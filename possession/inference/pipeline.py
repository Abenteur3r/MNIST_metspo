"""Load the frozen ensemble and infer H/A/D labels for each graded block."""

from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np

from possession.inference.contracts import (
    _member_filename, _decoder_contract, _tracking_contract, MATCHES, SEEDS,
)
from possession.inference.blocks import predict_blocks as _predict
from possession.inference.temporal import SCHEMA, KIND, TASK_ISOLATION, _parameters, probability
from possession.features.players import TOKEN_NAMES, extract_tokens
from possession.features.set_pieces import NAMES as TAG_NAMES
from possession.features.combined import SCHEMA as FEATURE_SCHEMA, extract
from possession.inference.decode import decode
from possession.inference.control import expand

CANDIDATE = "solution"
ENSEMBLE_SCHEMA = "mnist-shared-temporal-ensemble-v1"
PILOTS = ("J03WOH", "J03WQQ")
TRAINING_FREEZES = {"discovery":"2669cb4a902a03f4353ed7bcc905f0e92505174c",
                   "remaining":"d9c88ae261b962d1ffbfcb88a64f6755147d13bc"}
DECODER = {"dead_mix":0.,"confidence":.55,"switch_penalty":0.,"duration_weight":.1,"min_quantile":0.}


_ACCEPTANCE_FLAGS = ("accepted_external_comparison", "accepted_external_transfer",
                     "accepted_remaining_five", "accepted_transfer_r7",
                     "accepted_transfer_w3", "selected_by_campaign")
_FREEZE_FIELDS = ("evaluation_freeze_commit", "transfer_freeze_commit",
                  "campaign_selection_freeze_commit")


def _identity_contract(artifact):
    return (artifact["schema"] == ENSEMBLE_SCHEMA and artifact["candidate"] == CANDIDATE
            and artifact["feature_schema"] == FEATURE_SCHEMA
            and artifact["token_names"] == list(TOKEN_NAMES))


def _acceptance_contract(artifact):
    return (artifact["target_data_used"] is False and artifact["public_gt_used"] is False
            and all(artifact[key] is True for key in _ACCEPTANCE_FLAGS)
            and artifact["selection_protocol"] == "experiments/20261109-shared-temporal"
            and artifact["campaign_selection_protocol"] == "experiments/20261109-campaign-selection"
            and artifact["task_isolation"] == TASK_ISOLATION and artifact["arm"] == "shared_readout")


def _freeze_contract(artifact):
    return (artifact["training_freezes"] == TRAINING_FREEZES
            and all(isinstance(artifact[key], str) and re.fullmatch(r"[0-9a-f]{40}", artifact[key])
                    for key in _FREEZE_FIELDS))


def _names_contract(artifact):
    names = artifact["feature_names"]
    return (len(names) == 119 and len(set(names)) == 119
            and all(isinstance(name, str) and name for name in names)
            and names[-8:] == list(TAG_NAMES))


def _manifest_contract(artifact, models):
    members = artifact["members"]
    expected = [(held, seed) for held in MATCHES for seed in SEEDS]
    return (artifact["training_matches"] == list(MATCHES) and artifact["seeds"] == list(SEEDS)
            and len(members) == len(models) == 21
            and [(member["held_match"], member["seed"]) for member in members] == expected)


def _member_experiment(held):
    if held in PILOTS:
        return "20261109-r37-temporal-isolation", TRAINING_FREEZES["discovery"]
    return "20261109-shared-temporal", TRAINING_FREEZES["remaining"]


def _member_provenance(member, held, seed, train, source, freeze):
    return (member["file"] == f"mnist_{held}_{seed}.json"
            and _member_filename(member["file"])
            and member["training_matches"] == train
            and member["normalization_matches"] == train
            and member["training_freeze_commit"] == freeze
            and member["source_experiment"] == "experiments/" + source)


def _model_provenance(model, held, seed, train, source, freeze):
    return (model["source_experiment"] == "experiments/" + source
            and model["training_matches"] == train and model["seed"] == seed
            and model["schema"] == SCHEMA and model["kind"] == KIND
            and model["arm"] == "shared_readout"
            and model.get("normalization_matches", train) == train
            and (held in PILOTS or model.get("training_freeze_commit") == freeze))


def _member_contract(member, model):
    held, seed = member["held_match"], member["seed"]
    train = [match for match in MATCHES if match != held]
    source, freeze = _member_experiment(held)
    valid = (_member_provenance(member, held, seed, train, source, freeze)
             and _model_provenance(model, held, seed, train, source, freeze))
    _parameters(model)
    return valid


def _artifact_contract(artifact, models):
    return (_identity_contract(artifact) and _acceptance_contract(artifact)
            and _freeze_contract(artifact) and _names_contract(artifact)
            and _manifest_contract(artifact, models)
            and artifact["decoder"] == DECODER and _decoder_contract(artifact))


def validate(artifact, models):
    try:
        valid = _artifact_contract(artifact, models)
        for member, model in zip(artifact["members"], models):
            valid = _member_contract(member, model) and valid
    except (KeyError, TypeError, ValueError, IndexError):
        valid = False
    if not valid:
        raise ValueError("Invalid selected MNIST ensemble, original members or external provenance")


@lru_cache(maxsize=1)
def load_model():
    folder = Path(__file__).parents[1] / "models"
    try:
        artifact = json.loads((folder/f"{CANDIDATE}.json").read_text())
        if not all(_member_filename(m["file"]) for m in artifact["members"]):raise ValueError
        models = [json.loads((folder/m["file"]).read_text()) for m in artifact["members"]]
        validate(artifact,models)
    except (KeyError,TypeError,ValueError):
        raise ValueError("Invalid MNIST packaged manifest or members") from None
    return artifact,models


def half_prediction(tracking, intervals, artifact, models, *, team_names=None):
    _tracking_contract(tracking)
    values, names, features = extract(tracking, intervals, artifact["control"]["window"], team_names=team_names)
    if names != artifact["feature_names"]:
        raise ValueError("MNIST serving feature schema mismatch")
    tokens = extract_tokens(tracking)
    learned = probability(values, tokens, models)
    if (learned.shape != (len(values), 3) or not np.isfinite(learned).all()
            or (learned < 0).any() or (learned > 1).any()
            or not np.allclose(learned.sum(axis=1), 1., rtol=0., atol=1e-6)):
        raise ValueError("MNIST serving produced invalid H/A/D probabilities")
    path = decode(features, learned, learned[:, 2], artifact["control"], artifact["decoder"], artifact["priors"])
    labels = expand(path, features, 5)
    if labels.shape != (len(tracking.ball),) or not np.isin(labels, list("HAD")).all():
        raise ValueError("MNIST serving produced invalid half labels")
    return labels


def predict(tracking, intervals, blocks, *, team_names=None):
    return _predict(tracking, intervals, blocks, team_names=team_names,
                    half=half_prediction, loader=load_model)
