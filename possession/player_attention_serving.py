"""Conditional CPU serving for the original externally accepted player-attention fold bag.

Snapshot attention has no player-ID embedding or cross-frame identity state.
Token velocities still use local same-slot continuity over five frames; slot
order invariance after token construction does not imply invariance to arbitrary
frame-by-frame slot swaps in the raw tracking.
"""

from functools import lru_cache
import json
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np

from .io_atd import Tracking
from .player_attention import SCHEMA, probability
from .player_set import TOKEN_NAMES, extract_tokens
from .tag_features import NAMES as TAG_NAMES
from .w3_features import SCHEMA as FEATURE_SCHEMA, extract
from .b_decode import decode
from .m2_control import expand

CANDIDATE = "player_attention_tcn"
ENSEMBLE_SCHEMA = "player-attention-ensemble-v1"
MATCHES = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9", "J03WMX", "J03WN1")
SEEDS = (2026, 2027, 2028)


def _finite_array(value, shape, *, positive=False):
    array = np.asarray(value)
    return (array.shape == shape and np.issubdtype(array.dtype, np.number)
            and np.isrealobj(array) and np.isfinite(array).all()
            and (not positive or (array > 0).all()))


def _member_filename(value):
    return (isinstance(value, str) and value.endswith(".json") and bool(value)
            and Path(value).name == value and "/" not in value and "\\" not in value
            and value not in (".", ".."))


_CONTROL_REALS = ("keeper_distance", "control_distance", "relative_speed", "control_speed",
                  "team_margin", "restart_distance", "restart_speed", "confirmation",
                  "restart_window", "restart_votes")
_CONTROL_COUNTS = ("confirmation", "restart_window", "restart_votes")
_DECODER_REALS = ("confidence", "min_quantile", "switch_penalty", "duration_weight")


def _nonnegative_reals(control):
    return all(isinstance(control[key], Real) and np.isfinite(control[key]) and control[key] >= 0
               for key in _CONTROL_REALS)


def _positive_counts(control):
    return all(isinstance(control[key], Integral) and not isinstance(control[key], bool)
               and control[key] >= 1 for key in _CONTROL_COUNTS)


def _control_contract(control, decoder):
    return (control["stride"] == 5 and control["window"] == 5 and decoder["dead_mix"] == 0
            and _nonnegative_reals(control) and _positive_counts(control))


def _decoder_settings(decoder):
    return (all(isinstance(decoder[key], Real) and np.isfinite(decoder[key])
                for key in _DECODER_REALS)
            and 0 <= decoder["confidence"] <= 1 and 0 <= decoder["min_quantile"] <= 1
            and decoder["switch_penalty"] >= 0 and decoder["duration_weight"] >= 0)


def _progress_priors(priors):
    if (priors["training_matches"] != list(MATCHES)
            or not _finite_array(priors["transition"], (3, 3))
            or not _finite_array(priors["initial"], (3,))):
        return False
    transition, initial = np.asarray(priors["transition"]), np.asarray(priors["initial"])
    return not ((transition < 0).any() or (initial < 0).any()
                or not np.allclose(transition.sum(axis=1), 1., atol=1e-6, rtol=0.)
                or not np.isclose(initial.sum(), 1., atol=1e-6, rtol=0.))


def _duration_contract(durations):
    return (len(durations) == 3 and all(np.asarray(values).ndim == 1 and len(values)
            and _finite_array(values, (len(values),), positive=True) for values in durations))


def _decoder_contract(artifact):
    control, decoder, priors = (artifact[key] for key in ("control", "decoder", "priors"))
    return (_control_contract(control, decoder) and _decoder_settings(decoder)
            and _progress_priors(priors) and _duration_contract(priors["durations"]))


_HEX40 = re.compile(r"[0-9a-f]{40}")
_ATTENTION_SHAPES = {"qkv": ((96, 32), (96,)), "projection": ((32, 32), (32,)),
                     "norm1": ((32,), (32,)), "ff1": ((64, 32), (64,)),
                     "ff2": ((32, 64), (32,)), "norm2": ((32,), (32,))}


def _header_contract(artifact):
    return (artifact["schema"] == ENSEMBLE_SCHEMA and artifact["candidate"] == CANDIDATE
            and artifact["feature_schema"] == FEATURE_SCHEMA
            and artifact["training_matches"] == list(MATCHES) and artifact["seeds"] == list(SEEDS)
            and artifact["selection_protocol"] == "experiments/20261106-r7-player-attention")


def _flag_contract(artifact):
    return (artifact["target_data_used"] is False and artifact["public_gt_used"] is False
            and artifact["accepted_external_comparison"] is True
            and artifact["accepted_external_transfer"] is True)


def _freeze_contract(artifact):
    return all(isinstance(artifact[key], str) and _HEX40.fullmatch(artifact[key])
               for key in ("evaluation_freeze_commit", "transfer_freeze_commit"))


def _names_contract(names):
    return (len(names) == 119 and all(isinstance(name, str) and name for name in names)
            and len(set(names)) == 119 and list(names[-8:]) == list(TAG_NAMES))


def _manifest_contract(artifact, models):
    members = artifact["members"]
    expected = [(held, seed) for held in MATCHES for seed in SEEDS]
    return (len(members) == len(models) == 21
            and [(member["held_match"], member["seed"]) for member in members] == expected
            and len({member["file"] for member in members}) == 21
            and all(_member_filename(member["file"]) for member in members))


def _member_manifest(member, train):
    return member["training_matches"] == train and member["normalization_matches"] == train


def _model_identity(model, member, train):
    return (model["schema"] == SCHEMA and model["kind"] == "player_attention_tcn"
            and model["class_order"] == ["H", "A", "D"] and model["target_data_used"] is False
            and model["training_matches"] == train and model["seed"] == member["seed"])


def _optional_flags(model, train):
    return (("public_gt_used" not in model or model["public_gt_used"] is False)
            and ("normalization_matches" not in model or model["normalization_matches"] == train))


def _statistics(model):
    return all(_finite_array(model[prefix + "mean"], (size,))
               and _finite_array(model[prefix + "scale"], (size,), positive=True)
               for prefix, size in (("", 119), ("token_", 10)))


def _player_layers(model):
    return len(model["player_layers"]) == 2 and all(
        _finite_array(layer["weight"], (32, 20 if index == 0 else 32))
        and _finite_array(layer["bias"], (32,))
        for index, layer in enumerate(model["player_layers"]))


def _attention_contract(attention):
    return (set(attention) == set(_ATTENTION_SHAPES) | {"team_bias"}
            and all(_finite_array(attention[name]["weight"], weight)
                    and _finite_array(attention[name]["bias"], bias)
                    for name, (weight, bias) in _ATTENTION_SHAPES.items())
            and _finite_array(attention["team_bias"], (2,)))


def _layer_contract(index, layer):
    if index < 5:
        return (_finite_array(layer["weight"], (32, 366 if index == 0 else 32, 3))
                and _finite_array(layer["bias"], (32,))
                and layer["dilation"] == 2**index and layer["relu"] is True)
    return (_finite_array(layer["weight"], (3, 32)) and _finite_array(layer["bias"], (3,))
            and layer["dilation"] == 1 and layer["relu"] is False)


def _layers_contract(layers):
    return len(layers) == 6 and all(_layer_contract(index, layer) for index, layer in enumerate(layers))


def _member_contract(member, model):
    train = [match for match in MATCHES if match != member["held_match"]]
    return (_member_manifest(member, train) and _model_identity(model, member, train)
            and _optional_flags(model, train) and _statistics(model)
            and _player_layers(model) and _attention_contract(model["attention"])
            and _layers_contract(model["layers"]))


def _ensemble_contract(artifact, models):
    return (_header_contract(artifact) and _flag_contract(artifact) and _freeze_contract(artifact)
            and _names_contract(artifact["feature_names"])
            and artifact["token_names"] == list(TOKEN_NAMES)
            and _manifest_contract(artifact, models) and _decoder_contract(artifact))


def validate(artifact, models):
    """Reject incomplete bags, altered schemas and invalid external provenance."""
    try:
        valid = _ensemble_contract(artifact, models) and all(
            _member_contract(member, model) for member, model in zip(artifact["members"], models))
    except (KeyError, TypeError, ValueError, IndexError):
        valid = False
    if not valid:
        raise ValueError("Invalid accepted player-attention ensemble, parameters or external provenance")


@lru_cache(maxsize=1)
def load_model():
    folder = Path(__file__).parent / "models"
    try:
        artifact = json.loads((folder / f"{CANDIDATE}.json").read_text())
        if not all(_member_filename(member["file"]) for member in artifact["members"]):
            raise ValueError
        models = [json.loads((folder / member["file"]).read_text()) for member in artifact["members"]]
        validate(artifact, models)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid player-attention packaged manifest or member data") from None
    return artifact, models


def _shape_contract(ball, players, teams, keepers):
    return (ball.ndim == 2 and ball.shape[1] == 2 and len(ball) > 0
            and players.ndim == 3 and players.shape[0] == len(ball) and players.shape[2] == 2
            and players.shape[1] > 0 and teams.shape == keepers.shape == (players.shape[1],))


def _coordinate_contract(ball, players):
    return (np.issubdtype(ball.dtype, np.floating) and np.issubdtype(players.dtype, np.floating)
            and not np.isinf(ball).any() and not np.isinf(players).any())


def _team_contract(teams, keepers):
    return (np.isin(teams, [0, 1]).all() and all((teams == team).any() for team in (0, 1))
            and np.issubdtype(keepers.dtype, np.bool_))


def _tracking_contract(tracking):
    try:
        ball, players, teams, keepers = (np.asarray(getattr(tracking, key))
                                         for key in ("ball", "players", "teams", "keepers"))
        valid = (_shape_contract(ball, players, teams, keepers)
                 and _coordinate_contract(ball, players)
                 and _team_contract(teams, keepers)
                 and isinstance(tracking.home_defends_left, (bool, np.bool_))
                 and _finite_array(tracking.pitch, (2,), positive=True))
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("player-attention requires aligned XY tracking and two-team slot metadata")


def _feature_contract(names, artifact):
    return names == artifact["feature_names"]


def _probability_contract(learned, rows):
    return (learned.shape == (rows, 3) and np.isfinite(learned).all()
            and not (learned < 0).any() and not (learned > 1).any()
            and np.allclose(learned.sum(axis=1), 1., rtol=0., atol=1e-6))


def _label_contract(labels, frames):
    return labels.shape == (frames,) and np.isin(labels, list("HAD")).all()


def half_prediction(tracking, intervals, artifact, models, *, team_names=None):
    _tracking_contract(tracking)
    values, names, features = extract(tracking, intervals, artifact["control"]["window"], team_names=team_names)
    if not _feature_contract(names, artifact):
        raise ValueError("player-attention serving feature schema mismatch")
    tokens = extract_tokens(tracking)
    learned = probability(values, tokens, models)
    if not _probability_contract(learned, len(values)):
        raise ValueError("player-attention serving produced invalid H/A/D probabilities")
    path = decode(features, learned, learned[:, 2], artifact["control"], artifact["decoder"], artifact["priors"])
    labels = expand(path, features, 5)
    if not _label_contract(labels, len(tracking.ball)):
        raise ValueError("player-attention serving produced invalid half labels")
    return labels


def _integral_pair(row):
    first, last = row
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in (first, last)):
        raise ValueError
    return first, last


def _checked_blocks(blocks, frames):
    blocks = list(blocks)
    if not blocks:
        raise ValueError
    previous = -1
    for row in blocks:
        first, last = _integral_pair(row)
        if first < 0 or last < first or last >= frames or first <= previous:
            raise ValueError
        previous = last
    return blocks


def _checked_intervals(intervals):
    if not isinstance(intervals, (list, tuple)):
        raise ValueError
    for row in intervals:
        first, stop = _integral_pair((row["start_frame"], row["end_frame"]))
        if stop <= first:
            raise ValueError
    return intervals


def predict(tracking, intervals, blocks, *, team_names=None, label="player-attention", half=half_prediction, loader=load_model):
    _tracking_contract(tracking)
    try:
        blocks = _checked_blocks(blocks, len(tracking.ball))
        intervals = _checked_intervals(intervals)
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{label} requires valid nonoverlapping blocks and tag intervals") from None
    artifact, models = loader()
    labels = np.full(len(tracking.ball), "X", dtype="<U1")
    for k, (first, last) in enumerate(blocks):
        part = Tracking(tracking.players[first:last+1], tracking.ball[first:last+1], tracking.teams,
                        tracking.keepers, tracking.home_defends_left != bool(k % 2), tracking.pitch)
        tags = [dict(row, start_frame=row["start_frame"]-first, end_frame=row["end_frame"]-first)
                for row in intervals if row["end_frame"] > first and row["start_frame"] <= last]
        labels[first:last+1] = half(part, tags, artifact, models, team_names=team_names)
    return labels
