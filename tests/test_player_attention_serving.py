import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from possession import player_attention, player_attention_serving as serving
from possession.io_atd import Tracking
from possession.player_set import TOKEN_NAMES
from possession.tag_features import NAMES as TAG_NAMES
from possession.w3_features import SCHEMA as FEATURE_SCHEMA, extract

HELD = list(serving.MATCHES)
SEEDS = list(serving.SEEDS)
FREEZE = "a" * 40
NAMES = [f"f{i}" for i in range(111)] + list(TAG_NAMES)


def _zeros(*shape):
    return np.zeros(shape)


def _attention():
    return {"qkv": {"weight": _zeros(96, 32), "bias": _zeros(96)},
            "projection": {"weight": _zeros(32, 32), "bias": _zeros(32)},
            "norm1": {"weight": _zeros(32), "bias": _zeros(32)},
            "ff1": {"weight": _zeros(64, 32), "bias": _zeros(64)},
            "ff2": {"weight": _zeros(32, 64), "bias": _zeros(32)},
            "norm2": {"weight": _zeros(32), "bias": _zeros(32)},
            "team_bias": _zeros(2)}


def _model(held, seed):
    train = [match for match in HELD if match != held]
    layers = [{"weight": _zeros(32, 366 if index == 0 else 32, 3),
               "bias": _zeros(32), "dilation": 2 ** index, "relu": True}
              for index in range(5)]
    layers.append({"weight": _zeros(3, 32), "bias": _zeros(3), "dilation": 1, "relu": False})
    return {"schema": player_attention.SCHEMA, "kind": serving.CANDIDATE,
            "class_order": ["H", "A", "D"], "target_data_used": False,
            "training_matches": train, "seed": seed,
            "mean": _zeros(119), "scale": np.ones(119),
            "token_mean": _zeros(10), "token_scale": np.ones(10),
            "player_layers": [{"weight": _zeros(32, 20), "bias": _zeros(32)},
                              {"weight": _zeros(32, 32), "bias": _zeros(32)}],
            "attention": _attention(), "layers": layers}


def ensemble():
    """A minimal ensemble that satisfies every published serving contract."""
    artifact = {"schema": serving.ENSEMBLE_SCHEMA, "candidate": serving.CANDIDATE,
                "feature_schema": FEATURE_SCHEMA,
                "target_data_used": False, "public_gt_used": False,
                "accepted_external_comparison": True, "accepted_external_transfer": True,
                "training_matches": list(HELD), "seeds": list(SEEDS),
                "selection_protocol": "experiments/20261106-r7-player-attention",
                "evaluation_freeze_commit": FREEZE, "transfer_freeze_commit": FREEZE,
                "feature_names": list(NAMES), "token_names": list(TOKEN_NAMES),
                "members": [{"held_match": held, "seed": seed,
                             "file": f"member_{held}_{seed}.json",
                             "training_matches": [m for m in HELD if m != held],
                             "normalization_matches": [m for m in HELD if m != held]}
                            for held in HELD for seed in SEEDS],
                "control": {"stride": 5, "window": 5, "keeper_distance": 3.0,
                            "control_distance": 2.0, "relative_speed": 1.0,
                            "control_speed": 5.0, "team_margin": 0.5,
                            "restart_distance": 10.0, "restart_speed": 0.5,
                            "confirmation": 3, "restart_window": 3, "restart_votes": 1},
                "decoder": {"dead_mix": 0.0, "confidence": 0.55, "min_quantile": 0.0,
                            "switch_penalty": 0.0, "duration_weight": 0.1},
                "priors": {"training_matches": list(HELD),
                           "transition": np.full((3, 3), 1 / 3),
                           "initial": np.full(3, 1 / 3),
                           "durations": [np.array([10.0, 20.0]), np.array([10.0]),
                                         np.array([5.0, 15.0])]}}
    models = [_model(member["held_match"], member["seed"]) for member in artifact["members"]]
    return artifact, models


class ServingValidateTests(unittest.TestCase):
    def setUp(self):
        self.artifact, self.models = ensemble()

    def assertRejected(self, artifact=None, models=None):
        with self.assertRaisesRegex(ValueError, "^Invalid accepted player-attention ensemble, parameters or external provenance$"):
            serving.validate(self.artifact if artifact is None else artifact,
                            self.models if models is None else models)

    def test_published_ensemble_is_accepted(self):
        self.assertIsNone(serving.validate(self.artifact, self.models))

    def test_artifact_contract_rejections(self):
        cases = {"schema": ("schema", "other-ensemble"),
                 "candidate": ("candidate", "other"),
                 "feature_schema": ("feature_schema", "other"),
                 "target_data_used": ("target_data_used", True),
                 "public_gt_used": ("public_gt_used", True),
                 "external comparison": ("accepted_external_comparison", False),
                 "external transfer": ("accepted_external_transfer", False),
                 "training matches": ("training_matches", list(HELD[:-1])),
                 "seeds": ("seeds", [2026]),
                 "selection protocol": ("selection_protocol", "experiments/other"),
                 "evaluation freeze": ("evaluation_freeze_commit", "abc"),
                 "transfer freeze": ("transfer_freeze_commit", "A" * 40),
                 "feature names": ("feature_names", list(NAMES[:-1])),
                 "duplicate feature names": ("feature_names", [NAMES[0]] * 119),
                 "tag suffix": ("feature_names", list(NAMES[:-8]) + ["tag"] * 8),
                 "token names": ("token_names", list(TOKEN_NAMES[:-1]))}
        for label, (key, value) in cases.items():
            with self.subTest(label):
                artifact = copy.deepcopy(self.artifact)
                artifact[key] = value
                self.assertRejected(artifact=artifact)
        with self.subTest("missing key"):
            del self.artifact["seeds"]
            self.assertRejected()

    def test_member_manifest_rejections(self):
        cases = []
        artifact = copy.deepcopy(self.artifact)
        artifact["members"].pop()
        cases.append(("twenty members", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][1]["file"] = artifact["members"][0]["file"]
        cases.append(("duplicate member file", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["file"] = "nested/member.json"
        cases.append(("nested member file", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["seed"] = 1
        cases.append(("held and seed order", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["training_matches"] = list(HELD[:-1])[:6]
        cases.append(("member training matches", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["normalization_matches"] = list(HELD[:-1])[:6]
        cases.append(("member normalization matches", artifact))
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["file"] = "member.json.exe"
        cases.append(("member file suffix", artifact))
        for label, broken in cases:
            with self.subTest(label):
                self.assertRejected(artifact=broken)
        with self.subTest("model count"):
            self.assertRejected(models=self.models[:-1])

    def test_decoder_contract_rejections(self):
        cases = [("stride", ("control", "stride", 1)),
                 ("window", ("control", "window", 15)),
                 ("negative control distance", ("control", "control_distance", -1.0)),
                 ("non integral confirmation", ("control", "confirmation", 3.0)),
                 ("zero restart window", ("control", "restart_window", 0)),
                 ("dead mix", ("decoder", "dead_mix", 0.5)),
                 ("confidence range", ("decoder", "confidence", 1.5)),
                 ("quantile range", ("decoder", "min_quantile", -0.1)),
                 ("negative penalty", ("decoder", "switch_penalty", -1.0)),
                 ("negative duration weight", ("decoder", "duration_weight", -0.1))]
        for label, (section, key, value) in cases:
            with self.subTest(label):
                artifact = copy.deepcopy(self.artifact)
                artifact[section][key] = value
                self.assertRejected(artifact=artifact)
        priors = [("training matches", "training_matches", list(HELD[:-1])),
                  ("transition shape", "transition", np.ones((2, 2))),
                  ("initial shape", "initial", np.ones(2)),
                  ("transition sum", "transition", np.full((3, 3), 0.5)),
                  ("initial sum", "initial", np.full(3, 0.33)),
                  ("negative prior", "initial", np.array([-1.0, 1.0, 1.0])),
                  ("duration count", "durations", [np.array([1.0]), np.array([1.0])]),
                  ("empty duration", "durations", [np.array([1.0]), np.array([]), np.array([1.0])]),
                  ("infinite duration", "durations", [np.array([1.0]), np.array([np.inf]), np.array([1.0])]),
                  ("zero duration", "durations", [np.array([1.0]), np.array([0.0]), np.array([1.0])])]
        for label, key, value in priors:
            with self.subTest("priors " + label):
                artifact = copy.deepcopy(self.artifact)
                artifact["priors"][key] = value
                self.assertRejected(artifact=artifact)

    def test_member_model_rejections(self):
        cases = [("schema", ("schema", "other")),
                 ("kind", ("kind", "other")),
                 ("class order", ("class_order", ["A", "H", "D"])),
                 ("target data", ("target_data_used", True)),
                 ("public ground truth", ("public_gt_used", True)),
                 ("training matches", ("training_matches", list(HELD[:-1])[:6])),
                 ("seed", ("seed", 1)),
                 ("mean shape", ("mean", _zeros(118))),
                 ("scale positivity", ("scale", np.zeros(119))),
                 ("token scale positivity", ("token_scale", np.zeros(10))),
                 ("player layers", ("player_layers", [{"weight": _zeros(32, 20), "bias": _zeros(32)}])),
                 ("layers", ("layers", [{"weight": _zeros(32, 32, 3), "bias": _zeros(32),
                                         "dilation": 1, "relu": True}]))]
        for label, (key, value) in cases:
            with self.subTest(label):
                models = copy.deepcopy(self.models)
                models[0][key] = value
                self.assertRejected(models=models)
        with self.subTest("normalization matches"):
            models = copy.deepcopy(self.models)
            models[0]["normalization_matches"] = list(HELD[:-1])[:6]
            self.assertRejected(models=models)

    def test_attention_rejections(self):
        cases = [("missing key", lambda a: a.pop("team_bias")),
                 ("qkv shape", lambda a: a["qkv"].__setitem__("weight", _zeros(32, 96))),
                 ("ff1 shape", lambda a: a["ff1"].__setitem__("weight", _zeros(32, 64))),
                 ("norm shape", lambda a: a["norm1"].__setitem__("bias", _zeros(16)))]
        for label, mutate in cases:
            with self.subTest(label):
                models = copy.deepcopy(self.models)
                mutate(models[0]["attention"])
                self.assertRejected(models=models)
        with self.subTest("extra key"):
            models = copy.deepcopy(self.models)
            models[0]["attention"]["extra"] = _zeros(1)
            self.assertRejected(models=models)

    def test_layer_rejections(self):
        cases = [("conv shape", 0, "weight", _zeros(32, 32, 3)),
                 ("conv dilation", 1, "dilation", 3),
                 ("conv relu", 2, "relu", False),
                 ("dense shape", 5, "weight", _zeros(32, 3)),
                 ("dense relu", 5, "relu", True),
                 ("player layer shape", None, "weight", _zeros(32, 16))]
        for label, index, key, value in cases:
            with self.subTest(label):
                models = copy.deepcopy(self.models)
                if index is None:
                    models[0]["player_layers"][0][key] = value
                else:
                    models[0]["layers"][index][key] = value
                self.assertRejected(models=models)


class TrackingContractTests(unittest.TestCase):
    def _tracking(self, **overrides):
        n, slots = 10, 8
        values = dict(
            players=np.zeros((n, slots, 2)),
            ball=np.zeros((n, 2)),
            teams=np.array([0] * 4 + [1] * 4),
            keepers=np.zeros(slots, bool),
            home_defends_left=True,
        )
        values.update(overrides)
        return Tracking(**values)

    def _accepts(self, tracking):
        half = lambda part, tags, artifact, models, team_names=None: np.full(len(part.ball), "H")
        return serving.predict(tracking, [], [(0, 9)], half=half, loader=lambda: (object(), []))

    def test_aligned_two_team_tracking_is_accepted(self):
        self.assertEqual(self._accepts(self._tracking()).tolist(), ["H"] * 10)

    def test_malformed_tracking_is_rejected(self):
        n, slots = 10, 8
        ball = np.zeros((n, 2))
        players = np.zeros((n, slots, 2))
        teams = np.array([0] * 4 + [1] * 4)
        keepers = np.zeros(slots, bool)
        cases = {
            "ball is not xy": dict(ball=np.zeros((n, 3))),
            "empty ball": dict(ball=np.zeros((0, 2)), players=np.zeros((0, slots, 2))),
            "players are not xy": dict(players=np.zeros((n, slots, 3))),
            "players and ball disagree": dict(players=np.zeros((n + 1, slots, 2))),
            "no player slots": dict(players=np.zeros((n, 0, 2)), teams=np.array([], int), keepers=np.array([], bool)),
            "teams do not match slots": dict(teams=np.array([0] * 4 + [1] * 3)),
            "ball is not float": dict(ball=np.zeros((n, 2), int)),
            "players are not float": dict(players=np.zeros((n, slots, 2), int)),
            "infinite ball": dict(ball=np.full((n, 2), np.inf)),
            "infinite players": dict(players=np.full((n, slots, 2), -np.inf)),
            "unknown team label": dict(teams=np.array([0] * 4 + [2] * 4)),
            "one team only": dict(teams=np.array([0] * slots)),
            "keepers are not flags": dict(keepers=np.zeros(slots)),
            "home side is not a flag": dict(home_defends_left=1),
            "pitch is not positive": dict(pitch=np.array([0.0, 68.0])),
            "pitch is not two lengths": dict(pitch=np.array([105.0, 68.0, 0.0])),
        }
        for label, overrides in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, "^player-attention requires aligned XY tracking and two-team slot metadata$"):
                    self._accepts(self._tracking(**overrides))
        self.assertEqual(ball.shape, (n, 2))
        self.assertEqual(players.shape, (n, slots, 2))
        self.assertEqual(len(teams), slots)
        self.assertFalse(keepers.any())


class HalfPredictionTests(unittest.TestCase):
    def _tracking(self):
        n, slots = 10, 8
        return Tracking(np.zeros((n, slots, 2)), np.zeros((n, 2)),
                        np.array([0] * 4 + [1] * 4), np.zeros(slots, bool), True)

    def _artifact(self):
        artifact, models = ensemble()
        tracking = self._tracking()
        artifact["feature_names"] = extract(tracking, [])[1]
        return artifact, models, tracking

    def test_predicts_a_label_for_every_frame(self):
        artifact, models, tracking = self._artifact()
        labels = serving.half_prediction(tracking, [], artifact, models)
        self.assertEqual(labels.shape, (10,))
        self.assertTrue(np.isin(labels, list("HAD")).all())

    def test_rejects_a_feature_schema_mismatch(self):
        artifact, models, tracking = self._artifact()
        artifact["feature_names"] = list(reversed(artifact["feature_names"]))
        with self.assertRaisesRegex(ValueError, "^player-attention serving feature schema mismatch$"):
            serving.half_prediction(tracking, [], artifact, models)

    def test_rejects_probabilities_outside_a_simplex(self):
        artifact, models, tracking = self._artifact()
        with mock.patch.object(serving, "probability", return_value=np.full((2, 3), 0.5)):
            with self.assertRaisesRegex(ValueError, "^player-attention serving produced invalid H/A/D probabilities$"):
                serving.half_prediction(tracking, [], artifact, models)

    def test_rejects_half_labels_of_the_wrong_length(self):
        artifact, models, tracking = self._artifact()
        with mock.patch.object(serving, "expand", return_value=np.array(["H"])):
            with self.assertRaisesRegex(ValueError, "^player-attention serving produced invalid half labels$"):
                serving.half_prediction(tracking, [], artifact, models)


class LoadModelTests(unittest.TestCase):
    def _install(self, folder, artifact, models):
        (folder / "player_attention_tcn.json").write_text(
            json.dumps(artifact, default=lambda value: np.asarray(value).tolist()))
        for member, model in zip(artifact["members"], models):
            (folder / member["file"]).write_text(json.dumps(model, default=lambda value: np.asarray(value).tolist()))

    def test_reads_the_packaged_ensemble(self):
        artifact, models = ensemble()
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            folder = base / "models"
            folder.mkdir()
            self._install(folder, artifact, models)
            with mock.patch.object(serving, "__file__", str(base / "serving.py")):
                found, loaded = serving.load_model.__wrapped__()
        self.assertEqual(found["schema"], serving.ENSEMBLE_SCHEMA)
        self.assertEqual([member["file"] for member in found["members"]],
                         [member["file"] for member in artifact["members"]])
        self.assertEqual(len(loaded), 21)

    def test_wraps_a_bad_member_filename(self):
        artifact, models = ensemble()
        artifact["members"][0]["file"] = "../escape.json"
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            folder = base / "models"
            folder.mkdir()
            self._install(folder, artifact, models)
            with mock.patch.object(serving, "__file__", str(base / "serving.py")):
                with self.assertRaisesRegex(ValueError, "^Invalid player-attention packaged manifest or member data$"):
                    serving.load_model.__wrapped__()

    def test_missing_manifest_surfaces_as_a_file_error(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base / "models").mkdir()
            with mock.patch.object(serving, "__file__", str(base / "serving.py")):
                with self.assertRaises(FileNotFoundError):
                    serving.load_model.__wrapped__()


if __name__ == "__main__":
    unittest.main()
