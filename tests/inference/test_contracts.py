import copy
import unittest
from unittest import mock
import numpy as np
from possession.data.tracking import Tracking
from possession.features.combined import extract
from possession.inference import pipeline as serving
from possession.inference.blocks import predict_blocks


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
        return predict_blocks(tracking, [], [(0, 9)], half=half, loader=lambda: (object(), []))

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
                with self.assertRaisesRegex(ValueError, "^MNIST requires aligned XY tracking and two-team slot metadata$"):
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
        artifact, models = copy.deepcopy(serving.load_model())
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
        with self.assertRaisesRegex(ValueError, "^MNIST serving feature schema mismatch$"):
            serving.half_prediction(tracking, [], artifact, models)

    def test_rejects_probabilities_outside_a_simplex(self):
        artifact, models, tracking = self._artifact()
        with mock.patch.object(serving, "probability", return_value=np.full((2, 3), 0.5)):
            with self.assertRaisesRegex(ValueError, "^MNIST serving produced invalid H/A/D probabilities$"):
                serving.half_prediction(tracking, [], artifact, models)

    def test_rejects_half_labels_of_the_wrong_length(self):
        artifact, models, tracking = self._artifact()
        with mock.patch.object(serving, "expand", return_value=np.array(["H"])):
            with self.assertRaisesRegex(ValueError, "^MNIST serving produced invalid half labels$"):
                serving.half_prediction(tracking, [], artifact, models)
