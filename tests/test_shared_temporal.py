"""Behavior of the shared readout artifact contract and its binary logits."""

import copy
import unittest

import numpy as np

from possession import shared_temporal
from possession.shared_temporal_serving import load_model


def model():
    """A packaged ensemble member, the same artifact the submission ships."""
    return copy.deepcopy(load_model.__wrapped__()[1][0])


class ParametersTests(unittest.TestCase):
    def setUp(self):
        self.model = model()

    def assert_rejected(self, changed):
        with self.assertRaisesRegex(ValueError, "^shared-temporal artifact violates its fixed temporal-task recipe$"):
            shared_temporal._parameters(changed)

    def test_accepts_a_packaged_member(self):
        shared_temporal._parameters(self.model)

    def test_rejects_a_different_identity(self):
        for key, value in (("schema", "other"), ("kind", "other"), ("arm", "isolated"),
                           ("class_order", ["A", "H", "D"]), ("task_isolation", {}),
                           ("target_data_used", True), ("public_gt_used", True)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.model)
                changed[key] = value
                self.assert_rejected(changed)

    def test_rejects_broken_statistics(self):
        for prefix, size in (("", 119), ("token_", 10)):
            with self.subTest(prefix=prefix, field="mean"):
                changed = copy.deepcopy(self.model)
                changed[prefix + "mean"] = np.zeros(size + 1, np.float32)
                self.assert_rejected(changed)
            with self.subTest(prefix=prefix, field="scale"):
                changed = copy.deepcopy(self.model)
                changed[prefix + "scale"][0] = 0.
                self.assert_rejected(changed)

    def test_rejects_broken_player_layers(self):
        with self.subTest(case="count"):
            changed = copy.deepcopy(self.model)
            changed["player_layers"] = changed["player_layers"][:1]
            self.assert_rejected(changed)
        with self.subTest(case="width"):
            changed = copy.deepcopy(self.model)
            changed["player_layers"][0]["weight"] = np.zeros((32, 19), np.float32)
            self.assert_rejected(changed)

    def test_rejects_broken_attention(self):
        with self.subTest(case="missing"):
            changed = copy.deepcopy(self.model)
            del changed["attention"]["qkv"]
            self.assert_rejected(changed)
        with self.subTest(case="extra"):
            changed = copy.deepcopy(self.model)
            changed["attention"]["extra"] = np.zeros(2, np.float32)
            self.assert_rejected(changed)
        with self.subTest(case="shape"):
            changed = copy.deepcopy(self.model)
            changed["attention"]["ff1"]["weight"] = np.zeros((32, 32), np.float32)
            self.assert_rejected(changed)
        with self.subTest(case="team_bias"):
            changed = copy.deepcopy(self.model)
            changed["attention"]["team_bias"] = np.zeros(3, np.float32)
            self.assert_rejected(changed)

    def test_rejects_broken_paths(self):
        with self.subTest(case="count"):
            changed = copy.deepcopy(self.model)
            changed["paths"] = changed["paths"][:1]
            self.assert_rejected(changed)
        with self.subTest(case="layers"):
            changed = copy.deepcopy(self.model)
            changed["paths"][0].pop()
            self.assert_rejected(changed)
        with self.subTest(case="width"):
            changed = copy.deepcopy(self.model)
            changed["paths"][0][0]["weight"] = np.zeros((17, 366, 3), np.float32)
            self.assert_rejected(changed)
        with self.subTest(case="dilation"):
            changed = copy.deepcopy(self.model)
            changed["paths"][0][1]["dilation"] = 3
            self.assert_rejected(changed)
        with self.subTest(case="relu"):
            changed = copy.deepcopy(self.model)
            changed["paths"][0][1]["relu"] = False
            self.assert_rejected(changed)

    def test_rejects_a_broken_readout(self):
        with self.subTest(case="weight"):
            changed = copy.deepcopy(self.model)
            changed["readout"]["weight"] = np.zeros((3, 32), np.float32)
            self.assert_rejected(changed)
        with self.subTest(case="bias"):
            changed = copy.deepcopy(self.model)
            changed["readout"]["bias"] = np.zeros(3, np.float32)
            self.assert_rejected(changed)


class LogitsTests(unittest.TestCase):
    def setUp(self):
        self.model = model()
        self.values = np.zeros((6, 119), np.float32)
        self.tokens = np.zeros((6, 2, 3, 10), np.float32)

    def test_returns_two_finite_columns_per_frame(self):
        found = shared_temporal.logits(self.values, self.tokens, self.model)
        self.assertEqual(found.shape, (6, 2))
        self.assertTrue(np.isfinite(found).all())

    def test_an_empty_window_returns_no_rows(self):
        found = shared_temporal.logits(self.values[:0], self.tokens[:0], self.model)
        self.assertEqual(found.shape, (0, 2))

    def test_rejects_a_mismatched_schema_or_clock(self):
        cases = {"values": np.zeros((6, 118), np.float32),
                 "tokens": np.zeros((6, 3, 3, 10), np.float32),
                 "slots": np.zeros((6, 2, 0, 10), np.float32),
                 "width": np.zeros((6, 2, 3, 9), np.float32),
                 "clock": np.zeros((5, 2, 3, 10), np.float32)}
        for name, replacement in cases.items():
            with self.subTest(case=name):
                values = replacement if name == "values" else self.values
                tokens = replacement if name != "values" else self.tokens
                with self.assertRaisesRegex(ValueError, "^shared-temporal feature schema or clock mismatch$"):
                    shared_temporal.logits(values, tokens, self.model)


class ProbabilityTests(unittest.TestCase):
    def test_reads_two_logits_as_three_class_probabilities(self):
        found = shared_temporal.binary_probability(np.zeros((1, 2)))
        np.testing.assert_allclose(found[0], [.25, .25, .5])

    def test_reads_a_positive_logit_as_the_dead_class(self):
        found = shared_temporal.binary_probability(np.array([[10., 0.]]))
        self.assertAlmostEqual(float(found[0, 2]), 1., places=4)
        self.assertLess(float(found[0, 0]), 1e-4)

    def test_rejects_malformed_binary_logits(self):
        for name, logits in (("one", np.zeros((2, 1))), ("three", np.zeros((2, 3))),
                             ("nan", np.array([[0., np.nan]]))):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "^shared-temporal requires finite two-column binary logits$"):
                    shared_temporal.binary_probability(logits)

    def test_averages_one_registered_arm(self):
        member = model()
        values = np.zeros((4, 119), np.float32)
        tokens = np.zeros((4, 2, 3, 10), np.float32)
        one = shared_temporal.probability(values, tokens, [member])
        two = shared_temporal.probability(values, tokens, [member, copy.deepcopy(member)])
        np.testing.assert_allclose(one, two)
        self.assertEqual(one.shape, (4, 3))

    def test_rejects_an_empty_bag_or_mixed_arms(self):
        values = np.zeros((4, 119), np.float32)
        tokens = np.zeros((4, 2, 3, 10), np.float32)
        other = copy.deepcopy(model())
        other["arm"] = "isolated"
        for name, bag in (("empty", []), ("mixed", [model(), other])):
            with self.subTest(case=name):
                with self.assertRaisesRegex(
                        ValueError, "^shared-temporal requires a nonempty bag from one registered arm$"):
                    shared_temporal.probability(values, tokens, bag)
