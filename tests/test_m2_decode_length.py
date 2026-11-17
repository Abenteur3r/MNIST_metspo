"""Unit tests for the opt-in SALD-L decoder.

The contract that matters for the experiment is the zero-length-weight
equivalence: with the baseline duration prior retained, SALD-L must return the
same path as ``duration_decode``.
"""

import unittest

import numpy as np

from possession.m2_decode import duration_decode
from possession.m2_decode_length import duration_decode_length


class TestLengthDecode(unittest.TestCase):
    def setUp(self):
        self.n = 40
        self.minimum = np.array([2, 2, 2])
        self.means = np.array([8.0, 8.0, 8.0])
        self.transition = np.array(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        )
        self.initial = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        self.allowed = np.ones((self.n, 3, 3), dtype=bool)
        self.penalty = 0.0
        self.prior_weight = 0.1

    def _baseline(self, emissions):
        return duration_decode(
            emissions,
            self.allowed,
            self.minimum,
            self.means,
            self.transition,
            self.initial,
            self.penalty,
            self.prior_weight,
        )

    def _sald(self, emissions, length_weight):
        return duration_decode_length(
            emissions,
            self.allowed,
            self.minimum,
            self.means,
            self.transition,
            self.initial,
            self.penalty,
            self.prior_weight,
            length_weight,
            max_length=self.n,
        )

    def test_zero_length_weight_matches_baseline(self):
        for seed in range(5):
            emissions = np.random.default_rng(seed).normal(size=(self.n, 3))
            np.testing.assert_array_equal(
                self._baseline(emissions),
                self._sald(emissions, 0.0),
                err_msg=f"seed {seed}",
            )

    def test_nonzero_length_weight_returns_valid_path(self):
        emissions = np.random.default_rng(11).normal(size=(self.n, 3))
        for weight in (-0.5, 0.5):
            path = self._sald(emissions, weight)
            self.assertEqual(path.shape, (self.n,))
            self.assertTrue(np.isin(path, (0, 1, 2)).all())

    def test_rejects_bad_inputs(self):
        emissions = np.zeros((self.n, 3))
        with self.assertRaises(ValueError):
            duration_decode_length(
                emissions,
                self.allowed,
                self.minimum,
                self.means,
                self.transition,
                self.initial,
                self.penalty,
                self.prior_weight,
                0.0,
                max_length=1,
            )
        with self.assertRaises(ValueError):
            duration_decode_length(
                emissions,
                self.allowed,
                self.minimum,
                self.means,
                self.transition,
                self.initial,
                self.penalty,
                self.prior_weight,
                float("nan"),
                max_length=self.n,
            )
        with self.assertRaises(ValueError):
            duration_decode_length(
                np.zeros((0, 3)),
                np.zeros((0, 3, 3), dtype=bool),
                self.minimum,
                self.means,
                self.transition,
                self.initial,
                self.penalty,
                self.prior_weight,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
