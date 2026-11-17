"""Synthetic velocity cases, including missing interior observations."""

import unittest

import numpy as np

from possession.features import ball_velocity


class VelocityTests(unittest.TestCase):
    def test_requires_both_endpoints_and_every_intermediate_observation(self):
        for missing in range(6):
            with self.subTest(missing=missing):
                x = np.arange(7, dtype=float)
                seen = np.ones(7, dtype=bool)
                seen[missing] = False
                speed, valid = ball_velocity(x, np.zeros(7), seen)
                self.assertFalse(valid[5])
                self.assertTrue(np.isnan(speed[5]))
                # A finite visibility flag cannot make NaN coordinates usable.
                x[missing] = np.nan
                speed, valid = ball_velocity(x, np.zeros(7), np.ones(7, dtype=bool))
                self.assertFalse(valid[5])
        speed, valid = ball_velocity(np.arange(7.), np.zeros(7), np.ones(7, dtype=bool))
        np.testing.assert_array_equal(valid, [False] * 5 + [True, True])
        np.testing.assert_allclose(speed[valid], [25., 25.])

    def test_stationary_and_unobserved_velocity_are_distinct(self):
        speed, valid = ball_velocity(np.zeros(12), np.zeros(12), np.ones(12, dtype=bool))
        self.assertTrue(np.isnan(speed[:5]).all())
        self.assertFalse(valid[:5].any())
        np.testing.assert_array_equal(speed[5:], np.zeros(7))
        self.assertTrue(valid[5:].all())

    def test_short_series_and_invalid_window(self):
        for n in (0, 3, 5):
            speed, valid = ball_velocity(np.zeros(n), np.zeros(n), np.ones(n, dtype=bool))
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(speed).all())
        with self.assertRaises(ValueError):
            ball_velocity(np.zeros(2), np.zeros(2), np.ones(2, dtype=bool), window=0)
