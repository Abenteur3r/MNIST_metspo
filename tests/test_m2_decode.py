import unittest
from unittest import mock

import numpy as np

from possession.m2_decode import decode, duration_decode

CONTROL = {"stride": 5, "window": 5, "keeper_half_exclusion": False,
           "keeper_distance": 3.0, "control_distance": 2.0, "relative_speed": 1.0,
           "control_speed": 5.0, "team_margin": 0.5, "restart_distance": 10.0,
           "restart_speed": 0.5, "confirmation": 3, "restart_window": 3, "restart_votes": 1}
DECODER = {"dead_weight": 1.0, "owner_weight": 1.0, "min_quantile": 0.0,
           "switch_penalty": 0.0, "duration_weight": 0.0}
PRIORS = {"minimum_frames": [1, 1, 1], "mean_frames": [1, 1, 1],
          "transition": [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]],
          "initial": [1 / 3, 1 / 3, 1 / 3]}


def features(n=10, distance=1.0, speed=2.0):
    return {"distance": np.full((n, 2), distance), "relative_speed": np.zeros((n, 2, 3)),
            "ball_speed": np.full((n, 3), speed), "half_ok": np.ones((n, 2), bool),
            "teams": np.array([0, 1]), "keepers": np.zeros(2, bool),
            "tagged": np.zeros(n, bool), "n_frames": np.array(n * 5)}


class DurationDecodeTests(unittest.TestCase):
    def test_empty_input_returns_an_empty_path(self):
        path = duration_decode(np.zeros((0, 3)), np.ones((0, 3, 3)), [1, 1, 1], [1, 1, 1],
                               np.eye(3), np.full(3, 1 / 3), 0., 0.)
        self.assertEqual(path.dtype, np.int8)
        self.assertEqual(path.shape, (0,))

    def test_dead_evidence_saturates_the_dead_state(self):
        emissions = np.tile([-10., -10., 0.], (4, 1))
        path = duration_decode(emissions, np.ones((4, 3, 3)), [1, 1, 1], [1, 1, 1],
                               np.eye(3), np.full(3, 1 / 3), 0., 0.)
        np.testing.assert_array_equal(path, np.full(4, 2))

    def test_minimum_duration_holds_a_state_across_frames(self):
        emissions = np.array([[0., -10., -10.], [-10., 0., -10.]])
        path = duration_decode(emissions, np.ones((2, 3, 3)), [2, 1, 1], [2, 1, 1],
                               np.eye(3), np.full(3, 1 / 3), 0., 0.)
        np.testing.assert_array_equal(path, [0, 0])

    def test_forbidden_transition_keeps_the_previous_state(self):
        emissions = np.array([[0., -10., -10.], [-10., 0., -10.]])
        allowed = np.ones((2, 3, 3), bool)
        allowed[1, 0, 1] = False
        path = duration_decode(emissions, allowed, [2, 1, 1], [2, 1, 1],
                               np.eye(3), np.full(3, 1 / 3), 0., 0.)
        self.assertEqual(path[1], 0)

    def test_infeasible_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "^No feasible duration path$"):
            duration_decode(np.full((1, 3), -np.inf), np.ones((1, 3, 3)), [1, 1, 1],
                            [1, 1, 1], np.eye(3), np.full(3, 1 / 3), 0., 0.)


class DecodeTests(unittest.TestCase):
    def test_dead_probability_selects_the_dead_state(self):
        path = decode(features(), np.ones(10), CONTROL, DECODER, PRIORS)
        np.testing.assert_array_equal(path, np.full(10, 2))

    def test_low_dead_probability_keeps_play_alive(self):
        config = dict(DECODER, owner_weight=0.0)
        path = decode(features(), np.zeros(10), CONTROL, config, PRIORS)
        self.assertTrue(np.isin(path, [0, 1]).all())

    def test_dead_probability_is_sampled_at_the_control_stride(self):
        dead = np.zeros(10)
        dead[::2] = 1.0
        path = decode(features(), dead, dict(CONTROL, stride=10), DECODER, PRIORS)
        np.testing.assert_array_equal(path, np.full(5, 2))

    def test_duration_priors_become_minimum_durations_and_means(self):
        priors = {"durations": [[2., 4.], [8., 12.], [20., 40.]],
                  "transition": [[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]],
                  "initial": [1 / 3, 1 / 3, 1 / 3]}
        config = dict(DECODER, min_quantile=0.0)
        with mock.patch("possession.m2_decode.duration_decode", return_value=np.zeros(10, np.int8)) as walk:
            decode(features(), np.zeros(10), CONTROL, config, priors)
        minimum, means = walk.call_args.args[2:4]
        self.assertEqual(list(minimum), [1, 2, 4])
        np.testing.assert_allclose(list(means), [0.6, 2.0, 6.0])

    def test_explicit_frame_priors_are_scaled_by_the_stride(self):
        expected = np.zeros(10, np.int8)
        with mock.patch("possession.m2_decode.duration_decode", return_value=expected) as walk:
            result = decode(features(), np.zeros(10), CONTROL, DECODER, PRIORS)
        minimum, means = walk.call_args.args[2:4]
        self.assertEqual(list(minimum), [1, 1, 1])
        np.testing.assert_allclose(list(means), [0.2, 0.2, 0.2])
        np.testing.assert_array_equal(result, expected)

    def test_length_weight_delegates_to_the_length_aware_walk(self):
        config = dict(DECODER, length_weight=0.5, length_max=12)
        expected = np.zeros(10, np.int8)
        with mock.patch("possession.m2_decode_length.duration_decode_length", return_value=expected) as walk:
            result = decode(features(), np.zeros(10), CONTROL, config, PRIORS)
        self.assertIs(result, expected)
        self.assertEqual(walk.call_args.args[8:10], (0.5, 12))
        self.assertEqual(walk.call_args.args[6:8], (0.0, 0.0))
