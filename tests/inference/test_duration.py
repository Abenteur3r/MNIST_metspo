import unittest
import numpy as np
from possession.inference.duration import duration_decode


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
