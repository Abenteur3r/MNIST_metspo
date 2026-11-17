"""State-machine contracts on invented sequences, independent of match data."""

import unittest

import numpy as np

from possession.control import evidence
from possession.decode import decode
from possession.io_atd import Tracking


class DecoderTests(unittest.TestCase):
    def test_confirmed_turnover_is_backdated_to_control_only(self):
        control = np.array([-1] * 8 + [1] * 5 + [-1] * 2, dtype=np.int8)
        restart = np.array([0] * 3 + [-1] * 12, dtype=np.int8)
        labels = decode(control, restart, np.zeros(15, dtype=bool), [(0, 14)])
        np.testing.assert_array_equal(labels, list("DD" + "H" * 6 + "A" * 7))

    def test_restart_observations_expire_after_ten_frames(self):
        restart = np.full(20, -1, dtype=np.int8)
        restart[[0, 1, 12, 13, 14]] = 1
        labels = decode(np.full(20, -1), restart, np.zeros(20, dtype=bool), [(0, 19)])
        np.testing.assert_array_equal(labels[:14], np.full(14, "D"))
        np.testing.assert_array_equal(labels[14:], np.full(6, "A"))

    def test_tag_onset_resets_once_and_restart_can_precede_tag_end(self):
        restart = np.array([0] * 3 + [-1] * 3 + [1] * 3 + [-1] * 3)
        tagged = np.zeros(12, dtype=bool)
        tagged[6:] = True
        labels = decode(np.full(12, -1), restart, tagged, [(0, 11)])
        np.testing.assert_array_equal(labels, list("DDHHHHDDAAAA"))

    def test_half_boundary_prevents_relative_velocity_from_crossing(self):
        ball = np.column_stack((np.arange(30) * .12, np.zeros(30)))
        tracking = Tracking((ball + [1., 0.])[:, None, :], ball,
                            np.array([0]), np.array([False]), True)
        control, _ = evidence(tracking, [(0, 9), (20, 29)])
        np.testing.assert_array_equal(control[20:25], np.full(5, -1))
        np.testing.assert_array_equal(control[25:], np.zeros(5))

    def test_keeper_reliability_reverses_with_half(self):
        ball = np.tile([10., 30.], (40, 1))
        tracking = Tracking((ball + [.5, 0.])[:, None, :], ball,
                            np.array([0]), np.array([True]), True)
        control, _ = evidence(tracking, [(0, 19), (20, 39)])
        np.testing.assert_array_equal(control[5:20], np.zeros(15))
        np.testing.assert_array_equal(control[20:], np.full(20, -1))

    def test_empty_decode(self):
        self.assertEqual(len(decode(np.array([]), np.array([]),
                                    np.array([], dtype=bool), [])), 0)
