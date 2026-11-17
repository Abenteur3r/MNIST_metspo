"""Control, not proximity or flight, must establish an opponent possession."""
import unittest

import numpy as np

from possession.io_atd import Tracking
from possession.decode import decode
from possession.control import evidence


class ControlPossessionTests(unittest.TestCase):
    def track(self, ball, players, teams, keepers=None):
        return Tracking(np.asarray(players, dtype=float), np.asarray(ball, dtype=float),
                        np.asarray(teams, dtype=np.int8),
                        np.zeros(len(teams), dtype=bool) if keepers is None else keepers,
                        True)

    def test_flight_over_opponent_does_not_transfer_until_controlled_reception(self):
        n = 100
        ball = np.zeros((n, 2))
        ball[:, 0] = np.r_[np.arange(20) * .12,
                           2.28 + np.arange(1, 61) * .8,
                           50.28 + np.arange(1, 21) * .12]
        players = np.zeros((n, 2, 2))
        players[:, 0] = ball + [1., 0.]
        players[20:, 0] += [10., 0.]
        players[:, 1, 0] = 25.
        players[80:, 1] = ball[80:] + [1., 0.]
        track = self.track(ball, players, [0, 1])
        control, restart = evidence(track, [(0, n - 1)])
        labels = decode(control, restart, np.zeros(n, dtype=bool), [(0, n - 1)])
        np.testing.assert_array_equal(labels[10:80], np.full(70, "H"))
        np.testing.assert_array_equal(labels[85:], np.full(15, "A"))

    def test_stationary_keeper_hold_changes_team_without_creating_dead_ball(self):
        n = 80
        ball = np.zeros((n, 2))
        ball[:, 0] = np.r_[80 + np.arange(20) * .12, np.full(60, 90.)]
        players = np.full((n, 2, 2), 50.)
        players[:20, 0] = ball[:20] + [1., 0.]
        players[20:, 1] = ball[20:] + [1., 0.]
        track = self.track(ball, players, [0, 1], np.array([False, True]))
        control, restart = evidence(track, [(0, n - 1)])
        labels = decode(control, restart, np.zeros(n, dtype=bool), [(0, n - 1)])
        np.testing.assert_array_equal(labels[30:], np.full(50, "A"))

    def test_wrong_end_keeper_role_cannot_establish_opponent_control(self):
        ball = np.tile([10., 30.], (50, 1))
        players = np.tile([[[11., 30.], [10.2, 30.]]], (50, 1, 1))
        control, _ = evidence(self.track(ball, players, [0, 1],
                                        np.array([False, True])), [(0, 49)])
        np.testing.assert_array_equal(control[5:], np.zeros(45, dtype=np.int8))

    def test_occlusion_and_brief_deflection_retain_last_owner(self):
        control = np.r_[np.zeros(15), np.full(20, -1), np.ones(2),
                        np.full(13, -1)].astype(np.int8)
        restart = np.r_[np.zeros(3), np.full(47, -1)].astype(np.int8)
        labels = decode(control, restart, np.zeros(50, dtype=bool), [(0, 49)])
        np.testing.assert_array_equal(labels[2:], np.full(48, "H"))

    def test_stoppage_and_new_half_clear_previous_owner_and_pending_control(self):
        control = np.r_[np.zeros(15), np.ones(20)].astype(np.int8)
        restart = np.r_[np.zeros(3), np.full(32, -1)].astype(np.int8)
        tagged = np.zeros(35, dtype=bool)
        tagged[12] = True
        labels = decode(control, restart, tagged, [(0, 19), (25, 34)])
        np.testing.assert_array_equal(labels[:12], ["D", "D"] + ["H"] * 10)
        np.testing.assert_array_equal(labels[12:20], np.full(8, "D"))
        np.testing.assert_array_equal(labels[20:25], np.full(5, "X"))
        np.testing.assert_array_equal(labels[25:], np.full(10, "D"))


if __name__ == "__main__":
    unittest.main()
