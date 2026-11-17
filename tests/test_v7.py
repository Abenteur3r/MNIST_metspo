"""Historical v7 synthetic contracts; the real-data runner is not imported."""
import unittest
import numpy as np
from possession.v7 import v7_labels, TEAM_HOME, TEAM_AWAY, V1_D, STATE_HOME, STATE_AWAY


def selftest():
    """Check independently specified restart, re-entry, and motion contracts."""
    n = 100
    seen = np.ones(n, dtype=bool)
    gapd = np.zeros(n, dtype=bool)
    dists = np.full(n, 2.0, dtype=np.float32)
    teams = np.full(n, TEAM_HOME, dtype=np.int8)
    dead = np.zeros(n, dtype=bool)
    stoppage = dead.copy()
    stoppage[30:40] = True

    # Expected runs are hand-specified, not computed by historical candidates.
    cases = [
        ("restart after three strong moving observations", teams,
         np.r_[np.full(30, 15.), np.zeros(10), np.full(60, 15.)],
         stoppage, gapd, {0}, None,
         [(V1_D, 2), (STATE_HOME, 28), (V1_D, 12), (STATE_HOME, 58)]),
        ("no motion never restarts", teams,
         np.r_[np.full(30, 15.), np.zeros(70)],
         stoppage, gapd, {0}, None,
         [(V1_D, 2), (STATE_HOME, 28), (V1_D, 70)]),
        ("re-entry needs ten still observations and five slow moving ones", teams,
         np.r_[np.full(30, 15.), np.zeros(10), np.ones(60)],
         dead, gapd, {0}, None,
         [(V1_D, 2), (STATE_HOME, 37), (V1_D, 5), (STATE_HOME, 56)]),
        ("still emissions do not anticipate the restart team",
         np.r_[np.full(60, TEAM_HOME), np.full(40, TEAM_AWAY)].astype(np.int8),
         np.r_[np.full(60, 15.), np.zeros(20), np.full(20, 15.)],
         dead, gapd, {0}, None,
         [(V1_D, 2), (STATE_HOME, 67), (V1_D, 15), (STATE_AWAY, 16)]),
        ("ordinary restart threshold is strictly above two", teams,
         np.full(n, 2.), dead, gapd, {0}, None, [(V1_D, n)]),
        ("unknown velocity is not stationary evidence", teams,
         np.r_[np.full(10, 15.), np.full(20, np.nan), np.full(70, 15.)],
         dead, gapd, {0}, None, [(V1_D, 2), (STATE_HOME, 98)]),
        ("invalid finite velocity is not stationary evidence", teams,
         np.r_[np.full(10, 15.), np.zeros(20), np.full(70, 15.)],
         dead, gapd, {0}, np.r_[np.ones(10, dtype=bool),
                               np.zeros(20, dtype=bool), np.ones(70, dtype=bool)],
         [(V1_D, 2), (STATE_HOME, 98)]),
        ("half reset clears restart evidence", teams, np.full(n, 15.),
         dead, gapd, {0, 50}, None,
         [(V1_D, 2), (STATE_HOME, 48), (V1_D, 2), (STATE_HOME, 48)]),
        ("long gap starts a new restart window", teams, np.full(n, 15.),
         dead, np.arange(n) == 50, {0}, None,
         [(V1_D, 2), (STATE_HOME, 48), (V1_D, 3), (STATE_HOME, 47)]),
        ("re-entry restart threshold is strictly above one half", teams,
         np.r_[np.full(30, 15.), np.zeros(10), np.full(60, 0.5)],
         dead, gapd, {0}, None,
         [(V1_D, 2), (STATE_HOME, 37), (V1_D, 61)]),
    ]
    for name, tc, speed, dm, gm, reset, valid, runs in cases:
        expected = np.concatenate([np.full(length, state, dtype=np.int8)
                                   for state, length in runs])
        actual = v7_labels(tc, dists, seen, speed, dm, gm, reset,
                           [(0, n - 1)], n, velocity_valid=valid)
        np.testing.assert_array_equal(actual, expected, err_msg=name)


class V7Tests(unittest.TestCase):
    def test_ten_restart_reentry_motion_contracts(self):
        selftest()
