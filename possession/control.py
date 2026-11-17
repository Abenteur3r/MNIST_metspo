"""Control-gated evidence, ported without threshold or numerical changes."""

import numpy as np

from .features import ball_velocity
from .io_atd import Tracking

WINDOW = 5
CONTROL_DISTANCE = 2.0
KEEPER_DISTANCE = 2.5
RELATIVE_SPEED = 3.0
CONTROL_SPEED = 8.0
TEAM_MARGIN = 0.5


def _keeper_reliable(tracking, slot, team, blocks, visible, length):
    """A keeper-role track in the opposite half cannot establish a turnover."""
    reliable = np.zeros(len(visible), dtype=bool)
    x = tracking.players[:, slot, 0]
    for half, (first, last) in enumerate(blocks):
        home_left = tracking.home_defends_left != bool(half % 2)
        defends_left = home_left if team == 0 else not home_left
        span = slice(first, last + 1)
        reliable[span] = visible[span] & ((x[span] < length / 2) if defends_left else (x[span] > length / 2))
    return reliable


def evidence(tracking: Tracking, blocks: list[tuple[int, int]]):
    """Return control and restart observations, -1 where unsupported.

    Require continuous local co-motion, not just nearest projected position.
    A keeper-role track in the opposite half is unreliable for its metadata
    team and cannot establish a turnover. It is ignored, never relabelled.
    """
    ball, players = tracking.ball, tracking.players
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    speed, speed_valid = ball_velocity(ball[:, 0], ball[:, 1], seen, WINDOW)
    best_control = np.full((2, n), np.inf)
    best_near = np.full((2, n), np.inf)
    length = tracking.pitch[0]
    for slot, team in enumerate(tracking.teams):
        relative = ball - players[:, slot]
        visible = seen & np.isfinite(players[:, slot]).all(axis=1)
        distance = np.linalg.norm(relative, axis=1)
        if tracking.keepers[slot]:
            reliable = _keeper_reliable(tracking, slot, team, blocks, visible, length)
        else:
            reliable = visible.copy()
        best_near[team] = np.minimum(best_near[team], np.where(reliable, distance, np.inf))
        missing = np.r_[0, np.cumsum(~reliable)]
        valid = np.zeros(n, dtype=bool)
        relative_speed = np.full(n, np.nan)
        if n > WINDOW:
            valid[WINDOW:] = missing[WINDOW + 1:] - missing[:n - WINDOW] == 0
            relative_speed[WINDOW:] = np.linalg.norm(
                relative[WINDOW:] - relative[:-WINDOW], axis=1) * 25 / WINDOW
        # No co-motion estimate can straddle a half/block boundary.
        for first, _ in blocks:
            valid[first:first + WINDOW] = False
        radius = KEEPER_DISTANCE if tracking.keepers[slot] else CONTROL_DISTANCE
        controlled = (valid & speed_valid & (speed <= CONTROL_SPEED)
                      & (relative_speed <= RELATIVE_SPEED) & (distance <= radius))
        best_control[team] = np.minimum(
            best_control[team], np.where(controlled, distance, np.inf))
    control = np.full(n, -1, dtype=np.int8)
    near = np.full(n, -1, dtype=np.int8)
    for team in (0, 1):
        control[(best_control[team] + TEAM_MARGIN < best_control[1 - team])] = team
        near[(best_near[team] < 6.) & (best_near[team] < best_near[1 - team])] = team
    restart = np.where(speed_valid & (speed > 2.), near, -1).astype(np.int8)
    return control, restart
