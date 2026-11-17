"""Shared candidate API for tactical inference and external benchmark halves.

Inputs contain only XY tracking, metadata, automatic intervals and frame blocks.
Neither ground truth nor ball height is accepted at this boundary.
"""

import numpy as np

from .baseline import smooth
from .control import evidence
from .dead import dead_region_mask
from .decode import decode
from .features import ball_velocity

CANDIDATES = ("baseline_smooth50", "v7", "control_v1")
ALL_CANDIDATES = (*CANDIDATES, "solution")


def nearest(tracking):
    n = len(tracking.ball)
    best = np.full(n, np.inf, dtype=np.float32)
    teams = np.full(n, -1, dtype=np.int8)
    seen = np.isfinite(tracking.ball).all(axis=1)
    for slot, team in enumerate(tracking.teams):
        delta = tracking.players[:, slot] - tracking.ball
        distance2 = np.sum(delta * delta, axis=1)
        better = seen & np.isfinite(distance2) & (distance2 < best)
        best[better], teams[better] = distance2[better], team
    return teams, np.sqrt(best), seen


def _control_labels(tracking, intervals, blocks):
    n = len(tracking.ball)
    control, restart = evidence(tracking, blocks)
    return decode(control, restart, dead_region_mask(n, intervals), blocks)


def _smooth_labels(teams, blocks, n):
    labels = np.full(n, "X", dtype="<U1")
    # Independent blocks cannot borrow votes from another half.
    for first, last in blocks:
        labels[first:last + 1] = smooth(teams[first:last + 1], 50)
    return labels


def _v7_labels(teams, distances, seen, tracking, intervals, blocks, n):
    from .v7 import v7_labels
    velocity, valid = ball_velocity(tracking.ball[:, 0], tracking.ball[:, 1], seen)
    state = v7_labels(teams, distances, seen, velocity, dead_region_mask(n, intervals),
                      np.zeros(n, dtype=bool), {first for first, _ in blocks}, blocks, n,
                      velocity_valid=valid)
    labels = np.full(n, "X", dtype="<U1")
    letters = np.array(["D", "H", "A"])
    for first, last in blocks:
        labels[first:last + 1] = letters[state[first:last + 1]]
    return labels


def predict(name, tracking, intervals, blocks, *, team_names=None):
    if name == "solution":
        from .shared_temporal_serving import predict as predict_shared
        return predict_shared(tracking, intervals, blocks, team_names=team_names)
    if name not in CANDIDATES:
        raise ValueError("Unknown candidate")
    n = len(tracking.ball)
    if name == "control_v1":
        return _control_labels(tracking, intervals, blocks)
    teams, distances, seen = nearest(tracking)
    if name == "baseline_smooth50":
        return _smooth_labels(teams, blocks, n)
    return _v7_labels(teams, distances, seen, tracking, intervals, blocks, n)
