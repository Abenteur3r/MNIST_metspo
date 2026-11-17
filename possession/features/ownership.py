"""Shared anonymous XY features for externally trained ownership models."""

import warnings
import numpy as np

from possession.features.observations import BASE_STRIDE, extract as extract_observations

SCHEMA = "ownership-111-masked-v1"


def velocity(xy, indices, window):
    seen = np.isfinite(xy).all(axis=-1)
    missing = np.concatenate((np.zeros_like(seen[:1], dtype=np.int32), np.cumsum(~seen, axis=0)))
    left = np.maximum(0, indices - window)
    valid = missing[indices + 1] == missing[left]
    valid &= (indices >= window).reshape((-1,) + (1,) * (valid.ndim - 1))
    delta = (xy[indices] - xy[left]) * (25 / window)
    return np.where(valid[..., None], delta, np.nan), valid


def _rank_columns(add, tracking, relative, distance, pv, pm, bv, bm, rows, team):
    eligible = np.isfinite(distance) & (tracking.teams == team)
    ranked = np.argsort(np.where(eligible, distance, np.inf), axis=1, kind="stable")
    for rank in range(3):
        slot = ranked[:, min(rank, ranked.shape[1] - 1)]
        present = eligible[rows, slot] & (rank < eligible.sum(axis=1))
        moving = present & pm[rows, slot] & bm
        prefix = f"team{team}_rank{rank + 1}_"
        values = (relative[rows, slot, 0], relative[rows, slot, 1], distance[rows, slot],
                  pv[rows, slot, 0], pv[rows, slot, 1], np.linalg.norm(pv[rows, slot] - bv, axis=1))
        for k, (name, value) in enumerate(zip(("dx", "dy", "distance", "vx", "vy", "relative_speed"), values)):
            add(prefix + name, np.where(present if k < 3 else moving, value, np.nan))
        add(prefix + "seen", present)
        add(prefix + "motion_seen", moving)
        add(prefix + "keeper", np.where(present, tracking.keepers[slot], np.nan))


def _ball_columns(add, ball, bv, seen, bm, base, index):
    for j, name in enumerate(("ball_x", "ball_y")):
        add(name, ball[:, j])
    for j, name in enumerate(("ball_vx", "ball_vy")):
        add(name, bv[:, j])
    add("ball_speed", np.linalg.norm(bv, axis=1))
    add("ball_seen", seen)
    add("ball_motion_seen", bm)
    add("ball_absence_seconds", base["dead_features"][:, 4])
    add("half_start", index == 0)
    for j, name in enumerate(base["feature_names"]):
        add("m2_" + str(name), base["dead_features"][:, j])


def _team_columns(add, tracking, players, pv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team in (0, 1):
            slots = tracking.teams == team
            xy, vv = players[:, slots], pv[:, slots]
            center, spread = np.nanmean(xy, axis=1), np.nanstd(xy, axis=1)
            mean_v = np.nanmean(vv, axis=1)
            speed = np.nanmedian(np.linalg.norm(vv, axis=2), axis=1)
            line = np.nanquantile(xy[:, :, 0], [.1, .9], axis=1)
            values = (center[:, 0], center[:, 1], spread[:, 0], spread[:, 1], mean_v[:, 0], mean_v[:, 1],
                      speed, line[0], line[1], np.isfinite(xy).all(axis=2).sum(axis=1))
            for name, value in zip(("center_x", "center_y", "spread_x", "spread_y", "mean_vx", "mean_vy",
                                    "median_speed", "line_x10", "line_x90", "visible_players"), values):
                add(f"team{team}_" + name, value)


def extract(tracking, intervals, window=5):
    """Return raw NaN-preserving features and control evidence; no labels or height."""
    base = extract_observations(tracking, intervals)
    n = len(tracking.ball)
    index = np.arange(0, n, BASE_STRIDE)
    ball, players = tracking.ball[index], tracking.players[index]
    bv, bm = velocity(tracking.ball, index, window)
    pv, pm = velocity(tracking.players, index, window)
    relative = players - ball[:, None]
    distance = np.linalg.norm(relative, axis=2)
    seen = np.isfinite(ball).all(axis=1)
    columns, names = [], []

    def add(name, value):
        names.append(name)
        columns.append(value)

    rows = np.arange(len(index))
    for team in (0, 1):
        _rank_columns(add, tracking, relative, distance, pv, pm, bv, bm, rows, team)
    _ball_columns(add, ball, bv, seen, bm, base, index)
    _team_columns(add, tracking, players, pv)
    return np.column_stack(columns).astype(np.float32), names, base


def normalize(values, mean, scale):
    """Encode missingness separately from centered numeric features; never fit here."""
    valid = np.isfinite(values)
    centered = np.where(valid, (values - np.asarray(mean)) / np.asarray(scale), 0.)
    return np.concatenate((centered, valid.astype(np.float32)), axis=1).astype(np.float32)
