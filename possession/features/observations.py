"""Fixed observation feature bank; no fitting, labels, height or player identities."""

import warnings
import numpy as np

BASE_STRIDE = 5
WINDOWS = (5, 15, 25)


def velocity_at(xy, indices, window):
    seen = np.isfinite(xy).all(axis=-1)
    missing = np.concatenate((np.zeros_like(seen[:1], dtype=np.int32), np.cumsum(~seen, axis=0)))
    left = np.maximum(0, indices - window)
    valid = (missing[indices + 1] == missing[left])
    valid &= (indices >= window).reshape((-1,) + (1,) * (valid.ndim - 1))
    delta = (xy[indices] - xy[left]) * (25 / window)
    return np.where(valid, np.linalg.norm(delta, axis=-1), np.nan)


def rolling(values, width):
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - width)
    return np.divide(sums[end] - sums[start], counts[end] - counts[start],
                     out=np.full(len(values), np.nan), where=counts[end] > counts[start])


def _half_ok(players, tracking):
    """A keeper is valid only inside the half its team defends."""
    half_ok = np.ones(players.shape[:2], dtype=bool)
    for slot, team in enumerate(tracking.teams):
        if tracking.keepers[slot]:
            left = tracking.home_defends_left if team == 0 else not tracking.home_defends_left
            half_ok[:, slot] = (players[:, slot, 0] < tracking.pitch[0] / 2) == left
    return half_ok


def _absence(tracking, indices, n):
    """Ball-seen flags, seconds unseen, last point and its distance off the pitch."""
    seen = np.isfinite(tracking.ball).all(axis=1)
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    last_xy = tracking.ball[np.maximum(last[indices], 0)].copy()
    last_xy[last[indices] < 0] = np.nan
    absence = (indices - last[indices]) / 25
    off = np.max(np.column_stack((-last_xy, last_xy - tracking.pitch)), axis=1)
    return seen, absence, off, last_xy


def _speed_columns(add, player_speed, absence, off, players, tracking):
    """Player-speed quantiles, collective rates and per-team spread."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed_quantiles = np.nanquantile(player_speed, [.25, .5, .75], axis=1)
        for q, values in zip((25, 50, 75), speed_quantiles):
            add(f"player_speed_q{q}", values)
        for width in (5, 15):
            add(f"collective_mean_{width}", rolling(speed_quantiles[1], width))
        add("absence_times_collective", absence * speed_quantiles[1])
        add("absence_times_off_pitch", absence * off)
        centers = [np.nanmean(players[:, tracking.teams == t], axis=1) for t in (0, 1)]
        spread = [np.nanmean(np.linalg.norm(players[:, tracking.teams == t] - centers[t][:, None], axis=2), axis=1)
                  for t in (0, 1)]
        add("team_spread_min", np.minimum(*spread))
        add("team_spread_max", np.maximum(*spread))


def _distance_columns(add, distance):
    """Distance to the nearest and the third-nearest player."""
    sorted_distances = np.sort(np.where(np.isfinite(distance), distance, np.inf), axis=1)
    for k in (0, 2):
        add(f"nearest_player_{k + 1}_m", np.where(np.isfinite(sorted_distances[:, k]), sorted_distances[:, k], np.nan))


def _pitch_columns(add, last_xy, tracking):
    """Ball distance to the pitch landmarks."""
    length, width = tracking.pitch
    corners = np.array([[0, 0], [0, width], [length, 0], [length, width]])
    add("corner_distance", np.min(np.linalg.norm(last_xy[:, None] - corners, axis=2), axis=1))
    add("centre_distance", np.linalg.norm(last_xy - np.array(tracking.pitch) / 2, axis=1))
    add("touchline_distance", np.minimum(np.abs(last_xy[:, 1]), np.abs(last_xy[:, 1] - width)))
    add("goal_line_distance", np.minimum(np.abs(last_xy[:, 0]), np.abs(last_xy[:, 0] - length)))
    # Six-yard depth is pitch geometry, not a predictive threshold.
    add("six_yard_line_distance", np.minimum(np.abs(last_xy[:, 0] - 5.4864), np.abs(last_xy[:, 0] - length + 5.4864)))


def _tag_columns(add, intervals, indices, n):
    """Tag masks plus the time since and until each tag edge."""
    masks = {"SET PIECES": np.zeros(n, bool), "GOALS": np.zeros(n, bool)}
    starts, ends = [], []
    for row in intervals:
        if row["code"] not in masks:
            continue
        a, b = max(0, row["start_frame"]), min(n, row["end_frame"])
        if b > a:
            masks[row["code"]][a:b] = True
            starts.append(a)
            ends.append(b)
    for name, mask in masks.items():
        add(name.lower().replace(" ", "_"), mask[indices].astype(float))
    for name, stamps in (("tag_start", starts), ("tag_end", ends)):
        stamps = np.sort(stamps)
        if len(stamps):
            next_id = np.searchsorted(stamps, indices)
            previous = np.where(next_id > 0, (indices - stamps[np.maximum(0, next_id - 1)]) / 25, np.nan)
            following = np.where(next_id < len(stamps), (stamps[np.minimum(next_id, len(stamps) - 1)] - indices) / 25, np.nan)
        else:
            previous = following = np.full(len(indices), np.nan)
        add("since_" + name, previous)
        add("until_" + name, following)
    return masks


def extract(tracking, intervals):
    n = len(tracking.ball)
    indices = np.arange(0, n, BASE_STRIDE)
    ball, players = tracking.ball[indices], tracking.players[indices]
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    relative = tracking.ball[:, None] - tracking.players
    relative_speed = np.stack([velocity_at(relative, indices, w) for w in WINDOWS], axis=-1)
    ball_speed = np.column_stack([velocity_at(tracking.ball, indices, w) for w in WINDOWS])
    player_speed = velocity_at(tracking.players, indices, 25)
    half_ok = _half_ok(players, tracking)
    seen, absence, off, last_xy = _absence(tracking, indices, n)
    columns, names = [], []
    def add(name, values):
        names.append(name)
        columns.append(values)
    for j, w in enumerate(WINDOWS):
        add(f"ball_speed_{w}", ball_speed[:, j])
    add("ball_seen", seen[indices].astype(float))
    add("absence_seconds", absence)
    add("last_off_pitch_m", off)
    _speed_columns(add, player_speed, absence, off, players, tracking)
    _distance_columns(add, distance)
    _pitch_columns(add, last_xy, tracking)
    masks = _tag_columns(add, intervals, indices, n)
    return {"distance": distance, "relative_speed": relative_speed, "ball_speed": ball_speed,
            "half_ok": half_ok, "teams": tracking.teams, "keepers": tracking.keepers,
            "dead_features": np.column_stack(columns).astype(np.float32),
            "feature_names": np.asarray(names), "tagged": (masks["SET PIECES"] | masks["GOALS"])[indices],
            "n_frames": np.asarray(n)}
