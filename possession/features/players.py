"""Permutation-invariant player tokens."""

import numpy as np
from possession.features.ownership import velocity

TOKEN_NAMES = ("x", "y", "vx", "vy", "opponent_dx", "opponent_dy",
               "teammate_dx", "teammate_dy", "ball_dx", "ball_dy")


def extract_tokens(tracking):
    """Read only observed geometry; preserve unknowns and average exact neighbour ties."""
    indices = np.arange(0, len(tracking.ball), 5)
    xy = tracking.players[indices]
    valid = np.isfinite(xy).all(axis=-1)
    motion, _ = velocity(tracking.players, indices, 5)
    teams = np.asarray(tracking.teams)
    if not np.isin(teams, [0, 1]).all():
        raise ValueError("Unsupported player team metadata")
    values = np.full((*xy.shape[:2], 10), np.nan, np.float32)
    values[..., :2], values[..., 2:4] = xy, motion
    values[..., 8:] = tracking.ball[indices, None] - xy
    same = teams[:, None] == teams[None, :]
    identity = np.eye(len(teams), dtype=bool)
    for first in range(0, len(xy), 512):
        block = xy[first:first+512]
        available = valid[first:first+512]
        delta = block[:, None, :, :] - block[:, :, None, :]
        squared = np.sum(delta**2, axis=-1)
        usable = available[:, :, None] & available[:, None, :]
        for offset, eligible in ((4, ~same), (6, same & ~identity)):
            distances = np.where(usable & eligible[None], squared, np.inf)
            minimum = distances.min(axis=-1, keepdims=True)
            ties = (distances == minimum) & np.isfinite(distances)
            count = ties.sum(axis=-1)
            total = np.where(ties[..., None], delta, 0).sum(axis=2)
            values[first:first+len(block), :, offset:offset+2] = np.where(
                count[..., None] > 0, total / np.maximum(count[..., None], 1), np.nan)
    values[~valid] = np.nan
    slots = max(int((teams == t).sum()) for t in (0, 1))
    tokens = np.full((len(indices), 2, slots, 10), np.nan, np.float32)
    for team in (0, 1):
        selected = values[:, teams == team]
        tokens[:, team, :selected.shape[1]] = selected
    return tokens


def normalize_tokens(tokens, mean, scale):
    valid = np.isfinite(tokens)
    mask = valid[..., :2].all(axis=-1)
    centered = np.where(valid, (tokens - np.asarray(mean, np.float32)) /
                        np.asarray(scale, np.float32), 0).astype(np.float32)
    return np.concatenate((centered, valid.astype(np.float32)), axis=-1), mask
