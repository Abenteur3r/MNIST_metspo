"""Serve independent graded blocks and leave all other frames as X."""

from numbers import Integral
import numpy as np
from possession.data.tracking import Tracking
from possession.inference.contracts import _tracking_contract


def _integral_pair(row):
    first, last = row
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in (first, last)):
        raise ValueError
    return first, last


def _checked_blocks(blocks, frames):
    blocks = list(blocks)
    if not blocks:
        raise ValueError
    previous = -1
    for row in blocks:
        first, last = _integral_pair(row)
        if first < 0 or last < first or last >= frames or first <= previous:
            raise ValueError
        previous = last
    return blocks


def _checked_intervals(intervals):
    if not isinstance(intervals, (list, tuple)):
        raise ValueError
    for row in intervals:
        first, stop = _integral_pair((row["start_frame"], row["end_frame"]))
        if stop <= first:
            raise ValueError
    return intervals


def predict_blocks(tracking, intervals, blocks, *, team_names=None, half, loader):
    _tracking_contract(tracking)
    try:
        blocks = _checked_blocks(blocks, len(tracking.ball))
        intervals = _checked_intervals(intervals)
    except (KeyError, TypeError, ValueError):
        raise ValueError("MNIST requires valid nonoverlapping blocks and tag intervals") from None
    artifact, models = loader()
    labels = np.full(len(tracking.ball), "X", dtype="<U1")
    for k, (first, last) in enumerate(blocks):
        part = Tracking(tracking.players[first:last+1], tracking.ball[first:last+1], tracking.teams,
                        tracking.keepers, tracking.home_defends_left != bool(k % 2), tracking.pitch)
        tags = [dict(row, start_frame=row["start_frame"]-first, end_frame=row["end_frame"]-first)
                for row in intervals if row["end_frame"] > first and row["start_frame"] <= last]
        labels[first:last+1] = half(part, tags, artifact, models, team_names=team_names)
    return labels
