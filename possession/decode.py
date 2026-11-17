"""Original control_v1 ownership state machine."""

from collections import deque

import numpy as np

CONTROL_FRAMES = 5


def _starts(tagged_dead):
    """First frame of each tagged stretch; the tag itself starts at frame 0."""
    return tagged_dead & np.r_[True, ~tagged_dead[:-1]] if len(tagged_dead) else tagged_dead


def _owner_from_votes(moving):
    """The majority restart team, once three or more restart votes are in."""
    observed = [team for team in moving if team >= 0]
    if len(observed) < 3:
        return -1
    return max((0, 1), key=observed.count)


def _fill_block(labels, starts, control, restart, first, last):
    owner = -1
    pending = -1
    count = 0
    moving = deque(maxlen=10)
    for frame in range(first, last + 1):
        if starts[frame]:
            owner, pending, count = -1, -1, 0
            moving.clear()
        if owner == -1:
            moving.append(int(restart[frame]))
            owner = _owner_from_votes(moving)
            if owner < 0:
                labels[frame] = "D"
                continue
            moving.clear()
        candidate = int(control[frame])
        if candidate < 0:
            pending, count = -1, 0
        else:
            count = count + 1 if candidate == pending else 1
            pending = candidate
            if count >= CONTROL_FRAMES:
                owner = candidate
                labels[frame - CONTROL_FRAMES + 1:frame + 1] = "HA"[owner]
        labels[frame] = "HA"[owner]


def decode(control, restart, tagged_dead, blocks):
    """Changes require consecutive control; stillness alone never creates D.

    Confirmation is offline: backdate only to the first controlled frame,
    never into the preceding flight. Explicit stoppage starts and half resets
    clear ownership and all pending evidence.
    """
    n = len(control)
    labels = np.full(n, "X", dtype="<U1")
    starts = _starts(tagged_dead)
    for first, last in blocks:
        _fill_block(labels, starts, control, restart, first, last)
    return labels
