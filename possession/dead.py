"""The control_v1 Smart Tagging prior."""

import numpy as np


def dead_region_mask(n, intervals):
    """Mark SET PIECES and GOALS; the decoder uses their union's onsets.

    A tagged interval resets possession at its start, but does not force D
    until its end: moving-near observations may establish an earlier restart.
    Keep the original case-sensitive code matching for control_v1 parity.
    """
    mask = np.zeros(n, dtype=bool)
    for row in intervals:
        if row["code"] in ("SET PIECES", "GOALS"):
            start, end = row["start_frame"], row["end_frame"]
            mask[max(0, start):min(n, end)] = True
    return mask
