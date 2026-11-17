"""The frozen 111 ownership features plus eight set-piece metadata bits."""

import numpy as np

from possession.features.ownership import extract as base_extract
from possession.features.set_pieces import NAMES, bits

SCHEMA = "ownership-119-tagged-masked-v1"


def extract(tracking, intervals, window=5, *, team_names=None):
    values, names, features = base_extract(tracking, intervals, window)
    if values.shape[1] != 111 or len(names) != 111:
        raise ValueError("Feature bank requires the frozen original ownership feature prefix")
    tags = bits(intervals, len(tracking.ball), team_names=team_names)
    return np.concatenate((values, tags), axis=1), names+list(NAMES), features
