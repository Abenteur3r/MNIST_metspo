"""The frozen 111 ownership features plus eight set-piece metadata bits."""

import numpy as np

from .b_features import extract as base_extract
from .tag_features import NAMES, bits

SCHEMA = "ownership-119-tagged-masked-v1"


def extract(tracking, intervals, window=5, *, team_names=None):
    values, names, features = base_extract(tracking, intervals, window)
    if values.shape[1] != 111 or len(names) != 111:
        raise ValueError("W3 requires the frozen original ownership feature prefix")
    tags = bits(intervals, len(tracking.ball), team_names=team_names)
    return np.concatenate((values, tags), axis=1), names+list(NAMES), features
