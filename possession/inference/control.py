"""Control and restart evidence used to gate duration-decoder transitions."""

import numpy as np
from possession.features.observations import BASE_STRIDE, WINDOWS


def channels(features, config):
    step = config["stride"] // BASE_STRIDE
    distance = features["distance"][::step]
    speed = features["ball_speed"][::step, WINDOWS.index(config["window"])]
    relative = features["relative_speed"][::step, :, WINDOWS.index(config["window"])]
    valid = np.isfinite(distance)
    if config.get("keeper_half_exclusion", False):
        valid &= features["half_ok"][::step]
    radius = np.where(features["keepers"], config["keeper_distance"], config["control_distance"])
    controlled = valid & (distance <= radius) & (relative <= config["relative_speed"]) & (speed[:, None] <= config["control_speed"])
    best, near = [], []
    for team in (0, 1):
        slots = features["teams"] == team
        best.append(np.min(np.where(controlled[:, slots], distance[:, slots], np.inf), axis=1))
        near.append(np.min(np.where(valid[:, slots], distance[:, slots], np.inf), axis=1))
    control = np.full(len(speed), -1, dtype=np.int8)
    restart = np.full(len(speed), -1, dtype=np.int8)
    for team in (0, 1):
        control[best[team] + config["team_margin"] < best[1 - team]] = team
        restart[(near[team] < config["restart_distance"]) & (near[team] < near[1 - team])
                & (speed > config["restart_speed"])] = team
    return control, restart


def confirmations(control, restart, config):
    n = len(control)
    confirmed = np.full(n, -1, dtype=np.int8)
    votes = np.zeros((n, 2), dtype=int)
    end = np.arange(1, n + 1)
    for team in (0, 1):
        sums = np.r_[0, np.cumsum(control == team)]
        width = config["confirmation"]
        good = (end >= width) & (sums[end] - sums[np.maximum(0, end - width)] == width)
        confirmed[good] = team
        sums = np.r_[0, np.cumsum(restart == team)]
        votes[:, team] = sums[end] - sums[np.maximum(0, end - config["restart_window"])]
    voted = np.where(votes.sum(axis=1) >= config["restart_votes"], votes.argmax(axis=1), -1).astype(np.int8)
    return confirmed, voted


def expand(labels, features, stride):
    return np.array(["H", "A", "D"])[np.repeat(labels, stride)[:int(features["n_frames"])]]
