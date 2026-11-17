"""Control ownership family with every decision constant supplied by a DFL artifact."""

from collections import deque
import numpy as np
from .m2_features import BASE_STRIDE, WINDOWS


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


def _restart_owner(moving, votes):
    """The majority restart vote once enough votes were seen, else nobody."""
    observed = [team for team in moving if team >= 0]
    if len(observed) < votes:
        return -1
    return max((0, 1), key=observed.count)


def _run_length(candidate, pending, count):
    """Length of the unbroken control run ending at this frame."""
    if candidate < 0:
        return 0, candidate
    return (count + 1 if candidate == pending else 1), candidate


def ownership(features, config, forced_dead=None):
    control, restart = channels(features, config)
    step = config["stride"] // BASE_STRIDE
    tagged = features["tagged"][::step]
    starts = tagged & np.r_[True, ~tagged[:-1]]
    labels = np.full(len(control), 2, dtype=np.int8)
    owner, pending, count = -1, -1, 0
    moving = deque(maxlen=config["restart_window"])
    for i in range(len(control)):
        dead = forced_dead is not None and forced_dead[i]
        if starts[i] or dead:
            owner, pending, count = -1, -1, 0
            moving.clear()
        if dead:
            continue
        if owner < 0:
            moving.append(int(restart[i]))
            owner = _restart_owner(moving, config["restart_votes"])
        if owner >= 0:
            count, pending = _run_length(int(control[i]), pending, count)
            if count >= config["confirmation"]:
                owner = pending
                labels[max(0, i - config["confirmation"] + 1):i + 1] = owner
            labels[i] = owner
    return labels


def stillness(features, config, settings):
    """V7-style D channel, with reselected speed/persistence and control settings."""
    control, restart = channels(features, config)
    _, voted = confirmations(control, restart, config)
    step = config["stride"] // BASE_STRIDE
    speed = features["ball_speed"][::step, WINDOWS.index(config["window"])]
    near = np.min(np.where(np.isfinite(features["distance"][::step]), features["distance"][::step], np.inf), axis=1)
    quiet = (speed <= settings["speed"]) & (near < config["restart_distance"])
    tagged = features["tagged"][::step]
    starts = tagged & np.r_[True, ~tagged[:-1]]
    dead = np.zeros(len(speed), bool)
    state, count = True, 0
    for i in range(len(speed)):
        count = count + 1 if quiet[i] else 0
        if starts[i] or count >= settings["samples"]:
            state = True
        elif voted[i] >= 0:
            state = False
        dead[i] = state
    return dead


def expand(labels, features, stride):
    return np.array(["H", "A", "D"])[np.repeat(labels, stride)[:int(features["n_frames"])]]
