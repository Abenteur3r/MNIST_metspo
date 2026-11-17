"""Learned H/A/D evidence with control-gated duration decoding."""

import numpy as np
from possession.inference.control import channels, confirmations
from possession.inference.duration import duration_decode


def decode(features, learned, dead_probability, control, config, priors):
    if control["stride"] != 5 or learned.shape != (len(dead_probability), 3):
        raise ValueError("Learned ownership observation alignment mismatch")
    eps = np.finfo(float).eps
    owner = learned[:, :2] / np.maximum(learned[:, :2].sum(axis=1, keepdims=True), eps)
    dead = config["dead_mix"] * dead_probability + (1 - config["dead_mix"]) * learned[:, 2]
    dead = np.clip(dead, eps, 1 - eps)
    evidence = np.column_stack((np.log1p(-dead)[:, None] + np.log(np.clip(owner, eps, 1)), np.log(dead)))
    raw, restart = channels(features, control)
    confirmed, voted = confirmations(raw, restart, control)
    allowed = np.ones((len(dead), 3, 3), dtype=bool)
    for team in (0, 1):
        confident = owner[:, team] >= config["confidence"]
        allowed[:, 1 - team, team] = (confirmed == team) | confident
        allowed[:, 2, team] = (voted == team) | confident
    minimum = [max(1, round(np.quantile(x, config["min_quantile"]) / 5)) for x in priors["durations"]]
    means = [np.mean(x) / 5 for x in priors["durations"]]
    return duration_decode(evidence, allowed, minimum, means, np.asarray(priors["transition"]),
                           np.asarray(priors["initial"]), config["switch_penalty"], config["duration_weight"])
