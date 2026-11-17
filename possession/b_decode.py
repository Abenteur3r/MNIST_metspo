"""External learned ownership evidence with the exact M2 duration decoder."""

import numpy as np
from .m2_control import channels, confirmations
from .m2_decode import duration_decode


def decode(features, learned, m2_dead, control, config, priors):
    if control["stride"] != 5 or learned.shape != (len(m2_dead), 3):
        raise ValueError("Learned ownership observation alignment mismatch")
    eps = np.finfo(float).eps
    owner = learned[:, :2] / np.maximum(learned[:, :2].sum(axis=1, keepdims=True), eps)
    dead = config["dead_mix"] * m2_dead + (1 - config["dead_mix"]) * learned[:, 2]
    dead = np.clip(dead, eps, 1 - eps)
    evidence = np.column_stack((np.log1p(-dead)[:, None] + np.log(np.clip(owner, eps, 1)), np.log(dead)))
    raw, restart = channels(features, control)
    confirmed, voted = confirmations(raw, restart, control)
    allowed = np.ones((len(dead), 3, 3), dtype=bool)
    for team in (0, 1):
        confident = owner[:, team] >= config["confidence"]
        allowed[:, 1 - team, team] = (confirmed == team) | confident
        allowed[:, 2, team] = (voted == team) | confident
    if "durations" in priors:
        minimum = [max(1, round(np.quantile(x, config["min_quantile"]) / 5)) for x in priors["durations"]]
        means = [np.mean(x) / 5 for x in priors["durations"]]
    else:
        minimum = [max(1, round(x / 5)) for x in priors["minimum_frames"]]
        means = np.asarray(priors["mean_frames"]) / 5
    if float(config.get("length_weight", 0.0)):
        from .m2_decode_length import duration_decode_length
        return duration_decode_length(evidence, allowed, minimum, means, np.asarray(priors["transition"]),
                                      np.asarray(priors["initial"]), config["switch_penalty"],
                                      config["duration_weight"], config["length_weight"], config.get("length_max"))
    return duration_decode(evidence, allowed, minimum, means, np.asarray(priors["transition"]),
                           np.asarray(priors["initial"]), config["switch_penalty"], config["duration_weight"])
