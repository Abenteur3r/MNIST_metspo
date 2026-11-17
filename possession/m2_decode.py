"""Exact duration-state DP with shifted-geometric priors and evidence-gated exits."""

import numpy as np
from .m2_control import channels, confirmations
from .m2_features import BASE_STRIDE


def _walk_parameters(minimum, means, transition, initial, penalty, prior_weight):
    """Expand the duration priors into per-state bounds and switching costs."""
    minimum = np.maximum(1, np.asarray(minimum, dtype=int))
    start = np.r_[0, np.cumsum(minimum)[:-1]]
    end = np.cumsum(minimum) - 1
    state = np.repeat(np.arange(3), minimum)
    hazard = 1 / np.maximum(1. + np.asarray(means) - minimum, 1. + np.finfo(float).eps)
    log_stay = prior_weight * np.log1p(-hazard)
    with np.errstate(divide="ignore"):
        # A disabled prior contributes zero, including at structural zero entries.
        # Avoid 0 * -inf; positive-weight M2 inference follows the original path.
        log_turn = (np.full((3, 3), -float(penalty)) if prior_weight == 0 else
                    prior_weight * (np.log(hazard)[:, None] + np.log(transition)) - penalty)
        initial_log = np.zeros(3) if prior_weight == 0 else np.log(initial)
    np.fill_diagonal(log_turn, -np.inf)
    return minimum, start, end, state, log_stay, log_turn, initial_log


def _advance(score, emission, allowed_t, ordinary_parent, walk):
    """One observation step: stay in the tail, or enter a state at its first frame."""
    minimum, start, end, state, log_stay, log_turn = walk[:6]
    parent = ordinary_parent.copy()
    new = score[parent]
    # The terminal age bucket has both an entering path and an unbounded tail.
    for s in range(3):
        tail = score[end[s]] + log_stay[s]
        if minimum[s] == 1 or tail >= new[end[s]]:
            new[end[s]], parent[end[s]] = tail, end[s]
    switches = np.where(allowed_t, score[end, None] + log_turn, -np.inf)
    source = np.argmax(switches, axis=0)
    entry = switches[source, np.arange(3)]
    for s in range(3):
        if minimum[s] > 1 or entry[s] > new[start[s]]:
            new[start[s]], parent[start[s]] = entry[s], end[source[s]]
    return new + emission[state], parent


def _backtrack(parents, state, n, index):
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state[index]
        if t:
            index = int(parents[t, index])
    return path


def duration_decode(emissions, allowed, minimum, means, transition, initial, penalty, prior_weight):
    """Maximize log evidence + duration/transition log priors - switch penalties.

    The final segment may be right-censored. Each state has deterministic minimum
    duration followed by an unbounded geometric tail; no maximum truncation.
    allowed[t, source, destination] applies at the exact transition observation.
    """
    n = len(emissions)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    walk = _walk_parameters(minimum, means, transition, initial, penalty, prior_weight)
    size = int(walk[0].sum())
    score = np.full(size, -np.inf)
    score[walk[1]] = emissions[0] + prior_weight * walk[6]
    parents = np.empty((n, size), dtype=np.int32)
    ordinary_parent = np.maximum(0, np.arange(size) - 1)
    for t in range(1, n):
        score, parents[t] = _advance(score, emissions[t], allowed[t], ordinary_parent, walk)
    index = int(np.argmax(score))
    if not np.isfinite(score[index]):
        raise ValueError("No feasible duration path")
    return _backtrack(parents, walk[3], n, index)


def decode(features, dead_probability, control_config, config, priors):
    control, restart = channels(features, control_config)
    confirmed, voted = confirmations(control, restart, control_config)
    stride = control_config["stride"]
    p = np.clip(dead_probability[::stride // BASE_STRIDE], np.finfo(float).eps, 1 - np.finfo(float).eps)
    evidence = np.column_stack((np.log1p(-p), np.log1p(-p), np.log(p))) * config["dead_weight"]
    for team in (0, 1):
        evidence[:, team] += config["owner_weight"] * ((control == team).astype(float) - (control == 1 - team))
    allowed = np.ones((len(p), 3, 3), dtype=bool)
    allowed[:, 0, 1] = confirmed == 1
    allowed[:, 1, 0] = confirmed == 0
    allowed[:, 2, 0] = voted == 0
    allowed[:, 2, 1] = voted == 1
    if "durations" in priors:
        minimum = [max(1, round(np.quantile(priors["durations"][s], config["min_quantile"]) / stride)) for s in range(3)]
        means = [np.mean(priors["durations"][s]) / stride for s in range(3)]
    else:
        minimum = [max(1, round(x / stride)) for x in priors["minimum_frames"]]
        means = np.asarray(priors["mean_frames"]) / stride
    if float(config.get("length_weight", 0.0)):
        from .m2_decode_length import duration_decode_length
        return duration_decode_length(evidence, allowed, minimum, means, np.asarray(priors["transition"]),
                                      np.asarray(priors["initial"]), config["switch_penalty"],
                                      config["duration_weight"], config["length_weight"], config.get("length_max"))
    return duration_decode(evidence, allowed, minimum, means, np.asarray(priors["transition"]),
                           np.asarray(priors["initial"]), config["switch_penalty"], config["duration_weight"])
