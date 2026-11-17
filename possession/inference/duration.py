"""Exact duration-state dynamic programming with geometric tails."""

import numpy as np


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
        # Avoid 0 * -inf when the prior is disabled.
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
