"""Length-aware semi-Markov decoder (SALD-L).

This module is opt-in and does not change the MNIST decoder. The existing
``duration_decode`` collapses every duration at or above the minimum into one
geometric-tail state, so it cannot score an exact segment length. This
implementation keeps an explicit age state, which is what a
``length_weight * log(length)`` term needs.

``length_weight=0`` reproduces ``duration_decode`` when the same ``allowed``,
``minimum``, ``means``, ``transition``, ``initial``, ``penalty`` and
``prior_weight`` are supplied. That equivalence is covered by
``tests/test_m2_decode_length.py``.

Complexity is ``O(n * 3 * max_length)``. Passing ``max_length=None`` uses
``n``, which is exact but quadratic; a smaller cap trades exactness for speed.
"""

from __future__ import annotations

import numpy as np


def _validated(emissions, length_weight, prior_weight, penalty):
    """Shape and scalar guards; return the float matrix and its frame count."""
    emissions = np.asarray(emissions, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] != 3:
        raise ValueError("emissions must have shape (n, 3)")
    n = emissions.shape[0]
    if n == 0:
        raise ValueError("emissions must not be empty")
    if not np.isfinite(length_weight):
        raise ValueError("length_weight must be finite")
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    return emissions, n


def _lengths(minimum, max_length, n):
    """Clamp the minimum durations and the cap to the frame count."""
    minimum = np.maximum(1, np.asarray(minimum, dtype=np.int64))
    if max_length is None:
        max_length = n
    max_length = int(max_length)
    if max_length < int(minimum.max()):
        raise ValueError("max_length must be at least max(minimum)")
    return minimum, min(max_length, n)


def _prior(means, minimum, transition, initial, prior_weight, penalty):
    """Score one segment length and one class switch."""
    means = np.asarray(means, dtype=np.float64)
    transition = np.asarray(transition, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    hazard = 1.0 / np.maximum(1.0 + means - minimum, 1e-6)
    if prior_weight != 0:
        log_stay = prior_weight * np.log1p(-hazard)
        with np.errstate(divide="ignore"):
            log_turn = prior_weight * (np.log(hazard)[:, None] + np.log(transition)) - penalty
        initial_log = np.log(initial)
    else:
        log_stay = np.zeros(3, dtype=np.float64)
        log_turn = np.full((3, 3), -float(penalty), dtype=np.float64)
        initial_log = np.zeros(3, dtype=np.float64)
    np.fill_diagonal(log_turn, -np.inf)
    return log_stay, log_turn, initial_log


def _switch(score, t, c, minimum, max_length, log_turn, allowed, length_weight):
    """Best completed segment that may switch into class ``c`` at frame ``t``."""
    best = -np.inf
    best_source = -1
    best_age = -1
    for source in range(3):
        if source == c or not allowed[t, source, c]:
            continue
        lowest = int(minimum[source]) - 1
        if lowest >= max_length:
            continue
        ages = np.arange(lowest, max_length, dtype=np.int64)
        candidates = (score[source, ages] + log_turn[source, c]
                      + length_weight * np.log(ages + 1.0))
        index = int(np.argmax(candidates))
        if candidates[index] > best:
            best = float(candidates[index])
            best_source = source
            best_age = int(ages[index])
    return best, best_source, best_age


def _advance(score, emissions, t, minimum, max_length, log_stay, log_turn,
             allowed, length_weight):
    """One decoder step: grow each live segment or open the best switch."""
    nxt = np.full((3, max_length), -np.inf, dtype=np.float64)
    team = np.full((3, max_length), -1, dtype=np.int8)
    age = np.full((3, max_length), -1, dtype=np.int16)
    lengths = np.arange(1, max_length, dtype=np.int64) if max_length > 1 else None
    for c in range(3):
        # Stay: the current segment grows by one anchor. The shifted-
        # geometric prior applies only after the minimum duration.
        if max_length > 1:
            stay = np.where(lengths >= minimum[c], log_stay[c], 0.0)
            nxt[c, 1:] = score[c, :-1] + stay + emissions[t, c]
            team[c, 1:] = c
            age[c, 1:] = np.arange(max_length - 1, dtype=np.int16)

        # Switch: a completed segment of length a >= minimum[source].
        best, best_source, best_age = _switch(
            score, t, c, minimum, max_length, log_turn, allowed, length_weight)
        if best_source >= 0:
            nxt[c, 0] = best + emissions[t, c]
            team[c, 0] = best_source
            age[c, 0] = best_age
    return nxt, team, age


def _backtrack(parent_team, parent_age, n, team, age):
    """Walk the parent links back to the first frame."""
    path = np.empty(n, dtype=np.int8)
    t = n - 1
    c = int(team)
    a = int(age)
    while True:
        path[t] = c
        if t == 0:
            break
        pc = int(parent_team[t, c, a])
        pa = int(parent_age[t, c, a])
        if pc < 0 or pa < 0:
            raise ValueError("broken backtrack")
        c = pc
        a = pa
        t -= 1
    return path


def duration_decode_length(
    emissions: np.ndarray,
    allowed: np.ndarray,
    minimum: np.ndarray,
    means: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    penalty: float,
    prior_weight: float,
    length_weight: float,
    max_length: int | None = None,
) -> np.ndarray:
    """Decode a 3-state sequence with an exact segment-length score.

    Parameters follow ``possession.m2_decode.duration_decode``. ``length_weight``
    is the SALD-L coefficient: positive values reward splitting at fixed total
    length (``log`` is concave), negative values reward fewer, longer segments.
    """
    emissions, n = _validated(emissions, length_weight, prior_weight, penalty)
    minimum, max_length = _lengths(minimum, max_length, n)
    log_stay, log_turn, initial_log = _prior(
        means, minimum, transition, initial, prior_weight, penalty)

    score = np.full((3, max_length), -np.inf, dtype=np.float64)
    score[:, 0] = emissions[0] + prior_weight * initial_log

    parent_team = np.full((n, 3, max_length), -1, dtype=np.int8)
    parent_age = np.full((n, 3, max_length), -1, dtype=np.int16)

    for t in range(1, n):
        score, parent_team[t], parent_age[t] = _advance(
            score, emissions, t, minimum, max_length, log_stay, log_turn,
            allowed, length_weight)

    # The final segment is right-censored; score its observed length.
    final = score + length_weight * np.log(np.arange(1, max_length + 1))
    flat = int(np.argmax(final))
    team, age = np.unravel_index(flat, final.shape)
    if not np.isfinite(final[team, age]):
        raise ValueError("no finite path")
    return _backtrack(parent_team, parent_age, n, team, age)
