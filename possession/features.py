"""Coordinate features shared by inference and future benchmark inputs."""

import numpy as np

from .game import FPS


def ball_velocity(ball_x, ball_y, ball_seen, window=5):
    """Return (speed in m/s, velocity_valid) on the inclusive [f-window, f].

    Unknown speeds are NaN, never stationary zero. Both displacement endpoints
    and every intervening observation must be visible and finite.
    """
    if window < 1:
        raise ValueError("window must be positive")
    n = len(ball_x)
    v = np.full(n, np.nan, dtype=np.float32)
    velocity_valid = np.zeros(n, dtype=bool)
    if n <= window:
        return v, velocity_valid
    observed = ball_seen & np.isfinite(ball_x) & np.isfinite(ball_y)
    missing_cum = np.concatenate(([0], np.cumsum(~observed)))
    ok = missing_cum[window + 1:] - missing_cum[:n - window] == 0
    velocity_valid[window:] = ok
    frames = np.flatnonzero(velocity_valid)
    fps = FPS
    dx = (ball_x[frames] - ball_x[frames - window]) * fps / window
    dy = (ball_y[frames] - ball_y[frames - window]) * fps / window
    v[frames] = np.hypot(dx, dy)
    return v, velocity_valid
