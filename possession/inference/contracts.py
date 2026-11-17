"""Tracking, control and prior contracts for the packaged ensemble."""

from numbers import Integral, Real
from pathlib import Path
import numpy as np


MATCHES = ("J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9", "J03WMX", "J03WN1")


SEEDS = (2026, 2027, 2028)


def _finite_array(value, shape, *, positive=False):
    array = np.asarray(value)
    return (array.shape == shape and np.issubdtype(array.dtype, np.number)
            and np.isrealobj(array) and np.isfinite(array).all()
            and (not positive or (array > 0).all()))


def _member_filename(value):
    return (isinstance(value, str) and value.endswith(".json") and bool(value)
            and Path(value).name == value and "/" not in value and "\\" not in value
            and value not in (".", ".."))


_CONTROL_REALS = ("keeper_distance", "control_distance", "relative_speed", "control_speed",
                  "team_margin", "restart_distance", "restart_speed", "confirmation",
                  "restart_window", "restart_votes")


_CONTROL_COUNTS = ("confirmation", "restart_window", "restart_votes")


_DECODER_REALS = ("confidence", "min_quantile", "switch_penalty", "duration_weight")


def _nonnegative_reals(control):
    return all(isinstance(control[key], Real) and np.isfinite(control[key]) and control[key] >= 0
               for key in _CONTROL_REALS)


def _positive_counts(control):
    return all(isinstance(control[key], Integral) and not isinstance(control[key], bool)
               and control[key] >= 1 for key in _CONTROL_COUNTS)


def _control_contract(control, decoder):
    return (control["stride"] == 5 and control["window"] == 5 and decoder["dead_mix"] == 0
            and _nonnegative_reals(control) and _positive_counts(control))


def _decoder_settings(decoder):
    return (all(isinstance(decoder[key], Real) and np.isfinite(decoder[key])
                for key in _DECODER_REALS)
            and 0 <= decoder["confidence"] <= 1 and 0 <= decoder["min_quantile"] <= 1
            and decoder["switch_penalty"] >= 0 and decoder["duration_weight"] >= 0)


def _progress_priors(priors):
    if (priors["training_matches"] != list(MATCHES)
            or not _finite_array(priors["transition"], (3, 3))
            or not _finite_array(priors["initial"], (3,))):
        return False
    transition, initial = np.asarray(priors["transition"]), np.asarray(priors["initial"])
    return not ((transition < 0).any() or (initial < 0).any()
                or not np.allclose(transition.sum(axis=1), 1., atol=1e-6, rtol=0.)
                or not np.isclose(initial.sum(), 1., atol=1e-6, rtol=0.))


def _duration_contract(durations):
    return (len(durations) == 3 and all(np.asarray(values).ndim == 1 and len(values)
            and _finite_array(values, (len(values),), positive=True) for values in durations))


def _decoder_contract(artifact):
    control, decoder, priors = (artifact[key] for key in ("control", "decoder", "priors"))
    return (_control_contract(control, decoder) and _decoder_settings(decoder)
            and _progress_priors(priors) and _duration_contract(priors["durations"]))


def _shape_contract(ball, players, teams, keepers):
    return (ball.ndim == 2 and ball.shape[1] == 2 and len(ball) > 0
            and players.ndim == 3 and players.shape[0] == len(ball) and players.shape[2] == 2
            and players.shape[1] > 0 and teams.shape == keepers.shape == (players.shape[1],))


def _coordinate_contract(ball, players):
    return (np.issubdtype(ball.dtype, np.floating) and np.issubdtype(players.dtype, np.floating)
            and not np.isinf(ball).any() and not np.isinf(players).any())


def _team_contract(teams, keepers):
    return (np.isin(teams, [0, 1]).all() and all((teams == team).any() for team in (0, 1))
            and np.issubdtype(keepers.dtype, np.bool_))


def _tracking_contract(tracking):
    try:
        ball, players, teams, keepers = (np.asarray(getattr(tracking, key))
                                         for key in ("ball", "players", "teams", "keepers"))
        valid = (_shape_contract(ball, players, teams, keepers)
                 and _coordinate_contract(ball, players)
                 and _team_contract(teams, keepers)
                 and isinstance(tracking.home_defends_left, (bool, np.bool_))
                 and _finite_array(tracking.pitch, (2,), positive=True))
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("MNIST requires aligned XY tracking and two-team slot metadata")
