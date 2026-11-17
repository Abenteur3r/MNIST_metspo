"""Read tactical tracking locally; errors never include input values or rows."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .game import N_FRAMES, PITCH


@dataclass(frozen=True)
class Tracking:
    """Metre coordinates and per-slot metadata; slots are not persistent IDs."""

    players: np.ndarray
    ball: np.ndarray
    teams: np.ndarray
    keepers: np.ndarray
    home_defends_left: bool
    pitch: tuple[float, float] = PITCH


def _players(root):
    """Return the player elements and the home team element."""
    return list(root.iter("Player")), next(root.iter("Team"))


def _team_labels(players, home):
    """Home slots are 0, every other slot is 1."""
    return np.array([0 if p.get("teamId") == home.get("id") else 1
                     for p in players], dtype=np.int8)


def _keeper_flags(players):
    """A slot is a keeper when its provider metadata says Goalkeeper."""
    return np.array([
        any(p.findtext("Name") == "position_type"
            and p.findtext("Value") == "Goalkeeper"
            for p in player.iter("ProviderParameter"))
        for player in players
    ], dtype=bool)


def _attack_direction(home):
    """Read the home side's first-half attack direction."""
    return next(p.findtext("Value") for p in home.iter("ProviderParameter")
                if p.findtext("Name") == "attack_direction_first_half")


def _read_metadata(meta_path):
    """Return team labels, keeper flags, attack direction and the slot count."""
    try:
        root = ET.parse(meta_path).getroot()
        players, home = _players(root)
        teams = _team_labels(players, home)
        keepers = _keeper_flags(players)
        direction = _attack_direction(home)
        if not players or direction not in ("left_to_right", "right_to_left"):
            raise ValueError
    except (ET.ParseError, StopIteration, ValueError):
        raise ValueError("Invalid tactical metadata or unsupported attack direction.") from None
    return teams, keepers, direction, len(players)


def _player_values(groups, slots):
    """Split and parse exactly one three-field group per slot."""
    player_groups = groups.split(";")
    if len(player_groups) != slots or any(len(group.split(",")) != 3
                                         for group in player_groups):
        raise ValueError
    return [[float(v) for v in group.split(",")[:2]] for group in player_groups]


def _ball_value(ball_group):
    """Parse the ball group, which carries the same three fields."""
    if len(ball_group.split(",")) != 3:
        raise ValueError
    return [float(v) for v in ball_group.split(",")[:2]]


def _frame_row(line, slots, n_frames, filled):
    """Parse one ATD row, rejecting malformed counters, widths or repeats."""
    counter, groups, ball_group = line.strip().split(":")
    frame = int(counter) - 1
    if not 0 <= frame < n_frames or filled[frame]:
        raise ValueError
    return frame, _player_values(groups, slots), _ball_value(ball_group)


def _read_rows(raw_path, slots, n_frames):
    """Fill one XY array per slot and the ball array from the raw ATD stream."""
    xy = np.full((n_frames, slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    filled = np.zeros(n_frames, dtype=bool)
    try:
        with open(raw_path, encoding="utf-8") as handle:
            for line in handle:
                frame, row, point = _frame_row(line, slots, n_frames, filled)
                xy[frame], ball[frame], filled[frame] = row, point, True
    except (ValueError, IndexError):
        raise ValueError("Invalid ATD row, slot count, or frame counter.") from None
    return xy, ball


def load_tracking(raw_path, meta_path, *, n_frames=N_FRAMES, pitch=PITCH):
    """Preserve the original float32 parse, then scale normalised XY to metres.

    ATD counters are one-based; output arrays use video frame = counter - 1.
    Unobserved rows and NaN coordinates stay missing. Provided speeds are unused.
    """
    teams, keepers, direction, slots = _read_metadata(meta_path)
    xy, ball = _read_rows(raw_path, slots, n_frames)
    xy *= pitch
    ball *= pitch
    return Tracking(xy, ball, teams, keepers, direction == "left_to_right", tuple(pitch))
