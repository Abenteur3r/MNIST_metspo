"""Set-piece team/type observations; no timing extension, fitting or file access."""

import unicodedata
import numpy as np

NAMES = ("setpiece_home", "setpiece_away", "setpiece_kickoff", "setpiece_throw_in",
         "setpiece_corner", "setpiece_free_kick", "setpiece_goal_kick", "setpiece_penalty")
_TYPES = {"kickoff": "kickoff", "kickoffstart": "kickoff", "kickoffaftergoal": "kickoff",
          "throwin": "throw_in", "corner": "corner", "cornerkick": "corner",
          "freekick": "free_kick", "goalkick": "goal_kick", "penalty": "penalty",
          "penaltykick": "penalty"}


def token(value):
    if value is None:
        return ""
    return "".join(char for char in unicodedata.normalize("NFKD", str(value)).casefold()
                   if char.isalnum())


def metadata_team_names(root):
    """Read an already parsed metadata element, in the existing home/away order."""
    names = tuple(team.findtext("Name") for team in root.iter("Team"))
    if len(names) != 2 or any(not token(name) for name in names):
        raise ValueError("Expected two metadata team names")
    _team_mapping(names)
    return names


def _team_mapping(team_names):
    mapping = {"h": "H", "a": "A"}
    if team_names is None:
        return mapping
    if len(team_names) != 2:
        raise ValueError("Team names must follow the home/away metadata order")
    for name, side in zip(team_names, "HA"):
        key = token(name)
        if not key or key == "na" or (key in mapping and mapping[key] != side):
            raise ValueError("Ambiguous metadata team names")
        mapping[key] = side
    return mapping


def canonicalize_intervals(intervals, team_names=None):
    """Resolve only supplied team/type strings; never infer an unknown owner."""
    mapping = _team_mapping(team_names)
    result = []
    for row in intervals:
        tags = dict(row.get("tags", {}))
        values = {token(key): value for key, value in tags.items()}
        tags["Team"] = mapping.get(token(values.get("team")))
        kind = token(values.get("type"))
        tags["Type"] = _TYPES.get(kind)
        result.append(dict(row, tags=tags))
    return result


def bits(intervals, n_frames, *, team_names=None, stride=5):
    if n_frames < 0 or stride != 5:
        raise ValueError("Set-piece observations require the fixed five-frame stride")
    indices = np.arange(0, n_frames, stride)
    values = np.zeros((len(indices), len(NAMES)), np.float32)
    for row in canonicalize_intervals(intervals, team_names):
        if token(row["code"]) != "setpieces":
            continue  # GOALS auxiliary team metadata is intentionally unused.
        active = (indices >= max(0, row["start_frame"])) & (indices < min(n_frames, row["end_frame"]))
        team, kind = row["tags"]["Team"], row["tags"]["Type"]
        if team in ("H", "A"):
            values[active, "HA".index(team)] = 1
        if kind is not None:
            values[active, NAMES.index("setpiece_"+kind)] = 1
    return values
