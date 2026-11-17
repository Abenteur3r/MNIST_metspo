"""Tactical input contract, adapted from the official games.py (MIT).

Copyright (c) 2026-2027 Metrica Sports. See LICENSE.
Only the published pitch, time base, asset names and graded blocks live here.
"""

import os
from pathlib import Path

FPS = 25
N_FRAMES = 199319
PITCH = (105.3, 68.0)
BLOCKS = ((11185, 90651), (110808, 190197))  # Both ends inclusive.


def asset_paths(data_dir=None):
    """Resolve required assets without reading data or relying on sibling repos."""
    root = data_dir or os.environ.get("MTAH_DATA_DIR")
    if not root:
        raise ValueError("Set MTAH_DATA_DIR or pass --data-dir with the dataset root.")
    folder = Path(root).expanduser() / "tactical"
    return {
        "atd_meta": folder / "BAR-ATH_tactical_atd.xml",
        "atd_raw": folder / "BAR-ATH_tactical_atd.txt",
        "smart_tagging": folder / "BAR-ATH_tactical_smart_tagging.xml",
    }
