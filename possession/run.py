"""Regenerate possession labels on CPU, without network or historical code."""

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Set before importing NumPy. No GPU library is part of this candidate.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .candidates import ALL_CANDIDATES, predict
from .game import BLOCKS, asset_paths
from .io_atd import load_tracking
from .io_tagging import load_tagging


def write_submission(path, labels):
    """Write the original CSV bytes with LF newlines; refuse to overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, label) for f, label in enumerate(labels))
    return path


def regenerate(out, *, data_dir=None, candidate="control_v1"):
    if Path(out).exists():
        raise FileExistsError("Output exists; choose a new output path.")
    paths = asset_paths(data_dir)
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("Required tactical assets are missing; check MTAH_DATA_DIR.")
    tracking = load_tracking(paths["atd_raw"], paths["atd_meta"])
    options = {}
    if candidate == "solution":
        from .tag_features import metadata_team_names
        try:
            options["team_names"] = metadata_team_names(ET.parse(paths["atd_meta"]).getroot())
        except ET.ParseError:
            raise ValueError("Invalid tactical team metadata.") from None
    labels = predict(candidate, tracking, load_tagging(paths["smart_tagging"]), BLOCKS, **options)
    write_submission(out, labels)
    return {label: int(np.sum(labels == label)) for label in "HADX"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=ALL_CANDIDATES)
    parser.add_argument("--out", required=True, type=Path, metavar="OUTPUT.csv")
    parser.add_argument("--data-dir", type=Path, help="Overrides MTAH_DATA_DIR")
    args = parser.parse_args(argv)
    try:
        counts = regenerate(args.out, data_dir=args.data_dir, candidate=args.candidate)
    except (OSError, ValueError):
        # Do not echo exception strings: third-party parsers may include input data.
        parser.exit(1, "Regeneration failed. Check the data root, tactical input format, "
                    "and a writable output path that does not already exist.\n")
    print(json.dumps({"label_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
