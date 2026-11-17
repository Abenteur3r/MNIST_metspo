# MNIST — Football Possession Inference

CPU-only submission for More Than A Hack 2027. The algorithm assigns one
`H` (home), `A` (away), `D` (dead ball) or `X` (ungraded) label per video frame.
Team MNIST: Dimash Rakishev, Dauren Izdibay, Diyar Ildartov, Ayan Kokhan and
Aneliya Kuneyeva.

## Install and run

Use Python 3.12 (tested on 3.12.14). Run these commands from the repository
root, where `pyproject.toml` and `MNIST.csv` are located:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m possession.run --out reproduced.csv --data-dir "/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
```

Extract the competition dataset and pass the directory directly containing
`tactical/`. It must contain:

```text
tactical/BAR-ATH_tactical_atd.txt
tactical/BAR-ATH_tactical_atd.xml
tactical/BAR-ATH_tactical_smart_tagging.xml
```

Alternatively, set `MTAH_DATA_DIR` and omit `--data-dir`. Choose a new output
path; the command refuses to overwrite existing files. All 21 models are
packaged with the code. Inference uses NumPy only, with no network access or
runtime training. The command fixes numerical-library thread counts to one.

## Verify the output

The supplied [MNIST.csv](MNIST.csv) contains 199,319 data rows, with header
`frame,label`. With the specified competition inputs, the command reproduces
its exact bytes. Compare the files:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
expected = "16cd851d2f55c28f34f5ea0b30b45bc88d59c38f3e809182a0a132fcb8a96f4e"
for name in ("MNIST.csv", "reproduced.csv"):
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, (name, actual)
print("Both files match the submission SHA256.")
PY
```

To run the official format checker, use Git and an internet connection to
obtain it. Choose a clone destination that does not already exist:

```bash
.venv/bin/python -m pip install '.[bench]'
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git /tmp/mnist-official-checker
.venv/bin/python /tmp/mnist-official-checker/evaluation/validate_submission.py reproduced.csv
```

This validates the format, not prediction accuracy. Run regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Repository layout

```text
possession/
  run.py          Command-line entry point and CSV output
  data/           Match constants, tracking and Smart Tagging loaders
  features/       Geometry, motion, player tokens and set-piece observations
  inference/      Ensemble loading, attention, temporal network and decoding
  models/         Frozen ensemble manifest and 21 model members
tests/            Input, feature and inference regression tests
MNIST.csv         Submission labels
MNIST.md          Method, training data, evidence and limitations
LICENSE           Original-code licence, upstream notices and data attribution
```

The 21-member ensemble combines player attention with two temporal convolution
paths, then decodes possession spells using duration and transition priors.
All learned parameters were fitted on seven external DFL Bundesliga matches.
No target-match labels were used for training, tuning or candidate selection.
See [MNIST.md](MNIST.md) for the complete method and data disclosures.

Original code is MIT-licensed. [LICENSE](LICENSE) includes upstream notices
and attribution for parameters derived from CC BY 4.0 DFL data. Competition
tracking, tagging and media are not distributed here.
