# MNIST — Football Possession Inference

CPU-only possession labelling for More Than A Hack 2027: one `H`, `A`, `D` or
`X` label per tactical-video frame. Team MNIST: Dimash Rakishev, Dauren
Izdibay, Diyar Ildartov, Ayan Kokhan and Aneliya Kuneyeva.

## Install and reproduce

Run these commands from the repository root with Python 3.12 (tested on 3.12.14):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
export MTAH_DATA_DIR="/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
.venv/bin/python -m possession.run --candidate solution --out reproduced.csv --data-dir "$MTAH_DATA_DIR"
```

The data root must contain `tactical/BAR-ATH_tactical_atd.txt`,
`tactical/BAR-ATH_tactical_atd.xml` and
`tactical/BAR-ATH_tactical_smart_tagging.xml`. The output path must be new;
the CLI refuses to overwrite a file. An installed package also runs from
outside the repository. All 21 models and the manifest are included in the
package; inference uses NumPy only, no network or training libraries.

This checkout is independent of the original project: source, model files
and tests are included here. You can keep the dataset under `data/` (ignored
by Git) and pass `--data-dir ./data`. No sibling checkout is required.
Python and installed dependencies are still required; after moving or copying
the project, create a new `.venv` rather than copying an existing environment.

The supplied submission is [MNIST.csv](MNIST.csv): **199,319 data rows** with
header `frame,label`. The `solution` command regenerates those exact bytes.
Runtime is approximately 65 seconds on the original certification CPU and
varies with hardware. Thread counts are fixed before NumPy is imported.

Compare the result with the supplied file:

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

## Validate with the official checker

Install the pinned optional evaluation dependency and obtain the official
repository in the ignored `official-repo/` folder inside this project:

```bash
.venv/bin/python -m pip install '.[bench]'
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git official-repo
.venv/bin/python official-repo/evaluation/validate_submission.py MNIST.csv
.venv/bin/python official-repo/evaluation/validate_submission.py reproduced.csv
```

Validation checks the file contract, not prediction accuracy. The private
full-match ground truth determines the competition result; public labels are
for pipeline correctness checks only.

## Method and files

A 21-member ensemble combines snapshot player attention with two temporal
convolution paths. It predicts joint H/A/D probabilities, then uses
externally fitted duration and transition priors to decode possession spells.
All learned parameters were fitted on seven external DFL Bundesliga matches.
No target-match labels were used for training, tuning or candidate selection.

- [MNIST.md](MNIST.md): approach, data, possession rules, evidence and limitations.
- `possession/`: inference source and the CLI.
- `possession/models/solution.json`: ensemble manifest.
- `possession/models/mnist_*.json`: 21 frozen model members.
- `tests/`: synthetic fixtures and regression tests.
- [figures/](figures/README.md): reproducible publication figures, PNG/PDF/SVG exports.
- [VERIFICATION.md](VERIFICATION.md): clean-install reproduction and validation results.

Run the tests after installation:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Licences and submission

Original code is MIT-licensed; see [LICENSE](LICENSE).
Upstream notices and external-data attribution are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
Model parameters derive from DFL data released under CC BY 4.0. Competition
tracking, tagging, homography and media are separately licensed and are not
included. `MNIST.csv` is the algorithm's submission output.

Submit `MNIST.csv`, `MNIST.md`, and the URL of this repository after making it
public, using the organisers' emailed Google Form. Use the registered team
name MNIST. The published deadline is November 17, 2026; check the registration
email for the exact cutoff and timezone. See the official
[submission instructions](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md).
