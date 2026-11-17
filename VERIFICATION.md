# MNIST release verification

## Release check — November 16, 2026

The `solution` release was built as an ordinary wheel and installed offline
into a fresh Python 3.12.14 virtual environment with NumPy 2.5.4. No editable
installation or system-site packages were used. Tests and inference ran from
an unrelated directory with Python isolation (`-I`).

## Results

- All 33 tests present at the time of that check passed against both source and the installed wheel. The suite has since grown to 131 tests, all passing against the source tree.
- The wheel contains the 21 model members, `solution.json`, and both licence
  notices. Model loading resolves to the installed package's `site-packages`.
- Comparison with the pre-change exports found all 1,097,406 numeric values
  unchanged across the 22 JSON files. Only release identifiers and filenames
  were renamed.
- Two independent installed CLI processes, using `--candidate solution`,
  reproduced the submitted CSV in 68.36 and 68.44 seconds on this host.
- The two generated files and `MNIST.csv` have identical bytes and SHA256:

```text
16cd851d2f55c28f34f5ea0b30b45bc88d59c38f3e809182a0a132fcb8a96f4e
```

The unmodified official `evaluation/validate_submission.py` accepted all
three files: 199,319 rows, frames 0–199318, H=54,180, A=33,205, D=71,472,
X=40,462. The validator reports 309 possessions with mean duration 11.3 s.
These counts describe predictions and do not measure their correctness.

All relative Markdown links resolve. A release-file audit found no competition
tracking, tagging or media; `MNIST.csv` is the sole included frame-level output.
The removed release identifier is absent from current source, documentation,
model contents and filenames. `git diff --check` passed.

## Independence check — November 17, 2026

- Repaired the copied `.venv` entry points and activation scripts, and replaced
  its stale editable installation with an ordinary wheel installation.
- A scan of source, configuration and local environment text found no references
  to the original checkout or the obsolete copied-environment prefix.
- Built a wheel from a clean source copy and installed it offline in a fresh
  environment. All 131 tests passed with `-I` from a separate directory using
  copied tests; the wheel's build-source directory was no longer at its original
  path. The installed package loaded all 21 model members.
- Source tests also passed: 131 tests and 194 subtests.
- Rechecked the reviewer's ordinary installed wheel: every non-cache file in
  its `possession/` directory matches the current checkout byte for byte
  (runtime source commit `ebf0b73`). Its pip installation record identifies
  wheel SHA256
  `3188e3774d74c34f4b310b6fbae7b3d1e880789693f4fd1e9e373032bcb1ca46`.
- Repeated full real-match inference from that wheel with `-I`, outside the
  checkout, in 64.30 seconds. The new output, the reviewer's output, and
  `MNIST.csv` are byte-identical, with the SHA256 shown above. The dataset
  was supplied explicitly for verification; it is not bundled or a default
  dependency on another checkout.
- Reran the official checker on both outputs: accepted 199,319 rows, with the
  same label counts and possession summary as above. Checker revision:
  `f8f2629ff1e0985f81d7e59d8413678514ee1530`; its validator, metrics helper
  and `games.py` were unmodified. The reviewer's original run logs were not
  present, so these claims use the fresh run and direct file comparisons.

## Coverage and complexity — November 17, 2026

Python 3.12.14, NumPy 2.5.4, coverage.py 7.17.3 and Radon 6.0.1:

- Package statement coverage: **1,482/1,541 = 96.17%**.
- Package branch coverage: **374/420 = 89.05%**.
- **159/159 functions have CRAP < 10**; maximum 9.797012.

These totals come from `unittest` with explicit `--source=possession`.
The earlier report's 506-branch denominator used a different measurement
setup and is superseded here. Do not use coverage.py's combined
`percent_covered` as statement coverage; compute statements and branches
separately.

The checked-in [metric script](tools/check_metrics.py) uses Radon's
module-level function/method records, not nested closures as separate
functions. For each function's inclusive line range, coverage is executed
statements divided by executed plus missing statements. CRAP is
`CC² × (1 − coverage)³ + CC`, where CC is Radon's cyclomatic complexity.
It fails on unmeasured functions and scores **>= 10**. Branch coverage is
reported separately and is not substituted into that formula.

From the repository root:

```bash
.venv/bin/python -m pip install 'coverage==7.17.3' 'radon==6.0.1'
mkdir -p local_only
COVERAGE_FILE=local_only/.coverage .venv/bin/python -m coverage run --branch --source=possession -m unittest discover -s tests -q
COVERAGE_FILE=local_only/.coverage .venv/bin/python -m coverage json -o local_only/coverage.json
.venv/bin/python tools/check_metrics.py local_only/coverage.json
```

This run passed all 131 tests. The separate pytest run passed 131 tests and
194 subtests (pytest 9.2.0). Boundary tests exercise selected failure cases;
they are **not equivalent to mutation testing**. The earlier mutation tools
produced no usable results (zero coverage or no mutants), so no mutation
score or mutation-testing success is claimed.

## Verification commands

From the repository root, build and install a wheel into an isolated
environment under the ignored `local_only/` folder:

```bash
python3.12 -m pip wheel --wheel-dir local_only/wheels '.[bench]'
python3.12 -m venv local_only/check
local_only/check/bin/python -m pip install --no-index --find-links local_only/wheels 'mnist-possession[bench]==1.0.0'
```

Run from a working directory that does not contain the source package.
Place the dataset under `data/` and obtain `official-repo/` as described
in [README.md](README.md) before running inference and the checker:

```bash
cd local_only
check/bin/python -I -m unittest discover -s ../tests -v
check/bin/python -I -m possession.run --candidate solution --out reproduced.csv --data-dir ../data
check/bin/python ../official-repo/evaluation/validate_submission.py reproduced.csv
cmp reproduced.csv ../MNIST.csv
sha256sum reproduced.csv ../MNIST.csv
```

No public-ground-truth
scoring, model selection or parameter fitting was repeated in this release
verification. Hidden-match accuracy remains unknown. Publication and Google
Form submission are separate actions and have not been performed.
