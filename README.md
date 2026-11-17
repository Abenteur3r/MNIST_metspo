# MNIST_metspo — Football Possession Inference

MNIST estimates team possession from football tracking data for [More Than A Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), a competition that ranks possession-spell predictions. The implementation uses tactical XY tracking, team and goalkeeper metadata, and automatic Smart Tagging to produce one label per video frame: `H` (home possession), `A` (away possession), `D` (dead ball), or `X` (outside graded intervals). Inference runs on the CPU using frozen model parameters.

## Method

Possession depends on control over time: the nearest player may change during a loose ball, a challenge or a tracking gap without a corresponding change of possession. The model therefore combines geometric observations with player interactions and temporal context. Tracking and tag features describe ball motion, player proximity, team movement and event context at 5 Hz. Validity indicators distinguish missing observations from measured values.

A shared player embedding and two-head self-attention represent interactions between observed players. Masked mean and maximum pooling within each team provide a fixed-size representation despite variation in the number of visible players. Two parallel dilated temporal convolution paths combine this representation with tracking and tag features. A shared readout estimates dead-ball probability and home possession conditional on live play; these estimates define joint H/A/D probabilities.

Duration-aware dynamic programming combines the averaged probabilities with transition priors and geometric control evidence to select a possession sequence. Duration priors penalize fragmentation, while control-gated transitions prevent proximity alone from forcing an ownership change. Each graded half is processed independently, and decoded labels are expanded to the original 25 Hz time base.

> **Architecture diagram placeholder:** tracking/tag features and pooled player attention → two temporal convolution paths → H/A/D probabilities → ensemble averaging → duration-aware decoding. A distributable illustration is not included in this repository.

See [MNIST.md](MNIST.md) for the mathematical formulation, possession rules and experimental history.

## Model and training

The frozen ensemble contains **21 models**: seven held-match folds × three random seeds (`2026`, `2027`, `2028`). Each member was trained for 24 epochs on six of the seven external DFL Bundesliga matches, excluding its held-out match from both fitting and normalization. Inference averages all 21 probability outputs; decoder priors and control constants were fitted on all seven external matches.

All submitted learned parameters derive from [DFL/IDSSE Bundesliga data, version 1](https://doi.org/10.6084/m9.figshare.28196177.v1). No target-match labels were used for training, tuning or candidate selection; the published target labels were used only for pipeline correctness checks.

This is an **inference-only package** containing model weights, normalization statistics, decoder parameters and inference code. The training and external benchmarking harnesses are not included. Reproducibility here means regenerating the submitted CSV from the supplied models; it does not extend to retraining or rerunning the development benchmarks. [MNIST.md](MNIST.md) documents training provenance and the other datasets used during development.

## External evaluation

The table summarizes historical external development results from [MNIST.md](MNIST.md). DFL results describe development cross-validation; SkillCorner results describe evaluation of the final ensemble on a previously examined external panel. These are recorded development measurements rather than newly rerun benchmarks or competition scores. Higher spellF1 is better.

| External panel | Recorded candidate spellF1 | Recorded reference spellF1 |
| :--- | ---: | :--- |
| Seven-match degraded DFL cross-validation | 0.533151 | 0.504072 — earlier snapshot-attention reference |
| Previously examined 20-match SkillCorner panel | 0.392735 | 0.393600 and 0.395500 — two reference scores |

The recorded DFL improvement was 0.029079 spellF1, with gains on all seven matches. On SkillCorner, the final ensemble scored below both references but met the prospectively fixed transfer tolerance. The evidence therefore supports an improvement on the DFL development panel, without a corresponding improvement in mean SkillCorner spellF1.

Repeated use of the external panels, reuse of discovery models and adaptive follow-up experiments limit the independence of these comparisons. The results do not establish hidden-test accuracy or leaderboard placement; [MNIST.md](MNIST.md) describes the evaluation history and its qualifications.

## Quick start

Use Python 3.12 (tested on 3.12.14). The pinned runtime dependency is NumPy 2.5.4; no GPU or training framework is required. Run these commands from the repository root, where `pyproject.toml` and `MNIST.csv` are located:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m possession.run --out reproduced.csv --data-dir "/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
```

Extract the competition dataset and pass the directory directly containing `tactical/`. The required structure is:

```text
More Than A Hack 2027 - Initial Challenge - Dataset/
└── tactical/
    ├── BAR-ATH_tactical_atd.txt
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_smart_tagging.xml
```

Alternatively, set `MTAH_DATA_DIR` and omit `--data-dir`. Choose a new output path; the command refuses to overwrite existing files. All 21 models are packaged with the code. Inference requires no network access or runtime fitting and fixes numerical-library thread counts to one.

## Reproducibility

### Verify the CSV bytes

The supplied [MNIST.csv](MNIST.csv) contains 199,319 data rows, with header `frame,label` and frame indices `0` through `199318`. With the specified competition inputs, the inference command is intended to reproduce its exact bytes, matching the [official submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md). Verify both files:

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

### Run the official format validator

Use Git and an internet connection to obtain the unmodified official checker. The optional `bench` dependency installs pandas 3.1.0 for validation; it is not needed for inference. Choose a clone destination that does not already exist:

```bash
.venv/bin/python -m pip install '.[bench]'
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git /tmp/mnist-official-checker
.venv/bin/python /tmp/mnist-official-checker/evaluation/validate_submission.py reproduced.csv
```

The validator checks CSV format, row count, frame ordering and the H/A/D/X alphabet. It does not measure prediction accuracy; the SHA-256 comparison separately checks exact reproduction.

### Run regression tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover input parsing, feature construction, model and parameter contracts, ensemble probabilities, control rules, duration decoding and pipeline dispatch. They do not replace full-match reproduction or an independent accuracy evaluation.

## Repository structure

```text
possession/
  run.py          Command-line entry point and CSV output
  data/           Match constants, tracking and Smart Tagging loaders
  features/       Geometry, motion, player tokens and set-piece observations
  inference/      Ensemble loading, attention, temporal network and decoding
  models/         Frozen ensemble manifest and 21 model members
tests/            Input, feature and inference regression tests
MNIST.csv         Submission labels
MNIST.md          Technical report, provenance, evidence and limitations
pyproject.toml    Package metadata and pinned runtime/validator dependencies
requirements.txt Pinned NumPy runtime dependency
LICENSE           Original-code licence, upstream notices and data attribution
```

## Limitations

Differences in tracking quality and possession semantics between the external datasets and the competition match pose a risk to generalization. Provider labels may not follow the same controlled-possession definition. The external benchmarks do not isolate the contribution of these differences to prediction errors.

Noisy tracking and unmarked player identity switches can corrupt local velocity features. Missing-ball observations reduce the evidence available for control and restart decisions; validity masks retain information about missingness but cannot reconstruct absent measurements.

Sampling at 5 Hz and duration priors can merge or suppress very short possessions. Temporal context also uses future observations within each half, so the implementation is intended for offline processing. See [MNIST.md](MNIST.md) for detailed failure modes and the limits of the external evaluation.

## Documentation, data and licensing

- [Technical report — MNIST.md](MNIST.md): methodology, training data and annotation provenance, external experiments, possession rules, limitations and reproduction details.
- [Official submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) and [competition rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md): required deliverables and permitted data use.
- [LICENSE](LICENSE): MIT licence for original code, the Metrica Sports upstream MIT notice, and training-data/model attribution. DFL data are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the code's MIT licence does not replace obligations associated with the training data or derived parameters.

Competition tracking, tagging, homography, video and audio are not redistributed. Obtain the dataset from the organisers under their data-use terms. External source datasets are also not bundled; their sources and licences are listed in [MNIST.md](MNIST.md) and [LICENSE](LICENSE).

## Team

Dimash Rakishev, Dauren Izdibay, Diyar Ildartov, Ayan Kokhan and Aneliya Kuneyeva — Team MNIST.
