# MNIST technical-document figures

The three statistical/pipeline figures follow the palette, sans-serif typography, minimal
spines and vector-export conventions of
[scientific-figure-making](https://github.com/ChenLiu-1996/figures4papers/blob/main/scientific-figure-making/SKILL.md)
from [figures4papers](https://github.com/ChenLiu-1996/figures4papers).
The plotting code is specific to this project; upstream example measurements
are not used. Inference dependencies and model parameters are unchanged.

| Figure | Preview | Vector exports |
| --- | --- | --- |
| Pipeline and external training | [PNG](pipeline.png) | [PDF](pipeline.pdf), [SVG](pipeline.svg) |
| Detailed neural architecture | [PNG](architecture.png) | [PDF](architecture.pdf), [SVG](architecture.svg), [HTML](architecture.html) |
| Layer-by-layer PlotNeuralNet architecture | [PNG](plotneuralnet.png) | [PDF](plotneuralnet.pdf), [SVG](plotneuralnet.svg), [TeX](plotneuralnet.tex) |
| External scores and paired uncertainty | [PNG](external_validation.png) | [PDF](external_validation.pdf), [SVG](external_validation.svg) |
| Per-match gains and duration limitations | [PNG](diagnostics.png) | [PDF](diagnostics.pdf), [SVG](diagnostics.svg) |

## Reproduce

Use a separate environment; plotting is optional and is not part of inference.
From the repository root:

```bash
python3.12 -m venv .venv-figures
.venv-figures/bin/python -m pip install -r figures/requirements.txt
.venv-figures/bin/python figures/generate.py
```

The pinned plotting environment uses Matplotlib 3.10.8. Output is PNG at 300 dpi,
PDF with embedded TrueType text, and SVG with editable text. The script validates
score differences, paired-interval bounds, match means, architecture dimensions
and the duration-bin numerators/denominators before saving. It needs only the
included aggregate evidence and never opens competition data or public labels.

## Evidence and interpretation

[evidence.json](evidence.json) is a neutral-name extract from the completed
November 9 external development and transfer records. Their SHA256 hashes and
the evaluation freeze commit are retained in its provenance fields. These
records predate this figure-building task; no experiments or GT checks were
rerun. The export keeps full-precision aggregate values. The main scores,
paired intervals and limitations agree with [MNIST.md](../MNIST.md).

- **Reference:** the previous 21-member snapshot-attention ensemble. This is
  our internal comparator, not an official baseline or another team's entry.
- **DFL:** seven held-match development folds under the existing degraded
  tracking grid. Mean spellF1 is 0.5331507643 versus 0.5040721269.
- **SkillCorner:** the previously examined 20-match transfer panel. Mean
  spellF1 is 0.3927350842 versus 0.3936004732. The small negative difference is
  shown explicitly; passing a transfer tolerance is not a mean-score gain.
- **Uncertainty:** the original paired whole-match bootstrap, 20,000 draws,
  seed 20261106. Intervals describe the development comparison; repeated
  selection on these panels limits independent confirmation.
- **Durations:** possession recall uses temporal IoU 0.5. Matched/true counts
  are 52/3240 (≤1 s), 1305/7392 (1–3 s), 5003/8784 (3–10 s), and 4532/5072
  (>10 s). Truth spells repeat across degradation-grid units, so these are
  pooled diagnostic counts, not counts of unique independent possessions.
- **Units:** one percentage point (pp) is 0.01 on the spellF1 scale. The
  +2.908 pp DFL gain is the absolute +0.029079 change, not +2.908% relative.

These figures do not show public-window or hidden-match competition scores.
The fixed inputs make them reproducible without the original research folder.

## Detailed architecture diagram

[architecture.html](architecture.html) uses the
[diagram-design skill](https://github.com/cathrynlavery/diagram-design/blob/main/skills/diagram-design/SKILL.md)
with an architecture layout, a fitted 1280 × 1120 canvas, presentation-size
node typography, nine nodes and nine orthogonal connectors. The skin is
inherited from the existing MNIST figures: white paper, dark ink, blue focal
nodes and DejaVu Sans with Arial/sans-serif fallback. System fonts keep the
HTML and SVG offline; PNG and PDF preserve the checked rendering. Training
and the 21 repeated members are summarized, while one member is shown in
detail. The figure is original project-specific markup, with the skill's
layout and export conventions.

The generator verifies the dimensions of all 21 packaged JSON models,
including token MLPs, attention projections, both convolution paths and the
shared readout. It reads model parameters, not competition tracking or labels.

Generate HTML and PDF in the separate plotting environment:

```bash
.venv-figures/bin/python -m pip install -r figures/requirements-architecture.txt
PLAYWRIGHT_SKIP_BROWSER_GC=1 .venv-figures/bin/python -m playwright install chromium --only-shell
.venv-figures/bin/python figures/generate_architecture.py --pdf
```

For SVG/PNG exports and the skill's static checks, use the upstream helpers
from an optional checkout of `cathrynlavery/diagram-design` inside the
ignored `local_only/` folder:

```bash
mkdir -p local_only
git clone https://github.com/cathrynlavery/diagram-design.git local_only/diagram-design
.venv-figures/bin/python local_only/diagram-design/skills/diagram-design/scripts/self_check.py --offline figures/architecture.html
.venv-figures/bin/python local_only/diagram-design/skills/diagram-design/scripts/verify_geometry.py figures/architecture.html
.venv-figures/bin/python local_only/diagram-design/skills/diagram-design/scripts/export_svg.py figures/architecture.html figures/architecture.svg --system-fonts
.venv-figures/bin/python local_only/diagram-design/skills/diagram-design/scripts/export_png.py figures/architecture.html figures/architecture.png 3
```

The 3× PNG is 3840 × 3360 pixels. Release checks passed the upstream offline
self-check and geometry validator, confirmed every node's text fits in
Chromium, and checked that horizontal scrolling stays local at a 390px mobile
viewport. No remote requests or browser errors were observed. The exported
PDF was reopened and visually inspected as one searchable page. Model weights
and the submission CSV were unchanged.

## Layer-by-layer PlotNeuralNet diagram

[plotneuralnet.pdf](plotneuralnet.pdf) is a separate architecture figure made
with the actual [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
Python API and TikZ Box primitives, rather than a diagram in a similar style.
The visual references are the upstream `examples/fcn8s/fcn8.tex` and
`examples/Unet/Unet.tex`: full-depth tensor plates, grouped amber convolution
layers with orange ReLU bands, red pooling and serif labels. A single connected
network shows the player encoder, attention, feature fusion, all ten Conv1D
layers, shared readout and joint probabilities. The footer summarizes ensemble
averaging and decoding. Channel counts and dilations are checked against all
21 packaged models before source generation. Cuboid sizes are schematic; red
pooling aggregates players, and every temporal block preserves sequence length
$T$. There are no Conv2D operations or temporal downsampling.

The requested rendering methods now have separate outputs and build sources:

| Method | Artifact | Build source |
| --- | --- | --- |
| diagram-design | [architecture.html](architecture.html) | [generate_architecture.py](generate_architecture.py) |
| PlotNeuralNet / TikZ | [plotneuralnet.pdf](plotneuralnet.pdf) | [generate_plotneuralnet.py](generate_plotneuralnet.py), [generated TeX](plotneuralnet.tex) |

The upstream drawing primitives and MIT licence are included under
[vendor/plotneuralnet](vendor/plotneuralnet/README.md). Source generation needs
only Python's standard library. Rendering was verified with Tectonic 0.15.0
(the portable Linux musl build) and PyMuPDF 2.1:

```bash
.venv-figures/bin/python -m pip install -r figures/requirements-plotneuralnet.txt
.venv-figures/bin/python figures/generate_plotneuralnet.py --render --tectonic /path/to/working/tectonic
```

A working `tectonic` on PATH can be used without the `--tectonic` argument.
The compiler may fetch its TeX packages on first use. To generate only editable
LaTeX source, omit `--render`. The result is one vector PDF page, a 300 dpi
PNG and an SVG derived from the compiled page. TeX intermediates are ignored.
This build does not read competition data or modify the submitted CSV.
