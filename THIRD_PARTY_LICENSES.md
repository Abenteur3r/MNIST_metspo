# Third-party notices and attribution

## Metrica Sports — MIT

The tactical constants and asset naming in `possession/game.py`, Smart Tagging
parsing in `possession/io_tagging.py`, and majority vote in
`possession/baseline.py` are adapted from `games.py`,
`starter/load_smart_tagging.py`, and `starter/baseline.py` in the
[official More Than A Hack 2027 repository](https://github.com/metrica-sports/more-than-a-hack-2027).
The following licence applies to those upstream portions.

MIT License

Copyright (c) 2026-2027 Metrica Sports

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DFL/IDSSE training data and model parameters

Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert,
[DFL/IDSSE Bundesliga data, version 1](https://doi.org/10.6084/m9.figshare.28196177.v1).
The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The files in `possession/models/` contain parameters fitted from this dataset,
including normalization, duration and transition priors. They are transformed
model exports, not copies of source tracking or video. Release identifiers
and filenames were renamed for this package; learned numeric parameters were
preserved. Retain this attribution when redistributing these exports. The
MIT licence on original code does not replace rights or obligations associated
with the training data or derived parameters. Source training data is not
included, and its licence does not grant rights to broadcast footage.

## Other development sources

SkillCorner open data (MIT), SoccerTrack v2 (CC BY 4.0), and an optional
PretrainedSED prototype were used for validation or rejected development
experiments as described in `MNIST.md`. Their source data, code and checkpoints
are not included in this distribution. None of their trained model parameters
are part of the submitted ensemble. NumPy is installed separately and retains
its own bundled licence notices.

## Competition assets

The match tracking, Smart Tagging, homography, video and audio are separately
licensed for this competition. They are not distributed here and are not
covered by either MIT notice. Obtain the assets from the organisers after
registration and acceptance of their data-use terms. `MNIST.csv` contains the
algorithm-generated submission labels, not source tracking or media.

## PlotNeuralNet — MIT

The figure generator uses vendored Python and TikZ primitives from
[HarisIqbal88/PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet),
Copyright (c) 2018 HarisIqbal88. The complete MIT notice is retained in
[figures/vendor/plotneuralnet/LICENSE](figures/vendor/plotneuralnet/LICENSE).
Source provenance and file hashes are recorded alongside it. These tools are
used only to draw the architecture; they are not inference dependencies.
