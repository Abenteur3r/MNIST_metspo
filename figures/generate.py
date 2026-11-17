"""Generate manuscript figures from frozen external aggregates; never read match data.

Styling follows scientific-figure-making in ChenLiu-1996/figures4papers.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
PALETTE = {
    'blue_main': '#0F4D92', 'blue_secondary': '#3775BA',
    'green_1': '#DDF3DE', 'green_3': '#8BCF8B',
    'red_1': '#F6CFCB', 'red_strong': '#B64342',
    'neutral': '#CFCECE', 'teal': '#42949E', 'ink': '#272727',
}


def apply_publication_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 14, 'axes.titleweight': 'bold',
        'axes.labelsize': 12, 'axes.linewidth': 1.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'text.color': PALETTE['ink'], 'axes.labelcolor': PALETTE['ink'],
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.hashsalt': 'mnist-figures-v1',
    })


def finalize_figure(fig, basename):
    """Export a 300 dpi preview plus vector files with editable text."""
    fig.canvas.draw()
    for extension in ('png', 'pdf', 'svg'):
        metadata = {'Creator': 'MNIST figure generator'}
        if extension == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        if extension == 'svg':
            metadata['Date'] = None
        fig.savefig(ROOT / f'{basename}.{extension}', dpi=300,
                    bbox_inches='tight', pad_inches=0.12, metadata=metadata)
        if extension == 'svg':
            path = ROOT / f'{basename}.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def validate_evidence(data):
    """Check the plotted units, denominators and arithmetic before exporting."""
    for panel in ('dfl', 'skillcorner'):
        d = data[panel]
        assert 0 <= d['reference_spell_f1'] <= 1
        assert 0 <= d['solution_spell_f1'] <= 1
        assert np.isclose(d['solution_spell_f1'] - d['reference_spell_f1'],
                          d['paired_change'], atol=1e-10)
        assert d['paired_ci95'][0] <= d['paired_change'] <= d['paired_ci95'][1]
    assert len(data['dfl']['match_gains']) == data['dfl']['matches']
    assert np.isclose(np.mean(list(data['dfl']['match_gains'].values())),
                      data['dfl']['paired_change'])
    bins = data['dfl']['duration_bins']
    assert sum(b['matched'] for b in bins.values()) == data['dfl']['matched_possessions_iou50']
    for b in bins.values():
        assert 0 <= b['matched'] <= b['truth']
        assert np.isclose(b['matched'] / b['truth'], b['recall_iou50'])
    a = data['architecture']
    assert a['raw_features'] + a['validity_bits'] + a['pooled_attention_values'] == a['temporal_inputs']
    assert a['folds'] * a['seeds'] == a['ensemble_members']


def pipeline(data):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set(xlim=(0, 14), ylim=(0, 4.8))
    ax.axis('off')
    ax.text(0, 4.5, 'MNIST  /  From tracking to possession spells',
            fontsize=20, weight='bold')
    ax.text(0, 4.02, 'Offline CPU inference  ·  no video processing  ·  no target-label fitting',
            fontsize=12, color='#555555')
    stages = [
        (0.05, '01  PROVIDED INPUTS', 'Tactical tracking\nplayers + ball coordinates\n\nTeam / goalkeeper metadata\nAutomatic tactical tags', '#F1F2F3'),
        (3.57, '02  REPRESENTATION', '119 tracking / tag features\n+ 119 validity bits\n\nPlayer snapshot attention\n+ 128 pooled values', '#E8F0F8'),
        (7.09, '03  TEMPORAL ENSEMBLE', '366 inputs at 5 Hz\nTwo temporal convolution paths\n\n21 frozen models\nMean H / A / D probabilities', '#E8F0F8'),
        (10.61, '04  DECODE + EXPORT', 'Duration-aware dynamic program\nControl-gated team switches\n\nExpand to 25 Hz\nCSV: 199,319 H / A / D / X rows', '#EDF5F0'),
    ]
    for x, title, body, fill in stages:
        box = FancyBboxPatch((x, 1.25), 3.22, 2.27,
                             boxstyle='round,pad=0.015,rounding_size=0.1',
                             facecolor=fill, edgecolor='#72808D', linewidth=1.3)
        ax.add_patch(box)
        ax.text(x + 0.15, 3.23, title, fontsize=10.6, color=PALETTE['blue_main'], weight='bold')
        ax.text(x + 0.15, 2.86, body, fontsize=10.1, va='top', linespacing=1.6)
    for x in (3.3, 6.82, 10.34):
        ax.add_patch(FancyArrowPatch((x, 2.35), (x + 0.22, 2.35),
                     arrowstyle='-|>', mutation_scale=17, linewidth=1.8, color=PALETTE['blue_main']))
    band = FancyBboxPatch((0.05, 0.2), 13.78, 0.64,
                         boxstyle='round,pad=0.01,rounding_size=0.08',
                         facecolor=PALETTE['green_1'], edgecolor='none')
    ax.add_patch(band)
    ax.text(6.94, 0.52, 'EXTERNAL TRAINING ONLY   7 DFL matches  ×  3 seeds  =  21 leave-one-match-out members',
            ha='center', va='center', fontsize=11.2, weight='bold')
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.03, top=0.99)
    finalize_figure(fig, 'pipeline')


def validation(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    x = np.arange(2)
    ref = [data[p]['reference_spell_f1'] for p in ('dfl', 'skillcorner')]
    sol = [data[p]['solution_spell_f1'] for p in ('dfl', 'skillcorner')]
    for shift, values, label, color, hatch in (
        (-0.18, ref, 'Previous attention ensemble', PALETTE['neutral'], '//'),
        (0.18, sol, 'MNIST solution', PALETTE['blue_main'], None),
    ):
        bars = ax.bar(x + shift, values, width=0.32, label=label,
                      color=color, edgecolor=PALETTE['ink'], linewidth=1.1, hatch=hatch)
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in values], padding=5, fontsize=11)
    ax.set(xticks=x, xticklabels=['DFL\n7 matches', 'SkillCorner\n20 matches'],
           ylabel='Possession spellF1', ylim=(0, 1.0), yticks=np.arange(0, 1.01, 0.2))
    ax.set_title('a  External development scores', loc='left', pad=22)
    ax.legend(loc='upper right', fontsize=9.3)
    ax.yaxis.grid(True, color='#EEEEEE', linewidth=0.8)
    ax.set_axisbelow(True)

    ax = axes[1]
    for y, panel in ((1, 'dfl'), (0, 'skillcorner')):
        d = data[panel]; mean = d['paired_change'] * 100
        lo, hi = np.asarray(d['paired_ci95']) * 100
        color = PALETTE['blue_main'] if mean > 0 else PALETTE['red_strong']
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt='o',
                    markersize=9, color=color, elinewidth=2.5, capsize=6, capthick=1.7)
        ax.text(mean, y + 0.22, f'{mean:+.3f} pp', ha='center', fontsize=11, color=color, weight='bold')
    ax.axvline(0, color='#777777', linewidth=1.1, linestyle='--')
    ax.set(yticks=[1, 0], yticklabels=['DFL', 'SkillCorner'], ylim=(-0.6, 1.6),
           xlim=(-1.2, 4.3), xlabel='Change in spellF1 (percentage points)')
    ax.set_title('b  Paired change and 95% interval', loc='left', pad=22)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.text(0.07, 0.04, 'Whole-match bootstrap: 20,000 draws. Previously examined development panels; no hidden-match score.',
             fontsize=10, color='#555555')
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.84, wspace=0.55)
    finalize_figure(fig, 'external_validation')


def diagnostics(data):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), gridspec_kw={'width_ratios': [1, 1.1]})
    ax = axes[0]
    gains = data['dfl']['match_gains']
    names = list(gains); values = np.asarray(list(gains.values())) * 100
    y = np.arange(len(names))
    ax.hlines(y, 0, values, color=PALETTE['blue_secondary'], linewidth=2)
    ax.scatter(values, y, s=58, color=PALETTE['blue_main'], zorder=3)
    for row, value in zip(y, values):
        ax.text(value + 0.12, row, f'+{value:.2f}', va='center', fontsize=10)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 6.05),
           xlabel='Change in spellF1 (percentage points)')
    ax.invert_yaxis()
    ax.set_title('a  DFL improvement: 7 / 7 match wins', loc='left', pad=20)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax = axes[1]
    bins = data['dfl']['duration_bins']
    rows = [bins[key] for key in ('le1s', '1to3s', '3to10s', 'gt10s')]
    values = [r['recall_iou50'] for r in rows]
    bars = ax.bar(np.arange(4), values, width=0.64,
                  color=[PALETTE['red_1'], PALETTE['red_strong'], PALETTE['teal'], PALETTE['blue_main']],
                  edgecolor=PALETTE['ink'], linewidth=1.1)
    ax.bar_label(bars, labels=[f'{v:.1%}' for v in values], padding=5, fontsize=11)
    ax.set(xticks=np.arange(4), xticklabels=['≤ 1 s', '1–3 s', '3–10 s', '> 10 s'],
           ylim=(0, 1.06), ylabel='Possession recall at temporal IoU 0.5', xlabel='Ground-truth possession duration')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title('b  Short possessions remain difficult', loc='left', pad=20)
    fig.text(0.08, 0.065, 'Panel a: mean gain within each DFL match. Panel b: pooled counts across degradation-grid units;', fontsize=10, color='#555555')
    fig.text(0.08, 0.025, 'truth spells repeat across units. These are diagnostic development results, not independent test evidence.', fontsize=10, color='#555555')
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.84, wspace=0.5)
    finalize_figure(fig, 'diagnostics')


def main():
    data = json.loads((ROOT / 'evidence.json').read_text())
    validate_evidence(data)
    apply_publication_style()
    for build in (pipeline, validation, diagnostics):
        build(data)
    print('Exported 3 figures in PNG (300 dpi), PDF and SVG. All evidence checks passed.')


if __name__ == '__main__':
    main()
