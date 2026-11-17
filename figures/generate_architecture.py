"""Render the MNIST architecture as self-contained HTML with inline SVG.

The nine-node layout follows cathrynlavery/diagram-design. Numeric dimensions
are checked against the packaged manifest and all 21 model members.
"""

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / 'possession' / 'models'


def verify_architecture():
    manifest = json.loads((MODELS / 'solution.json').read_text())
    assert len(manifest['members']) == 21
    assert len(manifest['feature_names']) == 119
    for member in manifest['members']:
        m = json.loads((MODELS / member['file']).read_text())
        assert [len(layer['weight'][0]) for layer in m['player_layers']] == [20, 32]
        assert all(len(layer['weight']) == 32 for layer in m['player_layers'])
        assert len(m['attention']['qkv']['weight']) == 96
        assert len(m['attention']['qkv']['weight'][0]) == 32
        assert m['task_isolation']['paths'] == 2
        assert m['task_isolation']['receptive_field_anchors'] == 63
        assert m['task_isolation']['class_order'] == ['H', 'A', 'D']
        assert len(m['paths']) == 2
        for path in m['paths']:
            assert [layer['dilation'] for layer in path] == [1, 2, 4, 8, 16]
            assert all(len(layer['weight']) == 16 for layer in path)
            assert [len(layer['weight'][0]) for layer in path] == [366, 16, 16, 16, 16]
            assert all(len(layer['weight'][0][0]) == 3 for layer in path)
            assert all(layer['relu'] is True for layer in path)
        assert len(m['readout']['weight']) == 2
        assert all(len(row) == 32 for row in m['readout']['weight'])
    return manifest


def node(key, x, y, width, height, tag, title, lines, focal=False):
    stroke = '#0F4D92' if focal else '#84909B'
    fill = '#E8F0F8' if focal else '#ffffff'
    text = [f'<g id="architecture-{key}" data-node-id="{key}">',
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
            f'<text x="{x+16}" y="{y+24}" font-size="8" font-family="DejaVu Sans Mono, monospace" fill="#596775" letter-spacing="1">{html.escape(tag)}</text>',
            f'<text x="{x+16}" y="{y+52}" font-size="16" font-weight="600" fill="#272727">{html.escape(title)}</text>']
    for index, line in enumerate(lines):
        text.append(f'<text x="{x+16}" y="{y+76+index*20}" font-size="12" fill="#3F4C58">{html.escape(line)}</text>')
    return '\n'.join(text + ['</g>'])


def connector(d):
    return f'<path d="{d}" fill="none" stroke="#596775" stroke-width="1.8" marker-end="url(#architecture-arrow)"/>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', action='store_true', help='Also export a one-page PDF using Playwright Chromium.')
    args = parser.parse_args()
    verify_architecture()
    # Rounded, orthogonal routes. Separate ports keep forks and merges distinct.
    edges = [
        'M 744 208 H 808',
        'M 200 268 V 296 Q 200 304 208 304 H 496 Q 504 304 504 312 V 328',
        'M 1000 268 V 296 Q 1000 304 992 304 H 672 Q 664 304 664 312 V 328',
        'M 504 432 V 448 Q 504 456 496 456 H 368 Q 360 456 360 464 V 488',
        'M 664 432 V 448 Q 664 456 672 456 H 912 Q 920 456 920 464 V 488',
        'M 360 620 V 644 Q 360 652 368 652 H 496 Q 504 652 504 660 V 680',
        'M 920 620 V 644 Q 920 652 912 652 H 672 Q 664 652 664 660 V 680',
        'M 584 832 V 880',
        'M 744 928 H 808',
    ]
    blocks = [
        node('base', 40, 144, 320, 124, 'TRACKING / TAGS', 'Base representation',
             ['119 normalized numeric features', '+ 119 feature-validity bits', '238 values per sampled frame']),
        node('players', 424, 144, 320, 124, 'PLAYER SNAPSHOTS', 'Shared token encoder',
             ['10 numeric + 10 validity inputs', 'MLP: 20 → 32 → 32, with ReLU', 'Anonymous slots; no identity embedding']),
        node('attention', 808, 144, 384, 124, 'PER-SNAPSHOT ATTENTION', 'Team representation',
             ['2 attention heads × 16 dimensions', 'Masked mean + max pooling per team', '2 teams × (32 + 32) = 128 values']),
        node('concat', 424, 328, 320, 104, 'FEATURE FUSION', 'Concatenate: 238 + 128',
             ['366 inputs per frame, sampled at 5 Hz']),
        node('path-a', 184, 488, 352, 132, 'TEMPORAL PATH A', 'Dilated Conv1D → 16 channels',
             ['5 layers, kernel size 3, ReLU', 'Dilations: 1, 2, 4, 8, 16', '63-anchor receptive field']),
        node('path-b', 744, 488, 352, 132, 'TEMPORAL PATH B', 'Dilated Conv1D → 16 channels',
             ['5 layers, kernel size 3, ReLU', 'Dilations: 1, 2, 4, 8, 16', '63-anchor receptive field']),
        node('readout', 424, 680, 320, 152, 'PER-MEMBER OUTPUT', 'Joint H / A / D probabilities',
             ['Shared readout: (16 + 16) → 32 → 2', 'd = P(dead); h = P(home | live)',
              'P(H) = (1−d)h;  P(A) = (1−d)(1−h)', 'P(D) = d'], focal=True),
        node('ensemble', 424, 880, 320, 112, '7 HELD-MATCH FOLDS × 3 SEEDS', 'Average 21 probability outputs',
             ['Each member includes the blocks above', 'Average probabilities, not logits'], focal=True),
        node('decoder', 808, 864, 384, 128, 'SEQUENCE OUTPUT', 'Duration-aware decoder → CSV',
             ['Dynamic program + control-gated switches', 'Expand 5 Hz → 25 Hz; X outside blocks', '199,319 frame labels: H / A / D / X']),
    ]
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 1120" role="img" aria-labelledby="architecture-title architecture-desc">
<title id="architecture-title">MNIST attention and temporal architecture</title>
<desc id="architecture-desc">Tracking and pooled player attention feed two temporal convolution paths, whose shared probabilistic output is averaged over 21 models and decoded into possession spells.</desc>
<defs><marker id="architecture-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#596775"/></marker></defs>
<rect x="0" y="0" width="1280" height="1120" fill="#ffffff"/>
<text x="40" y="52" font-size="40" font-weight="600" fill="#272727">MNIST / Attention and temporal architecture</text>
<text x="40" y="88" font-size="16" fill="#596775">One member is shown in detail. All fitted parameters come from external DFL matches.</text>
'''
    svg += '\n'.join(connector(d) for d in edges) + '\n' + '\n'.join(blocks)
    svg += '''
<line x1="40" y1="1032" x2="1192" y2="1032" stroke="#CFCECE" stroke-width="1"/>
<rect x="40" y="1052" width="16" height="16" rx="2" fill="#ffffff" stroke="#84909B"/>
<text x="68" y="1066" font-size="12" fill="#596775">Representation / temporal processing</text>
<rect x="424" y="1052" width="16" height="16" rx="2" fill="#E8F0F8" stroke="#0F4D92"/>
<text x="452" y="1066" font-size="12" fill="#596775">Probability construction / aggregation</text>
<text x="40" y="1096" font-size="11" fill="#596775">Both output heads use all 32 temporal channels. No per-player recurrent state or target-label fitting.</text>
</svg>'''
    document = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MNIST attention and temporal architecture</title>
<style>
:root { --paper:#ffffff; --ink:#272727; --accent:#0F4D92; --soft:#596775; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font-family:"DejaVu Sans",Arial,sans-serif; }
.frame { max-width:1360px; min-width:0; margin:0 auto; padding:24px; }
.eyebrow { margin:0 0 8px; line-height:16px; font-size:12px; color:var(--soft); letter-spacing:0.12em; text-transform:uppercase; }
h1 { font-size:28px; line-height:36px; font-weight:600; margin:8px 0 24px; }
.diagram-container { width:100%; overflow-x:auto; border:1px solid #CFCECE; border-radius:6px; }
svg { width:100%; min-width:1280px; display:block; font-family:"DejaVu Sans",Arial,sans-serif; }
.note { max-width:84ch; font-size:14px; line-height:1.6; color:var(--soft); }
@media print { .frame { padding:0; } .eyebrow,h1,.note { display:none; } .diagram-container { overflow-x:visible; border:0; } svg { min-width:0; } }
</style></head><body><main class="frame">
<p class="eyebrow">MNIST · Technical architecture · Offline figure</p>
<h1>Attention, temporal context and sequence decoding</h1>
<div class="diagram-container">'''+svg+'''</div>
<p class="note">Architecture dimensions are validated against all 21 packaged model members. The diagram uses the project's existing white-and-blue palette and system fonts. Arrows show inference data flow; training and all 21 repeated members are summarized to keep nine readable blocks. The decoder's control evidence and duration priors are described in MNIST.md.</p>
</main></body></html>
'''
    (ROOT / 'architecture.html').write_text(document)
    print('Wrote architecture.html; checked all 21 model members.')
    if args.pdf:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto((ROOT / 'architecture.html').as_uri(), wait_until='networkidle')
                page.evaluate('document.fonts.ready')
                page.pdf(path=str(ROOT / 'architecture.pdf'), width='1280px', height='1120px',
                         print_background=True,
                         margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})
            finally:
                browser.close()
        print('Wrote architecture.pdf.')


if __name__ == '__main__':
    main()
