"""Report package coverage and enforce CRAP < 10 from coverage.py JSON."""

import json
from pathlib import Path
import subprocess
import sys


def crap(complexity, coverage):
    return complexity ** 2 * (1 - coverage) ** 3 + complexity


def line_rate(record, first, last):
    executed = sum(first <= line <= last for line in record["executed_lines"])
    missing = sum(first <= line <= last for line in record["missing_lines"])
    if executed + missing == 0:
        raise ValueError("Function has no measured statements")
    return executed / (executed + missing)


def main():
    # Small self-checks keep formula and range-boundary mistakes visible.
    assert crap(2, 0.5) == 2.5
    assert crap(3, 1) == 3 and crap(3, 0) == 12
    assert line_rate({"executed_lines": [1, 2, 4], "missing_lines": [3]}, 2, 3) == 0.5
    root = Path(__file__).resolve().parents[1]
    report = json.loads(Path(sys.argv[1]).read_text())
    files = {Path(path).resolve(): record for path, record in report["files"].items()}
    blocks = json.loads(subprocess.run(
        [sys.executable, "-m", "radon", "cc", "possession", "-j"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout)
    scores = []
    for path, functions in blocks.items():
        for function in functions:
            if function["type"] not in ("function", "method"):
                continue
            record = files[(root / path).resolve()]
            rate = line_rate(record, function["lineno"], function["endline"])
            score = crap(function["complexity"], rate)
            scores.append(score)
            if score >= 10:
                print(f'{path}:{function["name"]}: CRAP={score:.6f}')
    if not scores:
        raise ValueError("No functions measured")
    totals = report["totals"]
    lines, statements = totals["covered_lines"], totals["num_statements"]
    branches, possible = totals["covered_branches"], totals["num_branches"]
    print(f"Lines: {lines}/{statements} ({100 * lines / statements:.2f}%)")
    print(f"Branches: {branches}/{possible} ({100 * branches / possible:.2f}%)")
    passed = sum(score < 10 for score in scores)
    print(f"CRAP < 10: {passed}/{len(scores)} functions; max={max(scores):.6f}")
    return int(passed != len(scores))


if __name__ == "__main__":
    raise SystemExit(main())
