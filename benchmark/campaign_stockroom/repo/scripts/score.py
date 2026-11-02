"""Milestone score, 0-100: each milestone's share of passing tests times its weight."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def passed_fraction(test_file: Path) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "r.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(test_file),
                        f"--junit-xml={report}", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, check=False)
        if not report.exists():
            return 0.0
        root = ET.parse(report).getroot()
        s = root if root.tag == "testsuite" else root.find("testsuite")
        total = int(s.get("tests", 0))
        bad = sum(int(s.get(k, 0)) for k in ("failures", "errors", "skipped"))
        return (total - bad) / total if total else 0.0


def main() -> None:
    weights = json.loads((ROOT / ".scoreboard" / "weights.json").read_text())
    total = 0.0
    for m, w in weights.items():
        files = sorted((ROOT / "tests").glob(f"test_{m}_*.py"))
        frac = sum(passed_fraction(f) for f in files) / len(files) if files else 0.0
        total += frac * w
        print(f"{m.upper()}: {frac * 100:5.1f}% of tests  -> {frac * w:5.1f} / {w}")
    print(f"SCORE: {round(total)}/100")


if __name__ == "__main__":
    main()
