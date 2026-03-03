from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_check_ieee_manuscript_template_smoke(tmp_path: Path) -> None:
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        (
            "## Abstract\n"
            "This abstract has enough words to pass the basic range check in the smoke test and "
            "contains only synthetic text for deterministic CI execution.\n"
            "## I. Introduction\n"
            "See Fig. 1 and Fig. 2.\n"
            "## II. Method\n"
            "Method.\n"
            "## III. Experimental Setup\n"
            "Setup.\n"
            "## IV. Results\n"
            "Results with Tab. 1 and Table 2.\n"
            "## V. Theory Validation\n"
            "Validation.\n"
            "## VI. Discussion\n"
            "Discussion.\n"
            "## VII. Conclusion\n"
            "Conclusion.\n"
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "MANUSCRIPT_TEMPLATE_REPORT.json"
    out_md = tmp_path / "MANUSCRIPT_TEMPLATE_REPORT.md"
    cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_template.py",
        "--manuscript",
        str(manuscript),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--strict",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(payload.get("ok", False)) is True
    assert int(payload.get("errors_count", 0)) == 0

