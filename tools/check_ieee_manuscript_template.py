from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


REQUIRED_HEADINGS = [
    "Abstract",
    "I. Introduction",
    "II. Method",
    "III. Experimental Setup",
    "IV. Results",
    "V. Theory Validation",
    "VI. Discussion",
    "VII. Conclusion",
]


def _extract_headings(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            out.append(line[3:].strip())
    return out


def _extract_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"^##\s+.+$", text[start:], re.MULTILINE)
    end = (start + nxt.start()) if nxt else len(text)
    return text[start:end].strip()


def _count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def build_report(manuscript: Path) -> Dict[str, object]:
    text = manuscript.read_text(encoding="utf-8")
    headings = _extract_headings(text)

    missing_headings = [h for h in REQUIRED_HEADINGS if h not in headings]
    order_ok = True
    if not missing_headings:
        indices = [headings.index(h) for h in REQUIRED_HEADINGS]
        order_ok = all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    abstract = _extract_section(text, "Abstract")
    abstract_words = _count_words(abstract)

    fig_refs = sorted(
        {
            m.group(0)
            for m in re.finditer(r"\bFig\.\s*\d+\b", text, flags=re.IGNORECASE)
        }
    )
    table_refs = sorted(
        {
            m.group(0)
            for m in re.finditer(r"\b(Tab\.\s*\d+|Table\s+\d+)\b", text, flags=re.IGNORECASE)
        }
    )

    warnings: List[str] = []
    errors: List[str] = []

    if missing_headings:
        errors.append(f"missing required headings: {', '.join(missing_headings)}")
    if not order_ok:
        errors.append("required headings are out of order")
    if abstract_words < 60 or abstract_words > 260:
        warnings.append(f"abstract word count is outside typical IEEE range (60..260): {abstract_words}")
    if len(fig_refs) < 3:
        warnings.append(f"low number of figure references detected: {len(fig_refs)}")
    if len(table_refs) < 2:
        warnings.append(f"low number of table references detected: {len(table_refs)}")
    if "Index Terms" not in headings:
        warnings.append("missing 'Index Terms' heading")
    if "References" not in headings:
        warnings.append("missing 'References' heading")

    ok = len(errors) == 0
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": str(manuscript),
        "required_headings": REQUIRED_HEADINGS,
        "headings_found": headings,
        "missing_headings": missing_headings,
        "required_headings_order_ok": bool(order_ok),
        "abstract_word_count": int(abstract_words),
        "figure_references_detected": fig_refs,
        "table_references_detected": table_refs,
        "warnings_count": len(warnings),
        "errors_count": len(errors),
        "warnings": warnings,
        "errors": errors,
        "ok": bool(ok),
    }


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Manuscript Template Report")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- manuscript: `{payload.get('manuscript', '')}`")
    lines.append(f"- ok: `{payload.get('ok', False)}`")
    lines.append(f"- required_headings_order_ok: `{payload.get('required_headings_order_ok', False)}`")
    lines.append(f"- abstract_word_count: `{payload.get('abstract_word_count', 0)}`")
    lines.append(f"- figure_references_detected: `{len(list(payload.get('figure_references_detected', [])))}`")
    lines.append(f"- table_references_detected: `{len(list(payload.get('table_references_detected', [])))}`")
    lines.append("")
    missing = list(payload.get("missing_headings", []))
    if missing:
        lines.append("## Missing Required Headings")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
    errors = list(payload.get("errors", []))
    if errors:
        lines.append("## Errors")
        for item in errors:
            lines.append(f"- {item}")
        lines.append("")
    warnings = list(payload.get("warnings", []))
    if warnings:
        lines.append("## Warnings")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manuscript section structure against IEEE-ready template.")
    parser.add_argument("--manuscript", required=True)
    parser.add_argument("--out-json", default="", help="Default: <manuscript_dir>/MANUSCRIPT_TEMPLATE_REPORT.json")
    parser.add_argument("--out-md", default="", help="Default: <manuscript_dir>/MANUSCRIPT_TEMPLATE_REPORT.md")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when required template checks fail.")
    args = parser.parse_args()

    manuscript = Path(str(args.manuscript)).expanduser().resolve()
    if not manuscript.exists():
        raise FileNotFoundError(manuscript)

    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (manuscript.parent / "MANUSCRIPT_TEMPLATE_REPORT.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (manuscript.parent / "MANUSCRIPT_TEMPLATE_REPORT.md")
    )

    payload = build_report(manuscript)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"ok: {bool(payload.get('ok', False))}")

    if bool(args.strict) and not bool(payload.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
