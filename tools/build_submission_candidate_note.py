from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _check_ready(checklist_text: str) -> bool:
    text = str(checklist_text).lower()
    return "ready_for_submission: `true`" in text


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _render_motor_rows(step28_dir: Path, summary_csv: Path) -> List[str]:
    derived_summary = step28_dir / "derived_ieee" / "motor_tuning_acceptance_summary.json"
    if derived_summary.exists():
        payload = _read_json(derived_summary)
        raw_rows = payload.get("rows", [])
        if isinstance(raw_rows, list) and raw_rows:
            out: List[str] = []
            for row in sorted((r for r in raw_rows if isinstance(r, dict)), key=lambda x: str(x.get("motor", ""))):
                out.append(
                    "- motor={motor}, acceptance_pass={acc}, power_mean={power:+.3f}%, power_min={pmin:+.3f}%, eta_mean={eta:+.3f}%, eta_min={emin:+.3f}%, err_max={err:.2f}".format(
                        motor=str(row.get("motor", "")),
                        acc=bool(row.get("acceptance_pass", False)),
                        power=_to_float(row.get("avg_power_saving_pct_mean")),
                        pmin=_to_float(row.get("avg_power_saving_pct_min")),
                        eta=_to_float(row.get("avg_eta_gain_pct_mean")),
                        emin=_to_float(row.get("avg_eta_gain_pct_min")),
                        err=_to_float(row.get("err_failures_max")),
                    )
                )
            if out:
                return out

    # Fallback for legacy step28 summary format without per-motor rows.
    df = pd.read_csv(summary_csv)
    mic = df[df["controller"].astype(str).str.upper() == "MIC"].copy()
    if mic.empty:
        return ["- MIC rows are missing in step28 summary."]
    sort_cols = [c for c in ("mode", "motor") if c in mic.columns]
    mic_sorted = mic.sort_values(sort_cols) if sort_cols else mic
    rows: List[str] = []
    for _, row in mic_sorted.iterrows():
        rows.append(
            "- mode={mode}, motor={motor}, power_mean={power:+.3f}%, power_min={pmin:+.3f}%, eta_mean={eta:+.3f}%, eta_min={emin:+.3f}%, err_max={err:.2f}".format(
                mode=str(row.get("mode", "")),
                motor=str(row.get("motor", "ALL")),
                power=_to_float(row.get("avg_power_saving_pct_mean")),
                pmin=_to_float(row.get("avg_power_saving_pct_min")),
                eta=_to_float(row.get("avg_eta_gain_pct_mean")),
                emin=_to_float(row.get("avg_eta_gain_pct_min")),
                err=_to_float(row.get("err_failures_max")),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build submission-candidate note from frozen IEEE artifacts.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Candidate tag. Default: step28 directory name.")
    parser.add_argument("--checklist-md", default="", help="Path to FINAL_CHECKLIST_AUTO.md")
    parser.add_argument("--out-md", default="", help="Path to output markdown note.")
    parser.add_argument("--out-json", default="", help="Path to output json note.")
    args = parser.parse_args()

    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)

    tag = str(args.tag).strip() or step28_dir.name
    lock_path = step28_dir / "submission_candidate_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    summary_csv = step28_dir / "step28_ieee_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    checklist_md = (
        Path(str(args.checklist_md)).expanduser().resolve()
        if str(args.checklist_md).strip()
        else (ieee_root / "FINAL_CHECKLIST_AUTO.md")
    )
    if not checklist_md.exists():
        raise FileNotFoundError(checklist_md)

    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (ieee_root / "SUBMISSION_CANDIDATE.md")
    )
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (ieee_root / "SUBMISSION_CANDIDATE.json")
    )

    lock = _read_json(lock_path)
    checklist_text = checklist_md.read_text(encoding="utf-8")
    ready = _check_ready(checklist_text)

    aggregate_sha = str(lock.get("aggregate_sha256", ""))
    hashed_files_count = int(lock.get("hashed_files_count", 0))
    lock_ok = bool(lock.get("lock_ok", False))
    required_missing = lock.get("required_files_missing", [])
    if not isinstance(required_missing, list):
        required_missing = []

    motor_rows = _render_motor_rows(step28_dir, summary_csv)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "checklist_md": str(checklist_md),
        "ready_for_submission": bool(ready),
        "lock_ok": bool(lock_ok),
        "lock_aggregate_sha256": aggregate_sha,
        "lock_hashed_files_count": hashed_files_count,
        "lock_required_missing_count": len(required_missing),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Submission Candidate")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- candidate_tag: `{tag}`")
    lines.append(f"- step28_dir: `{step28_dir}`")
    lines.append(f"- ready_for_submission: `{ready}`")
    lines.append(f"- lock_ok: `{lock_ok}`")
    lines.append(f"- lock_hashed_files_count: `{hashed_files_count}`")
    lines.append(f"- lock_aggregate_sha256: `{aggregate_sha}`")
    lines.append(f"- lock_required_missing_count: `{len(required_missing)}`")
    lines.append("")
    lines.append("## MIC summary rows")
    lines.extend(motor_rows)
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {out_md}")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
