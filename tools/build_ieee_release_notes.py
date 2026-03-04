from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_optional_json(path: Path) -> Tuple[Dict[str, object], bool]:
    if not path.exists():
        return {}, False
    return _read_json(path), True


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mic_rows(summary_df: pd.DataFrame) -> List[Dict[str, object]]:
    if summary_df.empty:
        return []
    mic_df = summary_df[summary_df["controller"].astype(str).str.upper() == "MIC"].copy()
    rows: List[Dict[str, object]] = []
    for _, row in mic_df.iterrows():
        rows.append(
            {
                "mode": str(row.get("mode", "")),
                "avg_power_saving_pct_mean": _to_float(row.get("avg_power_saving_pct_mean")),
                "avg_power_saving_pct_min": _to_float(row.get("avg_power_saving_pct_min")),
                "avg_eta_gain_pct_mean": _to_float(row.get("avg_eta_gain_pct_mean")),
                "avg_eta_gain_pct_min": _to_float(row.get("avg_eta_gain_pct_min")),
                "err_failures_max": _to_float(row.get("err_failures_max")),
                "start_stop_power_saving_pct_mean": _to_float(row.get("start_stop_power_saving_pct_mean")),
                "start_stop_power_saving_pct_min": _to_float(row.get("start_stop_power_saving_pct_min")),
            }
        )
    return rows


def _build_aggregate(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "avg_power_saving_pct_mean_avg": 0.0,
            "avg_eta_gain_pct_mean_avg": 0.0,
            "err_failures_max_worst": 0.0,
        }
    return {
        "avg_power_saving_pct_mean_avg": float(
            sum(_to_float(r.get("avg_power_saving_pct_mean")) for r in rows) / float(len(rows))
        ),
        "avg_eta_gain_pct_mean_avg": float(
            sum(_to_float(r.get("avg_eta_gain_pct_mean")) for r in rows) / float(len(rows))
        ),
        "err_failures_max_worst": float(max(_to_float(r.get("err_failures_max")) for r in rows)),
    }


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Frozen Release Notes")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- release_note_ready: `{payload.get('release_note_ready', False)}`")
    lines.append(f"- strict_ready: `{payload.get('strict_ready', False)}`")
    lines.append("")

    aggregate = dict(payload.get("aggregate", {}))
    lines.append("## MIC Aggregate")
    lines.append(f"- avg_power_saving_pct_mean_avg: `{float(aggregate.get('avg_power_saving_pct_mean_avg', 0.0)):+.4f}`")
    lines.append(f"- avg_eta_gain_pct_mean_avg: `{float(aggregate.get('avg_eta_gain_pct_mean_avg', 0.0)):+.4f}`")
    lines.append(f"- err_failures_max_worst: `{float(aggregate.get('err_failures_max_worst', 0.0)):.2f}`")
    lines.append("")

    lines.append("## MIC Per Mode")
    lines.append("| mode | power_mean_pct | power_min_pct | eta_mean_pct | eta_min_pct | err_failures_max | start_stop_mean_pct | start_stop_min_pct |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in list(payload.get("mic_rows", [])):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {mode} | {power_mean:+.4f} | {power_min:+.4f} | {eta_mean:+.4f} | {eta_min:+.4f} | {err:.2f} | {ss_mean:+.4f} | {ss_min:+.4f} |".format(
                mode=row.get("mode", ""),
                power_mean=_to_float(row.get("avg_power_saving_pct_mean")),
                power_min=_to_float(row.get("avg_power_saving_pct_min")),
                eta_mean=_to_float(row.get("avg_eta_gain_pct_mean")),
                eta_min=_to_float(row.get("avg_eta_gain_pct_min")),
                err=_to_float(row.get("err_failures_max")),
                ss_mean=_to_float(row.get("start_stop_power_saving_pct_mean")),
                ss_min=_to_float(row.get("start_stop_power_saving_pct_min")),
            )
        )
    lines.append("")

    artifacts = dict(payload.get("artifacts", {}))
    lines.append("## Artifact Status")
    for key in ("summary", "verify", "dossier", "release_manifest", "bundle_manifest", "regression_guard"):
        row = dict(artifacts.get(key, {}))
        lines.append(
            "- {name}: exists=`{exists}` ok=`{ok}` path=`{path}`".format(
                name=key,
                exists=bool(row.get("exists", False)),
                ok=bool(row.get("ok", False)),
                path=row.get("path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen IEEE release notes from step28 package artifacts.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/IEEE_RELEASE_NOTES.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/IEEE_RELEASE_NOTES.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)

    tag = str(args.tag).strip() or step28_dir.name
    summary_csv = step28_dir / "step28_ieee_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    verify_json = step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json"
    dossier_json = step28_dir / "IEEE_SUBMISSION_DOSSIER.json"
    release_manifest_json = step28_dir / "RELEASE_COMMIT_MANIFEST.json"
    regression_guard_json = step28_dir / "STEP28_REGRESSION_GUARD.json"
    bundle_manifest_json = ieee_root / "submission_bundle" / tag / "submission_bundle_manifest.json"

    summary_df = pd.read_csv(summary_csv)
    mic_rows = _mic_rows(summary_df)
    aggregate = _build_aggregate(mic_rows)

    verify_payload, verify_exists = _read_optional_json(verify_json)
    dossier_payload, dossier_exists = _read_optional_json(dossier_json)
    release_manifest_payload, release_manifest_exists = _read_optional_json(release_manifest_json)
    regression_guard_payload, regression_guard_exists = _read_optional_json(regression_guard_json)
    bundle_payload, bundle_exists = _read_optional_json(bundle_manifest_json)

    verify_ok = (not verify_exists) or _as_bool(verify_payload.get("verification_ok", False))
    dossier_ok = (not dossier_exists) or _as_bool(dict(dossier_payload.get("status", {})).get("dossier_ok", False))
    release_manifest_ok = (not release_manifest_exists) or _as_bool(release_manifest_payload.get("manifest_ok", False))
    regression_guard_ok = (not regression_guard_exists) or _as_bool(regression_guard_payload.get("ok", False))
    bundle_ok = (not bundle_exists) or _as_bool(bundle_payload.get("bundle_ok", False))

    release_note_ready = bool(
        len(mic_rows) > 0
        and verify_ok
        and dossier_ok
        and release_manifest_ok
        and bundle_ok
        and regression_guard_ok
    )
    strict_ready = bool(
        release_note_ready
        and verify_exists
        and dossier_exists
        and release_manifest_exists
        and bundle_exists
        and regression_guard_exists
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "mic_rows_count": int(len(mic_rows)),
        "mic_rows": mic_rows,
        "aggregate": aggregate,
        "release_note_ready": release_note_ready,
        "strict_ready": strict_ready,
        "artifacts": {
            "summary": {"path": str(summary_csv), "exists": True, "ok": len(mic_rows) > 0},
            "verify": {"path": str(verify_json), "exists": verify_exists, "ok": verify_ok},
            "dossier": {"path": str(dossier_json), "exists": dossier_exists, "ok": dossier_ok},
            "release_manifest": {
                "path": str(release_manifest_json),
                "exists": release_manifest_exists,
                "ok": release_manifest_ok,
            },
            "bundle_manifest": {"path": str(bundle_manifest_json), "exists": bundle_exists, "ok": bundle_ok},
            "regression_guard": {
                "path": str(regression_guard_json),
                "exists": regression_guard_exists,
                "ok": regression_guard_ok,
            },
        },
    }

    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "IEEE_RELEASE_NOTES.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "IEEE_RELEASE_NOTES.md")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"release_note_ready: {release_note_ready}")
    print(f"strict_ready: {strict_ready}")

    if bool(args.strict) and not strict_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
