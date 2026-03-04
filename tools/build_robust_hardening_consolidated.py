from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _to_float(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def _threshold_pass(row: pd.Series, thresholds: Dict[str, float]) -> bool:
    return bool(
        _to_float(row.get("baseline_power")) >= float(thresholds.get("baseline_min_power", 0.0))
        and _to_float(row.get("baseline_eta")) >= float(thresholds.get("baseline_min_eta", 0.0))
        and _to_float(row.get("baseline_err")) <= float(thresholds.get("baseline_max_err", 2.0))
        and _to_float(row.get("baseline_start_stop")) >= float(thresholds.get("baseline_min_start_stop", -0.5))
    )


def _resolve_runs(args_runs: str) -> List[Path]:
    if str(args_runs).strip():
        out: List[Path] = []
        for token in _parse_csv_list(args_runs):
            p = Path(token).expanduser().resolve()
            if p.exists():
                out.append(p)
        return out
    outputs = Path("outputs").resolve()
    if not outputs.exists():
        return []
    return sorted([p for p in outputs.iterdir() if p.is_dir() and p.name.startswith("robust_hardening_")])


def _load_run(run_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_path = run_dir / "robust_hardening_summary.json"
    if not summary_path.exists():
        return [], []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = list(payload.get("results", []))
    rank_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for item in results:
        motor = str(item.get("motor", "")).strip().lower()
        if not motor:
            continue
        selected = dict(item.get("selected_candidate", {}))
        selected_tag = str(selected.get("tag", ""))
        thresholds = dict(item.get("thresholds", {}))
        stage2_csv = run_dir / motor / f"{motor}_robust_stage2_rank.csv"
        if stage2_csv.exists():
            df = pd.read_csv(stage2_csv)
            for _, row in df.iterrows():
                row_dict = {k: row[k] for k in row.index}
                row_payload: Dict[str, object] = {
                    "run_tag": run_dir.name,
                    "run_dir": str(run_dir),
                    "motor": motor,
                    "selected_tag": selected_tag,
                    "selected_in_run": bool(str(row_dict.get("tag", "")) == selected_tag),
                    "selection_policy": str(item.get("selection_policy", "")),
                    "improved_vs_baseline": bool(item.get("improved_vs_baseline", False)),
                    "config_applied": bool(item.get("config_applied", False)),
                    **row_dict,
                }
                row_payload["baseline_guard_pass"] = _threshold_pass(pd.Series(row_dict), thresholds)
                rank_rows.append(row_payload)

        if selected:
            selected_rows.append(
                {
                    "run_tag": run_dir.name,
                    "run_dir": str(run_dir),
                    "motor": motor,
                    "selected_tag": selected_tag,
                    "selection_policy": str(item.get("selection_policy", "")),
                    "improved_vs_baseline": bool(item.get("improved_vs_baseline", False)),
                    "config_applied": bool(item.get("config_applied", False)),
                    "robust_score": _to_float(selected.get("robust_score")),
                    "baseline_power": _to_float(selected.get("baseline_power")),
                    "perturb_power_min": _to_float(selected.get("perturb_power_min")),
                    "perturb_eta_min": _to_float(selected.get("perturb_eta_min")),
                    "perturb_err_max": _to_float(selected.get("perturb_err_max")),
                    "perturb_start_stop_min": _to_float(selected.get("perturb_start_stop_min")),
                    "robust_pass": bool(selected.get("robust_pass", False)),
                }
            )

    return rank_rows, selected_rows


def _build_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    best_rows: List[Dict[str, object]] = []
    for motor, part in df.groupby("motor", dropna=False):
        part = part.copy()
        part["robust_score"] = pd.to_numeric(part["robust_score"], errors="coerce")
        part["baseline_guard_pass"] = part["baseline_guard_pass"].astype(bool)
        part["robust_pass"] = part.get("robust_pass", False).astype(bool)
        part = part.sort_values(
            ["baseline_guard_pass", "robust_pass", "robust_score", "selected_in_run"],
            ascending=[False, False, True, False],
        )
        best = part.iloc[0].to_dict()
        best_rows.append({"motor": motor, **best})
    return pd.DataFrame(best_rows)


def _render_md(
    out_path: Path,
    *,
    runs: List[Path],
    selected_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Robust Hardening Consolidated Report")
    lines.append("")
    lines.append(f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- runs_count: `{len(runs)}`")
    for run in runs:
        lines.append(f"- run: `{run}`")
    lines.append("")

    lines.append("## Selected candidates by run")
    if selected_df.empty:
        lines.append("- no selected rows")
    else:
        lines.append("| run_tag | motor | selected_tag | robust_score | baseline_power % | perturb_power_min % | perturb_eta_min % | robust_pass | policy |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
        for _, r in selected_df.sort_values(["motor", "run_tag"]).iterrows():
            lines.append(
                "| {run_tag} | {motor} | {selected_tag} | {robust_score:+.3f} | {baseline_power:+.3f} | {perturb_power_min:+.3f} | {perturb_eta_min:+.3f} | {robust_pass} | {selection_policy} |".format(
                    **{k: r[k] for k in r.index}
                )
            )
    lines.append("")

    lines.append("## Best candidate per motor (global ranking)")
    if best_df.empty:
        lines.append("- no best rows")
    else:
        lines.append("| motor | run_tag | tag | robust_score | baseline_guard_pass | robust_pass | baseline_power % | perturb_power_min % | perturb_eta_min % |")
        lines.append("|---|---|---|---:|---|---|---:|---:|---:|")
        for _, r in best_df.sort_values(["motor"]).iterrows():
            lines.append(
                "| {motor} | {run_tag} | {tag} | {robust_score:+.3f} | {baseline_guard_pass} | {robust_pass} | {baseline_power:+.3f} | {perturb_power_min:+.3f} | {perturb_eta_min:+.3f} |".format(
                    **{k: r[k] for k in r.index}
                )
            )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated robust hardening ranking across robust_hardening runs.")
    parser.add_argument("--runs", default="", help="Comma list of robust_hardening run directories. If empty, scan outputs/robust_hardening_*.")
    parser.add_argument("--out-dir", default="outputs/robust_hardening_consolidated_20260304")
    args = parser.parse_args()

    runs = _resolve_runs(str(args.runs))
    if not runs:
        raise FileNotFoundError("No robust_hardening run directories found")

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    for run in runs:
        r_rows, s_rows = _load_run(run)
        rank_rows.extend(r_rows)
        selected_rows.extend(s_rows)

    rank_df = pd.DataFrame(rank_rows)
    selected_df = pd.DataFrame(selected_rows)
    best_df = _build_best(rank_df) if not rank_df.empty else pd.DataFrame()

    rank_csv = out_dir / "robust_hardening_consolidated_rank.csv"
    selected_csv = out_dir / "robust_hardening_consolidated_selected.csv"
    best_csv = out_dir / "robust_hardening_consolidated_best.csv"
    report_json = out_dir / "robust_hardening_consolidated.json"
    report_md = out_dir / "robust_hardening_consolidated.md"

    rank_df.to_csv(rank_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [str(p) for p in runs],
        "rank_csv": str(rank_csv),
        "selected_csv": str(selected_csv),
        "best_csv": str(best_csv),
        "rows_rank": int(len(rank_df)),
        "rows_selected": int(len(selected_df)),
        "rows_best": int(len(best_df)),
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_md(report_md, runs=runs, selected_df=selected_df, best_df=best_df)

    print(f"saved: {rank_csv}")
    print(f"saved: {selected_csv}")
    print(f"saved: {best_csv}")
    print(f"saved: {report_json}")
    print(f"saved: {report_md}")


if __name__ == "__main__":
    main()
