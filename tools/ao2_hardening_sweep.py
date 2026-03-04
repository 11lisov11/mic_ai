from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _to_float(row: pd.Series, key: str) -> float:
    return float(pd.to_numeric(row.get(key, 0.0), errors="coerce"))


def _score(row: pd.Series) -> float:
    # Lower is better. Hard penalize falling below AO2 v2 target margin.
    power = _to_float(row, "avg_power_saving_pct")
    eta = _to_float(row, "avg_eta_gain_pct")
    err = _to_float(row, "err_failures")
    start_stop = _to_float(row, "start_stop_power_saving_pct")
    peak = _to_float(row, "worst_current_peak_ratio")
    mean = _to_float(row, "worst_current_mean_ratio")

    penalty = 0.0
    penalty += max(0.0, 0.20 - power) * 40.0
    penalty += max(0.0, 0.00 - eta) * 20.0
    penalty += max(0.0, err - 2.0) * 12.0
    penalty += max(0.0, -0.50 - start_stop) * 8.0
    penalty += max(0.0, peak - 1.15) * 3.0
    penalty += max(0.0, mean - 1.03) * 2.0
    return float(penalty)


def _run(cmd: List[str], *, cwd: Path, dry_run: bool) -> None:
    print("[ao2-hardening] run:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def _rank_stage2(stage2_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(stage2_csv).copy()
    if df.empty:
        raise ValueError(f"Empty stage2 rank file: {stage2_csv}")
    df["v2_score"] = [_score(r) for _, r in df.iterrows()]
    df = df.sort_values(["v2_score", "score", "avg_power_saving_pct"], ascending=[True, True, False]).reset_index(drop=True)
    return df


def _acceptance(row: pd.Series) -> Dict[str, bool]:
    return {
        "power_margin_pass": _to_float(row, "avg_power_saving_pct") >= 0.20,
        "eta_pass": _to_float(row, "avg_eta_gain_pct") >= 0.0,
        "err_pass": _to_float(row, "err_failures") <= 2.0,
        "start_stop_pass": _to_float(row, "start_stop_power_saving_pct") >= -0.5,
    }


def _render_md(
    *,
    out_path: Path,
    stage1_csv: Path,
    stage2_csv: Path,
    shortlist: pd.DataFrame,
    selected_payload: Dict[str, object],
    acceptance: Dict[str, bool],
    summary: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# AO2 Hardening Sweep (v2)")
    lines.append("")
    lines.append(f"- generated_utc: `{summary['generated_utc']}`")
    lines.append(f"- stage1_csv: `{stage1_csv}`")
    lines.append(f"- stage2_csv: `{stage2_csv}`")
    lines.append(f"- shortlist_count: `{int(summary['shortlist_count'])}`")
    lines.append(f"- selected_tag: `{summary['selected_candidate']['tag']}`")
    lines.append("")
    lines.append("## Acceptance (selected)")
    lines.append(f"- power_margin_pass (>=0.20%): `{acceptance['power_margin_pass']}`")
    lines.append(f"- eta_pass (>=0): `{acceptance['eta_pass']}`")
    lines.append(f"- err_pass (<=2): `{acceptance['err_pass']}`")
    lines.append(f"- start_stop_pass (>=-0.5%): `{acceptance['start_stop_pass']}`")
    lines.append("")
    lines.append("## Shortlist (top-3)")
    lines.append("| rank | tag | avg_power_saving_pct | avg_eta_gain_pct | err_failures | start_stop_power_saving_pct | v2_score |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, (_, row) in enumerate(shortlist.iterrows(), start=1):
        lines.append(
            f"| {i} | {row.get('tag','')} | {float(row.get('avg_power_saving_pct',0.0)):+.3f} | "
            f"{float(row.get('avg_eta_gain_pct',0.0)):+.3f} | {float(row.get('err_failures',0.0)):.2f} | "
            f"{float(row.get('start_stop_power_saving_pct',0.0)):+.3f} | {float(row.get('v2_score',0.0)):.3f} |"
        )
    lines.append("")
    lines.append("## Selected Candidate")
    lines.append("```json")
    lines.append(json.dumps(selected_payload, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="AO2 v2 hardening sweep wrapper: stage1/2 tuning + shortlist + selected profile.")
    parser.add_argument("--out-dir", default="outputs/ao2_hardening_v2")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--stage1-trials", type=int, default=40)
    parser.add_argument("--stage2-topk", type=int, default=10)
    parser.add_argument("--search-seed", type=int, default=26027)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--sample-profile", default="global", choices=["global", "local_safe"])
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(str(args.out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tune_cmd = [
        sys.executable,
        "tools/tune_motor_step27.py",
        "--motor",
        "ao2",
        "--seeds",
        str(args.seeds),
        "--scenarios",
        str(args.scenarios),
        "--out-dir",
        str(out_dir),
        "--stage1-trials",
        str(int(args.stage1_trials)),
        "--stage2-topk",
        str(int(args.stage2_topk)),
        "--search-seed",
        str(int(args.search_seed)),
        "--window-frac",
        str(float(args.window_frac)),
        "--error-tol-rel",
        str(float(args.error_tol_rel)),
        "--error-tol-abs",
        str(float(args.error_tol_abs)),
        "--foc-feedback-mode",
        str(args.foc_feedback_mode),
        "--mic-feedback-mode",
        str(args.mic_feedback_mode),
        "--checkpoint-registry",
        str(args.checkpoint_registry),
        "--sample-profile",
        str(args.sample_profile),
        "--min-avg-power-saving-pct",
        "0.20",
        "--min-avg-eta-gain-pct",
        "0.0",
        "--max-err-failures",
        "2.0",
        "--min-start-stop-saving-pct",
        "-0.5",
        "--use-total-power",
    ]
    if bool(args.seed_perturbation):
        tune_cmd.extend(["--seed-perturbation", "--seed-perturb-level", str(float(args.seed_perturb_level))])
    if bool(args.foc_disable_lut):
        tune_cmd.append("--foc-disable-lut")
    else:
        tune_cmd.append("--allow-foc-lut")

    _run(tune_cmd, cwd=root, dry_run=bool(args.dry_run))
    if bool(args.dry_run):
        return

    stage1_csv = out_dir / "ao2_stage1_rank.csv"
    stage2_csv = out_dir / "ao2_stage2_rank.csv"
    if not stage1_csv.exists() or not stage2_csv.exists():
        raise FileNotFoundError("Expected ao2_stage1_rank.csv and ao2_stage2_rank.csv in output directory")

    ranked = _rank_stage2(stage2_csv)
    shortlist = ranked.head(3).copy()
    selected = shortlist.iloc[0].copy()
    acc = _acceptance(selected)

    shortlist_csv = out_dir / "ao2_shortlist_top3.csv"
    shortlist.to_csv(shortlist_csv, index=False)
    selected_json = out_dir / "ao2_selected_candidate_v2.json"
    selected_payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(selected).items()}
    selected_json.write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "motor": "ao2",
        "stage1_csv": str(stage1_csv),
        "stage2_csv": str(stage2_csv),
        "shortlist_csv": str(shortlist_csv),
        "selected_candidate_json": str(selected_json),
        "shortlist_count": int(len(shortlist)),
        "selected_candidate": selected_payload,
        "selected_acceptance": acc,
    }
    summary_json = out_dir / "ao2_hardening_summary_v2.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_md = out_dir / "ao2_hardening_summary_v2.md"
    _render_md(
        out_path=summary_md,
        stage1_csv=stage1_csv,
        stage2_csv=stage2_csv,
        shortlist=shortlist,
        selected_payload=selected_payload,
        acceptance=acc,
        summary=summary,
    )

    print(f"saved: {shortlist_csv}")
    print(f"saved: {selected_json}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")
    print(f"selected_tag: {selected_payload.get('tag', '')}")
    print(f"selected_power_saving_pct: {float(selected_payload.get('avg_power_saving_pct', 0.0)):+.3f}")


if __name__ == "__main__":
    main()
