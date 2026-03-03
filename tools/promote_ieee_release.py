from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_with_manifest(
    *,
    src: Path,
    dst: Path,
    manifest_rows: List[Dict[str, str]],
    label: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest_rows.append(
        {
            "label": label,
            "src": str(src.resolve()),
            "dst": str(dst.resolve()),
            "sha256": _sha256(dst),
        }
    )


def _copy_group(
    *,
    src_base: Path,
    dst_base: Path,
    exts: Tuple[str, ...],
    manifest_rows: List[Dict[str, str]],
    label: str,
    strict: bool,
) -> None:
    for ext in exts:
        src = src_base.with_suffix(ext)
        if not src.exists():
            if strict:
                raise FileNotFoundError(src)
            continue
        dst = dst_base.with_suffix(ext)
        _copy_with_manifest(src=src, dst=dst, manifest_rows=manifest_rows, label=label)


def _build_release_snapshot(step28_dir: Path, out_json: Path) -> None:
    summary_csv = step28_dir / "step28_ieee_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    df = pd.read_csv(summary_csv)
    mic = df[df["controller"].astype(str).str.upper() == "MIC"].copy()
    if mic.empty:
        raise ValueError("No MIC rows in step28_ieee_summary.csv")

    snapshot = {
        "step28_dir": str(step28_dir.resolve()),
        "rows_total": int(df.shape[0]),
        "mic_rows": int(mic.shape[0]),
        "avg_power_saving_pct_mean": float(mic["avg_power_saving_pct_mean"].mean()),
        "avg_power_saving_pct_min": float(mic["avg_power_saving_pct_min"].min()),
        "avg_eta_gain_pct_mean": float(mic["avg_eta_gain_pct_mean"].mean()),
        "avg_eta_gain_pct_min": float(mic["avg_eta_gain_pct_min"].min()),
        "err_failures_max": float(mic["err_failures_max"].max()),
        "start_stop_power_saving_pct_mean": float(mic["start_stop_power_saving_pct_mean"].mean()),
        "table_sha256_unique": sorted(set(mic["table_sha256"].astype(str).tolist())),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote frozen IEEE step28 package artifacts into paper/ieee_2026/{fig,data/release/<tag>}."
    )
    parser.add_argument("--step28-dir", required=True, help="Frozen step28 package directory.")
    parser.add_argument("--ieee-root", default="paper/ieee_2026", help="Target IEEE root directory.")
    parser.add_argument("--pgups-fig-dir", default="paper/pgups_2026/fig", help="Source directory with AIR56 and RU figures.")
    parser.add_argument("--tag", default="", help="Release tag. Default: step28 directory name.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing optional figure sources.")
    args = parser.parse_args()

    step28_dir = Path(args.step28_dir).expanduser().resolve()
    ieee_root = Path(args.ieee_root).expanduser().resolve()
    pgups_fig_dir = Path(args.pgups_fig_dir).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)

    tag = str(args.tag).strip() or step28_dir.name
    fig_dir = ieee_root / "fig"
    release_dir = ieee_root / "data" / "release" / tag
    tables_dir = release_dir / "tables"
    manifest_rows: List[Dict[str, str]] = []

    derived = step28_dir / "derived_ieee"
    passport = step28_dir / "passport"
    if not derived.exists():
        raise FileNotFoundError(derived)

    # 1) Core cross-motor IEEE figure from frozen package.
    _copy_group(
        src_base=derived / "fig_ieee_pi_foc_mic_power",
        dst_base=fig_dir / "fig2_pi_foc_mic_power",
        exts=(".png", ".pdf", ".svg"),
        manifest_rows=manifest_rows,
        label="fig2_pi_foc_mic_power",
        strict=True,
    )

    # 2) AIR56 detailed characteristics for fig3 (from validated PGUPS figure).
    _copy_group(
        src_base=pgups_fig_dir / "working_characteristics_air56_foc_mic",
        dst_base=fig_dir / "fig3_air56_working_characteristics",
        exts=(".png", ".pdf", ".svg"),
        manifest_rows=manifest_rows,
        label="fig3_air56_working_characteristics",
        strict=bool(args.strict),
    )

    # 3) Cross-motor robustness heatmap.
    _copy_group(
        src_base=pgups_fig_dir / "fig_multi_motor_scenario_heatmap_ru",
        dst_base=fig_dir / "fig4_cross_motor_robustness",
        exts=(".png", ".pdf", ".svg"),
        manifest_rows=manifest_rows,
        label="fig4_cross_motor_robustness",
        strict=bool(args.strict),
    )

    # 4) Training-to-performance curve.
    _copy_group(
        src_base=pgups_fig_dir / "fig_learning_vs_foc_ru",
        dst_base=fig_dir / "fig5_training_to_foc",
        exts=(".png", ".pdf", ".svg"),
        manifest_rows=manifest_rows,
        label="fig5_training_to_foc",
        strict=bool(args.strict),
    )

    # 5) Methodology block figure.
    _copy_group(
        src_base=pgups_fig_dir / "fig_algorithm_block_ru",
        dst_base=fig_dir / "fig1_mic_methodology",
        exts=(".png", ".pdf", ".svg"),
        manifest_rows=manifest_rows,
        label="fig1_mic_methodology",
        strict=False,
    )

    # Promote core tables/reports into release snapshot.
    table_files = [
        derived / "ieee_pi_foc_mic_stats.csv",
        derived / "ieee_pi_foc_mic_stats.md",
        derived / "motor_tuning_acceptance_summary.csv",
        derived / "motor_tuning_acceptance_summary.json",
        derived / "motor_air56_tuning_report.md",
        derived / "motor_al31_tuning_report.md",
        derived / "motor_ao2_tuning_report.md",
        step28_dir / "step28_ieee_summary.csv",
        step28_dir / "step28_ieee_summary.md",
        step28_dir / "package_manifest.json",
    ]
    for src in table_files:
        if not src.exists():
            if bool(args.strict):
                raise FileNotFoundError(src)
            continue
        dst = tables_dir / src.name
        _copy_with_manifest(src=src, dst=dst, manifest_rows=manifest_rows, label=f"table:{src.name}")

    for src in (
        passport / "passport_compare_3motors.csv",
        passport / "passport_compare_3motors.md",
        passport / "passport_compare_3motors.json",
    ):
        if not src.exists():
            if bool(args.strict):
                raise FileNotFoundError(src)
            continue
        dst = tables_dir / src.name
        _copy_with_manifest(src=src, dst=dst, manifest_rows=manifest_rows, label=f"table:{src.name}")

    # Build release snapshot with frozen key metrics.
    snapshot_path = release_dir / "release_snapshot.json"
    _build_release_snapshot(step28_dir, snapshot_path)
    manifest_rows.append(
        {
            "label": "release_snapshot",
            "src": str(snapshot_path.resolve()),
            "dst": str(snapshot_path.resolve()),
            "sha256": _sha256(snapshot_path),
        }
    )

    manifest = {
        "tag": tag,
        "step28_dir": str(step28_dir.resolve()),
        "ieee_root": str(ieee_root.resolve()),
        "promoted_count": len(manifest_rows),
        "files": manifest_rows,
    }
    manifest_path = release_dir / "promotion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Promoted release tag: {tag}")
    print(f"Figures dir: {fig_dir}")
    print(f"Release dir: {release_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
