from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_utils import json_dump as _json_dump_shared
from tools.common_utils import parse_csv_list as _parse_csv_list_shared
from tools.common_utils import parse_int_list as _parse_int_list_shared
from tools.common_utils import std as _std_shared
from tools.common_utils import write_csv as _write_csv_shared
from tools.step27_pipeline import (  # noqa: E402
    MOTOR_REGISTRY,
    SeedPerturbationSettings,
    _aggregate_rows,
    _id_ref_eval_params,
    _load_agent,
    _load_env_and_agent,
    _sensorless_params,
    _simulate_rows,
    _supervisor_from_env,
)


METRIC_KEYS: Tuple[str, ...] = (
    "avg_power_saving_pct",
    "avg_eta_gain_pct",
    "err_failures",
    "start_stop_power_saving_pct",
    "worst_current_peak_ratio",
    "worst_current_mean_ratio",
    "avg_controller_speed_err",
)


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_list(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _json_dump(path: Path, payload: object) -> None:
    _json_dump_shared(path, payload)


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _stage_rank(stage: str) -> float:
    s = str(stage or "").strip().lower()
    if s == "fine_tune":
        return 300.0
    if s == "separate":
        return 200.0
    if s.startswith("joint_cycle_"):
        tail = s.replace("joint_cycle_", "", 1).strip()
        try:
            n = int(tail)
        except Exception:
            n = 0
        return 100.0 + float(n)
    return 0.0


def _load_manifest_checkpoint_index(path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        return {}
    out: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in runs:
        if not isinstance(row, dict):
            continue
        motor = str(row.get("motor", "")).strip().lower()
        if not motor:
            continue
        try:
            seed = int(row.get("seed", 0))
        except Exception:
            continue
        ckpt = str(row.get("best_checkpoint", "")).strip()
        if not ckpt:
            continue
        rec = {
            "motor": motor,
            "seed": seed,
            "stage": str(row.get("stage", "")),
            "best_checkpoint": ckpt,
            "stage_rank": _stage_rank(str(row.get("stage", ""))),
        }
        key = (motor, seed)
        prev = out.get(key)
        if prev is None or float(rec["stage_rank"]) >= float(prev["stage_rank"]):
            out[key] = rec
    return out


def _resolve_checkpoint_for_source(
    *,
    source_motor: str,
    seed: int,
    manifest_index: Dict[Tuple[str, int], Dict[str, object]],
    checkpoint_registry_path: str,
    foc_disable_lut: bool,
) -> Tuple[Path, str]:
    key = (str(source_motor).strip().lower(), int(seed))
    row = manifest_index.get(key)
    if row is not None:
        ckpt = Path(str(row["best_checkpoint"])).expanduser().resolve()
        if ckpt.exists():
            return ckpt, f"manifest:{Path(str(row['best_checkpoint'])).resolve()}"
    source_spec = MOTOR_REGISTRY[str(source_motor)]
    _env, _agent, ckpt = _load_env_and_agent(
        source_spec.config_path,
        foc_disable_lut=bool(foc_disable_lut),
        require_agent=True,
        motor_key=str(source_motor),
        checkpoint_registry_path=str(checkpoint_registry_path),
    )
    if ckpt is None:
        raise FileNotFoundError(f"Cannot resolve checkpoint for source motor={source_motor}")
    return ckpt.resolve(), "registry_or_env"


def _group_stats(
    rows: Sequence[Dict[str, object]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets.setdefault(key, []).append(dict(row))

    out: List[Dict[str, object]] = []
    for key, items in buckets.items():
        rec: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        for mk in metric_keys:
            vals: List[float] = []
            for r in items:
                try:
                    vals.append(float(r.get(mk, 0.0)))
                except Exception:
                    pass
            if not vals:
                continue
            rec[f"{mk}_mean"] = float(sum(vals) / max(len(vals), 1))
            rec[f"{mk}_std"] = float(_std_shared(vals))
            rec[f"{mk}_min"] = float(min(vals))
            rec[f"{mk}_max"] = float(max(vals))
        rec["n"] = int(len(items))
        out.append(rec)
    out.sort(key=lambda r: tuple(str(r.get(k, "")) for k in group_keys))
    return out


def _build_report(
    *,
    args: argparse.Namespace,
    per_seed: Sequence[Dict[str, object]],
    summary: Sequence[Dict[str, object]],
    gaps: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# Cross-Motor Generalization Report")
    lines.append("")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- dry_run: `{bool(args.dry_run)}`")
    lines.append(f"- sources: `{args.source_motors}`")
    lines.append(f"- targets: `{args.target_motors}`")
    lines.append(f"- seeds: `{args.seeds}`")
    lines.append(f"- scenarios: `{args.scenarios}`")
    lines.append("")
    lines.append(f"- per_seed_rows: `{len(per_seed)}`")
    lines.append(f"- summary_rows: `{len(summary)}`")
    lines.append(f"- gap_rows: `{len(gaps)}`")
    lines.append("")
    if gaps:
        lines.append("## Gap vs Native (mean)")
        lines.append("")
        lines.append("| Source | Target | ΔPower, % | Δη, % | ΔErr |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in gaps:
            lines.append(
                "| {s} | {t} | {dp:+.3f} | {de:+.3f} | {dr:+.3f} |".format(
                    s=r.get("source_motor", ""),
                    t=r.get("target_motor", ""),
                    dp=float(r.get("delta_power_vs_native_mean", math.nan)),
                    de=float(r.get("delta_eta_vs_native_mean", math.nan)),
                    dr=float(r.get("delta_err_vs_native_mean", math.nan)),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-motor generalization evaluation: source-motor checkpoint on held-out target motor domains."
    )
    parser.add_argument("--mode", default="heldout", choices=["heldout", "all_pairs"])
    parser.add_argument("--source-motors", default="air56,al31")
    parser.add_argument("--target-motors", default="ao2")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--no-use-total-power", dest="use_total_power", action="store_false")
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--out-dir", default="outputs/cross_motor_generalization")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [m.strip().lower() for m in _parse_csv_list(args.source_motors)]
    targets = [m.strip().lower() for m in _parse_csv_list(args.target_motors)]
    seeds = _parse_int_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    for m in set([*sources, *targets]):
        if m not in MOTOR_REGISTRY:
            raise ValueError(f"Unknown motor key={m}")
    if not sources or not targets:
        raise ValueError("Empty sources or targets")
    if not seeds or not scenarios:
        raise ValueError("Empty seeds or scenarios")

    manifest_index: Dict[Tuple[str, int], Dict[str, object]] = {}
    manifest_path = Path(str(args.train_manifest)).expanduser().resolve() if str(args.train_manifest).strip() else None
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest_index = _load_manifest_checkpoint_index(manifest_path)

    seed_pert = SeedPerturbationSettings(
        enabled=bool(args.seed_perturbation),
        level=float(max(0.0, float(args.seed_perturb_level))),
    )

    native_pairs = sorted(set((t, t) for t in targets))
    eval_pairs: List[Tuple[str, str]] = []
    for s in sources:
        for t in targets:
            if str(args.mode) == "heldout" and s == t:
                continue
            eval_pairs.append((s, t))
    eval_pairs = sorted(set(eval_pairs))

    per_seed_rows: List[Dict[str, object]] = []
    pair_plan_rows: List[Dict[str, object]] = []

    all_pairs = sorted(set(native_pairs + eval_pairs))
    for source_motor, target_motor in all_pairs:
        target_spec = MOTOR_REGISTRY[str(target_motor)]
        target_env, _none_agent, _none_ckpt = _load_env_and_agent(
            target_spec.config_path,
            foc_disable_lut=bool(args.foc_disable_lut),
            require_agent=False,
            motor_key=str(target_motor),
            checkpoint_registry_path=str(args.checkpoint_registry),
        )
        id_ref = _id_ref_eval_params(target_env)
        sensorless = _sensorless_params(target_env)
        supervisor = _supervisor_from_env(target_env)

        for seed in seeds:
            ckpt_path, ckpt_source = _resolve_checkpoint_for_source(
                source_motor=str(source_motor),
                seed=int(seed),
                manifest_index=manifest_index,
                checkpoint_registry_path=str(args.checkpoint_registry),
                foc_disable_lut=bool(args.foc_disable_lut),
            )
            pair_plan_rows.append(
                {
                    "source_motor": str(source_motor),
                    "target_motor": str(target_motor),
                    "seed": int(seed),
                    "checkpoint": str(ckpt_path),
                    "checkpoint_source": str(ckpt_source),
                }
            )
            if bool(args.dry_run):
                continue
            agent = _load_agent(ckpt_path)
            rows = _simulate_rows(
                env_cfg=target_env,
                motor_key=str(target_motor),
                agent=agent,
                scenarios=scenarios,
                seed=int(seed),
                window_frac=float(args.window_frac),
                error_tol_rel=float(args.error_tol_rel),
                error_tol_abs=float(args.error_tol_abs),
                use_total_power=bool(args.use_total_power),
                foc_feedback_mode=str(args.foc_feedback_mode),
                mic_feedback_mode=str(args.mic_feedback_mode),
                controller="MIC",
                id_ref_params=id_ref,
                supervisor_cfg=supervisor,
                sensorless=sensorless,
                seed_perturbation=seed_pert,
                mic_mode="ai",
                mic_rule_params=None,
            )
            agg = _aggregate_rows(rows)
            per_seed_rows.append(
                {
                    "source_motor": str(source_motor),
                    "target_motor": str(target_motor),
                    "seed": int(seed),
                    "checkpoint": str(ckpt_path),
                    "checkpoint_source": str(ckpt_source),
                    "avg_power_saving_pct": float(agg["avg_power_saving_pct"]),
                    "avg_eta_gain_pct": float(agg["avg_eta_gain_pct"]),
                    "err_failures": float(agg["err_failures"]),
                    "start_stop_power_saving_pct": float(agg["start_stop_power_saving_pct"]),
                    "worst_current_peak_ratio": float(agg["worst_current_peak_ratio"]),
                    "worst_current_mean_ratio": float(agg["worst_current_mean_ratio"]),
                    "avg_controller_speed_err": float(agg["avg_mic_err"]),
                }
            )
            print(
                "[cross-generalization] source={} target={} seed={} power={:.3f}% eta={:.3f}% err={:.1f}".format(
                    source_motor,
                    target_motor,
                    int(seed),
                    float(agg["avg_power_saving_pct"]),
                    float(agg["avg_eta_gain_pct"]),
                    float(agg["err_failures"]),
                ),
                flush=True,
            )

    summary_rows = _group_stats(
        per_seed_rows,
        group_keys=("source_motor", "target_motor"),
        metric_keys=METRIC_KEYS,
    )

    native_idx: Dict[str, Dict[str, object]] = {}
    for row in summary_rows:
        s = str(row.get("source_motor", ""))
        t = str(row.get("target_motor", ""))
        if s == t:
            native_idx[t] = row

    gap_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        s = str(row.get("source_motor", ""))
        t = str(row.get("target_motor", ""))
        if s == t:
            continue
        native = native_idx.get(t)
        if native is None:
            continue
        gap_rows.append(
            {
                "source_motor": s,
                "target_motor": t,
                "delta_power_vs_native_mean": float(row.get("avg_power_saving_pct_mean", 0.0))
                - float(native.get("avg_power_saving_pct_mean", 0.0)),
                "delta_eta_vs_native_mean": float(row.get("avg_eta_gain_pct_mean", 0.0))
                - float(native.get("avg_eta_gain_pct_mean", 0.0)),
                "delta_err_vs_native_mean": float(row.get("err_failures_mean", 0.0))
                - float(native.get("err_failures_mean", 0.0)),
            }
        )
    gap_rows.sort(key=lambda r: (str(r["target_motor"]), str(r["source_motor"])))

    _write_csv(out_dir / "cross_motor_generalization_pair_plan.csv", pair_plan_rows)
    _json_dump(out_dir / "cross_motor_generalization_pair_plan.json", pair_plan_rows)
    _write_csv(out_dir / "cross_motor_generalization_per_seed.csv", per_seed_rows)
    _json_dump(out_dir / "cross_motor_generalization_per_seed.json", per_seed_rows)
    _write_csv(out_dir / "cross_motor_generalization_summary.csv", summary_rows)
    _json_dump(out_dir / "cross_motor_generalization_summary.json", summary_rows)
    _write_csv(out_dir / "cross_motor_generalization_gap_vs_native.csv", gap_rows)
    _json_dump(out_dir / "cross_motor_generalization_gap_vs_native.json", gap_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(float(time.time() - t0), 3),
        "mode": str(args.mode),
        "source_motors": sources,
        "target_motors": targets,
        "seeds": [int(s) for s in seeds],
        "scenarios": scenarios,
        "dry_run": bool(args.dry_run),
        "checkpoint_registry": str(Path(str(args.checkpoint_registry)).resolve()),
        "train_manifest": "" if manifest_path is None else str(manifest_path),
        "artifacts": {
            "pair_plan_csv": str((out_dir / "cross_motor_generalization_pair_plan.csv").resolve()),
            "pair_plan_json": str((out_dir / "cross_motor_generalization_pair_plan.json").resolve()),
            "per_seed_csv": str((out_dir / "cross_motor_generalization_per_seed.csv").resolve()),
            "per_seed_json": str((out_dir / "cross_motor_generalization_per_seed.json").resolve()),
            "summary_csv": str((out_dir / "cross_motor_generalization_summary.csv").resolve()),
            "summary_json": str((out_dir / "cross_motor_generalization_summary.json").resolve()),
            "gap_csv": str((out_dir / "cross_motor_generalization_gap_vs_native.csv").resolve()),
            "gap_json": str((out_dir / "cross_motor_generalization_gap_vs_native.json").resolve()),
            "report_md": str((out_dir / "cross_motor_generalization_report.md").resolve()),
        },
    }
    _json_dump(out_dir / "cross_motor_generalization_manifest.json", manifest)
    (out_dir / "cross_motor_generalization_report.md").write_text(
        _build_report(args=args, per_seed=per_seed_rows, summary=summary_rows, gaps=gap_rows),
        encoding="utf-8",
    )
    print(f"[cross-generalization] out_dir={out_dir}")


if __name__ == "__main__":
    main()
