from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisorConfig  # noqa: E402
from tools.common_utils import json_dump as _json_dump_shared  # noqa: E402
from tools.common_utils import write_csv as _write_csv_shared  # noqa: E402
from tools.step27_pipeline import (  # noqa: E402
    _id_ref_eval_params,
    _load_agent,
    _load_env_and_agent,
    _supervisor_from_env,
    _supervisor_to_candidate,
)
from tools.tune_motor_step27 import (  # noqa: E402
    MOTOR_REGISTRY,
    SeedPerturbationSettings,
    _eval_candidate,
    _load_custom_candidates,
    _parse_csv_list,
    _parse_int_list,
    _pass,
    _score,
)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _json_dump(path: Path, payload: object) -> None:
    _json_dump_shared(path, payload)


def _build_base_candidate(env_cfg: object) -> Dict[str, object]:
    base_sup = _supervisor_from_env(env_cfg)
    if base_sup is None:
        base_sup = AiIdRefSupervisorConfig(enabled=True)
    base = _supervisor_to_candidate(base_sup, tag="baseline", source="config")
    base_id_ref = _id_ref_eval_params(env_cfg)
    base.update(
        {
            "id_ref_alpha": float(base_id_ref["id_ref_alpha"]),
            "delta_id_max": float(base_id_ref["delta_id_max"]),
            "id_ref_gate_speed_tol_rel": float(base_id_ref["id_ref_gate_speed_tol_rel"] or 0.05),
            "id_ref_gate_min_scale": float(base_id_ref["id_ref_gate_min_scale"]),
            "id_ref_gate_exponent": float(base_id_ref["id_ref_gate_exponent"]),
        }
    )
    return base


def _select_candidate(
    candidate_json: Path,
    *,
    base: Dict[str, object],
    candidate_index: int,
    candidate_tag: str,
) -> Tuple[Dict[str, object], int]:
    candidates = _load_custom_candidates(candidate_json.expanduser().resolve(), base=base)
    if not candidates:
        raise ValueError(f"No candidates found in {candidate_json}")

    tag = str(candidate_tag).strip()
    if tag:
        matches = [cand for cand in candidates if str(cand.get("tag", "")) == tag]
        if not matches:
            raise ValueError(f"Candidate tag '{tag}' was not found in {candidate_json}")
        if len(matches) > 1:
            raise ValueError(f"Candidate tag '{tag}' is ambiguous in {candidate_json}")
        return matches[0], len(candidates)

    idx = int(candidate_index)
    if idx < 0 or idx >= len(candidates):
        raise ValueError(f"Candidate index {idx} is out of range for {candidate_json} ({len(candidates)} candidates)")
    return candidates[idx], len(candidates)


def _collect_checkpoint_paths(pattern: str) -> List[Path]:
    text = str(pattern).strip()
    if not text:
        raise ValueError("Empty --checkpoint-glob")
    path = Path(text)
    if path.exists() and path.is_dir():
        return sorted(path.glob("*.pth"))

    if any(ch in text for ch in "*?[]"):
        return sorted(Path().glob(text))

    if path.exists():
        return [path]

    return sorted(Path().glob(text))


def _mean_score(rows: List[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(statistics.fmean(float(r.get("score", 0.0)) for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate many checkpoints against step27 metrics with one fixed candidate.")
    parser.add_argument("--motor", required=True, choices=sorted(MOTOR_REGISTRY.keys()))
    parser.add_argument("--checkpoint-glob", required=True, help="Checkpoint file, directory, or glob pattern (e.g. outputs/run/eval/actor_ep*.pth).")
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--candidate-tag", default="")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--out-dir", default="outputs/scan_step27_checkpoints")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--no-use-total-power", dest="use_total_power", action="store_false")
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--min-avg-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--min-avg-eta-gain-pct", type=float, default=0.0)
    parser.add_argument("--max-avg-eta-gain-pct", type=float, default=25.0)
    parser.add_argument("--max-err-failures", type=float, default=2.0)
    parser.add_argument("--min-start-stop-saving-pct", type=float, default=-0.5)
    parser.add_argument("--max-start-stop-saving-pct", type=float, default=20.0)
    parser.add_argument("--max-worst-current-peak-ratio", type=float, default=1.30)
    parser.add_argument("--max-worst-current-mean-ratio", type=float, default=1.20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    if not seeds:
        raise ValueError("Empty seeds list")
    if not scenarios:
        raise ValueError("Empty scenarios list")

    checkpoints = _collect_checkpoint_paths(str(args.checkpoint_glob))
    if not checkpoints:
        raise ValueError(f"No checkpoints matched: {args.checkpoint_glob}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, _, _ = _load_env_and_agent(
        MOTOR_REGISTRY[str(args.motor)].config_path,
        foc_disable_lut=bool(args.foc_disable_lut),
        require_agent=False,
        motor_key=str(args.motor),
        checkpoint_registry_path=None,
    )
    base_candidate = _build_base_candidate(env_cfg)
    candidate, candidate_count = _select_candidate(
        Path(args.candidate_json),
        base=base_candidate,
        candidate_index=int(args.candidate_index),
        candidate_tag=str(args.candidate_tag),
    )

    seed_perturb = SeedPerturbationSettings(
        enabled=bool(args.seed_perturbation),
        level=float(max(0.0, float(args.seed_perturb_level))),
    )

    rows: List[Dict[str, object]] = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        agent = _load_agent(ckpt.resolve())
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=str(args.motor),
            agent=agent,
            candidate=candidate,
            scenarios=scenarios,
            seeds=seeds,
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            seed_perturbation=seed_perturb,
        )
        row = {
            "rank_input": idx,
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_name": ckpt.name,
            **metrics,
        }
        row["score"] = _score(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_eta=float(args.max_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
            max_start_stop=float(args.max_start_stop_saving_pct),
            max_peak_ratio=float(args.max_worst_current_peak_ratio),
            max_mean_ratio=float(args.max_worst_current_mean_ratio),
        )
        row["acceptance_pass"] = _pass(
            metrics,
            min_power=float(args.min_avg_power_saving_pct),
            min_eta=float(args.min_avg_eta_gain_pct),
            max_eta=float(args.max_avg_eta_gain_pct),
            max_err=float(args.max_err_failures),
            min_start_stop=float(args.min_start_stop_saving_pct),
            max_start_stop=float(args.max_start_stop_saving_pct),
            max_peak_ratio=float(args.max_worst_current_peak_ratio),
            max_mean_ratio=float(args.max_worst_current_mean_ratio),
        )
        rows.append(row)
        print(
            f"[scan:{args.motor}] {idx}/{len(checkpoints)} {ckpt.name} "
            f"power={row['avg_power_saving_pct']:.3f}% eta={row['avg_eta_gain_pct']:.3f}% "
            f"start_stop={row['start_stop_power_saving_pct']:.3f}% err={row['err_failures']:.1f}"
        )

    rows.sort(key=lambda r: (float(r["score"]), -float(r["avg_power_saving_pct"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    _write_csv(out_dir / f"{args.motor}_checkpoint_scan.csv", rows)
    _json_dump(out_dir / f"{args.motor}_checkpoint_scan.json", rows)
    _json_dump(out_dir / f"{args.motor}_selected_candidate.json", candidate)

    summary = {
        "motor": str(args.motor),
        "checkpoint_glob": str(args.checkpoint_glob),
        "candidate_json": str(args.candidate_json),
        "candidate_index": int(args.candidate_index),
        "candidate_tag": str(args.candidate_tag),
        "candidate_count": int(candidate_count),
        "seeds": [int(s) for s in seeds],
        "scenarios": [str(s) for s in scenarios],
        "top_k": int(max(1, int(args.top_k))),
        "scan_rows": int(len(rows)),
        "acceptance": {
            "min_avg_power_saving_pct": float(args.min_avg_power_saving_pct),
            "min_avg_eta_gain_pct": float(args.min_avg_eta_gain_pct),
            "max_avg_eta_gain_pct": float(args.max_avg_eta_gain_pct),
            "max_err_failures": float(args.max_err_failures),
            "min_start_stop_saving_pct": float(args.min_start_stop_saving_pct),
            "max_start_stop_saving_pct": float(args.max_start_stop_saving_pct),
            "max_worst_current_peak_ratio": float(args.max_worst_current_peak_ratio),
            "max_worst_current_mean_ratio": float(args.max_worst_current_mean_ratio),
        },
        "score_mean": _mean_score(rows),
        "best": rows[0] if rows else {},
        "top_rows": rows[: max(1, int(args.top_k))],
    }
    _json_dump(out_dir / f"{args.motor}_checkpoint_scan_summary.json", summary)
    print(f"[scan:{args.motor}] done. best={rows[0]['checkpoint_name']} score={rows[0]['score']:.3f}")
    print(f"[scan:{args.motor}] summary={out_dir / f'{args.motor}_checkpoint_scan_summary.json'}")


if __name__ == "__main__":
    main()
