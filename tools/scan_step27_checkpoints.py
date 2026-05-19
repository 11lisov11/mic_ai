from __future__ import annotations

import argparse
import hashlib
import json
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
    AI_ID_REF_HYBRID_MODE,
    _id_ref_eval_params,
    _load_agent,
    _load_env_and_agent,
    _normalize_ai_control_mode,
    _supervisor_from_env,
    _supervisor_to_candidate,
)
from tools.tune_motor_step27 import (  # noqa: E402
    DEFAULT_ACCEPTANCE_ENVELOPES,
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


def _parse_candidate_tags(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _select_candidates(
    candidate_json: Path,
    *,
    base: Dict[str, object],
    candidate_index: int,
    candidate_tag: str,
    candidate_tags: List[str] | None = None,
) -> Tuple[List[Tuple[Dict[str, object], int]], int]:
    candidates = _load_custom_candidates(candidate_json.expanduser().resolve(), base=base)
    if not candidates:
        raise ValueError(f"No candidates found in {candidate_json}")

    requested_tags = [str(tag).strip() for tag in (candidate_tags or []) if str(tag).strip()]
    if requested_tags:
        tag_to_rows: Dict[str, List[Tuple[Dict[str, object], int]]] = {}
        for idx, cand in enumerate(candidates):
            tag_to_rows.setdefault(str(cand.get("tag", "")), []).append((cand, idx))
        selected: List[Tuple[Dict[str, object], int]] = []
        for tag in requested_tags:
            matches = tag_to_rows.get(tag, [])
            if not matches:
                raise ValueError(f"Candidate tag '{tag}' was not found in {candidate_json}")
            if len(matches) > 1:
                raise ValueError(f"Candidate tag '{tag}' is ambiguous in {candidate_json}")
            selected.append(matches[0])
        return selected, len(candidates)

    row, total = _select_candidate(
        candidate_json,
        base=base,
        candidate_index=int(candidate_index),
        candidate_tag=str(candidate_tag),
    )
    selected_tag = str(row.get("tag", ""))
    for idx, cand in enumerate(candidates):
        if str(cand.get("tag", "")) == selected_tag:
            return [(cand, idx)], total
    idx = int(candidate_index)
    if 0 <= idx < len(candidates):
        return [(candidates[idx], idx)], total
    raise ValueError(f"Selected candidate '{selected_tag}' was not found in {candidate_json}")


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


def _resolve_existing_path_for_hash(path_text: str) -> Path | None:
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_fingerprints(checkpoints: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ckpt in checkpoints:
        path = ckpt.resolve()
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "exists": bool(path.exists()),
                "size": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "sha256": _file_sha256(path),
            }
        )
    return rows


def _mean_score(rows: List[Dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return float(statistics.fmean(float(r.get("score", 0.0)) for r in rows))


def _sorted_ranked_rows(
    evaluated_rows: List[Dict[str, object]],
    *,
    use_envelope_acceptance: bool,
) -> List[Dict[str, object]]:
    ranked_rows = [dict(row) for row in evaluated_rows]
    if not ranked_rows:
        return []
    ranked_rows.sort(key=lambda r: _candidate_row_sort_key(r, use_envelope_acceptance=bool(use_envelope_acceptance)))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    return ranked_rows


def _candidate_row_sort_key(
    row: Dict[str, object],
    *,
    use_envelope_acceptance: bool,
) -> Tuple[object, ...]:
    if bool(use_envelope_acceptance):
        return (
            not bool(row.get("acceptance_pass", False)),
            float(row.get("envelope_fail_count", 0.0)),
            float(row.get("envelope_gap_total", 0.0)),
            float(row.get("selector_score", row.get("aggregate_score", row.get("score", 0.0)))),
            -float(row["avg_power_saving_pct"]),
        )
    return (
        not bool(row.get("acceptance_pass", False)),
        float(row.get("selector_score", row.get("aggregate_score", row.get("score", 0.0)))),
        -float(row["avg_power_saving_pct"]),
    )


def _select_best_candidate_row(
    rows: List[Dict[str, object]],
    *,
    use_envelope_acceptance: bool,
) -> Dict[str, object]:
    if not rows:
        raise ValueError("Cannot select best candidate row from an empty list")
    ranked_rows = sorted(
        (dict(row) for row in rows),
        key=lambda r: _candidate_row_sort_key(r, use_envelope_acceptance=bool(use_envelope_acceptance)),
    )
    best = dict(ranked_rows[0])
    best["candidate_variants_evaluated"] = int(len(ranked_rows))
    best["candidate_tags_evaluated"] = [str(row.get("candidate_tag", "")) for row in ranked_rows]
    return best


def _write_progress_summary(
    *,
    out_dir_path: Path,
    motor: str,
    ai_control_mode: str,
    evaluated_rows: List[Dict[str, object]],
    skipped_rows: List[Dict[str, object]],
    processed_count: int,
    total_count: int,
    top_k: int,
    use_envelope_acceptance: bool,
    candidate_json: str,
    candidate_tag: str,
    candidate_tags: List[str],
    candidate_index: int,
    complete: bool,
    last_checkpoint_name: str,
) -> None:
    ranked_rows = _sorted_ranked_rows(evaluated_rows, use_envelope_acceptance=bool(use_envelope_acceptance))
    progress = {
        "motor": str(motor),
        "ai_control_mode": str(ai_control_mode),
        "complete": bool(complete),
        "processed_count": int(processed_count),
        "total_count": int(total_count),
        "evaluated_count": int(len(evaluated_rows)),
        "skipped_count": int(len(skipped_rows)),
        "candidate_json": str(candidate_json),
        "candidate_tag": str(candidate_tag),
        "candidate_tags": list(candidate_tags),
        "candidate_index": int(candidate_index),
        "selector_mode": "canonical_envelope_then_aggregate" if bool(use_envelope_acceptance) else "aggregate_only",
        "last_checkpoint_name": str(last_checkpoint_name),
        "best_so_far": ranked_rows[0] if ranked_rows else None,
        "top_rows": ranked_rows[: max(1, int(top_k))],
        "skipped_rows": [dict(row) for row in skipped_rows],
    }
    _json_dump(out_dir_path / f"{motor}_checkpoint_scan_progress.json", progress)


def _scan_state_signature(
    *,
    motor: str,
    config_path: str,
    checkpoint_glob: str,
    config_sha256: str = "",
    candidate_json_sha256: str = "",
    acceptance_envelopes_sha256: str = "",
    checkpoint_fingerprints: List[Dict[str, object]] | None = None,
    ai_control_mode: str,
    candidate_json: str,
    candidate_index: int,
    candidate_tag: str,
    candidate_tags: List[str] | None,
    seeds: List[int],
    scenarios: List[str],
    use_envelope_acceptance: bool,
) -> Dict[str, object]:
    return {
        "signature_version": 2,
        "motor": str(motor),
        "config_path": str(config_path),
        "config_sha256": str(config_sha256),
        "checkpoint_glob": str(checkpoint_glob),
        "checkpoint_fingerprints": [dict(row) for row in (checkpoint_fingerprints or [])],
        "ai_control_mode": str(ai_control_mode),
        "candidate_json": str(candidate_json),
        "candidate_json_sha256": str(candidate_json_sha256),
        "candidate_index": int(candidate_index),
        "candidate_tag": str(candidate_tag),
        "candidate_tags": list(candidate_tags or []),
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "use_envelope_acceptance": bool(use_envelope_acceptance),
        "acceptance_envelopes_sha256": str(acceptance_envelopes_sha256),
    }


def _load_scan_state(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid scan state format: {path}")
    return payload


def _write_scan_state(
    *,
    path: Path,
    signature: Dict[str, object],
    evaluated_rows: List[Dict[str, object]],
    skipped_rows: List[Dict[str, object]],
    complete: bool,
    last_checkpoint_name: str,
) -> None:
    payload = {
        "signature": dict(signature),
        "complete": bool(complete),
        "last_checkpoint_name": str(last_checkpoint_name),
        "evaluated_rows": [dict(row) for row in evaluated_rows],
        "skipped_rows": [dict(row) for row in skipped_rows],
    }
    _json_dump(path, payload)


def _skip_row(ckpt: Path, reason: str) -> Dict[str, object]:
    return {
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_name": ckpt.name,
        "status": "skipped",
        "skip_reason": reason,
        "score": float("inf"),
        "acceptance_pass": False,
    }


def scan_checkpoints(
    *,
    motor: str,
    config_path: str | None = None,
    checkpoint_glob: str,
    candidate_json: str = "",
    ai_control_mode: str = "ai_id_ref",
    candidate_index: int = 0,
    candidate_tag: str = "",
    candidate_tags: List[str] | None = None,
    seeds: List[int] | None = None,
    scenarios: List[str] | None = None,
    out_dir: Path | str = Path("outputs/scan_step27_checkpoints"),
    window_frac: float = 0.25,
    error_tol_rel: float = 0.05,
    error_tol_abs: float = 0.0,
    foc_feedback_mode: str = "encoder",
    mic_feedback_mode: str = "sensorless",
    seed_perturbation: bool = False,
    seed_perturb_level: float = 0.2,
    use_total_power: bool = True,
    foc_disable_lut: bool = True,
    min_avg_power_saving_pct: float = 0.0,
    min_avg_eta_gain_pct: float = 0.0,
    max_avg_eta_gain_pct: float = 25.0,
    max_err_failures: float = 2.0,
    min_start_stop_saving_pct: float = -0.5,
    max_start_stop_saving_pct: float = 20.0,
    max_worst_current_peak_ratio: float = 1.30,
    max_worst_current_mean_ratio: float = 1.20,
    min_avg_power_saving_pct_min_seed: float | None = None,
    min_avg_eta_gain_pct_min_seed: float | None = None,
    max_err_failures_max_seed: float | None = None,
    min_start_stop_saving_pct_min_seed: float | None = None,
    use_envelope_acceptance: bool = False,
    acceptance_envelopes: Path | str | None = DEFAULT_ACCEPTANCE_ENVELOPES,
    top_k: int = 10,
    feature_keys: List[str] | None = None,
    resume: bool = False,
) -> Dict[str, object]:
    ai_mode = str(ai_control_mode).strip().lower()
    eval_ai_mode = _normalize_ai_control_mode(ai_mode)
    seeds_list = [int(s) for s in (seeds or [101])]
    scenarios_list = [str(s) for s in (scenarios or ["speed_step", "ramp", "load_step", "start_stop"])]
    if not seeds_list:
        raise ValueError("Empty seeds list")
    if not scenarios_list:
        raise ValueError("Empty scenarios list")

    checkpoints = _collect_checkpoint_paths(str(checkpoint_glob))
    if not checkpoints:
        raise ValueError(f"No checkpoints matched: {checkpoint_glob}")

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    state_path = out_dir_path / f"{motor}_checkpoint_scan_state.json"
    if acceptance_envelopes is None:
        acceptance_envelopes_path = Path(DEFAULT_ACCEPTANCE_ENVELOPES).expanduser().resolve()
    else:
        acceptance_envelopes_path = Path(acceptance_envelopes).expanduser().resolve()
    config_path_text = "" if config_path is None else str(config_path)
    config_hash_path = _resolve_existing_path_for_hash(config_path_text)
    candidate_hash_path = _resolve_existing_path_for_hash(str(candidate_json))
    checkpoint_hashes = _checkpoint_fingerprints(checkpoints)

    env_cfg, _, _ = _load_env_and_agent(
        str(config_path or MOTOR_REGISTRY[str(motor)].config_path),
        foc_disable_lut=bool(foc_disable_lut),
        require_agent=False,
        motor_key=str(motor),
        checkpoint_registry_path=None,
    )
    base_candidate = _build_base_candidate(env_cfg)
    candidate_tags_list = [str(tag).strip() for tag in (candidate_tags or []) if str(tag).strip()]
    if eval_ai_mode == "ai_id_ref":
        if not str(candidate_json).strip():
            raise ValueError("--candidate-json is required for ai_id_ref scans.")
        selected_candidates, candidate_count = _select_candidates(
            Path(candidate_json),
            base=base_candidate,
            candidate_index=int(candidate_index),
            candidate_tag=str(candidate_tag),
            candidate_tags=candidate_tags_list,
        )
    else:
        candidate = dict(base_candidate)
        candidate["tag"] = "baseline"
        candidate["source"] = f"{ai_mode}_baseline"
        selected_candidates = [(candidate, 0)]
        candidate_count = 1

    seed_perturb = SeedPerturbationSettings(
        enabled=bool(seed_perturbation),
        level=float(max(0.0, float(seed_perturb_level))),
    )

    signature = _scan_state_signature(
        motor=str(motor),
        config_path=config_path_text,
        config_sha256=_file_sha256(config_hash_path),
        checkpoint_glob=str(checkpoint_glob),
        checkpoint_fingerprints=checkpoint_hashes,
        ai_control_mode=ai_mode,
        candidate_json=str(candidate_json),
        candidate_json_sha256=_file_sha256(candidate_hash_path),
        candidate_index=int(candidate_index),
        candidate_tag=str(candidate_tag),
        candidate_tags=candidate_tags_list,
        seeds=seeds_list,
        scenarios=scenarios_list,
        use_envelope_acceptance=bool(use_envelope_acceptance),
        acceptance_envelopes_sha256=_file_sha256(acceptance_envelopes_path),
    )

    evaluated_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    processed_checkpoints: set[str] = set()
    last_checkpoint_name = ""
    if bool(resume):
        state = _load_scan_state(state_path)
        if state is not None:
            state_signature = state.get("signature")
            if state_signature != signature:
                raise ValueError(f"Resume state does not match current scan configuration: {state_path}")
            evaluated_rows = [dict(row) for row in state.get("evaluated_rows", [])]
            skipped_rows = [dict(row) for row in state.get("skipped_rows", [])]
            processed_checkpoints = {
                str(row.get("checkpoint", ""))
                for row in list(evaluated_rows) + list(skipped_rows)
                if str(row.get("checkpoint", ""))
            }
            last_checkpoint_name = str(state.get("last_checkpoint_name", ""))
            if processed_checkpoints:
                print(f"[scan:{motor}] resume loaded {len(processed_checkpoints)} processed checkpoints from {state_path}")

    for idx, ckpt in enumerate(checkpoints, start=1):
        ckpt_resolved = str(ckpt.resolve())
        if ckpt_resolved in processed_checkpoints:
            continue
        if not ckpt.exists():
            skipped_rows.append(_skip_row(ckpt, "missing_file"))
            processed_checkpoints.add(ckpt_resolved)
            last_checkpoint_name = ckpt.name
            processed_count = len(processed_checkpoints)
            _write_scan_state(
                path=state_path,
                signature=signature,
                evaluated_rows=evaluated_rows,
                skipped_rows=skipped_rows,
                complete=False,
                last_checkpoint_name=last_checkpoint_name,
            )
            _write_progress_summary(
                out_dir_path=out_dir_path,
                motor=str(motor),
                ai_control_mode=ai_mode,
                evaluated_rows=evaluated_rows,
                skipped_rows=skipped_rows,
                processed_count=processed_count,
                total_count=len(checkpoints),
                top_k=top_k,
                use_envelope_acceptance=bool(use_envelope_acceptance),
                candidate_json=str(candidate_json),
                candidate_tag=str(candidate_tag),
                candidate_tags=candidate_tags_list,
                candidate_index=int(candidate_index),
                complete=False,
                last_checkpoint_name=last_checkpoint_name,
            )
            print(f"[scan:{motor}] {processed_count}/{len(checkpoints)} {ckpt.name} skipped=missing_file")
            continue
        if feature_keys is None:
            agent = _load_agent(ckpt.resolve())
        else:
            agent = _load_agent(ckpt.resolve(), feature_keys=feature_keys)
        candidate_rows: List[Dict[str, object]] = []
        for candidate, selected_index in selected_candidates:
            metrics = _eval_candidate(
                env_cfg=env_cfg,
                motor_key=str(motor),
                agent=agent,
                candidate=candidate,
                ai_control_mode=ai_mode,
                scenarios=scenarios_list,
                seeds=seeds_list,
                window_frac=float(window_frac),
                error_tol_rel=float(error_tol_rel),
                error_tol_abs=float(error_tol_abs),
                use_total_power=bool(use_total_power),
                foc_feedback_mode=str(foc_feedback_mode),
                mic_feedback_mode=str(mic_feedback_mode),
                seed_perturbation=seed_perturb,
                acceptance_envelopes_path=acceptance_envelopes_path,
            )
            row = {
                "rank_input": idx,
                "checkpoint": str(ckpt.resolve()),
                "checkpoint_name": ckpt.name,
                "status": "evaluated",
                "skip_reason": None,
                "candidate_tag": str(candidate.get("tag", "")),
                "candidate_source": str(candidate.get("source", "")),
                "candidate_index": int(selected_index),
                **metrics,
            }
            aggregate_score = _score(
                metrics,
                min_power=float(min_avg_power_saving_pct),
                min_eta=float(min_avg_eta_gain_pct),
                max_eta=float(max_avg_eta_gain_pct),
                max_err=float(max_err_failures),
                min_start_stop=float(min_start_stop_saving_pct),
                max_start_stop=float(max_start_stop_saving_pct),
                max_peak_ratio=float(max_worst_current_peak_ratio),
                max_mean_ratio=float(max_worst_current_mean_ratio),
                require_envelope_pass=False,
                min_power_min_seed=min_avg_power_saving_pct_min_seed,
                min_eta_min_seed=min_avg_eta_gain_pct_min_seed,
                max_err_max_seed=max_err_failures_max_seed,
                min_start_stop_min_seed=min_start_stop_saving_pct_min_seed,
            )
            selector_score = _score(
                metrics,
                min_power=float(min_avg_power_saving_pct),
                min_eta=float(min_avg_eta_gain_pct),
                max_eta=float(max_avg_eta_gain_pct),
                max_err=float(max_err_failures),
                min_start_stop=float(min_start_stop_saving_pct),
                max_start_stop=float(max_start_stop_saving_pct),
                max_peak_ratio=float(max_worst_current_peak_ratio),
                max_mean_ratio=float(max_worst_current_mean_ratio),
                require_envelope_pass=bool(use_envelope_acceptance),
                min_power_min_seed=min_avg_power_saving_pct_min_seed,
                min_eta_min_seed=min_avg_eta_gain_pct_min_seed,
                max_err_max_seed=max_err_failures_max_seed,
                min_start_stop_min_seed=min_start_stop_saving_pct_min_seed,
            )
            aggregate_pass = _pass(
                metrics,
                min_power=float(min_avg_power_saving_pct),
                min_eta=float(min_avg_eta_gain_pct),
                max_eta=float(max_avg_eta_gain_pct),
                max_err=float(max_err_failures),
                min_start_stop=float(min_start_stop_saving_pct),
                max_start_stop=float(max_start_stop_saving_pct),
                max_peak_ratio=float(max_worst_current_peak_ratio),
                max_mean_ratio=float(max_worst_current_mean_ratio),
                require_envelope_pass=False,
                min_power_min_seed=min_avg_power_saving_pct_min_seed,
                min_eta_min_seed=min_avg_eta_gain_pct_min_seed,
                max_err_max_seed=max_err_failures_max_seed,
                min_start_stop_min_seed=min_start_stop_saving_pct_min_seed,
            )
            acceptance_pass = _pass(
                metrics,
                min_power=float(min_avg_power_saving_pct),
                min_eta=float(min_avg_eta_gain_pct),
                max_eta=float(max_avg_eta_gain_pct),
                max_err=float(max_err_failures),
                min_start_stop=float(min_start_stop_saving_pct),
                max_start_stop=float(max_start_stop_saving_pct),
                max_peak_ratio=float(max_worst_current_peak_ratio),
                max_mean_ratio=float(max_worst_current_mean_ratio),
                require_envelope_pass=bool(use_envelope_acceptance),
                min_power_min_seed=min_avg_power_saving_pct_min_seed,
                min_eta_min_seed=min_avg_eta_gain_pct_min_seed,
                max_err_max_seed=max_err_failures_max_seed,
                min_start_stop_min_seed=min_start_stop_saving_pct_min_seed,
            )
            row["score"] = float(aggregate_score)
            row["aggregate_score"] = float(aggregate_score)
            row["selector_score"] = float(selector_score)
            row["acceptance_pass_aggregate"] = bool(aggregate_pass)
            row["acceptance_pass"] = bool(acceptance_pass)
            candidate_rows.append(row)
        evaluated_rows.append(_select_best_candidate_row(candidate_rows, use_envelope_acceptance=bool(use_envelope_acceptance)))
        processed_checkpoints.add(ckpt_resolved)
        last_checkpoint_name = ckpt.name
        processed_count = len(processed_checkpoints)
        _write_scan_state(
            path=state_path,
            signature=signature,
            evaluated_rows=evaluated_rows,
            skipped_rows=skipped_rows,
            complete=False,
            last_checkpoint_name=last_checkpoint_name,
        )
        _write_progress_summary(
            out_dir_path=out_dir_path,
            motor=str(motor),
            ai_control_mode=ai_mode,
            evaluated_rows=evaluated_rows,
            skipped_rows=skipped_rows,
            processed_count=processed_count,
            total_count=len(checkpoints),
            top_k=top_k,
            use_envelope_acceptance=bool(use_envelope_acceptance),
            candidate_json=str(candidate_json),
            candidate_tag=str(candidate_tag),
            candidate_tags=candidate_tags_list,
            candidate_index=int(candidate_index),
            complete=False,
            last_checkpoint_name=last_checkpoint_name,
        )
        print(
            f"[scan:{motor}] {processed_count}/{len(checkpoints)} {ckpt.name} "
            f"power={row['avg_power_saving_pct']:.3f}% eta={row['avg_eta_gain_pct']:.3f}% "
            f"start_stop={row['start_stop_power_saving_pct']:.3f}% err={row['err_failures']:.1f}"
        )

    evaluated_rows = _sorted_ranked_rows(evaluated_rows, use_envelope_acceptance=bool(use_envelope_acceptance))
    rows: List[Dict[str, object]] = list(evaluated_rows) + list(skipped_rows)
    for row in skipped_rows:
        row["rank"] = 10**9

    _write_csv(out_dir_path / f"{motor}_checkpoint_scan.csv", rows)
    _json_dump(out_dir_path / f"{motor}_checkpoint_scan.json", rows)
    best_selected_candidate = None
    if evaluated_rows:
        best_candidate_tag = str(evaluated_rows[0].get("candidate_tag", "")).strip()
        best_candidate_index = int(evaluated_rows[0].get("candidate_index", 0))
        if ai_mode == "ai_id_ref":
            for candidate, selected_index in selected_candidates:
                if best_candidate_tag and str(candidate.get("tag", "")) == best_candidate_tag:
                    best_selected_candidate = dict(candidate)
                    break
                if not best_candidate_tag and int(selected_index) == best_candidate_index:
                    best_selected_candidate = dict(candidate)
                    break
        else:
            best_selected_candidate = dict(selected_candidates[0][0])
    _json_dump(out_dir_path / f"{motor}_selected_candidate.json", best_selected_candidate)

    summary = {
        "motor": str(motor),
        "checkpoint_glob": str(checkpoint_glob),
        "checkpoint_fingerprints": checkpoint_hashes,
        "ai_control_mode": ai_mode,
        "candidate_json": str(candidate_json),
        "candidate_json_sha256": _file_sha256(candidate_hash_path),
        "candidate_index": int(candidate_index),
        "candidate_tag": str(candidate_tag),
        "candidate_tags": list(candidate_tags_list),
        "candidate_count": int(candidate_count),
        "seeds": list(seeds_list),
        "scenarios": list(scenarios_list),
        "top_k": int(max(1, int(top_k))),
        "scan_rows": int(len(rows)),
        "skipped_count": int(len(skipped_rows)),
        "acceptance": {
            "min_avg_power_saving_pct": float(min_avg_power_saving_pct),
            "min_avg_eta_gain_pct": float(min_avg_eta_gain_pct),
            "max_avg_eta_gain_pct": float(max_avg_eta_gain_pct),
            "max_err_failures": float(max_err_failures),
            "min_start_stop_saving_pct": float(min_start_stop_saving_pct),
            "max_start_stop_saving_pct": float(max_start_stop_saving_pct),
            "max_worst_current_peak_ratio": float(max_worst_current_peak_ratio),
            "max_worst_current_mean_ratio": float(max_worst_current_mean_ratio),
            "min_avg_power_saving_pct_min_seed": min_avg_power_saving_pct_min_seed,
            "min_avg_eta_gain_pct_min_seed": min_avg_eta_gain_pct_min_seed,
            "max_err_failures_max_seed": max_err_failures_max_seed,
            "min_start_stop_saving_pct_min_seed": min_start_stop_saving_pct_min_seed,
        },
        "use_envelope_acceptance": bool(use_envelope_acceptance),
        "acceptance_envelopes": str(acceptance_envelopes_path),
        "acceptance_envelopes_sha256": _file_sha256(acceptance_envelopes_path),
        "selector_mode": "canonical_envelope_then_aggregate" if bool(use_envelope_acceptance) else "aggregate_only",
        "score_mean": _mean_score(evaluated_rows),
        "best": evaluated_rows[0] if evaluated_rows else None,
        "top_rows": evaluated_rows[: max(1, int(top_k))],
        "skipped_rows": skipped_rows,
    }
    _write_scan_state(
        path=state_path,
        signature=signature,
        evaluated_rows=evaluated_rows,
        skipped_rows=skipped_rows,
        complete=True,
        last_checkpoint_name=last_checkpoint_name,
    )
    _write_progress_summary(
        out_dir_path=out_dir_path,
        motor=str(motor),
        ai_control_mode=ai_mode,
        evaluated_rows=evaluated_rows,
        skipped_rows=skipped_rows,
        processed_count=len(checkpoints),
        total_count=len(checkpoints),
        top_k=top_k,
        use_envelope_acceptance=bool(use_envelope_acceptance),
        candidate_json=str(candidate_json),
        candidate_tag=str(candidate_tag),
        candidate_tags=candidate_tags_list,
        candidate_index=int(candidate_index),
        complete=True,
        last_checkpoint_name=evaluated_rows[0]["checkpoint_name"] if evaluated_rows else "",
    )
    _json_dump(out_dir_path / f"{motor}_checkpoint_scan_summary.json", summary)
    if evaluated_rows:
        print(f"[scan:{motor}] done. best={evaluated_rows[0]['checkpoint_name']} score={evaluated_rows[0]['score']:.3f} skipped={len(skipped_rows)}")
    else:
        print(f"[scan:{motor}] done. no evaluated checkpoints skipped={len(skipped_rows)}")
    print(f"[scan:{motor}] summary={out_dir_path / f'{motor}_checkpoint_scan_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate many checkpoints against step27 metrics with one fixed candidate.")
    parser.add_argument("--motor", required=True, choices=sorted(MOTOR_REGISTRY.keys()))
    parser.add_argument("--config-path", default="", help="Optional explicit env config path. Overrides registry config for the selected motor.")
    parser.add_argument("--checkpoint-glob", required=True, help="Checkpoint file, directory, or glob pattern (e.g. outputs/run/eval/actor_ep*.pth).")
    parser.add_argument(
        "--ai-control-mode",
        default="ai_id_ref",
        choices=["ai_id_ref", AI_ID_REF_HYBRID_MODE, "ai_current", "ai_voltage", "foc_assist", "ai_speed"],
    )
    parser.add_argument("--candidate-json", default="")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--candidate-tag", default="")
    parser.add_argument("--candidate-tags", default="", help="Comma-separated candidate tags to rank per checkpoint.")
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
    parser.add_argument("--min-avg-power-saving-pct-min-seed", type=float, default=None)
    parser.add_argument("--min-avg-eta-gain-pct-min-seed", type=float, default=None)
    parser.add_argument("--max-err-failures-max-seed", type=float, default=None)
    parser.add_argument("--min-start-stop-saving-pct-min-seed", type=float, default=None)
    parser.add_argument("--use-envelope-acceptance", action="store_true")
    parser.add_argument("--acceptance-envelopes", default=str(DEFAULT_ACCEPTANCE_ENVELOPES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume a previous interrupted scan from the out-dir state file.")
    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    scan_checkpoints(
        motor=str(args.motor),
        config_path=None if not str(args.config_path).strip() else str(args.config_path),
        checkpoint_glob=str(args.checkpoint_glob),
        ai_control_mode=str(args.ai_control_mode),
        candidate_json=str(args.candidate_json),
        candidate_index=int(args.candidate_index),
        candidate_tag=str(args.candidate_tag),
        candidate_tags=_parse_candidate_tags(args.candidate_tags),
        seeds=_parse_int_list(args.seeds),
        scenarios=_parse_csv_list(args.scenarios),
        out_dir=Path(args.out_dir),
        window_frac=float(args.window_frac),
        error_tol_rel=float(args.error_tol_rel),
        error_tol_abs=float(args.error_tol_abs),
        foc_feedback_mode=str(args.foc_feedback_mode),
        mic_feedback_mode=str(args.mic_feedback_mode),
        seed_perturbation=bool(args.seed_perturbation),
        seed_perturb_level=float(args.seed_perturb_level),
        use_total_power=bool(args.use_total_power),
        foc_disable_lut=bool(args.foc_disable_lut),
        min_avg_power_saving_pct=float(args.min_avg_power_saving_pct),
        min_avg_eta_gain_pct=float(args.min_avg_eta_gain_pct),
        max_avg_eta_gain_pct=float(args.max_avg_eta_gain_pct),
        max_err_failures=float(args.max_err_failures),
        min_start_stop_saving_pct=float(args.min_start_stop_saving_pct),
        max_start_stop_saving_pct=float(args.max_start_stop_saving_pct),
        max_worst_current_peak_ratio=float(args.max_worst_current_peak_ratio),
        max_worst_current_mean_ratio=float(args.max_worst_current_mean_ratio),
        min_avg_power_saving_pct_min_seed=args.min_avg_power_saving_pct_min_seed,
        min_avg_eta_gain_pct_min_seed=args.min_avg_eta_gain_pct_min_seed,
        max_err_failures_max_seed=args.max_err_failures_max_seed,
        min_start_stop_saving_pct_min_seed=args.min_start_stop_saving_pct_min_seed,
        use_envelope_acceptance=bool(args.use_envelope_acceptance),
        acceptance_envelopes=Path(args.acceptance_envelopes),
        top_k=int(args.top_k),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
