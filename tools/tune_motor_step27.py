from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ACCEPTANCE_ENVELOPES = ROOT / "config" / "acceptance_envelopes_3motors.json"

from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisorConfig  # noqa: E402
from tools.step27_pipeline import (  # noqa: E402
    MOTOR_REGISTRY,
    SeedPerturbationSettings,
    _aggregate_rows,
    _build_handcrafted_candidates,
    _candidate_to_supervisor,
    _id_ref_eval_params,
    _load_env_and_agent,
    _sample_supervisor_candidate,
    _sensorless_params,
    _simulate_rows,
    _supervisor_from_env,
    _supervisor_to_candidate,
)
from tools.common_utils import json_dump as _json_dump_shared
from tools.common_utils import parse_csv_list as _parse_csv_list_shared
from tools.common_utils import parse_int_list as _parse_int_list_shared
from tools.common_utils import write_csv as _write_csv_shared


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_list(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _json_dump(path: Path, payload: object) -> None:
    _json_dump_shared(path, payload)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_candidate_value(key: str, value: object) -> object:
    if key in {"tag", "source", "objective"}:
        return str(value)
    if key in {"update_steps", "idle_exit_boost_steps"}:
        return int(float(value))
    if key == "idle_enable":
        return _to_bool(value)
    if key == "objective_clip":
        if value in {"", None, "None", "none"}:
            return None
        return float(value)
    return float(value)


def _load_custom_candidates(path: Path, *, base: Dict[str, object]) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        raw_candidates = payload.get("candidates", [])
    elif isinstance(payload, list):
        raw_candidates = payload
    else:
        raise ValueError(f"Unsupported candidate JSON payload in {path}")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"'candidates' must be a list in {path}")

    out: List[Dict[str, object]] = []
    for idx, raw in enumerate(raw_candidates, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Candidate #{idx} in {path} is not an object")
        cand = dict(base)
        for key, value in raw.items():
            if key not in cand:
                raise ValueError(f"Unknown candidate field '{key}' in {path}")
            cand[key] = _normalize_candidate_value(key, value)
        if not str(cand.get("tag", "")).strip():
            cand["tag"] = f"custom_{idx:03d}"
        if not str(cand.get("source", "")).strip():
            cand["source"] = str(path.name)
        out.append(cand)
    return out


def _score(
    metrics: Dict[str, float],
    *,
    min_power: float,
    min_eta: float,
    max_eta: float,
    max_err: float,
    min_start_stop: float,
    max_start_stop: float,
    max_peak_ratio: float,
    max_mean_ratio: float,
    require_envelope_pass: bool = False,
    min_power_min_seed: float | None = None,
    min_eta_min_seed: float | None = None,
    max_err_max_seed: float | None = None,
    min_start_stop_min_seed: float | None = None,
) -> float:
    power = float(metrics.get("avg_power_saving_pct", 0.0))
    eta = float(metrics.get("avg_eta_gain_pct", 0.0))
    err = float(metrics.get("err_failures", 0.0))
    start_stop = float(metrics.get("start_stop_power_saving_pct", 0.0))
    peak_ratio = float(metrics.get("worst_current_peak_ratio", 1.0))
    mean_ratio = float(metrics.get("worst_current_mean_ratio", 1.0))
    power_min_seed = float(metrics.get("avg_power_saving_pct_min_seed", power))
    eta_min_seed = float(metrics.get("avg_eta_gain_pct_min_seed", eta))
    err_max_seed = float(metrics.get("err_failures_max_seed", err))
    start_stop_min_seed = float(metrics.get("start_stop_power_saving_pct_min_seed", start_stop))
    envelope_fail_count = float(metrics.get("envelope_fail_count", 0.0))
    envelope_scenario_fail_count = float(metrics.get("envelope_scenario_fail_count", 0.0))
    envelope_gap_total = float(metrics.get("envelope_gap_total", 0.0))
    envelope_err_fail_count = float(metrics.get("envelope_err_fail_count", 0.0))
    envelope_all_rows_pass = bool(metrics.get("envelope_all_rows_pass", False))

    penalty = 0.0
    if power < min_power:
        penalty += 30.0 * (min_power - power)
    if eta < min_eta:
        penalty += 20.0 * (min_eta - eta)
    if eta > max_eta:
        penalty += 6.0 * (eta - max_eta)
    if err > max_err:
        penalty += 8.0 * (err - max_err)
    if start_stop < min_start_stop:
        penalty += 12.0 * (min_start_stop - start_stop)
    if start_stop > max_start_stop:
        penalty += 4.0 * (start_stop - max_start_stop)
    penalty += 5.0 * max(0.0, peak_ratio - max_peak_ratio)
    penalty += 3.0 * max(0.0, mean_ratio - max_mean_ratio)
    if min_power_min_seed is not None and power_min_seed < float(min_power_min_seed):
        penalty += 30.0 * (float(min_power_min_seed) - power_min_seed)
    if min_eta_min_seed is not None and eta_min_seed < float(min_eta_min_seed):
        penalty += 20.0 * (float(min_eta_min_seed) - eta_min_seed)
    if max_err_max_seed is not None and err_max_seed > float(max_err_max_seed):
        penalty += 8.0 * (err_max_seed - float(max_err_max_seed))
    if min_start_stop_min_seed is not None and start_stop_min_seed < float(min_start_stop_min_seed):
        penalty += 12.0 * (float(min_start_stop_min_seed) - start_stop_min_seed)
    if require_envelope_pass and not envelope_all_rows_pass:
        penalty += 100000.0 * max(1.0, envelope_fail_count)
        penalty += 1000.0 * envelope_scenario_fail_count
        penalty += 100.0 * envelope_gap_total
        penalty += 100.0 * envelope_err_fail_count
    return float(penalty)


def _pass(
    metrics: Dict[str, float],
    *,
    min_power: float,
    min_eta: float,
    max_eta: float,
    max_err: float,
    min_start_stop: float,
    max_start_stop: float,
    max_peak_ratio: float,
    max_mean_ratio: float,
    require_envelope_pass: bool = False,
    min_power_min_seed: float | None = None,
    min_eta_min_seed: float | None = None,
    max_err_max_seed: float | None = None,
    min_start_stop_min_seed: float | None = None,
) -> bool:
    aggregate_pass = bool(
        float(metrics.get("avg_power_saving_pct", 0.0)) >= min_power
        and float(metrics.get("avg_eta_gain_pct", 0.0)) >= min_eta
        and float(metrics.get("avg_eta_gain_pct", 0.0)) <= max_eta
        and float(metrics.get("err_failures", 0.0)) <= max_err
        and float(metrics.get("start_stop_power_saving_pct", 0.0)) >= min_start_stop
        and float(metrics.get("start_stop_power_saving_pct", 0.0)) <= max_start_stop
        and float(metrics.get("worst_current_peak_ratio", 0.0)) <= max_peak_ratio
        and float(metrics.get("worst_current_mean_ratio", 0.0)) <= max_mean_ratio
    )
    worst_seed_pass = bool(
        (min_power_min_seed is None or float(metrics.get("avg_power_saving_pct_min_seed", metrics.get("avg_power_saving_pct", 0.0))) >= float(min_power_min_seed))
        and (min_eta_min_seed is None or float(metrics.get("avg_eta_gain_pct_min_seed", metrics.get("avg_eta_gain_pct", 0.0))) >= float(min_eta_min_seed))
        and (max_err_max_seed is None or float(metrics.get("err_failures_max_seed", metrics.get("err_failures", 0.0))) <= float(max_err_max_seed))
        and (min_start_stop_min_seed is None or float(metrics.get("start_stop_power_saving_pct_min_seed", metrics.get("start_stop_power_saving_pct", 0.0))) >= float(min_start_stop_min_seed))
    )
    aggregate_pass = bool(aggregate_pass and worst_seed_pass)
    if require_envelope_pass:
        return bool(aggregate_pass and bool(metrics.get("envelope_all_rows_pass", False)))
    return aggregate_pass


def _envelope_rules_for_row(
    payload: Dict[str, object],
    *,
    motor_key: str,
    scenario: str,
) -> Dict[str, object]:
    common = dict(payload.get("common", {}))
    motors = dict(payload.get("motors", {}))
    motor_cfg = dict(motors.get(str(motor_key), {}))
    rules = dict(common.get(str(scenario), {}))
    rules.update(dict(motor_cfg.get(str(scenario), {})))
    return rules


def _envelope_gap_for_check(*, metric: str, value: float, limit: float, passed: bool) -> float:
    if passed:
        return 0.0
    if metric in {"power_saving_pct", "eta_gain_pct"}:
        return float(max(0.0, limit - value))
    if metric in {"current_peak_ratio", "current_mean_ratio", "mic_mean_err"}:
        return float(max(0.0, value - limit))
    if metric == "err_ok":
        return 1.0
    return 0.0


def _evaluate_envelope_rows(
    *,
    motor_key: str,
    rows_by_seed: List[tuple[int, List[Dict[str, object]]]],
    envelopes_path: Path = DEFAULT_ACCEPTANCE_ENVELOPES,
) -> Dict[str, object]:
    path = Path(envelopes_path).resolve()
    if not path.exists():
        return {
            "envelope_all_rows_pass": False,
            "envelope_fail_count": 0,
            "envelope_scenario_fail_count": 0,
            "envelope_gap_total": 0.0,
            "envelope_power_gap": 0.0,
            "envelope_eta_gap": 0.0,
            "envelope_peak_gap": 0.0,
            "envelope_mean_gap": 0.0,
            "envelope_err_fail_count": 0,
            "envelope_summary_rows": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported envelopes payload in {path}")

    power_gap = 0.0
    eta_gap = 0.0
    peak_gap = 0.0
    mean_gap = 0.0
    err_fail_count = 0
    fail_count = 0
    per_scenario: Dict[str, Dict[str, object]] = {}

    for seed, rows in rows_by_seed:
        for row in rows:
            scenario = str(row.get("scenario", ""))
            rules = _envelope_rules_for_row(payload, motor_key=str(motor_key), scenario=scenario)
            row_pass = True
            checks = []

            def add_min(metric: str, limit_key: str) -> None:
                nonlocal row_pass, power_gap, eta_gap
                if limit_key not in rules:
                    return
                value = float(row.get(metric, 0.0))
                limit = float(rules[limit_key])
                passed = bool(value >= limit)
                checks.append((metric, value, limit, passed))
                row_pass = row_pass and passed
                gap = _envelope_gap_for_check(metric=metric, value=value, limit=limit, passed=passed)
                if metric == "power_saving_pct":
                    power_gap += gap
                elif metric == "eta_gain_pct":
                    eta_gap += gap

            def add_max(metric: str, limit_key: str) -> None:
                nonlocal row_pass, peak_gap, mean_gap
                if limit_key not in rules:
                    return
                value = float(row.get(metric, 0.0))
                limit = float(rules[limit_key])
                passed = bool(value <= limit)
                checks.append((metric, value, limit, passed))
                row_pass = row_pass and passed
                gap = _envelope_gap_for_check(metric=metric, value=value, limit=limit, passed=passed)
                if metric == "current_peak_ratio":
                    peak_gap += gap
                elif metric == "current_mean_ratio":
                    mean_gap += gap

            add_min("power_saving_pct", "power_saving_pct_min")
            add_min("eta_gain_pct", "eta_gain_pct_min")
            add_max("current_peak_ratio", "current_peak_ratio_max")
            add_max("current_mean_ratio", "current_mean_ratio_max")
            if "mic_mean_err_max" in rules:
                add_max("mic_mean_err", "mic_mean_err_max")
            if "err_ok_required" in rules:
                err_ok = bool(row.get("err_ok", False))
                required = bool(rules.get("err_ok_required", False))
                passed = bool(err_ok or (not required))
                checks.append(("err_ok", float(err_ok), float(required), passed))
                row_pass = row_pass and passed
                if not passed:
                    err_fail_count += 1

            if not row_pass:
                fail_count += 1

            item = per_scenario.setdefault(
                scenario,
                {
                    "motor": str(motor_key),
                    "scenario": scenario,
                    "samples": 0,
                    "pass_count": 0,
                    "power_saving_pct_min": float("inf"),
                    "eta_gain_pct_min": float("inf"),
                    "current_peak_ratio_max": 0.0,
                    "current_mean_ratio_max": 0.0,
                },
            )
            item["samples"] = int(item["samples"]) + 1
            item["pass_count"] = int(item["pass_count"]) + int(row_pass)
            item["power_saving_pct_min"] = min(float(item["power_saving_pct_min"]), float(row.get("power_saving_pct", 0.0)))
            item["eta_gain_pct_min"] = min(float(item["eta_gain_pct_min"]), float(row.get("eta_gain_pct", 0.0)))
            item["current_peak_ratio_max"] = max(float(item["current_peak_ratio_max"]), float(row.get("current_peak_ratio", 0.0)))
            item["current_mean_ratio_max"] = max(float(item["current_mean_ratio_max"]), float(row.get("current_mean_ratio", 0.0)))

    summary_rows: List[Dict[str, object]] = []
    scenario_fail_count = 0
    for _, item in sorted(per_scenario.items(), key=lambda kv: str(kv[0])):
        samples = int(item["samples"])
        pass_count = int(item["pass_count"])
        pass_rate = float(pass_count / max(samples, 1))
        if pass_count < samples:
            scenario_fail_count += 1
        summary_rows.append(
            {
                **item,
                "pass_rate": pass_rate,
            }
        )

    gap_total = float(power_gap + eta_gap + peak_gap + mean_gap + float(err_fail_count))
    return {
        "envelope_all_rows_pass": bool(fail_count == 0 and scenario_fail_count == 0),
        "envelope_fail_count": int(fail_count),
        "envelope_scenario_fail_count": int(scenario_fail_count),
        "envelope_gap_total": gap_total,
        "envelope_power_gap": float(power_gap),
        "envelope_eta_gap": float(eta_gap),
        "envelope_peak_gap": float(peak_gap),
        "envelope_mean_gap": float(mean_gap),
        "envelope_err_fail_count": int(err_fail_count),
        "envelope_summary_rows": summary_rows,
    }


def _eval_candidate(
    *,
    env_cfg: object,
    motor_key: str,
    agent: object,
    candidate: Dict[str, object],
    scenarios: List[str],
    seeds: List[int],
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
    seed_perturbation: SeedPerturbationSettings,
    acceptance_envelopes_path: Path = DEFAULT_ACCEPTANCE_ENVELOPES,
) -> Dict[str, float]:
    sup_cfg = _candidate_to_supervisor(candidate)
    id_ref = _id_ref_eval_params(env_cfg)
    id_ref["id_ref_alpha"] = float(candidate["id_ref_alpha"])
    id_ref["delta_id_max"] = float(candidate["delta_id_max"])
    id_ref["id_ref_gate_speed_tol_rel"] = float(candidate["id_ref_gate_speed_tol_rel"])
    id_ref["id_ref_gate_min_scale"] = float(candidate["id_ref_gate_min_scale"])
    id_ref["id_ref_gate_exponent"] = float(candidate["id_ref_gate_exponent"])
    sensorless = _sensorless_params(env_cfg)

    per_seed: List[Dict[str, float]] = []
    rows_by_seed: List[tuple[int, List[Dict[str, object]]]] = []
    for seed in seeds:
        rows = _simulate_rows(
            env_cfg=env_cfg,
            motor_key=str(motor_key),
            agent=agent,
            scenarios=scenarios,
            seed=int(seed),
            window_frac=float(window_frac),
            error_tol_rel=float(error_tol_rel),
            error_tol_abs=float(error_tol_abs),
            use_total_power=bool(use_total_power),
            foc_feedback_mode=str(foc_feedback_mode),
            mic_feedback_mode=str(mic_feedback_mode),
            controller="MIC",
            id_ref_params=id_ref,
            supervisor_cfg=sup_cfg,
            sensorless=sensorless,
            seed_perturbation=seed_perturbation,
            mic_mode="ai",
            mic_rule_params=None,
        )
        rows_by_seed.append((int(seed), rows))
        per_seed.append(_aggregate_rows(rows))

    out = {
        "avg_power_saving_pct": _mean([float(x["avg_power_saving_pct"]) for x in per_seed]),
        "avg_eta_gain_pct": _mean([float(x["avg_eta_gain_pct"]) for x in per_seed]),
        "err_failures": _mean([float(x["err_failures"]) for x in per_seed]),
        "start_stop_power_saving_pct": _mean([float(x["start_stop_power_saving_pct"]) for x in per_seed]),
        "worst_current_peak_ratio": max(float(x["worst_current_peak_ratio"]) for x in per_seed),
        "worst_current_mean_ratio": max(float(x["worst_current_mean_ratio"]) for x in per_seed),
        "avg_power_saving_pct_min_seed": min(float(x["avg_power_saving_pct"]) for x in per_seed),
        "avg_eta_gain_pct_min_seed": min(float(x["avg_eta_gain_pct"]) for x in per_seed),
        "err_failures_max_seed": max(float(x["err_failures"]) for x in per_seed),
        "start_stop_power_saving_pct_min_seed": min(float(x["start_stop_power_saving_pct"]) for x in per_seed),
    }
    out.update(
        _evaluate_envelope_rows(
            motor_key=str(motor_key),
            rows_by_seed=rows_by_seed,
            envelopes_path=Path(acceptance_envelopes_path).resolve(),
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune AI MIC supervisor/id_ref params for selected motor using step27 metrics.")
    parser.add_argument("--motor", required=True, choices=sorted(MOTOR_REGISTRY.keys()))
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--out-dir", default="outputs/motor_tuning_step27")
    parser.add_argument("--stage1-trials", type=int, default=40)
    parser.add_argument("--stage2-topk", type=int, default=8)
    parser.add_argument("--stage1-seed", type=int, default=101)
    parser.add_argument("--stage2-seed", type=int, default=101)
    parser.add_argument("--search-seed", type=int, default=26027)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--candidate-json", default="")
    parser.add_argument("--candidate-json-mode", default="append", choices=["append", "replace"])
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--sample-profile", default="global", choices=["global", "local_safe"])
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
    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    scenarios = _parse_csv_list(args.scenarios)
    if not seeds:
        raise ValueError("Empty seeds list")
    if not scenarios:
        raise ValueError("Empty scenarios list")

    motor = str(args.motor)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_seed = int(args.stage1_seed)
    stage2_seed = int(args.stage2_seed)

    env_cfg, agent, ckpt = _load_env_and_agent(
        MOTOR_REGISTRY[motor].config_path,
        foc_disable_lut=bool(args.foc_disable_lut),
        require_agent=True,
        motor_key=str(motor),
        checkpoint_registry_path=str(args.checkpoint_registry),
    )
    if agent is None:
        raise RuntimeError("AI agent was not loaded.")

    base_sup = _supervisor_from_env(env_cfg)
    if base_sup is None:
        base_sup = AiIdRefSupervisorConfig(enabled=True)

    base_cand = _supervisor_to_candidate(base_sup, tag="baseline", source="config")
    base_id_ref = _id_ref_eval_params(env_cfg)
    base_cand.update(
        {
            "id_ref_alpha": float(base_id_ref["id_ref_alpha"]),
            "delta_id_max": float(base_id_ref["delta_id_max"]),
            "id_ref_gate_speed_tol_rel": float(base_id_ref["id_ref_gate_speed_tol_rel"] or 0.05),
            "id_ref_gate_min_scale": float(base_id_ref["id_ref_gate_min_scale"]),
            "id_ref_gate_exponent": float(base_id_ref["id_ref_gate_exponent"]),
        }
    )

    custom_candidates: List[Dict[str, object]] = []
    candidate_json = str(args.candidate_json).strip()
    if candidate_json:
        custom_candidates = _load_custom_candidates(Path(candidate_json).expanduser().resolve(), base=base_cand)

    candidates: List[Dict[str, object]] = []
    if str(args.candidate_json_mode) == "append":
        rng = random.Random(int(args.search_seed))
        candidates.append(dict(base_cand))
        candidates.extend(_build_handcrafted_candidates(base_cand))
        for i in range(int(args.stage1_trials)):
            candidates.append(
                _sample_supervisor_candidate(
                    rng,
                    idx=i + 1,
                    base=base_cand,
                    profile=str(args.sample_profile),
                )
            )
        candidates.extend(custom_candidates)
    else:
        candidates.extend(custom_candidates)
    if not candidates:
        raise ValueError("No candidates to evaluate. Provide --candidate-json or keep --candidate-json-mode=append.")

    seed_perturb = SeedPerturbationSettings(
        enabled=bool(args.seed_perturbation),
        level=float(max(0.0, float(args.seed_perturb_level))),
    )

    stage1_rows: List[Dict[str, object]] = []
    for i, cand in enumerate(candidates, start=1):
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=motor,
            agent=agent,
            candidate=cand,
            scenarios=scenarios,
            seeds=[stage1_seed],
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            seed_perturbation=seed_perturb,
            acceptance_envelopes_path=Path(args.acceptance_envelopes),
        )
        row = {**cand, **metrics}
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
            require_envelope_pass=bool(args.use_envelope_acceptance),
            min_power_min_seed=args.min_avg_power_saving_pct_min_seed,
            min_eta_min_seed=args.min_avg_eta_gain_pct_min_seed,
            max_err_max_seed=args.max_err_failures_max_seed,
            min_start_stop_min_seed=args.min_start_stop_saving_pct_min_seed,
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
            require_envelope_pass=bool(args.use_envelope_acceptance),
            min_power_min_seed=args.min_avg_power_saving_pct_min_seed,
            min_eta_min_seed=args.min_avg_eta_gain_pct_min_seed,
            max_err_max_seed=args.max_err_failures_max_seed,
            min_start_stop_min_seed=args.min_start_stop_saving_pct_min_seed,
        )
        stage1_rows.append(row)
        print(
            f"[tune:{motor}][stage1] {i}/{len(candidates)} tag={row['tag']} "
            f"power={float(row['avg_power_saving_pct']):.3f}% eta={float(row['avg_eta_gain_pct']):.3f}% "
            f"start_stop={float(row['start_stop_power_saving_pct']):.3f}% err={float(row['err_failures']):.1f}"
        )

    stage1_rows.sort(key=lambda r: float(r["score"]))
    top = stage1_rows[: max(1, int(args.stage2_topk))]

    stage2_rows: List[Dict[str, object]] = []
    for i, cand in enumerate(top, start=1):
        metrics = _eval_candidate(
            env_cfg=env_cfg,
            motor_key=motor,
            agent=agent,
            candidate=cand,
            scenarios=scenarios,
            seeds=seeds,
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            seed_perturbation=seed_perturb,
            acceptance_envelopes_path=Path(args.acceptance_envelopes),
        )
        row = {**cand, **metrics}
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
            require_envelope_pass=bool(args.use_envelope_acceptance),
            min_power_min_seed=args.min_avg_power_saving_pct_min_seed,
            min_eta_min_seed=args.min_avg_eta_gain_pct_min_seed,
            max_err_max_seed=args.max_err_failures_max_seed,
            min_start_stop_min_seed=args.min_start_stop_saving_pct_min_seed,
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
            require_envelope_pass=bool(args.use_envelope_acceptance),
            min_power_min_seed=args.min_avg_power_saving_pct_min_seed,
            min_eta_min_seed=args.min_avg_eta_gain_pct_min_seed,
            max_err_max_seed=args.max_err_failures_max_seed,
            min_start_stop_min_seed=args.min_start_stop_saving_pct_min_seed,
        )
        stage2_rows.append(row)
        print(
            f"[tune:{motor}][stage2] {i}/{len(top)} tag={row['tag']} "
            f"power={float(row['avg_power_saving_pct']):.3f}% eta={float(row['avg_eta_gain_pct']):.3f}% "
            f"start_stop={float(row['start_stop_power_saving_pct']):.3f}% err={float(row['err_failures']):.1f}"
        )

    stage2_rows.sort(key=lambda r: float(r["score"]))
    best = stage2_rows[0] if stage2_rows else stage1_rows[0]

    _write_csv(out_dir / f"{motor}_stage1_rank.csv", stage1_rows)
    _write_csv(out_dir / f"{motor}_stage2_rank.csv", stage2_rows)
    summary = {
        "motor": motor,
        "checkpoint": "" if ckpt is None else str(ckpt),
        "config_path": MOTOR_REGISTRY[motor].config_path,
        "seeds": seeds,
        "scenarios": scenarios,
        "stage1_trials": int(args.stage1_trials),
        "stage2_topk": int(args.stage2_topk),
        "stage1_seed": stage1_seed,
        "stage2_seed": stage2_seed,
        "search_seed": int(args.search_seed),
        "candidate_json": candidate_json,
        "candidate_json_mode": str(args.candidate_json_mode),
        "custom_candidate_count": int(len(custom_candidates)),
        "acceptance": {
            "min_avg_power_saving_pct": float(args.min_avg_power_saving_pct),
            "min_avg_eta_gain_pct": float(args.min_avg_eta_gain_pct),
            "max_avg_eta_gain_pct": float(args.max_avg_eta_gain_pct),
            "max_err_failures": float(args.max_err_failures),
            "min_start_stop_saving_pct": float(args.min_start_stop_saving_pct),
            "max_start_stop_saving_pct": float(args.max_start_stop_saving_pct),
            "max_worst_current_peak_ratio": float(args.max_worst_current_peak_ratio),
            "max_worst_current_mean_ratio": float(args.max_worst_current_mean_ratio),
            "min_avg_power_saving_pct_min_seed": args.min_avg_power_saving_pct_min_seed,
            "min_avg_eta_gain_pct_min_seed": args.min_avg_eta_gain_pct_min_seed,
            "max_err_failures_max_seed": args.max_err_failures_max_seed,
            "min_start_stop_saving_pct_min_seed": args.min_start_stop_saving_pct_min_seed,
        },
        "use_envelope_acceptance": bool(args.use_envelope_acceptance),
        "acceptance_envelopes": str(Path(args.acceptance_envelopes).resolve()),
        "sample_profile": str(args.sample_profile),
        "best": best,
        "top_stage2": stage2_rows,
    }
    _json_dump(out_dir / f"{motor}_tuning_summary.json", summary)
    _json_dump(out_dir / f"{motor}_best_candidate.json", best)
    print(f"[tune:{motor}] done. best={best.get('tag')} score={float(best.get('score', 0.0)):.3f}")
    print(f"[tune:{motor}] summary={out_dir / f'{motor}_tuning_summary.json'}")


if __name__ == "__main__":
    main()
