from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# ensure project root importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.validation import suite_conditions
from candidates.runner import load_candidate, load_env_config, run_candidate, validate_candidate
from config.env import create_default_env


@dataclass(frozen=True)
class SearchSpace:
    """Parameter ranges for random search."""

    kp_min: float
    kp_max: float
    ki_min: float
    ki_max: float
    integrator_min: float | None = None
    integrator_max: float | None = None
    vq_max_min: float | None = None
    vq_max_max: float | None = None
    k_age_min: float | None = None
    k_age_max: float | None = None
    tau_action_min: float | None = None
    tau_action_max: float | None = None
    dvq_max_min: float | None = None
    dvq_max_max: float | None = None


DEFAULT_SPACES: dict[str, SearchSpace] = {
    "pid_speed": SearchSpace(
        kp_min=0.1,
        kp_max=5.0,
        ki_min=0.0,
        ki_max=2.0,
        integrator_min=0.5,
        integrator_max=4.0,
    ),
    "mic_rule": SearchSpace(
        kp_min=0.1,
        kp_max=5.0,
        ki_min=0.0,
        ki_max=2.0,
        integrator_min=0.5,
        integrator_max=4.0,
        vq_max_min=120.0,
        vq_max_max=240.0,
        k_age_min=-0.5,
        k_age_max=0.5,
        tau_action_min=0.0,
        tau_action_max=0.05,
        dvq_max_min=500.0,
        dvq_max_max=3000.0,
    ),
}


@dataclass
class SearchResult:
    baseline_scores: dict[int, float]
    baseline_hashes: dict[int, str]
    best_score: float
    best_params: dict[str, float]
    best_improvement_rel: float
    best_improvement_abs: float
    best_seed_wins: int
    all_scores: list[float]
    iterations: list[dict[str, object]]


def run_search(
    policy: str,
    iters: int,
    base_candidate_path: str | Path,
    seed: int,
    with_identification: bool,
    leaderboard_path: str | Path = "leaderboard.json",
    log_root: str | Path = "logs",
    out_dir: str | Path = "search_runs",
    env_config_path: str | None = None,
    replicates: int = 3,
    min_rel_improve: float = 0.05,
    min_abs_improve: float = 2.0,
) -> SearchResult:
    policy = policy.strip().lower()
    if policy not in DEFAULT_SPACES:
        raise ValueError(f"Unsupported policy: {policy!r}")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    env = load_env_config(env_config_path) if env_config_path else create_default_env()
    base_candidate_path = Path(base_candidate_path)
    base_candidate = load_candidate(base_candidate_path)
    validate_candidate(base_candidate)

    if str(base_candidate.get("policy_type", "")).lower() != policy:
        raise ValueError("base-candidate policy_type does not match --policy")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) / f"random_search_{policy}_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "search_log.jsonl"

    seeds = [seed + idx for idx in range(replicates)]
    baseline_scores, baseline_hashes = _run_baseline(
        base_candidate,
        base_candidate_path,
        env,
        seeds,
        with_identification,
        leaderboard_path,
        log_root,
        candidates_dir,
    )

    baseline_mean = _mean_score(list(baseline_scores.values()))
    best_score = baseline_mean
    best_params: dict[str, float] = {}
    best_seed_wins = 0
    all_scores: list[float] = []
    iterations: list[dict[str, object]] = []

    rng = random.Random(seed)
    for idx in range(1, iters + 1):
        params = _sample_params(policy, base_candidate.get("params", {}), rng)
        scores_by_seed, hash_ok = _run_candidate_iteration(
            policy,
            idx,
            params,
            base_candidate,
            env,
            seeds,
            with_identification,
            leaderboard_path,
            log_root,
            candidates_dir,
            baseline_hashes,
        )
        if not scores_by_seed:
            entry = {
                "iteration": idx,
                "params": params,
                "score_mean": None,
                "hash_ok": False,
                "seed_scores": {},
                "best_so_far": best_score,
            }
            iterations.append(entry)
            _append_log(log_path, entry)
            continue

        score_mean = _mean_score(list(scores_by_seed.values()))
        all_scores.append(score_mean)
        seed_wins = _count_seed_wins(scores_by_seed, baseline_scores, min_rel_improve, min_abs_improve)
        stable = seed_wins >= _stable_threshold(len(scores_by_seed))
        if hash_ok and stable and score_mean > best_score:
            best_score = score_mean
            best_params = params
            best_seed_wins = seed_wins
        entry = {
            "iteration": idx,
            "params": params,
            "score_mean": score_mean,
            "seed_scores": scores_by_seed,
            "hash_ok": hash_ok,
            "stable": stable,
            "seed_wins": seed_wins,
            "best_so_far": best_score,
        }
        iterations.append(entry)
        _append_log(log_path, entry)

    best_improve_rel = (best_score - baseline_mean) / max(baseline_mean, 1e-9)
    best_improve_abs = best_score - baseline_mean
    summary = {
        "policy": policy,
        "baseline_mean": baseline_mean,
        "baseline_scores": baseline_scores,
        "baseline_hashes": baseline_hashes,
        "best_score": best_score,
        "best_params": best_params,
        "best_improvement_rel": best_improve_rel,
        "best_improvement_abs": best_improve_abs,
        "best_seed_wins": best_seed_wins,
        "min_rel_improve": min_rel_improve,
        "min_abs_improve": min_abs_improve,
        "iters": iters,
        "replicates": replicates,
    }
    _write_json(out_dir / "search_summary.json", summary)

    return SearchResult(
        baseline_scores=baseline_scores,
        baseline_hashes=baseline_hashes,
        best_score=best_score,
        best_params=best_params,
        best_improvement_rel=best_improve_rel,
        best_improvement_abs=best_improve_abs,
        best_seed_wins=best_seed_wins,
        all_scores=all_scores,
        iterations=iterations,
    )


def _run_baseline(
    base_candidate: Mapping[str, object],
    base_candidate_path: Path,
    env: object,
    seeds: list[int],
    with_identification: bool,
    leaderboard_path: str | Path,
    log_root: str | Path,
    candidates_dir: Path,
) -> tuple[dict[int, float], dict[int, str]]:
    scores: dict[int, float] = {}
    hashes: dict[int, str] = {}
    for seed in seeds:
        candidate = _build_candidate(
            base_candidate,
            policy_id=f"{base_candidate.get('policy_id', base_candidate_path.stem)}_baseline_s{seed}",
            seed=seed,
            log_root=log_root,
            with_identification=with_identification,
            run_prefix=f"baseline_{seed}",
        )
        candidate_path = candidates_dir / f"baseline_s{seed}.json"
        _write_json(candidate_path, candidate)
        summary = run_candidate(
            candidate_path,
            env=env,
            leaderboard_path=leaderboard_path,
            policy_id_override=candidate["policy_id"],
            with_identification_override=with_identification,
        )
        score = float(summary["summary"]["score"])
        scores[seed] = score
        run_dir = Path(summary["run_dir"])
        hashes[seed] = str(suite_conditions(run_dir)["suite_hash"])
    return scores, hashes


def _run_candidate_iteration(
    policy: str,
    iteration: int,
    params: dict[str, float],
    base_candidate: Mapping[str, object],
    env: object,
    seeds: list[int],
    with_identification: bool,
    leaderboard_path: str | Path,
    log_root: str | Path,
    candidates_dir: Path,
    baseline_hashes: Mapping[int, str],
) -> tuple[dict[int, float], bool]:
    scores: dict[int, float] = {}
    hash_ok = True
    for seed in seeds:
        policy_id = f"{policy}_search_{iteration}_s{seed}"
        candidate = _build_candidate(
            base_candidate,
            policy_id=policy_id,
            seed=seed,
            log_root=log_root,
            with_identification=with_identification,
            run_prefix=f"iter_{iteration}_s{seed}",
            params_override=params,
        )
        candidate_path = candidates_dir / f"iter_{iteration}_s{seed}.json"
        _write_json(candidate_path, candidate)
        summary = run_candidate(
            candidate_path,
            env=env,
            leaderboard_path=leaderboard_path,
            policy_id_override=policy_id,
            with_identification_override=with_identification,
        )
        score = float(summary["summary"]["score"])
        scores[seed] = score
        run_dir = Path(summary["run_dir"])
        suite_hash = str(suite_conditions(run_dir)["suite_hash"])
        baseline_hash = baseline_hashes.get(seed)
        if baseline_hash is None or suite_hash != baseline_hash:
            hash_ok = False
    return scores, hash_ok


def _build_candidate(
    base_candidate: Mapping[str, object],
    policy_id: str,
    seed: int,
    log_root: str | Path,
    with_identification: bool,
    run_prefix: str,
    params_override: Mapping[str, float] | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {}
    for key in ("policy_id", "policy_type", "params", "limits", "testsuite_overrides", "with_identification", "id_config"):
        if key in base_candidate:
            candidate[key] = base_candidate[key]

    candidate["policy_id"] = policy_id
    candidate["with_identification"] = bool(with_identification)

    overrides = dict(candidate.get("testsuite_overrides", {}) or {})
    overrides["seed"] = int(seed)
    overrides["log_root"] = str(log_root)
    overrides["run_prefix"] = run_prefix
    candidate["testsuite_overrides"] = overrides

    params = dict(candidate.get("params", {}) or {})
    if params_override:
        params.update(params_override)
    candidate["params"] = params

    validate_candidate(candidate)
    return candidate


def _sample_params(policy: str, base_params: Mapping[str, object], rng: random.Random) -> dict[str, float]:
    space = DEFAULT_SPACES[policy]
    params = dict(base_params)
    params["kp"] = rng.uniform(space.kp_min, space.kp_max)
    params["ki"] = rng.uniform(space.ki_min, space.ki_max)
    if space.integrator_min is not None and space.integrator_max is not None:
        params["integrator_limit"] = rng.uniform(space.integrator_min, space.integrator_max)
    if policy == "mic_rule":
        if space.vq_max_min is not None and space.vq_max_max is not None:
            params["vq_max"] = rng.uniform(space.vq_max_min, space.vq_max_max)
        params["k_age"] = rng.uniform(space.k_age_min, space.k_age_max)
        params["tau_action"] = rng.uniform(space.tau_action_min, space.tau_action_max)
        params["dvq_max"] = rng.uniform(space.dvq_max_min, space.dvq_max_max)
    return {key: float(value) for key, value in params.items()}


def _mean_score(values: list[float]) -> float:
    if not values:
        return float("-inf")
    return float(sum(values) / len(values))


def _count_seed_wins(
    scores: Mapping[int, float],
    baseline_scores: Mapping[int, float],
    min_rel_improve: float,
    min_abs_improve: float,
) -> int:
    wins = 0
    for seed, score in scores.items():
        base = baseline_scores.get(seed)
        if base is None:
            continue
        if score >= base * (1.0 + min_rel_improve) or score >= base + min_abs_improve:
            wins += 1
    return wins


def _stable_threshold(n_seeds: int) -> int:
    return max(1, (n_seeds // 2) + (n_seeds % 2))


def _append_log(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for MIC_AI candidate parameters")
    parser.add_argument("--policy", choices=["pid_speed", "mic_rule"], required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--base-candidate", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-identification", action="store_true")
    parser.add_argument("--env-config", default=None, help="env config .py (expects ENV)")
    parser.add_argument("--leaderboard", default="leaderboard.json", help="leaderboard path")
    parser.add_argument("--log-root", default="logs", help="logs root directory")
    parser.add_argument("--out-dir", default="search_runs", help="search output directory")
    parser.add_argument("--replicates", type=int, default=3, help="number of seeds for stability")
    parser.add_argument("--min-rel-improve", type=float, default=0.05, help="relative improvement threshold")
    parser.add_argument("--min-abs-improve", type=float, default=2.0, help="absolute improvement threshold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_search(
        policy=args.policy,
        iters=args.iters,
        base_candidate_path=args.base_candidate,
        seed=args.seed,
        with_identification=args.with_identification,
        leaderboard_path=args.leaderboard,
        log_root=args.log_root,
        out_dir=args.out_dir,
        env_config_path=args.env_config,
        replicates=args.replicates,
        min_rel_improve=args.min_rel_improve,
        min_abs_improve=args.min_abs_improve,
    )
    baseline_mean = _mean_score(list(result.baseline_scores.values()))
    print(f"Baseline mean: {baseline_mean:.4f}")
    print(f"Best mean: {result.best_score:.4f}")
    print(f"Improvement rel: {result.best_improvement_rel * 100.0:.2f}%")
    print(f"Improvement abs: {result.best_improvement_abs:.4f}")
    print(f"Best params: {json.dumps(result.best_params, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
