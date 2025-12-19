from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Mapping

# ensure project root importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.identification import IdentificationConfig
from bench.scoring import score_testsuite
from bench.testsuite import TestsuiteConfig, run_testsuite
from candidates.policies import create_policy
from config.env import EnvConfig, create_default_env
from sim.safety import SafetyLimits, SafetySupervisor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIC_AI candidate policies")
    parser.add_argument("--candidate", help="path to candidate JSON")
    parser.add_argument("--batch-dir", help="directory with candidate JSON files")
    parser.add_argument("--env-config", help="path to env config .py (expects ENV)")
    parser.add_argument("--leaderboard", default="leaderboard.json", help="leaderboard path")
    parser.add_argument("--policy-id", default=None, help="override policy_id")
    parser.add_argument("--with-identification", action="store_true", help="enable identification step")
    return parser.parse_args(argv)


def run_candidate(
    candidate_path: str | Path,
    env: EnvConfig | None = None,
    leaderboard_path: str | Path = "leaderboard.json",
    policy_id_override: str | None = None,
    with_identification_override: bool | None = None,
) -> dict[str, object]:
    candidate_path = Path(candidate_path)
    candidate = load_candidate(candidate_path)
    validate_candidate(candidate)

    env = env or create_default_env()
    testsuite_overrides = candidate.get("testsuite_overrides", {})
    id_config = _build_id_config(candidate.get("id_config", {}), env)
    with_identification = bool(candidate.get("with_identification", False))
    if with_identification_override:
        with_identification = True
    suite_cfg = _build_testsuite_config(
        testsuite_overrides,
        candidate_path,
        with_identification=with_identification,
        id_config=id_config,
    )
    if suite_cfg.dt is None:
        suite_cfg = replace(suite_cfg, dt=env.sim.dt)
    policy = create_policy(candidate, env, suite_cfg.dt)

    safety = _build_safety(candidate.get("limits", {}))
    suite_result = run_testsuite(env=env, suite_cfg=suite_cfg, policy=policy, safety=safety)

    policy_id = policy_id_override or str(candidate.get("policy_id") or candidate_path.stem)
    summary = score_testsuite(suite_result.run_dir, policy_id, leaderboard_path)
    return {
        "policy_id": policy_id,
        "run_dir": str(suite_result.run_dir),
        "results": suite_result.results,
        "summary": summary,
    }


def run_batch(
    batch_dir: str | Path,
    env: EnvConfig | None = None,
    leaderboard_path: str | Path = "leaderboard.json",
) -> list[dict[str, object]]:
    batch_dir = Path(batch_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f"batch_dir not found: {batch_dir}")
    summaries = []
    for candidate_path in sorted(batch_dir.glob("*.json")):
        if candidate_path.name == "schema.json":
            continue
        summaries.append(run_candidate(candidate_path, env=env, leaderboard_path=leaderboard_path))
    return summaries


def load_candidate(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_candidate(candidate: Mapping[str, object]) -> None:
    errors = []
    if not isinstance(candidate, Mapping):
        errors.append("candidate must be a JSON object")
    if "policy_type" not in candidate:
        errors.append("policy_type is required")
    if "params" not in candidate:
        errors.append("params is required")
    if "params" in candidate and not isinstance(candidate["params"], Mapping):
        errors.append("params must be an object")
    for key in ("limits", "testsuite_overrides", "id_config"):
        if key in candidate and not isinstance(candidate[key], Mapping):
            errors.append(f"{key} must be an object")
    if "with_identification" in candidate and not isinstance(candidate["with_identification"], bool):
        errors.append("with_identification must be a boolean")
    if errors:
        raise ValueError("; ".join(errors))


def load_env_config(path: str | Path) -> EnvConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"env config not found: {path}")
    spec = spec_from_file_location("candidate_env", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load env config: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    env = getattr(module, "ENV", None)
    if env is None:
        raise ValueError(f"ENV not found in {path}")
    return env


def _build_testsuite_config(
    overrides: Mapping[str, object],
    candidate_path: Path,
    with_identification: bool,
    id_config: IdentificationConfig | None,
) -> TestsuiteConfig:
    run_prefix = overrides.get("run_prefix", candidate_path.stem)
    identify_once = bool(overrides.get("identify_once_per_suite", True))
    identify_per_case = bool(overrides.get("identify_per_case", False))
    return TestsuiteConfig(
        duration=float(overrides.get("duration", 60.0)),
        dt=float(overrides["dt"]) if "dt" in overrides else None,
        controller=str(overrides.get("controller", "foc")),
        seed=int(overrides.get("seed", 1)),
        log_root=str(overrides.get("log_root", "logs")),
        run_prefix=str(run_prefix),
        with_identification=with_identification,
        identify_once_per_suite=identify_once,
        identify_per_case=identify_per_case,
        id_config=id_config,
    )


def _build_safety(limits: Mapping[str, object]) -> SafetySupervisor | None:
    if not limits:
        return None
    return SafetySupervisor(
        SafetyLimits(
            i_inst_max=_get_limit(limits, "i_inst_max"),
            i_rms_max=_get_limit(limits, "i_rms_max"),
            omega_max=_get_limit(limits, "omega_max"),
            vdc_max=_get_limit(limits, "vdc_max"),
        )
    )


def _get_limit(limits: Mapping[str, object], key: str) -> float | None:
    if key not in limits:
        return None
    return float(limits[key])


def _build_id_config(overrides: Mapping[str, object], env: EnvConfig) -> IdentificationConfig | None:
    if not overrides:
        return None
    numeric_keys = {
        "dt",
        "duration",
        "pulse_start",
        "pulse_end",
        "pulse_v_d",
        "pulse_v_q",
        "rs_nom",
        "rs_max",
        "current_ratio",
        "i_inst_max",
        "i_rms_max",
        "omega_max",
        "vdc_max",
    }
    kwargs: dict[str, object] = {"env": env}
    for key, value in overrides.items():
        if key in numeric_keys:
            kwargs[key] = float(value)
        elif key in ("log_root", "run_name"):
            kwargs[key] = str(value)
        elif key == "seed":
            kwargs[key] = int(value)
    return IdentificationConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = load_env_config(args.env_config) if args.env_config else create_default_env()

    if args.batch_dir:
        summaries = run_batch(args.batch_dir, env=env, leaderboard_path=args.leaderboard)
        print(f"Processed {len(summaries)} candidates")
        return 0
    if not args.candidate:
        raise SystemExit("Use --candidate or --batch-dir")

    summary = run_candidate(
        args.candidate,
        env=env,
        leaderboard_path=args.leaderboard,
        policy_id_override=args.policy_id,
        with_identification_override=args.with_identification,
    )
    score = summary["summary"]["score"]
    print(f"Score: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
