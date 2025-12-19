from pathlib import Path

from bench.validation import suite_conditions
from candidates.runner import load_candidate, run_candidate, validate_candidate
from config.env import create_default_env


def _write_candidate(tmp_path: Path, source: Path, policy_id: str, with_identification: bool) -> Path:
    candidate = load_candidate(source)
    candidate["policy_id"] = policy_id
    candidate["with_identification"] = bool(with_identification)
    overrides = dict(candidate.get("testsuite_overrides", {}) or {})
    overrides["log_root"] = str(tmp_path / "logs")
    overrides["run_prefix"] = policy_id
    candidate["testsuite_overrides"] = overrides
    validate_candidate(candidate)
    out_path = tmp_path / f"{policy_id}.json"
    out_path.write_text(_to_json(candidate), encoding="utf-8")
    return out_path


def _to_json(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def test_persisted_baselines(tmp_path) -> None:
    env = create_default_env()
    leaderboard_path = tmp_path / "leaderboard.json"

    pid_tuned_src = Path("candidates/examples/pid_tuned_stage11.json")
    mic_tuned_src = Path("candidates/examples/mic_tuned_stage11.json")
    pid_default_src = Path("candidates/examples/pid_default.json")
    mic_default_src = Path("candidates/examples/mic_default.json")

    pid_tuned_path = _write_candidate(tmp_path, pid_tuned_src, "pid_tuned_stage11", True)
    mic_tuned_path = _write_candidate(tmp_path, mic_tuned_src, "mic_tuned_stage11", True)
    pid_default_path = _write_candidate(tmp_path, pid_default_src, "pid_default_stage12", True)
    mic_default_path = _write_candidate(tmp_path, mic_default_src, "mic_default_stage12", True)

    pid_tuned = load_candidate(pid_tuned_path)
    mic_tuned = load_candidate(mic_tuned_path)
    ref_hash = pid_tuned["conditions_reference"]["suite_hash"]

    pid_tuned_summary = run_candidate(pid_tuned_path, env=env, leaderboard_path=leaderboard_path)
    pid_default_summary = run_candidate(pid_default_path, env=env, leaderboard_path=leaderboard_path)
    mic_tuned_summary = run_candidate(mic_tuned_path, env=env, leaderboard_path=leaderboard_path)
    mic_default_summary = run_candidate(mic_default_path, env=env, leaderboard_path=leaderboard_path)

    pid_tuned_score = float(pid_tuned_summary["summary"]["score"])
    pid_default_score = float(pid_default_summary["summary"]["score"])
    mic_tuned_score = float(mic_tuned_summary["summary"]["score"])
    mic_default_score = float(mic_default_summary["summary"]["score"])

    assert pid_tuned_score >= pid_default_score
    assert mic_tuned_score >= mic_default_score

    pid_tuned_hash = suite_conditions(pid_tuned_summary["run_dir"])["suite_hash"]
    mic_tuned_hash = suite_conditions(mic_tuned_summary["run_dir"])["suite_hash"]
    pid_default_hash = suite_conditions(pid_default_summary["run_dir"])["suite_hash"]
    mic_default_hash = suite_conditions(mic_default_summary["run_dir"])["suite_hash"]

    assert pid_tuned_hash == ref_hash
    assert mic_tuned_hash == mic_tuned["conditions_reference"]["suite_hash"]
    assert pid_default_hash == ref_hash
    assert mic_default_hash == mic_tuned["conditions_reference"]["suite_hash"]

    assert leaderboard_path.exists()
    data = leaderboard_path.read_text(encoding="utf-8")
    assert "entries" in data
