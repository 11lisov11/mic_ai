import json
from pathlib import Path

from candidates.runner import run_candidate


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "candidates" / "examples"


def _load_example(name: str) -> dict:
    with (EXAMPLES_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_candidate(tmp_path: Path, candidate: dict, name: str) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return path


def _run_example(tmp_path: Path, example_name: str, run_prefix: str, leaderboard_path: Path) -> dict:
    candidate = _load_example(example_name)
    overrides = dict(candidate.get("testsuite_overrides", {}))
    overrides.update(
        {
            "duration": 0.2,
            "dt": 0.001,
            "controller": "foc",
            "seed": 1,
            "log_root": str(tmp_path),
            "run_prefix": run_prefix,
        }
    )
    candidate["testsuite_overrides"] = overrides
    path = _write_candidate(tmp_path, candidate, example_name)
    return run_candidate(path, leaderboard_path=leaderboard_path)


def _assert_no_safety_trips(results: list) -> None:
    assert all(not result.safety_trip for result in results)


def test_pid_baseline_smoke(tmp_path) -> None:
    leaderboard_path = tmp_path / "leaderboard.json"
    summary = _run_example(tmp_path, "pid_default.json", "pid_smoke", leaderboard_path)
    assert summary["summary"]["score"] > 0.0
    _assert_no_safety_trips(summary["results"])


def test_mic_baseline_smoke(tmp_path) -> None:
    leaderboard_path = tmp_path / "leaderboard.json"
    summary = _run_example(tmp_path, "mic_default.json", "mic_smoke", leaderboard_path)
    assert summary["summary"]["score"] > 0.0
    _assert_no_safety_trips(summary["results"])


def test_leaderboard_presence(tmp_path) -> None:
    leaderboard_path = tmp_path / "leaderboard.json"
    _run_example(tmp_path, "pid_default.json", "pid_present", leaderboard_path)
    _run_example(tmp_path, "mic_default.json", "mic_present", leaderboard_path)

    with leaderboard_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("entries", [])
    ids = {entry.get("policy_id") for entry in entries}
    assert "pid_default" in ids
    assert "mic_default" in ids
