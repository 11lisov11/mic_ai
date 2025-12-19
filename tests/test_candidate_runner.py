import json
from pathlib import Path

from candidates.runner import run_candidate


def test_candidate_runner_end_to_end(tmp_path) -> None:
    candidate_path = tmp_path / "stub_candidate.json"
    candidate = {
        "policy_id": "stub_candidate",
        "policy_type": "nn_stub",
        "params": {"v_d": 0.0, "v_q": 0.0},
        "limits": {},
        "testsuite_overrides": {
            "duration": 0.02,
            "dt": 0.01,
            "controller": "scalar",
            "log_root": str(tmp_path),
            "run_prefix": "stub_run",
        },
    }
    with candidate_path.open("w", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")

    leaderboard_path = tmp_path / "leaderboard.json"
    summary = run_candidate(candidate_path, leaderboard_path=leaderboard_path)

    assert leaderboard_path.exists()
    assert summary["summary"]["score"] > 0.0
    run_dir = Path(summary["run_dir"])
    assert run_dir.exists()
