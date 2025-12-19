import json
from pathlib import Path

import numpy as np

from bench.scoring import score_testsuite


def _write_run(root: Path, name: str, safety_trip: bool = False) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    t = np.array([0.0, 0.01, 0.02], dtype=float)
    omega_ref = np.array([10.0, 10.0, 10.0], dtype=float)
    omega_m = np.array([0.0, 5.0, 9.0], dtype=float)
    i_a = np.array([0.1, 0.2, 0.1], dtype=float)
    i_b = np.array([-0.1, -0.2, -0.1], dtype=float)
    i_c = np.array([0.0, 0.0, 0.0], dtype=float)
    v_d = np.array([1.0, 1.0, 1.0], dtype=float)
    v_q = np.array([0.5, 0.5, 0.5], dtype=float)
    flag = np.array([0.0, 1.0, 1.0] if safety_trip else [0.0, 0.0, 0.0], dtype=float)

    npz_path = run_dir / "timeseries.npz"
    np.savez(
        npz_path,
        t=t,
        omega_m=omega_m,
        omega_ref=omega_ref,
        i_a=i_a,
        i_b=i_b,
        i_c=i_c,
        v_d=v_d,
        v_q=v_q,
        flag_safety=flag,
    )

    meta = {"npz_path": str(npz_path), "config": {"dt": 0.01}, "metrics": {}}
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def test_score_testsuite_updates_leaderboard(tmp_path) -> None:
    _write_run(tmp_path, "E1", safety_trip=False)
    _write_run(tmp_path, "E2", safety_trip=True)

    leaderboard_path = tmp_path / "leaderboard.json"
    summary = score_testsuite(tmp_path, policy_id="foc_baseline", leaderboard_path=leaderboard_path)

    assert summary["leaderboard_path"] == leaderboard_path
    assert leaderboard_path.exists()
    with leaderboard_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "entries" in data
    assert len(data["entries"]) == 1
    entry = data["entries"][0]
    assert entry["policy_id"] == "foc_baseline"
    assert entry["score"] > 0.0

    meta_path = tmp_path / "E1" / "run_meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert "metrics" in meta
    assert "score" in meta
