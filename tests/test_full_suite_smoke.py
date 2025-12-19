import json
from pathlib import Path

import numpy as np

from bench.identification import IdentificationConfig
from bench.scoring import score_testsuite
from bench.testsuite import TestsuiteConfig, run_testsuite
from config.env import create_default_env


def _run_suite(
    tmp_path: Path,
    controller: str,
    run_prefix: str,
    seed: int,
) -> tuple[Path, float]:
    env = create_default_env()
    id_cfg = IdentificationConfig(
        env=env,
        dt=0.001,
        duration=0.5,
        pulse_start=0.1,
        pulse_end=0.4,
    )
    suite_cfg = TestsuiteConfig(
        duration=1.0,
        dt=0.001,
        controller=controller,
        seed=seed,
        log_root=str(tmp_path),
        run_prefix=run_prefix,
        with_identification=True,
        id_config=id_cfg,
    )
    result = run_testsuite(env=env, suite_cfg=suite_cfg)
    leaderboard_path = tmp_path / f"leaderboard_{run_prefix}.json"
    summary = score_testsuite(result.run_dir, policy_id=run_prefix, leaderboard_path=leaderboard_path)
    return result.run_dir, float(summary["score"])


def _load_case_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("E")])


def test_full_suite_smoke(tmp_path) -> None:
    run_dir_foc, score_foc = _run_suite(tmp_path, "foc", "foc_smoke", seed=7)
    cases = _load_case_dirs(run_dir_foc)
    assert cases

    for case_dir in cases:
        meta_path = case_dir / "run_meta.json"
        assert meta_path.exists()
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        identification = meta.get("identification")
        assert identification is not None
        assert 0.0 <= identification["aging"] <= 1.0
        assert "score" in meta

        npz_path = case_dir / "timeseries.npz"
        data = np.load(npz_path)
        assert not np.any(data["flag_safety"] > 0.5)

    run_dir_foc_2, score_foc_2 = _run_suite(tmp_path, "foc", "foc_smoke_repeat", seed=7)
    assert abs(score_foc - score_foc_2) < 1e-9

    _, score_vf = _run_suite(tmp_path, "scalar", "vf_smoke", seed=7)
    assert score_foc >= score_vf
