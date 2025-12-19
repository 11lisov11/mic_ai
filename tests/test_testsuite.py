from bench.testsuite import TestsuiteConfig, run_testsuite
from config.env import create_default_env


def test_testsuite_runs_all_cases(tmp_path) -> None:
    env = create_default_env()
    suite_cfg = TestsuiteConfig(
        duration=0.02,
        dt=0.01,
        controller="scalar",
        seed=7,
        log_root=str(tmp_path),
        run_prefix="unit_suite",
    )
    result = run_testsuite(env=env, suite_cfg=suite_cfg)

    assert result.run_dir.exists()
    assert len(result.results) == 7
    for case_result in result.results:
        assert case_result.ok
        assert case_result.npz_path is not None
        assert case_result.json_path is not None
        assert case_result.npz_path.exists()
        assert case_result.json_path.exists()
