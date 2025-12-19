from pathlib import Path

from search.random_search import run_search


def test_random_search_smoke(tmp_path) -> None:
    leaderboard_path = tmp_path / "leaderboard.json"
    log_root = tmp_path / "logs"
    out_dir = tmp_path / "search"

    result = run_search(
        policy="pid_speed",
        iters=3,
        base_candidate_path=Path("candidates/examples/pid_default.json"),
        seed=0,
        with_identification=False,
        leaderboard_path=leaderboard_path,
        log_root=log_root,
        out_dir=out_dir,
        env_config_path=None,
        replicates=1,
        min_rel_improve=0.0,
        min_abs_improve=0.0,
    )

    assert leaderboard_path.exists()
    assert result.all_scores
    assert result.best_score >= min(result.all_scores)
