from __future__ import annotations

"""
Build `paper/pgups_2026/data/time_to_foc_summary_ru.csv` from committed training artifacts.

We intentionally derive the "time to match FOC" / "time to beat FOC" from the per-episode
evaluation snapshots saved under `results_run/.../eval/ep_XXX/summary.json`.

This keeps the paper reproducible without relying on external experiment trackers.
"""

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunInfo:
    motor_key: str
    motor_label: str
    run_dir: Path


RUNS: tuple[RunInfo, ...] = (
    RunInfo(
        motor_key="air56",
        motor_label="АИР56 0,25 кВт",
        run_dir=Path("results_run/20260215_084707_env_research_air56_025kw_ai_id_ref"),
    ),
    RunInfo(
        motor_key="al31",
        motor_label="АЛ-31-4 0,6 кВт",
        run_dir=Path("results_run/20260215_085913_env_research_al31_4_06kw_ai_id_ref"),
    ),
    RunInfo(
        motor_key="ao2",
        motor_label="АО2-32-4 3,0 кВт",
        run_dir=Path("results_run/20260215_090746_env_research_ao2_32_4_3kw_ai_id_ref"),
    ),
)



_RUN_TS_RE = re.compile(r"^(?P<ymd>\d{8})_(?P<hms>\d{6})_")
_EP_RE = re.compile(r"^ep_(?P<ep>\d+)$")


def _parse_run_start_ts(run_dir: Path) -> datetime:
    m = _RUN_TS_RE.match(run_dir.name)
    if not m:
        raise ValueError(f"Cannot parse start timestamp from run dir name: {run_dir.name}")
    return datetime.strptime(f"{m.group('ymd')}_{m.group('hms')}", "%Y%m%d_%H%M%S")


def _iter_eval_eps(eval_dir: Path) -> Iterable[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for d in eval_dir.iterdir():
        if not d.is_dir():
            continue
        m = _EP_RE.match(d.name)
        if not m:
            continue
        items.append((int(m.group("ep")), d))
    return sorted(items, key=lambda x: x[0])


def _read_run_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_summary(ep_dir: Path) -> list[dict]:
    path = ep_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if abs(den) <= 1e-12:
        return float("inf") if abs(float(num)) > 0 else 1.0
    return float(num) / den


def _read_trace(path: Path) -> dict[str, list[float]]:
    """
    Minimal CSV reader for the evaluation traces written by mic_ai.tools.scenario_compare.

    Expected columns: t, omega, omega_ref, i_rms, p_el, p_mech.
    """

    import pandas as pd

    df = pd.read_csv(path)
    return {k: df[k].to_numpy(dtype=float) for k in ("t", "omega", "omega_ref", "i_rms", "p_el", "p_mech")}


def _trapz(y, x) -> float:
    if len(y) <= 1:
        return float(sum(y))
    import numpy as np

    return float(np.trapezoid(y, x))


def _series_metrics_full(series: dict[str, "object"]) -> dict[str, float]:
    import numpy as np

    t = series["t"]
    omega = series["omega"]
    omega_ref = series["omega_ref"]
    p_in = np.maximum(series["p_el"], 0.0)
    err = np.abs(omega_ref - omega)
    return {
        "mae_full": float(np.mean(err)) if err.size else 0.0,
        "mean_p_in_full": float(np.mean(p_in)) if p_in.size else 0.0,
        "energy_in_full": _trapz(p_in, t),
    }


def _infer_scenario_from_tag(file_tag: str) -> str:
    tag = str(file_tag).strip()
    if tag.startswith("hold_"):
        pu = tag[len("hold_") :].replace("p", ".", 1)
        return f"hold:{pu}"
    return tag


def main() -> None:
    out_path = Path("paper/pgups_2026/data/time_to_foc_summary_ru.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for run in RUNS:
        if not run.run_dir.exists():
            raise FileNotFoundError(run.run_dir)
        eval_dir = run.run_dir / "eval"
        if not eval_dir.exists():
            raise FileNotFoundError(eval_dir)

        cfg = _read_run_config(run.run_dir)
        # Some runs keep eval_dt/eval_t_end explicitly as `null` in run_config.json.
        # Treat those as "use env default".
        t_end = float(cfg.get("eval_t_end") or 2.0)
        episodes_total = int(cfg.get("episodes", 0) or 0)
        dt = float(cfg.get("eval_dt") or cfg.get("dt") or 1e-3)
        # Training "sim time" is reported as N_episodes * t_end (same convention as earlier drafts).
        _ = dt  # kept for future; not used directly.

        start_ts = _parse_run_start_ts(run.run_dir)
        ep_items = list(_iter_eval_eps(eval_dir))
        if not ep_items:
            raise FileNotFoundError(f"No eval snapshots under: {eval_dir}")

        # Criteria used in the paper.
        mae_ratio_limit = 1.05
        better_saving_threshold = 0.0

        ep_equal = None
        ep_better = None
        ep_better_all = None
        last_ep = ep_items[-1][0]
        last_wall_s = None

        for ep, ep_dir in ep_items:
            # Compute metrics from traces (full interval), not from summary.json steady window.
            scenario_tags = []
            for p in ep_dir.glob("*_foc.csv"):
                tag = p.name[: -len("_foc.csv")]
                if (ep_dir / f"{tag}_mic_ai.csv").exists():
                    scenario_tags.append(tag)
            scenario_tags = sorted(set(scenario_tags))
            if not scenario_tags:
                continue

            mae_ratios = []
            savings = []
            for tag in scenario_tags:
                scenario = _infer_scenario_from_tag(tag)
                foc = _read_trace(ep_dir / f"{tag}_foc.csv")
                mic = _read_trace(ep_dir / f"{tag}_mic_ai.csv")
                m_f = _series_metrics_full(foc)
                m_m = _series_metrics_full(mic)
                mae_ratios.append(_safe_ratio(m_m["mae_full"], max(m_f["mae_full"], 1e-12)))
                # Same sign convention as the paper: saving on mean P_in+.
                base = float(m_f["mean_p_in_full"])
                alt = float(m_m["mean_p_in_full"])
                savings.append(100.0 * (1.0 - alt / max(base, 1e-12)))
                _ = scenario  # kept for possible per-scenario filtering

            max_rho = max(mae_ratios) if mae_ratios else float("inf")
            mean_saving = sum(savings) / max(len(savings), 1)
            min_saving = min(savings) if savings else float("-inf")

            # Use the timestamp of the ep snapshot as wall-clock marker.
            wall_s = (ep_dir / "summary.json").stat().st_mtime if (ep_dir / "summary.json").exists() else ep_dir.stat().st_mtime
            wall_ts = datetime.fromtimestamp(wall_s)
            wall_from_start_s = (wall_ts - start_ts).total_seconds()
            last_wall_s = wall_from_start_s

            if ep_equal is None and max_rho <= mae_ratio_limit:
                ep_equal = ep
            if ep_better is None and max_rho <= mae_ratio_limit and mean_saving > better_saving_threshold:
                ep_better = ep
            if ep_better_all is None and max_rho <= mae_ratio_limit and min_saving > better_saving_threshold:
                ep_better_all = ep

        def _wall_for(ep: int | None) -> float | None:
            if ep is None:
                return None
            # We recompute precisely for the chosen ep (avoid relying on last loop state).
            ep_dir = eval_dir / f"ep_{ep:03d}"
            if not ep_dir.exists():
                return None
            stamp = (ep_dir / "summary.json")
            wall_ts = datetime.fromtimestamp(stamp.stat().st_mtime if stamp.exists() else ep_dir.stat().st_mtime)
            return float((wall_ts - start_ts).total_seconds())

        rows.append(
            {
                "motor_key": run.motor_key,
                "motor_label": run.motor_label,
                "t_equal_foc_wall_s": _wall_for(ep_equal),
                "ep_equal_foc": ep_equal,
                "t_equal_foc_sim_s": None if ep_equal is None else float((ep_equal + 1) * t_end),
                "t_better_foc_wall_s": _wall_for(ep_better),
                "ep_better_foc": ep_better,
                "t_better_foc_sim_s": None if ep_better is None else float((ep_better + 1) * t_end),
                "t_better_all_wall_s": _wall_for(ep_better_all),
                "ep_better_all": ep_better_all,
                "t_better_all_sim_s": None if ep_better_all is None else float((ep_better_all + 1) * t_end),
                "not_reached_until_ep": None if (ep_better is not None) else float(last_ep),
                "not_reached_until_wall_s": None if (ep_better is not None) else float(last_wall_s or math.nan),
                "episodes_total": episodes_total,
            }
        )

    # Write CSV with stable column order.
    cols = [
        "motor_key",
        "motor_label",
        "t_equal_foc_wall_s",
        "ep_equal_foc",
        "t_equal_foc_sim_s",
        "t_better_foc_wall_s",
        "ep_better_foc",
        "t_better_foc_sim_s",
        "t_better_all_wall_s",
        "ep_better_all",
        "t_better_all_sim_s",
        "not_reached_until_ep",
        "not_reached_until_wall_s",
        "episodes_total",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    print(f"OK: wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
