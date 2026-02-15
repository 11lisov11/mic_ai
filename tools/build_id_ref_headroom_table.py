from __future__ import annotations

"""
Build `paper/pgups_2026/data/id_ref_headroom_table.csv` from the committed id_ref sweep JSON reports.

Source JSONs are expected under `outputs/id_ref_sweep_pgups/*/id_ref_sweep.json`.
"""

import csv
import json
from pathlib import Path


MOTOR_LABELS = {
    "env_research_air56_025kw": ("air56", "АИР56 0,25 кВт"),
    "env_research_al31_4_06kw": ("al31", "АЛ-31-4 0,6 кВт"),
    "env_research_ao2_32_4_3kw": ("ao2", "АО2-32-4 3,0 кВт"),
}


def _motor_from_env_config(env_config: str) -> tuple[str, str]:
    stem = Path(str(env_config)).stem
    if stem in MOTOR_LABELS:
        return MOTOR_LABELS[stem]
    return stem, stem


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    src_root = Path("outputs/id_ref_sweep_pgups")
    out_path = Path("paper/pgups_2026/data/id_ref_headroom_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []

    for report_path in sorted(src_root.glob("*/id_ref_sweep.json")):
        rep = _load(report_path)
        motor_key, motor_label = _motor_from_env_config(rep.get("env_config", ""))
        scenario = str(rep.get("scenario", "")).strip()

        baseline = rep.get("baseline", {}) or {}
        id_base = float(baseline.get("id_ref", 0.0))
        p_base = float(baseline.get("mean_p_el_pos", 0.0))
        err_base = float(baseline.get("mean_err", 0.0))
        err_limit = float(baseline.get("err_limit", 0.0))

        best = None
        for r in rep.get("rows", []) or []:
            if not bool(r.get("err_ok", False)):
                continue
            p = float(r.get("mean_p_el_pos", 0.0))
            if best is None or p < float(best.get("mean_p_el_pos", 0.0)):
                best = r

        if best is None:
            id_best = id_base
            p_best = p_base
            err_best = err_base
        else:
            id_best = float(best.get("id_ref", id_base))
            p_best = float(best.get("mean_p_el_pos", p_base))
            err_best = float(best.get("mean_err", err_base))

        saving_pct = 0.0 if p_base <= 1e-12 else 100.0 * (1.0 - p_best / p_base)

        rows_out.append(
            {
                "motor_key": motor_key,
                "motor_label": motor_label,
                "scenario": scenario,
                "id_base": id_base,
                "p_base_w": p_base,
                "id_best": id_best,
                "p_best_w": p_best,
                "saving_pct": saving_pct,
                "err_base": err_base,
                "err_best": err_best,
                "err_limit": err_limit,
            }
        )

    cols = [
        "motor_key",
        "motor_label",
        "scenario",
        "id_base",
        "p_base_w",
        "id_best",
        "p_best_w",
        "saving_pct",
        "err_base",
        "err_best",
        "err_limit",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k) for k in cols})

    print(f"OK: wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

