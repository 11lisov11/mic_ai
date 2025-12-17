from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricRow:
    ai_mean: float
    ai_ci95: float
    foc_mean: float
    foc_ci95: float


METRIC_LABELS_LATEX: Dict[str, str] = {
    "i_rms": r"$I_{\mathrm{rms}}$, А",
    "p_in_pos": r"$P_{in}^+$, Вт",
    "speed_err": r"$|\omega_{ref}-\omega|$, рад/с",
}

METRIC_LABELS_MD: Dict[str, str] = {
    "i_rms": "I_rms, А",
    "p_in_pos": "P_in⁺, Вт",
    "speed_err": "|ω_ref − ω|, рад/с",
}


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _finite(x: float, default: float = 0.0) -> float:
    try:
        x = float(x)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _fmt(metric: str, value: float) -> str:
    value = _finite(value, 0.0)
    if metric == "p_in_pos":
        return f"{value:.2f}"
    if metric == "speed_err":
        return f"{value:.3f}"
    return f"{value:.3f}"


def _fmt_pm_latex(metric: str, mean: float, ci95: float) -> str:
    mean = _finite(mean, 0.0)
    ci95 = _finite(ci95, 0.0)
    if ci95 > 0:
        return rf"{_fmt(metric, mean)} $\pm$ {_fmt(metric, ci95)}"
    return _fmt(metric, mean)


def _fmt_pm_md(metric: str, mean: float, ci95: float) -> str:
    mean = _finite(mean, 0.0)
    ci95 = _finite(ci95, 0.0)
    if ci95 > 0:
        return f"{_fmt(metric, mean)} ± {_fmt(metric, ci95)}"
    return _fmt(metric, mean)


def _read_summary_overall(path: Path) -> Dict[str, MetricRow]:
    rows: Dict[str, MetricRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            rows[metric] = MetricRow(
                ai_mean=_finite(row.get("ai_mean", 0.0)),
                ai_ci95=_finite(row.get("ai_ci95", 0.0)),
                foc_mean=_finite(row.get("foc_mean", 0.0)),
                foc_ci95=_finite(row.get("foc_ci95", 0.0)),
            )
    return rows


def _read_summary_by_stage(path: Path) -> Dict[int, Dict[str, MetricRow]]:
    out: Dict[int, Dict[str, MetricRow]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                stage = int(float(row.get("stage", 0)))
            except Exception:
                stage = 0
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            out.setdefault(stage, {})[metric] = MetricRow(
                ai_mean=_finite(row.get("ai_mean", 0.0)),
                ai_ci95=_finite(row.get("ai_ci95", 0.0)),
                foc_mean=_finite(row.get("foc_mean", 0.0)),
                foc_ci95=_finite(row.get("foc_ci95", 0.0)),
            )
    return out


def _read_summary_cases(path: Path) -> Dict[str, Dict[str, MetricRow]]:
    out: Dict[str, Dict[str, MetricRow]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            case = str(row.get("case", "")).strip()
            metric = str(row.get("metric", "")).strip()
            if not case or not metric:
                continue
            out.setdefault(case, {})[metric] = MetricRow(
                ai_mean=_finite(row.get("ai_mean", 0.0)),
                ai_ci95=_finite(row.get("ai_ci95", 0.0)),
                foc_mean=_finite(row.get("foc_mean", 0.0)),
                foc_ci95=_finite(row.get("foc_ci95", 0.0)),
            )
    return out


def _metric_order(rows: Iterable[str]) -> List[str]:
    preferred = ["i_rms", "p_in_pos", "speed_err"]
    rest = [m for m in rows if m not in preferred]
    return [m for m in preferred if m in rows] + sorted(rest)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8-sig")


def _latex_table_overall(rows: Dict[str, MetricRow]) -> str:
    metrics = _metric_order(rows.keys())
    lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Метрика & AI (mean $\pm$ 95\% ДИ) & FOC (mean $\pm$ 95\% ДИ) \\",
        r"\hline",
    ]
    for m in metrics:
        r0 = rows[m]
        label = METRIC_LABELS_LATEX.get(m, _latex_escape(m))
        lines.append(
            f"{label} & {_fmt_pm_latex(m, r0.ai_mean, r0.ai_ci95)} & {_fmt_pm_latex(m, r0.foc_mean, r0.foc_ci95)} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def _md_table_overall(rows: Dict[str, MetricRow]) -> str:
    metrics = _metric_order(rows.keys())
    lines = [
        "| Метрика | AI (mean ± 95% ДИ) | FOC (mean ± 95% ДИ) |",
        "|---|---:|---:|",
    ]
    for m in metrics:
        r0 = rows[m]
        label = METRIC_LABELS_MD.get(m, m)
        lines.append(f"| {label} | {_fmt_pm_md(m, r0.ai_mean, r0.ai_ci95)} | {_fmt_pm_md(m, r0.foc_mean, r0.foc_ci95)} |")
    return "\n".join(lines)


def _latex_tables_by_stage(rows: Dict[int, Dict[str, MetricRow]]) -> Dict[str, str]:
    stages = sorted(rows.keys())
    metrics = _metric_order({m for st in stages for m in rows.get(st, {}).keys()})
    out: Dict[str, str] = {}
    for m in metrics:
        lines = [
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Стадия & AI (mean $\pm$ 95\% ДИ) & FOC (mean $\pm$ 95\% ДИ) \\",
            r"\hline",
        ]
        for st in stages:
            r0 = rows.get(st, {}).get(m)
            if r0 is None:
                continue
            lines.append(
                f"{st} & {_fmt_pm_latex(m, r0.ai_mean, r0.ai_ci95)} & {_fmt_pm_latex(m, r0.foc_mean, r0.foc_ci95)} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}"]
        out[m] = "\n".join(lines)
    return out


def _md_tables_by_stage(rows: Dict[int, Dict[str, MetricRow]]) -> Dict[str, str]:
    stages = sorted(rows.keys())
    metrics = _metric_order({m for st in stages for m in rows.get(st, {}).keys()})
    out: Dict[str, str] = {}
    for m in metrics:
        lines = [
            "| Стадия | AI (mean ± 95% ДИ) | FOC (mean ± 95% ДИ) |",
            "|---:|---:|---:|",
        ]
        for st in stages:
            r0 = rows.get(st, {}).get(m)
            if r0 is None:
                continue
            lines.append(f"| {st} | {_fmt_pm_md(m, r0.ai_mean, r0.ai_ci95)} | {_fmt_pm_md(m, r0.foc_mean, r0.foc_ci95)} |")
        out[m] = "\n".join(lines)
    return out


def _latex_tables_by_case(rows: Dict[str, Dict[str, MetricRow]]) -> Dict[str, str]:
    cases = sorted(rows.keys())
    metrics = _metric_order({m for case in cases for m in rows.get(case, {}).keys()})
    out: Dict[str, str] = {}
    for m in metrics:
        lines = [
            r"\begin{tabular}{lcc}",
            r"\hline",
            r"Кейс & AI (mean $\pm$ 95\% ДИ) & FOC (mean $\pm$ 95\% ДИ) \\",
            r"\hline",
        ]
        for case in cases:
            r0 = rows.get(case, {}).get(m)
            if r0 is None:
                continue
            lines.append(
                f"{_latex_escape(case)} & {_fmt_pm_latex(m, r0.ai_mean, r0.ai_ci95)} & {_fmt_pm_latex(m, r0.foc_mean, r0.foc_ci95)} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}"]
        out[m] = "\n".join(lines)
    return out


def _md_tables_by_case(rows: Dict[str, Dict[str, MetricRow]]) -> Dict[str, str]:
    cases = sorted(rows.keys())
    metrics = _metric_order({m for case in cases for m in rows.get(case, {}).keys()})
    out: Dict[str, str] = {}
    for m in metrics:
        lines = [
            "| Кейс | AI (mean ± 95% ДИ) | FOC (mean ± 95% ДИ) |",
            "|---|---:|---:|",
        ]
        for case in cases:
            r0 = rows.get(case, {}).get(m)
            if r0 is None:
                continue
            lines.append(f"| {case} | {_fmt_pm_md(m, r0.ai_mean, r0.ai_ci95)} | {_fmt_pm_md(m, r0.foc_mean, r0.foc_ci95)} |")
        out[m] = "\n".join(lines)
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export IEEE-ready tables (LaTeX/Markdown) from evaluation CSVs.")
    p.add_argument("--summary-overall", type=str, default=None)
    p.add_argument("--summary-by-stage", type=str, default=None)
    p.add_argument("--summary-cases", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/paper_tables")
    p.add_argument("--prefix", type=str, default="motor1")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    prefix = str(args.prefix).strip() or "tables"

    if args.summary_overall:
        rows = _read_summary_overall(Path(args.summary_overall))
        _write_text(out_dir / f"table_overall_{prefix}.tex", _latex_table_overall(rows))
        _write_text(out_dir / f"table_overall_{prefix}.md", _md_table_overall(rows))

    if args.summary_by_stage:
        rows = _read_summary_by_stage(Path(args.summary_by_stage))
        for metric, latex in _latex_tables_by_stage(rows).items():
            _write_text(out_dir / f"table_by_stage_{metric}_{prefix}.tex", latex)
        for metric, md in _md_tables_by_stage(rows).items():
            _write_text(out_dir / f"table_by_stage_{metric}_{prefix}.md", md)

    if args.summary_cases:
        rows = _read_summary_cases(Path(args.summary_cases))
        for metric, latex in _latex_tables_by_case(rows).items():
            _write_text(out_dir / f"table_by_case_{metric}_{prefix}.tex", latex)
        for metric, md in _md_tables_by_case(rows).items():
            _write_text(out_dir / f"table_by_case_{metric}_{prefix}.md", md)

    print(f"Saved tables to: {out_dir}")


if __name__ == "__main__":
    main()

