from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure


def _load_csv(path: Path) -> Dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"CSV is empty: {path}")
    header = lines[0].split(",")
    cols: Dict[str, list[float]] = {name: [] for name in header}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            # Skip malformed lines to keep plots robust.
            continue
        for key, value in zip(header, parts):
            cols[key].append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def _parse_meta(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    meta: Dict[str, object] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _extract_float(meta: Dict[str, object], key: str) -> float | None:
    if key not in meta:
        return None
    try:
        return float(meta[key])
    except Exception:
        return None


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path | None]:
    if args.input_dir:
        base = Path(args.input_dir)
        foc_csv = base / "timeseries_foc.csv"
        mic_csv = base / "timeseries_mic_ai.csv"
        meta = base / "run_meta.json"
        return foc_csv, mic_csv, meta if meta.exists() else None
    if not args.foc_csv or not args.mic_csv:
        raise ValueError("Provide --input-dir or both --foc-csv and --mic-csv.")
    foc_csv = Path(args.foc_csv)
    mic_csv = Path(args.mic_csv)
    meta = Path(args.run_meta) if args.run_meta else None
    if meta is None:
        candidate = foc_csv.parent / "run_meta.json"
        meta = candidate if candidate.exists() else None
    return foc_csv, mic_csv, meta


def _clip_positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _compute_summary(
    foc: Dict[str, np.ndarray],
    mic: Dict[str, np.ndarray],
    omega_ref: float,
    window_frac: float,
) -> Dict[str, Dict[str, float | None]]:
    n = int(min(foc["t"].size, mic["t"].size))
    sl = _steady_slice(n, window_frac)
    omega_ref_scale = max(abs(float(omega_ref)), 1e-6)

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr[sl])) if arr.size else 0.0

    def _mean_abs_err(series: Dict[str, np.ndarray]) -> float:
        err = np.abs(float(omega_ref) - series["omega"])
        return float(np.mean(err[sl])) if err.size else 0.0

    foc_p_el = _mean(foc["p_el"])
    mic_p_el = _mean(mic["p_el"])
    foc_p_el_pos = _mean(_clip_positive(foc["p_el"]))
    mic_p_el_pos = _mean(_clip_positive(mic["p_el"]))
    foc_err = _mean_abs_err(foc)
    mic_err = _mean_abs_err(mic)
    foc_err_rel = foc_err / omega_ref_scale
    mic_err_rel = mic_err / omega_ref_scale
    power_saving_pct = 0.0
    if foc_p_el_pos > 1e-9:
        power_saving_pct = 100.0 * (1.0 - mic_p_el_pos / foc_p_el_pos)
    err_rel_loss_pp = 100.0 * (mic_err_rel - foc_err_rel)
    err_rel_loss_pct = None
    if foc_err_rel > 1e-9:
        err_rel_loss_pct = 100.0 * (mic_err_rel / foc_err_rel - 1.0)

    return {
        "foc": {
            "mean_p_el": foc_p_el,
            "mean_p_el_pos": foc_p_el_pos,
            "mean_abs_speed_err": foc_err,
            "mean_abs_speed_err_rel": foc_err_rel,
        },
        "mic": {
            "mean_p_el": mic_p_el,
            "mean_p_el_pos": mic_p_el_pos,
            "mean_abs_speed_err": mic_err,
            "mean_abs_speed_err_rel": mic_err_rel,
        },
        "delta": {
            "power_saving_pct": power_saving_pct,
            "speed_err_rel_loss_pp": err_rel_loss_pp,
            "speed_err_rel_loss_pct": err_rel_loss_pct,
        },
    }


def _plot_power(
    out_path: Path,
    foc: Dict[str, np.ndarray],
    mic: Dict[str, np.ndarray],
    clip_negative: bool,
) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    p_foc = _clip_positive(foc["p_el"]) if clip_negative else foc["p_el"]
    p_mic = _clip_positive(mic["p_el"]) if clip_negative else mic["p_el"]

    ax.plot(foc["t"], p_foc, color="black", label="FOC")
    ax.plot(mic["t"], p_mic, color="0.35", linestyle="--", label="MIC AI")

    ax.set_xlabel("t, с")
    ylabel = "P_эл, Вт" if not clip_negative else "P_эл^+, Вт"
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_speed_error(
    out_path: Path,
    foc: Dict[str, np.ndarray],
    mic: Dict[str, np.ndarray],
    omega_ref: float,
) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    err_foc = np.abs(omega_ref - foc["omega"])
    err_mic = np.abs(omega_ref - mic["omega"])

    ax.plot(foc["t"], err_foc, color="black", label="FOC")
    ax.plot(mic["t"], err_mic, color="0.35", linestyle="--", label="MIC AI")

    ax.set_xlabel("t, с")
    ax.set_ylabel("|ω_ref - ω|, рад/с")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-ready time-series comparison: power and speed error (FOC vs MIC AI)."
    )
    parser.add_argument("--input-dir", default=None, help="Directory with timeseries_foc.csv, timeseries_mic_ai.csv.")
    parser.add_argument("--foc-csv", default=None, help="Path to timeseries_foc.csv.")
    parser.add_argument("--mic-csv", default=None, help="Path to timeseries_mic_ai.csv.")
    parser.add_argument("--run-meta", default=None, help="Optional run_meta.json path to read omega_ref.")
    parser.add_argument("--omega-ref", type=float, default=None, help="Override omega_ref (rad/s).")
    parser.add_argument("--out-dir", default="outputs/paper_timeseries_compare")
    parser.add_argument("--prefix", default="fig")
    parser.add_argument("--window-frac", type=float, default=0.25, help="Steady-state window fraction for metrics.")
    parser.add_argument(
        "--clip-negative",
        action="store_true",
        help="Clip negative electric power to zero (P_эл^+).",
    )
    args = parser.parse_args()

    foc_csv, mic_csv, meta_path = _resolve_paths(args)
    if not foc_csv.exists():
        raise FileNotFoundError(f"FOC CSV not found: {foc_csv}")
    if not mic_csv.exists():
        raise FileNotFoundError(f"MIC CSV not found: {mic_csv}")

    meta = _parse_meta(meta_path) if meta_path is not None else {}
    omega_ref = float(args.omega_ref) if args.omega_ref is not None else _extract_float(meta, "omega_ref")
    if omega_ref is None:
        raise ValueError("omega_ref not provided; pass --omega-ref or supply run_meta.json.")

    foc = _load_csv(foc_csv)
    mic = _load_csv(mic_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_power(out_dir / f"{args.prefix}_power", foc, mic, bool(args.clip_negative))
    _plot_speed_error(out_dir / f"{args.prefix}_speed_error", foc, mic, omega_ref)

    summary = {
        "foc_csv": str(foc_csv),
        "mic_csv": str(mic_csv),
        "run_meta": str(meta_path) if meta_path is not None else "",
        "omega_ref": float(omega_ref),
        "clip_negative": bool(args.clip_negative),
        "plot_style": "vak_ru",
        "plot_formats": ["png", "pdf", "svg"],
        "window_frac": float(args.window_frac),
    }
    summary.update(_compute_summary(foc, mic, omega_ref, float(args.window_frac)))
    (out_dir / f"{args.prefix}_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
