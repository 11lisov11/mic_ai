from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    init: float


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be float-like, got {value!r}") from exc


def normalize_param_space(param_space: Any) -> List[ParamSpec]:
    specs: List[ParamSpec] = []

    if isinstance(param_space, dict):
        items = list(param_space.items())
        for name, bounds in items:
            if not isinstance(bounds, (list, tuple)) or len(bounds) not in (2, 3):
                raise ValueError(f"Param '{name}' must be (low, high) or (low, high, init).")
            low = _as_float(bounds[0], f"{name}.low")
            high = _as_float(bounds[1], f"{name}.high")
            init = _as_float(bounds[2], f"{name}.init") if len(bounds) == 3 else (low + high) / 2.0
            specs.append(ParamSpec(str(name), low, high, init))
        return specs

    if isinstance(param_space, (list, tuple)):
        for entry in param_space:
            if not isinstance(entry, dict):
                raise ValueError("Param space list entries must be dicts.")
            name = str(entry.get("name"))
            if not name:
                raise ValueError("Param space entry missing 'name'.")
            low = _as_float(entry.get("low"), f"{name}.low")
            high = _as_float(entry.get("high"), f"{name}.high")
            init = entry.get("init", (low + high) / 2.0)
            init = _as_float(init, f"{name}.init")
            specs.append(ParamSpec(name, low, high, init))
        return specs

    raise TypeError("param_space must be dict or list of dicts.")


def _to_unit(x: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    span = np.maximum(highs - lows, 1e-12)
    return (x - lows) / span


def _from_unit(u: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    span = np.maximum(highs - lows, 1e-12)
    return lows + u * span


def optimize(
    score_fn: Callable[[Dict[str, float]], float],
    param_space: Any,
    budget: int,
    seed: Optional[int] = None,
    sigma0: float = 0.3,
    popsize: Optional[int] = None,
    callback: Optional[Callable[[int, Dict[str, float], float], None]] = None,
) -> Dict[str, Any]:
    specs = normalize_param_space(param_space)
    if budget <= 0:
        raise ValueError("budget must be > 0")
    if not specs:
        raise ValueError("param_space must contain at least one parameter")

    names = [s.name for s in specs]
    lows = np.array([s.low for s in specs], dtype=float)
    highs = np.array([s.high for s in specs], dtype=float)
    init = np.array([s.init for s in specs], dtype=float)
    n = len(names)

    rng = np.random.default_rng(seed)
    lam = popsize or (4 + int(3 * math.log(n)))
    mu = max(1, lam // 2)

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights) ** 2 / np.sum(weights**2)

    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    m = _to_unit(init, lows, highs)
    sigma = float(sigma0)
    pc = np.zeros(n, dtype=float)
    ps = np.zeros(n, dtype=float)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    invsqrtC = np.eye(n)

    eval_count = 0
    best_score = float("inf")
    best_params: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    while eval_count < budget:
        arz = rng.standard_normal((lam, n))
        ary = (arz @ (B * D).T)
        arx = m + sigma * ary
        arx = np.clip(arx, 0.0, 1.0)

        scores: List[Tuple[float, int]] = []
        for k in range(lam):
            if eval_count >= budget:
                break
            x = _from_unit(arx[k], lows, highs)
            params = {name: float(val) for name, val in zip(names, x)}
            score = float(score_fn(params))
            if not math.isfinite(score):
                score = float("inf")
            scores.append((score, k))
            eval_count += 1

            if score < best_score:
                best_score = score
                best_params = dict(params)
            if callback is not None:
                callback(eval_count, params, score)
            history.append({"eval": eval_count, "score": score, "params": params})

        if not scores:
            break
        scores.sort(key=lambda x: x[0])
        mu_eff = min(mu, len(scores))
        idx = [k for _score, k in scores[:mu_eff]]

        x_selected = arx[idx]
        m_prev = m.copy()
        weights_eff = weights[:mu_eff]
        weights_eff = weights_eff / np.sum(weights_eff)
        m = np.sum(x_selected * weights_eff[:, None], axis=0)
        y_w = (m - m_prev) / max(sigma, 1e-12)

        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
        norm_ps = np.linalg.norm(ps)
        hsig = int(norm_ps / math.sqrt(1 - (1 - cs) ** (2 * eval_count / lam)) / chi_n < (1.4 + 2 / (n + 1)))

        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w

        artmp = (x_selected - m_prev) / max(sigma, 1e-12)
        delta_hsig = (1 - hsig) * cc * (2 - cc)
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C)
        for i, w in enumerate(weights_eff):
            C += cmu * w * np.outer(artmp[i], artmp[i])

        sigma = sigma * math.exp((cs / damps) * (norm_ps / chi_n - 1))

        C = np.triu(C) + np.triu(C, 1).T
        D_vals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D_vals, 1e-20))
        invsqrtC = B @ np.diag(1.0 / D) @ B.T

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "history": history,
        "evaluations": eval_count,
        "param_names": names,
    }


__all__ = ["optimize", "normalize_param_space", "ParamSpec"]
