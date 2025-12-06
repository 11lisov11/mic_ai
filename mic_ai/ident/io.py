"""
Помощники для чтения/записи JSON с результатами идентификации.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .ident_result import IdentificationResult
from .motor_params import MotorParamsEstimated, MotorParamsTrue


def _motor_params_true_to_dict(params: MotorParamsTrue | None) -> Dict[str, Any] | None:
    if params is None:
        return None
    return asdict(params)


def save_ident_result(result: IdentificationResult, path: str) -> None:
    payload = {
        "motor_name": result.motor_name,
        "source": result.source,
        "timestamp": result.timestamp,
        "tests": result.tests_meta,
        "estimated_params": result.estimated.as_dict(),
        "true_params_if_available": _motor_params_true_to_dict(result.true_params),
        "relative_error_percent": result.rel_error,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_estimated_params(path: str) -> MotorParamsEstimated:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params_dict = payload.get("estimated_params")
    if not isinstance(params_dict, dict):
        raise ValueError("JSON does not contain 'estimated_params' object")
    return MotorParamsEstimated(**params_dict)


def load_test_data(path: str) -> dict:
    """
    Загрузить записанные тестовые данные для идентификации.

    Поддерживает:
    - .npz: массивы с ключами (t, u_d, u_q, i_d, i_q, w_mech, torque и т.д.)
    - .json: словарь списков/чисел
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        data_npz = np.load(p)
        return {k: data_npz[k] for k in data_npz.files}
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported test data format for {path}; use .npz or .json")


__all__ = ["save_ident_result", "load_estimated_params", "load_test_data"]
