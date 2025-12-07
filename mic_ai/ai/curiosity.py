from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .world_model import SimpleWorldModel


class WorldModelCuriosity:
    """
    Curiosity-модуль:
      - хранит ссылку на world_model,
      - считает intrinsic reward как нормированную ошибку предсказания.
    """

    def __init__(self, model: SimpleWorldModel, beta: float = 0.1):
        self.model = model
        self.beta = beta
        self.eps = 1e-8

    def compute_intrinsic_reward(self, x: np.ndarray, y_true: np.ndarray) -> float:
        # prediction_error возвращает сумму квадратов в среднем по батчу; нормализуем на размерность
        y_true_arr = np.asarray(y_true, dtype=np.float32)
        dim = float(y_true_arr.shape[-1]) if y_true_arr.ndim > 0 else 1.0
        err2 = self.model.prediction_error(x, y_true)
        r_int = self.beta * float(np.sqrt(err2 / (dim + self.eps) + self.eps))
        return r_int

    def update_model(self, x: np.ndarray, y_true: np.ndarray) -> float:
        return self.model.update(x, y_true)


__all__ = ["WorldModelCuriosity"]
