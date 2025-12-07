from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple


def _init_layer(in_dim: int, out_dim: int, scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    w = np.random.randn(out_dim, in_dim).astype(np.float32) * scale
    b = np.zeros(out_dim, dtype=np.float32)
    return w, b


class SimpleWorldModel:
    """
    Компактная MLP-модель мира на numpy:
      x_t = concat(obs_t, action_t)
      y_t = obs_{t+1}
    2 скрытых слоя ReLU по умолчанию (64,64), online SGD, скользящая нормализация входов/выходов.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int] | None = (64, 64), lr: float = 1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.hidden_sizes = list(hidden_sizes) if hidden_sizes else [64, 64]

        dims = [input_dim] + self.hidden_sizes + [output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            w, b = _init_layer(dims[i], dims[i + 1], scale=0.1 / np.sqrt(max(dims[i], 1)))
            self.weights.append(w)
            self.biases.append(b)

        # Нормировочные статистики для входов/выходов
        self.x_mean = np.zeros(input_dim, dtype=np.float32)
        self.x_var = np.ones(input_dim, dtype=np.float32)
        self.y_mean = np.zeros(output_dim, dtype=np.float32)
        self.y_var = np.ones(output_dim, dtype=np.float32)
        self._count = 1e-6  # чтобы не делить на 0

    def _update_stats(self, xs: np.ndarray, ys: np.ndarray) -> None:
        batch = xs.shape[0]
        total = self._count + batch
        alpha = batch / total
        self.x_mean = (1 - alpha) * self.x_mean + alpha * xs.mean(axis=0)
        self.y_mean = (1 - alpha) * self.y_mean + alpha * ys.mean(axis=0)
        self.x_var = (1 - alpha) * self.x_var + alpha * xs.var(axis=0)
        self.y_var = (1 - alpha) * self.y_var + alpha * ys.var(axis=0)
        self._count = total

    def _norm(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        return (x - mean) / np.sqrt(var + 1e-6)

    def _denorm(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        return x * np.sqrt(var + 1e-6) + mean

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        h = x
        activations = [h]
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.T + b
            if idx < len(self.weights) - 1:
                h = np.maximum(h, 0.0)  # ReLU
            activations.append(h)
        return h, activations

    def _backward(self, activations: List[np.ndarray], grad_out: np.ndarray) -> None:
        grad = grad_out
        for idx in reversed(range(len(self.weights))):
            h_prev = activations[idx]
            w = self.weights[idx]

            grad_w = grad.T @ h_prev
            grad_b = grad.sum(axis=0)

            grad_h_prev = grad @ w
            if idx > 0:
                grad_h_prev = grad_h_prev * (activations[idx] > 0).astype(np.float32)

            self.weights[idx] -= self.lr * grad_w
            self.biases[idx] -= self.lr * grad_b

            grad = grad_h_prev

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x_n = self._norm(x, self.x_mean, self.x_var)
        y_hat, _ = self._forward(x_n)
        return self._denorm(y_hat, self.y_mean, self.y_var)

    def update(self, xs: np.ndarray, ys: np.ndarray) -> float:
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if xs.ndim == 1:
            xs = xs[None, :]
            ys = ys[None, :]

        self._update_stats(xs, ys)
        xs_n = self._norm(xs, self.x_mean, self.x_var)
        ys_n = self._norm(ys, self.y_mean, self.y_var)

        y_hat, activations = self._forward(xs_n)
        err = y_hat - ys_n
        loss = 0.5 * float(np.mean(np.sum(err * err, axis=1)))

        grad_out = err / max(xs.shape[0], 1)
        self._backward(activations, grad_out)
        return loss

    def prediction_error(self, x: np.ndarray, y_true: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
            y_true = y_true[None, :]
        y_pred = self.predict(x)
        err = y_pred - y_true
        return float(np.mean(np.sum(err * err, axis=1)))


class WorldModel(SimpleWorldModel):
    """
    Обёртка над SimpleWorldModel, принимающая obs и act отдельно.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int] | None = (64, 64), lr: float = 1e-4):
        super().__init__(obs_dim + act_dim, obs_dim, hidden_sizes=hidden_sizes, lr=lr)

    def predict_pair(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray(act, dtype=np.float32)], axis=-1)
        return self.predict(x)

    def update_pair(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
        x = np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray(act, dtype=np.float32)], axis=-1)
        return self.update(x, np.asarray(next_obs, dtype=np.float32))


__all__ = ["SimpleWorldModel"]
