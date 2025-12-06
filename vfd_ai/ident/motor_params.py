"""
Dataclasses describing true and estimated motor parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotorParamsTrue:
    Rs: float
    Rr: float
    Ls: float
    Lr: float
    Lm: float
    J: float
    B: float


@dataclass
class MotorParamsEstimated:
    Rs: float | None = None
    Rr: float | None = None
    Ls: float | None = None
    Lr: float | None = None
    Lm: float | None = None
    J: float | None = None
    B: float | None = None

    def as_dict(self) -> dict:
        return {
            "Rs": self.Rs,
            "Rr": self.Rr,
            "Ls": self.Ls,
            "Lr": self.Lr,
            "Lm": self.Lm,
            "J": self.J,
            "B": self.B,
        }


__all__ = ["MotorParamsTrue", "MotorParamsEstimated"]

