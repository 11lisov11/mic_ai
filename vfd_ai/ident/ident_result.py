"""
Container for identification results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .motor_params import MotorParamsEstimated, MotorParamsTrue


@dataclass
class IdentificationResult:
    motor_name: str
    source: str  # "simulation" or "hardware"
    timestamp: str
    tests_meta: Dict[str, Any]
    estimated: MotorParamsEstimated
    true_params: Optional[MotorParamsTrue] = None
    rel_error: Dict[str, float] = field(default_factory=dict)


__all__ = ["IdentificationResult"]

