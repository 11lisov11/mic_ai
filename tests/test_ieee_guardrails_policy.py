from __future__ import annotations

import json
from pathlib import Path


def test_ieee_guardrails_policy_schema() -> None:
    path = Path("paper/ieee_2026/guardrails_policy.json").resolve()
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert int(payload.get("version", 0)) >= 1
    raw = payload.get("motor_saving_thresholds_pct")
    assert isinstance(raw, dict)

    for key in ("air56", "al31", "ao2"):
        assert key in raw
        assert isinstance(raw[key], (int, float))

    # Keep current baseline guardrails explicit and non-negative.
    assert float(raw["air56"]) >= 0.5
    assert float(raw["al31"]) >= 0.0
    assert float(raw["ao2"]) >= 0.05
