from __future__ import annotations

"""CLI wrapper kept for discoverability: Delta MS300 Modbus bridge."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.delta_ms300_modbus import main


if __name__ == "__main__":
    raise SystemExit(main())
