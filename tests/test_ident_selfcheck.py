import sys
from pathlib import Path

import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vfd_ai.core.env import make_env_from_config
from vfd_ai.ident.selfcheck import self_check_full_identification


def test_selfcheck_passes_on_demo_env():
    cfg_path = Path("config/env_demo_true.py")
    if not cfg_path.exists():
        pytest.skip("demo config missing")

    def make_env():
        return make_env_from_config(str(cfg_path))

    # Should not raise if identification converges within tolerances
    self_check_full_identification(make_env_with_true_params=make_env, max_attempts=2, tol_percent_main=20.0, tol_percent_mech=50.0)
