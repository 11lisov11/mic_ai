import numpy as np
import pytest


SLOW_TEST_FILES = {
    "test_ao2_hardening_sweep_smoke.py",
    "test_frozen_mini_baselines.py",
    "test_reproduce_ieee_step28_smoke.py",
    "test_robust_motor_hardening_smoke.py",
    "test_run_integration_pipeline_smoke.py",
    "test_run_step27_extended_repro_smoke.py",
    "test_step27_pipeline_smoke.py",
    "test_train_3motors_pipeline_joint_and_finetune_smoke.py",
    "test_train_3motors_pipeline_resume_eval_first_smoke.py",
    "test_train_3motors_pipeline_smoke.py",
    "test_train_any_motor_pipeline_smoke.py",
}

SLOW_TEST_PREFIXES = (
    "test_run_external_step27",
    "test_promote_external_step27",
)


def pytest_configure() -> None:
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass


def pytest_collection_modifyitems(items) -> None:
    slow_marker = pytest.mark.slow
    for item in items:
        filename = item.path.name
        if filename in SLOW_TEST_FILES or any(item.name.startswith(prefix) for prefix in SLOW_TEST_PREFIXES):
            item.add_marker(slow_marker)
