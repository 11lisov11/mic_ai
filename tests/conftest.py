import numpy as np


def pytest_configure() -> None:
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except Exception:
        pass
