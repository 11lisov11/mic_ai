import importlib
import pkgutil
from types import ModuleType
from pathlib import Path


def _walk_package(pkg_name: str) -> None:
    pkg = importlib.import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(mod.name)


def test_import_packages() -> None:
    for name in ("mic_ai", "control", "models", "simulation", "drivers", "config"):
        _walk_package(name)


def test_import_tools_modules() -> None:
    # Import common tool modules that are part of the repo but not heavy to load.
    for mod_name in (
        "tools.uno_q_protocol",
        "tools.uno_q_ai_bridge",
        "mic_ai.tools.calibrate_losses",
    ):
        importlib.import_module(mod_name)
