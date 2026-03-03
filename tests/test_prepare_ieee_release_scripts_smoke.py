from __future__ import annotations

from pathlib import Path


def test_prepare_release_wrapper_scripts_present() -> None:
    ps1 = Path("scripts/prepare_ieee_release_commit.ps1").resolve()
    sh = Path("scripts/prepare_ieee_release_commit.sh").resolve()
    assert ps1.exists()
    assert sh.exists()

    ps1_text = ps1.read_text(encoding="utf-8")
    sh_text = sh.read_text(encoding="utf-8")

    for token in (
        "tools/prepare_ieee_release_commit.py",
        "--step28-dir",
        "--ieee-root",
    ):
        assert token in ps1_text
        assert token in sh_text

