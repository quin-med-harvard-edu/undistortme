"""Tests for the mode-dependent bundled TOPUP config default.

With no -c, the CLI resolves to the packaged pervol.cnf (whole-volume) or
perslice_1.cnf (-s/--slice); an explicit -c always wins.  The packaged copies
under src/undistortme/configs/ must stay identical to the user-facing copies
in the repo-root configs/ directory (the README tells users to pass those).
"""

import sys
from importlib import resources
from pathlib import Path

from undistortme import pipeline as cp

REPO_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
PACKAGED_CONFIGS = resources.files("undistortme") / "configs"


def _parsed(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["undistortme", *argv])
    return cp.get_args()


def test_get_args_leaves_config_unresolved(monkeypatch):
    """Resolution must wait until check_binaries has settled slice mode."""
    assert _parsed(monkeypatch, "-t", "-s").config_file is None


def test_resolve_config_defaults_by_mode():
    assert Path(cp.resolve_config(None, slice_check=False)).name == "pervol.cnf"
    assert Path(cp.resolve_config(None, slice_check=True)).name == "perslice_1.cnf"


def test_resolve_config_explicit_wins():
    assert cp.resolve_config("my.cnf", slice_check=True) == "my.cnf"


def test_slice_downgrade_changes_the_default(monkeypatch):
    """No slicenii on the host: check_binaries turns slice mode off, and the
    default must follow it to pervol.cnf, not keep the per-slice schedule."""
    monkeypatch.setattr(cp.shutil, "which", lambda name: None)
    _, _, _, slice_check = cp.check_binaries(False, False, True)
    assert slice_check is False
    assert Path(cp.resolve_config(None, slice_check)).name == "pervol.cnf"


def test_default_config_estmov_off():
    """Both bundled defaults must keep TOPUP movement estimation disabled."""
    for slice_mode in (False, True):
        text = Path(cp.default_config(slice_mode)).read_text()
        assert "--estmov=0,0,0,0,0,0,0,0,0" in text


def test_packaged_configs_match_repo_configs():
    """Guard against drift between configs/ and src/undistortme/configs/."""
    repo_names = {p.name for p in REPO_CONFIGS.glob("*.cnf")}
    packaged_names = {
        p.name for p in PACKAGED_CONFIGS.iterdir() if p.name.endswith(".cnf")
    }
    assert repo_names == packaged_names
    for name in repo_names:
        assert (REPO_CONFIGS / name).read_bytes() == (
            PACKAGED_CONFIGS / name
        ).read_bytes(), name
