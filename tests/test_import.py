"""Sanity-check imports and fixture wiring."""

import nibabel as nib
import numpy as np
from undistortme import pipeline as cp
from undistortme import contrastmatch


# ---------------------------------------------------------------------------
# Import-level callables
# ---------------------------------------------------------------------------


def test_corr_pipeline_has_process_run():
    assert callable(cp.process_run)


def test_corr_pipeline_has_get_topup_command():
    assert callable(cp.get_topup_command)


def test_contrastmatch_has_main():
    assert callable(contrastmatch.main)


def test_contrastmatch_has_load_image():
    assert callable(contrastmatch.load_image)


# ---------------------------------------------------------------------------
# Fixture wiring
# ---------------------------------------------------------------------------


def test_tiny_nii_roundtrip(tmp_path, tiny_nii):
    """tiny_nii writes a readable NIfTI file."""
    p = tmp_path / "test.nii"
    path_str = tiny_nii(p)
    img = nib.load(path_str)
    assert img.shape == (8, 8, 8)


def test_tiny_nii_fill(tmp_path, tiny_nii):
    """fill= produces a constant-valued array."""
    p = tmp_path / "filled.nii"
    path_str = tiny_nii(p, fill=3.0)
    data = nib.load(path_str).get_fdata()
    assert data.shape == (8, 8, 8)
    assert np.allclose(data, 3.0)


def test_check_dict_initial_state(check_dict):
    """All gates start off: the boolean ones False, "denoise" (a method) None."""
    for key in check_dict:
        if key == "denoise":
            assert cp.check_dict[key] is None
        else:
            assert cp.check_dict[key] is False, f"Expected {key!r} to be False"


def test_check_dict_gate_flip(check_dict):
    """Flipping a key is reflected through the module-global reference."""
    assert cp.check_dict["topup"] is False
    check_dict["topup"] = True
    assert cp.check_dict["topup"] is True


def test_fixtures_combined(tmp_path, tiny_nii, check_dict):
    """Both fixtures work together in the same test."""
    p = tmp_path / "combined.nii"
    path_str = tiny_nii(p, seed=7)
    img = nib.load(path_str)
    assert img.shape == (8, 8, 8)

    assert cp.check_dict["topup"] is False
    check_dict["topup"] = True
    assert cp.check_dict["topup"] is True


def test_contrastmatch_command_shape():
    """Pin the contrastmatch invocation (the snapshot placeholder is derived
    from this constant, so snapshots alone cannot catch a regression here)."""
    import sys

    assert cp.CONTRASTMATCH_CMD == f"{sys.executable} -m undistortme.contrastmatch"
