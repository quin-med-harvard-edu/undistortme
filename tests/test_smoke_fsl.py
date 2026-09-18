"""Real-binaries end-to-end smoke test for FSL TOPUP integration.

This test is intentionally excluded from the default ``pytest`` run (which
filters ``not slow and not needs_fsl`` per pytest.ini).  Run it explicitly:

    PY -m pytest tests/test_smoke_fsl.py -m "slow or needs_fsl" -v

WHAT IT TESTS
-------------
* Build a two-echo, one-volume synthetic BIDS run with 32x32x16 NIfTI files
  (the default conftest tiny_nii 8x8x8 is too small for FSL topup's b02b0.cnf
  subsampling steps; 32x32x16 works reliably with pervol.cnf).
* Wire corr_pipeline.check_dict with topup=True, everything else False.
* Call cp.process_run() with pervol.cnf so topup actually runs.
* Assert corrected output files (*desc-undistorted*) exist and are loadable via
  nibabel with finite data.

No numerical quality assertions are made — this is a wiring / integration
check only.

NOTE ON CONFIG
--------------
``pervol.cnf`` (from the repo root) is used instead of ``b02b0.cnf`` because
b02b0.cnf's subsampling schedule requires larger images.  pervol.cnf runs
topup in ~5-8 seconds on a 32x32x16 phantom.

NOTE ON ENVIRONMENT
-------------------
FSL must be discoverable: either ``topup`` is already on PATH, or ``$FSLDIR``
is set (its ``share/fsl/bin`` / ``bin`` subdirectory is then prepended to
PATH).  Otherwise the test skips.  FSLOUTPUTTYPE=NIFTI is set so FSL emits
.nii not .nii.gz, matching the glob patterns used by run_topup.
"""

import glob
import json
import os
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from undistortme import denoise as dn
from undistortme import pipeline as cp

# ---------------------------------------------------------------------------
# Marks — excluded from default run; both marks needed to be safe
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.slow, pytest.mark.needs_fsl]

# Config that works on small phantoms — relative to repo root
_REPO_ROOT = Path(__file__).parent.parent
PERVOL_CNF = str(_REPO_ROOT / "configs" / "pervol.cnf")


# ---------------------------------------------------------------------------
# Module-level skip if topup is not on PATH (after trying $FSLDIR)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_if_no_fsl(monkeypatch):
    """Ensure topup is reachable (PATH, else $FSLDIR), or skip.

    With UNDISTORTME_REQUIRE_FSL=1 a missing FSL FAILS instead of skipping —
    CI uses this so a broken image can never pass its smoke gate by skipping.
    """
    if shutil.which("topup") is None:
        fsldir = os.environ.get("FSLDIR", "")
        candidates = (
            [
                os.path.join(fsldir, "share", "fsl", "bin"),
                os.path.join(fsldir, "bin"),
            ]
            if fsldir
            else []
        )
        for bin_dir in candidates:
            if os.path.isfile(os.path.join(bin_dir, "topup")):
                monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
                break
    if shutil.which("topup") is None:
        msg = "FSL topup not found: put it on PATH or set $FSLDIR"
        if os.environ.get("UNDISTORTME_REQUIRE_FSL"):
            pytest.fail(msg)
        pytest.skip(msg)


# ---------------------------------------------------------------------------
# Local tree builder — 32x32x16 volumes (too big for the default 8x8x8
# tiny_nii fixture, but small enough to be fast with pervol.cnf)
# ---------------------------------------------------------------------------


def _make_large_nii(path: Path, shape=(32, 32, 16), seed: int = 0) -> str:
    """Write a real uncompressed NIfTI at *path* and return path as str."""
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    aff = np.eye(4)
    aff[0, 0] = 3.0
    aff[1, 1] = 3.0
    aff[2, 2] = 3.0
    nib.save(nib.Nifti1Image(data, aff), str(path))
    return str(path)


def _make_smoke_bids_run(tmp_path: Path):
    """Build a two-echo, one-volume BIDS run tree with 32x32x16 NIfTIs.

    Returns a dict compatible with the conftest bids_run factory output:
    output_dir, deriv_dir, subject, session, run, run_dir, tmp_path.
    """
    subject = "sub-01"
    session = "ses-01"
    run = "run-01"

    out = tmp_path / "out"
    deriv = tmp_path / "deriv"
    run_dir = out / subject / session / run
    run_dir.mkdir(parents=True)

    for e in [1, 2]:
        stem = f"{subject}_{session}_{run}_echo-{e}"
        sidecar = {
            "EchoNumber": e,
            "EchoTime": 0.07 + (e - 1) * 0.02,
            "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.106487,
            "PulseSequenceName": "ep_seg_35",
            "SeriesDescription": "ME_GRE",
            "ImageType": ["ORIGINAL", "PRIMARY", "M"],
            "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
        }
        (run_dir / f"{stem}.json").write_text(json.dumps(sidecar))
        _make_large_nii(run_dir / f"{stem}_1.nii", seed=e * 100 + 1)

    return {
        "output_dir": str(out),
        "deriv_dir": str(deriv),
        "subject": subject,
        "session": session,
        "run": run,
        "run_dir": str(run_dir),
        "tmp_path": str(tmp_path),
    }


# ---------------------------------------------------------------------------
# The smoke test
# ---------------------------------------------------------------------------


def test_topup_smoke(tmp_path, monkeypatch):
    """End-to-end: process_run with real FSL topup on a 32x32x16 phantom.

    Wall time with pervol.cnf: ~5-8 s (dominated by topup itself).
    """
    # --- environment setup ---
    # (PATH already carries FSL_BIN via the autouse _skip_if_no_fsl fixture.)
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")

    # --- build synthetic run ---
    run_info = _make_smoke_bids_run(tmp_path)

    # --- wire check_dict (real dispatchers, no recorder) ---
    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,  # keep single-threaded for reproducibility
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)

    # --- run the pipeline ---
    t0 = time.time()
    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        PERVOL_CNF,
    )
    elapsed = time.time() - t0
    print(f"\n[smoke] process_run elapsed: {elapsed:.1f}s")

    # --- assertions ---
    topup_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / "whole-volume"
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )

    corr_files = sorted(glob.glob(str(topup_dir / "*desc-undistorted*")))

    # Helpful failure message listing actual directory contents
    dir_contents = sorted(glob.glob(str(topup_dir / "*"))) if topup_dir.exists() else []
    assert len(corr_files) > 0, (
        f"No *desc-undistorted* files found in {topup_dir}.\n"
        f"Directory contents ({len(dir_contents)} items):\n"
        + "\n".join(f"  {p}" for p in dir_contents)
    )

    for path in corr_files:
        img = nib.load(path)
        data = img.get_fdata()
        assert np.isfinite(data).all(), f"Non-finite values in corrected output: {path}"

    print(f"[smoke] corrected files found: {[os.path.basename(f) for f in corr_files]}")


# ---------------------------------------------------------------------------
# Slice-mode smoke test — additionally requires slicenii + combinenii.
# Caught a real incompatibility: slicenii 0.2.0 release binaries slice fine
# but their combinenii cannot recombine the pipeline's per-slice outputs
# (needs > 0.2.0 with combinenii axis-guessing / sorting fixes).
# ---------------------------------------------------------------------------


def test_topup_slice_smoke(tmp_path, monkeypatch):
    """End-to-end slice mode: slicenii -> per-slice topup -> combinenii."""
    if shutil.which("slicenii") is None or shutil.which("combinenii") is None:
        msg = "slicenii/combinenii not found on PATH"
        if os.environ.get("UNDISTORTME_REQUIRE_FSL"):
            pytest.fail(msg)
        pytest.skip(msg)

    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_smoke_bids_run(tmp_path)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": True,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
        "jobs": 4,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    perslice_cnf = str(_REPO_ROOT / "configs" / "perslice_1.cnf")
    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        perslice_cnf,
    )

    assert cp.failed_commands == [], (
        f"{len(cp.failed_commands)} command(s) failed, first: {cp.failed_commands[0]}"
    )
    results_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / "per-slice"
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    recombined = sorted(glob.glob(str(results_dir / "*recombined*")))
    assert len(recombined) == 2, (
        f"expected 2 recombined volumes (one per echo) in {results_dir}, "
        f"found {len(recombined)}"
    )
    stray_slices = sorted(glob.glob(str(results_dir / "*_sv-*")))
    assert stray_slices == [], (
        "per-slice corrected files belong in the work tree, but derivatives "
        f"contains: {stray_slices[:3]}"
    )
    for path in recombined:
        img = nib.load(path)
        assert img.shape == (32, 32, 16), (
            f"recombined shape {img.shape} != input shape (32, 32, 16): {path}"
        )
        assert np.isfinite(img.get_fdata()).all()


# ---------------------------------------------------------------------------
# Denoised-estimation smoke test - runs the real undistortme-denoise
# subprocess, then real topup/applytopup, and checks the invariant that only
# the FIELD saw denoised data.
# ---------------------------------------------------------------------------


def test_topup_denoise_smoke(tmp_path, monkeypatch):
    """End-to-end with --denoise median: field from denoised, apply to originals."""
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_smoke_bids_run(tmp_path)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": "median",
        "jobs": 2,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        PERVOL_CNF,
    )

    assert cp.failed_commands == [], (
        f"{len(cp.failed_commands)} command(s) failed, first: {cp.failed_commands[0]}"
    )

    variant = "whole-volume_denoise-median"
    work_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme-work"
        / variant
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    suffix = dn.suffix_for("median")
    denoised = sorted(glob.glob(str(work_dir / f"*{suffix}.nii")))
    assert len(denoised) == 2, (
        f"expected one denoised copy per echo in {work_dir}, found {len(denoised)}"
    )

    # the denoised images really are different from their sources
    for path in denoised:
        source = Path(run_info["run_dir"]) / os.path.basename(path).replace(
            f"_{suffix}", ""
        )
        assert not np.allclose(
            nib.load(path).get_fdata(), nib.load(str(source)).get_fdata()
        )

    topup_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / variant
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    corr_files = sorted(glob.glob(str(topup_dir / "*desc-undistorted*")))
    assert len(corr_files) == 2, (
        f"expected 2 corrected outputs in {topup_dir}, found {len(corr_files)}"
    )
    for path in corr_files:
        assert np.isfinite(nib.load(path).get_fdata()).all()


# ---------------------------------------------------------------------------
# End-to-end identity: does output bv-N / sv-K carry input volume N / slice K?
#
# The structural argument is that it must: get_topup_command RETURNS the topup
# base and the same loop iteration hands that variable to
# get_applytopup_command, so the field applied to a (b-volume, slice) is the
# one estimated for it -- no filename is ever looked up, globbed or sorted to
# make that association. The one place names ARE matched by pattern is
# combinenii, which reassembles per-slice outputs.
#
# These tests stop arguing and measure it. Each volume and each slice is given
# a unique multiplicative tag in the data itself, real FSL runs, and the tag is
# read back from the outputs. A mismatch anywhere -- estimation, application,
# or recombination -- moves a tag and fails.
#
# Both use >= 10 b-volumes on purpose: bv- is NOT zero-padded, so bv-10 sorts
# before bv-2. If anything in the chain depended on lexicographic filename
# order, these are the cases that would expose it.
# ---------------------------------------------------------------------------


def _tagged_nii(
    path: Path, volume_tag: float, shape=(32, 32, 16), seed: int = 0
) -> str:
    """A phantom whose every slice carries a tag identifying (volume, slice).

    Voxel value = noise + volume_tag * 1000 + slice_index * 10, so the mean of
    any slice in the output identifies which input volume and which slice it
    came from. applytopup --method=jac rescales by the Jacobian and warps
    in-plane, neither of which reorders volumes or slices.
    """
    rng = np.random.default_rng(seed)
    data = rng.random(shape, dtype=np.float32)
    for k in range(shape[2]):
        data[:, :, k] += volume_tag * 1000.0 + k * 10.0
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = 3.0
    nib.save(nib.Nifti1Image(data, aff), str(path))
    return str(path)


def _make_tagged_run(
    tmp_path: Path, n_bvols: int, n_echoes: int = 2, bvals: list[int] | None = None
):
    """A BIDS run whose volumes are individually identifiable (see _tagged_nii).

    ``bvals`` selects the code path: more than one distinct value routes
    process_run to run_topup_diffusion_special, one value routes it to
    run_topup. Both build their own original_file_list, so both need proving.
    """
    subject, session, run = "sub-01", "ses-01", "run-01"
    out = tmp_path / "out"
    run_dir = out / subject / session / run
    run_dir.mkdir(parents=True)
    width = len(str(n_bvols))

    for e in range(1, n_echoes + 1):
        stem = f"{subject}_{session}_{run}_echo-{e}"
        (run_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "EchoNumber": e,
                    "EchoTime": 0.07 + (e - 1) * 0.02,
                    "PhaseEncodingDirection": "j-",
                    "TotalReadoutTime": 0.106487,
                    "PulseSequenceName": "ep_seg_35",
                    "SeriesDescription": "ME_GRE",
                    "ImageType": ["ORIGINAL", "PRIMARY", "M"],
                    "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                }
            )
        )
        values = bvals if bvals is not None else ([0] + [1000] * (n_bvols - 1))
        (run_dir / f"{stem}.bval").write_text(" ".join(str(b) for b in values))
        for i in range(1, n_bvols + 1):
            _tagged_nii(
                run_dir / f"{stem}_{str(i).zfill(width)}.nii",
                volume_tag=float(i),
                seed=e * 100 + i,
            )

    return {
        "output_dir": str(out),
        "deriv_dir": str(tmp_path / "deriv"),
        "subject": subject,
        "session": session,
        "run": run,
        "run_dir": str(run_dir),
        "tmp_path": str(tmp_path),
    }


def _volume_tag_of(path) -> int:
    """Recover the volume tag from a corrected image (see _tagged_nii)."""
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    # slice tags average out; the volume tag dominates the whole-volume mean
    mean_slice_tag = (data.shape[2] - 1) * 10.0 / 2.0
    return int(round((float(data.mean()) - mean_slice_tag - 0.5) / 1000.0))


def _slice_tag_of(plane) -> int:
    """Recover the slice tag from one plane of a corrected volume."""
    return int(round((float(plane.mean()) % 1000.0 - 0.5) / 10.0))


def test_whole_volume_outputs_carry_their_own_input_volume(tmp_path, monkeypatch):
    """bv-N must be the correction of input volume N, for N past 9.

    12 volumes, so bv-10/11/12 exist alongside bv-1/2: any lexicographic
    filename ordering in the chain would mis-assign them.
    """
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_tagged_run(tmp_path, n_bvols=12, n_echoes=2)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
        "jobs": 8,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        PERVOL_CNF,
        None,
        3000,
    )

    assert cp.failed_commands == [], cp.failed_commands
    out_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / "whole-volume"
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    corrected = sorted(glob.glob(str(out_dir / "*desc-undistorted*_echo-1_*.nii")))
    assert len(corrected) == 12, f"expected 12 outputs, got {len(corrected)}"

    for path in corrected:
        declared = int(os.path.basename(path).split("_bv-")[1].split("_")[0])
        measured = _volume_tag_of(path)
        assert measured == declared, (
            f"{os.path.basename(path)} says b-volume {declared} but carries "
            f"the data of input volume {measured}"
        )


def test_per_slice_outputs_carry_their_own_input_slice(tmp_path, monkeypatch):
    """After slicing, per-slice TOPUP and recombination, slice K of the output
    must still be slice K of the input -- and of the right b-volume."""
    if shutil.which("slicenii") is None or shutil.which("combinenii") is None:
        msg = "slicenii/combinenii not found on PATH"
        if os.environ.get("UNDISTORTME_REQUIRE_FSL"):
            pytest.fail(msg)
        pytest.skip(msg)

    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_tagged_run(tmp_path, n_bvols=11, n_echoes=2)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": True,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
        "jobs": 8,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    perslice_cnf = str(_REPO_ROOT / "configs" / "perslice_1.cnf")
    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        perslice_cnf,
        None,
        3000,
    )

    assert cp.failed_commands == [], cp.failed_commands
    out_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / "per-slice"
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    recombined = sorted(glob.glob(str(out_dir / "*recombined*echo-1.nii")))
    assert len(recombined) == 11, f"expected 11 volumes, got {len(recombined)}"

    for path in recombined:
        declared = int(os.path.basename(path).split("_bv-")[1].split("_")[0])
        data = np.asarray(nib.load(path).dataobj, dtype=np.float64)
        assert data.shape == (32, 32, 16)
        assert _volume_tag_of(path) == declared, (
            f"{os.path.basename(path)} carries another b-volume's data"
        )
        for k in range(data.shape[2]):
            assert _slice_tag_of(data[:, :, k]) == k, (
                f"{os.path.basename(path)} slice {k} carries the data of "
                f"input slice {_slice_tag_of(data[:, :, k])}"
            )


def test_single_bval_outputs_carry_their_own_input_volume(tmp_path, monkeypatch):
    """The same identity proof for the run_topup path.

    One distinct b-value routes process_run to run_topup rather than
    run_topup_diffusion_special; that function builds its own
    original_file_list, so the pairing has to be proved there separately. 12
    volumes again, to exercise bv-10..12 alongside bv-1..9.
    """
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    run_info = _make_tagged_run(tmp_path, n_bvols=12, n_echoes=2, bvals=[1000] * 12)

    check_dict = {
        "dcm2niix": False,
        "topup": True,
        "topup_multithread": False,
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
        "jobs": 8,
        "oversubscribe": 1.0,
    }
    monkeypatch.setattr(cp, "check_dict", check_dict, raising=False)
    monkeypatch.setattr(cp, "failed_commands", [])

    cp.process_run(
        run_info["subject"],
        run_info["session"],
        run_info["run"],
        run_info["output_dir"],
        run_info["deriv_dir"],
        PERVOL_CNF,
        None,
        3000,
    )

    assert cp.failed_commands == [], cp.failed_commands
    out_dir = (
        Path(run_info["deriv_dir"])
        / "undistortme"
        / "whole-volume"
        / run_info["subject"]
        / run_info["session"]
        / run_info["run"]
    )
    corrected = sorted(glob.glob(str(out_dir / "*desc-undistorted*_echo-1_*.nii")))
    assert len(corrected) == 12, f"expected 12 outputs, got {len(corrected)}"

    for path in corrected:
        declared = int(os.path.basename(path).split("_bv-")[1].split("_")[0])
        measured = _volume_tag_of(path)
        assert measured == declared, (
            f"{os.path.basename(path)} says b-volume {declared} but carries "
            f"the data of input volume {measured}"
        )
