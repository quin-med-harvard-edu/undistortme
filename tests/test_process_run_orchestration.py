"""Orchestration-level characterization tests for ``process_run``.

WHAT THIS PINS
--------------
``corr_pipeline.process_run`` is the top-level per-run driver.  Every external
command it would run is funnelled through exactly three batch dispatchers:

    parallel_bash_commands(commands, description)
    serial_bash_commands(commands, description)

These tests install a *capture seam* at those three dispatchers (the
``recorder`` / ``slicing_recorder`` fixtures in conftest) so that process_run
executes its full decision logic and command assembly but NOTHING is run.  The
captured ``(description, [commands])`` batches are normalized to text (absolute
paths replaced by ``<TMP>`` / ``<OUT>`` / ``<DERIV>`` / ``<CMATCH>``
placeholders) and pinned as golden snapshots in ``tests/_snapshots/``.

WHY DISPATCH-LEVEL
------------------
All command execution converges on these three functions, so patching them
captures the complete command surface of a run with a single, stable seam.  We
deliberately do NOT patch ``run_bash_command``: parallel dispatch uses a
ProcessPoolExecutor and patches would not reach the worker processes.  We also
no-op ``cp.shuffle`` (the pipeline shuffles work lists) so batch order is the
deterministic source order.

THESE GOLDENS PIN CURRENT BEHAVIOR
----------------------------------
The snapshots encode the pipeline EXACTLY as it behaves today, including any
quirks (e.g. the always-present ``bval`` column meaning the "bval not in
columns" branch is never taken; the ``--inindex`` echo remap under contrast
match).  They are a refactor safety net, not an assertion that the current
behavior is correct.

SNAPSHOT WORKFLOW
-----------------
Regenerate goldens after an intentional change with::

    UPDATE_SNAPSHOTS=1 PY -m pytest tests/test_process_run_orchestration.py

(each test writes its golden and is skipped).  Then run without the env var to
compare.  A mismatch fails with a unified diff.

DEVIATIONS FROM THE ORIGINAL CASE TABLE (verified against source)
-----------------------------------------------------------------
* topup_3echo_avg / topup_2echo_avg / topup_single_bval reach ``run_topup`` via
  the ``else`` branch of process_run's bval dispatch, NOT the
  ``"bval" not in curr_df.columns`` branch: get_run_info ALWAYS emits a "bval"
  column (None for the averages case), so that first branch is effectively dead
  for dataframes coming from get_run_info.  The destination (run_topup) and the
  resulting commands are unchanged, so the table's intent holds.
* topup_single_bval: bvals [1000, 1000] yields ``nunique() == 1`` so the
  ``> 1`` diffusion branch is false and it falls to run_topup, producing one
  per-bvol command for EACH of the two (equal-bval) volumes -> 2 commands.
* topup_with_mask: there is no slicing, so ``slicing_recorder`` only materialises
  the masked outputs; no slicenii batch appears.  The masked .nii files are not
  strictly required by the whole-volume topup path (inputs are only string-
  interpolated into commands), but we write them per the harness contract.

ORDERING
--------
``find_files`` sorts its glob results (pinned by
TestFindFilesOrdering::test_results_are_sorted), so batch-internal ordering
is deterministic across hosts and filesystems.  A snapshot ordering diff is
therefore a REAL behavior change — investigate it; do not regenerate goldens
to make it go away.
"""

import os

import pytest
from conftest import _reps
from conftest import _run_process_run as _run

from undistortme import pipeline as cp

CONFIG = "b02b0.cnf"


# ===========================================================================
# TOPUP whole-volume paths
# ===========================================================================


def test_topup_3echo_avg(check_dict, bids_run, recorder, normalize, assert_snapshot):
    """topup, 3 echoes, no bval, 1 average -> run_topup, single per-bvol batch."""
    check_dict["topup"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    _run(info)
    assert_snapshot("topup_3echo_avg", normalize(recorder, _reps(info)))


def test_topup_2echo_avg(check_dict, bids_run, recorder, normalize, assert_snapshot):
    """topup, 2 echoes, no bval -> run_topup with two applytopup per command."""
    check_dict["topup"] = True
    info = bids_run(n_echoes=2, n_avgs=1)
    _run(info)
    assert_snapshot("topup_2echo_avg", normalize(recorder, _reps(info)))


def test_topup_3echo_match(check_dict, bids_run, recorder, normalize, assert_snapshot):
    """topup+match, 3 echoes -> geomean contrast-match command incl <CMATCH>.

    Under match, applytopup remaps odd echoes to --inindex=1 and even echoes to
    --inindex=2 (so echo-3 -> --inindex=1) — pinned quirk.
    """
    check_dict["topup"] = True
    check_dict["match"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    _run(info)
    text = normalize(recorder, _reps(info))
    assert "<CMATCH>" in text
    assert_snapshot("topup_3echo_match", text)


def test_topup_twoecho_filter(
    check_dict, bids_run, recorder, normalize, assert_snapshot
):
    """topup+two_echo, 4 echoes -> filtered to echoes <3, 'two-echo' in paths."""
    check_dict["topup"] = True
    check_dict["two_echo"] = True
    info = bids_run(n_echoes=4, n_avgs=1)
    _run(info)
    text = normalize(recorder, _reps(info))
    assert "two-echo" in text
    # filtered: only echo-1 / echo-2 applytopup outputs appear
    assert "echo-3" not in text
    assert "echo-4" not in text
    assert_snapshot("topup_twoecho_filter", text)


def test_topup_diffusion_special(
    check_dict, bids_run, recorder, normalize, assert_snapshot
):
    """topup, 2 echoes, bvals [0,1000,2000] -> run_topup_diffusion_special.

    Two batches: per-bvol topup below the b cutoff, then a separate
    high-b applytopup batch.  The closest low-b volume chosen for the high-b
    field is data-dependent (find_closest_volume_nmi over seeded tiny niftis)
    but fully deterministic, so the snapshot pins whichever it selects.
    """
    check_dict["topup"] = True
    info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])
    _run(info, cutoff=1000)
    text = normalize(recorder, _reps(info))
    descs = [d for d, _ in recorder]
    assert "running TOPUP per-bvol below b cutoff" in descs
    assert "ApplyTOPUP Commands for high b diffusion" in descs
    assert_snapshot("topup_diffusion_special", text)


def test_topup_single_bval(check_dict, bids_run, recorder, normalize, assert_snapshot):
    """topup, 2 echoes, bvals [1000,1000] -> nunique==1 falls to run_topup.

    Two equal-bval volumes -> two per-bvol topup commands in one batch.
    """
    check_dict["topup"] = True
    info = bids_run(n_echoes=2, bvals=[1000, 1000])
    _run(info, cutoff=1000)
    text = normalize(recorder, _reps(info))
    assert len(recorder) == 1
    assert recorder[0][0] == "topup commands per-bvol"
    assert len(recorder[0][1]) == 2
    assert_snapshot("topup_single_bval", text)


# ===========================================================================
# Masked whole-volume path
# ===========================================================================

# ===========================================================================
# Masking
# ===========================================================================


def test_topup_with_mask(
    check_dict,
    bids_run,
    slicing_recorder,
    normalize,
    assert_snapshot,
    tmp_path,
    tiny_nii,
):
    """topup + mask: masking batch first, then topup on the masked niis.

    Mask is discovered in mask_dir by find_files using the patterns
    ``{sub}_{ses}_{run}*-label.nii`` and ``{sub}_{ses}_{run}*mask.nii``; we
    place a ``..._desc-brain_mask.nii`` matching the second pattern.
    """
    check_dict["topup"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tiny_nii(
        mask_dir
        / f"{info['subject']}_{info['session']}_{info['run']}_desc-brain_mask.nii"
    )
    _run(info, mask_dir=str(mask_dir))
    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    assert descs[0] == "masking commands"
    assert "topup commands per-bvol" in descs
    assert_snapshot("topup_with_mask", text)


# ===========================================================================
# Per-slice paths
# ===========================================================================


def test_topup_3echo_slice(
    check_dict, bids_run, slicing_recorder, normalize, assert_snapshot
):
    """topup+slice: slicenii -> per-slice topup -> combinenii."""
    check_dict["topup"] = True
    check_dict["slice"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    _run(info)
    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    assert descs[0] == "slicenii commands"
    assert "topup commands per-slice" in descs
    assert "combinenii commands" in descs
    assert_snapshot("topup_3echo_slice", text)


def test_topup_3echo_slice_match(
    check_dict, bids_run, slicing_recorder, normalize, assert_snapshot
):
    """topup+slice+match: the highest-fanout advertised configuration.

    Per slice: geomean contrast-match -> merge -> topup, then combinenii.
    Pins that each slice gets its own desc-cm_geomean file (distinct sv-).
    """
    check_dict["topup"] = True
    check_dict["slice"] = True
    check_dict["match"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    _run(info)
    text = normalize(slicing_recorder, _reps(info))
    cm_lines = [l for l in text.splitlines() if "<CMATCH>" in l]
    assert len(cm_lines) >= 2, "expected per-slice contrast-match commands"
    sv_tags = {
        l.split("_sv-")[1].split("_")[0].split(".")[0] for l in cm_lines if "_sv-" in l
    }
    assert len(sv_tags) >= 2, "contrast-match outputs must differ per slice"
    assert_snapshot("topup_3echo_slice_match", text)


def test_topup_3echo_slice_mask(
    check_dict,
    bids_run,
    slicing_recorder,
    normalize,
    assert_snapshot,
    tmp_path,
    tiny_nii,
):
    """topup+slice+mask: masked copies and slices share one work dir.

    Pins that the masking batch runs first, slicenii consumes the masked
    files, and the shared work_dir layout produces no colliding filenames.
    """
    check_dict["topup"] = True
    check_dict["slice"] = True
    info = bids_run(n_echoes=3, n_avgs=1)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tiny_nii(
        mask_dir
        / f"{info['subject']}_{info['session']}_{info['run']}_desc-brain_mask.nii"
    )
    _run(info, mask_dir=str(mask_dir))
    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    assert descs[0] == "masking commands"
    assert "slicenii commands" in descs
    assert_snapshot("topup_3echo_slice_mask", text)


# ===========================================================================
# Denoised variants (estimation-only; see tests/test_denoise_pipeline.py for
# the invariant these snapshots illustrate)
# ===========================================================================


def test_topup_3echo_slice_match_denoise(
    check_dict, bids_run, slicing_recorder, normalize, assert_snapshot
):
    """The advertised per-slice+match config with median denoising.

    Pins the whole surface: a denoising batch per echo, slicenii on both the
    originals and the denoised copies, contrast-match/merge/topup on the
    denoised slices, and applytopup on the original ones.
    """
    check_dict["topup"] = True
    check_dict["slice"] = True
    check_dict["match"] = True
    check_dict["denoise"] = "median"
    info = bids_run(n_echoes=3, n_avgs=1)
    _run(info)
    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    assert descs[0] == "denoising commands"
    assert "denoise-median" in text
    assert_snapshot("topup_3echo_slice_match_denoise", text)


# ===========================================================================
# BIDS snapshot reuse: staged input must drive the UNCHANGED core.
#
# A BIDS dataset equivalent to the legacy fixtures is staged with the
# ingest adapter, process_run is pointed at the staging tree, and the
# captured command batches must equal the existing goldens VERBATIM: the
# staging root simply takes the <OUT> (input root) placeholder.  The
# goldens are read directly and never (re)generated by these tests.
# ===========================================================================


def _golden_text(name):
    """Read a committed golden snapshot directly (no regen path)."""
    path = os.path.join(os.path.dirname(__file__), "_snapshots", f"{name}.txt")
    with open(path, "r") as f:
        return f.read()


def _stage(tmp_path, info):
    """Stage a BIDS dataset; return an info dict for _reps()/_run().

    The staging tree is process_run's input root, so it fills the
    ``output_dir`` slot and normalizes to ``<OUT>`` like any input root.
    """
    from undistortme import bids as ub

    stage_dir = str(tmp_path / "stage")
    staged = ub.stage_bids(info["bids_dir"], stage_dir)
    assert staged == [("sub-01", "ses-01", "run-01")]
    return {
        "subject": "sub-01",
        "session": "ses-01",
        "run": "run-01",
        "output_dir": stage_dir,
        "deriv_dir": str(tmp_path / "deriv"),
        "tmp_path": str(tmp_path),
    }


def test_bids_staged_3echo_avg_matches_golden(
    check_dict, tmp_path, bids_dwi_dataset, recorder, normalize
):
    """Staged 3-echo/1-volume BIDS input reproduces topup_3echo_avg
    verbatim: strongest proof the core is untouched."""
    check_dict["topup"] = True
    info = _stage(tmp_path, bids_dwi_dataset(n_echoes=3, n_vols=1))
    _run(info)
    text = normalize(recorder, _reps(info))
    assert text == _golden_text("topup_3echo_avg")


def test_bids_staged_diffusion_special_matches_golden(
    check_dict, tmp_path, bids_dwi_dataset, recorder, normalize
):
    """Staged 2-echo diffusion BIDS input (bvals 0/1000/2000) reproduces
    topup_diffusion_special verbatim.

    The factory uses seed_for(echo, vol) — the same data contract as the
    legacy _make_bids_run — so the NMI-based closest-low-b selection picks
    the same volume and the high-b applytopup batch matches too.
    """
    check_dict["topup"] = True
    info = _stage(tmp_path, bids_dwi_dataset(n_echoes=2, bvals=[0, 1000, 2000]))
    _run(info, cutoff=1000)
    text = normalize(recorder, _reps(info))
    assert text == _golden_text("topup_diffusion_special")


# ===========================================================================
# Gate cases: process_run returns early, nothing is dispatched.
# ===========================================================================


def test_gate_single_volume(check_dict, bids_run, recorder):
    """1 echo, 1 volume -> fewer than two volumes -> early return, no dispatch."""
    check_dict["topup"] = True
    info = bids_run(n_echoes=1, n_avgs=1)
    _run(info)
    assert recorder == []


def test_gate_match_lt3echo(check_dict, bids_run, recorder):
    """match + only 2 echoes -> contrast match needs >=3 echoes -> early return."""
    check_dict["topup"] = True
    check_dict["match"] = True
    info = bids_run(n_echoes=2, n_avgs=1)
    _run(info)
    assert recorder == []
