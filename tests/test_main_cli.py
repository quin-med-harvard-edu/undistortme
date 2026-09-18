"""Tests for main(): argument wiring, subject selection, --fix-names gating,
and the failure exit status.

main() is the only entry point outside users call; these tests stub
check_binaries (no real binaries needed) and record process_run calls.
"""

import argparse
import os

import pytest

from undistortme import pipeline as cp

CONFIG = "b02b0.cnf"


def _make_args(**over):
    """A Namespace with every attribute main() reads, defaults all-off."""
    ns = argparse.Namespace(
        input_dir="./sourcedata",
        subject_dir=None,
        output_dir="./",
        derivdir="./derivatives",
        workdir=None,
        config_file=CONFIG,
        run_topup=False,
        run_dcm2niix=False,
        slice=False,
        matchcontrast=False,
        dryrun=False,
        twoecho=False,
        maskdir=None,
        fix_names=False,
        cutoff="1000",
        jobs=None,
        oversubscribe=None,
        denoise=None,
        bids=None,
        echo_entity="echo",
        echo_from_acq=None,
    )
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub binaries/process_run; return helpers to build trees and run main."""
    calls = []

    monkeypatch.setattr(cp, "failed_commands", [])
    monkeypatch.setattr(
        cp, "check_binaries", lambda dcm, topup, slc: (dcm, topup, False, slc)
    )
    monkeypatch.setattr(cp, "process_run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(cp, "shuffle", lambda seq: None)

    def build_run(subject, session="ses-01", run="run-01"):
        (tmp_path / subject / session / run).mkdir(parents=True)

    def run_main(**over):
        over.setdefault("output_dir", str(tmp_path))
        over.setdefault("derivdir", str(tmp_path / "derivatives"))
        monkeypatch.setattr(cp, "get_args", lambda: _make_args(**over))
        cp.main()

    return argparse.Namespace(
        calls=calls, build_run=build_run, run_main=run_main, root=tmp_path
    )


# ===========================================================================
# Argument wiring
# ===========================================================================


def test_process_run_positional_wiring(harness):
    """maskdir / cutoff / workdir arrive in the right positional slots."""
    harness.build_run("sub-01")
    harness.run_main(run_topup=True, maskdir="MASKS", cutoff="850", workdir="WORKROOT")
    assert len(harness.calls) == 1
    (subject, session, run, out, deriv, config, mask_dir, cutoff, work_root) = (
        harness.calls[0]
    )
    assert (subject, session, run) == ("sub-01", "ses-01", "run-01")
    assert config == CONFIG
    assert mask_dir == "MASKS"
    assert cutoff == 850  # int() applied to the string default type
    assert work_root == "WORKROOT"


def test_jobs_and_oversubscribe_reach_check_dict(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True, jobs=3, oversubscribe=1.5)
    assert cp.check_dict["jobs"] == 3
    assert cp.check_dict["oversubscribe"] == 1.5


def test_denoise_method_reaches_check_dict(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True, denoise="median")
    assert cp.check_dict["denoise"] == "median"


def test_denoise_defaults_to_off(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True)
    assert cp.check_dict["denoise"] is None


def test_jobs_default_when_unset(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True)
    assert cp.check_dict["jobs"] == cp.default_jobs()
    assert cp.check_dict["oversubscribe"] == cp.DEFAULT_OVERSUBSCRIBE


def test_twoecho_mutually_exclusive_with_matchcontrast(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True, twoecho=True, matchcontrast=True)
    assert cp.check_dict["two_echo"] is False
    harness.calls.clear()
    harness.run_main(run_topup=True, twoecho=True)
    assert cp.check_dict["two_echo"] is True


def test_subject_dir_limits_to_one_subject(harness):
    harness.build_run("sub-01")
    harness.build_run("sub-02")
    harness.run_main(run_topup=True, subject_dir="sub-02")
    assert [c[0] for c in harness.calls] == ["sub-02"]


def test_no_subjects_exits_nonzero(harness):
    with pytest.raises(SystemExit):
        harness.run_main(run_topup=True)


# ===========================================================================
# --fix-names gating (the one in-place destructive operation)
# ===========================================================================


def test_input_tree_untouched_without_fix_names(harness):
    harness.build_run("sub-0.5")
    harness.run_main(run_topup=True)
    assert (harness.root / "sub-0.5").exists(), (
        "input tree was mutated without --fix-names"
    )
    assert harness.calls[0][0] == "sub-0.5"


def test_fix_names_renames_dotted_subject(harness):
    harness.build_run("sub-0.5")
    harness.run_main(run_topup=True, fix_names=True)
    assert not (harness.root / "sub-0.5").exists()
    renamed = harness.calls[0][0]
    assert "." not in renamed
    assert (harness.root / renamed).exists()


# ===========================================================================
# --bids routing
# ===========================================================================


def _staging_root(harness):
    """Where main() stages BIDS input given the harness's default derivdir."""
    return os.path.join(
        str(harness.root / "derivatives"), "undistortme-work", "bids-staged"
    )


def test_bids_flag_stages_and_reroutes_input(harness, bids_dwi_dataset):
    """--bids stages the dataset and process_run gets the staging tree as
    its input root; the synthesized (sub, ses, run-token) triple is used."""
    info = bids_dwi_dataset(n_echoes=2)
    harness.run_main(run_topup=True, bids=info["bids_dir"])
    staging_root = _staging_root(harness)
    assert len(harness.calls) == 1
    (subject, session, run, out, deriv, config, mask_dir, cutoff, work_root) = (
        harness.calls[0]
    )
    assert (subject, session, run) == ("sub-01", "ses-01", "run-01")
    assert out == staging_root
    assert deriv == str(harness.root / "derivatives")
    # staged per-volume layout exists where process_run will look
    assert os.path.exists(
        os.path.join(
            staging_root,
            "sub-01",
            "ses-01",
            "run-01",
            "sub-01_ses-01_run-01_echo-1_1.nii",
        )
    )


def test_bids_staging_respects_workdir(harness, bids_dwi_dataset):
    info = bids_dwi_dataset(n_echoes=2)
    workdir = str(harness.root / "WORKROOT")
    harness.run_main(run_topup=True, bids=info["bids_dir"], workdir=workdir)
    assert harness.calls[0][3] == os.path.join(workdir, "bids-staged")


def test_bids_conflicts_with_dcm2niix(monkeypatch, tmp_path, capsys):
    """--bids and -d are argparse-level mutually exclusive (exit code 2)."""
    monkeypatch.setattr("sys.argv", ["undistortme", "--bids", str(tmp_path), "-d"])
    with pytest.raises(SystemExit) as excinfo:
        cp.get_args()
    assert excinfo.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_bids_empty_dataset_exits_nonzero(harness, tmp_path):
    empty = tmp_path / "empty-bids"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        harness.run_main(run_topup=True, bids=str(empty))
    assert excinfo.value.code == 1


def test_bids_staging_error_is_reported_not_raised(
    harness, tmp_path, monkeypatch, capsys
):
    """A malformed dataset (e.g. .bval count != volume count) raises ValueError
    inside stage_bids; the CLI must report it and exit 1, not traceback."""

    def _bad_dataset(*args, **kwargs):
        raise ValueError(".bval lists 30 volumes but the image has 31")

    monkeypatch.setattr(cp.bids_input, "stage_bids", _bad_dataset)
    with pytest.raises(SystemExit) as excinfo:
        harness.run_main(run_topup=True, bids=str(tmp_path))
    assert excinfo.value.code == 1
    assert "ERROR:" in capsys.readouterr().out


def test_bids_staging_runs_even_under_dryrun(harness, bids_dwi_dataset, capsys):
    """Documented decision (report risk 8): staging is real Python I/O and
    is NOT skipped by --dryrun, so downstream dryrun output references real
    files; a NOTE says so."""
    info = bids_dwi_dataset(n_echoes=2)
    harness.run_main(run_topup=True, bids=info["bids_dir"], dryrun=True)
    staging_root = _staging_root(harness)
    assert os.path.exists(
        os.path.join(
            staging_root,
            "sub-01",
            "ses-01",
            "run-01",
            "sub-01_ses-01_run-01_echo-1_1.nii",
        )
    )
    assert "--dryrun does not skip BIDS staging" in capsys.readouterr().out


def test_bids_shaped_output_dir_prints_hint_only(harness, capsys):
    """A BIDS-looking -o tree without --bids gets a printed hint, never a
    silent mode switch."""
    (harness.root / "sub-01" / "dwi").mkdir(parents=True)
    harness.run_main(run_topup=True)
    out = capsys.readouterr().out
    assert "HINT" in out and "--bids" in out
    # legacy flow proceeded (and found no run dirs to process)
    assert harness.calls == []


def test_legacy_tree_prints_no_hint(harness, capsys):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True)
    assert "HINT" not in capsys.readouterr().out


# ===========================================================================
# Failure exit status
# ===========================================================================


def test_failed_commands_exit_nonzero(harness):
    harness.build_run("sub-01")
    cp.failed_commands.append(("topup ...", 1))
    with pytest.raises(SystemExit) as excinfo:
        harness.run_main(run_topup=True)
    assert excinfo.value.code == 1


def test_clean_run_exits_normally(harness):
    harness.build_run("sub-01")
    harness.run_main(run_topup=True)  # no SystemExit raised
    assert cp.failed_commands == []
