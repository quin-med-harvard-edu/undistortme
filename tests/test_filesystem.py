"""Characterization tests for filesystem-layer functions in corr_pipeline.py.

These tests pin the CURRENT behavior exactly as-is — ahead of a refactor.
Do NOT modify corr_pipeline.py; do NOT change expected values to match a
"corrected" implementation — the source wins.

Source behaviors confirmed before writing (READ source before each class):

- find_files (L260):
    * Thin wrapper around glob.glob with recursive=True.
    * Returns a plain list; order follows glob (OS-dependent, not sorted).

- check_for_bvals (L443):
    * Reads dir_dict["current_dir"] for *.bval files via find_files.
    * With bval file: returns (len(bvals.split()), raw file contents, [path]).
      Raw contents are NOT stripped (the source reads and splits to count, but
      returns the raw string from bval_file.read()).
    * Without bval file: returns (1, "", []).

- make_acq_params (L460):
    * phase_sign = (-1)**(echo + 1):  echo1→+1, echo2→-1, echo3→+1, echo4→-1.
    * phase_dir "j*" → "0 {sign} 0 {readout}\n"
    * phase_dir "i*" → "{sign} 0 0 {readout}\n"
    * phase_dir "k*" → "0 0 {sign} {readout}\n"
    * match=True AND nunique(echo) > 2 → only echoes [1, 2] written.
    * Unrecognised phase_dir → exit(1).
    * Pre-existing acqparams file → returned immediately without rewrite.

- get_echo_info (L496):
    * bval_path = json_path.replace("json", "bval")  (substring, not suffix).
    * No bval file: bvals=[None], num_bvals=0; falls back to glob for "_*.nii".
    * Nii naming: json_path.replace(".json", f"_{bb}.nii") where
      bb = str(i).zfill(len(str(num_bvals))), i in range(1, num_bvals+1).
    * Returns 16-key dict (see TestGetEchoInfo for exact keys).
    * phase_sign = (-1)**(echo_num + 1); IS stored in the returned dict at key
      "phase_sign".
    * orientation [1,0,0,0,1,0] → "axial", orientation_num=2.
    * orientation [1,0,0,0,0,-1] → "coronal", orientation_num=1.
    * orientation [0,1,0,0,0,-1] → "sagittal", orientation_num=0.
    * other/undefined → "unknown", orientation_num=3.
    * Multiple glob candidates when expected nii missing → FileNotFoundError.

- set_dirs (L387):
    * Reads check_dict["slice"], ["match"], ["mask"].
    * Missing current_dir → exit(1).
    * Always sets "current_dir", "ave_dir", "topup_dir",
      "inner_dir".
    * slice=True → also sets "slice_dir"; inner_dir starts as "per-slice".
    * slice=False → inner_dir starts as "whole-volume".
    * match=True → appends "_contrast-matched" to inner_dir.
    * mask=True → also sets "masked_dir"; appends "_masked" to inner_dir.

- get_run_info (L1071):
    * Signature: get_run_info(curr_dir, subject, session, run) → pd.DataFrame.
    * Calls get_echo_info for each *.json in curr_dir.
    * No bval: rows have bval=None, bvol_num enumerated 1..N by average.
    * With bval: rows have bval=int, bvol_num enumerated 1..N by bval position.
    * 17 columns: subject, session, run, echo, bvol_num, bval, nii, echo_time,
      readout_time, phase_dir, slice_orientation, orientation_num,
      pulse_sequence, series_description, image_type, orientation_list,
      orientation_rounded.

- rename_entity (L175):
    * extension = old_path.suffixes[-1] if old_path.suffixes else "".
    * new_name  = old_path.stem.replace(".", "").
    * Returns new Path if names differ, None if the name already has no dots.
    * NOTE: for a path like "a.b.c.nii", stem="a.b.c", suffixes=[".b",".c",".nii"],
      so extension=".nii", new_name="abc", result="abc.nii".

- rename_entities (L147):
    * Renames root_dir itself first (dots removed from its own stem).
    * Then walks the tree (topdown=False) renaming files, then queued dirs.
    * Returns Path(new_name).stem (string, not Path).
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from undistortme import pipeline as cp

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
SUBJ = "sub-01"
SESS = "ses-baseline"
RUN = "run-01"
READOUT = 0.106487

# Path to committed golden files
_GOLDEN = Path(__file__).parent / "_golden"


# ===========================================================================
# Helpers
# ===========================================================================


def _write_json(
    path: Path,
    echo_num: int,
    phase_dir: str,
    readout: float = READOUT,
    orientation: list | None = None,
) -> Path:
    """Write a minimal JSON sidecar that get_echo_info can read."""
    data = {
        "EchoNumber": echo_num,
        "EchoTime": 0.07,
        "PhaseEncodingDirection": phase_dir,
        "TotalReadoutTime": readout,
        "PulseSequenceName": "ep_seg_35",
        "SeriesDescription": "ME_GRE",
        "ImageType": ["ORIGINAL", "PRIMARY", "M"],
        "ImageOrientationPatientDICOM": orientation or [1, 0, 0, 0, 1, 0],
    }
    path.write_text(json.dumps(data))
    return path


def _make_sidecar_tree(
    run_dir: Path,
    stem: str,
    *,
    n_vols: int,
    bvals: list[str] | None = None,
    sidecar_overrides: dict | None = None,
    echo_num: int = 1,
    phase_dir: str = "j-",
    orientation: list | None = None,
) -> Path:
    """Create a JSON sidecar and the expected nii files for get_echo_info.

    Nii naming follows the source convention:
        {stem}_{bb}.nii  where bb = str(i).zfill(len(str(n_vols))), i in 1..n_vols

    If bvals is given, also writes a .bval file alongside the JSON
    (using the substring-replace convention: stem.bval).

    Returns the Path to the written JSON sidecar.
    """
    json_path = run_dir / f"{stem}.json"

    data: dict = {
        "EchoNumber": echo_num,
        "EchoTime": 0.07,
        "PhaseEncodingDirection": phase_dir,
        "TotalReadoutTime": READOUT,
        "PulseSequenceName": "ep_seg_35",
        "SeriesDescription": "ME_GRE",
        "ImageType": ["ORIGINAL", "PRIMARY", "M"],
        "ImageOrientationPatientDICOM": orientation or [1, 0, 0, 0, 1, 0],
    }
    if sidecar_overrides:
        data.update(sidecar_overrides)
    json_path.write_text(json.dumps(data))

    if bvals is not None:
        bval_path = run_dir / f"{stem}.bval"
        bval_path.write_text(" ".join(bvals))

    num_digits = len(str(n_vols))
    for i in range(1, n_vols + 1):
        bb = str(i).zfill(num_digits)
        (run_dir / f"{stem}_{bb}.nii").touch()

    return json_path


def _make_run_df(
    echoes: list[int], phase_dir: str, readout: float = READOUT
) -> pd.DataFrame:
    """Return a minimal run_df with echo, phase_dir, readout_time columns."""
    rows = [
        {"echo": e, "phase_dir": phase_dir, "readout_time": readout} for e in echoes
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# find_files
# ===========================================================================


class TestFindFiles:
    """find_files is a sorted glob wrapper (see TestFindFilesOrdering)."""

    def test_returns_matching_files(self, tmp_path):
        """Files matching the pattern are included in the result."""
        (tmp_path / "alpha.nii").touch()
        (tmp_path / "beta.nii").touch()
        (tmp_path / "gamma.json").touch()

        result = cp.find_files(str(tmp_path), "*.nii")

        result_names = {Path(p).name for p in result}
        assert result_names == {"alpha.nii", "beta.nii"}

    def test_non_matching_files_excluded(self, tmp_path):
        """Files not matching the pattern are absent from the result."""
        (tmp_path / "alpha.nii").touch()
        (tmp_path / "gamma.json").touch()

        result = cp.find_files(str(tmp_path), "*.bval")
        assert result == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """An empty directory yields an empty list."""
        result = cp.find_files(str(tmp_path), "*.nii")
        assert result == []

    def test_returns_list_type(self, tmp_path):
        """Return type is list (from glob.glob)."""
        result = cp.find_files(str(tmp_path), "*.nii")
        assert isinstance(result, list)

    def test_recursive_finds_nested_files(self, tmp_path):
        """With **/ pattern, files in subdirectories are also found."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.nii").touch()

        result = cp.find_files(str(tmp_path), "**/*.nii")
        assert any(Path(p).name == "deep.nii" for p in result)


# ===========================================================================
# check_for_bvals
# ===========================================================================


class TestCheckForBvals:
    """check_for_bvals reads dir_dict["current_dir"] for *.bval files."""

    def test_with_bval_file_returns_count_contents_paths(self, tmp_path):
        """With a .bval file: (count, raw_string, [path]).

        Source reads the file and returns bval_file.read() — the raw string
        WITHOUT stripping. Count = len(raw.split()).
        """
        curr_dir = tmp_path / "run"
        curr_dir.mkdir()
        bval_content = "0 1000 2000"
        bval_file = curr_dir / "sub-01_run-01.bval"
        bval_file.write_text(bval_content)

        num_bvals, bvals_str, bval_paths = cp.check_for_bvals(
            {"current_dir": str(curr_dir)}
        )

        assert num_bvals == 3
        assert bvals_str == bval_content  # raw, un-stripped
        assert len(bval_paths) == 1
        assert Path(bval_paths[0]).name == "sub-01_run-01.bval"

    def test_without_bval_file_returns_one_empty_empty(self, tmp_path):
        """Without any .bval file: (1, "", []).

        The source hard-codes num_bvals=1 and bvals="" when no file found.
        PINNED: the hard-coded 1 (not 0) means "no .bval" is treated downstream
        as a single volume.  This diverges from get_echo_info's internal
        num_bvals=0 for the same absent-bval condition — both behaviours are
        pinned as-is.
        """
        curr_dir = tmp_path / "run"
        curr_dir.mkdir()

        num_bvals, bvals_str, bval_paths = cp.check_for_bvals(
            {"current_dir": str(curr_dir)}
        )

        assert num_bvals == 1
        assert bvals_str == ""
        assert bval_paths == []

    def test_bval_count_matches_split_count(self, tmp_path):
        """num_bvals matches the number of whitespace-separated tokens."""
        curr_dir = tmp_path / "run"
        curr_dir.mkdir()
        # 5 values with mixed whitespace
        bval_content = "0\t500 1000\n1500 2000"
        (curr_dir / "sub-01.bval").write_text(bval_content)

        num_bvals, _, _ = cp.check_for_bvals({"current_dir": str(curr_dir)})

        assert num_bvals == 5

    def test_raw_string_returned_not_stripped(self, tmp_path):
        """The returned string is the raw file.read(), preserving trailing newline."""
        curr_dir = tmp_path / "run"
        curr_dir.mkdir()
        bval_content = "0 1000 2000\n"  # trailing newline
        (curr_dir / "sub-01.bval").write_text(bval_content)

        _, bvals_str, _ = cp.check_for_bvals({"current_dir": str(curr_dir)})

        # Source returns bval_file.read() unmodified; trailing \n preserved.
        assert bvals_str == bval_content


# ===========================================================================
# make_acq_params  (golden tests)
# ===========================================================================


class TestMakeAcqParams:
    """make_acq_params writes acqparams.txt; phase_sign=(-1)**(echo+1).

    PINNED ODDITY: column choice uses substring matching ("j" in phase_dir) and
    sign is purely (-1)**(echo+1); the +/- polarity suffix of phase_dir is
    intentionally NOT honored — "j-" and "j" produce identical output.
    """

    def _golden_path(self, name: str) -> Path:
        return _GOLDEN / name

    # --- (a) 3-echo, "j-", match=False ---

    def test_golden_3echo_jminus_nomatch(self, tmp_path, check_dict):
        """3 echoes, phase_dir='j-', match=False → matches committed golden file.

        Derived sign pattern (phase_sign = (-1)**(echo+1)):
          echo 1: +1  → '0 1 0 0.106487'
          echo 2: -1  → '0 -1 0 0.106487'
          echo 3: +1  → '0 1 0 0.106487'
        """
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2, 3], "j-")

        acq_file = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        actual = Path(acq_file).read_text()
        expected = self._golden_path("acqparams_3echo_jminus_nomatch.txt").read_text()
        assert actual == expected, (
            f"Acqparams content differs from golden.\n"
            f"Actual:\n{actual!r}\nExpected:\n{expected!r}"
        )

    # --- (b) 3-echo, "j-", match=True → only echoes 1 and 2 ---

    def test_golden_3echo_jminus_match(self, tmp_path, check_dict):
        """match=True with 3 echoes → only echoes 1,2 written (source L472).

        Source: if check_dict['match'] and run_df['echo'].nunique() > 2:
                    echo_list = [1, 2]
        Derived content:
          echo 1: +1  → '0 1 0 0.106487'
          echo 2: -1  → '0 -1 0 0.106487'
        """
        check_dict["match"] = True
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2, 3], "j-")

        acq_file = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        actual = Path(acq_file).read_text()
        expected = self._golden_path("acqparams_3echo_jminus_match.txt").read_text()
        assert actual == expected, (
            f"Acqparams content differs from golden.\n"
            f"Actual:\n{actual!r}\nExpected:\n{expected!r}"
        )

    # --- (c) 2-echo, "i" ---

    def test_golden_2echo_i(self, tmp_path, check_dict):
        """phase_dir='i' → sign column is first; zeros in j,k columns.

        Derived:
          echo 1: +1  → '1 0 0 0.106487'
          echo 2: -1  → '-1 0 0 0.106487'
        """
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2], "i")

        acq_file = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        actual = Path(acq_file).read_text()
        expected = self._golden_path("acqparams_2echo_i.txt").read_text()
        assert actual == expected, (
            f"Acqparams content differs from golden.\n"
            f"Actual:\n{actual!r}\nExpected:\n{expected!r}"
        )

    # --- (d) 2-echo, "k" ---

    def test_golden_2echo_k(self, tmp_path, check_dict):
        """phase_dir='k' → sign column is last; zeros in i,j columns.

        Derived:
          echo 1: +1  → '0 0 1 0.106487'
          echo 2: -1  → '0 0 -1 0.106487'
        """
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2], "k")

        acq_file = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        actual = Path(acq_file).read_text()
        expected = self._golden_path("acqparams_2echo_k.txt").read_text()
        assert actual == expected, (
            f"Acqparams content differs from golden.\n"
            f"Actual:\n{actual!r}\nExpected:\n{expected!r}"
        )

    # --- unrecognised phase_dir ---

    def test_unrecognised_phase_dir_exits(self, tmp_path, check_dict):
        """Unrecognised phase_dir calls exit(1) → SystemExit is raised."""
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1], "x")  # 'x' is not 'i', 'j', or 'k'

        with pytest.raises(SystemExit):
            cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

    # --- pre-existing file is returned without rewrite ---

    def test_preexisting_file_returned_unchanged(self, tmp_path, check_dict):
        """If acqparams.txt already exists it is returned without being rewritten.

        Pin the 'early return on exists' behavior (source L466-L468).
        """
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()

        # Write sentinel content to the acqparams file first.
        sentinel = "SENTINEL CONTENT DO NOT OVERWRITE\n"
        acq_file = topup_dir / f"{SUBJ}_{SESS}_{RUN}_acqparams.txt"
        acq_file.write_text(sentinel)

        run_df = _make_run_df([1, 2], "j-")
        result = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        assert Path(result).read_text() == sentinel, (
            "Pre-existing acqparams.txt must NOT be overwritten."
        )

    def test_returns_path_object(self, tmp_path, check_dict):
        """make_acq_params always returns a Path (not str)."""
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2], "j-")

        result = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        assert isinstance(result, Path)

    def test_acq_file_name_format(self, tmp_path, check_dict):
        """Output filename is '{subject}_{session}_{run}_acqparams.txt'."""
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2], "j-")

        result = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))

        assert Path(result).name == f"{SUBJ}_{SESS}_{RUN}_acqparams.txt"

    def test_match_true_with_only_2_echoes_writes_all_echoes(
        self, tmp_path, check_dict
    ):
        """match=True but nunique(echo) == 2 → echo_list from unique() (both echoes).

        Source condition: match AND nunique > 2. With 2 echoes the branch is
        skipped and all echoes are written.
        """
        check_dict["match"] = True
        topup_dir = tmp_path / "topup"
        topup_dir.mkdir()
        run_df = _make_run_df([1, 2], "j-")

        result = cp.make_acq_params(SUBJ, SESS, RUN, run_df, str(topup_dir))
        lines = Path(result).read_text().splitlines()

        assert len(lines) == 2, (
            "With only 2 echoes and match=True, both echoes must be written "
            "(source condition is nunique > 2)."
        )

    def test_polarity_suffix_ignored(self, tmp_path, check_dict):
        """phase_dir 'j-' and 'j' produce byte-identical acqparams files.

        PINNED ODDITY: the source tests 'j' in phase_dir (substring match), so
        both 'j-' and 'j' enter the same branch.  Sign is solely (-1)**(echo+1);
        the trailing '-' in 'j-' is never inspected.
        """
        topup_a = tmp_path / "topup_a"
        topup_b = tmp_path / "topup_b"
        topup_a.mkdir()
        topup_b.mkdir()

        run_df_jminus = _make_run_df([1, 2], "j-")
        run_df_j = _make_run_df([1, 2], "j")

        file_a = cp.make_acq_params(SUBJ, SESS, RUN, run_df_jminus, str(topup_a))
        file_b = cp.make_acq_params(SUBJ, SESS, RUN, run_df_j, str(topup_b))

        content_a = Path(file_a).read_bytes()
        content_b = Path(file_b).read_bytes()
        assert content_a == content_b, (
            "phase_dir 'j-' and 'j' must produce identical acqparams content; "
            "polarity suffix is not honored by the source."
        )


# ===========================================================================
# get_echo_info
# ===========================================================================


class TestGetEchoInfo:
    """get_echo_info parses a JSON sidecar and locates associated nii files."""

    EXPECTED_KEYS = {
        "json_path",
        "echo",
        "bvals",
        "bvals_list",
        "nii_list",
        "phase_dir",
        "phase_sign",
        "echo_time",
        "readout_time",
        "pulse_sequence",
        "series_description",
        "image_type",
        "orientation_list",
        "orientation_rounded",
        "slice_orientation",
        "orientation_num",
    }

    def _write_sidecar_and_niis(
        self,
        run_dir: Path,
        echo_num: int,
        num_niis: int,
        phase_dir: str = "j-",
        orientation: list | None = None,
    ) -> Path:
        """Create a JSON sidecar and the expected nii files for get_echo_info.

        Delegates to the module-level _make_sidecar_tree helper which encodes
        the canonical bb = str(i).zfill(len(str(n_vols))) naming convention.
        """
        stem = f"sub-01_echo-{echo_num}"
        return _make_sidecar_tree(
            run_dir,
            stem,
            n_vols=num_niis,
            echo_num=echo_num,
            phase_dir=phase_dir,
            orientation=orientation,
        )

    def test_no_bval_two_average_volumes(self, tmp_path):
        """No .bval file, 2 average volumes → bvals=2, bvals_list=[None], nii_list len=2.

        Source: when no bval, bvals=[None]; num_bvals falls back to len(found_niis).
        The function then constructs nii paths as json_path.replace(".json", "_{bb}.nii").
        With num_niis=2: bb in {"1","2"}, num_digits=1.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(run_dir, echo_num=1, num_niis=2)

        result = cp.get_echo_info(str(json_path))

        assert result.keys() == self.EXPECTED_KEYS
        assert result["bvals"] == 2
        assert result["bvals_list"] == [None]
        assert len(result["nii_list"]) == 2
        assert result["slice_orientation"] == "axial"
        assert result["orientation_num"] == 2
        assert result["phase_dir"] == "j-"
        # phase_sign = (-1)**(echo_num + 1) = (-1)**2 = 1
        assert result["phase_sign"] == 1
        assert result["echo_time"] == 0.07
        assert result["readout_time"] == READOUT

    @pytest.mark.parametrize("dirname", ["run", "jsondata"])
    def test_with_bval_file_three_bvals(self, tmp_path, dirname):
        """With a .bval file containing 3 values → bvals=3, bvals_list parsed.

        Source: bval_path = splitext(json_path)[0] + ".bval" (extension swap).
        bvals_list = f.read().split() → list of string tokens.
        nii names: {stem}_{bb}.nii where bb=str(i).zfill(len("3"))="1","2","3".

        "jsondata" is the regression case: the old code substring-replaced
        "json"→"bval" over the WHOLE path, so a directory name containing
        "json" made it look in /bvaldata/ and silently miss the bval file.
        """
        run_dir = tmp_path / dirname
        run_dir.mkdir()
        stem = "sub-01_echo-1"
        # _make_sidecar_tree uses the zfill convention; 3 vols → num_digits=1,
        # so bb in {"1","2","3"} — same as the previous inline loop.
        json_path = _make_sidecar_tree(
            run_dir,
            stem,
            n_vols=3,
            bvals=["0", "1000", "2000"],
            echo_num=1,
            phase_dir="j-",
        )

        result = cp.get_echo_info(str(json_path))

        assert result["bvals"] == 3
        assert result["bvals_list"] == ["0", "1000", "2000"]
        assert len(result["nii_list"]) == 3

    def test_axial_orientation(self, tmp_path):
        """ImageOrientationPatientDICOM [1,0,0,0,1,0] → axial, orientation_num=2."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(
            run_dir,
            echo_num=1,
            num_niis=1,
            orientation=[1, 0, 0, 0, 1, 0],
        )

        result = cp.get_echo_info(str(json_path))

        assert result["slice_orientation"] == "axial"
        assert result["orientation_num"] == 2
        assert result["orientation_rounded"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def test_coronal_orientation(self, tmp_path):
        """ImageOrientationPatientDICOM [1,0,0,0,0,-1] → coronal, orientation_num=1."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(
            run_dir,
            echo_num=1,
            num_niis=1,
            orientation=[1, 0, 0, 0, 0, -1],
        )

        result = cp.get_echo_info(str(json_path))

        assert result["slice_orientation"] == "coronal"
        assert result["orientation_num"] == 1

    def test_sagittal_orientation(self, tmp_path):
        """ImageOrientationPatientDICOM [0,1,0,0,0,-1] → sagittal, orientation_num=0."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(
            run_dir,
            echo_num=1,
            num_niis=1,
            orientation=[0, 1, 0, 0, 0, -1],
        )

        result = cp.get_echo_info(str(json_path))

        assert result["slice_orientation"] == "sagittal"
        assert result["orientation_num"] == 0

    def test_unknown_orientation(self, tmp_path):
        """Non-standard ImageOrientationPatientDICOM → 'unknown', orientation_num=3."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Oblique orientation — doesn't match any canonical pattern
        json_path = self._write_sidecar_and_niis(
            run_dir,
            echo_num=1,
            num_niis=1,
            orientation=[0.707, 0.707, 0, 0, 0, -1],
        )

        result = cp.get_echo_info(str(json_path))

        assert result["slice_orientation"] == "unknown"
        assert result["orientation_num"] == 3

    def test_phase_sign_echo2(self, tmp_path):
        """Echo 2 → phase_sign = (-1)**(2+1) = -1."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(run_dir, echo_num=2, num_niis=1)

        result = cp.get_echo_info(str(json_path))

        assert result["phase_sign"] == -1

    def test_multiple_glob_candidates_raises_file_not_found(self, tmp_path):
        """When expected nii is missing and glob finds 2+ candidates → FileNotFoundError.

        Source (L548-L553): if len(possible_files) > 1: raise FileNotFoundError.
        The glob pattern is '*echo-{echo_num}_{bb}.nii' in the json's directory.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        echo_num = 1
        stem = f"sub-01_echo-{echo_num}"
        json_path = run_dir / f"{stem}.json"
        _write_json(json_path, echo_num=echo_num, phase_dir="j-")

        # Do NOT create the expected nii (sub-01_echo-1_1.nii).
        # Create two glob candidates matching '*echo-1_1.nii' pattern.
        (run_dir / f"candidate_A_echo-{echo_num}_1.nii").touch()
        (run_dir / f"candidate_B_echo-{echo_num}_1.nii").touch()

        # Also create a bval file so num_bvals=1 (avoids the average fallback path)
        bval_path = run_dir / f"{stem}.bval"
        bval_path.write_text("0")

        with pytest.raises(FileNotFoundError, match="multiple candidate"):
            cp.get_echo_info(str(json_path))

    def test_returned_dict_has_exact_key_set(self, tmp_path):
        """The returned dict has exactly the 16 expected keys (no more, no less)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        json_path = self._write_sidecar_and_niis(run_dir, echo_num=1, num_niis=1)

        result = cp.get_echo_info(str(json_path))

        assert result.keys() == self.EXPECTED_KEYS


# ===========================================================================
# set_dirs
# ===========================================================================


class TestSetDirs:
    """set_dirs returns a directory dict; reads check_dict for conditional keys."""

    ALWAYS_PRESENT = {
        "current_dir",
        "ave_dir",
        "work_dir",
        "work_root",
        "topup_dir",
        "inner_dir",
    }

    def _make_current_dir(self, tmp_path):
        """Create the current_dir structure that set_dirs expects."""
        curr = tmp_path / "output" / SUBJ / SESS / RUN
        curr.mkdir(parents=True)
        return tmp_path / "output", tmp_path / "deriv"

    def test_all_false_returns_whole_volume(self, tmp_path, check_dict):
        """All flags False → inner_dir='whole-volume'; mandatory keys only."""
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert result["inner_dir"] == "whole-volume"
        assert set(result.keys()) == self.ALWAYS_PRESENT
        # No optional keys
        assert "slice_dir" not in result
        assert "masked_dir" not in result

    def test_slice_true_adds_slice_dir_and_per_slice_prefix(self, tmp_path, check_dict):
        """slice=True → inner_dir starts with 'per-slice'; 'slice_dir' key added."""
        check_dict["slice"] = True
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert result["inner_dir"] == "per-slice"
        assert "slice_dir" in result

    def test_match_true_appends_contrast_matched(self, tmp_path, check_dict):
        """match=True → '_contrast-matched' appended to inner_dir."""
        check_dict["match"] = True
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert result["inner_dir"] == "whole-volume_contrast-matched"
        assert "masked_dir" not in result

    def test_mask_true_appends_masked_and_adds_masked_dir(self, tmp_path, check_dict):
        """mask=True → '_masked' appended to inner_dir; 'masked_dir' key added."""
        check_dict["mask"] = True
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert result["inner_dir"] == "whole-volume_masked"
        assert "masked_dir" in result

    def test_slice_and_match_and_mask(self, tmp_path, check_dict):
        """All three flags True → inner_dir='per-slice_contrast-matched_masked'."""
        check_dict["slice"] = True
        check_dict["match"] = True
        check_dict["mask"] = True
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert result["inner_dir"] == "per-slice_contrast-matched_masked"
        assert "slice_dir" in result
        assert "masked_dir" in result

    def test_current_dir_path_composition(self, tmp_path, check_dict):
        """current_dir = output_dir / subject / session / run."""
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        expected_current = os.path.join(str(output_dir), SUBJ, SESS, RUN)
        assert result["current_dir"] == expected_current

    def test_topup_dir_path_composition(self, tmp_path, check_dict):
        """topup_dir = deriv_dir / 'undistortme' / inner_dir / subj/sess/run."""
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        expected_topup = os.path.join(
            str(deriv_dir), "undistortme", "whole-volume", SUBJ, SESS, RUN
        )
        assert result["topup_dir"] == expected_topup

    def test_missing_current_dir_exits(self, tmp_path, check_dict):
        """If current_dir does not exist, set_dirs calls exit(1) → SystemExit."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Do NOT create the run subdirectory.
        with pytest.raises(SystemExit):
            cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(tmp_path / "deriv"))

    def test_always_returns_topup_dir(self, tmp_path, check_dict):
        """'topup_dir' is always present."""
        output_dir, deriv_dir = self._make_current_dir(tmp_path)

        result = cp.set_dirs(SUBJ, SESS, RUN, str(output_dir), str(deriv_dir))

        assert "topup_dir" in result


class TestFindFilesOrdering:
    def test_results_are_sorted(self, tmp_path):
        """find_files returns sorted paths regardless of creation order.

        Every golden snapshot depends on this determinism guarantee: glob
        order is filesystem-dependent, so an unsorted find_files would make
        processing order (and the pinned command batches) vary by host.
        """
        for name in ["c.nii", "a.nii", "b.nii"]:
            (tmp_path / name).touch()
        result = cp.find_files(str(tmp_path), "*.nii")
        assert result == sorted(result)
        assert [p.rsplit("/", 1)[-1] for p in result] == ["a.nii", "b.nii", "c.nii"]


# ===========================================================================
# get_run_info
# ===========================================================================


class TestGetRunInfo:
    """get_run_info builds a DataFrame from JSON sidecars in curr_dir."""

    EXPECTED_COLUMNS = {
        "subject",
        "session",
        "run",
        "echo",
        "bvol_num",
        "bval",
        "nii",
        "echo_time",
        "readout_time",
        "phase_dir",
        "slice_orientation",
        "orientation_num",
        "pulse_sequence",
        "series_description",
        "image_type",
        "orientation_list",
        "orientation_rounded",
    }

    def _setup_run_dir(
        self, run_dir: Path, echoes: int, vols_per_echo: int, with_bvals: bool = False
    ) -> Path:
        """Create JSON sidecars and nii files for get_run_info.

        Delegates to _make_sidecar_tree for each echo, which encodes the
        canonical bb = str(i).zfill(len(str(n_vols))) naming convention.
        Files named: sub-01_echo-{e}.json + sub-01_echo-{e}_{bb}.nii.
        """
        for e in range(1, echoes + 1):
            stem = f"sub-01_echo-{e}"
            if with_bvals:
                # bval_path = json_path.replace("json", "bval") — substring
                bval_tokens = [str(i * 500) for i in range(vols_per_echo)]
                _make_sidecar_tree(
                    run_dir,
                    stem,
                    n_vols=vols_per_echo,
                    bvals=bval_tokens,
                    echo_num=e,
                    phase_dir="j-",
                )
            else:
                # No bval: average mode.
                _make_sidecar_tree(
                    run_dir,
                    stem,
                    n_vols=vols_per_echo,
                    echo_num=e,
                    phase_dir="j-",
                )
        return run_dir

    def test_column_set_no_bvals(self, tmp_path):
        """Returned DataFrame has exactly the 17 expected columns (no bvals case)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=2, vols_per_echo=2, with_bvals=False)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        assert set(df.columns) == self.EXPECTED_COLUMNS

    def test_row_count_no_bvals(self, tmp_path):
        """Without bvals: rows = echoes × volumes_per_echo."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=2, vols_per_echo=2, with_bvals=False)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        assert len(df) == 4  # 2 echoes × 2 averages

    def test_bval_is_none_without_bval_file(self, tmp_path):
        """Without .bval file, bval column contains None."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=1, vols_per_echo=2, with_bvals=False)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        assert df["bval"].isna().all(), "bval column should be None without bval file"

    def test_bvol_num_enumeration_no_bvals(self, tmp_path):
        """Without bvals, bvol_num is 1-indexed average number per echo."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=1, vols_per_echo=3, with_bvals=False)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)
        bvol_nums = sorted(df["bvol_num"].tolist())

        # 3 averages → bvol_nums = [1, 2, 3]
        assert bvol_nums == [1, 2, 3]

    def test_bval_is_int_with_bval_file(self, tmp_path):
        """With .bval file, bval column contains integers (not strings)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=1, vols_per_echo=3, with_bvals=True)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        # Guard: 1 echo × 3 bvals = 3 rows — ensures the per-row loop below
        # is not vacuous (would trivially pass on an empty DataFrame).
        assert len(df) == 3, f"Expected 3 rows (1 echo × 3 bvals), got {len(df)}"
        assert not df["bval"].isna().any(), "bval should not be None with bval file"
        for v in df["bval"]:
            assert isinstance(v, int), f"bval should be int, got {type(v)}"

    def test_bvol_num_enumeration_with_bvals(self, tmp_path):
        """With bvals, bvol_num is 1-indexed position in the bval list per echo."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=1, vols_per_echo=3, with_bvals=True)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)
        bvol_nums = sorted(df["bvol_num"].tolist())

        assert bvol_nums == [1, 2, 3]

    def test_row_count_with_bvals(self, tmp_path):
        """With bvals: rows = echoes × num_bvals."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=2, vols_per_echo=3, with_bvals=True)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        assert len(df) == 6  # 2 echoes × 3 bvals

    def test_subject_session_run_columns(self, tmp_path):
        """subject, session, run columns match the passed arguments."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._setup_run_dir(run_dir, echoes=1, vols_per_echo=1, with_bvals=False)

        df = cp.get_run_info(str(run_dir), SUBJ, SESS, RUN)

        assert (df["subject"] == SUBJ).all()
        assert (df["session"] == SESS).all()
        assert (df["run"] == RUN).all()


# ===========================================================================
# rename_entity
# ===========================================================================


class TestRenameEntity:
    """rename_entity removes dots from the stem (not extension), renames on disk."""

    def test_dots_in_stem_removed(self, tmp_path):
        """File with dots in stem: stem dots removed, extension preserved.

        Source logic (L178-L186):
          extension = old_path.suffixes[-1]  → ".nii" for "a.b.c.nii"
          stem      = old_path.stem          → "a.b.c"  (stem strips ONE suffix)

        Path("a.b.c.nii").stem == "a.b.c" and suffixes == [".b", ".c", ".nii"],
        so: extension=".nii", new_name="abc", result="abc.nii".
        """
        original = tmp_path / "sub-1_desc-a.b.c.nii"
        original.touch()

        result = cp.rename_entity(original)

        assert result is not None
        # stem "sub-1_desc-a.b.c" → dots removed → "sub-1_desc-abc"
        expected = tmp_path / "sub-1_desc-abc.nii"
        assert result == expected
        assert expected.exists()
        assert not original.exists()

    def test_no_dots_in_stem_returns_none(self, tmp_path):
        """File with no dots in stem → returns None (file unchanged)."""
        original = tmp_path / "sub-01_echo-1.nii"
        original.touch()

        result = cp.rename_entity(original)

        assert result is None
        assert original.exists()

    def test_extension_preserved(self, tmp_path):
        """Only the last suffix is preserved; dots inside the stem are stripped."""
        original = tmp_path / "foo.bar.baz"
        original.touch()

        result = cp.rename_entity(original)

        # stem="foo.bar", extension=".baz" → new_name="foobar", result="foobar.baz"
        assert result == tmp_path / "foobar.baz"

    def test_returns_path_object(self, tmp_path):
        """Return value is a Path object (not str)."""
        original = tmp_path / "a.b.nii"
        original.touch()

        result = cp.rename_entity(original)

        assert isinstance(result, Path)

    def test_no_suffix_file(self, tmp_path):
        """File with no extension: extension='', stem=full name; dots stripped."""
        original = tmp_path / "my.dotted.name"
        original.touch()

        result = cp.rename_entity(original)

        # suffixes=[".dotted", ".name"], extension=".name"
        # stem="my.dotted" → new_name="mydotted" → result="mydotted.name"
        assert result == tmp_path / "mydotted.name"

    def test_directory_rename(self, tmp_path):
        """rename_entity also works on directories (same logic)."""
        dotted_dir = tmp_path / "sub.dir.name"
        dotted_dir.mkdir()

        result = cp.rename_entity(dotted_dir)

        # stem="sub.dir", extension=".name" → new_name="subdir" → "subdir.name"
        assert result == tmp_path / "subdir.name"
        assert (tmp_path / "subdir.name").exists()


# ===========================================================================
# rename_entities
# ===========================================================================


class TestRenameEntities:
    """rename_entities renames the root dir and walks the tree renaming all names."""

    def test_root_dir_dots_removed(self, tmp_path):
        """The root directory itself has dots stripped from its stem (not name).

        Source L151-L154:
          dir  = Path(root_dir)
          name = dir.stem                      # strips LAST suffix
          new_name = dir.parent / name.replace(".", "")   # NO extension added
          os.rename(root_dir, new_name)

        For "sub.01.root":
          dir.stem = "sub.01"  (Path strips ".root")
          name.replace(".", "") = "sub01"
          new_name = parent / "sub01"           # extension dropped entirely

        Returns Path(new_name).stem = "sub01".

        PINNED ODDITY: the root dir loses its last suffix entirely (rename_entity
        preserves the extension but rename_entities uses dir.stem not dir.name).
        """
        dotted_root = tmp_path / "sub.01.root"
        dotted_root.mkdir()
        (dotted_root / "plain.nii").touch()  # plain file inside

        result = cp.rename_entities(str(dotted_root))

        # stem "sub.01" → dots removed → "sub01"; extension ".root" dropped
        assert result == "sub01"
        assert (tmp_path / "sub01").exists()
        assert not (tmp_path / "sub.01.root").exists()

    def test_nested_file_dots_removed(self, tmp_path):
        """Files with dots in stem are renamed inside the walked tree."""
        dotted_root = tmp_path / "rootdir"
        dotted_root.mkdir()
        dotted_file = dotted_root / "sub-01_desc-a.b.c.nii"
        dotted_file.touch()

        cp.rename_entities(str(dotted_root))

        # root has no dots → stays "rootdir"; file has dots removed
        assert (tmp_path / "rootdir" / "sub-01_desc-abc.nii").exists()
        assert not dotted_file.exists()

    def test_returns_stem_string(self, tmp_path):
        """Return value is a string (Path.stem), not a Path object."""
        root = tmp_path / "myroot"
        root.mkdir()

        result = cp.rename_entities(str(root))

        assert isinstance(result, str)
        assert result == "myroot"

    def test_plain_files_unchanged(self, tmp_path):
        """Files without dots in the stem are left alone (rename_entity returns None)."""
        root = tmp_path / "root"
        root.mkdir()
        plain = root / "plain_file.nii"
        plain.touch()

        cp.rename_entities(str(root))

        assert (tmp_path / "root" / "plain_file.nii").exists()
