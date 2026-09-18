"""Characterization tests for command-builder functions in corr_pipeline.py.

These tests pin the CURRENT behavior exactly as-is — ahead of a refactor.
Do NOT modify corr_pipeline.py; do NOT change expected strings to match a
"corrected" implementation — the source wins.

Expected strings are derived independently from the f-strings in the source;
they are NEVER produced by calling the function under test (no tautologies).

Source behaviors confirmed before writing:
- convert_dcms:            dcm2niix call shape; -z 3 not -z y
- get_fslmerge_command:    returns (None, filename) when output already exists
- get_topup_command:       replaces '.' with '-' in out_file_base; checks
                           corr_filename (not field_filename) for existence;
                           appends " --nthr=<os.cpu_count()>" when topup_multithread
- get_applytopup_command:  match=False → inindex=echo; match=True → odd echo→1 even→2;
                           zero-pads slice_num to 3 digits in "_sv-NNN.nii" suffix;
                           returns None command when out_file already exists
- get_slicenii_cmd:        orientation_num param is accepted but not used in output
- get_combinenii_cmd:      four-arg one-liner
- linear_contrast_match:   hardcoded absolute path to contrastmatch.py;
                           skip when cm_file already exists
- geomean_contrast_match:  same path; appends " -m geomean"; skip when exists
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from undistortme import pipeline as cp

# ---------------------------------------------------------------------------
# Shared test-vector parameters (fixed; nothing computed by the function).
# ---------------------------------------------------------------------------
SUBJ = "sub-01"
SESS = "ses-baseline"
RUN = "run-01"
BV = 3
SV = 7
SUF = "echo1-2"

# The contrast-match invocation prefix (same-interpreter module run).
_CONTRASTMATCH_PATH = cp.CONTRASTMATCH_CMD

# Module-level echo-time constants shared by contrast-match test classes.
_TE1, _TE2, _TE3 = 0.07, 0.09, 0.11


def _make_slice_df(tmp_path):
    """Return a 3-echo slice DataFrame used by contrast-match tests."""
    rows = [
        {"echo": 1, "nii": str(tmp_path / "e1.nii"), "echo_time": _TE1},
        {"echo": 2, "nii": str(tmp_path / "e2.nii"), "echo_time": _TE2},
        {"echo": 3, "nii": str(tmp_path / "e3.nii"), "echo_time": _TE3},
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# convert_dcms
# ===========================================================================


class TestConvertDcms:
    def test_returns_dcm2niix_command(self, tmp_path):
        """convert_dcms builds a plain dcm2niix call."""
        in_dir = str(tmp_path / "source")
        out_dir = str(tmp_path / "niftis")

        result = cp.convert_dcms(in_dir, out_dir)

        assert isinstance(result, str)
        assert result.startswith("dcm2niix")

    def test_exact_command_string(self, tmp_path):
        """Full command string matches the f-string in the source exactly."""
        in_dir = str(tmp_path / "source")
        out_dir = str(tmp_path / "niftis")

        result = cp.convert_dcms(in_dir, out_dir)

        # Derived independently from the source f-string.
        expected = (
            "dcm2niix"
            " -d 9"
            " -z 3"
            " -w 0"
            " -v 1"
            " -f sub-%i/ses-%t/run-%s_desc-%d/sub-%i_ses-%t_run-%s_desc-%d_echo-%e"
            f" -o {out_dir} " + in_dir
        )
        assert result == expected

    def test_creates_output_dir(self, tmp_path):
        """convert_dcms calls os.makedirs so the output dir is created."""
        out_dir = str(tmp_path / "new_subdir")
        cp.convert_dcms(str(tmp_path / "src"), out_dir)
        assert os.path.isdir(out_dir)


# ===========================================================================
# get_fslmerge_command
# ===========================================================================


class TestGetFslmergeCommand:
    def _expected_filename(self, topup_dir):
        return os.path.join(
            topup_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_merged_{SUF}.nii",
        )

    def test_build_command(self, tmp_path):
        """Normal path: returns the fslmerge -t command and the merged filename."""
        topup_dir = str(tmp_path / "topup")
        file_list = [str(tmp_path / "a.nii"), str(tmp_path / "b.nii")]

        cmd, fname = cp.get_fslmerge_command(
            SUBJ, SESS, RUN, BV, SV, topup_dir, file_list, SUF
        )

        expected_fname = self._expected_filename(topup_dir)
        expected_cmd = f"fslmerge -t {expected_fname} {' '.join(file_list)}"

        assert fname == expected_fname
        assert cmd == expected_cmd

    def test_skip_when_output_exists(self, tmp_path):
        """When the merged file already exists, command is None and path is returned."""
        topup_dir = str(tmp_path / "topup")
        os.makedirs(topup_dir, exist_ok=True)

        expected_fname = self._expected_filename(topup_dir)
        Path(expected_fname).touch()  # pre-create the output

        cmd, fname = cp.get_fslmerge_command(
            SUBJ, SESS, RUN, BV, SV, topup_dir, ["a.nii", "b.nii"], SUF
        )

        assert cmd is None
        assert fname == expected_fname


# ===========================================================================
# get_topup_command
# ===========================================================================


class TestGetTopupCommand:
    def _out_file_base(self, topup_dir, suffix=SUF):
        """Reproduce the out_file_base with the dot-to-dash replacement."""
        raw = f"topup-result_{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_{suffix}"
        return os.path.join(topup_dir, raw.replace(".", "-"))

    def test_single_thread_exact_string(self, tmp_path, check_dict):
        """Single-thread topup command matches the source f-string exactly."""
        topup_dir = str(tmp_path / "topup")
        acq_file = str(tmp_path / "acq.txt")
        config = "b02b0.cnf"
        merged = str(tmp_path / "merged.nii")

        # check_dict["topup_multithread"] stays False (default from fixture)
        cmd, base = cp.get_topup_command(
            SUBJ, SESS, RUN, BV, SV, topup_dir, acq_file, config, merged, SUF
        )

        expected_base = self._out_file_base(topup_dir)
        field_file = expected_base + "_field.nii"
        corr_file = expected_base + "_desc-topupcorr.nii"

        expected_cmd = (
            f"topup --imain={merged}"
            f" --datain={acq_file}"
            f" --config={config}"
            " --scale=1"
            f" --out={expected_base}"
            f" --fout={field_file}"
            f" --iout={corr_file}"
        )

        assert base == expected_base
        assert cmd == expected_cmd
        assert "--nthr=" not in cmd

    def test_dot_replaced_with_dash_in_suffix(self, tmp_path, check_dict):
        """Dots in the suffix become dashes in out_file_base (source L1447)."""
        suffix_with_dot = "b0.cnf-echo1"
        topup_dir = str(tmp_path / "topup")

        cmd, base = cp.get_topup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            topup_dir,
            "acq.txt",
            "b02b0.cnf",
            "merged.nii",
            suffix_with_dot,
        )

        # The replacement acts on the basename only; restrict check to avoid
        # false failures from dots in the tmp_path directory name.
        assert "." not in os.path.basename(base), (
            "out_file_base must have all dots replaced with dashes"
        )

    def test_multithread_appends_nthr(self, tmp_path, check_dict):
        """When topup_multithread is True, --nthr=<cpu_count> is appended."""
        check_dict["topup_multithread"] = True

        topup_dir = str(tmp_path / "topup")
        cmd, base = cp.get_topup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            topup_dir,
            "acq.txt",
            "b02b0.cnf",
            "merged.nii",
            SUF,
        )

        assert cmd is not None
        assert "--nthr=" in cmd

        # The value is os.cpu_count() — compute it independently.
        expected_nthr = str(os.cpu_count())
        assert cmd.endswith("--nthr=" + expected_nthr), (
            f"Expected command to end with '--nthr={expected_nthr}'; got: {cmd!r}"
        )

    def test_skip_when_corr_output_exists(self, tmp_path, check_dict):
        """Returns (None, base) when the *_desc-topupcorr.nii file already exists.

        Note: the source checks corr_filename (not field_filename) for existence.
        """
        topup_dir = str(tmp_path / "topup")
        os.makedirs(topup_dir, exist_ok=True)

        expected_base = self._out_file_base(topup_dir)
        corr_file = expected_base + "_desc-topupcorr.nii"
        Path(corr_file).touch()  # pre-create the checked output

        cmd, base = cp.get_topup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            topup_dir,
            "acq.txt",
            "b02b0.cnf",
            "merged.nii",
            SUF,
        )

        assert cmd is None
        assert base == expected_base


# ===========================================================================
# get_applytopup_command
# ===========================================================================


class TestGetApplytopupCommand:
    def _out_file_base(self, topup_dir):
        return os.path.join(
            topup_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_desc-undistorted-{SUF}",
        )

    def _ending(self, slice_num):
        return f"_sv-{str(slice_num).zfill(3)}.nii"

    def test_match_off_inindex_equals_echo(self, tmp_path, check_dict):
        """match=False → inindex is the raw echo number."""
        topup_dir = str(tmp_path / "topup")
        acq_file = "acq.txt"
        topup_base = "topup_base"

        nii1 = str(tmp_path / "e1.nii")
        nii2 = str(tmp_path / "e2.nii")

        cmd_list, out_list = cp.get_applytopup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            acq_file,
            topup_dir,
            topup_base,
            [nii1, nii2],
            [1, 2],
            SUF,
        )

        base = self._out_file_base(topup_dir)
        ending = self._ending(SV)

        expected_out1 = base + "_echo-1" + ending
        expected_out2 = base + "_echo-2" + ending

        expected_cmd1 = (
            f"applytopup --imain={nii1}"
            f" --datain={acq_file}"
            " --inindex=1"
            f" --topup={topup_base}"
            f" --out={expected_out1}"
            " --method=jac"
        )
        expected_cmd2 = (
            f"applytopup --imain={nii2}"
            f" --datain={acq_file}"
            " --inindex=2"
            f" --topup={topup_base}"
            f" --out={expected_out2}"
            " --method=jac"
        )

        assert out_list == [expected_out1, expected_out2]
        assert cmd_list == [expected_cmd1, expected_cmd2]

    def test_match_on_odd_echo_gets_index1_even_gets_index2(self, tmp_path, check_dict):
        """match=True → odd echo → inindex=1; even echo → inindex=2.

        Source L1502: e = 2 if echo % 2 == 0 else 1
        """
        check_dict["match"] = True

        topup_dir = str(tmp_path / "topup")
        acq_file = "acq.txt"
        topup_base = "topup_base"
        nii1 = str(tmp_path / "e1.nii")
        nii2 = str(tmp_path / "e2.nii")
        nii3 = str(tmp_path / "e3.nii")

        cmd_list, _ = cp.get_applytopup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            acq_file,
            topup_dir,
            topup_base,
            [nii1, nii2, nii3],
            [1, 2, 3],
            SUF,
        )

        # echo 1 (odd)  → inindex=1
        # echo 2 (even) → inindex=2
        # echo 3 (odd)  → inindex=1
        assert " --inindex=1" in cmd_list[0]
        assert " --inindex=2" in cmd_list[1]
        assert " --inindex=1" in cmd_list[2]

    def test_zero_padded_sv_in_output_name(self, tmp_path, check_dict):
        """slice_num is zero-padded to 3 digits in the output filename."""
        topup_dir = str(tmp_path / "topup")
        cmd_list, out_list = cp.get_applytopup_command(
            SUBJ, SESS, RUN, BV, 5, "acq.txt", topup_dir, "base", ["e1.nii"], [1], SUF
        )
        # slice 5 → _sv-005.nii
        assert out_list[0].endswith("_sv-005.nii")

    def test_slice_num_padded_double_digit(self, tmp_path, check_dict):
        """slice_num=42 → _sv-042.nii  (2-digit padded to 3)."""
        topup_dir = str(tmp_path / "topup")
        cmd_list, out_list = cp.get_applytopup_command(
            SUBJ, SESS, RUN, BV, 42, "acq.txt", topup_dir, "base", ["e1.nii"], [1], SUF
        )
        assert out_list[0].endswith("_sv-042.nii")

    def test_method_jac_in_every_command(self, tmp_path, check_dict):
        """Every generated applytopup command must include --method=jac."""
        topup_dir = str(tmp_path / "topup")
        cmd_list, _ = cp.get_applytopup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            "acq.txt",
            topup_dir,
            "base",
            ["e1.nii", "e2.nii"],
            [1, 2],
            SUF,
        )
        assert len(cmd_list) == 2
        for cmd in cmd_list:
            assert cmd is not None
            assert "--method=jac" in cmd

    def test_skip_when_out_file_exists(self, tmp_path, check_dict):
        """If an output file already exists its command slot is None."""
        topup_dir = str(tmp_path / "topup")
        os.makedirs(topup_dir, exist_ok=True)

        base = self._out_file_base(topup_dir)
        ending = self._ending(SV)
        out1 = base + "_echo-1" + ending
        out2 = base + "_echo-2" + ending
        Path(out1).touch()  # pre-create only echo-1 output

        nii1 = str(tmp_path / "e1.nii")
        nii2 = str(tmp_path / "e2.nii")

        cmd_list, out_list = cp.get_applytopup_command(
            SUBJ,
            SESS,
            RUN,
            BV,
            SV,
            "acq.txt",
            topup_dir,
            "base",
            [nii1, nii2],
            [1, 2],
            SUF,
        )

        assert cmd_list[0] is None, "echo-1 output exists → command must be None"
        assert cmd_list[1] is not None, "echo-2 output absent → command must be built"
        assert out_list == [out1, out2]


# ===========================================================================
# get_slicenii_cmd
# ===========================================================================


class TestGetSliceniiCmd:
    def test_exact_command_string(self):
        """Output is the one-liner from source L1154; orientation_num is unused."""
        nii_path = "/some/path/img.nii"
        slice_dir = "/some/slice_dir"
        orient_num = 3

        result = cp.get_slicenii_cmd(nii_path, slice_dir, orient_num)

        # Derived directly from source L1154 (orientation_num is NOT in the output).
        expected = f"slicenii -i {nii_path} -o {slice_dir} -p 6"
        assert result == expected

    def test_orientation_num_not_in_output(self):
        """Confirm orientation_num=99 does not appear in the command string."""
        result = cp.get_slicenii_cmd("/img.nii", "/out", 99)
        assert "99" not in result


# ===========================================================================
# get_combinenii_cmd
# ===========================================================================


class TestGetCombineniiCmd:
    def test_exact_command_string(self):
        """Output matches the one-liner in source L1160-L1162."""
        in_dir = "/input_dir"
        ref_nii = "/ref.nii"
        start_string = "sub-01"
        output_filename = "/out.nii"

        result = cp.get_combinenii_cmd(in_dir, ref_nii, start_string, output_filename)

        expected = (
            f"combinenii -i {in_dir} -r {ref_nii}"
            f" -s {start_string} -o {output_filename}"
        )
        assert result == expected


# ===========================================================================
# linear_contrast_match
# ===========================================================================


class TestLinearContrastMatch:
    def test_contains_hardcoded_contrastmatch_path(self, tmp_path):
        """Command must contain the literal hardcoded path to contrastmatch.py.

        Known refactor target — pinned as-is.
        """
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        cmd, _ = cp.linear_contrast_match(SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir)

        assert cmd is not None
        assert _CONTRASTMATCH_PATH in cmd, (
            f"Command must contain the literal hardcoded path "
            f"{_CONTRASTMATCH_PATH!r}; got: {cmd!r}"
        )

    def test_exact_command_string(self, tmp_path):
        """Full linear_contrast_match command matches source L335."""
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        e1 = str(tmp_path / "e1.nii")
        e3 = str(tmp_path / "e3.nii")
        cm_file = os.path.join(
            ave_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_desc-cm_linear_echoes-1-3.nii",
        )

        expected_cmd = (
            f"{_CONTRASTMATCH_PATH}"
            f" -i {e1} {e3}"
            f" -t {_TE1} {_TE3}"
            f" -n {_TE2}"
            f" -o {cm_file}"
        )

        cmd, files = cp.linear_contrast_match(
            SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir
        )

        assert cmd == expected_cmd
        assert files == [cm_file, str(tmp_path / "e2.nii")]

    def test_no_geomean_flag(self, tmp_path):
        """Linear method must NOT contain '-m geomean'."""
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        cmd, _ = cp.linear_contrast_match(SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir)
        assert cmd is not None
        assert "-m geomean" not in cmd

    def test_skip_when_cm_file_exists(self, tmp_path):
        """Returns (None, [cm_file, echo2]) when cm_file already exists."""
        ave_dir = str(tmp_path / "ave")
        os.makedirs(ave_dir)
        slice_df = _make_slice_df(tmp_path)

        cm_file = os.path.join(
            ave_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_desc-cm_linear_echoes-1-3.nii",
        )
        Path(cm_file).touch()

        cmd, files = cp.linear_contrast_match(
            SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir
        )

        assert cmd is None
        assert files[0] == cm_file


# ===========================================================================
# geomean_contrast_match
# ===========================================================================


class TestGeomeanContrastMatch:
    def test_contains_hardcoded_contrastmatch_path(self, tmp_path):
        """Command must contain the literal hardcoded path to contrastmatch.py.

        Known refactor target — pinned as-is.
        """
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        cmd, _ = cp.geomean_contrast_match(SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir)

        assert cmd is not None
        assert _CONTRASTMATCH_PATH in cmd, (
            f"Command must contain the literal hardcoded path "
            f"{_CONTRASTMATCH_PATH!r}; got: {cmd!r}"
        )

    def test_contains_m_geomean_flag(self, tmp_path):
        """geomean method appends ' -m geomean' (source L382)."""
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        cmd, _ = cp.geomean_contrast_match(SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir)

        assert cmd is not None
        assert "-m geomean" in cmd

    def test_exact_command_string(self, tmp_path):
        """Full geomean command matches source L381-L382."""
        ave_dir = str(tmp_path / "ave")
        slice_df = _make_slice_df(tmp_path)

        e1 = str(tmp_path / "e1.nii")
        e3 = str(tmp_path / "e3.nii")
        cm_file = os.path.join(
            ave_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_desc-cm_geomean_echoes-1-3.nii",
        )

        expected_cmd = (
            f"{_CONTRASTMATCH_PATH}"
            f" -i {e1} {e3}"
            f" -t {_TE1} {_TE3}"
            f" -n {_TE2}"
            f" -o {cm_file}"
            " -m geomean"
        )

        cmd, files = cp.geomean_contrast_match(
            SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir
        )

        assert cmd == expected_cmd
        assert files == [cm_file, str(tmp_path / "e2.nii")]

    def test_skip_when_cm_file_exists(self, tmp_path):
        """Returns (None, [cm_file, echo2]) when cm_file already exists."""
        ave_dir = str(tmp_path / "ave")
        os.makedirs(ave_dir)
        slice_df = _make_slice_df(tmp_path)

        cm_file = os.path.join(
            ave_dir,
            f"{SUBJ}_{SESS}_{RUN}_bv-{BV}_sv-{SV}_desc-cm_geomean_echoes-1-3.nii",
        )
        Path(cm_file).touch()

        cmd, files = cp.geomean_contrast_match(
            SUBJ, SESS, RUN, BV, SV, slice_df, ave_dir
        )

        assert cmd is None
        assert files[0] == cm_file
