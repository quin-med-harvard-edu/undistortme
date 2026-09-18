"""Pipeline wiring for estimation-only denoising.

THE INVARIANT
-------------
Denoising exists to stabilise the ESTIMATED FIELD.  Every command that feeds
topup (contrast match, fslmerge, topup itself) must consume denoised images,
and every applytopup command must still consume the ORIGINAL ones.  The tests
in TestEstimationOnlyInvariant are the ones that would catch a regression that
silently starts correcting denoised data.
"""

import os

import pandas as pd
import pytest

from conftest import _reps, _run_process_run
from undistortme import pipeline as cp

SUBJ = "sub-01"
SESS = "ses-01"
RUN = "run-01"

_DENOISE_PATH = cp.DENOISE_CMD


def _run_df(tmp_path, *, echoes=(1, 2), bvols=(1, 2), bvals=(0, 1000)):
    """A minimal run table of the shape get_run_info produces."""
    rows = []
    for e in echoes:
        for bvol, bval in zip(bvols, bvals):
            rows.append(
                {
                    "subject": SUBJ,
                    "session": SESS,
                    "run": RUN,
                    "echo": e,
                    "bvol_num": bvol,
                    "bval": bval,
                    "nii": str(tmp_path / f"{SUBJ}_echo-{e}_{bvol}.nii"),
                    "orientation_num": 2,
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# The command builder
# ===========================================================================


class TestGetDenoiseCommand:
    def test_builds_a_median_command(self, tmp_path):
        niis = [str(tmp_path / "a.nii"), str(tmp_path / "b.nii")]
        out_dir = str(tmp_path / "work")

        command, outputs = cp.get_denoise_command(niis, out_dir, "median")

        assert command == (
            f"{_DENOISE_PATH} -m median -i {niis[0]} {niis[1]}"
            f" -o {out_dir} --suffix desc-denoised-median"
        )
        assert outputs == [
            os.path.join(out_dir, "a_desc-denoised-median.nii"),
            os.path.join(out_dir, "b_desc-denoised-median.nii"),
        ]

    def test_the_slice_axis_is_left_to_the_denoiser(self, tmp_path):
        """The pipeline must not pass a slice axis.

        Only the image header knows the acquired slice axis; deriving it from
        the anatomical orientation is wrong (dcm2niix writes slices along the
        last array axis whatever the anatomy), so undistortme-denoise reads it
        per file instead.
        """
        niis = [str(tmp_path / "a.nii")]

        command, _ = cp.get_denoise_command(niis, str(tmp_path), "median")

        assert "--slice_axis" not in command

    def test_returns_no_command_when_outputs_exist(self, tmp_path, tiny_nii):
        out_dir = tmp_path / "work"
        out_dir.mkdir()
        niis = [str(tiny_nii(tmp_path / "a.nii"))]
        tiny_nii(out_dir / "a_desc-denoised-median.nii")

        command, outputs = cp.get_denoise_command(niis, str(out_dir), "median")

        assert command is None
        assert outputs == [str(out_dir / "a_desc-denoised-median.nii")]


# ===========================================================================
# handle_denoising
# ===========================================================================


class TestHandleDenoising:
    def test_no_denoising_makes_est_nii_the_original(
        self, check_dict, tmp_path, recorder
    ):
        check_dict["denoise"] = None
        run_df = _run_df(tmp_path)

        out = cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        assert out["est_nii"].tolist() == out["nii"].tolist()
        assert recorder == []

    def test_one_command_per_echo_over_the_whole_series(
        self, check_dict, tmp_path, recorder
    ):
        check_dict["denoise"] = "mppca"
        run_df = _run_df(
            tmp_path,
            echoes=(1, 2, 3),
            bvols=(1, 2, 3, 4, 5),
            bvals=(0, 1000, 1000, 1000, 1000),
        )

        cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        assert len(recorder) == 1
        description, commands = recorder[0]
        assert description == "denoising commands"
        assert len(commands) == 3

    def test_est_nii_points_at_the_denoised_copies(
        self, check_dict, tmp_path, recorder
    ):
        check_dict["denoise"] = "median"
        work = tmp_path / "work"
        run_df = _run_df(tmp_path)

        out = cp.handle_denoising(run_df, {"work_dir": str(work)})

        assert all(p.startswith(str(work)) for p in out["est_nii"])
        assert all("desc-denoised-median" in p for p in out["est_nii"])
        assert out["nii"].tolist() == run_df["nii"].tolist()

    def test_volumes_follow_bvol_num_order(self, check_dict, tmp_path, recorder):
        check_dict["denoise"] = "mppca"
        run_df = _run_df(
            tmp_path,
            echoes=(1,),
            bvols=(3, 1, 2, 5, 4),
            bvals=(2000, 0, 1000, 1000, 1000),
        )

        cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        command = recorder[0][1][0]
        # sorted by bvol_num, whatever order the frame arrived in
        assert command.index("_1.nii") < command.index("_2.nii")

    def test_a_failed_denoise_command_skips_the_run(
        self, check_dict, tmp_path, monkeypatch, capsys
    ):
        """Otherwise topup would estimate from originals while writing into the
        _denoise-<method> tree, and resume would keep that result forever."""
        check_dict["denoise"] = "mppca"
        monkeypatch.setattr(cp, "failed_commands", [])

        def _fail(commands, description):
            return [(command, 1) for command in commands or []]

        monkeypatch.setattr(cp, "parallel_bash_commands", _fail)
        run_df = _run_df(
            tmp_path, bvols=(1, 2, 3, 4, 5), bvals=(0, 1000, 1000, 1000, 1000)
        )

        out = cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        assert out is None
        assert len(cp.failed_commands) == 1
        assert "failed" in cp.failed_commands[0][0]
        assert "SKIPPING" in capsys.readouterr().out

    def test_too_few_volumes_skips_the_run_before_dispatch(
        self, check_dict, tmp_path, recorder, monkeypatch, capsys
    ):
        """undistortme-denoise would raise per echo; refuse once, up front."""
        check_dict["denoise"] = "mppca"
        monkeypatch.setattr(cp, "failed_commands", [])
        run_df = _run_df(tmp_path, echoes=(1, 2), bvols=(1, 2), bvals=(0, 1000))

        out = cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        assert out is None
        assert recorder == [], "no denoise command may be dispatched"
        assert "volume" in cp.failed_commands[0][0]


# ===========================================================================
# Slicing: a missing estimation slice must never pass as the original
# ===========================================================================


class TestSlicingRequiresEstimationSlices:
    def test_missing_estimation_slices_skips_the_run(
        self, check_dict, tmp_path, recorder, tiny_nii, monkeypatch, capsys
    ):
        """slicenii failed for the denoised copy: falling back to the original
        slice would put undenoised data into a _denoise-<method> field."""
        check_dict["denoise"] = "median"
        monkeypatch.setattr(cp, "failed_commands", [])
        slice_dir = tmp_path / "work"
        slice_dir.mkdir()
        run_df = _run_df(tmp_path, echoes=(1,), bvols=(1,), bvals=(0,))
        run_df["est_nii"] = str(slice_dir / "denoised.nii")
        # only the ORIGINAL gets slices on disk
        original_slices = slice_dir / (
            os.path.basename(run_df["nii"][0]).split(".")[0] + "_slices"
        )
        original_slices.mkdir()
        base = os.path.basename(run_df["nii"][0]).split(".")[0]
        tiny_nii(original_slices / f"{base}_axis-2_slice-padded-001.nii")

        out = cp.handle_slicing(run_df, {"slice_dir": str(slice_dir)})

        assert out is None
        assert len(cp.failed_commands) == 1
        assert "slice" in cp.failed_commands[0][0]
        assert "SKIPPING" in capsys.readouterr().out

    def test_process_run_skips_the_run_when_denoising_fails(
        self, check_dict, bids_run, monkeypatch
    ):
        """A failed denoise must stop the run before any topup is dispatched,
        and must still fail the exit status."""
        check_dict["topup"] = True
        check_dict["denoise"] = "median"
        monkeypatch.setattr(cp, "failed_commands", [])
        calls = []

        def _fail_denoising(commands, description):
            calls.append(description)
            if description == "denoising commands":
                return [(command, 1) for command in commands or []]
            return []

        monkeypatch.setattr(cp, "parallel_bash_commands", _fail_denoising)
        monkeypatch.setattr(cp, "shuffle", lambda seq: None)
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        assert "topup commands per-bvol" not in calls
        assert cp.failed_commands, "the skipped run must fail the exit status"

    def test_a_failed_slicenii_command_skips_the_run(
        self, check_dict, tmp_path, monkeypatch, capsys
    ):
        """A volume whose slicenii failed would otherwise contribute zero
        slice rows and the run would quietly proceed with fewer echoes."""
        check_dict["denoise"] = None
        monkeypatch.setattr(cp, "failed_commands", [])

        def _fail(commands, description):
            return [(command, 1) for command in commands or []]

        monkeypatch.setattr(cp, "parallel_bash_commands", _fail)
        slice_dir = tmp_path / "work"
        slice_dir.mkdir()
        run_df = _run_df(tmp_path, echoes=(1,), bvols=(1,), bvals=(0,))
        run_df["est_nii"] = run_df["nii"]

        out = cp.handle_slicing(run_df, {"slice_dir": str(slice_dir)})

        assert out is None
        assert len(cp.failed_commands) == 1
        assert "slicenii" in cp.failed_commands[0][0]
        assert "SKIPPING" in capsys.readouterr().out


# ===========================================================================
# Variant naming (resume-by-existence depends on it)
# ===========================================================================


class TestVariantNaming:
    def test_denoise_method_is_part_of_the_variant(self, check_dict, tmp_path):
        check_dict["denoise"] = "mppca"
        run_dir = tmp_path / SUBJ / SESS / RUN
        run_dir.mkdir(parents=True)

        dir_dict = cp.set_dirs(SUBJ, SESS, RUN, str(tmp_path), str(tmp_path / "deriv"))

        assert dir_dict["inner_dir"] == "whole-volume_denoise-mppca"

    def test_conditions_do_not_share_a_work_tree(self, check_dict, tmp_path):
        """Two methods must not resume into each other's merged files/fields."""
        run_dir = tmp_path / SUBJ / SESS / RUN
        run_dir.mkdir(parents=True)

        check_dict["denoise"] = None
        plain = cp.set_dirs(SUBJ, SESS, RUN, str(tmp_path), str(tmp_path / "deriv"))
        check_dict["denoise"] = "median"
        denoised = cp.set_dirs(SUBJ, SESS, RUN, str(tmp_path), str(tmp_path / "deriv"))

        assert plain["work_dir"] != denoised["work_dir"]
        assert plain["topup_dir"] != denoised["topup_dir"]


# ===========================================================================
# The estimation-only invariant, end to end through process_run
# ===========================================================================


class TestEstimationOnlyInvariant:
    def test_merge_uses_denoised_and_apply_uses_originals(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        check_dict["topup"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        text = normalize(slicing_recorder, _reps(info))
        merge_lines = [l for l in text.splitlines() if "fslmerge" in l]
        apply_lines = [l for l in text.splitlines() if "applytopup" in l]
        assert merge_lines and apply_lines
        assert all("desc-denoised-median" in l for l in merge_lines)
        for line in apply_lines:
            imain = line.split("--imain=")[1].split()[0]
            assert "desc-denoised" not in imain
            assert imain.startswith("<OUT>")

    def test_contrast_match_consumes_denoised_echoes(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        check_dict["topup"] = True
        check_dict["match"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        text = normalize(slicing_recorder, _reps(info))
        cm_lines = [l for l in text.splitlines() if "<CMATCH>" in l]
        assert cm_lines
        for line in cm_lines:
            inputs = line.split(" -i ")[1].split(" -t ")[0]
            assert all("desc-denoised-median" in tok for tok in inputs.split())

    def test_per_slice_estimation_slices_come_from_denoised_volumes(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        check_dict["topup"] = True
        check_dict["slice"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        text = normalize(slicing_recorder, _reps(info))
        merge_lines = [l for l in text.splitlines() if "fslmerge" in l]
        apply_lines = [l for l in text.splitlines() if "applytopup" in l]
        assert merge_lines and apply_lines
        for line in merge_lines:
            # "fslmerge -t {merged} {inputs...}"; the normalized text may end
            # the line with the "&&" that joins the compound command
            niis = [tok for tok in line.split() if tok.endswith(".nii")]
            assert all("desc-denoised-median" in tok for tok in niis[1:])
        for line in apply_lines:
            imain = line.split("--imain=")[1].split()[0]
            assert "desc-denoised" not in imain

    def test_diffusion_path_estimates_on_denoised_data(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        check_dict["topup"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])

        _run_process_run(info, cutoff=1000)

        text = normalize(slicing_recorder, _reps(info))
        merge_lines = [l for l in text.splitlines() if "fslmerge" in l]
        apply_lines = [l for l in text.splitlines() if "applytopup" in l]
        assert merge_lines and apply_lines
        assert all("desc-denoised-median" in l for l in merge_lines)
        for line in apply_lines:
            imain = line.split("--imain=")[1].split()[0]
            assert "desc-denoised" not in imain


# ===========================================================================
# Gating
# ===========================================================================


class TestDryrun:
    def test_dryrun_still_matches_volumes_when_the_images_exist(
        self, check_dict, bids_run, recorder, monkeypatch
    ):
        """--dryrun must print the plan the real run would execute.

        With denoising off, the estimation images ARE the original files, which
        exist, so the high-b volume matching can and must be done for real.
        Skipping it prints an --topup= base that the real run would not use.
        """
        check_dict["topup"] = True
        check_dict["dryrun"] = True
        check_dict["denoise"] = None
        calls = []
        real = cp.find_closest_volume_nmi
        monkeypatch.setattr(
            cp,
            "find_closest_volume_nmi",
            lambda img, comparisons: calls.append(1) or real(img, comparisons),
        )
        info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])

        _run_process_run(info, cutoff=1000)

        assert calls, "volume matching was skipped even though the images exist"

    def test_diffusion_dryrun_does_not_read_estimation_images(
        self, check_dict, bids_run, recorder, normalize
    ):
        """Under --dryrun the denoised copies were never written, so the
        diffusion path must not try to load them for volume matching."""
        check_dict["topup"] = True
        check_dict["dryrun"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])

        _run_process_run(info, cutoff=1000)

        text = normalize(recorder, _reps(info))
        assert "fslmerge" in text
        merge_lines = [l for l in text.splitlines() if "fslmerge" in l]
        niis = [tok for l in merge_lines for tok in l.split() if tok.endswith(".nii")]
        assert any("desc-denoised-median" in tok for tok in niis)


class TestDenoiseGating:
    def test_dipy_backed_methods_stop_the_run_when_dipy_is_missing(
        self, monkeypatch, capsys
    ):
        """Quietly continuing would write un-denoised results under a plain
        variant name and exit 0, so the user keeps derivatives believing they
        were denoised. Every other unusable-denoising path on this branch
        stops; this one does too."""
        monkeypatch.setattr(cp, "_dipy_available", lambda: False)

        with pytest.raises(SystemExit):
            cp.check_denoise("mppca")
        assert "dipy" in capsys.readouterr().out

    def test_median_needs_no_dipy(self, monkeypatch):
        monkeypatch.setattr(cp, "_dipy_available", lambda: False)

        assert cp.check_denoise("median") == "median"

    def test_no_method_stays_none(self):
        assert cp.check_denoise(None) is None


# ===========================================================================
# The invariant across CLI combinations
#
# Denoising has to compose with every other flag, not just the configuration
# it was developed against.  Each case runs process_run end to end through the
# capture seam and asserts the same two things: topup estimates from denoised
# images, applytopup corrects original ones.
# ===========================================================================


def _estimation_and_apply(text):
    """Split a normalized command log into (merge inputs, applytopup inputs).

    Under --matchcontrast a merge input can be the contrast-match OUTPUT, whose
    filename says nothing about denoising. Those are resolved back to the
    images the contrast-match command consumed, so the check follows the whole
    estimation chain rather than trusting the leaf name.
    """
    cm_sources: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if "<CMATCH>" in stripped:
            inputs = stripped.split(" -i ")[1].split(" -t ")[0].split()
            output = stripped.split(" -o ")[1].split()[0]
            cm_sources[output] = inputs

    merge_inputs, apply_inputs = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("fslmerge"):
            niis = [tok for tok in stripped.split() if tok.endswith(".nii")]
            for nii in niis[1:]:  # [0] is the merged output
                merge_inputs.extend(cm_sources.get(nii, [nii]))
        elif stripped.startswith("applytopup"):
            apply_inputs.append(stripped.split("--imain=")[1].split()[0])
    return merge_inputs, apply_inputs


CLI_COMBINATIONS = [
    ("whole_3echo", {}, {"n_echoes": 3, "n_avgs": 1}, 1000),
    ("whole_2echo", {}, {"n_echoes": 2, "n_avgs": 1}, 1000),
    ("whole_3echo_match", {"match": True}, {"n_echoes": 3, "n_avgs": 1}, 1000),
    ("slice_3echo", {"slice": True}, {"n_echoes": 3, "n_avgs": 1}, 1000),
    ("slice_2echo", {"slice": True}, {"n_echoes": 2, "n_avgs": 1}, 1000),
    (
        "slice_3echo_match",
        {"slice": True, "match": True},
        {"n_echoes": 3, "n_avgs": 1},
        1000,
    ),
    ("whole_diffusion", {}, {"n_echoes": 2, "bvals": [0, 1000, 2000]}, 1000),
    (
        "slice_diffusion",
        {"slice": True},
        {"n_echoes": 2, "bvals": [0, 1000, 2000]},
        1000,
    ),
    (
        "whole_diffusion_match",
        {"match": True},
        {"n_echoes": 3, "bvals": [0, 1000, 2000]},
        1000,
    ),
    ("twoecho_filter", {"two_echo": True}, {"n_echoes": 4, "n_avgs": 1}, 1000),
    ("single_bval", {}, {"n_echoes": 2, "bvals": [1000, 1000]}, 1000),
]


@pytest.mark.parametrize("case", CLI_COMBINATIONS, ids=[c[0] for c in CLI_COMBINATIONS])
def test_estimation_only_invariant_holds_for_every_cli_combination(
    case, check_dict, bids_run, slicing_recorder, normalize
):
    _, flags, run_kwargs, cutoff = case
    check_dict["topup"] = True
    check_dict["denoise"] = "median"
    check_dict.update(flags)
    info = bids_run(**run_kwargs)

    _run_process_run(info, cutoff=cutoff)

    text = normalize(slicing_recorder, _reps(info))
    merge_inputs, apply_inputs = _estimation_and_apply(text)

    assert merge_inputs, "no fslmerge command was produced"
    assert apply_inputs, "no applytopup command was produced"
    for nii in merge_inputs:
        assert "desc-denoised-median" in nii, (
            f"topup would estimate from an un-denoised image: {nii}"
        )
    for nii in apply_inputs:
        assert "desc-denoised" not in nii, (
            f"applytopup would correct a denoised image: {nii}"
        )


@pytest.mark.parametrize("case", CLI_COMBINATIONS, ids=[c[0] for c in CLI_COMBINATIONS])
def test_without_denoising_nothing_is_denoised(
    case, check_dict, bids_run, slicing_recorder, normalize
):
    """The same matrix with --denoise off: no denoise command, no denoise
    variant in any path, and both stages read the originals."""
    _, flags, run_kwargs, cutoff = case
    check_dict["topup"] = True
    check_dict["denoise"] = None
    check_dict.update(flags)
    info = bids_run(**run_kwargs)

    _run_process_run(info, cutoff=cutoff)

    text = normalize(slicing_recorder, _reps(info))
    assert "<DENOISE>" not in text
    assert "denoise-" not in text, "a denoise variant leaked into the paths"
    merge_inputs, apply_inputs = _estimation_and_apply(text)
    assert merge_inputs and apply_inputs
    assert not any("desc-denoised" in nii for nii in merge_inputs + apply_inputs)


# ===========================================================================
# Slice-number and b-volume-number correspondence
#
# Everything downstream is addressed by (bvol_num, slice_num): the merged
# file, the topup result base, the field, and the corrected output all carry
# bv-/sv- tags, and the estimation image for a slice is looked up by its
# parsed slice number. A mismatch here would correct the right voxels with the
# wrong slice's field and raise nothing, so these pin the correspondence
# rather than trusting it.
# ===========================================================================


def _tag(name, key):
    """Read the integer after ``_{key}-`` in a pipeline filename."""
    return int(name.split(f"_{key}-")[1].split("_")[0].split(".")[0])


class TestSliceAndVolumeIndexing:
    def test_every_slice_row_pairs_the_same_slice_number(
        self, check_dict, bids_run, slicing_recorder
    ):
        """est_nii must be the SAME slice of the denoised volume, not merely a
        slice of it."""
        check_dict["topup"] = True
        check_dict["slice"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, bvals=[0, 1000, 2000])

        _run_process_run(info, cutoff=3000)

        # rebuild the frame process_run worked from
        run_df = cp.get_run_info(
            os.path.join(
                info["output_dir"], info["subject"], info["session"], info["run"]
            ),
            info["subject"],
            info["session"],
            info["run"],
        )
        dir_dict = cp.set_dirs(
            info["subject"],
            info["session"],
            info["run"],
            info["output_dir"],
            info["deriv_dir"],
        )
        run_df = cp.handle_denoising(run_df, dir_dict)
        run_df["volume_type"] = "volume"
        run_df = cp.handle_slicing(run_df, dir_dict)

        slices = run_df[run_df["volume_type"] == "slice"]
        assert not slices.empty
        for _, row in slices.iterrows():
            assert cp.slice_number(row["nii"]) == row["slice_num"]
            assert cp.slice_number(row["est_nii"]) == row["slice_num"], (
                f"slice {row['slice_num']} paired with estimation slice "
                f"{cp.slice_number(row['est_nii'])}"
            )
            # and the estimation slice belongs to the denoised copy of the
            # SAME echo and b-volume, not another one
            assert f"echo-{row['echo']}_" in os.path.basename(row["est_nii"])

    def test_merge_inputs_all_come_from_one_slice(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        """Each per-slice topup merges the echoes OF THAT SLICE. Mixing slices
        would fit a field to two different pieces of anatomy."""
        check_dict["topup"] = True
        check_dict["slice"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        text = normalize(slicing_recorder, _reps(info))
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("fslmerge"):
                continue
            niis = [tok for tok in stripped.split() if tok.endswith(".nii")]
            merged, inputs = niis[0], niis[1:]
            expected = _tag(os.path.basename(merged), "sv")
            for nii in inputs:
                assert cp.slice_number(nii) == expected, (
                    f"merge for sv-{expected} pulled in slice "
                    f"{cp.slice_number(nii)}: {nii}"
                )

    def test_topup_and_apply_agree_on_bvol_and_slice(
        self, check_dict, bids_run, slicing_recorder, normalize
    ):
        """The field applied to (bvol, slice) must be the field estimated for
        that same (bvol, slice)."""
        check_dict["topup"] = True
        check_dict["slice"] = True
        check_dict["denoise"] = "median"
        info = bids_run(n_echoes=3, n_avgs=1)

        _run_process_run(info)

        text = normalize(slicing_recorder, _reps(info))
        checked = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("applytopup"):
                continue
            base = os.path.basename(stripped.split("--topup=")[1].split()[0])
            out = os.path.basename(stripped.split("--out=")[1].split()[0])
            assert _tag(base, "bv") == _tag(out, "bv")
            assert _tag(base, "sv") == _tag(out, "sv")
            checked += 1
        assert checked, "no applytopup commands to check"

    def test_bvol_numbers_index_the_bvals_they_were_read_with(
        self, check_dict, bids_run
    ):
        """bvol_num n must address the n-th volume AND the n-th bval; the
        denoise command's -i list is built from these two columns."""
        bvals = [0, 1000, 2000, 50]
        info = bids_run(n_echoes=2, bvals=bvals)

        run_df = cp.get_run_info(
            os.path.join(
                info["output_dir"], info["subject"], info["session"], info["run"]
            ),
            info["subject"],
            info["session"],
            info["run"],
        )

        for _, row in run_df.iterrows():
            assert row["bval"] == bvals[row["bvol_num"] - 1]
            assert os.path.basename(row["nii"]).endswith(f"_{row['bvol_num']}.nii")

    def test_denoise_command_lists_stay_aligned_under_shuffled_rows(
        self, check_dict, tmp_path, recorder
    ):
        """The -i list is positional and the outputs are mapped back by
        position. If the frame arrives in any order, sorting by bvol_num must
        still hand the volumes over in volume order."""
        check_dict["denoise"] = "mppca"
        run_df = _run_df(
            tmp_path,
            echoes=(1,),
            bvols=(3, 1, 2, 5, 4),
            bvals=(2000, 0, 1000, 1000, 1000),
        )
        run_df = run_df.sample(frac=1, random_state=0).reset_index(drop=True)

        cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        command = recorder[0][1][0]
        inputs = command.split(" -i ")[1].split(" -o ")[0].split()
        assert [_tag(os.path.basename(p) + "_", "echo") for p in inputs] == [
            1,
            1,
            1,
            1,
            1,
        ]
        assert [os.path.basename(p) for p in inputs] == [
            f"{SUBJ}_echo-1_1.nii",
            f"{SUBJ}_echo-1_2.nii",
            f"{SUBJ}_echo-1_3.nii",
            f"{SUBJ}_echo-1_4.nii",
            f"{SUBJ}_echo-1_5.nii",
        ]

    def test_volume_order_follows_bvol_num_not_the_filename(
        self, check_dict, tmp_path, recorder
    ):
        """The -i order must come from bvol_num, not from how the paths
        happen to sort.

        Every other fixture here names volumes so that lexicographic order
        matches numeric order, which cannot tell the two apart: replacing
        sort_values("bvol_num") with sort_values("nii") passes them all. These
        names sort in the OPPOSITE order to their b-volume numbers, so only
        the correct key gives volume 1 first.
        """
        check_dict["denoise"] = "mppca"
        # names sort z, y, x, w, v -- the exact reverse of bvol_num 1..5
        names = [
            "z_volume.nii",
            "y_volume.nii",
            "x_volume.nii",
            "w_volume.nii",
            "v_volume.nii",
        ]
        bvals = [0, 1000, 1000, 1000, 1000]
        run_df = pd.DataFrame(
            [
                {
                    "subject": SUBJ,
                    "session": SESS,
                    "run": RUN,
                    "echo": 1,
                    "bvol_num": i + 1,
                    "bval": bvals[i],
                    "nii": str(tmp_path / names[i]),
                    "orientation_num": 2,
                }
                for i in range(5)
            ]
        )

        cp.handle_denoising(run_df, {"work_dir": str(tmp_path)})

        command = recorder[0][1][0]
        inputs = command.split(" -i ")[1].split(" -o ")[0].split()
        assert [os.path.basename(p) for p in inputs] == names


# ===========================================================================
# Combinations the first matrix missed
# ===========================================================================


@pytest.mark.parametrize(
    "cutoff", [500, 2500], ids=["highb_reuse_active", "all_volumes_are_lowb"]
)
def test_cutoff_selects_the_path_and_the_invariant_holds_in_both(
    cutoff, check_dict, bids_run, slicing_recorder, normalize
):
    """--cutoff decides whether high-b volumes get their own TOPUP fit or reuse
    a low-b field chosen by NMI matching.

    The reuse branch reads the ESTIMATION images to pick the closest low-b
    volume, and corrects the ORIGINAL high-b ones. At --cutoff 1000 with b=0
    and b=1000 data that branch never runs, which is every configuration this
    feature was evaluated on -- so it needs its own case.
    """
    check_dict["topup"] = True
    check_dict["denoise"] = "median"
    info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])

    _run_process_run(info, cutoff=cutoff)

    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    reuse_batch = [
        c
        for d, cmds in slicing_recorder
        if d == "ApplyTOPUP Commands for high b diffusion"
        for c in cmds
    ]
    if cutoff == 500:
        assert reuse_batch, "expected high-b volumes to reuse a low-b field"
    else:
        assert not reuse_batch, "no volume is above a 2500 cutoff"

    merge_inputs, apply_inputs = _estimation_and_apply(text)
    assert merge_inputs and apply_inputs
    assert all("desc-denoised-median" in nii for nii in merge_inputs)
    assert not any("desc-denoised" in nii for nii in apply_inputs)


def test_masking_and_denoising_compose(
    check_dict, bids_run, slicing_recorder, normalize, tmp_path, tiny_nii
):
    """--maskdir rewrites nii to the masked copies BEFORE denoising runs.

    So the estimation images are denoised masked copies, and applytopup
    corrects the masked originals -- masked, but never denoised.
    """
    check_dict["topup"] = True
    check_dict["denoise"] = "median"
    info = bids_run(n_echoes=3, n_avgs=1)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tiny_nii(
        mask_dir / f"{info['subject']}_{info['session']}_{info['run']}"
        "_desc-brain_mask.nii"
    )

    _run_process_run(info, mask_dir=str(mask_dir))

    text = normalize(slicing_recorder, _reps(info))
    descs = [d for d, _ in slicing_recorder]
    assert descs[0] == "masking commands"
    assert descs[1] == "denoising commands", (
        f"denoising must follow masking, got {descs[:3]}"
    )

    # the denoiser must be handed the masked copies, not the raw inputs
    denoise_lines = [l for l in text.splitlines() if "<DENOISE>" in l]
    assert denoise_lines
    for line in denoise_lines:
        for nii in line.split(" -i ")[1].split(" -o ")[0].split():
            assert "desc-masked" in nii, f"denoised a raw input: {nii}"

    merge_inputs, apply_inputs = _estimation_and_apply(text)
    assert all("desc-denoised-median" in nii for nii in merge_inputs)
    for nii in apply_inputs:
        assert "desc-denoised" not in nii
        assert "desc-masked" in nii, "applytopup should correct masked images"


@pytest.mark.parametrize("method", ["median", "mppca"])
def test_every_method_composes_with_slice_and_match(
    method, check_dict, bids_run, slicing_recorder, normalize
):
    """The matrix runs median throughout because it needs no dipy; the
    flagship configuration is checked for each method here."""
    check_dict["topup"] = True
    check_dict["slice"] = True
    check_dict["match"] = True
    check_dict["denoise"] = method
    info = bids_run(n_echoes=3, bvals=[0, 1000, 1000, 1000, 1000])

    _run_process_run(info, cutoff=1000)

    text = normalize(slicing_recorder, _reps(info))
    denoise_lines = [l for l in text.splitlines() if "<DENOISE>" in l]
    assert denoise_lines, f"no denoise command for {method}"
    for line in denoise_lines:
        assert f" -m {method} " in line

    merge_inputs, apply_inputs = _estimation_and_apply(text)
    assert all(f"desc-denoised-{method}" in nii for nii in merge_inputs)
    assert not any("desc-denoised" in nii for nii in apply_inputs)


def test_workdir_moves_the_denoised_copies_too(
    check_dict, bids_run, slicing_recorder, tmp_path
):
    """--workdir relocates the whole work tree; the denoised estimation images
    are intermediates and must follow it, not stay under derivdir."""
    check_dict["topup"] = True
    check_dict["denoise"] = "median"
    info = bids_run(n_echoes=3, n_avgs=1)
    work_root = tmp_path / "elsewhere"

    cp.process_run(
        info["subject"],
        info["session"],
        info["run"],
        info["output_dir"],
        info["deriv_dir"],
        "b02b0.cnf",
        None,
        1000,
        str(work_root),
    )

    # raw commands, not the normalized text: normalize() rewrites tmp_path to
    # a placeholder, which would defeat a path-prefix check
    denoise_commands = [
        c
        for description, cmds in slicing_recorder
        if description == "denoising commands"
        for c in cmds
    ]
    assert denoise_commands
    for command in denoise_commands:
        out_dir = command.split(" -o ")[1].split()[0]
        assert out_dir.startswith(str(work_root)), (
            f"denoised copies went to {out_dir}, not under --workdir"
        )


@pytest.mark.parametrize(
    "cutoff,expected_own_fit,expected_reuse",
    [
        (1000, {1, 2}, {3}),  # b=1000 is AT the cutoff -> its own fit
        (999, {1}, {2, 3}),  # one below -> b=1000 reuses the b=0 field
    ],
)
def test_cutoff_is_inclusive_on_the_low_b_side(
    cutoff,
    expected_own_fit,
    expected_reuse,
    check_dict,
    bids_run,
    slicing_recorder,
    normalize,
):
    """`bval <= cutoff` is low-b, so a volume AT the cutoff gets its own TOPUP
    fit rather than reusing another volume's field.

    This is the boundary that decides how the default behaves on ordinary
    b=0/b=1000 data: at --cutoff 1000 every volume is fitted individually and
    the reuse path never runs at all.
    """
    check_dict["topup"] = True
    check_dict["denoise"] = "median"
    info = bids_run(n_echoes=2, bvals=[0, 1000, 2000])

    _run_process_run(info, cutoff=cutoff)

    own_fit, reuse = set(), set()
    for description, commands in slicing_recorder:
        for command in commands:
            for part in command.split(" && "):
                if not part.strip().startswith("applytopup"):
                    continue
                out = os.path.basename(part.split("--out=")[1].split()[0])
                bvol = int(out.split("_bv-")[1].split("_")[0])
                if description == "ApplyTOPUP Commands for high b diffusion":
                    reuse.add(bvol)
                else:
                    own_fit.add(bvol)

    assert own_fit == expected_own_fit, (
        f"cutoff {cutoff}: volumes fitted individually were {sorted(own_fit)}"
    )
    assert reuse == expected_reuse, (
        f"cutoff {cutoff}: volumes reusing a field were {sorted(reuse)}"
    )
