"""Unit tests for the BIDS ingest adapter (undistortme.bids).

Pure adapter tests, no snapshots: entity parsing, sidecar inheritance,
grouping and run-token synthesis, collision detection, 4D splitting
(naming, zero-padding, atomic round-trip), EchoNumber injection, bval
propagation, session synthesis, idempotency, and the PE-sign warning.

The staged trees produced here are consumed unmodified by process_run;
tests proving THAT are the snapshot-reuse cases in
test_process_run_orchestration.py.
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from undistortme import bids as ub


# ===========================================================================
# Entity parsing
# ===========================================================================


class TestParseEntities:
    def test_full_dwi_filename(self):
        entities, suffix, ext = ub.parse_entities(
            "/x/sub-01_ses-02_acq-highres_dir-AP_run-03_echo-2_dwi.nii.gz"
        )
        assert entities == {
            "sub": "01",
            "ses": "02",
            "acq": "highres",
            "dir": "AP",
            "run": "03",
            "echo": "2",
        }
        assert suffix == "dwi"
        assert ext == ".nii.gz"

    def test_plain_nii_extension(self):
        entities, suffix, ext = ub.parse_entities("sub-01_dwi.nii")
        assert entities == {"sub": "01"}
        assert suffix == "dwi"
        assert ext == ".nii"

    def test_suffix_only_json(self):
        """Root-level inheritance files can be suffix-only (dwi.json)."""
        entities, suffix, ext = ub.parse_entities("dwi.json")
        assert entities == {}
        assert suffix == "dwi"
        assert ext == ".json"

    def test_label_containing_dash(self):
        """Only the first dash separates key from label."""
        entities, _, _ = ub.parse_entities("sub-01_acq-b-1000_dwi.nii")
        assert entities["acq"] == "b-1000"


# ===========================================================================
# Run-token synthesis
# ===========================================================================


class TestMakeRunToken:
    """Tokens always lead with run- because main()'s legacy glob is run-*;
    the exact-token assertions below pin that along with the entity order."""

    def test_run_entity_passthrough(self):
        assert ub.make_run_token({"run": "02"}) == "run-02"

    def test_no_entities_synthesizes_run01(self):
        assert ub.make_run_token({}) == "run-01"

    def test_extra_entities_appended_in_canonical_order(self):
        token = ub.make_run_token({"dir": "AP", "acq": "highres"})
        assert token == "run-01_acq-highres_dir-AP"


# ===========================================================================
# Indexing and grouping
# ===========================================================================


class TestIndexAndGroup:
    def test_multi_echo_files_form_one_group(self, bids_dwi_dataset):
        info = bids_dwi_dataset(n_echoes=3, n_vols=2)
        records = ub.index_bids_dwi(info["bids_dir"])
        assert len(records) == 3
        groups = ub.group_runs(records)
        assert list(groups) == [("sub-01", "ses-01", "run-01")]
        assert sorted(r["echo"] for r in groups[("sub-01", "ses-01", "run-01")]) == [
            1,
            2,
            3,
        ]

    def test_different_acq_different_groups(self, tmp_path, bids_dwi_dataset):
        bids_dwi_dataset(n_echoes=2, acq="fast")
        info = bids_dwi_dataset(n_echoes=2, acq="slow")
        records = ub.index_bids_dwi(info["bids_dir"])
        groups = ub.group_runs(records)
        assert sorted(groups) == [
            ("sub-01", "ses-01", "run-01_acq-fast"),
            ("sub-01", "ses-01", "run-01_acq-slow"),
        ]

    def test_part_phase_skipped_by_default(self, bids_dwi_dataset, capsys):
        info = bids_dwi_dataset(n_echoes=2, parts=("mag", "phase"))
        records = ub.index_bids_dwi(info["bids_dir"])
        assert len(records) == 2
        assert all("part-mag" in r["path"] for r in records)
        assert "part-phase" in capsys.readouterr().out

    def test_token_collision_is_hard_error(self, bids_dwi_dataset):
        """Two groups differing only in a token-dropped entity collide.

        part- is excluded from the run token (only part-mag is staged by
        default), so forcing part-mag AND no-part images into the index
        must raise rather than silently merge or overwrite.
        """
        info = bids_dwi_dataset(n_echoes=2, parts=("mag",))
        # add an identical image WITHOUT the part entity
        dwi = Path(info["dwi_dir"])
        for e in (1, 2):
            src = dwi / f"sub-01_ses-01_run-01_echo-{e}_part-mag_dwi.nii.gz"
            (dwi / f"sub-01_ses-01_run-01_echo-{e}_dwi.nii.gz").write_bytes(
                src.read_bytes()
            )
        records = ub.index_bids_dwi(info["bids_dir"])
        with pytest.raises(RuntimeError, match="collision"):
            ub.group_runs(records)

    def test_duplicate_echo_numbers_are_hard_error(self, bids_dwi_dataset):
        """Files whose echoes cannot be told apart (all default to 1) fail."""
        info = bids_dwi_dataset(n_echoes=2, echo_in_acq=True)
        # without --echo-from-acq the acq-e1/acq-e2 files fall back to
        # EchoNumber (absent) -> both echo 1, but acq differs so they land
        # in different groups.  Force the collision by dropping acq from
        # the records' entities.
        records = ub.index_bids_dwi(info["bids_dir"])
        for rec in records:
            rec["entities"].pop("acq")
        with pytest.raises(RuntimeError, match="[Dd]uplicate echo"):
            ub.group_runs(records)

    def test_echo_from_acq_regex(self, bids_dwi_dataset):
        info = bids_dwi_dataset(n_echoes=3, echo_in_acq=True)
        records = ub.index_bids_dwi(info["bids_dir"], echo_from_acq=r"e(\d+)$")
        assert sorted(r["echo"] for r in records) == [1, 2, 3]
        # acq carried the echo, so it is dropped from grouping: one group
        groups = ub.group_runs(records)
        assert list(groups) == [("sub-01", "ses-01", "run-01")]

    def test_echo_from_acq_without_group_is_hard_error(self, bids_dwi_dataset):
        """A group-less --echo-from-acq pattern is rejected up front."""
        info = bids_dwi_dataset(n_echoes=2, echo_in_acq=True)
        with pytest.raises(ValueError, match="group 1"):
            ub.index_bids_dwi(info["bids_dir"], echo_from_acq=r"e\d+")

    def test_echo_from_sidecar_echonumber_fallback(self, bids_dwi_dataset):
        """No echo entity, no acq match -> sidecar EchoNumber decides."""
        info = bids_dwi_dataset(n_echoes=2, echo_in_acq=True, include_echo_number=True)
        records = ub.index_bids_dwi(info["bids_dir"])
        assert sorted(r["echo"] for r in records) == [1, 2]


# ===========================================================================
# Sidecar inheritance
# ===========================================================================


class TestInheritance:
    def test_leaf_sidecar_resolved(self, bids_dwi_dataset):
        info = bids_dwi_dataset(n_echoes=2)
        records = ub.index_bids_dwi(info["bids_dir"])
        for rec in records:
            assert rec["sidecar"]["TotalReadoutTime"] == 0.106487
            assert rec["sidecar"]["EchoTime"] in (0.07, 0.09)

    def test_subject_level_sidecar_applies(self, bids_dwi_dataset):
        """A sidecar at sub-01/ with a subset of the image's entities
        (sub + echo) is picked up via the inheritance walk."""
        info = bids_dwi_dataset(n_echoes=2, sidecar_level="sub")
        records = ub.index_bids_dwi(info["bids_dir"])
        assert len(records) == 2
        for rec in records:
            assert rec["sidecar"]["TotalReadoutTime"] == 0.106487

    def test_root_common_fields_merged_with_leaf(self, bids_dwi_dataset):
        """Root dwi.json supplies common fields, the leaf sidecar the
        per-echo ones; deeper values win on conflict."""
        info = bids_dwi_dataset(n_echoes=2, sidecar_level="split")
        # give the root file a value the leaf overrides
        root_json = Path(info["bids_dir"]) / "dwi.json"
        data = json.loads(root_json.read_text())
        data["EchoTime"] = 99.0
        root_json.write_text(json.dumps(data))

        records = ub.index_bids_dwi(info["bids_dir"])
        for rec in records:
            assert rec["sidecar"]["TotalReadoutTime"] == 0.106487  # root
            assert rec["sidecar"]["EchoTime"] in (0.07, 0.09)  # leaf wins

    def test_bval_resolved_next_to_image(self, bids_dwi_dataset):
        info = bids_dwi_dataset(n_echoes=2, bvals=[0, 1000])
        records = ub.index_bids_dwi(info["bids_dir"])
        for rec in records:
            assert rec["bval_path"] is not None
            assert rec["bvec_path"] is not None
            assert Path(rec["bval_path"]).read_text() == "0 1000"


# ===========================================================================
# Splitting
# ===========================================================================


class TestSplitEchoVolumes:
    def test_split_naming_and_zero_pad(self, bids_dwi_dataset, tmp_path):
        """12 volumes -> _01.. _12 (width = len(str(N)))."""
        info = bids_dwi_dataset(n_echoes=1, n_vols=12)
        out_dir = tmp_path / "staged"
        out_dir.mkdir()
        paths = ub.split_echo_volumes(info["image_paths"][0], str(out_dir), "stem")
        assert len(paths) == 12
        assert os.path.basename(paths[0]) == "stem_01.nii"
        assert os.path.basename(paths[-1]) == "stem_12.nii"
        assert all(os.path.exists(p) for p in paths)

    def test_split_round_trip(self, bids_dwi_dataset, tmp_path):
        """Staged volume k must equal the 4D's [..., k] exactly."""
        info = bids_dwi_dataset(n_echoes=1, n_vols=3)
        src = info["image_paths"][0]
        out_dir = tmp_path / "staged"
        out_dir.mkdir()
        paths = ub.split_echo_volumes(src, str(out_dir), "stem")
        full = np.asanyarray(nib.load(src).dataobj)
        for k, path in enumerate(paths):
            vol = np.asanyarray(nib.load(path).dataobj)
            assert np.array_equal(vol, full[..., k])

    def test_3d_input_treated_as_single_volume(self, tmp_path):
        img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
        src = tmp_path / "img.nii.gz"
        nib.save(img, str(src))
        paths = ub.split_echo_volumes(str(src), str(tmp_path), "stem")
        assert [os.path.basename(p) for p in paths] == ["stem_1.nii"]

    def test_bval_count_mismatch_is_hard_error(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=1, n_vols=3)
        with pytest.raises(ValueError, match="volumes"):
            ub.split_echo_volumes(
                info["image_paths"][0], str(tmp_path), "stem", expected_vols=5
            )

    def test_split_writes_are_atomic(self, bids_dwi_dataset, tmp_path, monkeypatch):
        """A kill mid-save must not leave a truncated output that the next run
        treats as already staged: every save goes to a temp name first."""
        info = bids_dwi_dataset(n_echoes=1, n_vols=3)
        saved = []
        real_save = nib.save

        def _spy(img, path):
            saved.append(str(path))
            real_save(img, path)

        monkeypatch.setattr(ub.nib, "save", _spy)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        nii = next(Path(info["bids_dir"]).rglob("*_dwi.nii*"))

        outputs = ub.split_echo_volumes(str(nii), str(out_dir), "stem")

        assert all(p.endswith(".tmp.nii") for p in saved), saved
        assert all(os.path.exists(p) for p in outputs)
        assert not list(out_dir.glob("*.tmp.nii"))


# ===========================================================================
# Staging end-to-end
# ===========================================================================


class TestStageBids:
    def test_staged_tree_matches_legacy_layout(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=3, n_vols=2)
        stage = tmp_path / "stage"
        staged = ub.stage_bids(info["bids_dir"], str(stage))
        assert staged == [("sub-01", "ses-01", "run-01")]
        run_dir = stage / "sub-01" / "ses-01" / "run-01"
        for e in (1, 2, 3):
            stem = f"sub-01_ses-01_run-01_echo-{e}"
            assert (run_dir / f"{stem}.json").exists()
            assert (run_dir / f"{stem}_1.nii").exists()
            assert (run_dir / f"{stem}_2.nii").exists()

    def test_echonumber_injected_when_absent(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=2, include_echo_number=False)
        stage = tmp_path / "stage"
        ub.stage_bids(info["bids_dir"], str(stage))
        run_dir = stage / "sub-01" / "ses-01" / "run-01"
        for e in (1, 2):
            sidecar = json.loads(
                (run_dir / f"sub-01_ses-01_run-01_echo-{e}.json").read_text()
            )
            assert sidecar["EchoNumber"] == e
            assert sidecar["EchoTime"] == (0.07, 0.09)[e - 1]

    def test_bval_copied_per_echo(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=2, bvals=[0, 1000, 2000])
        stage = tmp_path / "stage"
        ub.stage_bids(info["bids_dir"], str(stage))
        run_dir = stage / "sub-01" / "ses-01" / "run-01"
        for e in (1, 2):
            bval = run_dir / f"sub-01_ses-01_run-01_echo-{e}.bval"
            assert bval.read_text() == "0 1000 2000"
            assert (run_dir / f"sub-01_ses-01_run-01_echo-{e}.bvec").exists()

    def test_session_synthesized_when_absent(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=2, session=None)
        stage = tmp_path / "stage"
        staged = ub.stage_bids(info["bids_dir"], str(stage))
        assert staged == [("sub-01", "ses-01", "run-01")]
        assert (
            stage / "sub-01" / "ses-01" / "run-01" / "sub-01_ses-01_run-01_echo-1_1.nii"
        ).exists()

    def test_idempotent_rerun_keeps_files(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=2, n_vols=2)
        stage = tmp_path / "stage"
        first = ub.stage_bids(info["bids_dir"], str(stage))
        snapshot = {
            str(p): p.stat().st_mtime_ns for p in stage.rglob("*") if p.is_file()
        }
        second = ub.stage_bids(info["bids_dir"], str(stage))
        assert first == second
        after = {str(p): p.stat().st_mtime_ns for p in stage.rglob("*") if p.is_file()}
        assert after == snapshot, "re-staging must not rewrite staged files"

    def test_plain_nii_input_also_staged(self, bids_dwi_dataset, tmp_path):
        info = bids_dwi_dataset(n_echoes=2, gz=False)
        stage = tmp_path / "stage"
        staged = ub.stage_bids(info["bids_dir"], str(stage))
        assert staged == [("sub-01", "ses-01", "run-01")]

    def test_empty_dataset_returns_no_runs(self, tmp_path):
        """Empty dataset -> [] (the CLI entry points own the error print)."""
        empty = tmp_path / "empty-bids"
        empty.mkdir()
        assert ub.stage_bids(str(empty), str(tmp_path / "stage")) == []

    def test_missing_bids_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ub.stage_bids(str(tmp_path / "nope"), str(tmp_path / "stage"))


# ===========================================================================
# PE-sign warning (report risk 4)
# ===========================================================================


class TestPeSignWarning:
    def test_no_warning_when_signs_alternate(self, bids_dwi_dataset, tmp_path, capsys):
        info = bids_dwi_dataset(n_echoes=2)  # default: "j", "j-"
        ub.stage_bids(info["bids_dir"], str(tmp_path / "stage"))
        out = capsys.readouterr().out
        assert "do not alternate" not in out

    def test_warning_when_signs_do_not_alternate(
        self, bids_dwi_dataset, tmp_path, capsys
    ):
        info = bids_dwi_dataset(n_echoes=2, phase_dirs=["j-", "j-"])
        ub.stage_bids(info["bids_dir"], str(tmp_path / "stage"))
        out = capsys.readouterr().out
        assert "do not alternate" in out
        assert "MISMATCH" in out

    def test_warning_on_mixed_axes(self, bids_dwi_dataset, tmp_path, capsys):
        info = bids_dwi_dataset(n_echoes=2, phase_dirs=["j", "i-"])
        ub.stage_bids(info["bids_dir"], str(tmp_path / "stage"))
        assert "different phase-encoding AXES" in capsys.readouterr().out


# ===========================================================================
# BIDS-shape hint helper + standalone CLI
# ===========================================================================


class TestLooksLikeBids:
    def test_true_for_bids_tree(self, bids_dwi_dataset):
        info = bids_dwi_dataset(n_echoes=2)
        assert ub.looks_like_bids(info["bids_dir"]) is True

    def test_false_for_legacy_run_tree(self, tmp_path):
        (tmp_path / "sub-01" / "ses-01" / "run-01").mkdir(parents=True)
        assert ub.looks_like_bids(str(tmp_path)) is False

    def test_false_for_empty_dir(self, tmp_path):
        assert ub.looks_like_bids(str(tmp_path)) is False


class TestBidsifyCli:
    def test_main_stages_and_prints_next_step(
        self, bids_dwi_dataset, tmp_path, monkeypatch, capsys
    ):
        info = bids_dwi_dataset(n_echoes=2)
        stage = tmp_path / "stage"
        monkeypatch.setattr(
            "sys.argv", ["undistortme-bidsify", info["bids_dir"], str(stage)]
        )
        ub.main()
        assert (
            stage / "sub-01" / "ses-01" / "run-01" / "sub-01_ses-01_run-01_echo-1_1.nii"
        ).exists()
        assert "undistortme -t -o" in capsys.readouterr().out

    def test_main_exits_nonzero_when_nothing_staged(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(
            "sys.argv", ["undistortme-bidsify", str(empty), str(tmp_path / "stage")]
        )
        with pytest.raises(SystemExit) as excinfo:
            ub.main()
        assert excinfo.value.code == 1
