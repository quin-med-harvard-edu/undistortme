"""Unit tests for ``undistortme.denoise``.

The denoise module is a standalone CLI (like contrastmatch) that the pipeline
invokes as a subprocess.  It denoises a whole echo's volume series so that the
result can be used for FIELD ESTIMATION ONLY -- the pipeline keeps applying the
correction to the original images, so nothing here writes over its inputs.
"""

import os

import nibabel as nib
import numpy as np
import pytest

from undistortme import denoise as dn


# ===========================================================================
# In-plane median
# ===========================================================================


class TestMedianDenoise:
    def test_removes_isolated_in_plane_spike(self):
        """A lone hot voxel is a 3x3 in-plane median's textbook target."""
        data = np.zeros((8, 8, 4, 1), dtype=np.float32)
        data[4, 4, 2, 0] = 100.0

        out = dn.denoise_median(data)

        assert out[4, 4, 2, 0] == 0.0

    def test_preserves_contrast_between_slices(self):
        """The filter is 2D in-plane: it must NOT mix neighbouring slices.

        Each slice is uniform but different from its neighbours; an in-plane
        median returns each slice unchanged, while a 3D median would pull the
        slices toward each other.
        """
        data = np.zeros((8, 8, 4, 1), dtype=np.float32)
        for k in range(4):
            data[:, :, k, 0] = float(k + 1)

        out = dn.denoise_median(data)

        np.testing.assert_allclose(out[:, :, :, 0], data[:, :, :, 0])

    def test_denoises_each_volume_independently(self):
        """Volume 1's spike must not leak into volume 0."""
        data = np.zeros((8, 8, 4, 2), dtype=np.float32)
        data[4, 4, 2, 1] = 100.0

        out = dn.denoise_median(data)

        assert out.shape == data.shape
        assert out[4, 4, 2, 1] == 0.0
        assert np.all(out[:, :, :, 0] == 0.0)

    def test_slice_axis_selects_the_through_plane_direction(self):
        """With slice_axis=0 the uniform-per-slice test flips to the x axis."""
        data = np.zeros((4, 8, 8, 1), dtype=np.float32)
        for i in range(4):
            data[i, :, :, 0] = float(i + 1)

        out = dn.denoise_median(data, slice_axis=0)

        np.testing.assert_allclose(out[:, :, :, 0], data[:, :, :, 0])


# ===========================================================================
# Method dispatch and validation
# ===========================================================================


class TestApplyDenoise:
    def test_dispatches_median(self):
        data = np.zeros((8, 8, 4, 2), dtype=np.float32)
        data[4, 4, 2, 0] = 100.0

        out = dn.apply_denoise(data, "median")

        assert out[4, 4, 2, 0] == 0.0

    def test_unknown_method_is_an_error(self):
        data = np.zeros((8, 8, 4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="nonsense"):
            dn.apply_denoise(data, "nonsense")

    def test_dipy_methods_refuse_a_series_too_short_to_denoise(self):
        """mppca on one volume returns altered data without complaining, and
        the pipeline discards subprocess stdout on success, so a warning would
        never reach the user. It has to be an error."""
        data = np.zeros((8, 8, 4, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="volume"):
            dn.apply_denoise(data, "mppca")

    def test_median_is_fine_on_a_single_volume(self):
        """It is a spatial filter: it needs no redundancy across volumes."""
        data = np.zeros((8, 8, 4, 1), dtype=np.float32)
        data[4, 4, 2, 0] = 100.0

        assert dn.apply_denoise(data, "median")[4, 4, 2, 0] == 0.0


# ===========================================================================
# Slice axis
# ===========================================================================


class TestSliceAxis:
    def test_comes_from_the_image_header(self, tmp_path, tiny_nii):
        """dcm2niix writes the acquired slice direction as the last array axis
        whatever the anatomy, and records it in dim_info; an axial and a
        coronal acquisition from the same session both report slice axis 2.
        Deriving it from the anatomical orientation instead would filter across
        slices on non-axial runs."""
        path = tiny_nii(tmp_path / "img.nii")
        img = nib.load(path)
        img.header.set_dim_info(freq=0, phase=1, slice=2)

        assert dn.slice_axis_of(img) == 2

    def test_falls_back_to_the_last_spatial_axis(self, tmp_path, tiny_nii):
        """Headers that never had dim_info set still have to yield an axis."""
        img = nib.load(tiny_nii(tmp_path / "img.nii"))

        assert dn.slice_axis_of(img) == 2


# ===========================================================================
# Output naming (the pipeline predicts these paths without running anything)
# ===========================================================================


class TestOutputPath:
    def test_appends_suffix_in_the_output_directory(self):
        out = dn.output_path("/in/sub-01_echo-1_01.nii", "/work", "desc-denoised-mppca")

        assert out == "/work/sub-01_echo-1_01_desc-denoised-mppca.nii"

    def test_relative_output_dir_is_joined_once(self):
        """The original standalone script joined the output dir twice; this
        pins the fix."""
        out = dn.output_path("in/img.nii", "work", "desc-denoised-median")

        assert out == os.path.join("work", "img_desc-denoised-median.nii")


# ===========================================================================
# CLI
# ===========================================================================


class TestCli:
    def test_writes_one_output_per_input_volume(self, tmp_path, tiny_nii, monkeypatch):
        inputs = [
            tiny_nii(tmp_path / f"img_{i}.nii", shape=(8, 8, 4), seed=i)
            for i in range(3)
        ]
        out_dir = tmp_path / "work"
        out_dir.mkdir()
        monkeypatch.setattr(
            "sys.argv",
            [
                "undistortme-denoise",
                "-m",
                "median",
                "-o",
                str(out_dir),
                "--suffix",
                "desc-denoised-median",
                "-i",
                *inputs,
            ],
        )

        dn.main()

        for path in inputs:
            expected = dn.output_path(path, str(out_dir), "desc-denoised-median")
            assert os.path.exists(expected)

    def test_preserves_affine_and_shape(self, tmp_path, tiny_nii, monkeypatch):
        inputs = [
            tiny_nii(tmp_path / f"img_{i}.nii", shape=(8, 8, 4), seed=i)
            for i in range(2)
        ]
        out_dir = tmp_path / "work"
        out_dir.mkdir()
        monkeypatch.setattr(
            "sys.argv",
            [
                "undistortme-denoise",
                "-m",
                "median",
                "-o",
                str(out_dir),
                "--suffix",
                "dn",
                "-i",
                *inputs,
            ],
        )

        dn.main()

        src = nib.load(inputs[0])
        got = nib.load(dn.output_path(inputs[0], str(out_dir), "dn"))
        assert got.shape == src.shape
        np.testing.assert_allclose(got.affine, src.affine)

    def test_slice_axis_outside_the_volume_is_rejected(self, tmp_path):
        """3 is exactly what get_echo_info assigns to an unknown orientation,
        so an unvalidated value reaches here as an IndexError deep in the
        filter rather than a usable message."""
        with pytest.raises(SystemExit):
            dn.parse_args(
                [
                    "-m",
                    "median",
                    "-o",
                    str(tmp_path),
                    "--suffix",
                    "dn",
                    "--slice_axis",
                    "3",
                    "-i",
                    "a.nii",
                ]
            )


# ===========================================================================
# dipy-backed methods (optional dependency)
# ===========================================================================


class TestDipyMethods:
    def test_mppca_returns_same_shape(self):
        pytest.importorskip("dipy")
        rng = np.random.default_rng(0)
        data = rng.random((10, 10, 6, 8)).astype(np.float32)

        out = dn.apply_denoise(data, "mppca", patch_radius=1)

        assert out.shape == data.shape

    def test_mppca_is_deterministic(self):
        """The same series must give the same field on every run; mppca has
        no random step, and this pins that the call stays that way."""
        pytest.importorskip("dipy")
        rng = np.random.default_rng(0)
        data = rng.random((10, 10, 6, 8)).astype(np.float32)

        first = dn.apply_denoise(data, "mppca", patch_radius=1)
        second = dn.apply_denoise(data, "mppca", patch_radius=1)

        assert np.array_equal(first, second)


class TestSuffixDefault:
    def test_cli_default_suffix_matches_what_the_pipeline_predicts(
        self, tmp_path, tiny_nii, monkeypatch
    ):
        """A standalone run with no --suffix must land on the same filename the
        pipeline's resume check looks for, or the two produce parallel outputs
        that never see each other."""
        inputs = [tiny_nii(tmp_path / "img.nii", shape=(8, 8, 4))]
        out_dir = tmp_path / "work"
        monkeypatch.setattr(
            "sys.argv",
            ["undistortme-denoise", "-m", "median", "-o", str(out_dir), "-i", *inputs],
        )

        dn.main()

        expected = dn.output_path(inputs[0], str(out_dir), dn.suffix_for("median"))
        assert os.path.exists(expected)


class TestOutputPathExtensions:
    def test_strips_a_gzipped_extension(self):
        """foo.nii.gz must not become foo.nii.gz_desc-....nii -- and the
        stripped stem must differ from the input's, since handle_slicing keys
        slice directories on basename.split('.')[0]."""
        out = dn.output_path("/in/foo.nii.gz", "/work", "desc-denoised-mppca")

        assert out == "/work/foo_desc-denoised-mppca.nii"
        assert (
            os.path.basename(out).split(".")[0]
            != os.path.basename("/in/foo.nii.gz").split(".")[0]
        )


class TestInputOrderIsAuthoritative:
    """The CLI must never re-sort its inputs.

    The pipeline hands over -i in bvol_num order and maps each input to its
    own output by position. If the denoiser re-sorted the paths it received,
    a run whose filenames do not sort in volume order would silently get each
    denoised image written under another volume's name, with no error
    anywhere.
    """

    def test_each_output_matches_its_own_input(self, tmp_path, tiny_nii, monkeypatch):
        # given in an order that is the REVERSE of how the names sort
        given = [
            tiny_nii(tmp_path / "z.nii", shape=(8, 8, 4), fill=3.0),
            tiny_nii(tmp_path / "m.nii", shape=(8, 8, 4), fill=2.0),
            tiny_nii(tmp_path / "a.nii", shape=(8, 8, 4), fill=1.0),
        ]
        assert [os.path.basename(p) for p in sorted(given)] != [
            os.path.basename(p) for p in given
        ], "fixture must not be sorted"
        out_dir = tmp_path / "work"
        monkeypatch.setattr(
            "sys.argv",
            ["undistortme-denoise", "-m", "median", "-o", str(out_dir), "-i", *given],
        )

        dn.main()

        # a median of a constant volume is that constant, so each output must
        # still carry the value of the input it is named after
        for path, expected in zip(given, [3.0, 2.0, 1.0]):
            out = dn.output_path(path, str(out_dir), dn.suffix_for("median"))
            data = np.asarray(nib.load(out).dataobj, dtype=np.float64)
            assert np.allclose(data, expected), (
                f"{os.path.basename(out)} carries {data.mean():.1f}, "
                f"expected {expected}"
            )

    def test_the_stacked_series_keeps_the_given_order(self, tmp_path, tiny_nii):
        """load_series stacks along the volume axis in argument order, which is
        what keeps each output paired with its own input."""
        given = [
            tiny_nii(tmp_path / "z.nii", shape=(8, 8, 4), fill=3.0),
            tiny_nii(tmp_path / "m.nii", shape=(8, 8, 4), fill=2.0),
            tiny_nii(tmp_path / "a.nii", shape=(8, 8, 4), fill=1.0),
        ]

        data, images = dn.load_series(given)

        assert data.shape[-1] == 3
        assert [float(data[..., i].mean()) for i in range(3)] == [3.0, 2.0, 1.0]
        assert [os.path.basename(img.get_filename()) for img in images] == [
            "z.nii",
            "m.nii",
            "a.nii",
        ]
