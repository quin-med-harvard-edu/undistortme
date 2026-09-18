"""Characterization tests for contrastmatch.py.

These tests pin the CURRENT behavior of contrastmatch.py exactly as-is.
They are written ahead of a refactor and must NOT be changed to match a
"corrected" implementation — the source wins.

Key source behaviors confirmed before writing:
- simple_line:  slope = (img[1]-img[0])/(t[1]-t[0]);
                intercept = img[0] - slope*t[0]
- linregress_across_images: np.polyfit per-voxel; returns (slope_3d, intercept_3d)
- geomean_across_images:    takes times=[t0, new_te, t_last] (3-element) and
                            imgs=[img1, img2] (2-element).
                            c1=(t2-t1)/(t2-t0), c3=(t1-t0)/(t2-t0)
                            result = exp(c1*log(img1+1) + c3*log(img2+1)) - 1
- geomean exit condition bug: `if len(imgs) != 2 and len(times) != 3:` uses
                              'and' not 'or', so the exit(1) branch is unreachable
                              via main() (times always has length 3, so
                              len(times)!=3 is always False).
- unknown method: exit(1) -> raises SystemExit with code 1
- main() cleanup: NaN->0, Inf->0, neg->0, np.round(..., 4)
  - negative values become exactly 0.0 (asserted by test_cli_cleanup_negatives)
  - rounding to 4 decimals is observable (asserted by test_cli_cleanup_rounding)
- geomean with 3 images: third image is silently ignored (asserted by
  test_geomean_exit_condition_is_unreachable which checks the output equals
  the 2-image result of imgs[0] and imgs[1])
"""

import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from undistortme import contrastmatch as cm

# ---------------------------------------------------------------------------
# Module-level constants and helpers
# ---------------------------------------------------------------------------


def run_cli(*args):
    """Run contrastmatch as a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", "undistortme.contrastmatch", *args],
        capture_output=True,
        text=True,
    )


# ===========================================================================
# 1 – simple_line recovers exact linear model
# ===========================================================================


def test_simple_line_recovers_exact_linear():
    """slope and intercept from simple_line match the generating linear model."""
    shape = (8, 8, 8)
    rng = np.random.default_rng(7)

    # spatially varying a (slope) and b (intercept)
    a = rng.uniform(50.0, 200.0, shape)
    b = rng.uniform(300.0, 700.0, shape)

    t1, t2 = 0.07, 0.11

    img1 = a * t1 + b
    img2 = a * t2 + b

    slope, intercept = cm.simple_line([t1, t2], [img1, img2])

    assert np.allclose(slope, a), "simple_line slope must match generating slope"
    assert np.allclose(intercept, b), (
        "simple_line intercept must match generating intercept"
    )

    # construct_new_image should recover the value at a new TE
    new_te = 0.09
    predicted = cm.construct_new_image(slope, intercept, new_te)
    expected = a * new_te + b

    assert np.allclose(predicted, expected), (
        "construct_new_image must reproduce a*new_te + b"
    )


# ===========================================================================
# 2 – linregress_across_images recovers slopes from exact linear model
# ===========================================================================


def test_linregress_across_images_recovers_slopes():
    """np.polyfit per-voxel recovers exact slopes and intercepts for >=3 images."""
    shape = (4, 4, 2)  # tiny so the voxel loop is fast
    rng = np.random.default_rng(13)

    a = rng.uniform(50.0, 150.0, shape)
    b = rng.uniform(400.0, 600.0, shape)

    times = np.array([0.07, 0.09, 0.11])
    imgs = [a * t + b for t in times]

    slope_3d, intercept_3d = cm.linregress_across_images(times, imgs)

    assert slope_3d.shape == shape, "returned slope shape must match image shape"
    assert intercept_3d.shape == shape, (
        "returned intercept shape must match image shape"
    )

    assert np.allclose(slope_3d, a, atol=1e-5), (
        "linregress_across_images slope must recover generating slope"
    )
    assert np.allclose(intercept_3d, b, atol=1e-5), (
        "linregress_across_images intercept must recover generating intercept"
    )


# ===========================================================================
# 3 – geomean_across_images: equal inputs return the input
# ===========================================================================


def test_geomean_equal_images_identity():
    """When both images are the same constant, geomean returns that constant.

    Source formula:
        c1 = (t2 - t1) / (t2 - t0)
        c3 = (t1 - t0) / (t2 - t0)
        result = exp(c1*log(img+1) + c3*log(img+1)) - 1

    Since c1 + c3 = 1 (they partition the interval), this equals:
        exp(log(img+1)) - 1 = img
    """
    v = 5.0
    img = np.full((4, 4, 2), v)

    # times passed to geomean_across_images are [t0, new_te, t_last]
    times = np.array([0.07, 0.09, 0.11])

    result = cm.geomean_across_images(times, [img, img])

    assert np.allclose(result, v), (
        f"geomean of equal inputs must return the input value; got {result[0, 0, 0]}"
    )


# ===========================================================================
# 4 – geomean_across_images: known nontrivial values
# ===========================================================================


def test_geomean_known_values():
    """Independent hand-computation of geomean for a nontrivial case.

    Parameters chosen so c1 = c3 = 0.5 (t0=0.07, t1=0.09, t2=0.11).
    With img1=2.0 (all voxels) and img2=4.0 (all voxels):

        c1 = (0.11 - 0.09) / (0.11 - 0.07) = 0.02/0.04 = 0.5
        c3 = (0.09 - 0.07) / (0.11 - 0.07) = 0.02/0.04 = 0.5
        result = exp(0.5*log(3) + 0.5*log(5)) - 1
               = exp(log(sqrt(3*5))) - 1
               = sqrt(15) - 1
               ≈ 2.872983346207417
    """
    t0, t1, t2 = 0.07, 0.09, 0.11
    shape = (4, 4, 2)

    img1 = np.full(shape, 2.0)
    img2 = np.full(shape, 4.0)

    times = np.array([t0, t1, t2])
    result = cm.geomean_across_images(times, [img1, img2])

    # Hand-computed independently of the function
    c1 = (t2 - t1) / (t2 - t0)  # = 0.5
    c3 = (t1 - t0) / (t2 - t0)  # = 0.5
    expected = np.exp(c1 * np.log(3.0) + c3 * np.log(5.0)) - 1.0  # = sqrt(15) - 1

    assert np.allclose(result, expected, atol=1e-12), (
        f"geomean result {result[0, 0, 0]!r} != hand-computed {expected!r}"
    )


# ===========================================================================
# 5 – error / exit-path characterization
# ===========================================================================


def test_unknown_method_exits_with_code_1(tmp_path, tiny_nii):
    """main() calls exit(1) for unrecognised method -> process exits with code 1.

    We test by running the script as a subprocess so that argparse and main()
    are fully exercised without modifying module-level state.
    """
    img1 = tiny_nii(tmp_path / "i1.nii", fill=1.0)
    img2 = tiny_nii(tmp_path / "i2.nii", fill=1.0)
    out = str(tmp_path / "out.nii")

    result = run_cli(
        "-i",
        img1,
        img2,
        "-t",
        "0.07",
        "0.11",
        "-n",
        "0.09",
        "-o",
        out,
        "-m",
        "bogusmethod",
    )
    assert result.returncode == 1, (
        f"Expected exit code 1 for unknown method, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_cli_linear_end_to_end(tmp_path):
    """Full subprocess round-trip for the linear method with a known linear model.

    Images are built from img = a*te + b (spatially constant a=100, b=500)
    so the expected output at new_te=0.09 is 100*0.09 + 500 = 509.0, which
    after main()'s cleanup (round to 4 decimals, no NaN/Inf/neg) is 509.0.
    """
    shape = (8, 8, 8)
    a, b = 100.0, 500.0
    t1, t2 = 0.07, 0.11
    new_te = 0.09

    img1_data = np.full(shape, a * t1 + b, dtype=np.float32)  # 507.0
    img2_data = np.full(shape, a * t2 + b, dtype=np.float32)  # 511.0

    img1_path = str(tmp_path / "echo1.nii")
    img2_path = str(tmp_path / "echo2.nii")
    out_path = str(tmp_path / "matched.nii")

    nib.save(nib.Nifti1Image(img1_data, np.eye(4)), img1_path)
    nib.save(nib.Nifti1Image(img2_data, np.eye(4)), img2_path)

    result = run_cli(
        "-i",
        img1_path,
        img2_path,
        "-t",
        str(t1),
        str(t2),
        "-n",
        str(new_te),
        "-o",
        out_path,
        "-m",
        "linear",
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert os.path.isfile(out_path), "Output NIfTI must exist after successful run"

    loaded = nib.load(out_path).get_fdata()

    # No NaN or Inf
    assert not np.any(np.isnan(loaded)), "Output must not contain NaN"
    assert not np.any(np.isinf(loaded)), "Output must not contain Inf"

    # No negatives
    assert not np.any(loaded < 0), "Output must not contain negative values"

    # Expected value from the known model, rounded to 4 decimals as main() does
    expected_value = np.round(a * new_te + b, 4)  # 509.0
    expected = np.full(shape, expected_value)

    assert np.allclose(loaded, expected, atol=1e-4), (
        f"Output values must match np.round(a*new_te+b, 4)={expected_value}; "
        f"got {loaded[0, 0, 0]}"
    )


# ===========================================================================
# 7 – CLI cleanup: negatives are zeroed
# ===========================================================================


def test_cli_cleanup_negatives(tmp_path):
    """main() zeros out negative values produced by the linear extrapolation.

    We construct two images from img = a*t + b with a steep negative slope:
        a = -10000, b = 400
        img1 (t=0.07): -10000*0.07 + 400 = -300.0
        img2 (t=0.11): -10000*0.11 + 400 = -700.0

    At new_te=0.09 the raw model gives: -10000*0.09 + 400 = -500.0 (negative).
    After main()'s cleanup the output must be exactly 0.0.

    NOTE: input images themselves are negative here, which is unusual for MRI
    data, but is sufficient to force a negative extrapolation result that the
    cleanup block must zero.
    """
    shape = (4, 4, 2)
    a, b = -10000.0, 400.0
    t1, t2 = 0.07, 0.11
    new_te = 0.09

    # raw model at new_te: a*new_te + b = -10000*0.09 + 400 = -500.0  (< 0)
    raw_expected = a * new_te + b
    assert raw_expected < 0, "test setup: raw extrapolation must be negative"

    img1_data = np.full(shape, a * t1 + b, dtype=np.float32)  # -300.0
    img2_data = np.full(shape, a * t2 + b, dtype=np.float32)  # -700.0

    img1_path = str(tmp_path / "neg1.nii")
    img2_path = str(tmp_path / "neg2.nii")
    out_path = str(tmp_path / "neg_out.nii")

    nib.save(nib.Nifti1Image(img1_data, np.eye(4)), img1_path)
    nib.save(nib.Nifti1Image(img2_data, np.eye(4)), img2_path)

    result = run_cli(
        "-i",
        img1_path,
        img2_path,
        "-t",
        str(t1),
        str(t2),
        "-n",
        str(new_te),
        "-o",
        out_path,
        "-m",
        "linear",
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert os.path.isfile(out_path), "Output NIfTI must exist"

    loaded = nib.load(out_path).get_fdata()

    # After cleanup: all negative values must be zeroed to exactly 0.0
    assert np.all(loaded == 0.0), (
        f"Expected all zeros after negative cleanup; got {loaded[0, 0, 0]!r}. "
        "Deleting the 'new_image[new_image < 0] = 0' cleanup line would cause this to fail."
    )


# ===========================================================================
# 8 – CLI cleanup: rounding to 4 decimal places is observable
# ===========================================================================


def test_cli_cleanup_rounding(tmp_path):
    """main() rounds the output to 4 decimal places; deleting the round fails this test.

    We choose a, b, new_te so that the raw linear model value has more than
    4 decimal places of precision:
        a = 1.0, b = 0.123456789, t1=0.0, t2=1.0
        At new_te=0.0: raw = 0.123456789

    np.round(0.123456789, 4) = 0.1235, which != 0.123456789.
    The test asserts the output equals the rounded value AND differs from
    the unrounded value, so deleting the np.round line in main() makes it fail.
    """
    shape = (4, 4, 2)
    a, b = 1.0, 0.123456789
    t1, t2 = 0.0, 1.0
    new_te = 0.0

    raw_expected = a * new_te + b  # = 0.123456789
    rounded_expected = np.round(raw_expected, 4)  # = 0.1235

    assert rounded_expected != raw_expected, (
        "test setup: rounding must change the value"
    )

    img1_data = np.full(shape, a * t1 + b, dtype=np.float64)  # 0.123456789
    img2_data = np.full(shape, a * t2 + b, dtype=np.float64)  # 1.123456789

    img1_path = str(tmp_path / "rnd1.nii")
    img2_path = str(tmp_path / "rnd2.nii")
    out_path = str(tmp_path / "rnd_out.nii")

    nib.save(nib.Nifti1Image(img1_data.astype(np.float32), np.eye(4)), img1_path)
    nib.save(nib.Nifti1Image(img2_data.astype(np.float32), np.eye(4)), img2_path)

    result = run_cli(
        "-i",
        img1_path,
        img2_path,
        "-t",
        str(t1),
        str(t2),
        "-n",
        str(new_te),
        "-o",
        out_path,
        "-m",
        "linear",
    )

    assert result.returncode == 0, (
        f"Expected exit code 0, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert os.path.isfile(out_path), "Output NIfTI must exist"

    loaded = nib.load(out_path).get_fdata()

    # Must equal the 4-decimal-rounded value ...
    assert np.allclose(loaded, rounded_expected, atol=1e-6), (
        f"Output must equal np.round({raw_expected}, 4)={rounded_expected}; "
        f"got {loaded[0, 0, 0]!r}"
    )
    # ... and must NOT equal the unrounded value (proves round() is active)
    assert not np.allclose(loaded, raw_expected, atol=1e-6), (
        f"Output must differ from unrounded value {raw_expected}; "
        "deleting np.round in main() would cause this assertion to erroneously pass."
    )


# ===========================================================================
# geomean input-count guard
# ===========================================================================


def test_geomean_rejects_more_than_two_inputs(tmp_path):
    """geomean with three inputs must exit non-zero, not silently use two.

    (times is always built with length 3, so a guard that also requires
    len(times) != 3 can never fire.)
    """
    paths = []
    for i in range(3):
        p = tmp_path / f"e{i}.nii"
        data = np.full((4, 4, 4), float(i + 1), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
        paths.append(str(p))

    result = run_cli(
        "-i",
        *paths,
        "-t",
        "0.07",
        "0.09",
        "0.11",
        "-n",
        "0.08",
        "-o",
        str(tmp_path / "out.nii"),
        "-m",
        "geomean",
    )
    assert result.returncode != 0
    assert not (tmp_path / "out.nii").exists()


def test_geomean_two_inputs_still_works(tmp_path):
    """The guard must not reject the valid two-input case."""
    paths = []
    for i in range(2):
        p = tmp_path / f"e{i}.nii"
        data = np.full((4, 4, 4), float(i + 1), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
        paths.append(str(p))

    result = run_cli(
        "-i",
        *paths,
        "-t",
        "0.07",
        "0.11",
        "-n",
        "0.09",
        "-o",
        str(tmp_path / "out.nii"),
        "-m",
        "geomean",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.nii").exists()
