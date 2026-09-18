#!/usr/bin/env python3
"""Denoise a whole echo's volume series, for FIELD ESTIMATION ONLY.

The pipeline calls this as a subprocess (like ``undistortme-contrastmatch``)
and feeds the results to contrast matching / fslmerge / topup.  ``applytopup``
keeps consuming the ORIGINAL images, so denoising here only stabilises the
estimated field; it never changes the corrected output data.

Two methods, both operating on one echo's full 4-D series (x, y, z, volume):

* ``median`` -- 2-D in-plane 3x3 median, per volume.  No dependencies beyond
                scipy, and no assumption about volume redundancy.
* ``mppca``  -- dipy's Marchenko-Pastur PCA denoiser: a local PCA over small
                spatial patches across the volume series, so each volume keeps
                its own geometry and contrast.

``mppca`` borrows signal across volumes, so it needs a real series (many
volumes); running it on the 2-4 volume merge that feeds topup would be
meaningless.  That is why this runs before slicing, on whole volumes.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

METHODS = ("mppca", "median")

# Which methods need what. Kept here, next to the implementations that impose
# the requirements, so the pipeline has one place to ask rather than its own
# copy of the taxonomy.
DIPY_METHODS = ("mppca",)

# mppca infers its noise model from the spread across volumes, so below this
# many it is not denoising in any meaningful sense -- on a single volume it
# happily returns altered data.  This is a hard error rather than a warning:
# the pipeline runs this as a subprocess and discards its output on success, so
# a warning would never reach anyone.
MIN_VOLUMES = 5

# Below this the Marchenko-Pastur fit is thin; worth saying, not worth refusing.
COMFORTABLE_VOLUMES = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (argv defaults to sys.argv[1:])."""
    parser = argparse.ArgumentParser(
        description=(
            "Denoise one echo's volume series for distortion-field "
            "estimation. Writes one output per input volume."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input images: one 4-D nifti, or the 3-D volumes of one echo "
        "in volume order.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory to write the denoised images into.",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=METHODS,
        required=True,
        help="Denoising method.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Suffix added to each input's basename "
        "(default: desc-denoised-<method>, the name the pipeline predicts).",
    )
    parser.add_argument(
        "--patch_radius",
        type=int,
        default=2,
        help="Patch radius for mppca (default: 2).",
    )
    parser.add_argument(
        "--slice_axis",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Through-plane axis for the median filter: the filter has zero "
        "extent along it. Read from the NIfTI header by default.",
    )
    return parser.parse_args(argv)


def slice_axis_of(img: nib.Nifti1Image) -> int:
    """The acquired slice axis, from the image's own header.

    NOT derived from the anatomical orientation: dcm2niix writes the acquired
    slice direction as the LAST array axis whatever the anatomy, so a coronal
    acquisition still has its slices along axis 2. Verified on this project's
    data -- an axial and a coronal T2 from the same session both report
    ``get_dim_info() == (.., .., 2)`` with the thick (2.5 mm) dimension on axis
    2 -- so mapping anatomy to an array axis would filter across slices on
    exactly the non-axial runs it was meant to protect.
    """
    slice_axis = img.header.get_dim_info()[2]
    return 2 if slice_axis is None else int(slice_axis)


def suffix_for(method: str) -> str:
    """The filename suffix a method's outputs carry.

    One definition, so the pipeline's predicted paths and the CLI's written
    paths cannot drift apart.
    """
    return f"desc-denoised-{method}"


def output_path(nii_path: str, output_dir: str, suffix: str) -> str:
    """Where the denoised version of ``nii_path`` is written.

    The pipeline calls this to predict outputs without running anything, so it
    must stay a pure function of its arguments.
    """
    base = os.path.basename(nii_path)
    for extension in (".nii.gz", ".nii"):
        if base.endswith(extension):
            base = base[: -len(extension)]
            break
    return os.path.join(output_dir, f"{base}_{suffix}.nii")


def denoise_median(data: NDArray, slice_axis: int = 2) -> NDArray:
    """2-D in-plane 3x3 median filter of a 4-D series, per volume.

    The filter has zero extent along ``slice_axis``, so it is the same
    operation as filtering each 2-D slice on its own -- deliberately NOT a 3-D
    median, which would blur across slices (the pipeline corrects distortion
    slice by slice, so through-plane mixing would be self-defeating).
    """
    from scipy.ndimage import median_filter

    size = [3, 3, 3]
    size[slice_axis] = 1
    out = np.empty_like(data, dtype=np.float32)
    for v in range(data.shape[-1]):
        out[..., v] = median_filter(data[..., v].astype(np.float32), size=tuple(size))
    return out


def denoise_mppca(data: NDArray, patch_radius: int = 2) -> NDArray:
    """dipy's Marchenko-Pastur PCA denoiser over the volume series."""
    from dipy.denoise.localpca import mppca

    return mppca(data, patch_radius=patch_radius)


def apply_denoise(
    data: NDArray,
    method: str,
    patch_radius: int = 2,
    slice_axis: int = 2,
) -> NDArray:
    """Denoise a 4-D series with ``method``, validating its requirements."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown denoising method {method!r}; choose one of {', '.join(METHODS)}"
        )

    if method == "median":
        return denoise_median(data, slice_axis=slice_axis)

    n_volumes = data.shape[-1]
    if n_volumes < MIN_VOLUMES:
        raise ValueError(
            f"{method} needs at least {MIN_VOLUMES} volumes to estimate "
            f"anything from, and this series has {n_volumes}. Use "
            "'median' for a series this short: it is a spatial filter and "
            "needs no redundancy across volumes."
        )
    if n_volumes < COMFORTABLE_VOLUMES:
        print(
            f"WARNING: {method} on {n_volumes} volumes is below the {COMFORTABLE_VOLUMES} "
            "it works best with; denoising may be weak."
        )

    if method == "mppca":
        return denoise_mppca(data, patch_radius=patch_radius)
    raise ValueError(f"No implementation for method {method!r}")


def load_series(input_paths: list[str]) -> tuple[NDArray, list[nib.Nifti1Image]]:
    """Load inputs as one 4-D array, plus each input's image for its header.

    One input is taken as a 4-D series; several are stacked in the given order.
    The images are returned so the writer can reuse each volume's own affine
    and header instead of opening every file a second time.
    """
    images = [nib.load(path) for path in input_paths]
    if len(input_paths) == 1:
        data = np.asarray(images[0].dataobj, dtype=np.float32)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        return data, images

    # filled in place rather than stacked from a list, which would hold both
    # the per-volume arrays and the 4-D result at once (2x peak for a series
    # that is already ~100 MB per echo)
    first = np.asarray(images[0].dataobj, dtype=np.float32)
    if first.ndim != 3:
        raise ValueError(
            f"{input_paths[0]} is {first.ndim}-D; give either one 4-D file or "
            "a list of 3-D volumes."
        )
    data = np.empty(first.shape + (len(input_paths),), dtype=np.float32)
    data[..., 0] = first
    for i, (path, img) in enumerate(zip(input_paths[1:], images[1:]), start=1):
        volume = np.asarray(img.dataobj, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(
                f"{path} is {volume.ndim}-D; give either one 4-D file or a "
                "list of 3-D volumes."
            )
        data[..., i] = volume
    return data, images


def save_nifti(data: NDArray, ref: nib.Nifti1Image, out_path: str) -> None:
    """Write one image, via a temporary file so a kill cannot truncate it.

    The pipeline decides an echo is already denoised when every output exists,
    so a half-written file would be silently reused on the next run.
    """
    tmp_path = out_path + ".tmp.nii"
    nib.save(nib.Nifti1Image(data, ref.affine, ref.header), tmp_path)
    os.replace(tmp_path, out_path)


def save_series(
    data: NDArray,
    input_paths: list[str],
    output_dir: str,
    suffix: str,
    images: list[nib.Nifti1Image],
) -> list[str]:
    """Write the denoised data back out in the shape it arrived in.

    Each output keeps its own input's affine and header, using the images
    ``load_series`` already opened.
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(input_paths) == 1:
        out_path = output_path(input_paths[0], output_dir, suffix)
        out_data = data if data.shape[-1] > 1 else data[..., 0]
        save_nifti(out_data, images[0], out_path)
        return [out_path]

    written = []
    for i, (path, img) in enumerate(zip(input_paths, images)):
        out_path = output_path(path, output_dir, suffix)
        save_nifti(data[..., i], img, out_path)
        written.append(out_path)
    return written


def main() -> None:
    """Denoise the given series and write one output per input."""
    args = parse_args()

    for path in args.input:
        if not os.path.isfile(path):
            print(f"ERROR: input image {path} does not exist")
            sys.exit(1)

    suffix = args.suffix if args.suffix is not None else suffix_for(args.method)
    data, images = load_series(args.input)
    slice_axis = (
        args.slice_axis if args.slice_axis is not None else slice_axis_of(images[0])
    )
    denoised = apply_denoise(
        data,
        args.method,
        patch_radius=args.patch_radius,
        slice_axis=slice_axis,
    )
    written = save_series(denoised, args.input, args.output_dir, suffix, images)
    print(f"Wrote {len(written)} denoised image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
