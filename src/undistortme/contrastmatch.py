#!/usr/bin/env python3
import argparse
import os

import nibabel as nib
import numpy as np

# from concurrent.futures import ProcessPoolExecutor, as_completed


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contrast-match two echoes of a multi-echo dataset to a new TE by fitting a function to them and interpolating the new values at each voxel of the image"
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input images to contrast-match",
    )
    parser.add_argument(
        "-t",
        "--te",
        nargs="+",
        required=True,
        help="Echo times of input images",
    )
    parser.add_argument(
        "-n",
        "--new",
        type=float,
        required=True,
        help="New TE to contrast-match to",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output image name",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=False,
        default="linear",
        help="Method to use for contrast matching. Options are 'linear' or 'geomean'. Default is 'linear'.",
    )
    return parser.parse_args()


def check_args(args):
    if len(args.input) != len(args.te):
        raise ValueError("Number of input images and TEs must be equal")
    for i in range(len(args.input)):
        print(args.input[i])
        if not os.path.isfile(args.input[i]):
            raise ValueError("Input image {} does not exist".format(args.input[i]))
    if os.path.isfile(args.output):
        raise ValueError("Output image {} already exists".format(args.output))


def load_image(nii_path: str) -> np.ndarray:
    nii = nib.load(nii_path)
    return nii.get_fdata()


def linregress_across_images(times, images: list[np.ndarray]) -> tuple:
    # Number of voxels per image and number of images
    num_voxels: np.int64 = np.prod(images[0].shape)
    num_images: int = len(images)
    times = np.array(times).astype(np.float64)

    # Reshape images for easier processing
    reshaped_images: list[np.ndarray] = [img.reshape(-1) for img in images]
    print(reshaped_images[0].shape)

    # Combine all images into a single array
    combined_data = np.vstack(reshaped_images)
    print(combined_data.shape)

    # Prepare arrays to hold regression results
    slopes = np.empty(num_voxels).astype(np.float64)
    intercepts = np.empty(num_voxels).astype(np.float64)

    # Perform linear regression for each voxel
    for i in range(num_voxels):
        # print(f"Voxel {i+1} of {num_voxels}")
        # print(combined_data[:, i])
        # print(times)
        slopes[i], intercepts[i] = np.polyfit(times, combined_data[:, i], 1)
        # try:
        #     # slopes[i], intercepts[i], _, _, _ = linregress(times, combined_data[:, i])
        #     slopes[i], intercepts[i] = np.polyfit(times, combined_data[:, i], 1)
        # except:
        #     print("ValueError")
        #     slopes[i] = np.nan
        #     intercepts[i] = np.nan
        #     continue

    # Reshape the regression results back to the 3D shape of the original images
    slope_3d = slopes.reshape(images[0].shape)
    intercept_3d = intercepts.reshape(images[0].shape)

    return slope_3d, intercept_3d


def simple_line(times, imgs):
    sig_diff = imgs[1] - imgs[0]
    te_diff = times[1] - times[0]
    slope_3d = sig_diff / te_diff
    intercept_3d = imgs[0] - slope_3d * times[0]
    return slope_3d, intercept_3d


def geomean_across_images(times: np.ndarray, imgs: list[np.ndarray]) -> np.ndarray:
    # Method introduced in Weiskopf et al. 2005
    # https://doi.org/10.1016/j.neuroimage.2004.12.012
    c1 = (times[2] - times[1]) / (times[2] - times[0])
    c3 = (times[1] - times[0]) / (times[2] - times[0])
    img1_offset = imgs[0] + 1
    img3_offset = imgs[1] + 1
    return np.exp(c1 * np.log(img1_offset) + c3 * np.log(img3_offset)) - 1


def construct_new_image(
    slope_3d: np.ndarray, intercept_3d: np.ndarray, new_te: float
) -> np.ndarray:
    new_image = slope_3d * new_te + intercept_3d
    return new_image


def main() -> None:
    args = get_args()
    check_args(args)
    new_te = args.new
    input_images = args.input
    input_tes = args.te
    output_image = args.output

    if args.method == "linear":
        input_tes = np.array(input_tes).astype(np.float64)

        imgs: list[np.ndarray] = []
        for img in input_images:
            imgs.append(load_image(img))

        if len(imgs) == 2:
            slopes, intercepts = simple_line(input_tes, imgs)
        else:
            slopes, intercepts = linregress_across_images(input_tes, imgs)

        new_image = construct_new_image(slopes, intercepts, new_te)
    elif args.method == "geomean":
        imgs: list[np.ndarray] = []
        for img in input_images:
            imgs.append(load_image(img))
        times = np.array([input_tes[0], new_te, input_tes[-1]]).astype(np.float64)
        if len(imgs) != 2:
            print("Not implemented for more than two images yet")
            exit(1)
        new_image = geomean_across_images(times, imgs)
    else:
        print("Method not recognized")
        exit(1)

    # clean up the output image for nans, inf, negatives, etc.
    new_image[np.isnan(new_image)] = 0
    new_image[np.isinf(new_image)] = 0
    new_image[new_image < 0] = 0
    new_image = np.round(new_image, 4)

    ref_nii = nib.load(input_images[0])
    new_nii = nib.Nifti1Image(new_image, ref_nii.affine, ref_nii.header)
    nib.save(new_nii, output_image)
    pass


if __name__ == "__main__":
    main()
