#!/usr/bin/env python
# coding: utf-8
# Liam Timms 2024
# Corrects distortion in multiecho EPI data using FSL's TOPUP

import argparse
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import resources
from pathlib import Path
from random import shuffle

import nibabel as nib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim

# from nilearn.image import crop_img
from tqdm import tqdm

from undistortme import denoise
from undistortme import bids as bids_input

# Contrast matching runs as a subprocess of the same interpreter, so the
# command works regardless of how/where the package is installed.
CONTRASTMATCH_CMD = f"{sys.executable} -m undistortme.contrastmatch"

# Denoising runs the same way, and for the same reason.
DENOISE_CMD = f"{sys.executable} -m undistortme.denoise"

# take advantage of newer pandas features
pd.options.mode.copy_on_write = True

# TOPUP threads rarely saturate a core (~25% utilization is typical), so by
# default each concurrent TOPUP is offered more threads than a strict
# cores/workers split would allow. Tune per machine with --oversubscribe.
DEFAULT_OVERSUBSCRIBE = 4.0


def default_jobs() -> int:
    """Number of usable cores (cgroup/affinity aware where supported)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def topup_threads(batch_size: int) -> int:
    """Thread count for each topup in a batch of ``batch_size`` commands."""
    jobs = check_dict.get("jobs") or default_jobs()
    oversub = check_dict.get("oversubscribe") or DEFAULT_OVERSUBSCRIBE
    concurrent = max(1, min(jobs, batch_size))
    return int(max(1, min(jobs, round(jobs * oversub / concurrent))))


def default_config(slice_mode: bool) -> str:
    """Path to the bundled TOPUP config for the given correction mode.

    Whole-volume fits use ``pervol.cnf``: FSL's default b02b0.cnf schedule
    with movement estimation disabled (the opposite-blip echoes come from a
    single acquisition, so estimating motion between them only invents
    spurious rigid transforms that applytopup then applies).  Slice-wise
    fits use ``perslice_1.cnf``, which additionally drops all subsampling
    so single-slice inputs are accepted.
    """
    name = "perslice_1.cnf" if slice_mode else "pervol.cnf"
    return str(resources.files("undistortme") / "configs" / name)


def resolve_config(config_file: str | None, slice_check: bool) -> str:
    """The TOPUP config to use: an explicit -c, else the bundled default for
    the FINAL correction mode (call after check_binaries, which can turn
    slice mode off when slicenii is missing)."""
    if config_file is not None:
        return config_file
    return default_config(slice_check)


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Defaults
    input_dir = "./sourcedata"
    output_dir = "./"
    deriv_dir = "./derivatives"

    # Create the parser and add arguments
    parser = argparse.ArgumentParser(
        description="Corrects for distortion in multiecho EPI data."
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        help="directory of DICOMs to convert; only used with -d "
        "(default: ./sourcedata)",
        default=input_dir,
    )
    parser.add_argument(
        "-u",
        "--subject_dir",
        help="Process a single subject LABEL (e.g. sub-01) under "
        "--output_dir instead of globbing sub-*",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="output directory for nifti files (default: ./)",
        default=output_dir,
    )
    parser.add_argument(
        "--derivdir",
        help="directory for derivatives (default: ./derivatives)",
        default=deriv_dir,
    )
    parser.add_argument(
        "--workdir",
        help=(
            "directory for intermediate files, safe to delete after a "
            "run (default: {derivdir}/undistortme-work)"
        ),
        default=None,
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="config file for topup (default: the bundled pervol.cnf, or "
        "perslice_1.cnf when -s/--slice is given)",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--run_topup",
        help="Whether to run FSL TOPUP correction.",
        action="store_true",
    )
    # -d converts DICOMs into the input tree; --bids replaces the input
    # tree with a staging of an already-converted dataset - never both.
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "-d",
        "--run_dcm2niix",
        help="Run dcm2niix conversion.",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--slice",
        help="Run the correction slice-by-slice.",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--matchcontrast",
        help="TE-weighted geometric-mean combination of echoes 1 & 3 to "
        "match echo 2's contrast. Requires >= 3 echoes; runs with fewer "
        "are skipped entirely.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--dryrun",
        help="Whether to run the script without actually running any commands.",
        action="store_true",
    )
    parser.add_argument(
        "--twoecho",
        help="Use only the first two echoes of multi-echo data (no effect "
        "on 2-echo runs; ignored when --matchcontrast is given)",
        action="store_true",
    )
    parser.add_argument(
        "--denoise",
        choices=list(denoise.METHODS),
        default=None,
        help="Denoise the images used to ESTIMATE the field (the correction "
        "is still applied to the original images). 'mppca' needs dipy. The "
        "method becomes part of the output variant name. (default: no "
        "denoising)",
    )
    parser.add_argument(
        "--maskdir",
        help="Directory containing masks for each run.",
        default=None,
    )
    parser.add_argument(
        "--fix-names",
        dest="fix_names",
        help=(
            "Strip '.' characters from subject file/directory names "
            "(only triggered when a sub-* name contains a dot). "
            "WARNING: modifies the input tree in place."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1000,
        help="b-value at or below which TOPUP is fit (b <= CUTOFF is "
        "low-b); higher-b volumes reuse the field of the most similar "
        "low-b volume. Only used for runs with more than one distinct "
        "b-value. (default: 1000)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum parallel worker processes (default: all available cores).",
    )
    parser.add_argument(
        "--oversubscribe",
        type=float,
        default=None,
        help=(
            "TOPUP thread oversubscription factor (default: "
            f"{DEFAULT_OVERSUBSCRIBE}). TOPUP threads rarely saturate a "
            "core, so each concurrent TOPUP is given about "
            "jobs*FACTOR/concurrent threads. Use 1 for strict budgeting "
            "on shared machines."
        ),
    )
    input_mode.add_argument(
        "--bids",
        metavar="BIDS_DIR",
        default=None,
        help=(
            "Read input from a BIDS dataset instead of the legacy run-"
            "directory layout: 4D sub-*/[ses-*/]dwi/*_dwi.nii[.gz] images "
            "are staged as per-volume 3D .nii under "
            f"{{workdir}}/{bids_input.DEFAULT_STAGING_SUBDIR}/ and that "
            "staging tree becomes the input root (-o/--output_dir is "
            "then not used for input). Staging always writes files, even "
            "under --dryrun, so the printed commands reference real "
            "inputs. Mutually exclusive with -d/--run_dcm2niix."
        ),
    )
    bids_input.add_echo_source_args(parser)
    # parser.add_argument(
    # "--t2",
    # help="whether to fit t2",
    # action="store_true",
    # )

    # Parse the arguments
    args = parser.parse_args()
    return args


def rename_entities(root_dir):
    """Recursively rename files and directories in a directory tree."""
    # A list to keep track of directories to rename
    dirs_to_rename = []
    dir = Path(root_dir)
    name = dir.stem
    new_name = dir.parent / name.replace(".", "")
    os.rename(root_dir, new_name)

    for path, subdirs, files in os.walk(new_name, topdown=False):
        for name in files:
            old_file_path = Path(path) / name
            new_file_path = rename_entity(old_file_path)
            if new_file_path:
                print(f"Renamed file '{old_file_path}' to '{new_file_path}'")

        # Add directories to the list (renaming will be handled after this loop)
        dirs_to_rename.extend([Path(path) / subdir for subdir in subdirs])

    # Rename directories after all files have been processed
    for dir_path in dirs_to_rename:
        new_dir_path = rename_entity(dir_path)
        if new_dir_path:
            print(f"Renamed directory '{dir_path}' to '{new_dir_path}'")

    return Path(new_name).stem


def rename_entity(old_path):
    """Rename a file or directory by removing dots from its name."""
    # Split the path and extension (if any)
    parent = old_path.parent
    name = old_path.stem
    extension = old_path.suffixes[-1] if old_path.suffixes else ""

    # Replace dots in the name part only
    new_name = name.replace(".", "")

    # Construct the new path
    new_path = Path(parent) / f"{new_name}{extension}"

    # Rename the entity
    if new_path != old_path:
        os.rename(old_path, new_path)
        return new_path
    return None


def check_binaries(
    dcm2niix_check: bool, topup_check: bool, slice_check: bool
) -> tuple[bool, bool, bool, bool]:
    """Check for required binaries & turn off the relevant flags if not found."""
    convert_binaries = ["dcm2niix"]
    for binary in convert_binaries:
        if shutil.which(binary) is None:
            print(f"{binary} not found, TURNING OFF DCM2NIIX")
            dcm2niix_check = False

    fsl_binaries = ["fslmerge", "fslmaths", "topup", "applytopup"]
    for binary in fsl_binaries:
        if shutil.which(binary) is None and topup_check:
            print(f"{binary} not found, TURNING OFF TOPUP")
            topup_check = False

    topup_multithread = True
    if topup_check and slice_check:
        print("setting topup calls to single thread because slice-by-slice is on")
        topup_multithread = False
    elif topup_multithread and topup_check and not slice_check:
        # check if fsl's topup actually supports multithreading using it's help message
        topup_help = subprocess.run(
            "topup --help", capture_output=True, text=True, shell=True
        )
        if "nthr" not in topup_help.stderr:
            print(
                "WARNING: this version of fsl's topup might not support multithreading"
                + "SWITCHING TO SINGLE THREAD TOPUP."
            )
            topup_multithread = False

    slicenii_binaries = ["slicenii", "combinenii"]
    for binary in slicenii_binaries:
        if shutil.which(binary) is None and slice_check == 1:
            print(f"{binary} not found, TURNING OFF SLICE BY SLICE")
            slice_check = False

    return (dcm2niix_check, topup_check, topup_multithread, slice_check)


def _dipy_available() -> bool:
    """Whether dipy can be imported (it is an optional dependency)."""
    return importlib.util.find_spec("dipy") is not None


def check_denoise(method: str | None) -> str | None:
    """Return the requested denoising method, or refuse to run without it.

    Deliberately NOT check_binaries' warn-and-disable: a disabled binary
    downgrades to a correctly-labelled plain run, whereas silently skipping
    denoising would leave un-denoised results that no output name distinguishes
    from denoised ones.
    """
    if method is None:
        return None
    if method in denoise.DIPY_METHODS and not _dipy_available():
        print(
            f"ERROR: --denoise {method} needs dipy, which is not installed "
            "(pip install 'undistortme[denoise]').\n"
            "Refusing to continue: carrying on would write UN-DENOISED "
            "results under a plain variant name and exit 0, which is "
            "indistinguishable from a run that was never asked to denoise."
        )
        exit(1)
    return method


def get_denoise_command(
    nii_list: list[str],
    out_dir: str,
    method: str,
) -> tuple[str | None, list[str]]:
    """Build the command denoising one echo's volume series.

    The denoised images are for FIELD ESTIMATION ONLY; ``nii_list`` itself is
    what applytopup keeps using.  ``nii_list`` must be in volume order: the
    outputs are mapped back to the inputs by position.

    The median filter's through-plane axis is not passed: undistortme-denoise
    reads it from each image's own header, which is the only place that knows
    it (see denoise.slice_axis_of).
    """
    suffix = denoise.suffix_for(method)
    out_list = [denoise.output_path(nii, out_dir, suffix) for nii in nii_list]
    if all(os.path.exists(out) for out in out_list):
        return (None, out_list)

    command = (
        f"{DENOISE_CMD} -m {method}"
        + f" -i {' '.join(nii_list)}"
        + f" -o {out_dir} --suffix {suffix}"
    )
    return (command, out_list)


def handle_denoising(run_df: pd.DataFrame, dir_dict: dict) -> pd.DataFrame | None:
    """Add an ``est_nii`` column of images to estimate the field from.

    One denoising call per echo, over that echo's whole volume series (mppca
    borrows signal across volumes, so it needs the series, not the 2-4 volume
    merge that feeds topup).  ``nii`` is left untouched: it is what applytopup
    corrects.

    Returns ``None`` when the run cannot be denoised as asked, after reporting
    the skip and recording it in ``failed_commands``; the caller then skips
    the run. Results written into a ``_denoise-{method}`` tree without
    denoising would be indistinguishable from real ones, and the
    resume-by-existence logic would keep them for good.
    """
    method = check_dict.get("denoise")
    if not method:
        run_df["est_nii"] = run_df["nii"]
        return run_df

    denoise_cmd_list: list[str] = []
    est_map: dict[str, str] = {}
    for echo, echo_df in run_df.groupby("echo", sort=True):
        echo_df = echo_df.sort_values("bvol_num")
        nii_list = echo_df["nii"].tolist()
        if method in denoise.DIPY_METHODS and len(nii_list) < denoise.MIN_VOLUMES:
            return skip_run(
                run_df,
                f"echo {echo} has {len(nii_list)} volumes and {method} needs "
                f"at least {denoise.MIN_VOLUMES}",
            )
        if (
            method in denoise.DIPY_METHODS
            and len(nii_list) < denoise.COMFORTABLE_VOLUMES
        ):
            # the same warning inside undistortme-denoise would be swallowed:
            # run_bash_command discards subprocess output on success
            print(
                f"WARNING: echo {echo} has {len(nii_list)} volumes, below "
                f"the {denoise.COMFORTABLE_VOLUMES} {method} works best "
                "with; denoising may be weak."
            )
        (command, out_list) = get_denoise_command(
            nii_list, dir_dict["work_dir"], method
        )
        denoise_cmd_list.append(command)
        est_map.update(dict(zip(nii_list, out_list)))

    failed = parallel_bash_commands(denoise_cmd_list, "denoising commands")
    if failed:
        # the individual command failures are already recorded by the
        # dispatcher; this entry names the run that is skipped because of them
        return skip_run(
            run_df,
            f"{len(failed)} {method} command(s) failed, so the field would be "
            "estimated from un-denoised images",
        )

    run_df["est_nii"] = run_df["nii"].map(est_map)
    return run_df


def estimation_df(df: pd.DataFrame) -> pd.DataFrame:
    """View of ``df`` whose ``nii`` column is the field-estimation image.

    Lets the field-estimation builders (contrast match, merge, the diffusion
    path's volume matching) stay unaware of denoising: they read ``nii`` as
    before, but on this view it is ``est_nii``.  ``est_nii`` is always present
    -- handle_denoising sets it on every path, before any consumer runs.
    """
    return df.assign(nii=df["est_nii"])


def convert_dcms(input_dir: str, output_dir: str) -> str:
    """Convert dicoms to niftis using dcm2niix."""
    os.makedirs(output_dir, exist_ok=True)
    command = (
        # "dcm2niix" + " -d 9" + " -z 3" + " -w 0" + " -v 1" + " -m 1" + # this line is for when HASTE scans are failing
        "dcm2niix"
        + " -d 9"
        + " -z 3"
        + " -w 0"
        + " -v 1"
        + " -f sub-%i/ses-%t/run-%s_desc-%d/sub-%i_ses-%t_run-%s_desc-%d_echo-%e"
        + f" -o {output_dir} "
        + input_dir
    )
    return command


def find_files(directory: str, pattern: str) -> list[str]:
    """Find files in a directory matching a pattern, in sorted order.

    glob order is filesystem-dependent; sorting keeps processing order (and
    the command batches built from it) deterministic across machines.
    """
    return sorted(glob.glob(f"{directory}/{pattern}", recursive=True))


def get_fslmerge_command(
    subject: str,
    session: str,
    run: str,
    bvol_num: int,
    slice_num: int,
    topup_dir: str,
    topup_file_list: list[str],
    suffix: str,
) -> tuple[str | None, str]:
    """generate fslmerge command"""
    merged_filename = os.path.join(
        topup_dir,
        f"{subject}_{session}_{run}_bv-{bvol_num}_sv-{slice_num}_merged_{suffix}.nii",
    )
    # print(topup_file_list)
    if os.path.exists(merged_filename):
        command = None
    else:
        command = f"fslmerge -t {merged_filename} {' '.join(topup_file_list)}"
    return (command, merged_filename)


def linear_contrast_match(
    subject: str,
    session: str,
    run: str,
    bvol_num: int,
    slice_num: int,
    slice_df: pd.DataFrame,
    ave_dir: str,
) -> tuple[str | None, list[str]]:
    """Use the time-to-echo values to fit a linear model to the contrast."""
    echo1_nii = slice_df[slice_df["echo"] == 1]["nii"].values[0]
    echo2_nii = slice_df[slice_df["echo"] == 2]["nii"].values[0]
    echo3_nii = slice_df[slice_df["echo"] == 3]["nii"].values[0]
    te1 = slice_df[slice_df["echo"] == 1]["echo_time"].values[0]
    te2 = slice_df[slice_df["echo"] == 2]["echo_time"].values[0]
    te3 = slice_df[slice_df["echo"] == 3]["echo_time"].values[0]
    cm_file = os.path.join(
        ave_dir,
        f"{subject}_{session}_{run}_bv-{bvol_num}_sv-{slice_num}_desc-cm_linear_echoes-1-3.nii",
    )
    if os.path.exists(cm_file):
        command = None
    else:
        command = f"{CONTRASTMATCH_CMD} -i {echo1_nii} {echo3_nii} -t {te1} {te3} -n {te2} -o {cm_file}"

    return (command, [cm_file, echo2_nii])


# def get_filt_command(
#     subject: str,
#     session: str,
#     run: str,
#     bvol_num: int,
#     slice_num: int,
#     dir: str,
#     merged_filename: str,
# ):
#     filt_filename = merged_filename.replace(".nii", "_filt.nii")
#     if os.path.exists(filt_filename):
#         command = None
#     else:
#         command = f"fslmaths {merged_filename} -kernel box 1 -fmedian {filt_filename}"
#     return (command, filt_filename)


def geomean_contrast_match(
    subject: str,
    session: str,
    run: str,
    bvol_num: int,
    slice_num: int,
    slice_df: pd.DataFrame,
    ave_dir: str,
) -> tuple[str | None, list[str]]:
    """Use the time-to-echo values in a weighted geometric mean to mimic contrast."""
    echo1_nii = slice_df[slice_df["echo"] == 1]["nii"].values[0]
    echo2_nii = slice_df[slice_df["echo"] == 2]["nii"].values[0]
    echo3_nii = slice_df[slice_df["echo"] == 3]["nii"].values[0]
    te1 = slice_df[slice_df["echo"] == 1]["echo_time"].values[0]
    te2 = slice_df[slice_df["echo"] == 2]["echo_time"].values[0]
    te3 = slice_df[slice_df["echo"] == 3]["echo_time"].values[0]
    cm_file = os.path.join(
        ave_dir,
        f"{subject}_{session}_{run}_bv-{bvol_num}_sv-{slice_num}_desc-cm_geomean_echoes-1-3.nii",
    )
    if os.path.exists(cm_file):
        command = None
    else:
        command = (
            f"{CONTRASTMATCH_CMD} -i {echo1_nii} {echo3_nii}"
            + f" -t {te1} {te3} -n {te2} -o {cm_file} -m geomean"
        )

    return (command, [cm_file, echo2_nii])


def default_work_root(deriv_dir: str) -> str:
    """Default intermediates root when --workdir is not given."""
    return os.path.join(deriv_dir, "undistortme-work")


def set_dirs(
    subject: str,
    session: str,
    run: str,
    output_dir: str,
    deriv_dir: str,
    work_root: str | None = None,
) -> dict[str, str]:
    """Generate the directory structure for a given subject and run.

    Final corrected outputs live under ``{deriv_dir}/undistortme/``; every
    intermediate (merged echoes, contrast-matched volumes, masked copies,
    slices, acqparams, raw TOPUP results) lives in one work tree (default
    ``{deriv_dir}/undistortme-work/``) that can be deleted after a run.
    """

    current_dir = os.path.join(output_dir, subject, session, run)
    if not os.path.exists(current_dir):
        print(f"{current_dir} does not exist, exiting")
        exit(1)

    if work_root is None:
        work_root = default_work_root(deriv_dir)

    dir_dict = {"current_dir": current_dir}
    if check_dict["slice"]:
        inner_dir = "per-slice"
    else:
        inner_dir = "whole-volume"

    if check_dict["match"]:
        inner_dir = inner_dir + "_contrast-matched"

    if check_dict["mask"]:
        inner_dir = inner_dir + "_masked"

    if check_dict.get("denoise"):
        # Intermediates are reused when they already exist, so a denoised run
        # must not share a work tree with a plain one: without this it would
        # silently resume onto the other condition's merged file and field.
        inner_dir = inner_dir + f"_denoise-{check_dict['denoise']}"

    work_dir = os.path.join(work_root, inner_dir, subject, session, run)
    if check_dict["slice"]:
        dir_dict["slice_dir"] = work_dir
    if check_dict["mask"]:
        dir_dict["masked_dir"] = work_dir
    dir_dict["ave_dir"] = work_dir
    dir_dict["work_dir"] = work_dir
    dir_dict["work_root"] = work_root

    dir_dict["topup_dir"] = os.path.join(
        deriv_dir,
        "undistortme",
        inner_dir,
        subject,
        session,
        run,
    )

    dir_dict["inner_dir"] = inner_dir
    return dir_dict


def check_for_bvals(dir_dict: dict[str, str]) -> tuple[int, str, list[str]]:
    """Check if bvals exist, return number of bvals and contents of the bval file."""
    # check if bvals exist (if they do, we assume bvecs do too)
    bval_files = find_files(dir_dict["current_dir"], "*.bval")
    # read the contents of the first bval file
    if len(bval_files) > 0:
        with open(bval_files[0], "r") as bval_file:
            bvals = bval_file.read()
        num_bvals = len(bvals.split())
    else:
        # if there's no bvals files
        num_bvals = 1
        bvals = ""

    return (num_bvals, bvals, bval_files)


def make_acq_params(
    subject, session, run, run_df: pd.DataFrame, topup_dir: str
) -> Path:
    """make an acqparams.txt file for run, considering contrast matching"""
    run_df = run_df.sort_values("echo")
    acq_file = Path(os.path.join(topup_dir, f"{subject}_{session}_{run}_acqparams.txt"))
    if os.path.exists(acq_file):
        print(f"{acq_file} exists.")
        return acq_file
    print(f"Making {acq_file}")
    # we want to write a line for each echo in the normal case
    # but in contrast matched case, we want to write a line for 1,3 combined and then 2
    if check_dict["match"] and run_df["echo"].nunique() > 2:
        echo_list: list[int] = [1, 2]
    else:
        echo_list = run_df["echo"].unique().tolist()

    acqparams_lines = []
    for echo in sorted(echo_list):
        echo_df = run_df[run_df["echo"] == echo]
        phase_sign = bids_input.expected_phase_sign(echo)
        phase_dir = echo_df["phase_dir"].unique()[0]
        readout_time = echo_df["readout_time"].unique()[0]
        if "j" in phase_dir:
            acqparams_lines.append(f"0 {phase_sign} 0 {readout_time}\n")
        elif "i" in phase_dir:
            acqparams_lines.append(f"{phase_sign} 0 0 {readout_time}\n")
        elif "k" in phase_dir:
            acqparams_lines.append(f"0 0 {phase_sign} {readout_time}\n")
        else:
            print("Phase encoding direction not recognized")
            exit(1)
    acq_file.write_text("".join(acqparams_lines))
    return acq_file


def get_echo_info(json_path: str) -> dict:
    """Get the echo info for a given json file."""
    with open(json_path, "r") as f:
        json_data = json.load(f)

    echo_num = json_data.get("EchoNumber", 1)
    # look for a bvals file in the same directory (swap only the extension:
    # a bare substring replace would corrupt paths containing "json" in a
    # directory or file name)
    bval_path = os.path.splitext(json_path)[0] + ".bval"
    if os.path.exists(bval_path):
        with open(bval_path, "r") as f:
            bvals = f.read().split()
        num_bvals = len(bvals)
        print(f"Found {num_bvals} bvals for {json_path}")
    else:
        bvals = [None]
        num_bvals = 0

    if num_bvals == 0:
        # we handle averages as though they are bvols
        nii_pattern = os.path.basename(json_path)
        nii_pattern = nii_pattern.replace(".json", "_*.nii")
        found_niis = find_files(os.path.dirname(json_path), nii_pattern)
        if found_niis is not None:
            print(f"Found {len(found_niis)} nifti files for {json_path} but no bvals.")
            num_bvals = len(found_niis)
        else:
            print(
                f"WARNING: niis not found in {os.path.dirname(json_path)}"
                + f"using pattern {nii_pattern}"
            )

    nii_file_list = []
    for i in range(1, num_bvals + 1):
        # per-volume naming shared with the staging adapter: {stem}_{bb}.nii
        nii_path = bids_input.volume_filename(
            json_path.removesuffix(".json"), i, num_bvals
        )

        if os.path.exists(nii_path):
            nii_file_list.append(nii_path)
        else:
            # try globbing for one with a close name
            print(f"WARNING: {nii_path} does not exist, globbing")
            bb = bids_input.volume_index_token(i, num_bvals)
            nii_pattern = os.path.dirname(json_path)
            nii_pattern = os.path.join(nii_pattern, f"*echo-{echo_num}_{bb}.nii")
            # nii_file_list = find_files(os.path.dirname(json_path), nii_pattern)
            possible_files = glob.glob(nii_pattern)
            if possible_files is None or len(possible_files) == 0:
                print(f"WARNING: no nifti files found for {json_path}")
            elif len(possible_files) == 1:
                nii_path = possible_files[0]
                nii_file_list.append(nii_path)
            else:
                print(
                    f"Found {possible_files} nifti files for {json_path} with pattern {nii_pattern}"
                )
                raise FileNotFoundError(
                    "multiple candidate niftis found when there should be one"
                )

    echo_time = json_data.get("EchoTime", "undefined")
    phase_encoding_direction = json_data.get("PhaseEncodingDirection", "undefined")
    orientation_vals = json_data.get("ImageOrientationPatientDICOM", "undefined")
    pulse_sequence = json_data.get("PulseSequenceName", "undefined")
    series_description = json_data.get("SeriesDescription", "undefined")
    image_type = json_data.get("ImageType", "undefined")
    readout_time = json_data.get("TotalReadoutTime", "undefined")
    phase_sign = bids_input.expected_phase_sign(echo_num)

    # now guess orientation by rounding the values in orienation_vals
    if orientation_vals == "undefined":
        orientation_rounded = [0, 0, 0, 0, 0, 0]
        slice_orientation = "unknown"
        orientation_num = 3
    else:
        orientation_rounded = [round(x, 0) for x in orientation_vals]
        if orientation_rounded == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
            slice_orientation = "axial"
            orientation_num = 2
        elif orientation_rounded == [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]:
            slice_orientation = "coronal"
            orientation_num = 1
        elif orientation_rounded == [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]:
            slice_orientation = "sagittal"
            orientation_num = 0
        else:
            slice_orientation = "unknown"
            orientation_num = 3

    # alternative idea take the difference from each list
    # then pick the direction that fits the best (removing a fail state)
    # but this is ok for pciking the orientation as a human might expect it to be
    # i.e. NOT axial, coronal, or sagittal if it's very twisted

    return {
        "json_path": json_path,
        "echo": echo_num,
        "bvals": num_bvals,
        "bvals_list": bvals,
        "nii_list": nii_file_list,
        "phase_dir": phase_encoding_direction,
        "phase_sign": phase_sign,
        "echo_time": echo_time,
        "readout_time": readout_time,
        "pulse_sequence": pulse_sequence,
        "series_description": series_description,
        "image_type": image_type,
        "orientation_list": orientation_vals,
        "orientation_rounded": orientation_rounded,
        "slice_orientation": slice_orientation,
        "orientation_num": orientation_num,
    }


def run_topup(
    subject: str,
    session: str,
    run: str,
    config: str,
    dir_dict: dict,
    curr_df: pd.DataFrame,
) -> None:
    """Run topup on a given subject and run."""
    suffix = dir_dict["inner_dir"]
    # make an acqparams.txt file for topup
    acq_file = make_acq_params(subject, session, run, curr_df, dir_dict["work_dir"])
    if acq_file is None:
        print("Could not make or find acqparams.txt file, skipping")
        return

    # setup topup commands
    # we want to iterate over each bvol_num and each slice_num
    # for each we want to make a long command that will:
    # 1. merge the right volumes
    # 2. run topup on the merged volumes
    # 3. apply the topup correction to the original volumes
    # --------------------
    # we need to take into account contrast matching setting in choosing the volumes
    # if it's on we also need to average 1,3 and collect that result
    # we also need to take into account the slice setting to decide if we merge
    # --------------------
    # so what we want to do is figure out volume lists for each bvol_num and slice_num
    # then we will refine that based on contrast matching
    # then we will make the commands

    bvol_num_list = curr_df["bvol_num"].unique().tolist()
    shuffle(bvol_num_list)
    topup_cmd_list = []
    n_bvols = len(bvol_num_list)

    for bvol_num in bvol_num_list:
        bvol_df = curr_df[curr_df["bvol_num"] == bvol_num]
        for slice_num in bvol_df["slice_num"].unique():
            # slice_num = int(slice_num)
            # we set the full volumes to be labled as "slice 1" so this works for both
            slice_df = bvol_df[bvol_df["slice_num"] == slice_num].sort_values("echo")
            original_file_list = slice_df["nii"].tolist()
            print(original_file_list)
            echo_list = slice_df["echo"].tolist()
            if original_file_list is None or echo_list is None:
                print(
                    f"Missing files or info for {subject} {session} {run}"
                    + f"bvol {bvol_num} slice {slice_num}"
                )
                exit(1)

            # everything from here to the topup call estimates the field, so
            # it works on the (possibly denoised) estimation images
            est_slice_df = estimation_df(slice_df)

            if check_dict["match"] and slice_df["echo"].nunique() > 2:
                # === Now we try the linear contrast matching ===
                # (average_command, topup_file_list) = linear_contrast_match(
                #     subject,
                #     session,
                #     run,
                #     bvol_num,
                #     slice_num,
                #     slice_df,
                #     dir_dict["ave_dir"],
                # )
                (average_command, topup_file_list) = geomean_contrast_match(
                    subject,
                    session,
                    run,
                    bvol_num,
                    slice_num,
                    est_slice_df,
                    dir_dict["ave_dir"],
                )
                # echo_list = [2 if e % 2 == 0 else 1 for e in echo_list]
            else:
                average_command = None
                topup_file_list = est_slice_df["nii"].tolist()

            # merge the niis
            (merge_command, merged_filename) = get_fslmerge_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                dir_dict["work_dir"],
                topup_file_list,
                suffix,
            )
            # median filter the merged file
            # (filt_command, merged_filename) = get_filt_command(
            #     subject,
            #     session,
            #     run,
            #     bvol_num,
            #     slice_num,
            #     dir_dict["topup_dir"],
            #     merged_filename,
            # )
            # run topup
            (topup_command, topup_base) = get_topup_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                dir_dict["work_dir"],
                str(acq_file.absolute()),
                config,
                merged_filename,
                suffix,
                batch_size=n_bvols,
            )

            # apply topup to the orignal images
            # per-slice corrected files are combinenii intermediates and stay
            # in the work tree; whole-volume outputs are final products
            (apply_command_list, corrected_file_list) = get_applytopup_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                str(acq_file.absolute()),
                dir_dict["work_dir"] if check_dict["slice"] else dir_dict["topup_dir"],
                topup_base,
                original_file_list,
                echo_list,
                suffix,
            )

            commands = [
                average_command,
                merge_command,
                # filt_command,
                topup_command,
            ]
            commands = commands + apply_command_list

            # concatenate the commands and add them to the list
            full_topup_command = " && ".join(cmd for cmd in commands if cmd is not None)
            topup_cmd_list.append(full_topup_command)
            # TODO: need to deal with the case there is an odd dimension in the volume and topup fails

        if check_dict["slice"]:
            # if we are going slice by slice we dispatch now and reinitialize
            # the topup command list
            parallel_bash_commands(topup_cmd_list, "topup commands per-slice")
            # serial_bash_commands(topup_cmd_list, "topup commands per-slice")
            topup_cmd_list = []

    if not check_dict["slice"]:
        parallel_bash_commands(topup_cmd_list, "topup commands per-bvol")

    return


def find_closest_volume(img: NDArray, comparison_imgs: list[NDArray]) -> np.intp:
    """Find the index of the closest image volume in a list to a given image."""
    # we want to find the volume in comparison_imgs that is most similar to img
    # we will do this by calculating the SSIM between img and each volume in comparison_imgs
    # then we will return the index of the volume with the highest SSIM
    data_range = np.max(np.stack([img, *comparison_imgs])) - np.min(
        np.stack([img, *comparison_imgs])
    )
    ssim_list: list[float] = [
        ssim(img, comp_img, data_range=data_range, win_size=5)
        for comp_img in comparison_imgs
    ]
    # print(f"SSIM list: {ssim_list}")
    return np.argmax(ssim_list)


def find_closest_volume_nmi(img: NDArray, comparison_imgs: list[NDArray]) -> np.intp:
    """Find the index of the closest volume in comparison_imgs to img.

    NMI is a similarity: identical images -> 2.0, independent -> ~1.0.
    """
    nmi_list: list[float] = [nmi(img, comp_img) for comp_img in comparison_imgs]
    return np.argmax(nmi_list)


# def get_applytopup_to_othervol_command(
#     subject: str,
#     session: str,
#     run: str,
#     curr_bvol_num: int,
#     topup_bvol_num: int,
#     slice_num: int,
#     acq_file: str,
#     topup_dir: str,
#     topup_base: str,
#     original_file_list: list[str],
#     echo_list: list[int],
#     suffix: str,
# ) -> tuple[str, list[str]]:
#     topup_out_file_base = os.path.join(
#         topup_dir,
#         f"{subject}_{session}_{run}_bv-{topup_bvol_num}_desc-topupcorr-{suffix}",
#     )
#     curr_out_file_base = os.path.join(
#         topup_dir,
#         f"{subject}_{session}_{run}_bv-{curr_bvol_num}_desc-topupcorr-{suffix}",
#     )
#     ending_string = f"_sv-{str(slice_num).zfill(3)}.nii"
#     out_file_list = []
#     command_list = []
#     for echo, nii in zip(echo_list, original_file_list):
#
#
#     return


def run_topup_diffusion_special(
    subject: str,
    session: str,
    run: str,
    config: str,
    dir_dict: dict,
    curr_df: pd.DataFrame,
    cutoff: int = 1000,
) -> None:
    """Run topup on a given subject and run but do so with special considerations for diffusion data."""
    print(f"Running TOPUP Diffusion Special for {subject} {session} {run}")
    suffix: str = dir_dict["inner_dir"]
    acq_file = make_acq_params(subject, session, run, curr_df, dir_dict["work_dir"])
    if acq_file is None:
        print("Could not make or find acqparams.txt file, skipping")
        return
    # we want to iterate over each bvol_num and each slice_num FOR BVALS UNDER A CUTOFF
    # for each we want to make a long command that will:
    # 1. merge
    # 2. topup
    # 3. applytopup
    # BUT crucially for each volume with a bval above the cutoff we to select a field from a volume below the cutoff
    # and then apply it to those volumes
    # cutoff = 1000
    curr_df.sort_values(["bvol_num", "echo"], inplace=True)
    lowb_df: pd.DataFrame = curr_df[curr_df["bval"] <= cutoff]
    # print(lowb_df.head())
    highb_df: pd.DataFrame = curr_df[curr_df["bval"] > cutoff]
    low_bvol_num_list: list[int] = lowb_df["bvol_num"].unique().tolist()
    high_bvol_num_list: list[int] = highb_df["bvol_num"].unique().tolist()
    # for each slice we will want to load all the low bval images and find the closest volume to the high bval image
    # then we will use that volume as the field for the high bval image and apply it
    slice_list: list[int] = curr_df["slice_num"].unique().tolist()
    shuffle(slice_list)

    phase2_cmd_list: list[str] = []
    for slice_num in slice_list:
        apply_dict = {}
        print(f"Working on slice {slice_num}")
        lowb_slice_df = lowb_df[lowb_df["slice_num"] == slice_num].sort_values(
            "bvol_num"
        )
        lowb_slice_echo1_df: pd.DataFrame = lowb_slice_df[lowb_slice_df["echo"] == 1]

        # load all the lowb first echo images for this slice.  Choosing which
        # low-b field to reuse is an estimation decision, so it reads the
        # (possibly denoised) estimation images.
        lowb_slice_echo1_df = estimation_df(lowb_slice_echo1_df)
        lowb_niis: list[str] = lowb_slice_echo1_df["nii"].tolist()
        lowb_bvol_num_list: list[int] = lowb_slice_echo1_df["bvol_num"].tolist()
        # In a real run these images are guaranteed to exist by now, so they
        # are loaded unguarded and a missing one is a genuine error.  Only
        # --dryrun can legitimately reach here before they are written.
        if check_dict["dryrun"] and not all(os.path.exists(nii) for nii in lowb_niis):
            print(
                "Estimation images for volume matching do not exist yet "
                "(--dryrun): every high-b volume in this slice is shown "
                "using the first low-b field, so the printed high-b plan is "
                "a placeholder."
            )
            lowb_echo1_imgs: list[NDArray] = []
        else:
            lowb_echo1_imgs = [nib.load(nii).get_fdata() for nii in lowb_niis]

        print(f"Low bval images: {lowb_bvol_num_list}")

        highb_slice_df: pd.DataFrame = highb_df[
            highb_df["slice_num"] == slice_num
        ].sort_values("echo")
        highb_slice_echo1_df: pd.DataFrame = estimation_df(
            highb_slice_df[highb_slice_df["echo"] == 1]
        ).sort_values("bvol_num")
        # for bvol_num in high_bvol_num_list:
        for bvol_num in tqdm(high_bvol_num_list):
            echo1_nii = highb_slice_echo1_df[
                highb_slice_echo1_df["bvol_num"] == bvol_num
            ]["nii"].values[0]
            if not lowb_echo1_imgs:
                closest_lowb_index = 0
            else:
                echo1_img = nib.load(echo1_nii).get_fdata()
                # closest_lowb_index = find_closest_volume(echo1_img,
                # lowb_echo1_imgs)
                closest_lowb_index = find_closest_volume_nmi(echo1_img, lowb_echo1_imgs)
            closest_lowb_bvol_num = lowb_bvol_num_list[closest_lowb_index]
            # print(
            # f"Closest low b volume to high b {bvol_num} is {closest_lowb_bvol_num} for slice {slice_num}"
            # )
            apply_dict[bvol_num] = closest_lowb_bvol_num

        # use these to setup a mapping from the low bvals to the high bvals which we will then use to construct apply commands

        inv_apply_dict: dict[int, list[int]] = {}
        for k, v in apply_dict.items():
            inv_apply_dict[v] = inv_apply_dict.get(v, []) + [k]

        print(f"APPLY DICT: {inv_apply_dict}")

        slice_df: pd.DataFrame = curr_df[curr_df["slice_num"] == slice_num].sort_values(
            ["bvol_num", "echo"]
        )
        topup_cmd_list: list[str] = []
        for bvol_num in low_bvol_num_list:
            bvol_df: pd.DataFrame = lowb_slice_df[
                lowb_slice_df["bvol_num"] == bvol_num
            ].sort_values("echo")
            original_file_list: list[str] = bvol_df["nii"].tolist()
            print(f"ORIGINAL FILE LIST: {original_file_list}")
            echo_list: list[int] = bvol_df["echo"].tolist()
            if original_file_list is None or echo_list is None:
                print(
                    f"Missing files or info for {subject} {session} {run}"
                    + f"bvol {bvol_num} slice {slice_num}"
                )
                exit(1)
            est_bvol_df = estimation_df(bvol_df)
            if check_dict["match"] and bvol_df["echo"].nunique() > 2:
                (average_command, topup_file_list) = geomean_contrast_match(
                    subject,
                    session,
                    run,
                    bvol_num,
                    slice_num,
                    est_bvol_df,
                    dir_dict["ave_dir"],
                )
            else:
                average_command = None
                topup_file_list = est_bvol_df["nii"].tolist()

            (merge_command, merged_filename) = get_fslmerge_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                dir_dict["work_dir"],
                topup_file_list,
                suffix,
            )
            (topup_command, topup_base) = get_topup_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                dir_dict["work_dir"],
                str(acq_file.absolute()),
                config,
                merged_filename,
                suffix,
                batch_size=len(low_bvol_num_list),
            )
            (apply_command_list, _) = get_applytopup_command(
                subject,
                session,
                run,
                bvol_num,
                slice_num,
                str(acq_file.absolute()),
                dir_dict["work_dir"] if check_dict["slice"] else dir_dict["topup_dir"],
                topup_base,
                original_file_list,
                echo_list,
                suffix,
            )
            highb_apply_command_list: list[str] = []
            apply_bvol_list: list[int] = inv_apply_dict.get(bvol_num, [])
            print(f"Applying to {apply_bvol_list}")
            if len(apply_bvol_list) > 0:
                for high_bvol_num in apply_bvol_list:
                    curr_highb_df: pd.DataFrame = highb_slice_df[
                        highb_slice_df["bvol_num"] == high_bvol_num
                    ]
                    highb_echo_list = curr_highb_df["echo"].tolist()
                    highb_echo_list.sort()
                    highb_nii_list = curr_highb_df["nii"].tolist()
                    highb_nii_list.sort()
                    (one_highb_apply_cmd_list, _) = get_applytopup_command(
                        subject,
                        session,
                        run,
                        high_bvol_num,
                        slice_num,
                        str(acq_file.absolute()),
                        dir_dict["work_dir"]
                        if check_dict["slice"]
                        else dir_dict["topup_dir"],
                        topup_base,
                        highb_nii_list,
                        highb_echo_list,
                        suffix,
                    )
                    if one_highb_apply_cmd_list is not None:
                        highb_apply_command_list = (
                            highb_apply_command_list + one_highb_apply_cmd_list
                        )

            # print(f"High bval apply commands: {highb_apply_command_list}")
            commands = [average_command, merge_command, topup_command]
            commands = commands + apply_command_list  # + highb_apply_command_list
            phase2_cmd_list = phase2_cmd_list + highb_apply_command_list
            full_topup_command = " && ".join(cmd for cmd in commands if cmd is not None)
            topup_cmd_list.append(full_topup_command)
        parallel_bash_commands(topup_cmd_list, "running TOPUP per-bvol below b cutoff")
    parallel_bash_commands(phase2_cmd_list, "ApplyTOPUP Commands for high b diffusion")
    pass


def get_run_info(curr_dir: str, subject: str, session: str, run: str) -> pd.DataFrame:
    """Make a dataframe with the run information for a given subject and run.

    we want to construct a dataframe with the following columns:
        nifti_filename, subject_ID, run_num, echo_num, TR, TE, phase_dir,
        readout_time, bvols, bvals
    we will want to load the json file for each nifti file
    and extract the relevant information we will also want to check if the bvals file
    exists and if so, read it in
    """
    # find all the jsons
    json_files = find_files(curr_dir, "*.json")
    # echo_dict_list = []
    vol_dict_list = []
    for json_file in json_files:
        echo_dict = get_echo_info(json_file)
        # echo_dict_list.append(get_echo_info(json_file))
        # now make an dict for each volume

        # hacky to get the list back out of the dict correctly
        nii_list = [nii for nii in echo_dict["nii_list"]]

        # we need to check if we have bvals
        if echo_dict["bvals_list"][0] is None:
            # check for multiple averages if we don't have bvals
            for av, nii in enumerate(nii_list):
                vol_dict = {
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "echo": echo_dict["echo"],
                    "bvol_num": av + 1,  # we are abusing bvol_num to be average number
                    "bval": None,
                    "nii": nii,
                    "echo_time": echo_dict["echo_time"],
                    "readout_time": echo_dict["readout_time"],
                    "phase_dir": echo_dict["phase_dir"],
                    "slice_orientation": echo_dict["slice_orientation"],
                    "orientation_num": echo_dict["orientation_num"],
                    "pulse_sequence": echo_dict["pulse_sequence"],
                    "series_description": echo_dict["series_description"],
                    "image_type": echo_dict["image_type"],
                    "orientation_list": echo_dict["orientation_list"],
                    "orientation_rounded": echo_dict["orientation_rounded"],
                }
                vol_dict_list.append(vol_dict)

        else:
            bvals_list_int = [int(i) for i in echo_dict["bvals_list"]]
            for bn, (b, nii) in enumerate(
                zip(
                    bvals_list_int,
                    nii_list,
                )
            ):
                vol_dict = {
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "echo": echo_dict["echo"],
                    "bvol_num": bn + 1,  # enumerate starts at 0 so we add 1
                    "bval": b,
                    "nii": nii,
                    "echo_time": echo_dict["echo_time"],
                    "readout_time": echo_dict["readout_time"],
                    "phase_dir": echo_dict["phase_dir"],
                    "slice_orientation": echo_dict["slice_orientation"],
                    "orientation_num": echo_dict["orientation_num"],
                    "pulse_sequence": echo_dict["pulse_sequence"],
                    "series_description": echo_dict["series_description"],
                    "image_type": echo_dict["image_type"],
                    "orientation_list": echo_dict["orientation_list"],
                    "orientation_rounded": echo_dict["orientation_rounded"],
                }
                vol_dict_list.append(vol_dict)

    # make a dataframe that contains a row for each nifti file
    return pd.DataFrame(vol_dict_list)


def get_slicenii_cmd(nii_path: str, slice_dir: str, orientation_num: int) -> str:
    """Get the slicenii command for a given nifti file."""
    return f"slicenii -i {nii_path} -o {slice_dir} -p 6"


def get_combinenii_cmd(
    input_dir: str, ref_nii: str, start_string: str, output_filename: str
) -> str:
    """Get the combinenii command for a given directory and reference nifti file."""
    return (
        f"combinenii -i {input_dir} -r {ref_nii} -s {start_string} -o {output_filename}"
    )


def run_bash_command(cmd: str, dryrun: bool = False) -> tuple[str, int, str]:
    """Run a shell command; return (cmd, returncode, error output).

    ``dryrun`` is passed explicitly (never read from module state) because
    this function runs inside worker processes: with Python >= 3.14 the
    default multiprocessing start method re-imports the module, so globals
    created by main() do not exist there.
    """
    if dryrun:
        print(f"\nDryrun - command: {cmd}\n")
        return (cmd, 0, "")
    print(f"\nRunning command: {cmd}\n")
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return (cmd, 0, "")
    except subprocess.CalledProcessError as e:
        output = (
            e.output.decode(errors="replace")
            if isinstance(e.output, bytes)
            else str(e.output)
        )
        print(f"Command '{cmd}' failed. Output: {output}, code: {e.returncode}")
        return (cmd, e.returncode, output[-2000:])
    except OSError as e:
        print(f"Command '{cmd}' failed with OSError: {e}")
        return (cmd, -1, str(e))


# Failed (cmd, returncode) pairs collected by the dispatchers in the parent
# process; main() reports them and sets the exit status.
failed_commands: list[tuple[str, int]] = []


def parallel_bash_commands(
    bash_commands: list[str] | None, description: str
) -> list[tuple[str, int]]:
    """Run a list of bash commands in parallel; return the ones that failed.

    Failures are also recorded in the module-global ``failed_commands`` for the
    exit status, but callers that must react to a failure of their own batch
    (denoising, which cannot let the run continue) read the return value.
    """
    if bash_commands is None:
        print(f"No commands to run for {description}")
        return []
    bash_commands = [cmd for cmd in bash_commands if cmd is not None]
    if len(bash_commands) == 0:
        print(f"No commands to run for {description}")
        return []

    print("\n")
    print(f"Running {description} in parallel.")
    print("\n")
    jobs = check_dict.get("jobs") or default_jobs()
    n_workers = max(1, min(jobs, len(bash_commands)))
    dryrun = check_dict["dryrun"]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_bash_command, cmd, dryrun): cmd for cmd in bash_commands
        }

        progress_bar = tqdm(total=len(bash_commands), desc=description, ncols=100)
        failed: list[tuple[str, int]] = []
        for future in as_completed(futures):
            _cmd, returncode, _err = future.result()
            if returncode != 0:
                failed.append((_cmd, returncode))
                failed_commands.append((_cmd, returncode))
            progress_bar.update(1)
        progress_bar.close()
    return failed


def serial_bash_commands(
    bash_commands: list[str] | None, description: str
) -> list[tuple[str, int]]:
    """Run a list of bash commands in serial; return the ones that failed."""
    if bash_commands is None:
        print(f"No commands to run for {description}")
        return []
    bash_commands = [cmd for cmd in bash_commands if cmd is not None]
    if len(bash_commands) == 0:
        print(f"No commands to run for {description}")
        return []

    print("\n")
    print(f"Running {description} in serial.")
    print("\n")

    dryrun = check_dict["dryrun"]
    if dryrun:
        print("Dryrun - not running commands")

    progress_bar = tqdm(total=len(bash_commands), desc=description, ncols=100)
    failed: list[tuple[str, int]] = []
    for cmd in bash_commands:
        _cmd, returncode, _err = run_bash_command(cmd, dryrun)
        if returncode != 0:
            failed.append((_cmd, returncode))
            failed_commands.append((_cmd, returncode))
        progress_bar.update(1)
    progress_bar.close()
    return failed


def find_slices(slice_dir: str, nii: str) -> list[str]:
    """The slice files slicenii wrote for ``nii``, in sorted order."""
    nii_basename = os.path.basename(nii).split(".")[0]
    return find_files(
        os.path.join(slice_dir, f"{nii_basename}_slices"),
        f"{nii_basename}_*",
    )


def slice_number(nii: str) -> int:
    """Slice index parsed out of a slicenii output filename."""
    return int(nii.split("_")[-1].split("-")[-1].split(".")[0])


def skip_run(run_df: pd.DataFrame, reason: str) -> None:
    """Report that this run is skipped and make the invocation exit non-zero.

    Used where a helper, not ``process_run`` itself, discovers that the run
    cannot proceed; the helper returns this ``None`` and the caller returns.
    """
    subject, session, run = (run_df[c].iloc[0] for c in ("subject", "session", "run"))
    print(f"SKIPPING {subject} {session} {run}: {reason}")
    failed_commands.append((f"skipped {subject} {session} {run}: {reason}", 1))
    return None


def handle_slicing(run_df: pd.DataFrame, dir_dict: dict) -> pd.DataFrame | None:
    """Slice the volumes into individual padded slices."""
    # print("here")
    # print(run_df)
    run_df["slice_dir"] = dir_dict["slice_dir"]
    row_list = [pd.Series(row, index=run_df.columns) for _, row in run_df.iterrows()]

    slicenii_cmd_list = []
    for row in row_list:
        # the estimation image is a different file when denoising is on, and
        # topup needs its slices too; dict.fromkeys drops the duplicate when
        # denoising is off and the two columns are the same file
        for nii in dict.fromkeys((row["nii"], row["est_nii"])):
            slicenii_cmd_list.append(
                get_slicenii_cmd(nii, row["slice_dir"], row["orientation_num"])
            )
    failed = parallel_bash_commands(slicenii_cmd_list, "slicenii commands")
    if failed:
        return skip_run(
            run_df,
            f"{len(failed)} slicenii command(s) failed, so some volumes would "
            "have no slices",
        )
    # now we need to make a dataframe that contains a row for each slice
    # slice_niis = find_files(dir_dict["slice_dir"], "*.nii").sort()
    # print("there")
    # print(run_df)
    df_list = [run_df]
    for row in row_list:
        # print(row)
        slice_niis = find_slices(dir_dict["slice_dir"], row["nii"])
        est_nii = row["est_nii"]
        if est_nii == row["nii"]:
            # no denoising: the estimation slices ARE the original slices, so
            # do not glob the same directory twice
            est_by_num = {slice_number(nii): nii for nii in slice_niis}
        else:
            est_by_num = {
                slice_number(nii): nii
                for nii in find_slices(dir_dict["slice_dir"], est_nii)
            }
            # Quietly reusing the original slice here would estimate the field
            # from un-denoised data inside a _denoise-{method} tree.
            missing = [nii for nii in slice_niis if slice_number(nii) not in est_by_num]
            if missing:
                return skip_run(
                    run_df,
                    f"{len(missing)} of {len(slice_niis)} estimation slices "
                    f"are missing for {os.path.basename(est_nii)} "
                    f"(looked in {dir_dict['slice_dir']})",
                )
        slice_rows = []
        for nii in slice_niis:
            number = slice_number(nii)
            slice_rows.append(
                {
                    "subject": row["subject"],
                    "session": row["session"],
                    "run": row["run"],
                    "echo": row["echo"],
                    "bvol_num": row["bvol_num"],
                    "bval": row["bval"],
                    "nii": nii,
                    "est_nii": est_by_num[number],
                    "phase_dir": row["phase_dir"],
                    "echo_time": row["echo_time"],
                    "slice_dir": row[
                        "slice_dir"
                    ],  # TODO: slice directory vs direction might be confusing here
                    "readout_time": row["readout_time"],
                    "volume_type": "slice",
                    "slice_num": number,
                }
            )
        slice_df = pd.DataFrame(slice_rows)
        df_list.append(slice_df)
    run_df = pd.concat(df_list, ignore_index=True)
    return run_df


def handle_combining(
    run_df: pd.DataFrame,
    slices_dir: str,
    out_dir: str,
    inner_dir: str,
    corr: str,
) -> pd.DataFrame:
    """Combine the corrected slices in ``slices_dir`` into volumes in ``out_dir``."""

    curr_df = run_df[run_df["volume_type"] == "volume"]
    original_niis = curr_df["nii"].unique()

    combine_cmd_list = []
    combined_df_list = []
    for nii in original_niis:
        nii_df = curr_df[curr_df["nii"] == nii]
        # print(f"{nii_df}")
        subject = nii_df["subject"].iloc[0]
        session = nii_df["session"].iloc[0]
        run = nii_df["run"].iloc[0]
        bvol_num = nii_df["bvol_num"].iloc[0]
        echo_num = nii_df["echo"].iloc[0]

        start_string = (
            f"{subject}_{session}_{run}_bv-{bvol_num}_"
            + f"desc-{corr}-{inner_dir}_echo-{echo_num}_sv-"
        )
        output_filename = os.path.join(
            out_dir,
            f"{subject}_{session}_{run}_bv-{bvol_num}_desc-{corr}"
            + f"-recombined-volume_echo-{echo_num}.nii",
        )
        combine_cmd_list.append(
            get_combinenii_cmd(slices_dir, nii, start_string, output_filename)
        )
        nii_df["combined_nii"] = output_filename
        combined_df_list.append(nii_df)

    parallel_bash_commands(combine_cmd_list, "combinenii commands")
    return pd.concat(combined_df_list, ignore_index=True)


# def handle_combining_fields(
#         run_df: pd.DataFrame,
#         output_dir: str,
#         inner_dir: str,
#         corr: str,
#         ) -> None:
#     """combine the slices of the calculated fields back into volumes"""
#     output_filename = os.path.join(
#         output_dir,
#         f"{subject}_{session}_{run}_desc-{corr}corr-recombined-field.nii",
#     )
#
#     return


def get_topup_command(
    subject: str,
    session: str,
    run: str,
    bvol_num: int,
    slice_num: int,
    topup_dir: str,
    acq_file: str,
    config: str,
    merged_filename: str,
    suffix: str,
    batch_size: int = 1,
) -> tuple[None, str] | tuple[str, str]:
    """Create the topup command for a given slice, volume, etc.

    ``batch_size`` is how many of these commands will be dispatched together;
    it sets the per-process thread budget (see topup_threads).
    """
    out_file_base: str = os.path.join(
        topup_dir,
        f"topup-result_{subject}_{session}_{run}_bv-{bvol_num}_sv-{slice_num}_{suffix}".replace(
            ".", "-"
        ),
    )

    field_filename = out_file_base + "_field.nii"
    corr_filename = out_file_base + "_desc-topupcorr.nii"
    if os.path.exists(corr_filename):
        print(f"Topup already run for {out_file_base}")
        return (None, out_file_base)
    else:
        command = (
            f"topup --imain={merged_filename}"
            + f" --datain={acq_file}"
            + f" --config={config}"
            + " --scale=1"
            + f" --out={out_file_base}"
            + f" --fout={field_filename}"
            + f" --iout={corr_filename}"
        )

        if check_dict["topup_multithread"]:
            command = command + " --nthr=" + str(topup_threads(batch_size))

    return (command, out_file_base)


def get_applytopup_command(
    subject: str,
    session: str,
    run: str,
    bvol_num: int,
    slice_num: int,
    acq_file: str,
    topup_dir: str,
    topup_base: str,
    original_file_list: list[str],
    echo_list: list[int],
    suffix: str,
) -> tuple[list, list]:
    """Apply the topup field correction back to the original volumes."""
    out_file_base = os.path.join(
        topup_dir,
        f"{subject}_{session}_{run}_bv-{bvol_num}_desc-undistorted-{suffix}",
    )

    ending_string = f"_sv-{str(slice_num).zfill(3)}.nii"
    out_file_list = []
    command_list = []

    for echo, nii in zip(echo_list, original_file_list):
        out_file = out_file_base + f"_echo-{echo}" + ending_string
        print(f"out_file: {out_file}, nii: {nii}")

        if os.path.exists(out_file):
            print(f"Applytopup already run for {out_file}")
            command = None
        else:
            # Since we don't have bvol_df here, simply check if match is enabled
            # The actual echo count check is done in the contrast match functions
            if check_dict["match"]:
                e = 2 if echo % 2 == 0 else 1
            else:
                e = echo

            command = (
                f"applytopup --imain={nii}"
                + f" --datain={acq_file}"
                + f" --inindex={e}"
                + f" --topup={topup_base}"
                + f" --out={out_file}"
                + " --method=jac"
            )

        out_file_list.append(out_file)
        command_list.append(command)

    return (command_list, out_file_list)


def process_run(
    subject: str,
    session: str,
    run: str,
    output_dir: str,
    deriv_dir: str,
    config: str,
    mask_dir: str | None = None,
    cutoff: int = 1000,
    work_root: str | None = None,
) -> None:
    """Call selected functions for a given subject and run."""
    print(f"Processing {subject}, {session}, {run}")
    mask_file = None
    if mask_dir is not None:
        # check if we have a mask for the current run
        mask_file_list = find_files(mask_dir, f"{subject}_{session}_{run}*-label.nii")
        mask_file_list_alt = find_files(mask_dir, f"{subject}_{session}_{run}*mask.nii")
        mask_file_list = mask_file_list + mask_file_list_alt

        if len(mask_file_list) == 0:
            mask_file = None
            print(f"Mask file not found in {mask_dir}, skipping masking")
            check_dict["mask"] = False
        else:
            mask_file = mask_file_list[0]
            print(f"Found mask: {mask_file}")
            check_dict["mask"] = True

    dir_dict: dict = set_dirs(subject, session, run, output_dir, deriv_dir, work_root)

    # find the number of volumes in the run
    volume_list: list = find_files(dir_dict["current_dir"], "*.nii")
    if len(volume_list) < 2:
        # we need two or more volumes to do anything
        print(f"Found {len(volume_list)} volumes, at least two are required, skipping")
        return

    # make all the output dirs we need
    for key, value in dir_dict.items():
        if not key == "inner_dir":
            os.makedirs(value, exist_ok=True)

    run_df: pd.DataFrame = get_run_info(dir_dict["current_dir"], subject, session, run)
    # print(run_df)
    # check if we even have "echo" in run_df
    if "echo" not in run_df.columns:
        print("No echo information found, skipping")
        return

    if run_df["echo"].nunique() < 2:
        # delete empty output directories
        if os.listdir(dir_dict["topup_dir"]) == []:
            os.rmdir(dir_dict["topup_dir"])
        print("might have multiple volumes but not multiple echoes, skipping")
        return

    if check_dict["match"] and run_df["echo"].nunique() < 3:
        print("Contrast matching enabled but fewer than 3 echoes found, skipping run.")
        return

    if check_dict["two_echo"] and run_df["echo"].nunique() > 2:
        # filter to only use the first two echoes in some cases
        run_df: pd.DataFrame = run_df[run_df["echo"] < 3]
        dir_dict["inner_dir"] = dir_dict["inner_dir"] + "_two-echo"
        dir_dict["topup_dir"] = os.path.join(
            deriv_dir,
            "undistortme",
            dir_dict["inner_dir"],
            subject,
            session,
            run,
        )
        work_dir = os.path.join(
            dir_dict["work_root"], dir_dict["inner_dir"], subject, session, run
        )
        dir_dict["work_dir"] = work_dir
        dir_dict["ave_dir"] = work_dir
        if "slice_dir" in dir_dict:
            dir_dict["slice_dir"] = work_dir
        if "masked_dir" in dir_dict:
            dir_dict["masked_dir"] = work_dir
        os.makedirs(dir_dict["topup_dir"], exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

    # handle masking if we have a mask
    if check_dict["mask"] and mask_file is not None:
        print(f"Masking with {mask_file}")
        mask_command_list: list = []
        for nii in run_df["nii"].unique():
            mask_out = os.path.join(
                dir_dict["masked_dir"],
                os.path.basename(nii.replace(".nii", "_desc-masked.nii")),
            )
            mask_command = f"fslmaths {nii} -mas {mask_file} {mask_out}"
            mask_command_list.append(mask_command)
            # print(f"Masking {nii} with {mask_file}")
            # mask_nii = nib.load(mask_file)
            # mask = mask_nii.get_fdata()
            # # check for odd number dimensions in the mask
            # # find max and min x, y, z of the Mask
            # x_min = np.min(np.where(mask != 0)[0])
            # x_max = np.max(np.where(mask != 0)[0])
            # y_min = np.min(np.where(mask != 0)[1])
            # y_max = np.max(np.where(mask != 0)[1])
            # if (x_max - x_min + 1) % 2 != 0:
            #     try:
            #         mask[x_max + 1, y_min:y_max, :] = 1
            #     except IndexError:
            #         mask[x_min - 1, y_min:y_max, :] = 1
            # if (y_max - y_min + 1) % 2 != 0:
            #     try:
            #         mask[x_min:x_max, y_max + 1, :] = 1
            #     except IndexError:
            #         mask[x_min:x_max, y_min - 1, :] = 1
            #
            # img = nib.load(nii)
            # img_data = img.get_fdata()
            # img_data[mask == 0] = 0
            # img_masked = nib.Nifti1Image(img_data, img.affine, img.header)
            # img_crop = crop_img(img_masked)
            # nib.save(img_crop, mask_out)

        parallel_bash_commands(mask_command_list, "masking commands")
        run_df["nii"] = [
            os.path.join(
                dir_dict["masked_dir"],
                os.path.basename(nii.replace(".nii", "_desc-masked.nii")),
            )
            for nii in run_df["nii"]
        ]

    # denoise for field estimation only: this adds an "est_nii" column and
    # leaves "nii" (what applytopup corrects) alone
    run_df = handle_denoising(run_df, dir_dict)
    if run_df is None:
        return
    run_df["volume_type"] = "volume"
    if check_dict["slice"]:
        run_df = handle_slicing(run_df, dir_dict)
        if run_df is None:
            return
    else:
        run_df["slice_num"] = 1

    # filter to either slice volumes or full volumes
    if check_dict["slice"]:
        curr_df = run_df[run_df["volume_type"] == "slice"]
        curr_df["slice_num"] = curr_df["slice_num"].astype(int)
    else:
        curr_df = run_df[run_df["volume_type"] == "volume"]

    if check_dict["topup"]:
        if "bval" not in curr_df.columns:
            run_topup(subject, session, run, config, dir_dict, curr_df)
        elif curr_df["bval"].nunique() > 1:
            run_topup_diffusion_special(
                subject, session, run, config, dir_dict, curr_df, cutoff
            )
        else:
            run_topup(subject, session, run, config, dir_dict, curr_df)

        if check_dict["slice"]:
            # combine slices back into volumes
            curr_df = run_df[run_df["volume_type"] == "volume"]
            run_df = handle_combining(
                run_df,
                dir_dict["work_dir"],
                dir_dict["topup_dir"],
                dir_dict["inner_dir"],
                "undistorted",
            )

    # if check_dict["t2"]:
    #     run_t2fit()

    return


def main() -> None:
    """Get arguments, verify requirements, and run the pipeline."""
    # Get arguments
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    deriv_dir = args.derivdir
    topup_check = args.run_topup
    dcm2niix_check = args.run_dcm2niix
    slice_check = args.slice
    match_check = args.matchcontrast
    cutoff = int(args.cutoff)
    (
        dcm2niix_check,
        topup_check,
        topup_multithread,
        slice_check,
    ) = check_binaries(dcm2niix_check, topup_check, slice_check)
    config = resolve_config(args.config_file, slice_check)

    # Set environment variable
    os.environ["FSLOUTPUTTYPE"] = (
        "NIFTI"  # we choose not to compress for now for simplicity
    )
    # make check_dict global because we will refer to it a lot
    # and it is constant for each run of this script
    global check_dict

    two_echo_check: bool = args.twoecho and not match_check

    check_dict = {
        "dcm2niix": dcm2niix_check,
        "topup": topup_check,
        "topup_multithread": topup_multithread,
        "slice": slice_check,
        "match": match_check,
        "dryrun": args.dryrun,
        "two_echo": two_echo_check,
        "jobs": args.jobs or default_jobs(),
        "oversubscribe": args.oversubscribe or DEFAULT_OVERSUBSCRIBE,
        "denoise": check_denoise(args.denoise),
    }
    if args.maskdir is not None:
        check_dict["mask"] = True
    else:
        check_dict["mask"] = False

    if args.bids is not None:
        # BIDS ingest mode: stage the dataset into the per-volume 3D layout
        # and use the staging tree as the input root for the unchanged
        # process_run flow below. (--bids/-d exclusivity is enforced by
        # argparse.)
        work_root = args.workdir or default_work_root(deriv_dir)
        staging_root = os.path.join(work_root, bids_input.DEFAULT_STAGING_SUBDIR)
        print(f"BIDS input: staging {args.bids} into {staging_root}")
        if check_dict["dryrun"]:
            print(
                "NOTE: --dryrun does not skip BIDS staging; staged files "
                "are written so the printed commands reference real "
                "inputs."
            )
        try:
            staged = bids_input.stage_bids(
                args.bids, staging_root, args.echo_entity, args.echo_from_acq
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"ERROR: {e}")
            exit(1)
        if len(staged) == 0:
            print(
                f"ERROR: no BIDS dwi runs found under {args.bids} "
                "(expected sub-*/[ses-*/]dwi/*_dwi.nii[.gz])"
            )
            exit(1)
        output_dir = staging_root
    elif bids_input.looks_like_bids(output_dir):
        # hint only - never a silent mode switch
        print(
            f"HINT: {output_dir} looks like a BIDS dataset "
            "(sub-*/[ses-*/]dwi/ and no run-* directories); did you mean "
            f"'undistortme --bids {output_dir} ...'? "
            "Proceeding with the legacy run-directory layout."
        )

    if check_dict["dcm2niix"]:
        command = convert_dcms(input_dir, output_dir)
        _cmd, returncode, _err = run_bash_command(command, check_dict["dryrun"])
        if returncode != 0:
            failed_commands.append((_cmd, returncode))

    print("\n--------------------------")
    if check_dict["topup"]:
        subjects = (
            [args.subject_dir]
            if args.subject_dir is not None
            else [
                os.path.basename(x)
                for x in glob.glob(os.path.join(output_dir, "sub-*"))
            ]
        )

        if len(subjects) == 0:
            print(
                "ERROR: No subjects found."
                + "(Try using -d flag to convert source DICOMS to NIFTI.)"
            )
            exit(1)

        shuffle(subjects)
        for subject in subjects:
            if "." in subject and args.fix_names:
                dir = os.path.join(output_dir, subject)
                subject = rename_entities(dir)

            sessions = [
                os.path.basename(x)
                for x in glob.glob(os.path.join(output_dir, subject, "ses-*"))
            ]

            if len(sessions) == 0:
                print(f"WARNING: No sessions found for {subject}.")
                continue
            shuffle(sessions)

            for session in sessions:
                # get number of runs for this subject
                runs = [
                    os.path.basename(x)
                    for x in glob.glob(
                        os.path.join(output_dir, subject, session, "run-*")
                    )
                ]
                # runs.sort(key=lambda x: int(x.split("run-")[-1].split("_")[0]))
                shuffle(runs)

                if len(runs) == 0:
                    print(f"WARNING: No runs found for {subject} {session}.")
                    continue

                for run in runs:
                    process_run(
                        subject,
                        session,
                        run,
                        output_dir,
                        deriv_dir,
                        config,
                        args.maskdir,
                        cutoff,
                        args.workdir,
                    )
    else:
        print("WARNING: TOPUP correction not selected, nothing to do.")

    if failed_commands:
        print(f"\nERROR: {len(failed_commands)} command(s) failed:")
        for cmd, returncode in failed_commands:
            print(f"  [exit {returncode}] {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
