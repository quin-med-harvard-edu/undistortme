#!/usr/bin/env python3
"""Verify undistortme runs on the bundled example data.

Usage:
    python examples/verify_example.py [DERIVATIVES_DIR]

DERIVATIVES_DIR defaults to examples/derivatives (the location produced by
the quick-test commands in the README). Two runs are verified, matching the
two README commands:

  whole-volume               plain `-t`
  per-slice_contrast-matched `-t -s -m -c .../perslice_1.cnf`

For each variant the script checks the expected corrected volumes exist and
compares the echo-2 volumes voxelwise against reference outputs bundled in
examples/reference/<variant>/ (produced by the released Docker image).

The PASS threshold (Pearson correlation per volume) is calibrated from
measured runs: TOPUP's result varies slightly with the FSL build/CPU
(matching environments agree to ~1e-4; different builds still correlate at
>= 0.9996 for both variants), while an uncorrected or wrongly-corrected
volume correlates at <= 0.992. 0.995 separates the two cleanly. Requires
numpy and nibabel (installed with undistortme).
"""

import glob
import os
import sys

import nibabel as nib
import numpy as np

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join("sub-example", "ses-01", "run-1_desc-phantom4echo")
# variant -> (glob for its final corrected volumes (2 b-volumes x 4 echoes),
#             per-volume correlation PASS threshold)
VARIANTS = {
    "whole-volume": ("*desc-undistorted*_sv-001.nii", 0.995),
    "per-slice_contrast-matched": ("*recombined-volume*.nii", 0.995),
}
N_EXPECTED = 8


def check_variant(variant: str, deriv: str) -> bool:
    pattern, threshold = VARIANTS[variant]
    run_dir = os.path.join(deriv, "undistortme", variant, RUN)
    print(f"\n== {variant} ==")
    if not os.path.isdir(run_dir):
        print(
            f"FAIL: no outputs at {run_dir}\n"
            "Run the matching quick-test command from the README first."
        )
        return False

    corrected = sorted(glob.glob(os.path.join(run_dir, pattern)))
    print(f"corrected volumes found: {len(corrected)} (expected {N_EXPECTED})")
    ok = len(corrected) == N_EXPECTED

    refs = sorted(
        glob.glob(os.path.join(EXAMPLES_DIR, "reference", variant, "*.nii.gz"))
    )
    print(f"{'volume':<28} {'correlation':>12}  result")
    for ref in refs:
        name = os.path.basename(ref)[:-3]  # strip .gz
        out = os.path.join(run_dir, name)
        label = "bv-" + name.split("_bv-")[1].split("_desc")[0] + " echo-2"
        if not os.path.exists(out):
            print(f"{label:<28} {'missing':>12}  FAIL")
            ok = False
            continue
        a = np.asarray(nib.load(out).dataobj, dtype=np.float64)
        b = np.asarray(nib.load(ref).dataobj, dtype=np.float64)
        if a.shape != b.shape:
            print(f"{label:<28} {'shape!':>12}  FAIL")
            ok = False
            continue
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        good = corr >= threshold
        ok = ok and good
        print(f"{label:<28} {corr:>12.6f}  {'ok' if good else 'FAIL'}")
    return ok


def main() -> int:
    deriv = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXAMPLES_DIR, "derivatives")
    )
    ok = all([check_variant(v, deriv) for v in VARIANTS])
    print(
        "\nPASS: your installation reproduces the reference corrections."
        if ok
        else "\nFAIL: see above — missing runs or outputs deviating from the "
        "references. Check the FSL/slicenii install and the exact commands "
        "from the README."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
