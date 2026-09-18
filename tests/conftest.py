import difflib
import json
import os
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from undistortme import denoise as dn
from undistortme import pipeline as cp


@pytest.fixture
def check_dict(monkeypatch):
    """Install a fresh all-False gate dict as the module-global corr_pipeline.check_dict.

    The real dict is created only inside main(), so the module attribute does not
    exist at import time.  monkeypatch.setattr with raising=False creates it and
    automatically deletes it again on teardown.
    """
    d = {
        "dcm2niix": False,
        "topup": False,
        "topup_multithread": False,
        "slice": False,
        "match": False,
        "dryrun": False,
        "two_echo": False,
        "mask": False,
        "denoise": None,
    }
    monkeypatch.setattr(cp, "check_dict", d, raising=False)
    return d


# ===========================================================================
# Shared data contracts
#
# _make_bids_run (legacy layout) and _make_bids_dwi_dataset (BIDS layout)
# must produce EQUIVALENT runs so that the snapshot-reuse tests can compare
# staged-BIDS command batches against the legacy goldens.  The pieces both
# builders share are defined once here so the contract is structural.
# ===========================================================================

# Sidecar fields common to every echo (per-echo fields - EchoNumber,
# EchoTime, PhaseEncodingDirection - are added by the builders).
SIDECAR_COMMON = {
    "TotalReadoutTime": 0.106487,
    "PulseSequenceName": "ep_seg_35",
    "SeriesDescription": "ME_GRE",
    "ImageType": ["ORIGINAL", "PRIMARY", "M"],
    "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
}

# Echo times assigned to echoes 1..4.
DEFAULT_TES = (0.07, 0.09, 0.11, 0.13)


def seed_for(echo, vol):
    """RNG seed for volume ``vol`` (1-based) of ``echo`` in both builders."""
    return echo * 100 + vol


def _seeded_data(seed, shape=(8, 8, 8)):
    """Deterministic float32 voxel data for ``seed``."""
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _bval_text(bvals):
    """The exact on-disk .bval encoding both builders write."""
    return " ".join(str(b) for b in bvals)


@pytest.fixture
def tiny_nii():
    """Return a factory that writes a real uncompressed NIfTI and returns the path str.

    Usage:
        path = tiny_nii(tmp_path / "img.nii")
        path = tiny_nii(tmp_path / "img.nii", shape=(4, 4, 4), fill=1.0)
        path = tiny_nii(tmp_path / "img.nii", seed=42)

    Default shape (8, 8, 8) is large enough for skimage SSIM's default win_size.
    """

    def _make(path, shape=(8, 8, 8), fill=None, seed=0):
        if fill is not None:
            data = np.full(shape, float(fill), dtype=np.float32)
        else:
            data = _seeded_data(seed, shape)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        return str(path)

    return _make


# ===========================================================================
# Orchestration-test harness (Task 5)
#
# These fixtures/helpers support tests/test_process_run_orchestration.py.  They
# let process_run run end-to-end WITHOUT executing any external binary: every
# shell command the pipeline would dispatch is captured at the three batch
# dispatchers (parallel_bash_commands / serial_bash_commands) and
# recorded instead of run.
# ===========================================================================

# The contrast-match subcommand baked into the pipeline's commands.
CMATCH_PATH = cp.CONTRASTMATCH_CMD

# The denoise subcommand; like CMATCH_PATH it embeds sys.executable, so it is
# always replaced by a placeholder before snapshotting.
DENOISE_PATH = cp.DENOISE_CMD


def _make_bids_run(
    tmp_path,
    tiny_nii,
    *,
    n_echoes=3,
    te=DEFAULT_TES,
    phase_dir="j-",
    bvals=None,
    n_avgs=1,
    subject="sub-01",
    session="ses-01",
    run="run-01",
    output_root=None,
    deriv_root=None,
):
    """Build a BIDS-ish run tree the pipeline can glob and pin.

    Layout produced under ``{output_root}/{subject}/{session}/{run}/``:

      * sidecar  ``{subject}_{session}_{run}_echo-{e}.json``  (e = 1..n_echoes)
      * volumes  ``{subject}_{session}_{run}_echo-{e}_{bb}.nii``
        with ``bb = str(i).zfill(len(str(N)))``, i = 1..N where
        N = len(bvals) when bvals is given else n_avgs.
      * (optional) ``{subject}_{session}_{run}_echo-{e}.bval`` containing the
        space-separated bvals (one identical .bval per echo).

    All .nii are real (tiny, 8x8x8, seeded) so that code paths which call
    ``nib.load`` (run_topup_diffusion_special / find_closest_volume_nmi) work.

    Returns a dict with output_dir, deriv_dir, subject, session, run, run_dir,
    tmp_path (all strings).
    """
    out = Path(output_root) if output_root else (tmp_path / "out")
    deriv = Path(deriv_root) if deriv_root else (tmp_path / "deriv")
    run_dir = out / subject / session / run
    run_dir.mkdir(parents=True, exist_ok=True)

    n_vols = len(bvals) if bvals is not None else n_avgs
    num_digits = len(str(n_vols))

    for e in range(1, n_echoes + 1):
        stem = f"{subject}_{session}_{run}_echo-{e}"
        data = {
            "EchoNumber": e,
            "EchoTime": te[e - 1],
            "PhaseEncodingDirection": phase_dir,
            **SIDECAR_COMMON,
        }
        (run_dir / f"{stem}.json").write_text(json.dumps(data))
        if bvals is not None:
            (run_dir / f"{stem}.bval").write_text(_bval_text(bvals))
        for i in range(1, n_vols + 1):
            bb = str(i).zfill(num_digits)
            tiny_nii(run_dir / f"{stem}_{bb}.nii", seed=seed_for(e, i))

    return {
        "output_dir": str(out),
        "deriv_dir": str(deriv),
        "subject": subject,
        "session": session,
        "run": run,
        "run_dir": str(run_dir),
        "tmp_path": str(tmp_path),
    }


@pytest.fixture
def bids_run(tmp_path, tiny_nii):
    """Factory building a BIDS run tree (see _make_bids_run for the contract)."""

    def _build(**kwargs):
        return _make_bids_run(tmp_path, tiny_nii, **kwargs)

    return _build


def _make_bids_dwi_dataset(
    tmp_path,
    *,
    session="01",
    n_echoes=3,
    n_vols=1,
    acq=None,
    parts=None,
    bvals=None,
    gz=True,
    phase_dirs=None,
    sidecar_level="leaf",
    include_echo_number=False,
    echo_in_acq=False,
):
    """Build a real (gzipped-4D) BIDS dwi dataset for the ingest adapter.

    Layout produced under ``{tmp_path}/bids/``::

        dataset_description.json
        sub-01/[ses-{session}/]dwi/
            sub-01[_ses-..][_acq-..]_run-01_echo-{e}[_part-..]_dwi.nii[.gz]
            + one JSON sidecar per image (or higher-level sidecars, see
              ``sidecar_level``) and optional .bval/.bvec per echo.

    Knobs:
      * ``session=None`` -> no session level (tests ses-01 synthesis).
      * ``acq`` -> extra acq- entity label (None omits it).
      * ``parts`` -> iterable of part labels (e.g. ("mag", "phase")); each
        echo is written once per part. None omits the part entity.
      * ``bvals`` -> list of b-values; sets n_vols = len(bvals) and writes
        .bval/.bvec companions per echo.
      * ``gz`` -> .nii.gz (default) vs plain .nii.
      * ``phase_dirs`` -> per-echo PhaseEncodingDirection strings; the
        default alternates sign with echo parity ("j", "j-", "j", ...),
        matching the pipeline's blip-sign assumption.
      * ``sidecar_level`` -> "leaf" (one full sidecar next to each image),
        "sub" (per-echo sidecars at the subject level), or "split" (common
        fields in a root-level dwi.json, per-echo fields at the leaf).
      * ``include_echo_number`` -> whether sidecars carry EchoNumber
        (default False: tests the adapter's EchoNumber injection).
      * ``echo_in_acq`` -> encode the echo as ``acq-e{n}`` instead of an
        ``echo-`` entity (for --echo-from-acq tests); overrides ``acq``.

    Volume ``i`` of echo ``e`` uses ``seed_for(e, i)`` - the same seeds
    ``_make_bids_run`` gives ``tiny_nii`` - so command batches from staged
    data can be compared against the legacy-layout goldens.

    Returns a dict with bids_dir, subject/session tokens (session synthesis
    NOT applied: "session" is None when the dataset has no session level),
    dwi_dir, n_vols and the list of image paths.
    """
    if bvals is not None:
        n_vols = len(bvals)

    bids_dir = tmp_path / "bids"
    sub_tok = "sub-01"
    ses_tok = f"ses-{session}" if session is not None else None
    dwi_dir = bids_dir / sub_tok / (ses_tok or "") / "dwi"
    dwi_dir.mkdir(parents=True, exist_ok=True)
    (bids_dir / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "undistortme test dataset",
                "BIDSVersion": "1.9.0",
            }
        )
    )

    if phase_dirs is None:
        phase_dirs = ["j" if e % 2 == 1 else "j-" for e in range(1, n_echoes + 1)]

    if sidecar_level == "split":
        (bids_dir / "dwi.json").write_text(json.dumps(SIDECAR_COMMON))

    image_paths = []
    for e in range(1, n_echoes + 1):
        chunks = [sub_tok]
        if ses_tok:
            chunks.append(ses_tok)
        if echo_in_acq:
            chunks.append(f"acq-e{e}")
        elif acq is not None:
            chunks.append(f"acq-{acq}")
        chunks.append("run-01")
        if not echo_in_acq:
            chunks.append(f"echo-{e}")

        sidecar = {
            "EchoTime": DEFAULT_TES[e - 1],
            "PhaseEncodingDirection": phase_dirs[e - 1],
        }
        if sidecar_level != "split":
            sidecar.update(SIDECAR_COMMON)
        if include_echo_number:
            sidecar["EchoNumber"] = e

        data = np.stack(
            [_seeded_data(seed_for(e, i)) for i in range(1, n_vols + 1)], axis=-1
        )

        for part in parts if parts is not None else [None]:
            img_chunks = list(chunks)
            if part is not None:
                img_chunks.append(f"part-{part}")
            stem = "_".join(img_chunks) + "_dwi"
            ext = ".nii.gz" if gz else ".nii"
            img_path = dwi_dir / f"{stem}{ext}"
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(img_path))
            image_paths.append(str(img_path))

            if sidecar_level == "sub":
                # subject-level sidecar: only entities shared with the
                # image (sub + echo), so it applies via inheritance
                name = f"{sub_tok}_echo-{e}_dwi.json"
                (bids_dir / sub_tok / name).write_text(json.dumps(sidecar))
            else:
                (dwi_dir / f"{stem}.json").write_text(json.dumps(sidecar))

            if bvals is not None:
                (dwi_dir / f"{stem}.bval").write_text(_bval_text(bvals))
                (dwi_dir / f"{stem}.bvec").write_text(
                    "\n".join(" ".join("0" for _ in bvals) for _ in range(3))
                )

    return {
        "bids_dir": str(bids_dir),
        "subject": sub_tok,
        "session": ses_tok,
        "dwi_dir": str(dwi_dir),
        "n_vols": n_vols,
        "n_echoes": n_echoes,
        "image_paths": image_paths,
        "tmp_path": str(tmp_path),
    }


@pytest.fixture
def bids_dwi_dataset(tmp_path):
    """Factory building a real BIDS dwi dataset (see _make_bids_dwi_dataset)."""

    def _build(**kwargs):
        return _make_bids_dwi_dataset(tmp_path, **kwargs)

    return _build


def _clean(commands):
    """Match dispatcher behavior: drop None entries from a command list."""
    if commands is None:
        return []
    return [c for c in commands if c is not None]


def _install_recorder(monkeypatch, record_fn):
    """Patch the three batch dispatchers + shuffle to use ``record_fn``.

    All three dispatchers share the signature ``(bash_commands, description)``.
    ``shuffle`` is no-op'd so work lists keep deterministic (source) order for
    stable snapshots.  ``record_fn`` receives ``(bash_commands, description)``
    and is responsible for building/appending to whatever call log it owns.
    """
    monkeypatch.setattr(cp, "parallel_bash_commands", record_fn)
    monkeypatch.setattr(cp, "serial_bash_commands", record_fn)
    monkeypatch.setattr(cp, "shuffle", lambda seq: None)


@pytest.fixture
def recorder(monkeypatch):
    """Capture every dispatched batch instead of executing it.

    Monkeypatches the batch dispatchers on the pipeline module so
    that each call appends ``(description, [non-None commands])`` to the
    returned list, executing nothing.  Also no-ops ``cp.shuffle`` so that work
    lists keep deterministic (source) order for stable snapshots.

    All three dispatchers share the signature ``(bash_commands, description)``.
    """
    calls = []

    def _record(bash_commands, description):
        calls.append((description, _clean(bash_commands)))

    _install_recorder(monkeypatch, _record)
    return calls


@pytest.fixture
def slicing_recorder(monkeypatch, tiny_nii):
    """Like ``recorder`` but also fakes the on-disk outputs the pipeline reads.

    The whole-volume paths never touch disk for their outputs, but two batches
    have outputs that LATER code globs/reads, so we must materialise them:

      * slicenii batch (``slicenii -i {nii} -o {slice_dir} -p 6``): the real
        slicenii would write
        ``{slice_dir}/{base}_slices/{base}_axis-2_slice-padded-{NNN}.nii``
        (base = basename(nii) without extension, NNN 3-digit 1-based).
        handle_slicing then globs ``{slice_dir}/{base}_slices/{base}_*`` and
        parses the slice number from the filename, so we write 3 such slices
        per input volume (real tiny niftis).
      * masking batch (``fslmaths {nii} -mas {mask} {out}``): process_run then
        rewrites run_df["nii"] to the masked outputs.  We write each ``{out}``
        (the last token) as a real tiny nifti so downstream steps can proceed.

    Parsing is intentionally literal (token splitting on the exact command
    shapes above); anything else is recorded but not faked.
    """
    calls = []

    def _fake_outputs(cmd):
        tokens = cmd.split()
        if cmd.startswith("slicenii "):
            nii_path = tokens[tokens.index("-i") + 1]
            slice_dir = tokens[tokens.index("-o") + 1]
            base = os.path.basename(nii_path).split(".")[0]
            out_dir = Path(slice_dir) / f"{base}_slices"
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in range(1, 4):
                fname = f"{base}_axis-2_slice-padded-{str(n).zfill(3)}.nii"
                tiny_nii(out_dir / fname, seed=1000 + n)
        elif " -m undistortme.denoise " in cmd:
            # "... -m {method} -i {nii...} -o {out_dir} --suffix {suffix}
            #  [-b {bvals...}]": the real denoiser writes one output per input,
            # named by denoise.output_path, and handle_slicing later slices
            # those files, so they must exist on disk.
            out_dir = tokens[tokens.index("-o") + 1]
            suffix = tokens[tokens.index("--suffix") + 1]
            i_start = tokens.index("-i") + 1
            i_end = tokens.index("-o")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for nii_path in tokens[i_start:i_end]:
                tiny_nii(dn.output_path(nii_path, out_dir, suffix), seed=3000)
        elif cmd.startswith("fslmaths ") and " -mas " in cmd:
            # tokens[-1] is assumed to be the output path, mirroring the current
            # "fslmaths {nii} -mas {mask} {out}" command shape.  If the masking
            # command shape changes (extra flags, reordered args, etc.) this
            # assumption will silently miss the real output — revisit then.
            out_path = tokens[-1]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            tiny_nii(out_path, seed=2000)

    def _record(bash_commands, description):
        cmds = _clean(bash_commands)
        for cmd in cmds:
            _fake_outputs(cmd)
        calls.append((description, cmds))

    _install_recorder(monkeypatch, _record)
    return calls


def _normalize(calls, replacements):
    """Render captured (description, commands) batches to snapshot text.

    For each batch emit ``## {description}`` then one command per line, in the
    captured order (NOT sorted: order is characterized behavior).  Path
    substitutions are applied longest-find-first so that nested roots (deriv /
    output dirs under tmp_path) win over their tmp_path prefix.  The hardcoded
    contrastmatch.py path is always mapped to ``<CMATCH>``, and a
    ``--nthr=<N>`` backstop guards against any leaked cpu-count.

    Compound commands joined by ``" && "`` are split so that each subcommand
    appears on its own continuation line (indented 4 spaces after the first),
    i.e. ``" && "`` is replaced with ``" &&\\n    "``.  This makes line-granular
    diffs point at the exact subcommand that changed rather than a single
    multi-kilobyte line.

    ``replacements`` is an iterable of ``(find, placeholder)`` pairs.
    """
    reps = list(replacements) + [(CMATCH_PATH, "<CMATCH>"), (DENOISE_PATH, "<DENOISE>")]
    reps.sort(key=lambda kv: len(kv[0]), reverse=True)

    lines = []
    for description, commands in calls:
        lines.append(f"## {description}")
        for cmd in commands:
            s = cmd
            for find, placeholder in reps:
                s = s.replace(find, placeholder)
            s = re.sub(r"--nthr=\d+", "--nthr=<N>", s)
            s = s.replace(" && ", " &&\n    ")
            lines.append(s)
    return "\n".join(lines) + "\n"


@pytest.fixture
def normalize():
    """Return the ``_normalize(calls, replacements)`` snapshot renderer."""
    return _normalize


def _reps(info):
    """Path replacements for normalize(): longest-find-first handled inside."""
    return [
        (info["deriv_dir"], "<DERIV>"),
        (info["output_dir"], "<OUT>"),
        (info["tmp_path"], "<TMP>"),
    ]


@pytest.fixture
def reps():
    """Return the ``_reps(info)`` path-placeholder builder."""
    return _reps


def _run_process_run(info, mask_dir=None, cutoff=1000, config="b02b0.cnf"):
    """Call process_run with the positional arguments a bids_run dict implies."""
    cp.process_run(
        info["subject"],
        info["session"],
        info["run"],
        info["output_dir"],
        info["deriv_dir"],
        config,
        mask_dir,
        cutoff,
    )


@pytest.fixture
def run_process_run():
    """Return the ``_run_process_run(info, ...)`` driver."""
    return _run_process_run


_SNAPSHOT_DIR = Path(__file__).parent / "_snapshots"


def _assert_snapshot(name, text):
    """Compare ``text`` against the committed golden snapshot ``{name}.txt``.

    With env ``UPDATE_SNAPSHOTS`` set, (over)write the golden and skip.  Without
    it, fail with a unified diff on mismatch (or if the golden is missing).
    """
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = _SNAPSHOT_DIR / f"{name}.txt"

    if os.environ.get("UPDATE_SNAPSHOTS", "").lower() in {"1", "true", "yes"}:
        path.write_text(text)
        pytest.skip(f"snapshot written: {path.name}")

    if not path.exists():
        pytest.fail(
            f"Snapshot {path.name} missing. Regenerate with UPDATE_SNAPSHOTS=1."
        )

    expected = path.read_text()
    if text != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"golden/{path.name}",
                tofile="actual",
            )
        )
        pytest.fail(f"Snapshot mismatch for {path.name}:\n{diff}")


@pytest.fixture
def assert_snapshot():
    """Return the ``_assert_snapshot(name, text)`` comparator."""
    return _assert_snapshot
