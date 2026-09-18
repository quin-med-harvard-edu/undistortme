#!/usr/bin/env python
# coding: utf-8
# Liam Timms 2026
# BIDS ingest adapter: stage a BIDS dwi dataset into the per-volume 3D
# layout the undistortme pipeline expects.

"""Stage BIDS-organized dwi data into the undistortme input layout.

The pipeline consumes ``{root}/sub-*/ses-*/run-*/`` directories holding one
3D ``.nii`` per volume plus one JSON sidecar (and optional ``.bval``) per
echo (see README "Input layout"). BIDS datasets instead hold 4D
``sub-*/[ses-*/]dwi/*_dwi.nii[.gz]`` files. This module bridges the two:

1. index ``{bids_dir}/sub-*/[ses-*/]dwi/*_dwi.nii[.gz]``, parsing BIDS
   entities from the filenames;
2. resolve JSON / ``.bval`` / ``.bvec`` sidecars via the BIDS inheritance
   principle (walk up ``dwi/`` -> ``ses-*/`` -> ``sub-*/`` -> root);
3. group files into runs by every entity except the echo entity, and
   synthesize a ``run-``-prefixed run token per group (hard error on
   collisions);
4. split each echo's 4D image into per-volume 3D plain ``.nii`` files under
   ``{staging_root}/{sub}/{ses}/{run-token}/`` named exactly as the legacy
   layout expects, copying sidecars (injecting ``EchoNumber``) and bvals.

Multi-echo DWI is NOT valid BIDS (``echo`` is not an accepted entity for the
``_dwi`` suffix), so real datasets may encode echoes in ``acq-`` labels; the
echo source is configurable (``echo_entity`` / ``echo_from_acq``).

Staging is idempotent: files already present in the staging tree are kept.
The splitter loads the whole 4-D image once (per-volume reads would restart
gzip decompression each time), so peak memory is the full series.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from functools import lru_cache

import nibabel as nib
import numpy as np

# Entities folded into the synthesized run token, in canonical BIDS order.
# Excluded on purpose: sub/ses (directory levels), run (leads the token),
# echo (varies within a run), part (only part-mag is staged by default, so
# keeping it would only create collision-prone noise).
TOKEN_ENTITY_ORDER = [
    "task",
    "acq",
    "ce",
    "rec",
    "dir",
    "mod",
    "flip",
    "inv",
    "mt",
    "chunk",
]

# Known sidecar/companion extensions, longest first so .nii.gz wins over .gz.
KNOWN_EXTENSIONS = [".nii.gz", ".nii", ".json", ".bval", ".bvec"]

# Where dwi images may live relative to the BIDS root, and the directory
# staged runs are written into under the work root.
DWI_DIR_GLOBS = (os.path.join("sub-*", "dwi"), os.path.join("sub-*", "ses-*", "dwi"))
DEFAULT_STAGING_SUBDIR = "bids-staged"


def volume_index_token(i: int, n_vols: int) -> str:
    """1-based volume index zero-padded to the width of N (legacy naming)."""
    return str(i).zfill(len(str(n_vols)))


def volume_filename(stem: str, i: int, n_vols: int) -> str:
    """Per-volume 3D filename for volume ``i`` of ``n_vols``: the
    ``{stem}_{bb}.nii`` convention shared by dcm2niix output, the staging
    tree, and get_echo_info's lookup."""
    return f"{stem}_{volume_index_token(i, n_vols)}.nii"


def expected_phase_sign(echo_num: int) -> int:
    """Blip sign the pipeline assumes for an echo: alternating with echo
    parity, echo 1 positive (sidecar PE signs are not consulted)."""
    return (-1) ** (echo_num + 1)


def parse_entities(path: str) -> tuple[dict[str, str], str, str]:
    """Parse BIDS entities from a filename.

    Returns ``(entities, suffix, extension)`` where entities maps entity key
    to label (e.g. ``{"sub": "01", "echo": "2"}``), suffix is the trailing
    dash-less chunk (e.g. ``dwi``, possibly ``""``), and extension is the
    full extension including ``.nii.gz``.
    """
    base = os.path.basename(path)
    ext = ""
    for known in KNOWN_EXTENSIONS:
        if base.endswith(known):
            ext = known
            break
    else:
        ext = os.path.splitext(base)[1]
    stem = base[: -len(ext)] if ext else base

    entities: dict[str, str] = {}
    suffix = ""
    for chunk in stem.split("_"):
        if "-" in chunk:
            key, _, value = chunk.partition("-")
            entities[key] = value
        else:
            # a dash-less chunk is the suffix (last one wins)
            suffix = chunk
    return (entities, suffix, ext)


def _levels_up_to_root(nii_path: str, bids_dir: str) -> list[str]:
    """Directories from the BIDS root down to the data file's directory.

    ``nii_path`` always comes from globbing under ``bids_dir``, so this is
    a simple relative-path expansion.
    """
    root = os.path.normpath(os.path.abspath(bids_dir))
    file_dir = os.path.normpath(os.path.abspath(os.path.dirname(nii_path)))
    levels = [root]
    rel = os.path.relpath(file_dir, root)
    if rel != ".":
        for part in rel.split(os.sep):
            levels.append(os.path.join(levels[-1], part))
    return levels


def _applies_to(
    candidate_entities: dict[str, str], data_entities: dict[str, str]
) -> bool:
    """BIDS inheritance rule: candidate entities must be a subset of the
    data file's entities (same key AND same label)."""
    return all(
        data_entities.get(key) == value for key, value in candidate_entities.items()
    )


@lru_cache(maxsize=None)
def _dir_index(directory: str, extension: str) -> tuple[tuple[dict, str], ...]:
    """Parsed ``(entities, path)`` pairs for the dwi-suffixed files with
    ``extension`` in ``directory``, in sorted path order.

    Cached for the process lifetime: staging scans the same directories
    once per echo and per companion type, and assumes the dataset does not
    change while the process runs.
    """
    index = []
    for path in sorted(glob.glob(os.path.join(directory, f"*{extension}"))):
        entities, suffix, _ext = parse_entities(path)
        if suffix == "dwi":
            index.append((entities, path))
    return tuple(index)


def _applicable_at_level(
    level: str, extension: str, data_entities: dict[str, str]
) -> list[tuple[int, str]]:
    """Inheritance candidates at one directory level as sorted
    ``(specificity, path)`` tuples (least specific first)."""
    return sorted(
        (len(entities), path)
        for entities, path in _dir_index(level, extension)
        if _applies_to(entities, data_entities)
    )


def resolve_sidecar_json(
    nii_path: str, data_entities: dict[str, str], bids_dir: str
) -> dict:
    """Merge all applicable JSON sidecars per the BIDS inheritance principle.

    Walks from the dataset root down to the data file's directory; at each
    level every applicable ``*_dwi.json`` (entities a subset of the data
    file's, least-specific first) is merged in, so deeper/more-specific
    values override shallower ones.
    """
    merged: dict = {}
    for level in _levels_up_to_root(nii_path, bids_dir):
        for _specificity, json_path in _applicable_at_level(
            level, ".json", data_entities
        ):
            try:
                with open(json_path, "r") as f:
                    merged.update(json.load(f))
            except (OSError, json.JSONDecodeError) as e:
                print(f"WARNING: could not read sidecar {json_path}: {e}")
    return merged


def resolve_companion_file(
    nii_path: str, data_entities: dict[str, str], bids_dir: str, extension: str
) -> str | None:
    """Find the most specific applicable ``.bval``/``.bvec`` file.

    Unlike JSON sidecars these are not merged: the nearest (deepest) level
    wins, and within a level the candidate with the most entities wins.
    """
    for level in reversed(_levels_up_to_root(nii_path, bids_dir)):
        candidates = _applicable_at_level(level, extension, data_entities)
        if candidates:
            return candidates[-1][1]
    return None


def index_bids_dwi(
    bids_dir: str, echo_entity: str = "echo", echo_from_acq: str | None = None
) -> list[dict]:
    """Index every dwi image in a BIDS dataset.

    Returns one record dict per staged-eligible file with keys: ``path``,
    ``entities``, ``echo``, ``sidecar`` (merged JSON dict), ``bval_path``,
    ``bvec_path``, ``echo_from_acq_used``.

    Only magnitude images are kept: files carrying a ``part-`` entity other
    than ``part-mag`` are skipped (phase/real/imag parts would break the
    intensity-based pipeline steps).
    """
    nii_files: list[str] = []
    for dwi_glob in DWI_DIR_GLOBS:
        for ext in (".nii", ".nii.gz"):
            nii_files.extend(glob.glob(os.path.join(bids_dir, dwi_glob, f"*_dwi{ext}")))
    nii_files = sorted(set(nii_files))

    acq_pattern = None
    if echo_from_acq is not None:
        acq_pattern = re.compile(echo_from_acq)
        if acq_pattern.groups < 1:
            raise ValueError(
                f"--echo-from-acq pattern '{echo_from_acq}' must capture "
                "the echo number as group 1"
            )

    records: list[dict] = []
    for path in nii_files:
        entities, suffix, _ext = parse_entities(path)
        if "sub" not in entities:
            print(f"WARNING: no sub- entity in {path}, skipping")
            continue
        part = entities.get("part")
        if part is not None and part != "mag":
            print(f"Skipping non-magnitude image (part-{part}): {path}")
            continue

        sidecar = resolve_sidecar_json(path, entities, bids_dir)
        bval_path = resolve_companion_file(path, entities, bids_dir, ".bval")
        bvec_path = resolve_companion_file(path, entities, bids_dir, ".bvec")

        # echo number: the acq regex first, then the configurable entity,
        # then the sidecar's EchoNumber, else 1
        echo: int | None = None
        echo_from_acq_used = False
        if acq_pattern is not None and "acq" in entities:
            m = acq_pattern.search(entities["acq"])
            if m is not None:
                echo = int(m.group(1))
                echo_from_acq_used = True
        if echo is None and echo_entity in entities:
            try:
                echo = int(entities[echo_entity])
            except ValueError:
                print(
                    f"WARNING: non-integer {echo_entity}- label in "
                    f"{path}, falling back to sidecar EchoNumber"
                )
        if echo is None:
            echo = int(sidecar.get("EchoNumber", 1))

        records.append(
            {
                "path": path,
                "entities": entities,
                "echo": echo,
                "sidecar": sidecar,
                "bval_path": bval_path,
                "bvec_path": bvec_path,
                "echo_from_acq_used": echo_from_acq_used,
            }
        )
    return records


def make_run_token(entities: dict[str, str]) -> str:
    """Synthesize a filesystem-safe, ``run-``-prefixed run token.

    The token leads with ``run-{label}`` (``run-01`` when the dataset has
    no run entity) so the legacy ``run-*`` glob in main() still finds the
    staged directories, then appends the remaining grouping entities in
    BIDS canonical order (e.g. ``run-01_acq-highres_dir-AP``).
    """
    parts = [f"run-{entities.get('run', '01')}"]
    for key in TOKEN_ENTITY_ORDER:
        if key in entities:
            parts.append(f"{key}-{entities[key]}")
    # deterministic fallback for entities outside the canonical table
    known = set(TOKEN_ENTITY_ORDER) | {"sub", "ses", "run", "part", "echo"}
    for key in sorted(set(entities) - known):
        parts.append(f"{key}-{entities[key]}")
    return "_".join(parts)


def group_runs(
    records: list[dict], echo_entity: str = "echo"
) -> dict[tuple[str, str, str], list[dict]]:
    """Group indexed files into runs keyed by ``(sub, ses, run_token)``.

    The grouping key is every entity except the echo entity (and acq when
    the echo was extracted from it). Sessions are synthesized as ``ses-01``
    when the dataset has no session level. Distinct groups mapping to the
    same run token (e.g. differing only in a dropped ``part-`` entity) are
    a hard error, as are duplicate echo numbers within one group.
    """
    token_sources: dict[tuple[str, str, str], tuple] = {}
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for rec in records:
        entities = dict(rec["entities"])
        subject = f"sub-{entities.pop('sub')}"
        session = f"ses-{entities.pop('ses')}" if "ses" in entities else "ses-01"
        entities.pop(echo_entity, None)
        if rec["echo_from_acq_used"]:
            entities.pop("acq", None)
        group_id = (subject, session, tuple(sorted(entities.items())))

        token = make_run_token(entities)
        key = (subject, session, token)
        if key in token_sources and token_sources[key] != group_id:
            raise RuntimeError(
                f"Run-token collision: groups {token_sources[key]} and "
                f"{group_id} both map to {subject}/{session}/{token}. "
                "Rename the input files so the run token is unambiguous."
            )
        token_sources[key] = group_id

        groups.setdefault(key, []).append(rec)

    for (subject, session, token), recs in groups.items():
        echoes = [r["echo"] for r in recs]
        if len(echoes) != len(set(echoes)):
            raise RuntimeError(
                f"Duplicate echo numbers {sorted(echoes)} in "
                f"{subject}/{session}/{token}: check the echo source "
                "(--echo-entity / --echo-from-acq) and the sidecars' "
                "EchoNumber fields."
            )
    return groups


def warn_on_pe_signs(
    subject: str, session: str, token: str, records: list[dict]
) -> None:
    """Warn loudly when sidecar PE signs contradict the echo-parity model.

    The pipeline derives the blip sign from echo parity (echo 1 positive,
    echo 2 negative, ...) and ignores the sign in the sidecar's
    PhaseEncodingDirection because dcm2niix signs are unreliable. Curated
    BIDS datasets may carry meaningful signs, so surface any disagreement.
    """
    # TODO: add --pe-from-sidecar to trust sidecar signs instead of parity.
    pe_by_echo: dict[int, str] = {}
    for rec in records:
        pe = rec["sidecar"].get("PhaseEncodingDirection")
        if isinstance(pe, str) and pe and pe != "undefined":
            pe_by_echo[rec["echo"]] = pe
    if len(pe_by_echo) < 2:
        return

    axes = {pe[0] for pe in pe_by_echo.values()}
    if len(axes) > 1:
        print(
            f"WARNING: {subject} {session} {token}: echoes report "
            f"different phase-encoding AXES {sorted(axes)}; TOPUP "
            "correction across axes is not supported."
        )

    mismatched = {
        echo: pe
        for echo, pe in pe_by_echo.items()
        if (-1 if pe.endswith("-") else 1) != expected_phase_sign(echo)
    }
    if mismatched:
        print("=" * 70)
        print(
            f"WARNING: {subject} {session} {token}: sidecar "
            "PhaseEncodingDirection signs do not alternate with echo "
            "parity as the pipeline assumes (echo 1 positive, echo 2 "
            "negative, ...):"
        )
        for echo, pe in sorted(pe_by_echo.items()):
            flag = "  <-- MISMATCH" if echo in mismatched else ""
            print(f"    echo {echo}: PhaseEncodingDirection={pe}{flag}")
        print(
            "undistortme IGNORES sidecar signs and derives blip signs "
            "from echo parity; if these sidecar signs are correct the "
            "correction may be wrong. Verify the acquisition scheme."
        )
        print("=" * 70)


def _save_atomic(image: nib.Nifti1Image, out_path: str) -> None:
    """Write via a temporary name so a kill cannot leave a truncated file that
    the next run's resume-by-existence would accept as staged."""
    tmp_path = out_path + ".tmp.nii"
    nib.save(image, tmp_path)
    os.replace(tmp_path, out_path)


def split_echo_volumes(
    nii_path: str, out_dir: str, stem: str, expected_vols: int | None = None
) -> list[str]:
    """Split a (possibly gzipped) 3D/4D image into per-volume plain .nii.

    Output naming follows the legacy convention (see volume_filename).
    ``nib.load`` only reads the header, so when every output already
    exists (idempotent resume) no voxel data is touched. Otherwise the
    image is decompressed ONCE into memory (peak: one 4D array) and
    sliced per volume — per-volume ``dataobj[..., k]`` reads would restart
    the gzip stream from byte 0 for every volume, making staging O(N^2)
    in decompression.
    """
    img = nib.load(nii_path)
    shape = img.shape
    n_vols = shape[3] if len(shape) >= 4 else 1
    if expected_vols is not None and expected_vols != n_vols:
        raise ValueError(
            f"{nii_path}: .bval lists {expected_vols} volumes but the "
            f"image has {n_vols}"
        )

    out_paths = [
        os.path.join(out_dir, volume_filename(stem, i, n_vols))
        for i in range(1, n_vols + 1)
    ]
    missing = [
        i for i, path in enumerate(out_paths, start=1) if not os.path.exists(path)
    ]
    if len(missing) > 0:
        data = np.asanyarray(img.dataobj)
        for i in missing:
            vol = data[..., i - 1] if len(shape) >= 4 else data
            _save_atomic(nib.Nifti1Image(vol, img.affine, img.header), out_paths[i - 1])
    return out_paths


def _copy_if_absent(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` unless it is already staged (idempotency)."""
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def stage_bids(
    bids_dir: str,
    staging_root: str,
    echo_entity: str = "echo",
    echo_from_acq: str | None = None,
) -> list[tuple[str, str, str]]:
    """Stage a BIDS dwi dataset into ``staging_root``.

    Returns the sorted list of ``(subject, session, run_token)`` triples
    staged (empty when no dwi images are found — the caller owns the
    error message); ``{staging_root}/{subject}/{session}/{run_token}/``
    then holds the per-volume 3D layout process_run expects. Raises
    FileNotFoundError when ``bids_dir`` does not exist.
    """
    if not os.path.isdir(bids_dir):
        raise FileNotFoundError(f"BIDS directory {bids_dir} does not exist")

    records = index_bids_dwi(bids_dir, echo_entity, echo_from_acq)
    if len(records) == 0:
        return []

    groups = group_runs(records, echo_entity)
    staged: list[tuple[str, str, str]] = []
    for subject, session, token in sorted(groups):
        recs = sorted(groups[(subject, session, token)], key=lambda r: r["echo"])
        warn_on_pe_signs(subject, session, token, recs)
        run_dir = os.path.join(staging_root, subject, session, token)
        os.makedirs(run_dir, exist_ok=True)

        for rec in recs:
            echo = rec["echo"]
            stem = f"{subject}_{session}_{token}_echo-{echo}"

            expected_vols = None
            if rec["bval_path"] is not None:
                with open(rec["bval_path"], "r") as f:
                    expected_vols = len(f.read().split())

            nii_paths = split_echo_volumes(rec["path"], run_dir, stem, expected_vols)

            sidecar_out = os.path.join(run_dir, f"{stem}.json")
            if not os.path.exists(sidecar_out):
                sidecar = dict(rec["sidecar"])
                existing = sidecar.get("EchoNumber")
                if existing is None:
                    sidecar["EchoNumber"] = echo
                elif int(existing) != echo:
                    print(
                        f"WARNING: sidecar EchoNumber={existing} "
                        f"disagrees with parsed echo {echo} for "
                        f"{rec['path']}; using {echo}"
                    )
                    sidecar["EchoNumber"] = echo
                with open(sidecar_out, "w") as f:
                    json.dump(sidecar, f, indent=2)

            if rec["bval_path"] is not None:
                _copy_if_absent(rec["bval_path"], os.path.join(run_dir, f"{stem}.bval"))

            # the pipeline never reads bvecs, but carry them along so they
            # are available next to the staged data (and for future
            # BIDS-derivatives export)
            if rec["bvec_path"] is not None:
                _copy_if_absent(rec["bvec_path"], os.path.join(run_dir, f"{stem}.bvec"))

            print(f"Staged {stem}: {len(nii_paths)} volume(s) in {run_dir}")

        staged.append((subject, session, token))
    print(f"Staged {len(staged)} run(s) from {bids_dir} into {staging_root}")
    return staged


def looks_like_bids(root: str) -> bool:
    """Heuristic: does ``root`` look like a BIDS dataset rather than the
    legacy run-directory layout? Used only for a printed hint."""
    has_dwi_dir = any(
        glob.glob(os.path.join(root, dwi_glob)) for dwi_glob in DWI_DIR_GLOBS
    )
    run_dirs = glob.glob(os.path.join(root, "sub-*", "ses-*", "run-*"))
    return has_dwi_dir and len(run_dirs) == 0


def add_echo_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the echo-source options shared by undistortme --bids and
    undistortme-bidsify (single definition, no drifting help text)."""
    parser.add_argument(
        "--echo-entity",
        dest="echo_entity",
        default="echo",
        help="BIDS filename entity carrying the echo number when reading "
        "BIDS input (default: echo, i.e. echo-1, echo-2, ...)",
    )
    parser.add_argument(
        "--echo-from-acq",
        dest="echo_from_acq",
        default=None,
        help="regex applied to the BIDS acq- label to extract the echo "
        r"number as capture group 1 (e.g. 'e(\d+)$' for acq-e1/acq-e2); "
        "when it matches, the acq entity is dropped from run grouping",
    )


def get_args() -> argparse.Namespace:
    """Parse command line arguments for the standalone staging tool."""
    parser = argparse.ArgumentParser(
        description="Stage a BIDS dwi dataset into the per-volume 3D input "
        "layout expected by undistortme (the same conversion `undistortme "
        "--bids` performs before processing)."
    )
    parser.add_argument(
        "bids_dir",
        help="root of the BIDS dataset (sub-*/[ses-*/]dwi/*_dwi.nii[.gz])",
    )
    parser.add_argument(
        "staging_dir",
        help="directory to write the staged tree into (it becomes the "
        "input root, i.e. undistortme's -o/--output_dir)",
    )
    add_echo_source_args(parser)
    return parser.parse_args()


def main() -> None:
    """Entry point for the undistortme-bidsify console script."""
    args = get_args()
    try:
        staged = stage_bids(
            args.bids_dir, args.staging_dir, args.echo_entity, args.echo_from_acq
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if len(staged) == 0:
        print(
            f"ERROR: no BIDS dwi runs found under {args.bids_dir} "
            "(expected sub-*/[ses-*/]dwi/*_dwi.nii[.gz])"
        )
        sys.exit(1)
    print(
        f"\nRun the pipeline on the staged tree with e.g.:\n"
        f"  undistortme -t -o {args.staging_dir}"
    )


if __name__ == "__main__":
    main()
