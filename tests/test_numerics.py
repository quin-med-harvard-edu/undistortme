"""Characterization tests for image-similarity functions in corr_pipeline.py.

These tests pin the CURRENT behavior exactly as-is — ahead of a refactor.
Do NOT change expected behavior to match a "corrected" implementation;
the source wins.

Source behaviors confirmed before writing:
- find_closest_volume (L777):
    * Computes SSIM between img and each comparison volume using
      skimage.metrics.structural_similarity with data_range computed from
      the global max-min across all inputs and win_size=5.
    * Returns np.argmax(ssim_list) — the index of the MOST similar volume.
    * 8x8x8 arrays work with win_size=5 (each dimension ≥ win_size).

- find_closest_volume_nmi (L793):
    * Computes skimage.metrics.normalized_mutual_information between img and
      each comparison volume.
    * Returns np.argmax(nmi_list) — the index of the MOST mutually-informative
      (i.e. most similar) volume.
"""

import numpy as np

from undistortme import pipeline as cp


# ===========================================================================
# find_closest_volume  (SSIM-based; returns argmax)
# ===========================================================================


class TestFindClosestVolume:
    """SSIM-based closest-volume finder; should return the index of the most
    similar volume (highest SSIM).
    """

    def test_identical_copy_is_found(self):
        """When one comparison volume IS the target, its index is returned.

        The identical copy has SSIM=1.0; all others are < 1.0 for random data.
        """
        rng = np.random.default_rng(0)
        img = rng.random((8, 8, 8), dtype=np.float32)

        # Build comparisons: [noise, noise, img_copy, noise]
        comparisons = [
            rng.random((8, 8, 8), dtype=np.float32),  # index 0
            rng.random((8, 8, 8), dtype=np.float32),  # index 1
            img.copy(),  # index 2  ← target
            rng.random((8, 8, 8), dtype=np.float32),  # index 3
        ]

        result = cp.find_closest_volume(img, comparisons)
        assert result == 2, (
            f"Expected index 2 (identical copy); got {result}. "
            "find_closest_volume returns np.argmax(ssim_list)."
        )

    def test_most_similar_volume_wins(self):
        """Among two noise arrays and a high-similarity variant, the variant wins."""
        rng = np.random.default_rng(42)
        img = rng.random((8, 8, 8), dtype=np.float32)

        # slight perturbation — much more similar to img than pure noise
        similar = img + rng.random((8, 8, 8), dtype=np.float32) * 0.01

        comparisons = [
            rng.random((8, 8, 8), dtype=np.float32),  # index 0 — unrelated
            similar,  # index 1 — very similar
        ]

        result = cp.find_closest_volume(img, comparisons)
        assert result == 1, (
            f"Expected index 1 (near-identical perturbed volume); got {result}."
        )

    def test_returns_argmax_not_value(self):
        """Return type is np.intp (an integer index), not the SSIM value itself."""
        rng = np.random.default_rng(7)
        img = rng.random((8, 8, 8), dtype=np.float32)
        comp = [img.copy()]

        result = cp.find_closest_volume(img, comp)
        assert isinstance(result, (int, np.integer)), (
            f"Expected integer index; got {type(result)}"
        )
        assert result == 0

    def test_first_of_two_identical_comparisons(self):
        """When two comparisons are equally similar (both identical), argmax
        returns the first (index 0) due to np.argmax tie-breaking behavior."""
        rng = np.random.default_rng(11)
        img = rng.random((8, 8, 8), dtype=np.float32)

        comparisons = [img.copy(), img.copy()]  # both identical

        result = cp.find_closest_volume(img, comparisons)
        # np.argmax returns the first occurrence on ties
        assert result == 0


# ===========================================================================
# find_closest_volume_nmi  (NMI-based; returns argmax = most similar)
# ===========================================================================


class TestFindClosestVolumeNmi:
    """NMI-based closest-volume finder: argmax = most similar volume."""

    def test_returns_most_similar_volume(self):
        """An identical copy beats an independent array.

        Setup:
          img     = seeded random array
          comp[0] = identical copy of img       → HIGH NMI  (most informative)
          comp[1] = independent random array    → LOW NMI   (least informative)
        """
        rng = np.random.default_rng(0)
        img = rng.random((8, 8, 8), dtype=np.float32)

        identical = img.copy()  # NMI should be max
        independent = rng.random((8, 8, 8), dtype=np.float32)  # NMI should be low

        comparisons = [identical, independent]

        result = cp.find_closest_volume_nmi(img, comparisons)

        assert result == 0, (
            f"expected index 0 (the identical, highest-NMI volume) but got {result}"
        )

    def test_most_similar_of_three_comparisons(self):
        """With three comparisons the function still returns the lowest-NMI index.

        Ordering: [identical, similar, independent]
        Expected argmax: index 0 (identical → highest NMI with the target).
        """
        rng = np.random.default_rng(5)
        img = rng.random((8, 8, 8), dtype=np.float32)

        comparisons = [
            img.copy(),  # index 0: identical  (high NMI)
            img
            + rng.random((8, 8, 8), dtype=np.float32)
            * 0.05,  # index 1: similar (medium NMI)
            rng.random((8, 8, 8), dtype=np.float32),  # index 2: unrelated  (low NMI)
        ]

        result = cp.find_closest_volume_nmi(img, comparisons)

        assert result == 0, f"Expected index 0 (identical, highest NMI); got {result}."

    def test_returns_integer_index(self):
        """Return value is a numpy integer index, not the NMI score."""
        rng = np.random.default_rng(99)
        img = rng.random((8, 8, 8), dtype=np.float32)
        comp = [img.copy()]

        result = cp.find_closest_volume_nmi(img, comp)
        assert isinstance(result, (int, np.integer)), (
            f"Expected integer index; got {type(result)}"
        )

    def test_single_comparison_returns_0(self):
        """With exactly one comparison the index must be 0 regardless."""
        rng = np.random.default_rng(13)
        img = rng.random((8, 8, 8), dtype=np.float32)
        comp = [rng.random((8, 8, 8), dtype=np.float32)]

        result = cp.find_closest_volume_nmi(img, comp)
        assert result == 0


# ===========================================================================
# find_closest_volume_nmi: data-driven selection with a wide margin
# ===========================================================================


def test_nmi_selection_unambiguous_margin():
    """Pin NMI selection on data where the choice is clear-cut.

    The orchestration snapshot pins this choice on two near-identical noise
    volumes (NMI gap < 0.5% — a flake risk across skimage versions). Here the
    candidates are constructed so the winner is separated by a wide margin,
    and the margin itself is asserted so the test fails loudly if the inputs
    ever stop discriminating.
    """
    rng = np.random.default_rng(7)
    base = rng.random((16, 16, 8))
    near_copy = base + rng.normal(0, 0.01, base.shape)  # high NMI vs base
    unrelated = rng.random((16, 16, 8))  # low NMI vs base

    from skimage.metrics import normalized_mutual_information as nmi

    nmi_copy = nmi(base, near_copy)
    nmi_unrel = nmi(base, unrelated)
    assert (nmi_copy - nmi_unrel) / nmi_unrel > 0.10, (
        "test inputs no longer discriminate; rebuild them"
    )

    idx = cp.find_closest_volume_nmi(base, [near_copy, unrelated])
    assert idx == 0  # the near-copy is the most similar volume
