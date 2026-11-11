"""The estimators' contract, checked against brute force.

A catalogue is k partitions of a box, of which m are available, and the
estimators return N-point statistics of the full box from those m:

* with m = k the estimate is exactly the full box's count;
* with N <= m < k it is unbiased: averaged over every choice of m of the k
  partitions it equals the full box's count (scale-up), or that count times
  (m / k)^N (scale-down), up to rounding;
* with m < N it is rejected, since tuples spanning more than m partitions
  could never be observed.

The full-box counts come from the brute-force references in reference.py,
which ignore partitions entirely, so these tests do not rely on the
counters' own split into partition classes.
"""

from functools import cache
from itertools import combinations

import numpy as np
import pytest
import reference as ref
from pysauce import compute_2pcf, compute_npcf, compute_xi, compute_xi_smu

BOX = 20.0
EDGES = np.array([0.3, 1.5, 3.0, 4.5])
PI_EDGES = np.array([0.0, 1.0, 2.5, 4.0])
N_MU, MU_MAX = 3, 1.0


@cache
def catalogue(k, uneven):
    """Clustered points in k partitions, equal in size on average or very
    unequal. The estimate is unbiased either way: it averages over which
    partitions are chosen, not over how points are assigned to them."""
    coords, _ = ref.clustered_catalogue(70, BOX, 1, seed=100 + k, n_clusters=6, spread=1.0)
    rng = np.random.default_rng(200 + k)
    p = np.arange(1, k + 1, dtype=float) ** (3 if uneven else 0)
    parts = rng.choice(k, len(coords), p=p / p.sum()).astype(np.int32)
    return coords, parts


def selections(parts, k, m):
    """Masks selecting each of the C(k, m) choices of m partitions."""
    for chosen in combinations(range(k), m):
        yield np.isin(parts, chosen)


@cache
def full_box_tuples(k, uneven, n_order):
    coords, parts = catalogue(k, uneven)
    return ref.tuples(coords, np.zeros_like(parts), EDGES, BOX, n_order).sum(axis=0)


def full_box_pairs(coords, statistic):
    """Pair counts of the whole catalogue with partitions ignored."""
    one = np.zeros(len(coords), np.int32)
    if statistic == "r":
        return ref.radial_pairs(coords, one, EDGES, BOX)[0]
    if statistic == "rp_pi":
        return ref.projected_pairs(coords, one, EDGES, PI_EDGES, BOX)[0]
    return ref.redshift_space_pairs(coords, one, EDGES, N_MU, MU_MAX, BOX)[0]


def pair_estimate(statistic, coords, parts, k, m):
    """The corrected pair counts and the power of m / k they are scaled by."""
    if statistic == "r":
        return compute_xi(coords, parts, EDGES, BOX, k, m)["dd_corr"], 2
    if statistic == "rp_pi":
        return compute_2pcf(coords, parts, EDGES, PI_EDGES, BOX, k, m)["dd_corr"], 0
    result = compute_xi_smu(
        coords,
        parts,
        EDGES,
        BOX,
        n_partitions=k,
        n_partitions_selected=m,
        n_mu_bins=N_MU,
        mu_max=MU_MAX,
    )
    return result["dd_corr"], 2


PAIR_STATISTICS = ["r", "rp_pi", "s_mu"]
SPLITS = [(k, uneven) for k in (2, 3, 4, 6) for uneven in (False, True)]
# Tuple orders N with every split that has k >= N partitions.
ORDERS_AND_SPLITS = [(n, k, uneven) for n in (2, 3, 4) for k, uneven in SPLITS if k >= n]


class TestFullSelection:
    @pytest.mark.parametrize("statistic", PAIR_STATISTICS)
    @pytest.mark.parametrize("k, uneven", SPLITS)
    def test_pairs_are_the_full_box_counts(self, statistic, k, uneven):
        coords, parts = catalogue(k, uneven)
        estimate, _ = pair_estimate(statistic, coords, parts, k, k)
        np.testing.assert_array_equal(estimate, full_box_pairs(coords, statistic))

    @pytest.mark.parametrize("scale_method", ["scale-up", "scale-down"])
    @pytest.mark.parametrize("n_order, k, uneven", ORDERS_AND_SPLITS)
    def test_tuples_are_the_full_box_counts(self, n_order, scale_method, k, uneven):
        coords, parts = catalogue(k, uneven)
        result = compute_npcf(coords, parts, EDGES, BOX, k, k, n_order, scale_method)
        np.testing.assert_array_equal(result["weights"], 1.0)
        np.testing.assert_array_equal(result["t_corr"], full_box_tuples(k, uneven, n_order))

    def test_the_estimate_does_not_depend_on_how_the_box_was_split(self):
        """With every partition present, relabelling the points into any
        number of partitions leaves the estimate unchanged."""
        coords, _ = catalogue(6, False)
        rng = np.random.default_rng(7)
        estimates = []
        for k in (2, 3, 6, 11):
            parts = rng.integers(0, k, len(coords)).astype(np.int32)
            estimates.append(compute_npcf(coords, parts, EDGES, BOX, k, k, 2)["t_corr"])
            estimates.append(compute_xi(coords, parts, EDGES, BOX, k, k)["dd_corr"])
        for estimate in estimates[1:]:
            np.testing.assert_array_equal(estimate, estimates[0])


class TestUnbiased:
    @pytest.mark.parametrize("statistic", PAIR_STATISTICS)
    @pytest.mark.parametrize("k, uneven", SPLITS)
    def test_pairs(self, statistic, k, uneven):
        coords, parts = catalogue(k, uneven)
        full = full_box_pairs(coords, statistic)
        assert full.sum() > 0
        for m in range(2, k + 1):
            estimates = []
            for s in selections(parts, k, m):
                estimate, power = pair_estimate(statistic, coords[s], parts[s], k, m)
                estimates.append(estimate)
            np.testing.assert_allclose(
                np.mean(estimates, axis=0), full * (m / k) ** power, rtol=1e-12, err_msg=f"m={m}"
            )

    @pytest.mark.parametrize("scale_method", ["scale-up", "scale-down"])
    @pytest.mark.parametrize("n_order, k, uneven", ORDERS_AND_SPLITS)
    def test_tuples(self, n_order, scale_method, k, uneven):
        coords, parts = catalogue(k, uneven)
        full = full_box_tuples(k, uneven, n_order)
        assert full.sum() > 0
        for m in range(n_order, k + 1):
            estimates = [
                compute_npcf(coords[s], parts[s], EDGES, BOX, k, m, n_order, scale_method)["t_corr"]
                for s in selections(parts, k, m)
            ]
            expected = full * (m / k) ** n_order if scale_method == "scale-down" else full
            np.testing.assert_allclose(
                np.mean(estimates, axis=0), expected, rtol=1e-12, err_msg=f"m={m}"
            )

    def test_a_single_selection_is_not_the_full_count(self):
        """Unbiased is not exact: individual selections scatter about the
        full count, which is what the average over selections removes."""
        coords, parts = catalogue(4, False)
        estimates = [
            compute_npcf(coords[s], parts[s], EDGES, BOX, 4, 2, 2)["t_corr"]
            for s in selections(parts, 4, 2)
        ]
        assert np.ptp(estimates, axis=0).max() > 0


class TestTooFewSelected:
    @pytest.mark.parametrize("n_order", [2, 3, 4])
    def test_tuples(self, n_order):
        coords, parts = catalogue(6, False)
        for m in range(0, n_order):
            keep = parts < max(m, 1)
            with pytest.raises(ValueError, match="order of the statistic"):
                compute_npcf(coords[keep], parts[keep], EDGES, BOX, 6, m, n_order)

    @pytest.mark.parametrize("statistic", PAIR_STATISTICS)
    def test_pairs(self, statistic):
        coords, parts = catalogue(6, False)
        keep = parts == 0
        with pytest.raises(ValueError, match="order of the statistic"):
            pair_estimate(statistic, coords[keep], parts[keep], 6, 1)
