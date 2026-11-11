"""Sparse-partition estimators: weights, unbiasedness and output contracts."""

from itertools import combinations
from math import comb

import numpy as np
import pytest
import reference as ref
from pysauce import (
    analytic_rr,
    analytic_rr_1d,
    analytic_rr_smu,
    compute_2pcf,
    compute_npcf,
    compute_npcf3_shape,
    compute_npcf3_smu,
    compute_xi,
    compute_xi_smu,
    count_npoint,
    count_npoint3_shape,
    count_npoint3_smu,
    count_pairs_1d,
    count_pairs_2d,
    count_pairs_smu,
    partition_weights,
)

K = 5


@pytest.fixture(scope="module")
def clustered():
    """A clustered catalogue with K partitions, so that the auto and cross
    counts differ strongly and a wrong weight would show."""
    coords, parts = ref.clustered_catalogue(400, 40.0, K, seed=31, n_clusters=15)
    return coords, parts, 40.0, np.array([0.2, 1.0, 2.5, 4.0])


def subsets(parts, m):
    """Masks selecting each possible choice of m of the K partitions."""
    for chosen in combinations(range(K), m):
        yield np.isin(parts, chosen)


class TestPartitionWeights:
    @pytest.mark.parametrize("k, m", [(4, 4), (8, 3), (27, 9), (10, 2), (2, 2)])
    def test_pair_weights_match_closed_forms(self, k, m):
        up = partition_weights(k, m, 2, "scale-up")
        down = partition_weights(k, m, 2, "scale-down")
        np.testing.assert_allclose(up, [k / m, k * (k - 1) / (m * (m - 1))], rtol=1e-15)
        np.testing.assert_allclose(down, [m / k, m * (k - 1) / (k * (m - 1))], rtol=1e-15)

    def test_triplet_example(self):
        # w_s = (k)_s / (m)_s * (m / k)^3 with k = 8, m = 4
        np.testing.assert_allclose(
            partition_weights(8, 4, 3, "scale-down"), [0.25, 7 / 12, 1.75], rtol=1e-15
        )

    @pytest.mark.parametrize("n_order", [2, 3, 5])
    def test_full_selection_has_unit_weights(self, n_order):
        np.testing.assert_array_equal(partition_weights(6, 6, n_order, "scale-up"), 1.0)

    def test_scale_down_is_scale_up_times_selected_fraction(self):
        k, m, n = 9, 4, 3
        np.testing.assert_allclose(
            partition_weights(k, m, n, "scale-down"),
            partition_weights(k, m, n, "scale-up") * (m / k) ** n,
            rtol=1e-15,
        )

    @pytest.mark.parametrize("n_order", [2, 3, 5, 8])
    def test_rejects_fewer_selected_partitions_than_the_order(self, n_order):
        """With m < N, tuples spanning more than m partitions can never be
        observed, so no weighting is unbiased."""
        partition_weights(10, n_order, n_order, "scale-up")
        with pytest.raises(ValueError, match="order of the statistic"):
            partition_weights(10, n_order - 1, n_order, "scale-up")

    @pytest.mark.parametrize(
        "k, m, n_order, error",
        [
            (3, 4, 2, ValueError),
            (5, 2, 0, ValueError),
            (5, 2.0, 2, TypeError),
            (5.5, 2, 2, TypeError),
            (5, True, 1, TypeError),
            (5, 2, None, TypeError),
        ],
    )
    def test_rejects_invalid_selection(self, k, m, n_order, error):
        with pytest.raises(error):
            partition_weights(k, m, n_order, "scale-up")

    def test_accepts_numpy_integers(self):
        np.testing.assert_array_equal(
            partition_weights(np.int64(6), np.int32(3), 2, "scale-up"),
            partition_weights(6, 3, 2, "scale-up"),
        )

    @pytest.mark.parametrize("k", [2, 5, 13])
    def test_weights_decrease_with_the_selection(self, k):
        """Selecting more partitions never needs a larger correction, and
        the classes spanning more partitions are always corrected more."""
        previous = None
        for m in range(2, k + 1):
            w = partition_weights(k, m, 2, "scale-up")
            assert np.all(np.diff(w) >= 0)
            if previous is not None:
                assert np.all(w <= previous)
            previous = w

    def test_rejects_unknown_scale_method(self):
        with pytest.raises(ValueError, match="scale_method"):
            partition_weights(4, 2, 2, "scale-sideways")


class TestUnbiasedOverSelections:
    """Averaged over every choice of m of the K partitions, the corrected
    counts must equal the counts of the full catalogue (scale-up) or those
    counts times (m / K)^N (scale-down), exactly up to rounding."""

    @pytest.mark.parametrize("m", [2, 3, K])
    def test_projected_pairs(self, clustered, m):
        coords, parts, box, edges = clustered
        pi_edges = np.array([0.0, 1.0, 4.0])
        full = sum(count_pairs_2d(coords, parts, edges, pi_edges, box))
        corrected = [
            compute_2pcf(coords[s], parts[s], edges, pi_edges, box, K, m)["dd_corr"]
            for s in subsets(parts, m)
        ]
        np.testing.assert_allclose(np.mean(corrected, axis=0), full, rtol=1e-12)

    @pytest.mark.parametrize("m", [2, 4])
    def test_radial_pairs(self, clustered, m):
        coords, parts, box, edges = clustered
        full = sum(count_pairs_1d(coords, parts, edges, box))
        corrected = [
            compute_xi(coords[s], parts[s], edges, box, K, m)["dd_corr"] for s in subsets(parts, m)
        ]
        np.testing.assert_allclose(np.mean(corrected, axis=0), full * (m / K) ** 2, rtol=1e-12)

    @pytest.mark.parametrize("m", [2, 3])
    def test_redshift_space_pairs(self, clustered, m):
        coords, parts, box, edges = clustered
        full = sum(count_pairs_smu(coords, parts, edges, 4, 1.0, box))
        corrected = [
            compute_xi_smu(
                coords[s],
                parts[s],
                edges,
                box,
                n_partitions=K,
                n_partitions_selected=m,
                n_mu_bins=4,
            )["dd_corr"]
            for s in subsets(parts, m)
        ]
        np.testing.assert_allclose(np.mean(corrected, axis=0), full * (m / K) ** 2, rtol=1e-12)

    @pytest.mark.parametrize("n_order, m", [(3, 3), (3, 4), (4, 4), (3, K)])
    @pytest.mark.parametrize("scale_method", ["scale-up", "scale-down"])
    def test_tuples(self, clustered, n_order, m, scale_method):
        coords, parts, box, edges = clustered
        full = count_npoint(coords, parts, edges, box, n_order)[1]
        if scale_method == "scale-down":
            full = full * (m / K) ** n_order
        corrected = [
            compute_npcf(coords[s], parts[s], edges, box, K, m, n_order, scale_method)["t_corr"]
            for s in subsets(parts, m)
        ]
        np.testing.assert_allclose(np.mean(corrected, axis=0), full, rtol=1e-12)

    @pytest.mark.parametrize("m", [3, 4])
    def test_binned_triplets(self, clustered, m):
        coords, parts, box, edges = clustered
        full_shape = count_npoint3_shape(coords, parts, edges, 4, box)[1].reshape(-1, 4)
        full_mu = count_npoint3_smu(coords, parts, edges, 4, 1.0, box)[1].reshape(-1, 4)
        shape, mu = [], []
        for s in subsets(parts, m):
            shape.append(compute_npcf3_shape(coords[s], parts[s], edges, 4, box, K, m)["t_corr"])
            mu.append(compute_npcf3_smu(coords[s], parts[s], edges, 4, box, K, m)["t_corr"])
        np.testing.assert_allclose(np.mean(shape, axis=0), full_shape, rtol=1e-12)
        np.testing.assert_allclose(np.mean(mu, axis=0), full_mu, rtol=1e-12)


@pytest.fixture(scope="module")
def poisson():
    """Uniform points in equal-sized partitions, of which m are selected."""
    rng = np.random.default_rng(32)
    n, box, k, m = 24_000, 150.0, 8, 3
    coords = rng.uniform(0, box, (n, 3))
    parts = rng.permutation(np.arange(n) % k).astype(np.int32)
    keep = parts < m
    return coords[keep], parts[keep], box, k, m


def assert_consistent_with_zero(result, alpha, beta, n_sigma=5.0):
    """|xi| within n_sigma of its Poisson noise, estimated from the counts."""
    sigma = np.sqrt(alpha**2 * result["dd_auto"] + beta**2 * result["dd_cross"]) / result["rr"]
    assert np.all(np.abs(result["xi"]) < n_sigma * sigma + 1e-3)


class TestPoissonNull:
    """For an unclustered catalogue the corrected xi vanishes for m < k."""

    def test_xi(self, poisson):
        coords, parts, box, k, m = poisson
        result = compute_xi(coords, parts, np.array([2.0, 5.0, 10.0, 20.0]), box, k, m)
        assert_consistent_with_zero(result, *partition_weights(k, m, 2, "scale-down"))

    def test_xi_smu(self, poisson):
        coords, parts, box, k, m = poisson
        result = compute_xi_smu(
            coords,
            parts,
            np.array([5.0, 10.0, 20.0]),
            box,
            n_partitions=k,
            n_partitions_selected=m,
            n_mu_bins=5,
        )
        result["xi"] = result["xi_grid"]
        assert_consistent_with_zero(result, *partition_weights(k, m, 2, "scale-down"))

    def test_xi_rp_pi(self, poisson):
        coords, parts, box, k, m = poisson
        result = compute_2pcf(
            coords, parts, np.array([2.0, 8.0, 20.0]), np.array([0.0, 10.0, 20.0]), box, k, m
        )
        assert_consistent_with_zero(result, *partition_weights(k, m, 2, "scale-up"))


def selected(clustered, m):
    """The catalogue restricted to partitions 0 ... m - 1."""
    coords, parts, box, edges = clustered
    keep = parts < m
    return coords[keep], parts[keep], box, edges


class TestOutputs:
    def test_compute_xi(self, clustered):
        coords, parts, box, edges = selected(clustered, 3)
        res = compute_xi(coords, parts, edges, box, K, 3)
        auto, cross = count_pairs_1d(coords, parts, edges, box)
        alpha, beta = partition_weights(K, 3, 2, "scale-down")
        np.testing.assert_array_equal(res["dd_auto"], auto)
        np.testing.assert_array_equal(res["dd_cross"], cross)
        np.testing.assert_allclose(res["dd_corr"], alpha * auto + beta * cross, rtol=1e-15)
        np.testing.assert_allclose(res["rr"], analytic_rr_1d(edges, box, len(coords)))
        np.testing.assert_allclose(res["xi"], res["dd_corr"] / res["rr"] - 1)
        np.testing.assert_allclose(res["r_mid"], np.sqrt(edges[:-1] * edges[1:]))

    def test_compute_2pcf(self, clustered):
        coords, parts, box, edges = selected(clustered, 2)
        pi_edges = np.array([0.0, 1.0, 3.0])
        res = compute_2pcf(coords, parts, edges, pi_edges, box, K, 2)
        n_total = round(len(coords) * K / 2)
        np.testing.assert_allclose(res["rr"], analytic_rr(edges, pi_edges, box, n_total))
        np.testing.assert_allclose(res["xi"], res["dd_corr"] / res["rr"] - 1)
        np.testing.assert_allclose(res["wp"], 2 * (res["xi"] * np.diff(pi_edges)).sum(axis=1))
        explicit = compute_2pcf(coords, parts, edges, pi_edges, box, K, 2, n_total=10_000)
        np.testing.assert_allclose(explicit["rr"], analytic_rr(edges, pi_edges, box, 10_000))

    def test_compute_xi_smu(self, clustered):
        coords, parts, box, edges = clustered
        res = compute_xi_smu(
            coords, parts, edges, box, n_partitions=K, n_partitions_selected=K, n_mu_bins=4
        )
        mu = np.array([0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(res["mu_mid"], mu)
        np.testing.assert_allclose(res["dd_corr"], res["dd_auto"] + res["dd_cross"])
        np.testing.assert_allclose(res["rr"], analytic_rr_smu(edges, box, len(coords), 4))
        np.testing.assert_array_equal(res["xi_smu"], res["xi_grid"])
        xi = res["xi_grid"]
        np.testing.assert_allclose(res["xi0"], np.nansum(xi * 0.25, axis=1))
        legendre_2 = 0.5 * (3 * mu**2 - 1)
        np.testing.assert_allclose(res["xi2"], 5 * np.nansum(xi * legendre_2 * 0.25, axis=1))

    def test_compute_npcf(self, clustered):
        coords, parts, box, edges = selected(clustered, 4)
        res = compute_npcf(coords, parts, edges, box, K, 4, 3)
        t_by_s, t_total = count_npoint(coords, parts, edges, box, 3)
        np.testing.assert_array_equal(res["t_total"], t_total)
        np.testing.assert_allclose(res["t_corr"], res["weights"] @ t_by_s, rtol=1e-14)
        np.testing.assert_allclose(res["zeta"], res["t_corr"] / comb(len(coords), 3))

    def test_full_selection_needs_no_correction(self, clustered):
        coords, parts, box, edges = clustered
        np.testing.assert_array_equal(
            compute_npcf(coords, parts, edges, box, K, K, 3)["t_corr"],
            compute_npcf(coords, parts, edges, box, K, K, 3)["t_total"],
        )
        for res in (
            compute_npcf3_shape(coords, parts, edges, 3, box, K, K),
            compute_npcf3_smu(coords, parts, edges, 3, box, K, K),
        ):
            np.testing.assert_array_equal(res["t_corr"], res["t_total"])
            assert res["t_corr"].shape == (len(edges) - 1, 3)

    def test_binned_triplets_marginalise_to_compute_npcf(self, clustered):
        coords, parts, box, edges = selected(clustered, 3)
        npcf = compute_npcf(coords, parts, edges, box, K, 3, 3, "scale-down")
        for res in (
            compute_npcf3_shape(coords, parts, edges, 5, box, K, 3, "scale-down"),
            compute_npcf3_smu(coords, parts, edges, 5, box, K, 3, scale_method="scale-down"),
        ):
            np.testing.assert_allclose(res["t_corr"].sum(axis=1), npcf["t_corr"], rtol=1e-12)
            np.testing.assert_array_equal(res["weights"], npcf["weights"])

    def test_accepts_other_dtypes(self, clustered):
        coords, parts, box, edges = clustered
        expected = compute_xi(coords, parts, edges, box, K, K)["dd_corr"]
        got = compute_xi(coords.tolist(), parts.astype(np.int64), edges.tolist(), box, K, K)
        np.testing.assert_array_equal(got["dd_corr"], expected)

    def test_rejects_non_integer_labels(self, clustered):
        coords, parts, box, edges = clustered
        with pytest.raises(TypeError, match="integers"):
            compute_xi(coords, parts.astype(float), edges, box, K, K)

    def test_rejects_labels_beyond_int32(self, clustered):
        coords, parts, box, edges = clustered
        with pytest.raises(ValueError, match="32-bit"):
            compute_xi(coords, parts.astype(np.int64) + 2**40, edges, box, K, K)


SELECTION_ERRORS = [
    (lambda c, p, e, b, m: compute_xi(c, p, e, b, K, m), 2),
    (lambda c, p, e, b, m: compute_2pcf(c, p, e, e, b, K, m), 2),
    (
        lambda c, p, e, b, m: compute_xi_smu(
            c, p, e, b, n_partitions=K, n_partitions_selected=m, n_mu_bins=2
        ),
        2,
    ),
    (lambda c, p, e, b, m: compute_npcf(c, p, e, b, K, m, 4), 4),
    (lambda c, p, e, b, m: compute_npcf3_shape(c, p, e, 2, b, K, m), 3),
    (lambda c, p, e, b, m: compute_npcf3_smu(c, p, e, 2, b, K, m), 3),
]


@pytest.mark.parametrize("estimator, min_selected", SELECTION_ERRORS)
def test_rejects_invalid_selection(clustered, estimator, min_selected):
    """m must satisfy N <= m <= k, whatever the catalogue holds."""
    coords, parts, box, edges = clustered
    with pytest.raises(ValueError, match="cannot exceed"):
        estimator(coords, parts, edges, box, K + 1)
    one_partition = parts == 0
    for m in range(0, min_selected):
        with pytest.raises(ValueError, match="order of the statistic"):
            estimator(coords[one_partition], parts[one_partition], edges, box, m)


@pytest.mark.parametrize("estimator, min_selected", SELECTION_ERRORS)
def test_rejects_more_labels_than_selected_partitions(clustered, estimator, min_selected):
    """A catalogue of m partitions cannot hold more than m distinct labels;
    weighting it as if it did would bias the estimate."""
    coords, parts, box, edges = clustered
    keep = parts < min_selected + 1
    with pytest.raises(ValueError, match="distinct labels"):
        estimator(coords[keep], parts[keep], edges, box, min_selected)
    # Fewer labels than m is fine: a selected partition may be empty.
    fewer = parts < min_selected - 1 if min_selected > 1 else parts == 0
    estimator(coords[fewer], parts[fewer], edges, box, min_selected)


@pytest.mark.parametrize(
    "estimator",
    [
        lambda c, p, e, b: compute_npcf(c, p, e, b, K, K, 3, "bad"),
        lambda c, p, e, b: compute_npcf3_shape(c, p, e, 2, b, K, K, "bad"),
        lambda c, p, e, b: compute_npcf3_smu(c, p, e, 2, b, K, K, scale_method="bad"),
    ],
)
def test_rejects_unknown_scale_method(clustered, estimator):
    coords, parts, box, edges = clustered
    with pytest.raises(ValueError, match="scale_method"):
        estimator(coords, parts, edges, box)
