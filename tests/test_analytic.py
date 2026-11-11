"""Analytic random pair counts."""

import numpy as np
import pytest
from pysauce import (
    analytic_rr,
    analytic_rr_1d,
    analytic_rr_smu,
    count_pairs_1d,
    count_pairs_2d,
    count_pairs_smu,
)

BOX = 100.0
N = 1000


def test_rr_1d_formula():
    edges = np.array([2.0, 8.0, 25.0])
    shells = 4 / 3 * np.pi * np.diff(edges**3)
    np.testing.assert_allclose(analytic_rr_1d(edges, BOX, N), N * (N - 1) / 2 * shells / BOX**3)


def test_rr_1d_sums_over_adjacent_bins():
    fine = analytic_rr_1d([1.0, 2.0, 3.0, 5.0], BOX, N)
    coarse = analytic_rr_1d([1.0, 5.0], BOX, N)
    np.testing.assert_allclose(fine.sum(), coarse[0])


def test_rr_1d_scaling():
    edges = [1.0, 10.0]
    assert analytic_rr_1d(edges, BOX, 200)[0] / analytic_rr_1d(edges, BOX, 100)[0] == pytest.approx(
        200 * 199 / (100 * 99)
    )
    assert analytic_rr_1d(edges, 2 * BOX, N)[0] / analytic_rr_1d(edges, BOX, N)[0] == pytest.approx(
        1 / 8
    )


def test_rr_smu_splits_each_shell_evenly_in_mu():
    edges = np.array([1.0, 4.0, 9.0])
    rr = analytic_rr_smu(edges, BOX, N, n_mu_bins=5)
    assert rr.shape == (2, 5)
    np.testing.assert_allclose(rr.sum(axis=1), analytic_rr_1d(edges, BOX, N))
    np.testing.assert_allclose(rr, np.broadcast_to(rr[:, :1], rr.shape))


def test_rr_smu_with_mu_max_covers_that_fraction_of_the_shell():
    edges = np.array([1.0, 4.0])
    rr = analytic_rr_smu(edges, BOX, N, n_mu_bins=4, mu_max=0.5)
    np.testing.assert_allclose(rr.sum(axis=1), 0.5 * analytic_rr_1d(edges, BOX, N))


def test_rr_projected_formula():
    rr = analytic_rr([2.0, 6.0], [3.0, 9.0], BOX, N)
    volume = 2 * np.pi * (6.0**2 - 2.0**2) * (9.0 - 3.0)
    np.testing.assert_allclose(rr, [[N * (N - 1) / 2 * volume / BOX**3]])


def test_rr_projected_shape():
    assert analytic_rr([0.0, 5.0, 10.0], [0.0, 5.0, 15.0, 30.0], BOX, N).shape == (2, 3)


@pytest.fixture(scope="module")
def uniform():
    rng = np.random.default_rng(41)
    n = 20_000
    return rng.uniform(0, BOX, (n, 3)), np.zeros(n, np.int32)


def assert_poisson_consistent(counts, expected, n_sigma=5.0):
    assert np.all(np.abs(counts - expected) < n_sigma * np.sqrt(expected))


class TestAgainstUniformCatalogue:
    """Pair counts of uniform random points scatter around the analytic RR."""

    def test_radial(self, uniform):
        coords, parts = uniform
        edges = np.array([1.0, 2.0, 4.0, 8.0])
        counts, _ = count_pairs_1d(coords, parts, edges, BOX)
        assert_poisson_consistent(counts, analytic_rr_1d(edges, BOX, len(coords)))

    def test_redshift_space(self, uniform):
        coords, parts = uniform
        edges = np.array([2.0, 4.0, 8.0])
        counts, _ = count_pairs_smu(coords, parts, edges, 4, 1.0, BOX)
        assert_poisson_consistent(counts, analytic_rr_smu(edges, BOX, len(coords), 4))

    def test_projected(self, uniform):
        coords, parts = uniform
        rp, pi = np.array([1.0, 3.0, 6.0]), np.array([0.0, 2.0, 5.0])
        counts, _ = count_pairs_2d(coords, parts, rp, pi, BOX)
        assert_poisson_consistent(counts, analytic_rr(rp, pi, BOX, len(coords)))
