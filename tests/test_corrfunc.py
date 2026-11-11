"""Pair counts checked against Corrfunc, an independent and widely used code.

Corrfunc's auto-correlation counts ordered pairs, so each pair appears twice;
its cross-correlation of two samples counts each pair once. Its bins are
half-open like ours, its pi bins have unit width, and it bins mu = |dz| / s
from the z axis, so the counts must agree exactly.
"""

import numpy as np
import pytest
import reference as ref
from pysauce import (
    count_pairs_1d,
    count_pairs_1d_partition_matrix,
    count_pairs_2d,
    count_pairs_smu,
)

theory = pytest.importorskip("Corrfunc.theory")

BOX = 100.0
K = 4
EDGES = np.array([0.3, 1.0, 2.5, 5.0, 9.0])

CATALOGUES = {
    "uniform": ref.uniform_catalogue(5000, BOX, K, seed=51),
    "clustered": ref.clustered_catalogue(5000, BOX, K, seed=52, n_clusters=100, spread=2.0),
}


@pytest.fixture(params=list(CATALOGUES), ids=list(CATALOGUES))
def catalogue(request):
    return CATALOGUES[request.param]


def corrfunc_auto(counter, coords, *args, **kwargs):
    """Unordered pair counts of one sample."""
    x, y, z = coords.T
    result = counter(1, 1, *args, x, y, z, periodic=True, boxsize=BOX, verbose=False, **kwargs)
    return result["npairs"] / 2


def corrfunc_cross(coords1, coords2, edges):
    """Pair counts between two samples."""
    x1, y1, z1 = coords1.T
    x2, y2, z2 = coords2.T
    result = theory.DD(
        0, 1, edges, x1, y1, z1, X2=x2, Y2=y2, Z2=z2, periodic=True, boxsize=BOX, verbose=False
    )
    return result["npairs"].astype(np.float64)


def test_radial_counts(catalogue):
    coords, parts = catalogue
    auto, cross = count_pairs_1d(coords, parts, EDGES, BOX)
    np.testing.assert_array_equal(auto + cross, corrfunc_auto(theory.DD, coords, EDGES))


@pytest.mark.parametrize("mu_max, n_mu", [(1.0, 5), (0.6, 3)])
def test_redshift_space_counts(catalogue, mu_max, n_mu):
    coords, parts = catalogue
    auto, cross = count_pairs_smu(coords, parts, EDGES, n_mu, mu_max, BOX)
    expected = corrfunc_auto(theory.DDsmu, coords, EDGES, mu_max, n_mu)
    np.testing.assert_array_equal(auto + cross, expected.reshape(auto.shape))


def test_projected_counts(catalogue):
    coords, parts = catalogue
    pi_max = 6
    auto, cross = count_pairs_2d(coords, parts, EDGES, np.arange(pi_max + 1.0), BOX)
    expected = corrfunc_auto(theory.DDrppi, coords, float(pi_max), EDGES)
    np.testing.assert_array_equal(auto + cross, expected.reshape(auto.shape))


def test_auto_counts_are_the_pairs_within_each_partition(catalogue):
    coords, parts = catalogue
    auto, _ = count_pairs_1d(coords, parts, EDGES, BOX)
    per_partition = [corrfunc_auto(theory.DD, coords[parts == a], EDGES) for a in range(K)]
    np.testing.assert_array_equal(auto, np.sum(per_partition, axis=0))


def test_partition_matrix(catalogue):
    coords, parts = catalogue
    auto, cross = count_pairs_1d_partition_matrix(coords, parts, EDGES, BOX, K)
    for a in range(K):
        np.testing.assert_array_equal(
            auto[:, a], corrfunc_auto(theory.DD, coords[parts == a], EDGES)
        )
        for b in range(a + 1, K):
            np.testing.assert_array_equal(
                cross[:, a, b], corrfunc_cross(coords[parts == a], coords[parts == b], EDGES)
            )


# A larger, denser catalogue with bins out to a fifth of the box, so that most
# z windows span several blocks of candidates and neighbouring columns wrap
# around every face of the box.
WIDE_EDGES = np.geomspace(0.5, 20.0, 12)


@pytest.fixture(scope="module")
def dense():
    return ref.clustered_catalogue(40_000, BOX, K, seed=53, n_clusters=400, spread=3.0)


def test_radial_counts_to_a_fifth_of_the_box(dense):
    coords, parts = dense
    auto, cross = count_pairs_1d(coords, parts, WIDE_EDGES, BOX)
    np.testing.assert_array_equal(auto + cross, corrfunc_auto(theory.DD, coords, WIDE_EDGES))


@pytest.mark.parametrize("mu_max, n_mu", [(1.0, 20), (0.5, 7), (1.0, 1)])
def test_redshift_space_counts_to_a_fifth_of_the_box(dense, mu_max, n_mu):
    coords, parts = dense
    auto, cross = count_pairs_smu(coords, parts, WIDE_EDGES, n_mu, mu_max, BOX)
    expected = corrfunc_auto(theory.DDsmu, coords, WIDE_EDGES, mu_max, n_mu)
    np.testing.assert_array_equal(auto + cross, expected.reshape(auto.shape))


@pytest.mark.parametrize("pi_max", [3, 20, 30])
def test_projected_counts_to_a_fifth_of_the_box(dense, pi_max):
    """pi_max of 30 is beyond a quarter of the box, where the counter wraps
    each separation individually."""
    coords, parts = dense
    auto, cross = count_pairs_2d(coords, parts, WIDE_EDGES, np.arange(pi_max + 1.0), BOX)
    expected = corrfunc_auto(theory.DDrppi, coords, float(pi_max), WIDE_EDGES)
    np.testing.assert_array_equal(auto + cross, expected.reshape(auto.shape))


def test_auto_counts_per_partition_to_a_fifth_of_the_box(dense):
    coords, parts = dense
    auto, _ = count_pairs_1d(coords, parts, WIDE_EDGES, BOX)
    per_partition = [corrfunc_auto(theory.DD, coords[parts == a], WIDE_EDGES) for a in range(K)]
    np.testing.assert_array_equal(auto, np.sum(per_partition, axis=0))
