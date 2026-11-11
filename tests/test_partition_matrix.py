"""count_pairs_1d_partition_matrix: pair counts resolved by partition."""

import numpy as np
import pytest
import reference as ref
from pysauce import count_pairs_1d, count_pairs_1d_partition_matrix

CASES = {
    "uniform": (ref.uniform_catalogue(400, 60.0, 5, seed=11), 60.0, [0.5, 3.0, 7.0], 5),
    "coarse": (ref.clustered_catalogue(300, 30.0, 4, seed=12), 30.0, [0.0, 2.0, 9.0], 4),
    # partition 3 of 6 is empty, and k is larger than the largest label
    "unused": (ref.uniform_catalogue(300, 40.0, 3, seed=13), 40.0, [1.0, 5.0], 6),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request):
    (coords, parts), box, edges, k = CASES[request.param]
    return coords, parts, box, np.array(edges), k


def test_matches_brute_force(case):
    coords, parts, box, edges, k = case
    auto, cross = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
    expected_auto, expected_cross = ref.partition_matrix(coords, parts, edges, box, k)
    assert expected_cross.sum() > 0
    np.testing.assert_array_equal(auto, expected_auto)
    np.testing.assert_array_equal(cross, expected_cross)


def test_sums_to_count_pairs_1d(case):
    coords, parts, box, edges, k = case
    auto, cross = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
    dd_auto, dd_cross = count_pairs_1d(coords, parts, edges, box)
    np.testing.assert_array_equal(auto.sum(axis=1), dd_auto)
    np.testing.assert_array_equal(cross.sum(axis=(1, 2)), dd_cross)


def test_cross_counts_are_strictly_upper_triangular(case):
    coords, parts, box, edges, k = case
    _, cross = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
    lower = np.tril(np.ones((k, k), dtype=bool))
    assert np.all(cross[:, lower] == 0)


def test_shapes(case):
    coords, parts, box, edges, k = case
    auto, cross = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
    n_r = len(edges) - 1
    assert auto.shape == (n_r, k)
    assert cross.shape == (n_r, k, k)


def test_relabelling_partitions_permutes_the_matrix():
    (coords, parts), box, edges, k = CASES["uniform"]
    edges = np.array(edges)
    relabel = np.array([3, 0, 4, 1, 2], dtype=np.int32)
    auto, cross = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
    auto2, cross2 = count_pairs_1d_partition_matrix(coords, relabel[parts], edges, box, k)
    np.testing.assert_array_equal(auto2[:, relabel], auto)
    symmetric = cross + cross.transpose(0, 2, 1)
    symmetric2 = cross2 + cross2.transpose(0, 2, 1)
    np.testing.assert_array_equal(symmetric2[:, relabel][:, :, relabel], symmetric)
