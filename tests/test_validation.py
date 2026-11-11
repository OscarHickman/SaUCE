"""Input validation of the compiled counters.

Invalid input must raise ValueError rather than return wrong counts or crash
the interpreter.
"""

import numpy as np
import pytest
from pysauce import (
    count_npoint,
    count_npoint3_shape,
    count_npoint3_smu,
    count_pairs_1d,
    count_pairs_1d_partition_matrix,
    count_pairs_2d,
    count_pairs_smu,
)

BOX = 50.0
EDGES = np.array([1.0, 5.0])

# Every counter, called with (coords, partition_ids, bin_edges, box_size).
COUNTERS = {
    "count_pairs_1d": lambda c, p, e, b: count_pairs_1d(c, p, e, b),
    "count_pairs_1d_partition_matrix": lambda c, p, e, b: count_pairs_1d_partition_matrix(
        c, p, e, b, 4
    ),
    "count_pairs_2d": lambda c, p, e, b: count_pairs_2d(c, p, e, e, b),
    "count_pairs_smu": lambda c, p, e, b: count_pairs_smu(c, p, e, 4, 1.0, b),
    "count_npoint": lambda c, p, e, b: count_npoint(c, p, e, b, 3),
    "count_npoint3_shape": lambda c, p, e, b: count_npoint3_shape(c, p, e, 4, b),
    "count_npoint3_smu": lambda c, p, e, b: count_npoint3_smu(c, p, e, 4, 1.0, b),
}


@pytest.fixture(params=list(COUNTERS), ids=list(COUNTERS))
def counter(request):
    return COUNTERS[request.param]


@pytest.fixture
def catalogue():
    rng = np.random.default_rng(0)
    return rng.uniform(0, BOX, (20, 3)), rng.integers(0, 4, 20).astype(np.int32)


def test_accepts_valid_input(counter, catalogue):
    coords, parts = catalogue
    counter(coords, parts, EDGES, BOX)
    counter(coords, parts, np.array([0.0, 0.49 * BOX]), BOX)


def test_accepts_empty_catalogue(counter):
    counter(np.empty((0, 3)), np.empty(0, np.int32), EDGES, BOX)


@pytest.mark.parametrize("box", [0.0, -1.0, np.nan, np.inf])
def test_rejects_bad_box_size(counter, catalogue, box):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="box_size"):
        counter(coords, parts, EDGES, box)


def test_rejects_wrong_number_of_columns(counter, catalogue):
    coords, parts = catalogue
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        counter(np.ascontiguousarray(coords[:, :2]), parts, EDGES, BOX)


def test_rejects_mismatched_lengths(counter, catalogue):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="must match"):
        counter(coords, parts[:-1], EDGES, BOX)


@pytest.mark.parametrize("bad", [-0.1, BOX, np.nan, np.inf])
def test_rejects_coordinates_outside_box(counter, catalogue, bad):
    coords, parts = catalogue
    coords[3, 1] = bad
    with pytest.raises(ValueError, match=r"\[0, box_size\)"):
        counter(coords, parts, EDGES, BOX)


@pytest.mark.parametrize(
    "edges, message",
    [
        ([5.0], "at least 2 edges"),
        ([5.0, 1.0, 10.0], "strictly increasing"),
        ([1.0, 1.0], "strictly increasing"),
        ([-1.0, 2.0], "non-negative"),
        ([1.0, np.nan], "finite"),
    ],
)
def test_rejects_bad_bin_edges(counter, catalogue, edges, message):
    coords, parts = catalogue
    with pytest.raises(ValueError, match=message):
        counter(coords, parts, np.array(edges), BOX)


def test_names_the_offending_bin_argument(catalogue):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="pi_bins"):
        count_pairs_2d(coords, parts, EDGES, np.array([5.0, 1.0]), BOX)
    with pytest.raises(ValueError, match="r_p_bins"):
        count_pairs_2d(coords, parts, np.array([5.0, 1.0]), EDGES, BOX)


@pytest.mark.parametrize("counter_with_mu", [count_pairs_smu, count_npoint3_smu])
def test_rejects_bad_mu_binning(catalogue, counter_with_mu):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="mu_max"):
        counter_with_mu(coords, parts, EDGES, 4, 0.0, BOX)
    with pytest.raises(ValueError, match="n_mu_bins"):
        counter_with_mu(coords, parts, EDGES, 0, 1.0, BOX)


@pytest.mark.parametrize("n_order", [0, 1, 9])
def test_rejects_unsupported_tuple_order(catalogue, n_order):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="n_order"):
        count_npoint(coords, parts, EDGES, BOX, n_order)


def test_rejects_zero_shape_bins(catalogue):
    coords, parts = catalogue
    with pytest.raises(ValueError, match="n_ratio"):
        count_npoint3_shape(coords, parts, EDGES, 0, BOX)


@pytest.mark.parametrize(
    "k, labels, message",
    [(0, None, "k_partitions"), (3, None, r"\[0, k_partitions\)"), (4, -1, r"\[0, k_partitions\)")],
)
def test_partition_matrix_rejects_labels_outside_range(catalogue, k, labels, message):
    coords, parts = catalogue
    if labels is not None:
        parts[0] = labels
    with pytest.raises(ValueError, match=message):
        count_pairs_1d_partition_matrix(coords, parts, EDGES, BOX, k)


@pytest.mark.parametrize(
    "n_order, shard_idx, n_shards, message",
    [
        (3, 0, None, "together"),
        (3, None, 2, "together"),
        (3, 0, 0, "n_shards"),
        (3, 2, 2, "shard_idx"),
        (2, 0, 2, "n_order >= 3"),
    ],
)
def test_rejects_bad_sharding(catalogue, n_order, shard_idx, n_shards, message):
    coords, parts = catalogue
    with pytest.raises(ValueError, match=message):
        count_npoint(coords, parts, EDGES, BOX, n_order, shard_idx, n_shards)


def test_requires_float64_and_int32_arrays(catalogue):
    """The compiled counters do not convert dtypes; the compute_* functions do."""
    coords, parts = catalogue
    with pytest.raises(TypeError):
        count_pairs_1d(coords.astype(np.float32), parts, EDGES, BOX)
    with pytest.raises(TypeError):
        count_pairs_1d(coords, parts.astype(np.int64), EDGES, BOX)
