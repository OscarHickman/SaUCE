"""Speed of the counters on a million uniform random points.

Run with ``pytest benchmarks`` (needs pytest-benchmark). The counters use
every available core.
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

BOX = 500.0
N_POINTS = 1_000_000
K = 8

R_BINS = np.logspace(-1, np.log10(20.0), 31)
S_BINS = np.logspace(-1, np.log10(40.0), 21)
N_MU = 50
PI_MAX = 40
TRIPLET_BINS = np.logspace(-1, np.log10(10.0), 16)
TUPLE_BINS = np.logspace(-1, np.log10(6.0), 11)


@pytest.fixture(scope="module")
def catalogue():
    rng = np.random.default_rng(0)
    return rng.uniform(0, BOX, (N_POINTS, 3)), rng.integers(0, K, N_POINTS).astype(np.int32)


def run(benchmark, group, counter, *args):
    benchmark.group = group
    benchmark.pedantic(counter, args=args, rounds=5, warmup_rounds=1)


def test_pairs_r(benchmark, catalogue):
    run(benchmark, "pairs in r", count_pairs_1d, *catalogue, R_BINS, BOX)


def test_pairs_s_mu(benchmark, catalogue):
    run(benchmark, "pairs in (s, mu)", count_pairs_smu, *catalogue, S_BINS, N_MU, 1.0, BOX)


def test_pairs_rp_pi(benchmark, catalogue):
    pi_bins = np.arange(PI_MAX + 1.0)
    run(benchmark, "pairs in (r_p, pi)", count_pairs_2d, *catalogue, R_BINS, pi_bins, BOX)


def test_partition_matrix(benchmark, catalogue):
    run(benchmark, "partition matrix", count_pairs_1d_partition_matrix, *catalogue, R_BINS, BOX, K)


@pytest.mark.parametrize("n_order", [3, 4, 5])
def test_tuples(benchmark, catalogue, n_order):
    bins = TRIPLET_BINS if n_order == 3 else TUPLE_BINS
    run(benchmark, f"{n_order}-tuples", count_npoint, *catalogue, bins, BOX, n_order)


def test_triplets_by_shape(benchmark, catalogue):
    run(benchmark, "3-tuples", count_npoint3_shape, *catalogue, TRIPLET_BINS, 10, BOX)


def test_triplets_by_mu(benchmark, catalogue):
    run(benchmark, "3-tuples", count_npoint3_smu, *catalogue, TRIPLET_BINS, 10, 1.0, BOX)
