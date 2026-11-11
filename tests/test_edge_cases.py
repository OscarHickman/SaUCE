"""Edge cases for the counters: unusual catalogues, binnings and inputs.

Each test compares against the brute-force references in reference.py or
checks an exact property, so any traversal or binning shortcut that loses or
double-counts a pair fails here.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import reference as ref
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


def assert_pairs_match_reference(coords, parts, edges, box):
    """All pair counters against their brute-force references."""
    edges = np.asarray(edges, dtype=np.float64)
    k = int(parts.max()) + 1 if parts.min() >= 0 else None
    if k is not None and k <= 64:
        got = count_pairs_1d_partition_matrix(coords, parts, edges, box, k)
        expected = ref.partition_matrix(coords, parts, edges, box, k)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
    for got, expected in [
        (count_pairs_1d(coords, parts, edges, box), ref.radial_pairs(coords, parts, edges, box)),
        (
            count_pairs_2d(coords, parts, edges, edges, box),
            ref.projected_pairs(coords, parts, edges, edges, box),
        ),
        (
            count_pairs_smu(coords, parts, edges, 7, 1.0, box),
            ref.redshift_space_pairs(coords, parts, edges, 7, 1.0, box),
        ),
    ]:
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


class TestCatalogueShapes:
    @pytest.mark.parametrize(
        "n", [2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200]
    )
    def test_every_occupancy_of_a_single_cell(self, n):
        """All points in one cell, in numbers around typical vector and block
        sizes, so that the remainder handling in the inner loops, and both
        ways of choosing candidates (the whole cell or a z window), are hit."""
        rng = np.random.default_rng(n)
        coords = 20.0 + rng.uniform(0, 3.0, (n, 3))
        parts = rng.integers(0, 3, n).astype(np.int32)
        assert_pairs_match_reference(coords, parts, [0.0, 0.5, 1.0, 2.0, 4.0], BOX)

    def test_dense_clump_in_sparse_background(self):
        rng = np.random.default_rng(1)
        clump = 25.0 + rng.normal(0, 0.3, (300, 3))
        background = rng.uniform(0, BOX, (200, 3))
        coords = np.mod(np.vstack([clump, background]), BOX)
        parts = rng.integers(0, 4, len(coords)).astype(np.int32)
        assert_pairs_match_reference(coords, parts, np.logspace(-2, np.log10(6.0), 25), BOX)

    def test_clump_straddling_the_periodic_corner(self):
        rng = np.random.default_rng(2)
        coords = np.mod(rng.normal(0, 0.8, (300, 3)), BOX)
        coords[coords >= BOX] = 0.0
        parts = rng.integers(0, 3, 300).astype(np.int32)
        assert_pairs_match_reference(coords, parts, [0.0, 0.3, 1.0, 2.0, 3.0], BOX)

    def test_points_on_the_box_faces(self):
        """Coordinates of exactly 0 and just below box_size."""
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, BOX, (300, 3))
        edge = np.nextafter(BOX, 0.0)
        coords[:100, 0] = rng.choice([0.0, edge], 100)
        coords[50:150, 1] = rng.choice([0.0, edge], 100)
        coords[100:200, 2] = rng.choice([0.0, edge], 100)
        parts = rng.integers(0, 3, 300).astype(np.int32)
        assert_pairs_match_reference(coords, parts, [0.1, 1.0, 3.0, 6.0], BOX)

    def test_coincident_points(self):
        """Pairs at zero separation are counted when the first edge is 0.
        Their mu is undefined, so the (s, mu) counter skips them."""
        coords = np.array([[5.0, 5.0, 5.0]] * 4 + [[6.0, 5.0, 5.0]])
        parts = np.array([0, 0, 1, 2, 0], np.int32)
        edges = np.array([0.0, 0.5, 2.0])
        auto, cross = count_pairs_1d(coords, parts, edges, BOX)
        np.testing.assert_array_equal(auto + cross, [6, 4])
        auto, cross = count_pairs_smu(coords, parts, edges, 4, 1.0, BOX)
        np.testing.assert_array_equal((auto + cross).sum(axis=1), [0, 4])
        assert_pairs_match_reference(coords, parts, edges, BOX)

    def test_all_points_along_one_line_of_sight(self):
        """dx = dy = 0 for every pair: mu = 1 is excluded, r_p = 0."""
        coords = np.column_stack([np.full(40, 7.0), np.full(40, 9.0), np.linspace(0, 45, 40)])
        parts = np.arange(40, dtype=np.int32) % 3
        assert_pairs_match_reference(coords, parts, [0.0, 2.0, 5.0, 10.0], BOX)

    def test_negative_and_large_partition_labels(self):
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, BOX, (300, 3))
        labels = np.array([-(2**31), -7, 0, 2**31 - 1], np.int32)[rng.integers(0, 4, 300)]
        assert_pairs_match_reference(coords, labels, [0.5, 2.0, 5.0], BOX)


class TestBinnings:
    @pytest.mark.parametrize(
        "edges",
        [
            np.logspace(-3, np.log10(8.0), 120),
            np.linspace(0.0, 8.0, 200),
            np.array([7.999, 8.0]),
            np.array([0.0, 1e-9, 8.0]),
            np.array([2.0, 2.0 + 1e-12, 2.0 + 2e-12, 8.0]),
        ],
        ids=["many-log", "many-linear", "one-thin", "tiny-first", "near-duplicate"],
    )
    def test_unusual_bin_edges(self, edges):
        coords, parts = ref.clustered_catalogue(500, BOX, 3, seed=5, spread=1.5)
        assert_pairs_match_reference(coords, parts, edges, BOX)

    @pytest.mark.parametrize("r_max", [0.05, 0.3, 12.4, 12.6, 16.0, 24.9])
    def test_search_radius_from_tiny_to_half_the_box(self, r_max):
        """Covers the 128-cell cap (cells much wider than r_max), the switch
        to per-pair wrapping at a quarter of the box, and a 3-cell grid."""
        coords, parts = ref.clustered_catalogue(400, BOX, 3, seed=6, spread=0.5 * r_max + 0.1)
        edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * r_max + 1e-3
        assert_pairs_match_reference(coords, parts, edges, BOX)

    @pytest.mark.parametrize("rp_max, pi_max", [(1.0, 20.0), (20.0, 1.0), (13.0, 2.0), (2.0, 13.0)])
    def test_projected_counts_with_very_different_ranges(self, rp_max, pi_max):
        coords, parts = ref.clustered_catalogue(400, BOX, 3, seed=7, spread=3.0)
        rp = np.linspace(0.1, rp_max, 5)
        pi = np.linspace(0.0, pi_max, 4)
        got = count_pairs_2d(coords, parts, rp, pi, BOX)
        expected = ref.projected_pairs(coords, parts, rp, pi, BOX)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("n_mu, mu_max", [(1, 1.0), (1, 0.1), (100, 1.0), (13, 0.37), (4, 2.0)])
    def test_mu_binnings(self, n_mu, mu_max):
        coords, parts = ref.clustered_catalogue(400, BOX, 3, seed=8, spread=1.5)
        edges = np.array([0.2, 1.0, 3.0, 5.0])
        got = count_pairs_smu(coords, parts, edges, n_mu, mu_max, BOX)
        expected = ref.redshift_space_pairs(coords, parts, edges, n_mu, mu_max, BOX)
        np.testing.assert_array_equal(got, expected)


class TestTuples:
    @pytest.mark.parametrize("n_order", [6, 7, 8])
    def test_highest_orders_match_brute_force(self, n_order):
        coords, parts = ref.clustered_catalogue(13, 10.0, 4, seed=n_order, n_clusters=1, spread=0.8)
        edges = np.array([0.5, 1.5, 2.5, 4.0])
        expected = ref.tuples(coords, parts, edges, 10.0, n_order)
        assert expected.sum() > 0
        np.testing.assert_array_equal(
            count_npoint(coords, parts, edges, 10.0, n_order)[0], expected
        )

    def test_dense_clump(self):
        coords, parts = ref.clustered_catalogue(90, BOX, 3, seed=9, n_clusters=2, spread=0.4)
        edges = np.linspace(0.1, 1.5, 15)
        for n_order in (3, 4):
            expected = ref.tuples(coords, parts, edges, BOX, n_order)
            np.testing.assert_array_equal(
                count_npoint(coords, parts, edges, BOX, n_order)[0], expected
            )

    def test_triplets_with_many_fine_bins(self):
        coords, parts = ref.clustered_catalogue(120, BOX, 3, seed=10, spread=1.0)
        edges = np.logspace(-2, np.log10(4.0), 150)
        np.testing.assert_array_equal(
            count_npoint(coords, parts, edges, BOX, 3)[0], ref.tuples(coords, parts, edges, BOX, 3)
        )

    def test_equilateral_triangles_have_ratio_one(self):
        """Many exact equilateral triangles: every one lands in the top
        shape bin whatever the number of bins."""
        h = np.sqrt(3.0) / 2
        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]])
        coords = np.vstack([base + [5.0 * i + 2.0, 20.0, 20.0] for i in range(8)])
        for n_ratio in (1, 2, 7):
            _, total = count_npoint3_shape(
                coords, np.zeros(24, np.int32), np.array([0.5, 1.5]), n_ratio, BOX
            )
            assert total[-1] == 8 and total.sum() == 8

    def test_triplets_perpendicular_to_the_line_of_sight_have_mu_zero(self):
        rng = np.random.default_rng(11)
        coords = rng.uniform(10, 40, (60, 3))
        coords[:, 2] = 25.0
        _, total = count_npoint3_smu(
            coords, np.zeros(60, np.int32), np.array([0.1, 8.0]), 5, 1.0, BOX
        )
        assert total[0] > 0 and total[1:].sum() == 0


class TestInputs:
    @pytest.fixture
    def catalogue(self):
        return ref.clustered_catalogue(400, BOX, 4, seed=12, spread=1.5)

    def test_non_contiguous_arrays(self, catalogue):
        coords, parts = catalogue
        edges = np.array([0.1, 1.0, 3.0, 6.0])
        expected = count_pairs_1d(coords, parts, edges, BOX)
        fortran = np.asfortranarray(coords)
        strided = np.repeat(coords, 2, axis=0)[::2]
        wide = np.hstack([coords, coords])[:, :3]
        for view in (fortran, strided, wide):
            assert not view.flags.c_contiguous
            np.testing.assert_array_equal(count_pairs_1d(view, parts, edges, BOX), expected)
        np.testing.assert_array_equal(
            count_pairs_1d(coords, np.repeat(parts, 2)[::2], edges, BOX), expected
        )
        np.testing.assert_array_equal(
            count_pairs_1d(coords, parts, np.repeat(edges, 2)[::2], BOX), expected
        )

    def test_repeated_calls_are_identical(self, catalogue):
        coords, parts = catalogue
        edges = np.array([0.1, 1.0, 3.0])
        for count in (
            lambda: count_pairs_smu(coords, parts, edges, 5, 1.0, BOX),
            lambda: count_npoint(coords, parts, edges, BOX, 4),
            lambda: count_pairs_1d_partition_matrix(coords, parts, edges, BOX, 4),
        ):
            first = count()
            for _ in range(3):
                for a, b in zip(first, count()):
                    np.testing.assert_array_equal(a, b)

    def test_inputs_are_not_modified(self, catalogue):
        coords, parts = catalogue
        coords_copy, parts_copy = coords.copy(), parts.copy()
        count_npoint(coords, parts, np.array([0.1, 2.0]), BOX, 3)
        count_pairs_2d(coords, parts, np.array([0.1, 2.0]), np.array([0.0, 2.0]), BOX)
        np.testing.assert_array_equal(coords, coords_copy)
        np.testing.assert_array_equal(parts, parts_copy)


SINGLE_THREAD_SCRIPT = """
import sys
import numpy as np
import reference as ref
from pysauce import count_pairs_1d, count_pairs_smu, count_npoint, count_pairs_1d_partition_matrix

coords, parts = ref.clustered_catalogue(3000, 60.0, 5, seed=13, spread=2.0)
edges = np.logspace(-1, np.log10(6.0), 12)
results = [
    *count_pairs_1d(coords, parts, edges, 60.0),
    *count_pairs_smu(coords, parts, edges, 6, 1.0, 60.0),
    *count_npoint(coords, parts, edges, 60.0, 3),
    *count_npoint(coords, parts, edges[:6], 60.0, 4),
    *count_pairs_1d_partition_matrix(coords, parts, edges, 60.0, 5),
]
np.savez(sys.argv[1], *results)
"""


def test_results_do_not_depend_on_the_number_of_threads(tmp_path):
    """Rayon's thread pool is fixed per process, so each thread count runs in
    its own interpreter."""
    outputs = []
    for threads in ("1", "3", str(os.cpu_count())):
        out = tmp_path / f"threads_{threads}.npz"
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        subprocess.run(
            [sys.executable, "-c", SINGLE_THREAD_SCRIPT, str(out)],
            check=True,
            env=env,
            cwd=Path(__file__).parent,
        )
        outputs.append(np.load(out))
    for other in outputs[1:]:
        for key in outputs[0].files:
            np.testing.assert_array_equal(outputs[0][key], other[key])
