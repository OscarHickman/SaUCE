"""Tuple counters: count_npoint, count_npoint3_shape and count_npoint3_smu."""

import numpy as np
import pytest
import reference as ref
from pysauce import count_npoint, count_npoint3_shape, count_npoint3_smu, count_pairs_1d

# (catalogue, box, r_bins). "coarse" has r_max > box / 4 (per-pair wrapping)
# and "fine" has 40 narrow bins, which stresses the constant-time bin lookup.
TRIPLET_CASES = {
    "uniform": (ref.uniform_catalogue(120, 20.0, 3, seed=21), 20.0, [0.5, 2.0, 3.5, 5.0]),
    "clustered": (ref.clustered_catalogue(120, 60.0, 4, seed=22), 60.0, [0.1, 1.0, 2.5, 4.0]),
    "coarse": (ref.uniform_catalogue(45, 40.0, 4, seed=23), 40.0, [1.0, 5.0, 12.0, 19.0]),
    "fine": (ref.uniform_catalogue(80, 20.0, 3, seed=24), 20.0, np.linspace(0.5, 6.0, 41)),
}


@pytest.fixture(params=list(TRIPLET_CASES), ids=list(TRIPLET_CASES))
def triplet_case(request):
    (coords, parts), box, edges = TRIPLET_CASES[request.param]
    return coords, parts, box, np.asarray(edges, dtype=np.float64)


class TestCountNpoint:
    def test_triplets_match_brute_force(self, triplet_case):
        coords, parts, box, edges = triplet_case
        t_by_s, t_total = count_npoint(coords, parts, edges, box, 3)
        expected = ref.tuples(coords, parts, edges, box, 3)
        assert expected.sum() > 0
        np.testing.assert_array_equal(t_by_s, expected)
        np.testing.assert_array_equal(t_total, expected.sum(axis=0))

    @pytest.mark.parametrize(
        "n_order, n, box, edges",
        [
            (4, 30, 15.0, [0.5, 2.0, 4.0]),
            (4, 25, 16.0, [1.0, 3.0, 7.0]),  # r_max > box / 4
            (5, 18, 12.0, [1.0, 3.0, 5.0, 8.0]),
            (6, 14, 10.0, [1.0, 4.0, 7.0]),
        ],
    )
    def test_higher_orders_match_brute_force(self, n_order, n, box, edges):
        coords, parts = ref.clustered_catalogue(n, box, n_order, seed=n_order, n_clusters=3)
        edges = np.array(edges)
        t_by_s, t_total = count_npoint(coords, parts, edges, box, n_order)
        expected = ref.tuples(coords, parts, edges, box, n_order)
        assert expected.sum() > 0
        np.testing.assert_array_equal(t_by_s, expected)
        np.testing.assert_array_equal(t_total, expected.sum(axis=0))

    def test_every_partition_class_is_counted(self):
        """With one large partition and several small ones, all classes
        s = 1 ... 5 of 5-tuples occur and must match the brute force."""
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 12.0, (15, 3))
        parts = rng.permutation([0] * 7 + [1, 1, 2, 2, 3, 3, 4, 4]).astype(np.int32)
        edges = np.linspace(1.0, 8.0, 6)
        expected = ref.tuples(coords, parts, edges, 12.0, 5)
        assert np.all(expected.sum(axis=1) > 0)
        np.testing.assert_array_equal(count_npoint(coords, parts, edges, 12.0, 5)[0], expected)

    def test_pairs_equal_count_pairs_1d(self, triplet_case):
        coords, parts, box, edges = triplet_case
        t_by_s, t_total = count_npoint(coords, parts, edges, box, 2)
        auto, cross = count_pairs_1d(coords, parts, edges, box)
        np.testing.assert_array_equal(t_by_s, [auto, cross])
        np.testing.assert_array_equal(t_total, auto + cross)

    def test_tuples_wider_than_r_max_are_excluded(self):
        coords = np.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [2.0, 1.0, 1.0], [50.0, 50.0, 50.0]])
        _, t_total = count_npoint(coords, np.zeros(4, np.int32), np.array([0.1, 5.0]), 100.0, 3)
        np.testing.assert_array_equal(t_total, [1])

    def test_triplet_is_binned_by_its_longest_side(self):
        # sides 3, 4, 5 -> bin [4.5, 6)
        coords = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 13.0], [14.0, 10.0, 10.0]])
        t_by_s, _ = count_npoint(
            coords, np.array([0, 1, 1], np.int32), np.array([1.0, 4.5, 6.0]), 50.0, 3
        )
        np.testing.assert_array_equal(t_by_s, [[0, 0], [0, 1], [0, 0]])

    @pytest.mark.parametrize("n_order", [3, 4])
    def test_shards_sum_to_the_full_count(self, triplet_case, n_order):
        coords, parts, box, edges = triplet_case
        full_by_s, full_total = count_npoint(coords, parts, edges, box, n_order)
        shards = [count_npoint(coords, parts, edges, box, n_order, i, 3) for i in range(3)]
        np.testing.assert_array_equal(sum(s[0] for s in shards), full_by_s)
        np.testing.assert_array_equal(sum(s[1] for s in shards), full_total)

    def test_input_order_does_not_matter(self, triplet_case):
        coords, parts, box, edges = triplet_case
        perm = np.random.default_rng(1).permutation(len(coords))
        for n_order in (3, 4):
            np.testing.assert_array_equal(
                count_npoint(coords, parts, edges, box, n_order)[0],
                count_npoint(coords[perm], parts[perm], edges, box, n_order)[0],
            )

    def test_adding_an_outer_bin_leaves_inner_bins_unchanged(self, triplet_case):
        """A wider r_max changes the grid and admits more tuples, but only
        into the new bin."""
        coords, parts, box, edges = triplet_case
        wider = np.append(edges, 1.25 * edges[-1])
        for n_order in (3, 4):
            narrow = count_npoint(coords, parts, edges, box, n_order)[0]
            wide = count_npoint(coords, parts, wider, box, n_order)[0]
            np.testing.assert_array_equal(wide[:, :-1], narrow)


def shape_bin(n_ratio):
    def bin_of(longest_sq, shortest_sq, _dz):
        ratio = np.sqrt(shortest_sq / longest_sq)
        return np.minimum((ratio * n_ratio).astype(np.int64), n_ratio - 1)

    return bin_of


def mu_bin(n_mu, mu_max):
    def bin_of(longest_sq, _shortest_sq, dz):
        return ref.mu_index(np.abs(dz) / np.sqrt(longest_sq), mu_max, n_mu)

    return bin_of


class TestCountNpoint3Shape:
    def test_matches_brute_force(self, triplet_case):
        coords, parts, box, edges = triplet_case
        t_by_s, t_total = count_npoint3_shape(coords, parts, edges, 6, box)
        expected = ref.triplets(coords, parts, edges, box, 6, shape_bin(6))
        np.testing.assert_array_equal(t_by_s, expected.reshape(3, -1))
        np.testing.assert_array_equal(t_total, expected.sum(axis=0).ravel())

    def test_marginalises_to_count_npoint(self, triplet_case):
        coords, parts, box, edges = triplet_case
        t_by_s, t_total = count_npoint3_shape(coords, parts, edges, 8, box)
        ref_by_s, ref_total = count_npoint(coords, parts, edges, box, 3)
        np.testing.assert_array_equal(t_by_s.reshape(3, -1, 8).sum(axis=2), ref_by_s)
        np.testing.assert_array_equal(t_total.reshape(-1, 8).sum(axis=1), ref_total)

    def test_equilateral_triangle_is_in_the_last_shape_bin(self):
        h = np.sqrt(3.0) / 2 * 4.0
        coords = np.array([[10.0, 10.0, 10.0], [14.0, 10.0, 10.0], [12.0, 10.0 + h, 10.0]])
        _, t_total = count_npoint3_shape(
            coords, np.zeros(3, np.int32), np.array([1.0, 5.0]), 4, 50.0
        )
        np.testing.assert_array_equal(t_total, [0, 0, 0, 1])


class TestCountNpoint3Smu:
    @pytest.mark.parametrize("mu_max", [1.0, 0.6])
    def test_matches_brute_force(self, triplet_case, mu_max):
        coords, parts, box, edges = triplet_case
        t_by_s, t_total = count_npoint3_smu(coords, parts, edges, 5, mu_max, box)
        expected = ref.triplets(coords, parts, edges, box, 5, mu_bin(5, mu_max))
        np.testing.assert_array_equal(t_by_s, expected.reshape(3, -1))
        np.testing.assert_array_equal(t_total, expected.sum(axis=0).ravel())

    def test_marginalises_to_count_npoint(self, triplet_case):
        coords, parts, box, edges = triplet_case
        t_by_s, _ = count_npoint3_smu(coords, parts, edges, 5, 1.0, box)
        np.testing.assert_array_equal(
            t_by_s.reshape(3, -1, 5).sum(axis=2), count_npoint(coords, parts, edges, box, 3)[0]
        )

    def test_mu_is_taken_from_the_longest_side(self):
        """Right triangle with legs 3 (along z) and 4 (along x): the
        hypotenuse has dz = 3 and length 5, so mu = 0.6."""
        coords = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 13.0], [14.0, 10.0, 10.0]])
        t_by_s, t_total = count_npoint3_smu(
            coords, np.zeros(3, np.int32), np.array([0.1, 6.0]), 10, 1.0, 100.0
        )
        expected = np.zeros(10)
        expected[6] = 1
        np.testing.assert_array_equal(t_total, expected)
        np.testing.assert_array_equal(t_by_s[0], expected)


def test_output_shapes():
    coords, parts = ref.uniform_catalogue(40, 10.0, 3, seed=0)
    edges = np.array([0.5, 1.0, 2.0])
    for n_order in (2, 3, 5):
        t_by_s, t_total = count_npoint(coords, parts, edges, 10.0, n_order)
        assert t_by_s.shape == (n_order, 2) and t_total.shape == (2,)
    t_by_s, t_total = count_npoint3_shape(coords, parts, edges, 4, 10.0)
    assert t_by_s.shape == (3, 8) and t_total.shape == (8,)
    t_by_s, t_total = count_npoint3_smu(coords, parts, edges, 7, 1.0, 10.0)
    assert t_by_s.shape == (3, 14) and t_total.shape == (14,)
