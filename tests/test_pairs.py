"""Pair counters: count_pairs_1d, count_pairs_2d and count_pairs_smu."""

import numpy as np
import pytest
import reference as ref
from pysauce import count_pairs_1d, count_pairs_2d, count_pairs_smu

# Catalogues covering the regimes the traversal distinguishes. "coarse" has
# r_max above a quarter of the box, which forces a 3x3x3 grid and per-pair
# periodic wrapping; "lattice" has exact separations lying on bin edges.
CATALOGUES = {
    "uniform": (ref.uniform_catalogue(500, 100.0, 4, seed=1), 100.0, [0.5, 3.0, 8.0, 15.0]),
    "clustered": (ref.clustered_catalogue(600, 80.0, 5, seed=2), 80.0, [0.1, 1.0, 4.0, 10.0]),
    "coarse": (ref.uniform_catalogue(300, 40.0, 3, seed=3), 40.0, [1.0, 6.0, 12.0, 19.0]),
    "zero_edge": (ref.clustered_catalogue(500, 60.0, 4, seed=4), 60.0, [0.0, 2.0, 5.0]),
    "lattice": (ref.lattice_catalogue(600, 16.0, 3, seed=5), 16.0, [1.0, 2.0, 3.0, 5.0, 7.0]),
}


@pytest.fixture(params=list(CATALOGUES), ids=list(CATALOGUES))
def catalogue(request):
    (coords, parts), box, edges = CATALOGUES[request.param]
    return coords, parts, box, np.array(edges)


class TestAgainstBruteForce:
    def test_radial(self, catalogue):
        coords, parts, box, edges = catalogue
        auto, cross = count_pairs_1d(coords, parts, edges, box)
        expected_auto, expected_cross = ref.radial_pairs(coords, parts, edges, box)
        assert expected_auto.sum() > 0 and expected_cross.sum() > 0
        np.testing.assert_array_equal(auto, expected_auto)
        np.testing.assert_array_equal(cross, expected_cross)

    def test_projected(self, catalogue):
        coords, parts, box, edges = catalogue
        pi_edges = np.array([0.0, 1.0, 3.0, edges[-1]])
        auto, cross = count_pairs_2d(coords, parts, edges, pi_edges, box)
        expected_auto, expected_cross = ref.projected_pairs(coords, parts, edges, pi_edges, box)
        np.testing.assert_array_equal(auto, expected_auto)
        np.testing.assert_array_equal(cross, expected_cross)

    def test_projected_with_distinct_pi_range(self, catalogue):
        """The grid uses different cell widths across and along the line of
        sight when pi_max differs from r_p_max."""
        coords, parts, box, edges = catalogue
        pi_edges = np.linspace(0.0, 0.45 * box, 5)
        auto, cross = count_pairs_2d(coords, parts, edges[:2], pi_edges, box)
        expected = ref.projected_pairs(coords, parts, edges[:2], pi_edges, box)
        np.testing.assert_array_equal(auto, expected[0])
        np.testing.assert_array_equal(cross, expected[1])

    @pytest.mark.parametrize("mu_max", [1.0, 0.6])
    def test_redshift_space(self, catalogue, mu_max):
        coords, parts, box, edges = catalogue
        auto, cross = count_pairs_smu(coords, parts, edges, 5, mu_max, box)
        expected = ref.redshift_space_pairs(coords, parts, edges, 5, mu_max, box)
        np.testing.assert_array_equal(auto, expected[0])
        np.testing.assert_array_equal(cross, expected[1])


class TestExactGeometry:
    def test_pair_across_periodic_boundary(self):
        coords = np.array([[1.0, 50.0, 50.0], [99.0, 50.0, 50.0]])
        auto, cross = count_pairs_1d(
            coords, np.zeros(2, np.int32), np.array([1.0, 3.0, 10.0]), 100.0
        )
        np.testing.assert_array_equal(auto, [1, 0])
        np.testing.assert_array_equal(cross, [0, 0])

    def test_bins_are_half_open(self):
        """A separation equal to an edge belongs to the bin above it, and one
        equal to the last edge is not counted."""
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        auto, _ = count_pairs_1d(coords, np.zeros(3, np.int32), np.array([1.0, 2.0, 5.0]), 100.0)
        # separations: 2 (edge -> bin 1), 5 (last edge -> dropped), sqrt(29) (beyond)
        np.testing.assert_array_equal(auto, [0, 1])

    def test_pairs_outside_range_are_dropped(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
        auto, cross = count_pairs_1d(coords, np.zeros(3, np.int32), np.array([5.0, 50.0]), 200.0)
        assert auto.sum() + cross.sum() == 0

    def test_projected_components(self):
        coords = np.array([[10.0, 10.0, 10.0], [13.0, 14.0, 12.0]])  # r_p = 5, pi = 2
        auto, _ = count_pairs_2d(
            coords, np.zeros(2, np.int32), np.array([4.0, 6.0]), np.array([0.0, 1.0, 3.0]), 50.0
        )
        np.testing.assert_array_equal(auto, [[0, 1]])

    def test_mu_is_measured_from_the_z_axis(self):
        coords = np.array([[10.0, 10.0, 10.0], [13.0, 10.0, 14.0]])  # s = 5, mu = 0.8
        auto, _ = count_pairs_smu(
            coords, np.zeros(2, np.int32), np.array([1.0, 6.0]), 10, 1.0, 50.0
        )
        expected = np.zeros((1, 10))
        expected[0, 8] = 1
        np.testing.assert_array_equal(auto, expected)

    def test_mu_equal_to_mu_max_is_dropped(self):
        coords = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 12.0]])  # mu = 1
        auto, _ = count_pairs_smu(coords, np.zeros(2, np.int32), np.array([1.0, 6.0]), 4, 1.0, 50.0)
        assert auto.sum() == 0

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points(self, n):
        coords = np.full((n, 3), 1.0)
        auto, cross = count_pairs_1d(coords, np.zeros(n, np.int32), np.array([0.0, 5.0]), 10.0)
        assert auto.sum() == cross.sum() == 0


class TestInvariances:
    @pytest.fixture
    def lattice(self):
        (coords, parts), box, edges = CATALOGUES["lattice"]
        return coords, parts, box, np.array(edges)

    def test_labels_only_split_auto_and_cross(self, catalogue):
        coords, parts, box, edges = catalogue
        auto, cross = count_pairs_1d(coords, parts, edges, box)
        total, none = count_pairs_1d(coords, np.zeros_like(parts), edges, box)
        np.testing.assert_array_equal(auto + cross, total)
        assert none.sum() == 0
        none, total = count_pairs_1d(coords, np.arange(len(parts), dtype=np.int32), edges, box)
        assert none.sum() == 0
        np.testing.assert_array_equal(auto + cross, total)

    def test_input_order_does_not_matter(self, catalogue):
        coords, parts, box, edges = catalogue
        perm = np.random.default_rng(0).permutation(len(coords))
        for count in (
            lambda c, p: count_pairs_1d(c, p, edges, box),
            lambda c, p: count_pairs_smu(c, p, edges, 4, 1.0, box),
        ):
            for a, b in zip(count(coords, parts), count(coords[perm], parts[perm])):
                np.testing.assert_array_equal(a, b)

    def test_periodic_translation_does_not_matter(self, lattice):
        """Shifting every point by the same vector (modulo the box) leaves
        all separations unchanged. Integer coordinates keep this exact."""
        coords, parts, box, edges = lattice
        shifted = np.mod(coords + np.array([5.0, 11.0, 3.0]), box)
        for count in (
            lambda c: count_pairs_1d(c, parts, edges, box),
            lambda c: count_pairs_2d(c, parts, edges, edges, box),
            lambda c: count_pairs_smu(c, parts, edges, 6, 1.0, box),
        ):
            for a, b in zip(count(coords), count(shifted)):
                np.testing.assert_array_equal(a, b)

    def test_radial_counts_are_isotropic(self, lattice):
        coords, parts, box, edges = lattice
        expected = count_pairs_1d(coords, parts, edges, box)
        for axes in ([1, 2, 0], [2, 0, 1], [0, 2, 1]):
            got = count_pairs_1d(np.ascontiguousarray(coords[:, axes]), parts, edges, box)
            np.testing.assert_array_equal(got, expected)

    def test_projected_counts_are_symmetric_in_x_and_y(self, lattice):
        coords, parts, box, edges = lattice
        swapped = np.ascontiguousarray(coords[:, [1, 0, 2]])
        pi_edges = np.array([0.0, 2.0, 4.0])
        np.testing.assert_array_equal(
            count_pairs_2d(coords, parts, edges, pi_edges, box),
            count_pairs_2d(swapped, parts, edges, pi_edges, box),
        )

    def test_adding_an_outer_bin_leaves_inner_bins_unchanged(self, catalogue):
        """The cell grid depends on r_max; the counts must not."""
        coords, parts, box, edges = catalogue
        wider = np.append(edges, 0.49 * box)
        for a, b in zip(
            count_pairs_1d(coords, parts, edges, box), count_pairs_1d(coords, parts, wider, box)
        ):
            np.testing.assert_array_equal(a, b[:-1])

    def test_mu_bins_marginalise_to_radial_counts(self, catalogue):
        """With mu_max = 1 every pair is kept except those along the line of
        sight (mu exactly 1), which fall on the excluded upper edge."""
        coords, parts, box, edges = catalogue
        auto, cross = count_pairs_smu(coords, parts, edges, 7, 1.0, box)
        radial_auto, radial_cross = count_pairs_1d(coords, parts, edges, box)

        _, _, d = ref.all_pairs(coords, box)
        r_bin = ref.bin_index((d**2).sum(axis=1), edges**2)
        on_axis = (d[:, 0] == 0) & (d[:, 1] == 0) & (r_bin >= 0)
        lost = np.bincount(r_bin[on_axis], minlength=len(edges) - 1)
        np.testing.assert_array_equal(
            auto.sum(axis=1) + cross.sum(axis=1) + lost, radial_auto + radial_cross
        )

    def test_counts_are_whole_numbers(self, catalogue):
        coords, parts, box, edges = catalogue
        for counts in count_pairs_2d(coords, parts, edges, edges, box):
            np.testing.assert_array_equal(counts, np.round(counts))
            assert counts.dtype == np.float64


def test_output_shapes():
    coords, parts = ref.uniform_catalogue(50, 20.0, 3, seed=0)
    edges = np.array([1.0, 2.0, 3.0, 4.0])
    for counts in count_pairs_1d(coords, parts, edges, 20.0):
        assert counts.shape == (3,)
    for counts in count_pairs_2d(coords, parts, edges, edges[:3], 20.0):
        assert counts.shape == (3, 2)
    for counts in count_pairs_smu(coords, parts, edges, 6, 1.0, 20.0):
        assert counts.shape == (3, 6)
