"""Every pair counter against brute force over a broad sweep of geometries.

The traversal changes strategy with the search radius (columns up to a
quarter of the box, wrapped cells beyond) and treats particles near the box
faces and column boundaries specially, so the sweep covers radii on both
sides of the switch, box sizes over several orders of magnitude, and
catalogues built to put particles on those boundaries. The cases are drawn
with a fixed seed so that failures are reproducible.
"""

import itertools

import numpy as np
import pytest
import reference as ref
from pysauce import (
    count_pairs_1d,
    count_pairs_1d_partition_matrix,
    count_pairs_2d,
    count_pairs_smu,
)


def _in_box(coords, box):
    coords = np.mod(coords, box)
    coords[coords >= box] = 0.0
    return np.ascontiguousarray(coords)


def boundary_catalogue(n, box, k, seed, r_max):
    """Coordinates on and one ulp either side of the box faces and of the
    column boundaries the traversal uses for this search radius."""
    rng = np.random.default_rng(seed)
    n_columns = max(int(2 * box / r_max), 1)
    edges = np.arange(n_columns) * (box / n_columns)
    values = np.concatenate(
        [edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf), [box / 2]]
    )
    values = values[(values >= 0) & (values < box)]
    coords = rng.choice(values, (n, 3))
    return coords, rng.integers(0, k, n).astype(np.int32)


def duplicate_catalogue(n, box, k, seed):
    """Few distinct positions, each shared by many particles."""
    rng = np.random.default_rng(seed)
    sites = rng.uniform(0, box, (max(n // 25, 2), 3))
    return sites[rng.integers(0, len(sites), n)], rng.integers(0, k, n).astype(np.int32)


def slab_catalogue(n, box, k, seed):
    """Particles confined to a thin slab, so that most columns (or all but a
    few z ranges) are empty."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, box, (n, 3))
    axis = rng.integers(0, 3)
    coords[:, axis] = rng.uniform(0.4 * box, 0.42 * box, n)
    return _in_box(coords, box), rng.integers(0, k, n).astype(np.int32)


def make_catalogue(kind, n, box, k, seed, r_max):
    if kind == "uniform":
        return ref.uniform_catalogue(n, box, k, seed)
    if kind == "clustered":
        return ref.clustered_catalogue(n, box, k, seed, spread=box / 100)
    if kind == "lattice":
        scale = box / 16
        coords, parts = ref.lattice_catalogue(n, 16.0, k, seed)
        return coords * scale, parts
    if kind == "boundary":
        return boundary_catalogue(n, box, k, seed, r_max)
    if kind == "duplicates":
        return duplicate_catalogue(n, box, k, seed)
    if kind == "slab":
        return slab_catalogue(n, box, k, seed)
    raise ValueError(kind)


def make_edges(style, r_max):
    if style == "log":
        return np.geomspace(r_max / 200, r_max, 9)
    if style == "linear_from_zero":
        return np.linspace(0.0, r_max, 6)
    if style == "single":
        return np.array([0.3 * r_max, r_max])
    raise ValueError(style)


KINDS = ["uniform", "clustered", "lattice", "boundary", "duplicates", "slab"]
BOXES = [1.0, 37.5, 1000.0]
# Fractions of the box, either side of the switch to wrapped cells at 0.25.
R_FRACTIONS = [0.01, 0.06, 0.13, 0.2, 0.2499, 0.25, 0.2501, 0.3, 0.45, 0.6]
EDGE_STYLES = ["log", "linear_from_zero", "single"]


def _draw_cases(n_cases, seed=2024):
    rng = np.random.default_rng(seed)
    grid = list(itertools.product(KINDS, BOXES, R_FRACTIONS, EDGE_STYLES))
    picks = rng.choice(len(grid), n_cases, replace=False)
    cases = []
    for i, pick in enumerate(sorted(picks)):
        kind, box, frac, style = grid[pick]
        n = int(rng.integers(40, 420))
        k = int(rng.integers(1, 7))
        cases.append(
            pytest.param(
                (kind, box, frac, style, n, k, 1000 + i), id=f"{kind}-L{box}-r{frac}-{style}"
            )
        )
    # Every catalogue kind with the radius at the switch, whatever the draw.
    for j, kind in enumerate(KINDS):
        cases.append(
            pytest.param((kind, 37.5, 0.25, "log", 300, 3, 2000 + j), id=f"{kind}-at-switch")
        )
    return cases


CASES = _draw_cases(90)


@pytest.fixture(params=CASES)
def case(request):
    kind, box, frac, style, n, k, seed = request.param
    r_max = frac * box
    coords, parts = make_catalogue(kind, n, box, k, seed, r_max)
    return np.ascontiguousarray(coords), parts, box, make_edges(style, r_max), k


def assert_same(got, expected):
    for g, e in zip(got, expected):
        np.testing.assert_array_equal(g, e)


def test_radial(case):
    coords, parts, box, edges, _ = case
    assert_same(
        count_pairs_1d(coords, parts, edges, box), ref.radial_pairs(coords, parts, edges, box)
    )


def test_partition_matrix(case):
    coords, parts, box, edges, k = case
    assert_same(
        count_pairs_1d_partition_matrix(coords, parts, edges, box, k),
        ref.partition_matrix(coords, parts, edges, box, k),
    )


@pytest.mark.parametrize("pi_fraction", [0.05, 0.24, 0.26, 0.5])
def test_projected(case, pi_fraction):
    """The line-of-sight range is chosen independently of r_p, so the two
    can fall on different sides of the switch to wrapped cells."""
    coords, parts, box, edges, _ = case
    pi_edges = np.linspace(0.0, pi_fraction * box, 4)
    assert_same(
        count_pairs_2d(coords, parts, edges, pi_edges, box),
        ref.projected_pairs(coords, parts, edges, pi_edges, box),
    )


@pytest.mark.parametrize("n_mu, mu_max", [(1, 1.0), (6, 1.0), (4, 0.5), (9, 0.123)])
def test_redshift_space(case, n_mu, mu_max):
    coords, parts, box, edges, _ = case
    assert_same(
        count_pairs_smu(coords, parts, edges, n_mu, mu_max, box),
        ref.redshift_space_pairs(coords, parts, edges, n_mu, mu_max, box),
    )
