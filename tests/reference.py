"""Brute-force reference implementations and test catalogues.

Every function here enumerates all pairs or tuples directly, with no cell
grid or pruning, so it is slow but simple enough to check by eye. The binning
conventions match the compiled counters: separations use the nearest periodic
image, bins are half-open [lo, hi), and radial bins are compared on squared
separations.
"""

from itertools import combinations

import numpy as np


def uniform_catalogue(n, box, k, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box, (n, 3)), rng.integers(0, k, n).astype(np.int32)


def clustered_catalogue(n, box, k, seed, n_clusters=None, spread=None):
    """Points scattered around random centres, so small scales are populated."""
    rng = np.random.default_rng(seed)
    n_clusters = n_clusters or max(1, n // 20)
    spread = spread or box / 40
    centres = rng.uniform(0, box, (n_clusters, 3))
    coords = centres[rng.integers(0, n_clusters, n)] + rng.normal(0, spread, (n, 3))
    coords = np.mod(coords, box)
    coords[coords >= box] = 0.0  # np.mod can round up to box itself
    return coords, rng.integers(0, k, n).astype(np.int32)


def lattice_catalogue(n, box, k, seed):
    """Integer coordinates: all separations are exact and many fall exactly
    on integer bin edges."""
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, int(box), (n, 3)).astype(np.float64)
    return coords, rng.integers(0, k, n).astype(np.int32)


def min_image(d, box):
    return np.where(d > box / 2, d - box, np.where(d < -box / 2, d + box, d))


def all_pairs(coords, box):
    """Indices i < j of every pair and their separation vectors."""
    i, j = np.triu_indices(len(coords), k=1)
    return i, j, min_image(coords[j] - coords[i], box)


def bin_index(values, edges):
    """Half-open bin of each value, or -1 outside [edges[0], edges[-1])."""
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.where((values >= edges[0]) & (values < edges[-1]), idx, -1)


def mu_index(mu, mu_max, n_mu):
    with np.errstate(invalid="ignore"):
        idx = np.minimum((mu / mu_max * n_mu).astype(np.int64), n_mu - 1)
        return np.where(mu < mu_max, idx, -1)


def _split_counts(flat_bin, same_partition, n_bins):
    """Histogram flat bin indices (-1 = discard) into (auto, cross)."""
    keep = flat_bin >= 0
    auto = np.bincount(flat_bin[keep & same_partition], minlength=n_bins)
    cross = np.bincount(flat_bin[keep & ~same_partition], minlength=n_bins)
    return auto.astype(np.float64), cross.astype(np.float64)


def radial_pairs(coords, parts, r_bins, box):
    i, j, d = all_pairs(coords, box)
    r_bin = bin_index((d**2).sum(axis=1), np.asarray(r_bins) ** 2)
    return _split_counts(r_bin, parts[i] == parts[j], len(r_bins) - 1)


def projected_pairs(coords, parts, rp_bins, pi_bins, box):
    i, j, d = all_pairs(coords, box)
    rp = bin_index(d[:, 0] ** 2 + d[:, 1] ** 2, np.asarray(rp_bins) ** 2)
    pi = bin_index(np.abs(d[:, 2]), np.asarray(pi_bins))
    n_rp, n_pi = len(rp_bins) - 1, len(pi_bins) - 1
    flat = np.where((rp >= 0) & (pi >= 0), rp * n_pi + pi, -1)
    auto, cross = _split_counts(flat, parts[i] == parts[j], n_rp * n_pi)
    return auto.reshape(n_rp, n_pi), cross.reshape(n_rp, n_pi)


def redshift_space_pairs(coords, parts, s_bins, n_mu, mu_max, box):
    i, j, d = all_pairs(coords, box)
    s_sq = (d**2).sum(axis=1)
    s = bin_index(s_sq, np.asarray(s_bins) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = mu_index(np.abs(d[:, 2]) / np.sqrt(s_sq), mu_max, n_mu)
    n_s = len(s_bins) - 1
    flat = np.where((s >= 0) & (mu >= 0), s * n_mu + mu, -1)
    auto, cross = _split_counts(flat, parts[i] == parts[j], n_s * n_mu)
    return auto.reshape(n_s, n_mu), cross.reshape(n_s, n_mu)


def partition_matrix(coords, parts, r_bins, box, k):
    i, j, d = all_pairs(coords, box)
    r_bin = bin_index((d**2).sum(axis=1), np.asarray(r_bins) ** 2)
    n_r = len(r_bins) - 1
    auto = np.zeros((n_r, k))
    cross = np.zeros((n_r, k, k))
    for b, a, c in zip(r_bin, parts[i], parts[j]):
        if b < 0:
            continue
        if a == c:
            auto[b, a] += 1
        else:
            cross[b, min(a, c), max(a, c)] += 1
    return auto, cross


def _distance_matrix(coords, box):
    d = min_image(coords[None, :, :] - coords[:, None, :], box)
    return d, (d**2).sum(axis=2)


def _combinations(n, k):
    return np.array(list(combinations(range(n), k)), dtype=np.intp).reshape(-1, k)


def _n_distinct(labels):
    """Number of distinct values in each row."""
    labels = np.sort(labels, axis=1)
    return 1 + (np.diff(labels, axis=1) != 0).sum(axis=1)


def tuples(coords, parts, r_bins, box, n_order):
    """Counts of n_order-tuples by diameter, shape (n_order, n_r)."""
    _, dist_sq = _distance_matrix(coords, box)
    tup = _combinations(len(coords), n_order)
    sides = combinations(range(n_order), 2)
    diameter_sq = np.max([dist_sq[tup[:, a], tup[:, b]] for a, b in sides], axis=0)
    r_bin = bin_index(diameter_sq, np.asarray(r_bins) ** 2)
    keep = r_bin >= 0
    counts = np.zeros((n_order, len(r_bins) - 1))
    np.add.at(counts, (_n_distinct(parts[tup[keep]]) - 1, r_bin[keep]), 1)
    return counts


def triplets(coords, parts, r_bins, box, n_extra, extra_bin):
    """Triplet counts by longest side and a second quantity, shape
    (3, n_r, n_extra).

    ``extra_bin(longest_sq, shortest_sq, dz_longest)`` receives arrays with
    one entry per triplet and returns bin indices, -1 to discard.
    """
    d, dist_sq = _distance_matrix(coords, box)
    tup = _combinations(len(coords), 3)
    ends = np.array([(0, 1), (0, 2), (1, 2)])
    a, b = tup[:, ends[:, 0]], tup[:, ends[:, 1]]
    lengths = dist_sq[a, b]
    longest = np.argmax(lengths, axis=1)
    rows = np.arange(len(tup))
    longest_sq = lengths[rows, longest]
    dz_longest = d[a[rows, longest], b[rows, longest], 2]
    r_bin = bin_index(longest_sq, np.asarray(r_bins) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        extra = extra_bin(longest_sq, lengths.min(axis=1), dz_longest)
    keep = (r_bin >= 0) & (extra >= 0)
    counts = np.zeros((3, len(r_bins) - 1, n_extra))
    np.add.at(counts, (_n_distinct(parts[tup[keep]]) - 1, r_bin[keep], extra[keep]), 1)
    return counts
