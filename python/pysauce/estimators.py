"""Correlation-function estimators with the sparse-partition correction.

The full catalogue is the union of k disjoint partitions, each an independent
sample of the whole box at 1/k of the total density, and only m of them are
available. A pair or N-tuple whose members span s distinct partitions is
observed only if all s partitions are among the m selected, which for a
random selection happens with probability

    P_s = m (m - 1) ... (m - s + 1) / [k (k - 1) ... (k - s + 1)].

Weighting the observed counts of each class by w_s = 1 / P_s therefore gives
an unbiased estimate of the counts in the full catalogue ("scale-up").
Multiplying every weight by (m / k)^N instead estimates the full counts
diluted to the observed density ("scale-down"), which is what random counts
built from the observed number of points expect.

For pairs the s = 1 and s = 2 classes are the auto and cross counts, and the
weights are usually written alpha = w_1 and beta = w_2.

With m = k every weight is 1 and the estimate is the count of the full box.
With m < k it is unbiased, provided m >= N: with fewer selected partitions,
tuples spanning more than m partitions could never be observed, so no
weighting could recover them. Every estimator therefore requires
N <= m <= k, and rejects catalogues with more than m distinct partition
labels.
"""

from math import comb, perm

import numpy as np

from . import _pysauce
from .analytic import analytic_rr, analytic_rr_1d, analytic_rr_smu

SCALE_METHODS = ("scale-up", "scale-down")


def partition_weights(n_partitions, n_partitions_selected, n_order, scale_method):
    """Weights w_s for tuples spanning s = 1 ... n_order distinct partitions.

    Parameters
    ----------
    n_partitions : int
        Total number of partitions k.
    n_partitions_selected : int
        Number of selected partitions m.
    n_order : int
        Tuple order N (2 for pairs).
    scale_method : {"scale-up", "scale-down"}
        "scale-up" estimates the counts of the full catalogue; "scale-down"
        multiplies those by (m / k)^N, the counts at the observed density.

    Returns
    -------
    ndarray, shape (n_order,)
        ``weights[s - 1]`` is w_s.

    Raises
    ------
    ValueError
        Unless ``n_order <= n_partitions_selected <= n_partitions``: with
        m < N some tuples could never be observed, so no weights give an
        unbiased estimate.
    """
    k, m = n_partitions, n_partitions_selected
    _check_selection(k, m, n_order)
    if scale_method not in SCALE_METHODS:
        raise ValueError(f"scale_method must be one of {SCALE_METHODS}, got {scale_method!r}")
    # Ratios of exact integers, so each weight is correctly rounded.
    if scale_method == "scale-up":
        weights = [perm(k, s) / perm(m, s) for s in range(1, n_order + 1)]
    else:
        weights = [
            perm(k, s) * m**n_order / (perm(m, s) * k**n_order) for s in range(1, n_order + 1)
        ]
    return np.array(weights, dtype=np.float64)


def _check_selection(n_partitions, n_partitions_selected, n_order):
    """Require whole numbers with 1 <= n_order <= m <= k."""
    k, m = n_partitions, n_partitions_selected
    for name, value in [("n_partitions", k), ("n_partitions_selected", m), ("n_order", n_order)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {value!r}")
    if n_order < 1:
        raise ValueError(f"n_order must be >= 1, got {n_order}")
    if m > k:
        raise ValueError(f"n_partitions_selected ({m}) cannot exceed n_partitions ({k})")
    if m < n_order:
        raise ValueError(
            f"n_partitions_selected must be >= the order of the statistic ({n_order}), got {m}: "
            f"tuples spanning more than {m} partitions could never be observed"
        )


def _check_labels(partition_ids, n_partitions_selected):
    """A catalogue of m selected partitions has at most m distinct labels."""
    n_labels = len(np.unique(partition_ids))
    if n_labels > n_partitions_selected:
        raise ValueError(
            f"partition_ids has {n_labels} distinct labels, more than "
            f"n_partitions_selected ({n_partitions_selected})"
        )


def _as_edges(edges):
    return np.ascontiguousarray(edges, dtype=np.float64)


def _as_catalogue(coords, partition_ids, n_partitions_selected):
    """Convert inputs to the float64 / int32 arrays the counters require."""
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    ids = np.asarray(partition_ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"partition_ids must be integers, got dtype {ids.dtype}")
    info = np.iinfo(np.int32)
    if ids.size and (ids.min() < info.min or ids.max() > info.max):
        raise ValueError("partition_ids must fit in a 32-bit integer")
    _check_labels(ids, n_partitions_selected)
    return coords, np.ascontiguousarray(ids, dtype=np.int32)


def _weighted_sum(weights, counts_by_class):
    """Sum over classes s of w_s * counts[s - 1]."""
    return sum(w * c for w, c in zip(weights, counts_by_class))


def compute_xi(coords, partition_ids, r_bins, box_size, n_partitions, n_partitions_selected):
    """Real-space correlation function xi(r) of a sparse-partition catalogue.

    Uses the scale-down weights, so the corrected pair counts are compared
    with analytic random counts for the observed number of points.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Positions in the periodic box [0, box_size)^3.
    partition_ids : array_like, shape (N,)
        Partition label of each point.
    r_bins : array_like, shape (n_r + 1,)
        Bin edges in r.
    box_size : float
        Side length of the periodic box.
    n_partitions : int
        Total number of partitions k.
    n_partitions_selected : int
        Number of partitions m present in ``coords``; at least 2 and at most
        ``n_partitions``.

    Returns
    -------
    dict
        ``dd_auto``, ``dd_cross``: raw pair counts; ``dd_corr``: corrected
        counts ``alpha * dd_auto + beta * dd_cross``; ``rr``: analytic random
        counts; ``xi``: ``dd_corr / rr - 1``; ``r_mid``: geometric bin
        centres. All have shape (n_r,).
    """
    _check_selection(n_partitions, n_partitions_selected, 2)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    r_bins = _as_edges(r_bins)
    dd_auto, dd_cross = _pysauce.count_pairs_1d(coords, partition_ids, r_bins, box_size)

    alpha, beta = partition_weights(n_partitions, n_partitions_selected, 2, "scale-down")
    dd_corr = alpha * dd_auto + beta * dd_cross
    rr = analytic_rr_1d(r_bins, box_size, len(coords))
    with np.errstate(divide="ignore", invalid="ignore"):  # empty bins give nan
        xi = dd_corr / rr - 1.0
    return {
        "xi": xi,
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
        "r_mid": np.sqrt(r_bins[:-1] * r_bins[1:]),
    }


def compute_2pcf(
    coords,
    partition_ids,
    rp_bins,
    pi_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
    n_total=None,
):
    """Anisotropic correlation function xi(r_p, pi) and projected w_p(r_p).

    The line of sight is the z axis. Uses the scale-up weights, so the
    corrected pair counts estimate those of the full catalogue and are
    compared with random counts for ``n_total`` points.

    Parameters
    ----------
    coords, partition_ids, box_size, n_partitions, n_partitions_selected
        As for :func:`compute_xi`.
    rp_bins : array_like, shape (n_rp + 1,)
        Bin edges in transverse separation r_p.
    pi_bins : array_like, shape (n_pi + 1,)
        Bin edges in line-of-sight separation pi.
    n_total : int, optional
        Number of points in the full catalogue. Defaults to
        ``round(N * k / m)``.

    Returns
    -------
    dict
        ``dd_auto``, ``dd_cross``, ``dd_corr``, ``rr``, ``xi`` with shape
        (n_rp, n_pi), and ``wp = 2 * sum_pi xi * d_pi`` with shape (n_rp,).
    """
    _check_selection(n_partitions, n_partitions_selected, 2)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    rp_bins, pi_bins = _as_edges(rp_bins), _as_edges(pi_bins)
    dd_auto, dd_cross = _pysauce.count_pairs_2d(coords, partition_ids, rp_bins, pi_bins, box_size)

    alpha, beta = partition_weights(n_partitions, n_partitions_selected, 2, "scale-up")
    dd_corr = alpha * dd_auto + beta * dd_cross
    if n_total is None:
        n_total = int(round(len(coords) * n_partitions / n_partitions_selected))
    rr = analytic_rr(rp_bins, pi_bins, box_size, n_total)
    with np.errstate(divide="ignore", invalid="ignore"):  # empty bins give nan
        xi = dd_corr / rr - 1.0
    delta_pi = np.diff(pi_bins)
    return {
        "xi": xi,
        "wp": 2.0 * np.sum(xi * delta_pi[np.newaxis, :], axis=1),
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
    }


def compute_xi_smu(
    coords,
    partition_ids,
    s_bins,
    box_size,
    *,
    n_partitions,
    n_partitions_selected,
    n_mu_bins,
    mu_max=1.0,
):
    """Redshift-space correlation function xi(s, mu) and its multipoles.

    The line of sight is the z axis and mu = |dz| / s. Uses the scale-down
    weights, as :func:`compute_xi`. The monopole and quadrupole are the
    midpoint-rule integrals

        xi_0(s) = sum_mu xi(s, mu) d_mu,
        xi_2(s) = 5 * sum_mu xi(s, mu) P_2(mu) d_mu,

    which are the Legendre multipoles only for ``mu_max = 1``. Empty bins
    (``nan``) are skipped.

    Parameters
    ----------
    coords, partition_ids, box_size, n_partitions, n_partitions_selected
        As for :func:`compute_xi`.
    s_bins : array_like, shape (n_s + 1,)
        Bin edges in s.
    n_mu_bins : int
        Number of equal-width mu bins over [0, mu_max).
    mu_max : float, optional
        Upper mu limit.

    Returns
    -------
    dict
        ``dd_auto``, ``dd_cross``, ``dd_corr``, ``rr`` and ``xi_grid`` (also
        available as ``xi_smu``) with shape (n_s, n_mu_bins); ``xi0``, ``xi2``
        and ``s_mid`` (geometric bin centres) with shape (n_s,); ``mu_mid``
        with shape (n_mu_bins,).
    """
    _check_selection(n_partitions, n_partitions_selected, 2)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    s_bins = _as_edges(s_bins)
    dd_auto, dd_cross = _pysauce.count_pairs_smu(
        coords, partition_ids, s_bins, n_mu_bins, mu_max, box_size
    )

    alpha, beta = partition_weights(n_partitions, n_partitions_selected, 2, "scale-down")
    dd_corr = alpha * dd_auto + beta * dd_cross
    rr = analytic_rr_smu(s_bins, box_size, len(coords), n_mu_bins, mu_max)
    with np.errstate(divide="ignore", invalid="ignore"):  # empty bins give nan
        xi_grid = dd_corr / rr - 1.0

    mu_bins = np.linspace(0.0, mu_max, n_mu_bins + 1)
    mu_mid = 0.5 * (mu_bins[:-1] + mu_bins[1:])
    d_mu = mu_max / n_mu_bins
    legendre_2 = 0.5 * (3.0 * mu_mid**2 - 1.0)
    return {
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
        "xi_grid": xi_grid,
        "xi_smu": xi_grid,
        "xi0": np.nansum(xi_grid * d_mu, axis=1),
        "xi2": 5.0 * np.nansum(xi_grid * legendre_2 * d_mu, axis=1),
        "s_mid": np.sqrt(s_bins[:-1] * s_bins[1:]),
        "mu_mid": mu_mid,
    }


def compute_npcf(
    coords,
    partition_ids,
    r_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
    n_order,
    scale_method="scale-up",
):
    """Sparse-partition-corrected N-tuple counts binned by tuple diameter.

    A tuple is counted when all its pairwise separations are below
    ``r_bins[-1]`` and binned by the largest of them.

    Parameters
    ----------
    coords, partition_ids, r_bins, box_size, n_partitions, n_partitions_selected
        As for :func:`compute_xi`; ``n_partitions_selected`` must be at least
        ``n_order``.
    n_order : int
        Tuple order N, from 2 to 8.
    scale_method : {"scale-up", "scale-down"}, optional
        See :func:`partition_weights`.

    Returns
    -------
    dict
        ``t_total``: raw counts, shape (n_r,); ``weights``: w_s for
        s = 1 ... N; ``t_corr``: ``sum_s w_s * counts_s``, shape (n_r,);
        ``zeta``: ``t_corr`` divided by the number of N-tuples in the
        observed catalogue, C(N_points, n_order).
    """
    weights = partition_weights(n_partitions, n_partitions_selected, n_order, scale_method)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    r_bins = _as_edges(r_bins)
    t_by_s, t_total = _pysauce.count_npoint(coords, partition_ids, r_bins, box_size, n_order)

    t_corr = _weighted_sum(weights, t_by_s)
    n_points = len(coords)
    n_tuples = comb(n_points, n_order) if n_points >= n_order else 1.0
    return {
        "t_corr": t_corr,
        "t_total": t_total,
        "weights": weights,
        "zeta": t_corr / n_tuples,
    }


def _corrected_triplets(weights, counts, n_extra):
    """Result of a binned-triplet estimator from its raw counts."""
    t_by_s, t_total = counts
    shape = (-1, n_extra)
    return {
        "t_corr": _weighted_sum(weights, t_by_s).reshape(shape),
        "t_total": np.asarray(t_total).reshape(shape),
        "weights": weights,
    }


def compute_npcf3_shape(
    coords,
    partition_ids,
    r_bins,
    n_ratio,
    box_size,
    n_partitions,
    n_partitions_selected,
    scale_method="scale-up",
):
    """Sparse-partition-corrected triplet counts binned by size and shape.

    Each triplet is binned by its longest side r_max (``r_bins``) and by the
    ratio r_min / r_max of its shortest to longest side, in ``n_ratio`` equal
    bins over [0, 1]: ratios near 0 are squeezed triangles, 1 is equilateral.
    The ratio does not determine the triangle uniquely. Summing over the ratio
    axis gives :func:`compute_npcf` with ``n_order=3``.

    Parameters
    ----------
    coords, partition_ids, r_bins, box_size, n_partitions, n_partitions_selected
        As for :func:`compute_npcf`; ``n_partitions_selected`` must be >= 3.
    n_ratio : int
        Number of shape bins.
    scale_method : {"scale-up", "scale-down"}, optional
        See :func:`partition_weights`.

    Returns
    -------
    dict
        ``t_corr`` and ``t_total`` with shape (n_r, n_ratio), and the
        weights for s = 1, 2, 3. No normalised ``zeta`` is returned, since
        random triplet counts in these bins are not available analytically.
    """
    weights = partition_weights(n_partitions, n_partitions_selected, 3, scale_method)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    counts = _pysauce.count_npoint3_shape(
        coords, partition_ids, _as_edges(r_bins), int(n_ratio), box_size
    )
    return _corrected_triplets(weights, counts, n_ratio)


def compute_npcf3_smu(
    coords,
    partition_ids,
    r_bins,
    n_mu_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
    mu_max=1.0,
    scale_method="scale-up",
):
    """Sparse-partition-corrected triplet counts binned by size and
    orientation.

    Each triplet is binned by its longest side r_max (``r_bins``) and by mu,
    the cosine of the angle between that side and the line of sight (the z
    axis), in ``n_mu_bins`` equal bins over [0, mu_max). The weights depend
    only on the number of partitions a triplet spans, so they are the same as
    for :func:`compute_npcf3_shape`. With ``mu_max = 1`` summing over the mu
    axis gives :func:`compute_npcf` with ``n_order=3``.

    Parameters
    ----------
    coords, partition_ids, r_bins, box_size, n_partitions, n_partitions_selected
        As for :func:`compute_npcf`; ``n_partitions_selected`` must be >= 3.
    n_mu_bins : int
        Number of mu bins.
    mu_max : float, optional
        Upper mu limit; triplets with mu >= mu_max are not counted.
    scale_method : {"scale-up", "scale-down"}, optional
        See :func:`partition_weights`.

    Returns
    -------
    dict
        ``t_corr`` and ``t_total`` with shape (n_r, n_mu_bins), and the
        weights for s = 1, 2, 3.
    """
    weights = partition_weights(n_partitions, n_partitions_selected, 3, scale_method)
    coords, partition_ids = _as_catalogue(coords, partition_ids, n_partitions_selected)
    counts = _pysauce.count_npoint3_smu(
        coords, partition_ids, _as_edges(r_bins), int(n_mu_bins), float(mu_max), box_size
    )
    return _corrected_triplets(weights, counts, n_mu_bins)
