"""Expected pair counts for an unclustered (Poisson) catalogue in a periodic box.

For N points in a box of volume V, a bin covering volume V_bin of separation
space holds on average N(N-1)/2 * V_bin / V of the N(N-1)/2 distinct pairs.
These "RR" counts replace a random catalogue in the estimator
xi = DD / RR - 1.
"""

import numpy as np


def analytic_rr_1d(r_bins, box_size, n):
    """Expected pair counts in spherical shells of separation r.

    Parameters
    ----------
    r_bins : array_like, shape (n_r + 1,)
        Bin edges in r.
    box_size : float
        Side length of the periodic box.
    n : int
        Number of points.

    Returns
    -------
    ndarray, shape (n_r,)
    """
    r_bins = np.asarray(r_bins, dtype=np.float64)
    v_shell = (4.0 * np.pi / 3.0) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    return n * (n - 1) / 2.0 * v_shell / box_size**3


def analytic_rr_smu(s_bins, box_size, n, n_mu_bins, mu_max=1.0):
    """Expected pair counts in bins of separation s and mu.

    For random pairs mu = |cos(angle to the line of sight)| is uniform on
    [0, 1], so a mu bin of width d_mu holds a fraction d_mu of each shell.

    Parameters
    ----------
    s_bins : array_like, shape (n_s + 1,)
        Bin edges in s.
    box_size : float
        Side length of the periodic box.
    n : int
        Number of points.
    n_mu_bins : int
        Number of equal-width mu bins over [0, mu_max).
    mu_max : float, optional
        Upper mu limit.

    Returns
    -------
    ndarray, shape (n_s, n_mu_bins)
    """
    rr_shell = analytic_rr_1d(s_bins, box_size, n)
    mu_bin_width = mu_max / n_mu_bins
    return rr_shell[:, None] * np.full(n_mu_bins, mu_bin_width, dtype=np.float64)


def analytic_rr(rp_bins, pi_bins, box_size, n):
    """Expected pair counts in bins of transverse separation r_p and
    line-of-sight separation pi = |dz|.

    Each (r_p, pi) bin is an annulus of area pi (r_p,hi^2 - r_p,lo^2) and
    thickness d_pi on each side of the plane z = 0.

    Parameters
    ----------
    rp_bins : array_like, shape (n_rp + 1,)
        Bin edges in r_p.
    pi_bins : array_like, shape (n_pi + 1,)
        Bin edges in pi.
    box_size : float
        Side length of the periodic box.
    n : int
        Number of points.

    Returns
    -------
    ndarray, shape (n_rp, n_pi)
    """
    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    pi_bins = np.asarray(pi_bins, dtype=np.float64)
    annulus_area = np.pi * (rp_bins[1:] ** 2 - rp_bins[:-1] ** 2)
    v_bin = 2.0 * annulus_area[:, np.newaxis] * np.diff(pi_bins)[np.newaxis, :]
    return n * (n - 1) / 2.0 * v_bin / box_size**3
