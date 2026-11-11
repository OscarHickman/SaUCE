"""SaUCE: Sparse and Unbiased Clustering Estimator.

Pair and N-point counts for catalogues made of k independent partitions of a
periodic box, of which only m are available, with the weights that correct
for the missing partitions.

The ``count_*`` functions (compiled, in :mod:`pysauce._pysauce`) return raw
counts split by the number of distinct partitions each pair or tuple spans.
The ``compute_*`` functions apply the correction and form the correlation
functions; the ``analytic_rr*`` functions give the matching random counts.
"""

from importlib.metadata import PackageNotFoundError, version

from ._pysauce import (
    count_npoint,
    count_npoint3_shape,
    count_npoint3_smu,
    count_pairs_1d,
    count_pairs_1d_partition_matrix,
    count_pairs_2d,
    count_pairs_smu,
)
from .analytic import analytic_rr, analytic_rr_1d, analytic_rr_smu
from .estimators import (
    compute_2pcf,
    compute_npcf,
    compute_npcf3_shape,
    compute_npcf3_smu,
    compute_xi,
    compute_xi_smu,
    partition_weights,
)

try:
    __version__ = version("pysauce")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "analytic_rr",
    "analytic_rr_1d",
    "analytic_rr_smu",
    "compute_2pcf",
    "compute_npcf",
    "compute_npcf3_shape",
    "compute_npcf3_smu",
    "compute_xi",
    "compute_xi_smu",
    "count_npoint",
    "count_npoint3_shape",
    "count_npoint3_smu",
    "count_pairs_1d",
    "count_pairs_1d_partition_matrix",
    "count_pairs_2d",
    "count_pairs_smu",
    "partition_weights",
]
