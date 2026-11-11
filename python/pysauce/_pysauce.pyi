"""Type stubs for the compiled counters. See each function's docstring."""

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I32 = NDArray[np.int32]

def count_pairs_1d(
    coords: F64, partition_ids: I32, r_bins: F64, box_size: float
) -> tuple[F64, F64]: ...
def count_pairs_1d_partition_matrix(
    coords: F64, partition_ids: I32, r_bins: F64, box_size: float, k_partitions: int
) -> tuple[F64, F64]: ...
def count_pairs_2d(
    coords: F64, partition_ids: I32, r_p_bins: F64, pi_bins: F64, box_size: float
) -> tuple[F64, F64]: ...
def count_pairs_smu(
    coords: F64, partition_ids: I32, s_bins: F64, n_mu_bins: int, mu_max: float, box_size: float
) -> tuple[F64, F64]: ...
def count_npoint(
    coords: F64,
    partition_ids: I32,
    r_bins: F64,
    box_size: float,
    n_order: int,
    shard_idx: int | None = None,
    n_shards: int | None = None,
) -> tuple[F64, F64]: ...
def count_npoint3_shape(
    coords: F64, partition_ids: I32, r_bins: F64, n_ratio: int, box_size: float
) -> tuple[F64, F64]: ...
def count_npoint3_smu(
    coords: F64, partition_ids: I32, r_bins: F64, n_mu_bins: int, mu_max: float, box_size: float
) -> tuple[F64, F64]: ...
