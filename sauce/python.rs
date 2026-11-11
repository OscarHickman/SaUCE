//! Python entry points.
//!
//! Each function converts its NumPy arguments, validates them, runs a counter
//! from [`crate::pairs`] or [`crate::npoint`] with the GIL released, and
//! reshapes the result. The doc comments become the Python docstrings.

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::npoint::{self, Shard};
use crate::pairs;
use crate::validate::{self, Check};

type Array1Out<'py> = Bound<'py, PyArray1<f64>>;
type Array2Out<'py> = Bound<'py, PyArray2<f64>>;

fn check(result: Check) -> PyResult<()> {
    result.map_err(PyValueError::new_err)
}

/// Validated particle positions and partition labels.
struct Catalogue {
    coords: Vec<[f64; 3]>,
    partition: Vec<i32>,
}

fn read_catalogue(
    coords: &PyReadonlyArray2<f64>,
    partition_ids: &PyReadonlyArray1<i32>,
    box_size: f64,
) -> PyResult<Catalogue> {
    let coords = coords.as_array();
    let partition = partition_ids.as_array();
    check(validate::box_size(box_size))?;
    check(validate::coords_shape(coords.ncols()))?;
    check(validate::matching_lengths(coords.nrows(), partition.len()))?;
    let coords: Vec<[f64; 3]> = match coords.as_slice() {
        Some(flat) => flat.par_chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect(),
        None => coords.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
    };
    check(validate::coords_in_box(coords.as_flattened(), box_size))?;
    Ok(Catalogue { coords, partition: partition.to_vec() })
}

fn read_edges(name: &str, edges: &PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let edges = edges.as_array().to_vec();
    check(validate::bin_edges(name, &edges))?;
    Ok(edges)
}

fn reshape2(counts: Vec<f64>, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), counts).expect("counter returned the wrong number of bins")
}

/// Splits a flat `(n_classes, n_bins)` histogram into the per-class array
/// and its sum over classes.
fn classes_and_total(counts: Vec<f64>, n_classes: usize) -> (Array2<f64>, Array1<f64>) {
    let n_bins = counts.len() / n_classes;
    let by_class = reshape2(counts, n_classes, n_bins);
    let total = by_class.sum_axis(numpy::ndarray::Axis(0));
    (by_class, total)
}

/// Count pairs in bins of 3D separation r, split by partition membership.
///
/// Parameters
/// ----------
/// coords : ndarray, shape (N, 3), float64
///     Positions in the periodic box [0, box_size)^3.
/// partition_ids : ndarray, shape (N,), int32
///     Partition label of each particle.
/// r_bins : ndarray, shape (n_r + 1,), float64
///     Increasing, non-negative bin edges. Bins are half-open [lo, hi).
/// box_size : float
///     Side length of the periodic box.
///
/// Returns
/// -------
/// dd_auto, dd_cross : ndarray, shape (n_r,), float64
///     Numbers of unordered pairs whose members share a partition (auto) or
///     do not (cross). Separations use the nearest periodic image.
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_bins, box_size))]
pub fn count_pairs_1d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
) -> PyResult<(Array1Out<'py>, Array1Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let r_bins = read_edges("r_bins", &r_bins)?;
    let (auto, cross) =
        py.allow_threads(|| pairs::radial_counts(&cat.coords, &cat.partition, box_size, &r_bins));
    Ok((auto.into_pyarray(py), cross.into_pyarray(py)))
}

/// Count pairs in bins of r, resolved by the partitions of both members.
///
/// Gives the individual per-partition and per-partition-pair counts that
/// ``count_pairs_1d`` sums over, as needed for example by the variance of the
/// sparse-partition estimator.
///
/// Parameters
/// ----------
/// coords, partition_ids, r_bins, box_size
///     As for ``count_pairs_1d``.
/// k_partitions : int
///     Number of partitions k. Every partition id must lie in [0, k).
///
/// Returns
/// -------
/// c_auto : ndarray, shape (n_r, k), float64
///     ``c_auto[r, a]`` is the number of pairs within partition a.
/// c_cross : ndarray, shape (n_r, k, k), float64
///     ``c_cross[r, a, b]`` is the number of pairs between partitions a and b
///     for a < b. Entries on and below the diagonal are zero; add the
///     transpose for the symmetric matrix.
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_bins, box_size, k_partitions))]
pub fn count_pairs_1d_partition_matrix<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
    k_partitions: usize,
) -> PyResult<(Array2Out<'py>, Bound<'py, PyArray3<f64>>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let r_bins = read_edges("r_bins", &r_bins)?;
    check(validate::partition_ids_below(&cat.partition, k_partitions))?;
    let k = k_partitions;
    let (auto, cross) = py.allow_threads(|| {
        pairs::partition_pair_matrix(&cat.coords, &cat.partition, box_size, &r_bins, k)
    });
    let n_r = r_bins.len() - 1;
    let cross = Array3::from_shape_vec((n_r, k, k), cross)
        .expect("counter returned the wrong number of bins");
    Ok((reshape2(auto, n_r, k).into_pyarray(py), cross.into_pyarray(py)))
}

/// Count pairs in bins of transverse separation r_p and line-of-sight
/// separation pi, split by partition membership.
///
/// The line of sight is the z axis: r_p is the separation in the x-y plane
/// and pi = |dz|.
///
/// Parameters
/// ----------
/// coords, partition_ids, box_size
///     As for ``count_pairs_1d``.
/// r_p_bins : ndarray, shape (n_rp + 1,), float64
///     Bin edges in r_p.
/// pi_bins : ndarray, shape (n_pi + 1,), float64
///     Bin edges in pi.
///
/// Returns
/// -------
/// dd_auto, dd_cross : ndarray, shape (n_rp, n_pi), float64
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_p_bins, pi_bins, box_size))]
pub fn count_pairs_2d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_p_bins: PyReadonlyArray1<'py, f64>,
    pi_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
) -> PyResult<(Array2Out<'py>, Array2Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let rp_bins = read_edges("r_p_bins", &r_p_bins)?;
    let pi_bins = read_edges("pi_bins", &pi_bins)?;
    let (auto, cross) = py.allow_threads(|| {
        pairs::projected_counts(&cat.coords, &cat.partition, box_size, &rp_bins, &pi_bins)
    });
    let shape = (rp_bins.len() - 1, pi_bins.len() - 1);
    Ok((
        reshape2(auto, shape.0, shape.1).into_pyarray(py),
        reshape2(cross, shape.0, shape.1).into_pyarray(py),
    ))
}

/// Count pairs in bins of separation s and mu, split by partition membership.
///
/// mu = |dz| / s is the cosine of the angle between the pair separation and
/// the line of sight, which is the z axis.
///
/// Parameters
/// ----------
/// coords, partition_ids, box_size
///     As for ``count_pairs_1d``.
/// s_bins : ndarray, shape (n_s + 1,), float64
///     Bin edges in s.
/// n_mu_bins : int
///     Number of equal-width mu bins over [0, mu_max).
/// mu_max : float
///     Upper mu limit; pairs with mu >= mu_max are not counted.
///
/// Returns
/// -------
/// dd_auto, dd_cross : ndarray, shape (n_s, n_mu_bins), float64
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, s_bins, n_mu_bins, mu_max, box_size))]
pub fn count_pairs_smu<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    s_bins: PyReadonlyArray1<'py, f64>,
    n_mu_bins: usize,
    mu_max: f64,
    box_size: f64,
) -> PyResult<(Array2Out<'py>, Array2Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let s_bins = read_edges("s_bins", &s_bins)?;
    check(validate::mu_binning(mu_max, n_mu_bins))?;
    let (auto, cross) = py.allow_threads(|| {
        pairs::redshift_space_counts(
            &cat.coords,
            &cat.partition,
            box_size,
            &s_bins,
            n_mu_bins,
            mu_max,
        )
    });
    let n_s = s_bins.len() - 1;
    Ok((
        reshape2(auto, n_s, n_mu_bins).into_pyarray(py),
        reshape2(cross, n_s, n_mu_bins).into_pyarray(py),
    ))
}

/// Count N-tuples binned by their largest pairwise separation, split by the
/// number of distinct partitions they span.
///
/// A tuple is counted when all of its pairwise separations are below
/// ``r_bins[-1]``, and is binned by the largest of them.
///
/// Parameters
/// ----------
/// coords, partition_ids, r_bins, box_size
///     As for ``count_pairs_1d``.
/// n_order : int
///     Tuple order N, from 2 to 8.
/// shard_idx, n_shards : int, optional
///     Count only the share ``shard_idx`` of ``n_shards`` equal shares of the
///     work (N >= 3 only). Summing the results of all shards gives the full
///     count, which lets a large calculation be split across separate jobs.
///
/// Returns
/// -------
/// t_by_s : ndarray, shape (n_order, n_r), float64
///     ``t_by_s[s - 1]`` counts tuples spanning s distinct partitions.
/// t_total : ndarray, shape (n_r,), float64
///     Sum of ``t_by_s`` over s.
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_bins, box_size, n_order, shard_idx=None, n_shards=None))]
#[allow(clippy::too_many_arguments)]
pub fn count_npoint<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
    n_order: usize,
    shard_idx: Option<usize>,
    n_shards: Option<usize>,
) -> PyResult<(Array2Out<'py>, Array1Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let r_bins = read_edges("r_bins", &r_bins)?;
    check(validate::n_order(n_order))?;
    check(validate::sharding(shard_idx, n_shards, n_order))?;
    let shard = shard_idx.zip(n_shards).map(|(index, count)| Shard { index, count });

    let counts = py.allow_threads(|| {
        if n_order == 2 {
            let (mut auto, cross) =
                pairs::radial_counts(&cat.coords, &cat.partition, box_size, &r_bins);
            auto.extend(cross);
            auto
        } else {
            npoint::tuple_counts(&cat.coords, &cat.partition, box_size, &r_bins, n_order, shard)
        }
    });
    let (by_class, total) = classes_and_total(counts, n_order);
    Ok((by_class.into_pyarray(py), total.into_pyarray(py)))
}

/// Count triplets binned by their longest side and by triangle shape.
///
/// The shape is the ratio of the shortest to the longest side, from near 0
/// (squeezed) to 1 (equilateral), in ``n_ratio`` equal bins over [0, 1].
/// Summing over the ratio bins gives ``count_npoint(..., n_order=3)``.
///
/// Parameters
/// ----------
/// coords, partition_ids, r_bins, box_size
///     As for ``count_pairs_1d``.
/// n_ratio : int
///     Number of shape bins.
///
/// Returns
/// -------
/// t_by_s : ndarray, shape (3, n_r * n_ratio), float64
///     Counts by number of distinct partitions s = 1, 2, 3; the second axis
///     is ``r_bin * n_ratio + ratio_bin``.
/// t_total : ndarray, shape (n_r * n_ratio,), float64
///     Sum of ``t_by_s`` over s.
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_bins, n_ratio, box_size))]
pub fn count_npoint3_shape<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    n_ratio: usize,
    box_size: f64,
) -> PyResult<(Array2Out<'py>, Array1Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let r_bins = read_edges("r_bins", &r_bins)?;
    check(validate::n_ratio(n_ratio))?;
    let counts = py.allow_threads(|| {
        npoint::triplet_counts_by_shape(&cat.coords, &cat.partition, box_size, &r_bins, n_ratio)
    });
    let (by_class, total) = classes_and_total(counts, 3);
    Ok((by_class.into_pyarray(py), total.into_pyarray(py)))
}

/// Count triplets binned by their longest side and its orientation to the
/// line of sight.
///
/// mu = |dz| / r is the cosine of the angle between the triangle's longest
/// side and the line of sight (the z axis), in ``n_mu_bins`` equal bins over
/// [0, mu_max). If two sides tie for longest, one is chosen by a fixed
/// internal order. With ``mu_max = 1`` summing over the mu bins gives
/// ``count_npoint(..., n_order=3)``.
///
/// Parameters
/// ----------
/// coords, partition_ids, r_bins, box_size
///     As for ``count_pairs_1d``.
/// n_mu_bins : int
///     Number of mu bins.
/// mu_max : float
///     Upper mu limit; triplets with mu >= mu_max are not counted.
///
/// Returns
/// -------
/// t_by_s : ndarray, shape (3, n_r * n_mu_bins), float64
///     Counts by number of distinct partitions s = 1, 2, 3; the second axis
///     is ``r_bin * n_mu_bins + mu_bin``.
/// t_total : ndarray, shape (n_r * n_mu_bins,), float64
///     Sum of ``t_by_s`` over s.
#[pyfunction]
#[pyo3(signature = (coords, partition_ids, r_bins, n_mu_bins, mu_max, box_size))]
pub fn count_npoint3_smu<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    partition_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    n_mu_bins: usize,
    mu_max: f64,
    box_size: f64,
) -> PyResult<(Array2Out<'py>, Array1Out<'py>)> {
    let cat = read_catalogue(&coords, &partition_ids, box_size)?;
    let r_bins = read_edges("r_bins", &r_bins)?;
    check(validate::mu_binning(mu_max, n_mu_bins))?;
    let counts = py.allow_threads(|| {
        npoint::triplet_counts_by_mu(
            &cat.coords,
            &cat.partition,
            box_size,
            &r_bins,
            n_mu_bins,
            mu_max,
        )
    });
    let (by_class, total) = classes_and_total(counts, 3);
    Ok((by_class.into_pyarray(py), total.into_pyarray(py)))
}
