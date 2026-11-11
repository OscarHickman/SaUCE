//! Checks on user input, run before any counting.
//!
//! The counters assume valid input and would otherwise return wrong counts or
//! panic (which aborts the Python process, since the release profile sets
//! `panic = "abort"`), so every public entry point validates first. Each check
//! returns a message suitable for a Python `ValueError`.

use rayon::prelude::*;

use crate::npoint::MAX_ORDER;

pub type Check = Result<(), String>;

pub fn box_size(box_size: f64) -> Check {
    if box_size.is_finite() && box_size > 0.0 {
        Ok(())
    } else {
        Err(format!("box_size must be a positive finite number, got {box_size}"))
    }
}

pub fn coords_shape(n_columns: usize) -> Check {
    if n_columns == 3 {
        Ok(())
    } else {
        Err(format!("coords must have shape (N, 3), got {n_columns} columns"))
    }
}

pub fn matching_lengths(n_coords: usize, n_partitions: usize) -> Check {
    if n_coords == n_partitions {
        Ok(())
    } else {
        Err(format!(
            "coords has {n_coords} rows but partition_ids has {n_partitions} elements; \
             they must match"
        ))
    }
}

/// Coordinates must lie in the periodic box `[0, box_size)`. Checked in
/// parallel; the first offending value is reported.
pub fn coords_in_box(coords: &[f64], box_size: f64) -> Check {
    match coords.par_iter().find_first(|&&c| !(c.is_finite() && (0.0..box_size).contains(&c))) {
        None => Ok(()),
        Some(c) => Err(format!(
            "all coordinates must be finite and lie in [0, box_size) = [0, {box_size}); found {c}"
        )),
    }
}

/// Bin edges must be finite, non-negative and strictly increasing, with at
/// least one bin.
pub fn bin_edges(name: &str, edges: &[f64]) -> Check {
    if edges.len() < 2 {
        return Err(format!("{name} must have at least 2 edges (>=1 bin), got {}", edges.len()));
    }
    if let Some(e) = edges.iter().find(|e| !e.is_finite()) {
        return Err(format!("{name} must contain only finite values, found {e}"));
    }
    if edges[0] < 0.0 {
        return Err(format!("{name} must be non-negative, found {}", edges[0]));
    }
    if let Some(w) = edges.windows(2).find(|w| w[1] <= w[0]) {
        return Err(format!(
            "{name} must be strictly increasing, found {} followed by {}",
            w[0], w[1]
        ));
    }
    Ok(())
}

pub fn mu_binning(mu_max: f64, n_mu_bins: usize) -> Check {
    if !(mu_max.is_finite() && mu_max > 0.0) {
        return Err(format!("mu_max must be a positive finite number, got {mu_max}"));
    }
    if n_mu_bins < 1 {
        return Err("n_mu_bins must be >= 1".into());
    }
    Ok(())
}

/// Tuple orders are limited to [`MAX_ORDER`], the size of the fixed buffer
/// used to count distinct partitions within a tuple.
pub fn n_order(n_order: usize) -> Check {
    if (2..=MAX_ORDER).contains(&n_order) {
        Ok(())
    } else {
        Err(format!("n_order must be between 2 and {MAX_ORDER} (inclusive), got {n_order}"))
    }
}

pub fn n_ratio(n_ratio: usize) -> Check {
    if n_ratio >= 1 {
        Ok(())
    } else {
        Err("n_ratio must be >= 1".into())
    }
}

/// Partition ids index a dense `(k, k)` array, so must lie in `[0, k)`.
pub fn partition_ids_below<'a>(ids: impl IntoIterator<Item = &'a i32>, k: usize) -> Check {
    if k == 0 {
        return Err("k_partitions must be >= 1".into());
    }
    match ids.into_iter().find(|&&p| p < 0 || p as usize >= k) {
        None => Ok(()),
        Some(p) => {
            Err(format!("partition_ids must lie in [0, k_partitions) = [0, {k}); found {p}"))
        }
    }
}

/// `shard_idx` and `n_shards` must be given together, with
/// `0 <= shard_idx < n_shards`, and only for tuple orders of 3 or more.
pub fn sharding(shard_idx: Option<usize>, n_shards: Option<usize>, n_order: usize) -> Check {
    match (shard_idx, n_shards) {
        (None, None) => Ok(()),
        (Some(_), Some(0)) => Err("n_shards must be >= 1".into()),
        (Some(i), Some(n)) if i >= n => Err("shard_idx must be < n_shards".into()),
        (Some(_), Some(_)) if n_order == 2 => {
            Err("sharding is only supported for n_order >= 3".into())
        }
        (Some(_), Some(_)) => Ok(()),
        _ => Err("shard_idx and n_shards must be given together".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_input() {
        assert!(box_size(1.0).is_ok());
        assert!(coords_shape(3).is_ok());
        assert!(matching_lengths(4, 4).is_ok());
        assert!(coords_in_box(&[0.0, 0.5, 9.999], 10.0).is_ok());
        assert!(bin_edges("r", &[0.0, 1.0, 2.5]).is_ok());
        assert!(mu_binning(1.0, 1).is_ok());
        assert!(n_order(2).is_ok() && n_order(MAX_ORDER).is_ok());
        assert!(n_ratio(1).is_ok());
        assert!(partition_ids_below(&[0, 2, 1], 3).is_ok());
        assert!(sharding(None, None, 2).is_ok());
        assert!(sharding(Some(1), Some(2), 3).is_ok());
    }

    #[test]
    fn rejects_invalid_input() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(box_size(bad).is_err());
        }
        assert!(coords_shape(2).is_err());
        assert!(matching_lengths(4, 3).is_err());
        for bad in [-0.1, 10.0, f64::NAN] {
            assert!(coords_in_box(&[1.0, bad], 10.0).is_err());
        }
        for bad in [&[1.0][..], &[1.0, 1.0], &[2.0, 1.0], &[-1.0, 1.0], &[0.0, f64::INFINITY]] {
            assert!(bin_edges("r", bad).is_err(), "{bad:?}");
        }
        assert!(mu_binning(0.0, 3).is_err() && mu_binning(1.0, 0).is_err());
        assert!(n_order(1).is_err() && n_order(MAX_ORDER + 1).is_err());
        assert!(n_ratio(0).is_err());
        assert!(partition_ids_below(&[0, 3], 3).is_err());
        assert!(partition_ids_below(&[-1], 3).is_err());
        assert!(partition_ids_below(&[], 0).is_err());
        assert!(sharding(Some(0), None, 3).is_err());
        assert!(sharding(Some(0), Some(0), 3).is_err());
        assert!(sharding(Some(2), Some(2), 3).is_err());
        assert!(sharding(Some(0), Some(2), 2).is_err());
    }
}
