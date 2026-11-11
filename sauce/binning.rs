//! Histogram bin lookup.
//!
//! All bins are half-open, `[edges[b], edges[b + 1])`. Radial bins are looked
//! up on squared distances against squared edges, which avoids a square root
//! per pair and gives the same bin because squaring is monotonic on
//! non-negative numbers.

/// Squares of `values`, for binning squared separations.
pub fn squares(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&v| v * v).collect()
}

/// Index `b` with `edges[b] <= value < edges[b + 1]`, by binary search. The
/// tests use it as the reference for [`TableBins::find`].
#[cfg(test)]
pub fn find_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value >= edges[edges.len() - 1] {
        return None;
    }
    Some(edges.partition_point(|&e| e <= value) - 1)
}

/// Bin of `mu` among `n_mu` equal-width bins covering `[0, mu_max)`.
#[inline(always)]
pub fn mu_bin(mu: f64, mu_max: f64, n_mu: usize) -> Option<usize> {
    if mu < mu_max {
        Some(((mu / mu_max * n_mu as f64) as usize).min(n_mu - 1))
    } else {
        None
    }
}

/// [`mu_bin`] as a float, `-1.0` for `None`. It has no branches, so that a
/// loop over many values compiles to SIMD instructions.
#[inline(always)]
pub fn mu_bin_value(mu: f64, mu_max: f64, n_mu: usize) -> f64 {
    let bin = (mu / mu_max * n_mu as f64).floor().min((n_mu - 1) as f64);
    if mu < mu_max {
        bin
    } else {
        -1.0
    }
}

/// Number of slots in a [`BinTable`].
const LOOKUP_SIZE: usize = 4096;

/// Table entry for a slot that contains a bin edge.
const STRADDLES_EDGE: i32 = -2;

/// Table entry for a slot entirely below the first bin edge.
const BELOW_FIRST_EDGE: i32 = -1;

/// Storage for a constant-time bin lookup, used for every separation that
/// passes the range test in the inner loops.
///
/// `[0, edges[last]]` is divided into [`LOOKUP_SIZE`] equal slots. A slot that
/// falls inside a single bin stores that bin; a slot containing a bin edge is
/// marked [`STRADDLES_EDGE`] and resolved by exact binary search, so the
/// result always equals a binary search of the edges. Radial bins are looked
/// up on squared separations, so their table is built from the squared edges.
pub struct BinTable {
    edges: Vec<f64>,
    table: Box<[i32; LOOKUP_SIZE]>,
    slots_per_unit: f64,
}

impl BinTable {
    pub fn new(edges: &[f64]) -> Self {
        let slots_per_unit = (LOOKUP_SIZE - 1) as f64 / edges[edges.len() - 1];
        let table = std::array::from_fn(|slot| {
            let lo = exact_bin(slot as f64 / slots_per_unit, edges);
            let hi = exact_bin((slot + 1) as f64 / slots_per_unit, edges);
            if lo == hi {
                lo
            } else {
                STRADDLES_EDGE
            }
        });
        BinTable { edges: edges.to_vec(), table: Box::new(table), slots_per_unit }
    }

    /// The lookup itself. It borrows the table and is `Copy`, so a hot loop
    /// can hold it by value and keep its fields in registers.
    pub fn bins(&self) -> TableBins<'_> {
        TableBins {
            edges: &self.edges,
            table: &self.table,
            slots_per_unit: self.slots_per_unit,
            lo: self.edges[0],
            hi: self.edges[self.edges.len() - 1],
        }
    }
}

/// Constant-time bin lookup; see [`BinTable`].
#[derive(Clone, Copy)]
pub struct TableBins<'a> {
    edges: &'a [f64],
    table: &'a [i32; LOOKUP_SIZE],
    slots_per_unit: f64,
    lo: f64,
    hi: f64,
}

impl<'a> TableBins<'a> {
    /// Whether `value` lies in `[edges[0], edges[last])`, i.e. in some bin.
    /// Both comparisons are always made (`&`, not `&&`), which lets loops
    /// calling this compile to SIMD instructions.
    #[inline(always)]
    pub fn contains(self, value: f64) -> bool {
        (value >= self.lo) & (value < self.hi)
    }

    /// Upper end of the binned range (exclusive).
    pub fn hi(self) -> f64 {
        self.hi
    }

    /// The bin edges.
    pub fn edges(self) -> &'a [f64] {
        self.edges
    }

    pub fn n_bins(self) -> usize {
        self.edges.len() - 1
    }

    /// Bin of a `value` in `[0, hi)`, where `hi` is [`Self::hi`].
    #[inline(always)]
    pub fn find(self, value: f64) -> Option<usize> {
        let mut bin = self.table[self.slot(value)];
        if bin == STRADDLES_EDGE {
            bin = exact_bin(value, self.edges);
        }
        (bin >= 0).then_some(bin as usize)
    }

    /// Sets `bins[j]` to the bin of `values[j]`, for values that
    /// [`Self::contains`]. The table lookups compile to SIMD instructions;
    /// values in slots containing an edge are then resolved one at a time.
    #[inline(always)]
    pub fn find_all(self, values: &[f64], bins: &mut [i32]) {
        for (bin, &value) in bins.iter_mut().zip(values) {
            *bin = self.table[self.slot(value)];
        }
        for (bin, &value) in bins.iter_mut().zip(values) {
            if *bin == STRADDLES_EDGE {
                *bin = exact_bin(value, self.edges);
            }
        }
    }

    /// Table slot of a `value` in `[0, hi)`.
    #[inline(always)]
    fn slot(self, value: f64) -> usize {
        debug_assert!((0.0..self.hi).contains(&value));
        // SAFETY: `value` is finite and in `[0, hi)`, so the product lies in
        // `[0, LOOKUP_SIZE)` (up to rounding, which the clamp absorbs) and
        // converts exactly to an integer. The unchecked conversion avoids the
        // saturation handling of `as`, which is measurable in the inner loop.
        let slot = unsafe { (value * self.slots_per_unit).to_int_unchecked::<i32>() };
        slot.clamp(0, LOOKUP_SIZE as i32 - 1) as usize
    }
}

/// Bin of `value` assuming it is below the last edge, or
/// [`BELOW_FIRST_EDGE`]. Values at or above the last edge map to the number
/// of bins, which only happens for table slots beyond the binnable range.
fn exact_bin(value: f64, edges: &[f64]) -> i32 {
    match edges.partition_point(|&e| e <= value) {
        0 => BELOW_FIRST_EDGE,
        idx => (idx - 1) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_bin_uses_half_open_bins() {
        let edges = [1.0, 2.0, 4.0];
        assert_eq!(find_bin(0.999, &edges), None);
        assert_eq!(find_bin(1.0, &edges), Some(0));
        assert_eq!(find_bin(1.999, &edges), Some(0));
        assert_eq!(find_bin(2.0, &edges), Some(1));
        assert_eq!(find_bin(3.999, &edges), Some(1));
        assert_eq!(find_bin(4.0, &edges), None);
    }

    #[test]
    fn mu_bin_covers_zero_to_mu_max() {
        assert_eq!(mu_bin(0.0, 1.0, 4), Some(0));
        assert_eq!(mu_bin(0.25, 1.0, 4), Some(1));
        assert_eq!(mu_bin(0.999_999, 1.0, 4), Some(3));
        assert_eq!(mu_bin(1.0, 1.0, 4), None);
        assert_eq!(mu_bin(0.5, 0.5, 4), None);
        assert_eq!(mu_bin(0.3, 0.5, 5), Some(3));
    }

    #[test]
    fn mu_bin_value_agrees_with_mu_bin() {
        let mut values: Vec<f64> = (0..=10_000).map(|i| i as f64 / 8_000.0).collect();
        values.extend([f64::NAN, f64::INFINITY, 0.1 + 0.2, 1.0 - f64::EPSILON / 2.0]);
        for (mu_max, n_mu) in [(1.0, 1), (1.0, 50), (0.7, 9), (0.3, 7), (1.0, 3)] {
            let edges = (0..=n_mu).map(|j| j as f64 * mu_max / n_mu as f64);
            let near_edges = edges.flat_map(|e: f64| {
                [e, f64::from_bits(e.to_bits().saturating_sub(1)), f64::from_bits(e.to_bits() + 1)]
            });
            for mu in values.iter().copied().chain(near_edges) {
                let expected = mu_bin(mu, mu_max, n_mu).map_or(-1.0, |b| b as f64);
                assert_eq!(mu_bin_value(mu, mu_max, n_mu), expected, "mu = {mu}");
            }
        }
    }

    /// Values spread over `[0, hi)` plus the neighbours of every edge.
    fn probe_values(edges: &[f64]) -> Vec<f64> {
        let hi = edges[edges.len() - 1];
        let mut values: Vec<f64> = (0..20_000).map(|i| hi * i as f64 / 20_000.0).collect();
        for &e in edges {
            let (below, above) = (e.to_bits().saturating_sub(1), e.to_bits() + 1);
            values.extend([e, f64::from_bits(below), f64::from_bits(above)]);
        }
        values.retain(|v| (0.0..hi).contains(v));
        values
    }

    #[test]
    fn find_all_agrees_with_find() {
        let log_edges: Vec<f64> = (0..=30).map(|i| 0.01 * 1e4_f64.powf(i as f64 / 30.0)).collect();
        let edge_sets: [&[f64]; 3] = [&log_edges, &[0.0, 1.0, 2.0, 3.0], &[2.0, 2.0000001, 5.0]];
        for edges in edge_sets {
            let table = BinTable::new(edges);
            let bins = table.bins();
            let values: Vec<f64> =
                probe_values(edges).into_iter().filter(|&v| bins.contains(v)).collect();
            for chunk in values.chunks(37) {
                let mut found = vec![-7; chunk.len()];
                bins.find_all(chunk, &mut found);
                for (&v, &b) in chunk.iter().zip(&found) {
                    assert_eq!(Some(b as usize), bins.find(v), "value = {v}");
                }
            }
        }
    }

    #[test]
    fn table_lookup_agrees_with_binary_search_everywhere() {
        let log_edges: Vec<f64> =
            (0..=120).map(|i| 1e-3 * 1e4_f64.powf(i as f64 / 120.0)).collect();
        let unit_edges: Vec<f64> = (0..=40).map(f64::from).collect();
        let squared: Vec<f64> = [0.5, 1.0, 2.5, 3.0, 8.0].iter().map(|e| e * e).collect();
        let edge_sets: [&[f64]; 6] = [
            &[0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            &[1.0, 1.0001, 1.0002, 1.0003, 2.0],
            &[7.5, 8.0],
            &log_edges,
            &unit_edges,
            &squared,
        ];
        for edges in edge_sets {
            let table = BinTable::new(edges);
            let bins = table.bins();
            assert_eq!(bins.n_bins(), edges.len() - 1);
            for value in probe_values(edges) {
                assert_eq!(bins.find(value), find_bin(value, edges), "value = {value}");
                assert_eq!(bins.contains(value), find_bin(value, edges).is_some());
            }
            assert!(!bins.contains(bins.hi()));
        }
    }
}
