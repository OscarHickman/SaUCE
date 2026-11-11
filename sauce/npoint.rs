//! Tuple (N-point) counters.
//!
//! An N-tuple is counted when every pairwise separation among its members is
//! below `r_max`, and binned by its largest pairwise separation (its
//! "diameter"). Tuples are also classified by `s`, the number of distinct
//! partitions among their members, which is what the sparse-partition
//! correction needs.
//!
//! Each tuple is enumerated exactly once, from its member with the lowest
//! input index (the "anchor"). Every other member is within `r_max` of the
//! anchor, so it is found among the anchor's neighbours in the 27 surrounding
//! cells. Anchoring on input order rather than grid position makes the result
//! independent of the grid layout.

use rayon::prelude::*;

use crate::binning::{mu_bin, squares, BinTable, TableBins};
use crate::grid::{min_image, CellGrid};
use crate::pairs::add_histograms;

/// Largest supported tuple order.
pub const MAX_ORDER: usize = 8;

/// Selects the cells `c` with `c % count == index`, so that `count`
/// independent jobs can each count a disjoint share of the tuples and sum
/// their results. Interleaving rather than contiguous blocks spreads densely
/// populated regions evenly across jobs.
#[derive(Clone, Copy)]
pub struct Shard {
    pub index: usize,
    pub count: usize,
}

impl Shard {
    fn contains(self, cell: usize) -> bool {
        cell % self.count == self.index
    }
}

/// A later-indexed particle within `r_max` of the anchor.
#[derive(Clone, Copy)]
struct Neighbour {
    /// Grid position.
    pos: usize,
    /// Nearest-image separation from the anchor.
    d: [f64; 3],
    r_sq: f64,
}

/// Collects into `out` every particle within `r_max` of the anchor at grid
/// position `anchor` (in cell `at`) whose input index is larger than the
/// anchor's.
#[inline(always)]
fn gather_neighbours(
    grid: &CellGrid,
    anchor: usize,
    at: [usize; 3],
    r_max: f64,
    wrap_each: bool,
    out: &mut Vec<Neighbour>,
) {
    out.clear();
    let (x, y, z) = (grid.x[anchor], grid.y[anchor], grid.z[anchor]);
    let first = grid.original_index[anchor];
    let r_sq_max = r_max * r_max;
    let (box_size, half_box) = (grid.box_size, grid.half_box);

    let mut keep_if_close = |q: usize, [dx, dy, dz]: [f64; 3]| {
        let r_sq = dx * dx + dy * dy + dz * dz;
        if r_sq < r_sq_max {
            out.push(Neighbour { pos: q, d: [dx, dy, dz], r_sq });
        }
    };

    for dx_cell in -1..=1 {
        for dy_cell in -1..=1 {
            for dz_cell in -1..=1 {
                let (c, [sx, sy, sz]) = grid.neighbour(at, [dx_cell, dy_cell, dz_cell]);
                let cell = grid.cell(c);
                if wrap_each {
                    for q in cell.filter(|&q| grid.original_index[q] > first) {
                        let d = [grid.x[q] - x + sx, grid.y[q] - y + sy, grid.z[q] - z + sz];
                        keep_if_close(q, d.map(|d| min_image(d, box_size, half_box)));
                    }
                } else {
                    for q in grid.z_window(cell, z, sz, r_max) {
                        if grid.original_index[q] > first {
                            let d = [grid.x[q] - x + sx, grid.y[q] - y + sy, grid.z[q] - z + sz];
                            keep_if_close(q, d);
                        }
                    }
                }
            }
        }
    }
}

/// Runs `visit(hist, anchor, neighbours)` for every particle in parallel,
/// accumulating into per-worker histograms of length `n_bins` that are summed
/// at the end.
fn histogram_anchors(
    grid: &CellGrid,
    r_max: f64,
    shard: Option<Shard>,
    n_bins: usize,
    visit: impl Fn(&mut [f64], usize, &[Neighbour]) + Sync + Send,
) -> Vec<f64> {
    let wrap_each = grid.needs_min_image();
    // The work is split by anchor, not by cell: an anchor's work grows as its
    // number of neighbours to the power N - 1, so for large N one dense cell
    // can hold most of the total and would otherwise run on a single thread.
    // The counts are integers, so the order of summation does not matter.
    (0..grid.n_cells())
        .into_par_iter()
        .filter(|&c| shard.is_none_or(|s| s.contains(c)))
        .flat_map(|c| {
            let at = grid.cell_coords(c);
            grid.cell(c).into_par_iter().map(move |anchor| (anchor, at))
        })
        .fold(
            || (vec![0.0; n_bins], Vec::with_capacity(512)),
            |(mut hist, mut neighbours), (anchor, at)| {
                gather_neighbours(grid, anchor, at, r_max, wrap_each, &mut neighbours);
                visit(&mut hist, anchor, &neighbours);
                (hist, neighbours)
            },
        )
        .map(|(hist, _)| hist)
        .reduce(|| vec![0.0; n_bins], add_histograms)
}

/// Geometry and partition class of one counted triplet.
pub struct Triangle {
    /// Squared length of the longest side.
    pub longest_sq: f64,
    /// Squared length of the shortest side.
    pub shortest_sq: f64,
    /// z component of the longest side. When two sides tie for longest, the
    /// side listed first among (anchor-j, anchor-k, j-k) is used.
    pub dz_longest: f64,
    /// Number of distinct partitions among the three members (1 to 3).
    pub n_partitions: usize,
}

/// Calls `on_triangle` for every pair `{j, k}` of the anchor's neighbours
/// whose separation is also below `r_sq_max`.
#[inline(always)]
fn for_each_triangle(
    grid: &CellGrid,
    anchor: usize,
    neighbours: &[Neighbour],
    r_sq_max: f64,
    wrap_each: bool,
    mut on_triangle: impl FnMut(&Triangle),
) {
    let (box_size, half_box) = (grid.box_size, grid.half_box);
    let part_i = grid.partition[anchor];
    for (idx, j) in neighbours.iter().enumerate() {
        let part_j = grid.partition[j.pos];
        for k in &neighbours[idx + 1..] {
            // The j-k side, testing each component against r_max as soon as
            // it is known.
            let mut dz = k.d[2] - j.d[2];
            if wrap_each {
                dz = min_image(dz, box_size, half_box);
            }
            let dz_sq = dz * dz;
            if dz_sq >= r_sq_max {
                continue;
            }
            let mut dy = k.d[1] - j.d[1];
            if wrap_each {
                dy = min_image(dy, box_size, half_box);
            }
            let dyz_sq = dy * dy + dz_sq;
            if dyz_sq >= r_sq_max {
                continue;
            }
            let mut dx = k.d[0] - j.d[0];
            if wrap_each {
                dx = min_image(dx, box_size, half_box);
            }
            let jk_sq = dx * dx + dyz_sq;
            if jk_sq >= r_sq_max {
                continue;
            }

            let longest_sq = j.r_sq.max(k.r_sq).max(jk_sq);
            let dz_longest = if longest_sq == j.r_sq {
                j.d[2]
            } else if longest_sq == k.r_sq {
                k.d[2]
            } else {
                dz
            };
            let part_k = grid.partition[k.pos];
            let n_partitions = if part_i == part_j && part_j == part_k {
                1
            } else if part_i != part_j && part_j != part_k && part_i != part_k {
                3
            } else {
                2
            };
            on_triangle(&Triangle {
                longest_sq,
                shortest_sq: j.r_sq.min(k.r_sq).min(jk_sq),
                dz_longest,
                n_partitions,
            });
        }
    }
}

/// Triplet counts binned by the longest side (`n_r` bins from `r_edges`) and
/// a secondary quantity with `n_extra` bins chosen by `extra_bin`, which may
/// return `None` to discard a triplet.
///
/// Returns a flat array of shape `(3, n_r, n_extra)`, indexed by
/// `(n_partitions - 1, radial bin, extra bin)`.
pub fn triplet_counts(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    n_extra: usize,
    shard: Option<Shard>,
    extra_bin: impl Fn(&Triangle) -> Option<usize> + Sync + Send,
) -> Vec<f64> {
    let table = BinTable::new(&squares(r_edges));
    let bins = table.bins();
    let r_max = r_edges[r_edges.len() - 1];
    let block = bins.n_bins() * n_extra;
    let grid = CellGrid::new(coords, partition, box_size, r_max, r_max);
    let wrap_each = grid.needs_min_image();

    let (grid, extra_bin) = (&grid, &extra_bin);
    histogram_anchors(grid, r_max, shard, 3 * block, move |hist, anchor, neighbours| {
        for_each_triangle(grid, anchor, neighbours, bins.hi(), wrap_each, |t| {
            if let (Some(r_bin), Some(extra)) = (bins.find(t.longest_sq), extra_bin(t)) {
                hist[(t.n_partitions - 1) * block + r_bin * n_extra + extra] += 1.0;
            }
        });
    })
}

/// Triplet counts binned by the longest side only; shape `(3, n_r)`.
pub fn triplet_counts_by_size(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    shard: Option<Shard>,
) -> Vec<f64> {
    triplet_counts(coords, partition, box_size, r_edges, 1, shard, |_| Some(0))
}

/// Triplet counts binned by the longest side and by the ratio of the
/// shortest to the longest side, in `n_ratio` equal bins over `[0, 1]`;
/// shape `(3, n_r, n_ratio)`.
pub fn triplet_counts_by_shape(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    n_ratio: usize,
) -> Vec<f64> {
    triplet_counts(coords, partition, box_size, r_edges, n_ratio, None, |t| {
        let ratio = (t.shortest_sq / t.longest_sq).sqrt();
        Some(((ratio * n_ratio as f64) as usize).min(n_ratio - 1))
    })
}

/// Triplet counts binned by the longest side and by `mu`, the cosine of the
/// angle between the longest side and the line of sight (the z axis), in
/// `n_mu` equal bins over `[0, mu_max)`; shape `(3, n_r, n_mu)`.
pub fn triplet_counts_by_mu(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    n_mu: usize,
    mu_max: f64,
) -> Vec<f64> {
    triplet_counts(coords, partition, box_size, r_edges, n_mu, None, |t| {
        mu_bin(t.dz_longest.abs() / t.longest_sq.sqrt(), mu_max, n_mu)
    })
}

/// Depth-first enumeration of the tuples of order `n_order` that contain one
/// anchor, built by adding the anchor's neighbours in increasing list order.
struct TupleSearch<'a> {
    grid: &'a CellGrid,
    bins: TableBins<'a>,
    neighbours: &'a [Neighbour],
    n_order: usize,
    wrap_each: bool,
    /// Grid positions of the members chosen so far; `members[0]` is the anchor.
    members: [usize; MAX_ORDER],
    /// Separations of the chosen members from the anchor.
    offsets: [[f64; 3]; MAX_ORDER],
}

impl TupleSearch<'_> {
    /// Extends the first `depth` members with neighbours from `start` on;
    /// `diameter_sq` is the largest squared separation among them so far.
    fn extend(&mut self, depth: usize, start: usize, diameter_sq: f64, hist: &mut [f64]) {
        if depth == self.n_order {
            self.record(diameter_sq, hist);
            return;
        }
        let r_sq_max = self.bins.hi();
        let (box_size, half_box) = (self.grid.box_size, self.grid.half_box);
        let neighbours = self.neighbours;

        'candidates: for (idx, candidate) in neighbours.iter().enumerate().skip(start) {
            let mut diameter_sq = diameter_sq.max(candidate.r_sq);
            for offset in &self.offsets[1..depth] {
                let d = [0, 1, 2].map(|axis| {
                    let d = candidate.d[axis] - offset[axis];
                    if self.wrap_each {
                        min_image(d, box_size, half_box)
                    } else {
                        d
                    }
                });
                let r_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r_sq >= r_sq_max {
                    continue 'candidates;
                }
                diameter_sq = diameter_sq.max(r_sq);
            }
            self.members[depth] = candidate.pos;
            self.offsets[depth] = candidate.d;
            self.extend(depth + 1, idx + 1, diameter_sq, hist);
        }
    }

    fn record(&self, diameter_sq: f64, hist: &mut [f64]) {
        let Some(bin) = self.bins.find(diameter_sq) else {
            return;
        };
        let mut distinct = [0i32; MAX_ORDER];
        let mut n_distinct = 0;
        for &member in &self.members[..self.n_order] {
            let part = self.grid.partition[member];
            if !distinct[..n_distinct].contains(&part) {
                distinct[n_distinct] = part;
                n_distinct += 1;
            }
        }
        hist[(n_distinct - 1) * self.bins.n_bins() + bin] += 1.0;
    }
}

/// Counts of tuples of order `n_order` (3 to [`MAX_ORDER`]) binned by
/// diameter; shape `(n_order, n_r)`, indexed by `(n_partitions - 1, bin)`.
pub fn tuple_counts(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    n_order: usize,
    shard: Option<Shard>,
) -> Vec<f64> {
    assert!((3..=MAX_ORDER).contains(&n_order));
    if n_order == 3 {
        return triplet_counts_by_size(coords, partition, box_size, r_edges, shard);
    }
    let table = BinTable::new(&squares(r_edges));
    let bins = table.bins();
    let r_max = r_edges[r_edges.len() - 1];
    let grid = CellGrid::new(coords, partition, box_size, r_max, r_max);
    let wrap_each = grid.needs_min_image();

    histogram_anchors(&grid, r_max, shard, n_order * bins.n_bins(), |hist, anchor, neighbours| {
        let mut search = TupleSearch {
            grid: &grid,
            bins,
            neighbours,
            n_order,
            wrap_each,
            members: [anchor; MAX_ORDER],
            offsets: [[0.0; 3]; MAX_ORDER],
        };
        search.extend(1, 0, 0.0, hist);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::find_bin;
    use crate::pairs::tests::test_catalogue;

    /// Squared nearest-image distance between two points.
    fn dist_sq(a: [f64; 3], b: [f64; 3], box_size: f64) -> f64 {
        let d = [0, 1, 2].map(|i| min_image(b[i] - a[i], box_size, 0.5 * box_size));
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }

    /// Visits every tuple of `n` indices in increasing order.
    fn for_each_combination(n_items: usize, n: usize, mut visit: impl FnMut(&[usize])) {
        let mut idx: Vec<usize> = (0..n).collect();
        if n > n_items {
            return;
        }
        loop {
            visit(&idx);
            let Some(pos) = (0..n).rev().find(|&i| idx[i] != i + n_items - n) else {
                return;
            };
            idx[pos] += 1;
            for i in pos + 1..n {
                idx[i] = idx[i - 1] + 1;
            }
        }
    }

    fn brute_force_tuples(
        coords: &[[f64; 3]],
        parts: &[i32],
        box_size: f64,
        edges: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let n_r = edges.len() - 1;
        let mut hist = vec![0.0; n * n_r];
        for_each_combination(coords.len(), n, |tuple| {
            let mut diameter: f64 = 0.0;
            for (a, &i) in tuple.iter().enumerate() {
                for &j in &tuple[a + 1..] {
                    diameter = diameter.max(dist_sq(coords[i], coords[j], box_size).sqrt());
                }
            }
            if let Some(b) = find_bin(diameter, edges) {
                let mut ps: Vec<i32> = tuple.iter().map(|&i| parts[i]).collect();
                ps.sort_unstable();
                ps.dedup();
                hist[(ps.len() - 1) * n_r + b] += 1.0;
            }
        });
        hist
    }

    #[test]
    fn tuple_counts_match_brute_force() {
        // The second case has r_max > box / 4, which uses per-pair wrapping.
        let cases: [(usize, f64, &[f64], usize); 4] = [
            (250, 60.0, &[0.5, 3.0, 6.0, 8.0], 3),
            (70, 16.0, &[0.5, 2.0, 5.0, 7.5], 3),
            (60, 30.0, &[1.0, 4.0, 7.0], 4),
            (30, 12.0, &[1.0, 3.0, 5.5], 5),
        ];
        for (n, box_size, edges, order) in cases {
            let (coords, parts) = test_catalogue(n, box_size, 4, 21);
            let expected = brute_force_tuples(&coords, &parts, box_size, edges, order);
            assert!(expected.iter().sum::<f64>() > 0.0);
            assert_eq!(tuple_counts(&coords, &parts, box_size, edges, order, None), expected);
        }
    }

    #[test]
    fn triplet_extras_marginalise_to_size_counts() {
        let (coords, parts) = test_catalogue(400, 40.0, 3, 22);
        let edges = [0.5, 2.0, 4.0, 6.0];
        let by_size = triplet_counts_by_size(&coords, &parts, 40.0, &edges, None);
        let marginal = |counts: Vec<f64>, n_extra: usize| -> Vec<f64> {
            counts.chunks(n_extra).map(|c| c.iter().sum()).collect()
        };
        assert_eq!(marginal(triplet_counts_by_shape(&coords, &parts, 40.0, &edges, 6), 6), by_size);
        assert_eq!(
            marginal(triplet_counts_by_mu(&coords, &parts, 40.0, &edges, 5, 1.0), 5),
            by_size
        );
    }

    #[test]
    fn shards_sum_to_the_full_count() {
        let (coords, parts) = test_catalogue(300, 40.0, 3, 23);
        let edges = [0.5, 2.0, 5.0];
        for order in [3, 4] {
            let full = tuple_counts(&coords, &parts, 40.0, &edges, order, None);
            let mut summed = vec![0.0; full.len()];
            for index in 0..4 {
                let part = tuple_counts(
                    &coords,
                    &parts,
                    40.0,
                    &edges,
                    order,
                    Some(Shard { index, count: 4 }),
                );
                summed = add_histograms(summed, part);
            }
            assert_eq!(summed, full);
        }
    }
}
