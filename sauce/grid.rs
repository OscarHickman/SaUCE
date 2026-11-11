//! Periodic cell grid shared by every counter.
//!
//! The box is divided into a regular grid whose cells are at least as wide as
//! the search radius along each axis, so every partner of a particle lies in
//! its own cell or one of the 26 adjacent cells (with periodic wrapping).
//!
//! Two layout choices keep the inner loops fast:
//!
//! * particles are stored cell by cell in structure-of-arrays form (`x`, `y`,
//!   `z`, `partition`), so a cell is a contiguous slice of each array;
//! * within a cell, particles are sorted by `z`, so a search window in `z` can
//!   be found by binary search and the scan stopped early.

use std::ops::Range;

use rayon::prelude::*;

/// Cells per axis are capped to bound memory when the search radius is tiny.
const MAX_CELLS_PER_AXIS: usize = 128;

/// With fewer than three cells per axis the "left" and "right" neighbours of a
/// cell would coincide and pairs would be visited twice.
const MIN_CELLS_PER_AXIS: usize = 3;

/// Unused positions after the last particle in the per-particle arrays of a
/// [`CellGrid`], so that loops can read a fixed number of values starting
/// from any particle.
pub const PADDING: usize = 64;

/// Half of the 26 neighbour offsets: exactly one of each pair `{d, -d}`.
///
/// Visiting a cell's own particles plus these 13 neighbours from every cell
/// visits each unordered pair of neighbouring cells exactly once.
pub const HALF_SHELL: [[i32; 3]; 13] = [
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [-1, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [-1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [-1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
];

/// Maps a separation component onto its nearest periodic image.
#[inline(always)]
pub fn min_image(d: f64, box_size: f64, half_box: f64) -> f64 {
    if d > half_box {
        d - box_size
    } else if d < -half_box {
        d + box_size
    } else {
        d
    }
}

/// Moves a cell index by `step` (with `|step| < n`) with periodic wrapping.
///
/// Returns the wrapped index and the shift to add to a coordinate in the
/// destination cell so that it is measured in the same periodic image as the
/// source cell.
#[inline(always)]
fn step_cell(index: usize, step: i32, n: usize, box_size: f64) -> (usize, f64) {
    let moved = index as i32 + step;
    if moved < 0 {
        ((moved + n as i32) as usize, -box_size)
    } else if moved >= n as i32 {
        ((moved - n as i32) as usize, box_size)
    } else {
        (moved as usize, 0.0)
    }
}

/// `values` followed by [`PADDING`] zeros.
fn padded<T: Clone + Default>(mut values: Vec<T>) -> Vec<T> {
    values.resize(values.len() + PADDING, T::default());
    values
}

pub struct CellGrid {
    /// Coordinates and partitions by grid position, each followed by
    /// [`PADDING`] zeros.
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub partition: Vec<i32>,
    /// Position of each particle in the caller's input arrays.
    pub original_index: Vec<usize>,
    /// Particles of cell `c` occupy `cell_start[c]..cell_start[c + 1]`.
    cell_start: Vec<usize>,
    pub n_xy: usize,
    pub n_z: usize,
    /// Cell width along x and y.
    width_xy: f64,
    pub box_size: f64,
    pub half_box: f64,
}

impl CellGrid {
    /// Bins `coords` (which must lie in `[0, box_size)`) into cells at least
    /// `min_width_xy` wide in x and y and `min_width_z` wide in z.
    pub fn new(
        coords: &[[f64; 3]],
        partition: &[i32],
        box_size: f64,
        min_width_xy: f64,
        min_width_z: f64,
    ) -> Self {
        let cells_per_axis = |width: f64| {
            ((box_size / width) as usize).clamp(MIN_CELLS_PER_AXIS, MAX_CELLS_PER_AXIS)
        };
        let (n_xy, n_z) = (cells_per_axis(min_width_xy), cells_per_axis(min_width_z));
        Self::with_cells(coords, partition, box_size, n_xy, n_z)
    }

    /// Bins `coords` (which must lie in `[0, box_size)`) into `n_xy` by
    /// `n_xy` columns, each spanning the whole box in z.
    pub fn columns(coords: &[[f64; 3]], partition: &[i32], box_size: f64, n_xy: usize) -> Self {
        Self::with_cells(coords, partition, box_size, n_xy, 1)
    }

    fn with_cells(
        coords: &[[f64; 3]],
        partition: &[i32],
        box_size: f64,
        n_xy: usize,
        n_z: usize,
    ) -> Self {
        let width_xy = box_size / n_xy as f64;
        let width_z = box_size / n_z as f64;
        let n_cells = n_xy * n_xy * n_z;

        let cell_of: Vec<usize> = coords
            .par_iter()
            .map(|&[x, y, z]| {
                let ix = ((x / width_xy) as usize).min(n_xy - 1);
                let iy = ((y / width_xy) as usize).min(n_xy - 1);
                let iz = ((z / width_z) as usize).min(n_z - 1);
                (ix * n_xy + iy) * n_z + iz
            })
            .collect();

        // Counting sort of particle indices by cell.
        let mut cell_start = vec![0usize; n_cells + 1];
        for &c in &cell_of {
            cell_start[c + 1] += 1;
        }
        for c in 0..n_cells {
            cell_start[c + 1] += cell_start[c];
        }
        let mut next_slot = cell_start[..n_cells].to_vec();
        let mut order = vec![0usize; coords.len()];
        for (i, &c) in cell_of.iter().enumerate() {
            order[next_slot[c]] = i;
            next_slot[c] += 1;
        }

        // Sort each cell by z, working on batches of whole cells in parallel.
        let cells_per_batch = n_cells.div_ceil(8 * rayon::current_num_threads());
        let mut batches = Vec::new();
        let mut rest = order.as_mut_slice();
        for first in (0..n_cells).step_by(cells_per_batch) {
            let cells = first..(first + cells_per_batch).min(n_cells);
            let (batch, tail) = rest.split_at_mut(cell_start[cells.end] - cell_start[first]);
            batches.push((cells, batch));
            rest = tail;
        }
        batches.into_par_iter().for_each(|(cells, batch)| {
            let base = cell_start[cells.start];
            for c in cells {
                batch[cell_start[c] - base..cell_start[c + 1] - base]
                    .sort_unstable_by(|&a, &b| coords[a][2].total_cmp(&coords[b][2]));
            }
        });

        let gather = |axis: usize| padded(order.par_iter().map(|&i| coords[i][axis]).collect());
        CellGrid {
            x: gather(0),
            y: gather(1),
            z: gather(2),
            partition: padded(order.par_iter().map(|&i| partition[i]).collect()),
            original_index: order,
            cell_start,
            n_xy,
            n_z,
            width_xy,
            box_size,
            half_box: 0.5 * box_size,
        }
    }

    pub fn n_cells(&self) -> usize {
        self.cell_start.len() - 1
    }

    /// Range of grid positions holding the particles of cell `c`.
    #[inline(always)]
    pub fn cell(&self, c: usize) -> Range<usize> {
        self.cell_start[c]..self.cell_start[c + 1]
    }

    /// Cell coordinates `[ix, iy, iz]` of flat cell index `c`.
    #[inline(always)]
    pub fn cell_coords(&self, c: usize) -> [usize; 3] {
        [c / (self.n_z * self.n_xy), (c / self.n_z) % self.n_xy, c % self.n_z]
    }

    /// The cell at `offset` from `[ix, iy, iz]` after periodic wrapping,
    /// together with the shift that brings its coordinates into the image
    /// adjacent to the starting cell.
    #[inline(always)]
    pub fn neighbour(&self, [ix, iy, iz]: [usize; 3], offset: [i32; 3]) -> (usize, [f64; 3]) {
        let (nx, sx) = step_cell(ix, offset[0], self.n_xy, self.box_size);
        let (ny, sy) = step_cell(iy, offset[1], self.n_xy, self.box_size);
        let (nz, sz) = step_cell(iz, offset[2], self.n_z, self.box_size);
        ((nx * self.n_xy + ny) * self.n_z + nz, [sx, sy, sz])
    }

    /// Lower bound on |dx| (or |dy|) between a particle at `coord` in cell
    /// `index` along x (or y) and any particle in the cell `step` cells away,
    /// measured in the particle's periodic image.
    ///
    /// The bound is reduced by more than the rounding error in assigning
    /// particles to cells and in computing separations, so it never exceeds
    /// a separation the caller computes.
    #[inline(always)]
    pub fn gap_xy(&self, index: usize, step: i32, coord: f64) -> f64 {
        let gap = match step.signum() {
            1 => (index as f64 + step as f64) * self.width_xy - coord,
            -1 => coord - (index as f64 + step as f64 + 1.0) * self.width_xy,
            _ => return 0.0,
        };
        (gap - 8.0 * f64::EPSILON * self.box_size).max(0.0)
    }

    /// Whether the per-cell shift from [`CellGrid::neighbour`] can fail to give
    /// the nearest periodic image, so each separation must be wrapped
    /// individually with [`min_image`].
    ///
    /// Particles in adjacent cells are less than two cell widths apart,
    /// which is within half the box as long as every axis has at least four
    /// cells.
    pub fn needs_min_image(&self) -> bool {
        self.n_xy < 4 || self.n_z < 4
    }

    /// Positions in `cell` whose z separation from `z0`, after adding
    /// `shift`, may be at most `reach` (positions are sorted by z).
    ///
    /// The limits are widened by more than the rounding error of the
    /// separation, so the window never excludes a particle that the caller's
    /// exact distance test would accept; it may include a few that the test
    /// then rejects.
    #[inline(always)]
    pub fn z_window(&self, cell: Range<usize>, z0: f64, shift: f64, reach: f64) -> Range<usize> {
        let (lo, hi) = self.z_limits(z0, shift, reach);
        let zs = &self.z[cell.clone()];
        let start = zs.partition_point(|&z| z < lo);
        let end = start + zs[start..].partition_point(|&z| z <= hi);
        cell.start + start..cell.start + end
    }

    /// Limits `(lo, hi)` such that every particle whose z separation from
    /// `z0`, after adding `shift`, is at most `reach` has `lo <= z <= hi`.
    /// See [`CellGrid::z_window`] for how rounding is allowed for.
    #[inline(always)]
    pub fn z_limits(&self, z0: f64, shift: f64, reach: f64) -> (f64, f64) {
        let margin = 4.0 * f64::EPSILON * (2.0 * self.box_size + reach);
        let centre = z0 - shift;
        (centre - reach - margin, centre + reach + margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_points(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * box_size
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn half_shell_and_its_mirror_cover_the_26_neighbours_once() {
        let mut seen = std::collections::HashSet::new();
        for d in HALF_SHELL {
            assert!(seen.insert(d));
            assert!(seen.insert([-d[0], -d[1], -d[2]]));
        }
        assert!(!seen.contains(&[0, 0, 0]));
        assert_eq!(seen.len(), 26);
    }

    #[test]
    fn periodic_wrapping_is_needed_only_on_coarse_grids() {
        let parts = vec![0; 0];
        assert!(CellGrid::new(&[], &parts, 100.0, 30.0, 10.0).needs_min_image());
        assert!(CellGrid::new(&[], &parts, 100.0, 10.0, 30.0).needs_min_image());
        assert!(!CellGrid::new(&[], &parts, 100.0, 25.0, 25.0).needs_min_image());
    }

    #[test]
    fn min_image_folds_into_half_open_interval() {
        let (l, h) = (10.0, 5.0);
        assert_eq!(min_image(6.0, l, h), -4.0);
        assert_eq!(min_image(-6.0, l, h), 4.0);
        assert_eq!(min_image(3.0, l, h), 3.0);
        assert_eq!(min_image(5.0, l, h), 5.0);
        assert_eq!(min_image(-5.0, l, h), -5.0);
    }

    #[test]
    fn every_particle_is_stored_once_in_the_cell_containing_it() {
        let box_size = 50.0;
        let pts = lcg_points(2000, box_size, 1);
        let parts: Vec<i32> = (0..2000).map(|i| i % 7).collect();
        let grid = CellGrid::new(&pts, &parts, box_size, 4.0, 6.0);

        let mut seen = vec![false; pts.len()];
        let width_xy = box_size / grid.n_xy as f64;
        let width_z = box_size / grid.n_z as f64;
        assert!(width_xy >= 4.0 && width_z >= 6.0);

        for c in 0..grid.n_cells() {
            let [ix, iy, iz] = grid.cell_coords(c);
            for p in grid.cell(c) {
                let i = grid.original_index[p];
                assert!(!seen[i]);
                seen[i] = true;
                assert_eq!([grid.x[p], grid.y[p], grid.z[p]], pts[i]);
                assert_eq!(grid.partition[p], parts[i]);
                assert_eq!((pts[i][0] / width_xy) as usize, ix);
                assert_eq!((pts[i][1] / width_xy) as usize, iy);
                assert_eq!((pts[i][2] / width_z) as usize, iz);
            }
            let zs = &grid.z[grid.cell(c)];
            assert!(zs.windows(2).all(|w| w[0] <= w[1]));
        }
        assert!(seen.iter().all(|&s| s));
        for axis in [&grid.x, &grid.y, &grid.z] {
            assert_eq!(axis.len(), pts.len() + PADDING);
        }
        assert_eq!(grid.partition.len(), pts.len() + PADDING);
    }

    #[test]
    fn layout_does_not_depend_on_the_number_of_threads() {
        // Many equal z values, so the order of ties within a cell is tested.
        let pts: Vec<[f64; 3]> =
            lcg_points(20_000, 40.0, 6).into_iter().map(|[x, y, z]| [x, y, z.floor()]).collect();
        let parts: Vec<i32> = (0..20_000).map(|i| i % 5).collect();
        let build = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| CellGrid::new(&pts, &parts, 40.0, 3.0, 3.0))
        };
        let (one, many) = (build(1), build(7));
        assert_eq!(one.original_index, many.original_index);
        assert_eq!(one.x, many.x);
        assert_eq!(one.y, many.y);
        assert_eq!(one.z, many.z);
        assert_eq!(one.partition, many.partition);
        assert_eq!(one.cell_start, many.cell_start);
    }

    #[test]
    fn cells_per_axis_is_clamped() {
        let pts = lcg_points(10, 100.0, 2);
        let parts = vec![0; 10];
        let coarse = CellGrid::new(&pts, &parts, 100.0, 60.0, 60.0);
        assert_eq!((coarse.n_xy, coarse.n_z), (3, 3));
        let fine = CellGrid::new(&pts, &parts, 100.0, 0.01, 0.01);
        assert_eq!((fine.n_xy, fine.n_z), (128, 128));
    }

    #[test]
    fn neighbour_wraps_and_reports_the_image_shift() {
        let grid = CellGrid::new(&[], &[], 10.0, 2.0, 2.0);
        assert_eq!((grid.n_xy, grid.n_z), (5, 5));
        let (c, shift) = grid.neighbour([0, 4, 2], [-1, 1, 0]);
        assert_eq!(grid.cell_coords(c), [4, 0, 2]);
        assert_eq!(shift, [-10.0, 10.0, 0.0]);
        let (c, shift) = grid.neighbour([2, 2, 2], [1, -1, 1]);
        assert_eq!(grid.cell_coords(c), [3, 1, 3]);
        assert_eq!(shift, [0.0; 3]);
    }

    #[test]
    fn gap_xy_bounds_every_separation_from_a_neighbouring_cell() {
        let box_size = 30.0;
        let pts = lcg_points(3000, box_size, 4);
        let parts = vec![0; 3000];
        let grid = CellGrid::new(&pts, &parts, box_size, 5.0, 5.0);
        for c in 0..grid.n_cells() {
            let at = grid.cell_coords(c);
            for offset in HALF_SHELL {
                let (other, shift) = grid.neighbour(at, offset);
                for p in grid.cell(c) {
                    let gap_x = grid.gap_xy(at[0], offset[0], grid.x[p]);
                    let gap_y = grid.gap_xy(at[1], offset[1], grid.y[p]);
                    assert!(gap_x >= 0.0 && gap_y >= 0.0);
                    for q in grid.cell(other) {
                        assert!((grid.x[q] - grid.x[p] + shift[0]).abs() >= gap_x);
                        assert!((grid.y[q] - grid.y[p] + shift[1]).abs() >= gap_y);
                    }
                }
            }
        }
    }

    #[test]
    fn columns_span_the_box_in_z_and_are_sorted() {
        let box_size = 30.0;
        let pts = lcg_points(2000, box_size, 8);
        let parts: Vec<i32> = (0..2000).map(|i| i % 3).collect();
        let grid = CellGrid::columns(&pts, &parts, box_size, 7);
        assert_eq!((grid.n_xy, grid.n_z, grid.n_cells()), (7, 1, 49));
        let width = box_size / 7.0;
        let mut total = 0;
        for c in 0..grid.n_cells() {
            let [ix, iy, iz] = grid.cell_coords(c);
            assert_eq!(iz, 0);
            let cell = grid.cell(c);
            total += cell.len();
            for p in cell.clone() {
                assert_eq!(((grid.x[p] / width) as usize, (grid.y[p] / width) as usize), (ix, iy));
            }
            assert!(grid.z[cell].windows(2).all(|w| w[0] <= w[1]));
        }
        assert_eq!(total, 2000);
    }

    #[test]
    fn gap_xy_bounds_separations_from_columns_two_steps_away() {
        let box_size = 20.0;
        let pts = lcg_points(1500, box_size, 9);
        let parts = vec![0; 1500];
        let grid = CellGrid::columns(&pts, &parts, box_size, 9);
        for c in 0..grid.n_cells() {
            let at = grid.cell_coords(c);
            for (sx, sy) in [(2, 0), (-2, 1), (1, -2), (2, 2), (-2, -2), (0, -1)] {
                let (other, shift) = grid.neighbour(at, [sx, sy, 0]);
                assert_eq!(shift[2], 0.0);
                for p in grid.cell(c) {
                    let gap_x = grid.gap_xy(at[0], sx, grid.x[p]);
                    let gap_y = grid.gap_xy(at[1], sy, grid.y[p]);
                    for q in grid.cell(other) {
                        assert!((grid.x[q] - grid.x[p] + shift[0]).abs() >= gap_x);
                        assert!((grid.y[q] - grid.y[p] + shift[1]).abs() >= gap_y);
                    }
                }
            }
        }
    }

    #[test]
    fn z_limits_contain_the_reach_with_a_margin() {
        let grid = CellGrid::columns(&[], &[], 10.0, 4);
        let (lo, hi) = grid.z_limits(3.0, 0.0, 1.5);
        assert!(lo < 1.5 && lo > 1.5 - 1e-12);
        assert!(hi > 4.5 && hi < 4.5 + 1e-12);
        let (lo, hi) = grid.z_limits(9.5, 10.0, 1.0);
        assert!(lo < -1.5 && hi > 0.5 && hi < 0.5 + 1e-12);
    }

    #[test]
    fn z_window_contains_every_particle_within_reach() {
        let pts = lcg_points(500, 20.0, 3);
        let parts = vec![0; 500];
        let grid = CellGrid::new(&pts, &parts, 20.0, 20.0, 20.0);
        for c in 0..grid.n_cells() {
            for &(z0, shift) in &[(3.0, 0.0), (0.5, -20.0), (19.0, 20.0), (7.25, 0.0)] {
                let window = grid.z_window(grid.cell(c), z0, shift, 2.5);
                for p in grid.cell(c) {
                    let dz: f64 = (grid.z[p] - z0) + shift;
                    if dz.abs() <= 2.5 {
                        assert!(window.contains(&p));
                    }
                    if !window.contains(&p) {
                        assert!(dz.abs() > 2.5 - 1e-9);
                    }
                }
            }
        }
    }
}
