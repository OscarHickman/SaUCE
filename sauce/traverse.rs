//! Enumeration of the pairs of particles within a given reach.
//!
//! The pair counters in [`crate::pairs`] differ only in how a pair is binned
//! (a [`PairBinning`]); they all find pairs with [`Layout::for_each_pair`].
//!
//! Particles are arranged in columns spanning the box along the line of sight
//! (z) and sorted by z within each column. Each particle's partners are found
//! in its own column and in the columns up to one or two steps away in x and
//! y, within the range of z they can reach. Candidates are tested a block at
//! a time in loops that compile to SIMD instructions, and the kept ones are
//! packed together and binned a block at a time too.

use std::ops::Range;

use rayon::prelude::*;

use crate::grid::{min_image, CellGrid, HALF_SHELL, PADDING};

/// Number of candidate partners examined together in the inner loop. It is
/// the width of the bit mask of passing candidates.
pub const BLOCK: usize = 64;

const _: () = assert!(BLOCK <= 64 && BLOCK <= PADDING && BLOCK.is_multiple_of(8));

/// Windows of at most this many candidates are examined as one short block.
/// Most windows are this small when the search radius is small.
const SHORT_BLOCK: usize = 16;

/// Blocks with at least this many kept candidates are counted with
/// [`PairBinning::count_block`], when the binning provides it. Its cost
/// hardly depends on the number kept, so smaller blocks are cheaper to pack
/// and bin pair by pair.
const COUNT_BLOCKS_FROM: u32 = 28;

/// Relative slack in [`Reach`] tests, far larger than rounding errors.
const SLACK: f64 = 1.0 + 1e-12;

/// Columns are normally half the search radius wide, with partners up to two
/// columns away. Narrow columns fit the sphere (or cylinder) of accepted
/// separations closely, so few candidates are tested per counted pair. Each
/// column searched has a fixed cost, though, so when fewer than this many
/// particles are expected in a window, columns as wide as the search radius
/// are used instead, with partners at most one column away.
const MIN_EXPECTED_WINDOW: f64 = 8.0;

/// Columns per axis are capped to bound the overhead per column when the
/// search radius is tiny.
const MAX_COLUMNS_PER_AXIS: usize = 1024;

/// Relative distance from the top or bottom of the box within which a z
/// window is treated as possibly crossing it. Far larger than rounding
/// errors; searching a window that turns out to be empty costs little.
const NEAR_EDGE: f64 = 1e-6;

/// The separations a counter can accept, used to skip candidates that
/// cannot pass before testing them.
#[derive(Clone, Copy, Debug)]
pub enum Reach {
    /// 3D separation below `r`.
    Sphere(f64),
    /// Transverse separation below `r_p` and `|dz|` below `pi`.
    Cylinder { r_p: f64, pi: f64 },
}

impl Reach {
    /// Largest `|dz|` a counted pair can have, with the same slack.
    fn z_max(self) -> f64 {
        match self {
            Reach::Sphere(r) => r * SLACK,
            Reach::Cylinder { pi, .. } => pi * SLACK,
        }
    }

    /// Whether a pair whose transverse separation is at least `sqrt(gap_sq)`
    /// can be counted. The small relative slack covers rounding in the
    /// caller's distance.
    #[inline(always)]
    fn allows(self, gap_sq: f64) -> bool {
        let r = match self {
            Reach::Sphere(r) => r,
            Reach::Cylinder { r_p, .. } => r_p,
        };
        gap_sq < r * r * SLACK
    }

    /// Largest `|dz|` a counted pair can have when its transverse separation
    /// is at least `sqrt(gap_sq)`, assuming [`Self::allows`] holds.
    #[inline(always)]
    fn z_reach(self, gap_sq: f64) -> f64 {
        match self {
            Reach::Sphere(r) => (r * r * SLACK - gap_sq).sqrt() * SLACK,
            Reach::Cylinder { pi, .. } => pi * SLACK,
        }
    }
}

/// Kept pairs of one particle, packed together for binning.
pub struct Kept {
    /// Separations of the partners, one array per axis.
    pub d: [[f64; BLOCK]; 3],
    /// Partitions of the partners.
    pub partition: [i32; BLOCK],
    /// Number of pairs held.
    pub len: usize,
}

/// How one pair counter tests and bins separations.
///
/// The methods are called for whole blocks of candidates, and should be
/// simple loops that the compiler can vectorise.
pub trait PairBinning: Sync {
    /// The separations that can be counted.
    fn reach(&self) -> Reach;

    /// Whether a separation may be counted. Must reject every separation
    /// outside [`Self::reach`], since those may be skipped without a test.
    fn keep(&self, d: [f64; 3]) -> bool;

    /// Number of histogram bins. Bin `n_bins()` itself receives the pairs
    /// that are discarded.
    fn n_bins(&self) -> usize;

    /// Sets `bins[j]` to the bin of kept pair `j`, for pairs between a
    /// particle in `partition` and the partners in `kept`.
    fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]);

    /// Whether [`Self::count_block`] is used instead of [`Self::bin_kept`].
    const COUNTS_BLOCKS: bool = false;

    /// Counts the pairs between a particle in `partition` and the candidates
    /// of a block (separations `d`, partitions `partitions`) whose bits are
    /// set in `kept`, calling `record(bin, n)` to add `n` pairs to a bin.
    /// Binnings with few bins can do this without packing the pairs.
    fn count_block(
        &self,
        _partition: i32,
        _d: &[[f64; BLOCK]; 3],
        _partitions: &[i32; BLOCK],
        _kept: u64,
        _record: &mut impl FnMut(usize, u64),
    ) {
        unreachable!("count_block is used only when COUNTS_BLOCKS is set")
    }
}

/// Buffers for the inner loop, one per worker thread and reused for every
/// block of candidates.
struct Scratch {
    /// Separations of the block's candidates, one array per axis.
    d: [[f64; BLOCK]; 3],
    /// 1 for candidates that pass [`PairBinning::keep`], else 0.
    passed: [u8; BLOCK],
    kept: Kept,
    /// Bins of the kept pairs.
    bins: [usize; BLOCK],
}

impl Scratch {
    fn new() -> Box<Self> {
        Box::new(Scratch {
            d: [[0.0; BLOCK]; 3],
            passed: [0; BLOCK],
            kept: Kept { d: [[0.0; BLOCK]; 3], partition: [0; BLOCK], len: 0 },
            bins: [0; BLOCK],
        })
    }

    /// Bins the kept pairs of a particle in `partition`, calls `record(bin, 1)`
    /// for each and empties the buffer.
    #[inline(always)]
    fn flush<B: PairBinning>(
        &mut self,
        partition: i32,
        binning: &B,
        record: &mut impl FnMut(usize, u64),
    ) {
        binning.bin_kept(partition, &self.kept, &mut self.bins);
        for &bin in &self.bins[..self.kept.len] {
            record(bin, 1);
        }
        self.kept.len = 0;
    }
}

/// Bit `k` of the result is `flags[k]`, for 0/1 flags and `N <= 64`.
#[inline(always)]
pub fn pack_flags<const N: usize>(flags: &[u8; N]) -> u64 {
    // Multiplying eight 0/1 bytes by this constant gathers them into the top
    // byte, in order.
    const GATHER: u64 = 0x0102_0408_1020_4080;
    let mut mask = 0;
    for (i, eight) in flags.as_chunks::<8>().0.iter().enumerate() {
        mask |= (u64::from_le_bytes(*eight).wrapping_mul(GATHER) >> 56) << (8 * i);
    }
    mask
}

/// The first `N` elements of `values` as an array.
#[inline(always)]
fn first<const N: usize, T>(values: &mut [T]) -> &mut [T; N] {
    (&mut values[..N]).try_into().expect("at least N values")
}

/// Sets `d[k] = coords[k] - origin + shift`: separations along one axis
/// from a particle at `origin` to a block of candidates.
#[inline(always)]
fn separations<const N: usize>(d: &mut [f64; N], coords: &[f64], origin: f64, shift: f64) {
    let coords: &[f64; N] = coords[..N].try_into().expect("a whole block");
    for (d, &c) in d.iter_mut().zip(coords) {
        *d = c - origin + shift;
    }
}

/// Examines grid positions `candidates` as partners of `p`, and calls
/// `record(bin, n)` to count the pairs whose separation from `p` (after
/// adding `shift`) is kept by `binning`.
///
/// Candidates are handled a block at a time (see [`bin_block`]).
#[inline(always)]
fn bin_candidates<B: PairBinning>(
    grid: &CellGrid,
    p: usize,
    shift: [f64; 3],
    candidates: Range<usize>,
    binning: &B,
    scratch: &mut Scratch,
    record: &mut impl FnMut(usize, u64),
) {
    let mut start = candidates.start;
    while start < candidates.end {
        let n = (candidates.end - start).min(BLOCK);
        if n <= SHORT_BLOCK {
            bin_block::<SHORT_BLOCK, B>(grid, p, shift, start, n, binning, scratch, record);
        } else {
            bin_block::<BLOCK, B>(grid, p, shift, start, n, binning, scratch, record);
        }
        start += n;
    }
}

/// Counts the pairs of `p` with the `n <= N` candidates from grid position
/// `start`.
///
/// The separations and the `keep` test for the whole block are computed in
/// fixed-length loops that compile to SIMD instructions (the grid's padding
/// makes every block readable), and the passing candidates are found from a
/// bit mask. They are then packed together to be binned, or, if there are
/// many, counted directly by the binning when it can.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn bin_block<const N: usize, B: PairBinning>(
    grid: &CellGrid,
    p: usize,
    shift: [f64; 3],
    start: usize,
    n: usize,
    binning: &B,
    scratch: &mut Scratch,
    record: &mut impl FnMut(usize, u64),
) {
    let (x, y, z) = (grid.x[p], grid.y[p], grid.z[p]);
    let [dx, dy, dz] = &mut scratch.d;
    separations(first::<N, _>(dx), &grid.x[start..], x, shift[0]);
    separations(first::<N, _>(dy), &grid.y[start..], y, shift[1]);
    separations(first::<N, _>(dz), &grid.z[start..], z, shift[2]);
    let passed = first::<N, _>(&mut scratch.passed);
    for (k, pk) in passed.iter_mut().enumerate() {
        *pk = binning.keep([dx[k], dy[k], dz[k]]) as u8;
    }

    let mut mask = pack_flags(passed) & (u64::MAX >> (64 - n));
    if N == BLOCK && B::COUNTS_BLOCKS && mask.count_ones() >= COUNT_BLOCKS_FROM {
        let partitions = grid.partition[start..start + BLOCK].try_into().expect("padded");
        binning.count_block(grid.partition[p], &scratch.d, partitions, mask, record);
        return;
    }
    let kept = &mut scratch.kept;
    let mut j = 0;
    while mask != 0 {
        let k = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        kept.d[0][j] = dx[k];
        kept.d[1][j] = dy[k];
        kept.d[2][j] = dz[k];
        kept.partition[j] = grid.partition[start + k];
        j += 1;
    }
    kept.len = j;
    scratch.flush(grid.partition[p], binning, record);
}

/// Offsets `[dx, dy, 0]` of the neighbouring columns searched from each
/// column: one of each pair `{d, -d}` with `|dx|, |dy| <= steps`, so that
/// every pair of columns is visited once.
fn column_offsets(steps: i32) -> Vec<[i32; 3]> {
    let steps = -steps..=steps;
    steps
        .clone()
        .flat_map(|dx| steps.clone().map(move |dy| [dx, dy, 0]))
        .filter(|&[dx, dy, _]| dx > 0 || (dx == 0 && dy > 0))
        .collect()
}

/// Particles arranged for pair enumeration.
pub enum Layout {
    /// Columns spanning the box in z, in which each particle's partners are
    /// found by searching a z window of every nearby column.
    Columns {
        grid: CellGrid,
        /// How many columns away along x and y partners may be.
        steps: i32,
    },
    /// Cells a third of the box wide or more, used when the search reaches
    /// beyond a quarter of the box. Every separation is wrapped to its
    /// nearest image individually.
    WrappedCells(CellGrid),
}

impl Layout {
    /// Layout for counting pairs within `r_xy` across the line of sight and
    /// `r_z` along it.
    pub fn new(coords: &[[f64; 3]], partition: &[i32], box_size: f64, r_xy: f64, r_z: f64) -> Self {
        Self::with_steps(coords, partition, box_size, r_xy, r_z, None)
    }

    /// [`Self::new`], with the number of column steps (1 or 2) fixed by
    /// `steps` instead of chosen from the density, when columns are used.
    fn with_steps(
        coords: &[[f64; 3]],
        partition: &[i32],
        box_size: f64,
        r_xy: f64,
        r_z: f64,
        steps: Option<i32>,
    ) -> Self {
        // The condition of `CellGrid::needs_min_image`, for cells as wide as
        // the reach: beyond a quarter of the box, z windows could overlap.
        let fits = |r: f64| (box_size / r) as usize >= 4;
        if fits(r_xy) && fits(r_z) {
            // Mean number of particles in the z window of a column half the
            // radius wide.
            let density = coords.len() as f64 / box_size.powi(3);
            let expected_window = density * (0.5 * r_xy).powi(2) * 2.0 * r_z;
            let steps = steps.unwrap_or(if expected_window < MIN_EXPECTED_WINDOW { 1 } else { 2 });
            let columns = (steps as f64 * box_size / r_xy) as usize;
            let n_xy = columns.min(MAX_COLUMNS_PER_AXIS);
            let grid = CellGrid::columns(coords, partition, box_size, n_xy);
            Layout::Columns { grid, steps }
        } else {
            Layout::WrappedCells(CellGrid::new(coords, partition, box_size, r_xy, r_z))
        }
    }

    pub fn grid(&self) -> &CellGrid {
        match self {
            Layout::Columns { grid, .. } | Layout::WrappedCells(grid) => grid,
        }
    }

    /// Counts every pair within reach, calling `record(acc, bin, n)` to add
    /// `n` pairs to `bin`. Cells are processed in parallel; each worker
    /// accumulates into its own `init()`, and the accumulators are returned.
    pub fn for_each_pair<B: PairBinning, T: Send>(
        &self,
        binning: &B,
        init: impl Fn() -> T + Sync + Send,
        record: impl Fn(&mut T, usize, u64) + Sync + Send,
    ) -> Vec<T> {
        self.for_each_pair_using(Simd::detect(), binning, init, record)
    }

    /// [`Self::for_each_pair`] with the loops compiled for `simd`.
    fn for_each_pair_using<B: PairBinning, T: Send>(
        &self,
        simd: Simd,
        binning: &B,
        init: impl Fn() -> T + Sync + Send,
        record: impl Fn(&mut T, usize, u64) + Sync + Send,
    ) -> Vec<T> {
        match simd {
            Simd::Baseline => fold_pairs!(self, binning, init, record),
            // SAFETY: `Simd::detect` found these instructions on this CPU.
            #[cfg(target_arch = "x86_64")]
            Simd::Avx2 => unsafe { for_each_pair_avx2(self, binning, init, record) },
        }
    }
}

/// The body of [`Layout::for_each_pair`]. It is a macro so that each
/// instruction set's copy defines its own closures, which then inherit the
/// instructions of the function they are written in.
macro_rules! fold_pairs {
    ($layout:expr, $binning:expr, $init:expr, $record:expr) => {{
        let (layout, binning, init, record) = ($layout, $binning, $init, $record);
        let offsets = match layout {
            Layout::Columns { steps, .. } => column_offsets(*steps),
            Layout::WrappedCells(_) => Vec::new(),
        };
        (0..layout.grid().n_cells())
            .into_par_iter()
            .fold(
                || (init(), Scratch::new()),
                |(mut acc, mut scratch), c| {
                    let record = &mut |b, n| record(&mut acc, b, n);
                    visit_cell(layout, c, &offsets, binning, &mut scratch, record);
                    (acc, scratch)
                },
            )
            .map(|(acc, _)| acc)
            .collect()
    }};
}
use fold_pairs;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
fn for_each_pair_avx2<B: PairBinning, T: Send>(
    layout: &Layout,
    binning: &B,
    init: impl Fn() -> T + Sync + Send,
    record: impl Fn(&mut T, usize, u64) + Sync + Send,
) -> Vec<T> {
    fold_pairs!(layout, binning, init, record)
}

/// The SIMD instructions the pair loops are compiled for.
///
/// Distributed builds target a baseline CPU, so the loops are compiled a
/// second time with wider instructions, and the widest the CPU supports is
/// chosen at run time. The instructions used do not change any result.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Simd {
    Baseline,
    /// AVX2 with the bit-manipulation instructions of the same CPU
    /// generation (x86-64-v3).
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Simd {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("lzcnt")
            && is_x86_feature_detected!("popcnt")
        {
            return Simd::Avx2;
        }
        Simd::Baseline
    }
}

/// Counts the pairs belonging to cell (or column) `c` of `layout`.
#[inline(always)]
fn visit_cell<B: PairBinning>(
    layout: &Layout,
    c: usize,
    offsets: &[[i32; 3]],
    binning: &B,
    scratch: &mut Scratch,
    record: &mut impl FnMut(usize, u64),
) {
    match layout {
        Layout::Columns { grid, .. } => {
            visit_column_pairs(grid, c, offsets, binning, scratch, record)
        }
        Layout::WrappedCells(grid) => visit_wrapped_cell_pairs(grid, c, binning, scratch, record),
    }
}

/// Counts, through `record(bin, n)`, every unordered pair of grid positions
/// `p`, `q` with `p` in column `c` and `q` in the same column or one of the
/// columns at `offsets`, whose nearest-image separation (of `q` from `p`) is
/// kept by `binning`.
///
/// Candidates outside `binning.reach()` are skipped without being tested:
/// whole columns are skipped when they are too far from `p` across the line
/// of sight, and only candidates within the remaining `|dz|` range, which may
/// wrap around the box, are examined.
///
/// Columns are sorted by z and their particles are taken in order, so the
/// limits of each z window only move up the column being searched. They are
/// found by advancing indices rather than by searching.
#[inline(always)]
fn visit_column_pairs<B: PairBinning>(
    grid: &CellGrid,
    c: usize,
    offsets: &[[i32; 3]],
    binning: &B,
    scratch: &mut Scratch,
    record: &mut impl FnMut(usize, u64),
) {
    let own = grid.cell(c);
    if own.is_empty() {
        return;
    }
    let reach = binning.reach();
    let z_max = reach.z_max();
    let box_size = grid.box_size;
    let near_top = box_size * (1.0 - NEAR_EDGE);

    // Pairs within the column: the partners of p within reach follow it
    // directly, or, across the top of the box, start the column.
    let zs = &grid.z[own.clone()];
    let (mut above, mut wrapped) = (0, 0);
    for (i, &z) in zs.iter().enumerate() {
        let p = own.start + i;
        above = advance(zs, above.max(i + 1), |zq| zq - z <= z_max);
        bin_candidates(grid, p, [0.0; 3], p + 1..own.start + above, binning, scratch, record);
        if z + z_max >= near_top {
            let (_, hi) = grid.z_limits(z, box_size, z_max);
            wrapped = advance(zs, wrapped, |zq| zq <= hi);
            let partners = own.start..own.start + wrapped;
            bin_candidates(grid, p, [0.0, 0.0, box_size], partners, binning, scratch, record);
        }
    }

    let at = grid.cell_coords(c);
    for &offset in offsets {
        let (other_cell, [shift_x, shift_y, _]) = grid.neighbour(at, offset);
        let other = grid.cell(other_cell);
        if other.is_empty() {
            continue;
        }
        let window = |range: Range<usize>| other.start + range.start..other.start + range.end;
        let mut sweep = Sweep::new(&grid.z[other.clone()]);
        for p in own.clone() {
            let gap_x = grid.gap_xy(at[0], offset[0], grid.x[p]);
            let gap_y = grid.gap_xy(at[1], offset[1], grid.y[p]);
            let gap_sq = gap_x * gap_x + gap_y * gap_y;
            if !reach.allows(gap_sq) {
                continue;
            }
            let z = grid.z[p];
            let z_reach = reach.z_reach(gap_sq);
            let limits = |shift, reach| grid.z_limits(z, shift, reach);

            let near = sweep.middle(limits(0.0, z_max), limits(0.0, z_reach));
            let shift = [shift_x, shift_y, 0.0];
            bin_candidates(grid, p, shift, window(near), binning, scratch, record);
            if z - z_reach < box_size - near_top {
                let near = sweep.top(limits(-box_size, z_max).0, limits(-box_size, z_reach).0);
                let shift = [shift_x, shift_y, -box_size];
                bin_candidates(grid, p, shift, window(near), binning, scratch, record);
            }
            if z + z_reach > near_top {
                let near = sweep.bottom(limits(box_size, z_max).1, limits(box_size, z_reach).1);
                let shift = [shift_x, shift_y, box_size];
                bin_candidates(grid, p, shift, window(near), binning, scratch, record);
            }
        }
    }
}

/// Finds z windows in a z-sorted column for a sequence of particles taken in
/// increasing z.
///
/// Each window is found in two steps: the window for the largest reach,
/// whose limits only increase from one particle to the next and so are
/// tracked by indices that only move forward, and then the particle's own,
/// narrower window inside it, by stepping in from its ends.
struct Sweep<'a> {
    zs: &'a [f64],
    /// First index with `z >= lo` of the widest window without wrapping.
    lo: usize,
    /// First index with `z > hi` of the widest window without wrapping.
    hi: usize,
    /// First index with `z >= lo` of the widest window wrapped from below.
    top: usize,
    /// First index with `z > hi` of the widest window wrapped from above.
    bottom: usize,
}

impl<'a> Sweep<'a> {
    fn new(zs: &'a [f64]) -> Self {
        Sweep { zs, lo: 0, hi: 0, top: 0, bottom: 0 }
    }

    /// Indices with `z` in `[near.0, near.1]`, given the widest limits
    /// `wide`, which contain `near` and must not decrease between calls.
    #[inline(always)]
    fn middle(&mut self, wide: (f64, f64), near: (f64, f64)) -> Range<usize> {
        let zs = self.zs;
        self.lo = advance(zs, self.lo, |z| z < wide.0);
        self.hi = advance(zs, self.hi.max(self.lo), |z| z <= wide.1);
        let start = advance(&zs[..self.hi], self.lo, |z| z < near.0);
        let end = retreat(&zs[start..], self.hi - start, |z| z > near.1) + start;
        start..end
    }

    /// Indices with `z >= near`, given the widest lower limit `wide <= near`,
    /// which must not decrease between calls.
    #[inline(always)]
    fn top(&mut self, wide: f64, near: f64) -> Range<usize> {
        self.top = advance(self.zs, self.top, |z| z < wide);
        advance(self.zs, self.top, |z| z < near)..self.zs.len()
    }

    /// Indices with `z <= near`, given the widest upper limit `wide >= near`,
    /// which must not decrease between calls.
    #[inline(always)]
    fn bottom(&mut self, wide: f64, near: f64) -> Range<usize> {
        self.bottom = advance(self.zs, self.bottom, |z| z <= wide);
        0..retreat(self.zs, self.bottom, |z| z > near)
    }
}

/// The first index from `i` at which `before(zs[index])` fails, or
/// `zs.len()`.
#[inline(always)]
fn advance(zs: &[f64], mut i: usize, before: impl Fn(f64) -> bool) -> usize {
    while i < zs.len() && before(zs[i]) {
        i += 1;
    }
    i
}

/// Steps `end` down while `after(zs[end - 1])` holds.
#[inline(always)]
fn retreat(zs: &[f64], mut end: usize, after: impl Fn(f64) -> bool) -> usize {
    while end > 0 && after(zs[end - 1]) {
        end -= 1;
    }
    end
}

/// Counts, through `record(bin, n)`, every unordered pair of grid positions
/// `p`, `q` in cell `c`, or in `c` and one of its half-shell neighbours,
/// whose nearest-image separation (of `q` from `p`) is kept by `binning`.
/// Every separation is wrapped individually (see [`Layout`]).
#[inline(always)]
fn visit_wrapped_cell_pairs<B: PairBinning>(
    grid: &CellGrid,
    c: usize,
    binning: &B,
    scratch: &mut Scratch,
    record: &mut impl FnMut(usize, u64),
) {
    let own = grid.cell(c);
    if own.is_empty() {
        return;
    }
    let reach = binning.reach();

    // Pairs within the cell. A cell is at most a third of the box wide, so
    // the plain difference is already the nearest image. Positions are sorted
    // by z, so the partners of p within reach follow it directly.
    let zs = &grid.z[own.clone()];
    for (i, &z) in zs.iter().enumerate() {
        let p = own.start + i;
        let above = advance(zs, i + 1, |zq| zq - z <= reach.z_max());
        bin_candidates(grid, p, [0.0; 3], p + 1..own.start + above, binning, scratch, record);
    }

    let at = grid.cell_coords(c);
    // Each separation is wrapped once, from the plain difference, as in the
    // definition of the nearest image.
    let (box_size, half_box) = (grid.box_size, grid.half_box);
    for offset in HALF_SHELL {
        let (other_cell, _) = grid.neighbour(at, offset);
        for p in own.clone() {
            let (x, y, z) = (grid.x[p], grid.y[p], grid.z[p]);
            for q in grid.cell(other_cell) {
                let d = [grid.x[q] - x, grid.y[q] - y, grid.z[q] - z]
                    .map(|d| min_image(d, box_size, half_box));
                if binning.keep(d) {
                    let kept = &mut scratch.kept;
                    let j = kept.len;
                    (kept.d[0][j], kept.d[1][j], kept.d[2][j]) = (d[0], d[1], d[2]);
                    kept.partition[j] = grid.partition[q];
                    kept.len += 1;
                    if kept.len == BLOCK {
                        scratch.flush(grid.partition[p], binning, record);
                    }
                }
            }
            scratch.flush(grid.partition[p], binning, record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform numbers in `[0, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Whether a separation lies within `reach`, by the plain definition.
    fn within(reach: Reach, [dx, dy, dz]: [f64; 3]) -> bool {
        match reach {
            Reach::Sphere(r) => dx * dx + dy * dy + dz * dz < r * r,
            Reach::Cylinder { r_p, pi } => (dx * dx + dy * dy < r_p * r_p) & (dz.abs() < pi),
        }
    }

    /// Bins every pair within reach by the ids of its members, which the
    /// tests store as the particles' partitions, and checks that each
    /// separation is the nearest-image one.
    struct PairIds<'a> {
        reach: Reach,
        coords: &'a [[f64; 3]],
        box_size: f64,
    }

    impl PairBinning for PairIds<'_> {
        fn reach(&self) -> Reach {
            self.reach
        }

        fn keep(&self, d: [f64; 3]) -> bool {
            within(self.reach, d)
        }

        fn n_bins(&self) -> usize {
            self.coords.len() * self.coords.len()
        }

        fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]) {
            let (p, n, l) = (partition as usize, self.coords.len(), self.box_size);
            for (j, bin) in bins[..kept.len].iter_mut().enumerate() {
                let q = kept.partition[j] as usize;
                assert_ne!(p, q, "a particle was paired with itself");
                for axis in 0..3 {
                    let d = min_image(self.coords[q][axis] - self.coords[p][axis], l, 0.5 * l);
                    assert!((kept.d[axis][j] - d).abs() <= 1e-9 * l, "wrong image");
                }
                *bin = p.min(q) * n + p.max(q);
            }
        }
    }

    /// How many times each pair `(i, j)`, `i < j`, is visited (at `i * n + j`).
    fn visits(coords: &[[f64; 3]], box_size: f64, reach: Reach, threads: usize) -> Vec<u32> {
        visits_using(Simd::detect(), None, coords, box_size, reach, threads)
    }

    fn visits_using(
        simd: Simd,
        steps: Option<i32>,
        coords: &[[f64; 3]],
        box_size: f64,
        reach: Reach,
        threads: usize,
    ) -> Vec<u32> {
        let ids: Vec<i32> = (0..coords.len() as i32).collect();
        let (r_xy, r_z) = match reach {
            Reach::Sphere(r) => (r, r),
            Reach::Cylinder { r_p, pi } => (r_p, pi),
        };
        let binning = PairIds { reach, coords, box_size };
        let n_bins = binning.n_bins();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let layout = Layout::with_steps(coords, &ids, box_size, r_xy, r_z, steps);
            let partial = layout.for_each_pair_using(
                simd,
                &binning,
                || vec![0; n_bins + 1],
                |h, b, n| h[b] += n as u32,
            );
            let mut total = vec![0; n_bins + 1];
            for h in partial {
                total.iter_mut().zip(h).for_each(|(t, c)| *t += c);
            }
            assert_eq!(total[n_bins], 0, "no pair should be discarded");
            total.truncate(n_bins);
            total
        })
    }

    /// The same by direct enumeration.
    fn brute_force_visits(coords: &[[f64; 3]], box_size: f64, reach: Reach) -> Vec<u32> {
        let n = coords.len();
        let mut counts = vec![0; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let d = [0, 1, 2]
                    .map(|a| min_image(coords[j][a] - coords[i][a], box_size, 0.5 * box_size));
                counts[i * n + j] = within(reach, d) as u32;
            }
        }
        counts
    }

    fn uniform(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| [0; 3].map(|_| rng.uniform() * box_size)).collect()
    }

    /// Tight clumps, so that many pairs are close and some columns are empty.
    fn clumped(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = Lcg(seed);
        let centres = uniform(5, box_size, seed + 1);
        (0..n)
            .map(|i| {
                centres[i % 5]
                    .map(|c| (c + 0.03 * box_size * (rng.uniform() - 0.5)).rem_euclid(box_size))
            })
            .map(|p| p.map(|c| if c >= box_size { 0.0 } else { c }))
            .collect()
    }

    /// Integer coordinates, many of them shared, so that separations fall
    /// exactly on the reach and on cell boundaries.
    fn lattice(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| [0; 3].map(|_| (rng.uniform() * box_size).floor())).collect()
    }

    /// Points on and next to the box faces and the column boundaries of an
    /// `n_columns`-wide grid.
    fn on_boundaries(box_size: f64, n_columns: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = Lcg(seed);
        let below = |v: f64| f64::from_bits(v.to_bits() - 1);
        let mut values = vec![0.0, f64::MIN_POSITIVE, below(box_size), 0.5 * box_size];
        for c in 1..n_columns {
            let edge = c as f64 * box_size / n_columns as f64;
            values.extend([edge, below(edge), f64::from_bits(edge.to_bits() + 1)]);
        }
        (0..400)
            .map(|_| [0; 3].map(|_| values[(rng.uniform() * values.len() as f64) as usize]))
            .collect()
    }

    fn check_every_pair_is_visited_once(coords: &[[f64; 3]], box_size: f64, reach: Reach) {
        let expected = brute_force_visits(coords, box_size, reach);
        let layouts = [(None, 1), (Some(1), 3), (Some(2), 1), (Some(2), 3)];
        for (steps, threads) in layouts {
            let got = visits_using(Simd::detect(), steps, coords, box_size, reach, threads);
            let wrong = (0..got.len()).find(|&i| got[i] != expected[i]);
            if let Some(i) = wrong {
                let (n, a, b) = (coords.len(), i / coords.len(), i % coords.len());
                panic!(
                    "{reach:?} in box {box_size} ({steps:?} steps, {threads} threads): pair ({a}, {b}) of {n} \
                     visited {} times, expected {}; positions {:?} {:?}",
                    got[i], expected[i], coords[a], coords[b]
                );
            }
        }
    }

    /// Reaches from tiny to beyond half the box, including those either side
    /// of the switch from columns to wrapped cells at a quarter of the box.
    fn reaches(box_size: f64) -> Vec<Reach> {
        let mut reaches: Vec<Reach> =
            [0.003, 0.02, 0.1, 0.19, 0.2499, 0.25, 0.2501, 0.33, 0.49, 0.7]
                .iter()
                .map(|f| Reach::Sphere(f * box_size))
                .collect();
        for (r_p, pi) in
            [(0.05, 0.2), (0.2, 0.05), (0.1, 0.3), (0.3, 0.1), (0.24, 0.24), (0.01, 0.6)]
        {
            reaches.push(Reach::Cylinder { r_p: r_p * box_size, pi: pi * box_size });
        }
        reaches
    }

    #[test]
    fn every_pair_within_reach_is_visited_once_uniform() {
        for (seed, box_size) in [(1, 100.0), (2, 1.0), (3, 7.3)] {
            let coords = uniform(300, box_size, seed);
            for reach in reaches(box_size) {
                check_every_pair_is_visited_once(&coords, box_size, reach);
            }
        }
    }

    #[test]
    fn every_pair_within_reach_is_visited_once_clumped() {
        let box_size = 50.0;
        let coords = clumped(300, box_size, 4);
        for reach in reaches(box_size) {
            check_every_pair_is_visited_once(&coords, box_size, reach);
        }
    }

    #[test]
    fn every_pair_within_reach_is_visited_once_on_a_lattice() {
        let box_size = 24.0;
        let coords = lattice(300, box_size, 5);
        for r in [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 12.0] {
            check_every_pair_is_visited_once(&coords, box_size, Reach::Sphere(r));
            check_every_pair_is_visited_once(
                &coords,
                box_size,
                Reach::Cylinder { r_p: r, pi: 2.0 },
            );
            check_every_pair_is_visited_once(
                &coords,
                box_size,
                Reach::Cylinder { r_p: 2.0, pi: r },
            );
        }
    }

    #[test]
    fn every_pair_within_reach_is_visited_once_on_boundaries() {
        let box_size = 30.0;
        for r in [1.0, 3.0, 5.0, 7.5] {
            // The columns for reach r are 2 * box / r per axis.
            let n_columns = (2.0 * box_size / r) as usize;
            let coords = on_boundaries(box_size, n_columns, 6);
            check_every_pair_is_visited_once(&coords, box_size, Reach::Sphere(r));
            check_every_pair_is_visited_once(&coords, box_size, Reach::Cylinder { r_p: r, pi: r });
        }
    }

    #[test]
    fn every_instruction_set_gives_the_same_pairs() {
        let mut simds = vec![Simd::Baseline];
        #[cfg(target_arch = "x86_64")]
        if Simd::detect() == Simd::Avx2 {
            simds.push(Simd::Avx2);
        }
        let box_size = 40.0;
        let coords = clumped(300, box_size, 8);
        for reach in
            [Reach::Sphere(4.0), Reach::Cylinder { r_p: 2.0, pi: 7.0 }, Reach::Sphere(15.0)]
        {
            let expected = brute_force_visits(&coords, box_size, reach);
            for &simd in &simds {
                for steps in [Some(1), Some(2)] {
                    let got = visits_using(simd, steps, &coords, box_size, reach, 2);
                    assert_eq!(got, expected, "{simd:?} {steps:?}");
                }
            }
        }
    }

    #[test]
    fn coincident_particles_are_all_paired() {
        let coords = vec![[3.0, 4.0, 5.0]; 70];
        let visited = visits(&coords, 20.0, Reach::Sphere(1.0), 2);
        assert_eq!(visited.iter().sum::<u32>(), 70 * 69 / 2);
    }

    #[test]
    fn empty_and_single_particle_catalogues_have_no_pairs() {
        for coords in [vec![], vec![[1.0, 2.0, 3.0]]] {
            assert!(visits(&coords, 10.0, Reach::Sphere(2.0), 1).iter().all(|&c| c == 0));
        }
    }

    #[test]
    fn layout_depends_on_the_reach_and_the_density() {
        let columns = |layout: Layout| match layout {
            Layout::Columns { grid, steps } => Some((grid.n_xy, steps)),
            Layout::WrappedCells(_) => None,
        };
        // 10^5 particles in a box of side 100: a mean density of 0.1.
        let coords = vec![[1.0, 2.0, 3.0]; 100_000];
        let parts = vec![0; coords.len()];
        let layout = |r_xy, r_z| Layout::new(&coords, &parts, 100.0, r_xy, r_z);
        assert_eq!(columns(layout(25.0, 25.0)), Some((8, 2)));
        assert_eq!(columns(layout(10.0, 25.0)), Some((20, 2)));
        assert_eq!(columns(layout(7.0, 1.0)), Some((14, 1)));
        assert_eq!(columns(layout(25.1, 1.0)), None);
        assert_eq!(columns(layout(1.0, 25.1)), None);
        assert_eq!(columns(layout(1e-3, 1e-3)), Some((MAX_COLUMNS_PER_AXIS, 1)));
        // Just either side of the expected window at which half-width
        // columns are used: 0.1 * (r / 2)^2 * 2 r = 8 at r = 160^(1/3).
        let r = 160f64.cbrt();
        assert_eq!(columns(layout(r * 0.999, r * 0.999)).map(|c| c.1), Some(1));
        assert_eq!(columns(layout(r * 1.001, r * 1.001)).map(|c| c.1), Some(2));
    }

    #[test]
    fn column_offsets_and_their_mirrors_cover_the_neighbourhood_once() {
        for steps in [1, 2] {
            let offsets = column_offsets(steps);
            let side = 2 * steps as usize + 1;
            assert_eq!(offsets.len(), (side * side - 1) / 2);
            let mut seen = std::collections::HashSet::new();
            for [dx, dy, dz] in offsets {
                assert_eq!(dz, 0);
                assert!(seen.insert([dx, dy]) && seen.insert([-dx, -dy]));
                assert!(dx.abs() <= steps && dy.abs() <= steps);
            }
            assert_eq!(seen.len(), side * side - 1);
        }
    }

    #[test]
    fn sweep_finds_the_same_windows_as_a_filter() {
        let mut rng = Lcg(7);
        for len in [0, 1, 2, 5, 40, 300] {
            let mut zs: Vec<f64> = (0..len).map(|_| (rng.uniform() * 50.0).floor() / 5.0).collect();
            zs.sort_by(f64::total_cmp);
            let mut sweep = Sweep::new(&zs);
            let mut centre = -2.0;
            while centre < 12.0 {
                centre += rng.uniform() * 0.3;
                let (wide, near) = (1.5, 1.5 * rng.uniform());
                let range =
                    sweep.middle((centre - wide, centre + wide), (centre - near, centre + near));
                let expected: Vec<usize> = (0..len)
                    .filter(|&i| (centre - near..=centre + near).contains(&zs[i]))
                    .collect();
                assert_eq!(range.collect::<Vec<_>>(), expected);

                let top = sweep.top(centre + 8.0 - wide, centre + 8.0 - near);
                assert_eq!(
                    top.collect::<Vec<_>>(),
                    (0..len).filter(|&i| zs[i] >= centre + 8.0 - near).collect::<Vec<_>>()
                );
                let bottom = sweep.bottom(centre - 8.0 + wide, centre - 8.0 + near);
                assert_eq!(
                    bottom.collect::<Vec<_>>(),
                    (0..len).filter(|&i| zs[i] <= centre - 8.0 + near).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn advance_and_retreat_stop_at_the_first_failure() {
        let zs = [1.0, 2.0, 2.0, 3.0];
        assert_eq!(advance(&zs, 0, |z| z < 2.0), 1);
        assert_eq!(advance(&zs, 2, |z| z < 2.0), 2);
        assert_eq!(advance(&zs, 0, |z| z < 10.0), 4);
        assert_eq!(advance(&zs, 4, |_| true), 4);
        assert_eq!(retreat(&zs, 4, |z| z > 2.0), 3);
        assert_eq!(retreat(&zs, 4, |z| z > 0.0), 0);
        assert_eq!(retreat(&zs, 2, |z| z > 5.0), 2);
    }

    #[test]
    fn pack_flags_sets_one_bit_per_flag() {
        let mut flags = [0u8; BLOCK];
        assert_eq!(pack_flags(&flags), 0);
        for k in 0..BLOCK {
            flags = [0; BLOCK];
            flags[k] = 1;
            assert_eq!(pack_flags(&flags), 1u64 << k);
        }
        let mut rng = Lcg(12345);
        for _ in 0..1000 {
            let expected = rng.uniform().to_bits() ^ (rng.uniform().to_bits() << 11);
            let flags: [u8; BLOCK] = std::array::from_fn(|k| ((expected >> k) & 1) as u8);
            assert_eq!(pack_flags(&flags), expected);
        }
    }

    #[test]
    fn reach_never_excludes_a_pair_the_distance_test_accepts() {
        let reaches = [Reach::Sphere(3.0), Reach::Cylinder { r_p: 3.0, pi: 1.5 }];
        let mut rng = Lcg(99);
        for reach in reaches {
            for _ in 0..200_000 {
                let d = [0, 1, 2].map(|_| 6.0 * rng.uniform() - 3.0);
                if !within(reach, d) {
                    continue;
                }
                // Any lower bounds on |dx| and |dy| up to their true values.
                let gap = [d[0].abs() * rng.uniform(), d[1].abs() * rng.uniform()];
                let gap_sq = gap[0] * gap[0] + gap[1] * gap[1];
                assert!(reach.allows(gap_sq), "{d:?}");
                assert!(d[2].abs() <= reach.z_reach(gap_sq), "{d:?}");
                assert!(d[2].abs() <= reach.z_max());
            }
        }
    }
}
