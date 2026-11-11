//! Pair counters.
//!
//! Every counter finds pairs with [`Layout::for_each_pair`] and differs only
//! in how a pair's separation is mapped to a histogram bin (its
//! [`PairBinning`]). Pairs are split by whether the two particles share a
//! partition ("auto") or not ("cross").

use std::sync::atomic::{AtomicU64, Ordering};

use crate::binning::{mu_bin_value, squares, BinTable, TableBins};
use crate::traverse::{pack_flags, Kept, Layout, PairBinning, Reach, BLOCK};

/// Histogram of all pairs in `layout` binned by `binning`.
fn histogram_pairs(layout: &Layout, binning: impl PairBinning) -> Vec<f64> {
    // One extra bin receives the discarded pairs.
    let n_bins = binning.n_bins();
    let partial =
        layout.for_each_pair(&binning, || vec![0.0; n_bins + 1], |h, b, n| h[b] += n as f64);
    let mut counts =
        partial.into_iter().reduce(add_histograms).unwrap_or_else(|| vec![0.0; n_bins + 1]);
    counts.truncate(n_bins);
    counts
}

/// Element-wise sum of two histograms of the same length.
pub fn add_histograms(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
    a
}

/// Splits `[auto..., cross...]` into its two halves.
fn split_auto_cross(mut counts: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let cross = counts.split_off(counts.len() / 2);
    (counts, cross)
}

/// Offset of a pair's bin in an `[auto..., cross...]` histogram with
/// `n_half` bins per half, for members in partitions `a` and `b`.
#[inline(always)]
fn auto_or_cross(a: i32, b: i32, n_half: usize) -> usize {
    (a != b) as usize * n_half
}

/// Squared lengths of the first `n` separations in `d`, using x and y only
/// when `AXES` is 2 and all three axes when it is 3.
#[inline(always)]
fn squared_lengths<const AXES: usize>(d: &[[f64; BLOCK]; 3], n: usize) -> [f64; BLOCK] {
    let mut out = [0.0; BLOCK];
    for j in 0..n {
        out[j] = d[0][j] * d[0][j] + d[1][j] * d[1][j];
        if AXES == 3 {
            out[j] += d[2][j] * d[2][j];
        }
    }
    out
}

/// Binning by 3D separation `r`, split into auto and cross pairs.
struct Radial<'a> {
    r_bins: TableBins<'a>,
}

impl PairBinning for Radial<'_> {
    fn reach(&self) -> Reach {
        Reach::Sphere(self.r_bins.hi().sqrt())
    }

    #[inline(always)]
    fn keep(&self, [dx, dy, dz]: [f64; 3]) -> bool {
        self.r_bins.contains(dx * dx + dy * dy + dz * dz)
    }

    fn n_bins(&self) -> usize {
        2 * self.r_bins.n_bins()
    }

    #[inline(always)]
    fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]) {
        let n = kept.len;
        let r_sq = squared_lengths::<3>(&kept.d, n);
        let mut r_bin = [0; BLOCK];
        self.r_bins.find_all(&r_sq[..n], &mut r_bin[..n]);
        let n_r = self.r_bins.n_bins();
        for j in 0..n {
            bins[j] = r_bin[j] as usize + auto_or_cross(partition, kept.partition[j], n_r);
        }
    }

    const COUNTS_BLOCKS: bool = true;

    /// Works down the bins from the outermost, where most pairs are: the
    /// kept candidates at or above a bin's lower edge that are not yet
    /// counted are in that bin. Each step compares the whole block at once,
    /// and the pairs are counted with population counts of bit masks.
    #[inline(always)]
    fn count_block(
        &self,
        partition: i32,
        d: &[[f64; BLOCK]; 3],
        partitions: &[i32; BLOCK],
        kept: u64,
        record: &mut impl FnMut(usize, u64),
    ) {
        let r_sq = squared_lengths::<3>(d, BLOCK);
        let mut same = [0u8; BLOCK];
        for (s, &b) in same.iter_mut().zip(partitions) {
            *s = (b == partition) as u8;
        }
        let same = pack_flags(&same);

        let (edges, n_r) = (self.r_bins.edges(), self.r_bins.n_bins());
        let mut remaining = kept;
        let mut bin = n_r;
        let mut at_or_above = [0u8; BLOCK];
        while remaining != 0 {
            // Every kept separation is at or above the first edge, so the
            // loop ends by bin 0.
            bin -= 1;
            let lower = edges[bin];
            for (a, &r_sq) in at_or_above.iter_mut().zip(&r_sq) {
                *a = (r_sq >= lower) as u8;
            }
            let in_bin = pack_flags(&at_or_above) & remaining;
            remaining &= !in_bin;
            record(bin, (in_bin & same).count_ones() as u64);
            record(bin + n_r, (in_bin & !same).count_ones() as u64);
        }
    }
}

/// Binning by transverse separation `r_p` and line-of-sight separation
/// `pi = |dz|`, split into auto and cross pairs.
struct Projected<'a> {
    rp_bins: TableBins<'a>,
    pi_bins: TableBins<'a>,
}

impl PairBinning for Projected<'_> {
    fn reach(&self) -> Reach {
        Reach::Cylinder { r_p: self.rp_bins.hi().sqrt(), pi: self.pi_bins.hi() }
    }

    #[inline(always)]
    fn keep(&self, [dx, dy, dz]: [f64; 3]) -> bool {
        self.rp_bins.contains(dx * dx + dy * dy) & self.pi_bins.contains(dz.abs())
    }

    fn n_bins(&self) -> usize {
        2 * self.rp_bins.n_bins() * self.pi_bins.n_bins()
    }

    #[inline(always)]
    fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]) {
        let n = kept.len;
        let rp_sq = squared_lengths::<2>(&kept.d, n);
        let mut pi = [0.0; BLOCK];
        for (pi, dz) in pi[..n].iter_mut().zip(&kept.d[2]) {
            *pi = dz.abs();
        }
        let (mut rp_bin, mut pi_bin) = ([0; BLOCK], [0; BLOCK]);
        self.rp_bins.find_all(&rp_sq[..n], &mut rp_bin[..n]);
        self.pi_bins.find_all(&pi[..n], &mut pi_bin[..n]);
        let n_pi = self.pi_bins.n_bins();
        let n_half = self.n_bins() / 2;
        for j in 0..n {
            let bin = rp_bin[j] as usize * n_pi + pi_bin[j] as usize;
            bins[j] = bin + auto_or_cross(partition, kept.partition[j], n_half);
        }
    }
}

/// Binning by separation `s` and `mu = |dz| / s`, split into auto and cross
/// pairs.
struct RedshiftSpace<'a> {
    s_bins: TableBins<'a>,
    n_mu: usize,
    mu_max: f64,
}

impl PairBinning for RedshiftSpace<'_> {
    fn reach(&self) -> Reach {
        Reach::Sphere(self.s_bins.hi().sqrt())
    }

    #[inline(always)]
    fn keep(&self, [dx, dy, dz]: [f64; 3]) -> bool {
        self.s_bins.contains(dx * dx + dy * dy + dz * dz)
    }

    fn n_bins(&self) -> usize {
        2 * self.s_bins.n_bins() * self.n_mu
    }

    #[inline(always)]
    fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]) {
        let n = kept.len;
        let s_sq = squared_lengths::<3>(&kept.d, n);
        // mu bins in [-1, n_mu), where -1 means the pair is discarded.
        let mut mu_bin = [0i32; BLOCK];
        let mu = |j: usize| kept.d[2][j].abs() / s_sq[j].sqrt();
        if self.mu_max == 1.0 {
            // The usual case, in which the division by mu_max can be left out
            // without changing any result.
            for (j, bin) in mu_bin[..n].iter_mut().enumerate() {
                *bin = mu_bin_value(mu(j), 1.0, self.n_mu) as i32;
            }
        } else {
            for (j, bin) in mu_bin[..n].iter_mut().enumerate() {
                *bin = mu_bin_value(mu(j), self.mu_max, self.n_mu) as i32;
            }
        }
        let mut s_bin = [0; BLOCK];
        self.s_bins.find_all(&s_sq[..n], &mut s_bin[..n]);
        let n_half = self.n_bins() / 2;
        for j in 0..n {
            let bin = s_bin[j] as usize * self.n_mu + mu_bin[j].max(0) as usize;
            let bin = bin + auto_or_cross(partition, kept.partition[j], n_half);
            bins[j] = if mu_bin[j] < 0 { 2 * n_half } else { bin };
        }
    }
}

/// Binning by `r` and by the partitions of both members: pairs within a
/// partition get bins `r * k + a`, and pairs between partitions `a < b` get
/// bins `n_r * k + (r * k + a) * k + b`.
struct PartitionMatrix<'a> {
    r_bins: TableBins<'a>,
    k: usize,
}

impl PairBinning for PartitionMatrix<'_> {
    fn reach(&self) -> Reach {
        Reach::Sphere(self.r_bins.hi().sqrt())
    }

    #[inline(always)]
    fn keep(&self, [dx, dy, dz]: [f64; 3]) -> bool {
        self.r_bins.contains(dx * dx + dy * dy + dz * dz)
    }

    fn n_bins(&self) -> usize {
        self.r_bins.n_bins() * self.k * (self.k + 1)
    }

    #[inline(always)]
    fn bin_kept(&self, partition: i32, kept: &Kept, bins: &mut [usize; BLOCK]) {
        let n = kept.len;
        let r_sq = squared_lengths::<3>(&kept.d, n);
        let mut r_bin = [0; BLOCK];
        self.r_bins.find_all(&r_sq[..n], &mut r_bin[..n]);
        let (n_r, k, a) = (self.r_bins.n_bins(), self.k, partition as usize);
        for j in 0..n {
            let (r, b) = (r_bin[j] as usize, kept.partition[j] as usize);
            bins[j] = if a == b { r * k + a } else { n_r * k + (r * k + a.min(b)) * k + a.max(b) };
        }
    }
}

/// Pair counts binned by 3D separation `r`, returned as `(auto, cross)`,
/// each of length `r_edges.len() - 1`.
pub fn radial_counts(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let r_max = r_edges[r_edges.len() - 1];
    let table = BinTable::new(&squares(r_edges));
    let layout = Layout::new(coords, partition, box_size, r_max, r_max);
    split_auto_cross(histogram_pairs(&layout, Radial { r_bins: table.bins() }))
}

/// Pair counts binned by transverse separation `r_p` (in the x-y plane) and
/// line-of-sight separation `pi = |dz|`, returned as `(auto, cross)` with
/// shape `(n_rp, n_pi)` flattened row-major.
pub fn projected_counts(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    rp_edges: &[f64],
    pi_edges: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let rp_max = rp_edges[rp_edges.len() - 1];
    let pi_max = pi_edges[pi_edges.len() - 1];
    let (rp_table, pi_table) = (BinTable::new(&squares(rp_edges)), BinTable::new(pi_edges));
    let layout = Layout::new(coords, partition, box_size, rp_max, pi_max);
    let binning = Projected { rp_bins: rp_table.bins(), pi_bins: pi_table.bins() };
    split_auto_cross(histogram_pairs(&layout, binning))
}

/// Pair counts binned by separation `s` and `mu = |dz| / s`, the cosine of
/// the angle to the line of sight (the z axis), returned as `(auto, cross)`
/// with shape `(n_s, n_mu)` flattened row-major. Pairs with `mu >= mu_max`
/// are discarded.
pub fn redshift_space_counts(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    s_edges: &[f64],
    n_mu: usize,
    mu_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let s_max = s_edges[s_edges.len() - 1];
    let table = BinTable::new(&squares(s_edges));
    let layout = Layout::new(coords, partition, box_size, s_max, s_max);
    let binning = RedshiftSpace { s_bins: table.bins(), n_mu, mu_max };
    split_auto_cross(histogram_pairs(&layout, binning))
}

/// Pair counts binned by `r` and resolved by the partitions of both members.
///
/// Returns `(auto, cross)`: `auto[r * k + a]` counts pairs inside partition
/// `a`, and `cross[(r * k + a) * k + b]` counts pairs between partitions
/// `a < b` (entries with `a >= b` are zero). Partition ids must lie in
/// `[0, k)`.
///
/// The `(n_r, k, k)` array can be large (hundreds of MB for k ~ 1000), so
/// instead of giving every worker its own copy the counts go straight into
/// one shared array of atomic counters.
pub fn partition_pair_matrix(
    coords: &[[f64; 3]],
    partition: &[i32],
    box_size: f64,
    r_edges: &[f64],
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_r = r_edges.len() - 1;
    let r_max = r_edges[n_r];
    let table = BinTable::new(&squares(r_edges));
    let layout = Layout::new(coords, partition, box_size, r_max, r_max);
    let binning = PartitionMatrix { r_bins: table.bins(), k };

    let n_auto = n_r * k;
    let counters: Vec<AtomicU64> = (0..n_auto + n_r * k * k).map(|_| AtomicU64::new(0)).collect();
    layout.for_each_pair(
        &binning,
        || (),
        |_, i, n| {
            counters[i].fetch_add(n, Ordering::Relaxed);
        },
    );

    let mut counts: Vec<f64> = counters.into_iter().map(|c| c.into_inner() as f64).collect();
    let cross = counts.split_off(n_auto);
    (counts, cross)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::binning::{find_bin, mu_bin};
    use crate::grid::min_image;

    /// Deterministic pseudo-random points with a clustered component, so that
    /// small separations are well populated.
    pub fn test_catalogue(n: usize, box_size: f64, k: i32, seed: u64) -> (Vec<[f64; 3]>, Vec<i32>) {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut coords = Vec::with_capacity(n);
        let mut parts = Vec::with_capacity(n);
        for i in 0..n {
            let p = if i % 2 == 1 {
                let base: [f64; 3] = coords[i - 1];
                base.map(|c| (c + 0.05 * box_size * (uniform() - 0.5)).rem_euclid(box_size))
            } else {
                [uniform() * box_size, uniform() * box_size, uniform() * box_size]
            };
            coords.push(p.map(|c| if c >= box_size { 0.0 } else { c }));
            parts.push((uniform() * k as f64) as i32);
        }
        (coords, parts)
    }

    /// All pairs by direct enumeration.
    pub fn brute_force_pairs(
        coords: &[[f64; 3]],
        box_size: f64,
        mut visit: impl FnMut(usize, usize, [f64; 3]),
    ) {
        let half = 0.5 * box_size;
        for i in 0..coords.len() {
            for j in i + 1..coords.len() {
                let d = [0, 1, 2].map(|a| min_image(coords[j][a] - coords[i][a], box_size, half));
                visit(i, j, d);
            }
        }
    }

    const BOXES_AND_EDGES: [(f64, &[f64]); 3] = [
        (100.0, &[0.5, 2.0, 5.0, 9.0]),
        (30.0, &[0.0, 1.0, 4.0, 11.0]),
        (20.0, &[1.0, 3.0, 7.0, 9.5]),
    ];

    #[test]
    fn radial_counts_match_brute_force() {
        for (box_size, edges) in BOXES_AND_EDGES {
            let (coords, parts) = test_catalogue(600, box_size, 4, 11);
            let n_r = edges.len() - 1;
            let (mut auto, mut cross) = (vec![0.0; n_r], vec![0.0; n_r]);
            brute_force_pairs(&coords, box_size, |i, j, [dx, dy, dz]| {
                if let Some(b) = find_bin((dx * dx + dy * dy + dz * dz).sqrt(), edges) {
                    if parts[i] == parts[j] {
                        auto[b] += 1.0
                    } else {
                        cross[b] += 1.0
                    }
                }
            });
            assert_eq!(radial_counts(&coords, &parts, box_size, edges), (auto, cross));
        }
    }

    #[test]
    fn projected_counts_match_brute_force() {
        let pi_edges = [0.0, 2.0, 6.0, 9.0];
        for (box_size, rp_edges) in BOXES_AND_EDGES {
            let (coords, parts) = test_catalogue(600, box_size, 3, 12);
            let n_pi = pi_edges.len() - 1;
            let n = (rp_edges.len() - 1) * n_pi;
            let (mut auto, mut cross) = (vec![0.0; n], vec![0.0; n]);
            brute_force_pairs(&coords, box_size, |i, j, [dx, dy, dz]| {
                let rp = (dx * dx + dy * dy).sqrt();
                if let (Some(a), Some(b)) = (find_bin(rp, rp_edges), find_bin(dz.abs(), &pi_edges))
                {
                    let hist = if parts[i] == parts[j] { &mut auto } else { &mut cross };
                    hist[a * n_pi + b] += 1.0;
                }
            });
            assert_eq!(
                projected_counts(&coords, &parts, box_size, rp_edges, &pi_edges),
                (auto, cross)
            );
        }
    }

    #[test]
    fn redshift_space_counts_match_brute_force() {
        for (box_size, edges) in BOXES_AND_EDGES {
            for mu_max in [1.0, 0.55] {
                let (coords, parts) = test_catalogue(600, box_size, 3, 13);
                let n_mu = 4;
                let n = (edges.len() - 1) * n_mu;
                let (mut auto, mut cross) = (vec![0.0; n], vec![0.0; n]);
                brute_force_pairs(&coords, box_size, |i, j, [dx, dy, dz]| {
                    let s = (dx * dx + dy * dy + dz * dz).sqrt();
                    if let (Some(a), Some(b)) =
                        (find_bin(s, edges), mu_bin(dz.abs() / s, mu_max, n_mu))
                    {
                        let hist = if parts[i] == parts[j] { &mut auto } else { &mut cross };
                        hist[a * n_mu + b] += 1.0;
                    }
                });
                let got = redshift_space_counts(&coords, &parts, box_size, edges, n_mu, mu_max);
                assert_eq!(got, (auto, cross));
            }
        }
    }

    #[test]
    fn partition_matrix_sums_to_radial_counts() {
        for (box_size, edges) in BOXES_AND_EDGES {
            let k = 5;
            let (coords, parts) = test_catalogue(600, box_size, k as i32, 14);
            let (auto, cross) = partition_pair_matrix(&coords, &parts, box_size, edges, k);
            let (auto_total, cross_total) = radial_counts(&coords, &parts, box_size, edges);
            for r in 0..edges.len() - 1 {
                let a: f64 = auto[r * k..(r + 1) * k].iter().sum();
                let c: f64 = cross[r * k * k..(r + 1) * k * k].iter().sum();
                assert_eq!((a, c), (auto_total[r], cross_total[r]));
                for i in 0..k {
                    for j in 0..=i {
                        assert_eq!(cross[(r * k + i) * k + j], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn empty_and_single_particle_catalogues_have_no_pairs() {
        let edges = [1.0, 2.0];
        assert_eq!(radial_counts(&[], &[], 10.0, &edges), (vec![0.0], vec![0.0]));
        assert_eq!(radial_counts(&[[1.0, 2.0, 3.0]], &[0], 10.0, &edges), (vec![0.0], vec![0.0]));
    }

    /// Random kept blocks for `binning`: separations it keeps, with random
    /// partitions in `[0, k)`.
    fn kept_blocks(binning: &impl PairBinning, scale: f64, k: i32, seed: u64) -> Vec<(i32, Kept)> {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..=BLOCK)
            .map(|len| {
                let mut kept = Kept { d: [[0.0; BLOCK]; 3], partition: [0; BLOCK], len: 0 };
                let mut attempt = 0;
                while kept.len < len {
                    attempt += 1;
                    let d = [0; 3].map(|_| scale * (2.0 * uniform() - 1.0));
                    // Some separations along the line of sight and some zero.
                    let d = match attempt % 17 {
                        3 => [0.0, 0.0, d[2]],
                        5 => [0.0; 3],
                        _ => d,
                    };
                    if binning.keep(d) {
                        let j = kept.len;
                        (kept.d[0][j], kept.d[1][j], kept.d[2][j]) = (d[0], d[1], d[2]);
                        kept.partition[j] = (uniform() * k as f64) as i32;
                        kept.len += 1;
                    }
                }
                ((uniform() * k as f64) as i32, kept)
            })
            .collect()
    }

    /// Checks `binning.bin_kept` against `expected(partition_p, partition_q,
    /// d)` for every pair of random blocks of every length.
    fn check_bin_kept(
        binning: &impl PairBinning,
        scale: f64,
        k: i32,
        expected: impl Fn(i32, i32, [f64; 3]) -> usize,
    ) {
        for (a, kept) in kept_blocks(binning, scale, k, 3) {
            let mut bins = [usize::MAX; BLOCK];
            binning.bin_kept(a, &kept, &mut bins);
            for (j, &bin) in bins[..kept.len].iter().enumerate() {
                let d = [kept.d[0][j], kept.d[1][j], kept.d[2][j]];
                assert_eq!(bin, expected(a, kept.partition[j], d), "d = {d:?}");
                assert!(bin <= binning.n_bins());
            }
        }
    }

    fn squared_edges(edges: &[f64]) -> Vec<f64> {
        edges.iter().map(|e| e * e).collect()
    }

    #[test]
    fn radial_bin_kept_matches_each_pair() {
        let edges = [0.0, 0.5, 1.0, 1.7, 3.0];
        let table = BinTable::new(&squares(&edges));
        let binning = Radial { r_bins: table.bins() };
        let n_r = edges.len() - 1;
        check_bin_kept(&binning, 3.0, 3, |a, b, [dx, dy, dz]| {
            let bin = find_bin(dx * dx + dy * dy + dz * dz, &squared_edges(&edges)).unwrap();
            bin + if a == b { 0 } else { n_r }
        });
    }

    #[test]
    fn radial_count_block_matches_bin_kept() {
        let edges = [0.0, 0.5, 1.0, 1.7, 3.0];
        let table = BinTable::new(&squares(&edges));
        let binning = Radial { r_bins: table.bins() };
        let mut state = 5u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for trial in 0..500 {
            let d: [[f64; BLOCK]; 3] =
                [0; 3].map(|_| std::array::from_fn(|_| 6.0 * uniform() - 3.0));
            let partitions: [i32; BLOCK] = std::array::from_fn(|_| (uniform() * 3.0) as i32);
            let partition = trial % 3;
            // Kept candidates are those in range, less a random subset, as
            // when a block runs past the end of a window.
            let mut kept_mask = (uniform().to_bits() << 11) | uniform().to_bits();
            let mut kept = Kept { d: [[0.0; BLOCK]; 3], partition: [0; BLOCK], len: 0 };
            for k in 0..BLOCK {
                let dk = [d[0][k], d[1][k], d[2][k]];
                if !binning.keep(dk) {
                    kept_mask &= !(1 << k);
                } else if kept_mask >> k & 1 == 1 {
                    let j = kept.len;
                    (kept.d[0][j], kept.d[1][j], kept.d[2][j]) = (dk[0], dk[1], dk[2]);
                    kept.partition[j] = partitions[k];
                    kept.len += 1;
                }
            }
            let mut expected = vec![0u64; binning.n_bins()];
            let mut bins = [0; BLOCK];
            binning.bin_kept(partition, &kept, &mut bins);
            bins[..kept.len].iter().for_each(|&b| expected[b] += 1);

            let mut counted = vec![0u64; binning.n_bins()];
            binning.count_block(partition, &d, &partitions, kept_mask, &mut |b, n| counted[b] += n);
            assert_eq!(counted, expected);
        }
    }

    #[test]
    fn projected_bin_kept_matches_each_pair() {
        let (rp_edges, pi_edges) = ([0.2, 1.0, 2.0], [0.0, 0.5, 1.5, 4.0]);
        let (rp_table, pi_table) = (BinTable::new(&squares(&rp_edges)), BinTable::new(&pi_edges));
        let binning = Projected { rp_bins: rp_table.bins(), pi_bins: pi_table.bins() };
        let n_half = 2 * 3;
        check_bin_kept(&binning, 4.0, 2, |a, b, [dx, dy, dz]| {
            let rp = find_bin(dx * dx + dy * dy, &squared_edges(&rp_edges)).unwrap();
            let pi = find_bin(dz.abs(), &pi_edges).unwrap();
            rp * 3 + pi + if a == b { 0 } else { n_half }
        });
    }

    #[test]
    fn redshift_space_bin_kept_matches_each_pair() {
        let edges = [0.0, 0.3, 1.0, 2.5];
        let table = BinTable::new(&squares(&edges));
        for (n_mu, mu_max) in [(5, 1.0), (4, 0.6), (1, 1.0), (7, 0.99)] {
            let binning = RedshiftSpace { s_bins: table.bins(), n_mu, mu_max };
            let n_half = 3 * n_mu;
            check_bin_kept(&binning, 2.5, 4, |a, b, [dx, dy, dz]| {
                let s_sq = dx * dx + dy * dy + dz * dz;
                let s = find_bin(s_sq, &squared_edges(&edges)).unwrap();
                match mu_bin(dz.abs() / s_sq.sqrt(), mu_max, n_mu) {
                    Some(mu) => s * n_mu + mu + if a == b { 0 } else { n_half },
                    None => 2 * n_half,
                }
            });
        }
    }

    #[test]
    fn partition_matrix_bin_kept_matches_each_pair() {
        let edges = [0.1, 1.0, 2.0];
        let table = BinTable::new(&squares(&edges));
        let k = 6;
        let binning = PartitionMatrix { r_bins: table.bins(), k };
        check_bin_kept(&binning, 2.0, k as i32, |a, b, [dx, dy, dz]| {
            let r = find_bin(dx * dx + dy * dy + dz * dz, &squared_edges(&edges)).unwrap();
            let (a, b) = (a as usize, b as usize);
            if a == b {
                r * k + a
            } else {
                2 * k + (r * k + a.min(b)) * k + a.max(b)
            }
        });
    }

    /// Dense clumps, so that many z windows hold more than one block of
    /// candidates.
    fn dense_clumps(n: usize, box_size: f64, k: i32, seed: u64) -> (Vec<[f64; 3]>, Vec<i32>) {
        let (centres, _) = test_catalogue(8, box_size, 1, seed);
        let (offsets, parts) = test_catalogue(n, 0.08 * box_size, k, seed + 1);
        let coords = offsets
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let c = centres[i % 8];
                [0, 1, 2].map(|a| {
                    let v = (c[a] + o[a]).rem_euclid(box_size);
                    if v >= box_size {
                        0.0
                    } else {
                        v
                    }
                })
            })
            .collect();
        (coords, parts)
    }

    #[test]
    fn counts_match_brute_force_when_windows_span_many_blocks() {
        let box_size = 50.0;
        let (coords, parts) = dense_clumps(3000, box_size, 3, 21);
        let edges = [0.0, 0.2, 0.7, 1.5, 3.0, 6.0];
        let n_r = edges.len() - 1;
        let (mut auto, mut cross) = (vec![0.0; n_r], vec![0.0; n_r]);
        let n_mu = 4;
        let (mut s_auto, mut s_cross) = (vec![0.0; n_r * n_mu], vec![0.0; n_r * n_mu]);
        let sq_edges = squared_edges(&edges);
        brute_force_pairs(&coords, box_size, |i, j, [dx, dy, dz]| {
            let r_sq = dx * dx + dy * dy + dz * dz;
            if let Some(b) = find_bin(r_sq, &sq_edges) {
                let same = parts[i] == parts[j];
                *if same { &mut auto[b] } else { &mut cross[b] } += 1.0;
                if let Some(m) = mu_bin(dz.abs() / r_sq.sqrt(), 1.0, n_mu) {
                    *if same { &mut s_auto[b * n_mu + m] } else { &mut s_cross[b * n_mu + m] } +=
                        1.0;
                }
            }
        });
        assert!(auto.iter().sum::<f64>() > 1e5, "the test needs dense windows");
        assert_eq!(radial_counts(&coords, &parts, box_size, &edges), (auto, cross));
        let got = redshift_space_counts(&coords, &parts, box_size, &edges, n_mu, 1.0);
        assert_eq!(got, (s_auto, s_cross));
    }
}
