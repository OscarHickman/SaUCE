# Changelog

## 0.3.2 (2026-11-11)

- `count_npoint` can count one of `n_shards` shares of the work for N >= 3,
  to split a calculation across jobs. The shares sum to the full count.
- New `count_pairs_1d_partition_matrix`: pair counts for every pair of
  partitions.
- New `count_npoint3_smu` and `compute_npcf3_smu`: triplets binned by longest
  side and its orientation to the line of sight.
- The estimators require `N <= m <= k`.
- Faster pair and N-point counting. Counts are unchanged.
- The core was rewritten and is tested against brute force and Corrfunc.

## 0.3.1 (2026-07-04)

- Fixed the bin lookup for N >= 3: tuples with a diameter close to a bin edge
  could be counted in the neighbouring bin. Pair counts were not affected.
- Invalid input raises `ValueError`.
- The GIL is released while counting.
- Package renamed to `pysauce`.

## 0.3.0 (2026-06-11)

- New `count_npoint3_shape` and `compute_npcf3_shape`: triplets binned by size
  and shape.

## 0.2.0 (2026-05-30)

- `compute_npcf` takes `scale_method` ("scale-up" or "scale-down") and returns
  `zeta`, the corrected counts divided by the number of N-tuples.

## 0.1.1 (2026-05-28)

- `compute_xi_smu` returns the analytic random counts, the multipoles and the
  bin centres.

## 0.1.0 (2026-05-27)

- First release: pair counters in r, (r_p, pi) and (s, mu), the N-tuple
  counter, and the sparse-partition estimators.
