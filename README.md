# SaUCE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**S**parse **a**nd **U**nbiased **C**lustering **E**stimator: pair and N-point
counts for catalogues made of k partitions of a periodic box, of which only m
are available, corrected for the missing partitions. The counting is written
in Rust and used from the Python package `pysauce`.

## Method

Each partition is an independent sample of the whole box at 1/k of the total
density, for example one of k runs of a semi-analytic model on the same
N-body simulation. An N-tuple whose members lie in s distinct partitions is
observed only if all s are among the m selected, which happens with
probability

    P_s = m (m − 1) ⋯ (m − s + 1) / [k (k − 1) ⋯ (k − s + 1)].

The counters split their counts by s (for pairs, "auto" and "cross"), and the
estimators weight each class by 1 / P_s. With m = k the result is the count of
the full box; with m < k it is an unbiased estimate of it. The estimators
require N ≤ m ≤ k, since with m < N some tuples could never be observed.

The weights can instead be multiplied by (m / k)^N ("scale-down"), giving the
counts at the observed density, to compare with random counts for the
observed number of points.

## Installation

```bash
pip install pysauce
```

This needs Python 3.9 or newer. To build from source instead, install Rust
1.88 or newer (from [rustup](https://rustup.rs)) and run

```bash
pip install "git+https://github.com/OscarHickman/SaUCE.git"
```

For development, clone the repository and run
`pip install maturin && maturin develop --release` in a virtual environment.

## Usage

```python
import numpy as np
from pysauce import compute_xi

rng = np.random.default_rng(0)
k, box_size = 8, 500.0
coords = rng.uniform(0, box_size, (200_000, 3))
partition_ids = rng.integers(0, k, 200_000)

selected = partition_ids < 3  # m = 3 of the k partitions
result = compute_xi(
    coords[selected], partition_ids[selected],
    r_bins=np.linspace(5.0, 50.0, 10), box_size=box_size,
    n_partitions=k, n_partitions_selected=3,
)
result["xi"]  # consistent with 0 for this unclustered field
```

| Function | Returns |
|---|---|
| `compute_xi`, `compute_2pcf`, `compute_xi_smu` | corrected ξ(r); ξ(r_p, π) and w_p(r_p); ξ(s, μ) and its multipoles |
| `compute_npcf`, `compute_npcf3_shape`, `compute_npcf3_smu` | corrected N-point counts by size, and triplets by shape or orientation |
| `count_pairs_1d`, `count_pairs_2d`, `count_pairs_smu` | raw auto and cross pair counts |
| `count_pairs_1d_partition_matrix` | pair counts for every pair of partitions |
| `count_npoint`, `count_npoint3_shape`, `count_npoint3_smu` | raw N-tuple counts split by s |
| `partition_weights` | the weights |
| `analytic_rr_1d`, `analytic_rr`, `analytic_rr_smu` | expected random pair counts |

Conventions: positions lie in `[0, box_size)³` and separations use the nearest
periodic image; the line of sight is the z axis (π = |Δz|, μ = |Δz| / s); bins
are half-open, `[lo, hi)`; an N-tuple is binned by its largest pairwise
separation. The `count_*` functions need `float64` positions and `int32`
labels; the `compute_*` functions convert their input. See the docstrings for
details.

## Examples

The notebooks in [`examples/`][examples] run in seconds on a mock catalogue
they build themselves (`pip install ".[examples]"` adds Matplotlib and
Jupyter):

* [`01_xi_r.ipynb`][ex1]: ξ(r) from m of k partitions against the full box;
* [`02_redshift_space.ipynb`][ex2]: ξ(s, μ), its multipoles and w_p(r_p);
* [`03_npoint.ipynb`][ex3]: triplet counts by size, shape and orientation.

## How it works

Particles are sorted into columns that span the box along z and are about
half the search radius wide. Each particle's partners are searched for in
nearby columns, only within the range of z they can reach, and candidates are
tested and binned in SIMD blocks. The result is exactly that of comparing
every pair. On x86-64 the inner loops are also compiled for AVX2, chosen at
run time when the CPU supports it.

The counters use every core; set `RAYON_NUM_THREADS` to use fewer. The counts
do not depend on the number of threads.

## Testing

```bash
pip install -e ".[test]"
pytest
cargo test
```

The tests compare every counter with brute-force references
([`tests/reference.py`][reference]) and with
[Corrfunc](https://github.com/manodeep/Corrfunc), and check that the
estimators return the full-box counts for m = k and are unbiased for m < k.
`cargo test` links against libpython: if it cannot find it, set `PYO3_PYTHON`
to your Python and add its `lib` directory to `LD_LIBRARY_PATH`.

`pytest benchmarks` (with `pip install -e ".[benchmark]"`) times every counter
on a million points.

## Citation

If you use SaUCE, please cite the paper for the statistic you measure:

* two-point functions: Hickman, O. J. et al. (2026), *Less is More: Unbiased
  Galaxy Clustering from Fewer Merger Trees*, arXiv preprint;
* N-point functions: Hickman, O. J. et al. (2026), *SaUCE for the N-point
  Correlation Function: Unbiased Higher-Order Galaxy Clustering from Fewer
  Merger Trees*, in prep.

The software itself can be cited with [`CITATION.cff`][citation].

## License

MIT; see [LICENSE][license].

[examples]: https://github.com/OscarHickman/SaUCE/tree/main/examples
[ex1]: https://github.com/OscarHickman/SaUCE/blob/main/examples/01_xi_r.ipynb
[ex2]: https://github.com/OscarHickman/SaUCE/blob/main/examples/02_redshift_space.ipynb
[ex3]: https://github.com/OscarHickman/SaUCE/blob/main/examples/03_npoint.ipynb
[reference]: https://github.com/OscarHickman/SaUCE/blob/main/tests/reference.py
[citation]: https://github.com/OscarHickman/SaUCE/blob/main/CITATION.cff
[license]: https://github.com/OscarHickman/SaUCE/blob/main/LICENSE
