# Multidimensional Scaling (sklearn.manifold.MDS)

<!--
tier: 3-component
status: shipped-metric-smacof
baseline-commit: b8ff9e61
upstream: scikit-learn 1.5.2 behavior oracle, scikit-learn 1.9.0 public API surface
upstream-paths:
  - sklearn/manifold/_mds.py
ferrolearn-module: ferrolearn-decomp/src/mds.rs
parity-ops: MDS, smacof
crosslink-issue: 1451
-->

## Summary

`ferrolearn-decomp/src/mds.rs` now implements metric SMACOF for `MDS` and
exports the standalone public helper `ferrolearn_decomp::smacof`, matching the
exact sklearn public helper name. The implementation covers the metric path:
Guttman majorization, `n_init` restarts, fixed `init`, `max_iter`, `eps`,
`random_state`, raw stress, normalized Stress-1 reporting, precomputed
dissimilarities, and fitted accessors for the embedding, stress, dissimilarity
matrix, and iteration count.

The fixed-init path is the value-parity path. With `init=X0`, ferrolearn follows
the same deterministic metric SMACOF trajectory as sklearn 1.5.2
`smacof(init=X0, n_init=1)` and matches the live oracle element-wise to about
`1e-6`. The default random-init path remains a documented coordinate-parity
carve-out because sklearn uses numpy `RandomState.uniform` while ferrolearn uses
Rust `Xoshiro256PlusPlus`.

The retained `classical_mds`, `eigh_faer`, and `pairwise_sq_distances` helpers
are internal `pub(crate)` support for Isomap, whose embedding step is classical
MDS on geodesic distances. They are not the algorithm used by `MDS::fit`.

Current status: 7 shipped requirements (REQ-2 through REQ-6, REQ-8, REQ-9) and
4 not-started requirements (REQ-1 default numpy-RNG coordinate parity, REQ-7
non-metric MDS, REQ-10 PyO3 binding, REQ-11 ferray substrate).

## Requirements

| REQ | Status | Evidence |
|---|---|---|
| REQ-1: exact coordinate parity for the default random init | NOT-STARTED | RNG carve-out. sklearn initializes from numpy `RandomState.uniform`; ferrolearn initializes from `Xoshiro256PlusPlus`. Exact default coordinates differ even though the metric SMACOF loop is implemented. Fixed-init parity is tested under REQ-6. |
| REQ-2: distance preservation / stress descent | SHIPPED | `smacof_single` minimizes raw stress by Guttman majorization. The fixed-init full-rank fixture reconstructs pairwise distances closely; `test_mds_preserves_distances` guards this. |
| REQ-3: embedding shape and deterministic fixed-init behavior | SHIPPED | `FittedMDS::embedding()` has shape `(n_samples, n_components)`, or the supplied `init` column count. Fixed-init runs are bitwise deterministic in `green_fixed_init_determinism_identical_runs`. |
| REQ-4: normalized Stress-1 | SHIPPED | `MDS::with_normalized_stress(true)` and public `smacof(..., normalized_stress=true, ...)` report Kruskal Stress-1. `smacof_normalized_stress_is_kruskal1` guards the magnitude change. |
| REQ-5: scoped validation contracts | SHIPPED | `MDS::fit` rejects `n_components == 0`, too few samples, non-square precomputed matrices, non-finite inputs, and wrong-row-count `init`. Public `smacof` rejects non-square dissimilarities, fewer than two samples, non-finite inputs, `n_components == 0`, and wrong-row-count `init`. ferrolearn remains stricter than sklearn on `MDS::fit` when `n_components > n_samples`. |
| REQ-6: metric SMACOF algorithm, `n_init`, `random_state`, fixed `init` | SHIPPED | `smacof_impl` runs one or more `smacof_single` restarts and keeps the minimum-stress result; fixed `init` forces one run. `green_smacof_fixed_init_*_parity`, `smacof_fixed_init_*_parity`, and `public_smacof_fixed_init_precomputed_matches_sklearn` match sklearn 1.5.2 fixed-init oracles. |
| REQ-7: `metric=False` non-metric MDS | NOT-STARTED | sklearn's non-metric path performs an isotonic-regression disparity update each iteration. ferrolearn currently implements only the metric path. |
| REQ-8: raw sklearn `stress_`, `max_iter`, and `eps` | SHIPPED | Default `stress()` is raw SSR/2; `with_max_iter` and `with_eps` govern the convergence loop. `divergence_mds_stress_2404` and MDS in-module parity guards pin the raw stress value. |
| REQ-9: precomputed SMACOF plus fitted attrs | SHIPPED | `MDS::fit` stores `embedding`, `stress`, `dissimilarity_matrix`, and `n_iter`; Euclidean mode stores computed pairwise distances and precomputed mode stores the supplied matrix. `smacof_dissimilarity_matrix_is_euclidean` and precomputed parity guards cover this. |
| REQ-10: PyO3 binding | NOT-STARTED | `ferrolearn-python` does not register MDS or expose `fit_transform`, `embedding_`, `stress_`, `dissimilarity_matrix_`, or `n_iter_`. |
| REQ-11: ferray substrate | NOT-STARTED | The implementation still uses `ndarray::Array2` and faer directly rather than ferray array/linalg primitives. |

## Public Surface

- `ferrolearn_decomp::MDS`
- `ferrolearn_decomp::FittedMDS`
- `ferrolearn_decomp::Dissimilarity`
- `ferrolearn_decomp::smacof`

`MDS` exposes the Rust builder surface `with_dissimilarity`, `with_init`,
`with_n_init`, `with_max_iter`, `with_eps`, `with_normalized_stress`, and
`with_random_state`. It implements `Fit<Array2<f64>, ()>`. Rust callers retrieve
the fitted embedding through `FittedMDS::embedding()` rather than a separate
`fit_transform` method.

`smacof` accepts a precomputed dissimilarity matrix and returns
`(embedding, stress, n_iter)`. This corresponds to sklearn's public helper for
the metric path, with `metric=True` and `return_n_iter=True` semantics.

## Oracle Evidence

The committed fixed-init expected values come from the installed local sklearn
1.5.2 oracle, not from ferrolearn output.

```bash
python3 - <<'PY'
import numpy as np
from sklearn.manifold import MDS, smacof

X0 = np.array([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]])
D = np.array([
    [0.0, 2.0, 5.0, 9.0],
    [2.0, 0.0, 3.0, 4.0],
    [5.0, 3.0, 0.0, 6.0],
    [9.0, 4.0, 6.0, 0.0],
])
print(smacof(D, metric=True, init=X0, n_init=1,
             normalized_stress=False, max_iter=300, eps=1e-3,
             return_n_iter=True))

X = np.array([[0., 0.], [3., 0.], [0., 4.], [3., 4.]])
m = MDS(n_components=2, dissimilarity="euclidean", n_init=1,
        max_iter=300, eps=1e-3, normalized_stress=False)
print(m.fit_transform(X, init=X0))
print(m.stress_, m.n_iter_)
PY
```

Pinned values:

- precomputed helper stress: `3.148219331054871`
- precomputed helper `n_iter`: `13`
- precomputed helper embedding:
  `[[-3.333717200034, -1.658330631573], [-0.431085112947, -0.700165295708],
  [-0.786750476780, 2.465105803376], [4.551552789761, -0.106609876095]]`
- Euclidean MDS stress: `0.0013111846996572488`
- Euclidean MDS `n_iter`: `13`
- Euclidean MDS embedding:
  `[[-2.164424557023, -1.234049962647], [0.57663887645, -2.435876213413],
  [-0.587315085045, 2.433308813391], [2.175100765618, 1.236617362669]]`

## Verification

Targeted commands for this unit:

```bash
cargo fmt --all --check
cargo test -p ferrolearn-decomp mds
cargo test -p ferrolearn-decomp --test divergence_mds
cargo test -p ferrolearn-decomp --test divergence_smacof
cargo test -p ferrolearn-decomp --test divergence_mds_stress_2404
cargo test -p ferrolearn-decomp --test divergence_mds_grid_2406
cargo test -p ferrolearn-decomp --test api_proof api_proof_mds
cargo test -p ferrolearn-decomp --test conformance_surface_coverage
```

Broader crate verification:

```bash
cargo test -p ferrolearn-decomp
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
```

## Remaining Blockers

- `#1452`: exact default random-init coordinate parity. This requires numpy RNG
  parity or a deliberate compatibility shim for sklearn's default coordinate
  stream.
- `#1454`: non-metric MDS (`metric=False`) with isotonic-regression disparities.
- `#1457`: PyO3/Python binding for `MDS` and its fitted attributes.
- `#1458`: migrate the implementation substrate from direct `ndarray`/faer to
  ferray primitives.
