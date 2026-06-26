# ClassicalMDS (sklearn.manifold.ClassicalMDS)

<!--
tier: 3-component
status: shipped
upstream: scikit-learn 1.9.0 public API surface and source contract
upstream-paths:
  - sklearn/manifold/_classical_mds.py
ferrolearn-module: ferrolearn-decomp/src/mds.rs
parity-ops: ClassicalMDS
-->

## Summary

`ferrolearn_decomp::ClassicalMDS` closes the exact public-name gap for
`sklearn.manifold.ClassicalMDS`. It implements the source contract from
`_classical_mds.py`: compute pairwise distances for Euclidean input or validate
a precomputed symmetric dissimilarity matrix, double-centre squared distances,
eigendecompose the centred matrix, reverse eigenpairs into descending order, and
apply sklearn's deterministic `svd_flip(U, None)` sign convention before scaling
the embedding by `sqrt(eigenvalues_)`.

The Rust surface follows ferrolearn's fitted-state style:

- `ClassicalMDS::new(n_components)`
- `ClassicalMDS::with_metric(Dissimilarity::{Euclidean, Precomputed})`
- `ClassicalMDS::fit_transform(&Array2<f64>)`
- `FittedClassicalMDS::{embedding, dissimilarity_matrix, eigenvalues}`

## Evidence

The parity guard is `ferrolearn-decomp/tests/divergence_classical_mds.rs`. The
local installed sklearn runtime is 1.5.2 and does not expose `ClassicalMDS`, so
the expected values are generated from the 1.9 source algorithm using numpy and
scipy: pairwise distances, double centering, `scipy.linalg.eigh`, descending
eigen order, and sklearn's `svd_flip` rule.

Targeted verification:

```bash
cargo test -p ferrolearn-decomp --test divergence_classical_mds
cargo test -p ferrolearn-decomp --test api_proof api_proof_mds
cargo test -p ferrolearn-decomp --test conformance_surface_coverage
```

## Remaining Gaps

- Generic sklearn `metric` strings/callables and `metric_params` are scoped down
  to Euclidean and precomputed modes through the existing `Dissimilarity` enum.
- Python `BaseEstimator` behavior, feature-name attributes, and PyO3 exposure
  are still part of the broader Python protocol gap documented in
  `GAP-REPORT.md`.
