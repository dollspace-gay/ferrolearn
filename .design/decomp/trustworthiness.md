# trustworthiness (sklearn.manifold.trustworthiness)

<!--
tier: 3-component
status: shipped-partial
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/manifold/_t_sne.py  # trustworthiness (:456-558)
ferrolearn-module: ferrolearn-decomp/src/trustworthiness.rs
parity-ops: trustworthiness
-->

## Summary

`ferrolearn-decomp/src/trustworthiness.rs` implements the dense Euclidean Rust
analogue of `sklearn.manifold.trustworthiness`: rank original-space neighbors,
take each embedded-space k-nearest-neighbor set, sum the positive rank penalties
`rank(i,j)-k`, and return
`1 - 2 * penalty / (n * k * (2n - 3k - 1))` (`_t_sne.py:456-558`).

The public surface is `pub fn trustworthiness(&Array2<f64>, &Array2<f64>, usize)
-> Result<f64, FerroError>`, re-exported from `ferrolearn_decomp`. It is narrower
than sklearn: dense Euclidean only; no sparse input, `metric="precomputed"`,
cosine/callable metrics, or Python target validation ABI.

## Requirements

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (rank-penalty formula) | SHIPPED | `trustworthiness.rs` builds the inverted original-space rank matrix, takes the first `n_neighbors` embedded neighbors, sums positive `rank-k` penalties, and applies sklearn's normalizer (`_t_sne.py:545-558`). Verification: `tests/divergence_trustworthiness.rs` pins identity/affine `1.0`, sklearn's five-point shuffled fixture `0.2`, and a no-tie seven-point fixture for k=1/2/3 to the live sklearn 1.5.2 oracle. |
| REQ-2 (validation) | SHIPPED | rejects `n_neighbors < 1`, `n_neighbors >= n_samples/2`, sample-count mismatch, and non-finite input. sklearn's `n_neighbors >= n/2` error is explicit at `_t_sne.py:527-532`; other validation maps to ferrolearn's `FerroError` family. Verification: `trustworthiness_validation_errors`. |
| REQ-3 (surface/evidence) | SHIPPED | crate-root re-export in `lib.rs`; `tests/api_proof.rs::api_proof_trustworthiness`; surface inventory includes `ferrolearn_decomp::trustworthiness`. |
| REQ-X (broader metric surface) | NOT-STARTED | sklearn accepts sparse input, `metric="precomputed"`, cosine and callable metrics through `pairwise_distances`; ferrolearn currently supports dense Euclidean only. |
