# Covariance Estimation

<!--
tier: 3-component
status: draft
baseline-commit: fa5b01d1
upstream-paths:
  - sklearn/covariance/_empirical_covariance.py
  - sklearn/covariance/_shrunk_covariance.py
  - sklearn/covariance/_robust_covariance.py
  - sklearn/covariance/_elliptic_envelope.py
-->

## Summary

`ferrolearn-covariance/src/covariance.rs` mirrors the six estimators of
`sklearn.covariance`: `EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf`,
`OAS` (`sklearn/covariance/_empirical_covariance.py`,
`sklearn/covariance/_shrunk_covariance.py`), `MinCovDet` (FastMCD,
`sklearn/covariance/_robust_covariance.py`) and `EllipticEnvelope`
(`sklearn/covariance/_elliptic_envelope.py`). All estimators share a
`FittedCovariance` return type holding `covariance_`, `location_`, `precision_`
and follow the ferrolearn `Fit` trait pattern.

The deterministic estimators split sharply by value parity against the live
sklearn 1.5.2 oracle: `EmpiricalCovariance`, `ShrunkCovariance`, and
`LedoitWolf` are element-wise exact (~1e-13), so their value REQs are SHIPPED
and oracle-pinnable. `OAS` uses a *superseded* shrinkage formula and diverges
(ferrolearn `shrinkage_ = 0.5387` vs sklearn `0.6706` on the probe set), so its
value REQ is NOT-STARTED. `MinCovDet`/`EllipticEnvelope` are RNG-coupled
(FastMCD draws random h-subsets with a Rust `Xoshiro256PlusPlus`, not numpy
`RandomState`), so exact-value parity is an R-DEFER-3 carve-out (blocker, no
failing test); only structural REQs are testable, and several of those also
diverge (empirical-quantile `offset_` vs chi-squared threshold; squared vs
non-squared `mahalanobis`). Two `clippy::collapsible_if` lints block the crate
gauntlet.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/covariance/_empirical_covariance.py` — `empirical_covariance`
  (`:67`, `covariance = np.dot(X.T, X) / X.shape[0]`, ddof=0 / MLE),
  `log_likelihood` (`:33`), `class EmpiricalCovariance` (`:116`),
  `get_precision` (`:216`), `score` (`:258`), `error_norm` (`:289`),
  `mahalanobis` (`:340`, returns **squared** distances `:353`).
- `sklearn/covariance/_shrunk_covariance.py` — `shrunk_covariance` (`:111`,
  `(1-s)*emp_cov + s*mu*I`, `:153-156`), `class ShrunkCovariance` (`:161`,
  default `shrinkage=0.1`), `ledoit_wolf_shrinkage`/`ledoit_wolf` and
  `class LedoitWolf` (`:469`, `block_size=1000`), `oas` (`:618`) and
  `class OAS` (`:686`). The 1.5.2 OAS formula is `alpha = mean(emp_cov**2)`,
  `mu = trace(emp_cov)/p`, `num = alpha + mu**2`,
  `den = (n_samples + 1) * (alpha - mu**2/p)`,
  `shrinkage = 1.0 if den==0 else min(num/den, 1.0)` (`:79-87`).
- `sklearn/covariance/_robust_covariance.py` — `class MinCovDet` (`:580`) +
  `fast_mcd`/`_c_step`/`select_candidates`; support size
  `n_support = (n_samples + n_features + 1) // 2` default; consistency
  correction via `correction = median(dist) / chi2(p).isf(0.5)` and reweighting
  at the 0.975 chi-squared quantile.
- `sklearn/covariance/_elliptic_envelope.py` — `class EllipticEnvelope`
  (`:16`), constructor `contamination=0.1` in `Interval(Real, 0, 0.5,
  closed="right")` (`:147`), `offset_ = np.percentile(-self.dist_,
  100.0*contamination)` (`:185`), `decision_function = score_samples - offset_`
  (`:206`), `predict` returns `+1` inlier / `-1` outlier with `values >= 0`
  the inlier branch (`:224-243`).

The installed `sklearn` 1.5.2 package is the live oracle.

## Requirements

Grouped per estimator, then cross-cutting. Each REQ is tagged with its
verification class: **oracle-pinnable** (deterministic value parity against the
live oracle), **structural** (shape/sign/range invariant, RNG-independent),
**RNG carve-out** (exact values un-bit-matchable, R-DEFER-3), or
**code-quality**.

### EmpiricalCovariance
- REQ-1 (MLE covariance, oracle-pinnable): `covariance_ = (X-mean)^T(X-mean)/n`
  (ddof=0, biased), `location_` = column mean or 0 if `assume_centered`,
  `precision_` = inverse. Mirrors `empirical_covariance`
  (`_empirical_covariance.py:67`).

### ShrunkCovariance
- REQ-2 (fixed shrinkage, oracle-pinnable): `(1-s)*emp + s*(trace(emp)/p)*I`,
  default `shrinkage=0.1`. Mirrors `shrunk_covariance`
  (`_shrunk_covariance.py:153-156`).

### LedoitWolf
- REQ-3 (Ledoit-Wolf shrinkage + covariance, oracle-pinnable): data-driven
  `shrinkage_` and `covariance_` match `LedoitWolf().fit(X)` element-wise.

### OAS
- REQ-4 (OAS shrinkage + covariance, oracle-pinnable): `shrinkage_`/`covariance_`
  must match the 1.5.2 OAS formula (`_shrunk_covariance.py:79-87`).

### MinCovDet
- REQ-5 (FastMCD exact attributes, RNG carve-out): `support_`,
  `raw_location_`/`raw_covariance_`, `location_`/`covariance_` exact values.
- REQ-6 (support size, structural): support cardinality equals
  `h = (n + p + 1)//2` (default) or `ceil(support_fraction*n)`.
- REQ-7 (consistency correction + reweighting, structural): the raw MCD
  covariance is scaled by the empirical consistency factor and a reweighting
  step at the 0.975 chi-squared quantile is applied; output covariance is SPD.
- REQ-8 (raw_* attribute exposure, structural): `raw_location_`,
  `raw_covariance_`, `dist_`, `correction_` are exposed (sklearn names).

### EllipticEnvelope
- REQ-9 (offset_ definition, structural / oracle-pinnable-given-MCD):
  `offset_ = percentile(-dist_, 100*contamination)` — an empirical quantile of
  the MCD Mahalanobis distances (`_elliptic_envelope.py:185`).
- REQ-10 (predict / decision_function sign, structural): `predict` returns `+1`
  inlier / `-1` outlier; `decision_function = score_samples - offset_` with the
  inlier set `values >= 0` (`:206,:241`).
- REQ-11 (contamination domain, structural): `contamination` constraint
  `Interval(Real, 0, 0.5, closed="right")` — `0.5` is ACCEPTED (`:147`).
- REQ-12 (EllipticEnvelope exact labels, RNG carve-out): exact `offset_`,
  `dist_`, and label vector depend on the underlying FastMCD RNG.

### Cross-cutting
- REQ-13 (`mahalanobis` returns squared distances, structural): sklearn returns
  the **squared** Mahalanobis distance (`_empirical_covariance.py:340,353`).
- REQ-14 (`score`/`log_likelihood`, oracle-pinnable): a `score` method
  returning the Gaussian average log-likelihood (`:258`,
  `log_likelihood :33`) on the estimator types.
- REQ-15 (`error_norm`, oracle-pinnable): Frobenius/spectral error norm between
  two covariances (`:289`).
- REQ-16 (`get_precision`, structural): a `get_precision` accessor /
  `store_precision` flag (`:216`).
- REQ-17 (code-quality — collapsible_if lints): the two
  `clippy::collapsible_if` lints in the MinCovDet section block the crate
  gauntlet (`-D warnings`).
- REQ-18 (R-SUBSTRATE — ferray): owned computation must run on `ferray-core` /
  `ferray::linalg` / `ferray::random`, not `ndarray` + hand-rolled
  `spd_inverse`/`log_det_spd` + `rand_xoshiro`.
- REQ-19 (consumer): the estimator types are re-exported and reachable by a
  non-test production consumer.

## Acceptance criteria

- AC-1 (REQ-1): `covariance_`/`location_`/`precision_` match
  `EmpiricalCovariance(assume_centered=...).fit(X)` element-wise within 1e-10
  on the probe set (oracle-pinnable).
- AC-2 (REQ-2): `covariance_` matches `ShrunkCovariance(shrinkage=0.1).fit(X)`
  element-wise within 1e-10 (oracle-pinnable).
- AC-3 (REQ-3): `shrinkage_` and `covariance_` match `LedoitWolf().fit(X)`
  within 1e-10 (oracle-pinnable).
- AC-4 (REQ-4): `shrinkage_` and `covariance_` match `OAS().fit(X)` within
  1e-10 (oracle-pinnable — currently FAILS, see status).
- AC-5 (REQ-6): `support().iter().filter(|&&b| b).count()` ≥ `h` (the support
  cardinality bound).
- AC-6 (REQ-7): the fitted `covariance_` is SPD and the raw covariance is
  scaled (output ≠ raw subset covariance).
- AC-7 (REQ-9): `offset_` equals `np.percentile(-dist_, 100*contamination)` of
  the model's own MCD distances (given the same support set).
- AC-8 (REQ-10): `predict` ∈ {+1, -1}; the inlier set is exactly
  `decision_function(X) >= 0`.
- AC-9 (REQ-11): `EllipticEnvelope(contamination=0.5)` fits successfully (no
  error), matching sklearn's closed-right interval.
- AC-10 (REQ-13): `mahalanobis(X)` returns squared distances matching
  `EmpiricalCovariance.fit(X).mahalanobis(X)` element-wise within 1e-10.
- AC-11 (REQ-14): `score(X)` matches `EmpiricalCovariance().fit(X).score(X)`
  within 1e-10 (oracle: -2.8378770664093453 on the symmetric probe).
- AC-12 (REQ-19): the estimator types are constructed by a non-test consumer.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (empirical MLE) | SHIPPED | `pub fn fit for EmpiricalCovariance in covariance.rs` calls `empirical_cov` (`cov = xt.dot(&x_c); cov.mapv_inplace(\|v\| v / n_f)`, ddof=0) and `spd_inverse`; `location` is `col_mean` or `Array1::zeros` when `assume_centered`. Mirrors `_empirical_covariance.py:67` (`covariance = np.dot(X.T, X) / X.shape[0]`). Live oracle `X=[[0,0],[2,0],[0,2],[2,2]]`: sklearn `covariance_=[[1,0],[0,1]]`, `location_=[1,1]`, `precision_=[[1,0],[0,1]]`; ferrolearn matches. Asymmetric probe `covariance_[0][0]=3.8888888888888893` matches sklearn `3.8888888888888893`. Non-test consumer: `pub fn empirical_covariance in helpers.rs` (`EmpiricalCovariance::<F>::new().assume_centered(...).fit(x, &())`), itself consumed by `graphical_lasso.rs` (`let emp_cov = empirical_covariance(x, self.assume_centered)?`). Verification: `cargo test -p ferrolearn-covariance --lib` 53 passed. Oracle-pinnable. |
| REQ-2 (shrunk fixed) | SHIPPED | `pub fn fit for ShrunkCovariance in covariance.rs` calls `shrink_covariance(&cov_emp, self.shrinkage, p)` = `(1-s)*cov + s*(trace(cov)/p)*I`. Mirrors `_shrunk_covariance.py:153-156`. Live oracle (`shrinkage=0.1`, asymmetric probe): sklearn `covariance_[0]=[4.148148148148149, 4.4, 3.45]`; ferrolearn `[4.148148148148149, 4.3999999999999995, 3.45]` — match to ~1e-13. Non-test consumer: re-exported in `lib.rs` (`pub use covariance::{... ShrunkCovariance}`); the function-form `pub fn shrunk_covariance in helpers.rs` is the parity surface (re-exported, consumed by tests + external users). Verification: tests green. Oracle-pinnable. NOTE: ferrolearn's default constructor takes shrinkage as an explicit arg (`ShrunkCovariance::new(0.1)`) rather than defaulting it — sklearn's class default `shrinkage=0.1` is matched only when the caller passes 0.1; see Architecture. |
| REQ-3 (Ledoit-Wolf) | SHIPPED | `pub fn fit for LedoitWolf in covariance.rs` computes `S = X_c^T X_c / n`, `mu = trace(S)/p`, `delta = sum((s_ij - mu*kron_ij)^2)/p^2`, `beta = sum_k ||x_k x_k^T - S||_F^2 / (n^2 p^2)`, `shrinkage = clip(beta/delta, 0, 1)`, then `shrink_covariance(&s, shrinkage, p)`. Live oracle (asymmetric probe): sklearn `shrinkage_=0.4087180724344861`, `covariance_[0][0]=4.94852833594126`; ferrolearn `shrinkage=0.4087180724344862`, `covariance[0][0]=4.948528335941261` — match to ~1e-13. Non-test consumer: `pub fn ledoit_wolf in helpers.rs` (`LedoitWolf::<F>::new().assume_centered(...).fit(x,&())` → `(cov, shrinkage)`), re-exported in `lib.rs`. Verification: tests green; element-wise oracle parity. Oracle-pinnable. NOTE: `block_size` (sklearn default 1000) is not a ferrolearn param — it affects only chunking, not the result, so parity holds. |
| REQ-4 (OAS) | SHIPPED | `Fit::fit for OAS` uses the sklearn 1.5.2 formula (`_shrunk_covariance.py:79-87`, fixed #1702): `mu = trace(S)/p`, `alpha = mean(emp_cov²)` (`trace_sq(&s)/p²`), `num = alpha + mu²`, `den = (n+1)·(alpha − mu²/p)`, `shrinkage = min(num/den, 1)`, `cov = (1−s)·S + s·mu·I`. Verified vs the live sklearn 1.5.2 oracle (this row's prior claim of the SUPERSEDED formula was STALE): test `divergence_oas_formula` pins the asymmetric probe to `shrinkage_=0.6705854984555868`, `covariance_[0][0]=5.627443884884855` (full 3×3, ≤1e-9). |
| REQ-5 (MCD exact attrs) | NOT-STARTED | open prereq blocker (tracking #1701; R-DEFER-3 RNG carve-out — NO failing test). `pub fn fit for MinCovDet in covariance.rs` draws subsets via `Xoshiro256PlusPlus::seed_from_u64` + `Uniform`, which cannot bit-match numpy `RandomState` in sklearn's `fast_mcd` (`_robust_covariance.py`). `support_`/`location_`/`covariance_` exact values are un-bit-matchable; per R-DEFER-3 this is a documented carve-out, not a pinned failing test. |
| REQ-6 (MCD support size) | SHIPPED | `pub fn fit for MinCovDet in covariance.rs` computes `h = (n + p + 1).div_ceil(2).max(p+1).min(n)` (default) or `ceil(frac*n).max(p+1).min(n)`, matching sklearn's `n_support = (n_samples + n_features + 1) // 2`. The C-step selects the `h` closest by Mahalanobis distance (`indices_dists ... take(h)`). Structural test `test_mincovdet_support_fraction` asserts `n_support >= 4` for `support_fraction(0.7)` on 7 samples (`ceil(0.7*7)=5`). Non-test consumer: `pub fn fast_mcd in helpers.rs`. Verification: tests green. Structural. |
| REQ-7 (consistency + reweight) | SHIPPED | `pub fn fit for MinCovDet in covariance.rs` applies the consistency correction (`correction = median(mahal²) / chi2_quantile_approx(p, 0.5)`, `cov.mapv_inplace(\|v\| v * correction)`) then reweighting (`thresh_sq = chi2_quantile_approx(p, 0.975)`; recompute location/cov on the `dist² < thresh_sq` mask), mirroring sklearn's correction + reweighting (`_robust_covariance.py`). Output is forced SPD via `invert_with_shrinkage`. Structural test `test_mincovdet_symmetric` asserts symmetry; `test_mincovdet_with_outlier` asserts the outlier has max distance. Consumer: `fast_mcd in helpers.rs`. Structural — NOTE: the chi-squared quantile is the Wilson-Hilferty `chi2_quantile_approx`, not an exact `scipy.stats.chi2.isf`, so the *magnitude* of the correction diverges slightly from sklearn even before the RNG carve-out (sub-finding under REQ-5; the correction is structurally present and applied). |
| REQ-8 (raw_* attributes) | NOT-STARTED | open prereq blocker (tracking #1701). `FittedMinCovDet` exposes `covariance()`, `location()`, `precision()`, `support()` but NOT `raw_location_`, `raw_covariance_`, `dist_`, or `correction_` (sklearn attributes, `_robust_covariance.py` `MinCovDet`). The raw (pre-correction) estimates are computed internally (`best_location`/`best_cov`) but discarded — not surfaced as fitted attributes. |
| REQ-9 (offset_ definition) | NOT-STARTED | open prereq blocker (tracking #1701). `pub fn fit for EllipticEnvelope in covariance.rs` sets `threshold_ = chi2_quantile_approx(p, 1 - contamination)` — a chi-squared quantile of the *theoretical* distribution. sklearn instead sets `offset_ = np.percentile(-self.dist_, 100*contamination)` (`_elliptic_envelope.py:185`) — an EMPIRICAL quantile of the model's own training Mahalanobis distances. Different quantity (theoretical chi² vs empirical percentile), and ferrolearn exposes `threshold()` not `offset_`. Oracle-pinnable given a fixed MCD support set. |
| REQ-10 (predict/decision sign) | SHIPPED | `impl Predict for FittedEllipticEnvelope in covariance.rs` returns `if d*d > self.threshold_ { -1 } else { 1 }` (∈ {+1,-1}); `pub fn decision_function in covariance.rs` returns `threshold_ - d*d` (positive = inlier). sklearn: `predict` returns `+1`/`-1` with inliers at `values >= 0` (`_elliptic_envelope.py:241`); `decision_function = score_samples - offset_` (`:206`). The {+1,-1} convention and "inlier ⇔ decision ≥ 0" relationship match. Test `test_elliptic_envelope_decision_function` asserts `scores[i] > 0 ⟹ labels[i] == 1`. Consumer: re-exported in `lib.rs`. Structural. NOTE: ferrolearn's tie at `d*d == threshold` maps to inlier (`else` branch) and `decision == 0` maps to inlier — consistent with sklearn's `>= 0` inlier rule. The *value* of the threshold diverges (REQ-9), but the sign convention is correct. |
| REQ-11 (contamination domain) | SHIPPED | `EllipticEnvelope::fit` accepts the interval `(0, 0.5]` (rejects `≤0` and `>0.5`), matching sklearn `Interval(Real, 0, 0.5, closed="right")` (`_elliptic_envelope.py:147`, fixed #1704); `0.5` is accepted, `0.6` rejected. Verified vs live sklearn 1.5.2 (`contamination=0.5`→OK, `0.6`→error; ferrolearn matches — this row's prior over-rejection claim was STALE). Test `divergence_contamination_right_endpoint`. |
| REQ-12 (EE exact labels) | NOT-STARTED | open prereq blocker (tracking #1701; R-DEFER-3 RNG carve-out — NO failing test). `EllipticEnvelope` wraps `MinCovDet`, so `dist_`, `offset_`, and the label vector inherit the FastMCD RNG (`Xoshiro256PlusPlus` ≠ numpy `RandomState`). Exact-value parity is a documented carve-out, not a pinned failing test. |
| REQ-13 (mahalanobis squared) | SHIPPED | The public `FittedCovariance::mahalanobis` now returns the SQUARED distance: `Ok(mahalanobis_distances(...).mapv(|d| d*d))`, matching sklearn `EmpiricalCovariance.mahalanobis` ("Squared Mahalanobis distances", `_empirical_covariance.py:340`). The private `mahalanobis_distances` helper and its four INTERNAL MCD/EllipticEnvelope fitting call sites (non-squared semantics) are unchanged; squaring once at the public boundary fixes all delegating public APIs (Shrunk/LedoitWolf/OAS/MinCovDet/EllipticEnvelope all route to `FittedCovariance::mahalanobis`). EllipticEnvelope `decision_function`/`predict` updated to consume the now-squared value (previously re-squared). Verification (live sklearn 1.5.2, R-CHAR-3, `X=[[0,0],[2,0],[0,2],[2,2]]`): `mahalanobis = [2,2,2,2]` (was `sqrt(2)≈1.414`). Tests `mahalanobis_returns_squared_matches_sklearn` + un-ignored `divergence_mahalanobis_squared` (tol 1e-7 absorbs a ~2e-8 residual from the separately-tracked precision-reg #1705). |
| REQ-14 (score/log_likelihood) | SHIPPED | `FittedCovariance::score(&self, x) -> Result<F, FerroError>` centers `x` by the TRAINING `location_`, forms the assume-centered empirical covariance `test_cov = Xcᵀ·Xc / n`, and returns `helpers::log_likelihood(&test_cov, &self.precision_)`, mirroring sklearn `EmpiricalCovariance.score` (`_empirical_covariance.py:258`). Delegating `score` added to `FittedLedoitWolf`/`FittedOAS`/`FittedMinCovDet` (→ `self.inner.score`) and `FittedEllipticEnvelope` (→ `self.mcd.score`); `ShrunkCovariance` returns a bare `FittedCovariance` so it has `score` directly. Shape-guards `n_features` (`ShapeMismatch`). Verification (live sklearn 1.5.2, R-CHAR-3): train `X=[[1,2],[2,1],[3,4],[4,3],[5,5],[2,3]]` → `score(X)=-2.9419860169581122`; `score([[1,1],[2,2],[3,1]])=-3.5585273703415714`. Tests `score_on_training_data_matches_sklearn`, `score_on_test_data_matches_sklearn`, `score_shape_mismatch_errors`. |
| REQ-15a (error_norm Frobenius) | SHIPPED | `FittedCovariance::error_norm(&self, comp_cov, scaling: bool, squared: bool) -> Result<F, FerroError>` computes the FROBENIUS error (sklearn default norm, `_empirical_covariance.py:289`): `error = comp_cov − covariance_`, `squared_norm = Σ error²`; if `scaling` divide by `n_features` (`error.shape[0]`); return `squared_norm` or `sqrt`. Shape-guards `(p,p)` (`ShapeMismatch`). Delegating `error_norm` on `FittedLedoitWolf`/`FittedOAS`/`FittedMinCovDet`/`FittedEllipticEnvelope`; `ShrunkCovariance` returns bare `FittedCovariance`. Verification (live sklearn 1.5.2, R-CHAR-3, `cov_=[[1.80555556,1.33333333],[1.33333333,1.66666667]]`, `comp=[[2,0.5],[0.5,2]]`): `(scaling=T,squared=T)=0.7689043209876543`; `(F,T)=1.5378086419753085`; `(T,F)=0.8768718954258109`. Tests `error_norm_frobenius_default_matches_sklearn`, `error_norm_unscaled_matches_sklearn`, `error_norm_not_squared_matches_sklearn`, `error_norm_shape_mismatch_errors`. |
| REQ-15b (error_norm spectral) | NOT-STARTED | open prereq blocker (tracking #1701). `norm='spectral'` = `max(svdvals(errorᵀ·error))` (`_empirical_covariance.py`) requires an eigenvalue/SVD routine the crate lacks (only `spd_inverse`/`cholesky`/`log_det_spd` exist). Needs the `ferray::linalg` SVD/eigh substrate. The `error_norm` signature currently exposes only the Frobenius path. |
| REQ-16 (get_precision / store_precision) | NOT-STARTED | open prereq blocker (tracking #1701). `precision()` always returns the stored precision; there is no `store_precision` flag and no `get_precision()` accessor that recomputes a pseudo-inverse when precision is not stored (`_empirical_covariance.py:216`). The `store_precision=True` default path is covered by `precision()`, but the API name (`get_precision`) and the `store_precision=False` branch are absent. |
| REQ-17 (collapsible_if lints) | SHIPPED | `cargo clippy -p ferrolearn-covariance --all-targets -- -D warnings` is GREEN (exit 0) — the `collapsible_if` lints were resolved by the workspace-wide `clippy --fix` (rust-1.95 drift cleanup). This row's prior FAILS claim was STALE; the crate gauntlet is unblocked (verified this iteration). |
| REQ-18 (ferray substrate) | NOT-STARTED | open prereq blocker (tracking #1701; R-SUBSTRATE). `covariance.rs` uses `ndarray::{Array1, Array2}`, hand-rolled `fn spd_inverse`/`fn cholesky`/`fn log_det_spd`, and `rand_xoshiro::Xoshiro256PlusPlus` + `rand_distr::Uniform` for the MCD RNG. Destination substrate is `ferray-core` (array), `ferray::linalg` (Cholesky/inverse/logdet), `ferray::random` (the numpy `RandomState` analog — also the lever for closing the REQ-5/REQ-12 RNG carve-out). Not migrated. |
| REQ-19 (consumer) | SHIPPED | `lib.rs` re-exports all six estimator types (`pub use covariance::{EllipticEnvelope, EmpiricalCovariance, ..., ShrunkCovariance}`). Dedicated non-test production consumers: `helpers.rs` constructs `EmpiricalCovariance`, `LedoitWolf`, `OAS`, `MinCovDet` (the `empirical_covariance`/`ledoit_wolf`/`oas`/`fast_mcd` functions); `graphical_lasso.rs` consumes `empirical_covariance` (`let emp_cov = empirical_covariance(x, self.assume_centered)?` in `GraphicalLasso::fit` and `GraphicalLassoCV`). There is NO `ferrolearn-python` binding for any covariance estimator (boundary S5: the estimator types ARE the public API, grandfathered; `helpers`/`graphical_lasso` are internal consumers). |

## Architecture

ferrolearn collapses sklearn's class hierarchy (`EmpiricalCovariance` base;
`ShrunkCovariance`/`LedoitWolf`/`OAS`/`MinCovDet` subclasses;
`EllipticEnvelope` wrapping `MinCovDet`) into six unfitted builder structs
sharing a `FittedCovariance<F> { covariance_, location_, precision_ }` core.
`LedoitWolf`/`OAS`/`MinCovDet`/`EllipticEnvelope` each have a dedicated fitted
type (`FittedLedoitWolf`, `FittedOAS`, `FittedMinCovDet`,
`FittedEllipticEnvelope`) wrapping `FittedCovariance` plus their extra
attributes (`shrinkage_`, `support_`, `threshold_`). sklearn keeps one class and
sets attributes on `self` in `fit`.

Shared numerical kernels: `fn empirical_cov` (`X_c^T X_c / n`, ddof=0),
`fn col_mean`, `fn spd_inverse` (Cholesky `L Lᵀ` + forward/back substitution),
`fn log_det_spd` (`2·Σ log diag(L)`), `fn shrink_covariance`,
`fn mahalanobis_distances`, and `fn chi2_quantile_approx` /
`fn normal_quantile_approx` (Wilson-Hilferty + Beasley-Springer-Moro
approximations). The chi-squared approximation is a known divergence source
versus sklearn's exact `scipy.stats.chi2` in the MCD correction and the
EllipticEnvelope threshold — folded into REQ-7's sub-finding and REQ-9.

**ShrunkCovariance default.** sklearn's class defaults `shrinkage=0.1`
(`_shrunk_covariance.py:161`); ferrolearn's `ShrunkCovariance::new(s)` takes the
coefficient as a required argument and has no `Default`. Parity holds when the
caller passes 0.1, but the "default" is not encoded in the type — a constructor
ergonomics gap (not a numerical divergence) noted under REQ-2.

**OAS divergence (REQ-4).** The ferrolearn OAS formula is the pre-1.5
formulation; sklearn 1.5.2 changed it (the docstring at
`_shrunk_covariance.py:653-658` explicitly notes the `2/p` term is omitted in
the implemented formula). This is the single deterministic value divergence and
is oracle-pinnable as a failing test.

**MCD / EllipticEnvelope RNG (REQ-5, REQ-12).** FastMCD's random h-subset draws
use `Xoshiro256PlusPlus` seeded from a `u64`; sklearn uses numpy `RandomState`.
The two RNG streams cannot be bit-matched, so exact `support_`/`covariance_`/
label values are an R-DEFER-3 carve-out — a documented blocker, NOT a failing
test. Closing this REQ requires the `ferray::random` numpy-`RandomState` analog
(REQ-18). The structural REQs (support size, SPD output, predict sign,
contamination domain) are RNG-independent and testable independently.

**EllipticEnvelope threshold (REQ-9).** The largest structural divergence:
sklearn derives `offset_` empirically from the *training* Mahalanobis distances
(`percentile(-dist_, 100*contamination)`), whereas ferrolearn uses a theoretical
chi-squared quantile. The sign convention (REQ-10) is correct; the threshold
*value* and its derivation are not.

Invariants: `covariance_` is symmetric PSD (forced SPD via Cholesky
regularisation / Tikhonov shrinkage in MCD); `precision_ = covariance_^{-1}`;
`mahalanobis` requires `X.ncols() == p` else `ShapeMismatch`.

## Verification

Commands establishing the SHIPPED claims (run at baseline `fa5b01d1`):

- `cargo test -p ferrolearn-covariance --lib` → 53 passed, 0 failed
  (REQ-1, REQ-2, REQ-3, REQ-6, REQ-7, REQ-10, REQ-19 structural tests).
- `cargo clippy -p ferrolearn-covariance --all-targets -- -D warnings` →
  **FAILS** with two `collapsible_if` errors (REQ-17 NOT-STARTED, blocks the
  gauntlet). All value-parity REQs that route through the lib are blocked from a
  clean gauntlet commit until REQ-17 lands.
- Live sklearn oracle (sklearn 1.5.2), asymmetric probe
  `X=[[1,2,3],[4,5,6],[7,8,10],[2,3,5],[3,1,2],[5,9,4]]`:
  - `EmpiricalCovariance().fit(X).covariance_[0]` =
    `[3.8888888888888893, 4.888888888888888, 3.833333333333333]`; ferrolearn
    `[3.8888888888888893, 4.888888888888888, 3.8333333333333335]` (REQ-1 ✓).
  - `ShrunkCovariance(shrinkage=0.1).fit(X).covariance_[0]` =
    `[4.148148148148149, 4.4, 3.45]`; ferrolearn
    `[4.148148148148149, 4.3999999999999995, 3.45]` (REQ-2 ✓).
  - `LedoitWolf().fit(X)` → `shrinkage_=0.4087180724344861`,
    `covariance_[0][0]=4.94852833594126`; ferrolearn `0.4087180724344862`,
    `4.948528335941261` (REQ-3 ✓).
  - `OAS().fit(X)` → `shrinkage_=0.6705854984555868`,
    `covariance_[0][0]=5.627443884884855`; ferrolearn `0.5386971696301457`,
    `5.2855111805226` — **DIVERGES** (REQ-4 NOT-STARTED, oracle-pinnable).
  - `EmpiricalCovariance().fit([[0,0],[2,0],[0,2],[2,2]]).mahalanobis(...)` =
    `[2.0, 2.0, 2.0, 2.0]` (squared); ferrolearn returns `sqrt(2)≈1.414` each —
    **DIVERGES by a square** (REQ-13 NOT-STARTED, oracle-pinnable).
  - `EmpiricalCovariance().fit([[0,0],[2,0],[0,2],[2,2]]).score(...)` =
    `-2.8378770664093453`; ferrolearn has no `score` method (REQ-14
    NOT-STARTED, oracle-pinnable once added).

A critic should pin REQ-4 (OAS formula), REQ-13 (squared mahalanobis), and (once
the method exists) REQ-14 (`score`) as failing `#[test]`s with expected values
from the live oracle above — never copied from the ferrolearn side (R-CHAR-3).
REQ-5 and REQ-12 are RNG carve-outs (blocker, NO failing test per R-DEFER-3).

Per R-DEFER-2 the table is binary. SHIPPED: REQ-1, REQ-2, REQ-3, REQ-6, REQ-7,
REQ-10, REQ-19 (impl + non-test consumer + green structural/oracle verification).
NOT-STARTED (tracking #1701; the critic files per-REQ blockers): REQ-4 (OAS
formula), REQ-5 (MCD exact values — RNG carve-out), REQ-8 (raw_* attributes),
REQ-9 (offset_ definition), REQ-11 (contamination right endpoint), REQ-12 (EE
exact labels — RNG carve-out), REQ-13 (squared mahalanobis), REQ-14 (score),
REQ-15 (error_norm), REQ-16 (get_precision/store_precision), REQ-17
(collapsible_if lints), REQ-18 (ferray substrate).
