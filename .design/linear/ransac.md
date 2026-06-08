# RANSAC Regressor

<!--
tier: 3-component
status: draft
baseline-commit: 44c12d6370ac5c23cc1cc100223610fa18e09198
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/linear_model/_ransac.py
ferrolearn-module: ferrolearn-linear/src/ransac.rs
parity-op: RANSACRegressor
crosslink-issue: 511
-->

## Summary

`ferrolearn-linear/src/ransac.rs` mirrors scikit-learn's
`sklearn.linear_model.RANSACRegressor` (`sklearn/linear_model/_ransac.py`): a
meta-estimator that wraps a base regressor and fits it robustly by repeatedly
sampling random subsets, classifying each sample as an inlier or outlier by a
residual threshold, and keeping the best consensus set. ferrolearn implements
the random-subset loop, the MAD residual-threshold default, the inlier
classification, prediction delegation, and the unfitted/fitted builder API
(`RANSACRegressor::new` / `FittedRANSACRegressor::inlier_mask`). However the
**decision rule** of the loop diverges from sklearn in several concrete ways
(selection criterion, refit semantics, acceptance gate, dynamic max-trials,
MAD-zero handling, loss), and the introspection attributes
(`estimator_`, `n_trials_`, `n_skips_*`, `n_features_in_`) and several
constructor parameters (`is_data_valid`, `is_model_valid`, `max_skips`,
`stop_n_inliers`, `stop_score`, `stop_probability`, `loss`) are absent. The
crate is also on the wrong substrate (`ndarray`, not ferray).

## Algorithm (sklearn — the contract)

`RANSACRegressor.fit` (`_ransac.py:325-606`) runs:

1. **Init** (`_ransac.py:451-465`): `n_inliers_best = 1`, `score_best = -inf`,
   `inlier_mask_best = None`, `n_skips_* = 0`, `n_trials_ = 0`,
   `max_trials = self.max_trials`.
2. **Loop** `while self.n_trials_ < max_trials` (`_ransac.py:467`):
   - Draw a random subset of `min_samples` indices without replacement
     (`sample_without_replacement`, `_ransac.py:478-480`).
   - Optional `is_data_valid(X_subset, y_subset)` gate; on fail
     `n_skips_invalid_data_ += 1; continue` (`_ransac.py:485-489`).
   - **Fit the base estimator on the SUBSET** (`_ransac.py:497`).
   - Optional `is_model_valid(estimator, X_subset, y_subset)` gate; on fail
     `n_skips_invalid_model_ += 1; continue` (`_ransac.py:500-504`).
   - Residuals of ALL data from the subset model:
     `residuals = loss(y, estimator.predict(X))` (`_ransac.py:507-508`).
   - Inlier mask `residuals <= residual_threshold`,
     `n_inliers_subset = sum(mask)` (`_ransac.py:511-512`).
   - **Acceptance**: if `n_inliers_subset < n_inliers_best`:
     `n_skips_no_inliers_ += 1; continue` (`_ransac.py:514-517`).
   - **Score-rank**: `score_subset = estimator.score(X_inlier, y_inlier)` =
     the base estimator's R² on the inlier set (`_ransac.py:530-534`); if
     `n_inliers_subset == n_inliers_best and score_subset < score_best:
     continue` (`_ransac.py:538-539`). **Higher score wins ties on inlier
     count.**
   - **Save best**: record `n_inliers_best`, `score_best`,
     `inlier_mask_best = inlier_mask_subset` (from the SUBSET model — NOT
     recomputed), `X_inlier_best`, `y_inlier_best`, `inlier_best_idxs_subset`
     (`_ransac.py:542-547`).
   - **Dynamic max-trials**: `max_trials = min(max_trials,
     _dynamic_max_trials(n_inliers_best, n_samples, min_samples,
     stop_probability))` (`_ransac.py:549-554`, helper at `_ransac.py:48-79`).
   - **Stop**: break if `n_inliers_best >= stop_n_inliers or
     score_best >= stop_score` (`_ransac.py:557-558`).
   - Also break if `sum(n_skips_*) > max_skips` (`_ransac.py:470-475`).
3. **Single final refit** (`_ransac.py:597-602`): `estimator.fit(X_inlier_best,
   y_inlier_best)` ONCE, AFTER the loop. The inlier mask is **never recomputed**
   from this refitted model.
4. **Attributes** (`_ransac.py:604-605`): `self.estimator_ = estimator`,
   `self.inlier_mask_ = inlier_mask_best`. Also exposed:
   `n_trials_`, `n_skips_no_inliers_`, `n_skips_invalid_data_`,
   `n_skips_invalid_model_`, `n_features_in_`.

`residual_threshold` defaults to the MAD of `y`:
`np.median(np.abs(y - np.median(y)))` with NO special-casing of zero
(`_ransac.py:399-401`). `min_samples` defaults to `X.shape[1] + 1` for the
`LinearRegression` base, and accepts a float fraction in `(0,1)` →
`ceil(min_samples * n_samples)` (`_ransac.py:382-397`). `loss` is
`'absolute_error'` (default), `'squared_error'`, or a callable
(`_ransac.py:405-421`).

## Parity boundary / RNG

**The random-draw path is inherently non-parityable.** sklearn draws subsets
with `sample_without_replacement(n_samples, min_samples,
random_state=random_state)` over numpy's Mersenne-Twister
(`check_random_state`, `_ransac.py:423`, `:478`). ferrolearn draws with a
Fisher-Yates partial shuffle over `rand::rngs::StdRng` seeded from a `u64`
(`fn sample_indices in ransac.rs`, seeded at `fn fit in ransac.rs`,
`ransac.rs:257-260, 173-181`). These two PRNGs produce **different sequences of
subsets** from the same seed; no seed mapping makes ferrolearn draw the same
candidate sets as sklearn. Cross-implementation parity on WHICH subsets are
drawn — and therefore on the exact fitted `coef_`/`intercept_`/`inlier_mask_`
for a stochastic dataset — is **infeasible** and is NOT a contract REQ.

Additionally, ferrolearn defaults the seed to **42** when `random_state` is
`None` (`ransac.rs:259`), so ferrolearn's `random_state=None` is deterministic,
whereas sklearn's `None` consumes fresh global entropy (`check_random_state`,
`_ransac.py:423`). These are not equivalent and parity on the `None` case is
also out of scope.

Parity is therefore asserted **only on the deterministic decision rules** —
given the SAME candidate inlier set, do the following match sklearn?

- **Threshold default** (MAD of `y`), including the MAD-zero case (REQ-2, REQ-9).
- **Inlier classification** (`residual <= threshold`) (REQ-3).
- **Selection criterion** — rank by `(n_inliers, base-estimator R² score)`,
  higher score wins ties (REQ-4).
- **Refit semantics** — single final refit after the loop; `inlier_mask_` taken
  from the subset model, not recomputed (REQ-5).
- **Acceptance gate** — `n_inliers_best` init and the
  `n_inliers_subset >= n_inliers_best` accept rule (REQ-6).
- **Dynamic max-trials + stop criteria** (REQ-7).
- **Loss family** — `absolute_error` / `squared_error` (REQ-8).

REQ-1 (the sampling loop) is SHIPPED as a *structural* requirement (a loop that
draws `min_samples`-sized subsets exists and is seedable), explicitly carrying
the RNG non-parity caveat above; it does NOT claim subset-sequence parity.

## Requirements

- REQ-1: Random-subset sampling loop — draw `min_samples` distinct indices per
  trial, up to `max_trials`, deterministically reproducible from a seed
  (structural; RNG-sequence parity with sklearn explicitly out of scope).
- REQ-2: `residual_threshold` defaults to the MAD of `y`
  (`median(|y - median(y)|)`).
- REQ-3: Inlier classification — a sample is an inlier iff its residual
  (`|y - y_pred|` under absolute-error loss) `<= residual_threshold`.
- REQ-4: Selection criterion — rank candidate consensus sets by
  `(n_inliers_best, score_best)` where `score_best` is the base estimator's R²
  on the inlier set; higher score wins ties on inlier count.
- REQ-5: Refit-once-after-loop semantics — the inlier refit happens exactly once
  AFTER the loop, and `inlier_mask_` is taken from the subset model (never
  recomputed from the refitted model).
- REQ-6: `n_inliers_best` initialized to `1` with acceptance gate
  `n_inliers_subset >= n_inliers_best` (skip only when strictly fewer).
- REQ-7: Dynamic `max_trials` shrink (`_dynamic_max_trials`) and stop criteria
  (`stop_n_inliers`, `stop_score`, `stop_probability`, `max_skips`), with
  `n_trials_`/`n_skips_*` tracking.
- REQ-8: `loss='squared_error'` (and the callable-loss option) in addition to
  the absolute-error default.
- REQ-9: MAD-zero parity — when the MAD of `y` is `0` (e.g. constant target),
  `residual_threshold = 0` (no `1e-6` substitution).
- REQ-10: Introspection attributes — `estimator_`, `n_trials_`,
  `n_skips_no_inliers_`, `n_skips_invalid_data_`, `n_skips_invalid_model_`,
  `n_features_in_`.
- REQ-11: `is_data_valid`, `is_model_valid`, `max_skips` parameters and the
  skip-accounting / `ConvergenceWarning` they drive.
- REQ-12: `min_samples` as a float fraction in `(0,1)` →
  `ceil(min_samples * n_samples)`.
- REQ-13: ferray substrate migration (array type → `ferray-core`; sampling →
  `ferray::random`) per R-SUBSTRATE.

## Acceptance criteria

- AC-1 (REQ-1): two fits with the same `random_state` produce identical
  `inlier_mask` (`test_ransac_reproducible_with_seed`); the loop draws
  `min_samples`-sized subsets. (No sklearn subset-sequence comparison — see
  Parity boundary.)
- AC-2 (REQ-2/REQ-9): for `y` with a nonzero MAD and `residual_threshold=None`,
  the auto threshold equals `median(|y - median(y)|)`; for a constant `y` the
  threshold equals `0.0` (matching the live oracle
  `np.median(np.abs(yc - np.median(yc))) == 0.0`).
- AC-3 (REQ-3): a sample with `|y - y_pred| == threshold` is classified as an
  inlier (boundary inclusive, per `_ransac.py:511`).
- AC-4 (REQ-4): given two candidate inlier sets of equal size, the one whose
  base-estimator R² on its inliers is HIGHER is selected (oracle:
  `estimator.score(X_inlier, y_inlier)`, higher wins).
- AC-5 (REQ-5): `inlier_mask_` equals the mask computed from the SUBSET model
  that won, and the refitted estimator is fit exactly once after the loop; the
  reported mask is not recomputed from the refitted model.
- AC-6 (REQ-6): a candidate with `n_inliers == n_inliers_best` is NOT skipped on
  count alone (it advances to the score tiebreak); `n_inliers_best` starts at 1.
- AC-7 (REQ-7): after convergence, `n_trials_ <= max_trials` and may be `<`
  `max_trials` when a stop criterion fires; `n_skips_no_inliers_` matches the
  oracle (e.g. `2` for the `[2,4,6,8,10,100]` / `threshold=2.0` / `seed=0`
  case).
- AC-8 (REQ-8): with `loss='squared_error'`, residuals are `(y - y_pred)**2`.
- AC-10 (REQ-10): `n_features_in_`, `n_trials_`, `n_skips_*`, and `estimator_`
  are exposed on the fitted object (oracle: `m.n_features_in_ == 1`,
  `m.n_trials_ == 4`, `m.estimator_.coef_ == [2.0]` for the seed-0 fixture).
- AC-12 (REQ-12): `min_samples=0.5` on `n_samples=10` yields an effective
  `min_samples = ceil(5.0) = 5`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ tests + green; NOT-STARTED = concrete open blocker referenced by `#`-number.
`RANSACRegressor`/`FittedRANSACRegressor` are boundary estimator types re-exported
at the crate root (`pub use ransac::{FittedRANSACRegressor, RANSACRegressor} in
lib.rs`); under S5/R-DEFER-1 the public estimator type IS the consumer surface,
grandfathered (there is no `ferrolearn-python` binding for RANSAC yet).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (sampling loop) | SHIPPED | impl `fn sample_indices in ransac.rs` draws `k` distinct indices via Fisher-Yates (`ransac.rs:173-181`: `let j = rng.random_range(i..n); indices.swap(i, j); ... indices.truncate(k)`), called per trial in `fn fit in ransac.rs` (`ransac.rs:267-269`: `for _ in 0..self.max_trials { let indices = sample_indices(&mut rng, n_samples, min_samples);`), seeded deterministically (`ransac.rs:257-260`). Boundary type re-exported at crate root (`pub use ransac::{...RANSACRegressor} in lib.rs`). Tests: `test_ransac_reproducible_with_seed in ransac.rs` (same seed → identical mask). **Caveat: RNG-sequence parity with sklearn's Mersenne-Twister `sample_without_replacement` is non-parityable (see Parity boundary); structural REQ only.** Verification: `cargo test -p ferrolearn-linear ransac`. |
| REQ-2 (MAD threshold default) | SHIPPED | impl `fn fit in ransac.rs` computes the auto threshold from `fn mad in ransac.rs` (`ransac.rs:162-166`: `let med = median(values); ... median(&abs_devs)`) when `residual_threshold` is `None` (`ransac.rs:245-255`), mirroring `_ransac.py:401` (`residual_threshold = np.median(np.abs(y - np.median(y)))`). Tests: `test_ransac_auto_threshold in ransac.rs`. Verification: `cargo test -p ferrolearn-linear ransac`. **Note:** the zero-MAD sub-case diverges and is split out as REQ-9 (NOT-STARTED). |
| REQ-3 (inlier classification) | SHIPPED | impl `fn fit in ransac.rs` classifies `let residual = (preds[i] - y[i]).abs(); if residual <= threshold { inlier_mask[i] = true; ... }` (`ransac.rs:288-295`), mirroring `_ransac.py:511` (`inlier_mask_subset = residuals_subset <= residual_threshold`) with the boundary-inclusive `<=`. Tests: `test_ransac_with_outlier`, `test_ransac_multiple_outliers in ransac.rs` (outliers excluded, clean points retained). Verification: `cargo test -p ferrolearn-linear ransac`. |
| REQ-4 (selection criterion: n_inliers then R² score) | NOT-STARTED | open prereq blocker #512. ferrolearn ranks ties by `residual_sum < best_residual_sum` (sum of absolute residuals over inliers, LOWER wins) — `fn fit in ransac.rs` (`ransac.rs:298-299`: `let is_better = n_inliers > best_n_inliers \|\| (n_inliers == best_n_inliers && residual_sum < best_residual_sum);`). sklearn ranks ties by `score_subset = estimator.score(X_inlier, y_inlier)` (base-estimator R², HIGHER wins; `_ransac.py:530-543`). Different decision rule; the base estimator's `score`/R² is never called. |
| REQ-5 (refit-once-after-loop; mask from subset model) | NOT-STARTED | open prereq blocker #513. ferrolearn refits on inliers INSIDE the loop on every improvement AND recomputes the mask/count from the refitted model, storing that (`fn fit in ransac.rs`, `ransac.rs:301-337`: `if let Ok(refit) = ... { if let Ok(new_preds) = refit.predict(x) { ... best_inlier_mask = Some(new_mask); ... } }`). sklearn refits exactly ONCE after the loop (`_ransac.py:602`) and reports `inlier_mask_ = inlier_mask_best` from the SUBSET model, never recomputed (`_ransac.py:544, :605`). Divergent refit count and reported mask. |
| REQ-6 (n_inliers_best init / acceptance gate) | NOT-STARTED | open prereq blocker #514. ferrolearn starts `best_n_inliers = 0usize` (`ransac.rs:264`) and gates acceptance on `n_inliers >= min_samples` (`fn fit in ransac.rs`, `ransac.rs:301`: `if is_better && n_inliers >= min_samples`). sklearn starts `n_inliers_best = 1` (`_ransac.py:451`) and accepts any candidate with `n_inliers_subset >= n_inliers_best`, skipping only when strictly fewer (`_ransac.py:515`); it never gates on `min_samples`. Different init and acceptance condition. |
| REQ-7 (dynamic max_trials + stop criteria) | NOT-STARTED | open prereq blocker #515. ferrolearn runs a FIXED `for _ in 0..self.max_trials` loop (`ransac.rs:267`) with no `_dynamic_max_trials` shrink, no `stop_n_inliers`/`stop_score`/`stop_probability` break, no `max_skips`, and no `n_trials_`/`n_skips_*` accounting. sklearn shrinks `max_trials` via `_dynamic_max_trials` (`_ransac.py:48-79, :549-554`) and breaks on stop criteria (`_ransac.py:557`). |
| REQ-8 (loss='squared_error') | SHIPPED | impl `pub enum RansacLoss { #[default] AbsoluteError, SquaredError } in ransac.rs` + field `loss` + `with_loss in ransac.rs`. `fn fit in ransac.rs` branches the per-sample residual on `self.loss` (`AbsoluteError → (preds[i]-y[i]).abs()`, `SquaredError → { let d = preds[i]-y[i]; d*d }`), applied at the `residual <= threshold` classification, mirroring `_ransac.py:407,414,508,511`; the MAD-default threshold is loss-independent (`_ransac.py:399-401`). sklearn-callable loss is not supported (Rust analog is the enum; callable-loss omitted, no separate blocker — covered by the str-options contract). Consumer: boundary type re-export + `pub use ransac::{...RansacLoss} in lib.rs`. Tests (live-oracle, RNG-independent on a clean line + one outlier): `ransac_loss_squared_error_recovers_line`, `ransac_loss_default_absolute_error_byte_identical` (tests/divergence_ransac_fit.rs); oracle `coef≈2.0, intercept≈1.0`, 9 inliers. Closed #516. |
| REQ-9 (MAD-zero parity) | NOT-STARTED | open prereq blocker #517. ferrolearn substitutes `1e-6` when MAD `<= epsilon` (`fn fit in ransac.rs`, `ransac.rs:249-254`: `if y_mad <= F::epsilon() { F::from(1e-6).unwrap() } else { y_mad }`). sklearn applies no special-case: `residual_threshold = np.median(np.abs(y - np.median(y)))` can be exactly `0` (`_ransac.py:401`; live oracle: MAD of a constant `y` is `0.0`). Divergent threshold for constant/near-constant targets. |
| REQ-10 (introspection attributes) | NOT-STARTED | open prereq blocker #518. `FittedRANSACRegressor` stores only `fitted_estimator` and `inlier_mask` (`ransac.rs:127-132`) and exposes only `inlier_mask()` (`ransac.rs:137`). Missing vs sklearn Attributes (`_ransac.py:192-224`): `estimator_` accessor, `n_trials_`, `n_skips_no_inliers_`, `n_skips_invalid_data_`, `n_skips_invalid_model_`, `n_features_in_`. (Live oracle exposes `n_trials_=4`, `n_skips_no_inliers_=2`, `n_features_in_=1` on the seed-0 fixture.) |
| REQ-11 (is_data_valid / is_model_valid / max_skips) | NOT-STARTED | open prereq blocker #519. `RANSACRegressor` struct (`ransac.rs:59-71`) has no `is_data_valid`, `is_model_valid`, or `max_skips` fields. sklearn gates each subset/model on these callables and accumulates `n_skips_*`, raising `ValueError`/`ConvergenceWarning` when `sum(n_skips_*) > max_skips` (`_ransac.py:485-504, :561-595`). |
| REQ-12 (min_samples float fraction) | SHIPPED | impl `pub enum MinSamples<F> { Count(usize), Fraction(F) } in ransac.rs` + field `min_samples: Option<MinSamples<F>>` + builders `with_min_samples (→ Count)`, `with_min_samples_fraction (→ Fraction)`, getter `min_samples in ransac.rs`. `fn fit in ransac.rs` resolves `None → n_features+1`, `Count(k) → k`, `Fraction(f) → ceil(f·n_samples)` validating `0 < f < 1` (else `FerroError::InvalidParameter`), plus the resolved-count `> n_samples → InvalidParameter` guard, mirroring `_ransac.py:382-397` (sklearn `ValueError`; constraint `Interval(RealNotInt, 0, 1, closed="both")` `_ransac.py:264`). Note: sklearn treats an integral-valued float (e.g. `1.0`) as the Integral count branch; ferrolearn's API splits these into the explicit `Count`/`Fraction` variants, so `Fraction` requires a genuine `(0,1)` value. Consumer: boundary type re-export + `pub use ransac::{...MinSamples} in lib.rs`. Tests (live-oracle): `ransac_min_samples_fraction_resolves_ceil` (0.5 on n=10 → ceil(5.0)=5 → sklearn 9 inliers), `ransac_min_samples_fraction_out_of_range_errors` (0.0 / 1.5 / -0.1 → Err), `ransac_min_samples_count_unchanged` (tests/divergence_ransac_fit.rs). Closed #520. |
| REQ-13 (ferray substrate) | NOT-STARTED | open prereq blocker #521. `ransac.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` (`ransac.rs:39`) and samples via `rand::rngs::StdRng` (`ransac.rs:41-42, 258`), not `ferray-core` arrays or `ferray::random` (R-SUBSTRATE-1/2). |

## Architecture

### sklearn (the contract)

`RANSACRegressor(MetaEstimatorMixin, RegressorMixin, MultiOutputMixin,
BaseEstimator)` (`_ransac.py:82-87`). The base estimator defaults to
`LinearRegression()` when `None` (`_ransac.py:377-380`). `fit` validates input,
resolves `min_samples` / `residual_threshold` / `loss_function`, runs the
sampling-and-scoring loop (Algorithm §2), refits once on the best inlier set,
and stores `estimator_` + `inlier_mask_` plus the `n_trials_`/`n_skips_*`
diagnostics. `predict`/`score` are thin wrappers over `estimator_`
(`_ransac.py:608-696`). Tie-break ordering is `(n_inliers, R²)` with higher R²
winning (`_ransac.py:538`).

### ferrolearn (what exists)

`RANSACRegressor<F, E> { estimator, min_samples: Option<usize>,
residual_threshold: Option<F>, max_trials: usize, random_state: Option<u64> }`
(`ransac.rs:59-71`), with a builder API (`new`, `with_min_samples`,
`with_residual_threshold`, `with_max_trials`, `with_random_state`). `fn fit`
(`impl Fit for RANSACRegressor in ransac.rs`):

- Validates shapes (`FerroError::ShapeMismatch`) and sample count
  (`FerroError::InsufficientSamples`), defaults `min_samples = n_features + 1`
  (`ransac.rs:234`), computes the MAD threshold (with the `1e-6` zero-MAD
  substitution — REQ-9 divergence), and seeds `StdRng` (default 42 — RNG
  divergence).
- Runs a FIXED `for _ in 0..self.max_trials` loop (no dynamic trials / stop
  criteria — REQ-7): sample a subset, fit the base estimator, predict on all
  `X`, classify inliers by absolute residual `<= threshold`, and track the best
  by `(n_inliers, residual_sum)` with `residual_sum` LOWER-wins (REQ-4
  divergence). On improvement it refits on inliers INSIDE the loop and
  recomputes the stored mask from the refitted model (REQ-5 divergence), gating
  acceptance on `n_inliers >= min_samples` (REQ-6 divergence).
- Returns `FittedRANSACRegressor<Ef> { fitted_estimator, inlier_mask }` or
  `FerroError::ConvergenceFailure` if no model was accepted.

`FittedRANSACRegressor<Fitted>` (`ransac.rs:127-140`) exposes only
`inlier_mask()` and implements `Predict` by delegating to `fitted_estimator`
(`ransac.rs:353-369`). No `estimator_`/`n_trials_`/`n_skips_*`/`n_features_in_`
accessors (REQ-10).

### Why the loop diverges from the contract

The two implementations agree on the *shape* of RANSAC (sample → fit → residual
→ inlier mask → keep-best → final refit) but disagree on five deterministic
decision rules — selection criterion (REQ-4), refit count and reported-mask
source (REQ-5), acceptance init/gate (REQ-6), dynamic trials/stop (REQ-7), and
zero-MAD threshold (REQ-9) — plus the loss family (REQ-8). Because the subset
draw is non-parityable (Parity boundary), these decision-rule REQs are the
contract: the divergence tests pin them by constructing candidate inlier sets
directly (or by comparing the auto-threshold / tie-break / refit-count behavior)
against the live sklearn oracle, never by comparing the stochastic
subset-by-subset trajectory.

## Verification

Commands that establish the SHIPPED claims (baseline `44c12d6`):

- `cargo test -p ferrolearn-linear ransac` — the module unit tests
  (`test_ransac_no_outliers`, `test_ransac_with_outlier`,
  `test_ransac_multiple_outliers`, `test_ransac_shape_mismatch`,
  `test_ransac_insufficient_samples`, `test_ransac_reproducible_with_seed`,
  `test_ransac_auto_threshold`) pin REQ-1/REQ-2/REQ-3.
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the NOT-STARTED gaps; expected values per
R-CHAR-3 come from sklearn, never copied from ferrolearn):

```bash
python3 -c "import numpy as np; from sklearn.linear_model import RANSACRegressor; \
X=np.arange(6).reshape(-1,1).astype(float); y=np.array([2.,4.,6.,8.,10.,100.]); \
m=RANSACRegressor(random_state=0, residual_threshold=2.0).fit(X,y); \
print(m.inlier_mask_.tolist(), m.n_trials_, m.n_skips_no_inliers_, \
m.estimator_.coef_.tolist(), m.n_features_in_)"
# -> [True,True,True,True,True,False] 4 2 [2.0] 1   (REQ-7/REQ-10 gaps)
python3 -c "import numpy as np; yc=np.full(6,5.0); \
print(np.median(np.abs(yc-np.median(yc))))"
# -> 0.0   (REQ-9 gap: ferrolearn substitutes 1e-6)
```

REQ-4, REQ-5, REQ-6, REQ-7, REQ-8, REQ-9, REQ-10, REQ-11, REQ-12, REQ-13 have no
green verification against the sklearn contract and are NOT-STARTED, each gated
on the blocker below. A NOT-STARTED REQ closes only when its fix lands AND a
divergence test (expected values from the live oracle / a sklearn `file:line`
constant per R-CHAR-3) goes green.

## Blockers to open

- **#512** — REQ-4 of ransac: selection criterion must rank ties by the base
  estimator's R² on the inlier set (`estimator.score`, higher wins), not by
  `residual_sum` (lower wins). Requires the base-estimator `score`/R² surface.
- **#513** — REQ-5 of ransac: refit the base estimator exactly ONCE after the
  loop and report `inlier_mask_` from the winning SUBSET model (do not recompute
  the mask from the refitted model).
- **#514** — REQ-6 of ransac: initialize `n_inliers_best = 1` and accept on
  `n_inliers_subset >= n_inliers_best` (skip only when strictly fewer); drop the
  `n_inliers >= min_samples` acceptance gate.
- **#515** — REQ-7 of ransac: add `_dynamic_max_trials` shrink and the
  `stop_n_inliers` / `stop_score` / `stop_probability` / `max_skips` stop
  criteria, with `n_trials_` / `n_skips_*` tracking.
- **#516** — REQ-8 of ransac: add the `loss` parameter with `'absolute_error'`
  (default) and `'squared_error'` (`(y - y_pred)**2`); callable loss optional.
- **#517** — REQ-9 of ransac: remove the `1e-6` zero-MAD substitution; the auto
  `residual_threshold` must equal the MAD of `y` exactly, including `0`.
- **#518** — REQ-10 of ransac: expose `estimator_`, `n_trials_`,
  `n_skips_no_inliers_`, `n_skips_invalid_data_`, `n_skips_invalid_model_`,
  `n_features_in_` on the fitted object.
- **#519** — REQ-11 of ransac: add `is_data_valid`, `is_model_valid`,
  `max_skips` parameters with the skip accounting and `ConvergenceWarning` /
  `ValueError` they drive.
- **#520** — REQ-12 of ransac: accept `min_samples` as a float fraction in
  `(0,1)` resolving to `ceil(min_samples * n_samples)`.
- **#521** — REQ-13 of ransac: migrate `ransac.rs` off `ndarray` / `rand` onto
  the ferray substrate (`ferray-core` arrays, `ferray::random`) per
  R-SUBSTRATE.
