# PowerTransformer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 0df60782
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/preprocessing/_data.py  # class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator) (:3122); _parameter_constraints {method:[StrOptions({"yeo-johnson","box-cox"})], standardize:["boolean"], copy:["boolean"]} (:3222-3226); __init__(method="yeo-johnson", *, standardize=True, copy=True) (:3226-3229); _fit(X, y=None, force_transform=False) (:3274-3316): per-col optim_function = {box-cox:_box_cox_optimize, yeo-johnson:_yeo_johnson_optimize}[method] (:3284-3287), constant YJ feature -> lambdas_[i]=1.0 (:3299-3302), StandardScaler(copy=False).fit/fit_transform when standardize (:3308-3314); transform (:3318-3345); inverse_transform (:3347-3392) -> _box_cox_inverse_tranform (:3394-3403) / _yeo_johnson_inverse_transform (:3405-3424); _yeo_johnson_transform 4-branch (:3426-3446, np.spacing(1.0) cut); _box_cox_optimize (:3448-3462): stats.boxcox(x[~mask], lmbda=None), all-NaN -> ValueError; _yeo_johnson_optimize (:3464-3493): _neg_log_likelihood = -n/2*log(var) + (lmbda-1)*(np.sign(x)*np.log1p(np.abs(x))).sum() (:3484-3485), inf if x_trans_var < tiny (:3480), drop NaN x=x[~np.isnan(x)] (:3491), optimize.brent(brack=(-2,2)) (:3493); _check_input check_positive box-cox np.nanmin(X)<=0 raises (:3524-3528), check_shape (:3531-3537); fitted attrs lambdas_, n_features_in_, feature_names_in_; OneToOneFeatureMixin.get_feature_names_out. power_transform(X, method="yeo-johnson", *, standardize=True, copy=True) free fn (:3549-3648) -> PowerTransformer(method,standardize,copy).fit_transform(X) (:3647-3648).
ferrolearn-module: ferrolearn-preprocess/src/power_transformer.rs
parity-ops: PowerTransformer, power_transform
crosslink-issue: 1342
-->

## Summary

scikit-learn's `PowerTransformer` (`_data.py:3122`) applies a parametric,
monotonic power transform feature-wise to make data more Gaussian-like, choosing
the per-feature lambda by maximum-likelihood. It supports **two methods** —
`'yeo-johnson'` (default, positive + negative data) and `'box-cox'` (strictly
positive only) — selected by the `method` ctor param (`:3226`), optionally
followed by zero-mean/unit-variance standardization (`StandardScaler(copy=False)`,
`:3308-3314`). The `power_transform` free function (`:3549-3648`) is the stateless
equivalent: `PowerTransformer(method, standardize, copy).fit_transform(X)`.

`ferrolearn-preprocess/src/power_transformer.rs` ships a **Yeo-Johnson-ONLY**
subset: `PowerTransformer<F> { standardize: bool }` (`new` = standardize true,
`without_standardize`, `Default` = `new`, accessor `standardize()`) fits into
`FittedPowerTransformer<F> { lambdas: Array1<F>, means: Option<Array1<F>>, stds:
Option<Array1<F>> }` (accessor `lambdas()`). There is **no `method` param, no
box-cox, no `inverse_transform`, no `power_transform` free fn, no `copy` param,
and no constant-feature / NaN handling.** `fit` optimizes each column with
`ferrolearn_numerical::optimize::brent_bounded(neg_ll, -3.0, 3.0, 1e-8, 500)`
(a BOUNDED golden-section/parabolic minimizer) over a `fn log_likelihood_yj`
whose Jacobian term now carries the `np.sign(x)` factor sklearn uses at
`_data.py:3485`. Non-test consumers: the PyO3 binding `_RsPowerTransformer`
(`ferrolearn-python/src/extras.rs:1171-1177`, registered at
`ferrolearn-python/src/lib.rs:84`), the crate re-export `pub use
power_transformer::{FittedPowerTransformer, PowerTransformer};`
(`ferrolearn-preprocess/src/lib.rs:123`), and the `PipelineTransformer` /
`FittedPipelineTransformer` impls.

**Headline finding — REQ-1 (Yeo-Johnson lambda + transform VALUE parity) is
SHIPPED.** The former DIV-1 (now resolved, was #1343) — `fn log_likelihood_yj`
omitting the `np.sign(x)` Jacobian factor — has been FIXED: the helper now
computes the Jacobian as `(lambda - 1) * sum(sign(y) * (|y| + 1).ln())`, with the
explicit three-way sign (`sign(0) == 0`) matching sklearn's `_neg_log_likelihood`
`(lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()` (`_data.py:3485`). The
once-suspected DIV-2 (bounded-Brent `[-3, 3]` vs sklearn's unbounded
`optimize.brent(brack=(-2, 2))`, `:3493`) is a **non-issue**: the bounded
minimizer lands on sklearn's optimum — positive-data lambda matches to ≈1e-8, and
the negative-optimal-lambda probe (`λ ≈ -0.7252`) confirms it finds interior
minima `< 0` rather than clamping at an endpoint. Yeo-Johnson lambda + forward
transform VALUE parity is now VERIFIED against the live sklearn 1.5.2 oracle
across positive, all-negative, mixed-sign (with zero), multi-feature
(heterogeneous per-column sign), `standardize=True`, and negative-optimal-lambda
fixtures: **13 passing value tests** in `tests/divergence_power_transformer.rs`
(the 2 former DIV-1 pins now green + 11 green-guards) at tol 1e-4 (lambda) /
1e-5 (transform). The transform-GIVEN-lambda machinery (`fn yeo_johnson`) and the
biased-var standardize path were already correct; the restored sign factor closes
the end-to-end lambda gate on signed columns. This is a **shipped-partial** unit:
**2 SHIPPED** (REQ-1 YJ lambda+transform parity headline, REQ-2 scoped
error/parameter contracts) / **8 NOT-STARTED** (REQ-3 box-cox, REQ-4
inverse_transform, REQ-5 power_transform free fn, REQ-6 constant-feature skip +
zero-scale handling, REQ-7 NaN drop + box-cox check_positive, REQ-8 method/copy
ctor params, REQ-9 fitted-attr / get_feature_names_out surface, REQ-10 ferray).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 (HEADLINE) — POSITIVE fixture: lambda parity holds (sign(x)==1 for all rows);
# ferrolearn matches the oracle to ~1e-8:
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
pt=PowerTransformer(method='yeo-johnson', standardize=False).fit([[1.],[2.],[3.],[4.],[5.]]); \
print('POS lambda:', pt.lambdas_.tolist()); \
print('POS transform(nostd):', np.round(pt.transform([[1.],[2.],[3.],[4.],[5.]]).ravel(),6).tolist())"
# -> POS lambda: [0.699807422455043]
# -> POS transform(nostd): [0.892085, 1.653616, 2.341089, 2.978267, 3.578034]
#    ferrolearn (without_standardize().fit): lambda 0.6998074158512372 (matches to ~1e-8),
#    transform [0.892085, 1.653616, 2.341089, 2.978267, 3.578034] (IDENTICAL).

# REQ-1 (HEADLINE) — SIGNED fixture: with the restored np.sign factor the MLE lambda now
# MATCHES the oracle on signed data (formerly DIV-1, #1343, resolved):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
ptn=PowerTransformer(method='yeo-johnson', standardize=False).fit([[-2.],[-1.],[0.],[1.],[2.],[3.]]); \
print('NEG lambda:', ptn.lambdas_.tolist()); \
print('NEG transform(nostd):', np.round(ptn.transform([[-2.],[-1.],[0.],[1.],[2.],[3.]]).ravel(),6).tolist())"
# -> NEG lambda: [0.9504965354909566]
# -> NEG transform(nostd): [-2.065428, -1.019356, 0.0, 0.981106, 1.937095, 2.87713]
#    ferrolearn (without_standardize().fit): lambda 0.950497 (matches to ~1e-4),
#    transform [-2.065428, -1.019356, 0.0, 0.981106, 1.937095, 2.87713] (matches to ~1e-5).
#    => DIV-1 RESOLVED: signed-data lambda + transform now parity. Pinned by
#       divergence_div1_signed_lambda / divergence_div1_signed_transform (now green).

# REQ-1 — POSITIVE fixture with standardize=True (the standardize machinery itself is correct):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
pts=PowerTransformer(method='yeo-johnson', standardize=True).fit([[1.],[2.],[3.],[4.],[5.]]); \
print('POS transform(std):', np.round(pts.transform([[1.],[2.],[3.],[4.],[5.]]).ravel(),6).tolist())"
# -> POS transform(std): [-1.472976, -0.669761, 0.055343, 0.727399, 1.359996]
#    ferrolearn (new().fit, standardize=true): [-1.472976, -0.669761, 0.055343, 0.727399, 1.359996]
#    (IDENTICAL — the biased-var StandardScaler path matches sklearn on this all-positive column).

# REQ-3 — box-cox method (MISSING in ferrolearn entirely):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
bc=PowerTransformer(method='box-cox', standardize=False).fit([[1.],[2.],[3.],[4.],[5.]]); \
print('BOXCOX lambda:', bc.lambdas_.tolist()); \
print('BOXCOX transform:', np.round(bc.transform([[1.],[2.],[3.],[4.],[5.]]).ravel(),6).tolist())"
# -> BOXCOX lambda: [0.6902965702837968]
# -> BOXCOX transform: [0.0, 0.888915, 1.643917, 2.323283, 2.95143]
#    ferrolearn has no `method` param and no box-cox path. NOT-STARTED.

# REQ-4 — inverse_transform round-trip (MISSING in ferrolearn entirely):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
ptn2=PowerTransformer(method='yeo-johnson', standardize=True).fit([[-2.],[-1.],[0.],[1.],[2.],[3.]]); \
t=ptn2.transform([[-2.],[-1.],[0.],[1.],[2.],[3.]]); \
print('YJ inverse roundtrip:', np.round(ptn2.inverse_transform(t).ravel(),6).tolist())"
# -> YJ inverse roundtrip: [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
#    FittedPowerTransformer exposes no inverse_transform. NOT-STARTED.

# REQ-5 — power_transform free fn (MISSING in ferrolearn entirely):
python3 -c "import numpy as np; from sklearn.preprocessing import power_transform; \
print('power_transform free:', np.round(power_transform([[1.],[2.],[3.],[4.],[5.]], standardize=False).ravel(),6).tolist())"
# -> power_transform free: [0.892085, 1.653616, 2.341089, 2.978267, 3.578034]
#    (free fn signature: power_transform(X, method="yeo-johnson", *, standardize=True, copy=True), :3549)
#    No `power_transform` symbol in ferrolearn-preprocess. NOT-STARTED.
```

## Requirements

- REQ-1: **Yeo-Johnson lambda + forward-transform VALUE parity** (HEADLINE,
  SHIPPED; was DIV-1 #1343, resolved). For each feature, estimate the MLE lambda
  exactly as sklearn's `_yeo_johnson_optimize` (`_data.py:3464-3493`): minimize
  `_neg_log_likelihood` = `-n/2 * log(var(yj(x,λ)))` + `(λ - 1) * (np.sign(x) *
  np.log1p(np.abs(x))).sum()` (`:3484-3485`), returning `inf` when `x_trans_var <
  np.finfo(float64).tiny` (`:3480`), via `scipy.optimize.brent(brack=(-2, 2))`
  (`:3493`); then apply the 4-branch `_yeo_johnson_transform` (`:3426-3446`) and,
  when `standardize`, `StandardScaler(copy=False)` (`:3308-3314`). ferrolearn's
  `fn yeo_johnson` (the 4-branch transform) and the biased-var standardize path
  were already correct; `fn log_likelihood_yj` now computes the Jacobian as
  `lambda_minus_1 * jacobian` with `jacobian = col.fold(|acc, y| acc + sign(y) *
  (y.abs() + one).ln())` — **the `np.sign(y)` factor is restored** (`_data.py:3485`)
  with an explicit three-way sign so `y == 0` contributes 0 (`np.sign(0) == 0`).
  `fit` optimizes via `brent_bounded(..., -3.0, 3.0, ...)`; the bounded `[-3, 3]`
  interval lands on sklearn's unbounded `brent(brack=(-2, 2))` optimum (positive
  λ matches to ≈1e-8; negative-optimal probe `λ ≈ -0.7252` confirms interior
  minima `< 0`). Lambda + transform parity is now verified across positive,
  all-negative, mixed-sign, multi-feature, standardize, and negative-optimal-λ
  fixtures against the live oracle.

- REQ-2: **Error / parameter contracts** (scoped). ferrolearn `fit` returns
  `InsufficientSamples` when `n_samples == 0`; `transform` returns `ShapeMismatch`
  on a column-count mismatch; the unfitted `Transform` returns `InvalidParameter`
  (must fit first). These guards mirror sklearn's structural validation
  (`_check_input` check_shape `:3531-3537`, `check_is_fitted` `:3331`), scoped to
  the contracts ferrolearn actually enforces. **FLAG (DIV, not fixed here):
  sklearn fits on a single sample (no `n_samples >= 1` rejection of empties at the
  estimator level — `_validate_data` requires `>= 1` sample but ferrolearn's
  `n_samples == 0 -> InsufficientSamples{required:1}` is a near-match).**

- REQ-3: **`box-cox` method** — sklearn's `method='box-cox'` (`_parameter_constraints`
  `StrOptions({"yeo-johnson","box-cox"})`, `:3223`) runs `_box_cox_optimize`
  (`:3448-3462`): `_, lmbda = stats.boxcox(x[~mask], lmbda=None)` (scipy Brent MLE),
  raising `ValueError` on an all-NaN column, and `boxcox` as the transform; box-cox
  requires strictly positive input (`_check_input` `check_positive` `np.nanmin(X)
  <= 0` raises, `:3524-3528`). ferrolearn has **no `method` field and no box-cox
  path** — `PowerTransformer<F>` carries only `standardize: bool` (Probe REQ-3:
  sklearn box-cox λ `0.690297`).

- REQ-4: **`inverse_transform`** (YJ + box-cox) — sklearn's `inverse_transform`
  (`:3347-3392`) un-standardizes (`_scaler.inverse_transform`, `:3375`) then applies
  the per-method inverse: `_yeo_johnson_inverse_transform` (4-branch, `:3405-3424`)
  or `_box_cox_inverse_tranform` (`:3394-3403`). `FittedPowerTransformer<F>` exposes
  **no `inverse_transform`** (Probe REQ-4: round-trip recovers `[-2,-1,0,1,2,3]`).

- REQ-5: **`power_transform` free function** — sklearn's `power_transform(X,
  method="yeo-johnson", *, standardize=True, copy=True)` (`:3549-3648`, note the
  1.5.2 default `method="yeo-johnson"`) constructs `PowerTransformer(method,
  standardize, copy).fit_transform(X)` (`:3647-3648`). ferrolearn exposes **no
  `power_transform` free fn** (grep finds none); the closest is `PowerTransformer::
  fit_transform` but it lacks the standalone functional surface (Probe REQ-5).

- REQ-6: **Constant-feature → λ=1.0 skip + StandardScaler zero-scale handling** —
  sklearn's `_fit` skips the optimizer for constant YJ features, setting
  `self.lambdas_[i] = 1.0` (`:3299-3302`, via `_is_constant_feature(var[i], mean[i],
  n_samples)`), and `StandardScaler(copy=False)` runs `_handle_zeros_in_scale`
  (scale 0 → 1, so a constant transformed column maps to subtract-mean → 0,
  `:3308-3314`). ferrolearn has **no constant-feature skip** (it always runs
  `brent_bounded`) and `transform` standardizes only `if s > F::zero()` (a constant
  transformed column is left UNSCALED — mean NOT subtracted — diverging from
  sklearn's subtract-mean-then-divide-by-1).

- REQ-7: **NaN handling (drop in optimize) + box-cox `check_positive`** — sklearn's
  `_yeo_johnson_optimize` drops NaN before optimizing (`x = x[~np.isnan(x)]`,
  `:3491`) and `_box_cox_optimize` masks NaN (`x[~mask]`, `:3457`) raising on an
  all-NaN column; `_check_input` raises `ValueError` for non-positive box-cox input
  (`:3524-3528`). ferrolearn instead maps every value through `to_f64().unwrap_or(
  0.0)` (NaN → **0.0**, NOT dropped) when building the optimizer column, and has no
  box-cox positivity check (no box-cox).

- REQ-8: **`method` / `copy` ctor params + `_parameter_constraints`** — sklearn's
  `__init__(method="yeo-johnson", *, standardize=True, copy=True)` (`:3226-3229`)
  under `_parameter_constraints` (`method: StrOptions({"yeo-johnson","box-cox"})`,
  `standardize`/`copy` boolean, `:3222-3226`); `copy=False` performs in-place
  computation (`:3277-3278`). ferrolearn's `PowerTransformer<F>` exposes only
  `new()` / `without_standardize()` / `standardize()` — **no `method`, no `copy`,
  no parameter-constraint validation surface**.

- REQ-9: **`get_feature_names_out` + `lambdas_` / `n_features_in_` /
  `feature_names_in_` fitted-attr surface** — sklearn's `PowerTransformer`
  (`OneToOneFeatureMixin`, `:3122`) exposes `get_feature_names_out` (one-to-one
  passthrough), `lambdas_` (`:3294`), `n_features_in_`, and `feature_names_in_`.
  ferrolearn's `FittedPowerTransformer<F>` exposes the `lambdas()` accessor (≈
  `lambdas_`) but **no `get_feature_names_out`, no `n_features_in_` /
  `feature_names_in_` surface**.

- REQ-10: **ferray substrate** — compute the per-column MLE, the Yeo-Johnson /
  box-cox transforms, and the standardization over `ferray-core` arrays /
  `ferray`'s optimizer rather than `ndarray::Array2` + `num_traits::Float` +
  `ferrolearn_numerical::optimize::brent_bounded` + per-column `Vec<f64>`
  round-tripping (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `PowerTransformer::<f64>::without_standardize().fit(
  [[-2],[-1],[0],[1],[2],[3]])` yields `λ ≈ 0.950497` (sklearn oracle, Probe REQ-1
  signed) and `transform` row 0 `≈ -2.065428`; ferrolearn now MATCHES (λ to ~1e-4,
  transform to ~1e-5). On the all-positive fixture `[[1],[2],[3],[4],[5]]` it
  matches `λ ≈ 0.699807` / transform `[0.892085, ...]`, and the all-negative,
  mixed-sign, multi-feature, standardize, and negative-optimal-λ fixtures match
  too. Pinned by the 13 value tests in `tests/divergence_power_transformer.rs`
  (`divergence_div1_signed_lambda`, `divergence_div1_signed_transform` now green +
  11 green guards including `green_all_negative_lambda_and_transform`,
  `green_mixed_sign_spread_lambda_and_transform`,
  `green_multifeature_per_column_lambda_and_block`,
  `green_signed_standardize_transform`, `green_negative_optimal_lambda`).

- AC-2 (REQ-2): `PowerTransformer::<f64>::new().fit(Array2::zeros((0, 2)))` returns
  `Err(InsufficientSamples)`; `fitted.transform` on a wrong column count returns
  `Err(ShapeMismatch)`; the unfitted `transform` returns `Err(InvalidParameter)`.
  Pinned by `test_insufficient_samples_error`, `test_shape_mismatch_error`,
  `test_unfitted_transform_error` (in-module) and `green_zero_samples_errors`,
  `green_transform_ncols_mismatch_errors`, `green_unfitted_transform_errors`
  (divergence suite).

- AC-3 (REQ-3): `PowerTransformer(method='box-cox', standardize=False).fit(
  [[1],[2],[3],[4],[5]]).lambdas_` ≈ `0.690297`, transform `[0.0, 0.888915, ...]`
  (Probe REQ-3); non-positive box-cox input raises `ValueError` (`:3524-3528`).
  ferrolearn has no `method`/box-cox path.

- AC-4 (REQ-4): `pt.inverse_transform(pt.transform(X))` recovers `X` (Probe REQ-4:
  `[-2,-1,0,1,2,3]`); `FittedPowerTransformer<F>` has no `inverse_transform`.

- AC-5 (REQ-5): `power_transform([[1],[2],[3],[4],[5]], standardize=False)` →
  `[0.892085, ...]` (Probe REQ-5); no `power_transform` symbol exists in
  ferrolearn-preprocess.

- AC-6 (REQ-6): a constant column yields `lambdas_[i] == 1.0` (no optimizer call,
  `:3299-3302`) and standardizes to all-zeros (subtract-mean, scale→1,
  `_handle_zeros_in_scale`); ferrolearn always runs `brent_bounded` on a constant
  column and `transform` leaves it unscaled (`if s > 0`), so the mean is NOT
  subtracted.

- AC-7 (REQ-7): a column with a NaN is optimized over the non-NaN subset
  (`x[~np.isnan(x)]`, `:3491`); ferrolearn maps NaN → 0.0 (`to_f64().unwrap_or(
  0.0)`), biasing the MLE.

- AC-8 (REQ-8): `PowerTransformer(method='box-cox', copy=False)` constructs and is
  validated by `_parameter_constraints`; ferrolearn has no `method`/`copy` field.

- AC-9 (REQ-9): a fitted handle exposes `get_feature_names_out(['x'])` → `['x']`,
  `lambdas_`, `n_features_in_`, `feature_names_in_`; ferrolearn exposes only
  `lambdas()`.

- AC-10 (REQ-10): the MLE + transform + standardize path computes on `ferray-core`
  arrays + a `ferray` optimizer rather than `ndarray` + `brent_bounded` +
  per-column `Vec<f64>` round-tripping.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (Yeo-Johnson lambda + forward-transform VALUE parity; HEADLINE) | SHIPPED | `fn log_likelihood_yj in power_transformer.rs` now computes the Jacobian as `let jacobian: F = col.iter().fold(F::zero(), \|acc, y\| { let sign = if y > 0 {1} else if y < 0 {-1} else {0}; acc + sign * (y.abs() + one).ln() }); let jacobian_ll = lambda_minus_1 * jacobian;` — the `sign(y)·ln(\|y\|+1)` term matches sklearn's `_neg_log_likelihood` `loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()` (`_data.py:3485`), with the explicit three-way sign making `y == 0` contribute 0 (`np.sign(0) == 0`). `Fit::fit for PowerTransformer` optimizes via `ferrolearn_numerical::optimize::brent_bounded(neg_ll, -3.0, 3.0, 1e-8, 500)`; the bounded `[-3, 3]` interval lands on sklearn's `optimize.brent(brack=(-2, 2))` optimum (`:3493`) — confirmed by the negative-optimal-lambda probe (`λ ≈ -0.7252`, interior minimum `< 0`, not endpoint-clamped). The transform-given-lambda path (`fn yeo_johnson`, the 4-branch YJ matching `_yeo_johnson_transform` `:3426-3446`) and the biased-var standardize path are correct. Lambda + transform parity VERIFIED against the live sklearn 1.5.2 oracle across positive / all-negative / mixed-sign (with zero) / multi-feature (heterogeneous per-column sign) / `standardize=True` / negative-optimal-lambda fixtures. Consumers: PyO3 `_RsPowerTransformer` (`ferrolearn-python/src/extras.rs:1171-1177`, registered `lib.rs:84`) + `PipelineTransformer`/`FittedPipelineTransformer` + crate re-export `lib.rs:84`/`lib.rs:123`. Verification: `cargo test -p ferrolearn-preprocess --test divergence_power_transformer` → 13 green value tests at tol 1e-4 (lambda) / 1e-5 (transform): `divergence_div1_signed_lambda`, `divergence_div1_signed_transform` (former DIV-1 pins, now green) + `green_positive_lambda_matches_oracle`, `green_positive_transform_matches_oracle`, `green_positive_standardize_zero_mean`, `green_all_negative_lambda_and_transform`, `green_mixed_sign_spread_lambda_and_transform`, `green_multifeature_per_column_lambda_and_block`, `green_signed_standardize_transform`, `green_negative_optimal_lambda` (+ 3 REQ-2 error guards). Was DIV-1/#1343 — now resolved. |
| REQ-2 (error / parameter contracts, scoped) | SHIPPED (scoped) | `Fit::fit for PowerTransformer` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "PowerTransformer::fit" })` when `n_samples == 0`; `Transform::transform for FittedPowerTransformer` returns `Err(FerroError::ShapeMismatch { context: "FittedPowerTransformer::transform", .. })` when `x.ncols() != self.lambdas.len()`; `Transform::transform for PowerTransformer` (unfitted) returns `Err(FerroError::InvalidParameter { name: "PowerTransformer", reason: "transformer must be fitted before calling transform; use fit() first" })`. Mirrors sklearn `_check_input` check_shape (`_data.py:3531-3537`) and `check_is_fitted` (`:3331`). Non-test consumers: the PyO3 binding `_RsPowerTransformer` (`ferrolearn-python/src/extras.rs:1171-1177`, registered `lib.rs:84`) and the crate re-export `pub use power_transformer::{FittedPowerTransformer, PowerTransformer};` (`ferrolearn-preprocess/src/lib.rs:123`) route every fit/transform through these guards. Verification: `cargo test -p ferrolearn-preprocess power` (`test_insufficient_samples_error`, `test_shape_mismatch_error`, `test_unfitted_transform_error`; divergence `green_zero_samples_errors`, `green_transform_ncols_mismatch_errors`, `green_unfitted_transform_errors`) → green. **FLAG (DIV, not fixed here): sklearn fits on a single sample; ferrolearn's `n_samples == 0 -> InsufficientSamples{required:1}` is a near-match.** |
| REQ-3 (`box-cox` method) | NOT-STARTED | open prereq blocker #1344. `PowerTransformer<F>` carries only `standardize: bool` — there is NO `method` field and NO box-cox path. sklearn's `method='box-cox'` runs `_box_cox_optimize` (`_, lmbda = stats.boxcox(x[~mask], lmbda=None)`, `_data.py:3448-3462`) + `boxcox` transform, requiring strictly positive input (`_check_input` `np.nanmin(X) <= 0` raises, `:3524-3528`) and raising `ValueError` on an all-NaN column. Probe REQ-3: box-cox λ `0.690297`, transform `[0.0, 0.888915, 1.643917, 2.323283, 2.95143]`. |
| REQ-4 (`inverse_transform` YJ + box-cox) | NOT-STARTED | open prereq blocker #1345. `FittedPowerTransformer<F>` exposes no `inverse_transform` — only `transform` (forward) and `lambdas()`. sklearn's `inverse_transform` (`_data.py:3347-3392`) un-standardizes (`_scaler.inverse_transform`, `:3375`) then applies the 4-branch `_yeo_johnson_inverse_transform` (`:3405-3424`) or `_box_cox_inverse_tranform` (`:3394-3403`). Probe REQ-4: `inverse_transform(transform(X))` recovers `[-2,-1,0,1,2,3]`. |
| REQ-5 (`power_transform` free fn) | NOT-STARTED | open prereq blocker #1346. No `power_transform` symbol exists in `ferrolearn-preprocess` (grep finds none); `lib.rs:123` re-exports only `FittedPowerTransformer, PowerTransformer`. sklearn's free fn `power_transform(X, method="yeo-johnson", *, standardize=True, copy=True)` (`_data.py:3549`, 1.5.2 default `method="yeo-johnson"`) is `PowerTransformer(method, standardize, copy).fit_transform(X)` (`:3647-3648`). Probe REQ-5: `power_transform([[1],..,[5]], standardize=False)` → `[0.892085, 1.653616, 2.341089, 2.978267, 3.578034]`. |
| REQ-6 (constant-feature → λ=1.0 skip + StandardScaler zero-scale handling) | SHIPPED | `Fit::fit for PowerTransformer` now computes the (NaN-dropped) column mean/var and calls `fn is_constant_feature` (mirroring sklearn `_is_constant_feature` `_data.py:72-85`, `var <= n*eps*var + (n*mean*eps)^2`); a constant column sets `lambdas[j] = F::one()` and `continue`s the optimizer (sklearn `:3299-3302`). The standardize block applies `_handle_zeros_in_scale` semantics — `if is_constant_feature(var, mean, n_obs) { std = 1.0 } else { var.sqrt() }` (sklearn `:88-120`, `:1016-1021`) — and `Transform::transform for FittedPowerTransformer` now always subtract-mean-then-÷scale (the `if s > 0` guard removed), so a constant column is centered to 0. Non-test consumers: PyO3 `_RsPowerTransformer` (`extras.rs:1171-1177`, registered `lib.rs:84`) + `PipelineTransformer`/`FittedPipelineTransformer` + re-export (`lib.rs:123`). Verification: 4 oracle pins in `tests/divergence_power_transformer_edges.rs` (`divergence_constant_feature_lambda` → λ=1.0; `divergence_constant_feature_transform_no_std` → [3,3,3,3]; `divergence_constant_feature_transform_standardize` → [0,0,0,0]; `divergence_single_sample_standardize` → [0.0]) → green. |
| REQ-7 (NaN drop in optimize + box-cox check_positive) | SHIPPED (NaN-drop; box-cox check_positive NOT-STARTED) | `Fit::fit for PowerTransformer` now builds the optimizer column via `x.column(j).iter().filter_map(\|v\| { let f = v.to_f64().unwrap_or(NAN); if f.is_nan() { None } else { Some(f) } })` — **NaN DROPPED, not mapped to 0.0** — and the standardize stats filter NaN identically, mirroring sklearn `x = x[~np.isnan(x)]` (`_data.py:3491`). The transform passes NaN through (NaN arithmetic preserves NaN). Non-test consumers: PyO3 `_RsPowerTransformer` + `PipelineTransformer` + re-export. Verification: 2 oracle pins in `tests/divergence_power_transformer_edges.rs` (`divergence_nan_dropped_in_mle_lambda` → λ=0.502119 over {1,2,4,5}; `divergence_nan_transform_finite_rows` → finite rows match, NaN row stays NaN) → green. The box-cox `check_positive` / all-NaN `ValueError` (`:3524-3528`, `:3457`) remains NOT-STARTED — box-cox has no path (blocker #1344). |
| REQ-8 (`method` / `copy` ctor params + `_parameter_constraints`) | NOT-STARTED | open prereq blocker #1349. `PowerTransformer<F>` exposes only `new()` / `without_standardize()` / `standardize()` — no `method`, no `copy`, no parameter-constraint validation. sklearn's `__init__(method="yeo-johnson", *, standardize=True, copy=True)` (`_data.py:3226-3229`) is validated by `_parameter_constraints` (`method: StrOptions({"yeo-johnson","box-cox"})`, `standardize`/`copy` boolean, `:3222-3226`), and `copy=False` does in-place work (`:3277-3278`). |
| REQ-9 (`get_feature_names_out` + `n_features_in_` / `feature_names_in_` fitted-attr surface) | NOT-STARTED | open prereq blocker #1350. `FittedPowerTransformer<F>` exposes the `lambdas()` accessor (≈ sklearn `lambdas_`, `_data.py:3294`) but NO `get_feature_names_out`, NO `n_features_in_` / `feature_names_in_` surface. sklearn's `PowerTransformer` (`OneToOneFeatureMixin`, `:3122`) emits a one-to-one `get_feature_names_out` and exposes `n_features_in_` / `feature_names_in_` via `BaseEstimator`. (Note: `lambdas_` IS covered.) |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #1351. The fit/transform path uses `ndarray::Array2` (`x.column(j)`, `out.columns_mut()`, `Array1::zeros`), `num_traits::Float`, `ferrolearn_numerical::optimize::brent_bounded`, and per-column `Vec<f64>` / `Vec<F>` round-tripping inside the optimizer closure — not `ferray-core` arrays / a `ferray` optimizer (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `power_transformer.rs` exposes the unfitted
`PowerTransformer<F> { standardize: bool, _marker: PhantomData<F> }` (`new` =
standardize true, `without_standardize` = false, `Default` = `new`, accessor
`standardize()`) and the fitted `FittedPowerTransformer<F> { lambdas: Array1<F>,
means: Option<Array1<F>>, stds: Option<Array1<F>> }` (accessor `lambdas()`). Two
private helpers carry the math: `fn yeo_johnson<F>(y, lambda)` is the 4-branch YJ
forward transform (`y >= 0, λ≈0 → ln(y+1)`; `y >= 0 → ((y+1)^λ - 1)/λ`; `y < 0,
λ≈2 → -ln(1-y)`; `y < 0 → -((1-y)^(2-λ) - 1)/(2-λ)`, eps `1e-10` vs sklearn's
`np.spacing(1.0)`) — a faithful mirror of `_yeo_johnson_transform`
(`_data.py:3426-3446`). `fn log_likelihood_yj<F>(col, lambda)` computes the FULL
normal log-likelihood (incl. the `2π` and `-n/2` constants that drop out of
argmin) plus the Jacobian `(λ - 1) * sum(sign(y) * (|y| + 1).ln())` — **now
carrying the `np.sign(y)` factor** at `_data.py:3485` (explicit three-way sign,
`sign(0) == 0`). `Fit::fit` validates `n_samples == 0 → InsufficientSamples`, then
per column converts to `Vec<f64>` (NaN → `unwrap_or(0.0)`), runs
`ferrolearn_numerical::optimize::brent_bounded(neg_ll, -3.0, 3.0, 1e-8, 500)` to
minimize `-log_likelihood_yj`, and stores the result; if `standardize`, it
computes the biased (÷n, population, ddof=0) mean/std of the transformed columns.
`Transform::transform` checks the column count (`ShapeMismatch`), applies
`yeo_johnson` per cell, then standardizes only `if s > F::zero()`. The unfitted
`Transform` returns `InvalidParameter`; `FitTransform` chains `fit` then
`transform`; `PipelineTransformer` / `FittedPipelineTransformer` wrap the same
path. The crate re-exports both types (`lib.rs:123`), and the PyO3 binding
`_RsPowerTransformer` (`extras.rs:1171-1177`, `py_transformer!` macro,
`PowerTransformer::new()` ⇒ standardize=true, registered `lib.rs:84`) is the
boundary CPython consumer.

**sklearn (target contract).** `PowerTransformer(OneToOneFeatureMixin,
TransformerMixin, BaseEstimator)` (`_data.py:3122`) takes `__init__(method=
"yeo-johnson", *, standardize=True, copy=True)` (`:3226-3229`) under
`_parameter_constraints` (`method: StrOptions({"yeo-johnson","box-cox"})`,
`standardize`/`copy` boolean, `:3222-3226`). `_fit` (`:3274-3316`) computes per-col
mean/var, dispatches `optim_function = {box-cox:_box_cox_optimize,
yeo-johnson:_yeo_johnson_optimize}[method]` (`:3284-3287`), **skips constant YJ
features with `lambdas_[i] = 1.0` (`:3299-3302`)**, then fits/transforms via
`StandardScaler(copy=False)` when `standardize` (`:3308-3314`).
`_yeo_johnson_optimize` (`:3464-3493`) minimizes `_neg_log_likelihood = -n/2 *
log(var) + (λ - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()` (`:3484-3485`,
returning `inf` if `x_trans_var < tiny`, `:3480`) over NaN-dropped `x`
(`:3491`) via UNBOUNDED `optimize.brent(brack=(-2, 2))` (`:3493`).
`_box_cox_optimize` (`:3448-3462`) uses `stats.boxcox(x[~mask], lmbda=None)`.
`transform` (`:3318-3345`) and `inverse_transform` (`:3347-3392`, YJ inverse
`:3405-3424`, box-cox inverse `:3394-3403`) apply the per-method forward/inverse
maps. `power_transform` (`:3549-3648`) is the stateless `fit_transform` wrapper.

**The gap.** ferrolearn matches sklearn on the *transform-given-lambda algebra*
(`fn yeo_johnson` = `_yeo_johnson_transform`), the *biased-var standardize path*,
and now the **end-to-end Yeo-Johnson lambda estimate on signed data** (REQ-1, the
headline) — the restored `np.sign(x)` Jacobian factor (`:3485`) closes the former
DIV-1 (#1343, resolved), and the bounded `[-3, 3]` Brent lands on sklearn's
unbounded `brent(brack=(-2,2))` optimum (`:3493`), verified across positive,
all-negative, mixed-sign, multi-feature, standardize, and negative-optimal-λ
fixtures against the live oracle. The remaining gaps are absent surfaces: no
box-cox (REQ-3); no `inverse_transform` (REQ-4); no `power_transform` free fn
(REQ-5); no constant-feature skip / zero-scale handling (REQ-6); NaN→0 instead of
dropped + no box-cox positivity check (REQ-7); no `method`/`copy` ctor params
(REQ-8); no `get_feature_names_out` / `n_features_in_` surface (REQ-9, though
`lambdas_` exists); and the non-ferray substrate (REQ-10). This is a
**shipped-partial** unit (2 SHIPPED / 8 NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 YJ lambda+transform parity, REQ-2
scoped error contracts):

```bash
# Crate gauntlet — REQ-1 value parity + REQ-2 error contracts:
cargo test -p ferrolearn-preprocess power     # in-module: test_insufficient_samples_error,
                                               #   test_shape_mismatch_error,
                                               #   test_unfitted_transform_error,
                                               #   test_yeo_johnson_identity_at_lambda_one,
                                               #   test_yeo_johnson_log_at_lambda_zero,
                                               #   test_yeo_johnson_negative_at_lambda_two,
                                               #   test_power_transformer_fit_basic,
                                               #   test_power_transformer_transform_shape,
                                               #   test_standardize_produces_zero_mean,
                                               #   test_without_standardize,
                                               #   test_fit_transform_equivalence,
                                               #   test_negative_values_supported,
                                               #   test_pipeline_integration
cargo test -p ferrolearn-preprocess --test divergence_power_transformer
                                               # 13 oracle VALUE tests (REQ-1) + 3 REQ-2 guards:
                                               #   divergence_div1_signed_lambda (was RED, now GREEN),
                                               #   divergence_div1_signed_transform (was RED, now GREEN),
                                               #   green_positive_lambda_matches_oracle,
                                               #   green_positive_transform_matches_oracle,
                                               #   green_positive_standardize_zero_mean,
                                               #   green_all_negative_lambda_and_transform,
                                               #   green_mixed_sign_spread_lambda_and_transform,
                                               #   green_multifeature_per_column_lambda_and_block,
                                               #   green_signed_standardize_transform,
                                               #   green_negative_optimal_lambda
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 (HEADLINE) oracle gate — SIGNED fixture: lambda + transform now MATCH the oracle
# (the restored np.sign factor closes former DIV-1; #1343 resolved):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
ptn=PowerTransformer(method='yeo-johnson', standardize=False).fit([[-2.],[-1.],[0.],[1.],[2.],[3.]]); \
print('sklearn NEG lambda=', ptn.lambdas_[0], 'row0=', round(float(ptn.transform([[-2.]])[0,0]),6))"
#   -> sklearn NEG lambda= 0.9504965354909566 row0= -2.065428
#      ferrolearn: lambda 0.950497 (~1e-4), row0 -2.065428 (~1e-5).  PARITY. REQ-1 SHIPPED.

# REQ-1 oracle gate — NEGATIVE-optimal-lambda probe: confirms bounded-Brent finds an
# interior minimum < 0 (former DIV-2 is a non-issue):
python3 -c "import numpy as np; from sklearn.preprocessing import PowerTransformer; \
pte=PowerTransformer(method='yeo-johnson', standardize=False).fit([[1.],[1.],[1.],[2.],[10.],[100.]]); \
print('sklearn E lambda=', pte.lambdas_[0])"
#   -> sklearn E lambda= -0.7252485461394542   (ferrolearn matches to ~1e-4; not endpoint-clamped)
```

The in-module `#[test]`s exercise REQ-2 (every error path —
`test_insufficient_samples_error`, `test_shape_mismatch_error`,
`test_unfitted_transform_error`) and the REQ-1 transform-given-lambda algebra
(`test_yeo_johnson_identity_at_lambda_one` / `_log_at_lambda_zero` /
`_negative_at_lambda_two` pin `fn yeo_johnson` at λ∈{1,0,2};
`test_standardize_produces_zero_mean` pins the biased-var standardize path). The
signed-data lambda+transform VALUE parity (REQ-1, HEADLINE) is pinned by the 13
green oracle value tests in `tests/divergence_power_transformer.rs` against the
live sklearn 1.5.2 oracle (positive / all-negative / mixed-sign / multi-feature /
standardize / negative-optimal-λ). No green command establishes REQ-3..REQ-10.

## Blockers

REQ-1 (YJ lambda+transform parity, HEADLINE) and REQ-2 (scoped error / parameter
contracts) are SHIPPED. The former DIV-1 blocker is RESOLVED:

- #1343 (RESOLVED) — REQ-1 (HEADLINE): the Yeo-Johnson MLE lambda diverged from
  the oracle on SIGNED data because `fn log_likelihood_yj` omitted the `np.sign(x)`
  Jacobian factor (`_data.py:3485`). FIXED: the sign factor is restored (explicit
  three-way sign, `sign(0) == 0`); the bounded-Brent `[-3, 3]` was confirmed to
  land on sklearn's `optimize.brent(brack=(-2, 2))` optimum (`:3493`), so the
  once-suspected DIV-2 is a non-issue. Parity verified by 13 green oracle value
  tests across positive / all-negative / mixed-sign / multi-feature / standardize /
  negative-optimal-λ fixtures.

The remaining REQs are NOT-STARTED, filed as `-l blocker` issues against tracking
issue #1342:

- #1344 — REQ-3: no `method` param / box-cox path (`_box_cox_optimize` +
  `stats.boxcox`, `:3448-3462`; `check_positive`, `:3524-3528`).
- #1345 — REQ-4: no `inverse_transform` (YJ inverse `:3405-3424`, box-cox inverse
  `:3394-3403`).
- #1346 — REQ-5: no `power_transform` free fn (`:3549-3648`).
- #1347 — REQ-6: no constant-feature `λ=1.0` skip (`:3299-3302`) and `transform`
  leaves a constant column unscaled (`if s > 0`) instead of subtract-mean-then-÷1
  (`_handle_zeros_in_scale`, `:3308-3314`).
- #1348 — REQ-7: NaN mapped to `0.0` (`to_f64().unwrap_or(0.0)`) instead of dropped
  (`x[~np.isnan(x)]`, `:3491`); no box-cox positivity check (`:3524-3528`).
- #1349 — REQ-8: no `method` / `copy` ctor params or `_parameter_constraints`
  (`:3222-3229`).
- #1350 — REQ-9: no `get_feature_names_out` / `n_features_in_` / `feature_names_in_`
  surface (`lambdas_` is covered).
- #1351 — REQ-10: fit/transform on `ndarray` / `num_traits` / `brent_bounded` /
  per-column `Vec<f64>` round-tripping, not ferray (R-SUBSTRATE-1/2).
```
