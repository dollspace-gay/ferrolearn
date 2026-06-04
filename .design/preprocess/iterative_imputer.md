# IterativeImputer

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 7d819dce
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/impute/_iterative.py  # class IterativeImputer(_BaseImputer) (:51). EXPERIMENTAL — requires `from sklearn.experimental import enable_iterative_imputer` (:63-70). __init__(estimator=None, *, missing_values=np.nan, sample_posterior=False, max_iter=10, tol=1e-3, n_nearest_features=None, initial_strategy="mean", fill_value=None, imputation_order="ascending", skip_complete=False, min_value=-inf, max_value=inf, verbose=0, random_state=None, add_indicator=False, keep_empty_features=False) (:305-343). estimator=None -> BayesianRidge() (:74,732-735). _impute_one_feature (:345-466): X_train = neighbor cols on ~missing rows (:408-412), y_train = feat_idx on ~missing rows (:413-417), estimator.fit (:418); imputed_values = estimator.predict(X_test) (:454); imputed_values = np.clip(imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]) (:455-457); _safe_assign updates the feature (:460-465). _get_neighbor_feat_idx (:468-502): if n_nearest_features < n_features choose by abs-corr probability (:493-497) else all other features (:499-501). _get_ordered_idx (:504-542): frac_of_missing = mask.mean(axis=0) (:524); skip_complete -> flatnonzero else arange (:525-528); 'roman' = missing_values_idx i.e. column order (:529-530); 'arabic' = reversed (:531-532); 'ascending' = argsort(frac, kind="mergesort")[n:] fewest-missing first (:533-535); 'descending' = reversed ascending (:536-538); 'random' = shuffle (:539-541). fit_transform (:732-829): estimator=None -> BayesianRidge (:732-735); imputation_sequence_=[] (:739); _initial_imputation via SimpleImputer(strategy=initial_strategy) (:743); max_iter==0 or all-missing -> n_iter_=0 return (:750-752); single feature -> n_iter_=0 return initial (:755-757); _validate_limit min/max (:759-760); ordered_idx = _get_ordered_idx (:769); abs_corr_mat = _get_abs_corr_mat(Xt) (:772); normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values])) (:780); round-robin loop range(1, max_iter+1) over ordered_idx (:781-801); inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf) (:811); if inf_norm < normalized_tol break (:818-821); else ConvergenceWarning (:823-828). sample_posterior path samples truncated normal (:430-452).
ferrolearn-module: ferrolearn-preprocess/src/iterative_imputer.rs
parity-ops: IterativeImputer
crosslink-issue: 1403
-->

## Summary

scikit-learn's `IterativeImputer` (`_iterative.py:51`, EXPERIMENTAL) imputes missing
values by modeling each feature with missing values as a regression on all the other
features, in a **round-robin** fashion repeated until convergence or `max_iter`
(`fit_transform:781-829`). The default per-feature estimator is `BayesianRidge`
(`:74,732-735`), features are visited in `imputation_order="ascending"` (fewest-missing
first, `_get_ordered_idx:533-535`), imputed values are clipped to `[min_value,
max_value]` (`_impute_one_feature:455-457`), and convergence is the **inf-norm** of the
change scaled by `tol * max|X_observed|` (`:780,811,818`).

`ferrolearn-preprocess/src/iterative_imputer.rs` ships the **round-robin regression
structure plus the initial mean/median fill**: `IterativeImputer<F> { max_iter, tol,
initial_strategy }` with `InitialStrategy::{Mean, Median}` (`new`, accessors `max_iter`,
`tol`, `initial_strategy`; `Default` = `(10, 1e-3, Mean)`) produces
`FittedIterativeImputer<F> { initial_fill, feature_models, missing_features, n_iter,
max_iter, tol, initial_strategy }` (accessors `n_iter()`, `initial_fill()`,
`initial_strategy()`). `Fit::fit` (`fn fit in iterative_imputer.rs`) computes the initial
column fill (`column_means_nan` / `column_medians_nan`), then for each feature with
missing values — in **column-index order** — fits a closed-form **Ridge regression with
`alpha=1`** (`ridge_fit`) on the other features over the non-missing rows and predicts the
missing entries, looping until an **L2-relative** convergence test
(`(total_change/total_value).sqrt() < tol`) or `max_iter`. `Transform::transform`
(`fn transform in iterative_imputer.rs`) re-applies the stored per-feature models.

This unit is an **honest verify-and-document** unit. ferrolearn implements the round-robin
*structure* and the *initial-fill values* (which match sklearn's `SimpleImputer`-backed
initialization), but the **exact imputed VALUES diverge** from sklearn because four
algorithm choices differ: the per-feature **estimator** (Ridge `alpha=1` vs sklearn default
`BayesianRidge`), the feature **order** (column-index == sklearn `'roman'` vs the
`'ascending'` default), the **convergence** criterion (L2-relative vs sklearn inf-norm),
and **clipping** (none vs sklearn `min_value`/`max_value`). Exact-value parity is a
**NOT-STARTED CARVE-OUT** (REQ-4), an algorithm divergence gated on a `BayesianRidge`
estimator, not a minimal fix — analogous to the bayesian-GMM heuristic-vs-VB carve-out
(#1067). The SHIPPED claims are STRUCTURAL/contract plus the initial-fill values. This is
a **shipped-partial** unit: **3 SHIPPED** (REQ-1 round-robin structure + initial-fill
values + non-missing-preserved + output shape; REQ-2 determinism + termination + `n_iter`;
REQ-3 error / parameter contracts + `max_iter==0` initial-fill parity) / **11 NOT-STARTED**
(REQ-4 exact imputed-value parity carve-out, REQ-5 `estimator` / `sample_posterior`, REQ-6
`imputation_order`, REQ-7 `min_value`/`max_value` clipping, REQ-8 `n_nearest_features`,
REQ-9 `initial_strategy` most_frequent/constant + `fill_value` + `missing_values`, REQ-10
`random_state` / `skip_complete` / `add_indicator` / `keep_empty_features` / `verbose`,
REQ-11 inf-norm convergence, REQ-12 fitted-attribute / `get_feature_names_out` surface,
REQ-13 PyO3 binding, REQ-14 ferray substrate).

## Probes (live sklearn oracle, 1.5.2)

All values below are live output from `python3` against scikit-learn 1.5.2, run from
`/tmp` (R-CHAR-3), with `from sklearn.experimental import enable_iterative_imputer`
(EXPERIMENTAL gate, `_iterative.py:63-70`). They pin the SHIPPED initial-fill + structural
claims (REQ-1/2/3) and **anchor the value carve-out** (REQ-4) — Probe B is a divergence
demonstration, NOT a parity claim.

```bash
# PROBE A (REQ-1) — the INITIAL fill equals SimpleImputer(strategy=initial_strategy).
# IterativeImputer initializes via SimpleImputer (_initial_imputation, _iterative.py:743):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
print('mean stats=', SimpleImputer(strategy='mean').fit(X).statistics_.tolist())
print('median stats=', SimpleImputer(strategy='median').fit(X).statistics_.tolist())"
#   -> mean stats= [2.0, 4.0]   median stats= [2.0, 4.0]
#   col0 observed {1,3} -> mean 2.0 / median 2.0; col1 observed {2,6} -> mean 4.0 / median 4.0.
#   ferrolearn column_means_nan([[1,2],[3,NaN],[NaN,6]]) = [2.0, 4.0],
#   column_medians_nan(same) = [2.0, 4.0]  (IDENTICAL initial fill — REQ-1).

# PROBE B (REQ-4 CARVE-OUT) — EXACT imputed VALUES diverge: sklearn default estimator is
# BayesianRidge (_iterative.py:74,732-735), NOT ferrolearn's Ridge(alpha=1). This is a
# DIVERGENCE DEMONSTRATION anchoring the carve-out, NOT a parity assertion:
python3 -c "import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
ii=IterativeImputer(max_iter=10, tol=1e-3, random_state=0)
out=ii.fit_transform(X)
print('default estimator =', type(ii._estimator).__name__)
print('sklearn out =', out.tolist())
print('n_iter_ =', ii.n_iter_)"
#   -> default estimator = BayesianRidge
#   -> sklearn out = [[1.0, 2.0], [3.0, 4.000002999996018], [4.999994000015464, 6.0]]
#   -> n_iter_ = 2
#   ferrolearn (Ridge alpha=1, column-order, L2-tol, no clip) will produce DIFFERENT
#   imputed values for [1,1] and [2,0] — exact-value parity is NOT claimed (REQ-4 carve-out).

# PROBE C (REQ-1) — output shape is preserved (n_samples, n_features); no column drop:
python3 -c "import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
ii=IterativeImputer(max_iter=10, tol=1e-3, random_state=0)
print('out.shape =', ii.fit_transform(X).shape, 'n_features_in_ =', ii.n_features_in_)"
#   -> out.shape = (3, 2)   n_features_in_ = 2
#   ferrolearn transform returns an (n_samples, n_features) Array2 with NaNs filled (REQ-1).

# PROBE D (REQ-6) — imputation_order='ascending' (fewest-missing first) DIFFERS from
# ferrolearn's column-index order (== sklearn 'roman') when missing fractions differ
# (_get_ordered_idx:524,529-535):
python3 -c "import numpy as np
X=np.array([[1.,2.,3.],[np.nan,np.nan,6.],[7.,np.nan,9.],[10.,11.,np.nan]])
mask=np.isnan(X); frac=mask.mean(axis=0)
print('frac_of_missing =', frac.tolist())
print('ascending (default) =', np.argsort(frac, kind='mergesort').tolist())
print('roman == column order =', list(range(3)))"
#   -> frac_of_missing = [0.25, 0.5, 0.25]
#   -> ascending (default) = [0, 2, 1]   roman == column order = [0, 1, 2]
#   ferrolearn visits missing_features in COLUMN ORDER [0,1,2] (== 'roman'), NOT the
#   'ascending' default [0,2,1] (REQ-6 NOT-STARTED).

# PROBE E (REQ-7) — min_value/max_value clipping changes imputed values
# (_impute_one_feature:455-457). ferrolearn does NO clipping:
python3 -c "import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.],[5.,8.]])
unclipped=IterativeImputer(random_state=0).fit_transform(X)
clipped=IterativeImputer(random_state=0, min_value=0, max_value=5).fit_transform(X)
print('unclipped =', unclipped.tolist())
print('clipped[0,5] =', clipped.tolist())"
#   -> unclipped   = [[1.0, 2.0], [3.0, 5.000437960566561], [3.6657584513916404, 6.0], [5.0, 8.0]]
#   -> clipped[0,5] = [[1.0, 2.0], [3.0, 5.0], [3.6666665833336247, 6.0], [5.0, 8.0]]
#   the [1,1] imputation is clamped from 5.0004... to 5.0 — ferrolearn has no min/max clip (REQ-7).
```

## Requirements

- REQ-1: **Round-robin imputation structure + initial-fill values + non-missing
  preservation + output shape** (HEADLINE, SHIPPED scoped). Mirror sklearn's overall
  contract: initialize each missing column with `SimpleImputer(strategy=initial_strategy)`
  (`_iterative.py:743`); then for each feature with missing values, fit a regression on the
  other features over the non-missing rows (`_impute_one_feature:408-418`) and predict the
  missing entries (`:454,460-465`); repeat per round; return an `(n_samples, n_features)`
  matrix with **observed values preserved** (`_assign_where(Xt, X, cond=~mask):829`) and
  every missing entry filled. ferrolearn's `fn fit in iterative_imputer.rs` computes the
  initial fill via `column_means_nan` / `column_medians_nan` — matching `SimpleImputer`'s
  mean/median (Probe A: `[2.0, 4.0]`) — builds `missing_features`, runs the round-robin
  `ridge_fit`/`ridge_predict` loop over the missing features, and stores per-feature
  `FeatureModel { coefficients, intercept }`; `fn transform in iterative_imputer.rs`
  re-applies the stored models and returns an `(n_samples, n_features)` `Array2<F>` with
  NaNs filled and observed values untouched (`initial_fill` only overwrites `v.is_nan()`).
  Scoped: this is the round-robin STRUCTURE + the initial-fill VALUES + the output-shape
  contract; it is explicitly **NOT** exact imputed-value parity — see REQ-4 (Probe B).
  Pinned by `test_iterative_imputer_basic`, `test_iterative_imputer_no_missing` (observed
  values preserved), `test_iterative_imputer_median_strategy`,
  `test_iterative_imputer_fit_transform`, `test_iterative_imputer_f32` (in-module).

- REQ-2: **Determinism + termination + `n_iter` accessor** (SHIPPED scoped). The fit/transform
  path is deterministic — there is **no RNG** in ferrolearn's code path (no `random_state`),
  so repeated fits on the same input yield identical models. The round-robin loop runs at most
  `max_iter` rounds (`for iter_idx in 0..self.max_iter`) and breaks early on the convergence
  test, mirroring sklearn's bounded `for self.n_iter_ in range(1, self.max_iter + 1)` with an
  early-stop break (`_iterative.py:781,818-821`). `FittedIterativeImputer::n_iter()` exposes the
  number of rounds performed (sklearn's `n_iter_`, `:781`). NOTE: the *convergence criterion
  itself* (L2-relative vs sklearn inf-norm) diverges — that is REQ-11; REQ-2 covers only that
  termination is bounded by `max_iter`, early-stops on a tol test, and reports `n_iter`. Pinned
  by `test_iterative_imputer_convergence`, `test_iterative_imputer_n_iter_accessor` (`0 <
  n_iter() <= 10`), `test_iterative_imputer_default` (`max_iter()==10`).

- REQ-3: **Error / parameter contracts + `max_iter==0` initial-fill parity** (scoped, SHIPPED).
  `fn fit in iterative_imputer.rs` returns `InsufficientSamples { required: 1, actual: 0, .. }`
  when `n_samples == 0`. `max_iter == 0` is **ACCEPTED**: the round-robin loop (`for iter_idx in
  0..self.max_iter`) runs zero times, so `fit` returns the initial mean/median fill with
  `n_iter() == 0` — matching sklearn, which sets `n_iter_ = 0` and returns the
  `SimpleImputer(strategy=initial_strategy)` initial imputation for `max_iter==0` (`:750-752`).
  This was the #1404 divergence (ferrolearn previously rejected `max_iter==0` with
  `InvalidParameter`); the rejection guard was REMOVED and the behavior now matches sklearn —
  **RESOLVED**. `fn transform in iterative_imputer.rs` returns `ShapeMismatch` when `x.ncols()`
  differs from the fitted feature count. The unfitted `Transform for IterativeImputer in
  iterative_imputer.rs` returns `InvalidParameter` (the "must fit first" guard, mirroring
  sklearn's `check_is_fitted`). Scoped to the contracts ferrolearn actually enforces over the
  dense NaN-only API. Pinned by `test_iterative_imputer_zero_rows_error`,
  `test_iterative_imputer_zero_max_iter_returns_initial_fill` (in-module),
  `divergence_max_iter_zero_returns_initial_fill` (`tests/divergence_iterative_imputer.rs`),
  `test_iterative_imputer_shape_mismatch_error`, `test_iterative_imputer_unfitted_transform_error`.

- REQ-4: **Exact imputed-VALUE parity** (NOT-STARTED CARVE-OUT). Reproduce sklearn's exact
  imputed values, which requires the full default algorithm: per-feature **`BayesianRidge`**
  (`_iterative.py:74,732-735`) — NOT ferrolearn's closed-form `ridge_fit(alpha=1)` — plus
  **`imputation_order="ascending"`** (REQ-6), the **inf-norm convergence** criterion (REQ-11),
  and **`min_value`/`max_value` clipping** (REQ-7). ferrolearn diverges on all four axes
  simultaneously: `ridge_fit` solves `(XᵀX + I)⁻¹Xᵀy` (Ridge `alpha=1`) where sklearn fits a
  Bayesian-Ridge posterior; it visits `missing_features` in **column order** (== sklearn
  `'roman'`, NOT `'ascending'`, Probe D); it converges on `(total_change/total_value).sqrt() <
  tol` (L2-relative) where sklearn uses `inf_norm < tol * max|X_observed|` (Probe E shows
  clipping further perturbs values). This is an **algorithm divergence** in the same class as
  the bayesian-GMM heuristic-vs-VB carve-out (#1067) — it is **gated on a `BayesianRidge`
  estimator** existing in `ferrolearn-linear` and on REQ-6/REQ-7/REQ-11 landing, and is **NOT**
  minimally fixable. Per R-DEFER-3 **no failing parity test is committed** (closing the divergence
  requires the prerequisite `BayesianRidge` + order + clip + inf-norm to land together). Probe B
  anchors the divergence. Open prereq blocker #1405.

- REQ-5: **`estimator` parameter (pluggable; default `BayesianRidge`) + `sample_posterior`**
  (NOT-STARTED). sklearn's ctor accepts `estimator=None` → `BayesianRidge()`
  (`_iterative.py:74,307,732-735`), clones it per feature (`_impute_one_feature:404`), and
  `sample_posterior=True` (`:85-89,310`) draws from a truncated-normal posterior
  (`:430-452`). ferrolearn hardwires its internal closed-form `ridge_fit(alpha=1)` — there is
  **no `estimator` field, no pluggable estimator trait object, and no `sample_posterior`**.
  Open prereq blocker #1406.

- REQ-6: **`imputation_order` parameter** (NOT-STARTED). sklearn's `imputation_order="ascending"`
  (`:316`) orders features fewest-missing-first via `argsort(frac, kind="mergesort")[n:]`
  (`_get_ordered_idx:533-535`), with `'descending'`/`'roman'`/`'arabic'`/`'random'` variants
  (`:529-541`). ferrolearn visits `missing_features` in **column-index order** (the order they
  are discovered in `for j in 0..n_features`, == sklearn `'roman'`, `:529-530`) — there is **no
  `imputation_order` param**, so it cannot select the `'ascending'` default (Probe D:
  ascending `[0,2,1]` vs ferrolearn `[0,1,2]`). Open prereq blocker #1407.

- REQ-7: **`min_value` / `max_value` clipping** (NOT-STARTED). sklearn clips every imputed value
  to `[min_value, max_value]` via `np.clip(imputed_values, self._min_value[feat_idx],
  self._max_value[feat_idx])` (`_impute_one_feature:455-457`), defaulting to `-inf`/`+inf`
  (`:318-319`) and supporting per-feature array-likes (`_validate_limit:759-760`). ferrolearn
  stores `ridge_predict` output **unclipped** (`imputed[[i, j]] = predictions[row_idx]`) — there
  are **no `min_value`/`max_value` fields and no clip** (Probe E: `5.0004...` not clamped to
  `5.0`). Open prereq blocker #1408.

- REQ-8: **`n_nearest_features` + abs-correlation feature selection** (NOT-STARTED). sklearn's
  `n_nearest_features=None` (`:313`); when set below `n_features`, each feature is predicted from
  a random subsample of the others chosen with probability proportional to absolute correlation
  (`_get_neighbor_feat_idx:493-497`, `_get_abs_corr_mat:544`). ferrolearn always uses **all other
  features** as predictors (`predictor_cols = (0..n_features).filter(|&k| k != j)`) — there is
  **no `n_nearest_features` param and no correlation-based selection**. Open prereq blocker #1409.

- REQ-9: **`initial_strategy` most_frequent/constant + `fill_value` + non-NaN `missing_values`**
  (NOT-STARTED). sklearn's `initial_strategy` accepts `"mean"|"median"|"most_frequent"|"constant"`
  (`:314`, fed to `SimpleImputer`), `fill_value=None` (`:315`) supplies the constant, and
  `missing_values=np.nan` (`:309`) may be any scalar sentinel (`_get_mask`). ferrolearn's
  `InitialStrategy` enum has only `Mean`/`Median` (no `MostFrequent`, no `Constant`), exposes no
  `fill_value`, and detects missingness **only** via `v.is_nan()` (`fn fit`/`fn transform in
  iterative_imputer.rs`). Open prereq blocker #1410.

- REQ-10: **`random_state` + `skip_complete` + `add_indicator` + `keep_empty_features` +
  `verbose`** (NOT-STARTED). sklearn's ctor accepts `random_state=None` (`:321`, seeds the
  posterior sampling and random order), `skip_complete=False` (`:317`, omit complete features from
  the order, `_get_ordered_idx:525-526`), `add_indicator=False` (`:322`, append a missing mask),
  `keep_empty_features=False` (`:323`), and `verbose=0` (`:320`). ferrolearn exposes **none** of
  these. Open prereq blocker #1411.

- REQ-11: **inf-norm convergence (`tol * max|X_observed|`)** (NOT-STARTED). sklearn's early-stop is
  `inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf)` compared against `normalized_tol =
  self.tol * np.max(np.abs(X[~mask_missing_values]))` (`_iterative.py:780,811,818`), emitting a
  `ConvergenceWarning` if never reached (`:823-828`). ferrolearn uses an **L2-relative** test
  `(total_change/total_value).sqrt() < self.tol` (`fn fit in iterative_imputer.rs`) — a different
  metric, un-scaled by `max|X_observed|`, and no `ConvergenceWarning`. Distinct from REQ-4 (it
  also folds into the value carve-out) but pinned separately. Open prereq blocker #1412.

- REQ-12: **`get_feature_names_out` + `imputation_sequence_` / `n_iter_` (name) /
  `n_features_in_` / `random_state_` fitted-attribute surface** (NOT-STARTED). sklearn's
  `_BaseImputer` / `TransformerMixin` expose `get_feature_names_out`, and `fit_transform` records
  `imputation_sequence_` (`:739,798-801`), `n_iter_` (`:781`), `n_features_with_missing_`
  (`:770`), `n_features_in_`, and `random_state_` (`:728`). ferrolearn's `FittedIterativeImputer`
  exposes only `n_iter()`, `initial_fill()`, `initial_strategy()` — **no `get_feature_names_out`,
  no `imputation_sequence_`, no `n_features_in_`, no `random_state_`** (and `feature_models` is a
  private field). Open prereq blocker #1413.

- REQ-13: **PyO3 binding** (NOT-STARTED). There is no `IterativeImputer` CPython binding in
  `ferrolearn-python` — `grep -rn "IterativeImputer" ferrolearn-python/src` finds none — so the
  imputer is unreachable from `import ferrolearn`. Open prereq blocker #1414.

- REQ-14: **ferray substrate** (NOT-STARTED). Compute the initial fill, round-robin regression,
  and convergence over `ferray-core` arrays and `ferray::linalg` rather than `ndarray::Array2` /
  `Array1`, `num_traits::Float`, and the hand-rolled `solve_linear_system` Gaussian elimination
  (R-SUBSTRATE-1/2). Open prereq blocker #1415.

## Acceptance criteria

- AC-1 (REQ-1): the initial fill matches `SimpleImputer` — `column_means_nan([[1,2],[3,NaN],
  [NaN,6]]) == [2.0, 4.0]` and `column_medians_nan(same) == [2.0, 4.0]` (Probe A);
  `fitted.transform(x)` returns an `(n_samples, n_features)` matrix (Probe C) with every NaN
  filled and every observed value unchanged. Pinned by `test_iterative_imputer_basic` (no NaN in
  output), `test_iterative_imputer_no_missing` (output equals input when nothing is missing),
  `test_iterative_imputer_median_strategy`, `test_iterative_imputer_fit_transform`,
  `test_iterative_imputer_f32`.

- AC-2 (REQ-2): `fitted.n_iter()` is in `(0, max_iter]` and the loop terminates without RNG;
  `IterativeImputer::default().max_iter() == 10` and `initial_strategy() == Mean`. Pinned by
  `test_iterative_imputer_n_iter_accessor`, `test_iterative_imputer_convergence`,
  `test_iterative_imputer_default`.

- AC-3 (REQ-3): `IterativeImputer::<f64>::new(10, 1e-3, Mean).fit(Array2::zeros((0,3)))` →
  `Err(InsufficientSamples)`; `new(0, ..).fit([[1,2],[nan,3],[5,nan],[7,8]])` → `Ok` with
  `n_iter() == 0` whose `transform` returns the per-column mean initial fill
  `[[1,2],[13/3,3],[5,13/3],[7,8]]`, matching sklearn (`max_iter==0` → initial fill,
  `:750-752`); a fitted handle's `transform` on a wrong column count → `Err(ShapeMismatch)`;
  calling `transform` on the unfitted `IterativeImputer` → `Err(InvalidParameter)`. Pinned by
  `test_iterative_imputer_zero_rows_error`,
  `test_iterative_imputer_zero_max_iter_returns_initial_fill`,
  `divergence_max_iter_zero_returns_initial_fill`, `test_iterative_imputer_shape_mismatch_error`,
  `test_iterative_imputer_unfitted_transform_error`.

- AC-4 (REQ-4): `IterativeImputer().fit_transform([[1,2],[3,nan],[nan,6]])` under sklearn yields
  `[[1.0,2.0],[3.0,4.000002999996018],[4.999994000015464,6.0]]` with `_estimator ==
  BayesianRidge` (Probe B); ferrolearn's Ridge-`alpha=1` / column-order / L2-tol / no-clip path
  produces materially different imputed values — exact-value parity is NOT claimed (carve-out
  gated on a `BayesianRidge` estimator + REQ-6/7/11). No failing parity test is committed.

- AC-5 (REQ-5): `IterativeImputer(estimator=KNeighborsRegressor())` plugs in a custom estimator;
  `sample_posterior=True` draws posterior samples (`:430-452`); ferrolearn hardwires `ridge_fit`.

- AC-6 (REQ-6): on `[[1,2,3],[nan,nan,6],[7,nan,9],[10,11,nan]]`, `imputation_order='ascending'`
  visits features `[0,2,1]` (fewest-missing first, Probe D) while ferrolearn visits `[0,1,2]`
  (column order == `'roman'`).

- AC-7 (REQ-7): `IterativeImputer(min_value=0, max_value=5)` clamps the imputed `5.0004...` to
  `5.0` (Probe E); ferrolearn stores the unclipped prediction.

- AC-8 (REQ-8): `IterativeImputer(n_nearest_features=2)` predicts each feature from a
  correlation-weighted subset of the others (`_get_neighbor_feat_idx:493-497`); ferrolearn always
  uses all other features.

- AC-9 (REQ-9): `IterativeImputer(initial_strategy='most_frequent')` / `'constant'` +
  `fill_value=0` initializes via `SimpleImputer` (`:314,743`); `missing_values=-1` treats `-1` as
  missing; ferrolearn's `InitialStrategy` is `Mean`/`Median` only and detects only `NaN`.

- AC-10 (REQ-10): `IterativeImputer(random_state=0, imputation_order='random')` is
  reproducible; `skip_complete=True` omits complete features; `add_indicator=True` appends a mask;
  ferrolearn exposes none of these.

- AC-11 (REQ-11): sklearn converges when `inf_norm(Xt - Xt_previous) < tol * max|X_observed|`
  (`:780,811,818`); ferrolearn uses `(total_change/total_value).sqrt() < tol` (L2-relative).

- AC-12 (REQ-12): a fitted imputer exposes `imputation_sequence_`, `n_iter_`, `n_features_in_`,
  `random_state_`, and `get_feature_names_out`; ferrolearn exposes only `n_iter()`,
  `initial_fill()`, `initial_strategy()`.

- AC-13 (REQ-13): a CPython `IterativeImputer` binding fits and transforms from Python; no such
  binding exists in `ferrolearn-python`.

- AC-14 (REQ-14): the initial-fill, round-robin regression, and convergence path computes on
  `ferray-core` arrays + `ferray::linalg` rather than `ndarray` + `num_traits::Float` + the
  hand-rolled `solve_linear_system`.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (round-robin structure + initial-fill values + non-missing preserved + output shape; HEADLINE) | SHIPPED (scoped) | impl `fn fit in iterative_imputer.rs` computes the initial fill via `match self.initial_strategy { InitialStrategy::Mean => column_means_nan(x), InitialStrategy::Median => column_medians_nan(x) }` — `column_means_nan` = per-column `sum/count` over `!v.is_nan()`, `column_medians_nan` = sorted middle (avg of two for even n) — matching sklearn's `SimpleImputer(strategy=initial_strategy)` initialization (`_initial_imputation`, `_iterative.py:743`); then builds `missing_features` and runs the round-robin `ridge_fit`/`ridge_predict` loop (`for &j in &missing_features { ... ridge_fit(&x_train, &y_train, alpha) ... }`) mirroring `_impute_one_feature`'s fit-on-non-missing / predict-on-missing (`:408-418,454,460-465`), storing `FeatureModel { coefficients, intercept }`. `fn transform in iterative_imputer.rs` re-applies the models and returns an `(n_samples, n_features)` `Array2<F>` with NaNs filled and observed values preserved (`initial_fill` overwrites only `v.is_nan()`, mirroring `_assign_where(Xt, X, cond=~mask)` `:829`). Initial-fill VALUES confirmed live: Probe A `mean=[2.0,4.0]`, `median=[2.0,4.0]` (IDENTICAL); output shape Probe C `(3,2)`. Non-test consumer: the boundary re-export `pub use iterative_imputer::{FittedIterativeImputer, InitialStrategy, IterativeImputer};` at `lib.rs:150` (grandfathered S5 / R-DEFER-1 boundary estimator API — `IterativeImputer` IS the public surface). Verification: `cargo test -p ferrolearn-preprocess iterative_imputer` → `test_iterative_imputer_basic`, `test_iterative_imputer_no_missing` (observed preserved), `test_iterative_imputer_median_strategy`, `test_iterative_imputer_fit_transform`, `test_iterative_imputer_f32` green. SCOPED: structural contract + initial-fill VALUES only — exact imputed-value parity is REQ-4 (Probe B). |
| REQ-2 (determinism + termination + `n_iter` accessor) | SHIPPED (scoped) | impl `fn fit in iterative_imputer.rs` has NO RNG (no `random_state` anywhere in the file — deterministic per input), runs `for iter_idx in 0..self.max_iter { n_iter = iter_idx + 1; ... }` (bounded, mirroring sklearn `for self.n_iter_ in range(1, self.max_iter + 1)` `_iterative.py:781`) and breaks early on the tol test (mirroring sklearn's early-stop break `:818-821`); `FittedIterativeImputer::n_iter()` returns the rounds performed (sklearn `n_iter_` `:781`). Non-test consumer: the boundary re-export at `lib.rs:150`. Verification: `cargo test -p ferrolearn-preprocess iterative_imputer` → `test_iterative_imputer_n_iter_accessor` (`0 < n_iter() <= 10`), `test_iterative_imputer_convergence`, `test_iterative_imputer_default` (`max_iter()==10`, `initial_strategy()==Mean`) green. NOTE: the convergence METRIC (L2-relative vs sklearn inf-norm) diverges — that is REQ-11; REQ-2 covers bounded termination + early-stop + `n_iter` reporting only. |
| REQ-3 (error / parameter contracts + `max_iter==0` initial-fill parity, scoped) | SHIPPED | impl `fn fit in iterative_imputer.rs` returns `Err(FerroError::InsufficientSamples { required: 1, actual: 0, context: "IterativeImputer::fit".into() })` when `n_samples == 0`. `max_iter == 0` is now **ACCEPTED**: the rejection guard was REMOVED (was the #1404 divergence, RESOLVED), so the round-robin loop `for iter_idx in 0..self.max_iter` runs zero times and `fit` returns the initial mean/median fill with `n_iter() == 0` — matching sklearn, which sets `self.n_iter_ = 0` and returns the `SimpleImputer(strategy=initial_strategy)` initial imputation for `max_iter==0` (`_iterative.py:750-752`). impl `fn transform in iterative_imputer.rs` returns `Err(FerroError::ShapeMismatch { expected: vec![x.nrows(), n_features], actual: vec![x.nrows(), x.ncols()], context: "FittedIterativeImputer::transform".into() })` when `x.ncols() != n_features` (mirroring sklearn's `check_is_fitted` + feature-count validation). impl `Transform for IterativeImputer in iterative_imputer.rs` (unfitted handle) returns `Err(FerroError::InvalidParameter { name: "IterativeImputer".into(), reason: "imputer must be fitted before calling transform; use fit() first".into() })` (mirroring `check_is_fitted`). Non-test consumer: the boundary re-export at `lib.rs:150` routes every fit/transform through these guards. Verification: `cargo test -p ferrolearn-preprocess iterative_imputer` → `test_iterative_imputer_zero_rows_error`, `test_iterative_imputer_zero_max_iter_returns_initial_fill` (asserts `fit` Ok, `n_iter()==0`, output == per-column mean fill `13/3`), `test_iterative_imputer_shape_mismatch_error`, `test_iterative_imputer_unfitted_transform_error` green; `cargo test -p ferrolearn-preprocess --test divergence_iterative_imputer` → `divergence_max_iter_zero_returns_initial_fill` (sklearn oracle `[[1,2],[13/3,3],[5,13/3],[7,8]]`) GREEN. |
| REQ-4 (exact imputed-VALUE parity) | NOT-STARTED (CARVE-OUT) | open prereq blocker #1405. ALGORITHM divergence on four axes simultaneously: (1) estimator — `ridge_fit(&x_train, &y_train, F::one())` solves closed-form Ridge `(XᵀX + I)⁻¹Xᵀy` vs sklearn default `BayesianRidge()` (`_iterative.py:74,732-735`); (2) order — visits `missing_features` in column order (== sklearn `'roman'` `:529-530`) vs the `'ascending'` default (`:533-535`, Probe D `[0,2,1]` vs `[0,1,2]`); (3) convergence — `(total_change/total_value).sqrt() < tol` (L2-relative) vs `inf_norm < tol*max|X_observed|` (`:780,811,818`); (4) no clip vs `np.clip(.., min_value, max_value)` (`:455-457`, Probe E). Same class as the bayesian-GMM heuristic-vs-VB carve-out (#1067); gated on a `BayesianRidge` estimator in `ferrolearn-linear` + REQ-6/REQ-7/REQ-11. Per R-DEFER-3 NO failing parity test is committed (the fix requires the prerequisites to land together). Probe B anchors the divergence (`sklearn out=[[1.0,2.0],[3.0,4.000002999996018],[4.999994000015464,6.0]]`). |
| REQ-5 (`estimator` param + default `BayesianRidge` + `sample_posterior`) | NOT-STARTED | open prereq blocker #1406. `IterativeImputer<F> { max_iter, tol, initial_strategy }` has NO `estimator` or `sample_posterior` field; the per-feature model is the hardwired internal `ridge_fit`/`ridge_predict` with `alpha = F::one()`. sklearn's `estimator=None` → `BayesianRidge()` cloned per feature (`_iterative.py:74,307,404,732-735`); `sample_posterior=True` draws truncated-normal posterior samples (`:85-89,430-452`). |
| REQ-6 (`imputation_order` param) | NOT-STARTED | open prereq blocker #1407. `fn fit in iterative_imputer.rs` iterates `for &j in &missing_features`, where `missing_features` is built in column-index order (`for j in 0..n_features { ... if has_missing { missing_features.push(j); } }`) — == sklearn `'roman'` (`_get_ordered_idx:529-530`), NOT the `'ascending'` default `argsort(frac, kind="mergesort")[n:]` (`:533-535`, Probe D ascending `[0,2,1]`). No `imputation_order` field; `'descending'`/`'arabic'`/`'random'` (`:531-541`) also absent. |
| REQ-7 (`min_value` / `max_value` clipping) | NOT-STARTED | open prereq blocker #1408. `fn fit`/`fn transform in iterative_imputer.rs` store `ridge_predict` output unclipped (`imputed[[i, j]] = predictions[row_idx]`). sklearn clips every imputed value via `np.clip(imputed_values, self._min_value[feat_idx], self._max_value[feat_idx])` (`_impute_one_feature:455-457`), defaults `-inf`/`+inf` (`:318-319`), supports per-feature array-likes (`:759-760`). No `min_value`/`max_value` fields (Probe E: `5.0004...` not clamped to `5.0`). |
| REQ-8 (`n_nearest_features` + abs-corr feature selection) | NOT-STARTED | open prereq blocker #1409. `fn fit in iterative_imputer.rs` always uses ALL other features as predictors (`let predictor_cols: Vec<usize> = (0..n_features).filter(\|&k\| k != j).collect()`). sklearn's `n_nearest_features` (`:313`) selects a correlation-weighted random subsample (`_get_neighbor_feat_idx:493-497`, `_get_abs_corr_mat:544`). No `n_nearest_features` field and no correlation matrix. |
| REQ-9 (`initial_strategy` most_frequent/constant + `fill_value` + non-NaN `missing_values`) | NOT-STARTED | open prereq blocker #1410. `InitialStrategy` enum is `{ Mean, Median }` only — no `MostFrequent`, no `Constant`; no `fill_value`; missingness detected ONLY via `v.is_nan()` (`column_means_nan`/`initial_fill`/`fn transform in iterative_imputer.rs`). sklearn's `initial_strategy` accepts `"mean"\|"median"\|"most_frequent"\|"constant"` fed to `SimpleImputer` (`:314,743`), `fill_value=None` (`:315`), and `missing_values=np.nan` may be any scalar sentinel (`:309`). |
| REQ-10 (`random_state` + `skip_complete` + `add_indicator` + `keep_empty_features` + `verbose`) | NOT-STARTED | open prereq blocker #1411. `IterativeImputer<F>` exposes NONE of these fields. sklearn's ctor accepts `random_state=None` (`:321`, seeds posterior + random order `:728`), `skip_complete=False` (`:317`, `_get_ordered_idx:525-526`), `add_indicator=False` (`:322`, `_fit_indicator`/`_concatenate_indicator`), `keep_empty_features=False` (`:323`), `verbose=0` (`:320`). |
| REQ-11 (inf-norm convergence `tol * max\|X\|`) | NOT-STARTED | open prereq blocker #1412. `fn fit`/`fn transform in iterative_imputer.rs` converge on `(total_change / total_value).sqrt() < self.tol` (L2-relative, un-scaled). sklearn uses `inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf)` vs `normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))` (`_iterative.py:780,811,818`) and emits `ConvergenceWarning` if never reached (`:823-828`) — different metric, scaling, and warning. Distinct from but folds into REQ-4. |
| REQ-12 (`get_feature_names_out` + `imputation_sequence_`/`n_iter_`(name)/`n_features_in_`/`random_state_` surface) | NOT-STARTED | open prereq blocker #1413. `FittedIterativeImputer<F>` exposes only `n_iter()`, `initial_fill()`, `initial_strategy()` (`feature_models`/`missing_features` are private). sklearn's `fit_transform` records `imputation_sequence_` (`:739,798-801`), `n_iter_` (`:781`), `n_features_with_missing_` (`:770`), `n_features_in_`, `random_state_` (`:728`), plus `get_feature_names_out` from `_BaseImputer`/`TransformerMixin`. |
| REQ-13 (PyO3 binding) | NOT-STARTED | open prereq blocker #1414. No `IterativeImputer` CPython binding exists — `grep -rn "IterativeImputer" ferrolearn-python/src` finds none — so the imputer is unreachable from `import ferrolearn`. |
| REQ-14 (ferray substrate) | NOT-STARTED | open prereq blocker #1415. The initial fill, round-robin regression, and convergence use `ndarray::Array2`/`Array1` (`x.column(j)`, `Array2::zeros`, `out.columns_mut()`), `num_traits::Float`, and the hand-rolled `solve_linear_system` Gaussian-elimination — not `ferray-core` arrays + `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `iterative_imputer.rs` exposes the unfitted `IterativeImputer<F> {
max_iter: usize, tol: F, initial_strategy: InitialStrategy }` (`new(max_iter, tol,
initial_strategy)`, accessors `max_iter()`, `tol()`, `initial_strategy()`; `Default` = `(10,
1e-3, Mean)`) with the `Copy` `InitialStrategy` enum `{ Mean, Median }`, and the fitted
`FittedIterativeImputer<F> { initial_fill: Array1<F>, feature_models: Vec<Option<FeatureModel<F>>>,
missing_features: Vec<usize>, n_iter, max_iter, tol, initial_strategy }` (accessors `n_iter()`,
`initial_fill()`, `initial_strategy()`), where the private `FeatureModel<F> { coefficients:
Array1<F>, intercept: F }` is one closed-form Ridge model per missing feature. `Fit<Array2<F>, ()>
for IterativeImputer` (`fn fit in iterative_imputer.rs`) rejects zero rows
(`InsufficientSamples`), then computes the per-column initial fill (`column_means_nan` /
`column_medians_nan`, ignoring NaN), builds the boolean `missing_mask` and the column-order
`missing_features` list, runs the initial `initial_fill` substitution, then loops up to `max_iter`
rounds (`for iter_idx in 0..self.max_iter`): for each missing feature `j` it assembles `x_train`
(all other columns, non-missing rows) and `y_train` (column `j`, non-missing rows), fits
`ridge_fit(.., alpha = F::one())` (centered `(XᵀX + αI)⁻¹Xᵀy` solved via `solve_linear_system`
Gaussian elimination with partial pivoting), predicts the missing rows with `ridge_predict`, writes
them back, and stores the `FeatureModel`; the round ends with the L2-relative convergence test
`(total_change / total_value).sqrt() < tol`. When `max_iter == 0` the loop body never executes, so
`fit` returns the initial fill with `n_iter() == 0` — matching sklearn's `max_iter==0` short-circuit
(`:750-752`) and resolving the former #1404 divergence. `Transform<Array2<F>>
for FittedIterativeImputer` (`fn transform in iterative_imputer.rs`) validates the column count
(`ShapeMismatch`), re-runs the initial fill, then re-applies the stored per-feature models (or
re-fits on the transform data when a stored model is absent) under the same loop and convergence
test, returning an `(n_samples, n_features)` matrix. The unfitted `Transform for IterativeImputer`
is an error stub (`InvalidParameter`) satisfying the `FitTransform: Transform` supertrait;
`FitTransform` wraps the fit→transform path. The grandfathered boundary re-export at `lib.rs:150`
(`pub use iterative_imputer::{FittedIterativeImputer, InitialStrategy, IterativeImputer}`) is the
non-test production consumer that pins REQ-1/REQ-2/REQ-3 SHIPPED.

**sklearn (target contract).** `IterativeImputer(_BaseImputer)` (`_iterative.py:51`, EXPERIMENTAL,
`:63-70`) takes `__init__(estimator=None, *, missing_values=np.nan, sample_posterior=False,
max_iter=10, tol=1e-3, n_nearest_features=None, initial_strategy="mean", fill_value=None,
imputation_order="ascending", skip_complete=False, min_value=-inf, max_value=inf, verbose=0,
random_state=None, add_indicator=False, keep_empty_features=False)` (`:305-343`). `fit_transform`
(`:732-829`) clones `estimator` (default `BayesianRidge`, `:732-735`), initializes via
`SimpleImputer(strategy=initial_strategy)` (`_initial_imputation:743`), short-circuits when
`max_iter==0`/all-missing/single-feature (`:750-757`), validates `min_value`/`max_value`
(`:759-763`), orders features via `_get_ordered_idx` (`:769`, default `'ascending'` fewest-missing
first `:533-535`), computes the abs-correlation matrix for `n_nearest_features` (`:772`), and runs
the round-robin loop `for self.n_iter_ in range(1, self.max_iter + 1)` (`:781`): each feature is
imputed by `_impute_one_feature` (`:345-466`) — fit the cloned estimator on neighbor cols over
non-missing rows (`:408-418`), `predict` (`:454`), `np.clip` to `[min_value, max_value]`
(`:455-457`), `_safe_assign` back (`:460-465`) — recording an `_ImputerTriplet` in
`imputation_sequence_` (`:798-801`). Convergence is `inf_norm(Xt - Xt_previous) < tol *
max|X_observed|` (`:780,811,818`), else `ConvergenceWarning` (`:823-828`); finally observed values
are restored via `_assign_where(Xt, X, cond=~mask)` (`:829`).

**The gap.** ferrolearn matches sklearn on the *round-robin STRUCTURE* (per-feature regression on
the other features over non-missing rows, repeated per round, observed values preserved, output
shape `(n_samples, n_features)` — REQ-1, Probes A/C), on the *initial-fill VALUES* (mean/median ==
`SimpleImputer` — REQ-1, Probe A), on *bounded RNG-free termination + `n_iter`* (REQ-2), and on the
scoped structural contracts (zero-rows, `max_iter==0` → initial fill matching `:750-752`,
column-count, unfitted — REQ-3). It does **not** match the *exact imputed VALUES* (REQ-4 CARVE-OUT)
— four algorithm axes diverge at once: the per-feature estimator (Ridge `alpha=1` vs default
`BayesianRidge`, REQ-5), the feature order (column == `'roman'` vs `'ascending'`, REQ-6, Probe D),
clipping (none vs `min/max`, REQ-7, Probe E), and convergence (L2-relative vs inf-norm, REQ-11). The
remaining gaps are configuration and surface: no `estimator`/`sample_posterior` (REQ-5), no
`imputation_order` (REQ-6), no `min/max` clip (REQ-7), no `n_nearest_features` (REQ-8), no
`most_frequent`/`constant`/`fill_value`/non-NaN `missing_values` (REQ-9), no `random_state`/
`skip_complete`/`add_indicator`/`keep_empty_features`/`verbose` (REQ-10), no inf-norm convergence
(REQ-11), no `imputation_sequence_`/`n_iter_`-named/`n_features_in_`/`get_feature_names_out` surface
(REQ-12), no PyO3 binding (REQ-13), and the non-ferray substrate (REQ-14). This is a
**shipped-partial** unit (3 SHIPPED / 11 NOT-STARTED), with the exact-value parity (REQ-4) framed as
an honest NOT-STARTED **carve-out** in the class of #1067, gated on a `BayesianRidge` estimator and
REQ-6/7/11 — no failing parity test committed.

## Verification

Commands establishing the SHIPPED claims (REQ-1 round-robin structure + initial-fill values +
output shape, REQ-2 determinism + termination + `n_iter`, REQ-3 scoped error contracts +
`max_iter==0` initial-fill parity):

```bash
# Consumer / module wiring check:
grep -rn "pub mod iterative_imputer" ferrolearn-preprocess/src/lib.rs           # :100
grep -rn "pub use iterative_imputer::" ferrolearn-preprocess/src/lib.rs         # :150 boundary re-export consumer

# REQ-1/REQ-2/REQ-3 (in-module tests):
cargo test -p ferrolearn-preprocess iterative_imputer
#   REQ-1: test_iterative_imputer_basic (no NaN in output),
#          test_iterative_imputer_no_missing (output == input),
#          test_iterative_imputer_median_strategy, test_iterative_imputer_fit_transform,
#          test_iterative_imputer_f32
#   REQ-2: test_iterative_imputer_convergence, test_iterative_imputer_n_iter_accessor (0<n_iter<=10),
#          test_iterative_imputer_default (max_iter==10, initial_strategy==Mean)
#   REQ-3: test_iterative_imputer_zero_rows_error,
#          test_iterative_imputer_zero_max_iter_returns_initial_fill (max_iter=0 -> Ok, n_iter()==0,
#                output == per-column mean fill 13/3 == sklearn :750-752),
#          test_iterative_imputer_shape_mismatch_error, test_iterative_imputer_unfitted_transform_error

# REQ-3 max_iter==0 parity (divergence-suite, now GREEN after #1404 fix):
cargo test -p ferrolearn-preprocess --test divergence_iterative_imputer
#   divergence_max_iter_zero_returns_initial_fill -> ok
#     (sklearn oracle [[1,2],[13/3,3],[5,13/3],[7,8]] == ferrolearn max_iter=0 initial fill)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — the INITIAL fill equals SimpleImputer(mean/median):
python3 -c "import numpy as np
from sklearn.impute import SimpleImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
print('mean=', SimpleImputer(strategy='mean').fit(X).statistics_.tolist())
print('median=', SimpleImputer(strategy='median').fit(X).statistics_.tolist())"
#   -> mean= [2.0, 4.0]   median= [2.0, 4.0]
#   ferrolearn column_means_nan / column_medians_nan = [2.0, 4.0]  (IDENTICAL — REQ-1)

# REQ-3 max_iter==0 oracle gate — sklearn returns the initial fill with n_iter_=0:
python3 -c "import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
X=np.array([[1.,2.],[np.nan,3.],[5.,np.nan],[7.,8.]])
ii=IterativeImputer(max_iter=0); out=ii.fit_transform(X)
print('n_iter_=', ii.n_iter_, 'out=', out.tolist())"
#   -> n_iter_= 0 out= [[1.0,2.0],[4.333333333333333,3.0],[5.0,4.333333333333333],[7.0,8.0]]
#   ferrolearn IterativeImputer::new(0,..).fit(X) -> Ok, n_iter()==0, transform == same fill (REQ-3)

# REQ-4 CARVE-OUT anchor (value DIVERGENCE, NOT parity) — sklearn default is BayesianRidge:
python3 -c "import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
X=np.array([[1.,2.],[3.,np.nan],[np.nan,6.]])
ii=IterativeImputer(max_iter=10, tol=1e-3, random_state=0); out=ii.fit_transform(X)
print('estimator=', type(ii._estimator).__name__, 'out=', out.tolist())"
#   -> estimator= BayesianRidge out= [[1.0,2.0],[3.0,4.000002999996018],[4.999994000015464,6.0]]
#   ferrolearn (Ridge alpha=1 / column-order / L2-tol / no-clip) DIFFERS — exact-value parity
#   NOT claimed; carve-out gated on a BayesianRidge estimator + REQ-6/7/11 (no failing test committed).
```

The in-module `#[test]`s exercise REQ-1 (round-robin produces a no-NaN
`(n_samples, n_features)` output, observed values preserved, mean + median initial strategies,
f32), REQ-2 (bounded RNG-free termination + `n_iter()` accessor + `Default`), and REQ-3 (every
error / edge path — zero rows, `max_iter==0` → initial fill with `n_iter()==0`, column-count
mismatch, unfitted transform); the divergence suite's `divergence_max_iter_zero_returns_initial_fill`
pins the `max_iter==0` sklearn parity (#1404, RESOLVED). No green ferrolearn command establishes
REQ-4..REQ-14 (exact-value parity, `estimator`/`sample_posterior`, `imputation_order`, `min`/`max`
clip, `n_nearest_features`, `most_frequent`/`constant`/`fill_value`/non-NaN `missing_values`,
`random_state`/`skip_complete`/`add_indicator`/`keep_empty_features`/`verbose`, inf-norm convergence,
fitted-attr/`get_feature_names_out` surface, PyO3, ferray). REQ-4 is an honest carve-out with **no
committed failing parity test** (R-DEFER-3): closing it requires a `BayesianRidge` estimator plus
REQ-6/REQ-7/REQ-11 to land together.

## Blockers

REQ-1 (round-robin structure + initial-fill values + non-missing preserved + output shape,
HEADLINE), REQ-2 (determinism + termination + `n_iter`), and REQ-3 (scoped error / parameter
contracts + `max_iter==0` initial-fill parity) are SHIPPED, with the boundary re-export at
`lib.rs:150` as the grandfathered (S5 / R-DEFER-1) non-test production consumer. The former
`max_iter==0` divergence (#1404) is **RESOLVED** — the rejection guard was removed and `max_iter==0`
now returns the initial fill with `n_iter()==0`, matching sklearn `_iterative.py:750-752`
(pinned by `divergence_max_iter_zero_returns_initial_fill` +
`test_iterative_imputer_zero_max_iter_returns_initial_fill`).

The remaining REQs are NOT-STARTED, filed as `-l blocker` issues against tracking issue
#1403. REQ-4 is an exact-value-parity **carve-out** (algorithm divergence in the class of #1067),
NOT minimally fixable, with **no committed failing test** (R-DEFER-3):

- #1405 — REQ-4: exact imputed-VALUE parity CARVE-OUT — Ridge `alpha=1` ≠ default `BayesianRidge`
  (`_iterative.py:74,732-735`) + column order ≠ `'ascending'` (`:533-535`) + L2-tol ≠ inf-norm
  (`:780,811,818`) + no clip ≠ `np.clip` (`:455-457`). Gated on a `BayesianRidge` estimator
  (`ferrolearn-linear`) + REQ-6/REQ-7/REQ-11.
- #1406 — REQ-5: no `estimator` param / default `BayesianRidge` / `sample_posterior`
  (`_iterative.py:74,307,310,404,430-452,732-735`).
- #1407 — REQ-6: no `imputation_order` param — column order == `'roman'`, not `'ascending'`
  default (`_get_ordered_idx:529-541`).
- #1408 — REQ-7: no `min_value`/`max_value` clipping (`_impute_one_feature:455-457`,
  ctor `:318-319`).
- #1409 — REQ-8: no `n_nearest_features` / abs-correlation feature selection
  (`_get_neighbor_feat_idx:493-497`, `_get_abs_corr_mat:544`).
- #1410 — REQ-9: `InitialStrategy` is `Mean`/`Median` only — no `most_frequent`/`constant`,
  no `fill_value`, no non-NaN `missing_values` (`_iterative.py:309,314-315,743`).
- #1411 — REQ-10: no `random_state`/`skip_complete`/`add_indicator`/`keep_empty_features`/
  `verbose` (`_iterative.py:317,320-323,728`).
- #1412 — REQ-11: L2-relative convergence, not inf-norm scaled by `tol*max|X_observed|`
  (`_iterative.py:780,811,818,823-828`).
- #1413 — REQ-12: no `imputation_sequence_`/`n_iter_`-named/`n_features_in_`/`random_state_`
  fitted attrs, no `get_feature_names_out` (`_iterative.py:739,770,781,798-801`).
- #1414 — REQ-13: no PyO3 `IterativeImputer` binding in `ferrolearn-python`.
- #1415 — REQ-14: fit/transform on `ndarray` + `num_traits` + hand-rolled `solve_linear_system`,
  not ferray (R-SUBSTRATE-1/2).
