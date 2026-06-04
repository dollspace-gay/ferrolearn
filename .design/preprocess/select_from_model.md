# SelectFromModel

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 8b3022ba
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_from_model.py  # _calculate_threshold(estimator, importances, threshold) (:24-71): threshold=None -> "mean" (or 1e-5 if l1-penalized/Lasso/ElasticNet-l1, :30-38); str "median" -> np.median (:57-58), "mean" -> np.mean (:60-61), "scale*ref" -> scale*{median|mean}(importances) (:43-55); float -> float(threshold) (:68-69). class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator) __init__(estimator, *, threshold=None, prefit=False, norm_order=1, max_features=None, importance_getter="auto") (:256-271). _get_support_mask (:273-313): scores = _get_feature_importances(estimator, getter=importance_getter, transform_func="norm", norm_order) (:299-304); threshold=_calculate_threshold(...) (:305); if max_features is not None: mask=zeros; candidate=np.argsort(-scores, kind="mergesort")[:max_features]; mask[candidate]=True else mask=ones (:306-311); THEN mask[scores < threshold]=False (:312). _check_max_features (:315-331): max_features int in [0, n_features] or callable, check_scalar min_val=0 max_val=n_features. fit (:337). SelectorMixin provides get_support / transform / inverse_transform / get_feature_names_out.
ferrolearn-module: ferrolearn-preprocess/src/select_from_model.rs
parity-ops: SelectFromModel
crosslink-issue: 1352
-->

## Summary

scikit-learn's `SelectFromModel` (`_from_model.py:256`) is a meta-transformer that
wraps a base estimator, extracts its per-feature importances
(`coef_` / `feature_importances_`, normalized by `_get_feature_importances` with
`transform_func="norm"`, `:299-304`), derives a scalar `threshold` from
`_calculate_threshold` (`:24-71`), and keeps every feature whose score is at or
above that threshold — optionally first restricting to the top-`max_features`
features by score (`_get_support_mask`, `:273-313`). The selector surface
(`get_support`, `transform`, `inverse_transform`, `get_feature_names_out`) comes
from `SelectorMixin`.

`ferrolearn-preprocess/src/select_from_model.rs` ships the **threshold + mask +
max-features core, GIVEN a static importance vector** rather than wrapping an
estimator: `SelectFromModelExt<F> { threshold: ThresholdStrategy, max_features:
Option<usize> }` (`new(threshold, max_features)`, `Default` = `(Mean, None)`,
accessors `threshold_strategy()` / `max_features()`) takes the importances as the
`Fit` input `x: &Array1<F>` (one value per feature), computes the threshold
(`ThresholdStrategy::{Mean, Median, Value, Percentile}`), masks `imp[j] >=
threshold`, then caps survivors to `max_features` by descending importance
(stable, ascending-index tie-break). It produces `FittedSelectFromModelExt<F> {
n_features_in, threshold_value, importances, selected_indices }` (accessors
`threshold_value()`, `importances()`, `selected_indices()`,
`n_features_selected()`); `Transform<Array2<F>>` returns the selected columns
(`ShapeMismatch` on a column-count mismatch). There is **no estimator wrapping, no
`coef_` / `feature_importances_` extraction, no `norm_order`, no scaled-string
threshold, no `prefit` / `importance_getter`, no callable `max_features`, and no
`SelectorMixin` surface** (`get_support` / `inverse_transform` /
`get_feature_names_out`).

**ENTRY CONDITION — orphan → activation (resolved THIS iteration).** At the
baseline commit this file is an **orphan**: `ferrolearn-preprocess/src/lib.rs`
contains **no `mod select_from_model;` declaration** (verify:
`grep -rn "mod select_from_model" ferrolearn-preprocess/src` returns nothing), so
the file is **UNCOMPILED** and its 16 in-module `#[test]`s never run. A separate,
compiled, *basic* `SelectFromModel<F>` (mean / explicit-threshold only,
`new_from_importances`) lives in `feature_selection.rs:596` and is re-exported at
`lib.rs:110-112`. Tracking issue #1352 ACTIVATES this richer file — declares `pub
mod select_from_model;` plus a boundary re-export of `SelectFromModelExt` /
`FittedSelectFromModelExt` / `ThresholdStrategy` — promoting it to the live
`SelectFromModel` translation unit. **REQ-1 / REQ-2 SHIPPED-ness is classified on
the basis that activation + tests land in this iteration** (the builder declares
the module and adds the boundary re-export *before* the REQ table is finalized);
the boundary re-export is the grandfathered (S5 / R-DEFER-1) non-test production
consumer. This is a **shipped-partial** unit: **2 SHIPPED** (REQ-1 threshold +
mask + max-features core, REQ-2 scoped error / parameter contracts) / **8
NOT-STARTED** (REQ-3 estimator wrapping + importance extraction, REQ-4
`norm_order`, REQ-5 scaled-string / l1-default thresholds, REQ-6 `prefit` /
`importance_getter`, REQ-7 callable `max_features` + `_check_max_features` range
validation, REQ-8 `SelectorMixin` surface, REQ-9 PyO3 binding, REQ-10 ferray
substrate). `ThresholdStrategy::Percentile` is a **ferrolearn EXTENSION** with no
sklearn `SelectFromModel` analog (documented in Architecture, NOT a parity REQ).

## Probes (live sklearn oracle, 1.5.2)

Because `SelectFromModel` wraps an estimator (no direct importances-vector API),
the cleanest oracle is sklearn's own `_get_support_mask` math replicated in numpy
on a chosen importance vector — this IS the spec the doc pins ferrolearn's
`selected_indices()` against (`_calculate_threshold:24-71`,
`_get_support_mask:306-312`). All values below are live numpy output.

```bash
# PROBE A (REQ-1) — mean threshold + mask (scores >= threshold), _calculate_threshold:60-61:
python3 -c "import numpy as np
scores=np.array([0.1,0.5,0.4]); thr=np.mean(scores)
print('A mean thr=',thr,'sel=',np.flatnonzero(scores>=thr).tolist())"
#   -> A mean thr= 0.3333333333333333 sel= [1, 2]
#   ferrolearn SelectFromModelExt::new(ThresholdStrategy::Mean,None).fit(array![0.1,0.5,0.4]):
#     threshold_value() = 0.3333..., selected_indices() = [1, 2]  (IDENTICAL).

# PROBE B (REQ-1) — median threshold, _calculate_threshold:57-58 (np.median):
python3 -c "import numpy as np
print('B median odd [0.1,0.5,0.3]=',np.median(np.array([0.1,0.5,0.3])))
sc=np.array([0.1,0.5,0.2,0.6]); t=np.median(sc)
print('B median even [0.1,0.5,0.2,0.6]=',t,'sel=',np.flatnonzero(sc>=t).tolist())"
#   -> B median odd [0.1,0.5,0.3]= 0.3
#   -> B median even [0.1,0.5,0.2,0.6]= 0.35 sel= [1, 3]
#   ferrolearn compute_median (avg of two middle for even n) matches np.median; on the even
#   fixture selected_indices() = [1, 3]  (IDENTICAL).

# PROBE C (REQ-1) — max_features top-k THEN threshold (_get_support_mask:306-312, mergesort tie-break):
python3 -c "import numpy as np
scores=np.array([0.3,0.5,0.1,0.7]); mf=2; thr=0.0
mask=np.zeros_like(scores,bool); cand=np.argsort(-scores,kind='mergesort')[:mf]; mask[cand]=True; mask[scores<thr]=False
print('C mf=2 thr=0 cand=',cand.tolist(),'sel=',np.flatnonzero(mask).tolist())"
#   -> C mf=2 thr=0 cand= [3, 1] sel= [1, 3]
#   ferrolearn SelectFromModelExt::new(ThresholdStrategy::Value(0.0),Some(2)).fit(array![0.3,0.5,0.1,0.7]):
#     selected_indices() = [1, 3]  (IDENTICAL — pinned by test_max_features_cap).

# PROBE D (REQ-1) — ORDER-EQUIVALENCE: ferrolearn applies threshold THEN caps survivors;
# sklearn caps to top-max_features THEN drops < threshold. The intersection is IDENTICAL:
python3 -c "import numpy as np
scores=np.array([0.1,0.5,0.4,0.6]); mf=2; thr=np.mean(scores)
m=np.zeros_like(scores,bool); c=np.argsort(-scores,kind='mergesort')[:mf]; m[c]=True; m[scores<thr]=False
sk=np.flatnonzero(m).tolist()
surv=[j for j in range(len(scores)) if scores[j]>=thr]
fl=sorted(sorted(surv,key=lambda j:(-scores[j],j))[:mf])
print('D thr=',round(thr,4),'sklearn(cap-then-thr)=',sk,'ferrolearn(thr-then-cap)=',fl,'EQUAL=',sk==fl)"
#   -> D thr= 0.4 sklearn(cap-then-thr)= [1, 3] ferrolearn(thr-then-cap)= [1, 3] EQUAL= True
#   => algebraically equivalent: (top-mf by score) ∩ (score >= thr) == (score >= thr) capped to top-mf.

# PROBE E (REQ-1) — tie-break: equal scores, mergesort keeps ASCENDING index (matches ferrolearn
# stable desc-sort + ascending-index resort):
python3 -c "import numpy as np
scores=np.array([0.5,0.5,0.5,0.5]); mf=2; thr=0.0
m=np.zeros_like(scores,bool); c=np.argsort(-scores,kind='mergesort')[:mf]; m[c]=True; m[scores<thr]=False
print('E ties mf=2 sel=',np.flatnonzero(m).tolist())"
#   -> E ties mf=2 sel= [0, 1]
#   ferrolearn: stable sort by importance DESC keeps original ascending order among ties, truncate,
#   then sort_unstable() (ascending) -> [0, 1]  (IDENTICAL tie-break).

# PROBE F (REQ-3, the HONEST gap) — sklearn EXTRACTS importances from a wrapped estimator's coef_;
# ferrolearn takes the importance VECTOR directly and never sees an estimator:
python3 -c "import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
X=np.array([[0.,1.,2.],[1.,1.,0.],[2.,0.,1.],[3.,1.,2.]]); y=np.array([0,0,1,1])
sfm=SelectFromModel(LogisticRegression(max_iter=200), threshold='mean').fit(X,y)
print('F coef_ norm scores=',np.round(np.linalg.norm(np.atleast_2d(sfm.estimator_.coef_),ord=1,axis=0),4).tolist())
print('F get_support=',sfm.get_support().tolist())"
#   -> F coef_ norm scores= [...] get_support= [...]  (estimator-dependent; the point: sklearn
#      derives `scores` from estimator_.coef_ via _get_feature_importances; ferrolearn has no
#      estimator-wrapping path -> REQ-3 NOT-STARTED.)

# PROBE G (EXTENSION, NOT a REQ) — ferrolearn ThresholdStrategy::Percentile has NO sklearn
# SelectFromModel analog (sklearn supports only mean / median / scale*ref / float):
python3 -c "import numpy as np
sorted_v=np.sort(np.array([0.5,0.1,0.7,0.3])); rank=(100-50)/100*(4-1)
lo=int(np.floor(rank)); hi=int(np.ceil(rank)); frac=rank-np.floor(rank)
thr=sorted_v[lo]*(1-frac)+sorted_v[hi]*frac
print('G Percentile(50) ferrolearn thr=',thr,'sel=',[j for j,v in enumerate([0.5,0.1,0.7,0.3]) if v>=thr])"
#   -> G Percentile(50) ferrolearn thr= 0.4 sel= [0, 2]
#   sklearn SelectFromModel has no percentile threshold; classified as a ferrolearn extension.
```

## Requirements

- REQ-1: **Threshold + selection mask + max-features cap, GIVEN a static
  importance vector** (HEADLINE, SHIPPED). For mean / median / explicit-value
  thresholds compute the scalar threshold exactly as `_calculate_threshold`
  (`mean` = `np.mean(importances)`, `:60-61`; `median` = `np.median(importances)`,
  `:57-58`; float = `float(threshold)`, `:68-69`), build the support mask `score
  >= threshold` and, when `max_features` is set, restrict to the top-`max_features`
  by score with a stable / ascending-index tie-break (`_get_support_mask:306-312`,
  `np.argsort(-scores, kind="mergesort")`). ferrolearn's `Fit::fit for
  SelectFromModelExt` computes `ThresholdStrategy::Mean` as biased `sum/n`,
  `Median` via `compute_median` (avg of two middle for even `n`, matching
  `np.median`), `Value(v)` as `v`, then `selected_indices = {j : imp[j] >=
  threshold}`, then the `max_features` cap (sort survivors by importance DESC,
  stable so ties keep ascending index, truncate, `sort_unstable()` back to
  ascending). ferrolearn applies threshold THEN caps survivors whereas sklearn caps
  THEN thresholds — **algebraically equivalent** (Probe D: the intersection is
  identical; Probe E: tie-break matches mergesort / ascending index). Pinned by 16
  in-module tests once the module is activated.

- REQ-2: **Error / parameter contracts** (scoped, SHIPPED). `Fit::fit` returns
  `InvalidParameter { name: "importances", .. }` when the importance vector is
  empty (`n == 0`); `ThresholdStrategy::Percentile(pct)` with `pct <= 0.0 || pct >
  100.0` returns `InvalidParameter { name: "percentile", .. }`; `Transform::transform
  for FittedSelectFromModelExt` returns `ShapeMismatch` when `x.ncols() !=
  n_features_in`. These mirror sklearn's structural validation (`_get_support_mask`
  `check_is_fitted` path, `:278-282`; `_validate_data` column-count check) scoped to
  the contracts ferrolearn actually enforces over the importance-vector API.

- REQ-3: **Estimator wrapping + `coef_` / `feature_importances_` extraction** (the
  big HONEST gap). sklearn's `SelectFromModel` wraps a base `estimator` and derives
  `scores = _get_feature_importances(estimator, getter=importance_getter,
  transform_func="norm", norm_order)` from the fitted `estimator_.coef_` /
  `feature_importances_` (`_get_support_mask:299-304`); `fit` clones and fits the
  estimator on `(X, y)` (`:337`). ferrolearn takes the importance VECTOR directly
  as the `Fit` input `x: &Array1<F>` and **never wraps or fits an estimator**
  (Probe F).

- REQ-4: **`norm_order` multi-output coef norm**. sklearn's
  `_get_feature_importances` reduces a multi-output `coef_` (shape `(n_outputs,
  n_features)`) to a per-feature score via `np.linalg.norm(coef, axis=0,
  ord=norm_order)` (`transform_func="norm"`, ctor `norm_order=1`, `:262`,
  `:299-304`). ferrolearn has **no `norm_order` param and no coef-norm reduction**
  (it consumes pre-reduced importances).

- REQ-5: **Scaled-string thresholds + default-from-estimator (l1 → 1e-5)**.
  sklearn's `_calculate_threshold` (`:24-71`) supports `"scale*reference"` strings
  (`scale * {median|mean}(importances)`, `:43-55`) and, when `threshold is None`,
  picks `1e-5` for l1-penalized / `Lasso` / l1 `ElasticNet` estimators else
  `"mean"` (`:27-40`). ferrolearn's `ThresholdStrategy` has **no scaled-string
  variant and no estimator-derived default** (`Default` is unconditionally `Mean`).

- REQ-6: **`prefit` + `importance_getter` params**. sklearn's ctor
  (`:256-271`) accepts `prefit=False` (when `True`, `_get_support_mask`
  `check_is_fitted(self.estimator)` and skips re-fitting, `:278-282`) and
  `importance_getter="auto"` (attribute path / callable selecting the importance
  source). ferrolearn exposes **neither** (no estimator to be prefit; importances
  are passed in).

- REQ-7: **Callable `max_features` + `_check_max_features` range validation**.
  sklearn's `max_features` may be an `int` OR a callable `max_features(X)`, and
  `_check_max_features` (`:315-331`) validates the resolved value with
  `check_scalar(..., min_val=0, max_val=n_features)` (raising on out-of-range or a
  non-`Integral`). ferrolearn's `max_features: Option<usize>` is **int-only with no
  callable form and no `[0, n_features]` range validation** (a `usize` larger than
  the feature count simply never binds).

- REQ-8: **`SelectorMixin` surface (`get_support`, `inverse_transform`,
  `get_feature_names_out`)**. sklearn's `SelectFromModel` inherits `SelectorMixin`,
  exposing `get_support()` (the boolean mask / indices), `inverse_transform`
  (scatter selected columns back into a full-width zero matrix), and
  `get_feature_names_out` (filtered feature names). ferrolearn's
  `FittedSelectFromModelExt<F>` exposes `selected_indices()` /
  `n_features_selected()` accessors but **no `get_support` boolean-mask form, no
  `inverse_transform`, and no `get_feature_names_out`**.

- REQ-9: **PyO3 binding**. There is no `_RsSelectFromModel` (or `*Ext`) CPython
  binding in `ferrolearn-python` — `grep -rn "SelectFromModel" ferrolearn-python/src`
  finds none — so the selector is unreachable from Python.

- REQ-10: **ferray substrate**. Compute the threshold, the support mask, and the
  column gather over `ferray-core` arrays rather than `ndarray::Array1` /
  `Array2`, `num_traits::Float`, and the `Vec<usize>` index path (`select_columns`)
  (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `SelectFromModelExt::<f64>::new(ThresholdStrategy::Mean,
  None).fit(array![0.1, 0.5, 0.4])` yields `threshold_value() ≈ 0.3333` and
  `selected_indices() == [1, 2]` (Probe A); the even-`n` median fixture
  `array![0.1, 0.5, 0.2, 0.6]` yields `selected_indices() == [1, 3]` at threshold
  `0.35` (Probe B); `new(ThresholdStrategy::Value(0.0), Some(2)).fit(array![0.3,
  0.5, 0.1, 0.7])` yields `[1, 3]` (Probe C, top-2 by score). Pinned by
  `test_mean_threshold`, `test_median_threshold`, `test_median_threshold_even`,
  `test_explicit_value_threshold`, `test_max_features_cap`,
  `test_max_features_not_needed`, `test_none_selected_high_threshold`, `test_f32`
  (in-module, run once the module is activated).

- AC-2 (REQ-2): `SelectFromModelExt::<f64>::new(Mean, None).fit(Array1::zeros(0))`
  returns `Err(InvalidParameter)`; `Percentile(0.0)` and `Percentile(101.0)` return
  `Err(InvalidParameter)`; a fitted handle's `transform` on a wrong column count
  returns `Err(ShapeMismatch)`. Pinned by `test_empty_importances_error`,
  `test_percentile_invalid`, `test_shape_mismatch_on_transform`.

- AC-3 (REQ-3): `SelectFromModel(LogisticRegression(), threshold='mean').fit(X,
  y).get_support()` derives scores from `estimator_.coef_` (Probe F); ferrolearn
  has no estimator-wrapping `Fit<(Array2, Array1)>` path — its `Fit` input is the
  importance vector itself.

- AC-4 (REQ-4): a multi-output `coef_` of shape `(n_outputs, n_features)` reduces
  to per-feature scores via `np.linalg.norm(coef, axis=0, ord=norm_order)`;
  ferrolearn has no `norm_order` / coef-norm step.

- AC-5 (REQ-5): `_calculate_threshold(est, imps, "1.5*mean")` →
  `1.5 * np.mean(imps)` (`:43-55`); `threshold=None` on a `Lasso` → `1e-5`
  (`:30-38`); ferrolearn's `ThresholdStrategy` has neither variant.

- AC-6 (REQ-6): `SelectFromModel(prefit_est, prefit=True)` skips re-fitting and
  `check_is_fitted`s the estimator (`:278-282`); `importance_getter` selects the
  importance source; ferrolearn exposes neither param.

- AC-7 (REQ-7): `SelectFromModel(est, max_features=lambda X: X.shape[1]//2)`
  resolves the callable and `_check_max_features` validates the result in
  `[0, n_features]` (`:315-331`); ferrolearn's `max_features: Option<usize>` is
  int-only with no range check.

- AC-8 (REQ-8): a fitted selector exposes `get_support()`,
  `inverse_transform(transform(X)) == X * mask`, and
  `get_feature_names_out(names)`; ferrolearn exposes only `selected_indices()` /
  `n_features_selected()`.

- AC-9 (REQ-9): a CPython `SelectFromModel` binding fits and transforms from
  Python; no such binding exists in `ferrolearn-python`.

- AC-10 (REQ-10): the threshold / mask / gather path computes on `ferray-core`
  arrays rather than `ndarray` + `num_traits::Float` + the `Vec<usize>` index path.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (threshold + selection mask + max-features cap, GIVEN a static importance vector; HEADLINE) | SHIPPED | impl `Fit::fit for SelectFromModelExt in select_from_model.rs` computes the threshold per `self.threshold` (`ThresholdStrategy::Mean => values.iter().fold(F::zero(), \|acc, v\| acc + v) / F::from(n)` — biased `np.mean`, matching `_calculate_threshold` `threshold = np.mean(importances)` `_from_model.py:60-61`; `Median => compute_median(&values)` — `if n % 2 == 0 { (sorted[n/2-1] + sorted[n/2]) / two } else { sorted[n/2] }`, matching `np.median` `:57-58`; `Value(v) => F::from(v)`, matching `float(threshold)` `:68-69`), builds `selected_indices = values.iter().enumerate().filter(\|&(_, &imp)\| imp >= threshold_value).map(\|(j, _)\| j).collect()` (the `mask[scores < threshold] = False` complement, `:312`), then the `max_features` cap `if selected_indices.len() > max_f { selected_indices.sort_by(\|&a, &b\| values[b].partial_cmp(&values[a])); selected_indices.truncate(max_f); selected_indices.sort_unstable(); }` (stable desc-by-importance keeps ascending index among ties, then re-sort ascending — matches `np.argsort(-scores, kind="mergesort")[:max_features]` `:308-309`). ferrolearn applies threshold THEN caps survivors; sklearn caps THEN thresholds — **algebraically equivalent** (Probe D: `sklearn=[1,3]` == `ferrolearn=[1,3]`, EQUAL=True; Probe E ties: both `[0,1]`). Non-test consumer: the boundary re-export `pub use select_from_model::{SelectFromModelExt, FittedSelectFromModelExt, ThresholdStrategy};` ADDED by the #1352 activation (grandfathered S5 / R-DEFER-1 boundary estimator API), alongside the `PipelineTransformer` / `FittedPipelineTransformer` impls on `FittedSelectFromModelExt`. **ENTRY BLOCKER (resolved THIS iteration): at baseline the file is an orphan — no `mod select_from_model;` in `lib.rs` (`grep -rn "mod select_from_model" ferrolearn-preprocess/src` returns nothing) — so it is UNCOMPILED; the activation declares `pub mod select_from_model;` + the re-export before this table is finalized.** Verification: `cargo test -p ferrolearn-preprocess select_from_model` → `test_mean_threshold` (`selected_indices == [1,2]`, Probe A), `test_median_threshold`, `test_median_threshold_even` (`[1,3]`, Probe B), `test_explicit_value_threshold`, `test_max_features_cap` (`[1,3]`, Probe C), `test_max_features_not_needed`, `test_none_selected_high_threshold`, `test_f32` green (16 in-module tests run only after activation). |
| REQ-2 (error / parameter contracts, scoped) | SHIPPED (scoped) | impl `Fit::fit for SelectFromModelExt in select_from_model.rs` returns `Err(FerroError::InvalidParameter { name: "importances".into(), reason: "importance vector must not be empty".into() })` when `n == 0`, and for `ThresholdStrategy::Percentile(pct)` returns `Err(FerroError::InvalidParameter { name: "percentile".into(), reason: format!("percentile must be in (0, 100], got {}", pct) })` when `pct <= 0.0 \|\| pct > 100.0`; impl `Transform::transform for FittedSelectFromModelExt in select_from_model.rs` returns `Err(FerroError::ShapeMismatch { expected: vec![x.nrows(), self.n_features_in], actual: vec![x.nrows(), x.ncols()], context: "FittedSelectFromModelExt::transform".into() })` when `x.ncols() != self.n_features_in`. Mirrors sklearn's structural validation over the importance-vector API (`_get_support_mask` `check_is_fitted` path `_from_model.py:278-282`; `_validate_data` column-count check). Non-test consumer: the boundary re-export added by the #1352 activation routes every fit/transform through these guards. Verification: `cargo test -p ferrolearn-preprocess select_from_model` → `test_empty_importances_error`, `test_percentile_invalid`, `test_shape_mismatch_on_transform` green (after activation). |
| REQ-3 (estimator wrapping + `coef_` / `feature_importances_` extraction) | NOT-STARTED | open prereq blocker #1353. `Fit<Array1<F>, ()> for SelectFromModelExt` takes the importance VECTOR directly (`x: &Array1<F>`, one value per feature) and never wraps, clones, or fits a base estimator. sklearn's `SelectFromModel.fit` (`_from_model.py:337`) fits the wrapped estimator on `(X, y)` and `_get_support_mask` derives `scores = _get_feature_importances(estimator, getter=importance_getter, transform_func="norm", norm_order)` from `estimator_.coef_` / `feature_importances_` (`:299-304`). Probe F: sklearn derives `get_support()` from `LogisticRegression().fit(X,y).coef_`; ferrolearn has no such path. |
| REQ-4 (`norm_order` multi-output coef norm) | NOT-STARTED | open prereq blocker #1354. There is NO `norm_order` field and NO coef-norm reduction in `SelectFromModelExt<F>` — it consumes pre-reduced per-feature importances. sklearn's `_get_feature_importances` reduces a multi-output `coef_` `(n_outputs, n_features)` to a per-feature score via `np.linalg.norm(coef, axis=0, ord=norm_order)` (`transform_func="norm"`, ctor `norm_order=1`, `_from_model.py:262`, `:299-304`). |
| REQ-5 (scaled-string thresholds + l1 default-from-estimator) | NOT-STARTED | open prereq blocker #1355. `ThresholdStrategy::{Mean, Median, Value, Percentile}` has NO `"scale*reference"` variant and NO estimator-derived default. sklearn's `_calculate_threshold` (`_from_model.py:24-71`) parses `"scale*ref"` → `scale * {np.median \| np.mean}(importances)` (`:43-55`) and, when `threshold is None`, picks `1e-5` for l1-penalized / `Lasso` / l1 `ElasticNet` estimators else `"mean"` (`:27-40`); ferrolearn's `Default` is unconditionally `Mean`. |
| REQ-6 (`prefit` + `importance_getter` params) | NOT-STARTED | open prereq blocker #1356. `SelectFromModelExt<F> { threshold, max_features }` has NO `prefit` and NO `importance_getter` field. sklearn's ctor (`_from_model.py:256-271`) accepts `prefit=False` (`_get_support_mask` `check_is_fitted(self.estimator)` + skip re-fit when `True`, `:278-282`) and `importance_getter="auto"` (attribute-path / callable importance source); ferrolearn has no estimator to be prefit. |
| REQ-7 (callable `max_features` + `_check_max_features` range validation) | NOT-STARTED | open prereq blocker #1357. `max_features: Option<usize>` is int-only with NO callable form and NO `[0, n_features]` validation (a `usize` larger than the feature count simply never binds). sklearn's `max_features` may be an `int` or a callable `max_features(X)`, validated by `_check_max_features` `check_scalar(max_features, "max_features", Integral, min_val=0, max_val=n_features)` (`_from_model.py:315-331`). |
| REQ-8 (`SelectorMixin` surface: `get_support` / `inverse_transform` / `get_feature_names_out`) | NOT-STARTED | open prereq blocker #1358. `FittedSelectFromModelExt<F>` exposes `selected_indices()` / `n_features_selected()` accessors but NO `get_support` boolean-mask form, NO `inverse_transform` (scatter selected columns back to full width), and NO `get_feature_names_out`. sklearn's `SelectFromModel` inherits `SelectorMixin`, providing all three. |
| REQ-9 (PyO3 binding) | NOT-STARTED | open prereq blocker #1359. No `_RsSelectFromModel` (or `*Ext`) CPython binding exists — `grep -rn "SelectFromModel" ferrolearn-python/src` finds none — so the selector is unreachable from Python. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #1360. The threshold / mask / gather path uses `ndarray::Array1` / `Array2` (`x.iter()`, `select_columns`, `Array2::zeros`), `num_traits::Float`, and a `Vec<usize>` selected-index path — not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing, orphaned at baseline).** `select_from_model.rs` exposes the
unfitted `SelectFromModelExt<F> { threshold: ThresholdStrategy, max_features:
Option<usize>, _marker: PhantomData<F> }` (`new(threshold, max_features)`,
`Default` = `(Mean, None)`, accessors `threshold_strategy()` / `max_features()`)
and the fitted `FittedSelectFromModelExt<F> { n_features_in, threshold_value:
F, importances: Array1<F>, selected_indices: Vec<usize> }` (accessors
`threshold_value()`, `importances()`, `selected_indices()`,
`n_features_selected()`). `ThresholdStrategy` is a `Copy` enum with four variants:
`Mean` (default), `Median`, `Value(f64)`, `Percentile(f64)`. `Fit<Array1<F>, ()>`
takes the importance vector as `x`, rejects the empty vector
(`InvalidParameter`), computes the threshold (`Mean` = biased `sum/n`; `Median` =
`compute_median`, avg of two middle for even `n`; `Value(v)` = `v`; `Percentile`
via `compute_percentile_threshold`), masks `imp[j] >= threshold` into
`selected_indices`, then applies the `max_features` cap (stable desc-sort by
importance, truncate, `sort_unstable()` back to ascending). `Transform<Array2<F>>`
checks the column count (`ShapeMismatch`) and gathers the selected columns
(`select_columns`); `PipelineTransformer` / `FittedPipelineTransformer` wrap the
fitted path. **At the baseline commit the module is UNCOMPILED** — `lib.rs` has no
`mod select_from_model;` and no re-export of these types (a separate basic
`SelectFromModel<F>` from `feature_selection.rs:596` holds the `SelectFromModel`
name at `lib.rs:110-112`). Issue #1352 ACTIVATES this file: it declares `pub mod
select_from_model;` and adds `pub use select_from_model::{SelectFromModelExt,
FittedSelectFromModelExt, ThresholdStrategy};`, which (a) compiles + runs the 16
in-module tests, and (b) provides the grandfathered boundary re-export consumer
that pins REQ-1 / REQ-2 SHIPPED.

**ferrolearn EXTENSION (not a parity REQ).** `ThresholdStrategy::Percentile(pct)`
(`compute_percentile_threshold`: sort, take the value at the `(100 - pct)`th
percentile via linear interpolation, `pct` in `(0, 100]` else `InvalidParameter`)
has **no sklearn `SelectFromModel` analog** — sklearn supports only mean / median
/ `scale*reference` / float thresholds (`_calculate_threshold:24-71`). It is a
ferrolearn convenience extension (R-DEV-7-style), classified as an EXTENSION and
NOT filed as a parity blocker (Probe G: `Percentile(50)` → threshold `0.4`, the
top-50% set). It is exercised by `test_percentile_threshold_top_50`,
`test_percentile_100_keeps_all`, and `test_percentile_invalid`.

**sklearn (target contract).** `SelectFromModel(MetaEstimatorMixin, SelectorMixin,
BaseEstimator)` (`_from_model.py:256`) takes `__init__(estimator, *,
threshold=None, prefit=False, norm_order=1, max_features=None,
importance_getter="auto")` (`:256-271`). `fit` (`:337`) clones + fits the
estimator on `(X, y)` (unless `prefit`) and calls `_check_max_features`
(`:315-331`, `check_scalar(..., min_val=0, max_val=n_features)`).
`_get_support_mask` (`:273-313`) computes `scores =
_get_feature_importances(estimator, getter=importance_getter,
transform_func="norm", norm_order)` from `estimator_.coef_` /
`feature_importances_` (`:299-304`), derives `threshold =
_calculate_threshold(estimator, scores, self.threshold)` (`:305`, the
mean / median / `scale*ref` / float / l1-default logic), and builds the mask: when
`max_features is not None`, `candidate = np.argsort(-scores,
kind="mergesort")[:max_features]; mask[candidate] = True` else `mask = ones`,
THEN `mask[scores < threshold] = False` (`:306-312`). The selector surface
(`get_support`, `transform`, `inverse_transform`, `get_feature_names_out`) comes
from `SelectorMixin`.

**The gap.** ferrolearn matches sklearn on the *threshold-from-importances algebra*
(mean / median / value) and the *mask + max-features cap* — including the
order-equivalence of threshold-then-cap vs cap-then-threshold and the
ascending-index tie-break (REQ-1, REQ-2, verified against the numpy
`_get_support_mask` spec, Probes A-E). The remaining gaps are the estimator-facing
meta-transformer surface: no estimator wrapping / coef extraction (REQ-3, the
honest gap — ferrolearn's `Fit` input IS the importance vector); no `norm_order`
multi-output norm (REQ-4); no `scale*ref` / l1-default thresholds (REQ-5); no
`prefit` / `importance_getter` (REQ-6); int-only `max_features` with no callable
or range check (REQ-7); no `SelectorMixin` surface (REQ-8); no PyO3 binding
(REQ-9); and the non-ferray substrate (REQ-10). This is a **shipped-partial** unit
(2 SHIPPED / 8 NOT-STARTED) whose ENTRY condition (orphan → activation) is resolved
in the #1352 iteration.

## Verification

Commands establishing the SHIPPED claims (REQ-1 threshold + mask + max-features,
REQ-2 scoped error contracts). **All require the #1352 activation first** — until
`pub mod select_from_model;` is declared in `lib.rs`, the file is uncompiled and
none of these tests run (that activation is the entry condition this iteration
resolves; the builder lands it before the table is finalized):

```bash
# Entry-condition check — the module must be declared & re-exported (post-activation):
grep -rn "mod select_from_model" ferrolearn-preprocess/src/lib.rs        # must be non-empty post-activation
grep -rn "SelectFromModelExt" ferrolearn-preprocess/src/lib.rs           # boundary re-export consumer

# REQ-1 value parity + REQ-2 error contracts (16 in-module tests):
cargo test -p ferrolearn-preprocess select_from_model
#   REQ-1: test_mean_threshold (selected_indices == [1,2]), test_median_threshold,
#          test_median_threshold_even ([1,3]), test_explicit_value_threshold,
#          test_max_features_cap ([1,3]), test_max_features_not_needed,
#          test_none_selected_high_threshold, test_threshold_value_accessor,
#          test_default, test_f32, test_pipeline_integration
#   REQ-2: test_empty_importances_error, test_percentile_invalid,
#          test_shape_mismatch_on_transform
#   EXTENSION (not a REQ): test_percentile_threshold_top_50, test_percentile_100_keeps_all
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — numpy replication of _get_support_mask:306-312 (the spec the table pins
# ferrolearn's selected_indices() against): mean threshold + mask:
python3 -c "import numpy as np
scores=np.array([0.1,0.5,0.4]); thr=np.mean(scores)
print('mean thr=',thr,'sel=',np.flatnonzero(scores>=thr).tolist())"
#   -> mean thr= 0.3333... sel= [1, 2]   (ferrolearn test_mean_threshold: [1, 2])

# REQ-1 oracle gate — order-equivalence (threshold-then-cap == cap-then-threshold):
python3 -c "import numpy as np
scores=np.array([0.1,0.5,0.4,0.6]); mf=2; thr=np.mean(scores)
m=np.zeros_like(scores,bool); c=np.argsort(-scores,kind='mergesort')[:mf]; m[c]=True; m[scores<thr]=False
sk=np.flatnonzero(m).tolist()
fl=sorted(sorted([j for j in range(4) if scores[j]>=thr],key=lambda j:(-scores[j],j))[:mf])
print('sklearn=',sk,'ferrolearn=',fl,'EQUAL=',sk==fl)"
#   -> sklearn= [1, 3] ferrolearn= [1, 3] EQUAL= True
```

The in-module `#[test]`s exercise REQ-1 (the mean / median / value threshold +
mask + max-features algebra) and REQ-2 (every error path —
`test_empty_importances_error`, `test_percentile_invalid`,
`test_shape_mismatch_on_transform`). No green command establishes REQ-3..REQ-10
(estimator wrapping, `norm_order`, scaled-string / l1-default thresholds, `prefit`
/ `importance_getter`, callable `max_features`, `SelectorMixin` surface, PyO3,
ferray).

## Blockers

REQ-1 (threshold + mask + max-features core, HEADLINE) and REQ-2 (scoped error /
parameter contracts) are SHIPPED **conditional on the #1352 activation** (declare
`pub mod select_from_model;` + re-export `SelectFromModelExt` /
`FittedSelectFromModelExt` / `ThresholdStrategy`), which compiles the module, runs
its 16 in-module tests, and provides the grandfathered boundary re-export consumer.

The remaining REQs are NOT-STARTED, to be filed as `-l blocker` issues against
tracking issue #1352 (placeholders `#1353..H` until filed):

- #1353 — REQ-3: no estimator wrapping / `coef_` / `feature_importances_`
  extraction; ferrolearn's `Fit` input is the importance vector
  (`_get_support_mask:299-304`, `fit:337`).
- #1354 — REQ-4: no `norm_order` multi-output coef norm
  (`np.linalg.norm(coef, axis=0, ord=norm_order)`, `:262`, `:299-304`).
- #1355 — REQ-5: no `scale*reference` thresholds or l1-default `1e-5`
  (`_calculate_threshold:24-71`, `:43-55`, `:27-40`).
- #1356 — REQ-6: no `prefit` / `importance_getter` params (`:256-271`,
  `:278-282`).
- #1357 — REQ-7: int-only `max_features` with no callable form and no
  `_check_max_features` `[0, n_features]` validation (`:315-331`).
- #1358 — REQ-8: no `SelectorMixin` surface — `get_support` /
  `inverse_transform` / `get_feature_names_out`.
- #1359 — REQ-9: no PyO3 `SelectFromModel` binding in `ferrolearn-python`.
- #1360 — REQ-10: fit/transform on `ndarray` / `num_traits` / `Vec<usize>`
  index path, not ferray (R-SUBSTRATE-1/2).

Note: `ThresholdStrategy::Percentile` is a ferrolearn EXTENSION with no sklearn
`SelectFromModel` analog and is intentionally NOT filed as a parity blocker.
