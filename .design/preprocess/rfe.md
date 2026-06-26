# RFE / RFECV

<!--
tier: 3-component
status: draft
baseline-commit: 58cf2824
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/feature_selection/_rfe.py  # class RFE(_RoutingNotSupportedMixin, SelectorMixin, MetaEstimatorMixin, BaseEstimator) (:72); __init__(estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter="auto") (:214-227); _parameter_constraints {estimator:[HasMethods(["fit"])], n_features_to_select:[None, Interval(RealNotInt,0,1,closed="right"), Interval(Integral,0,None,closed="neither")], step:[Interval(Integral,0,None,closed="neither"), Interval(RealNotInt,0,1,closed="neither")], verbose:["verbose"], importance_getter:[str, callable]} (:199-212); fit(X, y, **fit_params) -> _fit (:247-268); _fit(X, y, step_score=None, **fit_params) (:270-361): _validate_data(accept_sparse="csc", ensure_min_features=2, force_all_finite=False, multi_output=True) (:275-282); n_features_to_select None -> n_features//2 (:286-287); Integral -> as-is, and if > n_features warnings.warn("Found n_features_to_select=N > n_features=M. There will be no feature selection and all features will be kept.", UserWarning) (:288-297) WARNS-NOT-RAISE keeps-all; float -> int(n_features * frac) (:298-299); step float (0,1) -> int(max(1, step*n_features)) else int(step) (:301-304); support_=np.ones(n_features,bool) (:306), ranking_=np.ones(n_features,int) (:307); elimination while np.sum(support_) > n_features_to_select (:314): features=arange(n_features)[support_] (:316), estimator=clone(self.estimator) (:319), estimator.fit(X[:,features], y) (:323), importances=_get_feature_importances(estimator, importance_getter, transform_func="square") (:326-330) SQUARES, ranks=np.argsort(importances) (:331), threshold=min(step, np.sum(support_)-n_features_to_select) (:337), support_[features[ranks][:threshold]]=False (:345), ranking_[~support_]+=1 (:346); final estimator_=clone(self.estimator).fit(X[:,features], y) (:350-351), n_features_=support_.sum() (:357), support_ (:358), ranking_ (:359). predict/score/decision_function/predict_proba/predict_log_proba delegate to estimator_ via available_if (:363-469). _get_support_mask -> support_ (:407-409); SelectorMixin supplies get_support/transform/inverse_transform/get_feature_names_out (sklearn/feature_selection/_base.py). class RFECV(RFE) (:485): __init__(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter="auto") (:675-694) DROPS n_features_to_select (:673); fit(X, y, groups=None) (:700-816): _validate_data(ensure_min_features=2,...) (:726-733); cv=check_cv(self.cv, y, classifier=is_classifier(estimator)) (:736); scorer=check_scoring(estimator, scoring) (:737); per-fold _rfe_single_fit(rfe, estimator, X, y, train, test, scorer) (:30-50,:777-780) runs RFE._fit down to min_features_to_select collecting step_scores_/step_n_features_; scores summed across folds reversed (:783-787) and n_features_to_select = step_n_features_rev[argmax(scores_sum_rev)] (:788) lowest-n-on-tie; re-fit RFE over full data (:791-799); cv_results_ {mean_test_score, std_test_score, split{i}_test_score, n_features} (:810-815).
ferrolearn-module: ferrolearn-preprocess/src/rfe.rs
parity-ops: RFE, RFECV
crosslink-issue: 1294
-->

## Summary

scikit-learn's `RFE` (`_rfe.py:72`) is a recursive-feature-elimination wrapper
around an **unfitted estimator**: it re-fits the estimator on the surviving
feature subset at every elimination round, extracts per-feature importances via
`_get_feature_importances(estimator, importance_getter, transform_func="square")`
(`:326-330` — note importances are **squared**), ranks them with
`np.argsort(importances)` (`:331`), and drops the `threshold = min(step,
sum(support_) - n_features_to_select)` (`:337`) least-important features per round
until `n_features_to_select` remain (`:314-346`). `RFECV` (`_rfe.py:485`) subclasses
`RFE` and tunes the feature count by **cross-validation**: it runs an `RFE` per CV
fold down to `min_features_to_select`, sums per-feature-count scores across folds
(`:777-788`), and picks the count maximizing the mean CV score (lowest-count on
ties), exposing `cv_results_`. Both expose the `SelectorMixin` surface
(`support_`, `ranking_`, `get_support`, `transform`, `inverse_transform`,
`get_feature_names_out`) plus `estimator_` / `n_features_`.

`ferrolearn-preprocess/src/rfe.rs` ships the **ranking/elimination SHAPE** with a
**fundamentally different importance interface**: `RFE<F> { ranking, support,
selected_indices, n_features_in }` is constructed by `RFE::new(importances:
&Array1<F>, n_features_to_select: usize, step: usize)` from a **pre-computed
static importance vector** — there is NO wrapped estimator, NO per-round re-fit, NO
`_get_feature_importances` / squaring. The elimination loop in `RFE::new` sorts the
remaining features ascending by the (fixed) importance, removes `step.min(remaining
- n_features_to_select)` per round, and accumulates ranking (selected → 1,
last-removed-round → 2, earlier rounds → higher), mirroring sklearn's
`support_`/`ranking_` evolution GIVEN static per-round importances. `RFECV<F> { rfe,
cv_scores, optimal_n_features }` (`RFECV::new(importances, cv_scores: &[f64],
step)`) likewise takes **pre-computed per-feature-count CV scores** and picks
`optimal_n_features = argmax(cv_scores) + 1` (first-max / lowest-count on ties) —
the cross-validation machinery (cv splitter, scoring, per-fold RFE, `cv_results_`)
is entirely absent. Both expose accessors (`ranking()`, `support()`,
`selected_indices()`, `n_features_selected()`; RFECV adds `cv_scores()` /
`optimal_n_features()`) and a `Transform` that gathers the selected columns.
Non-test consumer: the crate re-export `pub use rfe::{RFE, RFECV};`
(`ferrolearn-preprocess/src/lib.rs`, the boundary public API). There is **no PyO3
binding**.

**Headline finding (the structural gap, REQ-3).** The two APIs are not 1:1. The
RANKING/ELIMINATION SHAPE matches given static importances (REQ-1 SHIPPED), the
scoped transform + error contracts ship (REQ-2 SHIPPED), and RFECV's
optimal-count argmax matches sklearn's lowest-count-on-tie selection given static
scores (REQ-9 SHIPPED). But the **importance SOURCE** — an unfitted estimator
**re-fit each round**, with importances pulled from `coef_`/`feature_importances_`
and **squared** (`_get_feature_importances(..., transform_func="square")`,
`:326-330`) — is **entirely absent**: ferrolearn takes a fixed importance vector.
Everything sklearn layers on the estimator+CV machinery (the re-fit loop REQ-3,
the `n_features_to_select=None`/float defaults REQ-5, float `step` REQ-6,
`importance_getter`/`verbose` REQ-7, the RFECV internal cross-validation REQ-8,
and the `estimator_`/`n_features_` attrs plus delegation residuals in REQ-10)
remains absent. The scoped dense `SelectorMixin` helpers in REQ-10 now ship via
`crate::SelectorMixin`. This is a **mostly-NOT-STARTED** unit: 5 SHIPPED
(REQ-1/2/4/9 plus scoped REQ-10, REQ-4 fixed #1296 this iteration) / 7
NOT-STARTED, with REQ-10 residual open.

**One boundary divergence fixed (DIV-1, REQ-4 — #1296).** sklearn `RFE` with
`n_features_to_select > n_features` does NOT raise — it **warns** (`UserWarning`)
and **keeps all features**, because the `while np.sum(support_) >
n_features_to_select` loop never runs (`:288-297`,`:314`). ferrolearn now clamps
the requested count and keeps all features too; there is no Rust warning surface.
Live oracle (Probe DIV-1): `RFE(LinearRegression(), n_features_to_select=5).fit(X_2feat, y)`
does NOT raise (warns), `support_ == [True, True]`, `ranking_ == [1, 1]`; ferrolearn
`RFE::new(&[0.5, 0.3], 5, 1)` returns `Ok` with the same keep-all support/ranking.

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-3 — the IMPORTANCE SOURCE is an unfitted estimator RE-FIT each round, importances
# SQUARED from coef_/feature_importances_ (_get_feature_importances transform_func="square",
# :326-330). RFE(LinearRegression(), n_features_to_select=2, step=1) on a 6x4 fixture:
python3 -c "import numpy as np; \
from sklearn.feature_selection import RFE; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1,5.],[2.,20.,0.2,4.],[3.,30.,0.3,3.],[4.,40.,0.4,2.],[5.,50.,0.5,1.],[6.,5.,0.6,9.]]); \
y=np.array([1.,2.,3.,4.,5.,6.]); \
r=RFE(LinearRegression(), n_features_to_select=2, step=1).fit(X,y); \
print('ranking_', r.ranking_.tolist(), 'support_', r.support_.tolist(), 'n_features_', int(r.n_features_))"
# -> ranking_ [1, 2, 3, 1] support_ [True, False, False, True] n_features_ 2
#    The importances driving this ranking are RE-COMPUTED from a refit LinearRegression at each
#    round and SQUARED. ferrolearn::RFE::new takes a FIXED importance vector — no estimator/refit.

# REQ-4 (DIV-1) — n_features_to_select > n_features WARNS and KEEPS ALL (the while loop never runs,
# :288-297, :314). Does NOT raise:
python3 -c "import warnings, numpy as np; \
from sklearn.feature_selection import RFE; \
from sklearn.linear_model import LinearRegression; \
X2=np.array([[1.,10.],[2.,20.],[3.,30.],[4.,40.]]); y=np.array([1.,2.,3.,4.]); \
w=warnings.catch_warnings(record=True); w.__enter__(); warnings.simplefilter('always'); \
r=RFE(LinearRegression(), n_features_to_select=5).fit(X2,y); \
print('raised? NO. support_', r.support_.tolist(), 'ranking_', r.ranking_.tolist())"
# -> raised? NO. support_ [True, True] ranking_ [1, 1]
#    Warning: 'Found n_features_to_select=5 > n_features=2. There will be no feature selection
#    and all features will be kept.'
#    ferrolearn::RFE::new now keeps all features too; no Rust warning surface.

# REQ-8 — RFECV runs cross-validation INTERNALLY (cv splitter, per-fold RFE over a step grid,
# cv_results_), picking the feature count that maximizes the mean CV score:
python3 -c "import numpy as np; \
from sklearn.feature_selection import RFECV; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1,5.],[2.,20.,0.2,4.],[3.,30.,0.3,3.],[4.,40.,0.4,2.],[5.,50.,0.5,1.],[6.,5.,0.6,9.]]); \
y=np.array([1.,2.,3.,4.,5.,6.]); \
r=RFECV(LinearRegression(), cv=2).fit(X,y); \
print('n_features_', int(r.n_features_), 'support_', r.support_.tolist(), \
'cv_results_ keys', sorted(r.cv_results_.keys()))"
# -> n_features_ 2 support_ [True, False, False, True]
#    cv_results_ keys ['mean_test_score', 'n_features', 'split0_test_score', 'split1_test_score', 'std_test_score']
#    ferrolearn::RFECV::new takes PRE-COMPUTED cv_scores: &[f64] — NO cv splitter, NO per-fold RFE,
#    NO scoring, NO cv_results_.
```

## Requirements

- REQ-1: **RFE ranking / support / elimination given static importances** — from
  a fixed per-feature importance vector, repeatedly sort the surviving features
  ascending by importance, drop `min(step, sum(support) - n_features_to_select)`
  least-important per round, and accumulate ranking (selected → 1, last-removed
  round → 2, earlier rounds → higher), producing `support_` (bool mask) and
  `ranking_`. Mirrors sklearn's `_fit` elimination loop (`:314-346`): `threshold =
  min(step, np.sum(support_) - n_features_to_select)` (`:337`), `support_[features[
  ranks][:threshold]] = False` (`:345`), `ranking_[~support_] += 1` (`:346`) — GIVEN
  static per-round importances. **Scope: importances are user-supplied; no estimator
  / re-fit / squaring (that is REQ-3).** Supports `f32`/`f64`.
- REQ-2: **RFE transform + error contracts (scoped)** — `InvalidParameter` on empty
  `importances`, `step == 0`, and `n_features_to_select == 0`; `ShapeMismatch` on a
  `transform` column-count mismatch; `Transform` returns the selected columns. (The
  analogous sklearn validation lives in `_validate_data`, `:275-282`, and the
  `_parameter_constraints` `step`/`n_features_to_select` intervals, `:199-212`.)
  **FLAG:** ferrolearn's `n_features_to_select > n_features` guard is DIV-1 — sklearn
  WARNS and keeps all features there (`:288-297`), it does not raise; that boundary is
  REQ-4, not this row.
- REQ-3: **Wrapped estimator + per-round re-fit + `_get_feature_importances`** (the
  load-bearing structural gap, HEADLINE) — clone + `fit` the wrapped unfitted
  estimator on the surviving feature subset each round (`:319-323`), pull per-feature
  importances from `coef_`/`feature_importances_` via `_get_feature_importances(
  estimator, importance_getter, transform_func="square")` (importances **squared**,
  `:326-330`), and `ranks = np.argsort(importances)` (`:331`). ferrolearn's
  `RFE::new(importances, ...)` instead takes a **fixed** importance vector — no
  estimator, no per-round re-fit, no squaring (Probe REQ-3).
- REQ-4: **`n_features_to_select > n_features` = warn + keep-all** (DIV-1,
  SHIPPED / closed #1296) — sklearn does NOT raise: it emits a `UserWarning` and
  keeps all features (the `while np.sum(support_) > n_features_to_select` loop
  never runs, `:288-297`,`:314`). ferrolearn now clamps the requested count and
  keeps all features too; the only residual difference is that Rust has no warning
  facade here.
- REQ-5: **`n_features_to_select=None` default + float fraction** (`:286-287`,
  `:298-299`) — `None` selects `n_features // 2`; a float in `(0, 1]` selects `int(
  n_features * frac)` (`Interval(RealNotInt, 0, 1, closed="right")`, `:203`).
  ferrolearn's `n_features_to_select: usize` is a required absolute count with no
  `None`/fraction mode.
- REQ-6: **Float `step` in (0, 1)** (`:301-302`) — a float step selects `int(max(1,
  step * n_features))` features to remove per round (`Interval(RealNotInt, 0, 1,
  closed="neither")`, `:208`). ferrolearn's `step: usize` is an absolute count only.
- REQ-7: **`importance_getter` + `verbose` parameters** (`:211`,`:210`,`:112-127`,
  `:320-321`) — `importance_getter` ("auto" / attribute-path str / callable) selects
  which fitted attribute supplies importances (`:326-330`); `verbose` controls
  progress printing (`:320-321`). Both absent in ferrolearn (the importance vector is
  passed directly).
- REQ-8: **RFECV internal cross-validation** (`:485`,`:700-816`) — wrap an unfitted
  estimator and run an `RFE` per CV fold (`check_cv`, `:736`; `_rfe_single_fit`,
  `:30-50`,`:777-780`) over a step grid down to `min_features_to_select`, score each
  feature count with `scorer` (`check_scoring`, `:737`), sum across folds reversed
  (`:783-787`), pick `n_features_to_select = step_n_features_rev[argmax(scores_sum_rev)]`
  (`:788`, lowest-count on ties), and expose `cv_results_` (`:810-815`), with `cv` /
  `scoring` / `n_jobs` / `min_features_to_select` parameters. ferrolearn's
  `RFECV::new(importances, cv_scores, step)` takes **pre-computed** per-count CV
  scores — no cv splitter, no per-fold RFE, no scoring, no `cv_results_` (Probe
  REQ-8).
- REQ-9: **RFECV optimal-count selection given static scores** — from a vector of
  per-feature-count CV scores, pick `optimal_n_features = argmax(cv_scores) + 1` with
  first-max (lowest-count) tie-break, and delegate ranking/support to an inner
  `RFE` built with that count. Mirrors sklearn's count-selection semantics
  (`scores_sum_rev[::-1]` + `np.argmax`, `:786-788`), whose reversal picks the LOWEST
  feature count on ties — identical to ferrolearn's first-max-on-ascending-count.
  **Scope: the scores are user-supplied; no per-fold CV produces them (that is REQ-8).**
- REQ-10: **`SelectorMixin` surface + `estimator_` / `n_features_` fitted attrs**
  (`:407-409`,`:350-359`; `SelectorMixin` in `sklearn/feature_selection/_base.py`) —
  `get_support()` / `get_support(indices=True)`, `inverse_transform` (zero-pad dropped
  columns), `get_feature_names_out`, plus the fitted `estimator_` (the refit estimator
  on selected features, `:350-351`) and `n_features_ = support_.sum()` (`:357`), and
  the `predict`/`score`/`decision_function`/`predict_proba` delegation (`:363-469`).
  ferrolearn exposes `support()` (bool mask) and `selected_indices()` (the
  `get_support(indices=True)` analog) and `transform`, but not `inverse_transform` /
  `get_feature_names_out` / `estimator_` / the delegating predict surface.
- REQ-11: **PyO3 binding** — `import ferrolearn` exposing registered `RFE` / `RFECV`
  marshalling `fit`/`transform`, the project boundary CPython consumer. Absent (no
  `ferrolearn-python` reference to RFE).
- REQ-12: **ferray substrate** — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::{Array1, Array2}` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `RFE::<f64>::new(&array![0.6, 0.3, 0.1], 1, 1)` yields `ranking() ==
  [1, 2, 3]`, `support() == [true, false, false]`, `selected_indices() == [0]`
  (feature 2 dropped round 1 → rank 2, feature 1 round 2 → rank 3); `new(&imp, 2, 1)`
  drops only the least-important (rank 2); `new(&array![0.5,0.3,0.2,0.1], 2, 2)`
  removes 2 at once. Pinned by `test_rfe_basic_ranking`, `test_rfe_select_two`,
  `test_rfe_step_two`, `test_rfe_all_features_selected`.
- AC-2 (REQ-2): empty `importances` → `Err(InvalidParameter)`; `step == 0` → `Err`;
  `n_features_to_select == 0` → `Err`; a `transform` column-count mismatch → `Err(
  ShapeMismatch)`; `transform` on selected `[0]` returns column 0. Pinned by
  `test_rfe_empty_importances_error`, `test_rfe_zero_step_error`,
  `test_rfe_n_features_zero_error`, `test_rfe_shape_mismatch_error`,
  `test_rfe_transform`.
- AC-3 (REQ-3): a fitted handle re-fits the wrapped estimator each round and ranks by
  `_get_feature_importances(..., transform_func="square")` (`:326-331`), reproducing
  the Probe REQ-3 result `ranking_ [1, 2, 3, 1]`, `support_ [True, False, False,
  True]` for `LinearRegression()`, `n_features_to_select=2`, `step=1` on the 6x4
  fixture.
- AC-4 (REQ-4): `RFE::new(&imp_2feat, 5, 1)` does NOT error — it keeps all features
  (`support == [true, true]`, `ranking == [1, 1]`) and surfaces a keep-all warning,
  matching sklearn (Probe REQ-4 / DIV-1) — NOT the current `Err(InvalidParameter)`.
- AC-5 (REQ-5): `n_features_to_select=None` selects `n_features // 2`; a float `0.5`
  on 4 features selects `int(4 * 0.5) = 2` (`:286-287`,`:298-299`).
- AC-6 (REQ-6): a float `step=0.5` on 4 features removes `int(max(1, 0.5*4)) = 2` per
  round (`:301-302`).
- AC-7 (REQ-7): `importance_getter="auto"` pulls `coef_`/`feature_importances_`; an
  attribute-path str / callable overrides it; `verbose>0` prints progress (`:320-321`,
  `:326-330`).
- AC-8 (REQ-8): `RFECV(LinearRegression(), cv=2)` on the 6x4 fixture reproduces the
  Probe REQ-8 result `n_features_ 2`, `support_ [True, False, False, True]`, and
  populates `cv_results_` with `mean_test_score` / `std_test_score` /
  `split{i}_test_score` / `n_features` (`:810-815`).
- AC-9 (REQ-9): `RFECV::<f64>::new(&array![0.5,0.3,0.2], &[0.85, 0.95, 0.90], 1)`
  selects `optimal_n_features() == 2` (argmax at index 1); a leading-max `&[0.9, 0.8]`
  selects 1 (first-max), and a trailing tie resolves to the lowest count, matching
  sklearn's reversed-argmax lowest-count tie-break (`:786-788`). Pinned by
  `test_rfecv_selects_optimal`, `test_rfecv_cv_scores_accessor`,
  `test_rfecv_ranking_and_support`.
- AC-10 (REQ-10): a fitted handle exposes `get_support()`, `inverse_transform`
  (zero-pad dropped columns), `get_feature_names_out`, `estimator_`, and `n_features_`
  alongside `selected_indices()`; `predict`/`score` delegate to `estimator_`.
- AC-11 (REQ-11): `python3 -c "import ferrolearn; ..."` resolves registered
  `RFE`/`RFECV`; `.fit(X, y).transform(X)` matches the Probe REQ-3 selection.
- AC-12 (REQ-12): the ranking/elimination path computes on `ferray-core` arrays.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (RFE ranking/support/elimination given static importances) | SHIPPED | impl `pub fn new in rfe.rs`: the `while remaining.len() > n_features_to_select` loop sorts `remaining` ascending by `imp` (`remaining.sort_by(|&a, &b| imp[a].partial_cmp(&imp[b]))`), removes `n_to_remove = step.min(remaining.len() - n_features_to_select)` least-important (`removed = remaining[..n_to_remove]`), and records each elimination round; ranking is then assigned selected → 1, last-removed round → 2, earlier rounds → higher (`for (round_idx, round) in elimination_rounds.iter().rev().enumerate() { ranking[idx] = round_idx + 2 }`), and `support` derives from `ranking == 1`. This mirrors sklearn `_fit` (`_rfe.py:337` `threshold = min(step, np.sum(support_) - n_features_to_select)`, `:345` `support_[features[ranks][:threshold]] = False`, `:346` `ranking_[~support_] += 1`) GIVEN static per-round importances. Doctest pins `ranking() == [1, 2, 3]` / `support() == [true, false, false]` for `[0.6, 0.3, 0.1]`. Generic `F: Float + Send + Sync + 'static` covers `f32`/`f64`. **Scope (R-HONEST-3):** importances are the user-supplied vector, NOT estimator-refit-and-squared (that is REQ-3). Non-test consumer: crate re-export `pub use rfe::{RFE, RFECV};` (`ferrolearn-preprocess/src/lib.rs`), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess rfe` (`test_rfe_basic_ranking`, `test_rfe_select_two`, `test_rfe_step_two`, `test_rfe_all_features_selected`) → 18 passed, 0 failed. |
| REQ-2 (RFE transform + error contracts, scoped) | SHIPPED | `pub fn new in rfe.rs` returns `Err(FerroError::InvalidParameter { name:"importances", reason:"importance vector must not be empty" })` when `n_features == 0`; `Err(InvalidParameter { name:"step", reason:"step must be at least 1" })` when `step == 0`; `Err(InvalidParameter { name:"n_features_to_select", .. })` when `n_features_to_select == 0`. `impl Transform<Array2<F>> for RFE` returns `Err(FerroError::ShapeMismatch { context:"RFE::transform", .. })` when `x.ncols() != self.n_features_in`, else `select_columns(x, &self.selected_indices)`. The analogous sklearn validation is `_validate_data(accept_sparse="csc", ensure_min_features=2, ...)` (`:275-282`) + the `step`/`n_features_to_select` `_parameter_constraints` (`:199-212`). Non-test consumer: the error path guards every instance reached through the crate re-export (`lib.rs`, `pub use rfe::{RFE, RFECV}`). Verification: `cargo test -p ferrolearn-preprocess rfe` (`test_rfe_empty_importances_error`, `test_rfe_zero_step_error`, `test_rfe_n_features_zero_error`, `test_rfe_shape_mismatch_error`, `test_rfe_transform`) → green. **Scoped + DIV flag:** the `n_features_to_select > n_features` arm of the same guard is DIV-1 — sklearn WARNS + keeps-all there (`:288-297`), it does not raise; that boundary is REQ-4, not this row. |
| REQ-3 (wrapped estimator + per-round re-fit + `_get_feature_importances`) | NOT-STARTED | open prereq blocker #1295. **The load-bearing structural gap.** `RFE::new(importances: &Array1<F>, n_features_to_select, step)` takes a FIXED importance vector and the elimination loop reads that single vector every round. sklearn `_fit` clones + re-fits the wrapped unfitted `estimator` on the surviving feature subset each round (`estimator = clone(self.estimator)` `:319`, `estimator.fit(X[:, features], y)` `:323`), then recomputes `importances = _get_feature_importances(estimator, self.importance_getter, transform_func="square")` (importances SQUARED, `:326-330`) and `ranks = np.argsort(importances)` (`:331`). ferrolearn has NO estimator, NO per-round re-fit, NO squaring — the importance source is entirely absent (Probe REQ-3: `LinearRegression()`, `n_features_to_select=2` → `ranking_ [1, 2, 3, 1]`). |
| REQ-4 (`n_features_to_select > n_features` = warn + keep-all; DIV-1) | SHIPPED (closed #1296) | `RFE::new` now errors ONLY on `n_features_to_select == 0`; the `> n_features` arm was removed and replaced with `let n_features_to_select = n_features_to_select.min(n_features);` (clamp), so the `while remaining.len() > n_features_to_select` loop is a no-op and ALL features are kept — matching sklearn `_fit` warn-and-keep-all (`_rfe.py:290-297`,`:314`; the `UserWarning` has no Rust analog, no log facade). Live oracle (R-CHAR-3): `RFE(LinearRegression(), n_features_to_select=5).fit(X_2feat, y)` → no raise, `support_ [True, True]`, `ranking_ [1, 1]`; ferrolearn `RFE::new(&[0.5,0.3], 5, 1)` → Ok, `support()==[true,true]`, `ranking()==[1,1]`. In-module `test_rfe_n_features_too_large_error` rewritten to `test_rfe_n_features_too_large_keeps_all` (R-HONEST-4). Guard `divergence_rfe_n_features_to_select_gt_n_features_keeps_all` + 4 clamp-boundary guards (==n, ==n+1, 100-of-4, valid-count-unperturbed, zero-still-errors) PASS. Two-round critic-verified CLEAN. Consumer: re-export lib.rs. |
| REQ-5 (`n_features_to_select=None` default + float fraction) | NOT-STARTED | open prereq blocker #1297. `RFE { .. }` has `n_features_to_select: usize` as a required absolute count (`new(importances, n_features_to_select: usize, step)`); there is no `None`/fraction mode. sklearn resolves `None → n_features // 2` (`:286-287`) and a float in `(0, 1]` → `int(n_features * self.n_features_to_select)` (`:298-299`, `Interval(RealNotInt, 0, 1, closed="right")`, `:203`). |
| REQ-6 (float `step` in (0, 1)) | NOT-STARTED | open prereq blocker #1298. `new(.., step: usize)` is an absolute removal count (`n_to_remove = step.min(..)`). sklearn accepts a float step in `(0, 1)` and computes `step = int(max(1, self.step * n_features))` (`:301-302`, `Interval(RealNotInt, 0, 1, closed="neither")`, `:208`). |
| REQ-7 (`importance_getter` + `verbose`) | NOT-STARTED | open prereq blocker #1299. `RFE::new` has only `importances` / `n_features_to_select` / `step`; there is no `importance_getter` (sklearn "auto"/attr-path str/callable selecting `coef_` vs `feature_importances_`, `:211`,`:112-127`,`:326-330`) and no `verbose` (progress printing, `:210`,`:320-321`) — both are moot because the importance vector is supplied directly (REQ-3). |
| REQ-8 (RFECV internal cross-validation) | NOT-STARTED | open prereq blocker #1300. `RFECV::new(importances, cv_scores: &[f64], step)` takes PRE-COMPUTED per-feature-count CV scores. sklearn `RFECV.fit` builds `cv = check_cv(self.cv, y, classifier=...)` (`:736`) and `scorer = check_scoring(...)` (`:737`), runs `_rfe_single_fit` per fold (an RFE `_fit` down to `min_features_to_select` collecting `step_scores_`/`step_n_features_`, `:30-50`,`:777-780`), sums scores across folds (`:783-787`), picks `n_features_to_select = step_n_features_rev[argmax(scores_sum_rev)]` (`:788`), and populates `cv_results_` (`:810-815`). ferrolearn has NO cv splitter, NO per-fold RFE, NO scoring, NO `cv_results_` — the CV machinery is absent (Probe REQ-8: `cv=2` → `n_features_ 2`, `cv_results_` keys present). |
| REQ-9 (RFECV optimal-count selection given static scores) | SHIPPED | impl `pub fn new in rfe.rs` (RFECV): scans `cv_scores` keeping `best_idx` under strict `if score > best_score` (first-max), sets `optimal_n_features = best_idx + 1`, and delegates to `RFE::new(importances, optimal_n_features, step)`. The strict-`>` first-max over ascending feature count picks the LOWEST count on ties — identical to sklearn's `n_features_to_select = step_n_features_rev[np.argmax(scores_sum_rev)]` (`:786-788`), where the `[::-1]` reversal makes `np.argmax` resolve ties to the lowest feature count (verified: reversed-argmax on `[0.85, 0.95, 0.95]` → lowest of the two tied counts). Accessors `optimal_n_features()` / `cv_scores()` plus delegated `ranking()`/`support()`/`selected_indices()`. **Scope (R-HONEST-3):** the scores are user-supplied, NOT per-fold-CV-produced (that is REQ-8). Non-test consumer: crate re-export `pub use rfe::{RFE, RFECV};` (`lib.rs`). Verification: `cargo test -p ferrolearn-preprocess rfe` (`test_rfecv_selects_optimal`, `test_rfecv_cv_scores_accessor`, `test_rfecv_ranking_and_support`) → green. |
| REQ-10 (SelectorMixin surface + `estimator_`/`n_features_` attrs) | SHIPPED scoped / residual open | `ferrolearn_preprocess::SelectorMixin` is implemented for `RFE<F>` and `RFECV<F>`, providing dense `get_support()`, `get_support_indices()`, `inverse_transform` zero-fill, and `get_feature_names_out`, matching sklearn `SelectorMixin` on dense arrays (`sklearn/feature_selection/_base.py:54,136,176`). Verification: `cargo test -p ferrolearn-preprocess --test divergence_selector_mixin`. Residual blocker #1301 remains for the fitted `estimator_`, sklearn `n_features_` attr, prediction/score delegation, sparse/pandas output, and Python fitted-state protocol. |
| REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1302. No `ferrolearn-python` registration of `RFE`/`RFECV` (grep `RFE`/`rfe` across `ferrolearn-python/` matches only `README.md`); the only non-test consumer is the crate re-export (`lib.rs`, `pub use rfe::{RFE, RFECV}`). The boundary CPython `import ferrolearn` selector surface is absent. |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1303. The ranking/elimination/transform path uses `ndarray::{Array1, Array2}` (`importances.len()`, `Array2::zeros`, the `select_columns` gather) + `num_traits::Float` and `Vec` bookkeeping — not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `rfe.rs` exposes two unfitted-shaped types and a private
`select_columns<F>(x, indices) -> Array2<F>` column-gather helper. `RFE<F> { ranking:
Vec<usize>, support: Vec<bool>, selected_indices: Vec<usize>, n_features_in: usize,
_marker: PhantomData<F> }` is constructed by `RFE::new(importances: &Array1<F>,
n_features_to_select: usize, step: usize) -> Result<Self, FerroError>`, which
validates `n_features == 0` / `step == 0` / `n_features_to_select == 0 ||
n_features_to_select > n_features` (`InvalidParameter`; the `> n_features` arm is
DIV-1), then runs the elimination simulation: while `remaining.len() >
n_features_to_select`, sort `remaining` ascending by the fixed `imp`, drop
`step.min(remaining.len() - n_features_to_select)` least-important, and accumulate
ranking (selected → 1, last-removed round → 2, earlier rounds → higher). Accessors
`ranking()` / `support()` / `selected_indices()` / `n_features_selected()`; `impl
Transform<Array2<F>>` returns `ShapeMismatch` on a column-count mismatch else the
selected columns. `RFECV<F> { rfe: RFE<F>, cv_scores: Vec<f64>, optimal_n_features:
usize }` is constructed by `RFECV::new(importances: &Array1<F>, cv_scores: &[f64],
step: usize)`, which validates empty `importances` / `cv_scores.len() != n_features`
(`InvalidParameter`), picks `optimal_n_features = argmax(cv_scores) + 1` (first-max),
and builds an inner `RFE::new(importances, optimal_n_features, step)`. It exposes
`cv_scores()` / `optimal_n_features()` plus the RFE accessors (delegated) and a
`Transform` that forwards to the inner RFE. The crate re-exports both
(`lib.rs`, `pub use rfe::{RFE, RFECV}`); there is no PyO3 binding.

**sklearn (target contract).** `RFE(_RoutingNotSupportedMixin, SelectorMixin,
MetaEstimatorMixin, BaseEstimator)` (`:72`) takes `__init__(estimator, *,
n_features_to_select=None, step=1, verbose=0, importance_getter="auto")` (`:214-227`)
under `_parameter_constraints` requiring `estimator` to have `fit`
(`HasMethods(["fit"])`, `:200`). `_fit(X, y)` (`:270`) validates via `_validate_data(
accept_sparse="csc", ensure_min_features=2, force_all_finite=False, multi_output=
True)` (`:275-282`); resolves `n_features_to_select` from `None` (`n_features // 2`,
`:286-287`), an `Integral` (as-is, WARNING + keep-all if `> n_features`, `:288-297`),
or a float (`int(n_features * frac)`, `:298-299`); resolves a float `step` to `int(
max(1, step * n_features))` (`:301-302`). The elimination loop (`:314-346`) re-fits
`clone(self.estimator)` on the surviving features each round (`:319-323`), pulls
squared importances (`_get_feature_importances(..., transform_func="square")`,
`:326-330`), `ranks = np.argsort` (`:331`), drops `threshold = min(step, sum(support_)
- n_features_to_select)` features (`:337`,`:345`), and bumps `ranking_[~support_]`
(`:346`). Final attributes: `estimator_ = clone(self.estimator).fit(X[:, features], y)`
(`:350-351`), `n_features_ = support_.sum()` (`:357`), `support_` (`:358`), `ranking_`
(`:359`). `SelectorMixin` supplies `get_support`/`transform`/`inverse_transform`/
`get_feature_names_out`; `predict`/`score`/`decision_function`/`predict_proba` delegate
to `estimator_` via `available_if` (`:363-469`). `RFECV(RFE)` (`:485`) takes
`__init__(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None,
verbose=0, n_jobs=None, importance_getter="auto")` (`:675-694`, dropping
`n_features_to_select`, `:673`); `fit(X, y, groups=None)` (`:700`) builds `cv =
check_cv(...)` (`:736`) and `scorer = check_scoring(...)` (`:737`), runs
`_rfe_single_fit` per fold (`:30-50`,`:777-780`), sums fold scores reversed (`:783-787`),
picks the count maximizing the sum (`step_n_features_rev[argmax(scores_sum_rev)]`,
`:788`, lowest count on ties), re-fits an RFE over the full data (`:791-799`), and
exposes `cv_results_` (`:810-815`).

**The structural gap.** ferrolearn matches sklearn on the *ranking/elimination shape*
(REQ-1: the ascending-sort + threshold-removal + ranking accumulation reproduces
`support_`/`ranking_` evolution), the scoped transform/error contracts (REQ-2), and
RFECV's *optimal-count argmax* given static scores (REQ-9: first-max = sklearn's
reversed-argmax lowest-count tie-break). But the *importance source* (REQ-3) is
fundamentally different: ferrolearn takes a fixed importance vector, whereas sklearn
**re-fits a cloned estimator each round and squares the extracted `coef_`/
`feature_importances_`** (`:319-330`). Every contract sklearn layers on the
estimator+CV machinery is therefore NOT-STARTED: the `None`/float `n_features_to_select`
defaults (REQ-5), float `step` (REQ-6), `importance_getter`/`verbose` (REQ-7), the
RFECV internal cross-validation (REQ-8), the `estimator_` / `n_features_` attrs
and delegation residuals in REQ-10, the PyO3 binding (REQ-11), the ferray
substrate (REQ-12). The scoped dense `SelectorMixin` helpers in REQ-10 now ship
via `crate::SelectorMixin`.
The `n_features_to_select > n_features` boundary (REQ-4, DIV-1) is fixed by
#1296: ferrolearn now clamps and keeps all features (without sklearn's warning
surface). This is a **mostly-NOT-STARTED** unit (5 SHIPPED / 7 NOT-STARTED, with
REQ-10 residual open; REQ-4 fixed #1296).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-2, REQ-9):

```bash
# Crate gauntlet — REQ-1 (ranking/elimination shape), REQ-2 (transform + error contracts),
# REQ-9 (RFECV optimal-count argmax):
cargo test -p ferrolearn-preprocess rfe   # incl. test_rfe_basic_ranking,
                                          #       test_rfe_select_two,
                                          #       test_rfe_step_two,
                                          #       test_rfe_all_features_selected,
                                          #       test_rfe_empty_importances_error,
                                          #       test_rfe_zero_step_error,
                                          #       test_rfe_n_features_zero_error,
                                          #       test_rfe_shape_mismatch_error,
                                          #       test_rfe_transform,
                                          #       test_rfecv_selects_optimal,
                                          #       test_rfecv_cv_scores_accessor,
                                          #       test_rfecv_ranking_and_support,
                                          #       test_rfecv_transform,
                                          #       test_rfecv_mismatched_scores_error,
                                          #       test_rfecv_empty_importances_error
#   -> 18 passed; 0 failed (16 unit + 2 doctests).
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# Oracle (Probe REQ-3) — the estimator-refit + SQUARED-importance source ferrolearn does NOT model:
python3 -c "import numpy as np; \
from sklearn.feature_selection import RFE; \
from sklearn.linear_model import LinearRegression; \
X=np.array([[1.,10.,0.1,5.],[2.,20.,0.2,4.],[3.,30.,0.3,3.],[4.,40.,0.4,2.],[5.,50.,0.5,1.],[6.,5.,0.6,9.]]); \
y=np.array([1.,2.,3.,4.,5.,6.]); \
print(RFE(LinearRegression(), n_features_to_select=2, step=1).fit(X,y).ranking_.tolist())"
#   -> [1, 2, 3, 1]   (re-fit-driven, squared importances; ferrolearn's fixed-vector RFE::new is a different interface)

# Oracle (Probe REQ-4 / DIV-1) — n_features_to_select > n_features WARNS + keeps-all (does NOT raise):
python3 -c "import warnings, numpy as np; \
from sklearn.feature_selection import RFE; \
from sklearn.linear_model import LinearRegression; \
X2=np.array([[1.,10.],[2.,20.],[3.,30.],[4.,40.]]); y=np.array([1.,2.,3.,4.]); \
w=warnings.catch_warnings(record=True); w.__enter__(); warnings.simplefilter('always'); \
r=RFE(LinearRegression(), n_features_to_select=5).fit(X2,y); \
print('support_', r.support_.tolist())"
#   -> support_ [True, True]   (ferrolearn::RFE::new now keeps all; no Rust warning surface)
```

The existing `#[test]`s exercise REQ-1 (ranking/elimination shape, step, all-features),
REQ-2 (every error path + transform), REQ-4 (warn-equivalent keep-all boundary),
and REQ-9 (optimal-count + accessors) with fixed
importance / cv-score fixtures; they are **vector-grounded, not estimator-refit
oracle-grounded** — by construction, since the importance interface diverges (REQ-3).
No currently-green command establishes REQ-3, REQ-5..REQ-8, REQ-10 residuals, or
REQ-11..REQ-12. `cargo test -p ferrolearn-preprocess --test divergence_selector_mixin`
establishes the scoped dense `SelectorMixin` helpers in REQ-10.

## Blockers

REQ-1, REQ-2, REQ-4, REQ-9, and scoped REQ-10 are SHIPPED (ranking/elimination shape,
scoped transform/error contracts, keep-all boundary, RFECV optimal-count argmax, and
dense `SelectorMixin` helpers). The remaining NOT-STARTED REQs are open
`-l blocker` issues referenced by the REQ status table:

- #1295 — REQ-3 (HEADLINE): `RFE::new(importances, ...)` takes a FIXED importance
  vector; sklearn re-fits `clone(estimator)` each round and recomputes squared
  importances via `_get_feature_importances(..., transform_func="square")`
  (`_rfe.py:319-331`). The importance SOURCE (estimator + per-round re-fit + squaring)
  is entirely absent.
- #1296 — REQ-4 (DIV-1, CLOSED/fixed): `RFE::new` now clamps
  `n_features_to_select > n_features` and keeps all features, matching sklearn's
  warn-and-keep-all outcome (`UserWarning` has no Rust warning facade;
  `:288-297`,`:314`).
- #1297 — REQ-5: no `n_features_to_select=None` (`n_features // 2`, `:286-287`) or float
  fraction (`int(n_features * frac)`, `:298-299`) — the count is a required `usize`.
- #1298 — REQ-6: `step: usize` is an absolute count; no float step `int(max(1,
  step * n_features))` (`:301-302`).
- #1299 — REQ-7: no `importance_getter` (`:211`,`:326-330`) or `verbose` (`:210`,`:320-321`).
- #1300 — REQ-8: `RFECV::new(importances, cv_scores, step)` takes pre-computed CV
  scores; sklearn runs cross-validation internally (`check_cv` `:736`, per-fold RFE
  `:30-50`,`:777-780`, `cv_results_` `:810-815`) with `cv`/`scoring`/`n_jobs`/
  `min_features_to_select`.
- #1301 — REQ-10 residual: scoped dense `SelectorMixin` helpers now ship via
  `crate::SelectorMixin` (`get_support` / `get_support_indices` /
  `inverse_transform` / `get_feature_names_out`). Residual parity still lacks
  `estimator_`, sklearn-named `n_features_`, `predict`/`score` delegation,
  sparse/pandas output, and Python fitted-state protocol (`:350-359`,`:363-469`,
  `:407-409`).
- #1302 — REQ-11: no `ferrolearn-python` `RFE`/`RFECV` binding (boundary CPython
  consumer absent; grep matches only `README.md`).
- #1303 — REQ-12: ranking/elimination/transform path on `ndarray`/`num_traits`/`Vec`,
  not ferray (R-SUBSTRATE-1/2).
