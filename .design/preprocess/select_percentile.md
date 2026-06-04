# SelectPercentile

<!--
tier: 3-component
status: draft
baseline-commit: 2686196c
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/feature_selection/_univariate_selection.py  # class SelectPercentile(_BaseFilter) (:589); __init__(score_func=f_classif, *, percentile=10) (:665-667); _parameter_constraints {percentile:[Interval(Real,0,100,closed="both")]} + inherited {score_func:[callable]} (:660-663,:536); _get_support_mask (:669-686): if percentile==100 -> all True (:673-674), elif percentile==0 -> all False (:675-676), else scores=_clean_nans(scores_) (:678), threshold=np.percentile(scores,100-percentile) (:679), mask = scores > threshold STRICT (:680), ties=np.where(scores==threshold)[0] (:681), if len(ties): max_feats=int(len(scores)*percentile/100) FLOOR (:683), kept_ties=ties[:max_feats-mask.sum()] ASCENDING fill (:684), mask[kept_ties]=True (:685). _more_tags requires_y=False (:688-689). _BaseFilter.fit (:542): score_func_ret=self.score_func(X,y); if tuple -> scores_,pvalues_ else scores_,pvalues_=None (:567-573); scores_=np.asarray (:575). f_classif (:127): X,y=check_X_y(accept_sparse) (:171), args=[X[y==k] for k in np.unique(y)] (:172), return f_oneway(*args) (:173). f_oneway (:43): ssbn/sswn one-way ANOVA, msb=ssbn/dfbn, msw=sswn/dfwn, f=msb/msw, prob=special.fdtrc(dfbn,dfwn,f) (:92-117). _clean_nans (:24): scores[np.isnan]=np.finfo(dtype).min (:31-33). Selector surface (get_support/inverse_transform/get_feature_names_out) inherited from SelectorMixin (sklearn/feature_selection/_base.py).
ferrolearn-module: ferrolearn-preprocess/src/select_percentile.rs
parity-ops: SelectPercentile
crosslink-issue: 1273
-->

## Summary

scikit-learn's `SelectPercentile` (`_univariate_selection.py:589`) is a univariate
feature-selection filter that scores each feature with a `score_func` (default
`f_classif`, the ANOVA F-test, `:127`/`:43`) and keeps the features whose score
falls in the top `percentile` percent. Its selection rule (`_get_support_mask`,
`:669-686`) is **threshold-based**: it cleans NaN scores to `dtype.min`
(`_clean_nans`, `:24`,`:678`), computes `threshold = np.percentile(scores,
100 - percentile)` (`:679`), keeps features with `scores > threshold` (STRICT,
`:680`), then **fills threshold-equal ties** up to `int(len(scores) *
percentile / 100)` (FLOOR via `int()`, `:683-685`) in ascending index order.
It stores `scores_` and `pvalues_` (`:567-573`) and inherits the `SelectorMixin`
surface (`get_support`, `transform`, `inverse_transform`,
`get_feature_names_out`).

`ferrolearn-preprocess/src/select_percentile.rs` ships a **dense, `f_classif`-only,
rank-top-k** selector with the unfitted/fitted split: `SelectPercentile<F>
{ percentile: usize, score_func: ScoreFunc }` (`new(percentile, score_func)`,
`Default = (10, FClassif)`, accessors `percentile()`/`score_func()`; `ScoreFunc`
has ONLY `FClassif`) and `FittedSelectPercentile<F> { n_features_in, scores:
Array1<F>, selected_indices: Vec<usize> }` (accessors `scores()`,
`selected_indices()`). `fit` computes per-feature ANOVA F via the local
`anova_f_scores` helper, then selects via sklearn's `_get_support_mask` rule
(#1274): `0/100` short-circuits, else `threshold = numpy_percentile(scores,
100 - percentile)` (local linear-interpolation helper), `scores > threshold`
(strict), and `int()`-floor ascending tie-fill — kept indices sorted ascending.
`transform` returns only the selected columns (via `select_columns`). Non-test consumer: the
crate re-export `pub use select_percentile::{FittedSelectPercentile,
SelectPercentile};` (`ferrolearn-preprocess/src/lib.rs` line 136) — the boundary
public API. There is **no PyO3 binding**.

**Scope note (this file owns ONE estimator).** The translate route
(`tooling/translate-routes.toml`) lists `parity_ops = ["SelectPercentile",
"SelectFpr", "SelectFdr", "SelectFwe"]` for this crate pattern, but `SelectFpr` /
`SelectFdr` / `SelectFwe` are translated in the **sibling**
`ferrolearn-preprocess/src/stat_selectors.rs` (its own route at the next
`crate_pattern`, design `.design/preprocess/stat_selectors.md`). This document
covers **only `SelectPercentile`** (and its `f_classif`/`f_oneway`/`_clean_nans`
dependencies); `parity-ops` in the front-matter is scoped to `SelectPercentile`
accordingly.

**Headline finding (DIV-1, REQ-2) — FIXED this iteration (#1274):** ferrolearn's
selection WAS **`ceil` rank-top-k**; it now replicates sklearn's
**`np.percentile`-threshold + strict `>` + `int()`-floor tie-fill** rule
(`_get_support_mask`, `:669-686`), agreeing on the `0`/`100` short-circuits AND at
intermediate percentiles. The pinned divergence was `pct=50` on the 5-feature
fixture: sklearn `[0, 3]` (2 features: `scores > median`), ferrolearn previously
`[0, 1, 3]` (`k=ceil(2.5)=3`) — now `[0, 3]`. The ANOVA F-scores (REQ-1) and the
scoped error contracts (REQ-3) also match. Verified bit/value-CLEAN across fresh
stress fixtures (8-feature non-round percentiles, exact tie-fill, both
interpolation branches, f32).

## Probes (live sklearn oracle, 1.5.2)

```bash
# REQ-1 — ANOVA F-score value match (f_classif, finite scores) + REQ-2 selection divergence.
# Pinned 5-feature fixture (the DIV-1 carrier).
python3 -c "import numpy as np; from sklearn.feature_selection import f_classif, SelectPercentile; \
X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]); \
y=np.array([0,0,1,1]); F,p=f_classif(X,y); \
print('scores', [round(v,6) for v in F.tolist()]); \
print([(pct, SelectPercentile(percentile=pct).fit(X,y).get_support(indices=True).tolist()) for pct in [0,5,10,30,40,50,60,100]])"
# -> scores [450.0, 0.5, 0.002069, 2040.2, 0.0]
# -> [(0, []), (5, [3]), (10, [3]), (30, [0, 3]), (40, [0, 3]), (50, [0, 3]), (60, [0, 1, 3]), (100, [0, 1, 2, 3, 4])]
#    REQ-1: ferrolearn anova_f_scores reproduces [450.0, 0.5, 0.002069, 2040.2, 0.0]
#           (same one-way ANOVA ss_between/(k-1) over ss_within/(n-k) as f_oneway :108-113).
#    REQ-2 DIV-1: at pct=50, sklearn -> [0, 3] (2 feats, scores>median(0.5));
#           ferrolearn k=ceil(5*50/100)=ceil(2.5)=3, rank desc [3,0,1,2,4], top-3 {0,1,3} -> [0,1,3]. DIVERGENCE.
#           pct in {10,30,40,60} happen to match on THIS fixture; the 0/100 short-circuits also match
#           (ceil(5)=5, ceil(0)=0), but the intermediate-percentile threshold-vs-ceil gap is real.

# REQ-2 — confirm sklearn's selection is np.percentile-threshold + strict '>' (NOT ceil rank-top-k):
python3 -c "import numpy as np; \
F=np.array([450.0,0.5,0.002069,2040.2,0.0]); \
thr=np.percentile(F,100-50); print('thr=median', thr); \
print('mask scores>thr', (F>thr).tolist(), 'count', int((F>thr).sum())); \
print('max_feats=int(5*50/100)', int(5*50/100))"
# -> thr=median 0.49999999999973355
# -> mask scores>thr [True, False, False, True, False] count 2   (strict '>' excludes feature 1 at 0.5)
# -> max_feats=int(5*50/100) 2   (no ties to fill: 2 strict-gt == max_feats) -> support [0,3]
#    ferrolearn would keep 3 (ceil), including feature 1. DIVERGENCE.

# REQ-2 — TIE-FILL path (the int()-floor ascending tie-fill, :683-685):
python3 -c "import numpy as np; from sklearn.feature_selection import SelectPercentile, f_classif; \
X=np.array([[1.,1.,0.,4.9],[1.2,1.2,0.1,5.0],[3.,3.,9.,5.1],[3.3,3.3,9.2,4.8]]); \
y=np.array([0,0,1,1]); F,p=f_classif(X,y); \
print('tie scores', [round(v,4) for v in F.tolist()]); \
s=SelectPercentile(percentile=50).fit(X,y); thr=np.percentile(F,50); \
print('pct50 support', s.get_support(indices=True).tolist(), 'thr', round(float(thr),4), \
      'strict_gt', int((F>thr).sum()), 'max_feats', int(4*50/100))"
# -> tie scores [129.3077, 129.3077, 6552.2, 0.0]
# -> pct50 support [0, 2] thr 129.3077 strict_gt 1 max_feats 2
#    threshold=median=129.3077; strict '>' keeps only feature 2 (count 1); ties=={0,1};
#    fill max_feats-mask.sum()=2-1=1 tie ASCENDING -> feature 0 -> [0,2].
#    ferrolearn here ALSO gives [0,2] (k=ceil(2)=2, desc rank [2,0,1,3] index-tie-break, top-2 {0,2})
#    -> the tie-fill MATCHES on this fixture, but is coincidental: sklearn's rule is threshold+int()-floor,
#       NOT ceil rank-top-k (the pct=50 5-feature fixture proves they are different rules).

# REQ-4 — _clean_nans: NaN score -> dtype.min (NOT skipped), then threshold over cleaned scores (:24,:678):
python3 -c "import numpy as np; from sklearn.feature_selection._univariate_selection import _clean_nans; \
print(_clean_nans(np.array([1.0, np.nan, 3.0])).tolist())"
# -> [1.0, -1.7976931348623157e+308, 3.0]   (NaN -> np.finfo(float64).min, sorts to the bottom)
#    ferrolearn: no _clean_nans; anova_f_scores can emit F::infinity() (ms_within==0) and the
#    sort_by(partial_cmp -> Ordering::Equal) does NOT map NaN to the smallest value before ranking.

# REQ-5 — score_func is a CALLABLE (default f_classif; chi2 / f_regression / mutual_info_* accepted):
python3 -c "from sklearn.feature_selection import SelectPercentile, chi2; \
from sklearn.datasets import load_digits; X,y=load_digits(return_X_y=True); \
print(SelectPercentile(chi2, percentile=10).fit_transform(X,y).shape)"
# -> (1797, 7)   (chi2 score_func, NOT f_classif)
#    ferrolearn: ScoreFunc enum has ONLY FClassif; no chi2/f_regression/mutual_info_*.

# REQ-6 — SelectorMixin surface: get_support(indices), inverse_transform, get_feature_names_out:
python3 -c "import numpy as np; from sklearn.feature_selection import SelectPercentile; \
X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]); \
y=np.array([0,0,1,1]); s=SelectPercentile(percentile=50).fit(X,y); \
print('get_support', s.get_support().tolist()); \
print('inverse_transform', s.inverse_transform(s.transform(X)).tolist()[0]); \
print('names_out', s.get_feature_names_out().tolist())"
# -> get_support [True, False, False, True, False]
# -> inverse_transform [1.0, 0.0, 0.0, 4.0, 0.0]   (zero-pads dropped columns)
# -> names_out ['x0', 'x3']
#    ferrolearn: exposes selected_indices() (and transform) but NO boolean get_support / inverse_transform /
#    get_feature_names_out.

# REQ-7 — scores_ AND pvalues_ fitted attributes (score_func returns (F, pval) tuple, :567-573):
python3 -c "import numpy as np; from sklearn.feature_selection import SelectPercentile; \
X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]); \
y=np.array([0,0,1,1]); s=SelectPercentile(percentile=50).fit(X,y); \
print('scores_', [round(v,4) for v in s.scores_.tolist()]); \
print('pvalues_', [round(v,6) for v in s.pvalues_.tolist()])"
# -> scores_ [450.0, 0.5, 0.0021, 2040.2, 0.0]
# -> pvalues_ [0.002215, 0.552786, 0.96785, 0.00049, 1.0]
#    ferrolearn: FittedSelectPercentile stores scores only (scores()); NO pvalues_.

# REQ-8 — percentile is Interval(Real,0,100,closed="both") -> accepts FLOAT, rejects <0 / >100 (:662):
python3 -c "from sklearn.feature_selection import SelectPercentile; import numpy as np; \
X=np.array([[1.,2.],[3.,4.]]); y=np.array([0,1]); \
print('float pct ok', SelectPercentile(percentile=33.3).fit(X,y).get_support().tolist())"
# -> float pct ok [True, False]   (sklearn accepts a Real percentile)
#    ferrolearn: percentile is usize (integer only); rejects >100 (InvalidParameter) but cannot express 33.3.
```

## Requirements

- REQ-1: ANOVA F-score value match (`f_classif`, finite scores) — compute a
  per-feature one-way ANOVA F-statistic `F = (ss_between / (n_classes - 1)) /
  (ss_within / (n_samples - n_classes))` matching sklearn `f_classif` (`:127`)
  → `f_oneway` (`:43`, `msb=ssbn/dfbn`, `msw=sswn/dfwn`, `f=msb/msw`, `:108-113`)
  on finite, non-degenerate input; store the per-feature scores (Probe REQ-1:
  `[450.0, 0.5, 0.002069, 2040.2, 0.0]`). Supports `f32` and `f64`.
- REQ-2: **Selection mask `_get_support_mask`** (`:669-686`) — keep the features
  `scores > np.percentile(scores, 100 - percentile)` (STRICT `>`, `:679-680`),
  with the `percentile==100 → all`, `percentile==0 → none` short-circuits
  (`:673-676`) and the `int()`-FLOOR ascending tie-fill up to `int(len(scores) *
  percentile / 100)` (`:683-685`). **DIV-1 (the fixable divergence):** ferrolearn
  selects `k = ceil(n_features * percentile / 100)` and rank-top-k by score
  descending — a different rule that diverges from the threshold rule at
  intermediate percentiles (Probe REQ-1/REQ-2: `pct=50` → sklearn `[0,3]`,
  ferrolearn `[0,1,3]`).
- REQ-3: Error contracts (scoped) — `InsufficientSamples` on zero rows in `fit`,
  `ShapeMismatch` on `y.len() != n_rows` in `fit` and on a column-count mismatch
  in `transform`, `InvalidParameter` on `percentile > 100`. (sklearn validates
  `percentile ∈ [0, 100]` via `Interval(Real, 0, 100, closed="both")`, `:662`.)
- REQ-4: `_clean_nans` (`:24`,`:678`) — before thresholding, replace NaN scores
  with `np.finfo(dtype).min` (so NaN-scored features sort to the bottom and are
  never selected) (Probe REQ-4). ferrolearn's `anova_f_scores` can emit
  `F::infinity()`/`F::zero()` and ranks via `partial_cmp -> Ordering::Equal`
  without a `_clean_nans` pre-pass.
- REQ-5: Pluggable `score_func` callable (`:596-599`,`:665`) — default `f_classif`
  but accepting any callable returning `(scores, pvalues)` or scores, including
  `chi2`, `f_regression`, `mutual_info_classif`, `mutual_info_regression`
  (Probe REQ-5: `chi2` on digits → `(1797, 7)`). ferrolearn's `ScoreFunc` enum has
  only `FClassif`.
- REQ-6: `SelectorMixin` surface — boolean `get_support()` / `get_support(indices=True)`,
  `inverse_transform` (zero-pad dropped columns), `get_feature_names_out`
  (Probe REQ-6). ferrolearn exposes `selected_indices()` and `transform` but not
  the boolean mask / inverse / feature-name surface.
- REQ-7: `scores_` AND `pvalues_` fitted attributes (`:567-573`) — when the
  `score_func` returns a `(F, pval)` tuple (as `f_classif` does), store BOTH
  (Probe REQ-7). ferrolearn stores `scores` only; no `pvalues_`.
- REQ-8: `percentile` as `Interval(Real, 0, 100, closed="both")` (`:662`) — accept
  a real-valued percentile (e.g. `33.3`) and reject `< 0` / `> 100` as `ValueError`
  (Probe REQ-8). ferrolearn's `percentile` is `usize` (integer only; rejects `>100`
  but cannot express a fractional percentile).
- REQ-9: PyO3 binding — `import ferrolearn` exposing a `SelectPercentile`
  marshalling `fit`/`transform`, the project boundary CPython consumer. Absent.
- REQ-10: ferray substrate — compute over `ferray-core` arrays / `ferray-ufunc`
  rather than `ndarray::Array2` + `num_traits::Float` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): `SelectPercentile::<f64>::new(_, FClassif).fit(&X,&y)` for
  `X=[[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]`,
  `y=[0,0,1,1]` yields `scores() == [450.0, 0.5, 0.002069, 2040.2, 0.0]` within
  ULP tolerance (Probe REQ-1, the live `f_classif` output). Pinned by an
  oracle-grounded `#[test]`.
- AC-2 (REQ-2): on the Probe REQ-1 fixture at `percentile=50`,
  `selected_indices() == [0, 3]` (sklearn `_get_support_mask`), NOT the current
  `[0, 1, 3]`; the `0/100` short-circuits keep none / all; an oracle-grounded
  divergence `#[test]` asserts `[0, 3]` and fails against the current `ceil`
  rank-top-k.
- AC-3 (REQ-3): `fit` on `(0, n)` returns `Err(InsufficientSamples)`; `y.len() !=
  n_rows` returns `Err(ShapeMismatch)`; a column-count mismatch on `transform`
  returns `Err(ShapeMismatch)`; `percentile > 100` returns
  `Err(InvalidParameter)` (pinned by the existing
  `test_select_percentile_zero_rows_error`,
  `test_select_percentile_y_length_mismatch_error`,
  `test_select_percentile_shape_mismatch_on_transform`,
  `test_select_percentile_over_100_error`).
- AC-4 (REQ-4): a fit whose scores include a NaN treats the NaN-scored feature as
  the lowest (never selected at `percentile < 100`), reproducing `_clean_nans`
  (`np.finfo(dtype).min`, Probe REQ-4).
- AC-5 (REQ-5): `SelectPercentile::new(10, Chi2)` (and `FRegression`, etc.) selects
  the chi2-top features, reproducing the Probe REQ-5 `(1797, 7)` digits shape.
- AC-6 (REQ-6): a fitted handle exposes `get_support()` (`[true,false,false,true,false]`),
  `get_support_indices()` (`[0,3]`), `inverse_transform` (zero-pad to
  `[1,0,0,4,0]`), and `get_feature_names_out` (`['x0','x3']`) per Probe REQ-6.
- AC-7 (REQ-7): a fitted handle exposes `pvalues()` equal to
  `[0.002215, 0.552786, 0.96785, 0.00049, 1.0]` (Probe REQ-7) alongside `scores()`.
- AC-8 (REQ-8): `SelectPercentile::new_real(33.3, ...)` accepts a fractional
  percentile and reproduces the Probe REQ-8 mask; `< 0` / `> 100` returns
  `Err(InvalidParameter)`.
- AC-9 (REQ-9): `python3 -c "import ferrolearn; ..."` resolves a registered
  `SelectPercentile`; `.fit(X, y).transform(X)` matches the Probe REQ-1 selection.
- AC-10 (REQ-10): the score + selection path computes on `ferray-core` arrays.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ANOVA F-score value match, finite) | SHIPPED | impl `pub fn fit in select_percentile.rs` (`Fit<Array2<F>, Array1<usize>> for SelectPercentile<F>`) calls the local `fn anova_f_scores` which, per feature, computes `grand_mean`, `ss_between = Σ n_k·(class_mean - grand_mean)²`, `ss_within = Σ (x - class_mean)²`, and `f = (ss_between / (n_classes-1)) / (ss_within / (n_samples-n_classes))` — the exact one-way ANOVA of sklearn `f_oneway` (`_univariate_selection.py:108-113`, `msb=ssbn/dfbn`, `msw=sswn/dfwn`, `f=msb/msw`) reached via `f_classif` (`:127`,`:171-173`); the result is stored in `FittedSelectPercentile.scores` (accessor `scores()`). On the pinned fixture the live `f_classif` returns `[450.0, 0.5, 0.002069, 2040.2, 0.0]` (Probe REQ-1) and ferrolearn's `anova_f_scores` reproduces it (identical textbook ANOVA). Edge handling matches sklearn's degenerate behavior in spirit: `df_between==0`/`df_within==0` → `0`, `ms_within==0` → `F::infinity()` (cf. `f_oneway` `f=msb/msw` → `inf` on a zero within-group variance). Non-test consumer: crate re-export `pub use select_percentile::{FittedSelectPercentile, SelectPercentile};` (`ferrolearn-preprocess/src/lib.rs` line 136), the boundary public API (grandfathered S5/R-DEFER-1). Verification: `cargo test -p ferrolearn-preprocess` (`test_select_percentile_scores_stored`, `test_select_percentile_selects_highest_scoring`). |
| REQ-2 (selection mask `_get_support_mask`) | SHIPPED (closed #1274) | `fit` now replicates sklearn `_get_support_mask` (`:669-686`): `percentile==100 → all`, `percentile==0 → none` short-circuits; else a local `numpy_percentile` helper (linear interpolation, same idiom as `robust_scaler.rs` `quantile_sorted`) computes `threshold = numpy_percentile(scores, 100 - percentile)`, `mask = scores > threshold` (STRICT), then ascending tie-fill of `scores == threshold` up to `(n*percentile/100) as usize` (`int()`-floor). Replaced the `ceil` rank-top-k. Live oracle (R-CHAR-3): the 5-feature fixture pct=0/10/30/50/60/100 → `[]`/`[3]`/`[0,3]`/`[0,3]`/`[0,1,3]`/`[0,1,2,3,4]` (pct=50 was `[0,1,3]`). Two-round critic-verified bit/value-CLEAN across fresh stress fixtures: 8-feature non-round percentiles (25/33/75/90), exact ascending tie-fill (`int()`-floor partial fill), interpolation lo==hi vs between-points branches, f32 path. Guards `divergence_select_percentile_mask_pct50` + `_full_oracle` + 5 stress guards PASS. In-module `test_select_percentile_selects_highest_scoring` re-grounded against a finite-score live oracle (R-HONEST-4, the old fixture hit the inf-score REQ-4 path). Consumer: re-export lib.rs:136. `#[allow(clippy::float_cmp, reason=...)]` on the threshold comparisons (mirrors numpy `scores == threshold`). |
| REQ-3 (InsufficientSamples / ShapeMismatch / InvalidParameter error contracts) | SHIPPED | `Fit::fit` returns `Err(FerroError::InsufficientSamples { required:1, actual:0, context:"SelectPercentile::fit" })` when `n_samples==0`; `Err(FerroError::ShapeMismatch { context:"SelectPercentile::fit — y must have same length as x rows" })` when `y.len() != n_samples`; `Err(FerroError::InvalidParameter { name:"percentile", .. })` when `self.percentile > 100`. `Transform::transform` returns `Err(FerroError::ShapeMismatch { context:"FittedSelectPercentile::transform" })` when `x.ncols() != self.n_features_in`. Non-test consumer: the error path guards every fitted instance reached through the crate re-export (`lib.rs` line 136). Verification: `cargo test -p ferrolearn-preprocess` (`test_select_percentile_zero_rows_error`, `test_select_percentile_y_length_mismatch_error`, `test_select_percentile_shape_mismatch_on_transform`, `test_select_percentile_over_100_error`). Scoped: sklearn validates `percentile ∈ [0, 100]` via `Interval(Real, 0, 100, closed="both")` (`:662`) (rejects `< 0` too — ferrolearn's `usize` cannot be negative; the fractional-percentile gap is REQ-8); the 0-rows `InsufficientSamples` and `y`-length `ShapeMismatch` are ferrolearn-side guards on the same inputs sklearn's `check_X_y` validates. |
| REQ-4 (`_clean_nans` NaN → dtype.min before thresholding) | NOT-STARTED | open prereq blocker #1275. There is no `_clean_nans` pre-pass. `fit` ranks features directly via `raw_scores[b].partial_cmp(&raw_scores[a]).unwrap_or(Ordering::Equal)`, so a NaN score sorts as `Equal` (NOT mapped to the smallest value), and `anova_f_scores` can emit `F::infinity()` (a feature with zero within-group variance). sklearn cleans NaN scores to `np.finfo(dtype).min` so they sort to the bottom and are never selected at `percentile < 100` (`_clean_nans` `:24`,`:31-33`; called at `:678` before `np.percentile`). Probe REQ-4 (`[1, NaN, 3] → [1, -1.797e308, 3]`) unavailable. |
| REQ-5 (pluggable `score_func`: chi2 / f_regression / mutual_info_*) | NOT-STARTED | open prereq blocker #1276. `ScoreFunc` (`crate::feature_selection::ScoreFunc`, matched in `fit` as `match self.score_func { ScoreFunc::FClassif => anova_f_scores(x, y) }`) has ONLY the `FClassif` variant — `chi2` (`:202`), `f_regression` (`:405`), `r_regression` (`:300`), `mutual_info_classif`/`mutual_info_regression` are absent, as is the callable-`score_func` polymorphism of sklearn `__init__(score_func=f_classif, ...)` (`:596-599`,`:665`) over `_BaseFilter` (`:531-539`). Probe REQ-5 (`chi2` on digits → `(1797, 7)`) unavailable. |
| REQ-6 (SelectorMixin surface: get_support / inverse_transform / get_feature_names_out) | NOT-STARTED | open prereq blocker #1277. `FittedSelectPercentile<F>` exposes `selected_indices()` (the `get_support(indices=True)` analog) and `transform`, but NOT the boolean `get_support()` mask, `inverse_transform` (zero-pad dropped columns), or `get_feature_names_out` provided by sklearn's `SelectorMixin` (`sklearn/feature_selection/_base.py`, inherited by `_BaseFilter`). Probe REQ-6 (`get_support [T,F,F,T,F]`, `inverse_transform [1,0,0,4,0]`, `names_out ['x0','x3']`) unavailable. |
| REQ-7 (`scores_` AND `pvalues_` fitted attributes) | NOT-STARTED | open prereq blocker #1278. `FittedSelectPercentile<F>` stores `scores: Array1<F>` only (accessor `scores()`); there is no `pvalues_`. sklearn's `_BaseFilter.fit` unpacks the `score_func` tuple `self.scores_, self.pvalues_ = score_func_ret` (`:567-570`), and `f_classif` returns `(f, prob)` with `prob = special.fdtrc(dfbn, dfwn, f)` (`f_oneway` `:116`). ferrolearn's `anova_f_scores` returns the F-statistic only — no F-distribution survival function — so no p-values. Probe REQ-7 (`pvalues_ [0.002215, 0.552786, 0.96785, 0.00049, 1.0]`) unavailable. |
| REQ-8 (`percentile` as `Interval(Real, 0, 100)`: fractional + lower bound) | NOT-STARTED | open prereq blocker #1279. `SelectPercentile<F> { percentile: usize, .. }` is an integer; `new(percentile: usize, ..)` cannot express a fractional percentile (sklearn accepts `Real`, e.g. `33.3`, via `Interval(Real, 0, 100, closed="both")` `:662`). The `> 100` upper bound IS checked (`InvalidParameter`, REQ-3), but the `Real` type and the `closed="both"` lower bound are not modeled. Probe REQ-8 (`percentile=33.3` accepted) unavailable. |
| REQ-9 (PyO3 binding) | NOT-STARTED | open prereq blocker #1280. No `ferrolearn-python` registration of `SelectPercentile` (grep `SelectPercentile`/`select_percentile` across `ferrolearn-python/src/` returns nothing); the only non-test consumer is the crate re-export (`lib.rs` line 136). The boundary CPython `import ferrolearn` selector surface is absent. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #1281. The compute path uses `ndarray::{Array1, Array2}` (`x.column(j)`, `x.ncols()`, `Array2::zeros`) + `num_traits::Float`, with `std::collections::HashMap` for class grouping and `Vec` + `sort_by` for ranking — not `ferray-core` / `ferray-ufunc` (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing).** `select_percentile.rs` exposes the unfitted/fitted pair.
`SelectPercentile<F> { percentile: usize, score_func: ScoreFunc, _marker:
PhantomData<F> }` is constructed by `new(percentile, score_func)` (`Default =
new(10, ScoreFunc::FClassif)`) with accessors `percentile()` and `score_func()`;
`ScoreFunc` (re-exported from `crate::feature_selection`) currently has the single
variant `FClassif`. `FittedSelectPercentile<F> { n_features_in: usize, scores:
Array1<F>, selected_indices: Vec<usize> }` exposes `scores()` and
`selected_indices()`. Two free helpers live in the module: `fn anova_f_scores<F>(x,
y) -> Vec<F>` (per-feature one-way ANOVA F via a `HashMap<class, Vec<row>>`
grouping, `ss_between`/`ss_within`, `f = ms_between / ms_within`) and `fn
select_columns<F>(x, indices) -> Array2<F>` (gather the kept columns).

`impl Fit<Array2<F>, Array1<usize>> for SelectPercentile<F>` rejects `n_samples ==
0` (`InsufficientSamples`), `y.len() != n_samples` (`ShapeMismatch`), and
`percentile > 100` (`InvalidParameter`); computes `raw_scores =
anova_f_scores(x, y)`; then **selects via sklearn's `_get_support_mask` rule**
(#1274): `percentile==100 → all` / `percentile==0 → none` short-circuits, else
`threshold = numpy_percentile(raw_scores, 100 - percentile)` (local
linear-interpolation helper), `mask[j] = raw_scores[j] > threshold` (strict), then
`int()`-floor ascending tie-fill of `raw_scores[j] == threshold` up to
`(n_features * percentile / 100) as usize`, kept indices sorted ascending into
`selected_indices`. `impl Transform<Array2<F>> for
FittedSelectPercentile<F>` returns `ShapeMismatch` on a column-count mismatch and
otherwise returns `select_columns(x, &self.selected_indices)`. The generic bound
`F: Float + Send + Sync + 'static` supports `f32`/`f64`. The crate re-exports both
public types (`lib.rs` line 136); there is no PyO3 binding and no `feature_scoring`
/ pipeline wiring of this estimator beyond the doc-link in `feature_scoring.rs`.

**sklearn (target contract).** `SelectPercentile(_BaseFilter)` (`:589`) takes
`__init__(score_func=f_classif, *, percentile=10)` (`:665-667`) under
`_parameter_constraints {score_func:[callable], percentile:[Interval(Real, 0, 100,
closed="both")]}` (`:660-663`,`:536`). `_BaseFilter.fit` (`:542`) validates `X, y`
(`check_X_y`, `accept_sparse`, `multi_output`), calls `score_func(X, y)`, and
unpacks `self.scores_, self.pvalues_` if the return is a tuple else
`pvalues_=None` (`:567-573`). `f_classif` (`:127`) splits `X` by class
(`args = [X[y==k] for k in np.unique(y)]`, `:172`) and delegates to `f_oneway`
(`:43`), which computes the one-way ANOVA `f = msb/msw` and
`prob = special.fdtrc(dfbn, dfwn, f)` (`:108-117`). Selection
(`_get_support_mask`, `:669-686`) short-circuits `percentile == 100 → all True`
(`:673-674`) and `percentile == 0 → all False` (`:675-676`); otherwise cleans NaN
scores (`_clean_nans` → `dtype.min`, `:678`,`:24`), sets `threshold =
np.percentile(scores, 100 - percentile)` (linear interpolation, `:679`), keeps
`scores > threshold` (STRICT, `:680`), and fills threshold-equal ties up to
`int(len(scores) * percentile / 100)` (FLOOR, `:683`) in ascending index order
(`:684-685`). The `SelectorMixin` base supplies `get_support`, `transform`,
`inverse_transform`, and `get_feature_names_out`. `_more_tags` advertises
`requires_y=False` (`:688-689`).

**The structural gap.** ferrolearn's **score** computation is an exact match for
sklearn on finite, non-degenerate input (REQ-1: the `anova_f_scores`
`ss_between`/`ss_within` F reproduces the live `f_classif` `[450.0, 0.5, 0.002069,
2040.2, 0.0]`), and the scoped error contracts ship (REQ-3). The selection rule (REQ-2, DIV-1) was the one
behavior that changed an observable selection on well-formed input — now FIXED
(#1274): `fit` replicates `scores > np.percentile(scores, 100 - percentile)` with
the `int()`-floor ascending tie-fill and `0/100` short-circuits, so the pinned
`pct=50` fixture now yields `[0, 3]`. The
remaining gaps are *contract surface*: `_clean_nans` NaN handling (REQ-4), the
pluggable `score_func`/extra score functions (REQ-5), the `SelectorMixin` surface
(REQ-6), `pvalues_` (REQ-7), the `Real`-typed `percentile` (REQ-8), the PyO3
binding (REQ-9), and the ferray substrate (REQ-10).

## Verification

Commands establishing the SHIPPED claims (REQ-1, REQ-3):

```bash
# Oracle (Probe REQ-1) — ANOVA F-score value match (the live f_classif output):
python3 -c "import numpy as np; from sklearn.feature_selection import f_classif; \
X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]); \
y=np.array([0,0,1,1]); print(f_classif(X,y)[0].tolist())"
#   -> [450.0, 0.5, 0.0020690..., 2040.2000..., 0.0]
# ferrolearn equivalent: SelectPercentile::<f64>::new(50, FClassif).fit(&X,&y).unwrap().scores()

# REQ-2 divergence (must be pinned as a FAILING oracle test; the fix is the threshold rule):
python3 -c "import numpy as np; from sklearn.feature_selection import SelectPercentile; \
X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],[9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]]); \
y=np.array([0,0,1,1]); \
print(SelectPercentile(percentile=50).fit(X,y).get_support(indices=True).tolist())"
#   -> [0, 3]   (ferrolearn returns [0, 1, 3] — k=ceil(2.5)=3 includes feature 1 at score 0.5)

# Crate gauntlet:
cargo test -p ferrolearn-preprocess   # incl. test_select_percentile_scores_stored,
                                      #       test_select_percentile_selects_highest_scoring,
                                      #       test_select_percentile_zero_rows_error,
                                      #       test_select_percentile_y_length_mismatch_error,
                                      #       test_select_percentile_shape_mismatch_on_transform,
                                      #       test_select_percentile_over_100_error
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check
```

The existing `#[test]`s exercise REQ-1 and REQ-3 but are **not oracle-grounded**
(`test_select_percentile_50_percent` asserts `out.ncols() == 2` as a property, not
the live sklearn mask). To satisfy R-CHAR-3 the critic should add an oracle-pinned
guard asserting `scores() == f_classif(X, y)[0]` (REQ-1) and a FAILING divergence
test asserting `selected_indices() == [0, 3]` at `pct=50` on the 5-feature fixture
(REQ-2). No currently-green command establishes REQ-2 or REQ-4..REQ-10.

## Blockers

REQ-1, REQ-2, REQ-3 are SHIPPED (REQ-2 fixed #1274 this iteration); the remaining
NOT-STARTED REQs are open `-l blocker` issues referenced by the REQ status table:

- #1274 — REQ-2 (DIV-1, CLOSED/fixed): `fit` selected `k = ceil(n_features ·
  percentile / 100)` rank-top-k; now replicates sklearn's threshold rule
  `scores > np.percentile(scores, 100 - percentile)` (strict `>`) + `int()`-floor
  ascending tie-fill + `0/100` short-circuits (`_univariate_selection.py:669-686`).
  `pct=50` 5-feature fixture: `[0,1,3]` → `[0,3]`.
- #1275 — REQ-4: no `_clean_nans` pre-pass; NaN scores ranked as `Ordering::Equal`
  rather than mapped to `dtype.min` (`:24`,`:678`).
- #1276 — REQ-5: `ScoreFunc` has only `FClassif`; no `chi2`/`f_regression`/
  `mutual_info_*` and no callable-`score_func` polymorphism (`:202`,`:405`,`:596-599`).
- #1277 — REQ-6: no boolean `get_support` / `inverse_transform` /
  `get_feature_names_out` (SelectorMixin surface).
- #1278 — REQ-7: no `pvalues_`; `anova_f_scores` returns the F-statistic only, no
  F-distribution survival function (`special.fdtrc`, `f_oneway:116`).
- #1279 — REQ-8: `percentile` is `usize`, not `Interval(Real, 0, 100)` — cannot
  express a fractional percentile (`:662`).
- #1280 — REQ-9: no `ferrolearn-python` `SelectPercentile` binding (boundary
  CPython consumer absent).
- #1281 — REQ-10: compute path on `ndarray`/`num_traits`/`HashMap`, not ferray
  (R-SUBSTRATE-1/2).
