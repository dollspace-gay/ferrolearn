# VarianceThreshold / SelectKBest / GenericUnivariateSelect — feature-selection transformers

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 17c7dc2b
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/feature_selection/_variance_threshold.py  # VarianceThreshold: variances_ = np.nanvar(X, axis=0) (:112); threshold==0 -> peak_to_peaks = np.ptp(X, axis=0) (:114) then variances_ = np.nanmin([variances_, peak_to_peaks], axis=0) (:116-120) to defeat constant-feature FP noise; _get_support_mask returns variances_ > threshold (:133); default threshold=0.0 (:77); _parameter_constraints threshold in [0, None) (:73-75).
  - sklearn/feature_selection/_univariate_selection.py  # _clean_nans (:24-33) maps NaN scores to np.finfo(dtype).min so NaN-scored features rank LAST; f_oneway (:43-117) one-way ANOVA: msb=ssbn/dfbn, msw=sswn/dfwn, f=msb/msw (:108-113); CONSTANT feature msw==0 with msb==0 -> f=0/0=NaN + "Features ... are constant." UserWarning (:110-113); f_classif (:127-173) groups X by np.unique(y) -> f_oneway(*args) (:172-173); SelectKBest (:692) __init__(score_func=f_classif, *, k=10) (:770-772), _check_params warns (not raises) when k>n_features (:774-779), _get_support_mask (:781-795): k=='all'->all True (:784-785), k==0->all False (:786-787), else scores=_clean_nans(scores_) (:789); mask[argsort(scores, kind="mergesort")[-k:]]=1 (:794) -> k-boundary ties broken toward HIGHER index; GenericUnivariateSelect (:1054) mode meta-selector {percentile,k_best,fpr,fdr,fwe} (:1119-1125), default mode='percentile', param=1e-5 (:1133).
ferrolearn-module: ferrolearn-preprocess/src/feature_selection.rs
parity-ops: VarianceThreshold, SelectKBest, GenericUnivariateSelect
crosslink-issue: 1424
-->

## Summary

scikit-learn's filter-style feature selectors `VarianceThreshold`
(`_variance_threshold.py:14`), `SelectKBest` (`_univariate_selection.py:692`) and
the mode-configurable meta-selector `GenericUnivariateSelect`
(`_univariate_selection.py:1054`) each fit a per-feature criterion and expose a
boolean support mask through the shared `SelectorMixin._get_support_mask`.
`VarianceThreshold` removes columns with `variances_ <= threshold`; `SelectKBest`
keeps the `k` highest-scoring columns of a pluggable `score_func` (default
`f_classif`); `GenericUnivariateSelect` dispatches to one of five concrete filters
by `mode`.

`ferrolearn-preprocess/src/feature_selection.rs` ships `VarianceThreshold<F>` (a
`Fit<Array2<F>,()>` computing a per-column Welford population variance and selecting
`var > threshold`), `SelectKBest<F>` (a `Fit<Array2<F>, Array1<usize>>` computing
in-module `anova_f_scores` then a top-`k` selection), and a SECOND, basic
`SelectFromModel<F>` surface (importance-threshold selection — distinct from the
richer `SelectFromModelExt` in `select_from_model.rs`, covered by
`.design/preprocess/select_from_model.md`). All three implement the ferrolearn
`Fit`/`Transform` + `PipelineTransformer` pattern and are re-exported at the crate
boundary (`lib.rs:131-133`).

This is a **shipped-partial** unit: **4 SHIPPED** (REQ-1 `VarianceThreshold`
mask + population variances on the common case, REQ-2 `SelectKBest` ANOVA F-scores
on finite non-constant features, REQ-3 `SelectKBest` top-`k` SELECTION — tie-break,
constant-feature ranking, and `k>n_features` clamp now match sklearn, REQ-4
error/parameter contracts) / **7 NOT-STARTED** (REQ-5
`VarianceThreshold` `threshold==0` peak-to-peak guard + `np.nanvar` NaN-handling;
REQ-6 `SelectKBest` `k='all'`/`k==0` + pluggable `score_func` + `_clean_nans`;
REQ-7 `GenericUnivariateSelect`; REQ-8 `SelectorMixin` surface +
`scores_`/`pvalues_`/`n_features_in_` attrs; REQ-9 basic `SelectFromModel`
duplicate; REQ-10 PyO3 binding; REQ-11 ferray substrate).

**`SelectKBest` top-`k` selection (REQ-3 — now faithful).** Verified against the
live sklearn oracle on the `X=[[1,1,5],[2,2,5],[7,7,5],[8,8,5]]`, `y=[0,0,1,1]`
fixture whose `f_classif` scores are `[72, 72, nan]`. ferrolearn now replicates
sklearn `scores = _clean_nans(scores_); mask[np.argsort(scores, kind="mergesort")
[-k:]] = 1` (`:794`, `:24-33`): the `k`-boundary tie is broken toward the HIGHER
column index (`argsort` ascending + stable, `[-k:]` keeps the later/higher index);
a constant feature scores `NaN` (`anova_f_scores` returns `F::nan()` when
`ms_within == 0 && ms_between == 0`), `_clean_nans` maps it to `F::min_value()`, so
it ranks LAST and is never selected unless `k` exhausts every feature; and
`k > n_features` clamps (`k_eff = k.min(n_features)`) and keeps all features
(sklearn warns+keeps-all, `:774-779`). Live oracle: cols 0,1 identical (score 72
each), `k=1` → **sklearn support `[1]`**, ferrolearn `[1]`; constant col 2 has
**sklearn score `nan`**, ferrolearn `nan`; `k=10` on 3 features → **sklearn `[0,1,2]`**,
ferrolearn `[0,1,2]`. Was DIV-A/DIV-B (resolved by #1425) + DIV-C (resolved by
#1426).

## Probes (live sklearn oracle, scikit-learn 1.5.2)

All values below are live oracle output captured from `/tmp` at baseline
`17c7dc2b`; ferrolearn values are the behavior of the shipped code on the identical
inputs. Re-run with the commands shown (R-CHAR-3 — expected values are the live
sklearn call, never copied from the ferrolearn side).

```bash
# ---------------------------------------------------------------------------
# PROBE A (REQ-2, REQ-3 / DIV-A, DIV-B) — SelectKBest scores + support
#   X = [[1,1,5],[2,2,5],[7,7,5],[8,8,5]], y = [0,0,1,1]
#   cols 0 and 1 are identical (tied F=72); col 2 is constant (F=NaN in sklearn)
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np, warnings
from sklearn.feature_selection import SelectKBest, f_classif
X=np.array([[1,1,5],[2,2,5],[7,7,5],[8,8,5]],float); y=np.array([0,0,1,1])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    s,p=f_classif(X,y); print('scores', s)
    for k in [1,2,'all',0]:
        m=SelectKBest(f_classif,k=k).fit(X,y)
        print('k=%r support'%k, np.where(m.get_support())[0].tolist())"
#   sklearn scores  = [72. 72. nan]            (col 2 constant -> NaN, NOT +inf)
#   sklearn k=1     support = [1]              (TIE broken toward HIGHER index)
#   sklearn k=2     support = [0, 1]           (constant col 2 NaN -> ranks LAST, excluded)
#   sklearn k='all' support = [0, 1, 2]        (k=='all' bypasses selection)
#   sklearn k=0     support = []               (k==0 -> all False)
#
#   ferrolearn anova_f_scores = [72.0, 72.0, NaN]    (FIXED #1425: constant col 2 -> NaN, matches sklearn)
#   ferrolearn k=1  selected_indices = [1]    (FIXED #1425: TIE broken toward HIGHER index)
#   ferrolearn k=2  selected_indices = [0, 1] (FIXED #1425: constant col 2 NaN->finfo.min ranks LAST, excluded)
#   ferrolearn k=10 (>n_features) selected_indices = [0, 1, 2]  (FIXED #1426: k_eff=k.min(n_features), keep-all)
#   ferrolearn k=0  selected_indices = []     (slicing idx[n_features-0..] is empty);
#               note SelectKBest still carries a plain usize k (no 'all' sentinel -> REQ-6).

# ---------------------------------------------------------------------------
# PROBE B (REQ-1, REQ-5) — VarianceThreshold variances_ + support, threshold=0
#   X = [[1,5,2],[2,5,2],[3,5,2]]  (cols 1 and 2 constant)
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.feature_selection import VarianceThreshold
X=np.array([[1,5,2],[2,5,2],[3,5,2]],float)
vt=VarianceThreshold(threshold=0.0).fit(X)
print('variances_', vt.variances_)
print('support', np.where(vt.get_support())[0].tolist())
print('np.nanvar', np.nanvar(X,axis=0), 'np.ptp', np.ptp(X,axis=0))"
#   sklearn variances_ = [0.66666667 0.         0.        ]   (== np.nanvar; ptp=[2,0,0] -> nanmin keeps var)
#   sklearn support    = [0]                                  (variances_ > 0)
#
#   ferrolearn variances() = [0.6666666666666666, 0.0, 0.0]   (Welford population var; matches to ~1e-15)
#   ferrolearn selected_indices = [0]                         (var > threshold; MATCHES on this case)

# ---------------------------------------------------------------------------
# PROBE C (REQ-5) — VarianceThreshold threshold==0 peak-to-peak guard.
#   sklearn replaces variances_ by nanmin(variances_, ptp) ONLY when threshold==0,
#   so a column whose naive variance is a tiny FP epsilon but ptp==0 is still dropped.
# ---------------------------------------------------------------------------
python3 -c "
import numpy as np
from sklearn.feature_selection import VarianceThreshold
# constant column whose float repr accumulates FP noise under naive var:
X=np.full((5,1), 1e8, float)
vt=VarianceThreshold(threshold=0.0).fit(X) if np.any(np.nanvar(X,axis=0)>0 or np.ptp(X,0)>0) else None
print('nanvar', np.nanvar(X,axis=0), 'ptp', np.ptp(X,axis=0))"
#   sklearn: ptp == 0 -> nanmin(var, ptp) == 0 -> column dropped at threshold=0 regardless of var FP noise.
#   ferrolearn: uses Welford (exact 0 for constant cols) so the COMMON constant case agrees,
#               but there is NO ptp-guard and NO np.nanvar NaN-handling (REQ-5 NOT-STARTED).
```

**SelectKBest selection parity summary (REQ-3 now faithful — DIV-A/B/C resolved).**

| input | sklearn `scores_` | sklearn `k=1` support | ferrolearn score | ferrolearn `k=1` | parity |
|---|---|---|---|---|---|
| tied cols 0,1 (F=72 each) | `[72, 72, nan]` | `[1]` | `[72, 72, nan]` | `[1]` | MATCH (tie → higher idx, #1425) |
| constant col 2 | `nan` (`f_classif`) | excluded for `k<3` | `nan` (`anova_f_scores` NaN) | excluded for `k<3` | MATCH (NaN→finfo.min, #1425) |
| `k=10` on 3 features | — | `[0,1,2]` | — | `[0,1,2]` | MATCH (k_eff clamp keep-all, #1426) |

REQ-2 is scoped to the F-SCORE VALUE on FINITE, non-constant features (where
ferrolearn's `anova_f_scores` matches `f_classif`'s `f = msb/msw` to ~1e-15); the
SELECTION step that consumes those scores (REQ-3) — tie-break, constant-feature
NaN ranking, and the `k>n_features` clamp — now matches sklearn and is SHIPPED.

## Requirements

- REQ-1: **`VarianceThreshold` mask (`var > threshold`) + population variances**
  (SHIPPED). For each feature compute the population variance and select columns
  whose variance is strictly greater than the threshold, matching sklearn's
  `variances_ = np.nanvar(X, axis=0)` (`_variance_threshold.py:112`) and
  `_get_support_mask` `return self.variances_ > self.threshold` (`:133`) with default
  `threshold=0.0` (`:77`). ferrolearn's `VarianceThreshold<F>` `Fit` impl
  (`feature_selection.rs`, `impl Fit<Array2<F>,()>`) computes a per-column Welford
  population variance (`var = m2 / n`) and pushes `j` when `var > self.threshold`,
  producing the identical `variances_`/support on the common (finite, non-FP-noise)
  case (Probe B). The `threshold==0` peak-to-peak guard and `np.nanvar` NaN-handling
  are split out into REQ-5.

- REQ-2: **`SelectKBest` ANOVA F-score VALUES (finite, non-constant features)**
  (SHIPPED, scoped). For each feature compute the one-way ANOVA F-statistic
  `F = (SSB/(k-1)) / (SSW/(n-k))`, matching `f_oneway` `msb = ssbn/dfbn`,
  `msw = sswn/dfwn`, `f = msb/msw` (`_univariate_selection.py:108-113`) reached via
  `f_classif` → `f_oneway(*args)` (`:172-173`). ferrolearn's in-module
  `anova_f_scores` (`fn anova_f_scores in feature_selection.rs`) computes
  `ss_between = Σ n_k·(class_mean-grand_mean)²` / `ss_within = Σ (x-class_mean)²`
  then `f = (ss_between/df_b)/(ss_within/df_w)` (`df_b = n_classes-1`,
  `df_w = n_samples-n_classes`), producing the identical F-VALUE on finite,
  non-constant features (Probe A: cols 0,1 → `72.0` each, matching `f_classif`).
  The CONSTANT-feature value (now `NaN`, matching sklearn) and the SELECTION
  step are REQ-3 (SHIPPED — tie-break + constant ranking + `k>n_features` clamp).

- REQ-3: **`SelectKBest` top-`k` SELECTION (tie-break + constant-feature ranking +
  `k>n_features` clamp)** (SHIPPED). ferrolearn now replicates sklearn's
  `_get_support_mask`: `scores = _clean_nans(scores_)` (`_univariate_selection.py:24-33`,
  NaN → `np.finfo.min`) then `mask[np.argsort(scores, kind="mergesort")[-k:]] = 1`
  (`:794`). In `impl Fit<Array2<F>, Array1<usize>> for SelectKBest` the cleaned
  vector maps `NaN → F::min_value()`, indices are sorted ASCENDING + stably, and the
  last-`k_eff` slice is kept — so a `k`-boundary tie keeps the HIGHER index (it
  appears later) and a constant feature ranks LAST. Three previously-divergent
  behaviors are now faithful:
  - **tie-break** (was DIV-A): `k`-boundary tie → HIGHER index. Probe A `k=1`:
    sklearn `[1]`, ferrolearn `[1]`. RESOLVED by #1425.
  - **constant feature** (was DIV-B): `anova_f_scores` now returns `F::nan()` when
    `ms_within == 0 && ms_between == 0` (instead of `+inf`), so `_clean_nans` ranks
    it LAST. Probe A constant col 2: sklearn `NaN`, ferrolearn `NaN`. RESOLVED by
    #1425.
  - **`k > n_features`** (was DIV-C): clamps `k_eff = k.min(n_features)` and keeps
    all features (sklearn warns+keeps-all, `:774-779`) instead of erroring. Live
    oracle `k=10`/3 features: sklearn `[0,1,2]`, ferrolearn `[0,1,2]`. RESOLVED by
    #1426.
  Verified faithful across multi-tie, multi-constant (`k=3`/`k=4`), `k ∈ {0, ==n,
  >n}`, mixed, real-ish 8×4×3-class, scores-NaN-at-constant, and f32 — 21 tests in
  `tests/divergence_feature_selection.rs` (full gauntlet GREEN). The original REQ-3
  prereq blocker is RESOLVED as #1425 (tie-break + constant NaN) + #1426
  (`k>n_features` clamp).

- REQ-4: **error / parameter contracts** (SHIPPED). `VarianceThreshold::fit`
  returns `Err(InvalidParameter)` on `threshold < 0` (mirroring
  `_parameter_constraints {"threshold": [Interval(Real, 0, None, closed="left")]}`
  `_variance_threshold.py:73-75`) and `Err(InsufficientSamples)` on zero rows.
  `SelectKBest::fit` returns `Err(InsufficientSamples)` on zero rows,
  `Err(ShapeMismatch)` when `y.len() != n_samples`, and `Err(InvalidParameter)` when
  `k > n_features`. These are the ferrolearn error contracts scoped to the dense
  `Array2`/`Array1` API. (NOTE the `k > n_features` BEHAVIOR diverges from sklearn,
  which only *warns* and returns all features — `SelectKBest._check_params`
  `warnings.warn(...)` `_univariate_selection.py:774-779`; the ferrolearn `Err`
  surface is a stricter contract folded into REQ-6.)

- REQ-5: **`VarianceThreshold` `threshold==0` peak-to-peak guard + `np.nanvar`
  NaN-handling** (NOT-STARTED). When `threshold == 0`, sklearn replaces
  `variances_` with `np.nanmin([variances_, np.ptp(X, axis=0)], axis=0)`
  (`_variance_threshold.py:113-120`) so a near-constant column with FP-noise variance
  but `ptp == 0` is still dropped; sklearn also allows NaN in `X` (`force_all_finite=
  "allow-nan"` `:103`) and uses `np.nanvar` (`:112`). ferrolearn's Welford variance
  is exactly `0` for constant columns (so the COMMON case agrees, Probe B) but there
  is no `ptp` guard and no NaN-aware variance (NaN values poison the Welford
  accumulation), and no "no feature meets threshold → ValueError" path (`:122-126`).
  NOT-STARTED on prereq blocker #1427.

- REQ-6: **`SelectKBest` `k='all'` + `k==0` semantics + pluggable `score_func`
  (chi2/f_regression/mutual_info) + general `_clean_nans`** (NOT-STARTED). sklearn's
  `SelectKBest` accepts `k="all"` → all-True mask (`_univariate_selection.py:784-785`)
  and `k==0` → all-False mask (`:786-787`), takes any `score_func` (default
  `f_classif`, `:770`), and only *warns* when `k > n_features` (`:774-779`).
  ferrolearn's `SelectKBest<F>` carries a plain `k: usize` (no `"all"` sentinel) and
  a `ScoreFunc` enum with the single `FClassif` variant (`enum ScoreFunc`), erroring
  on `k > n_features` rather than warning. The general `_clean_nans` (`:24-33`,
  NaN → `finfo.min`) is absent (folded into DIV-B / REQ-3 for the f_classif path but
  not generalized). NOT-STARTED on prereq blocker #1428.

- REQ-7: **`GenericUnivariateSelect` (mode meta-selector)** (NOT-STARTED). sklearn's
  `GenericUnivariateSelect` (`_univariate_selection.py:1054`) dispatches to one of
  five concrete filters by `mode ∈ {percentile, k_best, fpr, fdr, fwe}`
  (`_selection_modes` `:1119-1125`), default `mode='percentile'`, `param=1e-5`
  (`:1133`), forwarding `param` into the chosen selector (`_make_selector` `:1138`).
  ferrolearn has no `GenericUnivariateSelect` analog and no `fpr`/`fdr`/`fwe`
  selectors. This is a routed `parity-op` that is ABSENT. NOT-STARTED on prereq
  blocker #1429.

- REQ-8: **`SelectorMixin` surface + `scores_`/`pvalues_`/`n_features_in_` attrs**
  (NOT-STARTED). sklearn's selectors inherit `SelectorMixin`
  (`feature_selection/_base.py`) exposing `get_support(indices=...)`,
  `inverse_transform`, and `get_feature_names_out`, and `_BaseFilter` sets
  `scores_`/`pvalues_`/`n_features_in_` (`_univariate_selection.py:567-575`).
  ferrolearn surfaces `selected_indices()`/`variances()`/`scores()` accessors but
  has no `get_support(indices=True)` boolean-mask API, no `inverse_transform`, no
  `get_feature_names_out`, and `SelectKBest` exposes no `pvalues_` (its
  `anova_f_scores` returns scores only). NOT-STARTED on prereq blocker #1430.

- REQ-9: **basic `SelectFromModel` (duplicate surface)** (NOT-STARTED / cross-ref).
  `feature_selection.rs` ALSO defines a basic `SelectFromModel<F>`
  (`new_from_importances`, mean / explicit threshold, `imp >= threshold`) — a
  SECOND, simpler surface distinct from `select_from_model.rs::SelectFromModelExt`.
  Its sklearn parity (vs `SelectFromModel` in `_from_model.py`) is owned by
  `.design/preprocess/select_from_model.md` — do NOT re-litigate it here. The
  DUPLICATION is tech-debt: two `SelectFromModel`-shaped types in one crate (the
  basic one here, the rich `SelectFromModelExt` in `select_from_model.rs`).
  NOT-STARTED on prereq blocker #1431 (cross-ref `.design/preprocess/select_from_model.md`).

- REQ-10: **PyO3 binding** (NOT-STARTED). There is no `VarianceThreshold` /
  `SelectKBest` / `GenericUnivariateSelect` CPython binding in `ferrolearn-python`
  (`grep -rln "SelectKBest\|VarianceThreshold" ferrolearn-python/src` finds none),
  so these transformers are unreachable from Python. NOT-STARTED on prereq blocker
  #1432.

- REQ-11: **ferray substrate** (NOT-STARTED). The selectors compute over
  `ndarray::Array2`/`Array1` (`x.column(j)`, `Array2::zeros`) and
  `num_traits::Float`, not `ferray-core` arrays (R-SUBSTRATE-1/2). NOT-STARTED on
  prereq blocker #1433.

## Acceptance criteria

- AC-1 (REQ-1): `VarianceThreshold::<f64>::new(0.0)` on Probe B's `X` yields
  `variances() == [0.6666…, 0.0, 0.0]` matching `VarianceThreshold().fit(X).variances_`
  to `|Δ| < 1e-12` and `selected_indices() == [0]`. Pinned by in-module
  `test_variance_threshold_removes_constant_column`,
  `test_variance_threshold_custom_threshold`, `test_variance_threshold_stores_variances`,
  `test_variance_threshold_all_constant_columns`, plus the Probe B oracle gate.

- AC-2 (REQ-2): `SelectKBest` `anova_f_scores` on Probe A's `(X, y)` finite columns
  0,1 yields `[72.0, 72.0, …]` matching `f_classif`'s `[72., 72., …]` to
  `|Δ| < 1e-9`. Pinned by in-module `test_select_k_best_scores_stored`,
  `test_select_k_best_f_score_zero_within_class_variance`, plus the Probe A oracle
  gate (finite columns only).

- AC-3 (REQ-3): on Probe A `(X, y)`, `SelectKBest(k=1).fit(X,y).selected_indices()`
  == sklearn `[1]` (tie → higher index) and the constant column scores `NaN` (not
  `+inf`), ranking LAST so it is excluded for `k<3`; `k=10` on the 3-feature `X`
  keeps `[0,1,2]` (clamp + keep-all). Pinned by
  `divergence_selectkbest_tiebreak_keeps_higher_index`,
  `divergence_selectkbest_constant_feature_ranks_last`,
  `divergence_selectkbest_k_over_nfeatures_keeps_all`, and the RE-AUDIT suite
  (`reaudit_a`..`reaudit_h`) in `tests/divergence_feature_selection.rs` — **all
  GREEN** (21 tests). SHIPPED (resolved by #1425 + #1426).

- AC-4 (REQ-4): `VarianceThreshold::new(-0.1).fit(x)` → `Err(InvalidParameter)`;
  `VarianceThreshold::new(0.0).fit(Array2::zeros((0,2)))` → `Err(InsufficientSamples)`;
  `SelectKBest::new(1,..).fit(Array2::zeros((0,3)), zeros(0))` →
  `Err(InsufficientSamples)`; `SelectKBest` with `y.len()` mismatch →
  `Err(ShapeMismatch)`; `SelectKBest::new(5,..)` on a 2-feature `X` →
  `Err(InvalidParameter)`. Pinned by in-module
  `test_variance_threshold_negative_threshold_error`,
  `test_variance_threshold_zero_rows_error`, `test_select_k_best_zero_rows_error`,
  `test_select_k_best_y_length_mismatch_error`,
  `test_select_k_best_k_exceeds_n_features_error`.

- AC-5 (REQ-5): `VarianceThreshold(threshold=0)` drops a column whose naive variance
  is FP-noise but `ptp == 0` (sklearn `nanmin(var, ptp)` guard), and tolerates NaN in
  `X` via `np.nanvar`; ferrolearn has no ptp guard and no NaN-aware variance.
  NOT-STARTED.

- AC-6 (REQ-6): `SelectKBest(k="all")` → all features; `SelectKBest(k=0)` → none;
  `SelectKBest(chi2, k=…)` / `SelectKBest(f_regression, …)`; `k > n_features` *warns*
  and keeps all. ferrolearn has a plain `k: usize` (no `"all"`), a single-variant
  `ScoreFunc::FClassif`, and *errors* on `k > n_features`. NOT-STARTED.

- AC-7 (REQ-7): `GenericUnivariateSelect(f_classif, mode="k_best", param=2)` matches
  `SelectKBest(k=2)`; ferrolearn has no `GenericUnivariateSelect`. NOT-STARTED.

- AC-8 (REQ-8): `selector.get_support(indices=True)` returns selected column
  indices; `inverse_transform` round-trips; `SelectKBest.pvalues_` is populated;
  ferrolearn exposes none of these. NOT-STARTED.

- AC-9 (REQ-9): the basic `SelectFromModel` here vs the rich
  `SelectFromModelExt` in `select_from_model.rs` — parity owned by
  `.design/preprocess/select_from_model.md`; duplication is tech-debt. NOT-STARTED.

- AC-10 (REQ-10): a CPython `VarianceThreshold`/`SelectKBest` binding selects
  features from Python; no such binding exists. NOT-STARTED.

- AC-11 (REQ-11): the selectors compute on `ferray-core` arrays rather than
  `ndarray` + `num_traits::Float`. NOT-STARTED.

`## REQ status`

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`VarianceThreshold` mask + population variances) | SHIPPED | impl `pub fn fit in feature_selection.rs` (`impl Fit<Array2<F>,()> for VarianceThreshold<F>`) computes a per-column Welford population variance (`var = m2 / n`) and pushes `j` when `var > self.threshold`, mirroring sklearn `variances_ = np.nanvar(X, axis=0)` (`sklearn/feature_selection/_variance_threshold.py:112`) + `_get_support_mask` `return self.variances_ > self.threshold` (`:133`) with default `threshold=0.0` (`:77`). Non-test consumer: boundary re-export `pub use feature_selection::{... FittedVarianceThreshold ... VarianceThreshold}` (`lib.rs:131-133`) + `impl PipelineTransformer<F> for VarianceThreshold<F>` (used by `ferrolearn_core::pipeline::Pipeline`). Verification: Probe B — ferrolearn `variances() == [0.6666…, 0.0, 0.0]` vs sklearn `[0.66666667, 0., 0.]` (`|Δ| ~ 1e-15`), `selected_indices() == [0]` vs sklearn support `[0]`; `cargo test -p ferrolearn-preprocess --lib` → `test_variance_threshold_removes_constant_column`, `test_variance_threshold_custom_threshold`, `test_variance_threshold_all_constant_columns`, `test_variance_threshold_stores_variances`, `test_variance_threshold_f32` green. |
| REQ-2 (`SelectKBest` ANOVA F-score VALUES, finite features) | SHIPPED | impl `fn anova_f_scores in feature_selection.rs` computes per-feature `ss_between = Σ n_k·(class_mean-grand_mean)²` / `ss_within = Σ (x-class_mean)²` then `f = (ss_between/df_between)/(ss_within/df_within)` (`df_between = n_classes-1`, `df_within = n_samples-n_classes`), mirroring `f_oneway` `msb = ssbn/dfbn; msw = sswn/dfwn; f = msb/msw` (`sklearn/feature_selection/_univariate_selection.py:108-113`) reached from `f_classif` `return f_oneway(*args)` (`:173`); called from `impl Fit<Array2<F>, Array1<usize>> for SelectKBest<F>` via `match self.score_func { ScoreFunc::FClassif => anova_f_scores(x, y) }`. Non-test consumer: boundary re-export `pub use feature_selection::{FittedSelectKBest, ... ScoreFunc, ... SelectKBest, ...}` (`lib.rs:131-133`) + `impl PipelineTransformer<F> for SelectKBest<F>`. Verification: Probe A — ferrolearn finite scores `[72.0, 72.0, …]` vs sklearn `f_classif` `[72., 72., nan]` (finite cols match ~1e-15); `cargo test -p ferrolearn-preprocess --lib` → `test_select_k_best_scores_stored`, `test_select_k_best_f_score_zero_within_class_variance` green. SCOPED to the finite/non-constant F-VALUE; the constant-feature value (`+inf` vs `NaN`, DIV-B) and the selection step (DIV-A) are REQ-3. |
| REQ-3 (`SelectKBest` top-`k` SELECTION: tie-break + constant-feature ranking + `k>n_features` clamp) | SHIPPED | impl `impl Fit<Array2<F>, Array1<usize>> for SelectKBest` replicates sklearn `_get_support_mask`: cleaned scores map `NaN → F::min_value()` (`_clean_nans`, `sklearn/feature_selection/_univariate_selection.py:24-33`), indices sort ASCENDING + stably, the last-`k_eff` slice is kept → `mask[np.argsort(scores, kind="mergesort")[-k:]] = 1` (`:794`, k-boundary tie keeps the HIGHER index). `fn anova_f_scores` returns `F::nan()` when `ms_within == F::zero() && ms_between == F::zero()` (was `F::infinity()`), so a constant feature ranks LAST; `let k_eff = self.k.min(n_features)` clamps `k > n_features` and keeps all (sklearn warns+keeps-all `:774-779`). Non-test consumer: boundary re-export (`lib.rs:131-133`) + `impl PipelineTransformer<F> for SelectKBest<F>`. Verification: live oracle — `k=1` sklearn `[1]` == ferrolearn `[1]`; constant col 2 sklearn `NaN` == ferrolearn `NaN`; `k=10`/3-feat sklearn `[0,1,2]` == ferrolearn `[0,1,2]`; `cargo test -p ferrolearn-preprocess --test divergence_feature_selection` → 21 tests GREEN (`divergence_selectkbest_tiebreak_keeps_higher_index`, `divergence_selectkbest_constant_feature_ranks_last`, `divergence_selectkbest_k_over_nfeatures_keeps_all`, `reaudit_a`..`reaudit_h`). Was DIV-A/DIV-B (resolved #1425) + DIV-C (resolved #1426). |
| REQ-4 (error / parameter contracts) | SHIPPED | impl `VarianceThreshold::fit` returns `Err(InvalidParameter{name:"threshold"})` on `self.threshold < F::zero()` (mirroring `_parameter_constraints {"threshold": [Interval(Real, 0, None, closed="left")]}` `sklearn/feature_selection/_variance_threshold.py:73-75`) and `Err(InsufficientSamples)` on `n_samples == 0`; `SelectKBest::fit` returns `Err(InsufficientSamples)` on `n_samples == 0`, `Err(ShapeMismatch)` on `y.len() != n_samples`, and `Err(InvalidParameter{name:"k"})` on `self.k > n_features`. Non-test consumer: boundary re-export (`lib.rs:131-133`). Verification: `cargo test -p ferrolearn-preprocess --lib` → `test_variance_threshold_negative_threshold_error`, `test_variance_threshold_zero_rows_error`, `test_select_k_best_zero_rows_error`, `test_select_k_best_y_length_mismatch_error`, `test_select_k_best_k_exceeds_n_features_error` green. NOTE the `k > n_features` Err diverges from sklearn's *warn-and-keep-all* (`_check_params` `:774-779`); that behavioral gap is REQ-6. |
| REQ-5 (`VarianceThreshold` `threshold==0` ptp-guard + `np.nanvar` NaN) | NOT-STARTED | open prereq blocker #1427. sklearn, when `threshold == 0`, sets `variances_ = np.nanmin([variances_, np.ptp(X, axis=0)], axis=0)` (`sklearn/feature_selection/_variance_threshold.py:113-120`) and computes `np.nanvar` allowing NaN (`force_all_finite="allow-nan"` `:103`, `:112`), plus raises ValueError when no feature meets the threshold (`:122-126`). ferrolearn's `VarianceThreshold::fit` uses Welford (exactly `0` on constant cols, so the COMMON case matches Probe B) but has NO ptp guard, NO NaN-aware variance, and NO "no feature meets threshold" error. |
| REQ-6 (`k='all'`/`k==0` + pluggable `score_func` + `_clean_nans`) | NOT-STARTED | open prereq blocker #1428. sklearn `SelectKBest` accepts `k="all"` → all-True (`sklearn/feature_selection/_univariate_selection.py:784-785`), `k==0` → all-False (`:786-787`), any `score_func` (default `f_classif` `:770`), and only WARNS on `k > n_features` (`:774-779`). ferrolearn's `SelectKBest<F>` has a plain `k: usize` (no `"all"` sentinel) and a single-variant `enum ScoreFunc { FClassif }` — no `chi2`/`f_regression`/`mutual_info` dispatch, no generalized `_clean_nans` (`:24-33`), and it ERRORS on `k > n_features`. |
| REQ-7 (`GenericUnivariateSelect` mode meta-selector) | NOT-STARTED | open prereq blocker #1429 (routed parity-op, ABSENT). sklearn `GenericUnivariateSelect` (`sklearn/feature_selection/_univariate_selection.py:1054`) dispatches by `mode ∈ {percentile, k_best, fpr, fdr, fwe}` (`_selection_modes` `:1119-1125`) with default `mode='percentile'`, `param=1e-5` (`:1133`). ferrolearn has no `GenericUnivariateSelect` and no `fpr`/`fdr`/`fwe`/`percentile` filters in `feature_selection.rs`. |
| REQ-8 (`SelectorMixin` surface + `scores_`/`pvalues_`/`n_features_in_`) | NOT-STARTED | open prereq blocker #1430. sklearn's selectors inherit `SelectorMixin` (`get_support`/`inverse_transform`/`get_feature_names_out`) and `_BaseFilter` sets `scores_`/`pvalues_`/`n_features_in_` (`sklearn/feature_selection/_univariate_selection.py:567-575`). ferrolearn exposes `selected_indices()`/`variances()`/`scores()` accessors only — no boolean `get_support(indices=…)`, no `inverse_transform`, no `get_feature_names_out`, and `SelectKBest` surfaces no `pvalues_` (`anova_f_scores` returns scores only). |
| REQ-9 (basic `SelectFromModel` duplicate surface) | NOT-STARTED | open prereq blocker #1431 (cross-ref). `feature_selection.rs` defines a basic `pub struct SelectFromModel<F>` (`new_from_importances`, mean/explicit threshold, `imp >= thr`) DISTINCT from the rich `select_from_model.rs::SelectFromModelExt`. Its sklearn parity (vs `SelectFromModel` in `sklearn/feature_selection/_from_model.py`) is owned by `.design/preprocess/select_from_model.md` — see that doc; not re-litigated here. The DUPLICATION (two `SelectFromModel`-shaped types in one crate) is tech-debt. |
| REQ-10 (PyO3 binding) | NOT-STARTED | open prereq blocker #1432. No CPython binding for `VarianceThreshold`/`SelectKBest`/`GenericUnivariateSelect` exists in `ferrolearn-python/src` (`grep -rln "SelectKBest\|VarianceThreshold" ferrolearn-python/src` → none), so these transformers are unreachable from Python. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker #1433. The selectors + `select_columns`/`anova_f_scores` helpers compute over `ndarray::Array2`/`Array1` (`x.column(j)`, `Array2::zeros`) and `num_traits::Float`, not `ferray-core` arrays (R-SUBSTRATE-1/2). |

## Architecture

**ferrolearn (existing, compiled at baseline `17c7dc2b`).** `feature_selection.rs`
exposes three selector surfaces plus a shared column-gather helper:

- `select_columns(x: &Array2<F>, indices: &[usize]) -> Array2<F>` — builds a new
  matrix of the selected columns in `indices` order (returns an `(nrows, 0)` matrix
  when `indices` is empty). Used by every `transform`.
- `VarianceThreshold<F> { threshold: F }` (`new`, `Default` = `F::zero()`,
  `threshold()`). Its `Fit<Array2<F>,()>` impl validates `threshold >= 0` and
  `n_samples > 0`, then for each column runs Welford's online algorithm
  (`count`/`mean`/`m2`) to get `var = m2 / n` (population variance, exactly `0` for
  constant columns — the comment notes the naive `Σ(v-mean)²/n` accumulates ~1e-34
  FP noise that defeats the `threshold=0 ⇒ drop constants` contract), selecting `j`
  when `var > threshold`. `FittedVarianceThreshold<F> { selected_indices, variances }`
  exposes `selected_indices()`/`variances()` and a `Transform` that `ShapeMismatch`-checks
  `ncols`.
- `SelectKBest<F> { k: usize, score_func: ScoreFunc, _marker }` (`new`, `k()`,
  `score_func()`); `enum ScoreFunc { FClassif }`. Its `Fit<Array2<F>, Array1<usize>>`
  impl validates `n_samples > 0` and `y.len() == n_samples`, clamps `k_eff =
  k.min(n_features)` (so `k > n_features` keeps all — the #1426 fix, no longer an
  error), computes `anova_f_scores`, maps `NaN -> F::min_value()` (`_clean_nans`),
  sorts indices ASCENDING + stably by cleaned score, slices the last `k_eff` (a
  k-boundary tie keeps the HIGHER index, matching sklearn `argsort mergesort[-k:]`),
  then sorts the kept indices ascending for a stable output layout.
  `FittedSelectKBest<F> { n_features_in, scores, selected_indices }`
  exposes `scores()`/`selected_indices()` and a shape-checked `Transform`.
- `anova_f_scores(x, y: &Array1<usize>) -> Vec<F>` — one-way ANOVA F per column:
  groups rows by class into a `HashMap<usize, Vec<usize>>`, computes
  `ss_between`/`ss_within`, `df_between = n_classes-1`, `df_within =
  n_samples-n_classes`, then `f = ms_between/ms_within`. **`ms_within == 0`** splits:
  `ms_between > 0` (perfect separator) → `F::infinity()`; `ms_between == 0` (constant
  feature) → `F::nan()` — matching sklearn `f = 0/0 = NaN`, then `_clean_nans` →
  `finfo.min` ranks it last (the #1425 fix; was a bare `+inf`). `df_between == 0 ||
  df_within == 0` → `F::zero()`.
- `SelectFromModel<F> { importances, threshold, selected_indices }`
  (`new_from_importances`, mean / explicit threshold, `imp >= thr`) — a SECOND, basic
  importance-threshold surface. Its parity is owned by
  `.design/preprocess/select_from_model.md` (the rich `SelectFromModelExt` lives in
  `select_from_model.rs`); the two co-existing `SelectFromModel`-shaped types are
  tech-debt (REQ-9).

All three implement `PipelineTransformer<F>` / `FittedPipelineTransformer<F>` and are
re-exported `pub use feature_selection::{FittedSelectKBest, FittedVarianceThreshold,
ScoreFunc, SelectFromModel, SelectKBest, VarianceThreshold};` (`lib.rs:131-133`) —
that boundary re-export + the pipeline integration are the grandfathered consumers
(S5 / R-DEFER-1) pinning the SHIPPED rows. There is NO PyO3 binding (REQ-10).

**sklearn (target contract).** `VarianceThreshold` (`_variance_threshold.py:14`) is a
`SelectorMixin`/`BaseEstimator` whose `fit` computes `variances_ = np.nanvar(X,
axis=0)` (`:112`), applies the `threshold==0` peak-to-peak guard
`variances_ = np.nanmin([variances_, np.ptp(X, axis=0)], axis=0)` (`:113-120`), raises
when no feature meets the threshold (`:122-126`), and selects via `_get_support_mask`
`variances_ > threshold` (`:133`). `SelectKBest` (`_univariate_selection.py:692`) is a
`_BaseFilter` whose `fit` (`:567-575`) calls `score_func(X, y)` (default `f_classif`),
stores `scores_`/`pvalues_`/`n_features_in_`, and whose `_get_support_mask` (`:781-795`)
handles `k=="all"` (all-True), `k==0` (all-False), else `_clean_nans(scores_)` (`:789`,
NaN → `finfo.min` `:32`) then `mask[argsort(scores, kind="mergesort")[-k:]] = 1` (`:794`,
higher-index tie-break). `GenericUnivariateSelect` (`:1054`) is the mode meta-selector
dispatching to `{percentile, k_best, fpr, fdr, fwe}` (`:1119-1125`).

**The gap.** ferrolearn matches sklearn on `VarianceThreshold`'s mask + population
variances on the common case (REQ-1), `SelectKBest`'s F-SCORE VALUES on finite
non-constant features (REQ-2), the `SelectKBest` top-`k` SELECTION step — k-boundary
tie-break (HIGHER index), constant-feature `NaN` ranking, and the `k>n_features`
clamp+keep-all (REQ-3, resolved by #1425 + #1426) — and the dense error/parameter
contracts (REQ-4). The remaining gaps are surface/parameterization: no `threshold==0`
ptp-guard / NaN-handling (REQ-5), no `k='all'`/`k==0` string sentinel / pluggable
`score_func` / generalized `_clean_nans` (REQ-6), no `GenericUnivariateSelect`
(REQ-7, absent parity-op), no `SelectorMixin` surface / `pvalues_` (REQ-8), the
duplicate `SelectFromModel` (REQ-9, cross-ref), no PyO3 binding (REQ-10), and the
non-ferray substrate (REQ-11). This is a **shipped-partial** unit (4 SHIPPED / 7
NOT-STARTED).

## Verification

Commands establishing the SHIPPED claims (REQ-1 `VarianceThreshold` mask + variances,
REQ-2 `SelectKBest` finite F-scores, REQ-3 top-`k` selection, REQ-4 error contracts):

```bash
# Module is compiled + re-exported (the boundary consumer):
grep -n "mod feature_selection" ferrolearn-preprocess/src/lib.rs            # lib.rs:97
grep -n "FittedSelectKBest\|VarianceThreshold\|SelectKBest" ferrolearn-preprocess/src/lib.rs  # lib.rs:131-133

# REQ-1/2/4 (in-module tests):
cargo test -p ferrolearn-preprocess --lib
#   REQ-1: test_variance_threshold_removes_constant_column, test_variance_threshold_custom_threshold,
#          test_variance_threshold_stores_variances, test_variance_threshold_all_constant_columns,
#          test_variance_threshold_keeps_all_when_above, test_variance_threshold_f32,
#          test_variance_threshold_pipeline_integration
#   REQ-2: test_select_k_best_scores_stored, test_select_k_best_f_score_zero_within_class_variance,
#          test_select_k_best_k_equals_n_features_keeps_all
#   REQ-4: test_variance_threshold_negative_threshold_error, test_variance_threshold_zero_rows_error,
#          test_variance_threshold_shape_mismatch_on_transform, test_select_k_best_zero_rows_error,
#          test_select_k_best_y_length_mismatch_error, test_select_k_best_k_exceeds_n_features_error,
#          test_select_k_best_shape_mismatch_on_transform
# REQ-3 (top-k selection — tie-break + constant ranking + k>n_features clamp):
cargo test -p ferrolearn-preprocess --test divergence_feature_selection   # 21 tests GREEN
#   divergence_selectkbest_tiebreak_keeps_higher_index, divergence_selectkbest_constant_feature_ranks_last,
#   divergence_selectkbest_k_over_nfeatures_keeps_all, reaudit_a..reaudit_h (multi-tie/multi-constant/
#   k in {0,==n,>n}/mixed/realish-8x4-3class/scores-NaN-at-constant/f32)
cargo clippy -p ferrolearn-preprocess --all-targets -- -D warnings
cargo fmt --all --check

# REQ-1 oracle gate — VarianceThreshold variances_ + support (threshold=0, constant cols):
python3 -c "
import numpy as np
from sklearn.feature_selection import VarianceThreshold
X=np.array([[1,5,2],[2,5,2],[3,5,2]],float)
vt=VarianceThreshold(threshold=0.0).fit(X)
print('variances_', vt.variances_, 'support', np.where(vt.get_support())[0].tolist())"
#   -> variances_ [0.66666667 0. 0.]  support [0]   (ferrolearn matches to ~1e-15)

# REQ-2 oracle gate — f_classif scores on the finite columns (constant col DIVERGES, REQ-3):
python3 -c "
import numpy as np, warnings
from sklearn.feature_selection import f_classif
X=np.array([[1,1,5],[2,2,5],[7,7,5],[8,8,5]],float); y=np.array([0,0,1,1])
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); s,p=f_classif(X,y); print('scores', s)"
#   -> scores [72. 72. nan]   (ferrolearn finite cols match 72.0; constant col -> NaN, matches sklearn — REQ-3)

# REQ-3 parity gate (now GREEN) — tie-break + constant ranking + k>n_features clamp:
python3 -c "
import numpy as np, warnings
from sklearn.feature_selection import SelectKBest, f_classif
X=np.array([[1,1,5],[2,2,5],[7,7,5],[8,8,5]],float); y=np.array([0,0,1,1])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for k in [1,2,10]:
        m=SelectKBest(f_classif,k=k).fit(X,y)
        print('k=%d support'%k, np.where(m.get_support())[0].tolist())"
#   -> k=1  support [1]     (ferrolearn selects [1]    — tie → HIGHER index, #1425)
#   -> k=2  support [0,1]   (ferrolearn selects [0,1]  — constant col 2 NaN ranks last, #1425)
#   -> k=10 support [0,1,2] (ferrolearn selects [0,1,2] — k_eff clamp keep-all, #1426)
```

The in-module `#[test]`s pin REQ-1 (`VarianceThreshold` mask + variances), REQ-2
(the finite F-score VALUE), and REQ-4 (every dense error path); the 21-test
`tests/divergence_feature_selection.rs` gauntlet pins REQ-3 (top-`k` selection
tie-break + constant-feature NaN ranking + `k>n_features` clamp), all GREEN. No green
command establishes REQ-5 (`threshold==0` ptp-guard / NaN-handling), REQ-6
(`k='all'`/`k==0` / pluggable `score_func`), REQ-7 (`GenericUnivariateSelect`), REQ-8
(`SelectorMixin` surface / `pvalues_`), REQ-9 (duplicate `SelectFromModel`,
cross-ref), REQ-10 (PyO3), or REQ-11 (ferray).

## Blockers

REQ-1 (`VarianceThreshold` mask + population variances), REQ-2 (`SelectKBest` finite
ANOVA F-score VALUES), REQ-3 (`SelectKBest` top-`k` SELECTION), and REQ-4 (error /
parameter contracts) are SHIPPED — the module is compiled (`lib.rs:97`) and
re-exported (`lib.rs:131-133`, the grandfathered boundary consumer) plus
pipeline-integrated, the `VarianceThreshold` variances/support and the finite
F-scores match the live oracle to ~1e-15, and the tests are green.

The remaining REQs (REQ-5..11) are NOT-STARTED. Each is filed as a `-l blocker` issue
against tracking issue #1424:

- RESOLVED — REQ-3: `SelectKBest` top-`k` SELECTION now matches sklearn. The two
  former divergences DIV-A (k-boundary tie → higher index) + DIV-B (constant feature
  → `NaN` not `+inf`) were resolved by **#1425** (`anova_f_scores` returns `F::nan()`
  when `ms_within==0 && ms_between==0`; selection replicates `_clean_nans` →
  `argsort mergesort [-k:]`, `_univariate_selection.py:24-33`/`:794`), and DIV-C
  (`k>n_features`) was resolved by **#1426** (`k_eff = k.min(n_features)` clamp +
  keep-all, `:774-779`). Both blockers are CLOSED; 21 tests in
  `tests/divergence_feature_selection.rs` GREEN.
- #1427 — REQ-5: no `threshold==0` peak-to-peak guard
  (`_variance_threshold.py:113-120`) and no `np.nanvar` NaN-handling (`:103`, `:112`)
  / no "no feature meets threshold" ValueError (`:122-126`).
- #1428 — REQ-6: no `k='all'` string sentinel (`_univariate_selection.py:784-785`;
  ferrolearn's `usize` `k=0` already gives an empty selection), no pluggable
  `score_func` (chi2/f_regression/mutual_info; single `ScoreFunc::FClassif`), and no
  generalized `_clean_nans` exposed for arbitrary `score_func` outputs (`:24-33`).
  (The `k>n_features` clamp+keep-all is now done — REQ-3 / #1426.)
- #1429 — REQ-7: no `GenericUnivariateSelect` mode meta-selector
  (`_univariate_selection.py:1054`, modes `:1119-1125`) — routed parity-op, ABSENT.
- #1430 — REQ-8: no `SelectorMixin` surface (`get_support`/`inverse_transform`/
  `get_feature_names_out`) and no `pvalues_`/`scores_`/`n_features_in_` attribute
  contract (`_univariate_selection.py:567-575`).
- #1431 — REQ-9: duplicate basic `SelectFromModel` in `feature_selection.rs` vs the
  rich `SelectFromModelExt` in `select_from_model.rs` — parity owned by
  `.design/preprocess/select_from_model.md` (cross-ref, do not re-litigate); the
  two co-existing types are tech-debt.
- #1432 — REQ-10: no PyO3 `VarianceThreshold`/`SelectKBest`/`GenericUnivariateSelect`
  binding in `ferrolearn-python`.
- #1433 — REQ-11: selectors + helpers compute on `ndarray` / `num_traits`, not
  ferray (R-SUBSTRATE-1/2).
