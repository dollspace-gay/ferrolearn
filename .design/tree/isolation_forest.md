# IsolationForest — anomaly detection via isolation trees

<!--
tier: 3-component
status: draft
baseline-commit: c59f45ba
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/ensemble/_iforest.py   # IsolationForest(OutlierMixin, BaseBagging) (:28); _parameter_constraints (:199, max_samples=StrOptions{auto}|Interval(Integral,1,None)|Interval(RealNotInt,0,1); contamination=StrOptions{auto}|Interval(Real,0,0.5,closed=right); max_features=Integral|Interval(Real,0,1,closed=right); bootstrap=boolean); __init__ defaults (:221, n_estimators=100/max_samples='auto'/contamination='auto'/max_features=1.0/bootstrap=False/random_state=None/verbose=0/warm_start=False); _get_estimator -> ExtraTreeRegressor(max_features=1, splitter='random') (:250); fit (:268): max_samples_='auto'->min(256,n_samples) (:303-304) / int passthrough+warn (:306-316) / float->int(frac*n) (:317-318), max_depth=ceil(log2(max(max_samples,2))) (:321); offset_=-0.5 when contamination=='auto' (:341-345) else np.percentile(_score_samples(X), 100*contamination) (:353); predict (:357): is_inlier=ones; is_inlier[decision_function(X)<0]=-1 (:374-378); decision_function (:380) = score_samples(X) - offset_ (:410); score_samples (:412) = _score_samples(X) (:439); _score_samples (:441) = -_compute_chunked_score_samples(X) (:451, SIGN FLIP — opposite of paper); _compute_score_samples (:485): depths = decision_path + per-tree apl - 1, scores = 2**(-depths / (n_est * apl([max_samples]))) (:501-523); _average_path_length (:535): mask_1(n<=1)->0, mask_2(n==2)->1, else 2*(ln(n-1)+euler_gamma) - 2*(n-1)/n (:561-566)
  - sklearn/ensemble/_bagging.py   # BaseBagging._fit — the per-tree subsample (bootstrap=False => sample_without_replacement) + feature draw IsolationForest inherits
ferrolearn-module: ferrolearn-tree/src/isolation_forest.rs
parity-ops: IsolationForest
crosslink-issue: 725
-->

## Summary

`ferrolearn-tree/src/isolation_forest.rs` mirrors
`sklearn.ensemble.IsolationForest` (`_iforest.py:28`) — an unsupervised anomaly
detector that builds an ensemble of randomly-split isolation trees and scores a
point by its mean path length, normalized by `c(n)`, the average path length of
an unsuccessful BST search.

**The headline divergence is the scoring-object contract (R-DEV-3).** sklearn's
`score_samples` returns the **opposite** of the paper's anomaly score:
`_score_samples = -_compute_chunked_score_samples` (`_iforest.py:451`), so
`score_samples(X) = -2^(-mean_path / c(max_samples)) ∈ [-1, 0]` — **higher means
more NORMAL**, lower (more negative) means more anomalous (live-verified: all
`score_samples(X) <= 0`, min `-0.85`, max `-0.36`). ferrolearn's `score_samples`
returns the **positive** paper score `+2^(-mean/c) ∈ (0, 1]` — higher means more
**ANOMALOUS** — the inverted sign convention. Riding on that inversion, sklearn
thresholds through two attributes ferrolearn lacks entirely: `offset_`
(`= -0.5` when `contamination='auto'`, the paper threshold; else
`np.percentile(score_samples(X_train), 100*contamination)`) and
`decision_function(X) = score_samples(X) - offset_`; `predict` is then
`-1 where decision_function(X) < 0 else 1`. ferrolearn has **no
`decision_function`, no `offset_`, no `contamination='auto'`** — it thresholds
the positive scores at the contamination quantile (descending sort) and predicts
`-1 where score >= threshold`. These five pieces — score sign/range,
`decision_function`, `offset_`, `contamination='auto'`, and `predict` —
form **one coherent scoring-contract rework** (see Blockers).

The **deterministic, RNG-independent** parts — `c(n)` (`_average_path_length`,
exact match incl. Euler-gamma `0.5772156649015329`), the isolation-tree
structure + `max_depth = ceil(log2(max_samples))`, and `random_state`
reproducibility — are the shippable/pinnable contract. The **exact tree ensemble
and exact scores at a `random_state`** are a documented RNG boundary: sklearn
draws the per-tree subsample WITHOUT replacement via numpy MT19937
(`bootstrap=False`, `sample_without_replacement`) and uses an `ExtraTreeRegressor`
(`splitter='random'`); ferrolearn draws WITH replacement via `StdRng`
(`next_u64() % n_samples`). numpy-MT-vs-StdRng cannot bit-match — the same
boundary already accepted for `random_forest`/`extra_trees`/`bagging`.

sklearn's `IsolationForest` is structurally a `BaseBagging` of
`ExtraTreeRegressor` (`_iforest.py:28`/`:250`); ferrolearn re-implements the
isolation-tree ensemble **natively** (its own `IsoNode`/`build_isolation_tree`),
which is faithful to the paper and observably equivalent in structure — the
divergences that matter are the scoring contract above, not the bagging
substructure.

This doc adapts to the **existing** code under R-HONEST-3 (underclaim beats
overclaim). `c(n)`, the tree structure, and reproducibility are SHIPPED; the
entire scoring/threshold/predict contract is NOT-STARTED, as are the missing
params (`max_features`, `bootstrap`, `contamination='auto'`), the
sample-without-replacement draw, and the ferray substrate.

## Algorithm (sklearn — the contract)

### Estimator surface & defaults (live `get_params()`, sklearn 1.5.2)

`IsolationForest(*, n_estimators=100, max_samples='auto', contamination='auto',
max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0,
warm_start=False)` (`__init__`, `:221`). Live `get_params()`: `{'bootstrap':
False, 'contamination': 'auto', 'max_features': 1.0, 'max_samples': 'auto',
'n_estimators': 100, 'n_jobs': None, 'random_state': None, 'verbose': 0,
'warm_start': False}`.

**Defaults ferrolearn matches**: `n_estimators=100` (`fn new`),
`random_state=None`. **`max_samples` default**: sklearn `'auto' = min(256,
n_samples)` (`:303-304`); ferrolearn defaults the field to `256` then applies
`.min(n_samples)` in `fit` (`effective_max_samples`) — **numerically equivalent
for the default** (live `max_samples_=52` on a 52-row X equals ferrolearn's
`256.min(52)=52`). The remaining gap is representational: ferrolearn cannot
express the `'auto'` string nor an `int > 256` that means "exactly that many"
(it always `.min(n_samples)`s, which is correct for `'auto'` but not for an
explicit int) — a minor surface gap folded into REQ-1.

**Defaults ferrolearn DIVERGES on**: `contamination` — ferrolearn defaults `0.1`
(the pre-0.22 sklearn default; sklearn changed it to `'auto'` in 0.22, `:76-78`)
and validates `0.0..=0.5`, which **cannot express `'auto'`** at all. sklearn's
default `'auto'` path sets `offset_ = -0.5` (the paper threshold), a completely
different mechanism (REQ-6).

**Params ABSENT in ferrolearn** (REQ-1 flags each): `max_features` (feature
subsampling; sklearn `Integral | Interval(Real,0,1]`, `:210`), `bootstrap`
(sample-with-vs-without-replacement toggle, default `False`, `:214`/`:89`),
`n_jobs`, `verbose`, `warm_start` (Python/parallel ergonomics — `n_jobs`
subsumed by future rayon use; `warm_start`/`verbose` R-DEV-4 non-divergences).

### `c(n)` — average path length (`_average_path_length`, `:535`)

The normalization constant: `c(1)=0`, `c(2)=1`, else `2*(ln(n-1) + euler_gamma)
- 2*(n-1)/n` (`:561-566`). ferrolearn's `fn average_path_length` matches the
general branch exactly (Euler-gamma literal `0.5772156649015329` ==
`np.euler_gamma`; live `_average_path_length([256]) = 10.2447709...`). **Minor
edge note**: sklearn special-cases `n==2 -> 1.0` exactly (`:562`); ferrolearn
computes `c(2) = 2*0.5772... - 1 = 0.1544...` from the general formula and does
NOT special-case it. (sklearn's `n==2` special-case yields `1.0`, which the
general formula does NOT — `2*(ln(1)+γ) - 2*(1)/2 = 2γ - 1 ≈ 0.1544`.) This is a
real numerical divergence on the `n==2` leaf, folded into REQ-3.

### Subsample + tree build (`fit` / `BaseBagging._fit`, `_iforest.py:268`)

`max_samples_` resolved (`:303-318`): `'auto' -> min(256, n)`; `int` ->
passthrough (warn + clamp if `> n`); `float -> int(frac * n)`.
`max_depth = ceil(log2(max(max_samples, 2)))` (`:321`). Each of `n_estimators`
trees is an `ExtraTreeRegressor(max_features=1, splitter='random')` (`:250`) fit
on a subsample drawn by `BaseBagging._fit` — **WITHOUT replacement** because
`bootstrap=False` (default), via `sample_without_replacement` off numpy MT19937.
At each node the extra-tree splitter picks one random feature and a random split
in `[min, max]`.

ferrolearn (`fit`): `effective_max_samples = max_samples.min(n_samples)`,
`max_depth = (effective_max_samples as f64).log2().ceil()`. Per tree it draws
`effective_max_samples` indices **WITH replacement** (`(rng.next_u64() as usize)
% n_samples`, the loop ~270-275) and recursively builds an isolation tree
(`build_isolation_tree`): pick a random feature (`next_u64() % n_features`), find
its `[min, max]` over the node's points, pick a uniform threshold
(`random_threshold`), partition, recurse to `max_depth` or a singleton leaf.
**Structurally faithful** (random feature + random split in `[min,max]`,
`ceil(log2)` depth cap). **Divergences**: (1) WITH vs WITHOUT replacement
(REQ-7); (2) numpy MT19937 vs `StdRng` and `next_u64() % n` vs
`sample_without_replacement` (RNG boundary, REQ-7); (3) no `max_features` feature
subsampling (REQ-7). ferrolearn's `max_depth` uses `effective_max_samples`
directly while sklearn uses `max(max_samples, 2)` — identical for the default
(`max_samples >= 2` always in practice).

### Scoring (`_compute_score_samples`, `:485`; `_score_samples`, `:441`)

Per sample, sum over trees of `decision_path_length + c(n_leaf) - 1`, divide by
`n_estimators * c(max_samples)`, then `raw = 2^(-mean)`. The **public**
`score_samples` returns `-raw` (`_score_samples = -_compute_chunked_score_samples`,
`:451`): **negative, range [-1, 0], HIGHER = more NORMAL** (live: all `<= 0`).
ferrolearn's `score_samples` (~191-216) computes the SAME inner term
(`total_path / n_trees`, then `f64::powf(2.0, -mean_path / c_n)`) but returns it
**positive** (`+raw ∈ (0,1]`, HIGHER = more ANOMALOUS) — the **inverted sign**
(REQ-4). The mean-path arithmetic and `c(n)` normalization match; only the sign
(and the per-tree `-1` decision-path bookkeeping, a sklearn-internal artifact of
counting from `compute_node_depths`) differ.

### offset_, decision_function, predict (`:341`, `:380`, `:357`)

- `offset_` (`:341-353`): `-0.5` when `contamination=='auto'` (paper threshold —
  inliers' scores near 0, outliers' near -1); else `np.percentile(score_samples(
  X_train), 100*contamination)` (live: `contamination=0.1 -> offset_ =
  -0.49982...` == the 10th percentile of the negative scores).
- `decision_function(X) = score_samples(X) - offset_` (`:410`): subtracting
  `offset_` makes `0` the outlier threshold (live-verified equal).
- `predict` (`:374-378`): `is_inlier = ones; is_inlier[decision_function(X) < 0]
  = -1` — i.e. `-1` where `decision_function < 0`, else `+1` (live-verified
  exactly equal to `np.where(df < 0, -1, 1)`).

ferrolearn has **none** of these. `fit` thresholds the **positive** scores at the
contamination quantile: sort descending, take `sorted[ceil(contamination*n) -
1]` as `threshold` (~301-321); `predict` (~337-341) returns `-1 where score >=
threshold else 1`. On the default this is a *different mechanism* (quantile of
positive scores, no `offset_`, no `'auto'` -0.5 path) on the *inverted* sign
convention. It produces a similar inlier/outlier partition by accident on
contaminated data but is not the sklearn contract — REQ-5/REQ-6/REQ-8.

## ferrolearn (what exists)

- **Unfitted**: `pub struct IsolationForest<F>` (public fields `n_estimators`,
  `max_samples`, `contamination: f64`, `random_state: Option<u64>`); builders
  `with_n_estimators`, `with_max_samples`, `with_contamination`,
  `with_random_state`; `Default` / `fn new` (`n_estimators=100`,
  `max_samples=256`, `contamination=0.1`, `random_state=None`).
- **Fitted**: `pub struct FittedIsolationForest<F>` (`trees: Vec<Vec<IsoNode<F>>>`,
  `n_features`, `threshold: f64`, `max_samples`); accessors `fn n_estimators`,
  `fn n_features`, `fn threshold`, `fn score_samples`.
- **Traits**: `Fit<Array2<F>, ()>` (unsupervised — `y = ()`); `Predict<Array2<F>>`
  with `Output = Array1<isize>` (+1 / -1).
- **Internal**: `enum IsoNode<F>` (`Split`/`Leaf`), `fn build_isolation_tree`,
  `fn path_length`, `fn average_path_length` (the `c(n)` mirror),
  `fn random_threshold`.
- **NO** `decision_function`, **NO** `offset_`, **NO** `contamination='auto'`,
  **NO** `max_features`, **NO** `bootstrap`.
- **Consumers (non-test)**: crate re-export — `ferrolearn-tree/src/lib.rs`
  (`pub use isolation_forest::{FittedIsolationForest, IsolationForest}`, plus the
  module-doc reference). There is **NO PyO3 binding** for `IsolationForest`
  (verified: nothing matching `Isolation`/`iforest` under
  `ferrolearn-python/src/`) — the crate re-export is the only non-test
  production consumer. `IsolationForest` is a grandfathered boundary estimator
  type (S5/R-DEFER-1: the public estimator type IS the public API; its consumers
  are external users + the future PyO3 binding).

## Requirements

- REQ-1: **Param surface + defaults (R-DEV-2).** `n_estimators=100`,
  `random_state=None` match; `max_samples` default is numerically equivalent to
  `'auto'` (`256.min(n)` == `min(256,n)`). DIVERGES: `contamination` default
  `0.1` vs sklearn `'auto'` and cannot express `'auto'` (REQ-6). ABSENT:
  `max_features`, `bootstrap` (REQ-7), `n_jobs`, `verbose`, `warm_start`; the
  `'auto'`/`int`/`float` string-or-numeric `max_samples` representation
  (`_iforest.py:199`/`:221`).
- REQ-2: **Isolation-tree build + `max_depth = ceil(log2(max_samples))`.** Random
  feature, random split in `[min,max]`, recurse to `ceil(log2(effective_max_
  samples))` depth or singleton leaf — structurally faithful to sklearn's
  `ExtraTreeRegressor(splitter='random')` ensemble (`:250`, `:321`). Exact tree
  at a seed = RNG boundary (REQ-7).
- REQ-3: **`c(n)` average path length.** `c(1)=0`, general branch `2*(ln(n-1) +
  euler_gamma) - 2*(n-1)/n` matches `_average_path_length` (`:561-566`) incl.
  Euler-gamma. KNOWN micro-gap: sklearn special-cases `c(2)=1.0` (`:562`);
  ferrolearn computes `c(2)=0.1544...` from the general formula.
- REQ-4: **`score_samples` sign/range (HEADLINE, R-DEV-3).** sklearn returns
  `-2^(-mean/c) ∈ [-1, 0]` (HIGHER = more NORMAL, `_score_samples` SIGN FLIP
  `:451`); ferrolearn returns `+2^(-mean/c) ∈ (0,1]` (HIGHER = more ANOMALOUS).
  Inverted convention.
- REQ-5: **`decision_function`.** `decision_function(X) = score_samples(X) -
  offset_` (`:410`); ferrolearn has no such method.
- REQ-6: **`offset_` + `contamination='auto'`.** `offset_ = -0.5` when
  `contamination=='auto'` (`:341-345`), else `np.percentile(score_samples(
  X_train), 100*contamination)` (`:353`). Needs a `Contamination{Auto,
  Value(f64)}` enum (default `Auto`). ferrolearn has neither `offset_` nor an
  `'auto'` path; it thresholds the positive scores at the contamination quantile.
- REQ-7: **Sample-without-replacement + `max_features` + `bootstrap`.** sklearn
  `bootstrap=False` (default) draws WITHOUT replacement
  (`sample_without_replacement`); ferrolearn draws WITH replacement
  (`next_u64() % n_samples`). `max_features`/`bootstrap` params absent. Exact
  ensemble/score at a `random_state` = numpy-MT-vs-StdRng RNG boundary.
- REQ-8: **`predict` via `decision_function < 0`.** sklearn `-1 where
  decision_function(X) < 0 else 1` (`:374-378`); ferrolearn `-1 where score >=
  quantile-threshold else 1` (~337-341) — different mechanism on the inverted
  sign.
- REQ-9: **`random_state` determinism** — same seed -> identical ensemble
  (ferrolearn reproducibility, NOT numpy parity). RNG boundary for cross-sklearn.
- REQ-10: **ferray substrate (R-SUBSTRATE).** Imports `ndarray`/`rand`, not
  `ferray-core`/`ferray::random`.

## Acceptance criteria

- AC-1: live `IsolationForest().get_params()` equals the REQ-1 defaults
  (live-verified: `n_estimators=100`, `max_samples='auto'`,
  `contamination='auto'`, `max_features=1.0`, `bootstrap=False`,
  `random_state=None`); `max_samples_` on a 52-row X is `52` == ferrolearn
  `256.min(52)`; the `contamination='auto'` and `max_features`/`bootstrap` gaps
  are enumerated.
- AC-2: an isolation tree's depth never exceeds `ceil(log2(effective_max_
  samples))`; a clear outlier gets a shorter mean path than the bulk (covered by
  `test_anomaly_scores`, `test_anomaly_detected`).
- AC-3: `average_path_length(n)` matches `_average_path_length([n])` for `n in
  {1, 3, 10, 256}` to 1e-12 (live `_average_path_length([256])=10.2447709...`);
  the `n==2` value DIVERGES (sklearn `1.0` vs ferrolearn `0.1544...`) — pins the
  REQ-3 micro-gap.
- AC-4: live `IsolationForest(random_state=0).fit(X).score_samples(X)` is `<= 0`
  for every row (live-verified True, min `-0.85`); ferrolearn `score_samples`
  returns positive values — pins the REQ-4 sign inversion.
- AC-5: live `decision_function(X) == score_samples(X) - offset_` (True) and
  `offset_ == -0.5` for `contamination='auto'`, `== np.percentile(score_samples,
  100*contamination)` otherwise (live: `-0.49982...` for `0.1`) — pins
  REQ-5/REQ-6.
- AC-6: live `predict(X) == np.where(decision_function(X) < 0, -1, 1)` (True) —
  pins REQ-8.
- AC-7: `random_state` reproducibility — two `fit` calls with the same seed
  produce identical `predict` (covered by `test_reproducibility`).

## REQ status table

Binary (R-DEFER-2). `IsolationForest`/`FittedIsolationForest` are grandfathered
boundary estimator types re-exported at the crate root (`lib.rs`); there is no
PyO3 binding yet (S5/R-DEFER-1: the estimator type is the public API; the crate
re-export is the non-test consumer). Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (param surface + defaults) | SHIPPED (with gaps flagged) | `fn new` sets `n_estimators=100`, `random_state=None`; `max_samples=256` then `effective_max_samples = max_samples.min(n_samples)` in `fit` == sklearn `'auto' = min(256, n)` (`_iforest.py:303-304`) — live `max_samples_=52` on 52-row X == `256.min(52)`. Consumer: crate re-export (`lib.rs`). Verification: `python3 -c "from sklearn.ensemble import IsolationForest as I; print(I().get_params())"` (matches the exposed subset). Tests: `test_isolation_forest_default` (now asserts `contamination == Contamination::Auto`), `test_isolation_forest_builder`, `test_contamination_auto_builder`, `test_max_samples_larger_than_data`. The `contamination` default now matches sklearn `'auto'` via `Contamination::Auto` (#726 contamination-default part closed); STILL ABSENT: `max_features`/`bootstrap` and the `'auto'`/`int`/`float` `max_samples` string-or-numeric representation → remainder of #726. |
| REQ-2 (isolation-tree build + max_depth) | SHIPPED (structural; RNG boundary) | `build_isolation_tree` (random feature `next_u64() % n_features`, random `threshold` in `[min,max]` via `random_threshold`, recurse to `max_depth`) mirrors sklearn's `ExtraTreeRegressor(splitter='random')` ensemble (`_iforest.py:250`); `max_depth = (effective_max_samples as f64).log2().ceil()` in `fit` == `ceil(log2(max(max_samples,2)))` (`:321`). Consumer: `score_samples`/`predict` traverse via `path_length`; crate re-export. Tests: `test_fit_predict_basic`, `test_anomaly_detected`, `test_anomaly_scores`, `test_single_sample`, `test_f32`. Exact tree at a seed = RNG boundary (#732). |
| REQ-3 (c(n) average path length) | SHIPPED | `fn average_path_length`: `c(1)=0`, `c(2)=1.0` (special-cased, mirroring `mask_2`), else `2*((n-1).ln() + 0.5772156649015329) - 2*(n-1)/n` — matches `_average_path_length` (`_iforest.py:557-566`) incl. the `n<=1`/`n==2` special-cases; Euler-gamma literal == `np.euler_gamma`. Consumer: called by `score_samples` and `path_length`. Verification: live `_average_path_length([1])==0., ([2])==1., ([256])==10.244770920119917` all match. Tests: `test_average_path_length_values` (`c(1)==0`, `c(2)==1.0`, `c(256)==10.2447709...`). The `c(2)` micro-gap (#727) is now closed: ferrolearn returns `1.0` for `n==2` (`_iforest.py:562`). |
| REQ-4 (score_samples sign/range — HEADLINE) | SHIPPED | `fn score_samples` now returns `-f64::powf(2.0, -mean_path / c_n)` — NEGATIVE ∈ [-1,0] (HIGHER = NORMAL), matching sklearn `_score_samples = -_compute_chunked_score_samples` (`_iforest.py:451`). Consumer: `decision_function` and `predict` (in-crate) + crate re-export (`lib.rs`). Tests: `test_score_samples_sign_in_minus_one_zero`, `test_anomaly_scores` (in-src), and `divergence_score_samples_sign_le_zero` (now GREEN). The inner mean-path/`c(n)` arithmetic was already correct; the SIGN is now flipped. |
| REQ-5 (decision_function) | SHIPPED | `fn decision_function(&self, x) -> Result<Array1<f64>, FerroError>` = `score_samples(x) - offset_` (sklearn `_iforest.py:410`). Consumer: called by `predict` (in-crate) + crate re-export. Tests: `test_decision_function_equals_score_minus_offset` (in-src), `contract_decision_function_equals_score_minus_offset` (divergence). |
| REQ-6 (offset_ + contamination='auto') | SHIPPED | New `pub enum Contamination { Auto, Value(f64) }` (default `Auto`, re-exported from `lib.rs`); `IsolationForest.contamination` is now `Contamination`. `fit` sets `offset_ = -0.5` for `Contamination::Auto` (`_iforest.py:341-345`) else `percentile(score_samples(X_train), 100*v)` via `fn percentile` (numpy default linear interpolation, `_iforest.py:353`); stored on `FittedIsolationForest` and exposed via `fn offset`. `with_contamination(v)` sets `Value(v)` (validated `0 < v <= 0.5` at fit, `_iforest.py:199`); `with_contamination_auto()` sets `Auto`. Consumer: `offset_` feeds `decision_function`/`predict`; the enum is re-exported. Tests: `test_offset_auto_is_minus_half`, `test_offset_value_is_percentile_of_scores`, `test_percentile_linear_interpolation` (in-src), `contract_offset_auto_is_minus_half` (divergence). |
| REQ-7 (sample-without-replacement + max_features + bootstrap) | NOT-STARTED | open prereq blocker #731. `fit` draws `(rng.next_u64() as usize) % n_samples` WITH replacement (~270-275); sklearn `bootstrap=False` (default, `_iforest.py:228`) draws WITHOUT replacement (`sample_without_replacement` via `_bagging.py`). `max_features` (`:210`) and `bootstrap` (`:214`) params absent. Exact ensemble/score at a `random_state` = numpy-MT-vs-StdRng RNG boundary (#732). |
| REQ-8 (predict via decision_function < 0) | SHIPPED | `predict` now computes `decision_function(X)` and returns `-1` where `d < 0` else `1` (sklearn `is_inlier[decision_function(X) < 0] = -1`, `_iforest.py:374-378`); the old quantile-threshold field is removed. Consumer: crate re-export + Predict impl. Tests: `test_predict_agrees_with_decision_function_sign` (in-src), `contract_predict_agrees_with_decision_function_sign` (divergence), `test_anomaly_detected`/`conformance_isolation_forest` (predict agreement). |
| REQ-9 (random_state determinism) | SHIPPED | `fit` seeds `StdRng::seed_from_u64(random_state)` (else `from_os_rng`); same seed → identical trees → identical `predict`. Consumer: crate re-export. Tests: `test_reproducibility` (two `fit` calls, same seed, equal `predict`). ferrolearn reproducibility; exact numpy-MT parity is the RNG boundary (#732). |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #733. Imports `ndarray`/`rand`, not `ferray-core`/`ferray::random` (R-SUBSTRATE). |

## Architecture

`IsolationForest<F>` is the unfitted boundary type (public `n_estimators`,
`max_samples`, `contamination: f64`, `random_state: Option<u64>`; `with_*`
builders; `Default`). `fit` validates (`InsufficientSamples` for zero rows,
`InvalidParameter` for `n_estimators==0`/`max_samples==0`/`contamination` outside
`0.0..=0.5`), computes `effective_max_samples = max_samples.min(n_samples)` and
`max_depth = ceil(log2(effective_max_samples))`, seeds a `StdRng`, then for each
of `n_estimators` draws `effective_max_samples` row indices WITH replacement and
calls `build_isolation_tree` (recursive: random feature, `[min,max]` random
threshold, partition, recurse). After building, it scores the training rows and
sets a `threshold` by descending-sorting the POSITIVE scores and indexing the
contamination quantile.

`FittedIsolationForest<F>` stores `trees`, `n_features`, `threshold`,
`max_samples`. `score_samples` traverses every tree via `path_length` (which adds
`c(n_leaf)` at each leaf), means over trees, and returns `2^(-mean / c(max_
samples))` — POSITIVE (the REQ-4 sign inversion). `predict` returns `-1 where
score >= threshold else 1`.

The faithful sklearn contract requires a coherent rework of the scoring path,
**all in `isolation_forest.rs`** plus a one-line `lib.rs` re-export for the new
`Contamination` enum: (a) flip `score_samples` to return `-2^(-mean/c)` (REQ-4);
(b) add a `Contamination{Auto, Value(f64)}` enum (default `Auto`) replacing the
`f64` field (REQ-6); (c) compute `offset_` in `fit` (`-0.5` for `Auto`, else the
`contamination`-percentile of the NEGATIVE scores) and store it on
`FittedIsolationForest` (REQ-6); (d) add `fn decision_function(X) =
score_samples(X) - offset_` (REQ-5); (e) rewire `predict` to `-1 where
decision_function(X) < 0 else 1` (REQ-8). This is a builder-scale **single-file**
change (the manifest is `isolation_forest.rs` + the `lib.rs` re-export line for
`Contamination`), not cross-crate.

**Deterministically pinnable** (no RNG dependence): the `score_samples <= 0`
contract (REQ-4 — assertable on any fitted model), `decision_function`'s
existence and `== score_samples - offset_` identity (REQ-5), `offset_ == -0.5`
for `'auto'` and `== percentile` otherwise (REQ-6), the `predict ==
decision_function < 0` partition (REQ-8), `c(n)` values (REQ-3), and `random_
state` reproducibility (REQ-9). **RNG boundary** (NOT pinnable cross-sklearn): the
exact tree ensemble, the exact path lengths, and therefore exact `score_samples`
*values* at a given `random_state` — because the subsample draw diverges
(WITH-replacement `StdRng` vs WITHOUT-replacement numpy MT19937) and the tree
splitter's random stream differs. Same boundary as
`random_forest`/`extra_trees`/`bagging`.

## Verification

Library crate (green at baseline `c59f45ba`):
```
cargo test -p ferrolearn-tree --lib isolation_forest   # existing suite passes
cargo clippy -p ferrolearn-tree --all-targets -- -D warnings
cargo fmt --all --check
```
Live sklearn oracle (installed 1.5.2, run from `/tmp`):
```
# REQ-1 defaults
python3 -c "from sklearn.ensemble import IsolationForest as I; print(I().get_params())"
# REQ-3 c(n)  (general branch matches; n==2 micro-gap: sklearn 1.0 vs ferrolearn 0.1544)
python3 -c "from sklearn.ensemble._iforest import _average_path_length as a; print(a([1]),a([2]),a([256]))"  # [0.] [1.] [10.2447709...]
# REQ-4 score_samples sign (all <= 0)  +  REQ-5/6 offset_/decision_function  +  REQ-8 predict
python3 -c "
import numpy as np; from sklearn.ensemble import IsolationForest
rng=np.random.RandomState(0); X=np.vstack([rng.randn(50,2),[[100.,100.],[-80.,90.]]])
m=IsolationForest(random_state=0).fit(X); s=m.score_samples(X); d=m.decision_function(X)
print('score<=0', bool((s<=0).all()), 'offset_', m.offset_, 'df==s-off', bool(np.allclose(d,s-m.offset_)),
      'pred==df<0', bool((m.predict(X)==np.where(d<0,-1,1)).all()))
m2=IsolationForest(random_state=0,contamination=0.1).fit(X)
print('offset_(0.1)', m2.offset_, '==pctile', float(np.percentile(m2.score_samples(X),10.0)))"
```
NOT-STARTED REQs (4, 5, 6, 7, 8, 10) have no green verification by construction —
each carries an open prereq blocker. The HEADLINE REQ-4 + REQ-5/6/8 are pinnable
deterministically (no RNG): the sign contract, `decision_function`/`offset_`
identities, and the `predict == decision_function < 0` partition assert on ANY
fitted model regardless of the exact ensemble. A characterization pin (R-CHAR-3)
belongs in `ferrolearn-tree/tests/divergence_isolation_forest.rs` with expected
values from the live oracle above: `score_samples <= 0` (currently fails —
ferrolearn returns positive), `decision_function` existence + `offset_=-0.5` for
`'auto'`. REQ-1/2/3(general)/9 are verified by the in-crate `#[test]`s named in
the table.

## Blockers to open

- #726 — REQ-1: PARTIALLY RESOLVED — the `contamination` default now matches
  sklearn `'auto'` via the `Contamination::Auto` default. STILL OPEN:
  `max_features` (`:210`), `bootstrap` (`:214`), and the `'auto'`/`int`/`float`
  `max_samples` string-or-numeric representation remain absent/under-typed.
- #727 — REQ-3: RESOLVED — `average_path_length` now special-cases
  `n==2 -> 1.0` (`_iforest.py:562`); was `0.1544...` from the general formula.
- #728 — REQ-4 (HEADLINE, R-DEV-3): RESOLVED — `score_samples` now returns
  `-2^(-mean/c) ∈ [-1,0]` (`_score_samples = -_compute_chunked_score_samples`,
  `_iforest.py:451`); pin `divergence_score_samples_sign_le_zero` is GREEN.
- #729 — REQ-5/REQ-8: RESOLVED — `fn decision_function(X) = score_samples(X) -
  offset_` (`_iforest.py:410`) added; `predict = -1 where decision_function < 0
  else 1` (`:374-378`).
- #730 — REQ-6: RESOLVED — added `Contamination{Auto, Value(f64)}` enum
  (default `Auto`, re-exported from `lib.rs`), the `offset_` field/accessor, and
  `offset_ = -0.5` for `'auto'` (`_iforest.py:341-345`) else
  `np.percentile(score_samples, 100*contamination)` (`:353`).
- #731 — REQ-7: subsample drawn WITH replacement (`next_u64() % n_samples`,
  ~270-275) vs sklearn `bootstrap=False` WITHOUT replacement
  (`sample_without_replacement`); `max_features` (`:210`) / `bootstrap` (`:214`)
  params absent.
- #732 — REQ-2/REQ-7/REQ-9: exact tree ensemble + exact `score_samples` values at
  a `random_state` is a numpy-MT-vs-StdRng RNG boundary — documented like
  random_forest/extra_trees/bagging, NOT a fixable divergence.
- #733 — REQ-10: migrate `isolation_forest.rs` off `ndarray`/`rand` to the ferray
  substrate (R-SUBSTRATE).

> NOTE: REQ-4 (#728), REQ-5 (#729), REQ-6 (#730), and REQ-8 (folded into
> #729/#730) form ONE coherent scoring-contract rework — a builder-scale
> single-file change in `isolation_forest.rs` (+ a `lib.rs` re-export for the
> `Contamination` enum): flip the `score_samples` sign, add the `Contamination`
> enum + `offset_` field, add `decision_function`, and rewire `predict` through
> `decision_function < 0`. Ship them together (one builder), not piecemeal.
