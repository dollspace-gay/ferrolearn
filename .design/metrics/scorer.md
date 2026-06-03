# Scorer Machinery (sklearn.metrics scorer registry)

<!--
tier: 3-component
status: draft
baseline-commit: db2cb592
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/metrics/_scorer.py   # _BaseScorer (:197); _BaseScorer.__call__ (:241); _Scorer._score (:331, applies self._sign at :376); get_scorer (:385); make_scorer (:655, greater_is_better=True default :659, response_method=None :658, sign=1 if greater_is_better else -1 :754); _SCORERS dict (:867); get_scorer_names (:908, sorted keys :930); the precision/recall/f1/jaccard average-variant loop (:933-942)
ferrolearn-module: ferrolearn-metrics/src/scorer.rs
parity-ops: make_scorer, get_scorer, get_scorer_names, Scorer, check_scoring
crosslink-issue: 778  # ferrolearn-metrics scorer unit tracking issue
-->

## Summary

`ferrolearn-metrics/src/scorer.rs` mirrors the scorer machinery of
scikit-learn's `sklearn/metrics/_scorer.py`: `make_scorer`, `get_scorer`,
`get_scorer_names`, the `_SCORERS` name→scorer registry, and the `_BaseScorer`/
`_Scorer` callable. ferrolearn provides a `Scorer<F>` struct, `make_scorer`,
`Scorer::score`, `Scorer::sign`, a `get_scorer(name)` registry over **11** of
sklearn's **56** scorer names, `get_scorer_names`, and a `check_scoring`
convenience.

Under honest underclaim (R-HONEST-3), the scorer machinery **diverges from the
live sklearn 1.5.2 oracle on two deterministic, structural contracts** plus the
registry-coverage gap:

1. **The sign-application contract is inverted (the headline divergence).**
   sklearn's scorer is a *callable* whose evaluation **bakes in `self._sign`** —
   `_Scorer._score` returns `self._sign * self._score_func(y_true, y_pred)`
   (`_scorer.py:376`). So `get_scorer("neg_mean_squared_error")(est, X, y)`
   returns **`-mse`**. ferrolearn's `Scorer::score` returns the **raw** metric
   (`+mse`); `sign()` is a *separate accessor* the caller must remember to apply.
   A model-selection loop that does `scorer.score(...)` and maximises gets the
   sign backwards for every `neg_*` / loss scorer. This is the single cleanest
   deterministic critic-pin this iteration.
2. **`make_scorer` ABI is narrower than sklearn (R-DEV-2).** sklearn:
   `make_scorer(score_func, *, response_method=None, greater_is_better=True,
   **kwargs)` (`:655-663`) — keyword-only after `score_func`, a **`True`
   default** for `greater_is_better`, a `response_method` selector, and `**kwargs`
   forwarded to the metric. ferrolearn: `make_scorer(score_fn, greater_is_better,
   name)` — `greater_is_better` is a **required positional** (no default), there
   is **no `response_method`** and **no `**kwargs`** forwarding, and there is a
   ferrolearn-only `name` argument with no sklearn analog.
3. **The registry covers 11 of 56 names, and one of the 11 uses a non-sklearn
   name.** ferrolearn registers `neg_max_error`; **sklearn has no
   `neg_max_error`** — the canonical name is **`max_error`** with `_sign == -1`
   (`:761,870`). 45 sklearn scorer names are unregistered (classification,
   clustering, and the two regression scorers `max_error` + `d2_absolute_error_score`).

`scorer.rs` is re-exported at the crate root (`lib.rs`: `pub use scorer::{Scorer,
make_scorer, get_scorer, get_scorer_names, check_scoring, BUILTIN_SCORER_NAMES,
ScoringInput}`) and consumes `regression.rs` metric functions inside
`get_scorer`. The re-export is the non-test production-consumer surface; the
`Scorer`/`make_scorer`/`get_scorer`/`get_scorer_names` public API is grandfathered
(S5/R-DEFER-1). The existing `#[test]`s pin only ferrolearn's narrower (raw-score,
11-name) behavior and do not establish sklearn parity.

## Algorithm (sklearn — the contract)

A **scorer = `sign * metric`**, evaluated as a callable on `(estimator, X, y)`:

- **`make_scorer(score_func, *, response_method=None, greater_is_better=True,
  **kwargs)`** (`:655`): `sign = 1 if greater_is_better else -1` (`:754`);
  `response_method` defaults to `"predict"` (`:635,694`); returns a `_Scorer`
  closing over `(score_func, sign, kwargs, response_method)` (`:755`). A score
  function (`accuracy`, `r2`) keeps the default `greater_is_better=True`
  (`sign=+1`); a loss (`mean_squared_error`, `log_loss`) passes
  `greater_is_better=False` (`sign=-1`), and the scorer name is prefixed `neg_`.
- **`_BaseScorer.__call__(estimator, X, y_true, sample_weight=None, **kwargs)`**
  (`:241`): obtains `y_pred` via the estimator's `response_method`
  (`predict`/`predict_proba`/`decision_function`), then delegates to `_score`.
- **`_Scorer._score(...)`** (`:331`): the sign is applied here —
  **`return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)`**
  (`:376`). The returned value is *already signed* so a maximiser always works
  ("greater is better" by construction).
- **`get_scorer(scoring)`** (`:385`): `copy.deepcopy(_SCORERS[scoring])` if a
  string (a fresh copy each call, `:423`), else returns the callable as-is; raises
  `ValueError` listing `get_scorer_names()` on an unknown string (`:425-429`).
- **`get_scorer_names()`** (`:908`): `sorted(_SCORERS.keys())` (`:930`) — the **56**
  names below.
- **`_SCORERS`** (`:867`) is built statically (`explained_variance`, `r2`,
  `max_error`, regression `neg_*`, `d2_absolute_error_score`, classification,
  clustering) and then extended (`:933-942`) by a loop that, for each of
  `precision`/`recall`/`f1`/`jaccard`, registers a bare name
  (`average="binary"`) plus four `_macro`/`_micro`/`_samples`/`_weighted`
  variants (`average=<m>, pos_label=None`) — 4×5 = 20 names from that loop alone.

### The full 56-name sklearn canon (live `get_scorer_names()`, sklearn 1.5.2)

```
accuracy, adjusted_mutual_info_score, adjusted_rand_score, average_precision,
balanced_accuracy, completeness_score, d2_absolute_error_score,
explained_variance, f1, f1_macro, f1_micro, f1_samples, f1_weighted,
fowlkes_mallows_score, homogeneity_score, jaccard, jaccard_macro, jaccard_micro,
jaccard_samples, jaccard_weighted, matthews_corrcoef, max_error,
mutual_info_score, neg_brier_score, neg_log_loss, neg_mean_absolute_error,
neg_mean_absolute_percentage_error, neg_mean_gamma_deviance,
neg_mean_poisson_deviance, neg_mean_squared_error, neg_mean_squared_log_error,
neg_median_absolute_error, neg_negative_likelihood_ratio,
neg_root_mean_squared_error, neg_root_mean_squared_log_error,
normalized_mutual_info_score, positive_likelihood_ratio, precision,
precision_macro, precision_micro, precision_samples, precision_weighted, r2,
rand_score, recall, recall_macro, recall_micro, recall_samples, recall_weighted,
roc_auc, roc_auc_ovo, roc_auc_ovo_weighted, roc_auc_ovr, roc_auc_ovr_weighted,
top_k_accuracy, v_measure_score
```

### Per-name `_sign` for the 11 ferrolearn registers (live probe)

| ferrolearn name | sklearn name | sklearn `_sign` | metric `greater_is_better` |
|---|---|---|---|
| `neg_mean_absolute_error` | same | `-1` | false ✓ |
| `neg_mean_squared_error` | same | `-1` | false ✓ |
| `neg_root_mean_squared_error` | same | `-1` | false ✓ |
| `neg_mean_squared_log_error` | same | `-1` | false ✓ |
| `neg_root_mean_squared_log_error` | same | `-1` | false ✓ |
| `neg_mean_absolute_percentage_error` | same | `-1` | false ✓ |
| `neg_median_absolute_error` | same | `-1` | false ✓ |
| `neg_max_error` | **`max_error`** (no `neg_max_error` exists) | `-1` | false ✓ (sign) / **wrong name** |
| `r2` | same | `+1` | true ✓ |
| `explained_variance` | same | `+1` | true ✓ |
| `neg_mean_poisson_deviance` | same | `-1` | false ✓ |
| `neg_mean_gamma_deviance` | same | `-1` | false ✓ |

The 11 names ferrolearn shares with sklearn all carry the **correct
`greater_is_better`/sign metadata** (the `false`→`-1`, `true`→`+1` mapping is
right). The metadata is correct; the *application* of it in `score()` is the
divergence (REQ-2).

## ferrolearn (what exists)

In `ferrolearn-metrics/src/scorer.rs`:

- **`pub struct Scorer<F>`** — fields `score_fn: fn(&Array1<F>, &Array1<F>) ->
  Result<F, FerroError>`, `greater_is_better: bool`, `name: String`. The function
  pointer signature is **regression-shaped** (`&Array1<F>` → `&Array1<F>`); it
  cannot wrap a classification metric (`&Array1<usize>` labels), a probabilistic
  metric (`&Array2<f64>` proba), a clustering metric (label arrays), or a
  ranking/threshold metric (score arrays). No `response_method`, no `kwargs`, no
  `sign` baked into the type.
- **`pub fn make_scorer<F: Float>(score_fn, greater_is_better, name)`** — three
  **positional** args; `greater_is_better` has no default; no `response_method`,
  no `**kwargs`.
- **`pub fn Scorer::score(&self, y_true, y_pred)`** — returns the **raw**
  `(self.score_fn)(y_true, y_pred)`; **does not apply the sign**.
- **`pub fn Scorer::sign(&self) -> F`** — separate accessor returning `+1`/`-1`
  from `greater_is_better`; the caller must multiply manually.
- **`pub const BUILTIN_SCORER_NAMES: &[&str]`** — the **11**-name (regression-only)
  list; **not sorted** (sklearn `get_scorer_names` returns `sorted(...)`).
- **`pub fn get_scorer_names() -> &'static [&'static str]`** — returns
  `BUILTIN_SCORER_NAMES` (11 names; sklearn returns 56 sorted).
- **`pub fn get_scorer(name) -> Result<Scorer<f64>, FerroError>`** — `match` over
  the 11 names, each mapping to a `regression.rs` function via `make_scorer`;
  unknown name → `FerroError::InvalidParameter` (sklearn raises `ValueError`).
- **`pub fn check_scoring(ScoringInput) -> Result<Scorer<f64>, FerroError>`** +
  **`pub enum ScoringInput<'a> { Name(&str), Scorer(Scorer<f64>) }`** — a
  pass-through helper loosely mirroring sklearn `check_scoring`'s string/callable
  branches (no estimator argument, no multimetric `list`/`dict`/`set` path, no
  `_PassthroughScorer` for `None`).

**Underlying metric functions already present in the crate** (so the 45 missing
scorers are blocked on the *scorer type + registration*, NOT on the metric math):
`classification.rs` has `accuracy_score`, `precision_score`, `recall_score`,
`f1_score`, `jaccard_score`, `balanced_accuracy_score`, `matthews_corrcoef`,
`log_loss`, `brier_score_loss`, `roc_auc_score`, `average_precision_score`,
`top_k_accuracy_score`; `clustering.rs` has `adjusted_rand_score`,
`adjusted_mutual_info`, `mutual_info_score`, `normalized_mutual_info_score`,
`v_measure_score`, `homogeneity_score`, `completeness_score`,
`fowlkes_mallows_score`, `rand_score`; `regression.rs` has `max_error` and
`d2_absolute_error_score`. (Not present as a function: `class_likelihood_ratios`
for the two likelihood-ratio scorers.)

**Consumers (non-test):** crate re-export (`lib.rs`: `pub use scorer::{Scorer,
make_scorer, get_scorer, get_scorer_names, check_scoring, BUILTIN_SCORER_NAMES,
ScoringInput}`); `get_scorer` itself consumes `regression.rs` functions. Existing
pub API, grandfathered (S5/R-DEFER-1). **No `ferrolearn-python` binding** exposes
the scorer machinery (REQ-7).

## Requirements

- REQ-1: **`make_scorer` / `Scorer` / `get_scorer` / `get_scorer_names` machinery
  for the 11-name regression subset (R-DEV-1/2).** The registry resolves the 11
  shared names to the correct metric with the correct `greater_is_better`/sign
  *metadata*, and `make_scorer`/`get_scorer_names` exist. **SHIPPED IFF the sign
  is correctly baked into evaluation (REQ-2)** — otherwise the machinery produces
  the wrong scored value and is not a faithful mirror.
- REQ-2: **Sign-application contract — `sign * metric` (THE HEADLINE, R-DEV-3).**
  A scorer's evaluation must return `self._sign * score_func(y_true, y_pred)`
  (`_scorer.py:376`), so `get_scorer("neg_mean_squared_error").score(y, yhat)`
  returns **`-mse`**. ferrolearn's `score()` returns **`+mse`** (raw); `sign()` is
  a separate accessor never applied inside `score()`.
- REQ-3: **`make_scorer` ABI parity (R-DEV-2).** Match `make_scorer(score_func, *,
  response_method=None, greater_is_better=True, **kwargs)` (`:655-663`):
  keyword-only after `score_func`, `greater_is_better` **defaults to `True`**,
  a `response_method` selector, `**kwargs` forwarded to the metric. ferrolearn:
  positional `greater_is_better` (no default), no `response_method`, no `kwargs`.
- REQ-4: **Full 56-name registry — classification scorers (R-DEV-1/2).** Register
  the classification scorers sklearn's `_SCORERS` defines (`:790-942`):
  `accuracy`, `balanced_accuracy`, `matthews_corrcoef`, `neg_log_loss`,
  `neg_brier_score`, `top_k_accuracy`, `average_precision`, the `roc_auc` family
  (`roc_auc`/`roc_auc_ovo`/`roc_auc_ovr`/`*_weighted`), the likelihood-ratio
  scorers, and the `f1`/`precision`/`recall`/`jaccard` bare+`macro`/`micro`/
  `samples`/`weighted` variants. The `Scorer<F>` `fn(&Array1<F>,&Array1<F>)` type
  cannot wrap `&Array1<usize>`/`&Array2<f64>` metrics — blocked on a
  heterogeneous scorer type + `average`/`pos_label`/`multi_class` parameterization.
- REQ-5: **Full 56-name registry — clustering scorers (R-DEV-1/2).** Register
  `adjusted_rand_score`, `rand_score`, `homogeneity_score`, `completeness_score`,
  `v_measure_score`, `mutual_info_score`, `adjusted_mutual_info_score`,
  `normalized_mutual_info_score`, `fowlkes_mallows_score` (`:855-904`). Metric
  functions exist in `clustering.rs`; blocked on the same heterogeneous scorer
  type + registration.
- REQ-6: **Regression scorer-name canon fix (R-DEV-2).** Register **`max_error`**
  (sklearn name, `_sign == -1`, `:761,870`) and **`d2_absolute_error_score`**
  (`_sign == +1`, `:788,881`); **remove the non-sklearn `neg_max_error`** name.
  Make `get_scorer_names()` return the **sorted** name list (`:930`).
- REQ-7: **PyO3 binding (R-DEFER-1).** `import sklearn.metrics` exposes
  `make_scorer`/`get_scorer`/`get_scorer_names`/`check_scoring`;
  `ferrolearn-python` exposes no scorer shim.
- REQ-8: **ferray substrate (R-SUBSTRATE).** `scorer.rs` imports `ndarray::Array1`
  + `num_traits::Float`, not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (run from `/tmp`),
never literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-2 pin — THE HEADLINE): for a fitted estimator with raw
  `mean_squared_error == m`, `get_scorer("neg_mean_squared_error")(est, X, y)`
  must equal **`-m`** (sklearn applies `_sign`, `_scorer.py:376`). Live oracle on
  `LinearRegression().fit([[1],[2],[3],[4]],[2,4,6,7])`: raw mse ≈ **0.075**,
  scorer returns ≈ **-0.075**. ferrolearn's `Scorer::score` returns ≈ **+0.075**
  and FAILS. The signed-value contract is what model selection relies on.
- AC-2 (REQ-2, value form): `get_scorer("r2")` evaluated must equal `+r2`
  (sign `+1`) and `get_scorer("neg_mean_absolute_error")` must equal `-mae` —
  ferrolearn's `score()` returns `+r2` (coincidentally correct, sign `+1`) but
  `+mae` (wrong; should be `-mae`). The asymmetry is the tell that `sign()` is
  not applied.
- AC-3 (REQ-3): `make_scorer(r2_score)` must succeed with `greater_is_better`
  **defaulting to `True`** (sklearn `:659`); ferrolearn requires the argument
  and does not compile without it. `make_scorer(fbeta_score, beta=2)` must forward
  `beta` (sklearn `**kwargs`, `:732-733`); ferrolearn has no `kwargs`.
- AC-4 (REQ-4/5/6 — coverage): `len(get_scorer_names())` must equal **56**
  (live oracle); ferrolearn returns **11**. `"max_error" in get_scorer_names()`
  must be **True** (ferrolearn: False — it has `neg_max_error` instead);
  `"accuracy"`, `"f1_macro"`, `"adjusted_rand_score"` must each be present
  (ferrolearn: absent).
- AC-5 (REQ-6 — name canon): `get_scorer("neg_max_error")` must raise sklearn's
  `ValueError` ("not a valid scoring value") — **`neg_max_error` is not a sklearn
  name**; `get_scorer("max_error")._sign` must equal **`-1`**. ferrolearn accepts
  `neg_max_error` and rejects `max_error`.
- AC-6 (REQ-4/5 — full evaluation, post-type-fix): a classification scorer such
  as `get_scorer("accuracy")(clf, X, y)` and a clustering scorer such as
  `get_scorer("adjusted_rand_score")` must evaluate against the live sklearn value;
  ferrolearn's `Scorer<F>` type cannot wrap a `usize`-label metric and FAILS to
  express it.
- AC-7 (REQ-1 baseline — must stay green): the 11 shared names resolve to a
  `Scorer` with the correct `greater_is_better` flag (`neg_*` → false, `r2`/
  `explained_variance` → true) — this metadata is value-correct today and bounds
  the correctness that IS present. It does NOT make REQ-1 SHIPPED, because the
  flag is not applied by `score()` (REQ-2).

## REQ status table

Binary (R-DEFER-2). The scorer machinery is an existing pub API re-exported at the
crate root (the non-test production-consumer surface; grandfathered S5/R-DEFER-1).
Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle =
installed sklearn 1.5.2, run from `/tmp`. **No REQ is SHIPPED**: REQ-1 is gated on
the inverted sign-application contract (REQ-2), so even the 11-name subset is not a
faithful mirror. Honest underclaim (R-HONEST-3).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (11-name machinery) | NOT-STARTED | open prereq blocker #790 (depends on #791). `pub fn get_scorer` / `pub fn make_scorer` / `pub fn get_scorer_names` / `pub struct Scorer` resolve the 11 shared names with the correct `greater_is_better` *metadata* (AC-7), but the machinery is **not a faithful mirror** because `Scorer::score` returns the raw metric, not `sign * metric` (REQ-2). A `neg_*` scorer therefore yields the wrong scored value (`+loss` instead of `-loss`). SHIPPED-gated-on-REQ-2 → NOT-STARTED. |
| REQ-2 (**sign application**, headline) | NOT-STARTED | open prereq blocker #791. `pub fn Scorer::score in scorer.rs` returns raw `(self.score_fn)(y_true, y_pred)`; sklearn `_Scorer._score` returns `self._sign * self._score_func(...)` (`_scorer.py:376`). `pub fn Scorer::sign` is a separate accessor never applied inside `score()`. Live pin: `get_scorer("neg_mean_squared_error")(est,X,y)` → sklearn **-0.075**, ferrolearn `score()` → **+0.075**. Cleanest deterministic critic-pin this iteration. |
| REQ-3 (`make_scorer` ABI) | NOT-STARTED | open prereq blocker #792. `pub fn make_scorer in scorer.rs` is `make_scorer(score_fn, greater_is_better, name)` — `greater_is_better` **required positional, no default**; no `response_method`, no `**kwargs`. sklearn `make_scorer(score_func, *, response_method=None, greater_is_better=True, **kwargs)` (`_scorer.py:655-663`): keyword-only, `True` default, response_method, kwargs. Pin: `make_scorer(r2_score)` must compile with the default; ferrolearn requires the flag. |
| REQ-4 (classification scorers) | NOT-STARTED | open prereq blocker #793 (depends on #794). 36 sklearn classification scorer names unregistered (`accuracy`, `balanced_accuracy`, `matthews_corrcoef`, `neg_log_loss`, `neg_brier_score`, `top_k_accuracy`, `average_precision`, `roc_auc`/`roc_auc_ovo`/`roc_auc_ovr`/`*_weighted`, `positive_likelihood_ratio`/`neg_negative_likelihood_ratio`, and `f1`/`precision`/`recall`/`jaccard` ×{bare,macro,micro,samples,weighted}) (`_scorer.py:790-942`). The metric functions exist in `classification.rs`, but `Scorer<F>`'s `fn(&Array1<F>,&Array1<F>)` type cannot wrap `&Array1<usize>`/`&Array2<f64>` metrics, and there is no `average`/`pos_label`/`multi_class` parameterization. Blocked on #794. |
| REQ-5 (clustering scorers) | NOT-STARTED | open prereq blocker #795 (depends on #794). 9 clustering scorer names unregistered (`adjusted_rand_score`, `rand_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`, `mutual_info_score`, `adjusted_mutual_info_score`, `normalized_mutual_info_score`, `fowlkes_mallows_score`) (`_scorer.py:855-904`). Metric functions exist in `clustering.rs`; blocked on the same heterogeneous `Scorer` type (#794) + registration. |
| REQ-6 (regression name canon) | NOT-STARTED | open prereq blocker #796. `const BUILTIN_SCORER_NAMES in scorer.rs` registers `neg_max_error` — **not a sklearn name**; sklearn registers `max_error` (`_sign == -1`, `:761,870`) and `d2_absolute_error_score` (`_sign == +1`, `:788,881`), both unregistered in ferrolearn though the functions exist in `regression.rs`. `pub fn get_scorer_names` returns the list **unsorted**; sklearn returns `sorted(...)` (`:930`). Pin: `get_scorer("max_error")._sign == -1`; `get_scorer("neg_max_error")` → sklearn `ValueError`. |
| REQ-7 (PyO3 binding) | NOT-STARTED | open prereq blocker #797. `ferrolearn-python` exposes no scorer shim; `import ferrolearn` cannot call `make_scorer`/`get_scorer`/`get_scorer_names`/`check_scoring` that `import sklearn.metrics` provides. |
| REQ-8 (ferray substrate) | NOT-STARTED | open prereq blocker #798. `scorer.rs` imports `ndarray::Array1` + `num_traits::Float`, not `ferray-core` (R-SUBSTRATE). |

## Architecture

`scorer.rs` is a flat module: one generic struct `Scorer<F>`, free functions
`make_scorer`/`get_scorer`/`get_scorer_names`/`check_scoring`, a `const`
`BUILTIN_SCORER_NAMES`, and an enum `ScoringInput`. There are no fitted/unfitted
types — a `Scorer` is a stateless wrapper around a function pointer plus metadata.
The three structural divergences from sklearn:

1. **Sign is metadata, not behavior (REQ-2 — the headline).** In sklearn the sign
   is *part of the callable's evaluation* (`_Scorer._score` multiplies by
   `self._sign`, `_scorer.py:376`); a scorer is by construction "greater is
   better". In ferrolearn the sign lives only in the `greater_is_better` flag and
   the `sign()` accessor — `score()` returns the raw metric, pushing the
   sign-application burden onto every caller. A model-selection loop that
   maximises `scorer.score(...)` minimises the loss exactly backwards. This is the
   single most consequential, fully deterministic divergence (no estimator,
   `response_method`, or `kwargs` involved): a one-line semantic difference in
   `Scorer::score`.

2. **The function-pointer type is regression-shaped (REQ-4/5 — the big chunk).**
   `score_fn: fn(&Array1<F>, &Array1<F>) -> Result<F, FerroError>` can only wrap
   metrics whose `y_true`/`y_pred` are both `&Array1<F>`. sklearn's metrics are
   heterogeneous: classification takes integer labels (`&Array1<usize>`),
   probabilistic scorers take `predict_proba` output (`&Array2<f64>`),
   ranking/threshold scorers (`roc_auc`, `average_precision`, `top_k_accuracy`)
   take score arrays, and clustering scorers take two label arrays. sklearn
   absorbs this by closing over `score_func` + `response_method` + `kwargs` as
   opaque Python callables; ferrolearn's monomorphic `fn` pointer cannot. So all
   45 missing scorers are blocked NOT on the metric math (the functions exist in
   `classification.rs`/`clustering.rs`/`regression.rs`) but on a richer scorer
   abstraction (#794) — e.g. an enum of input-kinds or a boxed trait object — plus
   the `average`/`pos_label`/`multi_class` parameterization the `_SCORERS` loop
   (`:933-942`) and the `roc_auc_ov*` entries (`:824-841`) require.

3. **The registry is a 12-arm `match` over an unsorted 11-name list (REQ-6).**
   sklearn's `_SCORERS` is a 56-entry dict, `get_scorer_names()` returns
   `sorted(...)`. ferrolearn's `match` includes the non-sklearn `neg_max_error`
   and omits `max_error`/`d2_absolute_error_score` even though both functions are
   in `regression.rs`. `BUILTIN_SCORER_NAMES` is hand-maintained and unsorted,
   so it can (and does) drift from both the `match` arms and the sklearn canon.

**Invariants held vs sklearn:** the `greater_is_better`→sign metadata for the 11
shared names is correct (`neg_*` → `false`/`-1`, `r2`/`explained_variance` →
`true`/`+1`, AC-7); `get_scorer` raises on unknown names (ferrolearn
`InvalidParameter` vs sklearn `ValueError`); `make_scorer` round-trips the
function pointer + flag; `check_scoring` distinguishes a name from a pre-built
scorer (a subset of sklearn's string/callable branches). **Invariants NOT held:**
the sign is not applied in evaluation (REQ-2); `make_scorer` lacks the keyword-only
shape, `True` default, `response_method`, and `**kwargs` (REQ-3); 45 of 56 names
unregistered (REQ-4/5); `neg_max_error` is a non-sklearn name and `max_error`/
`d2_absolute_error_score` are missing, list unsorted (REQ-6); the scorer is not a
callable on `(estimator, X, y)` — it takes `(y_true, y_pred)` directly, so the
`response_method` / `_get_response_values` layer (`_scorer.py:84-96,369-373`) has
no analog; no multimetric `_MultimetricScorer`, no `_PassthroughScorer`.

## Verification

Library crate (green at baseline `db2cb592` for the existing — narrower —
contract):
```
cargo test -p ferrolearn-metrics --lib scorer    # 8 passed, 0 failed at baseline
cargo clippy -p ferrolearn-metrics --all-targets -- -D warnings
cargo fmt --all --check
```
The existing 8 `#[test]`s (`test_make_scorer_basic`, `test_scorer_evaluate`,
`test_scorer_sign_greater_is_better`, `test_scorer_sign_less_is_better`,
`test_scorer_debug`, `test_scorer_error_propagation`, `test_scorer_with_real_metric`,
`test_scorer_clone`) pin only ferrolearn's narrower behavior — notably
`test_scorer_evaluate`/`test_scorer_with_real_metric` assert `score()` returns the
**raw** (unsigned) metric, i.e. they pin the REQ-2 divergence as if it were
correct. They make no REQ SHIPPED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin first (R-CHAR-3 expected values):
```
# REQ-2 (HEADLINE — scorer applies sign): sklearn -mse vs ferro +mse
python3 -c "
import numpy as np
from sklearn.metrics import get_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression
X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,4.,6.,7.])
est=LinearRegression().fit(X,y); yhat=est.predict(X)
print('raw mse', mean_squared_error(y,yhat))                    # 0.07499...
print('scorer', get_scorer('neg_mean_squared_error')(est,X,y))  # -0.07499...  (ferro score() => +0.075)
print('sign  ', get_scorer('neg_mean_squared_error')._sign)     # -1
"
# REQ-3 (make_scorer greater_is_better default True): sklearn _sign == +1
python3 -c "from sklearn.metrics import make_scorer, r2_score; print(make_scorer(r2_score)._sign)"  # 1
# REQ-4/5/6 (coverage): sklearn 56 names, ferro 11
python3 -c "from sklearn.metrics import get_scorer_names as g; n=g(); print(len(n), 'max_error' in n, 'neg_max_error' in n, 'accuracy' in n, 'f1_macro' in n, 'adjusted_rand_score' in n)"  # 56 True False True True True
# REQ-6 (neg_max_error is NOT a sklearn name -> ValueError; max_error sign -1):
python3 -c "from sklearn.metrics import get_scorer; print(get_scorer('max_error')._sign)"  # -1
python3 -c "from sklearn.metrics import get_scorer; get_scorer('neg_max_error')"           # ValueError
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-metrics/tests/divergence_scorer.rs`, asserting the live-sklearn
expected values above and FAILING against current `scorer.rs`. The REQ-2 pin (a
`Scorer::score` on a `neg_*` name returning the *signed* value) is the cleanest:
it is fully deterministic, needs no estimator/`response_method`, and exposes the
single-line semantic gap. Every REQ is NOT-STARTED; each carries an open prereq
blocker.

## Blockers to open

- #790 — REQ-1 (11-name machinery): a faithful mirror is gated on the
  sign-application fix (#791); the 11-name subset resolves correct metadata but
  `score()` does not apply it.
- #791 — REQ-2 (**headline**): `Scorer::score` returns raw metric, not
  `sign * metric` (`_scorer.py:376`). Pin: `get_scorer("neg_mean_squared_error")`
  evaluated → sklearn -mse, ferro +mse.
- #792 — REQ-3: `make_scorer` ABI — positional `greater_is_better` (no `True`
  default), no `response_method`, no `**kwargs` (`_scorer.py:655-663`).
- #793 — REQ-4 (classification scorers): 36 names unregistered (`accuracy`,
  `balanced_accuracy`, `matthews_corrcoef`, `neg_log_loss`, `neg_brier_score`,
  `top_k_accuracy`, `average_precision`, `roc_auc` family, likelihood ratios,
  `f1`/`precision`/`recall`/`jaccard` ×5) (`_scorer.py:790-942`); metric functions
  exist in `classification.rs`. Blocked on #794.
- #794 — REQ-4/5 prereq: the `Scorer<F>` `fn(&Array1<F>,&Array1<F>)` type cannot
  wrap heterogeneous metrics (`&Array1<usize>` labels, `&Array2<f64>` proba, score
  arrays, clustering label pairs) and has no `average`/`pos_label`/`multi_class`
  parameterization — needs a richer scorer abstraction before any classification/
  clustering scorer can be registered.
- #795 — REQ-5 (clustering scorers): 9 names unregistered (`adjusted_rand_score`,
  `rand_score`, `homogeneity_score`, `completeness_score`, `v_measure_score`,
  `mutual_info_score`, `adjusted_mutual_info_score`, `normalized_mutual_info_score`,
  `fowlkes_mallows_score`) (`_scorer.py:855-904`); functions exist in
  `clustering.rs`. Blocked on #794.
- #796 — REQ-6: `neg_max_error` is a non-sklearn name; register `max_error`
  (`_sign -1`, `:761,870`) + `d2_absolute_error_score` (`_sign +1`, `:788,881`);
  `get_scorer_names()` must be sorted (`:930`).
- #797 — REQ-7: no `ferrolearn-python` scorer-machinery binding.
- #798 — REQ-8: migrate `scorer.rs` off `ndarray`/`num-traits` to the ferray
  substrate (R-SUBSTRATE).
