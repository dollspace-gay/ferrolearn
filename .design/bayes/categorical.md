# Categorical Naive Bayes (sklearn.naive_bayes.CategoricalNB)

<!--
tier: 3-component
status: draft
baseline-commit: e170c3c5
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # CategoricalNB(_BaseDiscreteNB) :1222-1516; __init__(*, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, min_categories=None) (:1336-1351); _parameter_constraints alpha=Interval(Real,0,None,closed="left") (:1333 — alpha=0 ALLOWED, only alpha<0 rejected) + min_categories=[None,"array-like",Interval(Integral,1,None,closed="left")] (:1328-1332); _check_X/_check_X_y dtype="int" + check_non_negative "CategoricalNB (input X)" (:1427-1440); _validate_n_categories n_categories_ = max(X.max(0)+1, min_categories) (:1446-1466); _count category_count_[i] shape (n_classes, n_categories_i), np.pad to n_categories_[i] (:1468-1496); _update_feature_log_prob smoothed_cat_count=category_count_[i]+alpha, log(smoothed_cat_count)-log(smoothed_cat_count.sum(axis=1).reshape(-1,1)) (:1498-1506); _joint_log_likelihood jll += feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_ (:1508-1515); shared _update_class_log_prior LENGTH-only (:580-602); _check_alpha floor 1e-10 unless force_alpha (:604-626); shared fit/partial_fit (:628-762)
ferrolearn-module: ferrolearn-bayes/src/categorical.rs
parity-ops: CategoricalNB (.__init__, .fit, .partial_fit, .predict, .predict_proba, .predict_log_proba, .predict_joint_log_proba, .score)
crosslink-issue: 919
-->

## Summary

`ferrolearn-bayes/src/categorical.rs` mirrors scikit-learn's `CategoricalNB`
(`sklearn/naive_bayes.py`, `class CategoricalNB(_BaseDiscreteNB)` `:1222-1516`) —
the naive-Bayes classifier for discrete categorical features (each column drawn
from its own categorical distribution, values encoded as non-negative integers
`0..K_j-1`) whose per-(feature, class, category) log-likelihood is the smoothed
log-probability `log P(x_j = k | c) = log((N_cjk + alpha) / (N_c + alpha * K_j))`.
It exposes the unfitted `CategoricalNB<F>` (`alpha=1.0`, `class_prior:
Option<Vec<F>>`, `fit_prior=true`, `force_alpha=true`, `min_categories:
Option<MinCategories{Scalar|PerFeature}>`), the fitted `FittedCategoricalNB<F>`
(per-feature `categories` / `category_counts` / `feature_log_prob` plus
`class_log_prior` / `class_counts` and `alpha`/`class_prior`/`fit_prior` carried
for `partial_fit`), and delegates the entire prediction pipeline to the shared
`BaseNB<F>` trait (`base.rs`, the `_BaseNB` analog — see `.design/bayes/base.md`).

Under honest underclaim (R-HONEST-3), the behaviors that are genuinely present
**and value-match the live sklearn 1.5.2 oracle** (verified on the categorical
fixture `X = [[0,0,1],[1,0,0],[0,1,0],[0,0,0],[2,2,1],[2,1,2],[1,2,2],[2,2,2]]`,
`y = [0,0,0,0,1,1,1,1]`, query `q = [[0,0,0],[2,2,2]]`, run from `/tmp`) are:

- **`feature_log_prob_` smoothing VALUE** — sklearn's `_update_feature_log_prob`
  computes `smoothed_cat_count = category_count_[i] + alpha`; `smoothed_class_count
  = smoothed_cat_count.sum(axis=1)`; `feature_log_prob_[i] = log(smoothed_cat_count)
  - log(smoothed_class_count.reshape(-1,1))` (`:1498-1506`). ferrolearn's
  `recompute_feature_log_prob` computes `((N_cjk + alpha) / (N_c + alpha*K_j)).ln()`
  per (feature, class, category) — the algebraic identity (`smoothed_cat_count.sum(
  axis=1) == N_c + alpha*K_j` because `category_count_[i].sum(axis=1) == N_c`).
  ferrolearn has NO public `feature_log_prob_` accessor, so the VALUE is verified
  indirectly through `predict_joint_log_proba` / `predict_proba`.
- **`_joint_log_likelihood` + `predict` / `predict_proba` / `predict_log_proba` /
  `predict_joint_log_proba` VALUE** — sklearn's `jll += feature_log_prob_[i][:,
  X[:,i]].T` then `jll += class_log_prior_` (`:1508-1515`) feeding the `_BaseNB`
  pipeline. ferrolearn's `fn joint_log_likelihood` computes `class_log_prior[ci] +
  sum_j log_prob_for(j, ci, x[i,j])`, and the delegated `predict_*` match the
  oracle to ~1e-12 (oracle `predict_joint_log_proba(q) = [[-2.3719945443662134,
  -6.530877627725885], [-6.530877627725885, -2.3719945443662134]]`,
  `predict_proba(q) = [[0.9846153846153846, 0.015384615384615389],
  [0.015384615384615389, 0.9846153846153846]]`, `predict(q) = [0,1]`; ferrolearn
  reproduces `jll = [[-2.3719945443662134, -6.530877627725886], ...]`, `pp =
  [[0.9846153846153846, 0.015384615384615375], ...]`, `pred = [0,1]`).
- **`min_categories` n_categories_ semantics VALUE** — sklearn's
  `_validate_n_categories` sets `n_categories_[i] = max(X[:,i].max()+1,
  min_categories[i])` (`:1446-1466`) and `_count` pads `category_count_[i]` to that
  width (`:1491-1493`), so unobserved-but-allocated categories get the smoothed
  weight `alpha / (N_c + alpha*K_j)`. ferrolearn's `fn fit` ensures `categories[j]`
  covers `0..min_cats[j]` (`MinCategories::Scalar`/`PerFeature`) so the count tables
  have those slots. Oracle `CategoricalNB(min_categories=4).fit(X4,y4).
  n_categories_ = [4,4]`, `feature_log_prob_[0] = [[-0.6931..,-1.7918..,-1.7918..,
  -1.7918..],[...]]`, `predict_joint_log_proba([[3,0]]) = [[-3.58351893845611,
  -3.58351893845611]]`, `predict_proba = [[0.5,0.5]]`, `predict = [0]`; ferrolearn
  `with_min_categories(4)` reproduces the IDENTICAL `jll = [[-3.58351893845611,
  -3.58351893845611]]`, `pp = [[0.5,0.5]]` for the allocated-but-unobserved
  category 3.
- **`class_prior` LENGTH-ONLY validation** — sklearn's shared
  `_update_class_log_prior` checks ONLY length (`:589-590`), with NO sum-to-1 and
  NO non-negativity check for discrete NB. ferrolearn's `resolve_class_log_prior`
  checks ONLY length — a **MATCH** (`CategoricalNB(class_prior=[0.5,0.3])`, sum 0.8,
  fits on both sides → oracle `class_log_prior_ = [-0.6931471805599453,
  -1.2039728043259361]` = `log([0.5,0.3])`).
- **`class_log_prior_` empirical / uniform paths VALUE** — `log(class_count_) -
  log(class_count_.sum())` (fit_prior, `:600`) and `-log(n_classes)`
  (fit_prior=False, `:602`); ferrolearn's `resolve_class_log_prior` sets
  `ln(count_c / total)` (empirical) / `(1/n_classes).ln()` (uniform) — value-matches
  (oracle empirical `class_log_prior_ = [-0.6931471805599452, -0.6931471805599452]`
  on the balanced fixture).
- **`force_alpha` / `_check_alpha` floor + `fit_prior` toggle** — the
  `base::check_alpha` / `clamp_alpha` floor (`1e-10` unless `force_alpha`) and the
  empirical-vs-uniform prior selection.
- **`partial_fit` VALUE on existing classes/categories** — increments the per-(
  feature, class, category) counts then recomputes `feature_log_prob` / class
  priors (the same smoothing); ferrolearn additionally EXTENDS `categories[j]` and
  `classes` for never-seen values — a non-sklearn flexibility (see divergence #4
  below).
- **`score`** — mean accuracy (`ClassifierMixin.score` analog).

The behaviors that **diverge** from the `CategoricalNB` contract (each pinned to a
NOT-STARTED REQ with a concrete prereq blocker — the director creates the real
issues; the numbers below are SUGGESTIONS continuing the bayes layer past
complement #914-917):

1. **`alpha = 0` over-rejection + `alpha < 0` reject (R-DEV-2 — THE key fixable
   divergence).** sklearn's `_parameter_constraints` declares `alpha:
   [Interval(Real, 0, None, closed="left")]` (`:1333`) → `alpha >= 0` is allowed,
   so **`alpha = 0` is ACCEPTED** (`CategoricalNB(alpha=0.0).fit(X,y)` succeeds,
   producing `-inf` log-probs where a count is zero, with a `divide by zero
   encountered in log` RuntimeWarning — NOT an error); only `alpha < 0` is a HARD
   reject (`CategoricalNB(alpha=-1.0).fit(X,y)` → `InvalidParameterError("The
   'alpha' parameter of CategoricalNB must be a float in the range [0.0, inf). Got
   -1.0 instead.")`). ferrolearn `fn fit` rejects `self.alpha <= F::zero() &&
   self.force_alpha` (`categorical.rs` line 238) — so with the default
   `force_alpha=true` it **OVER-REJECTS `alpha = 0`** (returns
   `InvalidParameter`), diverging from sklearn which accepts it. The reject for
   `alpha < 0` matches sklearn's decision (both reject; the type/message differ).
   **The single-file-fixable divergence in `categorical.rs` `fn fit`** — change the
   `alpha <= 0` guard to `alpha < 0` (reject only negatives, allow zero), same
   class as multinomial #900 / complement #914 except CategoricalNB's bug is
   over-rejection of zero rather than under-rejection of negatives. The critic
   should pin this FIRST.
2. **negative / non-integer feature validation (R-DEV-2).** sklearn's `_check_X` /
   `_check_X_y` validate `X` via `_validate_data(X, dtype="int", ...)` then
   `check_non_negative(X, "CategoricalNB (input X)")` (`:1427-1440`): a negative
   value raises `ValueError("Negative values in data passed to CategoricalNB (input
   X)")`; a non-integer float is TRUNCATED toward zero by the `dtype="int"` cast
   (e.g. `0.9 → 0`, observed: `n_categories_=[2,2]`, `category_count_[0]=[[2,0],
   [0,2]]`). ferrolearn has NO validation: `fn fit` maps each value via
   `x[[i,j]].to_usize().unwrap_or(0)` — a negative value silently becomes category
   `0` (no error; `fit(X_with_-1, y)` is `is_ok=true`), and a non-negative float is
   truncated toward zero by `to_usize` (so `0.9 → 0` MATCHES sklearn for
   non-negatives). DIVERGENCE only on negatives: sklearn rejects, ferrolearn maps
   to 0.
3. **unseen-category at predict (R-DEV-3).** sklearn requires category indices `<
   n_categories_[i]` (set at fit, possibly enlarged by `min_categories`); the
   `_joint_log_likelihood` `feature_log_prob_[i][:, X[:,i]]` fancy-index raises
   `IndexError("index 5 is out of bounds for axis 1 with size 2")` for an index
   beyond `n_categories_` (`:1513`). ferrolearn's `log_prob_for` returns a uniform
   `(1/(n_known_cats+1)).ln()` fallback for any category value not in
   `categories[j]` — NO error (`predict([[5,0]])` → `[0]`, `predict_proba` →
   `[[0.5,0.5]]`). DIVERGENCE (sklearn errors / requires the category to be sized in
   via `min_categories`; ferrolearn silently degrades to a uniform fallback). NOTE:
   when the category IS sized in via `min_categories` (e.g. category 3 under
   `min_categories=4`), both sides agree (REQ-3, the allocated-slot path); the
   divergence is the genuinely-unsized index (`>= n_categories_`).
4. **`partial_fit` new-category / new-class EXTENSION (R-DEV-7 deviation — NOT a
   bug).** sklearn's shared `partial_fit(X, y, classes=None, ...)` (`:628-709`)
   takes the full `classes` list on the first call and binarizes against it, and
   `_count` pads `category_count_` only up to `n_categories_` (fixed by the first
   fit / `min_categories`); a category index `>= n_categories_` in a later chunk
   would `IndexError`. ferrolearn's `FittedCategoricalNB::partial_fit` (no `classes`
   argument) APPENDS never-seen category values to `categories[j]` (allocating a
   count slot for every class) AND inserts never-seen class labels into `classes`
   — a deliberate non-sklearn flexibility documented in the method's doc-comment
   ("Unlike sklearn's strict integer-index model …"). On data that stays within the
   fitted categories/classes the recompute-after-accumulate VALUE matches; the
   extension behavior is an observable deviation from sklearn's fixed-`n_categories_`
   contract.
5. **`sample_weight` (R-DEV-1).** sklearn `fit(X, y, sample_weight=None)` (`:1353`,
   `:712`) weights the binarized `Y` so `class_count_ = Y.sum(axis=0)` and the
   `np.bincount(X_feature[mask], weights=...)` per-category counts become weighted
   (`:1468-1496`). ferrolearn's `Fit` trait is `fn fit(&self, x, y)` — NO
   `sample_weight` parameter on `fit` or `partial_fit`.
6. **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn exposes
   `category_count_` (list of `(n_classes, n_categories_i)` arrays),
   `feature_log_prob_` (same shape), `class_count_`, `class_log_prior_`,
   `n_categories_`, `classes_`, `n_features_in_` (`:1266-1303`). ferrolearn
   `FittedCategoricalNB` exposes ONLY `classes()` (via `HasClasses`);
   `feature_log_prob` / `category_counts` / `categories` / `class_log_prior` /
   `class_counts` are PRIVATE fields with no accessor. **CategoricalNB has NO PyO3
   binding at all** — unlike its siblings (`_RsMultinomialNB` / `_RsBernoulliNB` /
   `_RsComplementNB` exist in `ferrolearn-python/src/extras.rs`), there is NO
   `_RsCategoricalNB`, and `ferrolearn.CategoricalNB` does not exist in
   `ferrolearn-python/python/ferrolearn/`. So `import ferrolearn` cannot reach
   CategoricalNB at all.
7. **ferray substrate (R-SUBSTRATE).** `categorical.rs` imports `ndarray::{Array1,
   Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`, not `ferray-core`.

`CategoricalNB` / `FittedCategoricalNB` are existing pub APIs (grandfathered per
S5/R-DEFER-1) re-exported at the crate root (`ferrolearn-bayes/src/lib.rs`, `pub
use categorical::{CategoricalNB, FittedCategoricalNB}`). Their non-test production
consumer is the in-crate pipeline integration (`impl PipelineEstimator for
CategoricalNB` — the same `Box<dyn FittedPipelineEstimator>`-producing surface
that `pipeline.rs` cites for GaussianNB/BernoulliNB). Unlike the other discrete NB
variants there is NO `ferrolearn-python` binding (REQ-6/#923).

## Algorithm (sklearn — the contract)

### Construction (`naive_bayes.py:1336-1351`)

`CategoricalNB(*, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
min_categories=None)` — all keyword-only. `_parameter_constraints` (`:1326-1334`,
merging `_BaseDiscreteNB`'s `:529-534` then OVERRIDING `alpha`): `alpha:
[Interval(Real, 0, None, closed="left")]` (**>= 0**, so `alpha = 0` is ALLOWED,
`:1333`); `min_categories: [None, "array-like", Interval(Integral, 1, None,
closed="left")]` (`:1328-1332`); `fit_prior: ["boolean"]`; `class_prior:
["array-like", None]`; `force_alpha: ["boolean"]`. `_more_tags` declares
`requires_positive_X: True` (`:1424-1425`).

### Fit (`_BaseDiscreteNB.fit` `:712-762` → `CategoricalNB._count` `:1468-1496`)

`fit(X, y, sample_weight=None)`: `_check_X_y` validates `X` via `_validate_data(X,
y, dtype="int", ...)` + `check_non_negative(X, "CategoricalNB (input X)")`
(`:1435-1440`); binarize `y` → one-hot `Y`; optionally `Y *= sample_weight.T`;
`_init_counters` (`:1442-1444`) sets `class_count_ = zeros(n_classes)` and
`category_count_ = [zeros((n_classes, 0)) for _ in range(n_features)]`; `_count(X,
Y)` accumulates; `alpha = self._check_alpha()`; `_update_feature_log_prob(alpha)`;
`_update_class_log_prior(class_prior)`.

- **`_validate_n_categories`** (`:1446-1466`): `n_categories_X = X.max(axis=0) +
  1`; if `min_categories is not None` → `n_categories_ = np.maximum(n_categories_X,
  min_categories_, dtype=np.int64)` (must be integral type, matching shape); else
  `n_categories_X`.
- **`_count`** (`:1468-1496`): `class_count_ += Y.sum(axis=0)`; `n_categories_ =
  _validate_n_categories(X, min_categories)`; for each feature `i`, pad
  `category_count_[i]` to width `n_categories_[i]` (`_update_cat_count_dims`,
  `:1469-1474`), then `np.bincount(X[:,i][mask], weights=...)` per class accumulates
  the per-category counts (`:1476-1485`).
- **`_check_alpha`** (`:604-626`): `alpha_lower_bound = 1e-10`; if `alpha_min <
  alpha_lower_bound and not self.force_alpha` warn + return `np.maximum(alpha,
  alpha_lower_bound)`; else return alpha unchanged. (The `>= 0` HARD constraint is
  enforced earlier by `_validate_params` against `_parameter_constraints` `:1333`,
  NOT inside `_check_alpha` — and `alpha = 0` is INSIDE the allowed interval.)
- **`_update_feature_log_prob`** (`:1498-1506`): for each feature `i`,
  `smoothed_cat_count = category_count_[i] + alpha`; `smoothed_class_count =
  smoothed_cat_count.sum(axis=1)`; `feature_log_prob_[i] = np.log(smoothed_cat_count)
  - np.log(smoothed_class_count.reshape(-1, 1))`. Algebraically `log((N_cjk + alpha)
  / (N_c + alpha * K_j))` (since `smoothed_cat_count.sum(axis=1) = N_c + alpha*K_j`).
- **`_update_class_log_prior`** (`:580-602`): if `class_prior is not None` → `if
  len(class_prior) != n_classes: raise ValueError("Number of priors must match
  number of classes.")` (`:589-590`); `class_log_prior_ = np.log(class_prior)`
  (`:591`) — **LENGTH-ONLY check, NO sum-to-1, NO non-negativity**. elif `fit_prior`
  → `log(class_count_) - log(class_count_.sum())` (`:600`). else → `np.full(
  n_classes, -np.log(n_classes))` (uniform, `:602`).

### `_joint_log_likelihood` (`:1508-1515`)

`jll = zeros((n_samples, n_classes))`; for each feature `i`: `jll +=
feature_log_prob_[i][:, X[:,i]].T` (fancy-index — selects the log-prob column for
each sample's category, **raises `IndexError` if any category index `>=
n_categories_[i]`**); then `total_ll = jll + class_log_prior_` (`:1514`). Shape
`(n_samples, n_classes)`. The shared `_BaseNB` pipeline (`predict` /
`predict_proba` / `predict_log_proba` / `predict_joint_log_proba`) consumes this
(`.design/bayes/base.md`).

### `partial_fit` (`_BaseDiscreteNB.partial_fit` `:628-709`)

`partial_fit(X, y, classes=None, sample_weight=None)`. First call requires
`classes` and initializes the counters; each call binarizes `y` against the FULL
`classes` list, `_count(X, Y)` accumulates (re-validating `n_categories_` and
padding `category_count_`), then recomputes `alpha = _check_alpha()`,
`_update_feature_log_prob`, `_update_class_log_prior`. `n_categories_` is fixed by
the first fit (and `min_categories`); a category `>= n_categories_` in a later
chunk would `IndexError` in `_count`'s `np.bincount` indexing path.

### Edge cases (live oracle, sklearn 1.5.2, run from /tmp)

- **feature_log_prob_ VALUE** (fixture `X = [[0,0,1],[1,0,0],[0,1,0],[0,0,0],
  [2,2,1],[2,1,2],[1,2,2],[2,2,2]]`, `y = [0,0,0,0,1,1,1,1]`, `alpha=1`):
  `n_categories_ = [3,3,3]`; `category_count_[0] = [[3,1,0],[0,1,3]]`;
  `feature_log_prob_[0] = [[-0.5596157879354227, -1.252762968495368,
  -1.9459101490553132], [-1.9459101490553132, -1.252762968495368,
  -0.5596157879354227]]` (= `log((category_count_[0]+1)/((category_count_[0]+1).
  sum(axis=1, keepdims=True)))`); `class_log_prior_ = [-0.6931471805599452,
  -0.6931471805599452]`.
- **predict VALUE** (`q = [[0,0,0],[2,2,2]]`): `predict_joint_log_proba(q) =
  [[-2.3719945443662134, -6.530877627725885], [-6.530877627725885,
  -2.3719945443662134]]`; `predict_log_proba(q) = [[-0.015504186535965303,
  -4.174387269895637], [-4.174387269895637, -0.015504186535965303]]`;
  `predict_proba(q) = [[0.9846153846153846, 0.015384615384615389],
  [0.015384615384615389, 0.9846153846153846]]`; `predict(q) = [0,1]`.
- **alpha = 0 (force_alpha=True default)**: `CategoricalNB(alpha=0.0).fit(X,y)` is
  **ACCEPTED** (`0` is in `[0, inf)`); produces `feature_log_prob_[0] =
  [[-0.2876820724517808, -1.3862943611198906, -inf], [-inf, -1.3862943611198906,
  -0.2876820724517808]]` (the `-inf` where a count is zero) with a `divide by zero
  encountered in log` RuntimeWarning — NOT an error. ferrolearn `with_alpha(0.0).
  fit(&X,&y)` → **rejects** (`is_ok=false`, `InvalidParameter { name: "alpha" }`).
- **alpha = -1**: `CategoricalNB(alpha=-1.0).fit(X,y)` → `InvalidParameterError("The
  'alpha' parameter of CategoricalNB must be a float in the range [0.0, inf). Got
  -1.0 instead.")` (raised at `fit` by `_validate_params`). ferrolearn rejects too
  (`InvalidParameter`) — both reject negatives.
- **negative features**: `CategoricalNB().fit([[-1,0],[0,1],[1,0],[1,1]], [0,0,1,
  1])` → `ValueError("Negative values in data passed to CategoricalNB (input X)")`.
  ferrolearn `fn fit` is `is_ok=true` (maps `-1 → 0` via `to_usize().unwrap_or(0)`).
- **non-integer features**: `CategoricalNB().fit([[0.9,0.0],...], y)` SUCCEEDS —
  `0.9` is TRUNCATED toward zero to category `0` by the `dtype="int"` cast
  (observed `n_categories_=[2,2]`, `category_count_[0]=[[2,0],[0,2]]`). ferrolearn's
  `to_usize` also truncates toward zero, so non-negative floats MATCH.
- **unseen category at predict**: fit on `[[0,0],[0,1],[1,0],[1,1]]`, `[0,0,1,1]`
  (`n_categories_=[2,2]`), then `predict([[5,0]])` → `IndexError("index 5 is out of
  bounds for axis 1 with size 2")`. ferrolearn `predict([[5,0]])` → `[0]`,
  `predict_proba` → `[[0.49999999999999994, 0.49999999999999994]]` (uniform
  `1/(n_cats+1)` fallback, no error).
- **min_categories**: `CategoricalNB(min_categories=4).fit([[0,0],[0,1],[1,0],
  [1,1]], [0,0,1,1])` → `n_categories_ = [4,4]`; `feature_log_prob_[0] =
  [[-0.6931471805599452, -1.791759469228055, -1.791759469228055,
  -1.791759469228055], [-1.791759469228055, -0.6931471805599452,
  -1.791759469228055, -1.791759469228055]]` (the allocated-but-unobserved
  categories 2,3 get the smoothed `alpha/(N_c+alpha*K) = 1/6` → `log(1/6) =
  -1.791759…`); `predict_joint_log_proba([[3,0]]) = [[-3.58351893845611,
  -3.58351893845611]]`; `predict_proba = [[0.5,0.5]]`; `predict = [0]`. ferrolearn
  `with_min_categories(4)` reproduces the IDENTICAL `jll = [[-3.58351893845611,
  -3.58351893845611]]`, `pp = [[0.5,0.5]]`. (A category `>= n_categories_`, e.g.
  `predict([[4,0]])`, still `IndexError`s in sklearn — `index 4 is out of bounds
  for axis 1 with size 4` — while ferrolearn's uniform fallback handles it; this is
  the REQ-4 unseen-category divergence, distinct from the allocated-slot match.)
- **class_prior length-only**: `CategoricalNB(class_prior=[0.5,0.3]).fit(X,y)` (sum
  0.8) ACCEPTED → `class_log_prior_ = [-0.6931471805599453, -1.2039728043259361]`
  (= `log([0.5,0.3])`), NO sum/non-neg error. Wrong length `class_prior=[0.5]` →
  `ValueError("Number of priors must match number of classes.")`. ferrolearn
  MATCHES all three (length-only via `resolve_class_log_prior`).

## ferrolearn (what exists)

All in `ferrolearn-bayes/src/categorical.rs`, generic over `F: Float + Send + Sync
+ 'static`; `ndarray` substrate. Every public method returns `Result<_,
FerroError>` (no panics in library code, R-CODE-2).

- **`pub enum MinCategories { Scalar(usize), PerFeature(Vec<usize>) }`** — the
  `min_categories` analog (scalar broadcast or explicit per-feature).
- **`pub struct CategoricalNB<F> { alpha, class_prior: Option<Vec<F>>, fit_prior,
  force_alpha, min_categories: Option<MinCategories> }`** — `pub fn new` sets `alpha
  = 1.0`, `class_prior = None`, `fit_prior = true`, `force_alpha = true`,
  `min_categories = None` (matching sklearn defaults, `:1336-1351`); builder setters
  `with_alpha` / `with_class_prior` / `with_fit_prior` / `with_force_alpha` /
  `with_min_categories` / `with_min_categories_per_feature`; `impl Default → new()`.
- **`pub struct FittedCategoricalNB<F>`** — private fields `classes: Vec<usize>`,
  `class_log_prior: Vec<F>` (the `class_log_prior_` analog), `feature_log_prob:
  Vec<Vec<Vec<F>>>` (the `feature_log_prob_` analog, indexed `[feature][class]
  [category]`), `category_counts: Vec<Vec<Vec<usize>>>` (the `category_count_`
  analog), `categories: Vec<Vec<usize>>` (per-feature sorted known category values,
  including `min_categories` padding), `class_counts: Vec<usize>` (the `class_count_`
  analog), `n_features`, plus `alpha` / `class_prior` / `fit_prior` carried for
  `partial_fit`. **No public accessor** for any of these (only `classes()` via
  `HasClasses`).
- **`impl Fit<Array2<F>, Array1<usize>> for CategoricalNB<F>` / `fn fit`** — rejects
  `n_samples == 0` (`InsufficientSamples`), `n_samples != y.len()` (`ShapeMismatch`),
  and `self.alpha <= F::zero() && self.force_alpha` (`InvalidParameter`, line 238 —
  **OVER-REJECTS `alpha = 0`**, REQ-1). Validates `MinCategories::PerFeature` length
  against `n_features`. `alpha = crate::clamp_alpha(self.alpha, self.force_alpha)`
  (the `_check_alpha` floor — REQ-5). Collects sorted-deduped `classes`. For each
  feature: discovers observed unique category values (`x[[i,j]].to_usize().
  unwrap_or(0)` — **no negative/non-integer guard**, REQ-2), then ensures
  `categories[j]` covers `0..min_cats[j]` (the `min_categories` padding — REQ-3);
  builds the per-(class, category) `category_counts`. `recompute_feature_log_prob`
  computes `((N_cjk + alpha) / (N_c + alpha*K_j)).ln()` (REQ-1).
  `resolve_class_log_prior` resolves explicit (LENGTH-only — REQ-7) / empirical /
  uniform priors (REQ-7). **No `alpha < 0` vs `= 0` distinction** (REQ-1), **no
  `sample_weight`** (REQ-5).
- **`fn recompute_feature_log_prob`** — for each (feature, class, category) sets
  `log_prob = ((count + alpha) / (N_c + alpha*K_j)).ln()`, mirroring
  `_update_feature_log_prob` (`:1498-1506`, REQ-1).
- **`fn resolve_class_log_prior`** — explicit `class_prior` (LENGTH-only check then
  `p.ln()` — REQ-7) / `fit_prior` empirical `ln(count_c / total)` / uniform
  `(1/n_classes).ln()`, mirroring `_update_class_log_prior` (`:580-602`).
- **`FittedCategoricalNB::partial_fit(&mut self, x, y)`** — increments the per-(
  feature, class, category) `category_counts`, then recomputes `feature_log_prob` /
  `class_log_prior` (REQ-8 VALUE on existing categories/classes). **EXTENDS**:
  inserts never-seen class labels into `classes` (allocating count slots) and
  appends never-seen category values to `categories[j]` — a non-sklearn flexibility
  (REQ-9 deviation). Rejects feature-count mismatch (`ShapeMismatch`). No `classes`
  argument, no `sample_weight` (REQ-5).
- **`fn log_prob_for(feature_idx, class_idx, cat_value)`** — looks up
  `feature_log_prob[feature_idx][class_idx][cat_idx]` if `cat_value` is in
  `categories[feature_idx]`, ELSE returns the uniform `(1/(n_known_cats+1)).ln()`
  fallback (REQ-4 divergence — sklearn `IndexError`s instead).
- **`impl BaseNB<F> for FittedCategoricalNB<F>` / `fn joint_log_likelihood`** —
  `score[i,ci] = class_log_prior[ci] + sum_j log_prob_for(j, ci, x[i,j])`, shape
  `(n_samples, n_classes)`, mirroring `jll += feature_log_prob_[i][:, X[:,i]].T; jll
  += class_log_prior_` (`:1508-1515`, REQ-1). `fn nb_classes` returns
  `&self.classes`.
- **`pub fn predict_proba` / `pub fn predict_log_proba` / `pub fn
  predict_joint_log_proba`** — delegate to `BaseNB::nb_predict_proba` /
  `nb_predict_log_proba` / `nb_predict_joint_log_proba` (REQ-1 pipeline).
- **`impl Predict for FittedCategoricalNB<F>` / `fn predict`** — delegates to
  `BaseNB::nb_predict` (`classes_[argmax(jll)]`, first-max tie-break).
- **`pub fn score(&self, x, y)`** — mean accuracy (`correct / n`), the
  `ClassifierMixin.score` analog.
- **`impl HasClasses for FittedCategoricalNB<F>`** — `classes()` / `n_classes()`.
- **Pipeline**: `impl PipelineEstimator<F> for CategoricalNB<F>` (`fn fit_pipeline`,
  maps float labels → `usize`) + `FittedCategoricalNBPipeline` (`fn
  predict_pipeline`) — the non-test production consumer.

**Consumers (non-test).** Crate re-export (`ferrolearn-bayes/src/lib.rs`, `pub use
categorical::{CategoricalNB, FittedCategoricalNB}`) plus the in-crate **pipeline
integration** — `impl PipelineEstimator<F> for CategoricalNB<F>` produces a `Box<dyn
FittedPipelineEstimator<F>>` (`FittedCategoricalNBPipeline`) that consumes `fit` /
`predict`, the same production-consumer surface `pipeline.rs` cites for
GaussianNB/BernoulliNB. **There is NO `ferrolearn-python` binding** for CategoricalNB
(no `_RsCategoricalNB` in `extras.rs`, no `ferrolearn.CategoricalNB` in
`_extras.py` — unlike its discrete-NB siblings), so `import ferrolearn` cannot
reach it (REQ-6/#923).

## Requirements

- REQ-1: **`feature_log_prob_` smoothing VALUE + `_joint_log_likelihood` /
  `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
  VALUE (R-DEV-1/3).** Mirror `_update_feature_log_prob` (`:1498-1506`,
  `log(category_count_[i]+alpha) - log((category_count_[i]+alpha).sum(axis=1).
  reshape(-1,1))`) and `_joint_log_likelihood` (`:1508-1515`, `jll +=
  feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_`) feeding the `_BaseNB`
  pipeline. ferrolearn `recompute_feature_log_prob` computes the algebraically
  identical `((N_cjk + alpha) / (N_c + alpha*K_j)).ln()`, `fn joint_log_likelihood`
  computes `class_log_prior[ci] + sum_j log_prob_for(j, ci, x[i,j])`, and the
  delegated `predict_*` value-match the oracle to ~1e-12. (No public
  `feature_log_prob_` accessor — verified indirectly via `predict_joint_log_proba` /
  `predict_proba`.)
- REQ-2: **`class_log_prior_` empirical / uniform + LENGTH-only `class_prior` VALUE
  (R-DEV-1/2 — MATCH).** Mirror `_update_class_log_prior` (`:580-602`): explicit
  branch LENGTH-only (`if len != n_classes: ValueError; log(class_prior)`,
  `:589-591`), empirical `log(class_count_) - log(class_count_.sum())` (`:600`),
  uniform `-log(n_classes)` (`:602`). ferrolearn `resolve_class_log_prior` checks
  ONLY length then `p.ln()` (the MATCH — sum-0.8 prior fits on both sides),
  empirical `ln(count_c / total)`, uniform `(1/n_classes).ln()`; value-matches. The
  wrong-length error TYPE differs (`InvalidParameter` vs `ValueError`).
- REQ-3: **`min_categories` n_categories_ semantics VALUE (R-DEV-1).** Mirror
  `_validate_n_categories` (`:1446-1466`, `n_categories_ = max(X.max(0)+1,
  min_categories)`) + the `_count` padding (`:1491-1493`) so allocated-but-
  unobserved categories get the smoothed weight `alpha/(N_c+alpha*K_j)`. ferrolearn
  `fn fit` ensures `categories[j]` covers `0..min_cats[j]`
  (`MinCategories::Scalar`/`PerFeature`); value-matches the oracle (`min_categories=4`
  → `n_categories_=[4,4]`, allocated cat-3 `predict_joint_log_proba([[3,0]]) =
  [[-3.58351893845611, -3.58351893845611]]`).
- REQ-4: **unseen-category at predict (R-DEV-3 — DIVERGENCE).** sklearn requires
  category indices `< n_categories_[i]`; the `feature_log_prob_[i][:, X[:,i]]`
  fancy-index (`:1513`) raises `IndexError` for an index beyond `n_categories_`.
  ferrolearn's `log_prob_for` returns a uniform `(1/(n_known_cats+1)).ln()` fallback
  for any unknown category — NO error (`predict([[5,0]])` → `[0]`,
  `predict_proba` → `[[0.5,0.5]]`). DIVERGENCE: sklearn errors / requires the
  category sized in via `min_categories`; ferrolearn silently degrades.
- REQ-5: **`alpha = 0` accepted + `alpha < 0` rejected (R-DEV-2 — THE key fixable
  divergence).** Mirror `_parameter_constraints` `alpha: [Interval(Real, 0, None,
  closed="left")]` (`:1333`): `alpha >= 0` allowed (so `alpha = 0` ACCEPTED, with
  `-inf` where a count is zero), only `alpha < 0` rejected. ferrolearn `fn fit`
  rejects `self.alpha <= F::zero() && self.force_alpha` (line 238) — OVER-REJECTS
  `alpha = 0` under the default `force_alpha=true`. The fix: reject only `alpha <
  0`, allow `alpha = 0`.
- REQ-6: **`force_alpha` floor + `fit_prior` toggle (R-DEV-1).** Mirror
  `_check_alpha` (`:604-626`, floor `1e-10` unless `force_alpha`) and the empirical-
  vs-uniform prior selection on `fit_prior`. ferrolearn `fn fit` calls
  `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`) and honors
  `fit_prior` via `resolve_class_log_prior`.
- REQ-7: **negative / non-integer feature validation (R-DEV-2 — DIVERGENCE).**
  Mirror `_check_X_y` `dtype="int"` + `check_non_negative(X, "CategoricalNB (input
  X)")` (`:1435-1440`): negative → `ValueError`, non-integer float truncated toward
  zero. ferrolearn `fn fit` maps every value via `x[[i,j]].to_usize().unwrap_or(0)`
  — a negative value silently becomes category `0` (no error), a non-negative float
  truncates toward zero (MATCHES for non-negatives). DIVERGENCE only on negatives:
  sklearn rejects, ferrolearn maps to 0.
- REQ-8: **`partial_fit` VALUE on existing categories/classes (R-DEV-1).** Mirror
  the shared `partial_fit` (`:628-709`): increment counts then recompute
  `feature_log_prob_` / `class_log_prior_`. ferrolearn `FittedCategoricalNB::
  partial_fit` increments `category_counts` then recomputes; on data within the
  fitted categories/classes the recompute VALUE matches.
- REQ-9: **`sample_weight` + `partial_fit` new-category/new-class EXTENSION (R-DEV-1
  / R-DEV-7).** (a) sklearn `fit(X, y, sample_weight=None)` (`:712`) weights
  `class_count_` / the `np.bincount` per-category counts; ferrolearn's `impl Fit` is
  `fn fit(&self, x, y)` — NO `sample_weight` parameter. (b) sklearn's `partial_fit`
  keeps `n_categories_` fixed (a new category `>= n_categories_` would `IndexError`)
  and binarizes against the full `classes` list; ferrolearn's `partial_fit` APPENDS
  never-seen categories and inserts never-seen class labels — a non-sklearn
  flexibility (R-DEV-7 deviation, documented in the method doc-comment), observably
  diverging from sklearn's fixed-`n_categories_` contract.
- REQ-10: **fitted-attribute + PyO3 surface (R-DEV-3 / R-DEFER-1/3).** sklearn
  exposes `category_count_`, `feature_log_prob_`, `class_count_`, `class_log_prior_`,
  `n_categories_`, `classes_`, `n_features_in_` (`:1266-1303`). ferrolearn
  `FittedCategoricalNB` exposes ONLY `classes()`; `feature_log_prob` /
  `category_counts` / `categories` / `class_log_prior` / `class_counts` are PRIVATE
  fields with no accessor. **CategoricalNB has NO PyO3 binding** (no
  `_RsCategoricalNB`, no `ferrolearn.CategoricalNB`) — unlike its discrete-NB
  siblings.
- REQ-11: **ferray substrate (R-SUBSTRATE).** `categorical.rs` imports
  `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}`,
  not `ferray-core`.

## Acceptance criteria

All expected values are from the live sklearn 1.5.2 oracle (`from
sklearn.naive_bayes import CategoricalNB`, run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). The categorical fixture is `X = [[0,0,1],[1,0,0],
[0,1,0],[0,0,0],[2,2,1],[2,1,2],[1,2,2],[2,2,2]]`, `y = [0,0,0,0,1,1,1,1]`, query
`q = [[0,0,0],[2,2,2]]`; the binary fixture (for min_categories / unseen) is `X4 =
[[0,0],[0,1],[1,0],[1,1]]`, `y4 = [0,0,1,1]`.

- AC-1 (REQ-1, present & matching): `CategoricalNB().fit(X,y).feature_log_prob_[0]`
  → `[[-0.5596157879354227, -1.252762968495368, -1.9459101490553132],
  [-1.9459101490553132, -1.252762968495368, -0.5596157879354227]]` (= `log((
  category_count_[0]+1)/((category_count_[0]+1).sum(axis=1, keepdims=True)))` with
  `category_count_[0] = [[3,1,0],[0,1,3]]`); `predict_joint_log_proba(q)` →
  `[[-2.3719945443662134, -6.530877627725885], [-6.530877627725885,
  -2.3719945443662134]]`; `predict_proba(q)` → `[[0.9846153846153846,
  0.015384615384615389], [0.015384615384615389, 0.9846153846153846]]`; `predict(q)`
  → `[0,1]`. ferrolearn matches to ~1e-12 (verified by scratch integration run:
  `jll = [[-2.3719945443662134, -6.530877627725886], [...]]`, `pp =
  [[0.9846153846153846, 0.015384615384615375], [...]]`, `pred = [0,1]`).
- AC-2 (REQ-2, present & matching — the MATCH): `CategoricalNB(class_prior=[0.5,
  0.3]).fit(X4,y4).class_log_prior_` → `[-0.6931471805599453, -1.2039728043259361]`
  (sum 0.8, NO error). `class_prior=[0.5]` → `ValueError("Number of priors must
  match number of classes.")`. ferrolearn `with_class_prior([0.5,0.3]).fit`
  succeeds (length-only), `with_class_prior([0.5]).fit` errors (`InvalidParameter`)
  — MATCHES the accept/reject decisions (wrong-length error TYPE differs).
- AC-3 (REQ-3, present & matching): `CategoricalNB(min_categories=4).fit(X4,y4).
  n_categories_` → `[4,4]`; `feature_log_prob_[0]` → `[[-0.6931471805599452,
  -1.791759469228055, -1.791759469228055, -1.791759469228055],
  [-1.791759469228055, -0.6931471805599452, -1.791759469228055,
  -1.791759469228055]]`; `predict_joint_log_proba([[3,0]])` → `[[-3.58351893845611,
  -3.58351893845611]]`; `predict_proba([[3,0]])` → `[[0.5,0.5]]`; `predict([[3,0]])`
  → `[0]`. ferrolearn `with_min_categories(4)` reproduces the IDENTICAL `jll =
  [[-3.58351893845611, -3.58351893845611]]`, `pp = [[0.5,0.5]]` (scratch run).
- AC-4 (REQ-4 pin): `CategoricalNB().fit(X4,y4)` (`n_categories_=[2,2]`) then
  `predict([[5,0]])` → `IndexError("index 5 is out of bounds for axis 1 with size
  2")`. ferrolearn `predict([[5,0]])` → `[0]`, `predict_proba` →
  `[[0.49999999999999994, 0.49999999999999994]]` (uniform fallback, no error) —
  DIVERGENCE.
- AC-5 (REQ-5 pin — THE key divergence): `CategoricalNB(alpha=0.0).fit(X,y)` is
  **ACCEPTED** (`feature_log_prob_[0] = [[-0.2876820724517808, -1.3862943611198906,
  -inf], [-inf, -1.3862943611198906, -0.2876820724517808]]`, with a `divide by zero
  encountered in log` RuntimeWarning, NOT an error). ferrolearn `with_alpha(0.0).
  fit(&X,&y)` → **rejects** (`is_ok=false`, `InvalidParameter { name: "alpha" }`,
  line 238 `alpha <= 0 && force_alpha`). `CategoricalNB(alpha=-1.0).fit(X,y)` →
  `InvalidParameterError("The 'alpha' parameter of CategoricalNB must be a float in
  the range [0.0, inf). Got -1.0 instead.")`; ferrolearn rejects too (both reject
  negatives). FAILS the `alpha = 0` accept contract until the line-238 guard is
  changed from `<= 0` to `< 0`.
- AC-6 (REQ-6, present & matching): with `force_alpha=True` default and `alpha=1`
  the AC-1 `feature_log_prob_` is reproduced; `clamp_alpha(1, true) = 1`. With
  `fit_prior=False`, `class_log_prior_` is uniform `[-log(2), -log(2)]`.
- AC-7 (REQ-7 pin): `CategoricalNB().fit([[-1,0],[0,1],[1,0],[1,1]], [0,0,1,1])` →
  `ValueError("Negative values in data passed to CategoricalNB (input X)")`.
  ferrolearn `fn fit` is `is_ok=true` (maps `-1 → 0`). For non-negative floats both
  truncate toward zero (`0.9 → 0`) — MATCH.
- AC-8 (REQ-8, present & matching, existing categories): chunked `partial_fit` over
  data within the fitted categories/classes reproduces the whole-`fit`
  `feature_log_prob_` (recompute-after-accumulate). The in-tree predict/proba tests
  pin the recompute path.
- AC-9 (REQ-9 pin): `CategoricalNB().fit(X, y, sample_weight=...)` weights
  `class_count_` / `category_count_`; ferrolearn has no `sample_weight` parameter.
  `partial_fit` with a category `>= n_categories_` `IndexError`s in sklearn but is
  silently appended by ferrolearn (the R-DEV-7 extension).
- AC-10 (REQ-10 surface): `hasattr(CategoricalNB().fit(X,y), 'feature_log_prob_')`
  / `'category_count_'` / `'class_count_'` / `'class_log_prior_'` /
  `'n_categories_'` / `'classes_'` all True in sklearn; ferrolearn
  `FittedCategoricalNB` exposes only `classes()`, and `ferrolearn.CategoricalNB`
  does NOT EXIST (no PyO3 binding, unlike the other discrete NB variants).

## REQ status table

Binary (R-DEFER-2). `CategoricalNB` / `FittedCategoricalNB` are existing pub APIs
re-exported at the crate root and consumed non-test by the in-crate pipeline
integration (`impl PipelineEstimator for CategoricalNB` — the production-consumer
surface; grandfathered S5/R-DEFER-1). Unlike the other discrete-NB variants there
is NO `ferrolearn-python` binding. Cites use symbol anchors (ferrolearn) /
`file:line` (sklearn 1.5.2, commit 156ef14). Live oracle = installed sklearn 1.5.2,
run from `/tmp`. Honest underclaim (R-HONEST-3): the smoothing + predict VALUES,
the `min_categories` n_categories_ semantics, the `class_prior` length-only check,
the empirical/uniform priors, the floor/toggle, and the same-categories
`partial_fit` all match and are SHIPPED; the `alpha = 0` over-rejection, the
unseen-category fallback, the negative-feature silent-map, `sample_weight` + the
partial_fit extension, the fitted-attribute/PyO3 surface, and the ferray substrate
are NOT-STARTED. Suggested blocker numbers — the director creates the real issues
(continuing the bayes layer past complement #914-917).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`feature_log_prob_` smoothing + `_joint_log_likelihood` / predict VALUE) | SHIPPED | impl `fn recompute_feature_log_prob` in `categorical.rs` sets `((N_cjk + alpha) / (N_c + alpha*K_j)).ln()` per (feature, class, category), the algebraic identity of `_update_feature_log_prob` (`naive_bayes.py:1498-1506`, `log(category_count_[i]+alpha) - log((category_count_[i]+alpha).sum(axis=1).reshape(-1,1))`); `fn joint_log_likelihood` for `FittedCategoricalNB` computes `class_log_prior[ci] + sum_j log_prob_for(j, ci, x[i,j])`, mirroring `jll += feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_` (`:1508-1515`); `predict`/`predict_proba`/`predict_log_proba`/`predict_joint_log_proba` delegate to `BaseNB` (`.design/bayes/base.md`). Non-test consumer: `impl PipelineEstimator for CategoricalNB` (`fn fit_pipeline` → `FittedCategoricalNBPipeline::predict_pipeline` → `fitted.predict`), the same pipeline-consumer surface `pipeline.rs` cites for GaussianNB/BernoulliNB. Live oracle (AC-1): `feature_log_prob_[0]` → `[[-0.5596..,-1.2528..,-1.9459..],[-1.9459..,-1.2528..,-0.5596..]]`; `predict_joint_log_proba(q)` → `[[-2.3719945443662134, -6.530877627725885], [...]]`; `predict_proba(q)` → `[[0.9846153846153846, 0.015384615384615389], [...]]`; `predict(q)` → `[0,1]`; ferrolearn matches to ~1e-12 (scratch run: `jll = [[-2.3719945443662134, -6.530877627725886], ...]`, `pp = [[0.9846153846153846, 0.015384615384615375], ...]`, `pred = [0,1]`). In-tree `test_categorical_nb_fit_predict` / `test_categorical_nb_predict_proba_sums_to_one` / `test_categorical_nb_predict_proba_ordering` / `test_categorical_nb_three_classes` pin labels + normalization. |
| REQ-2 (`class_log_prior_` empirical/uniform + LENGTH-only `class_prior` — MATCH) | SHIPPED | impl `fn resolve_class_log_prior` validates ONLY `priors.len() != n_classes` then `p.ln()` (explicit), else `ln(count_c / total)` (empirical), else `(1/n_classes).ln()` (uniform), mirroring `_update_class_log_prior` (`naive_bayes.py:580-602`, `if len != n_classes: ValueError; log(class_prior)` `:589-591`; `log(class_count_) - log(class_count_.sum())` `:600`; `-log(n_classes)` `:602`) — discrete NB has NO sum/non-neg check (deliberate MATCH). Non-test consumer: `impl PipelineEstimator` (the `with_class_prior` / empirical-prior path feeds the jll). Live oracle (AC-2): `class_prior=[0.5,0.3]` (sum 0.8) ACCEPTED → `class_log_prior_ = log([0.5,0.3])`; empirical balanced fixture → `[-0.6931471805599452, -0.6931471805599452]`; ferrolearn `with_class_prior([0.5,0.3]).fit` succeeds, `with_class_prior([0.5]).fit` errors. In-tree `test_categorical_nb_single_class` / `test_categorical_nb_unordered_classes` exercise the empirical prior. (Wrong-length error TYPE differs — `InvalidParameter` vs `ValueError` — folded into REQ-10's surface gap.) |
| REQ-3 (`min_categories` n_categories_ semantics VALUE) | SHIPPED | impl `fn fit` ensures `categories[j]` covers `0..min_cats[j]` (`MinCategories::Scalar(m)` broadcast / `PerFeature(v)[j]`, validated against `n_features`), so the count tables hold allocated-but-unobserved slots that `recompute_feature_log_prob` smooths to `alpha/(N_c+alpha*K_j)` — mirroring `_validate_n_categories` (`naive_bayes.py:1446-1466`, `n_categories_ = max(X.max(0)+1, min_categories)`) + `_count` padding (`:1491-1493`). Non-test consumer: `impl PipelineEstimator` (the padded `feature_log_prob` feeds predict). Live oracle (AC-3): `CategoricalNB(min_categories=4).fit(X4,y4).n_categories_` → `[4,4]`, `feature_log_prob_[0]` → `[[-0.6931..,-1.7918..,-1.7918..,-1.7918..],[...]]`, `predict_joint_log_proba([[3,0]])` → `[[-3.58351893845611, -3.58351893845611]]`, `predict_proba` → `[[0.5,0.5]]`, `predict` → `[0]`; ferrolearn `with_min_categories(4)` reproduces the IDENTICAL `jll = [[-3.58351893845611, -3.58351893845611]]`, `pp = [[0.5,0.5]]` (scratch run). `with_min_categories_per_feature` length is validated against `n_features`. |
| REQ-6 (`force_alpha` floor + `fit_prior` toggle) | SHIPPED | impl `fn fit` calls `crate::clamp_alpha(self.alpha, self.force_alpha)` (`base::check_alpha`, the `_check_alpha` floor `1e-10` unless `force_alpha`, `naive_bayes.py:604-626`) and selects empirical/uniform prior on `fit_prior` via `resolve_class_log_prior`. Non-test consumer: `impl PipelineEstimator`. Live oracle (AC-6): `force_alpha=true` default + `alpha=1` reproduces the AC-1 `feature_log_prob_`; `clamp_alpha(1, true) = 1`; `fit_prior=False` → uniform `class_log_prior_`. In-tree `test_categorical_nb_alpha_smoothing_effect` / `test_categorical_nb_default`; `base.rs` `test_check_alpha_*`. |
| REQ-8 (`partial_fit` VALUE on existing categories/classes) | SHIPPED | `FittedCategoricalNB::partial_fit` increments the per-(feature, class, category) `category_counts` then recomputes `feature_log_prob`/`class_log_prior` (same smoothing), mirroring the shared `_BaseDiscreteNB.partial_fit` accumulate-then-recompute (`naive_bayes.py:628-709`, `_count` → `_update_feature_log_prob` → `_update_class_log_prior`). Non-test consumer: in-crate (`FittedCategoricalNB` is produced by `fit`/`fit_pipeline`). On data within the fitted categories/classes the recompute VALUE matches sklearn's reapply-smoothing-to-accumulated-counts (AC-8). In-tree `test_categorical_nb_fit_predict` (the recompute path); the new-category/new-class extension is the REQ-9 deviation. |
| REQ-4 (unseen-category at predict) | NOT-STARTED | open prereq blocker **#920**. sklearn requires category indices `< n_categories_[i]`; the `_joint_log_likelihood` `feature_log_prob_[i][:, X[:,i]]` fancy-index (`naive_bayes.py:1513`) raises `IndexError` for an index beyond `n_categories_` (sized at fit / via `min_categories`). ferrolearn's `fn log_prob_for` returns a uniform `(1/(n_known_cats+1)).ln()` fallback for any category not in `categories[j]` — NO error. Pin (AC-4): fit on `X4,y4` (`n_categories_=[2,2]`), `predict([[5,0]])` → sklearn `IndexError("index 5 is out of bounds for axis 1 with size 2")` vs ferrolearn `[0]`, `predict_proba = [[0.49999999999999994, 0.49999999999999994]]` (scratch run, no error). DIVERGENCE: sklearn errors / requires the category sized in via `min_categories`; ferrolearn silently degrades. (For categories sized in via `min_categories`, both agree — that is REQ-3 SHIPPED; this REQ is the genuinely-unsized index `>= n_categories_`.) In-tree `test_categorical_nb_unseen_category` PINS the (divergent) fallback, so it stays green despite the divergence. |
| REQ-5 (`alpha = 0` accepted + `alpha < 0` rejected) | NOT-STARTED | open prereq blocker **#921**. sklearn `_parameter_constraints` declares `alpha: [Interval(Real, 0, None, closed="left")]` (`naive_bayes.py:1333`) → `alpha >= 0` allowed, so **`alpha = 0` is ACCEPTED** (`CategoricalNB(alpha=0.0).fit(X,y)` succeeds, `feature_log_prob_[0] = [[-0.2876820724517808, -1.3862943611198906, -inf], [-inf, ...]]`, with a `divide by zero encountered in log` RuntimeWarning, NOT an error); only `alpha < 0` is a HARD reject (`CategoricalNB(alpha=-1.0).fit(X,y)` → `InvalidParameterError("The 'alpha' parameter of CategoricalNB must be a float in the range [0.0, inf). Got -1.0 instead.")`). ferrolearn `fn fit` rejects `self.alpha <= F::zero() && self.force_alpha` (`categorical.rs` line 238) — **OVER-REJECTS `alpha = 0`** under the default `force_alpha=true`. Pin (AC-5): `with_alpha(0.0).fit(&X,&y)` is `is_ok=false` in ferrolearn (scratch run) vs sklearn ACCEPTS. **THE single-file fixable divergence** — change the line-238 guard from `alpha <= 0` to `alpha < 0` (reject only negatives, allow zero) in `categorical.rs` `fn fit`; the existing `test_categorical_nb_invalid_alpha_zero` then needs updating (it currently pins the over-rejection). Same class as multinomial #900 / complement #914 (the `alpha >= 0` family) except CategoricalNB's bug is over-rejection of zero. The critic should pin this FIRST. |
| REQ-7 (negative / non-integer feature validation) | NOT-STARTED | open prereq blocker **#922**. sklearn `_check_X_y` validates `X` via `_validate_data(X, y, dtype="int", ...)` + `check_non_negative(X, "CategoricalNB (input X)")` (`naive_bayes.py:1435-1440`): a negative value raises `ValueError("Negative values in data passed to CategoricalNB (input X)")`; a non-integer float is truncated toward zero by the `dtype="int"` cast. ferrolearn `fn fit` maps every value via `x[[i,j]].to_usize().unwrap_or(0)` — a negative value silently becomes category `0` (no error; scratch run `fit(X_with_-1, y)` `is_ok=true`), a non-negative float truncates toward zero (`0.9 → 0`, MATCHES sklearn for non-negatives). Pin (AC-7): `CategoricalNB().fit([[-1,0],...], y)` → sklearn `ValueError` vs ferrolearn `is_ok=true`. DIVERGENCE only on negatives. (May be fixed alongside REQ-4/#920 since both concern silently-tolerated invalid category indices.) |
| REQ-9 (`sample_weight` + `partial_fit` extension) | NOT-STARTED | open prereq blocker **#924**. (a) sklearn `fit(X, y, sample_weight=None)` (`:712`) weights `class_count_ = Y.sum(axis=0)` and the `np.bincount(X_feature[mask], weights=...)` per-category counts (`:1468-1496`); ferrolearn's `impl Fit<Array2<F>, Array1<usize>>` has signature `fn fit(&self, x, y)` — NO `sample_weight` parameter on `fit` or `partial_fit`. (b) sklearn's `partial_fit` keeps `n_categories_` fixed (a category `>= n_categories_` in a later chunk `IndexError`s — `predict([[4,0]])` after `min_categories=4` → `IndexError("index 4 is out of bounds for axis 1 with size 4")`) and binarizes against the full `classes` list; ferrolearn's `FittedCategoricalNB::partial_fit` APPENDS never-seen category values to `categories[j]` and inserts never-seen class labels into `classes` — a non-sklearn flexibility (R-DEV-7 deviation, documented in the method doc-comment), observably diverging from sklearn's fixed-`n_categories_` contract. Pin (AC-9): both the missing `sample_weight` and the extension behavior. |
| REQ-10 (fitted-attribute + PyO3 surface) | NOT-STARTED | open prereq blocker **#923**. sklearn exposes `category_count_` (list of `(n_classes, n_categories_i)` arrays), `feature_log_prob_` (same shape), `class_count_`, `class_log_prior_`, `n_categories_`, `classes_`, `n_features_in_` (`naive_bayes.py:1266-1303`). `FittedCategoricalNB` stores `feature_log_prob`/`category_counts`/`categories`/`class_log_prior`/`class_counts` as PRIVATE fields with no accessor — only `classes()` (via `HasClasses`) is public. **CategoricalNB has NO PyO3 binding** — unlike `_RsMultinomialNB`/`_RsBernoulliNB`/`_RsComplementNB` (`ferrolearn-python/src/extras.rs`) + `ferrolearn.MultinomialNB`/`BernoulliNB`/`ComplementNB` (`_extras.py`), there is NO `_RsCategoricalNB` and NO `ferrolearn.CategoricalNB`, so `import ferrolearn` cannot reach CategoricalNB at all. Pin (AC-10): `hasattr(sklearn fitted, 'feature_log_prob_')`/`'category_count_'`/`'n_categories_'` True; `ferrolearn.CategoricalNB` raises `AttributeError`. Also subsumes the `class_prior` wrong-length error-TYPE sub-item (REQ-2) and the negative-feature MESSAGE/TYPE sub-item (REQ-7). |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker **#925**. `categorical.rs` imports `ndarray::{Array1, Array2}` + `num_traits::{Float, FromPrimitive, ToPrimitive}` (the wrong substrate); not migrated to `ferray-core` (R-SUBSTRATE-1/2). |

## Architecture

`categorical.rs` follows the unfitted/fitted split (CLAUDE.md naming) for a single
estimator that delegates its entire prediction pipeline to the shared `BaseNB<F>`
trait (`base.rs`):

- `MinCategories { Scalar(usize) | PerFeature(Vec<usize>) }` — the `min_categories`
  analog.
- `CategoricalNB<F>` (`alpha`, `class_prior: Option<Vec<F>>`, `fit_prior`,
  `force_alpha`, `min_categories: Option<MinCategories>`) → `Fit<Array2<F>,
  Array1<usize>>` → `FittedCategoricalNB<F>` (`classes`, `class_log_prior`,
  `feature_log_prob: Vec<Vec<Vec<F>>>` indexed `[feature][class][category]`,
  `category_counts: Vec<Vec<Vec<usize>>>`, `categories: Vec<Vec<usize>>`,
  `class_counts`, `n_features`, plus `alpha`/`class_prior`/`fit_prior` for
  `partial_fit`).

Generic over `F: Float + Send + Sync + 'static`; every public method returns
`Result<_, FerroError>` (R-CODE-2). The ragged `Vec<Vec<Vec<F>>>` shape mirrors
sklearn's `feature_log_prob_` being a *list* of `(n_classes, n_categories_i)`
arrays (each feature can have a different `K_j`) — unlike Multinomial/Complement
whose `feature_log_prob_` is a single rectangular `(n_classes, n_features)` array.

**Fit path (`fn fit`).** Validation rejects empty `X` (`InsufficientSamples`),
`n_samples != y.len()` (`ShapeMismatch`), and `self.alpha <= 0 && force_alpha`
(`InvalidParameter`, line 238 — which OVER-REJECTS `alpha = 0`, REQ-5/#921, vs
sklearn's `Interval(Real, 0, None, closed="left")` `:1333` that ALLOWS zero). It
validates `MinCategories::PerFeature` length against `n_features`, then `alpha =
clamp_alpha(self.alpha, force_alpha)` (the `_check_alpha` floor, REQ-6). Per
feature it discovers observed unique category values (`x[[i,j]].to_usize().
unwrap_or(0)` — NO negative/non-integer guard, REQ-7/#922: negatives silently map
to category 0), ensures `categories[j]` covers `0..min_cats[j]` (the
`min_categories` padding, REQ-3), and builds the per-(class, category)
`category_counts`. `recompute_feature_log_prob` then computes `((N_cjk + alpha) /
(N_c + alpha*K_j)).ln()` (the `_update_feature_log_prob` identity, `:1498-1506`,
REQ-1) and `resolve_class_log_prior` resolves the priors (explicit LENGTH-only /
empirical / uniform, REQ-2). No `sample_weight` (REQ-9).

**Prediction (delegated to `BaseNB`).** `joint_log_likelihood` computes
`class_log_prior[ci] + sum_j log_prob_for(j, ci, x[i,j])`, mirroring `jll +=
feature_log_prob_[i][:, X[:,i]].T; jll += class_log_prior_` (`:1508-1515`, REQ-1).
`log_prob_for` looks up the cached `feature_log_prob[j][ci][cat_idx]` when the
category is known, ELSE returns a uniform `(1/(n_known_cats+1)).ln()` fallback —
the REQ-4/#920 divergence (sklearn `IndexError`s on a category `>= n_categories_`).
`predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
delegate to the `BaseNB` provided methods (the `_BaseNB` pipeline; see
`.design/bayes/base.md`).

**`partial_fit` (`fn partial_fit`).** Increments the per-(feature, class, category)
`category_counts` then recomputes `feature_log_prob`/`class_log_prior` (same
smoothing), so on data within the fitted categories/classes chunked `partial_fit`
matches whole `fit` (REQ-8). DEVIATION (R-DEV-7, REQ-9): it APPENDS never-seen
category values to `categories[j]` (allocating a slot per class) AND inserts
never-seen class labels into `classes` — a non-sklearn flexibility documented in the
method doc-comment ("Unlike sklearn's strict integer-index model …"), where sklearn
keeps `n_categories_` fixed and would `IndexError` on an out-of-range category.

**Scoring.** `score` = `correct / n` (mean accuracy), the `ClassifierMixin.score`
analog.

**Consumer wiring.** The non-test production consumer is the in-crate **pipeline
integration**: `impl PipelineEstimator<F> for CategoricalNB<F>` (`fn fit_pipeline`,
maps float labels → `usize`) produces `FittedCategoricalNBPipeline` whose
`fn predict_pipeline` consumes `fitted.predict` — the same `Box<dyn
FittedPipelineEstimator>`-producing surface `pipeline.rs` cites for
GaussianNB/BernoulliNB. **There is NO `ferrolearn-python` binding** for CategoricalNB
(REQ-10/#923).

**Invariants held vs sklearn:** the `feature_log_prob_` smoothing VALUE + the full
predict pipeline VALUE to ~1e-12 (AC-1); the `min_categories` n_categories_
semantics + allocated-slot smoothing (AC-3); the `class_prior` LENGTH-only
accept/reject decisions (AC-2, the MATCH); the empirical/uniform `class_log_prior_`
(REQ-2); the `force_alpha` floor + `fit_prior` toggle (AC-6); non-negative-float
truncation toward zero (AC-7, the MATCH); same-categories `partial_fit` == `fit`
(AC-8); `classes_` ordering; `predict_proba` rows sum to 1.

**Invariants NOT held vs sklearn:** `alpha = 0` accept (REQ-5/#921 — the key
single-file divergence; ferrolearn OVER-REJECTS zero); the unseen-category
`IndexError` (REQ-4/#920 — ferrolearn uniform fallback); the negative-feature
reject (REQ-7/#922 — ferrolearn maps to 0); `sample_weight` + the fixed-
`n_categories_` `partial_fit` contract (REQ-9/#924 — ferrolearn extends); the
fitted-attribute + PyO3 surface (REQ-10/#923 — no binding at all); the ferray
substrate (REQ-11/#925).

## Verification

Library crate (green at baseline `e170c3c5` for the existing contract):
```
cargo test -p ferrolearn-bayes --lib categorical
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s (`test_categorical_nb_fit_predict`,
`test_categorical_nb_predict_proba_sums_to_one`, `test_categorical_nb_has_classes`,
`test_categorical_nb_alpha_smoothing_effect`, `test_categorical_nb_invalid_alpha_zero`,
`test_categorical_nb_invalid_alpha_negative`, `test_categorical_nb_single_class`,
`test_categorical_nb_unseen_category`, `test_categorical_nb_three_classes`,
`test_categorical_nb_unordered_classes`, `test_categorical_nb_pipeline`,
`test_categorical_nb_predict_proba_ordering`, `test_categorical_nb_f32`, …) pin
ferrolearn's current behavior. **None compares against the live sklearn oracle**;
two PIN the divergent behavior and so stay green despite the divergence:
`test_categorical_nb_invalid_alpha_zero` asserts `alpha=0` is REJECTED (the REQ-5
over-rejection) and `test_categorical_nb_unseen_category` asserts the uniform
fallback succeeds (the REQ-4 divergence). The SHIPPED REQs (REQ-1 smoothing+predict
VALUE, REQ-2 priors+class_prior MATCH, REQ-3 min_categories, REQ-6 floor/toggle,
REQ-8 partial_fit) value-match the oracle; the rest are NOT-STARTED.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the divergences a critic
should pin (R-CHAR-3 expected values). **Pin the deterministic single-file one
FIRST**: REQ-5 (`alpha = 0` accept — change the line-238 guard `<= 0` → `< 0`):
```
# REQ-5 (#921) alpha=0 — sklearn ACCEPTS (force_alpha=True default), ferrolearn over-rejects (THE key fix)
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np, warnings; X=np.array([[0,0,1],[1,0,0],[0,1,0],[0,0,0],[2,2,1],[2,1,2],[1,2,2],[2,2,2]]); y=np.array([0,0,0,0,1,1,1,1]);
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); m=CategoricalNB(alpha=0.0).fit(X,y)
print('alpha=0 OK; flp[0]=', m.feature_log_prob_[0].tolist())"  # SUCCESS; flp[0] has -inf where count=0   (ferro: with_alpha(0.0).fit -> InvalidParameter)
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]);
try: CategoricalNB(alpha=-1.0).fit(X,y)
except Exception as e: print(type(e).__name__,'::',e)"  # InvalidParameterError :: The 'alpha' parameter of CategoricalNB must be a float in the range [0.0, inf). Got -1.0 instead.  (ferro rejects too)
# REQ-1 (present) feature_log_prob_ + predict VALUE
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[0,0,1],[1,0,0],[0,1,0],[0,0,0],[2,2,1],[2,1,2],[1,2,2],[2,2,2]]); y=np.array([0,0,0,0,1,1,1,1]); m=CategoricalNB().fit(X,y); q=np.array([[0,0,0],[2,2,2]]); print(m.feature_log_prob_[0].tolist()); print(m.predict_joint_log_proba(q).tolist()); print(m.predict_proba(q).tolist()); print(m.predict(q).tolist())"  # flp[0] [[-0.5596..,-1.2528..,-1.9459..],[-1.9459..,-1.2528..,-0.5596..]]; jll [[-2.3720..,-6.5309..],[...]]; pp [[0.9846..,0.0154..],[...]]; pred [0,1]
# REQ-3 (present) min_categories n_categories_ + allocated-slot smoothing
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]); m=CategoricalNB(min_categories=4).fit(X,y); print(m.n_categories_.tolist()); print(m.predict_joint_log_proba(np.array([[3,0]])).tolist())"  # [4,4]; [[-3.58351893845611, -3.58351893845611]]
# REQ-4 (#920) unseen category >= n_categories_ — sklearn IndexError, ferrolearn uniform fallback
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]); m=CategoricalNB().fit(X,y);
try: m.predict(np.array([[5,0]]))
except Exception as e: print(type(e).__name__,'::',e)"  # IndexError :: index 5 is out of bounds for axis 1 with size 2   (ferro: predict -> [0], proba [[0.5,0.5]])
# REQ-7 (#922) negative features — sklearn ValueError, ferrolearn maps to 0
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[-1,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]);
try: CategoricalNB().fit(X,y)
except Exception as e: print(type(e).__name__,'::',e)"  # ValueError :: Negative values in data passed to CategoricalNB (input X)   (ferro: is_ok=true)
# REQ-2 (present, MATCH) class_prior length-only (sum 0.8 accepted)
python3 -c "from sklearn.naive_bayes import CategoricalNB; import numpy as np; X=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]); print(CategoricalNB(class_prior=[0.5,0.3]).fit(X,y).class_log_prior_.tolist())"  # [-0.6931471805599453, -1.2039728043259361]  (ferro MATCHES)
```
A characterization pin (R-CHAR-3) for each NOT-STARTED REQ belongs in
`ferrolearn-bayes/tests/divergence_categorical.rs`, asserting the live-sklearn
expected values above and FAILING against current `categorical.rs` (REQ-5 the
`alpha = 0` accept is the cleanest single-file fix — note the existing
`test_categorical_nb_invalid_alpha_zero` must be UPDATED, as it currently pins the
over-rejection). REQ-1/REQ-2/REQ-3/REQ-6/REQ-8 already match and should be guarded
by non-regression pins.

ferrolearn-python: CategoricalNB has NO binding (REQ-10/#923). Adding
`_RsCategoricalNB` (`extras.rs`) + `ferrolearn.CategoricalNB` (`_extras.py`) — with
`min_categories` kwarg, `predict_proba`/`predict_log_proba`/`score`/`partial_fit`,
and the `category_count_`/`feature_log_prob_`/`class_count_`/`class_log_prior_`/
`n_categories_` getters — is the binding work; until then there is no pytest
surface to assert.

## Blockers to open

(Director creates the real issues; the numbers are SUGGESTIONS continuing the bayes
layer past complement #914-917. #919 is this doc's crosslink tracking issue.)

- **#920** — REQ-4 (unseen-category at predict): match sklearn's `IndexError` for a
  category `>= n_categories_` (`naive_bayes.py:1513`) instead of the uniform
  `1/(n_cats+1)` fallback in `fn log_prob_for` (or require sizing via
  `min_categories`); update `test_categorical_nb_unseen_category` which pins the
  fallback. (May fold with #922 — both are silently-tolerated invalid category
  indices.)
- **#921** — REQ-5 (`alpha = 0` accept + `alpha < 0` reject): change the
  `categorical.rs` `fn fit` line-238 guard from `self.alpha <= F::zero() &&
  force_alpha` to reject only `alpha < 0` (mirroring `_parameter_constraints` `alpha:
  Interval(Real, 0, None, closed="left")`, `naive_bayes.py:1333` — alpha=0 ALLOWED).
  **The cleanest single-file deterministic fix** — the critic should pin this FIRST.
  Update `test_categorical_nb_invalid_alpha_zero` (currently pins the over-rejection).
- **#922** — REQ-7 (negative / non-integer feature validation): reject negative `X`
  with a `check_non_negative`-equivalent (`naive_bayes.py:1432`/`:1439`) instead of
  silently mapping `to_usize().unwrap_or(0)`; non-negative-float truncation already
  matches. (May fold with #920.)
- **#923** — REQ-10 (fitted-attribute + PyO3 surface): expose `category_count_`/
  `feature_log_prob_`/`class_count_`/`class_log_prior_`/`n_categories_` accessors on
  `FittedCategoricalNB`, and ADD a `_RsCategoricalNB` binding (`ferrolearn-python/
  src/extras.rs`) + `ferrolearn.CategoricalNB` (`_extras.py`) with the
  `min_categories` kwarg + `predict_proba`/`predict_log_proba`/`score`/`partial_fit`
  + those getters (CategoricalNB currently has NO Python binding at all, unlike its
  siblings). Also align the `class_prior` wrong-length error TYPE (REQ-2) and the
  negative-feature MESSAGE/TYPE (REQ-7) with sklearn.
- **#924** — REQ-9 (`sample_weight` + `partial_fit` extension): add weighted
  `class_count_`/`category_count_` (needs a `sample_weight` parameter on `fit`/
  `partial_fit`, `:712`/`:1468-1496`), and decide on the `partial_fit`
  new-category/new-class extension — either keep it as a documented R-DEV-7 deviation
  or match sklearn's fixed-`n_categories_` + full-`classes`-list contract.
- **#925** — REQ-11 (ferray substrate): migrate `categorical.rs` off
  `ndarray`/`num-traits` to `ferray-core` (R-SUBSTRATE).
