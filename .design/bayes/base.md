# Naive-Bayes base (sklearn `_BaseNB` / `_BaseDiscreteNB`)

<!--
tier: 3-component
status: draft
baseline-commit: e91f4de7
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # _BaseNB(ClassifierMixin, BaseEstimator): abstract _joint_log_likelihood (:42-53), abstract _check_X (:55-60), predict_joint_log_proba (:62-84), predict = classes_[argmax(jll)] (:86-103), predict_log_proba = jll - logsumexp(jll) (:105-126), predict_proba = exp(predict_log_proba) (:128-144). _BaseDiscreteNB(_BaseNB): _check_alpha floor 1e-10 unless force_alpha (:604-626), _update_class_log_prior (:580-602), abstract _count (:542-557) / _update_feature_log_prob (:559-570), partial_fit (:628-709), coef_/intercept_ properties.
ferrolearn-module: ferrolearn-bayes/src/base.rs
parity-ops: _BaseNB (._joint_log_likelihood, .predict, .predict_log_proba, .predict_proba, .predict_joint_log_proba), _BaseDiscreteNB (._check_alpha)
crosslink-issue: 889
-->

## Summary

`ferrolearn-bayes/src/base.rs` mirrors scikit-learn's abstract naive-Bayes
class hierarchy (`sklearn/naive_bayes.py`): the abstract `_BaseNB` — the
prediction pipeline shared by every NB variant — and the `_BaseDiscreteNB`
smoothing helper `_check_alpha`. ferrolearn expresses the `_BaseNB` contract as
the `BaseNB<F>` trait: an implementor supplies the two abstract pieces —
`joint_log_likelihood` (sklearn's abstract `_joint_log_likelihood`, the
unnormalized `log P(c) + log P(x|c)`) and `nb_classes` (the sorted `classes_`)
— and inherits the full prediction pipeline as provided methods:

- `nb_predict` — `classes_[argmax(jll, axis=1)]` (`_BaseNB.predict`,
  `naive_bayes.py:103`), `np.argmax` first-max / smallest-label tie-break.
- `nb_predict_log_proba` — `jll - logsumexp(jll, axis=1)`
  (`_BaseNB.predict_log_proba`, `naive_bayes.py:123-126`), via the crate-level
  numerically stable `log_softmax_rows`.
- `nb_predict_proba` — `exp(predict_log_proba)` (`_BaseNB.predict_proba`,
  `naive_bayes.py:144`).
- `nb_predict_joint_log_proba` — the unnormalized joint log-probability
  (`_BaseNB.predict_joint_log_proba`, `naive_bayes.py:84`).

`base.rs` also re-homes the `_BaseDiscreteNB._check_alpha` smoothing floor as
`pub(crate) fn check_alpha<F: Float>(alpha, force_alpha) -> F` (formerly
`lib.rs::clamp_alpha`); `lib.rs` re-exports it as `clamp_alpha` so the four
discrete variants' fit call sites are unchanged. `log_softmax_rows` stays in
`lib.rs` (the variants and the trait default both call `crate::log_softmax_rows`)
— no numerics move.

The five fitted Naive Bayes types implement `BaseNB<F>` and **delegate** their
inherent `predict_proba` / `predict_log_proba` / `predict_joint_log_proba`
methods and their `impl Predict` to the trait defaults. They are the non-test
production consumers of this base (R-DEFER-1); the existing 91 in-tree variant
tests are the green verification (the refactor is behavior-preserving — no
observable output changes). `HasClasses` is unchanged.

This is a behavior-preserving refactor that lifts the previously-duplicated
predict pipeline (per-variant private `joint_log_likelihood` + inherent
`predict_*` + `Predict::predict`) onto one shared, sklearn-grounded base. The
per-variant **value** parity REQs (each `_joint_log_likelihood` matching its
sklearn variant to ULPs) live in the per-variant design docs
(`.design/bayes/{gaussian,multinomial,bernoulli,complement,categorical}.md`);
this doc covers only the SHARED base contract.

`conjugate.rs` (`posterior_normal_normal`, the Normal-Normal conjugate-prior
closed-form update) has **no scikit-learn analog** — it is not a naive-Bayes
estimator and not part of `_BaseNB`/`_BaseDiscreteNB`. It was mis-routed to
`_BaseNB`; with this iteration that route is re-pointed to `base.rs`, and
`conjugate.rs` becomes **UNROUTED** (excluded like `ferrolearn-decomp/src/umap.rs`
— a no-sklearn-analog module, out of scope per goal.md "Out of scope"). The
route swap is one-for-one, so routed-count is unchanged.

## Algorithm (sklearn — the contract)

### `_BaseNB` — the abstract prediction pipeline (`naive_bayes.py:39-144`)

`_BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta)` declares two
abstract methods — `_joint_log_likelihood(X)` (`:42-53`, "the unnormalized
posterior log probability of X, i.e. `log P(c) + log P(x|c)`, shape
`(n_samples, n_classes)`") and `_check_X(X)` (`:55-60`, input validation only
used in `predict*`) — and provides the full pipeline on top:

- **`predict(X)`** (`:86-103`): `jll = self._joint_log_likelihood(X)`;
  `return self.classes_[np.argmax(jll, axis=1)]`. `np.argmax` returns the
  **first** (smallest-index) maximum on ties; since `classes_` is sorted, the
  tie-break is "smallest class label wins".
- **`predict_log_proba(X)`** (`:105-126`): `jll = self._joint_log_likelihood(X)`;
  `log_prob_x = logsumexp(jll, axis=1)`; `return jll - np.atleast_2d(log_prob_x).T`.
- **`predict_proba(X)`** (`:128-144`): `return np.exp(self.predict_log_proba(X))`.
- **`predict_joint_log_proba(X)`** (`:62-84`): validates then returns
  `self._joint_log_likelihood(X)` (the unnormalized scores).

### `_BaseDiscreteNB._check_alpha` (`naive_bayes.py:604-626`)

`alpha_lower_bound = 1e-10`; when the minimum alpha `< alpha_lower_bound and not
self.force_alpha`, sklearn warns ("alpha too small will result in numeric
errors, setting alpha = 1.0e-10. Use `force_alpha=True` to keep alpha
unchanged.") and returns `np.maximum(alpha, alpha_lower_bound)`; otherwise it
returns alpha unchanged. (The array-alpha shape/positivity checks `:609-617` are
not part of ferrolearn's scalar-alpha surface.)

### `_BaseDiscreteNB` — the rest (NOT-STARTED on this base)

`coef_` / `intercept_` properties (feature_log_prob / class_log_prior views),
abstract `_count` / `_update_feature_log_prob` (`:542-570`),
`_update_class_log_prior` (`:580-602`), and the shared `partial_fit`
(`:628-709`) are part of `_BaseDiscreteNB` but are **not** lifted onto this
base in this iteration (the variants implement `partial_fit` and prior handling
individually). See REQ-5/6/7.

## ferrolearn (what exists / what this iteration adds)

All in `ferrolearn-bayes/src/base.rs`, generic over `F: Float`. Every method
returns `Result<_, FerroError>` (no panics, R-CODE-2).

- **`pub trait BaseNB<F: Float>`** — abstract `fn joint_log_likelihood(&self,
  &Array2<F>) -> Result<Array2<F>, FerroError>` (sklearn `_joint_log_likelihood`)
  and `fn nb_classes(&self) -> &[usize]` (sklearn `classes_`); provided
  `fn nb_predict` / `fn nb_predict_log_proba` / `fn nb_predict_proba` /
  `fn nb_predict_joint_log_proba` (the pipeline). `nb_predict` uses a `fold`
  with `partial_cmp(...)` matching `Ordering::Greater` for strict-`>` first-max
  argmax (no `.unwrap()`). `nb_predict_log_proba` calls `crate::log_softmax_rows`.
- **`pub(crate) fn check_alpha<F: Float>(alpha, force_alpha) -> F`** — the
  re-homed `clamp_alpha` (sklearn `_BaseDiscreteNB._check_alpha`); uses
  `unwrap_or_else(F::epsilon)` for the `1e-10` literal conversion (no `.unwrap()`).

**Consumers (non-test).** Each `Fitted*NB` implements `BaseNB<F>`:

- `FittedGaussianNB` / `FittedMultinomialNB` / `FittedBernoulliNB` /
  `FittedCategoricalNB` — `fn joint_log_likelihood` wraps the existing private
  body (feature-count shape guard + the per-variant scores) and returns
  `Result`; `fn nb_classes` returns `&self.classes`.
- `FittedComplementNB` — `fn joint_log_likelihood` is the **extracted**
  `complement_scores` body (with the feature-count shape guard); `fn nb_classes`
  returns `&self.classes`. (ComplementNB stores `feature_log_prob_` with
  sklearn's positive sign so `argmax(X @ weights.T)` predicts directly; the jll
  IS `complement_scores`.)

The variants' inherent `pub fn predict_proba` / `pub fn predict_log_proba` /
`pub fn predict_joint_log_proba` and `impl Predict::predict` become thin
delegators: `BaseNB::nb_predict_proba(self, x)` etc. The public API is
unchanged — callers using `fitted.predict_proba(x)` without importing the trait
still work (inherent methods shadow / wrap the trait). `BernoulliNB` binarizes
inside its `joint_log_likelihood` so the delegating wrappers need no special
casing. `lib.rs` adds `pub mod base; pub use base::BaseNB;` and
`pub(crate) use base::check_alpha as clamp_alpha;`.

## Requirements

- REQ-1: **`_BaseNB.predict` — `classes_[argmax(jll)]` + tie-break (R-DEV-1/3).**
  Mirror `_BaseNB.predict` (`naive_bayes.py:86-103`): per-row argmax over the
  joint log-likelihood with `np.argmax` first-max (smallest-index → smallest
  label) tie-break. ferrolearn `BaseNB::nb_predict`.
- REQ-2: **`_BaseNB.predict_log_proba` — `jll - logsumexp(jll)` (R-DEV-3).**
  Mirror `_BaseNB.predict_log_proba` (`naive_bayes.py:105-126`). ferrolearn
  `BaseNB::nb_predict_log_proba` via `crate::log_softmax_rows`.
- REQ-3: **`_BaseNB.predict_proba` — `exp(predict_log_proba)` (R-DEV-3).**
  Mirror `_BaseNB.predict_proba` (`naive_bayes.py:128-144`). ferrolearn
  `BaseNB::nb_predict_proba`.
- REQ-4: **`_BaseDiscreteNB._check_alpha` — floor 1e-10 unless `force_alpha`
  (R-DEV-1).** Mirror `_check_alpha` (`naive_bayes.py:604-626`). ferrolearn
  `base::check_alpha` (re-exported `clamp_alpha`).
- REQ-5: **`_BaseDiscreteNB.coef_` / `intercept_` (R-DEV-3).** sklearn exposes
  `coef_`/`intercept_` views over `feature_log_prob_`/`class_log_prior_`. No
  ferrolearn analog on the fitted types or the trait.
- REQ-6: **`_BaseDiscreteNB.partial_fit` / `_count` / `_update_feature_log_prob`
  (R-DEV-2).** sklearn factors fitting through abstract `_count` /
  `_update_feature_log_prob` (`naive_bayes.py:542-570`) driven by a shared
  `partial_fit` (`:628-709`). ferrolearn implements `partial_fit` per variant
  with no shared seam on this base.
- REQ-7: **`_BaseDiscreteNB._update_class_log_prior` / `class_prior`
  (R-DEV-2).** sklearn centralizes prior selection in `_update_class_log_prior`
  (`naive_bayes.py:580-602`). ferrolearn duplicates it per discrete variant.

## Acceptance criteria

Expected values are from the live sklearn 1.5.2 oracle, never literal-copied
from ferrolearn (R-CHAR-3). The base predict pipeline is verified through one
variant (MultinomialNB), since the base IS the pipeline every variant's
`predict*` now routes through.

- AC-1 (REQ-1/2/3, present & matching): MultinomialNB fixture
  `X=[[1,2],[0,3],[4,0],[3,1]]`, `y=[0,0,1,1]`, query `[[2,2]]`. Live sklearn:
  `predict_log_proba` → `[[-0.5470675457484475, -0.8642776061017265]]`;
  `predict_proba` → `[[0.5786441724102462, 0.4213558275897536]]`; `predict` →
  `[0]`. ferrolearn (now routed through `BaseNB`) matches to ~1e-12.
- AC-2 (REQ-1 tie-break): a stub `BaseNB` with `jll=[[2.0, 2.0]]`,
  `classes=[5, 9]` → `nb_predict` → `[5]` (smallest label, `np.argmax` first-max).
  In-tree `base.rs` unit test `test_nb_predict_tie_breaks_to_smallest_index`.
- AC-3 (REQ-4): `check_alpha(0.0, force_alpha=true)` → `0.0`;
  `check_alpha(0.0, force_alpha=false)` → `1e-10`;
  `check_alpha(1.0, force_alpha=false)` → `1.0` (sklearn
  `alpha_lower_bound = 1e-10`, `naive_bayes.py:618`). In-tree `base.rs` unit
  tests `test_check_alpha_*`.
- AC-4 (behavior-preserving): the 91 in-tree variant lib tests + 9 doctests stay
  green after the variants delegate to `BaseNB` — no observable output change.

## REQ status table

Binary (R-DEFER-2). The non-test production consumers are the five `Fitted*NB`
types whose predict pipeline delegates to `BaseNB`; green verification is the
existing 91-test variant suite (exercising the delegated pipeline) plus the live
sklearn oracle sanity-check (AC-1) and the four new `base.rs` unit tests. Cites
use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`_BaseNB.predict`) | SHIPPED | provided method `fn nb_predict` for `BaseNB` (first-max argmax over `joint_log_likelihood`, `classes_[best]`) mirrors `_BaseNB.predict` (`naive_bayes.py:103`). Non-test consumers: `impl Predict::predict` for all five `Fitted*NB` delegate to `BaseNB::nb_predict`. Verified: live oracle (AC-1) `MultinomialNB.predict([[2,2]])` → `[0]`; ferrolearn matches; 91 variant tests green; `base.rs` `test_nb_predict_argmax_to_classes` + `test_nb_predict_tie_breaks_to_smallest_index`. |
| REQ-2 (`_BaseNB.predict_log_proba`) | SHIPPED | provided method `fn nb_predict_log_proba` (calls `crate::log_softmax_rows`) mirrors `_BaseNB.predict_log_proba` (`naive_bayes.py:123-126`). Non-test consumers: each variant's `pub fn predict_log_proba` delegates here. Verified: live oracle (AC-1) `predict_log_proba([[2,2]])` → `[[-0.5470675457484475, -0.8642776061017265]]`; ferrolearn matches to ~1e-12. |
| REQ-3 (`_BaseNB.predict_proba`) | SHIPPED | provided method `fn nb_predict_proba` (`exp` of `nb_predict_log_proba`) mirrors `_BaseNB.predict_proba` (`naive_bayes.py:144`). Non-test consumers: each variant's `pub fn predict_proba` delegates here (value-identical to the prior row-softmax). Verified: live oracle (AC-1) `predict_proba([[2,2]])` → `[[0.5786441724102462, 0.4213558275897536]]`; ferrolearn matches; `*_predict_proba_sums_to_one` tests green; `base.rs` `test_nb_predict_proba_is_exp_of_log_proba_and_sums_to_one`. |
| REQ-4 (`_BaseDiscreteNB._check_alpha`) | SHIPPED | `pub(crate) fn check_alpha` (re-homed `clamp_alpha`) mirrors `_check_alpha` (`naive_bayes.py:604-626`, floor `1e-10` unless `force_alpha`). Non-test consumers: `MultinomialNB`/`BernoulliNB`/`ComplementNB`/`CategoricalNB` `fn fit` call `crate::clamp_alpha` (`pub(crate) use base::check_alpha as clamp_alpha`). Verified (AC-3): `base.rs` `test_check_alpha_force_alpha_keeps_value` + `test_check_alpha_floors_when_not_forced`; `*_alpha_smoothing_effect` / `*_default` variant tests green. |
| REQ-5 (`_BaseDiscreteNB.coef_` / `intercept_`) | NOT-STARTED | open prereq blocker. sklearn exposes `coef_ = feature_log_prob_[1:]` (binary) / `intercept_ = class_log_prior_[1:]` (`_BaseDiscreteNB` properties). No ferrolearn `coef_`/`intercept_` accessor on the fitted discrete types or `BaseNB`/a `BaseDiscreteNB` trait. |
| REQ-6 (`_BaseDiscreteNB.partial_fit` / `_count` / `_update_feature_log_prob`) | NOT-STARTED | open prereq blocker. sklearn's shared `partial_fit` (`naive_bayes.py:628-709`) drives abstract `_count` / `_update_feature_log_prob` (`:542-570`). ferrolearn implements `partial_fit` per variant (each `Fitted*NB::partial_fit`); no shared count/feature-log-prob seam on this base. |
| REQ-7 (`_BaseDiscreteNB._update_class_log_prior` / `class_prior`) | NOT-STARTED | open prereq blocker. sklearn's `_update_class_log_prior` (`naive_bayes.py:580-602`) centralizes empirical/uniform/explicit prior selection. ferrolearn duplicates this per discrete variant (`fit`/`partial_fit` prior blocks); not lifted onto this base. |

## Architecture

`base.rs` is a trait + free function, not an estimator — it has no
unfitted/fitted split of its own. The shape is:

- **`trait BaseNB<F: Float>`** with two required methods (`joint_log_likelihood`
  = sklearn abstract `_joint_log_likelihood`; `nb_classes` = `classes_`) and
  four provided methods implementing the `_BaseNB` pipeline. The provided
  `nb_predict` reproduces `np.argmax`'s first-max via a `fold` that replaces the
  running-best index only on `Ordering::Greater` (strict `>`), so ties keep the
  earlier/smaller index — byte-for-byte the variants' prior inline argmax.
  `nb_predict_log_proba` defers to `crate::log_softmax_rows` (the
  `jll - logsumexp` step kept in `lib.rs`), `nb_predict_proba` is `exp(...)`,
  `nb_predict_joint_log_proba` is the identity over `joint_log_likelihood`.
- **`pub(crate) fn check_alpha`** — the `_check_alpha` floor; re-exported from
  `lib.rs` as `clamp_alpha` so the four discrete variants' `crate::clamp_alpha`
  call sites are unchanged.

**Delegation wiring (the non-test consumers).** Each `Fitted*NB`:
- `impl BaseNB<F>`: `joint_log_likelihood` = the variant's prior private
  `joint_log_likelihood` body, now feature-shape-guarded and `Result`-returning
  (complement: the extracted `complement_scores`); `nb_classes` = `&self.classes`.
- inherent `pub fn predict_proba` → `BaseNB::nb_predict_proba(self, x)`;
  `pub fn predict_log_proba` → `nb_predict_log_proba`;
  `pub fn predict_joint_log_proba` → `nb_predict_joint_log_proba`.
- `impl Predict::predict` → `BaseNB::nb_predict(self, x)`.

`HasClasses` (`classes()`/`n_classes()`) and `PipelineEstimator` wiring are
unchanged. `BernoulliNB` binarizes inside its `joint_log_likelihood` (so the
delegators carry binarization transparently). The refactor is
behavior-preserving: same numerics, same tie-break, same error paths.

**What this base does NOT lift (vs sklearn `_BaseDiscreteNB`):** `coef_` /
`intercept_` (REQ-5), the abstract `_count` / `_update_feature_log_prob` /
shared `partial_fit` seam (REQ-6), and `_update_class_log_prior` / `class_prior`
selection (REQ-7) remain per-variant.

**`conjugate.rs` is UNROUTED.** `posterior_normal_normal` (Normal-Normal
conjugate-prior closed form) has no sklearn analog and is not a naive-Bayes
estimator; it is excluded like `umap.rs` (goal.md "Out of scope"). The
`conjugate.rs` route is re-pointed to `base.rs` (a one-for-one swap; routed-count
unchanged), and no separate `conjugate.rs` route is added.

## Verification

Library crate (green at baseline `e91f4de7` for the existing variant contract;
this iteration is behavior-preserving):
```
cargo test -p ferrolearn-bayes
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```
All existing variant tests (91 lib + 9 doctests) stay green, plus the four new
`base.rs` unit tests (`test_nb_predict_argmax_to_classes`,
`test_nb_predict_tie_breaks_to_smallest_index`,
`test_nb_predict_proba_is_exp_of_log_proba_and_sums_to_one`,
`test_nb_predict_joint_log_proba_is_jll`, `test_check_alpha_*`).

Live sklearn oracle sanity-check (installed 1.5.2 — the base predict pipeline
must match, R-CHAR-3 expected values):
```
python3 -c "import numpy as np; from sklearn.naive_bayes import MultinomialNB; m=MultinomialNB().fit(np.array([[1,2],[0,3],[4,0],[3,1]]), np.array([0,0,1,1])); print(m.predict_log_proba(np.array([[2,2]])).tolist(), m.predict_proba(np.array([[2,2]])).tolist(), m.predict(np.array([[2,2]])).tolist())"
# [[-0.5470675457484475, -0.8642776061017265]] [[0.5786441724102462, 0.4213558275897536]] [0]
```
ferrolearn's MultinomialNB predict/predict_proba/predict_log_proba — now routed
through `BaseNB` — match these values. (Per-variant value REQs come in later
iterations; this is the base-contract sanity check only.)

## Blockers to open

(Director creates the real issues; these are the NOT-STARTED REQs as concrete
prereq blockers.)

- REQ-5 — `_BaseDiscreteNB.coef_` / `intercept_` accessors on the discrete
  fitted types (or a `BaseDiscreteNB` trait): `coef_ = feature_log_prob_[1:]`
  (binary collapses), `intercept_ = class_log_prior_[1:]`.
- REQ-6 — shared `partial_fit` / `_count` / `_update_feature_log_prob` seam:
  lift the per-variant `partial_fit` onto a `BaseDiscreteNB` trait with abstract
  `_count` / `_update_feature_log_prob` (`naive_bayes.py:628-709`, `:542-570`).
- REQ-7 — shared `_update_class_log_prior` / `class_prior` selection: centralize
  the empirical/uniform/explicit prior logic (`naive_bayes.py:580-602`) now
  duplicated across the four discrete variants.
