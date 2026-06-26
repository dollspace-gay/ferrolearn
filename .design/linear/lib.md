# ferrolearn-linear — Crate-Root Re-Export Boundary

<!--
tier: 3-component
status: draft
baseline-commit: 02b52351f7a64ec6665e775d086c300008a44803
upstream-paths:
  - sklearn/linear_model/__init__.py
  - sklearn/base.py
-->

## Summary

This is the crate-root translation unit (`ferrolearn-linear/src/lib.rs`): the public-API
**surface** of `ferrolearn-linear`, not an estimator. It mirrors the re-export boundary of
scikit-learn's `sklearn/linear_model/__init__.py` `__all__` (`__init__.py:48-98`) — but
**broader**: per the workspace's dependency-order grouping ("linear_model, svm,
discriminant_analysis, isotonic", goal.md scope §2), `lib.rs` also surfaces the public types of
`sklearn.svm` and `sklearn.discriminant_analysis` and `sklearn.isotonic` at the same crate root.
The sklearn helpers `enet_path`, `lars_path`, `lars_path_gram`, `lasso_path`, `orthogonal_mp`, `orthogonal_mp_gram`, `ridge_regression`, and `l1_min_c` are also surfaced from this
boundary. Beyond the re-exports, `lib.rs` defines three cross-cutting crate-level helpers: two public score
traits (`ClassifierScore`/`RegressorScore`) backing the `score(&x, &y)` convenience method, and a
`pub(crate)` `log_proba` used as the body of every classifier `predict_log_proba`. The
per-estimator behavior lives in the sibling modules (each separately routed, e.g.
`.design/linear/linear_regression.md`); this doc is the contract for the **boundary** and the
helpers only.

## Probes (live sklearn 1.5.2 oracle)

The re-export boundary mirrored (`__init__.py:48-98`, the `__all__` list):

```bash
python3 -c "import sklearn.linear_model as m; print(sorted(m.__all__))"
```

The two `score` delegations this crate reimplements (`sklearn/base.py`):

```bash
# ClassifierMixin.score (base.py:738-764) -> accuracy_score (mean accuracy)
# RegressorMixin.score (base.py:805-849) -> r2_score (R^2 = 1 - SSres/SStot)
```

`r2_score` constant-`y_true` edge oracle (`sklearn.metrics.r2_score`, the function
`RegressorMixin.score` delegates to at `base.py:847`):

```bash
python3 -c "import warnings, numpy as np; from sklearn.metrics import r2_score
with warnings.catch_warnings():
    warnings.simplefilter('always')
    print('const y, resid!=0:', r2_score([5.,5.,5.],[4.,5.,6.]))   # -> 0.0  (+UndefinedMetricWarning)
    print('const y, resid==0:', r2_score([5.,5.,5.],[5.,5.,5.]))   # -> 1.0
    print('in-regime:        ', r2_score([3.,5.,2.,7.],[2.5,5.,2.,8.]))  # -> 0.9152542372881356"
```

Observed oracle values: constant `y_true` with non-zero residual → **0.0** (with
`UndefinedMetricWarning`); constant `y_true` with zero residual → **1.0**; an in-regime case →
`0.9152542372881356`.

## Requirements

- REQ-1: The crate root re-exports every linear/SVM/discriminant/isotonic estimator ferrolearn
  implements as the crate's public API, mirroring sklearn's `__all__` re-export boundary
  (`sklearn/linear_model/__init__.py:48-98`), broadened to also surface `sklearn.svm`,
  `sklearn.discriminant_analysis`, and `sklearn.isotonic` public types plus `sklearn.svm.l1_min_c`
  (goal.md scope §2 grouping).
- REQ-2: `ClassifierScore::score(x, y)` returns mean accuracy of `predict(x)` vs `y`, mirroring
  `ClassifierMixin.score` → `accuracy_score` (`base.py:738-764`, body `base.py:764`).
- REQ-3: `RegressorScore::score(x, y)` returns the in-regime R² coefficient of determination
  `1 − SSres/SStot` of `predict(x)` vs `y`, mirroring `RegressorMixin.score` → `r2_score`
  (`base.py:805-849`, body `base.py:847-848`; R² definition `base.py:807-816`).
- REQ-4: `RegressorScore::score`'s constant-`y_true` edge matches `sklearn.metrics.r2_score`:
  `SStot == 0 ∧ SSres ≠ 0` → `0.0`; `SStot == 0 ∧ SSres == 0` → `1.0` (oracle values above;
  the function `base.py:847` delegates to).
- REQ-5: `log_proba` computes the element-wise `ln` of a probability matrix (clamping tiny
  values to avoid `-inf`), used as the body of every classifier `predict_log_proba` in the crate,
  mirroring sklearn `predict_log_proba = np.log(predict_proba)` (e.g.
  `discriminant_analysis.py:1058-1059`).
- REQ-6: The `score` methods accept per-sample `sample_weight`, matching sklearn's
  `score(self, X, y, sample_weight=None)` signature on both mixins (`base.py:738`, `base.py:805`).
- REQ-substrate: The crate-level helpers and score traits run on the ferray substrate
  (`ferray-core` arrays / `ferray::linalg`) rather than `ndarray::Array1`/`Array2` +
  `num_traits::Float` (goal.md R-SUBSTRATE-1).

## Acceptance criteria

- AC-1 (REQ-1): The set of estimator types named in the `pub use` block at the crate root is a
  superset of sklearn `linear_model.__all__` minus the types ferrolearn lacks (see Architecture),
  plus the `sklearn.svm`/`discriminant_analysis`/`isotonic` public types and `l1_min_c`; each is reachable as
  `ferrolearn_linear::<Type>` and routed by its own `.design/linear/<doc>.md`.
- AC-2 (REQ-2): For a fitted classifier with integer-label predictions, `score(x, y)` equals
  `(# predicted == true) / n`, matching `sklearn …ClassifierMixin.score` (= `accuracy_score`)
  to within float rounding.
- AC-3 (REQ-3): For an in-regime (non-degenerate `y`) regressor, `score(x, y)` matches
  `sklearn …RegressorMixin.score` (= `metrics.r2_score`) to within `1e-8` (oracle in-regime case
  `0.9152542372881356`).
- AC-4 (REQ-4): `score` on constant `y` returns `0.0` when residual ≠ 0 and `1.0` when residual
  = 0 (oracle values above).
- AC-5 (REQ-5): `predict_log_proba(X)` of a fitted classifier equals `predict_proba(X).ln()`
  element-wise (clamped), matching live sklearn `predict_log_proba`.
- AC-6 (REQ-6): `score(x, y, sample_weight)` reproduces sklearn's weighted accuracy / weighted R².

## REQ status

Consumer basis for the score traits (REQ-2/REQ-3/REQ-4): `ClassifierScore`/`RegressorScore`
are **pre-existing** crate-root `pub trait`s (not added this commit), re-exported through the
meta-crate (`ferrolearn/src/lib.rs`, `pub use ferrolearn_linear as linear;`). Per goal.md **S5**
("R-DEFER-1 binds on NEWLY-ADDED pub APIs only; existing pub API surface is grandfathered;
boundary types ARE the public API"), the grandfathered re-export is the sanctioned consumer —
the same basis REQ-1 ships on. Honest underclaim (R-HONEST-3): no production code *calls*
`.score()` yet (only `tests/api_proof.rs` + the divergence pins); surfacing `.score` through the
`ferrolearn-python` shim is future work, and `sample_weight` is unsupported (REQ-6).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (re-export boundary) | SHIPPED | impl: the `pub use` block `in lib.rs` surfaces every implemented estimator at the crate root — `LinearRegression`, `Ridge`/`RidgeCV`/`RidgeClassifier`, `Lasso`/`LassoCV`, `ElasticNet`/`ElasticNetCV`, `BayesianRidge`, `ARDRegression`, `HuberRegressor`, `QuantileRegressor`, `Lars`/`LassoLars`, `OrthogonalMatchingPursuit`/`OrthogonalMatchingPursuitCV`, `RANSACRegressor`, `LogisticRegression`/`LogisticRegressionCV`, `LinearSVC`/`LinearSVR`, `SVC`/`SVR`/`NuSVC`/`NuSVR`/`OneClassSVM`, `SGDClassifier`/`SGDRegressor`/`SGDOneClassSVM`, `LDA`/`QDA`, `IsotonicRegression`, the GLM family (`GLMRegressor`/`PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`), and the helpers `enet_path`, `lars_path`, `lars_path_gram`, `lasso_path`, `orthogonal_mp`, `orthogonal_mp_gram`, `ridge_regression`, and `l1_min_c`/`L1MinCLoss` — mirroring sklearn `linear_model.__all__` (`__init__.py:48-98`) and the grouped sklearn.svm boundary, broadened per goal.md scope §2. Non-test consumers: the meta-crate `ferrolearn/src/lib.rs` (`pub use ferrolearn_linear as linear;`) and the PyO3 shim `ferrolearn-python/src/{regressors,classifiers,extras}.rs` (each imports concrete estimator types from `ferrolearn_linear`); helpers are covered by `tests/api_proof.rs`, `tests/divergence_enet_path.rs`, `tests/divergence_lars_path.rs`, `tests/divergence_lars_path_gram.rs`, `tests/divergence_lasso_path.rs`, `tests/divergence_omp_cv.rs`, `tests/divergence_omp_default.rs`, `tests/divergence_ridge_numeric.rs`, and `tests/divergence_svm_bounds.rs`. Boundary estimator types are grandfathered public API (goal.md S5/R-DEFER-1). Verification: `cargo build -p ferrolearn` (green); per-estimator/helper routes in `.design/linear/*.md`. |
| REQ-2 (ClassifierScore == mean accuracy) | SHIPPED | impl `pub trait ClassifierScore` + blanket impl over `Predict<Array2<F>, Output=Array1<usize>>` `in lib.rs`, body `mean_accuracy in lib.rs` (`correct / n`) mirrors `ClassifierMixin.score` → `accuracy_score` (`base.py:738-764`). Critic-verified clean vs the live oracle (`accuracy_score([0,1,2,1],[0,1,1,1]) = 0.75 = correct/n`). Consumer: grandfathered crate/meta-crate re-export of the pre-existing `pub trait` (goal.md S5). Verification: `cargo test -p ferrolearn-linear` green; `tests/api_proof.rs` exercises `.score`. Underclaim: no production `.score()` caller yet; multilabel "subset accuracy" N/A (`Output=Array1<usize>`, single-label). |
| REQ-3 (RegressorScore == in-regime R²) | SHIPPED | impl `pub trait RegressorScore` + blanket impl over `Predict<Array2<F>, Output=Array1<F>>` `in lib.rs`, body `r2_score in lib.rs` computes `1 − ss_res/ss_tot` mirroring `RegressorMixin.score` → `metrics.r2_score` (`base.py:805-849`; R² def `base.py:807-816`). Critic-verified clean: matches the live oracle `r2_score([3.,5.,2.,7.],[2.5,5.,2.,8.]) = 0.9152542372881356` within 1e-8 (`r2_in_regime_matches_oracle`, `tests/divergence_lib.rs`). Consumer: grandfathered re-export (S5). Underclaim: no production `.score()` caller yet. |
| REQ-4 (constant-y edge parity) | SHIPPED | FIXED #1104. `r2_score in lib.rs` now returns `F::zero()` (was `F::neg_infinity()`) when `ss_tot == 0 ∧ ss_res != 0`, matching `sklearn.metrics.r2_score` (`_regression.py:891`: `output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0`, the `RegressorMixin.score` delegate `base.py:849`). The zero-residual sub-case stays `1.0`. Live oracle: `r2_score([5.,5.,5.],[4.,5.,6.])=0.0`, `r2_score([5.,5.,5.],[5.,5.,5.])=1.0`. Verification (both green): `divergence_r2_constant_ytrue_nonzero_residual_returns_zero` + `r2_constant_ytrue_zero_residual_returns_one` in `tests/divergence_lib.rs`. |
| REQ-5 (log_proba behind predict_log_proba) | SHIPPED | FIXED #1105. `pub(crate) fn log_proba in lib.rs` is now the UNCLAMPED element-wise `p.ln()` (was clamping `p ≤ 1e-300` to `ln(1e-300) ≈ -690.78`), matching sklearn `predict_log_proba = np.log(predict_proba)` (`discriminant_analysis.py:1059`: `return np.log(probas_)`) — a `0.0` probability now maps to `-inf`. Non-test consumers: `FittedLogisticRegression::predict_log_proba in logistic_regression.rs`, `FittedLogisticRegressionCV::predict_log_proba in logistic_regression_cv.rs`, `FittedQDA::predict_log_proba in qda.rs` (`Ok(crate::log_proba(&proba))`). Verification (green): `divergence_log_proba_zero_clamps_instead_of_neg_inf` (QDA zero-proba → `[[-inf,0.0]]`) + `cargo test -p ferrolearn-linear`. (Note: `lda.rs` deliberately does **not** use this helper — it floors zeros with `smallest_normal`, see `.design/linear/lda.md`.) |
| REQ-6 (sample_weight on score) | NOT-STARTED | open prereq blocker #1106. Both blanket impls take only `(&self, x, y)`; neither `mean_accuracy` nor `r2_score in lib.rs` accepts per-sample weights. sklearn's `score(self, X, y, sample_weight=None)` (`base.py:738`, `base.py:805`) forwards `sample_weight` into `accuracy_score`/`r2_score`, so weighted scoring is unsupported here. |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker #1107. `lib.rs` imports `ndarray::{Array1, Array2}` and `num_traits::Float`; the score traits, `mean_accuracy`, `r2_score`, and `log_proba` all operate on `ndarray` arrays. Migrating the helper signatures + the trait `Output` bounds onto `ferray-core` arrays (goal.md R-SUBSTRATE-1) cascades through every fitted estimator's `Predict::Output`. |

## Architecture

**This is a boundary, not an estimator.** `lib.rs` has three responsibilities, in source order:

1. **Module declarations + re-export block (REQ-1).** `pub mod` for every estimator module
   (`lib.rs:43-71`; `linalg` and `optim` are private internals — `mod linalg;`, `mod optim;`),
   followed by the `pub use` re-export block (`lib.rs:74-108`) that hoists each estimator's
   unfitted + fitted types (and supporting types like `Kernel`/`RbfKernel`, `LinearSVCLoss`,
   `GLMFamily`) to the crate root. This mirrors the *function* of sklearn's
   `linear_model/__init__.py` `__all__` (`__init__.py:48-98`): defining the importable public
   surface. **Breadth divergence (intended, per goal.md scope §2):** ferrolearn groups four
   sklearn modules under one crate, so `lib.rs` *adds* the public types of `sklearn.svm`
   (`SVC`/`SVR`/`NuSVC`/`NuSVR`/`OneClassSVM`/`LinearSVC`/`LinearSVR` + kernels + `l1_min_c`) and
   `sklearn.discriminant_analysis` (`LDA`/`QDA`) and `sklearn.isotonic` (`IsotonicRegression`),
   which are *not* in `linear_model.__all__`.
   **Omissions (estimators ferrolearn lacks — documented, not a `lib.rs` gap):** relative to
   `linear_model.__all__`, the crate root omits `MultiTaskElasticNet[CV]`, `MultiTaskLasso[CV]`,
   `Perceptron`, `PassiveAggressiveClassifier`/`Regressor`, `TheilSenRegressor`, the cross-validated
   `LarsCV`/`LassoLarsCV`/`LassoLarsIC`/`RidgeClassifierCV`, the loss
   classes (`Hinge`/`Log`/`ModifiedHuber`/`SquaredLoss`/`Huber`). Those are missing-estimator gaps owned by
   (future) per-estimator routes, not boundary gaps; `lib.rs` correctly re-exports exactly what
   exists. The non-test consumers of the boundary are the meta-crate
   (`ferrolearn/src/lib.rs:26`, `pub use ferrolearn_linear as linear`) and the PyO3 shim crate.

2. **Score convenience traits (REQ-2/REQ-3/REQ-4/REQ-6).** `ClassifierScore<F>` (`lib.rs:121-147`)
   and `RegressorScore<F>` (`lib.rs:154-180`) each have a blanket impl over the corresponding
   `Predict` shape, giving every fitted classifier / regressor a `score(&x, &y)` method without
   per-estimator boilerplate. They are the structural analog of sklearn's `ClassifierMixin.score`
   / `RegressorMixin.score` (`base.py:738`, `base.py:805`), which inject `score` via mixin
   inheritance. The classifier impl guards `x.nrows() == y.len()` (`ShapeMismatch`), calls
   `predict`, and returns `mean_accuracy`. The regressor impl is identical in shape but returns
   `r2_score`. Because `Output = Array1<usize>` for the classifier, the multilabel
   "subset accuracy" remark in sklearn's docstring (`base.py:741-743`) is N/A here — predictions
   are single-label integer class indices. Status: the constant-`y` R² edge (REQ-4, #1104) and
   the `log_proba` clamp (REQ-5, #1105) were FIXED this iteration; the traits ship on the
   grandfathered-re-export basis (goal.md S5, see the REQ-status preamble). Remaining gap: no
   `sample_weight` parameter (REQ-6, #1106) — and no production `.score()` caller yet (honest
   underclaim; future PyO3 surfacing).

3. **Crate helpers.** `mean_accuracy` (`lib.rs:186-197`, `correct / n`, empty → `0.0`), `r2_score`
   (`lib.rs:203-226`), and `log_proba` (`lib.rs:231-234`) are `pub(crate)` bodies shared by the
   above traits and by the classifier modules. `log_proba` is the one helper with a clean
   non-test consumer chain (`logistic_regression.rs:581`, `logistic_regression_cv.rs:190`,
   `qda.rs:397`) → REQ-5 SHIPPED. `mean_accuracy`/`r2_score` are reached *only* through the
   (test-only-consumed) score traits, which is why REQ-2/REQ-3 are NOT-STARTED.

**Note on namesakes (not consumers of these helpers).** `ransac.rs` and `sgd.rs` each define their
*own* private `r2_score` (`ransac.rs:224`, `sgd.rs:578`) with their own signatures and edge
semantics (used for RANSAC subset selection and SGD early-stopping validation, respectively); they
do **not** call `crate::r2_score`. Those are covered by `.design/linear/ransac.md` and
`.design/linear/sgd.md`, not this unit.

## Verification

Commands establishing the SHIPPED claims:

- `cargo build -p ferrolearn` — the meta-crate compiles against the re-export boundary (REQ-1),
  exercising `pub use ferrolearn_linear as linear`.
- `cargo test -p ferrolearn-linear` — 470/470 lib tests pass at baseline, including
  `qda::tests::qda_predict_log_proba` and the logistic `predict_log_proba` tests that exercise the
  `log_proba` helper (REQ-5).
- Re-export surface (REQ-1): every type in the `pub use` block (`lib.rs:74-108`) is named by its
  own routed doc under `.design/linear/`.
- `log_proba` oracle (REQ-5):
  `python3 -c "import numpy as np; from sklearn.linear_model import LogisticRegression; X=np.array([[0.],[1.],[2.],[3.]]); y=np.array([0,0,1,1]); m=LogisticRegression().fit(X,y); print(np.allclose(m.predict_log_proba(X), np.log(m.predict_proba(X))))"`
  → `True` (sklearn's own `predict_log_proba` is `log(predict_proba)`, the relation `log_proba` mirrors).
- Constant-`y` R² divergence reference (REQ-4 — a NOT-STARTED divergence, not a parity claim):
  `python3 -c "from sklearn.metrics import r2_score; print(r2_score([5.,5.,5.],[4.,5.,6.]), r2_score([5.,5.,5.],[5.,5.,5.]))"`
  → `0.0 1.0`; ferrolearn `r2_score in lib.rs` returns `neg_infinity` for the first case.

REQ-1..REQ-5 are SHIPPED (REQ-4/REQ-5 fixed this iteration, pinned green in
`tests/divergence_lib.rs`; REQ-2/REQ-3 on the grandfathered-re-export basis, goal.md S5). REQ-6
(sample_weight, #1106) and REQ-substrate (#1107) are NOT-STARTED; their acceptance criteria
(AC-6) have no green verification until those blockers land.

## Blockers

Fixed this iteration (divergences pinned in `tests/divergence_lib.rs`, fix landed, tests green):

- **#1104 (REQ-4) — FIXED.** `r2_score in lib.rs` returned `F::neg_infinity()` on constant `y_true`
  with non-zero residual; `sklearn.metrics.r2_score` (the `RegressorMixin.score` delegate,
  `base.py:849`; `_regression.py:891`) returns `0.0`. The `ss_tot == 0 ∧ ss_res != 0` branch now
  returns `F::zero()`.
- **#1105 (REQ-5) — FIXED.** `log_proba in lib.rs` clamped `p ≤ 1e-300` to `ln(1e-300) ≈ -690.78`;
  sklearn `predict_log_proba = np.log(predict_proba)` (`discriminant_analysis.py:1059`) is unclamped
  (`log(0) = -inf`). The clamp is removed; `log_proba` is now `proba.mapv(|p| p.ln())`.

Open (NOT-STARTED):

- **#1106 (REQ-6):** neither score trait accepts `sample_weight`; sklearn's
  `score(self, X, y, sample_weight=None)` (`base.py:738`, `base.py:805`) does. Needs a weighted
  path through `mean_accuracy`/`r2_score`.
- **#1107 (REQ-substrate):** the score traits and helpers are on `ndarray` + `num_traits`, not the
  ferray substrate (goal.md R-SUBSTRATE-1).
