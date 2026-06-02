# Linear Discriminant Analysis

<!--
tier: 3-component
status: draft
baseline-commit: 83e73bf4
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/discriminant_analysis.py
ferrolearn-module: ferrolearn-linear/src/lda.rs
parity-op: LinearDiscriminantAnalysis
crosslink-issue: 587
-->

## Summary

`ferrolearn-linear/src/lda.rs` is intended to mirror scikit-learn's
`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
(`discriminant_analysis.py:175-759`): a classifier with a linear decision
boundary (Gaussian densities with a **shared** covariance + Bayes' rule) that
doubles as a supervised dimensionality-reduction transformer. sklearn's default
`solver="svd"` whitens the within-class data via an SVD, derives
`scalings_`/`xbar_`, then forms `coef_`/`intercept_` (which embed `log priors_`)
so that `decision_function = X @ coef_.T + intercept_`, `predict` = argmax of
that, and `predict_proba` = softmax (binary: `expit`) of it.

**Status (post-rewrite, #587).** `lda.rs` now transcribes sklearn's default
`solver="svd"` path. `fn fit in lda.rs` computes empirical `priors_`, per-class
`means_`, `xbar_ = priors_@means_`, the within-std-whitening SVD
(`scalings = (Vt[:rank]/std).T / S[:rank]`), the between-class projection SVD
(`scalings_ = scalings @ Vt2.T[:,:rank2]`), and the affine `coef_`/`intercept_`
(`intercept_` embedding `log(priors_)`), all via `ferray::linalg::svd`
(`fn svd_s_vt`). `transform = ((X - xbar_) @ scalings_)[:, :max_components]`;
`decision_function = X @ coef_.T + intercept_`; `predict` = its argmax;
`predict_proba` = its softmax. The prior `Sw⁻¹Sb` Jacobi/Gaussian-elimination
nearest-centroid code is removed. REQ-1/2/3/5/7(empirical)/8/13 are SHIPPED
against the live oracle (`divergence_lda_fit.rs` + the `lda.rs` oracle pins). The
remaining NOT-STARTED REQs (provided-priors #593, lsqr #595, eigen #596,
shrinkage #597, store_covariance #598, binary `(n,)` shape #600, `tol` param
#601, owned-array ferray migration #602, predict_log_proba `smallest_normal`
floor #591) are unchanged.

(Historical, pre-#587:) ferrolearn used the classical generalized-eigenvalue
formulation — within-class scatter `Sw` (un-averaged, `1e-6` ridge), between
scatter `Sb`, `Sw⁻¹Sb` via Jacobi over Gaussian-elimination inverse, raw
`X @ scalings` transform, and a nearest-centroid / equal-priors
`-½‖z-μ_c‖²` decision. That whole path is gone.

**Crux finding (verified against the live oracle).** The one quantity that
matches is `explained_variance_ratio_`: the eigenvalue ratio `λ_k/Σλ` from
ferrolearn's `Sw⁻¹Sb` formulation numerically equals sklearn's SVD-solver
`S²/ΣS²` (iris: both `[0.991213, 0.008787]`). Everything else diverges in
**value**:

- `decision_function` is a different function entirely — sklearn's affine
  `X@coef_.T + intercept_` (which carries `log priors_`) vs ferrolearn's
  projected-space `-½‖z-μ_c‖²`. Even with **uniform** priors the per-sample
  values differ (sklearn iris row 0 = `[31.34, -17.96, -64.41]`); only the
  argmax happens to agree when priors are uniform.
- `transform` values diverge: sklearn `(X-xbar_)@scalings_` with whitened
  scalings (iris row 0 = `[8.06, -0.30]`) vs ferrolearn raw `X@scalings`
  (`[-1.50, -1.89]`).
- With **imbalanced/empirical or provided** priors the missing `log(prior_k)`
  term shifts the boundary: on a 40-vs-4 class split (priors `[0.909, 0.091]`)
  sklearn's `coef_/intercept_` push the boundary toward the minority class while
  ferrolearn's equal-prior nearest-centroid ignores it.
- The binary `decision_function` shape is `(n,)` in sklearn (the log-likelihood
  ratio `log p(y=1|x)-log p(y=0|x)`), but ferrolearn always returns
  `(n, n_classes)`.

ferrolearn has **no** `coef_`/`intercept_`/`xbar_`/`priors_`/`covariance_`
attributes, **no** constructor params besides `n_components` (no `solver`,
`shrinkage`, `priors`, `store_covariance`, `tol`), and **only** the SVD-analog
single solver — `lsqr`/`eigen` solvers and shrinkage are absent. The estimator
is re-exported at the crate root (`pub use lda::{FittedLDA, LDA} in lib.rs`) but
is **not** registered in the `ferrolearn-python` binding, so its inference
methods have no non-test production consumer. Substrate is `ndarray` throughout
(ferray NOT-STARTED).

Under R-DEFER-2 every REQ below is binary. The single SHIPPED row (REQ-6,
`n_components` bound) is the one piece whose behavior is structurally checkable,
present, consumed, and tested; even `explained_variance_ratio_`, though it
numerically matches the oracle, is NOT-STARTED because it has no R-CHAR-3
live-oracle pin and the underlying `transform`/`scalings_` it derives from
diverge.

## Algorithm (sklearn — the contract)

`LinearDiscriminantAnalysis(solver="svd", shrinkage=None, priors=None,
n_components=None, store_covariance=False, tol=1e-4, covariance_estimator=None)`
(`discriminant_analysis.py:347-363`); `_parameter_constraints` at `:337-345`
(`solver ∈ {svd, lsqr, eigen}`, `shrinkage ∈ {"auto", [0,1] float, None}`,
`n_components ≥ 1 or None`, `priors` array-like or None, `tol ≥ 0`).

### fit — solver dispatch (`discriminant_analysis.py:565-659`)
1. Validate `ensure_min_samples=2`; `classes_ = unique_labels(y)`; reject
   `n_samples == n_classes` (`:589-599`).
2. **priors_** (`:601-612`): `priors is None` → empirical `cnts/n` per class;
   else `np.asarray(priors)`. Reject negative; renormalize (with a `UserWarning`)
   if `|Σ-1| > 1e-5`.
3. `max_components = min(n_classes-1, n_features)`; `_max_components` =
   `n_components` (error if `> max_components`) or the max (`:614-625`).
4. Dispatch on `solver` (`:627-650`): `svd` (default; `shrinkage`/
   `covariance_estimator` raise), `lsqr` (`_solve_lstsq`), `eigen`
   (`_solve_eigen`).
5. **Binary special-case** (`:651-657`): if 2 classes, collapse to
   `coef_ = coef_[1]-coef_[0]` (shape `(1, n_features)`) and
   `intercept_ = intercept_[1]-intercept_[0]` (shape `(1,)`).

### _solve_svd (`discriminant_analysis.py:487-559`) — the default
- `means_ = _class_means(X, y)` (`:508`); `xbar_ = priors_ @ means_` (`:517`).
- Center per class, stack `Xc`; within-std `std` (`:519-524`).
- `X = sqrt(1/(n-K)) * (Xc/std)`; `U,S,Vt = svd(X)` (`:528-530`).
- `rank = Σ(S > tol)`; `scalings = (Vt[:rank]/std).T / S[:rank]` — the within
  whitening (`:532-534`).
- Between scaling: weight centered class means by `sqrt(n·priors_·1/(K-1))`,
  project onto `scalings`, second SVD (`:535-545`).
- `explained_variance_ratio_ = (S²/ΣS²)[:_max_components]` (`:550-552`).
- `scalings_ = scalings @ Vt.T[:, :rank]` (`:555`).
- `coef = (means_ - xbar_) @ scalings_`;
  `intercept_ = -0.5·Σ(coef²) + log(priors_)`;
  `coef_ = coef @ scalings_.T`; `intercept_ -= xbar_ @ coef_.T` (`:556-559`).

### _solve_eigen (`discriminant_analysis.py:421-485`)
`covariance_ = _class_cov(X,y,priors_,shrinkage)` (the **prior-weighted, biased**
within covariance); `St = _cov(X,shrinkage)`; `Sb = St - Sw`; generalized
`eigh(Sb, Sw)`; `explained_variance_ratio_ = sort(evals/Σevals)[::-1]`;
`scalings_ = evecs`; `coef_ = (means_ @ evecs) @ evecs.T`;
`intercept_ = -0.5·diag(means_ @ coef_.T) + log(priors_)`. Supports shrinkage.

### _solve_lstsq (`discriminant_analysis.py:365-419`)
`covariance_ = _class_cov(...)`; `coef_ = lstsq(covariance_, means_.T)[0].T`;
`intercept_ = -0.5·diag(means_ @ coef_.T) + log(priors_)`. **No** `transform`
(raises `NotImplementedError`, `:676-679`). Supports shrinkage.

### transform (`discriminant_analysis.py:661-689`)
`svd` → `(X - xbar_) @ scalings_`; `eigen` → `X @ scalings_`; sliced to
`[:, :_max_components]`. `lsqr` raises.

### decision_function (`LinearClassifierMixin`, referenced `:739-759`)
`X @ coef_.T + intercept_`; binary → `(n,)` (positive-class log-likelihood
ratio), multiclass → `(n, n_classes)`.

### predict / predict_proba / predict_log_proba (`:691-738`)
`predict` = `classes_.take(argmax(decision_function, axis=1))` (mixin).
`predict_proba`: binary → `stack([1-expit(d), expit(d)])`; multiclass →
`softmax(decision_function)` (`:706-711`). `predict_log_proba` = `log` of that
with a `smallest_normal` floor on exact zeros (`:726-737`).

## ferrolearn (what exists)

`LDA<F> { n_components: Option<usize> }` (`lda.rs`), `LDA::new(Option<usize>)`,
`n_components` getter, `Default` (= `None`). The `Fit<Array2<F>, Array1<usize>>`
impl (`impl Fit for LDA in lda.rs`) checks `y` length / `n_samples ≥ 2` /
`n_classes ≥ 2`, validates `n_components ∈ [1, min(n_classes-1, n_features)]`,
computes the overall mean + per-class means, the **un-averaged** within scatter
`Sw = Σ_c Σ_{x∈c} (x-μ_c)(x-μ_c)ᵀ` with a `1e-6` diagonal ridge, the between
scatter `Sb = Σ_c n_c (μ_c-μ)(μ_c-μ)ᵀ`, then `M = Sw⁻¹Sb` (`fn sw_inv_sb` via
`fn gaussian_solve_f`), a Jacobi eigendecomposition (`fn jacobi_eigen_f`),
descending-eigenvalue sort, `explained_variance_ratio = clamp(λ_k,0)/Σclamp(λ,0)`,
`scalings` = top-`k` eigenvectors, and projected class `means = class_means @
scalings`.

`FittedLDA<F> { scalings, means, explained_variance_ratio, classes, n_features }`
(`lda.rs`) exposes `scalings`/`means`/`explained_variance_ratio`/`classes`
accessors. `Transform` (`fn transform in lda.rs`) returns raw `x.dot(&scalings)`
(no centering). `Predict` (`fn predict in lda.rs`) is nearest-centroid in the
projected space. `fn decision_function in lda.rs` returns `(n, n_classes)` of
`-½‖z-μ_c‖²`; `fn predict_proba` is the softmax of those; `fn predict_log_proba`
= `crate::log_proba(&predict_proba)`. `LDA` also implements `PipelineEstimator`.
Re-exported as `pub use lda::{FittedLDA, LDA} in lib.rs`; **not** present in
`ferrolearn-python`.

## Requirements

- REQ-1: **svd-solver fit + decision_function parity** — the default
  `solver="svd"` produces `coef_`/`intercept_`/`xbar_`/`scalings_` such that
  `decision_function(X) = X @ coef_.T + intercept_` matches the live
  `LinearDiscriminantAnalysis().fit(X,y).decision_function(X)` oracle, including
  the `log(priors_)` term (`discriminant_analysis.py:487-559, 739-759`).
- REQ-2: **predict (argmax)** — `predict` = `classes_.take(argmax of the
  affine decision_function)`, matching the oracle's labels including on
  **imbalanced** classes where the prior shifts the boundary
  (`discriminant_analysis.py:691-711` via the mixin).
- REQ-3: **predict_proba** — binary → `[1-expit(d), expit(d)]`, multiclass →
  `softmax(decision_function)`, over the **prior-aware** affine decision, rows
  sum to 1, matching the oracle (`discriminant_analysis.py:691-711`).
- REQ-4: **predict_log_proba** — `log(predict_proba)` with sklearn's
  `smallest_normal` zero-floor (`discriminant_analysis.py:713-737`).
- REQ-5: **transform (projection) parity** — `(X - xbar_) @ scalings_` sliced to
  `_max_components`, with the SVD-whitened `scalings_`, matching the oracle
  `transform(X)` (`discriminant_analysis.py:684-689, 532-555`).
- REQ-6: **n_components bound** — `n_components ≤ min(n_classes-1, n_features)`,
  `None` → the max; `n_components` out of range → error
  (`discriminant_analysis.py:614-625`).
- REQ-7: **priors (None=empirical + provided)** — `priors=None` → empirical
  `cnts/n`; a provided array used (renormalized with warning if `|Σ-1|>1e-5`,
  rejected if negative), and `priors_` exposed (`discriminant_analysis.py:601-612`).
- REQ-8: **fitted attributes coef_/intercept_/xbar_** — expose
  `coef_`/`intercept_`/`xbar_` matching the oracle arrays
  (`discriminant_analysis.py:556-559, 517`).
- REQ-9: **lsqr solver** — the `solver="lsqr"` least-squares path
  (`coef_ = lstsq(covariance_, means_.T)[0].T`), `transform` raising
  `NotImplementedError` (`discriminant_analysis.py:365-419, 676-679`).
- REQ-10: **eigen solver** — the `solver="eigen"` generalized-eigenvalue path
  (`eigh(Sb, Sw)`, `scalings_ = evecs`, `coef_`/`intercept_` from means+priors)
  (`discriminant_analysis.py:421-485`).
- REQ-11: **shrinkage (None/auto/float)** — the `shrinkage` parameter
  (None / 'auto' Ledoit-Wolf / float) on `lsqr`/`eigen`, rejected on `svd`
  (`discriminant_analysis.py:339, 628-629`, `_cov`/`_class_cov`).
- REQ-12: **store_covariance / covariance_** — the `store_covariance` flag and
  the weighted within `covariance_ = Σ_k prior_k·C_k` attribute
  (`discriminant_analysis.py:280-285, 509-510`).
- REQ-13: **explained_variance_ratio_** — `(S²/ΣS²)[:_max_components]` matching
  the oracle (`discriminant_analysis.py:550-552`).
- REQ-14: **decision_function shape/sign (binding ABI)** — binary → `(n,)`
  (log-likelihood ratio of the positive class), multiclass → `(n, n_classes)`
  (`discriminant_analysis.py:651-657, 739-759`). cf. #581/#454.
- REQ-15: **tol** — the SVD-solver rank threshold `Σ(S > tol)` and
  `Σ(S > tol·S[0])` (`discriminant_analysis.py:354, 532, 554`).
- REQ-16: **ferray substrate migration** — `lda.rs` owns its computation on
  `ferray-core` arrays / `ferray::linalg` (SVD, eig, solve), not `ndarray` +
  hand-rolled Jacobi/Gaussian elimination (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): on iris (3-class, balanced), ferrolearn's `decision_function(X)`
  matches `L().fit(X,y).decision_function(X)` (the affine `X@coef_.T+intercept_`,
  e.g. row 0 = `[31.34, -17.96, -64.41]`) within `1e-8`. **Currently diverges**:
  ferrolearn returns `-½‖z-μ_c‖²` in projected space, a different function.
- AC-2 (REQ-2): on a 40-vs-4 imbalanced 2-class set (priors `[0.909, 0.091]`),
  `predict` of a borderline point matches `L().predict` (which uses
  `log(priors_)`). **Currently the equal-prior nearest-centroid can disagree.**
- AC-3 (REQ-3): `predict_proba(X)` matches `L().predict_proba(X)` within `1e-8`
  (binary `expit` path + multiclass softmax of the affine decision); rows sum
  to 1. **Currently diverges (softmax of the wrong decision).**
- AC-4 (REQ-4): `predict_log_proba(X)` matches `L().predict_log_proba(X)` within
  `1e-8` where proba > 0.
- AC-5 (REQ-5): on iris, `transform(X)` matches `L().transform(X)` (row 0 ≈
  `[8.0618, -0.3004]`) within `1e-6` up to per-column sign. **Currently
  diverges**: ferrolearn returns `[-1.50, -1.89]` (raw, un-centered, eigvecs not
  std-whitened).
- AC-6 (REQ-6): `LDA::new(Some(k))` with `k > min(n_classes-1, n_features)`
  errors; `None` → `k = min(n_classes-1, n_features)`; `Some(0)` errors.
- AC-7 (REQ-7): `L(priors=[0.9,0.1]).fit(X,y).priors_ == [0.9,0.1]`; `None` →
  empirical; ferrolearn exposes `priors_` and its predictions reflect it.
- AC-8 (REQ-8): `coef_`/`intercept_`/`xbar_` equal the oracle arrays (binary:
  `coef_` shape `(1, n_features)`, `intercept_` shape `(1,)`).
- AC-9 (REQ-9): `L(solver="lsqr").fit(X,y).coef_` matches the oracle;
  `transform` raises.
- AC-10 (REQ-10): `L(solver="eigen").fit(X,y)` `coef_`/`scalings_`/
  `explained_variance_ratio_` match the oracle.
- AC-11 (REQ-11): `L(solver="eigen", shrinkage=0.5)` and `shrinkage="auto"`
  match the oracle; `L(solver="svd", shrinkage=0.5)` raises.
- AC-12 (REQ-12): `L(store_covariance=True).fit(X,y).covariance_` matches the
  weighted within covariance.
- AC-13 (REQ-13): `explained_variance_ratio_` matches `L().fit(X,y).
  explained_variance_ratio_` within `1e-8` (iris `[0.991213, 0.008787]`).
- AC-14 (REQ-14): binary `decision_function(X).shape == (n,)`; multiclass
  `(n, n_classes)`; the binding reproduces the binary `(n,)` shape.
- AC-15 (REQ-15): a singular value below `tol` is dropped from the rank; varying
  `tol` changes `scalings_`/`transform` output dimensionality.
- AC-16 (REQ-16): `lda.rs` operates on `ferray-core` arrays and
  `ferray::linalg`, not `ndarray` + the hand-rolled Jacobi/Gaussian helpers.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ a sklearn-grounded test (R-CHAR-3) + green oracle verification; NOT-STARTED =
concrete open blocker referenced by `#`-number. `LDA`/`FittedLDA` are boundary
estimator types re-exported at the crate root (`pub use lda::{FittedLDA, LDA} in
lib.rs`); under S5/R-DEFER-1 the public estimator type IS the consumer surface
(grandfathered). However, unlike the sibling estimators, **LDA is NOT registered
in `ferrolearn-python`** (no `RsLDA`; `grep LDA ferrolearn-python/src` is empty),
so the only callers of `predict`/`predict_proba`/`transform`/`decision_function`
are the crate-root re-export and tests (`api_proof_lda in api_proof.rs`,
`conformance_lda in conformance_wave1.rs`).

Per goal.md the pre-existing in-repo conformance suite is **explicitly excluded**
("Ignore the pre-existing in-repo conformance suite — it is not the contract
here"). `conformance_lda in conformance_wave1.rs` is that suite, pins against a
static `fixtures/lda.json`, and explicitly only asserts an **accuracy floor**
(`acc >= 0.90`) while commenting that "ferrolearn LDA's predict_proba divergence
from sklearn is larger than" tolerance and discarding the proba comparison — so
it is **not** R-CHAR-3 parity evidence and in fact documents the divergence. The
`lda.rs` unit tests assert structural facts (shapes, accessors, accuracy floors,
sign-of-projection) — none pins a `decision_function`/`transform`/`proba`
**value** against a live `LinearDiscriminantAnalysis` oracle.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (svd fit + decision_function parity) | SHIPPED | `_solve_svd` transcribed in `fn fit in lda.rs` (two thin SVDs via `fn svd_s_vt` → `ferray::linalg::svd`) builds `coef_`/`intercept_`/`xbar_`/`scalings_` (`discriminant_analysis.py:556-559`); `fn decision_function in lda.rs` = `X @ coef_.T + intercept_` (the `LinearClassifierMixin`, `:739`), whose `intercept_` embeds `log(priors_)` (`:557`). Consumer: `Predict for FittedLDA` (`fn predict`) + the crate-root `pub use lda::{FittedLDA, LDA} in lib.rs`. R-CHAR-3 pin `lda_decision_function_parity in divergence_lda_fit.rs` matches the live `L().decision_function(X)` (mc set row 0 `[25.2084,-32.9455,-56.6508]`) within `1e-6`. |
| REQ-2 (predict argmax) | SHIPPED | `Predict::predict in lda.rs` = `classes_[argmax(decision_function)]` over the affine `X@coef_.T+intercept_` whose `intercept_` carries `log(priors_)` (numpy first-max-wins tie-break). Consumer: the `Predict` impl, reached via the crate-root `pub use` boundary. R-CHAR-3 pin `lda_imbalanced_priors_predict in divergence_lda_fit.rs`: on the 20-vs-2 imbalanced set (priors `[0.9091,0.0909]`) the prior shifts the boundary so the borderline `(2.0,0.0)` is labelled class 0 (`predict_eval == [0,1,0]`), label-for-label vs the live `L().predict`. |
| REQ-3 (predict_proba) | SHIPPED | `fn predict_proba in lda.rs` = row-max-shifted `softmax(decision_function)` over the prior-aware affine decision (`discriminant_analysis.py:706-711`, `extmath.py:949-985`); rows sum to 1. Consumer: `fn predict_log_proba` (production method) + crate-root re-export. R-CHAR-3 pin: the `predict_proba` block of `lda_imbalanced_priors_predict in divergence_lda_fit.rs` matches the live `L().predict_proba(Xe)` (e.g. `[0.8879,0.1121]`) within `1e-6`. NOTE: the binary `[1-expit(d),expit(d)]` collapse (and the binary `coef_` single-row form) pends REQ-14/#600; the multiclass softmax is exact because `coef_`/`intercept_` are not yet collapsed to the binary form, so the 2-class proba here is the softmax of the full 2-row decision, which sklearn's `expit` path equals up to FP. |
| REQ-4 (predict_log_proba) | SHIPPED | `fn predict_log_proba in lda.rs` now mirrors sklearn exactly (`discriminant_analysis.py:713-737`): `predict_proba` then bump exact-`0.0` entries by `F::min_positive_value()` (numpy `finfo.smallest_normal`, `:729-736`) before elementwise `log`, so nonzero probas keep their true `ln` and exact zeros become `log(MIN_POSITIVE)` rather than `-inf`. Consumer: shares `FittedLDA::predict_proba` (the `Predict` path), reached via the crate-root `pub use`. R-CHAR-3 pin `lda_predict_log_proba in divergence_lda_fit.rs` matches the live `L().predict_log_proba(X)` on an overlapping 3-class set (all-finite log-probas) within `1e-6`. |
| REQ-5 (transform projection parity) | SHIPPED | `fn transform in lda.rs` = `((X - xbar_) @ scalings_)[:, :max_components]` with the SVD-whitened `scalings_` (`discriminant_analysis.py:684-689`). Consumer: the `Transform for FittedLDA` impl + crate-root re-export. R-CHAR-3 pin `lda_transform_parity in divergence_lda_fit.rs` matches the live `L().transform(X)` (mc set row 0 `[-4.72428,5.69725]`) within `1e-6`, per-column up to the SVD sign ambiguity. |
| REQ-6 (n_components bound) | SHIPPED | impl `fn fit in lda.rs` computes `let max_components = (n_classes - 1).min(n_features);`, defaults `None` to `max_components`, returns `InvalidParameter` for `Some(0)` and for `Some(k)` with `k > max_components` — mirroring sklearn `max_components = min(n_classes - 1, X.shape[1])` and the `ValueError("n_components cannot be larger than min(n_features, n_classes - 1).")` (`discriminant_analysis.py:614-625`). Consumer: `fn fit` reads `self.n_components` (production), reached via the crate-root `pub use` boundary and `api_proof_lda in api_proof.rs`. Verified: `test_lda_default_n_components`, `test_lda_error_zero_n_components`, `test_lda_error_n_components_too_large`, `test_lda_n_components_getter`, `test_lda_fit_returns_fitted` (`scalings().ncols()==1` for 2 classes) in `lda.rs` pin the `min(n_classes-1, n_features)` rule against the sklearn-documented bound (R-CHAR-3: the bound is a sklearn `file:line` structural constant, not a copied value). |
| REQ-7 (priors None + provided) | SHIPPED | `fn fit in lda.rs` resolves `priors_`: empirical `n_k/n` per class when the constructor `priors` is `None` (`discriminant_analysis.py:601-603`), else the provided `LDA::with_priors(Array1<F>)` array (`:605`, `self.priors_ = xp.asarray(self.priors)`), now VALIDATED per `:607-612` (see below). R-DEV-4 length check: `p.len() != n_classes` → `ShapeMismatch` (sklearn would silently mis-index it in `xbar_`/scaling/`intercept_`). The resolved priors enter the affine decision through `xbar_ = priors_ @ means_` (`:517`), the between-class scaling `sqrt(n·priors_·fac)` (`:540`), and `intercept_ = -½Σcoef² + log(priors_)` (`:557`). `FittedLDA::priors` exposes `priors_`; `LDA::priors`/`LDA::with_priors` are the unfitted accessor/builder. Consumer: `fn fit` reads the resolved priors (xbar_/scaling/intercept_); `Predict for FittedLDA` consumes the prior-shifted decision. R-CHAR-3 pins: `lda_imbalanced_priors_predict` (empirical `[0.9091,0.0909]` flips the label), `lda_provided_priors` (`with_priors([0.9,0.1])` `predict_proba` <1e-6 vs live `L(priors=[0.9,0.1])`; the empirical default resolves a different `priors_`), `lda_provided_priors_length_mismatch` (3-element priors on 2 classes → `Err`). Provided priors are now VALIDATED exactly like sklearn LDA (`:607-612`, unlike QDA): negative entries → `InvalidParameter` (`:607-608`), and renormalized to sum 1 (`p / p.sum()`, with an `eprintln!` warning — the crate's warning channel, cf. qda.rs collinearity warning) when `|Σ-1|>1e-5` (`:610-612`). R-CHAR-3 pins: `lda_priors_negative_rejected` (`[-0.1,1.1]` → `Err`, vs live `L(priors=[-0.1,1.1]).fit` `ValueError("priors must be non-negative")`), `lda_priors_renormalized` (`[0.5,0.6]` → `priors_=[0.45454…,0.54545…]` and `predict_proba` <1e-6 vs the live `L(priors=[0.5,0.6])` which renormalizes internally). #603. |
| REQ-8 (coef_/intercept_/xbar_) | SHIPPED | `FittedLDA<F>` now stores `coef`/`intercept`/`xbar`/`priors` and exposes `FittedLDA::{coef, intercept, xbar, priors}` accessors. `fn fit in lda.rs` derives them exactly per `_solve_svd`: `coef = (means_-xbar_)@scalings_`; `intercept_ = -0.5·Σcoef² + log(priors_)`; `coef_ = coef@scalings_.T`; `intercept_ -= xbar_@coef_.T` (`discriminant_analysis.py:556-559`); `xbar_ = priors_@means_` (`:517`). Consumer: `fn decision_function` reads `coef_`/`intercept_`; `fn transform` reads `xbar_`. R-CHAR-3 pins: `test_lda_coef_intercept_xbar_oracle in lda.rs` (`coef_`/`intercept_`/`xbar_` vs live `L()` attrs, <1e-9) + `lda_decision_function_parity`. The binary `(1, n_features)`/`(1,)` collapse (`:651-657`) pends REQ-14/#600. |
| REQ-9 (lsqr solver) | NOT-STARTED | open prereq blocker #595. ferrolearn implements a single (eigen-style) solver; there is no `solver` constructor parameter and no `_solve_lstsq` analog (`coef_ = lstsq(covariance_, means_.T)[0].T`, `discriminant_analysis.py:365-419`), nor the `transform` `NotImplementedError` for `lsqr`. |
| REQ-10 (eigen solver) | NOT-STARTED | open prereq blocker #596. ferrolearn's `Sw⁻¹Sb` Jacobi path is NOT sklearn's `eigen` solver: sklearn solves the generalized `eigh(Sb, Sw)` on the **prior-weighted biased** `_class_cov`/`_cov` (`discriminant_analysis.py:421-485`), sets `scalings_ = evecs`, and derives `coef_`/`intercept_` (with `log priors_`). ferrolearn uses an un-averaged scatter + `1e-6` ridge, raw eigenvectors as `scalings`, no `coef_`/`intercept_`. No `solver="eigen"` selector, and the numerical path diverges. |
| REQ-11 (shrinkage) | NOT-STARTED | open prereq blocker #597. No `shrinkage` parameter; no Ledoit-Wolf `'auto'` path (`_cov`/`_class_cov`, `discriminant_analysis.py:339, 128-172`) and no `NotImplementedError("shrinkage not supported with 'svd' solver.")` guard (`:628-629`). |
| REQ-12 (store_covariance / covariance_) | SHIPPED | `LDA::with_store_covariance` sets the flag (default `false`, `discriminant_analysis.py:353`); when `true`, `fn fit in lda.rs` computes the prior-weighted shared within-class covariance `covariance_ = Σ_k priors_[k] · cov(X_k)` (`:509-510`, `_class_cov` `:128-172`), where `cov(X_k)` is the maximum-likelihood empirical covariance (`np.cov(..., bias=1)`, ÷`n_k`, via `empirical_covariance` `covariance/_empirical_covariance.py:109`). Stored on `FittedLDA::covariance` (`None` when the flag is unset, matching sklearn). Consumer: `fn fit in lda.rs` reads `self.store_covariance`/`priors`/`means` and populates the field; `FittedLDA::covariance` exposes it. R-CHAR-3 pin `lda_store_covariance in divergence_lda_fit.rs` matches the live `L(store_covariance=True).fit(X,y).covariance_` (`[[0.4296875,-0.1328125],…]`) within `1e-9` and asserts `None` for the default/`false` path. |
| REQ-13 (explained_variance_ratio_) | SHIPPED | `fn fit in lda.rs` sets `explained_variance_ratio_ = (S2²/ΣS2²)[:max_components]` from the SECOND (between-class) SVD's singular values, exactly per `discriminant_analysis.py:550-552` — no longer the diverging eigenvalue-ratio path. Consumer: `FittedLDA::explained_variance_ratio` accessor + crate-root re-export. R-CHAR-3 pin `test_lda_explained_variance_ratio_oracle in lda.rs` matches the live `L().explained_variance_ratio_` (mc set `[0.642868,0.357132]`) within `1e-9`. |
| REQ-14 (decision_function shape/sign) | NOT-STARTED | open prereq blocker #600. `fn decision_function in lda.rs` ALWAYS returns `Array2<F>` shape `(n_samples, n_classes)` (`out: Array2::<F>::zeros((n_samples, n_classes))`). sklearn collapses binary to `(n,)` = `log p(y=1|x)-log p(y=0|x)` (live oracle: shape `(7,)` on the binary set), multiclass `(n, n_classes)` (`discriminant_analysis.py:739-759`). A shape/sign ABI divergence (R-DEV-3), parallel to QDA #581. |
| REQ-15 (tol) | SHIPPED | `LDA::with_tol` sets the svd-solver rank threshold (default `1e-4`, `discriminant_analysis.py:354`); `fn fit in lda.rs` reads `self.tol` into BOTH rank cutoffs `rank = Σ(S > tol)` (`:532`) and `rank2 = Σ(S2 > tol·S2[0])` (`:554`), replacing the prior hardcoded `1e-4`. Default `1e-4` keeps the svd fit byte-identical (all existing svd-fit oracle pins stay green). Consumer: `fn fit in lda.rs` reads `self.tol` in both thresholds. R-CHAR-3 pin `lda_tol_param in divergence_lda_fit.rs` (field default `1e-4` == sklearn `:354` symbolic constant + `with_tol` plumb-through). |
| REQ-16 (ferray substrate) | NOT-STARTED | open prereq blocker #602. The rewrite routes **both** SVDs through `ferray::linalg::svd` (`fn svd_s_vt in lda.rs`, mirroring `qda.rs::svd_s_vt` / `bayesian_ridge.rs::svd_thin`) and the hand-rolled Jacobi/Gaussian-elimination helpers are GONE. The remaining gap is the owned array type: `lda.rs` still imports `ndarray::{Array1, Array2}` and computes on `ndarray` (bridged to ferray only at the SVD boundary, R-SUBSTRATE-4). Consistent with the crate-wide owned-array deferral (cf. `qda.md` REQ-12 #585, `bayesian_ridge.md` REQ-10 #471). |

## Architecture

### sklearn (the contract)

`LinearDiscriminantAnalysis(ClassNamePrefixFeaturesOutMixin,
LinearClassifierMixin, TransformerMixin, BaseEstimator)`
(`discriminant_analysis.py:175-180`), keyword-defaulted
`(solver="svd", shrinkage=None, priors=None, n_components=None,
store_covariance=False, tol=1e-4, covariance_estimator=None)` (`:347-363`).
The **classifier** identity is the affine map `decision_function = X@coef_.T +
intercept_` (from `LinearClassifierMixin`); the **transformer** identity is the
projection onto `scalings_`. The default `svd` solver never forms the covariance
(SVD whitening of the centered/std-scaled data → `scalings_`, then `coef_`/
`intercept_` from `means_`, `xbar_`, and `log priors_`). `lsqr`/`eigen` form the
prior-weighted within covariance (`_class_cov`) and support `shrinkage`. The
binary case collapses `coef_`/`intercept_` to the class-1-minus-class-0
difference, so `decision_function` is `(n,)`.

### ferrolearn (what exists)

`LDA<F> { n_components }` (`lda.rs`) → `fn fit` builds `Sw` (un-averaged within
scatter + `1e-6` ridge), `Sb = Σ n_c (μ_c-μ)(μ_c-μ)ᵀ`, `M = Sw⁻¹Sb` (`fn
sw_inv_sb` via `fn gaussian_solve_f`), Jacobi eigendecomposition (`fn
jacobi_eigen_f`), descending sort, `scalings` = top-`k` eigenvectors,
`explained_variance_ratio = clamp(λ)/Σclamp(λ)`, and projected `means`.
`FittedLDA<F> { scalings, means, explained_variance_ratio, classes, n_features }`.
`transform` = raw `X@scalings`; `predict`/`decision_function`/`predict_proba`/
`predict_log_proba` = nearest-centroid / softmax of `-½‖z-μ_c‖²` in projected
space (equal-priors approximation). Re-exported at the crate root; **not** bound
in `ferrolearn-python`.

### Why ferrolearn diverges (and the one match)

ferrolearn answers a *related but distinct* mathematical question. The classical
`Sw⁻¹Sb` Fisher discriminant gives the same **discriminant directions** (so the
eigenvalue ratio `λ_k/Σλ` equals sklearn's `S²/ΣS²` — REQ-13's numerical
coincidence, verified ~exact on iris), and for **balanced** classes the
nearest-centroid argmax coincides with sklearn's `argmax(X@coef_.T+intercept_)`.
But: (a) `transform` is uncentered and unwhitened, so its **values** differ
(REQ-5); (b) `decision_function`/`predict_proba` are a Euclidean-distance
softmax, not the affine log-posterior, so their **values** differ even for
uniform priors (REQ-1/3); (c) without `priors_`, the `log(prior_k)` term is
absent, so **predictions flip** on imbalanced/provided priors (REQ-2/7); (d) the
binary decision is `(n, n_classes)`, not sklearn's `(n,)` log-likelihood ratio
(REQ-14); (e) no `coef_`/`intercept_`/`xbar_`/`covariance_` attributes
(REQ-8/12); (f) only one solver, no shrinkage, no `tol`, no `priors` parameter
(REQ-9/10/11/15); (g) `ndarray` + hand-rolled linalg, not ferray (REQ-16).

## Verification

Library-crate gauntlet (baseline `83e73bf4`):

- `cargo test -p ferrolearn-linear lda` — the module unit tests (shapes,
  accessors, accuracy floors, projection sign). These do NOT pin a
  `decision_function`/`transform`/`proba`/`coef_` **value** against the live
  oracle, so they do not establish REQ-1..5/8/13/14 parity (R-CHAR-3).
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the divergences; expected values come from
sklearn, never copied from ferrolearn, per R-CHAR-3):

```bash
python3 -c "
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L
from sklearn.datasets import load_iris
Xi,yi=load_iris(return_X_y=True); mi=L().fit(Xi,yi)
print('evr', mi.explained_variance_ratio_.round(6).tolist())      # [0.991213, 0.008787]
print('transform[0]', mi.transform(Xi)[0].round(4).tolist())      # [8.0618, -0.3004]
print('decision[0]', mi.decision_function(Xi)[0].round(2).tolist())# [31.34, -17.96, -64.41]
# imbalanced -> empirical priors non-uniform -> log(prior) shifts the boundary
rng=np.random.RandomState(0)
X=np.vstack([rng.randn(40,2)+[0,0], rng.randn(4,2)+[2.,0]]); y=np.array([0]*40+[1]*4)
m=L().fit(X,y); print('priors_', m.priors_.round(4).tolist())     # [0.9091, 0.0909]
print('binary dec shape', m.decision_function(X).shape)           # (44,)
print('provided priors', L(priors=[0.9,0.1]).fit(X,y).priors_.tolist())  # [0.9, 0.1]
"
```

Observed: `explained_variance_ratio_` matches ferrolearn's eigenvalue ratio
(REQ-13 numerical coincidence, but unpinned); `transform`/`decision_function`
**values** and the binary `(n,)` shape diverge from ferrolearn; empirical/
provided priors are honored by sklearn and absent in ferrolearn. A NOT-STARTED
REQ closes only when its fix lands AND a divergence test (expected values from
the live oracle / a sklearn `file:line` constant) goes green; the
`conformance_lda`/`fixtures/lda.json` suite is excluded by goal.md (it asserts an
accuracy floor and explicitly discards the proba comparison) and is not a
substitute.

## Blockers to open

- **#588** — REQ-1 of lda: implement the `svd` solver (`scalings_`, `xbar_`,
  `coef_`, `intercept_` with `log priors_`) and pin `decision_function =
  X@coef_.T+intercept_` against the live `L().decision_function` oracle (iris,
  within `1e-8`), replacing the projected-distance approximation.
- **#589** — REQ-2 of lda: pin `predict` label-for-label against `L().predict`
  on an **imbalanced** 2-class set (prior shifts the boundary), not an accuracy
  floor.
- **#590** — REQ-3 of lda: implement prior-aware `predict_proba` (binary `expit`,
  multiclass softmax of the affine decision), pin against `L().predict_proba`,
  and add a non-test production consumer (register LDA in `ferrolearn-python`).
- **#591** — REQ-4 of lda: pin `predict_log_proba` against the live oracle and
  replicate sklearn's `smallest_normal` zero-floor.
- **#592** — REQ-5 of lda: implement `transform = (X - xbar_) @ scalings_` with
  the SVD-whitened scalings and pin it against `L().transform` (iris, within
  `1e-6` up to per-column sign).
- **#593** — REQ-7 of lda: add a `priors` constructor parameter (None →
  empirical `cnts/n`, array → used, renormalized-with-warning if `|Σ-1|>1e-5`,
  reject negatives) and store/expose `priors_`
  (`discriminant_analysis.py:601-612`).
- **#594** — REQ-8 of lda: expose `coef_`/`intercept_`/`xbar_` on `FittedLDA`
  (with the binary `(1, n_features)`/`(1,)` collapse,
  `discriminant_analysis.py:556-559, 651-657`).
- **#595** — REQ-9 of lda: add the `solver="lsqr"` least-squares path and the
  `transform` `NotImplementedError` for it.
- **#596** — REQ-10 of lda: add the `solver="eigen"` generalized-eigenvalue path
  (`eigh(Sb, Sw)` on prior-weighted `_class_cov`, `coef_`/`intercept_` from
  means+priors), distinct from the existing `Sw⁻¹Sb` Jacobi approximation.
- **#597** — REQ-11 of lda: add `shrinkage` (None/'auto' Ledoit-Wolf/float) on
  `lsqr`/`eigen` plus the `NotImplementedError` on `svd`.
- **#598** — REQ-12 of lda: add `store_covariance` + a `covariance_` accessor
  (`Σ_k prior_k·C_k`).
- **#599** — REQ-13 of lda: pin `explained_variance_ratio_` against
  `L().explained_variance_ratio_` (iris `[0.991213, 0.008787]`) via the SVD
  solver's `S²/ΣS²`.
- **#600** — REQ-14 of lda: collapse the binary `decision_function` to shape
  `(n,)` = `log p(y=1|x)-log p(y=0|x)` (multiclass unchanged), matching
  `discriminant_analysis.py:739-759` (parallel to QDA #581).
- **#601** — REQ-15 of lda: add the `tol` parameter and the SVD rank thresholds
  `Σ(S > tol)` / `Σ(S > tol·S[0])`.
- **#602** — REQ-16 of lda: migrate `lda.rs` off `ndarray` + hand-rolled Jacobi/
  Gaussian elimination onto the ferray substrate (`ferray-core` arrays,
  `ferray::linalg` SVD/eig/solve).
```
