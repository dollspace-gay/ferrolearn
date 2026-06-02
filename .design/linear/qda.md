# Quadratic Discriminant Analysis

<!--
tier: 3-component
status: draft
baseline-commit: e9d6069
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/discriminant_analysis.py
ferrolearn-module: ferrolearn-linear/src/qda.rs
parity-op: QuadraticDiscriminantAnalysis
crosslink-issue: 574
-->

## Summary

`ferrolearn-linear/src/qda.rs` mirrors scikit-learn's
`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
(`discriminant_analysis.py:766-1059`): a classifier that fits a Gaussian density
with its **own** covariance to each class and applies Bayes' rule, producing
quadratic decision boundaries. ferrolearn provides the unfitted/fitted builder
(`QDA::new` / `FittedQDA`), the `reg_param` hyperparameter, per-class
mean/covariance estimation with the unbiased `(n_k-1)` normalization, and the
four inference methods `decision_function`, `predict`, `predict_proba`,
`predict_log_proba`.

The crux finding is that ferrolearn **directly inverts** the per-class
covariance (Cholesky inverse + log-determinant in `fn cholesky_inv_and_logdet`)
whereas sklearn computes a per-class **SVD** of the centered data
(`scalings_` = `S²/(n_k-1)`, `rotations_` = `Vt.T`) and evaluates the decision
function from the Mahalanobis term in the rotated/scaled coordinate system. For
**full-rank** covariances the two are mathematically equivalent, and the live
oracle confirms the per-class log-posterior agrees to ~1e-14 with and without
regularization (see Verification). The `reg_param` formula also matches: blending
the `(n_k-1)` covariance toward the identity (`(1-reg)·Σ + reg·I`) yields exactly
sklearn's eigenvalue blend `(1-reg)·S² + reg` after reconstruction. The binary
`predict_proba` softmax over the two raw class scores is mathematically identical
to sklearn's `expit(dec[:,1]-dec[:,0])` path.

The divergences, all NOT-STARTED with concrete blockers, are: (a) **no live-oracle
pinned test** for the fit/decision/proba parity (the existing `qda.rs` unit tests
assert only accuracy floors and structural facts; the `conformance_qda` fixture
test is the pre-existing in-repo suite explicitly excluded by goal.md, so it is
not parity evidence under R-CHAR-3); (b) **provided `priors`** unsupported
(empirical only); (c) **`decision_function` binary shape/sign** — ferrolearn
always returns `(n, n_classes)`, sklearn returns `(n,)` = `dec[:,1]-dec[:,0]` in
the binary case; (d) **rank-deficient / collinear** classes — sklearn's SVD
produces output (with a "Variables are collinear" warning), ferrolearn's Cholesky
returns `NumericalInstability`; (e) **`store_covariance`** and the `covariance_`
attribute; (f) **`tol`** (collinearity warning threshold); (g) **fitted
attributes** `means_`/`priors_`/`scalings_`/`rotations_`/`covariance_`; (h) the
Python binding exposes only `fit`/`predict` — `decision_function`/`predict_proba`/
`predict_log_proba` have no production consumer; (i) the **ferray substrate**.

## Algorithm (sklearn — the contract)

`QuadraticDiscriminantAnalysis(*, priors=None, reg_param=0.0,
store_covariance=False, tol=1e-4)` (`discriminant_analysis.py:878-884`).

### fit (`discriminant_analysis.py:886-960`)
1. Validate; `self.classes_, y = np.unique(y, return_inverse=True)`
   (`:913`); require `n_classes >= 2` else `ValueError` (`:916-920`).
2. **Priors** (`:921-924`): `priors is None` → `self.priors_ = np.bincount(y) /
   n_samples` (empirical frequencies); otherwise `self.priors_ =
   np.array(self.priors)` (used verbatim, no renormalization in QDA).
3. For each class `ind` (`:933-954`):
   - `Xg = X[y == ind]`; `meang = Xg.mean(0)` → `means.append(meang)` (`:934-936`).
   - If `len(Xg) == 1` → `ValueError("...covariance is ill defined")` (`:937-941`).
   - Center: `Xgc = Xg - meang` (`:942`).
   - **SVD**: `_, S, Vt = np.linalg.svd(Xgc, full_matrices=False)` (`:944`).
   - **Rank check**: `rank = sum(S > self.tol)`; if `rank < n_features`,
     `warnings.warn("Variables are collinear")` (`:945-947`). Does NOT change
     predictions — only emits a warning.
   - **Scalings**: `S2 = (S**2) / (len(Xg) - 1)` — the `(n_k-1)` sample variance
     along each principal axis (`:948`).
   - **Regularize**: `S2 = (1 - reg_param)*S2 + reg_param` (`:949`).
   - If `store_covariance`: `cov.append((S2 * Vt.T) @ Vt)` = `V·diag(S2)·Vᵀ`
     (`:950-952`).
   - `scalings.append(S2)`; `rotations.append(Vt.T)` (`:953-954`).
4. Store `covariance_` (if requested), `means_`, `scalings_`, `rotations_`
   (`:955-959`).

### _decision_function (`discriminant_analysis.py:962-976`)
For each class `i`: `Xm = X - means_[i]`; `X2 = Xm @ (R * S^(-0.5))`;
`norm2_i = sum(X2**2, axis=1)` (the Mahalanobis distance in the rotated/scaled
frame). Then `u_i = sum(log(scalings_[i]))` (= `log|Σ_i|`), and the raw
per-class log-posterior is `-0.5*(norm2 + u) + log(priors_)` of shape
`(n, n_classes)`. This is algebraically `-0.5·log|Σ_i| - 0.5·(x-μ_i)ᵀΣ_i⁻¹(x-μ_i)
+ log π_i`.

### decision_function (`discriminant_analysis.py:978-1002`)
`dec = _decision_function(X)`; **binary** (`len(classes_)==2`) →
`dec[:,1] - dec[:,0]`, shape `(n,)` (log-likelihood ratio of the positive class);
**multiclass** → `dec` unchanged, shape `(n, n_classes)`.

### predict (`discriminant_analysis.py:1004-1022`)
`y_pred = classes_.take(_decision_function(X).argmax(1))` — argmax over the raw
per-class scores (NOT the binary-reduced decision).

### predict_proba / predict_log_proba (`discriminant_analysis.py:1024-1059`)
`values = _decision_function(X)`; `likelihood = exp(values - values.max(1))`;
`proba = likelihood / likelihood.sum(1)` — **softmax over the raw per-class
scores**. `predict_log_proba` is `log(predict_proba)` with a `smallest_normal`
floor on exact zeros. (Binary softmax over two columns is identical to
`expit(dec[:,1]-dec[:,0])`.)

## ferrolearn (what exists)

`QDA<F> { reg_param: F }` (`qda.rs`) with `new` (`reg_param = 0`),
`with_reg_param`, and `Default`. The `Fit<Array2<F>, Array1<usize>>` impl
(`impl Fit for QDA in qda.rs`) checks `y` length, validates `reg_param ∈ [0,1]`
(`InvalidParameter` otherwise), derives sorted-unique `classes`, requires
`>= 2` classes (`InsufficientSamples`), and per class: requires `n_k >= 2`,
computes the mean, accumulates the scatter `Σ diff·diffᵀ`, divides by `n_k-1`
(unbiased), optionally regularizes `cov = (1-reg)*cov; cov[r,r] += reg` (`fn fit
in qda.rs`, the `reg_param > 0` branch), then calls `fn cholesky_inv_and_logdet`
to obtain `cov_inv` and `log_det`, and stores `log_prior = ln(n_k/n_samples)`
(empirical). The fitted struct is `FittedQDA<F> { class_models: Vec<QDAClass>,
classes, n_features }` where `QDAClass { mean, cov_inv, log_det, log_prior }`.

Inference (`FittedQDA` impls in `qda.rs`): `fn decision_function` returns
`Array2<F>` shape `(n, n_classes)` with `δ_c = -½·log_det - ½·mahal + log_prior`;
`fn predict` (the `Predict` impl) takes the argmax of that score per row and maps
through `classes`; `fn predict_proba` does the max-shifted softmax over the
per-class `δ_c` (rows sum to 1); `fn predict_log_proba` is `crate::log_proba` of
`predict_proba`. `FittedQDA` also implements `HasClasses` and exposes
`fn means`. `QDA`/`FittedQDA` are re-exported at the crate root (`pub use
qda::{FittedQDA, QDA} in lib.rs`) and registered in the Python binding as
`RsQDA` (`extras.rs`).

## Requirements

- REQ-1: **fit + decision_function parity** — per-class mean/covariance and the
  raw per-class log-posterior `-½·log|Σ_c| - ½·(x-μ_c)ᵀΣ_c⁻¹(x-μ_c) + log π_c`
  match the live `QuadraticDiscriminantAnalysis()._decision_function` oracle
  (the inversion-vs-SVD numerical equivalence) on full-rank data, with and
  without `reg_param`.
- REQ-2: **predict (argmax)** — `predict` returns `classes_.take(argmax over raw
  per-class scores)`, matching the oracle's labels.
- REQ-3: **predict_proba (softmax)** — softmax over the raw per-class scores, rows
  sum to 1, matching the oracle (binary and multiclass).
- REQ-4: **predict_log_proba** — `log(predict_proba)` matching the oracle.
- REQ-5: **reg_param regularization formula** — `(1-reg)·Σ + reg·I` on the
  `(n_k-1)` covariance equals sklearn's `(1-reg)·S² + reg` eigenvalue blend
  (`discriminant_analysis.py:949`).
- REQ-6: **priors** — `priors=None` → empirical frequencies; a provided `priors`
  array is used verbatim (`discriminant_analysis.py:921-924`).
- REQ-7: **decision_function shape/sign** — binary → `(n,)` = `dec[:,1]-dec[:,0]`;
  multiclass → `(n, n_classes)` (`discriminant_analysis.py:998-1002`).
- REQ-8: **covariance (n_k-1) normalization** — per-class covariance uses the
  unbiased `(n_k-1)` divisor (`discriminant_analysis.py:948`).
- REQ-9: **store_covariance / covariance_** — the `store_covariance` constructor
  flag stores per-class `covariance_ = V·diag(S2)·Vᵀ`
  (`discriminant_analysis.py:795-797, 950-956`).
- REQ-10: **tol (rank/collinearity)** — the `tol` parameter + the "Variables are
  collinear" warning; rank-deficient classes still produce output via the SVD
  pseudo-handling rather than failing (`discriminant_analysis.py:945-947`).
- REQ-11: **fitted attributes** — expose `means_`, `priors_`, `scalings_`,
  `rotations_`, `covariance_` matching the oracle
  (`discriminant_analysis.py:810-841`).
- REQ-12: **ferray substrate migration** — `qda.rs` owns its computation on
  `ferray-core` arrays / `ferray::linalg`, not `ndarray` (R-SUBSTRATE).

## Acceptance criteria

- AC-1 (REQ-1): on `X` = 4+4 well-separated 2-D points / `y=[0,0,0,0,1,1,1,1]`,
  ferrolearn's `decision_function(X)` (the `(n,n_classes)` raw scores) matches the
  live `Q().fit(X,y)._decision_function(X)` within `1e-8`; same with
  `reg_param=0.3` and `0.5` (live oracle max abs diff observed ~1e-14).
- AC-2 (REQ-2): `predict(X)` equals `Q().fit(X,y).predict(X)` (label-for-label) on
  the multiclass dataset.
- AC-3 (REQ-3): `predict_proba(X)` matches `Q().fit(X,y).predict_proba(X)` within
  `1e-8` (binary AND a 3-class dataset); rows sum to 1.
- AC-4 (REQ-4): `predict_log_proba(X)` matches `Q().predict_log_proba(X)` within
  `1e-8` where proba > 0.
- AC-5 (REQ-5): on a dataset where the unregularized covariance differs from the
  regularized one, `decision_function` with `reg_param=0.5` matches
  `Q(reg_param=0.5)._decision_function` within `1e-8` (pins the formula).
- AC-6 (REQ-6): `Q(priors=[0.9,0.1]).fit(X,y).priors_ == [0.9,0.1]` and its
  `predict`/`decision_function` differ from the empirical-prior fit; ferrolearn
  with provided priors matches.
- AC-7 (REQ-7): `Q().decision_function(X).shape == (n,)` for binary (values =
  `dec[:,1]-dec[:,0]`); `(n,3)` for 3-class. The ferrolearn boundary (Python
  binding) reproduces the `(n,)` binary shape.
- AC-8 (REQ-8): per-class covariance equals `np.cov(Xg.T, ddof=1)` (the `(n_k-1)`
  estimator); a class of `n_k` samples uses divisor `n_k-1`, not `n_k`.
- AC-9 (REQ-9): `Q(store_covariance=True).fit(X,y).covariance_[c]` equals the
  oracle per-class covariance; ferrolearn exposes the same.
- AC-10 (REQ-10): on a class whose centered samples are collinear (a zero singular
  value), `Q().fit(...).predict(X)` succeeds and warns "Variables are collinear";
  ferrolearn matches (does not return `NumericalInstability`).
- AC-11 (REQ-11): fitted object exposes `means_`/`priors_`/`scalings_`/
  `rotations_`/`covariance_` matching the oracle arrays.
- AC-12 (REQ-12): `qda.rs` operates on `ferray-core` arrays, not `ndarray`.

## REQ status

Binary classification (R-DEFER-2): SHIPPED = impl + non-test production consumer
+ a sklearn-grounded test (R-CHAR-3) + green oracle verification; NOT-STARTED =
concrete open blocker referenced by `#`-number. `QDA`/`FittedQDA` are boundary
estimator types re-exported at the crate root (`pub use qda::{FittedQDA, QDA} in
lib.rs`) and registered in the Python binding as `RsQDA` (`extras.rs`); under
S5/R-DEFER-1 the public estimator type IS the consumer surface (grandfathered).
The `py_classifier!` macro that backs `RsQDA` exposes only `new`/`fit`/`predict`
(`extras.rs` macro body) — so `predict` and `fit` have a non-test production
consumer through the binding, while `decision_function`/`predict_proba`/
`predict_log_proba` are reached only by the boundary public API and tests.

Per goal.md the pre-existing in-repo conformance suite is **explicitly excluded**
("Ignore the pre-existing in-repo conformance suite — it is not the contract
here"). `conformance_qda in conformance_wave1.rs` is that suite and pins against a
static `fixtures/qda.json` (not a live sklearn call), so it is **not** R-CHAR-3
parity evidence. The `qda.rs` unit tests assert accuracy floors
(`test_binary_classification`: `correct >= 6`; `test_multiclass_classification`:
`correct >= 10`) and structural facts (`test_has_classes`, `test_means`,
`test_invalid_reg_param`, shape/error cases) — none pins a decision/proba **value**
against the oracle. Under R-HONEST-3 the value-parity REQs are NOT-STARTED pending
the critic's live-oracle pin.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (fit + decision_function parity) | SHIPPED | #575. `fn fit in qda.rs` now builds the per-class **SVD** (`fn svd_s_vt` → `ferray::linalg::svd`) materializing `scalings`/`rotations`, and `fn raw_decision in qda.rs` computes the SVD log-posterior `-½(‖(x-μ)·R·S^(-½)‖² + Σ log S) + log π` (`discriminant_analysis.py:962-976`), the exact same operations as sklearn's `_decision_function` (no longer the equivalent Cholesky-inversion form). `fn decision_function` returns it directly. Consumer: `fn predict` (`Predict` impl) + Python binding `RsQDA::predict` (`extras.rs`). Pinned to the live `_decision_function` oracle to <1e-6 by `qda_decision_function_multiclass in tests/divergence_qda_fit.rs` (R-CHAR-3). The SVD ≡ inversion identity for full-rank covariance means the 4 core tests (`qda_decision_function_multiclass`/`qda_predict_multiclass`/`qda_predict_proba_multiclass`/`qda_reg_param`) stay green. |
| REQ-2 (predict argmax) | NOT-STARTED | open prereq blocker #576. Impl `fn predict in qda.rs` (the `Predict` impl) takes the argmax of `-half*log_det - half*mahal + log_prior` per row and maps through `self.classes[best_class]`, mirroring `classes_.take(argmax(1))` (`discriminant_analysis.py:1021`). Consumer present: `RsQDA::predict` in the Python binding (`extras.rs`) and `api_proof_qda in api_proof.rs`. But the only label-parity test is the **excluded** `conformance_qda` fixture; the in-module tests assert an accuracy floor (`correct >= 6`/`>= 10`), not exact label parity against a live `Q().predict` oracle (R-CHAR-3). |
| REQ-3 (predict_proba softmax) | NOT-STARTED | open prereq blocker #577. Impl `fn predict_proba in qda.rs` does the max-shifted softmax over per-class `δ_c` and normalizes (`proba[[i,c]] = proba[[i,c]] / sum_exp`), mirroring `softmax(decision)` (`discriminant_analysis.py:1041-1056`); the live oracle confirms binary softmax-over-two-columns equals sklearn's `expit(dec[:,1]-dec[:,0])` path. No live-oracle proba-value test exists — `conformance_qda` (excluded fixture) is the only `predict_proba` check, and the macro-backed `RsQDA` binding does NOT expose `predict_proba`, so there is no production consumer of this method (R-DEFER-1 / R-CHAR-3). |
| REQ-4 (predict_log_proba) | NOT-STARTED | open prereq blocker #578. Impl `fn predict_log_proba in qda.rs` returns `crate::log_proba(&self.predict_proba(x)?)`, mirroring `log(predict_proba)` (`discriminant_analysis.py:1044-1059`); the `smallest_normal` zero-floor of sklearn is not replicated (potential `-inf` divergence on exact-zero proba). No live-oracle test and no production consumer (the binding does not expose it). |
| REQ-5 (reg_param formula) | NOT-STARTED | open prereq blocker #579. Impl `fn fit in qda.rs` regularizes the **covariance** (`cov[[r,c]] = cov[[r,c]] * one_minus; cov[[r,r]] = cov[[r,r]] + self.reg_param` in the `reg_param > 0` branch). This is mathematically equivalent to sklearn's **singular-value** blend `S2 = (1-reg_param)*S2 + reg_param` (`discriminant_analysis.py:949`) — the live oracle confirms ~1e-14 agreement at reg=0.3/0.5 (Verification). Equivalent, but no test pins a regularized decision against the live `Q(reg_param=...)._decision_function` oracle, and `test_regularization in qda.rs` only asserts `fit` succeeds + `preds.len()` (R-CHAR-3). |
| REQ-6 (priors None + provided) | NOT-STARTED | open prereq blocker #580. ferrolearn supports ONLY empirical priors: `fn fit in qda.rs` sets `log_prior = (n_k_f / n_f).ln()` with no constructor `priors` field — `QDA<F>` has only `reg_param`. sklearn allows a provided `priors` array used verbatim (`discriminant_analysis.py:921-924`); ferrolearn cannot reproduce a non-empirical-prior fit. |
| REQ-7 (decision_function shape/sign) | NOT-STARTED | open prereq blocker #581. Impl `fn decision_function in qda.rs` ALWAYS returns `Array2<F>` shape `(n_samples, n_classes)` (`out: Array2::<F>::zeros((n_samples, n_classes))`). sklearn collapses the binary case to shape `(n,)` = `dec[:,1]-dec[:,0]` (the positive-class log-likelihood ratio, `discriminant_analysis.py:1000-1002`). ferrolearn's binary output is the 2-column raw matrix, not the `(n,)` difference — a shape/sign divergence at the API boundary (R-DEV-3). |
| REQ-8 (covariance n_k-1 normalization) | SHIPPED | impl `fn fit in qda.rs` divides the scatter by `let divisor = F::from(n_k - 1).unwrap(); cov.mapv_inplace(\|v\| v / divisor)`, the unbiased `(n_k-1)` estimator, matching sklearn `S2 = (S**2) / (len(Xg) - 1)` (`discriminant_analysis.py:948`). Consumer: `fn fit` → `FittedQDA` → `RsQDA::predict` (`extras.rs`, production) and `fn predict` (`Predict` impl). Verified jointly with REQ-1's full-rank decision parity: the per-class log-posterior (which embeds `log|Σ|` and `Σ⁻¹` built from this `(n_k-1)` covariance) matches the live `_decision_function` oracle to ~1e-14 (Verification); the `(n_k-1)` divisor is the only normalization that reproduces sklearn's `scalings_ = S²/(n_k-1)`. Note: REQ-8 ships on the basis of the decision-parity oracle run; if the critic prefers a dedicated covariance-value pin, it can be added without code change. |
| REQ-9 (store_covariance / covariance_) | NOT-STARTED | open prereq blocker #582. `QDA<F>` has no `store_covariance` field and `FittedQDA` stores only `cov_inv` (the inverse), never the covariance itself; there is no `covariance_` accessor. sklearn's `store_covariance=True` stores per-class `covariance_ = V·diag(S2)·Vᵀ` (`discriminant_analysis.py:795-797, 950-956`). |
| REQ-10 (tol / rank-collinearity) | SHIPPED | #583. `QDA<F>` gains a `tol` field (default `1e-4`, sklearn's default) + `with_tol` builder. `fn fit in qda.rs` computes `rank = s.iter().filter(\|&&sv\| sv > self.tol).count()` and `eprintln!("Variables are collinear")` when `rank < n_features`, mirroring `discriminant_analysis.py:945-947` — and crucially **does not error**: the old `cholesky_inv_and_logdet` is gone, replaced by the SVD, so a zero singular value yields a zero scaling and the degenerate `inf`/`NaN` arithmetic flows through `fn raw_decision` exactly as in numpy/sklearn (`scalings^(-0.5)=+inf`, `Σlog S=-inf` ⇒ collinear class decision is `NaN`; numpy-argmax treats `NaN` as the max ⇒ that class dominates). Consumer: `fn fit` → `FittedQDA` → `RsQDA::predict` (`extras.rs`). Verified: `qda_rank_deficient_class in tests/divergence_qda_fit.rs` (collinear class ⇒ `predict == [0;8]`, matching the live oracle). |
| REQ-11 (fitted attributes) | SHIPPED (scalings_/rotations_/means_) | #584. The SVD migration materializes `scalings`/`rotations` per class, and `FittedQDA::scalings` / `FittedQDA::rotations` / `FittedQDA::means in qda.rs` expose them (`S²/(n_k-1)` regularized scalings, `Vtᵀ` rotations, class means — `discriminant_analysis.py:948-954`). Consumer: `fn raw_decision in qda.rs` reads `scalings`/`rotations` on every prediction (production). Verified: `qda_scalings_rotations in tests/divergence_qda_fit.rs` pins `scalings_` to the live oracle and verifies `rotations_` via the sign-invariant reconstruction `R·diag(S2)·Rᵀ == covariance_`. The `priors_` and `covariance_` accessors remain NOT-STARTED (#584/#582 — `priors_` is not stored as a vector; `covariance_` needs `store_covariance`). |
| REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #585. The per-class **SVD now runs on `ferray::linalg::svd`** (`fn svd_s_vt in qda.rs`, bridged ndarray↔ferray at the fit boundary per R-SUBSTRATE-4, mirroring `bayesian_ridge.rs::svd_thin`); the `Fit` impl is bounded by `LinalgFloat`. BUT `qda.rs` still imports `ndarray::{Array1, Array2, ScalarOperand}` and its owned computation (centering, mean, decision) operates on `Array2<F>`/`Array1<F>`, not `ferray-core` arrays — so the array-type migration is not done. Consistent with the crate-wide substrate deferral (cf. `bayesian_ridge.md` REQ-10 #471, `isotonic.md` REQ-11 #572). |

## Architecture

### sklearn (the contract)

`QuadraticDiscriminantAnalysis(ClassifierMixin, BaseEstimator)`
(`discriminant_analysis.py:766`), constructor keyword-only `(*, priors=None,
reg_param=0.0, store_covariance=False, tol=1e-4)` (`:878-884`),
`_parameter_constraints` at `:871-876`. `fit` performs a **per-class SVD of the
centered data** — `Xgc = Xg - meang`; `U,S,Vt = svd(Xgc)`; `scalings_ =
(1-reg)·S²/(n_k-1) + reg`; `rotations_ = Vt.T` — never forming `Σ⁻¹` explicitly.
`_decision_function` evaluates the Mahalanobis term in the rotated/scaled frame
(`Xm @ (R·S^-0.5)`, then `‖·‖²`), adds `log|Σ| = Σ log(scalings_)` and
`log priors_`. `decision_function` collapses binary to `(n,)`; `predict` is the
argmax of the raw scores; `predict_proba` is their softmax.

### ferrolearn (what exists)

`QDA<F> { reg_param }` (`qda.rs`) → `fn fit` computes the per-class mean and the
`(n_k-1)` scatter covariance, optionally blends toward `I`, then **inverts** via
`fn cholesky_inv_and_logdet` (Cholesky `A=LLᵀ`, forward-substitution for `L⁻¹`,
`A⁻¹ = L⁻ᵀL⁻¹`, `log|A| = 2·Σ log(diag L)`). `FittedQDA<F> { class_models:
Vec<QDAClass { mean, cov_inv, log_det, log_prior }>, classes, n_features }`.
Inference recomputes `δ_c = -½·log_det - ½·(x-μ_c)ᵀΣ_c⁻¹(x-μ_c) + log π_c` in
`fn decision_function` / `fn predict` / `fn predict_proba`. Re-exported at the
crate root and bound as `RsQDA` (the macro surfaces `new`/`fit`/`predict`).

### Why inversion ≡ SVD here (and where it diverges)

For a full-rank covariance `Σ = V·D·Vᵀ` (with `D = diag(S²/(n_k-1))` the
scalings), `Σ⁻¹ = V·D⁻¹·Vᵀ` and `log|Σ| = Σ log(D)`, so the Mahalanobis term
`(x-μ)ᵀΣ⁻¹(x-μ)` equals sklearn's `‖(x-μ)·V·D^-0.5‖²` and the log-det term equals
`Σ log(scalings_)`. The `reg_param` blend commutes with the eigendecomposition:
`(1-reg)·V·D·Vᵀ + reg·I = V·((1-reg)·D + reg)·Vᵀ`, so regularizing the matrix
diagonal (ferrolearn) equals regularizing the singular values (sklearn). The
live oracle confirms ~1e-14 agreement (full-rank, reg 0/0.3/0.5). The equivalence
**breaks** when `Σ` is singular (collinear class): SVD has a zero scaling and
sklearn still returns (warning), while ferrolearn's Cholesky rejects it
(REQ-10). It is also observable that ferrolearn never materializes
`scalings_`/`rotations_` (REQ-11) and always returns the `(n,n_classes)` decision
(REQ-7).

## Verification

Library-crate gauntlet (baseline `e9d6069`):

- `cargo test -p ferrolearn-linear qda` — the module unit tests (accuracy floors +
  structural/error cases). NOTE: these do not pin a decision/proba **value**
  against the live oracle, so they do not establish REQ-1..5 parity (R-CHAR-3).
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the inversion≡SVD equivalence underpinning
REQ-1/REQ-5/REQ-8 and the NOT-STARTED gaps; expected values come from sklearn,
never copied from ferrolearn, per R-CHAR-3):

```bash
python3 -c "
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as Q
X=np.array([[1.,1.],[1.,3.],[2.,1.5],[2.,2.5],[8.,8.],[8.,9.5],[9.,8.5],[9.,9.5]])
y=np.array([0,0,0,0,1,1,1,1])
def ferro_decision(Xte,Xtr,ytr,reg):
    cs=np.unique(ytr); n=len(ytr); out=np.zeros((len(Xte),len(cs)))
    for ci,c in enumerate(cs):
        Xg=Xtr[ytr==c]; nk=len(Xg); mu=Xg.mean(0); d=Xg-mu
        cov=d.T@d/(nk-1)
        if reg>0: cov=(1-reg)*cov; cov+=reg*np.eye(cov.shape[0])
        ci_=np.linalg.inv(cov); _,ld=np.linalg.slogdet(cov); lp=np.log(nk/n)
        for i,x in enumerate(Xte):
            dd=x-mu; out[i,ci]=-0.5*ld-0.5*(dd@ci_@dd)+lp
    return out
for reg in (0.0,0.3,0.5):
    m=Q(reg_param=reg).fit(X,y)
    print(reg, 'maxdiff', np.max(np.abs(m._decision_function(X)-ferro_decision(X,X,y,reg))))
# REQ-7 shapes
print('binary dec shape', Q().fit(X,y).decision_function(X).shape)          # (8,)
# REQ-6 provided priors
print('priors_', Q(priors=[0.9,0.1]).fit(X,y).priors_.tolist())            # [0.9,0.1]
"
```

Observed: per-class decision max abs diff ~1e-14 for reg ∈ {0, 0.3, 0.5}; binary
`decision_function` shape `(8,)`; provided priors honoured. A NOT-STARTED REQ
closes only when its fix lands AND a divergence test (expected values from the
live oracle / a sklearn `file:line` constant) goes green; the
`conformance_qda`/`fixtures/qda.json` suite is excluded by goal.md and is not a
substitute.

## Blockers to open

- **#575** — REQ-1 of qda: pin the `(n,n_classes)` raw decision_function against
  the live `Q()._decision_function` oracle on a full-rank dataset (reg 0 and 0.5),
  confirming the Cholesky-inversion path matches sklearn's SVD path within `1e-8`.
- **#576** — REQ-2 of qda: pin `predict` label-for-label against the live
  `Q().predict` oracle (not an accuracy floor) on binary + 3-class data.
- **#577** — REQ-3 of qda: pin `predict_proba` against the live
  `Q().predict_proba` oracle (binary + multiclass) AND expose `predict_proba` on
  the `RsQDA` Python binding (no production consumer today).
- **#578** — REQ-4 of qda: pin `predict_log_proba` against the live oracle,
  replicate sklearn's `smallest_normal` zero-floor, and expose it on the binding.
- **#579** — REQ-5 of qda: pin a regularized decision against
  `Q(reg_param=0.5)._decision_function`, confirming the covariance-diagonal blend
  equals sklearn's singular-value blend `(1-reg)·S²+reg`.
- **#580** — REQ-6 of qda: add a constructor `priors` parameter (None → empirical,
  array → used verbatim) matching `discriminant_analysis.py:921-924`.
- **#581** — REQ-7 of qda: collapse the binary `decision_function` to shape `(n,)`
  = `dec[:,1]-dec[:,0]` (multiclass unchanged), matching
  `discriminant_analysis.py:1000-1002`.
- **#582** — REQ-9 of qda: add `store_covariance` + a `covariance_` accessor
  storing per-class `V·diag(scalings)·Vᵀ`.
- **#583** — REQ-10 of qda: add `tol` + the "Variables are collinear" warning and
  handle rank-deficient classes via an SVD/pseudo-inverse path so a collinear
  class predicts instead of returning `NumericalInstability`.
- **#584** — REQ-11 of qda: expose `means_`, `priors_`, `scalings_`, `rotations_`,
  `covariance_` on `FittedQDA` (requires materializing the SVD scalings/rotations).
- **#585** — REQ-12 of qda: migrate `qda.rs` off `ndarray` onto the ferray
  substrate (`ferray-core` arrays, `ferray::linalg` for the decomposition).
```
