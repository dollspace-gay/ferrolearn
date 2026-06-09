# Kernel Ridge Regression

<!--
tier: 3-component
status: draft
baseline-commit: 4ccf98d2
upstream-paths:
  - sklearn/kernel_ridge.py
  - sklearn/linear_model/_ridge.py        # _solve_cholesky_kernel (solve semantics)
  - sklearn/metrics/pairwise.py           # pairwise_kernels / PAIRWISE_KERNEL_FUNCTIONS
-->

## Summary

`ferrolearn-kernel/src/kernel_ridge.rs` mirrors scikit-learn's
`sklearn.kernel_ridge.KernelRidge` — ridge regression combined with the kernel
trick, solved in closed form in the dual (kernel) space. sklearn computes the
kernel via `pairwise_kernels` (`sklearn/metrics/pairwise.py`) and delegates the
solve to `_solve_cholesky_kernel` (`sklearn/linear_model/_ridge.py:226`).
ferrolearn computes `K = kernel(X, X)` (`compute_kernel_matrix`/`kernel_value`
in `nystroem.rs`), adds `alpha` to the diagonal, and solves
`(K + alpha*I) dual_coef = y` via `fn cholesky_solve` with a `fn gaussian_solve`
fallback, then predicts `K(X, X_fit) @ dual_coef`.

The current implementation is value-exact against the live sklearn 1.5.2 oracle
for the **single-output, scalar-alpha, no-sample-weight** case with the
`linear` and `rbf` kernels (verified element-wise to ~1e-13 below). The
remaining sklearn surface — the `coef0=1` default, multi-output `y`,
`sample_weight`, per-target array `alpha`, the kernels beyond
linear/rbf/poly/sigmoid, `precomputed`, callable kernels, `kernel_params`, the
`X_fit_`/`n_features_in_` attribute names, full parameter validation, and the
ferray substrate — is NOT-STARTED.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/kernel_ridge.py` — `class KernelRidge` (constructor `:146-161`,
  `_parameter_constraints` `:134-144`, `_get_kernel` `:163-168`, `fit` `:174`,
  `predict` `:218`).
- `sklearn/linear_model/_ridge.py:226` — `_solve_cholesky_kernel(K, y, alpha,
  sample_weight, copy)`: `K.flat[::n_samples+1] += alpha[0]` then
  `linalg.solve(K, y, assume_a="pos")` with an `lstsq` fallback on
  `LinAlgError` (`:245-268`).
- `sklearn/metrics/pairwise.py` — `PAIRWISE_KERNEL_FUNCTIONS`,
  `pairwise_kernels`.

## Requirements

- REQ-1: Closed-form dual solve. `fit` forms `K`, adds `alpha` to the diagonal,
  and solves `(K + alpha*I) dual_coef = y`; `predict` returns
  `K(X, X_fit) @ dual_coef`. Mirrors `_solve_cholesky_kernel`
  (`_ridge.py:226`, `:247`, `:253`) and `predict` (`kernel_ridge.py:237`).
- REQ-2: Linear-kernel value parity (single-output, scalar alpha,
  no sample_weight) — `dual_coef_` and predictions match sklearn element-wise.
- REQ-3: RBF-kernel value parity (single-output, scalar alpha,
  no sample_weight), including the `gamma=None → 1/n_features` default
  (`kernel_ridge.py:66-70`; pairwise rbf default).
- REQ-4: Polynomial-kernel formula `(gamma·<x,y> + coef0)^degree` and
  sigmoid-kernel formula `tanh(gamma·<x,y> + coef0)` match sklearn's pairwise
  definitions when `coef0`/`degree` are set explicitly.
- REQ-5: `coef0` constructor default equals sklearn's `coef0=1`
  (`kernel_ridge.py:153`), so poly/sigmoid kernels match at default params.
- REQ-6: Multi-output `y` of shape `(n_samples, n_targets)`, ravelled back to
  1-D when 1-D was passed (`kernel_ridge.py:204-212`).
- REQ-7: `sample_weight` support (`kernel_ridge.py:174,198-199`;
  `_ridge.py:238-243,265-266`).
- REQ-8: Per-target array `alpha` of shape `(n_targets,)`
  (`kernel_ridge.py:202`; `_ridge.py:269-285`).
- REQ-9: Full kernel coverage — `laplacian`, `chi2`, `additive_chi2`,
  `cosine`, plus `precomputed`, callable kernels, and `kernel_params`
  (`kernel_ridge.py:52-64,79-81,134-144,163-168`).
- REQ-10: Parameter validation matching `_parameter_constraints`
  (`kernel_ridge.py:134-144`): `alpha ≥ 0`, `gamma ≥ 0 or None`,
  `degree ≥ 0`, `coef0` any real.
- REQ-11: Fitted-attribute contract — expose `dual_coef_`, `X_fit_`, and
  `n_features_in_` with sklearn's names (`kernel_ridge.py:85-94`).
- REQ-12: ferray substrate (R-SUBSTRATE) — array type, linear-algebra solve,
  and PyO3 bridge on `ferray-core` / `ferray::linalg` / `ferray::numpy_interop`
  rather than `ndarray` + hand-rolled `cholesky_solve`/`gaussian_solve`.
- REQ-13: Non-test production consumer — re-exported and exposed through the
  Python binding so `import ferrolearn` mirrors `import sklearn`.

## Acceptance criteria

- AC-1 (REQ-1): `fit` then `predict` reconstructs `K(X,X_fit) @ dual_coef`;
  `dual_coef` solves the regularized normal equations to machine precision
  (deterministic; oracle-pinnable).
- AC-2 (REQ-2): `dual_coef_` and `predict(X)` for `kernel='linear'`,
  single-output `y`, scalar `alpha` match `sklearn.kernel_ridge.KernelRidge`
  element-wise within 1e-10 (deterministic; oracle-pinnable).
- AC-3 (REQ-3): same for `kernel='rbf'` with both explicit `gamma` and the
  `gamma=None` default (deterministic; oracle-pinnable).
- AC-4 (REQ-4): `kernel_value` for poly/sigmoid equals
  `sklearn.metrics.pairwise.{polynomial,sigmoid}_kernel` element-wise with
  matching `gamma/degree/coef0` (deterministic; oracle-pinnable).
- AC-5 (REQ-5): `KernelRidge::new().with_kernel(Polynomial)` (no `with_coef0`)
  matches `KernelRidge(kernel='poly')` (which defaults `coef0=1`)
  (deterministic; oracle-pinnable — currently FAILS, see status).
- AC-6 (REQ-6): fitting `(n_samples, n_targets)` `y` yields a
  `(n_samples, n_targets)` `dual_coef_` matching sklearn; 1-D `y` ravels back.
- AC-7 (REQ-7): `fit(X, y, sample_weight=w)` matches sklearn's weighted solve.
- AC-8 (REQ-8): array `alpha` of length `n_targets` solves each target with its
  own penalty (`_ridge.py:269-285`).
- AC-9 (REQ-9): each added kernel / `precomputed` / callable matches the
  corresponding `pairwise_kernels` output; unknown kernel strings raise the
  sklearn-equivalent error.
- AC-10 (REQ-10): out-of-domain `gamma`/`degree` are rejected with a
  `ValueError`-equivalent (`FerroError::InvalidParameter`), as sklearn's
  `_parameter_constraints` does.
- AC-11 (REQ-11): a fitted model exposes `X_fit_`, `dual_coef_`, and
  `n_features_in_` (sklearn attribute names).
- AC-12 (REQ-12): no `ndarray`/`faer`-direct/hand-rolled-solver usage in the
  owned computation; solve routes through `ferray::linalg`.
- AC-13 (REQ-13): `ferrolearn::KernelRidge` (Python) fits/predicts via the
  Rust core through `RsKernelRidge`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (dual solve) | SHIPPED | `pub fn fit` in `kernel_ridge.rs` forms `K` via `compute_kernel_matrix`, does `k[[i, i]] = k[[i, i]] + self.alpha`, then `cholesky_solve(&k, y).or_else(|_| gaussian_solve(&k, y)).unwrap_or_else(|_| lstsq_min_norm_solve(&k, y))` — mirrors `_solve_cholesky_kernel` (`_ridge.py:247` `K.flat[::n_samples+1] += alpha[0]`, `:253` `linalg.solve(K, y, assume_a="pos")` with `:254-259` `lstsq` fallback on `LinAlgError`). The lstsq fallback is now value-matching: `fn lstsq_min_norm_solve` computes the minimum-norm least-squares solution via a symmetric (Jacobi) eigendecomposition pseudo-inverse with cutoff `n·eps·max\|λ\|` (= `scipy.linalg.lstsq` default `cond=None` rank cutoff `max(M,N)·eps·max(s)`). Singular kernels (`alpha=0` + duplicate rows) that previously returned `Err(NumericalInstability)` now succeed with sklearn's min-norm `dual_coef_` (tests `divergence_singular_kernel_lstsq_fallback{,_min_norm}`, live oracle `~1e-9`). The well-conditioned (positive-definite) path is unchanged — Cholesky still wins, bit-exact. `predict` returns `k_new.dot(&self.dual_coef)` = `kernel_ridge.py:237` `np.dot(K, self.dual_coef_)`. Non-test consumer: `ferrolearn-python/src/extras.rs` (`RsKernelRidge` → `KernelRidge::<f64>::new()`), re-exported in `lib.rs` (`pub use kernel_ridge::{FittedKernelRidge, KernelRidge}`). Verification: oracle parity below; `cargo test -p ferrolearn-kernel` 301 passed, 0 failed. Warning-absent gap: sklearn emits a "Singular matrix in solving dual problem" `warnings.warn` (`_ridge.py:255-258`); ferrolearn has no warning facade, so the fit succeeds silently (documented divergence, not a value gap). |
| REQ-2 (linear parity) | SHIPPED | `KernelType::Linear` in `kernel_value` returns `<x,y>`. Live oracle (`alpha=0.5`, `kernel='linear'`, `X=[[0],[1],[2],[3],[4]]`, `y=[0,1,4,9,16]`): sklearn `dual_coef_ = [0.0, -4.55737704918033, -5.1147540983606525, -1.6721311475409892, 5.770491803278692]`; ferrolearn `[0.0, -4.5573770491803325, -5.114754098360658, -1.6721311475409832, 5.7704918032786905]` — match to ~1e-13. `predict` likewise matches. Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-3 (rbf parity) | SHIPPED | `KernelType::Rbf` in `kernel_value` returns `exp(-gamma·||x-y||²)`; `fit` resolves `gamma=None → 1/n_features` (`self.gamma.unwrap_or_else(|| F::one()/F::from(n_features))`), matching `pairwise` rbf default. Live oracle (`alpha=0.5`, `gamma=0.5`): sklearn `dual_coef_[4]=9.953434595974752`, ferrolearn `9.953434595974752`; default-gamma case also matches to ~1e-13 (full vectors in Verification). Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-4 (poly/sigmoid formula) | SHIPPED | `KernelType::Polynomial` computes `(gamma·dot + coef0)^degree` and `KernelType::Sigmoid` computes `(gamma·dot + coef0).tanh()` in `kernel_value` — identical to sklearn's `polynomial_kernel`/`sigmoid_kernel`. The kernel *value* formula matches when `gamma/degree/coef0` are supplied explicitly; the **default** mismatch is REQ-5, not this REQ. Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-5 (coef0 default = 1) | NOT-STARTED | tracking #1661 (critic to file specific blocker). `KernelRidge::new` sets `coef0: F::zero()` (`kernel_ridge.rs`), but sklearn defaults `coef0=1` (`kernel_ridge.py:153`). The doc-comment is self-contradictory ("Coefficient ... default 0.0" vs "match scikit-learn"). Live oracle confirms divergence: `KernelRidge(alpha=0.5, kernel='poly')` (default `coef0=1`) `dual_coef_=[0.0343..., -0.1559..., ...]` vs `coef0=0` `dual_coef_=[0.0, 1.4683..., 3.7468..., ...]` — different results at default params for poly/sigmoid. Oracle-pinnable. |
| REQ-6 (multi-output y) | NOT-STARTED | tracking #1661. `impl Fit<Array2<F>, Array1<F>>` accepts only 1-D `y`; sklearn supports `(n_samples, n_targets)` and ravels back (`kernel_ridge.py:204-212`). `dual_coef` is `Array1<F>` (no target axis). Blocker: add `Fit<Array2<F>, Array2<F>>` / 2-D `dual_coef_`. |
| REQ-7 (sample_weight) | NOT-STARTED | tracking #1661. `fn fit(&self, x, y)` has no `sample_weight` parameter; sklearn `fit(X, y, sample_weight=None)` (`kernel_ridge.py:174`) scales `y` and `K` by `sqrt(sw)` (`_ridge.py:241-243`) and post-scales `dual_coef` (`:265-266`). The `Fit` trait signature carries no weight slot. |
| REQ-8 (array alpha) | NOT-STARTED | tracking #1661. `alpha: F` is scalar; sklearn allows `(n_targets,)` (`kernel_ridge.py:202` `np.atleast_1d(self.alpha)`; per-target loop `_ridge.py:269-285`). Tied to REQ-6 (multi-output). |
| REQ-9 (kernel coverage) | NOT-STARTED | tracking #1661. `enum KernelType` (`nystroem.rs`) has only `{Rbf, Polynomial, Linear, Sigmoid}`. sklearn accepts all `PAIRWISE_KERNEL_FUNCTIONS` (adds `laplacian`, `chi2`, `additive_chi2`, `cosine`) plus `"precomputed"`, callable kernels, and `kernel_params` (`kernel_ridge.py:52-64,79-81,134-144,163-168`). Missing kernels + precomputed + callable + kernel_params are absent. |
| REQ-10 (param validation) | NOT-STARTED | tracking #1661. `fit` rejects `alpha < 0` (`FerroError::InvalidParameter`), matching `_parameter_constraints` `alpha ≥ 0` (`kernel_ridge.py:135`). But `gamma ≥ 0` (`:140`) and `degree ≥ 0` (`:141`) are NOT validated; `degree` is `usize` (≥0 by type but no explicit check), and a negative explicit `gamma` is accepted silently. Partial coverage → NOT-STARTED until gamma/degree validation matches. |
| REQ-11 (fitted attributes — `n_features_in_`) | SHIPPED | `FittedKernelRidge` exposes `dual_coef()` (`dual_coef_`) + `x_fit()` (`X_fit_`); added `#[must_use] pub fn n_features_in(&self) -> usize` (= `x_fit.ncols()`, sklearn `n_features_in_` `kernel_ridge.py:93`). Verification (live sklearn 1.5.2, R-CHAR-3, `X` 3×2): `n_features_in_ == 2`; the `n_features_in() == x_fit().ncols()` invariant holds. Test `kernel_ridge_n_features_in_matches_sklearn`. (The PyO3 re-export of `X_fit_`/`n_features_in_` rides the separate binding/substrate REQ.) |
| REQ-12 (ferray substrate) | NOT-STARTED | tracking #1661 (R-SUBSTRATE). `kernel_ridge.rs` uses `ndarray::{Array1, Array2}` and hand-rolled `fn cholesky_solve`/`fn gaussian_solve`; the kernel/eigen path in `nystroem.rs` uses `faer` directly. Destination substrate is `ferray-core` (array) + `ferray::linalg` (solve) per R-SUBSTRATE-1. Not migrated. |
| REQ-13 (consumer) | SHIPPED | `lib.rs` re-exports `pub use kernel_ridge::{FittedKernelRidge, KernelRidge}`. Python binding: `ferrolearn-python/src/extras.rs` (`RsKernelRidge` → `ferrolearn_kernel::KernelRidge::<f64>::new().with_alpha(alpha)`), registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsKernelRidge>()`), wrapped by `ferrolearn-python/python/ferrolearn/_extras.py` (`class KernelRidge(_RegressorWrapper)`), exported in `__init__.py`. Note: the binding exposes only `alpha` — `kernel`/`gamma`/`degree`/`coef0` are NOT plumbed through, so the Python surface is narrower than sklearn's constructor (sub-finding tracked under REQ-9/REQ-11). |

## Architecture

ferrolearn splits the estimator into the unfitted builder `KernelRidge<F>`
(fields `alpha`, `kernel`, `gamma: Option<F>`, `degree`, `coef0`) and the
fitted `FittedKernelRidge<F>` (`x_fit`, `dual_coef: Array1<F>`, plus the kernel
hyperparameters), matching sklearn's `KernelRidge` / post-`fit` attribute split.
sklearn keeps one class and sets `dual_coef_`/`X_fit_` on `self` in `fit`
(`kernel_ridge.py:210,214`).

The kernel evaluation is shared with `Nystroem`: `KernelType` (defined in
`nystroem.rs`) enumerates `{Rbf, Polynomial, Linear, Sigmoid}`, and
`compute_kernel_matrix`/`kernel_value` (also in `nystroem.rs`) implement the
four formulas. sklearn instead routes every kernel through
`pairwise_kernels(... metric=self.kernel ...)` (`kernel_ridge.py:168`), which
admits the full `PAIRWISE_KERNEL_FUNCTIONS` set plus `precomputed` and
callables — the source of REQ-9's gap.

The solve is the contract's numerical core. ferrolearn's `fn cholesky_solve`
performs an explicit `L Lᵀ` factorization and forward/back substitution,
returning `NumericalInstability` if a diagonal pivot is `≤ 0` (non-PD); on that
error `fn fit` falls back to `fn gaussian_solve` (partial-pivot Gaussian
elimination), and if that also fails (a genuinely singular kernel — e.g.
`alpha=0` with duplicate rows) to `fn lstsq_min_norm_solve`. sklearn calls
`linalg.solve(K, y, assume_a="pos")` (a LAPACK Cholesky) and on `LinAlgError`
falls back to `linalg.lstsq` (`_ridge.py:253-259`). For the PD case (the common
path, and every value-parity check above) both take the Cholesky branch and
agree to ~1e-13. For the singular case ferrolearn now matches sklearn's
`linalg.lstsq` minimum-norm least-squares result: `fn lstsq_min_norm_solve`
forms the symmetric eigendecomposition `K = V diag(λ) Vᵀ` (cyclic Jacobi,
unconditionally convergent for real symmetric matrices), drops eigenvalues with
`|λ| ≤ n·eps·max|λ|` (matching `scipy.linalg.lstsq`'s `cond=None` rank cutoff
`max(M,N)·eps·max(s)`; `K` square so `M==N==n`), and returns
`x = Σ_{|λ_i|>cutoff} (vᵢᵀy / λ_i) vᵢ`. The lstsq path is a fallback only — the
well-conditioned `dual_coef_` is bit-exact-unchanged. sklearn additionally emits
a `warnings.warn` on the singular branch (`_ridge.py:255-258`); ferrolearn has no
warning facade so the fit succeeds silently (documented divergence). The
`gamma=None` default resolves identically (`1/n_features`).

Invariants: `K` is symmetric PSD; adding `alpha ≥ 0` to the diagonal makes it PD
for `alpha > 0`; `predict` requires `X.ncols() == X_fit.ncols()` and otherwise
returns `ShapeMismatch`.

## Verification

Commands establishing the SHIPPED claims (run at baseline `4ccf98d2`):

- `cargo test -p ferrolearn-kernel --lib` → 192 passed, 0 failed (REQ-1, REQ-4).
- Live sklearn oracle (REQ-2, REQ-3), `X=[[0],[1],[2],[3],[4]]`, `y=[0,1,4,9,16]`:
  - `python3 -c "from sklearn.kernel_ridge import KernelRidge; ..."`
    `KernelRidge(alpha=0.5, kernel='linear')` →
    `dual_coef_ = [0.0, -4.55737704918033, -5.1147540983606525, -1.6721311475409892, 5.770491803278692]`;
    ferrolearn `[0.0, -4.5573770491803325, -5.114754098360658, -1.6721311475409832, 5.7704918032786905]`.
  - `KernelRidge(alpha=0.5, kernel='rbf', gamma=0.5)` →
    `dual_coef_ = [-0.1358903569851363, 0.04557666340439703, 1.1536231760735174, 1.5057135979384975, 9.953434595974752]`;
    ferrolearn `[-0.13589035698513627, 0.04557666340439698, 1.1536231760735178, 1.505713597938496, 9.953434595974752]`.
  - `KernelRidge(alpha=0.5, kernel='rbf')` (default gamma) →
    `dual_coef_ = [-0.07633659717810444, 0.22464834449569865, 1.7182558136275892, 3.1547282092376903, 9.871961120746237]`;
    ferrolearn `[-0.07633659717810445, 0.22464834449569876, 1.718255813627589, 3.154728209237691, 9.871961120746237]`.
- coef0-default divergence (REQ-5, NOT-STARTED, oracle-pinnable):
  `KernelRidge(alpha=0.5, kernel='poly')` (default `coef0=1`) →
  `dual_coef_ = [0.03432932041431303, -0.15597606574755987, 0.009205422932366645, 0.16174855663851495, -0.06647189444479128]`
  vs `coef0=0` →
  `[0.0, 1.4683570187097572, 3.7468561496779755, 3.645639505163571, -2.0251508025766385]`;
  ferrolearn's `new()` uses `coef0=0`, so it matches the wrong branch.

A critic-pinned value-parity `#[test]` (expected values from the live oracle
above, never copied from ferrolearn) should pin REQ-2/REQ-3 green and REQ-5 as a
failing test until `new()`'s `coef0` is fixed to `F::one()`.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1, REQ-2,
REQ-3, REQ-4, REQ-13 (impl + Python-binding consumer + green verification).
NOT-STARTED (tracking #1661; the critic files per-REQ blockers): REQ-5
(coef0 default), REQ-6 (multi-output), REQ-7 (sample_weight), REQ-8 (array
alpha), REQ-9 (kernel coverage / precomputed / callable / kernel_params),
REQ-10 (gamma/degree validation), REQ-11 (attribute names + n_features_in_),
REQ-12 (ferray substrate).
