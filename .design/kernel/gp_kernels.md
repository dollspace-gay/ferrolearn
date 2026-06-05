# Gaussian-Process Covariance Kernels

<!--
tier: 3-component
status: draft
baseline-commit: d853cf241
upstream-paths:
  - sklearn/gaussian_process/kernels.py    # Kernel base (:153), Hyperparameter (:52), RBF (:1445), Matern (:1598), ConstantKernel (:1184), WhiteKernel (:1322), DotProduct (:2099), Sum (:796), Product (:893), RationalQuadratic (:1798), ExpSineSquared (:1954), Exponentiation (:993), CompoundKernel (:514)
-->

## Summary

`ferrolearn-kernel/src/gp_kernels.rs` mirrors scikit-learn's GP covariance
kernels in `sklearn/gaussian_process/kernels.py` — the `Kernel` family used to
build covariance matrices for `GaussianProcessRegressor`/`Classifier`. ferrolearn
exposes the `GPKernel` trait (`compute`/`diagonal`/`n_params`/`get_params`/
`set_params`/`clone_box`) and seven kernels: `RBFKernel`, `MaternKernel`,
`ConstantKernel`, `WhiteKernel`, `DotProductKernel`, `SumKernel`, `ProductKernel`.
The non-test production consumer is the in-crate GPR/GPC: `compute`/`diagonal`
drive the covariance-matrix construction in `gaussian_process.rs`
(`fn fit`/`fn predict`) and `gp_classifier.rs` (`fn fit_binary_gpc`/`fn predict`).
There is NO Python binding for GP — the consumer is wholly the Rust GPR/GPC.

The **base kernel-matrix formulas** for RBF, Matern (`nu ∈ {0.5, 1.5, 2.5}`),
Constant, DotProduct, Sum and Product are value-exact against the live sklearn
1.5.2 oracle and have a real consumer — these are SHIPPED. GP kernels are
deterministic, so every divergence below is oracle-pinnable *now* (unlike the
RNG/generator carve-outs in `nystroem.md`); the gaps are missing-feature
blockers, not value-of-existing-output divergences.

The gaps (all NOT-STARTED, all oracle-pinnable):

1. **`eval_gradient` / analytic `dK/dθ` — MISSING (major).** Every sklearn
   `__call__(X, eval_gradient=True)` returns `(K, K_gradient)` with
   `K_gradient.shape == (n, n, n_dims)` (RBF `:1569-1582`, Matern `:1742-1781`,
   Constant `:1280-1291`, White `:1405-1412`, DotProduct, Sum, Product). The
   `GPKernel` trait has NO gradient method — `compute`/`diagonal` only.
2. **`theta`/`bounds`/`Hyperparameter` machinery — MISSING.** sklearn's `Kernel`
   base exposes `theta` (flattened log-transformed non-fixed hyperparameters,
   `:287-309`), `bounds` (`:342-358`), `hyperparameters`, `clone_with_theta`,
   `n_dims`, plus per-param `Hyperparameter` objects (`:52`) carrying bounds and
   a `fixed` flag. ferrolearn has `get_params`/`set_params`/`n_params` returning
   log-space values but NO bounds, NO `Hyperparameter` objects, NO `fixed`
   support.
3. **Matern unsupported-`nu` fallback (DIVERGENCE).** ferrolearn's
   `MaternKernel::compute` silently returns an RBF formula for `nu ∉ {0.5, 1.5,
   2.5}`. sklearn evaluates general `nu` via the modified Bessel function
   (`:1729-1735`) and `nu=inf` → RBF (`:1727-1728`).
4. **WhiteKernel `Y is None` semantics (DIVERGENCE).** sklearn: `Y is None` →
   `noise_level*eye(n)` (`:1404`); `Y is not None` → `zeros((n,m))` (`:1416`).
   ferrolearn's `compute(x1, x2)` instead checks ROW EQUALITY, so
   `compute(X, X)` returns `noise*I` where sklearn's `kernel(X, Y=X)` returns
   zeros.
5. **Anisotropic `length_scale` — MISSING.** sklearn RBF/Matern accept an
   array `length_scale` (`:1472-1475`, `anisotropic` property `:1512-1514`);
   ferrolearn holds a scalar `length_scale: F`.
6. **Missing kernels.** `RationalQuadratic` (`:1798`), `ExpSineSquared`
   (`:1954`), `Exponentiation` (`:993`), `CompoundKernel` (`:514`) have NO
   ferrolearn analog.
7. **Constructor defaults / `Default`.** sklearn defaults: RBF/Matern
   `length_scale=1.0` (`:1508`, `:1678`), Matern `nu=1.5` (`:1678`),
   ConstantKernel `constant_value=1.0` (`:1233`), DotProduct `sigma_0=1.0`
   (`:2156`), WhiteKernel `noise_level=1.0`. ferrolearn's `new()` requires an
   explicit value (no sklearn-default `Default` impl). The `theta`/`get_params`
   log-space ordering for Sum/Product (k1 then k2) MATCHES sklearn.
8. **Substrate (R-SUBSTRATE-1).** `ndarray` instead of `ferray-core` /
   `ferray::linalg`.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/gaussian_process/kernels.py`
  - `class Kernel` `:153` — base. `theta` `:287-309` (flattened log non-fixed
    hyperparameters), `bounds` `:342-358`, `hyperparameters` `:277-285`,
    `n_dims` `:272-275`, `clone_with_theta`, `set_params`/`get_params`.
  - `class Hyperparameter` `:52` — `name`, `value_type`, `bounds`, `n_elements`,
    `fixed` (a `NamedTuple` carrying per-param bounds + fixed flag).
  - `class RBF` `:1445` — `__init__(length_scale=1.0,
    length_scale_bounds=(1e-5,1e5))` `:1508`; `anisotropic` `:1512`;
    `__call__` `:1527`: `K = exp(-0.5 * sqdist(X/l))` (`:1558-1562`, with
    `Y is None` filling the diagonal to 1); `eval_gradient` branch
    `:1569-1582` returns `K_gradient` of shape `(n,n,n_dims)`
    (`(K*sqdist)[:,:,newaxis]` isotropic `:1574`).
  - `class Matern(RBF)` `:1598` — `__init__(length_scale=1.0,
    length_scale_bounds=(1e-5,1e5), nu=1.5)` `:1678`; `__call__` `:1682`:
    `nu==0.5` → `exp(-dists)` `:1720`; `nu==1.5` → `(1+√3·d)exp(-√3·d)`
    `:1722-1723`; `nu==2.5` → `(1+√5·d+5d²/3)exp(-√5·d)` `:1725-1726`;
    `nu==inf` → `exp(-d²/2)` (RBF) `:1727-1728`; **general `nu`** via modified
    Bessel `kv(nu, √(2ν)·d)` `:1729-1735`; gradient branch `:1742-1781`.
  - `class ConstantKernel` `:1184` — `__init__(constant_value=1.0, ...)` `:1233`;
    `__call__` `:1241`: `K = full((n,m), constant_value)` `:1275`; gradient
    `:1280-1291`; `diag` returns `full(n, constant_value)`.
  - `class WhiteKernel` `:1322` — `__init__(noise_level=1.0, ...)`; `__call__`:
    `Y is None` → `noise_level*eye(n)` `:1404`; `Y is not None` →
    `zeros((n,m))` `:1416`; gradient `:1405-1412`; `diag` → `full(n,
    noise_level)` `:1435`.
  - `class DotProduct` `:2099` — `__init__(sigma_0=1.0, ...)` `:2156`;
    `__call__` `:2164`: `K = sigma_0² + X·Yᵀ` (`:2113` formula); gradient
    branch returns `2·sigma_0²` along the diagonal direction.
  - `class Sum` `:796` / `class Product` `:893` — `__call__` combines `k1`, `k2`
    matrices; `theta` concatenates `k1.theta` then `k2.theta`; `eval_gradient`
    `dstack`s the two sub-gradients.
  - `class RationalQuadratic` `:1798`, `class ExpSineSquared` `:1954`,
    `class Exponentiation` `:993`, `class CompoundKernel` `:514` — no ferrolearn
    analog.

## Requirements

- REQ-1: RBF kernel-matrix formula. `RBFKernel::compute` computes
  `exp(-‖x−x'‖²/(2·l²))` and `diagonal` returns 1, mirroring `RBF.__call__`
  (`:1558-1562`). (Deterministic / oracle-pinnable.)
- REQ-2: Matern `nu ∈ {0.5, 1.5, 2.5}` formulas. `MaternKernel::compute`
  computes `exp(-r/l)` (0.5), `(1+√3·r/l)exp(-√3·r/l)` (1.5),
  `(1+√5·r/l+5r²/3l²)exp(-√5·r/l)` (2.5), and `diagonal` returns 1 — mirroring
  `Matern.__call__` (`:1719-1726`). (Deterministic / oracle-pinnable.)
- REQ-3: ConstantKernel. `ConstantKernel::compute` returns
  `full((n,m), constant_value)` and `diagonal` returns `full(n, constant_value)`
  — mirrors `ConstantKernel.__call__`/`diag` (`:1275`). (Oracle-pinnable.)
- REQ-4: DotProductKernel formula. `DotProductKernel::compute` returns
  `sigma_0² + X·X'ᵀ` and `diagonal` returns `‖xᵢ‖²+sigma_0²` — mirrors
  `DotProduct.__call__` (`:2113`). (Oracle-pinnable.)
- REQ-5: Sum/Product composition. `SumKernel`/`ProductKernel` return
  `k1+k2`/`k1*k2` (matrix and diagonal), and `get_params` concatenates
  `k1.get_params()` then `k2.get_params()` — matching sklearn's `Sum`/`Product`
  `__call__` and `theta` concatenation order (`:796`, `:893`). (Oracle-pinnable.)
- REQ-6: log-space parameter round-trip. `get_params`/`set_params`/`n_params`
  expose each positive hyperparameter as `ln(value)` / `exp(θ)`, matching the
  log transform sklearn applies in `theta` (`:287-309`). (Oracle-pinnable.)
- REQ-7: GPR/GPC production consumer. The kernel covariance drives the in-crate
  GP estimators' fit/predict via `compute`/`diagonal`. (Non-test consumer.)
- REQ-8: `eval_gradient` / analytic `dK/dθ`. A trait method returning the
  per-log-hyperparameter gradient `K_gradient` of shape `(n,n,n_dims)`, mirroring
  every kernel's `eval_gradient=True` branch (RBF `:1569-1582`, Matern
  `:1742-1781`, Constant `:1280-1291`, White `:1405-1412`, DotProduct, Sum,
  Product). (Oracle-pinnable; required by GPR's LML optimizer.)
- REQ-9: `theta`/`bounds`/`Hyperparameter`/`fixed` machinery. Expose log-space
  `theta`, per-param `bounds` (`:342-358`), `Hyperparameter` objects (`:52`)
  with a `fixed` flag, `n_dims`, and `clone_with_theta`. (Oracle-pinnable on
  `bounds`/`n_dims`.)
- REQ-10: Matern general `nu` (Bessel) + `nu=inf`. `compute` evaluates the
  modified-Bessel Matern for `nu ∉ {0.5,1.5,2.5,inf}` (`:1729-1735`) and the RBF
  limit for `nu=inf` (`:1727-1728`), instead of silently returning RBF.
  (Oracle-pinnable.)
- REQ-11: WhiteKernel `Y is None` channel. The kernel distinguishes
  `kernel(X)` (`noise_level*I`) from `kernel(X, Y)` (`zeros((n,m))`,
  `:1416`) by a "Y is None" signal, instead of row equality. (Oracle-pinnable.)
- REQ-12: Anisotropic `length_scale`. RBF/Matern accept an array `length_scale`
  (one per feature, `:1472-1475`) with the `anisotropic` gradient path
  (`:1576-1582`). (Oracle-pinnable.)
- REQ-13: Missing kernels. `RationalQuadratic` (`:1798`), `ExpSineSquared`
  (`:1954`), `Exponentiation` (`:993`), `CompoundKernel` (`:514`).
  (Oracle-pinnable; mirrored surface, not out-of-scope.)
- REQ-14: Constructor defaults / `Default`. sklearn-matching defaults
  (RBF/Matern `length_scale=1.0`, Matern `nu=1.5`, `constant_value=1.0`,
  `sigma_0=1.0`, `noise_level=1.0`) exposed as a `Default` impl, matching
  sklearn's keyword defaults (`:1508`, `:1678`, `:1233`, `:2156`).
  (Oracle-pinnable.)
- REQ-15: ferray substrate (R-SUBSTRATE-1). Array type → `ferray-core`;
  distance/dot computations → `ferray-ufunc`/`ferray::linalg` — instead of
  `ndarray`.

## Acceptance criteria

- AC-1 (REQ-1): `RBFKernel::new(1.5).compute(X, X)` for
  `X=[[0,0],[1,0],[0,1]]` equals `RBF(length_scale=1.5)(X) =
  [[1, 0.800737, 0.800737],[0.800737, 1, 0.64118],[0.800737, 0.64118, 1]]`
  element-wise to ~1e-12 (live oracle, R-CHAR-3). `diagonal` ≈ 1.
- AC-2 (REQ-2): `MaternKernel::new(1.0, 1.5).compute(X, X)` equals
  `Matern(length_scale=1.0, nu=1.5)(X) = [[1, 0.483358, 0.483358], ...]`;
  `nu=2.5` → `[[1, 0.523994, 0.523994], ...]`; `nu=0.5` →
  `[[1, 0.367879, 0.367879], ...]` (live oracle, same `X`).
- AC-3 (REQ-3): `ConstantKernel::new(2.0).compute(X, X)` is all-2.0, equals
  `ConstantKernel(constant_value=2.0)(X)`; `diagonal` is all-2.0.
- AC-4 (REQ-4): `DotProductKernel::new(1.0).compute(X, X)` equals
  `DotProduct(sigma_0=1.0)(X) = [[1,1,1],[1,2,1],[1,1,2]]`.
- AC-5 (REQ-5): `SumKernel(Const(1), Const(2))` is all-3.0; `ProductKernel(
  Const(2), RBF(1))` equals `2·RBF(1)(X,X)`. `get_params()` of
  `Sum(RBF(1.5), White(0.1))` is `[ln 1.5, ln 0.1] = [0.405465, -2.302585]`,
  matching `(RBF(1.5)+WhiteKernel(0.1)).theta`.
- AC-6 (REQ-6): `RBFKernel::new(2.0).get_params() == [ln 2.0]`;
  `set_params(&[ln 1.0])` sets `length_scale = 1.0`. (Round-trip = sklearn
  `theta` log transform.)
- AC-7 (REQ-7): `GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)))`
  fits and predicts through `kernel.compute`/`diagonal` (`gaussian_process.rs`
  `fn fit` line `let mut k_mat = self.kernel.compute(x, x)`, `fn predict`
  `let k_star = self.kernel.compute(x, &self.x_train)`).
- AC-8 (REQ-8): `RBF(length_scale=1.0)(X, eval_gradient=True)` returns
  `K_gradient` of shape `(3,3,1)` with `K_gradient[:,:,0] = [[0, 0.606531,
  0.606531],[0.606531, 0, 0.735759],[0.606531, 0.735759, 0]]`. ferrolearn has
  no gradient method to compare. Oracle-pinnable once a trait method exists.
- AC-9 (REQ-9): `(ConstantKernel(2.0)*RBF(1.5)+WhiteKernel(0.1)).theta =
  [0.693147, 0.405465, -2.302585]`, `.bounds.shape == (3,2)`, `.n_dims == 3`.
  ferrolearn has no `bounds`/`n_dims`/`Hyperparameter`. Oracle-pinnable.
- AC-10 (REQ-10): `Matern(length_scale=1.0, nu=3.5)(X) = [[1, 0.544942,
  0.544942], ...]` (true Bessel Matern) whereas ferrolearn's fallback returns
  `RBF(length_scale=1.0)(X) = [[1, 0.606531, 0.606531], ...]` — `0.544942 ≠
  0.606531`. Also `Matern(nu=inf)(X) == RBF(1.0)(X)`. Oracle-pinnable (the
  fallback is the divergence).
- AC-11 (REQ-11): `WhiteKernel(0.1)(X)` (Y=None) is `0.1·I`; `WhiteKernel(0.1)(
  X, X)` (Y=X explicit) is the all-zeros `(3,3)` matrix. ferrolearn's
  `compute(X, X)` returns `0.1·I` (row-equality path) — diverges from sklearn's
  `zeros` for the explicit-`Y` call. Oracle-pinnable.
- AC-12 (REQ-12): `RBF(length_scale=[1.0, 2.0])(X)` (anisotropic) differs from
  any scalar-`l` RBF; ferrolearn cannot represent an array length scale.
  Oracle-pinnable once an anisotropic constructor exists.
- AC-13 (REQ-13): `RationalQuadratic()(X)`, `ExpSineSquared()(X)`,
  `(DotProduct()**2)(X)`, `CompoundKernel([...])` all evaluate in sklearn; no
  ferrolearn type exists. Oracle-pinnable per kernel.
- AC-14 (REQ-14): `ConstantKernel()` defaults `constant_value=1.0`,
  `DotProduct()` defaults `sigma_0=1.0`, `Matern()` defaults `length_scale=1.0,
  nu=1.5`; ferrolearn's `new()` requires explicit args (no `Default`).
- AC-15 (REQ-15): no `ndarray` in the owned computation; arrays/distances route
  through `ferray-core`/`ferray-ufunc`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (RBF formula) | SHIPPED | `RBFKernel::compute` in `gp_kernels.rs` does `sq.mapv(\|d\| (-d/(two*ls2)).exp())` over `squared_distances`, and `diagonal` returns `from_elem(n, 1)` — mirrors `RBF.__call__` `K = exp(-0.5·sqdist)` (`kernels.py:1558-1562`). Live oracle (`X=[[0,0],[1,0],[0,1]]`): `RBF(length_scale=1.5)(X) = [[1, 0.800737, 0.800737],[0.800737, 1, 0.64118],[0.800737, 0.64118, 1]]`. Non-test consumer: `gaussian_process.rs` `fn fit` (`let mut k_mat = self.kernel.compute(x, x)`) / `fn predict` (`self.kernel.compute(x, &self.x_train)`); `gp_classifier.rs` `fn fit_binary_gpc` (`kernel.compute(x, x)`). In-crate `rbf_self_covariance_is_one`/`rbf_cross_covariance`/`rbf_length_scale_effect`. Verification: `cargo test -p ferrolearn-kernel --lib gp_kernels` (32 passed). Deterministic / oracle-pinnable. |
| REQ-2 (Matern 0.5/1.5/2.5) | SHIPPED | `MaternKernel::compute` branches on `nu ≈ 0.5/1.5/2.5`: `(-r/ls).exp()`; `(1+√3·r/ls)·exp(-√3·r/ls)`; `(1+√5·r/ls+5/3·(r/ls)²)·exp(-√5·r/ls)` — mirrors `Matern.__call__` (`kernels.py:1719-1726`). Live oracle: `Matern(1.0, nu=1.5)(X) = [[1, 0.483358, 0.483358], ...]`, `nu=2.5 → [[1, 0.523994, ...]]`, `nu=0.5 → [[1, 0.367879, ...]]`. Consumer: `gaussian_process.rs` test `predict_with_matern` uses `MaternKernel::new(1.0, 1.5)` through `GaussianProcessRegressor` (production fit path). In-crate `matern_05_is_exponential`/`matern_15_at_zero`/`matern_25_at_zero`. Deterministic / oracle-pinnable. (The `nu ∉ {0.5,1.5,2.5}` fallback is REQ-10, NOT-STARTED.) |
| REQ-3 (Constant) | SHIPPED | `ConstantKernel::compute` returns `from_elem((n,m), constant_value)`; `diagonal` returns `from_elem(n, constant_value)` — mirrors `ConstantKernel.__call__`/`diag` (`kernels.py:1275`). Live oracle: `ConstantKernel(2.0)(X)` all-2.0. Non-test consumer: used inside `ProductKernel`/`SumKernel` in `gaussian_process.rs`/`gp_classifier.rs` GP fit paths. In-crate `constant_kernel`/`constant_diagonal`. Deterministic / oracle-pinnable. |
| REQ-4 (DotProduct) | SHIPPED | `DotProductKernel::compute` does `x1.dot(&x2.t()).mapv(\|v\| v + s0_sq)`; `diagonal` does `row.dot(&row) + s0_sq` — mirrors `DotProduct.__call__` `K = sigma_0² + X·Yᵀ` (`kernels.py:2113`). Live oracle: `DotProduct(1.0)(X) = [[1,1,1],[1,2,1],[1,1,2]]`. Non-test consumer: `gaussian_process.rs` test `predict_with_dot_product` drives `GaussianProcessRegressor::new(Box::new(DotProductKernel::new(1.0)))` through the production fit/predict. In-crate `dot_product_at_origin`/`dot_product_linear`/`dot_product_diagonal`. Deterministic / oracle-pinnable. |
| REQ-5 (Sum/Product + theta order) | SHIPPED | `SumKernel::compute` returns `m1+m2`, `ProductKernel::compute` returns `m1*m2`; both `diagonal` combine `d1`/`d2`; `get_params` does `params = k1.get_params(); params.extend(k2.get_params())` — matching sklearn's `Sum`/`Product` `__call__` and `theta` concatenation (`kernels.py:796`, `:893`). Live oracle: `(RBF(1.5)+WhiteKernel(0.1)).theta = [0.405465, -2.302585] = [ln 1.5, ln 0.1]` (k1 then k2). Non-test consumer: `gaussian_process.rs` `predict_with_sum_kernel` (`SumKernel::new(RBF, White)`) and `predict_with_product_kernel` (`ProductKernel::new(Const, RBF)`) drive `GaussianProcessRegressor` production fit. In-crate `sum_kernel`/`product_kernel_scaling`/`sum_kernel_params`. Deterministic / oracle-pinnable. |
| REQ-6 (log-space round-trip) | SHIPPED | each kernel's `get_params` returns `vec![value.ln()]` and `set_params` does `value = params[0].exp()` (`RBFKernel`, `MaternKernel`, `ConstantKernel`, `WhiteKernel`, `DotProductKernel`) — matching the log transform sklearn applies in `theta` (`kernels.py:287-309` `np.log(np.hstack(theta))`). Live oracle: `RBF(2.0).theta = [ln 2.0 = 0.693147]`. Non-test consumer: `n_params`/`get_params`/`set_params` are part of the trait the GP estimators store and clone (`clone_box`); the *flat-vector contract* ships even though no optimizer currently calls `set_params` in a search loop (that loop is REQ-8/REQ-9). In-crate `rbf_params_roundtrip`/`matern_params_roundtrip`. Deterministic / oracle-pinnable. |
| REQ-7 (GPR/GPC consumer) | SHIPPED | `lib.rs` re-exports `pub use gp_kernels::{ConstantKernel, DotProductKernel, GPKernel, MaternKernel, ProductKernel, RBFKernel, SumKernel, WhiteKernel}`. Non-test production consumers: `gaussian_process.rs` holds `kernel: Box<dyn GPKernel<F>>` and calls `compute` in `fn fit` (`let mut k_mat = self.kernel.compute(x, x)`) and `fn predict` (`compute(x, &self.x_train)`), and `diagonal` in the predictive-variance path (`let k_star_diag = self.kernel.diagonal(x)`); `gp_classifier.rs` calls `kernel.compute(x, x)` in `fn fit_binary_gpc` and `model.kernel.compute(x, &model.x_train)` in `fn predict`. Verification: `cargo test -p ferrolearn-kernel --lib gaussian_process gp_classifier`. (No Python binding for GP — consumer is the Rust GPR/GPC, which IS the public estimator API per R-DEFER-1/S5.) |
| REQ-8 (eval_gradient / dK/dθ) | NOT-STARTED | blocker issue to be filed by critic. The `GPKernel` trait has only `compute`/`diagonal` — NO gradient method. sklearn every `__call__(X, eval_gradient=True)` returns `(K, K_gradient)` of shape `(n,n,n_dims)` (RBF `:1569-1582`, Matern `:1742-1781`, Constant `:1280-1291`, White `:1405-1412`, DotProduct/Sum/Product). Live oracle: `RBF(1.0)(X, eval_gradient=True)[1].shape == (3,3,1)`, `[:,:,0] = [[0, 0.606531, 0.606531],[0.606531, 0, 0.735759],[0.606531, 0.735759, 0]]`. Consequence: GPR's log-marginal-likelihood gradient optimizer cannot be driven — `gaussian_process.rs` stores `n_restarts_optimizer` but `fn fit` performs a single fixed-kernel Cholesky solve with NO optimizer loop, so hyperparameters are never tuned (REQ-8/REQ-9 are jointly the missing optimization capability). Oracle-pinnable per kernel. |
| REQ-9 (theta/bounds/Hyperparameter/fixed) | NOT-STARTED | blocker issue to be filed by critic. ferrolearn exposes `get_params`/`set_params`/`n_params` (flat log vector) but NO `bounds`, NO `Hyperparameter` objects (sklearn `:52`), NO `fixed` flag, NO `n_dims`/`clone_with_theta`. sklearn's `Kernel` base exposes `theta` `:287-309`, `bounds` `:342-358` (log-transformed `(n_dims,2)`), `hyperparameters` `:277-285`, `n_dims` `:272-275`. Live oracle: `(Const(2.0)*RBF(1.5)+White(0.1)).bounds.shape == (3,2)`, `.n_dims == 3`. Without bounds + the `fixed` flag, GPR's L-BFGS-B optimizer (REQ-8) has no search box and cannot pin a hyperparameter. Oracle-pinnable on `bounds`/`n_dims`. |
| REQ-10 (Matern general nu / inf) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-1 numerical contract). `MaternKernel::compute` `else { ... RBF ... }` silently returns the RBF formula (`length_scale=ls`) for any `nu ∉ {0.5,1.5,2.5}`. sklearn evaluates the true modified-Bessel Matern `(2^{1-ν}/Γ(ν))·t^ν·K_ν(t)`, `t=√(2ν)·d` (`kernels.py:1729-1735`) and `nu=inf → exp(-d²/2)` (RBF, `:1727-1728`). Live oracle: `Matern(1.0, nu=3.5)(X) = [[1, 0.544942, 0.544942], ...]` vs ferrolearn's fallback `RBF(1.0)(X) = [[1, 0.606531, ...]]` — `0.544942 ≠ 0.606531`. Needs a `gamma`/`besselk` (`kv`) primitive (likely a ferray/special-function gap, R-SUBSTRATE-5). Oracle-pinnable. |
| REQ-11 (WhiteKernel Y-is-None) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-3 output contract). `WhiteKernel::compute(x1, x2)` checks ROW EQUALITY (`if n1==n2 { ... rows identical → noise_level on the diagonal }`). sklearn distinguishes by argument: `kernel(X)` (Y=None) → `noise_level*eye(n)` (`:1404`), `kernel(X, Y)` → `zeros((n,m))` (`:1416`). Live oracle: `WhiteKernel(0.1)(X)` is `0.1·I` but `WhiteKernel(0.1)(X, X)` (explicit Y) is the all-zeros `(3,3)`; ferrolearn returns `0.1·I` for the explicit-Y call. The trait has no "Y is None" channel — `compute` always takes two arrays. For the GPR training path `K(X,X)` both give `noise*I` (REQ-7 unaffected), but `compute(X, X)` diverges from `kernel(X, Y=X)`. Oracle-pinnable. |
| REQ-12 (anisotropic length_scale) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2 ABI). `RBFKernel`/`MaternKernel` hold `length_scale: F` (scalar). sklearn accepts `length_scale: float OR ndarray(n_features,)` (`:1472-1475`) with the `anisotropic` property `:1512-1514` and a dimension-wise gradient path (`:1576-1582`). Live oracle: `RBF(length_scale=[1.0, 2.0])(X)` (per-feature scaling) cannot be expressed by any scalar-`l` ferrolearn RBF. Oracle-pinnable once an array-`length_scale` constructor exists. |
| REQ-13 (missing kernels) | NOT-STARTED | blocker issue to be filed by critic (builder territory). No ferrolearn analog for `RationalQuadratic` (`kernels.py:1798`, `k = (1 + d²/(2αl²))^{-α}`), `ExpSineSquared` (`:1954`, periodic), `Exponentiation` (`:993`, `kernel ** exponent`), `CompoundKernel` (`:514`, the per-class GPC kernel stack). These are real mirrored surface (each evaluates in the live oracle), not out-of-scope. Oracle-pinnable per kernel once built. |
| REQ-14 (constructor defaults / Default) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2). ferrolearn's `new()` requires an explicit value for every kernel and there is NO `Default` impl with sklearn's keyword defaults: RBF/Matern `length_scale=1.0` (`:1508`, `:1678`), Matern `nu=1.5` (`:1678`), `ConstantKernel` `constant_value=1.0` (`:1233`), `DotProduct` `sigma_0=1.0` (`:2156`), `WhiteKernel` `noise_level=1.0`. sklearn users write `RBF()`/`ConstantKernel()`/`DotProduct()` and get those defaults. Oracle-pinnable (default-constructed kernel output). (The Sum/Product `theta` ordering MATCHES — REQ-5.) |
| REQ-15 (ferray substrate) | NOT-STARTED | blocker issue to be filed by critic (R-SUBSTRATE-1). `gp_kernels.rs` imports `ndarray::{Array1, Array2}`; `squared_distances`/`euclidean_distances`/`compute`/`diagonal` operate on `ndarray`. Destination: `ferray-core` (array type), `ferray-ufunc` (elementwise `exp`/distance), `ferray::linalg` (the `X·Yᵀ` dot for DotProduct). Not migrated. The Bessel primitive for REQ-10 belongs in `ferray::stats`/a special-functions analog (`scipy.special.kv`/`gamma`). |

## Architecture

ferrolearn models the kernel hierarchy as a single trait `GPKernel<F>`
(`compute`/`diagonal`/`n_params`/`get_params`/`set_params`/`clone_box`) with one
struct per kernel, where sklearn uses an abstract `Kernel` base (`kernels.py:153`)
with `__call__`/`diag` plus the `theta`/`bounds`/`hyperparameters`/`n_dims`
property surface (`:272-358`). The composite kernels `SumKernel`/`ProductKernel`
hold `Box<dyn GPKernel<F>>` children and delegate `compute`/`diagonal` and the
flat-vector `get_params` (k1 then k2), mirroring sklearn's `Sum`/`Product`
(`:796`, `:893`) and their `theta` concatenation order. Trait objects are cloned
via `clone_box` (the `impl Clone for Box<dyn GPKernel<F>>` shim), the Rust analog
of sklearn's `clone`.

The fundamental structural gap is that `GPKernel` is a **value-only** kernel:
`compute` returns just `K`, never `(K, K_gradient)`. sklearn's `__call__` carries
the `eval_gradient` channel (REQ-8) and the `Kernel` base carries the
`theta`/`bounds`/`Hyperparameter`/`fixed` machinery (REQ-9) that the
log-marginal-likelihood optimizer in `_gpr.py` consumes. ferrolearn's
`GaussianProcessRegressor` reflects this: `fn fit` (`gaussian_process.rs`) does a
single fixed-kernel `K + alpha·I` Cholesky solve and NO hyperparameter
optimization — `n_restarts_optimizer` is stored but unused, and
`log_marginal_likelihood` is an introspection method, not an objective driving a
search. So the missing gradient/theta/bounds surface has no *current* ferrolearn
consumer either; it is the prerequisite for ever matching sklearn's *tuned*
kernel, hence NOT-STARTED rather than absent-by-design.

The kernel formulas themselves (REQ-1..5) are value-exact: `squared_distances`/
`euclidean_distances` build the pairwise distance matrix, and each `compute`
applies the closed-form covariance. Three element-wise divergences in the
existing kernels are pinned NOT-STARTED: the Matern `else` RBF fallback for
unsupported `nu` (REQ-10), the WhiteKernel row-equality vs `Y is None` semantics
(REQ-11), and the scalar-only `length_scale` (REQ-12). Constructor defaults
(REQ-14) and four missing kernels (REQ-13) round out the surface gap.

Invariants: `compute` returns `(n1, n2)`; `diagonal` returns `(n,)`; for every
stationary kernel `diagonal(X)` ≡ `compute(X, X)` diagonal (1 for RBF/Matern,
`constant_value` for Constant, `noise_level` for White, `‖xᵢ‖²+σ₀²` for
DotProduct). `K(X,X)` is symmetric PSD — pinned by `rbf_positive_semidefinite`/
`matern_15_positive_semidefinite`. The covariance matrix `K + alpha·I` consumed
by GPR is PD for `alpha > 0`.

## Verification

Commands establishing the SHIPPED claims (run at baseline `d853cf241`):

- `cargo test -p ferrolearn-kernel --lib gp_kernels` → 32 passed, 0 failed
  (REQ-1..6: `rbf_*`, `matern_*`, `constant_*`, `white_*`, `dot_product_*`,
  `sum_*`, `product_*`, `*_params_roundtrip`, `*_positive_semidefinite`).
- `cargo test -p ferrolearn-kernel --lib gaussian_process gp_classifier` →
  exercises the production consumer (REQ-7): `predict_with_matern`,
  `predict_with_dot_product`, `predict_with_sum_kernel`,
  `predict_with_product_kernel` drive `GaussianProcessRegressor` through
  `kernel.compute`/`diagonal`.
- Base-formula oracle (REQ-1..5, R-CHAR-3 — expected from sklearn, never copied
  from ferrolearn), `X=[[0,0],[1,0],[0,1]]`:
  `python3 -c "import numpy as np; from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, DotProduct; X=np.array([[0.,0.],[1.,0.],[0.,1.]]); print(RBF(length_scale=1.5)(X).tolist()); print(Matern(length_scale=1.0,nu=2.5)(X).tolist()); print(DotProduct(sigma_0=1.0)(X).tolist())"`
  → RBF `[[1, 0.800737, ...]]`, Matern `[[1, 0.523994, ...]]`, DotProduct
  `[[1,1,1],[1,2,1],[1,1,2]]`. A critic pins these as Rust `#[test]`s comparing
  `compute` to the live oracle.

Open divergences pinned as FAILING tests (NOT-STARTED, oracle expected values —
all deterministic, all oracle-pinnable now):

- REQ-8 (eval_gradient): `RBF(1.0)(X, eval_gradient=True)[1]` has shape
  `(3,3,1)` with `[:,:,0] = [[0, 0.606531, 0.606531],[0.606531, 0, 0.735759],
  [0.606531, 0.735759, 0]]`. ferrolearn has no gradient method — pinned once the
  trait method is added.
- REQ-9 (theta/bounds): `(Const(2.0)*RBF(1.5)+White(0.1)).bounds.shape ==
  (3,2)`, `.n_dims == 3`. ferrolearn exposes neither.
- REQ-10 (Matern general nu): `Matern(1.0, nu=3.5)(X) = [[1, 0.544942,
  0.544942], ...]` vs ferrolearn's fallback `RBF(1.0)(X) = [[1, 0.606531, ...]]`
  — the off-by-Bessel mismatch is the pinned divergence; `Matern(nu=inf)(X) ==
  RBF(1.0)(X)`.
- REQ-11 (WhiteKernel Y-is-None): `WhiteKernel(0.1)(X, X)` (explicit Y) is the
  all-zeros `(3,3)` in sklearn; ferrolearn's `compute(X, X)` returns `0.1·I`.
- REQ-12 (anisotropic): `RBF(length_scale=[1.0, 2.0])(X)` ≠ any scalar-`l` RBF.
- REQ-13 (missing kernels): `RationalQuadratic()(X)`, `ExpSineSquared()(X)`,
  `(DotProduct()**2)(X)`, `CompoundKernel([...])` evaluate in sklearn; no
  ferrolearn type.
- REQ-14 (defaults): `ConstantKernel()`/`DotProduct()`/`Matern()` use
  `constant_value=1.0`/`sigma_0=1.0`/`length_scale=1.0,nu=1.5`; ferrolearn's
  `new()` requires explicit args.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1 (RBF),
REQ-2 (Matern 0.5/1.5/2.5), REQ-3 (Constant), REQ-4 (DotProduct), REQ-5
(Sum/Product + theta order), REQ-6 (log-space round-trip), REQ-7 (GPR/GPC
consumer) — impl + non-test GP consumer + green verification. NOT-STARTED (the
critic files per-REQ `-l blocker` issues): REQ-8 (eval_gradient / `dK/dθ`),
REQ-9 (theta/bounds/Hyperparameter/fixed), REQ-10 (Matern general-`nu`/Bessel +
`nu=inf`), REQ-11 (WhiteKernel `Y is None` semantics), REQ-12 (anisotropic
`length_scale`), REQ-13 (RationalQuadratic/ExpSineSquared/Exponentiation/
CompoundKernel), REQ-14 (constructor defaults / `Default`), REQ-15 (ferray
substrate). GP kernels are deterministic, so every NOT-STARTED is a
missing-feature or element-wise-divergence blocker that is oracle-pinnable now —
there is no RNG/generator carve-out here (cf. `nystroem.md` REQ-8).
