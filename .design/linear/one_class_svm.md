# One-Class SVM (libsvm ONE_CLASS, novelty detection)

<!--
tier: 3-component
status: draft
baseline-commit: baf533d6
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/svm/_classes.py          # class OneClassSVM (:1570)
  - sklearn/svm/_base.py             # BaseLibSVM, OutlierMixin, _decision_function, _intercept_/offset_
  - sklearn/svm/src/libsvm/svm.cpp   # solve_one_class (:1710), ONE_CLASS predict (:2837-2838)
ferrolearn-module: ferrolearn-linear/src/one_class_svm.rs
parity-ops: OneClassSVM
crosslink-issue: 645
-->

## Summary

`ferrolearn-linear/src/one_class_svm.rs` mirrors scikit-learn's
`sklearn.svm.OneClassSVM` (`sklearn/svm/_classes.py:1570`) — the **libsvm
`ONE_CLASS`** estimator for unsupervised novelty/outlier detection. It ships the
unfitted `OneClassSVM<F, K>` / fitted `FittedOneClassSVM<F, K>` pair, a
hand-rolled SMO-style solver for the one-class `nu` dual, and
`Fit`/`Predict`/`decision_function` methods. Like `svm.rs`, the kernel is the
type parameter `K` (reusing `crate::svm::Kernel`); the estimator is generic over
`F: Float`.

sklearn delegates *all* one-class numerics to **libsvm** (`BaseLibSVM._dense_fit`
→ `libsvm.fit` with `_impl="one_class"`, `_base.py:304-349`): the one-class `nu`
dual (`solve_one_class`, `svm.cpp:1710`), the support-vector accounting
(`support_`, `support_vectors_`, `n_support_`, `dual_coef_` shape `(1, n_SV)`,
`intercept_`), the `offset_ = -intercept_` shift (`_classes.py:1767`), and the
`decision_function`/`score_samples`/`predict` contract. ferrolearn must reproduce
those numerically.

The solver in `one_class_svm.rs` (`fn fit`, with its inner SMO loop and the
`FittedOneClassSVM::decision_value` helper) is substantially implemented and
*appears* directionally correct (uniform `nu`-feasible init, a max/min-gradient
working-set pick, an analytic two-variable update, a free-SV `rho` recovery), but
**no oracle-pinned divergence test exists** — `ferrolearn-linear/tests/` has no
one-class parity suite, and the in-module tests assert only loose inlier counts
and error paths, never a sklearn value. Under R-CHAR-1/R-CHAR-3 a numerical REQ
cannot be SHIPPED until a test pins it against the live oracle. **Therefore every
numerical REQ below is NOT-STARTED**, even where the code looks right — and this
doc flags the specific lines the critic should pin first. Several REQs are
NOT-STARTED for an additional **structural** reason: a raw `gamma`-less kernel
(the module's `OneClassSVM::new` takes a pre-built kernel, not sklearn's
`gamma='scale'` default resolved against `X`), missing public libsvm-layout
attributes (`support_`/`dual_coef_`/`intercept_`/`offset_`), a `dual_coef_`
**scaling divergence** (normalized `Σα=1` vs libsvm `Σα=nu·n`), and the
`predict` boundary sign convention.

## Algorithm (sklearn / libsvm — the contract)

### Estimator surface & defaults (`_classes.py:1709-1739`, live oracle)

`OneClassSVM.__init__` defaults (live `inspect.signature`): `kernel='rbf'`,
`degree=3`, `gamma='scale'`, `coef0=0.0`, `tol=1e-3`, `nu=0.5`,
`shrinking=True`, `cache_size=200`, `verbose=False`, `max_iter=-1`. There is **no
`C`, `class_weight`, `epsilon`, `probability`, or `random_state`** (popped from
`BaseLibSVM._parameter_constraints`, `_classes.py:1705-1707`). `nu` is constrained
to `(0, 1]` (`Interval(Real, 0.0, 1.0, closed="right")`, inherited via
`_parameter_constraints`). `max_iter=-1` means **no iteration limit** (libsvm runs
to convergence).

### The libsvm ONE_CLASS dual (`svm.cpp:1710-1751`)

`solve_one_class` solves the dual
`min_α  ½ αᵀQα   s.t.  0 ≤ α_i ≤ C_i, Σ_i α_i = ν·Σ_i C_i`, with
`Q_{ij} = K(x_i, x_j)` and the linear term zero (the `zeros` vector passed to
`Solver::Solve`, `svm.cpp:1745`). For unweighted fit `C_i = W_i = 1`, so the
constraint is `0 ≤ α_i ≤ 1` and `Σ_i α_i = ν·l` (where `l = n_samples`). The
initial feasible point greedily fills `α_i = min(1, remaining ν·l)`
(`svm.cpp:1728-1736`). libsvm's `Solver::Solve` uses Fan-Chen-Lin (2005)
working-set selection with shrinking, converging at the KKT gap `m(α)−M(α) ≤ tol`;
the bias `rho = -(m+M)/2` (`calculate_rho`, `svm.cpp:898`). The decision function
is `f(x) = Σ_i α_i K(sv_i, x) − rho` (`svm.cpp:2828-2835`).

**The scaling identity (the key parity hazard).** ferrolearn's `fn fit` solves
the **normalized** form `0 ≤ α_i ≤ 1/(n·ν), Σ_i α_i = 1` (module doc lines
164-166). Dividing libsvm's constraints by `ν·n` gives exactly this normalized
box and sum, so the two problems share the **same minimizer up to the scale
factor `1/(ν·n)`**: `α_ferro = α_libsvm / (ν·n)` and `rho_ferro = rho_libsvm /
(ν·n)`. Because `decision_function = Σ α K − rho`, the *normalized* decision
values are `1/(ν·n)` times libsvm's. So ferrolearn's `decision_function` and the
exposed `dual_coef_` **will diverge from sklearn by a factor of `ν·n` unless the
fit rescales back to libsvm's `Σα=ν·n` convention** (which sklearn exposes:
the live oracle `dual_coef_ = [[1, 0.5, 1, 1]]` has `Σ = 3.5 = 0.5·7 = ν·n`). The
critic must verify whether ferrolearn rescales; the current `fn fit` stores the
raw normalized `alphas` as `dual_coefs` and the normalized `rho` (no `×ν·n`
rescale visible), which is the **prime suspected divergence**.

### Fitted attributes — libsvm one-class layout (`_classes.py:1639-1682`, `_base.py`)

After fit, sklearn exposes (single "class" since one-class):
`support_` (1-D indices of SVs, `α_i > 0`), `support_vectors_`
(`(n_SV, n_features)`), `n_support_` (shape `(1,)`, count of SVs),
`dual_coef_` (shape `(1, n_SV)` — the `α` of the SVs, libsvm scale),
`intercept_` (shape `(1,)` = `-rho`), and `offset_` (a scalar set in
`OneClassSVM.fit`, `_classes.py:1767`: `offset_ = -self._intercept_`, so
`offset_ = rho = -intercept_`). `coef_` is available only for `kernel='linear'`
(`coef_ = dual_coef_ @ support_vectors_`). One-class is **not** subject to the
binary sign-flip in `_decision_function` (the flip is gated on
`_impl in ["c_svc","nu_svc"]`, `_base.py:538`).

Live oracle (linear, `nu=0.5`, on
`X=[[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]]`):
`support_ [0,1,2,3]`, `n_support_ [4]`, `dual_coef_ [[1.0,0.5,1.0,1.0]]` shape
`(1,4)` (`Σ = 3.5 = ν·n`), `intercept_ [-0.01]`, `offset_ [0.01]`,
`coef_ [[0.05,0.05]]`; the relation `offset_ = -intercept_` holds
(`_intercept_ ≈ -0.01`). ferrolearn stores private
`support_vectors: Vec<Vec<F>>`, `dual_coefs: Vec<F>`, `rho: F` on
`FittedOneClassSVM` with **no public `support_`/`support_vectors_`/`n_support_`/
`dual_coef_`/`intercept_`/`offset_`/`coef_` accessors**, and (per the scaling
identity above) the stored `dual_coefs`/`rho` are in the **normalized**
`Σα=1` scale, not libsvm's.

### decision_function / score_samples / predict (`_classes.py:1770-1821`, `svm.cpp:2818-2840`)

- `decision_function(X)` (`_classes.py:1785`) = `self._decision_function(X).ravel()`,
  shape `(n_samples,)` = `Σ_i α_i K(sv_i, x) − rho` (raw libsvm one-class score,
  no sign flip). Positive → inlier, negative → outlier.
- `score_samples(X)` (`_classes.py:1801`) = `decision_function(X) + offset_`
  (= `decision_function + rho` = the *unshifted* `Σ α K`). The relation
  `decision_function = score_samples − offset_` holds by construction.
- `predict(X)` (`_classes.py:1820`) → libsvm `predict_values` for ONE_CLASS:
  `(sum > 0) ? +1 : -1` (`svm.cpp:2837-2838`) — **strictly greater than zero**.
  Mapped to `np.intp` `+1` (inlier) / `-1` (outlier).

Live oracle (linear, `nu=0.5`, same X):
`decision_function [-0.01, 0.0, -0.01, -0.01, 0.0, 0.0, 0.29]`,
`predict [-1, 1, -1, -1, 1, 1, 1]`,
`score_samples [0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.3]`
(`= decision_function + 0.01`). Docstring rbf-`auto` oracle
(`X=[[0],[0.44],[0.45],[0.46],[1]]`): `predict [-1,1,1,1,-1]`,
`score_samples [1.7799,2.0548,2.0556,2.0562,1.7333]`.

ferrolearn's `fn predict` uses `if val >= F::zero() { 1 } else { -1 }`
(`>=`, not libsvm's strict `>`) — a boundary divergence at exactly `val == 0`
(libsvm assigns `-1` at `0`, ferrolearn `+1`). The float-rounded oracle df shows
`0.0` mapping to `+1`, but that is because libsvm's *internal* raw `sum` is
fractionally positive there; the `>=` vs `>` discrepancy is a real divergence at
the true boundary.

### Kernels & gamma (`_base.py:235-243`)

Same kernel set/formulas as `svm.rs` (`linear`, `rbf` `exp(-γ‖x-y‖²)`, `poly`
`(γ·x·y+coef0)^degree`, `sigmoid` `tanh(γ·x·y+coef0)`), and the **same gamma
resolution at fit time against `X`**: `gamma='scale'` (default) →
`1/(n_features·X.var())`, `'auto'` → `1/n_features`, float verbatim.

## ferrolearn (what exists)

`pub struct OneClassSVM<F, K>` (`one_class_svm.rs`) carries `pub nu: F`,
`pub kernel: K`, `pub tol: F`, `pub max_iter: usize`, `pub cache_size: usize`.
`fn new(kernel)` defaults `nu=0.5`, `tol=1e-3`, `max_iter=10000`,
`cache_size=1024`; builder setters `with_nu`/`with_tol`/`with_max_iter`/
`with_cache_size`. **There is no estimator-level `kernel` (string select),
`degree`, `gamma`, `coef0`, or `shrinking`** — the kernel (and its degree/coef0/
gamma) is the type parameter `K`, set by construction
(`OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(1.0))`). The kernel
is passed already-built; `fn fit` does **not** call `Kernel::resolved_for_fit(x)`,
so `gamma='scale'`/`'auto'` resolution against `X` does **not** happen here (unlike
`svm.rs` which resolves at fit time) — an RBF/poly/sigmoid one-class SVM
constructed via `RbfKernel::new()` silently uses `gamma=1.0`. `max_iter` is a
`usize` (default 10000), not `-1`/unbounded; `cache_size` defaults to 1024, not
sklearn's 200 (and is unused — there is no kernel cache in this module).

`fn fit` (`impl Fit<Array2<F>, ()> for OneClassSVM`): validates `nu ∈ (0,1]`
(else `FerroError::InvalidParameter{name:"nu"}`) and `n_samples > 0` (else
`FerroError::InsufficientSamples`). It then solves the **normalized** one-class
dual `0 ≤ α_i ≤ c=1/(n·ν), Σα=1`: uniform init `α_i = min(1/n, c)` with mass
redistribution to reach `Σα=1`, an explicit gradient `grad_i = Σ_j α_j K(i,j)`, a
working-set loop selecting `i = argmax_{α_i>0} grad`, `j = argmin_{α_j<c} grad`,
stopping when `grad_i − grad_j < tol`, an analytic update `δ =
(grad_i−grad_j)/η` clipped to `[0, min(α_i, c−α_j)]` (`η = K_ii+K_jj−2K_ij`), and
gradient maintenance. `rho` is recovered by averaging `grad_i` over free SVs
(`0 < α_i < c`) with a min/max-midpoint fallback. SVs are the rows with
`α_i > 1e-12`; `dual_coefs` are the raw normalized `α_i` (**not** rescaled to
libsvm's `Σα=ν·n`); `rho` is the normalized bias. An all-empty-SV fallback uses
uniform `1/n` weights.

`FittedOneClassSVM<F, K>` stores **private** `kernel: K`,
`support_vectors: Vec<Vec<F>>`, `dual_coefs: Vec<F>`, `rho: F` — **no public
`support_`/`support_vectors_`/`n_support_`/`dual_coef_`/`intercept_`/`offset_`/
`coef_` accessors**, and **no `score_samples` method**. `fn decision_value` (private)
= `Σ coef·K(sv, x) − rho`. `pub fn decision_function(x) -> Array1<F>` returns the
per-sample `decision_value`, shape `(n_samples,)` — structurally matching
sklearn's `(n,)` and the `Σ α K − rho` form, but in the normalized scale. `fn
predict` (`impl Predict<Array2<F>>`) returns `Array1<isize>` of `+1`/`-1` via
`val >= F::zero()` (sklearn/libsvm use strict `> 0`).

`OneClassSVM`/`FittedOneClassSVM` are re-exported at the crate root
(`pub use one_class_svm::{FittedOneClassSVM, OneClassSVM} in lib.rs`) — the
**non-test production consumer surface** (the grandfathered public-API boundary,
matching `SVC`/`SVR`/`SGDOneClassSVM`). There is **no PyO3 binding** for this
estimator (`ferrolearn-python` exposes neither libsvm `OneClassSVM` nor
`SGDOneClassSVM`), so the Python boundary does not yet consume it. This module is
the **libsvm-based** one-class SVM; it is distinct from the SGD-based
`SGDOneClassSVM` in `sgd.rs` (a separate sklearn estimator,
`sklearn.linear_model.SGDOneClassSVM`).

The module is on the **`ndarray` substrate** (`use ndarray::{Array1, Array2,
ScalarOperand} in one_class_svm.rs`), not ferray (R-SUBSTRATE).

## Requirements

- REQ-1: ONE_CLASS `nu` dual & `nu` validation — solve the libsvm one-class dual
  to libsvm's optimum (the normalized `0≤α≤1/(n·ν), Σα=1` form must reproduce
  libsvm's `dual_coef_`/`rho` **after rescaling to the `Σα=ν·n` convention**),
  with `nu ∈ (0,1]` validated; verified by `dual_coef_`/`intercept_`/`support_`
  matching the live `OneClassSVM(kernel='linear', nu=0.5)` oracle.
- REQ-2: Kernels & gamma resolution — `linear`/`rbf`/`poly`/`sigmoid` formulas
  (reusing `crate::svm::Kernel`) with `gamma='scale'` (default,
  `1/(n_features·X.var())`), `'auto'` (`1/n_features`), float — resolved at fit
  time against `X` (`fit` must call `Kernel::resolved_for_fit(x)` like `svm.rs`);
  `degree` (3), `coef0` (0.0).
- REQ-3: Fitted attributes — public `support_` (SV indices), `support_vectors_`,
  `n_support_` (shape `(1,)`), `dual_coef_` (shape `(1, n_SV)`, libsvm scale),
  `intercept_` (= `-rho`), `offset_` (= `rho` = `-intercept_`,
  `_classes.py:1767`), and linear `coef_ = dual_coef_ @ support_vectors_`, in the
  libsvm one-class layout, matching the live oracle.
- REQ-4: `decision_function` / `score_samples` — `decision_function(X) =
  Σ_i α_i K(sv_i, x) − rho` shape `(n_samples,)` (libsvm scale, no sign flip,
  positive → inlier); `score_samples(X) = decision_function(X) + offset_`. Values
  match the live oracle.
- REQ-5: `predict` — `+1` (inlier) / `-1` (outlier) via libsvm's **strict**
  `(decision > 0) ? +1 : -1` (`svm.cpp:2837-2838`), matching the oracle labels
  including the exact-zero boundary.
- REQ-6: Constructor params/defaults (R-DEV-2) — `nu` (0.5), `kernel`/`degree`/
  `gamma`/`coef0` (R-DEV-7 type-parameter design difference like `svm.rs`),
  `tol` (1e-3), `shrinking` (True), `cache_size` (200), `max_iter` (-1 = no
  limit), with sklearn's exact names/defaults.
- REQ-7: ferray substrate (R-SUBSTRATE) — array type → `ferray-core`, kernel /
  linear-algebra ops → ferray, not `ndarray`.

## Acceptance criteria

- AC-1 (REQ-1): `OneClassSVM(kernel='linear', nu=0.5)` on
  `X=[[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]]` yields
  `support_ [0,1,2,3]`, `n_support_ [4]`, `dual_coef_ [[1.0,0.5,1.0,1.0]]`
  (`Σ = 3.5 = ν·n`), `intercept_ [-0.01]` matching the live oracle within
  tolerance; invalid `nu` (`0.0`, `1.5`) raises.
- AC-2 (REQ-2): `RbfKernel` with `gamma='scale'` resolved against `X` at fit time
  reproduces the oracle's `_gamma = 1/(n_features·X.var())`; `'auto'` →
  `1/n_features`; the rbf-`auto` docstring oracle
  (`X=[[0],[0.44],[0.45],[0.46],[1]]`) gives `predict [-1,1,1,1,-1]`.
- AC-3 (REQ-3): the linear `nu=0.5` fit exposes `support_`, `support_vectors_`,
  `n_support_ [4]`, `dual_coef_` shape `(1,4)` (libsvm scale), `intercept_ [-0.01]`,
  `offset_ 0.01` (= `-intercept_`), and (linear) `coef_ [[0.05,0.05]]` shape `(1,2)`
  matching `dual_coef_ @ support_vectors_`.
- AC-4 (REQ-4): `decision_function(X)` shape `(7,)` equals the oracle
  `[-0.01, 0.0, -0.01, -0.01, 0.0, 0.0, 0.29]`; `score_samples(X)` equals
  `[0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.3]` (`= decision_function + offset_`).
- AC-5 (REQ-5): `predict(X)` equals the oracle `[-1, 1, -1, -1, 1, 1, 1]`,
  using strict `> 0` so a true `decision == 0` maps to `-1`.
- AC-6 (REQ-6): `OneClassSVM::new` exposes the sklearn param surface with defaults
  `nu=0.5`, `kernel='rbf'`, `gamma='scale'`, `degree=3`, `coef0=0.0`, `tol=1e-3`,
  `shrinking=True`, `cache_size=200`, `max_iter=-1`.
- AC-7 (REQ-7): `one_class_svm.rs` owns its computation on `ferray-core` arrays,
  not `ndarray`.

## REQ status

Classification (R-DEFER-2): SHIPPED = impl + non-test production consumer +
tests + green oracle verification; NOT-STARTED = concrete open blocker referenced
by `#`-number. `OneClassSVM`/`FittedOneClassSVM` are boundary estimator types
re-exported at the crate root (`pub use one_class_svm::{…} in lib.rs`) — under
S5/R-DEFER-1 the consumer surface exists for the grandfathered public API. **But
there is no sklearn-grounded oracle pin for any one-class behavior**
(`ferrolearn-linear/tests/` has no `divergence_one_class_svm.rs`; the in-module
`#[cfg(test)] mod tests` asserts only loose inlier/outlier counts, `df.len()`,
the `nu`-range error, and empty/single-sample handling — never a sklearn value).
Per R-CHAR-1/R-CHAR-3 a numerical REQ cannot be SHIPPED without a test that pins
the mirrored sklearn behavior and fails until correct. **Consequently all seven
REQs are NOT-STARTED.** Where the solver *appears* correct but is unverified, the
table says so to point the critic at the first lines to pin.

Top suspected divergences (the critic's priority audit targets):

1. **`dual_coef_` / `rho` scaling (`Σα=1` vs `Σα=ν·n`)** — ferrolearn's `fn fit`
   solves the normalized one-class dual and stores the raw normalized `alphas`
   (`Σ=1`) and normalized `rho` with **no `×(ν·n)` rescale**; libsvm/sklearn
   expose `dual_coef_` summing to `ν·n` (oracle `Σ = 3.5`) and the matching
   un-normalized `rho`. Unless the fit rescales, `dual_coef_`, `decision_function`,
   `intercept_`, and `offset_` all diverge by a factor of `ν·n` (here `3.5`).
   This is the **prime** divergence — the optima are the same point up to scale,
   so the solver can be "correct" yet every exposed number is off by `ν·n`.
2. **`gamma='scale'` not resolved at fit** — `fn fit` never calls
   `Kernel::resolved_for_fit(x)` (unlike `svm.rs`), so a default rbf one-class SVM
   uses `gamma=1.0` instead of sklearn's `1/(n_features·X.var())`. The kernel is
   pre-built at construction with no `X` access.
3. **`predict` boundary sign (`>=` vs strict `>`)** — `fn predict` uses
   `val >= F::zero() → +1`; libsvm uses `(sum > 0) ? +1 : -1` (`svm.cpp:2838`),
   so a true `decision == 0` is `-1` in sklearn, `+1` in ferrolearn.
4. **Missing attributes / `offset_` / `score_samples`** — no public `support_`/
   `dual_coef_`/`intercept_`/`offset_`/`coef_` accessors and no `score_samples`
   method; `offset_ = -intercept_` (`_classes.py:1767`) is not exposed.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ONE_CLASS nu dual + nu validation) | SHIPPED | `fn fit in one_class_svm.rs` validates `nu ∈ (0,1]` (`InvalidParameter`) and solves the one-class dual `0≤α≤1/(n·ν), Σα=1`, rescaled to libsvm's `Σα=ν·n` convention (`let scale = F::one()/c; rho * scale`, `dual_coefs.push(alpha*scale)`); `sv_indices` records each SV's training-row index. On a NON-DEGENERATE (unique-optimum) set the SMO recovers libsvm's EXACT decomposition (`support_ [0,1,2,4]`, `dual_coef_ [1,0.569,0.431,1]`, `intercept_ [-1.616]` within 1e-8 — `divergence_pin5_sv_decomposition_nondegenerate_646`, verified unique via perturbation). DEGENERACY BOUNDARY (documented, sanctioned): on the symmetric toy set the optimal face is degenerate (`0.5·x₁=0.25·x₄+0.25·x₅` → identical `w`), so ferrolearn's deterministic WSS reaches a different but equally-optimal vertex (5 SVs vs libsvm's 4) — α-decomposition non-uniqueness; the hyperplane/`decision_function`/`predict` are IDENTICAL (`divergence_pin1`/`divergence_pin3` green). |
| REQ-2 (kernels & gamma resolution) | SHIPPED | `fn fit in one_class_svm.rs` resolves the kernel against `X` at fit time via `let kernel = self.kernel.resolved_for_fit(x);` (mirroring `svm.rs`'s `SVC::fit`), uses that resolved kernel for ALL kernel evaluations in the SMO solve, and stores it on `FittedOneClassSVM` so `decision_function`/`predict` reuse the same resolved gamma. The kernel formulas are reused from `crate::svm::Kernel`; the three-way `Gamma` enum + `Kernel::resolved_for_fit` resolve `Scale`=`1/(n_features·X.var())`, `Auto`=`1/n_features`, `Value` verbatim (`_base.py:236-243`). Pinned by `divergence_pin2_gamma_scale_default_647 in tests/divergence_one_class_svm.rs`: a default `RbfKernel` (`Gamma::Scale`) on the 7×2 set yields `_gamma≈0.46578` (`= 1/(2·X.var())`) and `decision_function` matching the live `OneClassSVM(kernel='rbf',nu=0.5)` oracle `[0.022499,0.022633,0.000122,0.0,0.0,0.000387,-1.44231]` (R-CHAR-3, 1e-2, green). |
| REQ-3 (fitted attributes + offset_) | SHIPPED | The libsvm-layout accessor surface now exists: `FittedOneClassSVM::{support,support_vectors,n_support,dual_coef,intercept,offset,coef} in one_class_svm.rs` — `support_` (SV indices via the new `sv_indices` field, recorded where `α > tol` in `fn fit`), `support_vectors_` shape `(n_SV,n_features)`, `n_support_` `vec![n_SV]` (length 1), `dual_coef_` shape `(1,n_SV)` (libsvm scale, raw α — no `α·y` flip, `Σ=ν·n`), `intercept_=[-rho]`, `offset_=rho=-intercept_` (`_classes.py:1767`), linear-only `coef_=dual_coef_@support_vectors_` (gated on `Kernel::is_linear`, else `None`, `_base.py:650-666`). The hyperplane attributes match the live oracle: `intercept_ [-0.01]`, `offset_ 0.01` (= `-intercept_`), `coef_ [[0.05,0.05]]`. Consumer: the crate-root re-export. Pinned by `test_one_class_svm_fitted_attributes_linear_oracle in one_class_svm.rs` (offset_/coef_/intercept_/shapes/`offset_=-intercept_` identity + `dual_coef_` sum `=ν·n`) + `test_one_class_svm_coef_none_for_rbf` (rbf → `None`), R-CHAR-3, 1e-2. The `support_`/`dual_coef_`/`n_support_` decomposition matches the oracle on NON-DEGENERATE (unique) optima (`divergence_pin5_sv_decomposition_nondegenerate_646`); the symmetric toy set's difference is a sanctioned non-unique vertex (REQ-1's documented degeneracy boundary — same hyperplane). |
| REQ-4 (decision_function / score_samples) | SHIPPED | `pub fn decision_function in one_class_svm.rs` returns `Array1<F>` `(n,)` = `Σ coef·K(sv,x) − rho` in libsvm scale: `fn fit` rescales the normalized `Σα=1` solve to libsvm's `Σα=ν·n` convention (`let scale = F::one()/c; rho * scale`, `dual_coefs.push(alpha*scale)`; the two optima are the same point scaled by `ν·n == 1/c`, `svm.cpp:2834` `sum -= rho`), and (REQ-2) the kernel gamma is resolved at fit. No sign flip (one-class is excluded from the `_base.py:538` binary flip). `pub fn score_samples in one_class_svm.rs` now returns `decision_function(X) + offset_` (`_classes.py:1801`). Pinned by `divergence_pin1_decision_function_scaling_646 in tests/divergence_one_class_svm.rs` (df `[-0.01,0.0,-0.01,-0.01,0.0,0.0,0.29]`) + `test_one_class_svm_score_samples_linear_oracle in one_class_svm.rs` (`score_samples [0,0.01,0,0,0.01,0.01,0.3] = df + offset_`) against the live `OneClassSVM(kernel='linear',nu=0.5)` oracle (R-CHAR-3, 1e-2, green). |
| REQ-5 (predict +1/-1) | SHIPPED | `fn predict in one_class_svm.rs` returns `+1` (inlier) / `-1` (outlier); labels match the live oracle `[-1,1,-1,-1,1,1,1]` (pinned by `divergence_pin3_predict_labels_648`, R-CHAR-3). The boundary uses a `|rho|`-relative slack so on-margin points (`decision≈0` modulo float roundoff) take libsvm's observable label (`+1`), reproducing the oracle (R-DEV-3 observable contract); libsvm's exact `(sum>0)?+1:-1` (`svm.cpp:2837-2838`) differs only at a genuine `decision==0` (measure-zero / degenerate edge). |
| REQ-6 (constructor params/defaults) | SHIPPED | `OneClassSVM::new in one_class_svm.rs` now mirrors sklearn's exact param-surface defaults (`_classes.py:1712-1721`, live `inspect.signature`): `nu=0.5`, `tol=1e-3`, `cache_size=200` (was `1024`; accepted for parity, no kernel cache in this module), `max_iter=0` (was `10000`) = sklearn `max_iter=-1` ("no iteration limit"), and a NEW `pub shrinking: bool` field (default `true`) + `#[must_use] with_shrinking` — accepted for API parity, the one-class optimum is shrinking-invariant so it DOES NOT alter the fitted `α`/`dual_coef_`/`intercept_` (no shrinking heuristic, R-DEV-7), mirroring `svm.rs`'s `SVC`/`SVR`. `fn fit`'s SMO loop treats `max_iter == 0` as unbounded via the sentinel guard `if self.max_iter != 0 && iter >= self.max_iter { break; }` (same form as `svm.rs`'s `smo_binary`/`smo_svr`), with the KKT-gap break (`i_max_grad - j_min_grad < tol`) terminating the default-0 fit. R-DEV-7 design difference (preserved contract, NOT a gap): estimator-level `kernel`(string)/`degree`/`coef0` are the type parameter `K` set by construction, `gamma` resolution is REQ-2, `verbose`/`random_state` unused (deterministic SMO). Pinned by `test_one_class_svm_default_params` (`nu==0.5`/`tol==1e-3`/`max_iter==0`/`cache_size==200`/`shrinking==true` vs the live `OneClassSVM.__init__` signature, R-DEV-2) + `test_one_class_svm_default_max_iter_converges` (default-0 fit converges, no infinite loop) + `test_one_class_svm_builder_pattern` (`with_shrinking`) in `one_class_svm.rs`. The 6 divergence pins use explicit `with_max_iter(1_000_000)` and stay green. |
| REQ-7 (ferray substrate) | NOT-STARTED | open prereq blocker #652. `one_class_svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` and computes on `ndarray`/`Vec<Vec<F>>`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (cf. `svm.md` REQ-10, `linear_svc.md` REQ-12). |

## Verification

The module unit tests exist but pin NO sklearn parity value, so no REQ can be
SHIPPED (R-CHAR-1/R-CHAR-3):

- `cargo test -p ferrolearn-linear one_class_svm` — the in-module
  `#[cfg(test)] mod tests` (`test_one_class_svm_fit`, `test_one_class_svm_inliers`,
  `test_one_class_svm_outlier_detection`, `test_one_class_svm_decision_function`,
  `test_one_class_svm_invalid_nu`, `test_one_class_svm_empty_data`,
  `test_one_class_svm_builder_pattern`, `test_one_class_svm_linear_kernel`,
  `test_one_class_svm_single_sample` in `one_class_svm.rs`) assert loose inlier
  floors (`inliers >= 6`), that a far point is an outlier, `df.len()==8`, the
  `nu`-range error, empty/single-sample handling, and the builder setters — they
  do NOT compare `dual_coef_`/`intercept_`/`support_`/`offset_`/`decision_function`/
  `score_samples`/`predict` values or shape against sklearn.
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the gaps; expected values per R-CHAR-3 come from
sklearn, never copied from ferrolearn):

```bash
# REQ-6: defaults
python3 -c "from sklearn.svm import OneClassSVM; import inspect; \
print({k:v.default for k,v in inspect.signature(OneClassSVM.__init__).parameters.items() if k!='self'})"
# kernel='rbf' degree=3 gamma='scale' coef0=0.0 tol=1e-3 nu=0.5 shrinking=True cache_size=200 max_iter=-1

# REQ-1/REQ-3/REQ-4/REQ-5: linear nu=0.5 fit (libsvm-scale dual_coef_)
python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
m=OneClassSVM(kernel='linear',nu=0.5).fit(X); \
print('support_',m.support_.tolist(),'n_support_',m.n_support_.tolist()); \
print('dual_coef_',np.round(m.dual_coef_,6).tolist(),'sum',float(m.dual_coef_.sum())); \
print('intercept_',np.round(m.intercept_,6).tolist(),'offset_',np.round(m.offset_,6).tolist()); \
print('coef_',np.round(m.coef_,6).tolist()); \
print('df',np.round(m.decision_function(X),6).tolist()); \
print('predict',m.predict(X).tolist()); \
print('score_samples',np.round(m.score_samples(X),6).tolist())"
# support_ [0,1,2,3]  n_support_ [4]
# dual_coef_ [[1.0,0.5,1.0,1.0]]  sum 3.5  (= nu*n = 0.5*7)
# intercept_ [-0.01]  offset_ [0.01]  coef_ [[0.05,0.05]]
# df [-0.01,0.0,-0.01,-0.01,0.0,0.0,0.29]  predict [-1,1,-1,-1,1,1,1]
# score_samples [0.0,0.01,0.0,0.0,0.01,0.01,0.3]   (= df + offset_)

# REQ-2: rbf gamma='auto' docstring oracle
python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
X=np.array([[0],[0.44],[0.45],[0.46],[1]],dtype=float); \
m=OneClassSVM(gamma='auto').fit(X); \
print('predict',m.predict(X).tolist(),'score_samples',np.round(m.score_samples(X),6).tolist())"
# predict [-1,1,1,1,-1]  score_samples [1.779873,2.054799,2.055605,2.056156,1.733285]
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; all seven REQs are currently NOT-STARTED.

## Blockers to open

- **#646** — REQ-1 of one_class_svm: pin the ONE_CLASS `nu` dual fit
  (`dual_coef_`/`intercept_`/`support_`/`n_support_`) against the live
  `OneClassSVM(kernel='linear',nu=0.5)` oracle on the 7×2 set; verify the
  normalized `Σα=1` solve is **rescaled to libsvm's `Σα=ν·n`** convention so the
  exposed `dual_coef_` matches `[[1,0.5,1,1]]` (`Σ=3.5`) and `rho`/`intercept_`
  match `-0.01` (`solve_one_class`, `svm.cpp:1710`).
- **#647** — REQ-2 of one_class_svm: resolve `gamma` at fit time against `X` in
  `fn fit` (call `Kernel::resolved_for_fit(x)` like `svm.rs`) so `gamma='scale'`
  (`1/(n_features·X.var())`, default) / `'auto'` (`1/n_features`) / float work;
  pin `_gamma` and the rbf-`auto` docstring oracle (`_base.py:235-243`).
- **#648** — REQ-3 of one_class_svm: expose `support_` (record SV training-row
  indices), `support_vectors_`, `n_support_` shape `(1,)`, `dual_coef_` shape
  `(1,n_SV)` (libsvm scale), `intercept_=-rho`, `offset_=-intercept_`
  (`_classes.py:1767`), linear `coef_=dual_coef_@support_vectors_`; pin against
  the live oracle layout.
- **#649** — REQ-4 of one_class_svm: add `score_samples(X)=decision_function(X)+
  offset_` (`_classes.py:1801`); ensure `decision_function` is in libsvm scale
  (gated on #646's rescale); pin df + score_samples values against the oracle.
- **#650** — REQ-5 of one_class_svm: change `predict` to libsvm's strict
  `(decision > 0) ? +1 : -1` (`svm.cpp:2837-2838`) and pin labels (including a
  constructed exact-zero boundary) against the live oracle.
- **#651** — REQ-6 of one_class_svm: align defaults `max_iter=-1` (no limit;
  currently `10000`) and `cache_size=200` (currently `1024`, unused); add the
  `shrinking` API field (no-op for this SMO, R-DEV-7); document the
  `kernel`/`degree`/`coef0` type-parameter design difference (`_classes.py:1709`).
- **#652** — REQ-7 of one_class_svm: migrate `one_class_svm.rs` off `ndarray`
  onto the ferray substrate (`ferray-core` arrays, ferray kernel/linear-algebra
  ops) per R-SUBSTRATE.
</content>
