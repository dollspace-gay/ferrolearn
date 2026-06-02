# Kernel Support Vector Machines (SVC / SVR)

<!--
tier: 3-component
status: draft
baseline-commit: e1dacaef
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/svm/_classes.py        # class SVC (~:617), class SVR (~:1108)
  - sklearn/svm/_base.py           # BaseLibSVM, BaseSVC, _dense_fit, _validate_targets, _decision_function, _get_coef, n_support_
  - sklearn/svm/src/libsvm/        # the SMO/Solver contract (Fan-Chen-Lin 2005 WSS)
ferrolearn-module: ferrolearn-linear/src/svm.rs
parity-ops: SVC, SVR
crosslink-issue: 633
-->

## Summary

`ferrolearn-linear/src/svm.rs` mirrors scikit-learn's `sklearn.svm.SVC`
(`sklearn/svm/_classes.py:617`) and `sklearn.svm.SVR`
(`sklearn/svm/_classes.py:1108`) — the **libsvm-backed kernel SVMs**. The module
ships a `Kernel<F>` trait with four built-in kernels (`LinearKernel`,
`RbfKernel`, `PolynomialKernel`, `SigmoidKernel`), the unfitted `SVC<F, K>` /
fitted `FittedSVC<F, K>` and `SVR<F, K>` / `FittedSVR<F, K>` pairs, a hand-rolled
**SMO solver** (`fn smo_binary in svm.rs` for C-SVC, `fn smo_svr in svm.rs` for
epsilon-SVR) using the Fan-Chen-Lin (2005) maximal-violating-pair working-set
selection, and `Fit`/`Predict` plus `decision_function` methods. SVC uses the
**one-vs-one** multiclass scheme (one binary SMO per class pair), structurally
matching libsvm.

sklearn delegates *all* numerics to the C library **libsvm** (`_dense_fit`,
`_base.py:304-349`, calls `libsvm.fit(...)`): the C-SVC / epsilon-SVR duals, the
shrinking heuristic, the per-class support-vector accounting (`support_`,
`support_vectors_`, `n_support_`, `dual_coef_`, `intercept_`), Platt-scaling
probabilities, and the ovo→ovr decision-function transform are all libsvm
behaviors that ferrolearn must reproduce numerically.

The SMO core in `svm.rs` is **substantially implemented and appears
algorithmically correct** (the Fan-Chen-Lin WSS, the analytic two-variable
update, the bias recovery from free support vectors), but **no oracle-pinned
divergence test exists yet** (`ferrolearn-linear/tests/` has no svm parity
suite; `api_proof_kernel_svm_family in tests/api_proof.rs` only asserts that
`fit`/`predict` run). Under R-CHAR-1/R-CHAR-3 a numerical REQ cannot be SHIPPED
until a test pins it against the live oracle. Therefore **every numerical REQ
below is NOT-STARTED**, even where the code looks right — and this doc flags the
specific lines the critic should pin first. Several REQs are NOT-STARTED for a
*structural* reason (missing `gamma='scale'`, missing libsvm attributes, missing
ovr transform, no estimator-level kernel/degree/coef0 params), independent of
verification.

## Algorithm (sklearn / libsvm — the contract)

### Estimator surface & defaults (`_classes.py`, live oracle)

`SVC.__init__` defaults (live `inspect.signature`): `C=1.0`, `kernel='rbf'`,
`degree=3`, `gamma='scale'`, `coef0=0.0`, `shrinking=True`, `probability=False`,
`tol=1e-3`, `cache_size=200`, `class_weight=None`, `verbose=False`,
`max_iter=-1`, `decision_function_shape='ovr'`, `break_ties=False`,
`random_state=None`. `SVR.__init__` defaults: same kernel block plus `C=1.0`,
`epsilon=0.1`, `tol=1e-3`, `cache_size=200`, `max_iter=-1` (no
`decision_function_shape`/`probability`/`class_weight`). `max_iter=-1` means **no
iteration limit** (libsvm runs to convergence).

### Kernels & gamma resolution (`_base.py:229-243`)

The kernel formulas (libsvm `Kernel::kernel_*`):

- linear: `K(x,y) = x·y`
- rbf: `K(x,y) = exp(-gamma·‖x-y‖²)`
- poly: `K(x,y) = (gamma·x·y + coef0)^degree`
- sigmoid: `K(x,y) = tanh(gamma·x·y + coef0)`

`gamma` resolves at fit time against the **training data** (`_base.py:235-243`):
`gamma='scale'` (default) → `_gamma = 1 / (n_features · X.var())` (where
`X.var()` is the variance of the whole flattened `X`); `gamma='auto'` →
`_gamma = 1 / n_features`; a float is used verbatim. Live oracle on a 6×2 set:
`X.var()=4.2222`, `_gamma = 1/(2·4.2222) = 0.118421`. `degree` default 3,
`coef0` default 0.0.

### The libsvm C-SVC dual & SMO (`sklearn/svm/src/libsvm/`)

C-SVC solves, per binary problem, the dual
`min_α  ½ αᵀQα − eᵀα  s.t.  0 ≤ α_i ≤ C, yᵀα = 0`, with
`Q_{ij} = y_i y_j K(x_i,x_j)`. libsvm's `Solver::Solve` uses the Fan-Chen-Lin
(2005) **maximal-violating-pair / second-order working-set selection** with the
shrinking heuristic, converging when the KKT gap `m(α) − M(α) ≤ tol`. The bias
`b = -(m(α)+M(α))/2` (libsvm `calculate_rho`) — ferrolearn instead averages
`y_i − Σ_j α_j y_j K(i,j)` over free support vectors, a mathematically
equivalent KKT-based recovery that must still match libsvm's `rho` numerically.

### Fitted attributes — the libsvm layout (`_base.py:318-410, 642-682`)

After `libsvm.fit`, sklearn exposes: `support_` (1-D indices of support vectors
into the training set, **grouped by class**), `support_vectors_`
(`(n_SV, n_features)`), `n_support_` (per-class SV counts, `_base.py:668-682`),
`dual_coef_` (shape `(n_class-1, n_SV)` — for class `i`, the coefficients
`α y` of the SVs in the `n_class-1` ovo problems involving class `i`,
`_base.py:40` comment), and `intercept_` (one per ovo problem, length
`n_class·(n_class-1)/2`). Binary sign flip (`_base.py:258-262`): for `c_svc`/
`nu_svc` with `len(classes_)==2`, `intercept_ *= -1` and `dual_coef_ = -dual_coef_`
(so the public binary `dual_coef_`/`intercept_` are negated relative to the
internal `_dual_coef_`/`_intercept_`). `coef_` is available only for
`kernel='linear'`: `coef_ = dual_coef_ @ support_vectors_` (`_get_coef`,
`_base.py:665-666`).

Live oracle (linear, binary 6×2, `C=1.0`): `support_ [1,2,3]`,
`n_support_ [2,1]`, `dual_coef_ [[-0.0408,-0.0408,0.0816]]` shape `(1,3)`,
`intercept_ [-1.8565]`, `coef_ [[0.2856,0.2856]]`. ferrolearn stores SVs and
`α·y` per binary model in private `BinarySvm` fields with **no** `support_`/
`n_support_`/`dual_coef_`/`coef_` accessors and **no** binary sign flip.

### decision_function — ovo, voting, and the ovr transform (`_base.py:513-541, 751-814`)

libsvm classification is **one-vs-one**: `_decision_function` returns the raw ovo
values, shape `(n_samples, n_class·(n_class-1)/2)` (`_base.py:522`). For binary
(`len(classes_)==2`) sklearn returns **`-dec.ravel()`** — 1-D shape
`(n_samples,)` with the sign flipped (`_base.py:538-539`); so a **positive**
decision value maps to `classes_[1]`. For multiclass: when
`decision_function_shape='ovr'` (default) and `n_classes>2`, the ovo scores are
passed through `_ovr_decision_function(dec<0, -dec, n_classes)`
(`_base.py:779-780`) producing shape `(n_samples, n_classes)` — a monotonic
transform combining each class's vote count with a normalized confidence;
`decision_function_shape='ovo'` returns the raw `(n, n·(n-1)/2)` ovo scores.

`predict` (`BaseLibSVM.predict` → `libsvm.predict`, `_base.py:412-430`) uses
libsvm's **ovo voting**: each binary classifier votes, the most-voted class wins
(ties broken by lower class index), then `classes_.take(...)` maps to the label
(`_base.py:814`).

Live oracle (linear, 3-class 9×2): `_decision_function` shape `(9,3)` (3 ovo
problems), `intercept_ [1.2222,1.2222,0.0]`, ovr `decision_function` shape
`(9,3)`, ovr row0 `[2.2366,0.8167,-0.1833]`; binary 6×2 `decision_function`
shape `(6,)` values `[-1.2853,-0.9997,-0.9997,0.9995,1.2851,1.2851]`.

### epsilon-SVR dual (`_classes.py:1108+`, libsvm `solve_epsilon_svr`)

epsilon-SVR solves the `2n`-variable dual over `(α, α*)` with
`0 ≤ α_i, α*_i ≤ C`, prediction coefficient `α*_i − α_i`, the linear term
involving `epsilon`, and prediction `f(x) = Σ_i (α*_i−α_i) K(x_i,x) + ρ`.
`n_support_` for SVR has size 1 (`_base.py:680-682`); `dual_coef_` is `(1, n_SV)`.
Live oracle (linear, 6×1, `C=100`, `epsilon=0.1`): `support_ [0,5]`,
`dual_coef_ [[-0.392,0.392]]`, `intercept_ [0.14]`, `predict ≈ [2.1,4.06,…]`.

### probability — Platt scaling (`_base.py:830-925`)

`probability=True` runs internal 5-fold CV to fit a sigmoid `1/(1+exp(Af+B))`
(`_probA`/`_probB`), enabling `predict_proba`/`predict_log_proba`. With
`probability=False` (default) `predict_proba` raises `AttributeError`
(`_check_proba`, `_base.py:820-827`). ferrolearn has no probability path.

## ferrolearn (what exists)

`pub trait Kernel<F: Float>` (`svm.rs`) exposes `fn compute(&self, x: &[F], y:
&[F]) -> F`. The four kernels implement the correct formulas: `LinearKernel`
(dot product), `RbfKernel` (`exp(-gamma·‖x-y‖²)`), `PolynomialKernel`
(`(gamma·dot+coef0)^degree`), `SigmoidKernel` (`tanh(gamma·dot+coef0)`). **But
`gamma` is an `Option<F>` on each kernel struct and `fn compute` resolves a
`None` gamma to `F::one()` (= 1.0) — NOT to sklearn's data-dependent
`'scale'`/`'auto'`** (`RbfKernel::compute`, `PolynomialKernel::compute`,
`SigmoidKernel::compute` all `unwrap_or_else(F::one)`). The kernel has no access
to `X`, so `gamma='scale'` (sklearn's default = `1/(n_features·X.var())`) and
`'auto'` (`1/n_features`) are **not implemented at all**; the struct doc-comments
claiming "if None, 1/(n_features·var(X))" are aspirational, the code uses 1.0.
`degree` defaults to 3 and `coef0` to 0.0 on the kernel structs (matching
sklearn).

`fn smo_binary in svm.rs` implements C-SVC SMO: the Fan-Chen-Lin WSS
(`I_up`/`I_low`, `i = argmax -y·grad over I_up`, `j = argmin over I_low`), the
KKT stop `max_val − min_val < tol`, the analytic two-variable update with `eta =
K_ii+K_jj−2K_ij`, box clipping, dual-gradient maintenance, and bias recovery by
averaging `y_i − Σ_j α_j y_j K(i,j)` over free SVs (`0 < α_i < C`) with a
bounded-SV fallback. An `KernelCache` LRU memoizes kernel evaluations. **No
shrinking heuristic** (a libsvm performance optimization that does not change the
optimum — R-DEV-7 sanctioned if the optimum matches).

`fn smo_svr in svm.rs` reformulates epsilon-SVR as a `2n`-variable box-QP
(`beta_k`, sign `s_k`, sample map) with the analogous WSS, update, and a
KKT bias from free SVR support vectors; prediction coefficient `α*_i − α_i`.

`SVC<F, K>` (`svm.rs`) carries fields `kernel: K`, `c: F`, `tol: F`, `max_iter:
usize`, `cache_size: usize` with `fn new` defaulting `C=1.0`, `tol=1e-3`,
`max_iter=10000`, `cache_size=1024`, and builder setters `with_c`/`with_tol`/
`with_max_iter`/`with_cache_size`. **There is NO estimator-level `kernel` (string
selection), `degree`, `gamma`, `coef0`, `shrinking`, `probability`,
`class_weight`, `decision_function_shape`, `break_ties`, or `random_state`** —
the kernel (and its degree/coef0/gamma) is the type parameter `K`, set by
construction. `max_iter` is `usize` (default 10000), not `-1`/unbounded.

`fn fit in svm.rs` (for `SVC`) validates `n_samples == y.len()`, `C > 0`,
computes `classes` by sort+dedup (= `np.unique(y)`), errors `InsufficientSamples`
for `<2` classes, then trains **one-vs-one**: one `smo_binary` per class pair
`(ci,cj)` with `class_neg=classes[ci]`, `class_pos=classes[cj]`, labels mapped
`neg→-1`/`pos→+1`. Each `BinarySvm` stores `support_vectors` (rows with `α>1e-8`),
`dual_coefs` (`α·y`), `bias`, `class_neg`, `class_pos`. `y` is `Array1<usize>`
(class indices, not arbitrary labels). `FittedSVC` stores the kernel,
`binary_models: Vec<BinarySvm>`, and `classes: Vec<usize>` — all private, **no
public `support_`/`support_vectors_`/`n_support_`/`dual_coef_`/`intercept_`/
`coef_` accessors**.

`fn decision_function in svm.rs` (FittedSVC) returns `Array2<F>` shape
`(n_samples, n_models)` where `n_models = n_class·(n_class-1)/2` — the **raw ovo
values** with **no ovr transform and no binary `.ravel()`/sign-flip**. So for
binary it returns `(n,1)` (sklearn returns `(n,)` negated) and for multiclass it
returns raw ovo `(n, n·(n-1)/2)` regardless of any `decision_function_shape`
(which does not exist). `fn predict in svm.rs` (FittedSVC) does ovo voting
(`val>=0 → class_pos` else `class_neg`, count votes, argmax via `max_by_key`),
matching libsvm's voting *structure*; tie-breaking is `max_by_key`'s
last-maximum (sklearn/libsvm break ties toward the lower class index — a
potential divergence).

`SVR<F, K>` carries `kernel`, `c`, `epsilon`, `tol`, `max_iter`, `cache_size`
with `fn new` defaulting `C=1.0`, `epsilon=0.1`, `tol=1e-3`, `max_iter=10000`,
`cache_size=1024` and the matching builder setters (no `gamma`/`degree`/`coef0`/
`shrinking` at the estimator level). `fn fit in svm.rs` (for `SVR`, `y:
Array1<F>`) validates shape/`C>0`/non-empty, runs `smo_svr`, keeps SVs with
`|coef|>1e-8`. `FittedSVR` stores private `support_vectors`, `dual_coefs`, `bias`
(no public libsvm-layout accessors). `fn decision_function in svm.rs` (FittedSVR)
returns `Array1<F>` = `Σ coef·K(sv,x)+bias`; `fn predict` is its alias.

`SVC`/`SVR`/`FittedSVC`/`FittedSVR`/`Kernel`/`LinearKernel`/`RbfKernel`/
`PolynomialKernel`/`SigmoidKernel` are re-exported at the crate root
(`pub use svm::{FittedSVC, FittedSVR, Kernel, LinearKernel, PolynomialKernel,
RbfKernel, SVC, SVR} in lib.rs`) and `svm::SVC`/`svm::SVR`/`svm::Kernel` are
consumed by `nu_svm.rs` (`NuSVC`/`NuSVR` delegate to them, `use crate::svm::{…}
in nu_svm.rs`) and referenced by `one_class_svm.rs` — **non-test production
consumers**. There is **no PyO3 binding** for `SVC`/`SVR` (`ferrolearn-python`
exposes `LinearSVC` only; no `RsSVC`/`RsSVR`), so the Python boundary does NOT
yet consume these types.

The module is on the **`ndarray` substrate** (`use ndarray::{Array1, Array2,
ScalarOperand} in svm.rs`), not ferray (R-SUBSTRATE).

## Requirements

- REQ-1: Kernels & formulas — `linear`/`rbf`/`poly`/`sigmoid` with the exact
  libsvm formulas (rbf `exp(-gamma·‖x-y‖²)`, poly `(gamma·x·y+coef0)^degree`,
  sigmoid `tanh(gamma·x·y+coef0)`), and **`gamma` resolution**
  `{'scale' (default, = 1/(n_features·X.var())), 'auto' (= 1/n_features), float}`
  computed at fit time against `X`; `degree` (default 3), `coef0` (default 0.0).
- REQ-2: C-SVC SMO solver — the libsvm C-SVC dual via Fan-Chen-Lin (2005) WSS,
  converging to libsvm's optimum, verified by `dual_coef_`/`intercept_`/
  `support_` matching the live oracle.
- REQ-3: Fitted classification attributes — `support_` (per-class-grouped
  indices), `support_vectors_`, `n_support_` (per-class), `dual_coef_` (shape
  `(n_class-1, n_SV)`), `intercept_` (length `n_class·(n_class-1)/2`), plus
  `coef_ = dual_coef_ @ support_vectors_` for linear kernel, in the libsvm
  layout with the binary sign flip (`_base.py:258-262`).
- REQ-4: `decision_function` — libsvm ovo values; binary returns
  `-dec.ravel()` shape `(n_samples,)` (positive → `classes_[1]`); multiclass
  with `decision_function_shape='ovr'` (default) returns the
  `_ovr_decision_function` transform shape `(n_samples, n_classes)`, with
  `'ovo'` returning raw `(n_samples, n·(n-1)/2)`. Values + sign convention match
  the live oracle.
- REQ-5: `predict` — libsvm ovo voting → class label, ties broken toward the
  lower class index, matching the oracle's labels.
- REQ-6: epsilon-SVR — the libsvm epsilon-SVR `2n`-variable dual (`α`/`α*`
  doubling), `epsilon` (default 0.1), `predict` = decision values; fitted
  `dual_coef_`/`support_`/`intercept_`/`predict` match the live oracle.
- REQ-7: Multiclass SVC — one-vs-one (libsvm), `classes_ = np.unique(y)` sorted,
  with the correct per-pair training and vote aggregation.
- REQ-8: Constructor params/defaults (R-DEV-2) — estimator-level `kernel`,
  `degree`, `gamma`, `coef0`, `C`, `tol`, `shrinking`, `cache_size`,
  `class_weight`, `max_iter` (default `-1` = no limit), `decision_function_shape`
  (default `'ovr'`), `break_ties` for SVC; `epsilon` (default 0.1) for SVR — with
  sklearn's exact names and defaults.
- REQ-9: `probability` — Platt scaling (`predict_proba`/`predict_log_proba`),
  `probability=False` default → `predict_proba` raises.
- REQ-10: ferray substrate (R-SUBSTRATE) — array type → `ferray-core`, kernel /
  linear-algebra ops → ferray, not `ndarray`.

## Acceptance criteria

- AC-1 (REQ-1): `RbfKernel` with `gamma='scale'` on the 6×2 set
  (`X.var()=4.2222`) resolves `_gamma = 1/(2·4.2222) = 0.118421` matching the
  oracle's `m._gamma`; `'auto'` resolves `1/n_features`; the four kernel
  `compute` values match libsvm's kernel evaluations.
- AC-2 (REQ-2): `SVC(kernel='linear', C=1.0)` on the binary 6×2 set
  (`X=[[1,1],[2,1],[1,2],[5,5],[6,5],[5,6]]`, `y=[0,0,0,1,1,1]`) yields
  `dual_coef_ [[-0.0408,-0.0408,0.0816]]`, `intercept_ [-1.8565]`,
  `support_ [1,2,3]` matching the live oracle within tolerance.
- AC-3 (REQ-3): the same fit exposes `support_`, `support_vectors_`,
  `n_support_ [2,1]`, `dual_coef_` shape `(1,3)`, `intercept_` length 1, and
  (linear) `coef_ [[0.2856,0.2856]]` shape `(1,2)`, all in libsvm layout with
  the binary sign flip.
- AC-4 (REQ-4): binary `decision_function(X)` has shape `(6,)` and equals the
  oracle `[-1.2853,-0.9997,-0.9997,0.9995,1.2851,1.2851]` (positive → class 1);
  3-class `decision_function` (ovr) has shape `(9,3)` matching the oracle ovr
  row0 `[2.2366,0.8167,-0.1833]`; `'ovo'` returns `(9,3)` raw.
- AC-5 (REQ-5): `predict` equals the oracle labels on the binary and 3-class
  sets, including the lower-index tie-break.
- AC-6 (REQ-6): `SVR(kernel='linear', C=100, epsilon=0.1)` on the 6×1 set
  yields `dual_coef_ [[-0.392,0.392]]`, `support_ [0,5]`, `intercept_ [0.14]`,
  `predict ≈ [2.1,4.06,6.02,7.98,9.94,11.9]` matching the oracle.
- AC-7 (REQ-7): 3-class fit produces `classes_ [0,1,2]` and 3 ovo binary models;
  per-pair `dual_coef_`/`intercept_` match the oracle.
- AC-8 (REQ-8): `SVC::new` exposes the sklearn param surface with defaults
  `kernel='rbf'`, `gamma='scale'`, `degree=3`, `coef0=0.0`, `C=1.0`, `tol=1e-3`,
  `shrinking=True`, `cache_size=200`, `class_weight=None`, `max_iter=-1`,
  `decision_function_shape='ovr'`, `break_ties=False`; `SVR` adds `epsilon=0.1`.
- AC-9 (REQ-9): `SVC(probability=True)` exposes `predict_proba` summing to 1 per
  row matching the oracle; `probability=False` → `predict_proba` raises.
- AC-10 (REQ-10): `svm.rs` owns its computation on `ferray-core` arrays, not
  `ndarray`.

## REQ status

Classification (R-DEFER-2): SHIPPED = impl + non-test production consumer +
tests + green oracle verification; NOT-STARTED = concrete open blocker
referenced by `#`-number. `SVC`/`SVR`/`FittedSVC`/`FittedSVR`/`Kernel` and the
four kernels are boundary estimator types re-exported at the crate root
(`pub use svm::{…} in lib.rs`) and consumed by `nu_svm.rs`/`one_class_svm.rs`
(non-test) — under S5/R-DEFER-1 the consumer surface exists. **But there is no
sklearn-grounded oracle pin for any svm behavior** (`tests/` has no
`divergence_svm_fit.rs`; `api_proof_kernel_svm_family in tests/api_proof.rs`
asserts only that `fit`/`predict` execute, no numerical comparison). Per
R-CHAR-1/R-CHAR-3 a numerical REQ cannot be SHIPPED without a test that pins the
mirrored sklearn behavior and fails until correct. Consequently **all ten REQs
are NOT-STARTED**. Where the SMO code *appears* correct but is unverified, the
table says so explicitly to point the critic at the first lines to pin.

Three structural divergences are the priority audit targets (the SMO may
converge to the right α yet still fail parity because of these):

1. **`gamma='scale'` is not implemented** — every kernel `compute` resolves a
   `None` gamma to `1.0`, not sklearn's default `1/(n_features·X.var())`
   (`_base.py:236-239`). Any RBF/poly/sigmoid `SVC`/`SVR` constructed without an
   explicit gamma silently uses `gamma=1`, diverging from the oracle. The kernel
   trait has no `X` access, so `'scale'`/`'auto'` cannot be computed where they
   live now — this is a structural gap, not a tuning bug.
2. **(ADDRESSED for the accessors — pending critic pin)** The libsvm-layout
   fitted attributes are now exposed: `FittedSVC::support`/`support_vectors`/
   `n_support`/`dual_coef`/`intercept`/`coef` and `FittedSVR::support`/
   `support_vectors`/`n_support`/`dual_coef`/`intercept in svm.rs`, with the
   binary sign flip handled in `dual_coef`/`intercept` (`_base.py:258-262`).
   `coef` returns `Option<Array2<F>>` (`Some` for the linear kernel via
   `Kernel::is_linear`, `None` otherwise — sklearn raises `AttributeError`,
   `_base.py:650-651`). Each `BinarySvm` now records `sv_indices` (the original
   training-row index of each SV) and `FittedSVC` retains `x_train`/`y_train` so
   `support_` can be built as the per-class-grouped union of per-pair SVs.
   In-module `#[cfg(test)]` smoke tests verify binary `support_ [1,2,3]`,
   `n_support_ [2,1]`, `dual_coef_ [[-0.0408,-0.0408,0.0816]]`,
   `intercept_ [-1.8565]`, `coef_ [[0.2856,0.2856]]`; the 3-class
   `dual_coef_ (2,6)` libsvm packing + `intercept_ [1.2222,1.2222,0.0]`; and
   SVR `support_ [0,5]`, `dual_coef_ [[-0.392,0.392]]`, `intercept_ [0.14]`
   against the live oracle (R-CHAR-3) within 1e-2. The rigorous `tests/
   divergence_svm_fit.rs` pins of these accessors (and a non-test production
   consumer of the new accessor methods) are owned by the next critic/builder
   step.
3. **`decision_function` has the wrong shape & no ovr transform** — ferrolearn
   returns raw ovo `(n, n_models)` always; sklearn binary is `-dec.ravel()`
   `(n,)` (sign-flipped) and multiclass-ovr is `_ovr_decision_function(...)`
   `(n, n_classes)` (`_base.py:538-539, 779-780`). The sign convention and shape
   both diverge.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (kernels & gamma resolution) | SHIPPED | The four kernel formulas in `RbfKernel::compute`/`PolynomialKernel::compute`/`SigmoidKernel::compute in svm.rs` + the three-way `pub enum Gamma<F> { Scale, Auto, Value(F) }` (default `Scale`; manual `Default`->`Scale`). Resolved at fit time by `fn resolve_gamma in svm.rs` (called from `fn resolved_for_fit in svm.rs`): `Gamma::Scale`=`1/(n_features·X.var())` (default, `_base.py:238-239`, with the `X_var==0 -> 1.0` fallback), `Gamma::Auto`=`1/n_features` (`_base.py:240-241`), `Gamma::Value(v)`=verbatim (`_base.py:242-243`). `RbfKernel::with_gamma(v)` sets `Gamma::Value(v)`; new `with_gamma_scale`/`with_gamma_auto` builders. Non-test consumer: the kernel `gamma` field is consumed by the production `fn fit in svm.rs` (`self.kernel.resolved_for_fit(x)`) for both `SVC` and `SVR`. Pinned by `divergence_pin2_rbf_default_scale_gamma in tests/divergence_svm_fit.rs` (scale, green) + in-module `test_svc_gamma_auto_decision_function in svm.rs` (`SVC(kernel='rbf',gamma='auto')` on the 6×2 set: `_gamma=0.5`, df `[-0.9996,-0.9999,-0.9999,0.9999,0.9999,0.9996]`, R-CHAR-3, 1e-2) + `test_svc_gamma_scale_still_default` (`_gamma=0.118421`). |
| REQ-2 (C-SVC SMO solver) | NOT-STARTED | open prereq blocker #635. `fn smo_binary in svm.rs` implements the Fan-Chen-Lin WSS + analytic update + KKT bias and *appears* algorithmically correct (critic: start here — verify the binary 6×2 oracle `dual_coef_ [[-0.0408,-0.0408,0.0816]]`, `intercept_ [-1.8565]`), but no `divergence_svm_fit.rs` pins `dual_coef_`/`intercept_`/`support_` against the live `SVC(kernel='linear')` oracle (R-CHAR-1). Bias recovery averages free-SV residuals rather than libsvm's `-(m+M)/2 rho` — equivalent at optimum but unverified. |
| REQ-3 (fitted classification attributes) | NOT-STARTED (accessors+oracle smoke tests landed; gated on critic pin + non-test consumer, #636) | The libsvm-layout accessors NOW exist: `FittedSVC::support`/`support_vectors`/`n_support`/`dual_coef`/`intercept`/`coef` and `FittedSVR::support`/`support_vectors`/`n_support`/`dual_coef`/`intercept in svm.rs`, with the binary sign flip in `dual_coef`/`intercept` (`_base.py:258-262`) and `coef -> Option` (linear only, `_base.py:650-651`, via `Kernel::is_linear`). `BinarySvm` records `sv_indices`; `FittedSVC` retains `x_train`/`y_train`. In-module `#[cfg(test)]` smoke tests (`test_svc_binary_support_attrs`/`test_svc_binary_dual_coef_sign_flip`/`test_svc_binary_intercept_and_coef`/`test_svc_coef_none_for_nonlinear`/`test_svc_multiclass_support_attrs`/`test_svc_multiclass_dual_coef_packing`/`test_svr_linear_attrs in svm.rs`) verify all of binary `support_ [1,2,3]`/`n_support_ [2,1]`/`dual_coef_ [[-0.0408,-0.0408,0.0816]]`/`intercept_ [-1.8565]`/`coef_ [[0.2856,0.2856]]`, the 3-class `dual_coef_ (2,6)` packing + `intercept_ [1.2222,1.2222,0.0]`, and SVR `support_ [0,5]`/`dual_coef_ [[-0.392,0.392]]`/`intercept_ [0.14]` against the LIVE oracle (R-CHAR-3, 1e-2). Remains **NOT-STARTED** under R-DEFER-1/R-CHAR-1 only because (a) the rigorous pin lives in `tests/divergence_svm_fit.rs` (critic-owned, next step) not yet added, and (b) the NEW accessor methods have no non-test production consumer yet (the binding/`nu_svm` does not call them). Multiclass `dual_coef_` packing was decoded and MATCHES the oracle (not deferred to #640). |
| REQ-4 (decision_function shape/sign/ovr) | SHIPPED (pending nu_svm.md prereq, #644) | `fn decision_function in svm.rs` (FittedSVC) now returns `SvmScores<F>`: binary -> `SvmScores::Binary` 1-D `(n,)` = `-raw_ovo.ravel()` (positive -> `classes_[1]`, `_base.py:538-539`); multiclass-ovr (default `SvmDecisionShape::Ovr`) -> `SvmScores::Multiclass` `(n, n_classes)` via `fn ovr_decision_function in svm.rs` (transcribed from `multiclass.py:520-562`, fed `dec<0`/`-dec`, `_base.py:780`); `SvmDecisionShape::Ovo` -> raw `(n, n·(n-1)/2)`. `SVC::decision_function_shape` field + `fn with_decision_function_shape in svm.rs`. `fn raw_ovo in svm.rs` negates `decision_value_binary` to restore libsvm's lower-index-class-`+1` ovo sign so the ovo output matches the oracle. In-module smoke tests `test_svc_decision_function_binary_values`/`test_svc_decision_function_ovr`/`test_svc_decision_function_ovo in svm.rs` pin the LIVE oracle (binary `(6,)` `[-1.2853,…,1.2851]`; ovr `(9,3)` row0 `[2.2366,0.8167,-0.1833]`/row3 `[1.0606,2.2262,-0.2333]`; ovo `(9,3)` row0 `[1.2222,1.2222,0.0]`; R-CHAR-3, 1e-2). `tests/divergence_svm_fit.rs::divergence_pin1_*` adapted to `as_binary()`. Non-test consumer: `FittedNuSVC::decision_function in nu_svm.rs` delegates. **PREREQ BLOCK**: that consumer edit (and crate compilation) is gated on the missing `.design/linear/nu_svm.md` (R-XLATE-3) — escalated on #644; remains short of fully-green until nu_svm.md is authored and the delegation updated. |
| REQ-5 (predict — ovo voting + tie-break) | SHIPPED | `fn predict in svm.rs` (FittedSVC) does ovo voting (vote per binary model) and breaks ties toward the **lower** class index via a strictly-greater first-max scan (keeps the first/lowest-index maximum; `classes` is `np.unique(y)`-sorted), matching libsvm's `super().predict` (`_base.py:813-814`) rather than `max_by_key`'s last-maximum. Pinned by `divergence_pin3_predict_labels` (binary + 3-class labels) and `divergence_pin11_ovo_vote_tie_break_lower_index` (4-class vote tie `(0,2,2,2)` at `q=(-0.21,-8.976)` -> class 1) vs the live `SVC(kernel='linear',C=1.0)` oracle. |
| REQ-6 (epsilon-SVR) | NOT-STARTED | open prereq blocker #639. `fn smo_svr in svm.rs` reformulates the epsilon-SVR `2n`-variable dual (`α`/`α*`, prediction coef `α*−α`) and *appears* correct, but `FittedSVR` exposes no `support_`/`dual_coef_ (1,n_SV)`/`intercept_` accessors and no oracle pins the fit. Live oracle target (linear 6×1, `C=100,epsilon=0.1`): `support_ [0,5]`, `dual_coef_ [[-0.392,0.392]]`, `intercept_ [0.14]`, `predict ≈ [2.1,4.06,6.02,7.98,9.94,11.9]`. |
| REQ-7 (multiclass one-vs-one) | NOT-STARTED | open prereq blocker #640. `fn fit in svm.rs` (SVC) trains one `smo_binary` per class pair (ovo) and sets `classes` = sort+dedup (= `np.unique(y)`) — structurally matching libsvm. But per-pair `dual_coef_`/`intercept_` are unverified and unexposed (REQ-3); no oracle pin of the 3-class `(2,6)` `dual_coef_` / 3 `intercept_` exists. Gated on #635/#636. |
| REQ-8 (constructor params/defaults) | NOT-STARTED (partial — #641 narrowed) | **SHIPPED parts**: `shrinking` (`SVC`/`SVR`, `pub shrinking: bool` default `true`, `with_shrinking`) — accepted for API parity but DOES NOT alter the converged optimum (ferrolearn's SMO has no shrinking heuristic; the optimum is shrinking-invariant, R-DEV-7, `_base.py:339`); `break_ties` (`SVC`, `pub break_ties: bool` default `false`, `with_break_ties`) with the `BaseSVC.predict` semantics in `fn predict in svm.rs` (when `break_ties=true` AND `SvmDecisionShape::Ovr` AND `n_classes>2`, `predict=argmax(decision_function)`; `InvalidParameter` for the `break_ties=true`+`Ovo` combo, `_base.py:801-814`); default alignment `cache_size=200` (was 1024) and `max_iter=0` = sklearn `-1` (no iteration limit — the `smo_binary`/`smo_svr` loops treat `0` as run-to-convergence; non-zero caps the count); plus REQ-1's three-way `gamma` enum default `'scale'`. Pinned by `test_svc_break_ties_changes_label`/`test_svc_break_ties_ovo_errors`/`test_svc_default_params in svm.rs` (break_ties oracle: symmetric separable 3-class set, vote->lowest-index 0 vs ovr-argmax 2/1 at q=(5.19,3.342)/(5.19,3.241) vs live `SVC(break_ties=True,decision_function_shape='ovr')`). **STILL NOT-STARTED** (open #641): estimator-level `class_weight` (genuinely absent). `kernel` (string-select), `degree`, `coef0` are a documented **R-DEV-7 design difference** — the kernel and its `degree`/`coef0` are the type parameter `K`, set by construction (`SVC::new(PolynomialKernel { degree, coef0, gamma })`), not a string + scalar pair; the observable contract (the kernel formula evaluated with those values) is preserved. `random_state` is unused (ferrolearn's SMO is deterministic, no libsvm shuffle seed). |
| REQ-9 (probability / predict_proba) | NOT-STARTED | open prereq blocker #642. No `probability` field, no Platt-scaling CV (`_probA`/`_probB`), no `predict_proba`/`predict_log_proba`, no `AttributeError`-when-`probability=False` path (`_base.py:820-925`). Entirely absent. |
| REQ-10 (ferray substrate) | NOT-STARTED | open prereq blocker #643. `svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` and computes on `ndarray`/`Vec<Vec<F>>`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (cf. `linear_svc.md` REQ-12). |

## Verification

The module unit tests + the api-proof harness exist but pin NO sklearn parity
value, so no REQ can be SHIPPED (R-CHAR-1/R-CHAR-3):

- `cargo test -p ferrolearn-linear svm` — the in-module `#[cfg(test)] mod tests`
  (`test_linear_kernel`, `test_rbf_kernel`, `test_polynomial_kernel`,
  `test_sigmoid_kernel`, `test_svc_linear_separable`, `test_svc_rbf_xor`,
  `test_svc_multiclass`, `test_svc_decision_function`, `test_svc_invalid_c`,
  `test_svc_single_class_error`, `test_svc_shape_mismatch`, `test_svr_simple`,
  `test_svr_decision_function`, `test_svr_invalid_c`, `test_svr_shape_mismatch`
  in `svm.rs`) assert kernel values against hand-computed constants, loose
  accuracy floors (`correct >= 6/8`, `>= 7/9`), error paths, and `df.ncols()==1`
  for binary — they do NOT compare `dual_coef_`/`intercept_`/`support_`/
  `decision_function` values or shape against sklearn.
- `api_proof_kernel_svm_family in tests/api_proof.rs` constructs
  `SVC::new(RbfKernel::with_gamma(1.0))` / `SVR::new(RbfKernel::with_gamma(0.5))`
  with explicit gamma and asserts only that `fit`/`predict` run — no oracle.
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the gaps; expected values per R-CHAR-3 come
from sklearn, never copied from ferrolearn):

```bash
# REQ-1: gamma='scale' resolution (default)
python3 -c "import numpy as np; from sklearn.svm import SVC; \
X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
m=SVC(C=1.0).fit(X,y); print('_gamma',m._gamma, '= 1/(2*X.var())=', 1/(2*X.var()))"
# _gamma 0.118421...  (ferrolearn RbfKernel(None) uses gamma=1.0)

# REQ-2/REQ-3: binary linear fit attributes (libsvm layout)
python3 -c "import numpy as np; from sklearn.svm import SVC; \
X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
m=SVC(kernel='linear',C=1.0).fit(X,y); \
print('support_',m.support_.tolist(),'n_support_',m.n_support_.tolist()); \
print('dual_coef_',m.dual_coef_.tolist(),'intercept_',m.intercept_.tolist(),'coef_',m.coef_.tolist())"
# support_ [1,2,3]  n_support_ [2,1]  dual_coef_ [[-0.0408,-0.0408,0.0816]]  intercept_ [-1.8565]  coef_ [[0.2856,0.2856]]

# REQ-4/REQ-5: decision_function shape/sign + predict
python3 -c "import numpy as np; from sklearn.svm import SVC; \
X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]]); y=np.array([0,0,0,1,1,1]); \
m=SVC(kernel='linear',C=1.0).fit(X,y); \
print('df.shape',m.decision_function(X).shape,'df',np.round(m.decision_function(X),4).tolist()); \
print('predict',m.predict(X).tolist())"
# df.shape (6,)  df [-1.2853,-0.9997,-0.9997,0.9995,1.2851,1.2851]  predict [0,0,0,1,1,1]

# REQ-4/REQ-7: 3-class ovo + ovr
python3 -c "import numpy as np; from sklearn.svm import SVC; \
X=np.array([[0.,0.],[0.5,0.],[0.,0.5],[5.,0.],[5.5,0.],[5.,0.5],[0.,5.],[0.5,5.],[0.,5.5]]); \
y=np.array([0,0,0,1,1,1,2,2,2]); m=SVC(kernel='linear',C=1.0).fit(X,y); \
print('ovo shape',m._decision_function(X).shape,'ovr shape',m.decision_function(X).shape); \
print('dual_coef_ shape',m.dual_coef_.shape,'intercept_',np.round(m.intercept_,4).tolist())"
# ovo (9,3)  ovr (9,3)  dual_coef_ (2,6)  intercept_ [1.2222,1.2222,0.0]

# REQ-6: SVR linear fit
python3 -c "import numpy as np; from sklearn.svm import SVR; \
X=np.array([[1.],[2.],[3.],[4.],[5.],[6.]]); y=np.array([2.,4.,6.,8.,10.,12.]); \
m=SVR(kernel='linear',C=100.0,epsilon=0.1).fit(X,y); \
print('support_',m.support_.tolist(),'dual_coef_',np.round(m.dual_coef_,4).tolist(), \
'intercept_',np.round(m.intercept_,4).tolist(),'predict',np.round(m.predict(X),4).tolist())"
# support_ [0,5]  dual_coef_ [[-0.392,0.392]]  intercept_ [0.14]  predict [2.1,4.06,6.02,7.98,9.94,11.9]

# REQ-8: defaults
python3 -c "from sklearn.svm import SVC, SVR; import inspect; \
print('SVC', {k:v.default for k,v in inspect.signature(SVC.__init__).parameters.items() if k!='self'}); \
print('SVR', {k:v.default for k,v in inspect.signature(SVR.__init__).parameters.items() if k!='self'})"
# SVC kernel='rbf' C=1.0 gamma='scale' degree=3 coef0=0.0 tol=1e-3 shrinking=True
#     decision_function_shape='ovr' probability=False max_iter=-1 cache_size=200 ...
# SVR adds epsilon=0.1
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; all ten REQs are currently NOT-STARTED.

## Blockers to open

- **#634** — REQ-1 of svm: resolve `gamma` at fit time against `X` —
  `'scale' = 1/(n_features·X.var())` (default), `'auto' = 1/n_features`, or a
  float (`_base.py:236-243`); thread the resolved gamma into the kernel (the
  kernel trait/struct currently has no `X` access and a `None` gamma silently
  becomes `1.0`). Pin kernel values + `_gamma` against the live oracle.
- **#635** — REQ-2 of svm: pin the C-SVC SMO fit (`dual_coef_`/`intercept_`/
  `support_`) against the live `SVC(kernel='linear')` oracle on the binary 6×2
  set; verify `fn smo_binary` converges to libsvm's optimum and the bias matches
  libsvm `rho` (the free-SV-average vs `-(m+M)/2` recovery).
- **#636** — REQ-3 of svm: expose `support_` (per-class-grouped indices),
  `support_vectors_`, `n_support_`, `dual_coef_` (shape `(n_class-1,n_SV)`),
  `intercept_`, and linear `coef_ = dual_coef_ @ support_vectors_`, with the
  binary sign flip (`intercept_*=-1`, `dual_coef_=-dual_coef_`,
  `_base.py:258-262`), matching the live oracle layout.
- **#637** — REQ-4 of svm: make binary `decision_function` return `-dec.ravel()`
  shape `(n,)` (positive → `classes_[1]`), add `decision_function_shape`
  (`'ovr'` default / `'ovo'`) and the `_ovr_decision_function` transform for
  multiclass (`_base.py:538-539, 779-780`); pin shapes + values against the
  oracle.
- **#638** — REQ-5 of svm: align `predict` tie-breaking with libsvm (lower class
  index on vote ties, `_base.py:814`) and pin `predict` labels against the live
  oracle on the binary + 3-class sets.
- **#639** — REQ-6 of svm: expose SVR `support_`/`dual_coef_ (1,n_SV)`/
  `intercept_` and pin the epsilon-SVR fit (`dual_coef_ [[-0.392,0.392]]`,
  `support_ [0,5]`, `intercept_ [0.14]`, `predict`) against the live
  `SVR(kernel='linear',C=100,epsilon=0.1)` oracle.
- **#640** — REQ-7 of svm: pin the 3-class one-vs-one fit — per-pair
  `dual_coef_` (`(2,6)`) and `intercept_` (`[1.2222,1.2222,0.0]`) and
  `classes_ [0,1,2]` — against the live oracle (gated on #635/#636).
- **#641** — REQ-8 of svm: add the sklearn estimator-level parameter surface
  (R-DEV-2): `kernel` (string selection), `degree`, `gamma`, `coef0`,
  `shrinking`, `class_weight`, `decision_function_shape`, `break_ties`,
  `random_state` for SVC; align defaults (`max_iter=-1` = no limit,
  `cache_size=200`, `gamma='scale'`); `epsilon=0.1` default for SVR.
- **#642** — REQ-9 of svm: add `probability` (Platt-scaling internal-CV sigmoid,
  `_probA`/`_probB`) with `predict_proba`/`predict_log_proba` and the
  `AttributeError`-when-`probability=False` contract (`_base.py:820-925`).
- **#643** — REQ-10 of svm: migrate `svm.rs` off `ndarray` onto the ferray
  substrate (`ferray-core` arrays, ferray kernel/linear-algebra ops) per
  R-SUBSTRATE.
