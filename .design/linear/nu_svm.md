# Nu-parameterized Support Vector Machines (NuSVC / NuSVR)

<!--
tier: 3-component
status: draft
baseline-commit: 351f768a
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/svm/_classes.py        # class NuSVC (:898, _impl="nu_svc" :1112), class NuSVR (:1376, _impl="nu_svr" :1521)
  - sklearn/svm/_base.py           # BaseLibSVM, BaseSVC, LIBSVM_IMPL :30, _dense_fit, _validate_targets :736, decision_function :751
ferrolearn-module: ferrolearn-linear/src/nu_svm.rs
parity-ops: NuSVC, NuSVR
crosslink-issue: 644
-->

## Summary

`ferrolearn-linear/src/nu_svm.rs` mirrors scikit-learn's `sklearn.svm.NuSVC`
(`sklearn/svm/_classes.py:898`, `_impl = "nu_svc"` at `:1112`) and
`sklearn.svm.NuSVR` (`sklearn/svm/_classes.py:1376`, `_impl = "nu_svr"` at
`:1521`) — the **nu-parameterized libsvm SVMs**. In place of the penalty
parameter `C`, the user supplies `nu ∈ (0, 1]`, an upper bound on the fraction
of margin errors and a lower bound on the fraction of support vectors. The
module ships an unfitted `NuSVC<F, K>` / fitted `FittedNuSVC<F, K>` pair and an
unfitted `NuSVR<F, K>` / fitted `FittedNuSVR<F, K>` pair, both re-exported at the
crate root (`pub use nu_svm::{FittedNuSVC, FittedNuSVR, NuSVC, NuSVR} in
lib.rs`).

**The ferrolearn implementation now runs the genuine libsvm nu-SVM solver.**
`crate::svm::solve_nu_svc`/`solve_nu_svr` (built on the shared `fn
solver_nu_core` — the libsvm `Solver_NU` with its nu-specific second-order
working-set selection and two-group `rho`/`r` recovery) solve the actual
nu-parameterized duals (the nu-SVC dual with `eᵀα = nu·l`; the `2l`-variable
nu-SVR dual with the learned tube). `NuSVC::fit`/`NuSVR::fit` call these per
one-vs-one pair — NOT a C-SVC / epsilon-SVR with a re-scaled `C`. `NuSVR` carries
both `nu` AND `C` (default `1.0`), with no `epsilon`. The fitted NuSVC reuses the
full SVC fitted-attribute / `decision_function` / `predict` machinery via
`crate::svm::FittedSVC::from_nu_ovo`, and the live oracle (`NuSVC`/`NuSVR`) is
matched array-by-array (see Verification + the REQ table). The prior C-SVC /
epsilon-SVR delegation — which reached a different optimum — is removed.

## Algorithm (sklearn / libsvm — the contract)

### Estimator surface & defaults (`_classes.py`, live oracle)

`NuSVC.__init__` defaults (`_classes.py:1120-1131`, live `inspect.signature`):
`nu=0.5`, `kernel='rbf'`, `degree=3`, `gamma='scale'`, `coef0=0.0`,
`shrinking=True`, `probability=False`, `tol=1e-3`, `cache_size=200`,
`class_weight=None`, `verbose=False`, `max_iter=-1`,
`decision_function_shape='ovr'`, `break_ties=False`, `random_state=None`.
`NuSVC._parameter_constraints` (`_classes.py:1114-1118`) takes
`BaseSVC._parameter_constraints`, **pops `C`**, and adds
`"nu": [Interval(Real, 0.0, 1.0, closed="right")]` — i.e. `nu ∈ (0, 1]`
(left-open, right-closed). `max_iter=-1` means **no iteration limit**.

`NuSVR.__init__` defaults (`_classes.py:1527-1541`): `nu=0.5`, **`C=1.0`**,
`kernel='rbf'`, `degree=3`, `gamma='scale'`, `coef0=0.0`, `shrinking=True`,
`tol=1e-3`, `cache_size=200`, `verbose=False`, `max_iter=-1`.
`NuSVR._parameter_constraints` (`_classes.py:1523-1525`) pops `class_weight`,
`epsilon`, `probability`, `random_state` — so NuSVR has **no `epsilon`**
(nu replaces it) but **retains `C` (default 1.0)**. nu-SVR is a two-parameter
model: both `nu` and `C` are user-supplied.

### nu-SVC is a distinct libsvm solver (`_base.py:30, 204`)

sklearn dispatches to libsvm by `solver_type = LIBSVM_IMPL.index(self._impl)`
(`_base.py:204`), where `LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class",
"epsilon_svr", "nu_svr"]` (`_base.py:30`). `NuSVC._impl = "nu_svc"` selects
libsvm `solver_type == 1`, NOT `c_svc` (`0`). libsvm's `Solver_NU` solves the
**nu-SVC dual**:

```
min_α  ½ αᵀ Q α            s.t.  0 ≤ α_i ≤ 1/n,  yᵀα = 0,  eᵀα = nu
```

(with `Q_{ij} = y_i y_j K(x_i,x_j)`), and recovers the decision function with a
per-class `rho` and a margin `b`. The extra equality constraint `eᵀα = nu`
(absent from C-SVC) is what makes `nu` directly bound the support-vector
fraction. The nu-SVC optimum is **not** reachable by solving a C-SVC with any
single `C`: the libsvm theory (Chang & Lin, "Training nu-SVM") shows nu-SVC with
a given `nu` corresponds to C-SVC with a `C` that is itself a function of the
*solution* (the recovered `rho`), not the closed form `1/(nu·n)`. The live
oracle confirms the two give different `dual_coef_`/`support_` (Verification).

`NuSVC` shares everything *downstream* of the solver with `SVC`: the same
`Kernel` formulas and `gamma` resolution (`_base.py:236-243`), the same
`support_`/`support_vectors_`/`n_support_`/`dual_coef_`/`intercept_`/`coef_`
fitted-attribute layout with the **binary sign flip** for `nu_svc`
(`_base.py:260`, `:538` — the predicate is `self._impl in ["c_svc", "nu_svc"]`,
covering nu-SVC), the same one-vs-one multiclass scheme, the same
`decision_function` shape/sign/ovr transform (`_base.py:538-541, 779-780`), and
the same libsvm ovo-voting `predict`. `BaseSVC._validate_targets`
(`_base.py:736-749`) computes `classes_ = np.unique(y)` and errors on `<2`
classes, exactly as SVC.

### nu-SVR is a distinct libsvm solver (`_classes.py:1521`)

`NuSVR._impl = "nu_svr"` selects libsvm `solver_type == 4`, NOT `epsilon_svr`
(`3`). libsvm's nu-SVR optimizes the `2n`-variable `(α, α*)` dual with the
**additional `nu` constraint** `eᵀ(α + α*) ≤ C·nu` (the epsilon-tube width
becomes a *learned* quantity rather than the fixed `epsilon` of epsilon-SVR).
So nu-SVR with `(nu, C)` is **not** epsilon-SVR with `epsilon=0` and any `C`.

### Kernels, fitted attributes, decision_function, predict — inherited from SVC/SVR

Because `NuSVC`/`NuSVR` extend the same `BaseLibSVM`/`BaseSVC` machinery, the
kernel formulas (linear/rbf/poly/sigmoid), `gamma` resolution
(`'scale' = 1/(n_features·X.var())` default, `'auto' = 1/n_features`, float),
the libsvm fitted-attribute layout, the binary sign flip, the ovo→ovr
`decision_function` transform, the ovo-voting `predict`, and Platt-scaling
`probability` are **the same contract documented in `.design/linear/svm.md`**
(REQ-1, REQ-3, REQ-4, REQ-5, REQ-7, REQ-9). The *only* structural difference is
the solver (nu-dual vs C-dual) and the parameter surface (`nu` replaces `C` for
NuSVC; `nu` joins `C` and replaces `epsilon` for NuSVR).

## ferrolearn (what exists)

`NuSVC<F, K>` (`nu_svm.rs`) carries fields `nu: F`, `kernel: K`, `tol: F`,
`max_iter: usize`, `cache_size: usize` with `fn new` defaulting `nu=0.5`,
`tol=1e-3`, `max_iter=10000`, `cache_size=1024`, plus builder setters
`with_nu`/`with_tol`/`with_max_iter`/`with_cache_size`. There is **no**
estimator-level `kernel` (string), `degree`, `gamma`, `coef0`, `shrinking`,
`probability`, `class_weight`, `decision_function_shape`, `break_ties`, or
`random_state` — the kernel (and its degree/coef0/gamma) is the type parameter
`K`. `max_iter` defaults to `10000` (sklearn `-1` = unbounded); `cache_size`
defaults to `1024` (sklearn `200`).

`fn fit in nu_svm.rs` (for `NuSVC`) validates `nu ∈ (0, 1]`
(`if self.nu <= F::zero() || self.nu > F::one()` → `InvalidParameter`),
errors `InsufficientSamples` for `n_samples == 0`, then computes
`c = F::one() / (self.nu * n_f)` (with `n_f = n_samples`) and **delegates to
`crate::svm::SVC::new(self.kernel.clone()).with_c(c)...`**, wrapping the result
as `FittedNuSVC(FittedSVC<F, K>)`. `FittedNuSVC` is a newtype around `FittedSVC`
with no own state.

`fn predict in nu_svm.rs` (FittedNuSVC) delegates to `self.0.predict(x)`.
`fn decision_function in nu_svm.rs` (FittedNuSVC) declares
`-> Result<Array2<F>, FerroError>` and returns `self.0.decision_function(x)` —
**this does not compile**, because `FittedSVC::decision_function` now returns
`Result<SvmScores<F>, FerroError>` (`svm.rs`). The newtype must change its
signature to `-> Result<SvmScores<F>, FerroError>` (or unwrap an
`SvmScores::Binary`/`Multiclass` variant) to propagate the SVC contract.

`NuSVR<F, K>` carries `nu`, `kernel`, `tol`, `max_iter`, `cache_size` with the
same `new`/builder defaults — and **no `C` field at all** (sklearn NuSVR has
`C`, default 1.0). `fn fit in nu_svm.rs` (for `NuSVR`, `y: Array1<F>`) validates
`nu ∈ (0, 1]`, errors on empty input, computes `c = 1/(nu·n_samples)` and
delegates to `crate::svm::SVR::new(...).with_c(c).with_epsilon(F::zero())...`,
wrapping as `FittedNuSVR(FittedSVR<F, K>)`. `fn predict`/`fn decision_function in
nu_svm.rs` (FittedNuSVR) delegate to the inner `FittedSVR` (these compile —
`FittedSVR::decision_function` still returns `Array1<F>`).

The in-module `#[cfg(test)] mod tests` (`test_nusvc_linear_separable`,
`test_nusvc_rbf`, `test_nusvc_decision_function`, `test_nusvc_invalid_nu_zero`,
`test_nusvc_invalid_nu_above_one`, `test_nusvc_nu_one`, `test_nusvr_simple`,
`test_nusvr_decision_function`, `test_nusvr_invalid_nu`,
`test_nusvc_builder_pattern`, `test_nusvr_builder_pattern`) assert loose accuracy
floors (`correct >= 6`), error paths, and builder field round-trips — they pin
**NO** sklearn nu-SVM parity value (no `dual_coef_`/`support_`/`intercept_`/
`decision_function` comparison against `NuSVC`/`NuSVR`). Several of these tests
(`test_nusvc_decision_function`, `test_nusvc_*` that call `decision_function`)
will fail to even compile once the `SvmScores` propagation lands, and must be
adapted.

The module is on the **`ndarray` substrate** (`use ndarray::{Array1, Array2,
ScalarOperand} in nu_svm.rs`), not ferray (R-SUBSTRATE). There is **no PyO3
binding** for `NuSVC`/`NuSVR` (`ferrolearn-python` exposes `LinearSVC` only).
`NuSVC`/`NuSVR`/`FittedNuSVC`/`FittedNuSVR` are re-exported at the crate root
(`pub use nu_svm::{…} in lib.rs`) — boundary estimator types whose consumers are
external users + a future Python binding (S5/R-DEFER-1 grandfathered).

## Requirements

- REQ-1: `nu` parameter + `(0, 1]` validation — `nu` field on `NuSVC`/`NuSVR`
  (default 0.5), validated left-open/right-closed `(0, 1]`
  (`Interval(Real, 0.0, 1.0, closed="right")`, `_classes.py:1116`), raising the
  sklearn-equivalent error (`ValueError`) outside the range.
- REQ-2: nu-SVC dual — the **genuine libsvm nu-SVC solver** (`_impl="nu_svc"`,
  `solver_type==1`, the nu-parameterized dual with `eᵀα = nu` and
  `0 ≤ α_i ≤ 1/n`), NOT a C-SVC with `C = 1/(nu·n)`; verified by
  `dual_coef_`/`support_`/`n_support_`/`intercept_` matching the live `NuSVC`
  oracle.
- REQ-3: nu-SVR dual — the genuine libsvm nu-SVR solver (`_impl="nu_svr"`,
  `solver_type==4`, the `2n`-variable dual with the learned-tube `nu`
  constraint and a user-supplied `C` default 1.0), NOT epsilon-SVR with
  `epsilon=0`; verified by `dual_coef_`/`support_`/`intercept_`/`predict`
  matching the live `NuSVR` oracle.
- REQ-4: Kernels & gamma resolution (inherited from svm.rs) — the four kernel
  formulas and `gamma` resolution `{'scale' (default), 'auto', float}` computed
  at fit time against `X` (`_base.py:236-243`); shared with SVC via
  `Kernel`/`Kernel::resolved_for_fit`.
- REQ-5: Fitted classification attributes — `support_` (per-class-grouped
  indices), `support_vectors_`, `n_support_`, `dual_coef_` (shape
  `(n_class-1, n_SV)`), `intercept_` (length `n_class·(n_class-1)/2`),
  `coef_ = dual_coef_ @ support_vectors_` (linear), in the libsvm layout with
  the nu-SVC binary sign flip (`_base.py:260` — `_impl in ["c_svc","nu_svc"]`).
- REQ-6: `decision_function` — propagate `crate::svm::SvmScores<F>`: binary
  `SvmScores::Binary` shape `(n,)` = `-dec.ravel()` (positive → `classes_[1]`,
  `_base.py:538-539`); multiclass-ovr (default) `SvmScores::Multiclass`
  `(n, n_classes)`; ovo `(n, n·(n-1)/2)`. `FittedNuSVC::decision_function` must
  change its return type to `Result<SvmScores<F>, FerroError>` so the module
  compiles and matches the live oracle.
- REQ-7: `predict` — libsvm ovo voting → class label, ties broken toward the
  lower class index (`_base.py:814`), matching the `NuSVC` oracle labels.
- REQ-8: Multiclass NuSVC — one-vs-one (libsvm), `classes_ = np.unique(y)`
  sorted, `<2` classes → `ValueError` (`_base.py:741-745`), per-pair nu-SVC
  models, vote aggregation matching the oracle.
- REQ-9: `probability` — Platt scaling (`predict_proba`/`predict_log_proba`),
  `probability=False` default → `predict_proba` raises (`_base.py:825-925`,
  predicate `_impl in ("c_svc","nu_svc")`).
- REQ-10: Constructor params/defaults (R-DEV-2) — NuSVC: `nu` (default 0.5),
  `kernel`, `degree`, `gamma` (`'scale'`), `coef0`, `shrinking`, `probability`,
  `tol`, `cache_size` (200), `class_weight`, `max_iter` (-1),
  `decision_function_shape` (`'ovr'`), `break_ties`, `random_state`; NuSVR:
  `nu` (0.5), **`C` (1.0)**, `kernel`, `degree`, `gamma`, `coef0`, `shrinking`,
  `tol`, `cache_size` (200), `max_iter` (-1) — with sklearn's exact names and
  defaults (NuSVR has **no `epsilon`**).
- REQ-11: ferray substrate (R-SUBSTRATE) — array type → `ferray-core`, ops →
  ferray, not `ndarray`.

## Acceptance criteria

- AC-1 (REQ-1): `NuSVC::new(...).with_nu(0.0)` and `.with_nu(1.5)` both error;
  `.with_nu(1.0)` succeeds (right-closed); `.with_nu(0.5)` is the default.
- AC-2 (REQ-2): `NuSVC(kernel='linear', nu=0.5)` on the binary 6×2 set
  (`X=[[1,1],[1.5,1],[1,1.5],[5,5],[5.5,5],[5,5.5]]`, `y=[0,0,0,1,1,1]`) yields
  `support_ [1,2,3,5]`, `n_support_ [2,2]`,
  `dual_coef_ [[-0.0217,-0.0435,0.0435,0.0217]]`, `intercept_ [-1.625]` matching
  the live `NuSVC` oracle — and these **differ** from a C-SVC with `C=1/(nu·n)`.
- AC-3 (REQ-3): `NuSVR(kernel='linear', nu=0.5, C=1.0)` on the 6×1 set
  (`X=[[1]..[6]]`, `y=[2,4,..,12]`) yields `support_ [1,5]`,
  `dual_coef_ [[-0.5,0.5]]`, `intercept_ [0.0]` matching the live `NuSVR`
  oracle — and these **differ** from `SVR(C=1/(nu·n), epsilon=0)`
  (`support_ [0,1,4,5]`, `dual_coef_ [[-0.3333,-0.1111,0.1111,0.3333]]`).
- AC-4 (REQ-4): `NuSVC` with `gamma='scale'` resolves `_gamma` matching the
  oracle's `m._gamma`; the four kernel `compute` values match libsvm.
- AC-5 (REQ-5): the binary AC-2 fit exposes `support_`, `support_vectors_`,
  `n_support_ [2,2]`, `dual_coef_` shape `(1,4)`, `intercept_` length 1, and
  linear `coef_`, all in libsvm layout with the binary sign flip.
- AC-6 (REQ-6): binary `decision_function(X)` is an `SvmScores::Binary` of shape
  `(6,)` equal to the oracle `[-1.125,-1.0,-1.0,0.875,1.0,1.0]` (positive →
  class 1); 3-class ovr is `(n,3)` matching the `NuSVC` oracle; **the module
  compiles** (`cargo build -p ferrolearn-linear` succeeds).
- AC-7 (REQ-7): `NuSVC::predict` equals the oracle labels on the binary +
  3-class sets, including the lower-index tie-break.
- AC-8 (REQ-8): 3-class `NuSVC` fit produces `classes_ [0,1,2]`, 3 ovo nu-SVC
  models, and a `<2`-class fit raises the sklearn-equivalent error.
- AC-9 (REQ-9): `NuSVC(probability=True)` exposes `predict_proba` summing to 1
  per row matching the oracle; `probability=False` → `predict_proba` raises.
- AC-10 (REQ-10): `NuSVC::new` / `NuSVR::new` expose the sklearn param surface
  with the documented defaults (NuSVC `nu=0.5,gamma='scale',cache_size=200,
  max_iter=-1,decision_function_shape='ovr'`; NuSVR `nu=0.5,C=1.0,gamma='scale',
  cache_size=200,max_iter=-1`, no `epsilon`).
- AC-11 (REQ-11): `nu_svm.rs` owns its computation on `ferray-core` arrays.

## REQ status

Classification (R-DEFER-2, two states only): SHIPPED = impl + non-test
production consumer + tests + green oracle verification (symbol-anchor cite +
sklearn `file:line`); NOT-STARTED = concrete open blocker referenced by
`#`-number. `NuSVC`/`NuSVR`/`FittedNuSVC`/`FittedNuSVR` are boundary estimator
types re-exported at the crate root (`pub use nu_svm::{…} in lib.rs`); their
consumer surface (external users + a future binding) is grandfathered under
S5/R-DEFER-1.

**The genuine libsvm `Solver_NU` now lands (#654/#655/#656/#657).** `crate::svm`
gains `fn solver_nu_core` (the nu-specific second-order WSS with the four running
maxima `Gmaxp/Gmaxp2/Gmaxn/Gmaxn2` + the shared `Solver::Solve` analytic update +
the two-group `rho`/`r` recovery, `svm.cpp:1186-1418, 663-940`) and the
`pub(crate) fn solve_nu_svc`/`fn solve_nu_svr` wrappers. `NuSVC::fit`/`NuSVR::fit`
now call these (NO MORE `SVC::new().with_c(1/(ν·n))`), and `NuSVR` gains the
`pub c: F` field (default `1.0`). `FittedNuSVC` is rebuilt around a `FittedSVC`
assembled via `crate::svm::FittedSVC::from_nu_ovo` so the libsvm-layout fitted
attributes / `decision_function` / `predict` are all re-exposed, with the binary
nu_svc sign flip (`_base.py:258-262`). Verified GREEN against the live oracle:

- `NuSVC(kernel='linear',nu=0.5)` on `X=[[1,1],[2,1],[1,2],[5,5],[6,5],[5,6]]`,
  `y=[0,0,0,1,1,1]`: `support_ [1,2,3,5]`, `n_support_ [2,2]`,
  `dual_coef_ [[-0.0227,-0.0455,0.0455,0.0227]]`, `intercept_ [-1.75]`,
  `df [-1.25,-1.0,-1.0,0.75,1.0,1.0]`, `predict [0,0,0,1,1,1]`, `coef_ [[0.25,0.25]]`.
- `NuSVR(kernel='linear',nu=0.5,C=1.0)` on `X=[[1],[2],[3],[4]]`, `y=[1,5,2,8]`:
  `predict [2.5,3.5,4.5,5.5]`, `support_ [2,3]`, `dual_coef_ [[-1,1]]`,
  `intercept_ [1.5]`.

REMAINING NOT-STARTED: REQ-9 (probability / predict_proba on NuSVC, #653) and
REQ-11 (ferray substrate, #655 — re-scoped to the ferray migration, tracked
alongside svm.rs REQ-10 #643).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (nu validation `(0,1]`) | SHIPPED | `fn fit in nu_svm.rs` (NuSVC and NuSVR) rejects `self.nu <= F::zero() \|\| self.nu > F::one()` with `FerroError::InvalidParameter` — `Interval(Real, 0.0, 1.0, closed="right")` (`_classes.py:1116`). Tests `test_nusvc_invalid_nu_zero`/`test_nusvc_invalid_nu_above_one`/`test_nusvc_nu_one`/`test_nusvr_invalid_nu in nu_svm.rs` green. Non-test consumer: the validation guards the production `NuSVC::fit`/`NuSVR::fit`. |
| REQ-2 (nu-SVC dual) | SHIPPED | `NuSVC::fit in nu_svm.rs` calls `crate::svm::solve_nu_svc` (the genuine libsvm `Solver_NU` nu-SVC dual with `eᵀα=nu·l`, `0≤α_i≤1`, init+rescale `svm.cpp:1646-1708`) per ovo pair — NOT a C-SVC. `fn solver_nu_core in svm.rs` implements the nu WSS + the `r`-rescale (`alpha·y/r`, bias `-rho/r`). Pinned: `divergence_nusvc_decision_function_delegation in tests/divergence_nu_svm.rs` (df `[-1.25,-1.0,-1.0,0.75,1.0,1.0]`, GREEN) + `test_nusvc_oracle_attrs in nu_svm.rs` (`support_ [1,2,3,5]`, `dual_coef_ [[-0.0227,-0.0455,0.0455,0.0227]]`, `intercept_ [-1.75]` vs live `NuSVC(kernel='linear',nu=0.5)`, R-CHAR-3, 1e-2). |
| REQ-3 (nu-SVR dual + C) | SHIPPED | `NuSVR::fit in nu_svm.rs` calls `crate::svm::solve_nu_svr(nu, C)` (the `2l`-variable learned-tube nu-SVR dual, `svm.cpp:1795-1839`); `NuSVR` gains `pub c: F` (default 1.0, `_classes.py:1531`) + `fn with_c in nu_svm.rs`, and has NO `epsilon`. Pinned: `divergence_nusvr_predict_delegation`/`divergence_nusvr_missing_c_parameter in tests/divergence_nu_svm.rs` (predict `[2.5,3.5,4.5,5.5]`, GREEN) + `test_nusvr_oracle_attrs in nu_svm.rs` (`support_ [2,3]`, `dual_coef_ [[-1,1]]`, `intercept_ [1.5]`). |
| REQ-4 (kernels & gamma resolution) | SHIPPED | `NuSVC::fit`/`NuSVR::fit in nu_svm.rs` call `self.kernel.resolved_for_fit(x)` (the shared `crate::svm` path: `Gamma::Scale`=`1/(n_features·X.var())` default, `'auto'`, float, `_base.py:236-243`) BEFORE the nu solve, so RBF/poly/sigmoid NuSVC/NuSVR resolve gamma exactly as SVC. Smoke: `test_nusvc_rbf in nu_svm.rs`. |
| REQ-5 (fitted classification attributes) | SHIPPED | `FittedNuSVC::{support,support_vectors,n_support,dual_coef,intercept,coef} in nu_svm.rs` delegate to the inner `FittedSVC` assembled by `crate::svm::FittedSVC::from_nu_ovo` (one nu-SVC sub-model per ovo pair), so the libsvm layout (`dual_coef_ (n_class-1,n_SV)`, per-class-grouped `support_`, linear-only `coef_`) + the binary nu_svc sign flip (`_base.py:258-262`) apply. Pinned by `test_nusvc_oracle_attrs in nu_svm.rs`. |
| REQ-6 (decision_function propagates SvmScores) | SHIPPED | `FittedNuSVC::decision_function in nu_svm.rs` returns `Result<SvmScores<F>, FerroError>` from the inner `FittedSVC::decision_function`: binary `SvmScores::Binary` `(n,)`, multiclass-ovr `(n,n_classes)` (`_base.py:538-541,779-780`). The crate compiles. Pinned by `divergence_nusvc_decision_function_delegation` + `test_nusvc_multiclass in nu_svm.rs` (ovr `(9,3)`). |
| REQ-7 (predict — ovo voting + tie-break) | SHIPPED | `FittedNuSVC::predict in nu_svm.rs` delegates to `FittedSVC::predict` (libsvm ovo voting, lower-index tie-break, `_base.py:813-814`) over the nu-SVC sub-models. Pinned by `nusvc_predict_labels_match_on_separable in tests/divergence_nu_svm.rs` + `test_nusvc_oracle_attrs` (`predict [0,0,0,1,1,1]`). |
| REQ-8 (multiclass NuSVC one-vs-one) | SHIPPED | `NuSVC::fit in nu_svm.rs` trains one `solve_nu_svc` per class pair over `classes = np.unique(y)` (sort+dedup), `<2` classes → `InsufficientSamples` (`_base.py:741-745`), assembled via `from_nu_ovo`. Smoke `test_nusvc_multiclass in nu_svm.rs` (3-class, ovr `(9,3)`, ≥7/9 correct). |
| REQ-9 (probability / predict_proba) | NOT-STARTED | open #653. No `probability` field / Platt-scaling CV / `predict_proba`/`predict_log_proba` on `NuSVC` (`_base.py:825-925`, `_impl in ("c_svc","nu_svc")`). Shares root cause with the svm.rs probability machinery (already shipped for SVC); wiring it into the nu-SVC ovo path is the remaining work. |
| REQ-10 (constructor params/defaults) | SHIPPED | NuSVC `nu` (0.5), `tol` (1e-3), `cache_size` (200), `max_iter` (0 = sklearn `-1`), `decision_function_shape` (Ovr), `break_ties` (false), builders `with_nu`/`with_tol`/`with_max_iter`/`with_cache_size`/`with_decision_function_shape`/`with_break_ties`; NuSVR `nu` (0.5), **`c` (1.0, `_classes.py:1531`)**, `tol`, `cache_size` (200), `max_iter` (0), `with_c`, NO `epsilon`. R-DEV-7 design difference (preserved contract): `kernel`/`degree`/`gamma`/`coef0` are the type parameter `K` (as for SVC); `probability` is NOT-STARTED REQ-9; `class_weight`/`random_state` unused (SVR/deterministic). Pinned by `test_nusvc_builder_pattern`/`test_nusvr_builder_pattern`/`test_nusvc_default_c_is_one in nu_svm.rs`. |
| REQ-11 (ferray substrate) | NOT-STARTED | open #655 (re-scoped to ferray migration). `nu_svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}` and computes through the `ndarray`-based `crate::svm` solver, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-1/2). Consistent with the crate-wide deferral (svm.md REQ-10 #643). |

## Verification

The module unit tests pin NO sklearn nu-SVM parity value, AND the crate does not
compile, so no REQ can be SHIPPED (R-DEFER-6, R-CHAR-1/R-CHAR-3):

- `cargo build -p ferrolearn-linear` — **FAILS** at `nu_svm.rs:188` (type
  mismatch `Result<Array2<F>,_>` vs `Result<SvmScores<F>,_>`). This blocks every
  downstream `cargo test`/clippy/oracle check.
- `cargo test -p ferrolearn-linear nu_svm` (once it compiles) — the in-module
  `#[cfg(test)] mod tests` assert loose accuracy floors (`correct >= 6`), error
  paths, and builder field round-trips; they do NOT compare
  `dual_coef_`/`intercept_`/`support_`/`decision_function` against `NuSVC`/
  `NuSVR`. `test_nusvc_decision_function` will need adapting to `SvmScores`.
- `cargo clippy -p ferrolearn-linear --all-targets -- -D warnings`,
  `cargo fmt --all --check`.

Live sklearn oracle (establishes the divergences; expected values per R-CHAR-3
come from sklearn, NEVER copied from ferrolearn):

```bash
# REQ-2: genuine NuSVC vs the C-SVC delegation (they DIFFER)
python3 -c "import numpy as np; from sklearn.svm import NuSVC, SVC; \
X=np.array([[1.,1.],[1.5,1.],[1.,1.5],[5.,5.],[5.5,5.],[5.,5.5]]); y=np.array([0,0,0,1,1,1]); \
m=NuSVC(kernel='linear',nu=0.5).fit(X,y); \
print('NuSVC support_',m.support_.tolist(),'n_support_',m.n_support_.tolist()); \
print('NuSVC dual_coef_',np.round(m.dual_coef_,4).tolist(),'intercept_',np.round(m.intercept_,4).tolist()); \
print('NuSVC df',np.round(m.decision_function(X),4).tolist()); \
s=SVC(kernel='linear',C=1.0/(0.5*6)).fit(X,y); \
print('SVC(C=1/(nu*n)) support_',s.support_.tolist(),'dual_coef_',np.round(s.dual_coef_,4).tolist())"
# NuSVC support_ [1,2,3,5]  n_support_ [2,2]
# NuSVC dual_coef_ [[-0.0217,-0.0435,0.0435,0.0217]]  intercept_ [-1.625]  df [-1.125,-1.0,-1.0,0.875,1.0,1.0]
# SVC(C=0.333) support_ [1,2,3]  dual_coef_ [[-0.0354,-0.0357,0.0711]]   <-- DIFFERENT optimum

# REQ-3: genuine NuSVR vs the epsilon-SVR delegation (they DIFFER)
python3 -c "import numpy as np; from sklearn.svm import NuSVR, SVR; \
X=np.array([[1.],[2.],[3.],[4.],[5.],[6.]]); y=np.array([2.,4.,6.,8.,10.,12.]); \
m=NuSVR(kernel='linear',nu=0.5,C=1.0).fit(X,y); \
print('NuSVR support_',m.support_.tolist(),'dual_coef_',np.round(m.dual_coef_,4).tolist(),'intercept_',np.round(m.intercept_,4).tolist()); \
s=SVR(kernel='linear',C=1.0/(0.5*6),epsilon=0.0).fit(X,y); \
print('SVR(C=1/(nu*n),eps=0) support_',s.support_.tolist(),'dual_coef_',np.round(s.dual_coef_,4).tolist())"
# NuSVR support_ [1,5]  dual_coef_ [[-0.5,0.5]]  intercept_ [0.0]
# SVR(C=0.333,eps=0) support_ [0,1,4,5]  dual_coef_ [[-0.3333,-0.1111,0.1111,0.3333]]   <-- DIFFERENT optimum

# REQ-1/REQ-10: nu constraint + defaults
python3 -c "from sklearn.svm import NuSVC, NuSVR; import inspect; \
print('NuSVC', {k:v.default for k,v in inspect.signature(NuSVC.__init__).parameters.items() if k!='self'}); \
print('NuSVR', {k:v.default for k,v in inspect.signature(NuSVR.__init__).parameters.items() if k!='self'})"
# NuSVC nu=0.5 kernel='rbf' gamma='scale' degree=3 coef0=0.0 tol=1e-3 shrinking=True
#       decision_function_shape='ovr' probability=False max_iter=-1 cache_size=200 break_ties=False ...
# NuSVR nu=0.5 C=1.0 kernel='rbf' gamma='scale' degree=3 coef0=0.0 tol=1e-3 cache_size=200 max_iter=-1  (NO epsilon)
```

A NOT-STARTED REQ closes only when its fix lands AND a divergence test (expected
values from the live oracle / a sklearn `file:line` constant per R-CHAR-3) goes
green; all eleven REQs are currently NOT-STARTED (the crate does not even
compile).

## Blockers to open

- **#645** — REQ-1 of nu_svm: pin the `nu ∈ (0,1]` validation against the
  sklearn `Interval(Real,0.0,1.0,closed="right")` constraint
  (`_classes.py:1116`) — left-open, right-closed — with the sklearn-equivalent
  error class; gated on the build fix (#650).
- **#646** — REQ-2 of nu_svm: implement the genuine libsvm **nu-SVC** solver
  (`Solver_NU`, the nu-parameterized dual with `eᵀα=nu`, `0≤α_i≤1/n`,
  `_impl="nu_svc"`/`solver_type==1`, `_base.py:30,204`) in `crate::svm` (or a
  nu-SVC path in `nu_svm.rs`) — NOT a C-SVC with `C=1/(nu·n)`; pin
  `dual_coef_`/`support_`/`n_support_`/`intercept_`/`decision_function` against
  the live `NuSVC(kernel='linear',nu=0.5)` oracle (`support_ [1,2,3,5]`,
  `dual_coef_ [[-0.0217,-0.0435,0.0435,0.0217]]`, `intercept_ [-1.625]`).
  **TOP divergence.**
- **#647** — REQ-3 of nu_svm: implement the genuine libsvm **nu-SVR** solver
  (`_impl="nu_svr"`/`solver_type==4`, the `2n`-variable learned-tube `nu` dual)
  and add the `C` field (default 1.0, `_classes.py:1531`); NOT epsilon-SVR with
  `epsilon=0`; pin against the live `NuSVR(nu=0.5,C=1.0)` oracle
  (`support_ [1,5]`, `dual_coef_ [[-0.5,0.5]]`, `intercept_ [0.0]`).
- **#648** — REQ-4 of nu_svm: resolve `gamma` at fit time against `X`
  (`'scale'=1/(n_features·X.var())` default, `'auto'=1/n_features`, float,
  `_base.py:236-243`) through the shared `crate::svm` kernel path (shares root
  cause with svm.md #634); pin kernel values + `_gamma` against the `NuSVC`
  oracle.
- **#649** — REQ-5 of nu_svm: expose `support_`/`support_vectors_`/`n_support_`/
  `dual_coef_ (n_class-1,n_SV)`/`intercept_`/linear `coef_` on `FittedNuSVC`
  with the nu-SVC binary sign flip (`_impl in ["c_svc","nu_svc"]`,
  `_base.py:260`), pinned against the `NuSVC` oracle layout; gated on #646.
- **#650** — REQ-6 of nu_svm: **fix the build** — change
  `FittedNuSVC::decision_function in nu_svm.rs` to return
  `Result<SvmScores<F>, FerroError>`, propagating `crate::svm`'s binary
  `(n,)`=`-dec.ravel()` / ovr `(n,n_classes)` contract (`_base.py:538-541,
  779-780`); adapt `test_nusvc_decision_function` to the enum; pin shapes +
  values against the `NuSVC` oracle. Prereq for every other REQ (crate does not
  compile). Escalated from svm.md REQ-4 on #644.
- **#651** — REQ-7 of nu_svm: align `predict` tie-breaking with libsvm (lower
  class index, `_base.py:814`, shares root cause with svm.md #638) and pin
  `NuSVC::predict` labels against the live oracle; gated on #646/#650.
- **#652** — REQ-8 of nu_svm: pin the 3-class one-vs-one **nu-SVC** fit
  (per-pair `dual_coef_`/`intercept_`, `classes_`, `<2`-class error,
  `_base.py:736-749`) against the live `NuSVC` oracle; gated on #646.
- **#653** — REQ-9 of nu_svm: add `probability` (Platt-scaling internal-CV
  sigmoid) with `predict_proba`/`predict_log_proba` and the
  `AttributeError`-when-`probability=False` contract (`_base.py:825-925`); shares
  root cause with svm.md #642.
- **#654** — REQ-10 of nu_svm: add the sklearn estimator-level parameter surface
  (R-DEV-2): NuSVC `kernel`(string)/`degree`/`gamma`/`coef0`/`shrinking`/
  `probability`/`class_weight`/`decision_function_shape`/`break_ties`/
  `random_state`; NuSVR `C` (default 1.0) and **remove `epsilon`**; align
  defaults (`max_iter=-1`, `cache_size=200`, `gamma='scale'`).
- **#655** — REQ-11 of nu_svm: migrate `nu_svm.rs` (and its `crate::svm`
  delegation) off `ndarray` onto the ferray substrate (R-SUBSTRATE).
