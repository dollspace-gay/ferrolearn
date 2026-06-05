# Nystroem Kernel Approximation

<!--
tier: 3-component
status: draft
baseline-commit: 3aef2e880
upstream-paths:
  - sklearn/kernel_approximation.py        # class Nystroem (:827-1088)
  - sklearn/metrics/pairwise.py            # pairwise_kernels / KERNEL_PARAMS (poly/rbf defaults)
-->

## Summary

`ferrolearn-kernel/src/nystroem.rs` mirrors scikit-learn's
`sklearn.kernel_approximation.Nystroem` — a low-rank kernel approximation that
samples a subset of the training data as a basis, eigendecomposes the basis
kernel matrix, and builds a dense feature embedding `Z` such that
`Z·Zᵀ ≈ K`. sklearn's `fit` permutes the rows, computes
`basis_kernel = pairwise_kernels(basis, metric=self.kernel, ...)`, takes its SVD,
floors the singular values to `1e-12`, and stores the **symmetric** inverse
square root `normalization_ = U/√S · V` (`kernel_approximation.py:1030-1033`);
`transform` returns `embedded · normalization_ᵀ` (`:1066`). ferrolearn shuffles
with `Xoshiro256PlusPlus`, computes the same four kernel formulas
(`compute_kernel_matrix`/`kernel_value`), eigendecomposes `K_qq = V D Vᵀ` via
`faer` (`symmetric_eigen`), and stores `normalization[i,j] = V[i,j]/√D_j`
(i.e. `V·D^{-1/2}`, **not** the symmetric `V·D^{-1/2}·Vᵀ`); `transform`
returns `k_new · normalization`.

The **rotation-invariant** correctness slice — the kernel reconstruction
`Z·Zᵀ ≈ K` at `n_components == n_samples`, and the default
`gamma = 1/n_features` — is SHIPPED and oracle-pinnable against
`sklearn.metrics.pairwise.rbf_kernel`. Everything that depends on a specific
basis permutation OR on the element-wise `transform` output diverges from
sklearn and is NOT-STARTED: the `V·D^{-1/2}` vs symmetric `V·D^{-1/2}·Vᵀ`
normalization form (an orthogonal `Vᵀ`-rotation of the embedding), the
eigenvalue floor (`zero` vs `1e-12`), the RNG basis-permutation bit-stream
(`Xoshiro256++` vs numpy Mersenne-Twister, plus a no-shuffle path at full
basis), the polynomial `coef0` default (`0` vs sklearn `1`), the silent clamp
with no `n_components > n_samples` warning, the missing
`component_indices_`/`normalization_`/string-or-callable-kernel/`kernel_params`/
`n_jobs` surface, and the ferray array/linalg/random substrate.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/kernel_approximation.py:827-1088` — `class Nystroem`.
  - constructor `:957-976` — `__init__(self, kernel="rbf", *, gamma=None,
    coef0=None, degree=None, kernel_params=None, n_components=100,
    random_state=None, n_jobs=None)`.
  - `_parameter_constraints` `:943-955` — `kernel: StrOptions(...) | callable`
    (string OR callable, incl. `"precomputed"`); `gamma: Interval(Real, 0, None,
    closed="left") | None`; `coef0: Interval(Real, None, None) | None`;
    `degree: Interval(Real, 1, None, closed="left") | None`;
    `n_components: Interval(Integral, 1, None, closed="left")`.
  - `fit` `:979-1036` — `n_components > n_samples` → clamp to `n_samples` **and**
    `warnings.warn("n_components > n_samples. ...")` (`:1005-1012`);
    `inds = rnd.permutation(n_samples); basis_inds = inds[:n_components];
    basis = X[basis_inds]` (`:1017-1019`, ALWAYS permutes);
    `basis_kernel = pairwise_kernels(basis, metric=self.kernel, ...)` (`:1021`);
    `U, S, V = svd(basis_kernel); S = np.maximum(S, 1e-12);
    self.normalization_ = np.dot(U / np.sqrt(S), V)` (`:1030-1032`);
    `self.components_ = basis; self.component_indices_ = basis_inds` (`:1033-1034`).
  - `transform` `:1038-1066` — `embedded = pairwise_kernels(X, self.components_,
    ...); return np.dot(embedded, self.normalization_.T)` (`:1058-1066`).
  - `_get_kernel_params` `:1068-1088` — pulls `gamma/coef0/degree` from
    `KERNEL_PARAMS[self.kernel]`, only forwarding those that are not `None`.
- `sklearn/metrics/pairwise.py` — `pairwise_kernels` / `KERNEL_PARAMS`. Live
  oracle: `KERNEL_PARAMS["polynomial"] == {degree, coef0, gamma}` with
  `polynomial_kernel` defaults `degree=3, gamma=None (→1/n_features), coef0=1`;
  `KERNEL_PARAMS["rbf"] == {gamma}` with `gamma=None → 1/n_features`.

## Requirements

- REQ-1: Kernel-reconstruction correctness (full basis). At
  `n_components == n_samples` the embedding satisfies `Z·Zᵀ ≈ K(X,X)`, the
  Nystroem identity (`kernel_approximation.py:1030-1066`). This is invariant to
  basis permutation AND to the `Vᵀ`-rotation, so ferrolearn MATCHES sklearn
  here. (Deterministic / oracle-pinnable via `rbf_kernel`.)
- REQ-2: Default `gamma = 1/n_features`. With `gamma=None`, `fit` resolves the
  effective gamma to `1/n_features`, matching sklearn's pairwise rbf default
  (`KERNEL_PARAMS["rbf"]`, `gamma=None`). (Deterministic / oracle-pinnable.)
- REQ-3: Four kernel value formulas. `kernel_value` computes rbf
  `exp(-γ‖x−y‖²)`, polynomial `(γ⟨x,y⟩+c)^d`, linear `⟨x,y⟩`, sigmoid
  `tanh(γ⟨x,y⟩+c)` — matching sklearn's pairwise definitions when
  `gamma/degree/coef0` are supplied explicitly. (Deterministic / oracle-pinnable.)
- REQ-4: `n_components` clamp + warning at full/over-full basis. `fit` clamps
  `n_components` to `n_samples` when it exceeds it; sklearn ALSO emits a
  `warnings.warn` (`:1005-1012`). ferrolearn clamps but does NOT warn.
  (Deterministic / oracle-pinnable on the warning.)
- REQ-5: `n_components ≥ 1` validation. `fit` rejects `n_components == 0`,
  matching `_parameter_constraints` `Interval(Integral, 1, None, closed="left")`
  (`:952`). (Deterministic / oracle-pinnable.)
- REQ-6: Symmetric-normalization output contract (`transform` value parity).
  sklearn stores `normalization_ = U/√S · V` (the SYMMETRIC `K_qq^{-1/2}`,
  `:1032`) and transforms `embedded · normalization_ᵀ` (`:1066`); ferrolearn
  stores `V·D^{-1/2}` (no trailing `Vᵀ`) and transforms `k_new · normalization`.
  The two embeddings differ by an orthogonal `Vᵀ` rotation, so element-wise
  `transform` output diverges from sklearn even with an identical basis. (R-DEV-3
  output-contract.)
- REQ-7: Small-eigenvalue handling parity. sklearn FLOORS singular values to
  `1e-12` (`S = np.maximum(S, 1e-12)`, `:1031`) and keeps every column;
  ferrolearn ZEROES any column whose eigenvalue `≤ 1e-12` (the `if ev > eps`
  branch leaves the column zero). On rank-deficient `K_qq` these differ.
  (R-DEV-1 numerical contract.)
- REQ-8: Basis-permutation / exact `transform` value parity (RNG). sklearn
  permutes via numpy `check_random_state` (Mersenne-Twister) and ALWAYS permutes
  even at `n_components == n_samples`; ferrolearn shuffles with
  `Xoshiro256PlusPlus` and at full basis takes `x.clone()` (original order, NO
  permutation). Exact `transform`-value parity is unachievable across the
  generator boundary + the no-shuffle path. (RNG carve-out — **no failing test**.)
- REQ-9: Polynomial `coef0` default = 1. ferrolearn's `Nystroem::new` sets
  `coef0 = F::zero()`; sklearn's polynomial kernel defaults `coef0=1` (via
  `pairwise_kernels`/`polynomial_kernel`). (Deterministic / oracle-pinnable.)
  (`degree` default 3 and `gamma` default `1/n_features` MATCH — see REQ-2/-3.)
- REQ-10: Fitted-attribute + constructor surface. Expose `components_`,
  `component_indices_`, `normalization_` (sklearn names, `:885-893`), accept
  `kernel` as string-OR-callable (incl. `"precomputed"`), `kernel_params`, and
  `n_jobs` (`:957-976`). ferrolearn exposes `basis()`/`n_components()` only, a
  fixed `KernelType` enum, and no `kernel_params`/`n_jobs`.
- REQ-11: ferray substrate (R-SUBSTRATE-1). Array type → `ferray-core`;
  eigendecomposition → `ferray::linalg`; basis sampling → `ferray::random`;
  PyO3 bridge → `ferray::numpy_interop` — instead of `ndarray` + `faer` +
  `rand`/`rand_xoshiro`/`rand_distr`.
- REQ-12: Non-test production consumer. Re-exported and registered in the
  Python binding so `import ferrolearn` mirrors `import sklearn`.

## Acceptance criteria

- AC-1 (REQ-1): with `n_components == n_samples` (all points are the basis),
  ferrolearn's `Z·Zᵀ` equals `sklearn.metrics.pairwise.rbf_kernel(X, X,
  gamma=γ)` element-wise to ~1e-13. Expected from the live oracle
  (`rbf_kernel`), NOT from ferrolearn's transform (R-CHAR-3). The in-crate
  `kernel_approximation_quality_rbf` test pins the self-kernel diagonal ≈ 1.
- AC-2 (REQ-2): `Nystroem::new().with_n_components(k).fit(X)` with `X` of
  `n_features=5` yields effective `gamma == 0.2 == 1/5`, matching sklearn's
  pairwise rbf default. The `default_gamma_scales_with_features` test pins this.
- AC-3 (REQ-3): `kernel_value` for each `KernelType` equals the corresponding
  `sklearn.metrics.pairwise.{rbf,polynomial,linear,sigmoid}_kernel` entry with
  matching `gamma/degree/coef0` (deterministic; oracle-pinnable).
- AC-4 (REQ-4): `Nystroem(n_components=100).fit(X)` on `n_samples=3` emits a
  `UserWarning` "n_components > n_samples" in sklearn; ferrolearn clamps to 3
  (`n_components_exceeds_n_samples` test) but raises NO warning. Oracle-pinnable
  on the warning emission.
- AC-5 (REQ-5): `Nystroem::new().with_n_components(0).fit(X)` errors, as sklearn
  raises `InvalidParameterError` for `n_components=0`. The `rejects_zero_components`
  test pins this.
- AC-6 (REQ-6): for an IDENTICAL basis, ferrolearn's `transform(X)` and
  sklearn's `transform(X)` differ by an orthogonal matrix `Q` (`Q·Qᵀ = I`):
  `‖Z_ferro·Z_ferroᵀ − Z_sk·Z_skᵀ‖ ≈ 0` but `‖Z_ferro − Z_sk‖ ≉ 0`. A critic
  pins this by fixing the basis (full-basis case) and asserting the element-wise
  mismatch — failing until ferrolearn adopts the symmetric `V·D^{-1/2}·Vᵀ` form.
- AC-7 (REQ-7): on a rank-deficient `K_qq` (e.g. duplicated rows), sklearn's
  floored column contributes `1/√(1e-12)=1e6` while ferrolearn zeroes it;
  `Z_sk` and `Z_ferro` differ in the floored direction. Oracle-pinnable once a
  rank-deficient fixture is constructed.
- AC-8 (REQ-8): N/A — RNG carve-out; no test (R-DEFER-3). Exact `transform`
  values cannot match across `Xoshiro256++` ↔ Mersenne-Twister, compounded by
  the no-shuffle-at-full-basis path.
- AC-9 (REQ-9): `Nystroem::new().with_kernel(Polynomial)` (no `with_coef0`)
  uses `coef0=0`, whereas `Nystroem(kernel='poly')` defaults `coef0=1`. Live
  oracle: `polynomial_kernel(X,X)` (default `coef0=1`) is
  `[[42.875, 274.625, ...], ...]` vs `coef0=0`
  `[[15.625, 166.375, ...], ...]` for `X=[[1,2],[3,4],[5,6]]` — different basis
  kernel, hence different embedding. Oracle-pinnable.
- AC-10 (REQ-10): a fitted model exposes `components_`, `component_indices_`,
  `normalization_` (sklearn names) and the constructor accepts a string/callable
  kernel + `kernel_params` + `n_jobs`. Currently absent.
- AC-11 (REQ-11): no `ndarray`/`faer`/`rand`/`rand_xoshiro`/`rand_distr` in the
  owned computation; arrays/eigen/sampling route through ferray.
- AC-12 (REQ-12): `ferrolearn`'s Python `Nystroem` fit/transforms via the Rust
  core through `RsNystroem`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (reconstruction, full basis) | SHIPPED | At `n_components == n_samples`, `Fit::fit` takes `basis = x.clone()` and builds `normalization` from `symmetric_eigen(&k_basis)`; `Transform::transform` returns `k_new.dot(&self.normalization)`. The Nystroem identity `Z·Zᵀ = K_nq·K_qq^{-1}·K_qn` is rotation- AND permutation-invariant, so ferrolearn MATCHES sklearn. Live oracle (R-CHAR-3): `Nystroem(kernel='rbf', gamma=0.5, n_components=8, random_state=0).fit(X); max abs(Z@Z.T − rbf_kernel(X,X,gamma=0.5)) = 1.55e-15` on `X` of shape `(8,3)`. In-crate `kernel_approximation_quality_rbf` pins the self-kernel diagonal ≈ 1. Non-test consumer: `ferrolearn-python/src/extras.rs` (`py_transformer!(RsNystroem, ..., FittedNystroem<f64>, (), { Nystroem::<f64>::new() })`), re-exported via `lib.rs` (`pub use nystroem::{FittedNystroem, KernelType, Nystroem}`). Verification: `cargo test -p ferrolearn-kernel --lib nystroem`. Deterministic / oracle-pinnable. |
| REQ-2 (default gamma) | SHIPPED | `Fit::fit` resolves `effective_gamma = self.gamma.unwrap_or_else(\|\| F::one()/F::from(n_features))` — matches sklearn's pairwise rbf default (`KERNEL_PARAMS["rbf"]`, `gamma=None → 1/n_features`; live `1/5 = 0.2`). In-crate `default_gamma_scales_with_features` pins `fitted.gamma == 0.2` for `n_features=5`. Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-3 (kernel formulas) | SHIPPED | `kernel_value` computes rbf `(-gamma*sq_dist).exp()`, polynomial `(gamma*dot+coef0)^degree` (explicit `for _ in 0..degree` product), linear `<x,y>`, sigmoid `(gamma*dot+coef0).tanh()` — identical to sklearn's `rbf/polynomial/linear/sigmoid_kernel` when `gamma/degree/coef0` are supplied explicitly. (The poly **default** mismatch is REQ-9, not this REQ.) `compute_kernel_matrix` fills the pairwise matrix used by both `fit` and `transform`. Consumer: same as REQ-1. In-crate `polynomial_kernel`/`linear_kernel`/`sigmoid_kernel` tests exercise each branch. Deterministic / oracle-pinnable. |
| REQ-4 (clamp + warn) | NOT-STARTED | blocker issue to be filed by critic. `Fit::fit` does `let n_basis = self.n_components.min(n_samples)` and at `n_basis >= n_samples` uses `x.clone()` — it clamps but emits NO warning. sklearn (`:1005-1012`) clamps AND `warnings.warn("n_components > n_samples. ...")`. Live oracle: `Nystroem(n_components=100).fit(X)` on `n_samples=3` raises a `UserWarning`; ferrolearn is silent. The clamp itself is correct (`n_components_exceeds_n_samples` test confirms `n_components()==10` for `n_components=100, n_samples=10`); only the warning is missing. Deterministic / oracle-pinnable on warning emission. |
| REQ-5 (n_components ≥ 1) | SHIPPED | `Fit::fit` does `if self.n_components == 0 { return Err(InvalidParameter{ name:"n_components", reason:"must be at least 1" }) }` — matches `_parameter_constraints` `n_components: Interval(Integral, 1, None, closed="left")` (`:952`). Live oracle: `Nystroem(n_components=0).fit(...)` → `InvalidParameterError`. In-crate `rejects_zero_components` (green). Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-6 (normalization form / transform parity) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-3 output contract). `Fit::fit` builds `normalization[[i,j]] = eigenvectors[[i,j]] * inv_sqrt` from `K_qq = V D Vᵀ` — i.e. `V·D^{-1/2}`, NO trailing `Vᵀ`; `transform` does `k_new.dot(&normalization)`. sklearn (`:1032`) stores the SYMMETRIC `normalization_ = U/√S · V` (`= K_qq^{-1/2}`) and transforms `embedded · normalization_ᵀ` (`:1066`). The embeddings differ by an orthogonal `Vᵀ` rotation: `Z·Zᵀ` is identical (REQ-1), but element-wise `transform` output diverges from sklearn even with an identical basis. Fix: post-multiply `normalization` by `Vᵀ` to restore the symmetric inverse square root. Deterministic / oracle-pinnable (mismatch is the divergence). |
| REQ-7 (eigenvalue floor) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-1 numerical contract). `Fit::fit` does `if ev > eps { ... } // else column stays zero` with `eps = 1e-12` — ZEROES near-zero-eigenvalue columns. sklearn FLOORS to `1e-12` (`S = np.maximum(S, 1e-12)`, `:1031`) and KEEPS every column (the floored direction scales by `1/√(1e-12)=1e6`). On a rank-deficient `K_qq` (duplicate basis rows) the two embeddings differ in the floored direction. Deterministic / oracle-pinnable on a rank-deficient fixture. |
| REQ-8 (basis permutation / exact value parity) | NOT-STARTED | blocker issue to be filed by critic (R-DEFER-3 carve-out — **NO failing test**). `Fit::fit` uses `rand_xoshiro::Xoshiro256PlusPlus` (`seed_from_u64`/`from_os_rng`) + `SliceRandom::shuffle`, and at `n_basis >= n_samples` takes `x.clone()` (ORIGINAL order, NO permutation). sklearn permutes via numpy `check_random_state` (Mersenne-Twister, `:1001,1017`) and ALWAYS permutes even at `n_components == n_samples`. Exact `transform`-value parity vs the live oracle is structurally unachievable (different generator bit-stream AND the no-shuffle path) until ferray exposes a numpy-compatible RNG (`ferray::random`) — R-SUBSTRATE-1/5. This is the dominant value-parity blocker; REQ-1 (reconstruction) is the permutation-invariant slice that ships regardless. Carve-out: blocker, no failing test. |
| REQ-9 (poly coef0 default) | SHIPPED | FIXED #1903: `Nystroem::new` default `coef0: F::zero()` → `F::one()`, matching sklearn's `coef0=None` → `pairwise_kernels`/`polynomial_kernel` default `coef0=1` (`:962`, `:1068-1086`); unused by rbf/linear (see `kernel_value`). Live oracle (`X=[[1,2],[3,4],[5,6]]`): `polynomial_kernel(X,X)` default `coef0=1` `= [[42.875, ...]]` vs `coef0=0` `[[15.625, ...]]`. Pinned by `divergence_poly_default_coef0` in `tests/divergence_nystroem.rs` (poly degree=1 full-basis reconstruction vs the live oracle; was off by exactly 1.0, now green). (`degree` default 3 MATCHES; `gamma` default `1/n_features` MATCHES — REQ-2.) Deterministic / oracle-pinnable. |
| REQ-10 (attribute + constructor surface) | NOT-STARTED | blocker issue to be filed by critic (R-DEV-2/3). `FittedNystroem` exposes `basis()` and `n_components()` only — NO `components_`/`component_indices_`/`normalization_` accessors (sklearn attrs `:885-893`; `component_indices_` is `basis_inds`, untracked because the shuffle indices are dropped after building `basis`). The constructor uses a fixed `KernelType` enum (`Rbf/Polynomial/Linear/Sigmoid`) — no string/callable/`"precomputed"` kernel, no `kernel_params`, no `n_jobs` (sklearn `:957-976`). Live oracle exposes `m.component_indices_ = [6 2 1 7 3 0 5 4]` and `m.normalization_.shape = (8,8)`. Attribute/constructor-coverage gap. |
| REQ-11 (ferray substrate) | NOT-STARTED | blocker issue to be filed by critic (R-SUBSTRATE-1). `nystroem.rs` imports `ndarray::{Array1, Array2}`, `rand::{SeedableRng, seq::SliceRandom}`, `rand_xoshiro::Xoshiro256PlusPlus`, and `faer::Mat`/`self_adjoint_eigen` (in `symmetric_eigen`); tests use `rand_distr::Normal`. Destination: `ferray-core` (array), `ferray::linalg` (eigendecomposition), `ferray::random` (basis sampling — the `numpy.random` analog), `ferray::numpy_interop` (PyO3 bridge). Not migrated. The RNG migration (REQ-8) and this REQ are linked: a numpy-compatible `ferray::random` is the prerequisite for exact basis-permutation parity. |
| REQ-12 (consumer) | SHIPPED | `lib.rs` re-exports `pub use nystroem::{FittedNystroem, KernelType, Nystroem}`. Python binding: `ferrolearn-python/src/extras.rs` (`py_transformer!(RsNystroem, "_RsNystroem", FittedNystroem<f64>, (), { Nystroem::<f64>::new() })`), registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsNystroem>()`), wrapped by `ferrolearn-python/python/ferrolearn/_extras.py` (`class Nystroem(_TransformerWrapper)` → `_RsNystroem()`), exported in `__init__.py`. Note: the binding's constructor block is bare `{ Nystroem::<f64>::new() }` — `kernel`/`gamma`/`degree`/`coef0`/`n_components`/`random_state` are NOT plumbed through, so the Python surface hardcodes defaults (rbf, `gamma=None→1/n_features`, `n_components=100`) and cannot reach sklearn's constructor params (sub-finding tracked under REQ-9/REQ-10). |

## Architecture

ferrolearn splits the estimator into the unfitted builder `Nystroem<F>` (fields
`kernel: KernelType`, `gamma: Option<F>`, `degree: usize`, `coef0: F`,
`n_components: usize`, `random_state: Option<u64>`) and the fitted
`FittedNystroem<F>` (`basis: Array2<F>`, `normalization: Array2<F>`, plus the
kernel hyperparameters), matching sklearn's `Nystroem` / post-`fit` attribute
split — sklearn keeps one class and sets `components_`/`component_indices_`/
`normalization_` on `self` in `fit` (`kernel_approximation.py:1032-1034`). The
builder API is method-chained (`with_kernel`, `with_gamma`, `with_n_components`,
`with_random_state`, …) where sklearn uses keyword constructor args (`:957-976`).

The kernel evaluation is shared with `KernelRidge`: `KernelType` enumerates
`{Rbf, Polynomial, Linear, Sigmoid}`, and `kernel_value`/`compute_kernel_matrix`
implement the four formulas. sklearn instead routes every kernel through
`pairwise_kernels(metric=self.kernel, ...)` (`:1021,1058`), which admits the
full `PAIRWISE_KERNEL_FUNCTIONS` set plus `"precomputed"` and callables — the
source of REQ-10's gap.

The numerical core is `fn fit` and `fn symmetric_eigen`. `fit` selects the basis
(at `n_basis >= n_samples`, `x.clone()` in original order — sklearn always
permutes, REQ-8; otherwise a seeded `Xoshiro256++` shuffle truncated to
`n_basis`), computes `K_qq = compute_kernel_matrix(basis, basis, ...)`,
eigendecomposes `K_qq = V D Vᵀ` (`symmetric_eigen` converts to `faer::Mat<f64>`,
calls `self_adjoint_eigen(Side::Lower)`, sorts eigenpairs by descending
eigenvalue, converts back to `F`), and builds `normalization[i,j] =
V[i,j]/√D_j` for `D_j > 1e-12` (else the column stays zero — REQ-7). sklearn
takes the SVD `U,S,V = svd(K_qq)`, floors `S` to `1e-12`, and stores the
SYMMETRIC `normalization_ = U/√S · V` (`= K_qq^{-1/2}`, `:1032`). `transform`
computes `K_nq = compute_kernel_matrix(X, basis, ...)` and returns
`K_nq · normalization`; sklearn returns `embedded · normalization_ᵀ` (`:1066`).

The crucial structural point (REQ-6): ferrolearn's `V·D^{-1/2}` and sklearn's
symmetric `V·D^{-1/2}·Vᵀ` (`= U/√S · V` for symmetric PSD `K_qq`) differ by an
orthogonal `Vᵀ`. Since the Nystroem reconstruction `Z·Zᵀ =
K_nq·(V·D^{-1/2})·(V·D^{-1/2})ᵀ·K_qnᵀ = K_nq·K_qq^{-1}·K_qn` is rotation-
invariant, the kernel approximation (REQ-1) is identical, but the element-wise
embedding `Z` is rotated relative to sklearn — so any test that compares
`transform` outputs element-wise (rather than the Gram matrix `Z·Zᵀ`) diverges.

Invariants: `K_qq` is symmetric PSD; `basis` is `(n_basis, n_features)` with
`n_basis = min(n_components, n_samples)`; `normalization` is
`(n_basis, n_basis)`; `transform` requires `X.ncols() == basis.ncols()` and
otherwise returns `ShapeMismatch`. `fit` rejects `n_components == 0`
(`InvalidParameter`) and empty input (`n_samples == 0` → `InsufficientSamples`);
sklearn's `_validate_data` raises on 0 samples, and `_parameter_constraints`
rejects `n_components < 1` — both reject, though the exception class differs.

## Verification

Commands establishing the SHIPPED claims (run at baseline `3aef2e880`):

- `cargo test -p ferrolearn-kernel --lib nystroem` — exercises
  `basic_fit_transform_rbf`, `output_shape`, `n_components_exceeds_n_samples`
  (REQ-4 clamp), `kernel_approximation_quality_rbf` (REQ-1),
  `polynomial_kernel`/`linear_kernel`/`sigmoid_kernel` (REQ-3),
  `reproducible_with_seed`, `rejects_zero_components` (REQ-5),
  `rejects_empty_input`, `default_gamma_scales_with_features` (REQ-2),
  `single_sample`, `f32_support`.
- Reconstruction oracle (REQ-1, R-CHAR-3 — expected from `rbf_kernel`, never
  from ferrolearn's transform). At `n_components == n_samples` (all points are
  the basis):
  `python3 -c "import numpy as np; from sklearn.kernel_approximation import Nystroem; from sklearn.metrics.pairwise import rbf_kernel; np.random.seed(0); X=np.random.randn(8,3); m=Nystroem(kernel='rbf',gamma=0.5,n_components=8,random_state=0).fit(X); Z=m.transform(X); print(np.max(np.abs(Z@Z.T - rbf_kernel(X,X,gamma=0.5))))"`
  → `1.55e-15`. A critic pins this as a Rust `#[test]` that fits a ferrolearn
  `Nystroem` at full basis, computes `Z·Zᵀ`, and asserts equality with
  `rbf_kernel(X,X,gamma)` (the kernel formula recomputed in-test) — verifying
  the reconstruction without RNG bit-match. The in-crate
  `kernel_approximation_quality_rbf` already pins the self-kernel diagonal ≈ 1.
- Default-gamma oracle (REQ-2): `1/n_features` (live `1/5 = 0.2`); the in-crate
  `default_gamma_scales_with_features` asserts `fitted.gamma == 0.2`.

Open divergences pinned as FAILING tests (NOT-STARTED, oracle expected values):

- REQ-4 (warning): `Nystroem(n_components=100).fit(X)` on `n_samples=3` emits a
  `UserWarning` "n_components > n_samples" in sklearn; ferrolearn is silent. The
  warning emission is the pinned divergence (the clamp is already correct).
- REQ-6 (normalization form): with an IDENTICAL basis (full-basis case),
  `Z_ferro·Z_ferroᵀ ≈ Z_sk·Z_skᵀ` (REQ-1) but `‖Z_ferro − Z_sk‖ ≉ 0` — a
  critic fixes the basis and asserts element-wise mismatch, failing until
  ferrolearn restores the symmetric `V·D^{-1/2}·Vᵀ` (`= normalization·Vᵀ`).
- REQ-7 (eigenvalue floor): on a rank-deficient `K_qq` (duplicate basis rows),
  sklearn's floored column scales by `1/√(1e-12)=1e6` while ferrolearn zeroes
  it; the embeddings differ in the floored direction.
- REQ-9 (poly coef0): `Nystroem(kernel='poly')` defaults `coef0=1`; ferrolearn's
  `new()` uses `coef0=0`. Live `polynomial_kernel(X,X)` (default `coef0=1`) `=
  [[42.875, 274.625, 857.375], ...]` vs `coef0=0` `[[15.625, 166.375,
  614.125], ...]` — ferrolearn's default matches the wrong branch.

The RNG carve-out (REQ-8) gets a blocker but **no failing test** (R-DEFER-3):
ferrolearn's `Xoshiro256++` basis permutation cannot equal numpy's
Mersenne-Twister permutation, and at full basis ferrolearn does not permute at
all — so exact `transform`-value parity is structurally unachievable and is
documented, not tested. REQ-1 (reconstruction, permutation-invariant) is the
slice that ships regardless.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1
(reconstruction at full basis), REQ-2 (default gamma), REQ-3 (kernel formulas),
REQ-5 (n_components validation), REQ-9 (poly/sigmoid coef0 default — FIXED
#1903), REQ-12 (Python-binding consumer) — impl + non-test consumer + green
verification. NOT-STARTED (the critic files per-REQ `-l blocker` issues): REQ-4
(missing clamp warning #1908), REQ-6 (symmetric normalization / transform value
parity #1905), REQ-7 (eigenvalue floor #1907 / SVD-vs-eigen #1906), REQ-8 (exact
RNG / basis-permutation parity #1904 — carve-out, no test), REQ-10
(`component_indices_`/`normalization_`/string-or-callable kernel/`kernel_params`/
`n_jobs` surface #1909), REQ-11 (ferray substrate #1910).
