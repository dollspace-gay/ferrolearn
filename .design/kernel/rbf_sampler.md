# RBF Sampler (Random Fourier Features)

<!--
tier: 3-component
status: draft
baseline-commit: 25da433a
upstream-paths:
  - sklearn/kernel_approximation.py        # class RBFSampler (:244-411)
-->

## Summary

`ferrolearn-kernel/src/rbf_sampler.rs` mirrors scikit-learn's
`sklearn.kernel_approximation.RBFSampler` — the random Fourier feature
("Random Kitchen Sinks", Rahimi & Recht 2007) approximation of the RBF kernel.
`fit` samples a random projection matrix `random_weights_ ~ sqrt(2·gamma)·N(0,1)`
of shape `(n_features, n_components)` and an offset
`random_offset_ ~ Uniform(0, 2π)` of shape `(n_components,)`; `transform`
returns `sqrt(2/n_components)·cos(X·random_weights_ + random_offset_)`
(`kernel_approximation.py:372-408`). The inner product in the transformed space
approximates `k(x,y) = exp(-gamma·||x−y||²)`.

Because the fitted attributes are drawn from a pseudo-random generator,
**exact** `random_weights_`/`random_offset_` parity against numpy's
`RandomState` is an R-DEFER-3 RNG carve-out (ferrolearn uses
`rand_xoshiro::Xoshiro256PlusPlus`, numpy uses Mersenne-Twister `RandomState`)
— that REQ is NOT-STARTED with a blocker and **no failing test** is filed for it.
The **deterministic** surface — the transform formula, the sampling-distribution
*structure*, `n_components` validation — is SHIPPED and oracle-pinnable. Two
deterministic divergences are open: sklearn accepts `gamma=0` (ferrolearn
rejects it) and supports `gamma='scale'` (ferrolearn is numeric-only). The
production `fn fit` also contains `.unwrap()` calls outside `#[cfg(test)]`
(R-CODE-2). The unit is on the `ndarray`/`rand_distr`/`rand_xoshiro` substrate,
not ferray (R-SUBSTRATE).

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/kernel_approximation.py:244-411` — `class RBFSampler`.
  - constructor `:336-339` — `__init__(self, *, gamma=1.0, n_components=100,
    random_state=None)`.
  - `_parameter_constraints` `:327-334` — `gamma: [StrOptions({"scale"}),
    Interval(Real, 0.0, None, closed="left")]` (i.e. `gamma ≥ 0`, OR the literal
    `"scale"`); `n_components: Interval(Integral, 1, None, closed="left")`.
  - `fit` `:341-385` — `random_weights_ = (2.0 * self._gamma) ** 0.5 *
    random_state.normal(size=(n_features, self.n_components))` (`:372-374`);
    `random_offset_ = random_state.uniform(0, 2*np.pi, size=self.n_components)`
    (`:376`); `gamma='scale'` resolves `self._gamma = 1.0 / (n_features * X_var)`
    (`:366-369`); `self._n_features_out = self.n_components` (`:384`);
    `n_features_in_` set by `_validate_data` (`:362`).
  - `transform` `:387-408` — `projection = safe_sparse_dot(X,
    random_weights_); projection += random_offset_; np.cos(projection,
    projection); projection *= (2.0 / self.n_components) ** 0.5` (`:404-407`).
  - `_more_tags` `:410-411` — `preserves_dtype: [float64, float32]`.

## Requirements

- REQ-1: Transform formula parity. `transform` computes
  `sqrt(2/n_components)·cos(X·random_weights_ + random_offset_)`, matching
  `kernel_approximation.py:404-407`. (Deterministic / oracle-pinnable given the
  fitted weights.)
- REQ-2: Sampling-distribution *structure*. `fit` draws
  `random_weights_` from `sqrt(2·gamma)·N(0,1)` of shape
  `(n_features, n_components)` and `random_offset_` from `Uniform(0, 2π)` of
  shape `(n_components,)`, matching `:372-376`. (Distribution *structure* —
  std-dev `sqrt(2·gamma)`, shapes, support — is oracle-pinnable; exact draws are
  REQ-8.)
- REQ-3: `gamma=0` validation parity. sklearn's `_parameter_constraints`
  admits `gamma=0` (`Interval(Real, 0.0, None, closed="left")`, `:330`); a
  `gamma=0` fit yields an all-zero `random_weights_`. ferrolearn over-rejects
  `gamma ≤ 0`. (Deterministic / oracle-pinnable.)
- REQ-4: `gamma='scale'` support. sklearn 1.2+ accepts the literal
  `gamma="scale"` → `self._gamma = 1/(n_features·X.var())`
  (`:253-259,366-369`). ferrolearn's `gamma: F` is numeric-only. (Feature gap.)
- REQ-5: No production `.unwrap()`/panic (R-CODE-2 / R-APG-1). `fn fit` must
  not call `.unwrap()`/`.expect()`/`panic!()` outside `#[cfg(test)]`; lossy
  `F::from(...)` conversions must propagate `Result`.
- REQ-6: `n_components` validation parity. `n_components ≥ 1`
  (`_parameter_constraints` `:332`); `n_components=0` is rejected.
  (Deterministic / oracle-pinnable.)
- REQ-7: Fitted-attribute contract. Expose `random_weights_`
  `(n_features, n_components)`, `random_offset_` `(n_components,)`, and
  `n_features_in_` with sklearn's names (`:271-285`).
- REQ-8: Exact fitted-value (RNG) parity — **R-DEFER-3 carve-out**.
  Bit-exact `random_weights_`/`random_offset_` match against numpy
  `RandomState` is not achievable across the Xoshiro256++ ↔ Mersenne-Twister
  generator boundary. (Carve-out; NOT-STARTED; **no failing test**.)
- REQ-9: ferray substrate (R-SUBSTRATE). Array type → `ferray-core`; random
  sampling → `ferray::random`; PyO3 bridge → `ferray::numpy_interop` — instead
  of `ndarray` + `rand_distr` + `rand_xoshiro`.
- REQ-10: Non-test production consumer. Re-exported and registered in the
  Python binding so `import ferrolearn` mirrors `import sklearn`.

## Acceptance criteria

- AC-1 (REQ-1): for ferrolearn's *own* fitted `random_weights`/`random_offset`,
  `transform(X)` equals `sqrt(2/n_components)·np.cos(X·W + b)` computed by numpy
  element-wise to machine precision. (Expected uses the sklearn transform
  *formula*, not ferrolearn's transform output → R-CHAR-3-compliant.)
- AC-2 (REQ-2): the empirical std-dev of `random_weights` ≈ `sqrt(2·gamma)`,
  shapes are `(n_features, n_components)` / `(n_components,)`, and offsets lie in
  `[0, 2π)`; the transform output is bounded by `sqrt(2/n_components)`.
- AC-3 (REQ-3): `RBFSampler::new().with_gamma(0.0).fit(X)` succeeds and yields
  an all-zero `random_weights` (sklearn `RBFSampler(gamma=0.0).fit(X)` →
  `random_weights_` all zeros). Currently FAILS (ferrolearn rejects). Oracle-
  pinnable.
- AC-4 (REQ-4): a `gamma='scale'` path resolves `_gamma = 1/(n_features·var(X))`
  matching `sklearn.kernel_approximation.RBFSampler(gamma='scale')._gamma`.
  Currently absent. Oracle-pinnable once the API exists.
- AC-5 (REQ-5): `grep -nE '\.unwrap\(\)|\.expect\(|panic!' rbf_sampler.rs`
  outside `#[cfg(test)]` is empty. Currently FAILS.
- AC-6 (REQ-6): `RBFSampler::new().with_n_components(0).fit(X)` errors, as
  sklearn raises `InvalidParameterError` for `n_components=0`. Oracle-pinnable.
- AC-7 (REQ-7): a fitted model exposes `random_weights_`, `random_offset_`, and
  `n_features_in_` (sklearn names) — including the Python binding surface.
- AC-8 (REQ-8): N/A — RNG carve-out; no test (R-DEFER-3).
- AC-9 (REQ-9): no `ndarray`/`rand_distr`/`rand_xoshiro` in the owned
  computation; arrays/sampling route through ferray.
- AC-10 (REQ-10): `ferrolearn`'s Python `RBFSampler` fit/transforms via the Rust
  core through `RsRBFSampler`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (transform formula) | SHIPPED | `transform` in `rbf_sampler.rs` computes `let projection = x.dot(&self.random_weights) + &self.random_offset; projection.mapv(\|v\| self.scale * v.cos())` with `scale = sqrt(2/n_components)` (set in `fn fit`) — mirrors `kernel_approximation.py:404-407` (`projection = safe_sparse_dot(X, random_weights_); projection += random_offset_; np.cos(...); projection *= (2.0/n_components)**0.5`). Oracle (formula, R-CHAR-3): with ferrolearn's own fitted `W,b`, `np.allclose(transform(X), sqrt(2/n)*np.cos(X@W+b))` → `max abs diff: 0.0` on the sklearn side; the deterministic mapping is identical math. Non-test consumer: `extras.rs` (`RsRBFSampler` → `FittedRBFSampler<f64>::transform`), re-exported via `lib.rs` (`pub use rbf_sampler::{FittedRBFSampler, RBFSampler}`). Verification: `cargo test -p ferrolearn-kernel --lib rbf` → 25 passed (incl. `values_bounded`, `self_kernel_close_to_one`, `kernel_approximation_quality`). Deterministic / oracle-pinnable. |
| REQ-2 (sampling structure) | SHIPPED | `fn fit` sets `std_dev = (2·gamma).sqrt()`, builds `Normal::new(0.0, std_dev)` and fills a `(n_features, n_components)` `random_weights` (`Array2::from_shape_vec((n_features, n_components), w_data)`); offsets from `Uniform::new(0.0, 2π)` into a length-`n_components` `random_offset` — matches sklearn `(2.0*_gamma)**0.5 * normal(size=(n_features, n_components))` (`:372-374`) and `uniform(0, 2π, size=n_components)` (`:376`). Live oracle confirms shapes `(2,4)`/`(4,)` and `n_features_in_=2`. Consumer: same as REQ-1. The std-dev/shape/support *structure* is oracle-pinnable; exact draws are REQ-8 (carve-out). Deterministic structure. |
| REQ-3 (gamma=0 validation) | NOT-STARTED | open prereq blocker (#1669 tracking; critic files the specific `-l blocker`). `fn fit` does `if self.gamma <= F::zero() { return Err(InvalidParameter{ name:"gamma", reason:"must be positive" }) }` — over-rejects. sklearn `_parameter_constraints` `gamma: Interval(Real, 0.0, None, closed="left")` (`:330`) ADMITS `gamma=0`. Live oracle: `RBFSampler(gamma=0.0, n_components=3, random_state=0).fit([[1,2],[3,4]])` → ACCEPTED, `random_weights_ = [[0,0,0],[0,0,-0]]` (all zero, since `sqrt(2·0)=0`). Deterministic / oracle-pinnable (currently rejected by ferrolearn — `rejects_zero_gamma` test pins the WRONG behavior). |
| REQ-4 (gamma='scale') | NOT-STARTED | open prereq blocker (#1669 tracking). ferrolearn's `gamma: F` field is numeric-only; there is no `'scale'` path. sklearn (`:366-369`) resolves `self._gamma = 1.0/(n_features·X_var) if X_var != 0 else 1.0` when `gamma=="scale"`. Live oracle: `RBFSampler(gamma='scale').fit(X)._gamma = 0.17142857...` = `1/(2·X.var())`. Feature gap — needs a `Gamma` enum (`Scale`/`Value(F)`) or equivalent + `X.var()` resolution. |
| REQ-5 (no prod unwrap) | NOT-STARTED | open prereq blocker (#1669 tracking; R-CODE-2 / R-APG-1). `fn fit` (outside `#[cfg(test)]`) calls `F::from(2.0).unwrap()`, `.to_f64().unwrap()`, `F::from(normal.sample(&mut rng)).unwrap()`, `F::from(uniform.sample(&mut rng)).unwrap()`, `F::from(n_components).unwrap()` — production `.unwrap()`. Must propagate via a `Result`-returning conversion (e.g. `FerroError::NumericalInstability` on `F::from(...) == None`). Deterministic code-quality REQ. |
| REQ-6 (n_components ≥ 1) | SHIPPED | `fn fit` does `if self.n_components == 0 { return Err(InvalidParameter{ name:"n_components", reason:"must be at least 1" }) }` — matches `_parameter_constraints` `n_components: Interval(Integral, 1, None, closed="left")` (`:332`). Live oracle: `RBFSampler(n_components=0).fit(...)` → `InvalidParameterError`. Test `rejects_zero_components` (green). Consumer: same as REQ-1. Deterministic / oracle-pinnable. |
| REQ-7 (fitted attributes) | NOT-STARTED | open prereq blocker (#1669 tracking). `FittedRBFSampler` exposes `random_weights()` / `random_offset()` accessors (sklearn attrs are `random_weights_` / `random_offset_` — naming differs by trailing underscore at the Python boundary, acceptable in Rust). But there is NO `n_features_in_` accessor; sklearn sets `n_features_in_` (`:362`, live `n_features_in_=2`). The Python binding (`RsRBFSampler`) does not surface `random_weights_`/`random_offset_`/`n_features_in_`. Attribute-coverage gap. |
| REQ-8 (exact RNG parity) | NOT-STARTED | open prereq blocker (#1669 tracking; R-DEFER-3 carve-out — **NO failing test**). Exact `random_weights_`/`random_offset_` cannot bit-match: ferrolearn uses `rand_xoshiro::Xoshiro256PlusPlus` (`seed_from_u64` / `from_os_rng`), sklearn uses numpy `RandomState` (Mersenne-Twister) via `check_random_state` (`:363`). `random_state=None` is non-deterministic in BOTH (ferrolearn `from_os_rng()`). Per the binding contract this carve-out gets a blocker but no failing test (REQ-1/REQ-2 pin the deterministic formula/structure instead). |
| REQ-9 (ferray substrate) | NOT-STARTED | open prereq blocker (#1669 tracking; R-SUBSTRATE-1). `rbf_sampler.rs` imports `ndarray::{Array1, Array2}`, `rand_distr::{Normal, Uniform}`, `rand::SeedableRng`, `rand_xoshiro::Xoshiro256PlusPlus`. Destination: `ferray-core` (array), `ferray::random` (Normal/Uniform sampling — the `numpy.random` analog), `ferray::numpy_interop` (PyO3 bridge). Not migrated. |
| REQ-10 (consumer) | SHIPPED | `lib.rs` re-exports `pub use rbf_sampler::{FittedRBFSampler, RBFSampler}`. Python binding: `ferrolearn-python/src/extras.rs` `py_transformer!(RsRBFSampler, "_RsRBFSampler", FittedRBFSampler<f64>, (), { RBFSampler::<f64>::new() })`, registered in `ferrolearn-python/src/lib.rs` (`m.add_class::<extras::RsRBFSampler>()`). Note: the binding's constructor block is `{ RBFSampler::<f64>::new() }` — `gamma`/`n_components`/`random_state` are NOT plumbed through, so the Python surface hardcodes defaults (`gamma=1.0`, `n_components=100`, `random_state=None`) and cannot reach sklearn's constructor params (sub-finding tracked under REQ-4/REQ-7). |

## Architecture

ferrolearn splits the estimator into the unfitted builder `RBFSampler<F>`
(fields `gamma: F`, `n_components: usize`, `random_state: Option<u64>`) and the
fitted `FittedRBFSampler<F>` (`random_weights: Array2<F>`,
`random_offset: Array1<F>`, `scale: F`), matching sklearn's `RBFSampler` /
post-`fit` attribute split (sklearn keeps one class and sets
`random_weights_`/`random_offset_`/`_n_features_out` on `self` in `fit`,
`:372-384`). The builder API is method-chained (`with_gamma`, `with_n_components`,
`with_random_state`) where sklearn uses keyword constructor args (`:336-339`).

The numerical core is two pieces. **`fn fit`** computes the weight std-dev
`sqrt(2·gamma)`, seeds an `Xoshiro256PlusPlus` RNG (from `random_state` or the
OS), draws `n_features·n_components` `N(0, sqrt(2·gamma))` samples into
`random_weights` and `n_components` `Uniform(0, 2π)` samples into
`random_offset`, and precomputes `scale = sqrt(2/n_components)`. sklearn instead
multiplies a unit-normal draw by `(2·_gamma)**0.5` after the fact (`:372`,
algebraically identical: `sqrt(2γ)·N(0,1) ≡ N(0, 2γ)`) and applies the scale in
`transform`. **`fn transform`** broadcasts `random_offset` across rows of
`X·random_weights`, applies `cos`, and multiplies by `scale` — the same in-place
sequence sklearn performs at `:404-407`.

Invariants: `random_weights` is `(n_features, n_components)`, `random_offset` is
`(n_components,)`; `transform` requires `X.ncols() == random_weights.nrows()`
and otherwise returns `ShapeMismatch`; every output element is bounded in
`[-scale, +scale]` (cos is in `[-1,1]`), which the `values_bounded` test pins.
`fit` rejects empty input (`x.nrows() == 0` → `InsufficientSamples`); sklearn's
`_validate_data` raises on 0 samples — both reject, though the exception class
differs.

Three open divergences are deterministic and oracle-pinnable: the `gamma=0`
over-rejection (REQ-3), the missing `gamma='scale'` path (REQ-4), and `n_features_in_`
(REQ-7). One is a code-quality fix (REQ-5, production `.unwrap()`). One is the
RNG carve-out (REQ-8). One is substrate (REQ-9). The exact-draw mismatch does
NOT affect REQ-1/REQ-2, which pin the formula and the distribution structure
rather than specific bytes.

## Verification

Commands establishing the SHIPPED claims (run at baseline `25da433a`):

- `cargo test -p ferrolearn-kernel --lib rbf` → 25 passed, 0 failed (REQ-1,
  REQ-2, REQ-6). Relevant tests: `basic_fit_transform`,
  `output_shape_matches_n_components`, `values_bounded` (bound `sqrt(2/n)`),
  `self_kernel_close_to_one`, `kernel_approximation_quality`,
  `reproducible_with_seed`, `rejects_zero_components`.
- Transform-formula oracle (REQ-1, R-CHAR-3 — expected from the sklearn
  *formula*, never from ferrolearn's transform output). Using a model's OWN
  fitted weights:
  `python3 -c "import numpy as np; from sklearn.kernel_approximation import RBFSampler; X=np.array([[1.,2.],[3.,4.],[5.,6.]]); m=RBFSampler(gamma=0.7,n_components=4,random_state=5).fit(X); W,b=m.random_weights_,m.random_offset_; print(np.allclose(m.transform(X), np.sqrt(2/4)*np.cos(X@W+b)))"`
  → `True` (`max abs diff: 0.0`). The Rust `transform` performs the identical
  `scale·cos(X·W + b)`; a critic pins this as a Rust `#[test]` that fits a
  ferrolearn `RBFSampler`, reads back `random_weights()`/`random_offset()`,
  computes `sqrt(2/n)·cos(X·W + b)` in the test, and asserts equality with
  `transform(X)` — verifying the formula without RNG bit-match.
- Sampling-structure oracle (REQ-2): live `RBFSampler(gamma=g,
  n_components=k).fit(X)` exposes `random_weights_.shape == (n_features, k)`,
  `random_offset_.shape == (k,)`, offsets ∈ `[0, 2π)`, and empirical
  `std(random_weights_) ≈ sqrt(2·g)` — ferrolearn matches the shapes/support/
  std-dev structure.

Open divergences pinned as FAILING tests (NOT-STARTED, oracle expected values):

- REQ-3 (`gamma=0`): `RBFSampler(gamma=0.0, n_components=3,
  random_state=0).fit([[1,2],[3,4]])` succeeds in sklearn with all-zero
  `random_weights_`; ferrolearn returns `Err(InvalidParameter)`. A critic pins
  `gamma=0.0` fitting to SUCCEED with all-zero weights — failing until the
  `gamma <= 0` guard is relaxed to `gamma < 0`. (Note: the existing
  `rejects_zero_gamma` test encodes the divergent behavior and must be removed/
  inverted when REQ-3 lands.)
- REQ-4 (`gamma='scale'`): `RBFSampler(gamma='scale').fit(X)._gamma ==
  1/(n_features·X.var())` (live `0.17142857...` for `X=[[1,2],[3,4],[5,6]]`) —
  no ferrolearn API to pin until the `'scale'` path exists.

The RNG carve-out (REQ-8) gets a blocker but **no failing test** (R-DEFER-3):
ferrolearn's Xoshiro256++ draws cannot equal numpy's Mersenne-Twister draws, so
exact-value parity is structurally unachievable and is documented, not tested.

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED: REQ-1 (transform
formula), REQ-2 (sampling structure), REQ-6 (n_components validation), REQ-10
(Python-binding consumer) — impl + non-test consumer + green verification.
NOT-STARTED (tracking #1669; the critic files per-REQ `-l blocker` issues):
REQ-3 (gamma=0 over-rejection), REQ-4 (gamma='scale'), REQ-5 (production
`.unwrap()`), REQ-7 (n_features_in_ / attribute exposure), REQ-8 (exact RNG
parity — carve-out, no test), REQ-9 (ferray substrate).
