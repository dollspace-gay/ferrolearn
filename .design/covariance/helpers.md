# Covariance Function-Export Helpers

<!--
tier: 3-component
status: draft
baseline-commit: 2475c0d180ddf32d9a2569e1607a760406ea445b
upstream-paths:
  - sklearn/covariance/_empirical_covariance.py
  - sklearn/covariance/_shrunk_covariance.py
  - sklearn/covariance/_robust_covariance.py
-->

## Summary

`ferrolearn-covariance/src/helpers.rs` mirrors the stateless **function exports**
of `sklearn.covariance` — `empirical_covariance`, `shrunk_covariance`,
`ledoit_wolf`, `ledoit_wolf_shrinkage`, `oas`, `log_likelihood`, and `fast_mcd`.
Six of the seven are thin one-shot wrappers that delegate to the estimator
structs in `covariance.rs` (audited as the sibling doc `.design/covariance/covariance.md`,
tracking #1701); `shrunk_covariance` and `log_likelihood` are computed directly
in `helpers.rs`. Because the underlying estimators ship element-wise-exact value
parity (`EmpiricalCovariance`, `LedoitWolf`, and — since the #1702 fix —
`OAS`), the function-form value parity is SHIPPED and oracle-pinnable. The gaps
are API-shape only (no `shrinkage=0.1` default, no `block_size` param) plus the
`fast_mcd` RNG carve-out, none of which are deterministic value divergences.

## Upstream reference (scikit-learn 1.5.2, commit 156ef14)

- `sklearn/covariance/_empirical_covariance.py:67`
  `def empirical_covariance(X, *, assume_centered=False)` — biased MLE: when
  `assume_centered` it returns `np.dot(X.T, X) / X.shape[0]` (`:107`), else the
  mean-centered `np.cov(X.T, bias=1)` (`:109`). Returns the covariance array.
- `sklearn/covariance/_empirical_covariance.py:33` `def log_likelihood(emp_cov, precision)`:
  ```python
  p = precision.shape[0]
  log_likelihood_ = -np.sum(emp_cov * precision) + fast_logdet(precision)
  log_likelihood_ -= p * np.log(2 * np.pi)
  log_likelihood_ /= 2.0
  return log_likelihood_
  ```
  `np.sum(emp_cov * precision)` is the elementwise-product sum = `tr(emp_cov @ precision)`
  for symmetric inputs. `fast_logdet(precision)` = `np.linalg.slogdet` (sign·log|det|).
  Net formula: `(-tr + logdet(precision) - p·log(2π)) / 2`.
- `sklearn/covariance/_shrunk_covariance.py:111`
  `def shrunk_covariance(emp_cov, shrinkage=0.1)`: `shrunk = (1-shrinkage)*emp_cov`,
  `mu = trace(emp_cov)/n_features` (`:154`), `shrunk += shrinkage*mu*np.eye(n_features)`
  (`:156`). **DEFAULT `shrinkage=0.1`**, constraint `Interval(Real, 0, 1, closed="both")`.
- `sklearn/covariance/_shrunk_covariance.py:299`
  `def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000)`: returns the
  shrinkage scalar only.
- `sklearn/covariance/_shrunk_covariance.py:409`
  `def ledoit_wolf(X, *, assume_centered=False, block_size=1000)`: returns
  `(estimator.covariance_, estimator.shrinkage_)` (`:80`).
- `sklearn/covariance/_shrunk_covariance.py:619`
  `def oas(X, *, assume_centered=False)`: returns `(shrunk_cov, shrinkage)`; the
  1.5.2 OAS formula (`:79-87` of the module) was matched in ferrolearn under #1702.
- `sklearn/covariance/_robust_covariance.py:358`
  `def fast_mcd(X, support_fraction=None, cov_computation_method=empirical_covariance, random_state=None)`:
  returns `(location, covariance, support)` (`:577`); FastMCD — RNG-dependent.

The installed `sklearn` 1.5.2 package is the live oracle.

## Requirements

Each REQ is tagged with its verification class: **oracle-pinnable** (deterministic
value parity vs the live oracle), **structural** (shape/range invariant),
**RNG carve-out** (R-DEFER-3, exact values un-bit-matchable), **API-shape**
(constructor/signature parity gap, no value impact), or **substrate**.

- REQ-EMPIRICAL (empirical_covariance value parity, oracle-pinnable): MATCH.
  `empirical_covariance(X, assume_centered)` delegates to the SHIPPED
  `EmpiricalCovariance` estimator (covariance.md REQ-1) and returns its
  `covariance_`. Mirrors `_empirical_covariance.py:67`.
- REQ-SHRUNK (shrunk_covariance formula, oracle-pinnable): MATCH on the formula.
  `(1-s)*emp_cov + s*(trace(emp_cov)/p)*I`, computed in-file. Mirrors
  `_shrunk_covariance.py:153-156`.
- REQ-LEDOIT-WOLF (ledoit_wolf / ledoit_wolf_shrinkage value parity, oracle-pinnable):
  MATCH. Delegates to the SHIPPED `LedoitWolf` estimator (covariance.md REQ-3),
  returning `(covariance_, shrinkage_)` and the scalar respectively. Mirrors
  `_shrunk_covariance.py:409,299`.
- REQ-OAS (oas value parity, oracle-pinnable): MATCH. Delegates to the `OAS`
  estimator whose 1.5.2 formula was fixed under #1702. Mirrors
  `_shrunk_covariance.py:619`.
- REQ-LOG-LIKELIHOOD (log_likelihood value parity, oracle-pinnable): MATCH.
  ferrolearn returns `-0.5*(p·log(2π) - logdet(precision) + tr(precision·emp_cov))`,
  algebraically identical to sklearn's `(-tr + logdet(precision) - p·log(2π))/2`.
  Verified equal to the live oracle to machine precision on three SPD precisions
  (see Acceptance criteria / Verification). Mirrors `_empirical_covariance.py:33`.
- REQ-FAST-MCD (fast_mcd exact attributes, RNG carve-out): DEVIATE (R-DEFER-3).
  `fast_mcd` delegates to `MinCovDet`, whose h-subset draws use
  `Xoshiro256PlusPlus`, not numpy `RandomState`. Exact `(location, covariance,
  support)` are un-bit-matchable; this is a documented carve-out (no failing
  test), tracked under covariance.md REQ-5 and blocker #1875. The returned tuple
  **shape** `(Array1, Array2, Vec<bool>)` matches sklearn's `(location,
  covariance, support)` (`_robust_covariance.py:577`).
- REQ-DEFAULTS (API-shape parity gap): DEVIATE — minor. `shrunk_covariance`
  takes `shrinkage: F` as a mandatory argument with no `shrinkage=0.1` default
  (sklearn `:111`); `ledoit_wolf`/`ledoit_wolf_shrinkage` omit the `block_size=1000`
  parameter (sklearn `:409,299`). `block_size` is a memory-chunking knob that does
  not change the result. These are signature-shape gaps, not value divergences.
  Blocker #1874.
- REQ-X-1 (R-SUBSTRATE — ferray): DEVIATE. `helpers.rs` uses `ndarray::{Array1,
  Array2}` and a hand-rolled `fn log_det_spd` (Cholesky + adaptive diagonal
  regularisation) rather than `ferray-core` arrays and `ferray::linalg`. Blocker
  #1876.
- REQ-X-2 (non-test consumer): MATCH. All seven functions are re-exported from
  `lib.rs` (`pub use helpers::{empirical_covariance, fast_mcd, ledoit_wolf,
  ledoit_wolf_shrinkage, log_likelihood, oas, shrunk_covariance}`).

## Acceptance criteria

Each AC carries a live oracle command (sklearn 1.5.2). Probe set
`X = [[1,2,3],[4,5,6],[7,8,10],[2,3,5],[3,1,2],[5,9,4]]` unless noted.

- AC-EMPIRICAL (REQ-EMPIRICAL): `empirical_covariance(X, false)` matches
  `python3 -c "import numpy as np; from sklearn.covariance import empirical_covariance; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); print(empirical_covariance(X)[0].tolist())"`
  → `[3.8888888888888893, 4.888888888888888, 3.833333333333333]` within 1e-10.
- AC-SHRUNK (REQ-SHRUNK): `shrunk_covariance(emp_cov, 0.1)[0][0]` matches
  `python3 -c "import numpy as np; from sklearn.covariance import empirical_covariance, shrunk_covariance; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); print(shrunk_covariance(empirical_covariance(X), shrinkage=0.1)[0][0])"`
  → `4.148148148148149` within 1e-10. (ferrolearn: `4.148148148148149`.)
- AC-LEDOIT-WOLF (REQ-LEDOIT-WOLF): `ledoit_wolf(X, false).1` matches
  `python3 -c "import numpy as np; from sklearn.covariance import ledoit_wolf; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); c,s=ledoit_wolf(X); print(s, c[0][0])"`
  → `shrinkage=0.4087180724344861`, `cov[0][0]=4.94852833594126` within 1e-12.
  (ferrolearn: `0.4087180724344862`, `4.948528335941261`.) `ledoit_wolf_shrinkage`
  returns the same scalar.
- AC-OAS (REQ-OAS): `oas(X, false).1` matches
  `python3 -c "import numpy as np; from sklearn.covariance import oas; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); c,s=oas(X); print(s, c[0][0])"`
  → `shrinkage=0.6705854984555868`, `cov[0][0]=5.627443884884855` within 1e-12.
  (ferrolearn: `0.6705854984555868`, exact.)
- AC-LOG-LIKELIHOOD (REQ-LOG-LIKELIHOOD): `log_likelihood(emp_cov, precision)`
  matches sklearn on three SPD precisions —
  `python3 -c "import numpy as np; from sklearn.covariance import empirical_covariance, log_likelihood; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); ec=empirical_covariance(X); print(log_likelihood(np.eye(3), np.eye(3)), log_likelihood(0.5*np.eye(3), 2.0*np.eye(3)), log_likelihood(ec, np.linalg.inv(ec)))"`
  → `(-4.2568155996140185, -3.2170948287741004, -5.970663124061147)`. ferrolearn
  returns `(-4.2568155996140185, -3.2170948287741, -5.970663124061147)` — match to
  machine precision (the real-precision case is bit-exact).
- AC-FAST-MCD (REQ-FAST-MCD): `fast_mcd(X, 0.75, Some(seed))` returns a tuple
  `(location: Array1, covariance: Array2, support: Vec<bool>)` of the correct
  shapes; exact values are NOT oracle-compared (RNG carve-out). Structural smoke:
  `support.len() == n_samples`, `location.len() == n_features`,
  `covariance.dim() == (p, p)`.
- AC-CONSUMER (REQ-X-2): all seven functions are reachable as
  `ferrolearn_covariance::{empirical_covariance, ...}` via the `lib.rs` re-export.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-EMPIRICAL (empirical_covariance) | SHIPPED | `pub fn empirical_covariance in helpers.rs` delegates `EmpiricalCovariance::<F>::new().assume_centered(...).fit(x, &())` and returns `fitted.covariance().clone()` (covariance.md REQ-1, SHIPPED). Mirrors `_empirical_covariance.py:67` (`covariance = np.dot(X.T, X) / X.shape[0]`). Live oracle: `empirical_covariance(X)[0] = [3.8888888888888893, 4.888888888888888, 3.833333333333333]`; ferrolearn `[3.8888888888888893, 4.888888888888888, 3.8333333333333335]` — match ~1e-15. Non-test consumer: re-exported in `lib.rs` (`pub use helpers::{empirical_covariance, ...}`); also consumed by `graphical_lasso.rs` (`let emp_cov = empirical_covariance(x, self.assume_centered)?`). Oracle-pinnable. |
| REQ-SHRUNK (shrunk_covariance formula) | SHIPPED | `pub fn shrunk_covariance in helpers.rs` computes `mu = diag_sum/n`, `out = (1-shrinkage)*emp_cov`, `out[[i,i]] += shrinkage*mu`. Mirrors `_shrunk_covariance.py:153-156`. Live oracle (`shrinkage=0.1`): `shrunk_covariance(emp_cov, 0.1)[0][0] = 4.148148148148149`; ferrolearn `4.148148148148149` — exact. Non-test consumer: re-exported in `lib.rs`. Oracle-pinnable. The absent `shrinkage=0.1` DEFAULT is folded into REQ-DEFAULTS (API-shape, NOT-STARTED) — the formula itself is bit-exact. |
| REQ-LEDOIT-WOLF (ledoit_wolf, ledoit_wolf_shrinkage) | SHIPPED | `pub fn ledoit_wolf in helpers.rs` delegates `LedoitWolf::<F>::new().assume_centered(...).fit(x,&())` → `(covariance().clone(), shrinkage())`; `pub fn ledoit_wolf_shrinkage in helpers.rs` is `ledoit_wolf(...).1` (covariance.md REQ-3, SHIPPED). Mirrors `_shrunk_covariance.py:409,299,:80`. Live oracle: `ledoit_wolf(X)` → `shrinkage=0.4087180724344861`, `cov[0][0]=4.94852833594126`; ferrolearn `0.4087180724344862`, `4.948528335941261` — match ~1e-16. Non-test consumer: re-exported in `lib.rs`. Oracle-pinnable. The absent `block_size` param (memory-chunking, no value impact) is folded into REQ-DEFAULTS. |
| REQ-OAS (oas) | SHIPPED | `pub fn oas in helpers.rs` delegates `OAS::<F>::new().assume_centered(...).fit(x,&())` → `(covariance().clone(), shrinkage())`. Mirrors `_shrunk_covariance.py:619`. The 1.5.2 OAS formula was fixed in the `OAS` estimator under #1702. Live oracle: `oas(X)` → `shrinkage=0.6705854984555868`, `cov[0][0]=5.627443884884855`; ferrolearn `0.6705854984555868`, `5.627443884884855` — exact. Non-test consumer: re-exported in `lib.rs`. Oracle-pinnable. NOTE: covariance.md REQ-4 still records the PRE-#1702 divergence; the live oracle at this baseline (`2475c0d`) shows `OAS` now matches, so the helper function value-parity is SHIPPED. |
| REQ-LOG-LIKELIHOOD (log_likelihood) | SHIPPED | `pub fn log_likelihood in helpers.rs` returns `-half*(n_f*two_pi.ln() - logdet_prec + tr)` where `logdet_prec = log_det_spd(precision)` and `tr = sum_{i,k} precision[[i,k]]*emp_cov[[k,i]]`. Algebraically `0.5*(logdet(precision) - tr - p*log(2π))` = sklearn's `(-tr + logdet(precision) - p*log(2π))/2` (`_empirical_covariance.py:33`, `-np.sum(emp_cov*precision) + fast_logdet(precision) - p*np.log(2π)`, all `/2`). VERIFIED against the live oracle on three SPD precisions: sklearn `(-4.2568155996140185, -3.2170948287741004, -5.970663124061147)`; ferrolearn `(-4.2568155996140185, -3.2170948287741, -5.970663124061147)` — match to machine precision, real-precision case bit-exact. Non-test consumer: re-exported in `lib.rs`. Oracle-pinnable. CAVEAT (does not affect parity): for a VALID SPD precision `log_det_spd` adds zero shift, so it coincides with `slogdet`; it diverges only on a near-singular/non-SPD precision (adaptive `tau` regularisation), where sklearn's `slogdet` would instead return a near-`-inf`/sign-flipped result — outside the SPD contract this function documents. |
| REQ-FAST-MCD (fast_mcd exact attrs) | NOT-STARTED | open prereq blocker #1875 (R-DEFER-3 RNG carve-out — NO failing test). `pub fn fast_mcd in helpers.rs` delegates `MinCovDet::<F>::new().support_fraction(...).random_state(...).fit(x,&())` → `(location().clone(), covariance().clone(), support().to_vec())`. MinCovDet's h-subset draws use `Xoshiro256PlusPlus`, which cannot bit-match numpy `RandomState` in sklearn's `fast_mcd` (`_robust_covariance.py:358`, returns `(location, covariance, support)` at `:577`). Exact `(location, covariance, support)` are un-bit-matchable (covariance.md REQ-5); documented carve-out. The returned tuple SHAPE matches sklearn. |
| REQ-DEFAULTS (shrinkage=0.1, block_size=1000) | NOT-STARTED | open prereq blocker #1874 (API-shape, minor). `pub fn shrunk_covariance in helpers.rs` requires `shrinkage: F` (no `shrinkage=0.1` default; sklearn `_shrunk_covariance.py:111`). `pub fn ledoit_wolf`/`ledoit_wolf_shrinkage in helpers.rs` omit `block_size=1000` (sklearn `:409,299`); `block_size` is a memory-chunking knob with no value impact. Signature-shape gaps, not value divergences. |
| REQ-X-1 (R-SUBSTRATE ferray) | NOT-STARTED | open prereq blocker #1876 (R-SUBSTRATE). `helpers.rs` imports `ndarray::{Array1, Array2}` and uses a hand-rolled `fn log_det_spd` (Cholesky + adaptive diagonal regularisation). Destination substrate is `ferray-core` (array) + `ferray::linalg` (logdet/Cholesky). Not migrated (consistent with covariance.md REQ-18). |
| REQ-X-2 (non-test consumer) | SHIPPED | `lib.rs` re-exports all seven functions: `pub use helpers::{empirical_covariance, fast_mcd, ledoit_wolf, ledoit_wolf_shrinkage, log_likelihood, oas, shrunk_covariance};`. `empirical_covariance` is additionally consumed by `graphical_lasso.rs` (`GraphicalLasso::fit`). The functions ARE the public function-export surface (sklearn `from sklearn.covariance import empirical_covariance`); boundary S5 grandfathers them. |

## Architecture

`helpers.rs` is the function-export layer over the estimator structs in
`covariance.rs` (sibling doc `.design/covariance/covariance.md`, tracking #1701).
sklearn exposes both forms — `EmpiricalCovariance().fit(X).covariance_` AND
`empirical_covariance(X)` — and ferrolearn mirrors that duality: the helper
functions construct the estimator, call `.fit`, and project the relevant fitted
attribute(s).

**Delegating wrappers.** `empirical_covariance` → `EmpiricalCovariance`;
`ledoit_wolf`/`ledoit_wolf_shrinkage` → `LedoitWolf` (the scalar form is
`ledoit_wolf(...).1`); `oas` → `OAS`; `fast_mcd` → `MinCovDet`. Each returns
`fitted.<attr>().clone()`. Their value parity therefore inherits directly from
the estimator REQs in covariance.md (REQ-1, REQ-3, REQ-4-via-#1702, REQ-5).

**In-file computations.** `shrunk_covariance` and `log_likelihood` do not
delegate. `shrunk_covariance` recomputes `mu = trace(emp_cov)/p` and the convex
combination directly (matching `_shrunk_covariance.py:153-156`).
`log_likelihood` computes `tr(precision·emp_cov)` as a double loop and
`log|precision|` via `fn log_det_spd` — a Cholesky `2·Σ log diag(L)` with an
adaptive diagonal shift `tau` that grows geometrically until `A + tau·I`
factorises. For a valid SPD precision the shift stays zero and `log_det_spd`
coincides with sklearn's `fast_logdet` (= `slogdet`); the regularisation only
engages on near-singular/non-SPD inputs (e.g. a truncated graphical-lasso
precision), which is outside the SPD contract the docstring states.

**log_likelihood algebra (REQ-LOG-LIKELIHOOD).** ferrolearn writes
`-0.5*(p·log(2π) - logdet(precision) + tr)`; distributing the `-0.5` gives
`0.5·logdet(precision) - 0.5·tr - 0.5·p·log(2π)`, identical term-by-term to
sklearn's `(fast_logdet(precision) - np.sum(emp_cov*precision) - p·log(2π))/2`.
The two are the same expression; the live oracle confirms bit-exactness on the
real-precision probe (`-5.970663124061147`).

**OAS note (REQ-OAS).** The sibling covariance.md records REQ-4 (the OAS
estimator) as NOT-STARTED against the PRE-#1702 superseded formula. At this
doc's baseline commit (`2475c0d`) the live oracle shows `oas(X)` matching sklearn
exactly (`shrinkage=0.6705854984555868`), i.e. the #1702 fix has landed. The
helper `oas` is therefore SHIPPED on value parity; covariance.md REQ-4 should be
re-audited to SHIPPED on its next pass.

**fast_mcd RNG (REQ-FAST-MCD).** `MinCovDet` draws random h-subsets with
`Xoshiro256PlusPlus` seeded from a `u64`; sklearn uses numpy `RandomState`. The
streams cannot be bit-matched, so exact `(location, covariance, support)` are an
R-DEFER-3 carve-out (blocker #1875, NO failing test). Closing it requires the
`ferray::random` numpy-`RandomState` analog (REQ-X-1 / blocker #1876). The tuple
shape `(Array1, Array2, Vec<bool>)` matches sklearn's
`(location, covariance, support)` (`_robust_covariance.py:577`).

**Invariants.** `shrunk_covariance` and `log_likelihood` require square
`emp_cov`/`precision` (else `FerroError::ShapeMismatch`); `log_likelihood`
returns `FerroError::NumericalInstability` if the precision does not factorise
even after regularisation.

## Verification

Commands establishing the SHIPPED claims (run at baseline `2475c0d`):

- `cargo test -p ferrolearn-covariance --lib` → in-file `helpers` smoke tests
  (`test_empirical_covariance`, `test_shrunk_covariance`, `test_ledoit_wolf`,
  `test_oas_helper`, `test_log_likelihood_basic`, `test_fast_mcd_smoke`) pass.
- Live sklearn oracle (sklearn 1.5.2), probe
  `X=[[1,2,3],[4,5,6],[7,8,10],[2,3,5],[3,1,2],[5,9,4]]`, cross-checked against
  the ferrolearn functions via a `tests/` harness calling the re-exported
  functions:
  - `empirical_covariance(X)[0]` sklearn
    `[3.8888888888888893, 4.888888888888888, 3.833333333333333]`; ferrolearn
    matches ~1e-15 (REQ-EMPIRICAL ✓).
  - `shrunk_covariance(emp_cov, 0.1)[0][0]` sklearn `4.148148148148149`;
    ferrolearn `4.148148148148149` (REQ-SHRUNK ✓).
  - `ledoit_wolf(X)` sklearn `(0.4087180724344861, cov[0][0]=4.94852833594126)`;
    ferrolearn `(0.4087180724344862, 4.948528335941261)` (REQ-LEDOIT-WOLF ✓).
  - `oas(X)` sklearn `(0.6705854984555868, cov[0][0]=5.627443884884855)`;
    ferrolearn `(0.6705854984555868, 5.627443884884855)` (REQ-OAS ✓, post-#1702).
  - `log_likelihood` on `(I,I)`, `(0.5I, 2I)`, `(emp_cov, inv(emp_cov))`:
    sklearn `(-4.2568155996140185, -3.2170948287741004, -5.970663124061147)`;
    ferrolearn `(-4.2568155996140185, -3.2170948287741, -5.970663124061147)` —
    match to machine precision (REQ-LOG-LIKELIHOOD ✓).

A critic should pin REQ-EMPIRICAL, REQ-SHRUNK, REQ-LEDOIT-WOLF, REQ-OAS, and
REQ-LOG-LIKELIHOOD as `#[test]`s whose expected values come from the live oracle
commands above (never copied from the ferrolearn side, R-CHAR-3). REQ-FAST-MCD is
an RNG carve-out (blocker #1875, NO failing test per R-DEFER-3).

Per R-DEFER-2 the table is binary. SHIPPED: REQ-EMPIRICAL, REQ-SHRUNK,
REQ-LEDOIT-WOLF, REQ-OAS, REQ-LOG-LIKELIHOOD, REQ-X-2 (impl + non-test consumer +
green oracle/structural verification). NOT-STARTED: REQ-FAST-MCD (RNG carve-out,
blocker #1875), REQ-DEFAULTS (API-shape, blocker #1874), REQ-X-1 (ferray
substrate, blocker #1876).
