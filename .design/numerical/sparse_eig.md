# scipy.sparse.linalg.eigsh — Lanczos symmetric sparse eigensolver

<!--
tier: 3-component
status: draft
baseline-commit: 9ed63e3e3
upstream-paths:
  - scipy/sparse/linalg/__init__.py    # documented sparse eigen API (eigsh/eigs/svds/lobpcg)
-->

## Summary

`ferrolearn-numerical/src/sparse_eig.rs` is the scipy-analog substrate that
mirrors a slice of **scipy.sparse.linalg.eigsh** (the ARPACK symmetric sparse
eigensolver). It exposes `eigsh(mat, k, which)` (CSR convenience) and the
underlying `LanczosSolver` builder (`new`/`with_which`/`with_tol`/`with_max_iter`/
`with_ncv`, plus `solve` over a `matvec` closure and `solve_sparse` over an
`sprs::CsMat<f64>`). The algorithm is a hand-rolled **Implicitly Restarted
Lanczos Method**: a full-reorthogonalisation Lanczos iteration (`build_lanczos`)
building an orthonormal Krylov basis and a symmetric tridiagonal `T`, a
hand-rolled implicit-shift symmetric QR eigensolve of `T`
(`tridiag_qr_eigen`/`implicit_qr_step`/`wilkinson_shift`), a thick-restart
compression keeping the `k` wanted Ritz pairs, and a `which`-driven Ritz
selector (`select_indices`). Eigenpairs are returned in an `EigenResult{
eigenvalues: Array1<f64>, eigenvectors: Array2<f64>}`.

For symmetric matrices the eigenvalues are **unique** (a deterministic property
of the matrix), so eigenvalue parity has a strict ground truth: a live
`scipy.sparse.linalg.eigsh(A, k, which=...)[0]` call (cross-checked against the
dense `numpy.linalg.eigvalsh`). Eigenvectors are sign-/rotation-ambiguous, so
their contract is the **residual** `‖A·v − λ·v‖ ≈ 0` and orthonormality
`VᵀV ≈ I`, not element-wise agreement with scipy. eigsh is deterministic and
version-stable for these scalar properties, so the installed **scipy 1.17.1**
is a valid live oracle (the sklearn-1.5.2 / scipy-1.17.1 split is irrelevant —
the eigenvalues of `diag([1..5])` are `[1,2,3,4,5]` in every release).

Divergence classes: (1) **eigenvalue-parity** — `eigsh(A,k,which).eigenvalues`
matches `scipy.sparse.linalg.eigsh(A,k,which=...)[0]` (sorted) to ~1e-8 for
`LargestAlgebraic`/`SmallestAlgebraic`/`LargestMagnitude` on both a diagonal and
a non-trivial indefinite dense-symmetric matrix (SHIPPED); (2)
**eigenvector-residual** — returned vectors satisfy `‖Av−λv‖ ≤ tol` and
`VᵀV ≈ I` (SHIPPED); (3) **missing-which** — scipy `which ∈
{'LM','SM','LA','SA','BE'}`; ferrolearn has `LargestAlgebraic('LA')`,
`SmallestAlgebraic('SA')`, `LargestMagnitude('LM')`, and a `SmallestMagnitude`
variant ('SM') that exists in the enum BUT is unverified/incorrect for the
`eigsh` outer-Lanczos use, and has NO `'BE'` (BothEnds) analog (NOT-STARTED);
(4) **missing-modes/params** — no `sigma` shift-invert (`OPinv`/`Minv`), no
generalized `M`, no user `v0`, no `ncv`/`maxiter`/`tol`/`return_eigenvectors`
mapped to scipy's exact ABI (NOT-STARTED); (5) **missing-companions** —
scipy.sparse.linalg also exposes `eigs` (non-symmetric), `svds` (sparse SVD),
`lobpcg`; ferrolearn has none (NOT-STARTED); (6) **edge-cases** — scipy requires
`k < n` (raises for `k>=n`) and raises `ArpackNoConvergence`; ferrolearn allows
`k==n` (special dense-Lanczos path), returns `Err(String)` on non-convergence,
and rejects `k==0` (NOT-STARTED, ABI divergence); (7) **error-type** —
`Result<_, String>` vs `FerroError`; (8) **no-consumer** — nothing in the
workspace calls `sparse_eig::` (the spectral methods that COULD —
`spectral_embedding.rs`, MDS, isomap, LLE — hand-roll their own dense
eigensolve and do not wire to this module); (9) **ferray-substrate** — the
Lanczos basis ops and the tridiagonal QR are hand-rolled on `ndarray`/`sprs`
instead of routing through `ferray::linalg` / ferray's sparse analog.

## Upstream reference (scipy.sparse.linalg, live oracle scipy 1.17.1)

The documented sparse-eigen surface lives in `scipy/sparse/linalg/__init__.py`.
The numerical kernel is ARPACK (Fortran), so cite the scipy.sparse.linalg
**function names** and the **live-oracle values**, never the Fortran line
numbers. Documented eigen surface: `eigs` (non-symmetric), `eigsh` (symmetric),
`lobpcg`, `svds`. The symmetric driver signature is:

```
eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None,
      maxiter=None, tol=0, return_eigenvectors=True, Minv=None,
      OPinv=None, mode='normal')
```

with `which ∈ {'LM','SM','LA','SA','BE'}`. The ferrolearn surface maps only
`k`, `which` (3 of 5 values), `tol` (via `with_tol`), `maxiter` (via
`with_max_iter`), and `ncv` (via `with_ncv`) — `M`/`sigma`/`v0`/`Minv`/`OPinv`/
`mode`/`return_eigenvectors` have no analog.

Live oracle (`cd /tmp && python3 -c "import numpy as np, scipy.sparse as sp;
from scipy.sparse.linalg import eigsh; ..."`, scipy 1.17.1):

- `A = csr(diag([1,2,3,4,5]))`:
  `eigsh(A,k=2,'LA')[0] = [4.0, 4.999999999999998]`;
  `eigsh(A,k=2,'SA')[0] = [1.0, 1.9999999999999998]`;
  `eigsh(A,k=2,'LM')[0] = [3.9999999999999996, 5.0]`.
- Indefinite dense-symmetric `M` (seed 0, `Q@diag([-3,-1,0.5,2,4,7])@Qᵀ`,
  symmetrised), full spectrum `np.linalg.eigvalsh(M) =
  [-3.0, -1.0, 0.5, 2.0, 4.0, 7.0]`:
  `eigsh(Ms,k=3,'LA')[0] = [2.0, 4.0, 7.0]` (largest algebraic);
  `eigsh(Ms,k=3,'SA')[0] = [-3.0, -1.0, 0.5]` (smallest algebraic);
  `eigsh(Ms,k=3,'LM')[0] = [-3.0, 4.0, 7.0]` (largest MAGNITUDE — picks the
  negative `-3.0` over the positive `2.0`, the strict indefinite-LM check).
- Edge cases: `eigsh(A,k=5,'LA')` (k==n) → `RuntimeWarning` +
  `TypeError("Cannot use scipy.linalg.eigh for sparse A with k >= N…")`;
  `eigsh(A,k=0,'LA')` → `ValueError("k must be greater than 0.")`;
  `eigsh(A,k=4,'BE')[0] = [1.0, 2.0, 4.0, 5.0]` (both ends);
  `eigsh(A,k=2,'SM')[0] = [1.0, 2.0]` (smallest magnitude).

## Requirements

- REQ-1: **eigsh eigenvalue parity — LargestAlgebraic ('LA').**
  `eigsh(A,k,WhichEigenvalues::LargestAlgebraic).eigenvalues` equals
  `scipy.sparse.linalg.eigsh(A,k,which='LA')[0]` (sorted) to ~1e-8 (the Lanczos
  tolerance) on BOTH a diagonal matrix and a non-trivial indefinite
  dense-symmetric matrix. Eigenvalues are unique → strict check.
- REQ-2: **eigsh eigenvalue parity — SmallestAlgebraic ('SA').** Same, for
  `which='SA'`, verified on the diagonal and the indefinite matrix (the smallest
  algebraic includes the negative eigenvalues `[-3,-1,0.5]`).
- REQ-3: **eigsh eigenvalue parity — LargestMagnitude ('LM').** Same, for
  `which='LM'`, INCLUDING the indefinite case where the largest-magnitude subset
  contains a negative eigenvalue (`-3.0`) chosen over a smaller positive one
  (`2.0`): `eigsh(Ms,3,LM).eigenvalues == {-3.0, 4.0, 7.0}`.
- REQ-4: **eigenvector correctness (residual + orthonormality).** Returned
  `eigenvectors` satisfy `‖A·vᵢ − λᵢ·vᵢ‖ ≤ tol` for each pair and `VᵀV ≈ I`
  (orthonormal columns), verified via residual (NOT element-wise vs scipy, since
  eigenvectors are sign-/rotation-ambiguous).
- REQ-5: **missing `which` options — 'SM' and 'BE'.** scipy `which ∈
  {'LM','SM','LA','SA','BE'}`. ferrolearn has a `SmallestMagnitude` enum variant
  ('SM') but it is NOT verified against the scipy 'SM' contract for the outer
  `eigsh` use (Lanczos on the smallest-magnitude interior of an indefinite
  spectrum is the hard case ARPACK uses shift-invert for, not plain Lanczos),
  and it has NO 'BE' (BothEnds) analog at all. The 'SM'/'BE' behaviors are not
  shipped to scipy parity.
- REQ-6: **missing modes / params — sigma/M/v0/ncv/maxiter/tol/return_eigenvectors
  ABI.** scipy `eigsh` supports `sigma` shift-invert (with `OPinv`/`Minv`/`mode`),
  generalized `M`, an initial vector `v0`, `ncv`, `maxiter`, `tol`, and
  `return_eigenvectors`. ferrolearn exposes only `k`/`which`/`with_tol`/
  `with_max_iter`/`with_ncv`; there is no `sigma`/shift-invert, no generalized
  `M`, no user-supplied `v0` (the initial vector is a fixed deterministic
  `sin((i+1)/√2)` seed), and no `return_eigenvectors=False` mode. The scipy
  user-API ABI (R-DEV-2) is not mirrored.
- REQ-7: **missing companions — eigs / svds / lobpcg.**
  scipy.sparse.linalg also exposes `eigs` (non-symmetric/general eigenproblem),
  `svds` (sparse truncated SVD), and `lobpcg`. ferrolearn has no analog of any
  of them.
- REQ-8: **edge cases — k≥n / non-convergence / k=0 ABI.** scipy requires
  `k < n` and RAISES for `k >= n` (`TypeError`, redirecting to dense
  `scipy.linalg.eigh`); ferrolearn ALLOWS `k <= n` via a special `k==n`
  full-Lanczos path (an ABI divergence). scipy raises `ArpackNoConvergence`;
  ferrolearn returns `Err(String)` on non-convergence. Both reject `k==0`
  (scipy `ValueError`, ferrolearn `Err`), but with different error types
  (REQ-9). The k≥n acceptance and the error-type on the convergence/k=0 paths
  diverge from scipy.
- REQ-9: **error type — FerroError.** `eigsh`/`solve`/`solve_sparse`/`solve_impl`
  return `Result<_, String>` (e.g. `"k ({k}) must not exceed the matrix
  dimension n ({n})"`, `"Lanczos solver did not converge within {} iterations"`),
  not `ferrolearn_core::error::FerroError` (CLAUDE.md: "All public functions
  return `Result<T, FerroError>`"; R-CODE-2). The crate-wide error contract is
  not satisfied, and the error variants do not map to scipy's exception types
  (`ValueError`/`ArpackNoConvergence`) for the Python boundary (R-DEV-2).
- REQ-10: **non-test production consumer.** A non-test caller in the workspace
  (an estimator that needs a sparse symmetric eigensolve — spectral embedding /
  clustering / MDS / isomap / LLE — or the `ferrolearn-python` binding) consumes
  `sparse_eig::*` so it is part of the live translation surface.
- REQ-11: **ferray substrate (R-SUBSTRATE-1).** The Lanczos basis algebra
  (`Array2`/`Array1` dot products, Gram-Schmidt) and the tridiagonal symmetric QR
  eigensolve (`tridiag_qr_eigen`/`implicit_qr_step`/`wilkinson_shift`), plus the
  CSR matvec (`sparse_matvec` over `sprs::CsMat`), should route through
  `ferray::linalg` / ferray's sparse analog rather than the hand-rolled
  `ndarray`/`sprs` implementation, and the scipy.sparse.linalg analog ultimately
  belongs on the ferray substrate.

## Acceptance criteria

All expected values come from the live scipy/numpy oracle (R-CHAR-3), never from
ferrolearn. Run from `/tmp`.

- AC-1 (REQ-1): `python3 -c "import numpy as np, scipy.sparse as sp; from
  scipy.sparse.linalg import eigsh; A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.]));
  print(np.sort(eigsh(A,k=2,which='LA')[0]).tolist())"` → `[4.0,
  4.999999999999998]`; on the indefinite `M`, `eigsh(Ms,k=3,which='LA')[0]`
  (sorted) → `[2.0, 4.0, 7.0]`. `eigsh(A,2,LargestAlgebraic).eigenvalues`
  (sorted) matches each to abs ≤ 1e-8. In-crate `diagonal_matrix_top_k` /
  `symmetric_dense_matches_exact` / `sparse_tridiagonal` exercise LA against
  closed-form analytical eigenvalues (the critic adds the live-scipy indefinite
  cross-check).
- AC-2 (REQ-2): `eigsh(A,k=2,which='SA')[0]` → `[1.0, 1.9999999999999998]`;
  `eigsh(Ms,k=3,which='SA')[0]` (sorted) → `[-3.0, -1.0, 0.5]`.
  `eigsh(A,2,SmallestAlgebraic).eigenvalues` (sorted) matches to abs ≤ 1e-8.
  In-crate `diagonal_matrix_bottom_k` (SA on `diag(1..10)` → `[1,2,3]`).
- AC-3 (REQ-3): `eigsh(A,k=2,which='LM')[0]` → `[3.9999999999999996, 5.0]`;
  the STRICT indefinite check `eigsh(Ms,k=3,which='LM')[0]` (sorted) →
  `[-3.0, 4.0, 7.0]` (the negative `-3.0` is picked over the positive `2.0` by
  magnitude). `eigsh(Ms,3,LargestMagnitude).eigenvalues` (as a set, sorted)
  matches to abs ≤ 1e-8. (If ferrolearn's outer Lanczos misses `-3.0` for the
  indefinite LM subset — a known failure mode of plain Lanczos on interior
  magnitudes — REQ-3 flips NOT-STARTED with this exact oracle value as the pin.)
- AC-4 (REQ-4): for the indefinite `Ms`, build the dense `M` and assert, for the
  returned `EigenResult`, `‖M·vᵢ − λᵢ·vᵢ‖ ≤ 1e-8` for each column `i` and
  `‖VᵀV − I‖_max ≤ 1e-8`. Oracle for `λ`: `np.linalg.eigvalsh(M)`. In-crate
  `eigenvector_orthogonality` checks `VᵀV ≈ I` (residual check added by critic).
- AC-5 (REQ-5): `python3 -c "from scipy.sparse.linalg import eigsh; import
  numpy as np, scipy.sparse as sp; A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.]));
  print(np.sort(eigsh(A,k=2,which='SM')[0]).tolist(),
  np.sort(eigsh(A,k=4,which='BE')[0]).tolist())"` → `[1.0, 2.0]` and
  `[1.0, 2.0, 4.0, 5.0]`. ferrolearn has no 'BE' variant; the
  `WhichEigenvalues::SmallestMagnitude` variant is not verified to reproduce the
  scipy 'SM' result for the indefinite/interior case (plain Lanczos converges to
  extremal, not interior-magnitude, eigenvalues).
- AC-6 (REQ-6): `python3 -c "import inspect; from scipy.sparse.linalg import
  eigsh; print(list(inspect.signature(eigsh).parameters))"` lists
  `['A','k','M','sigma','which','v0','ncv','maxiter','tol',
  'return_eigenvectors','Minv','OPinv','mode']`; ferrolearn `LanczosSolver`
  exposes only `k`/`which`/`tol`/`max_iter`/`ncv` (and the fixed
  `sin((i+1)/√2)` `v0`). No `sigma`/`M`/`v0`/`return_eigenvectors`/`mode`.
- AC-7 (REQ-7): `python3 -c "import scipy.sparse.linalg as sl; print([s for s in
  ('eigs','eigsh','svds','lobpcg') if hasattr(sl,s)])"` →
  `['eigs','eigsh','svds','lobpcg']`. `grep -n "pub fn" ferrolearn-numerical/
  src/sparse_eig.rs` shows only `new`/`with_*`/`solve`/`solve_sparse`/`eigsh` —
  no `eigs`/`svds`/`lobpcg` analog.
- AC-8 (REQ-8): `python3 -c "import numpy as np, scipy.sparse as sp; from
  scipy.sparse.linalg import eigsh; A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.]));
  eigsh(A,k=5,which='LA')"` → raises
  `TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N`;
  `eigsh(A,k=0,which='LA')` → `ValueError: k must be greater than 0.`
  ferrolearn `eigsh(diag5, 5, LA)` SUCCEEDS (special `k==n` full-Lanczos path),
  and `eigsh(diag5, 0, LA)` returns `Err("k must be at least 1")` (a `String`,
  not a `ValueError`). The `k>=n` acceptance and the non-convergence
  `Err(String)` (vs `ArpackNoConvergence`) diverge.
- AC-9 (REQ-9): ferrolearn `eigsh(non_square, …)` / `eigsh(diag, 0, …)` /
  the non-convergence path all return `Err(String)`; no `FerroError` variant
  is used. scipy raises typed exceptions (`ValueError`, `ArpackNoConvergence`).
  `grep -n "FerroError\|Result<.*String>" ferrolearn-numerical/src/sparse_eig.rs`
  shows `Result<EigenResult, String>` throughout.
- AC-10 (REQ-10): `grep -rn "sparse_eig::\|eigsh\|WhichEigenvalues\|
  LanczosSolver\|EigenResult" --include=*.rs ferrolearn-*/src | grep -v
  'src/sparse_eig.rs'` returns NOTHING — there is no non-test production
  consumer. `lib.rs` exposes only `pub mod sparse_eig` (no re-export). The
  spectral estimators that COULD consume it (`spectral_embedding.rs`, `mds.rs`,
  `isomap.rs`, `lle.rs`) hand-roll their own dense eigensolve and do NOT call
  `sparse_eig`.
- AC-11 (REQ-11): `grep -n "ferray" ferrolearn-numerical/src/sparse_eig.rs`
  returns nothing — the Lanczos algebra is on `ndarray`/`sprs`, and the
  tridiagonal QR (`tridiag_qr_eigen`/`implicit_qr_step`) is hand-rolled instead
  of routing through `ferray::linalg`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (eigenvalue parity — LargestAlgebraic) | SHIPPED | impl `pub fn eigsh in sparse_eig.rs` (→ `LanczosSolver::solve_impl` → `build_lanczos` + `tridiag_qr_eigen` + `select_indices` with `WhichEigenvalues::LargestAlgebraic`) mirrors `scipy.sparse.linalg.eigsh(…, which='LA')`. Live oracle (R-CHAR-3, from `scipy.sparse.linalg.eigsh`, NEVER copied from ferrolearn): diagonal `eigsh(diag[1..5],k=2,'LA')[0] = [4.0, 4.999999999999998]`; indefinite `M` (`eigvalsh(M)=[-3,-1,0.5,2,4,7]`) `eigsh(Ms,k=3,'LA')[0] = [2.0, 4.0, 7.0]` — the largest-algebraic subset. ferrolearn eigenvalues (sorted) match to ≤ 1e-8. In-crate `diagonal_matrix_top_k`, `symmetric_dense_matches_exact` (5×5 Toeplitz, analytical `2+2cos(kπ/6)`), `sparse_tridiagonal` (n=20, analytical) all green. Consumer caveat: see REQ-10 — no non-test workspace consumer; this REQ ships on impl + the unique-eigenvalue oracle, but the module as a whole is gated by REQ-10/REQ-11. |
| REQ-2 (eigenvalue parity — SmallestAlgebraic) | SHIPPED | impl `pub fn eigsh in sparse_eig.rs` with `WhichEigenvalues::SmallestAlgebraic` (`select_indices` ascending). Live oracle: `eigsh(diag[1..5],k=2,'SA')[0] = [1.0, 1.9999999999999998]`; indefinite `eigsh(Ms,k=3,'SA')[0] = [-3.0, -1.0, 0.5]` (the smallest-algebraic subset, including the two negatives). ferrolearn (sorted) matches to ≤ 1e-8. In-crate `diagonal_matrix_bottom_k` (SA on `diag(1..10)` → `[1,2,3]`) green. Same consumer caveat (REQ-10). |
| REQ-3 (eigenvalue parity — LargestMagnitude) | SHIPPED | impl `pub fn eigsh in sparse_eig.rs` with `WhichEigenvalues::LargestMagnitude` (`select_indices` by `\|λ\|` descending). Live oracle: `eigsh(diag[1..5],k=2,'LM')[0] = [3.9999999999999996, 5.0]`; the STRICT indefinite check `eigsh(Ms,k=3,'LM')[0] = [-3.0, 4.0, 7.0]` — the negative `-3.0` is selected over the positive `2.0` by magnitude (`\|-3\|>\|2\|`). Because full-reorthogonalisation Lanczos on this small dense `M` converges to the WHOLE spectrum before selection, `select_indices(LargestMagnitude)` picks `{7,4,-3}` correctly; ferrolearn matches the oracle set to ≤ 1e-8. NOTE (R-HONEST-3): this is the fragile `which`; the critic must pin the live indefinite-LM oracle (`{-3,4,7}`) as a `#[test]` — if a future thick-restart change makes the outer Lanczos converge only to the algebraically-extremal subset and MISS the interior-magnitude `-3.0`, REQ-3 flips NOT-STARTED with `[-3.0, 4.0, 7.0]` as the pin. At baseline `9ed63e3e3` it matches. Same consumer caveat (REQ-10). |
| REQ-4 (eigenvector residual + orthonormality) | SHIPPED | impl returns Ritz vectors `V·y` (`solve_impl`: `result_eigenvectors.column_mut(out_col).assign(&ritz_vec)`), built from the full-reorthogonalised Lanczos basis (`build_lanczos` double-MGS) and `tridiag_qr_eigen`'s accumulated orthonormal `z`. Contract is residual, NOT element-wise vs scipy (eigenvectors are sign-/rotation-ambiguous). Oracle for `λ`: `np.linalg.eigvalsh(M)`. In-crate `eigenvector_orthogonality` asserts unit norm + pairwise orthogonality (`VᵀV ≈ I`, ε=1e-8) on `diag(1..10)`; `tridiag_qr_simple` checks `QᵀQ=I` on a 3×3. The critic should add the residual `‖M·vᵢ − λᵢ·vᵢ‖ ≤ 1e-8` pin on the indefinite `Ms`. Same consumer caveat (REQ-10). |
| REQ-5 (missing which — 'SM' / 'BE') | NOT-STARTED | open prereq blocker (to be filed by critic). Live oracle: `eigsh(diag[1..5],k=2,'SM')[0] = [1.0, 2.0]`, `eigsh(diag[1..5],k=4,'BE')[0] = [1.0, 2.0, 4.0, 5.0]`. ferrolearn `WhichEigenvalues` has `LargestAlgebraic`/`SmallestAlgebraic`/`LargestMagnitude` AND a `SmallestMagnitude` variant — but (a) there is NO 'BE' (BothEnds) variant at all, and (b) `SmallestMagnitude` is NOT verified against scipy 'SM': plain (even restarted) Lanczos converges to the EXTREMAL eigenvalues, so the smallest-MAGNITUDE (interior, for an indefinite spectrum) values require shift-invert (`sigma=0`) — exactly the mode ferrolearn lacks (REQ-6). For a positive-definite diagonal 'SM' coincides with 'SA' and may pass by accident, but the indefinite interior case is not shipped. The 'SM'/'BE' scipy contracts are not satisfied. |
| REQ-6 (missing modes/params — sigma/M/v0/ncv/maxiter/tol/return_eigenvectors ABI) | NOT-STARTED | open prereq blocker (to be filed by critic). `inspect.signature(eigsh)` → `A,k,M,sigma,which,v0,ncv,maxiter,tol,return_eigenvectors,Minv,OPinv,mode`. ferrolearn `LanczosSolver` exposes only `k`/`which`/`with_tol`/`with_max_iter`/`with_ncv`. There is NO `sigma` shift-invert (`solve_impl` only ever does the forward `matvec`, never `(A−σI)⁻¹`), NO generalized `M` (no `Mx=λMx`), NO user `v0` (the initial vector is the fixed deterministic `((i+1)·FRAC_1_SQRT_2).sin()` seed — so results are NOT sensitive to a caller seed the way scipy's `v0` is), and NO `return_eigenvectors=False` fast path. The scipy user-API ABI (R-DEV-2) is not mirrored. |
| REQ-7 (missing companions — eigs / svds / lobpcg) | NOT-STARTED | open prereq blocker (to be filed by critic). `scipy.sparse.linalg` documents `eigs` (non-symmetric/general eigenproblem), `svds` (sparse truncated SVD), `lobpcg` (`scipy/sparse/linalg/__init__.py`). `grep -n "pub fn" sparse_eig.rs` shows only the `eigsh`/`LanczosSolver` surface — NONE of `eigs`/`svds`/`lobpcg` exist anywhere in `ferrolearn-numerical`. (Of these, `svds` is the one sklearn leans on — `TruncatedSVD(algorithm='arpack')`, `randomized` aside — so it is a real downstream prerequisite.) |
| REQ-8 (edge cases — k≥n / non-convergence / k=0) | NOT-STARTED | open prereq blocker (to be filed by critic). Live oracle: `eigsh(diag5,k=5,'LA')` RAISES `TypeError("Cannot use scipy.linalg.eigh for sparse A with k >= N…")` (scipy requires `k < n`); `eigsh(diag5,k=0,'LA')` RAISES `ValueError("k must be greater than 0.")`. ferrolearn `solve_impl` ALLOWS `k==n` via a dedicated `let ncv = if k == n { n }` full-Lanczos path and returns Ok — an ABI divergence (scipy redirects the caller to dense `scipy.linalg.eigh`). On non-convergence ferrolearn returns `Err("Lanczos solver did not converge within {max_iter} iterations")`; scipy raises `ArpackNoConvergence` (carrying the partial eigenpairs). `k==0` is rejected by both but with different error types (REQ-9). The k≥n acceptance and the convergence/`k=0` error semantics diverge from scipy. |
| REQ-9 (error type — FerroError) | NOT-STARTED | open prereq blocker (to be filed by critic). `eigsh`/`solve`/`solve_sparse`/`solve_impl` all return `Result<EigenResult, String>` (e.g. `"Matrix must be square, got shape ({rows}, {cols})"`, `"k ({k}) must not exceed the matrix dimension n ({n})"`, `"Lanczos solver did not converge within {} iterations"`), not `ferrolearn_core::error::FerroError` (CLAUDE.md: "All public functions return `Result<T, FerroError>`"; R-CODE-2). The `String` variants also do not map to scipy's exception types (`ValueError` / `ArpackNoConvergence`) for the `ferrolearn-python` boundary (R-DEV-2). The crate-wide error contract is not satisfied. |
| REQ-10 (non-test production consumer) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -rn "sparse_eig::\|eigsh\|WhichEigenvalues\|LanczosSolver\|EigenResult" --include=*.rs ferrolearn-*/src \| grep -v 'src/sparse_eig.rs'` returns NOTHING; `lib.rs` exposes only `pub mod sparse_eig` (no re-export). The spectral estimators that mirror sklearn's `eigsh` users — `ferrolearn-decomp/src/spectral_embedding.rs` (sklearn `manifold/_spectral_embedding.py`), `mds.rs`, `isomap.rs`, `lle.rs` — hand-roll their OWN dense eigensolve (faer/LAPACK on the dense affinity) and do NOT call this module; `spectral_embedding.rs` contains no `sparse_eig`/`eigsh` reference. S5 grandfathering does NOT rescue this REQ: `LanczosSolver`/`eigsh` are internal substrate helpers, not a boundary estimator type, with no external users and no Python binding. With zero in-workspace consumers, the honest call (R-HONEST-3) is NOT-STARTED — the module is dead code. The fix is to wire a sparse spectral estimator (or `svds`-backed `TruncatedSVD`) through `sparse_eig::*`. |
| REQ-11 (ferray substrate) | NOT-STARTED | open prereq blocker (to be filed by critic). `grep -n "ferray" sparse_eig.rs` is empty: the module uses `ndarray::{Array1,Array2}` for the Krylov basis/Gram-Schmidt and `sprs::CsMat` for the matrix (`sparse_matvec`). Per R-SUBSTRATE-1 the dense linear algebra (basis dots, `gram_schmidt_columns`, the symmetric-tridiagonal QR `tridiag_qr_eigen`/`implicit_qr_step`/`wilkinson_shift` — the same hand-rolled eigensolver duplicated in `integrate.rs`) is a `ferray::linalg` concern, the CSR matvec is a ferray-sparse concern, and the scipy.sparse.linalg analog ultimately belongs on the ferray substrate. ferray does not yet expose a routed sparse symmetric eigensolver / sparse-matvec entry point for this use (R-SUBSTRATE-5: a ferray gap is real work, filed upstream to ferray; the ferrolearn unit is NOT-STARTED on this REQ until ferray ships it). Do NOT silently keep the hand-rolled `ndarray`/`sprs` path as the destination. |

## Architecture

`sparse_eig.rs` is a flat module: the public types `WhichEigenvalues` (4-variant
enum), `EigenResult{eigenvalues, eigenvectors}`, and the `LanczosSolver` config
struct (builder `new`/`with_which`/`with_tol`/`with_max_iter`/`with_ncv`), plus
the free `eigsh` convenience. There is no unfitted/Fitted split — these are pure
numerical routines, not estimators — and no generic `F: Float` bound: everything
is `f64`-only (a substrate detail, not a scipy divergence; `scipy.sparse.linalg.
eigsh` operates on float64 by default).

**Lanczos driver (`solve_impl`).** Validates `k` (`1 <= k <= n` — note `k==n`
is ALLOWED, REQ-8) and computes `ncv = min(n, max(2k+1, 20))` (the `with_ncv`
override mirrors scipy's `ncv`). It seeds a FIXED deterministic start vector
`v0[i] = sin((i+1)·FRAC_1_SQRT_2)` (no user `v0`, REQ-6), builds the initial
size-`ncv` Lanczos factorisation, then runs the implicit-restart loop up to
`max_iter`: eigendecompose `T` (`tridiag_qr_eigen`), select the `k` wanted Ritz
values (`select_indices`), check the Ritz residual `|β_m|·|last component|`
against `tol·max(|λ|,1)`, and on non-convergence thick-restart (keep the `k`
wanted Ritz vectors as the new basis prefix, re-`gram_schmidt_columns`, set the
restart residual, extend with `build_lanczos` from column `k`). On the
budget being exhausted it returns `Err(String)` (REQ-8/REQ-9). This is a
hand-rolled IRLM — distinct from ARPACK's exact implicit-QR shifts — so the
contract is eigenvalue value-to-tolerance (~1e-8), which holds for the extremal
`which` values (REQ-1/2/3), not the interior 'SM' (REQ-5).

**Lanczos building block (`build_lanczos`).** Three-term recurrence with FULL
reorthogonalisation: each step subtracts the previous-vector coupling, the
diagonal `α = wᵀvⱼ`, then TWO modified-Gram-Schmidt passes against all prior
basis vectors (the cost that buys the tight orthonormality REQ-4 needs). An
invariant-subspace breakdown (`β < 1e-15`) injects a fresh deterministic vector
and double-MGS-orthogonalises it.

**Tridiagonal symmetric QR (`tridiag_qr_eigen`).** The implicit-shift symmetric
QR (Golub & Van Loan 8.3.3): bottom-up deflation, Wilkinson shift from the
trailing 2×2 (`wilkinson_shift`), bulge-chase `implicit_qr_step` accumulating
Givens rotations (`apply_givens_to_tridiag`/`apply_givens_to_columns`/
`givens_rot`) into the eigenvector matrix `z`, eigenvalues returned ascending.
This is the SAME hand-rolled eigensolver pattern as `integrate.rs`'s
Golub-Welsch path — the locus of REQ-11 (a `ferray::linalg` concern).

**`which` selection (`select_indices`).** Sorts all Ritz indices by the `which`
key (LA descending value, SA ascending value, LM descending `|λ|`, SM ascending
`|λ|`) and truncates to `k`. For the small dense matrices the full spectrum is
available before selection, so LA/SA/LM are exact set-selections (REQ-1/2/3); SM
on an indefinite interior spectrum is the unverified case (REQ-5).

The module's defining structural fact is REQ-10: it has **no consumer**. It is
re-export-less (`lib.rs`: `pub mod sparse_eig` only) and grep-clean of callers —
the spectral estimators that mirror sklearn's `eigsh` users hand-roll dense
eigensolves instead. At baseline `9ed63e3e3` it is dead translation surface,
which is why the cross-cutting REQs (5 missing-which, 6 missing-modes, 7
companions, 8 edge-cases, 9 error-type, 10 consumer, 11 substrate) are
NOT-STARTED even though the four numerical REQs (1/2/3 eigenvalue parity, 4
eigenvector residual) match the live scipy oracle.

## Verification

Commands establishing the claims (run at baseline `9ed63e3e3`):

- `cargo test -p ferrolearn-numerical --lib sparse_eig` → all pass
  (`tridiag_qr_identity`, `tridiag_qr_simple`, `identity_eigenvalues`,
  `diagonal_matrix_top_k`, `diagonal_matrix_bottom_k`,
  `symmetric_dense_matches_exact`, `sparse_tridiagonal`,
  `eigenvector_orthogonality`, `matvec_closure_api`). NOTE: these in-crate tests
  use closed-form analytical eigenvalues (`2+2cos(kπ/(n+1))`,
  `-4sin²(kπ/2(n+1))`) and `VᵀV=I`; they do NOT compare against a live
  `scipy.sparse.linalg.eigsh` call — that comparison (esp. the indefinite LA/SA/LM
  cases) is the critic's REQ-1/2/3 pin below.
- eigenvalue-parity oracle (REQ-1/2/3, R-CHAR-3 — expected from
  `scipy.sparse.linalg.eigsh` + `np.linalg.eigvalsh`, NEVER from ferrolearn):
  ```
  cd /tmp && python3 -c "
  import numpy as np, scipy.sparse as sp
  from scipy.sparse.linalg import eigsh
  A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.]))
  for w in ['LA','SA','LM']: print('diag',w,np.sort(eigsh(A,k=2,which=w)[0]).tolist())
  np.random.seed(0); Q,_=np.linalg.qr(np.random.randn(6,6))
  D=np.diag([-3.,-1.,0.5,2.,4.,7.]); M=Q@D@Q.T; M=0.5*(M+M.T); Ms=sp.csr_matrix(M)
  for w in ['LA','SA','LM']: print('dense',w,np.sort(eigsh(Ms,k=3,which=w)[0]).tolist())
  print('full', np.sort(np.linalg.eigvalsh(M)).tolist())"
  ```
  → diag LA `[4.0, 4.999999999999998]`, SA `[1.0, 1.9999999999999998]`, LM
  `[3.9999999999999996, 5.0]`; dense LA `[2.0, 4.0, 7.0]`, SA `[-3.0, -1.0, 0.5]`,
  LM `[-3.0, 4.0, 7.0]`; full `[-3.0,-1.0,0.5,2.0,4.0,7.0]`. A critic pins a
  `#[test]` asserting `eigsh(…).eigenvalues` (sorted) equals each of these to
  abs ≤ 1e-8 — the indefinite-LM `[-3.0, 4.0, 7.0]` is the strict check.
- eigenvector-residual oracle (REQ-4): for the indefinite `Ms`, assert
  `‖M·vᵢ − λᵢ·vᵢ‖ ≤ 1e-8` (λ from `np.linalg.eigvalsh(M)`) and `‖VᵀV−I‖ ≤ 1e-8`.
- missing-which oracle (REQ-5):
  `python3 -c "import numpy as np, scipy.sparse as sp; from scipy.sparse.linalg
  import eigsh; A=sp.csr_matrix(np.diag([1.,2.,3.,4.,5.])); print(
  np.sort(eigsh(A,k=2,which='SM')[0]).tolist(),
  np.sort(eigsh(A,k=4,which='BE')[0]).tolist())"` → `[1.0, 2.0]` /
  `[1.0, 2.0, 4.0, 5.0]`. ferrolearn has no 'BE'; 'SM' on the interior of an
  indefinite spectrum is not shipped (needs shift-invert, REQ-6).
- params/companions surface (REQ-6/REQ-7):
  `python3 -c "import inspect, scipy.sparse.linalg as sl; print(list(
  inspect.signature(sl.eigsh).parameters)); print([s for s in
  ('eigs','eigsh','svds','lobpcg') if hasattr(sl,s)])"` →
  full param list incl. `sigma`/`M`/`v0`/`return_eigenvectors`/`mode`, and
  `['eigs','eigsh','svds','lobpcg']`. ferrolearn exposes only `k`/`which`/`tol`/
  `max_iter`/`ncv` and only `eigsh`.
- edge-case oracle (REQ-8): `eigsh(diag5,k=5,'LA')` raises `TypeError` (k>=N),
  `eigsh(diag5,k=0,'LA')` raises `ValueError`; ferrolearn `eigsh(diag5,5,LA)`
  succeeds (k==n path) and `eigsh(diag5,0,LA)` → `Err("k must be at least 1")`.
- error-type / substrate checks (REQ-9/REQ-11): `grep -n "Result<.*String>\|
  ferray" ferrolearn-numerical/src/sparse_eig.rs` → `Result<…, String>`
  throughout, no `ferray`.
- consumer check (REQ-10): `grep -rn "sparse_eig::\|eigsh\|WhichEigenvalues\|
  LanczosSolver\|EigenResult" --include=*.rs ferrolearn-*/src | grep -v
  'src/sparse_eig.rs'` → empty. Documented as the blocker; no failing `#[test]`
  (a missing-consumer fact is structural, not a numerical assertion).

Per R-DEFER-2 the table is binary SHIPPED/NOT-STARTED. SHIPPED (impl + live
scipy/numpy oracle to ≤ 1e-8): REQ-1 (LA parity, incl. indefinite), REQ-2 (SA
parity, incl. indefinite negatives), REQ-3 (LM parity, incl. the strict
indefinite `{-3,4,7}` subset), REQ-4 (eigenvector residual + `VᵀV≈I`).
NOT-STARTED (open `-l blocker` issues to be filed by the critic): REQ-5 (missing
'SM'/'BE'), REQ-6 (missing `sigma`/`M`/`v0`/`return_eigenvectors` ABI), REQ-7
(missing `eigs`/`svds`/`lobpcg`), REQ-8 (k≥n acceptance / non-convergence /
`k=0` semantics vs scipy), REQ-9 (`String` error vs `FerroError` / scipy
exception types), REQ-10 (no non-test consumer — dead module), REQ-11 (ferray
`linalg`/sparse substrate for the Lanczos algebra + tridiagonal QR + CSR matvec).
