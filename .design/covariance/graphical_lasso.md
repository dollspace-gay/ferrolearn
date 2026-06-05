# Graphical Lasso (Sparse Inverse Covariance)

<!--
tier: 3-component
status: draft
baseline-commit: 63f797ba92963e34756bf0da2219587b2d4fd84d
upstream-paths:
  - sklearn/covariance/_graph_lasso.py
-->

## Summary

`ferrolearn-covariance/src/graphical_lasso.rs` mirrors the L1-penalised
sparse-inverse-covariance estimators of `sklearn.covariance`: `GraphicalLasso`,
`GraphicalLassoCV`, and the function `graphical_lasso`
(`sklearn/covariance/_graph_lasso.py`). It implements the Friedman–Hastie–
Tibshirani (2008) block coordinate-descent outer loop with a hand-rolled inner
soft-thresholding lasso, exposes `covariance()`/`precision()`/`location()`/
`n_iter()` on `FittedGraphicalLasso`, and a sequential-fold cross-validator
`GraphicalLassoCV` selecting `alpha` by mean log-likelihood over an explicit
grid.

This is a **numerical coordinate-descent unit**: value parity (R-DEV-1) is the
contract. The headline finding is that **ferrolearn does NOT match the live
sklearn 1.5.2 oracle**. On a well-conditioned 4-feature probe (60 samples,
`alpha=0.1`, both converging in `n_iter=2`):

- `covariance_` max-abs-diff = **1.68e-2** (diagonal matches to 4e-13 — the
  `+alpha` strip works — but off-diagonals diverge),
- `precision_` max-abs-diff = **2.68e-1** (the precision diagonal is off by
  0.27),
- the **sparsity pattern matches** (same zero/non-zero entries),
- the divergence **grows with `alpha`** (at `alpha=0.2`, `precision_[0,0]` =
  0.873 ferrolearn vs 1.114 sklearn).

The root cause is three coupled deviations: (1) ferrolearn's inner
`coord_descent_lasso` penalty scaling differs from sklearn's
`cd_fast.enet_coordinate_descent_gram`; (2) ferrolearn converges on the
Frobenius change `||W − W_old||_F < tol` rather than the **duality gap**
`|_dual_gap| < tol`; (3) the init differs (`W = S + alpha·I` vs sklearn's
`covariance_ *= 0.95` off-diagonal Tikhonov shrink with the empirical diagonal
restored). So `REQ-GLASSO-VALUE`, `REQ-GLASSO-FN`, and `REQ-CONVERGENCE` are all
**NOT-STARTED** with filed blockers — this is the load-bearing divergence the
critic should pin. The covariance/precision **diagonals and shapes** are
structurally correct; only the penalised off-diagonal solution diverges.

## Upstream reference (scikit-learn 1.5.2)

`sklearn/covariance/_graph_lasso.py` (live oracle: installed sklearn 1.5.2).

**`_graphical_lasso` (`:70`) — the CD core.** Init (`:91-104`):
`covariance_ = emp_cov.copy(); covariance_ *= 0.95;
diagonal = emp_cov.flat[::n+1]; covariance_.flat[::n+1] = diagonal;
precision_ = linalg.pinvh(covariance_)` — the off-diagonals are shrunk 5%,
the diagonal is the empirical diagonal, and the precision is seeded by a
pseudo-inverse. Outer loop `for i in range(max_iter)` (`:120`); per feature
`idx` build the contiguous `sub_covariance = covariance_[indices!=idx].T[indices!=idx]`
and `row = emp_cov[idx, indices != idx]` (`:131`); for `mode=="cd"`:

```python
coefs = -(precision_[indices != idx, idx]
          / (precision_[idx, idx] + 1000 * eps))                       # :135-138
coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
    coefs, alpha, 0, sub_covariance, row, row,
    max_iter, enet_tol, check_random_state(None), False)               # :139-150
```

i.e. the inner lasso is the **Gram elastic net with `l1=alpha`, `l2=0`,
tolerance `enet_tol`** (a *separate* tolerance from the outer `tol`),
warm-started from the previous precision column. The precision update
(`:163-168`) is `precision_[idx,idx] = 1/(covariance_[idx,idx] −
covariance_[indices!=idx,idx]·coefs)`, `precision_[indices!=idx,idx] =
−precision_[idx,idx]·coefs`; then `covariance_[idx, indices!=idx] =
sub_covariance @ coefs` (`:169-171`).

**Convergence — duality gap (`:57-66`, `:176-185`).**

```python
def _dual_gap(emp_cov, precision_, alpha):
    gap  = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += alpha * (np.abs(precision_).sum() - np.abs(np.diag(precision_)).sum())
    return gap
...
d_gap = _dual_gap(emp_cov, precision_, alpha)                          # :176
if np.abs(d_gap) < tol:                                                # :184
    break
```

Returns `(covariance_, precision_, costs, i+1)` (`:200`).

**`alpha_max(emp_cov)` (`:203`)** — zero the diagonal of a copy of `emp_cov`,
return `max(abs(.))`; the smallest `alpha` that drives all off-diagonal lasso
coefficients to zero (on the probe, `alpha_max = 0.7937`).

**`graphical_lasso(emp_cov, alpha, *, mode="cd", tol=1e-4, enet_tol=1e-4,
max_iter=100, ..., return_costs=False, ..., return_n_iter=False)` (`:230`)** —
function wrapper; builds a `GraphicalLasso(...).fit(emp_cov)` (with
`covariance="precomputed"`) and returns `[covariance_, precision_]`
(`:350-357`).

**`class GraphicalLasso(BaseGraphicalLasso)` (`:391`)** —
`__init__(alpha=0.01, *, mode="cd", tol=1e-4, enet_tol=1e-4, max_iter=100,
verbose=False, assume_centered=False)`. `fit` (`:542`):
`emp_cov = empirical_covariance(X, assume_centered=...)` (`:565`),
`self.location_ = X.mean(0)` or zeros (`:566-569`), then
`self.covariance_, self.precision_, self.costs_, self.n_iter_ =
_graphical_lasso(emp_cov, alpha=self.alpha, ...)`.

**`class GraphicalLassoCV(BaseGraphicalLasso)` (`:714`)** —
`__init__(*, alphas=4, n_refinements=4, cv=None, tol=1e-4, enet_tol=1e-4,
max_iter=100, mode="cd", n_jobs=None, verbose=False, assume_centered=False)`
(`:903-933`). `fit` (`:935`): if `alphas` is an **array** ⇒ use it verbatim and
`n_refinements = 1` (`:979-990`); if `alphas` is an **int** ⇒
`alpha_1 = alpha_max(emp_cov)`, `alpha_0 = 1e-2*alpha_1`, `alphas =
logspace(log10(alpha_0), log10(alpha_1), n_alphas)[::-1]` (`:991-995`). For each
of `n_refinements` rounds (`:1003`), CV-score every alpha via
`graphical_lasso_path` on each `(train,test)` split (`cv=None ⇒ KFold(5)`,
`max_iter=int(0.1*self.max_iter)` per inner fit, `:1014-1027`), pick the alpha
maximising mean test log-likelihood (smallest alpha on ties, `:1036-1049`), then
**refine** the grid around it (`:1051-1064`). Sets `alpha_`, `cv_results_`, and
refits on full data (`covariance_`/`precision_`).

## Requirements

Each REQ tags a **MATCH/DEVIATE** rationale against the live oracle. Value REQs
are *oracle-pinnable* (deterministic, no RNG in the CD solver).

- **REQ-GLASSO-VALUE** (`GraphicalLasso.covariance_`/`precision_` value parity,
  R-DEV-1 — THE KEY REQ): `GraphicalLasso::new(alpha).fit(X)` covariance and
  precision must match `GraphicalLasso(alpha=...).fit(X)` element-wise within a
  numerical tolerance. **DEVIATE**: measured covariance max-abs-diff 1.68e-2,
  precision max-abs-diff 2.68e-1 (see Acceptance/Verification). Inner-lasso
  penalty scaling + convergence criterion + init all differ.
- **REQ-CONVERGENCE** (duality-gap vs Frobenius-W): sklearn breaks on
  `|_dual_gap(emp_cov, precision_, alpha)| < tol` (`:184`); ferrolearn on
  `||W − W_old||_F < tol` (`solve_glasso`). **DEVIATE** — and since the values
  diverge, this is the *root cause*, not a benign internal difference.
- **REQ-GLASSO-FN** (`graphical_lasso` function value parity): the
  `(cov, precision)` wrapper must match `graphical_lasso(emp_cov, alpha)`.
  **DEVIATE** (same solver as REQ-GLASSO-VALUE; ferrolearn `graphical_lasso`
  output equals `GraphicalLasso::fit` to 1e-15, both diverge from sklearn).
- **REQ-GLASSOCV-CV** (CV fold scheme + scoring + alpha selection on an explicit
  grid): ferrolearn uses sequential consecutive-row folds + `log_likelihood(
  test_emp, precision)` mean; sklearn `cv=None ⇒ KFold(5, shuffle=False)` +
  the same log-likelihood path scoring. The **selection** (argmax over an
  explicit grid) is structurally present and, on the probe, picks the same
  `alpha_ = 0.05` as sklearn — but the per-fold scores differ because they
  consume the divergent precision (REQ-GLASSO-VALUE) and use 3 folds vs sklearn
  KFold(5). **DEVIATE** (selection agrees on the probe but is not oracle-pinned
  to sklearn's `cv_results_`).
- **REQ-GLASSOCV-REFINEMENT** (adaptive alpha grid + `n_refinements`): sklearn
  generates the grid from `alpha_max` and refines it `n_refinements` times when
  `alphas` is an int (`:991-1064`); ferrolearn takes an explicit `alphas: Vec`
  and never refines. **DEVIATE/architectural** — note: when `alphas` is an
  explicit array sklearn ALSO sets `n_refinements = 1` (`:990`), so ferrolearn's
  no-refinement behaviour matches sklearn's *explicit-array* path; the gap is
  the missing int-`alphas` auto-grid (`alpha_max` + `logspace`) and refinement
  loop.
- **REQ-ENET-TOL** (separate inner-lasso tolerance): sklearn passes a distinct
  `enet_tol=1e-4` to the Gram CD (`:147`); ferrolearn reuses the outer `tol`
  (and `max_inner_iter`) for `coord_descent_lasso`. **DEVIATE** (parameter gap).
- **REQ-DEFAULTS** (`alpha=0.01`, `max_iter=100`, `mode='cd'`; CV `alphas=4`,
  `n_refinements=4`, `cv=5`): ferrolearn `GraphicalLasso::new(alpha)` *requires*
  alpha (no `0.01` default), `GraphicalLassoCV::new(alphas)` requires an
  explicit grid + defaults `n_folds=3` (sklearn KFold(5)). **DEVIATE**.
- **REQ-MODE-LARS** (`mode='lars'`): ferrolearn is CD-only; sklearn supports
  `lars_path_gram` (`:151-161`). **DEVIATE** (unimplemented).
- **REQ-REFIT-ESTIMATOR** (CV fitted attributes): ferrolearn exposes
  `best_alpha()`/`cv_scores()`; sklearn sets `alpha_` + `cv_results_` (a dict
  `alphas`/`split*_test_score`/`mean_test_score`/`std_test_score`) plus
  `covariance_`/`precision_`. **PARTIAL/DEVIATE** — `best_alpha`↔`alpha_` and
  `covariance`/`precision` are present; the structured `cv_results_` is absent.
- **REQ-X-1** (R-SUBSTRATE — ferray): `solve_glasso`/`coord_descent_lasso` run on
  `ndarray` + hand-rolled CD + Friedman inversion, not `ferray::linalg`.
  **DEVIATE** (not migrated).
- **REQ-X-2** (non-test consumer): the estimator types are re-exported from
  `lib.rs` and reachable by a production consumer. **MATCH** (SHIPPED).

## Acceptance criteria

Each AC carries a LIVE sklearn 1.5.2 oracle on a fixed probe.

**Probe data** (deterministic, saved at `/tmp/glasso_X.npy` during authoring;
regenerable):
`X = RandomState(0).multivariate_normal(zeros(4), cov_true, size=60)` with
`cov_true = [[1,.5,.2,0],[.5,1,.3,.1],[.2,.3,1,.4],[0,.1,.4,1]]`. Both solvers
converge in `n_iter = 2` at `alpha = 0.1`, so divergence is **algorithmic, not
under-iteration**.

- **AC-1 (REQ-GLASSO-VALUE):** `GraphicalLasso::new(0.1).fit(X)` covariance and
  precision match `GraphicalLasso(alpha=0.1).fit(X)` within 1e-6.
  Oracle: `m.covariance_[0] =
  [1.230591687718, 0.69368865292, 0.301931948369, 0.143553743945]`;
  `m.precision_[0] =
  [1.292753985237, -0.741742330948, -0.252779734338, 0.0]`. **CURRENTLY FAILS** —
  ferrolearn `covariance[0] = [1.2305916877, 0.6936896682, 0.3019512635,
  0.1267303777]` (cov[0,3] 0.1267 vs 0.1436), `precision[0] = [1.1019091373,
  -0.5823974309, -0.2059828939, 0.0]` (prec[0,0] 1.102 vs 1.293). cov
  max-abs-diff = 1.68e-2, precision max-abs-diff = 2.68e-1.
- **AC-2 (REQ-GLASSO-FN):** `graphical_lasso(empirical_covariance(X), 0.1, 100,
  1e-4)` matches `sklearn.covariance.graphical_lasso(emp_cov, 0.1)` within 1e-6.
  **CURRENTLY FAILS** (same solver; ferrolearn `graphical_lasso` equals
  `GraphicalLasso::fit` output bit-for-bit, both diverge from sklearn).
- **AC-3 (REQ-GLASSO-VALUE, alpha sweep):** at `alpha ∈ {0.01, 0.2, 0.5}` the
  precision diagonal must match. Oracle `precision_[0,0]`: 1.54418848 / 1.11371268
  / 0.86701586. ferrolearn: 1.51039084 / 0.87253820 / 0.59621252 — **FAILS**,
  divergence grows with alpha.
- **AC-4 (REQ-GLASSOCV-CV):** `GraphicalLassoCV::new([0.01,0.05,0.1,0.2,0.5])
  .n_folds(3).fit(X).best_alpha()` selects the alpha maximising mean CV
  log-likelihood. Oracle `GraphicalLassoCV(alphas=[...], cv=3).fit(X).alpha_ =
  0.05`; ferrolearn `best_alpha = 0.05` — **selection AGREES on this probe**
  (per-fold scores differ; ferrolearn `cv_scores =
  [-5.327, -5.303, -5.341, -5.483, -5.817]`, sklearn `mean_test_score` over a
  6-entry grid `[..., -5.382, -5.301, -5.297, -5.332, ...]`).
- **AC-5 (REQ-GLASSOCV-REFINEMENT):** `GraphicalLassoCV` with an integer
  `alphas` builds a `logspace(1e-2·alpha_max, alpha_max, n)` grid and refines it.
  Oracle `alpha_max(empirical_covariance(X)) = 0.7936880139362154`. **No
  ferrolearn surface accepts an integer `alphas`** → NOT-STARTED.
- **AC-6 (REQ-X-2):** the estimator types are constructed/re-exported by a
  non-test consumer (`lib.rs`). SATISFIED.

## REQ status table

Per R-DEFER-2 the table is binary (SHIPPED / NOT-STARTED). Per R-HONEST-2 the
value REQ is classified on the *measured* oracle diff, not optimism.

| REQ | Status | Evidence |
|---|---|---|
| REQ-GLASSO-VALUE (cov/prec parity) | NOT-STARTED | open prereq blocker #1880. `pub fn fit for GraphicalLasso in graphical_lasso.rs` → `solve_glasso(&emp_cov, alpha, max_iter, max_inner_iter, tol)`; the inner `fn coord_descent_lasso` solves `min 0.5 β'Wβ − s'β + alpha·‖β‖₁` with per-coordinate `soft_threshold(residual, alpha)/W[j,j]` — a penalty scaling that differs from sklearn `cd_fast.enet_coordinate_descent_gram(coefs, alpha, 0, sub_covariance, row, row, ...)` (`_graph_lasso.py:139-150`). Live oracle (probe, alpha=0.1, both n_iter=2): sklearn `precision_[0]=[1.2928,-0.7417,-0.2528,0]`, ferrolearn `[1.1019,-0.5824,-0.2060,0]`. **covariance_ max-abs-diff = 1.68e-2, precision_ max-abs-diff = 2.68e-1**; divergence grows with alpha (prec[0,0] 0.873 vs 1.114 at alpha=0.2). Sparsity pattern + cov diagonal (4e-13) match. Oracle-pinnable as a failing test. |
| REQ-CONVERGENCE (duality gap) | NOT-STARTED | open prereq blocker #1881. `fn solve_glasso in graphical_lasso.rs` breaks when `(Σ(w−w_old)²).sqrt() < tol` (Frobenius change of `W`); sklearn breaks when `abs(_dual_gap(emp_cov, precision_, alpha)) < tol` (`_graph_lasso.py:184`) with `_dual_gap = Σ(emp_cov·precision_) − n_features + alpha·(‖precision_‖₁ − ‖diag(precision_)‖₁)` (`:57-66`). Root-cause coupling to REQ-GLASSO-VALUE: even with a matched inner lasso, a different stopping rule yields a different fixed point at finite `tol`. |
| REQ-GLASSO-FN (function parity) | NOT-STARTED | open prereq blocker #1880. `pub fn graphical_lasso in graphical_lasso.rs` calls `solve_glasso(emp_cov, alpha, max_iter, 100, tol)` (max_inner_iter hardcoded 100) and returns `(cov, prec)`. Shares the divergent solver: ferrolearn `graphical_lasso(emp, 0.1, 100, 1e-4)` output equals `GraphicalLasso::fit` to 1e-15 and diverges from sklearn `graphical_lasso(emp, 0.1)` by the same 1.68e-2 / 2.68e-1. |
| REQ-GLASSOCV-CV (CV scoring/selection) | NOT-STARTED | open prereq blocker #1880 (coupled). `impl Fit for GraphicalLassoCV in graphical_lasso.rs`: sequential consecutive-row folds (`lo = fold*fold_size`), per-fold `GraphicalLasso::new(alpha).fit(train)` then `helpers::log_likelihood(&test_emp, fitted.precision())`, mean, argmax. Structure mirrors sklearn's KFold + log-likelihood path scoring; on the probe `best_alpha = 0.05` matches sklearn `alpha_ = 0.05`. BUT the per-fold scores consume the divergent precision (REQ-GLASSO-VALUE) and use n_folds=3 vs sklearn KFold(5), so the scores are not oracle-pinned. Classified NOT-STARTED until the underlying value parity is closed. |
| REQ-GLASSOCV-REFINEMENT (adaptive grid) | NOT-STARTED | open prereq blocker #1882. `GraphicalLassoCV<F> { alphas: Vec<F>, n_folds, ... }` accepts only an explicit grid and never refines. sklearn: int `alphas` ⇒ `alpha_max(emp_cov)` (`_graph_lasso.py:203`) + `logspace(log10(1e-2·alpha_max), log10(alpha_max), n)[::-1]` (`:993-995`) + `n_refinements` refinement rounds (`:1003,:1051-1064`). NOTE: for an *explicit-array* `alphas` sklearn also sets `n_refinements = 1` (`:990`), so the no-refinement path matches; the gap is the int-`alphas` auto-grid + refinement. No `alpha_max`/logspace in `graphical_lasso.rs`. |
| REQ-ENET-TOL (inner-lasso tol) | NOT-STARTED | open prereq blocker #1883. `fn coord_descent_lasso in graphical_lasso.rs` uses the outer `tol` and `max_inner_iter` for its `if max_change < tol break`. sklearn passes a separate `enet_tol=1e-4` (`_graph_lasso.py:147`) to `cd_fast.enet_coordinate_descent_gram`, independent of the outer `tol`. Parameter gap affecting inner-lasso accuracy. |
| REQ-DEFAULTS (constructor defaults) | NOT-STARTED | open prereq blocker #1884. `GraphicalLasso::new(alpha)` requires `alpha` (sklearn class default `alpha=0.01`, `_graph_lasso.py:404`); `GraphicalLassoCV::new(alphas)` requires an explicit grid and defaults `n_folds = 3` (sklearn `alphas=4` int auto-grid, `n_refinements=4`, `cv=None ⇒ KFold(5)`, `:903-933`). max_iter=100 + mode='cd' defaults DO match. Constructor-default gaps. |
| REQ-MODE-LARS (mode='lars') | NOT-STARTED | open prereq blocker #1885. `solve_glasso` is coordinate-descent only; sklearn's `mode=="lars"` branch uses `lars_path_gram(Xy=row, Gram=sub_covariance, ...)` (`_graph_lasso.py:151-161`). No LARS solver in `graphical_lasso.rs`. |
| REQ-REFIT-ESTIMATOR (CV attributes) | NOT-STARTED | open prereq blocker #1886. `FittedGraphicalLassoCV` exposes `best_alpha()` (↔ sklearn `alpha_`), `cv_scores()`, `covariance()`/`precision()`/`location()`/`n_iter()`. sklearn additionally sets `cv_results_` — a dict (`alphas`, `split0_test_score`…`split{k-1}_test_score`, `mean_test_score`, `std_test_score`, `_graph_lasso.py` `fit`). The structured per-split `cv_results_` is absent (ferrolearn keeps only the mean per alpha in `cv_scores`). |
| REQ-X-1 (ferray substrate) | NOT-STARTED | open prereq blocker #1887 (R-SUBSTRATE). `graphical_lasso.rs` uses `ndarray::{Array1, Array2}` and hand-rolled `fn solve_glasso`/`fn coord_descent_lasso`/`fn soft_threshold` + the Friedman block-inversion identity for the precision. Destination substrate is `ferray-core` (array) / `ferray::linalg` (Cholesky/inverse). Not migrated. |
| REQ-X-2 (non-test consumer) | SHIPPED | `lib.rs` re-exports `pub use graphical_lasso::{FittedGraphicalLasso, FittedGraphicalLassoCV, GraphicalLasso, GraphicalLassoCV, graphical_lasso}`. The unfitted estimator types ARE the public API (boundary R-DEFER-2/S5, grandfathered); `GraphicalLasso::fit` and `GraphicalLassoCV::fit` consume `helpers::empirical_covariance` (`let emp_cov = empirical_covariance(x, self.assume_centered)?`) and `helpers::log_likelihood` (non-test production helpers, themselves re-exported). No `ferrolearn-python` binding exists for the covariance crate, so the re-exported estimator types are the consumer surface. Verification: `cargo test -p ferrolearn-covariance --lib` green. |

## Architecture

ferrolearn collapses sklearn's `BaseGraphicalLasso → GraphicalLasso /
GraphicalLassoCV` hierarchy into two unfitted builder structs.
`pub struct GraphicalLasso<F> { alpha, max_iter (100), max_inner_iter (100),
tol (1e-4), assume_centered }` with chained setters; `impl Fit<Array2<F>, ()>`
computes `empirical_covariance(x, assume_centered)`, calls
`solve_glasso(&emp_cov, alpha, max_iter, max_inner_iter, tol)`, and returns
`FittedGraphicalLasso<F> { covariance, precision, location, n_iter }`. The
`location` is the per-column mean (or zeros under `assume_centered`).

**The CD solver (`fn solve_glasso`).** Init `W = emp_cov.clone()` with
`W[i,i] += alpha` (`graphical_lasso.rs` solve_glasso, the Friedman
`W = S + alpha·I` stabiliser) — **this differs from sklearn's
`covariance_ *= 0.95` off-diagonal Tikhonov shrink + restored empirical
diagonal + `pinvh` precision seed** (`_graph_lasso.py:101-104`). Outer loop: for
each column `j` extract `W11` (W minus row/col j) and `s12 = emp_cov[:,j]`
(minus diag), `beta = coord_descent_lasso(&w11, &s12, alpha, max_inner_iter,
tol)`, write back `w_12 = W11·beta`. Convergence is `||W − W_old||_F < tol`
(Frobenius change) — **sklearn uses `|_dual_gap| < tol`**. After the loop the
precision is rebuilt column-wise via the Friedman inversion identity
(`theta_jj = 1/(W[j,j] − w_12·beta_j)`, `theta_12 = −beta_j·theta_jj`), then the
`+alpha` diagonal shift is stripped from `W` (#343) so `covariance_`'s diagonal
equals the empirical diagonal exactly — which is why the **cov diagonal matches
the oracle to 4e-13** while the off-diagonals (driven by the divergent inner
lasso) and the entire precision do not.

**The inner lasso (`fn coord_descent_lasso`).** Cyclic coordinate descent on
`min 0.5 β'Wβ − s'β + alpha·‖β‖₁`: per coordinate `j`,
`residual = s[j] − Σ_{k≠j} W[j,k]·β[k]`, `β[j] = soft_threshold(residual,
alpha) / W[j,j]`, break on `max_change < tol`. sklearn's
`enet_coordinate_descent_gram` solves the analogous Gram problem but with the
elastic-net parameterisation `(alpha=l1, beta=l2=0)` and a different residual/
penalty normalisation; this scaling mismatch is the primary source of the
precision divergence.

**`GraphicalLassoCV`.** `pub struct GraphicalLassoCV<F> { alphas: Vec<F>,
n_folds (3), max_iter, tol, assume_centered }`. `fit` does sequential
consecutive-row K-fold (NOT shuffled — matches sklearn KFold default
`shuffle=False`): per alpha, per fold, refit `GraphicalLasso` on the train block,
score the held-out block with `helpers::log_likelihood(&test_emp,
fitted.precision())`, average, then argmax over the explicit grid and refit on
full data. There is **no `alpha_max` grid generation and no refinement loop**
(REQ-GLASSOCV-REFINEMENT); this matches sklearn's *explicit-array* path
(`n_refinements = 1`) but not its int-`alphas` auto-grid.

Invariants: `emp_cov` is square (else `ShapeMismatch` in `graphical_lasso`);
`n ≥ 2`, `p ≥ 2` (else `InsufficientSamples`); the recovered `precision` shares
the lasso sparsity pattern (off-diagonal zeros where `beta` soft-thresholds to
zero) — verified to match sklearn's zero pattern on the probe.

## Verification

Commands establishing the (one) SHIPPED claim and quantifying the divergences,
run at baseline `63f797ba`:

- `cargo test -p ferrolearn-covariance --lib` → green (the in-file structural
  tests `test_graphical_lasso_basic`/`_function`/`_cv`/`_too_small` assert
  shapes and `best_alpha ∈ grid`; none pin oracle values — they are
  shape-only). Establishes REQ-X-2.
- **Live sklearn 1.5.2 oracle** (probe `X = RandomState(0).multivariate_normal(
  zeros(4), [[1,.5,.2,0],[.5,1,.3,.1],[.2,.3,1,.4],[0,.1,.4,1]], 60)`,
  `alpha=0.1`, both `n_iter=2`):
  - sklearn `GraphicalLasso(alpha=0.1).fit(X).precision_[0] =
    [1.292753985237, -0.741742330948, -0.252779734338, 0.0]`; ferrolearn
    `GraphicalLasso::new(0.1).fit(X).precision()[0] = [1.1019091373,
    -0.5823974309, -0.2059828939, -0.0]`. **precision_ max-abs-diff = 2.68e-1**.
  - sklearn `.covariance_[0] = [1.230591687718, 0.69368865292, 0.301931948369,
    0.143553743945]`; ferrolearn `[1.2305916877, 0.6936896682, 0.3019512635,
    0.1267303777]`. **covariance_ max-abs-diff = 1.68e-2** (diagonal diff
    4.15e-13, off-diagonal `[0,3]` 0.1267 vs 0.1436).
  - Alpha sweep `precision_[0,0]` sklearn / ferrolearn:
    `0.01 → 1.54418848 / 1.51039084`, `0.2 → 1.11371268 / 0.87253820`,
    `0.5 → 0.86701586 / 0.59621252` — divergence grows with alpha.
  - `alpha_max(empirical_covariance(X)) = 0.7936880139362154` (for the
    REQ-GLASSOCV-REFINEMENT grid).
  - `GraphicalLassoCV(alphas=[0.01,0.05,0.1,0.2,0.5], cv=3).fit(X).alpha_ =
    0.05`; ferrolearn `GraphicalLassoCV::new([...]).n_folds(3).fit(X)
    .best_alpha() = 0.05` — selection agrees (scores not pinned).

The above ferrolearn numbers were produced by a throwaway integration test
(`GraphicalLasso::new(0.1).fit(X)` on the probe via the public API) that was
**removed after measurement** — no `.rs` file is modified by this doc.

A critic should pin **REQ-GLASSO-VALUE** (#1880) and **REQ-GLASSO-FN** (#1880) as
failing `#[test]`s comparing `covariance()`/`precision()` to the sklearn oracle
values above (expected values copied from sklearn, never from ferrolearn,
R-CHAR-3). **REQ-CONVERGENCE** (#1881) is the root-cause fix: replace the
Frobenius-W break with the `_dual_gap` criterion, adopt the `covariance_*=0.95`
init, and align the inner lasso with `enet_coordinate_descent_gram` (the latter
is REQ-ENET-TOL #1883). Once those land the value REQs flip to SHIPPED-with-
tolerance and REQ-GLASSOCV-CV becomes oracle-pinnable.

Per R-DEFER-2 the table is binary. **SHIPPED: 1** (REQ-X-2). **NOT-STARTED: 10**
(REQ-GLASSO-VALUE #1880, REQ-CONVERGENCE #1881, REQ-GLASSO-FN #1880,
REQ-GLASSOCV-CV #1880, REQ-GLASSOCV-REFINEMENT #1882, REQ-ENET-TOL #1883,
REQ-DEFAULTS #1884, REQ-MODE-LARS #1885, REQ-REFIT-ESTIMATOR #1886, REQ-X-1
#1887). The load-bearing divergence is the **CD solver value parity**: the
covariance diagonal and sparsity pattern are correct, but the penalised
off-diagonal covariance (≤1.7e-2) and the precision matrix (≤2.7e-1) do not
match sklearn and the gap grows with `alpha`.
