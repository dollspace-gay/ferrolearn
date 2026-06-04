# MiniBatchNMF (sklearn.decomposition.MiniBatchNMF)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 02e0f811
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_nmf.py  # class MiniBatchNMF(_BaseNMF) (:1792-2454). ctor (:2007-2045): n_components="warn", *, init=None, batch_size=1024, beta_loss="frobenius", tol=1e-4, max_no_improvement=10, max_iter=200, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, forget_factor=0.7, fresh_restarts=False, fresh_restarts_max_iter=30, transform_max_iter=None, random_state=None, verbose=0. _check_params (:2047-2071): _batch_size = min(batch_size, n_samples) (:2050); _rho = forget_factor ** (_batch_size / n_samples) (:2054); _gamma per beta_loss (:2057-2062); _transform_max_iter defaults to max_iter (:2065-2069). _solve_W (:2073-2098) = transform/fresh-restart heart: W init = np.full((n_samples, n_components), sqrt(X.mean()/n_components)) (:2079-2080), then _multiplicative_update_w for max_iter iters (:2086-2096) with relative-norm tol stop (:2090-2093). _minibatch_step (:2100-2147): MU for W (or _solve_W if fresh_restarts :2117), then H via _multiplicative_update_h(..., A=_components_numerator, B=_components_denominator, rho=_rho) (:2130-2141) — the online EWA aggregates. _minibatch_convergence (:2149-2208): EWA cost _ewa_cost (:2167-2172), H-change early stop (:2182-2186), max_no_improvement heuristic (:2190-2204). _fit_transform (:2254-2349): _check_w_h NNDSVDa-default init (:2308), _components_numerator/_denominator (:2312-2313), gen_batches+itertools.cycle (:2319-2320), n_steps = max_iter * ceil(n_samples/_batch_size) (:2321-2322), ConvergenceWarning at max_iter (:2334-2341). transform (:2351-2371): W = self._solve_W(X, self.components_, self._transform_max_iter) (:2369) — deterministic given fitted H. partial_fit (:2373+). _BaseNMF (:1139): _parameter_constraints (:1147), _compute_regularization (:1275). _initialize_nmf (:225-374): NNDSVD/nndsvda/nndsvdar (:330-372). _multiplicative_update_w (:530), _multiplicative_update_h (:638), _beta_divergence (:89).
ferrolearn-module: ferrolearn-decomp/src/minibatch_nmf.rs
parity-ops: MiniBatchNMF
crosslink-issue: 1485
-->

## Summary

`ferrolearn-decomp/src/minibatch_nmf.rs` mirrors scikit-learn's `MiniBatchNMF`
(`sklearn/decomposition/_nmf.py`, `class MiniBatchNMF(_BaseNMF)` `:1792`): factor a
non-negative matrix `X ≈ W·H` into non-negative `W` (transformed data,
`n_samples × n_components`) and `H` (components, `n_components × n_features`),
processing the data in mini-batches for scalability. The exposed surface is the
unfitted `MiniBatchNMF<F> { n_components, max_iter (200), batch_size (1024), tol
(1e-4), random_state, init: MiniBatchNMFInit{Random|Nndsvd} }` (`minibatch_nmf.rs`,
struct at line 61; builders `with_max_iter`/`with_batch_size`/`with_tol`/
`with_random_state`/`with_init`, accessors) and the fitted `FittedMiniBatchNMF<F> {
components_ (n_components, n_features), reconstruction_err_, n_iter_ }`
(`minibatch_nmf.rs`, struct at line 170, accessors `components`/`reconstruction_err`/
`n_iter`), re-exported at the crate root (`pub use minibatch_nmf::{FittedMiniBatchNMF,
MiniBatchNMF, MiniBatchNMFInit}`, `lib.rs:96`). There is NO PyO3 binding (a
`grep -rn MiniBatchNMF ferrolearn-python/src/` is empty) and NO
`tests/divergence_minibatch_nmf.rs`.

**EXACT `components_` / `transform` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4
NOT-STARTED, CARVE-OUT `#1486`).** sklearn's `MiniBatchNMF._fit_transform`
(`_nmf.py:2254-2349`) drives a MULTIPLICATIVE-UPDATE (MU) online solver: NNDSVDa
default init (`_check_w_h` → `_initialize_nmf`, `_nmf.py:2308`/`:225`), `_minibatch_step`
(`:2100`) updates `W` by `_multiplicative_update_w` (`:2118-2120`) and `H` by
`_multiplicative_update_h(..., A=_components_numerator, B=_components_denominator,
rho=_rho)` (`:2130-2141`) — an Exponentially-Weighted-Average (EWA) of the H numerator/
denominator aggregates with the `_rho = forget_factor ** (_batch_size/n_samples)`
forget factor (`:2054`) — over `gen_batches` cycled by `itertools.cycle` with a
numpy `RandomState`-seeded init. ferrolearn's `fn fit` (`minibatch_nmf.rs`, impl at
line 365) instead uses a DIFFERENT algorithm: per-batch `W` solved by a 5-iteration
COORDINATE DESCENT (`update_w_batch`, `minibatch_nmf.rs` fn at line 315) and a PLAIN
MU for `H` `H *= (Wᵀ X_batch)/(Wᵀ W H + eps)` (`minibatch_nmf.rs:458-472`) with NO EWA
aggregates `A`/`B`, NO `rho` forget factor; batches are produced by a deterministic
`indices.rotate_left` (`minibatch_nmf.rs:423`) with NO `random_state` shuffle; init is
a Rust `StdRng` uniform (`init_random`, fn at line 210, default seed 42) or a
power-iteration pseudo-NNDSVD (`init_nndsvd_simple`, fn at line 233 — NOT sklearn's
NNDSVDa). Different algorithm + different RNG + no forget_factor/EWA ⇒ the
`components_` and `transform` VALUES diverge (same class as the cluster / sparse_pca
RNG carve-outs); no failing test is asserted (R-DEFER-3).

**`transform` = `_solve_W` MU FOLDS INTO THE VALUE CARVE-OUT (REQ-5, NOT-STARTED,
`#1487`).** sklearn's `transform` (`_nmf.py:2351-2371`) is DETERMINISTIC given the
fitted `H`: `W = self._solve_W(X, self.components_, self._transform_max_iter)`
(`:2369`), where `_solve_W` (`:2073-2098`) inits `W = np.full((n_samples,
n_components), sqrt(X.mean()/n_components))` (`:2079-2080`) and runs
`_multiplicative_update_w` for `_transform_max_iter` (= `max_iter`, default 200)
iterations with a relative-norm tol stop (`:2090-2093`). ferrolearn's `transform`
(`impl Transform for FittedMiniBatchNMF`, fn at line 507) instead runs a 5-iteration
coordinate-descent `update_w_batch` (`minibatch_nmf.rs:524`) from a CONSTANT `0.1`
init (`minibatch_nmf.rs:519-522`). **Critic verdict (live oracle, R-CHAR-3): NOT a
meaningfully observable divergence.** Both target the same convex NNLS optimum
`min_{W≥0}||X − W·H||²` for the fixed fitted `H`; on ferrolearn's own `H` the
residuals match — `||X − W_ferro·H|| = 4.598322` vs sklearn `_solve_W` `4.598258`
(relative ~1.4e-5), ferro's even below `non_negative_factorization` `4.598462`. The
~0.022 elementwise `W` gap is a flat-valley artifact (near-zero `H` entry; 2000 extra
MU steps move the objective only ~2.8e-5). No failing test is pinned; REQ-5 folds
into the REQ-4 value carve-out, and the green-guard
`div5_transform_residual_matches_solve_w_optimum` records the residual equivalence.

As of this iteration: the STRUCTURAL non-negativity of `components_` / `W`, the
shape `(n_components, n_features)`, the error & parameter contracts (n_components
0/>n_features, negative input, empty data, transform col-mismatch), determinism given
seed, and the finite `reconstruction_err_` (REQ-1,2,3) are SHIPPED scoped; exact
`components_` value parity (REQ-4, CARVE-OUT `#1486`), the `transform`/`_solve_W` MU
formula (REQ-5, CARVE-OUT `#1487`), `beta_loss` + `_gamma` (REQ-6, `#1488`),
`solver`/MU-for-W (REQ-7, `#1489`), `forget_factor`/`_rho` + EWA aggregates
(REQ-8, `#1490`), `_minibatch_convergence` EWA cost + `max_no_improvement` (REQ-9,
`#1491`), real NNDSVDa init + `random_state` shuffle (REQ-10, `#1492`),
regularization `alpha_W`/`alpha_H`/`l1_ratio` (REQ-11, `#1493`),
`fresh_restarts`/`fresh_restarts_max_iter` (REQ-12, `#1494`), `partial_fit`
(REQ-13, `#1495`), fitted attrs `n_components_`/`n_features_in_`/`n_steps_`
(REQ-14, `#1496`), the PyO3 binding (REQ-15, `#1497`), and the ferray substrate
(REQ-16, `#1498`) are NOT-STARTED — **3 SHIPPED / 13 NOT-STARTED**.

`MiniBatchNMF` / `FittedMiniBatchNMF` are existing pub APIs whose non-test consumer
is the crate re-export (`lib.rs:96`, boundary public API, grandfathered
S5/R-DEFER-1). There is NO PyO3 binding (REQ-15 NOT-STARTED).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2 SHIPPED scoped + REQ-4/14 NOT-STARTED) — components_ shape
# (n_components, n_features) + non-negativity; fitted attrs n_components_/
# n_features_in_/n_steps_. Small fixed non-negative X (5x3). VALUES generated by
# sklearn, never copied from ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import MiniBatchNMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[2,1,0],[0,3,4]])
m=MiniBatchNMF(n_components=2, random_state=0).fit(X)
print('components_ shape:', m.components_.shape)
print('components_ non-negative:', bool((m.components_>=0).all()))
print('components_ row0:', np.round(m.components_[0],6).tolist())
print('n_components_:', m.n_components_, 'n_features_in_:', m.n_features_in_, 'n_steps_:', m.n_steps_)
print('reconstruction_err_:', round(float(m.reconstruction_err_),6))"
# -> components_ shape: (2, 3)
# -> components_ non-negative: True             => structural non-negativity (REQ-2)
# -> components_ row0: [3.476503, 2.115604, 1.748483]   => VALUES (REQ-4 CARVE-OUT, NOT reproduced)
# -> n_components_: 2 n_features_in_: 3 n_steps_: 200    => fitted attrs (REQ-14 NOT-STARTED)
# -> reconstruction_err_: 0.97347

# PROBE 2 (REQ-5 CARVE-OUT #1487 — transform converges to NNLS optimum) — transform = _solve_W is DETERMINISTIC
# given the fitted H, non-negative, shape (n_samples, n_components).
python3 -c "
import numpy as np
from sklearn.decomposition import MiniBatchNMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[2,1,0],[0,3,4]])
m=MiniBatchNMF(n_components=2, random_state=0).fit(X)
W1=m.transform(X); W2=m.transform(X)
print('transform shape:', W1.shape)
print('transform deterministic:', bool(np.allclose(W1,W2)))
print('transform W row0:', np.round(W1[0],6).tolist())
print('transform non-negative:', bool((W1>=0).all()))"
# -> transform shape: (5, 2)
# -> transform deterministic: True          => _solve_W is deterministic given H (REQ-5)
# -> transform W row0: [0.248862, 0.493079] => MU-from-sqrt(X.mean()/k) values (NOT ferrolearn's 5-iter CD from 0.1)
# -> transform non-negative: True

# PROBE 3 (REQ-3/6..16 NOT-STARTED) — ctor defaults.
python3 -c "
from sklearn.decomposition import MiniBatchNMF
m=MiniBatchNMF()
for p in ['n_components','init','batch_size','beta_loss','tol','max_no_improvement','max_iter','alpha_W','alpha_H','l1_ratio','forget_factor','fresh_restarts','fresh_restarts_max_iter','transform_max_iter','random_state']:
    print(f'{p} =', getattr(m,p))"
# -> n_components = warn  init = None  batch_size = 1024  beta_loss = frobenius
# -> tol = 0.0001  max_no_improvement = 10  max_iter = 200  alpha_W = 0.0
# -> alpha_H = same  l1_ratio = 0.0  forget_factor = 0.7  fresh_restarts = False
# -> fresh_restarts_max_iter = 30  transform_max_iter = None  random_state = None
#    => ferrolearn has n_components/max_iter/batch_size/tol/random_state/init only;
#       NO beta_loss/alpha_*/l1_ratio/forget_factor/fresh_restarts/max_no_improvement/transform_max_iter.
```

## Requirements

- REQ-1: **Structural: `components_` (H) shape `(n_components, n_features)`,
  finite `reconstruction_err_`, positive `n_iter_`, determinism given seed
  (SHIPPED scoped).** `fn fit` (`minibatch_nmf.rs`, impl at line 365) stores
  `components_ = h` of shape `(n_components, n_features)` (`minibatch_nmf.rs:488` /
  struct field `:172`, = sklearn `components_` `H` shape `_nmf.py:2349`), the
  Frobenius `reconstruction_err_ = ||X − W·H||_F` (`reconstruction_error`, fn at line
  303; stored `:489`), and `n_iter_` (`:490`). The seeded `StdRng` (`init_random`,
  fn at line 210, seed `self.random_state.unwrap_or(42)` `:402`) plus the
  deterministic `indices.rotate_left` batching (`:423`) make the fit reproducible
  given a seed. **Scope: STRUCTURAL (shape / finiteness / determinism), NOT
  component VALUES (REQ-4).** Non-test consumer: re-export `lib.rs:96`.

- REQ-2: **Structural: `components_` (H) and `transform` (W) are NON-NEGATIVE; the
  fit factors a non-negative `X` (SHIPPED scoped).** The H multiplicative update
  clamps negatives to `eps` (`minibatch_nmf.rs:468-470`) and the CD `update_w_batch`
  clamps each `W` entry to `≥ 0` (`minibatch_nmf.rs:340-344`), mirroring NMF's
  non-negativity constraint (sklearn objective `_nmf.py:1804`, `W,H` non-negative).
  Probe 1 confirms sklearn `components_` are non-negative; Probe 2 confirms
  `transform` `W` is non-negative. Pinned by
  `test_minibatch_nmf_components_nonnegative`. **Scope: STRUCTURAL non-negativity,
  NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:96`.

- REQ-3: **Error / parameter contracts (SHIPPED scoped).** `fn fit`
  (`minibatch_nmf.rs`, impl at line 365) returns `InvalidParameter { name:
  "n_components" }` for `n_components == 0` (`minibatch_nmf.rs:368-373`) and for
  `n_components > n_features` (`:374-382`), `InsufficientSamples { required: 1 }`
  for `0` samples (`:383-389`), and `InvalidParameter { name: "X" }` on any negative
  input entry (`:392-399`); `transform` returns `ShapeMismatch` on a column-count
  mismatch (`:509-515`). Pinned by `test_minibatch_nmf_zero_components_error`,
  `test_minibatch_nmf_too_many_components_error`, `test_minibatch_nmf_empty_data`,
  `test_minibatch_nmf_negative_input_error`,
  `test_minibatch_nmf_transform_shape_mismatch`. **FLAG (candidate DIVs):** sklearn
  validates via `_parameter_constraints` (`_nmf.py:1147`) and `check_non_negative`
  (`_fit_transform` `:2294`) raising `InvalidParameterError`/`ValueError`, NOT
  `FerroError`; sklearn accepts `n_components=None` (→ default) which ferrolearn
  requires as an explicit `usize`; sklearn does not pre-reject `n_components >
  n_features` (MU surfaces it later). Non-test consumer: re-export `lib.rs:96`.

- REQ-4: **EXACT `components_` value parity with sklearn's online MU solver
  (NOT-STARTED, CARVE-OUT; `#1486`).** sklearn's `_fit_transform`
  (`_nmf.py:2254-2349`) inits via NNDSVDa (`_check_w_h` → `_initialize_nmf`,
  `:2308`/`:225`), then over `gen_batches`/`itertools.cycle` (`:2319-2320`) runs
  `_minibatch_step` (`:2100`): MU for `W` (`_multiplicative_update_w` `:2118-2120`)
  and MU for `H` with the EWA aggregates `A=_components_numerator`,
  `B=_components_denominator`, `rho=_rho` (`_multiplicative_update_h` `:2130-2141`),
  numpy `RandomState`-seeded. ferrolearn's `fn fit` (`minibatch_nmf.rs`, impl at line
  365) uses a DIFFERENT algorithm — 5-iter CD for `W` (`update_w_batch`, fn at line
  315) + plain MU for `H` with NO EWA / NO `rho` (`:458-472`), deterministic
  `rotate_left` batching (`:423`), Rust `StdRng` init (seed 42). Probe 1 sklearn
  `components_ row0 = [3.476503, 2.115604, 1.748483]` is NOT reproduced element-wise.
  **CARVE-OUT (R-DEFER-3):** matching sklearn requires reimplementing the MU `W`/`H`
  updates + EWA aggregates + `rho` forget factor + NNDSVDa + numpy RNG; no failing
  test is asserted (same class as the cluster / sparse_pca RNG carve-outs).

- REQ-5: **`transform` = `_solve_W` MU formula (NOT-STARTED, CARVE-OUT — folds into
  REQ-4; `#1487`).** sklearn's `transform` (`_nmf.py:2351-2371`) is DETERMINISTIC
  given the fitted `H`: `W = self._solve_W(X, self.components_,
  self._transform_max_iter)` (`:2369`); `_solve_W` (`:2073-2098`) inits `W =
  np.full((n_samples, n_components), sqrt(X.mean()/n_components))` (`:2079-2080`)
  and runs `_multiplicative_update_w` for `_transform_max_iter` (= `max_iter`,
  default 200) iters with a relative-norm tol stop (`:2090-2093`). ferrolearn's
  `transform` (`impl Transform for FittedMiniBatchNMF`, fn at line 507) runs a
  5-iter CD `update_w_batch` (`minibatch_nmf.rs:524`) from a CONSTANT `0.1` init
  (`:519-522`) — different init/solver/iter-count. **Critic verdict (live oracle):
  NOT a meaningfully observable divergence** — both reach the same convex NNLS
  optimum for the fixed `H` (residuals match to relative ~1.4e-5 on ferrolearn's own
  `H`; the ~0.022 elementwise gap is a flat-valley artifact). No failing test is
  pinned; this folds into REQ-4's value carve-out. The green-guard
  `div5_transform_residual_matches_solve_w_optimum` records the residual equivalence
  (Probe 2 sklearn `transform W row0 = [0.248862, 0.493079]` is the
  deterministic-given-H oracle).

- REQ-6: **`beta_loss` (`frobenius`/`kullback-leibler`/`itakura-saito`) + `_gamma`
  (NOT-STARTED; `#1488`).** sklearn's `MiniBatchNMF(beta_loss="frobenius")`
  (`_nmf.py:2011`, `StrOptions({"frobenius","kullback-leibler","itakura-saito"})`
  + numeric `_nmf.py:1157`) selects the beta-divergence loss, and `_check_params`
  sets `_gamma` for the Maximization-Minimization step per `_beta_loss`
  (`_nmf.py:2057-2062`); `_beta_divergence` (`_nmf.py:89`) and the MU helpers branch
  on it. ferrolearn's `MiniBatchNMF<F>` (`minibatch_nmf.rs`, struct at line 61) has
  NO `beta_loss` field — only the Frobenius-squared `reconstruction_error` (fn at
  line 303) and a hard-coded Frobenius MU; no `_gamma`, no KL/IS path.

- REQ-7: **`solver` / multiplicative-update for W (NOT-STARTED; `#1489`).**
  sklearn's `MiniBatchNMF` is MU-only (it has no `solver` param; W is updated by
  `_multiplicative_update_w` `_nmf.py:530`, called from `_minibatch_step` `:2118`
  and `_solve_W` `:2086`). ferrolearn updates `W` by a 5-iteration COORDINATE
  DESCENT (`update_w_batch`, fn at line 315; `for _cd_iter in 0..5`
  `minibatch_nmf.rs:324`), NOT the sklearn MU `W *= numerator/denominator`. The W
  solver itself is a divergence from sklearn's MU.

- REQ-8: **`forget_factor` / `_rho` + online EWA aggregates `A`/`B`
  (NOT-STARTED; `#1490`).** sklearn's `MiniBatchNMF(forget_factor=0.7)`
  (`_nmf.py:2018`) sets `_rho = forget_factor ** (_batch_size / n_samples)`
  (`_nmf.py:2054`), and `_minibatch_step` passes the running aggregates
  `A=self._components_numerator`, `B=self._components_denominator`, `rho=self._rho`
  to `_multiplicative_update_h` (`_nmf.py:2130-2141`) — the EWA over mini-batches that
  makes the H update ONLINE (`_components_numerator`/`_denominator` initialised
  `_nmf.py:2312-2313`). ferrolearn's H update is a PLAIN per-batch MU
  `H *= (Wᵀ X_batch)/(Wᵀ W H + eps)` (`minibatch_nmf.rs:458-472`) with NO `A`/`B`
  aggregates and NO `rho` field — each batch overwrites `H` directly.

- REQ-9: **`_minibatch_convergence` EWA cost + `max_no_improvement` early stop
  (NOT-STARTED; `#1491`).** sklearn's `_minibatch_convergence`
  (`_nmf.py:2149-2208`) tracks an Exponentially-Weighted-Average `_ewa_cost`
  (`:2167-2172`), early-stops on relative H-change `≤ tol` (`:2182-2186`), and
  early-stops via the `max_no_improvement` (default 10, `_nmf.py:2015`) heuristic on
  `_ewa_cost` lack-of-improvement (`:2190-2204`). ferrolearn's convergence check is a
  relative-RECONSTRUCTION-error stop `|prev_err − err|/prev_err < tol`
  (`minibatch_nmf.rs:479-482`) computed over the WHOLE `X` each outer iteration — no
  EWA cost, no `max_no_improvement` field, no per-step H-change stop.

- REQ-10: **Real NNDSVDa init + `random_state` shuffle (NOT-STARTED; `#1492`).**
  sklearn defaults `init=None` → NNDSVDa (`_initialize_nmf`, `_nmf.py:225`/`:362`,
  SVD-based with zeros filled by the average of `X`) and shuffles batches via a numpy
  `RandomState` (`_fit_transform` `gen_batches`/`itertools.cycle` `:2319-2320`,
  `check_random_state`). ferrolearn's `init_nndsvd_simple` (fn at line 233) is a
  power-iteration pseudo-NNDSVD (`xtx = XᵀX`, 20-step power iteration
  `minibatch_nmf.rs:268-276`, clamp negatives), NOT the sklearn SVD-based NNDSVDa,
  and `init_random` (fn at line 210) is a Rust `StdRng` uniform; batching is the
  deterministic `indices.rotate_left` (`minibatch_nmf.rs:423`) with NO random shuffle.

- REQ-11: **Regularization `alpha_W` / `alpha_H` / `l1_ratio`
  (NOT-STARTED; `#1493`).** sklearn's `MiniBatchNMF(alpha_W=0.0, alpha_H="same",
  l1_ratio=0.0)` (`_nmf.py:2013-2014`,`:2016`) adds L1/L2 penalties via
  `_compute_regularization` (`_nmf.py:1275`), threaded into every `_minibatch_step`
  (`l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H` `_nmf.py:2106`) and `_solve_W`
  (`:2082`). ferrolearn's `MiniBatchNMF<F>` (`minibatch_nmf.rs`, struct at line 61)
  has NO `alpha_W`/`alpha_H`/`l1_ratio` fields and applies NO regularization — the MU
  for H and CD for W are unpenalised.

- REQ-12: **`fresh_restarts` / `fresh_restarts_max_iter` (NOT-STARTED; `#1494`).**
  sklearn's `MiniBatchNMF(fresh_restarts=False, fresh_restarts_max_iter=30)`
  (`_nmf.py:2019-2020`) optionally re-solves `W` from scratch each mini-batch via
  `_solve_W(X, H, self.fresh_restarts_max_iter)` (`_nmf.py:2117`) and a final
  `_solve_W` at end-of-fit (`:2335`). ferrolearn has NO `fresh_restarts` field — `W`
  is always warm-continued from the previous batch (`update_w_batch` mutates the
  carried `w` slice, `minibatch_nmf.rs:440-456`).

- REQ-13: **`partial_fit` (online out-of-core fit) (NOT-STARTED; `#1495`).**
  sklearn's `MiniBatchNMF.partial_fit(X, y=None, W=None, H=None)`
  (`_nmf.py:2373+`) updates the model incrementally on consecutive chunks for
  out-of-core learning. ferrolearn exposes only the batch `Fit::fit`
  (`minibatch_nmf.rs`, impl at line 354) — no `partial_fit` method, no incremental
  state carried across calls.

- REQ-14: **Fitted attrs `n_components_` / `n_features_in_` / `n_steps_`
  (NOT-STARTED; `#1496`).** sklearn exposes `n_components_`, `n_features_in_`, and
  `n_steps_` (the number of mini-batches processed; Probe 1 `n_steps_ = 200`).
  `FittedMiniBatchNMF<F>` (`minibatch_nmf.rs`, struct at line 170) exposes only
  `components()` / `reconstruction_err()` / `n_iter()` (fns at lines 182/188/194) —
  no `n_components_` (derivable from `components_.nrows()` but not exposed), no
  `n_features_in_`, no `n_steps_` (only outer-iteration `n_iter_`, not the mini-batch
  step count).

- REQ-15: **PyO3 binding (NOT-STARTED; `#1497`).** sklearn exposes `MiniBatchNMF`
  through `import sklearn.decomposition`. ferrolearn has NO PyO3 binding for
  `MiniBatchNMF` — a `grep -rn MiniBatchNMF ferrolearn-python/src/` is empty; the
  only non-test consumer of `MiniBatchNMF`/`FittedMiniBatchNMF` is the crate
  re-export (`lib.rs:96`). The CPython surface (a `_RsMiniBatchNMF` class with a
  ctor + `fit` + `transform`) is absent.

- REQ-16: **ferray substrate (NOT-STARTED; `#1498`).** `minibatch_nmf.rs` computes on
  `ndarray::Array2` (`minibatch_nmf.rs:33`) and uses `rand`/`rand_distr`
  (`minibatch_nmf.rs:35-36`, `StdRng` + `Uniform`) for init, not `ferray-core`
  arrays / `ferray::random` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `MiniBatchNMF::new(3).with_random_state(0).fit(&X)
  .unwrap().components().dim()` is `(3, n_features)`; `reconstruction_err()` is
  finite and `≥ 0`; `n_iter()` is `> 0`; two fits with the same seed are identical.
  Pinned by `test_minibatch_nmf_components_shape` `(3,4)`,
  `test_minibatch_nmf_reconstruction_err_positive`,
  `test_minibatch_nmf_n_iter_positive`, `test_minibatch_nmf_basic`. (Structural
  shape / finiteness / determinism only — NOT the exact component values, REQ-4.)

- AC-2 (REQ-2, SHIPPED scoped): every entry of `fitted.components()` and of
  `fitted.transform(&X)` is `≥ 0`. Probe 1 / Probe 2 confirm sklearn `components_` /
  `W` are non-negative. Pinned by `test_minibatch_nmf_components_nonnegative`.
  (Structural non-negativity only — NOT value parity, REQ-4.)

- AC-3 (REQ-3, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_components > n_features`, `n_samples = 0`, and any negative input;
  `transform` returns `Err` for a column-count mismatch. Pinned by
  `test_minibatch_nmf_zero_components_error`,
  `test_minibatch_nmf_too_many_components_error`, `test_minibatch_nmf_empty_data`,
  `test_minibatch_nmf_negative_input_error`,
  `test_minibatch_nmf_transform_shape_mismatch`. FLAG: sklearn raises
  `InvalidParameterError`/`ValueError` (not `FerroError`), accepts
  `n_components=None`, and does not pre-reject `n_components > n_features`.

- AC-4 (REQ-4, NOT-STARTED, CARVE-OUT): `MiniBatchNMF(n_components=2,
  random_state=0).fit(X).components_` (Probe 1: shape `(2,3)`, `row0 =
  [3.476503, 2.115604, 1.748483]`) is NOT reproduced element-wise by ferrolearn
  (different CD-vs-MU + EWA/rho + numpy RNG + NNDSVDa init). No failing test asserts
  this (R-DEFER-3).

- AC-5 (REQ-5, NOT-STARTED, CARVE-OUT — folds into REQ-4): sklearn's `transform`
  computes `_solve_W(X, H, transform_max_iter)` — `W = np.full((n,k),
  sqrt(X.mean()/k))` then `_multiplicative_update_w` to `transform_max_iter` (Probe
  2: deterministic given H). ferrolearn runs 5-iter CD from `0.1`
  (`minibatch_nmf.rs:519-524`). **Critic verdict (live oracle): no observable
  divergence** — both reach the same convex NNLS optimum for the fixed `H` (residuals
  match to relative ~1.4e-5 on ferrolearn's own `H`), so no failing test is pinned;
  REQ-5 folds into the REQ-4 value carve-out.

- AC-6 (REQ-6..14, DIVERGES): `MiniBatchNMF()` defaults `n_components="warn",
  init=None, batch_size=1024, beta_loss="frobenius", tol=1e-4,
  max_no_improvement=10, max_iter=200, alpha_W=0.0, alpha_H="same", l1_ratio=0.0,
  forget_factor=0.7, fresh_restarts=False, fresh_restarts_max_iter=30,
  transform_max_iter=None, random_state=None` (Probe 3, `_nmf.py:2007-2045`);
  sklearn exposes `partial_fit`, `n_components_`/`n_features_in_`/`n_steps_`, the EWA
  `_minibatch_convergence`, NNDSVDa init, and regularization. ferrolearn has no
  `beta_loss`/`alpha_*`/`l1_ratio`/`forget_factor`/`fresh_restarts`/
  `max_no_improvement`/`transform_max_iter` params, no `partial_fit`, no
  `n_components_`/`n_features_in_`/`n_steps_` attrs, no EWA convergence, no real
  NNDSVDa, and no random-state batch shuffle.

- AC-7 (REQ-15/16): `import ferrolearn` exposes NO `_RsMiniBatchNMF`
  (`grep -rn MiniBatchNMF ferrolearn-python/src/` is empty); the only non-test
  consumer is the crate re-export (`lib.rs:96`). The module imports `ndarray`
  (`minibatch_nmf.rs:33`) + `rand`/`rand_distr` (`:35-36`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `MiniBatchNMF` / `FittedMiniBatchNMF` are existing pub APIs; the
non-test consumer is the crate re-export (`lib.rs:96`, boundary public API,
grandfathered S5/R-DEFER-1) — there is NO PyO3 binding (REQ-15 NOT-STARTED). Cites
use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle =
installed sklearn 1.5.2, run from `/tmp`.
**EXACT `components_`/`transform` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4
NOT-STARTED, CARVE-OUT `#1486`):** ferrolearn's 5-iter CD `W` + plain MU `H` (no
EWA / no `rho`), deterministic `rotate_left` batching, Rust `StdRng` init
(`minibatch_nmf.rs:365`) ≠ sklearn's online MU `W`/`H` with EWA aggregates +
`forget_factor` + NNDSVDa + numpy RNG (`_nmf.py:2254-2349`).
**`transform` = `_solve_W` MU FOLDS INTO THE VALUE CARVE-OUT (REQ-5
NOT-STARTED, `#1487`):** sklearn `transform` (`_nmf.py:2369`) is the deterministic
`_solve_W` (MU from `sqrt(X.mean()/k)` init to `transform_max_iter`); ferrolearn runs
5-iter CD from `0.1` (`minibatch_nmf.rs:519-524`). The critic confirmed via the live
oracle that ferrolearn's transform reaches the SAME convex NNLS optimum (residual
match relative ~1.4e-5 on its own fitted `H`), so REQ-5 is not an observable
divergence — no failing test, it folds into REQ-4's value carve-out.
The least-confident SHIPPED claim is REQ-1 — it is STRUCTURAL (shape / finiteness /
determinism), explicitly NOT the component VALUES (REQ-4); the in-tree tests assert
shapes / finiteness / positivity, not oracle parity. #1485 is this doc's crosslink
tracking issue. Count: **3 SHIPPED (REQ-1,2,3) / 13 NOT-STARTED
(REQ-4,5,6,7,8,9,10,11,12,13,14,15,16)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: shape / finite err / n_iter / determinism) | SHIPPED | `fn fit` (`minibatch_nmf.rs` impl at line 365) stores `components_ = h` shape `(n_components, n_features)` (`minibatch_nmf.rs:488`, field `:172` = sklearn `H` `_nmf.py:2349`), Frobenius `reconstruction_err_` (`reconstruction_error` fn at line 303; `:489`), `n_iter_` (`:490`). Seeded `StdRng` (`init_random` fn at line 210, seed `unwrap_or(42)` `:402`) + deterministic `indices.rotate_left` (`:423`) ⇒ reproducible. **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:96`. Verification: `cargo test -p ferrolearn-decomp minibatch_nmf` → `test_minibatch_nmf_components_shape` `(3,4)`, `test_minibatch_nmf_reconstruction_err_positive`, `test_minibatch_nmf_n_iter_positive`, `test_minibatch_nmf_basic`, `test_minibatch_nmf_small_batch` PASS. |
| REQ-2 (structural: non-negativity of components_ + W) | SHIPPED | H MU clamps negatives to `eps` (`minibatch_nmf.rs:468-470`); `update_w_batch` (fn at line 315) clamps each `W` entry to `≥ 0` (`minibatch_nmf.rs:340-344`) — NMF non-negativity (sklearn objective `_nmf.py:1804`). Probe 1 sklearn `components_` non-negative; Probe 2 sklearn `transform W` non-negative. **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:96`. Verification: `cargo test -p ferrolearn-decomp minibatch_nmf` → `test_minibatch_nmf_components_nonnegative`, `test_minibatch_nmf_f32` PASS. |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`minibatch_nmf.rs` impl at line 365) returns `Err(InvalidParameter{name:"n_components", reason:"must be at least 1"})` for `n_components==0` (`:368-373`), `Err(InvalidParameter{name:"n_components", ... "exceeds n_features"})` for `>n_features` (`:374-382`), `Err(InsufficientSamples{required:1,...})` for `0` samples (`:383-389`), `Err(InvalidParameter{name:"X", ... "non-negative input"})` on a negative entry (`:392-399`); `transform` returns `Err(ShapeMismatch)` on column mismatch (`:509-515`). Non-test consumer: re-export `lib.rs:96`. Verification: `cargo test -p ferrolearn-decomp minibatch_nmf` (`test_minibatch_nmf_zero_components_error`, `_too_many_components_error`, `_empty_data`, `_negative_input_error`, `_transform_shape_mismatch`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_nmf.py:1147`) + `check_non_negative` (`_nmf.py:2294`) raising `InvalidParameterError`/`ValueError` (not `FerroError`); accepts `n_components=None`; does NOT pre-reject `n_components > n_features`. |
| REQ-4 (EXACT `components_` value parity with online MU) | NOT-STARTED | open prereq blocker **#1486** (CARVE-OUT, R-DEFER-3). sklearn `_fit_transform` (`_nmf.py:2254-2349`): NNDSVDa init (`:2308`/`:225`), `gen_batches`/`itertools.cycle` (`:2319-2320`), `_minibatch_step` (`:2100`) — MU for `W` (`_multiplicative_update_w` `:2118-2120`) + MU for `H` with EWA `A=_components_numerator`, `B=_components_denominator`, `rho=_rho` (`:2130-2141`), numpy `RandomState`. ferrolearn `fn fit` (`minibatch_nmf.rs` impl at line 365): DIFFERENT — 5-iter CD `W` (`update_w_batch` fn at line 315) + plain MU `H` NO EWA / NO `rho` (`:458-472`), `rotate_left` batching (`:423`), Rust `StdRng` (seed 42). Probe 1 sklearn `components_ row0 = [3.476503, 2.115604, 1.748483]` NOT reproduced. No failing test (same class as cluster / sparse_pca RNG carve-outs). |
| REQ-5 (`transform` = `_solve_W` MU formula) | NOT-STARTED | open prereq blocker **#1487** (CARVE-OUT — folds into REQ-4, divergence UNOBSERVABLE). sklearn `transform` (`_nmf.py:2351-2371`): `W = self._solve_W(X, self.components_, self._transform_max_iter)` (`:2369`); `_solve_W` (`:2073-2098`) inits `W = np.full((n,k), sqrt(X.mean()/k))` (`:2079-2080`) then `_multiplicative_update_w` to `_transform_max_iter` (= `max_iter`, default 200) with relative-norm tol stop (`:2090-2093`). ferrolearn `transform` (`impl Transform for FittedMiniBatchNMF` fn at line 507) runs 5-iter CD `update_w_batch` (`:524`) from CONSTANT `0.1` init (`:519-522`) — different init/solver/iter-count. **Critic verdict (live oracle, R-CHAR-3): NOT a meaningfully observable divergence.** Both target the same convex NNLS optimum `min_{W≥0}||X − W·H||²` for the FIXED fitted H; on ferrolearn's own H the residuals match — `||X − W_ferro·H|| = 4.598322` vs sklearn `_solve_W` `4.598258` (relative ~1.4e-5), ferro's even marginally below `non_negative_factorization` `4.598462`. The ~0.022 elementwise `W` gap is a flat-valley artifact (near-zero H entry; 2000 extra MU steps move the objective only ~2.8e-5). So no failing test is pinned; this folds into the REQ-4 value carve-out. Green-guard `div5_transform_residual_matches_solve_w_optimum` records the residual equivalence. |
| REQ-6 (`beta_loss` kl/is + `_gamma`) | NOT-STARTED | open prereq blocker **#1488**. sklearn `MiniBatchNMF(beta_loss="frobenius")` (`_nmf.py:2011`, `StrOptions({"frobenius","kullback-leibler","itakura-saito"})` `_nmf.py:1157`) + `_check_params` `_gamma` per `_beta_loss` (`_nmf.py:2057-2062`); `_beta_divergence` (`_nmf.py:89`) branches on it. ferrolearn `MiniBatchNMF<F>` (`minibatch_nmf.rs` struct at line 61) has NO `beta_loss` field — only Frobenius-squared `reconstruction_error` (fn at line 303) + hard-coded Frobenius MU; no `_gamma`, no KL/IS path. |
| REQ-7 (`solver` / MU-for-W) | NOT-STARTED | open prereq blocker **#1489**. sklearn `MiniBatchNMF` updates `W` by MU `_multiplicative_update_w` (`_nmf.py:530`), called from `_minibatch_step` (`:2118`) and `_solve_W` (`:2086`). ferrolearn updates `W` by 5-iter COORDINATE DESCENT (`update_w_batch` fn at line 315; `for _cd_iter in 0..5` `minibatch_nmf.rs:324`), NOT the sklearn MU `W *= numerator/denominator`. |
| REQ-8 (`forget_factor`/`_rho` + EWA aggregates A/B) | NOT-STARTED | open prereq blocker **#1490**. sklearn `MiniBatchNMF(forget_factor=0.7)` (`_nmf.py:2018`) → `_rho = forget_factor ** (_batch_size/n_samples)` (`_nmf.py:2054`); `_minibatch_step` passes `A=_components_numerator`, `B=_components_denominator`, `rho=_rho` to `_multiplicative_update_h` (`_nmf.py:2130-2141`) — the ONLINE EWA (`_components_numerator`/`_denominator` init `:2312-2313`). ferrolearn's H update is PLAIN per-batch MU `H *= (Wᵀ X_batch)/(Wᵀ W H + eps)` (`minibatch_nmf.rs:458-472`), NO `A`/`B`, NO `rho` field. |
| REQ-9 (`_minibatch_convergence` EWA cost + `max_no_improvement`) | NOT-STARTED | open prereq blocker **#1491**. sklearn `_minibatch_convergence` (`_nmf.py:2149-2208`): EWA `_ewa_cost` (`:2167-2172`), H-change stop `≤ tol` (`:2182-2186`), `max_no_improvement` (default 10, `_nmf.py:2015`) heuristic (`:2190-2204`). ferrolearn uses a relative-RECONSTRUCTION-error stop `|prev_err−err|/prev_err < tol` over the WHOLE `X` (`minibatch_nmf.rs:479-482`) — no EWA cost, no `max_no_improvement` field, no per-step H-change stop. |
| REQ-10 (real NNDSVDa init + `random_state` shuffle) | NOT-STARTED | open prereq blocker **#1492**. sklearn defaults `init=None` → NNDSVDa (`_initialize_nmf` `_nmf.py:225`/`:362`, SVD-based, zeros filled by avg of X) + numpy `RandomState` batch shuffle (`_fit_transform` `gen_batches`/`itertools.cycle` `:2319-2320`). ferrolearn `init_nndsvd_simple` (fn at line 233) is a power-iteration pseudo-NNDSVD (`xtx=XᵀX`, 20-step power iter `minibatch_nmf.rs:268-276`), NOT SVD-based NNDSVDa; `init_random` (fn at line 210) is Rust `StdRng` uniform; batching is deterministic `indices.rotate_left` (`:423`), NO shuffle. |
| REQ-11 (`alpha_W`/`alpha_H`/`l1_ratio` regularization) | NOT-STARTED | open prereq blocker **#1493**. sklearn `MiniBatchNMF(alpha_W=0.0, alpha_H="same", l1_ratio=0.0)` (`_nmf.py:2013-2014`,`:2016`) → `_compute_regularization` (`_nmf.py:1275`) threaded into every `_minibatch_step` (`:2106`) + `_solve_W` (`:2082`). ferrolearn `MiniBatchNMF<F>` (`minibatch_nmf.rs` struct at line 61) has NO `alpha_W`/`alpha_H`/`l1_ratio` fields — H MU and W CD are unpenalised. |
| REQ-12 (`fresh_restarts`/`fresh_restarts_max_iter`) | NOT-STARTED | open prereq blocker **#1494**. sklearn `MiniBatchNMF(fresh_restarts=False, fresh_restarts_max_iter=30)` (`_nmf.py:2019-2020`) re-solves `W` from scratch per-batch via `_solve_W(X, H, fresh_restarts_max_iter)` (`_nmf.py:2117`) + final `_solve_W` (`:2335`). ferrolearn has NO `fresh_restarts` field — `W` is warm-continued from the previous batch (`minibatch_nmf.rs:440-456`). |
| REQ-13 (`partial_fit` online fit) | NOT-STARTED | open prereq blocker **#1495**. sklearn `MiniBatchNMF.partial_fit(X, y=None, W=None, H=None)` (`_nmf.py:2373+`) updates the model incrementally on consecutive chunks. ferrolearn exposes only batch `Fit::fit` (`minibatch_nmf.rs` impl at line 354) — no `partial_fit`, no incremental state. |
| REQ-14 (fitted attrs `n_components_`/`n_features_in_`/`n_steps_`) | NOT-STARTED | open prereq blocker **#1496**. sklearn exposes `n_components_`, `n_features_in_`, `n_steps_` (Probe 1 `n_steps_ = 200`). `FittedMiniBatchNMF<F>` (`minibatch_nmf.rs` struct at line 170) exposes only `components()`/`reconstruction_err()`/`n_iter()` (fns at lines 182/188/194) — no `n_components_`, no `n_features_in_`, no `n_steps_` (only outer-iter `n_iter_`, not the mini-batch step count). |
| REQ-15 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1497**. sklearn exposes `MiniBatchNMF` via `import sklearn.decomposition`. ferrolearn has NO PyO3 binding — `grep -rn MiniBatchNMF ferrolearn-python/src/` is empty; the only non-test consumer of `MiniBatchNMF`/`FittedMiniBatchNMF` is the crate re-export (`lib.rs:96`). No `_RsMiniBatchNMF` class. |
| REQ-16 (ferray substrate) | NOT-STARTED | open prereq blocker **#1498**. `minibatch_nmf.rs` computes on `ndarray::Array2` (`minibatch_nmf.rs:33`) and uses `rand`/`rand_distr` `StdRng`+`Uniform` (`minibatch_nmf.rs:35-36`) for init, not `ferray-core` arrays / `ferray::random` (R-SUBSTRATE-1/2). |

## Architecture

`minibatch_nmf.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`MiniBatchNMF<F> { n_components, max_iter (200), batch_size (1024), tol (1e-4),
random_state, init: MiniBatchNMFInit{Random|Nndsvd} }` (struct at line 61;
`new(n_components)` fn at line 83, builders `with_max_iter`/`with_batch_size`/
`with_tol`/`with_random_state`/`with_init` fns at lines 97-128, accessors
`n_components()`/`max_iter()`/`batch_size()`/`tol()`/`init()` fns at lines 132-158) →
`Fit<Array2<F>, ()>` → `FittedMiniBatchNMF<F> { components_, reconstruction_err_,
n_iter_ }` (struct at line 170, accessors `components()`/`reconstruction_err()`/
`n_iter()` fns at lines 182/188/194). The path is generic over `F: Float + Send +
Sync + 'static` (both f32 and f64, `test_minibatch_nmf_f32`); `fit`/`transform`
return `Result<_, FerroError>` (R-CODE-2).

**Fit path (`fn fit`, impl at line 365) — REQ-1/2/3/4.** Validates `n_components !=
0`, `n_components <= n_features`, `n_samples >= 1`, and non-negativity of `X`
(`minibatch_nmf.rs:368-399`) — REQ-3. Initialises `(W, H)` via `init_random` (fn at
line 210, Rust `StdRng` uniform, seed `unwrap_or(42)`) or `init_nndsvd_simple` (fn at
line 233, power-iteration pseudo-NNDSVD) per `init` (`:406-409`) — REQ-10
NOT-STARTED. Outer loop to `max_iter` (`:419`): (a) `indices.rotate_left(batch_size %
n_samples)` — DETERMINISTIC batching, NO shuffle (`:423`); (b) for each batch, extract
`X_batch`/`W_batch`, solve `W_batch` by 5-iter coordinate descent (`update_w_batch`,
fn at line 315 — REQ-7 NOT-STARTED), write `W_batch` back, then update `H` by PLAIN MU
`H *= (W_batchᵀ X_batch)/(W_batchᵀ W_batch H + eps)` (`:458-472` — NO EWA / NO `rho`,
REQ-8 NOT-STARTED); (c) break on relative reconstruction-error change `< tol`
(`:479-482` — NOT sklearn's EWA `_minibatch_convergence`, REQ-9 NOT-STARTED). Stores
`components_ = h`, `reconstruction_err_`, `n_iter_` (`:487-491`). **This is NOT
sklearn's online MU solver (REQ-4):** sklearn runs NNDSVDa-init MU `W`/`H` with EWA
aggregates + `forget_factor` over `itertools.cycle`d batches with numpy RNG
(`_nmf.py:2254-2349`); ferrolearn's CD-`W` + plain-MU-`H` + deterministic batching +
Rust RNG produce DIFFERENT component values (CARVE-OUT).

**Transform (`impl Transform for FittedMiniBatchNMF`, fn at line 507) — REQ-5
(CARVE-OUT, folds into REQ-4).** Validates the column count (`:509-515` — REQ-3),
inits `W` with the CONSTANT `0.1` (`:519-522`), and solves `W` by a 5-iter coordinate
descent `update_w_batch` (`:524`). This is NOT sklearn's `_solve_W`
(`_nmf.py:2073-2098`): sklearn inits `W = np.full((n,k), sqrt(X.mean()/k))` and runs
MU for `_transform_max_iter` (= `max_iter`, default 200) iters with a relative-norm
tol stop. Both target the same convex NNLS optimum, and the **critic verdict (live
oracle) is that the divergence is NOT meaningfully observable** — on ferrolearn's own
fitted `H` the residuals match to relative ~1.4e-5; the ~0.022 elementwise gap is a
flat-valley artifact. So no failing test is pinned and REQ-5 folds into REQ-4's value
carve-out.

**sklearn (target contract).** `class MiniBatchNMF(_BaseNMF)` (`_nmf.py:1792`) takes
`__init__(n_components="warn", *, init=None, batch_size=1024, beta_loss="frobenius",
tol=1e-4, max_no_improvement=10, max_iter=200, alpha_W=0.0, alpha_H="same",
l1_ratio=0.0, forget_factor=0.7, fresh_restarts=False, fresh_restarts_max_iter=30,
transform_max_iter=None, random_state=None, verbose=0)` (`:2007-2045`).
`_check_params` (`:2047-2071`) sets `_batch_size = min(batch_size, n_samples)`,
`_rho = forget_factor ** (_batch_size/n_samples)` (`:2054`), `_gamma` per
`beta_loss` (`:2057-2062`), `_transform_max_iter` defaulting to `max_iter`
(`:2065-2069`). `_fit_transform` (`:2254`) NNDSVDa-inits, cycles `gen_batches`, and
runs `_minibatch_step` (`:2100`: MU `W`/`H` + EWA aggregates) gated by
`_minibatch_convergence` (`:2149`: EWA cost + `max_no_improvement`). `transform`
(`:2351`) is the deterministic `_solve_W` (MU from `sqrt(X.mean()/k)`).
`partial_fit` (`:2373`) is the online variant. Fitted attrs: `components_`,
`reconstruction_err_`, `n_components_`, `n_features_in_`, `n_steps_`, `n_iter_`.

**The remaining gap.** ferrolearn ships the STRUCTURAL shape / finiteness /
determinism (REQ-1), non-negativity (REQ-2), and the scoped error & parameter
contracts (REQ-3). It lacks: exact `components_` value parity (REQ-4, CARVE-OUT
`#1486`); the `transform`/`_solve_W` MU formula (REQ-5, CARVE-OUT `#1487`);
`beta_loss` + `_gamma` (REQ-6, `#1488`); `solver`/MU-for-W (REQ-7, `#1489`);
`forget_factor`/`_rho` + EWA aggregates (REQ-8, `#1490`); `_minibatch_convergence`
EWA cost + `max_no_improvement` (REQ-9, `#1491`); real NNDSVDa + `random_state`
shuffle (REQ-10, `#1492`); `alpha_W`/`alpha_H`/`l1_ratio` (REQ-11, `#1493`);
`fresh_restarts`/`fresh_restarts_max_iter` (REQ-12, `#1494`); `partial_fit`
(REQ-13, `#1495`); `n_components_`/`n_features_in_`/`n_steps_` attrs (REQ-14,
`#1496`); the PyO3 binding (REQ-15, `#1497`); and the ferray substrate (REQ-16,
`#1498`). This is a **structure-SHIPPED-algorithm-NOT-STARTED** unit (3 SHIPPED / 13
NOT-STARTED).

## Verification

Library crate (green at baseline `02e0f811`):
```bash
cargo test -p ferrolearn-decomp minibatch_nmf               # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3 (STRUCTURAL): `test_minibatch_nmf_basic`
`(3,2)`, `test_minibatch_nmf_components_shape` `(3,4)`,
`test_minibatch_nmf_nndsvd_init`, `test_minibatch_nmf_reconstruction_err_positive`,
`test_minibatch_nmf_n_iter_positive`, `test_minibatch_nmf_small_batch`,
`test_minibatch_nmf_f32`, `test_minibatch_nmf_builder_methods` (REQ-1);
`test_minibatch_nmf_components_nonnegative` (REQ-2);
`test_minibatch_nmf_zero_components_error`,
`test_minibatch_nmf_too_many_components_error`, `test_minibatch_nmf_empty_data`,
`test_minibatch_nmf_negative_input_error`,
`test_minibatch_nmf_transform_shape_mismatch` (REQ-3); plus the module doctest.
`tests/divergence_minibatch_nmf.rs` now holds 6 green-guards (shape, non-negativity,
reconstruction/n_iter, determinism, error contracts, and
`div5_transform_residual_matches_solve_w_optimum`). REQ-4 (components value parity)
and REQ-5 (`transform` = `_solve_W` MU) are both CARVE-OUTs (R-DEFER-3, no failing
test) — the critic confirmed via the live oracle that the transform reaches the same
NNLS optimum as sklearn's `_solve_W` (residual match relative ~1.4e-5), so REQ-5 is
not a standalone observable divergence.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-4 components value
gap and the REQ-5 `transform`/`_solve_W` divergence:
```bash
# REQ-1/2 structural + REQ-4 value gap (sklearn components_ non-negative; values NOT reproduced):
python3 -c "import numpy as np; from sklearn.decomposition import MiniBatchNMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[2,1,0],[0,3,4]])
m=MiniBatchNMF(n_components=2, random_state=0).fit(X)
print(m.components_.shape, bool((m.components_>=0).all()), np.round(m.components_[0],6).tolist())"
# -> (2, 3) True [3.476503, 2.115604, 1.748483]

# REQ-5 transform = _solve_W (deterministic given fitted H):
python3 -c "import numpy as np; from sklearn.decomposition import MiniBatchNMF
X=np.array([[1.,2,3],[4,5,6],[7,8,9],[2,1,0],[0,3,4]])
m=MiniBatchNMF(n_components=2, random_state=0).fit(X)
print(m.transform(X).shape, np.round(m.transform(X)[0],6).tolist())"
# -> (5, 2) [0.248862, 0.493079]  (sklearn _solve_W MU; ferrolearn 5-iter CD from 0.1 differs)
```
REQ-5's oracle is `_solve_W(X, H, transform_max_iter)` — `W = np.full((n,k),
sqrt(X.mean()/k))` then `_multiplicative_update_w` to `transform_max_iter` — on
ferrolearn's OWN fitted `H` (R-CHAR-3). The critic confirmed ferrolearn's `transform`
already reaches the same NNLS optimum (residual match relative ~1.4e-5), so REQ-5 is a
CARVE-OUT with no parity test, like REQ-4.

ferrolearn-python (REQ-15, ABSENT at baseline): there is NO `_RsMiniBatchNMF`
binding — `grep -rn MiniBatchNMF ferrolearn-python/src/` is empty. The only non-test
consumer of `MiniBatchNMF`/`FittedMiniBatchNMF` is the crate re-export (`lib.rs:96`).

## Blockers

(#1485 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1486** — REQ-4 (CARVE-OUT): reimplement sklearn's online MU solver — MU `W`
  (`_multiplicative_update_w` `_nmf.py:530`), MU `H` with EWA aggregates
  `A`/`B` + `rho` forget factor (`_multiplicative_update_h` `_nmf.py:638`,
  `_minibatch_step` `:2130-2141`), NNDSVDa init (`_nmf.py:225`), and numpy
  `RandomState` batch shuffle — to reach EXACT `components_` value parity; inherently
  RNG/algorithm-bound (no failing test, R-DEFER-3).
- **#1487** — REQ-5 (CARVE-OUT, folds into #1486): align the `transform` solver with
  sklearn's `_solve_W` (`_nmf.py:2073-2098`): `W = np.full((n,k), sqrt(X.mean()/k))`
  then `_multiplicative_update_w` for `transform_max_iter` iters with the
  relative-norm tol stop. The critic confirmed (live oracle) ferrolearn's existing
  5-iter CD already reaches the same NNLS optimum (residual match relative ~1.4e-5),
  so this is cosmetic/value-carve-out work, not an observable parity fix — no failing
  test (R-DEFER-3).
- **#1488** — REQ-6: add a `beta_loss` field (`frobenius`/`kullback-leibler`/
  `itakura-saito`, `_nmf.py:2011`) + `_gamma` (`_nmf.py:2057-2062`) and the
  beta-divergence MU branches (`_beta_divergence` `_nmf.py:89`).
- **#1489** — REQ-7: switch the `W` update from 5-iter coordinate descent
  (`update_w_batch`) to sklearn's multiplicative update `_multiplicative_update_w`
  (`_nmf.py:530`, `_minibatch_step` `:2118`).
- **#1490** — REQ-8: add a `forget_factor` field + `_rho = forget_factor **
  (_batch_size/n_samples)` (`_nmf.py:2054`) and the online EWA H aggregates
  `A=_components_numerator` / `B=_components_denominator` (`_nmf.py:2130-2141`).
- **#1491** — REQ-9: replace the whole-`X` reconstruction-error stop with sklearn's
  `_minibatch_convergence` (`_nmf.py:2149-2208`): EWA `_ewa_cost`, per-step H-change
  stop, and a `max_no_improvement` field + heuristic.
- **#1492** — REQ-10: replace `init_nndsvd_simple` (power iteration) with real
  SVD-based NNDSVDa (`_initialize_nmf` `_nmf.py:225`/`:362`) and add a numpy-`RandomState`
  batch shuffle (`gen_batches`/`itertools.cycle` `_nmf.py:2319-2320`).
- **#1493** — REQ-11: add `alpha_W`/`alpha_H`/`l1_ratio` fields (`_nmf.py:2013-2016`)
  and the `_compute_regularization` (`_nmf.py:1275`) penalties in the W/H updates.
- **#1494** — REQ-12: add `fresh_restarts`/`fresh_restarts_max_iter` fields
  (`_nmf.py:2019-2020`) and the per-batch + end-of-fit `_solve_W` re-solve
  (`_nmf.py:2117`/`:2335`).
- **#1495** — REQ-13: add a `partial_fit` method (`_nmf.py:2373+`) carrying
  incremental online state across calls.
- **#1496** — REQ-14: expose `n_components_` / `n_features_in_` / `n_steps_` fitted
  attrs on `FittedMiniBatchNMF` (Probe 1: `n_steps_ = max_iter * ceil(n_samples /
  batch_size)`, `_nmf.py:2321-2322`/`:2347`).
- **#1497** — REQ-15: add a `_RsMiniBatchNMF` PyO3 binding in `ferrolearn-python`
  (ctor + `fit` + `transform`), registered in `lib.rs`, as the CPython consumer.
- **#1498** — REQ-16: migrate `minibatch_nmf.rs` off `ndarray` + `rand`/`rand_distr`
  to `ferray-core` arrays / `ferray::random` (R-SUBSTRATE).
