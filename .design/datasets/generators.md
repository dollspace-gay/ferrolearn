# Synthetic dataset generators (`make_*`)

<!--
tier: 3-component
status: draft
baseline-commit: 309f1ab184dff2addfde320fbd5ef84969002105
upstream-paths:
  - sklearn/datasets/_samples_generator.py
-->

## Summary

`ferrolearn-datasets/src/generators.rs` mirrors scikit-learn's synthetic data
generators in `sklearn/datasets/_samples_generator.py`: `make_classification`,
`make_regression`, `make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`,
`make_s_curve`, `make_sparse_uncorrelated`, `make_friedman1/2/3`,
`make_low_rank_matrix`, `make_spd_matrix`, `make_sparse_spd_matrix`,
`make_gaussian_quantiles`, `make_hastie_10_2`, `make_multilabel_classification`
(17 public functions, all re-exported at the crate root in
`ferrolearn-datasets/src/lib.rs`).

This is a **value-parity** contract over the **existing** code. The module is
broadly NOT a 1:1 port: nearly every generator diverges from sklearn in at least
one of five classes, and most diverge in several at once:

1. **RNG-substrate** (every stochastic generator). ferrolearn samples from
   `rand::rngs::SmallRng` + `rand_distr::{Normal, Uniform}` (`fn make_rng in
   generators.rs`). sklearn samples from numpy's `RandomState` (Mersenne Twister)
   via `check_random_state` and `numpy.random` distribution methods
   (`_samples_generator.py:210` etc.). These are different bit-streams, so
   **exact value parity against the live oracle is impossible** until ferray
   exposes a numpy-compatible RNG (`ferray::random`) — R-SUBSTRATE-1/5. This is a
   real upstream blocker, not an excuse: the value-parity REQ of each stochastic
   generator is NOT-STARTED on it.
2. **Wrong-signature** (`make_classification`, `make_regression`, `make_blobs`,
   `make_low_rank_matrix`, `make_spd_matrix`, `make_sparse_spd_matrix`,
   `make_gaussian_quantiles`, `make_multilabel_classification`,
   `make_sparse_uncorrelated`): the ferrolearn function exposes a different
   parameter list (missing or renamed params, different defaults) than the
   sklearn signature — an R-DEV-2 ABI divergence independent of the RNG.
3. **Wrong-algorithm** (`make_classification`, `make_low_rank_matrix`,
   `make_spd_matrix`, `make_sparse_spd_matrix`,
   `make_multilabel_classification`): the ferrolearn body computes a *different
   construction* than sklearn's — value parity is unreachable even with a
   matching RNG.
4. **Deterministic-geometry** (`make_moons`, `make_friedman3`): a divergence in
   the deterministic (RNG-independent) part of the construction. **Oracle-
   pinnable now**, no RNG match needed.
5. **Label-encoding / output-contract** (`make_hastie_10_2`,
   `make_gaussian_quantiles`, `make_multilabel_classification`,
   `make_regression`): the returned dtype / label set / row ordering / number of
   return values diverges from sklearn's output contract (R-DEV-3). Several of
   these are oracle-pinnable without an RNG match.

`make_sparse_coded_signal`, `make_biclusters`, `make_checkerboard` exist in
`_samples_generator.py` (`:1473`, `:2110`, `:2232`) but have **no ferrolearn
analog** in `generators.rs`; they are out of scope for this unit (missing-
estimator work, builder territory) and not given REQs here.

No `.rs` edits are proposed here. Every gap is recorded as a NOT-STARTED REQ with
a concrete prerequisite; the acto-critic files the `#NNN` blocker issues.

## Upstream cites (read-only, tag 1.5.2, commit 156ef14)

`sklearn/datasets/_samples_generator.py`:
- `make_classification` signature `:62-79`, hypercube-vertex algorithm
  `:210-322` (centroids `:258-265`, informative draw `:268`, per-cluster
  covariance `:277-280`, redundant `:283-287`, repeated `:290-293`, useless
  `:296-297`, `flip_y` `:300-302`, `shift`/`scale` `:305-311`, shuffle
  `:313-320`).
- `make_multilabel_classification` `:340-504` (Poisson/Multinomial generative
  process `:451-482`, CSR build `:484-496`, indicator/`p_c`/`p_w_c` returns
  `:498-503`).
- `make_hastie_10_2` `:514-570` (label `:567-568`: `y = ((X**2).sum > 9.34)
  .astype(float64)`, then `y[y==0.0] = -1.0`).
- `make_regression` `:589-735` (`n_targets`/`bias`/`effective_rank`/
  `tail_strength`/`shuffle`/`coef` params `:589-601`; ground truth
  `100 * generator.uniform` `:709-712`; noise `:717-718`; shuffle `:721-727`;
  `coef` return `:731-732`).
- `make_circles` `:748-832` (`linspace(0, 2π, n, endpoint=False)` `:813-814`,
  `factor` default 0.8 `:749`, `shuffle` `:826-827`).
- `make_moons` `:844-919` (`linspace(0, π, n)` endpoint=True `:901-904`,
  `shuffle` `:913-914`, noise `:916-917`).
- `make_blobs` `:935-1102` (`center_box` `:941`, per-center `cluster_std` array
  `:1066-1074`, contiguous per-center counts `:1079-1094`, `shuffle` `:1096`,
  `return_centers` `:1099-1100`).
- `make_friedman1` `:1114-1186`, `make_friedman2` `:1197-1270`, `make_friedman3`
  `:1281-1354` (X drawn as a whole then column-scaled `:1259-1264` / `:1343-1348`;
  friedman3 has NO `+1e-6` on X0).
- `make_low_rank_matrix` `:1367-1460` (QR of two Gaussians `:1441-1450`,
  bell-curve singular profile `:1456-1458`, `U·S·Vᵀ` `:1460`; `tail_strength`
  default 0.5, `effective_rank` default 10).
- `make_sparse_uncorrelated` `:1568-1625` (`y ~ N(X0 + 2·X1 − 2·X2 − 1.5·X3, 1)`
  `:1619-1623` — 4 informative weights `[1, 2, −2, −1.5]`, NOT 5).
- `make_spd_matrix` `:1635-1672` (signature `make_spd_matrix(n_dim, *,
  random_state)`; `A=uniform`, SVD of `AᵀA`, `U·(1+diag(uniform))·Vᵀ`
  `:1666-1670`).
- `make_sparse_spd_matrix` `:1694-1825` (sparsity on the **Cholesky factor**,
  `alpha` = P(zero) default 0.95, `norm_diag`/`smallest_coef`/`largest_coef`/
  `sparse_format`, `prec = cholᵀ·chol` `:1797-1815`).
- `make_swiss_roll` `:1837-1908` (`t = 1.5π·(1+2·uniform)` `:1889`, `y =
  21·uniform` `:1890`, `x=t·cos t`, `z=t·sin t` `:1900-1901`, `hole` param).
- `make_s_curve` `:1919-1965` (`t = 3π·(uniform − 0.5)` `:1957`, `X1 =
  2·uniform` `:1960`, `z = sign(t)·(cos t − 1)` `:1961`).
- `make_gaussian_quantiles` `:1980-2086` (keyword-only `mean`/`cov`,
  `multivariate_normal` `:2067`, argsort by squared distance + **X reordered**
  `:2070-2071`, contiguous quantile labels `:2074-2081`, `shuffle` `:2083-2084`).

## Requirements

One or more REQs per generator. The "value-parity" REQ of each stochastic
generator is the element-wise match against the live oracle (RNG-blocked); a
separate REQ captures each non-RNG divergence (signature / algorithm / geometry /
output-contract) that is independently checkable.

- REQ-1 (make_classification algorithm + signature): match sklearn's
  hypercube-vertex clusters method and its 16-parameter signature
  (`n_informative`, `n_redundant`, `n_repeated`, `n_clusters_per_class`,
  `weights`, `flip_y`, `class_sep`, `hypercube`, `shift`, `scale`, `shuffle`).
- REQ-2 (make_regression signature + value parity): match the 11-param signature
  (`n_targets`, `bias`, `effective_rank`, `tail_strength`, `shuffle`, `coef`),
  the `100 * uniform` ground-truth, and the optional `coef` return.
- REQ-3 (make_blobs signature + assignment): match `center_box`, per-center
  `cluster_std` array, `shuffle`, `return_centers`, and the **contiguous
  per-center** sample assignment (not `i % centers`).
- REQ-4 (make_moons geometry): match the deterministic geometry —
  `linspace(0, π, n)` **endpoint=True** spacing `i/(n−1)`. Oracle-pinnable at
  `noise=None`.
- REQ-5 (make_moons shuffle + value parity): match `shuffle=True` default and
  Gaussian-noise value parity.
- REQ-6 (make_circles geometry): match the deterministic geometry —
  `linspace(0, 2π, n, endpoint=False)`, `factor` default 0.8. Oracle-pinnable at
  `noise=None`.
- REQ-7 (make_circles shuffle + value parity): match `shuffle=True` default and
  noise value parity.
- REQ-8 (make_swiss_roll value parity + hole): match the `t`/`y` sampling, the
  `hole` parameter, and noise value parity.
- REQ-9 (make_s_curve value parity): match `t = 3π(uniform−0.5)`, `X1 =
  2·uniform`, and noise value parity.
- REQ-10 (make_sparse_uncorrelated weights + value parity): match the **4**
  informative weights `[1, 2, −2, −1.5]` and the `y ~ N(·, 1)` noise draw (NOT 5
  fixed weights `[1,2,3,4,5]`).
- REQ-11 (make_friedman1 value parity): formula matches; element-wise parity is
  RNG-blocked.
- REQ-12 (make_friedman2 value parity + RNG order): formula matches but X is
  drawn per-row in ferrolearn vs whole-matrix-then-column-scale in sklearn;
  parity RNG-blocked.
- REQ-13 (make_friedman3 deterministic 1e-6 + value parity): ferrolearn adds a
  spurious `+1e-6` to X0 that sklearn does not. Deterministic divergence in the
  X0 column.
- REQ-14 (make_low_rank_matrix algorithm + signature): match the QR-orthonormal
  `U·S·Vᵀ` construction, the `(1−tail_strength)·exp(…)` singular profile, and
  the `tail_strength`/`effective_rank` defaults.
- REQ-15 (make_spd_matrix algorithm + signature): match the `n_dim` signature
  and the SVD-based `U·(1+diag(uniform))·Vᵀ` construction (not `XᵀX + nI`).
- REQ-16 (make_sparse_spd_matrix algorithm + signature): match the Cholesky-
  factor sparsity construction, `alpha`=P(zero), and the
  `norm_diag`/`smallest_coef`/`largest_coef`/`sparse_format` params.
- REQ-17 (make_gaussian_quantiles output contract + signature): match the
  keyword-only `mean`/`cov`/`shuffle` signature, the **X row reordering** by
  squared distance, and the contiguous quantile labelling.
- REQ-18 (make_hastie_10_2 label encoding): return labels in `{−1.0, +1.0}` as
  float (R-DEV-3), not `{0, 1}` as `usize`. Oracle-pinnable on the label set /
  dtype.
- REQ-19 (make_multilabel_classification algorithm + return contract): match the
  Poisson/Multinomial generative process, the `n_labels`/`length`/
  `allow_unlabeled`/`sparse`/`return_indicator`/`return_distributions` params,
  and the dense/sparse indicator + `p_c`/`p_w_c` return options.
- REQ-20 (RNG substrate): every stochastic generator samples from ferray's
  numpy-compatible RNG (`ferray::random` / numpy `RandomState`), not `SmallRng` +
  `rand_distr`. This is the shared prerequisite for all value-parity REQs.
- REQ-21 (ferray array substrate): array types are `ferray-core`, not `ndarray`.
- REQ-22 (production consumer): each public generator has a non-test consumer
  (the `lib.rs` re-export is the boundary surface).

## Acceptance criteria

- AC-1 (REQ-1): `make_classification(random_state=42)` in sklearn returns
  `X.shape==(100, 20)`, `y[:5]==[0,0,1,1,0]` with hypercube-vertex structure;
  ferrolearn's `(n_samples, n_features, n_classes, random_state)` signature
  cannot accept these defaults and produces one Gaussian blob per class. FAILS.
- AC-2 (REQ-2): live `make_regression(n_samples=5, n_features=2, noise=1,
  random_state=42, coef=True)` returns a 3-tuple `(X, y, coef)`; ferrolearn
  returns a 2-tuple and has no `coef`/`n_targets`/`bias`/`effective_rank`.
  FAILS.
- AC-3 (REQ-3): live `make_blobs(n_samples=[3,3,4], centers=None, random_state=0)`
  gives contiguous labels `[0,0,0,1,1,1,2,2,2,2]` (before shuffle); ferrolearn's
  `i % centers` gives interleaved `[0,1,2,0,1,2,0,1,2,0]`. FAILS (deterministic
  assignment-order divergence, independent of RNG).
- AC-4 (REQ-4, deterministic): live `make_moons(n_samples=6, shuffle=False,
  noise=None)` → `X==[[1,0],[0,1],[-1,0],[0,0.5],[1,-0.5],[2,0.5]]`. ferrolearn's
  `θ=π·i/n_upper` (endpoint=False) yields different upper-moon points (e.g. row 1
  uses `θ=π/3` not `π/2`). FAILS — oracle-pinnable now.
- AC-5 (REQ-6, deterministic): live `make_circles(n_samples=6, shuffle=False,
  noise=None, factor=0.8)` → outer `[[1,0],[-0.5,0.866],[-0.5,-0.866]]`, inner
  scaled by 0.8. ferrolearn's `2π·i/n_outer` reproduces the outer/inner geometry
  exactly (endpoint=False matches). The geometry slice of REQ-6 PASSES; the
  `shuffle`/value slice (REQ-7) does not.
- AC-6 (REQ-10, deterministic structure): live `make_sparse_uncorrelated` uses 4
  weights `[1,2,−2,−1.5]`; ferrolearn uses 5 weights `[1,2,3,4,5]` and a
  noise-free `y` (no `N(·,1)` draw). The weight set + noise model diverge
  independently of the RNG. FAILS.
- AC-7 (REQ-13, deterministic): `min(X[:,0])` for ferrolearn `make_friedman3` is
  `≥ 1e-6` (the spurious offset); sklearn's X0 is `uniform(0,100)` with no
  offset. Oracle-pinnable on the X0 lower bound at fixed seed structure.
- AC-8 (REQ-18, deterministic): live `make_hastie_10_2(n_samples=20,
  random_state=0)` → `np.unique(y)==[-1.0, 1.0]`, `y.dtype==float64`. ferrolearn
  returns `Array1<usize>` with unique set `{0, 1}`. FAILS — oracle-pinnable on
  the label set + return type, no RNG match needed.
- AC-9 (REQ-15): live `make_spd_matrix(n_dim=2, random_state=42)` returns a
  non-symmetric-looking SVD product `[[2.09…,0.34…],[0.34…,0.21…]]`; ferrolearn
  builds `XᵀX + nI` (a different, diagonally-dominant matrix). FAILS.
- AC-10 (REQ-17): live `make_gaussian_quantiles` returns X **reordered** so that
  rows are sorted by squared distance with contiguous class blocks; ferrolearn
  keeps X in draw order and labels each row by its rank. The row-ordering
  contract diverges independently of the RNG. FAILS.
- AC-11 (REQ-20): every stochastic generator imports `rand_distr` / `SmallRng`;
  no `ferray::random` usage exists. FAILS.

## REQ status

Headline: of the 22 REQs, **4 are SHIPPED** (REQ-4 make_moons geometry and
REQ-18 make_hastie label encoding, both FIXED in the #1890 iteration; REQ-6
make_circles geometry slice; REQ-22 production-consumer for the grandfathered
boundary re-exports) and **18 are NOT-STARTED**. The dominant blocker is the RNG substrate (REQ-20): it gates every
value-parity REQ. Independently of the RNG, the wrong-signature / wrong-algorithm
/ output-contract divergences (REQ-1/2/3/10/14/15/16/17/18/19) would *still* be
NOT-STARTED even with a numpy-compatible RNG. The deterministic-geometry REQs
(REQ-4 moons, REQ-13 friedman3, REQ-18 hastie label set, REQ-3 blobs assignment
order, REQ-10 weights, REQ-17 row ordering) are oracle-pinnable **now** by the
critic without any RNG match.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (make_classification algo+sig) | NOT-STARTED | `pub fn make_classification in generators.rs` signature is `(n_samples, n_features, n_classes, random_state)` and draws one `Normal(0,3)` centre per class then `class = i % n_classes` (`let class = i % n_classes;`). sklearn is the hypercube-vertex method with 12 keyword params (`_samples_generator.py:62-79`, `:258-320`). Wrong-algorithm **and** wrong-signature, plus RNG-substrate. Blocker: needs the hypercube-vertex rewrite + the 9 missing params (`n_informative`/`n_redundant`/`n_repeated`/`n_clusters_per_class`/`weights`/`flip_y`/`class_sep`/`hypercube`/`shift`/`scale`/`shuffle`) + REQ-20. Blocker issue to be filed by critic. |
| REQ-2 (make_regression sig+parity) | NOT-STARTED | `pub fn make_regression in generators.rs` signature `(n_samples, n_features, n_informative, noise, random_state)` lacks `n_targets`/`bias`/`effective_rank`/`tail_strength`/`shuffle`/`coef`; coefficients drawn `Uniform(-10,10)` (`Uniform::new(-10.0_f64, 10.0_f64)`) vs sklearn `100 * generator.uniform` (`_samples_generator.py:709-712`). No `coef` return (`:731-732`). Wrong-signature + value (RNG). Blocker: add the 6 missing params + `100*uniform` ground truth + `coef` return + REQ-20. Blocker issue to be filed by critic. |
| REQ-3 (make_blobs sig+assignment) | NOT-STARTED | `pub fn make_blobs in generators.rs` assigns `cluster = i % centers` (`let cluster = i % centers;`) — interleaved — vs sklearn's **contiguous** per-center counts (`_samples_generator.py:1079-1094`, `cum_sum_n_samples` blocks). Lacks `center_box` (uses hardcoded `Uniform(-10,10)`), per-center `cluster_std` array, `shuffle`, `return_centers`. The assignment-order divergence is deterministic and oracle-pinnable (AC-3); the rest is signature + RNG. Blocker: contiguous-block assignment + 4 missing params + REQ-20. Blocker issue to be filed by critic. |
| REQ-4 (make_moons geometry) | SHIPPED | FIXED #1891: `pub fn make_moons in generators.rs` now uses `theta = PI * i / (n_upper.saturating_sub(1).max(1))` — endpoint=True spacing `i/(n−1)`, matching `np.linspace(0, π, n_samples_out)` (`_samples_generator.py:901-904`), with numpy's `linspace(0,π,1)==[0.0]` single-point edge. Verified element-wise vs the live oracle (`make_moons(shuffle=False, noise=None)`) at n=10 and edge cases n=2/3/11 within 1e-12 (`divergence_moons_geometry` + `guard_moons_*` in `tests/divergence_generators*.rs`). `shuffle` default + noise value parity remain REQ-5 (NOT-STARTED, RNG). |
| REQ-5 (make_moons shuffle+parity) | NOT-STARTED | `pub fn make_moons in generators.rs` never shuffles (rows emitted upper-then-lower in order) and adds `rand_distr::Normal` noise. sklearn defaults `shuffle=True` and adds `generator.normal` noise (`_samples_generator.py:913-917`). Value parity RNG-blocked. Blocker: REQ-20 + `shuffle` param/default. Blocker issue to be filed by critic. |
| REQ-6 (make_circles geometry) | SHIPPED (geometry slice only) | `pub fn make_circles in generators.rs` uses `theta = 2.0 * PI * (i as f64) / (n_outer.max(1) as f64)` and inner radius `factor_f64 * cos/sin` — matching sklearn's `np.linspace(0, 2π, n, endpoint=False)` (= `2π·i/n`) and `factor` scaling (`_samples_generator.py:813-818`). Verified at `noise=None`: live `make_circles(6, shuffle=False, noise=None, factor=0.8)` → outer `[[1,0],[-0.5,0.866025],[-0.5,-0.866025]]`, inner ×0.8 — reproduced by the ferrolearn formula element-wise. Non-test consumer: re-exported in `lib.rs` (`pub use generators::{... make_circles ...}`). SCOPE: SHIPPED ONLY for the deterministic outer/inner geometry; `factor`'s default (0.8) is exposed as a required arg here, and the `shuffle` default + noise parity are REQ-7 (NOT-STARTED). |
| REQ-7 (make_circles shuffle+parity) | NOT-STARTED | `pub fn make_circles in generators.rs` never shuffles and `factor` is a required argument with no default; sklearn defaults `factor=0.8`, `shuffle=True` (`_samples_generator.py:748-749`, `:826-827`) and noise is a `generator.normal` draw (`:829-830`). Value parity + default RNG-blocked. Blocker: REQ-20 + `shuffle`/`factor` defaults. Blocker issue to be filed by critic. |
| REQ-8 (make_swiss_roll parity+hole) | NOT-STARTED | `pub fn make_swiss_roll in generators.rs` draws `t ~ Uniform(1.5π, 4.5π)` and `height ~ Uniform(0,21)` per-row via `rand_distr`; sklearn computes `t = 1.5π·(1 + 2·uniform)` (mathematically the same support but a different RNG draw) and `y = 21·uniform`, then `x=t·cos t`, `z=t·sin t` (`_samples_generator.py:1889-1901`) and adds noise as a `(3, n)` block. Lacks the `hole` parameter (`:1837`, `:1891-1898`). Value parity RNG-blocked; `hole` is a missing feature. Blocker: REQ-20 + `hole`. Blocker issue to be filed by critic. |
| REQ-9 (make_s_curve parity) | NOT-STARTED | `pub fn make_s_curve in generators.rs` draws `t ~ Uniform(−1.5π, 1.5π)`, `height ~ Uniform(0,2)`; sklearn uses `t = 3π·(uniform − 0.5)`, `X1 = 2·uniform`, `z = sign(t)·(cos t − 1)` (`_samples_generator.py:1957-1961`). The formula for X matches but the per-row RNG draw order/values differ. Value parity RNG-blocked. Blocker: REQ-20. Blocker issue to be filed by critic. |
| REQ-10 (make_sparse_uncorrelated weights+parity) | NOT-STARTED | `pub fn make_sparse_uncorrelated in generators.rs` uses **5** fixed weights `let weights = [1.0, 2.0, 3.0, 4.0, 5.0];` and a noise-free `y = Σ wᵢ·Xᵢ`; sklearn uses **4** weights `[1, 2, −2, −1.5]` and `y ~ N(X0 + 2·X1 − 2·X2 − 1.5·X3, 1)` (`_samples_generator.py:1619-1623`). Wrong weight set + missing target noise — both deterministic-structural divergences (oracle-pinnable on the weight pattern, AC-6) plus RNG for the noise draw. Blocker: correct weights `[1,2,−2,−1.5]` + the `N(·,1)` target draw + REQ-20. Blocker issue to be filed by critic. |
| REQ-11 (make_friedman1 parity) | NOT-STARTED | `pub fn make_friedman1 in generators.rs` formula matches sklearn (`10·sin(π·x0·x1) + 20·(x2−0.5)² + 10·x3 + 5·x4 + noise`, `_samples_generator.py:1178-1184`), but X is drawn per-row from `rand_distr::Uniform` and noise from `rand_distr::Normal` — different bit-stream from numpy. Value parity RNG-blocked. Blocker: REQ-20. Blocker issue to be filed by critic. |
| REQ-12 (make_friedman2 parity+RNG order) | NOT-STARTED | `pub fn make_friedman2 in generators.rs` draws each row's 4 uniforms then scales (`r0 = u01.sample()*100.0; r1 = 40π + u01.sample()*(560−40)π; …`); sklearn draws the **whole** `(n,4)` uniform matrix first, then column-scales (`X = generator.uniform(size=(n,4)); X[:,0]*=100; …`, `_samples_generator.py:1259-1264`). Both the RNG substrate and the draw *order* diverge. Value parity RNG-blocked + ordering. Blocker: REQ-20 + whole-matrix draw order. Blocker issue to be filed by critic. |
| REQ-13 (make_friedman3 1e-6 + parity) | NOT-STARTED | `pub fn make_friedman3 in generators.rs` adds a spurious offset `let r0 = u01.sample(&mut rng) * 100.0 + 1e-6;` to X0 that sklearn does NOT (`X[:, 0] *= 100`, `_samples_generator.py:1344`). DETERMINISTIC divergence in the X0 column (oracle-pinnable: ferrolearn `min(X[:,0]) ≥ 1e-6`, AC-7), plus the same per-row vs whole-matrix RNG-order issue as REQ-12. Blocker: remove the `+1e-6` + REQ-20 + draw order. Blocker issue to be filed by critic. |
| REQ-14 (make_low_rank_matrix algo+sig) | NOT-STARTED | `pub fn make_low_rank_matrix in generators.rs` builds `A·B` from two raw Gaussians with B's columns scaled by `sigma` — explicitly an "approximation" (the comment says "simple approximation"), NOT sklearn's `U·S·Vᵀ` with `U`,`V` from economic **QR** of Gaussians (`_samples_generator.py:1441-1460`). The singular profile also diverges: ferrolearn `exp(−(i/eff)²) + tail·exp(−0.1·i)`, sklearn `(1−tail)·exp(−(i/eff)²) + tail·exp(−0.1·i/eff)` (`:1456-1457`) — note the missing `(1−tail_strength)` factor and the `/eff` in the tail. Wrong-algorithm + signature (no defaults). Blocker: QR-orthonormal construction + corrected singular profile + REQ-20. Blocker issue to be filed by critic. |
| REQ-15 (make_spd_matrix algo+sig) | NOT-STARTED | `pub fn make_spd_matrix in generators.rs` signature is `(n, random_state)` and builds `A = XᵀX + nI` from a Gaussian `X` (`let sym = a.t().dot(&a); … + if i==j { n_f }`). sklearn's `make_spd_matrix(n_dim, …)` builds `U·(1 + diag(uniform))·Vᵀ` from the SVD of `AᵀA` with `A` **uniform** (`_samples_generator.py:1666-1670`). Wrong-algorithm (different matrix) + the `n` vs `n_dim` parameter name (R-DEV-2). Blocker: SVD-based construction + `n_dim` name + REQ-20. Blocker issue to be filed by critic. |
| REQ-16 (make_sparse_spd_matrix algo+sig) | NOT-STARTED | `pub fn make_sparse_spd_matrix in generators.rs` zeroes off-diagonals with prob `1−alpha` directly on the precision matrix then diagonally shifts for SPD; sklearn imposes sparsity on the **Cholesky factor** (`chol = −I + tril(sparse aux); prec = cholᵀ·chol`, `_samples_generator.py:1797-1815`) — a different matrix distribution, and `alpha` is P(zero) on the factor not on the precision off-diagonals. Lacks `norm_diag`/`smallest_coef`/`largest_coef`/`sparse_format`. Wrong-algorithm + signature. Blocker: Cholesky-factor construction + 4 missing params + REQ-20. Blocker issue to be filed by critic. |
| REQ-17 (make_gaussian_quantiles contract+sig) | NOT-STARTED | `pub fn make_gaussian_quantiles in generators.rs` signature `(n_samples, n_features, n_classes, random_state)` lacks the keyword-only `mean`/`cov`/`shuffle`; it sorts indices by squared norm and labels each row **in place by rank** (`y[*idx] = cls`), keeping X in draw order. sklearn **reorders X** so rows are sorted by distance with contiguous class blocks (`idx = np.argsort(...); X = X[idx, :]`, `_samples_generator.py:2070-2081`). Output-contract (row ordering) divergence — oracle-pinnable (AC-10) — plus signature + RNG (`multivariate_normal`). Blocker: reorder X by argsort + add `mean`/`cov`/`shuffle` + REQ-20. Blocker issue to be filed by critic. |
| REQ-18 (make_hastie_10_2 label encoding) | SHIPPED | FIXED #1892: `pub fn make_hastie_10_2 in generators.rs` now returns `Array1<F>` with `y[i] = if s > 9.34 { F::one() } else { -F::one() }` — label set `{−1.0, +1.0}` as float, matching sklearn (`y = ((X**2).sum > 9.34).astype(float64); y[y==0.0] = −1.0`, `_samples_generator.py:567-568`); strict `> 9.34` threshold preserved. Pinned by `divergence_hastie_label_encoding` (label set + dtype, oracle-grounded). The X *values* remain RNG-blocked (REQ-20). |
| REQ-19 (make_multilabel_classification algo+return) | NOT-STARTED | `pub fn make_multilabel_classification in generators.rs` picks `n_labels.clamp(1, n_classes)` distinct random classes per sample and fabricates X as `uniform − 0.5 + Σ(c+1)·0.1` — NOT sklearn's Poisson/Multinomial word-generation process (`_samples_generator.py:451-496`). Returns `(X, Array2<usize>)` only; lacks `length`/`allow_unlabeled`/`sparse`/`return_indicator`/`return_distributions` and the `p_c`/`p_w_c` returns (`:340-352`, `:498-503`). Wrong-algorithm + return-contract + signature. Blocker: full Poisson/Multinomial rewrite + CSR/indicator returns + 5 missing params + REQ-20. Blocker issue to be filed by critic. |
| REQ-20 (RNG substrate) | NOT-STARTED | `generators.rs` imports `use rand::rngs::SmallRng;` and `use rand_distr::{Distribution, Normal, Uniform};`; `fn make_rng` returns a `SmallRng`. R-SUBSTRATE-1 requires `ferray::random` (the numpy `RandomState` analog). Without it, no stochastic generator can value-match the live oracle (Mersenne-Twister bit-stream). This is the shared prerequisite for REQ-2/5/7/8/9/11/12/13/14/15/16/17/18(X)/19 value parity. Per R-SUBSTRATE-5 the fix belongs in ferray; until it ships, every value-parity REQ is NOT-STARTED on it. Blocker issue to be filed by critic. |
| REQ-21 (ferray array substrate) | NOT-STARTED | `generators.rs` uses `use ndarray::{Array1, Array2};` and returns `ndarray::Array2<F>`/`Array1<F>`/`Array1<usize>`. R-SUBSTRATE-1 requires `ferray-core` array types. Blocker: migrate the array types to the ferray analog (bridge via `into_ndarray()` during transition). Blocker issue to be filed by critic. |
| REQ-22 (production consumer) | SHIPPED (boundary re-export) | All 17 generators are re-exported at the crate root: `ferrolearn-datasets/src/lib.rs` `pub use generators::{ make_blobs, make_circles, make_classification, make_friedman1, make_friedman2, make_friedman3, make_gaussian_quantiles, make_hastie_10_2, make_low_rank_matrix, make_moons, make_multilabel_classification, make_regression, make_s_curve, make_sparse_spd_matrix, make_sparse_uncorrelated, make_spd_matrix, make_swiss_roll };`. These are existing grandfathered boundary pub APIs (R-DEFER-1/S5: consumers are external users + the future `ferrolearn-python` `datasets` binding). Verification: `grep -rn "make_classification\|make_blobs\|make_moons" ferrolearn-* --include=*.rs | grep -v '#\[cfg(test\|/tests/'` → only the `lib.rs` re-exports; no in-workspace non-test caller located. Underclaim: this is the boundary-grandfather clause only — no internal production consumer exists today (the `ferrolearn-python` datasets binding is not yet present). |

## Architecture

`generators.rs` is a single file of 17 free functions plus `fn make_rng` (the
RNG factory). There is no fitted/unfitted estimator split — these are pure data-
generating functions, mirroring sklearn's `@validate_params`-decorated
module-level `make_*` functions. The crate-root re-exports in
`ferrolearn-datasets/src/lib.rs` are the public boundary surface.

`fn make_rng in generators.rs` is the substrate fault line: it returns a
`SmallRng` (`SmallRng::seed_from_u64` / `SmallRng::from_os_rng`), and every
stochastic generator threads that through `rand_distr` `Normal`/`Uniform`
samplers. sklearn instead calls `check_random_state(random_state)` to obtain a
numpy `RandomState` and draws via `generator.standard_normal` /
`generator.uniform` / `generator.normal` / `generator.poisson` /
`generator.multivariate_normal` / `util_shuffle`. The two RNGs produce different
bit-streams, which is why REQ-20 gates all value parity.

Divergence map (function → divergence classes):

- **Wrong-signature + wrong-algorithm**: `make_classification` (one-blob-per-class
  vs hypercube vertices), `make_low_rank_matrix` (`A·B` approximation vs
  `U·S·Vᵀ` QR), `make_spd_matrix` (`XᵀX+nI` vs SVD product),
  `make_sparse_spd_matrix` (precision-off-diagonal zeroing vs Cholesky-factor
  sparsity), `make_multilabel_classification` (distinct-class pick vs
  Poisson/Multinomial).
- **Wrong-signature + value(RNG)**: `make_regression` (missing
  `n_targets`/`bias`/`effective_rank`/`coef`, `Uniform(−10,10)` vs `100·uniform`),
  `make_blobs` (`i % centers` interleave vs contiguous blocks; missing
  `center_box`/per-center-std/`shuffle`/`return_centers`).
- **Deterministic-geometry**: `make_moons` (endpoint-False `i/n` vs sklearn
  endpoint-True `i/(n−1)`), `make_friedman3` (spurious `+1e-6` on X0).
- **Output-contract**: `make_hastie_10_2` (`{0,1}` `usize` vs `{−1,+1}` float),
  `make_gaussian_quantiles` (X not reordered; rank-labelled in place vs argsort-
  reordered with contiguous blocks).
- **Deterministic-structure + value(RNG)**: `make_sparse_uncorrelated` (5 weights
  `[1,2,3,4,5]` vs 4 weights `[1,2,−2,−1.5]`; no target noise).
- **Geometry-correct, value(RNG)-only**: `make_circles` (outer/inner geometry
  matches at `noise=None`; only `shuffle`/`factor` default + noise diverge),
  `make_friedman1`/`make_friedman2` (formula matches; RNG bit-stream + (friedman2)
  draw order diverge), `make_swiss_roll`/`make_s_curve` (manifold formula matches;
  RNG draw + (swiss_roll) `hole` diverge).

Invariant the existing code DOES honor across all generators: shape correctness
(every `X` has the documented `(n_samples, n_features)` / `(·, 3)` / `(·, 2)`
shape) and seed-reproducibility within ferrolearn (same `random_state` → same
output, pinned by the in-crate `*_reproducible` tests). Shape + self-
reproducibility is NOT value parity (R-HONEST-1/3): a generator is SHIPPED for
value parity only when it matches the live sklearn oracle element-wise, which
none of the stochastic generators can until REQ-20 lands.

## Verification

Commands establishing the SHIPPED claims and grounding the NOT-STARTED rows
(expected values from the live sklearn 1.5.2 oracle, never copied from the
ferrolearn side, per R-CHAR-3):

```bash
cargo test -p ferrolearn-datasets --lib generators   # in-crate shape/reproducibility tests pass
```

Deterministic, oracle-pinnable NOW (the critic adds these as FAILING `#[test]`s,
no RNG match required):

```bash
# REQ-4 make_moons geometry (endpoint off-by-one):
python3 -c "from sklearn.datasets import make_moons; \
X,y=make_moons(n_samples=6, shuffle=False, noise=None); print(X.tolist())"
# -> [[1,0],[0,1],[-1,0],[0,0.5],[1,-0.5],[2,0.5]] ; ferrolearn's pi*i/n_upper differs

# REQ-6 make_circles geometry (SHIPPED slice — ferrolearn MATCHES):
python3 -c "from sklearn.datasets import make_circles; \
X,y=make_circles(n_samples=6, shuffle=False, noise=None, factor=0.8); print(X.tolist())"
# -> outer [[1,0],[-0.5,0.866...],[-0.5,-0.866...]], inner x0.8 ; ferrolearn reproduces

# REQ-18 make_hastie_10_2 label set + dtype:
python3 -c "from sklearn.datasets import make_hastie_10_2; import numpy as np; \
X,y=make_hastie_10_2(n_samples=20, random_state=0); print(np.unique(y).tolist(), y.dtype)"
# -> [-1.0, 1.0] float64 ; ferrolearn returns Array1<usize> {0,1}

# REQ-3 make_blobs contiguous assignment:
python3 -c "from sklearn.datasets import make_blobs; \
_,y=make_blobs(n_samples=[3,3,4], centers=None, n_features=2, random_state=0); print(y.tolist())"
# (with shuffle=False the labels are contiguous blocks; ferrolearn interleaves i%centers)

# REQ-10 make_sparse_uncorrelated weight set:
python3 -c "from sklearn.datasets import make_sparse_uncorrelated  # y ~ N(X0+2X1-2X2-1.5X3, 1)"
# ferrolearn uses [1,2,3,4,5] with no target noise

# REQ-13 make_friedman3 spurious 1e-6: ferrolearn min(X[:,0]) >= 1e-6 ; sklearn uniform(0,100)

# REQ-17 make_gaussian_quantiles X reordering:
python3 -c "from sklearn.datasets import make_gaussian_quantiles  # X = X[argsort(dist)]"
```

RNG-BLOCKED value-parity REQs (REQ-2/5/7/8/9/11/12/13/14/15/16/17/18-X/19) cannot
be pinned to element-wise equality until ferray exposes a numpy-compatible
`RandomState` (REQ-20). Per R-SUBSTRATE-5 the critic files the ferray-RNG blocker
and marks these NOT-STARTED; the fix lands in ferray, then the value-parity tests
go green. Do NOT pin a tautological "ferrolearn == ferrolearn" test for these
(R-CHAR-3).

Consumer check (REQ-22):

```bash
grep -rn "make_classification\|make_blobs\|make_moons\|make_circles\|make_regression" \
  ferrolearn-* --include=*.rs | grep -v '#\[cfg(test' | grep -v '/tests/'
# -> only the ferrolearn-datasets/src/lib.rs re-exports (boundary surface)
```

Each NOT-STARTED REQ above is an open work item; the acto-critic pins the
deterministic divergences (REQ-3/4/10/13/17/18) as failing characterization
tests against the live-oracle commands here and files the specific `#NNN`
blockers — including the shared ferray-RNG substrate blocker (REQ-20) that gates
all value parity.
