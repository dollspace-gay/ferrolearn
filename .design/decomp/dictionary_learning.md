# DictionaryLearning (sklearn.decomposition.DictionaryLearning)

<!--
tier: 3-component
status: shipped-partial
baseline-commit: 7443a1a9
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_dict_learning.py  # class DictionaryLearning(_BaseSparseCoding, BaseEstimator) (:1372-1711). ctor (:1588-1629): n_components=None, *, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm="lars", transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None, dict_init=None, callback=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000. _parameter_constraints (:1565-1586): fit_algorithm StrOptions({"lars","cd"}), transform_algorithm StrOptions({"lasso_lars","lasso_cd","lars","omp","threshold"}). fit_transform (:1651-1702): method = "lasso_" + self.fit_algorithm (:1671, default "lars" -> "lasso_lars"); n_components = X.shape[1] if None (:1676-1679); V, U, E, self.n_iter_ = _dict_learning(X, n_components, alpha, tol, max_iter, method, method_max_iter=transform_max_iter, ..., code_init, dict_init, ..., random_state, positive_dict, positive_code) (:1681-1698); self.components_ = U (:1699); self.error_ = E (:1700); returns V (the codes). _dict_learning (:554-674): SVD-based init (code, S, dictionary = linalg.svd(X, full_matrices=False) + svd_flip, DETERMINISTIC given X, :581-584) when dict_init/code_init None; alternating sparse_encode (:620, LARS/CD lasso) + _update_dict (:632-640). _update_dict (:474-551): block-coordinate descent per atom dictionary[k] += (B[:,k] - A[k] @ dictionary)/A[k,k] (:531); UNUSED-ATOM RESAMPLING newd = Y[random_state.choice(n_samples)] + noise (:534-540) when A[k,k] <= 1e-6; optional positive clip (:544-545); unit-ball projection dictionary[k] /= max(norm, 1) (:548). _BaseSparseCoding._transform (:1110-1139) / transform (:1141-1159): sparse_encode(X, components_, algorithm=transform_algorithm, transform_n_nonzero_coefs, alpha=transform_alpha or alpha, max_iter=transform_max_iter, positive=positive_code) then optional split_sign (:1131-1137). class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator) (:1715) — online variant (batch_size, max_no_improvement, partial_fit), ABSENT in ferrolearn.
ferrolearn-module: ferrolearn-decomp/src/dictionary_learning.rs
parity-ops: DictionaryLearning, sparse_encode
crosslink-issue: 1512
-->

## Summary

`ferrolearn-decomp/src/dictionary_learning.rs` mirrors scikit-learn's
`DictionaryLearning` (`sklearn/decomposition/_dict_learning.py`,
`class DictionaryLearning(_BaseSparseCoding, BaseEstimator)` `:1372`): learn an
overcomplete dictionary `D` (the `components_`, `n_components × n_features`) and
sparse codes `A` such that `X ≈ A·D`, solving
`argmin 0.5·||X − U·V||_Fro² + alpha·||U||_{1,1}` subject to `||V_k||₂ ≤ 1`
(docstring `:1378-1386`) by alternating a sparse-coding step (with `D` fixed) and a
dictionary-update step (with `A` fixed). The exposed surface is the unfitted
`DictionaryLearning { n_components, alpha (1.0), max_iter (1000), tol (1e-8),
fit_algorithm: DictFitAlgorithm{CoordinateDescent}, transform_algorithm:
DictTransformAlgorithm{Omp|LassoCd|Threshold}, transform_n_nonzero_coefs, random_state }`
(`dictionary_learning.rs`, struct at line 70; builders `with_alpha`/`with_max_iter`/
`with_tol`/`with_fit_algorithm`/`with_transform_algorithm`/
`with_transform_n_nonzero_coefs`/`with_random_state`, accessors) and the fitted
`FittedDictionaryLearning { components_ (n_components, n_features), alpha_, n_iter_,
reconstruction_err_, transform_algorithm_, transform_n_nonzero_coefs_ }`
(`dictionary_learning.rs`, struct at line 210, accessors `components`/`n_iter`/
`reconstruction_err`), re-exported at the crate root (`pub use
dictionary_learning::{DictFitAlgorithm, DictTransformAlgorithm, DictionaryLearning,
FittedDictionaryLearning}`, `lib.rs:84`). The standalone `sparse_encode` helper
also exposes fixed-dictionary sparse coding for OMP, lasso-cd, and threshold,
returning `(n_samples, n_components)` codes. The path is **f64-ONLY** — `impl
Fit<Array2<f64>, ()>` (`dictionary_learning.rs`, impl at line 483), NOT generic over
`<F>`. There is NO PyO3 binding (a `grep -rn DictionaryLearning
ferrolearn-python/src/` is empty) and NO `tests/divergence_dictionary_learning.rs`.

**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED,
CARVE-OUT `#1513`).** ferrolearn's `components_` VALUES diverge from sklearn through
THREE algorithmic differences plus the RNG. (1) **Init:** ferrolearn seeds a random
Gaussian dictionary `Normal(0,1)` via `Xoshiro256PlusPlus::seed_from_u64(random_state
.unwrap_or(0))` then `normalise_dictionary` (`dictionary_learning.rs:530-536`),
whereas sklearn's `_dict_learning` default (`dict_init=None`/`code_init=None`) is the
SVD-based `code, S, dictionary = linalg.svd(X, full_matrices=False)` + `svd_flip`
(`_dict_learning.py:581-584`) — DETERMINISTIC given `X`. (2) **Sparse coder:**
ferrolearn uses a hand-rolled soft-threshold coordinate descent `lasso_cd_single`
(`dictionary_learning.rs:271`), whereas sklearn's default `fit_algorithm="lars"`
yields `method = "lasso_lars"` (`_dict_learning.py:1671`), an LARS-solved lasso. (3)
**Dictionary update:** ferrolearn solves normal equations
`D[:,j] = solve((AᵀA + 1e-10·I), AᵀX[:,j])` then `normalise_dictionary`
(`dictionary_learning.rs:554-580`), whereas sklearn's `_update_dict`
(`_dict_learning.py:474-551`) is per-atom block-coordinate descent
(`dictionary[k] += (B[:,k] − A[k]@dictionary)/A[k,k]` `:531`) with UNUSED-ATOM
RESAMPLING from a numpy `RandomState` (`newd = Y[random_state.choice(n_samples)]` +
noise `:534-540`) and unit-BALL projection (`/= max(norm, 1)` `:548`). Different
init + different solver + different dict update + different RNG ⇒ the `components_`
VALUES diverge (same class as the `minibatch_nmf` / `sparse_pca` RNG carve-outs); no
failing test is asserted (R-DEFER-3).

**SPARSE_ENCODE VALUE PARITY (SHIPPED scoped).**
`ferrolearn_decomp::sparse_encode` covers
`sklearn.decomposition.sparse_encode` for dense f64 OMP, lasso-cd, and threshold
paths. `tests/divergence_sparse_encode.rs` pins OMP, threshold, and lasso-cd
against the live sklearn 1.5.2 oracle on a fixed dictionary. Residual unsupported
paths are `lasso_lars`, `lars`, positive coding, precomputed `gram`/`cov`, `init`,
and `n_jobs`.

As of this iteration: the STRUCTURAL unit-L2 dictionary atoms, the `components_`
shape `(n_components, n_features)`, the sparse-code `transform` (shape + OMP
n-nonzero cap + LassoCd/threshold sparsity), the standalone `sparse_encode`
OMP/lasso-cd/threshold value-parity slice, the monitored/finite `reconstruction_err_` and
positive `n_iter_`, the error & parameter contracts (n_components 0, n_samples 0,
n_features 0, alpha < 0, transform col mismatch), and determinism given a seed
(REQ-1,2,3) are SHIPPED scoped; exact `components_` value parity (REQ-4, CARVE-OUT
`#1513`), `fit_algorithm="lars"`/LARS solver (REQ-5, `#1514`), SVD-based
`dict_init`/`code_init` init (REQ-6, `#1515`), `_update_dict` BCD + atom resampling
(REQ-7, `#1516`), `transform_alpha` + the full transform-algorithm set
(`lasso_lars`/`lars`; ferrolearn ships `omp`/`lasso_cd`/`threshold`) plus positive
coding/precomputed sparse_encode paths (REQ-8 residual, `#1517`), `split_sign` (REQ-9, `#1518`), `positive_code`/`positive_dict` (REQ-10,
`#1519`), `transform_max_iter` (REQ-11, `#1520`), `MiniBatchDictionaryLearning`
(REQ-12, `#1521`), `error_`/`n_components_`/`n_features_in_` fitted attrs (REQ-13,
`#1522`), generic `F` (f64-only) (REQ-14, `#1523`), the PyO3 binding (REQ-15,
`#1524`), and the ferray substrate (REQ-16, `#1525`) are NOT-STARTED —
**3 SHIPPED (REQ-1,2,3) + REQ-8 sparse_encode/threshold scoped / residual gaps open**.

`DictionaryLearning` / `FittedDictionaryLearning` are existing pub APIs whose
non-test consumer is the crate re-export (`lib.rs:84`, boundary public API,
grandfathered S5/R-DEFER-1). There is NO PyO3 binding (REQ-15 NOT-STARTED).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2 SHIPPED scoped + REQ-4 residual) — components_ shape
# (n_components, n_features), atoms UNIT L2-NORM, n_iter_, error_ vector,
# n_features_in_. Fixed 6x4 X. VALUES generated by sklearn, never copied from
# ferrolearn (R-CHAR-3).
python3 -c "
import numpy as np
from sklearn.decomposition import DictionaryLearning
X=np.array([[1.,2,3,0],[4,5,6,1],[7,8,9,2],[2,1,0,3],[0,3,4,1],[3,0,1,2]])
m=DictionaryLearning(n_components=3, alpha=1, random_state=0).fit(X)
print('components_ shape:', m.components_.shape)
print('atom L2 norms:', [round(float(np.linalg.norm(r)),6) for r in m.components_])
print('components_ row0:', np.round(m.components_[0],6).tolist())
print('n_iter_:', m.n_iter_, 'error_ len/last:', len(m.error_), round(float(m.error_[-1]),4))
print('n_features_in_:', m.n_features_in_)"
# -> components_ shape: (3, 4)                                  => structural shape (REQ-1)
# -> atom L2 norms: [1.0, 1.0, 1.0]                             => UNIT L2 atoms (REQ-2)
# -> components_ row0: [0.487906, 0.566806, 0.650717, 0.131324] => VALUES (REQ-4 CARVE-OUT, NOT reproduced)
# -> n_iter_: 11 error_ len/last: 11 37.3122                    => error_ VECTOR (REQ-13 NOT-STARTED)
# -> n_features_in_: 4                                          => fitted attr (REQ-13 NOT-STARTED)

# PROBE 2 (REQ-1 SHIPPED scoped: atoms unit-norm) — the unit-ball projection in
# _update_dict (:548) normalises every atom; ferrolearn's normalise_dictionary
# does the same (unit L2). Demonstrated above (all norms == 1.0).

# PROBE 3 (REQ-5/6/8..16 NOT-STARTED) — ctor defaults + the LARS default flow.
python3 -c "
from sklearn.decomposition import DictionaryLearning
m=DictionaryLearning()
for p in ['n_components','alpha','max_iter','tol','fit_algorithm','transform_algorithm','transform_n_nonzero_coefs','transform_alpha','n_jobs','code_init','dict_init','split_sign','positive_code','positive_dict','transform_max_iter']:
    print(f'{p} =', getattr(m,p))
print('method = lasso_ + fit_algorithm =', 'lasso_'+m.fit_algorithm)"
# -> n_components = None  alpha = 1  max_iter = 1000  tol = 1e-08
# -> fit_algorithm = lars  transform_algorithm = omp  transform_n_nonzero_coefs = None
# -> transform_alpha = None  n_jobs = None  code_init = None  dict_init = None
# -> split_sign = False  positive_code = False  positive_dict = False  transform_max_iter = 1000
# -> method = lasso_ + fit_algorithm = lasso_lars
#    => ferrolearn has n_components/alpha/max_iter/tol/fit_algorithm(CD-only)/
#       transform_algorithm(Omp|LassoCd)/transform_n_nonzero_coefs/random_state only;
#       NO transform_alpha / split_sign / positive_code / positive_dict /
#       transform_max_iter / n_jobs / code_init / dict_init; fit_algorithm default is
#       "lars" (NOT CD), transform_algorithm has lasso_lars/lars/threshold too.

# PROBE 4 (REQ-1 SHIPPED scoped: transform sparse codes + OMP n-nonzero cap).
python3 -c "
import numpy as np
from sklearn.decomposition import DictionaryLearning
X=np.array([[1.,2,3,0],[4,5,6,1],[7,8,9,2],[2,1,0,3],[0,3,4,1],[3,0,1,2]])
m=DictionaryLearning(n_components=3, alpha=1, transform_algorithm='omp', transform_n_nonzero_coefs=2, random_state=0).fit(X)
Xt=m.transform(X)
print('transform shape:', Xt.shape)
print('max nnz per row:', int(max((np.abs(r)>1e-12).sum() for r in Xt)))"
# -> transform shape: (6, 3)         => sparse codes (n_samples, n_components) (REQ-1)
# -> max nnz per row: 2              => OMP n_nonzero_coefs cap (REQ-1)
```

## Requirements

- REQ-1: **Structural: `components_` shape `(n_components, n_features)`, sparse-code
  `transform` shape `(n_samples, n_components)` with OMP n-nonzero cap / LassoCd
  sparsity, finite `reconstruction_err_`, positive `n_iter_`, determinism given seed
  (SHIPPED scoped).** `fn fit` (`dictionary_learning.rs`, impl at line 495) stores
  `components_ = d` of shape `(n_components, n_features)`
  (`dictionary_learning.rs:601-602`, field `:213`, = sklearn `components_ = U`
  `_dict_learning.py:1699`, shape `_dict_learning.py:1501`), the Frobenius
  `reconstruction_err_` (`reconstruction_error`, fn at line 469; stored `:605`), and
  `n_iter_` (`:604`). `transform` (`impl Transform for FittedDictionaryLearning`, fn
  at line 622) returns codes of shape `(n_samples, n_components)`
  (`dictionary_learning.rs:632-651`); OMP (`omp_single`, fn at line 327) caps the
  active set at `max_nonzero` (`dictionary_learning.rs:333-335`) and LassoCd
  (`lasso_cd_single`, fn at line 271) yields sparse codes via the L1 prox. The seeded
  `Xoshiro256PlusPlus` (`dictionary_learning.rs:530`, seed `random_state.unwrap_or(0)`
  `:526`) plus the deterministic CD/normal-equations loop make the fit reproducible
  given a seed. **Scope: STRUCTURAL (shapes / sparsity-cap / finiteness /
  determinism), NOT component VALUES (REQ-4).** Pinned by `test_dictlearn_basic_shape`
  `(5,10)`, `test_dictlearn_transform_shape` `(20,5)`,
  `test_dictlearn_omp_nonzero_coefs` (≤2 nnz/row), `test_dictlearn_sparsity_of_codes`,
  `test_dictlearn_reconstruction_error_decreases`, `test_dictlearn_fitted_accessors`.
  Non-test consumer: re-export `lib.rs:84`.

- REQ-2: **Structural: dictionary atoms (rows of `components_`) are UNIT L2-NORM
  (SHIPPED scoped).** `normalise_dictionary` (fn at line 251) divides each atom row by
  its L2 norm (`dictionary_learning.rs:254-265`), and `fn fit` calls it both at init
  (`dictionary_learning.rs:536`) and after every dictionary update
  (`dictionary_learning.rs:580`), so the stored `components_` rows are unit-norm —
  mirroring sklearn's unit-ball constraint `||V_k||₂ ≤ 1` (docstring
  `_dict_learning.py:1382`), enforced in `_update_dict` by `dictionary[k] /=
  max(linalg.norm(dictionary[k]), 1)` (`_dict_learning.py:548`). Probe 1 confirms
  sklearn atoms are unit-norm (`[1.0, 1.0, 1.0]`). **FLAG (candidate DIV):** sklearn
  projects onto the unit BALL (`max(norm, 1)`, so an already-short atom keeps its
  norm), whereas ferrolearn always normalises to the unit SPHERE (`/= norm`); for a
  converged dictionary both coincide, but the constraint differs (folds into REQ-7).
  Pinned by `test_dictlearn_dictionary_atoms_normalised`. **Scope: STRUCTURAL
  unit-norm, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:84`.

- REQ-3: **Error / parameter contracts (SHIPPED scoped).** `fn fit`
  (`dictionary_learning.rs`, impl at line 495) returns `InvalidParameter { name:
  "n_components" }` for `n_components == 0` (`dictionary_learning.rs:499-504`),
  `InsufficientSamples { required: 1 }` for `0` samples
  (`dictionary_learning.rs:505-511`), `InvalidParameter { name: "X" }` for `0`
  features (`dictionary_learning.rs:512-517`), and `InvalidParameter { name: "alpha"
  }` for `alpha < 0` (`dictionary_learning.rs:518-523`); `transform` returns
  `ShapeMismatch` on a column-count mismatch (`dictionary_learning.rs:624-630`).
  Pinned by `test_dictlearn_invalid_n_components_zero`, `test_dictlearn_empty_data`,
  `test_dictlearn_zero_features`, `test_dictlearn_invalid_alpha_negative`,
  `test_dictlearn_transform_shape_mismatch`. **FLAG (candidate DIVs):** sklearn
  validates via `_parameter_constraints` (`_dict_learning.py:1565-1586`,
  `n_components` in `[None, [1,∞)]`, `alpha`/`tol` in `[0,∞)`, `max_iter` in
  `[0,∞)`) raising `InvalidParameterError`, NOT `FerroError`; sklearn accepts
  `n_components=None` (→ `n_features`, `_dict_learning.py:1676-1679`) which ferrolearn
  requires as an explicit `usize`; sklearn does NOT pre-reject `0` features (SVD
  surfaces it later). Non-test consumer: re-export `lib.rs:84`.

- REQ-4: **EXACT `components_` value parity with sklearn's `_dict_learning`
  (NOT-STARTED, CARVE-OUT; `#1513`).** sklearn's `fit_transform`
  (`_dict_learning.py:1651-1702`) calls `_dict_learning` (`:554-674`) which (a)
  SVD-inits `code, S, dictionary = linalg.svd(X, full_matrices=False)` + `svd_flip`
  (`:581-584`, DETERMINISTIC given `X`), (b) sparse-codes via `sparse_encode` with the
  LARS lasso `method = "lasso_lars"` (`:1671`, `:620`), and (c) updates the dictionary
  via `_update_dict` BCD with unused-atom resampling from a numpy `RandomState`
  (`:474-551`), then sets `components_ = U` (`:1699`). ferrolearn's `fn fit`
  (`dictionary_learning.rs`, impl at line 495) uses a DIFFERENT algorithm on all three
  axes — random-Gaussian `Xoshiro` init + normalise (`dictionary_learning.rs:530-536`)
  vs SVD init; hand-rolled soft-threshold CD `lasso_cd_single` (fn at line 271) vs
  LARS lasso; normal-equations LS dict update `D[:,j] = solve((AᵀA + 1e-10·I),
  AᵀX[:,j])` + normalise (`dictionary_learning.rs:554-580`) vs `_update_dict` BCD +
  resampling. Probe 1 sklearn `components_ row0 = [0.487906, 0.566806, 0.650717,
  0.131324]` is NOT reproduced element-wise. **CARVE-OUT (R-DEFER-3):** matching
  sklearn requires reimplementing the SVD init, the LARS lasso solver, and
  `_update_dict`'s BCD + numpy-`RandomState` resampling; no failing test is asserted
  (same class as the `minibatch_nmf` / `sparse_pca` RNG carve-outs). **INVESTIGATE
  (for the critic):** sklearn's `transform` (OMP / lasso_cd) is DETERMINISTIC given a
  FIXED dictionary, so a transform value pin against an injected dictionary would be
  fixable — but ferrolearn's struct fields are PRIVATE and there is no API to
  construct a `FittedDictionaryLearning` from an arbitrary `D`, so transform value
  parity is gated on `components_` (this carve-out). The critic decides whether to add
  a constructor for an injectable-dictionary transform pin.

- REQ-5: **`fit_algorithm="lars"` (LARS) + the `cd` option (NOT-STARTED;
  `#1514`).** sklearn's `DictionaryLearning(fit_algorithm="lars")`
  (`_dict_learning.py:1595`, default `"lars"`, constraint `StrOptions({"lars","cd"})`
  `:1570`) sets `method = "lasso_" + fit_algorithm` (`:1671`, default →
  `"lasso_lars"`), an LARS-solved lasso sparse-coding step in `_dict_learning`
  (`:620`). ferrolearn's `DictFitAlgorithm` enum (`dictionary_learning.rs:47`) has a
  SINGLE variant `CoordinateDescent` (no `Lars`), and `fn fit` always sparse-codes via
  `lasso_cd_single` (fn at line 271) — there is no LARS path, and the default
  `fit_algorithm` is CD (not `"lars"` as in sklearn).

- REQ-6: **SVD-based `dict_init` / `code_init` init (NOT-STARTED; `#1515`).**
  sklearn's `_dict_learning` (`_dict_learning.py:554`) defaults (`dict_init=None`,
  `code_init=None`) to the SVD-based init `code, S, dictionary = linalg.svd(X,
  full_matrices=False)` + `svd_flip` + `S[:, np.newaxis] * dictionary`
  (`:581-584`) — DETERMINISTIC given `X` — and supports warm-restart via the
  `code_init`/`dict_init` ctor params (`:1600-1601`, `:576-579`). ferrolearn inits the
  dictionary from a random Gaussian `Normal(0,1)` via
  `Xoshiro256PlusPlus::seed_from_u64(random_state.unwrap_or(0))` + `normalise_dictionary`
  (`dictionary_learning.rs:530-536`) — NOT SVD-based, and has no `code_init`/`dict_init`
  fields.

- REQ-7: **`_update_dict` BCD + unused-atom resampling + unit-BALL projection
  (NOT-STARTED; `#1516`).** sklearn's `_update_dict` (`_dict_learning.py:474-551`)
  updates each atom by block-coordinate descent `dictionary[k] += (B[:,k] −
  A[k]@dictionary)/A[k,k]` (`:531`), and for a near-unused atom (`A[k,k] ≤ 1e-6`)
  RESAMPLES it from a random data point `newd = Y[random_state.choice(n_samples)]`
  plus noise (`:534-540`) using a numpy `RandomState`, then projects onto the unit
  BALL `dictionary[k] /= max(linalg.norm(dictionary[k]), 1)` (`:548`). ferrolearn
  updates the whole dictionary by normal-equations least squares `D[:,j] =
  solve((AᵀA + 1e-10·I), AᵀX[:,j])` (`dictionary_learning.rs:554-578`) — NO per-atom
  BCD, NO unused-atom resampling (so a dead atom stays dead), and `normalise_dictionary`
  (`:580`) projects onto the unit SPHERE (`/= norm`), not the unit ball.

- REQ-8: **`sparse_encode` + transform-algorithm coverage
  (SHIPPED scoped / residual open; `#1517`).** sklearn's
  `transform_algorithm` (`_dict_learning.py:1596`, default `"omp"`, constraint
  `StrOptions({"lasso_lars","lasso_cd","lars","omp","threshold"})` `:1571-1573`)
  selects among five sparse coders in `sparse_encode`, and `transform_alpha`
  (`:1598`, defaulting to `alpha` `_dict_learning.py:1115-1116`) sets the L1 penalty /
  threshold for the lasso / threshold algorithms. ferrolearn now exposes
  `sparse_encode` for dense f64 `Omp`, `LassoCd`, and `Threshold`, and
  `FittedDictionaryLearning::transform` delegates through the same fixed-dictionary
  path. Residual gaps: no `lasso_lars`/`lars`, no `transform_alpha` field, no positive
  coding, no precomputed `gram`/`cov`, no `init`, and no `n_jobs`.

- REQ-9: **`split_sign` (NOT-STARTED; `#1518`).** sklearn's
  `DictionaryLearning(split_sign=False)` (`_dict_learning.py:1604`), when `True`,
  splits the sparse code into its positive and negative parts
  (`split_code[:, :n] = max(code, 0)`, `split_code[:, n:] = -min(code, 0)`,
  `_BaseSparseCoding._transform` `:1131-1137`), doubling the transformed feature
  count. ferrolearn's `transform` (`impl Transform for FittedDictionaryLearning`, fn
  at line 622) has NO `split_sign` field — it returns the raw `(n_samples,
  n_components)` codes only.

- REQ-10: **`positive_code` / `positive_dict` constraints (NOT-STARTED;
  `#1519`).** sklearn's `DictionaryLearning(positive_code=False,
  positive_dict=False)` (`_dict_learning.py:1606-1607`) optionally enforces
  non-negativity on the codes (threaded into `sparse_encode`, `positive=positive_code`
  `:1697`/`:627`) and on the dictionary atoms (`np.clip(dictionary[k], 0, None)` in
  `_update_dict` `:544-545`), gated by `_check_positive_coding` (`:30-34`/`:1669`).
  ferrolearn's `DictionaryLearning` (`dictionary_learning.rs`, struct at line 70) has
  NO `positive_code`/`positive_dict` fields and applies no non-negativity constraint.

- REQ-11: **`transform_max_iter` (NOT-STARTED; `#1520`).** sklearn's
  `DictionaryLearning(transform_max_iter=1000)` (`_dict_learning.py:1608`) caps the
  inner iterations of the `lasso_cd`/`lasso_lars` transform sparse coder (passed as
  `max_iter` to `sparse_encode`, `_dict_learning.py:1126`/`:1688`). ferrolearn's
  `transform` hard-codes `200` lasso-CD iterations for LassoCd
  (`lasso_cd_single(&self.components_, self.alpha_, 200)`,
  `dictionary_learning.rs:643`) — no `transform_max_iter` field.

- REQ-12: **`MiniBatchDictionaryLearning` (online variant) (NOT-STARTED;
  `#1521`).** sklearn's `class MiniBatchDictionaryLearning(_BaseSparseCoding,
  BaseEstimator)` (`_dict_learning.py:1715`) is the online / mini-batch variant
  (`batch_size`, `max_no_improvement`, `shuffle`, `partial_fit`) for scalable
  dictionary learning. It is ABSENT in ferrolearn (`grep -rn
  MiniBatchDictionaryLearning ferrolearn-decomp/src/` is empty) — no type, no online
  dictionary learner.

- REQ-13: **Fitted attrs `error_` / `n_components_` / `n_features_in_`
  (NOT-STARTED; `#1522`).** sklearn exposes `error_` (the per-iteration error vector
  `_dict_learning.py:1504`/`:1700`; Probe 1 `error_ len = 11`), `n_features_in_`
  (`:1507`; Probe 1 `n_features_in_ = 4`), and `n_components_` (the `_n_features_out`
  property `:1704-1707`). `FittedDictionaryLearning` (`dictionary_learning.rs`, struct
  at line 210) exposes only `components()` / `n_iter()` / `reconstruction_err()` (fns
  at lines 229/235/241) — it stores a SCALAR `reconstruction_err_` (the final
  Frobenius error), NOT sklearn's per-iteration `error_` VECTOR, and has no
  `n_components_` / `n_features_in_` accessors.

- REQ-14: **Generic `F: Float` (f32 + f64) (NOT-STARTED; `#1523`).** Per CLAUDE.md
  (Numeric Generics: `F: Float + Send + Sync + 'static`, support both f32 and f64),
  the estimator should be generic. ferrolearn's `DictionaryLearning` is **f64-ONLY**:
  the struct fields are `f64` (`dictionary_learning.rs:73-78`), and the trait impl is
  `impl Fit<Array2<f64>, ()> for DictionaryLearning` (`dictionary_learning.rs:483`) /
  `impl Transform<Array2<f64>>` (`:612`) — not `<F>`-generic (contrast the
  sibling `MiniBatchNMF<F>` / `SparsePCA<F>`). sklearn's `_more_tags`
  (`_dict_learning.py:1709-1711`) declares `preserves_dtype: [np.float64,
  np.float32]`.

- REQ-15: **PyO3 binding (NOT-STARTED; `#1524`).** sklearn exposes
  `DictionaryLearning` through `import sklearn.decomposition`. ferrolearn has NO PyO3
  binding for `DictionaryLearning` — a `grep -rn DictionaryLearning
  ferrolearn-python/src/` is empty; the only non-test consumer of
  `DictionaryLearning`/`FittedDictionaryLearning` is the crate re-export
  (`lib.rs:84`). The CPython surface (a `_RsDictionaryLearning` class with a ctor +
  `fit` + `transform`) is absent.

- REQ-16: **ferray substrate (NOT-STARTED; `#1525`).** `dictionary_learning.rs`
  computes on `ndarray::Array2` (`dictionary_learning.rs:36`) and uses
  `rand`/`rand_distr` + `rand_xoshiro` (`dictionary_learning.rs:37-39`,
  `Xoshiro256PlusPlus` + `Normal`) for init, with a hand-rolled Gaussian-elimination
  linear solve (`solve_symmetric`, fn at line 404), not `ferray-core` arrays /
  `ferray::random` / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never
literal-copied from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED scoped): `DictionaryLearning::new(5).with_max_iter(20)
  .with_random_state(42).fit(&X, &()).unwrap().components().dim()` is
  `(5, n_features)`; `transform(&X)` has shape `(n_samples, 5)`; an OMP transform with
  `transform_n_nonzero_coefs(2)` yields ≤ 2 non-zeros per row; `reconstruction_err()`
  is finite and `≥ 0`; `n_iter() > 0`. Pinned by `test_dictlearn_basic_shape`
  `(5,10)`, `test_dictlearn_transform_shape` `(20,5)`,
  `test_dictlearn_omp_nonzero_coefs`, `test_dictlearn_sparsity_of_codes`,
  `test_dictlearn_reconstruction_error_decreases`, `test_dictlearn_fitted_accessors`,
  `test_dictlearn_single_component`. (Structural shapes / sparsity-cap / finiteness /
  determinism only — NOT the exact component values, REQ-4.)

- AC-2 (REQ-2, SHIPPED scoped): every row of `fitted.components()` has L2 norm
  `≈ 1.0` (within 1e-6). Probe 1 confirms sklearn atoms are unit-norm. Pinned by
  `test_dictlearn_dictionary_atoms_normalised`. FLAG: sklearn projects onto the unit
  BALL (`max(norm, 1)`) whereas ferrolearn normalises to the unit SPHERE (`/= norm`)
  — coincident for a converged dictionary (folds into REQ-7).

- AC-3 (REQ-3, SHIPPED scoped): `fit` returns `Err` for `n_components=0`,
  `n_samples=0`, `n_features=0`, and `alpha < 0`; `transform` returns `Err` for a
  column-count mismatch. Pinned by `test_dictlearn_invalid_n_components_zero`,
  `test_dictlearn_empty_data`, `test_dictlearn_zero_features`,
  `test_dictlearn_invalid_alpha_negative`, `test_dictlearn_transform_shape_mismatch`.
  FLAG: sklearn raises `InvalidParameterError` (not `FerroError`), accepts
  `n_components=None`, and does not pre-reject `0` features.

- AC-4 (REQ-4, NOT-STARTED, CARVE-OUT): `DictionaryLearning(n_components=3, alpha=1,
  random_state=0).fit(X).components_` (Probe 1: shape `(3,4)`, `row0 = [0.487906,
  0.566806, 0.650717, 0.131324]`) is NOT reproduced element-wise by ferrolearn
  (random-Gaussian-`Xoshiro` init + soft-threshold CD + normal-equations LS update
  vs sklearn SVD init + LARS lasso + `_update_dict` BCD + numpy RNG resampling). No
  failing test asserts this (R-DEFER-3).

- AC-5 (REQ-5..13, DIVERGES): `DictionaryLearning()` defaults `n_components=None,
  alpha=1, max_iter=1000, tol=1e-8, fit_algorithm="lars",
  transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None,
  split_sign=False, positive_code=False, positive_dict=False, transform_max_iter=1000`
  (Probe 3, `_dict_learning.py:1588-1629`); `fit_algorithm` default `"lars"` →
  `method="lasso_lars"`; sklearn exposes the `error_` vector,
  `n_components_`/`n_features_in_`, and the `MiniBatchDictionaryLearning` class.
  ferrolearn has `fit_algorithm` CD-only (no LARS, default CD),
  `transform_algorithm` `Omp`/`LassoCd`/`Threshold` (no `lasso_lars`/`lars`), no
  `transform_alpha`/`split_sign`/`positive_code`/`positive_dict`/`transform_max_iter`/
  `code_init`/`dict_init`, a scalar `reconstruction_err_` (not the `error_` vector),
  no `n_components_`/`n_features_in_`, and no `MiniBatchDictionaryLearning`.

- AC-6 (REQ-14/15/16): `DictionaryLearning` is f64-only (`impl Fit<Array2<f64>, ()>`,
  `dictionary_learning.rs:483`), not `<F>`-generic; `import ferrolearn` exposes NO
  `_RsDictionaryLearning` (`grep -rn DictionaryLearning ferrolearn-python/src/` is
  empty); the only non-test consumer is the crate re-export (`lib.rs:84`). The module
  imports `ndarray` (`dictionary_learning.rs:36`) + `rand`/`rand_distr`/`rand_xoshiro`
  (`:37-39`), not ferray.

`## REQ status`

## REQ status

Binary (R-DEFER-2). `DictionaryLearning` / `FittedDictionaryLearning` are existing
pub APIs; the non-test consumer is the crate re-export (`lib.rs:84`, boundary public
API, grandfathered S5/R-DEFER-1) — there is NO PyO3 binding (REQ-15 NOT-STARTED).
Cites use symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle =
installed sklearn 1.5.2, run from `/tmp`.
**EXACT `components_` VALUE PARITY DIVERGES (R-HONEST-3, REQ-4 NOT-STARTED, CARVE-OUT
`#1513`):** ferrolearn's random-Gaussian `Xoshiro` init + soft-threshold CD
`lasso_cd_single` + normal-equations LS dict update + normalise
(`dictionary_learning.rs:495`) ≠ sklearn's SVD init + LARS lasso (`fit_algorithm=
"lars"` → `"lasso_lars"`) + `_update_dict` BCD with numpy-`RandomState` atom
resampling (`_dict_learning.py:554-674`/`:474-551`). The least-confident SHIPPED claim
is REQ-2 — it is STRUCTURAL unit-norm (the in-tree test asserts each atom's L2 norm
≈ 1, not oracle component parity); it also carries a FLAG that sklearn projects onto
the unit BALL (`max(norm, 1)`) whereas ferrolearn uses the unit SPHERE (`/= norm`),
coincident only for a converged dictionary. #1512 is this doc's crosslink tracking
issue. Count: **3 SHIPPED (REQ-1,2,3) + REQ-8 sparse_encode/threshold scoped /
residual gaps open (REQ-4,5,6,7,8,9,10,11,12,13,14,15,16)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (structural: component/code shapes / OMP-cap / finite err / n_iter / determinism) | SHIPPED | `fn fit` (`dictionary_learning.rs` impl at line 495) stores `components_ = d` shape `(n_components, n_features)` (`dictionary_learning.rs:601-602`, field `:213` = sklearn `components_ = U` `_dict_learning.py:1699`), Frobenius `reconstruction_err_` (`reconstruction_error` fn at line 469; `:605`), `n_iter_` (`:604`). `transform` (`impl Transform for FittedDictionaryLearning` fn at line 622) returns `(n_samples, n_components)` codes; OMP `omp_single` (fn at line 327) caps the active set at `max_nonzero` (`:333-335`), LassoCd `lasso_cd_single` (fn at line 271) yields sparse codes. Seeded `Xoshiro256PlusPlus` (`:530`, seed `unwrap_or(0)` `:526`) ⇒ reproducible. **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:84`. Verification: `cargo test -p ferrolearn-decomp dictlearn` → `test_dictlearn_basic_shape` `(5,10)`, `test_dictlearn_transform_shape` `(20,5)`, `test_dictlearn_omp_nonzero_coefs`, `test_dictlearn_sparsity_of_codes`, `test_dictlearn_reconstruction_error_decreases`, `test_dictlearn_fitted_accessors`, `test_dictlearn_single_component` PASS. |
| REQ-2 (structural: dictionary atoms UNIT L2-NORM) | SHIPPED | `normalise_dictionary` (fn at line 251) divides each atom row by its L2 norm (`dictionary_learning.rs:254-265`); `fn fit` calls it at init (`:536`) and after every dict update (`:580`), so stored `components_` rows are unit-norm — sklearn unit-ball constraint `||V_k||₂ ≤ 1` (`_dict_learning.py:1382`), `_update_dict` `dictionary[k] /= max(linalg.norm(dictionary[k]), 1)` (`:548`). Probe 1 sklearn atom L2 norms `[1.0,1.0,1.0]`. **Scope: STRUCTURAL, NOT value parity (REQ-4).** Non-test consumer: re-export `lib.rs:84`. Verification: `cargo test -p ferrolearn-decomp dictlearn` → `test_dictlearn_dictionary_atoms_normalised` PASS. **FLAG (candidate DIV):** sklearn projects onto the unit BALL (`max(norm, 1)`); ferrolearn always normalises to the unit SPHERE (`/= norm`) — coincident for a converged dictionary (folds into REQ-7). |
| REQ-3 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`dictionary_learning.rs` impl at line 495) returns `Err(InvalidParameter{name:"n_components", reason:"must be at least 1"})` for `n_components==0` (`:499-504`), `Err(InsufficientSamples{required:1,...})` for `0` samples (`:505-511`), `Err(InvalidParameter{name:"X", ... "at least 1 feature"})` for `0` features (`:512-517`), `Err(InvalidParameter{name:"alpha", ... "non-negative"})` for `alpha<0` (`:518-523`); `transform` returns `Err(ShapeMismatch)` on column mismatch (`:624-630`). Non-test consumer: re-export `lib.rs:84`. Verification: `cargo test -p ferrolearn-decomp dictlearn` (`test_dictlearn_invalid_n_components_zero`, `_empty_data`, `_zero_features`, `_invalid_alpha_negative`, `_transform_shape_mismatch`) PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_dict_learning.py:1565-1586`) raising `InvalidParameterError` (not `FerroError`); accepts `n_components=None` → `n_features` (`:1676-1679`); does NOT pre-reject `0` features. |
| REQ-4 (EXACT `components_` value parity with `_dict_learning`) | NOT-STARTED | open prereq blocker **#1513** (CARVE-OUT, R-DEFER-3). sklearn `fit_transform` (`_dict_learning.py:1651-1702`) → `_dict_learning` (`:554-674`): SVD init `linalg.svd(X)` + `svd_flip` (`:581-584`, deterministic), LARS lasso `method="lasso_lars"` (`:1671`/`:620`), `_update_dict` BCD `dictionary[k] += (B[:,k]−A[k]@dictionary)/A[k,k]` (`:531`) + unused-atom resampling `Y[random_state.choice(n_samples)]` (`:534-540`, numpy `RandomState`); `components_ = U` (`:1699`). ferrolearn `fn fit` (`dictionary_learning.rs` impl at line 495): DIFFERENT on all three — random-Gaussian `Xoshiro` init + normalise (`:530-536`), soft-threshold CD `lasso_cd_single` (fn at line 271), normal-equations LS `D[:,j]=solve((AᵀA+1e-10·I), AᵀX[:,j])` + normalise (`:554-580`). Probe 1 sklearn `components_ row0 = [0.487906,0.566806,0.650717,0.131324]` NOT reproduced. No failing test (same class as `minibatch_nmf` / `sparse_pca` RNG carve-outs). **INVESTIGATE (critic):** transform (OMP/lasso_cd) is deterministic given a FIXED dictionary, but ferrolearn exposes no API to inject an arbitrary `D` (struct fields private), so transform value parity is gated on `components_` (this carve-out). |
| REQ-5 (`fit_algorithm="lars"` LARS + `cd` option) | NOT-STARTED | open prereq blocker **#1514**. sklearn `DictionaryLearning(fit_algorithm="lars")` (`_dict_learning.py:1595`, `StrOptions({"lars","cd"})` `:1570`) → `method = "lasso_" + fit_algorithm` (`:1671`, default `"lasso_lars"`) → LARS lasso in `sparse_encode` (`:620`). ferrolearn `DictFitAlgorithm` (`dictionary_learning.rs:47`) has a SINGLE `CoordinateDescent` variant (no `Lars`); `fn fit` always sparse-codes via `lasso_cd_single` (fn at line 271) — no LARS path, default is CD not `"lars"`. |
| REQ-6 (SVD-based `dict_init`/`code_init` init) | NOT-STARTED | open prereq blocker **#1515**. sklearn `_dict_learning` (`_dict_learning.py:554`) defaults to SVD init `code, S, dictionary = linalg.svd(X, full_matrices=False)` + `svd_flip` + `S[:,np.newaxis]*dictionary` (`:581-584`, deterministic given X) and supports warm-restart `code_init`/`dict_init` (`:1600-1601`,`:576-579`). ferrolearn inits a random Gaussian `Normal(0,1)` via `Xoshiro256PlusPlus::seed_from_u64(random_state.unwrap_or(0))` + `normalise_dictionary` (`dictionary_learning.rs:530-536`) — NOT SVD, no `code_init`/`dict_init` fields. |
| REQ-7 (`_update_dict` BCD + atom resampling + unit-BALL projection) | NOT-STARTED | open prereq blocker **#1516**. sklearn `_update_dict` (`_dict_learning.py:474-551`): per-atom BCD `dictionary[k] += (B[:,k]−A[k]@dictionary)/A[k,k]` (`:531`), unused-atom (`A[k,k]≤1e-6`) RESAMPLE `newd = Y[random_state.choice(n_samples)] + noise` (`:534-540`, numpy `RandomState`), unit-BALL projection `/= max(norm, 1)` (`:548`). ferrolearn updates the whole `D` by normal-equations LS `D[:,j]=solve((AᵀA+1e-10·I), AᵀX[:,j])` (`dictionary_learning.rs:554-578`) — NO per-atom BCD, NO unused-atom resampling, and `normalise_dictionary` (`:580`) projects onto the unit SPHERE (`/= norm`). |
| REQ-8 (`sparse_encode` + transform algorithm coverage) | SHIPPED scoped / residual open | open prereq blocker **#1517** remains for the residual. sklearn `transform_algorithm` (`_dict_learning.py:1596`, `StrOptions({"lasso_lars","lasso_cd","lars","omp","threshold"})` `:1571-1573`) + `transform_alpha` (`:1598`, defaults to `alpha` `:1115-1116`) select among five `sparse_encode` coders. ferrolearn `sparse_encode` and `DictTransformAlgorithm` now cover dense f64 `Omp`, `LassoCd`, and `Threshold`, with value pins in `tests/divergence_sparse_encode.rs`. Residual gaps: no `lasso_lars`/`lars`, no `transform_alpha` field, no positive coding, no precomputed `gram`/`cov`, no `init`, and no `n_jobs`. |
| REQ-9 (`split_sign`) | NOT-STARTED | open prereq blocker **#1518**. sklearn `DictionaryLearning(split_sign=False)` (`_dict_learning.py:1604`), when True, splits the code into positive/negative parts (`split_code[:,:n]=max(code,0)`, `split_code[:,n:]=-min(code,0)`, `_BaseSparseCoding._transform` `:1131-1137`), doubling the output features. ferrolearn `transform` (`impl Transform for FittedDictionaryLearning` fn at line 622) has NO `split_sign` field — raw `(n_samples, n_components)` codes only. |
| REQ-10 (`positive_code` / `positive_dict`) | NOT-STARTED | open prereq blocker **#1519**. sklearn `DictionaryLearning(positive_code=False, positive_dict=False)` (`_dict_learning.py:1606-1607`) enforces non-negativity on codes (`sparse_encode(positive=positive_code)` `:1697`/`:627`) and dict atoms (`np.clip(dictionary[k],0,None)` `:544-545`), gated by `_check_positive_coding` (`:30-34`/`:1669`). ferrolearn `DictionaryLearning` (`dictionary_learning.rs` struct at line 70) has NO `positive_code`/`positive_dict` fields, no non-negativity constraint. |
| REQ-11 (`transform_max_iter`) | NOT-STARTED | open prereq blocker **#1520**. sklearn `DictionaryLearning(transform_max_iter=1000)` (`_dict_learning.py:1608`) caps the `lasso_cd`/`lasso_lars` transform inner iterations (`sparse_encode(max_iter=...)` `:1126`/`:1688`). ferrolearn `transform` hard-codes `200` lasso-CD iters (`lasso_cd_single(&self.components_, self.alpha_, 200)` `dictionary_learning.rs:643`) — no `transform_max_iter` field. |
| REQ-12 (`MiniBatchDictionaryLearning`) | NOT-STARTED | open prereq blocker **#1521**. sklearn `class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator)` (`_dict_learning.py:1715`) is the online / mini-batch variant (`batch_size`, `max_no_improvement`, `shuffle`, `partial_fit`). ABSENT in ferrolearn (`grep -rn MiniBatchDictionaryLearning ferrolearn-decomp/src/` is empty) — no type, no online learner. |
| REQ-13 (fitted attrs `error_` / `n_components_` / `n_features_in_`) | NOT-STARTED | open prereq blocker **#1522**. sklearn exposes `error_` (per-iteration error VECTOR `_dict_learning.py:1504`/`:1700`; Probe 1 len 11), `n_features_in_` (`:1507`; Probe 1 = 4), `n_components_` (`_n_features_out` `:1704-1707`). `FittedDictionaryLearning` (`dictionary_learning.rs` struct at line 210) exposes only `components()`/`n_iter()`/`reconstruction_err()` (fns at 229/235/241) — stores a SCALAR `reconstruction_err_` (final Frobenius error), NOT the per-iteration `error_` vector, and no `n_components_`/`n_features_in_`. |
| REQ-14 (generic `F: Float`, f32 + f64) | NOT-STARTED | open prereq blocker **#1523**. CLAUDE.md mandates `F: Float + Send + Sync + 'static` (f32 + f64); sklearn `_more_tags` `preserves_dtype: [np.float64, np.float32]` (`_dict_learning.py:1709-1711`). ferrolearn is **f64-ONLY**: struct fields `f64` (`dictionary_learning.rs:73-78`), `impl Fit<Array2<f64>, ()>` (`:483`) / `impl Transform<Array2<f64>>` (`:612`) — not `<F>`-generic (contrast `MiniBatchNMF<F>` / `SparsePCA<F>`). |
| REQ-15 (PyO3 binding) | NOT-STARTED | open prereq blocker **#1524**. sklearn exposes `DictionaryLearning` via `import sklearn.decomposition`. ferrolearn has NO PyO3 binding — `grep -rn DictionaryLearning ferrolearn-python/src/` is empty; the only non-test consumer of `DictionaryLearning`/`FittedDictionaryLearning` is the crate re-export (`lib.rs:84`). No `_RsDictionaryLearning` class. |
| REQ-16 (ferray substrate) | NOT-STARTED | open prereq blocker **#1525**. `dictionary_learning.rs` computes on `ndarray::Array2` (`dictionary_learning.rs:36`), uses `rand`/`rand_distr`/`rand_xoshiro` `Xoshiro256PlusPlus`+`Normal` (`:37-39`) for init, and a hand-rolled Gaussian-elimination solve (`solve_symmetric` fn at line 404), not `ferray-core` arrays / `ferray::random` / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`dictionary_learning.rs` follows the unfitted/fitted split (CLAUDE.md naming):
`DictionaryLearning { n_components, alpha (1.0), max_iter (1000), tol (1e-8),
fit_algorithm: DictFitAlgorithm{CoordinateDescent}, transform_algorithm:
DictTransformAlgorithm{Omp|LassoCd}, transform_n_nonzero_coefs, random_state }`
(struct at line 70; `new(n_components)` fn at line 95, builders `with_alpha` `:110` /
`with_max_iter` `:117` / `with_tol` `:124` / `with_fit_algorithm` `:131` /
`with_transform_algorithm` `:138` / `with_transform_n_nonzero_coefs` `:145` /
`with_random_state` `:152`, accessors `n_components()`..`random_state()` fns at lines
159-197) → `Fit<Array2<f64>, ()>` → `FittedDictionaryLearning { components_, alpha_,
n_iter_, reconstruction_err_, transform_algorithm_, transform_n_nonzero_coefs_ }`
(struct at line 210, accessors `components()`/`n_iter()`/`reconstruction_err()` fns at
lines 229/235/241). The path is **f64-ONLY** (`impl Fit<Array2<f64>, ()>` `:483`,
`impl Transform<Array2<f64>>` `:612`) — NOT `<F>`-generic (REQ-14 NOT-STARTED);
`fit`/`transform` return `Result<_, FerroError>` (R-CODE-2). NOTE: the init uses
`Normal::new(0.0, 1.0).unwrap()` (`dictionary_learning.rs:531`) in production — an
`unwrap()` outside `#[cfg(test)]` (R-CODE-2 / R-APG-1 flag), surfaced for the critic.

**Fit path (`fn fit`, impl at line 495) — REQ-1/2/3/4.** Validates `n_components !=
0`, `n_samples >= 1`, `n_features >= 1`, `alpha >= 0`
(`dictionary_learning.rs:499-523`) — REQ-3. Inits the dictionary `D` (n_components,
n_features) from a seeded `Xoshiro256PlusPlus` Gaussian `Normal(0,1)`, then
`normalise_dictionary` to unit L2 (`dictionary_learning.rs:530-536`) — NOT sklearn's
SVD init (REQ-6 NOT-STARTED). Outer loop to `max_iter` (`:541`): (a) sparse-coding —
for each sample, `lasso_cd_single` (fn at line 271, `min_a 0.5||x − Dᵀa||² +
alpha||a||₁` via soft-threshold CD, 200 inner iters) builds the codes `A`
(`:545-552`) — the CD coder is REQ-5 NOT-STARTED (sklearn uses LARS lasso); (b)
dictionary update — solve the normal equations `D[:,j] = solve((AᵀA + 1e-10·I),
AᵀX[:,j])` per feature column via `solve_symmetric` (fn at line 404), then
`normalise_dictionary` (`:554-580`) — REQ-7 NOT-STARTED (sklearn does `_update_dict`
BCD + atom resampling + unit-ball); (c) break on `|prev_err − err| < tol` (`:584`).
A final sparse-coding pass computes `reconstruction_err_` (`:591-599`). Stores
`components_ = d`, `alpha_`, `n_iter_`, `reconstruction_err_`, `transform_algorithm_`,
`transform_n_nonzero_coefs_` (`:601-608`). **This is NOT sklearn's `_dict_learning`
(REQ-4):** sklearn SVD-inits, LARS-lasso sparse-codes, and `_update_dict`-BCDs with
numpy-RNG atom resampling (`_dict_learning.py:554-674`); ferrolearn's
random-Gaussian-`Xoshiro` init + soft-threshold-CD + normal-equations-LS update
produce DIFFERENT component values (CARVE-OUT).

**Transform (`impl Transform for FittedDictionaryLearning`, fn at line 622) —
REQ-1/8 scoped.** Validates the column count (`:624-630` — REQ-3), then delegates to
the same fixed-dictionary sparse coding path as public `sparse_encode`: OMP capped by
`transform_n_nonzero_coefs_`, LassoCd with the fit `alpha_`, or Threshold with the fit
`alpha_`, returning codes `(n_samples, n_components)`. This is sklearn's
`_BaseSparseCoding._transform` → `sparse_encode` (`_dict_learning.py:1110-1139`)
restricted to `omp`/`lasso_cd`/`threshold` — NO `split_sign` (REQ-9), NO
`transform_alpha`, NO `lasso_lars`/`lars`, NO positive coding / precomputed
`gram`/`cov` / `init` / `n_jobs` (REQ-8 residual), and NO `transform_max_iter`
(REQ-11, hard-coded 200). `tests/divergence_sparse_encode.rs` pins public
`sparse_encode` against sklearn for a fixed dictionary.

**sklearn (target contract).** `class DictionaryLearning(_BaseSparseCoding,
BaseEstimator)` (`_dict_learning.py:1372`) takes `__init__(n_components=None, *,
alpha=1, max_iter=1000, tol=1e-8, fit_algorithm="lars", transform_algorithm="omp",
transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None,
dict_init=None, callback=None, verbose=False, split_sign=False, random_state=None,
positive_code=False, positive_dict=False, transform_max_iter=1000)` (`:1588-1629`).
`fit_transform` (`:1651`) sets `method = "lasso_" + fit_algorithm` (`:1671`,
`"lasso_lars"` by default), defaults `n_components = n_features` if `None`
(`:1676-1679`), and calls `_dict_learning` (`:1681-1698`) → `V` (codes), `U` (dict),
`E` (error vector), `n_iter_`; stores `components_ = U` (`:1699`), `error_ = E`
(`:1700`). `_dict_learning` (`:554`) SVD-inits (`:581-584`), alternates `sparse_encode`
(`:620`, LARS/CD lasso) and `_update_dict` (`:632-640`, BCD + atom resampling).
`transform` (`:1141`) is `sparse_encode(X, components_, transform_algorithm, ...)` +
optional `split_sign`. `MiniBatchDictionaryLearning` (`:1715`) is the online variant.
Fitted attrs: `components_`, `error_`, `n_features_in_`, `n_iter_`, `n_components_`
(`_n_features_out`).

**The remaining gap.** ferrolearn ships the STRUCTURAL component/code shapes, OMP
n-nonzero cap and LassoCd sparsity, finiteness & determinism (REQ-1), the unit-L2
dictionary atoms (REQ-2), and the scoped error & parameter contracts (REQ-3). It
lacks: exact `components_` value parity (REQ-4, CARVE-OUT `#1513`);
`fit_algorithm="lars"`/LARS (REQ-5, `#1514`); SVD-based `dict_init`/`code_init`
init (REQ-6, `#1515`); `_update_dict` BCD + atom resampling (REQ-7, `#1516`);
`transform_alpha` + `lasso_lars`/`lars` + positive/precomputed sparse_encode paths
(REQ-8 residual, `#1517`); `split_sign` (REQ-9, `#1518`);
`positive_code`/`positive_dict` (REQ-10, `#1519`);
`transform_max_iter` (REQ-11, `#1520`); `MiniBatchDictionaryLearning` (REQ-12,
`#1521`); `error_`/`n_components_`/`n_features_in_` attrs (REQ-13, `#1522`);
generic `F` (REQ-14, `#1523`); the PyO3 binding (REQ-15, `#1524`); and the ferray
substrate (REQ-16, `#1525`). This is a
**structure-SHIPPED plus sparse_encode-scoped** unit (3 SHIPPED + REQ-8 scoped /
residual gaps open).

## Verification

Library crate (green at baseline `7443a1a9`):
```bash
cargo test -p ferrolearn-decomp dictlearn                  # in-module #[test]s + doctest
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```
The in-tree `#[test]`s pin REQ-1/2/3 (STRUCTURAL): `test_dictlearn_basic_shape`
`(5,10)`, `test_dictlearn_transform_shape` `(20,5)`,
`test_dictlearn_reconstruction_error_decreases`, `test_dictlearn_sparsity_of_codes`,
`test_dictlearn_omp_transform`, `test_dictlearn_lasso_cd_transform`,
`test_dictlearn_omp_nonzero_coefs` (≤2 nnz/row), `test_dictlearn_single_component`,
`test_dictlearn_fitted_accessors` (REQ-1); `test_dictlearn_dictionary_atoms_normalised`
(REQ-2); `test_dictlearn_invalid_n_components_zero`, `test_dictlearn_empty_data`,
`test_dictlearn_zero_features`, `test_dictlearn_invalid_alpha_negative`,
`test_dictlearn_transform_shape_mismatch` (REQ-3); plus `test_soft_threshold`,
`test_dictlearn_getters`, and the module doctest. There is NO
`tests/divergence_dictionary_learning.rs` yet. REQ-4 (`components_` value parity) is a
CARVE-OUT (R-DEFER-3) with NO failing test.

Live sklearn oracle (installed 1.5.2, run from `/tmp`) — the REQ-1/2 structure and
the REQ-4 components value gap:
```bash
# REQ-1/2 structural (shape, unit-norm atoms) + REQ-4 value gap (values NOT reproduced):
python3 -c "import numpy as np; from sklearn.decomposition import DictionaryLearning
X=np.array([[1.,2,3,0],[4,5,6,1],[7,8,9,2],[2,1,0,3],[0,3,4,1],[3,0,1,2]])
m=DictionaryLearning(n_components=3, alpha=1, random_state=0).fit(X)
print(m.components_.shape, [round(float(np.linalg.norm(r)),6) for r in m.components_], np.round(m.components_[0],6).tolist())"
# -> (3, 4) [1.0, 1.0, 1.0] [0.487906, 0.566806, 0.650717, 0.131324]

# REQ-1 transform sparse codes + OMP n-nonzero cap:
python3 -c "import numpy as np; from sklearn.decomposition import DictionaryLearning
X=np.array([[1.,2,3,0],[4,5,6,1],[7,8,9,2],[2,1,0,3],[0,3,4,1],[3,0,1,2]])
m=DictionaryLearning(n_components=3, alpha=1, transform_algorithm='omp', transform_n_nonzero_coefs=2, random_state=0).fit(X)
Xt=m.transform(X); print(Xt.shape, int(max((np.abs(r)>1e-12).sum() for r in Xt)))"
# -> (6, 3) 2
```
REQ-4 remains a CARVE-OUT (no parity test) — matching sklearn's `components_` needs
the SVD init + LARS lasso + `_update_dict` BCD + numpy-RNG atom resampling. The
transform (OMP/lasso_cd) IS deterministic given a fixed dictionary (INVESTIGATE for
the critic — gated on the carved-out `components_` since the struct fields are
private).

ferrolearn-python (REQ-15, ABSENT at baseline): there is NO `_RsDictionaryLearning`
binding — `grep -rn DictionaryLearning ferrolearn-python/src/` is empty. The only
non-test consumer of `DictionaryLearning`/`FittedDictionaryLearning` is the crate
re-export (`lib.rs:84`).

## Blockers

(#1512 is this doc's crosslink tracking issue. The blockers below are the open work
items the dispatcher files / numbers; none are filed by this doc — markdown only.)

- **#1513** — REQ-4 (CARVE-OUT): reimplement sklearn's `_dict_learning`
  (`_dict_learning.py:554-674`) — SVD init (`linalg.svd` + `svd_flip` `:581-584`),
  LARS lasso `sparse_encode` (`method="lasso_lars"` `:1671`/`:620`), and `_update_dict`
  BCD with numpy-`RandomState` unused-atom resampling (`:474-551`) — to reach EXACT
  `components_` value parity; inherently RNG/algorithm-bound (no failing test,
  R-DEFER-3). The transform-on-fixed-dictionary determinism is INVESTIGATE-for-critic
  (gated on this carve-out — no injectable-dictionary API).
- **#1514** — REQ-5: turn `DictFitAlgorithm` (`dictionary_learning.rs:47`) into
  `{Lars, Cd}` with `"lars"` default and an LARS-lasso sparse-coding path
  (`fit_algorithm` → `method="lasso_"+algo` `_dict_learning.py:1671`).
- **#1515** — REQ-6: replace the random-Gaussian `Xoshiro` init with the SVD-based
  `linalg.svd(X)` + `svd_flip` init (`_dict_learning.py:581-584`) and add
  `code_init`/`dict_init` warm-restart fields (`:1600-1601`).
- **#1516** — REQ-7: switch the dictionary update from normal-equations LS to
  sklearn's `_update_dict` (`_dict_learning.py:474-551`): per-atom BCD (`:531`),
  unused-atom resampling from a numpy `RandomState` (`:534-540`), and the unit-BALL
  projection `/= max(norm, 1)` (`:548`).
- **#1517** — REQ-8 residual: add a `transform_alpha` field (default `alpha`,
  `_dict_learning.py:1598`/`:1115-1116`), the `lasso_lars`/`lars`
  transform-algorithm variants (`_dict_learning.py:1571-1573`), and the positive /
  precomputed sparse_encode paths.
- **#1518** — REQ-9: add a `split_sign` field + the positive/negative code split
  (`_BaseSparseCoding._transform` `_dict_learning.py:1131-1137`).
- **#1519** — REQ-10: add `positive_code`/`positive_dict` fields + the
  non-negativity constraints in sparse coding (`sparse_encode(positive=...)`
  `_dict_learning.py:627`) and the dict update (`np.clip` `:544-545`), gated by
  `_check_positive_coding` (`:30-34`).
- **#1520** — REQ-11: add a `transform_max_iter` field
  (`_dict_learning.py:1608`) threaded into the lasso-CD/lasso-LARS transform inner
  iterations (currently hard-coded 200 `dictionary_learning.rs:643`).
- **#1521** — REQ-12: add a `MiniBatchDictionaryLearning` type mirroring the online
  variant (`_dict_learning.py:1715`, `batch_size`/`max_no_improvement`/`partial_fit`).
- **#1522** — REQ-13: store and expose the per-iteration `error_` VECTOR
  (`_dict_learning.py:1700`) plus `n_components_` / `n_features_in_` fitted attrs on
  `FittedDictionaryLearning` (`_dict_learning.py:1504`/`:1507`/`:1704-1707`).
- **#1523** — REQ-14: make `DictionaryLearning`/`FittedDictionaryLearning` generic
  over `F: Float + Send + Sync + 'static` (f32 + f64), replacing the f64-only `impl
  Fit<Array2<f64>, ()>` (`dictionary_learning.rs:483`) — CLAUDE.md / sklearn
  `preserves_dtype` (`_dict_learning.py:1709-1711`).
- **#1524** — REQ-15: add a `_RsDictionaryLearning` PyO3 binding in
  `ferrolearn-python` (ctor + `fit` + `transform`), registered in `lib.rs`, as the
  CPython consumer.
- **#1525** — REQ-16: migrate `dictionary_learning.rs` off `ndarray` +
  `rand`/`rand_distr`/`rand_xoshiro` + the hand-rolled `solve_symmetric`
  (`dictionary_learning.rs:404`) to `ferray-core` / `ferray::random` /
  `ferray::linalg` (R-SUBSTRATE).
