# IncrementalPCA (sklearn.decomposition.IncrementalPCA)

<!--
tier: 3-component
status: value-parity-shipped
baseline-commit: a5d211b9
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_incremental_pca.py  # class IncrementalPCA(_BasePCA) (:19). ctor (:194-198): n_components=None, *, whiten=False, copy=True, batch_size=None; _parameter_constraints (:187-192) n_components Interval[1,None] or None, whiten/copy boolean, batch_size Interval[1,None] or None. fit (:201-249): resets state (:218-225), validates accept_sparse csr/csc/lil + dtype float64/float32 (:227-233), batch_size_ = 5*n_features if None else batch_size (:236-239), gen_batches(n_samples, batch_size_, min_batch_size=n_components or 0) (:241-243), partial_fit per batch (:247). partial_fit (:252-373): col_mean, col_var, n_total_samples = _incremental_mean_and_var(X, last_mean=mean_, last_variance=var_, last_sample_count=repeat(n_samples_seen_)) (:329-334); n_components_ = min(n_samples,n_features) if None (:294) / must be <= n_features (:297-302) AND <= n_samples (:303-308). Whitening/stacking (:337-355): if n_samples_seen==0 -> X -= col_mean (:340); else col_batch_mean=mean(X,axis=0) (:342), X -= col_batch_mean (:343), mean_correction = sqrt((n_samples_seen/n_total)*n_samples)*(mean_ - col_batch_mean) (:345-347), X = vstack([singular_values_.reshape(-1,1)*components_, X, mean_correction]) (:348-354). U,S,Vt = linalg.svd(X, full_matrices=False) (:356); U,Vt = svd_flip(U,Vt,u_based_decision=False) (:357); explained_variance = S**2/(n_total-1) (:358); explained_variance_ratio = S**2/sum(col_var*n_total) (:359); components_ = Vt[:n_components_] (:362); singular_values_ = S[:n_components_] (:363); mean_ = col_mean (:364); var_ = col_var (:365); explained_variance_/_ratio_ truncated (:366-367); noise_variance_ = explained_variance[n_components_:].mean() unless n_components_ in (n_samples,n_features) -> 0.0 (:369-372). transform (:375-414) -> _BasePCA.transform: (X-mean_) @ components_.T (+ /sqrt(explained_variance_) if whiten); inverse_transform (_base.py): X @ components_*sqrt(exp_var) + mean_ (whiten) / X @ components_ + mean_.
  - sklearn/utils/extmath.py  # svd_flip(u, v, u_based_decision=True) (:848-906). u_based_decision=False branch (:897-905) operates on v (Vt) ROWS: max_abs_v_rows = argmax(abs(v), axis=1) (:899, numpy argmax -> FIRST max on ties); signs = sign(v[arange, max_abs]) (:902); v *= signs[:, newaxis] (:905) -> each component (Vt) row's max-abs entry becomes POSITIVE. _incremental_mean_and_var (:1057-1180): Youngs-and-Cramer running mean+variance update (Chan/Golub/LeVeque) -> updated_mean (:1137), updated_variance (:1178), updated_sample_count (:1135).
ferrolearn-module: ferrolearn-decomp/src/incremental_pca.rs
parity-ops: IncrementalPCA
crosslink-issue: 1584
-->

## Summary

`ferrolearn-decomp/src/incremental_pca.rs` mirrors scikit-learn's `IncrementalPCA`
(`sklearn/decomposition/_incremental_pca.py`, `class IncrementalPCA(_BasePCA)` `:19`):
PCA computed incrementally batch-by-batch via the sklearn incremental-SVD update, so a
dataset too large for a single PCA can be processed in mini-batches with constant memory.

**A critic→fixer→re-audit cycle (closing #1585/#1586/#1587) reimplemented
`fn partial_fit_batch` as a faithful port of sklearn's `partial_fit`
(`_incremental_pca.py:329-372`), and ferrolearn now has FULL VALUE PARITY with the live,
deterministic sklearn 1.5.2 `IncrementalPCA` on BOTH single- and multi-batch fits.** On a
fresh SINGLE-batch fixture (7×4, `n_components=2`, `batch_size=7`) `components_` (INCLUDING
sign) match to 2.2e-16, `explained_variance_` to 1.8e-15, `explained_variance_ratio_` to
2.2e-16, `singular_values_` to 1.3e-15, and `mean_`/`var_` exactly. On a fresh MULTI-batch
fixture (9×4, `batch_size=3`, three batches) `components_` (incl. sign) match to 6.3e-15,
`explained_variance_` to 2.8e-14, `singular_values_` to 2.0e-14, `explained_variance_ratio_`
to 2.4e-15, and `mean_`/`var_` exactly. The three previously-divergent points are FIXED:

1. **`svd_flip(u_based_decision=False)` sign (was #1585, FIXED).** `fn partial_fit_batch`
   now finds each `Vt` row's max-abs column (numpy `argmax`, first-on-ties via strict `>`)
   and negates the whole row if that entry is negative, pinning each `components_` row's
   max-abs entry POSITIVE (`_incremental_pca.py:357`, `extmath.py:897-905`).
2. **multi-batch `mean_correction` + batch-mean centring (was #1586, FIXED).** For
   `n_samples_seen > 0` `fn partial_fit_batch` now centres the batch by the BATCH mean and
   stacks the 3-block `M = [singular_values · components_ ; X_batch_centred ;
   mean_correction]` with `mean_correction = sqrt((n_seen/n_total) · n_batch) · (mean_ −
   col_batch_mean)` (`_incremental_pca.py:342-354`).
3. **`explained_variance_ratio_` denominator + running `var_` (was #1587, FIXED).** A
   faithful port of `_incremental_mean_and_var` (Youngs-Cramer/Chan, `extmath.py:1057-1180`)
   now tracks a per-feature running population variance (ddof=0) in the new `var_` field;
   `explained_variance_ratio_ = S² / Σ(col_var · n_total)` is the fraction of TOTAL feature
   variance (`_incremental_pca.py:359`). `explained_variance_ = S²/(n_total − 1)` is
   unchanged (`:358`).

The exposed surface is the unfitted `IncrementalPCA<F> { n_components, batch_size }`
(`pub struct IncrementalPCA in incremental_pca.rs`, `new`/`with_batch_size`/`n_components`/
`batch_size` only — NO `whiten`/`copy`) and the fitted `FittedIncrementalPCA<F> {
components_, explained_variance_, explained_variance_ratio_, mean_, var_, n_samples_seen_,
singular_values_ }` (`pub struct FittedIncrementalPCA in incremental_pca.rs`, accessors
`components`/`explained_variance`/`explained_variance_ratio`/`mean`/`var`/`n_samples_seen`/
`singular_values`, `inverse_transform`, and the streaming `partial_fit`/`partial_fit_batch`),
re-exported at the crate root (`pub use incremental_pca::{FittedIncrementalPCA,
IncrementalPCA}`, `lib.rs:88`) and bound to CPython as `_RsIncrementalPCA`
(`ferrolearn-python/src/extras.rs:1094-1099` via the `py_transformer!` macro `extras.rs:107`,
registered `ferrolearn-python/src/lib.rs:72`).

`IncrementalPCA` / `FittedIncrementalPCA` are existing pub APIs whose non-test consumers are
the crate re-export (`lib.rs:88`, boundary public API, grandfathered S5/R-DEFER-1) and the
`_RsIncrementalPCA` PyO3 binding (`extras.rs:1094`, registered `lib.rs:72`). There is NO
`PipelineTransformer`/`FittedPipelineTransformer` impl (unlike `pca.rs` — REQ-11 gap). The
value parity is pinned by `tests/divergence_incremental_pca.rs` (3 ex-divergence tests now
live/green + 11 structural green-guards).

## Probes (live sklearn oracle, 1.5.2, run from /tmp)

```bash
# PROBE 1 (REQ-1/2/3/5/6/7/15 — MULTI-batch fitted attrs, the headline parity case).
# IncrementalPCA(n_components=2, batch_size=3) on a fixed 9x4 X => 3 batches. VALUES generated
# by sklearn (R-CHAR-3), never copied from ferrolearn. DEMONSTRATES: each component-row max-abs
# entry POSITIVE (svd_flip), ratio sums to < 1 (fraction of total feature variance), running
# var_ (population, ddof=0).
python3 -c "
import numpy as np
from sklearn.decomposition import IncrementalPCA
X=np.array([[2.5,2.4,1.0,0.3],[0.5,0.7,3.0,1.1],[2.2,2.9,1.5,0.8],[1.9,2.2,2.0,1.4],
            [3.1,3.0,0.5,0.2],[2.3,2.7,1.2,0.9],[1.2,1.8,2.4,1.7],[2.8,2.6,0.9,0.4],
            [0.9,1.1,2.7,1.5]])
m=IncrementalPCA(n_components=2, batch_size=3).fit(X)
for i,row in enumerate(m.components_):
    k=int(np.argmax(np.abs(row))); print(f'  comp[{i}]=', np.round(row,8).tolist(), f'argmax-abs idx={k} val={row[k]:.8f} (positive=>svd_flip)')
print('  singular_values_:', np.round(m.singular_values_,8).tolist())
print('  explained_variance_:', np.round(m.explained_variance_,8).tolist())
print('  explained_variance_ratio_:', np.round(m.explained_variance_ratio_,8).tolist())
print('  ratio sum (FRACTION of total, < 1):', round(float(m.explained_variance_ratio_.sum()),8))
print('  mean_:', np.round(m.mean_,8).tolist(), ' var_:', np.round(m.var_,8).tolist())
print('  n_samples_seen_:', int(m.n_samples_seen_), ' noise_variance_:', round(float(m.noise_variance_),8))"
# ->   comp[0]= [0.57981816, 0.50018606, -0.57113503, -0.29568495] argmax-abs idx=0 val=0.57981816 (positive=>svd_flip)
# ->   comp[1]= [0.01541405, 0.62305771, 0.16562627, 0.76428361] argmax-abs idx=3 val=0.76428361 (positive=>svd_flip)
# ->   singular_values_: [4.30815437, 1.10648545]
# ->   explained_variance_: [2.32002426, 0.15303876]
# ->   explained_variance_ratio_: [0.92770047, 0.0611951]
# ->   ratio sum (FRACTION of total, < 1): 0.98889557
# ->   mean_: [1.93333333, 2.15555556, 1.68888889, 0.92222222]  var_: [0.7, 0.57580247, 0.68098765, 0.26617284]
# ->   n_samples_seen_: 9  noise_variance_: 0.00469617
#   => ferrolearn IncrementalPCA::<f64>::new(2).with_batch_size(3).fit(X) matches ALL of the above:
#      components (incl. sign) to 6.3e-15, explained_variance_ 2.8e-14, singular_values_ 2.0e-14,
#      ratio 2.4e-15, mean_/var_ exact. noise_variance_ (REQ-16) is NOT exposed by ferrolearn.

# PROBE 2 (REQ-1/3/5/6/7/15 — SINGLE batch; full attr parity).
# IncrementalPCA(n_components=2, batch_size=7) on a fixed 7x4 X (one batch).
python3 -c "
import numpy as np
from sklearn.decomposition import IncrementalPCA
X=np.array([[2.5,2.4,1.0,0.3],[0.5,0.7,3.0,1.1],[2.2,2.9,1.5,0.8],[1.9,2.2,2.0,1.4],
            [3.1,3.0,0.5,0.2],[2.3,2.7,1.2,0.9],[1.2,1.8,2.4,1.7]])
m=IncrementalPCA(n_components=2, batch_size=7).fit(X)
print('  components_:', np.round(m.components_,8).tolist())
print('  explained_variance_:', np.round(m.explained_variance_,8).tolist())
print('  explained_variance_ratio_:', np.round(m.explained_variance_ratio_,8).tolist())
print('  singular_values_:', np.round(m.singular_values_,8).tolist())
print('  mean_:', np.round(m.mean_,8).tolist(), ' var_:', np.round(m.var_,8).tolist())"
# ->   components_: [[0.5786644, 0.50446422, -0.5768673, -0.27908327], [0.03931978, 0.58642035, 0.16953871, 0.79108897]]
# ->   explained_variance_: [2.20603452, 0.19163081]
# ->   explained_variance_ratio_: [0.90908016, 0.07896874]
# ->   singular_values_: [3.63815985, 1.0722802]
# ->   mean_: [1.95714286, 2.24285714, 1.65714286, 0.91428571]  var_: [0.63959184, 0.54530612, 0.63959184, 0.2555102]
#   => ferrolearn matches ALL of the above: components (incl. sign) to 2.2e-16, explained_variance_
#      1.8e-15, ratio 2.2e-16, singular_values_ 1.3e-15, mean_/var_ exact.

# PROBE 3 (REQ-13/15/17 — ctor defaults + batch_size_ auto = 5*n_features).
python3 -c "
import numpy as np
from sklearn.decomposition import IncrementalPCA
d=IncrementalPCA()
for p in ['n_components','whiten','copy','batch_size']: print(f'  {p} =', getattr(d,p))
X=np.zeros((20,3)); X[:,0]=np.arange(20)
dd=IncrementalPCA().fit(X); print('  batch_size_ (None default, n_features=3) =', dd.batch_size_, '(= 5*n_features = 15)')"
# ->   n_components = None  whiten = False  copy = True  batch_size = None
# ->   batch_size_ (None default, n_features=3) = 15 (= 5*n_features = 15)
#   => ferrolearn requires n_components as an explicit usize (no None, REQ-14), has NO whiten/copy
#      (REQ-17), and defaults batch_size to the FULL dataset (n_samples), NOT 5*n_features (REQ-13).

# PROBE 4 (REQ-9 — param contract: sklearn allows n_components <= min(n_samples, n_features),
# so n_components == n_features is VALID; ferrolearn rejects n_components >= n_features).
python3 -c "
import numpy as np
from sklearn.decomposition import IncrementalPCA
X=np.array([[2.5,2.4,1.0],[0.5,0.7,3.0],[2.2,2.9,1.5],[1.9,2.2,2.0],[3.1,3.0,0.5],[2.3,2.7,1.2]])
m=IncrementalPCA(n_components=3).fit(X); print('  n_components=3 (== n_features) OK, n_components_ =', m.n_components_)"
# ->   n_components=3 (== n_features) OK, n_components_ = 3
#   => sklearn requires n_components <= min(n_samples, n_features) (_incremental_pca.py:297-308);
#      ferrolearn rejects n_components >= n_features (fn fit / fn partial_fit) — candidate DIV (REQ-9, #1590).
```

## Requirements

- REQ-1: **`components_` / `transform` VALUE parity via `svd_flip(u_based_decision=False)`
  (SHIPPED; was #1585, FIXED).** sklearn pins each `components_` row's sign deterministically:
  `U, Vt = svd_flip(U, Vt, u_based_decision=False)` (`_incremental_pca.py:357`), where
  `svd_flip`'s `u_based_decision=False` branch (`extmath.py:897-905`) takes `argmax(abs(v),
  axis=1)` per Vt ROW (`:899`, numpy first-max on ties), `signs = sign(v[row, max_abs])`
  (`:902`), `v *= signs[:, newaxis]` (`:905`) so each component row's max-abs entry is POSITIVE.
  ferrolearn's `fn partial_fit_batch` (`incremental_pca.rs`) now applies the same convention
  after `fn thin_svd`: for each Vt row it finds `j_max` (the index of the max abs value, strict
  `>` so ties take the FIRST index, matching numpy `argmax`) and negates the whole row if
  `components_[[k, j_max]] < 0`. `components_` (and the sign-dependent `transform`) match sklearn
  ROW-FOR-ROW including sign (single-batch 2.2e-16, multi-batch 6.3e-15). Non-test consumers:
  re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`.

- REQ-2: **multi-batch `mean_correction` row + batch-mean centring (SHIPPED; was #1586, FIXED).**
  For `n_samples_seen > 0` sklearn centres the batch by the BATCH mean `col_batch_mean =
  mean(X, axis=0)` (`_incremental_pca.py:342-343`) and stacks THREE blocks: the σ-weighted
  prior components, the batch-centred X, and a `mean_correction` row `sqrt((n_samples_seen /
  n_total) * n_samples) * (mean_ − col_batch_mean)` (`:345-354`). ferrolearn's `fn
  partial_fit_batch` (`incremental_pca.rs`) now, when `last_count > 0`, centres `x_batch` by
  `col_batch_mean`, builds `weighted = singular_values_[:,None] * components_`, builds the
  length-`n_features` `mean_correction` row with `scale = sqrt(last_count/new_n * batch_n)`,
  and assembles `M` via two `stack_vertical` calls (`[weighted ; x_centred ; mean_correction]`).
  The multi-batch `explained_variance_`/`singular_values_`/`components_` now match sklearn
  (Probe 1; explained_variance_ 2.8e-14, singular_values_ 2.0e-14). Non-test consumers:
  re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`.

- REQ-3: **`explained_variance_ratio_` total-variance denominator + running `var_` parity
  (SHIPPED; was #1587, FIXED).** sklearn computes `explained_variance_ratio = S² / sum(col_var
  * n_total_samples)` (`_incremental_pca.py:359`) — the denominator is the SUM of per-feature
  running variances times `n_total`, the TOTAL feature variance (Probe 1: ratio sums to
  `0.98889557 < 1`). This requires the running per-feature population variance (ddof=0), which
  ferrolearn now tracks in the `var_` field via a faithful port of `_incremental_mean_and_var`
  (Youngs-Cramer/Chan, `extmath.py:1057-1180`): `fn partial_fit_batch` computes the batch's
  corrected 2-pass `new_unnorm_var` and combines it with the prior `var_ * last_count` via the
  Chan update (`:1142-1178`), then `col_var = updated_unnorm_var / new_n`. The ratio loop then
  divides each `σ²` by `total_feature_var = Σ(col_var · new_n)`. `explained_variance_ratio_`
  matches sklearn (single-batch 2.2e-16, multi-batch 2.4e-15) and `var_` matches sklearn
  EXACTLY (Probe 1/2). Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA`
  `extras.rs:1094`. (`explained_variance_ = S²/(n_total − 1)` is REQ-6, unchanged.)

- REQ-4: **Incremental running MEAN (SHIPPED).** sklearn updates the per-feature mean via
  `_incremental_mean_and_var` (`_incremental_pca.py:329-334`, `extmath.py:1137` `updated_mean
  = (last_sum + new_sum) / updated_sample_count`) and sets `mean_ = col_mean` (`:364`).
  ferrolearn's `fn partial_fit_batch` (`incremental_pca.rs`) computes the same update
  `col_mean[j] = (mean_[j]·last_count + new_sum[j]) / new_n` (the `col_mean` loop) and stores
  it as `mean_`. The running mean matches sklearn's `col_mean` exactly regardless of batching
  (Probe 1/2: `mean_` matches to 0). Non-test consumers: re-export `lib.rs:88`,
  `_RsIncrementalPCA` `extras.rs:1094`. Pinned by `test_mean_is_correct` (col means `[1, 2]`
  to 1e-10), `test_batch_size_single_batch`, and the divergence-suite green-guards
  `green_multibatch_mean_matches_sklearn` / `green_fixture_a_mean_matches_sklearn`.

- REQ-5: **`components_` shape `(n_components, n_features)` + element-wise parity (SHIPPED).**
  sklearn `components_` shape `(n_components, n_features)`, `= Vt[:n_components_]`
  (`_incremental_pca.py:77-81,362`). ferrolearn's `FittedIncrementalPCA<F>.components_` is
  `Array2<F>` allocated `(n_components, n_features)` (`fn partial_fit` state init) and filled
  row-by-row in `fn partial_fit_batch` (the component-copy loop, with `svd_flip` REQ-1), accessor
  `fn components`. Shape and (post-fix) element-wise VALUES both match sklearn (Probe 1/2; the
  VALUE parity itself is REQ-1's claim). Non-test consumers: re-export `lib.rs:88`,
  `_RsIncrementalPCA` `extras.rs:1094`. Pinned by `test_fit_output_shape` (`(1,2)`),
  `test_fit_two_components` (`(2,3)`), `green_shapes_and_unit_norm`.

- REQ-6: **`explained_variance_ = S² / (n_total − 1)` (SHIPPED).** sklearn `explained_variance
  = S**2 / (n_total_samples − 1)` (`_incremental_pca.py:358`). ferrolearn's `fn
  partial_fit_batch` (`incremental_pca.rs`) computes `explained_variance_[k] = σ[k]² /
  (new_n − 1)` (the explained-variance loop, `denom = new_n.saturating_sub(1).max(1)`). The
  denominator MATCHES sklearn and, with REQ-1/2 fixed, the VALUE now matches element-wise
  (single-batch 1.8e-15, multi-batch 2.8e-14). Non-test consumers: re-export `lib.rs:88`,
  `_RsIncrementalPCA` `extras.rs:1094`. Pinned by `test_explained_variance_positive`,
  `green_single_batch_explained_variance_matches`,
  `green_fixture_a_explained_variance_and_singular_values_match`.

- REQ-7: **`singular_values_` length `n_components`, non-negative + element-wise parity
  (SHIPPED).** sklearn `singular_values_ = S[:n_components_]` (`_incremental_pca.py:363`).
  ferrolearn's `fn thin_svd` (`incremental_pca.rs`) clamps each `sv = sqrt(max(eigval, 0))` and
  sorts the eigenvalue indices DESCENDING; `fn partial_fit_batch` copies the top `n_components`
  into `singular_values_` (the singular-value copy loop) and zero-fills any beyond `max_rank`.
  Length, non-negativity, and (post-fix) element-wise VALUES all match sklearn (single-batch
  1.3e-15, multi-batch 2.0e-14). Non-test consumer: re-export `lib.rs:88`. Pinned by
  `green_single_batch_singular_values_matches`,
  `green_fixture_a_explained_variance_and_singular_values_match`.

- REQ-8: **`components_` rows approximately UNIT-NORM (SHIPPED).** sklearn's `components_` are
  right singular vectors `Vt` (`_incremental_pca.py:79-80,362`), orthonormal rows. ferrolearn's
  `fn thin_svd` recovers `V = MᵀU/σ` (or stores eigenvectors directly), unit-norm; `fn
  partial_fit_batch` copies these rows into `components_`. Non-test consumer: re-export
  `lib.rs:88`. Pinned by `test_components_approx_unit_length` (each row 2-norm == 1 to 1e-6)
  and `green_shapes_and_unit_norm`.

- REQ-9: **Error / parameter contracts (SHIPPED, scoped).** `fn fit` (`incremental_pca.rs`)
  returns `InvalidParameter { name: "n_components" }` for `n_components == 0`, `InvalidParameter
  { name: "n_features" }` for zero features, `InvalidParameter { name: "n_components" }` for
  `n_components >= n_features`, `InsufficientSamples { required: 2 }` for `< 2` samples, and
  `InvalidParameter { name: "batch_size" }` for `batch_size == 0`; `fn partial_fit` enforces the
  same `n_components`/`n_features` checks; `fn partial_fit_batch` returns `InsufficientSamples`
  on an empty batch and `ShapeMismatch` on a feature-count change after the first batch; `fn
  transform` and `fn inverse_transform` return `ShapeMismatch` on a column-count mismatch.
  Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Pinned by
  `test_invalid_n_components_zero_error`, `test_invalid_n_components_ge_n_features_error`,
  `test_insufficient_samples_error`, `test_zero_batch_size_error`,
  `test_transform_shape_mismatch_error`, and `green_error_contracts`. **FLAG (candidate DIVs):**
  sklearn validates via `_parameter_constraints` (`_incremental_pca.py:187-192`) raising
  `InvalidParameterError`/`ValueError` (not `FerroError`); sklearn requires `n_components <=
  min(n_samples, n_features)` (`:297-308`) so `n_components == n_features` is VALID (Probe 4),
  whereas ferrolearn rejects `n_components >= n_features` (pinned as current behavior by
  `green_n_components_eq_n_features_currently_rejected`, candidate DIV #1590); sklearn accepts
  `n_components=None` (REQ-14) and does not pre-reject `n_samples < 2`.

- REQ-10: **`n_samples_seen_` accumulation + `partial_fit` chaining + `batch_size` chunking
  (SHIPPED, scoped).** sklearn increments `n_samples_seen_ = n_total_samples` across
  `partial_fit` calls (`_incremental_pca.py:361`) and `fit` splits via `gen_batches(n_samples,
  batch_size_, min_batch_size=n_components or 0)` (`:241-243`). ferrolearn's `fn
  partial_fit_batch` sets `n_samples_seen_ = new_n` (running total); `fn partial_fit` threads
  an `Option<FittedIncrementalPCA>` state through successive batches; `fn fit` slices `x` into
  `batch_size`-row chunks and calls `partial_fit` per chunk. Non-test consumers: re-export
  `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094` (`fit`). Pinned by `test_n_samples_seen`
  (5), `test_partial_fit_chaining` (2 then 4), `test_batch_size_two_batches` (4),
  `test_getters`, and `green_n_samples_seen_accumulation` (chains to 6). **FLAG (candidate DIV):**
  sklearn's `min_batch_size = n_components or 0` (`:242`) drops a trailing batch smaller than
  `n_components`; ferrolearn processes every trailing chunk with `nrows > 0` — see also REQ-13.

- REQ-11: **`PipelineTransformer` integration (NOT-STARTED; #1588).** `pca.rs` (`pca.rs:565`)
  and `truncated_svd.rs` implement `PipelineTransformer<F>` + `FittedPipelineTransformer<F>` so a
  PCA/SVD step can sit in a `Pipeline` (the ferrolearn analogue of sklearn's `TransformerMixin`;
  `IncrementalPCA` is `_BasePCA` → `TransformerMixin` `_incremental_pca.py:19`). `incremental_pca.rs`
  has NO `impl PipelineTransformer for IncrementalPCA` / `impl FittedPipelineTransformer for
  FittedIncrementalPCA` (grep finds none), so it cannot be composed into a `Pipeline`.

- REQ-12: **f32 / f64 generic support (SHIPPED).** `IncrementalPCA<F>` / `FittedIncrementalPCA<F>`
  are generic over `F: Float + Send + Sync + 'static` (`pub struct IncrementalPCA in
  incremental_pca.rs` / `pub struct FittedIncrementalPCA in incremental_pca.rs`); `fn thin_svd`,
  `fn jacobi_eigen_symmetric`, the Youngs-Cramer variance update, the 3-block stack, and `svd_flip`
  are all generic. sklearn validates `dtype=[float64, float32]` (`_incremental_pca.py:231`).
  Non-test consumer: re-export `lib.rs:88`. Pinned by `test_f32_support` and `green_f32_path`.

- REQ-13: **`batch_size` auto-default = `5 * n_features` (NOT-STARTED; #1589).** sklearn sets
  `batch_size_ = 5 * n_features` when `batch_size is None` (`_incremental_pca.py:236-239`;
  Probe 3: `batch_size_ = 15` for `n_features = 3`) to balance accuracy and memory. ferrolearn's
  `fn fit` (`incremental_pca.rs`) defaults `batch_size` to `n_samples` (the FULL dataset, the
  `None => n_samples` branch), so the default fit is plain single-batch PCA, not a true
  mini-batch IncrementalPCA. (Both paths are now value-parity-correct for whatever batching is
  used — this is a default-value DIV, not a correctness one.) FIXABLE.

- REQ-14: **`n_components=None` default + `n_components == n_features` acceptance (NOT-STARTED;
  #1590).** sklearn's ctor default is `n_components=None` (`_incremental_pca.py:194`), resolved
  to `min(n_samples, n_features)` on the first `partial_fit` (`:294`; Probe 3), and sklearn
  accepts `n_components <= min(n_samples, n_features)` so `n_components == n_features` is VALID
  (Probe 4, `:297-308`). ferrolearn's `IncrementalPCA::new(n_components: usize)` (`fn new`
  `incremental_pca.rs`) requires an explicit integer (no `None` / auto-resolution) and `fn fit` /
  `fn partial_fit` reject `n_components >= n_features` (the current reject pinned by
  `green_n_components_eq_n_features_currently_rejected`).

- REQ-15: **`var_` running-variance fitted attr (SHIPPED).** sklearn tracks a per-feature running
  population variance (ddof=0) via `_incremental_mean_and_var` (`_incremental_pca.py:329-334`,
  Youngs-and-Cramer update `extmath.py:1057-1180`) and exposes `var_` (`:99-101,365`; Probe 1:
  `var_ = [0.7, 0.57580247, 0.68098765, 0.26617284]`). ferrolearn's `FittedIncrementalPCA<F>` now
  has a `var_` field (`pub struct FittedIncrementalPCA in incremental_pca.rs`), updated each call
  in `fn partial_fit_batch` via the same Chan combination (the `col_var` loop) and exposed by the
  `fn var` accessor (`pub fn var`). `var_` matches sklearn EXACTLY on both single- and multi-batch
  fixtures (Probe 1/2). Non-test consumer: re-export `lib.rs:88`. (The accessor is not yet surfaced
  through the PyO3 binding — that binding-surface gap is REQ-11/REQ-17 territory, not a computation
  gap. The computation underpins REQ-3's ratio denominator.)

- REQ-16: **`noise_variance_` fitted attr (NOT-STARTED; #1592).** sklearn exposes
  `noise_variance_ = explained_variance[n_components_:].mean()` (the mean of the discarded
  explained variances; `_incremental_pca.py:369-372`; Probe 1: `0.00469617`), and `0.0` when
  `n_components_ in (n_samples, n_features)`. ferrolearn's `FittedIncrementalPCA<F>` has no
  `noise_variance_` field — `fn partial_fit_batch` keeps only the top `n_components` rows of the
  thin SVD (zero-filling any beyond `max_rank`) and never retains the discarded-eigenvalue tail.

- REQ-17: **`whiten` + `copy` ctor params (NOT-STARTED; #1593).** sklearn's ctor takes
  `whiten=False` (`_incremental_pca.py:196`, scales `transform` by `1/sqrt(explained_variance_)`
  and un-scales `inverse_transform` via `_BasePCA`) and `copy=True` (`:197`, in-place vs copy
  validation; Probe 3). ferrolearn's `IncrementalPCA<F>` (`pub struct IncrementalPCA in
  incremental_pca.rs`) has `n_components` + `batch_size` ONLY — no `whiten` (so `fn transform`
  is the plain projection `(X − mean) · componentsᵀ` and `fn inverse_transform` is `X_reduced ·
  components + mean`, no whiten scaling) and no `copy`.

- REQ-18: **`n_features_in_` fitted attr (NOT-STARTED; #1594).** sklearn exposes `n_features_in_`
  (`_incremental_pca.py:120-123`). ferrolearn's `FittedIncrementalPCA<F>` exposes `components`/
  `explained_variance`/`explained_variance_ratio`/`mean`/`var`/`n_samples_seen`/`singular_values` —
  no `n_features_in_` accessor (derivable from `mean_.len()` but not exposed) and no `n_components_`.

- REQ-19: **ferray substrate (NOT-STARTED; #1595).** `incremental_pca.rs` computes on
  `ndarray::{Array1, Array2}` and uses a hand-rolled `fn thin_svd` + `fn jacobi_eigen_symmetric`,
  not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3).

- AC-1 (REQ-1, SHIPPED): each row of `IncrementalPCA::new(2).with_batch_size(b).fit(&X)
  .components()` has its max-abs entry POSITIVE and matches the live `IncrementalPCA(n_components=2,
  batch_size=b).components_` ROW-FOR-ROW INCLUDING SIGN (Probe 1/2). Confirmed by the cross-check
  (single-batch 2.2e-16, multi-batch 6.3e-15) and pinned by `divergence_svd_flip_sign`.

- AC-2 (REQ-2, SHIPPED): a MULTI-batch fit (`batch_size < n_samples`) matches the live
  multi-batch `IncrementalPCA` `explained_variance_`/`singular_values_`/`components_` (Probe 1:
  `explained_variance_ = [2.32002426, 0.15303876]`, `singular_values_ = [4.30815437, 1.10648545]`).
  Confirmed (explained_variance_ 2.8e-14, singular_values_ 2.0e-14) and pinned by
  `divergence_multibatch_mean_correction`.

- AC-3 (REQ-3/15, SHIPPED): `fitted.explained_variance_ratio()` matches `S² / sum(var_ ·
  n_samples_seen_)` (Probe 1: ratio `[0.92770047, 0.0611951]`, sums to `0.98889557 < 1`), and
  `fitted.var()` matches `IncrementalPCA(...).var_` exactly (Probe 1/2). Pinned by
  `divergence_explained_variance_ratio_denominator`.

- AC-4 (REQ-5/6/7/8, SHIPPED): `components()` is `(k, n_features)` and matches sklearn
  element-wise; `explained_variance()` is `S²/(n−1)` and matches; `singular_values()` is length
  `k` ≥ 0 and matches; each `components()` row is unit-norm to 1e-6 (Probe 1/2). Pinned by
  `green_shapes_and_unit_norm`, `green_single_batch_explained_variance_matches`,
  `green_single_batch_singular_values_matches`,
  `green_fixture_a_explained_variance_and_singular_values_match`,
  `test_components_approx_unit_length`.

- AC-5 (REQ-4/9/10/12, SHIPPED scoped): the running `mean()` equals the full-data column mean
  (Probe 1: `[1.93333333, 2.15555556, 1.68888889, 0.92222222]`); `fit` `Err`s for `n_components=0`,
  `n_components >= n_features`, `n_samples < 2`, `batch_size = 0`; `n_samples_seen()` accumulates;
  `partial_fit` chains; the f32 path fits/transforms. Pinned by `green_multibatch_mean_matches_sklearn`,
  `green_fixture_a_mean_matches_sklearn`, `green_error_contracts`, `green_n_samples_seen_accumulation`,
  `green_f32_path`, and the in-module `test_*` suite. FLAG: sklearn allows `n_components == n_features`
  (Probe 4, REQ-14) and `n_components=None`.

- AC-6 (REQ-11/13/14/16/17/18/19, NOT-STARTED): `IncrementalPCA()` defaults `n_components=None,
  whiten=False, copy=True, batch_size=None` with `batch_size_ = 5*n_features` (Probe 3,
  `_incremental_pca.py:194-198,236-239`); sklearn exposes `noise_variance_` (Probe 1: `0.00469617`),
  `n_features_in_`, `whiten`/`copy`, and a `Pipeline`-composable `TransformerMixin` surface.
  ferrolearn has none of these, defaults `batch_size` to the full dataset, rejects `n_components ==
  n_features`, has no `PipelineTransformer` impl, and computes on `ndarray` + hand-rolled Jacobi SVD,
  not ferray.

## REQ status

Binary (R-DEFER-2). `IncrementalPCA` / `FittedIncrementalPCA` are existing pub APIs; the non-test
consumers are the crate re-export (`lib.rs:88`, boundary public API, grandfathered S5/R-DEFER-1)
and the `_RsIncrementalPCA` PyO3 binding (`extras.rs:1094`, registered `lib.rs:72`). Cites use
symbol anchors (ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2,
run from `/tmp`.
**A critic→fixer→re-audit cycle (closing #1585/#1586/#1587) reimplemented `fn partial_fit_batch`
as a faithful port of sklearn's `partial_fit`, and ferrolearn now has FULL VALUE PARITY with the
deterministic live `IncrementalPCA` oracle on BOTH single-batch (7×4, components incl. sign 2.2e-16,
explained_variance_ 1.8e-15, ratio 2.2e-16, singular_values_ 1.3e-15, mean_/var_ 0) and multi-batch
(9×4, batch_size=3, components incl. sign 6.3e-15, explained_variance_ 2.8e-14, singular_values_
2.0e-14, ratio 2.4e-15, mean_/var_ 0) fits.** The three FIXED points are: (1) `svd_flip
(u_based_decision=False)` (REQ-1, was #1585); (2) the multi-batch 3-block `mean_correction` stack +
batch-mean centring (REQ-2, was #1586); (3) the `explained_variance_ratio_` total-variance denominator
backed by the running `var_` Youngs-Cramer update (REQ-3/15, was #1587). The 3 ex-divergence tests are
now live/green in `tests/divergence_incremental_pca.rs` alongside 11 structural green-guards (14 tests,
all pass). #1584 is this doc's crosslink tracking issue. Count: **12 SHIPPED (REQ-1,2,3,4,5,6,7,8,9,
10,12,15) / 7 NOT-STARTED (REQ-11,13,14,16,17,18,19)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (`components_`/`transform` parity via `svd_flip(u_based_decision=False)`) | SHIPPED | was #1585, FIXED. sklearn `U, Vt = svd_flip(U, Vt, u_based_decision=False)` (`_incremental_pca.py:357`); `u_based_decision=False` branch (`extmath.py:897-905`) `argmax(abs(v),axis=1)` per Vt row (`:899`, numpy first-max), `v *= sign(v[row,max_abs])` (`:905`) → max-abs entry POSITIVE. ferrolearn `fn partial_fit_batch` (`incremental_pca.rs`) finds `j_max` per row (strict `>` → first-on-ties) and negates the whole row if `components_[[k,j_max]] < 0`, after `fn thin_svd`. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Cross-check: components incl. sign match live `IncrementalPCA(...).components_` single-batch 2.2e-16, multi-batch 6.3e-15 (Probe 1/2). Verification: `cargo test -p ferrolearn-decomp --test divergence_incremental_pca` → `divergence_svd_flip_sign` PASS. |
| REQ-2 (multi-batch `mean_correction` row + batch-mean centring) | SHIPPED | was #1586, FIXED. For `n_samples_seen>0` sklearn centres by the BATCH mean (`_incremental_pca.py:342-343`) and stacks `[singular_values·components ; X_batch_centred ; mean_correction]`, `mean_correction = sqrt((n_seen/n_total)·n_batch)·(mean_ − col_batch_mean)` (`:345-354`). ferrolearn `fn partial_fit_batch` (`incremental_pca.rs`) builds `weighted = singular_values_[:,None]·components_`, the `mean_correction` row with `scale = sqrt(last_count/new_n · batch_n)`, and assembles `M` via two `fn stack_vertical` calls. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Cross-check (Probe 1): multi-batch `explained_variance_` 2.8e-14, `singular_values_` 2.0e-14. Verification: `divergence_multibatch_mean_correction` PASS. |
| REQ-3 (`explained_variance_ratio_` total-variance denominator + running `var_`) | SHIPPED | was #1587, FIXED. sklearn `explained_variance_ratio = S² / sum(col_var · n_total_samples)` (`_incremental_pca.py:359`) — fraction of total feature variance (Probe 1: ratio sums to `0.98889557 < 1`). ferrolearn `fn partial_fit_batch` (`incremental_pca.rs`) ports `_incremental_mean_and_var` (Youngs-Cramer/Chan, `extmath.py:1057-1180`) for the running `var_`, then divides each `σ²` by `total_feature_var = Σ(col_var · new_n)`. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Cross-check (Probe 1/2): ratio single-batch 2.2e-16, multi-batch 2.4e-15; `var_` matches sklearn EXACTLY. Verification: `divergence_explained_variance_ratio_denominator` PASS. |
| REQ-4 (incremental running MEAN) | SHIPPED | sklearn `mean_ = col_mean` via `_incremental_mean_and_var` (`_incremental_pca.py:329-334,364`; `extmath.py:1137`). ferrolearn `fn partial_fit_batch` (`incremental_pca.rs`) `col_mean[j] = (mean_[j]·last_count + new_sum[j])/new_n` (the `col_mean` loop), stored as `mean_`; matches sklearn exactly regardless of batching. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Verification: `test_mean_is_correct` (`[1,2]` to 1e-10), `test_batch_size_single_batch`, `green_multibatch_mean_matches_sklearn`, `green_fixture_a_mean_matches_sklearn` PASS. |
| REQ-5 (`components_` shape `(n_components, n_features)` + element-wise parity) | SHIPPED | sklearn `components_` shape `(n_components, n_features)` (`_incremental_pca.py:77-81`), `= Vt[:n_components_]` (`:362`). ferrolearn `FittedIncrementalPCA.components_` is `Array2<F>` allocated `(n_components, n_features)` (`fn partial_fit` state init), filled by `fn partial_fit_batch` (with `svd_flip` REQ-1), accessor `fn components`. Shape + (post-fix) element-wise VALUES match sklearn (Probe 1/2; VALUE parity is REQ-1's claim). Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Verification: `test_fit_output_shape`, `test_fit_two_components`, `green_shapes_and_unit_norm` PASS. |
| REQ-6 (`explained_variance_ = S²/(n−1)`) | SHIPPED | sklearn `explained_variance = S**2 / (n_total_samples − 1)` (`_incremental_pca.py:358`). ferrolearn `fn partial_fit_batch` (`incremental_pca.rs`) `explained_variance_[k] = σ[k]² / (new_n − 1)` (`denom = new_n.saturating_sub(1).max(1)`). With REQ-1/2 fixed, the VALUE matches element-wise (single-batch 1.8e-15, multi-batch 2.8e-14). Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Verification: `test_explained_variance_positive`, `green_single_batch_explained_variance_matches`, `green_fixture_a_explained_variance_and_singular_values_match` PASS. |
| REQ-7 (`singular_values_` length `n_components` + ≥ 0 + element-wise parity) | SHIPPED | sklearn `singular_values_ = S[:n_components_]` (`_incremental_pca.py:363`). ferrolearn `fn thin_svd` (`incremental_pca.rs`) clamps `sv = sqrt(max(eigval,0))`, sorts indices DESCENDING; `fn partial_fit_batch` copies the top `n_components` into `singular_values_`, zero-fills beyond `max_rank`. Length + non-negativity + (post-fix) VALUE match sklearn (single-batch 1.3e-15, multi-batch 2.0e-14). Non-test consumer: re-export `lib.rs:88`. Verification: `green_single_batch_singular_values_matches`, `green_fixture_a_explained_variance_and_singular_values_match` PASS. |
| REQ-8 (`components_` rows ~UNIT-NORM) | SHIPPED | sklearn `components_` are right singular vectors `Vt` (`_incremental_pca.py:79-80,362`), orthonormal rows. ferrolearn `fn thin_svd` recovers `V = MᵀU/σ` / stores eigenvectors (unit-norm); `fn partial_fit_batch` copies the rows into `components_`. Non-test consumer: re-export `lib.rs:88`. Verification: `test_components_approx_unit_length` (each row 2-norm == 1 to 1e-6), `green_shapes_and_unit_norm` PASS. |
| REQ-9 (error / parameter contracts, scoped) | SHIPPED | `fn fit` (`incremental_pca.rs`) returns `Err(InvalidParameter{name:"n_components"})` for `==0` and `>= n_features`, `Err(InvalidParameter{name:"n_features"})` for zero features, `Err(InsufficientSamples{required:2})` for `<2` samples, `Err(InvalidParameter{name:"batch_size"})` for `==0`; `fn partial_fit` repeats the `n_components`/`n_features` checks; `fn partial_fit_batch` `Err(InsufficientSamples)` on empty batch / `Err(ShapeMismatch)` on feature change; `fn transform` + `fn inverse_transform` `Err(ShapeMismatch)` on column mismatch. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094`. Verification: `test_invalid_n_components_zero_error`, `test_invalid_n_components_ge_n_features_error`, `test_insufficient_samples_error`, `test_zero_batch_size_error`, `test_transform_shape_mismatch_error`, `green_error_contracts` PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` (`_incremental_pca.py:187-192`); allows `n_components == n_features` (Probe 4, `:297-308`, REQ-14 #1590) — ferrolearn rejects `>= n_features` (pinned current behavior `green_n_components_eq_n_features_currently_rejected`); accepts `n_components=None`; does NOT pre-reject `n_samples < 2`. |
| REQ-10 (`n_samples_seen_` accumulation + `partial_fit` chaining + `batch_size` chunking) | SHIPPED | sklearn `n_samples_seen_ = n_total_samples` across `partial_fit` (`_incremental_pca.py:361`); `fit` splits via `gen_batches(...)` (`:241-247`). ferrolearn `fn partial_fit_batch` sets `n_samples_seen_ = new_n`; `fn partial_fit` threads `Option<FittedIncrementalPCA>` state; `fn fit` slices `x` into `batch_size`-row chunks per `partial_fit`. Non-test consumers: re-export `lib.rs:88`, `_RsIncrementalPCA` `extras.rs:1094` (`fit`). Verification: `test_n_samples_seen` (5), `test_partial_fit_chaining` (2→4), `test_batch_size_two_batches` (4), `test_getters`, `green_n_samples_seen_accumulation` (→6) PASS. **FLAG (candidate DIV):** sklearn's `min_batch_size = n_components or 0` (`:242`) drops a small trailing batch; ferrolearn processes every `nrows>0` chunk (see REQ-13). |
| REQ-11 (`PipelineTransformer` integration) | NOT-STARTED | open prereq blocker #1588. `pca.rs` (`PipelineTransformer` impl `pca.rs:565`) and `truncated_svd.rs` implement `PipelineTransformer<F>` + `FittedPipelineTransformer<F>` (analogue of sklearn `TransformerMixin`; `IncrementalPCA` is `_BasePCA` → `TransformerMixin` `_incremental_pca.py:19`). `incremental_pca.rs` has NO such impl (grep finds none), so `IncrementalPCA` cannot be a `Pipeline` step. |
| REQ-12 (f32 / f64 generic support) | SHIPPED | `IncrementalPCA<F>`/`FittedIncrementalPCA<F>` generic over `F: Float + Send + Sync + 'static` (`pub struct IncrementalPCA in incremental_pca.rs` / `pub struct FittedIncrementalPCA in incremental_pca.rs`); `fn thin_svd`, `fn jacobi_eigen_symmetric`, the Youngs-Cramer update, the 3-block stack, and `svd_flip` are all generic. sklearn `dtype=[float64,float32]` (`_incremental_pca.py:231`). Non-test consumer: re-export `lib.rs:88`. Verification: `test_f32_support`, `green_f32_path` PASS. |
| REQ-13 (`batch_size` auto-default = `5*n_features`) | NOT-STARTED | open prereq blocker #1589. sklearn `batch_size_ = 5 * n_features` when `batch_size is None` (`_incremental_pca.py:236-239`; Probe 3: `15` for `n_features=3`). ferrolearn `fn fit` (`incremental_pca.rs`) defaults `batch_size` to `n_samples` (full dataset, `None => n_samples`), so the default fit is single-batch plain PCA, not a true mini-batch IncrementalPCA. (Both paths are now value-parity-correct for the batching they use; this is a default-value DIV.) |
| REQ-14 (`n_components=None` default + `n_components == n_features` acceptance) | NOT-STARTED | open prereq blocker #1590. sklearn ctor default `n_components=None` (`_incremental_pca.py:194`) → `min(n_samples,n_features)` on first `partial_fit` (`:294`); sklearn accepts `n_components <= min(n_samples,n_features)` so `n_components == n_features` is VALID (Probe 4, `:297-308`). ferrolearn `IncrementalPCA::new(n_components: usize)` (`fn new` `incremental_pca.rs`) requires an explicit integer (no auto-resolution) and rejects `n_components >= n_features` (current reject pinned by `green_n_components_eq_n_features_currently_rejected`). |
| REQ-15 (`var_` running-variance fitted attr) | SHIPPED | sklearn tracks per-feature running population variance (ddof=0) via `_incremental_mean_and_var` (`_incremental_pca.py:329-334`, Youngs-Cramer `extmath.py:1057-1180`) and exposes `var_` (`:99-101,365`; Probe 1: `[0.7, 0.57580247, 0.68098765, 0.26617284]`). ferrolearn `FittedIncrementalPCA<F>` now has a `var_` field (`pub struct FittedIncrementalPCA in incremental_pca.rs`), updated in `fn partial_fit_batch` (the `col_var` Chan loop) and exposed by `pub fn var`. Non-test consumer: re-export `lib.rs:88`. Cross-check (Probe 1/2): `var()` matches `IncrementalPCA(...).var_` EXACTLY. The computation underpins REQ-3's ratio denominator. (NOTE: the `var()` accessor is not yet surfaced through the PyO3 binding — a binding-surface gap folded into REQ-11/17 territory, NOT a computation gap.) |
| REQ-16 (`noise_variance_` fitted attr) | NOT-STARTED | open prereq blocker #1592. sklearn `noise_variance_ = explained_variance[n_components_:].mean()` (`_incremental_pca.py:369-372`; Probe 1: `0.00469617`), `0.0` when `n_components_ in (n_samples, n_features)`. ferrolearn `FittedIncrementalPCA<F>` has no `noise_variance_` field — `fn partial_fit_batch` keeps only the top `n_components` thin-SVD rows (zero-filling beyond `max_rank`), discarding the eigenvalue tail. |
| REQ-17 (`whiten` + `copy` ctor params) | NOT-STARTED | open prereq blocker #1593. sklearn ctor `whiten=False` (`_incremental_pca.py:196`, scales `transform` by `1/sqrt(explained_variance_)` + un-scales `inverse_transform` via `_BasePCA`) and `copy=True` (`:197`; Probe 3). ferrolearn `IncrementalPCA<F>` (`pub struct IncrementalPCA in incremental_pca.rs`) has `n_components` + `batch_size` ONLY — `fn transform` is plain `(X−mean)·componentsᵀ` and `fn inverse_transform` is `x_reduced·components + mean`, no whiten scaling; no `copy`. |
| REQ-18 (`n_features_in_` fitted attr) | NOT-STARTED | open prereq blocker #1594. sklearn exposes `n_features_in_` (`_incremental_pca.py:120-123`). ferrolearn `FittedIncrementalPCA<F>` exposes `components`/`explained_variance`/`explained_variance_ratio`/`mean`/`var`/`n_samples_seen`/`singular_values` — no `n_features_in_` (derivable from `mean_.len()`) and no `n_components_`. |
| REQ-19 (ferray substrate) | NOT-STARTED | open prereq blocker #1595. `incremental_pca.rs` computes on `ndarray::{Array1, Array2}` and uses a hand-rolled `fn thin_svd` + `fn jacobi_eigen_symmetric`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`incremental_pca.rs` follows the unfitted/fitted split (CLAUDE.md naming): `IncrementalPCA<F> {
n_components, batch_size: Option<usize> }` (`pub struct IncrementalPCA in incremental_pca.rs`;
`new(n_components)` `fn new`, `with_batch_size(bs)` `fn with_batch_size`, accessors `n_components`/
`batch_size`) → `Fit<Array2<F>, ()>` → `FittedIncrementalPCA<F> { components_, explained_variance_,
explained_variance_ratio_, mean_, var_, n_samples_seen_, singular_values_ }` (`pub struct
FittedIncrementalPCA in incremental_pca.rs`, accessors `components`/`explained_variance`/
`explained_variance_ratio`/`mean`/`var`/`n_samples_seen`/`singular_values`, `inverse_transform`, and
the streaming `partial_fit_batch`). The unfitted type also exposes `partial_fit(x_batch, state)`
(`fn partial_fit`) for streaming use. The path is generic over `F: Float + Send + Sync + 'static`
(both f32 and f64, `test_f32_support`); `fit`/`transform`/`inverse_transform`/`partial_fit` return
`Result<_, FerroError>` (R-CODE-2). There is NO `PipelineTransformer` impl (REQ-11, #1588).

**Fit path (`fn fit` `incremental_pca.rs`) — REQ-9/10/13/14.** Validates `n_components != 0`,
`n_features != 0`, `n_components < n_features` (the `>= n_features` reject is a candidate DIV vs
sklearn's `<= min(n_samples, n_features)`, REQ-14 / Probe 4), `n_samples >= 2`, and `batch_size != 0`
when set. `batch_size` defaults to `n_samples` (the FULL dataset — DIV vs sklearn `5 * n_features`,
REQ-13 #1589). The fit slices `x` into `batch_size`-row chunks (the chunking loop) and threads an
`Option<FittedIncrementalPCA>` state through `fn partial_fit` per chunk (mirroring sklearn's
`gen_batches` loop `_incremental_pca.py:241-247`, except ferrolearn does not apply
`min_batch_size = n_components` so it never drops a small trailing batch).

**Incremental update (`fn partial_fit_batch` `incremental_pca.rs`) — REQ-1/2/3/4/5/6/7/8/15.** This
is a faithful port of sklearn `partial_fit` (`_incremental_pca.py:329-372`):

1. **Running mean + variance (REQ-4/15/3).** Compute `new_sum = X.sum(axis=0)`, `col_batch_mean =
   new_sum / batch_n`, and `col_mean = (mean_·last_count + new_sum)/new_n` (the convex update,
   `extmath.py:1137`). For variance, compute the batch's corrected 2-pass unnormalised variance
   `new_unnorm_var = Σ(X − col_batch_mean)² − correction²/batch_n` (`extmath.py:1142-1162`), combine
   with the prior `var_·last_count` via the Chan term `+ (last/new) / new_n · (last_sum/(last/new) −
   new_sum)²` (`:1178`), and divide by `new_n` to get the population variance `col_var` (ddof=0). This
   is the faithful `_incremental_mean_and_var` port; `last_sample_count` is the scalar `n_samples_seen_`
   broadcast over all features (`_incremental_pca.py:329-334`), so per-feature counts are equal and
   scalar arithmetic is exact.

2. **Build the stacked matrix `M` (REQ-2).** If `last_count == 0` (or `n_components == 0`): `M =
   X − col_mean` (first-batch centring, `_incremental_pca.py:340`). Otherwise centre the batch by the
   BATCH mean (`X − col_batch_mean`, `:343`) and assemble the THREE blocks (`:348-354`): block 1
   `weighted = singular_values_[:,None] · components_`; block 2 the batch-centred `x_centred`; block 3
   a single `mean_correction` row `scale · (mean_ − col_batch_mean)` with `scale = sqrt(last_count/new_n
   · batch_n)` (`:345-347`). Stacking is two `fn stack_vertical` calls.

3. **Thin SVD (REQ-7).** `fn thin_svd` (Jacobi eigendecomposition `fn jacobi_eigen_symmetric` of the
   smaller of `MMᵀ`/`MᵀM`, eigenvalues sorted DESCENDING, `σ = sqrt(max(eigval, 0))`) replaces sklearn's
   `scipy.linalg.svd` (`_incremental_pca.py:356`) — a Rust-analog substitution (R-DEV-7) that produces
   value-identical results within tolerance.

4. **`svd_flip(u_based_decision=False)` (REQ-1).** For each Vt row find `j_max` (max abs column, strict
   `>` → numpy first-on-ties) and negate the whole row if `components_[[k, j_max]] < 0`, pinning each
   component row's max-abs entry positive (`_incremental_pca.py:357`, `extmath.py:897-905`).
   `singular_values_` is set from `σ`; rows beyond `max_rank` are zeroed.

5. **Explained variance / ratio (REQ-6/3).** `explained_variance_ = σ²/(new_n − 1)` (`:358`);
   `explained_variance_ratio_ = σ² / Σ(col_var · new_n)` = fraction of total feature variance (`:359`).
   `mean_`, `var_`, `n_samples_seen_` are then committed.

**Transform / inverse (`fn transform` / `fn inverse_transform` `incremental_pca.rs`) — REQ-9/17.**
`fn transform` projects `(X − mean_) · components_ᵀ` (the centring loop + `dot`), mirroring
`_BasePCA.transform` no-whiten branch; `fn inverse_transform` is `x_reduced · components_ + mean_`.
Neither applies the `whiten` scale/un-scale (REQ-17 #1593); both `Err(ShapeMismatch)` on a
column-count mismatch (REQ-9).

**Boundary consumer.** `_RsIncrementalPCA` (`ferrolearn-python/src/extras.rs:1094-1099`, via the
`py_transformer!` macro `extras.rs:107`, registered `m.add_class::<extras::RsIncrementalPCA>()`
`lib.rs:72`) wraps `ferrolearn_decomp::FittedIncrementalPCA<f64>` with a `(n_components: usize = 2)`
ctor and `fit` + `transform`. **Like `_RsTruncatedSVD` and UNLIKE `_RsPCA`, the macro binds ONLY
fit + transform — NO `components_`/`explained_variance_`/`explained_variance_ratio_`/`mean_`/`var_`/
`singular_values_` getters, NO `inverse_transform`, NO `partial_fit`, NO `batch_size` ctor param**
(a binding-surface gap folded into REQ-11/REQ-17 territory). Its `transform` output now carries the
correct svd_flip sign and multi-batch values (REQ-1/2/3 fixed).

## Verification

All expected values come from the live sklearn 1.5.2 oracle (run from `/tmp`), never literal-copied
from ferrolearn (R-CHAR-3). Commands establishing the SHIPPED claims:

```bash
# Value-parity REQs (1,2,3,15) + structural green-guards (4,5,6,7,8,9,10,12) — the divergence suite.
# All 14 tests PASS (3 ex-divergence pins now green after the fixer landed + 11 green-guards):
cargo test -p ferrolearn-decomp --test divergence_incremental_pca
#   divergence_svd_flip_sign (REQ-1, was #1585)
#   divergence_multibatch_mean_correction (REQ-2, was #1586)
#   divergence_explained_variance_ratio_denominator (REQ-3, was #1587)
#   green_single_batch_explained_variance_matches / green_single_batch_singular_values_matches (REQ-6/7)
#   green_fixture_a_explained_variance_and_singular_values_match (REQ-6/7)
#   green_multibatch_mean_matches_sklearn / green_fixture_a_mean_matches_sklearn (REQ-4)
#   green_shapes_and_unit_norm (REQ-5/8)   green_n_samples_seen_accumulation (REQ-10)
#   green_error_contracts / green_n_components_eq_n_features_currently_rejected (REQ-9/14)
#   green_determinism   green_f32_path (REQ-12)

# In-module unit tests (structural):
cargo test -p ferrolearn-decomp incremental_pca

# Gauntlet:
cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check

# Cross-check vs the deterministic live oracle (Probes 1 & 2, run from /tmp), values regenerated
# by sklearn and compared to a throwaway ferrolearn fit (NEVER copied from ferrolearn into the doc):
#   SINGLE 7x4 n_components=2: components incl. sign 2.2e-16, explained_variance_ 1.8e-15,
#                              ratio 2.2e-16, singular_values_ 1.3e-15, mean_/var_ 0.
#   MULTI  9x4 batch_size=3:   components incl. sign 6.3e-15, explained_variance_ 2.8e-14,
#                              singular_values_ 2.0e-14, ratio 2.4e-15, mean_/var_ 0.

# NOT-STARTED REQs (11,13,14,16,17,18,19) — open prereq blockers #1588/#1589/#1590/#1592/#1593/
# #1594/#1595: no PipelineTransformer impl; batch_size default = n_samples (not 5*n_features);
# n_components required as explicit usize + n_components == n_features rejected; no noise_variance_;
# no whiten/copy; no n_features_in_ accessor; ndarray + hand-rolled Jacobi SVD, not ferray.
```
