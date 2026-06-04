# KernelPCA (sklearn.decomposition.KernelPCA)

<!--
tier: 3-component
status: value-parity-shipped
baseline-commit: 3615b8a6
upstream: scikit-learn 1.5.2
upstream-paths:
  - sklearn/decomposition/_kernel_pca.py  # class KernelPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator) (:32). ctor (:282-316): n_components=None, *, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver="auto", tol=0, max_iter=None, iterated_power="auto", remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None. NOTE coef0=1 (:289). _get_kernel (:319-326): pairwise_kernels(X, Y, metric=kernel, gamma=gamma_, degree, coef0); gamma_ = 1/n_features when gamma is None (fit, :439). _fit_transform (:328-404): K = self._centerer.fit_transform(K) (:331, KernelCenterer double-centering); n_components = K.shape[0] if None else min(K.shape[0], n_components) (:334-337); eigen_solver auto → 'dense' for K.shape[0]<=200 or n_components>=10 (:340-344); dense: eigenvalues_, eigenvectors_ = eigh(K, subset_by_index=(N-n_comp, N-1)) (:348-352); _check_psd_eigenvalues clamps negatives (:368-370); self.eigenvectors_, _ = svd_flip(u=self.eigenvectors_, v=None) (:373, COLUMN-based u_based_decision=True); argsort eigenvalues DESCENDING (:376-378); remove_zero_eig or n_components is None → drop <=0 eigenpairs (:381-383); eigenvectors kept norm-1, scaled by 1/sqrt(eigenvalues_) deferred to fit/transform (:385-404). fit (:418-451): gamma_ = 1/X.shape[1] if gamma is None (:439); _centerer = KernelCenterer (:440); K = _get_kernel(X) (:441); _fit_transform(K) (:442); if fit_inverse_transform: X_transformed = eigenvectors_*sqrt(eigenvalues_) (:446), _fit_inverse_transform (:448); X_fit_ = X (:450). fit_transform (:453-482): X_transformed = eigenvectors_*sqrt(eigenvalues_) (:477). transform (:484-512): K = _centerer.transform(_get_kernel(X, X_fit_)) (:502); scaled_alphas = eigenvectors_[:,nz] / sqrt(eigenvalues_[nz]) (:505-509); return K @ scaled_alphas (:512). inverse_transform (:514-563): NotFittedError unless fit_inverse_transform (:555-560); K = _get_kernel(X, X_transformed_fit_); return K @ dual_coef_ (:562-563). _fit_inverse_transform (:406-416): K = _get_kernel(X_transformed); K.flat[::n+1] += alpha; dual_coef_ = linalg.solve(K, X, assume_a='pos') (:415); X_transformed_fit_ = X_transformed (:416). _n_features_out = eigenvalues_.shape[0] (:571-574).
  - sklearn/utils/extmath.py  # svd_flip(u, v=None, u_based_decision=True) (:848-906). With v=None and u_based_decision=True the u-branch (:888-895) operates per COLUMN of u: max_abs_u_cols = argmax(abs(u), axis=0) (:889, numpy argmax → FIRST max on ties); signs = sign(u[max_abs_u_cols, range(ncols)]) (:892); u *= signs[newaxis, :] (:894) → each eigenvector COLUMN's max-abs entry (across ROWS) becomes POSITIVE. Depends only on u; v=None so no Vt step.
  - sklearn/preprocessing/_data.py  # KernelCenterer — double-centering K_c = K - 1_n K - K 1_n + 1_n K 1_n; fit stores K column means + grand mean, transform centers a test gram using stored train col means + the test row means + train grand mean.
ferrolearn-module: ferrolearn-decomp/src/kernel_pca.rs
parity-ops: KernelPCA
crosslink-issue: 1561
status-note: VALUE parity SHIPPED. A critic→fixer→re-audit cycle landed the two previously-blocking divergences: svd_flip(u=eigenvectors_, v=None) per-eigenvector-COLUMN max-abs-positive sign convention (REQ-3, was #1562, FIXED, kernel_pca.rs:523-545) and the coef0 default 0.0 → 1.0 (REQ-4, was #1563, FIXED, kernel_pca.rs:102). A re-audit cross-check on a fresh 6x4 fixture confirms ferrolearn's transform(X) matches live sklearn KernelPCA(eigen_solver='dense') ELEMENT-WISE INCLUDING SIGN across ALL FOUR kernels (linear 5.8e-13, rbf 3.4e-13, poly 2.2e-15, sigmoid 3.3e-14). The 3 divergence tests + 12 structural green-guards in tests/divergence_kernel_pca.rs are live/green. STILL NOT-STARTED: arpack/randomized eigen_solver + tol/max_iter/iterated_power/random_state (#1564); remove_zero_eig (#1565); fit_inverse_transform/inverse_transform/alpha ridge + dual_coef_/X_transformed_fit_ (#1566); n_components=None + kernel='precomputed' + cosine/laplacian/chi2 + kernel_params (#1567); fitted attrs n_features_in_/X_fit_ (#1568); degenerate repeated-eigenvalue subspace CARVE-OUT (#1569, R-DEFER-3); ferray substrate (#1570). 11 SHIPPED / 7 NOT-STARTED.
-->

## Summary

`ferrolearn-decomp/src/kernel_pca.rs` mirrors scikit-learn's `KernelPCA`
(`sklearn/decomposition/_kernel_pca.py`, `class KernelPCA` `:32`): non-linear
dimensionality reduction by mapping data into a (possibly infinite-dimensional)
feature space via a kernel, then performing PCA there. **ferrolearn's `transform(X)`
matches the live sklearn `KernelPCA(..., eigen_solver='dense')` embedding ELEMENT-WISE
INCLUDING SIGN across all four kernels** (re-audit cross-check on a fresh 6×4 fixture:
linear `5.8e-13`, rbf `3.4e-13`, poly default `coef0=1`/`degree=3` `2.2e-15`, sigmoid
default `coef0=1` `3.3e-14`). The exposed surface is the unfitted `KernelPCA<F> {
n_components, kernel: Kernel{Linear|RBF|Polynomial|Sigmoid}, gamma: Option<f64>, degree,
coef0 }` (`pub struct KernelPCA in kernel_pca.rs`, builders
`with_kernel`/`with_gamma`/`with_degree`/`with_coef0`, accessors
`n_components`/`kernel`/`gamma`/`degree`/`coef0`) and the fitted `FittedKernelPCA<F> {
alphas_, eigenvalues_, x_fit_, k_fit_col_means_, k_fit_grand_mean_, kernel, gamma,
degree, coef0 }` (`pub struct FittedKernelPCA in kernel_pca.rs`, accessors
`alphas`/`eigenvalues`), re-exported at the crate root (`pub use
kernel_pca::{FittedKernelPCA, Kernel, KernelPCA}`, `lib.rs:90`) and bound to CPython
as `_RsKernelPCA` (`extras.rs:1122-1127` via the `py_transformer!` macro, registered
`m.add_class::<extras::RsKernelPCA>()` `ferrolearn-python/src/lib.rs:76`).

**ferrolearn's fit MIRRORS sklearn's `dense` eigen_solver path with VALUE PARITY**
(`_kernel_pca.py:328-404`): `fn fit` (`pub fn fit in kernel_pca.rs`) builds the
training gram `K` via `compute_kernel_matrix`/`kernel_value` (the 4 kernels), saves
`K` column means + grand mean, double-centers `K` in feature space
(`centre_kernel_matrix` — the `KernelCenterer` analogue, `_kernel_pca.py:331,440`),
eigendecomposes the symmetric centered `K` via `jacobi_eigen_symmetric` (exact dense
analogue of sklearn's `eigh` `_kernel_pca.py:350-352`), clamps negative eigenvalues to
0 (= sklearn `_check_psd_eigenvalues` `:368-370`), scales each retained eigenvector by
`1/sqrt(eigenvalue)` to form `alphas_` (= sklearn's deferred
`eigenvectors_/sqrt(eigenvalues_)` scaling `:399`/`:507-509`), sorts eigenvalues
DESCENDING (= sklearn `argsort[::-1]` `:376-378`), and **applies
`svd_flip(u=eigenvectors_, v=None)` per eigenvector COLUMN** (the per-column
max-abs-row-positive sign convention, `_kernel_pca.py:373`, `kernel_pca.rs:523-545`).
`transform` computes the test gram against `x_fit_`, double-centers it with the stored
train col-means + grand-mean + the per-test-row mean (= `KernelCenterer.transform`,
`_kernel_pca.py:502`), and returns `K_centered @ alphas_` (= sklearn `K @
scaled_alphas` `:512`).

**A critic→fixer→re-audit cycle landed the two previously-blocking divergences:**

1. **`svd_flip` sign convention (REQ-3, was `#1562`, FIXED).** sklearn pins the
   otherwise-arbitrary eigenvector signs deterministically via `self.eigenvectors_, _
   = svd_flip(u=self.eigenvectors_, v=None)` (`_kernel_pca.py:373`): with `v=None` and
   `u_based_decision=True`, `svd_flip` operates per eigenvector COLUMN — for each
   column the max-abs entry ACROSS ROWS is made POSITIVE (`extmath.py:888-895`).
   ferrolearn's `fn fit` now applies exactly this per-column max-abs-row-positive flip
   to `alphas_` (`kernel_pca.rs:530-545`, numpy `argmax` first-on-ties via strict `>`
   on `abs`). After the flip and on DISTINCT eigenvalues the embedding matches sklearn
   EXACTLY, including a multi-component independent flip (one column flipped, one kept,
   within a single rbf fit — exercised by `divergence_svd_flip_alphas_max_abs_positive`).

2. **`coef0` default (REQ-4, was `#1563`, FIXED).** sklearn's ctor default is `coef0=1`
   (`_kernel_pca.py:289`, Probe 3: `KernelPCA(kernel='poly').coef0 == 1`); ferrolearn's
   `KernelPCA::new` now defaults `coef0: 1.0` (`kernel_pca.rs:102`), so the DEFAULT
   Polynomial `(gamma·dot + coef0)^degree` and Sigmoid `tanh(gamma·dot + coef0)`
   kernels match sklearn.

`KernelPCA` / `FittedKernelPCA` / `Kernel` are existing pub APIs whose non-test
consumers are the crate re-export (`lib.rs:90`, boundary public API, grandfathered
S5/R-DEFER-1), the `_RsKernelPCA` PyO3 binding (`extras.rs:1122`, registered
`lib.rs:76`), and the `PipelineTransformer`/`FittedPipelineTransformer` impls
(`pub fn fit_pipeline`/`transform_pipeline in kernel_pca.rs`). The 3 divergence tests
(`divergence_svd_flip_embedding_sign`, `divergence_svd_flip_alphas_max_abs_positive`,
`divergence_coef0_default`) plus 12 structural green-guards are live and green in
`tests/divergence_kernel_pca.rs`. **Count: 11 SHIPPED / 7 NOT-STARTED.**

## Probes (live sklearn oracle, 1.5.2, run from /tmp, `eigen_solver='dense'`)

```bash
# PROBE 1 (REQ-3 svd_flip + value parity) — NON-DEGENERATE 7x3 fixture, distinct
# eigenvalues, eigen_solver='dense' = the deterministic eigh oracle. VALUES generated
# by sklearn (R-CHAR-3), never copied from ferrolearn. DEMONSTRATES each eigenvector
# COLUMN's max-abs entry (across ROWS) is POSITIVE after svd_flip(u=eigenvectors_, v=None).
python3 -c "
import numpy as np
from sklearn.decomposition import KernelPCA
X=np.array([[1.49,-1.7,1.33],[1.35,-0.95,-1.36],[-0.42,-0.18,0.85],[0.19,2.94,-1.49],[-0.96,-1.19,1.27],[-0.73,0.33,-0.78],[-1.21,-1.45,1.28]])
m=KernelPCA(n_components=2, kernel='rbf', gamma=0.5, eigen_solver='dense').fit(X)
T=m.transform(X)
print('eigenvalues_:', np.round(m.eigenvalues_,8).tolist())
for i in range(m.eigenvectors_.shape[1]):
    col=m.eigenvectors_[:,i]; k=int(np.argmax(np.abs(col)))
    print(f'  eigvec col[{i}] argmax-abs ROW={k} val={col[k]:.6f} (positive => svd_flip column-based)')
print('transform row0:', np.round(T[0],6).tolist())
print('transform row1:', np.round(T[1],6).tolist())"
# -> eigenvalues_: [1.63354911, 1.0480561]
# ->   eigvec col[0] argmax-abs ROW=4 val=0.538896 (positive => svd_flip column-based)
# ->   eigvec col[1] argmax-abs ROW=5 val=0.726031 (positive => svd_flip column-based)
# -> transform row0: [-0.360628, -0.447626]
# -> transform row1: [-0.462286, -0.045225]
#   => each eigenvector COLUMN's max-abs entry (across ROWS) is POSITIVE
#      (svd_flip u_based_decision=True, v=None, _kernel_pca.py:373). ferrolearn's fit now
#      applies the same per-column max-abs-row-positive flip (kernel_pca.rs:523-545), so its
#      alphas_ / embedding K_centered @ alphas matches sklearn EXACTLY incl. SIGN (REQ-3).

# PROBE 2 (REQ-5/6 eigenvalues non-negative + DESCENDING) — same fixture, 4 components.
python3 -c "
import numpy as np
from sklearn.decomposition import KernelPCA
X=np.array([[1.49,-1.7,1.33],[1.35,-0.95,-1.36],[-0.42,-0.18,0.85],[0.19,2.94,-1.49],[-0.96,-1.19,1.27],[-0.73,0.33,-0.78],[-1.21,-1.45,1.28]])
m=KernelPCA(n_components=4, kernel='rbf', gamma=0.5, eigen_solver='dense').fit(X)
print('eigenvalues_ (descending, >=0):', np.round(m.eigenvalues_,8).tolist())"
# -> eigenvalues_ (descending, >=0): [1.63354911, 1.0480561, 0.50079..., 0.41...]
#   => sklearn argsort[::-1] (:376-378) + _check_psd_eigenvalues clamp (:368-370).

# PROBE 3 (4-kernel embedding VALUE parity, fresh 6x4 fixture, sklearn DEFAULTS coef0=1,
# degree=3) — the re-audit cross-check oracle for REQ-3/4.
python3 -c "
import numpy as np
from sklearn.decomposition import KernelPCA
X=np.array([[0.8,-1.2,0.3,1.1],[1.5,0.4,-0.7,0.2],[-0.6,1.3,0.9,-1.4],[0.1,-0.5,1.6,0.7],[-1.1,0.2,-0.3,1.0],[0.9,1.1,-1.2,-0.4]])
for kern,kw in [('linear',{}),('rbf',{'gamma':0.5}),('poly',{'gamma':0.5}),('sigmoid',{'gamma':0.1})]:
    m=KernelPCA(n_components=2, kernel=kern, eigen_solver='dense', **kw).fit(X)
    T=m.transform(X)
    print(f'{kern}: eigvals={np.round(m.eigenvalues_,6).tolist()} T[0]={np.round(T[0],6).tolist()} coef0={m.coef0}')"
# -> linear:  eigvals=[8.993334, 7.17047]    T[0]=[1.603331, -0.658291]  coef0=1
# -> rbf:     eigvals=[1.33765, 1.069635]    T[0]=[-0.397749, -0.44783]  coef0=1
# -> poly:    eigvals=[37.090875, 27.976115] T[0]=[-1.415538, -2.382276] coef0=1
# -> sigmoid: eigvals=[0.364844, 0.277378]   T[0]=[0.334946, -0.14363]   coef0=1
#   => sklearn coef0 DEFAULT is 1 (_kernel_pca.py:289); ferrolearn's KernelPCA::new now
#      defaults coef0=1.0 (kernel_pca.rs:102), so poly/sigmoid DEFAULT kernels match.
#      ferrolearn's transform matches each kernel ELEMENT-WISE INCLUDING SIGN (linear 5.8e-13,
#      rbf 3.4e-13, poly 2.2e-15, sigmoid 3.3e-14). ferrolearn exposes NEITHER alpha/
#      fit_inverse_transform (REQ-14), eigen_solver/tol/max_iter/iterated_power (REQ-12),
#      remove_zero_eig (REQ-13), nor n_components=None (REQ-15).
```

## Requirements

- REQ-1: **The 4 kernels (Linear / RBF / Polynomial / Sigmoid) compute correct kernel
  values (SHIPPED).** sklearn's `_get_kernel` (`_kernel_pca.py:319-326`) dispatches to
  `pairwise_kernels(metric=kernel, gamma, degree, coef0)`: Linear `x·y`, RBF
  `exp(-gamma·‖x−y‖²)`, Polynomial `(gamma·x·y + coef0)^degree`, Sigmoid
  `tanh(gamma·x·y + coef0)`. ferrolearn's `kernel_value` (`fn kernel_value in
  kernel_pca.rs`) computes exactly these per `Kernel` variant (Linear=dot, RBF=`(-γ·
  sq_dist).exp()`, Poly=`(γ·dot+coef0)` raised to `degree` by repeated multiply,
  Sigmoid=`(γ·dot+coef0).tanh()`), tiled into the gram by `compute_kernel_matrix`
  (`fn compute_kernel_matrix in kernel_pca.rs`). With the now-correct `coef0=1` default
  (REQ-4) the per-kernel embedding matches sklearn element-wise (Probe 3: linear
  `5.8e-13`, rbf `3.4e-13`, poly `2.2e-15`, sigmoid `3.3e-14`). Non-test consumers:
  re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Pinned by
  `test_kernel_pca_linear_basic`, `_rbf_basic`, `_polynomial_basic`, `_sigmoid_basic`,
  `_rbf_sensitivity_to_gamma` (in-module) and `green_four_kernels_finite_shape`
  (`divergence_kernel_pca.rs`).

- REQ-2: **Double-centering (`KernelCenterer`) of the train + test grams (SHIPPED).**
  sklearn centers the training gram via `K = self._centerer.fit_transform(K)`
  (`_kernel_pca.py:331`, `KernelCenterer`, `:440`) — `K_c = K − 1_n K − K 1_n + 1_n K
  1_n` — and centers a test gram via `self._centerer.transform(...)`
  (`_kernel_pca.py:502`) using stored train col-means + grand-mean + each test row's
  mean. ferrolearn's `fn fit` calls `centre_kernel_matrix` (`fn centre_kernel_matrix
  in kernel_pca.rs`, the `K[i,j] − col_mean[i] − col_mean[j] + grand_mean`
  double-centering for the symmetric train gram) and `transform` re-centers the test
  gram as `k_test[i,j] − k_fit_col_means_[j] − row_mean[i] + k_fit_grand_mean_`
  (the train-statistics + test-row-mean centering matching `KernelCenterer.transform`).
  With REQ-3's svd_flip the resulting embedding matches sklearn element-wise including
  sign (Probe 1). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA`
  `extras.rs:1122`. Pinned by `test_kernel_pca_transform_new_data`,
  `_linear_resembles_pca` (in-module) and `green_transform_new_data_shape`
  (`divergence_kernel_pca.rs`).

- REQ-3: **`alphas_` / embedding sign via `svd_flip(u=eigenvectors_, v=None)`
  (column-based) + EXACT value parity (SHIPPED — was `#1562`, FIXED).** sklearn pins
  the eigenvector signs deterministically: `self.eigenvectors_, _ =
  svd_flip(u=self.eigenvectors_, v=None)` (`_kernel_pca.py:373`); with `v=None` and
  `u_based_decision=True`, `svd_flip` (`extmath.py:888-895`) takes `argmax(abs(u),
  axis=0)` per eigenvector COLUMN (`:889`, numpy `argmax` → first-max on ties),
  `signs = sign(u[max_abs_rows, range(ncols)])` (`:892`), `u *= signs[newaxis, :]`
  (`:894`) → each eigenvector COLUMN's max-abs entry ACROSS ROWS becomes POSITIVE
  (Probe 1: col 0 max-abs row 4, col 1 max-abs row 5, both positive). ferrolearn's
  `fn fit` now applies the identical per-column max-abs-row-positive flip on `alphas`
  (`kernel_pca.rs:530-545`: for each column, find the max-`abs` row via strict `>`
  = numpy first-on-ties, negate the column when that entry is `< 0`). Because the
  positive `1/sqrt(eigenvalue)` scale is sign-preserving, flipping the scaled `alphas`
  column equals flipping the raw eigenvector column. On distinct eigenvalues the
  embedding `K_centered @ alphas_` matches sklearn EXACTLY including sign — a
  multi-component independent flip (one column flipped, one kept within a single rbf
  fit) is exercised. Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA`
  `extras.rs:1122`. Pinned by `divergence_svd_flip_embedding_sign` (element-wise incl.
  sign vs the live dense oracle, Probe 1) and `divergence_svd_flip_alphas_max_abs_positive`
  (every `alphas` column's max-abs entry is positive), `tests/divergence_kernel_pca.rs`.

- REQ-4: **`coef0` DEFAULT = 1 (SHIPPED — was `#1563`, FIXED).** sklearn's ctor default
  is `coef0=1` (`_kernel_pca.py:289`, Probe 3: `KernelPCA(kernel='poly').coef0 == 1`);
  ferrolearn's `KernelPCA::new` now defaults `coef0: 1.0` (`kernel_pca.rs:102`), so the
  DEFAULT Polynomial `(gamma·dot + coef0)^degree` and Sigmoid `tanh(gamma·dot + coef0)`
  kernels match sklearn. The poly/sigmoid embedding under defaults matches sklearn
  element-wise (Probe 3: poly `2.2e-15`, sigmoid `3.3e-14`). Non-test consumers:
  re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122` (the PyO3 binding uses this
  Rust-side default). Pinned by `divergence_coef0_default` (`coef0()` of a fresh
  `KernelPCA::new` equals `1.0`, the sklearn default), `tests/divergence_kernel_pca.rs`.

- REQ-5: **Eigenvalues NON-NEGATIVE (SHIPPED).** sklearn clamps via
  `_check_psd_eigenvalues` (`_kernel_pca.py:368-370`). ferrolearn's `fn fit` clamps
  each retained eigenvalue `eigval_clamped = if eigval > 0 { eigval } else { 0 }`. The
  retained eigenvalues also match sklearn element-wise (Probe 1: `[1.63354911,
  1.0480561]`). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA`
  `extras.rs:1122`. Pinned by `test_kernel_pca_eigenvalues_non_negative` (in-module),
  `green_eigenvalues_match_sklearn` + `green_eigenvalues_nonneg_descending`
  (`divergence_kernel_pca.rs`).

- REQ-6: **Eigenvalues SORTED DESCENDING (SHIPPED).** sklearn `indices =
  eigenvalues_.argsort()[::-1]` (`_kernel_pca.py:376-378`). ferrolearn's `fn fit`
  sorts the eigenvalue indices descending (`indices.sort_by(|&a,&b|
  eigenvalues[b].partial_cmp(&eigenvalues[a]))`) and takes the top `n_components`, then
  applies the REQ-3 svd_flip. The descending eigenvalues match sklearn element-wise
  (Probe 2). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`.
  Pinned by `test_kernel_pca_eigenvalues_sorted_descending` (in-module),
  `green_eigenvalues_nonneg_descending` (`divergence_kernel_pca.rs`).

- REQ-7: **Embedding shape `(n_samples, n_components)` + `alphas_` scaling
  `eigenvector/sqrt(eigenvalue)` (SHIPPED).** sklearn defers scaling each norm-1
  eigenvector by `1/sqrt(eigenvalues_)` to `transform`'s `scaled_alphas`
  (`_kernel_pca.py:505-509`), then `K @ scaled_alphas` is `(n_samples, n_components)`
  (`:512`). ferrolearn pre-scales `alphas[[i,k]] = eigenvectors[[i,idx]] · scale` with
  `scale = 1/sqrt(eigval_clamped)` (guarded to 0 for ~0 eigenvalues, matching sklearn's
  `np.flatnonzero(eigenvalues_)` null-space guard `:505`), then applies the REQ-3 flip,
  shape `(n_samples, n_components)`; `transform` returns `k_centered.dot(&alphas_)`.
  With REQ-3/4 the magnitudes AND signs match sklearn element-wise (Probe 1/3). Non-test
  consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`, `PipelineTransformer`.
  Pinned by `test_kernel_pca_single_component`, `_max_components_equals_samples`
  (in-module), `green_embedding_matches_sklearn_up_to_sign` +
  `divergence_svd_flip_embedding_sign` (`divergence_kernel_pca.rs`).

- REQ-8: **`transform` of NEW data (centered with train statistics) (SHIPPED).**
  sklearn's `transform` (`_kernel_pca.py:484-512`) computes the test gram
  `_get_kernel(X, X_fit_)`, centers it with the stored `KernelCenterer` train
  statistics (`:502`), and projects `K @ scaled_alphas` (`:512`). ferrolearn's
  `transform` (`pub fn transform in kernel_pca.rs`) computes `compute_kernel_matrix(x,
  x_fit_, ...)`, centers each test row with the stored `k_fit_col_means_` /
  `k_fit_grand_mean_` + that row's own mean, and returns `k_centered.dot(&alphas_)`,
  matching sklearn element-wise including sign (the projection inherits REQ-3's flipped
  `alphas_`). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`.
  Pinned by `test_kernel_pca_transform_new_data` (in-module),
  `green_transform_new_data_shape` (`divergence_kernel_pca.rs`).

- REQ-9: **Auto-gamma default `1/n_features` (SHIPPED).** sklearn sets `gamma_ = 1 /
  X.shape[1] if self.gamma is None` (`_kernel_pca.py:439`). ferrolearn's `fn fit`
  computes `effective_gamma = self.gamma.unwrap_or(1.0 / n_features as f64)` and
  threads it into both the train gram and the stored `gamma` used by `transform`.
  Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Pinned
  by `test_kernel_pca_auto_gamma` (in-module), `green_auto_gamma`
  (`divergence_kernel_pca.rs`).

- REQ-10: **Error / parameter contracts (SHIPPED scoped).** `fn fit` returns
  `InvalidParameter { name: "n_components" }` for `n_components == 0` and for
  `n_components > n_samples`, and `InsufficientSamples { required: 2 }` for
  `n_samples < 2`; `transform` returns `ShapeMismatch` on a feature-count mismatch
  against `x_fit_.ncols()`. **FLAG (candidate DIVs):** sklearn validates via
  `_parameter_constraints` raising `InvalidParameterError`/`ValueError`, NOT
  `FerroError`; sklearn accepts `n_components=None` (→ all dims, `_kernel_pca.py:334`)
  and CAPS `n_components` at `min(K.shape[0], n_components)` (`:337`) rather than
  erroring when it exceeds n_samples; sklearn does not pre-reject `n_samples < 2`.
  Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Pinned
  by `test_kernel_pca_invalid_n_components_zero`, `_invalid_n_components_too_large`,
  `_insufficient_samples`, `_shape_mismatch_transform` (in-module), and
  `green_err_n_components_zero` / `_too_large` / `_insufficient_samples` /
  `_transform_feature_mismatch` (`divergence_kernel_pca.rs`).

- REQ-11: **f32 / f64 generic support (SHIPPED).** `KernelPCA<F>` / `FittedKernelPCA<F>`
  / `kernel_value` / `compute_kernel_matrix` / `centre_kernel_matrix` /
  `jacobi_eigen_symmetric` are generic over `F: Float + Send + Sync + 'static`
  (`gamma`/`coef0` stored as `f64`, cast in `kernel_value` via `F::from`; the svd_flip
  comparisons use `F::zero()`). sklearn declares `preserves_dtype: [float64, float32]`
  (`_kernel_pca.py:565-569`). Non-test consumer: re-export `lib.rs:90`. Pinned by
  `test_kernel_pca_f32` (in-module), `green_f32_path` (`divergence_kernel_pca.rs`).

- REQ-12: **`eigen_solver` arpack / randomized + `tol` / `max_iter` /
  `iterated_power` / `random_state` (NOT-STARTED; `#1564`).** sklearn's
  `eigen_solver="auto"` (`_kernel_pca.py:293`) selects `'arpack'` (`eigsh`,
  `:353-357`) for large `K.shape[0] > 200` with `n_components < 10`, else `'dense'`
  (`eigh`, `:340-352`), and supports `'randomized'` (`_randomized_eigsh`, `:358-365`),
  governed by `tol` (`:294`), `max_iter` (`:295`), `iterated_power` (`:296`), and
  `random_state` (`:298`). ferrolearn implements ONLY the `'dense'`-equivalent exact
  `jacobi_eigen_symmetric` (`fn jacobi_eigen_symmetric in kernel_pca.rs`) — no
  `eigen_solver` field, no arpack/randomized path, no `tol`/`max_iter`/
  `iterated_power`/`random_state`.

- REQ-13: **`remove_zero_eig` (NOT-STARTED; `#1565`).** sklearn's
  `remove_zero_eig=False` (`_kernel_pca.py:297`) drops eigenpairs with ≤ 0 eigenvalue
  (`self.eigenvectors_[:, self.eigenvalues_ > 0]`, `:381-383`) — also forced when
  `n_components is None`. ferrolearn's `fn fit` always retains exactly the top
  `n_components` (clamping negatives to 0 but never dropping them), with no
  `remove_zero_eig` field.

- REQ-14: **`alpha` ridge + `fit_inverse_transform` / `inverse_transform` +
  `dual_coef_` / `X_transformed_fit_` (NOT-STARTED; `#1566`).** sklearn's
  `fit_inverse_transform=False` ctor (`_kernel_pca.py:292`) optionally learns a kernel
  ridge pre-image: `_fit_inverse_transform` (`:406-416`) solves `(K_tt + alpha·I)
  dual_coef_ = X` (`alpha=1.0` `:291`, `:414-415`) on `X_transformed = eigenvectors_·
  sqrt(eigenvalues_)` (`:446`), storing `dual_coef_` + `X_transformed_fit_`;
  `inverse_transform` (`:514-563`) returns `_get_kernel(X, X_transformed_fit_) @
  dual_coef_` (`:562-563`) or raises `NotFittedError` when not requested (`:555-560`).
  ferrolearn's `FittedKernelPCA<F>` has no `alpha`, `dual_coef_`, `X_transformed_fit_`,
  or `inverse_transform` — the inverse pre-image path is entirely absent.

- REQ-15: **`n_components=None` (all components) + `kernel='precomputed'` + extra
  kernels (cosine / laplacian / chi2) + `kernel_params` (NOT-STARTED; `#1567`).**
  sklearn's ctor default is `n_components=None` (`_kernel_pca.py:284`) → `n_components =
  K.shape[0]` (all dimensions, `:334-335`) and forces `remove_zero_eig` (`:381`); its
  `_get_kernel` (`:319-326`) routes through `pairwise_kernels`, which supports
  `metric='precomputed'`, the full pairwise-kernel catalogue (cosine, laplacian, chi2,
  additive_chi2, …), and a callable `kernel` with `kernel_params` (`:290`,`:320-321`).
  ferrolearn's `KernelPCA::new(n_components: usize)` requires an explicit integer count
  (no `None`/all-components default) and its `Kernel` enum has ONLY `{Linear, RBF,
  Polynomial, Sigmoid}` — no precomputed, no extra metrics, no `kernel_params`, no
  callable kernel.

- REQ-16: **Fitted attrs `n_features_in_` / `X_fit_` exposure (NOT-STARTED; `#1568`).**
  sklearn exposes `X_fit_` (`_kernel_pca.py:450`), `n_features_in_`, and
  `_n_features_out = eigenvalues_.shape[0]` (`:571-574`). ferrolearn stores `x_fit_`
  internally (used by `transform`) but exposes NO `X_fit_` accessor and has no
  `n_features_in_` / `_n_features_out` attribute.

- REQ-17: **Degenerate / repeated-eigenvalue subspace VALUE carve-out (NOT-STARTED,
  CARVE-OUT; `#1569`, R-DEFER-3).** On a repeated eigenvalue (e.g. a symmetric/isotropic
  fixture where two leading eigenvalues coincide), the Jacobi eigensolver and LAPACK
  `eigh` pick DIFFERENT orthonormal bases for the degenerate eigenspace, so the
  embedding COLUMNS are ambiguous even AFTER the REQ-3 sign flip (only the spanned
  subspace + the eigenvalues are well-defined). Same class as `spectral_embedding` /
  PCA degenerate carve-outs. **CARVE-OUT (R-DEFER-3):** no failing test is asserted —
  the ambiguity is inherent to the eigensolver, not a ferrolearn defect.

- REQ-18: **ferray substrate (NOT-STARTED; `#1570`).** `kernel_pca.rs` computes on
  `ndarray::{Array1, Array2}` and eigendecomposes via a hand-rolled
  `jacobi_eigen_symmetric` (`fn jacobi_eigen_symmetric in kernel_pca.rs`), not
  `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2).

## Acceptance criteria

All expected values from the live sklearn 1.5.2 oracle (run from `/tmp`,
`eigen_solver='dense'` for determinism), never literal-copied from ferrolearn
(R-CHAR-3).

- AC-1 (REQ-1/2/9, SHIPPED): `kernel_value` reproduces `pairwise_kernels` per kernel;
  `centre_kernel_matrix` reproduces `KernelCenterer` double-centering of the train
  gram; `transform` re-centers a test gram with train statistics; auto-gamma is
  `1/n_features`. Pinned by `test_kernel_pca_{linear,rbf,polynomial,sigmoid}_basic`,
  `_auto_gamma`, `_transform_new_data`, `_linear_resembles_pca`,
  `_rbf_sensitivity_to_gamma`, and `green_four_kernels_finite_shape`/`green_auto_gamma`.

- AC-2 (REQ-3, SHIPPED): `KernelPCA::new(k).with_kernel(RBF).with_gamma(g).fit(&X)
  .transform(&X)` equals the live `KernelPCA(n_components=k, kernel='rbf', gamma=g,
  eigen_solver='dense').transform(X)` element-wise INCLUDING sign on a non-degenerate
  fixture (Probe 1: `transform row0 = [-0.360628, -0.447626]`, row1 = `[-0.462286,
  -0.045225]`) because `fn fit` applies `svd_flip(u=eigenvectors_, v=None)`
  (per-eigenvector-column max-abs-positive). Pinned by `divergence_svd_flip_embedding_sign`
  and `divergence_svd_flip_alphas_max_abs_positive`.

- AC-3 (REQ-4, SHIPPED): `KernelPCA::new(k).with_kernel(Polynomial)` WITHOUT
  `with_coef0` uses `coef0 = 1.0`, matching the sklearn default `coef0 = 1` (Probe 3),
  so the default Polynomial/Sigmoid kernels agree (Probe 3: poly `2.2e-15`, sigmoid
  `3.3e-14`). Pinned by `divergence_coef0_default`.

- AC-4 (REQ-5/6/7, SHIPPED): `fitted.eigenvalues()` entries are ≥ 0, descending, and
  match sklearn element-wise (Probe 1: `[1.63354911, 1.0480561]`); `fitted.alphas()` /
  `transform(X)` are `(n_samples, n_components)` and match sklearn including sign.
  Pinned by `test_kernel_pca_eigenvalues_non_negative`,
  `_eigenvalues_sorted_descending`, `_single_component`,
  `_max_components_equals_samples`, and `green_eigenvalues_match_sklearn`.

- AC-5 (REQ-10/11, SHIPPED scoped): `fit` `Err`s for `n_components=0`,
  `n_components > n_samples`, `n_samples < 2`; `transform` `Err`s on a feature-count
  mismatch; `KernelPCA::<f32>::new(1).fit(&X).transform(&X)` has 1 column. Pinned by
  `test_kernel_pca_invalid_n_components_zero`, `_invalid_n_components_too_large`,
  `_insufficient_samples`, `_shape_mismatch_transform`, `_f32`, and the
  `green_err_*` / `green_f32_path` guards. FLAG: sklearn raises
  `InvalidParameterError`/`ValueError`, accepts `n_components=None`, caps at
  `min(K.shape[0], n_components)`, and does not pre-reject `n_samples < 2`.

- AC-6 (REQ-12..16/18, NOT-STARTED): `KernelPCA()` defaults `n_components=None,
  kernel="linear", gamma=None, degree=3, coef0=1, alpha=1.0,
  fit_inverse_transform=False, eigen_solver="auto", tol=0, max_iter=None,
  iterated_power="auto", remove_zero_eig=False, random_state=None` (Probe 3,
  `_kernel_pca.py:282-316`); sklearn exposes the arpack/randomized solvers + their
  params (`#1564`), `remove_zero_eig` (`#1565`), the `alpha`/`fit_inverse_transform`/
  `inverse_transform` ridge pre-image (`#1566`), `n_components=None` + `precomputed` +
  the extra pairwise-kernel catalogue + `kernel_params` (`#1567`), and
  `X_fit_`/`n_features_in_` (`#1568`). ferrolearn has none of these and computes on
  `ndarray` + hand-rolled Jacobi, not ferray (`#1570`).

- AC-7 (REQ-17, NOT-STARTED, CARVE-OUT): on a symmetric/isotropic input with a
  repeated leading eigenvalue, the embedding COLUMNS are basis-ambiguous (Jacobi ≠
  LAPACK) even after the REQ-3 flip; the eigenvalues and spanned subspace still match.
  No failing test asserts element-wise value parity (`#1569`, R-DEFER-3).

## REQ status

Binary (R-DEFER-2). `KernelPCA` / `FittedKernelPCA` / `Kernel` are existing pub APIs;
the non-test consumers are the crate re-export (`lib.rs:90`, boundary public API,
grandfathered S5/R-DEFER-1), the `_RsKernelPCA` PyO3 binding (`extras.rs:1122`,
registered `lib.rs:76`), and the `PipelineTransformer` impls (`pub fn
fit_pipeline`/`transform_pipeline in kernel_pca.rs`). Cites use symbol anchors
(ferrolearn) / `file:line` (sklearn 1.5.2). Live oracle = installed sklearn 1.5.2, run
from `/tmp` with `eigen_solver='dense'`.
**ferrolearn's fit MIRRORS sklearn's `dense` eigen_solver with VALUE PARITY**
(`_kernel_pca.py:328-404`) — the 4 kernels, `KernelCenterer` double-centering, exact
Jacobi eigendecomposition, descending sort, negative-clamp, `1/sqrt(eigenvalue)`
scaling, the `svd_flip(u=eigenvectors_, v=None)` per-column max-abs-positive sign
convention (`_kernel_pca.py:373`, `kernel_pca.rs:523-545`), and the `K_centered @
alphas` projection. A critic→fixer→re-audit cycle landed the two previously-blocking
divergences: REQ-3 svd_flip (was `#1562`, FIXED) and REQ-4 `coef0` default 0.0→1.0
(was `#1563`, FIXED). A re-audit cross-check confirms `transform(X)` matches the live
dense oracle ELEMENT-WISE INCLUDING SIGN across all 4 kernels (linear `5.8e-13`, rbf
`3.4e-13`, poly `2.2e-15`, sigmoid `3.3e-14`). The degenerate repeated-eigenvalue
subspace (REQ-17, `#1569`) is a CARVE-OUT with no failing test (R-DEFER-3). `#1561` is
this doc's crosslink tracking issue. Count: **11 SHIPPED (REQ-1,2,3,4,5,6,7,8,9,10,11)
/ 7 NOT-STARTED (REQ-12,13,14,15,16,17,18)**.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (4 kernels compute correct values) | SHIPPED | sklearn `_get_kernel` → `pairwise_kernels(metric=kernel, gamma, degree, coef0)` (`_kernel_pca.py:319-326`): Linear `x·y`, RBF `exp(-γ‖x−y‖²)`, Poly `(γ·dot+coef0)^degree`, Sigmoid `tanh(γ·dot+coef0)`. ferrolearn `fn kernel_value in kernel_pca.rs` computes exactly these per `Kernel` variant, tiled by `fn compute_kernel_matrix in kernel_pca.rs`; with the now-correct `coef0=1` default (REQ-4) the per-kernel embedding matches sklearn (Probe 3: linear `5.8e-13`, rbf `3.4e-13`, poly `2.2e-15`, sigmoid `3.3e-14`). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_{linear,rbf,polynomial,sigmoid}_basic`, `_rbf_sensitivity_to_gamma`, `green_four_kernels_finite_shape` PASS. |
| REQ-2 (`KernelCenterer` double-centering of train + test grams) | SHIPPED | sklearn `K = _centerer.fit_transform(K)` (`_kernel_pca.py:331`, `KernelCenterer` `:440`) and `_centerer.transform(...)` (`:502`). ferrolearn `fn centre_kernel_matrix in kernel_pca.rs` double-centers the symmetric train gram (`K[i,j] − col_mean[i] − col_mean[j] + grand_mean`); `transform` re-centers each test row with stored `k_fit_col_means_`/`k_fit_grand_mean_` + that row's mean (matching `KernelCenterer.transform`). With REQ-3's svd_flip the embedding matches sklearn element-wise incl. sign (Probe 1). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_transform_new_data`, `_linear_resembles_pca`, `green_transform_new_data_shape` PASS. |
| REQ-3 (`alphas_`/embedding sign via `svd_flip(u,v=None)` COLUMN-based + exact value parity) | SHIPPED | was `#1562`, FIXED. sklearn `self.eigenvectors_, _ = svd_flip(u=self.eigenvectors_, v=None)` (`_kernel_pca.py:373`); the u-branch (`extmath.py:888-895`) takes `argmax(abs(u), axis=0)` per eigenvector COLUMN (`:889`, numpy first-max), `signs = sign(u[max_abs_rows, range(ncols)])` (`:892`), `u *= signs[newaxis,:]` (`:894`) → each COLUMN's max-abs entry ACROSS ROWS POSITIVE (Probe 1: col 0 row 4, col 1 row 5). ferrolearn `fn fit` now applies the identical per-column max-abs-row-positive flip on `alphas` (`kernel_pca.rs:530-545`: strict `>` on `abs` = numpy first-on-ties, negate column when its max-abs entry is `< 0`); the positive `1/sqrt(eigenvalue)` scale is sign-preserving so flipping scaled `alphas` = flipping the raw eigenvector. On distinct eigenvalues the embedding `K_centered @ alphas` matches sklearn EXACTLY incl. sign (multi-component independent flip exercised). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp --test divergence_kernel_pca` → `divergence_svd_flip_embedding_sign` (row0 `[-0.360628,-0.447626]`, row1 `[-0.462286,-0.045225]`, element-wise incl. sign), `divergence_svd_flip_alphas_max_abs_positive` PASS. |
| REQ-4 (`coef0` DEFAULT = 1) | SHIPPED | was `#1563`, FIXED. sklearn `coef0=1` (`_kernel_pca.py:289`, Probe 3: `KernelPCA(kernel='poly').coef0 == 1`); ferrolearn `KernelPCA::new` now defaults `coef0: 1.0` (`kernel_pca.rs:102`), so the DEFAULT Polynomial `(γ·dot+coef0)^degree` and Sigmoid `tanh(γ·dot+coef0)` kernels match sklearn (Probe 3: poly `2.2e-15`, sigmoid `3.3e-14`). The PyO3 `_RsKernelPCA` binding uses this Rust-side default. Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp --test divergence_kernel_pca` → `divergence_coef0_default` (`coef0()` of a fresh `KernelPCA::new` == `1.0`) PASS. |
| REQ-5 (eigenvalues NON-NEGATIVE) | SHIPPED | sklearn `_check_psd_eigenvalues` clamps (`_kernel_pca.py:368-370`). ferrolearn `fn fit` clamps `eigval_clamped = if eigval > 0 { eigval } else { 0 }`; the retained eigenvalues match sklearn element-wise (Probe 1: `[1.63354911, 1.0480561]`). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_eigenvalues_non_negative`, `green_eigenvalues_match_sklearn`, `green_eigenvalues_nonneg_descending` PASS. |
| REQ-6 (eigenvalues SORTED DESCENDING) | SHIPPED | sklearn `indices = eigenvalues_.argsort()[::-1]` (`_kernel_pca.py:376-378`). ferrolearn `fn fit` sorts indices descending (`indices.sort_by(|&a,&b| eigenvalues[b].partial_cmp(&eigenvalues[a]))`), takes top `n_components`, then applies the REQ-3 svd_flip; the descending eigenvalues match sklearn (Probe 2). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_eigenvalues_sorted_descending`, `green_eigenvalues_nonneg_descending` PASS. |
| REQ-7 (embedding shape `(n_samples, n_components)` + `1/sqrt(eigenvalue)` scaling) | SHIPPED | sklearn defers scaling each norm-1 eigenvector by `1/sqrt(eigenvalues_)` (`scaled_alphas` `_kernel_pca.py:505-509`), `K @ scaled_alphas` is `(n_samples, n_components)` (`:512`). ferrolearn pre-scales `alphas[[i,k]] = eigenvectors[[i,idx]]·(1/sqrt(eigval_clamped))` (guarded to 0 for ~0 eigenvalues, matching sklearn's `np.flatnonzero` null-space guard `:505`), then applies the REQ-3 flip; `transform` returns `k_centered.dot(&alphas_)`. With REQ-3/4 the magnitudes AND signs match sklearn element-wise (Probe 1/3). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`, `PipelineTransformer`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_single_component`, `_max_components_equals_samples`, `green_embedding_matches_sklearn_up_to_sign`, `divergence_svd_flip_embedding_sign` PASS. |
| REQ-8 (`transform` of NEW data, centered with train statistics) | SHIPPED | sklearn `transform` (`_kernel_pca.py:484-512`): test gram `_get_kernel(X, X_fit_)`, `_centerer.transform` (`:502`), `K @ scaled_alphas` (`:512`). ferrolearn `pub fn transform in kernel_pca.rs`: `compute_kernel_matrix(x, x_fit_, ...)`, re-center each test row with stored `k_fit_col_means_`/`k_fit_grand_mean_` + row mean, return `k_centered.dot(&alphas_)` (inherits REQ-3's flipped `alphas_`, so matches sklearn incl. sign). Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_transform_new_data`, `green_transform_new_data_shape` PASS. |
| REQ-9 (auto-gamma `1/n_features`) | SHIPPED | sklearn `gamma_ = 1/X.shape[1] if gamma is None` (`_kernel_pca.py:439`). ferrolearn `fn fit`: `effective_gamma = self.gamma.unwrap_or(1.0 / n_features as f64)`, threaded into the train gram and the stored `gamma` for `transform`. Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_auto_gamma` (3 features → gamma 1/3), `green_auto_gamma` PASS. |
| REQ-10 (error / parameter contracts, scoped) | SHIPPED | `fn fit` returns `Err(InvalidParameter{name:"n_components"})` for `==0` and for `> n_samples`, `Err(InsufficientSamples{required:2})` for `< 2` samples; `transform` `Err(ShapeMismatch)` on feature-count mismatch vs `x_fit_.ncols()`. Non-test consumers: re-export `lib.rs:90`, `_RsKernelPCA` `extras.rs:1122`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_invalid_n_components_zero`, `_invalid_n_components_too_large`, `_insufficient_samples`, `_shape_mismatch_transform`, `green_err_n_components_zero`/`_too_large`/`_insufficient_samples`/`_transform_feature_mismatch` PASS. **FLAG (candidate DIVs):** sklearn validates via `_parameter_constraints` raising `InvalidParameterError`/`ValueError` (not `FerroError`); accepts `n_components=None` (`_kernel_pca.py:334`); caps at `min(K.shape[0], n_components)` (`:337`) instead of erroring; does NOT pre-reject `n_samples < 2`. |
| REQ-11 (f32 / f64 generic support) | SHIPPED | `KernelPCA<F>`/`FittedKernelPCA<F>`/`kernel_value`/`compute_kernel_matrix`/`centre_kernel_matrix`/`jacobi_eigen_symmetric` generic over `F: Float + Send + Sync + 'static` (`gamma`/`coef0` stored as `f64`, cast via `F::from`; svd_flip comparisons use `F::zero()`). sklearn `preserves_dtype: [float64, float32]` (`_kernel_pca.py:565-569`). Non-test consumer: re-export `lib.rs:90`. Verification: `cargo test -p ferrolearn-decomp` → `test_kernel_pca_f32`, `green_f32_path` PASS. |
| REQ-12 (`eigen_solver` arpack/randomized + `tol`/`max_iter`/`iterated_power`/`random_state`) | NOT-STARTED | open prereq blocker `#1564`. sklearn `eigen_solver="auto"` (`_kernel_pca.py:293`) selects `'arpack'` (`eigsh` `:353-357`) for `K.shape[0]>200` & `n_components<10`, else `'dense'` (`eigh` `:340-352`); `'randomized'` (`_randomized_eigsh` `:358-365`); governed by `tol`/`max_iter`/`iterated_power`/`random_state` (`:294-298`). ferrolearn implements ONLY the `'dense'`-equivalent exact `fn jacobi_eigen_symmetric in kernel_pca.rs` — no `eigen_solver` field, no arpack/randomized, no `tol`/`max_iter`/`iterated_power`/`random_state`. |
| REQ-13 (`remove_zero_eig`) | NOT-STARTED | open prereq blocker `#1565`. sklearn `remove_zero_eig=False` (`_kernel_pca.py:297`) drops eigenpairs with ≤ 0 eigenvalue (`eigenvectors_[:, eigenvalues_ > 0]` `:381-383`), forced when `n_components is None`. ferrolearn `fn fit` always retains exactly the top `n_components` (clamping negatives to 0, never dropping), no `remove_zero_eig` field. |
| REQ-14 (`alpha` ridge + `fit_inverse_transform`/`inverse_transform` + `dual_coef_`/`X_transformed_fit_`) | NOT-STARTED | open prereq blocker `#1566`. sklearn `_fit_inverse_transform` (`_kernel_pca.py:406-416`) solves `(K_tt + alpha·I) dual_coef_ = X` (`alpha=1.0` `:291`, `:414-415`) on `X_transformed = eigenvectors_·sqrt(eigenvalues_)` (`:446`), storing `dual_coef_`/`X_transformed_fit_`; `inverse_transform` (`:514-563`) returns `_get_kernel(X, X_transformed_fit_) @ dual_coef_` (`:562-563`) or raises `NotFittedError` (`:555-560`). ferrolearn `FittedKernelPCA<F>` has no `alpha`/`dual_coef_`/`X_transformed_fit_`/`inverse_transform` — the pre-image path is entirely absent. |
| REQ-15 (`n_components=None` all components + `kernel='precomputed'` + cosine/laplacian/chi2 + `kernel_params` + callable) | NOT-STARTED | open prereq blocker `#1567`. sklearn ctor default `n_components=None` (`_kernel_pca.py:284`) → `n_components = K.shape[0]` (`:334-335`), forces `remove_zero_eig` (`:381`); `_get_kernel` (`:319-326`) routes through `pairwise_kernels` (`metric='precomputed'`, the full kernel catalogue, callable + `kernel_params` `:290`,`:320-321`). ferrolearn `KernelPCA::new(n_components: usize)` requires an explicit integer (no `None`/all-components default); `Kernel` enum has ONLY `{Linear, RBF, Polynomial, Sigmoid}` — no precomputed/extra metrics/`kernel_params`/callable. |
| REQ-16 (fitted attrs `n_features_in_`/`X_fit_`) | NOT-STARTED | open prereq blocker `#1568`. sklearn exposes `X_fit_` (`_kernel_pca.py:450`), `n_features_in_`, and `_n_features_out = eigenvalues_.shape[0]` (`:571-574`). ferrolearn stores `x_fit_` internally (used by `transform`) but exposes NO `X_fit_` accessor and has no `n_features_in_`/`_n_features_out`. |
| REQ-17 (degenerate / repeated-eigenvalue subspace VALUE carve-out) | NOT-STARTED | open prereq blocker `#1569` (CARVE-OUT, R-DEFER-3). On a repeated eigenvalue (a symmetric/isotropic fixture where two leading eigenvalues coincide), `fn jacobi_eigen_symmetric in kernel_pca.rs` and LAPACK `eigh` (`_kernel_pca.py:350`) pick DIFFERENT orthonormal bases for the degenerate eigenspace, so the embedding COLUMNS are ambiguous even AFTER the REQ-3 sign flip (only the spanned subspace + eigenvalues are well-defined). Same class as `spectral_embedding` / PCA degenerate carve-outs. No failing test (R-DEFER-3). |
| REQ-18 (ferray substrate) | NOT-STARTED | open prereq blocker `#1570`. `kernel_pca.rs` computes on `ndarray::{Array1, Array2}` and eigendecomposes via a hand-rolled `fn jacobi_eigen_symmetric in kernel_pca.rs`, not `ferray-core` arrays / `ferray::linalg` (R-SUBSTRATE-1/2). |

## Architecture

`kernel_pca.rs` follows the unfitted/fitted split (CLAUDE.md naming): `KernelPCA<F> {
n_components, kernel, gamma: Option<f64>, degree, coef0 }` (`pub struct KernelPCA in
kernel_pca.rs`; `new(n_components)`, builders `with_kernel`/`with_gamma`/`with_degree`/
`with_coef0`, accessors `n_components`/`kernel`/`gamma`/`degree`/`coef0`) → `Fit<Array2<F>,
()>` → `FittedKernelPCA<F> { alphas_, eigenvalues_, x_fit_, k_fit_col_means_,
k_fit_grand_mean_, kernel, gamma, degree, coef0 }` (`pub struct FittedKernelPCA in
kernel_pca.rs`, accessors `alphas`/`eigenvalues`). The path is generic over `F: Float
+ Send + Sync + 'static` (both f32 and f64, `test_kernel_pca_f32`);
`fit`/`transform` return `Result<_, FerroError>` (R-CODE-2). The `Kernel` enum
(`pub enum Kernel in kernel_pca.rs`) has `{Linear, RBF, Polynomial, Sigmoid}`.
`KernelPCA::new` defaults `kernel=Linear`, `gamma=None`, `degree=3`, `coef0=1.0`
(`kernel_pca.rs:96-105`, the `coef0=1.0` matching sklearn `_kernel_pca.py:289` post-fix).

**Fit path (`pub fn fit in kernel_pca.rs`) — REQ-1/2/3/4/5/6/7/9/10.** Validates
`n_components != 0`, `n_components <= n_samples`, `n_samples >= 2` (REQ-10). Step 1:
`effective_gamma = self.gamma.unwrap_or(1.0 / n_features)` (REQ-9, = sklearn `gamma_ =
1/X.shape[1]` `_kernel_pca.py:439`). Step 2: training gram `compute_kernel_matrix(x, x,
kernel, gamma, degree, coef0)` via `kernel_value` (the 4 kernels with the now-correct
`coef0` default, REQ-1/4). Step 3: save `K` column means (`k_fit_col_means_`) + grand
mean (`k_fit_grand_mean_`) for `transform`. Step 4: `centre_kernel_matrix`
double-centers `K` in feature space (REQ-2, = `KernelCenterer.fit_transform`
`_kernel_pca.py:331`). Step 5: `jacobi_eigen_symmetric` (exact dense eigendecomposition,
= sklearn `eigh` `_kernel_pca.py:350-352`). Step 6: sort eigenvalues DESCENDING (REQ-6,
= `argsort[::-1]` `:376-378`); for the top `n_components`, clamp negatives to 0 (REQ-5,
= `_check_psd_eigenvalues` `:368-370`), scale each eigenvector by `1/sqrt(eigenvalue)`
to form `alphas_` (REQ-7, = the deferred `eigenvectors_/sqrt(eigenvalues_)` `:399`,
`:507-509`). **Step 7 (the post-fix sign step): `svd_flip(u=eigenvectors_, v=None)`**
(`kernel_pca.rs:523-545`, REQ-3) — for each `alphas` COLUMN, find the row of maximum
`abs` value via a strict `>` scan (numpy `argmax` first-on-ties), and if that entry is
`< 0` negate the whole column, so every column's max-abs entry is POSITIVE
(`_kernel_pca.py:373`, `extmath.py:888-895`). Because the `1/sqrt(eigenvalue)` scale is
non-negative, flipping the scaled `alphas` column is identical to flipping the raw
eigenvector column.

**VALUE-PARITY (post critic→fixer→re-audit).** The fix cycle closed the two divergences
that previously blocked exact embedding values: REQ-3 (svd_flip sign, was `#1562`) and
REQ-4 (`coef0` default 0.0→1.0, was `#1563`). A re-audit cross-check on a fresh 6×4
fixture confirms `transform(X)` matches the live `KernelPCA(eigen_solver='dense')`
embedding ELEMENT-WISE INCLUDING SIGN across all four kernels (linear `5.8e-13`, rbf
`3.4e-13`, poly `2.2e-15`, sigmoid `3.3e-14`), with a multi-component independent flip
exercised within a single rbf fit (one column flipped, one kept). The only residual
value ambiguity is the degenerate repeated-eigenvalue subspace (REQ-17, `#1569`,
CARVE-OUT) where Jacobi and LAPACK pick different orthonormal bases — inherent to the
eigensolver, not a defect.

**Transform path (`pub fn transform in kernel_pca.rs`) — REQ-8.** Validate the feature
count vs `x_fit_.ncols()` (REQ-10 `ShapeMismatch`). Compute the test gram
`compute_kernel_matrix(x, x_fit_, ...)`, double-center each test row with the stored
`k_fit_col_means_`/`k_fit_grand_mean_` + that row's own mean (= sklearn
`KernelCenterer.transform` `_kernel_pca.py:502`), and project `k_centered.dot(&alphas_)`
(= sklearn `K @ scaled_alphas` `:512`). The embedding inherits the sign-fixed `alphas_`,
so it matches sklearn including sign.

**Pipeline + binding.** `KernelPCA<F>` implements `PipelineTransformer<F>` (`pub fn
fit_pipeline in kernel_pca.rs` → `Fit::fit`) and `FittedKernelPCA<F>` implements
`FittedPipelineTransformer<F>` (`pub fn transform_pipeline in kernel_pca.rs` →
`Transform::transform`) — the ferrolearn analogue of sklearn's `TransformerMixin`
(KernelPCA is `ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator`
`_kernel_pca.py:32`). The PyO3 `_RsKernelPCA` (`extras.rs:1122-1127`, via the
`py_transformer!` macro, registered `lib.rs:76`) exposes a `(n_components: usize = 2)`
ctor + `fit` + `transform` over `ferrolearn_decomp::FittedKernelPCA<f64>` — the
boundary CPython consumer. **The macro binds ONLY the `n_components` ctor + fit +
transform: NO `kernel`/`gamma`/`degree`/`coef0` ctor params (so the binding always uses
the Rust-side defaults — Linear kernel, gamma `None`, degree 3, and the now-correct
`coef0=1.0`), NO `alphas_`/`eigenvalues_` getters, NO `inverse_transform`
(REQ-12/14/15/16 binding gaps); the `transform` output is sign-correct via REQ-3.**

## Verification

Library-crate gauntlet + live sklearn oracle (sklearn 1.5.2, run from `/tmp`,
`eigen_solver='dense'`):

```bash
# SHIPPED REQs — in-module #[cfg(test)] suite:
cargo test -p ferrolearn-decomp kernel_pca
#   test_kernel_pca_linear_basic / _rbf_basic / _polynomial_basic / _sigmoid_basic   (REQ-1)
#   test_kernel_pca_transform_new_data / _linear_resembles_pca                       (REQ-2/8)
#   test_kernel_pca_eigenvalues_non_negative / _eigenvalues_sorted_descending        (REQ-5/6)
#   test_kernel_pca_single_component / _max_components_equals_samples                (REQ-7)
#   test_kernel_pca_auto_gamma                                                        (REQ-9)
#   test_kernel_pca_invalid_n_components_zero / _too_large / _insufficient_samples /
#     _shape_mismatch_transform                                                       (REQ-10)
#   test_kernel_pca_f32                                                               (REQ-11)
#   test_kernel_pca_pipeline_integration                                              (pipeline)

# SHIPPED value-parity + structural green-guards — divergence suite (3 divergence
# tests + 12 green-guards, all GREEN post-fix vs the live dense oracle, Probes 1/3):
cargo test -p ferrolearn-decomp --test divergence_kernel_pca
#   divergence_svd_flip_embedding_sign            (REQ-3: transform == sklearn incl. SIGN)
#   divergence_svd_flip_alphas_max_abs_positive   (REQ-3: every alphas column max-abs > 0)
#   divergence_coef0_default                       (REQ-4: KernelPCA::new coef0 == 1.0)
#   green_eigenvalues_match_sklearn               (REQ-5: eigenvalues == sklearn)
#   green_embedding_matches_sklearn_up_to_sign    (REQ-7: |embedding| == sklearn)
#   green_four_kernels_finite_shape               (REQ-1/7)
#   green_eigenvalues_nonneg_descending           (REQ-5/6)
#   green_transform_new_data_shape                (REQ-8)
#   green_auto_gamma                              (REQ-9)
#   green_err_n_components_zero / _too_large / _insufficient_samples /
#     _transform_feature_mismatch                 (REQ-10)
#   green_f32_path                                (REQ-11)
#   green_determinism                             (Jacobi determinism)

cargo clippy -p ferrolearn-decomp --all-targets -- -D warnings
cargo fmt --all --check
```

All 11 SHIPPED REQs are green: the structural behavior via the in-module suite, and the
exact embedding VALUE parity (REQ-3 svd_flip sign + REQ-4 coef0 default) via the live-
oracle divergence tests, now matching sklearn element-wise INCLUDING SIGN across all 4
kernels (linear `5.8e-13`, rbf `3.4e-13`, poly `2.2e-15`, sigmoid `3.3e-14`). The 7
NOT-STARTED REQs (arpack/randomized solver `#1564`, `remove_zero_eig` `#1565`, inverse
transform / ridge `#1566`, `n_components=None`/precomputed/extra kernels `#1567`,
`n_features_in_`/`X_fit_` attrs `#1568`, degenerate carve-out `#1569`, ferray substrate
`#1570`) have open prereq blockers and are not claimed.

## Blockers

| REQ | Blocker | Kind | Resolution |
|---|---|---|---|
| REQ-3 | `#1562` (FIXED) | value-parity divergence | RESOLVED: `fn fit` applies `svd_flip(u=eigenvectors_, v=None)` per-eigenvector-COLUMN max-abs-positive flip (`kernel_pca.rs:523-545`, `_kernel_pca.py:373`, `extmath.py:888-895`); embedding matches sklearn incl. sign. Pinned green by `divergence_svd_flip_embedding_sign` / `divergence_svd_flip_alphas_max_abs_positive`. |
| REQ-4 | `#1563` (FIXED) | value-parity divergence | RESOLVED: `KernelPCA::new` `coef0` default changed `0.0` → `1.0` (`kernel_pca.rs:102` = `_kernel_pca.py:289`). Pinned green by `divergence_coef0_default`. |
| REQ-12 | `#1564` | missing feature | `eigen_solver` arpack/randomized + `tol`/`max_iter`/`iterated_power`/`random_state` (`_kernel_pca.py:340-365`). |
| REQ-13 | `#1565` | missing feature | `remove_zero_eig` drop-null-space (`_kernel_pca.py:381-383`). |
| REQ-14 | `#1566` | missing feature | `alpha` ridge + `fit_inverse_transform`/`inverse_transform` + `dual_coef_`/`X_transformed_fit_` (`_kernel_pca.py:406-416`,`:514-563`). |
| REQ-15 | `#1567` | missing feature | `n_components=None` → all components (`_kernel_pca.py:334-335`) + `kernel='precomputed'` + cosine/laplacian/chi2 + `kernel_params` + callable (`_kernel_pca.py:319-326`). |
| REQ-16 | `#1568` | missing attrs | fitted `n_features_in_` / `X_fit_` / `_n_features_out` exposure (`_kernel_pca.py:450`,`:571-574`). |
| REQ-17 | `#1569` | CARVE-OUT (R-DEFER-3) | Repeated-eigenvalue subspace basis ambiguity (Jacobi ≠ LAPACK); no failing test asserted (same class as PCA / spectral_embedding degenerate carve-outs). |
| REQ-18 | `#1570` | substrate | Migrate `ndarray` + hand-rolled Jacobi to `ferray-core` / `ferray::linalg` (R-SUBSTRATE-1/2). |
