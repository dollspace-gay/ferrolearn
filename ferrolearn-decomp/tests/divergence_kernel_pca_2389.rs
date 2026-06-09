//! Divergence pins for `KernelPCA` (ferrolearn-decomp) vs scikit-learn 1.5.2.
//!
//! Audited area: `ferrolearn-decomp/src/kernel_pca.rs` against
//! `sklearn/decomposition/_kernel_pca.py` (tag 1.5.2). The numerical core
//! (eigenvalues_, double-centering, eigenvector `1/sqrt(eigval)` scaling,
//! transform values, and the `svd_flip` sign convention) was found to match
//! the live sklearn 1.5.2 dense oracle to ~1e-12 for distinct-eigenvalue
//! inputs and is NOT pinned here (CLEAN).
//!
//! The two divergences pinned below are control-flow / contract divergences
//! where ferrolearn produces a DIFFERENT observable outcome than sklearn:
//!
//!  - #2389: `n_components > n_samples`. sklearn clamps `n_components =
//!    min(K.shape[0], n_components)` (`_kernel_pca.py:337`) and FITS; ferrolearn
//!    returns `Err(InvalidParameter)`.
//!  - #2390: a non-PSD centered kernel (e.g. `sigmoid`) with significant
//!    negative eigenvalues. sklearn RAISES `ValueError` via
//!    `_check_psd_eigenvalues` (`utils/validation.py:1958`, called at
//!    `_kernel_pca.py:368`); ferrolearn silently clamps the negatives to 0 and
//!    returns `Ok` with a garbage embedding.
//!
//! Expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3); the
//! commands are quoted in each test. NEVER literal-copied from ferrolearn.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{Kernel, KernelPCA};
use ndarray::{array, Array2};

/// Deterministic distinct-eigenvalue fixture (6 samples, 2 features).
fn fixture() -> Array2<f64> {
    array![
        [0.1, 0.3],
        [0.9, 0.2],
        [0.4, 0.8],
        [0.7, 0.5],
        [0.2, 0.6],
        [0.55, 0.15],
    ]
}

/// Divergence: `KernelPCA::fit` with `n_components > n_samples`.
///
/// sklearn `_kernel_pca.py:337`:
/// ```python
/// n_components = min(K.shape[0], self.n_components)
/// ```
/// so `KernelPCA(n_components=10, kernel='linear').fit(X)` on a 6-sample X
/// FITS and yields `eigenvalues_.shape == (6,)` and `transform(X).shape ==
/// (6, 6)` (the extra components are zero eigenvalues). ferrolearn instead
/// returns `Err(InvalidParameter{ name: "n_components", ... })`
/// (`kernel_pca.rs:508-516`).
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.decomposition import KernelPCA
/// X=np.array([[0.1,0.3],[0.9,0.2],[0.4,0.8],[0.7,0.5],[0.2,0.6],[0.55,0.15]])
/// kp=KernelPCA(n_components=10,kernel='linear',eigen_solver='dense').fit(X)
/// print(kp.eigenvalues_.shape, kp.transform(X).shape)"
/// # -> (6,) (6, 6)
/// ```
///
/// Tracking: #2389
#[test]
#[ignore = "divergence: KernelPCA rejects n_components>n_samples; sklearn clamps via min(); tracking #2389"]
fn divergence_n_components_exceeds_samples_clamps() {
    let x = fixture(); // 6 samples
    let fitted = KernelPCA::<f64>::new(10)
        .with_kernel(Kernel::Linear)
        .fit(&x, &())
        .expect("sklearn clamps n_components to n_samples and fits; ferrolearn must not Err");

    // sklearn returns 6 eigenvalues (== n_samples), not 10.
    assert_eq!(
        fitted.eigenvalues().len(),
        6,
        "sklearn clamps to n_samples=6 eigenvalues (_kernel_pca.py:337)"
    );
    let projected = fitted.transform(&x).expect("transform");
    assert_eq!(
        projected.dim(),
        (6, 6),
        "sklearn transform shape is (n_samples, min(n_samples, n_components)) = (6, 6)"
    );
}

/// Divergence: a non-PSD centered kernel with significant negative eigenvalues.
///
/// sklearn `_kernel_pca.py:368` calls
/// `_check_psd_eigenvalues(self.eigenvalues_, enable_warnings=False)`
/// which at `sklearn/utils/validation.py:1958` RAISES:
/// `ValueError("There are significant negative eigenvalues (... of the maximum
/// positive). Either the matrix is not PSD ...")`.
///
/// The `sigmoid` kernel with `gamma=2.0, coef0=1.0` on the fixture produces a
/// centered gram with a negative eigenvalue ~0.092 of the max positive, so
/// sklearn ABORTS the fit. ferrolearn instead clamps negatives to 0
/// (`kernel_pca.rs:566-570`) and returns `Ok` with a garbage embedding.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// python3 -c "
/// import numpy as np; from sklearn.decomposition import KernelPCA
/// X=np.array([[0.1,0.3],[0.9,0.2],[0.4,0.8],[0.7,0.5],[0.2,0.6],[0.55,0.15]])
/// KernelPCA(n_components=5,kernel='sigmoid',gamma=2.0,coef0=1.0,eigen_solver='dense').fit(X)"
/// # -> ValueError: There are significant negative eigenvalues (0.0924979 of
/// #    the maximum positive). Either the matrix is not PSD ...
/// ```
///
/// Tracking: #2390
#[test]
#[ignore = "divergence: KernelPCA clamps non-PSD negative eigenvalues; sklearn raises ValueError; tracking #2390"]
fn divergence_non_psd_kernel_raises() {
    let x = fixture();
    let result = KernelPCA::<f64>::new(5)
        .with_kernel(Kernel::Sigmoid)
        .with_gamma(2.0)
        .with_coef0(1.0)
        .fit(&x, &());

    // sklearn raises ValueError via _check_psd_eigenvalues; ferrolearn must
    // surface an error rather than silently returning a clamped embedding.
    assert!(
        matches!(result, Err(FerroError::InvalidParameter { .. })),
        "sklearn raises ValueError for a non-PSD centered kernel \
         (_check_psd_eigenvalues, validation.py:1958); ferrolearn returned {:?}",
        result.map(|f| f.eigenvalues().to_vec())
    );
}
