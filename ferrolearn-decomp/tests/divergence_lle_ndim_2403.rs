//! Divergence pin: `ferrolearn_decomp::LLE::fit` accepts `n_components`
//! greater than the number of input features (`d_in`), whereas scikit-learn
//! 1.5.2 `LocallyLinearEmbedding` raises a `ValueError` in that case.
//!
//! sklearn `sklearn/manifold/_locally_linear.py:222-225`
//! (`_locally_linear_embedding`):
//! ```python
//!     N, d_in = X.shape
//!     if n_components > d_in:
//!         raise ValueError(
//!             "output dimension must be less than or equal to input dimension"
//!         )
//! ```
//! This `n_components > d_in` guard fires BEFORE the `n_neighbors >= N` guard
//! (`:226-230`) and before any of the kNN / weight / eigen math.
//!
//! ferrolearn `ferrolearn-decomp/src/lle.rs` validates `n_components` only
//! against the SAMPLE count (`if self.n_components >= n { ... }`,
//! `lle.rs:359-367`, where `n = x.nrows()`). It never compares `n_components`
//! to `x.ncols()` (the feature count `d_in`). So for a fixture with more
//! samples than features, ferrolearn silently produces a `(n_samples,
//! n_components)` embedding instead of raising.
//!
//! LIVE sklearn 1.5.2 oracle (R-CHAR-3), X = 6 samples x 2 features,
//! n_components=3, n_neighbors=4, method='standard', eigen_solver='dense':
//!   >>> LocallyLinearEmbedding(n_components=3, n_neighbors=4,
//!   ...     method='standard', eigen_solver='dense').fit_transform(X)
//!   ValueError: output dimension must be less than or equal to input dimension
//! The boundary `n_components == d_in == 2` is ACCEPTED by sklearn (shape (6,2)).
//!
//! Expected (sklearn): `fit` returns `Err` for n_components=3 (> d_in=2).
//! Actual   (ferrolearn): `fit` returns `Ok` with embedding shape (6, 3).
//!
//! Tracking: #2403.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::LLE;
use ndarray::array;

/// 6 samples x 2 features. `n_components = 3 > d_in = 2` but
/// `n_components = 3 < n_samples = 6`, so ferrolearn's sample-count guard does
/// not fire while sklearn's `n_components > d_in` guard does.
#[test]
#[ignore = "divergence: LLE::fit accepts n_components > n_features (sklearn raises ValueError); tracking #2403"]
fn divergence_lle_n_components_gt_n_features() {
    let x = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ];
    // sklearn raises ValueError("output dimension must be less than or equal
    // to input dimension") (_locally_linear.py:222-225). ferrolearn must also
    // reject n_components (3) > d_in (2).
    let result = LLE::new(3).with_n_neighbors(4).with_reg(1e-3).fit(&x, &());
    assert!(
        result.is_err(),
        "sklearn raises ValueError for n_components (3) > n_features (2) \
         (_locally_linear.py:222-225); ferrolearn returned Ok"
    );
}
