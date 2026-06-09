//! Divergence pin (#2348): `n_components='auto'` resolving to a target
//! dimension LARGER than `n_features` must raise an error.
//!
//! scikit-learn 1.5.2 `sklearn/random_projection.py:399-405`:
//! ```python
//!     elif self.n_components_ > n_features:
//!         raise ValueError(
//!             "eps=%f and n_samples=%d lead to a target dimension of "
//!             "%d which is larger than the original space with "
//!             "n_features=%d"
//!             % (self.eps, n_samples, self.n_components_, n_features)
//!         )
//! ```
//!
//! Live sklearn 1.5.2 oracle (run from /tmp so the INSTALLED build is used):
//! ```text
//! python3 -c "import numpy as np
//!   from sklearn.random_projection import GaussianRandomProjection as G
//!   G(n_components='auto', eps=0.1, random_state=0).fit(np.ones((1000, 50)))"
//! # ValueError: eps=0.100000 and n_samples=1000 lead to a target dimension
//! #             of 5920 which is larger than the original space with n_features=50
//! ```
//! Here jl(1000, eps=0.1) == 5920 > n_features == 50, so sklearn ERRORS.
//!
//! ferrolearn `resolve_n_components` (`random_projection.py` mirror,
//! `ferrolearn-preprocess/src/random_projection.rs:147-168`) only rejects a
//! resolved value of `0`; it never compares the resolved dimension against
//! `n_features`, so `fit` SUCCEEDS and builds a (5920, 50) `components_`.

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::random_projection::{
    GaussianRandomProjection, SparseRandomProjection,
};
use ndarray::Array2;

/// Divergence: ferrolearn's auto `fit` diverges from
/// `sklearn/random_projection.py:399-405` when the JL-resolved `n_components_`
/// exceeds `n_features`. sklearn raises ValueError; ferrolearn returns Ok.
/// jl(1000, eps=0.1) == 5920 (live sklearn 1.5.2) > n_features == 50.
/// Tracking: #2348
#[test]
#[ignore = "divergence: auto n_components_ > n_features must error (sklearn random_projection.py:399-405); tracking #2348"]
fn gaussian_auto_n_components_gt_features_errors() {
    // jl(1000, 0.1) == 5920 (live sklearn 1.5.2), > n_features == 50.
    let x = Array2::<f64>::ones((1000, 50));
    let proj = GaussianRandomProjection::<f64>::new_auto()
        .eps(0.1)
        .random_state(0);
    let r = proj.fit(&x, &());
    assert!(
        r.is_err(),
        "sklearn raises ValueError when auto n_components_ (5920) > n_features (50) \
         (random_projection.py:399-405); ferrolearn must also error"
    );
}

/// Same divergence for SparseRandomProjection (shares `resolve_n_components`).
/// Tracking: #2348
#[test]
#[ignore = "divergence: auto n_components_ > n_features must error (sklearn random_projection.py:399-405); tracking #2348"]
fn sparse_auto_n_components_gt_features_errors() {
    let x = Array2::<f64>::ones((1000, 50));
    let proj = SparseRandomProjection::<f64>::new_auto()
        .eps(0.1)
        .random_state(0);
    let r = proj.fit(&x, &());
    assert!(
        r.is_err(),
        "sklearn raises ValueError when auto n_components_ (5920) > n_features (50) \
         (random_projection.py:399-405); ferrolearn must also error"
    );
}
