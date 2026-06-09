//! Divergence pin: `components_` orientation for `GaussianRandomProjection` /
//! `SparseRandomProjection` vs scikit-learn 1.5.2 `sklearn/random_projection.py`.
//!
//! sklearn stores the projection matrix as `components_` of shape
//! `(n_components, n_features)` and computes the transform as
//! `X @ components_.T` (`random_projection.py:418-421` generates
//! `_make_random_matrix(self.n_components_, n_features)` → shape
//! `(n_components, n_features)`; `BaseRandomProjection.transform` does
//! `X @ self.components_.T`). ferrolearn instead stores `projection` of shape
//! `(n_features, n_components)` and computes `X @ projection`. The transform
//! OUTPUT shape `(n_samples, n_components)` happens to coincide, but the
//! fitted-attribute orientation that a user reads via the public
//! `projection()` getter is TRANSPOSED relative to sklearn's `components_`.
//!
//! This is a DETERMINISTIC, structural divergence (R-DEV-1 fitted-attribute
//! orientation): for a NON-square `(n_features != n_components)` projection the
//! shapes differ, independent of the RNG that fills the values.
//!
//! Oracle (live sklearn 1.5.2):
//! ```text
//! python3 -c "import numpy as np; from sklearn.random_projection import GaussianRandomProjection; \
//!   g=GaussianRandomProjection(n_components=5, random_state=0).fit(np.ones((10,20))); \
//!   print(g.components_.shape)"   # (5, 20) == (n_components, n_features)
//! ```
//! sklearn `components_.shape == (n_components=5, n_features=20)`.
//! ferrolearn `projection().shape == (n_features=20, n_components=5)`.
//!
//! Tracking: #<filled at file time>

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::random_projection::{GaussianRandomProjection, SparseRandomProjection};
use ndarray::Array2;

/// sklearn `random_projection.py:191,418-421`: the random matrix (`components_`)
/// has shape `(n_components, n_features)`. With `n_components=5`, `n_features=20`
/// the oracle shape is `[5, 20]` (verified via the live sklearn oracle above).
const SK_N_COMPONENTS: usize = 5;
const SK_N_FEATURES: usize = 20;

#[test]
#[ignore = "divergence: components_ orientation (n_features,n_components) vs sklearn (n_components,n_features); tracking #2346"]
fn divergence_gaussian_components_orientation() {
    let x = Array2::<f64>::ones((10, SK_N_FEATURES));
    let proj = GaussianRandomProjection::<f64>::new(SK_N_COMPONENTS).random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();

    // sklearn components_.shape == (n_components, n_features) == (5, 20).
    // ferrolearn projection().shape == (n_features, n_components) == (20, 5).
    assert_eq!(
        fitted.projection().shape(),
        &[SK_N_COMPONENTS, SK_N_FEATURES],
        "ferrolearn projection() is transposed vs sklearn components_ \
         (random_projection.py:191,418-421)"
    );
}

#[test]
#[ignore = "divergence: components_ orientation (n_features,n_components) vs sklearn (n_components,n_features); tracking #2346"]
fn divergence_sparse_components_orientation() {
    let x = Array2::<f64>::ones((30, SK_N_FEATURES));
    let proj = SparseRandomProjection::<f64>::new(SK_N_COMPONENTS).random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.projection().shape(),
        &[SK_N_COMPONENTS, SK_N_FEATURES],
        "ferrolearn projection() is transposed vs sklearn components_ \
         (random_projection.py:191,418-421)"
    );
}
