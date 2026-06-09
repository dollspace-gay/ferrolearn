//! Oracle tests for the deterministic Johnson-Lindenstrauss surface of
//! `GaussianRandomProjection` / `SparseRandomProjection` vs scikit-learn 1.5.2
//! `sklearn/random_projection.py`.
//!
//! Covers (#2347):
//!   - the `johnson_lindenstrauss_min_dim` free fn
//!     (`random_projection.py:64-143`): value = `floor(4*ln(n_samples) /
//!     (eps^2/2 - eps^3/3))` (the `.astype(np.int64)` truncation `:142-143`),
//!     `eps∈(0,1)` reject (`:133`), `n_samples>0` reject (`:136`);
//!   - `n_components='auto'` resolution at fit (`:388-391`) setting
//!     `n_components_ = johnson_lindenstrauss_min_dim(n_samples, eps)`, with the
//!     resolved-`<=0` reject (`:393-397`);
//!   - the fitted attrs `n_components_`, `n_features_in_`, `density_` (`:788`).
//!
//! All expected values are LIVE sklearn 1.5.2 output (R-CHAR-3), e.g.:
//! ```text
//! python3 -c "from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl
//!   print(jl(1000, eps=0.1), jl(1e6, eps=0.5), jl(50, eps=0.2), jl(100, eps=0.9))"
//!   # 5920 663 902 113
//! python3 -c "from sklearn.random_projection import GaussianRandomProjection as G
//!   import numpy as np
//!   g=G(n_components='auto', eps=0.5, random_state=0).fit(np.ones((20,5000)))
//!   print(g.n_components_, g.components_.shape, g.n_features_in_)"  # 143 (143,5000) 5000
//! python3 -c "from sklearn.random_projection import SparseRandomProjection as S
//!   import numpy as np
//!   print(S(n_components=10, random_state=0).fit(np.ones((30,100))).density_)"  # 0.1
//! ```
//!
//! The RNG-generated matrix VALUES are a documented non-parity carve-out (Rust
//! `SmallRng` != numpy MT19937, #1388) and are NOT asserted here.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::random_projection::{
    GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim,
};
use ndarray::Array2;

// sklearn `johnson_lindenstrauss_min_dim` oracle (live sklearn 1.5.2):
//   jl(1000, 0.1) = 5920
//   jl(1e6,  0.5) = 663
//   jl(50,   0.2) = 902
//   jl(100,  0.9) = 113
// (random_projection.py:142-143 — `.astype(np.int64)` truncation toward zero.)

#[test]
fn jl_min_dim_oracle_1000_eps01() {
    let v = johnson_lindenstrauss_min_dim(1000_usize, 0.1_f64).unwrap();
    assert_eq!(
        v, 5920,
        "jl(1000, 0.1) must match sklearn (random_projection.py:142-143)"
    );
}

#[test]
fn jl_min_dim_oracle_1e6_eps05() {
    let v = johnson_lindenstrauss_min_dim(1_000_000_usize, 0.5_f64).unwrap();
    assert_eq!(v, 663, "jl(1e6, 0.5) must match sklearn");
}

#[test]
fn jl_min_dim_oracle_50_eps02() {
    let v = johnson_lindenstrauss_min_dim(50_usize, 0.2_f64).unwrap();
    assert_eq!(v, 902, "jl(50, 0.2) must match sklearn");
}

#[test]
fn jl_min_dim_oracle_100_eps09() {
    let v = johnson_lindenstrauss_min_dim(100_usize, 0.9_f64).unwrap();
    assert_eq!(v, 113, "jl(100, 0.9) must match sklearn");
}

#[test]
fn jl_min_dim_floor_truncates_not_ceil() {
    // sklearn truncates toward zero (.astype(np.int64)); the exact real value of
    // jl(1000, 0.1) = 4*ln(1000)/(0.005 - 0.000333..) ≈ 5920.9..., so the
    // truncated result is 5920, NOT a ceil of 5921.
    let v = johnson_lindenstrauss_min_dim(1000_usize, 0.1_f64).unwrap();
    assert_eq!(v, 5920);
    assert_ne!(
        v, 5921,
        "must FLOOR/truncate, not ceil (random_projection.py:143)"
    );
}

// eps validation: sklearn rejects eps outside ]0, 1[ (random_projection.py:133).

#[test]
fn jl_min_dim_eps_zero_errors() {
    let r = johnson_lindenstrauss_min_dim(1000_usize, 0.0_f64);
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

#[test]
fn jl_min_dim_eps_one_errors() {
    let r = johnson_lindenstrauss_min_dim(1000_usize, 1.0_f64);
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

#[test]
fn jl_min_dim_eps_above_one_errors() {
    let r = johnson_lindenstrauss_min_dim(1000_usize, 1.5_f64);
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

#[test]
fn jl_min_dim_eps_negative_errors() {
    let r = johnson_lindenstrauss_min_dim(1000_usize, -0.1_f64);
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

// n_samples validation: sklearn rejects n_samples <= 0 (random_projection.py:136).

#[test]
fn jl_min_dim_n_samples_zero_errors() {
    let r = johnson_lindenstrauss_min_dim(0_usize, 0.1_f64);
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

// n_components='auto' resolution at fit (random_projection.py:388-391):
//   n_components_ = johnson_lindenstrauss_min_dim(n_samples, eps).
// Oracle: G(n_components='auto', eps=0.5).fit(ones((20, 5000)))
//   -> n_components_ == 143 (== jl(20, 0.5)), components_.shape == (143, 5000),
//      n_features_in_ == 5000.

#[test]
fn gaussian_auto_resolves_n_components_via_jl() {
    let n_samples = 20_usize;
    let n_features = 5000_usize;
    let eps = 0.5_f64;
    let x = Array2::<f64>::ones((n_samples, n_features));
    let proj = GaussianRandomProjection::<f64>::new_auto()
        .eps(eps)
        .random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();

    let expected = johnson_lindenstrauss_min_dim(n_samples, eps).unwrap();
    assert_eq!(expected, 143, "sklearn jl(20, 0.5) == 143");
    assert_eq!(fitted.n_components_(), expected);
    assert_eq!(fitted.n_components_(), 143);
    // components_ orientation (n_components, n_features) — random_projection.py:419.
    assert_eq!(fitted.components().shape(), &[143, n_features]);
    assert_eq!(fitted.n_features_in_(), n_features);
}

#[test]
fn sparse_auto_resolves_n_components_via_jl() {
    let n_samples = 20_usize;
    let n_features = 5000_usize;
    let eps = 0.5_f64;
    let x = Array2::<f64>::ones((n_samples, n_features));
    let proj = SparseRandomProjection::<f64>::new_auto()
        .eps(eps)
        .random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();

    let expected = johnson_lindenstrauss_min_dim(n_samples, eps).unwrap();
    assert_eq!(fitted.n_components_(), expected);
    assert_eq!(fitted.n_components_(), 143);
    assert_eq!(fitted.components().shape(), &[143, n_features]);
    assert_eq!(fitted.n_features_in_(), n_features);
}

#[test]
fn gaussian_default_eps_is_point_one() {
    // sklearn default eps=0.1 (random_projection.py:328). With n_samples=1000,
    // n_components_ should equal jl(1000, 0.1) = 5920.
    let n_features = 6000_usize; // >= 5920 so the auto bound is valid.
    let x = Array2::<f64>::ones((1000, n_features));
    let proj = GaussianRandomProjection::<f64>::new_auto().random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();
    assert_eq!(fitted.n_components_(), 5920);
}

// Resolved n_components_ <= 0 reject (random_projection.py:393-397).
// jl can never be 0 for valid eps/n_samples >= 1 (4*ln(n)/denominator with
// n>=2 is positive and >= 1 for the small denominators here), but n_samples=1
// gives ln(1)=0 -> jl == 0 -> the auto path must reject.

#[test]
fn gaussian_auto_n_components_zero_errors() {
    // n_samples=1 -> ln(1)=0 -> jl == 0 -> invalid target dimension.
    let x = Array2::<f64>::ones((1, 100));
    let proj = GaussianRandomProjection::<f64>::new_auto().random_state(0);
    let r = proj.fit(&x, &());
    assert!(
        matches!(r, Err(FerroError::InvalidParameter { .. })),
        "auto n_components_ == 0 must error (random_projection.py:393-397)"
    );
}

// density_ attribute (random_projection.py:788): 'auto' -> 1/sqrt(n_features),
// explicit -> the given value.

#[test]
fn sparse_density_auto_oracle() {
    // Oracle: S(n_components=10).fit(ones((30,100))).density_ == 0.1 == 1/sqrt(100).
    let x = Array2::<f64>::ones((30, 100));
    let proj = SparseRandomProjection::<f64>::new(10).random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();
    assert!(
        (fitted.density_() - 0.1).abs() < 1e-12,
        "auto density_ == 1/sqrt(100) == 0.1"
    );
}

#[test]
fn sparse_density_explicit_oracle() {
    // Oracle: S(n_components=10, density=0.3).fit(...).density_ == 0.3.
    let x = Array2::<f64>::ones((30, 100));
    let proj = SparseRandomProjection::<f64>::new(10)
        .density(0.3)
        .random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();
    assert!((fitted.density_() - 0.3).abs() < 1e-12);
}

#[test]
fn sparse_density_auto_n_features_400() {
    // 1/sqrt(400) == 0.05 (sklearn _check_density:148-149).
    let x = Array2::<f64>::ones((30, 400));
    let proj = SparseRandomProjection::<f64>::new(10).random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();
    assert!((fitted.density_() - 0.05).abs() < 1e-12);
}

// Confirm the fixed-n_components path is unchanged (transform output shape and
// the n_features_in_ / n_components_ attrs).

#[test]
fn fixed_n_components_attrs_and_transform_unchanged() {
    use ferrolearn_core::traits::Transform;
    let x = Array2::<f64>::ones((10, 50));
    let proj = GaussianRandomProjection::<f64>::new(5).random_state(0);
    let fitted = proj.fit(&x, &()).unwrap();
    assert_eq!(fitted.n_components_(), 5);
    assert_eq!(fitted.n_features_in_(), 50);
    // components_ is (n_components, n_features) = (5, 50).
    assert_eq!(fitted.components().shape(), &[5, 50]);
    // transform output unchanged: (n_samples, n_components) = (10, 5).
    let out = fitted.transform(&x).unwrap();
    assert_eq!(out.shape(), &[10, 5]);
}
