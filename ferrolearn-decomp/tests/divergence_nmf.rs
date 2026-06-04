//! Divergence + green-guard suite for `ferrolearn-decomp::NMF` vs scikit-learn
//! 1.5.2 `class NMF(_BaseNMF)` (`sklearn/decomposition/_nmf.py:912`).
//!
//! Tracking issue: #1608. Design doc: `.design/decomp/nmf.md`.
//!
//! ## What this file pins
//!
//! These are STRUCTURAL green-guards for the 7 SHIPPED REQs (they MUST PASS
//! against the current `nmf.rs`). Every numeric bound is sourced from the live
//! sklearn 1.5.2 oracle (run from `/tmp`), never literal-copied from the
//! ferrolearn side (R-CHAR-3):
//!
//! - REQ-1: `components_` (H) shape `(n_components, n_features)`; transform W
//!   shape `(n_samples, n_components)`; finite `reconstruction_err_ >= 0` that
//!   DECREASES with more iterations / more components; positive `n_iter_`;
//!   determinism given a seed.
//! - REQ-2: W and H element-wise NON-NEGATIVE.
//! - REQ-3: both solvers (MU/CD) x both inits (Random/Nndsvd) fit + transform
//!   produce finite non-negative output.
//! - REQ-4: reconstruction QUALITY (residual small / decreasing).
//! - REQ-10: `inverse_transform` == `W @ components_` (exact algebra).
//! - REQ-11: error / parameter contracts.
//! - REQ-1 (f32): the f32 path fits.
//!
//! ## What this file does NOT pin (carve-outs, R-DEFER-3)
//!
//! Exact `components_`/W VALUES (REQ-5, carve-out #1609) are NOT pinned.
//! ferrolearn defaults to MU + Random init (Rust `StdRng` uniform) / pseudo-
//! NNDSVD (Jacobi of `X'X`, `nmf.rs:362`); sklearn defaults to `solver='cd'` +
//! `init='nndsvda'` (deterministic SVD). NMF is identifiable only up to
//! permutation/scaling, and numpy RNG != Rust RNG.
//!
//! The transform NNLS-W value (REQ-9, #1613) sits DOWNSTREAM of the carved-out
//! fitted H, and `FittedNMF`'s fields are PRIVATE (no `from_components` /
//! injectable-H constructor — see `nmf.rs:193`), so transform value parity
//! folds into the REQ-5 carve-out (same class as minibatch_nmf #1487 /
//! dictionary_learning).

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{NMF, NMFInit, NMFSolver};
use ndarray::{Array2, array};

/// Small fixed non-negative dataset (4x3). Same shape probed against the oracle.
fn small_x() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
}

/// Medium fixed non-negative dataset (6x4).
fn medium_x() -> Array2<f64> {
    array![
        [5.0, 3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 5.0, 4.0],
        [0.0, 0.0, 4.0, 3.0],
    ]
}

// ---------------------------------------------------------------------------
// REQ-1 / REQ-2: shapes + non-negativity (green-guards)
//
// Oracle (sklearn 1.5.2, /tmp): NMF(n_components=2, init='random',
// random_state=0).fit(X4x3) -> components_.shape == (2, 3); transform(X).shape
// == (4, 2); both element-wise >= 0. Shapes/non-negativity are STRUCTURAL
// (not the carved-out values).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-1): `components_` shape == `(n_components, n_features)`,
/// matching sklearn `_nmf.py:1118` (`self.components_ = H`).
#[test]
fn green_components_shape() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed on 4x3 non-negative X");
    // sklearn oracle: components_.shape == (2, 3).
    assert_eq!(fitted.components().dim(), (2, 3));
}

/// Green-guard (REQ-1): transform W shape == `(n_samples, n_components)`,
/// matching sklearn `transform` (`_nmf.py:1213`).
#[test]
fn green_transform_shape() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    let w = fitted
        .transform(&small_x())
        .expect("transform should succeed");
    // sklearn oracle: transform(X).shape == (4, 2).
    assert_eq!(w.dim(), (4, 2));
}

/// Green-guard (REQ-2): all `components_` (H) entries are non-negative.
/// sklearn oracle: `(components_ >= 0).all() == True`.
#[test]
fn green_components_non_negative() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    for &v in fitted.components() {
        assert!(v >= 0.0, "H entry must be non-negative, got {v}");
    }
}

/// Green-guard (REQ-2): all transform W entries are non-negative.
/// sklearn oracle: `(W >= 0).all() == True`.
#[test]
fn green_transform_w_non_negative() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    let w = fitted
        .transform(&small_x())
        .expect("transform should succeed");
    for &v in &w {
        assert!(v >= 0.0, "W entry must be non-negative, got {v}");
    }
}

// ---------------------------------------------------------------------------
// REQ-1 / REQ-4: reconstruction error finite >= 0, DECREASES with more
// iterations and more components ("did NMF work" signal). n_iter_ > 0.
//
// Oracle (sklearn 1.5.2 MU on medium 6x4, init='random', random_state=0,
// max_iter=1000): ||X-WH|| at k=1 -> 8.111155, k=2 -> 4.615357,
// k=3 -> 1.964775. Monotone decrease in k is the structural signal; the small
// converged residual (k=3 ~ 1.96) confirms a real factorization.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-1/REQ-4): `reconstruction_err_` is finite and `>= 0`.
#[test]
fn green_reconstruction_err_finite_nonneg() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    let err = fitted.reconstruction_err();
    assert!(
        err.is_finite(),
        "reconstruction_err must be finite, got {err}"
    );
    assert!(err >= 0.0, "reconstruction_err must be >= 0, got {err}");
}

/// Green-guard (REQ-1): `n_iter_ > 0` after a real fit.
#[test]
fn green_n_iter_positive() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    assert!(
        fitted.n_iter() > 0,
        "n_iter_ must be > 0, got {}",
        fitted.n_iter()
    );
}

/// Green-guard (REQ-4): more iterations do not increase the reconstruction
/// error. Mirrors NMF's monotone objective descent (`_nmf.py` MU/CD loop).
#[test]
fn green_more_iters_lower_or_equal_error() {
    let x = small_x();
    let few = NMF::<f64>::new(2)
        .with_random_state(0)
        .with_max_iter(10)
        .fit(&x, &())
        .expect("fit should succeed");
    let many = NMF::<f64>::new(2)
        .with_random_state(0)
        .with_max_iter(300)
        .fit(&x, &())
        .expect("fit should succeed");
    assert!(
        many.reconstruction_err() <= few.reconstruction_err() + 1e-6,
        "more iterations must not increase error: few={}, many={}",
        few.reconstruction_err(),
        many.reconstruction_err()
    );
}

/// Green-guard (REQ-4): more components do not increase the reconstruction
/// error. sklearn oracle (MU, medium 6x4): ||X-WH|| strictly decreases from
/// 8.11 (k=1) -> 4.62 (k=2) -> 1.96 (k=3).
#[test]
fn green_more_components_lower_or_equal_error() {
    let x = medium_x();
    let k1 = NMF::<f64>::new(1)
        .with_random_state(0)
        .with_max_iter(500)
        .fit(&x, &())
        .expect("fit should succeed");
    let k2 = NMF::<f64>::new(2)
        .with_random_state(0)
        .with_max_iter(500)
        .fit(&x, &())
        .expect("fit should succeed");
    assert!(
        k2.reconstruction_err() <= k1.reconstruction_err() + 1e-6,
        "more components must not increase error: k1={}, k2={}",
        k1.reconstruction_err(),
        k2.reconstruction_err()
    );
}

/// Green-guard (REQ-4): the converged residual is genuinely SMALL (the "did
/// NMF actually factor X" signal). sklearn oracle (MU, medium 6x4, k=3,
/// max_iter=1000) reaches ||X-WH|| ~ 1.96; we assert ferrolearn's residual is
/// comfortably below the trivial all-zero residual ||X||_F. This bounds the
/// quality WITHOUT pinning the carved-out component values (REQ-5).
#[test]
fn green_reconstruction_quality_small() {
    let x = medium_x();
    // ||X||_F (residual of the trivial WH=0 factorization).
    let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let fitted = NMF::<f64>::new(3)
        .with_random_state(0)
        .with_max_iter(1000)
        .fit(&x, &())
        .expect("fit should succeed");
    let err = fitted.reconstruction_err();
    // sklearn oracle k=3 residual is ~1.96, far below ||X||_F (~11.5). A real
    // factorization must beat half of ||X||_F.
    assert!(
        err < 0.5 * norm_x,
        "reconstruction err {err} should be well below 0.5*||X||_F = {}",
        0.5 * norm_x
    );
}

// ---------------------------------------------------------------------------
// REQ-3: all four (solver x init) combinations fit + transform produce finite
// non-negative output of the right shape. STRUCTURAL only (not values).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-3): every (MU|CD) x (Random|Nndsvd) combination fits and
/// transforms to finite, non-negative output of the contract shapes.
#[test]
fn green_all_solver_init_combos_finite_nonneg() {
    let x = medium_x();
    let (n_samples, n_features) = x.dim();
    let k = 2;
    for solver in [
        NMFSolver::MultiplicativeUpdate,
        NMFSolver::CoordinateDescent,
    ] {
        for init in [NMFInit::Random, NMFInit::Nndsvd] {
            let fitted = NMF::<f64>::new(k)
                .with_solver(solver)
                .with_init(init)
                .with_random_state(0)
                .with_max_iter(300)
                .fit(&x, &())
                .unwrap_or_else(|e| {
                    panic!("fit must succeed for solver={solver:?} init={init:?}: {e:?}")
                });
            assert_eq!(
                fitted.components().dim(),
                (k, n_features),
                "components shape for solver={solver:?} init={init:?}"
            );
            for &v in fitted.components() {
                assert!(
                    v.is_finite() && v >= 0.0,
                    "H entry must be finite non-negative (solver={solver:?} init={init:?}), got {v}"
                );
            }
            let w = fitted.transform(&x).unwrap_or_else(|e| {
                panic!("transform must succeed for solver={solver:?} init={init:?}: {e:?}")
            });
            assert_eq!(
                w.dim(),
                (n_samples, k),
                "W shape for solver={solver:?} init={init:?}"
            );
            for &v in &w {
                assert!(
                    v.is_finite() && v >= 0.0,
                    "W entry must be finite non-negative (solver={solver:?} init={init:?}), got {v}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-10: inverse_transform == W @ components_ (exact algebra).
//
// Oracle (sklearn 1.5.2): NMF.inverse_transform(W) returns W @ self.components_
// (`_nmf.py:1238`); np.allclose(m.inverse_transform(W), W @ m.components_) ==
// True. Deterministic, not RNG/solver-gated -> NOT a value carve-out.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-10): `inverse_transform(W)` equals `W @ components_`
/// exactly, mirroring sklearn `_nmf.py:1238`. We compute the reference `W @ H`
/// directly from the fitted components (the algebra sklearn performs), so this
/// is NOT tautological: it asserts ferrolearn's `inverse_transform` IS that
/// matrix product, not a literal-copied constant.
#[test]
fn green_inverse_transform_equals_w_dot_h() {
    let x = small_x();
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed");
    let w = fitted.transform(&x).expect("transform should succeed");
    let inv = fitted
        .inverse_transform(&w)
        .expect("inverse_transform should succeed");
    // sklearn semantics: inverse_transform(W) == W @ components_.
    let reference = w.dot(fitted.components());
    assert_eq!(
        inv.dim(),
        (4, 3),
        "inverse_transform shape (n_samples, n_features)"
    );
    for (a, b) in inv.iter().zip(reference.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
}

/// Green-guard (REQ-10): `inverse_transform` rejects a column-count mismatch
/// (sklearn `check_array` raises `ValueError`; ferrolearn returns
/// `ShapeMismatch`).
#[test]
fn green_inverse_transform_shape_mismatch_errs() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    // W with wrong column count (3 != n_components=2).
    let bad_w = array![[1.0, 2.0, 3.0]];
    assert!(
        fitted.inverse_transform(&bad_w).is_err(),
        "inverse_transform must reject W with wrong n_components"
    );
}

// ---------------------------------------------------------------------------
// REQ-11: error / parameter contracts. sklearn raises
// InvalidParameterError / ValueError; ferrolearn returns Err(FerroError).
// We assert the Err-vs-Ok contract (not the concrete sklearn exception type).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-11): `n_components == 0` is rejected.
#[test]
fn green_err_n_components_zero() {
    assert!(NMF::<f64>::new(0).fit(&small_x(), &()).is_err());
}

/// Green-guard (REQ-11): `n_components > min(n_samples, n_features)` is rejected
/// by ferrolearn. (FLAG: sklearn does NOT pre-reject this — `nndsvda` downgrades
/// to `random`; this is a documented contract DIVERGENCE, but ferrolearn's
/// stricter behavior is the SHIPPED scoped contract per REQ-11, so this is a
/// green-guard of the ferrolearn contract, not an sklearn parity pin.)
#[test]
fn green_err_n_components_too_large() {
    // small_x is 4x3 -> min = 3; 10 > 3.
    assert!(NMF::<f64>::new(10).fit(&small_x(), &()).is_err());
}

/// Green-guard (REQ-11): negative input to `fit` is rejected (sklearn
/// `check_non_negative` raises `ValueError`).
#[test]
fn green_err_negative_input() {
    let x = array![[1.0, -2.0], [3.0, 4.0]];
    assert!(NMF::<f64>::new(1).fit(&x, &()).is_err());
}

/// Green-guard (REQ-11): zero-sample input is rejected.
#[test]
fn green_err_zero_samples() {
    let x = Array2::<f64>::zeros((0, 3));
    assert!(NMF::<f64>::new(1).fit(&x, &()).is_err());
}

/// Green-guard (REQ-11): `transform` rejects negative input.
#[test]
fn green_err_transform_negative() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    let x_neg = array![[1.0, -2.0, 3.0]];
    assert!(fitted.transform(&x_neg).is_err());
}

/// Green-guard (REQ-11): `transform` rejects a feature-count mismatch
/// (sklearn raises `ValueError`; ferrolearn returns `ShapeMismatch`).
#[test]
fn green_err_transform_feature_mismatch() {
    let fitted = NMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&small_x(), &())
        .expect("fit should succeed");
    let x_bad = array![[1.0, 2.0]]; // 2 features != 3 seen at fit.
    assert!(fitted.transform(&x_bad).is_err());
}

// ---------------------------------------------------------------------------
// REQ-1: determinism — same random_state => identical components + transform.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-1): two fits with the same `random_state` produce
/// element-wise identical `components_` and identical transform W.
#[test]
fn green_determinism_same_seed() {
    let x = small_x();
    let a = NMF::<f64>::new(2)
        .with_random_state(7)
        .fit(&x, &())
        .expect("fit should succeed");
    let b = NMF::<f64>::new(2)
        .with_random_state(7)
        .fit(&x, &())
        .expect("fit should succeed");
    for (p, q) in a.components().iter().zip(b.components().iter()) {
        assert_abs_diff_eq!(p, q, epsilon = 1e-12);
    }
    let wa = a.transform(&x).expect("transform should succeed");
    let wb = b.transform(&x).expect("transform should succeed");
    for (p, q) in wa.iter().zip(wb.iter()) {
        assert_abs_diff_eq!(p, q, epsilon = 1e-12);
    }
}

// ---------------------------------------------------------------------------
// REQ-1: f32 path fits.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-1): the f32 generic path fits + transforms with finite,
/// non-negative output of the contract shapes.
#[test]
fn green_f32_path_fits() {
    let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let fitted = NMF::<f32>::new(1)
        .with_random_state(0)
        .fit(&x, &())
        .expect("f32 fit should succeed");
    assert_eq!(fitted.components().dim(), (1, 2));
    let w = fitted.transform(&x).expect("f32 transform should succeed");
    assert_eq!(w.dim(), (3, 1));
    for &v in &w {
        assert!(
            v.is_finite() && v >= 0.0,
            "f32 W entry must be finite non-negative, got {v}"
        );
    }
    assert!(fitted.reconstruction_err().is_finite());
}
