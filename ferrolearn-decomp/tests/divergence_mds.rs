//! Divergence / green-guard audit for `ferrolearn_decomp::mds`
//! (`MDS` / `FittedMDS`, SMACOF metric MDS) against scikit-learn 1.5.2
//! `sklearn/manifold/_mds.py`.
//!
//! ALGORITHM: ferrolearn's `MDS` is now SMACOF (matching sklearn), an iterative
//! Guttman-transform stress majorization (`_smacof_single`, `_mds.py:104-167`).
//! Parity is exact (~1e-6) on the FIXED-INIT path (`MDS::with_init(X0)` ↔
//! sklearn `smacof(init=X0, n_init=1)`): with a fixed init the Guttman
//! trajectory is deterministic and the embedding matches element-wise.
//!
//! The DEFAULT random-init path (`n_init` restarts, `random_state.uniform`,
//! `_mds.py:113`) is a documented NON-PARITY carve-out (Xoshiro ≠ numpy
//! RandomState, REQ-1, same class as KMeans #1388). It is NOT pinned for an
//! exact value here; instead the fixed-init parity tests below carry the
//! element-wise oracle (R-CHAR-3, live sklearn 1.5.2 from `/tmp`).
//!
//! These are GREEN-GUARDS: they MUST PASS where ferrolearn holds. A FAIL marks
//! a genuine divergence in the SHIPPED SMACOF scope.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Dissimilarity, MDS};
use ndarray::{Array2, array};

/// A fixed init shared by the parity guards (the sklearn-oracle init X0).
fn fixed_init() -> Array2<f64> {
    array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.05]]
}

/// Assert two embeddings are elementwise equal within `tol`.
fn assert_emb_eq(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim(), "embedding shape mismatch");
    let (rows, cols) = expected.dim();
    for i in 0..rows {
        for j in 0..cols {
            let a = actual[[i, j]];
            let e = expected[[i, j]];
            assert!(
                (a - e).abs() <= tol,
                "embedding [{i},{j}] = {a}, expected {e} (|diff| = {}, tol {tol})",
                (a - e).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (a) FIXED-INIT SMACOF parity (Euclidean) — element-wise vs sklearn.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-6, parity): `MDS::new(2).with_init(X0).fit(X)` matches
/// sklearn `MDS(dissimilarity='euclidean').fit_transform(X, init=X0)`
/// element-wise.
///
/// Live sklearn 1.5.2 oracle (`/tmp`, R-CHAR-3) on the 3-4-5 rectangle:
/// ```text
/// stress_ = 0.0013111846996572488, n_iter_ = 13,
/// embedding = [[-2.164424557023, -1.234049962647],
///              [ 0.57663887645,  -2.435876213413],
///              [-0.587315085045,  2.433308813391],
///              [ 2.175100765618,  1.236617362669]]
/// ```
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_smacof_fixed_init_euclidean_parity() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let sk_emb: Array2<f64> = array![
        [-2.164_424_557_023, -1.234_049_962_647],
        [0.576_638_876_45, -2.435_876_213_413],
        [-0.587_315_085_045, 2.433_308_813_391],
        [2.175_100_765_618, 1.236_617_362_669],
    ];
    let fitted = match MDS::new(2).with_init(fixed_init()).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fixed-init euclidean fit failed: {e:?}");
            return;
        }
    };
    assert_emb_eq(fitted.embedding(), &sk_emb, 1e-6);
    assert!((fitted.stress() - 0.001_311_184_699_657_248_8).abs() <= 1e-6);
    assert_eq!(fitted.n_iter(), 13);
}

// ---------------------------------------------------------------------------
// (b) FIXED-INIT SMACOF parity (precomputed) — element-wise vs sklearn.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-6/REQ-9 precomputed, parity): a precomputed non-Euclidean
/// dissimilarity matrix with a fixed init matches sklearn `smacof(D, init=X0,
/// n_init=1)` element-wise.
///
/// Live sklearn 1.5.2 oracle (`/tmp`, R-CHAR-3):
/// ```text
/// smacof(D, metric=True, init=X0, n_init=1, normalized_stress=False)
///   -> stress = 3.148219331054871, n_iter = 13,
///      embedding = [[-3.333717200034, -1.658330631573],
///                   [-0.431085112947, -0.700165295708],
///                   [-0.78675047678,   2.465105803376],
///                   [ 4.551552789761, -0.106609876095]]
/// ```
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_smacof_fixed_init_precomputed_parity() {
    let d: Array2<f64> = array![
        [0.0, 2.0, 5.0, 9.0],
        [2.0, 0.0, 3.0, 4.0],
        [5.0, 3.0, 0.0, 6.0],
        [9.0, 4.0, 6.0, 0.0],
    ];
    let sk_emb: Array2<f64> = array![
        [-3.333_717_200_034, -1.658_330_631_573],
        [-0.431_085_112_947, -0.700_165_295_708],
        [-0.786_750_476_78, 2.465_105_803_376],
        [4.551_552_789_761, -0.106_609_876_095],
    ];
    let fitted = match MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .with_init(fixed_init())
        .fit(&d, &())
    {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fixed-init precomputed fit failed: {e:?}");
            return;
        }
    };
    assert_emb_eq(fitted.embedding(), &sk_emb, 1e-6);
    assert!((fitted.stress() - 3.148_219_331_054_871).abs() <= 1e-6);
    assert_eq!(fitted.n_iter(), 13);
}

// ---------------------------------------------------------------------------
// (c) FIXED-INIT DETERMINISM — the Guttman trajectory is deterministic.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-3): two fixed-init fits on the same input produce
/// bitwise-identical embeddings (the Guttman iterates are deterministic from a
/// fixed init, unlike the default RNG path).
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_fixed_init_determinism_identical_runs() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let f1 = match MDS::new(2).with_init(fixed_init()).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "first fit failed: {e:?}");
            return;
        }
    };
    let f2 = match MDS::new(2).with_init(fixed_init()).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "second fit failed: {e:?}");
            return;
        }
    };
    let e1 = f1.embedding();
    let e2 = f2.embedding();
    assert_eq!(e1.dim(), e2.dim(), "embedding shapes must match");
    for i in 0..e1.nrows() {
        for j in 0..e1.ncols() {
            assert_eq!(
                e1[[i, j]],
                e2[[i, j]],
                "embedding [{i},{j}] differs across runs (non-deterministic)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (d) DEFAULT random-init runs and produces a finite, shaped embedding.
// ---------------------------------------------------------------------------

/// Green-guard (REQ-6 default path): the DEFAULT random-init SMACOF runs
/// (`n_init=4` restarts) and yields a finite shape-`(4,2)` embedding with a
/// finite, non-negative raw stress. The COORDINATE VALUE is a documented
/// non-parity carve-out (Xoshiro ≠ numpy RandomState, REQ-1) — only structure
/// and finiteness are asserted here.
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) reports the unexpected fit Err with diagnostics"
)]
fn green_default_random_init_finite() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    let fitted = match MDS::new(2).with_random_state(0).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "default random-init fit failed: {e:?}");
            return;
        }
    };
    assert_eq!(fitted.embedding().dim(), (4, 2));
    assert!(
        fitted.embedding().iter().all(|v| v.is_finite()),
        "embedding must be finite"
    );
    let s = fitted.stress();
    assert!(
        s.is_finite() && s >= 0.0,
        "stress must be finite >= 0, got {s}"
    );
}

// ---------------------------------------------------------------------------
// (e) ERROR CONTRACTS (REQ-5).
// ---------------------------------------------------------------------------

/// Green-guard (REQ-5): `n_components == 0` -> `fit` returns `Err`.
#[test]
fn green_error_n_components_zero() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]];
    assert!(
        MDS::new(0).fit(&x, &()).is_err(),
        "n_components=0 must error"
    );
}

/// Green-guard (REQ-5): `n_components > n_samples` -> `fit` returns `Err`
/// (ferrolearn's scoped guard; sklearn has no such upper bound —
/// `Interval(Integral, 1, None)`, `_mds.py:174` — so ferrolearn is stricter,
/// FLAGGED in the design doc REQ-5, not pinned as a divergence here).
#[test]
fn green_error_n_components_exceeds_n_samples() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]; // 4 samples
    assert!(
        MDS::new(10).fit(&x, &()).is_err(),
        "n_components(10) > n_samples(4) must error"
    );
}

/// Green-guard (REQ-5): a non-square Precomputed input -> `fit` returns `Err`.
#[test]
fn green_error_precomputed_not_square() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert!(
        MDS::new(1)
            .with_dissimilarity(Dissimilarity::Precomputed)
            .fit(&x, &())
            .is_err(),
        "non-square Precomputed must error"
    );
}

/// Green-guard (REQ-9): a fixed init with the wrong row count -> `Err`
/// (mirrors sklearn's shape check `_mds.py:118-121`).
#[test]
fn green_error_init_wrong_shape() {
    let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]; // 4 samples
    let bad_init = array![[0.1, 0.2], [0.3, -0.1]]; // 2 rows, not 4
    assert!(
        MDS::new(2).with_init(bad_init).fit(&x, &()).is_err(),
        "init with wrong n_samples must error"
    );
}
