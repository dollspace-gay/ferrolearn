//! Divergence / verify-and-document suite for `random_projection.rs`
//! (`GaussianRandomProjection` / `SparseRandomProjection` and their `Fitted*`)
//! against scikit-learn 1.5.2 `sklearn/random_projection.py`.
//!
//! This is an RNG-COUPLED VERIFY-AND-DOCUMENT unit. ferrolearn fills the
//! projection matrix from Rust `SmallRng`; sklearn fills it from numpy
//! `RandomState`. EXACT projection-matrix VALUE parity is therefore IMPOSSIBLE
//! and is a documented CARVE-OUT (REQ-4, NOT-STARTED): there is intentionally
//! NO test in this file pinning bit-exact matrix values against a numpy matrix.
//!
//! What IS deterministic given the sklearn FORMULA (and therefore verified here
//! as green guards, R-CHAR-3) are the DISTRIBUTIONAL / STRUCTURAL claims:
//!
//! - the sparse nonzero support/scale `sqrt(1/density)/sqrt(n_components)`
//!   (`_sparse_random_matrix:301`),
//! - the sparse default density `1/sqrt(n_features)` (`_check_density:148-149`),
//! - the `density == 1` dense special case `+-1/sqrt(n_components)`
//!   (`_sparse_random_matrix:271-274`),
//! - the Gaussian scale `1/sqrt(n_components)` (`_gaussian_random_matrix:200`)
//!   as a large-sample STATISTICAL guard,
//! - determinism given a seed,
//! - the `X @ R` transform contract and shapes,
//! - the error contracts (`n_components == 0`, bad density, empty input,
//!   shape mismatch, unfitted transform).
//!
//! Tracking issue: #1387.
//!
//! The oracle scale is the sklearn FORMULA `sqrt(1/density)/sqrt(n_components)`
//! (`_sparse_random_matrix:301`), recomputed here in [`sklearn_sparse_scale`]
//! (R-CHAR-3 option b: a named symbolic constant traceable to a sklearn
//! file:line, NOT a literal copied from the ferrolearn side). Live sklearn 1.5.2
//! values for cross-reference (computed from `/tmp`):
//!
//! ```text
//! python3 -c "import numpy as np; print(np.sqrt(1/0.5)/np.sqrt(4))"   # 0.7071067811865476
//! python3 -c "import numpy as np; print(np.sqrt(1/0.25)/np.sqrt(8))"  # 0.7071067811865475
//! python3 -c "import numpy as np; print(np.sqrt(1/0.1)/np.sqrt(10))"  # 1.0  (default density n_features=100, k=10)
//! python3 -c "import numpy as np; print(1/np.sqrt(4))"                # 0.5  (density==1, k=4)
//! python3 -c "import numpy as np; print(1/np.sqrt(8))"                # 0.35355339059327373 (gaussian scale k=8)
//! ```

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::random_projection::{GaussianRandomProjection, SparseRandomProjection};
use ndarray::Array2;

/// The sklearn sparse scale FORMULA `sqrt(1/density)/sqrt(n_components)`
/// (`sklearn/random_projection.py:301`). This is the oracle: it is the sklearn
/// formula, not a value read back from ferrolearn (R-CHAR-3).
fn sklearn_sparse_scale(density: f64, n_components: usize) -> f64 {
    (1.0 / density).sqrt() / (n_components as f64).sqrt()
}

// -------------------------------------------------------------------------
// REQ-2: sparse support / scale -- the load-bearing deterministic claim.
// Every nonzero entry must be EXACTLY +- sqrt(1/density)/sqrt(n_components);
// every other entry exactly 0.
// -------------------------------------------------------------------------

/// Verify (green guard): ferrolearn's `SparseRandomProjection::fit` matches
/// `sklearn/random_projection.py:301` -- every nonzero entry of the projection
/// matrix equals +- `sqrt(1/density)/sqrt(n_components)` and every zero is
/// exactly 0.0. Oracle scale from the sklearn formula (`sklearn_sparse_scale`).
/// Tracking issue: #1387.
#[test]
fn verify_sparse_support_scale_exact() {
    // (density, n_components) pairs. Oracle scale computed from the sklearn
    // formula. Live sklearn values: d=0.5 k=4 -> 0.7071067811865476;
    // d=0.25 k=8 -> 0.7071067811865475.
    let cases: &[(f64, usize)] = &[(0.5, 4), (0.25, 8)];

    for &(density, k) in cases {
        let oracle_scale = sklearn_sparse_scale(density, k);

        let x = Array2::<f64>::ones((6, 40));
        let proj = SparseRandomProjection::<f64>::new(k)
            .density(density)
            .random_state(7);
        let fitted = match proj.fit(&x, &()) {
            Ok(f) => f,
            Err(e) => panic!("fit failed for d={density} k={k}: {e:?}"),
        };
        let r = fitted.projection();

        let mut nonzero_count = 0usize;
        for &v in r.iter() {
            if v == 0.0 {
                continue;
            }
            nonzero_count += 1;
            assert!(
                (v.abs() - oracle_scale).abs() < 1e-12,
                "nonzero entry {v} != +-{oracle_scale} (sklearn :301) for d={density} k={k}"
            );
        }
        assert!(
            nonzero_count > 0,
            "expected at least one nonzero entry for d={density} k={k}"
        );
    }
}

// -------------------------------------------------------------------------
// REQ-2: sparse default density = 1/sqrt(n_features) (_check_density:148-149).
// n_features=100 -> d=0.1; with k=10 the nonzero magnitude is sqrt(1/0.1)/sqrt(10) = 1.0.
// -------------------------------------------------------------------------

/// Verify (green guard): with no explicit `.density()`, ferrolearn defaults to
/// `1/sqrt(n_features)` (sklearn `_check_density:148-149`). For n_features=100
/// the default density is 0.1, and with n_components=10 the nonzero magnitude
/// equals `sqrt(1/0.1)/sqrt(10) = 1.0`. Matching that magnitude confirms the
/// default is wired to `1/sqrt(n_features)`, not some other constant.
/// Tracking issue: #1387.
#[test]
fn verify_sparse_default_density_scale() {
    let n_features = 100usize;
    let k = 10usize;
    let default_density = 1.0 / (n_features as f64).sqrt(); // 0.1 (sklearn 'auto')
    let oracle_scale = sklearn_sparse_scale(default_density, k); // == 1.0

    // Self-check against the sklearn formula directly.
    assert!(
        (oracle_scale - 1.0).abs() < 1e-15,
        "default-density oracle scale should be 1.0, got {oracle_scale}"
    );

    let x = Array2::<f64>::ones((8, n_features));
    let proj = SparseRandomProjection::<f64>::new(k).random_state(11); // no .density()
    let fitted = match proj.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let r = fitted.projection();

    let mut nonzero_count = 0usize;
    for &v in r.iter() {
        if v == 0.0 {
            continue;
        }
        nonzero_count += 1;
        assert!(
            (v.abs() - oracle_scale).abs() < 1e-12,
            "default-density nonzero {v} != +-{oracle_scale} (sklearn _check_density:148-149 + :301)"
        );
    }
    assert!(
        nonzero_count > 0,
        "expected nonzero entries at default density"
    );
}

// -------------------------------------------------------------------------
// REQ-2: density == 1 -> every entry nonzero (+-1/sqrt(n_components)), no zeros.
// sklearn _sparse_random_matrix:271-274.
// -------------------------------------------------------------------------

/// Verify (green guard): with `.density(1.0)` every entry is nonzero and equals
/// +- `1/sqrt(n_components)` (sklearn dense special case `:271-274`). For k=4
/// the magnitude is `1/sqrt(4) = 0.5`.
/// Tracking issue: #1387.
#[test]
fn verify_sparse_density_one_dense() {
    let k = 4usize;
    let oracle_scale = sklearn_sparse_scale(1.0, k); // 1/sqrt(4) = 0.5

    let x = Array2::<f64>::ones((6, 30));
    let proj = SparseRandomProjection::<f64>::new(k)
        .density(1.0)
        .random_state(3);
    let fitted = match proj.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let r = fitted.projection();

    let zeros = r.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(
        zeros, 0,
        "density=1 must produce NO zeros (sklearn :271-274)"
    );
    for &v in r.iter() {
        assert!(
            (v.abs() - oracle_scale).abs() < 1e-12,
            "density=1 entry {v} != +-{oracle_scale} (1/sqrt(k), sklearn :271-274)"
        );
    }
}

// -------------------------------------------------------------------------
// REQ-2: f32 sparse support constant.
// -------------------------------------------------------------------------

/// Verify (green guard): the sparse support constant holds on the f32 path.
/// d=0.5, k=4 -> oracle scale 0.70710677 (f32 tolerance).
/// Tracking issue: #1387.
#[test]
fn verify_sparse_support_scale_f32() {
    let density = 0.5f64;
    let k = 4usize;
    let oracle_scale = sklearn_sparse_scale(density, k) as f32; // 0.70710677

    let x = Array2::<f32>::ones((6, 40));
    let proj = SparseRandomProjection::<f32>::new(k)
        .density(density)
        .random_state(7);
    let fitted = match proj.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let r = fitted.projection();

    let mut nonzero_count = 0usize;
    for &v in r.iter() {
        if v == 0.0 {
            continue;
        }
        nonzero_count += 1;
        assert!(
            (v.abs() - oracle_scale).abs() < 1e-6,
            "f32 nonzero {v} != +-{oracle_scale} (sklearn :301)"
        );
    }
    assert!(nonzero_count > 0, "expected nonzero entries (f32)");
}

// -------------------------------------------------------------------------
// REQ-1: Gaussian scale -- distributional, large-sample STATISTICAL guard.
// Entries ~ N(0, 1/n_components); empirical var ~ 1/k, mean ~ 0.
// sklearn _gaussian_random_matrix:200.
// -------------------------------------------------------------------------

/// Verify (STATISTICAL guard, large-sample): ferrolearn's Gaussian projection
/// entries are distributed `N(0, 1/n_components)` (sklearn
/// `_gaussian_random_matrix:200`, scale `1/sqrt(n_components)`). With
/// n_features=2000 and k=8 the empirical variance must be within 15% of
/// `1/8 = 0.125` and the empirical mean near 0. Seeded for reproducibility.
/// This is a distributional guard, NOT exact-value parity (REQ-4 carve-out).
/// Tracking issue: #1387.
#[test]
fn verify_gaussian_scale_statistical() {
    const SEED: u64 = 12345;
    let k = 8usize;
    let n_features = 2000usize;
    let oracle_var = 1.0 / (k as f64); // 0.125 (sklearn N(0, 1/n_components))

    let x = Array2::<f64>::ones((4, n_features));
    let proj = GaussianRandomProjection::<f64>::new(k).random_state(SEED);
    let fitted = match proj.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e:?}"),
    };
    let r = fitted.projection();

    let n = r.len() as f64; // 2000 * 8 = 16000 samples
    let mean: f64 = r.iter().copied().sum::<f64>() / n;
    let var: f64 = r.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;

    assert!(
        mean.abs() < 0.02,
        "Gaussian empirical mean {mean} not near 0 (expected N(0, 1/k) mean 0)"
    );
    let rel = (var - oracle_var).abs() / oracle_var;
    assert!(
        rel < 0.15,
        "Gaussian empirical variance {var} not within 15% of 1/k={oracle_var} (rel={rel}); sklearn :200"
    );
}

// -------------------------------------------------------------------------
// REQ-1 / property: determinism given a seed (bit-equal); different seeds differ.
// -------------------------------------------------------------------------

/// Verify (green guard): identical `random_state` seeds produce a bit-equal
/// projection matrix; different seeds produce a different one. (ferrolearn
/// determinism property; sklearn's analog is reproducibility across calls.)
/// Tracking issue: #1387.
#[test]
fn verify_determinism_seed() {
    let x = Array2::<f64>::ones((10, 50));

    let a = match GaussianRandomProjection::<f64>::new(6)
        .random_state(99)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("fit a: {e:?}"),
    };
    let b = match GaussianRandomProjection::<f64>::new(6)
        .random_state(99)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("fit b: {e:?}"),
    };
    assert_eq!(
        a.projection(),
        b.projection(),
        "same seed must yield bit-equal Gaussian projection"
    );

    let c = match GaussianRandomProjection::<f64>::new(6)
        .random_state(100)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("fit c: {e:?}"),
    };
    assert_ne!(
        a.projection(),
        c.projection(),
        "different seeds should yield different Gaussian projection"
    );

    // Sparse determinism too.
    let sa = match SparseRandomProjection::<f64>::new(6)
        .density(0.3)
        .random_state(99)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("sparse a: {e:?}"),
    };
    let sb = match SparseRandomProjection::<f64>::new(6)
        .density(0.3)
        .random_state(99)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("sparse b: {e:?}"),
    };
    assert_eq!(
        sa.projection(),
        sb.projection(),
        "same seed must yield bit-equal Sparse projection"
    );
}

// -------------------------------------------------------------------------
// REQ-1 / REQ-2: transform contract -- shape (n_samples, n_components),
// transform == X @ R exactly.
// -------------------------------------------------------------------------

/// Verify (green guard): `transform` output has shape (n_samples, n_components)
/// and equals `X @ projection` recomputed independently (sklearn `X @
/// components_.T`, `:810` analog). Both Gaussian and Sparse.
/// Tracking issue: #1387.
#[test]
fn verify_transform_equals_xr() {
    let x = Array2::<f64>::from_shape_fn((7, 30), |(i, j)| (i as f64) - 0.5 * (j as f64));

    let g = match GaussianRandomProjection::<f64>::new(5)
        .random_state(21)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("gaussian fit: {e:?}"),
    };
    let gout = match g.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("gaussian transform: {e:?}"),
    };
    assert_eq!(gout.shape(), &[7, 5]);
    let expected = x.dot(g.projection());
    for (a, b) in gout.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "gaussian transform != X@R: {a} vs {b}"
        );
    }

    let s = match SparseRandomProjection::<f64>::new(5)
        .density(0.4)
        .random_state(21)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("sparse fit: {e:?}"),
    };
    let sout = match s.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("sparse transform: {e:?}"),
    };
    assert_eq!(sout.shape(), &[7, 5]);
    let expected_s = x.dot(s.projection());
    for (a, b) in sout.iter().zip(expected_s.iter()) {
        assert!((a - b).abs() < 1e-12, "sparse transform != X@R: {a} vs {b}");
    }
}

// -------------------------------------------------------------------------
// REQ-3: error contracts.
// -------------------------------------------------------------------------

/// Verify (green guard): error contracts mirror sklearn's parameter guards.
///
/// - n_components == 0 -> Err (sklearn `_check_input_size:158-160`).
/// - density 0.0 and 1.5 -> Err (sklearn `_check_density:151-152`,
///   "Expected density in range ]0, 1]").
/// - zero-row input -> Err.
/// - shape mismatch on transform -> Err.
/// - unfitted transform -> Err (sklearn `check_is_fitted:450`).
///
/// Tracking issue: #1387.
#[test]
fn verify_error_contracts() {
    let x = Array2::<f64>::ones((5, 10));

    // n_components == 0
    assert!(
        GaussianRandomProjection::<f64>::new(0)
            .fit(&x, &())
            .is_err(),
        "gaussian n_components=0 must error"
    );
    assert!(
        SparseRandomProjection::<f64>::new(0).fit(&x, &()).is_err(),
        "sparse n_components=0 must error"
    );

    // density out of (0, 1]
    assert!(
        SparseRandomProjection::<f64>::new(4)
            .density(0.0)
            .fit(&x, &())
            .is_err(),
        "sparse density=0.0 must error (sklearn :151-152)"
    );
    assert!(
        SparseRandomProjection::<f64>::new(4)
            .density(1.5)
            .fit(&x, &())
            .is_err(),
        "sparse density=1.5 must error (sklearn :151-152)"
    );

    // zero-row input
    let empty = Array2::<f64>::zeros((0, 10));
    assert!(
        GaussianRandomProjection::<f64>::new(4)
            .fit(&empty, &())
            .is_err(),
        "gaussian zero-row input must error"
    );
    assert!(
        SparseRandomProjection::<f64>::new(4)
            .fit(&empty, &())
            .is_err(),
        "sparse zero-row input must error"
    );

    // shape mismatch on fitted transform
    let g = match GaussianRandomProjection::<f64>::new(4)
        .random_state(1)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("fit: {e:?}"),
    };
    let bad = Array2::<f64>::ones((5, 9));
    assert!(
        g.transform(&bad).is_err(),
        "transform with wrong n_features must error"
    );

    // unfitted transform error stub
    let unfitted = GaussianRandomProjection::<f64>::new(4);
    assert!(
        unfitted.transform(&x).is_err(),
        "unfitted gaussian transform must error (sklearn check_is_fitted:450)"
    );
    let unfitted_s = SparseRandomProjection::<f64>::new(4);
    assert!(
        unfitted_s.transform(&x).is_err(),
        "unfitted sparse transform must error"
    );
}

// -------------------------------------------------------------------------
// REQ-1: fit_transform consistency.
// -------------------------------------------------------------------------

/// Verify (green guard): `fit_transform` equals `fit` then `transform` for the
/// same seed (Gaussian).
///
/// Tracking issue: #1387.
#[test]
fn verify_fit_transform_matches_fit_then_transform() {
    let x = Array2::<f64>::from_shape_fn((9, 25), |(i, j)| 0.1 * (i + j) as f64);

    let ft = match GaussianRandomProjection::<f64>::new(4)
        .random_state(55)
        .fit_transform(&x)
    {
        Ok(o) => o,
        Err(e) => panic!("fit_transform: {e:?}"),
    };
    let fitted = match GaussianRandomProjection::<f64>::new(4)
        .random_state(55)
        .fit(&x, &())
    {
        Ok(f) => f,
        Err(e) => panic!("fit: {e:?}"),
    };
    let manual = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform: {e:?}"),
    };
    for (a, b) in ft.iter().zip(manual.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "fit_transform != fit;transform: {a} vs {b}"
        );
    }
}
