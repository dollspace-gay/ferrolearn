//! Divergence / green-guard audit for `ferrolearn_decomp::sparse_pca`
//! (`SparsePCA` / `FittedSparsePCA`) against scikit-learn 1.5.2
//! `sklearn/decomposition/_sparse_pca.py` (`_BaseSparsePCA` / `SparsePCA`).
//!
//! HEADLINE DIVERGENCE (DIV-1, REQ-1, FIXABLE; tracking #1476):
//! `_BaseSparsePCA.transform` (`_sparse_pca.py:93-123`) does NOT project. It
//! centers `X = X - mean_` (`:117`) then solves
//! `U = ridge_regression(components_.T, X.T, ridge_alpha=0.01, solver="cholesky")`
//! (`:119-121`) and returns `U` (`:123`) — a Ridge-regularized least-squares fit
//! of the NON-orthonormal sparse components to the centered data, precisely
//! because "Sparse PCA components orthogonality is not enforced as in PCA hence
//! one cannot use a simple linear projection" (docstring `:100-101`).
//! ferrolearn's `transform` (`impl Transform<Array2<F>> for FittedSparsePCA`,
//! `sparse_pca.rs:488`) instead centers (`sparse_pca.rs:498-503`) then returns
//! the PLAIN projection `x_centered.dot(&self.components_.t())`
//! (`sparse_pca.rs:505`) — it omits the `(C·Cᵀ + ridge_alpha·I)⁻¹` factor.
//!
//! ORACLE GROUNDING (R-CHAR-3): the expected `transform` value is NOT copied
//! from the ferrolearn side. It is computed by the LIVE sklearn 1.5.2 oracle
//! `sklearn.linear_model.ridge_regression(C.T, Xc.T, 0.01, solver="cholesky")`
//! fed with ferrolearn's OWN fitted components `C` and `mean` (extracted from a
//! green fit of `SparsePCA::<f64>::new(2).with_random_state(0)` on the fixed X
//! below), run from /tmp. The ridge FORMULA + `ridge_alpha=0.01` is the sklearn
//! spec (`_sparse_pca.py:119-121`). The `divergence_transform_ridge_formula_*`
//! probe additionally pins the formula on an INDEPENDENT fixed `C` (not from
//! ferrolearn) to prove the formula itself is the thing under test (non-tautology).
//!
//! CARVE-OUT (NOT pinned, REQ-4): exact `components_` VALUE parity. ferrolearn
//! uses random-init alternating soft-threshold coordinate-descent + a
//! least-squares dictionary update (`sparse_pca.rs:265`) seeded from a Rust
//! `StdRng`; sklearn uses LARS `dict_learning` + `svd_flip` + numpy RNG +
//! per-feature alpha scaling (`_sparse_pca.py:308-336`). Different algorithm +
//! different RNG ⇒ component VALUES diverge by design (R-DEFER-3). The green-guards
//! below assert only STRUCTURAL properties (shape / sparsity / centering /
//! determinism / contracts), never component values.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::SparsePCA;
use ndarray::{Array2, array};

/// The fixed 6x5 training matrix used across the transform divergence pin.
/// Generic data; deliberately NOT axis-aligned so the fitted sparse components
/// are non-orthonormal (the regime where ridge != projection).
fn fixed_x() -> Array2<f64> {
    array![
        [1.0, 2.0, 0.0, 3.0, 1.0],
        [4.0, 0.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 0.0, 9.0, 2.0],
        [0.0, 1.0, 2.0, 0.0, 3.0],
        [3.0, 0.0, 4.0, 1.0, 5.0],
        [2.0, 6.0, 0.0, 4.0, 1.0],
    ]
}

// ===========================================================================
// DIV-1 (FIXABLE, REQ-1): transform = ridge regression, NOT plain projection.
// ===========================================================================

/// Divergence: `FittedSparsePCA::transform` (`sparse_pca.rs:505`) returns the
/// PLAIN projection `(X - mean_) @ components_.t()`, but sklearn
/// `_BaseSparsePCA.transform` (`_sparse_pca.py:119-121`) returns the RIDGE
/// regression `ridge_regression(components_.T, X_centered.T, ridge_alpha=0.01,
/// solver="cholesky")` — it is missing the `(C·Cᵀ + 0.01·I)⁻¹` factor.
///
/// Oracle (R-CHAR-3): `SK_RIDGE_U` is the live sklearn 1.5.2 output of
/// `ridge_regression(C.T, Xc.T, 0.01, solver="cholesky")` where `C` and `mean`
/// are FERROLEARN's OWN fitted components/mean for `SparsePCA::<f64>::new(2)
/// .with_random_state(0)` on `fixed_x()`, computed from /tmp:
/// ```text
/// C    = [[0.40741329629780737, 0.5982817560122407, -0.24162079059833189,
///          0.6297490378403586, -0.14528898551660596],
///         [0.2957730585934991, -0.5842156448020392, 0.6019462474542632,
///          0.3875871614667855, -0.24217201659282414]]
/// mean = [2.8333333333333335, 2.8333333333333335, 1.8333333333333333,
///          3.8333333333333335, 2.0]
/// U = ridge_regression(C.T, (fixed_x()-mean).T, 0.01, solver="cholesky")
/// ```
/// ferrolearn returns the plain projection (max-abs diff from the ridge oracle
/// is 0.7999). Tracking: #1476.
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) guards unreachable Err/Ok arms in this divergence pin"
)]
#[test]
fn divergence_transform_is_ridge_not_projection() {
    // sklearn ridge_regression oracle, shape (n_samples=6, n_components=2),
    // computed live (1.5.2) from ferrolearn's OWN fitted C + mean (see doc).
    #[allow(
        clippy::excessive_precision,
        reason = "hard-coded live sklearn 1.5.2 ridge_regression oracle (R-CHAR-3)"
    )]
    const SK_RIDGE_U: [[f64; 2]; 6] = [
        [-1.2975484318260249, -1.349816587476714],
        [0.16293758138327855, 5.194195080631285],
        [8.393119431806431, -0.08720686487644684],
        [-4.977119644128173, -1.8498773447302455],
        [-4.25491777438539, 0.7713337715103278],
        [1.9735288371498747, -2.678628055058207],
    ];

    let x = fixed_x();
    let fitted = match SparsePCA::<f64>::new(2).with_random_state(0).fit(&x, &()) {
        Ok(f) => f,
        Err(e) => {
            assert!(false, "fit unexpectedly failed: {e:?}");
            return;
        }
    };
    let got = match fitted.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            assert!(false, "transform unexpectedly failed: {e:?}");
            return;
        }
    };

    assert_eq!(got.dim(), (6, 2), "transform shape");
    for i in 0..6 {
        for k in 0..2 {
            let diff = (got[[i, k]] - SK_RIDGE_U[i][k]).abs();
            assert!(
                diff < 1e-6,
                "transform[{i}][{k}] = {} diverges from sklearn ridge oracle {} (diff {diff:.6}); \
                 ferrolearn returns the plain projection, omitting (C·Cᵀ + 0.01·I)⁻¹",
                got[[i, k]],
                SK_RIDGE_U[i][k]
            );
        }
    }
}

/// Non-tautology / formula probe (R-CHAR-3): the sklearn RIDGE transform formula
/// `ridge_regression(C.T, Xc.T, 0.01, solver="cholesky")` differs from the plain
/// projection `Xc @ C.t()` even on an INDEPENDENT fixed non-orthonormal `C`
/// (not from ferrolearn). This pins the FORMULA under test and proves it is the
/// ridge form, not whatever ferrolearn happens to emit.
///
/// Oracle (live sklearn 1.5.2, run from /tmp):
/// ```text
/// C  = [[0.8, 0.6, 0.0], [0.0, 0.6, 0.8]]
/// X  = [[1,2,3],[4,5,6],[7,8,9]]; Xc = X - X.mean(axis=0)
/// ridge_regression(C.T, Xc.T, 0.01, solver="cholesky")
///   = [[-3.065693,-3.065693],[0,0],[3.065693,3.065693]]
/// Xc @ C.T (plain projection)
///   = [[-4.2,-4.2],[0,0],[4.2,4.2]]
/// ```
/// This is a standalone in-test check that the ridge form (closed-form
/// `Xc·Cᵀ·(C·Cᵀ + 0.01·I)⁻¹`) is NOT the projection `Xc·Cᵀ`. It is un-ignored:
/// it does not exercise ferrolearn's `transform`, only documents/asserts the
/// sklearn spec the headline pin is grounded against.
#[test]
fn divergence_transform_ridge_formula_differs_from_projection() {
    // C: 2x3 non-orthonormal sparse-like components.
    let c = array![[0.8_f64, 0.6, 0.0], [0.0, 0.6, 0.8]];
    let x = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    // Center by column means (X.mean(axis=0) = [4,5,6]).
    let mean = array![4.0_f64, 5.0, 6.0];
    let mut xc = x.clone();
    for mut row in xc.rows_mut() {
        for (v, &m) in row.iter_mut().zip(mean.iter()) {
            *v -= m;
        }
    }

    // Plain projection Xc @ C.T.
    let proj = xc.dot(&c.t());

    // Closed-form ridge: U = Xc · Cᵀ · (C·Cᵀ + 0.01·I)⁻¹.
    // C·Cᵀ is 2x2; invert in-test analytically.
    let cct = c.dot(&c.t());
    let alpha = 0.01_f64;
    let a = cct[[0, 0]] + alpha;
    let b = cct[[0, 1]];
    let d = cct[[1, 1]] + alpha;
    let det = a * d - b * b;
    let inv = array![[d / det, -b / det], [-b / det, a / det]];
    let ridge = xc.dot(&c.t()).dot(&inv);

    // sklearn 1.5.2 oracle for the ridge form (from /tmp).
    let sk_ridge = array![[-3.065693_f64, -3.065693], [0.0, 0.0], [3.065693, 3.065693]];

    // The in-test closed form matches the sklearn ridge oracle ...
    for i in 0..3 {
        for k in 0..2 {
            assert!(
                (ridge[[i, k]] - sk_ridge[[i, k]]).abs() < 1e-6,
                "closed-form ridge[{i}][{k}]={} != sklearn ridge oracle {}",
                ridge[[i, k]],
                sk_ridge[[i, k]]
            );
        }
    }
    // ... and the ridge form is NOT the projection (proves the divergence target).
    let max_diff = ridge
        .iter()
        .zip(proj.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1.0,
        "ridge and plain projection must differ (max diff {max_diff:.6}); \
         sklearn transform uses ridge, ferrolearn uses projection"
    );
}

// ===========================================================================
// GREEN-GUARDS (must PASS against current code) — REQ-2 / REQ-3 structural.
// ===========================================================================

/// REQ-2 structural: components shape is `(n_components, n_features)`
/// (`_sparse_pca.py:220`; `sparse_pca.rs:150`).
#[test]
fn green_components_shape() {
    let x = fixed_x();
    let fitted = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed on fixed_x");
    assert_eq!(fitted.components().dim(), (2, 5));
}

/// REQ-2 structural: the L1 penalty induces EXACT zeros in the fitted components
/// (`soft_threshold`, `sparse_pca.rs:189`, the analogue of `dict_learning`'s
/// lasso, `_sparse_pca.py:313`). With a reasonable alpha, at least one component
/// entry is exactly 0.0. Structural sparsity only — NOT a value comparison.
#[test]
fn green_components_have_exact_zeros() {
    let x = fixed_x();
    let fitted = SparsePCA::<f64>::new(2)
        .with_alpha(5.0)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed on fixed_x");
    let zeros = fitted.components().iter().filter(|v| **v == 0.0).count();
    assert!(
        zeros > 0,
        "L1 penalty should produce at least one exact zero, found {zeros}"
    );
}

/// REQ-2 structural: `mean_` equals the column means of X
/// (`_sparse_pca.py:83`, `mean_ = X.mean(axis=0)`; `sparse_pca.rs:296-307`).
/// Oracle = arithmetic column means computed in-test (a definitional constant,
/// not a ferrolearn value).
#[test]
fn green_mean_is_column_means() {
    let x = fixed_x();
    let (n, p) = x.dim();
    let mut expected = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += x[[i, j]];
        }
        expected[j] = s / n as f64;
    }
    let fitted = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed on fixed_x");
    let mean = fitted.mean();
    for j in 0..p {
        assert!(
            (mean[j] - expected[j]).abs() < 1e-9,
            "mean[{j}] = {} != column mean {}",
            mean[j],
            expected[j]
        );
    }
}

/// REQ-2 structural: same `random_state` seed → identical components AND identical
/// transform across two independent fits (`StdRng`, `sparse_pca.rs:310-311`).
#[test]
fn green_determinism_same_seed() {
    let x = fixed_x();
    let f1 = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit 1");
    let f2 = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit 2");
    let c1 = f1.components();
    let c2 = f2.components();
    assert_eq!(c1.dim(), c2.dim());
    for (a, b) in c1.iter().zip(c2.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "components differ across seeded runs"
        );
    }
    let t1 = f1.transform(&x).expect("transform 1");
    let t2 = f2.transform(&x).expect("transform 2");
    for (a, b) in t1.iter().zip(t2.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "transform differs across seeded runs"
        );
    }
}

/// REQ-2 structural: the fit converges within max_iter and the reconstruction
/// (transform output) is finite. `n_iter_` <= max_iter (`sparse_pca.rs:340-392`).
#[test]
fn green_converges_finite() {
    let x = fixed_x();
    let max_iter = 50;
    let fitted = SparsePCA::<f64>::new(2)
        .with_max_iter(max_iter)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed");
    assert!(fitted.n_iter() >= 1);
    assert!(fitted.n_iter() <= max_iter, "n_iter exceeds max_iter");
    let t = fitted.transform(&x).expect("transform");
    assert!(t.iter().all(|v| v.is_finite()), "transform must be finite");
}

/// REQ-3 contract: `n_components == 0` → `fit` returns `Err(InvalidParameter)`
/// (`sparse_pca.rs:268-273`).
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) marks an unreachable non-error match arm"
)]
#[test]
fn green_n_components_zero_errors() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
    match SparsePCA::<f64>::new(0).fit(&x, &()) {
        Err(FerroError::InvalidParameter { .. }) => {}
        other => assert!(false, "expected InvalidParameter, got {other:?}"),
    }
}

/// REQ-3 contract: `n_components > n_features` → `fit` returns
/// `Err(InvalidParameter)` (`sparse_pca.rs:274-282`).
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) marks an unreachable non-error match arm"
)]
#[test]
fn green_n_components_too_large_errors() {
    let x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    match SparsePCA::<f64>::new(5).fit(&x, &()) {
        Err(FerroError::InvalidParameter { .. }) => {}
        other => assert!(false, "expected InvalidParameter, got {other:?}"),
    }
}

/// REQ-3 contract: `transform` shape is `(n_samples, n_components)`, and a
/// column-count mismatch returns `Err(ShapeMismatch)` (`sparse_pca.rs:490-496`).
#[allow(
    clippy::assertions_on_constants,
    reason = "assert!(false, ...) marks an unreachable non-error match arm"
)]
#[test]
fn green_transform_shape_and_mismatch() {
    let x = fixed_x();
    let fitted = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.dim(), (6, 2), "transform shape (n_samples, n_components)");

    let bad = array![[1.0_f64, 2.0, 3.0]]; // 3 cols != 5 features
    match fitted.transform(&bad) {
        Err(FerroError::ShapeMismatch { .. }) => {}
        other => assert!(false, "expected ShapeMismatch, got {other:?}"),
    }
}
