//! Divergence pins for `PCA` / `FittedPCA` against scikit-learn 1.5.2
//! `class PCA(_BasePCA)`
//! (`/home/doll/scikit-learn/sklearn/decomposition/_pca.py:121`).
//!
//! Tracking: #1499. Blocker for DIV-1: see the `#[ignore]` annotations below.
//!
//! THE HEADLINE FIXABLE DIVERGENCE (DIV-1, REQ-1): ferrolearn's `components_`
//! (and therefore the sign-dependent `transform`) have ARBITRARY signs — `fn
//! fit` (`pca.rs:457-459`) copies the faer `self_adjoint_eigen` eigenvector
//! column straight into the component row with NO sign correction. sklearn pins
//! the signs deterministically via `svd_flip(U, Vt, u_based_decision=False)`
//! (`_pca.py:647`): for each component ROW of `Vt`, `max_abs = argmax(abs(row))`
//! (`extmath.py:899`, numpy argmax → FIRST max on ties), then the WHOLE row is
//! multiplied by `sign(row[max_abs])` (`extmath.py:902-905`) so the max-abs
//! entry becomes POSITIVE. ferrolearn's fit otherwise EXACTLY mirrors sklearn's
//! `covariance_eigh` solver (`_pca.py:593-644`), so after the same row-wise sign
//! flip the components / transform match sklearn EXACTLY on non-degenerate data.
//!
//! All expected values come from the live sklearn 1.5.2 oracle, run from /tmp
//! (R-CHAR-3) — never literal-copied from the ferrolearn side.
//!
//! FIXTURE A (6x3, distinct covariance eigenvalues) — chosen because faer's raw
//! eigenvector signs DISAGREE with sklearn's `svd_flip` on ALL THREE components
//! (each ferrolearn component row has a NEGATIVE max-abs entry; sklearn pins it
//! positive). The divergence is therefore genuinely observable on every row.
//! sklearn's "auto" solver picks `full` here (max(shape)=6<=500, `_pca.py:537`);
//! `full` and `covariance_eigh` both go through the SAME `svd_flip` +
//! ratio/singular postprocess and produce identical results (verified live).

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::PCA;
use ndarray::{Array2, array};

/// Fixture A: 6 samples, 3 features, clearly-separated covariance eigenvalues.
fn fixture_a() -> Array2<f64> {
    array![
        [1.0, 0.1, 8.0],
        [2.0, 0.4, 1.0],
        [3.0, 0.0, 5.0],
        [4.0, 0.9, 2.0],
        [5.0, 0.2, 9.0],
        [6.0, 0.7, 0.5],
    ]
}

// --- live sklearn 1.5.2 oracle for Fixture A, n_components=3 (R-CHAR-3) -------
// `from sklearn.decomposition import PCA; PCA(3).fit(X)` on fixture A.

/// sklearn `m.components_` (each ROW's max-abs entry is POSITIVE — svd_flip).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_COMPONENTS: [[f64; 3]; 3] = [
    [-0.161336219074942, -0.070935617130925, 0.984346871096189],
    [0.984685837730652, 0.055197220315576, 0.165369488848810],
    [-0.066063797856960, 0.995952511464120, 0.060943986750344],
];

/// sklearn `m.transform(X)` for fixture A.
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_TRANSFORM: [[f64; 3]; 6] = [
    [4.114739739151824, -1.857218223566339, 0.111512900041358],
    [-2.958305262735716, -2.013559641682686, -0.082373051628776],
    [0.846120249426467, -0.389474736683023, -0.303041907070008],
    [-2.332098638354874, 0.148780132785217, 0.344419595139708],
    [4.446648172235152, 2.252414338236639, 0.007796946510272],
    [-4.117104259722856, 1.859058130910190, -0.078314482992554],
];

/// sklearn `m.explained_variance_` (sign-independent).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_EXPLAINED_VARIANCE: [f64; 3] = [13.712096827412307, 3.241395108836682, 0.047174730417675];

/// sklearn `m.explained_variance_ratio_` (sums to 1 at n_components==n_features).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_EXPLAINED_VARIANCE_RATIO: [f64; 3] =
    [0.806562301130092, 0.190662823546332, 0.002774875323576];

/// sklearn `m.singular_values_` (sign-independent).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const SK_SINGULAR_VALUES: [f64; 3] = [8.280125852730835, 4.025788810181603, 0.485668253119733];

// ===========================================================================
// DIV-1 (REQ-1): components_ / transform signs are faer-arbitrary; sklearn pins
// them via svd_flip(u_based_decision=False) (_pca.py:647). These FAIL today.
// ===========================================================================

/// Divergence: ferrolearn's `components_` (`pca.rs:457-459`) stores faer's raw
/// eigenvector signs and SKIPS sklearn's `svd_flip(U, Vt,
/// u_based_decision=False)` (`_pca.py:647`, `extmath.py:897-905`). On fixture A
/// all three ferrolearn component rows have a NEGATIVE max-abs entry, whereas
/// sklearn pins each row's max-abs entry POSITIVE — so the rows are exactly
/// negated relative to sklearn.
/// sklearn comp[0] = [-0.1613, -0.0709, +0.9843]; ferrolearn comp[0] = [+0.1613,
/// +0.0709, -0.9843].
/// Tracking: #1499.
#[test]
fn divergence_components_sign_value_parity() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3)
        .fit(&x, &())
        .expect("fit must succeed on non-degenerate fixture");
    let c = fitted.components();
    for (i, sk_row) in SK_COMPONENTS.iter().enumerate() {
        for (j, &sk_val) in sk_row.iter().enumerate() {
            assert_abs_diff_eq!(c[[i, j]], sk_val, epsilon = 1e-6);
        }
    }
}

/// Divergence: the sign-dependent `transform` output inherits DIV-1. sklearn's
/// `transform` is `(X - mean_) @ components_.T` (`_base.py:148-166`); because
/// ferrolearn's `components_` rows are sign-flipped relative to sklearn, the
/// corresponding `transform` columns are negated.
/// sklearn transform row0 = [4.1147, -1.8572, 0.1115]; ferrolearn negates
/// columns 0 and 2 (the rows whose sign disagrees).
/// Tracking: #1499.
#[test]
fn divergence_transform_sign_value_parity() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3)
        .fit(&x, &())
        .expect("fit must succeed on non-degenerate fixture");
    let projected = fitted
        .transform(&x)
        .expect("transform must succeed on training data");
    for (i, sk_row) in SK_TRANSFORM.iter().enumerate() {
        for (j, &sk_val) in sk_row.iter().enumerate() {
            assert_abs_diff_eq!(projected[[i, j]], sk_val, epsilon = 1e-6);
        }
    }
}

/// Divergence (cleanest characterization of the fix target): the `svd_flip`
/// invariant is that each component ROW's max-abs entry is POSITIVE
/// (`extmath.py:899-905`). ferrolearn applies NO such flip, so on fixture A
/// every component row's max-abs entry is NEGATIVE. This is the structural
/// invariant the fixer must enforce.
/// Tracking: #1499.
#[test]
fn divergence_components_sign_convention_max_abs_positive() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3)
        .fit(&x, &())
        .expect("fit must succeed on non-degenerate fixture");
    let c = fitted.components();
    for i in 0..c.nrows() {
        let row = c.row(i);
        let mut max_j = 0usize;
        let mut max_abs = 0.0f64;
        for (j, &v) in row.iter().enumerate() {
            if v.abs() > max_abs {
                max_abs = v.abs();
                max_j = j;
            }
        }
        assert!(
            row[max_j] > 0.0,
            "svd_flip invariant violated: component row {i} max-abs entry \
             (idx {max_j}) = {} is not positive",
            row[max_j]
        );
    }
}

// ===========================================================================
// GREEN GUARDS: structural SHIPPED REQs that are sign/basis-independent. These
// MUST PASS against current code — they confirm the eigenvalues/magnitudes are
// already correct and that ONLY the signs are wrong (the DIV-1 fix target).
// ===========================================================================

/// GREEN (REQ-4): `explained_variance_` matches sklearn element-wise. This is
/// sign-independent (eigenvalue magnitudes) — confirms the covariance_eigh
/// eigenvalues are already correct.
#[test]
fn green_explained_variance_matches_sklearn() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let ev = fitted.explained_variance();
    for (k, &sk) in SK_EXPLAINED_VARIANCE.iter().enumerate() {
        assert_abs_diff_eq!(ev[k], sk, epsilon = 1e-6);
    }
}

/// GREEN (REQ-4): `explained_variance_ratio_` matches sklearn and sums to 1.0
/// when n_components == n_features.
#[test]
fn green_explained_variance_ratio_matches_sklearn() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let ratio = fitted.explained_variance_ratio();
    for (k, &sk) in SK_EXPLAINED_VARIANCE_RATIO.iter().enumerate() {
        assert_abs_diff_eq!(ratio[k], sk, epsilon = 1e-6);
    }
    let sum: f64 = ratio.iter().sum();
    assert!(sum <= 1.0 + 1e-9, "ratio sum {sum} exceeds 1");
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
}

/// GREEN (REQ-4): with fewer components than features, the ratio sum is < 1.
#[test]
fn green_explained_variance_ratio_partial_le_1() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(2).fit(&x, &()).expect("fit");
    let sum: f64 = fitted.explained_variance_ratio().iter().sum();
    assert!(sum <= 1.0 + 1e-9 && sum > 0.0, "partial ratio sum = {sum}");
    // sklearn live oracle: 0.806562301130092 + 0.190662823546332.
    assert_abs_diff_eq!(
        sum,
        SK_EXPLAINED_VARIANCE_RATIO[0] + SK_EXPLAINED_VARIANCE_RATIO[1],
        epsilon = 1e-6
    );
}

/// GREEN (REQ-5): `singular_values_` matches sklearn element-wise
/// (sign-independent).
#[test]
fn green_singular_values_matches_sklearn() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let sv = fitted.singular_values();
    for (k, &sk) in SK_SINGULAR_VALUES.iter().enumerate() {
        assert_abs_diff_eq!(sv[k], sk, epsilon = 1e-6);
    }
}

/// GREEN (REQ-3): each component row is unit-norm and rows are mutually
/// orthogonal — structural orthonormality (sign/basis-independent).
#[test]
fn green_components_orthonormal() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let c = fitted.components();
    for i in 0..c.nrows() {
        let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-8);
    }
    for i in 0..c.nrows() {
        for j in (i + 1)..c.nrows() {
            let dot: f64 = c
                .row(i)
                .iter()
                .zip(c.row(j).iter())
                .map(|(a, b)| a * b)
                .sum();
            assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-8);
        }
    }
}

/// GREEN (REQ-6): with n_components == n_features, `inverse_transform`
/// round-trips `transform` exactly. Sign-invariant (componentsᵀ·components = I
/// cancels the arbitrary signs).
#[test]
fn green_inverse_transform_roundtrip_exact() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let projected = fitted.transform(&x).expect("transform");
    let recovered = fitted
        .inverse_transform(&projected)
        .expect("inverse_transform");
    for (a, b) in x.iter().zip(recovered.iter()) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-8);
    }
}

/// GREEN (REQ-1 stability / REQ-7): two independent fits of the same data give
/// identical components (determinism), even though those components diverge in
/// sign from sklearn.
#[test]
fn green_fit_is_deterministic() {
    let x = fixture_a();
    let a = PCA::<f64>::new(3).fit(&x, &()).expect("fit a");
    let b = PCA::<f64>::new(3).fit(&x, &()).expect("fit b");
    let ca = a.components();
    let cb = b.components();
    assert_eq!(ca.dim(), cb.dim());
    for (va, vb) in ca.iter().zip(cb.iter()) {
        assert_abs_diff_eq!(va, vb, epsilon = 0.0);
    }
}

// --- error / parameter contracts (REQ-7) -----------------------------------

/// GREEN (REQ-7): n_components == 0 → Err.
#[test]
fn green_err_n_components_zero() {
    let x = fixture_a();
    assert!(PCA::<f64>::new(0).fit(&x, &()).is_err());
}

/// GREEN (REQ-7): n_components > n_features → Err.
#[test]
fn green_err_n_components_too_large() {
    let x = fixture_a();
    assert!(PCA::<f64>::new(4).fit(&x, &()).is_err());
}

/// GREEN (REQ-7): n_samples < 2 → Err.
#[test]
fn green_err_insufficient_samples() {
    let x = array![[1.0, 2.0, 3.0]];
    assert!(PCA::<f64>::new(1).fit(&x, &()).is_err());
}

/// GREEN (REQ-7): transform with mismatched column count → Err.
#[test]
fn green_err_transform_col_mismatch() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    let bad = array![[1.0, 2.0]];
    assert!(fitted.transform(&bad).is_err());
}

/// GREEN (REQ-7): inverse_transform with mismatched column count → Err.
#[test]
fn green_err_inverse_transform_col_mismatch() {
    let x = fixture_a();
    let fitted = PCA::<f64>::new(3).fit(&x, &()).expect("fit");
    // inverse_transform expects n_components (3) columns, not 2.
    let bad = array![[1.0, 2.0]];
    assert!(fitted.inverse_transform(&bad).is_err());
}
