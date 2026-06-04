//! Divergence + green-guard suite for `KernelPCA` / `FittedKernelPCA` / `Kernel`
//! (`ferrolearn-decomp/src/kernel_pca.rs`) against scikit-learn 1.5.2
//! `class KernelPCA` (`sklearn/decomposition/_kernel_pca.py:32`).
//!
//! All expected values come from the live sklearn 1.5.2 oracle (run from
//! `/tmp`, `eigen_solver='dense'` for determinism), hard-coded at full
//! precision (R-CHAR-3), never literal-copied from the ferrolearn side.
//!
//! Tracking issue: #1561.
//!
//! Two FIXABLE divergences are pinned as failing `#[ignore]`'d tests:
//!   - DIV-1 (REQ-3, blocker #1562): no `svd_flip(u=eigenvectors_, v=None)`
//!     per-eigenvector-COLUMN max-abs-positive sign convention
//!     (`_kernel_pca.py:373`) → embedding sign-diverges per component.
//!   - DIV-2 (REQ-4, blocker #1563): `coef0` default `0.0` (`kernel_pca.rs:102`)
//!     vs sklearn `coef0=1` (`_kernel_pca.py:289`).
//!
//! The remaining tests are green-guards confirming the SHIPPED structural
//! parity (the 4 kernels, eigenvalue sign+order, shapes, error contracts, f32,
//! determinism, auto-gamma) and that DIV-1 is SIGN-ONLY (eigenvalues +
//! magnitudes already match sklearn).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{Kernel, KernelPCA};
use ndarray::{Array2, array};

/// Non-degenerate 7x3 fixture with DISTINCT leading eigenvalues
/// (`[1.6336, 1.0481]`, gap ~0.59) so the svd_flip sign is unambiguous.
/// Chosen so ferrolearn's raw Jacobi sign on component 2 DISAGREES with
/// sklearn's svd_flip (component 2's max-abs entry is NEGATIVE in ferrolearn).
fn fixture() -> Array2<f64> {
    array![
        [1.49, -1.7, 1.33],
        [1.35, -0.95, -1.36],
        [-0.42, -0.18, 0.85],
        [0.19, 2.94, -1.49],
        [-0.96, -1.19, 1.27],
        [-0.73, 0.33, -0.78],
        [-1.21, -1.45, 1.28],
    ]
}

/// sklearn 1.5.2 oracle, run from `/tmp`:
/// ```text
/// X=np.array([[1.49,-1.7,1.33],[1.35,-0.95,-1.36],[-0.42,-0.18,0.85],
///   [0.19,2.94,-1.49],[-0.96,-1.19,1.27],[-0.73,0.33,-0.78],[-1.21,-1.45,1.28]])
/// m=KernelPCA(n_components=2, kernel='rbf', gamma=0.5, eigen_solver='dense').fit(X)
/// m.eigenvalues_.tolist(); m.transform(X).tolist()
/// ```
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3), _kernel_pca.py:368-370"
)]
const SK_EIGENVALUES: [f64; 2] = [1.633549109300395, 1.0480560989060168];

/// sklearn 1.5.2 oracle `KernelPCA(..., eigen_solver='dense').transform(X)`,
/// shape `(7, 2)`, full precision (R-CHAR-3). Sign INCLUDED — this is the
/// svd_flip(u=eigenvectors_, v=None) sign convention (`_kernel_pca.py:373`).
#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3), _kernel_pca.py:373 + :512"
)]
const SK_TRANSFORM: [[f64; 2]; 7] = [
    [-0.36062753121446034, -0.4476264648947989],
    [-0.4622855214487781, -0.045225416039812685],
    [0.2424469664734752, 0.3650775416777933],
    [-0.44567892167324646, -0.3494782796191525],
    [0.688764706508706, -0.09844927902441167],
    [-0.331844164884922, 0.7432716506367129],
    [0.6692244662392257, -0.16756975273633054],
];

fn rbf_2() -> KernelPCA<f64> {
    KernelPCA::<f64>::new(2)
        .with_kernel(Kernel::RBF)
        .with_gamma(0.5)
}

// ===========================================================================
// DIV-1 (REQ-3, blocker #1562): no svd_flip sign convention.
// ===========================================================================

/// Divergence: ferrolearn's `KernelPCA::fit` (`kernel_pca.rs:438`, scaling at
/// `:511-520`) stores the RAW Jacobi eigenvector columns with NO sign flip,
/// whereas sklearn applies `self.eigenvectors_, _ = svd_flip(u=self.eigenvectors_,
/// v=None)` (`_kernel_pca.py:373`; `u_based_decision=True`, `extmath.py:888-895`)
/// so each eigenvector COLUMN's max-abs entry (across rows) is POSITIVE.
///
/// On this non-degenerate fixture sklearn's `transform(X)` component 2 (column
/// index 1) has its leading entry NEGATIVE (`-0.4476...`); ferrolearn returns
/// the same magnitude with the OPPOSITE sign (`+0.4476...`). Component 1 already
/// matches. Expected values are the live sklearn dense oracle (R-CHAR-3).
///
/// Tracking: #1561, blocker #1562.
#[test]
fn divergence_svd_flip_embedding_sign() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.dim(), (7, 2));
    for (i, expected_row) in SK_TRANSFORM.iter().enumerate() {
        for (j, &expected) in expected_row.iter().enumerate() {
            assert!(
                (t[[i, j]] - expected).abs() < 1e-6,
                "embedding[{i},{j}]: ferrolearn={} sklearn={} (svd_flip sign divergence)",
                t[[i, j]],
                expected
            );
        }
    }
}

/// Divergence (focused characterization of the svd_flip invariant): sklearn's
/// `svd_flip(u=eigenvectors_, v=None)` (`_kernel_pca.py:373`,
/// `extmath.py:888-894`) forces each eigenvector COLUMN's max-abs entry (across
/// ROWS) to be POSITIVE. ferrolearn's `alphas` (`kernel_pca.rs:511-520`) is the
/// raw Jacobi eigenvector scaled by `1/sqrt(eigenvalue)` (sign-preserving), so
/// the invariant is NOT enforced.
///
/// On this fixture ferrolearn's `alphas` column 1 has its max-abs entry at row 5
/// with a NEGATIVE value (`-0.709...`), violating the svd_flip invariant.
///
/// Tracking: #1561, blocker #1562.
#[test]
fn divergence_svd_flip_alphas_max_abs_positive() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let alphas = fitted.alphas();
    for c in 0..alphas.ncols() {
        let col = alphas.column(c);
        let mut max_row = 0usize;
        let mut max_abs = 0.0f64;
        for (i, &v) in col.iter().enumerate() {
            if v.abs() > max_abs {
                max_abs = v.abs();
                max_row = i;
            }
        }
        // svd_flip invariant: the max-abs entry of every eigenvector column
        // must be positive (`extmath.py:892-894`).
        assert!(
            col[max_row] > 0.0,
            "alphas col[{c}] max-abs entry at row {max_row} = {} should be POSITIVE \
             per svd_flip(u, v=None); ferrolearn leaves the raw Jacobi sign",
            col[max_row]
        );
    }
}

// ===========================================================================
// DIV-2 (REQ-4, blocker #1563): coef0 default 0.0 vs sklearn 1.
// ===========================================================================

/// Divergence: ferrolearn's `KernelPCA::new` defaults `coef0: 0.0`
/// (`kernel_pca.rs:102`), whereas sklearn's ctor default is `coef0=1`
/// (`_kernel_pca.py:289`; live oracle Probe 3: `KernelPCA(kernel='poly').coef0
/// == 1`). For Polynomial `(gamma*dot + coef0)^degree` and Sigmoid
/// `tanh(gamma*dot + coef0)` the DEFAULT kernel — hence the embedding — differs.
///
/// Pinned precisely on the `coef0()` accessor of a fresh `KernelPCA::new` that
/// has NOT had `with_coef0` called: ferrolearn returns `0.0`, sklearn's default
/// is `1`.
///
/// Tracking: #1561, blocker #1563.
#[test]
fn divergence_coef0_default() {
    // sklearn KernelPCA() default coef0 (Probe 3, _kernel_pca.py:289).
    const SK_COEF0_DEFAULT: f64 = 1.0;
    let kpca = KernelPCA::<f64>::new(2).with_kernel(Kernel::Polynomial);
    assert!(
        (kpca.coef0() - SK_COEF0_DEFAULT).abs() < 1e-12,
        "ferrolearn KernelPCA::new default coef0 = {} but sklearn default coef0 = {} \
         (_kernel_pca.py:289)",
        kpca.coef0(),
        SK_COEF0_DEFAULT
    );
}

// ===========================================================================
// GREEN-GUARDS (must PASS against current code — structural SHIPPED).
// ===========================================================================

/// Green-guard (REQ-5/6): on the non-degenerate fixture the EIGENVALUES match
/// the live sklearn `eigen_solver='dense'` `.eigenvalues_` element-wise (sign-
/// invariant) — confirming the eigendecomposition is correct and ONLY signs
/// (DIV-1) diverge.
#[test]
fn green_eigenvalues_match_sklearn() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let ev = fitted.eigenvalues();
    assert_eq!(ev.len(), 2);
    for (i, &expected) in SK_EIGENVALUES.iter().enumerate() {
        assert!(
            (ev[i] - expected).abs() < 1e-6,
            "eigenvalue[{i}]: ferrolearn={} sklearn={}",
            ev[i],
            expected
        );
    }
}

/// Green-guard (REQ-3 is SIGN-ONLY): the embedding matches sklearn up to a
/// per-component SIGN — `abs(ferro) ~= abs(sklearn)` element-wise — confirming
/// the magnitudes are right and the only defect is the missing svd_flip.
#[test]
fn green_embedding_matches_sklearn_up_to_sign() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let t = fitted.transform(&x).expect("transform");
    for (i, expected_row) in SK_TRANSFORM.iter().enumerate() {
        for (j, &expected) in expected_row.iter().enumerate() {
            assert!(
                (t[[i, j]].abs() - expected.abs()).abs() < 1e-6,
                "abs(embedding[{i},{j}]): ferrolearn={} sklearn={}",
                t[[i, j]].abs(),
                expected.abs()
            );
        }
    }
}

/// Green-guard (REQ-1/7): all 4 kernels fit + transform to finite embeddings of
/// shape `(n_samples, n_components)`.
#[test]
fn green_four_kernels_finite_shape() {
    let x = fixture();
    let n = x.nrows();
    for kpca in [
        KernelPCA::<f64>::new(2).with_kernel(Kernel::Linear),
        KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::RBF)
            .with_gamma(0.5),
        KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::Polynomial)
            .with_degree(3)
            .with_gamma(1.0)
            .with_coef0(1.0),
        KernelPCA::<f64>::new(2)
            .with_kernel(Kernel::Sigmoid)
            .with_gamma(0.01)
            .with_coef0(1.0),
    ] {
        let kind = kpca.kernel();
        let fitted = kpca.fit(&x, &()).expect("fit");
        let t = fitted.transform(&x).expect("transform");
        assert_eq!(t.dim(), (n, 2), "{kind:?}");
        assert!(
            t.iter().all(|v| v.is_finite()),
            "{kind:?} embedding has non-finite entries"
        );
    }
}

/// Green-guard (REQ-5/6): eigenvalues non-negative and sorted descending.
#[test]
fn green_eigenvalues_nonneg_descending() {
    let x = fixture();
    let fitted = KernelPCA::<f64>::new(4)
        .with_kernel(Kernel::RBF)
        .with_gamma(0.5)
        .fit(&x, &())
        .expect("fit");
    let ev = fitted.eigenvalues();
    for &v in ev {
        assert!(v >= 0.0, "eigenvalue {v} negative");
    }
    for i in 1..ev.len() {
        assert!(ev[i - 1] >= ev[i] - 1e-12, "not descending at {i}");
    }
}

/// Green-guard (REQ-8): transform of NEW (test) data is `(n_test, n_components)`.
#[test]
fn green_transform_new_data_shape() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let x_test = array![[0.1, 0.2, 0.3], [-0.5, 1.0, -1.0]];
    let t = fitted.transform(&x_test).expect("transform");
    assert_eq!(t.dim(), (2, 2));
    assert!(t.iter().all(|v| v.is_finite()));
}

/// Green-guard (REQ-9): auto-gamma default `1/n_features` (fit without explicit
/// gamma succeeds and is deterministic).
#[test]
fn green_auto_gamma() {
    let x = fixture(); // 3 features -> gamma 1/3
    let fitted = KernelPCA::<f64>::new(2)
        .with_kernel(Kernel::RBF)
        .fit(&x, &())
        .expect("fit");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.dim(), (7, 2));
}

/// Green-guard (REQ-10): n_components==0 -> Err.
#[test]
fn green_err_n_components_zero() {
    let x = fixture();
    let r = KernelPCA::<f64>::new(0).fit(&x, &());
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

/// Green-guard (REQ-10): n_components > n_samples -> Err.
#[test]
fn green_err_n_components_too_large() {
    let x = fixture(); // 7 samples
    let r = KernelPCA::<f64>::new(20).fit(&x, &());
    assert!(matches!(r, Err(FerroError::InvalidParameter { .. })));
}

/// Green-guard (REQ-10): n_samples < 2 -> Err.
#[test]
fn green_err_insufficient_samples() {
    let x = array![[1.0, 2.0, 3.0]];
    let r = KernelPCA::<f64>::new(1).fit(&x, &());
    assert!(matches!(r, Err(FerroError::InsufficientSamples { .. })));
}

/// Green-guard (REQ-10): transform feature mismatch -> Err.
#[test]
fn green_err_transform_feature_mismatch() {
    let x = fixture();
    let fitted = rbf_2().fit(&x, &()).expect("fit");
    let bad = array![[1.0, 2.0]]; // 2 features instead of 3
    assert!(matches!(
        fitted.transform(&bad),
        Err(FerroError::ShapeMismatch { .. })
    ));
}

/// Green-guard (REQ-11): the f32 path fits + transforms.
#[test]
fn green_f32_path() {
    let x: Array2<f32> = array![
        [1.49f32, -1.7, 1.33],
        [1.35, -0.95, -1.36],
        [-0.42, -0.18, 0.85],
        [0.19, 2.94, -1.49],
        [-0.96, -1.19, 1.27],
    ];
    let fitted = KernelPCA::<f32>::new(2)
        .with_kernel(Kernel::RBF)
        .with_gamma(0.5)
        .fit(&x, &())
        .expect("fit");
    let t = fitted.transform(&x).expect("transform");
    assert_eq!(t.ncols(), 2);
    assert!(t.iter().all(|v| v.is_finite()));
}

/// Green-guard (determinism): same config -> identical alphas + embedding
/// (the Jacobi eigensolver is deterministic).
#[test]
fn green_determinism() {
    let x = fixture();
    let f1 = rbf_2().fit(&x, &()).expect("fit");
    let f2 = rbf_2().fit(&x, &()).expect("fit");
    let a1 = f1.alphas();
    let a2 = f2.alphas();
    assert_eq!(a1.dim(), a2.dim());
    for (v1, v2) in a1.iter().zip(a2.iter()) {
        assert_eq!(v1, v2, "alphas not bit-identical across fits");
    }
    let t1 = f1.transform(&x).expect("transform");
    let t2 = f2.transform(&x).expect("transform");
    for (v1, v2) in t1.iter().zip(t2.iter()) {
        assert_eq!(v1, v2, "embedding not bit-identical across fits");
    }
}
