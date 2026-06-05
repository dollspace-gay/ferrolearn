//! Divergence + green-guard tests for `ferrolearn-kernel::Nystroem` against
//! scikit-learn 1.5.2 `sklearn.kernel_approximation.Nystroem`
//! (`sklearn/kernel_approximation.py:827-1088`).
//!
//! Tracking: unit #1902. Per-REQ blockers filed by the critic:
//!
//! - poly/sigmoid default `coef0` (`0` vs sklearn `1`) — #1903 (DETERMINISTIC,
//!   pinned by `divergence_poly_default_coef0` below).
//! - RNG / exact transform value parity (umbrella) — #1904.
//! - normalization form `V·D^{-1/2}` vs symmetric `V·D^{-1/2}·Vᵀ` — #1905.
//! - SVD vs self-adjoint eigendecomposition (indefinite kernels) — #1906.
//! - eigenvalue floor (`zero` vs `1e-12`) — #1907.
//! - missing `n_components > n_samples` warning — #1908.
//! - missing API surface (`component_indices_`/`normalization_`/string kernel)
//!   — #1909.
//! - ferray substrate — #1910.
//!
//! R-CHAR-3: every expected value is recomputed in-test from the sklearn kernel
//! FORMULA (RBF `exp(-γ‖x−y‖²)`, polynomial `(γ⟨x,y⟩+coef0)^degree`) or comes
//! from the live sklearn 1.5.2 oracle (run from /tmp). NEVER copied from
//! ferrolearn's own `transform` output.

use ferrolearn_core::{Fit, Transform};
use ferrolearn_kernel::{KernelType, Nystroem};
use ndarray::{Array2, ArrayView1, array};

/// RBF kernel formula `exp(-gamma * ||x - y||^2)`
/// (`sklearn/metrics/pairwise.py` `rbf_kernel`). Used to recompute the expected
/// Gram in-test, NOT copied from ferrolearn (R-CHAR-3).
fn rbf(x: ArrayView1<f64>, y: ArrayView1<f64>, gamma: f64) -> f64 {
    let sq: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    (-gamma * sq).exp()
}

/// Polynomial kernel formula `(gamma * <x, y> + coef0)^degree`
/// (`sklearn/metrics/pairwise.py` `polynomial_kernel`). Used to recompute the
/// expected Gram in-test (R-CHAR-3).
fn poly(x: ArrayView1<f64>, y: ArrayView1<f64>, gamma: f64, coef0: f64, degree: i32) -> f64 {
    let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    (gamma * dot + coef0).powi(degree)
}

/// Build the Gram matrix `Z·Zᵀ` of a transformed embedding.
fn gram(z: &Array2<f64>) -> Array2<f64> {
    z.dot(&z.t())
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-1: RBF kernel reconstruction at n_components == n_samples.
// ---------------------------------------------------------------------------

/// Green guard: at `n_components == n_samples` the Nystroem identity
/// `Z·Zᵀ ≈ rbf_kernel(X, X, gamma)` holds
/// (`sklearn/kernel_approximation.py:1030-1066`). This reconstruction is
/// invariant to basis permutation AND to ferrolearn's `Vᵀ`-rotation of the
/// embedding, so ferrolearn MATCHES sklearn here even though element-wise `Z`
/// differs (#1905).
///
/// Expected Gram is recomputed in-test from the RBF FORMULA (R-CHAR-3), not
/// copied from ferrolearn. Live oracle cross-check (from /tmp):
/// `Nystroem(kernel='rbf', gamma=0.5, n_components=8, random_state=0)` on this
/// 8×3 X reconstructs `rbf_kernel(X,X,gamma=0.5)` to `max abs err 4.66e-15`.
///
/// Tracking: #1902 (REQ-1, SHIPPED).
#[test]
fn green_rbf_reconstruction_full_basis() {
    let x: Array2<f64> = array![
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [2.2, 2.3, 2.4],
    ];
    let gamma = 0.5;
    let n = x.nrows();

    let fitted = Nystroem::<f64>::new()
        .with_kernel(KernelType::Rbf)
        .with_gamma(gamma)
        .with_n_components(n) // full basis
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed");
    let z = fitted.transform(&x).expect("transform should succeed");
    let g = gram(&z);

    // Expected Gram = rbf_kernel(X, X, gamma), recomputed from the formula.
    for i in 0..n {
        for j in 0..n {
            let expected = rbf(x.row(i), x.row(j), gamma);
            let actual = g[[i, j]];
            assert!(
                (actual - expected).abs() <= 1e-10,
                "Z·Zᵀ[{i},{j}] = {actual} diverges from rbf_kernel = {expected} \
                 (reconstruction must hold at full basis)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// FAILING pin — REQ-9: polynomial default coef0 (sklearn 1 vs ferrolearn 0).
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `Nystroem::new` sets `coef0 = F::zero()`
/// (`nystroem.rs`, `coef0: F::zero()`), but sklearn's `Nystroem` constructor
/// `coef0=None` (`sklearn/kernel_approximation.py:962`) →
/// `_get_kernel_params` omits it (`:1068-1075`) → `polynomial_kernel` uses its
/// OWN default `coef0=1`. So at the default, the basis kernel — and hence the
/// reconstructed Gram `Z·Zᵀ` at full basis — uses coef0=1 in sklearn, coef0=0
/// in ferrolearn.
///
/// This fixture isolates coef0 cleanly: `degree=1, gamma=1` makes the basis
/// kernel `K = X·Xᵀ + coef0`, which is FULL-RANK PSD for both coef0=0 and
/// coef0=1 (live oracle from /tmp: min eigenvalue 0.122 and 0.130 resp.). So
/// the SVD-vs-eigen indefiniteness divergence (#1906) and the eigenvalue-floor
/// divergence (#1907) are NOT triggered — the ONLY thing that moves `Z·Zᵀ` is
/// coef0.
///
/// Live oracle (from /tmp): `Nystroem(kernel='polynomial', degree=1, gamma=1,
/// n_components=5, random_state=0).fit(X)` reconstructs
/// `polynomial_kernel(X,X,degree=1,gamma=1,coef0=1)` to `6.2e-15`; the coef0=0
/// Gram differs from it by exactly 1.0 in every entry.
///
/// sklearn returns: `Z·Zᵀ ≈ polynomial_kernel(X,X,coef0=1)`.
/// ferrolearn returns: `Z·Zᵀ ≈ polynomial_kernel(X,X,coef0=0)` (off by 1.0).
///
/// Tracking: #1903
#[test]
fn divergence_poly_default_coef0() {
    let x: Array2<f64> = array![
        [1.0, 0.5, 0.2, 0.1, 0.3],
        [0.3, 1.2, 0.4, 0.7, 0.2],
        [0.6, 0.1, 1.5, 0.2, 0.9],
        [0.2, 0.8, 0.3, 1.1, 0.4],
        [0.5, 0.4, 0.7, 0.3, 1.3],
    ];
    let gamma = 1.0;
    let n = x.nrows();

    // ferrolearn at DEFAULT coef0 (no with_coef0 call) — should be 1 per sklearn.
    let fitted = Nystroem::<f64>::new()
        .with_kernel(KernelType::Polynomial)
        .with_degree(1)
        .with_gamma(gamma)
        .with_n_components(n) // full basis -> reconstruction holds
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit should succeed");
    let z = fitted.transform(&x).expect("transform should succeed");
    let g = gram(&z);

    // Expected Gram = polynomial_kernel(X, X, degree=1, gamma=1, coef0=1),
    // recomputed from the sklearn formula with the sklearn DEFAULT coef0=1.
    for i in 0..n {
        for j in 0..n {
            let expected = poly(x.row(i), x.row(j), gamma, 1.0, 1);
            let actual = g[[i, j]];
            assert!(
                (actual - expected).abs() <= 1e-9,
                "Z·Zᵀ[{i},{j}] = {actual} diverges from polynomial_kernel(coef0=1) \
                 = {expected}; ferrolearn defaults coef0=0 (sklearn default is 1)"
            );
        }
    }
}
