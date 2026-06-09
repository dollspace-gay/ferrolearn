//! Divergence: ferrolearn's `GaussianProcessClassifier` Laplace posterior-mode
//! loop uses a different convergence criterion than scikit-learn, producing a
//! different log-marginal-likelihood (and `pi_`) when the Newton iteration is
//! stopped before convergence (low `max_iter` / `max_iter_predict`).
//!
//! sklearn `_BinaryGaussianProcessClassifierLaplace._posterior_mode`
//! (`sklearn/gaussian_process/_gpc.py:438-462`) iterates with:
//! ```text
//!   pi = expit(f)                       # uses the OLD f
//!   ... Newton step -> new f, a ...
//!   lml = -0.5*a@f - log1p(exp(-(2y-1)*f)).sum() - log(diag(L)).sum()
//!   if lml - log_marginal_likelihood < 1e-10: break   # LML-change criterion
//! ```
//! The returned temporary `pi` (-> `self.pi_`) is `expit` of the f from BEFORE
//! the final Newton update (a deliberate one-step lag), and the loop count is
//! governed by the LML-change-< 1e-10 criterion (`_gpc.py:458-461`).
//!
//! ferrolearn (`ferrolearn-kernel/src/gp_classifier.rs:339-417`) instead breaks
//! on `max(|f_new - f|) < tol` and recomputes `pi_hat = sigmoid(f)` from the
//! FINAL post-loop f (`:420`). At full convergence the two agree to ~1e-13
//! (audited clean), but at a user-supplied low `max_iter` they stop at
//! different states and the LML diverges by ~1e-2 (well outside R-DEV-1 1e-6).
//!
//! Expected values are the LIVE sklearn 1.5.2 oracle (R-CHAR-3):
//!   GaussianProcessClassifier(kernel=RBF(1.0), optimizer=None,
//!       max_iter_predict=N).fit(X, y).log_marginal_likelihood_value_
//! for the deterministic binary fixture below, at N = 1, 2, 3:
//!   N=1 -> -3.595122002617317
//!   N=2 -> -3.5282713908340484
//!   N=3 -> -3.5258870812145808
//! ferrolearn (`max_iter` builder == sklearn `max_iter_predict`) returns:
//!   N=1 -> -3.54957715032016585
//!   N=2 -> -3.52590767429169327
//!   N=3 -> -3.52588475660819389
//!
//! Tracking: #2378

use ferrolearn_core::Fit;
use ferrolearn_kernel::gp_classifier::GaussianProcessClassifier;
use ferrolearn_kernel::gp_kernels::RBFKernel;
use ndarray::{Array1, Array2};

/// Divergence: ferrolearn GPC LML at non-converged `max_iter` diverges from
/// sklearn `_gpc.py:438-462` (LML-change-criterion + one-step-lagged `pi_`).
#[test]
#[ignore = "divergence: GPC posterior-mode convergence criterion (LML-change vs max-f-change) + pi_ one-step lag; tracking #2378"]
fn divergence_gpc_lml_low_max_iter() {
    // Deterministic binary fixture: optimizer=None, fixed RBF(1.0), y in {0,1}.
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 3.0, 3.5, 4.0]).unwrap();
    let y = Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1]);

    // Live sklearn 1.5.2 oracle (max_iter_predict = 1, 2, 3); see header.
    // ferrolearn `max_iter` maps to sklearn `max_iter_predict`.
    let sklearn_lml: [(usize, f64); 3] = [
        (1, -3.595_122_002_617_317),
        (2, -3.528_271_390_834_048_4),
        (3, -3.525_887_081_214_580_8),
    ];

    for (max_iter, expected) in sklearn_lml {
        let fitted = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)))
            .max_iter(max_iter)
            .fit(&x, &y)
            .unwrap();
        let lml = fitted.log_marginal_likelihood();
        assert!(
            (lml - expected).abs() < 1e-6,
            "GPC LML at max_iter={max_iter}: ferrolearn={lml}, sklearn oracle={expected}, \
             diff={}",
            (lml - expected).abs()
        );
    }
}
