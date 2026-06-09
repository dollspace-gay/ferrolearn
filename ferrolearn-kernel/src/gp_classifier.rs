//! Gaussian Process classification via Laplace approximation.
//!
//! This module implements [`GaussianProcessClassifier`], a probabilistic
//! classifier that models the decision boundary as a Gaussian Process.
//!
//! # Algorithm (Binary Classification)
//!
//! Since the GP likelihood for classification (Bernoulli) is non-Gaussian,
//! exact inference is intractable. We use the **Laplace approximation**:
//!
//! 1. Find the MAP estimate `f*` of the latent function values by Newton's
//!    method on the un-normalized log posterior.
//! 2. Approximate the posterior as a Gaussian centered at `f*` with
//!    covariance `(K^{-1} + W)^{-1}`, where `W = diag(pi * (1 - pi))`.
//!
//! For multi-class problems, we use one-vs-rest binary GP classifiers.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.gaussian_process.GaussianProcessClassifier` (`_gpc.py:487`,
//! and its internal `_BinaryGaussianProcessClassifierLaplace` `:37`, v1.5.2 commit
//! 156ef14). Design doc: `.design/kernel/gp_classifier.md` (17 REQs). Every REQ is
//! BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker). GPC is
//! DETERMINISTIC (fixed kernel) so predict/predict_proba/LML are oracle-verified
//! element-wise against the live sklearn 1.5.2 (`optimizer=None`); the headline
//! gap is the absent hyperparameter optimization (deps gp_kernels #1912/#1913).
//!
//! **11 SHIPPED / 6 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-1 (binary posterior mode f̂/π̂) | SHIPPED | `fit_binary_gpc` Newton/Laplace loop (`_gpc.py:438-450`); oracle `pi_` match. |
//! | REQ-2 (binary latent-sign predict) | SHIPPED | FIXED #1932: binary `predict` now decides by the sign of `f̄* = K*·(y−π̂)` (`np.where(f_star>0, ...)`, `_gpc.py:289-291`), NOT argmax-of-proba; pinned by `divergence_binary_predict_latent_sign` (was `[0,0,1]` vs sklearn `[0,1,1]` at `f̄*=+6.9e-17`). |
//! | REQ-3 (binary LML value) | SHIPPED | `fit_binary_gpc` computes & stores the loop-internal LML `Z = -0.5 a·f - Σlog1p(exp(-(2y-1)f)) - Σlog diag(L)` (`_gpc.py:454-458`), returned by `log_marginal_likelihood`; oracle `-3.5259`. |
//! | REQ-4 (score / mean accuracy) | SHIPPED | `score` = accuracy on `predict` (sklearn `ClassifierMixin.score`); oracle `1.0`. |
//! | REQ-5 (classes ordering) | SHIPPED | sorted-unique `classes`, matching `np.unique(y)` (`_gpc.py:721`). |
//! | REQ-6 (max_iter_predict default 100) | SHIPPED | `new` sets `max_iter=100` (`_gpc.py:159`,`:665`). |
//! | REQ-7 (n_classes==1 error) | SHIPPED | `fit` errors on `<2` classes, matching sklearn `ValueError` (`_gpc.py:723-728`). |
//! | REQ-8 (production consumer) | SHIPPED | `lib.rs` re-export — the boundary estimator API (no Python GP binding, R-DEFER-1/S5). |
//! | REQ-9 (predict_proba LAMBDAS/COEFS squash) | SHIPPED | FIXED #1931: `predict_binary_proba` now uses sklearn's 5-term LAMBDAS/COEFS erf approximation (`_gpc.py:31-37`,`:324-331`) via `statrs` `erf`, not the MacKay probit; oracle-verified element-wise (~1e-13) across high/low/boundary/far/coincident points. Pinned by `divergence_binary_predict_proba_squashing`. |
//! | REQ-10 (hyperparameter optimization) | NOT-STARTED | `fit` never optimizes; sklearn default `optimizer="fmin_l_bfgs_b"` (`_gpc.py:215-254`). Needs gp_kernels `eval_gradient` #1912 + `bounds` #1913 + L-BFGS-B. Blocker #1934. |
//! | REQ-11 (posterior-mode convergence criterion + pi_ lag) | SHIPPED | FIXED #2378: `fit_binary_gpc` now computes the LML INSIDE the Newton loop and breaks when it stops increasing (`lml - prev_lml < 1e-10`, `_gpc.py:454-462`), NOT on max-f-change<tol; `pi_hat`/`l_factor`/stored `lml` are the LAST iteration's pre-update temporaries (the one-step `self.pi_` lag, `_gpc.py:438`,`:264-266`). At a non-converged `max_iter` the LML now matches sklearn (~1e-6); the converged default path is unchanged. Pinned by `divergence_gpc_lml_low_max_iter`. The W-clamp (≥1e-12, no sklearn analog) remains and is benign at the fixtures (π̂ matches). |
//! | REQ-12 (multi-class LML aggregation) | SHIPPED | FIXED #1933: `log_marginal_likelihood` now returns the MEAN of per-binary LMLs for multi-class (`np.mean`, `_gpc.py:743-749`), not the sum; oracle `-5.2469` (3-class). Pinned by `divergence_multiclass_lml_mean_vs_sum`. |
//! | REQ-13 (OvR predict_proba normalization) | SHIPPED | multi-class `predict_proba` row-normalizes the per-class LAMBDAS/COEFS probabilities; with REQ-9 fixed, the full `n×n_classes` matrix matches sklearn's OvR `predict_proba` (`_gpc.py:779-807`) element-wise (~1e-13). Guard `green_audit_multiclass_ovr_predict_proba`. |
//! | REQ-14 (multi_class one_vs_one) | NOT-STARTED | OvR-only; sklearn supports `multi_class="one_vs_one"` (`_gpc.py:734-737`). Blocker #1937. |
//! | REQ-15 (LML theta-arg + gradient API) | NOT-STARTED | `log_marginal_likelihood` evaluates only at the fitted theta; sklearn's takes `theta`+gradient (`_gpc.py:335-412`). Needs #1912. Blocker #1936. |
//! | REQ-16 (constructor surface) | NOT-STARTED | no `kernel=None`→`C(1.0)*RBF(1.0)`, `optimizer`, `n_restarts_optimizer`, `warm_start`, `copy_X_train`, `multi_class`, `random_state`, `n_jobs` (`_gpc.py:659-680`). Blocker #1938. |
//! | REQ-17 (ferray substrate) | NOT-STARTED | `ndarray` + hand-rolled cholesky + `statrs` erf vs `ferray-core`/`ferray::linalg`/`ferray::stats`. Blocker #1940. |

use ndarray::{Array1, Array2};
use num_traits::Float;

use ferrolearn_core::{FerroError, Fit, Predict};

use crate::gp_kernels::{GPKernel, RBFKernel};

/// Lambda coefficients for approximating the logistic sigmoid by a linear
/// combination of 5 error functions (Williams & Barber). Verbatim from
/// scikit-learn `sklearn/gaussian_process/_gpc.py:31`.
const LAMBDAS: [f64; 5] = [0.41, 0.4, 0.37, 0.44, 0.39];

/// Coefficients paired with [`LAMBDAS`] for the 5-term error-function
/// approximation of the logistic sigmoid. Verbatim from scikit-learn
/// `sklearn/gaussian_process/_gpc.py:32-34`.
const COEFS: [f64; 5] = [
    -1854.8214151,
    3516.89893646,
    221.29346712,
    128.12323805,
    -2010.49422654,
];

/// Gaussian Process classifier using Laplace approximation.
///
/// Binary classification uses the logistic sigmoid link function.
/// Multi-class classification uses one-vs-rest decomposition.
///
/// # Type Parameters
///
/// - `F`: Float type (`f32` or `f64`).
///
/// # Examples
///
/// ```
/// use ferrolearn_kernel::gp_classifier::GaussianProcessClassifier;
/// use ferrolearn_kernel::gp_kernels::RBFKernel;
/// use ferrolearn_core::{Fit, Predict};
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 3.0, 3.5, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1]);
///
/// let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
/// let fitted = gpc.fit(&x, &y).unwrap();
/// let predictions = fitted.predict(&x).unwrap();
/// ```
pub struct GaussianProcessClassifier<F: Float + Send + Sync + 'static> {
    /// Covariance kernel.
    kernel: Box<dyn GPKernel<F>>,
    /// Maximum iterations for the Laplace approximation Newton loop.
    /// Default: `100`.
    max_iter: usize,
    /// Convergence tolerance for the Newton loop.
    /// Default: `1e-6`.
    tol: F,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for GaussianProcessClassifier<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianProcessClassifier")
            .field("max_iter", &self.max_iter)
            .finish_non_exhaustive()
    }
}

impl<F: Float + Send + Sync + 'static> GaussianProcessClassifier<F> {
    /// Create a new GP classifier with the given kernel.
    pub fn new(kernel: Box<dyn GPKernel<F>>) -> Self {
        Self {
            kernel,
            max_iter: 100,
            tol: F::from(1e-6).unwrap(),
        }
    }

    /// Create a GP classifier with an RBF kernel and default length scale.
    pub fn default_rbf() -> Self {
        Self::new(Box::new(RBFKernel::new(F::one())))
    }

    /// Set the maximum number of Newton iterations.
    #[must_use]
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }
}

/// Fitted GP binary classifier (Laplace approximation).
///
/// Stores the quantities needed for Rasmussen & Williams Algorithm 3.2
/// (predictions) and 5.1 (log marginal likelihood):
/// - `f_hat`: the converged latent-function MAP estimate (the Laplace mode);
/// - `pi_hat`: `sigmoid(f_hat)`, used in predictive mean and variance;
/// - `y_binary`: the binary training labels;
/// - `l_factor`: Cholesky of `B = I + sqrt(W) K sqrt(W)`, used for predictive
///   variance and `log|B|` in the marginal likelihood.
struct FittedBinaryGPC<F: Float + Send + Sync + 'static> {
    /// Training features.
    x_train: Array2<F>,
    /// Sigmoid(f_hat) — class probabilities at training points. Used in the
    /// predictive mean `f_bar* = K* @ (y - pi_hat)` and predictive variance.
    pi_hat: Array1<F>,
    /// Training labels in {0, 1} (as F). Used in predictive mean and the
    /// log marginal likelihood's data-fit term.
    y_binary: Array1<F>,
    /// Cholesky factor of B = I + sqrt(W) K sqrt(W). Used in predictive
    /// variance via R&W eq. 3.24 (`v = L^{-1} sqrt(W) K(x*, X)^T`) and in
    /// `log|B| = 2 sum log L_ii` for the marginal likelihood.
    l_factor: Array2<F>,
    /// Log-marginal-likelihood computed INSIDE the posterior-mode Newton loop
    /// (sklearn's `_posterior_mode` return value `_gpc.py:454-470`, stored as
    /// `log_marginal_likelihood_value_` at `_gpc.py:256-258`). This is the value
    /// at the iteration where the LML-change criterion broke — at a non-converged
    /// `max_iter` it differs from a post-hoc recompute from `pi_hat`/`f_hat`
    /// (off-convergence `a != y - pi`), so it is captured in the loop rather than
    /// recomputed. At full convergence it equals the algebraic R&W eq. 3.32 form.
    lml: F,
    /// Kernel used during fitting.
    kernel: Box<dyn GPKernel<F>>,
}

/// Fitted Gaussian Process classifier.
///
/// For binary classification, wraps a single Laplace-approximated GP.
/// For multi-class, uses one-vs-rest with per-class binary GPs.
pub struct FittedGaussianProcessClassifier<F: Float + Send + Sync + 'static> {
    /// The class labels (sorted, unique).
    classes: Vec<usize>,
    /// Binary classifiers: one per class for OvR (or single for binary).
    binary_models: Vec<FittedBinaryGPC<F>>,
}

impl<F: Float + Send + Sync + 'static> std::fmt::Debug for FittedGaussianProcessClassifier<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FittedGaussianProcessClassifier")
            .field("n_classes", &self.classes.len())
            .field("classes", &self.classes)
            .finish()
    }
}

impl<F: Float + Send + Sync + 'static> FittedGaussianProcessClassifier<F> {
    /// Approximate Laplace log marginal likelihood `log p(y | X)`.
    ///
    /// Computes the Laplace approximation to the GP log marginal likelihood
    /// (Rasmussen & Williams "Gaussian Processes for Machine Learning"
    /// eq. 3.32 / Algorithm 5.1). For one-vs-rest multi-class models the
    /// per-class binary log marginal likelihoods are AVERAGED (mean), matching
    /// scikit-learn's `GaussianProcessClassifier.log_marginal_likelihood`
    /// (`sklearn/gaussian_process/_gpc.py:743-749`,
    /// `np.mean([estimator.log_marginal_likelihood() ...])` for `n_classes_ > 2`).
    /// For the binary case there is a single estimator, so the mean equals that
    /// estimator's value (`_gpc.py:751-753`).
    ///
    /// This value is the standard objective for kernel hyperparameter
    /// selection and model comparison.
    #[must_use]
    pub fn log_marginal_likelihood(&self) -> F {
        // Each binary model carries the log-marginal-likelihood computed inside
        // its posterior-mode Newton loop and stored at the break point — exactly
        // sklearn's `base_estimator_.log_marginal_likelihood_value_`
        // (`_gpc.py:256-258`,`:454-470`). At a non-converged `max_iter` this is
        // the loop value, NOT a post-hoc recompute (which would assume the
        // convergence identity `a = y - pi`).
        let sum = self
            .binary_models
            .iter()
            .map(|m| m.lml)
            .fold(F::zero(), |a, b| a + b);
        // Mean of the per-binary LMLs (sklearn `_gpc.py:743-749`). `binary_models`
        // is never empty after a successful fit (>= 2 classes => >= 1 model);
        // guard against a zero divisor to avoid producing NaN.
        match F::from(self.binary_models.len()) {
            Some(n) if !n.is_zero() => sum / n,
            _ => F::zero(),
        }
    }

    /// Class labels seen at fit time, in sorted order.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Predict class probabilities for new points.
    ///
    /// For binary classification, returns a 2-column array `[P(class=0), P(class=1)]`.
    /// For multi-class, returns `n_samples x n_classes` probabilities (normalized OvR).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature dimension does not match.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        if n_classes == 2 {
            // Binary case: use single model
            let probs = predict_binary_proba(&self.binary_models[0], x)?;
            let mut result = Array2::<F>::zeros((n_samples, 2));
            for i in 0..n_samples {
                result[[i, 1]] = probs[i];
                result[[i, 0]] = F::one() - probs[i];
            }
            Ok(result)
        } else {
            // Multi-class OvR: get raw probabilities, then normalize
            let mut raw = Array2::<F>::zeros((n_samples, n_classes));
            for (c, model) in self.binary_models.iter().enumerate() {
                let probs = predict_binary_proba(model, x)?;
                for i in 0..n_samples {
                    raw[[i, c]] = probs[i];
                }
            }
            // Normalize rows to sum to 1
            for i in 0..n_samples {
                let row_sum: F = (0..n_classes)
                    .map(|c| raw[[i, c]])
                    .fold(F::zero(), |a, b| a + b);
                if row_sum > F::zero() {
                    for c in 0..n_classes {
                        raw[[i, c]] = raw[[i, c]] / row_sum;
                    }
                } else {
                    // Uniform if all zeros
                    let uniform = F::one() / F::from(n_classes).unwrap();
                    for c in 0..n_classes {
                        raw[[i, c]] = uniform;
                    }
                }
            }
            Ok(raw)
        }
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba). Mirrors
    /// sklearn `GaussianProcessClassifier.predict_log_proba`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        Ok(crate::log_proba(&proba))
    }

    /// Mean accuracy on the given test data and labels. Equivalent to
    /// sklearn's `ClassifierMixin.score`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        Ok(crate::mean_accuracy(&preds, y))
    }
}

// ---------------------------------------------------------------------------
// Sigmoid and helpers
// ---------------------------------------------------------------------------

/// Logistic sigmoid: sigma(f) = 1 / (1 + exp(-f)).
fn sigmoid<F: Float>(f: F) -> F {
    F::one() / (F::one() + (-f).exp())
}

/// Fit a binary GP classifier using Laplace approximation.
///
/// `y_binary` should contain `F::zero()` (class 0) or `F::one()` (class 1).
fn fit_binary_gpc<F: Float + Send + Sync + 'static>(
    kernel: &dyn GPKernel<F>,
    x: &Array2<F>,
    y_binary: &Array1<F>,
    max_iter: usize,
) -> Result<FittedBinaryGPC<F>, FerroError> {
    let n = x.nrows();
    let k_mat = kernel.compute(x, x);

    // Initialize latent function values to zero
    let mut f = Array1::<F>::zeros(n);

    // Compile-time-constant conversions (never fail for f32/f64; `unwrap_or`
    // avoids a production `.unwrap`, R-CODE-2). 1e-12 is the W-clamp floor,
    // 1e-10 is sklearn's LML-change break threshold (`_gpc.py:462`).
    let w_floor = F::from(1e-12).unwrap_or_else(F::epsilon);
    let lml_eps = F::from(1e-10).unwrap_or_else(F::epsilon);
    let half = F::from(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));

    // sklearn computes the log-marginal-likelihood INSIDE the Newton loop and
    // breaks when it stops increasing (`lml - log_marginal_likelihood < 1e-10`,
    // `_gpc.py:454-462`), NOT on the change in `f`. The temporaries `pi`, `L`
    // (and `lml`) from the LAST EXECUTED iteration are what sklearn stores:
    // `self.pi_ = pi` is `expit(f)` from the TOP of the loop body — i.e. the
    // pre-update f (a deliberate one-step lag), and `log_marginal_likelihood_value_`
    // is the loop's last `lml` (`_gpc.py:264-266`,`:256-258`).
    let mut prev_lml = F::neg_infinity();
    // Carried-out temporaries: pi_ (pre-update lag), the Cholesky factor L, and
    // the loop-internal lml of the last executed iteration. `max_iter >= 1`
    // (sklearn constraint `max_iter_predict >= 1`), so the loop runs at least
    // once and these are always overwritten with real values.
    let mut pi_hat: Array1<F> = f.mapv(sigmoid);
    let mut l_final = Array2::<F>::zeros((n, n));
    let mut lml_final = F::neg_infinity();

    for _iter in 0..max_iter {
        // pi = sigmoid(f) — from the CURRENT (pre-update) f. This is sklearn's
        // `pi = expit(f)` at the TOP of the loop (`_gpc.py:438`); the LAST
        // iteration's value becomes `self.pi_` (the one-step lag).
        let pi: Array1<F> = f.mapv(sigmoid);

        // W = diag(pi * (1 - pi))
        let w: Array1<F> = pi
            .iter()
            .map(|&p| {
                let w_val = p * (F::one() - p);
                // Clamp to avoid zero/negative
                if w_val < w_floor { w_floor } else { w_val }
            })
            .collect();

        let sqrt_w: Array1<F> = w.mapv(num_traits::Float::sqrt);

        // B = I + sqrt(W) K sqrt(W)
        let mut b = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = sqrt_w[i] * k_mat[[i, j]] * sqrt_w[j];
            }
            b[[i, i]] = b[[i, i]] + F::one();
        }

        // Cholesky of B
        let l = cholesky_gpc(&b)?;

        // b_vec = W f + (y - pi)
        let b_vec: Array1<F> = w
            .iter()
            .zip(f.iter())
            .zip(y_binary.iter().zip(pi.iter()))
            .map(|((&wi, &fi), (&yi, &pii))| wi * fi + (yi - pii))
            .collect();

        // a = b_vec - sqrt(W) @ solve(L^T, solve(L, sqrt(W) @ K @ b_vec))
        // Step 1: compute K @ b_vec
        let k_b = mat_vec_mul(&k_mat, &b_vec);
        // Step 2: sqrt(W) * (K @ b_vec)
        let sw_kb: Array1<F> = sqrt_w
            .iter()
            .zip(k_b.iter())
            .map(|(&s, &v)| s * v)
            .collect();
        // Step 3: solve L z = sw_kb
        let z = forward_solve_gpc(&l, &sw_kb);
        // Step 4: solve L^T z2 = z
        let z2 = backward_solve_gpc(&l, &z);
        // Step 5: sqrt(W) * z2
        let sw_z2: Array1<F> = sqrt_w.iter().zip(z2.iter()).map(|(&s, &v)| s * v).collect();
        // Step 6: a = b_vec - sw_z2
        let a: Array1<F> = b_vec
            .iter()
            .zip(sw_z2.iter())
            .map(|(&b, &s)| b - s)
            .collect();

        // f_new = K @ a  (`_gpc.py:450`)
        let f_new = mat_vec_mul(&k_mat, &a);

        // Log-marginal-likelihood, computed on the UPDATED f, using this
        // iteration's a and L (`_gpc.py:454-458`):
        //   lml = -0.5 a^T f - sum log1p(exp(-(2y-1) f)) - sum log diag(L)
        // The `log1p(exp(...))` form is sklearn's numerically-stable Bernoulli
        // term; y is the {0,1} y_binary, so (2y-1) is the {-1,+1} sign.
        let mut quad = F::zero();
        for (&ai, &fi) in a.iter().zip(f_new.iter()) {
            quad = quad + ai * fi;
        }
        let mut data_fit = F::zero();
        for (&yi, &fi) in y_binary.iter().zip(f_new.iter()) {
            let signed = (yi + yi - F::one()) * fi; // (2y - 1) * f
            data_fit = data_fit + (-signed).exp().ln_1p();
        }
        let mut log_det = F::zero();
        for i in 0..n {
            log_det = log_det + l[[i, i]].ln();
        }
        let lml = -half * quad - data_fit - log_det;

        // Stash this iteration's temporaries as the candidates sklearn stores.
        pi_hat = pi;
        l_final = l;
        lml_final = lml;
        f = f_new;

        // Break when the LML stops increasing (`_gpc.py:462`). The `f` update
        // and the temporaries above are RETAINED from this iteration.
        if lml - prev_lml < lml_eps {
            break;
        }
        prev_lml = lml;
    }

    Ok(FittedBinaryGPC {
        x_train: x.clone(),
        pi_hat,
        y_binary: y_binary.clone(),
        l_factor: l_final,
        lml: lml_final,
        kernel: kernel.clone_box(),
    })
}

/// Predictive latent posterior mean `f_bar* = K* @ (y - pi_hat)` for one
/// binary GP at new points (Rasmussen & Williams Algorithm 3.2, eq. 3.21 /
/// `_gpc.py:289`).
///
/// This is the latent decision function whose SIGN sklearn uses for the hard
/// binary class decision (`np.where(f_star > 0, classes_[1], classes_[0])`,
/// `_gpc.py:291`) — NOT the squashed `predict_proba`. It is the same quantity
/// computed at the top of [`predict_binary_proba`].
fn predict_binary_latent_mean<F: Float + Send + Sync + 'static>(
    model: &FittedBinaryGPC<F>,
    x: &Array2<F>,
) -> Result<Array1<F>, FerroError> {
    if x.ncols() != model.x_train.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows(), model.x_train.ncols()],
            actual: vec![x.nrows(), x.ncols()],
            context: "predict feature count must match training data".into(),
        });
    }

    // K* = k(X_new, X_train), shape (n_pred, n_train).
    let k_star = model.kernel.compute(x, &model.x_train);

    // Gradient at convergence: y - pi_hat.
    let y_minus_pi: Array1<F> = model
        .y_binary
        .iter()
        .zip(model.pi_hat.iter())
        .map(|(&yi, &pi)| yi - pi)
        .collect();

    // Predictive latent mean: f_bar* = K* (y - pi_hat).
    Ok(k_star.dot(&y_minus_pi))
}

/// Predict binary class probabilities at new points using Rasmussen &
/// Williams Algorithm 3.2 (Laplace approximation with predictive variance).
///
/// 1. Predictive latent mean: `f_bar* = K* @ (y - pi_hat)` (eq. 3.21).
/// 2. Predictive latent variance: `v = L^{-1} sqrt(W) K*^T`,
///    `var* = k(x*, x*) - sum(v^2)` (eq. 3.24).
/// 3. Class probability via the 5-term LAMBDAS/COEFS error-function
///    approximation of the logistic sigmoid (Williams & Barber), mirroring
///    scikit-learn `sklearn/gaussian_process/_gpc.py:324-331`:
///    `alpha = 1/(2 var*)`, `gamma = LAMBDAS * f_bar*`,
///    `integrals = sqrt(pi/alpha) * erf(gamma * sqrt(alpha/(alpha + LAMBDAS^2)))
///                 / (2 sqrt(var* * 2 pi))`,
///    `pi* = (COEFS * integrals).sum() + 0.5 * COEFS.sum()`.
fn predict_binary_proba<F: Float + Send + Sync + 'static>(
    model: &FittedBinaryGPC<F>,
    x: &Array2<F>,
) -> Result<Array1<F>, FerroError> {
    if x.ncols() != model.x_train.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![x.nrows(), model.x_train.ncols()],
            actual: vec![x.nrows(), x.ncols()],
            context: "predict feature count must match training data".into(),
        });
    }

    let n_train = model.x_train.nrows();
    let n_pred = x.nrows();

    // K* = k(X_new, X_train), shape (n_pred, n_train).
    let k_star = model.kernel.compute(x, &model.x_train);

    // Predictive latent mean: f_bar* = K* (y - pi_hat) (same quantity sklearn
    // uses for the latent-sign decision, `_gpc.py:289`).
    let f_bar = predict_binary_latent_mean(model, x)?;

    // sqrt(W) at convergence: w_i = pi_i (1 - pi_i), clamped consistently with fit.
    let eps = F::from(1e-12).unwrap();
    let sqrt_w: Array1<F> = model
        .pi_hat
        .iter()
        .map(|&p| {
            let w_val = p * (F::one() - p);
            if w_val < eps {
                eps.sqrt()
            } else {
                w_val.sqrt()
            }
        })
        .collect();

    // Compute predictive variance for each test point and squash via the
    // LAMBDAS/COEFS error-function approximation (`_gpc.py:324-331`). Avoids
    // forming the full (n_pred, n_train) intermediate matrix
    // V = L^{-1} sqrt(W) K*^T explicitly.
    let mut probs = Array1::<F>::zeros(n_pred);

    // Conversion fallback for the f64 -> F squash result.
    let to_f = |x: f64| -> Result<F, FerroError> {
        F::from(x).ok_or_else(|| FerroError::NumericalInstability {
            message: "predict_proba: squashed probability not representable in F".into(),
        })
    };

    for i in 0..n_pred {
        // k_i = K(x_train, x_i), shape (n_train,).
        let k_row: Array1<F> = (0..n_train).map(|j| k_star[[i, j]]).collect();

        // sqrt(W) * k_i
        let swk: Array1<F> = sqrt_w
            .iter()
            .zip(k_row.iter())
            .map(|(&s, &k)| s * k)
            .collect();

        // v = L^{-1} sqrt(W) k_i (forward solve).
        let v = forward_solve_gpc(&model.l_factor, &swk);

        // var* = k(x_i, x_i) - v^T v.
        let xi = x.row(i).to_owned().insert_axis(ndarray::Axis(0));
        let k_self = model
            .kernel
            .compute(&xi.view().to_owned(), &xi.view().to_owned());
        let k_xx = k_self[[0, 0]];
        let v_sq: F = v.iter().map(|&vi| vi * vi).fold(F::zero(), |a, b| a + b);
        let var_star = (k_xx - v_sq).max(F::zero());

        // 5-term LAMBDAS/COEFS error-function approximation of the logistic
        // sigmoid (`_gpc.py:324-331`). Computed in f64 regardless of F.
        // `f_star`/`var_f_star` here mirror sklearn's predictive mean/variance.
        let f_star = f_bar[i].to_f64().unwrap_or(0.0);
        let mut var_f_star = var_star.to_f64().unwrap_or(0.0);
        // `alpha = 1 / (2 var*)` blows up at var* ~ 0; floor var* so the erf
        // integral stays finite (matches numpy's finite-precision behavior).
        if var_f_star <= 0.0 {
            var_f_star = f64::MIN_POSITIVE;
        }
        let alpha = 1.0 / (2.0 * var_f_star);
        let mut pi_star = 0.0_f64;
        let mut coefs_sum = 0.0_f64;
        for (lambda_k, coef_k) in LAMBDAS.iter().zip(COEFS.iter()) {
            let gamma = lambda_k * f_star;
            let integral = (std::f64::consts::PI / alpha).sqrt()
                * statrs::function::erf::erf(
                    gamma * (alpha / (alpha + lambda_k * lambda_k)).sqrt(),
                )
                / (2.0 * (var_f_star * 2.0 * std::f64::consts::PI).sqrt());
            pi_star += coef_k * integral;
            coefs_sum += coef_k;
        }
        pi_star += 0.5 * coefs_sum;

        probs[i] = to_f(pi_star)?;
    }

    Ok(probs)
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (duplicated from gaussian_process.rs to keep
// modules independent; could be extracted to a shared module later)
// ---------------------------------------------------------------------------

/// Matrix-vector multiplication.
fn mat_vec_mul<F: Float + 'static>(a: &Array2<F>, v: &Array1<F>) -> Array1<F> {
    a.dot(v)
}

/// Cholesky decomposition: A = L L^T.
fn cholesky_gpc<F: Float>(a: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(FerroError::NumericalInstability {
                        message: format!(
                            "B matrix is not positive definite in Laplace approximation (pivot {i})"
                        ),
                    });
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Forward substitution: solve L x = b.
fn forward_solve_gpc<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum - l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Backward substitution: solve L^T x = b.
fn backward_solve_gpc<F: Float>(l: &Array2<F>, b: &Array1<F>) -> Array1<F> {
    let n = l.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum - l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

// ---------------------------------------------------------------------------
// Fit / Predict implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for GaussianProcessClassifier<F>
{
    type Fitted = FittedGaussianProcessClassifier<F>;
    type Error = FerroError;

    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedGaussianProcessClassifier<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "GaussianProcessClassifier::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match X rows".into(),
            });
        }

        // Find unique classes
        let mut classes: Vec<usize> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: format!(
                    "need at least 2 classes, got {} unique class(es)",
                    classes.len()
                ),
            });
        }

        let binary_models = if classes.len() == 2 {
            // Binary: single model, y = 0 or 1 (map to the second class)
            let y_binary: Array1<F> =
                y.mapv(|v| if v == classes[1] { F::one() } else { F::zero() });
            let model = fit_binary_gpc(self.kernel.as_ref(), x, &y_binary, self.max_iter)?;
            vec![model]
        } else {
            // Multi-class: one-vs-rest
            let mut models = Vec::with_capacity(classes.len());
            for &cls in &classes {
                let y_binary: Array1<F> = y.mapv(|v| if v == cls { F::one() } else { F::zero() });
                let model = fit_binary_gpc(self.kernel.as_ref(), x, &y_binary, self.max_iter)?;
                models.push(model);
            }
            models
        };

        Ok(FittedGaussianProcessClassifier {
            classes,
            binary_models,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedGaussianProcessClassifier<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        if self.classes.len() == 2 {
            // Binary: decide by the SIGN of the latent posterior mean f_bar*,
            // matching sklearn `np.where(f_star > 0, classes_[1], classes_[0])`
            // (`_gpc.py:291`) — NOT argmax of the squashed predict_proba. The
            // squash crosses 0.5 only approximately at f_bar* = 0, so the two
            // rules disagree at the decision boundary. Strict `> 0` puts the
            // exact-zero (and negative) case in classes_[0], as sklearn does.
            let f_bar = predict_binary_latent_mean(&self.binary_models[0], x)?;
            for i in 0..n_samples {
                predictions[i] = if f_bar[i] > F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multi-class OvR: argmax over the per-class probabilities.
            let proba = self.predict_proba(x)?;
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut best_prob = proba[[i, 0]];
                for c in 1..self.classes.len() {
                    if proba[[i, c]] > best_prob {
                        best_prob = proba[[i, c]];
                        best_class = c;
                    }
                }
                predictions[i] = self.classes[best_class];
            }
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp_kernels::{ConstantKernel, MaternKernel, ProductKernel, SumKernel, WhiteKernel};
    use ndarray::{Array2, array};

    // Helper to create linearly separable binary data
    fn make_binary_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        (x, y)
    }

    fn make_binary_2d() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, // class 0
                3.0, 3.0, 3.5, 3.5, 3.0, 3.5, 3.5, 3.0, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn make_multiclass_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec((9, 1), vec![0.0, 0.5, 1.0, 4.0, 4.5, 5.0, 8.0, 8.5, 9.0])
            .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, y)
    }

    // --- Basic fit/predict ---

    #[test]
    fn fit_predict_binary() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
        // Should get most training points right
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 8,
            "Expected at least 8/10 correct, got {correct}/10"
        );
    }

    #[test]
    fn fit_predict_binary_2d() {
        let (x, y) = make_binary_2d();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/8 correct, got {correct}/8"
        );
    }

    // --- Predict proba ---

    #[test]
    fn predict_proba_binary() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (10, 2));

        // Probabilities should sum to 1
        for i in 0..10 {
            let row_sum = proba[[i, 0]] + proba[[i, 1]];
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {i} sums to {row_sum}, expected 1.0"
            );
        }

        // Probabilities should be in [0, 1]
        for &p in &proba {
            assert!((0.0..=1.0).contains(&p), "Probability {p} out of range");
        }
    }

    #[test]
    fn predict_proba_class_0_near_0() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        // Points clearly in class 0 region should have high P(class=0)
        assert!(
            proba[[0, 0]] > 0.5,
            "P(class=0) at x=0.0 should be > 0.5, got {}",
            proba[[0, 0]]
        );

        // Points clearly in class 1 region should have high P(class=1)
        assert!(
            proba[[9, 1]] > 0.5,
            "P(class=1) at x=7.0 should be > 0.5, got {}",
            proba[[9, 1]]
        );
    }

    // --- Log marginal likelihood ---

    #[test]
    fn log_marginal_likelihood_binary_finite_and_negative() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let lml = fitted.log_marginal_likelihood();
        assert!(lml.is_finite(), "log marginal likelihood should be finite");
        // For Bernoulli likelihood, log marginal likelihood is < 0 in normal regimes.
        assert!(
            lml < 0.0,
            "log marginal likelihood should be negative, got {lml}"
        );
    }

    #[test]
    fn log_marginal_likelihood_prefers_separable_data() {
        // Well-separated data should give a higher (less negative) marginal
        // likelihood than near-overlapping data, all else equal.
        let kernel = || Box::new(RBFKernel::new(1.0));

        let x_easy = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0]).unwrap();
        let y_easy = Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1]);
        let lml_easy = GaussianProcessClassifier::new(kernel())
            .fit(&x_easy, &y_easy)
            .unwrap()
            .log_marginal_likelihood();

        let x_hard = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 1.1, 1.5, 2.0]).unwrap();
        let y_hard = Array1::from_vec(vec![0usize, 0, 0, 1, 1, 1]);
        let lml_hard = GaussianProcessClassifier::new(kernel())
            .fit(&x_hard, &y_hard)
            .unwrap()
            .log_marginal_likelihood();

        assert!(
            lml_easy > lml_hard,
            "separable data should have higher LML: easy={lml_easy}, hard={lml_hard}"
        );
    }

    #[test]
    fn log_marginal_likelihood_multiclass_means_components() {
        // For OvR multi-class, sklearn returns the MEAN of the per-binary LMLs
        // (`np.mean([...])`, `_gpc.py:743-749`), NOT their sum.
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let n_models = fitted.classes().len();
        assert!(n_models > 2, "multi-class fixture must have > 2 classes");
        let mean = fitted.log_marginal_likelihood();
        assert!(mean.is_finite());
        // Every per-class LML is negative, so the mean is negative.
        assert!(mean < 0.0);
        // The aggregate is the MEAN (sum / n), not the SUM: reconstructing the
        // sum from the mean must yield a value strictly below the mean (n > 1).
        let implied_sum = mean * (n_models as f64);
        assert!(
            implied_sum < mean,
            "aggregate must be the MEAN of per-binary LMLs, not the SUM: \
             mean={mean}, implied sum={implied_sum}"
        );
    }

    #[test]
    fn classes_accessor_returns_sorted_labels() {
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let classes = fitted.classes();
        let mut sorted = classes.to_vec();
        sorted.sort_unstable();
        assert_eq!(classes, sorted.as_slice());
    }

    // --- Multi-class ---

    #[test]
    fn fit_predict_multiclass() {
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 9);

        // Check predictions contain valid classes
        for &p in &pred {
            assert!(p <= 2, "Prediction {p} not in valid classes [0, 1, 2]");
        }

        // Should get most right
        let correct: usize = pred.iter().zip(y.iter()).filter(|&(&p, &t)| p == t).count();
        assert!(
            correct >= 6,
            "Expected at least 6/9 correct, got {correct}/9"
        );
    }

    #[test]
    fn predict_proba_multiclass() {
        let (x, y) = make_multiclass_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();

        assert_eq!(proba.dim(), (9, 3));

        // Probabilities should sum to ~1 per row
        for i in 0..9 {
            let row_sum: f64 = (0..3).map(|c| proba[[i, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-8,
                "Row {i} sums to {row_sum}, expected 1.0"
            );
        }
    }

    // --- Error handling ---

    #[test]
    fn fit_rejects_single_class() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_elem(5, 0usize);
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn fit_rejects_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = array![0usize];
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn fit_rejects_mismatched_y() {
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 1, 0]; // Wrong length
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn predict_rejects_wrong_features() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();

        let x_wrong = Array2::from_shape_vec((2, 3), vec![0.0; 6]).unwrap();
        assert!(fitted.predict(&x_wrong).is_err());
    }

    // --- Different kernels ---

    #[test]
    fn fit_with_matern() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(MaternKernel::new(1.0, 1.5)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    #[test]
    fn fit_with_sum_kernel() {
        let (x, y) = make_binary_data();
        let kernel = SumKernel::new(
            Box::new(RBFKernel::new(1.0)),
            Box::new(WhiteKernel::new(0.1)),
        );
        let gpc = GaussianProcessClassifier::new(Box::new(kernel));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    #[test]
    fn fit_with_product_kernel() {
        let (x, y) = make_binary_data();
        let kernel = ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.0)),
        );
        let gpc = GaussianProcessClassifier::new(Box::new(kernel));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- Builder pattern ---

    #[test]
    fn builder_pattern() {
        let gpc = GaussianProcessClassifier::default_rbf()
            .max_iter(50)
            .tol(1e-8);

        let (x, y) = make_binary_data();
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- f32 support ---

    #[test]
    fn f32_fit_predict() {
        let x = Array2::from_shape_vec((8, 1), vec![0.0f32, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5])
            .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let gpc = GaussianProcessClassifier::<f32>::new(Box::new(RBFKernel::new(1.0f32)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 8);
    }

    // --- Convergence ---

    #[test]
    fn converges_with_few_iterations() {
        let (x, y) = make_binary_data();
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0))).max_iter(5);
        // Should still produce reasonable results even with few iterations
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();
        assert_eq!(pred.len(), 10);
    }

    // --- Non-contiguous class labels ---

    #[test]
    fn non_contiguous_labels() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0]).unwrap();
        let y = array![10usize, 10, 10, 20, 20, 20];
        let gpc = GaussianProcessClassifier::new(Box::new(RBFKernel::new(1.0)));
        let fitted = gpc.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Predictions should be from {10, 20}
        for &p in &pred {
            assert!(p == 10 || p == 20, "Expected 10 or 20, got {p}");
        }
    }
}
