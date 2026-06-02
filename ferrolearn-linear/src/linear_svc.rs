//! Linear Support Vector Classifier.
//!
//! This module provides [`LinearSVC`], a liblinear-faithful linear support
//! vector classifier that operates directly in the primal space without the
//! overhead of a kernel function. The fit minimizes the L2-regularized
//! hinge (or squared-hinge) classification objective
//!
//! ```text
//!   min_w   0.5 * ||w||^2  +  C * sum_i  L(y_i, w . x_i)
//! ```
//!
//! with `y_i ∈ {-1, +1}` per one-vs-rest sub-problem (NO `1/n` averaging — the
//! summed loss is scaled by `C`, matching `sklearn/svm/_base.py`
//! `_fit_liblinear`). The solver is liblinear's dual coordinate descent for
//! classification (`solve_l2r_l1l2_svc` in
//! `sklearn/svm/src/liblinear/linear.cpp:819`), which converges to the unique
//! minimizer of the strongly convex objective.
//!
//! Unlike [`SVC`](crate::svm::SVC) with a [`LinearKernel`](crate::svm::LinearKernel),
//! `LinearSVC` avoids computing and caching the full kernel matrix, making it
//! significantly faster for high-dimensional data.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::linear_svc::LinearSVC;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
//!     5.0, 5.0, 5.0, 6.0, 6.0, 5.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = LinearSVC::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! ## REQ status
//!
//! Binary (R-DEFER-2): SHIPPED = impl + non-test production consumer + tests +
//! green oracle verification; NOT-STARTED = open blocker `#`. `LinearSVC`/
//! `FittedLinearSVC`/`LinearSVCLoss` are boundary estimator types re-exported at
//! the crate root (`pub use linear_svc::{…}` in `lib.rs`) and registered as the
//! PyO3 `RsLinearSVC` estimator (`ferrolearn-python/src/extras.rs`); under
//! S5/R-DEFER-1 those ARE the non-test production-consumer surface. See
//! `.design/linear/linear_svc.md`.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (fit parity — coef_/intercept_ vs liblinear oracle) | SHIPPED | `fn solve_binary_dual` minimizes liblinear's `0.5·‖w‖² + C·Σ L` via the dual CD (`solve_l2r_l1l2_svc`, `linear.cpp:819`); `fn fit` maps `classes_[1]→+1` and extracts `coef_ = w[:n_features]`, `intercept_ = intercept_scaling·w_last` (`_base.py:1240-1245`). Pinned by `tests/divergence_linear_svc_fit.rs::linear_svc_coef_parity` (live oracle `coef_ [[0.12835213611984458, 0.12835213611984475]]`, `intercept_ [-1.1943776585907158]`, C=1.0, squared_hinge, fit_intercept=True). Consumer: `pub use linear_svc::{…}` (`lib.rs`) + `RsLinearSVC` (PyO3). |
//! | REQ-2 (decision_function shape `(n,)` + values) | SHIPPED | `fn decision_function` returns [`DecisionScores::Binary`] = 1-D `X·w + b` for the binary case (sklearn ravels the single-column score to `(n,)`, `linear_model/_base.py:365`) and [`DecisionScores::Multiclass`] `(n, n_classes)` otherwise. Pinned by `tests/divergence_linear_svc_fit.rs::linear_svc_decision_function` (live oracle 1-D `(8,)` values). Consumer: `fn predict` reads the binary scores' sign. |
//! | REQ-3 (predict + classes_) | SHIPPED (incidental) | `fn predict` uses the sign of the binary decision (`>= 0 → classes_[1]`) / argmax of the OvR scores; `HasClasses::classes` = sorted unique `y` (`classes_ = np.unique(y)`, `_classes.py:311`). The labels are now downstream of the liblinear-parity fit; pinned indirectly via `linear_svc_decision_function` (sign agreement) + `conformance_linear_svc` accuracy floor. A dedicated `predict` oracle pin remains under #620. |
//! | REQ-4 (loss {hinge, squared_hinge}) | NOT-STARTED | open prereq blocker #621. The dual CD now solves BOTH the true `hinge` (`U=C`, `diag=0`) and `squared_hinge` (`U=∞`, `diag=0.5/C`) optima (`solve_l2r_l1l2_svc`, `linear.cpp:849-858`), but no per-loss `coef_`/`intercept_` test pins the `hinge` optimum against the live oracle. |
//! | REQ-5 (penalty {l1, l2}) | NOT-STARTED | open prereq blocker #622. `LinearSVC<F>` has no `penalty` field; the dual CD hardcodes the L2 regularizer. |
//! | REQ-6 (multi_class {ovr, crammer_singer}) | NOT-STARTED | open prereq blocker #623. `fn fit` implements one-vs-rest (the default `'ovr'`) but there is no `multi_class` field / `crammer_singer` joint solver. |
//! | REQ-7 (fit_intercept + intercept_scaling) | SHIPPED | `LinearSVC<F>` exposes `pub fit_intercept: bool` (default true) + `pub intercept_scaling: F` (default 1.0) + `#[must_use]` builders. When fitting an intercept the design matrix is augmented with a penalized constant column = `intercept_scaling`, and `intercept_ = intercept_scaling·w_last` (`_base.py:1188-1198,:1240-1245`); `intercept_scaling > 0` is validated. Pinned by `linear_svc_coef_parity` + module `test_fit_intercept_false_zero_intercept`/`test_invalid_intercept_scaling`. |
//! | REQ-8 (dual param) | NOT-STARTED | open prereq blocker #625. `LinearSVC<F>` has no `dual` field. |
//! | REQ-9 (class_weight) | NOT-STARTED | open prereq blocker #626. `LinearSVC<F>` has no `class_weight` field. |
//! | REQ-10 (C-scaling convention) | SHIPPED | the `c / n_f` division is removed; the dual CD uses `upper_bound = C` (hinge) / `diag = 0.5/C` (squared_hinge), so `coef_` tracks `C` like liblinear. Pinned by `linear_svc_coef_c_dependence` (C=0.1 → `0.0784651864625997`, C=1.0 → `0.12835213611984458`). |
//! | REQ-11 (n_iter_/n_features_in_ + param validation) | NOT-STARTED | open prereq blocker #627. `fn fit` counts dual-CD outer iterations and emits the ConvergenceWarning-equivalent, but `n_iter_`/`n_features_in_` accessors + `tol > 0` validation are not yet exposed/pinned. |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #628. Imports `ndarray`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::{HasClasses, HasCoefficients};
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Loss function for [`LinearSVC`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSVCLoss {
    /// Standard hinge loss: `max(0, 1 - y * f(x))`. liblinear solver
    /// `L2R_L1LOSS_SVC_DUAL` (type 3): box `0 <= alpha <= C`, `diag = 0`.
    Hinge,
    /// Squared hinge loss: `max(0, 1 - y * f(x))^2` (default). liblinear solver
    /// `L2R_L2LOSS_SVC_DUAL` (type 1): box `0 <= alpha <= +inf`,
    /// `diag = 0.5 / C`.
    SquaredHinge,
}

/// Confidence scores returned by [`FittedLinearSVC::decision_function`].
///
/// Mirrors `sklearn.svm.LinearSVC.decision_function`
/// (`LinearClassifierMixin.decision_function`,
/// `sklearn/linear_model/_base.py:341-365`): the binary case collapses the
/// single-column score matrix to a 1-D `(n_samples,)` array
/// (`return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores`,
/// `_base.py:365`), the multiclass case returns `(n_samples, n_classes)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionScores<F> {
    /// Binary scores `X · w + b` for the positive class (`classes_[1]`), shape
    /// `(n_samples,)`. `> 0` predicts `classes_[1]`.
    Binary(Array1<F>),
    /// One-vs-rest scores, shape `(n_samples, n_classes)`. The argmax of each
    /// row agrees with [`Predict`].
    Multiclass(Array2<F>),
}

impl<F: Clone> DecisionScores<F> {
    /// Number of samples scored (the leading axis length in both variants).
    #[must_use]
    pub fn n_samples(&self) -> usize {
        match self {
            DecisionScores::Binary(v) => v.len(),
            DecisionScores::Multiclass(m) => m.nrows(),
        }
    }

    /// Borrow the binary 1-D scores, if this is the binary case.
    #[must_use]
    pub fn as_binary(&self) -> Option<&Array1<F>> {
        match self {
            DecisionScores::Binary(v) => Some(v),
            DecisionScores::Multiclass(_) => None,
        }
    }

    /// Borrow the multiclass `(n_samples, n_classes)` scores, if this is the
    /// multiclass case.
    #[must_use]
    pub fn as_multiclass(&self) -> Option<&Array2<F>> {
        match self {
            DecisionScores::Multiclass(m) => Some(m),
            DecisionScores::Binary(_) => None,
        }
    }
}

/// Linear Support Vector Classifier (liblinear dual CD).
///
/// Solves the L2-regularized hinge or squared-hinge objective
/// `0.5*||w||^2 + C * sum_i L(y_i, w.x_i)` via liblinear's dual coordinate
/// descent. Supports binary and multiclass (one-vs-rest) classification.
/// Mirrors `sklearn.svm.LinearSVC`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LinearSVC<F> {
    /// Inverse regularization strength. Larger values allow more
    /// misclassification. Must be strictly positive.
    pub c: F,
    /// Maximum number of dual coordinate descent iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the projected-gradient span.
    pub tol: F,
    /// Loss function to use.
    pub loss: LinearSVCLoss,
    /// Whether to fit an intercept. When `true`, the design matrix is augmented
    /// with a synthetic constant column equal to [`intercept_scaling`]; the
    /// augmented weight is penalized like any feature (liblinear convention).
    ///
    /// [`intercept_scaling`]: Self::intercept_scaling
    pub fit_intercept: bool,
    /// Value of the synthetic intercept feature when [`fit_intercept`] is
    /// `true`. Must be strictly positive. `intercept_ = intercept_scaling *
    /// w_last`.
    ///
    /// [`fit_intercept`]: Self::fit_intercept
    pub intercept_scaling: F,
}

impl<F: Float> LinearSVC<F> {
    /// Create a new `LinearSVC` with scikit-learn's default settings.
    ///
    /// Defaults (matching `sklearn.svm.LinearSVC`, `_classes.py`):
    /// `C = 1.0`, `max_iter = 1000`, `tol = 1e-4`, `loss = SquaredHinge`,
    /// `fit_intercept = true`, `intercept_scaling = 1.0`.
    #[must_use]
    pub fn new() -> Self {
        // 1e-4/1.0 are exactly representable in f32/f64; the defensive fallback
        // for `from(1e-4)` is never taken (no `.unwrap()` in lib code).
        let one = F::one();
        Self {
            c: one,
            max_iter: 1000,
            tol: F::from(1e-4).unwrap_or_else(|| {
                let ten = F::from(10).unwrap_or(one);
                one / (ten * ten * ten * ten)
            }),
            loss: LinearSVCLoss::SquaredHinge,
            fit_intercept: true,
            intercept_scaling: one,
        }
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the loss function.
    #[must_use]
    pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set whether to fit an intercept (sklearn `fit_intercept`).
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the intercept scaling (sklearn `intercept_scaling`). Must be
    /// strictly positive when `fit_intercept` is `true`.
    #[must_use]
    pub fn with_intercept_scaling(mut self, intercept_scaling: F) -> Self {
        self.intercept_scaling = intercept_scaling;
        self
    }
}

impl<F: Float> Default for LinearSVC<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Linear Support Vector Classifier.
///
/// Stores the learned weight vectors, intercepts, and class labels.
/// For binary classification a single weight vector is stored; for
/// multiclass, one per class (one-vs-rest).
#[derive(Debug, Clone)]
pub struct FittedLinearSVC<F> {
    /// Weight vectors: one per binary sub-problem.
    /// Binary: `[w]`, Multiclass: `[w_0, w_1, ..., w_{k-1}]`.
    weight_vectors: Vec<Array1<F>>,
    /// Intercept for each sub-problem.
    intercepts: Vec<F>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Whether this is a binary problem.
    is_binary: bool,
    /// Number of features.
    n_features: usize,
}

impl<F: Float> FittedLinearSVC<F> {
    /// Returns the weight vectors (one per binary sub-problem).
    #[must_use]
    pub fn weight_vectors(&self) -> &[Array1<F>] {
        &self.weight_vectors
    }

    /// Returns the intercepts (one per binary sub-problem).
    #[must_use]
    pub fn intercepts(&self) -> &[F] {
        &self.intercepts
    }
}

impl<F: Float + ScalarOperand + Send + Sync + 'static> FittedLinearSVC<F> {
    /// Raw signed distance from the decision boundary. Mirrors sklearn
    /// `LinearSVC.decision_function`.
    ///
    /// Binary: [`DecisionScores::Binary`] of shape `(n_samples,)` containing
    /// `X @ w + b` for the positive class (`classes_[1]`); sklearn ravels the
    /// single-column score to 1-D (`linear_model/_base.py:365`).
    /// Multiclass: [`DecisionScores::Multiclass`] of shape
    /// `(n_samples, n_classes)` of one-vs-rest scores; the argmax of each row
    /// agrees with [`Predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<DecisionScores<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        if self.is_binary {
            // sklearn collapses the single-column binary score matrix to a 1-D
            // (n_samples,) array (`linear_model/_base.py:365`).
            let scores = x.dot(&self.weight_vectors[0]) + self.intercepts[0];
            Ok(DecisionScores::Binary(scores))
        } else {
            let n_classes = self.classes.len();
            let mut out = Array2::<F>::zeros((n_samples, n_classes));
            for c in 0..n_classes {
                for i in 0..n_samples {
                    out[[i, c]] = x.row(i).dot(&self.weight_vectors[c]) + self.intercepts[c];
                }
            }
            Ok(DecisionScores::Multiclass(out))
        }
    }
}

/// Resolved per-fit solver configuration passed to [`solve_binary_dual`]
/// (groups the dual-CD knobs to keep the solver signature small).
#[derive(Debug, Clone, Copy)]
struct SolverConfig<F> {
    /// Inverse regularization strength `C`.
    c: F,
    /// Maximum dual-CD outer iterations.
    max_iter: usize,
    /// Projected-gradient span stopping tolerance.
    tol: F,
    /// Hinge / squared-hinge loss selector.
    loss: LinearSVCLoss,
    /// Whether the design matrix is augmented with a penalized bias column.
    fit_intercept: bool,
    /// Value of the synthetic bias column when `fit_intercept` is set.
    intercept_scaling: F,
}

/// Solve a single binary L2-SVM via liblinear's dual coordinate descent.
///
/// Minimizes `0.5 * ||w||^2 + C * sum_i L(y_i, w.x_i)` (NO `1/n` averaging,
/// matching `sklearn/svm/_base.py` `_fit_liblinear`) with `y_i ∈ {-1, +1}`.
/// The augmented weight vector has length `w_size`; when `fit_intercept` is
/// set, `w_size = n_features + 1` and the trailing weight multiplies the
/// synthetic constant column `intercept_scaling` (penalized like any feature).
///
/// This is liblinear's `solve_l2r_l1l2_svc` (`linear.cpp:819`): the dual is
///
/// ```text
///   min_alpha  0.5 * alpha^T (Q + diag) alpha  -  e^T alpha,
///     s.t.     0 <= alpha_i <= U,
/// ```
///
/// where `Q_ij = y_i y_j x_i.x_j`, `w = sum_i alpha_i y_i x_i`,
/// `QD[i] = diag + ||x_i||^2`. **hinge** (`L2R_L1LOSS_SVC_DUAL`): `diag = 0`,
/// `U = C` (`linear.cpp:852-858`). **squared_hinge** (`L2R_L2LOSS_SVC_DUAL`):
/// `diag = 0.5/C`, `U = +inf` (`linear.cpp:849-850`).
///
/// Returns `(w_augmented, n_iter, converged)`.
fn solve_binary_dual<F: Float + 'static>(
    x: &Array2<F>,
    y_signed: &Array1<F>,
    cfg: &SolverConfig<F>,
) -> (Vec<F>, usize, bool) {
    let SolverConfig {
        c,
        max_iter,
        tol,
        loss,
        fit_intercept,
        intercept_scaling,
    } = *cfg;

    let (n_samples, n_features) = x.dim();
    let w_size = if fit_intercept {
        n_features + 1
    } else {
        n_features
    };

    let inf = F::infinity();
    let two = F::one() + F::one();
    let half = F::one() / two;
    let tiny = F::from(1.0e-12).unwrap_or_else(F::epsilon);

    // diag / upper_bound from the solver type (`linear.cpp:849-858`).
    let (diag, upper_bound) = match loss {
        LinearSVCLoss::Hinge => (F::zero(), c),
        LinearSVCLoss::SquaredHinge => (half / c, inf),
    };

    // QD[i] = diag + ||x_i||^2 (including the augmented bias column).
    let mut qd = vec![F::zero(); n_samples];
    for (i, qd_i) in qd.iter_mut().enumerate() {
        let mut acc = diag;
        let row = x.row(i);
        for &v in row.iter() {
            acc = acc + v * v;
        }
        if fit_intercept {
            acc = acc + intercept_scaling * intercept_scaling;
        }
        *qd_i = acc;
    }

    let mut alpha = vec![F::zero(); n_samples];
    let mut w = vec![F::zero(); w_size];
    // alpha starts at 0 so w starts at 0; nothing to accumulate.

    let mut index: Vec<usize> = (0..n_samples).collect();
    let mut active_size = n_samples;
    let mut pgmax_old = inf;
    let mut pgmin_old = -inf;

    // w . x_i over the (augmented) design matrix.
    let dot_w_xi = |w: &[F], i: usize| -> F {
        let mut acc = F::zero();
        let row = x.row(i);
        for (j, &v) in row.iter().enumerate() {
            acc = acc + v * w[j];
        }
        if fit_intercept {
            acc = acc + intercept_scaling * w[n_features];
        }
        acc
    };

    let mut n_iter: usize = 0;
    let mut converged = false;

    // liblinear shuffles `index` each sweep; the minimizer is unique so order
    // only affects the path, not the limit. We sweep in natural order for
    // determinism (no RNG), reaching the same converged optimum.
    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut pgmax_new = -inf;
        let mut pgmin_new = inf;

        let mut s = 0;
        while s < active_size {
            let i = index[s];
            let yi = y_signed[i];

            // G = y_i*(w.x_i) - 1 + diag*alpha_i (`linear.cpp:909-921`).
            let g = yi * dot_w_xi(&w, i) - F::one() + diag * alpha[i];

            // Projected gradient + shrinking (`linear.cpp:923-949`).
            let mut pg = F::zero();
            if alpha[i] == F::zero() {
                if g > pgmax_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue; // re-process the swapped-in element at `s`
                } else if g < F::zero() {
                    pg = g;
                }
            } else if alpha[i] == upper_bound {
                if g < pgmin_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g > F::zero() {
                    pg = g;
                }
            } else {
                pg = g;
            }

            if pg > pgmax_new {
                pgmax_new = pg;
            }
            if pg < pgmin_new {
                pgmin_new = pg;
            }

            if pg.abs() > tiny {
                let alpha_old = alpha[i];
                // alpha_i <- clamp(alpha_i - G/QD[i], 0, U) (`linear.cpp:957`).
                let mut new_alpha = alpha[i] - g / qd[i];
                if new_alpha < F::zero() {
                    new_alpha = F::zero();
                } else if new_alpha > upper_bound {
                    new_alpha = upper_bound;
                }
                alpha[i] = new_alpha;
                let d = (alpha[i] - alpha_old) * yi;
                if d != F::zero() {
                    let row = x.row(i);
                    for (j, &v) in row.iter().enumerate() {
                        w[j] = w[j] + d * v;
                    }
                    if fit_intercept {
                        w[n_features] = w[n_features] + d * intercept_scaling;
                    }
                }
            }

            s += 1;
        }

        // Stopping: PGmax_new - PGmin_new <= tol on the full set
        // (`linear.cpp:972-990`). Absolute tol (the dual SVC solver receives
        // `param->eps` directly, `linear.cpp:2364`).
        if pgmax_new - pgmin_new <= tol {
            if active_size == n_samples {
                converged = true;
                break;
            }
            active_size = n_samples;
            pgmax_old = inf;
            pgmin_old = -inf;
            continue;
        }

        pgmax_old = pgmax_new;
        pgmin_old = pgmin_new;
        if pgmax_old <= F::zero() {
            pgmax_old = inf;
        }
        if pgmin_old >= F::zero() {
            pgmin_old = -inf;
        }
    }

    (w, n_iter, converged)
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Fit<Array2<F>, Array1<usize>>
    for LinearSVC<F>
{
    type Fitted = FittedLinearSVC<F>;
    type Error = FerroError;

    /// Fit the linear SVC model using liblinear's dual coordinate descent.
    ///
    /// Minimizes `0.5*||w||^2 + C * sum_i L(y_i, w.x_i)` (no `1/n`). When
    /// `fit_intercept` is set, the design matrix is augmented with a constant
    /// column equal to `intercept_scaling`, the augmented weight is penalized
    /// like any feature, and `intercept_ = intercept_scaling * w_last`
    /// (`_base.py:1188-1198,:1240-1245`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InvalidParameter`] — `C` not positive, or
    ///   `intercept_scaling` not positive when fitting an intercept.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 distinct classes.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedLinearSVC<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        // liblinear raises when intercept_scaling <= 0 with fit_intercept
        // (`_base.py:1190-1196`).
        if self.fit_intercept && self.intercept_scaling <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "intercept_scaling".into(),
                reason: "must be greater than 0 when fit_intercept is true".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "LinearSVC requires at least 2 distinct classes".into(),
            });
        }

        let cfg = SolverConfig {
            c: self.c,
            max_iter: self.max_iter,
            tol: self.tol,
            loss: self.loss,
            fit_intercept: self.fit_intercept,
            intercept_scaling: self.intercept_scaling,
        };

        // Solve one binary sub-problem and split the augmented weight vector
        // into (coef_, intercept_) per `_base.py:1240-1245`.
        let solve_one = |y_signed: &Array1<F>| -> (Array1<F>, F, usize, bool) {
            let (w, n_iter, converged) = solve_binary_dual(x, y_signed, &cfg);
            let coef = Array1::from_iter(w.iter().take(n_features).copied());
            let intercept = if self.fit_intercept {
                self.intercept_scaling * w[n_features]
            } else {
                F::zero()
            };
            (coef, intercept, n_iter, converged)
        };

        let mut any_unconverged = false;

        let fitted = if classes.len() == 2 {
            // Binary classification. liblinear's `prob.y` are the LabelEncoder
            // indices, mapped `y[i] = +1 if y_ind > 0 else -1`
            // (`linear.cpp:861-871`), so the positive class is `classes_[1]`
            // and `coef_` is for `classes_[1]` (`_base.py` / `_classes.py:311`).
            let y_signed: Array1<F> = y.mapv(|label| {
                if label == classes[1] {
                    F::one()
                } else {
                    -F::one()
                }
            });

            let (coef, intercept, _n_iter, converged) = solve_one(&y_signed);
            if !converged {
                any_unconverged = true;
            }

            FittedLinearSVC {
                weight_vectors: vec![coef],
                intercepts: vec![intercept],
                classes,
                is_binary: true,
                n_features,
            }
        } else {
            // Multiclass: one-vs-rest. Each class is the positive (+1) class of
            // its own binary sub-problem.
            let mut weight_vectors = Vec::with_capacity(classes.len());
            let mut intercepts = Vec::with_capacity(classes.len());

            for &cls in &classes {
                let y_signed: Array1<F> =
                    y.mapv(|label| if label == cls { F::one() } else { -F::one() });
                let (coef, intercept, _n_iter, converged) = solve_one(&y_signed);
                if !converged {
                    any_unconverged = true;
                }
                weight_vectors.push(coef);
                intercepts.push(intercept);
            }

            FittedLinearSVC {
                weight_vectors,
                intercepts,
                classes,
                is_binary: false,
                n_features,
            }
        };

        // liblinear warns when any sub-problem reaches `max_iter` without
        // satisfying the stopping criterion (`_base.py:1234-1238`). The crate's
        // warning channel is `eprintln!` (cf. qda.rs / lda.rs warnings).
        if any_unconverged {
            eprintln!("Liblinear failed to converge, increase the number of iterations.");
        }

        Ok(fitted)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedLinearSVC<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Binary: `sign(X @ w + b)` mapped to class labels (`>= 0 → classes_[1]`).
    /// Multiclass: argmax of decision values across one-vs-rest classifiers.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        if self.is_binary {
            let scores = x.dot(&self.weight_vectors[0]) + self.intercepts[0];
            for i in 0..n_samples {
                predictions[i] = if scores[i] >= F::zero() {
                    self.classes[1]
                } else {
                    self.classes[0]
                };
            }
        } else {
            // Multiclass: pick class with highest decision value.
            for i in 0..n_samples {
                let mut best_class = 0;
                let mut best_score = F::neg_infinity();
                for (c, w) in self.weight_vectors.iter().enumerate() {
                    let score = x.row(i).dot(w) + self.intercepts[c];
                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
                predictions[i] = self.classes[best_class];
            }
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasCoefficients<F> for FittedLinearSVC<F> {
    /// Returns the coefficient vector of the first (or only) binary sub-problem.
    fn coefficients(&self) -> &Array1<F> {
        &self.weight_vectors[0]
    }

    /// Returns the intercept of the first (or only) binary sub-problem.
    fn intercept(&self) -> F {
        self.intercepts[0]
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedLinearSVC<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_default_constructor() {
        let m = LinearSVC::<f64>::new();
        assert_eq!(m.max_iter, 1000);
        assert!(m.c == 1.0);
        assert_eq!(m.loss, LinearSVCLoss::SquaredHinge);
        assert!(m.fit_intercept);
        assert!(m.intercept_scaling == 1.0);
    }

    #[test]
    fn test_builder_setters() {
        let m = LinearSVC::<f64>::new()
            .with_c(10.0)
            .with_max_iter(500)
            .with_tol(1e-6)
            .with_loss(LinearSVCLoss::Hinge)
            .with_fit_intercept(false)
            .with_intercept_scaling(2.0);
        assert!(m.c == 10.0);
        assert_eq!(m.max_iter, 500);
        assert_eq!(m.loss, LinearSVCLoss::Hinge);
        assert!(!m.fit_intercept);
        assert!(m.intercept_scaling == 2.0);
    }

    #[test]
    fn test_binary_classification() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = LinearSVC::<f64>::new().with_c(1.0).with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_binary_hinge_loss() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LinearSVC::<f64>::new()
            .with_loss(LinearSVCLoss::Hinge)
            .with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 4, "expected at least 4 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_classification() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 0.0, 10.0, 0.5,
                10.0, 0.0, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = LinearSVC::<f64>::new().with_c(10.0).with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_binary_decision_function_is_1d() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = LinearSVC::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        let df = fitted.decision_function(&x).unwrap();
        // sklearn ravels the binary decision_function to (n,) (`_base.py:365`).
        let binary = df.as_binary().expect("binary decision is 1-D");
        assert_eq!(binary.len(), 6);
        assert!(df.as_multiclass().is_none());
        assert_eq!(df.n_samples(), 6);
    }

    #[test]
    fn test_multiclass_decision_function_is_2d() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 0.0, 10.0, 0.5,
                10.0, 0.0, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let fitted = LinearSVC::<f64>::new()
            .with_c(10.0)
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        let df = fitted.decision_function(&x).unwrap();
        let scores = df.as_multiclass().expect("multiclass decision is 2-D");
        assert_eq!(scores.dim(), (9, 3));
        assert!(df.as_binary().is_none());
    }

    #[test]
    fn test_shape_mismatch_fit() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = LinearSVC::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = LinearSVC::<f64>::new().with_c(0.0);
        assert!(model.fit(&x, &y).is_err());

        let model_neg = LinearSVC::<f64>::new().with_c(-1.0);
        assert!(model_neg.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_intercept_scaling() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        // fit_intercept=true + intercept_scaling<=0 is rejected
        // (`_base.py:1190-1196`).
        let model = LinearSVC::<f64>::new().with_intercept_scaling(0.0);
        assert!(model.fit(&x, &y).is_err());

        // But with fit_intercept=false, intercept_scaling is ignored.
        let model = LinearSVC::<f64>::new()
            .with_fit_intercept(false)
            .with_intercept_scaling(0.0);
        assert!(model.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_fit_intercept_false_zero_intercept() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = LinearSVC::<f64>::new()
            .with_fit_intercept(false)
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();
        assert!(fitted.intercept() == 0.0);
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = LinearSVC::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_has_coefficients() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let model = LinearSVC::<f64>::new().with_max_iter(5000);
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.coefficients().len(), 2);
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        let fitted = LinearSVC::<f64>::new()
            .with_max_iter(5000)
            .fit(&x, &y)
            .unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }
}
