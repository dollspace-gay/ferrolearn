//! Quadratic Discriminant Analysis (QDA).
//!
//! This module provides [`QDA`], a classifier that models each class with its
//! own covariance matrix, yielding quadratic decision boundaries. Unlike
//! [`LDA`](crate::lda::LDA), which assumes a shared covariance matrix, QDA
//! fits a separate covariance per class.
//!
//! # Algorithm
//!
//! Mirrors scikit-learn's per-class SVD formulation
//! (`sklearn/discriminant_analysis.py:940-976`). For each class `k`:
//! 1. Compute the class mean `mu_k` and center the samples: `Xkc = Xk - mu_k`.
//! 2. Take the thin SVD `Xkc = U·diag(S)·Vtᵀ` (`full_matrices=False`).
//! 3. Scalings: `scalings_k = (1 - reg) * (S² / (n_k - 1)) + reg`; rotations:
//!    `rotations_k = Vtᵀ`. (Algebraically identical to the eigendecomposition of
//!    the regularized covariance `(1 - reg) * Sigma_k + reg * I`.)
//! 4. Log-posterior in the rotated/scaled frame:
//!    `delta_k(x) = -0.5 * (norm2_k(x) + sum(log scalings_k)) + log(prior_k)`,
//!    where `norm2_k(x) = || (x - mu_k) · (rotations_k · scalings_k^(-0.5)) ||²`.
//! 5. Predict the class with the largest `delta_k`.
//!
//! Unlike the previous Cholesky covariance-inversion path, the SVD formulation
//! does **not** error on a rank-deficient (collinear) class: a zero singular
//! value yields a zero scaling, and the degenerate `inf`/`NaN` arithmetic
//! propagates exactly as in scikit-learn (which emits a "Variables are
//! collinear" warning and still predicts), reproducing the upstream behavior.
//!
//! ## REQ status (per `.design/linear/qda.md`, mirrors `sklearn/discriminant_analysis.py` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (fit + decision_function parity) | SHIPPED | `fn fit` per-class thin SVD (`fn svd_s_vt` → `ferray::linalg::svd`) + `fn raw_decision` SVD log-posterior `-½(‖(x-μ)·R·S^(-½)‖² + Σ log S) + log π` (`discriminant_analysis.py:962-976`). Consumer: `Predict for FittedQDA` + `RsQDA::predict`. Test `qda_decision_function_multiclass` <1e-6. #575. |
//! | REQ-2 (predict argmax) | SHIPPED | `Predict::predict` argmax of `raw_decision` (numpy first-NaN-wins replicated). Test `qda_predict_multiclass`. #576. |
//! | REQ-3 (predict_proba softmax) | SHIPPED | `FittedQDA::predict_proba`. Test `qda_predict_proba_multiclass` <1e-6. #577. |
//! | REQ-5 (reg_param) | SHIPPED | `with_reg_param`; `scalings = (1-reg)·(S²/(n_k-1)) + reg` (== sklearn singular-value blend). Test `qda_reg_param`. #579. |
//! | REQ-8 (covariance n_k-1 normalization) | SHIPPED | `S²/(n_k-1)` in `fn fit` (`discriminant_analysis.py:948`). |
//! | REQ-10 (tol + rank-deficient SVD, no error) | SHIPPED | `with_tol` (default 1e-4); `rank = Σ(S>tol)`, collinearity warning, no error (`:945-947`). Test `qda_rank_deficient_class` (`predict == [0;8]`). #583. |
//! | REQ-11 (scalings_/rotations_/means_/priors_ attrs) | SHIPPED | `FittedQDA::{scalings,rotations,means,priors}` (`:948-954`,`:921-924`). Test `qda_scalings_rotations`. #584. |
//! | REQ-6 (provided priors) | SHIPPED | `QDA::priors` field + `with_priors` builder; `fn fit` resolves `priors_` (`None`→empirical `n_k/n`, `Some(p)`→verbatim, no renorm/sign check — `discriminant_analysis.py:341,351,359,921-924`) and `log_prior = priors[k].ln()` enters `fn raw_decision` as `+ log π_k` (`:976`). R-DEV-4 deviation: length-mismatch → `ShapeMismatch` (sklearn silently mis-indexes). Consumer: `fn fit` reads `self.priors`; `FittedQDA::priors` exposes the resolved priors. Test `qda_provided_priors` + `qda_priors_length_mismatch` <1e-6. #580. |
//! | REQ-4 (predict_log_proba pin + consumer) | SHIPPED | `FittedQDA::predict_log_proba` = `log_proba(predict_proba(X))` (elementwise `log`, `discriminant_analysis.py:1058-1059` `return np.log(probas_)`). Consumer: `Predict for FittedQDA` shares `predict_proba`; test `qda_predict_log_proba` (overlapping 3-class, finite log-probas) <1e-6 vs live `QuadraticDiscriminantAnalysis().predict_log_proba`. #578. |
//! | REQ-7 (binary decision_function shape `(n,)`) | SHIPPED | #581 (binding-ABI layer, cf. logistic #454): the lib `FittedQDA::decision_function` (qda.rs:411) returns the raw `(n, n_classes)` log-posterior; the binding applies sklearn's binary `col1-col0` collapse OUTSIDE the lib. A SECOND `#[pymethods] impl RsQDA` block (pyo3 `multiple-pymethods`, mirroring the discrete-NB attr pattern) exposes `RsQDA::decision_function` (`ferrolearn-python/src/extras.rs`) marshalling the stored `FittedQDA` → `(n, n_classes)` `PyArray2<f64>` (not-fitted → `PyRuntimeError`, FerroError → `PyValueError`, no panic); `_extras.py::QuadraticDiscriminantAnalysis.decision_function` returns `raw[:, 1] - raw[:, 0]` (shape `(n,)`) for binary (`discriminant_analysis.py:1000-1001`), else the raw `(n, n_classes)`. Non-test consumer: `ferrolearn.QuadraticDiscriminantAnalysis.decision_function`. Verification (model B): `ferrolearn-python/tests/divergence_qda_py.py` (9 pass, live sklearn 1.5.2 oracle, atol 1e-6) — binary `(n,)` shape+values+sign (positive ⇒ `classes_[1]`), multiclass `(n, n_classes)` shape+values, predict unaffected, argmax==predict, not-fitted→NotFittedError. |
//! | REQ-9 (store_covariance + covariance_) | SHIPPED | `QDA::store_covariance` field + `with_store_covariance` builder (default `false`, `discriminant_analysis.py:353,361`); `fn fit` reconstructs per-class `Rₖ·diag(scalingsₖ)·Rₖᵀ` from the REGULARIZED scalings (`cov.append(np.dot(S2 * Vt.T, Vt))`, `:951-952`) only when set (`:955-956`). Consumer: `fn fit` reads `self.store_covariance`; `FittedQDA::covariance` exposes the per-class covariances (`None` when unset). Test `qda_store_covariance` <1e-9 vs live `QuadraticDiscriminantAnalysis(store_covariance=True).covariance_` (+ `with_store_covariance(false)` → `None`). #582. |
//! | REQ-12 (ferray array-type substrate) | NOT-STARTED | #585 (per-class SVD already on `ferray::linalg::svd`; owned array still `ndarray`, crate-wide-deferred cf. ridge #391). |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::qda::QDA;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 1.0, 1.5, 1.2, 1.2, 0.8, 5.0, 5.0, 5.5, 4.8, 4.8, 5.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let qda = QDA::<f64>::new();
//! let fitted = qda.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferray::linalg::{LinalgFloat, svd};
use ferray::{Array as FerrayArray, Ix2 as FerrayIx2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

/// Quadratic Discriminant Analysis configuration.
///
/// Holds hyperparameters. Calling [`Fit::fit`] computes per-class means
/// and covariance matrices and returns a [`FittedQDA`].
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct QDA<F> {
    /// Regularization parameter for covariance matrices.
    ///
    /// Blends each class covariance toward the identity:
    /// `Sigma_k = (1 - reg) * Sigma_k + reg * I`.
    /// Must be in `[0, 1]`. Default: `0.0`.
    pub reg_param: F,
    /// Absolute threshold for a singular value to be considered significant,
    /// used to estimate the rank of the centered per-class matrix and to emit
    /// the "Variables are collinear" warning. Mirrors sklearn's `tol`
    /// (`discriminant_analysis.py:801-806`). Does **not** affect predictions.
    /// Default: `1e-4`.
    pub tol: F,
    /// Class prior probabilities, one entry per class (in sorted class order).
    ///
    /// Mirrors sklearn's `priors` constructor argument
    /// (`discriminant_analysis.py:351,359`; `_parameter_constraints`:
    /// `"priors": ["array-like", None]`, `:341`). Default `None`: the empirical
    /// class frequencies `n_k / n` are used (`self.priors_ = np.bincount(y) /
    /// n_samples`, `:921-922`). When `Some(p)`, `p` is used **verbatim** as the
    /// per-class priors — exactly as sklearn does (`self.priors_ =
    /// np.array(self.priors)`, `:924`): the values are NOT renormalized to sum
    /// to 1 and negative/zero entries are NOT rejected (sklearn accepts e.g.
    /// `[0.5, 0.6]` and negatives verbatim). The priors enter the decision
    /// through the additive `+ log(prior_k)` term (`:976`); for a zero entry
    /// `log(0) = -inf` and for a negative entry `log(p) = NaN`, matching
    /// sklearn's verbatim arithmetic (no special-casing).
    ///
    /// The one deviation (R-DEV-4): the length is checked at [`Fit::fit`]. If
    /// `p.len() != n_classes`, ferrolearn returns
    /// [`FerroError::ShapeMismatch`]; sklearn would silently mis-index (a
    /// CPython/numpy footgun Rust eliminates).
    pub priors: Option<Array1<F>>,
    /// Whether to compute and store the per-class covariance matrices on the
    /// fitted model. Mirrors sklearn's `store_covariance` constructor argument
    /// (`discriminant_analysis.py:353,361`; default `False`). When `true`,
    /// [`Fit::fit`] reconstructs each class covariance from the regularized
    /// scalings/rotations (`Rₖ · diag(scalingsₖ) · Rₖᵀ`, matching
    /// `cov.append(np.dot(S2 * Vt.T, Vt))`, `:951-952`) and exposes them via
    /// [`FittedQDA::covariance`]. When `false` (the default), no covariance is
    /// stored and [`FittedQDA::covariance`] returns `None`, mirroring sklearn
    /// (`covariance_` only exists when `store_covariance=True`, `:955-956`).
    pub store_covariance: bool,
}

impl<F: Float> QDA<F> {
    /// Create a new `QDA` with default settings.
    ///
    /// Default: `reg_param = 0.0`, `tol = 1e-4` (sklearn's default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            reg_param: F::zero(),
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            priors: None,
            store_covariance: false,
        }
    }

    /// Set whether the per-class covariance matrices are stored on the fitted
    /// model (sklearn's `store_covariance`, `discriminant_analysis.py:353,361`;
    /// default `false`). When `true`, [`FittedQDA::covariance`] returns the
    /// reconstructed `Rₖ · diag(scalingsₖ) · Rₖᵀ` per class; when `false` it
    /// returns `None`.
    #[must_use]
    pub fn with_store_covariance(mut self, store_covariance: bool) -> Self {
        self.store_covariance = store_covariance;
        self
    }

    /// Set the regularization parameter.
    #[must_use]
    pub fn with_reg_param(mut self, reg_param: F) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set the rank/collinearity tolerance `tol`.
    ///
    /// A class whose centered matrix has a singular value `<= tol` for any
    /// principal axis triggers the "Variables are collinear" warning (the
    /// rank is `< n_features`). This does not change the predictions; it only
    /// controls the warning, mirroring sklearn (`discriminant_analysis.py:801`).
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the class priors (one entry per class, in sorted class order).
    ///
    /// Mirrors sklearn's `priors` constructor argument
    /// (`discriminant_analysis.py:351,359`). The values are used **verbatim**
    /// at [`Fit::fit`] (no renormalization, no non-negativity check — matching
    /// `self.priors_ = np.array(self.priors)`, `:924`). When unset (the default
    /// `None`), the empirical class frequencies `n_k / n` are used. The slice
    /// length must equal the number of distinct classes or `fit` returns
    /// [`FerroError::ShapeMismatch`] (the R-DEV-4 deviation; see the
    /// [`priors`](Self::priors) field).
    #[must_use]
    pub fn with_priors(mut self, priors: Array1<F>) -> Self {
        self.priors = Some(priors);
        self
    }
}

impl<F: Float> Default for QDA<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-class model component for QDA (sklearn's SVD formulation).
#[derive(Debug, Clone)]
struct QDAClass<F> {
    /// Class mean, shape `(n_features,)`. Mirrors `means_[i]`.
    mean: Array1<F>,
    /// Per-axis scalings `(1 - reg) * (S² / (n_k - 1)) + reg`, length
    /// `k = min(n_k, n_features)`. Mirrors sklearn's `scalings_[i]`
    /// (`discriminant_analysis.py:948-949,953`). A zero entry encodes a
    /// collapsed (rank-deficient) principal axis.
    scalings: Array1<F>,
    /// Rotation matrix `Vtᵀ`, shape `(n_features, k)`. Mirrors sklearn's
    /// `rotations_[i]` (`discriminant_analysis.py:954`).
    rotations: Array2<F>,
    /// Log-prior probability for this class, `log(n_k / n)`.
    log_prior: F,
}

/// Fitted Quadratic Discriminant Analysis model.
///
/// Stores per-class means, SVD scalings/rotations, and log-priors. Implements
/// [`Predict`] to produce class labels.
#[derive(Debug, Clone)]
pub struct FittedQDA<F> {
    /// Per-class QDA models.
    class_models: Vec<QDAClass<F>>,
    /// Sorted unique class labels.
    classes: Vec<usize>,
    /// Number of features seen during fitting.
    n_features: usize,
    /// Resolved per-class priors (in sorted class order). Mirrors sklearn's
    /// fitted `priors_` (`discriminant_analysis.py:921-924`): empirical `n_k /
    /// n` when [`QDA::priors`] is `None`, otherwise the provided priors
    /// verbatim. Each entry equals `exp(QDAClass::log_prior)` of the matching
    /// class.
    priors: Array1<F>,
    /// Per-class covariance matrices, present only when [`QDA::store_covariance`]
    /// was `true` at fit time. Mirrors sklearn's `covariance_`
    /// (`discriminant_analysis.py:951-956`): each is the reconstructed
    /// `Rₖ · diag(scalingsₖ) · Rₖᵀ` (`(n_features, n_features)`) built from the
    /// regularized scalings. `None` when `store_covariance` was `false` —
    /// matching sklearn, where `covariance_` only exists when
    /// `store_covariance=True`.
    covariance: Option<Vec<Array2<F>>>,
}

impl<F: Float> FittedQDA<F> {
    /// Returns the per-class covariance matrices when [`QDA::store_covariance`]
    /// was `true` at fit time, else `None`. Mirrors sklearn's `covariance_`
    /// (`discriminant_analysis.py:951-956`): each entry is the regularized
    /// `Rₖ · diag(scalingsₖ) · Rₖᵀ` of shape `(n_features, n_features)`, in
    /// sorted class order. `None` (no attribute) when `store_covariance` was
    /// `false` — matching sklearn, where `covariance_` is set only under
    /// `store_covariance=True`.
    #[must_use]
    pub fn covariance(&self) -> Option<&[Array2<F>]> {
        self.covariance.as_deref()
    }
    /// Returns the class means, one per class. Mirrors sklearn's `means_`.
    #[must_use]
    pub fn means(&self) -> Vec<&Array1<F>> {
        self.class_models.iter().map(|m| &m.mean).collect()
    }

    /// Returns the resolved per-class priors (in sorted class order). Mirrors
    /// sklearn's fitted `priors_` (`discriminant_analysis.py:921-924`):
    /// empirical class frequencies `n_k / n` when [`QDA::priors`] was `None`,
    /// otherwise the priors supplied via [`QDA::with_priors`] verbatim.
    #[must_use]
    pub fn priors(&self) -> &Array1<F> {
        &self.priors
    }

    /// Returns the per-class scalings (`S² / (n_k - 1)` regularized), one
    /// `Array1<F>` of length `min(n_k, n_features)` per class. Mirrors
    /// sklearn's `scalings_` (`discriminant_analysis.py:832-838`).
    #[must_use]
    pub fn scalings(&self) -> Vec<&Array1<F>> {
        self.class_models.iter().map(|m| &m.scalings).collect()
    }

    /// Returns the per-class rotations `Vtᵀ`, one `(n_features, k)` matrix per
    /// class. Mirrors sklearn's `rotations_` (`discriminant_analysis.py:824-830`).
    #[must_use]
    pub fn rotations(&self) -> Vec<&Array2<F>> {
        self.class_models.iter().map(|m| &m.rotations).collect()
    }
}

/// `0.5` as `F`, built panic-free from `1 / (1 + 1)` (exact for binary floats).
#[inline]
fn half<F: Float>() -> F {
    F::one() / (F::one() + F::one())
}

/// Convert a `usize` count to `F` without panicking. Returns
/// [`FerroError::NumericalInstability`] if the value is not representable
/// (effectively impossible for the sample/feature counts here, but `Float::from`
/// is fallible and the codebase forbids `.unwrap()` in production).
#[inline]
fn usize_to_f<F: Float>(v: usize) -> Result<F, FerroError> {
    F::from(v).ok_or_else(|| FerroError::NumericalInstability {
        message: format!("could not represent count {v} as the float type"),
    })
}

impl<F: Float + ndarray::ScalarOperand + Send + Sync + 'static> FittedQDA<F> {
    /// Raw per-class log-posterior `(n_samples, n_classes)`, the SVD form of
    /// sklearn's `_decision_function` (`discriminant_analysis.py:962-976`):
    ///
    /// ```text
    /// Xm    = X - means_[i]
    /// X2    = Xm @ (rotations_[i] * scalings_[i]^(-0.5))   # column-scaled
    /// norm2 = sum(X2², axis=1)
    /// u     = sum(log(scalings_[i]))
    /// dec_i = -0.5 * (norm2 + u) + log(priors_[i])
    /// ```
    ///
    /// For a rank-deficient class a zero scaling makes `scalings^(-0.5) = +inf`
    /// and `log(scalings) = -inf`, so the degenerate `inf`/`NaN` arithmetic
    /// propagates exactly as in numpy/sklearn (no special-casing).
    fn raw_decision(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        if n_features != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: vec![n_features],
                context: "number of features must match fitted model".into(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let half: F = half();
        let neg_half = -half;
        let mut out = Array2::<F>::zeros((n_samples, n_classes));

        for (c, model) in self.class_models.iter().enumerate() {
            // u = sum(log(scalings_[c]))  — -inf when any scaling is 0.
            let u = model
                .scalings
                .iter()
                .fold(F::zero(), |acc, &s| acc + s.ln());
            // Column-scaled rotations: rotations_[c] * scalings_[c]^(-0.5).
            // (n_features, k); scaled[:, j] = rotations[:, j] * scalings[j]^(-0.5).
            let k = model.scalings.len();
            let mut scaled = Array2::<F>::zeros((n_features, k));
            for j in 0..k {
                let inv_sqrt = model.scalings[j].powf(neg_half); // +inf when scaling==0
                for r in 0..n_features {
                    scaled[[r, j]] = model.rotations[[r, j]] * inv_sqrt;
                }
            }
            for i in 0..n_samples {
                // Xm = x_i - mean_c
                let diff: Array1<F> = x.row(i).to_owned() - &model.mean;
                // X2 = Xm @ scaled  (length k); norm2 = sum(X2²).
                let x2 = diff.dot(&scaled);
                let norm2 = x2.iter().fold(F::zero(), |acc, &v| acc + v * v);
                out[[i, c]] = neg_half * (norm2 + u) + model.log_prior;
            }
        }
        Ok(out)
    }

    /// Predict per-class probabilities. Mirrors sklearn
    /// `QuadraticDiscriminantAnalysis.predict_proba`
    /// (`discriminant_analysis.py:1024-1042`).
    ///
    /// Computes the max-shifted softmax over the per-class log-posteriors
    /// from [`decision_function`](Self::decision_function). Returns shape
    /// `(n_samples, n_classes)`; rows sum to 1.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let values = self.raw_decision(x)?;
        let n_samples = values.nrows();
        let n_classes = values.ncols();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let max_l = (0..n_classes)
                .map(|c| values[[i, c]])
                .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
            let mut sum_exp = F::zero();
            for c in 0..n_classes {
                let e = (values[[i, c]] - max_l).exp();
                proba[[i, c]] = e;
                sum_exp = sum_exp + e;
            }
            for c in 0..n_classes {
                proba[[i, c]] = proba[[i, c]] / sum_exp;
            }
        }
        Ok(proba)
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba).
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        Ok(crate::log_proba(&proba))
    }

    /// Per-class quadratic discriminant scores. Mirrors sklearn
    /// `QuadraticDiscriminantAnalysis.decision_function`
    /// (`discriminant_analysis.py:962-976`). Returns shape
    /// `(n_samples, n_classes)` with the SVD log-posterior
    /// `δ_c(x) = -½ (‖(x-μ_c)·R_c·S_c^(-½)‖² + Σ log S_c) + log π_c`.
    /// argmax of each row agrees with [`Predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.raw_decision(x)
    }
}

/// Singular values `S` and right singular vectors transposed `Vt` of the thin
/// SVD `A = U·diag(S)·Vt` (`full_matrices=False`), computed on the ferray
/// substrate (`ferray::linalg::svd`, `ferray-linalg/src/decomp/svd.rs:40`) —
/// the analog of scikit-learn's `_, S, Vt = np.linalg.svd(Xgc,
/// full_matrices=False)` (`discriminant_analysis.py:944`). Mirrors the bridging
/// pattern in `bayesian_ridge.rs::svd_thin` (R-SUBSTRATE-4): the caller keeps
/// its `ndarray` signature and the ndarray↔ferray conversion happens here.
///
/// Returns `(S, Vt)` with `S` of length `k = min(m, n)` and `Vt` of shape
/// `(k, n)`.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the ferray array build or
/// the SVD itself fails.
fn svd_s_vt<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let (m, n) = a.dim();

    // Bridge ndarray -> ferray (R-SUBSTRATE-4).
    let a_flat: Vec<F> = a.iter().copied().collect();
    let fa =
        FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([m, n]), a_flat).map_err(|e| {
            FerroError::NumericalInstability {
                message: format!("ferray svd: failed to build centered matrix: {e}"),
            }
        })?;

    // full_matrices=false => thin SVD, matching numpy's `full_matrices=False`.
    let (_u, s, vt) = svd(&fa, false).map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray svd failed: {e}"),
    })?;

    let s_nd = Array1::from_vec(s.iter().copied().collect());
    let vt_shape = vt.shape();
    let vt_nd = Array2::from_shape_vec((vt_shape[0], vt_shape[1]), vt.iter().copied().collect())
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray svd: Vt shape conversion failed: {e}"),
        })?;

    Ok((s_nd, vt_nd))
}

impl<F: LinalgFloat + ScalarOperand> Fit<Array2<F>, Array1<usize>> for QDA<F> {
    type Fitted = FittedQDA<F>;
    type Error = FerroError;

    /// Fit the QDA model via a per-class SVD of the centered data, mirroring
    /// scikit-learn (`discriminant_analysis.py:940-960`). A rank-deficient
    /// (collinear) class does **not** error — its zero singular value yields a
    /// zero scaling and a "Variables are collinear" warning is emitted, exactly
    /// as in sklearn.
    ///
    /// # Errors
    ///
    /// - [`FerroError::ShapeMismatch`] — sample count mismatch.
    /// - [`FerroError::InsufficientSamples`] — fewer than 2 distinct classes
    ///   or a class has too few samples.
    /// - [`FerroError::InvalidParameter`] — `reg_param` not in `[0, 1]`.
    /// - [`FerroError::NumericalInstability`] — the underlying SVD fails.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedQDA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        if self.reg_param < num_traits::Zero::zero() || self.reg_param > num_traits::One::one() {
            return Err(FerroError::InvalidParameter {
                name: "reg_param".into(),
                reason: "must be in [0, 1]".into(),
            });
        }

        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "QDA requires at least 2 distinct classes".into(),
            });
        }

        let n_classes = classes.len();
        let n_f = usize_to_f::<F>(n_samples)?;

        // Resolve per-class priors (in sorted class order), mirroring sklearn
        // (`discriminant_analysis.py:921-924`):
        //   priors is None  -> self.priors_ = np.bincount(y) / n_samples
        //   priors is Some(p) -> self.priors_ = np.array(self.priors) (verbatim)
        // The provided priors are used as-is: NOT renormalized to sum 1, and
        // negative/zero entries are NOT rejected (sklearn accepts them). The
        // ONE deviation (R-DEV-4) is the length check below — sklearn silently
        // mis-indexes a wrong-length `priors`; Rust eliminates that footgun by
        // returning `ShapeMismatch`.
        let priors: Array1<F> = match &self.priors {
            None => {
                // Empirical n_k / n per class. `n_k / n` here is bit-identical
                // to the previous `(n_k_f / n_f).ln()` path's argument, so the
                // empirical decision is byte-identical.
                let mut p = Array1::<F>::zeros(n_classes);
                for (idx, &cls) in classes.iter().enumerate() {
                    let n_k = y.iter().filter(|&&label| label == cls).count();
                    p[idx] = usize_to_f::<F>(n_k)? / n_f;
                }
                p
            }
            Some(provided) => {
                if provided.len() != n_classes {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n_classes],
                        actual: vec![provided.len()],
                        context: "priors length must equal the number of classes".into(),
                    });
                }
                provided.clone()
            }
        };

        let one_minus_reg = <F as num_traits::One>::one() - self.reg_param;
        let mut class_models = Vec::with_capacity(classes.len());
        // sklearn: `cov = []` only when store_covariance (`:928-929`); the
        // `covariance_` attribute is set only under store_covariance (`:955-956`).
        let mut covariance: Option<Vec<Array2<F>>> = if self.store_covariance {
            Some(Vec::with_capacity(classes.len()))
        } else {
            None
        };

        for (cls_idx, &cls) in classes.iter().enumerate() {
            // Extract samples for this class.
            let indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|&(_, label)| *label == cls)
                .map(|(i, _)| i)
                .collect();

            let n_k = indices.len();
            if n_k < 2 {
                return Err(FerroError::InsufficientSamples {
                    required: 2,
                    actual: n_k,
                    context: format!("class {cls} needs at least 2 samples for QDA"),
                });
            }

            let n_k_f = usize_to_f::<F>(n_k)?;

            // Compute class mean (`meang = Xg.mean(0)`, discriminant_analysis.py:935).
            let mut mean = Array1::<F>::zeros(n_features);
            for &i in &indices {
                for j in 0..n_features {
                    mean[j] += x[[i, j]];
                }
            }
            mean.mapv_inplace(|v| v / n_k_f);

            // Center: `Xgc = Xg - meang` (discriminant_analysis.py:942).
            let mut xgc = Array2::<F>::zeros((n_k, n_features));
            for (row, &i) in indices.iter().enumerate() {
                for j in 0..n_features {
                    xgc[[row, j]] = x[[i, j]] - mean[j];
                }
            }

            // Thin SVD `Xgc = U·diag(S)·Vt` (discriminant_analysis.py:944).
            let (s, vt) = svd_s_vt::<F>(&xgc)?;

            // Rank / collinearity warning (discriminant_analysis.py:945-947).
            // `rank = sum(S > tol)`; warn if `rank < n_features`. Does NOT
            // change predictions — only emits a warning.
            let rank = s.iter().filter(|&&sv| sv > self.tol).count();
            if rank < n_features {
                eprintln!("Variables are collinear");
            }

            // `S2 = (S**2) / (n_k - 1)`; `S2 = (1 - reg)*S2 + reg`
            // (discriminant_analysis.py:948-949). For a zero singular value the
            // scaling collapses to `reg` (== 0 when unregularized), encoding the
            // rank-deficient axis with the same degenerate float arithmetic as
            // numpy/sklearn.
            let n_k_minus_1 = usize_to_f::<F>(n_k - 1)?;
            let scalings: Array1<F> = s.mapv(|sv| {
                let s2 = (sv * sv) / n_k_minus_1;
                one_minus_reg * s2 + self.reg_param
            });

            // `rotations.append(Vt.T)` (discriminant_analysis.py:954):
            // (n_features, k) where k = len(S) = min(n_k, n_features).
            let rotations = vt.t().to_owned();

            // store_covariance: `cov.append(np.dot(S2 * Vt.T, Vt))`
            // (discriminant_analysis.py:951-952), using the REGULARIZED `S2`
            // computed above. With `rotations = Vt.T` of shape (n_features, k):
            //   cov[a,b] = Σ_j rotations[a,j] · scalings[j] · rotations[b,j]
            // i.e. `Rₖ · diag(scalingsₖ) · Rₖᵀ`, of shape (n_features, n_features).
            if let Some(cov) = covariance.as_mut() {
                let k = scalings.len();
                let mut cov_k = Array2::<F>::zeros((n_features, n_features));
                for a in 0..n_features {
                    for b in 0..n_features {
                        let mut acc = <F as num_traits::Zero>::zero();
                        for j in 0..k {
                            acc += rotations[[a, j]] * scalings[j] * rotations[[b, j]];
                        }
                        cov_k[[a, b]] = acc;
                    }
                }
                cov.push(cov_k);
            }

            // log(prior_k) of the resolved priors (empirical n_k/n for the
            // `None` path — bit-identical to the previous `(n_k_f / n_f).ln()`
            // — or the provided prior verbatim, where `ln` of a zero gives
            // `-inf` and of a negative gives `NaN`, matching sklearn `:976`).
            let log_prior = priors[cls_idx].ln();

            class_models.push(QDAClass {
                mean,
                scalings,
                rotations,
                log_prior,
            });
        }

        Ok(FittedQDA {
            class_models,
            classes,
            n_features,
            priors,
            covariance,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> Predict<Array2<F>> for FittedQDA<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Returns `classes_.take(argmax over the per-class decision)`, mirroring
    /// sklearn `predict` (`discriminant_analysis.py:1020-1022`). The argmax
    /// replicates numpy's `ndarray.argmax(1)` semantics exactly, including the
    /// degenerate rank-deficient case: a `NaN` decision (from a collinear class)
    /// is treated as the maximum, and the **first** index that is either `NaN`
    /// or strictly greater than the running best wins (ties → first index).
    /// This is what makes a singular class deterministically dominate.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the fitted model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let dec = self.raw_decision(x)?;
        let n_samples = dec.nrows();
        let n_classes = dec.ncols();
        let mut predictions = Array1::<usize>::zeros(n_samples);

        for i in 0..n_samples {
            // numpy argmax: best := dec[i,0]; once best is NaN nothing beats it;
            // otherwise dec[i,c] wins if it is NaN or strictly greater.
            let mut best_idx = 0;
            let mut best = dec[[i, 0]];
            for c in 1..n_classes {
                if best.is_nan() {
                    break;
                }
                let v = dec[[i, c]];
                if v.is_nan() || v > best {
                    best = v;
                    best_idx = c;
                }
            }
            predictions[i] = self.classes[best_idx];
        }

        Ok(predictions)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static> HasClasses for FittedQDA<F> {
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
        let m = QDA::<f64>::new();
        assert!(m.reg_param == 0.0);
    }

    #[test]
    fn test_builder() {
        let m = QDA::<f64>::new().with_reg_param(0.5);
        assert!(m.reg_param == 0.5);
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

        let model = QDA::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_multiclass_classification() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 10.5, 0.5,
                0.0, 10.0, 0.5, 10.0, 0.0, 10.5, 0.5, 10.5,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let model = QDA::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes(), &[0, 1, 2]);

        let preds = fitted.predict(&x).unwrap();
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 10, "expected at least 10 correct, got {correct}");
    }

    #[test]
    fn test_regularization() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // With regularization should still work.
        let model = QDA::<f64>::new().with_reg_param(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0, 0, 0];

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_reg_param() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let model = QDA::<f64>::new().with_reg_param(-0.1);
        assert!(model.fit(&x, &y).is_err());

        let model2 = QDA::<f64>::new().with_reg_param(1.5);
        assert!(model2.fit(&x, &y).is_err());
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let fitted = QDA::<f64>::new().fit(&x, &y).unwrap();

        let x_bad = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_has_classes() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let fitted = QDA::<f64>::new().fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0, 1]);
        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_means() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let fitted = QDA::<f64>::new().with_reg_param(0.1).fit(&x, &y).unwrap();
        let means = fitted.means();
        assert_eq!(means.len(), 2);
    }

    #[test]
    fn test_priors_default_none() {
        let m = QDA::<f64>::new();
        assert!(m.priors.is_none());
    }

    #[test]
    fn test_with_priors_builder() {
        let m = QDA::<f64>::new().with_priors(array![0.9, 0.1]);
        assert_eq!(m.priors.as_ref().map(|p| p.to_vec()), Some(vec![0.9, 0.1]));
    }

    #[test]
    fn test_priors_empirical_resolved() {
        // None path: fitted priors_ are the empirical n_k / n (here 4/8, 4/8).
        // `array!` builds the Array2 directly (no fallible from_shape_vec).
        let x = array![
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [8.0, 8.0],
            [8.0, 9.0],
            [9.0, 8.0],
            [9.0, 9.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];
        let fitted = QDA::<f64>::new().fit(&x, &y);
        assert!(fitted.is_ok(), "empirical fit should succeed");
        if let Ok(f) = fitted {
            assert_eq!(f.priors().to_vec(), vec![0.5, 0.5]);
        }
    }

    #[test]
    fn test_priors_length_mismatch_errors() {
        // R-DEV-4: a wrong-length priors vector errors (2 classes, 3 priors).
        let x = array![[1.0], [2.0], [5.0], [6.0]];
        let y = array![0, 0, 1, 1];
        let res = QDA::<f64>::new()
            .with_priors(array![0.3, 0.3, 0.4])
            .fit(&x, &y);
        assert!(matches!(res, Err(FerroError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_class_with_too_few_samples() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 5.0, 6.0]).unwrap();
        let y = array![0, 1, 1]; // class 0 has only 1 sample

        let model = QDA::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }
}
