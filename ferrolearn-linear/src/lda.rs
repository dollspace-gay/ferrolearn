//! Linear Discriminant Analysis (LDA).
//!
//! LDA is both a supervised dimensionality-reduction technique and a linear
//! classifier. This module mirrors scikit-learn's **default** `solver="svd"`
//! path (`sklearn/discriminant_analysis.py:487-559`, commit 156ef14): rather
//! than forming a covariance and solving the classical `Sw⁻¹·Sb` Fisher
//! eigenproblem, it whitens the within-class data with two SVDs, derives the
//! whitened projection `scalings_` and the weighted overall mean `xbar_`, then
//! forms the **affine** classifier `coef_`/`intercept_` (whose `intercept_`
//! embeds `log(priors_)`).
//!
//! The [`Solver::Lsqr`] least-squares path (`_solve_lstsq`,
//! `discriminant_analysis.py:365-419`) is also available (`LDA::with_solver`):
//! it forms the prior-weighted within-class covariance `covariance_` and solves
//! `coef_ = lstsq(covariance_, means_.T).T`, supporting covariance
//! [`Shrinkage`] (`None`/`Auto` Ledoit-Wolf/`Fixed`); it does NOT do
//! dimensionality reduction (no `transform`). [`Solver::Eigen`] is deferred
//! (#596).
//!
//! # Algorithm (`_solve_svd`, `discriminant_analysis.py:487-559`)
//!
//! With `n = n_samples`, `c = n_classes`:
//! 1. `priors_`: empirical `n_k / n` when the constructor `priors` is `None`
//!    (sklearn's default), else the provided `priors` used VERBATIM
//!    (`discriminant_analysis.py:601-605`).
//! 2. `means_` = per-class mean; `xbar_ = priors_ @ means_`.
//! 3. `Xc` = each sample minus its class mean (stacked); `std = std(Xc, axis=0)`
//!    (population, `ddof=0`), zeros replaced by `1`.
//! 4. `Xw = sqrt(1/(n-c)) · (Xc / std)`; thin SVD `Xw = U·diag(S)·Vt`;
//!    `rank = Σ(S > tol)`; `scalings = (Vt[:rank]/std).T / S[:rank]`.
//! 5. Between-class scaled centers `Xb = (sqrt(n·priors_·1/(c-1)) ⊙
//!    (means_-xbar_).T).T @ scalings`; thin SVD `Xb = U2·diag(S2)·Vt2`;
//!    `explained_variance_ratio_ = (S2²/ΣS2²)[:max_components]`;
//!    `rank2 = Σ(S2 > tol·S2[0])`; `scalings_ = scalings @ Vt2.T[:, :rank2]`.
//! 6. `coef = (means_-xbar_) @ scalings_`;
//!    `intercept_ = -½·Σ(coef²) + log(priors_)`;
//!    `coef_ = coef @ scalings_.T`; `intercept_ -= xbar_ @ coef_.T`.
//!
//! Inference (the `LinearClassifierMixin`, `discriminant_analysis.py:739`):
//! - `transform(X) = ((X - xbar_) @ scalings_)[:, :max_components]`
//!   (`discriminant_analysis.py:684-689`).
//! - `decision_function(X) = X @ coef_.T + intercept_`
//!   (`discriminant_analysis.py:739`).
//! - `predict(X)` = `classes_[argmax(decision_function)]`.
//! - `predict_proba(X)` = `softmax(decision_function)`
//!   (`discriminant_analysis.py:706-711`).
//!
//! The number of discriminant directions is at most `min(n_classes - 1,
//! n_features)`.
//!
//! ## REQ status (per `.design/linear/lda.md`, mirrors `sklearn/discriminant_analysis.py` @ 1.5.2)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (svd fit + decision_function parity) | SHIPPED | `_solve_svd` in `fn fit` (`fn svd_s_vt` → `ferray::linalg::svd`) builds `coef_`/`intercept_`/`xbar_`/`scalings_` (`discriminant_analysis.py:556-559`); `fn decision_function` = `X @ coef_.T + intercept_` (`:739`). Consumer: `Predict for FittedLDA` + crate-root `pub use`. Test `lda_decision_function_parity` <1e-6 vs live oracle. #588. |
//! | REQ-2 (predict argmax) | SHIPPED | `Predict::predict` = `classes_[argmax(decision_function)]`; the affine decision carries `log(priors_)` via `intercept_`. Test `lda_imbalanced_priors_predict` (prior shifts the boundary, label-for-label vs live oracle). #589. |
//! | REQ-3 (predict_proba prior-aware) | SHIPPED | `FittedLDA::predict_proba` = `softmax(decision_function)` (`discriminant_analysis.py:711`); rows sum to 1. Test `lda_imbalanced_priors_predict` proba block <1e-6 vs live oracle. #590 (partial: multiclass softmax; binary `expit` collapse pends #600). |
//! | REQ-5 (transform parity) | SHIPPED | `fn transform` = `((X - xbar_) @ scalings_)[:, :max_components]` (`discriminant_analysis.py:684-689`). Test `lda_transform_parity` <1e-6 (per-column sign) vs live oracle. #592. |
//! | REQ-6 (n_components bound) | SHIPPED | `fn fit` computes `max_components = min(n_classes-1, n_features)`, defaults `None` to it, errors `Some(0)`/`Some(k>max)` (`discriminant_analysis.py:614-625`). Tests `test_lda_default_n_components`, `test_lda_error_zero_n_components`, `test_lda_error_n_components_too_large`. |
//! | REQ-7 (priors: None=empirical + provided) | SHIPPED | `fn fit` resolves `priors_`: empirical `n_k/n` when `priors` is `None` (`discriminant_analysis.py:601-603`), else the provided `LDA::with_priors` array, now VALIDATED like sklearn LDA (`:607-612`, unlike QDA): R-DEV-4 length check (`p.len() != n_classes` → `ShapeMismatch`, sklearn would mis-index `:540,557`); negative entries → `InvalidParameter` (`:607-608`, `raise ValueError("priors must be non-negative")`); renormalized `p / p.sum()` with an `eprintln!` warning (the crate's warning channel, cf. qda.rs) when `|Σ-1| > 1e-5` (`:610-612`). The resolved priors flow into `xbar_ = priors_ @ means_` (`:517`), the between-class scaling `sqrt(n·priors_·fac)` (`:540`), and `intercept_ += log(priors_)` (`:557`). `FittedLDA::priors` exposes `priors_`. Consumer: the resolved `priors` is read by `fn fit` (xbar_/scaling/intercept_); `Predict for FittedLDA` consumes the prior-shifted decision. Tests: `lda_imbalanced_priors_predict` (empirical `[0.9091,0.0909]` flips the label), `lda_provided_priors` (`with_priors([0.9,0.1])` `predict_proba` <1e-6 vs live oracle; empirical default differs), `lda_priors_negative_rejected` (`[-0.1,1.1]` → `Err`), `lda_priors_renormalized` (`[0.5,0.6]` → `priors_=[0.4545…,0.5454…]`, `predict_proba` <1e-6 vs the live oracle which renormalizes internally). #593, #603. |
//! | REQ-8 (coef_/intercept_/xbar_) | SHIPPED | `FittedLDA::{coef, intercept, xbar}` accessors expose the `_solve_svd` arrays (`discriminant_analysis.py:556-559,517`). Consumer: `fn decision_function` reads `coef_`/`intercept_`; `fn transform` reads `xbar_`. Verified via `lda_decision_function_parity` (decision = `X@coef_.T+intercept_`) + `lda_transform_parity` (uses `xbar_`). |
//! | REQ-13 (explained_variance_ratio_) | SHIPPED | `fn fit` sets `explained_variance_ratio_ = (S2²/ΣS2²)[:max_components]` from the SECOND (between-class) SVD (`discriminant_analysis.py:550-552`). Test `test_lda_explained_variance_ratio_oracle` <1e-9 vs live `L().explained_variance_ratio_`. #599. |
//! | REQ-4 (predict_log_proba smallest_normal floor) | SHIPPED | `FittedLDA::predict_log_proba` mirrors sklearn exactly (`discriminant_analysis.py:713-737`): `predict_proba` then `prediction[prediction == 0.0] += smallest_normal` (`F::min_positive_value()` = numpy `finfo.smallest_normal`, `:729-736`) before `log`, so nonzero probas keep their true `ln` and exact zeros become `log(MIN_POSITIVE)` (not `-inf`). Consumer: shares `FittedLDA::predict_proba` (the `Predict` path). Test `lda_predict_log_proba` (overlapping 3-class, all-finite log-probas) <1e-6 vs live `LinearDiscriminantAnalysis().predict_log_proba`. #591. |
//! | REQ-9 (lsqr solver) | SHIPPED | `Solver::Lsqr` (`LDA::with_solver`) dispatches `fn fit` to `fn solve_lstsq` (sklearn `_solve_lstsq`, `discriminant_analysis.py:365-419`): `covariance_ = Σ_k priors_[k] · cov(X_k)` (`_class_cov` `:128-172`, ALWAYS populated for lsqr, `:413`); `coef_ = lstsq(covariance_, means_.T)[0].T` (`:416`) via `fn lstsq_multi` → `ferray::linalg::lstsq` (multi-RHS, `ferray-linalg/src/solve.rs:208`, R-SUBSTRATE-4 bridge); `intercept_ = -½·diag(means_ @ coef_.T) + log(priors_)` (`:417-418`). No `scalings_`/`xbar_`/`explained_variance_ratio_` / `transform` (sklearn raises for lsqr `transform`, `:676-679`; `max_components=0` ⇒ empty projection). Binary collapse `coef_[1]-coef_[0]` deferred to #600 (coef_ stays `(n_classes, n_features)`, matching the svd path). Consumer: `fn fit` reads `self.solver` and dispatches; `Predict`/`predict_proba` for `FittedLDA` consume the lsqr `coef_`/`intercept_`. Test `lda_lsqr_solver` (collapsed `coef_[1]-coef_[0]` = `[14.7368…, 14.7368…]`, predict/predict_proba) <1e-6 vs live `LinearDiscriminantAnalysis(solver='lsqr').fit(X,y)`. #595. |
//! | REQ-10 (eigen solver) | NOT-STARTED | open prereq blocker #596. The `Solver::Eigen` variant EXISTS (so the enum is complete) but `fn fit` returns `FerroError::InvalidParameter("eigen solver not yet implemented (#596)")`; no generalized-eigenvalue `_solve_eigen` path (`discriminant_analysis.py:421-485`). |
//! | REQ-11 (shrinkage None/auto/float) | SHIPPED | `Shrinkage::{None, Auto, Fixed(F)}` (`LDA::with_shrinkage`) drives `fn cov_shrunk` (sklearn `_cov`, `discriminant_analysis.py:36-93`) inside `fn solve_lstsq`: `None` → maximum-likelihood empirical covariance (`fn empirical_covariance`, `np.cov(...,bias=1)`, `:76-77`); `Fixed(s)` → `(1-s)·emp + s·(trace(emp)/p)·I` (`shrunk_covariance`, `covariance/_shrunk_covariance.py:153-156`), validated `0 ≤ s ≤ 1` (`Interval(Real,0,1,closed=both)`, `:339`) else `InvalidParameter`; `Auto` → analytical Ledoit-Wolf (`fn ledoit_wolf_shrinkage`, transcribed from `covariance/_shrunk_covariance.py:365-401` unblocked case) on StandardScaler-standardized data then rescaled (`_cov` `:70-75`). `Solver::Svd` + non-`None` shrinkage → `InvalidParameter("shrinkage not supported with svd solver")` mirroring sklearn `NotImplementedError` (`:628-629`). Consumer: `fn fit`/`fn solve_lstsq` read `self.shrinkage`. Tests `lda_shrinkage_fixed` (`Fixed(0.5)` coef = `[12.043…, 12.043…]`), `lda_shrinkage_auto` (`Auto` coef = `[11.3706…, 11.3706…]`, validates the Ledoit-Wolf transcription), `lda_svd_shrinkage_rejected` (svd+shrinkage → `Err`) <1e-6 vs the live oracle. #597. |
//! | REQ-12 (store_covariance / covariance_) | SHIPPED | `LDA::with_store_covariance` sets the flag (sklearn default `false`, `discriminant_analysis.py:353`); when `true`, `fn fit` computes the shared within-class covariance `covariance_ = Σ_k priors_[k] · cov(X_k)` (`:509-510`, `_class_cov` `:128-172`) with the maximum-likelihood (`bias=1`, ÷`n_k`) per-class empirical covariance (`empirical_covariance`, `np.cov(...,bias=1)`), stored on `FittedLDA::covariance` (`None` when the flag is unset, matching sklearn). Consumer: `fn fit` reads `self.store_covariance`/`priors`/`means` and populates the field; `FittedLDA::covariance` exposes it. Test `lda_store_covariance` matches the live oracle `LinearDiscriminantAnalysis(store_covariance=True).fit(X,y).covariance_` to 1e-9 and asserts `None` for the default/`false` path. #598. |
//! | REQ-14 (binary decision_function shape `(n,)`) | NOT-STARTED | open prereq blocker #600. `fn decision_function` always returns `(n, n_classes)`; sklearn collapses binary to `(n,)` (`discriminant_analysis.py:651-657,739`). Binding-ABI layer (parallel to QDA #581). |
//! | REQ-15 (tol parameter) | SHIPPED | `LDA::with_tol` sets the svd-solver rank threshold (sklearn default `1e-4`, `discriminant_analysis.py:354,362`); `fn fit` reads `self.tol` into BOTH rank cutoffs `rank = Σ(S > tol)` (`:532`) and `rank2 = Σ(S2 > tol·S2[0])` (`:554`), REPLACING the prior hardcoded `1e-4`. Default `1e-4` ⇒ byte-identical to prior behavior (all existing svd-fit oracle tests stay green). Consumer: `fn fit` reads `self.tol` in both rank thresholds. Test `lda_tol_param` (field default `1e-4` + `with_tol` plumb-through). #601. |
//! | REQ-16 (ferray array-type substrate) | NOT-STARTED | open prereq blocker #602. The two SVDs run on `ferray::linalg::svd`; the owned array type is still `ndarray` (crate-wide deferral, cf. qda.rs REQ-12 #585). |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::lda::LDA;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! let lda = LDA::new(Some(1));
//! let x = Array2::from_shape_vec(
//!     (6, 2),
//!     vec![1.0, 1.0, 1.5, 1.2, 1.2, 0.8, 5.0, 5.0, 5.5, 4.8, 4.8, 5.2],
//! ).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//! let fitted = lda.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```

use ferray::linalg::{LinalgFloat, svd};
use ferray::{Array as FerrayArray, Ix2 as FerrayIx2};
use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, PipelineEstimator};
use ferrolearn_core::traits::{Fit, Predict, Transform};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, NumCast};

// ---------------------------------------------------------------------------
// Solver / Shrinkage enums
// ---------------------------------------------------------------------------

/// LDA solver selector (sklearn's `solver`, `discriminant_analysis.py:204-216`,
/// `_parameter_constraints` `StrOptions({svd, lsqr, eigen})` `:338`).
///
/// - [`Solver::Svd`] (default) — the singular-value-decomposition path
///   (`_solve_svd`, `discriminant_analysis.py:487-559`); supports `transform`
///   (dimensionality reduction) but NOT `shrinkage`.
/// - [`Solver::Lsqr`] — the least-squares path (`_solve_lstsq`,
///   `discriminant_analysis.py:365-419`): `coef_ = lstsq(covariance_,
///   means_.T).T`, `intercept_ = -½·diag(means_ @ coef_.T) + log(priors_)`.
///   Supports `shrinkage`; does NOT support `transform` (sklearn raises
///   `NotImplementedError`, `:676-679`).
/// - [`Solver::Eigen`] — the generalized-eigenvalue path
///   (`_solve_eigen`, `discriminant_analysis.py:421-485`). NOT implemented in
///   ferrolearn yet (open prereq blocker #596); [`Fit::fit`] returns a
///   [`FerroError`] for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Solver {
    /// Singular-value-decomposition solver (sklearn default).
    #[default]
    Svd,
    /// Least-squares solver (`_solve_lstsq`).
    Lsqr,
    /// Generalized-eigenvalue solver. NOT implemented (#596); errors at fit.
    Eigen,
}

/// LDA covariance-shrinkage selector (sklearn's `shrinkage`,
/// `discriminant_analysis.py:218-225`, `_parameter_constraints`
/// `[StrOptions({auto}), Interval(Real, 0, 1, closed=both), None]` `:339`).
///
/// Drives the per-class covariance estimate `_cov`
/// (`discriminant_analysis.py:36-93`) inside the `lsqr`/`eigen` solvers
/// (sklearn note `:225`: shrinkage works only with the `lsqr` and `eigen`
/// solvers):
/// - [`Shrinkage::None`] — no shrinkage; the maximum-likelihood empirical
///   covariance (`np.cov(..., bias=1)`, `:76-77`).
/// - [`Shrinkage::Auto`] — automatic Ledoit-Wolf shrinkage (`:70-75`):
///   standardize features, run the Ledoit-Wolf lemma, then rescale.
/// - [`Shrinkage::Fixed`]`(s)` — fixed shrinkage `s ∈ [0, 1]`
///   (`shrunk_covariance`, `_shrunk_covariance.py:111-158`):
///   `(1 - s)·emp_cov + s·(trace(emp_cov)/p)·I`.
///
/// # Type Parameters
///
/// - `F`: the floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Shrinkage<F> {
    /// No shrinkage (sklearn `None`/`'empirical'`). Default.
    #[default]
    None,
    /// Automatic Ledoit-Wolf shrinkage (sklearn `'auto'`).
    Auto,
    /// Fixed shrinkage coefficient `s ∈ [0, 1]` (sklearn `float`).
    Fixed(F),
}

// ---------------------------------------------------------------------------
// LDA (unfitted)
// ---------------------------------------------------------------------------

/// Linear Discriminant Analysis configuration.
///
/// Holds hyperparameters. Calling [`Fit::fit`] runs sklearn's default
/// `solver="svd"` path (`discriminant_analysis.py:487-559`) and returns a
/// [`FittedLDA`].
///
/// Use [`LDA::with_priors`] to supply class priors (sklearn's `priors`,
/// `discriminant_analysis.py:359`); the default `None` infers the empirical
/// priors `n_k / n` at fit time.
///
/// # Type Parameters
///
/// - `F`: The floating-point scalar type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LDA<F> {
    /// Number of discriminant components to retain.
    ///
    /// If `None`, defaults to `min(n_classes - 1, n_features)` at fit time.
    n_components: Option<usize>,

    /// Class prior probabilities (sklearn's `priors`,
    /// `discriminant_analysis.py:351,359`).
    ///
    /// `None` (sklearn's default) ⇒ the empirical priors `n_k / n` are inferred
    /// from the training data at fit time. `Some(p)` ⇒ `p` is used VERBATIM as
    /// `priors_` (matching sklearn `:605`, `self.priors_ = xp.asarray(self.priors)`).
    priors: Option<Array1<F>>,

    /// Solver selector (sklearn's `solver`, default `"svd"`,
    /// `discriminant_analysis.py:204,349`). See [`Solver`].
    solver: Solver,

    /// Covariance-shrinkage selector (sklearn's `shrinkage`, default `None`,
    /// `discriminant_analysis.py:218,350`). See [`Shrinkage`]. Only honored by
    /// the `lsqr` solver here; combined with `Solver::Svd` it is rejected at fit
    /// (sklearn `NotImplementedError`, `:628-629`).
    shrinkage: Shrinkage<F>,

    /// Whether to compute and store the shared within-class covariance matrix
    /// `covariance_` during fit (sklearn's `store_covariance`, default `false`,
    /// `discriminant_analysis.py:353,361`). When `true`, the svd-solver `fit`
    /// computes `covariance_ = Σ_k priors_[k] · cov(X_k)` (`:509-510`,
    /// `_class_cov` `:128-172`).
    store_covariance: bool,

    /// Singular-value rank threshold used by the svd-solver (sklearn's `tol`,
    /// default `1e-4`, `discriminant_analysis.py:354,362`). It drives the two
    /// rank cutoffs `rank = Σ(S > tol)` (`:532`) and
    /// `rank2 = Σ(S2 > tol·S2[0])` (`:554`).
    tol: F,

    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> LDA<F> {
    /// Create a new `LDA`.
    ///
    /// - `n_components`: number of discriminant directions to retain.
    ///   Pass `None` to use `min(n_classes - 1, n_features)`.
    #[must_use]
    pub fn new(n_components: Option<usize>) -> Self {
        Self {
            n_components,
            priors: None,
            solver: Solver::Svd,
            shrinkage: Shrinkage::None,
            store_covariance: false,
            // sklearn default `tol=1e-4` (`discriminant_analysis.py:354`).
            // `1e-4` is exactly representable in f32/f64; the fallback to
            // `F::epsilon()` is unreachable for those but keeps `new`
            // panic-free for any conforming `Float`.
            tol: F::from(1e-4).unwrap_or_else(F::epsilon),
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the configured number of components (may be `None`).
    #[must_use]
    pub fn n_components(&self) -> Option<usize> {
        self.n_components
    }

    /// Set the class prior probabilities (sklearn's `priors`,
    /// `discriminant_analysis.py:351,359`).
    ///
    /// The provided vector is used VERBATIM as `priors_` (sklearn does not
    /// normalize it here when it already sums to 1, `:605`). Its length must
    /// equal the number of classes seen at fit time, or [`Fit::fit`] returns
    /// [`FerroError::ShapeMismatch`]. Pass nothing (the `None` default) to infer
    /// the empirical priors `n_k / n` from the training data (`:601-603`).
    #[must_use]
    pub fn with_priors(mut self, priors: Array1<F>) -> Self {
        self.priors = Some(priors);
        self
    }

    /// Return the configured class priors (`None` ⇒ empirical at fit time).
    /// Mirrors sklearn's constructor `priors` (`discriminant_analysis.py:359`).
    #[must_use]
    pub fn priors(&self) -> Option<&Array1<F>> {
        self.priors.as_ref()
    }

    /// Set the solver (sklearn's `solver`, `discriminant_analysis.py:204,349`).
    /// Default [`Solver::Svd`]. See [`Solver`].
    ///
    /// [`Solver::Lsqr`] enables the least-squares path (and `shrinkage`);
    /// [`Solver::Eigen`] is not yet implemented (#596) and makes [`Fit::fit`]
    /// return a [`FerroError`].
    #[must_use]
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Return the configured solver. Mirrors sklearn's constructor `solver`
    /// (`discriminant_analysis.py:349`, default `"svd"`).
    #[must_use]
    pub fn solver(&self) -> Solver {
        self.solver
    }

    /// Set the covariance shrinkage (sklearn's `shrinkage`,
    /// `discriminant_analysis.py:218,350`). Default [`Shrinkage::None`]. See
    /// [`Shrinkage`].
    ///
    /// Honored only by [`Solver::Lsqr`] here (sklearn note `:225`: shrinkage
    /// works only with the `lsqr`/`eigen` solvers). Combined with
    /// [`Solver::Svd`], [`Fit::fit`] returns a [`FerroError`] mirroring sklearn's
    /// `NotImplementedError("shrinkage not supported with 'svd' solver.")`
    /// (`:628-629`). [`Shrinkage::Fixed`]`(s)` requires `0 <= s <= 1` (sklearn
    /// `Interval(Real, 0, 1, closed="both")`, `:339`), else [`Fit::fit`] returns
    /// [`FerroError::InvalidParameter`].
    #[must_use]
    pub fn with_shrinkage(mut self, shrinkage: Shrinkage<F>) -> Self {
        self.shrinkage = shrinkage;
        self
    }

    /// Return the configured covariance shrinkage. Mirrors sklearn's constructor
    /// `shrinkage` (`discriminant_analysis.py:350`, default `None`).
    #[must_use]
    pub fn shrinkage(&self) -> Shrinkage<F> {
        self.shrinkage
    }

    /// Set whether to compute and store the shared within-class covariance
    /// matrix `covariance_` during fit (sklearn's `store_covariance`,
    /// `discriminant_analysis.py:353,361`). Default `false`.
    ///
    /// When `true`, [`Fit::fit`] computes `covariance_ = Σ_k priors_[k] ·
    /// cov(X_k)` (`:509-510`, `_class_cov` `:128-172`) and
    /// [`FittedLDA::covariance`] returns `Some`. When `false` it returns `None`
    /// (matching sklearn, where the attribute only exists when the flag is set).
    #[must_use]
    pub fn with_store_covariance(mut self, store_covariance: bool) -> Self {
        self.store_covariance = store_covariance;
        self
    }

    /// Return whether `covariance_` will be stored during fit (sklearn's
    /// `store_covariance`, `discriminant_analysis.py:353`).
    #[must_use]
    pub fn store_covariance(&self) -> bool {
        self.store_covariance
    }

    /// Set the singular-value rank threshold `tol` used by the svd-solver
    /// (sklearn's `tol`, `discriminant_analysis.py:354,362`). Default `1e-4`.
    ///
    /// It drives the two rank cutoffs `rank = Σ(S > tol)` (`:532`) and
    /// `rank2 = Σ(S2 > tol·S2[0])` (`:554`).
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Return the configured svd-solver rank threshold `tol`. Mirrors sklearn's
    /// constructor `tol` (`discriminant_analysis.py:354`, default `1e-4`).
    #[must_use]
    pub fn tol(&self) -> F {
        self.tol
    }
}

impl<F: Float + Send + Sync + 'static> Default for LDA<F> {
    fn default() -> Self {
        Self::new(None)
    }
}

// ---------------------------------------------------------------------------
// FittedLDA
// ---------------------------------------------------------------------------

/// A fitted LDA model (sklearn's `svd` solver).
///
/// Created by calling [`Fit::fit`] on an [`LDA`]. Implements:
/// - [`Transform<Array2<F>>`] — project data via `(X - xbar_) @ scalings_`.
/// - [`Predict<Array2<F>>`] — classify by argmax of the affine
///   `decision_function`.
#[derive(Debug, Clone)]
pub struct FittedLDA<F> {
    /// Whitened projection matrix `scalings_`, shape `(n_features, rank2)`.
    /// Mirrors sklearn's `scalings_` (`discriminant_analysis.py:555`).
    scalings: Array2<F>,

    /// Per-class means in the ORIGINAL feature space, shape
    /// `(n_classes, n_features)`. Mirrors sklearn's `means_`
    /// (`discriminant_analysis.py:508`).
    means: Array2<F>,

    /// Weighted overall mean `xbar_ = priors_ @ means_`, length `n_features`.
    /// Mirrors sklearn's `xbar_` (`discriminant_analysis.py:517`).
    xbar: Array1<F>,

    /// Resolved class priors `priors_`, length `n_classes`. Empirical `n_k / n`
    /// when the constructor `priors` was `None`, else the provided `priors`
    /// verbatim. Mirrors sklearn's `priors_` (`discriminant_analysis.py:601-605`).
    priors: Array1<F>,

    /// Affine classifier coefficients `coef_`, shape `(n_classes, n_features)`.
    /// Mirrors sklearn's `coef_` (`discriminant_analysis.py:558`). (Binary
    /// collapse to `(1, n_features)` pends #600.)
    coef: Array2<F>,

    /// Affine classifier intercepts `intercept_`, length `n_classes` (embeds
    /// `log(priors_)`). Mirrors sklearn's `intercept_`
    /// (`discriminant_analysis.py:557,559`).
    intercept: Array1<F>,

    /// Ratio of explained variance per discriminant direction, length
    /// `max_components`. Mirrors sklearn's `explained_variance_ratio_`
    /// (`discriminant_analysis.py:550-552`).
    explained_variance_ratio: Array1<F>,

    /// Shared within-class covariance matrix `covariance_`, shape
    /// `(n_features, n_features)`, present only when the model was configured
    /// with [`LDA::with_store_covariance`]`(true)`. Mirrors sklearn's
    /// `covariance_` (`discriminant_analysis.py:509-510`, `_class_cov`
    /// `:128-172`): `Σ_k priors_[k] · cov(X_k)`. `None` otherwise (matching
    /// sklearn, where the attribute only exists when `store_covariance=True`).
    covariance: Option<Array2<F>>,

    /// Class labels corresponding to rows of `means`/`coef`.
    classes: Vec<usize>,

    /// Number of components to keep on `transform` output (sklearn's
    /// `_max_components`, `discriminant_analysis.py:619/625`).
    max_components: usize,

    /// Number of features seen during fitting.
    n_features: usize,
}

impl<F: Float + Send + Sync + 'static> FittedLDA<F> {
    /// Whitened projection (`scalings_`) matrix, shape `(n_features, rank2)`.
    /// Mirrors sklearn's `scalings_` (`discriminant_analysis.py:555`).
    #[must_use]
    pub fn scalings(&self) -> &Array2<F> {
        &self.scalings
    }

    /// Per-class means in the original feature space, shape
    /// `(n_classes, n_features)`. Mirrors sklearn's `means_`
    /// (`discriminant_analysis.py:508`).
    #[must_use]
    pub fn means(&self) -> &Array2<F> {
        &self.means
    }

    /// Weighted overall mean `xbar_`, length `n_features`. Mirrors sklearn's
    /// `xbar_` (`discriminant_analysis.py:517`).
    #[must_use]
    pub fn xbar(&self) -> &Array1<F> {
        &self.xbar
    }

    /// Resolved class priors `priors_`, length `n_classes` (empirical `n_k / n`
    /// when the constructor `priors` was `None`, else the provided `priors`
    /// verbatim). Mirrors sklearn's `priors_` (`discriminant_analysis.py:601-605`).
    #[must_use]
    pub fn priors(&self) -> &Array1<F> {
        &self.priors
    }

    /// Affine classifier coefficients `coef_`, shape `(n_classes, n_features)`.
    /// Mirrors sklearn's `coef_` (`discriminant_analysis.py:558`).
    #[must_use]
    pub fn coef(&self) -> &Array2<F> {
        &self.coef
    }

    /// Affine classifier intercepts `intercept_`, length `n_classes`. Mirrors
    /// sklearn's `intercept_` (`discriminant_analysis.py:557,559`).
    #[must_use]
    pub fn intercept(&self) -> &Array1<F> {
        &self.intercept
    }

    /// Explained-variance ratio per discriminant direction. Mirrors sklearn's
    /// `explained_variance_ratio_` (`discriminant_analysis.py:550-552`).
    #[must_use]
    pub fn explained_variance_ratio(&self) -> &Array1<F> {
        &self.explained_variance_ratio
    }

    /// Shared within-class covariance matrix `covariance_`, shape
    /// `(n_features, n_features)`. Mirrors sklearn's `covariance_`
    /// (`discriminant_analysis.py:509-510`, `_class_cov` `:128-172`):
    /// `Σ_k priors_[k] · cov(X_k)` where `cov(X_k)` is the maximum-likelihood
    /// empirical covariance of class `k`'s samples (`np.cov(..., bias=1)`,
    /// normalized by `n_k`, via `empirical_covariance`,
    /// `covariance/_empirical_covariance.py:109`).
    ///
    /// Returns `Some` only when the model was configured with
    /// [`LDA::with_store_covariance`]`(true)`; `None` otherwise — matching
    /// sklearn, where the `covariance_` attribute only exists when
    /// `store_covariance=True`.
    #[must_use]
    pub fn covariance(&self) -> Option<&Array2<F>> {
        self.covariance.as_ref()
    }

    /// Sorted class labels as seen during fitting.
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Per-class discriminant scores. Mirrors sklearn
    /// `LinearDiscriminantAnalysis.decision_function` (the `LinearClassifierMixin`,
    /// `discriminant_analysis.py:739`): the affine map `X @ coef_.T + intercept_`.
    ///
    /// Returns shape `(n_samples, n_classes)`. (Binary collapse to `(n,)` pends
    /// REQ-14/#600.) argmax of each row agrees with [`Predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the fitted model.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedLDA::decision_function".into(),
            });
        }
        // X @ coef_.T + intercept_  (coef_ is (n_classes, n_features)).
        let mut out = x.dot(&self.coef.t());
        let n_classes = self.intercept.len();
        for mut row in out.rows_mut() {
            for c in 0..n_classes {
                row[c] = row[c] + self.intercept[c];
            }
        }
        Ok(out)
    }

    /// Predict per-class probabilities. Mirrors sklearn
    /// `LinearDiscriminantAnalysis.predict_proba` (`discriminant_analysis.py:706-711`):
    /// the multiclass `softmax(decision_function)` (the row-max-shifted softmax
    /// of `sklearn.utils.extmath.softmax`, `extmath.py:949-985`).
    ///
    /// Returns shape `(n_samples, n_classes)`; rows sum to 1. (The binary
    /// `[1-expit(d), expit(d)]` collapse pends REQ-14/#600; the multiclass
    /// softmax here is correct for `n_classes >= 2` because `coef_`/`intercept_`
    /// are not yet collapsed to the binary single-row form.)
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the model.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let decision = self.decision_function(x)?;
        let n_samples = decision.nrows();
        let n_classes = decision.ncols();
        let mut proba = Array2::<F>::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let max_l = (0..n_classes)
                .map(|c| decision[[i, c]])
                .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
            let mut sum_exp = F::zero();
            for c in 0..n_classes {
                let e = (decision[[i, c]] - max_l).exp();
                proba[[i, c]] = e;
                sum_exp = sum_exp + e;
            }
            for c in 0..n_classes {
                proba[[i, c]] = proba[[i, c]] / sum_exp;
            }
        }
        Ok(proba)
    }

    /// Element-wise log of [`predict_proba`](Self::predict_proba). Mirrors
    /// sklearn `predict_log_proba` exactly (`discriminant_analysis.py:713-737`):
    /// entries that are EXACTLY `0.0` are bumped by the dtype's
    /// `smallest_normal` (`f32`/`f64::MIN_POSITIVE`) before taking `log`
    /// (`:729-736`), so `log(0)` becomes `log(MIN_POSITIVE)` rather than `-inf`;
    /// every nonzero probability keeps its true `ln`.
    ///
    /// # Errors
    ///
    /// Forwards any error from [`predict_proba`](Self::predict_proba).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let proba = self.predict_proba(x)?;
        // sklearn: prediction[prediction == 0.0] += smallest_normal; log(prediction).
        // `F::min_positive_value()` is numpy's `finfo(dtype).smallest_normal`
        // (`f64::MIN_POSITIVE` ≈ 2.2250738585072014e-308).
        let smallest_normal = F::min_positive_value();
        Ok(proba.mapv(|p| {
            if p == F::zero() {
                (p + smallest_normal).ln()
            } else {
                p.ln()
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a `usize` count to `F` without panicking. Returns
/// [`FerroError::NumericalInstability`] if the value is not representable.
#[inline]
fn usize_to_f<F: Float>(v: usize) -> Result<F, FerroError> {
    F::from(v).ok_or_else(|| FerroError::NumericalInstability {
        message: format!("could not represent count {v} as the float type"),
    })
}

/// `0.5` as `F`, built panic-free from `1 / (1 + 1)` (exact for binary floats).
#[inline]
fn half<F: Float>() -> F {
    F::one() / (F::one() + F::one())
}

/// Singular values `S` and right singular vectors transposed `Vt` of the thin
/// SVD `A = U·diag(S)·Vt` (`full_matrices=False`), on the ferray substrate
/// (`ferray::linalg::svd`, the analog of `scipy.linalg.svd(X,
/// full_matrices=False)`, `discriminant_analysis.py:530,545`). Mirrors the
/// bridging pattern in `qda.rs::svd_s_vt` / `bayesian_ridge.rs::svd_thin`
/// (R-SUBSTRATE-4): the caller keeps its `ndarray` signature and the
/// ndarray↔ferray conversion happens here.
///
/// Returns `(S, Vt)` with `S` of length `k = min(m, n)` (descending) and `Vt`
/// of shape `(k, n)`.
///
/// # Errors
///
/// Returns [`FerroError::NumericalInstability`] if the ferray array build or
/// the SVD itself fails.
fn svd_s_vt<F: LinalgFloat>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError> {
    let (m, n) = a.dim();
    let a_flat: Vec<F> = a.iter().copied().collect();
    let fa =
        FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([m, n]), a_flat).map_err(|e| {
            FerroError::NumericalInstability {
                message: format!("ferray svd: failed to build matrix: {e}"),
            }
        })?;
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

/// Maximum-likelihood empirical covariance of `xg` (rows = samples), centered on
/// the per-column mean and normalized by `n` (NOT `n-1`). Mirrors sklearn's
/// `empirical_covariance` / `np.cov(..., bias=1)` (`discriminant_analysis.py:77`,
/// `covariance/_empirical_covariance.py:109`).
///
/// Returns the `(p, p)` covariance and the per-column means (length `p`).
fn empirical_covariance<F: Float>(xg: &Array2<F>) -> Result<(Array2<F>, Array1<F>), FerroError> {
    let (n, p) = xg.dim();
    let nf = usize_to_f::<F>(n)?;
    let mut mean = Array1::<F>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] = mean[j] + xg[[i, j]];
        }
    }
    for j in 0..p {
        mean[j] = mean[j] / nf;
    }
    let mut cov = Array2::<F>::zeros((p, p));
    for a in 0..p {
        for b in 0..p {
            let mut acc = F::zero();
            for i in 0..n {
                acc = acc + (xg[[i, a]] - mean[a]) * (xg[[i, b]] - mean[b]);
            }
            cov[[a, b]] = acc / nf;
        }
    }
    Ok((cov, mean))
}

/// Ledoit-Wolf analytical shrinkage coefficient of `x` (rows = samples), the
/// transcription of sklearn's `ledoit_wolf_shrinkage`
/// (`covariance/_shrunk_covariance.py:299-402`) for the unblocked case
/// (`block_size=1000 >> n_features`, so `n_splits = 0` and the blocked loops
/// collapse to the single tail term `beta_ = Σ(X²ᵀ·X²)`, `delta_ =
/// Σ((Xᵀ·X)²)`). `x` is assumed already centered (`assume_centered=True`, the
/// caller centers in [`cov_shrunk`]).
///
/// Formula (`:365-401`): with `X² = x⊙x`, `emp_cov_trace = Σ_i X²[i,:]/n`,
/// `mu = Σ(emp_cov_trace)/p`, `beta_ = Σ(X²ᵀ·X²)`, `delta_ = Σ((Xᵀ·X)²)/n²`,
/// `beta = (1/(p·n))·(beta_/n − delta_)`,
/// `delta = (delta_ − 2·mu·Σ(emp_cov_trace) + p·mu²)/p`,
/// `beta = min(beta, delta)`, `shrinkage = 0 if beta==0 else beta/delta`.
fn ledoit_wolf_shrinkage<F: Float>(x: &Array2<F>) -> Result<F, FerroError> {
    let (n, p) = x.dim();
    // sklearn `:345-346`: for a single feature the result is shrinkage-invariant.
    if p == 1 {
        return Ok(F::zero());
    }
    let nf = usize_to_f::<F>(n)?;
    let pf = usize_to_f::<F>(p)?;

    // emp_cov_trace[j] = Σ_i x[i,j]² / n   (:365-366)
    let mut emp_cov_trace = Array1::<F>::zeros(p);
    for j in 0..p {
        let mut acc = F::zero();
        for i in 0..n {
            acc = acc + x[[i, j]] * x[[i, j]];
        }
        emp_cov_trace[j] = acc / nf;
    }
    // mu = Σ(emp_cov_trace) / p   (:367)
    let mut trace_sum = F::zero();
    for j in 0..p {
        trace_sum = trace_sum + emp_cov_trace[j];
    }
    let mu = trace_sum / pf;

    // beta_ = Σ over (a,b) of (X²ᵀ·X²)[a,b] = Σ_{a,b} Σ_i x[i,a]²·x[i,b]²  (:388-390)
    // delta_ = Σ over (a,b) of ((Xᵀ·X)[a,b])²  (:384-386)
    let mut beta_acc = F::zero();
    let mut delta_acc = F::zero();
    for a in 0..p {
        for b in 0..p {
            let mut g_ab = F::zero(); // (Xᵀ·X)[a,b] = Σ_i x[i,a]·x[i,b]
            let mut h_ab = F::zero(); // (X²ᵀ·X²)[a,b] = Σ_i x[i,a]²·x[i,b]²
            for i in 0..n {
                let xa = x[[i, a]];
                let xb = x[[i, b]];
                g_ab = g_ab + xa * xb;
                h_ab = h_ab + (xa * xa) * (xb * xb);
            }
            beta_acc = beta_acc + h_ab;
            delta_acc = delta_acc + g_ab * g_ab;
        }
    }
    // delta_ /= n²   (:387)
    let delta_ = delta_acc / (nf * nf);
    // beta = (1/(p·n)) · (beta_/n − delta_)   (:392)
    let beta = (F::one() / (pf * nf)) * (beta_acc / nf - delta_);
    // delta = (delta_ − 2·mu·Σ(emp_cov_trace) + p·mu²) / p   (:394-395)
    let two = F::one() + F::one();
    let mut delta = delta_ - two * mu * trace_sum + pf * mu * mu;
    delta = delta / pf;
    // beta = min(beta, delta)   (:399)
    let beta = if beta < delta { beta } else { delta };
    // shrinkage = 0 if beta==0 else beta/delta   (:401)
    if beta == F::zero() {
        Ok(F::zero())
    } else {
        Ok(beta / delta)
    }
}

/// Per-class covariance estimate with optional shrinkage — sklearn's `_cov`
/// (`discriminant_analysis.py:36-93`) for `covariance_estimator=None`:
/// - [`Shrinkage::None`] → empirical maximum-likelihood covariance (`:76-77`).
/// - [`Shrinkage::Fixed`]`(s)` → `shrunk_covariance(emp_cov, s)`
///   (`:78-79`, `_shrunk_covariance.py:153-156`):
///   `(1 − s)·emp_cov + s·(trace(emp_cov)/p)·I`.
/// - [`Shrinkage::Auto`] → Ledoit-Wolf on the StandardScaler-standardized data,
///   then rescaled (`:70-75`): standardize `Xs = (X − mean)/scale` (population
///   std, `ddof=0`, zeros → 1), `s = ledoit_wolf(Xs)`, then `cov[a,b] =
///   scale[a]·s[a,b]·scale[b]`.
fn cov_shrunk<F: Float>(xg: &Array2<F>, shrinkage: Shrinkage<F>) -> Result<Array2<F>, FerroError> {
    let (n, p) = xg.dim();
    match shrinkage {
        Shrinkage::None => {
            let (cov, _mean) = empirical_covariance(xg)?;
            Ok(cov)
        }
        Shrinkage::Fixed(s) => {
            let (emp, _mean) = empirical_covariance(xg)?;
            let pf = usize_to_f::<F>(p)?;
            let mut trace = F::zero();
            for j in 0..p {
                trace = trace + emp[[j, j]];
            }
            let mu = trace / pf;
            let mut out = Array2::<F>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    let diag = if a == b { mu } else { F::zero() };
                    out[[a, b]] = (F::one() - s) * emp[[a, b]] + s * diag;
                }
            }
            Ok(out)
        }
        Shrinkage::Auto => {
            // StandardScaler: center + divide by POPULATION std (ddof=0); zeros
            // replaced by 1.0 (sklearn StandardScaler `_handle_zeros_in_scale`).
            let nf = usize_to_f::<F>(n)?;
            let mut mean = Array1::<F>::zeros(p);
            for i in 0..n {
                for j in 0..p {
                    mean[j] = mean[j] + xg[[i, j]];
                }
            }
            for j in 0..p {
                mean[j] = mean[j] / nf;
            }
            let mut scale = Array1::<F>::zeros(p);
            for j in 0..p {
                let mut var = F::zero();
                for i in 0..n {
                    let d = xg[[i, j]] - mean[j];
                    var = var + d * d;
                }
                var = var / nf;
                let sd = var.sqrt();
                scale[j] = if sd == F::zero() { F::one() } else { sd };
            }
            let mut xs = Array2::<F>::zeros((n, p));
            for i in 0..n {
                for j in 0..p {
                    xs[[i, j]] = (xg[[i, j]] - mean[j]) / scale[j];
                }
            }
            // ledoit_wolf re-centers; Xs already has ~0 mean, but follow sklearn
            // exactly and re-center (assume_centered=False, `:357-358`).
            let mut xs_mean = Array1::<F>::zeros(p);
            for i in 0..n {
                for j in 0..p {
                    xs_mean[j] = xs_mean[j] + xs[[i, j]];
                }
            }
            for j in 0..p {
                xs_mean[j] = xs_mean[j] / nf;
            }
            let mut xc = Array2::<F>::zeros((n, p));
            for i in 0..n {
                for j in 0..p {
                    xc[[i, j]] = xs[[i, j]] - xs_mean[j];
                }
            }
            // shrinkage coefficient from the (centered) standardized data.
            let shr = ledoit_wolf_shrinkage(&xc)?;
            // emp_cov of the standardized data = Xcᵀ·Xc / n, then shrink:
            // (1 − shr)·emp + shr·(trace(emp)/p)·I  (`_shrunk_covariance.py`).
            let (emp, _m) = empirical_covariance(&xs)?;
            let pf = usize_to_f::<F>(p)?;
            let mut trace = F::zero();
            for j in 0..p {
                trace = trace + emp[[j, j]];
            }
            let mu = trace / pf;
            // rescale: cov[a,b] = scale[a] · shrunk[a,b] · scale[b]  (`:75`).
            let mut out = Array2::<F>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    let diag = if a == b { mu } else { F::zero() };
                    let shrunk = (F::one() - shr) * emp[[a, b]] + shr * diag;
                    out[[a, b]] = scale[a] * shrunk * scale[b];
                }
            }
            Ok(out)
        }
    }
}

/// Solve the multi-RHS least-squares problem `A @ x = b` (with `b` having
/// `nrhs` columns) through [`ferray::linalg::lstsq`]
/// (`ferray-linalg/src/solve.rs:208`, the LAPACK-`gelsd`-equivalent single-SVD
/// min-norm solver), bridging ndarray↔ferray at this boundary (R-SUBSTRATE-4),
/// mirroring the bridge in `linalg.rs::solve_lstsq`. Returns the `(n, nrhs)`
/// solution. Used by `_solve_lstsq` to compute `coef_ = lstsq(covariance_,
/// means_.T)[0].T` (`discriminant_analysis.py:416`).
///
/// `rcond` is `Some(F::epsilon())`, pinning the singular-value cutoff to scipy's
/// `cond=eps` default (matching `linalg.lstsq` `:416`), as in `solve_lstsq`.
fn lstsq_multi<F: LinalgFloat>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>, FerroError> {
    let (m, n) = a.dim();
    let (bm, nrhs) = b.dim();
    if bm != m {
        return Err(FerroError::ShapeMismatch {
            expected: vec![m, nrhs],
            actual: vec![bm, nrhs],
            context: "LDA lsqr: covariance/means row mismatch".into(),
        });
    }
    let a_flat: Vec<F> = a.iter().copied().collect();
    let fa =
        FerrayArray::<F, FerrayIx2>::from_vec(FerrayIx2::new([m, n]), a_flat).map_err(|e| {
            FerroError::NumericalInstability {
                message: format!("ferray lstsq: failed to build matrix A: {e}"),
            }
        })?;
    let b_flat: Vec<F> = b.iter().copied().collect();
    let fb = FerrayArray::<F, ferray::IxDyn>::from_vec(ferray::IxDyn::new(&[bm, nrhs]), b_flat)
        .map_err(|e| FerroError::NumericalInstability {
            message: format!("ferray lstsq: failed to build RHS B: {e}"),
        })?;
    let (sol, _residuals, _rank, _singular) = ferray::linalg::lstsq(&fa, &fb, Some(F::epsilon()))
        .map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray lstsq solve failed: {e}"),
    })?;
    let sol_shape = sol.shape();
    let out = Array2::from_shape_vec((sol_shape[0], sol_shape[1]), sol.iter().copied().collect())
        .map_err(|e| FerroError::NumericalInstability {
        message: format!("ferray lstsq: solution shape conversion failed: {e}"),
    })?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fit (sklearn _solve_svd)
// ---------------------------------------------------------------------------

impl<F: LinalgFloat + ScalarOperand> Fit<Array2<F>, Array1<usize>> for LDA<F> {
    type Fitted = FittedLDA<F>;
    type Error = FerroError;

    /// Fit the LDA model via sklearn's default `solver="svd"` path
    /// (`discriminant_analysis.py:487-559`): two SVDs whiten the within-class
    /// data and project onto the between-class subspace, yielding `scalings_`,
    /// `xbar_`, `coef_`, `intercept_` (embedding `log(priors_)`), and
    /// `explained_variance_ratio_`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if fewer than 2 samples / classes.
    /// - [`FerroError::InvalidParameter`] if `n_components` is zero or exceeds
    ///   `min(n_classes - 1, n_features)`.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row counts.
    /// - [`FerroError::NumericalInstability`] if an SVD fails.
    #[allow(
        clippy::needless_range_loop,
        reason = "explicit index loops mirror sklearn's broadcasting per-column/per-class"
    )]
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedLDA<F>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "LDA: y length must match number of rows in X".into(),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LDA requires at least 2 samples".into(),
            });
        }

        // Sorted unique classes (sklearn `classes_ = unique_labels(y)`, :592).
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_classes,
                context: "LDA requires at least 2 distinct classes".into(),
            });
        }
        // sklearn rejects n_samples == n_classes (:596-599).
        if n_samples == n_classes {
            return Err(FerroError::InsufficientSamples {
                required: n_classes + 1,
                actual: n_samples,
                context: "LDA: number of samples must exceed number of classes".into(),
            });
        }

        // _max_components (sklearn :614-625).
        let max_components = (n_classes - 1).min(n_features);
        let user_max = match self.n_components {
            None => max_components,
            Some(0) => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: "must be at least 1".into(),
                });
            }
            Some(k) if k > max_components => {
                return Err(FerroError::InvalidParameter {
                    name: "n_components".into(),
                    reason: format!(
                        "n_components ({k}) cannot be larger than min(n_features, n_classes - 1) = {max_components}"
                    ),
                });
            }
            Some(k) => k,
        };

        let n_f = usize_to_f::<F>(n_samples)?;

        // --- per-class means_ and class indices (sklearn `_class_means`) ------
        let mut means = Array2::<F>::zeros((n_classes, n_features));
        let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        let mut class_pos = std::collections::HashMap::new();
        for (idx, &cls) in classes.iter().enumerate() {
            class_pos.insert(cls, idx);
        }
        for (i, &label) in y.iter().enumerate() {
            if let Some(&idx) = class_pos.get(&label) {
                class_indices[idx].push(i);
            }
        }
        for (idx, indices) in class_indices.iter().enumerate() {
            if indices.is_empty() {
                return Err(FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: format!("LDA: class {} has no samples", classes[idx]),
                });
            }
            let cnt_f = usize_to_f::<F>(indices.len())?;
            for &i in indices {
                for j in 0..n_features {
                    means[[idx, j]] += x[[i, j]];
                }
            }
            for j in 0..n_features {
                means[[idx, j]] /= cnt_f;
            }
        }

        // --- priors_  (sklearn :601-605) --------------------------------------
        // `priors=None` (default) ⇒ empirical `n_k / n` inferred from the data
        // (`:601-603`). `Some(p)` ⇒ `p` used VERBATIM (`:605`,
        // `self.priors_ = xp.asarray(self.priors)`). sklearn would mis-index a
        // wrong-length array, so reject it up front (R-DEV-4 length check).
        let priors = match &self.priors {
            None => {
                let mut priors = Array1::<F>::zeros(n_classes);
                for idx in 0..n_classes {
                    priors[idx] = usize_to_f::<F>(class_indices[idx].len())? / n_f;
                }
                priors
            }
            Some(p) => {
                if p.len() != n_classes {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n_classes],
                        actual: vec![p.len()],
                        context: "LDA: priors length must match number of classes".into(),
                    });
                }
                let mut p = p.clone();
                // sklearn rejects negative priors (:607-608,
                // `if xp.any(self.priors_ < 0): raise ValueError("priors must
                // be non-negative")`).
                if p.iter().any(|&v| v < <F as num_traits::Zero>::zero()) {
                    return Err(FerroError::InvalidParameter {
                        name: "priors".into(),
                        reason: "priors must be non-negative".into(),
                    });
                }
                // sklearn renormalizes (with a UserWarning) when the priors do
                // not sum to 1 (:610-612, `if xp.abs(xp.sum(self.priors_) - 1.0)
                // > 1e-5: warnings.warn(...); self.priors_ = self.priors_ /
                // self.priors_.sum()`). FerroError has no warning channel; the
                // crate emits warnings via `eprintln!` (cf. qda.rs collinearity
                // warning, `discriminant_analysis.py:947`). The observable
                // contract is the renormalized `priors_`.
                let s = p.sum();
                let tol_sum = F::from(1e-5).unwrap_or_else(F::epsilon);
                if (s - <F as num_traits::One>::one()).abs() > tol_sum {
                    eprintln!("The priors do not sum to 1. Renormalizing");
                    for v in p.iter_mut() {
                        *v /= s;
                    }
                }
                p
            }
        };

        // --- solver dispatch (sklearn :627-650) -------------------------------
        // `lsqr` and `eigen` need only means_/priors_/class_indices; resolve
        // them above this point, then branch. `svd` falls through to the
        // existing two-SVD path below (BYTE-IDENTICAL — `shrinkage` must be
        // `None` for svd, sklearn `NotImplementedError` `:628-629`).
        match self.solver {
            Solver::Lsqr => {
                return self.solve_lstsq(
                    x,
                    &classes,
                    &class_indices,
                    &means,
                    &priors,
                    user_max,
                    n_features,
                );
            }
            Solver::Eigen => {
                // The generalized-eigenvalue solver is not yet implemented
                // (open prereq blocker #596); the variant exists so the enum
                // is complete, but fit errors rather than silently mis-solving.
                return Err(FerroError::InvalidParameter {
                    name: "solver".into(),
                    reason: "eigen solver not yet implemented (#596)".into(),
                });
            }
            Solver::Svd => {
                // sklearn: svd + shrinkage != None → NotImplementedError
                // ("shrinkage not supported with 'svd' solver.", `:628-629`).
                if !matches!(self.shrinkage, Shrinkage::None) {
                    return Err(FerroError::InvalidParameter {
                        name: "shrinkage".into(),
                        reason: "shrinkage not supported with svd solver".into(),
                    });
                }
            }
        }

        // --- xbar_ = priors_ @ means_  (sklearn :517) -------------------------
        let mut xbar = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let mut acc = <F as num_traits::Zero>::zero();
            for idx in 0..n_classes {
                acc += priors[idx] * means[[idx, j]];
            }
            xbar[j] = acc;
        }

        // --- covariance_  (sklearn :509-510, `_class_cov` :128-172) -----------
        // When `store_covariance` is set, compute the shared within-class
        // covariance `Σ_k priors_[k] · cov(X_k)`, where `cov(X_k)` is the
        // MAXIMUM-LIKELIHOOD empirical covariance of class k's samples —
        // `empirical_covariance` calls `np.cov(Xg.T, bias=1)`
        // (`covariance/_empirical_covariance.py:109`), i.e. centered on the
        // class mean and normalized by `n_k` (NOT `n_k - 1`). Verified against
        // the live oracle: class-0 of the dispatch fixture yields the documented
        // `[[0.4296875, …], …]` only under the `bias=1` (÷n_k) normalization.
        let covariance = if self.store_covariance {
            let mut cov = Array2::<F>::zeros((n_features, n_features));
            for (idx, indices) in class_indices.iter().enumerate() {
                let nk = usize_to_f::<F>(indices.len())?;
                let prior_k = priors[idx];
                // cov(X_k)[a, b] = (1/n_k) Σ_i (x_ia - μ_ka)(x_ib - μ_kb).
                for a in 0..n_features {
                    for b in 0..n_features {
                        let mut acc = <F as num_traits::Zero>::zero();
                        for &i in indices {
                            acc += (x[[i, a]] - means[[idx, a]]) * (x[[i, b]] - means[[idx, b]]);
                        }
                        cov[[a, b]] += prior_k * (acc / nk);
                    }
                }
            }
            Some(cov)
        } else {
            None
        };

        // --- Xc = each sample minus its class mean (stacked; sklearn :512-519) -
        let mut xc = Array2::<F>::zeros((n_samples, n_features));
        for (idx, indices) in class_indices.iter().enumerate() {
            for &i in indices {
                for j in 0..n_features {
                    xc[[i, j]] = x[[i, j]] - means[[idx, j]];
                }
            }
        }

        // --- std = population std of Xc per column (ddof=0; sklearn :522-524) --
        // numpy std: sqrt(mean((Xc - mean(Xc))^2)). Xc columns already have ~0
        // mean by construction, but follow numpy exactly (subtract the column
        // mean) for ULP fidelity.
        let mut std = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let mut col_mean = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                col_mean += xc[[i, j]];
            }
            col_mean /= n_f;
            let mut var = <F as num_traits::Zero>::zero();
            for i in 0..n_samples {
                let d = xc[[i, j]] - col_mean;
                var += d * d;
            }
            var /= n_f;
            let s = var.sqrt();
            std[j] = if s == <F as num_traits::Zero>::zero() {
                <F as num_traits::One>::one()
            } else {
                s
            };
        }

        // --- Xw = sqrt(1/(n-c)) * (Xc / std)  (sklearn :525-528) --------------
        let denom = usize_to_f::<F>(n_samples - n_classes)?;
        let fac_sqrt = (<F as num_traits::One>::one() / denom).sqrt();
        let mut xw = Array2::<F>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                xw[[i, j]] = fac_sqrt * (xc[[i, j]] / std[j]);
            }
        }

        // --- first SVD: within whitening (sklearn :530-534) -------------------
        // sklearn's svd-solver rank threshold `tol` (constructor default `1e-4`,
        // `discriminant_analysis.py:354,362`), now configurable via
        // `LDA::with_tol`. Default `1e-4` ⇒ byte-identical to the prior hardcode.
        let tol = self.tol;
        let (s1, vt1) = svd_s_vt::<F>(&xw)?;
        let rank1 = s1.iter().filter(|&&v| v > tol).count();
        if rank1 == 0 {
            return Err(FerroError::NumericalInstability {
                message: "LDA: within-class scatter has rank 0 (all features constant)".into(),
            });
        }
        // scalings = (Vt[:rank]/std).T / S[:rank]   -> (n_features, rank1)
        let mut scalings1 = Array2::<F>::zeros((n_features, rank1));
        for k in 0..rank1 {
            let sk = s1[k];
            for j in 0..n_features {
                scalings1[[j, k]] = (vt1[[k, j]] / std[j]) / sk;
            }
        }

        // --- between-class scaled centers (sklearn :535-541) ------------------
        // Xb[i] = sqrt(n * priors_[i] * fac2) * (means_[i] - xbar_)  then @ scalings.
        let fac2 = if n_classes == 1 {
            <F as num_traits::One>::one()
        } else {
            <F as num_traits::One>::one() / usize_to_f::<F>(n_classes - 1)?
        };
        let mut xb_centers = Array2::<F>::zeros((n_classes, n_features));
        for idx in 0..n_classes {
            let w = (n_f * priors[idx] * fac2).sqrt();
            for j in 0..n_features {
                xb_centers[[idx, j]] = w * (means[[idx, j]] - xbar[j]);
            }
        }
        let xb = xb_centers.dot(&scalings1); // (n_classes, rank1)

        // --- second SVD: between-class projection (sklearn :545-555) ----------
        let (s2, vt2) = svd_s_vt::<F>(&xb)?;

        // explained_variance_ratio_ = (S2^2 / sum(S2^2))[:max_components] (:550-552)
        let mut sum_sq = <F as num_traits::Zero>::zero();
        for &v in s2.iter() {
            sum_sq += v * v;
        }
        let evr_len = user_max.min(s2.len());
        let mut explained_variance_ratio = Array1::<F>::zeros(evr_len);
        for k in 0..evr_len {
            explained_variance_ratio[k] = if sum_sq > <F as num_traits::Zero>::zero() {
                (s2[k] * s2[k]) / sum_sq
            } else {
                <F as num_traits::Zero>::zero()
            };
        }

        // rank2 = sum(S2 > tol * S2[0])  (sklearn :554)
        let s2_0 = if s2.is_empty() {
            <F as num_traits::Zero>::zero()
        } else {
            s2[0]
        };
        let rank2 = s2.iter().filter(|&&v| v > tol * s2_0).count();
        if rank2 == 0 {
            return Err(FerroError::NumericalInstability {
                message: "LDA: between-class scatter has rank 0 (classes coincide)".into(),
            });
        }

        // scalings_ = scalings @ Vt2.T[:, :rank2]   -> (n_features, rank2)
        // Vt2 is (k2, rank1); Vt2.T is (rank1, k2); take first rank2 columns.
        let mut scalings = Array2::<F>::zeros((n_features, rank2));
        for j in 0..n_features {
            for c in 0..rank2 {
                let mut acc = <F as num_traits::Zero>::zero();
                for k in 0..rank1 {
                    // Vt2.T[k, c] = Vt2[c, k]
                    acc += scalings1[[j, k]] * vt2[[c, k]];
                }
                scalings[[j, c]] = acc;
            }
        }

        // --- coef_ / intercept_  (sklearn :556-559) ---------------------------
        // coef = (means_ - xbar_) @ scalings_     (n_classes, rank2)
        let mut centered_means = Array2::<F>::zeros((n_classes, n_features));
        for idx in 0..n_classes {
            for j in 0..n_features {
                centered_means[[idx, j]] = means[[idx, j]] - xbar[j];
            }
        }
        let coef_lowrank = centered_means.dot(&scalings); // (n_classes, rank2)

        // intercept_ = -0.5 * sum(coef^2, axis=1) + log(priors_)
        let neg_half = -half::<F>();
        let mut intercept = Array1::<F>::zeros(n_classes);
        for idx in 0..n_classes {
            let mut sq = <F as num_traits::Zero>::zero();
            for c in 0..rank2 {
                sq += coef_lowrank[[idx, c]] * coef_lowrank[[idx, c]];
            }
            intercept[idx] = neg_half * sq + priors[idx].ln();
        }

        // coef_ = coef @ scalings_.T              (n_classes, n_features)
        let coef = coef_lowrank.dot(&scalings.t());

        // intercept_ -= xbar_ @ coef_.T           (subtract per class)
        for idx in 0..n_classes {
            let mut dot = <F as num_traits::Zero>::zero();
            for j in 0..n_features {
                dot += xbar[j] * coef[[idx, j]];
            }
            intercept[idx] -= dot;
        }

        Ok(FittedLDA {
            scalings,
            means,
            xbar,
            priors,
            coef,
            intercept,
            explained_variance_ratio,
            covariance,
            classes,
            max_components: user_max,
            n_features,
        })
    }
}

// ---------------------------------------------------------------------------
// Fit (sklearn _solve_lstsq) — the lsqr solver
// ---------------------------------------------------------------------------

impl<F: LinalgFloat + ScalarOperand> LDA<F> {
    /// The least-squares solver (sklearn's `_solve_lstsq`,
    /// `discriminant_analysis.py:365-419`), dispatched from [`Fit::fit`] when
    /// [`Solver::Lsqr`] is selected.
    ///
    /// Computes (`:412-418`):
    /// - `covariance_ = Σ_k priors_[k] · cov(X_k)` where `cov(X_k)` applies the
    ///   configured [`Shrinkage`] to class `k`'s empirical covariance
    ///   (`_class_cov` `:128-172`, `_cov` `:36-93`).
    /// - `coef_ = lstsq(covariance_, means_.T)[0].T` (`:416`), via
    ///   [`ferray::linalg::lstsq`] (multi-RHS).
    /// - `intercept_ = -½·diag(means_ @ coef_.T) + log(priors_)` (`:417-418`).
    ///
    /// Unlike sklearn's `svd` solver, the lsqr `coef_` is the FULL-space
    /// discriminant `(n_classes, n_features)` — NO `scalings_`/`xbar_`/
    /// `explained_variance_ratio_` and NO `transform` (sklearn raises
    /// `NotImplementedError` for `transform` under lsqr, `:676-679`); here
    /// [`Transform`] returns an error because `scalings_` is the zero matrix /
    /// `xbar_` is zero, and `transform` slices to `max_components` of a meaningless
    /// projection — we instead document that `transform` is unsupported by
    /// recording `max_components = 0` so the projection is empty (mirroring
    /// sklearn's "dimensionality reduction is not supported" for lsqr).
    ///
    /// `covariance_` is ALWAYS populated for lsqr (sklearn `:413`, the attribute
    /// is set regardless of `store_covariance`), exposed via
    /// [`FittedLDA::covariance`].
    ///
    /// `decision_function`/`predict`/`predict_proba` work identically to the svd
    /// path because they consume `coef_`/`intercept_` only.
    ///
    /// NOTE: the binary-collapse of `coef_`/`intercept_` to a single row
    /// (sklearn `:651-657`) is NOT applied here, matching the existing svd path
    /// (open prereq blocker #600); `coef_` stays `(n_classes, n_features)`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if [`Shrinkage::Fixed`]`(s)` has
    ///   `s ∉ [0, 1]` (sklearn `Interval(Real, 0, 1, closed="both")`, `:339`).
    /// - [`FerroError::NumericalInstability`] if the least-squares solve fails.
    #[allow(
        clippy::too_many_arguments,
        reason = "the lsqr solver consumes the same pre-resolved fit state (classes/indices/means/priors/dims) the svd path computes; threading them avoids recomputation"
    )]
    fn solve_lstsq(
        &self,
        x: &Array2<F>,
        classes: &[usize],
        class_indices: &[Vec<usize>],
        means: &Array2<F>,
        priors: &Array1<F>,
        _user_max: usize,
        n_features: usize,
    ) -> Result<FittedLDA<F>, FerroError> {
        let n_classes = classes.len();

        // Validate Fixed(s): sklearn Interval(Real, 0, 1, closed="both") (:339).
        if let Shrinkage::Fixed(s) = self.shrinkage
            && (s < <F as num_traits::Zero>::zero() || s > <F as num_traits::One>::one())
        {
            return Err(FerroError::InvalidParameter {
                name: "shrinkage".into(),
                reason: "shrinkage float must be in [0, 1]".into(),
            });
        }

        // covariance_ = Σ_k priors_[k] · cov(X_k)  (sklearn _class_cov :167-172).
        let mut covariance = Array2::<F>::zeros((n_features, n_features));
        for (idx, indices) in class_indices.iter().enumerate() {
            // Gather class-k rows.
            let nk = indices.len();
            let mut xg = Array2::<F>::zeros((nk, n_features));
            for (r, &i) in indices.iter().enumerate() {
                for j in 0..n_features {
                    xg[[r, j]] = x[[i, j]];
                }
            }
            let cov_k = cov_shrunk(&xg, self.shrinkage)?;
            let prior_k = priors[idx];
            for a in 0..n_features {
                for b in 0..n_features {
                    covariance[[a, b]] += prior_k * cov_k[[a, b]];
                }
            }
        }

        // coef_ = lstsq(covariance_, means_.T)[0].T  (sklearn :416).
        // Solve `covariance_ @ X = means_.T` for X of shape (n_features,
        // n_classes); X = lstsq(...)[0]; coef_ = X.T = (n_classes, n_features).
        let means_t = means.t().to_owned(); // (n_features, n_classes)
        let sol = lstsq_multi(&covariance, &means_t)?; // (n_features, n_classes)
        let coef = sol.t().to_owned(); // (n_classes, n_features)

        // intercept_ = -0.5 * diag(means_ @ coef_.T) + log(priors_)  (:417-418).
        // diag(means_ @ coef_.T)[k] = Σ_j means_[k,j] · coef_[k,j].
        let neg_half = -half::<F>();
        let mut intercept = Array1::<F>::zeros(n_classes);
        for k in 0..n_classes {
            let mut dot = <F as num_traits::Zero>::zero();
            for j in 0..n_features {
                dot += means[[k, j]] * coef[[k, j]];
            }
            intercept[k] = neg_half * dot + priors[k].ln();
        }

        // lsqr does NOT support dimensionality reduction (sklearn :372-373,
        // :676-679): no scalings_/xbar_/explained_variance_ratio_. Set them to
        // empty/zero and `max_components = 0` so `transform` yields a `(n, 0)`
        // projection (the lsqr "no transform" contract).
        Ok(FittedLDA {
            scalings: Array2::<F>::zeros((n_features, 0)),
            means: means.to_owned(),
            xbar: Array1::<F>::zeros(n_features),
            priors: priors.to_owned(),
            coef,
            intercept,
            explained_variance_ratio: Array1::<F>::zeros(0),
            covariance: Some(covariance),
            classes: classes.to_vec(),
            max_components: 0,
            n_features,
        })
    }
}

// ---------------------------------------------------------------------------
// Transform (sklearn svd transform)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedLDA<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Project `x` onto the discriminant axes: `((X - xbar_) @ scalings_)[:, :n]`
    /// where `n = max_components`. Mirrors sklearn's svd-solver `transform`
    /// (`discriminant_analysis.py:684-685,689`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.ncols()` does not match the
    /// number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedLDA::transform".into(),
            });
        }
        // (X - xbar_) @ scalings_
        let mut xc = x.to_owned();
        for mut row in xc.rows_mut() {
            for j in 0..self.n_features {
                row[j] = row[j] - self.xbar[j];
            }
        }
        let projected = xc.dot(&self.scalings);
        // Slice to [:, :max_components] (sklearn :689).
        let keep = self.max_components.min(projected.ncols());
        Ok(projected.slice(ndarray::s![.., ..keep]).to_owned())
    }
}

// ---------------------------------------------------------------------------
// Predict (argmax of the affine decision_function)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLDA<F> {
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Classify samples by argmax of the affine `decision_function`
    /// (`classes_[argmax(X @ coef_.T + intercept_)]`), mirroring sklearn's
    /// `predict` (the `LinearClassifierMixin`, `discriminant_analysis.py:739`).
    /// The argmax follows numpy's first-max-wins tie-breaking.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features does not
    /// match the model.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let decision = self.decision_function(x)?;
        let n_samples = decision.nrows();
        let n_classes = decision.ncols();
        let mut predictions = Array1::<usize>::zeros(n_samples);
        for i in 0..n_samples {
            let mut best_idx = 0usize;
            let mut best = decision[[i, 0]];
            for c in 1..n_classes {
                let v = decision[[i, c]];
                // numpy argmax: strictly-greater wins; ties keep first index.
                if v > best {
                    best = v;
                    best_idx = c;
                }
            }
            predictions[i] = self.classes[best_idx];
        }
        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> HasClasses for FittedLDA<F> {
    fn classes(&self) -> &[usize] {
        &self.classes
    }

    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: LinalgFloat + ScalarOperand> PipelineEstimator<F> for LDA<F> {
    /// Fit LDA using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineEstimator<F>>, FerroError> {
        let y_usize: Array1<usize> = y.mapv(|v| v.to_usize().unwrap_or(0));
        let fitted = self.fit(x, &y_usize)?;
        Ok(Box::new(FittedLDAPipeline(fitted)))
    }
}

/// Wrapper for pipeline integration that converts predictions to float.
struct FittedLDAPipeline<F>(FittedLDA<F>);

impl<F: Float + Send + Sync + 'static> FittedPipelineEstimator<F> for FittedLDAPipeline<F> {
    /// Predict via the pipeline interface, returning float class labels.
    fn predict_pipeline(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let preds = self.0.predict(x)?;
        Ok(preds.mapv(|v| NumCast::from(v).unwrap_or_else(F::nan)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn linearly_separable_2d() -> (Array2<f64>, Array1<usize>) {
        // Two well-separated Gaussian clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.2, 0.8, 0.9, 1.1, 1.3, // class 0
                6.0, 6.0, 6.2, 5.8, 5.9, 6.1, 6.3, 5.7, // class 1
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    fn three_class_data() -> (Array2<f64>, Array1<usize>) {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.1, 0.1, 0.5, // class 0
                5.0, 0.0, 5.2, 0.3, 4.8, 0.1, // class 1
                0.0, 5.0, 0.1, 5.2, 0.3, 4.8, // class 2
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (x, y)
    }

    // ------------------------------------------------------------------

    #[test]
    fn test_lda_fit_returns_fitted() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        // scalings_ is (n_features, rank2); for a binary 2-feature set rank2 = 1.
        assert_eq!(fitted.scalings().ncols(), 1);
        assert_eq!(fitted.scalings().nrows(), 2);
    }

    #[test]
    fn test_lda_default_n_components() {
        // With 2 classes the default n_components = min(1, n_features) = 1.
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::default();
        let fitted = lda.fit(&x, &y).unwrap();
        // transform output is truncated to max_components = 1.
        assert_eq!(fitted.transform(&x).unwrap().ncols(), 1);
    }

    #[test]
    fn test_lda_transform_shape() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let proj = fitted.transform(&x).unwrap();
        assert_eq!(proj.dim(), (8, 1));
    }

    #[test]
    fn test_lda_predict_accuracy_binary() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| *p == *a).count();
        assert_eq!(correct, 8, "All 8 samples should be classified correctly");
    }

    #[test]
    fn test_lda_predict_three_classes() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let correct = preds.iter().zip(y.iter()).filter(|(p, a)| *p == *a).count();
        assert!(correct >= 7, "Expected at least 7/9 correct, got {correct}");
    }

    #[test]
    fn test_lda_explained_variance_ratio_positive() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        for &v in fitted.explained_variance_ratio() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_lda_explained_variance_ratio_le_1() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        let total: f64 = fitted.explained_variance_ratio().iter().sum();
        assert!(total <= 1.0 + 1e-9, "total={total}");
    }

    /// R-CHAR-3 oracle pin for `explained_variance_ratio_` (REQ-13). Expected
    /// values are the live sklearn 1.5.2
    /// `LinearDiscriminantAnalysis().fit(X,y).explained_variance_ratio_` on the
    /// 3-class / 2-feature balanced set (same data as `divergence_lda_fit.rs`):
    /// ```text
    /// python3 -c "import numpy as np; \
    ///   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L; \
    ///   X=np.array([[0.,0.],[1.,.5],[.5,1.],[1.,1.],[4.,4.],[5.,4.5],[4.5,5.],[5.,5.],\
    ///               [0.,5.],[1.,6.],[.5,5.5],[1.,5.]]); \
    ///   y=np.array([0,0,0,0,1,1,1,1,2,2,2,2]); \
    ///   print(repr(L().fit(X,y).explained_variance_ratio_.tolist()))"
    /// # [0.6428683117561941, 0.3571316882438059]
    /// ```
    #[test]
    fn test_lda_explained_variance_ratio_oracle() {
        const SK_EVR: [f64; 2] = [0.6428683117561941, 0.3571316882438059];
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 4.0, 4.0, 5.0, 4.5, 4.5, 5.0, 5.0, 5.0,
                0.0, 5.0, 1.0, 6.0, 0.5, 5.5, 1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let fitted = LDA::<f64>::new(Some(2)).fit(&x, &y).unwrap();
        let evr = fitted.explained_variance_ratio();
        assert_eq!(evr.len(), 2);
        for k in 0..2 {
            assert_abs_diff_eq!(evr[k], SK_EVR[k], epsilon = 1e-9);
        }
    }

    /// R-CHAR-3 oracle pin for `coef_`/`intercept_`/`xbar_` (REQ-8). Live
    /// sklearn 1.5.2 attributes on the same 3-class / 2-feature set:
    /// ```text
    /// python3 -c "... ; m=L().fit(X,y); \
    ///   print(repr(m.coef_.tolist())); print(repr(m.intercept_.tolist())); \
    ///   print(repr(m.xbar_.tolist()))"
    /// # coef_ [[2.2582417582417564, -14.02747252747253],
    /// #        [13.335164835164827, -2.950549450549442],
    /// #        [-15.593406593406584, 16.978021978021978]]
    /// # intercept_ [25.208393205837393, -32.94545294800878, -56.65081009086592]
    /// # xbar_ [1.958333333333333, 3.541666666666666]
    /// ```
    #[test]
    fn test_lda_coef_intercept_xbar_oracle() {
        const SK_COEF: [[f64; 2]; 3] = [
            [2.2582417582417564, -14.02747252747253],
            [13.335164835164827, -2.950549450549442],
            [-15.593406593406584, 16.978021978021978],
        ];
        const SK_INTERCEPT: [f64; 3] = [25.208393205837393, -32.94545294800878, -56.65081009086592];
        const SK_XBAR: [f64; 2] = [1.958333333333333, 3.541666666666666];
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 4.0, 4.0, 5.0, 4.5, 4.5, 5.0, 5.0, 5.0,
                0.0, 5.0, 1.0, 6.0, 0.5, 5.5, 1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let fitted = LDA::<f64>::new(Some(2)).fit(&x, &y).unwrap();
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(fitted.coef()[[i, j]], SK_COEF[i][j], epsilon = 1e-9);
            }
            assert_abs_diff_eq!(fitted.intercept()[i], SK_INTERCEPT[i], epsilon = 1e-9);
        }
        for j in 0..2 {
            assert_abs_diff_eq!(fitted.xbar()[j], SK_XBAR[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lda_classes_accessor() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.classes(), &[0usize, 1]);
    }

    #[test]
    fn test_lda_means_shape() {
        // means_ is now in the ORIGINAL feature space (n_classes, n_features).
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.means().dim(), (3, 2));
    }

    #[test]
    fn test_lda_transform_shape_mismatch() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let x_bad = Array2::<f64>::zeros((3, 5));
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_lda_predict_shape_mismatch() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let x_bad = Array2::<f64>::zeros((3, 5));
        assert!(fitted.predict(&x_bad).is_err());
    }

    #[test]
    fn test_lda_error_zero_n_components() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(0));
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_n_components_too_large() {
        let (x, y) = linearly_separable_2d(); // 2 classes → max 1 component
        let lda = LDA::<f64>::new(Some(5));
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_single_class() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0usize, 0, 0, 0];
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_shape_mismatch_fit() {
        let x = Array2::<f64>::zeros((4, 2));
        let y = array![0usize, 1]; // wrong length
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_error_insufficient_samples() {
        let x = Array2::<f64>::zeros((1, 2));
        let y = array![0usize];
        let lda = LDA::<f64>::new(None);
        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_scalings_accessor() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        assert_eq!(fitted.scalings().nrows(), 2);
    }

    #[test]
    fn test_lda_pipeline_estimator() {
        use ferrolearn_core::pipeline::PipelineEstimator;

        let (x, y_usize) = linearly_separable_2d();
        let y_f64 = y_usize.mapv(|v| v as f64);
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit_pipeline(&x, &y_f64).unwrap();
        let preds = fitted.predict_pipeline(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_lda_n_components_getter() {
        let lda = LDA::<f64>::new(Some(2));
        assert_eq!(lda.n_components(), Some(2));
        let lda_none = LDA::<f64>::new(None);
        assert_eq!(lda_none.n_components(), None);
    }

    #[test]
    fn test_lda_priors_builder_default_none() {
        // Default (sklearn `priors=None`, discriminant_analysis.py:359).
        let lda = LDA::<f64>::new(None);
        assert!(lda.priors().is_none());
        // with_priors stores the vector verbatim.
        let lda = lda.with_priors(array![0.7, 0.3]);
        let p = lda.priors().cloned().unwrap_or_default();
        assert_eq!(p.len(), 2);
        assert_abs_diff_eq!(p[0], 0.7, epsilon = 1e-12);
        assert_abs_diff_eq!(p[1], 0.3, epsilon = 1e-12);
    }

    /// Re-oracled (was `test_lda_transform_then_predict_consistent`, which
    /// asserted the OLD nearest-centroid algorithm: `predict ==
    /// argmin ‖transform(x) - projected_mean‖`). The SVD solver's `predict` is
    /// the argmax of the affine `decision_function = X @ coef_.T + intercept_`
    /// (`discriminant_analysis.py:739`), NOT nearest-centroid in projected
    /// space, so this now checks the new contract.
    #[test]
    fn test_lda_predict_matches_decision_argmax() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let dec = fitted.decision_function(&x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        let n_samples = dec.nrows();
        let n_classes = dec.ncols();
        for i in 0..n_samples {
            let mut best = 0usize;
            let mut best_v = dec[[i, 0]];
            for c in 1..n_classes {
                if dec[[i, c]] > best_v {
                    best_v = dec[[i, c]];
                    best = c;
                }
            }
            assert_eq!(preds[i], fitted.classes()[best]);
        }
    }

    #[test]
    fn test_lda_projected_class_separation() {
        let (x, y) = linearly_separable_2d();
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let projected = fitted.transform(&x).unwrap();

        // Means of class 0 and class 1 in projected space should be far apart.
        let mean0: f64 = projected
            .rows()
            .into_iter()
            .zip(y.iter())
            .filter(|&(_, label)| *label == 0)
            .map(|(row, _)| row[0])
            .sum::<f64>()
            / 4.0;
        let mean1: f64 = projected
            .rows()
            .into_iter()
            .zip(y.iter())
            .filter(|&(_, label)| *label == 1)
            .map(|(row, _)| row[0])
            .sum::<f64>()
            / 4.0;

        assert!(
            (mean0 - mean1).abs() > 0.5,
            "Projected means should differ, got {mean0} vs {mean1}"
        );
    }

    #[test]
    fn test_lda_transform_known_data() {
        // With perfectly separated 1-D data the centered/whitened transform
        // should still place the two classes on opposite sides.
        let x = Array2::from_shape_vec((4, 1), vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let lda = LDA::<f64>::new(Some(1));
        let fitted = lda.fit(&x, &y).unwrap();
        let proj = fitted.transform(&x).unwrap();
        let sign0 = proj[[0, 0]].signum();
        let sign1 = proj[[2, 0]].signum();
        assert_ne!(
            sign0 as i32, sign1 as i32,
            "Classes should be on opposite sides"
        );
    }

    #[test]
    fn test_lda_predict_proba_rows_sum_to_one() {
        let (x, y) = three_class_data();
        let lda = LDA::<f64>::new(Some(2));
        let fitted = lda.fit(&x, &y).unwrap();
        let proba = fitted.predict_proba(&x).unwrap();
        assert_eq!(proba.dim(), (9, 3));
        for row in proba.rows() {
            let s: f64 = row.iter().sum();
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-12);
        }
    }
}
