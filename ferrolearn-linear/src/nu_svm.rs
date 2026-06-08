//! Nu-parameterized Support Vector Machines.
//!
//! This module provides [`NuSVC`] (classification) and [`NuSVR`] (regression),
//! the nu-parameterized libsvm SVMs (`sklearn.svm.NuSVC` /
//! `sklearn.svm.NuSVR`, `sklearn/svm/_classes.py`). Instead of the penalty
//! parameter `C` (NuSVC) / the `epsilon` tube (NuSVR), the user supplies
//! `nu ∈ (0, 1]`, an upper bound on the fraction of margin errors and a lower
//! bound on the fraction of support vectors.
//!
//! # The genuine nu-SVM solver (NOT a C-SVC delegation)
//!
//! `NuSVC`/`NuSVR` run the **true libsvm `Solver_NU`** dual
//! (`sklearn/svm/src/libsvm/svm.cpp:1166-1418`, `solve_nu_svc`/`solve_nu_svr`
//! at `:1646`/`:1795`), NOT a C-SVC / epsilon-SVR with `C = 1/(nu·n)`. The
//! nu-SVC dual carries the additional equality constraint `eᵀα = nu·l`
//! (`_impl = "nu_svc"`, libsvm `solver_type == 1`, `sklearn/svm/_base.py:30`);
//! the nu-SVR dual is a `2l`-variable problem with a LEARNED tube width and
//! BOTH `nu` and `C` as parameters (`_impl = "nu_svr"`, `solver_type == 4`).
//! These reach a genuinely different optimum from a re-scaled C-SVC — see
//! [`crate::svm::solve_nu_svc`]/[`crate::svm::solve_nu_svr`].
//!
//! NuSVC reuses the SVC fitted-attribute / `decision_function` / `predict`
//! machinery via [`crate::svm::FittedSVC::from_nu_ovo`] (one nu-SVC sub-model
//! per one-vs-one class pair), so the libsvm-layout `support_`/`dual_coef_`/
//! `intercept_`/`coef_` and the binary nu_svc sign flip (`_base.py:258-262`,
//! `_impl in ["c_svc","nu_svc"]`) are produced exactly as for SVC.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::nu_svm::{NuSVC, NuSVR};
//! use ferrolearn_linear::svm::LinearKernel;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.5, 1.0,  1.0, 1.5,
//!     5.0, 5.0,  5.5, 5.0,  5.0, 5.5,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = NuSVC::<f64, LinearKernel>::new(LinearKernel);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! ## REQ status
//!
//! Binary (R-DEFER-2): SHIPPED = impl + non-test production consumer + tests +
//! green oracle verification; NOT-STARTED = open blocker `#`. `NuSVC`/`NuSVR`/
//! `FittedNuSVC`/`FittedNuSVR` are boundary estimator types re-exported at the
//! crate root (`pub use nu_svm::{…}` in `lib.rs`); their consumer surface is
//! grandfathered under S5/R-DEFER-1, and the non-test production consumer of
//! the new svm.rs solver/constructor APIs is `fn fit in nu_svm.rs`. See
//! `.design/linear/nu_svm.md`.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (nu validation `(0,1]`) | SHIPPED | `fn fit in nu_svm.rs` (NuSVC and NuSVR) rejects `nu <= 0 \|\| nu > 1` with `InvalidParameter` — `Interval(Real, 0.0, 1.0, closed="right")` (`_classes.py:1116`). Tests `test_nusvc_invalid_nu_zero`/`test_nusvc_invalid_nu_above_one`/`test_nusvc_nu_one`/`test_nusvr_invalid_nu in nu_svm.rs`. |
//! | REQ-2 (nu-SVC dual) | SHIPPED | `NuSVC::fit in nu_svm.rs` calls `crate::svm::solve_nu_svc` (libsvm `Solver_NU`, `eᵀα=nu·l`, `svm.cpp:1646`) per ovo pair — NOT a C-SVC. Pinned: `divergence_nusvc_decision_function_delegation in tests/divergence_nu_svm.rs` (df `[-1.25,-1.0,-1.0,0.75,1.0,1.0]`) + `test_nusvc_oracle_attrs in nu_svm.rs` (`support_ [1,2,3,5]`, `dual_coef_ [[-0.0227,-0.0455,0.0455,0.0227]]`, `intercept_ [-1.75]` vs live `NuSVC(kernel='linear',nu=0.5)`, R-CHAR-3, 1e-2). |
//! | REQ-3 (nu-SVR dual + C) | SHIPPED | `NuSVR::fit in nu_svm.rs` calls `crate::svm::solve_nu_svr(nu, C)` (libsvm `solver_type==4`, learned tube, `svm.cpp:1795`); `NuSVR` gains `pub c: F` (default 1.0, `_classes.py:1531`) + `with_c`, NO `epsilon`. Pinned: `divergence_nusvr_predict_delegation`/`divergence_nusvr_missing_c_parameter in tests/divergence_nu_svm.rs` (predict `[2.5,3.5,4.5,5.5]`) + `test_nusvr_oracle_attrs in nu_svm.rs` (`support_ [2,3]`, `dual_coef_ [[-1,1]]`, `intercept_ [1.5]`). |
//! | REQ-4 (kernels & gamma resolution) | SHIPPED | NuSVC/NuSVR resolve `gamma` via `self.kernel.resolved_for_fit(x)` (the shared `crate::svm` path: `Gamma::Scale`=`1/(n_features·X.var())` default, `_base.py:236-243`) before the nu solve. |
//! | REQ-5 (fitted classification attrs) | SHIPPED | `FittedNuSVC` re-exposes `support`/`support_vectors`/`n_support`/`dual_coef`/`intercept`/`coef` by delegating to the inner `FittedSVC` built via `crate::svm::FittedSVC::from_nu_ovo`; the libsvm layout + binary nu_svc sign flip (`_base.py:258-262`) apply. Pinned by `test_nusvc_oracle_attrs in nu_svm.rs`. |
//! | REQ-6 (decision_function propagates SvmScores) | SHIPPED | `FittedNuSVC::decision_function in nu_svm.rs` returns `Result<SvmScores<F>, FerroError>` from the inner `FittedSVC`; binary `(n,)`, ovr `(n,n_classes)` (`_base.py:538-541`). Pinned by `divergence_nusvc_decision_function_delegation`. |
//! | REQ-7 (predict — ovo voting + tie-break) | SHIPPED | `FittedNuSVC::predict in nu_svm.rs` delegates to the inner `FittedSVC::predict` (libsvm ovo voting, lower-index tie-break, `_base.py:813-814`). Pinned by `nusvc_predict_labels_match_on_separable in tests/divergence_nu_svm.rs` + `test_nusvc_oracle_attrs`. |
//! | REQ-8 (multiclass NuSVC one-vs-one) | SHIPPED | `NuSVC::fit in nu_svm.rs` trains one `solve_nu_svc` per class pair over `classes = np.unique(y)`, `<2`-class -> `InsufficientSamples` (`_base.py:741-745`), assembled via `from_nu_ovo`. Smoke test `test_nusvc_multiclass in nu_svm.rs`. |
//! | REQ-9 (probability / predict_proba) | SHIPPED | `pub probability: bool` field on `NuSVC` (default `false`, `_classes.py:1129`, `with_probability`) + `predict_proba`/`predict_log_proba`/`prob_a`/`prob_b`/`probability` on `FittedNuSVC`. `NuSVC::fit in nu_svm.rs` (when `self.probability`) runs the shared `crate::svm::platt_cv_sigmoid` PER OVO PAIR with a `train_fold` closure wrapping the **NU-SVC** sub-solver `crate::svm::solve_nu_svc` (the SAME `svm_type` as the outer model, libsvm `svm_binary_svc_probability` `svm.cpp:2147-2150` — NOT `smo_binary`/C-SVC) → per-pair `(probA_, probB_)`, stored on the inner `FittedSVC` via `crate::svm::FittedSVC::from_nu_ovo`. `FittedNuSVC::predict_proba in nu_svm.rs` delegates to `FittedSVC::predict_proba` (the `sigmoid_predict` → `multiclass_probability` coupling is solver-agnostic, consuming only `raw_ovo` + `prob_a`/`prob_b`): binary `[P(classes[0]),P(classes[1])]`, multiclass pairwise → coupling, rows sum to 1, clamp `[1e-7,1-1e-7]`; `predict_log_proba` = `predict_proba.ln()` (`_base.py:866-894`). `probability=false` → `InvalidParameter` "predict_proba is not available when fitted with probability=False" (`_base.py:856-860`; no `NotFitted` variant by R-DEV-4 typestate). **RNG-CV value divergence (documented, NOT a gap, the SAME boundary SVC's REQ-9 carries):** libsvm seeds the CV fold permutation with `random_state` (`svm.cpp:2116-2122`), so sklearn's `probA_`/`probB_`/`predict_proba` are NON-DETERMINISTIC; ferrolearn uses a DETERMINISTIC contiguous 5-fold split, so it does NOT bit-match sklearn's VALUES — only the machinery + structural invariants + the raise contract are verified (R-CHAR-3: the asserted invariants are sklearn's DOCUMENTED contract). `class_weight`/`random_state` NOT-STARTED (deterministic solver, as for SVC). Non-test consumer: `fn fit in nu_svm.rs` consumes `self.probability` (the boundary `NuSVC`/`FittedNuSVC` types are re-exported at the crate root). Pinned by `nusvc_predict_proba_raises_when_probability_false`/`nusvc_predict_proba_binary_rows_sum_to_one`/`nusvc_predict_proba_binary_monotone_in_decision`/`nusvc_predict_log_proba_equals_log_of_proba`/`nusvc_predict_proba_multiclass_rows_sum_to_one in tests/divergence_nu_svm.rs`. |
//! | REQ-10 (constructor params/defaults) | SHIPPED (R-DEV-7 design difference) | NuSVC: `nu` (0.5), `tol` (1e-3), `cache_size` (200), `max_iter` (0 = sklearn `-1`), `decision_function_shape` (Ovr), `break_ties` (false); NuSVR: `nu` (0.5), **`c` (1.0)**, `tol`, `cache_size` (200), `max_iter` (0), NO `epsilon`. `kernel`/`degree`/`gamma`/`coef0` are the type parameter `K` (R-DEV-7, as for SVC); NuSVC now carries `probability` (default `false`, REQ-9 SHIPPED); `class_weight`/`random_state` unused (deterministic solver). |
//! | REQ-11 (ferray substrate) | NOT-STARTED | open #655. `nu_svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}`, not `ferray-core` (R-SUBSTRATE; consistent with svm.rs REQ-10 #643). |
//! | REQ-12 (non-finite input rejected) | SHIPPED | Both fit entries reject any NaN/+/-inf BEFORE the nu solve with `FerroError::InvalidParameter`, mirroring sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, …)` (`_base.py:190-197`, default `force_all_finite=True`) -> `ValueError`. **`NuSVC::fit`/`NuSVR::fit in nu_svm.rs` call `crate::svm::solve_nu_svc`/`solve_nu_svr` DIRECTLY — they do NOT route through the guarded `SVC::fit`/`SVR::fit`, so each carries its OWN guard.** `NuSVC::fit` checks `X` (`y` is `Array1<usize>` labels, finite by type); `NuSVR::fit` checks `X` AND the float target `y`. ferrolearn's `Fit::fit` has no `sample_weight` argument, so the sklearn `sample_weight`-finiteness raise has no fit-entry counterpart. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (the nu-SVC/nu-SVR oracle pins stay green). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): NaN/+inf/-inf in X for both, NaN/inf in y for NuSVR, all raise `ValueError` (`tests/divergence_svm_nonfinite.rs::{nusvc_*,nusvr_*}`). Non-test consumer: the existing `Fit::fit` consumers + the crate-root `pub use nu_svm::{NuSVC, NuSVR, …}` re-exports. (#2269) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use crate::svm::{
    FittedSVC, Kernel, NuOvoPair, SvmDecisionShape, SvmScores, platt_cv_sigmoid, solve_nu_svc,
    solve_nu_svr,
};

// ---------------------------------------------------------------------------
// NuSVC
// ---------------------------------------------------------------------------

/// Nu-parameterized Support Vector Classifier
/// (`sklearn.svm.NuSVC`, `sklearn/svm/_classes.py:898`, `_impl = "nu_svc"`).
///
/// Instead of specifying `C` directly, the user sets `nu ∈ (0, 1]`. The fit
/// runs the genuine libsvm nu-SVC dual ([`solve_nu_svc`]), one sub-model per
/// one-vs-one class pair — NOT a C-SVC with `C = 1/(nu·n)`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`LinearKernel`](crate::svm::LinearKernel)).
#[derive(Debug, Clone)]
pub struct NuSVC<F, K> {
    /// The nu parameter, in `(0, 1]`. Default: `0.5`
    /// (`Interval(Real, 0.0, 1.0, closed="right")`, `_classes.py:1116`).
    pub nu: F,
    /// The kernel function.
    pub kernel: K,
    /// Convergence tolerance. Default `1e-3`.
    pub tol: F,
    /// Maximum number of SMO iterations. `0` is the sklearn `max_iter=-1`
    /// sentinel (no iteration limit — the nu solver runs to convergence).
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache. Default `200`
    /// (matching sklearn's `cache_size=200`).
    pub cache_size: usize,
    /// The multiclass `decision_function` shape convention
    /// (`sklearn/svm/_base.py:778-781`); default
    /// [`SvmDecisionShape::Ovr`] (sklearn's `decision_function_shape='ovr'`).
    pub decision_function_shape: SvmDecisionShape,
    /// Whether `predict` breaks ties by the ovr decision confidence
    /// (`break_ties`, `sklearn/svm/_classes.py`, default `false`).
    pub break_ties: bool,
    /// Whether to enable Platt-scaling probability estimates (`probability`,
    /// `sklearn/svm/_classes.py:1129`, default `False`). When `true`,
    /// [`Fit::fit`] runs a per-ovo-pair 5-fold internal CV using the
    /// NU-SVC sub-solver ([`solve_nu_svc`]) — the SAME `svm_type` as the outer
    /// model, libsvm `svm_binary_svc_probability` (`svm.cpp:2147-2150`) — to fit
    /// the sigmoid `(probA_, probB_)`, enabling
    /// [`FittedNuSVC::predict_proba`]/[`FittedNuSVC::predict_log_proba`].
    ///
    /// Like SVC's `probability` (svm.rs REQ-9), the predict_proba VALUES do NOT
    /// bit-match sklearn: libsvm seeds the CV fold permutation with
    /// `random_state` (`svm.cpp:2116-2122`), making sklearn's
    /// `probA_`/`probB_`/`predict_proba` NON-DETERMINISTIC; ferrolearn uses a
    /// DETERMINISTIC contiguous 5-fold split, so only the machinery + the
    /// structural invariants (rows sum to 1, monotone in the decision value,
    /// the raise-when-`probability=false`) are verified (R-DEV-4 / R-CHAR-3).
    pub probability: bool,
}

impl<F: Float, K: Kernel<F>> NuSVC<F, K> {
    /// Create a new `NuSVC` with the given kernel and sklearn-matching defaults:
    /// `nu = 0.5`, `tol = 1e-3`, `max_iter = 0` (= sklearn `-1`, no iteration
    /// limit), `cache_size = 200`, `decision_function_shape = Ovr`,
    /// `break_ties = false`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            nu: F::from(0.5).unwrap_or_else(F::epsilon),
            kernel,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            max_iter: 0,
            cache_size: 200,
            decision_function_shape: SvmDecisionShape::Ovr,
            break_ties: false,
            probability: false,
        }
    }

    /// Set the nu parameter (`nu ∈ (0, 1]`).
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
        self
    }

    /// Enable/disable Platt-scaling probability estimates (`sklearn`
    /// `probability`, default `false`, `_classes.py:1129`). When `true`,
    /// [`Fit::fit`] runs the per-ovo-pair 5-fold internal CV with the NU-SVC
    /// sub-solver to fit the sigmoid `(probA_, probB_)`, enabling
    /// [`FittedNuSVC::predict_proba`]. See the [`NuSVC::probability`] field doc
    /// for the documented RNG-CV value divergence.
    #[must_use]
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO iterations (`0` = no limit).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the kernel cache size.
    #[must_use]
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Set the multiclass `decision_function` shape convention
    /// (`'ovr'` default / `'ovo'`, `sklearn/svm/_base.py:778-781`).
    #[must_use]
    pub fn with_decision_function_shape(mut self, shape: SvmDecisionShape) -> Self {
        self.decision_function_shape = shape;
        self
    }

    /// Set the `break_ties` flag (`sklearn` `break_ties`, default `false`).
    #[must_use]
    pub fn with_break_ties(mut self, break_ties: bool) -> Self {
        self.break_ties = break_ties;
        self
    }
}

/// Fitted Nu-SVC.
///
/// Wraps a [`FittedSVC`] assembled from the per-ovo-pair nu-SVC sub-models, so
/// every libsvm-layout fitted attribute / `decision_function` / `predict` is
/// re-exposed (the binary nu_svc sign flip applies, `_base.py:258-262`).
#[derive(Debug, Clone)]
pub struct FittedNuSVC<F, K>(FittedSVC<F, K>);

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<usize>> for NuSVC<F, K>
{
    type Fitted = FittedNuSVC<F, K>;
    type Error = FerroError;

    /// Fit the NuSVC model by running the genuine libsvm nu-SVC dual
    /// ([`solve_nu_svc`]) once per one-vs-one class pair.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`
    ///   (`Interval(Real, 0.0, 1.0, closed="right")`, `_classes.py:1116`).
    /// - [`FerroError::InsufficientSamples`] if there are no samples or fewer
    ///   than 2 distinct classes (`_base.py:741-745`).
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` disagree on sample count.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedNuSVC<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NuSVC requires at least one sample".into(),
            });
        }
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Reject non-finite input (NaN / +/-inf) in X BEFORE the nu-SVC solve.
        // `NuSVC::fit` calls `crate::svm::solve_nu_svc` DIRECTLY (the genuine
        // `Solver_NU` dual) — it does NOT route through the guarded
        // `crate::svm::SVC::fit`, so it needs its OWN finiteness guard. Mirrors
        // sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, …)`
        // (`sklearn/svm/_base.py:190-197`, default `force_all_finite=True`) ->
        // `ValueError("Input X contains NaN.")` / `"… contains infinity …"`. `y`
        // is class labels (`Array1<usize>`), finite by type, so only X is
        // checked. `.iter().any(|v| !v.is_finite())` catches NaN and +/-inf; on
        // finite input the guard never fires (the nu-SVC fitted attributes are
        // byte-identical).
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // classes_ = np.unique(y), sorted ascending (`_base.py:741`).
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "NuSVC requires at least 2 distinct classes".into(),
            });
        }

        // Resolve gamma='scale'/'auto'/float against X before fitting
        // (`_base.py:236-243`), shared with SVC via `resolved_for_fit`.
        let kernel = self.kernel.resolved_for_fit(x);
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        let n_classes = classes.len();
        let mut pairs: Vec<NuOvoPair<F>> = Vec::new();
        // Per-ovo-pair Platt sigmoid params (only filled when `probability`).
        let mut prob_a: Vec<F> = Vec::new();
        let mut prob_b: Vec<F> = Vec::new();

        for ci in 0..n_classes {
            for cj in (ci + 1)..n_classes {
                let class_neg = classes[ci]; // lower index (libsvm +1 side)
                let class_pos = classes[cj]; // higher index

                // Per-pair samples + ferrolearn sign (class_pos = +1).
                let mut sub_data = Vec::new();
                let mut sub_labels = Vec::new();
                let mut sub_indices = Vec::new();
                for s in 0..n_samples {
                    let label = y[s];
                    if label == class_neg {
                        sub_data.push(data[s].clone());
                        sub_labels.push(-F::one());
                        sub_indices.push(s);
                    } else if label == class_pos {
                        sub_data.push(data[s].clone());
                        sub_labels.push(F::one());
                        sub_indices.push(s);
                    }
                }

                // The genuine libsvm nu-SVC dual for this pair (NOT C-SVC).
                let model = solve_nu_svc(
                    &sub_data,
                    &sub_labels,
                    &kernel,
                    self.nu,
                    self.tol,
                    self.max_iter,
                    self.cache_size,
                )
                .ok_or_else(|| FerroError::InvalidParameter {
                    name: "nu".into(),
                    reason: "nu-SVC solve produced a degenerate (r ≈ 0) optimum; \
                             try a different nu (sklearn raises a similar \
                             infeasibility error)"
                        .into(),
                })?;

                // Platt-scaling CV for this ovo pair (only when probability).
                // libsvm's `svm_binary_svc_probability` trains the CV sub-models
                // with the SAME `svm_type` as the outer model
                // (`svm.cpp:2147-2150`): for NuSVC that is NU-SVC, so the
                // `train_fold` closure wraps `solve_nu_svc` (NOT `smo_binary`,
                // which is C-SVC). The returned `NuSvcModel` is already in this
                // crate's sign (`class_pos = +1`), exactly the `SubModel`
                // `(sv_data, sv_coefs, bias)` the shared CV expects.
                if self.probability {
                    let (nu, tol, max_iter, cache_size) =
                        (self.nu, self.tol, self.max_iter, self.cache_size);
                    let (a, b) = platt_cv_sigmoid(
                        &sub_data,
                        &sub_labels,
                        &kernel,
                        |tr_data: &[Vec<F>], tr_labels: &[F]| {
                            let sub = solve_nu_svc(
                                tr_data, tr_labels, &kernel, nu, tol, max_iter, cache_size,
                            )?;
                            Some((sub.sv_data, sub.sv_coefs, sub.bias_internal))
                        },
                    );
                    prob_a.push(a);
                    prob_b.push(b);
                }

                // Map per-pair SV rows back to ORIGINAL training-row indices.
                let sv_indices: Vec<usize> =
                    model.sv_indices.iter().map(|&k| sub_indices[k]).collect();

                pairs.push(NuOvoPair {
                    sv_data: model.sv_data,
                    sv_coefs: model.sv_coefs,
                    sv_indices,
                    bias_internal: model.bias_internal,
                    class_neg,
                    class_pos,
                });
            }
        }

        let inner = FittedSVC::from_nu_ovo(
            kernel,
            pairs,
            classes,
            x.clone(),
            y.to_vec(),
            self.decision_function_shape,
            self.break_ties,
            self.probability,
            prob_a,
            prob_b,
        );
        Ok(FittedNuSVC(inner))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedNuSVC<F, K>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels (libsvm ovo voting, lower-index tie-break),
    /// delegating to the inner [`FittedSVC::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count differs from
    /// training, or [`FerroError::InvalidParameter`] for the
    /// `break_ties=true`+`Ovo` combination (`_base.py:801-804`).
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        self.0.predict(x)
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> FittedNuSVC<F, K> {
    /// The raw decision function values
    /// (`sklearn/svm/_base.py:536-541, 778-781`), delegating to the inner
    /// [`FittedSVC::decision_function`]: binary [`SvmScores::Binary`] `(n,)`,
    /// multiclass [`SvmScores::Multiclass`] `(n, n_classes)` (ovr) /
    /// `(n, n·(n-1)/2)` (ovo).
    ///
    /// # Errors
    ///
    /// Returns `Ok` for valid input.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<SvmScores<F>, FerroError> {
        self.0.decision_function(x)
    }

    /// Support-vector training-row indices, grouped per class
    /// (`support_`, `sklearn/svm/_base.py:318-410`).
    #[must_use]
    pub fn support(&self) -> Array1<usize> {
        self.0.support()
    }

    /// The support vectors `X[support_]` (`support_vectors_`).
    #[must_use]
    pub fn support_vectors(&self) -> Array2<F> {
        self.0.support_vectors()
    }

    /// Number of support vectors per class (`n_support_`).
    #[must_use]
    pub fn n_support(&self) -> Vec<usize> {
        self.0.n_support()
    }

    /// Dual coefficients in the libsvm public layout `(n_class-1, n_SV)`
    /// (`dual_coef_`), with the binary nu_svc sign flip (`_base.py:258-262`).
    #[must_use]
    pub fn dual_coef(&self) -> Array2<F> {
        self.0.dual_coef()
    }

    /// Per-ovo-problem intercepts, length `n_class·(n_class-1)/2`
    /// (`intercept_`), with the binary nu_svc sign flip.
    #[must_use]
    pub fn intercept(&self) -> Array1<F> {
        self.0.intercept()
    }

    /// Primal weight vector `coef_ = dual_coef_ @ support_vectors_` — linear
    /// kernel only, else `None` (`sklearn/svm/_base.py:650-666`).
    #[must_use]
    pub fn coef(&self) -> Option<Array2<F>> {
        self.0.coef()
    }

    /// Class probability estimates, shape `(n_samples, n_classes)`; columns
    /// correspond to `classes_` in sorted order (`sklearn/svm/_base.py:829-864`,
    /// `_impl in ("c_svc","nu_svc")`). Delegates to the inner
    /// [`FittedSVC::predict_proba`], whose `prob_a`/`prob_b` were fitted by the
    /// per-ovo-pair NU-SVC Platt CV ([`platt_cv_sigmoid`] over
    /// [`solve_nu_svc`]). For the binary case the row is
    /// `[P(classes[0]), P(classes[1])]`; multiclass uses the Wu-Lin-Weng
    /// coupling. Rows sum to 1; values clamped to `[1e-7, 1-1e-7]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when the model was fitted with
    /// `probability=false` (`predict_proba is not available when fitted with
    /// probability=False`, `_base.py:856-860`).
    ///
    /// **RNG-CV value divergence (documented, NOT a gap):** the predict_proba
    /// VALUES do NOT bit-match sklearn — libsvm seeds the CV fold permutation
    /// with `random_state` (`svm.cpp:2116-2122`); ferrolearn uses a
    /// DETERMINISTIC contiguous 5-fold split (R-DEV-4). Only the machinery +
    /// structural invariants are verified. See [`NuSVC::probability`].
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.0.predict_proba(x)
    }

    /// Natural-log class probability estimates, shape `(n_samples, n_classes)`
    /// = `predict_proba(x).ln()` (`sklearn/svm/_base.py:866-894`), delegating to
    /// the inner [`FittedSVC::predict_log_proba`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when the model was fitted with
    /// `probability=false` (delegated from [`Self::predict_proba`]).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.0.predict_log_proba(x)
    }

    /// Whether Platt-scaling probability estimates were fitted (`probability`);
    /// when `false`, [`Self::predict_proba`]/[`Self::predict_log_proba`] raise.
    #[must_use]
    pub fn probability(&self) -> bool {
        self.0.probability()
    }

    /// The per-ovo-pair Platt sigmoid `A` parameters (`probA_`,
    /// `sklearn/svm/_base.py`), length `n_class·(n_class-1)/2`. Empty when
    /// fitted with `probability=false`.
    #[must_use]
    pub fn prob_a(&self) -> Array1<F> {
        self.0.prob_a()
    }

    /// The per-ovo-pair Platt sigmoid `B` parameters (`probB_`,
    /// `sklearn/svm/_base.py`), length `n_class·(n_class-1)/2`. Empty when
    /// fitted with `probability=false`.
    #[must_use]
    pub fn prob_b(&self) -> Array1<F> {
        self.0.prob_b()
    }
}

// ---------------------------------------------------------------------------
// NuSVR
// ---------------------------------------------------------------------------

/// Nu-parameterized Support Vector Regressor
/// (`sklearn.svm.NuSVR`, `sklearn/svm/_classes.py:1376`, `_impl = "nu_svr"`).
///
/// A TWO-parameter model: both `nu ∈ (0, 1]` AND `C` (default 1.0,
/// `_classes.py:1531`) are user-supplied; there is **no `epsilon`** (the tube
/// width is learned). The fit runs the genuine libsvm nu-SVR `2l`-variable dual
/// ([`solve_nu_svr`]), NOT epsilon-SVR with `epsilon = 0`.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type.
#[derive(Debug, Clone)]
pub struct NuSVR<F, K> {
    /// The nu parameter, in `(0, 1]`. Default: `0.5`.
    pub nu: F,
    /// The penalty parameter `C`. Default `1.0` (`_classes.py:1531`).
    pub c: F,
    /// The kernel function.
    pub kernel: K,
    /// Convergence tolerance. Default `1e-3`.
    pub tol: F,
    /// Maximum number of SMO iterations. `0` = sklearn `max_iter=-1` (no limit).
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache. Default `200`.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> NuSVR<F, K> {
    /// Create a new `NuSVR` with the given kernel and sklearn-matching defaults:
    /// `nu = 0.5`, `C = 1.0`, `tol = 1e-3`, `max_iter = 0` (= sklearn `-1`),
    /// `cache_size = 200`. NuSVR has **no `epsilon`**.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            nu: F::from(0.5).unwrap_or_else(F::epsilon),
            c: F::one(),
            kernel,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            max_iter: 0,
            cache_size: 200,
        }
    }

    /// Set the nu parameter (`nu ∈ (0, 1]`).
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
        self
    }

    /// Set the penalty parameter `C` (`sklearn` `C`, default 1.0,
    /// `_classes.py:1531`).
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO iterations (`0` = no limit).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the kernel cache size.
    #[must_use]
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }
}

/// Fitted Nu-SVR.
///
/// Stores the support vectors, prediction coefficients (`α*−α`), and bias from
/// the genuine libsvm nu-SVR dual.
#[derive(Debug, Clone)]
pub struct FittedNuSVR<F, K> {
    kernel: K,
    /// Support-vector feature rows.
    support_vectors: Vec<Vec<F>>,
    /// Original training-row index of each support vector.
    sv_indices: Vec<usize>,
    /// Prediction coefficients `α*_i − α_i` (`dual_coef_`).
    dual_coefs: Vec<F>,
    /// Bias term (`f(x) = Σ coef·K + bias`).
    bias: F,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<F>> for NuSVR<F, K>
{
    type Fitted = FittedNuSVR<F, K>;
    type Error = FerroError;

    /// Fit the NuSVR model by running the genuine libsvm nu-SVR dual
    /// ([`solve_nu_svr`]) with `(nu, C)` — NOT epsilon-SVR with `epsilon = 0`.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]` or `C` is
    ///   not positive.
    /// - [`FerroError::InsufficientSamples`] for empty input.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` disagree on sample count.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedNuSVR<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }
        if self.c <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "C".into(),
                reason: "must be positive".into(),
            });
        }

        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "NuSVR requires at least one sample".into(),
            });
        }
        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Reject non-finite input (NaN / +/-inf) in X or the float target y
        // BEFORE the nu-SVR solve. `NuSVR::fit` calls `crate::svm::solve_nu_svr`
        // DIRECTLY (the genuine `2l`-variable `Solver_NU` dual) — it does NOT
        // route through the guarded `crate::svm::SVR::fit`, so it needs its OWN
        // finiteness guard. Mirrors sklearn's `BaseLibSVM.fit` ->
        // `_validate_data(X, y, …)` (`sklearn/svm/_base.py:190-197`, default
        // `force_all_finite=True`) -> `ValueError("Input X contains NaN.")` /
        // `"Input y contains NaN."` / `"… contains infinity …"`. NuSVR's `y` is
        // float regression targets, so both X and y are checked.
        // `.iter().any(|v| !v.is_finite())` catches NaN and +/-inf; on finite
        // input the guard never fires (the fitted attributes are byte-identical).
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "Input y contains NaN or infinity.".into(),
            });
        }

        // Resolve gamma against X before fitting (`_base.py:236-243`).
        let kernel = self.kernel.resolved_for_fit(x);
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();
        let targets: Vec<F> = y.to_vec();

        let model = solve_nu_svr(
            &data,
            &targets,
            &kernel,
            self.nu,
            self.c,
            self.tol,
            self.max_iter,
            self.cache_size,
        );

        Ok(FittedNuSVR {
            kernel,
            support_vectors: model.sv_data,
            sv_indices: model.sv_indices,
            dual_coefs: model.sv_coefs,
            bias: model.bias,
        })
    }
}

impl<F: Float, K: Kernel<F>> FittedNuSVR<F, K> {
    /// Compute the decision function for a single sample.
    fn decision_value(&self, x: &[F]) -> F {
        let mut val = self.bias;
        for (sv, &coef) in self.support_vectors.iter().zip(self.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val
    }

    /// The raw decision function values for each sample
    /// (`f(x) = Σ coef·K(sv, x) + intercept_`).
    ///
    /// # Errors
    ///
    /// Returns `Ok` always (provided for API symmetry).
    pub fn decision_function(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let n_samples = x.nrows();
        let mut result = Array1::<F>::zeros(n_samples);
        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            result[s] = self.decision_value(&xi);
        }
        Ok(result)
    }

    /// Indices of the support vectors into the training set, ascending
    /// (`support_`, `sklearn/svm/_base.py:318-410`).
    #[must_use]
    pub fn support(&self) -> Array1<usize> {
        Array1::from_vec(self.sv_indices.clone())
    }

    /// The support vectors, shape `(n_SV, n_features)` (`support_vectors_`).
    #[must_use]
    pub fn support_vectors(&self) -> Array2<F> {
        let n_sv = self.support_vectors.len();
        let n_features = self.support_vectors.first().map_or(0, Vec::len);
        let mut out = Array2::<F>::zeros((n_sv, n_features));
        for (r, sv) in self.support_vectors.iter().enumerate() {
            for (c, &v) in sv.iter().enumerate() {
                out[[r, c]] = v;
            }
        }
        out
    }

    /// Number of support vectors. For nu-SVR `n_support_` has size 1.
    #[must_use]
    pub fn n_support(&self) -> Vec<usize> {
        vec![self.support_vectors.len()]
    }

    /// Prediction coefficients `α*_i − α_i`, shape `(1, n_SV)` (`dual_coef_`).
    /// No sign flip applies to SVR (`_base.py:260` restricts it to
    /// `c_svc`/`nu_svc`).
    #[must_use]
    pub fn dual_coef(&self) -> Array2<F> {
        let n_sv = self.dual_coefs.len();
        let mut out = Array2::<F>::zeros((1, n_sv));
        for (c, &v) in self.dual_coefs.iter().enumerate() {
            out[[0, c]] = v;
        }
        out
    }

    /// The intercept, length 1 (`intercept_`).
    #[must_use]
    pub fn intercept(&self) -> Array1<F> {
        Array1::from_vec(vec![self.bias])
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedNuSVR<F, K>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values (`= decision_function`).
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.decision_function(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::{LinearKernel, RbfKernel};
    use ndarray::array;

    /// Oracle: `NuSVC(kernel='linear', nu=0.5)` on the binary 6×2 set
    /// (R-CHAR-3, live sklearn 1.5.2):
    /// `support_ [1,2,3,5]`, `dual_coef_ [[-0.0227,-0.0455,0.0455,0.0227]]`,
    /// `intercept_ [-1.75]`, `df [-1.25,-1.0,-1.0,0.75,1.0,1.0]`,
    /// `predict [0,0,0,1,1,1]`.
    #[test]
    fn test_nusvc_oracle_attrs() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.5)
            .with_max_iter(200_000)
            .with_tol(1e-7);
        let fitted = model.fit(&x, &y).unwrap();

        // support_
        let support: Vec<usize> = fitted.support().to_vec();
        assert_eq!(support, vec![1, 2, 3, 5], "support_ mismatch");

        // n_support_
        assert_eq!(fitted.n_support(), vec![2, 2], "n_support_ mismatch");

        // dual_coef_
        let dc = fitted.dual_coef();
        assert_eq!(dc.dim(), (1, 4));
        let oracle_dc = [-0.022727, -0.045455, 0.045455, 0.022727];
        for (k, &o) in oracle_dc.iter().enumerate() {
            assert!(
                (dc[[0, k]] - o).abs() < 1e-2,
                "dual_coef_[{k}] = {} vs oracle {o}",
                dc[[0, k]]
            );
        }

        // intercept_
        let ic = fitted.intercept();
        assert!(
            (ic[0] - (-1.75)).abs() < 1e-2,
            "intercept_ = {} vs -1.75",
            ic[0]
        );

        // decision_function (binary)
        let df = fitted.decision_function(&x).unwrap();
        let b = df.as_binary().expect("binary scores");
        let oracle_df = [-1.25, -1.0, -1.0, 0.75, 1.0, 1.0];
        for (i, &o) in oracle_df.iter().enumerate() {
            assert!((b[i] - o).abs() < 1e-2, "df[{i}] = {} vs {o}", b[i]);
        }

        // predict
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds, array![0usize, 0, 0, 1, 1, 1]);

        // coef_ (linear): oracle [[0.25, 0.25]]
        let coef = fitted.coef().expect("linear coef_");
        assert!((coef[[0, 0]] - 0.25).abs() < 1e-2);
        assert!((coef[[0, 1]] - 0.25).abs() < 1e-2);
    }

    /// Oracle: `NuSVR(kernel='linear', nu=0.5, C=1.0)` on `X=[[1],[2],[3],[4]]`,
    /// `y=[1,5,2,8]` (R-CHAR-3, live sklearn 1.5.2):
    /// `predict [2.5,3.5,4.5,5.5]`, `support_ [2,3]`, `dual_coef_ [[-1,1]]`,
    /// `intercept_ [1.5]`.
    #[test]
    fn test_nusvr_oracle_attrs() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 5.0, 2.0, 8.0];

        let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.5)
            .with_c(1.0)
            .with_max_iter(500_000)
            .with_tol(1e-8);
        let fitted = model.fit(&x, &y).unwrap();

        let preds = fitted.predict(&x).unwrap();
        let oracle = [2.5, 3.5, 4.5, 5.5];
        for (i, &o) in oracle.iter().enumerate() {
            assert!(
                (preds[i] - o).abs() < 1e-2,
                "predict[{i}] = {} vs {o}",
                preds[i]
            );
        }

        let support: Vec<usize> = fitted.support().to_vec();
        assert_eq!(support, vec![2, 3], "support_ mismatch");

        let dc = fitted.dual_coef();
        assert_eq!(dc.dim(), (1, 2));
        assert!(
            (dc[[0, 0]] - (-1.0)).abs() < 1e-2,
            "dual_coef_[0] = {}",
            dc[[0, 0]]
        );
        assert!(
            (dc[[0, 1]] - 1.0).abs() < 1e-2,
            "dual_coef_[1] = {}",
            dc[[0, 1]]
        );

        let ic = fitted.intercept();
        assert!((ic[0] - 1.5).abs() < 1e-2, "intercept_ = {} vs 1.5", ic[0]);
    }

    #[test]
    fn test_nusvc_default_c_is_one() {
        // NuSVR default C must be 1.0 (sklearn `_classes.py:1531`).
        let model = NuSVR::<f64, LinearKernel>::new(LinearKernel);
        assert!((model.c - 1.0).abs() < 1e-12);
        assert!((model.nu - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_nusvc_rbf() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let model = NuSVC::new(RbfKernel::with_gamma(0.5)).with_nu(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_nusvc_multiclass() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(0.5);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();
        // 3 classes -> ovr decision function (n, 3).
        let df = fitted.decision_function(&x).unwrap();
        assert_eq!(df.as_multiclass().map(|m| m.dim()), Some((9, 3)));
        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 7, "Expected at least 7 correct, got {correct}");
    }

    #[test]
    fn test_nusvc_invalid_nu_zero() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvc_invalid_nu_above_one() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];
        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(1.5);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvc_nu_one() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1];
        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel).with_nu(1.0);
        assert!(model.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_nusvr_invalid_nu() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];
        assert!(NuSVR::new(LinearKernel).with_nu(0.0).fit(&x, &y).is_err());
        assert!(NuSVR::new(LinearKernel).with_nu(-0.5).fit(&x, &y).is_err());
    }

    #[test]
    fn test_nusvr_decision_function_equals_predict() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];
        let model = NuSVR::new(LinearKernel).with_nu(0.5).with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let df = fitted.decision_function(&x).unwrap();
        let preds = fitted.predict(&x).unwrap();
        for i in 0..4 {
            assert!((df[i] - preds[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nusvc_builder_pattern() {
        let model = NuSVC::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.3)
            .with_tol(1e-4)
            .with_max_iter(5000)
            .with_cache_size(2048);
        assert!((model.nu - 0.3).abs() < 1e-10);
        assert!((model.tol - 1e-4).abs() < 1e-10);
        assert_eq!(model.max_iter, 5000);
        assert_eq!(model.cache_size, 2048);
    }

    #[test]
    fn test_nusvr_builder_pattern() {
        let model = NuSVR::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.8)
            .with_c(2.0)
            .with_tol(1e-5)
            .with_max_iter(20000)
            .with_cache_size(512);
        assert!((model.nu - 0.8).abs() < 1e-10);
        assert!((model.c - 2.0).abs() < 1e-10);
        assert!((model.tol - 1e-5).abs() < 1e-10);
        assert_eq!(model.max_iter, 20000);
        assert_eq!(model.cache_size, 512);
    }
}
