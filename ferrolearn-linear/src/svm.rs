//! Support Vector Machine with kernel trick.
//!
//! This module provides [`SVC`] (classification) and [`SVR`] (regression)
//! support vector machines trained using the **Sequential Minimal Optimization
//! (SMO)** algorithm (Platt, 1998).
//!
//! # Kernels
//!
//! Four built-in kernels are provided:
//!
//! - [`LinearKernel`]: `K(x, y) = x . y`
//! - [`RbfKernel`]: `K(x, y) = exp(-gamma * ||x - y||^2)`
//! - [`PolynomialKernel`]: `K(x, y) = (gamma * x . y + coef0)^degree`
//! - [`SigmoidKernel`]: `K(x, y) = tanh(gamma * x . y + coef0)`
//!
//! Users can implement the [`Kernel`] trait for custom kernels.
//!
//! # Multiclass
//!
//! `SVC` uses a one-vs-one strategy for multiclass classification.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::svm::{SVC, LinearKernel};
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  2.0, 1.0,  1.0, 2.0,
//!     5.0, 5.0,  6.0, 5.0,  5.0, 6.0,
//! ]).unwrap();
//! let y = array![0usize, 0, 0, 1, 1, 1];
//!
//! let model = SVC::<f64, LinearKernel>::new(LinearKernel);
//! let fitted = model.fit(&x, &y).unwrap();
//! let preds = fitted.predict(&x).unwrap();
//! assert_eq!(preds.len(), 6);
//! ```
//!
//! ## REQ status
//!
//! Binary (R-DEFER-2): SHIPPED = impl + non-test production consumer + tests +
//! green oracle verification; NOT-STARTED = open blocker `#`. `SVC`/`SVR`/
//! `FittedSVC`/`FittedSVR`/`Kernel` + the four kernels are boundary estimator
//! types re-exported at the crate root (`pub use svm::{…}` in `lib.rs`) and
//! consumed by `nu_svm.rs` (`NuSVC`/`NuSVR` delegate to `SVC`/`SVR`) and
//! `one_class_svm.rs` (uses `Kernel`) — non-test production consumers; under
//! S5/R-DEFER-1 the fitted-attribute accessors are part of that boundary public
//! API surface. See `.design/linear/svm.md`.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (kernels + gamma scale/auto/float) | SHIPPED | The four kernel formulas + the three-way `pub enum Gamma<F> { Scale, Auto, Value }` (default `Scale`) resolved at fit time by `fn resolve_gamma in svm.rs` + `fn resolved_for_fit in svm.rs`: `Scale`=`1/(n_features·X.var())` (`_base.py:238-239`), `Auto`=`1/n_features` (`_base.py:240-241`), `Value(v)`=verbatim (`_base.py:242-243`). Builders `RbfKernel::with_gamma`/`with_gamma_scale`/`with_gamma_auto`. Non-test consumer: the kernel `gamma` field is resolved in the production `fn fit in svm.rs` (`self.kernel.resolved_for_fit(x)`). Pinned: `divergence_pin2_rbf_default_scale_gamma` (scale, green) + in-module `test_svc_gamma_auto_decision_function in svm.rs` (`_gamma=0.5`, df `[-0.9996,-0.9999,-0.9999,0.9999,0.9999,0.9996]` vs live `SVC(kernel='rbf',gamma='auto')`, R-CHAR-3, 1e-2) + `test_svc_gamma_scale_still_default` (`_gamma=0.118421`). |
//! | REQ-2 (C-SVC SMO fit) | SHIPPED | `fn smo_binary in svm.rs` (Fan-Chen-Lin WSS) converges to libsvm's `α`; pinned by `divergence_pin5_binary_fitted_attributes in tests/divergence_svm_fit.rs` (`dual_coef_ [[-0.0408,-0.0408,0.0816]]`, `support_ [1,2,3]`, `intercept_ [-1.8565]` vs live `SVC(kernel='linear',C=1.0)`). |
//! | REQ-3 (fitted attrs + binary sign flip) | SHIPPED | `FittedSVC::{support,support_vectors,n_support,dual_coef,intercept,coef} in svm.rs` emit the libsvm layout with the binary sign flip (`_base.py:258-262`); `coef_` is linear-only (`_base.py:642-666`). Pinned by `divergence_pin5_*` (binary) + `divergence_pin6_multiclass_dual_coef_packing` (multiclass `(n_class-1,n_SV)` packing). |
//! | REQ-4 (decision_function shape/sign/ovr) | SHIPPED | `FittedSVC::decision_function in svm.rs` returns the `SvmScores<F>` enum: binary -> `SvmScores::Binary` 1-D `(n,)` = `-raw_ovo.ravel()` (positive -> `classes_[1]`, `_base.py:538-539`); multiclass -> `SvmScores::Multiclass` `(n, n_classes)` via `fn ovr_decision_function in svm.rs` (default `SvmDecisionShape::Ovr`, transcribed from `multiclass.py:520-562`) applied to `dec<0`/`-dec` (`_base.py:780`), or raw `(n, n·(n-1)/2)` for `SvmDecisionShape::Ovo`. `SVC::decision_function_shape` field + `with_decision_function_shape`. Sign normalized: `fn raw_ovo` negates `decision_value_binary` to restore libsvm's lower-index-class-`+1` ovo convention. Pinned by `divergence_pin8_multiclass_ovr_decision_function` (ovr `(9,3)` row0 `[2.2366,0.8167,-0.1833]`, row3 `[1.0606,2.2262,-0.2333]`), `divergence_pin9_multiclass_ovo_decision_function` (ovo `(9,3)` row0 `[1.2222,1.2222,0.0]`), `divergence_pin10_binary_shape_contract` (binary 1-D `(6,)`) in `tests/divergence_svm_fit.rs` (R-CHAR-3, 1e-2). Consumer: `FittedNuSVC::decision_function in nu_svm.rs` delegates (non-test, propagates `SvmScores`). |
//! | REQ-5 (predict + tie-break) | SHIPPED | `fn predict in svm.rs` (FittedSVC) does libsvm ovo voting and breaks vote ties toward the LOWER class index via a strictly-greater first-max scan (keeps the first/lowest-index maximum since `classes` is `np.unique(y)`-sorted), matching libsvm/sklearn `super().predict` (`_base.py:813-814`) instead of `max_by_key`'s last-maximum. Pinned by `divergence_pin3_predict_labels` (separable-set labels) + `divergence_pin11_ovo_vote_tie_break_lower_index` (4-class vote tie `(0,2,2,2)` at `q=(-0.21,-8.976)` -> class 1) in `tests/divergence_svm_fit.rs` vs live `SVC(kernel='linear',C=1.0)`. |
//! | REQ-6 (epsilon-SVR) | SHIPPED | `fn smo_svr in svm.rs` + `FittedSVR::{support,support_vectors,n_support,dual_coef,intercept}`; pinned by `divergence_pin4_svr_predict_values` (predict) + `divergence_pin7_svr_fitted_attributes` (`support_ [0,5]`, `dual_coef_ [[-0.392,0.392]]`, `intercept_ [0.14]` vs live `SVR(kernel='linear',C=100,epsilon=0.1)`). |
//! | REQ-7 (multiclass one-vs-one) | SHIPPED | `fn fit in svm.rs` (SVC) trains one `smo_binary` per class pair, `classes` = `np.unique(y)`; pinned by `divergence_pin6_multiclass_dual_coef_packing` (3-class `dual_coef_ (2,6)` libsvm packing, `support_ [1,2,3,5,6,7]`, `n_support_ [2,2,2]`, `intercept_ [1.2222,1.2222,0.0]`). |
//! | REQ-8 (constructor param surface + defaults) | SHIPPED | `shrinking` (`SVC`/`SVR`, default `true`, `with_shrinking`; accepted for API parity, shrinking-invariant optimum so DOES NOT alter results — R-DEV-7); `break_ties` (`SVC`, default `false`, `with_break_ties`; `fn predict in svm.rs` ovr-argmax branch for `break_ties=true`+ovr+`n_classes>2`, `InvalidParameter` for the ovo combo, `_base.py:801-814`); default alignment `cache_size=200`, `max_iter=0` (= sklearn `-1`, no iteration limit; the `smo_binary`/`smo_svr` loops treat `0` as unbounded); REQ-1's `gamma` enum (`scale`/`auto`/float); and now **`class_weight`** (`SVC`, `pub class_weight: ClassWeight<F>` default `None`, `with_class_weight`). `fn compute_class_weight in svm.rs` mirrors `sklearn.utils.compute_class_weight` as called by `BaseSVC._validate_targets` (`class_weight_ = compute_class_weight(class_weight, classes, y)`, `_base.py:740`): `None`→1.0, `Balanced`→`n_samples/(n_classes·count_c)` (`_classes.py:122-124`), `Explicit`→1.0 default overridden by map. `fn smo_binary in svm.rs` now takes per-class box bounds `(cp, cn)` (the `y=+1`/`y=-1` upper bounds) instead of a scalar `c`, applied in the WSS `in_up`/`in_low` tests, the analytic-update box clip, and the free-SV bias recovery (`0<alpha_i<C_i`); when `cp==cn` the math is identical to before (the 13 divergence pins stay green). `fn fit in svm.rs` (SVC) computes `weights = compute_class_weight(...)` ONCE over the full `y`, then per ovo pair `(ci,cj)`: `cp = C·weights[cj]`, `cn = C·weights[ci]` (libsvm `weighted_C`, `_base.py:740`). Non-test consumer: `fn fit in svm.rs` consumes `self.class_weight` (the boundary `SVC`/`FittedSVC` types are re-exported at the crate root + consumed by `nu_svm.rs`). Pinned: `test_svc_class_weight_smoke`/`test_compute_class_weight_balanced`/`test_svc_break_ties_changes_label`/`test_svc_break_ties_ovo_errors`/`test_svc_default_params in svm.rs` (live oracle on the imbalanced 8×2 set: None `dual_coef_ [[-0.5,-1,1,0.5]]`/`intercept_ [-2.0]`/`support_ [1,3,5,6]`; balanced `[[-0.8,-0.8,1.3333,0.2667]]`/`-1.6667`; `{0:1,1:5}` `support_ [1,3,4,5]`/`-2.0`; R-CHAR-3, 1e-2; None≠balanced intercept). **R-DEV-7 design difference (preserved contract, NOT a gap):** estimator-level `kernel`(string-select)/`degree`/`coef0` are the type parameter `K`, set by construction; `random_state` is unused (ferrolearn's SMO is deterministic). `class_weight` is SVC-only (sklearn SVR has no `class_weight`). |
//! | REQ-9 (probability / predict_proba) | SHIPPED | `pub probability: bool` field on `SVC` (default `false`, `with_probability`) + `prob_a`/`prob_b`/`probability` on `FittedSVC`. The DETERMINISTIC Platt machinery is transcribed from libsvm: `fn sigmoid_train in svm.rs` (Newton iteration + prior init + target smoothing + step-halving line search, `svm.cpp:1919-2030`), `fn sigmoid_predict in svm.rs` (overflow-safe form, `svm.cpp:2032-2040`), `fn multiclass_probability in svm.rs` (Wu-Lin-Weng 2004 coupling, `svm.cpp:2043-2104`). `fn platt_cv_sigmoid in svm.rs` runs a per-ovo-pair 5-fold CV at fit time when `probability=true` (`svm.cpp:2107-2203`). `FittedSVC::predict_proba in svm.rs` builds the pairwise matrix via `sigmoid_predict` (clamped `[1e-7,1-1e-7]`, `svm.cpp:2937`) -> `multiclass_probability` -> `(n,n_classes)`; binary -> `[P(classes[0]),P(classes[1])]`; rows sum to 1. `FittedSVC::predict_log_proba` = `predict_proba.ln()` (`_base.py:866-894`). `probability=false` -> `InvalidParameter` carrying sklearn's `NotFittedError` text "predict_proba is not available when fitted with probability=False" (`_base.py:856-860`; no `NotFitted` variant by R-DEV-4 typestate). **RNG-CV boundary (documented divergence, NOT a gap):** libsvm's CV fold permutation is RNG-seeded, so sklearn's `probA_`/`probB_`/`predict_proba` are NON-DETERMINISTIC across `random_state` (`probA_` = -0.7749 at rs=0 vs -1.0541 at rs=1; the docstring admits CV-dependence). ferrolearn uses a DETERMINISTIC contiguous 5-fold split (analogous to the documented SGD shuffle boundary), so it does NOT bit-match sklearn's predict_proba VALUES — only the deterministic machinery + structural invariants + the raise contract are verified (R-CHAR-3: the asserted invariants are sklearn's DOCUMENTED contract, not copied values). Pinned by `test_svc_predict_proba_raises_when_probability_false`/`test_svc_predict_proba_binary_rows_sum_to_one`/`test_svc_predict_proba_binary_monotone_in_decision`/`test_svc_predict_log_proba_equals_log_of_proba`/`test_svc_predict_proba_multiclass_rows_sum_to_one`/`test_sigmoid_predict_overflow_safe`/`test_multiclass_probability_binary_reduces_to_pairwise in svm.rs`. Non-test consumer: `fn fit in svm.rs` (SVC) consumes `self.probability` (the boundary `SVC`/`FittedSVC` types are re-exported at the crate root + consumed by `nu_svm.rs`). |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open #643. `svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE). |
//! | REQ-11 (non-finite input rejected) | SHIPPED | Both fit entries reject any NaN/+/-inf BEFORE the SMO solve with `FerroError::InvalidParameter`, mirroring sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, …)` (`_base.py:190-197`, default `force_all_finite=True`) -> `ValueError("Input X contains NaN.")` / `"Input y contains NaN."` / `"… contains infinity …"`. `SVC::fit in svm.rs` checks `X` (`y` is `Array1<usize>` labels, finite by type); `SVR::fit in svm.rs` checks `X` AND the float target `y`. ferrolearn's `Fit::fit` signature has no `sample_weight` argument, so the sklearn `sample_weight`-finiteness raise has no fit-entry counterpart here. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (the guard never fires on finite input — the 13+ SVC/SVR divergence pins stay green). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): NaN/+inf/-inf in X for both, NaN/inf in y for SVR, all raise `ValueError` (`tests/divergence_svm_nonfinite.rs::{svc_*,svr_*}`). Non-test consumer: the existing `Fit::fit` consumers + the crate-root `pub use svm::{SVC, SVR, …}` re-exports. (#2269) |

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Kernel trait and built-in kernels
// ---------------------------------------------------------------------------

/// The `gamma` coefficient for the RBF / polynomial / sigmoid kernels,
/// mirroring scikit-learn's three-way `gamma` parameter
/// (`sklearn/svm/_base.py:235-243`,
/// `StrOptions({"scale", "auto"}) | Interval(Real, 0.0, None)`).
///
/// Resolved at fit time against the training matrix `X`
/// ([`Kernel::resolved_for_fit`]):
///
/// - [`Gamma::Scale`] (default): `1 / (n_features · X.var())` where `X.var()`
///   is the population variance (ddof=0) of the whole flattened `X`
///   (`_base.py:238-239`). When `X.var() == 0` sklearn falls back to `1.0`.
/// - [`Gamma::Auto`]: `1 / n_features` (`_base.py:240-241`).
/// - [`Gamma::Value`]: the float verbatim (`_base.py:242-243`).
///
/// The default is [`Gamma::Scale`], matching sklearn's `gamma='scale'`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gamma<F> {
    /// `gamma='scale'` (sklearn default): `1 / (n_features · X.var())`.
    Scale,
    /// `gamma='auto'`: `1 / n_features`.
    Auto,
    /// An explicit float gamma, used verbatim.
    Value(F),
}

impl<F> Default for Gamma<F> {
    /// sklearn's default is `gamma='scale'`.
    fn default() -> Self {
        Gamma::Scale
    }
}

/// Per-class scaling of the regularization parameter `C` for [`SVC`].
///
/// Mirrors `sklearn.svm.SVC`'s `class_weight` parameter
/// (`sklearn/svm/_classes.py:118-124`, constraint `{None, dict, 'balanced'}`):
/// it sets the C of class `i` to `class_weight[i]·C` (libsvm's per-class
/// `weighted_C[i] = C·class_weight_[i]`). The expanded per-class weights are
/// computed by [`compute_class_weight`] following
/// `sklearn.utils.compute_class_weight` semantics, as called from
/// `BaseSVC._validate_targets`
/// (`self.class_weight_ = compute_class_weight(self.class_weight, classes=cls,
/// y=y_)`, `sklearn/svm/_base.py:740`).
///
/// This mirrors [`crate::linear_svc::ClassWeight`] for cross-estimator
/// consistency, but is defined locally (no cross-import of `linear_svc`
/// internals).
#[derive(Debug, Clone, Default)]
pub enum ClassWeight<F> {
    /// Uniform weights (all classes weighted `1.0`). The default
    /// (`class_weight=None`).
    #[default]
    None,
    /// Balanced weights `n_samples / (n_classes · count_c)` per class `c`,
    /// matching `sklearn.utils.compute_class_weight("balanced", ...)`
    /// (`_classes.py:122-124`: `n_samples / (n_classes * np.bincount(y))`).
    Balanced,
    /// Explicit class-label -> weight map. Classes absent from the map default
    /// to `1.0`, matching the dict branch of `compute_class_weight`.
    Explicit(Vec<(usize, F)>),
}

/// Compute the expanded per-class weight vector aligned to `classes`
/// (sorted ascending, matching sklearn's `classes_ = np.unique(y)`).
///
/// Faithful to `sklearn.utils.compute_class_weight`, as called by
/// `BaseSVC._validate_targets`
/// (`compute_class_weight(self.class_weight, classes=cls, y=y_)`,
/// `sklearn/svm/_base.py:740`):
/// - `None` -> all `1.0`.
/// - `Balanced` -> `n_samples / (n_classes · count_c)` per class `c`,
///   where `count_c` is the number of samples with label `c`
///   (`_classes.py:122-124`).
/// - `Explicit(map)` -> `1.0` default, overridden by the map entries matched by
///   class label.
///
/// `classes` is the sorted unique label set; `y` is the per-sample label array.
/// Mirrors `ferrolearn_linear::linear_svc::compute_class_weight` exactly.
fn compute_class_weight<F: Float>(cw: &ClassWeight<F>, classes: &[usize], y: &[usize]) -> Vec<F> {
    match cw {
        ClassWeight::None => vec![F::one(); classes.len()],
        ClassWeight::Balanced => {
            // `recip_freq = len(y) / (n_classes * bincount(y))`, indexed per
            // class (`_classes.py:124`).
            let n_samples = F::from(y.len()).unwrap_or_else(F::zero);
            let n_classes = F::from(classes.len()).unwrap_or_else(F::one);
            classes
                .iter()
                .map(|&c| {
                    let count = y.iter().filter(|&&label| label == c).count();
                    let count_f = F::from(count).unwrap_or_else(F::one);
                    if count_f > F::zero() {
                        n_samples / (n_classes * count_f)
                    } else {
                        F::one()
                    }
                })
                .collect()
        }
        ClassWeight::Explicit(map) => classes
            .iter()
            .map(|&c| {
                map.iter()
                    .find(|(label, _)| *label == c)
                    .map_or_else(F::one, |(_, w)| *w)
            })
            .collect(),
    }
}

/// A kernel function for SVM.
///
/// Computes the inner product of two vectors in a (possibly implicit)
/// higher-dimensional feature space.
pub trait Kernel<F: Float>: Clone + Send + Sync {
    /// Compute the kernel value between two vectors.
    fn compute(&self, x: &[F], y: &[F]) -> F;

    /// Resolve any data-dependent kernel parameters against the training data
    /// at fit time, returning a copy of the kernel with those parameters fixed.
    ///
    /// For kernels with a [`Gamma<F>`] parameter, the three-way `gamma`
    /// resolution mirrors scikit-learn (`sklearn/svm/_base.py:235-243`):
    /// [`Gamma::Scale`] (default) -> `1 / (n_features * X.var())` where
    /// `X.var()` is the population variance (ddof=0) over the whole flattened
    /// training matrix; [`Gamma::Auto`] -> `1 / n_features`; [`Gamma::Value`]
    /// is left verbatim. After resolution the stored `gamma` is always a
    /// concrete [`Gamma::Value`].
    ///
    /// The default implementation is a no-op (returns `self.clone()`), which is
    /// correct for parameter-free kernels such as [`LinearKernel`].
    #[must_use]
    fn resolved_for_fit(&self, _x: &Array2<F>) -> Self
    where
        Self: Sized,
    {
        self.clone()
    }

    /// Whether this is the linear kernel `K(x, y) = x . y`.
    ///
    /// sklearn exposes `coef_` (the primal weight vector
    /// `dual_coef_ @ support_vectors_`) ONLY for the linear kernel and raises
    /// `AttributeError` otherwise (`sklearn/svm/_base.py:650-651`). The default
    /// is `false`; [`LinearKernel`] overrides it to `true`.
    #[must_use]
    fn is_linear(&self) -> bool {
        false
    }
}

/// Compute the population variance (ddof=0) of all elements of `x`, mirroring
/// numpy's `X.var()` (`mean((x - mean)^2)`). Returns `None` when `x` is empty.
fn population_variance<F: Float>(x: &Array2<F>) -> Option<F> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    let count = F::from(n)?;
    let sum = x.iter().fold(F::zero(), |acc, &v| acc + v);
    let mean = sum / count;
    let sq = x
        .iter()
        .fold(F::zero(), |acc, &v| acc + (v - mean) * (v - mean));
    Some(sq / count)
}

/// Extract the concrete float from a [`Gamma<F>`] for a direct `compute` call
/// without training data. After [`Kernel::resolved_for_fit`] the gamma is
/// always a [`Gamma::Value`], so this is the live path; an unresolved
/// `Scale`/`Auto` (e.g. a kernel used standalone outside a fit) has no `X` to
/// resolve against and falls back to `1.0`, matching the prior default-gamma
/// behavior of a directly-evaluated kernel.
fn gamma_value_or_one<F: Float>(gamma: Gamma<F>) -> F {
    match gamma {
        Gamma::Value(v) => v,
        Gamma::Scale | Gamma::Auto => F::one(),
    }
}

/// Resolve a [`Gamma<F>`] spec against the training matrix `X`, returning the
/// concrete float gamma, mirroring scikit-learn (`sklearn/svm/_base.py:235-243`):
///
/// - [`Gamma::Scale`] -> `1 / (n_features * X.var())` (`_base.py:238-239`).
///   When `X.var() == 0` (constant `X`) or `X` is empty, sklearn falls back to
///   `1.0` (`_base.py:239`: `if X_var != 0 else 1.0`), so we do the same
///   (avoiding a non-finite gamma).
/// - [`Gamma::Auto`] -> `1 / n_features` (`_base.py:240-241`).
/// - [`Gamma::Value`] -> the float verbatim (`_base.py:242-243`).
fn resolve_gamma<F: Float>(gamma: Gamma<F>, x: &Array2<F>) -> F {
    match gamma {
        Gamma::Value(v) => v,
        Gamma::Auto => match F::from(x.ncols()) {
            Some(nf) if nf > F::zero() => F::one() / nf,
            _ => F::one(),
        },
        Gamma::Scale => {
            let n_features = match F::from(x.ncols()) {
                Some(nf) if nf > F::zero() => nf,
                _ => return F::one(),
            };
            match population_variance(x) {
                Some(var) if var > F::zero() => F::one() / (n_features * var),
                // var == 0 (constant X) or empty: sklearn falls back to 1.0.
                _ => F::one(),
            }
        }
    }
}

/// Linear kernel: `K(x, y) = x . y`.
#[derive(Debug, Clone, Copy)]
pub struct LinearKernel;

impl<F: Float> Kernel<F> for LinearKernel {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        x.iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }

    fn is_linear(&self) -> bool {
        true
    }
}

/// Radial Basis Function (Gaussian) kernel.
///
/// `K(x, y) = exp(-gamma * ||x - y||^2)`
#[derive(Debug, Clone, Copy)]
pub struct RbfKernel<F> {
    /// The gamma parameter, a three-way [`Gamma<F>`] spec resolved at fit time
    /// (`sklearn/svm/_base.py:235-243`). Default [`Gamma::Scale`]
    /// (= `1 / (n_features * X.var())`); [`Gamma::Auto`] = `1 / n_features`;
    /// [`Gamma::Value`] is used verbatim.
    pub gamma: Gamma<F>,
}

impl<F: Float> RbfKernel<F> {
    /// Create a new RBF kernel with the default `gamma='scale'`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: Gamma::Scale,
        }
    }

    /// Create a new RBF kernel with an explicit float gamma
    /// (`gamma=<float>`, [`Gamma::Value`]).
    #[must_use]
    pub fn with_gamma(gamma: F) -> Self {
        Self {
            gamma: Gamma::Value(gamma),
        }
    }

    /// Create a new RBF kernel with `gamma='scale'` ([`Gamma::Scale`],
    /// sklearn's default = `1 / (n_features * X.var())`).
    #[must_use]
    pub fn with_gamma_scale() -> Self {
        Self {
            gamma: Gamma::Scale,
        }
    }

    /// Create a new RBF kernel with `gamma='auto'` ([`Gamma::Auto`]
    /// = `1 / n_features`).
    #[must_use]
    pub fn with_gamma_auto() -> Self {
        Self { gamma: Gamma::Auto }
    }
}

impl<F: Float> Default for RbfKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for RbfKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = gamma_value_or_one(self.gamma);
        let sq_dist = x.iter().zip(y.iter()).fold(F::zero(), |acc, (&a, &b)| {
            let d = a - b;
            acc + d * d
        });
        (-gamma * sq_dist).exp()
    }

    fn resolved_for_fit(&self, x: &Array2<F>) -> Self {
        Self {
            gamma: Gamma::Value(resolve_gamma(self.gamma, x)),
        }
    }
}

/// Polynomial kernel: `K(x, y) = (gamma * x . y + coef0)^degree`.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialKernel<F> {
    /// The gamma parameter, a three-way [`Gamma<F>`] spec resolved at fit time
    /// (`sklearn/svm/_base.py:235-243`). Default [`Gamma::Scale`].
    pub gamma: Gamma<F>,
    /// Polynomial degree.
    pub degree: usize,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> PolynomialKernel<F> {
    /// Create a new polynomial kernel with defaults (`gamma='scale'`,
    /// `degree=3`, `coef0=0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: Gamma::Scale,
            degree: 3,
            coef0: F::zero(),
        }
    }
}

impl<F: Float> Default for PolynomialKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for PolynomialKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = gamma_value_or_one(self.gamma);
        let dot: F = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        let val = gamma * dot + self.coef0;
        let mut result = F::one();
        for _ in 0..self.degree {
            result = result * val;
        }
        result
    }

    fn resolved_for_fit(&self, x: &Array2<F>) -> Self {
        Self {
            gamma: Gamma::Value(resolve_gamma(self.gamma, x)),
            degree: self.degree,
            coef0: self.coef0,
        }
    }
}

/// Sigmoid kernel: `K(x, y) = tanh(gamma * x . y + coef0)`.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidKernel<F> {
    /// The gamma parameter, a three-way [`Gamma<F>`] spec resolved at fit time
    /// (`sklearn/svm/_base.py:235-243`). Default [`Gamma::Scale`].
    pub gamma: Gamma<F>,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> SigmoidKernel<F> {
    /// Create a new sigmoid kernel with defaults (`gamma='scale'`, `coef0=0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: Gamma::Scale,
            coef0: F::zero(),
        }
    }
}

impl<F: Float> Default for SigmoidKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for SigmoidKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = gamma_value_or_one(self.gamma);
        let dot: F = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        (gamma * dot + self.coef0).tanh()
    }

    fn resolved_for_fit(&self, x: &Array2<F>) -> Self {
        Self {
            gamma: Gamma::Value(resolve_gamma(self.gamma, x)),
            coef0: self.coef0,
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel cache (LRU)
// ---------------------------------------------------------------------------

/// Simple LRU cache for kernel evaluations.
struct KernelCache<F> {
    cache: HashMap<(usize, usize), F>,
    order: Vec<(usize, usize)>,
    capacity: usize,
}

impl<F: Float> KernelCache<F> {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            order: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_or_compute<K: Kernel<F>>(
        &mut self,
        i: usize,
        j: usize,
        kernel: &K,
        data: &[Vec<F>],
    ) -> F {
        let key = if i <= j { (i, j) } else { (j, i) };
        if let Some(&val) = self.cache.get(&key) {
            return val;
        }
        let val = kernel.compute(&data[i], &data[j]);
        if self.order.len() >= self.capacity
            && let Some(old_key) = self.order.first().copied()
        {
            self.cache.remove(&old_key);
            self.order.remove(0);
        }
        self.cache.insert(key, val);
        self.order.push(key);
        val
    }
}

// ---------------------------------------------------------------------------
// SMO solver for binary SVM
// ---------------------------------------------------------------------------

/// Result of a binary SMO solve.
struct SmoResult<F> {
    alphas: Vec<F>,
    bias: F,
}

/// SMO implementation (Platt 1998, Fan-Chen-Lin 2005 WSS).
///
/// Uses the dual gradient `grad_i = (Q * alpha)_i - 1` where
/// `Q_{ij} = y_i * y_j * K(x_i, x_j)`. Bias is computed after
/// convergence from the KKT conditions.
#[allow(
    clippy::too_many_arguments,
    reason = "the per-class box bounds (cp, cn) are separate args mirroring \
              libsvm's per-sample upper bound C_i (Cp for y=+1, Cn for y=-1)"
)]
fn smo_binary<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    labels: &[F],
    kernel: &K,
    cp: F,
    cn: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Result<SmoResult<F>, FerroError> {
    let n = data.len();
    let mut alphas = vec![F::zero(); n];
    let mut cache = KernelCache::new(cache_size);

    // Per-sample box upper bound `C_i = (y_i > 0 ? Cp : Cn)` (libsvm `GETI`):
    // `class_weight` scales C per class so the +1 group (class_pos) gets `Cp`
    // and the -1 group (class_neg) gets `Cn`. When `cp == cn` the box is the
    // uniform `[0, C]` of the no-class-weight case.
    let c_of = |i: usize| -> F { if labels[i] > F::zero() { cp } else { cn } };

    // Gradient of the dual objective: grad_i = (Q*alpha)_i - 1
    // where Q_{ij} = y_i * y_j * K(x_i, x_j).
    // Initially alpha = 0, so grad_i = -1 for all i.
    let mut grad: Vec<F> = vec![-F::one(); n];

    let two = F::one() + F::one();
    let eps = F::from(1e-12).unwrap_or_else(F::epsilon);

    // `max_iter == 0` is the sklearn `max_iter=-1` ("no iteration limit",
    // libsvm runs to convergence) sentinel — the SMO loop then runs until the
    // KKT gap closes. A non-zero `max_iter` caps the iteration count.
    let mut iter = 0usize;
    loop {
        if max_iter != 0 && iter >= max_iter {
            break;
        }
        iter += 1;
        // Working set selection (Fan-Chen-Lin 2005):
        // I_up  = {i : (y_i=+1 and alpha_i < C) or (y_i=-1 and alpha_i > 0)}
        // I_low = {j : (y_j=+1 and alpha_j > 0) or (y_j=-1 and alpha_j < C)}
        // Select i = argmax_{t in I_up}  -y_t * grad_t
        // Select j = argmin_{t in I_low} -y_t * grad_t

        let mut i_up = None;
        let mut max_val = F::neg_infinity();
        let mut j_low = None;
        let mut min_val = F::infinity();

        for t in 0..n {
            let val = -labels[t] * grad[t];
            let c_t = c_of(t);

            let in_up = (labels[t] > F::zero() && alphas[t] < c_t - eps)
                || (labels[t] < F::zero() && alphas[t] > eps);

            let in_low = (labels[t] > F::zero() && alphas[t] > eps)
                || (labels[t] < F::zero() && alphas[t] < c_t - eps);

            if in_up && val > max_val {
                max_val = val;
                i_up = Some(t);
            }
            if in_low && val < min_val {
                min_val = val;
                j_low = Some(t);
            }
        }

        // Stopping criterion: KKT gap < tol
        if i_up.is_none() || j_low.is_none() || max_val - min_val < tol {
            break;
        }

        let i = i_up.unwrap();
        let j = j_low.unwrap();

        if i == j {
            break;
        }

        // Compute second-order info
        let kii = cache.get_or_compute(i, i, kernel, data);
        let kjj = cache.get_or_compute(j, j, kernel, data);
        let kij = cache.get_or_compute(i, j, kernel, data);
        let eta = kii + kjj - two * kij;

        if eta <= eps {
            continue;
        }

        // Bounds for alpha_j, respecting the per-sample box bounds
        // `0 <= alpha_i <= C_i` and `0 <= alpha_j <= C_j` (libsvm allows a
        // different upper bound per sample under `class_weight`).
        let old_ai = alphas[i];
        let old_aj = alphas[j];
        let ci = c_of(i);
        let cj = c_of(j);

        let (lo, hi) = if labels[i] == labels[j] {
            // alpha_i + alpha_j = sum (const): alpha_j in
            // [max(0, sum - C_i), min(C_j, sum)].
            let sum = old_ai + old_aj;
            ((sum - ci).max(F::zero()), sum.min(cj))
        } else {
            // alpha_i = alpha_j - diff (const diff): alpha_j in
            // [max(0, diff), min(C_j, C_i + diff)].
            let diff = old_aj - old_ai;
            (diff.max(F::zero()), (ci + diff).min(cj))
        };

        if (hi - lo).abs() < eps {
            continue;
        }

        // Analytic update for alpha_j (Platt 1998).
        // E_k = y_k * grad_k (dual error, where grad = Q*alpha - e).
        // alpha_j_new = alpha_j + y_j * (E_i - E_j) / eta
        //             = alpha_j + y_j * (y_i * grad_i - y_j * grad_j) / eta
        let mut new_aj = old_aj + labels[j] * (labels[i] * grad[i] - labels[j] * grad[j]) / eta;

        // Clip to [lo, hi]
        if new_aj > hi {
            new_aj = hi;
        }
        if new_aj < lo {
            new_aj = lo;
        }

        if (new_aj - old_aj).abs() < eps {
            continue;
        }

        let new_ai = old_ai + labels[i] * labels[j] * (old_aj - new_aj);

        alphas[i] = new_ai;
        alphas[j] = new_aj;

        // Update dual gradient: grad_k += delta_alpha_i * Q_{k,i} + delta_alpha_j * Q_{k,j}
        // where Q_{k,t} = y_k * y_t * K(k,t)
        let delta_ai = new_ai - old_ai;
        let delta_aj = new_aj - old_aj;

        for (k, grad_k) in grad.iter_mut().enumerate() {
            let kki = cache.get_or_compute(k, i, kernel, data);
            let kkj = cache.get_or_compute(k, j, kernel, data);
            *grad_k = *grad_k
                + delta_ai * labels[k] * labels[i] * kki
                + delta_aj * labels[k] * labels[j] * kkj;
        }
    }

    // Compute bias from KKT conditions.
    // For support vectors with 0 < alpha_i < C:
    //   y_i * (sum_j alpha_j * y_j * K(i,j) + b) = 1
    //   b = y_i - sum_j alpha_j * y_j * K(i,j)
    // (since y_i^2 = 1, y_i * (y_i * f) = f, so b = 1/y_i - sum = y_i - sum)
    let mut b_sum = F::zero();
    let mut b_count = 0usize;

    for i in 0..n {
        if alphas[i] > eps && alphas[i] < c_of(i) - eps {
            // This is a free support vector (`0 < alpha_i < C_i`).
            let mut f_no_b = F::zero();
            for j in 0..n {
                if alphas[j] > eps {
                    f_no_b =
                        f_no_b + alphas[j] * labels[j] * cache.get_or_compute(i, j, kernel, data);
                }
            }
            b_sum = b_sum + labels[i] - f_no_b;
            b_count += 1;
        }
    }

    let bias = if b_count > 0 {
        b_sum / F::from(b_count).unwrap()
    } else {
        // Fallback: use all support vectors (bounded ones too)
        let mut b_sum_all = F::zero();
        let mut b_count_all = 0usize;
        for i in 0..n {
            if alphas[i] > eps {
                let mut f_no_b = F::zero();
                for j in 0..n {
                    if alphas[j] > eps {
                        f_no_b = f_no_b
                            + alphas[j] * labels[j] * cache.get_or_compute(i, j, kernel, data);
                    }
                }
                b_sum_all = b_sum_all + labels[i] - f_no_b;
                b_count_all += 1;
            }
        }
        if b_count_all > 0 {
            b_sum_all / F::from(b_count_all).unwrap()
        } else {
            F::zero()
        }
    };

    Ok(SmoResult { alphas, bias })
}

// ---------------------------------------------------------------------------
// Platt scaling (probability estimates)
// ---------------------------------------------------------------------------

/// Fit the Platt sigmoid `P(y=+1 | f) = 1 / (1 + exp(A·f + B))` to a set of
/// decision values `dec_values` with binary labels `labels` (`+1` / `-1`),
/// returning the `(A, B)` parameters.
///
/// A faithful transcription of libsvm's `sigmoid_train`
/// (`sklearn/svm/src/libsvm/svm.cpp:1919-2030`): the prior-based initial point
/// (`A=0`, `B=log((prior0+1)/(prior1+1))`), the `t` target smoothing
/// (`hiTarget=(prior1+1)/(prior1+2)`, `loTarget=1/(prior0+2)`), the Newton
/// iteration with the regularized Hessian (`H' = H + sigma·I`,
/// `sigma=1e-12`), the gradient/Hessian accumulation, the step-halving line
/// search (`min_step=1e-10`, sufficient-decrease constant `0.0001`),
/// `max_iter=100`, and the `eps=1e-5` gradient stopping criterion. The
/// overflow-safe `fApB>=0` branching matches the C code exactly.
#[allow(
    clippy::too_many_lines,
    reason = "a faithful one-to-one transcription of libsvm's sigmoid_train \
              Newton loop (svm.cpp:1919-2030); splitting it would obscure the \
              line-by-line correspondence to the C oracle"
)]
fn sigmoid_train<F: Float>(dec_values: &[F], labels: &[F]) -> (F, F) {
    let l = dec_values.len();
    let zero = F::zero();
    let one = F::one();
    let two = one + one;

    let mut prior1 = zero;
    let mut prior0 = zero;
    for &lab in labels {
        if lab > zero {
            prior1 = prior1 + one;
        } else {
            prior0 = prior0 + one;
        }
    }

    let max_iter = 100usize;
    let min_step = F::from(1e-10).unwrap_or_else(F::epsilon);
    let sigma = F::from(1e-12).unwrap_or_else(F::epsilon);
    let eps = F::from(1e-5).unwrap_or_else(F::epsilon);
    let suff = F::from(0.0001).unwrap_or_else(F::epsilon);

    let hi_target = (prior1 + one) / (prior1 + two);
    let lo_target = one / (prior0 + two);

    // Per-sample target smoothed labels `t`.
    let t: Vec<F> = labels
        .iter()
        .map(|&lab| if lab > zero { hi_target } else { lo_target })
        .collect();

    // Initial point and initial function value.
    let mut a = zero;
    let mut b = ((prior0 + one) / (prior1 + one)).ln();

    let funcval = |a: F, b: F| -> F {
        let mut fval = zero;
        for i in 0..l {
            let f_ap_b = dec_values[i] * a + b;
            if f_ap_b >= zero {
                fval = fval + t[i] * f_ap_b + (one + (-f_ap_b).exp()).ln();
            } else {
                fval = fval + (t[i] - one) * f_ap_b + (one + f_ap_b.exp()).ln();
            }
        }
        fval
    };

    let mut fval = funcval(a, b);

    for _iter in 0..max_iter {
        // Update gradient and Hessian (H' = H + sigma·I).
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = zero;
        let mut g1 = zero;
        let mut g2 = zero;
        for i in 0..l {
            let f_ap_b = dec_values[i] * a + b;
            let (p, q) = if f_ap_b >= zero {
                let e = (-f_ap_b).exp();
                (e / (one + e), one / (one + e))
            } else {
                let e = f_ap_b.exp();
                (one / (one + e), e / (one + e))
            };
            let d2 = p * q;
            h11 = h11 + dec_values[i] * dec_values[i] * d2;
            h22 = h22 + d2;
            h21 = h21 + dec_values[i] * d2;
            let d1 = t[i] - p;
            g1 = g1 + dec_values[i] * d1;
            g2 = g2 + d1;
        }

        // Stopping criterion.
        if g1.abs() < eps && g2.abs() < eps {
            break;
        }

        // Newton direction: -inv(H')·g.
        let det = h11 * h22 - h21 * h21;
        let d_a = -(h22 * g1 - h21 * g2) / det;
        let d_b = -(-h21 * g1 + h11 * g2) / det;
        let gd = g1 * d_a + g2 * d_b;

        // Line search (step halving).
        let mut stepsize = one;
        while stepsize >= min_step {
            let new_a = a + stepsize * d_a;
            let new_b = b + stepsize * d_b;
            let newf = funcval(new_a, new_b);
            if newf < fval + suff * stepsize * gd {
                a = new_a;
                b = new_b;
                fval = newf;
                break;
            }
            stepsize = stepsize / two;
        }

        if stepsize < min_step {
            // Line search failed — libsvm bails out of the Newton loop.
            break;
        }
    }

    (a, b)
}

/// Evaluate the Platt sigmoid `P(y=+1 | f) = 1 / (1 + exp(A·f + B))` at a single
/// decision value, in the overflow-safe form of libsvm's `sigmoid_predict`
/// (`sklearn/svm/src/libsvm/svm.cpp:2032-2040`):
/// `fApB = decision·A + B`; if `fApB >= 0` return `exp(-fApB)/(1+exp(-fApB))`,
/// else `1/(1+exp(fApB))` (avoiding `exp` overflow / catastrophic
/// cancellation).
fn sigmoid_predict<F: Float>(decision: F, a: F, b: F) -> F {
    let f_ap_b = decision * a + b;
    if f_ap_b >= F::zero() {
        let e = (-f_ap_b).exp();
        e / (F::one() + e)
    } else {
        F::one() / (F::one() + f_ap_b.exp())
    }
}

/// Wu-Lin-Weng (2004) pairwise coupling ("Method 2"): given the `k×k` pairwise
/// probability matrix `r` (where `r[i][j] = P(class i | class i or j)`),
/// produce the `k` coupled class probabilities `p`.
///
/// A faithful transcription of libsvm's `multiclass_probability`
/// (`sklearn/svm/src/libsvm/svm.cpp:2043-2104`): build the `Q` matrix from the
/// pairwise probabilities, then run the fixed-point iteration
/// (`max_iter = max(100, k)`, `eps = 0.005/k`) that minimizes the coupling
/// objective, normalized so the returned probabilities sum to 1.
fn multiclass_probability<F: Float>(k: usize, r: &Array2<F>) -> Vec<F> {
    let zero = F::zero();
    let one = F::one();
    let k_f = F::from(k).unwrap_or(one);

    let mut p = vec![one / k_f; k];
    // Q[t][j].
    let mut q = Array2::<F>::zeros((k, k));
    for t in 0..k {
        for j in 0..t {
            q[[t, t]] = q[[t, t]] + r[[j, t]] * r[[j, t]];
            q[[t, j]] = q[[j, t]];
        }
        for j in (t + 1)..k {
            q[[t, t]] = q[[t, t]] + r[[j, t]] * r[[j, t]];
            q[[t, j]] = -r[[j, t]] * r[[t, j]];
        }
    }

    let max_iter = 100.max(k);
    let eps = F::from(0.005).unwrap_or_else(F::epsilon) / k_f;
    let mut qp = vec![zero; k];

    for _iter in 0..max_iter {
        // Recompute Qp, pQp for numerical accuracy.
        let mut p_qp = zero;
        for t in 0..k {
            qp[t] = zero;
            for j in 0..k {
                qp[t] = qp[t] + q[[t, j]] * p[j];
            }
            p_qp = p_qp + p[t] * qp[t];
        }
        let mut max_error = zero;
        for &qpt in qp.iter().take(k) {
            let error = (qpt - p_qp).abs();
            if error > max_error {
                max_error = error;
            }
        }
        if max_error < eps {
            break;
        }

        for t in 0..k {
            let qtt = q[[t, t]];
            if qtt == zero {
                continue;
            }
            let diff = (-qp[t] + p_qp) / qtt;
            p[t] = p[t] + diff;
            p_qp = (p_qp + diff * (diff * qtt + two_qp(qp[t]))) / (one + diff) / (one + diff);
            for j in 0..k {
                qp[j] = (qp[j] + diff * q[[t, j]]) / (one + diff);
                p[j] = p[j] / (one + diff);
            }
        }
    }

    p
}

/// `2·x` helper for [`multiclass_probability`] (libsvm `2*Qp[t]`).
#[inline]
fn two_qp<F: Float>(x: F) -> F {
    x + x
}

/// Decision value of a freshly-trained binary SMO sub-model on a query sample,
/// in ferrolearn's sign convention (positive favors the `+1` label, i.e. the
/// higher-index `class_pos` group).
fn sub_decision_value<F: Float, K: Kernel<F>>(
    sv_data: &[Vec<F>],
    sv_coefs: &[F],
    bias: F,
    kernel: &K,
    q: &[F],
) -> F {
    let mut val = bias;
    for (sv, &coef) in sv_data.iter().zip(sv_coefs.iter()) {
        val = val + coef * kernel.compute(sv, q);
    }
    val
}

/// A freshly-trained binary sub-model in this crate's (ferrolearn) sign
/// convention: support-vector feature rows, their coefficients
/// (`alpha_i·y_i`, `class_pos = +1` side), and the decision bias such that
/// [`sub_decision_value`] is positive favoring `class_pos`. Returned by the
/// per-solver TRAINER closure that [`platt_cv_sigmoid`] invokes on each CV
/// training fold.
pub(crate) type SubModel<F> = (Vec<Vec<F>>, Vec<F>, F);

/// Fit the per-ovo-pair Platt sigmoid `(A, B)` via a DETERMINISTIC 5-fold CV
/// over the pair's samples, mirroring libsvm's `svm_binary_svc_probability`
/// (`sklearn/svm/src/libsvm/svm.cpp:2107-2203`) EXCEPT for the fold
/// permutation.
///
/// libsvm shuffles the fold assignment with an RNG seeded by `random_state`
/// (`svm.cpp:2116-2122`), which makes the resulting `(A, B)` (sklearn's
/// `probA_`/`probB_`) and thus `predict_proba` NON-DETERMINISTIC across
/// `random_state`. To keep ferrolearn deterministic (it has no libsvm RNG
/// seed; cf. the documented SGD shuffle boundary, R-DEV-4), the folds here use a
/// DETERMINISTIC CLASS-STRATIFIED assignment instead of libsvm's random shuffle:
/// each sample's fold is its WITHIN-CLASS running index modulo `nr_fold`. Because
/// the per-ovo-pair samples arrive GROUPED by class, a naive contiguous
/// `[i·l/5, (i+1)·l/5)` split would make whole folds single-class — so the 4-fold
/// training set could miss a class entirely and `sigmoid_train` would collapse to
/// the degenerate `(A, B) = (0, 0)`. Stratifying within class keeps both classes
/// in every training set (when each class has ≥2 samples), restoring libsvm's
/// structural contract (a non-degenerate sigmoid) without its randomness. The
/// rest is a faithful transcription: train a binary sub-model on the 4 training
/// folds,
/// `predict_values` the held-out fold (in libsvm sign), with the degenerate
/// one-class-fold fallbacks (`+1` / `-1` / `0`, `svm.cpp:2161-2169`), then
/// [`sigmoid_train`] over all out-of-fold decisions.
///
/// # The `train_fold` trainer abstraction
///
/// libsvm's `svm_binary_svc_probability` trains each CV sub-model with the
/// SAME `svm_type` as the outer model (`svm.cpp:2147-2150`, a copy of the
/// outer `svm_parameter` with `probability=0`): C-SVC sub-models for `SVC`,
/// NU-SVC sub-models for `NuSVC`. ferrolearn threads that choice through a
/// `train_fold` closure: given the training-fold `(data, labels)` (in
/// ferrolearn sign, `class_pos = +1`), it returns the fitted [`SubModel`]
/// (`Some`) or `None` on a degenerate/failed sub-solve. `SVC` passes a
/// closure wrapping [`smo_binary`] (C-SVC); `NuSVC` passes a closure wrapping
/// [`solve_nu_svc`] (the genuine `Solver_NU`). The CV split, degenerate-fold
/// fallbacks, held-out scoring via [`sub_decision_value`], and the final
/// [`sigmoid_train`] are SOLVER-AGNOSTIC, so SVC's `(A, B)` is byte-identical
/// to the pre-refactor inline-`smo_binary` path.
///
/// `sub_labels` is ferrolearn's sign (`+1` = higher-index `class_pos`,
/// `-1` = lower-index `class_neg`). The decision values and labels passed to
/// [`sigmoid_train`] are converted to libsvm sign (`+1` = lower-index
/// `class_neg`, matching `raw_ovo`) so the fitted `(A, B)` is consistent with
/// the raw ovo decision used by [`FittedSVC::predict_proba`].
pub(crate) fn platt_cv_sigmoid<F: Float, K: Kernel<F>>(
    sub_data: &[Vec<F>],
    sub_labels: &[F],
    kernel: &K,
    train_fold: impl Fn(&[Vec<F>], &[F]) -> Option<SubModel<F>>,
) -> (F, F) {
    let l = sub_data.len();
    let nr_fold = 5usize;
    // Out-of-fold decision value per sample, in libsvm sign (+1 = class_neg).
    let mut dec_values = vec![F::zero(); l];

    // DETERMINISTIC class-stratified fold assignment. libsvm shuffles the fold
    // permutation with an RNG (`svm.cpp:2116-2122`) so each fold mixes both
    // classes; ferrolearn stays deterministic (no libsvm RNG seed; cf. the
    // sanctioned SGD-shuffle boundary, R-DEV-4) by instead assigning each sample
    // to a fold by its WITHIN-CLASS running index modulo `nr_fold`. The
    // per-ovo-pair samples arrive GROUPED by class (`[class_neg..., class_pos...]`,
    // built by the `FittedSVC::fit` loop), so a CONTIGUOUS `[i·l/5, (i+1)·l/5)`
    // split would make whole folds single-class and the 4-fold training set could
    // MISS a class entirely → a trivial sub-model → constant held-out decisions →
    // `sigmoid_train` returns the degenerate `(A, B) = (0, 0)`. Spreading each
    // class proportionally across all folds keeps BOTH classes in every training
    // set whenever each class has ≥2 samples, restoring the structural contract
    // (a non-degenerate sigmoid) at every input — matching libsvm's intent
    // without its randomness.
    let mut pos_seen = 0usize;
    let mut neg_seen = 0usize;
    let mut fold_of = vec![0usize; l];
    for (j, &lab) in sub_labels.iter().enumerate() {
        if lab > F::zero() {
            fold_of[j] = pos_seen % nr_fold;
            pos_seen += 1;
        } else {
            fold_of[j] = neg_seen % nr_fold;
            neg_seen += 1;
        }
    }

    for fold in 0..nr_fold {
        // Training set = all samples NOT assigned to this fold.
        let mut tr_data: Vec<Vec<F>> = Vec::with_capacity(l);
        let mut tr_labels: Vec<F> = Vec::with_capacity(l);
        for (j, row) in sub_data.iter().enumerate() {
            if fold_of[j] != fold {
                tr_data.push(row.clone());
                tr_labels.push(sub_labels[j]);
            }
        }

        // Count classes in the training folds (ferrolearn sign).
        let mut p_count = 0usize;
        let mut n_count = 0usize;
        for &lab in &tr_labels {
            if lab > F::zero() {
                p_count += 1;
            } else {
                n_count += 1;
            }
        }

        // Degenerate folds: libsvm assigns a constant decision
        // (`svm.cpp:2161-2169`). In ferrolearn sign a held-out sample gets
        // +1 (all-positive train), -1 (all-negative train), or 0 (empty); we
        // store the libsvm-sign value = negation. The held-out fold is now the
        // (non-contiguous) set `{ j : fold_of[j] == fold }`, not a slice.
        let held_out = (0..l).filter(|&j| fold_of[j] == fold);
        if p_count == 0 && n_count == 0 {
            for j in held_out {
                dec_values[j] = F::zero();
            }
            continue;
        } else if n_count == 0 {
            // train all +1 (class_pos) -> ferrolearn dec +1 -> libsvm -1.
            for j in held_out {
                dec_values[j] = -F::one();
            }
            continue;
        } else if p_count == 0 {
            for j in held_out {
                dec_values[j] = F::one();
            }
            continue;
        }

        // Train a probability-free sub-model on the training folds via the
        // per-solver trainer (C-SVC for SVC, NU-SVC for NuSVC).
        let Some((sv_data, sv_coefs, bias)) = train_fold(&tr_data, &tr_labels) else {
            // A failed/degenerate sub-solve falls back to a neutral 0 decision.
            for j in held_out {
                dec_values[j] = F::zero();
            }
            continue;
        };

        // Score the held-out fold; store in libsvm sign (negate ferrolearn).
        for j in held_out {
            let dec_ferro = sub_decision_value(&sv_data, &sv_coefs, bias, kernel, &sub_data[j]);
            dec_values[j] = -dec_ferro;
        }
    }

    // libsvm labels for sigmoid_train: +1 = lower-index class_neg, matching
    // the libsvm-sign decision values (so `-sub_labels`).
    let libsvm_labels: Vec<F> = sub_labels.iter().map(|&lab| -lab).collect();
    sigmoid_train(&dec_values, &libsvm_labels)
}

// ---------------------------------------------------------------------------
// decision_function shape + scores
// ---------------------------------------------------------------------------

/// The shape convention for [`FittedSVC::decision_function`] in the multiclass
/// case, mirroring scikit-learn's `SVC.decision_function_shape`
/// (`sklearn/svm/_base.py:778-781`).
///
/// - [`SvmDecisionShape::Ovr`] (default): one-vs-rest scores, shape
///   `(n_samples, n_classes)`, produced by the `_ovr_decision_function`
///   transform (`sklearn/utils/multiclass.py:520-562`).
/// - [`SvmDecisionShape::Ovo`]: the raw one-vs-one decision values, shape
///   `(n_samples, n_class·(n_class-1)/2)`.
///
/// The binary case is unaffected (it always collapses to a 1-D `(n_samples,)`
/// score, `_base.py:538-539`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvmDecisionShape {
    /// One-vs-rest: `(n_samples, n_classes)` via `_ovr_decision_function`
    /// (sklearn's default).
    #[default]
    Ovr,
    /// One-vs-one: raw `(n_samples, n_class·(n_class-1)/2)` decision values.
    Ovo,
}

/// The result of [`FittedSVC::decision_function`].
///
/// Mirrors scikit-learn's polymorphic `SVC.decision_function` return
/// (`sklearn/svm/_base.py:536-541, 778-781`): the binary case collapses the
/// single ovo column to a 1-D `(n_samples,)` array (`-dec.ravel()`,
/// `_base.py:538-539`), while the multiclass case returns
/// `(n_samples, n_classes)` (ovr, default) or
/// `(n_samples, n·(n-1)/2)` (ovo). Structurally parallels
/// [`crate::linear_svc::DecisionScores`] for cross-estimator consistency.
#[derive(Debug, Clone, PartialEq)]
pub enum SvmScores<F> {
    /// Binary decision values, shape `(n_samples,)`. A POSITIVE value predicts
    /// `classes_[1]` (`-dec.ravel()`, `_base.py:538-539`).
    Binary(Array1<F>),
    /// Multiclass decision values: `(n_samples, n_classes)` for
    /// [`SvmDecisionShape::Ovr`] or `(n_samples, n·(n-1)/2)` for
    /// [`SvmDecisionShape::Ovo`].
    Multiclass(Array2<F>),
}

impl<F: Clone> SvmScores<F> {
    /// Number of samples scored (the leading axis length in both variants).
    #[must_use]
    pub fn n_samples(&self) -> usize {
        match self {
            SvmScores::Binary(v) => v.len(),
            SvmScores::Multiclass(m) => m.nrows(),
        }
    }

    /// Borrow the binary 1-D scores, if this is the binary case.
    #[must_use]
    pub fn as_binary(&self) -> Option<&Array1<F>> {
        match self {
            SvmScores::Binary(v) => Some(v),
            SvmScores::Multiclass(_) => None,
        }
    }

    /// Borrow the multiclass score matrix, if this is the multiclass case.
    #[must_use]
    pub fn as_multiclass(&self) -> Option<&Array2<F>> {
        match self {
            SvmScores::Multiclass(m) => Some(m),
            SvmScores::Binary(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SVC (Support Vector Classifier)
// ---------------------------------------------------------------------------

/// Support Vector Classifier.
///
/// Uses Sequential Minimal Optimization (SMO) to solve the dual QP.
/// Supports multiclass via one-vs-one strategy.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`LinearKernel`], [`RbfKernel`]).
#[derive(Debug, Clone)]
pub struct SVC<F, K> {
    /// The kernel function.
    pub kernel: K,
    /// Regularization parameter (penalty for misclassification).
    pub c: F,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations. `0` is the sklearn `max_iter=-1`
    /// sentinel meaning **no iteration limit** (the SMO runs to convergence);
    /// a non-zero value caps the iteration count
    /// (`sklearn/svm/_classes.py`, `max_iter` default `-1`).
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache (perf-only; default `200` to
    /// match sklearn's `cache_size=200`).
    pub cache_size: usize,
    /// Whether to use libsvm's shrinking heuristic
    /// (`sklearn/svm/_base.py:339`, `_classes.py` `shrinking=True`).
    ///
    /// ferrolearn's SMO has no shrinking heuristic: shrinking is a libsvm
    /// performance optimization that does NOT change the converged optimum
    /// (R-DEV-7). This flag is accepted for API parity (default `true`,
    /// matching sklearn) but DOES NOT alter the fitted result — the converged
    /// `α`/`dual_coef_`/`intercept_` are shrinking-invariant.
    pub shrinking: bool,
    /// The multiclass `decision_function` shape convention
    /// (`sklearn/svm/_base.py:778-781`); default
    /// [`SvmDecisionShape::Ovr`] (sklearn's `decision_function_shape='ovr'`).
    pub decision_function_shape: SvmDecisionShape,
    /// Whether `predict` breaks ties by the one-vs-rest decision confidence
    /// instead of the libsvm vote (`break_ties`, `sklearn/svm/_classes.py`
    /// default `False`; semantics in `BaseSVC.predict`,
    /// `sklearn/svm/_base.py:801-814`).
    ///
    /// When `true` AND [`SvmDecisionShape::Ovr`] AND `n_classes > 2`,
    /// `predict = argmax(decision_function(X))` (the ovr decision, which breaks
    /// ties by confidence); otherwise the libsvm ovo vote (with lower-index
    /// tie-break) is used. `break_ties=true` with [`SvmDecisionShape::Ovo`] is
    /// rejected at predict time (`InvalidParameter`), matching sklearn
    /// (`_base.py:801-804`).
    pub break_ties: bool,
    /// Per-class scaling of `C` (`class_weight`, `sklearn/svm/_classes.py:118-124`).
    /// Default [`ClassWeight::None`] (all classes weighted `1.0`). For an ovo
    /// pair `(a, b)` with `a < b`, the C of the `y=+1` group (class `b`) is
    /// `C·class_weight_[b]` and the C of the `y=-1` group (class `a`) is
    /// `C·class_weight_[a]`; the weights are computed ONCE over the full `y`
    /// by [`compute_class_weight`] (`_base.py:740`).
    pub class_weight: ClassWeight<F>,
    /// Whether to enable Platt-scaling probability estimates
    /// (`probability`, `sklearn/svm/_classes.py`, default `False`).
    ///
    /// When `true`, [`Fit::fit`] runs an internal 5-fold cross-validation per
    /// one-vs-one pair, fits a sigmoid `1/(1+exp(A·f+B))` over the out-of-fold
    /// decision values ([`sigmoid_train`], libsvm `svm_binary_svc_probability`,
    /// `svm.cpp:2107-2203`), and stores the per-pair `(A, B)` so
    /// [`FittedSVC::predict_proba`]/[`FittedSVC::predict_log_proba`] are
    /// available. When `false` (the default) `predict_proba` returns an error
    /// (`_base.py:820-827`).
    ///
    /// **RNG boundary (documented divergence).** libsvm's
    /// `svm_binary_svc_probability` shuffles the CV fold assignment with an
    /// RNG seeded by `random_state`, so sklearn's `probA_`/`probB_` (and hence
    /// the exact `predict_proba` values) are NON-DETERMINISTIC across
    /// `random_state` — the docstring itself warns "the results can be slightly
    /// different than those obtained by predict". ferrolearn instead uses a
    /// DETERMINISTIC 5-fold split (contiguous folds, no RNG shuffle), so it
    /// CANNOT and DOES NOT bit-match sklearn's `predict_proba` values. What is
    /// reproduced exactly is the DETERMINISTIC machinery ([`sigmoid_train`],
    /// [`sigmoid_predict`], [`multiclass_probability`]) and the STRUCTURAL
    /// contract (rows sum to 1, entries in `[0, 1]`, monotone in the binary
    /// decision value, the raise-when-`probability=false`). This is analogous
    /// to the SGD shuffle boundary already documented in this codebase.
    pub probability: bool,
}

impl<F: Float, K: Kernel<F>> SVC<F, K> {
    /// Create a new `SVC` with the given kernel and default hyperparameters
    /// matching sklearn (`sklearn/svm/_classes.py` `SVC.__init__`).
    ///
    /// Defaults: `C = 1.0`, `tol = 1e-3`, `max_iter = 0` (= sklearn `-1`, no
    /// iteration limit), `cache_size = 200`, `shrinking = true`,
    /// `decision_function_shape = Ovr`, `break_ties = false`,
    /// `class_weight = None`, `probability = false`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            max_iter: 0,
            cache_size: 200,
            shrinking: true,
            decision_function_shape: SvmDecisionShape::Ovr,
            break_ties: false,
            class_weight: ClassWeight::None,
            probability: false,
        }
    }

    /// Enable/disable Platt-scaling probability estimates (`sklearn`
    /// `probability`, default `false`). When `true`, [`Fit::fit`] runs the
    /// internal per-pair 5-fold CV + [`sigmoid_train`] so
    /// [`FittedSVC::predict_proba`]/[`FittedSVC::predict_log_proba`] are
    /// available; when `false` they return an error.
    ///
    /// See the [`SVC::probability`] field doc for the documented RNG-CV
    /// exact-value boundary (sklearn is non-deterministic across
    /// `random_state`; only the deterministic machinery + structural
    /// invariants + the raise contract are reproduced).
    #[must_use]
    pub fn with_probability(mut self, probability: bool) -> Self {
        self.probability = probability;
        self
    }

    /// Set the per-class `C` scaling (`sklearn` `class_weight`,
    /// `_classes.py:118-124`). [`ClassWeight::None`] (default) leaves every
    /// class at `1.0`; [`ClassWeight::Balanced`] uses
    /// `n_samples / (n_classes · count_c)`; [`ClassWeight::Explicit`] takes a
    /// `(label, weight)` map (unlisted classes default to `1.0`).
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight<F>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Set the `shrinking` flag (`sklearn` `shrinking`, default `true`).
    ///
    /// Accepted for API parity; does NOT alter the converged optimum
    /// (ferrolearn's SMO has no shrinking heuristic — R-DEV-7).
    #[must_use]
    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    /// Set the `break_ties` flag (`sklearn` `break_ties`, default `false`,
    /// `sklearn/svm/_base.py:801-814`).
    #[must_use]
    pub fn with_break_ties(mut self, break_ties: bool) -> Self {
        self.break_ties = break_ties;
        self
    }

    /// Set the multiclass `decision_function` shape convention
    /// (`'ovr'` default / `'ovo'`, `sklearn/svm/_base.py:778-781`).
    #[must_use]
    pub fn with_decision_function_shape(mut self, shape: SvmDecisionShape) -> Self {
        self.decision_function_shape = shape;
        self
    }

    /// Set the regularization parameter C.
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

    /// Set the maximum number of SMO iterations.
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

/// A single binary SVM model (one pair of classes in one-vs-one).
#[derive(Debug, Clone)]
struct BinarySvm<F> {
    /// Support vectors (stored as rows).
    support_vectors: Vec<Vec<F>>,
    /// Original training-row index of each support vector (parallel to
    /// `support_vectors`/`dual_coefs`). Used to build the global, per-class
    /// grouped `support_` set (`sklearn/svm/_base.py:318-410`).
    sv_indices: Vec<usize>,
    /// Dual coefficients: `alpha_i * y_i` for each support vector, where this
    /// crate maps the lower-index class (`class_neg`) to `y = -1` and the
    /// higher-index class (`class_pos`) to `y = +1`. NOTE this is the OPPOSITE
    /// sign convention to libsvm internally (libsvm gives the lower-index class
    /// `+1`); the public-attribute layout compensates in
    /// [`FittedSVC::dual_coef`].
    dual_coefs: Vec<F>,
    /// Bias term.
    bias: F,
    /// The two class labels: (negative_class, positive_class). `class_neg` is
    /// the lower class index and `class_pos` the higher (the ovo pair `(a, b)`
    /// with `a < b`).
    class_neg: usize,
    class_pos: usize,
}

/// Fitted Support Vector Classifier.
///
/// Stores one binary SVM per pair of classes (one-vs-one). Implements
/// [`Predict`] to produce class labels.
#[derive(Debug, Clone)]
pub struct FittedSVC<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// One binary SVM per class pair, in libsvm ovo pair order
    /// `(0,1),(0,2),...,(0,k-1),(1,2),...` (the `(ci,cj)` double loop).
    binary_models: Vec<BinarySvm<F>>,
    /// Sorted unique classes (`classes_ = np.unique(y)`).
    classes: Vec<usize>,
    /// The training feature matrix, retained so the libsvm-layout fitted
    /// attributes (`support_`, `support_vectors_`) can index back into the
    /// original rows (`sklearn/svm/_base.py:318-410`).
    x_train: Array2<F>,
    /// The training labels (class index per row), retained so `support_` can
    /// be grouped by class.
    y_train: Vec<usize>,
    /// The multiclass `decision_function` shape convention carried over from
    /// the unfitted [`SVC`] (`sklearn/svm/_base.py:778-781`).
    decision_function_shape: SvmDecisionShape,
    /// The `break_ties` flag carried over from the unfitted [`SVC`]
    /// (`sklearn/svm/_base.py:801-814`).
    break_ties: bool,
    /// Whether probability estimates were fitted (`probability`,
    /// `sklearn/svm/_classes.py`). When `false`, [`Self::predict_proba`]
    /// returns an error (`_base.py:820-827`).
    probability: bool,
    /// Per-ovo-pair Platt sigmoid `A` parameter (`probA_`,
    /// `sklearn/svm/src/libsvm/svm.cpp:2200` via `sigmoid_train`), parallel to
    /// `binary_models`. Empty when `probability == false`.
    prob_a: Vec<F>,
    /// Per-ovo-pair Platt sigmoid `B` parameter (`probB_`), parallel to
    /// `binary_models`. Empty when `probability == false`.
    prob_b: Vec<F>,
}

/// One ovo binary sub-model in **this crate's sign convention** (higher-index
/// `class_pos` is the `+1` side, matching [`BinarySvm`] and
/// [`FittedSVC::decision_value_binary`]). Used by [`FittedSVC::from_nu_ovo`] to
/// assemble a nu-SVC fitted model that reuses all of [`FittedSVC`]'s accessors
/// / `decision_function` / `predict`.
///
/// The nu-SVC solver ([`solve_nu_svc`]) is fed the per-pair labels in this same
/// convention (`class_pos = +1`), so `sv_coefs`/`bias_internal` are already in
/// this-crate sign and `from_nu_ovo` stores them verbatim.
pub(crate) struct NuOvoPair<F> {
    /// Support-vector feature rows for this pair.
    pub sv_data: Vec<Vec<F>>,
    /// Per-SV coefficient `alpha·y/r` (this-crate sign, `class_pos = +1`),
    /// equal to the public binary `dual_coef_` value (the nu_svc binary flip
    /// `public = -internal` cancels with `internal = -stored`).
    pub sv_coefs: Vec<F>,
    /// Original training-row index of each support vector.
    pub sv_indices: Vec<usize>,
    /// Decision bias for the `+1`-side (`class_pos`) in this crate's
    /// convention (`f(x) = Σ sv_coef·K + bias_internal`).
    pub bias_internal: F,
    /// Lower-index class label (this crate's `-1` side).
    pub class_neg: usize,
    /// Higher-index class label (this crate's `+1` side).
    pub class_pos: usize,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> FittedSVC<F, K> {
    /// Assemble a [`FittedSVC`] from per-ovo-pair nu-SVC sub-models (in libsvm
    /// sign convention) so that [`NuSVC`](crate::nu_svm::NuSVC) reuses the full
    /// libsvm-layout fitted-attribute machinery (`support_`/`dual_coef_`/
    /// `intercept_`/`coef_`/`decision_function`/`predict`) without duplicating
    /// it (`sklearn/svm/_base.py:318-410`).
    ///
    /// Each [`NuOvoPair`] is already in this crate's [`BinarySvm`] sign
    /// convention (higher-index `class_pos` as the `+1` side, because
    /// [`solve_nu_svc`] is fed labels in that convention), so the coefficients
    /// and bias are stored verbatim. The resulting public `dual_coef_`/
    /// `intercept_` then carry the binary nu_svc sign flip exactly as `c_svc`
    /// does (`_base.py:258-262`, predicate `_impl in ["c_svc","nu_svc"]`).
    ///
    /// When `probability` is `true`, `prob_a`/`prob_b` are the per-ovo-pair
    /// sigmoid `(A, B)` parameters fitted by [`platt_cv_sigmoid`] with the
    /// NU-SVC sub-solver ([`solve_nu_svc`]) ([`NuSVC`](crate::nu_svm::NuSVC)
    /// REQ-9), so the assembled [`FittedSVC`]'s
    /// [`Self::predict_proba`]/[`Self::predict_log_proba`] work identically to
    /// a `probability=true` C-SVC fit — the coupling/`sigmoid_predict` path is
    /// solver-agnostic (it consumes only `raw_ovo` + `prob_a`/`prob_b`). When
    /// `probability` is `false`, `prob_a`/`prob_b` are empty.
    #[allow(
        clippy::too_many_arguments,
        reason = "carries the full nu-ovo assembly plus the probability state \
                  (probability flag + per-pair probA/probB) in one constructor"
    )]
    pub(crate) fn from_nu_ovo(
        kernel: K,
        pairs: Vec<NuOvoPair<F>>,
        classes: Vec<usize>,
        x_train: Array2<F>,
        y_train: Vec<usize>,
        decision_function_shape: SvmDecisionShape,
        break_ties: bool,
        probability: bool,
        prob_a: Vec<F>,
        prob_b: Vec<F>,
    ) -> Self {
        let binary_models = pairs
            .into_iter()
            .map(|pair| BinarySvm {
                support_vectors: pair.sv_data,
                sv_indices: pair.sv_indices,
                dual_coefs: pair.sv_coefs,
                bias: pair.bias_internal,
                class_neg: pair.class_neg,
                class_pos: pair.class_pos,
            })
            .collect();

        FittedSVC {
            kernel,
            binary_models,
            classes,
            x_train,
            y_train,
            decision_function_shape,
            break_ties,
            probability,
            prob_a,
            prob_b,
        }
    }
}

impl<F: Float, K: Kernel<F>> FittedSVC<F, K> {
    /// Compute the decision function value for a single sample against a
    /// binary model.
    fn decision_value_binary(&self, x: &[F], model: &BinarySvm<F>) -> F {
        let mut val = model.bias;
        for (sv, &coef) in model.support_vectors.iter().zip(model.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val
    }

    /// Raw one-vs-one decision values in **libsvm sign convention**, shape
    /// `(n_samples, n·(n-1)/2)`, columns in ovo pair order
    /// `(0,1),(0,2),...,(1,2),...` (the `(ci,cj)` double loop).
    ///
    /// libsvm/sklearn use the LOWER-index class as the `+1` side, so a POSITIVE
    /// value means the lower-index class wins (`sklearn/svm/_base.py:520-524`).
    /// This crate's [`Self::decision_value_binary`] uses the HIGHER-index class
    /// (`class_pos`) as `+1`, the opposite sign — so the raw ovo value is the
    /// negation of `decision_value_binary` to restore libsvm's convention.
    fn raw_ovo(&self, x: &Array2<F>) -> Array2<F> {
        let n_samples = x.nrows();
        let n_models = self.binary_models.len();
        let mut result = Array2::<F>::zeros((n_samples, n_models));

        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            for (m, model) in self.binary_models.iter().enumerate() {
                // Negate to match libsvm's "lower-index class = +1" sign.
                result[[s, m]] = self.decision_value_binary(&xi, model).neg();
            }
        }

        result
    }

    /// The continuous one-vs-rest decision function transformed from the
    /// one-vs-one scores, mirroring `_ovr_decision_function`
    /// (`sklearn/utils/multiclass.py:520-562`), shape `(n_samples, n_classes)`.
    ///
    /// `predictions[s,k] = if raw_ovo[s,k] < 0 { 1 } else { 0 }` and
    /// `confidences[s,k] = -raw_ovo[s,k]`, matching sklearn's call
    /// `_ovr_decision_function(dec < 0, -dec, n_classes)`
    /// (`sklearn/svm/_base.py:780`). The ovo pair iteration `(i,j)` with `i<j`,
    /// `k = 0,1,...`, is the SAME order as the `raw_ovo` columns.
    fn ovr_decision_function(&self, raw_ovo: &Array2<F>) -> Array2<F> {
        let n_samples = raw_ovo.nrows();
        let n_classes = self.classes.len();
        let mut votes = Array2::<F>::zeros((n_samples, n_classes));
        let mut sum_of_confidences = Array2::<F>::zeros((n_samples, n_classes));
        let one = F::one();

        let mut k = 0usize;
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                for s in 0..n_samples {
                    let dec = raw_ovo[[s, k]];
                    let confidence = dec.neg(); // -dec
                    // sum_of_confidences[:, i] -= confidences[:, k]
                    sum_of_confidences[[s, i]] = sum_of_confidences[[s, i]] - confidence;
                    // sum_of_confidences[:, j] += confidences[:, k]
                    sum_of_confidences[[s, j]] = sum_of_confidences[[s, j]] + confidence;
                    // predictions[s,k] = (dec < 0) ? 1 : 0
                    // votes[predictions==0, i] += 1; votes[predictions==1, j] += 1
                    if dec < F::zero() {
                        votes[[s, j]] = votes[[s, j]] + one;
                    } else {
                        votes[[s, i]] = votes[[s, i]] + one;
                    }
                }
                k += 1;
            }
        }

        // transformed = sum_of_confidences / (3 * (|sum_of_confidences| + 1))
        // return votes + transformed.
        let three = match F::from(3.0) {
            Some(v) => v,
            None => one + one + one,
        };
        let mut out = votes;
        for s in 0..n_samples {
            for c in 0..n_classes {
                let soc = sum_of_confidences[[s, c]];
                let transformed = soc / (three * (soc.abs() + one));
                out[[s, c]] = out[[s, c]] + transformed;
            }
        }
        out
    }

    /// The decision function for the samples in `x`
    /// (`sklearn/svm/_base.py:536-541, 778-781`).
    ///
    /// - **Binary** (`n_classes == 2`): [`SvmScores::Binary`], shape
    ///   `(n_samples,)` = `-raw_ovo.ravel()` (sklearn flips the sign for
    ///   `c_svc`/`nu_svc` binary, `_base.py:538-539`), so a POSITIVE value
    ///   predicts `classes_[1]`. Because this crate's `decision_value_binary`
    ///   already uses the higher-index class as `+1`, `-raw_ovo` equals
    ///   `decision_value_binary` directly.
    /// - **Multiclass [`SvmDecisionShape::Ovr`]** (default):
    ///   [`SvmScores::Multiclass`], shape `(n_samples, n_classes)` =
    ///   `_ovr_decision_function(raw_ovo)` (`_base.py:780`).
    /// - **Multiclass [`SvmDecisionShape::Ovo`]**: [`SvmScores::Multiclass`],
    ///   shape `(n_samples, n·(n-1)/2)` = the raw ovo values.
    ///
    /// # Errors
    ///
    /// Returns `Ok` for valid input.
    pub fn decision_function(&self, x: &Array2<F>) -> Result<SvmScores<F>, FerroError> {
        let raw_ovo = self.raw_ovo(x);

        if self.classes.len() == 2 {
            // Binary: -raw_ovo.ravel() = +decision_value_binary (1-D).
            let n_samples = raw_ovo.nrows();
            let mut binary = Array1::<F>::zeros(n_samples);
            for s in 0..n_samples {
                binary[s] = raw_ovo[[s, 0]].neg();
            }
            return Ok(SvmScores::Binary(binary));
        }

        match self.decision_function_shape {
            SvmDecisionShape::Ovo => Ok(SvmScores::Multiclass(raw_ovo)),
            SvmDecisionShape::Ovr => {
                Ok(SvmScores::Multiclass(self.ovr_decision_function(&raw_ovo)))
            }
        }
    }

    /// Class probability estimates, shape `(n_samples, n_classes)`; columns
    /// correspond to `classes_` in sorted order
    /// (`sklearn/svm/_base.py:829-864`, `libsvm.predict_probability`,
    /// `svm.cpp:2918-2955`).
    ///
    /// Built from the per-pair Platt sigmoids fitted at `fit` time when
    /// `probability=true`: the raw one-vs-one decision values (libsvm sign,
    /// lower-index class `+1`) are mapped to pairwise probabilities via
    /// [`sigmoid_predict`] (clamped to `[1e-7, 1-1e-7]`, `svm.cpp:2929-2938`),
    /// then coupled by [`multiclass_probability`] (Wu-Lin-Weng 2004,
    /// `svm.cpp:2941`). For the binary case `multiclass_probability` reduces to
    /// `[sigmoid_predict(dec), 1 - sigmoid_predict(dec)]` =
    /// `[P(classes_[0]), P(classes_[1])]`. Each row sums to 1.
    ///
    /// **RNG-CV exact-value boundary (documented divergence).** Because the
    /// underlying `(A, B)` come from a cross-validation whose fold assignment
    /// is RNG-dependent in libsvm/sklearn (sklearn's `probA_`/`probB_` and
    /// `predict_proba` values change with `random_state`), ferrolearn uses a
    /// DETERMINISTIC contiguous 5-fold split instead and therefore does NOT
    /// bit-match sklearn's `predict_proba` values. The reproduced contract is
    /// structural: rows sum to 1, entries in `[0, 1]`, and (binary) the
    /// `classes_[1]` column is monotone non-decreasing in the
    /// `decision_function` value. See [`SVC::probability`].
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] when the model was fitted with
    /// `probability=false`, with the message mirroring sklearn's
    /// `NotFittedError` text "predict_proba is not available when fitted with
    /// probability=False" (`_base.py:856-860`). (This crate has no `NotFitted`
    /// variant — predict-before-fit is a compile error via the typestate,
    /// R-DEV-4 — so the "fitted-without-probability" runtime condition maps to
    /// `InvalidParameter`; the binding maps it to the matching `PyErr`.)
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if !self.probability {
            return Err(FerroError::InvalidParameter {
                name: "probability".into(),
                reason: "predict_proba is not available when fitted with probability=False".into(),
            });
        }

        let raw_ovo = self.raw_ovo(x);
        let n_samples = raw_ovo.nrows();
        let n_classes = self.classes.len();
        let min_prob = F::from(1e-7).unwrap_or_else(F::epsilon);
        let max_prob = F::one() - min_prob;

        let mut out = Array2::<F>::zeros((n_samples, n_classes));

        for s in 0..n_samples {
            // Build the k×k pairwise probability matrix for this sample.
            let mut pairwise = Array2::<F>::zeros((n_classes, n_classes));
            let mut k = 0usize;
            for i in 0..n_classes {
                for j in (i + 1)..n_classes {
                    // dec_values[k] is the raw ovo value (libsvm sign: positive
                    // favors the lower-index class i = classes_[i]).
                    let dec = raw_ovo[[s, k]];
                    let (a, b) = (self.prob_a[k], self.prob_b[k]);
                    let mut pij = sigmoid_predict(dec, a, b);
                    // Clamp to [min_prob, 1-min_prob] (`svm.cpp:2937`).
                    if pij < min_prob {
                        pij = min_prob;
                    }
                    if pij > max_prob {
                        pij = max_prob;
                    }
                    pairwise[[i, j]] = pij;
                    pairwise[[j, i]] = F::one() - pij;
                    k += 1;
                }
            }
            let probs = multiclass_probability(n_classes, &pairwise);
            for (c, &pc) in probs.iter().enumerate() {
                out[[s, c]] = pc;
            }
        }

        Ok(out)
    }

    /// Natural-log class probability estimates, shape `(n_samples, n_classes)`
    /// = `predict_proba(x).ln()` elementwise (`sklearn/svm/_base.py:866-894`:
    /// `np.log(self.predict_proba(X))`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::NotFitted`] when the model was fitted with
    /// `probability=false` (delegated from [`Self::predict_proba`]).
    pub fn predict_log_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.predict_proba(x).map(|p| p.mapv(F::ln))
    }

    /// Whether Platt-scaling probability estimates were fitted
    /// (`probability`, `sklearn/svm/_classes.py`); when `false`,
    /// [`Self::predict_proba`]/[`Self::predict_log_proba`] raise.
    #[must_use]
    pub fn probability(&self) -> bool {
        self.probability
    }

    /// The per-ovo-pair Platt sigmoid `A` parameters (`probA_`,
    /// `sklearn/svm/_base.py`), length `n_class·(n_class-1)/2`. Empty when
    /// fitted with `probability=false`.
    #[must_use]
    pub fn prob_a(&self) -> Array1<F> {
        Array1::from_vec(self.prob_a.clone())
    }

    /// The per-ovo-pair Platt sigmoid `B` parameters (`probB_`,
    /// `sklearn/svm/_base.py`), length `n_class·(n_class-1)/2`. Empty when
    /// fitted with `probability=false`.
    #[must_use]
    pub fn prob_b(&self) -> Array1<F> {
        Array1::from_vec(self.prob_b.clone())
    }
}

impl<F: Float + ScalarOperand + 'static, K: Kernel<F>> FittedSVC<F, K> {
    /// Build the global, per-class-grouped support-vector index set, mirroring
    /// libsvm's `support_` layout (`sklearn/svm/_base.py:318-410`): the indices
    /// of the training rows that are a support vector in AT LEAST ONE ovo
    /// binary problem, deduplicated, grouped by class (all of class
    /// `classes_[0]` first, then `classes_[1]`, ...), ascending within a class.
    ///
    /// Returns `(support, per_class_indices)` where `support` is the flat
    /// grouped index vector and `per_class_indices[c]` is the (ascending)
    /// list of global training-row indices that are SVs for class
    /// `classes_[c]`.
    fn build_support(&self) -> (Vec<usize>, Vec<Vec<usize>>) {
        let n_classes = self.classes.len();
        // Per-class set of training-row indices that are an SV anywhere.
        let mut per_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        let mut seen: Vec<bool> = vec![false; self.y_train.len()];

        for model in &self.binary_models {
            for &idx in &model.sv_indices {
                if !seen[idx] {
                    seen[idx] = true;
                    let cls = self.y_train[idx];
                    if let Some(ci) = self.classes.iter().position(|&c| c == cls) {
                        per_class[ci].push(idx);
                    }
                }
            }
        }

        for group in &mut per_class {
            group.sort_unstable();
        }

        let support: Vec<usize> = per_class.iter().flatten().copied().collect();
        (support, per_class)
    }

    /// Indices of the support vectors into the training set, **grouped by
    /// class** (all class-`classes_[0]` SVs first, then `classes_[1]`, ...),
    /// ascending within each class.
    ///
    /// Mirrors `SVC.support_` (`sklearn/svm/_base.py:318-410`).
    #[must_use]
    pub fn support(&self) -> Array1<usize> {
        let (support, _) = self.build_support();
        Array1::from_vec(support)
    }

    /// The support vectors `X[support_]`, shape `(n_SV, n_features)`.
    ///
    /// Mirrors `SVC.support_vectors_` (`sklearn/svm/_base.py:318-410`).
    #[must_use]
    pub fn support_vectors(&self) -> Array2<F> {
        let (support, _) = self.build_support();
        let n_features = self.x_train.ncols();
        let mut out = Array2::<F>::zeros((support.len(), n_features));
        for (row, &idx) in support.iter().enumerate() {
            out.row_mut(row).assign(&self.x_train.row(idx));
        }
        out
    }

    /// Number of support vectors per class (`n_support_`,
    /// `sklearn/svm/_base.py:668-682`), parallel to `classes_`.
    #[must_use]
    pub fn n_support(&self) -> Vec<usize> {
        let (_, per_class) = self.build_support();
        per_class.iter().map(Vec::len).collect()
    }

    /// Dual coefficients in the libsvm public layout, shape
    /// `(n_class - 1, n_SV)` (`sklearn/svm/_base.py:318-410`, the `dual_coef_`
    /// attribute), columns in `support_` (per-class-grouped) order.
    ///
    /// For an SV belonging to class `i`, row `m` holds its coefficient in the
    /// binary classifier between class `i` and the `m`-th OTHER class (the
    /// other classes in increasing index order, skipping `i`). In the ovo pair
    /// `(a, b)` with `a < b`, libsvm uses class `a` as the `+1` side and `b` as
    /// `-1`; the stored coefficient is `alpha * y_libsvm`.
    ///
    /// This crate stores `alpha * y` per pair with the OPPOSITE sign
    /// (`class_neg = a` mapped to `-1`), so the libsvm-internal coefficient is
    /// the negation of the stored value. For `n_class == 2` sklearn negates the
    /// internal coefficient again to form the PUBLIC binary attribute
    /// (`sklearn/svm/_base.py:258-262`: `dual_coef_ = -dual_coef_`), which
    /// leaves the public binary value equal to this crate's stored value; for
    /// `n_class > 2` the public value IS the libsvm-internal value (no flip).
    #[must_use]
    pub fn dual_coef(&self) -> Array2<F> {
        let n_classes = self.classes.len();
        let (support, _per_class) = self.build_support();
        let n_sv = support.len();

        if n_classes == 2 {
            // Binary: public dual_coef_ = -internal, and internal = -stored,
            // so public = stored. The single ovo model holds one stored coef
            // per SV keyed by training index; map them into support_ column
            // order.
            let mut out = Array2::<F>::zeros((1, n_sv));
            if let Some(model) = self.binary_models.first() {
                for (sv_idx, &coef) in model.sv_indices.iter().zip(model.dual_coefs.iter()) {
                    if let Some(col) = support.iter().position(|&s| s == *sv_idx) {
                        out[[0, col]] = coef;
                    }
                }
            }
            return out;
        }

        // Multiclass: public dual_coef_ = libsvm-internal = -(stored). Row m
        // for an SV of class i is its coefficient in the pair (i, m-th other).
        let mut out = Array2::<F>::zeros((n_classes - 1, n_sv));

        // Column index in `support_` for a given training-row index.
        let col_of: HashMap<usize, usize> =
            support.iter().enumerate().map(|(c, &i)| (i, c)).collect();

        for model in &self.binary_models {
            let a = model.class_neg; // lower-index class in the pair
            let b = model.class_pos; // higher-index class
            let ai = self.classes.iter().position(|&c| c == a);
            let bi = self.classes.iter().position(|&c| c == b);
            let (ai, bi) = match (ai, bi) {
                (Some(ai), Some(bi)) => (ai, bi),
                _ => continue,
            };

            for (sv_idx, &stored) in model.sv_indices.iter().zip(model.dual_coefs.iter()) {
                let Some(&col) = col_of.get(sv_idx) else {
                    continue;
                };
                let cls = self.y_train[*sv_idx];
                let internal = stored.neg(); // libsvm internal = -(stored)
                // Determine which row this pair occupies for class `cls`:
                // the count of OTHER classes with index < the partner's index.
                let (own_class_index, partner_class_index) =
                    if cls == a { (ai, bi) } else { (bi, ai) };
                // Row m = number of other classes (excluding own) with class
                // index < partner_class_index.
                let mut row = 0usize;
                for ci in 0..n_classes {
                    if ci == own_class_index {
                        continue;
                    }
                    if ci < partner_class_index {
                        row += 1;
                    }
                }
                out[[row, col]] = internal;
            }
        }

        out
    }

    /// The per-ovo-problem intercepts, length `n_class * (n_class - 1) / 2`,
    /// in pair order `(0,1),(0,2),(1,2),...` (`intercept_`,
    /// `sklearn/svm/_base.py:318-410`). For `n_class == 2` sklearn negates the
    /// internal bias to form the public attribute
    /// (`sklearn/svm/_base.py:258-262`: `intercept_ *= -1`).
    ///
    /// This crate's per-pair `bias` is recovered for the decision function
    /// `sum coef*K + bias` with `class_pos` (the higher index) as the `+1`
    /// side. libsvm/sklearn use the lower-index class as `+1`, so the
    /// libsvm-internal intercept is the negation of this crate's `bias`. For
    /// binary, the public attribute negates the internal again, leaving the
    /// public value equal to this crate's stored `bias`; for multiclass the
    /// public value IS the internal `-bias`.
    #[must_use]
    pub fn intercept(&self) -> Array1<F> {
        let n_classes = self.classes.len();
        let vals: Vec<F> = if n_classes == 2 {
            self.binary_models.iter().map(|m| m.bias).collect()
        } else {
            self.binary_models.iter().map(|m| m.bias.neg()).collect()
        };
        Array1::from_vec(vals)
    }

    /// Primal weight vector `coef_ = dual_coef_ @ support_vectors_`, shape
    /// `(n_class - 1, n_features)` — available ONLY for the linear kernel
    /// (`sklearn/svm/_base.py:650-666`). Returns `None` for any other kernel
    /// (sklearn raises `AttributeError`).
    #[must_use]
    pub fn coef(&self) -> Option<Array2<F>> {
        if !self.kernel.is_linear() {
            return None;
        }
        let dual = self.dual_coef(); // (n_class-1, n_SV)
        let svs = self.support_vectors(); // (n_SV, n_features)
        Some(dual.dot(&svs))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<usize>> for SVC<F, K>
{
    type Fitted = FittedSVC<F, K>;
    type Error = FerroError;

    /// Fit the SVC model using SMO.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `C` is not positive.
    /// Returns [`FerroError::InsufficientSamples`] if fewer than 2 classes.
    fn fit(&self, x: &Array2<F>, y: &Array1<usize>) -> Result<FittedSVC<F, K>, FerroError> {
        let (n_samples, _n_features) = x.dim();

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

        // Reject non-finite input (NaN / +/-inf) in X BEFORE the SMO solve,
        // mirroring sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, ...)`
        // (`sklearn/svm/_base.py:190-197`) which keeps the default
        // `force_all_finite=True` and raises `ValueError("Input X contains NaN.")`
        // / `"... contains infinity ..."`. `y` is class labels (`Array1<usize>`),
        // already finite by type, so only `X` needs the float finiteness check.
        // `.iter().any(|v| !v.is_finite())` catches both NaN and +/-inf; on
        // finite input the guard never fires, so the fitted SVC attributes
        // (`support_`/`dual_coef_`/`intercept_`/`coef_`) are byte-identical.
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // Determine unique classes.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: classes.len(),
                context: "SVC requires at least 2 distinct classes".into(),
            });
        }

        // Resolve any data-dependent kernel parameters (e.g. a `None` gamma ->
        // sklearn's default `gamma='scale'` = 1/(n_features * X.var()),
        // `_base.py:236-239`) against the training data BEFORE fitting, and use
        // this resolved kernel for both fitting and prediction.
        let kernel = self.kernel.resolved_for_fit(x);

        // Convert data to Vec<Vec<F>> for kernel cache.
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        // Per-class weights computed ONCE over the FULL y (libsvm's
        // `class_weight_ = compute_class_weight(class_weight, classes, y)`,
        // `_base.py:740`); `weighted_C[c] = C·class_weight_[c]`.
        let y_vec: Vec<usize> = y.to_vec();
        let weights = compute_class_weight(&self.class_weight, &classes, &y_vec);

        // One-vs-one: train one binary SVM per pair.
        let n_classes = classes.len();
        let mut binary_models = Vec::new();
        // Per-pair Platt sigmoid params (only filled when `probability`).
        let mut prob_a: Vec<F> = Vec::new();
        let mut prob_b: Vec<F> = Vec::new();

        for ci in 0..n_classes {
            for cj in (ci + 1)..n_classes {
                let class_neg = classes[ci];
                let class_pos = classes[cj];

                // Per-class box bounds for this ovo pair `(class_neg, class_pos)`:
                // the `y=+1` group (class_pos = classes[cj]) gets `Cp = C·w[cj]`
                // and the `y=-1` group (class_neg = classes[ci]) gets
                // `Cn = C·w[ci]` (`weighted_C`, `_base.py:740`). The `weights`
                // vector is aligned to `classes`, so the class-index = the
                // position in `classes` (`ci`/`cj`).
                let cp = self.c * weights[cj];
                let cn = self.c * weights[ci];

                // Extract samples for these two classes.
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

                let result = smo_binary(
                    &sub_data,
                    &sub_labels,
                    &kernel,
                    cp,
                    cn,
                    self.tol,
                    self.max_iter,
                    self.cache_size,
                )?;

                // Extract support vectors (non-zero alphas).
                let eps = F::from(1e-8).unwrap_or_else(F::epsilon);
                let mut sv_data = Vec::new();
                let mut sv_coefs = Vec::new();
                let mut sv_idx = Vec::new();

                for (k, &alpha) in result.alphas.iter().enumerate() {
                    if alpha > eps {
                        sv_data.push(sub_data[k].clone());
                        sv_coefs.push(alpha * sub_labels[k]);
                        // Record the ORIGINAL training-row index of this SV
                        // (sub_indices maps the per-pair row k back to X).
                        sv_idx.push(sub_indices[k]);
                    }
                }

                // Platt-scaling CV for this ovo pair (only when probability).
                // The CV sub-models are C-SVC (the SAME svm_type as the outer
                // model, libsvm `svm.cpp:2147-2150`): the `train_fold` closure
                // wraps `smo_binary` + SV extraction, returning the fitted
                // sub-model in this crate's sign (`class_pos = +1`).
                if self.probability {
                    let (tol, max_iter, cache_size) = (self.tol, self.max_iter, self.cache_size);
                    let sub_eps = F::from(1e-8).unwrap_or_else(F::epsilon);
                    let (a, b) = platt_cv_sigmoid(
                        &sub_data,
                        &sub_labels,
                        &kernel,
                        |tr_data: &[Vec<F>], tr_labels: &[F]| {
                            let sub = smo_binary(
                                tr_data, tr_labels, &kernel, cp, cn, tol, max_iter, cache_size,
                            )
                            .ok()?;
                            let mut sv_d: Vec<Vec<F>> = Vec::new();
                            let mut sv_c: Vec<F> = Vec::new();
                            for (k, &alpha) in sub.alphas.iter().enumerate() {
                                if alpha > sub_eps {
                                    sv_d.push(tr_data[k].clone());
                                    sv_c.push(alpha * tr_labels[k]);
                                }
                            }
                            Some((sv_d, sv_c, sub.bias))
                        },
                    );
                    prob_a.push(a);
                    prob_b.push(b);
                }

                binary_models.push(BinarySvm {
                    support_vectors: sv_data,
                    sv_indices: sv_idx,
                    dual_coefs: sv_coefs,
                    bias: result.bias,
                    class_neg,
                    class_pos,
                });
            }
        }

        Ok(FittedSVC {
            kernel,
            binary_models,
            classes,
            x_train: x.clone(),
            y_train: y.to_vec(),
            decision_function_shape: self.decision_function_shape,
            break_ties: self.break_ties,
            probability: self.probability,
            prob_a,
            prob_b,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedSVC<F, K>
{
    type Output = Array1<usize>;
    type Error = FerroError;

    /// Predict class labels for the given feature matrix.
    ///
    /// Uses one-vs-one voting (each binary model casts a vote, the most-voted
    /// class wins, ties broken toward the lower class index), matching libsvm's
    /// `super().predict` (`sklearn/svm/_base.py:813-814`).
    ///
    /// When `break_ties == true` AND [`SvmDecisionShape::Ovr`] AND
    /// `n_classes > 2`, ties are instead broken by the one-vs-rest decision
    /// confidence: `predict = argmax(decision_function(X))`
    /// (`sklearn/svm/_base.py:806-811`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data. Returns
    /// [`FerroError::InvalidParameter`] when `break_ties == true` and the
    /// decision-function shape is [`SvmDecisionShape::Ovo`]
    /// (`sklearn/svm/_base.py:801-804`).
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        // sklearn raises when break_ties=True and decision_function_shape='ovo'
        // (`_base.py:801-804`), regardless of n_classes.
        if self.break_ties && self.decision_function_shape == SvmDecisionShape::Ovo {
            return Err(FerroError::InvalidParameter {
                name: "break_ties".into(),
                reason: "break_ties must be False when decision_function_shape is 'ovo'".into(),
            });
        }

        // break_ties=True, ovr, multiclass: predict = argmax(decision_function)
        // (`_base.py:806-811`). The ovr decision breaks vote ties by confidence.
        if self.break_ties && self.decision_function_shape == SvmDecisionShape::Ovr && n_classes > 2
        {
            let scores = self.decision_function(x)?;
            let mc = match scores {
                SvmScores::Multiclass(m) => m,
                // n_classes > 2 always yields the multiclass variant.
                SvmScores::Binary(_) => {
                    return Err(FerroError::InvalidParameter {
                        name: "break_ties".into(),
                        reason: "ovr decision function unavailable for break-tie predict".into(),
                    });
                }
            };
            let mut predictions = Array1::<usize>::zeros(n_samples);
            for s in 0..n_samples {
                // argmax over the row; ties keep the first (lowest) index via a
                // strictly-greater scan, matching numpy's argmax.
                let mut best_idx = 0usize;
                let mut best_val = mc[[s, 0]];
                for c in 1..n_classes {
                    if mc[[s, c]] > best_val {
                        best_val = mc[[s, c]];
                        best_idx = c;
                    }
                }
                predictions[s] = self.classes[best_idx];
            }
            return Ok(predictions);
        }

        let mut predictions = Array1::<usize>::zeros(n_samples);

        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            let mut votes = vec![0usize; n_classes];

            for model in &self.binary_models {
                let val = self.decision_value_binary(&xi, model);
                let winner = if val >= F::zero() {
                    model.class_pos
                } else {
                    model.class_neg
                };
                if let Some(idx) = self.classes.iter().position(|&c| c == winner) {
                    votes[idx] += 1;
                }
            }

            // libsvm/sklearn break ovo vote ties toward the LOWER class index
            // (`sklearn/svm/_base.py:813-814`: `super().predict` -> libsvm
            // `svm_predict` keeps the first argmax). `classes` is ascending
            // (`np.unique(y)`), so a strictly-greater scan keeps the
            // first/lowest-index maximum — unlike `max_by_key`, which returns
            // the LAST maximum (the higher index).
            let mut best_class_idx = 0usize;
            let mut best_votes = 0usize;
            for (i, &v) in votes.iter().enumerate() {
                if v > best_votes {
                    best_votes = v;
                    best_class_idx = i;
                }
            }

            predictions[s] = self.classes[best_class_idx];
        }

        Ok(predictions)
    }
}

// ---------------------------------------------------------------------------
// SVR (Support Vector Regressor)
// ---------------------------------------------------------------------------

/// Support Vector Regressor.
///
/// Uses SMO to solve the epsilon-SVR dual problem.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type.
#[derive(Debug, Clone)]
pub struct SVR<F, K> {
    /// The kernel function.
    pub kernel: K,
    /// Regularization parameter.
    pub c: F,
    /// Epsilon tube width (insensitive loss zone).
    pub epsilon: F,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations. `0` is the sklearn `max_iter=-1`
    /// sentinel meaning **no iteration limit** (the SMO runs to convergence).
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache (perf-only; default `200`).
    pub cache_size: usize,
    /// Whether to use libsvm's shrinking heuristic (`sklearn` `shrinking`,
    /// default `true`). Accepted for API parity; does NOT alter the converged
    /// optimum (ferrolearn's SMO has no shrinking heuristic — R-DEV-7).
    pub shrinking: bool,
}

impl<F: Float, K: Kernel<F>> SVR<F, K> {
    /// Create a new `SVR` with the given kernel and default hyperparameters
    /// matching sklearn (`sklearn/svm/_classes.py` `SVR.__init__`).
    ///
    /// Defaults: `C = 1.0`, `epsilon = 0.1`, `tol = 1e-3`, `max_iter = 0`
    /// (= sklearn `-1`, no iteration limit), `cache_size = 200`,
    /// `shrinking = true`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            epsilon: F::from(0.1).unwrap_or_else(F::epsilon),
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            max_iter: 0,
            cache_size: 200,
            shrinking: true,
        }
    }

    /// Set the `shrinking` flag (`sklearn` `shrinking`, default `true`).
    ///
    /// Accepted for API parity; does NOT alter the converged optimum
    /// (ferrolearn's SMO has no shrinking heuristic — R-DEV-7).
    #[must_use]
    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }

    /// Set the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: F) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon tube width.
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: F) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum number of SMO iterations.
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

/// Fitted Support Vector Regressor.
///
/// Stores the support vectors, dual coefficients, and bias.
#[derive(Debug, Clone)]
pub struct FittedSVR<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// Support vectors.
    support_vectors: Vec<Vec<F>>,
    /// Original training-row index of each support vector (parallel to
    /// `support_vectors`/`dual_coefs`), for the `support_` attribute.
    sv_indices: Vec<usize>,
    /// Dual coefficients (alpha_i* - alpha_i) for each support vector.
    dual_coefs: Vec<F>,
    /// Bias term.
    bias: F,
}

impl<F: Float, K: Kernel<F>> FittedSVR<F, K> {
    /// Compute the decision function value for a single sample.
    fn decision_value(&self, x: &[F]) -> F {
        let mut val = self.bias;
        for (sv, &coef) in self.support_vectors.iter().zip(self.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val
    }

    /// Compute the raw decision function values for each sample.
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
    /// (`SVR.support_`, `sklearn/svm/_base.py:318-410`). SVR has a single
    /// "class", so there is no per-class grouping; the SVs are kept in
    /// training-row order.
    #[must_use]
    pub fn support(&self) -> Array1<usize> {
        Array1::from_vec(self.sv_indices.clone())
    }

    /// The support vectors, shape `(n_SV, n_features)`
    /// (`SVR.support_vectors_`).
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

    /// Number of support vectors. For SVR `n_support_` has size 1
    /// (`sklearn/svm/_base.py:680-682`).
    #[must_use]
    pub fn n_support(&self) -> Vec<usize> {
        vec![self.support_vectors.len()]
    }

    /// Dual coefficients `alpha*_i - alpha_i`, shape `(1, n_SV)`
    /// (`SVR.dual_coef_`). No sign flip applies to SVR
    /// (`sklearn/svm/_base.py:260` restricts the flip to `c_svc`/`nu_svc`).
    #[must_use]
    pub fn dual_coef(&self) -> Array2<F> {
        let n_sv = self.dual_coefs.len();
        let mut out = Array2::<F>::zeros((1, n_sv));
        for (c, &v) in self.dual_coefs.iter().enumerate() {
            out[[0, c]] = v;
        }
        out
    }

    /// The intercept, length 1 (`SVR.intercept_`). The SVR decision function is
    /// `sum coef*K + bias`, matching libsvm's `f(x) = ... + rho`, so the public
    /// intercept equals this crate's stored `bias` (no sign flip).
    #[must_use]
    pub fn intercept(&self) -> Array1<F> {
        Array1::from_vec(vec![self.bias])
    }
}

/// Solve epsilon-SVR using SMO.
///
/// Reformulates the epsilon-SVR dual as a standard 2n-variable QP and
/// solves it with the Fan-Chen-Lin WSS approach, analogous to `smo_binary`.
///
/// The 2n variables are indexed 0..2n:
/// - Index `k` (k < n)  corresponds to alpha\*\_k  with label +1
/// - Index `k` (k >= n) corresponds to alpha\_{k-n} with label -1
///
/// The Q matrix is: `Q_{ij} = s_i * s_j * K(p_i, p_j)` where `s` is the
/// sign (+1 or -1) and `p` maps to the original sample index.
///
/// The linear term is: `q_k = epsilon - y_{p_k} * s_k`.
#[allow(clippy::too_many_arguments)]
fn smo_svr<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    targets: &[F],
    kernel: &K,
    c: F,
    epsilon: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Result<(Vec<F>, F), FerroError> {
    let n = data.len();
    let m = 2 * n; // Total number of dual variables.

    // Encoding: variable k in [0, m)
    //   k < n  => alpha*_k   (sign = +1, sample index = k)
    //   k >= n => alpha_{k-n} (sign = -1, sample index = k - n)
    //
    // The dual is: min 1/2 * beta^T Q beta + q^T beta
    //   s.t. 0 <= beta_k <= C, sum_k s_k * beta_k = 0
    // where beta_k = alpha*_k or alpha_{k-n},
    //       Q_{ij} = s_i * s_j * K(p_i, p_j),
    //       q_k    = epsilon - y_{p_k} * s_k.
    //
    // This has the same structure as the SVC dual.

    let sign = |k: usize| -> F { if k < n { F::one() } else { -F::one() } };
    let sample = |k: usize| -> usize { if k < n { k } else { k - n } };

    let mut beta = vec![F::zero(); m];
    let mut cache = KernelCache::new(cache_size);

    // Gradient: grad_k = (Q * beta)_k + q_k.  Initially beta=0 so grad_k = q_k.
    // q_k = epsilon - y_{p_k} * s_k
    let mut grad: Vec<F> = (0..m)
        .map(|k| epsilon - targets[sample(k)] * sign(k))
        .collect();

    let two = F::one() + F::one();
    let eps_num = F::from(1e-12).unwrap_or_else(F::epsilon);

    // `max_iter == 0` is the sklearn `max_iter=-1` ("no iteration limit")
    // sentinel — run until the KKT gap closes; non-zero caps the count.
    let mut iter = 0usize;
    loop {
        if max_iter != 0 && iter >= max_iter {
            break;
        }
        iter += 1;
        // WSS: same as SVC but with the extended variables.
        // I_up  = {k : (s_k=+1 and beta_k < C) or (s_k=-1 and beta_k > 0)}
        // I_low = {k : (s_k=+1 and beta_k > 0) or (s_k=-1 and beta_k < C)}
        // Select i = argmax_{k in I_up}  -s_k * grad_k
        // Select j = argmin_{k in I_low} -s_k * grad_k

        let mut i_up = None;
        let mut max_val = F::neg_infinity();
        let mut j_low = None;
        let mut min_val = F::infinity();

        for k in 0..m {
            let sk = sign(k);
            let val = -sk * grad[k];

            let in_up =
                (sk > F::zero() && beta[k] < c - eps_num) || (sk < F::zero() && beta[k] > eps_num);
            let in_low =
                (sk > F::zero() && beta[k] > eps_num) || (sk < F::zero() && beta[k] < c - eps_num);

            if in_up && val > max_val {
                max_val = val;
                i_up = Some(k);
            }
            if in_low && val < min_val {
                min_val = val;
                j_low = Some(k);
            }
        }

        if i_up.is_none() || j_low.is_none() || max_val - min_val < tol {
            break;
        }

        let i = i_up.unwrap();
        let j = j_low.unwrap();

        if i == j {
            break;
        }

        let si = sign(i);
        let sj = sign(j);
        let pi = sample(i);
        let pj = sample(j);

        // Q_{ii} = si*si*K(pi,pi) = K(pi,pi),  similarly for jj and ij
        let kii = cache.get_or_compute(pi, pi, kernel, data);
        let kjj = cache.get_or_compute(pj, pj, kernel, data);
        let kij = cache.get_or_compute(pi, pj, kernel, data);

        // eta = Q_{ii} + Q_{jj} - 2*Q_{ij} = K(pi,pi) + K(pj,pj) - 2*si*sj*K(pi,pj)
        let eta = kii + kjj - two * si * sj * kij;

        if eta <= eps_num {
            continue;
        }

        // Bounds for beta_j
        let old_bi = beta[i];
        let old_bj = beta[j];

        let (lo, hi) = if si == sj {
            let sum = old_bi + old_bj;
            ((sum - c).max(F::zero()), sum.min(c))
        } else {
            let diff = old_bj - old_bi;
            (diff.max(F::zero()), (c + diff).min(c))
        };

        if (hi - lo).abs() < eps_num {
            continue;
        }

        // Analytic update: beta_j += s_j * (E_i - E_j) / eta
        // where E_k = s_k * grad_k
        let mut new_bj = old_bj + sj * (si * grad[i] - sj * grad[j]) / eta;

        if new_bj > hi {
            new_bj = hi;
        }
        if new_bj < lo {
            new_bj = lo;
        }

        if (new_bj - old_bj).abs() < eps_num {
            continue;
        }

        let new_bi = old_bi + si * sj * (old_bj - new_bj);

        beta[i] = new_bi;
        beta[j] = new_bj;

        // Update gradient: grad_k += delta_bi * Q_{k,i} + delta_bj * Q_{k,j}
        // Q_{k,t} = s_k * s_t * K(p_k, p_t)
        let delta_bi = new_bi - old_bi;
        let delta_bj = new_bj - old_bj;

        for (k, grad_k) in grad.iter_mut().enumerate() {
            let sk = sign(k);
            let pk = sample(k);
            let kki = cache.get_or_compute(pk, pi, kernel, data);
            let kkj = cache.get_or_compute(pk, pj, kernel, data);
            *grad_k = *grad_k + delta_bi * sk * si * kki + delta_bj * sk * sj * kkj;
        }
    }

    // Recover alpha*_i = beta_i (i < n) and alpha_i = beta_{i+n} (i >= n).
    // Coefficient for prediction: coef_i = alpha*_i - alpha_i.
    let coefs: Vec<F> = (0..n).map(|i| beta[i] - beta[i + n]).collect();

    // Compute bias from KKT conditions on free support vectors.
    // For k where 0 < beta_k < C:
    //   grad_k = 0 at optimality => (Q*beta)_k + q_k = 0
    //   sum_t beta_t * s_k * s_t * K(p_k, p_t) + epsilon - y_{p_k} * s_k = 0
    //   s_k * sum_t (beta_t * s_t) * K(p_k, p_t) = y_{p_k} * s_k - epsilon
    //   sum_t coef_t_effective * K(p_k, p_t) = y_{p_k} - epsilon / s_k  (nah, let's use f directly)
    //
    // Prediction: f(x) = sum_i coef_i * K(x_i, x) + b
    // For free alpha*_i (0 < alpha*_i < C): y_i - f(x_i) = epsilon  => b = y_i - epsilon - sum coef_j * K(i,j)
    // For free alpha_i  (0 < alpha_i  < C): f(x_i) - y_i = epsilon  => b = y_i + epsilon - sum coef_j * K(i,j)

    let mut b_sum = F::zero();
    let mut b_count = 0usize;

    for i in 0..n {
        let mut kernel_sum = F::zero();
        let has_free = (beta[i] > eps_num && beta[i] < c - eps_num)
            || (beta[i + n] > eps_num && beta[i + n] < c - eps_num);

        if !has_free {
            continue;
        }

        for (j, &cj) in coefs.iter().enumerate() {
            if cj.abs() > eps_num {
                kernel_sum = kernel_sum + cj * cache.get_or_compute(i, j, kernel, data);
            }
        }

        if beta[i] > eps_num && beta[i] < c - eps_num {
            // alpha*_i is free: y_i - f(x_i) = epsilon => b = y_i - epsilon - kernel_sum
            b_sum = b_sum + targets[i] - epsilon - kernel_sum;
            b_count += 1;
        }
        if beta[i + n] > eps_num && beta[i + n] < c - eps_num {
            // alpha_i is free: f(x_i) - y_i = epsilon => b = y_i + epsilon - kernel_sum
            b_sum = b_sum + targets[i] + epsilon - kernel_sum;
            b_count += 1;
        }
    }

    let bias = if b_count > 0 {
        b_sum / F::from(b_count).unwrap()
    } else {
        F::zero()
    };

    Ok((coefs, bias))
}

// ---------------------------------------------------------------------------
// Solver_NU — the libsvm nu-parameterized solver (nu-SVC / nu-SVR)
// ---------------------------------------------------------------------------

/// Output of the generic [`solver_nu_core`] solve.
struct NuResult<F> {
    /// The dual variables `alpha` (length `l`, the solver-internal variables).
    alpha: Vec<F>,
    /// `rho = (r1 - r2) / 2` (the per-pair bias term, `Solver_NU::calculate_rho`
    /// returns this; `sklearn/svm/src/libsvm/svm.cpp:1417`).
    rho: F,
    /// `r = (r1 + r2) / 2` (`si->r`, the nu-SVC `/r` rescale factor /
    /// the nu-SVR `-epsilon`, `svm.cpp:1416`).
    r: F,
}

/// The generic libsvm `Solver_NU` core: solves
/// `min 0.5 αᵀQα + pᵀα  s.t.  0≤α_k≤C_k,  yᵀα=0,  eᵀα=const`
/// where `Q[k][t] = y_k·y_t·K(sample(k), sample(t))`, with the nu-specific
/// second-order working-set selection (separately over the `y=+1` / `y=-1`
/// groups, four running maxima `Gmaxp/Gmaxp2/Gmaxn/Gmaxn2`) and the
/// two-group `rho`/`r` recovery.
///
/// A faithful transcription of libsvm's `Solver_NU` + the shared
/// `Solver::Solve` update step (`sklearn/svm/src/libsvm/svm.cpp:1166-1418`
/// for the nu-specific `select_working_set`/`calculate_rho`, and `:663-940`
/// for the gradient init + analytic 2-variable update + objective). Like the
/// existing [`smo_binary`]/[`smo_svr`] solvers this is a NATURAL-ORDER,
/// no-shrinking, DETERMINISTIC variant: libsvm's shrinking heuristic
/// (`Solver_NU::do_shrinking`, `svm.cpp:1318`) is a performance optimization
/// that does NOT change the converged optimum (R-DEV-7), so it is omitted; the
/// `active_size` always equals `l`. The result (`alpha`, `rho`, `r`) is in
/// libsvm's convention so the caller can reconstruct `dual_coef_`/`intercept_`
/// that match the live `NuSVC`/`NuSVR` oracle after the public sign flip.
///
/// Arguments:
/// - `data`: the ORIGINAL per-sample feature rows (length `n`), keyed by the
///   [`KernelCache`] via `sample(i)` for the kernel evaluation.
/// - `m`: number of solver variables (`l` in libsvm; `= n` for nu-SVC,
///   `= 2n` for nu-SVR).
/// - `sample`: maps a solver variable index `k` to the original sample index
///   (into `data`) used for the kernel evaluation `K(sample(i), sample(j))`.
/// - `y`: the per-variable sign (`+1` / `-1`).
/// - `p`: the linear term (`p[k]`; `0` for nu-SVC, `∓prob.y` for nu-SVR).
/// - `c`: the per-variable upper bound `C_k`.
/// - `alpha`: the (greedily-initialized) starting dual variables, modified
///   in place into the solution.
#[allow(
    clippy::too_many_arguments,
    reason = "a faithful transcription of libsvm's Solver_NU::Solve threads the \
              problem (n, m, sample, y, p, c, alpha) + hyperparameters through \
              one call; splitting them would obscure the C oracle correspondence"
)]
#[allow(
    clippy::too_many_lines,
    reason = "a one-to-one transcription of libsvm's Solver_NU select_working_set \
              + the shared Solver analytic 2-variable update + calculate_rho \
              (svm.cpp:663-940, 1186-1418); kept inline to preserve the \
              line-by-line correspondence to the C oracle"
)]
fn solver_nu_core<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    m: usize,
    sample: &dyn Fn(usize) -> usize,
    y: &[F],
    p: &[F],
    c: &[F],
    mut alpha: Vec<F>,
    kernel: &K,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> NuResult<F> {
    let zero = F::zero();
    let two = F::one() + F::one();
    // libsvm's TAU (`svm.cpp:316`): a tiny positive floor for the quadratic
    // coefficient when the kernel is not strictly positive-definite.
    let tau = F::from(1e-12).unwrap_or_else(F::epsilon);
    let eps_bound = F::from(1e-12).unwrap_or_else(F::epsilon);

    let mut cache = KernelCache::new(cache_size);
    // `QD[i] = K(sample(i), sample(i))` (since `y_i^2 = 1`, `Q_ii = K(i,i)`).
    // The cache keys on the ORIGINAL-sample index `sample(i)` into `data`.
    let qd: Vec<F> = (0..m)
        .map(|i| {
            let si = sample(i);
            cache.get_or_compute(si, si, kernel, data)
        })
        .collect();

    // `q(i, j) = y_i·y_j·K(sample(i), sample(j))` (libsvm `SVC_Q::get_Q`,
    // `svm.cpp:1436-1446`). The cache is keyed by original-sample index.
    let q_entry = |i: usize, j: usize, cache: &mut KernelCache<F>| -> F {
        y[i] * y[j] * cache.get_or_compute(sample(i), sample(j), kernel, data)
    };

    // Bound predicates (`Solver::update_alpha_status`, `svm.cpp:588-598`).
    let is_upper = |k: usize, a: &[F]| -> bool { a[k] >= c[k] - eps_bound };
    let is_lower = |k: usize, a: &[F]| -> bool { a[k] <= eps_bound };

    // Gradient `G[k] = (Q·alpha)_k + p[k]` (`svm.cpp:693-715`). Since the
    // greedy init has some `alpha_k > 0`, accumulate the full product.
    let mut grad: Vec<F> = p.to_vec();
    #[allow(
        clippy::needless_range_loop,
        reason = "i indexes alpha while the inner loop forms Q[i][j]·alpha[i]"
    )]
    for i in 0..m {
        if alpha[i] > eps_bound {
            let ai = alpha[i];
            for (j, g) in grad.iter_mut().enumerate() {
                *g = *g + ai * q_entry(i, j, &mut cache);
            }
        }
    }

    let mut iter = 0usize;
    loop {
        if max_iter != 0 && iter >= max_iter {
            break;
        }
        iter += 1;

        // ---- Solver_NU::select_working_set (svm.cpp:1186-1296) ----
        let mut gmaxp = F::neg_infinity();
        let mut gmaxp2 = F::neg_infinity();
        let mut gmaxp_idx: isize = -1;
        let mut gmaxn = F::neg_infinity();
        let mut gmaxn2 = F::neg_infinity();
        let mut gmaxn_idx: isize = -1;

        for t in 0..m {
            if y[t] > zero {
                if !is_upper(t, &alpha) && -grad[t] >= gmaxp {
                    gmaxp = -grad[t];
                    gmaxp_idx = t as isize;
                }
            } else if !is_lower(t, &alpha) && grad[t] >= gmaxn {
                gmaxn = grad[t];
                gmaxn_idx = t as isize;
            }
        }

        let ip = gmaxp_idx;
        let in_ = gmaxn_idx;

        let mut gmin_idx: isize = -1;
        let mut obj_diff_min = F::infinity();

        for j in 0..m {
            if y[j] > zero {
                if !is_lower(j, &alpha) {
                    let grad_diff = gmaxp + grad[j];
                    if grad[j] >= gmaxp2 {
                        gmaxp2 = grad[j];
                    }
                    if grad_diff > zero && ip != -1 {
                        let ipi = ip as usize;
                        let quad_coef = qd[ipi] + qd[j] - two * q_entry(ipi, j, &mut cache);
                        let obj_diff = if quad_coef > zero {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / tau
                        };
                        if obj_diff <= obj_diff_min {
                            gmin_idx = j as isize;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else if !is_upper(j, &alpha) {
                let grad_diff = gmaxn - grad[j];
                if -grad[j] >= gmaxn2 {
                    gmaxn2 = -grad[j];
                }
                if grad_diff > zero && in_ != -1 {
                    let ini = in_ as usize;
                    let quad_coef = qd[ini] + qd[j] - two * q_entry(ini, j, &mut cache);
                    let obj_diff = if quad_coef > zero {
                        -(grad_diff * grad_diff) / quad_coef
                    } else {
                        -(grad_diff * grad_diff) / tau
                    };
                    if obj_diff <= obj_diff_min {
                        gmin_idx = j as isize;
                        obj_diff_min = obj_diff;
                    }
                }
            }
        }

        // Stopping criterion (`svm.cpp:1286`).
        if (gmaxp + gmaxp2).max(gmaxn + gmaxn2) < tol || gmin_idx == -1 {
            break;
        }

        let i = if y[gmin_idx as usize] > zero {
            gmaxp_idx
        } else {
            gmaxn_idx
        };
        if i == -1 {
            break;
        }
        let i = i as usize;
        let j = gmin_idx as usize;
        if i == j {
            break;
        }

        // ---- Solver::Solve analytic 2-variable update (svm.cpp:756-852) ----
        let q_ij = q_entry(i, j, &mut cache);
        let c_i = c[i];
        let c_j = c[j];
        let old_alpha_i = alpha[i];
        let old_alpha_j = alpha[j];

        if y[i] != y[j] {
            // `Q_i[j] = y_i·y_j·K = -K(i,j)` here, so `+2·Q_i[j]` in libsvm.
            let mut quad_coef = qd[i] + qd[j] + two * q_ij;
            if quad_coef <= zero {
                quad_coef = tau;
            }
            let delta = (-grad[i] - grad[j]) / quad_coef;
            let diff = alpha[i] - alpha[j];
            alpha[i] = alpha[i] + delta;
            alpha[j] = alpha[j] + delta;

            if diff > zero {
                if alpha[j] < zero {
                    alpha[j] = zero;
                    alpha[i] = diff;
                }
            } else if alpha[i] < zero {
                alpha[i] = zero;
                alpha[j] = -diff;
            }
            if diff > c_i - c_j {
                if alpha[i] > c_i {
                    alpha[i] = c_i;
                    alpha[j] = c_i - diff;
                }
            } else if alpha[j] > c_j {
                alpha[j] = c_j;
                alpha[i] = c_j + diff;
            }
        } else {
            let mut quad_coef = qd[i] + qd[j] - two * q_ij;
            if quad_coef <= zero {
                quad_coef = tau;
            }
            let delta = (grad[i] - grad[j]) / quad_coef;
            let sum = alpha[i] + alpha[j];
            alpha[i] = alpha[i] - delta;
            alpha[j] = alpha[j] + delta;

            if sum > c_i {
                if alpha[i] > c_i {
                    alpha[i] = c_i;
                    alpha[j] = sum - c_i;
                }
            } else if alpha[j] < zero {
                alpha[j] = zero;
                alpha[i] = sum;
            }
            if sum > c_j {
                if alpha[j] > c_j {
                    alpha[j] = c_j;
                    alpha[i] = sum - c_j;
                }
            } else if alpha[i] < zero {
                alpha[i] = zero;
                alpha[j] = sum;
            }
        }

        // ---- Update gradient (svm.cpp:856-862) ----
        let delta_alpha_i = alpha[i] - old_alpha_i;
        let delta_alpha_j = alpha[j] - old_alpha_j;
        #[allow(
            clippy::needless_range_loop,
            reason = "k indexes grad and is also the kernel column Q[k][i]/Q[k][j]"
        )]
        for k in 0..m {
            let q_ki = q_entry(k, i, &mut cache);
            let q_kj = q_entry(k, j, &mut cache);
            grad[k] = grad[k] + q_ki * delta_alpha_i + q_kj * delta_alpha_j;
        }
    }

    // ---- Solver_NU::calculate_rho (svm.cpp:1370-1418) ----
    let mut nr_free1 = 0usize;
    let mut nr_free2 = 0usize;
    let mut ub1 = F::infinity();
    let mut ub2 = F::infinity();
    let mut lb1 = F::neg_infinity();
    let mut lb2 = F::neg_infinity();
    let mut sum_free1 = zero;
    let mut sum_free2 = zero;

    for i in 0..m {
        let upper = is_upper(i, &alpha);
        let lower = is_lower(i, &alpha);
        if y[i] > zero {
            if upper {
                lb1 = lb1.max(grad[i]);
            } else if lower {
                ub1 = ub1.min(grad[i]);
            } else {
                nr_free1 += 1;
                sum_free1 = sum_free1 + grad[i];
            }
        } else if upper {
            lb2 = lb2.max(grad[i]);
        } else if lower {
            ub2 = ub2.min(grad[i]);
        } else {
            nr_free2 += 1;
            sum_free2 = sum_free2 + grad[i];
        }
    }

    let r1 = if nr_free1 > 0 {
        sum_free1 / F::from(nr_free1).unwrap_or_else(F::one)
    } else {
        (ub1 + lb1) / two
    };
    let r2 = if nr_free2 > 0 {
        sum_free2 / F::from(nr_free2).unwrap_or_else(F::one)
    } else {
        (ub2 + lb2) / two
    };

    NuResult {
        alpha,
        rho: (r1 - r2) / two,
        r: (r1 + r2) / two,
    }
}

/// The recovered nu-SVC binary sub-model: support vectors, their `dual_coef`
/// values (`alpha_i·y_i/r`, libsvm's `sv_coef`), original indices, and the
/// libsvm-internal bias `-rho/r` (so the decision function is
/// `f(x) = Σ sv_coef·K(sv, x) - rho/r`).
pub(crate) struct NuSvcModel<F> {
    pub sv_data: Vec<Vec<F>>,
    pub sv_coefs: Vec<F>,
    pub sv_indices: Vec<usize>,
    /// libsvm-internal bias term: `-rho/r`. Used directly as the `+1`-side
    /// decision bias for the LOWER-index class (libsvm convention).
    pub bias_internal: F,
}

/// Solve the libsvm **nu-SVC** dual for a single binary sub-problem
/// (`solve_nu_svc`, `sklearn/svm/src/libsvm/svm.cpp:1646-1708`):
/// `min 0.5 αᵀQα  s.t.  yᵀα=0, eᵀα=ν·l, 0≤α_i≤1`, `Q_ij=y_iy_jK`.
///
/// `data`/`labels` are the per-pair samples and signs (`+1` for the higher-index
/// `class_pos`, `-1` for the lower-index `class_neg`, matching this crate's
/// [`BinarySvm`] convention). The greedy `alpha` init (`svm.cpp:1667-1682`)
/// fills each class up to `min(C_i, nu·l/2 − running_sum)`. After
/// [`solver_nu_core`], libsvm rescales `alpha_i ← alpha_i·y_i/r` and
/// `rho ← rho/r` (`svm.cpp:1696-1702`); the support-vector coefficient is
/// `sv_coef = alpha·y/r` and the decision bias is `−rho/r`.
///
/// Returns `None` when `r ≈ 0` (degenerate — no usable rescale, e.g. a
/// pathological all-bound solution), letting the caller surface a clean error.
#[allow(
    clippy::too_many_arguments,
    reason = "the solver + per-pair samples/labels + hyperparameters thread \
              through one call mirroring libsvm's solve_nu_svc"
)]
pub(crate) fn solve_nu_svc<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    labels: &[F],
    kernel: &K,
    nu: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Option<NuSvcModel<F>> {
    let l = data.len();
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    // `C[i] = prob->W[i] = 1` (unit instance weights, `svm.cpp:1664`).
    let c = vec![one; l];

    // Greedy alpha init (`svm.cpp:1667-1682`): `nu_l = Σ nu·C[i] = nu·l`,
    // `sum_pos = sum_neg = nu_l/2`, fill each class greedily up to `C[i]`.
    let nu_l = nu * F::from(l).unwrap_or_else(F::zero);
    let mut sum_pos = nu_l / two;
    let mut sum_neg = nu_l / two;
    let mut alpha = vec![zero; l];
    for i in 0..l {
        if labels[i] > zero {
            alpha[i] = c[i].min(sum_pos);
            sum_pos = sum_pos - alpha[i];
        } else {
            alpha[i] = c[i].min(sum_neg);
            sum_neg = sum_neg - alpha[i];
        }
    }

    let p = vec![zero; l]; // nu-SVC linear term is 0 (`zeros`, svm.cpp:1684).
    let sample = |k: usize| k;
    let res = solver_nu_core(
        data, l, &sample, labels, &p, &c, alpha, kernel, tol, max_iter, cache_size,
    );

    let r = res.r;
    if r.abs() <= F::from(1e-12).unwrap_or_else(F::epsilon) {
        return None;
    }

    // libsvm: `alpha_i *= y_i / r`, `rho /= r` (`svm.cpp:1696-1702`).
    // `sv_coef = alpha·y/r`; decision bias = `-rho/r`.
    let eps_sv = F::from(1e-8).unwrap_or_else(F::epsilon);
    let mut sv_data = Vec::new();
    let mut sv_coefs = Vec::new();
    let mut sv_indices = Vec::new();
    #[allow(
        clippy::needless_range_loop,
        reason = "i indexes alpha, the sign labels[i], and the sample data[i] together"
    )]
    for i in 0..l {
        let coef = res.alpha[i] * labels[i] / r;
        if coef.abs() > eps_sv {
            sv_data.push(data[i].clone());
            sv_coefs.push(coef);
            sv_indices.push(i);
        }
    }

    Some(NuSvcModel {
        sv_data,
        sv_coefs,
        sv_indices,
        bias_internal: (res.rho / r).neg(),
    })
}

/// The recovered nu-SVR model: prediction coefficients `α*−α` per sample,
/// support indices, and the bias.
pub(crate) struct NuSvrModel<F> {
    pub sv_data: Vec<Vec<F>>,
    pub sv_coefs: Vec<F>,
    pub sv_indices: Vec<usize>,
    /// Prediction bias: `f(x) = Σ coef·K(sv, x) + bias`. libsvm's decision
    /// function is `Σ coef·K − rho`, so `bias = −rho`.
    pub bias: F,
}

/// Solve the libsvm **nu-SVR** dual (`solve_nu_svr`,
/// `sklearn/svm/src/libsvm/svm.cpp:1795-1839`): a `2l`-variable
/// `(α, α*)` dual with the learned-tube `nu` constraint, both `nu` AND `C`
/// used (`epsilon` is replaced by `nu`).
///
/// Variable layout (libsvm): `k < l` is `α*_k` with sign `+1` and linear term
/// `−y_k`; `k ≥ l` is `α_{k−l}` with sign `−1` and linear term `+y_{k−l}`;
/// `C_k = W·C` for all (`svm.cpp:1806-1824`). The greedy init fills both halves
/// to `min(sum, C)` where `sum = (Σ C·nu)/2` (`svm.cpp:1814-1817`). After
/// [`solver_nu_core`] the prediction coefficient is `coef_k = α*_k − α_k`
/// (`svm.cpp:1832-1833`) and the bias is `−rho` (libsvm `f = Σ coef·K − rho`).
#[allow(
    clippy::too_many_arguments,
    reason = "the solver + samples/targets + (nu, C) + hyperparameters thread \
              through one call mirroring libsvm's solve_nu_svr"
)]
pub(crate) fn solve_nu_svr<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    targets: &[F],
    kernel: &K,
    nu: F,
    c_param: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> NuSvrModel<F> {
    let l = data.len();
    let m = 2 * l;
    let zero = F::zero();
    let one = F::one();
    let two = one + one;

    // `C[i] = C[i+l] = W·C` (`svm.cpp:1809`); `sum = (Σ C·nu)/2`.
    let c = vec![c_param; m];
    let mut sum = c_param * nu * F::from(l).unwrap_or_else(F::zero) / two;

    let mut alpha = vec![zero; m];
    let mut y = vec![zero; m];
    let mut p = vec![zero; m];
    for i in 0..l {
        let a = sum.min(c[i]); // alpha2[i] = alpha2[i+l] = min(sum, C[i])
        alpha[i] = a;
        alpha[i + l] = a;
        sum = sum - a;
        p[i] = targets[i].neg(); // linear_term[i]   = -y_i
        y[i] = one;
        p[i + l] = targets[i]; // linear_term[i+l] = +y_i
        y[i + l] = -one;
    }

    let sample = |k: usize| if k < l { k } else { k - l };
    let res = solver_nu_core(
        data, m, &sample, &y, &p, &c, alpha, kernel, tol, max_iter, cache_size,
    );

    // coef_i = alpha2[i] - alpha2[i+l] = α*_i - α_i (`svm.cpp:1832-1833`).
    let eps_sv = F::from(1e-8).unwrap_or_else(F::epsilon);
    let mut sv_data = Vec::new();
    let mut sv_coefs = Vec::new();
    let mut sv_indices = Vec::new();
    #[allow(
        clippy::needless_range_loop,
        reason = "i pairs alpha2[i] with alpha2[i+l] and indexes the sample data[i]"
    )]
    for i in 0..l {
        let coef = res.alpha[i] - res.alpha[i + l];
        if coef.abs() > eps_sv {
            sv_data.push(data[i].clone());
            sv_coefs.push(coef);
            sv_indices.push(i);
        }
    }

    NuSvrModel {
        sv_data,
        sv_coefs,
        sv_indices,
        bias: res.rho.neg(),
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    Fit<Array2<F>, Array1<F>> for SVR<F, K>
{
    type Fitted = FittedSVR<F, K>;
    type Error = FerroError;

    /// Fit the SVR model using SMO.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::InvalidParameter`] if `C` is not positive.
    fn fit(&self, x: &Array2<F>, y: &Array1<F>) -> Result<FittedSVR<F, K>, FerroError> {
        let (n_samples, _n_features) = x.dim();

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

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SVR requires at least one sample".into(),
            });
        }

        // Reject non-finite input (NaN / +/-inf) in X or the float target y
        // BEFORE the SMO solve, mirroring sklearn's `BaseLibSVM.fit` ->
        // `_validate_data(X, y, ...)` (`sklearn/svm/_base.py:190-197`) which keeps
        // the default `force_all_finite=True` and raises
        // `ValueError("Input X contains NaN.")` / `"Input y contains NaN."` /
        // `"... contains infinity ..."`. SVR's `y` is float regression targets,
        // so both X and y are finiteness-checked. `.iter().any(|v|
        // !v.is_finite())` catches NaN and +/-inf; on finite input the guard
        // never fires, so the fitted SVR attributes are byte-identical.
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

        // Resolve any data-dependent kernel parameters (e.g. a `None` gamma ->
        // sklearn's default `gamma='scale'` = 1/(n_features * X.var()),
        // `_base.py:236-239`) against the training data BEFORE fitting, and use
        // this resolved kernel for both fitting and prediction.
        let kernel = self.kernel.resolved_for_fit(x);

        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();
        let targets: Vec<F> = y.to_vec();

        let (coefs, bias) = smo_svr(
            &data,
            &targets,
            &kernel,
            self.c,
            self.epsilon,
            self.tol,
            self.max_iter,
            self.cache_size,
        )?;

        // Extract support vectors (non-zero coefficients).
        let eps = F::from(1e-8).unwrap_or_else(F::epsilon);
        let mut sv_data = Vec::new();
        let mut sv_coefs = Vec::new();
        let mut sv_idx = Vec::new();

        for (i, &coef) in coefs.iter().enumerate() {
            if coef.abs() > eps {
                sv_data.push(data[i].clone());
                sv_coefs.push(coef);
                sv_idx.push(i);
            }
        }

        Ok(FittedSVR {
            kernel,
            support_vectors: sv_data,
            sv_indices: sv_idx,
            dual_coefs: sv_coefs,
            bias,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedSVR<F, K>
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values for the given feature matrix.
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
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_linear_kernel() {
        let k = LinearKernel;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert_relative_eq!(k.compute(&x, &y), 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rbf_kernel() {
        let k = RbfKernel::with_gamma(1.0);
        let x = vec![0.0, 0.0];
        let y = vec![0.0, 0.0];
        assert_relative_eq!(k.compute(&x, &y), 1.0, epsilon = 1e-10);

        // Different points should give value < 1
        let y2 = vec![1.0, 0.0];
        let val: f64 = k.compute(&x, &y2);
        assert!(val < 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_polynomial_kernel() {
        let k = PolynomialKernel {
            gamma: Gamma::Value(1.0),
            degree: 2,
            coef0: 1.0,
        };
        let x = vec![1.0f64, 0.0];
        let y = vec![1.0, 0.0];
        // (1.0 * 1.0 + 1.0)^2 = 4.0
        assert_relative_eq!(k.compute(&x, &y), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_kernel() {
        let k = SigmoidKernel {
            gamma: Gamma::Value(1.0),
            coef0: 0.0,
        };
        let x = vec![0.0f64];
        let y = vec![0.0];
        // tanh(0) = 0
        assert_relative_eq!(k.compute(&x, &y), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svc_linear_separable() {
        // Two well-separated clusters.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let model = SVC::<f64, LinearKernel>::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(correct >= 6, "Expected at least 6 correct, got {correct}");
    }

    #[test]
    fn test_svc_rbf_xor() {
        // XOR problem: not linearly separable, needs RBF kernel.
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, // class 0
                1.0, 1.0, 1.1, 1.1, // class 0
                1.0, 0.0, 1.1, 0.1, // class 1
                0.0, 1.0, 0.1, 1.1, // class 1
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];

        let kernel = RbfKernel::with_gamma(5.0);
        let model = SVC::new(kernel).with_c(100.0).with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 6,
            "Expected at least 6 correct for XOR, got {correct}"
        );
    }

    #[test]
    fn test_svc_multiclass() {
        // Three linearly separable clusters.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, // class 0
                5.0, 0.0, 5.5, 0.0, 5.0, 0.5, // class 1
                0.0, 5.0, 0.5, 5.0, 0.0, 5.5, // class 2
            ],
        )
        .unwrap();
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = SVC::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        let correct: usize = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count();
        assert!(
            correct >= 7,
            "Expected at least 7 correct for multiclass, got {correct}"
        );
    }

    #[test]
    fn test_svc_decision_function() -> TestResult {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.5, 1.0, 1.0, 1.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, // class 1
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1];

        let model = SVC::new(LinearKernel).with_c(10.0);
        let fitted = model.fit(&x, &y)?;

        let df = fitted.decision_function(&x)?;
        // Binary: 1-D (n,) score (sklearn ravels the single ovo column,
        // `_base.py:538-539`); the multiclass borrow is None.
        assert!(df.as_multiclass().is_none());
        let bin = df.as_binary().ok_or_else(|| err("binary 1-D"))?;
        assert_eq!(bin.len(), 6);

        // Class 0 samples should have negative decision values,
        // class 1 should have positive (positive -> classes_[1]).
        for (i, &v) in bin.iter().enumerate().take(3) {
            assert!(v < 0.5, "Class 0 sample {i} has decision value {v}");
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // REQ-4 smoke tests: decision_function shape + sign + ovr/ovo transform.
    //
    // Expected values from the LIVE sklearn 1.5.2 oracle (R-CHAR-3):
    //
    //   import numpy as np; from sklearn.svm import SVC
    //   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]])
    //   y=np.array([0,0,0,1,1,1]); m=SVC(kernel='linear',C=1.0).fit(X,y)
    //   m.decision_function(X)  # (6,) [-1.2853,-0.9997,-0.9997,0.9995,1.2851,1.2851]
    //
    //   X3=[[0,0],[.5,0],[0,.5],[5,0],[5.5,0],[5,.5],[0,5],[.5,5],[0,5.5]]
    //   y3=[0,0,0,1,1,1,2,2,2]; m3=SVC(kernel='linear',C=1.0).fit(X3,y3)
    //   m3.decision_function(X3)            # ovr (9,3) row0 [2.2366,0.8167,-0.1833]
    //                                       #         row3 [1.0606,2.2262,-0.2333]
    //   SVC(...,decision_function_shape='ovo').fit(X3,y3).decision_function(X3)
    //                                       # ovo (9,3) row0 [1.2222,1.2222,0.0]
    // -----------------------------------------------------------------------

    fn three_class_9x2() -> Result<(Array2<f64>, Array1<usize>), FerroError> {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        Ok((x, y))
    }

    #[test]
    fn test_svc_decision_function_binary_values() -> TestResult {
        let m = binary_fit()?;
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let df = m.decision_function(&x)?;
        let bin = df.as_binary().ok_or_else(|| err("binary"))?;
        assert_eq!(bin.len(), 6);
        let oracle = [-1.2853, -0.9997, -0.9997, 0.9995, 1.2851, 1.2851];
        for (i, &exp) in oracle.iter().enumerate() {
            assert!(
                (bin[i] - exp).abs() < 1e-2,
                "binary df[{i}] = {} vs oracle {exp}",
                bin[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_decision_function_ovr() -> TestResult {
        let (x, y) = three_class_9x2()?;
        let m = SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .fit(&x, &y)?;
        let df = m.decision_function(&x)?;
        let mc = df.as_multiclass().ok_or_else(|| err("multiclass"))?;
        assert_eq!(mc.dim(), (9, 3));
        // ovr (default): row0 [2.2366,0.8167,-0.1833], row3 [1.0606,2.2262,-0.2333].
        let row0 = [2.2366, 0.8167, -0.1833];
        let row3 = [1.0606, 2.2262, -0.2333];
        for (c, &v) in row0.iter().enumerate() {
            assert!(
                (mc[[0, c]] - v).abs() < 1e-2,
                "ovr row0[{c}] = {} vs oracle {v}",
                mc[[0, c]]
            );
        }
        for (c, &v) in row3.iter().enumerate() {
            assert!(
                (mc[[3, c]] - v).abs() < 1e-2,
                "ovr row3[{c}] = {} vs oracle {v}",
                mc[[3, c]]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_decision_function_ovo() -> TestResult {
        let (x, y) = three_class_9x2()?;
        let m = SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .with_decision_function_shape(SvmDecisionShape::Ovo)
            .fit(&x, &y)?;
        let df = m.decision_function(&x)?;
        let mc = df.as_multiclass().ok_or_else(|| err("multiclass"))?;
        assert_eq!(mc.dim(), (9, 3));
        // ovo: row0 [1.2222,1.2222,0.0].
        let row0 = [1.2222, 1.2222, 0.0];
        for (c, &v) in row0.iter().enumerate() {
            assert!(
                (mc[[0, c]] - v).abs() < 1e-2,
                "ovo row0[{c}] = {} vs oracle {v}",
                mc[[0, c]]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0usize, 0, 1, 1];

        let model = SVC::new(LinearKernel).with_c(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_single_class_error() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0usize, 0, 0];

        let model = SVC::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svc_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![0usize, 1];

        let model = SVC::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svr_simple() {
        // Simple linear regression: y = 2x
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let model = SVR::new(LinearKernel)
            .with_c(100.0)
            .with_epsilon(0.1)
            .with_max_iter(50000);
        let fitted = model.fit(&x, &y).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Check predictions are reasonably close.
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert!(
                (*p - actual).abs() < 2.0,
                "SVR prediction {p} too far from actual {actual}"
            );
        }
    }

    #[test]
    fn test_svr_decision_function() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SVR::new(LinearKernel).with_c(100.0).with_epsilon(0.1);
        let fitted = model.fit(&x, &y).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Decision function and predict should return the same values.
        for i in 0..4 {
            assert_relative_eq!(df[i], preds[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_svr_invalid_c() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let model = SVR::new(LinearKernel).with_c(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_svr_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let model = SVR::new(LinearKernel);
        assert!(model.fit(&x, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // REQ-3 smoke tests: libsvm-layout fitted attributes (binary sign flip).
    //
    // Expected values from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), never
    // copied from the ferrolearn side:
    //
    //   python3 -c "import numpy as np; from sklearn.svm import SVC, SVR
    //   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]])
    //   y=np.array([0,0,0,1,1,1]); m=SVC(kernel='linear',C=1.0).fit(X,y)
    //   print(m.support_.tolist(), m.n_support_.tolist(),
    //         m.dual_coef_.tolist(), m.intercept_.tolist(), m.coef_.tolist())"
    //   # [1, 2, 3] [2, 1] [[-0.0408,-0.0408,0.0816]] [-1.8565] [[0.2856,0.2856]]
    //
    //   X3=[[0,0],[.5,0],[0,.5],[5,0],[5.5,0],[5,.5],[0,5],[.5,5],[0,5.5]]
    //   y3=[0,0,0,1,1,1,2,2,2]; m3=SVC(kernel='linear',C=1.0).fit(X3,y3)
    //   # support_ [1,2,3,5,6,7] n_support_ [2,2,2]
    //   # dual_coef_ [[0.0988,0,-0.0988,0,-0.0988,0],[0,0.0988,0,0.0494,0,-0.0494]]
    //   # intercept_ [1.2222,1.2222,0.0]
    //
    //   Xr=[[1],[2],[3],[4],[5],[6]]; yr=[2,4,6,8,10,12]
    //   mr=SVR(kernel='linear',C=100,epsilon=0.1).fit(Xr,yr)
    //   # support_ [0,5] dual_coef_ [[-0.392,0.392]] intercept_ [0.14] n_support_ [2]
    //
    // The tests return `Result` and use `?`/`ok_or` (no unwrap/expect/panic).
    // -----------------------------------------------------------------------

    type TestResult = Result<(), FerroError>;

    fn err(msg: &str) -> FerroError {
        FerroError::InvalidParameter {
            name: "test".into(),
            reason: msg.into(),
        }
    }

    fn binary_fit() -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1];
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .fit(&x, &y)
    }

    #[test]
    fn test_svc_binary_support_attrs() -> TestResult {
        let m = binary_fit()?;
        // support_ [1,2,3], grouped by class (class0:[1,2], class1:[3]).
        assert_eq!(m.support().to_vec(), vec![1, 2, 3]);
        // n_support_ [2,1].
        assert_eq!(m.n_support(), vec![2, 1]);
        // support_vectors_ = X[support_].
        let svs = m.support_vectors();
        assert_eq!(svs.dim(), (3, 2));
        let expected = [[2.0, 1.0], [1.0, 2.0], [5.0, 5.0]];
        for (r, row) in expected.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert_relative_eq!(svs[[r, c]], v, epsilon = 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn test_svc_binary_dual_coef_sign_flip() -> TestResult {
        let m = binary_fit()?;
        // dual_coef_ shape (1,3) = [[-0.0408,-0.0408,0.0816]] (binary sign flip).
        let dc = m.dual_coef();
        assert_eq!(dc.dim(), (1, 3));
        let oracle = [-0.0408, -0.0408, 0.0816];
        for (c, &v) in oracle.iter().enumerate() {
            assert!(
                (dc[[0, c]] - v).abs() < 1e-2,
                "dual_coef_[0,{c}] = {} vs oracle {v}",
                dc[[0, c]]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_binary_intercept_and_coef() -> TestResult {
        let m = binary_fit()?;
        // intercept_ [-1.8565], length 1 (binary sign flip).
        let ic = m.intercept();
        assert_eq!(ic.len(), 1);
        assert!(
            (ic[0] - (-1.8565)).abs() < 1e-2,
            "intercept_ = {} vs oracle -1.8565",
            ic[0]
        );
        // coef_ [[0.2856,0.2856]] shape (1,2) for the linear kernel.
        let coef = m.coef().ok_or_else(|| err("linear kernel exposes coef_"))?;
        assert_eq!(coef.dim(), (1, 2));
        for c in 0..2 {
            assert!(
                (coef[[0, c]] - 0.2856).abs() < 1e-2,
                "coef_[0,{c}] = {} vs oracle 0.2856",
                coef[[0, c]]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_coef_none_for_nonlinear() -> TestResult {
        // coef_ is only available for the linear kernel; RBF -> None
        // (sklearn raises AttributeError, _base.py:650-651).
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1];
        let m = SVC::new(RbfKernel::with_gamma(0.5)).fit(&x, &y)?;
        assert!(m.coef().is_none());
        Ok(())
    }

    fn multiclass_fit() -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .fit(&x, &y)
    }

    #[test]
    fn test_svc_multiclass_support_attrs() -> TestResult {
        let m = multiclass_fit()?;
        // support_ [1,2,3,5,6,7] grouped by class; n_support_ [2,2,2].
        assert_eq!(m.support().to_vec(), vec![1, 2, 3, 5, 6, 7]);
        assert_eq!(m.n_support(), vec![2, 2, 2]);
        // intercept_ [1.2222,1.2222,0.0] (no sign flip for multiclass).
        let ic = m.intercept();
        assert_eq!(ic.len(), 3);
        let oracle_ic = [1.2222, 1.2222, 0.0];
        for (i, &v) in oracle_ic.iter().enumerate() {
            assert!(
                (ic[i] - v).abs() < 1e-2,
                "intercept_[{i}] = {} vs oracle {v}",
                ic[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_multiclass_dual_coef_packing() -> TestResult {
        let m = multiclass_fit()?;
        // dual_coef_ shape (2,6), libsvm packing (cols = SVs [1,2,3,5,6,7]):
        //   row0 = [0.0988, 0.0, -0.0988, 0.0, -0.0988, 0.0]
        //   row1 = [0.0, 0.0988, 0.0, 0.0494, 0.0, -0.0494]
        let dc = m.dual_coef();
        assert_eq!(dc.dim(), (2, 6));
        let oracle = [
            [0.0988, 0.0, -0.0988, 0.0, -0.0988, 0.0],
            [0.0, 0.0988, 0.0, 0.0494, 0.0, -0.0494],
        ];
        for (r, row) in oracle.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!(
                    (dc[[r, c]] - v).abs() < 1e-2,
                    "dual_coef_[{r},{c}] = {} vs oracle {v}",
                    dc[[r, c]]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_svr_linear_attrs() -> TestResult {
        // SVR(kernel='linear', C=100, epsilon=0.1) on the 6x1 set.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .map_err(|_| err("shape"))?;
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let m = SVR::new(LinearKernel)
            .with_c(100.0)
            .with_epsilon(0.1)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .fit(&x, &y)?;
        // support_ [0,5]; n_support_ [2]; dual_coef_ (1,2) [[-0.392,0.392]];
        // intercept_ [0.14].
        assert_eq!(m.support().to_vec(), vec![0, 5]);
        assert_eq!(m.n_support(), vec![2]);
        let dc = m.dual_coef();
        assert_eq!(dc.dim(), (1, 2));
        assert!(
            (dc[[0, 0]] - (-0.392)).abs() < 1e-2,
            "dual_coef_[0,0] = {} vs oracle -0.392",
            dc[[0, 0]]
        );
        assert!(
            (dc[[0, 1]] - 0.392).abs() < 1e-2,
            "dual_coef_[0,1] = {} vs oracle 0.392",
            dc[[0, 1]]
        );
        let ic = m.intercept();
        assert_eq!(ic.len(), 1);
        assert!(
            (ic[0] - 0.14).abs() < 1e-2,
            "intercept_ = {} vs oracle 0.14",
            ic[0]
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // REQ-1 (gamma='auto') + REQ-8 (break_ties) smoke tests.
    //
    // Expected values from the LIVE sklearn 1.5.2 oracle (R-CHAR-3):
    //
    //   import numpy as np; from sklearn.svm import SVC
    //   X=np.array([[1.,1.],[2.,1.],[1.,2.],[5.,5.],[6.,5.],[5.,6.]])
    //   y=np.array([0,0,0,1,1,1])
    //   m=SVC(kernel='rbf',C=1.0,gamma='auto').fit(X,y)
    //   m._gamma                       # 0.5  (= 1/n_features = 1/2)
    //   m.decision_function(X)         # [-0.9996,-0.9999,-0.9999,
    //                                  #   0.9999, 0.9999, 0.9996]
    //
    //   break_ties: a symmetric, cleanly-separable 3-class set so ferrolearn's
    //   SMO converges to libsvm's optimum (each class has 2 SVs). Near the
    //   centroid the three ovr scores are ~1.0 (a near 1-1-1 vote tie), so the
    //   libsvm vote breaks toward the LOWEST class index (0) while ovr-argmax
    //   breaks by confidence. Oracle (re-derived vs the live oracle):
    //   Xb=[[0,0],[.5,0],[0,.5],[10,0],[10.5,0],[10,.5],[5,9],[5.5,9],[5,9.5]]
    //   yb=[0,0,0,1,1,1,2,2,2]
    //   q1=[[5.19,3.342]]: vote -> 0, break_ties -> 2 (df [0.9942,0.999,1.0068])
    //   q2=[[5.19,3.241]]: vote -> 0, break_ties -> 1 (df [0.9999,1.0051,0.9949])
    //   SVC(...,decision_function_shape='ovo',break_ties=True) -> ValueError
    // -----------------------------------------------------------------------

    #[test]
    fn test_svc_gamma_auto_decision_function() -> TestResult {
        // gamma='auto' on the 6x2 set: _gamma = 1/n_features = 0.5.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1];

        // Confirm the resolved gamma is 1/n_features (sklearn `_base.py:241`).
        let resolved = RbfKernel::<f64>::with_gamma_auto().resolved_for_fit(&x);
        assert!(
            (gamma_value_or_one(resolved.gamma) - 0.5).abs() < 1e-12,
            "gamma='auto' resolved to {} vs oracle 0.5",
            gamma_value_or_one(resolved.gamma)
        );

        let m = SVC::new(RbfKernel::<f64>::with_gamma_auto())
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .fit(&x, &y)?;
        let df = m.decision_function(&x)?;
        let bin = df.as_binary().ok_or_else(|| err("binary"))?;
        assert_eq!(bin.len(), 6);
        let oracle = [-0.9996, -0.9999, -0.9999, 0.9999, 0.9999, 0.9996];
        for (i, &exp) in oracle.iter().enumerate() {
            assert!(
                (bin[i] - exp).abs() < 1e-2,
                "gamma=auto df[{i}] = {} vs oracle {exp}",
                bin[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_svc_gamma_scale_still_default() -> TestResult {
        // gamma='scale' (the default) must STILL resolve to
        // 1/(n_features * X.var()); on the 6x2 set X.var()=4.2222 ->
        // _gamma = 1/(2*4.2222) = 0.11842 (sklearn `_base.py:238-239`).
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let resolved = RbfKernel::<f64>::new().resolved_for_fit(&x);
        assert!(
            (gamma_value_or_one(resolved.gamma) - 0.118_421).abs() < 1e-4,
            "gamma='scale' resolved to {} vs oracle 0.118421",
            gamma_value_or_one(resolved.gamma)
        );
        Ok(())
    }

    fn break_ties_set() -> Result<(Array2<f64>, Array1<usize>), FerroError> {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 5.0, 9.0, 5.5, 9.0,
                5.0, 9.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        Ok((x, y))
    }

    fn break_ties_fit(
        break_ties: bool,
        shape: SvmDecisionShape,
    ) -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let (x, y) = break_ties_set()?;
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .with_break_ties(break_ties)
            .with_decision_function_shape(shape)
            .fit(&x, &y)
    }

    #[test]
    fn test_svc_break_ties_changes_label() -> TestResult {
        // q1=(5.19,3.342): vote -> 0, break_ties -> 2.
        let q1 = Array2::from_shape_vec((1, 2), vec![5.19, 3.342]).map_err(|_| err("shape"))?;
        // q2=(5.19,3.241): vote -> 0, break_ties -> 1.
        let q2 = Array2::from_shape_vec((1, 2), vec![5.19, 3.241]).map_err(|_| err("shape"))?;

        let vote = break_ties_fit(false, SvmDecisionShape::Ovr)?;
        let bt = break_ties_fit(true, SvmDecisionShape::Ovr)?;

        // break_ties=false (default): libsvm vote -> lowest-index class 0.
        assert_eq!(vote.predict(&q1)?[0], 0, "vote q1 should be 0");
        assert_eq!(vote.predict(&q2)?[0], 0, "vote q2 should be 0");

        // break_ties=true + ovr: argmax(decision_function).
        assert_eq!(
            bt.predict(&q1)?[0],
            2,
            "break_ties q1 (ovr-argmax) should be 2"
        );
        assert_eq!(
            bt.predict(&q2)?[0],
            1,
            "break_ties q2 (ovr-argmax) should be 1"
        );
        Ok(())
    }

    #[test]
    fn test_svc_break_ties_ovo_errors() -> TestResult {
        // sklearn raises when break_ties=True and decision_function_shape='ovo'
        // (`_base.py:801-804`).
        let m = break_ties_fit(true, SvmDecisionShape::Ovo)?;
        let q = Array2::from_shape_vec((1, 2), vec![5.19, 3.342]).map_err(|_| err("shape"))?;
        assert!(m.predict(&q).is_err());
        Ok(())
    }

    #[test]
    fn test_svc_default_params() {
        // sklearn defaults: cache_size=200, max_iter=-1 (= 0 sentinel),
        // shrinking=True, break_ties=False, decision_function_shape='ovr'.
        let m = SVC::<f64, LinearKernel>::new(LinearKernel);
        assert_eq!(m.cache_size, 200);
        assert_eq!(m.max_iter, 0);
        assert!(m.shrinking);
        assert!(!m.break_ties);
        assert_eq!(m.decision_function_shape, SvmDecisionShape::Ovr);
        let r = SVR::<f64, LinearKernel>::new(LinearKernel);
        assert_eq!(r.cache_size, 200);
        assert_eq!(r.max_iter, 0);
        assert!(r.shrinking);
    }

    /// The overlapping imbalanced binary set used to pin `class_weight`.
    fn class_weight_xy() -> Result<(Array2<f64>, Array1<usize>), FerroError> {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.5, 0.5, 2.0, 2.0, 2.5, 2.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1];
        Ok((x, y))
    }

    fn cw_fit(cw: ClassWeight<f64>) -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let (x, y) = class_weight_xy()?;
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-7)
            .with_max_iter(500_000)
            .with_class_weight(cw)
            .fit(&x, &y)
    }

    /// `class_weight` per-class C in the C-SVC SMO (REQ-8, #641).
    ///
    /// Oracle (live `SVC(kernel='linear', C=1.0, class_weight=...)` on the
    /// overlapping imbalanced binary set, R-CHAR-3):
    /// ```text
    /// X=[[0,0],[1,0],[0,1],[1,1],[0.5,0.5],[1.5,0.5],[2,2],[2.5,2.5]] y=[0,0,0,0,0,1,1,1]
    /// None     -> dual_coef_ [[-0.5,-1.0,1.0,0.5]]      intercept_ [-2.0]    support_ [1,3,5,6]
    /// balanced -> dual_coef_ [[-0.8,-0.8,1.3333,0.2667]] intercept_ [-1.6667] support_ [1,3,5,6]
    /// {0:1,1:5}-> support_ [1,3,4,5] intercept_ [-2.0]
    /// ```
    #[test]
    fn test_svc_class_weight_smoke() -> TestResult {
        // class_weight=None.
        let none = cw_fit(ClassWeight::None)?;
        assert_eq!(none.support().to_vec(), vec![1, 3, 5, 6]);
        let dc = none.dual_coef();
        for (c, &v) in [-0.5, -1.0, 1.0, 0.5].iter().enumerate() {
            assert!(
                (dc[[0, c]] - v).abs() < 1e-2,
                "None dual_coef_[0,{c}] = {} vs {v}",
                dc[[0, c]]
            );
        }
        let none_int = none.intercept()[0];
        assert!(
            (none_int - (-2.0)).abs() < 1e-2,
            "None intercept_ {none_int}"
        );

        // class_weight='balanced' (weights [0.8, 1.3333]).
        let bal = cw_fit(ClassWeight::Balanced)?;
        assert_eq!(bal.support().to_vec(), vec![1, 3, 5, 6]);
        let dcb = bal.dual_coef();
        for (c, &v) in [-0.8, -0.8, 1.3333, 0.2667].iter().enumerate() {
            assert!(
                (dcb[[0, c]] - v).abs() < 1e-2,
                "balanced dual_coef_[0,{c}] = {} vs {v}",
                dcb[[0, c]]
            );
        }
        let bal_int = bal.intercept()[0];
        assert!(
            (bal_int - (-1.6667)).abs() < 1e-2,
            "balanced intercept_ {bal_int}"
        );

        // class_weight={0:1, 1:5}.
        let exp = cw_fit(ClassWeight::Explicit(vec![(0, 1.0), (1, 5.0)]))?;
        assert_eq!(exp.support().to_vec(), vec![1, 3, 4, 5]);
        let exp_int = exp.intercept()[0];
        assert!(
            (exp_int - (-2.0)).abs() < 1e-2,
            "explicit intercept_ {exp_int}"
        );

        // None vs balanced MUST give different intercepts — fails if
        // class_weight were ignored (R-CHAR-1).
        assert!(
            (none_int - bal_int).abs() > 1e-2,
            "None intercept {none_int} must differ from balanced {bal_int}"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // REQ-9 smoke tests: probability / predict_proba (Platt scaling).
    //
    // These pin the DETERMINISTIC contract + STRUCTURAL invariants only, NOT
    // exact probA/probB or predict_proba values. sklearn's predict_proba is
    // RNG-CV-dependent (probA_ = -0.7749 at random_state=0 vs -1.0541 at
    // random_state=1 on the binary set), so exact values are NOT a stable
    // oracle (R-CHAR-3: the asserted invariants are sklearn's DOCUMENTED
    // contract — `_base.py:829-864` "columns correspond to classes_ in sorted
    // order", `predict_proba` rows are a probability distribution — never
    // copied from the ferrolearn side).
    // -----------------------------------------------------------------------

    #[test]
    fn test_svc_predict_proba_raises_when_probability_false() -> TestResult {
        // probability=false (default): predict_proba/predict_log_proba error,
        // mirroring sklearn's raise (`_base.py:820-827`/`856-860`).
        let m = binary_fit()?; // default probability=false
        let x = Array2::from_shape_vec((1, 2), vec![3.0, 3.0]).map_err(|_| err("shape"))?;
        assert!(m.predict_proba(&x).is_err());
        assert!(m.predict_log_proba(&x).is_err());
        Ok(())
    }

    fn proba_binary_fit() -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 1.5, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
                6.0, 6.0, 5.5, 5.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .with_probability(true)
            .fit(&x, &y)
    }

    #[test]
    fn test_svc_predict_proba_binary_rows_sum_to_one() -> TestResult {
        let m = proba_binary_fit()?;
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 5.0, 5.0, 5.5, 5.5])
            .map_err(|_| err("shape"))?;
        let p = m.predict_proba(&x)?;
        assert_eq!(p.dim(), (4, 2));
        for s in 0..4 {
            let row_sum = p[[s, 0]] + p[[s, 1]];
            assert!((row_sum - 1.0).abs() < 1e-9, "row {s} sums to {row_sum}");
            for c in 0..2 {
                assert!(
                    p[[s, c]] >= 0.0 && p[[s, c]] <= 1.0,
                    "p[{s},{c}] = {} out of [0,1]",
                    p[[s, c]]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_svc_predict_proba_binary_monotone_in_decision() -> TestResult {
        // STRUCTURAL invariant: P(classes_[1]) is monotone non-decreasing in
        // the (binary) decision_function value (higher decision -> higher
        // P(class_1)), per the sigmoid `1/(1+exp(A f + B))` contract.
        let m = proba_binary_fit()?;
        // A grid of query points sweeping from the class-0 to the class-1 side.
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5, 6.0, 6.0],
        )
        .map_err(|_| err("shape"))?;
        let p = m.predict_proba(&x)?;
        let df = m.decision_function(&x)?;
        let bin = df.as_binary().ok_or_else(|| err("binary"))?;

        // Sort sample indices by decision value, then P(class_1) must be
        // non-decreasing along that order.
        let mut order: Vec<usize> = (0..5).collect();
        order.sort_by(|&a, &b| {
            bin[a]
                .partial_cmp(&bin[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut prev = f64::NEG_INFINITY;
        for &s in &order {
            let p1 = p[[s, 1]];
            assert!(
                p1 >= prev - 1e-9,
                "P(class_1) not monotone in decision: sample {s} df={} p1={p1} prev={prev}",
                bin[s]
            );
            prev = p1;
        }
        Ok(())
    }

    #[test]
    fn test_svc_predict_log_proba_equals_log_of_proba() -> TestResult {
        let m = proba_binary_fit()?;
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 3.5, 3.5, 6.0, 6.0])
            .map_err(|_| err("shape"))?;
        let p = m.predict_proba(&x)?;
        let lp = m.predict_log_proba(&x)?;
        assert_eq!(lp.dim(), p.dim());
        for s in 0..3 {
            for c in 0..2 {
                assert_relative_eq!(lp[[s, c]], p[[s, c]].ln(), epsilon = 1e-12);
            }
        }
        Ok(())
    }

    fn proba_multiclass_fit() -> Result<FittedSVC<f64, LinearKernel>, FerroError> {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 0.0, 5.5, 0.0, 5.0, 0.5, 0.0, 5.0, 0.5, 5.0,
                0.0, 5.5,
            ],
        )
        .map_err(|_| err("shape"))?;
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        SVC::new(LinearKernel)
            .with_c(1.0)
            .with_tol(1e-6)
            .with_max_iter(200_000)
            .with_probability(true)
            .fit(&x, &y)
    }

    #[test]
    fn test_svc_predict_proba_multiclass_rows_sum_to_one() -> TestResult {
        // 3-class: predict_proba is (n, 3), each row a probability
        // distribution (Wu-Lin-Weng coupling, `svm.cpp:2941`).
        let m = proba_multiclass_fit()?;
        let x = Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 5.0, 0.25, 0.25, 5.0])
            .map_err(|_| err("shape"))?;
        let p = m.predict_proba(&x)?;
        assert_eq!(p.dim(), (3, 3));
        for s in 0..3 {
            let row_sum: f64 = (0..3).map(|c| p[[s, c]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-9, "row {s} sums to {row_sum}");
            for c in 0..3 {
                assert!(
                    p[[s, c]] >= 0.0 && p[[s, c]] <= 1.0,
                    "p[{s},{c}] = {} out of [0,1]",
                    p[[s, c]]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_sigmoid_predict_overflow_safe() {
        // sigmoid_predict matches `1/(1+exp(A f + B))` and is overflow-safe at
        // extreme decision values (`svm.cpp:2032-2040`).
        let a = -1.0f64;
        let b = 0.0;
        // f large positive -> fApB = -f large negative -> p -> 1.
        let p_pos = sigmoid_predict(1000.0, a, b);
        assert!(p_pos.is_finite() && (p_pos - 1.0).abs() < 1e-6);
        // f large negative -> p -> 0.
        let p_neg = sigmoid_predict(-1000.0, a, b);
        assert!(p_neg.is_finite() && p_neg.abs() < 1e-6);
        // f = 0 -> 1/(1+exp(0)) = 0.5.
        assert_relative_eq!(sigmoid_predict(0.0, a, b), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_multiclass_probability_binary_reduces_to_pairwise() {
        // For k=2 the Wu-Lin-Weng coupling reduces to [r01, 1-r01].
        let mut r = Array2::<f64>::zeros((2, 2));
        r[[0, 1]] = 0.7;
        r[[1, 0]] = 0.3;
        let p = multiclass_probability(2, &r);
        assert_relative_eq!(p[0], 0.7, epsilon = 1e-6);
        assert_relative_eq!(p[1], 0.3, epsilon = 1e-6);
    }

    /// `compute_class_weight` matches `sklearn.utils.compute_class_weight`
    /// (`_classes.py:122-124` balanced formula) on the imbalanced set.
    #[test]
    fn test_compute_class_weight_balanced() {
        // 8 samples, 2 classes; class0 count=5, class1 count=3.
        // balanced[c] = 8 / (2 * count_c): [8/10, 8/6] = [0.8, 1.3333].
        let classes = [0usize, 1];
        let y = [0usize, 0, 0, 0, 0, 1, 1, 1];
        let w = compute_class_weight::<f64>(&ClassWeight::Balanced, &classes, &y);
        assert_relative_eq!(w[0], 0.8, epsilon = 1e-9);
        assert_relative_eq!(w[1], 8.0 / 6.0, epsilon = 1e-9);
        // None -> all 1.0.
        let wn = compute_class_weight::<f64>(&ClassWeight::None, &classes, &y);
        assert_eq!(wn, vec![1.0, 1.0]);
        // Explicit map, unlisted defaults to 1.0.
        let we = compute_class_weight::<f64>(&ClassWeight::Explicit(vec![(1, 5.0)]), &classes, &y);
        assert_eq!(we, vec![1.0, 5.0]);
    }
}
