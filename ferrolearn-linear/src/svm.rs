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
//! | REQ-1 (kernels + gamma scale/auto/float) | NOT-STARTED | open #641. The four kernel formulas + `gamma='scale'` (default, `fn resolved_for_fit in svm.rs` = `1/(n_features·X.var())`) + explicit float are done and pinned (`divergence_pin2_rbf_default_scale_gamma`; #634 closed); the `'auto' = 1/n_features` variant is not yet expressible (no estimator-level gamma enum) — pending the param-surface work #641. |
//! | REQ-2 (C-SVC SMO fit) | SHIPPED | `fn smo_binary in svm.rs` (Fan-Chen-Lin WSS) converges to libsvm's `α`; pinned by `divergence_pin5_binary_fitted_attributes in tests/divergence_svm_fit.rs` (`dual_coef_ [[-0.0408,-0.0408,0.0816]]`, `support_ [1,2,3]`, `intercept_ [-1.8565]` vs live `SVC(kernel='linear',C=1.0)`). |
//! | REQ-3 (fitted attrs + binary sign flip) | SHIPPED | `FittedSVC::{support,support_vectors,n_support,dual_coef,intercept,coef} in svm.rs` emit the libsvm layout with the binary sign flip (`_base.py:258-262`); `coef_` is linear-only (`_base.py:642-666`). Pinned by `divergence_pin5_*` (binary) + `divergence_pin6_multiclass_dual_coef_packing` (multiclass `(n_class-1,n_SV)` packing). |
//! | REQ-4 (decision_function shape/sign/ovr) | SHIPPED | `FittedSVC::decision_function in svm.rs` returns the `SvmScores<F>` enum: binary -> `SvmScores::Binary` 1-D `(n,)` = `-raw_ovo.ravel()` (positive -> `classes_[1]`, `_base.py:538-539`); multiclass -> `SvmScores::Multiclass` `(n, n_classes)` via `fn ovr_decision_function in svm.rs` (default `SvmDecisionShape::Ovr`, transcribed from `multiclass.py:520-562`) applied to `dec<0`/`-dec` (`_base.py:780`), or raw `(n, n·(n-1)/2)` for `SvmDecisionShape::Ovo`. `SVC::decision_function_shape` field + `with_decision_function_shape`. Sign normalized: `fn raw_ovo` negates `decision_value_binary` to restore libsvm's lower-index-class-`+1` ovo convention. Pinned by `divergence_pin8_multiclass_ovr_decision_function` (ovr `(9,3)` row0 `[2.2366,0.8167,-0.1833]`, row3 `[1.0606,2.2262,-0.2333]`), `divergence_pin9_multiclass_ovo_decision_function` (ovo `(9,3)` row0 `[1.2222,1.2222,0.0]`), `divergence_pin10_binary_shape_contract` (binary 1-D `(6,)`) in `tests/divergence_svm_fit.rs` (R-CHAR-3, 1e-2). Consumer: `FittedNuSVC::decision_function in nu_svm.rs` delegates (non-test, propagates `SvmScores`). |
//! | REQ-5 (predict + tie-break) | NOT-STARTED | open #638. `fn predict in svm.rs` ovo voting matches the oracle labels on separable sets (`divergence_pin3_predict_labels`), but vote ties use `max_by_key` (last-maximum) instead of libsvm's lower-class-index tie-break (`_base.py:814`). |
//! | REQ-6 (epsilon-SVR) | SHIPPED | `fn smo_svr in svm.rs` + `FittedSVR::{support,support_vectors,n_support,dual_coef,intercept}`; pinned by `divergence_pin4_svr_predict_values` (predict) + `divergence_pin7_svr_fitted_attributes` (`support_ [0,5]`, `dual_coef_ [[-0.392,0.392]]`, `intercept_ [0.14]` vs live `SVR(kernel='linear',C=100,epsilon=0.1)`). |
//! | REQ-7 (multiclass one-vs-one) | SHIPPED | `fn fit in svm.rs` (SVC) trains one `smo_binary` per class pair, `classes` = `np.unique(y)`; pinned by `divergence_pin6_multiclass_dual_coef_packing` (3-class `dual_coef_ (2,6)` libsvm packing, `support_ [1,2,3,5,6,7]`, `n_support_ [2,2,2]`, `intercept_ [1.2222,1.2222,0.0]`). |
//! | REQ-8 (constructor param surface + defaults) | NOT-STARTED | open #641. The kernel is the type parameter `K`; missing estimator-level `kernel`(string)/`degree`/`gamma`/`coef0`/`shrinking`/`class_weight`/`decision_function_shape`/`break_ties`/`random_state`; defaults diverge (`max_iter=10000` vs sklearn `-1`, `cache_size=1024` vs `200`). |
//! | REQ-9 (probability / predict_proba) | NOT-STARTED | open #642. No `probability` Platt-scaling CV (`_probA`/`_probB`), no `predict_proba`/`predict_log_proba` (`_base.py:820-925`). |
//! | REQ-10 (ferray substrate) | NOT-STARTED | open #643. `svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE). |

use std::collections::HashMap;

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Kernel trait and built-in kernels
// ---------------------------------------------------------------------------

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
    /// For kernels with a `gamma: Option<F>` parameter, a `None` gamma mirrors
    /// scikit-learn's default `gamma='scale'`, resolving to
    /// `1 / (n_features * X.var())` where `X.var()` is the population variance
    /// (ddof=0) over the whole flattened training matrix
    /// (`sklearn/svm/_base.py:236-239`). An explicit `Some(gamma)` is left
    /// verbatim. `'auto'` (`1 / n_features`) is not expressible by the current
    /// `Option<F>` surface and is tracked under #641; only `None` -> `'scale'`
    /// is resolved here.
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

/// Resolve a `None` gamma to scikit-learn's `gamma='scale'`
/// = `1 / (n_features * X.var())` (`sklearn/svm/_base.py:236-239`). An explicit
/// `Some(gamma)` is returned unchanged. When `X.var() == 0` sklearn would divide
/// by zero; sklearn itself guards this by falling back to `1.0`
/// (`_base.py:239`), so we do the same (avoiding a non-finite gamma).
fn resolve_scale_gamma<F: Float>(gamma: Option<F>, x: &Array2<F>) -> Option<F> {
    if gamma.is_some() {
        return gamma;
    }
    let n_features = match F::from(x.ncols()) {
        Some(nf) if nf > F::zero() => nf,
        _ => return Some(F::one()),
    };
    match population_variance(x) {
        Some(var) if var > F::zero() => Some(F::one() / (n_features * var)),
        // var == 0 (constant X) or empty: sklearn falls back to gamma = 1.0.
        _ => Some(F::one()),
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
    /// The gamma parameter. `None` mirrors scikit-learn's default
    /// `gamma='scale'`, resolved at fit time to `1 / (n_features * X.var())`
    /// (`_base.py:236-239`). `'auto'` (`1 / n_features`) is tracked under #641.
    pub gamma: Option<F>,
}

impl<F: Float> RbfKernel<F> {
    /// Create a new RBF kernel with auto gamma.
    #[must_use]
    pub fn new() -> Self {
        Self { gamma: None }
    }

    /// Create a new RBF kernel with a specified gamma.
    #[must_use]
    pub fn with_gamma(gamma: F) -> Self {
        Self { gamma: Some(gamma) }
    }
}

impl<F: Float> Default for RbfKernel<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Send + Sync> Kernel<F> for RbfKernel<F> {
    fn compute(&self, x: &[F], y: &[F]) -> F {
        let gamma = self.gamma.unwrap_or_else(F::one);
        let sq_dist = x.iter().zip(y.iter()).fold(F::zero(), |acc, (&a, &b)| {
            let d = a - b;
            acc + d * d
        });
        (-gamma * sq_dist).exp()
    }

    fn resolved_for_fit(&self, x: &Array2<F>) -> Self {
        Self {
            gamma: resolve_scale_gamma(self.gamma, x),
        }
    }
}

/// Polynomial kernel: `K(x, y) = (gamma * x . y + coef0)^degree`.
#[derive(Debug, Clone, Copy)]
pub struct PolynomialKernel<F> {
    /// The gamma parameter. `None` mirrors scikit-learn's default
    /// `gamma='scale'`, resolved at fit time to `1 / (n_features * X.var())`
    /// (`_base.py:236-239`). `'auto'` (`1 / n_features`) is tracked under #641.
    pub gamma: Option<F>,
    /// Polynomial degree.
    pub degree: usize,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> PolynomialKernel<F> {
    /// Create a new polynomial kernel with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: None,
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
        let gamma = self.gamma.unwrap_or_else(F::one);
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
            gamma: resolve_scale_gamma(self.gamma, x),
            degree: self.degree,
            coef0: self.coef0,
        }
    }
}

/// Sigmoid kernel: `K(x, y) = tanh(gamma * x . y + coef0)`.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidKernel<F> {
    /// The gamma parameter. `None` mirrors scikit-learn's default
    /// `gamma='scale'`, resolved at fit time to `1 / (n_features * X.var())`
    /// (`_base.py:236-239`). `'auto'` (`1 / n_features`) is tracked under #641.
    pub gamma: Option<F>,
    /// Independent term.
    pub coef0: F,
}

impl<F: Float> SigmoidKernel<F> {
    /// Create a new sigmoid kernel with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: None,
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
        let gamma = self.gamma.unwrap_or_else(F::one);
        let dot: F = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        (gamma * dot + self.coef0).tanh()
    }

    fn resolved_for_fit(&self, x: &Array2<F>) -> Self {
        Self {
            gamma: resolve_scale_gamma(self.gamma, x),
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
fn smo_binary<F: Float, K: Kernel<F>>(
    data: &[Vec<F>],
    labels: &[F],
    kernel: &K,
    c: F,
    tol: F,
    max_iter: usize,
    cache_size: usize,
) -> Result<SmoResult<F>, FerroError> {
    let n = data.len();
    let mut alphas = vec![F::zero(); n];
    let mut cache = KernelCache::new(cache_size);

    // Gradient of the dual objective: grad_i = (Q*alpha)_i - 1
    // where Q_{ij} = y_i * y_j * K(x_i, x_j).
    // Initially alpha = 0, so grad_i = -1 for all i.
    let mut grad: Vec<F> = vec![-F::one(); n];

    let two = F::one() + F::one();
    let eps = F::from(1e-12).unwrap_or_else(F::epsilon);

    for _iter in 0..max_iter {
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

            let in_up = (labels[t] > F::zero() && alphas[t] < c - eps)
                || (labels[t] < F::zero() && alphas[t] > eps);

            let in_low = (labels[t] > F::zero() && alphas[t] > eps)
                || (labels[t] < F::zero() && alphas[t] < c - eps);

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

        // Bounds for alpha_j
        let old_ai = alphas[i];
        let old_aj = alphas[j];

        let (lo, hi) = if labels[i] == labels[j] {
            let sum = old_ai + old_aj;
            ((sum - c).max(F::zero()), sum.min(c))
        } else {
            let diff = old_aj - old_ai;
            (diff.max(F::zero()), (c + diff).min(c))
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
        if alphas[i] > eps && alphas[i] < c - eps {
            // This is a free support vector.
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
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
    /// The multiclass `decision_function` shape convention
    /// (`sklearn/svm/_base.py:778-781`); default
    /// [`SvmDecisionShape::Ovr`] (sklearn's `decision_function_shape='ovr'`).
    pub decision_function_shape: SvmDecisionShape,
}

impl<F: Float, K: Kernel<F>> SVC<F, K> {
    /// Create a new `SVC` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `C = 1.0`, `tol = 1e-3`, `max_iter = 10000`,
    /// `cache_size = 1024`, `decision_function_shape = Ovr`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            max_iter: 10000,
            cache_size: 1024,
            decision_function_shape: SvmDecisionShape::Ovr,
        }
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

        // One-vs-one: train one binary SVM per pair.
        let n_classes = classes.len();
        let mut binary_models = Vec::new();

        for ci in 0..n_classes {
            for cj in (ci + 1)..n_classes {
                let class_neg = classes[ci];
                let class_pos = classes[cj];

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
                    self.c,
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
    /// Uses one-vs-one voting: each binary model casts a vote for the
    /// winning class, and the class with the most votes wins.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of features
    /// does not match the training data.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
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

            let best_class_idx = votes
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(i, _)| i);

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
    /// Maximum number of SMO iterations.
    pub max_iter: usize,
    /// Size of the kernel evaluation LRU cache.
    pub cache_size: usize,
}

impl<F: Float, K: Kernel<F>> SVR<F, K> {
    /// Create a new `SVR` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `C = 1.0`, `epsilon = 0.1`, `tol = 1e-3`,
    /// `max_iter = 10000`, `cache_size = 1024`.
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            c: F::one(),
            epsilon: F::from(0.1).unwrap(),
            tol: F::from(1e-3).unwrap(),
            max_iter: 10000,
            cache_size: 1024,
        }
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

    for _iter in 0..max_iter {
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
            gamma: Some(1.0),
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
            gamma: Some(1.0),
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
}
