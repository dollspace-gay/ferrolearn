//! One-Class SVM for novelty detection.
//!
//! This module provides [`OneClassSVM`], which learns a decision boundary
//! around the training data and classifies new points as inliers (`+1`) or
//! outliers (`-1`).
//!
//! # Algorithm
//!
//! One-Class SVM trains a standard binary SVC where all training data is
//! assigned label `+1` and a synthetic origin point is assigned label `-1`.
//! The decision function then separates the data from the origin in kernel
//! feature space. Points with positive decision values are inliers; negative
//! are outliers.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::one_class_svm::OneClassSVM;
//! use ferrolearn_linear::svm::RbfKernel;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array2, Array1};
//!
//! let x_train = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 1.0,  1.5, 1.0,  1.0, 1.5,
//!     1.2, 1.3,  1.3, 1.2,  1.1, 1.1,
//! ]).unwrap();
//!
//! let model = OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(1.0));
//! let fitted = model.fit(&x_train, &()).unwrap();
//!
//! // Most training data should be classified as inliers.
//! let preds = fitted.predict(&x_train).unwrap();
//! let inliers: usize = preds.iter().filter(|&&p| p == 1).count();
//! assert!(inliers >= 4);
//! ```
//!
//! ## REQ status
//!
//! Classification (R-DEFER-2): SHIPPED = impl + non-test production consumer +
//! tests + green oracle verification; NOT-STARTED = open blocker `#`.
//! `OneClassSVM`/`FittedOneClassSVM` are boundary estimator types re-exported at
//! the crate root (`pub use one_class_svm::{…}` in `lib.rs`) — under S5/R-DEFER-1
//! the consumer surface exists for the grandfathered public API. See
//! `.design/linear/one_class_svm.md`.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (ONE_CLASS nu dual + nu validation) | SHIPPED | `fn fit in one_class_svm.rs` validates `nu ∈ (0,1]` (`InvalidParameter`) and solves the one-class dual `0≤α≤1/(n·ν), Σα=1`, rescaled to libsvm's `Σα=ν·n` convention (`let scale = F::one()/c; rho * scale`, `dual_coefs.push(alpha*scale)`). On a NON-DEGENERATE (unique-optimum) set the SMO recovers libsvm's EXACT decomposition — `support_`/`n_support_`/`dual_coef_`/`intercept_` match the live oracle within 1e-8 (pinned by `divergence_pin5_sv_decomposition_nondegenerate_646 in tests/divergence_one_class_svm.rs`: `support_ [0,1,2,4]`, `dual_coef_ [1,0.569,0.431,1]`, `intercept_ [-1.616]`, verified unique via perturbation). DEGENERACY BOUNDARY (documented, not a gap): on the symmetric toy set the optimal face is degenerate (margin points 1,4,5 satisfy `0.5·x₁=0.25·x₄+0.25·x₅` → identical `w`), so ferrolearn's deterministic WSS reaches a different but equally-optimal vertex (5 SVs vs libsvm's 4) — sanctioned α-decomposition non-uniqueness; the hyperplane/`decision_function`/`predict` are IDENTICAL (pin1/pin3 green). |
//! | REQ-2 (kernels & gamma resolution) | SHIPPED | `fn fit in one_class_svm.rs` resolves the kernel against X at fit time via `let kernel = self.kernel.resolved_for_fit(x);` (mirroring `svm.rs`'s `SVC::fit`), used for ALL kernel evaluations in the SMO solve and stored on `FittedOneClassSVM` so decision_function/predict reuse the same gamma. `Gamma::Scale` (default) resolves to `1/(n_features·X.var())`, `Auto` to `1/n_features`, `Value` verbatim (`crate::svm::Kernel::resolved_for_fit`, `_base.py:236-243`). Pinned: `divergence_pin2_gamma_scale_default_647 in tests/divergence_one_class_svm.rs` — default `RbfKernel` (`Gamma::Scale`) on the 7×2 set gives `_gamma≈0.46578` and df matching the live `OneClassSVM(kernel='rbf',nu=0.5)` oracle `[0.022499,0.022633,0.000122,0.0,0.0,0.000387,-1.44231]` (R-CHAR-3, 1e-2). |
//! | REQ-3 (fitted attributes + offset_) | SHIPPED | The libsvm-layout accessor surface now exists: `FittedOneClassSVM::{support,support_vectors,n_support,dual_coef,intercept,offset,coef} in one_class_svm.rs` — `support_` (SV indices via the new `sv_indices` field), `support_vectors_` shape `(n_SV,n_features)`, `n_support_` `vec![n_SV]` (length 1), `dual_coef_` shape `(1,n_SV)` (libsvm scale, raw α — no `α·y` flip, `Σ=ν·n`), `intercept_=[-rho]`, `offset_=rho=-intercept_` (`_classes.py:1767`), linear-only `coef_=dual_coef_@support_vectors_` (gated on `Kernel::is_linear`, else `None`, `_base.py:650-666`). The hyperplane attributes match the live oracle: `intercept_ [-0.01]`, `offset_ 0.01` (= `-intercept_`), `coef_ [[0.05,0.05]]`. Consumer: the crate-root re-export. Pinned by `test_one_class_svm_fitted_attributes_linear_oracle` (offset_/coef_/intercept_/shapes/the `offset_=-intercept_` identity + `dual_coef_` sum `=ν·n`) + `test_one_class_svm_coef_none_for_rbf` (rbf → `None`) in `one_class_svm.rs` (R-CHAR-3, 1e-2). The `support_`/`dual_coef_`/`n_support_` decomposition matches the oracle on NON-DEGENERATE (unique) optima (`divergence_pin5_sv_decomposition_nondegenerate_646`); on the symmetric toy set the decomposition is a sanctioned non-unique vertex (REQ-1's documented degeneracy boundary — same hyperplane). |
//! | REQ-4 (decision_function / score_samples) | SHIPPED | `pub fn decision_function in one_class_svm.rs` returns `Array1<F>` `(n,)` = `Σ coef·K(sv,x) − rho` in libsvm scale (the #646 rescale: `let scale = F::one()/c; rho * scale`, `dual_coefs.push(alpha*scale)`, `svm.cpp:2834` `sum -= rho`). `pub fn score_samples in one_class_svm.rs` now returns `decision_function(X) + offset_` (`_classes.py:1801`). Pinned: `divergence_pin1_decision_function_scaling_646 in tests/divergence_one_class_svm.rs` — linear `nu=0.5` on the 7×2 set gives df `[-0.01,0.0,-0.01,-0.01,0.0,0.0,0.29]` matching the live oracle; `test_one_class_svm_score_samples_linear_oracle in one_class_svm.rs` pins `score_samples [0,0.01,0,0,0.01,0.01,0.3]` against the live oracle (R-CHAR-3, 1e-2). |
//! | REQ-5 (predict +1/-1) | SHIPPED | `fn predict in one_class_svm.rs` returns `+1` (inlier) / `-1` (outlier); labels match the live oracle `[-1,1,-1,-1,1,1,1]` (pinned by `divergence_pin3_predict_labels_648 in tests/divergence_one_class_svm.rs`, R-CHAR-3). The boundary uses a `|rho|`-relative slack so on-margin points (`decision≈0` modulo float roundoff) take libsvm's observable label (`+1`), reproducing the oracle (R-DEV-3 observable contract); libsvm's exact `(sum>0)?+1:-1` (`svm.cpp:2837-2838`) differs only at a genuine `decision==0` (measure-zero / degenerate edge). |
//! | REQ-6 (constructor params/defaults) | SHIPPED | `OneClassSVM::new` now mirrors sklearn's exact param surface defaults (`_classes.py:1712-1721`, live `inspect.signature`): `nu=0.5`, `tol=1e-3`, `cache_size=200` (was `1024`; accepted for parity, no kernel cache in this module), `max_iter=0` (was `10000`) = sklearn `max_iter=-1` ("no iteration limit"), and a new `pub shrinking: bool` field (default `true`) + `#[must_use] with_shrinking` — accepted for API parity, shrinking-invariant one-class optimum so DOES NOT alter the fit (no shrinking heuristic, R-DEV-7), mirroring `svm.rs`'s `SVC`/`SVR`. `fn fit`'s SMO loop treats `max_iter == 0` as unbounded (run to convergence) via the sentinel guard `if self.max_iter != 0 && iter >= self.max_iter { break; }` (same as `svm.rs`'s `smo_binary`/`smo_svr`); the KKT-gap break (`i_max_grad - j_min_grad < tol`) terminates the default-0 fit. R-DEV-7 design difference (preserved contract, NOT a gap): estimator-level `kernel`(string)/`degree`/`coef0` are the type parameter `K` set by construction, `gamma` resolution is REQ-2; `verbose`/`random_state` are unused (deterministic SMO). Pinned by `test_one_class_svm_default_params` (asserts `nu==0.5`, `tol==1e-3`, `max_iter==0`, `cache_size==200`, `shrinking==true` against the live `OneClassSVM.__init__` signature, R-DEV-2) + `test_one_class_svm_default_max_iter_converges` (default-0 fit converges, no infinite loop) + `test_one_class_svm_builder_pattern` (`with_shrinking`) in `one_class_svm.rs`. The 6 divergence pins use explicit `with_max_iter(1_000_000)` and stay green. |
//! | REQ-7 (ferray substrate) | NOT-STARTED | open prereq blocker #652. `one_class_svm.rs` imports `ndarray::{Array1, Array2, ScalarOperand}`, not `ferray-core` (R-SUBSTRATE). |
//! | REQ-8 (non-finite input rejected) | SHIPPED | `fn fit in one_class_svm.rs` rejects any NaN/+/-inf in X BEFORE the one-class SMO solve with `FerroError::InvalidParameter`, mirroring sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, …)` (`_base.py:190-197`, default `force_all_finite=True`) -> `ValueError("Input X contains NaN.")` / `"… contains infinity …"`. OneClassSVM is unsupervised (`Fit<Array2<F>, ()>`, X only — sklearn's `OneClassSVM.fit(X)` passes a synthetic all-ones `y`), so only X is checked; ferrolearn's `Fit::fit` has no `sample_weight` argument. `.iter().any(|v| !v.is_finite())` catches both NaN and Inf; the finite path is byte-identical (the 6 one-class divergence pins stay green). Verified vs the live sklearn 1.5.2 oracle (R-CHAR-3): NaN/+inf/-inf in X all raise `ValueError` (`tests/divergence_svm_nonfinite.rs::ocs_*`). Non-test consumer: the existing `Fit::fit` consumer + the crate-root `pub use one_class_svm::{OneClassSVM, …}` re-export. (#2269) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;

use crate::svm::Kernel;

// ---------------------------------------------------------------------------
// OneClassSVM
// ---------------------------------------------------------------------------

/// One-Class SVM for novelty detection.
///
/// Learns a decision boundary around the training data. New points are
/// classified as inliers (`+1`) or outliers (`-1`).
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `K`: The kernel type (e.g., [`RbfKernel`](super::svm::RbfKernel)).
#[derive(Debug, Clone)]
pub struct OneClassSVM<F, K> {
    /// The nu parameter: upper bound on the fraction of outliers.
    /// Must be in `(0, 1]`. Default: `0.5`.
    pub nu: F,
    /// The kernel function.
    pub kernel: K,
    /// Convergence tolerance.
    pub tol: F,
    /// Maximum number of SMO iterations. `0` is the sklearn `max_iter=-1`
    /// sentinel ("no iteration limit"; libsvm runs to convergence) — the SMO
    /// loop then runs until the KKT gap closes; a non-zero value caps the
    /// iteration count (`sklearn/svm/_classes.py:1721`, `max_iter` default `-1`).
    pub max_iter: usize,
    /// Size of the kernel evaluation cache (`sklearn` `cache_size`, default
    /// `200`). Accepted for API parity; this module has no kernel cache.
    pub cache_size: usize,
    /// Whether to use libsvm's shrinking heuristic (`sklearn` `shrinking`,
    /// `_classes.py:1718`, default `true`).
    ///
    /// Accepted for API parity. The one-class optimum is shrinking-invariant
    /// and ferrolearn's SMO implements no shrinking heuristic, so this flag
    /// DOES NOT alter the fitted `α`/`dual_coef_`/`intercept_` (R-DEV-7).
    pub shrinking: bool,
}

impl<F: Float, K: Kernel<F>> OneClassSVM<F, K> {
    /// Create a new `OneClassSVM` with the given kernel and default hyperparameters.
    ///
    /// Defaults: `nu = 0.5`, `tol = 1e-3`, `max_iter = 0` (= sklearn `-1`, no
    /// iteration limit — runs to convergence), `cache_size = 200`,
    /// `shrinking = true` (`sklearn/svm/_classes.py:1712-1721`).
    #[must_use]
    pub fn new(kernel: K) -> Self {
        Self {
            nu: F::from(0.5).unwrap_or_else(F::zero),
            kernel,
            tol: F::from(1e-3).unwrap_or_else(F::zero),
            max_iter: 0,
            cache_size: 200,
            shrinking: true,
        }
    }

    /// Set the nu parameter.
    #[must_use]
    pub fn with_nu(mut self, nu: F) -> Self {
        self.nu = nu;
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

    /// Set the `shrinking` flag (`sklearn` `shrinking`, default `true`).
    ///
    /// Accepted for API parity; the one-class optimum is shrinking-invariant
    /// (ferrolearn's SMO has no shrinking heuristic — R-DEV-7), so this does
    /// not change the fit.
    #[must_use]
    pub fn with_shrinking(mut self, shrinking: bool) -> Self {
        self.shrinking = shrinking;
        self
    }
}

/// Fitted One-Class SVM.
///
/// Stores the support vectors and decision boundary. Points are classified
/// as inliers (+1) or outliers (-1) based on the sign of the decision
/// function.
#[derive(Debug, Clone)]
pub struct FittedOneClassSVM<F, K> {
    /// The kernel used for predictions.
    kernel: K,
    /// Support vectors (stored as rows).
    support_vectors: Vec<Vec<F>>,
    /// Original training-row index of each support vector, ascending. Mirrors
    /// libsvm's `support_` accounting (`sklearn/svm/_base.py:318-410`) so the
    /// public `support_`/`support_vectors_` attributes can index back into X.
    sv_indices: Vec<usize>,
    /// Dual coefficients for each support vector.
    dual_coefs: Vec<F>,
    /// Bias (rho) term. Decision function: sign(f(x) - rho).
    rho: F,
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Fit<Array2<F>, ()>
    for OneClassSVM<F, K>
{
    type Fitted = FittedOneClassSVM<F, K>;
    type Error = FerroError;

    /// Fit the One-Class SVM.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `nu` is not in `(0, 1]`.
    /// - [`FerroError::InsufficientSamples`] if no training data is provided.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedOneClassSVM<F, K>, FerroError> {
        if self.nu <= F::zero() || self.nu > F::one() {
            return Err(FerroError::InvalidParameter {
                name: "nu".into(),
                reason: "must be in (0, 1]".into(),
            });
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "OneClassSVM requires at least one sample".into(),
            });
        }

        // Reject non-finite input (NaN / +/-inf) in X BEFORE the one-class SMO
        // solve, mirroring sklearn's `BaseLibSVM.fit` -> `_validate_data(X, y, …)`
        // (`sklearn/svm/_base.py:190-197`, default `force_all_finite=True`) ->
        // `ValueError("Input X contains NaN.")` / `"… contains infinity …"`.
        // OneClassSVM is unsupervised (`Fit<Array2<F>, ()>`, X only — sklearn's
        // `OneClassSVM.fit(X)` passes a synthetic all-ones `y`), so only X is
        // finiteness-checked; ferrolearn's `Fit::fit` has no `sample_weight`
        // argument. `.iter().any(|v| !v.is_finite())` catches both NaN and
        // +/-inf; on finite input the guard never fires (the fitted
        // `support_`/`dual_coef_`/`intercept_`/`offset_` are byte-identical).
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidParameter {
                name: "X".into(),
                reason: "Input X contains NaN or infinity.".into(),
            });
        }

        // Resolve the kernel against X at fit time (gamma='scale'/'auto'/float),
        // exactly as `svm.rs`'s `SVC::fit` does (`self.kernel.resolved_for_fit(x)`).
        // sklearn resolves `gamma='scale'` to `1/(n_features·X.var())` and
        // `'auto'` to `1/n_features` against the training X (`_base.py:236-243`);
        // without this a default `RbfKernel` (`Gamma::Scale`) silently fits with
        // `gamma=1.0`. The resolved kernel is used for ALL kernel evaluations
        // below and stored on `FittedOneClassSVM` so decision_function/predict
        // reuse the same resolved gamma.
        let kernel = self.kernel.resolved_for_fit(x);

        // Solve the one-class SVM dual:
        // max sum_i alpha_i - 0.5 * sum_{i,j} alpha_i * alpha_j * K(x_i, x_j)
        // s.t. 0 <= alpha_i <= 1/(n * nu), sum alpha_i = 1
        //
        // We use a simplified approach: initialize alphas uniformly, then
        // iterate with SMO-style updates.

        let c = F::one() / (F::from(n_samples).unwrap() * self.nu);
        let data: Vec<Vec<F>> = (0..n_samples).map(|i| x.row(i).to_vec()).collect();

        // Initialize alphas uniformly: alpha_i = 1/n
        let init_alpha = F::one() / F::from(n_samples).unwrap();
        let mut alphas = vec![init_alpha.min(c); n_samples];

        // Ensure sum(alphas) = 1 after capping at c.
        let alpha_sum: F = alphas.iter().copied().fold(F::zero(), |a, b| a + b);
        if alpha_sum < F::one() {
            // Distribute remaining mass.
            let remaining = F::one() - alpha_sum;
            let per_sample = remaining / F::from(n_samples).unwrap();
            for alpha in &mut alphas {
                *alpha = (*alpha + per_sample).min(c);
            }
        }

        // Compute initial gradient: grad_i = sum_j alpha_j * K(x_i, x_j)
        let eps = F::from(1e-12).unwrap_or_else(F::epsilon);
        let two = F::one() + F::one();

        let mut grad = vec![F::zero(); n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                grad[i] = grad[i] + alphas[j] * kernel.compute(&data[i], &data[j]);
            }
        }

        // SMO iterations. `max_iter == 0` is the sklearn `max_iter=-1` ("no
        // iteration limit", libsvm runs to convergence) sentinel — the loop
        // then runs until the KKT gap closes (the `i_max_grad - j_min_grad <
        // tol` break below). A non-zero `max_iter` caps the iteration count,
        // mirroring `svm.rs`'s `smo_binary`/`smo_svr` loop guard.
        let mut iter = 0usize;
        loop {
            if self.max_iter != 0 && iter >= self.max_iter {
                break;
            }
            iter += 1;
            // Select working set: i with largest gradient (and alpha_i > 0),
            // j with smallest gradient (and alpha_j < c).
            let mut i_best = None;
            let mut i_max_grad = F::neg_infinity();
            let mut j_best = None;
            let mut j_min_grad = F::infinity();

            for k in 0..n_samples {
                if alphas[k] > eps && grad[k] > i_max_grad {
                    i_max_grad = grad[k];
                    i_best = Some(k);
                }
                if alphas[k] < c - eps && grad[k] < j_min_grad {
                    j_min_grad = grad[k];
                    j_best = Some(k);
                }
            }

            if i_best.is_none() || j_best.is_none() || i_max_grad - j_min_grad < self.tol {
                break;
            }

            let i = i_best.unwrap();
            let j = j_best.unwrap();

            if i == j {
                break;
            }

            let kii = kernel.compute(&data[i], &data[i]);
            let kjj = kernel.compute(&data[j], &data[j]);
            let kij = kernel.compute(&data[i], &data[j]);
            let eta = kii + kjj - two * kij;

            if eta <= eps {
                continue;
            }

            // Update: move mass from i to j.
            let delta = (grad[i] - grad[j]) / eta;
            let delta = delta.min(alphas[i]).min(c - alphas[j]);

            if delta.abs() < eps {
                continue;
            }

            alphas[i] = alphas[i] - delta;
            alphas[j] = alphas[j] + delta;

            // Update gradients.
            for k in 0..n_samples {
                let kki = kernel.compute(&data[k], &data[i]);
                let kkj = kernel.compute(&data[k], &data[j]);
                grad[k] = grad[k] - delta * kki + delta * kkj;
            }
        }

        // Compute rho from the KKT conditions.
        // For free SVs (0 < alpha_i < c): rho = grad_i = sum_j alpha_j * K(i, j).
        let mut rho_sum = F::zero();
        let mut rho_count = 0usize;

        for i in 0..n_samples {
            if alphas[i] > eps && alphas[i] < c - eps {
                rho_sum = rho_sum + grad[i];
                rho_count += 1;
            }
        }

        let rho = if rho_count > 0 {
            rho_sum / F::from(rho_count).unwrap()
        } else {
            // Fallback: use the midpoint of the gradient range among all SVs.
            let sv_grads: Vec<F> = (0..n_samples)
                .filter(|&i| alphas[i] > eps)
                .map(|i| grad[i])
                .collect();

            if sv_grads.is_empty() {
                F::zero()
            } else {
                let min_g = sv_grads.iter().fold(F::infinity(), |a, &b| a.min(b));
                let max_g = sv_grads.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
                (min_g + max_g) / two
            }
        };

        // Rescale from the normalized one-class dual (0<=a<=1/(n*nu), Sum a = 1)
        // to libsvm's un-normalized convention (0<=a<=1, Sum a = nu*n). The two
        // optima are the SAME point scaled by `nu*n`, so the reported
        // coefficients and `rho` must be multiplied by `nu*n` for the decision
        // function `Sum a_i K - rho` to match the sklearn/libsvm oracle
        // (libsvm `solve_one_class` svm.cpp:1722-1736, decision svm.cpp:2834).
        // `c == 1/(n*nu)`, so the scale factor `nu*n` is exactly `1/c`.
        let scale = F::one() / c;
        let rho = rho * scale;

        // Extract support vectors. `alphas` is in training-row order, so the
        // recorded `sv_indices` are already ascending — matching libsvm's
        // `support_` ordering (`svm.cpp` keeps SVs in input order for one-class).
        let mut support_vectors = Vec::new();
        let mut sv_indices = Vec::new();
        let mut dual_coefs = Vec::new();

        for (i, &alpha) in alphas.iter().enumerate() {
            if alpha > eps {
                support_vectors.push(data[i].clone());
                sv_indices.push(i);
                dual_coefs.push(alpha * scale);
            }
        }

        // If no support vectors found, use all data as fallback: distribute the
        // total mass `nu*n` (== scale) uniformly across all n samples. With
        // `c == 1/(n*nu)`, the per-sample weight `scale/n == scale*c*nu`.
        if support_vectors.is_empty() {
            let weight = scale * c * self.nu;
            for (i, row) in data.iter().enumerate() {
                support_vectors.push(row.clone());
                sv_indices.push(i);
                dual_coefs.push(weight);
            }
        }

        let _ = n_features; // used for validation context

        Ok(FittedOneClassSVM {
            kernel,
            support_vectors,
            sv_indices,
            dual_coefs,
            rho,
        })
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static>
    FittedOneClassSVM<F, K>
{
    /// Compute the decision function value for a single sample.
    ///
    /// Returns `f(x) - rho`, where `f(x) = sum_i alpha_i * K(sv_i, x)`.
    fn decision_value(&self, x: &[F]) -> F {
        let mut val = F::zero();
        for (sv, &coef) in self.support_vectors.iter().zip(self.dual_coefs.iter()) {
            val = val + coef * self.kernel.compute(sv, x);
        }
        val - self.rho
    }

    /// Compute the raw decision function values for each sample.
    ///
    /// Returns an array of shape `(n_samples,)`. Positive values indicate
    /// inliers, negative values indicate outliers.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
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
    /// (`OneClassSVM.support_`, `sklearn/svm/_base.py:318-410`). One-class has a
    /// single "class", so the SVs are kept in training-row order.
    #[must_use]
    pub fn support(&self) -> Array1<usize> {
        Array1::from_vec(self.sv_indices.clone())
    }

    /// The support vectors, shape `(n_SV, n_features)`
    /// (`OneClassSVM.support_vectors_`). Equals `X[support_]`.
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

    /// Number of support vectors. For one-class `n_support_` has size 1 — a
    /// single count of all SVs (`sklearn/svm/_classes.py:1664`,
    /// `sklearn/svm/_base.py:680-682`).
    #[must_use]
    pub fn n_support(&self) -> Vec<usize> {
        vec![self.support_vectors.len()]
    }

    /// Dual coefficients `alpha`, shape `(1, n_SV)` (`OneClassSVM.dual_coef_`).
    /// For one-class these are the raw `alpha` (NOT `alpha*y`), already rescaled
    /// in `fn fit` to libsvm's `Sum alpha = nu*n` convention
    /// (`sklearn/svm/_classes.py:1639`). No sign flip applies to one-class
    /// (`sklearn/svm/_base.py:258-262` restricts the flip to `c_svc`/`nu_svc`).
    #[must_use]
    pub fn dual_coef(&self) -> Array2<F> {
        let n_sv = self.dual_coefs.len();
        let mut out = Array2::<F>::zeros((1, n_sv));
        for (c, &v) in self.dual_coefs.iter().enumerate() {
            out[[0, c]] = v;
        }
        out
    }

    /// The intercept, length 1 (`OneClassSVM.intercept_`) = `-rho`. The
    /// one-class decision function is `Sum alpha*K - rho`, so the public
    /// intercept is `-rho` (libsvm `svm.cpp:2834` `sum -= rho`,
    /// `sklearn/svm/_base.py` `_intercept_`).
    #[must_use]
    pub fn intercept(&self) -> Array1<F> {
        Array1::from_vec(vec![-self.rho])
    }

    /// The offset, a scalar (`OneClassSVM.offset_`) = `rho` = `-intercept_`
    /// (`sklearn/svm/_classes.py:1767`: `self.offset_ = -self._intercept_`).
    /// Used to shift the decision function back to the raw score:
    /// `decision_function = score_samples - offset_`.
    #[must_use]
    pub fn offset(&self) -> F {
        self.rho
    }

    /// Primal weight vector `coef_ = dual_coef_ @ support_vectors_`, shape
    /// `(1, n_features)` — available ONLY for the linear kernel
    /// (`sklearn/svm/_base.py:650-666`). Returns `None` for any other kernel
    /// (sklearn raises `AttributeError`).
    #[must_use]
    pub fn coef(&self) -> Option<Array2<F>> {
        if !self.kernel.is_linear() {
            return None;
        }
        let dual = self.dual_coef(); // (1, n_SV)
        let svs = self.support_vectors(); // (n_SV, n_features)
        Some(dual.dot(&svs))
    }

    /// Raw (unshifted) scoring function of the samples,
    /// `score_samples(X) = decision_function(X) + offset_`
    /// (`sklearn/svm/_classes.py:1801`). Equals the unshifted `Sum alpha*K`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Self::decision_function`].
    pub fn score_samples(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        let dec = self.decision_function(x)?;
        let off = self.offset();
        Ok(dec.mapv(|v| v + off))
    }
}

impl<F: Float + Send + Sync + ScalarOperand + 'static, K: Kernel<F> + 'static> Predict<Array2<F>>
    for FittedOneClassSVM<F, K>
{
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict inlier (+1) or outlier (-1) for each sample.
    ///
    /// # Errors
    ///
    /// Returns `Ok` always for valid input.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let n_samples = x.nrows();
        let mut predictions = Array1::<isize>::zeros(n_samples);

        // libsvm ONE_CLASS uses `(sum > 0) ? +1 : -1` (svm.cpp:2837-2838). A
        // point exactly on the decision boundary (a free support vector, whose
        // decision value is mathematically 0) is reported `+1` by libsvm: its
        // converged `rho` lands fractionally below the on-boundary kernel sum
        // (e.g. 0.00999999977 vs 0.01), so `sum` is a small positive. ferrolearn
        // recovers `rho` exactly, leaving boundary points at machine-epsilon
        // noise of either sign. To reproduce libsvm's observable labels
        // (R-DEV-3), treat decision values within solver-precision of 0 as the
        // inlier side: a relative slack off `|rho|` separates true outliers
        // (whose `|sum|` is order `|rho|`) from on-boundary roundoff.
        let boundary = self.rho.abs() * F::from(1e-9).unwrap_or_else(F::epsilon);
        for s in 0..n_samples {
            let xi: Vec<F> = x.row(s).to_vec();
            let val = self.decision_value(&xi);
            predictions[s] = if val > -boundary { 1 } else { -1 };
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::{LinearKernel, RbfKernel};
    use ndarray::Array2;

    fn make_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 0.9, 0.9, 1.0, 0.9, 0.9, 1.0, 1.05, 1.05,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_one_class_svm_fit() {
        let x = make_cluster_data();
        let model = OneClassSVM::<f64, RbfKernel<f64>>::new(RbfKernel::with_gamma(10.0));
        let result = model.fit(&x, &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_one_class_svm_inliers() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();

        // Most training points should be classified as inliers.
        let inliers: usize = preds.iter().filter(|&&p| p == 1).count();
        assert!(inliers >= 6, "Expected at least 6 inliers, got {inliers}");
    }

    #[test]
    fn test_one_class_svm_outlier_detection() {
        let x_train = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.05, 0.05, -0.05,
                -0.05,
            ],
        )
        .unwrap();

        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x_train, &()).unwrap();

        // A far-away point should be an outlier.
        let x_outlier = Array2::from_shape_vec((1, 2), vec![100.0, 100.0]).unwrap();
        let preds = fitted.predict(&x_outlier).unwrap();
        assert_eq!(preds[0], -1, "Far-away point should be an outlier");
    }

    #[test]
    fn test_one_class_svm_decision_function() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(RbfKernel::with_gamma(10.0)).with_nu(0.1);
        let fitted = model.fit(&x, &()).unwrap();

        let df = fitted.decision_function(&x).unwrap();
        assert_eq!(df.len(), 8);

        // Most decision values should be non-negative for training data.
        let positive: usize = df.iter().filter(|&&v| v >= 0.0).count();
        assert!(
            positive >= 6,
            "Expected at least 6 positive df, got {positive}"
        );
    }

    #[test]
    fn test_one_class_svm_invalid_nu() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();

        let model = OneClassSVM::new(RbfKernel::<f64>::new()).with_nu(0.0);
        assert!(model.fit(&x, &()).is_err());

        let model2 = OneClassSVM::new(RbfKernel::<f64>::new()).with_nu(1.5);
        assert!(model2.fit(&x, &()).is_err());
    }

    #[test]
    fn test_one_class_svm_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let model = OneClassSVM::new(RbfKernel::<f64>::new());
        assert!(model.fit(&x, &()).is_err());
    }

    #[test]
    fn test_one_class_svm_builder_pattern() {
        let model = OneClassSVM::<f64, LinearKernel>::new(LinearKernel)
            .with_nu(0.3)
            .with_tol(1e-4)
            .with_max_iter(5000)
            .with_cache_size(2048)
            .with_shrinking(false);

        assert!((model.nu - 0.3).abs() < 1e-10);
        assert!((model.tol - 1e-4).abs() < 1e-10);
        assert_eq!(model.max_iter, 5000);
        assert_eq!(model.cache_size, 2048);
        assert!(!model.shrinking);
    }

    /// REQ-6 (R-DEV-2): `OneClassSVM::new` exposes sklearn's exact param
    /// surface defaults. Expected values from the live oracle:
    ///   python3 -c "from sklearn.svm import OneClassSVM; import inspect; \
    ///   print({k:v.default for k,v in \
    ///   inspect.signature(OneClassSVM.__init__).parameters.items() if k!='self'})"
    ///   # kernel='rbf' degree=3 gamma='scale' coef0=0.0 tol=1e-3 nu=0.5
    ///   #   shrinking=True cache_size=200 max_iter=-1
    /// (`max_iter=-1` maps to ferrolearn's `0` sentinel = no iteration limit).
    #[test]
    fn test_one_class_svm_default_params() {
        let model = OneClassSVM::<f64, LinearKernel>::new(LinearKernel);
        assert!((model.nu - 0.5).abs() < 1e-12, "nu default 0.5");
        assert!((model.tol - 1e-3).abs() < 1e-12, "tol default 1e-3");
        assert_eq!(model.max_iter, 0, "max_iter default 0 (= sklearn -1)");
        assert_eq!(model.cache_size, 200, "cache_size default 200");
        assert!(model.shrinking, "shrinking default true");
    }

    /// A default-`max_iter` (`0` = unbounded) fit must run to convergence, NOT
    /// run zero iterations or spin forever: the `i_max_grad - j_min_grad < tol`
    /// break terminates the loop. Verifies the sentinel loop guard.
    #[test]
    fn test_one_class_svm_default_max_iter_converges() {
        let x = oracle_7x2();
        let model = OneClassSVM::new(LinearKernel).with_nu(0.5);
        assert_eq!(model.max_iter, 0, "uses the default unbounded sentinel");
        let fit = model.fit(&x, &());
        assert!(
            fit.is_ok(),
            "default max_iter=0 fit converges, no infinite loop"
        );
        let Ok(fitted) = fit else { return };
        // Converged to a usable boundary: it produces a label per sample.
        let preds = fitted.predict(&x);
        assert!(preds.is_ok(), "predict succeeds on converged fit");
        let Ok(preds) = preds else { return };
        assert_eq!(preds.len(), 7);
    }

    #[test]
    fn test_one_class_svm_linear_kernel() {
        let x = make_cluster_data();
        let model = OneClassSVM::new(LinearKernel).with_nu(0.5);
        let fitted = model.fit(&x, &()).unwrap();
        let preds = fitted.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_one_class_svm_single_sample() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let model = OneClassSVM::new(RbfKernel::with_gamma(1.0)).with_nu(0.5);
        let result = model.fit(&x, &());
        assert!(result.is_ok());
    }

    /// The 7x2 contract set; built with `arr2` (no `Result`) so the smoke
    /// tests stay free of the forbidden `.unwrap()` token even under the
    /// Edit-path gate (which does not exempt `#[cfg(test)]`).
    fn oracle_7x2() -> Array2<f64> {
        ndarray::arr2(&[
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [0.0, 0.2],
            [0.2, 0.0],
            [3.0, 3.0],
        ])
    }

    // Expected values come from the live sklearn 1.5.2 oracle (R-CHAR-3),
    // NOT copied from ferrolearn:
    //
    //   python3 -c "import numpy as np; from sklearn.svm import OneClassSVM; \
    //     X=np.array([[0,0],[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[0,0.2],[0.2,0],[3,3]],dtype=float); \
    //     m=OneClassSVM(kernel='linear',nu=0.5).fit(X); \
    //     print(m.support_.tolist(), m.n_support_.tolist()); \
    //     print(np.round(m.dual_coef_,6).tolist()); \
    //     print(np.round(m.intercept_,6).tolist(), np.round(m.offset_,6).tolist()); \
    //     print(np.round(m.coef_,6).tolist()); \
    //     print(np.round(m.score_samples(X),6).tolist())"
    //
    //   support_ [0,1,2,3]  n_support_ [4]
    //   dual_coef_ [[1.0,0.5,1.0,1.0]]  (sum 3.5 = nu*n = 0.5*7)
    //   intercept_ [-0.01]  offset_ [0.01]  coef_ [[0.05,0.05]]
    //   score_samples [0.0,0.01,0.0,0.0,0.01,0.01,0.3]

    #[test]
    fn test_one_class_svm_fitted_attributes_linear_oracle() {
        let fit = OneClassSVM::new(LinearKernel)
            .with_nu(0.5)
            .fit(&oracle_7x2(), &());
        assert!(fit.is_ok(), "linear one-class fit should succeed");
        let Ok(fitted) = fit else { return };

        // The hyperplane-level attributes match the live oracle exactly
        // (coef_/intercept_/offset_ and decision_function/score_samples — see
        // the score_samples test). NOTE: the SV-decomposition attributes
        // (support_/dual_coef_/n_support_) DIVERGE — ferrolearn's SMO converges
        // to a different vertex of the same optimal face: it reports 5 SVs
        // {0,2,3,4,5} with dual_coef_ [[1,1,1,0.25,0.25]] vs the live oracle's
        // 4 SVs {0,1,2,3} with [[1,0.5,1,1]]. Both sum to nu*n=3.5 and yield
        // the SAME hyperplane (coef_=[[0.05,0.05]], intercept_=[-0.01]), so the
        // decision function matches. The SV-set divergence is a solver-optimum
        // divergence (REQ-1), filed as a new blocker for the critic to pin
        // rigorously and a fixer to resolve in the SMO working-set selection;
        // these accessors faithfully report whatever the solver converged to.

        // support_/support_vectors_ shapes are coherent with each other.
        let support = fitted.support();
        let svs = fitted.support_vectors();
        assert_eq!(
            svs.nrows(),
            support.len(),
            "support_vectors_ rows == |support_|"
        );
        assert_eq!(svs.ncols(), 2, "support_vectors_ n_features == 2");
        // support_ is ascending and indexes valid training rows.
        for w in support.windows(2).into_iter() {
            assert!(w[0] < w[1], "support_ strictly ascending");
        }
        for &i in support.iter() {
            assert!(i < 7, "support_ index in range");
        }

        // n_support_ has length 1 (one-class single "class") and equals |SV|.
        let n_support = fitted.n_support();
        assert_eq!(n_support.len(), 1, "n_support_ length 1 for one-class");
        assert_eq!(n_support[0], support.len(), "n_support_[0] == |support_|");

        // dual_coef_ shape (1, n_SV); its sum is the libsvm-scale total nu*n=3.5
        // (the rescale identity, scale-invariant of the SV decomposition).
        let dual = fitted.dual_coef();
        assert_eq!(dual.dim(), (1, support.len()), "dual_coef_ shape (1, n_SV)");
        let dual_sum: f64 = dual.iter().sum();
        assert!((dual_sum - 3.5).abs() < 1e-2, "dual_coef_ sum = nu*n = 3.5");

        // intercept_ = [-0.01], offset_ = 0.01 = -intercept_ (matches oracle).
        let intercept = fitted.intercept();
        assert_eq!(intercept.len(), 1, "intercept_ length 1");
        assert!(
            (intercept[0] - (-0.01)).abs() < 1e-2,
            "intercept_ vs oracle [-0.01]"
        );
        let offset = fitted.offset();
        assert!((offset - 0.01).abs() < 1e-2, "offset_ vs oracle 0.01");
        assert!(
            (offset - (-intercept[0])).abs() < 1e-12,
            "offset_ = -intercept_"
        );

        // coef_ = dual_coef_ @ support_vectors_ = [[0.05, 0.05]] (matches oracle:
        // the primal hyperplane is identical despite the different SV set).
        let coef = fitted.coef();
        assert!(coef.is_some(), "linear kernel exposes coef_");
        if let Some(coef) = coef {
            assert_eq!(coef.dim(), (1, 2), "coef_ shape (1, n_features)");
            assert!(
                (coef[[0, 0]] - 0.05).abs() < 1e-2,
                "coef_[0][0] vs oracle 0.05"
            );
            assert!(
                (coef[[0, 1]] - 0.05).abs() < 1e-2,
                "coef_[0][1] vs oracle 0.05"
            );
        }
    }

    #[test]
    fn test_one_class_svm_score_samples_linear_oracle() {
        let fit = OneClassSVM::new(LinearKernel)
            .with_nu(0.5)
            .fit(&oracle_7x2(), &());
        assert!(fit.is_ok(), "linear one-class fit should succeed");
        let Ok(fitted) = fit else { return };

        // score_samples = decision_function + offset_ = [0,0.01,0,0,0.01,0.01,0.3].
        let scores_res = fitted.score_samples(&oracle_7x2());
        assert!(scores_res.is_ok(), "score_samples should succeed");
        let df_res = fitted.decision_function(&oracle_7x2());
        assert!(df_res.is_ok(), "decision_function should succeed");
        let (Ok(scores), Ok(df)) = (scores_res, df_res) else {
            return;
        };
        let expected = [0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.3];
        assert_eq!(scores.len(), expected.len());
        for (i, &v) in expected.iter().enumerate() {
            assert!(
                (scores[i] - v).abs() < 1e-2,
                "score_samples[{i}] = {} vs oracle {v}",
                scores[i]
            );
        }

        // Cross-check the identity score_samples = decision_function + offset_.
        for i in 0..scores.len() {
            assert!((scores[i] - (df[i] + fitted.offset())).abs() < 1e-12);
        }
    }

    #[test]
    fn test_one_class_svm_coef_none_for_rbf() {
        // coef_ is linear-only; non-linear kernels return None (sklearn raises
        // AttributeError, `sklearn/svm/_base.py:650-651`).
        let fit = OneClassSVM::new(RbfKernel::with_gamma(1.0))
            .with_nu(0.5)
            .fit(&oracle_7x2(), &());
        assert!(fit.is_ok(), "rbf one-class fit should succeed");
        let Ok(fitted) = fit else { return };
        assert!(fitted.coef().is_none(), "rbf kernel has no coef_");
    }
}
