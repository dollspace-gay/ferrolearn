//! Label Spreading for semi-supervised classification.
//!
//! This module provides [`LabelSpreading`], a graph-based semi-supervised
//! learning algorithm similar to [`LabelPropagation`](super::label_propagation::LabelPropagation)
//! but using the **normalized graph Laplacian** for smoother label propagation.
//!
//! # Algorithm
//!
//! Mirrors `sklearn.semi_supervised.LabelSpreading`
//! (`_label_propagation.py`, `_variant="spreading"`):
//!
//! 1. Build an affinity matrix `W` using either an RBF or KNN kernel (RBF
//!    self-affinity diagonal `= 1`).
//! 2. Build the spreading graph matrix = the symmetric normalized Laplacian
//!    `S = D^{-1/2} W D^{-1/2}` with the diagonal zeroed
//!    (`csgraph_laplacian(normed=True)` then `-laplacian` then zero-diagonal,
//!    `_build_graph` `:609-623`). The degree `D` is the OFF-diagonal row sum
//!    (scipy `csgraph_laplacian` ignores self-loops).
//! 3. Initialize `label_distributions_` to a one-hot over `classes_` for
//!    labeled rows; **unlabeled rows start at zero**.
//! 4. Iterate `label_distributions_ = alpha * (graph @ label_distributions_)
//!    + y_static` with `y_static = one-hot * (1 - alpha)` (NO per-iteration
//!    normalization, NO clamping — soft clamping via `y_static`).
//! 5. Convergence: the L1 abs-sum `|label_distributions_ - l_previous|.sum() <
//!    tol`, checked at the START of each iteration against the previous
//!    iterate; `n_iter_` is the loop counter, set to `max_iter` exactly when
//!    `max_iter` is reached without convergence. A final row-normalization is
//!    applied once.
//!
//! Inductive `predict_proba(X)` is the kernel-weighted combination over all
//! training rows (`rbf_kernel(X_train, X).T @ label_distributions_`,
//! row-normalized); `predict(X) = classes_[argmax(predict_proba(X))]`.
//!
//! The `alpha` parameter controls the trade-off between the initial label
//! information and the graph structure. It must lie in the open interval
//! `(0, 1)` (both `0` and `1` are rejected): lower values keep labels closer
//! to the initial values, higher values give more weight to graph propagation.
//!
//! Labels of `-1` in the target vector indicate unlabeled points.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::LabelSpreading;
//! use ferrolearn_core::Fit;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);
//!
//! let model = LabelSpreading::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/semi_supervised/_label_propagation.py`
//! (`class LabelSpreading(BaseLabelPropagation)` `:486`, `_variant="spreading"`; base
//! `fit`/`predict`/`predict_proba` `:233-335`; `_build_graph` `:609-623`). Design doc:
//! `.design/cluster/label_spreading.md`. Cites use ferrolearn symbol anchors / sklearn
//! `file:line` (commit 156ef14); expected values from the live sklearn 1.5.2 oracle
//! (R-CHAR-3). The contiguous-label transduction PARTITION (REQ-1), the
//! `alpha ∈ (0,1)` validation (REQ-2), and the `classes_`/`n_classes`/label-VALUE
//! mapping (REQ-4) ship through the crate re-export. The `label_distributions_`
//! VALUES, `n_iter_`, the `tol` default, and `predict`/`predict_proba` now MATCH
//! sklearn bit-exactly: `fn build_rbf_affinity` keeps the self-affinity diagonal
//! `=1` (`rbf_kernel`, `:147`), `fn spreading_graph` builds the symmetric
//! normalized-Laplacian graph `D^{-1/2} W D^{-1/2}` (off-diagonal degree, diagonal
//! zeroed; `_build_graph` `:609-623`), `fn fit` zero-inits unlabeled rows and sets
//! `y_static = one-hot*(1-alpha)` (`:282,290-292`), `fn spread` runs the soft-clamp
//! update `alpha*graph@F + y_static` with NO per-iteration normalization and
//! converges on the L1-at-start rule tracking `n_iter_` (`:300-330`), `fn new`
//! defaults `tol = 1e-3` (`:595`), and `fn predict_proba` is the kernel-weighted
//! combination over all training rows (`:218-231`) — REQ-3/5/6/7 SHIPPED (closing
//! #1010/#1012/#1013/#1014). There is no CPython binding (#1020).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (contiguous-label transduction PARTITION) | SHIPPED | impl `fn fit` (graph build → `fn spreading_graph` → `fn spread` → per-row argmax) recovers sklearn's transduction on well-separated CONTIGUOUS-label data. Consumer: crate re-export `pub use label_spreading::{FittedLabelSpreading, LabelSpreading, LabelSpreadingKernel}` (`lib.rs:111`). Guards: `green_guard_req1_contiguous_partition_2blob`/`_3blob`/`_fresh` in `tests/divergence_label_spreading.rs` (live-oracle). The `label_distributions_` VALUES now match too (REQ-3/5). |
//! | REQ-2 (`alpha ∈ (0,1)` open-interval validation) | SHIPPED | impl `fn fit` now rejects `alpha <= 0 || alpha >= 1` (reason "must be in (0, 1)"), matching sklearn `_parameter_constraints["alpha"] = [Interval(Real, 0, 1, closed="neither")]` (`_label_propagation.py:585`) — alpha=0 AND alpha=1 both rejected. Guard: `divergence_req2_alpha_zero_rejected` + `confirm_alpha_one_already_rejected` + in-tree `test_alpha_zero_rejected`/`test_invalid_alpha`. Fixed #1009. |
//! | REQ-4 (`classes_` / `n_classes` / label-VALUE mapping) | SHIPPED | impl `fn fit` now builds `classes_` = sorted unique non-(-1) labels, `n_classes = classes_.len()`, one-hot indexed by class POSITION, and maps the final argmax index through `classes_` — matching sklearn `classes_ = unique(y)\{-1}` + `transduction_ = classes_[argmax]` (`:272-274,333`). Guard: `divergence_req4_noncontiguous_classes_mapping` (`{0,2}` fixture → `n_classes()==2`, `labels ⊆ {0,2}` = sklearn `[0,0,0,0,2,2,2,2]`; was `n_classes()==3` phantom). Fixed #1011. |
//! | REQ-3 (`label_distributions_` value — Laplacian graph + spreading iteration) | SHIPPED | closes #1010. `fn build_rbf_affinity` sets the self-affinity diagonal `=1` (`rbf_kernel`, `:147`); `fn spreading_graph` builds `D^{-1/2} W D^{-1/2}` with off-diagonal degree (scipy `csgraph_laplacian` ignores self-loops — verified `csgraph_laplacian([[5,1,0],[1,5,1],[0,1,5]],return_diag=True)` → degree `[1,2,1]`) and a zeroed diagonal (`_build_graph` `:609-623`); `fn fit` zero-inits unlabeled rows and sets `y_static = one-hot*(1-alpha)` (`:282,290-292`); `fn spread` runs `alpha*graph@F + y_static` with NO per-iteration normalization, one final row-normalize (`:316-330`). Consumer: crate re-export (`lib.rs:111`). Guards: `parity_req3_label_distributions_line_default_alpha`/`_gamma20`/`_alpha05`/`_alpha08`/`_three_class` assert live-oracle rows to 1e-6 across gamma {1,20}, alpha {0.2,0.5,0.8}, 2- and 3-class (e.g. `line` gamma=1 alpha=0.2 → `[[0.95249527,0.04750473],[0.57168677,0.42831323],[0.4557925,0.5442075],[0.04756047,0.95243953]]`). |
//! | REQ-5 (convergence — L1-at-start + `n_iter_`) | SHIPPED | closes #1012. `fn spread` checks `\|label_distributions_ - l_previous\|.sum() < tol` at the loop START (L1, against the previous iterate, `:301`), tracks `n_iter_` (the loop counter; `== max_iter` on non-convergence, `:321-326`), and applies the final row-normalization (`:328-330`). Guards: the `parity_req3_*` tests assert live-oracle `n_iter_` ∈ {4,5,6,9}; `parity_req5_n_iter_max_iter_hit` asserts `n_iter_ == max_iter` on a `tol=1e-12,max_iter=5` non-convergence case. |
//! | REQ-6 (`tol` default `1e-3`) | SHIPPED | closes #1013. `fn new` sets `tol = F::from(1e-3)` matching sklearn `LabelSpreading` `tol=1e-3` (`:595`). Consumer: crate re-export (`lib.rs:111`). Guard: `parity_req6_default_tol_and_params` (also pins `alpha=0.2`/`max_iter=30`/`gamma=20`/`n_neighbors=7`/`kernel=rbf`). |
//! | REQ-7 (`predict`/`predict_proba` kernel-weighted) | SHIPPED | closes #1014. `fn predict_proba` = `rbf_kernel(X_train,X).T @ label_distributions_` row-normalized (`:218-231`); `fn predict` = `classes_[argmax(predict_proba)]` (`:190-191`). Guard: `parity_req7_predict_proba_kernel_weighted` asserts live-oracle `predict_proba` rows (1e-6, sum-to-1) + `predict`. R-DEV-3. |
//! | REQ-8 (`transduction_`/`classes_`/`n_iter_`/`X_` attrs) | NOT-STARTED | open prereq blocker #1015. `FittedLabelSpreading` now exposes `fn classes` (`classes_`) and `fn n_iter` (`n_iter_`) accessors, but the labels are still named `fn labels` (not `transduction_`) and there is no `X_` / `n_features_in_` accessor — the full sklearn attribute surface (`:264,274,300,333`) is not yet mirrored. |
//! | REQ-9 (`ConvergenceWarning` + `n_iter_`) | NOT-STARTED | open prereq blocker #1016. sklearn warns `ConvergenceWarning` + `n_iter_ += 1` at `max_iter` (`:321-326`); ferrolearn `fn spread` breaks silently, no `n_iter_`. |
//! | REQ-10 (KNN connectivity graph — directed vs symmetrized) | NOT-STARTED | open prereq blocker #1017. sklearn knn → `kneighbors_graph(mode="connectivity")` directed (`:156-157`); ferrolearn `fn build_knn_affinity` SYMMETRIZES (`w[i,j]=w[j,i]=1`). Different graph. |
//! | REQ-11 (validation / error ABI) | NOT-STARTED | open prereq blocker #1018. sklearn `check_classification_targets` + `_parameter_constraints` raising `InvalidParameterError` (`:110-118,265`); ferrolearn `fn fit` raises `FerroError::InvalidParameter` (different type/ABI) and rejects `gamma>0` (stricter than sklearn's `[0,∞)`). |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1019. `label_spreading.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-1/2). |
//! | REQ-13 (PyO3 binding) | NOT-STARTED | open prereq blocker #1020. `grep LabelSpreading ferrolearn-python/` is EMPTY — no binding; `import ferrolearn` cannot reach `LabelSpreading`. Only consumer is the crate re-export. |
//! | REQ-14 (reject non-finite input) | SHIPPED | `fn reject_non_finite` called in `Fit::fit` (after the alpha/gamma/param checks, before the affinity build) rejects NaN AND infinity in the FEATURE matrix `X` with `FerroError::InvalidParameter{name:"X"}`, mirroring sklearn's `LabelSpreading.fit`→`BaseLabelPropagation.fit`→`_validate_data(X, y, force_all_finite=True)` default (`_label_propagation.py:258`), which raises `ValueError` (`validation.py:147-154`). `y` carries integer labels (the `-1` unlabeled sentinel) and is not finite-checked. Consumer: the existing `fit` entry — crate re-export `pub use label_spreading::{FittedLabelSpreading, LabelSpreading, LabelSpreadingKernel}` (`lib.rs`). Pinned by `divergence_nonfinite_reject_spillover.rs` (`divergence_label_spreading_fit_rejects_nan`/`_inf`) — live sklearn 1.5.2 raises, ferrolearn now `Err`. Finite input byte-identical (the module's oracle pins stay green). Closes #2286 for this estimator. |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// The kernel used to build the affinity matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSpreadingKernel {
    /// RBF (Gaussian) kernel: `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
    Rbf,
    /// KNN kernel: `W[i,j] = 1` if j is among the k nearest neighbors of i
    /// (or vice versa), `0` otherwise.
    Knn,
}

/// Label Spreading semi-supervised classifier (unfitted).
///
/// Holds hyperparameters. Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedLabelSpreading`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LabelSpreading<F> {
    /// The kernel to use for building the affinity matrix.
    pub kernel: LabelSpreadingKernel,
    /// Gamma parameter for the RBF kernel.
    pub gamma: F,
    /// Number of neighbors for the KNN kernel.
    pub n_neighbors: usize,
    /// Maximum number of spreading iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
    /// Clamping factor: controls the balance between initial labels and
    /// graph structure. Must be in `(0, 1)`.
    pub alpha: F,
}

impl<F: Float> LabelSpreading<F> {
    /// Create a new `LabelSpreading` with default parameters.
    ///
    /// Defaults: `kernel = Rbf`, `gamma = 20.0`, `n_neighbors = 7`,
    /// `max_iter = 30`, `tol = 1e-3`, `alpha = 0.2` — matching
    /// `sklearn.semi_supervised.LabelSpreading.__init__`
    /// (`_label_propagation.py:587-607`; `alpha=0.2` `:593`, `max_iter=30`
    /// `:594`, `tol=1e-3` `:595`, `gamma=20` `:591`, `n_neighbors=7` `:592`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernel: LabelSpreadingKernel::Rbf,
            gamma: F::from(20.0).unwrap_or_else(F::one),
            n_neighbors: 7,
            max_iter: 30,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
            alpha: F::from(0.2).unwrap_or_else(F::zero),
        }
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: LabelSpreadingKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the gamma parameter for the RBF kernel.
    #[must_use]
    pub fn with_gamma(mut self, gamma: F) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the number of neighbors for the KNN kernel.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
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

    /// Set the clamping factor `alpha`.
    ///
    /// Must be in the open interval `(0, 1)` (both `0` and `1` are rejected,
    /// matching sklearn `_parameter_constraints["alpha"]`). Lower values keep
    /// labels closer to the initial labels; higher values give more weight to
    /// the graph structure.
    #[must_use]
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<F: Float> Default for LabelSpreading<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Label Spreading model.
///
/// Stores the final transduction labels, the (row-normalized) label
/// distribution matrix, the distinct training classes, the number of
/// iterations run, and the training data + kernel needed for inductive
/// inference. Implements [`Predict`] via the sklearn kernel-weighted
/// combination over ALL training rows (`_label_propagation.py:190-231`).
#[derive(Debug, Clone)]
pub struct FittedLabelSpreading<F> {
    /// Final transduction labels for each training sample (`transduction_`).
    labels_: Array1<isize>,
    /// Label distribution matrix, shape `(n_samples, n_classes)`.
    label_distributions_: Array2<F>,
    /// The distinct labels used during fit (`classes_`), sorted ascending,
    /// excluding the `-1` unlabeled sentinel. Column `c` of
    /// `label_distributions_` corresponds to `classes_[c]`.
    classes_: Vec<isize>,
    /// Number of iterations the spreading loop ran (`n_iter_`).
    n_iter_: usize,
    /// Training data, stored for inductive `predict` / `predict_proba`.
    x_train_: Array2<F>,
    /// The kernel used to build the affinity matrix (for inductive inference).
    kernel_: LabelSpreadingKernel,
    /// Gamma parameter for the RBF kernel (for inductive inference).
    gamma_: F,
    /// Number of neighbors for the KNN kernel (for inductive inference).
    n_neighbors_: usize,
    /// Number of classes.
    n_classes_: usize,
}

impl<F: Float> FittedLabelSpreading<F> {
    /// Return the final labels for the training data.
    #[must_use]
    pub fn labels(&self) -> &Array1<isize> {
        &self.labels_
    }

    /// Return the label distribution matrix.
    ///
    /// Shape: `(n_samples, n_classes)`. Each row sums to approximately 1.
    #[must_use]
    pub fn label_distributions(&self) -> &Array2<F> {
        &self.label_distributions_
    }

    /// Return the number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes_
    }

    /// Return the distinct labels used during fit (`classes_`).
    ///
    /// Sorted ascending, excluding the `-1` unlabeled sentinel. Column `c` of
    /// [`label_distributions`](Self::label_distributions) corresponds to
    /// `classes_[c]`. Mirrors sklearn `classes_`
    /// (`_label_propagation.py:272-274`).
    #[must_use]
    pub fn classes(&self) -> &[isize] {
        &self.classes_
    }

    /// Return the number of iterations the spreading loop ran (`n_iter_`).
    ///
    /// Mirrors sklearn `n_iter_` — the L1-at-start convergence loop counter,
    /// set to `max_iter` exactly when `max_iter` is reached without convergence
    /// (`_label_propagation.py:300-326`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

impl<F: Float + Send + Sync + 'static> FittedLabelSpreading<F> {
    /// Per-class probability for each query sample. Mirrors sklearn
    /// `BaseLabelPropagation.predict_proba` (`_label_propagation.py:193-231`):
    /// a kernel-weighted combination over ALL training rows, row-normalized.
    ///
    /// For the RBF kernel:
    /// `weight_matrices = rbf_kernel(X_train, X_query)` (shape
    /// `(n_train, n_query)`), then `probabilities = weight_matrices.T @
    /// label_distributions_` (`:227-228`), then each row is divided by its sum
    /// (`:229-230`). For the KNN kernel: each query's `n_neighbors` nearest
    /// training rows' distributions are summed (`:218-225`) and row-normalized.
    ///
    /// Returns shape `(n_samples, n_classes)`; each row sums to `1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count does not
    /// match the training data.
    pub fn predict_proba(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = x.ncols();
        let expected = self.x_train_.ncols();
        if n_features != expected {
            return Err(FerroError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![n_features],
                context: "number of features must match the training data".into(),
            });
        }
        let n_new = x.nrows();
        let n_train = self.x_train_.nrows();
        let mut out = Array2::<F>::zeros((n_new, self.n_classes_));

        for i in 0..n_new {
            let ri = x.row(i);
            let si = ri.as_slice().unwrap_or(&[]);

            match self.kernel_ {
                LabelSpreadingKernel::Rbf => {
                    // probabilities[i, c] = sum_j rbf(X_train[j], X_query[i]) *
                    //                       label_distributions_[j, c]
                    // (sklearn weight_matrices.T @ label_distributions_, :227-228).
                    for j in 0..n_train {
                        let rj = self.x_train_.row(j);
                        let sj = rj.as_slice().unwrap_or(&[]);
                        let d = sq_euclidean(si, sj);
                        let w = (-self.gamma_ * d).exp();
                        for c in 0..self.n_classes_ {
                            out[[i, c]] = out[[i, c]] + w * self.label_distributions_[[j, c]];
                        }
                    }
                }
                LabelSpreadingKernel::Knn => {
                    // Sum the label distributions of the query's k nearest
                    // training rows (sklearn kneighbors + np.sum, :218-225).
                    let k = self.n_neighbors_.min(n_train);
                    let mut dists: Vec<(usize, F)> = (0..n_train)
                        .map(|j| {
                            let rj = self.x_train_.row(j);
                            let sj = rj.as_slice().unwrap_or(&[]);
                            (j, sq_euclidean(si, sj))
                        })
                        .collect();
                    dists
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    for &(j, _) in dists.iter().take(k) {
                        for c in 0..self.n_classes_ {
                            out[[i, c]] = out[[i, c]] + self.label_distributions_[[j, c]];
                        }
                    }
                }
            }

            // Row-normalize. sklearn divides by the row sum UNCONDITIONALLY
            // (`_label_propagation.py:229-230`, NO `normalizer[normalizer==0]=1`
            // zero-guard — that guard exists only in `fit`). A query far from all
            // training rows underflows every RBF weight to 0, so `row_sum == 0`
            // and sklearn yields `nan` (0.0/0.0) with a RuntimeWarning. We match
            // that bit-for-bit (#2184): divide unconditionally. Float 0/0 = NaN is
            // not a panic, so R-CODE-2 holds.
            let row_sum: F = (0..self.n_classes_).fold(F::zero(), |acc, c| acc + out[[i, c]]);
            for c in 0..self.n_classes_ {
                out[[i, c]] = out[[i, c]] / row_sum;
            }
        }
        Ok(out)
    }

    /// Mean accuracy on the given test data and labels. Mirrors sklearn
    /// `ClassifierMixin.score`. Test samples with `y == -1` are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()` or
    /// the feature count does not match the training data.
    pub fn score(&self, x: &Array2<F>, y: &Array1<isize>) -> Result<F, FerroError> {
        if x.nrows() != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows()],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }
        let preds = self.predict(x)?;
        let mut total = 0usize;
        let mut correct = 0usize;
        for (p, t) in preds.iter().zip(y.iter()) {
            if *t == -1 {
                continue;
            }
            total += 1;
            if p == t {
                correct += 1;
            }
        }
        if total == 0 {
            return Ok(F::zero());
        }
        Ok(F::from(correct).unwrap() / F::from(total).unwrap())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two slices.
#[inline]
fn sq_euclidean<F: Float>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + (ai - bi) * (ai - bi))
}

/// Reject `X` containing any non-finite value (NaN or infinity).
///
/// Mirrors sklearn's `LabelSpreading.fit` (via `BaseLabelPropagation.fit`) →
/// `self._validate_data(X, y, accept_sparse=["csr","csc"], reset=True)`
/// (`sklearn/semi_supervised/_label_propagation.py:258`), which keeps the
/// `force_all_finite=True` default and raises
/// `ValueError("Input X contains NaN.")` / `"... contains infinity ..."`
/// (`sklearn/utils/validation.py:147-154`). Only the FEATURE matrix `X` is
/// finite-checked here; `y` carries integer labels (the `-1` unlabeled
/// sentinel). NaN AND infinity are both rejected. Never panics (R-CODE-2).
fn reject_non_finite<F: Float>(x: &Array2<F>) -> Result<(), FerroError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

/// Build the RBF affinity matrix.
///
/// Mirrors sklearn `rbf_kernel(X, X, gamma)` (`_label_propagation.py:147`):
/// `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`, so the DIAGONAL is `exp(0) = 1`
/// (self-affinity, NOT zeroed). The self-affinity `1` is part of the degree in
/// the normalized graph Laplacian (`csgraph_laplacian(normed=True)`,
/// `_build_graph` `:615-616`).
fn build_rbf_affinity<F: Float>(x: &Array2<F>, gamma: F) -> Vec<F> {
    let n = x.nrows();
    let mut w = vec![F::zero(); n * n];

    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);
        // Diagonal: exp(-gamma * 0) = 1 (sklearn rbf_kernel self-affinity).
        w[i * n + i] = F::one();
        for j in (i + 1)..n {
            let rj = x.row(j);
            let sj = rj.as_slice().unwrap_or(&[]);
            let d = sq_euclidean(si, sj);
            let v = (-gamma * d).exp();
            w[i * n + j] = v;
            w[j * n + i] = v;
        }
    }
    w
}

/// Build the KNN affinity matrix.
fn build_knn_affinity<F: Float>(x: &Array2<F>, k: usize) -> Vec<F> {
    let n = x.nrows();
    let k = k.min(n - 1);
    let mut w = vec![F::zero(); n * n];

    for i in 0..n {
        let ri = x.row(i);
        let si = ri.as_slice().unwrap_or(&[]);

        let mut dists: Vec<(usize, F)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let rj = x.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                (j, sq_euclidean(si, sj))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, _) in dists.iter().take(k) {
            w[i * n + j] = F::one();
            w[j * n + i] = F::one();
        }
    }

    w
}

/// Build the spreading graph matrix, mirroring sklearn `LabelSpreading._build_graph`
/// (`_label_propagation.py:609-623`):
/// `laplacian = csgraph_laplacian(W, normed=True)` = `I - D^{-1/2} W D^{-1/2}`
/// (the SYMMETRIC normalized Laplacian), then `laplacian = -laplacian` (`:617`)
/// and the diagonal is ZEROED (`:622`, `laplacian.flat[::n+1] = 0`).
///
/// `scipy.sparse.csgraph.laplacian` treats `W` as a graph adjacency and IGNORES
/// the diagonal (self-loops) when computing the degree — verified against the live
/// scipy 1.x oracle: `csgraph_laplacian([[5,1,0],[1,5,1],[0,1,5]], return_diag=True)`
/// returns degree `[1, 2, 1]` (the OFF-diagonal row sums), NOT `[6,7,6]`. So even
/// though the RBF `W` diagonal is `1` (self-affinity), it does NOT contribute to
/// the degree: `D[i] = Σ_{j != i} W[i,j]`.
///
/// The returned graph matrix is `S = D^{-1/2} W D^{-1/2}` with its diagonal forced
/// to `0`: `graph[i,j] = W[i,j] / (sqrt(D[i]) * sqrt(D[j]))` for `i != j`,
/// `graph[i,i] = 0`. Verified bit-exact against the live sklearn 1.5.2
/// `LabelSpreading._build_graph` on the `line` fixture (off-diagonal row
/// `[0.415814, 0.315922, 0.193148, …]`).
fn spreading_graph<F: Float>(w: &[F], n: usize) -> Vec<F> {
    // D^{-1/2}, where D[i] = Σ_{j != i} W[i,j] (OFF-diagonal row sum — scipy
    // csgraph_laplacian ignores the diagonal/self-loops).
    let mut d_inv_sqrt = vec![F::zero(); n];
    for i in 0..n {
        let row_sum: F = (0..n).fold(
            F::zero(),
            |acc, j| {
                if i == j { acc } else { acc + w[i * n + j] }
            },
        );
        if row_sum > F::zero() {
            d_inv_sqrt[i] = F::one() / row_sum.sqrt();
        }
    }

    // graph[i,j] = D^{-1/2}[i] * W[i,j] * D^{-1/2}[j] for i != j; diagonal = 0.
    let mut graph = vec![F::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            graph[i * n + j] = d_inv_sqrt[i] * w[i * n + j] * d_inv_sqrt[j];
        }
    }

    graph
}

/// Run the label spreading iterations, mirroring `BaseLabelPropagation.fit`'s loop
/// (`_label_propagation.py:294-330`) for `_variant == "spreading"`.
///
/// Iterates `label_distributions_ = alpha * (graph_matrix @ label_distributions_)
/// + y_static` (`:316-320`) with NO per-iteration row-normalization and NO
/// clamping (spreading lets labeled rows evolve, soft-clamped through
/// `y_static = one-hot * (1 - alpha)`). Convergence is the L1 abs-sum against the
/// PREVIOUS iterate, checked at the START of the loop
/// (`np.abs(label_distributions_ - l_previous).sum() < tol`, `:301`). `n_iter_`
/// is the loop counter at the break — or `max_iter` (the loop `else: n_iter_ += 1`,
/// `:321-326`) if convergence was never reached. A single final row-normalization
/// is applied after the loop (`:328-330`).
///
/// Returns `(label_distributions_, n_iter_)`. `graph_matrix` is the spreading graph
/// (`fn spreading_graph`, row-major `n × n`); `initial_y` is the initial
/// `label_distributions_` (labeled rows one-hot, unlabeled rows zero); `y_static`
/// is the soft-clamp target (`one-hot * (1 - alpha)`, unlabeled zero).
fn spread<F: Float>(
    graph_matrix: &[F],
    initial_y: &Array2<F>,
    y_static: &Array2<F>,
    alpha: F,
    max_iter: usize,
    tol: F,
) -> (Array2<F>, usize) {
    let n = initial_y.nrows();
    let n_classes = initial_y.ncols();

    // self.label_distributions_ (starts at the one-hot/zero init).
    let mut ld = initial_y.clone();
    // l_previous starts at zeros (sklearn `:294`).
    let mut l_previous = Array2::<F>::zeros((n, n_classes));
    let mut buf = Array2::<F>::zeros((n, n_classes));

    let mut n_iter = max_iter;
    let mut converged = false;

    for it in 0..max_iter {
        // Convergence check at loop START: |ld - l_previous|.sum() < tol (L1).
        let mut diff = F::zero();
        for i in 0..n {
            for c in 0..n_classes {
                diff = diff + (ld[[i, c]] - l_previous[[i, c]]).abs();
            }
        }
        if diff < tol {
            n_iter = it;
            converged = true;
            break;
        }

        // l_previous = ld; ld = alpha * (graph_matrix @ ld) + y_static.
        l_previous.assign(&ld);
        for i in 0..n {
            for c in 0..n_classes {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + graph_matrix[i * n + j] * l_previous[[j, c]];
                }
                buf[[i, c]] = alpha * sum + y_static[[i, c]];
            }
        }
        std::mem::swap(&mut ld, &mut buf);
        // NO per-iteration row-normalization, NO clamping (spreading variant).
    }

    // sklearn: `for self.n_iter_ in range(max_iter)` leaves `n_iter_ == max_iter - 1`
    // on a no-break exit, then the loop `else:` does `self.n_iter_ += 1`
    // (`_label_propagation.py:321-326`) → `n_iter_ == max_iter` exactly on
    // non-convergence (#2183, same as LabelPropagation).
    if !converged {
        n_iter = max_iter;
    }

    // Final single row-normalization (sklearn `:328-330`).
    for i in 0..n {
        let row_sum: F = (0..n_classes).fold(F::zero(), |acc, c| acc + ld[[i, c]]);
        let denom = if row_sum == F::zero() {
            F::one()
        } else {
            row_sum
        };
        for c in 0..n_classes {
            ld[[i, c]] = ld[[i, c]] / denom;
        }
    }

    (ld, n_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait impls
// ─────────────────────────────────────────────────────────────────────────────

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<isize>> for LabelSpreading<F> {
    type Fitted = FittedLabelSpreading<F>;
    type Error = FerroError;

    /// Fit the Label Spreading model.
    ///
    /// Labels of `-1` indicate unlabeled points.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `alpha` is not in `(0, 1)`,
    /// `gamma` is not positive (for RBF), or there are no labeled points.
    fn fit(&self, x: &Array2<F>, y: &Array1<isize>) -> Result<FittedLabelSpreading<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedLabelSpreading {
                labels_: Array1::zeros(0),
                label_distributions_: Array2::zeros((0, 0)),
                classes_: Vec::new(),
                n_iter_: 0,
                x_train_: x.clone(),
                kernel_: self.kernel,
                gamma_: self.gamma,
                n_neighbors_: self.n_neighbors,
                n_classes_: 0,
            });
        }

        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y must have the same length as x rows".into(),
            });
        }

        if self.alpha <= F::zero() || self.alpha >= F::one() {
            return Err(FerroError::InvalidParameter {
                name: "alpha".into(),
                reason: "must be in (0, 1)".into(),
            });
        }

        if self.kernel == LabelSpreadingKernel::Rbf && self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive for RBF kernel".into(),
            });
        }

        // Reject non-finite X (NaN AND Inf), mirroring sklearn's
        // `_validate_data(force_all_finite=True)` reached from
        // `LabelSpreading.fit` (`_label_propagation.py:258`), which raises
        // `ValueError` (R-DEV-1, R-CODE-2). `y` is labels, not finite-checked.
        reject_non_finite(x)?;

        if self.kernel == LabelSpreadingKernel::Knn && self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1 for KNN kernel".into(),
            });
        }

        // Identify labeled points.
        let n_labeled = y.iter().filter(|&&l| l >= 0).count();
        if n_labeled == 0 {
            return Err(FerroError::InvalidParameter {
                name: "y".into(),
                reason: "must have at least one labeled sample (label >= 0)".into(),
            });
        }

        // Classes are the sorted unique non-negative labels (sklearn
        // `classes_ = np.unique(y); classes_ = classes_[classes_ != -1]`,
        // `_label_propagation.py:272-274`). `n_classes = len(classes_)`.
        let classes_: Vec<isize> = {
            let mut c: Vec<isize> = y.iter().copied().filter(|&l| l >= 0).collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let n_classes = classes_.len();

        // Build the affinity matrix. The RBF self-affinity diagonal is `1`
        // (`rbf_kernel(X,X)`, `:147`); the KNN graph is built per
        // `build_knn_affinity`.
        let w = match self.kernel {
            LabelSpreadingKernel::Rbf => build_rbf_affinity(x, self.gamma),
            LabelSpreadingKernel::Knn => build_knn_affinity(x, self.n_neighbors),
        };

        // Build the spreading graph matrix = `D^{-1/2} W D^{-1/2}` (degree `D`
        // INCLUDES the diagonal self-affinity) with the diagonal forced to `0`
        // (sklearn `_build_graph`, `csgraph_laplacian(normed=True)` then
        // `-laplacian` then zero diagonal, `:609-623`).
        let graph = spreading_graph(&w, n_samples);

        // Build the initial `label_distributions_`: zeros, then one-hot for
        // labeled rows (sklearn `:282-284`). Unlabeled rows START AT ZERO (NOT
        // uniform — the spreading iteration injects mass through `y_static`).
        let mut initial_y = Array2::from_elem((n_samples, n_classes), F::zero());
        for (i, &label) in y.iter().enumerate() {
            if label >= 0 {
                // Index by the POSITION of the label in `classes_`, not its raw
                // value (sklearn one-hots `classes_ == label`, `:283-284`).
                let c = classes_.iter().position(|&v| v == label).unwrap_or(0);
                if c < n_classes {
                    initial_y[[i, c]] = F::one();
                }
            }
            // Unlabeled rows remain zero (sklearn `:282`).
        }

        // `y_static` for the SPREADING variant = the one-hot init scaled by
        // `(1 - alpha)` (sklearn `y_static *= 1 - self.alpha`, `:290-292`).
        // Unlabeled rows are zero (they were zero in `initial_y`), so they stay
        // zero after scaling.
        let one_minus_alpha = F::one() - self.alpha;
        let mut y_static = initial_y.clone();
        y_static.mapv_inplace(|v| v * one_minus_alpha);

        // Run spreading (soft-clamp update `alpha*graph@F + y_static`, NO
        // per-iteration normalization, L1-at-start convergence, final
        // row-normalize). Returns (label_distributions_, n_iter_).
        let (label_distributions, n_iter) = spread(
            &graph,
            &initial_y,
            &y_static,
            self.alpha,
            self.max_iter,
            self.tol,
        );

        // Extract final labels (argmax of each row).
        let labels: Array1<isize> = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let mut best_c = 0;
                    let mut best_v = label_distributions[[i, 0]];
                    for c in 1..n_classes {
                        if label_distributions[[i, c]] > best_v {
                            best_v = label_distributions[[i, c]];
                            best_c = c;
                        }
                    }
                    // Map the argmax column INDEX through `classes_`
                    // (sklearn `transduction_ = classes_[argmax(...)]`, `:333`).
                    classes_.get(best_c).copied().unwrap_or(0)
                })
                .collect(),
        );

        Ok(FittedLabelSpreading {
            labels_: labels,
            label_distributions_: label_distributions,
            classes_,
            n_iter_: n_iter,
            x_train_: x.clone(),
            kernel_: self.kernel,
            gamma_: self.gamma,
            n_neighbors_: self.n_neighbors,
            n_classes_: n_classes,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLabelSpreading<F> {
    type Output = Array1<isize>;
    type Error = FerroError;

    /// Predict labels for new data by inductive inference, mirroring sklearn
    /// `BaseLabelPropagation.predict` (`_label_propagation.py:173-191`):
    /// `classes_[argmax(predict_proba(X), axis=1)]` — the per-row argmax of the
    /// kernel-weighted probabilities, mapped THROUGH `classes_`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the feature count does not match.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<isize>, FerroError> {
        let probas = self.predict_proba(x)?;
        let n_new = probas.nrows();
        let mut labels = Array1::zeros(n_new);

        for i in 0..n_new {
            // argmax over classes; ties resolve to the first (lowest) index,
            // matching numpy `np.argmax`.
            let mut best_c = 0;
            let mut best_v = if self.n_classes_ > 0 {
                probas[[i, 0]]
            } else {
                F::zero()
            };
            for c in 1..self.n_classes_ {
                if probas[[i, c]] > best_v {
                    best_v = probas[[i, c]];
                    best_c = c;
                }
            }
            // Map the argmax column index through `classes_`
            // (sklearn `classes_[argmax(...)]`, `:191`).
            labels[i] = self.classes_.get(best_c).copied().unwrap_or(0);
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two blobs with partially labeled data.
    fn make_semi_supervised() -> (Array2<f64>, Array1<isize>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1,
                10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);
        (x, y)
    }

    #[test]
    fn test_label_spreading_basic() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // Points near (0,0) should get label 0.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[3], 0);

        // Points near (10,10) should get label 1.
        assert_eq!(labels[4], 1);
        assert_eq!(labels[5], 1);
        assert_eq!(labels[6], 1);
        assert_eq!(labels[7], 1);
    }

    #[test]
    fn test_alpha_zero_rejected() {
        // sklearn `_parameter_constraints["alpha"] = [Interval(Real, 0, 1,
        // closed="neither")]` (`_label_propagation.py:585`) — the OPEN interval
        // (0, 1) rejects alpha=0. Mirror that: fit with alpha=0 must be Err.
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0])
            .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let model = LabelSpreading::<f64>::new().with_alpha(0.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_convergence() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new().with_max_iter(1000);
        let fitted = model.fit(&x, &y).unwrap();

        // Should have converged with reasonable results.
        assert_eq!(fitted.labels().len(), 8);

        // Label distributions should sum to ~1 for each sample.
        let dists = fitted.label_distributions();
        for i in 0..8 {
            let row_sum: f64 = (0..dists.ncols()).map(|c| dists[[i, c]]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_knn_kernel() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new()
            .with_kernel(LabelSpreadingKernel::Knn)
            .with_n_neighbors(3);
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);
    }

    #[test]
    fn test_predict_on_new_data() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        assert_eq!(new_labels[0], 0);
        assert_eq!(new_labels[1], 1);
    }

    #[test]
    fn test_invalid_alpha() {
        let (x, y) = make_semi_supervised();

        // alpha >= 1.0 is invalid.
        let model = LabelSpreading::<f64>::new().with_alpha(1.0);
        assert!(model.fit(&x, &y).is_err());

        // alpha < 0 is invalid.
        let model = LabelSpreading::<f64>::new().with_alpha(-0.1);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_no_labeled_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![-1, -1, -1, -1]);

        let model = LabelSpreading::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_label_distributions_shape() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let dist = fitted.label_distributions();
        assert_eq!(dist.nrows(), 8);
        assert_eq!(dist.ncols(), 2);
    }

    #[test]
    fn test_n_classes() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let (x, y) = make_semi_supervised();
        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let bad_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(fitted.predict(&bad_x).is_err());
    }

    #[test]
    fn test_y_length_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelSpreading::<f64>::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<isize>::zeros(0);

        let model = LabelSpreading::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.labels().len(), 0);
    }

    #[test]
    fn test_f32_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);

        let model = LabelSpreading::<f32>::new();
        let fitted = model.fit(&x, &y).unwrap();
        assert_eq!(fitted.labels().len(), 6);
    }

    #[test]
    fn test_three_classes() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 0.0, 10.0, 0.1,
                10.0, 0.0, 10.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1, 2, -1, -1]);

        let model = LabelSpreading::<f64>::new().with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[3], 1);
        assert_eq!(fitted.labels()[6], 2);
    }

    #[test]
    fn test_default_constructor() {
        let model = LabelSpreading::<f64>::default();
        assert_eq!(model.kernel, LabelSpreadingKernel::Rbf);
        assert!(model.gamma > 0.0);
        assert_eq!(model.n_neighbors, 7);
        assert_relative_eq!(model.alpha, 0.2);
    }

    #[test]
    fn test_invalid_gamma() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelSpreading::<f64>::new().with_gamma(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_alpha_affects_results() {
        let (x, y) = make_semi_supervised();

        // With alpha close to 0, results should be close to initial labels.
        let model_low = LabelSpreading::<f64>::new().with_alpha(0.01);
        let fitted_low = model_low.fit(&x, &y).unwrap();

        // With alpha close to 1, results should be more influenced by graph.
        let model_high = LabelSpreading::<f64>::new().with_alpha(0.99);
        let fitted_high = model_high.fit(&x, &y).unwrap();

        // Both should produce the same labels for well-separated clusters.
        assert_eq!(fitted_low.labels()[0], 0);
        assert_eq!(fitted_high.labels()[0], 0);
        assert_eq!(fitted_low.labels()[4], 1);
        assert_eq!(fitted_high.labels()[4], 1);
    }
}
