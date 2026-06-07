//! Label Propagation for semi-supervised classification.
//!
//! This module provides [`LabelPropagation`], a graph-based semi-supervised
//! learning algorithm that propagates known labels through a similarity graph
//! to classify unlabeled data points.
//!
//! # Algorithm
//!
//! Mirrors `sklearn.semi_supervised.LabelPropagation`
//! (`_label_propagation.py`, `_variant="propagation"`):
//!
//! 1. Build the affinity matrix `W` (RBF: `W[i,j] = exp(-gamma ||x_i-x_j||^2)`,
//!    self-affinity diagonal `= 1`; or KNN connectivity).
//! 2. Build the graph matrix by COLUMN-sum normalization:
//!    `graph[i,j] = W[i,j] / Σ_k W[k,i]` (sklearn `_build_graph`).
//! 3. Initialize `label_distributions_` to a one-hot over `classes_` for
//!    labeled rows; **unlabeled rows start at zero**.
//! 4. Iterate `label_distributions_ = graph @ label_distributions_`,
//!    row-normalize, then **clamp** labeled rows back to their one-hot
//!    (`y_static`).
//! 5. Convergence: the L1 abs-sum `|label_distributions_ - l_previous|.sum() <
//!    tol`, checked at the START of each iteration against the previous
//!    iterate; `n_iter_` is the loop counter, set to `max_iter` exactly when
//!    `max_iter` is reached without convergence (sklearn convention). A final row-normalization is
//!    applied.
//!
//! Inductive `predict_proba(X)` is the kernel-weighted combination over all
//! training rows (`rbf_kernel(X_train, X).T @ label_distributions_`,
//! row-normalized); `predict(X) = classes_[argmax(predict_proba(X))]`.
//!
//! Labels of `-1` in the target vector indicate unlabeled points.
//!
//! # Examples
//!
//! ```
//! use ferrolearn_cluster::LabelPropagation;
//! use ferrolearn_core::Fit;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     0.0, 0.0,  0.1, 0.0,  0.0, 0.1,
//!     5.0, 5.0,  5.1, 5.0,  5.0, 5.1,
//! ]).unwrap();
//! // First and fourth points are labeled; rest are unlabeled (-1).
//! let y = Array1::from_vec(vec![0, -1, -1, 1, -1, -1]);
//!
//! let model = LabelPropagation::<f64>::new();
//! let fitted = model.fit(&x, &y).unwrap();
//! assert_eq!(fitted.labels().len(), 6);
//! ```
//!
//! # `## REQ status`
//!
//! Binary (R-DEFER-2), translating `sklearn/semi_supervised/_label_propagation.py`
//! (`class LabelPropagation(BaseLabelPropagation)` `:338`, `_variant="propagation"`;
//! base `fit`/`predict`/`predict_proba` `:233-335`). Design doc:
//! `.design/cluster/label_propagation.md`. Cites use ferrolearn symbol anchors /
//! sklearn `file:line` (commit 156ef14); expected values from the live sklearn 1.5.2
//! oracle (R-CHAR-3). The contiguous-label transduction PARTITION (REQ-1) and the
//! `classes_`/`n_classes`/label-VALUE mapping (REQ-4) ship through the crate re-export.
//! The `label_distributions_` VALUES, `n_iter_`, the `tol` default, and
//! `predict`/`predict_proba` now MATCH sklearn bit-exactly: `fn build_rbf_affinity`
//! keeps the self-affinity diagonal `=1` (`rbf_kernel`, `:147`), `fn normalize_graph`
//! does column-sum normalization (`:457-461`), `fn fit` zero-inits unlabeled rows
//! (`:282`), `fn propagate` converges on the L1-at-start rule tracking `n_iter_`
//! (`:300-326`), `fn new` defaults `tol = 1e-3` (`:435`), and `fn predict_proba` is the
//! kernel-weighted combination over all training rows (`:218-231`) — REQ-2/3/5/6 SHIPPED
//! (closing #997/#998/#1000/#1001). There is no CPython binding (#1006).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (contiguous-label transduction PARTITION) | SHIPPED | impl `fn fit` (graph build → `fn normalize_graph` → `fn propagate` → per-row argmax) recovers sklearn's transduction on well-separated CONTIGUOUS-label data. Consumer: crate re-export `pub use label_propagation::{FittedLabelPropagation, LabelPropagation, LabelPropagationKernel}` (`lib.rs:110`). Guards: `green_guard_req1_contiguous_partition_2blob`, `green_guard_req1_contiguous_partition_3blob` in `tests/divergence_label_propagation.rs` (live-oracle). The `label_distributions_` VALUES now match too (REQ-2/3). |
//! | REQ-4 (`classes_` / `n_classes` / label-VALUE mapping) | SHIPPED | impl `fn fit` now builds `classes_` = sorted unique non-(-1) labels, `n_classes = classes_.len()`, one-hot indexed by class POSITION, and maps the final argmax index through `classes_` — matching sklearn `classes_ = unique(y)\{-1}` + `transduction_ = classes_[argmax]` (`_label_propagation.py:272-274,333`). Guard: `divergence_req4_noncontiguous_classes_mapping` (`{0,2}`-label fixture: ferrolearn now `n_classes()==2`, `labels ⊆ {0,2}` matching sklearn `[0,0,0,0,2,2,2,2]`; was `n_classes()==3` with a phantom class). Fixed #999. |
//! | REQ-2 (`label_distributions_` value — RBF diagonal + zero-init) | SHIPPED | closes #997. `fn build_rbf_affinity` sets the self-affinity diagonal `= 1` (`exp(0)`, `rbf_kernel(X,X)`, `:147`); `fn normalize_graph` does column-sum normalization (`:457-461`); `fn fit` zero-inits unlabeled rows of `initial_y` and `y_static` (`:282`). Consumer: crate re-export (`lib.rs:110`). Guard: `divergence_req2_3_label_distributions_line` (`tests/divergence_label_propagation.rs`) asserts the live-oracle `line` rows `[[1,0],[0.55810978,0.44189022],[0.49024013,0.50975987],[0,1]]` to 1e-6. |
//! | REQ-3 (convergence — L1-at-start) | SHIPPED | closes #998. `fn propagate` checks `\|label_distributions_ - l_previous\|.sum() < tol` at the loop START (L1, against the previous iterate, `:301`), tracks `n_iter_` (the loop counter; `+1` on `max_iter`, `:321-326`), and applies the final row-normalization (`:328-330`). Guard: `divergence_req2_3_label_distributions_line` asserts the live-oracle `n_iter_ == 4` on the `line` fixture. |
//! | REQ-5 (`tol` default `1e-3`) | SHIPPED | closes #1000. `fn new` sets `tol = F::from(1e-3)` matching sklearn `LabelPropagation` `tol=1e-3` (`:435`). Consumer: crate re-export (`lib.rs:110`). Guard: `divergence_req5_tol_default`. |
//! | REQ-6 (`predict`/`predict_proba` kernel-weighted) | SHIPPED | closes #1001. `fn predict_proba` = `rbf_kernel(X_train,X).T @ label_distributions_` row-normalized (`:218-231`); `fn predict` = `classes_[argmax(predict_proba)]` (`:190-191`). Guard: `divergence_req6_predict_proba_kernel_weighted` asserts live-oracle `predict_proba` rows (1e-6, sum-to-1) + `predict`. R-DEV-3. |
//! | REQ-7 (`transduction_`/`classes_`/`n_iter_`/`X_` attrs) | NOT-STARTED | open prereq blocker #1002. `FittedLabelPropagation` now exposes `fn classes` (`classes_`) and `fn n_iter` (`n_iter_`) accessors, but the labels are still named `fn labels` (not `transduction_`) and there is no `X_` / `n_features_in_` accessor — the full sklearn attribute surface (`:264,274,300,333`) is not yet mirrored. |
//! | REQ-8 (`ConvergenceWarning`) | NOT-STARTED | open prereq blocker #1003. `fn propagate` now tracks `n_iter_` and increments it by one when `max_iter` is reached without convergence (matching sklearn's `n_iter_ += 1`, `:321-326`), but it does NOT emit a `ConvergenceWarning` (no warning channel) — so the warning half of REQ-8 is unimplemented. |
//! | REQ-9 (KNN connectivity graph) | NOT-STARTED | open prereq blocker #1004. `fn normalize_graph` now does the sklearn COLUMN-sum normalization (`:457-461`), but `fn build_knn_affinity` still SYMMETRIZES (`w[i,j]=w[j,i]=1`) instead of building the DIRECTED `kneighbors_graph(mode="connectivity")` (`:156-157`) — the KNN graph topology still diverges (the RBF path now matches exactly). |
//! | REQ-10 (validation / error ABI) | NOT-STARTED | open prereq blocker #1005. sklearn `check_classification_targets` + `_parameter_constraints` (`gamma∈[0,∞)`, etc.) raising `InvalidParameterError` (`:110-118,265`); ferrolearn `fn fit` raises `FerroError::InvalidParameter` (different type/ABI) and rejects `gamma>0` (stricter than sklearn's `[0,∞)`). |
//! | REQ-11 (PyO3 binding) | NOT-STARTED | open prereq blocker #1006. `grep LabelPropagation ferrolearn-python/` is EMPTY — no binding; `import ferrolearn` cannot reach `LabelPropagation`. Only consumer is the crate re-export. |
//! | REQ-12 (ferray substrate) | NOT-STARTED | open prereq blocker #1007. `label_propagation.rs` imports `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`/`ferray::linalg` (R-SUBSTRATE-1/2). |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// The kernel used to build the affinity matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPropagationKernel {
    /// RBF (Gaussian) kernel: `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`.
    Rbf,
    /// KNN kernel: `W[i,j] = 1` if j is among the k nearest neighbors of i
    /// (or vice versa), `0` otherwise.
    Knn,
}

/// Label Propagation semi-supervised classifier (unfitted).
///
/// Holds hyperparameters. Call [`Fit::fit`] to run the algorithm and produce
/// a [`FittedLabelPropagation`].
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
#[derive(Debug, Clone)]
pub struct LabelPropagation<F> {
    /// The kernel to use for building the affinity matrix.
    pub kernel: LabelPropagationKernel,
    /// Gamma parameter for the RBF kernel.
    pub gamma: F,
    /// Number of neighbors for the KNN kernel.
    pub n_neighbors: usize,
    /// Maximum number of propagation iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: F,
}

impl<F: Float> LabelPropagation<F> {
    /// Create a new `LabelPropagation` with default parameters.
    ///
    /// Defaults: `kernel = Rbf`, `gamma = 20.0`, `n_neighbors = 7`,
    /// `max_iter = 1000`, `tol = 1e-3` — matching
    /// `sklearn.semi_supervised.LabelPropagation.__init__`
    /// (`_label_propagation.py:428-446`, `tol=1e-3` `:435`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernel: LabelPropagationKernel::Rbf,
            gamma: F::from(20.0).unwrap_or_else(F::one),
            n_neighbors: 7,
            max_iter: 1000,
            tol: F::from(1e-3).unwrap_or_else(F::epsilon),
        }
    }

    /// Set the kernel type.
    #[must_use]
    pub fn with_kernel(mut self, kernel: LabelPropagationKernel) -> Self {
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
}

impl<F: Float> Default for LabelPropagation<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fitted Label Propagation model.
///
/// Stores the final transduction labels, the (row-normalized) label
/// distribution matrix, the distinct training classes, the number of
/// iterations run, and the training data + kernel needed for inductive
/// inference. Implements [`Predict`] via the sklearn kernel-weighted
/// combination over ALL training rows (`_label_propagation.py:190-231`).
#[derive(Debug, Clone)]
pub struct FittedLabelPropagation<F> {
    /// Transduction labels for each training sample (`transduction_`).
    labels_: Array1<isize>,
    /// Label distribution matrix, shape `(n_samples, n_classes)`.
    label_distributions_: Array2<F>,
    /// The distinct labels used during fit (`classes_`), sorted ascending,
    /// excluding the `-1` unlabeled sentinel. Column `c` of
    /// `label_distributions_` corresponds to `classes_[c]`.
    classes_: Vec<isize>,
    /// Number of iterations the propagation loop ran (`n_iter_`).
    n_iter_: usize,
    /// Training data, stored for inductive `predict` / `predict_proba`.
    x_train_: Array2<F>,
    /// The kernel used to build the affinity matrix (for inductive inference).
    kernel_: LabelPropagationKernel,
    /// Gamma parameter for the RBF kernel (for inductive inference).
    gamma_: F,
    /// Number of neighbors for the KNN kernel (for inductive inference).
    n_neighbors_: usize,
    /// Number of classes.
    n_classes_: usize,
}

impl<F: Float> FittedLabelPropagation<F> {
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

    /// Return the number of iterations the propagation loop ran (`n_iter_`).
    ///
    /// Mirrors sklearn `n_iter_` — the L1-at-start convergence loop counter
    /// (`_label_propagation.py:300-326`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }
}

impl<F: Float + Send + Sync + 'static> FittedLabelPropagation<F> {
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
                LabelPropagationKernel::Rbf => {
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
                LabelPropagationKernel::Knn => {
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

            // Row-normalize; sklearn divides by the row sum (:229-230). A
            // zero row sum cannot occur (label_distributions_ rows sum to 1
            // and the weights are non-negative), but guard against it anyway
            // (R-CODE-2: never panic / divide-by-zero).
            let row_sum: F = (0..self.n_classes_).fold(F::zero(), |acc, c| acc + out[[i, c]]);
            if row_sum > F::zero() {
                for c in 0..self.n_classes_ {
                    out[[i, c]] = out[[i, c]] / row_sum;
                }
            }
        }
        Ok(out)
    }

    /// Mean accuracy on the given test data and labels. Mirrors sklearn
    /// `ClassifierMixin.score`. Test samples with `y == -1` (unlabeled
    /// sentinel) are skipped from the denominator.
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

/// Build the RBF affinity matrix.
///
/// Mirrors sklearn `rbf_kernel(X, X, gamma)` (`_label_propagation.py:147`):
/// `W[i,j] = exp(-gamma * ||x_i - x_j||^2)`, so the DIAGONAL is `exp(0) = 1`
/// (self-affinity, NOT zeroed). Verified against the live oracle
/// `sklearn.metrics.pairwise.rbf_kernel` whose `np.diag(K) == [1, 1, …]`.
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

        // Compute distances from i to all other points.
        let mut dists: Vec<(usize, F)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let rj = x.row(j);
                let sj = rj.as_slice().unwrap_or(&[]);
                (j, sq_euclidean(si, sj))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Set the k nearest neighbors to 1.
        for &(j, _) in dists.iter().take(k) {
            w[i * n + j] = F::one();
            w[j * n + i] = F::one(); // Symmetrize.
        }
    }

    w
}

/// Normalize the affinity matrix into the propagation graph matrix, mirroring
/// sklearn `LabelPropagation._build_graph` (`_label_propagation.py:457-461`):
/// `normalizer = affinity.sum(axis=0)` (COLUMN sums), then
/// `affinity /= normalizer[:, np.newaxis]` — i.e. row `i` is divided by the
/// `i`-th column sum, giving `graph[i,j] = affinity[i,j] / colsum(i)` where
/// `colsum(i) = Σ_k affinity[k,i]`.
///
/// For the SYMMETRIC RBF graph the `i`-th column sum equals the `i`-th row sum,
/// so this coincides with row-normalization; the column form is the literal
/// translation and is also correct for the directed KNN graph.
fn normalize_graph<F: Float>(w: &mut [F], n: usize) {
    // Column sums: col_sum[i] = Σ_k w[k, i].
    let mut col_sum = vec![F::zero(); n];
    for k in 0..n {
        for i in 0..n {
            col_sum[i] = col_sum[i] + w[k * n + i];
        }
    }
    for i in 0..n {
        // Divide row i by col_sum[i]. Guard against a zero column sum (cannot
        // occur for RBF — diagonal is 1 — but possible for a degenerate KNN
        // graph; R-CODE-2: never divide by zero).
        if col_sum[i] > F::zero() {
            for j in 0..n {
                w[i * n + j] = w[i * n + j] / col_sum[i];
            }
        }
    }
}

/// Run the label propagation iterations, mirroring `BaseLabelPropagation.fit`'s
/// loop (`_label_propagation.py:294-330`) for `_variant == "propagation"`.
///
/// Iterates `label_distributions_ = graph_matrix @ label_distributions_`, then
/// row-normalizes (zero row sums → `1`, `:310-312`), then re-clamps labeled
/// rows to `y_static` (`:313-315`). Convergence is the L1 abs-sum against the
/// PREVIOUS iterate, checked at the START of the loop
/// (`np.abs(label_distributions_ - l_previous).sum() < tol`, `:301`). `n_iter_`
/// is the loop counter at the point of the break — or `max_iter` (then `+= 1`)
/// if convergence was never reached (`:321-326`). A final row-normalization is
/// applied after the loop (`:328-330`).
///
/// Returns `(label_distributions_, n_iter_)`. `graph_matrix` is the
/// column-normalized affinity (row-major `n × n`); `initial_y` is the initial
/// `label_distributions_` (labeled rows one-hot, unlabeled rows zero);
/// `y_static` is the clamp target (labeled one-hot, unlabeled zero).
fn propagate<F: Float>(
    graph_matrix: &[F],
    initial_y: &Array2<F>,
    y_static: &Array2<F>,
    unlabeled_mask: &[bool],
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

        // l_previous = ld; ld = graph_matrix @ ld.
        l_previous.assign(&ld);
        for i in 0..n {
            for c in 0..n_classes {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + graph_matrix[i * n + j] * l_previous[[j, c]];
                }
                buf[[i, c]] = sum;
            }
        }
        std::mem::swap(&mut ld, &mut buf);

        // Row-normalize (zero row sums -> 1), then clamp labeled rows to
        // y_static (sklearn `:310-315`).
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
            if !unlabeled_mask[i] {
                for c in 0..n_classes {
                    ld[[i, c]] = y_static[[i, c]];
                }
            }
        }
    }

    // sklearn: `for self.n_iter_ in range(max_iter)` leaves `n_iter_ == max_iter - 1`
    // on a no-break exit, then the loop `else:` does `self.n_iter_ += 1`
    // (`_label_propagation.py:321-326`) → `n_iter_ == max_iter` exactly on
    // non-convergence (#2183).
    if !converged {
        n_iter = max_iter;
    }

    // Final row-normalization (sklearn `:328-330`).
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

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<isize>> for LabelPropagation<F> {
    type Fitted = FittedLabelPropagation<F>;
    type Error = FerroError;

    /// Fit the Label Propagation model.
    ///
    /// Labels of `-1` indicate unlabeled points.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `gamma` is not positive
    /// (for RBF kernel) or if there are no labeled points.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<isize>,
    ) -> Result<FittedLabelPropagation<F>, FerroError> {
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Ok(FittedLabelPropagation {
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

        if self.kernel == LabelPropagationKernel::Rbf && self.gamma <= F::zero() {
            return Err(FerroError::InvalidParameter {
                name: "gamma".into(),
                reason: "must be positive for RBF kernel".into(),
            });
        }

        if self.kernel == LabelPropagationKernel::Knn && self.n_neighbors == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_neighbors".into(),
                reason: "must be at least 1 for KNN kernel".into(),
            });
        }

        // Identify labeled and unlabeled points.
        let labeled_mask: Vec<bool> = y.iter().map(|&l| l >= 0).collect();
        let n_labeled = labeled_mask.iter().filter(|&&m| m).count();

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

        // Unlabeled mask (label == -1), the complement of `labeled_mask`.
        let unlabeled_mask: Vec<bool> = labeled_mask.iter().map(|&m| !m).collect();

        // Build the affinity matrix. The RBF self-affinity diagonal is `1`
        // (`rbf_kernel(X,X)`, `:147`); the KNN graph is built per
        // `build_knn_affinity`.
        let mut w = match self.kernel {
            LabelPropagationKernel::Rbf => build_rbf_affinity(x, self.gamma),
            LabelPropagationKernel::Knn => build_knn_affinity(x, self.n_neighbors),
        };

        // Column-sum normalize into the propagation graph matrix
        // (sklearn `_build_graph`, `:457-461`).
        normalize_graph(&mut w, n_samples);

        // Build the initial `label_distributions_`: zeros, then one-hot for
        // labeled rows (sklearn `:282-284`). Unlabeled rows START AT ZERO.
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
            // Unlabeled rows remain zero (sklearn `:282`, `y_static[unlabeled]=0`).
        }

        // `y_static` for the propagation variant = the labeled one-hot with
        // unlabeled rows zeroed (sklearn `:286-289`). Since unlabeled rows of
        // `initial_y` are already zero, `y_static == initial_y` here.
        let y_static = initial_y.clone();

        // Run propagation (L1-at-start convergence, clamp labeled rows to
        // `y_static`, final row-normalize). Returns (label_distributions_, n_iter_).
        let (label_distributions, n_iter) = propagate(
            &w,
            &initial_y,
            &y_static,
            &unlabeled_mask,
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

        Ok(FittedLabelPropagation {
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

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedLabelPropagation<F> {
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
        // Label first and fifth points; rest unlabeled.
        let y = Array1::from_vec(vec![0, -1, -1, -1, 1, -1, -1, -1]);
        (x, y)
    }

    #[test]
    fn test_label_propagation_basic() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        assert_eq!(labels.len(), 8);

        // Labeled points should keep their labels.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);

        // Points near (0,0) should get label 0.
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 0);
        assert_eq!(labels[3], 0);

        // Points near (10,10) should get label 1.
        assert_eq!(labels[5], 1);
        assert_eq!(labels[6], 1);
        assert_eq!(labels[7], 1);
    }

    #[test]
    fn test_knn_kernel() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new()
            .with_kernel(LabelPropagationKernel::Knn)
            .with_n_neighbors(3);
        let fitted = model.fit(&x, &y).unwrap();

        let labels = fitted.labels();
        // Same expected behavior.
        assert_eq!(labels[0], 0);
        assert_eq!(labels[4], 1);
    }

    #[test]
    fn test_predict_on_new_data() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
        let new_labels = fitted.predict(&new_x).unwrap();

        assert_eq!(new_labels[0], 0);
        assert_eq!(new_labels[1], 1);
    }

    #[test]
    fn test_all_labeled() {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0])
            .unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        // All labels should be preserved since all are labeled.
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[1], 0);
        assert_eq!(fitted.labels()[2], 1);
        assert_eq!(fitted.labels()[3], 1);
    }

    #[test]
    fn test_no_labeled_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![-1, -1, -1, -1]);

        let model = LabelPropagation::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_distributions_shape() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let dist = fitted.label_distributions();
        assert_eq!(dist.nrows(), 8);
        assert_eq!(dist.ncols(), 2); // 2 classes.
    }

    #[test]
    fn test_n_classes() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 2);
    }

    #[test]
    fn test_predict_shape_mismatch() {
        let (x, y) = make_semi_supervised();
        let model = LabelPropagation::<f64>::new();
        let fitted = model.fit(&x, &y).unwrap();

        let bad_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = fitted.predict(&bad_x);
        assert!(result.is_err());
    }

    #[test]
    fn test_y_length_mismatch() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]); // Wrong length.

        let model = LabelPropagation::<f64>::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<isize>::zeros(0);

        let model = LabelPropagation::<f64>::new();
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

        let model = LabelPropagation::<f32>::new();
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

        let model = LabelPropagation::<f64>::new().with_gamma(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.labels()[0], 0);
        assert_eq!(fitted.labels()[3], 1);
        assert_eq!(fitted.labels()[6], 2);
    }

    #[test]
    fn test_default_constructor() {
        let model = LabelPropagation::<f64>::default();
        assert_eq!(model.kernel, LabelPropagationKernel::Rbf);
        assert!(model.gamma > 0.0);
        assert_eq!(model.n_neighbors, 7);
    }

    #[test]
    fn test_invalid_gamma() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, -1]);

        let model = LabelPropagation::<f64>::new().with_gamma(-1.0);
        assert!(model.fit(&x, &y).is_err());
    }
}
