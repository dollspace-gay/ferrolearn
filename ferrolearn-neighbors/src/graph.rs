//! Neighbor graph constructors.
//!
//! Mirrors sklearn's free functions [`kneighbors_graph`] and
//! [`radius_neighbors_graph`], plus the [`sort_graph_by_row_values`]
//! utility. Returns CSR sparse matrices via [`ferrolearn_sparse::CsrMatrix`].
//!
//! For the method form, see e.g.
//! [`crate::FittedKNeighborsClassifier::kneighbors_graph`] which delegate
//! to these free functions after a `kneighbors` query against the fitted
//! data.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.neighbors` graph functions (`sklearn/neighbors/_graph.py`).
//! See `.design/neighbors/graph.md`. Non-test consumer: crate re-export +
//! the `Fitted*::kneighbors_graph` method forms.
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | `kneighbors_graph` value: `include_self=False` default (zero diagonal, `n_neighbors` non-self edges via query+1/drop-self, `_graph.py:34-43`) + CSR | SHIPPED |
//! | REQ-2 | `radius_neighbors_graph` value: self-edge dropped (zero diagonal, `_graph.py:255`) | SHIPPED |
//! | REQ-3 | `GraphMode`/`mode` value mapping: connectivity → 1.0, distance → actual (`_graph.py`) | SHIPPED |
//! | REQ-4 | CSR output contract: `(n×n)` shape, ascending column indices per row | SHIPPED |
//! | REQ-5 | `sort_graph_by_row_values` ascending-stored-value sort (`_base.py:271-289`) | NOT-STARTED (#825, blocked on ferrolearn-sparse `CsrMatrix::new_unchecked` #826) |
//! | REQ-6 | `include_self=True` toggle + `metric`/`p`/`metric_params`/`n_jobs` params | NOT-STARTED (#827) |
//! | REQ-7 | `KNeighborsTransformer` + `RadiusNeighborsTransformer` estimators | NOT-STARTED (#828) |
//! | REQ-8 | PyO3 binding + ferray sparse substrate | NOT-STARTED (#829) |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Fit;
use ferrolearn_sparse::CsrMatrix;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::knn::Algorithm;
use crate::nearest_neighbors::NearestNeighbors;
use crate::{FittedNearestNeighbors, RadiusNeighborsClassifier};

/// What value to store at every (i, j) cell of the neighbor graph.
///
/// Mirrors sklearn's `mode` parameter on `kneighbors_graph` and
/// `radius_neighbors_graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    /// Store `1.0` for every neighbor edge.
    Connectivity,
    /// Store the actual distance for every neighbor edge.
    Distance,
}

/// Compute the (weighted) graph of k-Neighbors for points in `x`.
///
/// Equivalent to sklearn `sklearn.neighbors.kneighbors_graph` (with
/// `metric="minkowski"`, `p=2`). Builds a transient `NearestNeighbors`
/// index over `x`, then for every row queries its `n_neighbors` nearest
/// neighbors in `x` itself.
///
/// Output shape is `(n_samples, n_samples)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] / [`FerroError::InvalidParameter`]
/// per the underlying neighbor query.
pub fn kneighbors_graph<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    n_neighbors: usize,
    mode: GraphMode,
) -> Result<CsrMatrix<F>, FerroError> {
    // sklearn's `kneighbors_graph` defaults `include_self=False`, routing the
    // query through `_query_include_self` (`sklearn/neighbors/_graph.py:34-43`:
    // "if not include_self: X = None") so each sample's OWN row is excluded —
    // the graph has a zero diagonal and reports `n_neighbors` NON-self
    // neighbors per row. We emulate that `X=None` semantics by querying
    // `n_neighbors + 1` neighbors and dropping each row's self entry. When
    // `n_neighbors >= n_samples` sklearn errors; querying `n_neighbors + 1`
    // preserves that (the underlying `kneighbors` rejects `k > n_train`).
    let query_k = n_neighbors + 1;
    let nn = NearestNeighbors::<F>::new()
        .with_n_neighbors(query_k)
        .with_algorithm(Algorithm::Auto)
        .fit(x, &())?;
    let (distances, indices) = nn.kneighbors(x, Some(query_k))?;
    let (distances, indices) = drop_self_neighbors(&distances, &indices, n_neighbors);
    knn_to_csr(&distances, &indices, x.nrows(), x.nrows(), mode)
}

/// Drop each query row's self entry from `(distances, indices)` results.
///
/// Mirrors sklearn's `include_self=False` (`X=None`) query semantics: the
/// query is the fitted data itself, so each row `i` contains its own index as
/// a 0-distance neighbor. We remove the FIRST occurrence of `index == i`; if no
/// self entry is present (only possible with exact-duplicate points, where the
/// self-match may not sort first), we drop the last/farthest entry so the
/// output stays exactly `n_neighbors` wide. Inputs are `(n_rows, n_neighbors+1)`
/// arrays; the output is `(n_rows, n_neighbors)`.
fn drop_self_neighbors<F: Float>(
    distances: &Array2<F>,
    indices: &Array2<usize>,
    n_neighbors: usize,
) -> (Array2<F>, Array2<usize>) {
    let n_rows = indices.nrows();
    let query_k = indices.ncols();
    let mut out_dist = Array2::<F>::zeros((n_rows, n_neighbors));
    let mut out_idx = Array2::<usize>::zeros((n_rows, n_neighbors));

    for i in 0..n_rows {
        // Index of the entry to drop: the first self-match, else the last.
        let drop = (0..query_k)
            .find(|&j| indices[[i, j]] == i)
            .unwrap_or(query_k - 1);
        let mut out_col = 0;
        for j in 0..query_k {
            if j == drop || out_col == n_neighbors {
                continue;
            }
            out_dist[[i, out_col]] = distances[[i, j]];
            out_idx[[i, out_col]] = indices[[i, j]];
            out_col += 1;
        }
    }

    (out_dist, out_idx)
}

/// Compute the (weighted) graph of radius-based neighbors for points in
/// `x`. Equivalent to sklearn `sklearn.neighbors.radius_neighbors_graph`.
///
/// Output shape is `(n_samples, n_samples)`.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] / [`FerroError::InvalidParameter`]
/// per the underlying radius search.
pub fn radius_neighbors_graph<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    radius: F,
    mode: GraphMode,
) -> Result<CsrMatrix<F>, FerroError> {
    let dummy_y: Array1<usize> = Array1::zeros(x.nrows());
    let clf = RadiusNeighborsClassifier::<F>::new()
        .with_radius(radius)
        .with_algorithm(Algorithm::Auto);
    let fitted = clf.fit(x, &dummy_y)?;
    let (distances, indices) = fitted.radius_neighbors(x, Some(radius))?;
    // sklearn's `radius_neighbors_graph` defaults `include_self=False`, routing
    // through `_query_include_self` (`sklearn/neighbors/_graph.py:255`:
    // "if not include_self: X = None") so the always-within-radius distance-0
    // self-edge is excluded — the graph has a zero diagonal. Unlike kneighbors
    // (fixed-width output), radius returns variable-length rows, so we simply
    // drop any entry where neighbor index == query row index.
    let n_rows = x.nrows();
    let distances_no_self: Vec<Vec<F>> = (0..n_rows)
        .map(|i| {
            distances[i]
                .iter()
                .zip(indices[i].iter())
                .filter(|&(_, &j)| j != i)
                .map(|(&d, _)| d)
                .collect()
        })
        .collect();
    let indices_no_self: Vec<Vec<usize>> = (0..n_rows)
        .map(|i| indices[i].iter().filter(|&&j| j != i).copied().collect())
        .collect();
    radius_to_csr(&distances_no_self, &indices_no_self, n_rows, n_rows, mode)
}

/// Build a CSR matrix from k-neighbor query results.
///
/// `distances` and `indices` are aligned `(n_rows, n_neighbors)` arrays
/// such as those returned by `FittedKNeighborsClassifier::kneighbors`.
/// `n_cols` should be the number of training samples.
pub(crate) fn knn_to_csr<F: Float>(
    distances: &Array2<F>,
    indices: &Array2<usize>,
    n_rows: usize,
    n_cols: usize,
    mode: GraphMode,
) -> Result<CsrMatrix<F>, FerroError> {
    let k = distances.ncols();
    let mut indptr: Vec<usize> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    let mut col_indices: Vec<usize> = Vec::with_capacity(n_rows * k);
    let mut data: Vec<F> = Vec::with_capacity(n_rows * k);

    for i in 0..n_rows {
        // sklearn requires CSR inner indices to be sorted per row, so we
        // build a (col, value) buffer per row, sort by column, then push.
        let mut row_pairs: Vec<(usize, F)> = (0..k)
            .map(|j| {
                let col = indices[[i, j]];
                let val = match mode {
                    GraphMode::Connectivity => F::one(),
                    GraphMode::Distance => distances[[i, j]],
                };
                (col, val)
            })
            .collect();
        row_pairs.sort_by_key(|(c, _)| *c);
        for (col, val) in row_pairs {
            col_indices.push(col);
            data.push(val);
        }
        indptr.push(col_indices.len());
    }

    CsrMatrix::new(n_rows, n_cols, indptr, col_indices, data)
}

/// Build a CSR matrix from radius-neighbor query results (jagged
/// `Vec<Vec<…>>`).
pub(crate) fn radius_to_csr<F: Float>(
    distances: &[Vec<F>],
    indices: &[Vec<usize>],
    n_rows: usize,
    n_cols: usize,
    mode: GraphMode,
) -> Result<CsrMatrix<F>, FerroError> {
    let mut indptr: Vec<usize> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    let mut col_indices: Vec<usize> = Vec::new();
    let mut data: Vec<F> = Vec::new();

    for i in 0..n_rows {
        let mut row_pairs: Vec<(usize, F)> = indices[i]
            .iter()
            .zip(distances[i].iter())
            .map(|(&col, &dist)| {
                let val = match mode {
                    GraphMode::Connectivity => F::one(),
                    GraphMode::Distance => dist,
                };
                (col, val)
            })
            .collect();
        row_pairs.sort_by_key(|(c, _)| *c);
        // Deduplicate any repeated columns (radius search may return the
        // query point itself or near-duplicates).
        let mut last_col: Option<usize> = None;
        for (col, val) in row_pairs {
            if Some(col) == last_col {
                continue;
            }
            last_col = Some(col);
            col_indices.push(col);
            data.push(val);
        }
        indptr.push(col_indices.len());
    }

    CsrMatrix::new(n_rows, n_cols, indptr, col_indices, data)
}

/// Sort each row of a sparse neighbor graph by ascending column index.
///
/// Mirrors sklearn `sort_graph_by_row_values`. The CSR inputs we
/// construct are already sorted per row; this function exists for
/// round-trips through external graph sources that may not preserve the
/// invariant. Returns a new matrix; never mutates the input.
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if the rebuilt matrix fails
/// the structural validity check.
pub fn sort_graph_by_row_values<F: Float + 'static>(
    graph: &CsrMatrix<F>,
) -> Result<CsrMatrix<F>, FerroError> {
    let n_rows = graph.n_rows();
    let n_cols = graph.n_cols();
    let dense = graph.to_dense();

    let mut indptr: Vec<usize> = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    let mut col_indices: Vec<usize> = Vec::new();
    let mut data: Vec<F> = Vec::new();

    for r in 0..n_rows {
        for c in 0..n_cols {
            let v = dense[[r, c]];
            if v != F::zero() {
                col_indices.push(c);
                data.push(v);
            }
        }
        indptr.push(col_indices.len());
    }

    CsrMatrix::new(n_rows, n_cols, indptr, col_indices, data)
}

// ---------------------------------------------------------------------------
// Method-form helpers — exposed via inherent impls on the fitted estimators.
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> FittedNearestNeighbors<F> {
    /// Compute the k-neighbors graph for the rows of `x` against the
    /// training data. Equivalent to sklearn
    /// `KNeighborsMixin.kneighbors_graph`.
    ///
    /// `n_neighbors = None` uses the value passed at construction.
    /// Output shape is `(x.nrows(), self.n_samples_fit())`.
    ///
    /// # Errors
    ///
    /// As [`FittedNearestNeighbors::kneighbors`].
    pub fn kneighbors_graph(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        let (distances, indices) = self.kneighbors(x, n_neighbors)?;
        knn_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }

    /// Compute the radius-neighbors graph for the rows of `x` against the
    /// training data. Equivalent to sklearn
    /// `RadiusNeighborsMixin.radius_neighbors_graph`.
    ///
    /// Output shape is `(x.nrows(), self.n_samples_fit())`.
    ///
    /// # Errors
    ///
    /// As [`FittedNearestNeighbors::radius_neighbors`].
    pub fn radius_neighbors_graph(
        &self,
        x: &Array2<F>,
        radius: F,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        // FittedNearestNeighbors::radius_neighbors returns
        // Vec<(Vec<F>, Vec<usize>)>: one (distances, indices) tuple per row.
        let rows = self.radius_neighbors(x, radius)?;
        let (distances, indices): (Vec<Vec<F>>, Vec<Vec<usize>>) = rows.into_iter().unzip();
        radius_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }
}

// Expose the same graph methods on the supervised KNN/Radius estimators.

impl<F: Float + Send + Sync + 'static> crate::FittedKNeighborsClassifier<F> {
    /// Compute the k-neighbors graph against the training data.
    ///
    /// # Errors
    ///
    /// As [`crate::FittedKNeighborsClassifier::kneighbors`].
    pub fn kneighbors_graph(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        let (distances, indices) = self.kneighbors(x, n_neighbors)?;
        knn_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }
}

impl<F: Float + Send + Sync + 'static> crate::FittedKNeighborsRegressor<F> {
    /// Compute the k-neighbors graph against the training data.
    ///
    /// # Errors
    ///
    /// As [`crate::FittedKNeighborsRegressor::kneighbors`].
    pub fn kneighbors_graph(
        &self,
        x: &Array2<F>,
        n_neighbors: Option<usize>,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        let (distances, indices) = self.kneighbors(x, n_neighbors)?;
        knn_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }
}

impl<F: Float + Send + Sync + 'static> crate::FittedRadiusNeighborsClassifier<F> {
    /// Compute the radius-neighbors graph against the training data.
    ///
    /// # Errors
    ///
    /// As [`crate::FittedRadiusNeighborsClassifier::radius_neighbors`].
    pub fn radius_neighbors_graph(
        &self,
        x: &Array2<F>,
        radius: Option<F>,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        let (distances, indices) = self.radius_neighbors(x, radius)?;
        radius_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }
}

impl<F: Float + Send + Sync + 'static> crate::FittedRadiusNeighborsRegressor<F> {
    /// Compute the radius-neighbors graph against the training data.
    ///
    /// # Errors
    ///
    /// As [`crate::FittedRadiusNeighborsRegressor::radius_neighbors`].
    pub fn radius_neighbors_graph(
        &self,
        x: &Array2<F>,
        radius: Option<F>,
        mode: GraphMode,
    ) -> Result<CsrMatrix<F>, FerroError> {
        let (distances, indices) = self.radius_neighbors(x, radius)?;
        radius_to_csr(&distances, &indices, x.nrows(), self.n_samples_fit(), mode)
    }
}
