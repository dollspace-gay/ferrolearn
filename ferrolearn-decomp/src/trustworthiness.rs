//! Trustworthiness score for manifold embeddings.
//!
//! This is the dense Euclidean Rust analogue of
//! `sklearn.manifold.trustworthiness` (`sklearn/manifold/_t_sne.py:456`).
//! It measures how many neighbors introduced by an embedding were not close in
//! the original input space.

use std::cmp::Ordering;

use ferrolearn_core::FerroError;
use ndarray::{Array2, ArrayView1};

fn squared_euclidean(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| {
            let d = av - bv;
            d * d
        })
        .sum()
}

fn sorted_neighbor_indices(x: &Array2<f64>, row: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f64)> = (0..x.nrows())
        .map(|idx| {
            let dist = if idx == row {
                f64::INFINITY
            } else {
                squared_euclidean(x.row(row), x.row(idx))
            };
            (idx, dist)
        })
        .collect();
    distances.sort_by(|&(idx_a, dist_a), &(idx_b, dist_b)| {
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(Ordering::Equal)
            .then_with(|| idx_a.cmp(&idx_b))
    });
    distances.into_iter().map(|(idx, _)| idx).collect()
}

fn validate_trustworthiness_input(
    x: &Array2<f64>,
    x_embedded: &Array2<f64>,
    n_neighbors: usize,
) -> Result<usize, FerroError> {
    let n_samples = x.nrows();
    if x_embedded.nrows() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![x_embedded.nrows()],
            context: "trustworthiness: X and X_embedded must have the same number of rows".into(),
        });
    }
    if n_neighbors < 1 {
        return Err(FerroError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: format!("must be at least 1, got {n_neighbors}"),
        });
    }
    if (n_neighbors as f64) >= (n_samples as f64) / 2.0 {
        return Err(FerroError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: format!(
                "n_neighbors ({n_neighbors}) should be less than n_samples / 2 ({})",
                (n_samples as f64) / 2.0
            ),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    if x_embedded.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "X_embedded".into(),
            reason: "Input X_embedded contains NaN or infinity.".into(),
        });
    }
    Ok(n_samples)
}

/// Compute the trustworthiness of an embedding.
///
/// The score lies in `[0, 1]` when `n_neighbors < n_samples / 2`. A score of
/// `1.0` means every embedded nearest neighbor is also within the same
/// neighborhood radius in the original input space. This function currently
/// supports dense Euclidean inputs; sklearn's sparse, precomputed, cosine, and
/// callable metric branches are not part of this Rust surface.
///
/// # Errors
///
/// Returns [`FerroError`] when `X` and `X_embedded` have different sample
/// counts, `n_neighbors < 1`, `n_neighbors >= n_samples / 2`, or either matrix
/// contains a non-finite value.
pub fn trustworthiness(
    x: &Array2<f64>,
    x_embedded: &Array2<f64>,
    n_neighbors: usize,
) -> Result<f64, FerroError> {
    let n_samples = validate_trustworthiness_input(x, x_embedded, n_neighbors)?;

    let mut original_ranks = vec![vec![0usize; n_samples]; n_samples];
    for (i, ranks_row) in original_ranks.iter_mut().enumerate() {
        let indices = sorted_neighbor_indices(x, i);
        for (rank_zero_based, idx) in indices.into_iter().enumerate() {
            ranks_row[idx] = rank_zero_based + 1;
        }
    }

    let mut rank_penalty = 0usize;
    for (i, ranks_row) in original_ranks.iter().enumerate() {
        let embedded_neighbors = sorted_neighbor_indices(x_embedded, i);
        for &neighbor in embedded_neighbors.iter().take(n_neighbors) {
            let rank = ranks_row[neighbor];
            if rank > n_neighbors {
                rank_penalty += rank - n_neighbors;
            }
        }
    }

    let n = n_samples as f64;
    let k = n_neighbors as f64;
    let normalizer = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0));
    Ok(1.0 - (rank_penalty as f64) * normalizer)
}
