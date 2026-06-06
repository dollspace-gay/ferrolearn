//! Ranking evaluation metrics.
//!
//! This module provides ranking metrics commonly used to evaluate recommendation
//! and information retrieval systems:
//!
//! - [`dcg_score`] — Discounted Cumulative Gain
//! - [`ndcg_score`] — Normalized Discounted Cumulative Gain
//!
//! ## REQ status
//!
//! Mirrors scikit-learn ranking metrics (`sklearn/metrics/_ranking.py`). See
//! `.design/metrics/ranking.md`. Non-test consumer: crate re-export. The
//! dcg/ndcg/label_ranking value divergences (#754/#755/#756) are fixed; the
//! remaining gaps are the 2D-multisample + `sample_weight` signature
//! extensions and the ROC/PR surface (owned by `classification.rs`).
//!
//! | REQ | Description | Status |
//! |-----|-------------|--------|
//! | REQ-1 | `dcg_score` 1D tie-averaged DCG (default `ignore_ties=False`, `_ranking.py:1528`): tied `y_score` share the mean discount over the ranks they span | SHIPPED |
//! | REQ-2 | `ndcg_score` 1D tie-averaged + negative-`y_true` `ValueError` guard (`_ranking.py:1770,:1868-1869`); ideal-DCG plain (`:1753`) | SHIPPED |
//! | REQ-3 | `coverage_error` 1D value: tie max-rank, empty-row → 0 (`_ranking.py:1301`) | SHIPPED |
//! | REQ-4 | `label_ranking_average_precision_score` 1D value (`_ranking.py:1202`) | SHIPPED |
//! | REQ-5 | `label_ranking_loss` 1D value: degenerate rows → 0, averaged over all `n_samples` (`_ranking.py:1461-1465`) | SHIPPED |
//! | REQ-6a | `dcg_score` `log_base` parameter (`_ranking.py:1511`, discount `ln(base)/ln(i+2)`); validated to `(0,∞)\{1}`; `ndcg_score` is base-invariant so its signature is unchanged | SHIPPED |
//! | REQ-6b | `dcg_score`/`ndcg_score` 2D `(n_samples,n_labels)` sample-mean + `sample_weight` | NOT-STARTED (#761) |
//! | REQ-7 | `coverage_error`/LRAP/`label_ranking_loss` `sample_weight`: weighted mean `sum_i(w_i * v_i)/sum_i(w_i)` (`_ranking.py:1365,:1281-1288,:1465`) | SHIPPED |
//! | REQ-8 | ROC/PR surface (`auc`/`roc_auc_score`/`roc_curve`/`precision_recall_curve`/`average_precision_score`/`det_curve`/`top_k_accuracy_score`) currently in `classification.rs` | NOT-STARTED (#763) |
//! | REQ-9 | PyO3 binding for ranking metrics | NOT-STARTED (#764) |
//! | REQ-10 | ferray substrate migration | NOT-STARTED (#765) |

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sort indices by descending `y_score`, breaking ties by index.
fn argsort_desc<F: Float>(arr: &Array1<F>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&a, &b| {
        arr[b]
            .partial_cmp(&arr[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    indices
}

/// Per-rank discount cumulative sum: `discount[i] = log(base) / log(i + 2)`
/// (i.e. `1 / log_base(i + 2)`), with `discount[i] = 0` for `i >= k` (the
/// DCG@k cutoff zeroes the discount past rank `k` *before* the cumsum), then
/// cumulative-summed over all `n` ranks.
///
/// sklearn computes `discount = 1 / (np.log(arange + 2) / np.log(log_base))`
/// (`sklearn/metrics/_ranking.py:1511`), equivalently `ln(base) / ln(i + 2)`.
///
/// Mirrors sklearn `_dcg_sample_scores` (`sklearn/metrics/_ranking.py:1511-1519`).
fn discount_cumsum<F: Float>(n: usize, k: usize, base: F) -> Vec<F> {
    let ln_base = base.ln();
    let mut cum = Vec::with_capacity(n);
    let mut acc = F::zero();
    for i in 0..n {
        if i < k {
            // discount_i = ln(base) / ln(i + 2) == 1 / log_base(i + 2)
            let d = ln_base / F::from(i + 2).unwrap_or_else(F::one).ln();
            acc = acc + d;
        }
        cum.push(acc);
    }
    cum
}

/// Compute the raw DCG with sklearn's default tie-averaging (`ignore_ties=False`).
///
/// Samples tied on `y_score` (in the descending-score order) form a group; each
/// group's gain is the mean `y_true` of its members times the sum of discounts
/// over the consecutive ranks the group spans. With all scores distinct this
/// reduces to the plain `sum_i y_true[ranked_i] / log2(i + 2)`.
///
/// `base` is the logarithm base of the discount (sklearn `log_base`, default 2).
///
/// Mirrors sklearn `_tie_averaged_dcg` (`sklearn/metrics/_ranking.py:1565-1573`).
fn compute_dcg_tie_averaged<F: Float>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
    k: usize,
    base: F,
) -> F {
    let n = y_true.len();
    let cum = discount_cumsum::<F>(n, k, base);

    // Order samples by descending y_score (ties keep their relative index order),
    // then walk consecutive equal-score runs as groups.
    let ranked_indices = argsort_desc(y_score);

    let mut dcg = F::zero();
    let mut prev_group_end_cumsum = F::zero();
    let mut group_start = 0usize;
    while group_start < n {
        // Extend the group while the score equals the group's first score.
        let group_score = y_score[ranked_indices[group_start]];
        let mut group_end = group_start;
        let mut gain_sum = F::zero();
        let mut count = 0usize;
        while group_end < n && y_score[ranked_indices[group_end]] == group_score {
            gain_sum = gain_sum + y_true[ranked_indices[group_end]];
            count += 1;
            group_end += 1;
        }
        // Last 0-based rank index this group spans is `group_end - 1`.
        let group_cumsum = cum[group_end - 1];
        let discount_sum = group_cumsum - prev_group_end_cumsum;
        let count_f = F::from(count).unwrap_or_else(F::one);
        let mean_gain = gain_sum / count_f;
        dcg = dcg + mean_gain * discount_sum;

        prev_group_end_cumsum = group_cumsum;
        group_start = group_end;
    }
    dcg
}

/// Compute the raw DCG ignoring ties (plain DCG over a pre-sorted relevance
/// ordering). Used for the ideal/normalizing DCG, where sklearn passes
/// `ignore_ties=True` (`sklearn/metrics/_ranking.py:1753`).
/// `base` is the logarithm base of the discount (sklearn `log_base`, default 2);
/// for NDCG normalization it must equal the base used for the numerator DCG.
fn compute_dcg_plain<F: Float>(relevances: &[F], k: usize, base: F) -> F {
    let ln_base = base.ln();
    let mut dcg = F::zero();
    for (i, &rel) in relevances.iter().take(k).enumerate() {
        // DCG_i = rel_i * ln(base) / ln(i + 2) == rel_i / log_base(i + 2)
        let denom = F::from(i + 2).unwrap_or_else(F::one).ln();
        dcg = dcg + rel * ln_base / denom;
    }
    dcg
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the Discounted Cumulative Gain (DCG).
///
/// Items are ranked by descending `y_score`. The DCG is defined as:
///
/// ```text
/// DCG@k = sum_{i=0}^{k-1} y_true[ranked_i] / log_base(i + 2)
/// ```
///
/// # Arguments
///
/// * `y_true` — relevance scores (ground-truth gains).
/// * `y_score` — predicted scores used to rank items.
/// * `k` — optional cutoff; if `None`, all items are used.
/// * `log_base` — base of the discount logarithm (sklearn `log_base`,
///   `_ranking.py:1492-1494`); `None` defaults to `2`. A low base means a
///   sharper discount (top results matter more).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` have
/// different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
/// Returns [`FerroError::InvalidParameter`] if `log_base` is `<= 0` or `== 1`
/// (sklearn constrains `log_base` to the open interval `(0, ∞)`,
/// `_ranking.py` `validate_params`; base `1` makes `ln(base) = 0` degenerate).
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::ranking::dcg_score;
/// use ndarray::array;
///
/// let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
/// let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let dcg = dcg_score(&y_true, &y_score, None, None).unwrap();
/// assert!(dcg > 0.0);
///
/// // A different log base rescales the DCG by ln(base)/ln(2).
/// let dcg10 = dcg_score(&y_true, &y_score, None, Some(10.0)).unwrap();
/// assert!((dcg10 - dcg * 10.0_f64.ln() / 2.0_f64.ln()).abs() < 1e-9);
/// ```
pub fn dcg_score<F: Float + Send + Sync + 'static>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
    k: Option<usize>,
    log_base: Option<F>,
) -> Result<F, FerroError> {
    let n = y_true.len();
    if n != y_score.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_score.len()],
            context: "dcg_score: y_true vs y_score".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "dcg_score".into(),
        });
    }

    // sklearn default log_base=2; validate_params constrains it to (0, ∞).
    let base = log_base.unwrap_or_else(|| F::from(2).unwrap_or_else(F::one));
    if base <= F::zero() || base == F::one() {
        return Err(FerroError::InvalidParameter {
            name: "log_base".into(),
            reason: "must be > 0 and != 1".into(),
        });
    }

    let k = k.unwrap_or(n).min(n);
    Ok(compute_dcg_tie_averaged(y_true, y_score, k, base))
}

/// Compute the Normalized Discounted Cumulative Gain (NDCG).
///
/// NDCG is the ratio of the DCG to the ideal DCG (computed by sorting
/// `y_true` in descending order). NDCG is always in `[0, 1]` when
/// relevances are non-negative.
///
/// ```text
/// NDCG@k = DCG@k / ideal_DCG@k
/// ```
///
/// When the ideal DCG is zero (all relevances are zero), the NDCG is
/// defined as `0.0`.
///
/// # Arguments
///
/// * `y_true` — relevance scores (ground-truth gains).
/// * `y_score` — predicted scores used to rank items.
/// * `k` — optional cutoff; if `None`, all items are used.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` have
/// different lengths.
/// Returns [`FerroError::InsufficientSamples`] if the arrays are empty.
///
/// # Examples
///
/// ```
/// use ferrolearn_metrics::ranking::ndcg_score;
/// use ndarray::array;
///
/// let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
/// let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
/// assert!(ndcg > 0.0 && ndcg <= 1.0);
///
/// // Perfect ranking yields NDCG = 1.0
/// let y_perfect = array![3.0_f64, 3.0, 2.0, 2.0, 1.0, 0.0];
/// let y_score_perf = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let ndcg_perf = ndcg_score(&y_perfect, &y_score_perf, None).unwrap();
/// assert!((ndcg_perf - 1.0).abs() < 1e-10);
/// ```
pub fn ndcg_score<F: Float + Send + Sync + 'static>(
    y_true: &Array1<F>,
    y_score: &Array1<F>,
    k: Option<usize>,
) -> Result<F, FerroError> {
    let n = y_true.len();
    if n != y_score.len() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n],
            actual: vec![y_score.len()],
            context: "ndcg_score: y_true vs y_score".into(),
        });
    }
    if n == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "ndcg_score".into(),
        });
    }

    // sklearn/metrics/_ranking.py:1868-1869: raise ValueError when y_true.min() < 0
    if y_true.iter().any(|&v| v < F::zero()) {
        return Err(FerroError::InvalidParameter {
            name: "y_true".into(),
            reason: "ndcg_score should not be used on negative y_true values".into(),
        });
    }

    let k = k.unwrap_or(n).min(n);

    // ndcg_score has NO log_base parameter (sklearn `ndcg_score`,
    // `_ranking.py:1770`): the base cancels in the DCG/ideal-DCG ratio, so
    // base 2 is used internally for both numerator and denominator.
    let base = F::from(2).unwrap_or_else(F::one);

    // Actual DCG: tie-averaged over the order induced by y_score (sklearn
    // default ignore_ties=False).
    let dcg = compute_dcg_tie_averaged(y_true, y_score, k, base);

    // Ideal (normalizing) DCG: sort y_true descending. sklearn normalizes with
    // ignore_ties=True (_ranking.py:1753) — permuting tied y_true entries does
    // not change the re-ordered relevances, so the plain DCG is used here.
    let mut ideal_relevances: Vec<F> = y_true.iter().copied().collect();
    ideal_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let ideal_dcg = compute_dcg_plain(&ideal_relevances, k, base);

    if ideal_dcg == F::zero() {
        return Ok(F::zero());
    }

    Ok(dcg / ideal_dcg)
}

// ---------------------------------------------------------------------------
// Multilabel ranking metrics
// ---------------------------------------------------------------------------

fn check_ranking_inputs<F: Float>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
    context: &str,
) -> Result<(), FerroError> {
    if y_true.shape() != y_score.shape() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![y_true.nrows(), y_true.ncols()],
            actual: vec![y_score.nrows(), y_score.ncols()],
            context: format!("{context}: y_true and y_score must have the same shape"),
        });
    }
    if y_true.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    Ok(())
}

/// Compute the coverage error: how far down the ranked list we have to go
/// to cover all true labels, averaged over samples.
///
/// Lower is better; 1.0 is perfect (every true label sits at the top of the
/// score-sorted list for its sample).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn coverage_error<F>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
    sample_weight: Option<&Array1<F>>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "coverage_error")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    if let Some(w) = sample_weight
        && w.len() != n_samples
    {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![w.len()],
            context: "coverage_error: sample_weight".into(),
        });
    }

    // sklearn returns `np.average(coverage, weights=sample_weight)`
    // (`_ranking.py:1365`): the per-sample coverage is averaged with
    // `sum_i(w_i * cov_i) / sum_i(w_i)`; with `sample_weight=None` every
    // `w_i = 1`, recovering the plain `sum/n_samples` mean.
    let mut weighted_acc = F::zero();
    let mut weight_sum = F::zero();
    for i in 0..n_samples {
        let w_i = sample_weight.map_or_else(F::one, |w| w[i]);
        weight_sum = weight_sum + w_i;
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        // Find the lowest score among the true (positive) labels for this row.
        let mut min_pos_score = F::infinity();
        let mut any = false;
        for j in 0..n_labels {
            if row_true[j] > 0 {
                any = true;
                if row_score[j] < min_pos_score {
                    min_pos_score = row_score[j];
                }
            }
        }
        if !any {
            // No positive labels — sklearn's masked min yields a coverage of 0
            // (`coverage.filled(0)`, `_ranking.py:1363`); the row still counts in
            // the average denominator (it contributes `0 * w_i` to the numerator
            // while `w_i` is added to `weight_sum`).
            continue;
        }
        // Coverage = number of labels with score >= min_pos_score
        let mut cov = 0usize;
        for j in 0..n_labels {
            if row_score[j] >= min_pos_score {
                cov += 1;
            }
        }
        weighted_acc = weighted_acc + w_i * F::from(cov).unwrap_or(F::zero());
    }
    Ok(weighted_acc / weight_sum)
}

/// Compute the label-ranking average precision (LRAP) score.
///
/// For each sample, considers the rank of each positive label among all
/// labels and averages the per-positive-label precision. Higher is better;
/// `1.0` is perfect.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn label_ranking_average_precision_score<F>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
    sample_weight: Option<&Array1<F>>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "label_ranking_average_precision_score")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    if let Some(w) = sample_weight
        && w.len() != n_samples
    {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![w.len()],
            context: "label_ranking_average_precision_score: sample_weight".into(),
        });
    }

    // sklearn accumulates `out += aux * sample_weight[i]` and divides by
    // `np.sum(sample_weight)` (`_ranking.py:1281-1288`); with `sample_weight=None`
    // each weight is 1 and the divisor is `n_samples`.
    let mut weighted_acc = F::zero();
    let mut weight_sum = F::zero();
    for i in 0..n_samples {
        let w_i = sample_weight.map_or_else(F::one, |w| w[i]);
        weight_sum = weight_sum + w_i;
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        let positives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] > 0).collect();
        if positives.is_empty() || positives.len() == n_labels {
            // sklearn convention: degenerate samples (all-positive or
            // all-negative) score 1.0.
            weighted_acc = weighted_acc + w_i * F::one();
            continue;
        }
        let mut sample_acc = F::zero();
        for &pos in &positives {
            let pos_score = row_score[pos];
            // Rank of `pos` is the number of labels with score >= its own.
            let mut rank = 0usize;
            let mut tp = 0usize;
            for j in 0..n_labels {
                if row_score[j] >= pos_score {
                    rank += 1;
                    if row_true[j] > 0 {
                        tp += 1;
                    }
                }
            }
            sample_acc =
                sample_acc + F::from(tp).unwrap_or(F::zero()) / F::from(rank).unwrap_or(F::one());
        }
        let aux = sample_acc / F::from(positives.len()).unwrap_or(F::one());
        weighted_acc = weighted_acc + w_i * aux;
    }
    Ok(weighted_acc / weight_sum)
}

/// Compute the label-ranking loss: the average number of badly-ordered
/// label pairs per sample, normalised by the number of possible label pairs.
///
/// Lower is better; `0.0` is perfect (all positives outrank all negatives
/// for every sample).
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `y_true` and `y_score` differ.
/// Returns [`FerroError::InsufficientSamples`] if either is empty.
pub fn label_ranking_loss<F>(
    y_true: &Array2<usize>,
    y_score: &Array2<F>,
    sample_weight: Option<&Array1<F>>,
) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    check_ranking_inputs(y_true, y_score, "label_ranking_loss")?;
    let n_samples = y_true.nrows();
    let n_labels = y_true.ncols();

    if let Some(w) = sample_weight
        && w.len() != n_samples
    {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![w.len()],
            context: "label_ranking_loss: sample_weight".into(),
        });
    }

    // sklearn returns `np.average(loss, weights=sample_weight)`
    // (`_ranking.py:1465`); degenerate rows have loss 0 and still count in the
    // denominator. With `sample_weight=None` every weight is 1 (mean over
    // n_samples).
    let mut weighted_acc = F::zero();
    let mut weight_sum = F::zero();
    for i in 0..n_samples {
        let w_i = sample_weight.map_or_else(F::one, |w| w[i]);
        weight_sum = weight_sum + w_i;
        let row_true = y_true.row(i);
        let row_score = y_score.row(i);
        let positives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] > 0).collect();
        let negatives: Vec<usize> = (0..n_labels).filter(|&j| row_true[j] == 0).collect();
        if positives.is_empty() || negatives.is_empty() {
            // Degenerate rows (all-relevant or all-irrelevant) contribute 0 to
            // the loss numerator and are still counted in the denominator.
            // sklearn/metrics/_ranking.py:1463-1465: loss[i] = 0 for degenerate
            // rows, then `np.average(loss, ...)` divides by sum(weights).
            continue;
        }
        let mut bad_pairs = 0usize;
        for &p in &positives {
            for &n in &negatives {
                if row_score[p] <= row_score[n] {
                    bad_pairs += 1;
                }
            }
        }
        let denom = positives.len() * negatives.len();
        let frac = F::from(bad_pairs).unwrap_or(F::zero()) / F::from(denom).unwrap_or(F::one());
        weighted_acc = weighted_acc + w_i * frac;
    }
    Ok(weighted_acc / weight_sum)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dcg_basic() -> Result<(), FerroError> {
        // y_true = [3, 2, 3, 0, 1, 2], y_score = [6, 5, 4, 3, 2, 1]
        // Ranking by y_score descending gives relevances: [3, 2, 3, 0, 1, 2]
        // DCG = 3/log2(0+2) + 2/log2(1+2) + 3/log2(2+2) + 0/log2(3+2) + 1/log2(4+2) + 2/log2(5+2)
        let y_true = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let y_score = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        let dcg = dcg_score(&y_true, &y_score, None, None)?;
        let expected = 3.0 / 2.0_f64.log2()
            + 2.0 / 3.0_f64.log2()
            + 3.0 / 4.0_f64.log2()
            + 0.0 / 5.0_f64.log2()
            + 1.0 / 6.0_f64.log2()
            + 2.0 / 7.0_f64.log2();
        assert!((dcg - expected).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_dcg_with_k() -> Result<(), FerroError> {
        let y_true = array![3.0_f64, 2.0, 1.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let dcg_k2 = dcg_score(&y_true, &y_score, Some(2), None)?;
        // Only first 2: 3/log2(2) + 2/log2(3)
        let expected = 3.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2();
        assert!((dcg_k2 - expected).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_ndcg_perfect_ranking() {
        let y_true = array![3.0_f64, 2.0, 1.0, 0.0];
        let y_score = array![4.0_f64, 3.0, 2.0, 1.0];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_all_zero_relevance() {
        let y_true = array![0.0_f64, 0.0, 0.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_with_k() {
        let y_true = array![0.0_f64, 1.0, 3.0, 2.0];
        let y_score = array![1.0_f64, 4.0, 3.0, 2.0];
        // Ranking by y_score desc: indices [1, 2, 3, 0] -> relevances [1, 3, 2, 0]
        // DCG@2 = 1/log2(0+2) + 3/log2(1+2) = 1/1 + 3/log2(3)
        // Ideal order of relevances: [3, 2, 1, 0]
        // ideal DCG@2 = 3/log2(2) + 2/log2(3) = 3/1 + 2/log2(3)
        let ndcg = ndcg_score(&y_true, &y_score, Some(2)).unwrap();
        let dcg = 1.0 / 2.0_f64.log2() + 3.0 / 3.0_f64.log2();
        let ideal = 3.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2();
        assert!((ndcg - dcg / ideal).abs() < 1e-10);
    }

    #[test]
    fn test_dcg_empty() {
        let y_true: Array1<f64> = Array1::zeros(0);
        let y_score: Array1<f64> = Array1::zeros(0);
        assert!(dcg_score(&y_true, &y_score, None, None).is_err());
    }

    #[test]
    fn test_dcg_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_score = array![1.0_f64, 2.0, 3.0];
        assert!(dcg_score(&y_true, &y_score, None, None).is_err());
    }

    #[test]
    fn test_ndcg_empty() {
        let y_true: Array1<f64> = Array1::zeros(0);
        let y_score: Array1<f64> = Array1::zeros(0);
        assert!(ndcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_ndcg_shape_mismatch() {
        let y_true = array![1.0_f64, 2.0];
        let y_score = array![1.0_f64];
        assert!(ndcg_score(&y_true, &y_score, None).is_err());
    }

    #[test]
    fn test_dcg_single_element() -> Result<(), FerroError> {
        let y_true = array![5.0_f64];
        let y_score = array![1.0_f64];
        let dcg = dcg_score(&y_true, &y_score, None, None)?;
        // 5 / log2(0+2) = 5 / log2(2) = 5 / 1 = 5
        assert!((dcg - 5.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_ndcg_single_element() {
        let y_true = array![5.0_f64];
        let y_score = array![1.0_f64];
        let ndcg = ndcg_score(&y_true, &y_score, None).unwrap();
        assert!((ndcg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dcg_f32() -> Result<(), FerroError> {
        let y_true = array![3.0_f32, 2.0, 1.0];
        let y_score = array![3.0_f32, 2.0, 1.0];
        let dcg = dcg_score(&y_true, &y_score, None, None)?;
        assert!(dcg > 0.0_f32);
        Ok(())
    }

    #[test]
    fn test_ndcg_k_larger_than_n() {
        let y_true = array![3.0_f64, 2.0, 1.0];
        let y_score = array![3.0_f64, 2.0, 1.0];
        let ndcg_all = ndcg_score(&y_true, &y_score, None).unwrap();
        let ndcg_big_k = ndcg_score(&y_true, &y_score, Some(100)).unwrap();
        assert!((ndcg_all - ndcg_big_k).abs() < 1e-10);
    }

    // -----------------------------------------------------------------
    // log_base divergence pins (#761, REQ-6a). Expected values from the
    // live sklearn 1.5.2 oracle:
    //   import numpy as np; from sklearn.metrics import dcg_score
    //   yt=np.array([[3,2,3,0,1,2]]); ys=np.array([[6,5,4,3,2,1]])
    //   dcg_score(yt,ys)            -> 6.8611266886  (log_base=2 default)
    //   dcg_score(yt,ys,log_base=10)-> 22.7921695094
    //   dcg_score(yt,ys,log_base=np.e) -> 9.8985134485
    // -----------------------------------------------------------------

    #[test]
    fn dcg_score_default_log2_unchanged() -> Result<(), FerroError> {
        let yt = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let ys = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        let got = dcg_score(&yt, &ys, None, None)?;
        assert!((got - 6.861_126_688_6).abs() < 1e-9, "got {got}");
        Ok(())
    }

    #[test]
    fn dcg_score_log_base_10_matches_sklearn() -> Result<(), FerroError> {
        let yt = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let ys = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        let got = dcg_score(&yt, &ys, None, Some(10.0))?;
        assert!((got - 22.792_169_509_4).abs() < 1e-9, "got {got}");
        Ok(())
    }

    #[test]
    fn dcg_score_log_base_e_matches_sklearn() -> Result<(), FerroError> {
        let yt = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let ys = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        let got = dcg_score(&yt, &ys, None, Some(std::f64::consts::E))?;
        assert!((got - 9.898_513_448_5).abs() < 1e-9, "got {got}");
        Ok(())
    }

    #[test]
    fn dcg_score_invalid_log_base_errors() {
        let yt = array![3.0_f64, 2.0, 3.0, 0.0, 1.0, 2.0];
        let ys = array![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0];
        // base == 1 (ln(1) = 0, degenerate discount), base == 0, and a
        // negative base are all rejected per sklearn's (0, ∞) constraint.
        assert!(dcg_score(&yt, &ys, None, Some(1.0)).is_err());
        assert!(dcg_score(&yt, &ys, None, Some(0.0)).is_err());
        assert!(dcg_score(&yt, &ys, None, Some(-2.0)).is_err());
    }

    // -----------------------------------------------------------------
    // sample_weight divergence pins (#762, REQ-7). Expected values from
    // the live sklearn 1.5.2 oracle:
    //   import numpy as np
    //   from sklearn.metrics import (coverage_error,
    //       label_ranking_average_precision_score as lrap,
    //       label_ranking_loss)
    //   yt=np.array([[1,0,0],[0,0,1],[0,1,1]])
    //   ys=np.array([[0.75,0.5,1.0],[1.0,0.2,0.1],[0.3,0.6,0.9]])
    //   coverage_error(yt,ys)                              -> 2.3333333333
    //   coverage_error(yt,ys,sample_weight=[3,1,1])        -> 2.2
    //   lrap(yt,ys)                                        -> 0.6111111111
    //   lrap(yt,ys,sample_weight=[1,2,3])                  -> 0.6944444444
    //   label_ranking_loss(yt,ys)                          -> 0.5
    //   label_ranking_loss(yt,ys,sample_weight=[1,2,3])    -> 0.4166666667
    // -----------------------------------------------------------------

    #[test]
    fn coverage_error_sample_weight_matches_sklearn() -> Result<(), FerroError> {
        let yt = array![[1usize, 0, 0], [0, 0, 1], [0, 1, 1]];
        let ys = array![[0.75_f64, 0.5, 1.0], [1.0, 0.2, 0.1], [0.3, 0.6, 0.9]];
        let none = coverage_error(&yt, &ys, None)?;
        assert!((none - 2.333_333_333_3).abs() < 1e-9, "none {none}");
        let w = array![3.0_f64, 1.0, 1.0];
        let weighted = coverage_error(&yt, &ys, Some(&w))?;
        assert!((weighted - 2.2).abs() < 1e-9, "weighted {weighted}");
        Ok(())
    }

    #[test]
    fn lrap_sample_weight_matches_sklearn() -> Result<(), FerroError> {
        let yt = array![[1usize, 0, 0], [0, 0, 1], [0, 1, 1]];
        let ys = array![[0.75_f64, 0.5, 1.0], [1.0, 0.2, 0.1], [0.3, 0.6, 0.9]];
        let none = label_ranking_average_precision_score(&yt, &ys, None)?;
        assert!((none - 0.611_111_111_1).abs() < 1e-9, "none {none}");
        let w = array![1.0_f64, 2.0, 3.0];
        let weighted = label_ranking_average_precision_score(&yt, &ys, Some(&w))?;
        assert!(
            (weighted - 0.694_444_444_4).abs() < 1e-9,
            "weighted {weighted}"
        );
        Ok(())
    }

    #[test]
    fn label_ranking_loss_sample_weight_matches_sklearn() -> Result<(), FerroError> {
        let yt = array![[1usize, 0, 0], [0, 0, 1], [0, 1, 1]];
        let ys = array![[0.75_f64, 0.5, 1.0], [1.0, 0.2, 0.1], [0.3, 0.6, 0.9]];
        let none = label_ranking_loss(&yt, &ys, None)?;
        assert!((none - 0.5).abs() < 1e-9, "none {none}");
        let w = array![1.0_f64, 2.0, 3.0];
        let weighted = label_ranking_loss(&yt, &ys, Some(&w))?;
        assert!(
            (weighted - 0.416_666_666_7).abs() < 1e-9,
            "weighted {weighted}"
        );
        Ok(())
    }

    #[test]
    fn ranking_sample_weight_length_mismatch_errors() {
        let yt = array![[1usize, 0, 0], [0, 0, 1], [0, 1, 1]];
        let ys = array![[0.75_f64, 0.5, 1.0], [1.0, 0.2, 0.1], [0.3, 0.6, 0.9]];
        // Wrong-length sample_weight (2 != n_samples=3) must error on all three.
        let bad = array![1.0_f64, 2.0];
        assert!(coverage_error(&yt, &ys, Some(&bad)).is_err());
        assert!(label_ranking_average_precision_score(&yt, &ys, Some(&bad)).is_err());
        assert!(label_ranking_loss(&yt, &ys, Some(&bad)).is_err());
    }
}
