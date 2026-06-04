//! Select features by percentile of highest scores.
//!
//! [`SelectPercentile`] retains features whose ANOVA F-score exceeds the
//! `(100 - percentile)`-th percentile of all scores (sklearn `_get_support_mask`).
//! It reuses the scoring infrastructure from [`crate::feature_selection`].
//!
//! Translation target: scikit-learn 1.5.2 `class SelectPercentile`
//! (`sklearn/feature_selection/_univariate_selection.py:589`) + `f_classif`
//! (`:127`). Design: `.design/preprocess/select_percentile.md`. Tracking: #1273.
//!
//! `## REQ status`
//!
//! | REQ | Status | Anchor |
//! |---|---|---|
//! | REQ-1 ANOVA F-score value match (f_classif, finite) | SHIPPED | `anova_f_scores`; sklearn `_univariate_selection.py:127`,`:43-117` |
//! | REQ-2 selection mask `_get_support_mask` (threshold + tie-fill) | SHIPPED (#1274) | `SelectPercentile::fit` `numpy_percentile`; sklearn `_univariate_selection.py:669-686` |
//! | REQ-3 InsufficientSamples / ShapeMismatch / InvalidParameter errors | SHIPPED | `fit` / `transform` guards; sklearn `_univariate_selection.py:662` |
//! | REQ-4 `_clean_nans` (NaN scores → dtype.min) | NOT-STARTED (#1275) | sklearn `_univariate_selection.py:24`,`:678` |
//! | REQ-5 pluggable score_func (chi2/f_regression/mutual_info_*) | NOT-STARTED (#1276) | sklearn `_univariate_selection.py:202`,`:405`,`:596-599` |
//! | REQ-6 SelectorMixin (get_support/inverse_transform/get_feature_names_out) | NOT-STARTED (#1277) | sklearn `feature_selection/_base.py` |
//! | REQ-7 `pvalues_` fitted attribute | NOT-STARTED (#1278) | sklearn `_univariate_selection.py:567-573`,`:116` |
//! | REQ-8 fractional `percentile` (`Interval(Real,0,100)`) | NOT-STARTED (#1279) | sklearn `_univariate_selection.py:662` |
//! | REQ-9 PyO3 binding | NOT-STARTED (#1280) | `ferrolearn-python/src/` (absent) |
//! | REQ-10 ferray substrate | NOT-STARTED (#1281) | R-SUBSTRATE |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::feature_selection::ScoreFunc;

// ---------------------------------------------------------------------------
// Helper: ANOVA F-scores (duplicated from feature_selection to avoid pub(crate))
// ---------------------------------------------------------------------------

/// Compute per-feature ANOVA F-scores.
fn anova_f_scores<F: Float>(x: &Array2<F>, y: &Array1<usize>) -> Vec<F> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }
    let n_classes = class_indices.len();

    let mut scores = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let grand_mean =
            col.iter().copied().fold(F::zero(), |acc, v| acc + v) / F::from(n_samples).unwrap();

        let mut ss_between = F::zero();
        let mut ss_within = F::zero();

        for rows in class_indices.values() {
            let n_k = F::from(rows.len()).unwrap();
            let class_mean = rows
                .iter()
                .map(|&i| col[i])
                .fold(F::zero(), |acc, v| acc + v)
                / n_k;
            let diff = class_mean - grand_mean;
            ss_between = ss_between + n_k * diff * diff;
            for &i in rows {
                let d = col[i] - class_mean;
                ss_within = ss_within + d * d;
            }
        }

        let df_between = F::from(n_classes.saturating_sub(1)).unwrap();
        let df_within = F::from(n_samples.saturating_sub(n_classes)).unwrap();

        let f = if df_between == F::zero() || df_within == F::zero() {
            F::zero()
        } else {
            let ms_between = ss_between / df_between;
            let ms_within = ss_within / df_within;
            if ms_within == F::zero() {
                F::infinity()
            } else {
                ms_between / ms_within
            }
        };

        scores.push(f);
    }

    scores
}

/// Compute `np.percentile(sorted, q)` with the default `'linear'` interpolation.
///
/// `sorted` must be ascending. `q` is in `[0, 100]`. Mirrors numpy's default
/// percentile (linear interpolation), the same idiom as `quantile_sorted` in
/// `robust_scaler.rs`. Used to reproduce sklearn `_get_support_mask`
/// (`_univariate_selection.py:679`).
fn numpy_percentile<F: Float>(sorted: &[F], q: f64) -> F {
    let n = sorted.len();
    if n == 0 {
        return F::nan();
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (q / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = F::from(idx - lo as f64).unwrap_or_else(F::zero);
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Build a new `Array2<F>` containing only the columns listed in `indices`.
fn select_columns<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let nrows = x.nrows();
    let ncols = indices.len();
    if ncols == 0 {
        return Array2::zeros((nrows, 0));
    }
    let mut out = Array2::zeros((nrows, ncols));
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..nrows {
            out[[i, new_j]] = x[[i, old_j]];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SelectPercentile
// ---------------------------------------------------------------------------

/// An unfitted percentile-based feature selector.
///
/// Retains the features whose ANOVA F-score ranks in the top `percentile`
/// percent.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::select_percentile::SelectPercentile;
/// use ferrolearn_preprocess::feature_selection::ScoreFunc;
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::{array, Array1};
///
/// let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
/// let x = array![[1.0, 10.0, 0.1, 0.01],
///                 [1.0, 20.0, 0.2, 0.02],
///                 [2.0, 10.0, 0.1, 0.01],
///                 [2.0, 20.0, 0.2, 0.02]];
/// let y: Array1<usize> = array![0, 0, 1, 1];
/// let fitted = sel.fit(&x, &y).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert_eq!(out.ncols(), 2); // 50% of 4 features = 2
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct SelectPercentile<F> {
    /// Percentile of features to keep (0-100).
    percentile: usize,
    /// Scoring function.
    score_func: ScoreFunc,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> SelectPercentile<F> {
    /// Create a new `SelectPercentile` selector.
    ///
    /// # Parameters
    ///
    /// - `percentile` — the percentile of top-scoring features to keep (0-100).
    /// - `score_func` — the scoring function to use.
    pub fn new(percentile: usize, score_func: ScoreFunc) -> Self {
        Self {
            percentile,
            score_func,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the percentile.
    #[must_use]
    pub fn percentile(&self) -> usize {
        self.percentile
    }

    /// Return the score function.
    #[must_use]
    pub fn score_func(&self) -> ScoreFunc {
        self.score_func
    }
}

impl<F: Float + Send + Sync + 'static> Default for SelectPercentile<F> {
    fn default() -> Self {
        Self::new(10, ScoreFunc::FClassif)
    }
}

// ---------------------------------------------------------------------------
// FittedSelectPercentile
// ---------------------------------------------------------------------------

/// A fitted percentile selector holding scores and selected indices.
///
/// Created by calling [`Fit::fit`] on a [`SelectPercentile`].
#[derive(Debug, Clone)]
pub struct FittedSelectPercentile<F> {
    /// Number of features seen during fitting.
    n_features_in: usize,
    /// Per-feature scores.
    scores: Array1<F>,
    /// Indices of selected columns (in original column order).
    selected_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSelectPercentile<F> {
    /// Return the per-feature scores.
    #[must_use]
    pub fn scores(&self) -> &Array1<F> {
        &self.scores
    }

    /// Return the indices of selected columns.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for SelectPercentile<F> {
    type Fitted = FittedSelectPercentile<F>;
    type Error = FerroError;

    /// Fit by computing per-feature scores and selecting the top percentile.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - [`FerroError::InvalidParameter`] if `percentile` > 100.
    /// - [`FerroError::ShapeMismatch`] if `x` and `y` have different row counts.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedSelectPercentile<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SelectPercentile::fit".into(),
            });
        }
        if y.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "SelectPercentile::fit — y must have same length as x rows".into(),
            });
        }
        if self.percentile > 100 {
            return Err(FerroError::InvalidParameter {
                name: "percentile".into(),
                reason: format!("percentile must be in [0, 100], got {}", self.percentile),
            });
        }

        let n_features = x.ncols();
        let raw_scores = match self.score_func {
            ScoreFunc::FClassif => anova_f_scores(x, y),
        };
        let scores = Array1::from_vec(raw_scores.clone());

        // Selection mirrors sklearn `_get_support_mask`
        // (`_univariate_selection.py:669-686`): an np.percentile threshold with
        // STRICT `>` plus an int()-floor tie-fill in ascending index order —
        // NOT a ceil rank-top-k rule.
        let selected_indices: Vec<usize> = if self.percentile == 100 {
            // :673-674 — keep all features.
            (0..n_features).collect()
        } else if self.percentile == 0 {
            // :675-676 — keep none.
            Vec::new()
        } else {
            // :679 — threshold = np.percentile(scores, 100 - percentile).
            let mut sorted = raw_scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = numpy_percentile(&sorted, 100.0 - self.percentile as f64);

            // :680 — mask = scores > threshold (strict greater), collected in
            // ascending index order.
            #[allow(
                clippy::float_cmp,
                reason = "mirrors sklearn np.where(scores == threshold), _univariate_selection.py:681"
            )]
            let mut selected: Vec<usize> = (0..n_features)
                .filter(|&j| raw_scores[j] > threshold)
                .collect();

            // :681-685 — fill ties (scores == threshold) in ascending index
            // order up to max_feats = int(n * percentile / 100) (floor).
            #[allow(
                clippy::float_cmp,
                reason = "mirrors sklearn np.where(scores == threshold), _univariate_selection.py:681"
            )]
            let ties: Vec<usize> = (0..n_features)
                .filter(|&j| raw_scores[j] == threshold)
                .collect();
            if !ties.is_empty() {
                let max_feats = ((n_features as f64) * (self.percentile as f64) / 100.0) as usize;
                let fill = max_feats.saturating_sub(selected.len());
                for &t in ties.iter().take(fill) {
                    selected.push(t);
                }
            }
            selected.sort_unstable();
            selected
        };

        Ok(FittedSelectPercentile {
            n_features_in: n_features,
            scores,
            selected_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSelectPercentile<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns differs
    /// from the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSelectPercentile::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_select_percentile_50_percent() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        // Feature 0 separates classes; features 1-3 do not
        let x = array![
            [1.0, 5.0, 0.1, 0.01],
            [1.0, 6.0, 0.2, 0.02],
            [10.0, 5.0, 0.1, 0.01],
            [10.0, 6.0, 0.2, 0.02]
        ];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        // 50% of 4 = 2 features
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_select_percentile_100_percent_keeps_all() {
        let sel = SelectPercentile::<f64>::new(100, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert_eq!(out.ncols(), 2);
    }

    #[test]
    fn test_select_percentile_selects_highest_scoring() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        // Feature 0 separates classes far better than feature 1 (finite scores).
        // Live sklearn 1.5.2 oracle (R-CHAR-3): f_classif -> [162.0, 0.0],
        // SelectPercentile(percentile=50).get_support(indices=True) -> [0].
        // (The previous fixture had an inf F-score for feature 0, exercising the
        // NaN/inf _clean_nans path which is REQ-4 NOT-STARTED — out of scope.)
        let x = array![[0.0, 5.0], [1.0, 5.5], [10.0, 5.0], [9.0, 5.5]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        // Feature 0 should be selected
        assert!(fitted.selected_indices().contains(&0));
    }

    #[test]
    fn test_select_percentile_scores_stored() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        assert_eq!(fitted.scores().len(), 2);
    }

    #[test]
    fn test_select_percentile_zero_rows_error() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        let x: Array2<f64> = Array2::zeros((0, 3));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_percentile_over_100_error() {
        let sel = SelectPercentile::<f64>::new(150, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_percentile_y_length_mismatch_error() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0]; // wrong length
        assert!(sel.fit(&x, &y).is_err());
    }

    #[test]
    fn test_select_percentile_shape_mismatch_on_transform() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_select_percentile_default() {
        let sel = SelectPercentile::<f64>::default();
        assert_eq!(sel.percentile(), 10);
    }

    #[test]
    fn test_select_percentile_indices_sorted() {
        let sel = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif);
        let x = array![
            [1.0, 100.0, 0.5, 0.01],
            [2.0, 200.0, 0.6, 0.02],
            [10.0, 100.0, 0.5, 0.01],
            [20.0, 200.0, 0.6, 0.02]
        ];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let fitted = sel.fit(&x, &y).unwrap();
        let indices = fitted.selected_indices();
        // Indices should be sorted
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }
}
