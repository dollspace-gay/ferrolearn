//! Generic univariate feature selector.
//!
//! Dense Rust analogue of scikit-learn's `GenericUnivariateSelect`
//! (`sklearn/feature_selection/_univariate_selection.py:1062`). The selector
//! computes `f_classif` scores/p-values through the existing [`ScoreFunc`]
//! surface and dispatches support-mask construction to one of five modes:
//! percentile, k-best, false-positive-rate, false-discovery-rate, or
//! family-wise-error.
//!
//! ## REQ status
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 public `GenericUnivariateSelect` surface | SHIPPED scoped / residual open | exact public name exported by crate root; typed mode/param enum maps sklearn `mode` + `param` |
//! | REQ-2 mode dispatch (`percentile`, `k_best`, `fpr`, `fdr`, `fwe`) | SHIPPED scoped | support masks match live sklearn 1.5.2 oracle for `f_classif`; `tests/divergence_generic_univariate_select.rs` |
//! | REQ-3 `scores_`/`pvalues_` fitted data | SHIPPED scoped | [`FittedGenericUnivariateSelect::scores`] / [`FittedGenericUnivariateSelect::p_values`] expose arrays computed at fit |
//! | REQ-4 `SelectorMixin` dense helpers | SHIPPED scoped / residual open | [`crate::SelectorMixin`] supplies support masks, inverse zero-fill, and feature-name filtering |
//! | REQ-5 full sklearn score_func/callable/sparse/Python protocol | NOT-STARTED | only [`ScoreFunc::FClassif`] and dense `Array2`; no callable `score_func`, sparse/pandas, PyO3, sklearn error ABI |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::feature_scoring::f_classif;
use crate::feature_selection::ScoreFunc;

/// Selection strategy used by [`GenericUnivariateSelect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenericUnivariateMode {
    /// Keep features above the requested score percentile.
    #[default]
    Percentile,
    /// Keep the `k` highest-scoring features.
    KBest,
    /// Keep features with p-value below alpha.
    Fpr,
    /// Benjamini-Hochberg false-discovery-rate control.
    Fdr,
    /// Bonferroni family-wise-error control.
    Fwe,
}

/// Mode parameter used by [`GenericUnivariateSelect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GenericUnivariateParam {
    /// sklearn's `"all"` sentinel. Only meaningful for `k_best`.
    All,
    /// Numeric mode parameter: percentile, `k`, or alpha depending on mode.
    Value(f64),
}

impl Default for GenericUnivariateParam {
    fn default() -> Self {
        Self::Value(1e-5)
    }
}

/// Configurable univariate feature selector.
#[must_use]
#[derive(Debug, Clone)]
pub struct GenericUnivariateSelect<F> {
    score_func: ScoreFunc,
    mode: GenericUnivariateMode,
    param: GenericUnivariateParam,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Float + Send + Sync + 'static> GenericUnivariateSelect<F> {
    /// Create a new generic univariate selector.
    pub fn new(
        score_func: ScoreFunc,
        mode: GenericUnivariateMode,
        param: GenericUnivariateParam,
    ) -> Self {
        Self {
            score_func,
            mode,
            param,
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the scoring function.
    #[must_use]
    pub fn score_func(&self) -> ScoreFunc {
        self.score_func
    }

    /// Return the selection mode.
    #[must_use]
    pub fn mode(&self) -> GenericUnivariateMode {
        self.mode
    }

    /// Return the mode parameter.
    #[must_use]
    pub fn param(&self) -> GenericUnivariateParam {
        self.param
    }
}

impl<F: Float + Send + Sync + 'static> Default for GenericUnivariateSelect<F> {
    fn default() -> Self {
        Self::new(
            ScoreFunc::FClassif,
            GenericUnivariateMode::Percentile,
            GenericUnivariateParam::default(),
        )
    }
}

/// Fitted generic univariate selector.
#[derive(Debug, Clone)]
pub struct FittedGenericUnivariateSelect<F> {
    n_features_in: usize,
    scores: Array1<F>,
    p_values: Array1<F>,
    selected_indices: Vec<usize>,
    mode: GenericUnivariateMode,
    param: GenericUnivariateParam,
}

impl<F: Float + Send + Sync + 'static> FittedGenericUnivariateSelect<F> {
    /// Return per-feature scores.
    #[must_use]
    pub fn scores(&self) -> &Array1<F> {
        &self.scores
    }

    /// Return per-feature p-values.
    #[must_use]
    pub fn p_values(&self) -> &Array1<F> {
        &self.p_values
    }

    /// Return selected feature indices in original input-column order.
    #[must_use]
    pub fn selected_indices(&self) -> &[usize] {
        &self.selected_indices
    }

    /// Return the number of selected features.
    #[must_use]
    pub fn n_features_selected(&self) -> usize {
        self.selected_indices.len()
    }

    /// Return the fitted selection mode.
    #[must_use]
    pub fn mode(&self) -> GenericUnivariateMode {
        self.mode
    }

    /// Return the fitted mode parameter.
    #[must_use]
    pub fn param(&self) -> GenericUnivariateParam {
        self.param
    }
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>>
    for GenericUnivariateSelect<F>
{
    type Fitted = FittedGenericUnivariateSelect<F>;
    type Error = FerroError;

    /// Fit by computing scores/p-values, then applying the configured mode.
    ///
    /// # Errors
    ///
    /// Propagates scoring errors and returns [`FerroError::InvalidParameter`]
    /// when `param` is invalid for the chosen mode.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<usize>,
    ) -> Result<FittedGenericUnivariateSelect<F>, FerroError> {
        let (scores, p_values) = match self.score_func {
            ScoreFunc::FClassif => f_classif(x, y)?,
        };

        let selected_indices = match self.mode {
            GenericUnivariateMode::Percentile => {
                select_percentile(scores.as_slice().unwrap_or(&[]), numeric_param(self.param)?)?
            }
            GenericUnivariateMode::KBest => select_k_best(
                scores.as_slice().unwrap_or(&[]),
                k_param(self.param, scores.len())?,
            ),
            GenericUnivariateMode::Fpr => select_fpr(
                p_values.as_slice().unwrap_or(&[]),
                numeric_param(self.param)?,
            )?,
            GenericUnivariateMode::Fdr => select_fdr(
                p_values.as_slice().unwrap_or(&[]),
                numeric_param(self.param)?,
            )?,
            GenericUnivariateMode::Fwe => select_fwe(
                p_values.as_slice().unwrap_or(&[]),
                numeric_param(self.param)?,
            )?,
        };

        Ok(FittedGenericUnivariateSelect {
            n_features_in: x.ncols(),
            scores,
            p_values,
            selected_indices,
            mode: self.mode,
            param: self.param,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedGenericUnivariateSelect<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Return a matrix containing only the selected columns.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the input column count differs
    /// from the number of features seen during fit.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        if x.ncols() != self.n_features_in {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), self.n_features_in],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedGenericUnivariateSelect::transform".into(),
            });
        }
        Ok(select_columns(x, &self.selected_indices))
    }
}

fn numeric_param(param: GenericUnivariateParam) -> Result<f64, FerroError> {
    match param {
        GenericUnivariateParam::Value(value) => Ok(value),
        GenericUnivariateParam::All => Err(FerroError::InvalidParameter {
            name: "param".into(),
            reason: "'all' is only valid for k_best mode".into(),
        }),
    }
}

fn k_param(param: GenericUnivariateParam, n_features: usize) -> Result<usize, FerroError> {
    match param {
        GenericUnivariateParam::All => Ok(n_features),
        GenericUnivariateParam::Value(value) => {
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return Err(FerroError::InvalidParameter {
                    name: "param".into(),
                    reason: "k_best param must be a non-negative integer or 'all'".into(),
                });
            }
            Ok(value as usize)
        }
    }
}

fn validate_percentile(percentile: f64) -> Result<(), FerroError> {
    if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
        return Err(FerroError::InvalidParameter {
            name: "param".into(),
            reason: "percentile param must be in [0, 100]".into(),
        });
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> Result<(), FerroError> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(FerroError::InvalidParameter {
            name: "param".into(),
            reason: "alpha param must be in [0, 1]".into(),
        });
    }
    Ok(())
}

fn select_k_best<F: Float>(scores: &[F], k: usize) -> Vec<usize> {
    let n_features = scores.len();
    let k_eff = k.min(n_features);
    let cleaned: Vec<F> = scores
        .iter()
        .map(|&v| if v.is_nan() { F::min_value() } else { v })
        .collect();
    let mut idx: Vec<usize> = (0..n_features).collect();
    idx.sort_by(|&a, &b| {
        cleaned[a]
            .partial_cmp(&cleaned[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut selected = idx[n_features - k_eff..].to_vec();
    selected.sort_unstable();
    selected
}

fn select_percentile<F: Float>(scores: &[F], percentile: f64) -> Result<Vec<usize>, FerroError> {
    validate_percentile(percentile)?;
    if percentile == 100.0 {
        return Ok((0..scores.len()).collect());
    }
    if percentile == 0.0 {
        return Ok(Vec::new());
    }

    let cleaned: Vec<F> = scores
        .iter()
        .map(|&v| if v.is_nan() { F::min_value() } else { v })
        .collect();
    let mut sorted = cleaned.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let threshold = numpy_percentile(&sorted, 100.0 - percentile);
    let mut mask: Vec<bool> = cleaned.iter().map(|&score| score > threshold).collect();
    let mut ties: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| (cleaned[idx] == threshold).then_some(idx))
        .collect();
    let max_feats = (scores.len() as f64 * percentile / 100.0) as usize;
    let needed = max_feats.saturating_sub(mask.iter().filter(|&&keep| keep).count());
    ties.truncate(needed);
    for idx in ties {
        mask[idx] = true;
    }
    Ok(mask
        .into_iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then_some(idx))
        .collect())
}

fn select_fpr<F: Float>(p_values: &[F], alpha: f64) -> Result<Vec<usize>, FerroError> {
    validate_alpha(alpha)?;
    let alpha_f = F::from(alpha).unwrap_or_else(F::zero);
    Ok(p_values
        .iter()
        .enumerate()
        .filter_map(|(idx, &p)| (p < alpha_f).then_some(idx))
        .collect())
}

fn select_fdr<F: Float>(p_values: &[F], alpha: f64) -> Result<Vec<usize>, FerroError> {
    validate_alpha(alpha)?;
    let n = p_values.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let alpha_f = F::from(alpha).unwrap_or_else(F::zero);
    let n_f = F::from(n).unwrap_or_else(F::one);
    let mut sorted = p_values.to_vec();
    sorted.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal),
    });
    let selected_max = sorted
        .iter()
        .enumerate()
        .filter_map(|(rank, &p)| {
            let rank_f = F::from(rank + 1).unwrap_or_else(F::one);
            (p <= alpha_f * rank_f / n_f).then_some(p)
        })
        .last();

    match selected_max {
        Some(max_p) => Ok(p_values
            .iter()
            .enumerate()
            .filter_map(|(idx, &p)| (p <= max_p).then_some(idx))
            .collect()),
        None => Ok(Vec::new()),
    }
}

fn select_fwe<F: Float>(p_values: &[F], alpha: f64) -> Result<Vec<usize>, FerroError> {
    validate_alpha(alpha)?;
    if p_values.is_empty() {
        return Ok(Vec::new());
    }
    let alpha_f = F::from(alpha).unwrap_or_else(F::zero);
    let n_f = F::from(p_values.len()).unwrap_or_else(F::one);
    let threshold = alpha_f / n_f;
    Ok(p_values
        .iter()
        .enumerate()
        .filter_map(|(idx, &p)| (p < threshold).then_some(idx))
        .collect())
}

fn numpy_percentile<F: Float>(sorted: &[F], q: f64) -> F {
    let n = sorted.len();
    if n == 0 {
        return F::nan();
    }
    if n == 1 {
        return sorted[0];
    }
    let vi = (q / 100.0) * (n - 1) as f64;
    let prev = vi.floor();
    let (lo, hi) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if vi < 0.0 {
        (0, 0)
    } else {
        let lo = prev as usize;
        (lo, lo + 1)
    };
    let t = F::from(vi - prev).unwrap_or_else(F::zero);
    let a = sorted[lo];
    let b = sorted[hi];
    let diff = b - a;
    if t >= F::from(0.5).unwrap_or_else(F::zero) {
        b - diff * (F::one() - t)
    } else {
        a + diff * t
    }
}

fn select_columns<F: Float>(x: &Array2<F>, indices: &[usize]) -> Array2<F> {
    let mut out = Array2::zeros((x.nrows(), indices.len()));
    for (new_j, &old_j) in indices.iter().enumerate() {
        for i in 0..x.nrows() {
            out[[i, new_j]] = x[[i, old_j]];
        }
    }
    out
}
