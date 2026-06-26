//! Shared feature-selector mixin surface.
//!
//! This is the dense `Array2` analogue of scikit-learn's
//! `feature_selection._base.SelectorMixin`: implementations provide the number
//! of input features and the retained column indices, and the trait supplies
//! support masks, support indices, inverse transformation, and selected feature
//! names.

use ferrolearn_core::error::FerroError;
use ndarray::Array2;
use num_traits::Float;

use crate::feature_selection::{FittedSelectKBest, FittedVarianceThreshold, SelectFromModel};
use crate::rfe::{RFE, RFECV};
use crate::select_from_model::FittedSelectFromModelExt;
use crate::select_percentile::FittedSelectPercentile;
use crate::sequential_feature_selector::FittedSequentialFeatureSelector;
use crate::stat_selectors::{FittedSelectFdr, FittedSelectFpr, FittedSelectFwe};

/// Dense feature-selector mixin for fitted selector states.
///
/// This trait mirrors the useful dense parts of sklearn's `SelectorMixin`:
/// `get_support()`, `get_support(indices=True)`, `inverse_transform`, and
/// `get_feature_names_out`. Sparse matrices, pandas output, estimator tags, and
/// Python fitted-state checks remain outside ferrolearn's Rust typestate model.
pub trait SelectorMixin<F: Float> {
    /// Number of input features seen by the selector.
    fn n_features_in(&self) -> usize;

    /// Selected feature indices in original input-column order.
    fn selected_indices(&self) -> &[usize];

    /// Return a boolean mask of retained features.
    #[must_use]
    fn get_support(&self) -> Vec<bool> {
        let mut mask = vec![false; self.n_features_in()];
        for &idx in self.selected_indices() {
            if let Some(slot) = mask.get_mut(idx) {
                *slot = true;
            }
        }
        mask
    }

    /// Return retained feature indices, matching sklearn
    /// `get_support(indices=True)`.
    #[must_use]
    fn get_support_indices(&self) -> Vec<usize> {
        self.get_support()
            .into_iter()
            .enumerate()
            .filter_map(|(idx, keep)| keep.then_some(idx))
            .collect()
    }

    /// Scatter a reduced matrix back to the original feature width.
    ///
    /// Removed columns are filled with zeros, matching sklearn's dense
    /// `SelectorMixin.inverse_transform`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when `x.ncols()` is not the number
    /// of selected features.
    fn inverse_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let selected = self.get_support_indices();
        if x.ncols() != selected.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), selected.len()],
                actual: vec![x.nrows(), x.ncols()],
                context: "SelectorMixin::inverse_transform".into(),
            });
        }

        let mut out = Array2::zeros((x.nrows(), self.n_features_in()));
        for (src_j, &dst_j) in selected.iter().enumerate() {
            for i in 0..x.nrows() {
                out[[i, dst_j]] = x[[i, src_j]];
            }
        }
        Ok(out)
    }

    /// Return selected feature names.
    ///
    /// When `input_features` is `None`, names are generated as `x0`, `x1`, ...
    /// like sklearn does when `feature_names_in_` is absent.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] when `input_features` length does
    /// not match [`Self::n_features_in`].
    fn get_feature_names_out(
        &self,
        input_features: Option<&[String]>,
    ) -> Result<Vec<String>, FerroError> {
        let n_features = self.n_features_in();
        let names: Vec<String> = match input_features {
            Some(features) => {
                if features.len() != n_features {
                    return Err(FerroError::ShapeMismatch {
                        expected: vec![n_features],
                        actual: vec![features.len()],
                        context: "SelectorMixin::get_feature_names_out".into(),
                    });
                }
                features.to_vec()
            }
            None => (0..n_features).map(|idx| format!("x{idx}")).collect(),
        };

        Ok(self
            .get_support_indices()
            .into_iter()
            .map(|idx| names[idx].clone())
            .collect())
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedVarianceThreshold<F> {
    fn n_features_in(&self) -> usize {
        self.variances().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectKBest<F> {
    fn n_features_in(&self) -> usize {
        self.scores().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for SelectFromModel<F> {
    fn n_features_in(&self) -> usize {
        self.importances().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectPercentile<F> {
    fn n_features_in(&self) -> usize {
        self.scores().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectFpr<F> {
    fn n_features_in(&self) -> usize {
        self.p_values().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectFdr<F> {
    fn n_features_in(&self) -> usize {
        self.p_values().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectFwe<F> {
    fn n_features_in(&self) -> usize {
        self.p_values().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSelectFromModelExt<F> {
    fn n_features_in(&self) -> usize {
        self.importances().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for RFE<F> {
    fn n_features_in(&self) -> usize {
        self.support().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for RFECV<F> {
    fn n_features_in(&self) -> usize {
        self.support().len()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}

impl<F: Float + Send + Sync + 'static> SelectorMixin<F> for FittedSequentialFeatureSelector<F> {
    fn n_features_in(&self) -> usize {
        self.n_features_in()
    }

    fn selected_indices(&self) -> &[usize] {
        self.selected_indices()
    }
}
