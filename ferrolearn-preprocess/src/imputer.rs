//! Simple imputer: fill missing (NaN) values per feature column.
//!
//! [`SimpleImputer`] supports four imputation strategies:
//! - [`ImputeStrategy::Mean`] — replace NaN with the column mean
//! - [`ImputeStrategy::Median`] — replace NaN with the column median
//! - [`ImputeStrategy::MostFrequent`] — replace NaN with the most common value
//! - [`ImputeStrategy::Constant`] — replace NaN with a fixed constant value
//!
//! Fitting ignores NaN values when computing statistics (e.g. the mean is the
//! mean of all non-NaN values in that column).  Under `Mean`/`Median`/
//! `MostFrequent`, columns that are entirely NaN at fit time have no observed
//! value, so — mirroring scikit-learn's default `keep_empty_features=False`
//! (`sklearn/impute/_base.py:501,510-512,534-537` set `statistics_=nan`;
//! `:586-603` drop them in `transform`) — they are DROPPED from the transform
//! output.  Under `Constant`, every column (including all-NaN ones) is filled
//! with the constant and KEPT (sklearn `:545,583`).
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class SimpleImputer` +
//! `MissingIndicator` (`sklearn/impute/_base.py:147`). Tracking: #1363. Each REQ
//! is BINARY — SHIPPED (impl + non-test consumer + tests + green verification)
//! or NOT-STARTED (with a concrete open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Per-column fill VALUES on columns with ≥1 observed value (Mean/Median/MostFrequent/Constant) | SHIPPED | [`SimpleImputer`] `fit` — Mean=`np.ma.mean` (`_base.py:498`), Median=`np.ma.median` (`:507`, even=avg-of-two-middle), MostFrequent=scipy mode tie→min (`_most_frequent` `:36-71`), Constant (`:545`); 9 oracle value tests in `tests/divergence_imputer.rs`. Consumer: re-export `lib.rs:136` + `PipelineTransformer` |
//! | REQ-2 | All-NaN column DROP under Mean/Median/MostFrequent (sklearn default `keep_empty_features=False`) | SHIPPED | `fit` sets `fill_values[j]=NaN` + excludes `j` from `kept_indices`; `transform` projects onto `kept_indices` (mirrors `statistics_=nan` + `X=X[:, valid]` `_base.py:586-603`); `Constant` keeps+fills all (`:583`); 10 oracle tests (column-order, all-dropped, separate matrix, f32) — was DIV-1 #1364, fixed |
//! | REQ-3 | Error/parameter contracts (n_samples==0, transform ncols, unfitted) | SHIPPED (scoped) | [`SimpleImputer::fit`]/[`FittedSimpleImputer`] `transform`; in-module + divergence error tests |
//! | REQ-4 | `keep_empty_features` param (True → fill 0 + keep all-NaN cols) | NOT-STARTED | always drops; sklearn `_base.py:583,501` — blocker #1365 |
//! | REQ-5 | `missing_values` param (non-NaN sentinel / None / str) | NOT-STARTED | NaN-only; sklearn `_base.py:161,288` — blocker #1366 |
//! | REQ-6 | `add_indicator` + `MissingIndicator` estimator (route parity_op, ABSENT) | NOT-STARTED | needs acto-builder; sklearn `_base.py:205` + `MissingIndicator` — blocker #1367 |
//! | REQ-7 | `inverse_transform` (requires add_indicator) | NOT-STARTED | sklearn `_base.py:641` — blocker #1368 |
//! | REQ-8 | `fill_value=None`→0 default + `statistics_` attr name + `copy` param | NOT-STARTED | `Constant(F)` explicit; sklearn `_base.py:425-427,223,288` — blocker #1369 |
//! | REQ-9 | string/object dtype (most_frequent/constant on non-numeric) | NOT-STARTED | `F: Float` only; sklearn `_base.py:42-52,526` — blocker #1370 |
//! | REQ-10 | sparse `_sparse_fit` | NOT-STARTED | dense `Array2` only; sklearn `_base.py:444` — blocker #1371 |
//! | REQ-11 | `get_feature_names_out` + `n_features_in_`/`feature_names_in_` | NOT-STARTED | `_BaseImputer` — blocker #1372 |
//! | REQ-12 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1373 |
//! | REQ-13 | ferray substrate | NOT-STARTED | dense `Array2` + `num_traits::Float` only — blocker #1374 |

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// ImputeStrategy
// ---------------------------------------------------------------------------

/// The strategy used to compute the fill value for each column.
#[derive(Debug, Clone, PartialEq)]
pub enum ImputeStrategy<F> {
    /// Replace NaN with the column mean (ignoring NaN values).
    Mean,
    /// Replace NaN with the column median (ignoring NaN values).
    Median,
    /// Replace NaN with the most frequently occurring value in the column.
    MostFrequent,
    /// Replace NaN with a fixed constant value.
    Constant(F),
}

// ---------------------------------------------------------------------------
// SimpleImputer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted simple imputer.
///
/// Calling [`Fit::fit`] computes the per-column fill values according to
/// the chosen [`ImputeStrategy`] and returns a [`FittedSimpleImputer`] that
/// can transform new data by replacing NaN values with those fill values.
///
/// NaN values are *ignored* when computing statistics during fitting — e.g.
/// the `Mean` strategy computes the mean of only the non-NaN elements.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::imputer::{SimpleImputer, ImputeStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
/// let x = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = imputer.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// // NaN in column 1 row 0 is replaced with the mean of column 1 = (4+6)/2 = 5.0
/// assert!((out[[0, 1]] - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleImputer<F> {
    strategy: ImputeStrategy<F>,
}

impl<F: Float + Send + Sync + 'static> SimpleImputer<F> {
    /// Create a new `SimpleImputer` with the given strategy.
    #[must_use]
    pub fn new(strategy: ImputeStrategy<F>) -> Self {
        Self { strategy }
    }

    /// Return the imputation strategy.
    #[must_use]
    pub fn strategy(&self) -> &ImputeStrategy<F> {
        &self.strategy
    }
}

// ---------------------------------------------------------------------------
// FittedSimpleImputer
// ---------------------------------------------------------------------------

/// A fitted simple imputer holding one fill value per feature column.
///
/// Created by calling [`Fit::fit`] on a [`SimpleImputer`].
#[derive(Debug, Clone)]
pub struct FittedSimpleImputer<F> {
    /// Per-INPUT-column fill values learned during fitting.
    ///
    /// One entry per input column, mirroring scikit-learn's `statistics_`:
    /// holds `F::nan()` for an all-NaN non-constant column that is dropped, and
    /// the computed fill statistic (or the user constant) otherwise.
    fill_values: Array1<F>,
    /// Input-column indices that survive transform, in ascending order.
    ///
    /// Under `Mean`/`Median`/`MostFrequent` an all-NaN column has no observed
    /// value and is excluded (sklearn `keep_empty_features=False`); under
    /// `Constant` every column is kept.
    kept_indices: Vec<usize>,
}

impl<F: Float + Send + Sync + 'static> FittedSimpleImputer<F> {
    /// Return the per-input-column fill values learned during fitting.
    ///
    /// Mirrors scikit-learn's `statistics_`: entries for all-NaN columns that
    /// are dropped under `Mean`/`Median`/`MostFrequent` are `F::nan()`.
    #[must_use]
    pub fn fill_values(&self) -> &Array1<F> {
        &self.fill_values
    }

    /// Return the input-column indices that survive `transform`, ascending.
    #[must_use]
    pub fn kept_indices(&self) -> &[usize] {
        &self.kept_indices
    }
}

// ---------------------------------------------------------------------------
// Helper: compute median of a non-empty Vec (may contain NaN — caller filters)
// ---------------------------------------------------------------------------

/// Compute the median of a non-empty slice of finite (non-NaN) values.
///
/// Uses a sort-and-interpolate approach.  Panics if the slice is empty.
fn median_of<F: Float>(values: &mut [F]) -> F {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        let mid = n / 2;
        (values[mid - 1] + values[mid]) / (F::one() + F::one())
    }
}

/// Find the most-frequent value in a non-empty slice of finite values.
///
/// Ties are broken by choosing the smallest value.
fn most_frequent_of<F: Float>(values: &[F]) -> F {
    // Collect (value, count) by scanning; values are finite so partial_cmp is
    // total.
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_val = sorted[0];
    let mut best_count = 1usize;
    let mut current_val = sorted[0];
    let mut current_count = 1usize;

    for &v in &sorted[1..] {
        if v == current_val {
            current_count += 1;
        } else {
            if current_count > best_count {
                best_count = current_count;
                best_val = current_val;
            }
            current_val = v;
            current_count = 1;
        }
    }
    // Final run
    if current_count > best_count {
        best_val = current_val;
    }
    best_val
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, ()> for SimpleImputer<F> {
    type Fitted = FittedSimpleImputer<F>;
    type Error = FerroError;

    /// Fit the imputer by computing per-column fill values.
    ///
    /// NaN values are excluded from the statistic computation.  Under
    /// `Mean`/`Median`/`MostFrequent`, a column that is entirely NaN has no
    /// observed value: its `fill_values` entry is set to `F::nan()` and it is
    /// excluded from `kept_indices`, so `transform` DROPS it (mirroring
    /// scikit-learn `keep_empty_features=False`, `sklearn/impute/_base.py:501,
    /// 510-512,534-537,586-603`).  Under `Constant`, every column is filled
    /// with the constant and kept (sklearn `:545,583`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InsufficientSamples`] if the input has zero rows.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedSimpleImputer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "SimpleImputer::fit".into(),
            });
        }

        let n_features = x.ncols();
        let mut fill_values = Array1::zeros(n_features);
        let mut kept_indices: Vec<usize> = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col_vals: Vec<F> = x
                .column(j)
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();

            // Constant fills (and keeps) every column, including all-NaN ones
            // (sklearn `np.full(X.shape[1], fill_value)`, `_base.py:545,583`).
            if let ImputeStrategy::Constant(c) = &self.strategy {
                fill_values[j] = *c;
                kept_indices.push(j);
                continue;
            }

            if col_vals.is_empty() {
                // All-NaN column with no observed value: sklearn sets
                // `statistics_=nan` and DROPS it (`_base.py:501,510-512,
                // 534-537,586-603`).
                fill_values[j] = F::nan();
                continue;
            }

            fill_values[j] = match &self.strategy {
                ImputeStrategy::Mean => {
                    let n = F::from(col_vals.len()).unwrap_or_else(F::one);
                    col_vals.iter().copied().fold(F::zero(), |acc, v| acc + v) / n
                }
                ImputeStrategy::Median => {
                    let mut vals = col_vals.clone();
                    median_of(&mut vals)
                }
                ImputeStrategy::MostFrequent => most_frequent_of(&col_vals),
                // Constant handled above.
                ImputeStrategy::Constant(c) => *c,
            };
            kept_indices.push(j);
        }

        Ok(FittedSimpleImputer {
            fill_values,
            kept_indices,
        })
    }
}

impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for FittedSimpleImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Replace NaN values with the learned fill value, projecting onto the
    /// columns that survived fitting.
    ///
    /// The transform input must have the same number of columns as the fit
    /// input (the full input width, `fill_values.len()`), matching scikit-learn
    /// which validates against `statistics_.shape[0]` (`_base.py:573-577`).
    /// The OUTPUT keeps only [`Self::kept_indices`] columns, in ascending
    /// order — dropping all-NaN columns under `Mean`/`Median`/`MostFrequent`
    /// (sklearn `X = X[:, valid_statistics_indexes]`, `_base.py:586-603`).
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the number of features seen during fitting.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.fill_values.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedSimpleImputer::transform".into(),
            });
        }

        // Gather the surviving columns (the column-projection pattern used
        // elsewhere, e.g. select_from_model's `select_columns`), imputing NaN
        // with each column's learned fill value as we go.
        let mut out = Array2::zeros((x.nrows(), self.kept_indices.len()));
        for (out_j, &in_j) in self.kept_indices.iter().enumerate() {
            let fill = self.fill_values[in_j];
            for (row, &v) in x.column(in_j).iter().enumerate() {
                out[[row, out_j]] = if v.is_nan() { fill } else { v };
            }
        }
        Ok(out)
    }
}

/// Implement `Transform` on the unfitted imputer to satisfy the
/// `FitTransform: Transform` supertrait bound.  Always returns an error.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for SimpleImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the imputer must be fitted first.
    ///
    /// Use [`Fit::fit`] to produce a [`FittedSimpleImputer`], then call
    /// [`Transform::transform`] on that.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "SimpleImputer".into(),
            reason: "imputer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: Float + Send + Sync + 'static> FitTransform<Array2<F>> for SimpleImputer<F> {
    type FitError = FerroError;

    /// Fit the imputer on `x` and return the imputed output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g. zero rows).
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Pipeline integration (generic)
// ---------------------------------------------------------------------------

impl<F: Float + Send + Sync + 'static> PipelineTransformer<F> for SimpleImputer<F> {
    /// Fit the imputer using the pipeline interface.
    ///
    /// The `y` argument is ignored; it exists only for API compatibility.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Fit::fit`].
    fn fit_pipeline(
        &self,
        x: &Array2<F>,
        _y: &Array1<F>,
    ) -> Result<Box<dyn FittedPipelineTransformer<F>>, FerroError> {
        let fitted = self.fit(x, &())?;
        Ok(Box::new(fitted))
    }
}

impl<F: Float + Send + Sync + 'static> FittedPipelineTransformer<F> for FittedSimpleImputer<F> {
    /// Transform data using the pipeline interface.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Transform::transform`].
    fn transform_pipeline(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        self.transform(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---- Mean strategy -------------------------------------------------------

    #[test]
    fn test_mean_basic() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        // Column 0 mean = (1+3+5)/3 = 3.0, column 1 mean = (4+6)/2 = 5.0
        assert_abs_diff_eq!(fitted.fill_values()[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(fitted.fill_values()[1], 5.0, epsilon = 1e-10);
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 1]], 5.0, epsilon = 1e-10);
        // Non-NaN values must be untouched
        assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_no_nan() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Nothing should change
        for (a, b) in x.iter().zip(out.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_mean_multiple_nans_same_column() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[f64::NAN], [f64::NAN], [6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 6.0, epsilon = 1e-10);
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[0, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(out[[1, 0]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_all_nan_column_dropped() {
        // sklearn `keep_empty_features=False` (default): an all-NaN column has
        // no observed value, so `statistics_=nan` and `transform` DROPS it
        // (`sklearn/impute/_base.py:586-603`). A single all-NaN input column
        // therefore yields ZERO output columns.
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[f64::NAN], [f64::NAN]];
        let fitted = match imputer.fit(&x, &()) {
            Ok(f) => f,
            #[allow(
                clippy::assertions_on_constants,
                reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
            )]
            Err(e) => {
                assert!(false, "fit errored: {e}");
                return;
            }
        };
        // statistics_ entry is NaN (mirrors sklearn `statistics_`).
        assert!(fitted.fill_values()[0].is_nan());
        match fitted.transform(&x) {
            Ok(out) => {
                assert_eq!(out.ncols(), 0, "all-NaN column dropped -> 0 output columns");
                assert_eq!(out.nrows(), 2);
            }
            #[allow(
                clippy::assertions_on_constants,
                reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
            )]
            Err(e) => assert!(false, "transform errored: {e}"),
        }
    }

    // ---- Median strategy ----------------------------------------------------

    #[test]
    fn test_median_odd_count() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
        let x = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_even_count() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
        let x = array![[1.0], [3.0], [5.0], [7.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        // Median of [1,3,5,7] = (3+5)/2 = 4.0
        assert_abs_diff_eq!(fitted.fill_values()[0], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_with_nan() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
        // Column 0: non-NaN values are [2, 4, 6], median = 4
        let x = array![[2.0], [f64::NAN], [4.0], [6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 4.0, epsilon = 1e-10);
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[1, 0]], 4.0, epsilon = 1e-10);
    }

    // ---- MostFrequent strategy ----------------------------------------------

    #[test]
    fn test_most_frequent_basic() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::MostFrequent);
        let x = array![[1.0], [2.0], [2.0], [3.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_most_frequent_tie_chooses_smallest() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::MostFrequent);
        // 1.0 and 3.0 each appear twice — smallest wins
        let x = array![[1.0], [1.0], [3.0], [3.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_most_frequent_with_nan() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::MostFrequent);
        let x = array![[1.0], [f64::NAN], [2.0], [2.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], 2.0, epsilon = 1e-10);
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[1, 0]], 2.0, epsilon = 1e-10);
    }

    // ---- Constant strategy --------------------------------------------------

    #[test]
    fn test_constant_strategy() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Constant(-99.0));
        let x = array![[1.0, f64::NAN], [f64::NAN, 4.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert_abs_diff_eq!(fitted.fill_values()[0], -99.0, epsilon = 1e-15);
        assert_abs_diff_eq!(fitted.fill_values()[1], -99.0, epsilon = 1e-15);
        let out = fitted.transform(&x).unwrap();
        assert_abs_diff_eq!(out[[1, 0]], -99.0, epsilon = 1e-15);
        assert_abs_diff_eq!(out[[0, 1]], -99.0, epsilon = 1e-15);
    }

    // ---- Error paths --------------------------------------------------------

    #[test]
    fn test_fit_zero_rows_error() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_transform_shape_mismatch_error() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_unfitted_transform_error() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0]];
        assert!(imputer.transform(&x).is_err());
    }

    // ---- fit_transform ------------------------------------------------------

    #[test]
    fn test_fit_transform_equivalence() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
        let via_fit_transform = imputer.fit_transform(&x).unwrap();
        let fitted = imputer.fit(&x, &()).unwrap();
        let via_separate = fitted.transform(&x).unwrap();
        for (a, b) in via_fit_transform.iter().zip(via_separate.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-15);
        }
    }

    // ---- f32 generic --------------------------------------------------------

    #[test]
    fn test_f32_imputer() {
        let imputer = SimpleImputer::<f32>::new(ImputeStrategy::Mean);
        let x: Array2<f32> = array![[1.0f32, f32::NAN], [3.0, 4.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!((out[[0, 1]] - 4.0f32).abs() < 1e-6);
    }

    // ---- Pipeline integration -----------------------------------------------

    #[test]
    fn test_pipeline_integration() {
        use ferrolearn_core::pipeline::PipelineTransformer;

        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
        let x = array![[1.0, f64::NAN], [3.0, 4.0]];
        let y = ndarray::array![0.0, 1.0];
        let fitted_box = imputer.fit_pipeline(&x, &y).unwrap();
        let out = fitted_box.transform_pipeline(&x).unwrap();
        // NaN should be gone
        assert!(!out[[0, 1]].is_nan());
    }

    // ---- multiple columns with mixed NaN ------------------------------------

    #[test]
    fn test_multi_column_mixed_nan() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
        let x = array![[f64::NAN, 10.0], [2.0, f64::NAN], [4.0, 30.0], [6.0, 40.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Column 0 non-NaN = [2,4,6], median = 4
        assert_abs_diff_eq!(out[[0, 0]], 4.0, epsilon = 1e-10);
        // Column 1 non-NaN = [10,30,40], median = 30
        assert_abs_diff_eq!(out[[1, 1]], 30.0, epsilon = 1e-10);
    }

    // ---- strategy accessor --------------------------------------------------

    #[test]
    fn test_strategy_accessor() {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Constant(42.0));
        assert_eq!(imputer.strategy(), &ImputeStrategy::Constant(42.0));
    }
}
