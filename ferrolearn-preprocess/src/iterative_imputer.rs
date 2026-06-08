//! Iterative imputer (MICE): fill missing values by modeling each feature as a
//! regression on all the other features, in a round-robin fashion.
//!
//! [`IterativeImputer`] mirrors scikit-learn's `IterativeImputer`
//! (`sklearn/impute/_iterative.py:51`, EXPERIMENTAL). For each feature with
//! missing values — visited in `imputation_order` (default `Ascending`,
//! fewest-missing first) — it fits the default per-feature estimator
//! [`ferrolearn_linear::BayesianRidge`] on the rows where the feature is
//! OBSERVED (predictors = the other features' current filled values), predicts
//! the missing rows, and clips the predictions to `[min_value, max_value]`. The
//! round-robin repeats for `max_iter` rounds or until the inf-norm of the change
//! falls below `tol * max|X_observed|`.
//!
//! # Initial Imputation
//!
//! Before the iterative process begins, missing values are filled using a simple
//! strategy (mean by default). This initial imputation provides the starting
//! point for the regression models (sklearn `_initial_imputation`,
//! `_iterative.py:743`).
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `class IterativeImputer`
//! (`sklearn/impute/_iterative.py:51`, EXPERIMENTAL). Tracking: #1403. Each REQ
//! is BINARY — SHIPPED (impl + non-test consumer + tests + green verification)
//! or NOT-STARTED (with a concrete open blocker). The round-robin now routes
//! through the real `BayesianRidge` default estimator with ascending order,
//! inf-norm convergence, and min/max clip, so the iterated imputed VALUES match
//! sklearn (~1e-6).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | Round-robin STRUCTURE + initial fill (mean/median == `SimpleImputer`) + non-missing values preserved + output shape | SHIPPED | [`IterativeImputer`] `fit`/`transform`; initial-fill VALUES match `SimpleImputer` (oracle); non-missing-preserved + shape + no-NaN tests in `tests/divergence_iterative_imputer.rs`. Consumer: re-export `lib.rs:150` |
//! | REQ-2 | Determinism (no RNG) + termination (≤ max_iter, tol break) + `n_iter` accessor | SHIPPED | [`FittedIterativeImputer::n_iter`]; determinism + bounded-termination tests |
//! | REQ-3 | Error/parameter contracts (n_samples==0, transform ncols, unfitted) + `max_iter==0` → initial fill | SHIPPED | [`IterativeImputer::fit`] accepts `max_iter==0` (returns initial fill, `n_iter=0`, matches sklearn `_iterative.py:750-752`); divergence + error tests |
//! | REQ-4 | Exact iterated imputed-VALUE parity (BayesianRidge default + ascending order + inf-norm tol + min/max clip) | SHIPPED | round-robin routes through [`ferrolearn_linear::BayesianRidge`] (`fn impute_one_feature`), ascending order, inf-norm convergence, min/max clip — matches sklearn `_iterative.py:454,732-735` (~1e-6). Verified by `divergence_round_robin_values_small`/`_three_features`/`min_max_clip_bound` (`tests/divergence_iterative_imputer_values.rs`, live sklearn oracle). Closes #1405 |
//! | REQ-5 | `estimator` param (pluggable; default BayesianRidge) + `sample_posterior` | SHIPPED (default only) | default per-feature estimator is `BayesianRidge` (`fn impute_one_feature`, sklearn `_iterative.py:74,732-735`); pluggable `estimator` + `sample_posterior` posterior sampling stay NOT-STARTED — blocker #1406 |
//! | REQ-6 | `imputation_order` param (ascending/descending/roman/arabic/random) | SHIPPED (ascending/descending/roman/arabic) | [`ImputationOrder`] enum + `imputation_order` field; default `Ascending` orders features by ascending missing-count (sklearn `_get_ordered_idx:533-535`). `Random` (RNG) stays NOT-STARTED — blocker #1407 |
//! | REQ-7 | `min_value`/`max_value` clipping | SHIPPED | `min_value`/`max_value` fields (default ±inf); per-iteration clip in `fn impute_one_feature` (sklearn `_iterative.py:455-457`); array-like per-feature limits stay scalar-only — blocker #1408 |
//! | REQ-8 | `n_nearest_features` + abs-correlation feature selection | NOT-STARTED | uses all other features; sklearn `:468-502` — blocker #1409 |
//! | REQ-9 | `initial_strategy` most_frequent/constant + `fill_value` + non-NaN `missing_values` | NOT-STARTED | Mean/Median only, NaN-only; sklearn `:112,183,743` — blocker #1410 |
//! | REQ-10 | `random_state` + `skip_complete` + `add_indicator` + `keep_empty_features` + `verbose` | NOT-STARTED | sklearn `:305-343` — blocker #1411 |
//! | REQ-11 | inf-norm convergence (`tol·max\|X\|`) | SHIPPED | `fn fit`/`fn transform` converge on `max\|Xt-Xt_prev\| < tol·max\|X_observed\|` (sklearn `_iterative.py:780,811,818`); `ConvergenceWarning` emission stays NOT-STARTED — blocker #1412 |
//! | REQ-12 | `get_feature_names_out` + `imputation_sequence_`/`n_iter_`/`n_features_in_`/`random_state_` attrs | NOT-STARTED | sklearn `:739` — blocker #1413 |
//! | REQ-13 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1414 |
//! | REQ-14 | ferray substrate | NOT-STARTED | dense `Array2` + `num_traits::Float` only — blocker #1415 |

use ferray::linalg::LinalgFloat;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, FitTransform, Predict, Transform};
use ferrolearn_linear::{BayesianRidge, FittedBayesianRidge};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

/// Float bound under which the round-robin can route through
/// [`ferrolearn_linear::BayesianRidge`]: the exact bound `BayesianRidge::fit` /
/// `FittedBayesianRidge::predict` require (`LinalgFloat + ScalarOperand +
/// FromPrimitive`, sklearn default estimator `_iterative.py:732-735`). A blanket
/// impl provides it for every such type (`f32`/`f64`), so the rest of the file
/// names a single tidy bound. The `ferray::linalg::LinalgFloat` bound enters via
/// `BayesianRidge` (which already sits on the ferray SVD substrate); the
/// `IterativeImputer`'s OWN compute remains `ndarray`/`num_traits` (REQ-14
/// substrate migration stays NOT-STARTED — blocker #1415).
pub trait ImputerFloat:
    LinalgFloat + ScalarOperand + FromPrimitive + Send + Sync + 'static
{
}

impl<F> ImputerFloat for F where
    F: LinalgFloat + ScalarOperand + FromPrimitive + Send + Sync + 'static
{
}

// ---------------------------------------------------------------------------
// InitialStrategy
// ---------------------------------------------------------------------------

/// Strategy for the initial imputation before iterative refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialStrategy {
    /// Replace NaN with the column mean.
    Mean,
    /// Replace NaN with the column median.
    Median,
}

// ---------------------------------------------------------------------------
// ImputationOrder
// ---------------------------------------------------------------------------

/// Order in which features are imputed each round, mirroring scikit-learn's
/// `imputation_order` (`sklearn/impute/_iterative.py:126-134`,
/// `_get_ordered_idx:504-542`). The `'random'` variant (which requires a seeded
/// RNG, `random_state`) is not yet modeled — blocker #1407.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationOrder {
    /// From features with fewest missing values to most (sklearn default,
    /// `argsort(frac_of_missing, kind="mergesort")`, ties keep column order).
    Ascending,
    /// From features with most missing values to fewest (reversed ascending).
    Descending,
    /// Left to right (column-index order, sklearn `'roman'`).
    Roman,
    /// Right to left (reversed column-index order, sklearn `'arabic'`).
    Arabic,
}

// ---------------------------------------------------------------------------
// IterativeImputer (unfitted)
// ---------------------------------------------------------------------------

/// An unfitted iterative imputer.
///
/// Calling [`Fit::fit`] learns the per-feature [`BayesianRidge`] models and
/// returns a [`FittedIterativeImputer`] that can impute missing values in new
/// data by deterministically replaying the learned imputation sequence.
///
/// # Parameters
///
/// - `max_iter` — maximum number of imputation rounds (default 10).
/// - `tol` — convergence tolerance on the inf-norm of the change scaled by
///   `max|X_observed|` (default 1e-3).
/// - `initial_strategy` — strategy for the initial fill (default `Mean`).
/// - `imputation_order` — feature visit order (default `Ascending`).
/// - `min_value` / `max_value` — imputed-value clip bounds (default ±∞).
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::iterative_imputer::{IterativeImputer, InitialStrategy};
/// use ferrolearn_core::traits::{Fit, Transform};
/// use ndarray::array;
///
/// let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
/// let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
/// let fitted = imputer.fit(&x, &()).unwrap();
/// let out = fitted.transform(&x).unwrap();
/// assert!(!out[[1, 1]].is_nan());
/// assert!(!out[[2, 0]].is_nan());
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct IterativeImputer<F> {
    /// Maximum number of imputation rounds.
    max_iter: usize,
    /// Convergence tolerance.
    tol: F,
    /// Initial imputation strategy.
    initial_strategy: InitialStrategy,
    /// Order in which features are imputed (default `Ascending`).
    imputation_order: ImputationOrder,
    /// Minimum imputed value (default `-inf`, sklearn `_iterative.py:318`).
    min_value: F,
    /// Maximum imputed value (default `+inf`, sklearn `_iterative.py:319`).
    max_value: F,
}

impl<F: Float + Send + Sync + 'static> IterativeImputer<F> {
    /// Create a new `IterativeImputer` with the given core parameters and the
    /// sklearn defaults for `imputation_order` (`Ascending`) and the
    /// `min_value`/`max_value` clip (`-inf`/`+inf`).
    pub fn new(max_iter: usize, tol: F, initial_strategy: InitialStrategy) -> Self {
        Self {
            max_iter,
            tol,
            initial_strategy,
            imputation_order: ImputationOrder::Ascending,
            min_value: F::neg_infinity(),
            max_value: F::infinity(),
        }
    }

    /// Set the feature imputation order (default `Ascending`), mirroring
    /// sklearn's `imputation_order` (`_iterative.py:316`).
    pub fn with_imputation_order(mut self, order: ImputationOrder) -> Self {
        self.imputation_order = order;
        self
    }

    /// Set the minimum imputed value (the clip lower bound), mirroring sklearn's
    /// `min_value` (`_iterative.py:318`, default `-inf`).
    pub fn with_min_value(mut self, min_value: F) -> Self {
        self.min_value = min_value;
        self
    }

    /// Set the maximum imputed value (the clip upper bound), mirroring sklearn's
    /// `max_value` (`_iterative.py:319`, default `+inf`).
    pub fn with_max_value(mut self, max_value: F) -> Self {
        self.max_value = max_value;
        self
    }

    /// Return the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Return the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> F {
        self.tol
    }

    /// Return the initial imputation strategy.
    #[must_use]
    pub fn initial_strategy(&self) -> InitialStrategy {
        self.initial_strategy
    }

    /// Return the feature imputation order.
    #[must_use]
    pub fn imputation_order(&self) -> ImputationOrder {
        self.imputation_order
    }

    /// Return the minimum imputed value (clip lower bound).
    #[must_use]
    pub fn min_value(&self) -> F {
        self.min_value
    }

    /// Return the maximum imputed value (clip upper bound).
    #[must_use]
    pub fn max_value(&self) -> F {
        self.max_value
    }
}

impl<F: Float + Send + Sync + 'static> Default for IterativeImputer<F> {
    fn default() -> Self {
        Self::new(
            10,
            F::from(1e-3).unwrap_or_else(F::epsilon),
            InitialStrategy::Mean,
        )
    }
}

// ---------------------------------------------------------------------------
// FittedIterativeImputer
// ---------------------------------------------------------------------------

/// A fitted iterative imputer that stores the ordered sequence of per-feature
/// [`BayesianRidge`] models learned during fitting, mirroring scikit-learn's
/// `imputation_sequence_` (`sklearn/impute/_iterative.py:739,798-801`).
///
/// Created by calling [`Fit::fit`] on an [`IterativeImputer`]. `transform`
/// replays this sequence deterministically (no re-fitting), exactly as sklearn's
/// inductive `transform` does (`_iterative.py:865-873`).
#[derive(Debug, Clone)]
pub struct FittedIterativeImputer<F> {
    /// Per-feature initial fill values (used for initial imputation of transform data).
    initial_fill: Array1<F>,
    /// Ordered imputation sequence: `(feat_idx, fitted BayesianRidge)` per round
    /// per feature, mirroring sklearn's `imputation_sequence_`.
    imputation_sequence: Vec<ImputationStep<F>>,
    /// Number of rounds performed (sklearn `n_iter_`).
    n_iter: usize,
    /// Minimum imputed value (clip lower bound).
    min_value: F,
    /// Maximum imputed value (clip upper bound).
    max_value: F,
    /// Initial strategy.
    initial_strategy: InitialStrategy,
}

/// One step of the imputation sequence: the feature being imputed plus the
/// fitted [`BayesianRidge`] model that imputes it.
#[derive(Debug, Clone)]
struct ImputationStep<F> {
    /// Index of the feature this step imputes.
    feat_idx: usize,
    /// The fitted per-feature regression model.
    model: FittedBayesianRidge<F>,
}

impl<F: Float + Send + Sync + 'static> FittedIterativeImputer<F> {
    /// Return the number of iterations (rounds) performed during fitting
    /// (sklearn `n_iter_`).
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Return the initial fill values per feature.
    #[must_use]
    pub fn initial_fill(&self) -> &Array1<F> {
        &self.initial_fill
    }

    /// Return the initial imputation strategy used during fitting.
    #[must_use]
    pub fn initial_strategy(&self) -> InitialStrategy {
        self.initial_strategy
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute column means, ignoring NaN values.
fn column_means_nan<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n_features = x.ncols();
    let mut means = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = x.column(j);
        let mut sum = F::zero();
        let mut count = 0usize;
        for &v in col {
            if !v.is_nan() {
                sum = sum + v;
                count += 1;
            }
        }
        means[j] = if count > 0 {
            sum / F::from(count).unwrap_or_else(F::one)
        } else {
            F::zero()
        };
    }
    means
}

/// Compute column medians, ignoring NaN values.
fn column_medians_nan<F: Float>(x: &Array2<F>) -> Array1<F> {
    let n_features = x.ncols();
    let mut medians = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = x.column(j);
        let mut vals: Vec<F> = col.iter().copied().filter(|v| !v.is_nan()).collect();
        if vals.is_empty() {
            medians[j] = F::zero();
        } else {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = vals.len();
            medians[j] = if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) / (F::one() + F::one())
            };
        }
    }
    medians
}

/// Fill NaN values in a matrix with the given fill values.
fn initial_fill<F: Float>(x: &Array2<F>, fill: &Array1<F>) -> Array2<F> {
    let mut out = x.to_owned();
    for (mut col, &f) in out.columns_mut().into_iter().zip(fill.iter()) {
        for v in &mut col {
            if v.is_nan() {
                *v = f;
            }
        }
    }
    out
}

/// Compute the visit order of features, mirroring scikit-learn's
/// `_get_ordered_idx` (`sklearn/impute/_iterative.py:504-542`) with
/// `skip_complete=False` (so `missing_values_idx = arange(n_features)`).
///
/// Returns ALL feature indices in the requested order. Features with no missing
/// values are kept in the order (matching sklearn) but produce no imputation
/// (the caller skips them when there are no missing rows). `Ascending` is a
/// STABLE sort by missing fraction (sklearn's `kind="mergesort"`): ties keep
/// ascending column order. `frac_of_missing[j]` is `missing_count[j] / n_samples`
/// but the per-sample denominator is identical across features, so ordering by
/// the raw missing count is identical to ordering by the fraction.
fn ordered_feature_idx(missing_counts: &[usize], order: ImputationOrder) -> Vec<usize> {
    let n_features = missing_counts.len();
    match order {
        ImputationOrder::Roman => (0..n_features).collect(),
        ImputationOrder::Arabic => (0..n_features).rev().collect(),
        ImputationOrder::Ascending => {
            let mut idx: Vec<usize> = (0..n_features).collect();
            // STABLE sort by missing count ascending (sklearn mergesort).
            idx.sort_by_key(|&j| missing_counts[j]);
            idx
        }
        ImputationOrder::Descending => {
            // sklearn `descending` is `ascending[::-1]` — the reverse of the
            // stable ascending order (NOT a stable descending sort).
            let mut idx: Vec<usize> = (0..n_features).collect();
            idx.sort_by_key(|&j| missing_counts[j]);
            idx.reverse();
            idx
        }
    }
}

/// Clip `v` into `[min_value, max_value]`, mirroring `np.clip` (sklearn
/// `_iterative.py:455-457`). With the default ±∞ bounds this is the identity.
fn clip<F: Float>(v: F, min_value: F, max_value: F) -> F {
    if v < min_value {
        min_value
    } else if v > max_value {
        max_value
    } else {
        v
    }
}

/// Build the predictor column indices for feature `feat_idx`: all other features
/// in ascending column order, mirroring scikit-learn's default
/// `_get_neighbor_feat_idx` (`concatenate((arange(feat_idx), arange(feat_idx+1,
/// n_features)))`, `_iterative.py:499-501`).
fn neighbor_feat_idx(n_features: usize, feat_idx: usize) -> Vec<usize> {
    (0..n_features).filter(|&k| k != feat_idx).collect()
}

/// Impute one feature `feat_idx` from the others, mirroring scikit-learn's
/// `_impute_one_feature` (`sklearn/impute/_iterative.py:345-466`) in the
/// `fit_mode=True`, `sample_posterior=False` path: fit the estimator on the rows
/// where the feature is OBSERVED (`X = the predictor columns' current filled
/// values, y = this feature's observed values`, `:408-418`), predict the missing
/// rows (`:454`), clip the predictions to `[min_value, max_value]` (`:455-457`),
/// and write them back into `imputed` (`:460-465`). Returns the fitted model.
///
/// `mask` marks missing entries of the ORIGINAL `x`. When the feature has no
/// missing rows the estimator is still fit (matching sklearn) but nothing is
/// written; `None` is returned so the caller records no replay step.
fn impute_one_feature<F>(
    imputed: &mut Array2<F>,
    mask: &Array2<bool>,
    feat_idx: usize,
    predictors: &[usize],
    min_value: F,
    max_value: F,
) -> Result<Option<FittedBayesianRidge<F>>, FerroError>
where
    F: ImputerFloat,
{
    let n_samples = imputed.nrows();
    let n_predictors = predictors.len();
    if n_predictors == 0 {
        return Ok(None);
    }

    // Observed (training) rows: feature is NOT missing.
    let observed_rows: Vec<usize> = (0..n_samples).filter(|&i| !mask[[i, feat_idx]]).collect();
    if observed_rows.is_empty() {
        return Ok(None);
    }

    let mut x_train = Array2::zeros((observed_rows.len(), n_predictors));
    let mut y_train = Array1::zeros(observed_rows.len());
    for (r, &i) in observed_rows.iter().enumerate() {
        for (c, &k) in predictors.iter().enumerate() {
            x_train[[r, c]] = imputed[[i, k]];
        }
        y_train[r] = imputed[[i, feat_idx]];
    }

    // Default per-feature estimator: BayesianRidge (sklearn `_iterative.py:732-735`).
    let model = BayesianRidge::<F>::new().fit(&x_train, &y_train)?;

    // Missing rows: predict + clip + write back.
    let missing_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[[i, feat_idx]]).collect();
    if !missing_rows.is_empty() {
        let mut x_test = Array2::zeros((missing_rows.len(), n_predictors));
        for (r, &i) in missing_rows.iter().enumerate() {
            for (c, &k) in predictors.iter().enumerate() {
                x_test[[r, c]] = imputed[[i, k]];
            }
        }
        let preds = model.predict(&x_test)?;
        for (r, &i) in missing_rows.iter().enumerate() {
            imputed[[i, feat_idx]] = clip(preds[r], min_value, max_value);
        }
    }

    Ok(Some(model))
}

/// Predict + clip + write back using an already-fitted per-feature model,
/// mirroring scikit-learn's `transform`-time `_impute_one_feature` with
/// `fit_mode=False` (`_iterative.py:865-873`).
fn replay_one_feature<F>(
    imputed: &mut Array2<F>,
    mask: &Array2<bool>,
    feat_idx: usize,
    predictors: &[usize],
    model: &FittedBayesianRidge<F>,
    min_value: F,
    max_value: F,
) -> Result<(), FerroError>
where
    F: ImputerFloat,
{
    let n_samples = imputed.nrows();
    let missing_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[[i, feat_idx]]).collect();
    if missing_rows.is_empty() {
        return Ok(());
    }
    let mut x_test = Array2::zeros((missing_rows.len(), predictors.len()));
    for (r, &i) in missing_rows.iter().enumerate() {
        for (c, &k) in predictors.iter().enumerate() {
            x_test[[r, c]] = imputed[[i, k]];
        }
    }
    let preds = model.predict(&x_test)?;
    for (r, &i) in missing_rows.iter().enumerate() {
        imputed[[i, feat_idx]] = clip(preds[r], min_value, max_value);
    }
    Ok(())
}

/// Inf-norm of `a - b` over all entries, mirroring `np.linalg.norm(Xt -
/// Xt_previous, ord=np.inf, axis=None)` (sklearn `_iterative.py:811`): the
/// maximum absolute element-wise difference.
fn inf_norm_diff<F: Float>(a: &Array2<F>, b: &Array2<F>) -> F {
    let mut m = F::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Maximum absolute value of the OBSERVED entries of the original `x` (entries
/// where `mask` is `false`), mirroring `np.max(np.abs(X[~mask_missing_values]))`
/// (sklearn `_iterative.py:780`). Returns `0` when every entry is missing.
fn max_abs_observed<F: Float>(x: &Array2<F>, mask: &Array2<bool>) -> F {
    let mut m = F::zero();
    for (&v, &is_missing) in x.iter().zip(mask.iter()) {
        if !is_missing {
            let a = v.abs();
            if a > m {
                m = a;
            }
        }
    }
    m
}

/// Build the boolean missing mask (`v.is_nan()`) and the per-feature missing
/// counts for `x`.
fn missing_mask_and_counts<F: Float>(x: &Array2<F>) -> (Array2<bool>, Vec<usize>) {
    let (n_samples, n_features) = x.dim();
    let mut mask = Array2::from_elem((n_samples, n_features), false);
    let mut counts = vec![0usize; n_features];
    for j in 0..n_features {
        for i in 0..n_samples {
            if x[[i, j]].is_nan() {
                mask[[i, j]] = true;
                counts[j] += 1;
            }
        }
    }
    (mask, counts)
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<F: ImputerFloat> Fit<Array2<F>, ()> for IterativeImputer<F> {
    type Fitted = FittedIterativeImputer<F>;
    type Error = FerroError;

    /// Fit the iterative imputer by round-robin [`BayesianRidge`] regression,
    /// mirroring `sklearn.impute.IterativeImputer.fit_transform`
    /// (`sklearn/impute/_iterative.py:693-831`) in the `sample_posterior=False`
    /// path.
    ///
    /// Features with missing values are visited in `imputation_order` (default
    /// `Ascending`); each is fit on the rows where it is observed and its
    /// missing rows are predicted and clipped to `[min_value, max_value]`. The
    /// round-robin repeats up to `max_iter` rounds, breaking early when
    /// `max|Xt - Xt_prev| < tol * max|X_observed|` (inf-norm convergence,
    /// `:780,811,818`).
    ///
    /// # Errors
    ///
    /// - [`FerroError::InsufficientSamples`] if the input has zero rows.
    /// - Propagates [`FerroError`] from the per-feature `BayesianRidge` fit.
    ///
    /// `max_iter == 0` is valid (matching sklearn `_iterative.py:750-752`): the
    /// iteration loop runs zero times and `fit` returns the initial fill with
    /// `n_iter() == 0`.
    fn fit(&self, x: &Array2<F>, _y: &()) -> Result<FittedIterativeImputer<F>, FerroError> {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "IterativeImputer::fit".into(),
            });
        }

        let n_features = x.ncols();

        // Initial fill values (sklearn `SimpleImputer(strategy=...)`, `:743`).
        let fill_values = match self.initial_strategy {
            InitialStrategy::Mean => column_means_nan(x),
            InitialStrategy::Median => column_medians_nan(x),
        };

        let (mask, missing_counts) = missing_mask_and_counts(x);
        let n_features_with_missing = missing_counts.iter().filter(|&&c| c > 0).count();

        // Initial imputation.
        let mut imputed = initial_fill(x, &fill_values);

        let mut imputation_sequence: Vec<ImputationStep<F>> = Vec::new();
        let mut n_iter = 0usize;

        // max_iter == 0, all-missing, or single-feature short-circuit
        // (sklearn `:750-757`).
        if self.max_iter == 0 || n_features_with_missing == 0 || n_features <= 1 {
            return Ok(FittedIterativeImputer {
                initial_fill: fill_values,
                imputation_sequence,
                n_iter: 0,
                min_value: self.min_value,
                max_value: self.max_value,
                initial_strategy: self.initial_strategy,
            });
        }

        // normalized_tol = tol * max|X_observed| (sklearn `:780`).
        let normalized_tol = self.tol * max_abs_observed(x, &mask);

        // Feature visit order (sklearn `_get_ordered_idx`, `:769`).
        let order = ordered_feature_idx(&missing_counts, self.imputation_order);

        let mut prev = imputed.clone();

        // Round-robin loop (sklearn `for self.n_iter_ in range(1, max_iter+1)`).
        for round in 1..=self.max_iter {
            n_iter = round;
            // APPEND every round's fitted models to `imputation_sequence`, matching
            // sklearn's `self.imputation_sequence_.append(estimator_triplet)` inside
            // the round loop (`_iterative.py:781,801`): the stored sequence holds
            // `n_features_with_missing * n_iter` models, in round-then-feature order.
            // `transform` then replays ALL of them from the initial fill
            // (`_iterative.py:865-873`), so the inductive transform and the
            // non-converged fit_transform match sklearn — not just the converged case.
            for &feat_idx in &order {
                let predictors = neighbor_feat_idx(n_features, feat_idx);
                if let Some(model) = impute_one_feature(
                    &mut imputed,
                    &mask,
                    feat_idx,
                    &predictors,
                    self.min_value,
                    self.max_value,
                )? {
                    imputation_sequence.push(ImputationStep { feat_idx, model });
                }
            }

            // Inf-norm convergence (sklearn `:811,818`).
            let inf_norm = inf_norm_diff(&imputed, &prev);
            if inf_norm < normalized_tol {
                break;
            }
            prev = imputed.clone();
        }

        Ok(FittedIterativeImputer {
            initial_fill: fill_values,
            imputation_sequence,
            n_iter,
            min_value: self.min_value,
            max_value: self.max_value,
            initial_strategy: self.initial_strategy,
        })
    }
}

impl<F: ImputerFloat> Transform<Array2<F>> for FittedIterativeImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Impute missing values in `x` by replaying the learned imputation sequence
    /// without re-fitting, mirroring scikit-learn's inductive `transform`
    /// (`sklearn/impute/_iterative.py:833-885`): the initial fill is applied,
    /// then each stored `(feat_idx, estimator)` predicts and clips its feature's
    /// missing rows in order.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if the number of columns does not
    /// match the training data; propagates [`FerroError`] from `predict`.
    fn transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let n_features = self.initial_fill.len();
        if x.ncols() != n_features {
            return Err(FerroError::ShapeMismatch {
                expected: vec![x.nrows(), n_features],
                actual: vec![x.nrows(), x.ncols()],
                context: "FittedIterativeImputer::transform".into(),
            });
        }

        // Initial imputation.
        let mut imputed = initial_fill(x, &self.initial_fill);

        // n_iter_ == 0 (or no recorded steps) → return the initial fill
        // (sklearn `:857-858`).
        if self.n_iter == 0 || self.imputation_sequence.is_empty() {
            return Ok(imputed);
        }

        let (mask, _) = missing_mask_and_counts(x);

        for step in &self.imputation_sequence {
            let predictors = neighbor_feat_idx(n_features, step.feat_idx);
            replay_one_feature(
                &mut imputed,
                &mask,
                step.feat_idx,
                &predictors,
                &step.model,
                self.min_value,
                self.max_value,
            )?;
        }

        Ok(imputed)
    }
}

/// Implement `Transform` on the unfitted imputer to satisfy the
/// `FitTransform: Transform` supertrait bound.
impl<F: Float + Send + Sync + 'static> Transform<Array2<F>> for IterativeImputer<F> {
    type Output = Array2<F>;
    type Error = FerroError;

    /// Always returns an error — the imputer must be fitted first.
    fn transform(&self, _x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "IterativeImputer".into(),
            reason: "imputer must be fitted before calling transform; use fit() first".into(),
        })
    }
}

impl<F: ImputerFloat> FitTransform<Array2<F>> for IterativeImputer<F> {
    type FitError = FerroError;

    /// Fit the imputer on `x` and return the imputed output in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&self, x: &Array2<F>) -> Result<Array2<F>, FerroError> {
        let fitted = self.fit(x, &())?;
        fitted.transform(x)
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
    fn test_iterative_imputer_basic() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // All values should be non-NaN.
        for v in &out {
            assert!(!v.is_nan(), "Output contains NaN");
        }
    }

    #[test]
    fn test_iterative_imputer_no_missing() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_iterative_imputer_convergence() {
        let imputer = IterativeImputer::<f64>::new(100, 1e-6, InitialStrategy::Mean);
        // Correlated features: feature 1 ≈ 2 * feature 0.
        let x = array![
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, f64::NAN],
            [f64::NAN, 10.0]
        ];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        // Feature 1 of row 3 should be close to 8.0 (2 * 4.0).
        assert!(
            (out[[3, 1]] - 8.0).abs() < 2.0,
            "Expected ~8.0, got {}",
            out[[3, 1]]
        );
        // Feature 0 of row 4 should be close to 5.0 (10.0 / 2).
        assert!(
            (out[[4, 0]] - 5.0).abs() < 2.0,
            "Expected ~5.0, got {}",
            out[[4, 0]]
        );
    }

    #[test]
    fn test_iterative_imputer_median_strategy() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Median);
        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, f64::NAN]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[2, 1]].is_nan());
    }

    #[test]
    fn test_iterative_imputer_fit_transform() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
        let out = imputer.fit_transform(&x).unwrap();
        for v in &out {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn test_iterative_imputer_zero_rows_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x: Array2<f64> = Array2::zeros((0, 3));
        assert!(imputer.fit(&x, &()).is_err());
    }

    #[test]
    fn test_iterative_imputer_zero_max_iter_returns_initial_fill() {
        // sklearn `_iterative.py:750-752`: max_iter == 0 is VALID — fit_transform
        // sets n_iter_ = 0 and returns the initial SimpleImputer fill with NO
        // regression rounds.
        let imputer = IterativeImputer::<f64>::new(0, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [f64::NAN, 3.0], [5.0, f64::NAN], [7.0, 8.0]];

        let fit_res = imputer.fit(&x, &());
        assert!(
            fit_res.is_ok(),
            "max_iter=0 must be accepted (sklearn parity), got {fit_res:?}"
        );
        let Ok(fitted) = fit_res else { return };
        assert_eq!(fitted.n_iter(), 0);

        let out_res = fitted.transform(&x);
        assert!(
            out_res.is_ok(),
            "transform after max_iter=0 fit failed: {out_res:?}"
        );
        let Ok(out) = out_res else { return };
        // Per-column mean initial fill (cols share mean 13/3), observed preserved.
        let mean_fill = 13.0 / 3.0;
        let expected = array![[1.0, 2.0], [mean_fill, 3.0], [5.0, mean_fill], [7.0, 8.0]];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-9,
                "max_iter=0 output {got} != initial fill {want}"
            );
        }
    }

    #[test]
    fn test_iterative_imputer_shape_mismatch_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let fitted = imputer.fit(&x_train, &()).unwrap();
        let x_bad = array![[1.0, 2.0, 3.0]];
        assert!(fitted.transform(&x_bad).is_err());
    }

    #[test]
    fn test_iterative_imputer_unfitted_transform_error() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0]];
        assert!(imputer.transform(&x).is_err());
    }

    #[test]
    fn test_iterative_imputer_default() {
        let imputer = IterativeImputer::<f64>::default();
        assert_eq!(imputer.max_iter(), 10);
        assert_eq!(imputer.initial_strategy(), InitialStrategy::Mean);
        // sklearn defaults: ascending order, ±inf clip (`_iterative.py:316,318-319`).
        assert_eq!(imputer.imputation_order(), ImputationOrder::Ascending);
        assert!(imputer.min_value().is_infinite() && imputer.min_value() < 0.0);
        assert!(imputer.max_value().is_infinite() && imputer.max_value() > 0.0);
    }

    #[test]
    fn test_iterative_imputer_n_iter_accessor() {
        let imputer = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        assert!(fitted.n_iter() > 0);
        assert!(fitted.n_iter() <= 10);
    }

    #[test]
    fn test_iterative_imputer_f32() {
        let imputer = IterativeImputer::<f32>::new(10, 1e-3, InitialStrategy::Mean);
        let x: Array2<f32> = array![[1.0f32, 2.0], [3.0, f32::NAN], [5.0, 6.0]];
        let fitted = imputer.fit(&x, &()).unwrap();
        let out = fitted.transform(&x).unwrap();
        assert!(!out[[1, 1]].is_nan());
    }

    #[test]
    fn test_ordered_feature_idx_ascending_stable() {
        // counts [1,2,1] -> ascending stable [0,2,1] (ties keep column order),
        // matching sklearn argsort(frac, kind="mergesort") (Probe D).
        assert_eq!(
            ordered_feature_idx(&[1, 2, 1], ImputationOrder::Ascending),
            vec![0, 2, 1]
        );
        assert_eq!(
            ordered_feature_idx(&[1, 2, 1], ImputationOrder::Roman),
            vec![0, 1, 2]
        );
        assert_eq!(
            ordered_feature_idx(&[1, 2, 1], ImputationOrder::Descending),
            vec![1, 2, 0]
        );
        assert_eq!(
            ordered_feature_idx(&[1, 2, 1], ImputationOrder::Arabic),
            vec![2, 1, 0]
        );
    }

    #[test]
    fn test_clip_bounds() {
        assert_eq!(clip(10.0, 0.0, 5.0), 5.0);
        assert_eq!(clip(-2.0, 0.0, 5.0), 0.0);
        assert_eq!(clip(3.0, 0.0, 5.0), 3.0);
        assert_eq!(clip(3.0, f64::NEG_INFINITY, f64::INFINITY), 3.0);
    }
}
