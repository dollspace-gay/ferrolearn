//! RANSAC (RANdom SAmple Consensus) robust regression.
//!
//! This module provides [`RANSACRegressor`], a meta-estimator that fits a
//! base regressor to inlier data, automatically detecting and excluding
//! outliers.
//!
//! # Algorithm
//!
//! Mirrors scikit-learn 1.5.2's `RANSACRegressor.fit` decision rule
//! (`sklearn/linear_model/_ransac.py:451-606`). Initialize `n_inliers_best = 1`,
//! `score_best = -inf`, `inlier_mask_best = None`. Then for each trial:
//!
//! 1. Draw a `min_samples`-sized subset of indices (seedable; RNG-sequence
//!    parity with numpy is out of scope — see `## REQ status`).
//! 2. Fit the base estimator on the SUBSET.
//! 3. Predict on ALL of `X`; classify a sample as an inlier iff
//!    `|y − y_pred| <= residual_threshold` (boundary inclusive).
//! 4. If `n_inliers_subset < n_inliers_best`, skip.
//! 5. Compute `score_subset` = R² of the SUBSET model on its inlier set
//!    (`1 − SS_res/SS_tot`; if `SS_tot == 0` then `1.0` when `SS_res == 0` else
//!    `0.0`). No refit inside the loop.
//! 6. If `n_inliers_subset == n_inliers_best` and `score_subset < score_best`,
//!    skip — so higher R² wins ties on inlier count.
//! 7. Otherwise record the new best (`n_inliers_best`, `score_best`, and the
//!    SUBSET model's mask — never recomputed from a refit).
//!
//! After the loop, refit the base estimator ONCE on the best inlier set; that is
//! the stored model. The reported `inlier_mask` is the winning subset model's
//! mask. The default `residual_threshold` is the MAD of `y`
//! (`median(|y − median(y)|)`), which may be exactly `0` for a constant target.
//!
//! ## REQ status (per `.design/linear/ransac.md`, mirrors `sklearn/linear_model/_ransac.py` @ 1.5.2)
//!
//! Binary classification (R-DEFER-2): SHIPPED means impl plus non-test consumer
//! plus tests, all green; NOT-STARTED means an open blocker referenced by number.
//! The boundary estimator types [`RANSACRegressor`] and
//! [`FittedRANSACRegressor`] are re-exported at the crate root
//! (`pub use ransac::{...} in lib.rs`); under S5/R-DEFER-1 the public estimator
//! type IS the consumer surface (no `ferrolearn-python` RANSAC binding yet).
//!
//! **RNG non-parity caveat:** subset draws use `rand::rngs::StdRng` (Fisher-Yates),
//! NOT numpy's Mersenne-Twister `sample_without_replacement`. Same-seed
//! cross-implementation subset-sequence parity is infeasible and out of scope;
//! parity is asserted only on the deterministic decision rules below.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (sampling loop) | SHIPPED | `fn sample_indices` draws `k` distinct indices via Fisher-Yates, called per trial in `fn fit`, seeded deterministically. Test: `test_ransac_reproducible_with_seed`. Structural only (RNG caveat above). |
//! | REQ-2 (MAD threshold default) | SHIPPED | `fn fit` sets the auto threshold to `fn mad` (`median(|y − median(y)|)`) when `residual_threshold` is `None`, mirroring `_ransac.py:401`. Test: `test_ransac_auto_threshold`. |
//! | REQ-3 (inlier classification) | SHIPPED | `fn fit`: `if (preds[i] - y[i]).abs() <= threshold { inlier_mask_subset[i] = true }`, boundary-inclusive `<=` per `_ransac.py:511`. Tests: `test_ransac_with_outlier`, `test_ransac_multiple_outliers`. |
//! | REQ-4 (selection: n_inliers then R²) | SHIPPED | `fn fit` ranks by `(n_inliers_subset, score_subset)` with `score_subset = fn r2_score` of the subset model on its inliers; ties skip when `score_subset < score_best` (higher R² wins), mirroring `_ransac.py:530-543`. Test: `ransac_selection_criterion_r2_not_residual_sum` (tests/divergence_ransac_fit.rs) — oracle picks group B `[F,F,F,T,T,T]`, predict([[1.0]])≈10.05. Closed #512. |
//! | REQ-5 (refit-once; mask from subset model) | SHIPPED | `fn fit` records `inlier_mask_best` from the SUBSET model (no in-loop refit/recompute) and refits the base estimator ONCE after the loop on `(x_inlier_best, y_inlier_best)`, mirroring `_ransac.py:544,602,605`. The stored `inlier_mask` is never recomputed from the refit. Verified by the green divergence suite + module unit tests. Closed #513. |
//! | REQ-6 (n_inliers_best init / acceptance gate) | SHIPPED | `fn fit` initializes `n_inliers_best = 1` (`_ransac.py:451`), skips only when `n_inliers_subset < n_inliers_best` (`_ransac.py:515`), and no longer gates on `n_inliers >= min_samples`. The up-front `n_samples < min_samples` guard mirrors `_ransac.py:393-397`. Closed #514. |
//! | REQ-7 (dynamic max_trials + stop criteria) | NOT-STARTED | open blocker #515. `fn fit` runs a FIXED `for _ in 0..self.max_trials` loop; no `_dynamic_max_trials` shrink, no `stop_n_inliers`/`stop_score`/`stop_probability`/`max_skips`, no `n_trials_`/`n_skips_*` tracking. |
//! | REQ-8 (loss='squared_error') | SHIPPED | impl: `pub enum RansacLoss { #[default] AbsoluteError, SquaredError }` + field `loss: RansacLoss` + `with_loss` builder (sklearn default `'absolute_error'`, `_ransac.py:301`). `fn fit` branches the per-sample residual: `AbsoluteError → (preds[i]-y[i]).abs()`, `SquaredError → { let d = preds[i]-y[i]; d*d }`, mirroring `_ransac.py:407,414` and applied at the `residuals <= residual_threshold` classification (`_ransac.py:508,511`); the MAD-default threshold stays loss-independent (`_ransac.py:399-401`). Consumer: boundary types re-exported at crate root + `RansacLoss` re-exported (`pub use ransac::{...RansacLoss} in lib.rs`). Tests (live-oracle, RNG-independent): `ransac_loss_squared_error_recovers_line`, `ransac_loss_default_absolute_error_byte_identical` (tests/divergence_ransac_fit.rs). Closed #516. |
//! | REQ-9 (MAD-zero parity) | SHIPPED | `fn fit` uses the MAD value directly (`mad(&y.to_vec())`) with no `1e-6` substitution, so a constant target yields threshold `0.0` per `_ransac.py:399-401`. Test: `ransac_mad_zero_threshold_excludes_tiny_deviation` (tests/divergence_ransac_fit.rs) — idx 7 (residual 1e-7) is an OUTLIER. Closed #517. |
//! | REQ-10 (introspection attributes) | NOT-STARTED | open blocker #518. `FittedRANSACRegressor` exposes only `inlier_mask()`; no `estimator_`/`n_trials_`/`n_skips_*`/`n_features_in_`. |
//! | REQ-11 (is_data_valid / is_model_valid / max_skips) | NOT-STARTED | open blocker #519. `RANSACRegressor` has no such fields. |
//! | REQ-12 (min_samples float fraction) | SHIPPED | impl: `pub enum MinSamples<F> { Count(usize), Fraction(F) }` + field `min_samples: Option<MinSamples<F>>` + builders `with_min_samples(usize) → Count`, `with_min_samples_fraction(F) → Fraction`, getter `min_samples()`. `fn fit` resolves `None → n_features+1` (unchanged), `Count(k) → k`, `Fraction(f) → ceil(f·n_samples)` validating `0 < f < 1` (else `FerroError::InvalidParameter`), with the resolved-count `> n_samples → InvalidParameter` guard mirroring `_ransac.py:382-397` (sklearn `ValueError`). Consumer: boundary types re-exported at crate root + `MinSamples` re-exported (`pub use ransac::{...MinSamples} in lib.rs`). Tests (live-oracle): `ransac_min_samples_fraction_resolves_ceil`, `ransac_min_samples_fraction_out_of_range_errors`, `ransac_min_samples_count_unchanged` (tests/divergence_ransac_fit.rs). Closed #520. |
//! | REQ-13 (ferray substrate) | NOT-STARTED | open blocker #521. Still on `ndarray` + `rand::rngs::StdRng`, not `ferray-core`/`ferray::random`. |
//!
//! # Examples
//!
//! ```
//! use ferrolearn_linear::ransac::RANSACRegressor;
//! use ferrolearn_linear::LinearRegression;
//! use ferrolearn_core::{Fit, Predict};
//! use ndarray::{array, Array1, Array2};
//!
//! // Data with an outlier at index 4.
//! let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let y = array![2.0, 4.0, 6.0, 8.0, 100.0]; // last point is outlier
//!
//! let base = LinearRegression::<f64>::new();
//! let model = RANSACRegressor::new(base);
//! let fitted = model.fit(&x, &y).unwrap();
//!
//! // The outlier should be detected.
//! let mask = fitted.inlier_mask();
//! assert!(!mask[4], "outlier at index 4 should be detected");
//! ```

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Predict};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// Loss family (REQ-8)
// ---------------------------------------------------------------------------

/// Per-sample residual function used to classify inliers.
///
/// Mirrors scikit-learn's `RANSACRegressor(loss=...)` string options
/// (`sklearn/linear_model/_ransac.py:284,405-418`), constrained by
/// `StrOptions({"absolute_error", "squared_error"})`. The residual produced
/// here is compared (boundary-inclusive) against `residual_threshold`; the
/// MAD-based default threshold itself is computed identically regardless of the
/// loss (`_ransac.py:399-401` is unconditional — only the per-sample residual at
/// `_ransac.py:508` changes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RansacLoss {
    /// Absolute error per sample: `|y_true − y_pred|`
    /// (`_ransac.py:407`, sklearn default).
    #[default]
    AbsoluteError,
    /// Squared error per sample: `(y_true − y_pred)²` (`_ransac.py:414`).
    SquaredError,
}

// ---------------------------------------------------------------------------
// min_samples specification (REQ-12)
// ---------------------------------------------------------------------------

/// How `min_samples` (the per-trial subset size) is specified.
///
/// Mirrors scikit-learn's `min_samples` parameter, which is `int (>= 1)` for an
/// absolute count or `float ([0, 1])` for a relative fraction
/// (`sklearn/linear_model/_ransac.py:115-125`; constraint
/// `Interval(Integral, 1, None) | Interval(RealNotInt, 0, 1, closed="both")`,
/// `_ransac.py:262-266`). The resolution at fit time is:
/// `0 < f < 1 → ceil(f · n_samples)` (`_ransac.py:389-390`); an integer count is
/// used directly (`_ransac.py:391-392`). A resolved count larger than
/// `n_samples` is a `ValueError` (`_ransac.py:393-397`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MinSamples<F> {
    /// An absolute number of samples per subset (`min_samples >= 1`).
    Count(usize),
    /// A fraction of `n_samples` per subset, resolved to
    /// `ceil(fraction · n_samples)` (sklearn `0 < min_samples < 1`).
    Fraction(F),
}

// ---------------------------------------------------------------------------
// RANSACRegressor (unfitted)
// ---------------------------------------------------------------------------

/// RANSAC robust regression meta-estimator.
///
/// Wraps a base regressor (e.g., [`LinearRegression`](crate::LinearRegression))
/// and repeatedly fits it on random subsets to find a model robust to
/// outliers.
///
/// # Type Parameters
///
/// - `F`: The floating-point type (`f32` or `f64`).
/// - `E`: The base estimator type.
#[derive(Debug, Clone)]
pub struct RANSACRegressor<F, E> {
    /// The base estimator.
    pub estimator: E,
    /// Minimum number of samples per subset. `None` resolves to
    /// `n_features + 1` (the `LinearRegression` default,
    /// `sklearn/linear_model/_ransac.py:388`); a [`MinSamples::Count`] is an
    /// absolute count, a [`MinSamples::Fraction`] resolves to
    /// `ceil(fraction · n_samples)` (`_ransac.py:389-390`).
    pub min_samples: Option<MinSamples<F>>,
    /// Residual threshold: points whose per-sample residual (under [`loss`]) is
    /// `<= threshold` are considered inliers. If `None`, uses the MAD of the
    /// target.
    ///
    /// [`loss`]: RANSACRegressor::loss
    pub residual_threshold: Option<F>,
    /// Maximum number of random trials.
    pub max_trials: usize,
    /// Per-sample residual loss used for inlier classification. Defaults to
    /// [`RansacLoss::AbsoluteError`] (sklearn default,
    /// `sklearn/linear_model/_ransac.py:301`).
    pub loss: RansacLoss,
    /// Optional random seed for reproducibility.
    pub random_state: Option<u64>,
}

impl<F: Float, E> RANSACRegressor<F, E> {
    /// Create a new `RANSACRegressor` with the given base estimator.
    ///
    /// Defaults: `min_samples = None` (auto: n_features + 1),
    /// `residual_threshold = None` (auto: MAD), `max_trials = 100`,
    /// `loss = AbsoluteError` (sklearn default), `random_state = None`.
    #[must_use]
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            min_samples: None,
            residual_threshold: None,
            max_trials: 100,
            loss: RansacLoss::AbsoluteError,
            random_state: None,
        }
    }

    /// Set the minimum number of samples per subset as an absolute count
    /// ([`MinSamples::Count`]). Mirrors sklearn `min_samples` as an `int >= 1`.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(MinSamples::Count(min_samples));
        self
    }

    /// Set the minimum number of samples per subset as a fraction of
    /// `n_samples` ([`MinSamples::Fraction`]), resolved at fit time to
    /// `ceil(fraction · n_samples)`. Mirrors sklearn `min_samples` as a
    /// `float` in `(0, 1)` (`sklearn/linear_model/_ransac.py:389-390`).
    ///
    /// The fraction is validated at fit time: a value outside `(0, 1)` (or one
    /// whose resolved count exceeds `n_samples`) yields
    /// [`FerroError::InvalidParameter`], mirroring sklearn's `ValueError`
    /// (`_ransac.py:393-397`).
    #[must_use]
    pub fn with_min_samples_fraction(mut self, fraction: F) -> Self {
        self.min_samples = Some(MinSamples::Fraction(fraction));
        self
    }

    /// Returns the configured `min_samples` specification (`None` = auto).
    #[must_use]
    pub fn min_samples(&self) -> Option<MinSamples<F>> {
        self.min_samples
    }

    /// Set the per-sample residual loss used for inlier classification.
    ///
    /// Mirrors sklearn `RANSACRegressor(loss=...)`
    /// (`sklearn/linear_model/_ransac.py:284`). Defaults to
    /// [`RansacLoss::AbsoluteError`].
    #[must_use]
    pub fn with_loss(mut self, loss: RansacLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the residual threshold for inlier detection.
    #[must_use]
    pub fn with_residual_threshold(mut self, threshold: F) -> Self {
        self.residual_threshold = Some(threshold);
        self
    }

    /// Set the maximum number of random trials.
    #[must_use]
    pub fn with_max_trials(mut self, max_trials: usize) -> Self {
        self.max_trials = max_trials;
        self
    }

    /// Set the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

// ---------------------------------------------------------------------------
// FittedRANSACRegressor
// ---------------------------------------------------------------------------

/// Fitted RANSAC robust regression model.
///
/// Stores the best estimator fitted on inlier data, and the inlier mask.
#[derive(Debug, Clone)]
pub struct FittedRANSACRegressor<Fitted> {
    /// The fitted base estimator (fitted on inliers).
    fitted_estimator: Fitted,
    /// Boolean mask: true if the sample was classified as an inlier.
    inlier_mask: Vec<bool>,
}

impl<Fitted> FittedRANSACRegressor<Fitted> {
    /// Returns the inlier mask. `true` indicates the sample was an inlier.
    #[must_use]
    pub fn inlier_mask(&self) -> &[bool] {
        &self.inlier_mask
    }
}

// ---------------------------------------------------------------------------
// Helper: Median Absolute Deviation
// ---------------------------------------------------------------------------

/// Compute the median of a slice of floats.
fn median<F: Float>(values: &[F]) -> F {
    let mut sorted: Vec<F> = values.to_vec();
    // Total order without `.unwrap()`: NaNs (absent for valid targets) sort last.
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / (F::one() + F::one())
    } else {
        sorted[n / 2]
    }
}

/// Compute the Median Absolute Deviation (MAD) of a slice.
fn mad<F: Float>(values: &[F]) -> F {
    let med = median(values);
    let abs_devs: Vec<F> = values.iter().map(|&v| (v - med).abs()).collect();
    median(&abs_devs)
}

/// Coefficient of determination R² of `y_pred` against `y_true`.
///
/// Mirrors sklearn's `r2_score` (used through `estimator.score`,
/// `_ransac.py:530`): `R² = 1 - SS_res / SS_tot` where
/// `SS_res = Σ(y_true − y_pred)²` and `SS_tot = Σ(y_true − mean(y_true))²`.
///
/// Matches sklearn's constant-target edge case (`metrics/_regression.py`):
/// when `SS_tot == 0`, R² is `1.0` if `SS_res == 0` (perfect prediction) and
/// `0.0` otherwise.
fn r2_score<F: Float>(y_true: &[F], y_pred: &[F]) -> F {
    let n = y_true.len();
    if n == 0 {
        return F::zero();
    }
    let mut sum = F::zero();
    for &v in y_true {
        sum = sum + v;
    }
    let mean = sum / F::from(n).unwrap_or_else(F::one);

    let mut ss_res = F::zero();
    let mut ss_tot = F::zero();
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let res = t - p;
        ss_res = ss_res + res * res;
        let dev = t - mean;
        ss_tot = ss_tot + dev * dev;
    }

    if ss_tot == F::zero() {
        if ss_res == F::zero() {
            F::one()
        } else {
            F::zero()
        }
    } else {
        F::one() - ss_res / ss_tot
    }
}

// ---------------------------------------------------------------------------
// Random subset sampling
// ---------------------------------------------------------------------------

/// Sample `k` distinct indices from `0..n` using Fisher-Yates.
fn sample_indices<R: Rng>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

/// Extract a subset of rows from a 2D array and a 1D array.
fn subset<F: Float>(x: &Array2<F>, y: &Array1<F>, indices: &[usize]) -> (Array2<F>, Array1<F>) {
    let n_features = x.ncols();
    let n = indices.len();
    let mut x_sub = Array2::<F>::zeros((n, n_features));
    let mut y_sub = Array1::<F>::zeros(n);
    for (row, &idx) in indices.iter().enumerate() {
        for col in 0..n_features {
            x_sub[[row, col]] = x[[idx, col]];
        }
        y_sub[row] = y[idx];
    }
    (x_sub, y_sub)
}

// ---------------------------------------------------------------------------
// Fit and Predict
// ---------------------------------------------------------------------------

impl<F, E, Ef> Fit<Array2<F>, Array1<F>> for RANSACRegressor<F, E>
where
    F: Float + Send + Sync + ScalarOperand + num_traits::FromPrimitive + 'static,
    E: Fit<Array2<F>, Array1<F>, Fitted = Ef, Error = FerroError> + Clone,
    Ef: Predict<Array2<F>, Output = Array1<F>, Error = FerroError> + Clone,
{
    type Fitted = FittedRANSACRegressor<Ef>;
    type Error = FerroError;

    /// Fit the RANSAC model by repeatedly sampling and fitting.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `x` and `y` have different
    /// sample counts.
    /// Returns [`FerroError::ConvergenceFailure`] if no valid model is found
    /// after `max_trials` iterations.
    fn fit(
        &self,
        x: &Array2<F>,
        y: &Array1<F>,
    ) -> Result<FittedRANSACRegressor<E::Fitted>, FerroError> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
                context: "y length must match number of samples in X".into(),
            });
        }

        // Resolve `min_samples` per sklearn (`_ransac.py:382-397`):
        //   None        -> n_features + 1 (the LinearRegression default).
        //   Count(k)     -> k (integer count, `>= 1` branch); `k < 1` is a
        //                   ValueError per the parameter constraint
        //                   `Interval(Integral, 1, None, closed="left")`
        //                   (`_ransac.py:263`) — never silently coerced.
        //   Fraction(f)  -> ceil(f * n_samples) for `0 < f < 1`; a fraction
        //                   outside (0, 1) is a ValueError.
        // A resolved count `> n_samples` is a ValueError (`_ransac.py:393-397`).
        let min_samples = match self.min_samples {
            None => (n_features + 1).max(1),
            Some(MinSamples::Count(k)) => {
                // sklearn rejects `min_samples < 1` at parameter validation
                // (`Interval(Integral, 1, None, closed="left")`,
                // `_ransac.py:263`); do not coerce `0` to `1`.
                if k < 1 {
                    return Err(FerroError::InvalidParameter {
                        name: "min_samples".into(),
                        reason: "min_samples must be >= 1".into(),
                    });
                }
                k
            }
            Some(MinSamples::Fraction(f)) => {
                // sklearn's float branch is `0 < min_samples < 1`
                // (`_ransac.py:389`); a float `>= 1` would take the integer
                // branch — for the explicit `Fraction` variant we require a
                // genuine fraction in (0, 1).
                if !(f > F::zero() && f < F::one()) {
                    return Err(FerroError::InvalidParameter {
                        name: "min_samples".into(),
                        reason: "min_samples fraction must be in the open \
                                 interval (0, 1); use with_min_samples for an \
                                 absolute count"
                            .into(),
                    });
                }
                // ceil(f * n_samples) (`_ransac.py:390`).
                let raw = f * F::from(n_samples).unwrap_or_else(F::one);
                let resolved = raw.ceil();
                // n_samples >= 1 and 0 < f < 1 keep `resolved` finite and >= 1.
                let resolved = resolved.to_usize().unwrap_or(1).max(1);
                if resolved > n_samples {
                    return Err(FerroError::InvalidParameter {
                        name: "min_samples".into(),
                        reason: format!(
                            "`min_samples` may not be larger than number of \
                             samples: n_samples = {n_samples}."
                        ),
                    });
                }
                resolved
            }
        };

        if n_samples < min_samples {
            return Err(FerroError::InsufficientSamples {
                required: min_samples,
                actual: n_samples,
                context: "RANSAC requires at least min_samples samples".into(),
            });
        }

        // Compute residual threshold if not provided.
        //
        // sklearn (`_ransac.py:399-401`) sets the default threshold to the MAD
        // (median absolute deviation) of `y` with NO special-casing of zero:
        // `residual_threshold = np.median(np.abs(y - np.median(y)))`. A constant
        // (or near-constant) target therefore yields a threshold of exactly 0.0,
        // under which only samples with a zero residual are inliers.
        let threshold = match self.residual_threshold {
            Some(t) => t,
            None => mad(&y.to_vec()),
        };

        let mut rng = match self.random_state {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        // sklearn-faithful selection state (`_ransac.py:451-456`):
        //   n_inliers_best = 1, score_best = -inf, inlier_mask_best = None.
        // The best inlier index set is remembered for the single final refit.
        let mut n_inliers_best: usize = 1;
        let mut score_best = F::neg_infinity();
        let mut inlier_mask_best: Option<Vec<bool>> = None;
        let mut inlier_best_idxs: Option<Vec<usize>> = None;

        // `while n_trials < max_trials` (`_ransac.py:467`). We keep the fixed
        // `self.max_trials` loop; dynamic max-trials / stop criteria are #515.
        for _ in 0..self.max_trials {
            // Choose a random sample set (`_ransac.py:478-482`). RNG-sequence
            // parity with numpy's `sample_without_replacement` is infeasible and
            // explicitly out of scope (see module REQ status).
            let subset_idxs = sample_indices(&mut rng, n_samples, min_samples);
            let (x_subset, y_subset) = subset(x, y, &subset_idxs);

            // Fit the base estimator on the SUBSET (`_ransac.py:497`).
            let fitted_subset = match self.estimator.fit(&x_subset, &y_subset) {
                Ok(f) => f,
                Err(_) => continue, // Skip failed fits (degenerate subset).
            };

            // Residuals of ALL data under the subset model (`_ransac.py:507-508`).
            let preds = match fitted_subset.predict(x) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Per-sample residual under the configured loss (`_ransac.py:508`,
            // `residuals_subset = loss_function(y, y_pred)`):
            //   AbsoluteError -> |y − y_pred|  (`_ransac.py:407`)
            //   SquaredError  -> (y − y_pred)² (`_ransac.py:414`)
            // The residual is then compared, boundary-inclusive, to
            // `residual_threshold` (`_ransac.py:511-512`); the MAD-default
            // threshold is loss-independent (`_ransac.py:399-401`).
            let mut inlier_mask_subset = vec![false; n_samples];
            let mut inlier_idxs_subset: Vec<usize> = Vec::new();
            for i in 0..n_samples {
                let residual = match self.loss {
                    RansacLoss::AbsoluteError => (preds[i] - y[i]).abs(),
                    RansacLoss::SquaredError => {
                        let d = preds[i] - y[i];
                        d * d
                    }
                };
                if residual <= threshold {
                    inlier_mask_subset[i] = true;
                    inlier_idxs_subset.push(i);
                }
            }
            let n_inliers_subset = inlier_idxs_subset.len();

            // Fewer inliers than the best so far -> skip (`_ransac.py:514-517`).
            if n_inliers_subset < n_inliers_best {
                continue;
            }

            // Score the SUBSET model on the inlier set: R² of the subset-fitted
            // model on `(X_inlier_subset, y_inlier_subset)` (`_ransac.py:530-534`).
            // No refit inside the loop.
            let y_inlier_subset: Vec<F> = inlier_idxs_subset.iter().map(|&i| y[i]).collect();
            let pred_inlier_subset: Vec<F> = inlier_idxs_subset.iter().map(|&i| preds[i]).collect();
            let score_subset = r2_score(&y_inlier_subset, &pred_inlier_subset);

            // Same inlier count but worse score -> skip (`_ransac.py:538-539`).
            // Higher R² wins ties on inlier count.
            if n_inliers_subset == n_inliers_best && score_subset < score_best {
                continue;
            }

            // Record the new best (`_ransac.py:542-547`). The stored mask is the
            // SUBSET model's mask — NOT recomputed from a refit.
            n_inliers_best = n_inliers_subset;
            score_best = score_subset;
            inlier_mask_best = Some(inlier_mask_subset);
            inlier_best_idxs = Some(inlier_idxs_subset);
        }

        // No valid consensus set found (`_ransac.py:561-580`).
        let (mask_best, idxs_best) = match (inlier_mask_best, inlier_best_idxs) {
            (Some(mask), Some(idxs)) => (mask, idxs),
            _ => {
                return Err(FerroError::ConvergenceFailure {
                    iterations: self.max_trials,
                    message: "RANSAC could not find a valid model after max_trials iterations"
                        .into(),
                });
            }
        };

        // Estimate the final model using the best inlier set ONCE, after the
        // loop (`_ransac.py:597-602`). `inlier_mask_` stays the subset model's
        // mask; it is never recomputed from this refit.
        let (x_inlier_best, y_inlier_best) = subset(x, y, &idxs_best);
        let fitted_estimator = self.estimator.fit(&x_inlier_best, &y_inlier_best)?;

        Ok(FittedRANSACRegressor {
            fitted_estimator,
            inlier_mask: mask_best,
        })
    }
}

impl<F, Fitted> Predict<Array2<F>> for FittedRANSACRegressor<Fitted>
where
    F: Float + Send + Sync + 'static,
    Fitted: Predict<Array2<F>, Output = Array1<F>, Error = FerroError>,
{
    type Output = Array1<F>;
    type Error = FerroError;

    /// Predict target values using the base estimator fitted on inliers.
    ///
    /// # Errors
    ///
    /// Returns any error from the base estimator's predict method.
    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        self.fitted_estimator.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearRegression;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_ransac_no_outliers() {
        // Perfect linear data, no outliers.
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_residual_threshold(1.0);
        let fitted = model.fit(&x, &y).unwrap();

        // All should be inliers.
        let mask = fitted.inlier_mask();
        assert!(mask.iter().all(|&v| v), "All should be inliers");

        // Predictions should be accurate.
        let preds = fitted.predict(&x).unwrap();
        for (p, &actual) in preds.iter().zip(y.iter()) {
            assert_relative_eq!(*p, actual, epsilon = 0.5);
        }
    }

    #[test]
    fn test_ransac_with_outlier() {
        // y = 2x, but one outlier.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0]; // outlier at idx 5

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_max_trials(200)
            .with_residual_threshold(2.0);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // The outlier at index 5 should be detected.
        assert!(!mask[5], "Outlier at index 5 should not be an inlier");

        // Most other points should be inliers.
        let n_inliers: usize = mask.iter().filter(|&&v| v).count();
        assert!(
            n_inliers >= 4,
            "Expected at least 4 inliers, got {n_inliers}"
        );

        // The prediction at x=3 should be close to 6.
        let x_test = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let pred = fitted.predict(&x_test).unwrap();
        assert!(
            (pred[0] - 6.0).abs() < 3.0,
            "Prediction at x=3 should be near 6.0, got {}",
            pred[0]
        );
    }

    #[test]
    fn test_ransac_multiple_outliers() {
        // y = x + 1, with two outliers.
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![2.0, 3.0, 50.0, 5.0, 6.0, -40.0, 8.0, 9.0]; // outliers at 2 and 5

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(123)
            .with_max_trials(500)
            .with_residual_threshold(2.0);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // Outliers at index 2 and 5 should be detected.
        assert!(!mask[2], "Outlier at index 2 should not be an inlier");
        assert!(!mask[5], "Outlier at index 5 should not be an inlier");
    }

    #[test]
    fn test_ransac_shape_mismatch() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = array![1.0, 2.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ransac_insufficient_samples() {
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let y = array![1.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base).with_min_samples(3);
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_ransac_reproducible_with_seed() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0];

        let base1 = LinearRegression::<f64>::new();
        let model1 = RANSACRegressor::new(base1)
            .with_random_state(42)
            .with_residual_threshold(2.0);
        let fitted1 = model1.fit(&x, &y).unwrap();

        let base2 = LinearRegression::<f64>::new();
        let model2 = RANSACRegressor::new(base2)
            .with_random_state(42)
            .with_residual_threshold(2.0);
        let fitted2 = model2.fit(&x, &y).unwrap();

        // Same seed should produce same inlier mask.
        assert_eq!(fitted1.inlier_mask(), fitted2.inlier_mask());
    }

    #[test]
    fn test_ransac_auto_threshold() {
        // No explicit threshold — should use MAD.
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 100.0];

        let base = LinearRegression::<f64>::new();
        let model = RANSACRegressor::new(base)
            .with_random_state(42)
            .with_max_trials(200);
        let fitted = model.fit(&x, &y).unwrap();

        let mask = fitted.inlier_mask();
        // At least some points should be inliers.
        let n_inliers: usize = mask.iter().filter(|&&v| v).count();
        assert!(
            n_inliers >= 3,
            "Expected at least 3 inliers, got {n_inliers}"
        );
    }
}
