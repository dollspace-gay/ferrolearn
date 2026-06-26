//! Feature scoring functions for feature selection.
//!
//! This module provides standalone univariate scoring functions that compute
//! per-feature statistics and p-values:
//!
//! - [`f_classif`] — ANOVA F-statistic for classification.
//! - [`r_regression`] — Pearson correlation between each feature and target.
//! - [`f_regression`] — univariate F-statistic via Pearson correlation.
//! - [`chi2`] — chi-squared statistic for non-negative features.
//!
//! These functions return `(F-statistics, p-values)` tuples and can be used
//! directly or passed to [`SelectKBest`](crate::feature_selection::SelectKBest)
//! / [`SelectPercentile`](crate::select_percentile::SelectPercentile) via the
//! [`ScoreFunc`](crate::feature_selection::ScoreFunc) enum.
//!
//! ## REQ status
//!
//! Translation target: scikit-learn 1.5.2 `f_classif`/`f_regression`/`chi2`
//! (`sklearn/feature_selection/_univariate_selection.py`). Tracking: #1416.
//! DETERMINISTIC numeric-parity unit — statistics + p-values verified against
//! the live scipy/sklearn oracle. Each REQ is BINARY — SHIPPED (impl + non-test
//! consumer + tests + green verification) or NOT-STARTED (open blocker).
//!
//! | REQ | Scope | Status | Evidence / Blocker |
//! |-----|-------|--------|--------------------|
//! | REQ-1 | [`f_classif`] F-statistic value parity (one-way ANOVA) | SHIPPED | matches sklearn `f_oneway` `_univariate_selection.py:43-117`; oracle stat tests (tol 1e-9) in `tests/divergence_feature_scoring.rs`. Consumer: re-export `lib.rs:173` |
//! | REQ-2 | [`f_regression`] F-statistic value parity (Pearson-F) | SHIPPED | `F=r²·(n-2)/(1-r²)` matches sklearn `:405+`; oracle stat tests (tol 1e-9) |
//! | REQ-3 | [`chi2`] statistic value parity | SHIPPED | matches sklearn `_chisquare` `:176-192`; oracle stat tests (tol 1e-9) |
//! | REQ-4 | p-values for all three (F-distribution + chi2 survival functions) | SHIPPED | `f_distribution_sf` (rewritten `regularized_incomplete_beta` + `betacf` Lentz CF, was DIV-1 #1417 fixed) matches scipy `fdtrc`; `chi2_distribution_sf` matches `chdtrc`; verified across small-p (1e-23 tail), large-p, moderate, varied df1∈{1,2,4}/df2∈{3..48} (29 oracle tests, ~13-15 sig figs) |
//! | REQ-5 | Error/parameter contracts (empty, shape mismatch, <2 classes, <3 samples f_regression, negative chi2 features) | SHIPPED (scoped) | per-fn guards; divergence error tests |
//! | REQ-6 | `f_regression` `center=False` + `force_finite` (nan/inf handling) | NOT-STARTED | sklearn `:405-465` — blocker #1418 |
//! | REQ-7 | `r_regression` free function (signed Pearson correlation) | SHIPPED | `r_regression` + `r_regression_with_options` mirror sklearn `:301-393`, including `center` and `force_finite`; oracle tests in `tests/divergence_r_regression.rs`. Consumer: re-export `lib.rs` + API proof. |
//! | REQ-8 | sparse `chi2` (CSR `observed = Y.T@X`) | NOT-STARTED | dense only; sklearn `:202-288` — blocker #1420 |
//! | REQ-9 | `mutual_info_classif`/`mutual_info_regression` | NOT-STARTED | `sklearn/feature_selection/_mutual_info.py` — blocker #1421 |
//! | REQ-10 | PyO3 binding | NOT-STARTED | no `ferrolearn-python` registration — blocker #1422 |
//! | REQ-11 | ferray substrate | NOT-STARTED | dense `Array1`/`Array2` + `num_traits::Float` only — blocker #1423 |

use ferrolearn_core::error::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ===========================================================================
// f_classif — ANOVA F-statistic
// ===========================================================================

/// Compute the ANOVA F-statistic and approximate p-values for each feature.
///
/// For each feature column the between-class and within-class sum of squares
/// are computed. The F-statistic is:
///
/// ```text
/// F = (SSB / (k - 1)) / (SSW / (n - k))
/// ```
///
/// where *k* is the number of distinct classes and *n* is the number of
/// samples.
///
/// P-values are approximated using the regularized incomplete beta function
/// from `ferrolearn-numerical`. If the numerical CDF is unavailable, `NaN`
/// is returned for the p-value.
///
/// # Returns
///
/// `(f_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has zero rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
/// - [`FerroError::InvalidParameter`] if fewer than 2 classes are present.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::f_classif;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 100.0], [2.0, 200.0], [10.0, 100.0], [11.0, 200.0]];
/// let y: Array1<usize> = array![0, 0, 1, 1];
/// let (f_stats, p_vals) = f_classif(&x, &y).unwrap();
/// assert_eq!(f_stats.len(), 2);
/// // Feature 0 separates classes well → high F
/// assert!(f_stats[0] > f_stats[1]);
/// ```
pub fn f_classif<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "f_classif".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "f_classif — y must have same length as x rows".into(),
        });
    }

    // Collect per-class row indices
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }
    let n_classes = class_indices.len();
    if n_classes < 2 {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: format!("f_classif requires at least 2 classes, got {n_classes}"),
        });
    }

    let n_features = x.ncols();
    let n_f = F::from(n_samples).unwrap();

    let df_between = n_classes - 1;
    let df_within = n_samples - n_classes;
    let df_b = F::from(df_between).unwrap();
    let df_w = F::from(df_within).unwrap();

    let mut f_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let grand_mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;

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

        let f = if df_w == F::zero() {
            F::zero()
        } else {
            let ms_between = ss_between / df_b;
            let ms_within = ss_within / df_w;
            if ms_within == F::zero() {
                // sklearn `f_oneway` computes `f = msb / msw`
                // (`_univariate_selection.py:113`) without guarding the
                // denominator. A CONSTANT feature has both msb == 0 and
                // msw == 0, so `0.0 / 0.0 = nan` (and `fdtrc(.., nan) = nan`).
                // Perfect separation has msb > 0 and msw == 0, so
                // `msb / 0 = +inf` (and `fdtrc(.., inf) = 0`). Match both.
                if ms_between == F::zero() {
                    F::nan()
                } else {
                    F::infinity()
                }
            } else {
                ms_between / ms_within
            }
        };

        f_stats[j] = f;
        p_vals[j] = f_distribution_sf(f, df_between, df_within);
    }

    Ok((f_stats, p_vals))
}

// ===========================================================================
// r_regression — Pearson correlation coefficient
// ===========================================================================

/// Compute Pearson's r between every feature and the regression target.
///
/// This mirrors `sklearn.feature_selection.r_regression(X, y)` with default
/// `center=true` and `force_finite=true`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has zero rows.
/// - [`FerroError::InvalidParameter`] if `x` has zero columns or `x`/`y`
///   contain NaN or infinity.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
pub fn r_regression<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Array1<F>, FerroError> {
    r_regression_with_options(x, y, true, true)
}

/// Compute Pearson's r with explicit sklearn `center` and `force_finite`
/// options.
///
/// `center=false` uses uncentered row norms and dot products. When
/// `force_finite=true`, undefined correlations produced by constant features
/// or a constant target are replaced with `0.0`, matching sklearn's
/// NaN-to-zero branch.
pub fn r_regression_with_options<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
    center: bool,
    force_finite: bool,
) -> Result<Array1<F>, FerroError> {
    validate_regression_xy(x, y, "r_regression")?;

    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_f = F::from(n_samples).unwrap();

    let y_mean = if center {
        y.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f
    } else {
        F::zero()
    };
    let y_norm_sq = y
        .iter()
        .copied()
        .map(|yi| {
            let centered = yi - y_mean;
            centered * centered
        })
        .fold(F::zero(), |acc, v| acc + v);
    let y_norm = y_norm_sq.sqrt();

    let mut corr = Array1::zeros(n_features);
    for j in 0..n_features {
        let col = x.column(j);
        let x_norm_sq = if center {
            let x_mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;
            let raw_norm_sq = col
                .iter()
                .copied()
                .map(|v| v * v)
                .fold(F::zero(), |acc, v| acc + v);
            raw_norm_sq - n_f * x_mean * x_mean
        } else {
            col.iter()
                .copied()
                .map(|v| v * v)
                .fold(F::zero(), |acc, v| acc + v)
        };
        let x_norm = x_norm_sq.sqrt();
        let numerator = col
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(xi, yi)| xi * (yi - y_mean))
            .fold(F::zero(), |acc, v| acc + v);

        let value = numerator / (x_norm * y_norm);
        corr[j] = if force_finite && value.is_nan() {
            F::zero()
        } else {
            value
        };
    }

    Ok(corr)
}

fn validate_regression_xy<F: Float>(
    x: &Array2<F>,
    y: &Array1<F>,
    context: &str,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    if x.ncols() == 0 {
        return Err(FerroError::InvalidParameter {
            name: "x".into(),
            reason: format!("{context} requires at least one feature"),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: format!("{context} — y must have same length as x rows"),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "x".into(),
            reason: "Input X contains NaN or infinity.".into(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidParameter {
            name: "y".into(),
            reason: "Input y contains NaN or infinity.".into(),
        });
    }
    Ok(())
}

// ===========================================================================
// f_regression — Pearson correlation-based F-statistic
// ===========================================================================

/// Compute univariate F-statistics via Pearson correlation for regression.
///
/// For each feature the Pearson correlation coefficient *r* with the target
/// is computed, then:
///
/// ```text
/// F = r^2 * (n - 2) / (1 - r^2)
/// ```
///
/// # Returns
///
/// `(f_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has fewer than 3 rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::f_regression;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 100.0], [2.0, 200.0], [3.0, 100.0], [4.0, 200.0]];
/// let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
/// let (f_stats, _p_vals) = f_regression(&x, &y).unwrap();
/// assert_eq!(f_stats.len(), 2);
/// ```
pub fn f_regression<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples < 3 {
        return Err(FerroError::InsufficientSamples {
            required: 3,
            actual: n_samples,
            context: "f_regression requires at least 3 samples".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "f_regression — y must have same length as x rows".into(),
        });
    }

    let n_f = F::from(n_samples).unwrap();
    let n_features = x.ncols();

    // Precompute y stats
    let y_mean = y.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;
    let y_var = y
        .iter()
        .copied()
        .map(|v| (v - y_mean) * (v - y_mean))
        .fold(F::zero(), |acc, v| acc + v);

    let two = F::from(2.0).unwrap();

    let mut f_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let x_mean = col.iter().copied().fold(F::zero(), |acc, v| acc + v) / n_f;
        let x_var = col
            .iter()
            .copied()
            .map(|v| (v - x_mean) * (v - x_mean))
            .fold(F::zero(), |acc, v| acc + v);

        let cov = col
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .fold(F::zero(), |acc, v| acc + v);

        let denom = x_var * y_var;
        let r = if denom == F::zero() {
            F::zero()
        } else {
            cov / denom.sqrt()
        };

        let r2 = r * r;
        let f = if r2 >= F::one() {
            // Perfect (anti-)correlation: r² == 1 → F would be +inf. sklearn's
            // `f_regression` has `force_finite=True` by DEFAULT, which replaces
            // the infinite F with `np.finfo(dtype).max`
            // (`_univariate_selection.py:447-461,509-513`); the p-value stays 0.
            // `F::max_value()` is the `np.finfo(dtype).max` analog (f64::MAX =
            // 1.7976931348623157e+308). A constant feature yields denom == 0 →
            // r == 0 → F == 0 above, p == sf(0) == 1, which already matches the
            // `force_finite` nan→0.0/p→1.0 branch.
            F::max_value()
        } else {
            r2 * (n_f - two) / (F::one() - r2)
        };

        f_stats[j] = f;
        // F-distribution with df1=1, df2=n-2
        p_vals[j] = f_distribution_sf(f, 1, n_samples - 2);
    }

    Ok((f_stats, p_vals))
}

// ===========================================================================
// chi2 — Chi-squared statistic
// ===========================================================================

/// Compute chi-squared statistics between each non-negative feature and the
/// class labels.
///
/// For each feature the observed and expected frequencies per class are
/// computed, then:
///
/// ```text
/// chi2 = sum_class (observed - expected)^2 / expected
/// ```
///
/// where `observed` is the sum of feature values for samples of that class,
/// and `expected` is the expected sum under the null hypothesis (proportional
/// to the class frequency and the overall feature sum).
///
/// # Returns
///
/// `(chi2_statistics, p_values)` — two `Array1<F>` of length `n_features`.
///
/// # Errors
///
/// - [`FerroError::InsufficientSamples`] if `x` has zero rows.
/// - [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
/// - [`FerroError::InvalidParameter`] if any feature value is negative.
///
/// # Examples
///
/// ```
/// use ferrolearn_preprocess::feature_scoring::chi2;
/// use ndarray::{array, Array1};
///
/// let x = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
/// let y: Array1<usize> = array![0, 1, 0, 1];
/// let (chi2_stats, _p_vals) = chi2(&x, &y).unwrap();
/// assert_eq!(chi2_stats.len(), 2);
/// ```
pub fn chi2<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
) -> Result<(Array1<F>, Array1<F>), FerroError> {
    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: "chi2".into(),
        });
    }
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "chi2 — y must have same length as x rows".into(),
        });
    }

    // Validate non-negative
    for j in 0..x.ncols() {
        for i in 0..n_samples {
            if x[[i, j]] < F::zero() {
                return Err(FerroError::InvalidParameter {
                    name: "x".into(),
                    reason: format!(
                        "chi2 requires non-negative features, found negative value at ({i}, {j})"
                    ),
                });
            }
        }
    }

    // Collect per-class row indices
    let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &label) in y.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }

    let n_classes = class_indices.len();
    let n_features = x.ncols();
    let n_f = F::from(n_samples).unwrap();

    let mut chi2_stats = Array1::zeros(n_features);
    let mut p_vals = Array1::zeros(n_features);

    for j in 0..n_features {
        let col = x.column(j);
        let total_sum = col.iter().copied().fold(F::zero(), |acc, v| acc + v);

        let mut chi2_val = F::zero();

        for rows in class_indices.values() {
            let n_k = F::from(rows.len()).unwrap();
            let observed = rows
                .iter()
                .map(|&i| col[i])
                .fold(F::zero(), |acc, v| acc + v);
            let expected = total_sum * n_k / n_f;

            // sklearn `_chisquare` divides `(obs - exp)² / exp` under
            // `np.errstate(invalid="ignore")` (`_univariate_selection.py:189-191`).
            // For an all-zero feature column `expected == 0` and `observed == 0`,
            // so IEEE `0.0 / 0.0 = nan`; the summed statistic is nan and
            // `chdtrc(.., nan) = nan`. We mirror this by dividing unconditionally
            // (the IEEE result is nan, never a trap — R-CODE-2) rather than
            // short-circuiting the all-zero column to stat=0/p=1.
            let diff = observed - expected;
            chi2_val = chi2_val + diff * diff / expected;
        }

        chi2_stats[j] = chi2_val;
        // Chi-squared distribution with df = n_classes - 1
        let df = n_classes.saturating_sub(1);
        p_vals[j] = chi2_distribution_sf(chi2_val, df);
    }

    Ok((chi2_stats, p_vals))
}

// ===========================================================================
// Distribution helper: F-distribution survival function (1 - CDF)
// ===========================================================================

/// Approximate the survival function (1 - CDF) of the F-distribution.
///
/// Uses the relationship between the F-distribution and the regularized
/// incomplete beta function:
///
/// ```text
/// P(F > x) = I_{d2/(d2 + d1*x)}(d2/2, d1/2)
/// ```
///
/// Returns `NaN` if the computation cannot be performed.
fn f_distribution_sf<F: Float>(x: F, df1: usize, df2: usize) -> F {
    if x <= F::zero() {
        return F::one();
    }
    if df1 == 0 || df2 == 0 {
        return F::nan();
    }

    let d1 = F::from(df1).unwrap();
    let d2 = F::from(df2).unwrap();

    // I_{d2/(d2 + d1*x)}(d2/2, d1/2)
    let z = d2 / (d2 + d1 * x);
    let a = d2 / F::from(2.0).unwrap();
    let b = d1 / F::from(2.0).unwrap();

    regularized_incomplete_beta(z, a, b)
}

/// Approximate the survival function (1 - CDF) of the chi-squared distribution.
///
/// Uses the relationship: chi2 with k df = Gamma(k/2, 2), and
/// P(X > x) = 1 - gamma_cdf = upper regularized gamma Q(k/2, x/2).
///
/// We use the relationship to the regularized incomplete beta function:
/// Q(a, x) = I_{x/(x+a)}(... ) — but more simply, chi2 with k df is
/// equivalent to F(k, inf) scaled. We use a direct series approximation.
fn chi2_distribution_sf<F: Float>(x: F, df: usize) -> F {
    if x <= F::zero() {
        return F::one();
    }
    if df == 0 {
        return F::nan();
    }

    // Use the upper regularized gamma function Q(k/2, x/2)
    let a = F::from(df).unwrap() / F::from(2.0).unwrap();
    let z = x / F::from(2.0).unwrap();

    upper_regularized_gamma(a, z)
}

/// Upper regularized gamma function Q(a, x) = 1 - P(a, x).
///
/// Uses a continued fraction expansion for x >= a + 1, and the series
/// expansion otherwise.
fn upper_regularized_gamma<F: Float>(a: F, x: F) -> F {
    if x <= F::zero() {
        return F::one();
    }

    let one = F::one();
    let two = F::from(2.0).unwrap();

    // Use series for P(a, x) when x < a + 1, then Q = 1 - P
    if x < a + one {
        let p = lower_regularized_gamma_series(a, x);
        return one - p;
    }

    // Continued fraction for Q(a, x) — Lentz's method
    let eps = F::from(1.0e-12).unwrap();
    let tiny = F::from(1.0e-30).unwrap();

    let mut c = tiny;
    let mut d = F::one() / (x + one - a);
    let mut f = d;

    for n_iter in 1..200 {
        let n = F::from(n_iter).unwrap();
        // Even term
        let an_even = n * (a - n);
        let bn_even = x + two * n + one - a;
        d = F::one() / (bn_even + an_even * d);
        c = bn_even + an_even / c;
        let delta = c * d;
        f = f * delta;

        if (delta - one).abs() < eps {
            break;
        }
    }

    // Q(a, x) = e^(-x) * x^a / Gamma(a) * f
    let log_prefix = a * x.ln() - x - ln_gamma(a);
    let prefix = log_prefix.exp();
    let result = prefix * f;

    // Clamp to [0, 1]
    if result < F::zero() {
        F::zero()
    } else if result > one {
        one
    } else {
        result
    }
}

/// Lower regularized gamma function P(a, x) via series expansion.
fn lower_regularized_gamma_series<F: Float>(a: F, x: F) -> F {
    let eps = F::from(1.0e-12).unwrap();
    let one = F::one();

    let mut sum = one / a;
    let mut term = one / a;

    for n in 1..200 {
        let n_f = F::from(n).unwrap();
        term = term * x / (a + n_f);
        sum = sum + term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - ln_gamma(a);
    let result = log_prefix.exp() * sum;

    // Clamp to [0, 1]
    if result < F::zero() {
        F::zero()
    } else if result > one {
        one
    } else {
        result
    }
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Numerical Recipes (§6.4) algorithm: the `front` prefactor
/// `x^a (1-x)^b / (a · B(a,b))` times the Lentz-method continued fraction
/// [`betacf`], using the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)` so the
/// continued fraction is only evaluated in its fast-converging regime
/// `x < (a+1)/(a+b+2)`.
fn regularized_incomplete_beta<F: Float>(x: F, a: F, b: F) -> F {
    let one = F::one();
    let two = F::from(2.0).unwrap_or_else(F::one);

    if x <= F::zero() {
        return F::zero();
    }
    if x >= one {
        return one;
    }

    // front = x^a (1-x)^b / (a · B(a,b)), computed in log space for stability.
    let ln_front = a * x.ln() + b * (one - x).ln() - ln_beta(a, b);
    let front = ln_front.exp();

    if x < (a + one) / (a + b + two) {
        front * betacf(a, b, x) / a
    } else {
        one - front * betacf(b, a, one - x) / b
    }
}

/// Lentz-method continued fraction for the regularized incomplete beta
/// function (Numerical Recipes §6.4 `betacf`). Converges rapidly only when
/// `x < (a+1)/(a+b+2)`; callers use the symmetry relation otherwise.
fn betacf<F: Float>(a: F, b: F, x: F) -> F {
    let one = F::one();
    let two = F::from(2.0).unwrap_or_else(F::one);
    let tiny = F::from(1.0e-30).unwrap_or_else(F::min_positive_value);
    let eps = F::from(1.0e-12).unwrap_or_else(F::epsilon);
    const MAXIT: usize = 200;

    let qab = a + b;
    let qap = a + one;
    let qam = a - one;

    let mut c = one;
    let mut d = one - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = one / d;
    let mut h = d;

    for m in 1..=MAXIT {
        let m_f = F::from(m).unwrap_or_else(F::one);
        let m2 = two * m_f;

        // Even step.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = one + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        h = h * d * c;

        // Odd step.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = one + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = one + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = one / d;
        let del = d * c;
        h = h * del;

        if (del - one).abs() < eps {
            break;
        }
    }

    h
}

/// Log of the beta function: ln(Beta(a, b)) = lnGamma(a) + lnGamma(b) - lnGamma(a+b).
fn ln_beta<F: Float>(a: F, b: F) -> F {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation of ln(Gamma(x)) for x > 0.
fn ln_gamma<F: Float>(x: F) -> F {
    // Lanczos coefficients (g=7, n=9)
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let one = F::one();
    let half = F::from(0.5).unwrap();
    let g = F::from(7.0).unwrap();

    if x < half {
        // Reflection formula
        let pi = F::from(std::f64::consts::PI).unwrap();
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(one - x);
    }

    let z = x - one;
    let mut sum = F::from(coefs[0]).unwrap();
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        sum = sum + F::from(c).unwrap() / (z + F::from(i).unwrap());
    }

    let t = z + g + half;
    let sqrt_2pi = F::from(2.506_628_274_631_000_5).unwrap();

    sqrt_2pi.ln() + (z + half) * t.ln() - t + sum.ln()
}

// ===========================================================================
// ScoreFunc integration
// ===========================================================================

/// Add `FRegression` and `Chi2` variants to `ScoreFunc`.
///
/// This cannot extend the existing enum directly, so we provide adapter
/// functions that compute scores in the format expected by `SelectKBest`.
///
/// Compute scores for the given score function name, returning F-scores only.
///
/// This is a convenience dispatcher for integration with feature selection.
pub fn compute_scores_classif<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<usize>,
    func: &str,
) -> Result<Vec<F>, FerroError> {
    match func {
        "f_classif" => {
            let (f_stats, _) = f_classif(x, y)?;
            Ok(f_stats.to_vec())
        }
        "chi2" => {
            let (chi2_stats, _) = chi2(x, y)?;
            Ok(chi2_stats.to_vec())
        }
        _ => Err(FerroError::InvalidParameter {
            name: "func".into(),
            reason: format!("unknown classification score function: {func}"),
        }),
    }
}

/// Compute regression scores, returning F-statistics only.
pub fn compute_scores_regression<F: Float + Send + Sync + 'static>(
    x: &Array2<F>,
    y: &Array1<F>,
) -> Result<Vec<F>, FerroError> {
    let (f_stats, _) = f_regression(x, y)?;
    Ok(f_stats.to_vec())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // -----------------------------------------------------------------------
    // f_classif tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f_classif_basic() {
        // Feature 0 separates classes well, feature 1 does not
        let x = array![
            [1.0_f64, 5.0],
            [1.5, 5.5],
            [2.0, 4.5],
            [10.0, 5.0],
            [10.5, 4.5],
            [11.0, 5.5]
        ];
        let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
        let (f_stats, p_vals) = f_classif(&x, &y).unwrap();
        assert_eq!(f_stats.len(), 2);
        assert_eq!(p_vals.len(), 2);
        // Feature 0 should have much higher F than feature 1
        assert!(f_stats[0] > f_stats[1]);
        // p-value for feature 0 should be very small
        assert!(p_vals[0] < 0.05);
    }

    #[test]
    fn test_f_classif_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_shape_mismatch() {
        let x = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let y: Array1<usize> = array![0, 1, 2]; // wrong length
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_single_class_error() {
        let x = array![[1.0_f64], [2.0], [3.0]];
        let y: Array1<usize> = array![0, 0, 0];
        assert!(f_classif(&x, &y).is_err());
    }

    #[test]
    fn test_f_classif_perfect_separation() {
        // Feature perfectly separates classes → infinite F
        let x = array![[0.0_f64], [0.0], [1.0], [1.0]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let (f_stats, _) = f_classif(&x, &y).unwrap();
        assert!(f_stats[0].is_infinite());
    }

    #[test]
    fn test_f_classif_p_values_bounded() {
        let x = array![
            [1.0_f64, 10.0],
            [2.0, 20.0],
            [3.0, 10.0],
            [4.0, 20.0],
            [5.0, 10.0],
            [6.0, 20.0]
        ];
        let y: Array1<usize> = array![0, 0, 0, 1, 1, 1];
        let (_, p_vals) = f_classif(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value {p} out of bounds");
        }
    }

    // -----------------------------------------------------------------------
    // f_regression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f_regression_perfect_correlation() {
        // Feature 0 = target → r=1 → F=infinity
        let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].is_infinite() || f_stats[0] > 1.0e6);
    }

    #[test]
    fn test_f_regression_no_correlation() {
        // Orthogonal feature → r≈0 → F≈0
        let x = array![[1.0_f64], [-1.0], [1.0], [-1.0]];
        let y: Array1<f64> = array![1.0, 1.0, -1.0, -1.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].abs() < 1.0e-6);
    }

    #[test]
    fn test_f_regression_too_few_samples() {
        let x = array![[1.0_f64], [2.0]];
        let y: Array1<f64> = array![1.0, 2.0];
        assert!(f_regression(&x, &y).is_err());
    }

    #[test]
    fn test_f_regression_shape_mismatch() {
        let x = array![[1.0_f64], [2.0], [3.0]];
        let y: Array1<f64> = array![1.0, 2.0]; // wrong length
        assert!(f_regression(&x, &y).is_err());
    }

    #[test]
    fn test_f_regression_p_values_bounded() {
        let x = array![
            [1.0_f64, 10.0],
            [2.0, 20.0],
            [3.0, 15.0],
            [4.0, 25.0],
            [5.0, 10.0]
        ];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, p_vals) = f_regression(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value {p} out of bounds");
        }
    }

    #[test]
    fn test_f_regression_constant_feature() {
        // Constant feature → r=0 → F=0
        let x = array![[5.0_f64], [5.0], [5.0], [5.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let (f_stats, _) = f_regression(&x, &y).unwrap();
        assert!(f_stats[0].abs() < 1.0e-6);
    }

    // -----------------------------------------------------------------------
    // chi2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chi2_basic() {
        // Feature 0 correlates with class, feature 1 is random
        let x = array![
            [1.0_f64, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ];
        let y: Array1<usize> = array![1, 1, 0, 0, 1, 1, 0, 0];
        let (chi2_stats, p_vals) = chi2(&x, &y).unwrap();
        assert_eq!(chi2_stats.len(), 2);
        assert_eq!(p_vals.len(), 2);
        // Feature 0 perfectly correlates → higher chi2
        assert!(chi2_stats[0] > chi2_stats[1]);
    }

    #[test]
    fn test_chi2_negative_value_error() {
        let x = array![[1.0_f64, -1.0], [0.0, 1.0]];
        let y: Array1<usize> = array![0, 1];
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_empty_input() {
        let x = Array2::<f64>::zeros((0, 2));
        let y: Array1<usize> = Array1::zeros(0);
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_shape_mismatch() {
        let x = array![[1.0_f64], [2.0]];
        let y: Array1<usize> = array![0]; // wrong length
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn test_chi2_all_zeros() {
        // An all-zero feature column has expected == 0, so sklearn's `_chisquare`
        // computes `0/0 = nan` for the statistic and `chdtrc(.., nan) = nan` for
        // the p-value (`_univariate_selection.py:189-191`). Oracle (sklearn 1.5.2):
        //   X=[[0,0],[0,0]], y=[0,1] -> chi2 -> stat=[nan,nan], p=[nan,nan].
        let x = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let y: Array1<usize> = array![0, 1];
        let (chi2_stats, p_vals) = chi2(&x, &y).unwrap();
        assert!(chi2_stats[0].is_nan());
        assert!(p_vals[0].is_nan());
    }

    #[test]
    fn test_chi2_p_values_bounded() {
        let x = array![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let y: Array1<usize> = array![0, 1, 0, 1, 0, 1];
        let (_, p_vals) = chi2(&x, &y).unwrap();
        for &p in p_vals.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value {p} out of bounds");
        }
    }

    // -----------------------------------------------------------------------
    // Distribution helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1 → ln = 0
        let val: f64 = ln_gamma(1.0);
        assert!((val).abs() < 1.0e-10);

        // Gamma(2) = 1 → ln = 0
        let val2: f64 = ln_gamma(2.0);
        assert!((val2).abs() < 1.0e-10);

        // Gamma(3) = 2 → ln = ln(2)
        let val3: f64 = ln_gamma(3.0);
        assert!((val3 - 2.0_f64.ln()).abs() < 1.0e-10);

        // Gamma(0.5) = sqrt(pi) → ln = 0.5 * ln(pi)
        let val4: f64 = ln_gamma(0.5);
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((val4 - expected).abs() < 1.0e-8);
    }

    #[test]
    fn test_regularized_incomplete_beta_boundaries() {
        // I_0(a, b) = 0
        let val: f64 = regularized_incomplete_beta(0.0, 1.0, 1.0);
        assert!((val).abs() < 1.0e-10);

        // I_1(a, b) = 1
        let val2: f64 = regularized_incomplete_beta(1.0, 1.0, 1.0);
        assert!((val2 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_f_distribution_sf_zero() {
        // P(F > 0) = 1
        let val: f64 = f_distribution_sf(0.0, 2, 10);
        assert!((val - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn test_f_distribution_sf_large_f() {
        // Very large F → p ≈ 0
        let val: f64 = f_distribution_sf(1000.0, 2, 100);
        assert!(val < 0.001);
    }

    // -----------------------------------------------------------------------
    // compute_scores_classif / compute_scores_regression
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_scores_classif_f_classif() {
        let x = array![[1.0_f64, 5.0], [1.5, 5.5], [10.0, 5.0], [10.5, 4.5]];
        let y: Array1<usize> = array![0, 0, 1, 1];
        let scores = compute_scores_classif(&x, &y, "f_classif").unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_compute_scores_classif_chi2() {
        let x = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let y: Array1<usize> = array![0, 1, 0, 1];
        let scores = compute_scores_classif(&x, &y, "chi2").unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_compute_scores_classif_unknown() {
        let x = array![[1.0_f64]];
        let y: Array1<usize> = array![0];
        assert!(compute_scores_classif(&x, &y, "unknown").is_err());
    }

    #[test]
    fn test_compute_scores_regression() {
        let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
        let scores = compute_scores_regression(&x, &y).unwrap();
        assert_eq!(scores.len(), 2);
    }
}
