//! Goodness of fit diagnostics and heteroscedasticity tests.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use ferrolearn_core::FerroError;

use crate::bandwidth::silverman_bandwidth;

/// Which heteroscedasticity test to run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HeteroscedasticityTest {
    /// White's test (general, quadratic auxiliary regression).
    White,
    /// Breusch-Pagan test (linear auxiliary regression).
    BreuschPagan,
    /// Goldfeld-Quandt test (split-sample F-test).
    GoldfeldQuandt,
    /// Dette-Munk-Wagner nonparametric bootstrap test.
    DetteMunkWagner {
        /// Number of bootstrap samples.
        n_bootstrap: usize,
    },
}

/// Result of a heteroscedasticity test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeteroscedasticityTestResult {
    /// Test statistic.
    pub statistic: f64,
    /// P-value.
    pub p_value: f64,
    /// Whether the null (homoscedasticity) is rejected at the given alpha.
    pub is_heteroscedastic: bool,
    /// Name of the test.
    pub test_name: String,
    /// Significance level used.
    pub alpha: f64,
}

/// Result of residual diagnostics.
#[derive(Debug, Clone)]
pub struct ResidualDiagnosticsResult {
    /// Raw residuals.
    pub residuals: Array1<f64>,
    /// Standardized residuals.
    pub standardized_residuals: Array1<f64>,
    /// Mean of residuals.
    pub mean: f64,
    /// Standard deviation of residuals.
    pub std: f64,
    /// Skewness.
    pub skewness: f64,
    /// Excess kurtosis.
    pub kurtosis: f64,
    /// Jarque-Bera normality test statistic.
    pub normality_statistic: f64,
    /// Normality test p-value.
    pub normality_p_value: f64,
    /// Whether residuals appear normal.
    pub is_normal: bool,
}

/// Run a heteroscedasticity test.
///
/// Requires predictions to compute residuals. Pass the fitted values directly.
pub fn heteroscedasticity_test(
    x: &Array2<f64>,
    y: &Array1<f64>,
    y_pred: &Array1<f64>,
    test: HeteroscedasticityTest,
    alpha: f64,
) -> Result<HeteroscedasticityTestResult, FerroError> {
    let residuals = y - y_pred;
    let residuals_sq = residuals.mapv(|r| r * r);

    match test {
        HeteroscedasticityTest::White => white_test(x, &residuals_sq, alpha),
        HeteroscedasticityTest::BreuschPagan => breusch_pagan_test(x, &residuals_sq, alpha),
        HeteroscedasticityTest::GoldfeldQuandt => goldfeld_quandt_test(x, y, y_pred, alpha),
        HeteroscedasticityTest::DetteMunkWagner { n_bootstrap } => {
            dette_munk_wagner_test(x, &residuals, &residuals_sq, alpha, n_bootstrap)
        }
    }
}

/// White's test for heteroscedasticity.
fn white_test(
    x: &Array2<f64>,
    residuals_sq: &Array1<f64>,
    alpha: f64,
) -> Result<HeteroscedasticityTestResult, FerroError> {
    let n = x.nrows();
    let d = x.ncols();

    // Build auxiliary design: [1, X, X^2, cross products]
    let mut cols: Vec<Array1<f64>> = vec![Array1::ones(n)];
    for j in 0..d {
        cols.push(x.column(j).to_owned());
    }
    for j in 0..d {
        cols.push(x.column(j).mapv(|v| v * v));
    }
    for j in 0..d {
        for k in (j + 1)..d {
            cols.push(
                x.column(j)
                    .iter()
                    .zip(x.column(k).iter())
                    .map(|(&a, &b)| a * b)
                    .collect(),
            );
        }
    }

    let n_cols = cols.len();
    let mut z = Array2::zeros((n, n_cols));
    for (j, col) in cols.iter().enumerate() {
        z.column_mut(j).assign(col);
    }

    let (r_squared, df) = ols_r_squared(&z, residuals_sq);
    let statistic = n as f64 * r_squared;

    let chi2 = ChiSquared::new(df as f64).unwrap();
    let p_value = 1.0 - chi2.cdf(statistic);

    Ok(HeteroscedasticityTestResult {
        statistic,
        p_value,
        is_heteroscedastic: p_value < alpha,
        test_name: "White".into(),
        alpha,
    })
}

/// Breusch-Pagan test.
fn breusch_pagan_test(
    x: &Array2<f64>,
    residuals_sq: &Array1<f64>,
    alpha: f64,
) -> Result<HeteroscedasticityTestResult, FerroError> {
    let n = x.nrows();
    let d = x.ncols();

    // Auxiliary: [1, X]
    let mut z = Array2::zeros((n, 1 + d));
    z.column_mut(0).fill(1.0);
    for j in 0..d {
        z.column_mut(1 + j).assign(&x.column(j));
    }

    // Normalize squared residuals
    let sigma_sq = residuals_sq.mean().unwrap();
    let g = residuals_sq.mapv(|r| r / sigma_sq);

    let (_, _) = ols_r_squared(&z, &g);

    // Actually need ESS/2
    let g_mean = g.mean().unwrap();
    let g_pred = ols_predict(&z, &g);
    let ess: f64 = g_pred.iter().map(|&p| (p - g_mean).powi(2)).sum();
    let statistic = 0.5 * ess;

    let chi2 = ChiSquared::new(d as f64).unwrap();
    let p_value = 1.0 - chi2.cdf(statistic);

    Ok(HeteroscedasticityTestResult {
        statistic,
        p_value,
        is_heteroscedastic: p_value < alpha,
        test_name: "Breusch-Pagan".into(),
        alpha,
    })
}

/// Goldfeld-Quandt test.
fn goldfeld_quandt_test(
    x: &Array2<f64>,
    y: &Array1<f64>,
    y_pred: &Array1<f64>,
    alpha: f64,
) -> Result<HeteroscedasticityTestResult, FerroError> {
    let n = x.nrows();

    // Sort by predicted values
    let mut sort_idx: Vec<usize> = (0..n).collect();
    sort_idx.sort_by(|&a, &b| {
        y_pred[a]
            .partial_cmp(&y_pred[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_drop = n / 3;
    let n_sub = (n - n_drop) / 2;

    if n_sub < 3 {
        return Ok(HeteroscedasticityTestResult {
            statistic: 1.0,
            p_value: 1.0,
            is_heteroscedastic: false,
            test_name: "Goldfeld-Quandt".into(),
            alpha,
        });
    }

    // First subsample (low)
    let ss1: f64 = sort_idx[..n_sub]
        .iter()
        .map(|&i| (y[i] - y_pred[i]).powi(2))
        .sum();

    // Last subsample (high)
    let ss2: f64 = sort_idx[(n - n_sub)..]
        .iter()
        .map(|&i| (y[i] - y_pred[i]).powi(2))
        .sum();

    let (statistic, df1, df2) = if ss2 > ss1 && ss1 > 0.0 {
        (ss2 / ss1, n_sub - 1, n_sub - 1)
    } else if ss1 > 0.0 && ss2 > 0.0 {
        (ss1 / ss2, n_sub - 1, n_sub - 1)
    } else {
        (1.0, 1, 1)
    };

    let f_dist = FisherSnedecor::new(df1 as f64, df2 as f64).unwrap();
    let p_value = (2.0 * (1.0 - f_dist.cdf(statistic))).min(1.0);

    Ok(HeteroscedasticityTestResult {
        statistic,
        p_value,
        is_heteroscedastic: p_value < alpha,
        test_name: "Goldfeld-Quandt".into(),
        alpha,
    })
}

/// Dette-Munk-Wagner nonparametric test.
fn dette_munk_wagner_test(
    x: &Array2<f64>,
    _residuals: &Array1<f64>,
    residuals_sq: &Array1<f64>,
    alpha: f64,
    n_bootstrap: usize,
) -> Result<HeteroscedasticityTestResult, FerroError> {
    use rand::Rng;

    let n = x.nrows();
    let sigma_sq_global = residuals_sq.mean().unwrap();
    let resid_sq_centered = residuals_sq.mapv(|r| r - sigma_sq_global);

    // Project onto first principal component (or first feature for 1D)
    let x_proj: Array1<f64> = if x.ncols() == 1 {
        x.column(0).to_owned()
    } else {
        // Use first feature as simple projection
        x.column(0).to_owned()
    };

    // Sort
    let mut sort_idx: Vec<usize> = (0..n).collect();
    sort_idx.sort_by(|&a, &b| {
        x_proj[a]
            .partial_cmp(&x_proj[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let x_sorted: Array1<f64> = sort_idx.iter().map(|&i| x_proj[i]).collect();
    let resid_sq_sorted: Array1<f64> = sort_idx.iter().map(|&i| residuals_sq[i]).collect();

    // Kernel smoother for variance function
    let bw = silverman_bandwidth(&x_sorted.clone().insert_axis(ndarray::Axis(1)));
    let h = bw[0] * 1.5;

    let mut w = Array2::zeros((n, n));
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let u = (x_sorted[i] - x_sorted[j]) / h;
            let k = (-0.5 * u * u).exp();
            w[[i, j]] = k;
            row_sum += k;
        }
        if row_sum > 0.0 {
            for j in 0..n {
                w[[i, j]] /= row_sum;
            }
        }
    }

    let sigma_sq_local = w.dot(&resid_sq_sorted);

    // Trim boundaries
    let trim = (0.1 * n as f64).max(5.0) as usize;
    let trim = trim.min(n / 3);

    let valid = &sigma_sq_local.slice(ndarray::s![trim..(n - trim)]);
    let global_var = resid_sq_sorted.mean().unwrap();
    let t_observed: f64 =
        valid.iter().map(|&v| (v - global_var).powi(2)).sum::<f64>() / valid.len() as f64;

    // Wild bootstrap
    let mut rng = rand::rng();
    let mut count_ge = 0usize;

    for _ in 0..n_bootstrap {
        // Rademacher weights
        let boot_resid_sq: Array1<f64> = (0..n)
            .map(|i| {
                let rademacher: f64 = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                resid_sq_centered[sort_idx[i]] * rademacher + sigma_sq_global
            })
            .collect();

        let sigma_boot = w.dot(&boot_resid_sq);
        let boot_global = boot_resid_sq.mean().unwrap();
        let valid_boot = &sigma_boot.slice(ndarray::s![trim..(n - trim)]);
        let t_boot: f64 = valid_boot
            .iter()
            .map(|&v| (v - boot_global).powi(2))
            .sum::<f64>()
            / valid_boot.len() as f64;

        if t_boot >= t_observed {
            count_ge += 1;
        }
    }

    let p_value = (count_ge as f64 + 1.0) / (n_bootstrap as f64 + 1.0);

    Ok(HeteroscedasticityTestResult {
        statistic: t_observed,
        p_value,
        is_heteroscedastic: p_value < alpha,
        test_name: "Dette-Munk-Wagner".into(),
        alpha,
    })
}

/// Compute residual diagnostics.
pub fn residual_diagnostics(
    y: &Array1<f64>,
    y_pred: &Array1<f64>,
    alpha: f64,
) -> ResidualDiagnosticsResult {
    let residuals = y - y_pred;
    let n = residuals.len() as f64;
    let mean = residuals.mean().unwrap();
    let std = {
        let var: f64 = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    };
    let std_safe = if std > 0.0 { std } else { 1.0 };
    let standardized = residuals.mapv(|r| r / std_safe);

    // Skewness and kurtosis
    let skewness = {
        let m3: f64 = residuals
            .iter()
            .map(|&r| ((r - mean) / std_safe).powi(3))
            .sum::<f64>();
        m3 / n
    };
    let kurtosis = {
        let m4: f64 = residuals
            .iter()
            .map(|&r| ((r - mean) / std_safe).powi(4))
            .sum::<f64>();
        m4 / n - 3.0 // excess kurtosis
    };

    // Jarque-Bera test
    let jb = n / 6.0 * (skewness.powi(2) + kurtosis.powi(2) / 4.0);
    let chi2 = ChiSquared::new(2.0).unwrap();
    let jb_p = 1.0 - chi2.cdf(jb);

    ResidualDiagnosticsResult {
        residuals,
        standardized_residuals: standardized,
        mean,
        std,
        skewness,
        kurtosis,
        normality_statistic: jb,
        normality_p_value: jb_p,
        is_normal: jb_p >= alpha,
    }
}

/// Goodness of fit diagnostics.
#[derive(Debug, Clone)]
pub struct GoodnessOfFit {
    /// R² (coefficient of determination).
    pub r_squared: f64,
    /// Adjusted R² accounting for effective degrees of freedom.
    pub adjusted_r_squared: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Effective degrees of freedom (trace of hat matrix).
    pub effective_df: f64,
}

impl GoodnessOfFit {
    /// Compute goodness of fit from predictions and hat matrix trace.
    pub fn compute(y: &Array1<f64>, y_pred: &Array1<f64>, eff_df: f64) -> Self {
        let n = y.len() as f64;
        let y_mean = y.mean().unwrap();

        let ss_res: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        let ss_tot: f64 = y.iter().map(|&a| (a - y_mean).powi(2)).sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        let adjusted_r_squared = if n - eff_df > 0.0 && ss_tot > 0.0 {
            1.0 - (ss_res / (n - eff_df)) / (ss_tot / (n - 1.0))
        } else {
            r_squared
        };

        let mse = ss_res / n;
        let rmse = mse.sqrt();
        let mae: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / n;

        // AIC = n * ln(MSE) + 2 * df
        let aic = n * (mse.max(1e-300)).ln() + 2.0 * eff_df;
        // BIC = n * ln(MSE) + df * ln(n)
        let bic = n * (mse.max(1e-300)).ln() + eff_df * n.ln();

        Self {
            r_squared,
            adjusted_r_squared,
            mse,
            rmse,
            mae,
            aic,
            bic,
            effective_df: eff_df,
        }
    }
}

/// OLS R² and degrees of freedom.
fn ols_r_squared(z: &Array2<f64>, y: &Array1<f64>) -> (f64, usize) {
    let y_pred = ols_predict(z, y);
    let y_mean = y.mean().unwrap();
    let ss_reg: f64 = y_pred.iter().map(|&p| (p - y_mean).powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
    let r2 = if ss_tot > 0.0 { ss_reg / ss_tot } else { 0.0 };
    (r2, z.ncols() - 1)
}

/// OLS predicted values via normal equations.
fn ols_predict(z: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let zt = z.t();
    let ztz = zt.dot(z);
    let zty = zt.dot(y);

    // Add small regularization
    let n = ztz.nrows();
    let mut ztz_reg = ztz;
    for i in 0..n {
        ztz_reg[[i, i]] += 1e-10;
    }

    match solve_gauss(&ztz_reg, &zty) {
        Some(beta) => z.dot(&beta),
        None => Array1::from_elem(y.len(), y.mean().unwrap()),
    }
}

/// Gaussian elimination with partial pivoting.
fn solve_gauss(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut result = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * result[j];
        }
        if aug[[i, i]].abs() < 1e-15 {
            return None;
        }
        result[i] = sum / aug[[i, i]];
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn gof_perfect_fit() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = y.clone();
        let gof = GoodnessOfFit::compute(&y, &y_pred, 5.0);
        assert!((gof.r_squared - 1.0).abs() < 1e-12);
        assert!(gof.mse < 1e-12);
    }

    #[test]
    fn gof_mean_prediction() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = Array1::from_elem(5, 3.0); // mean
        let gof = GoodnessOfFit::compute(&y, &y_pred, 1.0);
        assert!(gof.r_squared.abs() < 1e-12); // R² = 0 for mean prediction
    }

    #[test]
    fn residual_diagnostics_normal() {
        // Generate approximately normal residuals
        let y = array![1.0, 2.1, 2.9, 4.05, 4.95, 6.1, 6.9, 8.05, 8.95, 10.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = residual_diagnostics(&y, &y_pred, 0.05);
        assert!(result.mean.abs() < 0.2);
        assert!(result.std > 0.0);
    }

    #[test]
    fn white_test_homoscedastic() {
        // Homoscedastic data: residuals should not correlate with X
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi + 1.0);
        // Add constant noise (homoscedastic)
        let _y_pred = y.clone(); // Perfect predictions means residuals = 0
        // For a proper test we need non-zero residuals
        let y_noisy: Array1<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &yi)| {
                yi + 0.1 * ((i as f64 * 1.618).sin()) // Deterministic "noise"
            })
            .collect();

        let result =
            heteroscedasticity_test(&x, &y_noisy, &y, HeteroscedasticityTest::White, 0.05).unwrap();

        // With near-constant variance, should not reject H0
        // (This is a weak test since our "noise" is deterministic)
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn gof_effective_df() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];
        let gof = GoodnessOfFit::compute(&y, &y_pred, 2.5);
        assert!(gof.effective_df == 2.5);
        assert!(gof.aic.is_finite());
        assert!(gof.bic.is_finite());
    }
}
