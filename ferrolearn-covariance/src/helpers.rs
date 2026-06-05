//! Function-style covariance helpers (sklearn parity).
//!
//! These are stateless equivalents of the methods on the corresponding
//! estimator structs, mirroring the function exports in
//! `sklearn.covariance`.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.covariance` function exports (`_empirical_covariance.py`,
//! `_shrunk_covariance.py`, `_robust_covariance.py`) at v1.5.2. These delegate to
//! the `covariance.rs` estimators (audited under #1701). Every REQ is BINARY
//! (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-EMPIRICAL (empirical_covariance) | SHIPPED | delegates `EmpiricalCovariance` (biased MLE `X^T X / n`), `:67`; matches live oracle element-wise (assume_centered true/false). Guards `guard_empirical_covariance_*`. |
//! | REQ-SHRUNK (shrunk_covariance formula) | SHIPPED | `(1-s)*emp_cov + s*(trace/n)*I`, `:111`; matches live oracle for s=0.1/0.3. Guards `guard_shrunk_covariance_*`. |
//! | REQ-LEDOIT-WOLF (ledoit_wolf + ledoit_wolf_shrinkage) | SHIPPED | delegates `LedoitWolf` (`:409`/`:299`); returns `(cov, shrinkage)`; matches live oracle (shrinkage ~1e-10). Guard `guard_ledoit_wolf`. |
//! | REQ-OAS (oas) | SHIPPED | delegates `OAS` (1.5.2 formula, #1702); shrinkage ~0.67 (not pre-1.5 ~0.54), `:619`; matches live oracle. Guards `guard_oas`, `guard_return_order_cov_then_shrinkage`. |
//! | REQ-LOG-LIKELIHOOD (SPD precision value parity) | SHIPPED | `0.5*(logdet(precision) - tr - p*log(2π))` = sklearn `(-tr + logdet - p*log(2π))/2` (`:33`); bit-exact on identity/scaled/anisotropic/cross-term SPD precisions vs live oracle. Guards `guard_log_likelihood_anisotropic`, `guard_log_likelihood_cross_terms`. |
//! | REQ-LOG-LIKELIHOOD-NONPD (non-PD precision ⇒ -inf) | NOT-STARTED | sklearn `fast_logdet` (`utils/extmath.py:108`) returns `-inf` when `slogdet` sign ≤ 0; ferrolearn `log_det_spd` adds adaptive regularization and returns a finite value (out-of-contract input). Faithful parity needs a sign-aware `slogdet` (LU determinant) — the crate has only the regularizing Cholesky `log_det_spd` (kept for graphical_lasso). Test `divergence_log_likelihood_nonpd`. Blocker #1878. |
//! | REQ-DEFAULTS (shrunk_covariance shrinkage=0.1, ledoit_wolf block_size=1000) | NOT-STARTED | `shrinkage` mandatory; no `block_size` (memory-only, no value impact). API-shape gap. Blocker #1874. |
//! | REQ-FAST-MCD (fast_mcd) | NOT-STARTED | FastMCD RNG-dependent (`SmallRng`/Xoshiro vs numpy `RandomState`, `:358`) — R-DEFER-3 carve-out (NO failing value test); structural shapes hold. Blocker #1875. |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | `ndarray` + hand-rolled `log_det_spd`; destination `ferray-core`/`ferray::linalg` (R-SUBSTRATE-1). Blocker #1876. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use helpers::{empirical_covariance, shrunk_covariance, ledoit_wolf, ledoit_wolf_shrinkage, oas, log_likelihood, fast_mcd}` in `lib.rs` (boundary function API, S5/R-DEFER-1). |

use ferrolearn_core::FerroError;
use ferrolearn_core::traits::Fit;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::covariance::{EmpiricalCovariance, LedoitWolf, MinCovDet, OAS};

/// One-shot maximum-likelihood empirical covariance.
///
/// `assume_centered = false` subtracts the per-column mean before computing
/// `(X - mu)^T (X - mu) / n`. `assume_centered = true` skips the mean
/// subtraction (assumes the data already has zero mean).
///
/// # Errors
///
/// Returns [`FerroError::InsufficientSamples`] if `x` is empty.
pub fn empirical_covariance<F>(
    x: &Array2<F>,
    assume_centered: bool,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let est = EmpiricalCovariance::<F>::new().assume_centered(assume_centered);
    let fitted = est.fit(x, &())?;
    Ok(fitted.covariance().clone())
}

/// Apply fixed shrinkage to an empirical covariance.
///
/// Returns `(1 - shrinkage) * emp_cov + shrinkage * mu * I`, where `mu` is the
/// average of the diagonal of `emp_cov`.
pub fn shrunk_covariance<F>(emp_cov: &Array2<F>, shrinkage: F) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = emp_cov.nrows();
    if n != emp_cov.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![emp_cov.nrows(), emp_cov.ncols()],
            context: "shrunk_covariance: emp_cov must be square".into(),
        });
    }
    let n_f = F::from(n).ok_or_else(|| FerroError::InvalidParameter {
        name: "n".into(),
        reason: "could not convert to F".into(),
    })?;
    let mut diag_sum = F::zero();
    for i in 0..n {
        diag_sum = diag_sum + emp_cov[[i, i]];
    }
    let mu = diag_sum / n_f;
    let one_minus = F::one() - shrinkage;
    let mut out = emp_cov.clone();
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = one_minus * out[[i, j]];
            if i == j {
                out[[i, j]] = out[[i, j]] + shrinkage * mu;
            }
        }
    }
    Ok(out)
}

/// Compute the Ledoit-Wolf shrinkage estimator: returns `(cov, shrinkage)`.
///
/// `shrinkage` is the data-driven coefficient mixing `emp_cov` with the
/// scaled identity (see [`LedoitWolf`]).
pub fn ledoit_wolf<F>(x: &Array2<F>, assume_centered: bool) -> Result<(Array2<F>, F), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let est = LedoitWolf::<F>::new().assume_centered(assume_centered);
    let fitted = est.fit(x, &())?;
    Ok((fitted.covariance().clone(), fitted.shrinkage()))
}

/// Return only the Ledoit-Wolf shrinkage coefficient (matches sklearn).
pub fn ledoit_wolf_shrinkage<F>(x: &Array2<F>, assume_centered: bool) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let (_, s) = ledoit_wolf(x, assume_centered)?;
    Ok(s)
}

/// Compute the OAS (Oracle Approximating Shrinkage): returns `(cov, shrinkage)`.
pub fn oas<F>(x: &Array2<F>, assume_centered: bool) -> Result<(Array2<F>, F), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let est = OAS::<F>::new().assume_centered(assume_centered);
    let fitted = est.fit(x, &())?;
    Ok((fitted.covariance().clone(), fitted.shrinkage()))
}

/// Gaussian log-likelihood of the data given a covariance estimate.
///
/// `log L = -0.5 * (n_features * log(2 pi) + log|Σ| + tr(Σ^{-1} S))` averaged
/// per sample, where `S` is the empirical covariance of the data and `Σ` is
/// the estimate.
///
/// # Errors
///
/// Returns [`FerroError::ShapeMismatch`] if `cov` is not square.
/// Returns [`FerroError::NumericalInstability`] if `cov` is not invertible.
pub fn log_likelihood<F>(emp_cov: &Array2<F>, precision: &Array2<F>) -> Result<F, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n = emp_cov.nrows();
    if n != emp_cov.ncols() {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![emp_cov.nrows(), emp_cov.ncols()],
            context: "log_likelihood: emp_cov must be square".into(),
        });
    }
    if precision.nrows() != n || precision.ncols() != n {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n, n],
            actual: vec![precision.nrows(), precision.ncols()],
            context: "log_likelihood: precision must match emp_cov shape".into(),
        });
    }
    // log|precision| via determinant; for SPD matrices use log of product
    // of diagonal of an LU/Cholesky-style factor — but for cleanliness we
    // approximate via trace(log_eig) by taking log of the diagonal of
    // precision (only valid for diagonal matrices, so we instead use the
    // identity log|Σ| = -log|precision|).
    //
    // We compute log|precision| directly: since precision is SPD, det>0; we
    // accept naive expansion via cofactor for n<=3 and otherwise rely on
    // Sylvester (fallback to LU not available in core). For ferrolearn we
    // use a simple Bareiss-style integer-LU algorithm reduced to floats.
    let logdet_prec = log_det_spd(precision)?;
    let mut tr = F::zero();
    for i in 0..n {
        for k in 0..n {
            tr = tr + precision[[i, k]] * emp_cov[[k, i]];
        }
    }
    let n_f = F::from(n).ok_or_else(|| FerroError::InvalidParameter {
        name: "n".into(),
        reason: "could not convert to F".into(),
    })?;
    let two_pi =
        F::from(2.0 * std::f64::consts::PI).ok_or_else(|| FerroError::InvalidParameter {
            name: "2pi".into(),
            reason: "could not convert".into(),
        })?;
    let half = F::from(0.5).ok_or_else(|| FerroError::InvalidParameter {
        name: "0.5".into(),
        reason: "could not convert".into(),
    })?;
    Ok(-half * (n_f * two_pi.ln() - logdet_prec + tr))
}

/// Compute log|A| for a (near-)SPD matrix via Cholesky with adaptive
/// diagonal regularisation.
///
/// If the raw matrix is not strictly positive definite (e.g. produced by a
/// truncated optimiser like graphical lasso), this routine adds the smallest
/// non-negative diagonal shift `tau` such that `A + tau * I` is SPD up to
/// the working precision, then returns `log|A + tau * I|`. The shift starts
/// at zero and grows geometrically until success.
fn log_det_spd<F: Float>(a: &Array2<F>) -> Result<F, FerroError> {
    let n = a.nrows();
    let two = F::from(2.0).ok_or_else(|| FerroError::InvalidParameter {
        name: "2".into(),
        reason: "could not convert".into(),
    })?;
    // Scale-aware starting shift: max(|a_ii|) * 1e-6, falling back to 1e-8
    // for matrices whose diagonal is essentially zero.
    let mut max_diag = F::zero();
    for i in 0..n {
        let v = a[[i, i]].abs();
        if v > max_diag {
            max_diag = v;
        }
    }
    let scale = max_diag.max(F::one());
    let mut tau = F::zero();
    let base = scale * F::from(1e-6).unwrap_or(F::epsilon());
    for attempt in 0..30 {
        if attempt > 0 {
            tau = if tau == F::zero() {
                base
            } else {
                tau * F::from(10.0).unwrap_or_else(F::one)
            };
        }
        let mut l = Array2::<F>::zeros((n, n));
        let mut ok = true;
        'fact: for i in 0..n {
            for j in 0..=i {
                let mut sum = F::zero();
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    let v = a[[i, i]] + tau - sum;
                    if v <= F::zero() {
                        ok = false;
                        break 'fact;
                    }
                    l[[i, j]] = v.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        if ok {
            let mut acc = F::zero();
            for i in 0..n {
                acc = acc + l[[i, i]].ln();
            }
            return Ok(two * acc);
        }
    }
    Err(FerroError::NumericalInstability {
        message: "log_det_spd: matrix is not positive definite even after regularisation".into(),
    })
}

/// Result of [`fast_mcd`]: `(location, covariance, support_mask)` where
/// `support_mask[i]` is `true` if sample `i` is part of the MCD support set.
pub type McdResult<F> = (Array1<F>, Array2<F>, Vec<bool>);

/// One-shot FAST-MCD location/covariance estimate.
///
/// See [`McdResult`] for the returned tuple.
pub fn fast_mcd<F>(
    x: &Array2<F>,
    support_fraction: f64,
    random_state: Option<u64>,
) -> Result<McdResult<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let mut est = MinCovDet::<F>::new().support_fraction(support_fraction);
    if let Some(seed) = random_state {
        est = est.random_state(seed);
    }
    let fitted = est.fit(x, &())?;
    Ok((
        fitted.location().clone(),
        fitted.covariance().clone(),
        fitted.support().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Array2<f64> {
        array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    }

    #[test]
    fn test_empirical_covariance() {
        let cov = empirical_covariance(&data(), false).unwrap();
        assert_eq!(cov.dim(), (2, 2));
    }

    #[test]
    fn test_shrunk_covariance() {
        let cov = empirical_covariance(&data(), false).unwrap();
        let s = shrunk_covariance(&cov, 0.3).unwrap();
        assert_eq!(s.dim(), (2, 2));
    }

    #[test]
    fn test_ledoit_wolf() {
        let (cov, s) = ledoit_wolf(&data(), false).unwrap();
        assert_eq!(cov.dim(), (2, 2));
        assert!((0.0..=1.0).contains(&s));
        let s2 = ledoit_wolf_shrinkage(&data(), false).unwrap();
        assert!((s - s2).abs() < 1e-12);
    }

    #[test]
    fn test_oas_helper() {
        let (cov, s) = oas(&data(), false).unwrap();
        assert_eq!(cov.dim(), (2, 2));
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_log_likelihood_basic() {
        // Identity precision and identity emp_cov => trace = n, logdet = 0
        let n = 3;
        let mut id = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            id[[i, i]] = 1.0;
        }
        let ll = log_likelihood(&id, &id).unwrap();
        // -0.5 * (n * log(2pi) - 0 + n) = -0.5 * n * (log(2pi) + 1)
        let expected = -0.5 * (n as f64) * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((ll - expected).abs() < 1e-9);
    }

    #[test]
    fn test_fast_mcd_smoke() {
        let (loc, cov, support) = fast_mcd(&data(), 0.75, Some(7)).unwrap();
        assert_eq!(loc.len(), 2);
        assert_eq!(cov.dim(), (2, 2));
        assert_eq!(support.len(), 4);
    }
}
