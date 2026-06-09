//! Divergence audit (critic #2357): shrinkage covariance family vs scikit-learn 1.5.2.
//!
//! Pins VALUE divergences not already covered by `divergence_covariance.rs`.
//! Every expected value is from a LIVE `sklearn` 1.5.2 oracle call executed
//! from /tmp (R-CHAR-3) — NEVER copied from the ferrolearn side.
//!
//! FAILING guards (pinned real divergences, `#[ignore]` + tracking issue):
//!   - EmpiricalCovariance precision_ SEVERE blow-up on small-variance data
//!     (cholesky `reg=1e-8` vs sklearn exact `pinvh`)         — #2358
//!   - LedoitWolf shrinkage_ for n_features == 1 special case  — #2359
//!   - OAS         shrinkage_ for n_features == 1 special case  — #2360

use ferrolearn_core::traits::Fit;
use ferrolearn_covariance::{EmpiricalCovariance, LedoitWolf, OAS};
use ndarray::{Array2, array};

// ===========================================================================
// DIVERGENCE — EmpiricalCovariance precision_ SEVERE on small-variance data.
//
// sklearn `_empirical_covariance.py:216`:
//   `precision = linalg.pinvh(covariance, check_finite=False)`
// ferrolearn `covariance.rs:132-162` (`cholesky`) adds `reg = 1e-8` to the
// diagonal before inverting, so it returns `inv(cov + 1e-8*I)`. For a
// well-conditioned cov this is a ~1e-8 perturbation, but when the covariance
// has tiny eigenvalues (here entries ~1e-6) the additive `1e-8` is a large
// relative perturbation and the precision_ blows up.
//
// Fixture X (5x2, entries ~1e-3):
//   [[0.001,0.002],[0.003,0.001],[0.002,0.004],[0.005,0.002],[0.001,0.005]]
// Live oracle:
//   EmpiricalCovariance().fit(X).covariance_ =
//     [[2.24e-06,-1.12e-06],[-1.12e-06,2.16e-06]]
//   EmpiricalCovariance().fit(X).precision_ =
//     [[602678.5714285714, 312499.99999999994],
//      [312500.0,          624999.9999999999]]
//   ferrolearn returns precision_ ~[[598109.2, 308701.5],[308701.5, 620159.3]]
//   (max abs diff ~4569, ~0.76% relative) — DIVERGES >> 1e-7.
// ===========================================================================
#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit index compare vs oracle"
)]
fn divergence_precision_small_variance_blowup() {
    let x: Array2<f64> = array![
        [0.001, 0.002],
        [0.003, 0.001],
        [0.002, 0.004],
        [0.005, 0.002],
        [0.001, 0.005],
    ];
    // sklearn 1.5.2 live oracle:
    let sk_prec = [
        [602_678.571_428_571_4_f64, 312_499.999_999_999_94],
        [312_500.0, 624_999.999_999_999_9],
    ];
    let fitted = EmpiricalCovariance::<f64>::new().fit(&x, &()).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            // Relative tolerance 1e-6 (R-DEV-1 numerical parity).
            let denom = sk_prec[i][j].abs().max(1.0);
            assert!(
                (fitted.precision()[[i, j]] - sk_prec[i][j]).abs() / denom < 1e-6,
                "precision[{i},{j}]: ferro {} vs sklearn {}",
                fitted.precision()[[i, j]],
                sk_prec[i][j]
            );
        }
    }
}

// ===========================================================================
// DIVERGENCE — LedoitWolf shrinkage_ for n_features == 1.
//
// sklearn `_shrunk_covariance.py:30-33` (`_ledoit_wolf`):
//   `if len(X.shape) == 2 and X.shape[1] == 1: ... return ..., 0.0`
// i.e. for a single feature sklearn HARD-CODES `shrinkage_ = 0.0`.
// ferrolearn `covariance.rs:907-918` runs the general formula: with p=1,
// `delta = (s00 - mu)^2 / 1 = 0` (mu == s00), so it hits the
// `else { F::one() }` branch and returns `shrinkage_ = 1.0`.
//
// Fixture X = [[1],[2],[4],[8],[3]] (n=5, p=1).
// Live oracle: LedoitWolf().fit(X).shrinkage_ == 0.0
//   ferrolearn returns shrinkage_ == 1.0 — DIVERGES.
// ===========================================================================
#[test]
fn divergence_ledoit_wolf_single_feature_shrinkage() {
    let x: Array2<f64> = array![[1.0], [2.0], [4.0], [8.0], [3.0]];
    // sklearn 1.5.2 live oracle: LedoitWolf().fit(X).shrinkage_
    let sk_shrinkage = 0.0_f64;
    let fitted = LedoitWolf::<f64>::new().fit(&x, &()).unwrap();
    assert!(
        (fitted.shrinkage() - sk_shrinkage).abs() < 1e-9,
        "LedoitWolf p=1 shrinkage_: ferro {} vs sklearn {sk_shrinkage}",
        fitted.shrinkage()
    );
}

// ===========================================================================
// DIVERGENCE — OAS shrinkage_ for n_features == 1.
//
// sklearn `_shrunk_covariance.py:57-61` (`_oas`):
//   `if len(X.shape) == 2 and X.shape[1] == 1: ... return ..., 0.0`
// i.e. for a single feature sklearn HARD-CODES `shrinkage_ = 0.0`.
// ferrolearn `covariance.rs:1107-1118` runs the general formula: with p=1,
// `alpha = mu^2`, so `den = (n+1)*(alpha - mu^2/1) = 0`, hitting the
// `den == 0 -> F::one()` branch and returning `shrinkage_ = 1.0`.
//
// Fixture X = [[1],[2],[4],[8],[3]] (n=5, p=1).
// Live oracle: OAS().fit(X).shrinkage_ == 0.0
//   ferrolearn returns shrinkage_ == 1.0 — DIVERGES.
// ===========================================================================
#[test]
fn divergence_oas_single_feature_shrinkage() {
    let x: Array2<f64> = array![[1.0], [2.0], [4.0], [8.0], [3.0]];
    // sklearn 1.5.2 live oracle: OAS().fit(X).shrinkage_
    let sk_shrinkage = 0.0_f64;
    let fitted = OAS::<f64>::new().fit(&x, &()).unwrap();
    assert!(
        (fitted.shrinkage() - sk_shrinkage).abs() < 1e-9,
        "OAS p=1 shrinkage_: ferro {} vs sklearn {sk_shrinkage}",
        fitted.shrinkage()
    );
}
