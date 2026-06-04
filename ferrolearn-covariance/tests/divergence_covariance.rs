//! Divergence audit: ferrolearn-covariance vs scikit-learn 1.5.2.
//!
//! GREEN guards (should PASS — confirm shipped value parity):
//!   - EmpiricalCovariance covariance_/location_ (REQ-1)
//!   - ShrunkCovariance    (REQ-2)
//!   - LedoitWolf          (REQ-3)
//!
//! FAILING guards (pinned real divergences, `#[ignore]` + tracking issue):
//!   - EmpiricalCovariance precision_ regularization (REQ-1/AC-1, #1701)
//!   - OAS formula            (REQ-4,  #1701)
//!   - mahalanobis squared    (REQ-13, #1701)
//!   - contamination = 0.5    (REQ-11, #1701)
//!
//! Every expected value below is from a LIVE `sklearn` 1.5.2 oracle call
//! executed from /tmp (R-CHAR-3) — NEVER copied from the ferrolearn side.
//! The probe sets:
//!   Xa (asymmetric) = [[1,2,3],[4,5,6],[7,8,10],[2,3,5],[3,1,2],[5,9,4]]
//!   Xs (symmetric square) = [[0,0],[2,0],[0,2],[2,2]]

use ferrolearn_core::traits::Fit;
use ferrolearn_covariance::{
    EllipticEnvelope, EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance,
};
use ndarray::{Array2, array};

fn probe_asym() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0],
        [2.0, 3.0, 5.0],
        [3.0, 1.0, 2.0],
        [5.0, 9.0, 4.0],
    ]
}

fn probe_sym() -> Array2<f64> {
    array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
}

// ===========================================================================
// GREEN GUARD — REQ-1: EmpiricalCovariance MLE (ddof=0), covariance_ + location_
// Oracle: EmpiricalCovariance().fit(Xa) — sklearn 1.5.2
//   _empirical_covariance.py:67  `covariance = np.dot(X.T, X) / X.shape[0]`
// (precision_ is pinned separately as a divergence — see below.)
// ===========================================================================
#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit i/j index compare of ndarray vs nested-Vec sklearn oracle"
)]
fn green_empirical_covariance_parity() {
    // sklearn live oracle (Xa):
    let sk_cov = [
        [3.8888888888888893, 4.888888888888888, 3.833333333333333],
        [4.888888888888888, 8.88888888888889, 4.833333333333334],
        [3.833333333333333, 4.833333333333334, 6.666666666666666],
    ];
    let sk_loc = [3.6666666666666665, 4.666666666666667, 5.0];

    let fitted = EmpiricalCovariance::<f64>::new()
        .fit(&probe_asym(), &())
        .unwrap();
    for i in 0..3 {
        assert!(
            (fitted.location()[i] - sk_loc[i]).abs() < 1e-9,
            "location[{i}]"
        );
        for j in 0..3 {
            assert!(
                (fitted.covariance()[[i, j]] - sk_cov[i][j]).abs() < 1e-9,
                "covariance[{i},{j}]"
            );
        }
    }
}

// ===========================================================================
// FAILING — REQ-1 / AC-1: EmpiricalCovariance precision_ misses 1e-10 parity.
//
// AC-1 claims `precision_` matches sklearn element-wise within 1e-10.
// It does NOT. ferrolearn's `spd_inverse` -> `cholesky` (covariance.rs:99)
// adds `reg = 1e-8` to the diagonal BEFORE inverting, so the returned
// precision is `inv(cov + 1e-8*I)`, not `inv(cov)`. sklearn's
// `EmpiricalCovariance.get_precision` uses exact `linalg.pinvh(covariance_)`
// (`_empirical_covariance.py:216`) with no regularization.
//
// Live oracle EmpiricalCovariance().fit(Xa).precision_[0][0]
//   = 1.1653140967838913.
// ferrolearn returns 1.1653140799702713 (diff ~1.68e-8, exceeds 1e-10).
// Tracking: #1701
// ===========================================================================
#[test]
#[ignore = "divergence: precision_ inverts cov+1e-8*I (cholesky reg) vs sklearn exact pinvh _empirical_covariance.py:216; tracking #1701"]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit i/j index compare of ndarray vs nested-Vec sklearn oracle"
)]
fn divergence_empirical_precision_regularization() {
    // sklearn 1.5.2 live oracle (Xa) precision_:
    let sk_prec = [
        [
            1.1653140967838913,
            -0.4565674782085973,
            -0.33904418394950464,
        ],
        [
            -0.4565674782085972,
            0.3645927261797419,
            -0.001803426510369495,
        ],
        [
            -0.33904418394950464,
            -0.0018034265103694604,
            0.34625788999098306,
        ],
    ];
    let fitted = EmpiricalCovariance::<f64>::new()
        .fit(&probe_asym(), &())
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fitted.precision()[[i, j]] - sk_prec[i][j]).abs() < 1e-10,
                "precision[{i},{j}]: ferro {} vs sklearn {} (AC-1 tol 1e-10)",
                fitted.precision()[[i, j]],
                sk_prec[i][j]
            );
        }
    }
}

// ===========================================================================
// GREEN GUARD — REQ-2: ShrunkCovariance(shrinkage=0.1)
// Oracle: ShrunkCovariance(shrinkage=0.1).fit(Xa) — sklearn 1.5.2
//   _shrunk_covariance.py:153-156
// ===========================================================================
#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit i/j index compare of ndarray vs nested-Vec sklearn oracle"
)]
fn green_shrunk_covariance_parity() {
    let sk_cov = [
        [4.148148148148149, 4.3999999999999995, 3.4499999999999997],
        [4.3999999999999995, 8.648148148148149, 4.3500000000000005],
        [3.4499999999999997, 4.3500000000000005, 6.648148148148148],
    ];
    let fitted = ShrunkCovariance::<f64>::new(0.1)
        .fit(&probe_asym(), &())
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fitted.covariance()[[i, j]] - sk_cov[i][j]).abs() < 1e-9,
                "covariance[{i},{j}]"
            );
        }
    }
}

// ===========================================================================
// GREEN GUARD — REQ-3: LedoitWolf shrinkage_ + covariance_
// Oracle: LedoitWolf().fit(Xa) — sklearn 1.5.2
// ===========================================================================
#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit i/j index compare of ndarray vs nested-Vec sklearn oracle"
)]
fn green_ledoit_wolf_parity() {
    let sk_shrinkage = 0.4087180724344861;
    let sk_cov = [
        [4.94852833594126, 2.8907116458758453, 2.26658072233447],
        [2.8907116458758453, 7.9049379737688295, 2.8578626498999844],
        [2.26658072233447, 2.8578626498999844, 6.590978134734355],
    ];
    let fitted = LedoitWolf::<f64>::new().fit(&probe_asym(), &()).unwrap();
    assert!(
        (fitted.shrinkage() - sk_shrinkage).abs() < 1e-9,
        "shrinkage_: ferro {} vs sklearn {sk_shrinkage}",
        fitted.shrinkage()
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fitted.covariance()[[i, j]] - sk_cov[i][j]).abs() < 1e-9,
                "covariance[{i},{j}]"
            );
        }
    }
}

// ===========================================================================
// FAILING — REQ-4 (HEADLINE): OAS uses the pre-1.5 formula.
//
// Divergence: ferrolearn's `OAS::fit` (covariance.rs:893-910) computes the
// SUPERSEDED OAS shrinkage `rho_num = (1 - 2/p)*tr(S^2) + tr(S)^2`,
// `rho_den = (n + 1 - 2/p)*(tr(S^2) - tr(S)^2/p)`. sklearn 1.5.2's
// `oas` (`_shrunk_covariance.py:79-87`):
//   `alpha = np.mean(emp_cov**2)`  (:79)
//   `mu = np.trace(emp_cov)/n_features`  (:80)
//   `num = alpha + mu_squared`  (:85)
//   `den = (n_samples + 1) * (alpha - mu_squared / n_features)`  (:86)
//   `shrinkage = 1.0 if den == 0 else min(num / den, 1.0)`  (:87)
//
// Live oracle OAS().fit(Xa): shrinkage_=0.6705854984555868,
//   covariance_[0][0]=5.627443884884855.
// ferrolearn returns shrinkage ~0.5387, covariance[0][0] ~5.2855 — DIVERGES.
// Tracking: #1701
// ===========================================================================
#[test]
#[ignore = "divergence: OAS uses pre-1.5 formula vs sklearn _shrunk_covariance.py:79-87; tracking #1701"]
#[allow(
    clippy::needless_range_loop,
    reason = "explicit i/j index compare of ndarray vs nested-Vec sklearn oracle"
)]
fn divergence_oas_formula() {
    // sklearn 1.5.2 live oracle (Xa):
    let sk_shrinkage = 0.6705854984555868;
    let sk_cov = [
        [5.627443884884855, 1.6104708964393533, 1.262755589253584],
        [1.6104708964393533, 7.274516392606921, 1.5921700907979974],
        [1.262755589253584, 1.5921700907979974, 6.542484166952669],
    ];
    let fitted = OAS::<f64>::new().fit(&probe_asym(), &()).unwrap();
    assert!(
        (fitted.shrinkage() - sk_shrinkage).abs() < 1e-9,
        "OAS shrinkage_: ferro {} vs sklearn {sk_shrinkage}",
        fitted.shrinkage()
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fitted.covariance()[[i, j]] - sk_cov[i][j]).abs() < 1e-9,
                "OAS covariance[{i},{j}]: ferro {} vs sklearn {}",
                fitted.covariance()[[i, j]],
                sk_cov[i][j]
            );
        }
    }
}

// ===========================================================================
// FAILING — REQ-13: mahalanobis returns SQUARED distances in sklearn.
//
// Divergence: ferrolearn's `FittedCovariance::mahalanobis`
// (covariance.rs:196 `dists[i] = val.abs().sqrt()`) returns the
// NON-squared Mahalanobis distance. sklearn's
// `EmpiricalCovariance.mahalanobis` returns the SQUARED distance
// (`_empirical_covariance.py:341` docstring "Squared Mahalanobis distances",
//  :353 "Squared Mahalanobis distances of the observations").
//
// Live oracle EmpiricalCovariance().fit(Xs).mahalanobis(Xs):
//   [2.0, 2.0, 2.0, 2.0]  (squared).
// ferrolearn returns sqrt(2) ~ 1.4142 each.
// Tracking: #1701
// ===========================================================================
#[test]
#[ignore = "divergence: mahalanobis returns sqrt not squared vs sklearn _empirical_covariance.py:340,353; tracking #1701"]
fn divergence_mahalanobis_squared() {
    // sklearn 1.5.2 live oracle (Xs): squared Mahalanobis = 2.0 for each row.
    let sk_mahal = [
        2.0000000000000004,
        2.0000000000000004,
        2.0000000000000004,
        2.0000000000000004,
    ];
    let x = probe_sym();
    let fitted = EmpiricalCovariance::<f64>::new().fit(&x, &()).unwrap();
    let dists = fitted.mahalanobis(&x).unwrap();
    for i in 0..4 {
        assert!(
            (dists[i] - sk_mahal[i]).abs() < 1e-9,
            "mahalanobis[{i}]: ferro {} vs sklearn (squared) {}",
            dists[i],
            sk_mahal[i]
        );
    }
}

// ===========================================================================
// FAILING — REQ-11: EllipticEnvelope(contamination=0.5) must fit.
//
// Divergence: ferrolearn's `EllipticEnvelope::fit` (covariance.rs:1548)
// rejects `contamination >= 0.5`. sklearn's `_parameter_constraints`
// (`_elliptic_envelope.py:147`) declares
// `"contamination": [Interval(Real, 0, 0.5, closed="right")]` — the right
// endpoint 0.5 is VALID. Live oracle: `EllipticEnvelope(contamination=0.5,
// random_state=0).fit(Xa)` succeeds (no error). ferrolearn over-rejects.
// Tracking: #1701
// ===========================================================================
#[test]
#[ignore = "divergence: contamination=0.5 over-rejected vs sklearn _elliptic_envelope.py:147 closed-right; tracking #1701"]
fn divergence_contamination_right_endpoint() {
    // sklearn accepts contamination == 0.5 (closed right). ferrolearn must too.
    let result = EllipticEnvelope::<f64>::new()
        .contamination(0.5)
        .random_state(0)
        .fit(&probe_asym(), &());
    assert!(
        result.is_ok(),
        "contamination=0.5 must fit (sklearn closed-right interval); ferrolearn returned {:?}",
        result.err()
    );
}
