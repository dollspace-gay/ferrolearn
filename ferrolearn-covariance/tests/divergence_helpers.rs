//! ACToR critic guards for `ferrolearn-covariance/src/helpers.rs` vs the LIVE
//! scikit-learn 1.5.2 function oracle (`sklearn.covariance`).
//!
//! Every expected value below is the verbatim output of the live sklearn oracle
//! (sklearn 1.5.2), computed independently of the ferrolearn side (R-CHAR-3 —
//! NEVER copied from ferrolearn). Reproduction commands are inlined per test.
//!
//! Probe matrix (shared with `.design/covariance/helpers.md`):
//!   X = [[1,2,3],[4,5,6],[7,8,10],[2,3,5],[3,1,2],[5,9,4]]

use ferrolearn_covariance::{
    empirical_covariance, ledoit_wolf, ledoit_wolf_shrinkage, log_likelihood, oas,
    shrunk_covariance,
};
use ndarray::{Array2, array};

const TOL: f64 = 1e-10;

fn probe_x() -> Array2<f64> {
    array![
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 10.],
        [2., 3., 5.],
        [3., 1., 2.],
        [5., 9., 4.]
    ]
}

fn assert_close(a: f64, b: f64, tol: f64, ctx: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{ctx}: ferrolearn={a} vs sklearn={b} (|delta|={})",
        (a - b).abs()
    );
}

// ---------------------------------------------------------------------------
// REQ-EMPIRICAL — empirical_covariance value parity.
// Oracle:
//   python3 -c "import numpy as np; from sklearn.covariance import \
//   empirical_covariance; X=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],\
//   [2.,3.,5.],[3.,1.,2.],[5.,9.,4.]]); print(empirical_covariance(X).tolist())"
// => full 3x3 below.
// ---------------------------------------------------------------------------
#[test]
fn guard_empirical_covariance_centered() {
    // Live sklearn 1.5.2 oracle (assume_centered=False, the default).
    let sk = [
        [3.8888888888888893, 4.888888888888888, 3.833333333333333],
        [4.888888888888888, 8.88888888888889, 4.833333333333334],
        [3.833333333333333, 4.833333333333334, 6.666666666666666],
    ];
    let cov = empirical_covariance(&probe_x(), false).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(cov[[i, j]], sk[i][j], TOL, &format!("emp[{i},{j}]"));
        }
    }
}

#[test]
fn guard_empirical_covariance_assume_centered() {
    // Live sklearn oracle: empirical_covariance(X, assume_centered=True) = X.T@X/n
    let sk = [
        [17.333333333333332, 22.0, 22.166666666666668],
        [22.0, 30.666666666666668, 28.166666666666668],
        [22.166666666666668, 28.166666666666668, 31.666666666666668],
    ];
    let cov = empirical_covariance(&probe_x(), true).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(cov[[i, j]], sk[i][j], TOL, &format!("empc[{i},{j}]"));
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-SHRUNK — shrunk_covariance: (1-s)*emp_cov + s*(trace/p)*I.
// Oracle:
//   shrunk_covariance(empirical_covariance(X), shrinkage=s)
// ---------------------------------------------------------------------------
#[test]
fn guard_shrunk_covariance_s01() {
    // Live sklearn oracle, shrinkage=0.1.
    let sk = [
        [4.148148148148149, 4.3999999999999995, 3.4499999999999997],
        [4.3999999999999995, 8.648148148148149, 4.3500000000000005],
        [3.4499999999999997, 4.3500000000000005, 6.648148148148148],
    ];
    let ec = empirical_covariance(&probe_x(), false).unwrap();
    let sh = shrunk_covariance(&ec, 0.1).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(sh[[i, j]], sk[i][j], TOL, &format!("shrunk0.1[{i},{j}]"));
        }
    }
}

#[test]
fn guard_shrunk_covariance_s03() {
    // Live sklearn oracle, shrinkage=0.3.
    let sk = [
        [4.666666666666666, 3.4222222222222216, 2.683333333333333],
        [3.4222222222222216, 8.166666666666666, 3.3833333333333337],
        [2.683333333333333, 3.3833333333333337, 6.611111111111111],
    ];
    let ec = empirical_covariance(&probe_x(), false).unwrap();
    let sh = shrunk_covariance(&ec, 0.3).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_close(sh[[i, j]], sk[i][j], TOL, &format!("shrunk0.3[{i},{j}]"));
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-LEDOIT-WOLF — ledoit_wolf -> (cov, shrinkage); ledoit_wolf_shrinkage scalar.
// Oracle:
//   c,s = ledoit_wolf(X); print(s, c.tolist())
// ---------------------------------------------------------------------------
#[test]
fn guard_ledoit_wolf() {
    let sk_shrinkage = 0.4087180724344861;
    let sk_cov = [
        [4.94852833594126, 2.8907116458758453, 2.26658072233447],
        [2.8907116458758453, 7.9049379737688295, 2.8578626498999844],
        [2.26658072233447, 2.8578626498999844, 6.590978134734355],
    ];
    let (cov, s) = ledoit_wolf(&probe_x(), false).unwrap();
    assert_close(s, sk_shrinkage, 1e-10, "lw_shrinkage");
    for i in 0..3 {
        for j in 0..3 {
            assert_close(
                cov[[i, j]],
                sk_cov[i][j],
                1e-10,
                &format!("lw_cov[{i},{j}]"),
            );
        }
    }
    // ledoit_wolf_shrinkage returns the same scalar.
    let s2 = ledoit_wolf_shrinkage(&probe_x(), false).unwrap();
    assert_close(s2, sk_shrinkage, 1e-10, "lw_shrinkage_scalar");
}

// ---------------------------------------------------------------------------
// REQ-OAS — oas -> (cov, shrinkage). Exercises the #1702 / 1.5.2 OAS formula:
// shrinkage ~ 0.67 (NOT the pre-1.5 ~0.54). Confirms tuple order (cov, shrinkage).
// Oracle:
//   c,s = oas(X); print(s, c.tolist())
// ---------------------------------------------------------------------------
#[test]
fn guard_oas() {
    let sk_shrinkage = 0.6705854984555868;
    let sk_cov = [
        [5.627443884884855, 1.6104708964393533, 1.262755589253584],
        [1.6104708964393533, 7.274516392606921, 1.5921700907979974],
        [1.262755589253584, 1.5921700907979974, 6.542484166952669],
    ];
    let (cov, s) = oas(&probe_x(), false).unwrap();
    assert_close(
        s,
        sk_shrinkage,
        1e-10,
        "oas_shrinkage (1.5.2 #1702 formula)",
    );
    for i in 0..3 {
        for j in 0..3 {
            assert_close(
                cov[[i, j]],
                sk_cov[i][j],
                1e-10,
                &format!("oas_cov[{i},{j}]"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-LOG-LIKELIHOOD — log_likelihood value parity.
// sklearn: (-sum(emp_cov*precision) + fast_logdet(precision) - p*log(2pi))/2.
// Three SPD precisions incl. an ANISOTROPIC (non-diagonal, distinct eigenvalue)
// case so the -tr cross-term AND logdet both contribute non-degenerately.
// Oracle commands inline.
// ---------------------------------------------------------------------------
#[test]
fn guard_log_likelihood_identity() {
    // log_likelihood(eye(3), eye(3)) = -4.2568155996140185
    let id = Array2::<f64>::eye(3);
    let ll = log_likelihood(&id, &id).unwrap();
    assert_close(ll, -4.2568155996140185, 1e-10, "ll(I,I)");
}

#[test]
fn guard_log_likelihood_scaled() {
    // log_likelihood(0.5*eye(3), 2.0*eye(3)) = -3.2170948287741004
    let emp = Array2::<f64>::eye(3) * 0.5;
    let prec = Array2::<f64>::eye(3) * 2.0;
    let ll = log_likelihood(&emp, &prec).unwrap();
    assert_close(ll, -3.2170948287741004, 1e-10, "ll(0.5I,2I)");
}

#[test]
fn guard_log_likelihood_anisotropic() {
    // ADVERSARIAL: anisotropic SPD covariance, precision = its inverse.
    //   ec = [[2,0.6,0.3],[0.6,1.5,-0.2],[0.3,-0.2,1.0]]
    //   prec = inv(ec)
    //   log_likelihood(ec, prec) = -4.684661154486631  (live sklearn 1.5.2)
    let ec = array![[2.0, 0.6, 0.3], [0.6, 1.5, -0.2], [0.3, -0.2, 1.0]];
    // Precision = analytic inverse of ec (computed independently, NOT from ferrolearn).
    // We instead let the oracle's expected ll stand on its own and feed prec from
    // the live-oracle inverse to drive the same arithmetic sklearn used.
    let prec = invert_3x3(&ec);
    let ll = log_likelihood(&ec, &prec).unwrap();
    assert_close(ll, -4.684661154486631, 1e-9, "ll(aniso, inv(aniso))");
}

#[test]
fn guard_log_likelihood_cross_terms() {
    // ADVERSARIAL: emp_cov and precision are independent SPD matrices (NOT inverses),
    // so neither the -tr cross-term nor logdet collapses. A sign/term error surfaces.
    //   ec   = [[2,0.6,0.3],[0.6,1.5,-0.2],[0.3,-0.2,1.0]]
    //   prec = [[1.2,0.3,0.0],[0.3,0.9,0.25],[0.0,0.25,1.4]]
    //   log_likelihood(ec, prec) = -5.3264204972232365  (live sklearn 1.5.2)
    let ec = array![[2.0, 0.6, 0.3], [0.6, 1.5, -0.2], [0.3, -0.2, 1.0]];
    let prec = array![[1.2, 0.3, 0.0], [0.3, 0.9, 0.25], [0.0, 0.25, 1.4]];
    let ll = log_likelihood(&ec, &prec).unwrap();
    assert_close(ll, -5.3264204972232365, 1e-10, "ll(ec, indep prec)");
}

/// Analytic inverse of a symmetric 3x3 matrix (used only to feed `log_likelihood`
/// the inverse-of-covariance precision the same way sklearn's `np.linalg.inv`
/// would). This is plain Cramer's rule, not a ferrolearn code path.
fn invert_3x3(m: &Array2<f64>) -> Array2<f64> {
    let a = m[[0, 0]];
    let b = m[[0, 1]];
    let c = m[[0, 2]];
    let d = m[[1, 0]];
    let e = m[[1, 1]];
    let f = m[[1, 2]];
    let g = m[[2, 0]];
    let h = m[[2, 1]];
    let i = m[[2, 2]];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv = array![
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d]
    ];
    inv.mapv(|v| v / det)
}

// ---------------------------------------------------------------------------
// REQ-OAS / REQ-LEDOIT-WOLF return-order confirmation (cov, shrinkage):
// shrinkage is in [0,1]; covariance is a 3x3. Covered above by element checks;
// the explicit tuple-order assertion is that .0 is the 3x3 cov and .1 the scalar.
// ---------------------------------------------------------------------------
#[test]
fn guard_return_order_cov_then_shrinkage() {
    let (cov, s) = oas(&probe_x(), false).unwrap();
    assert_eq!(cov.dim(), (3, 3), "oas .0 must be the covariance matrix");
    assert!(
        (0.0..=1.0).contains(&s),
        "oas .1 must be the shrinkage scalar"
    );
    let (cov, s) = ledoit_wolf(&probe_x(), false).unwrap();
    assert_eq!(
        cov.dim(),
        (3, 3),
        "ledoit_wolf .0 must be the covariance matrix"
    );
    assert!(
        (0.0..=1.0).contains(&s),
        "ledoit_wolf .1 must be the shrinkage scalar"
    );
}

// ===========================================================================
// UNCLAIMED DIVERGENCE — log_likelihood on a NON-positive-definite precision.
//
// sklearn's `log_likelihood` uses `fast_logdet(precision)` = slogdet, which
// returns -inf when det(precision) is non-positive (see
// sklearn/utils/extmath.py:93-110 fast_logdet: "if not sign > 0: return -inf").
// Therefore for a precision with a negative eigenvalue sklearn returns -inf.
//
// ferrolearn's `log_likelihood` calls `log_det_spd`, which adds adaptive `tau`
// diagonal regularisation until the matrix factorises, then returns a FINITE
// logdet of (A + tau*I) -- or errors `NumericalInstability` if it never
// factorises. Either way it does NOT return -inf, so it DIVERGES from sklearn.
//
// Input (live sklearn 1.5.2 oracle):
//   ec    = [[2,0.6,0.3],[0.6,1.5,-0.2],[0.3,-0.2,1.0]]
//   nonpd = [[1,2,0],[2,1,0],[0,0,1]]   # eigenvalues {-1, 1, 3}
//   log_likelihood(ec, nonpd) -> -inf   (sklearn)
//
// ferrolearn: returns Err(NumericalInstability) or a finite value, NEVER -inf.
//
// Materiality: arguably out-of-contract (a "precision" is defined as an inverse
// covariance, i.e. SPD), so this is pinned as an #[ignore]'d boundary, not a
// release blocker. Tracking: #1878.
// ===========================================================================
#[test]
#[ignore = "divergence: log_likelihood on non-PD precision; sklearn fast_logdet -> -inf, ferrolearn regularizes/errors; tracking #1878"]
fn divergence_log_likelihood_non_pd_precision() {
    let ec = array![[2.0, 0.6, 0.3], [0.6, 1.5, -0.2], [0.3, -0.2, 1.0]];
    let nonpd = array![[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    // Live sklearn 1.5.2: log_likelihood(ec, nonpd) == -inf.
    let sk = f64::NEG_INFINITY;
    let got = log_likelihood(&ec, &nonpd);
    // sklearn returns a finite-typed -inf; ferrolearn must reproduce it to match.
    match got {
        Ok(v) => assert_eq!(v, sk, "ferrolearn ll(non-PD)={v}, sklearn=-inf"),
        Err(e) => panic!("ferrolearn ll(non-PD) errored ({e:?}); sklearn returns -inf"),
    }
}
