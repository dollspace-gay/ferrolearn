//! Divergence edge-case audit for `ferrolearn-preprocess/src/feature_scoring.rs`
//! (`f_classif`, `f_regression`, `chi2`) against scikit-learn 1.5.2
//! `sklearn/feature_selection/_univariate_selection.py`.
//!
//! These pin the *degenerate-input* divergences NOT covered by the value/p-value
//! parity suite in `divergence_feature_scoring.rs`: the constant-feature (0/0),
//! perfect-correlation (`force_finite` clamp), and all-zero-column edges.
//!
//! All expected values are from the LIVE sklearn 1.5.2 / scipy oracle (R-CHAR-3),
//! never copied from the ferrolearn side. Generators per test below.

use ferrolearn_preprocess::feature_scoring::{chi2, f_classif, f_regression};
use ndarray::{Array1, array};

// ===========================================================================
// DIV: f_classif constant feature -> sklearn (nan, nan) vs ferrolearn (inf, 0)
//
// sklearn `f_oneway` (_univariate_selection.py:113) computes `f = msb / msw`.
// For a constant feature both the between-MS (msb) and within-MS (msw) are 0,
// so `f = 0.0 / 0.0 = nan` and `special.fdtrc(dfbn, dfwn, nan) = nan`.
// ferrolearn `feature_scoring.rs:156-157` special-cases `ms_within == 0` to
// `F::infinity()` REGARDLESS of `ms_between`, giving f=inf and p=sf(inf)=0.
//
// LIVE oracle (sklearn 1.5.2):
//   python3 -c "import numpy as np; from sklearn.feature_selection import f_classif
//    X=np.array([[1.,7.],[2.,7.],[10.,7.],[11.,7.]]); y=np.array([0,0,1,1])
//    f,p=f_classif(X,y); print(f.tolist(), p.tolist())"
//   -> f=[162.0, nan]  p=[0.006116265326381102, nan]
// ferrolearn -> f=[162.0, inf]  p=[0.0061162653263810976, 0.0]
// Tracking: #2312
// ===========================================================================

/// Divergence: ferrolearn `f_classif` returns `f=inf, p=0` for a CONSTANT
/// feature where `sklearn/feature_selection/_univariate_selection.py:113`
/// (`f = msb / msw`, `0/0 = nan`) returns `f=nan, p=nan`.
/// Tracking: #2312
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_f_classif_constant_feature_nan() {
    // feature 0 separates classes (sklearn f=162); feature 1 is constant (=7).
    let x = array![[1.0_f64, 7.0], [2.0, 7.0], [10.0, 7.0], [11.0, 7.0]];
    let y: Array1<usize> = array![0, 0, 1, 1];
    match f_classif(&x, &y) {
        Ok((f, p)) => {
            // sklearn: constant feature -> f=nan, p=nan.
            assert!(
                f[1].is_nan(),
                "f_classif constant-feature F: ferrolearn {} vs sklearn nan",
                f[1]
            );
            assert!(
                p[1].is_nan(),
                "f_classif constant-feature p: ferrolearn {} vs sklearn nan",
                p[1]
            );
        }
        Err(e) => assert!(false, "f_classif returned Err: {e}"),
    }
}

// ===========================================================================
// DIV: f_regression perfect correlation -> sklearn finfo(f64).max vs ferro inf
//
// sklearn `f_regression` default `force_finite=True`
// (_univariate_selection.py:445-449) replaces an infinite F (r^2 == 1) with
// `np.finfo(F.dtype).max == 1.7976931348623157e+308`. ferrolearn
// `feature_scoring.rs:263-264` returns `F::infinity()` and never applies the
// finite clamp -> the SHIPPED default (force_finite=True) is not honored.
//
// LIVE oracle (sklearn 1.5.2):
//   python3 -c "import numpy as np; from sklearn.feature_selection import f_regression
//    X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([1.,2.,3.,4.])
//    f,p=f_regression(X,y); print(repr(f.tolist()), p.tolist())"
//   -> f=[1.7976931348623157e+308]  p=[0.0]
// ferrolearn -> f=[inf]  p=[0.0]
// Tracking: #2313
// ===========================================================================

/// Divergence: ferrolearn `f_regression` returns `f=inf` for a perfectly
/// correlated feature where sklearn (default `force_finite=True`,
/// `sklearn/feature_selection/_univariate_selection.py:445-449`) clamps to
/// `np.finfo(float64).max == 1.7976931348623157e+308`.
/// Tracking: #2313
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_f_regression_perfect_corr_force_finite() {
    // np.finfo(np.float64).max
    const SK_FINFO_MAX: f64 = 1.797_693_134_862_315_7e308;
    let x = array![[1.0_f64], [2.0], [3.0], [4.0]];
    let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    match f_regression(&x, &y) {
        Ok((f, _p)) => {
            assert!(
                f[0].is_finite() && (f[0] - SK_FINFO_MAX).abs() <= 0.0,
                "f_regression perfect-corr F: ferrolearn {} vs sklearn {SK_FINFO_MAX} (finite finfo.max, default force_finite=True)",
                f[0]
            );
        }
        Err(e) => assert!(false, "f_regression returned Err: {e}"),
    }
}

// ===========================================================================
// DIV: chi2 all-zero feature column -> sklearn (nan, nan) vs ferro (0.0, 1.0)
//
// sklearn `chi2` computes `expected = class_prob.T @ feature_count`
// (_univariate_selection.py:286). For an all-zero column feature_count==0 so
// every expected entry is 0; `_chisquare` (:189-191) divides `(obs-exp)^2`
// by 0 under `errstate(invalid="ignore")` giving `0/0 = nan`, summed to nan;
// `chdtrc(k-1, nan) = nan`. ferrolearn `feature_scoring.rs:368-372`
// short-circuits an all-zero column to `stat=0, p=1`.
//
// LIVE oracle (sklearn 1.5.2):
//   python3 -c "import numpy as np; from sklearn.feature_selection import chi2
//    X=np.array([[1.,0.],[2.,0.],[0.,0.],[3.,0.]]); y=np.array([0,1,0,1])
//    c,p=chi2(X,y); print(c.tolist(), p.tolist())"
//   -> stat=[2.6666666666666665, nan]  p=[0.10247043485974942, nan]
// ferrolearn -> stat=[2.666..., 0.0]  p=[0.102..., 1.0]
// Tracking: #2314
// ===========================================================================

/// Divergence: ferrolearn `chi2` returns `stat=0, p=1` for an all-zero feature
/// column where sklearn (`expected==0` -> `_chisquare` `0/0`,
/// `sklearn/feature_selection/_univariate_selection.py:189-192`) returns
/// `stat=nan, p=nan`.
/// Tracking: #2314
#[test]
#[allow(
    clippy::assertions_on_constants,
    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
)]
fn divergence_chi2_all_zero_column_nan() {
    // feature 0 has signal; feature 1 is the all-zero column.
    let x = array![[1.0_f64, 0.0], [2.0, 0.0], [0.0, 0.0], [3.0, 0.0]];
    let y: Array1<usize> = array![0, 1, 0, 1];
    match chi2(&x, &y) {
        Ok((stat, p)) => {
            assert!(
                stat[1].is_nan(),
                "chi2 all-zero-column stat: ferrolearn {} vs sklearn nan",
                stat[1]
            );
            assert!(
                p[1].is_nan(),
                "chi2 all-zero-column p: ferrolearn {} vs sklearn nan",
                p[1]
            );
        }
        Err(e) => assert!(false, "chi2 returned Err: {e}"),
    }
}
