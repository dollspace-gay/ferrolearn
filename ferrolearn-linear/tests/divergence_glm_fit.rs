//! Divergence pins for the GLM family (`PoissonRegressor`, `GammaRegressor`,
//! `TweedieRegressor`) against the live scikit-learn 1.5.2 oracle
//! (`sklearn/linear_model/_glm/glm.py`, commit 156ef14).
//!
//! GLM fitting is DETERMINISTIC (no RNG): sklearn (lbfgs) and ferrolearn (IRLS)
//! are both descent methods on the SAME convex penalized objective, so for a
//! correctly-implemented objective they reach the SAME minimizer (`glm.md`,
//! "Solver / optimum equivalence"). Therefore full `coef_`/`intercept_` parity
//! IS testable, and every assertion below compares ferrolearn's fitted
//! attributes against the live sklearn oracle.
//!
//! Every expected value is produced by RUNNING scikit-learn 1.5.2 (the live
//! oracle), never copied from ferrolearn (goal.md R-CHAR-3). The exact python
//! invocation is recorded above each oracle constant.
//!
//! # Paths already CORRECT (no test written — a passing path is not a divergence)
//!
//! For `alpha=0` (the penalty is moot, so only the data term / link matter)
//! ferrolearn's log-link IRLS already reaches sklearn's MLE on the design-doc
//! dataset `X=[[1],[2],[3],[4]]`, `y=[2,5,10,20]`:
//!   * `PoissonRegressor(alpha=0)`  — ferrolearn coef 0.729310511473,
//!     int 0.091029816 vs oracle coef 0.729310511475, int 0.091029816
//!     (#548 alpha=0 path CORRECT; the work is the alpha>0 objective #551).
//!   * `GammaRegressor(alpha=0)`    — ferrolearn coef 0.759308159803,
//!     int 0.003835132 vs oracle coef 0.759308159802, int 0.003835132
//!     (#550 alpha=0 path CORRECT).
//!   * `TweedieRegressor(power=1.5, alpha=0)` (log link) — ferrolearn coef
//!     0.742980981726, int 0.048291902 vs oracle coef 0.742980981243,
//!     int 0.048291904 (#549 log-link alpha=0 path CORRECT).
//! These three match to <1e-9, far inside any solver-tolerance band, so no
//! failing test is written for them. The divergences pinned below are all in
//! the alpha>0 objective, the always-log link, and the default power.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::{PoissonRegressor, TweedieRegressor};
use ndarray::{Array1, Array2};

// ===========================================================================
// #551 (crux) — intercept penalization. sklearn excludes the intercept from
// the L2 penalty (`l2_reg_strength = self.alpha`, glm.py:258; the intercept is
// the last coef entry and `LinearModelLoss` does not penalize it). ferrolearn's
// `weighted_ridge_solve` adds `alpha` to EVERY diagonal entry including the
// intercept column (`glm.rs`: `for i in 0..n_features { xtwx[[i,i]] += alpha }`),
// so a large alpha shrinks the intercept toward 0 instead of leaving it free.
// ===========================================================================

/// Divergence: `PoissonRegressor` penalizes the intercept; sklearn does not.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:258`
///   `l2_reg_strength = self.alpha`
/// applied only to the coefficient block — the intercept (last entry of the
/// `LinearModelLoss` coef array) is excluded from `||w||_2^2`. Consequently as
/// `alpha -> inf` the coefficients are driven to 0 but the UNPENALIZED intercept
/// converges to `link(mean y) = log(mean y)`.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs` `fn weighted_ridge_solve`
///   `for i in 0..n_features { xtwx[[i, i]] = xtwx[[i, i]] + alpha; }`
/// — the loop spans ALL design columns including column 0 (the intercept), so
/// the intercept IS penalized and is shrunk toward 0.
///
/// Dataset: `X=[[1],[2],[3],[4]]`, `y=[2,5,10,20]`, `alpha=1e6` (penalty so
/// large the coefficient is annihilated; only the intercept handling shows).
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,10.,20.]); \
///   m=PoissonRegressor(alpha=1e6,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_), repr(np.log(y.mean())))"
/// # -> [7.374230290567785e-06] 2.224623550344896  log(mean y)=2.224623551524334
/// ```
/// ferrolearn instead returns intercept ~3.3e-5 (shrunk toward 0 with the coef).
///
/// Tracking: #551
#[test]
fn glm_poisson_intercept_unpenalized() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![2.0, 5.0, 10.0, 20.0]);

    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(1e6)
        .with_max_iter(1000)
        .with_tol(1e-12)
        .fit(&x, &y)
        .expect("Poisson fit");

    // sklearn: intercept_ -> log(mean y) (UNpenalized), coef_ -> 0.
    const SK_INTERCEPT: f64 = 2.224_623_550_344_896; // = log(mean([2,5,10,20]))
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-3,
        "intercept penalization: sklearn leaves the intercept unpenalized so \
         alpha=1e6 gives intercept_ = log(mean y) = {SK_INTERCEPT}; ferrolearn \
         penalizes the intercept column and returns {} (shrunk toward 0)",
        fitted.intercept()
    );
}

// ===========================================================================
// #551 (crux) — mean-vs-sum deviance penalty scaling. sklearn minimizes the
// per-sample-MEAN half-deviance + 1/2*alpha*||coef||^2 (glm.py:229-258);
// ferrolearn's normal equations (X^T W X + alpha I) minimize the SUMMED weighted
// deviance + 1/2*alpha*||w||^2, making its effective penalty too weak by a
// factor ~n_samples. With a moderate alpha and a multi-feature dataset the two
// optima differ in BOTH coef and intercept (and the unpenalized-intercept bug
// compounds it).
// ===========================================================================

/// Divergence: `PoissonRegressor(alpha>0)` minimizes summed (not mean) deviance.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:229-242`
///   the data term is `average(loss, weights=sample_weight)` (a per-sample MEAN
///   of `1/2 * deviance`); the penalty `1/2 * alpha * L2` is weighed against the
///   mean, i.e. effective per-sample penalty `alpha / n_samples`.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs` `fn weighted_ridge_solve`
///   solves `(X^T W X + alpha I) w = X^T W z` — `X^T W X` accumulates the SUMMED
///   weighted deviance, so `alpha` is added to a sum-scale matrix; the penalty is
///   ~n_samples too weak relative to sklearn's mean-scale objective.
///
/// Dataset (n=5, two features so the scaling shifts the fitted coef vector):
/// `X=[[0,0],[1,0],[2,1],[3,2],[4,2]]`, `y=[0,1,2,3,4]`, `alpha=1.0`.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   m=PoissonRegressor(alpha=1.0,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # -> [0.34151720183203493, 0.18859744954079938] -0.3768013153839991
/// ```
/// ferrolearn returns coef ~[0.35139, 0.20082], int ~-0.38552 (penalty too weak).
///
/// Tracking: #551
#[test]
fn glm_poisson_penalty_scaling() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 2.0],
    )
    .unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(1.0)
        .with_max_iter(1000)
        .with_tol(1e-12)
        .fit(&x, &y)
        .expect("Poisson fit");

    // Oracle: mean-half-deviance objective.
    const SK_COEF: [f64; 2] = [0.341_517_201_832_034_93, 0.188_597_449_540_799_38];
    const SK_INTERCEPT: f64 = -0.376_801_315_383_999_1;

    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "penalty scaling: sklearn (mean half-deviance) fits coef_ = {SK_COEF:?}; \
         ferrolearn (summed deviance => penalty ~n too weak) returns [{}, {}]",
        coef[0],
        coef[1]
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "penalty scaling: sklearn intercept_ = {SK_INTERCEPT}; ferrolearn = {}",
        fitted.intercept()
    );
}

// ===========================================================================
// #554 — TweedieRegressor link='auto' selects IDENTITY for power<=0. sklearn's
// `TweedieRegressor._get_loss` returns `HalfTweedieLossIdentity` for power<=0
// (Normal/OLS, identity link); ferrolearn uses a log link ALWAYS, so for
// power=0 it fits the wrong model entirely.
// ===========================================================================

/// Divergence: `TweedieRegressor(power=0)` uses log link; sklearn uses identity.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:889-893`
///   `if self.link == "auto":`
///   `    if self.power <= 0:`
///   `        # identity link`
///   `        return HalfTweedieLossIdentity(power=self.power)`
/// — `link='auto'` (the default) with `power=0` is Normal + identity link, i.e.
/// ordinary (ridge) least squares; `coef_`/`intercept_` are the OLS solution.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs` (`fn fit_glm_irls` initializes
///   `eta = log(y_safe)` and `predict` applies `exp(...)`) — log link
///   unconditionally; `TweedieRegressor` has no `link` field, so power=0 cannot
///   select the identity link.
///
/// Dataset: `X=[[1],[2],[3],[4]]`, `y=[2,5,10,20]`, `power=0`, `alpha=0`.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,10.,20.]); \
///   m=TweedieRegressor(power=0.0,alpha=0.0,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # -> [5.9000000221289515] -5.500000066704759   (identity link = OLS line)
/// ```
/// ferrolearn (log link) returns coef ~[0.70956], int ~0.16030 — a different
/// model.
///
/// Tracking: #554
#[test]
fn glm_tweedie_power0_identity_link() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![2.0, 5.0, 10.0, 20.0]);

    let fitted = TweedieRegressor::<f64>::new()
        .with_power(0.0)
        .with_alpha(0.0)
        .with_max_iter(1000)
        .with_tol(1e-12)
        .fit(&x, &y)
        .expect("Tweedie fit");

    // sklearn power=0 link='auto' => identity link => OLS solution.
    const SK_COEF: f64 = 5.900_000_022_128_951;
    const SK_INTERCEPT: f64 = -5.500_000_066_704_759;

    assert!(
        (fitted.coefficients()[0] - SK_COEF).abs() < 1e-3,
        "Tweedie power=0 link='auto': sklearn fits the IDENTITY-link (OLS) model \
         coef_ = {SK_COEF}; ferrolearn uses a log link always and returns {}",
        fitted.coefficients()[0]
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-3,
        "Tweedie power=0 link='auto': sklearn intercept_ = {SK_INTERCEPT}; \
         ferrolearn (log link) returns {}",
        fitted.intercept()
    );
}

// ===========================================================================
// #553 — predict applies link.inverse, not an unconditional exp. Tied to #554:
// for the identity-link power=0 model, sklearn's predict returns the RAW linear
// predictor (X@coef + intercept), not exp(...). Even compared on identical
// (oracle) fitted coefficients, ferrolearn's predict would apply exp; here we
// pin the end-to-end predict against the oracle for the power=0 case.
// ===========================================================================

/// Divergence: `predict` applies `exp` unconditionally; sklearn applies the
/// link inverse (identity for power=0).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:362`
///   `y_pred = self._linear_predictor(X)` is passed through
///   `self._base_loss.link.inverse(...)` — identity for the power=0 identity
///   link, so `predict(X)` is the raw linear predictor.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs` `fn predict`
///   `let eta = x.dot(&self.coefficients) + self.intercept; Ok(eta.mapv(|v| v.exp()))`
///   — applies `exp` for every model, including identity-link Tweedie.
///
/// Dataset/oracle (sklearn 1.5.2, identity-link power=0):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,10.,20.]); \
///   m=TweedieRegressor(power=0.0,alpha=0.0,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.predict(X).tolist()))"
/// # -> [0.39999..., 6.29999..., 12.19999..., 18.10000...]  (raw linear predictor)
/// ```
/// ferrolearn returns exp(log-link-fit) ~[2.39, 4.85, 9.86, 20.06] — both the
/// model AND the link inverse are wrong.
///
/// Tracking: #553
#[test]
fn glm_tweedie_power0_predict_identity_inverse() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![2.0, 5.0, 10.0, 20.0]);

    let fitted = TweedieRegressor::<f64>::new()
        .with_power(0.0)
        .with_alpha(0.0)
        .with_max_iter(1000)
        .with_tol(1e-12)
        .fit(&x, &y)
        .expect("Tweedie fit");

    let preds = fitted.predict(&x).expect("predict");

    // sklearn identity link => predict is the raw linear predictor.
    const SK_PRED: [f64; 4] = [
        0.399_999_955_424_192_4,
        6.299_999_977_553_144,
        12.199_999_999_682_097,
        18.100_000_021_811_05,
    ];
    for (i, (&p, &sk)) in preds.iter().zip(SK_PRED.iter()).enumerate() {
        assert!(
            (p - sk).abs() < 1e-2,
            "predict link.inverse: sklearn (identity link) predicts SK_PRED[{i}] \
             = {sk}; ferrolearn applies exp(...) and returns {p}"
        );
    }
}

// ===========================================================================
// #555 — TweedieRegressor default power. sklearn's default is power=0.0
// (Normal); ferrolearn defaults to 1.5. Structural pin (no fit required).
// ===========================================================================

/// Divergence: `TweedieRegressor::new().power` is 1.5; sklearn default is 0.0.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:867`
///   `def __init__(self, *, power=0.0, alpha=1.0, ...)`
/// — the documented default (`glm.py:737` `power : float, default=0`) is Normal.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs` `fn TweedieRegressor::new`
///   `power: 1.5,`
///
/// Tracking: #555
#[test]
fn glm_tweedie_default_power() {
    // sklearn TweedieRegressor.__init__ default: power=0.0 (glm.py:867).
    const SK_DEFAULT_POWER: f64 = 0.0;
    let m = TweedieRegressor::<f64>::new();
    assert_eq!(
        m.power, SK_DEFAULT_POWER,
        "default power: sklearn TweedieRegressor default power=0.0 (Normal, \
         glm.py:867); ferrolearn defaults to {}",
        m.power
    );
}
