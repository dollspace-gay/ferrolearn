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
//!
//! These three match to <1e-9, far inside any solver-tolerance band, so no
//! failing test is written for them. The divergences pinned below are all in
//! the alpha>0 objective, the always-log link, and the default power.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_linear::glm::Solver;
use ferrolearn_linear::{GammaRegressor, PoissonRegressor, TweedieRegressor};
use ndarray::{Array1, Array2};

// ===========================================================================
// #560 — per-family y-domain validation (`in_y_true_range`) + `n_iter_`.
//
// sklearn validates `y` against the loss's `interval_y_true` BEFORE fitting:
//   `if not linear_loss.base_loss.in_y_true_range(y): raise ValueError(
//        "Some value(s) of y are out of the valid range of the loss ...")`
// (`sklearn/linear_model/_glm/glm.py:221-225`). The valid range depends on the
// family's Tweedie power (verified against the live oracle; the link does NOT
// change it):
//   * Poisson  (power 1): y >= 0   (closed at 0)
//   * Gamma    (power 2): y > 0    (open at 0 — y == 0 is INVALID)
//   * Tweedie(power): power<=0 unconstrained, 0<power<2 -> y>=0, power>=2 -> y>0
//
// sklearn also exposes the solver iteration count as fitted `n_iter_`
// (`glm.py:110-114, :283`); ferrolearn exposes the IRLS iteration count via
// `FittedGLMRegressor::n_iter()`.
// ===========================================================================

/// Divergence: `GammaRegressor` must reject `y == 0` (`HalfGammaLoss` domain is
/// `0 < y`, open at 0), and accept all-positive `y`.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:221-225` +
/// `HalfGammaLoss.interval_y_true == (0, inf)` (open at 0).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import GammaRegressor; \
///   X=np.array([[1.],[2.],[3.]]); \
///   GammaRegressor().fit(X, np.array([0.,1.,2.]))"
/// # -> ValueError: Some value(s) of y are out of the valid range of the loss 'HalfGammaLoss'.
/// # GammaRegressor().fit(X, [1.,2.,3.]) -> OK
/// ```
#[test]
fn glm_gamma_rejects_zero_y() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

    // y contains 0.0 — out of the Gamma domain (0 < y). sklearn raises ValueError.
    let y_bad = Array1::from(vec![0.0, 1.0, 2.0]);
    assert!(
        GammaRegressor::<f64>::new().fit(&x, &y_bad).is_err(),
        "Gamma domain is 0 < y (open at 0): sklearn raises ValueError for y==0, \
         ferrolearn must return Err"
    );

    // All-positive y is in-domain — fits fine.
    let y_ok = Array1::from(vec![1.0, 2.0, 3.0]);
    assert!(
        GammaRegressor::<f64>::new().fit(&x, &y_ok).is_ok(),
        "Gamma with all-positive y is in-domain and must fit"
    );
}

/// Divergence: `TweedieRegressor(power=2.0)` rejects `y == 0` (Gamma domain,
/// `power >= 2` -> `y > 0`), while `power=1.5` (`1 <= power < 2` -> `y >= 0`)
/// accepts `y == 0`.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:221-225` +
/// `HalfTweedieLoss(power).interval_y_true`: `power=2.0` -> `(0, inf)` (open),
/// `power=1.5` -> `[0, inf)` (closed).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   X=np.array([[1.],[2.],[3.]]); \
///   TweedieRegressor(power=2.0).fit(X, np.array([0.,1.,2.]))"
/// # -> ValueError (out of range of loss 'HalfTweedieLoss')
/// # TweedieRegressor(power=1.5).fit(X, [0.,1.,2.]) -> OK
/// ```
#[test]
fn glm_tweedie_power2_rejects_zero_y() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0]);

    // power=2.0 (Gamma): y > 0 strictly — y==0 is out of range.
    assert!(
        TweedieRegressor::<f64>::new()
            .with_power(2.0)
            .fit(&x, &y)
            .is_err(),
        "Tweedie(power=2.0) domain is y > 0 (open at 0): sklearn raises ValueError \
         for y==0, ferrolearn must return Err"
    );

    // power=1.5 (1 <= power < 2): y >= 0 — y==0 is allowed.
    assert!(
        TweedieRegressor::<f64>::new()
            .with_power(1.5)
            .fit(&x, &y)
            .is_ok(),
        "Tweedie(power=1.5) domain is y >= 0 (closed at 0): sklearn accepts y==0, \
         ferrolearn must return Ok"
    );
}

/// Divergence: `PoissonRegressor` rejects negative `y` (`HalfPoissonLoss` domain
/// is `0 <= y`), and accepts `y == 0`.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:221-225` +
/// `HalfPoissonLoss.interval_y_true == [0, inf)` (closed at 0).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[1.],[2.],[3.]]); \
///   PoissonRegressor().fit(X, np.array([-1.,1.,2.]))"
/// # -> ValueError (out of range of loss 'HalfPoissonLoss')
/// # PoissonRegressor().fit(X, [0.,1.,2.]) -> OK
/// ```
#[test]
fn glm_poisson_rejects_negative_y() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

    // Negative y is out of the Poisson domain (0 <= y).
    let y_neg = Array1::from(vec![-1.0, 1.0, 2.0]);
    assert!(
        PoissonRegressor::<f64>::new().fit(&x, &y_neg).is_err(),
        "Poisson domain is 0 <= y: sklearn raises ValueError for y<0, \
         ferrolearn must return Err"
    );

    // y == 0 is in-domain (closed at 0) — fits fine.
    let y_zero = Array1::from(vec![0.0, 1.0, 2.0]);
    assert!(
        PoissonRegressor::<f64>::new().fit(&x, &y_zero).is_ok(),
        "Poisson domain is closed at 0: sklearn accepts y==0, ferrolearn must \
         return Ok"
    );
}

/// `n_iter_` is exposed and reasonable: `1 <= n_iter() <= max_iter`.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:110-114, :283`
///   `n_iter_ : int` — "Actual number of iterations used in the solver."
///
/// Structural pin: ferrolearn's solver is IRLS (sklearn's default is lbfgs), so
/// the exact value need not match the oracle — only that the count is exposed
/// and in `[1, max_iter]`.
#[test]
fn glm_n_iter_exposed() {
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![2.0, 5.0, 10.0, 20.0]);

    let max_iter = 100;
    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(0.0)
        .with_max_iter(max_iter)
        .fit(&x, &y)
        .expect("Poisson fit");

    let n = fitted.n_iter();
    assert!(
        n >= 1 && n <= max_iter,
        "n_iter_ must be in [1, max_iter={max_iter}] (sklearn glm.py:283 exposes \
         the solver iteration count); ferrolearn returned {n}"
    );
}

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

// ---------------------------------------------------------------------------
// Family parity at alpha>0 (#548/#549/#550) — confirms the penalized objective
// (#551) + link (#554) fixes give full coef parity for all three families, not
// just the alpha=0 MLE. GLM fitting is deterministic, so we assert full
// coef_/intercept_ against the live sklearn 1.5.2 oracle.
// ---------------------------------------------------------------------------

/// `PoissonRegressor(alpha=0.5)` on a 5x2 dataset (#548).
/// Oracle (live sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   m=PoissonRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_), repr(m.intercept_))"
///   -> coef [0.38388476754733647, 0.2024000617918683], intercept -0.519356533563308
#[test]
fn glm_poisson_alpha_half_parity() {
    const SK_COEF: [f64; 2] = [0.383_884_767_547_336_47, 0.202_400_061_791_868_3];
    const SK_INTERCEPT: f64 = -0.519_356_533_563_308;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "Poisson alpha=0.5: sklearn coef {SK_COEF:?}, ferrolearn {coef:?}"
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "Poisson alpha=0.5: sklearn intercept {SK_INTERCEPT}, ferrolearn {}",
        fitted.intercept()
    );
}

/// `GammaRegressor(alpha=0.5)`, y>0 (#549).
/// Oracle (live sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import GammaRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.,5.]); \
///   m=GammaRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_), repr(m.intercept_))"
///   -> coef [0.24777318480127428, 0.11632310150595548], intercept 0.35993159442417294
#[test]
fn glm_gamma_alpha_half_parity() {
    const SK_COEF: [f64; 2] = [0.247_773_184_801_274_28, 0.116_323_101_505_955_48];
    const SK_INTERCEPT: f64 = 0.359_931_594_424_172_94;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let fitted = GammaRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "Gamma alpha=0.5: sklearn coef {SK_COEF:?}, ferrolearn {coef:?}"
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "Gamma alpha=0.5: sklearn intercept {SK_INTERCEPT}, ferrolearn {}",
        fitted.intercept()
    );
}

/// `TweedieRegressor(power=1.5, alpha=0.5)` (log link), y>0 (#550).
/// Oracle (live sklearn 1.5.2):
///   python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.,5.]); \
///   m=TweedieRegressor(power=1.5,alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_), repr(m.intercept_))"
///   -> coef [0.25606046404981164, 0.11657692670900446], intercept 0.3563978246931595
#[test]
fn glm_tweedie_alpha_half_parity() {
    const SK_COEF: [f64; 2] = [0.256_060_464_049_811_64, 0.116_576_926_709_004_46];
    const SK_INTERCEPT: f64 = 0.356_397_824_693_159_5;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let fitted = TweedieRegressor::<f64>::new()
        .with_power(1.5)
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "Tweedie(1.5) alpha=0.5: sklearn coef {SK_COEF:?}, ferrolearn {coef:?}"
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "Tweedie(1.5) alpha=0.5: sklearn intercept {SK_INTERCEPT}, ferrolearn {}",
        fitted.intercept()
    );
}

// ---------------------------------------------------------------------------
// #558 — sample_weight support. sklearn weights the per-sample deviance by
// `sample_weight` and normalizes the objective by `sum(sample_weight)`
// (glm.py:229-242). ferrolearn's `fit_with_sample_weight` multiplies the IRLS
// `W` diagonal by `s_i` and scales the L2 penalty by `S = sum_i s_i`. A
// non-uniform weight vector shifts `coef_`/`intercept_`; full parity is testable
// (deterministic objective). Verified against the live sklearn 1.5.2 oracle.
// ---------------------------------------------------------------------------

/// `PoissonRegressor(alpha=0.5).fit(X, y, sample_weight=w)` full parity (#558).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:229-242`
///   the data term is `average(½·deviance, weights=sample_weight)`, normalized
///   by `S = sum(sample_weight)`; in the IRLS normal equations the effective
///   per-sample weight is `s_i · w_irls,i` and the penalty scales with `S`.
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs`
///   `PoissonRegressor::fit_with_sample_weight` → `fn fit_glm_irls`
///   (`weights[i] = weights[i] * sample_weight[i]`, `weight_sum = Σ s_i`).
///
/// Non-uniform weights (not all equal) so the weighting is actually exercised.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   w=np.array([1.,2.,0.5,1.5,3.]); \
///   m=PoissonRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y,sample_weight=w); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # -> [0.3573882843613702, 0.19717461697342614] -0.4371920273289445
/// ```
#[test]
fn glm_poisson_sample_weight() {
    const SK_COEF: [f64; 2] = [0.357_388_284_361_370_2, 0.197_174_616_973_426_14];
    const SK_INTERCEPT: f64 = -0.437_192_027_328_944_5;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let w = Array1::from(vec![1.0, 2.0, 0.5, 1.5, 3.0]);
    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit_with_sample_weight(&x, &y, &w)
        .expect("weighted fit");
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "Poisson sample_weight: sklearn coef {SK_COEF:?}, ferrolearn {coef:?}"
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "Poisson sample_weight: sklearn intercept {SK_INTERCEPT}, ferrolearn {}",
        fitted.intercept()
    );
}

/// `GammaRegressor(alpha=0.5).fit(X, y, sample_weight=w)` full parity, y>0 (#558).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:229-242` (same weighting as
/// Poisson above; Gamma uses `V(mu)=mu^2`, log link).
///
/// ferrolearn site: `GammaRegressor::fit_with_sample_weight` → `fn fit_glm_irls`.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import GammaRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.,5.]); \
///   w=np.array([1.,2.,0.5,1.5,3.]); \
///   m=GammaRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y,sample_weight=w); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # -> [0.23049053907808104, 0.1135045392820989] 0.4195535739122035
/// ```
#[test]
fn glm_gamma_sample_weight() {
    const SK_COEF: [f64; 2] = [0.230_490_539_078_081_04, 0.113_504_539_282_098_9];
    const SK_INTERCEPT: f64 = 0.419_553_573_912_203_5;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let w = Array1::from(vec![1.0, 2.0, 0.5, 1.5, 3.0]);
    let fitted = GammaRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit_with_sample_weight(&x, &y, &w)
        .expect("weighted fit");
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
        "Gamma sample_weight: sklearn coef {SK_COEF:?}, ferrolearn {coef:?}"
    );
    assert!(
        (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
        "Gamma sample_weight: sklearn intercept {SK_INTERCEPT}, ferrolearn {}",
        fitted.intercept()
    );
}

// ---------------------------------------------------------------------------
// #559 — D² deviance `score`. sklearn's `_GeneralizedLinearRegressor.score`
// returns D² = 1 − (deviance + constant) / (deviance_null + constant), the
// fraction of (family) deviance explained — the GLM generalization of R²
// (`glm.py:365-438`). The null model predicts the (weighted) mean of y.
// ferrolearn's `FittedGLMRegressor::score` computes the same per-family unit
// deviance (`GLMFamily::unit_deviance`, verified term-for-term against
// `sklearn._loss.loss`) and replicates the `+ constant` algebra
// (`GLMFamily::constant_to_optimal_zero`). GLM fitting is deterministic, so for
// the same fit config the fitted model — and hence D² — matches the live
// sklearn 1.5.2 oracle.
// ---------------------------------------------------------------------------

/// `PoissonRegressor(alpha=0.5).score(X, y)` == live sklearn D² (#559).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:365-438`
///   D² = 1 − (deviance + constant) / (deviance_null + constant), Poisson unit
///   deviance `2·(y·ln(y/mu) − y + mu)` (`sklearn/_loss/loss.py:728-742`).
///
/// ferrolearn site: `ferrolearn-linear/src/glm.rs`
///   `FittedGLMRegressor::score` → `GLMFamily::unit_deviance` (Poisson arm).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   m=PoissonRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.score(X,y)))"
/// # -> 0.7979479374534378
/// ```
#[test]
fn glm_poisson_d2_score() {
    const SK_D2: f64 = 0.797_947_937_453_437_8;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let fitted = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let d2 = fitted.score(&x, &y).expect("score");
    assert!(
        (d2 - SK_D2).abs() < 1e-6,
        "Poisson D²: sklearn {SK_D2}, ferrolearn {d2}"
    );
}

/// `GammaRegressor(alpha=0.5).score(X, y)` == live sklearn D², y>0 (#559).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:365-438`; Gamma unit deviance
///   `2·(−ln(y/mu) + (y − mu)/mu)` (`sklearn/_loss/loss.py:754-773`).
///
/// ferrolearn site: `FittedGLMRegressor::score` → `GLMFamily::unit_deviance`
///   (Gamma arm).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import GammaRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.,5.]); \
///   m=GammaRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.score(X,y)))"
/// # -> 0.8987486959882107
/// ```
#[test]
fn glm_gamma_d2_score() {
    const SK_D2: f64 = 0.898_748_695_988_210_7;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let fitted = GammaRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let d2 = fitted.score(&x, &y).expect("score");
    assert!(
        (d2 - SK_D2).abs() < 1e-6,
        "Gamma D²: sklearn {SK_D2}, ferrolearn {d2}"
    );
}

/// `TweedieRegressor(power=0).score(X, y)` == live sklearn D² == R² (#559).
///
/// For `power = 0` (Normal, identity link) the Tweedie unit deviance is
/// `(y − mu)²`, so D² coincides exactly with the coefficient of determination
/// R² (`sklearn/_loss/loss.py:792-797` `p → 0` limit; `glm.py:368-369`
/// "D² is a generalization of R²").
///
/// ferrolearn site: `FittedGLMRegressor::score` → `GLMFamily::unit_deviance`
///   (Tweedie `p == 0` arm = squared error).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   from sklearn.metrics import r2_score; \
///   X=np.array([[1.],[2.],[3.],[4.]]); y=np.array([2.,5.,10.,20.]); \
///   m=TweedieRegressor(power=0.0,alpha=0.0,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.score(X,y)), repr(r2_score(y, m.predict(X))))"
/// # -> 0.9319946452476573 0.9319946452476573   (D² == R²)
/// ```
#[test]
fn glm_tweedie_power0_d2_score() {
    const SK_D2: f64 = 0.931_994_645_247_657_3;
    let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from(vec![2.0, 5.0, 10.0, 20.0]);
    let fitted = TweedieRegressor::<f64>::new()
        .with_power(0.0)
        .with_alpha(0.0)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let d2 = fitted.score(&x, &y).expect("score");
    assert!(
        (d2 - SK_D2).abs() < 1e-6,
        "Tweedie(power=0) D² (== R²): sklearn {SK_D2}, ferrolearn {d2}"
    );
}

/// `TweedieRegressor(power=1.5).score(X, y)` == live sklearn D², log link (#559).
///
/// Exercises the general-`p` Tweedie unit deviance
/// `2·( y^(2−p)/((1−p)(2−p)) − y·mu^(1−p)/(1−p) + mu^(2−p)/(2−p) )`
/// (`sklearn/_loss/loss.py:789-837`), `p ∉ {0,1,2}`.
///
/// ferrolearn site: `FittedGLMRegressor::score` → `GLMFamily::unit_deviance`
///   (Tweedie general-`p` arm).
///
/// Oracle (live sklearn 1.5.2):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import TweedieRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([1.,2.,3.,4.,5.]); \
///   m=TweedieRegressor(power=1.5,alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.score(X,y)))"
/// # -> 0.9277805586816806
/// ```
#[test]
fn glm_tweedie_d2_score() {
    const SK_D2: f64 = 0.927_780_558_681_680_6;
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let fitted = TweedieRegressor::<f64>::new()
        .with_power(1.5)
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("fit");
    let d2 = fitted.score(&x, &y).expect("score");
    assert!(
        (d2 - SK_D2).abs() < 1e-6,
        "Tweedie(power=1.5) D²: sklearn {SK_D2}, ferrolearn {d2}"
    );
}

// ===========================================================================
// #556 — `solver` parameter (lbfgs / newton-cholesky) API parity (R-DEV-2)
// via IRLS (R-DEV-7).
//
// sklearn exposes `solver ∈ {'lbfgs','newton-cholesky'}` (default 'lbfgs',
// `glm.py:140-145`, `StrOptions({"lbfgs","newton-cholesky"})`). Both sklearn
// solvers minimize the SAME convex penalized GLM objective and reach the SAME
// optimum. ferrolearn fits via IRLS/Fisher-scoring — a third solver that reaches
// that same optimum — so per R-DEV-2 (match the constructor parameter
// names/defaults/constraints) + R-DEV-7 (implementation may differ while
// preserving the observable contract), the `solver` param is exposed as a
// validated part of the ABI and the observable `coef_`/`intercept_` match
// sklearn for EITHER value (the optimum is solver-invariant).
// ===========================================================================

/// `PoissonRegressor(alpha=0.5)` fitted with `Solver::Lbfgs` AND
/// `Solver::NewtonCholesky` produces the SAME `coef_`/`intercept_`, matching the
/// (solver-invariant) live sklearn oracle for both values (#556).
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:140-145` — the `solver`
/// parameter constraint `StrOptions({"lbfgs", "newton-cholesky"})`, default
/// `"lbfgs"`. Both solvers reach the same convex optimum.
///
/// Oracle (live sklearn 1.5.2 — solver-invariant to ~1e-9):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   m=PoissonRegressor(alpha=0.5,solver='lbfgs',max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # lbfgs           -> [0.3838852306065439, 0.20239975413374428] -0.5193574932086451
/// # newton-cholesky -> [0.38388523050815737, 0.2023997532777797] -0.519357493320607
/// ```
/// Both ferrolearn solver values (fit via IRLS) must match this to 1e-4.
#[test]
fn glm_solver_param_invariant() {
    // Live sklearn 1.5.2 oracle (solver-invariant; see invocation above).
    const SK_COEF: [f64; 2] = [0.383_885_230_606_543_9, 0.202_399_754_133_744_28];
    const SK_INTERCEPT: f64 = -0.519_357_493_208_645_1;

    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    for solver in [Solver::Lbfgs, Solver::NewtonCholesky] {
        let fitted = PoissonRegressor::<f64>::new()
            .with_alpha(0.5)
            .with_solver(solver)
            .with_max_iter(1000)
            .with_tol(1e-10)
            .fit(&x, &y)
            .expect("fit");
        let coef = fitted.coefficients();
        assert!(
            (coef[0] - SK_COEF[0]).abs() < 1e-4 && (coef[1] - SK_COEF[1]).abs() < 1e-4,
            "solver={solver:?}: sklearn (solver-invariant) coef {SK_COEF:?}, \
             ferrolearn (IRLS) {coef:?}"
        );
        assert!(
            (fitted.intercept() - SK_INTERCEPT).abs() < 1e-4,
            "solver={solver:?}: sklearn (solver-invariant) intercept {SK_INTERCEPT}, \
             ferrolearn (IRLS) {}",
            fitted.intercept()
        );
    }
}

/// `score` re-validates the y-domain: Gamma rejects `y == 0` in `score`, mirroring
/// sklearn's `if not base_loss.in_y_true_range(y): raise ValueError(...)` in
/// `score` (`glm.py:413-417`).
#[test]
fn glm_score_rejects_out_of_domain_y() {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y_ok = Array1::from(vec![1.0, 2.0, 3.0]);
    let fitted = GammaRegressor::<f64>::new()
        .with_alpha(0.5)
        .fit(&x, &y_ok)
        .expect("fit");
    // y with 0.0 is out of the Gamma domain (0 < y) — score must Err, like sklearn.
    let y_bad = Array1::from(vec![0.0, 2.0, 3.0]);
    assert!(
        fitted.score(&x, &y_bad).is_err(),
        "score must re-validate y-domain (glm.py:413-417): Gamma rejects y==0"
    );
}

// ===========================================================================
// #557 — `warm_start` (REQ-11) API parity (R-DEV-2) via the explicit-init
// analog (R-DEV-7).
//
// sklearn's `warm_start=True` reuses the previous fit's stateful `self.coef_` /
// `self.intercept_` as the optimizer's starting point (`glm.py:243-254`):
//   if self.warm_start and hasattr(self, "coef_"):
//       coef = np.concatenate((self.coef_, [self.intercept_]))  # fit_intercept
//   else:
//       coef = linear_loss.init_zero_coef(X)                    # cold start
//
// ferrolearn's estimators are IMMUTABLE (`fit(&self, ...)` returns a fresh
// fitted object, never mutating `self`), so there is no `self.coef_` to reuse
// across calls. The R-DEV-7 analog is an EXPLICIT initial point supplied via
// `with_coef_init(coef, intercept)`; when `warm_start` is set and an init is
// provided, the IRLS seeds from it instead of cold-starting at `coef = 0`.
//
// Because the penalized GLM objective is convex, the converged
// `coef_` / `intercept_` are warm-start-INVARIANT: the init changes only the
// starting point (and the iteration count), never the optimum. So the
// warm-started fit equals the cold fit AND the sklearn oracle. These two tests
// pin (a) the observable contract is preserved and (b) the init is genuinely
// consumed (not a no-op).
// ===========================================================================

/// REQ-11 observable contract: a WARM fit (seeded from a perturbed init) reaches
/// the SAME `coef_` / `intercept_` as the COLD fit and as the live sklearn
/// oracle — the convex optimum is init-invariant (`glm.py:244-256` reaches the
/// same minimizer regardless of the seed).
///
/// Oracle (live sklearn 1.5.2 — same as `glm_poisson_alpha_half_parity`, the
/// warm/cold fit is solver- and seed-invariant):
/// ```text
/// python3 -c "import numpy as np; from sklearn.linear_model import PoissonRegressor; \
///   X=np.array([[0.,0.],[1.,0.],[2.,1.],[3.,2.],[4.,2.]]); y=np.array([0.,1.,2.,3.,4.]); \
///   m=PoissonRegressor(alpha=0.5,max_iter=1000,tol=1e-10).fit(X,y); \
///   print(repr(m.coef_.tolist()), repr(m.intercept_))"
/// # -> [0.38388476754733647, 0.2024000617918683] -0.519356533563308
/// ```
#[test]
fn glm_warm_start_observable_contract() {
    // Live sklearn 1.5.2 oracle (PoissonRegressor(alpha=0.5), solver/seed-invariant).
    const SK_COEF: [f64; 2] = [0.383_884_767_547_336_47, 0.202_400_061_791_868_3];
    const SK_INTERCEPT: f64 = -0.519_356_533_563_308;

    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    // COLD fit (default warm_start=false).
    let cold = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("cold fit");
    let coef_cold = cold.coefficients().clone();
    let int_cold = cold.intercept();

    // Cold fit matches the oracle (anchors the contract).
    assert!(
        (coef_cold[0] - SK_COEF[0]).abs() < 1e-4
            && (coef_cold[1] - SK_COEF[1]).abs() < 1e-4
            && (int_cold - SK_INTERCEPT).abs() < 1e-4,
        "cold fit must match the live sklearn oracle: oracle coef {SK_COEF:?} \
         int {SK_INTERCEPT}; ferrolearn coef {coef_cold:?} int {int_cold}"
    );

    // WARM fit seeded from a PERTURBED init (cold coef ± 0.1, intercept + 0.1) —
    // a reasonable but non-optimal starting point. The convex optimum is
    // init-invariant, so the converged attributes must equal the cold fit.
    let perturbed_coef = Array1::from(vec![coef_cold[0] + 0.1, coef_cold[1] - 0.1]);
    let warm = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .with_warm_start(true)
        .with_coef_init(perturbed_coef, int_cold + 0.1)
        .fit(&x, &y)
        .expect("warm fit");
    let coef_warm = warm.coefficients();
    let int_warm = warm.intercept();

    // warm == cold (convex optimum is init-invariant).
    assert!(
        (coef_warm[0] - coef_cold[0]).abs() < 1e-6
            && (coef_warm[1] - coef_cold[1]).abs() < 1e-6
            && (int_warm - int_cold).abs() < 1e-6,
        "warm_start observable contract: the convex optimum is init-invariant, so \
         the warm fit must equal the cold fit; cold coef {coef_cold:?} int {int_cold}, \
         warm coef {coef_warm:?} int {int_warm}"
    );

    // warm == oracle (so warm_start preserves the sklearn-observable contract).
    assert!(
        (coef_warm[0] - SK_COEF[0]).abs() < 1e-4
            && (coef_warm[1] - SK_COEF[1]).abs() < 1e-4
            && (int_warm - SK_INTERCEPT).abs() < 1e-4,
        "warm_start observable contract: the warm fit must match the live sklearn \
         oracle coef {SK_COEF:?} int {SK_INTERCEPT}; ferrolearn warm coef {coef_warm:?} \
         int {int_warm}"
    );
}

/// REQ-11 init genuinely used (not a no-op): seeding `with_coef_init` with the
/// EXACT converged solution and a tiny `max_iter` already lands at/near the
/// optimum, whereas a COLD fit with the same tiny `max_iter` has NOT converged.
/// This is a behavioral assertion that the init changes the trajectory.
///
/// sklearn site: `sklearn/linear_model/_glm/glm.py:243-254` — the warm-start
/// `coef` is handed straight to the optimizer as its starting point. With the
/// optimum as the seed and one step allowed, the optimizer is already converged.
#[test]
fn glm_warm_start_init_used() {
    let x = Array2::from_shape_vec((5, 2), vec![0., 0., 1., 0., 2., 1., 3., 2., 4., 2.]).unwrap();
    let y = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    // Reference: the converged solution (large max_iter, tight tol).
    let solved = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1000)
        .with_tol(1e-10)
        .fit(&x, &y)
        .expect("converged fit");
    let coef_star = solved.coefficients().clone();
    let int_star = solved.intercept();

    // COLD fit with a TINY max_iter (1 IRLS step) — starts at coef=0, so it is
    // NOT yet at the optimum.
    let cold_tiny = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1)
        .with_tol(1e-12)
        .fit(&x, &y)
        .expect("cold tiny fit");
    let cold_err = (cold_tiny.coefficients()[0] - coef_star[0]).abs()
        + (cold_tiny.coefficients()[1] - coef_star[1]).abs()
        + (cold_tiny.intercept() - int_star).abs();

    // WARM fit with the SAME tiny max_iter but seeded at the EXACT optimum — it
    // should already be at/near the solution (the init is genuinely consumed).
    let warm_tiny = PoissonRegressor::<f64>::new()
        .with_alpha(0.5)
        .with_max_iter(1)
        .with_tol(1e-12)
        .with_warm_start(true)
        .with_coef_init(coef_star.clone(), int_star)
        .fit(&x, &y)
        .expect("warm tiny fit");
    let warm_err = (warm_tiny.coefficients()[0] - coef_star[0]).abs()
        + (warm_tiny.coefficients()[1] - coef_star[1]).abs()
        + (warm_tiny.intercept() - int_star).abs();

    // The init is genuinely used: seeded at the optimum with 1 step, the warm fit
    // is far closer to the solution than the cold (coef=0) fit with 1 step. If the
    // init were a no-op (ignored), warm_err would equal cold_err.
    assert!(
        warm_err < 1e-6,
        "warm_start init used: seeded at the exact optimum with max_iter=1, the \
         warm fit must already be at the solution (||warm - star|| = {warm_err})"
    );
    assert!(
        cold_err > 1e-2,
        "control: a cold (coef=0) fit with max_iter=1 must NOT be converged \
         (||cold - star|| = {cold_err}); else the test cannot distinguish init-used"
    );
    assert!(
        warm_err < cold_err,
        "warm_start changes the trajectory: warm (seeded at optimum) must be closer \
         to the solution than cold (seeded at 0); warm_err {warm_err}, cold_err {cold_err}"
    );
}
