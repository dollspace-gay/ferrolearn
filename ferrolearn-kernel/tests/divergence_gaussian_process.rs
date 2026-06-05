//! Green-guard + divergence tests for
//! `ferrolearn-kernel::gaussian_process::GaussianProcessRegressor` against
//! scikit-learn 1.5.2 `sklearn.gaussian_process.GaussianProcessRegressor`
//! (`sklearn/gaussian_process/_gpr.py`).
//!
//! Translation unit #1920. All oracle expected values below were produced by a
//! live `sklearn` 1.5.2 call run from `/tmp` with `optimizer=None` (so the
//! kernel is FIXED and both sides use the SAME un-tuned hyperparameters) —
//! NEVER copied from ferrolearn's own output (R-CHAR-3). The exact oracle
//! command is recorded next to each constant.
//!
//! Shared fixture (matches `.design/kernel/gaussian_process.md` ACs):
//!   X  = [[0],[1],[2],[3],[4]]
//!   y  = [0, 1, 4, 9, 16]
//!   Xs = [[0.5],[2.5]]
//!   kernel = RBF(length_scale=1.0), alpha = 1e-10
//!
//! GREEN GUARDS (must PASS — the deterministic SHIPPED slice, normalize_y=False):
//!   - `green_predict_mean`        (REQ-1) posterior mean vs oracle
//!   - `green_predict_std`         (REQ-2) predictive std  vs oracle
//!   - `green_log_marginal`        (REQ-3) LML value        vs oracle
//!   - `green_score_r2`            (REQ-4) score (R²)        vs oracle
//!   - `green_alpha_default`       (REQ-5) alpha default == 1e-10
//!
//! RED PIN (must FAIL now — single-file-fixable divergence, REQ-7):
//!   - `divergence_normalize_y_std_scaling` — ferrolearn's `normalize_y=true`
//!     omits the population-std divide in `fit` and the `·y_std` / `·y_std²`
//!     rescales in `predict`/`predict_with_std`, so both the posterior mean and
//!     the predictive std diverge from sklearn. Tracking: see filed blocker.
//!     The fixer makes this green; it is intentionally un-ignored (RED now).
//!
//! Live oracle (run from /tmp):
//! ```text
//! python3 -c "
//! import numpy as np
//! from sklearn.gaussian_process import GaussianProcessRegressor as GPR
//! from sklearn.gaussian_process.kernels import RBF
//! X=np.array([[0.],[1.],[2.],[3.],[4.]]); y=np.array([0.,1.,4.,9.,16.])
//! Xs=np.array([[0.5],[2.5]])
//! m=GPR(kernel=RBF(1.0), alpha=1e-10, optimizer=None).fit(X,y)
//! print(m.predict(Xs, return_std=True))   # mean, std
//! print(m.log_marginal_likelihood_value_) # LML at fixed theta
//! print(m.score(X,y))                      # R²
//! mn=GPR(kernel=RBF(1.0), alpha=1e-10, optimizer=None, normalize_y=True).fit(X,y)
//! print(float(mn._y_train_std))            # 5.89915248150105
//! print(mn.predict(Xs, return_std=True))   # normalize_y mean, std
//! "
//! ```

use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::{GaussianProcessRegressor, RBFKernel};
use ndarray::{Array1, Array2, array};

/// Training design `X = [[0],[1],[2],[3],[4]]`.
fn x_train() -> Array2<f64> {
    Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap()
}

/// Targets `y = [0, 1, 4, 9, 16]` (population std ≈ 5.899, non-unit so the
/// normalize_y std divergence is large).
fn y_train() -> Array1<f64> {
    array![0.0, 1.0, 4.0, 9.0, 16.0]
}

/// Query points `Xs = [[0.5],[2.5]]`.
fn x_query() -> Array2<f64> {
    Array2::from_shape_vec((2, 1), vec![0.5, 2.5]).unwrap()
}

// ---------------------------------------------------------------------------
// GREEN GUARDS — deterministic SHIPPED behaviors, normalize_y=False.
// ---------------------------------------------------------------------------

/// REQ-1. `GPR(kernel=RBF(1.0), alpha=1e-10, optimizer=None).fit(X,y).predict(Xs)`
/// = [0.09962647811260122, 5.814671556548163] (live sklearn 1.5.2 oracle).
#[test]
fn green_predict_mean() {
    // Oracle: m.predict(Xs, return_std=True)[0]
    const ORACLE_MEAN: [f64; 2] = [0.099_626_478_112_601_22, 5.814_671_556_548_163];

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let pred = fitted.predict(&x_query()).unwrap();

    for (i, &expected) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (pred[i] - expected).abs() < 1e-8,
            "predict mean[{i}]: ferrolearn={}, sklearn={expected}",
            pred[i]
        );
    }
}

/// REQ-2. Predictive std vs oracle `m.predict(Xs, return_std=True)[1]`
/// = [0.11844729178698837, 0.09004190831342587].
#[test]
fn green_predict_std() {
    const ORACLE_STD: [f64; 2] = [0.118_447_291_786_988_37, 0.090_041_908_313_425_87];

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let (_, std) = fitted.predict_with_std(&x_query()).unwrap();

    for (i, &expected) in ORACLE_STD.iter().enumerate() {
        assert!(
            (std[i] - expected).abs() < 1e-8,
            "predict std[{i}]: ferrolearn={}, sklearn={expected}",
            std[i]
        );
    }
}

/// REQ-3. LML value at the fixed theta vs oracle
/// `m.log_marginal_likelihood_value_` = -138.99763790061525.
#[test]
fn green_log_marginal() {
    const ORACLE_LML: f64 = -138.997_637_900_615_25;

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let lml = fitted.log_marginal_likelihood(&y_train());

    assert!(
        (lml - ORACLE_LML).abs() < 1e-7,
        "LML: ferrolearn={lml}, sklearn={ORACLE_LML}"
    );
}

/// REQ-4. `score` (R²) on training data vs oracle `m.score(X,y)` = 1.0
/// (perfect near-interpolation at alpha=1e-10).
#[test]
fn green_score_r2() {
    const ORACLE_SCORE: f64 = 1.0;

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let score = fitted.score(&x_train(), &y_train()).unwrap();

    assert!(
        (score - ORACLE_SCORE).abs() < 1e-9,
        "score: ferrolearn={score}, sklearn={ORACLE_SCORE}"
    );
}

/// REQ-5. `alpha` default == 1e-10 (matches `GPR().alpha == 1e-10`,
/// `_gpr.py:204`). Verified via predict parity at the default alpha: the
/// `green_predict_mean` oracle was computed at alpha=1e-10, and a GPR
/// constructed WITHOUT `.alpha(..)` must reproduce it.
#[test]
fn green_alpha_default() {
    const ORACLE_MEAN: [f64; 2] = [0.099_626_478_112_601_22, 5.814_671_556_548_163];

    // No `.alpha(..)` builder call — relies on the constructor default 1e-10.
    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)));
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let pred = fitted.predict(&x_query()).unwrap();

    for (i, &expected) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (pred[i] - expected).abs() < 1e-8,
            "default-alpha predict mean[{i}]: ferrolearn={}, sklearn={expected} \
             (a mismatch means the default alpha != 1e-10)",
            pred[i]
        );
    }
}

// ---------------------------------------------------------------------------
// RED PIN — normalize_y std scaling divergence (REQ-7, single-file-fixable).
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `GaussianProcessRegressor` with `normalize_y(true)`
/// diverges from `sklearn/gaussian_process/_gpr.py:268-273` (fit normalization)
/// and `:443`/`:484` (predict rescale).
///
/// sklearn `fit` (`_gpr.py:268-273`):
///   `_y_train_mean = np.mean(y)`; `_y_train_std = _handle_zeros_in_scale(np.std(y))`
///   (POPULATION std, ddof=0; std=0 → 1); `y = (y - mean) / std`.
/// sklearn `predict` rescales: `y_mean = _y_train_std * y_mean + _y_train_mean`
///   (`:443`) and `y_var = outer(y_var, _y_train_std**2)` (`:484`).
///
/// ferrolearn `Fit::fit` computes only `y_mean = mean(y)` and centers
/// (`y - y_mean`) — NO std divide — and `predict`/`predict_with_std` only ADD
/// `y_mean` back (`.mapv(|v| v + self.y_mean)`), never multiply by a std.
///
/// Input: X=[[0],[1],[2],[3],[4]], y=[0,1,4,9,16] (np.std=5.89915248),
/// Xs=[[0.5],[2.5]], RBF(1.0), alpha=1e-10, normalize_y=True.
///
/// Live oracle:
///   GPR(kernel=RBF(1.0), alpha=1e-10, optimizer=None, normalize_y=True)
///     .fit(X,y).predict(Xs, return_std=True)
///   = (mean=[-0.06975414701411253, 5.853970673679483],
///      std =[0.6987386352722913, 0.5311709468662362])
///   _y_train_std = 5.89915248150105
///
/// ferrolearn returns (centering by y_mean=6.0 but NOT dividing/rescaling by
/// 5.899) a mean and std off by the y_std factor.
///
/// This pin is RED now and the fixer makes it GREEN by adding the std divide
/// in `fit` and the `·y_std`/`·y_std²` rescales in the predict paths.
#[test]
fn divergence_normalize_y_std_scaling() {
    // Oracle (normalize_y=True): see module header command.
    const ORACLE_MEAN: [f64; 2] = [-0.069_754_147_014_112_53, 5.853_970_673_679_483];
    const ORACLE_STD: [f64; 2] = [0.698_738_635_272_291_3, 0.531_170_946_866_236_2];

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)))
        .normalize_y(true)
        .alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let (mean, std) = fitted.predict_with_std(&x_query()).unwrap();

    for (i, &expected) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (mean[i] - expected).abs() < 1e-8,
            "normalize_y mean[{i}]: ferrolearn={}, sklearn={expected} \
             (ferrolearn omits the /y_std divide and ·y_std rescale)",
            mean[i]
        );
    }
    for (i, &expected) in ORACLE_STD.iter().enumerate() {
        assert!(
            (std[i] - expected).abs() < 1e-8,
            "normalize_y std[{i}]: ferrolearn={}, sklearn={expected} \
             (ferrolearn omits the ·y_std std rescale)",
            std[i]
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARDS — normalize_y=True FULL surface (unit #1920 / blocker #1921).
//
// These lock in the std-scaling fix across predict-mean / predict-std / LML /
// score / constant-y. A SEPARATE dataset from the pin above is used so both the
// `/y_std` divide (fit) and the `·y_std`+`y_mean` / `·y_std²` rescales
// (predict) are exercised with a non-unit std AND non-zero mean.
//
// Live oracle (run from /tmp, sklearn 1.5.2):
// ```text
// python3 -c "
// import numpy as np
// from sklearn.gaussian_process import GaussianProcessRegressor as GPR
// from sklearn.gaussian_process.kernels import RBF
// X=np.array([[0.],[1.],[2.],[3.],[4.]]); y=np.array([10.,12.,18.,28.,42.])  # mean=22, std=11.798
// m=GPR(kernel=RBF(1.5), alpha=1e-8, optimizer=None, normalize_y=True).fit(X,y)
// Xs=np.array([[0.5],[2.5],[5.0]])
// mean,std=m.predict(Xs, return_std=True)
// print(mean.tolist()); print(std.tolist())
// print(float(m.log_marginal_likelihood())); print(float(m.score(X,y)))
// "
// -> y_mean=22.0, y_std=11.7983049630021
//    mean=[10.247842715451888, 22.24577803279786, 47.369670041125275]
//    std =[0.30576654296609335, 0.17999967586905, 4.0060999923684175]
//    lml =-4.585355472853276
//    score=0.9999999999999974
// ```
// ---------------------------------------------------------------------------

/// Non-zero-mean, non-unit-std targets so the normalize_y rescale is real.
fn x_train_ny() -> Array2<f64> {
    Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap()
}
fn y_train_ny() -> Array1<f64> {
    array![10.0, 12.0, 18.0, 28.0, 42.0]
}
fn x_query_ny() -> Array2<f64> {
    Array2::from_shape_vec((3, 1), vec![0.5, 2.5, 5.0]).unwrap()
}

/// GREEN. `predict` MEAN at normalize_y=True exercises the `·y_std + y_mean`
/// rescale (sklearn `_gpr.py:443`). Oracle: `m.predict(Xs)` for
/// `GPR(RBF(1.5), alpha=1e-8, optimizer=None, normalize_y=True)` on
/// y=[10,12,18,28,42] (mean=22, std=11.798). Tolerance 1e-7 for the
/// hand-rolled Cholesky.
#[test]
fn green_normalize_y_predict_mean() {
    const ORACLE_MEAN: [f64; 3] = [
        10.247_842_715_451_888,
        22.245_778_032_797_86,
        47.369_670_041_125_275,
    ];

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.5)))
        .normalize_y(true)
        .alpha(1e-8);
    let fitted = gp.fit(&x_train_ny(), &y_train_ny()).unwrap();
    let pred = fitted.predict(&x_query_ny()).unwrap();

    for (i, &expected) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (pred[i] - expected).abs() < 1e-7,
            "normalize_y predict mean[{i}]: ferrolearn={}, sklearn={expected}",
            pred[i]
        );
    }
}

/// GREEN. `predict_with_std` STD at normalize_y=True exercises the `·y_std²`
/// variance rescale (sklearn `_gpr.py:484`). Oracle:
/// `m.predict(Xs, return_std=True)[1]` for the same model.
#[test]
fn green_normalize_y_predict_std() {
    const ORACLE_STD: [f64; 3] = [
        0.305_766_542_966_093_35,
        0.179_999_675_869_05,
        4.006_099_992_368_417_5,
    ];

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.5)))
        .normalize_y(true)
        .alpha(1e-8);
    let fitted = gp.fit(&x_train_ny(), &y_train_ny()).unwrap();
    let (_, std) = fitted.predict_with_std(&x_query_ny()).unwrap();

    for (i, &expected) in ORACLE_STD.iter().enumerate() {
        assert!(
            (std[i] - expected).abs() < 1e-7,
            "normalize_y predict std[{i}]: ferrolearn={}, sklearn={expected}",
            std[i]
        );
    }
}

/// GREEN. log_marginal_likelihood at normalize_y=True (HIGHEST-RISK site).
/// sklearn `_gpr.py:588-609` computes the LML on the NORMALIZED `y_train_`
/// and does NOT undo normalization, so `m.log_marginal_likelihood()` is the
/// LML of the normalized GP. ferrolearn's `log_marginal_likelihood(y)` takes
/// raw y and internally centers `(y - y_mean)/y_std` (`gaussian_process.rs`),
/// which must reproduce the SAME value. Oracle:
/// `m.log_marginal_likelihood()` = -4.585355472853276.
#[test]
fn green_normalize_y_log_marginal() {
    const ORACLE_LML: f64 = -4.585_355_472_853_276;

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.5)))
        .normalize_y(true)
        .alpha(1e-8);
    let fitted = gp.fit(&x_train_ny(), &y_train_ny()).unwrap();
    let lml = fitted.log_marginal_likelihood(&y_train_ny());

    assert!(
        (lml - ORACLE_LML).abs() < 1e-7,
        "normalize_y LML: ferrolearn={lml}, sklearn={ORACLE_LML}"
    );
}

/// GREEN. `score` (R²) at normalize_y=True vs oracle `m.score(X,y)`
/// = 0.9999999999999974 (near-perfect interpolation at alpha=1e-8).
#[test]
fn green_normalize_y_score_r2() {
    const ORACLE_SCORE: f64 = 0.999_999_999_999_997_4;

    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.5)))
        .normalize_y(true)
        .alpha(1e-8);
    let fitted = gp.fit(&x_train_ny(), &y_train_ny()).unwrap();
    let score = fitted.score(&x_train_ny(), &y_train_ny()).unwrap();

    assert!(
        (score - ORACLE_SCORE).abs() < 1e-9,
        "normalize_y score: ferrolearn={score}, sklearn={ORACLE_SCORE}"
    );
}

/// GREEN. Constant-y guard at normalize_y=True. y=[3,3,3,3,3] has population
/// std 0; sklearn's `_handle_zeros_in_scale` (`_gpr.py:270`) replaces std=0
/// with 1 so there is no divide-by-zero, and ferrolearn's `std<=0 -> 1` guard
/// (`gaussian_process.rs` fit) must do the same. Oracle:
/// `GPR(RBF(1.0), alpha=1e-8, optimizer=None, normalize_y=True)
///   .fit(X,[3,3,3,3,3]).predict(Xs)` = [3.0, 3.0, 3.0]; std finite.
#[test]
fn green_normalize_y_constant_target() {
    let x = x_train_ny();
    let y = Array1::from_elem(5, 3.0);
    let gp = GaussianProcessRegressor::new(Box::new(RBFKernel::new(1.0)))
        .normalize_y(true)
        .alpha(1e-8);
    let fitted = gp.fit(&x, &y).unwrap();
    let xs = x_query_ny();
    let (mean, std) = fitted.predict_with_std(&xs).unwrap();

    for (i, m) in mean.iter().enumerate() {
        assert!(
            m.is_finite() && (m - 3.0).abs() < 1e-7,
            "constant-y normalize_y mean[{i}]: ferrolearn={m}, sklearn=3.0 \
             (std=0 must be guarded to 1, no divide-by-zero)"
        );
    }
    for (i, s) in std.iter().enumerate() {
        assert!(
            s.is_finite() && !s.is_nan(),
            "constant-y normalize_y std[{i}] must be finite (no NaN/Inf), got {s}"
        );
    }
}
