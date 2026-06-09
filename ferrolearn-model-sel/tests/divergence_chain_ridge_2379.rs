//! Adversarial value-level audit of `RegressorChain` (`ferrolearn-model-sel/
//! src/chain.rs`) against scikit-learn 1.5.2 `RegressorChain` / `_BaseChain`
//! (`sklearn/multioutput.py` fit `:700-814`, `_get_predictions` `:659-697`,
//! predict `:816-829`, RegressorChain `:1106`). Tracking #2379.
//!
//! The existing `divergence_chain.rs` guards use a PERFECT-RECOVERY
//! nearest-neighbour base where `np.allclose(predict(X), Y) == True`. That base
//! CANNOT discriminate:
//!   * the chain coefficient on the prior link (a NON-trivial value, not just a
//!     copy of Y),
//!   * fit-TRUE-Y vs fit-PREDICTIONS (both collapse to ~Y under perfect
//!     recovery),
//!   * the un-permutation actually moving the right NUMBERS (not just shape).
//!
//! This file pins the VALUES with a DETERMINISTIC `Ridge(alpha=1.0,
//! fit_intercept=True)` base (closed form: center X and y, solve
//! `(Xc^T Xc + alpha I) w = Xc^T yc`, `b = ȳ - x̄·w`; verified bit-for-bit vs
//! sklearn `Ridge`). Every expected number is a LIVE sklearn 1.5.2 oracle value
//! (R-CHAR-3) quoted next to the assertion — never copied from ferrolearn.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_core::traits::Predict;
use ferrolearn_model_sel::chain::RegressorChain;
use ndarray::{Array1, Array2};

type PipelineFactory = Box<dyn Fn() -> Pipeline<f64> + Send + Sync>;

// ---------------------------------------------------------------------------
// Deterministic Ridge(alpha=1.0, fit_intercept=True) base estimator.
//
// Mirrors sklearn `Ridge` closed form (sklearn/linear_model/_ridge.py): center
// the (augmented) features and target, solve the regularised normal equations
// with a tiny Gaussian-elimination solver, then recover the intercept. Matches
// sklearn `Ridge(alpha=1.0).{coef_,intercept_}` to machine precision (verified
// live). This is the SAME base the oracle commands below pass to
// `RegressorChain`, so any value divergence is in the CHAIN logic, not the base.
// ---------------------------------------------------------------------------
const ALPHA: f64 = 1.0;

struct RidgeBase;
struct FittedRidge {
    coef: Vec<f64>,
    intercept: f64,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        // partial pivot
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col] / d;
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    (0..n).map(|i| b[i] / a[i][i]).collect()
}

impl PipelineEstimator<f64> for RidgeBase {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        let n = x.nrows();
        let p = x.ncols();
        let nf = n as f64;
        let xmean: Vec<f64> = (0..p).map(|j| x.column(j).sum() / nf).collect();
        let ymean = y.sum() / nf;
        // Centered design.
        let mut xc = vec![vec![0.0; p]; n];
        for i in 0..n {
            for j in 0..p {
                xc[i][j] = x[[i, j]] - xmean[j];
            }
        }
        let yc: Vec<f64> = (0..n).map(|i| y[i] - ymean).collect();
        // Normal equations (Xc^T Xc + alpha I) w = Xc^T yc.
        let mut ata = vec![vec![0.0; p]; p];
        let mut atb = vec![0.0; p];
        for j in 0..p {
            for k in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += xc[i][j] * xc[i][k];
                }
                ata[j][k] = s + if j == k { ALPHA } else { 0.0 };
            }
            let mut s = 0.0;
            for i in 0..n {
                s += xc[i][j] * yc[i];
            }
            atb[j] = s;
        }
        let coef = if p == 0 { vec![] } else { solve(ata, atb) };
        let mut intercept = ymean;
        for j in 0..p {
            intercept -= xmean[j] * coef[j];
        }
        Ok(Box::new(FittedRidge { coef, intercept }))
    }
}

impl FittedPipelineEstimator<f64> for FittedRidge {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        let mut out = Array1::<f64>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut v = self.intercept;
            for (j, c) in self.coef.iter().enumerate() {
                v += c * x[[i, j]];
            }
            out[i] = v;
        }
        Ok(out)
    }
}

fn ridge_factory() -> PipelineFactory {
    Box::new(|| Pipeline::<f64>::new().estimator_step("ridge", Box::new(RidgeBase)))
}

// Shared fixture (X, Y2, Y3, Xs) matching the oracle commands.
fn x_train() -> Array2<f64> {
    ndarray::array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [2.0, 2.0],
        [3.0, 1.0],
        [1.0, 3.0]
    ]
}
fn x_star() -> Array2<f64> {
    ndarray::array![[1.5, 0.5], [0.0, 0.0], [2.0, 3.0]]
}
// y0[i], y1[i] = 2*y0 + 0.5*x1 - 3 + noise ; y2 = -0.5*y0 + 1.3*y1 + 0.2*x0 + noise
fn targets() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let x = x_train();
    let n0 = [0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.05];
    let n1 = [-0.1, 0.25, -0.3, 0.15, 0.05, -0.4, 0.2, -0.05];
    let n2 = [0.1, -0.1, 0.2, -0.2, 0.05, 0.15, -0.05, 0.1];
    let mut y0 = Array1::zeros(8);
    let mut y1 = Array1::zeros(8);
    let mut y2 = Array1::zeros(8);
    for i in 0..8 {
        let (a, b) = (x[[i, 0]], x[[i, 1]]);
        y0[i] = 1.5 * a - 0.7 * b + 2.0 + n0[i];
        y1[i] = 2.0 * y0[i] + 0.5 * b - 3.0 + n1[i];
        y2[i] = -0.5 * y0[i] + 1.3 * y1[i] + 0.2 * a + n2[i];
    }
    (y0, y1, y2)
}
fn y2_matrix() -> Array2<f64> {
    let (y0, y1, _) = targets();
    let mut y = Array2::zeros((8, 2));
    for i in 0..8 {
        y[[i, 0]] = y0[i];
        y[[i, 1]] = y1[i];
    }
    y
}
fn y3_matrix() -> Array2<f64> {
    let (y0, y1, y2) = targets();
    let mut y = Array2::zeros((8, 3));
    for i in 0..8 {
        y[[i, 0]] = y0[i];
        y[[i, 1]] = y1[i];
        y[[i, 2]] = y2[i];
    }
    y
}

fn assert_close(p: &Array2<f64>, expected: &[[f64; 2]], tol: f64, label: &str) {
    assert_eq!(p.dim(), (expected.len(), 2), "{label}: shape");
    for (i, row) in expected.iter().enumerate() {
        for j in 0..2 {
            assert!(
                (p[[i, j]] - row[j]).abs() < tol,
                "{label}: P[{i},{j}] = {} expected {} (sklearn oracle); |diff| = {}",
                p[[i, j]],
                row[j],
                (p[[i, j]] - row[j]).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CORE: RegressorChain(Ridge(alpha=1.0)) predict, default order.
//
// LIVE ORACLE (R-CHAR-3):
//   import numpy as np
//   from sklearn.multioutput import RegressorChain
//   from sklearn.linear_model import Ridge
//   X = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[0.,2.],[2.,2.],[3.,1.],[1.,3.]])
//   y0 = 1.5*X[:,0]-0.7*X[:,1]+2.0+np.array([0.3,-0.2,0.1,0.4,-0.5,0.2,-0.1,0.05])
//   y1 = 2.0*y0+0.5*X[:,1]-3.0+np.array([-0.1,0.25,-0.3,0.15,0.05,-0.4,0.2,-0.05])
//   Y = np.column_stack([y0,y1]); Xs = np.array([[1.5,0.5],[0.,0.],[2.,3.]])
//   print(RegressorChain(Ridge(alpha=1.0), cv=None).fit(X,Y).predict(Xs))
//   # -> [[3.909449244060475, 5.139146868250540],
//   #     [2.084989200863931, 1.330669546436285],
//   #     [2.906695464362850, 4.084881209503241]]
//
// The discriminator: P[*,1] is link1 = Ridge fit on [X, TRUE y0], predicted with
// link0's PREDICTION cascaded in. If ferrolearn used predictions during fit
// (instead of true y0) P[0,1] would be ~5.306, not 5.139.
// ---------------------------------------------------------------------------
#[test]
fn regressor_chain_ridge_predict_default_order() {
    const SK: [[f64; 2]; 3] = [
        [3.909449244060475, 5.139146868250540],
        [2.084989200863931, 1.330669546436285],
        [2.906695464362850, 4.084881209503241],
    ];
    let p = RegressorChain::new(ridge_factory())
        .fit(&x_train(), &y2_matrix())
        .unwrap()
        .predict(&x_star())
        .unwrap();
    assert_close(&p, &SK, 1e-6, "RegressorChain Ridge default order");
}

// ---------------------------------------------------------------------------
// ORDER + UN-PERMUTE: RegressorChain(Ridge(alpha=1.0), order=[1,0]).
//
// LIVE ORACLE (R-CHAR-3):
//   print(RegressorChain(Ridge(alpha=1.0), order=[1,0], cv=None).fit(X,Y).predict(Xs))
//   # -> [[3.909449244060475, 5.139146868250540],
//   #     [2.084989200863931, 1.330669546436285],
//   #     [2.906695464362850, 4.084881209503239]]
//
// Output columns are in ORIGINAL target order: col0 is target0 even though
// target1 was fitted first. A chain-order leak would swap the columns.
// ---------------------------------------------------------------------------
#[test]
fn regressor_chain_ridge_predict_order_reversed() {
    const SK: [[f64; 2]; 3] = [
        [3.909449244060475, 5.139146868250540],
        [2.084989200863931, 1.330669546436285],
        [2.906695464362850, 4.084881209503239],
    ];
    let p = RegressorChain::new(ridge_factory())
        .order(vec![1, 0])
        .fit(&x_train(), &y2_matrix())
        .unwrap()
        .predict(&x_star())
        .unwrap();
    assert_close(&p, &SK, 1e-6, "RegressorChain Ridge order=[1,0]");
}

// ---------------------------------------------------------------------------
// 3 TARGETS, order=[2,0,1] — chain depth 3, full un-permutation.
//
// LIVE ORACLE (R-CHAR-3):
//   y2 = -0.5*y0 + 1.3*y1 + 0.2*X[:,0] +
//        np.array([0.1,-0.1,0.2,-0.2,0.05,0.15,-0.05,0.1])
//   Y3 = np.column_stack([y0,y1,y2])
//   print(RegressorChain(Ridge(alpha=1.0), order=[2,0,1], cv=None).fit(X,Y3).predict(Xs))
//   # -> [[3.909449244060475, 5.139146868250539, 5.024924406047516],
//   #     [2.084989200863930, 1.330669546436284, 0.733920086393089],
//   #     [2.906695464362851, 4.084881209503239, 4.299546436285097]]
// ---------------------------------------------------------------------------
#[test]
fn regressor_chain_ridge_predict_three_targets_order_201() {
    const SK: [[f64; 3]; 3] = [
        [3.909449244060475, 5.139146868250539, 5.024924406047516],
        [2.084989200863930, 1.330669546436284, 0.733920086393089],
        [2.906695464362851, 4.084881209503239, 4.299546436285097],
    ];
    let p = RegressorChain::new(ridge_factory())
        .order(vec![2, 0, 1])
        .fit(&x_train(), &y3_matrix())
        .unwrap()
        .predict(&x_star())
        .unwrap();
    assert_eq!(p.dim(), (3, 3));
    for (i, row) in SK.iter().enumerate() {
        for j in 0..3 {
            assert!(
                (p[[i, j]] - row[j]).abs() < 1e-6,
                "3-target order=[2,0,1]: P[{i},{j}] = {} expected {} (sklearn oracle)",
                p[[i, j]],
                row[j]
            );
        }
    }
}
