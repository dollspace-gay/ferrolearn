//! Divergence audit for the general-nu `MaternKernel` matrix and its GP
//! consumer, against live sklearn 1.5.2 (`sklearn.gaussian_process.kernels.Matern`,
//! `kernels.py:1729-1735`) and `GaussianProcessRegressor`.
//!
//! Every expected value below is a live sklearn 1.5.2 oracle (R-CHAR-3, never
//! copied from ferrolearn). Oracle commands are recorded next to each block.
//!
//! Relative/absolute tolerance 1e-6 (R-DEV-1).

use ferrolearn_core::Fit;
use ferrolearn_kernel::{GPKernel, GaussianProcessRegressor, MaternKernel, RBFKernel};
use ndarray::{Array1, Array2, array};

/// Shared 4-point design `X = [[0,0],[1,0],[0,1],[1.5,2.0]]` (mixed distances,
/// including a non-unit / non-sqrt2 pair to exercise the Bessel path off the
/// trivial grid).
fn make_x() -> Array2<f64> {
    Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.5, 2.0]).unwrap()
}

// ---------------------------------------------------------------------------
// Target #3 — full Matern matrix vs sklearn over (nu, length_scale) grid.
//
// Oracle (run from /tmp):
//   python3 -c "
//   import numpy as np
//   from sklearn.gaussian_process.kernels import Matern
//   X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.5,2.0]])
//   for nu in [0.7,3.5,5.5]:
//     for L in [0.5,1.0,2.3]:
//       print(nu, L, Matern(length_scale=L,nu=nu)(X).flatten().tolist())
//   "
// ---------------------------------------------------------------------------

/// `(nu, length_scale, flattened 4x4 Matern matrix)`.
const MATERN_ORACLE: &[(f64, f64, [f64; 16])] = &[
    (
        0.7,
        0.5,
        [
            1.0,
            0.13828069713920702,
            0.13828069713920702,
            0.004659343044852556,
            0.13828069713920702,
            1.0,
            0.05499154106561043,
            0.012699132490841826,
            0.13828069713920702,
            0.05499154106561043,
            1.0,
            0.022874378205733976,
            0.004659343044852556,
            0.012699132490841826,
            0.022874378205733976,
            1.0,
        ],
    ),
    (
        0.7,
        1.0,
        [
            1.0,
            0.406181840375756,
            0.406181840375756,
            0.07941967122077294,
            0.406181840375756,
            1.0,
            0.26180486407843745,
            0.12920677700694147,
            0.406181840375756,
            0.26180486407843745,
            1.0,
            0.17171828903750483,
            0.07941967122077294,
            0.12920677700694147,
            0.17171828903750483,
            1.0,
        ],
    ),
    (
        0.7,
        2.3,
        [
            1.0,
            0.7148576610326355,
            0.7148576610326355,
            0.37084097727436705,
            0.7148576610326355,
            1.0,
            0.6010123302913372,
            0.45227479736673865,
            0.7148576610326355,
            0.6010123302913372,
            1.0,
            0.5074706912545756,
            0.37084097727436705,
            0.45227479736673865,
            0.5074706912545756,
            1.0,
        ],
    ),
    (
        3.5,
        0.5,
        [
            1.0,
            0.13778061855662008,
            0.13778061855662008,
            0.0004289724370097413,
            0.13778061855662008,
            1.0,
            0.03308036271565073,
            0.002672462003681965,
            0.13778061855662008,
            0.03308036271565073,
            1.0,
            0.007542311542953075,
            0.0004289724370097413,
            0.002672462003681965,
            0.007542311542953075,
            1.0,
        ],
    ),
    (
        3.5,
        1.0,
        [
            1.0,
            0.5449424471128748,
            0.5449424471128748,
            0.05954656958951078,
            0.5449424471128748,
            1.0,
            0.3280670124332057,
            0.12478772930042893,
            0.5449424471128748,
            0.3280670124332057,
            1.0,
            0.18750506688966975,
            0.05954656958951078,
            0.12478772930042893,
            0.18750506688966975,
            1.0,
        ],
    ),
    (
        3.5,
        2.3,
        [
            1.0,
            0.8803125953553694,
            0.8803125953553694,
            0.4943397494173103,
            0.8803125953553694,
            1.0,
            0.7808056269023986,
            0.6076812434745492,
            0.8803125953553694,
            0.7808056269023986,
            1.0,
            0.6773949221824243,
            0.4943397494173103,
            0.6076812434745492,
            0.6773949221824243,
            1.0,
        ],
    ),
    (
        5.5,
        0.5,
        [
            1.0,
            0.13676937915110277,
            0.13676937915110277,
            0.00019936427430053568,
            0.13676937915110277,
            1.0,
            0.028755640601731965,
            0.001661041834744783,
            0.13676937915110277,
            0.028755640601731965,
            1.0,
            0.005444355498132258,
            0.00019936427430053568,
            0.001661041834744783,
            0.005444355498132258,
            1.0,
        ],
    ),
    (
        5.5,
        1.0,
        [
            1.0,
            0.5660530869645884,
            0.5660530869645884,
            0.05505723350068352,
            0.5660530869645884,
            1.0,
            0.3399358057083622,
            0.12299252787543835,
            0.5660530869645884,
            0.3399358057083622,
            1.0,
            0.18980576305684238,
            0.05505723350068352,
            0.12299252787543835,
            0.18980576305684238,
            1.0,
        ],
    ),
    (
        5.5,
        2.3,
        [
            1.0,
            0.892507575042342,
            0.892507575042342,
            0.5141601134322807,
            0.892507575042342,
            1.0,
            0.7991683256346874,
            0.6295329487542534,
            0.892507575042342,
            0.7991683256346874,
            1.0,
            0.6988669064582632,
            0.5141601134322807,
            0.6295329487542534,
            0.6988669064582632,
            1.0,
        ],
    ),
];

#[test]
fn matern_matrix_matches_sklearn_grid() {
    let x = make_x();
    let mut worst = 0.0_f64;
    let mut fails: Vec<String> = Vec::new();
    for &(nu, ls, expected) in MATERN_ORACLE {
        let km = MaternKernel::new(ls, nu).compute(&x, &x);
        for (idx, &exp) in expected.iter().enumerate() {
            let (i, j) = (idx / 4, idx % 4);
            let got = km[[i, j]];
            let err = (got - exp).abs();
            if err > worst {
                worst = err;
            }
            if err >= 1e-6 {
                fails.push(format!(
                    "nu={nu} L={ls} [{i},{j}]: ferro={got}, sklearn={exp}, abs={err:e}"
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "Matern matrix diverges from sklearn (worst abs {worst:e}):\n{}",
        fails.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Target #5 — GPR predict/std with general-nu Matern (a SECOND nu beyond the
// #2375 pin's 3.5). Oracle (run from /tmp):
//   python3 -c "
//   import numpy as np
//   from sklearn.gaussian_process import GaussianProcessRegressor as GPR
//   from sklearn.gaussian_process.kernels import Matern
//   X=np.array([[0.],[1.],[2.],[3.],[4.]]); y=np.array([0.,1.,4.,9.,16.])
//   Xs=np.array([[0.5],[2.5]])
//   for nu in [0.7,5.5]:
//     m=GPR(kernel=Matern(1.0,nu=nu),alpha=1e-10,optimizer=None).fit(X,y)
//     print(m.predict(Xs,return_std=True))
//   "
// ---------------------------------------------------------------------------

fn x_train() -> Array2<f64> {
    Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap()
}
fn y_train() -> Array1<f64> {
    array![0.0, 1.0, 4.0, 9.0, 16.0]
}
fn x_query() -> Array2<f64> {
    Array2::from_shape_vec((2, 1), vec![0.5, 2.5]).unwrap()
}

#[test]
fn gpr_matern_nu_07_predict_std_matches_sklearn() {
    const ORACLE_MEAN: [f64; 2] = [0.3905291639814565, 5.9040005268384705];
    const ORACLE_STD: [f64; 2] = [0.5976004093462246, 0.5971363941789382];

    let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 0.7))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let (mean, std) = fitted.predict_with_std(&x_query()).unwrap();
    for (i, &e) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (mean[i] - e).abs() < 1e-6,
            "nu=0.7 mean[{i}]: ferro={}, sklearn={e}",
            mean[i]
        );
    }
    for (i, &e) in ORACLE_STD.iter().enumerate() {
        assert!(
            (std[i] - e).abs() < 1e-6,
            "nu=0.7 std[{i}]: ferro={}, sklearn={e}",
            std[i]
        );
    }
}

#[test]
fn gpr_matern_nu_55_predict_std_matches_sklearn() {
    const ORACLE_MEAN: [f64; 2] = [0.2098343808248576, 5.804728994901039];
    const ORACLE_STD: [f64; 2] = [0.19941765187826288, 0.17713584767969587];

    let gp = GaussianProcessRegressor::new(Box::new(MaternKernel::new(1.0, 5.5))).alpha(1e-10);
    let fitted = gp.fit(&x_train(), &y_train()).unwrap();
    let (mean, std) = fitted.predict_with_std(&x_query()).unwrap();
    for (i, &e) in ORACLE_MEAN.iter().enumerate() {
        assert!(
            (mean[i] - e).abs() < 1e-6,
            "nu=5.5 mean[{i}]: ferro={}, sklearn={e}",
            mean[i]
        );
    }
    for (i, &e) in ORACLE_STD.iter().enumerate() {
        assert!(
            (std[i] - e).abs() < 1e-6,
            "nu=5.5 std[{i}]: ferro={}, sklearn={e}",
            std[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Target #6 — nu=inf reduces to RBF on the 4-point design.
// ---------------------------------------------------------------------------

#[test]
fn matern_nu_inf_equals_rbf() {
    let x = make_x();
    for &ls in &[0.5_f64, 1.0, 2.3] {
        let km = MaternKernel::new(ls, f64::INFINITY).compute(&x, &x);
        let rbf = RBFKernel::new(ls).compute(&x, &x);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (km[[i, j]] - rbf[[i, j]]).abs() < 1e-12,
                    "nu=inf L={ls} [{i},{j}]: matern={}, rbf={}",
                    km[[i, j]],
                    rbf[[i, j]]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Target #7 — edge / no-panic: large nu (10.5), tiny length_scale (huge d),
// large length_scale (tiny d), d=0 diagonal. The kernel must be finite (no
// NaN/Inf), diagonal exactly 1.0.
// ---------------------------------------------------------------------------

#[test]
fn matern_edge_no_panic_finite() {
    let x = make_x();
    for &(ls, nu) in &[
        (1e-6_f64, 10.5_f64), // huge d -> K_nu underflows, kernel -> 0 finite
        (1e6, 10.5),          // tiny d -> kernel -> 1
        (1.0, 10.5),          // large nu
        (1.0, 0.3),           // small nu
    ] {
        let km = MaternKernel::new(ls, nu).compute(&x, &x);
        for i in 0..4 {
            assert!(
                (km[[i, i]] - 1.0).abs() < 1e-12,
                "ls={ls} nu={nu} diagonal[{i}]={} != 1.0",
                km[[i, i]]
            );
            for j in 0..4 {
                let v = km[[i, j]];
                assert!(v.is_finite(), "ls={ls} nu={nu} [{i},{j}]={v} not finite");
            }
        }
    }
}
