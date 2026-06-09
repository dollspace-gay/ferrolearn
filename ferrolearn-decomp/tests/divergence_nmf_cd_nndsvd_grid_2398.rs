//! Config-GRID re-audit of the deterministic `init='nndsvd'`, `solver='cd'` NMF
//! path vs scikit-learn 1.5.2 `class NMF` (`sklearn/decomposition/_nmf.py:912`).
//! Tracking: #2398 (re-audit of #2394/#2395/#2396/#2397).
//!
//! Every expected number below is the LIVE sklearn 1.5.2 oracle (run from /tmp,
//! R-CHAR-3), NEVER copied from ferrolearn:
//!
//! ```python
//! import numpy as np
//! from sklearn.decomposition._nmf import _initialize_nmf
//! from sklearn.decomposition import NMF
//! def mk(seed,m,n): return (np.random.RandomState(seed).rand(m,n)*5).round(3)
//! for name,X,k in [("12x6",mk(0,12,6),3),("20x8",mk(1,20,8),4),
//!                  ("10x4",mk(2,10,4),2),("30x3",mk(3,30,3),2),
//!                  ("4x10",mk(4,4,10),3)]:
//!     W0,H0 = _initialize_nmf(X,k,init='nndsvd',random_state=0)
//!     m = NMF(n_components=k,init='nndsvd',solver='cd',max_iter=200,tol=1e-4,
//!             random_state=0); W = m.fit_transform(X)
//!     m.n_iter_, m.reconstruction_err_, m.components_[0], W[0], m.transform(X)[0]
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{NMF, NMFInit, NMFSolver};
use ndarray::{Array2, array};

fn fit_cd_nndsvd(x: &Array2<f64>, k: usize) -> ferrolearn_decomp::FittedNMF<f64> {
    NMF::<f64>::new(k)
        .with_solver(NMFSolver::CoordinateDescent)
        .with_init(NMFInit::Nndsvd)
        .with_max_iter(200)
        .with_tol(1e-4)
        .with_random_state(0)
        .fit(x, &())
        .expect("fit should succeed")
}

fn f_20x8() -> Array2<f64> {
    array![
        [2.085, 3.602, 0.001, 1.512, 0.734, 0.462, 0.931, 1.728],
        [1.984, 2.694, 2.096, 3.426, 1.022, 4.391, 0.137, 3.352],
        [2.087, 2.793, 0.702, 0.991, 4.004, 4.841, 1.567, 3.462],
        [4.382, 4.473, 0.425, 0.195, 0.849, 4.391, 0.492, 2.106],
        [4.789, 2.666, 3.459, 1.578, 3.433, 4.173, 0.091, 3.751],
        [4.944, 3.741, 1.402, 3.946, 0.516, 2.239, 4.543, 1.468],
        [1.439, 0.65, 0.097, 3.394, 1.058, 1.328, 2.458, 0.267],
        [2.871, 0.734, 2.947, 3.499, 0.512, 2.07, 3.472, 2.071],
        [0.25, 2.679, 3.319, 2.574, 4.723, 2.933, 4.517, 0.687],
        [0.696, 4.037, 1.988, 0.827, 4.638, 1.739, 3.754, 3.63],
        [4.417, 3.118, 3.755, 1.744, 1.35, 4.479, 2.14, 4.824],
        [3.317, 3.108, 0.574, 4.747, 2.25, 2.892, 2.041, 1.185],
        [4.517, 2.868, 0.014, 3.086, 1.633, 2.635, 4.43, 1.786],
        [4.543, 3.117, 0.079, 4.647, 3.454, 4.987, 0.862, 0.686],
        [4.663, 3.484, 0.33, 3.777, 3.769, 4.615, 3.558, 0.621],
        [0.099, 0.131, 0.142, 1.231, 4.3, 2.694, 2.764, 4.21],
        [0.621, 1.396, 2.929, 4.848, 2.805, 0.093, 4.003, 1.165],
        [4.036, 1.939, 4.318, 3.736, 2.781, 0.682, 0.3, 0.607],
        [0.223, 0.537, 1.129, 3.565, 2.799, 0.063, 0.36, 4.836],
        [2.841, 1.016, 1.262, 3.719, 0.977, 2.907, 4.85, 4.234],
    ]
}

fn f_10x4() -> Array2<f64> {
    array![
        [2.18, 0.13, 2.748, 2.177],
        [2.102, 1.652, 1.023, 3.096],
        [1.498, 1.334, 3.106, 2.646],
        [0.673, 2.568, 0.922, 3.927],
        [4.27, 2.471, 4.233, 0.398],
        [2.526, 0.326, 2.141, 0.483],
        [0.636, 2.984, 1.13, 0.535],
        [1.102, 1.749, 2.339, 1.009],
        [3.202, 2.415, 2.526, 1.934],
        [3.968, 2.9, 0.811, 3.504],
    ]
}

fn f_30x3() -> Array2<f64> {
    array![
        [2.754, 3.541, 1.455],
        [2.554, 4.465, 4.481],
        [0.628, 1.036, 0.257],
        [2.204, 0.149, 2.284],
        [3.246, 1.392, 3.381],
        [2.954, 0.12, 2.794],
        [1.296, 2.076, 1.418],
        [3.466, 2.202, 0.784],
        [2.723, 3.902, 1.532],
        [1.11, 1.94, 4.682],
        [4.88, 3.362, 4.514],
        [4.229, 1.89, 0.461],
        [3.267, 2.789, 1.808],
        [1.125, 2.033, 2.345],
        [1.346, 1.459, 2.288],
        [4.303, 2.931, 1.417],
        [1.39, 2.273, 1.027],
        [1.007, 2.57, 0.436],
        [2.418, 1.811, 3.538],
        [3.734, 3.455, 3.446],
        [1.868, 3.341, 1.699],
        [2.864, 1.629, 2.226],
        [0.308, 1.213, 4.858],
        [1.153, 3.457, 3.252],
        [3.62, 2.375, 2.983],
        [0.335, 0.363, 0.995],
        [0.759, 0.501, 0.646],
        [2.766, 0.939, 4.761],
        [3.408, 2.705, 3.536],
        [1.319, 4.634, 4.196],
    ]
}

fn f_4x10() -> Array2<f64> {
    array![
        [4.835, 2.736, 4.863, 3.574, 3.489, 1.08, 4.881, 0.031, 1.265, 2.174],
        [3.897, 0.988, 4.315, 4.917, 0.819, 2.987, 0.045, 1.933, 0.221, 4.783],
        [2.181, 4.745, 3.932, 4.331, 0.866, 0.375, 3.004, 0.84, 3.667, 2.042],
        [2.64, 4.688, 2.608, 0.541, 0.791, 2.726, 2.622, 3.188, 2.007, 3.249],
    ]
}

fn assert_close(name: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "{name}: sklearn {expected}, ferrolearn {actual} (diff {})",
        (actual - expected).abs()
    );
}

/// 20x8 k=4: n_iter_, reconstruction_err_, components_[0], transform W[0] all
/// match sklearn 1.5.2 oracle to 1e-6.
#[test]
fn divergence_grid_20x8_k4() {
    let x = f_20x8();
    let fitted = fit_cd_nndsvd(&x, 4);
    assert_eq!(fitted.n_iter(), 87, "n_iter_ 20x8");
    assert_close("recon 20x8", fitted.reconstruction_err(), 10.17540976664874);
    let sk_comp0 = [
        3.06265181944956,
        2.002670279990039,
        0.0,
        0.7206130448936773,
        0.0,
        2.2649691209528386,
        0.0,
        0.0,
    ];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp 20x8 [0][{j}]"), h[[0, j]], e);
    }
    let w = fitted.transform(&x).expect("transform");
    let sk_t0 = [
        0.6808891217760642,
        0.32675355025960257,
        0.28615908249844924,
        0.14373483506107285,
    ];
    for (k, &e) in sk_t0.iter().enumerate() {
        assert_close(&format!("transform 20x8 [0][{k}]"), w[[0, k]], e);
    }
}

/// 10x4 k=2.
#[test]
fn divergence_grid_10x4_k2() {
    let x = f_10x4();
    let fitted = fit_cd_nndsvd(&x, 2);
    assert_eq!(fitted.n_iter(), 68, "n_iter_ 10x4");
    assert_close("recon 10x4", fitted.reconstruction_err(), 4.051728100338134);
    let sk_comp0 = [2.15698444071336, 0.900294342023667, 2.336563544621748, 0.24532693030558828];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp 10x4 [0][{j}]"), h[[0, j]], e);
    }
    let w = fitted.transform(&x).expect("transform");
    let sk_t0 = [0.9404673445874812, 0.4678016117021916];
    for (k, &e) in sk_t0.iter().enumerate() {
        assert_close(&format!("transform 10x4 [0][{k}]"), w[[0, k]], e);
    }
}

/// 30x3 k=2 (tall).
#[test]
fn divergence_grid_30x3_k2() {
    let x = f_30x3();
    let fitted = fit_cd_nndsvd(&x, 2);
    assert_eq!(fitted.n_iter(), 141, "n_iter_ 30x3");
    assert_close("recon 30x3", fitted.reconstruction_err(), 5.646796096781988);
    let sk_comp0 = [3.406240492540472, 2.701236166217696, 0.2936470911207802];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp 30x3 [0][{j}]"), h[[0, j]], e);
    }
    let w = fitted.transform(&x).expect("transform");
    let sk_t0 = [0.968369002481738, 0.3113872234942161];
    for (k, &e) in sk_t0.iter().enumerate() {
        assert_close(&format!("transform 30x3 [0][{k}]"), w[[0, k]], e);
    }
}

/// 4x10 k=3 (wide).
#[test]
fn divergence_grid_4x10_k3() {
    let x = f_4x10();
    let fitted = fit_cd_nndsvd(&x, 3);
    assert_eq!(fitted.n_iter(), 97, "n_iter_ 4x10");
    assert_close("recon 4x10", fitted.reconstruction_err(), 3.431023356549271);
    let sk_comp0 = [
        1.4911664748676388,
        1.0421260964751025,
        1.7318358797523978,
        1.5850234326077928,
        1.0194370337828542,
        0.2182518022043106,
        1.5248637455985976,
        0.0,
        0.6746353430265573,
        0.7293634794480239,
    ];
    let h = fitted.components();
    for (j, &e) in sk_comp0.iter().enumerate() {
        assert_close(&format!("comp 4x10 [0][{j}]"), h[[0, j]], e);
    }
    let w = fitted.transform(&x).expect("transform");
    let sk_t0 = [2.860911614050451, 0.0, 0.0];
    for (k, &e) in sk_t0.iter().enumerate() {
        assert_close(&format!("transform 4x10 [0][{k}]"), w[[0, k]], e);
    }
}
