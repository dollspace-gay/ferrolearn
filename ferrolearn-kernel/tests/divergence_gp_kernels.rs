//! Green-guard + divergence tests for `ferrolearn-kernel::gp_kernels` against
//! scikit-learn 1.5.2 `sklearn.gaussian_process.kernels`
//! (`sklearn/gaussian_process/kernels.py`).
//!
//! Translation unit #1911. GP kernels are DETERMINISTIC, so every kernel-matrix
//! value is directly oracle-comparable. The SHIPPED kernels (RBF, Matern
//! nu∈{0.5,1.5,2.5}, RationalQuadratic, ExpSineSquared, Constant, DotProduct,
//! White, Sum, Product, Exponentiation) are pinned here
//! as PASSING oracle GREEN GUARDS: each `compute`/`diagonal` is asserted
//! element-wise (~1e-12) against the live sklearn 1.5.2 oracle.
//!
//! R-CHAR-3: every expected literal below was produced by a live `sklearn`
//! call run from `/tmp` (sklearn 1.5.2), NEVER copied from ferrolearn's own
//! output. The exact oracle command is recorded next to each constant.
//!
//! Per-REQ blockers filed by the critic (NOT-STARTED REQs). The faithful fix for
//! each needs a missing PREREQUISITE — a special function, a trait-API redesign
//! that ripples across files, or a whole new estimator — so per R-LOOP-3 they
//! are `-l blocker` issues WITHOUT a stranded red test:
//!
//! - REQ-8 eval_gradient / analytic dK/dθ (no trait method) — blocker.
//! - REQ-9 theta/bounds/Hyperparameter/fixed/n_dims machinery — blocker.
//! - REQ-11 WhiteKernel `Y is None` vs explicit-`Y` semantics — blocker (needs a
//!   self/Y=None channel across every kernel + GPR/GPC).
//! - REQ-12 anisotropic (array) length_scale — blocker.
//! - REQ-15 ferray substrate — blocker.
//! - REQ-18 CompoundKernel — blocker.
//!
//! VERDICT: the SHIPPED formulas all match the live oracle element-wise. No
//! genuine single-file-fixable element-wise mismatch was found in the existing
//! kernels, so this file is all green guards (the correct verify-and-document
//! outcome). The remaining oracle divergences are documented in the blocker
//! issues, not pinned as stranded red tests (R-LOOP-3).

use ferrolearn_kernel::{
    ConstantKernel, DotProductKernel, ExpSineSquared, Exponentiation, GPKernel, MaternKernel,
    ProductKernel, RBFKernel, RationalQuadratic, SumKernel, WhiteKernel,
};
use ndarray::{Array2, array};

/// `X = [[0,0],[1,0],[0,1]]` — the canonical 3-point design used by every
/// oracle command in `.design/kernel/gp_kernels.md`.
fn x3() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
}

/// A distinct cross-evaluation set `X2 = [[0,0],[1,1]]`.
fn x2() -> Array2<f64> {
    array![[0.0, 0.0], [1.0, 1.0]]
}

fn assert_mat(actual: &Array2<f64>, expected: &[&[f64]]) {
    assert_eq!(actual.nrows(), expected.len(), "row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(actual.ncols(), row.len(), "col count");
        for (j, &e) in row.iter().enumerate() {
            let a = actual[[i, j]];
            assert!(
                (a - e).abs() < 1e-12,
                "K[{i},{j}] = {a} but sklearn oracle = {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-1 — RBF  (GREEN GUARD)
// oracle: RBF(length_scale=1.5)(X) and RBF(length_scale=1.5)(X, X2)
// ---------------------------------------------------------------------------

#[test]
fn green_rbf_self() {
    // sklearn 1.5.2, /tmp: RBF(length_scale=1.5)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.8007374029168081, 0.8007374029168081],
        &[0.8007374029168081, 1.0, 0.6411803884299546],
        &[0.8007374029168081, 0.6411803884299546, 1.0],
    ];
    let k = RBFKernel::new(1.5);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_rbf_cross() {
    // sklearn 1.5.2, /tmp: RBF(length_scale=1.5)(X, X2)
    let expected: &[&[f64]] = &[
        &[1.0, 0.6411803884299546],
        &[0.8007374029168081, 0.8007374029168081],
        &[0.8007374029168081, 0.8007374029168081],
    ];
    let k = RBFKernel::new(1.5);
    assert_mat(&k.compute(&x3(), &x2()), expected);
}

#[test]
fn green_rbf_diag() {
    // sklearn 1.5.2, /tmp: RBF(1.5).diag(X) == [1, 1, 1]
    let k = RBFKernel::new(1.5);
    let d = k.diagonal(&x3());
    for &v in &d {
        assert!((v - 1.0).abs() < 1e-12, "diag {v} != 1");
    }
}

// ---------------------------------------------------------------------------
// REQ-2 — Matern nu ∈ {0.5, 1.5, 2.5}  (GREEN GUARD)
// oracle: Matern(length_scale=1.0, nu=nu)(X)
// ---------------------------------------------------------------------------

#[test]
fn green_matern_05() {
    // sklearn 1.5.2, /tmp: Matern(1.0, nu=0.5)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.36787944117144233, 0.36787944117144233],
        &[0.36787944117144233, 1.0, 0.2431167344342142],
        &[0.36787944117144233, 0.2431167344342142, 1.0],
    ];
    let k = MaternKernel::new(1.0, 0.5);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_matern_15() {
    // sklearn 1.5.2, /tmp: Matern(1.0, nu=1.5)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.4833577245965077, 0.4833577245965077],
        &[0.4833577245965077, 1.0, 0.29782076792963147],
        &[0.4833577245965077, 0.29782076792963147, 1.0],
    ];
    let k = MaternKernel::new(1.0, 1.5);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_matern_25() {
    // sklearn 1.5.2, /tmp: Matern(1.0, nu=2.5)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.5239941088318203, 0.5239941088318203],
        &[0.5239941088318203, 1.0, 0.3172833639540438],
        &[0.5239941088318203, 0.3172833639540438, 1.0],
    ];
    let k = MaternKernel::new(1.0, 2.5);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_matern_diag() {
    // sklearn 1.5.2, /tmp: Matern(1.0, nu=2.5).diag(X) == [1, 1, 1]
    let k = MaternKernel::new(1.0, 2.5);
    let d = k.diagonal(&x3());
    for &v in &d {
        assert!((v - 1.0).abs() < 1e-12, "diag {v} != 1");
    }
}

// ---------------------------------------------------------------------------
// REQ-3 — ConstantKernel  (GREEN GUARD)
// oracle: ConstantKernel(constant_value=2.0)(X), .diag(X)
// ---------------------------------------------------------------------------

#[test]
fn green_constant() {
    // sklearn 1.5.2, /tmp: ConstantKernel(2.0)(X) is all 2.0
    let k = ConstantKernel::new(2.0);
    let x = x3();
    let km = k.compute(&x, &x);
    for &v in &km {
        assert!((v - 2.0).abs() < 1e-12, "{v} != 2.0");
    }
    // .diag(X) is all 2.0
    for &v in &k.diagonal(&x) {
        assert!((v - 2.0).abs() < 1e-12, "diag {v} != 2.0");
    }
}

// ---------------------------------------------------------------------------
// REQ-4 — DotProduct  (GREEN GUARD)
// oracle: DotProduct(sigma_0=0.5)(X), (X, X2), .diag(X);  k = sigma_0^2 + x·x'
// ---------------------------------------------------------------------------

#[test]
fn green_dot_self() {
    // sklearn 1.5.2, /tmp: DotProduct(sigma_0=0.5)(X)
    let expected: &[&[f64]] = &[
        &[0.25, 0.25, 0.25],
        &[0.25, 1.25, 0.25],
        &[0.25, 0.25, 1.25],
    ];
    let k = DotProductKernel::new(0.5);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_dot_cross() {
    // sklearn 1.5.2, /tmp: DotProduct(sigma_0=0.5)(X, X2)
    let expected: &[&[f64]] = &[&[0.25, 0.25], &[0.25, 1.25], &[0.25, 1.25]];
    let k = DotProductKernel::new(0.5);
    assert_mat(&k.compute(&x3(), &x2()), expected);
}

#[test]
fn green_dot_diag() {
    // sklearn 1.5.2, /tmp: DotProduct(sigma_0=0.5).diag(X) == [0.25, 1.25, 1.25]
    let k = DotProductKernel::new(0.5);
    let d = k.diagonal(&x3());
    let expected = [0.25, 1.25, 1.25];
    for (i, &e) in expected.iter().enumerate() {
        assert!((d[i] - e).abs() < 1e-12, "diag[{i}] = {} != {e}", d[i]);
    }
}

// ---------------------------------------------------------------------------
// REQ-11 (SELF channel) — WhiteKernel(X) [Y=None]  (GREEN GUARD)
// oracle: WhiteKernel(noise_level=0.1)(X) == 0.1 * I  — the GPR training-path
// semantics (K(X,X), Y=None); ferrolearn compute(X,X) MATCHES it. The DIVERGENCE
// is the explicit-Y call WhiteKernel(0.1)(X, X) == zeros, which the GPKernel
// trait cannot express (no Y=None channel) — filed as a blocker, NOT pinned red
// here (R-LOOP-3).
// ---------------------------------------------------------------------------

#[test]
fn green_white_self_is_noise_times_identity() {
    // sklearn 1.5.2, /tmp: WhiteKernel(0.1)(X) == 0.1 on the diagonal, 0 off.
    let expected: &[&[f64]] = &[&[0.1, 0.0, 0.0], &[0.0, 0.1, 0.0], &[0.0, 0.0, 0.1]];
    let k = WhiteKernel::new(0.1);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_white_diag() {
    // sklearn 1.5.2, /tmp: WhiteKernel(0.1).diag(X) == [0.1, 0.1, 0.1]
    let k = WhiteKernel::new(0.1);
    for &v in &k.diagonal(&x3()) {
        assert!((v - 0.1).abs() < 1e-12, "diag {v} != 0.1");
    }
}

// ---------------------------------------------------------------------------
// REQ-5 — Sum / Product  (GREEN GUARD)
// oracle: (ConstantKernel(1)+ConstantKernel(2))(X),
//         (ConstantKernel(2)*RBF(1))(X), (RBF(1.5)+WhiteKernel(0.1))(X) + .theta
// ---------------------------------------------------------------------------

#[test]
fn green_sum_const() {
    // sklearn 1.5.2, /tmp: (ConstantKernel(1.0)+ConstantKernel(2.0))(X) all 3.0
    let k = SumKernel::new(
        Box::new(ConstantKernel::new(1.0)),
        Box::new(ConstantKernel::new(2.0)),
    );
    let x = x3();
    for &v in &k.compute(&x, &x) {
        assert!((v - 3.0).abs() < 1e-12, "{v} != 3.0");
    }
}

#[test]
fn green_product_const_rbf() {
    // sklearn 1.5.2, /tmp: (ConstantKernel(2.0)*RBF(length_scale=1.0))(X)
    let expected: &[&[f64]] = &[
        &[2.0, 1.2130613194252668, 1.2130613194252668],
        &[1.2130613194252668, 2.0, 0.7357588823428847],
        &[1.2130613194252668, 0.7357588823428847, 2.0],
    ];
    let k = ProductKernel::new(
        Box::new(ConstantKernel::new(2.0)),
        Box::new(RBFKernel::new(1.0)),
    );
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_sum_rbf_white_matrix() {
    // sklearn 1.5.2, /tmp: (RBF(1.5)+WhiteKernel(0.1))(X)
    let expected: &[&[f64]] = &[
        &[1.1, 0.8007374029168081, 0.8007374029168081],
        &[0.8007374029168081, 1.1, 0.6411803884299546],
        &[0.8007374029168081, 0.6411803884299546, 1.1],
    ];
    let k = SumKernel::new(
        Box::new(RBFKernel::new(1.5)),
        Box::new(WhiteKernel::new(0.1)),
    );
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

// ---------------------------------------------------------------------------
// REQ-5 / REQ-6 — theta concatenation order + log transform  (GREEN GUARD)
// oracle: (RBF(1.5)+WhiteKernel(0.1)).theta == [ln 1.5, ln 0.1]
//                                  == [0.4054651081081644, -2.3025850929940455]
// ---------------------------------------------------------------------------

#[test]
fn green_sum_theta_log_order() {
    // sklearn 1.5.2, /tmp: (RBF(1.5)+WhiteKernel(0.1)).theta (k1 then k2)
    let theta_oracle = [0.405_465_108_108_164_4_f64, -2.302_585_092_994_045_5];
    let k = SumKernel::new(
        Box::new(RBFKernel::new(1.5)),
        Box::new(WhiteKernel::new(0.1)),
    );
    let params = k.get_params();
    assert_eq!(params.len(), 2);
    for (i, &e) in theta_oracle.iter().enumerate() {
        assert!(
            (params[i] - e).abs() < 1e-12,
            "theta[{i}] = {} != sklearn {e}",
            params[i]
        );
    }
}

#[test]
fn green_rbf_theta_log_transform() {
    // sklearn 1.5.2, /tmp: RBF(2.0).theta == [ln 2.0]
    let oracle = 2.0_f64.ln();
    let k = RBFKernel::new(2.0);
    let params = k.get_params();
    assert_eq!(params.len(), 1);
    assert!(
        (params[0] - oracle).abs() < 1e-12,
        "theta {} != ln 2.0 = {oracle}",
        params[0]
    );
}

// ---------------------------------------------------------------------------
// REQ-13 — RationalQuadratic  (GREEN GUARD)
// oracle: RationalQuadratic(length_scale=1.3, alpha=0.7)(X), (X, X2),
// .diag(X), and theta order [ln alpha, ln length_scale].
// ---------------------------------------------------------------------------

#[test]
fn green_rational_quadratic_self() {
    // sklearn 1.5.2, /tmp:
    // RationalQuadratic(length_scale=1.3, alpha=0.7)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.7813226961811082, 0.7813226961811082],
        &[0.7813226961811082, 1.0, 0.6512559531780081],
        &[0.7813226961811082, 0.6512559531780081, 1.0],
    ];
    let k = RationalQuadratic::new(1.3, 0.7);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_rational_quadratic_cross() {
    // sklearn 1.5.2, /tmp:
    // RationalQuadratic(length_scale=1.3, alpha=0.7)(X, X2)
    let expected: &[&[f64]] = &[
        &[1.0, 0.6512559531780081],
        &[0.7813226961811082, 0.7813226961811082],
        &[0.7813226961811082, 0.7813226961811082],
    ];
    let k = RationalQuadratic::new(1.3, 0.7);
    assert_mat(&k.compute(&x3(), &x2()), expected);
}

#[test]
fn green_rational_quadratic_diag_and_theta() {
    // sklearn 1.5.2, /tmp:
    // RationalQuadratic(length_scale=1.3, alpha=0.7).diag(X) == [1,1,1]
    // theta == [ln(alpha), ln(length_scale)]
    //       == [-0.35667494393873245, 0.26236426446749106]
    let k = RationalQuadratic::new(1.3, 0.7);
    for &v in &k.diagonal(&x3()) {
        assert!((v - 1.0).abs() < 1e-12, "diag {v} != 1");
    }

    let theta_oracle = [-0.356_674_943_938_732_45_f64, 0.262_364_264_467_491_06];
    let params = k.get_params();
    assert_eq!(params.len(), 2);
    for (i, &e) in theta_oracle.iter().enumerate() {
        assert!(
            (params[i] - e).abs() < 1e-12,
            "theta[{i}] = {} != sklearn {e}",
            params[i]
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-16 — ExpSineSquared  (GREEN GUARD)
// oracle: ExpSineSquared(length_scale=1.3, periodicity=2.0)(X), (X, X2),
// .diag(X), and theta order [ln length_scale, ln periodicity].
// ---------------------------------------------------------------------------

#[test]
fn green_exp_sine_squared_self() {
    // sklearn 1.5.2, /tmp:
    // ExpSineSquared(length_scale=1.3, periodicity=2.0)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.3062259800580424, 0.3062259800580424],
        &[0.3062259800580424, 1.0, 0.472714571288163],
        &[0.3062259800580424, 0.472714571288163, 1.0],
    ];
    let k = ExpSineSquared::new(1.3, 2.0);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_exp_sine_squared_cross() {
    // sklearn 1.5.2, /tmp:
    // ExpSineSquared(length_scale=1.3, periodicity=2.0)(X, X2)
    let expected: &[&[f64]] = &[
        &[1.0, 0.472714571288163],
        &[0.3062259800580424, 0.3062259800580424],
        &[0.3062259800580424, 0.3062259800580424],
    ];
    let k = ExpSineSquared::new(1.3, 2.0);
    assert_mat(&k.compute(&x3(), &x2()), expected);
}

#[test]
fn green_exp_sine_squared_diag_and_theta() {
    // sklearn 1.5.2, /tmp:
    // ExpSineSquared(length_scale=1.3, periodicity=2.0).diag(X) == [1,1,1]
    // theta == [ln(length_scale), ln(periodicity)]
    //       == [0.26236426446749106, 0.6931471805599453]
    let k = ExpSineSquared::new(1.3, 2.0);
    for &v in &k.diagonal(&x3()) {
        assert!((v - 1.0).abs() < 1e-12, "diag {v} != 1");
    }

    let theta_oracle = [0.262_364_264_467_491_06_f64, std::f64::consts::LN_2];
    let params = k.get_params();
    assert_eq!(params.len(), 2);
    for (i, &e) in theta_oracle.iter().enumerate() {
        assert!(
            (params[i] - e).abs() < 1e-12,
            "theta[{i}] = {} != sklearn {e}",
            params[i]
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-17 — Exponentiation  (GREEN GUARD)
// oracle: Exponentiation(RBF(length_scale=1.5), exponent=2.0)(X), (X, X2),
// .diag(X), and theta delegation to the base kernel.
// ---------------------------------------------------------------------------

#[test]
fn green_exponentiation_rbf_self() {
    // sklearn 1.5.2, /tmp:
    // Exponentiation(RBF(length_scale=1.5), exponent=2.0)(X)
    let expected: &[&[f64]] = &[
        &[1.0, 0.6411803884299546, 0.6411803884299546],
        &[0.6411803884299546, 1.0, 0.4111122905071875],
        &[0.6411803884299546, 0.4111122905071875, 1.0],
    ];
    let k = Exponentiation::new(Box::new(RBFKernel::new(1.5)), 2.0);
    let x = x3();
    assert_mat(&k.compute(&x, &x), expected);
}

#[test]
fn green_exponentiation_rbf_cross() {
    // sklearn 1.5.2, /tmp:
    // Exponentiation(RBF(length_scale=1.5), exponent=2.0)(X, X2)
    let expected: &[&[f64]] = &[
        &[1.0, 0.4111122905071875],
        &[0.6411803884299546, 0.6411803884299546],
        &[0.6411803884299546, 0.6411803884299546],
    ];
    let k = Exponentiation::new(Box::new(RBFKernel::new(1.5)), 2.0);
    assert_mat(&k.compute(&x3(), &x2()), expected);
}

#[test]
fn green_exponentiation_diag_and_theta() {
    // sklearn 1.5.2, /tmp:
    // Exponentiation(RBF(length_scale=1.5), exponent=2.0).diag(X) == [1,1,1]
    // theta delegates to the base RBF kernel: [ln(1.5)].
    let k = Exponentiation::new(Box::new(RBFKernel::new(1.5)), 2.0);
    for &v in &k.diagonal(&x3()) {
        assert!((v - 1.0).abs() < 1e-12, "diag {v} != 1");
    }

    let params = k.get_params();
    assert_eq!(params.len(), 1);
    assert!(
        (params[0] - 0.405_465_108_108_164_4).abs() < 1e-12,
        "theta[0] = {} != sklearn ln(1.5)",
        params[0]
    );
}
