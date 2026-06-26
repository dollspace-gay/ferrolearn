//! Green-guard + divergence tests for `ferrolearn-kernel::gp_kernels` against
//! scikit-learn 1.5.2 `sklearn.gaussian_process.kernels`
//! (`sklearn/gaussian_process/kernels.py`).
//!
//! Translation unit #1911. GP kernels are DETERMINISTIC, so every kernel-matrix
//! value is directly oracle-comparable. The SHIPPED kernels (RBF, Matern
//! nu∈{0.5,1.5,2.5}, RationalQuadratic, ExpSineSquared, Constant, DotProduct,
//! White, Sum, Product, Exponentiation, CompoundKernel, and Hyperparameter
//! metadata) are pinned here as PASSING oracle GREEN GUARDS: each
//! `compute`/`diagonal` or metadata surface is asserted against the live sklearn
//! 1.5.2 oracle.
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
//! - REQ-9 full theta/bounds/fixed optimizer machinery — blocker. REQ-19 ships
//!   default Hyperparameter metadata, log-space bounds, and n_dims guards.
//! - REQ-11 WhiteKernel `Y is None` vs explicit-`Y` semantics — blocker (needs a
//!   self/Y=None channel across every kernel + GPR/GPC).
//! - REQ-12 anisotropic (array) length_scale — blocker.
//! - REQ-15 ferray substrate — blocker.
//!
//! VERDICT: the SHIPPED formulas all match the live oracle element-wise. No
//! genuine single-file-fixable element-wise mismatch was found in the existing
//! kernels, so this file is all green guards (the correct verify-and-document
//! outcome). The remaining oracle divergences are documented in the blocker
//! issues, not pinned as stranded red tests (R-LOOP-3).

use ferrolearn_kernel::{
    CompoundKernel, ConstantKernel, DotProductKernel, ExpSineSquared, Exponentiation, GPKernel,
    Hyperparameter, HyperparameterBounds, MaternKernel, ProductKernel, RBFKernel,
    RationalQuadratic, SumKernel, WhiteKernel,
};
use ndarray::{Array2, Array3, array};

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

fn assert_stack(actual: &Array3<f64>, expected: &[&[&[f64]]]) {
    assert_eq!(actual.shape()[0], expected.len(), "row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(actual.shape()[1], row.len(), "col count");
        for (j, cell) in row.iter().enumerate() {
            assert_eq!(actual.shape()[2], cell.len(), "stack depth");
            for (k, &e) in cell.iter().enumerate() {
                let a = actual[[i, j, k]];
                assert!(
                    (a - e).abs() < 1e-12,
                    "K[{i},{j},{k}] = {a} but sklearn oracle = {e}"
                );
            }
        }
    }
}

fn assert_default_log_bounds(actual: &Array2<f64>, rows: usize) {
    assert_eq!(actual.dim(), (rows, 2), "bounds shape");
    for (i, row) in actual.rows().into_iter().enumerate() {
        assert!(
            (row[0] - 1e-5_f64.ln()).abs() < 1e-12,
            "bounds[{i},0] = {} != sklearn ln(1e-5)",
            row[0]
        );
        assert!(
            (row[1] - 1e5_f64.ln()).abs() < 1e-12,
            "bounds[{i},1] = {} != sklearn ln(1e5)",
            row[1]
        );
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
// REQ-19 — Hyperparameter / bounds / n_dims metadata  (GREEN GUARD)
// oracle: sklearn Hyperparameter repeats a single numeric bounds row for
// vector-valued parameters, derives fixed=True from bounds="fixed", and scalar
// GP kernels expose default numeric bounds `(1e-5, 1e5)` in log space.
// ---------------------------------------------------------------------------

#[test]
fn green_hyperparameter_constructor_repeats_bounds_and_derives_fixed() {
    // sklearn 1.5.2, /tmp:
    // Hyperparameter("length_scale","numeric",(1e-5,1e5),3).bounds.tolist()
    // == [[1e-5,1e5],[1e-5,1e5],[1e-5,1e5]], fixed == False
    let hp = Hyperparameter::<f64>::new(
        "length_scale",
        "numeric",
        HyperparameterBounds::Numeric(vec![(1e-5_f64, 1e5_f64)]),
        3,
        None,
    );
    assert_eq!(hp.name, "length_scale");
    assert_eq!(hp.value_type, "numeric");
    assert_eq!(hp.n_elements, 3);
    assert!(!hp.fixed);
    let bounds = hp.bounds_array().unwrap();
    assert_eq!(bounds.dim(), (3, 2));
    for row in bounds.rows() {
        assert!((row[0] - 1e-5_f64).abs() < 1e-15);
        assert!((row[1] - 1e5_f64).abs() < 1e-9);
    }

    // sklearn 1.5.2, /tmp:
    // Hyperparameter("length_scale","numeric","fixed").fixed == True
    let fixed = Hyperparameter::<f64>::fixed("length_scale", "numeric", 1);
    assert!(fixed.fixed);
    assert!(fixed.bounds_array().is_none());
}

#[test]
fn green_scalar_kernel_hyperparameters_bounds_and_n_dims() {
    // sklearn 1.5.2, /tmp:
    // RBF(1.5).hyperparameters == [Hyperparameter(name="length_scale", ...)]
    // RBF(1.5).bounds == [[ln(1e-5), ln(1e5)]], n_dims == 1
    let rbf = RBFKernel::new(1.5);
    let rbf_hyperparameters = rbf.hyperparameters();
    assert_eq!(rbf_hyperparameters.len(), 1);
    assert_eq!(rbf_hyperparameters[0].name, "length_scale");
    assert_eq!(rbf_hyperparameters[0].value_type, "numeric");
    assert_eq!(rbf_hyperparameters[0].n_elements, 1);
    assert!(!rbf_hyperparameters[0].fixed);
    assert_eq!(rbf.n_dims(), 1);
    assert_default_log_bounds(&rbf.bounds(), 1);

    // sklearn 1.5.2, /tmp:
    // RationalQuadratic(...).hyperparameters names: ["alpha", "length_scale"]
    let rq = RationalQuadratic::new(1.3, 0.7);
    let rq_names: Vec<_> = rq
        .hyperparameters()
        .into_iter()
        .map(|hyperparameter| hyperparameter.name)
        .collect();
    assert_eq!(rq_names, vec!["alpha", "length_scale"]);
    assert_eq!(rq.n_dims(), 2);
    assert_default_log_bounds(&rq.bounds(), 2);

    // sklearn 1.5.2, /tmp:
    // ExpSineSquared(...).hyperparameters names: ["length_scale", "periodicity"]
    let periodic = ExpSineSquared::new(1.3, 2.0);
    let periodic_names: Vec<_> = periodic
        .hyperparameters()
        .into_iter()
        .map(|hyperparameter| hyperparameter.name)
        .collect();
    assert_eq!(periodic_names, vec!["length_scale", "periodicity"]);
    assert_eq!(periodic.n_dims(), 2);
    assert_default_log_bounds(&periodic.bounds(), 2);
}

#[test]
fn green_composite_hyperparameter_names_bounds_and_n_dims() {
    // sklearn 1.5.2, /tmp:
    // (ConstantKernel(2.0)*RBF(1.5)+WhiteKernel(0.1)).hyperparameters names
    // == ["k1__k1__constant_value", "k1__k2__length_scale", "k2__noise_level"]
    // theta == [ln(2.0), ln(1.5), ln(0.1)], bounds.shape == (3,2), n_dims == 3
    let k = SumKernel::new(
        Box::new(ProductKernel::new(
            Box::new(ConstantKernel::new(2.0)),
            Box::new(RBFKernel::new(1.5)),
        )),
        Box::new(WhiteKernel::new(0.1)),
    );
    let names: Vec<_> = k
        .hyperparameters()
        .into_iter()
        .map(|hyperparameter| hyperparameter.name)
        .collect();
    assert_eq!(
        names,
        vec![
            "k1__k1__constant_value",
            "k1__k2__length_scale",
            "k2__noise_level"
        ]
    );
    let theta = k.get_params();
    let theta_oracle = [
        std::f64::consts::LN_2,
        0.405_465_108_108_164_4_f64,
        -2.302_585_092_994_045_5,
    ];
    assert_eq!(theta.len(), theta_oracle.len());
    for (i, &e) in theta_oracle.iter().enumerate() {
        assert!(
            (theta[i] - e).abs() < 1e-12,
            "theta[{i}] = {} != sklearn {e}",
            theta[i]
        );
    }
    assert_eq!(k.n_dims(), 3);
    assert_default_log_bounds(&k.bounds(), 3);
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

// ---------------------------------------------------------------------------
// REQ-18 — CompoundKernel  (GREEN GUARD)
// oracle: CompoundKernel([RBF(1.5), DotProduct(0.5), WhiteKernel(0.1)])(X),
// (X, X2), .diag(X), and theta child-concatenation order.
// ---------------------------------------------------------------------------

fn compound_oracle_kernel() -> CompoundKernel<f64> {
    CompoundKernel::new(vec![
        Box::new(RBFKernel::new(1.5)),
        Box::new(DotProductKernel::new(0.5)),
        Box::new(WhiteKernel::new(0.1)),
    ])
}

#[test]
fn green_compound_kernel_self_stack() {
    // sklearn 1.5.2, /tmp:
    // CompoundKernel([RBF(length_scale=1.5), DotProduct(sigma_0=0.5),
    //                 WhiteKernel(noise_level=0.1)])(X)
    let expected: &[&[&[f64]]] = &[
        &[
            &[1.0, 0.25, 0.1],
            &[0.8007374029168081, 0.25, 0.0],
            &[0.8007374029168081, 0.25, 0.0],
        ],
        &[
            &[0.8007374029168081, 0.25, 0.0],
            &[1.0, 1.25, 0.1],
            &[0.6411803884299546, 0.25, 0.0],
        ],
        &[
            &[0.8007374029168081, 0.25, 0.0],
            &[0.6411803884299546, 0.25, 0.0],
            &[1.0, 1.25, 0.1],
        ],
    ];
    let k = compound_oracle_kernel();
    assert_stack(&k.compute_stack(&x3(), &x3()), expected);
}

#[test]
fn green_compound_kernel_cross_and_diag_stack() {
    // sklearn 1.5.2, /tmp:
    // CompoundKernel([...])(X, X2)
    let expected_cross: &[&[&[f64]]] = &[
        &[&[1.0, 0.25, 0.0], &[0.6411803884299546, 0.25, 0.0]],
        &[
            &[0.8007374029168081, 0.25, 0.0],
            &[0.8007374029168081, 1.25, 0.0],
        ],
        &[
            &[0.8007374029168081, 0.25, 0.0],
            &[0.8007374029168081, 1.25, 0.0],
        ],
    ];
    let k = compound_oracle_kernel();
    assert_stack(&k.compute_stack(&x3(), &x2()), expected_cross);

    // sklearn 1.5.2, /tmp: CompoundKernel([...]).diag(X)
    let expected_diag: &[&[f64]] = &[&[1.0, 0.25, 0.1], &[1.0, 1.25, 0.1], &[1.0, 1.25, 0.1]];
    assert_mat(&k.diagonal_stack(&x3()), expected_diag);
}

#[test]
fn green_compound_kernel_theta_log_order() {
    // sklearn 1.5.2, /tmp:
    // CompoundKernel([RBF(1.5), DotProduct(0.5), WhiteKernel(0.1)]).theta
    // == [ln(1.5), ln(0.5), ln(0.1)].
    let k = compound_oracle_kernel();
    let params = k.get_params();
    let theta_oracle = [
        0.405_465_108_108_164_4_f64,
        -std::f64::consts::LN_2,
        -2.302_585_092_994_045_5,
    ];
    assert_eq!(params.len(), theta_oracle.len());
    for (i, &e) in theta_oracle.iter().enumerate() {
        assert!(
            (params[i] - e).abs() < 1e-12,
            "theta[{i}] = {} != sklearn {e}",
            params[i]
        );
    }
    assert_eq!(k.hyperparameters().len(), 0);
    assert_eq!(k.n_dims(), 3);
    assert_default_log_bounds(&k.bounds(), 3);
}
