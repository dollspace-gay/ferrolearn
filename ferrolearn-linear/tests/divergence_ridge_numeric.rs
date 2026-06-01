//! Adversarial divergence audit of `ferrolearn_linear::Ridge` (single-output,
//! dense Cholesky path) against the live scikit-learn 1.5.2 oracle.
//!
//! Mirrors `sklearn.linear_model.Ridge` (`sklearn/linear_model/_ridge.py:893`).
//! All expected values are computed by live-calling sklearn 1.5.2 (R-CHAR-3).
//!
//! The well-conditioned numerical probes (coef_/intercept_ parity across alpha,
//! alpha=0 -> OLS, fit_intercept=False, f32) PASS and are kept as live
//! regression guards. The rank-deficient alpha=0 case is a real divergence and
//! is pinned `#[ignore]` with its tracking issue.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::Ridge;
use ndarray::{Array1, Array2, array};

/// Fixed 20x5 design matrix, row-major, from
/// `np.random.RandomState(42).randn(20,5) * [1,10,0.5,3,2]`.
const X_FLAT: [f64; 100] = [
    0.4967141530112327,
    -1.3826430117118464,
    0.32384426905034625,
    4.569089569224076,
    -0.46830674944667194,
    -0.23413695694918055,
    15.792128155073915,
    0.3837173645764544,
    -1.4084231578048563,
    1.0851200871719293,
    -0.46341769281246226,
    -4.657297535702568,
    0.12098113578301706,
    -5.739840733973393,
    -3.4498356650260655,
    -0.5622875292409727,
    -10.128311203344238,
    0.15712366629763694,
    -2.724072226563633,
    -2.824607402670583,
    1.465648768921554,
    -2.2577630048653567,
    0.03376410234396192,
    -4.274244558640371,
    -1.0887654490503653,
    0.11092258970986608,
    -11.509935774223027,
    0.18784900917283598,
    -1.801916069756415,
    -0.5833874995865536,
    -0.6017066122293969,
    18.522781845089376,
    -0.0067486123689669605,
    -3.173132786867701,
    1.645089824206378,
    -1.2208436499710222,
    2.088635950047554,
    -0.9798350619398878,
    -3.9845581466952913,
    0.39372247173824704,
    0.7384665799954104,
    1.713682811899705,
    -0.057824141194120264,
    -0.9033110867678664,
    -2.9570439807348547,
    -0.7198442083947086,
    -4.606387709597875,
    0.5285611131094579,
    1.0308548687053842,
    -3.526080310725468,
    0.324083969394795,
    -3.8508228041631654,
    -0.33846100015297936,
    1.8350288665226038,
    2.061999044991902,
    0.9312801191161986,
    -8.392175232226386,
    -0.1546061879256073,
    0.9937902942106919,
    1.9510902542447184,
    -0.47917423784528995,
    -1.8565897666381712,
    -0.5531674870030141,
    -3.5886198722420124,
    1.625051644788396,
    1.356240028570823,
    -0.7201012158033384,
    0.5017664489460121,
    1.0849080751429026,
    -1.2902395092102485,
    0.36139560550841393,
    15.380365664659692,
    -0.01791301955497577,
    4.693930967442019,
    -5.239490208179489,
    0.8219025043752238,
    0.8704706823817122,
    -0.14950367523293373,
    0.2752823296065069,
    -3.9751378292017856,
    -0.21967188783751193,
    3.5711257151174642,
    0.738947022370758,
    -1.5548106548209422,
    -1.6169872057863752,
    -0.5017570435845365,
    9.154021177020741,
    0.16437555482984223,
    -1.5892806113011164,
    1.0265348662267122,
    0.09707754934804039,
    9.686449905328892,
    -0.3510265469386762,
    -0.9829864397933046,
    -0.7842163062643153,
    -1.4635149481321186,
    2.9612027706457607,
    0.13052763608994467,
    0.01534036992738267,
    -0.46917426675029383,
];

const Y: [f64; 20] = [
    23.188644169901533,
    -33.55035263928976,
    -2.476771315001646,
    17.560579747590857,
    -2.185859585970603,
    22.05355804446233,
    -46.045166050350495,
    -16.259100740132915,
    1.9374341789044796,
    19.776066702729473,
    15.546368050815033,
    23.203703615260313,
    -7.016778431065476,
    13.071583308076502,
    -4.997001709667257,
    8.698497853193157,
    -6.310156757846288,
    -21.436297204942417,
    -17.739937102468453,
    -3.384400400772374,
];

fn design() -> Array2<f64> {
    Array2::from_shape_vec((20, 5), X_FLAT.to_vec()).unwrap()
}
fn target() -> Array1<f64> {
    Array1::from_vec(Y.to_vec())
}

/// Regression guard: ferrolearn's `Ridge::fit` coef_/intercept_ vs
/// `sklearn/linear_model/_ridge.py:893` `Ridge(alpha=a).fit(X, y)` across
/// alpha in {0.1, 1.0, 10.0, 100.0}. PASSES against current ferrolearn.
#[test]
fn ridge_coef_intercept_parity_across_alphas() {
    let x = design();
    let y = target();

    let cases: [(f64, [f64; 5], f64); 4] = [
        (
            0.1,
            [
                1.5154967733145186,
                -1.9961436421178604,
                0.694892362237892,
                3.2883375231752066,
                -1.104818962265875,
            ],
            3.98405016849177,
        ),
        (
            1.0,
            [
                1.4284907624832863,
                -1.9959970541985754,
                0.5639141411179915,
                3.2802196120812455,
                -1.1071406225174127,
            ],
            3.980059877867675,
        ),
        (
            10.0,
            [
                0.9618615401525885,
                -1.9869595296376286,
                0.27050263677029857,
                3.1411073582035556,
                -1.073093855624853,
            ],
            3.8932311446812036,
        ),
        (
            100.0,
            [
                0.3685919420338514,
                -1.848055005376819,
                0.11074662545246021,
                2.0240296190790312,
                -0.7181191552973881,
            ],
            3.060461685226366,
        ),
    ];

    for (alpha, sk_coef, sk_int) in cases {
        let fitted = Ridge::<f64>::new().with_alpha(alpha).fit(&x, &y).unwrap();
        let coef = fitted.coefficients();
        for j in 0..5 {
            let diff = (coef[j] - sk_coef[j]).abs();
            assert!(
                diff < 1e-8,
                "alpha={alpha} coef[{j}]: ferrolearn={} sklearn={} diff={diff}",
                coef[j],
                sk_coef[j]
            );
        }
        let idiff = (fitted.intercept() - sk_int).abs();
        assert!(
            idiff < 1e-8,
            "alpha={alpha} intercept: ferrolearn={} sklearn={} diff={idiff}",
            fitted.intercept(),
            sk_int
        );
    }
}

/// Regression guard: alpha=0 reduces to OLS on a well-conditioned design.
/// Oracle: `Ridge(alpha=0.0).fit(X,y).coef_` ==
/// [1.5259566279689956, -1.9961436404050774, 0.7149519903369858,
///  3.2890152574944063, -1.1042717959640993], intercept 3.9843511477437166.
/// PASSES against current ferrolearn.
#[test]
fn ridge_alpha_zero_equals_ols() {
    let x = design();
    let y = target();

    let sk_coef = [
        1.5259566279689956,
        -1.9961436404050774,
        0.7149519903369858,
        3.2890152574944063,
        -1.1042717959640993,
    ];
    let sk_int = 3.9843511477437166;

    let fitted = Ridge::<f64>::new().with_alpha(0.0).fit(&x, &y).unwrap();
    let coef = fitted.coefficients();
    for j in 0..5 {
        let diff = (coef[j] - sk_coef[j]).abs();
        assert!(
            diff < 1e-7,
            "alpha=0 coef[{j}]: ferrolearn={} sklearn={} diff={diff}",
            coef[j],
            sk_coef[j]
        );
    }
    let idiff = (fitted.intercept() - sk_int).abs();
    assert!(
        idiff < 1e-7,
        "alpha=0 intercept: ferrolearn={} sklearn={} diff={idiff}",
        fitted.intercept(),
        sk_int
    );
}

/// Regression guard: fit_intercept=False penalized fit through origin. Oracle:
/// `Ridge(alpha=1.0, fit_intercept=False).fit(X,y)` ->
/// coef=[1.911425502342288, -1.8980263525490693, 0.740870161513888,
///       2.780939597386463, -1.773843724145664], intercept=0.0.
/// PASSES against current ferrolearn.
#[test]
fn ridge_fit_intercept_false_parity() {
    let x = design();
    let y = target();

    let sk_coef = [
        1.911425502342288,
        -1.8980263525490693,
        0.740870161513888,
        2.780939597386463,
        -1.773843724145664,
    ];

    let fitted = Ridge::<f64>::new()
        .with_alpha(1.0)
        .with_fit_intercept(false)
        .fit(&x, &y)
        .unwrap();
    let coef = fitted.coefficients();
    for j in 0..5 {
        let diff = (coef[j] - sk_coef[j]).abs();
        assert!(
            diff < 1e-8,
            "fit_intercept=False coef[{j}]: ferrolearn={} sklearn={} diff={diff}",
            coef[j],
            sk_coef[j]
        );
    }
    assert!(
        fitted.intercept().abs() < 1e-12,
        "fit_intercept=False intercept should be 0, got {}",
        fitted.intercept()
    );
}

/// Regression guard: f32 parity at alpha=10.0. PASSES against current ferrolearn.
#[test]
fn ridge_f32_parity_alpha10() {
    let x32: Array2<f32> =
        Array2::from_shape_vec((20, 5), X_FLAT.iter().map(|&v| v as f32).collect()).unwrap();
    let y32: Array1<f32> = Array1::from_vec(Y.iter().map(|&v| v as f32).collect());

    let sk_coef: [f32; 5] = [
        0.9618615401525885,
        -1.9869595296376286,
        0.27050263677029857,
        3.1411073582035556,
        -1.073093855624853,
    ]
    .map(|v: f64| v as f32);
    let sk_int = 3.8932311446812036_f64 as f32;

    let fitted = Ridge::<f32>::new()
        .with_alpha(10.0)
        .fit(&x32, &y32)
        .unwrap();
    let coef = fitted.coefficients();
    for j in 0..5 {
        let diff = (coef[j] - sk_coef[j]).abs();
        assert!(
            diff < 1e-3,
            "f32 alpha=10 coef[{j}]: ferrolearn={} sklearn={} diff={diff}",
            coef[j],
            sk_coef[j]
        );
    }
    let idiff = (fitted.intercept() - sk_int).abs();
    assert!(
        idiff < 1e-2,
        "f32 alpha=10 intercept: ferrolearn={} sklearn={} diff={idiff}",
        fitted.intercept(),
        sk_int
    );
}

/// Divergence: ferrolearn's `Ridge::fit` (single-output) diverges from
/// `sklearn/linear_model/_ridge.py:753-756` for a rank-deficient design at
/// `alpha=0`.
///
/// sklearn's default solver `'auto'` -> `'cholesky'` solves `(X^T X + alpha*I)`
/// with `scipy.linalg.solve(assume_a="pos")` and, when that factorization
/// reports a singular/ill-conditioned matrix, falls through `except
/// linalg.LinAlgError: solver = "svd"` (`_ridge.py:754-756`) to a min-norm SVD
/// solve. On `X = [[1,2],[2,4],[3,6],[4,8]]` (col2 = 2*col1, rank 1), `y =
/// [1,2,3,4]`, `Ridge(alpha=0.0).fit(X,y)` returns a finite minimum-norm
/// solution `coef = [0.2, 0.4]`, `intercept ~ 0`.
///
/// ferrolearn's `crate::linalg::solve_ridge` forms `X^T X` (singular for
/// alpha=0), the Cholesky solve fails, and the Gaussian-elimination fallback
/// (`gaussian_solve`) detects the singular pivot and returns
/// `Err(FerroError::NumericalInstability)` instead of the min-norm coef.
/// The OLS path (`solve_lstsq` via ferray gelsd) already produces min-norm
/// solutions, but the Ridge alpha=0 path does not route through it.
///
/// sklearn returns coef=[0.2, 0.4]; ferrolearn returns Err(NumericalInstability).
/// Fix belongs in the owning crate (ferrolearn-linear): the Ridge cholesky
/// path must fall back to the min-norm SVD/lstsq solve on a singular system,
/// mirroring `_ridge.py:754-756`.
///
/// Tracking: #392
#[test]
fn divergence_ridge_alpha_zero_rank_deficient_min_norm() {
    // Oracle: Ridge(alpha=0.0).fit([[1,2],[2,4],[3,6],[4,8]], [1,2,3,4])
    //   -> coef=[0.19999999999999998, 0.39999999999999997], intercept~4.4e-16
    let x = Array2::from_shape_vec((4, 2), vec![1., 2., 2., 4., 3., 6., 4., 8.]).unwrap();
    let y: Array1<f64> = array![1., 2., 3., 4.];

    let fitted = Ridge::<f64>::new()
        .with_alpha(0.0)
        .fit(&x, &y)
        .expect("sklearn returns a finite min-norm coef here; ferrolearn must not error");

    let coef = fitted.coefficients();
    assert!(
        (coef[0] - 0.2).abs() < 1e-9,
        "coef[0]: ferrolearn={} sklearn=0.2",
        coef[0]
    );
    assert!(
        (coef[1] - 0.4).abs() < 1e-9,
        "coef[1]: ferrolearn={} sklearn=0.4",
        coef[1]
    );
    assert!(
        fitted.intercept().abs() < 1e-9,
        "intercept: ferrolearn={} sklearn~0",
        fitted.intercept()
    );
    // Reference the error type so the import is load-bearing if the call path
    // changes; the divergence is the `.expect` panic above.
    let _ = std::any::type_name::<FerroError>();
}
