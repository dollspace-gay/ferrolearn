//! Divergence: `BayesianRidge::fit` diverges from sklearn's `BayesianRidge`.
//!
//! Upstream contract: `sklearn/linear_model/_bayes.py`. The fitted attributes
//! `coef_`, `alpha_`, and `lambda_` are produced by an SVD-based MacKay (1992)
//! evidence-maximization loop WITH Gamma hyperpriors (`alpha_1 == alpha_2 ==
//! lambda_1 == lambda_2 == 1e-6`) and init `alpha_ = 1/var(y)`, `lambda_ = 1`:
//!
//!   `_bayes.py:305` — `gamma_ = np.sum((alpha_*eigen_vals_)/(lambda_ + alpha_*eigen_vals_))`
//!   `_bayes.py:306` — `lambda_ = (gamma_ + 2*lambda_1) / (np.sum(coef_**2) + 2*lambda_2)`
//!   `_bayes.py:307` — `alpha_ = (n_samples - gamma_ + 2*alpha_1) / (rmse_ + 2*alpha_2)`
//!   `_bayes.py:269` — `alpha_ = 1.0 / (np.var(y) + eps)`  (init when alpha_init is None)
//!
//! ferrolearn's `bayesian_ridge.rs` instead:
//!   - omits the four Gamma hyperprior terms (`+ 2*lambda_1`, etc.) — see `new_alpha`
//!     `= (n_f - gamma) / sse` and `new_lambda = gamma / w_norm_sq`,
//!   - approximates `gamma` with a Cholesky-diagonal sum
//!     `sum_i alpha * xtx[[i,i]] * sigma_diag[i]` instead of the exact eigenvalue sum,
//!   - initializes `alpha_init = 1.0` instead of `1/var(y)`.
//!
//! These differences change the converged regularization strength, so on a
//! regularization-sensitive dataset the fitted `coef_`, `alpha_`, and `lambda_`
//! all diverge from sklearn.
//!
//! Tracking: #464 (hyperprior + exact-gamma fit), contributing #466 (alpha_init default).

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::BayesianRidge;
use ndarray::{Array1, Array2};

/// Regularization-sensitive design: 30x5, RandomState(0), with two collinear
/// feature pairs (col4 ~ 0.9*col0, col3 ~ 0.8*col1) and moderate noise so the
/// Bayesian evidence loop actually shrinks the coefficients. Built by the live
/// sklearn 1.5.2 oracle (see module-level header for the generating script).
fn dataset() -> (Array2<f64>, Array1<f64>) {
    let x_data: Vec<f64> = vec![
        1.764052345967664, 0.4001572083672233, 0.9787379841057392, 0.24932698444308188, 1.5808229508384346,
        -0.977277879876411, 0.9500884175255894, -0.1513572082976979, 0.48508047533686777, -0.7082158197238333,
        0.144043571160878, 1.454273506962975, 0.7610377251469934, 1.034695125003802, 0.0551637318399462,
        0.33367432737426683, 1.4940790731576061, -0.20515826376580087, 0.7505826280811996, 0.21766304077093873,
        -2.5529898158340787, 0.6536185954403606, 0.8644361988595057, 0.6479411665577259, -2.307536086693214,
        -1.4543656745987648, 0.04575851730144607, -0.1871838500258336, -0.28380471728019274, -1.375276935775099,
        0.1549474256969163, 0.37816251960217356, -0.8877857476301128, 0.08165334779604874, 0.2521162753378754,
        0.15634896910398005, 1.2302906807277207, 1.2023798487844113, 0.9946655604343716, 0.032720921357239724,
        -1.0485529650670926, -1.4200179371789752, -1.7062701906250126, -1.2839269490214429, -1.0584445338014936,
        -0.4380743016111864, -1.2527953600499262, 0.7774903558319101, -0.6936333689585937, -0.43804887592451114,
        -0.8954665611936756, 0.386902497859262, -0.510805137568873, 0.05095061634271991, -0.8557231501435386,
        0.42833187053041766, 0.06651722238316789, 0.3024718977397814, 0.1066239517763709, 0.5784518888590745,
        -0.672460447775951, -0.3595531615405413, -0.813146282044454, -0.2954990928779322, -0.5102723223057799,
        -0.4017809362082619, -1.6301983469660446, 0.4627822555257742, -1.5377773771210752, -0.3528477184489166,
        0.7290905621775369, 0.12898291075741067, 1.1394006845433007, 0.20784166071227927, 0.5336379540767664,
        -0.6848100909403132, -0.8707971491818818, -0.5788496647644155, -0.7309469855899551, -0.5318927842061272,
        -1.1651498407833565, 0.9008264869541871, 0.46566243973045984, 0.8750192998060832, -1.1486563914439771,
        1.8958891760305832, 1.1787795711596507, -0.17992483581235091, 1.1077244877204668, 1.5518231487497638,
        -0.40317694697317963, 1.2224450703824274, 0.2082749780768603, 1.4106032461620799, -0.2440562730406315,
        0.7065731681919482, 0.010500020720820478, 1.7858704939058352, 0.2757056064639348, 0.6676101125652383,
        1.8831506970562544, -1.3477590611424464, -1.2704849984857336, -1.152043616502446, 1.7869215097287108,
        1.9436211856492926, -0.41361898075974735, -0.7474548114407578, -0.37877102012298314, 1.7811318323786656,
        1.8675589604265699, 0.9060446582753853, -0.8612256850547025, 0.9447676457977309, 1.766486125574182,
        0.8024563957963952, 0.947251967773748, -0.1550100930908342, 0.888854320363518, 0.6571081968867409,
        0.37642553115562943, -1.0994007905841945, 0.298238174206056, -0.7514943272478373, 0.23535869386162006,
        -0.14963454032767076, -0.43515355172163744, 1.8492637284793418, -0.6715140502394769, -0.06651163446674098,
        -0.7699160744453164, 0.5392491912918173, -0.6743326606573761, 0.4265341281536667, -0.7732654334181689,
        0.6764332949464997, 0.5765908166149409, -0.20829875557799488, 0.31366647145081505, 0.5398349876768298,
        -1.4912575927056055, 0.4393917012645369, 0.16667349537252904, 0.4074982808202772, -1.3876850837867793,
        0.9444794869904138, -0.9128222254441586, 1.117016288095853, -0.7498878582839186, 0.8517794541938781,
    ];
    let y_data: Vec<f64> = vec![
        7.3181665014724935, -5.735725096670511, -2.0005067216289083, -1.8938501514039752, -12.899733896381903,
        -6.722554308663109, 0.07818144396249455, -1.7528772043301009, -0.7636353462001417, 0.5131659255492051,
        -5.221761904761247, 1.8466481122638356, -2.2954814910284873, 1.7665230445206597, 1.9593642889106269,
        -1.079044153666201, -6.941833814441722, 5.7739336269972314, -4.319163425728022, 2.9811737671140577,
        10.313322023555456, 9.083139325354274, 6.168886015081853, 1.7065525703025137, 3.103024999516751,
        0.7122350816151478, -3.800902483801152, 0.6508782206180959, -7.220954440937009, 6.2752061106655,
    ];
    (
        Array2::from_shape_vec((30, 5), x_data).unwrap(),
        Array1::from_vec(y_data),
    )
}

/// sklearn 1.5.2 oracle output for `BayesianRidge().fit(X, y)` on `dataset()`:
///   python3 -c "from sklearn.linear_model import BayesianRidge; ...;
///               m = BayesianRidge().fit(X, y);
///               print(m.coef_.tolist(), m.intercept_, m.alpha_, m.lambda_)"
/// These are the EXPECTED values (NOT copied from ferrolearn — R-CHAR-3).
const SK_COEF: [f64; 5] = [
    3.072698040292568,
    -1.7597074435726576,
    0.019418762922079454,
    -0.3275335822258911,
    1.3450680544293254,
];
const SK_INTERCEPT: f64 = -0.041720682724152955;
const SK_ALPHA: f64 = 4.460716675567307;
const SK_LAMBDA: f64 = 0.315315465242343;

/// Divergence: ferrolearn's `BayesianRidge::fit` diverges from
/// `sklearn/linear_model/_bayes.py:305-307` (hyperprior-free, gamma-approximated,
/// wrong init). sklearn returns coef_ ≈ SK_COEF, alpha_ ≈ 4.46, lambda_ ≈ 0.315;
/// ferrolearn returns a different converged regularization strength.
/// Tracking: #464.
#[test]
#[ignore = "divergence: BayesianRidge fit omits Gamma hyperpriors + approximates gamma + alpha_init=1.0; tracking #464"]
fn divergence_bayesian_ridge_fit_coef_alpha_lambda() {
    let (x, y) = dataset();
    let fitted = BayesianRidge::<f64>::new().fit(&x, &y).unwrap();

    let coef = fitted.coefficients();
    let alpha = fitted.alpha();
    let lambda = fitted.lambda();
    let intercept = fitted.intercept();

    // If the algorithm matched sklearn's SVD/MacKay loop, coef_ would agree to
    // ~1e-3 and alpha_/lambda_ to ~1e-2. They do not.
    for (i, (&got, &want)) in coef.iter().zip(SK_COEF.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "coef_[{i}]: ferrolearn={got}, sklearn={want} (diff {})",
            (got - want).abs()
        );
    }
    assert!(
        (intercept - SK_INTERCEPT).abs() < 1e-3,
        "intercept: ferrolearn={intercept}, sklearn={SK_INTERCEPT}"
    );
    assert!(
        (alpha - SK_ALPHA).abs() / SK_ALPHA < 1e-2,
        "alpha_: ferrolearn={alpha}, sklearn={SK_ALPHA} (rel diff {})",
        (alpha - SK_ALPHA).abs() / SK_ALPHA
    );
    assert!(
        (lambda - SK_LAMBDA).abs() / SK_LAMBDA < 1e-2,
        "lambda_: ferrolearn={lambda}, sklearn={SK_LAMBDA} (rel diff {})",
        (lambda - SK_LAMBDA).abs() / SK_LAMBDA
    );
}
