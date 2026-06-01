//! Divergence + stable-contract: ferrolearn `LinearRegression::fit` (via
//! `linalg::solve_lstsq` -> `ferray::linalg::lstsq`) vs scikit-learn 1.5.2
//! `sklearn/linear_model/_base.py:687` (`self.coef_, _, self.rank_, self.singular_
//! = linalg.lstsq(X, y)`).
//!
//! scikit-learn calls `linalg.lstsq(X, y)` with NO `cond` argument. scipy's
//! `cond=None` default sets the singular-value cutoff to `eps * s_max` (machine
//! epsilon, `scipy/linalg/_basic.py`). ferray's `lstsq(_, _, None)` instead uses
//! the larger numpy convention `max(m, n) * eps * s_max`
//! (`ferray-linalg/src/solve.rs:222`). For a singular-value ratio in
//! `(eps, max(m,n)*eps)` the two make a DIFFERENT rank decision. The fix
//! (`linalg.rs`, `solve_lstsq`) passes `Some(F::epsilon())`, pinning ferray to
//! scipy's `cond=eps` cutoff so the RANK decision matches scipy/sklearn.
//!
//! ## Why the individual coefficients (and `X @ coef`) are NOT asserted on the
//! ## near-singular design.
//!
//! Per the ferray investigation (#382, closed — ferray's SVD is correct /
//! backward-stable), the residual divergence on the `cond~1e14` design is an
//! INHERENT floating-point limit. The 50x2 design's two columns are anti-parallel
//! to within FP noise (`cos == -1.0`), i.e. the matrix is effectively rank-1; the
//! second singular value (`~4.97e-15`) is pure rounding noise. With the `eps`
//! cutoff scipy and ferray BOTH KEEP it (`rank == 2`) — that is the deterministic
//! contract this test pins — but the resulting coefficient is `1/s_min`-amplified
//! noise (`~1e14`), and the fitted vector `X @ coef` is then the projection of `y`
//! onto that noise direction. No implementation (scipy SVD/gelsd, faer QR,
//! LAPACK) has a "true" answer for either the coefficients OR `X @ coef` there, so
//! asserting their values would be asserting numerically-meaningless quantities.
//!
//! The deterministic, meaningful contract is therefore:
//!   (a) RANK parity on the near-singular design — the `eps` cutoff KEEPS the tiny
//!       singular value (`rank == 2`, coefficient magnitude `~1e14`), exactly as
//!       scipy does; the OLD `max(m,n)*eps` cutoff ZEROED it (small-norm `~0.2`
//!       solution, `rank == 1`). The coefficient *magnitude* is the observable
//!       signature of which rank decision was taken.
//!   (b) Well-conditioned COEFFICIENT parity — on a `cond == 1e6` (genuinely
//!       full-rank) design, ferrolearn `coef_` matches scipy/sklearn within a
//!       tolerance scaled by `eps * cond ~ 2e-10`. The stable regime.
//!   (c) Stable FITTED-VALUE parity — on the same full-rank design, `X @ coef`
//!       (the projection onto `col(X)`) matches scipy's `X @ x_scipy`.
//!
//! All expected values come from the live scipy/sklearn 1.5.2 oracle (R-CHAR-3).
//!
//! Tracking: #381 (rcond `eps` cutoff in `solve_lstsq`).
use ferrolearn_core::Fit;
use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_linear::LinearRegression;
use ndarray::{Array1, Array2};

/// The near-rank-deficient 50x2 design: singular values [1.0, ~4.97e-15], whose
/// ratio (~4.97e-15) lies between `eps` (~2.22e-16) and `max(m,n)*eps` (50*eps
/// ~1.11e-14). scipy's `cond=eps` cutoff KEEPS the tiny sv (`rank == 2`); ferray's
/// old `max(m,n)*eps` default ZEROED it (`rank == 1`).
#[rustfmt::skip]
fn near_singular_design() -> (Array2<f64>, Array1<f64>) {
    let x_flat: Vec<f64> = vec![0.09289440750930184, -0.1640736342519232, 0.046945866816749796, -0.08291757478468112, 0.1693302302542488, -0.29907749036181763, 0.10609717384592196, -0.18739286210545492, 0.02608350871887028, -0.046069684755977024, 0.1751801147500193, -0.30940977876226666, 0.012223378140501577, -0.021589395186646692, 0.10565602614583605, -0.1866136902657766, 0.04732819650327305, -0.08359286001263108, -0.03390320879480568, 0.0598811363236119, -0.10723701700085382, 0.18940609642086034, 0.0004830811609903808, -0.000853236312575793, 0.07241288486604643, -0.12789839028198452, 0.05254233326008073, -0.09280226658632948, -0.04137106955620785, 0.07307115591750254, 0.00031254097254846433, -0.0005520217480655455, 0.027669328426919666, -0.048870619814873235, -0.02228479304176698, 0.03936024870548315, -0.05259188904165563, 0.09288979389178245, 0.011991949295553164, -0.02118063676210956, 0.04725434968250264, -0.08346242895023873, -0.12832269625563947, 0.22664842476723346, 0.04038443313105772, -0.07132852115293443, -0.12084262490080074, 0.21343683835898775, -0.03331222408817604, 0.05883731666639187, 0.03167605147231999, -0.05594744638738518, -0.018047422401897245, 0.031876043582733236, -0.04275815119224335, 0.07552107223787605, -0.02766599173527174, 0.04886472642322889, -0.04388366801207094, 0.07750900283555377, -0.10190646281023732, 0.17999106894954955, 0.035697871092292284, -0.06305093710385741, 0.021361579969206797, -0.03772963467752207, -0.053326312997542376, 0.09418696140439511, 0.15977060249433422, -0.2821929123692405, -0.01899569131843541, 0.03355091219491436, -0.04765126974100375, 0.08416348424788123, 0.13350472312006073, -0.23580111762822906, -0.060740255085035603, 0.1072817477865165, -0.03832054357989921, 0.06768320096162808, 0.004542230503793759, -0.008022660204736451, -0.018466126554874365, 0.032615574776241524, -0.0004086159567661836, 0.0007217130378196633, -0.04593560075753748, 0.08113320446206529, -0.06432956900259099, 0.11362133048814951, 0.029943766181322173, -0.05288781824747923, 0.02585610062940491, -0.04566802794264308, 0.009852954298172652, -0.017402662476289188, -0.014453717485691053, 0.0255287053323465, -0.0636349836627169, 0.11239452745374738];
    let y_vec: Vec<f64> = vec![-0.4471285647859982, 1.2245077048054989, 0.4034916417908, 0.593578523237067, -1.0949118457410418, 0.1693824330586681, 0.7405564510962748, -0.9537006018079346, -0.26621850600362207, 0.03261454669335856, -1.3731173202467557, 0.31515939204229176, 0.8461606475850334, -0.8595159408319863, 0.35054597866410736, -1.3122834112374318, -0.038695509266051115, -1.6157723547032947, 1.121417708235664, 0.4089005379368278, -0.024616955875778355, -0.7751616191691596, 1.2737559301587766, 1.9671017492547347, -1.857981864446752, 1.2361640304528203, 1.6276507531489064, 0.3380116965744758, -1.199268032335186, 0.8633453175440216, -0.18092030207815046, -0.6039206277932573, -1.2300581356669618, 0.5505374959762154, 0.7928068659193477, -0.6235307296797916, 0.5205763370733708, -1.1443413896231427, 0.8018610318713447, 0.04656729842414554, -0.18656977190734877, -0.10174587252914521, 0.8688861570058679, 0.7504116398650081, 0.5294653243527092, 0.13770120999738608, 0.07782112791270591, 0.6183802619985245, 0.23249455917873788, 0.6825514068644851];
    (
        Array2::from_shape_vec((50, 2), x_flat).unwrap(),
        Array1::from_vec(y_vec),
    )
}

/// A moderately-conditioned (`cond == 1e6`), genuinely full-rank 30x3 design,
/// SVD-constructed with singular values `[1e3, 1.0, 1e-3]`. Here both the
/// coefficients AND the fitted values are numerically stable, so they ARE a
/// meaningful deterministic contract. Oracle: `scipy.linalg.lstsq(X, y)` /
/// `LinearRegression(fit_intercept=False)` — `rank == 3`.
#[rustfmt::skip]
fn well_conditioned_design() -> (Array2<f64>, Array1<f64>) {
    let x_flat: Vec<f64> = vec![211.20614841370246, 78.41130270149003, -312.2434526666515, -92.87318673289026, -34.31008439526096, 137.11520364919315, -45.84254743637187, -17.4118424165533, 68.20417398603108, 68.4603283782553, 25.57807490259558, -101.38840294727116, -48.66093230521012, -18.251971157057334, 72.14448778299914, 2.020375121068331, 0.40018194058032064, -2.6024681363428623, -59.80097295983826, -22.151812494235518, 88.35432585856941, -141.99429515745678, -52.839241646500014, 210.05743890946312, -16.36222822298427, -6.453038809139176, 24.605991229085724, -1.7866150503588727, -0.3916208628391409, 2.342948217177239, 41.20692044923831, 15.28193473438514, -60.90220587273226, -32.53496192958811, -12.04758606021432, 48.06461380062625, 78.14960248834637, 29.243690524353095, -115.78821623122995, 19.81156360608132, 7.51767376399146, -29.467989475446743, -82.86226608622746, -31.069342641123573, 122.83884003206963, -74.96085399430251, -28.055164675089866, 111.06866355423476, 133.7501637443228, 49.889876415656744, -197.99164442115296, -109.24627868531638, -40.95337675456653, 161.94156570241674, -63.34858364449959, -23.57138903101345, 93.71192585761008, -41.953251518686336, -15.44794722899852, 61.88298235369845, 3.062353456964556, 0.9151793565425164, -4.284337529063124, -29.67432292256012, -11.044810135842482, 43.90127552416691, -141.55604046719913, -52.82038856276858, 209.56797136234377, -80.26423066345383, -29.919656618394647, 118.79468359353044, -232.31857822275762, -86.44661808748344, 343.67175236229434, 265.2089734122867, 98.60583559493274, -392.24051835565695, 6.772891644529122, 2.3512136006727, -9.83311129159264, -15.951988679851118, -5.9574676902705255, 23.621670077063445, 69.29079672748718, 25.66532496365281, -102.37298938539206, -2.298876432526871, -0.9796941314051848, 3.537065894767793];
    let y_vec: Vec<f64> = vec![0.28295585957645847, 1.518797191287471, 1.0692786247821269, -0.16825826394636972, 0.8025375945506402, -0.3854920006420483, -0.05514806731326931, -1.2343649203237947, -0.315427971290001, 0.14056545721159885, -0.16904205798971872, 0.7699631232905594, 1.8836741015084058, 1.321637648302766, -0.3049527446832278, -0.7311830168750274, -0.6526596562005005, -0.9777989792054598, 0.03401336924214472, 0.4211186682176753, 0.18303950058853105, 0.19365000462550794, 0.15820019276916414, -0.13205410073674043, -0.742356236499713, 1.7246458962284568, 1.0475316571321267, -0.9082558525031752, -0.9052415680914347, -0.5862703257750248];
    (
        Array2::from_shape_vec((30, 3), x_flat).unwrap(),
        Array1::from_vec(y_vec),
    )
}

/// (a) RANK parity on the near-singular design: with the `eps * s_max` cutoff
/// ferrolearn keeps the SAME number of singular values as scipy. scipy's
/// `linalg.lstsq(X, y)` returns `rank == 2` (`sv == [1.0, 4.97e-15]`, the second
/// `> eps*s_max ~2.22e-16`). The observable signature of "rank 2 was kept" is the
/// coefficient magnitude: keeping the tiny `s_min` amplifies it to `~1e14`,
/// whereas the OLD `max(m,n)*eps` cutoff ZEROED it and returned a small-norm
/// (`~0.2`) solution. The individual coefficient values are NOT asserted — they
/// are `1/s_min`-amplified noise (inherent FP limit, #382).
#[test]
fn lstsq_rcond_eps_cutoff_keeps_rank_like_scipy() {
    let (x, y) = near_singular_design();
    let model = LinearRegression::<f64>::new().with_fit_intercept(false);
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let coef = fitted.coefficients();

    // scipy keeps s_min (rank 2): both coefficients are ~1e14 in magnitude.
    // The old max(m,n)*eps cutoff zeroed s_min (rank 1): coefficients ~0.2.
    // A threshold of 1e6 cleanly separates the two rank decisions.
    assert!(
        coef[0].abs() > 1e6 && coef[1].abs() > 1e6,
        "rank parity: with the eps cutoff ferrolearn must KEEP the tiny singular \
         value like scipy (rank 2 -> coef magnitude ~1e14); got coef={coef:?}. A \
         small-norm solution here would mean the tiny sv was zeroed (rank 1), the \
         pre-fix max(m,n)*eps behaviour.",
    );
}

/// (b) Well-conditioned COEFFICIENT parity and (c) stable FITTED-VALUE parity on a
/// `cond == 1e6`, full-rank design — the deterministic, meaningful regime. Oracle
/// (live scipy/sklearn 1.5.2):
///   python3 -c "import numpy as np; from scipy.linalg import lstsq; \
///     from sklearn.linear_model import LinearRegression; ...; \
///     print(lstsq(X, y)[0].tolist(), (X@lstsq(X,y)[0]).tolist())"
#[test]
fn lstsq_well_conditioned_coef_and_fitted_parity() {
    // (b) coef oracle: scipy.linalg.lstsq == sklearn LinearRegression(coef_).
    const SCIPY_COEF: [f64; 3] = [
        -128.554_885_012_787_3,
        -130.853_244_135_044_12,
        -119.817_856_656_339_23,
    ];
    // (c) fitted oracle: (X @ coef)[..4] and the full fitted-vector norm.
    const SCIPY_FITTED_HEAD: [f64; 4] = [
        0.385_794_831_159_724_57,
        0.037_874_641_093_239_26,
        -0.398_461_072_094_109_97,
        0.257_408_613_243_677_5,
    ];
    const SCIPY_FITTED_NORM: f64 = 1.688_609_900_639_871;

    let (x, y) = well_conditioned_design();
    let model = LinearRegression::<f64>::new().with_fit_intercept(false);
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let coef = fitted.coefficients();

    // (b) coefficient parity: tolerance scaled by eps * cond ~ 2.2e-10 (+ margin).
    for (i, &expected) in SCIPY_COEF.iter().enumerate() {
        assert!(
            (coef[i] - expected).abs() <= 1e-7,
            "well-conditioned coef parity: coef_[{i}] = {} != scipy/sklearn {expected}",
            coef[i],
        );
    }

    // (c) fitted-value parity: X @ coef is the stable projection onto col(X).
    let x_coef = x.dot(coef);
    for (i, &expected) in SCIPY_FITTED_HEAD.iter().enumerate() {
        assert!(
            (x_coef[i] - expected).abs() <= 1e-6,
            "fitted-value parity: (X @ coef)[{i}] = {} != scipy {expected}",
            x_coef[i],
        );
    }
    let fitted_norm = x_coef.dot(&x_coef).sqrt();
    assert!(
        (fitted_norm - SCIPY_FITTED_NORM).abs() <= 1e-6,
        "fitted-value parity: ||X @ coef|| = {fitted_norm} != scipy {SCIPY_FITTED_NORM}",
    );
}
