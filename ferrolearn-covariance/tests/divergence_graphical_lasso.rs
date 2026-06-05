//! Divergence tests pinning `ferrolearn-covariance` `GraphicalLasso` /
//! `graphical_lasso` against the live scikit-learn 1.5.2 oracle.
//!
//! These tests were authored adversarially (ACToR critic role). Every expected
//! value below was computed by running scikit-learn 1.5.2 on the EXACT probe
//! matrix `X` embedded in [`probe_x`] and copied from the sklearn side ONLY
//! (R-CHAR-3 — never from ferrolearn). Reproduce with:
//!
//! ```text
//! import numpy as np
//! from sklearn.covariance import GraphicalLasso, graphical_lasso, empirical_covariance
//! cov_true = np.array([[1,.5,.2,0],[.5,1,.3,.1],[.2,.3,1,.4],[0,.1,.4,1]], float)
//! X = np.random.RandomState(0).multivariate_normal(np.zeros(4), cov_true, size=60)
//! m = GraphicalLasso(alpha=0.1).fit(X)        # m.covariance_, m.precision_
//! emp = empirical_covariance(X)
//! graphical_lasso(emp, alpha=0.1)             # (covariance_, precision_)
//! ```
//!
//! The Rust `X` literals round-trip byte-for-byte to the saved `.npy` (verified
//! during authoring), so the embedded oracle values are exact for this `X`.

use ferrolearn_core::traits::Fit;
use ferrolearn_covariance::helpers::empirical_covariance;
use ferrolearn_covariance::{GraphicalLasso, graphical_lasso};
use ndarray::{Array2, array};

/// Deterministic probe: `RandomState(0).multivariate_normal(zeros(4),
/// cov_true, 60)` with `cov_true = [[1,.5,.2,0],[.5,1,.3,.1],[.2,.3,1,.4],
/// [0,.1,.4,1]]`. n=60, p=4. Both sklearn and ferrolearn converge at
/// `alpha=0.1` in `n_iter=2`, so any divergence is algorithmic, not
/// under-iteration. Floats are Python `repr` (shortest round-trip) of the saved
/// `numpy` array; they reproduce sklearn's oracle output bit-for-bit.
fn probe_x() -> Array2<f64> {
    array![
        [
            -2.1830553704479407,
            -0.11086008285715779,
            -1.1516767306754998,
            -1.6100940796555128
        ],
        [
            -1.9038426645549114,
            -1.9318940877901725,
            -0.37343998365844283,
            -0.5853997101463247
        ],
        [
            -0.3979189830091436,
            0.9727090137981843,
            -0.19097274193335237,
            -0.3448564690863293
        ],
        [
            -0.6838355049144865,
            -0.39370939701009944,
            -0.3914210047513665,
            -0.6590828206594799
        ],
        [
            -0.7921468216628673,
            -1.6843858989163767,
            -0.7077720666065835,
            -0.6793473507902001
        ],
        [
            2.271022873563251,
            1.7822914828472733,
            2.085336679540243,
            0.2985638200093048
        ],
        [
            -2.2555660114920157,
            -2.38880919739361,
            -0.9387384426852304,
            0.012734142789000612
        ],
        [
            -0.4412786425941927,
            -0.4494597856774002,
            -1.6857700192491945,
            -1.8900397844894263
        ],
        [
            -0.4845241151566568,
            0.04772002366332218,
            1.2834432206297,
            2.05744163401671
        ],
        [
            0.028758642008467797,
            -0.6363652141218192,
            -1.5716859821436981,
            -1.2848105344861969
        ],
        [
            -0.6434799887678783,
            1.376717080459487,
            0.21598295128587883,
            2.3337983111261393
        ],
        [
            -0.02017448288088149,
            0.6978440755457788,
            -0.2143167171645765,
            1.148173279836998
        ],
        [
            0.9657119917854676,
            1.412512140582219,
            0.71140833970777,
            1.3289103133960363
        ],
        [
            -0.48724099250135017,
            0.177091025353516,
            0.8110711011224059,
            1.1289459521304286
        ],
        [
            0.40115432353546165,
            -0.09133629775669619,
            -0.48091924533811414,
            0.04731809089473521
        ],
        [
            1.1986000273589326,
            -0.45965539252678755,
            0.3889964750630159,
            0.9873784965208019
        ],
        [
            -0.2540209559550808,
            0.04009472057780468,
            -0.887659300907291,
            0.9837948767888307
        ],
        [
            0.4522938783481495,
            0.7457375097628682,
            1.003317868357488,
            0.036644865505794696
        ],
        [
            -1.2060031424906505,
            -1.7112094163023621,
            0.019778980819213497,
            0.21159583394438844
        ],
        [
            0.3068283518153014,
            0.501946002191701,
            0.6951006630711352,
            0.9848156850367767
        ],
        [
            1.89963527048415,
            0.41716676404977937,
            0.8647787768237222,
            -0.3336594323179712
        ],
        [
            -0.1099009591029147,
            -0.5991257652527405,
            -1.2210712800923067,
            -2.666102398010718
        ],
        [
            0.8153118646324671,
            1.8679924304961761,
            -0.051775170036221445,
            -0.11469758884170309
        ],
        [
            0.0024050905746194505,
            0.5466483711269137,
            -0.4581329853903994,
            -1.0046157527720228
        ],
        [
            0.7586219047376872,
            0.855391895502662,
            -0.757807567349853,
            -1.4057144717700576
        ],
        [
            -2.224455126589831,
            -1.3849598644008416,
            -1.553805639384185,
            0.7216631398712848
        ],
        [
            2.2723723992447464,
            1.264093007906492,
            -0.14684980231759268,
            -0.7034904214871696
        ],
        [
            -1.2535950709950001,
            -0.5668671371846769,
            -1.1001479718828684,
            -2.896611311423122
        ],
        [
            1.3066974120394086,
            1.8009203262364988,
            -0.4507107538165783,
            -0.911093722111815
        ],
        [
            -1.2612430310392722,
            -0.3463762168387437,
            -0.3784321794260149,
            -0.6263783485639426
        ],
        [
            -1.5155963354363648,
            -0.03215310954825625,
            0.21990668137729671,
            0.49017050002038
        ],
        [
            -0.3768967680339198,
            1.4549567281222926,
            0.11393264885358906,
            0.6185925338881626
        ],
        [
            -0.3408402756618631,
            -0.047826515628582866,
            -1.1328766546480264,
            -0.2597424654478149
        ],
        [
            0.2814064753571032,
            0.9140649676605921,
            0.0449346793806417,
            0.5825615048082879
        ],
        [
            -0.07686303923961796,
            -1.1057310921883887,
            0.018262895048689342,
            -0.28657068146397197
        ],
        [
            0.930649167767102,
            1.6306074904880299,
            0.8889756991848579,
            0.27846183326186513
        ],
        [
            -1.447915467707915,
            -0.8621394364065517,
            -2.7253991469674417,
            -1.3966502113556891
        ],
        [
            -0.12973253274228286,
            1.728291479977065,
            0.9052286969292752,
            0.9690035149794681
        ],
        [
            0.3768523395907495,
            -0.06903091347223049,
            0.909562229996713,
            1.0192528722203211
        ],
        [
            -0.9436059693498735,
            -1.4330756552045596,
            -0.9057492407209308,
            0.8320082161133575
        ],
        [
            1.176069159714863,
            1.0945762369731953,
            0.02288430832466254,
            -1.6620183625008413
        ],
        [
            2.1807765894162054,
            0.5173915540880591,
            0.13899566870698868,
            0.4303363009832269
        ],
        [
            -0.9473679778498133,
            -0.6837596896033604,
            -0.5196800685165093,
            -1.2292963324312167
        ],
        [
            -1.0596492837100404,
            -0.4958034624081547,
            -0.9649428907022553,
            0.5695738454129177
        ],
        [
            0.2447200312352925,
            0.39673516674107767,
            0.6214950193694438,
            1.1048731661094628
        ],
        [
            0.6258934906984039,
            -1.3473549695525888,
            0.7503787305573552,
            1.524596970947637
        ],
        [
            -1.1197197901693139,
            -0.9907084647245097,
            -0.35498527161258775,
            1.4295181401639878
        ],
        [
            1.455872763848017,
            1.3505630026913167,
            -0.8728579653671347,
            -0.2017162066309073
        ],
        [
            -0.6264687822181014,
            -0.5201985006922745,
            0.8368187825275599,
            0.6465397650599597
        ],
        [
            -1.0809478500171106,
            0.2820109505736908,
            0.11484303110586759,
            -2.0147602524218176
        ],
        [
            -0.3799706505270462,
            0.47351039776596254,
            0.8878840502473628,
            -0.17744258829020493
        ],
        [
            -0.9731784626175332,
            -1.4726594589644124,
            0.31537831248214754,
            0.9322228382317297
        ],
        [
            -0.5544428649836787,
            -0.1381347244106719,
            0.3005412159987665,
            -0.4912925415913273
        ],
        [
            -0.42811662946592816,
            -0.9394802711635557,
            -0.8225744844863685,
            0.43496936472523984
        ],
        [
            -0.17421604458023457,
            0.010293746440704736,
            1.0765616205172635,
            -1.344238978320111
        ],
        [
            0.32457192249943334,
            0.8811723520600777,
            0.524306448467055,
            0.9170323005494528
        ],
        [
            0.9455649999365787,
            1.322300824311515,
            1.1272295158268415,
            0.5925467075262294
        ],
        [
            0.7610162738676401,
            0.19506293897172222,
            -0.18762110132533064,
            1.1356135379767613
        ],
        [
            0.44832760914612385,
            1.0374548584393457,
            -0.5087609092712599,
            0.47893247952599094
        ],
        [
            -2.529361527616984,
            -1.5957541395468582,
            -0.012198505780888755,
            0.6434016505882416
        ],
    ]
}

/// sklearn 1.5.2 `GraphicalLasso(alpha=0.1).fit(probe_x()).covariance_`.
/// Computed live; copied from the sklearn side only (R-CHAR-3).
fn sklearn_covariance_alpha_0_1() -> Array2<f64> {
    array![
        [
            1.230591687717977,
            0.693688652920049,
            0.301931948369093,
            0.143553743944686
        ],
        [
            0.693688652920049,
            1.117113918447415,
            0.269651270204874,
            0.12820587422577
        ],
        [
            0.301931948369093,
            0.269651270204874,
            0.752888780813091,
            0.357961467289138
        ],
        [
            0.143553743944686,
            0.12820587422577,
            0.357961467289138,
            1.234160733810159
        ],
    ]
}

/// sklearn 1.5.2 `GraphicalLasso(alpha=0.1).fit(probe_x()).precision_`.
/// Computed live; copied from the sklearn side only (R-CHAR-3).
fn sklearn_precision_alpha_0_1() -> Array2<f64> {
    array![
        [
            1.292753985237026,
            -0.741742330948294,
            -0.252779734337547,
            0.0
        ],
        [
            -0.741742330948294,
            1.405463703394481,
            -0.205910873460996,
            0.0
        ],
        [
            -0.252779734337547,
            -0.205910873460996,
            1.715798015625492,
            -0.446865681434971
        ],
        [0.0, 0.0, -0.446865681434971, 0.939878140043022],
    ]
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Divergence: ferrolearn `GraphicalLasso::fit` covariance_/precision_ diverge
/// from `sklearn/covariance/_graph_lasso.py:70` (`_graphical_lasso`) for the
/// deterministic 60x4 probe at `alpha=0.1`.
///
/// sklearn `precision_[0]` = `[1.2928, -0.7417, -0.2528, 0]` (line:163-168 the
/// Friedman precision update fed by `cd_fast.enet_coordinate_descent_gram`
/// :139-150, init `covariance_ *= 0.95` + restored diagonal + `pinvh` seed
/// :101-104, convergence `|_dual_gap| < tol` :184).
/// ferrolearn `precision()[0]` = `[1.1019, -0.5824, -0.2060, 0]` — its
/// `solve_glasso`/`coord_descent_lasso` (graphical_lasso.rs) uses a different
/// inner-lasso penalty scaling, a Frobenius `||W-W_old||_F < tol` stop, and a
/// `W = S + alpha*I` init.
///
/// Measured (authoring): covariance_ max-abs-diff ~1.68e-2, precision_
/// max-abs-diff ~2.68e-1; both grow with alpha.
///
/// FIXER NOTE (coupled — ONE fixer for the whole port; #1880 is the contract,
/// fixing any sub-part alone will NOT green this):
///   * init: `covariance_ *= 0.95`, restore empirical diagonal, seed
///     `precision_ = pinvh(covariance_)`  (`_graph_lasso.py:101-104`) — REQ-X-2 init.
///   * inner lasso: replace `coord_descent_lasso` with the Gram elastic-net
///     `enet_coordinate_descent_gram(coefs, alpha, 0, sub_cov, row, row,
///     max_iter, enet_tol, ...)` warm-started from `-precision_[!=idx,idx]/
///     (precision_[idx,idx]+1000*eps)` (`:135-150`) — REQ-ENET-TOL #1883.
///   * convergence: break when `|_dual_gap(emp_cov, precision_, alpha)| < tol`,
///     `d_gap = sum(emp_cov*precision_) - p + alpha*(|precision_|.sum() -
///     |diag(precision_)|.sum())` (`:57-66`, `:176-185`) — REQ-CONVERGENCE #1881.
///
/// Tracking: #1880 (covers REQ-GLASSO-VALUE; root causes REQ-CONVERGENCE #1881,
/// REQ-ENET-TOL #1883).
#[test]
#[ignore = "divergence: GraphicalLasso covariance_/precision_ differ from sklearn _graphical_lasso; tracking #1880"]
fn divergence_graphical_lasso_value_alpha_0_1() {
    let x = probe_x();
    let fitted = GraphicalLasso::<f64>::new(0.1).fit(&x, &()).unwrap();

    let exp_cov = sklearn_covariance_alpha_0_1();
    let exp_prec = sklearn_precision_alpha_0_1();

    let cov_diff = max_abs_diff(fitted.covariance(), &exp_cov);
    let prec_diff = max_abs_diff(fitted.precision(), &exp_prec);

    // sklearn's documented convergence tolerance is 1e-4; pin to a tight 1e-4.
    assert!(
        cov_diff < 1e-4,
        "covariance_ diverges from sklearn: max-abs-diff = {cov_diff:e} (>= 1e-4)\n\
         ferrolearn = {:?}\nsklearn = {:?}",
        fitted.covariance(),
        exp_cov,
    );
    assert!(
        prec_diff < 1e-4,
        "precision_ diverges from sklearn: max-abs-diff = {prec_diff:e} (>= 1e-4)\n\
         ferrolearn = {:?}\nsklearn = {:?}",
        fitted.precision(),
        exp_prec,
    );
}

/// Divergence: the free function `graphical_lasso(emp_cov, alpha, max_iter, tol)`
/// (graphical_lasso.rs) wraps the same `solve_glasso` and therefore diverges
/// identically from `sklearn.covariance.graphical_lasso(emp_cov, alpha=0.1)`
/// (`sklearn/covariance/_graph_lasso.py:230`, which delegates to
/// `_graphical_lasso` :70).
///
/// sklearn `graphical_lasso(emp, 0.1)` returns the SAME `(covariance_,
/// precision_)` as `GraphicalLasso(alpha=0.1).fit(X)` (verified live), so the
/// expected arrays are the oracle matrices above. ferrolearn's function output
/// equals its `GraphicalLasso::fit` output (shared solver) and diverges.
///
/// Tracking: #1880 (REQ-GLASSO-FN; same coupled fix as the class path).
#[test]
#[ignore = "divergence: graphical_lasso() fn differs from sklearn.covariance.graphical_lasso; tracking #1880"]
fn divergence_graphical_lasso_function_alpha_0_1() {
    let x = probe_x();
    // ferrolearn's `empirical_covariance` is sklearn's empirical_covariance
    // (1/N normalisation), the same input sklearn's graphical_lasso receives.
    let emp = empirical_covariance(&x, false).unwrap();
    let (cov, prec) = graphical_lasso(&emp, 0.1, 100, 1e-4).unwrap();

    let exp_cov = sklearn_covariance_alpha_0_1();
    let exp_prec = sklearn_precision_alpha_0_1();

    let cov_diff = max_abs_diff(&cov, &exp_cov);
    let prec_diff = max_abs_diff(&prec, &exp_prec);

    assert!(
        cov_diff < 1e-4,
        "graphical_lasso covariance diverges: max-abs-diff = {cov_diff:e} (>= 1e-4)",
    );
    assert!(
        prec_diff < 1e-4,
        "graphical_lasso precision diverges: max-abs-diff = {prec_diff:e} (>= 1e-4)",
    );
}

/// Divergence (alpha sweep): the precision diagonal divergence GROWS with alpha
/// (`sklearn/covariance/_graph_lasso.py:163-168`). At `alpha=0.2` sklearn
/// `precision_[0,0]` = 1.113712683465427 (computed live); ferrolearn ~0.873.
/// Same coupled root cause as #1880 — pins that the gap is not a fixed offset.
///
/// Tracking: #1880.
#[test]
#[ignore = "divergence: precision diagonal diverges and grows with alpha; tracking #1880"]
fn divergence_graphical_lasso_precision_diag_alpha_0_2() {
    let x = probe_x();
    let fitted = GraphicalLasso::<f64>::new(0.2).fit(&x, &()).unwrap();
    // sklearn 1.5.2 GraphicalLasso(alpha=0.2).fit(probe_x()).precision_[0,0]:
    const SKLEARN_PREC_00: f64 = 1.113712683465427;
    let got = fitted.precision()[[0, 0]];
    assert!(
        (got - SKLEARN_PREC_00).abs() < 1e-4,
        "precision_[0,0] at alpha=0.2 diverges: ferrolearn {got} vs sklearn {SKLEARN_PREC_00}",
    );
}

/// GREEN GUARD (currently-faithful facet): sklearn 1.5.2 zeroes the same
/// off-diagonal precision entries on this probe — `precision_[0,3]`,
/// `precision_[3,0]`, `precision_[1,3]`, `precision_[3,1]` are exactly 0
/// (the lasso soft-thresholds those coefficients to zero;
/// `sklearn/covariance/_graph_lasso.py:167-168`). The doc claims ferrolearn's
/// sparsity pattern matches even though magnitudes diverge; this guard pins
/// that the structural zeros agree. If the fixer's port changes magnitudes it
/// MUST preserve this zero structure.
///
/// NOT ignored: this is the one facet measured as faithful. If it ever fails,
/// the sparsity pattern has regressed.
#[test]
fn guard_graphical_lasso_sparsity_pattern_alpha_0_1() {
    let x = probe_x();
    let fitted = GraphicalLasso::<f64>::new(0.1).fit(&x, &()).unwrap();
    let p = fitted.precision();
    // sklearn zero mask (computed live): off-diagonal (0,3),(3,0),(1,3),(3,1)==0.
    for &(i, j) in &[(0usize, 3usize), (3, 0), (1, 3), (3, 1)] {
        assert!(
            p[[i, j]].abs() <= 1e-12,
            "expected sklearn zero at precision[{i},{j}], got {}",
            p[[i, j]],
        );
    }
    // And the entries sklearn keeps non-zero must be non-zero in ferrolearn too.
    for &(i, j) in &[(0usize, 1usize), (0, 2), (1, 2), (2, 3)] {
        assert!(
            p[[i, j]].abs() > 1e-6,
            "expected non-zero at precision[{i},{j}], got {}",
            p[[i, j]],
        );
    }
}
