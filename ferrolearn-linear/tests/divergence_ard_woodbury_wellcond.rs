//! Divergence pin: well-conditioned `n_samples < n_features` ARD coef_ envelope.
//!
//! The `divergence_ard_woodbury.rs` suite was re-characterised (#2164) from an
//! exact-parity pin to an "observable contract": its only coef-pinned n<p case
//! (6x25, RandomState(23)) lands at max|coef diff| ~= 8.06e-5 against the live
//! sklearn 1.5.2 oracle and is asserted with a 5e-4 tolerance. That suggests
//! well-conditioned n<p matches sklearn to ~1e-4.
//!
//! This pin records the TRUE well-conditioned n<p envelope: a 10x15 design
//! (RandomState(11)) with a clean sparse signal on features {0,3,7,12} that
//! sklearn fits in n_iter_ = 7 (NOT chaotic). The kept SET matches sklearn, but
//! ferrolearn's coef_ diverges from the live oracle by ~6.9e-4 — well beyond the
//! suite's own 5e-4 contract bound and the implied 1e-4 "tight" tolerance.
//!
//! ROOT CAUSE (substrate, R-SUBSTRATE-5): the divergence traces to the
//! scipy-`pinvh` vs. ferray-LU `inv` backend. A faithful numpy-LU-`inv` ARD
//! reproduction matches ferrolearn (coef[3]: ferro 1.996500 vs numpy-LU 1.996586)
//! and BOTH differ from sklearn-pinvh (1.995814) by ~7e-4: the LU trajectory
//! splits from pinvh (8 vs 7 iters) and the final refresh widens the gap. This is
//! the SAME ferray `pinvh` gap that blocks the chaotic 5x8 tail.
//!
//! Expected values: LIVE sklearn 1.5.2 oracle (R-CHAR-3), NOT copied from
//! ferrolearn.
//!
//! Tracking: #2166 (this divergence) / substrate-blocked on #2165 (ferray
//! scipy.pinvh primitive). This pin documents that the
//! well-conditioned n<p coef envelope (~7e-4) is wider than the re-characterised
//! suite's single 8e-5 case implies; it is substrate-blocked, not a builder bug.

use ferrolearn_core::introspection::HasCoefficients;
use ferrolearn_core::traits::Fit;
use ferrolearn_linear::ard::ARDRegression;
use ndarray::{Array2, array};

/// Well-conditioned n<p (10 samples, 15 features), sklearn n_iter_ = 7. The
/// kept set {0,3,7,12} matches; ferrolearn's coef_ must match the live sklearn
/// 1.5.2 oracle to <= 1e-4 (the "tight" tolerance for a NON-chaotic n<p design).
/// FAILS: max|coef diff| ~= 6.9e-4, traceable to scipy-pinvh vs ferray-LU (#2165).
#[test]
#[ignore = "divergence: well-conditioned n<p coef envelope ~7e-4 (scipy-pinvh vs ferray-LU); tracking #2166, substrate #2165"]
fn divergence_ard_woodbury_well_conditioned_coef_envelope() {
    // Live sklearn 1.5.2 oracle: rng=RandomState(11); X=rng.randn(10,15);
    // y=X[:,0]+2*X[:,3]-X[:,7]+0.5*X[:,12]+0.01*rng.randn(10).
    let sk_coef: [f64; 15] = [
        1.0012031591123924,
        0.0,
        0.0,
        1.9958142764948812,
        0.0,
        0.0,
        0.0,
        -1.0053554849480593,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5023684764235373,
        0.0,
        0.0,
    ];

    let x = Array2::from_shape_vec(
        (10, 15),
        vec![
            1.7494547413051793,
            -0.28607299681629417,
            -0.4845651322211443,
            -2.653318559261479,
            -0.00828462937293584,
            -0.31963136376429824,
            -0.53662936223473,
            0.3154026684256969,
            0.42105071625547846,
            -1.0656029804488336,
            -0.886239669955473,
            -0.4757334926833952,
            0.6896823102014212,
            0.5611921760332811,
            -1.3055485097805806,
            -1.1194752615996189,
            0.7368373912295207,
            1.5746340731042836,
            -0.03107508846786004,
            -0.6834466274652374,
            1.095629698519445,
            -0.309576637220662,
            0.7257522247990231,
            1.5490716333717869,
            0.6300798224934151,
            0.07349323700188944,
            0.7322713538852552,
            -0.6425753924333545,
            -0.17809317531243007,
            -0.5739545594102069,
            -0.20437531566217204,
            -0.4864951001088207,
            -0.18577532395900828,
            -0.38053641675867705,
            0.0889776353222905,
            0.06367166142477786,
            0.29634710809428044,
            1.4027711194838541,
            -1.5468625676023853,
            1.2956185848026505,
            -0.2372504457468327,
            -1.2323462066022715,
            -0.1724197671774095,
            0.09183837492938327,
            1.0675584570477854,
            -1.0616344482718332,
            0.21734820797221702,
            0.11781950435342614,
            -1.68411089296333,
            -1.1857552653505647,
            0.6001020056275471,
            0.6955672553748646,
            1.0877108600968604,
            0.5338217156453108,
            0.3952120129351584,
            0.12286752936260395,
            1.2091016420933416,
            -0.8430661010803197,
            -0.14189358101621208,
            0.3853541380984869,
            -1.5774943085916198,
            1.3109436402150754,
            -0.7928650077765562,
            -0.07174941217099023,
            2.156674425806779,
            -0.8294372522768853,
            -0.5293720276479461,
            1.5617036936504114,
            -1.0801938251972851,
            -0.4320576187374964,
            0.51608403733497,
            0.45539286028580667,
            0.5936862029814544,
            0.370506334286746,
            1.3453780718487536,
            1.0159421463860647,
            0.5952784540779213,
            -0.6828016163681212,
            -0.7135599294040993,
            -1.9082895416901746,
            0.6873123485844095,
            -1.8230095835236662,
            0.8791137997169072,
            1.8463648729317061,
            -1.0616544037884463,
            -0.6844846684032199,
            -0.4762144798666495,
            0.8303104295602717,
            -0.8633525162339086,
            -0.13054146707773523,
            -0.5230876326157433,
            -0.25127692435468146,
            1.2912747284232993,
            -0.9642048531813225,
            0.07175977417228958,
            0.27160629615640114,
            0.8586671738730903,
            -1.2640736790434945,
            1.1148702849563317,
            0.43477698670760984,
            0.8742727658458852,
            0.07165210025884777,
            -1.6390516267392266,
            -0.6473026312724075,
            0.8177695662661595,
            0.03679475399862955,
            -0.04870254236748755,
            1.7915310965993816,
            2.2018563138041243,
            -0.03706690441882626,
            1.9329054284744887,
            -1.9935715313257563,
            -2.0490572604691897,
            0.8670552083195185,
            -0.26196107320485784,
            0.5789711070908105,
            0.5165523855469867,
            -0.08940364049732401,
            0.6821297104769012,
            0.1507220091744646,
            -1.5321425223126923,
            0.21886857740027746,
            0.1856632544475102,
            1.832776536801211,
            -1.1288944013131346,
            0.01699688084106283,
            -0.42442882494572987,
            -0.13290989911115147,
            1.6592046176964066,
            0.3786406842289673,
            -0.4645660624836016,
            0.15384124503486238,
            0.7661606204038456,
            -0.9940276932279378,
            -0.26434232875389496,
            1.5422092203408082,
            0.856152045689601,
            -0.04480261697554768,
            -0.47748922897960105,
            -0.1540655244375879,
            -1.551079461180946,
            0.9430359810936154,
            0.34115343851036384,
            0.13827321731793896,
            -1.2974226175245323,
            -0.7149624441521972,
            0.5144796332603411,
            0.2577163810737392,
            -0.4967304772912818,
            0.5050219167601375,
        ],
    )
    .unwrap();
    let y = array![
        -3.518356059290559,
        -2.2354155622978307,
        -2.4597377346531175,
        -5.947798764270625,
        -2.9974063673414513,
        -0.8671695788896187,
        -2.001849193086151,
        6.448249961192287,
        2.6376795602574434,
        0.36447583786077403
    ];

    let fitted = ARDRegression::<f64>::new().fit(&x, &y).unwrap();
    let coef = fitted.coefficients();

    // Kept set matches (documented; this part holds).
    let fl_kept: Vec<usize> = (0..15).filter(|&i| coef[i] != 0.0).collect();
    let sk_kept: Vec<usize> = (0..15).filter(|&i| sk_coef[i] != 0.0).collect();
    assert_eq!(fl_kept, sk_kept, "kept set must match sklearn");

    // Tight tolerance for a WELL-CONDITIONED (n_iter_=7, non-chaotic) n<p design.
    // sklearn (scipy pinvh) vs ferrolearn (ferray LU inv) differ by ~6.9e-4 here.
    let max_diff = (0..15)
        .map(|i| (coef[i] - sk_coef[i]).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff <= 1e-4,
        "well-conditioned n<p coef diverges from sklearn beyond tight tol:          max|diff|={max_diff:.3e} (> 1e-4); scipy-pinvh vs ferray-LU, #2165",
    );
}
