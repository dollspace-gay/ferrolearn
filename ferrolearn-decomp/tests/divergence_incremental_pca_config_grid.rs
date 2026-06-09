//! Config-grid re-audit of `ferrolearn_decomp::IncrementalPCA` vs scikit-learn
//! 1.5.2 `IncrementalPCA(n_components=K, batch_size=B).fit(X)`
//! (`sklearn/decomposition/_incremental_pca.py:241-367`,
//! `sklearn/utils/_chunking.py:67-75` gen_batches), re-auditing the #2386 fix
//! (hand-rolled Jacobi `thin_svd` replaced with `ferray::linalg::svd_lapack`
//! (LAPACK gesdd) + gen_batches min_batch_size merge in `fit`).
//!
//! Each config exercises a DIFFERENT gen_batches remainder/merge pattern:
//!   cfg0 (20,5,3,6)  -> batches 6+6+8  (trailing remainder 2 < K=3 merges)
//!   cfg1 (20,5,2,7)  -> batches 7+7+6  (remainder 6 >= K=2, no merge)
//!   cfg2 (17,4,3,5)  -> batches 5+5+7  (remainder 2 < K=3 merges)
//!   cfg3 (30,6,4,10) -> batches 10+10+10 (even, no merge)
//!   cfg4 (13,3,2,4)  -> batches 4+4+5  (remainder 1 < K=2 merges)
//!   cfg5 (25,5,3,6)  -> batches 6+6+6+7 (remainder 1 < K=3 merges into last full)
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle, run from `/tmp`
//! (R-CHAR-3), on `X = RandomState(0).randn(n, p)`. NEVER copied from ferrolearn.
//! sklearn applies `svd_flip(u_based_decision=False)` (`:357`) so component-row
//! max-abs is positive — ferrolearn matches that convention, so sign-aware
//! compares are valid (target #4). Tolerance ~1e-6 (R-DEV-1).
//!
//! Tracking re-audit: #2386.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::IncrementalPCA;
use ndarray::Array2;

#[allow(
    clippy::too_many_arguments,
    reason = "oracle-config check helper: n/p/k/b dims + 5 expected arrays per fixture"
)]
fn check(
    name: &str,
    n: usize,
    p: usize,
    k: usize,
    b: usize,
    x_flat: &[f64],
    comp: &[f64],
    sv: &[f64],
    ev: &[f64],
    evr: &[f64],
    tr2: &[f64],
) {
    let x = Array2::from_shape_vec((n, p), x_flat.to_vec()).expect("X shape");
    let f = IncrementalPCA::<f64>::new(k)
        .with_batch_size(b)
        .fit(&x, &())
        .unwrap_or_else(|e| panic!("{name}: fit failed: {e:?}"));

    // components_ (sign-aware, svd_flip u_based_decision=False)
    let fc = f.components();
    assert_eq!(fc.dim(), (k, p), "{name}: components shape");
    for kk in 0..k {
        for j in 0..p {
            let got = fc[[kk, j]];
            let want = comp[kk * p + j];
            let diff = (got - want).abs();
            assert!(
                diff < 1e-6,
                "{name}: components_[{kk}][{j}] = {got} but sklearn = {want} (diff {diff})"
            );
        }
    }
    // singular_values_
    let fsv = f.singular_values();
    for kk in 0..k {
        let diff = (fsv[kk] - sv[kk]).abs();
        assert!(
            diff < 1e-6,
            "{name}: singular_values_[{kk}] = {} but sklearn = {} (diff {diff})",
            fsv[kk],
            sv[kk]
        );
    }
    // explained_variance_
    let fev = f.explained_variance();
    for kk in 0..k {
        let diff = (fev[kk] - ev[kk]).abs();
        assert!(
            diff < 1e-6,
            "{name}: explained_variance_[{kk}] = {} but sklearn = {} (diff {diff})",
            fev[kk],
            ev[kk]
        );
    }
    // explained_variance_ratio_
    let fevr = f.explained_variance_ratio();
    for kk in 0..k {
        let diff = (fevr[kk] - evr[kk]).abs();
        assert!(
            diff < 1e-6,
            "{name}: explained_variance_ratio_[{kk}] = {} but sklearn = {} (diff {diff})",
            fevr[kk],
            evr[kk]
        );
    }
    // transform(X[:2])
    let x2 = x.slice(ndarray::s![0..2, ..]).to_owned();
    let t = f
        .transform(&x2)
        .unwrap_or_else(|e| panic!("{name}: transform: {e:?}"));
    for r in 0..2 {
        for kk in 0..k {
            let got = t[[r, kk]];
            let want = tr2[r * k + kk];
            let diff = (got - want).abs();
            assert!(
                diff < 1e-6,
                "{name}: transform[{r}][{kk}] = {got} but sklearn = {want} (diff {diff})"
            );
        }
    }
}

#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const X0: [f64; 100] = [
    1.764052345968,
    0.400157208367,
    0.978737984106,
    2.240893199201,
    1.86755799015,
    -0.977277879876,
    0.950088417526,
    -0.151357208298,
    -0.103218851794,
    0.410598501938,
    0.144043571161,
    1.454273506963,
    0.761037725147,
    0.121675016493,
    0.443863232745,
    0.333674327374,
    1.494079073158,
    -0.205158263766,
    0.313067701651,
    -0.854095739302,
    -2.552989815834,
    0.65361859544,
    0.86443619886,
    -0.742165020406,
    2.269754623988,
    -1.454365674599,
    0.045758517301,
    -0.187183850026,
    1.532779214358,
    1.4693587699,
    0.154947425697,
    0.378162519602,
    -0.88778574763,
    -1.980796468224,
    -0.347912149326,
    0.156348969104,
    1.230290680728,
    1.202379848784,
    -0.387326817408,
    -0.302302750575,
    -1.048552965067,
    -1.420017937179,
    -1.706270190625,
    1.950775395232,
    -0.509652181752,
    -0.438074301611,
    -1.25279536005,
    0.777490355832,
    -1.613897847558,
    -0.212740280214,
    -0.895466561194,
    0.386902497859,
    -0.510805137569,
    -1.180632184122,
    -0.028182228339,
    0.42833187053,
    0.066517222383,
    0.30247189774,
    -0.634322093681,
    -0.362741165987,
    -0.672460447776,
    -0.359553161541,
    -0.813146282044,
    -1.726282602332,
    0.177426142254,
    -0.401780936208,
    -1.630198346966,
    0.462782255526,
    -0.907298364383,
    0.051945395796,
    0.729090562178,
    0.128982910757,
    1.139400684543,
    -1.234825820354,
    0.402341641178,
    -0.68481009094,
    -0.870797149182,
    -0.578849664764,
    -0.311552532127,
    0.05616534223,
    -1.165149840783,
    0.900826486954,
    0.46566243973,
    -1.536243686277,
    1.488252193796,
    1.895889176031,
    1.17877957116,
    -0.179924835812,
    -1.070752621511,
    1.054451726931,
    -0.403176946973,
    1.222445070382,
    0.208274978077,
    0.976639036484,
    0.356366397174,
    0.706573168192,
    0.010500020721,
    1.785870493906,
    0.126912092704,
    0.401989363445,
];

#[test]
fn cfg0_20_5_3_6_merge_6_6_8() {
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.378702224658,
        0.261289371371,
        0.209950719843,
        0.829048695623,
        0.238561197539,
        0.601364022411,
        0.453011647525,
        0.40740037114,
        -0.51669996915,
        -0.01370544535,
        -0.565953676034,
        0.316998021295,
        0.35123441504,
        -0.121443430862,
        0.664149508673,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [5.479984302786, 5.078676476423, 4.389804029254];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [1.58053831362, 1.357523934325, 1.01423049554];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.31918880528, 0.274151179381, 0.204823266453];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        3.422541787602,
        0.2467046350904,
        -0.1252910059508,
        -0.0001362251889413,
        -0.3819377909271,
        0.5206117754415,
    ];
    check("cfg0", 20, 5, 3, 6, &X0, &comp, &sv, &ev, &evr, &tr2);
}

#[test]
fn cfg1_20_5_2_7_no_merge_7_7_6() {
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.376112964529,
        0.257111889669,
        0.184593733049,
        0.837904493469,
        0.23722084151,
        0.804327819329,
        0.26477361698,
        0.134697977695,
        -0.371003817148,
        -0.356601056185,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [5.469731567547, 4.794265971758];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [1.574629653738, 1.209736116208];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.317995554808, 0.244305514336];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        3.417278932147,
        0.269886222984,
        0.009251279697,
        -0.552428390255,
    ];
    check("cfg1", 20, 5, 2, 7, &X0, &comp, &sv, &ev, &evr, &tr2);
}

#[test]
fn cfg2_17_4_3_5_merge_5_5_7() {
    let x: Vec<f64> = X0[..17 * 4].to_vec();
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.92951827134,
        0.199092314331,
        0.308244285705,
        0.036653703082,
        0.136938039051,
        -0.475042525927,
        -0.206517318194,
        0.844353699174,
        -0.242004133192,
        0.801083149541,
        0.149741559626,
        0.526571222496,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [5.727710997617, 4.087073359729, 3.706478249772];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [2.050417079514, 1.044010540488, 0.858623813502];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.4586703085, 0.233541088526, 0.192070800307];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        2.032170973152,
        1.344489696695,
        1.360496665531,
        1.757628719725,
        -0.0009850490042663,
        -1.031972419323,
    ];
    check("cfg2", 17, 4, 3, 5, &x, &comp, &sv, &ev, &evr, &tr2);
}

#[allow(
    clippy::excessive_precision,
    reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
)]
const X3: [f64; 180] = [
    1.764052345968,
    0.400157208367,
    0.978737984106,
    2.240893199201,
    1.86755799015,
    -0.977277879876,
    0.950088417526,
    -0.151357208298,
    -0.103218851794,
    0.410598501938,
    0.144043571161,
    1.454273506963,
    0.761037725147,
    0.121675016493,
    0.443863232745,
    0.333674327374,
    1.494079073158,
    -0.205158263766,
    0.313067701651,
    -0.854095739302,
    -2.552989815834,
    0.65361859544,
    0.86443619886,
    -0.742165020406,
    2.269754623988,
    -1.454365674599,
    0.045758517301,
    -0.187183850026,
    1.532779214358,
    1.4693587699,
    0.154947425697,
    0.378162519602,
    -0.88778574763,
    -1.980796468224,
    -0.347912149326,
    0.156348969104,
    1.230290680728,
    1.202379848784,
    -0.387326817408,
    -0.302302750575,
    -1.048552965067,
    -1.420017937179,
    -1.706270190625,
    1.950775395232,
    -0.509652181752,
    -0.438074301611,
    -1.25279536005,
    0.777490355832,
    -1.613897847558,
    -0.212740280214,
    -0.895466561194,
    0.386902497859,
    -0.510805137569,
    -1.180632184122,
    -0.028182228339,
    0.42833187053,
    0.066517222383,
    0.30247189774,
    -0.634322093681,
    -0.362741165987,
    -0.672460447776,
    -0.359553161541,
    -0.813146282044,
    -1.726282602332,
    0.177426142254,
    -0.401780936208,
    -1.630198346966,
    0.462782255526,
    -0.907298364383,
    0.051945395796,
    0.729090562178,
    0.128982910757,
    1.139400684543,
    -1.234825820354,
    0.402341641178,
    -0.68481009094,
    -0.870797149182,
    -0.578849664764,
    -0.311552532127,
    0.05616534223,
    -1.165149840783,
    0.900826486954,
    0.46566243973,
    -1.536243686277,
    1.488252193796,
    1.895889176031,
    1.17877957116,
    -0.179924835812,
    -1.070752621511,
    1.054451726931,
    -0.403176946973,
    1.222445070382,
    0.208274978077,
    0.976639036484,
    0.356366397174,
    0.706573168192,
    0.010500020721,
    1.785870493906,
    0.126912092704,
    0.401989363445,
    1.883150697056,
    -1.347759061142,
    -1.270484998486,
    0.969396708158,
    -1.173123405114,
    1.943621185649,
    -0.41361898076,
    -0.747454811441,
    1.92294202648,
    1.480514791434,
    1.867558960427,
    0.906044658275,
    -0.861225685055,
    1.910064953099,
    -0.268003370951,
    0.802456395796,
    0.947251967774,
    -0.155010093091,
    0.614079370346,
    0.922206671567,
    0.376425531156,
    -1.099400790584,
    0.298238174206,
    1.326385896687,
    -0.694567859731,
    -0.149634540328,
    -0.435153551722,
    1.849263728479,
    0.672294757012,
    0.407461836241,
    -0.769916074445,
    0.539249191292,
    -0.674332660657,
    0.031830558274,
    -0.635846078379,
    0.676433294946,
    0.576590816615,
    -0.208298755578,
    0.396006712662,
    -1.093061508731,
    -1.491257592706,
    0.439391701265,
    0.166673495373,
    0.635031436892,
    2.383144774864,
    0.94447948699,
    -0.912822225444,
    1.117016288096,
    -1.315907410512,
    -0.461584604815,
    -0.068241605325,
    1.713342721649,
    -0.744754822048,
    -0.826438538659,
    -0.098452524425,
    -0.663478286362,
    1.126635922107,
    -1.079931508363,
    -1.147468652411,
    -0.437820044744,
    -0.498032450692,
    1.929532053817,
    0.949420806926,
    0.087551241385,
    -1.22543551883,
    0.844362976402,
    -1.00021534739,
    -1.544771096778,
    1.188029792352,
    0.316942611925,
    0.920858823781,
    0.318727652943,
    0.856830611903,
    -0.6510255933,
    -1.034242841784,
    0.681594518282,
    -0.803409664174,
    -0.68954977775,
    -0.455532503517,
    0.017479159025,
];

#[test]
fn cfg3_30_6_4_10_even_no_merge() {
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.735011695392,
        -0.00845900731,
        0.531329974495,
        0.106835709352,
        0.066518838431,
        0.401915521443,
        -0.314781355613,
        0.70572758776,
        0.33159146275,
        -0.164427003219,
        -0.440083708941,
        0.268697714172,
        0.017357395598,
        0.409274838088,
        0.232669469808,
        0.650032773426,
        0.244245362959,
        -0.543929537431,
        -0.500686280086,
        -0.06556514038,
        0.179270155498,
        0.188077238278,
        0.636464820573,
        0.52193501249,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [
        7.112885603119,
        6.281199257269,
        5.758280579785,
        4.70050365535,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [
        1.744591089761,
        1.360464279638,
        1.143372249501,
        0.761887400481,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [
        0.29150574301,
        0.227321550019,
        0.191047391611,
        0.127304646957,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        1.723947995414,
        -1.437878227342,
        2.593681549872,
        0.562288524731,
        1.222555388922,
        -0.216851357729,
        -0.831210605148,
        0.639945382251,
    ];
    check("cfg3", 30, 6, 4, 10, &X3, &comp, &sv, &ev, &evr, &tr2);
}

#[test]
fn cfg4_13_3_2_4_merge_4_4_5() {
    let x: Vec<f64> = X0[..13 * 3].to_vec();
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        -0.325318533098,
        0.259553678338,
        0.909285290811,
        0.843455311895,
        0.514370122202,
        0.15494035698,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [3.913117813958, 3.785633781957];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [1.276040918827, 1.194251927591];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.375248083955, 0.351196220259];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        0.630691362015,
        1.076954308322,
        -0.922140988084,
        1.930869582133,
    ];
    check("cfg4", 13, 3, 2, 4, &x, &comp, &sv, &ev, &evr, &tr2);
}

#[test]
fn cfg5_25_5_3_6_merge_6_6_6_7() {
    let x: Vec<f64> = X3[..25 * 5].to_vec();
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.585982524627,
        -0.077863029629,
        -0.182718558519,
        0.779540674301,
        -0.097427382864,
        0.161801115671,
        0.641126248818,
        0.481962598733,
        0.125496151964,
        0.561017202329,
        0.712782446254,
        0.197757711625,
        0.089261476779,
        -0.543453195268,
        -0.386683443038,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [7.221489809803, 5.546929611159, 5.069974400687];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [2.172913128045, 1.282017837965, 1.071026684317];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.391157355553, 0.230782676398, 0.19280106515];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        2.363397745407,
        2.021173405553,
        -0.484209405076,
        -0.764686745651,
        0.273978455921,
        -0.593005002295,
    ];
    check("cfg5", 25, 5, 3, 6, &x, &comp, &sv, &ev, &evr, &tr2);
}

// gen_batches boundary: n divisible by batch_size (remainder 0), 3 equal batches.
// gen_batches(12,4,min=3) = [(0,4),(4,8),(8,12)]. Target #2.
#[test]
fn boundary_rem0_12_4_3_4() {
    let x: Vec<f64> = X3[..12 * 4].to_vec();
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.932661151098,
        0.21347585714,
        0.269530783007,
        -0.1091988675,
        -0.320669615612,
        0.811621104355,
        0.315950308356,
        -0.372313823594,
        0.035202253561,
        0.468112072335,
        -0.139320451934,
        0.871906933538,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [5.109036112279, 3.744705096023, 3.341089365649];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [2.37293181787, 1.274801477835, 1.014807104478];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.463499515471, 0.249004148728, 0.198220023719];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        1.378593949399,
        -0.308983389566,
        1.771436685621,
        1.434589601692,
        -0.578513835383,
        -0.951541922606,
    ];
    check("rem0", 12, 4, 3, 4, &x, &comp, &sv, &ev, &evr, &tr2);
}

// n_components edge: K=1 (single component), gen_batches(10,4,min=1)=[(0,4),(4,8),(8,10)].
// Target #5 (n_components=1 edge).
#[test]
fn edge_k1_10_3_1_4() {
    let x: Vec<f64> = X3[..10 * 3].to_vec();
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [-0.193990387902, 0.342605852674, 0.919232810073];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [3.735920254992];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [1.55078890574];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.45948177286];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [0.760568828681, -0.627227576544];
    check("k1", 10, 3, 1, 4, &x, &comp, &sv, &ev, &evr, &tr2);
}

// No-panic / no-NaN (R-CODE-2, target #7): degenerate rank-deficient merge
// matrices (a constant zero feature + duplicate rows). sklearn handles this
// cleanly via LAPACK gesdd (no NaN); ferray svd_lapack must too, and the
// fitted values must match the oracle (sign-aware, svd_flip).
#[test]
fn nopanic_degenerate_rank_deficient_12_4_2_5() {
    let x = Array2::from_shape_vec(
        (12, 4),
        vec![
            1., 2., 3., 0., 1., 2., 3., 0., 4., 5., 6., 0., 4., 5., 6., 0., 7., 8., 9., 0., 2., 1.,
            0., 0., 3., 3., 3., 0., 0., 0., 0., 0., 1., 0., 2., 0., 5., 5., 1., 0., 2., 2., 2., 0.,
            9., 1., 4., 0.,
        ],
    )
    .unwrap();
    let f = IncrementalPCA::<f64>::new(2)
        .with_batch_size(5)
        .fit(&x, &())
        .expect("degenerate fit must not error");
    // No NaN anywhere.
    for v in f.components().iter() {
        assert!(v.is_finite(), "component NaN/Inf on degenerate input");
    }
    for v in f.singular_values().iter() {
        assert!(v.is_finite(), "singular value NaN/Inf on degenerate input");
    }
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let comp = [
        0.561654659514,
        0.545781728899,
        0.621825013847,
        0.0,
        0.826413473016,
        -0.406232705641,
        -0.389891985664,
        0.0,
    ];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let sv = [12.988473822229, 6.20856790871];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let ev = [15.336404748248, 3.504210497915];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let evr = [0.752567073148, 0.17195382369];
    #[allow(clippy::excessive_precision, reason = "oracle")]
    let tr2 = [
        -1.873997344785,
        -1.423430063169,
        -1.873997344785,
        -1.423430063169,
    ];
    let fc = f.components();
    for k in 0..2 {
        for j in 0..4 {
            let diff = (fc[[k, j]] - comp[k * 4 + j]).abs();
            assert!(
                diff < 1e-6,
                "comp[{k}][{j}] = {} sklearn {} diff {diff}",
                fc[[k, j]],
                comp[k * 4 + j]
            );
        }
    }
    for k in 0..2 {
        assert!((f.singular_values()[k] - sv[k]).abs() < 1e-6, "sv[{k}]");
        assert!((f.explained_variance()[k] - ev[k]).abs() < 1e-6, "ev[{k}]");
        assert!(
            (f.explained_variance_ratio()[k] - evr[k]).abs() < 1e-6,
            "evr[{k}]"
        );
    }
    let t = f
        .transform(&x.slice(ndarray::s![0..2, ..]).to_owned())
        .unwrap();
    for r in 0..2 {
        for k in 0..2 {
            assert!((t[[r, k]] - tr2[r * 2 + k]).abs() < 1e-6, "tr[{r}][{k}]");
        }
    }
}
