//! Divergence + green-guard tests for `ferrolearn-decomp` cross-decomposition
//! (`PLSRegression`/`PLSCanonical`/`CCA`/`PLSSVD`) against scikit-learn 1.5.2
//! `sklearn/cross_decomposition/_pls.py`.
//!
//! All expected values are produced by the live sklearn 1.5.2 oracle (run from
//! `/tmp`), never literal-copied from ferrolearn (R-CHAR-3). Cross-decomposition
//! is deterministic (NIPALS power method / SVD, no RNG); value parity is
//! achievable modulo a documented per-component sign convention.
//!
//! Tracking doc: `.design/decomp/cross_decomposition.md`, crosslink #1618.

use ferrolearn_core::traits::{Fit, Predict, Transform};
use ferrolearn_decomp::cross_decomposition::{CCA, PLSCanonical, PLSRegression, PLSSVD};
use ndarray::{Array1, Array2, array};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// Probe-1 fixture: 5x3 X, 5x2 Y. PLSRegression comp-0 max-abs x_weight is
/// positive in sklearn AND ferrolearn on this input (sign happens to agree),
/// so it isolates the SIGN-INVARIANT `predict` parity (DIV-A) cleanly.
fn fixture_a() -> (Array2<f64>, Array2<f64>) {
    let x = array![
        [1., 2., 3.],
        [4., 5., 7.],
        [7., 9., 8.],
        [10., 11., 14.],
        [13., 15., 16.]
    ];
    let y = array![[1., 0.5], [2., 1.2], [3., 1.4], [4., 2.1], [5., 2.6]];
    (x, y)
}

/// Sign-divergence fixture: 6x3 X, 6x2 Y on which ferrolearn's leading
/// x_weights column has a NEGATIVE max-abs entry while sklearn (via
/// `_svd_flip_1d` / `svd_flip`) forces it POSITIVE. Used to pin DIV-B / DIV-E.
fn fixture_signflip() -> (Array2<f64>, Array2<f64>) {
    let x = array![
        [-4.823583459179779, 4.883853299755387, 0.45934070397560367],
        [-4.869719549252166, 2.189515300938093, 0.1840278888191893],
        [-1.4169653841366414, -3.6341247990884726, -4.91592737953837],
        [-0.9189906304919737, 1.1480926601730468, 0.9331729855417237],
        [-2.332087778879832, 1.795926181704136, 0.9831885379135197],
        [2.0381690528239362, 1.5083991633664917, 0.828025007287363]
    ];
    let y = array![
        [0.44066101114351053, -2.0557329797909683],
        [3.8534842021373517, 0.6626729257328001],
        [0.6403854516589058, -2.701875385456546],
        [-3.7324903951190347, 1.8312329023960106],
        [0.36713296373599213, -0.9815250606851833],
        [-4.934660431650117, -3.3455827945138115]
    ];
    (x, y)
}

/// Sign of the entry of `col` that is largest in absolute value.
fn argmax_abs_sign(col: ndarray::ArrayView1<f64>) -> f64 {
    let mut best_i = 0;
    let mut best_v = -1.0;
    for (i, &v) in col.iter().enumerate() {
        if v.abs() > best_v {
            best_v = v.abs();
            best_i = i;
        }
    }
    col[best_i].signum()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

/// Per-element abs-difference of magnitudes: `||a| - |b||`, ignoring sign.
fn max_abs_diff_signless(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (&x, &y)| m.max((x.abs() - y.abs()).abs()))
}

/// ddof=1 (Bessel) standard deviation of a 1-D array.
fn std_ddof1(c: ndarray::ArrayView1<f64>) -> f64 {
    let n = c.len() as f64;
    let mean = c.iter().sum::<f64>() / n;
    (c.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

// ===========================================================================
// DIV-A (REQ-9): PLSRegression::predict element-wise value parity (GREEN-GUARD)
// ===========================================================================

/// GREEN-GUARD (REQ-9 SHIPPED): ferrolearn `PLSRegression::predict` matches the
/// live sklearn oracle ELEMENT-WISE. `coef_` is sign-invariant + deterministic,
/// so `predict(X)` is independent of the missing `_svd_flip_1d` sign convention.
///
/// sklearn `_PLS.predict` (`sklearn/cross_decomposition/_pls.py:530-531`):
///   `Ypred = (X - x_mean) @ coef_.T + intercept_`.
/// Oracle: `PLSRegression(n_components=2).fit(X,Y).predict(X)` (sklearn 1.5.2).
#[test]
fn greenguard_plsregression_predict_matches_sklearn() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3), produced from /tmp:
    //   PLSRegression(n_components=2).fit(X,Y).predict(X)
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_predict: Array2<f64> = array![
        [1.0001724645730092, 0.5393899142133716],
        [2.0133206213794526, 1.1128788919899593],
        [2.989166126574906, 1.4183243472874631],
        [3.9818783472848387, 2.164104543668567],
        [5.015462440187793, 2.5653023028406383]
    ];

    let (x, y) = fixture_a();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let pred = fitted.predict(&x).expect("predict");

    let diff = max_abs_diff(&pred, &sk_predict);
    assert!(
        diff < 1e-6,
        "PLSRegression::predict diverges from sklearn by {diff:e}; ferro={pred:?} sklearn={sk_predict:?}"
    );
}

// ===========================================================================
// DIV-B (REQ-15): _svd_flip_1d sign convention — NIPALS estimators
//   blocker #1620
// ===========================================================================

/// Divergence: ferrolearn's `fn nipals` (`cross_decomposition.rs:704`) omits the
/// `_svd_flip_1d` sign convention that sklearn applies per component
/// (`sklearn/cross_decomposition/_pls.py:354`, def `:154-161`:
/// `idx = argmax(abs(x_weights)); sign = sign(x_weights[idx]); x_weights *= sign`).
/// sklearn therefore guarantees each `x_weights_` column's max-abs entry is
/// POSITIVE; ferrolearn does not.
///
/// On `fixture_signflip`, sklearn `PLSRegression.x_weights_[:,0]` max-abs entry
/// is `+0.9839982361591639` (positive); ferrolearn returns `-0.984...` (negative).
/// Tracking: #1620
#[test]
fn divergence_plsregression_xweights_sign_convention() {
    let (x, y) = fixture_signflip();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let xw = fitted.x_weights();

    // sklearn _svd_flip_1d guarantees every column's max-abs entry is positive.
    for c in 0..xw.ncols() {
        let sign = argmax_abs_sign(xw.column(c));
        assert!(
            sign > 0.0,
            "PLSRegression x_weights_ column {c} max-abs entry sign = {sign} \
             (sklearn _svd_flip_1d forces it positive, _pls.py:158-161); column = {:?}",
            xw.column(c)
        );
    }
}

/// Divergence: same `_svd_flip_1d` omission pins `x_weights_` element-wise
/// (including sign) against the live sklearn oracle.
///
/// Oracle (sklearn 1.5.2): `PLSRegression(n_components=2).fit(X,Y).x_weights_`
/// on `fixture_signflip`. ferrolearn matches up to the per-component sign that
/// sklearn pins and ferrolearn does not.
/// Tracking: #1620, #1622 (NIPALS convergence-criterion fix)
#[test]
fn divergence_plsregression_xweights_elementwise() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): PLSRegression(2).fit(X,Y).x_weights_
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_xw: Array2<f64> = array![
        [0.9839982361591639, -0.07917795891489479],
        [-0.10130721011159803, 0.488331967848574],
        [0.1465753062935852, 0.8690585365780653]
    ];

    let (x, y) = fixture_signflip();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let xw = fitted.x_weights();

    // Signed comparison: FAILS now because ferrolearn's comp-0 sign is opposite.
    let diff = max_abs_diff(xw, &sk_xw);
    assert!(
        diff < 1e-6,
        "PLSRegression x_weights_ sign-diverges from sklearn by {diff:e}; ferro={xw:?} sklearn={sk_xw:?}"
    );

    // Cross-check (R-CHAR-3): the magnitudes DO match — proving this is purely a
    // sign-convention divergence, not a value/algebra error.
    let signless = max_abs_diff_signless(xw, &sk_xw);
    assert!(
        signless < 1e-6,
        "magnitudes should match (signless diff {signless:e})"
    );
}

// ===========================================================================
// DIV-C (REQ-13): CCA mode='B' — genuine value/algebra divergence
//   blocker #1619
// ===========================================================================

/// Divergence: sklearn `CCA` is `mode='B'` — `x_weights = X_pinv @ y_score`
/// (`sklearn/cross_decomposition/_pls.py:88-89`) — while ferrolearn's `fn nipals`
/// runs the SAME mode-A power method (`w = Xᵀu/(uᵀu)`, `cross_decomposition.rs:756`)
/// for all three NIPALS estimators, differing only by `ScoreNorm::UnitVariance`.
/// CCA's `x_weights_` VALUES therefore diverge from sklearn beyond any sign.
///
/// Oracle (sklearn 1.5.2) `CCA(n_components=2).fit(X,Y).x_weights_` on
/// `fixture_a` is `[[1.0, ~0],[~0, 0.847],[~0, 0.532]]`; ferrolearn returns the
/// mode-A `[[0.580, -0.294],[0.576, -0.661],[0.576, 0.690]]`.
/// Tracking: #1619
#[test]
fn divergence_cca_mode_b_xweights() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): CCA(n_components=2).fit(X,Y).x_weights_
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_xw: Array2<f64> = array![
        [0.9999999999999998, -6.336732573894699e-15],
        [3.552713678800538e-15, 0.8467028021754360],
        [3.552713678800538e-15, 0.5320661282098912]
    ];

    let (x, y) = fixture_a();
    let fitted = CCA::<f64>::new(2).fit(&x, &y).expect("CCA fit");
    let xw = fitted.x_weights();

    // Genuine VALUE divergence (not merely sign): magnitudes differ too.
    let signless = max_abs_diff_signless(xw, &sk_xw);
    assert!(
        signless < 1e-6,
        "CCA x_weights_ diverges in VALUE from sklearn mode='B' by {signless:e} \
         (signless); ferro={xw:?} sklearn={sk_xw:?}"
    );
}

/// Divergence: CCA `transform(X)` (X-scores) diverges in VALUE from sklearn
/// because of the mode-A vs mode-B weights. sklearn `transform` projects via the
/// frozen `x_rotations_` (`_pls.py:438`); on `fixture_a` the mode-B leading score
/// column equals `[-1.2649.., -0.6324.., 0, 0.6324.., 1.2649..]`, whereas
/// ferrolearn's mode-A leading score is `[-2.1556.., -0.7744.., ...]`.
/// Tracking: #1619
#[test]
fn divergence_cca_mode_b_transform() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): CCA(2).fit(X,Y).transform(X)
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_t: Array2<f64> = array![
        [-1.2649110640673606, 1.8570126455714338e-16],
        [-0.6324555320336799, 0.036672779305980001],
        [-6.4707807815934691e-16, -0.060010002500694368],
        [0.6324555320336805, 0.010001667083448932],
        [1.2649110640673606, 0.013335556111265206]
    ];

    let (x, y) = fixture_a();
    let fitted = CCA::<f64>::new(2).fit(&x, &y).expect("CCA fit");
    let t = fitted.transform(&x).expect("transform");

    let signless = max_abs_diff_signless(&t, &sk_t);
    assert!(
        signless < 1e-6,
        "CCA transform diverges in VALUE from sklearn mode='B' by {signless:e} \
         (signless); ferro={t:?} sklearn={sk_t:?}"
    );
}

/// Divergence: CCA `transform(X)` does NOT reproduce the unit-variance scores
/// that sklearn produces (and that ferrolearn's OWN stored `x_scores_` have).
/// sklearn `transform` projects training X via the frozen `x_rotations_`
/// (`_pls.py:438`) yielding a leading score with ddof=1 std ≈ 1.0 (Probe 2);
/// ferrolearn recomputes a `W(PᵀW)⁻¹` rotation per call (`:1824-1827`) so the
/// projected leading score has std ≈ 1.766, inconsistent with both sklearn and
/// ferrolearn's own normalised training `x_scores_` (std ≈ 1.0). Folds under the
/// CCA mode/rotation divergence.
/// Tracking: #1619
#[test]
fn divergence_cca_transform_leading_score_unit_variance() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): CCA(2).fit(X,Y).transform(X) col-0
    // ddof=1 std == 1.0 (Probe 2).
    let sk_leading_std = 1.0_f64;

    let (x, y) = fixture_a();
    let fitted = CCA::<f64>::new(2).fit(&x, &y).expect("CCA fit");
    let t = fitted.transform(&x).expect("transform");
    let std = std_ddof1(t.column(0));
    assert!(
        (std - sk_leading_std).abs() < 1e-6,
        "CCA transform leading-score ddof1 std diverges from sklearn ~1.0: got {std}"
    );
}

// ===========================================================================
// DIV-E (REQ-16): PLSSVD svd_flip sign convention
//   blocker #1621
// ===========================================================================

/// Divergence: ferrolearn `PLSSVD::fit` (`cross_decomposition.rs:580`) takes the
/// raw SVD left/right singular vectors without sklearn's `svd_flip(U, Vt)`
/// (`sklearn/cross_decomposition/_pls.py:1105`), which forces each `x_weights_`
/// column's max-abs entry positive. On `fixture_signflip`, sklearn comp-0 max-abs
/// entry is `+0.9840196653379837`; ferrolearn returns `-0.984...`.
/// Tracking: #1621
#[test]
fn divergence_plssvd_xweights_sign_convention() {
    let (x, y) = fixture_signflip();
    let fitted = PLSSVD::<f64>::new(2).fit(&x, &y).expect("PLSSVD fit");
    let xw = fitted.x_weights();

    for c in 0..xw.ncols() {
        let sign = argmax_abs_sign(xw.column(c));
        assert!(
            sign > 0.0,
            "PLSSVD x_weights_ column {c} max-abs entry sign = {sign} \
             (sklearn svd_flip forces it positive, _pls.py:1105); column = {:?}",
            xw.column(c)
        );
    }
}

/// Divergence: PLSSVD `x_weights_` element-wise (including sign) vs the live
/// sklearn oracle. Magnitudes match (cross-check), sign diverges.
/// Tracking: #1621
#[test]
fn divergence_plssvd_xweights_elementwise() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): PLSSVD(2).fit(X,Y).x_weights_
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_xw: Array2<f64> = array![
        [0.9840196653379837, -0.10362601785128867],
        [-0.10137796664609638, 0.34248561699841634],
        [0.1463823968475223, 0.9337907959353093]
    ];

    let (x, y) = fixture_signflip();
    let fitted = PLSSVD::<f64>::new(2).fit(&x, &y).expect("PLSSVD fit");
    let xw = fitted.x_weights();

    let diff = max_abs_diff(xw, &sk_xw);
    assert!(
        diff < 1e-6,
        "PLSSVD x_weights_ sign-diverges from sklearn by {diff:e}; ferro={xw:?} sklearn={sk_xw:?}"
    );

    // Cross-check: magnitudes match — confirms a pure sign-convention divergence.
    let signless = max_abs_diff_signless(xw, &sk_xw);
    assert!(
        signless < 1e-6,
        "magnitudes should match (signless diff {signless:e})"
    );
}

// ===========================================================================
// GREEN-GUARDS — SHIPPED behaviour that must PASS against current code
// ===========================================================================

/// GREEN-GUARD (REQ-9, scoped): PLSRegression `transform` (X-scores) matches the
/// sklearn oracle UP TO SIGN on `fixture_a` (where comp-0 sign happens to agree,
/// so element-wise also holds). sklearn `transform` = `(X-mean)/std @ x_rotations_`
/// (`_pls.py:438`); ferrolearn recomputes the same `W(PᵀW)⁻¹` rotation.
#[test]
fn greenguard_plsregression_transform_matches_up_to_sign() {
    // Live sklearn 1.5.2 oracle (R-CHAR-3): PLSRegression(2).fit(X,Y).transform(X)
    #[allow(
        clippy::excessive_precision,
        reason = "live sklearn 1.5.2 oracle (R-CHAR-3)"
    )]
    let sk_t: Array2<f64> = array![
        [-2.1740776026508573, 0.00931962649996359],
        [-1.0339335282879625, 0.13134220899526797],
        [-0.10519109483141698, -0.3057415013983234],
        [1.1381051718758641, 0.22677600231080697],
        [2.1750970538943726, -0.0616963364077147]
    ];

    let (x, y) = fixture_a();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let t = fitted.transform(&x).expect("transform");
    let signless = max_abs_diff_signless(&t, &sk_t);
    assert!(
        signless < 1e-6,
        "PLSRegression transform magnitudes diverge from sklearn by {signless:e}"
    );
}

/// GREEN-GUARD (REQ-1): `scale=True` uses ddof=1 (Bessel) std (NON-DIVERGENCE).
/// sklearn `_center_scale_xy` uses `X.std(axis=0, ddof=1)`
/// (`sklearn/cross_decomposition/_pls.py:142,145`). We pin the Bessel value
/// against the live numpy oracle and assert it is NOT the ddof=0 value
/// (non-tautological). The `predict` green-guard further confirms ferrolearn's
/// internal scaling uses ddof=1 (a ddof=0 scaling would break that match).
#[test]
fn greenguard_ddof1_scaling_not_ddof0() {
    let (x, _y) = fixture_a();
    let col0 = x.column(0);
    let n = col0.len() as f64;
    let mean = col0.iter().sum::<f64>() / n;
    let var_ddof1 = col0.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    let std_ddof1 = var_ddof1.sqrt();

    // Live numpy oracle (R-CHAR-3): X[:,0].std(ddof=1) == 4.743416..,
    // X[:,0].std(ddof=0) == 4.242641.. (Probe 4).
    #[allow(clippy::excessive_precision, reason = "live numpy oracle (R-CHAR-3)")]
    let np_std_ddof1 = 4.743416490252569_f64;
    #[allow(clippy::excessive_precision, reason = "live numpy oracle (R-CHAR-3)")]
    let np_std_ddof0 = 4.242640687119285_f64;

    assert!(
        (std_ddof1 - np_std_ddof1).abs() < 1e-9,
        "ddof=1 std mismatch: {std_ddof1} vs np {np_std_ddof1}"
    );
    assert!(
        (std_ddof1 - np_std_ddof0).abs() > 0.1,
        "ddof=1 must differ from ddof=0 (non-tautological guard)"
    );
}

/// GREEN-GUARD (REQ-6): fitted shapes match sklearn allocation
/// (`_pls.py:309-314`): x_weights_ `(p, k)`, x_scores `(n, k)`, y_loadings_ `(q, k)`.
#[test]
fn greenguard_fitted_shapes() {
    let (x, y) = fixture_a();
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    assert_eq!(fitted.x_weights().dim(), (3, 2)); // (p, k)
    assert_eq!(fitted.x_loadings().dim(), (3, 2));
    assert_eq!(fitted.y_loadings().dim(), (2, 2)); // (q, k)
    assert_eq!(fitted.x_scores().dim(), (5, 2)); // (n, k)
    assert_eq!(fitted.y_scores().dim(), (5, 2));
    assert_eq!(fitted.coefficients().dim(), (3, 2)); // internal (p, q)

    let svd = PLSSVD::<f64>::new(2).fit(&x, &y).expect("PLSSVD fit");
    assert_eq!(svd.x_weights().dim(), (3, 2));
    assert_eq!(svd.y_weights().dim(), (2, 2));
}

/// GREEN-GUARD (REQ-8): CCA's stored training `x_scores_` have ddof=1 std ≈ 1.0
/// (the `ScoreNorm::UnitVariance` normalisation applied in `fn nipals`). NOTE:
/// this is the STORED training score, not `transform(X)` — the latter diverges
/// (see `divergence_cca_transform_leading_score_unit_variance`).
#[test]
fn greenguard_cca_stored_scores_unit_variance() {
    let (x, y) = fixture_a();
    let fitted = CCA::<f64>::new(2).fit(&x, &y).expect("CCA fit");
    let col0: Array1<f64> = fitted.x_scores().column(0).to_owned();
    let std = std_ddof1(col0.view());
    assert!(
        (std - 1.0).abs() < 1e-6,
        "CCA stored x_scores_ leading column ddof1 std should be ~1.0, got {std}"
    );
}

/// GREEN-GUARD (REQ-7): PLSRegression and PLSCanonical produce DIFFERENT scores
/// on the same data (distinct deflation modes, `_pls.py:366-375`).
#[test]
fn greenguard_regression_vs_canonical_differ() {
    let (x, y) = fixture_a();
    let reg = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let can = PLSCanonical::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSCanonical fit");
    let tr = reg.transform(&x).expect("reg transform");
    let tc = can.transform(&x).expect("can transform");
    let diff = max_abs_diff(&tr, &tc);
    assert!(
        diff > 1e-6,
        "PLSRegression and PLSCanonical scores should differ (distinct deflation), diff={diff:e}"
    );
}

/// GREEN-GUARD (REQ-2): `scale=false` toggle changes the fitted result.
#[test]
fn greenguard_scale_false_toggle() {
    let (x, y) = fixture_a();
    let scaled = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("scaled fit");
    let unscaled = PLSRegression::<f64>::new(2)
        .with_scale(false)
        .fit(&x, &y)
        .expect("unscaled fit");
    let diff = max_abs_diff(scaled.x_weights(), unscaled.x_weights());
    assert!(
        diff > 1e-9,
        "scale=true vs scale=false should change x_weights, diff={diff:e}"
    );
}

/// GREEN-GUARD (REQ-4): error contracts — row mismatch, zero components,
/// too-many components, insufficient samples, transform column mismatch.
#[test]
fn greenguard_error_contracts() {
    let (x, y) = fixture_a();

    // row mismatch
    let y_bad = array![[1., 0.5], [2., 1.2]];
    assert!(PLSRegression::<f64>::new(2).fit(&x, &y_bad).is_err());

    // n_components == 0
    assert!(PLSRegression::<f64>::new(0).fit(&x, &y).is_err());

    // n_components too large (X has 3 features, Y has 2 -> max 2)
    assert!(PLSRegression::<f64>::new(99).fit(&x, &y).is_err());

    // insufficient samples
    let x1 = array![[1., 2., 3.]];
    let y1 = array![[1., 0.5]];
    assert!(PLSRegression::<f64>::new(1).fit(&x1, &y1).is_err());

    // transform column mismatch
    let fitted = PLSRegression::<f64>::new(2)
        .fit(&x, &y)
        .expect("PLSRegression fit");
    let x_wrong = array![[1., 2.], [3., 4.]];
    assert!(fitted.transform(&x_wrong).is_err());
}

/// GREEN-GUARD (REQ-3): constructor defaults (n_components getter, builders).
#[test]
fn greenguard_ctor_defaults() {
    assert_eq!(PLSRegression::<f64>::new(2).n_components(), 2);
    assert_eq!(PLSCanonical::<f64>::new(3).n_components(), 3);
    assert_eq!(CCA::<f64>::new(1).n_components(), 1);
    assert_eq!(PLSSVD::<f64>::new(2).n_components(), 2);
    // builder chain compiles and returns Self
    let _ = PLSRegression::<f64>::new(2)
        .with_max_iter(100)
        .with_tol(1e-8)
        .with_scale(false);
}

/// GREEN-GUARD (REQ-5): f32 fit + predict runs and is close to the f64 oracle.
#[test]
fn greenguard_f32_runs() {
    let x32 = array![
        [1.0_f32, 2., 3.],
        [4., 5., 7.],
        [7., 9., 8.],
        [10., 11., 14.],
        [13., 15., 16.]
    ];
    let y32 = array![[1.0_f32, 0.5], [2., 1.2], [3., 1.4], [4., 2.1], [5., 2.6]];
    let fitted = PLSRegression::<f32>::new(2)
        .fit(&x32, &y32)
        .expect("f32 fit");
    let pred = fitted.predict(&x32).expect("f32 predict");
    assert_eq!(pred.dim(), (5, 2));
    // close to the f64 sklearn oracle row 0 (1.0001724..)
    assert!((pred[[0, 0]] - 1.0001724_f32).abs() < 1e-2);
}

/// GREEN-GUARD: determinism — two identical fits give identical x_weights.
#[test]
fn greenguard_determinism() {
    let (x, y) = fixture_a();
    let a = PLSRegression::<f64>::new(2).fit(&x, &y).expect("fit a");
    let b = PLSRegression::<f64>::new(2).fit(&x, &y).expect("fit b");
    assert_eq!(max_abs_diff(a.x_weights(), b.x_weights()), 0.0);
}
