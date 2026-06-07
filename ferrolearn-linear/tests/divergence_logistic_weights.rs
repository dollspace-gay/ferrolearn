//! Divergence tests for `LogisticRegression` sample_weight / class_weight /
//! n_iter_ (REQ-17/18/19, #450/#451/#445/#452).
//!
//! All expected coef/intercept values are the LIVE scikit-learn 1.5.2 oracle
//! (`LogisticRegression(solver='lbfgs').fit(...)`), captured with a tight
//! `tol=1e-6`..`1e-9` so the optimum is fully resolved (R-CHAR-3 — never
//! literal-copied from ferrolearn). ferrolearn's L-BFGS is a different
//! implementation than scipy's, so we assert agreement to the shared L-BFGS
//! tolerance, not ULP equality (goal.md R-DEV-1/R-DEV-7).

use ferrolearn_core::{Fit, HasCoefficients};
use ferrolearn_linear::LogisticRegression;
use ferrolearn_linear::logistic_regression::ClassWeight;
use ndarray::{Array1, Array2, array};

/// Shared L-BFGS agreement tolerance: ferrolearn's optimizer ≠ scipy's, so the
/// weighted optima agree to ~1e-3 at the default-ish tol (same margin already
/// documented for the unweighted path in the design doc, AC-1).
const LBFGS_TOL: f64 = 5e-3;

fn binary_data() -> (Array2<f64>, Array1<usize>) {
    let x = Array2::from_shape_vec((6, 2), vec![1., 2., 2., 3., 3., 4., 5., 6., 6., 7., 7., 8.])
        .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    (x, y)
}

#[test]
fn sample_weight_none_is_byte_identical_to_unweighted() {
    // fit_with_sample_weight(.., None) MUST equal Fit::fit exactly (same code
    // path; the weight loop collapses to w_i = 1).
    let (x, y) = binary_data();
    let m = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_tol(1e-9);
    let a = m.fit(&x, &y).unwrap();
    let b = m.fit_with_sample_weight(&x, &y, None).unwrap();
    assert_eq!(a.coefficients(), b.coefficients());
    assert_eq!(a.intercept(), b.intercept());
    assert_eq!(a.n_iter(), b.n_iter());
}

#[test]
fn binary_sample_weight_matches_oracle() {
    // Oracle (sklearn 1.5.2, tol=1e-6):
    //   LogisticRegression(C=1.0, solver='lbfgs').fit(X, y, sample_weight=w)
    //   coef_      = [0.843278520512189, 0.8432767550338345]
    //   intercept_ = [-7.123813438934774]
    let (x, y) = binary_data();
    let w = array![1.0f64, 2.0, 1.0, 3.0, 1.0, 2.0];
    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(3000)
        .with_tol(1e-9)
        .fit_with_sample_weight(&x, &y, Some(&w))
        .unwrap();
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - 0.843278520512189).abs() < LBFGS_TOL,
        "coef0 {} vs oracle 0.8432785",
        coef[0]
    );
    assert!(
        (coef[1] - 0.8432767550338345).abs() < LBFGS_TOL,
        "coef1 {} vs oracle 0.8432768",
        coef[1]
    );
    assert!(
        (fitted.intercept() - (-7.123813438934774)).abs() < 5e-2,
        "intercept {} vs oracle -7.1238",
        fitted.intercept()
    );
}

#[test]
fn integer_sample_weight_equals_row_duplication() {
    // Invariant: integer sample_weight ≡ duplicating those rows. Oracle confirms
    // sklearn satisfies this (both give coef ≈ [0.73002, 0.73002]); ferrolearn
    // must too, computed entirely within ferrolearn (oracle only validates the
    // invariant target, not a copied number).
    let (x, _) = binary_data();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let w = array![2.0f64, 1.0, 1.0, 1.0, 1.0, 2.0];
    let weighted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .fit_with_sample_weight(&x, &y, Some(&w))
        .unwrap();

    // Duplicate the weight-2 rows (row 0 and row 5).
    let xd = Array2::from_shape_vec(
        (8, 2),
        vec![
            1., 2., 1., 2., // row 0 x2
            2., 3., 3., 4., 5., 6., 6., 7., //
            7., 8., 7., 8., // row 5 x2
        ],
    )
    .unwrap();
    let yd = array![0usize, 0, 0, 0, 1, 1, 1, 1];
    let dup = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .fit(&xd, &yd)
        .unwrap();

    let cw = weighted.coefficients();
    let cd = dup.coefficients();
    assert!(
        (cw[0] - cd[0]).abs() < 1e-2 && (cw[1] - cd[1]).abs() < 1e-2,
        "weighted {cw:?} vs dup {cd:?}"
    );
}

#[test]
fn n_iter_contract() {
    // R-DEV-7: not asserted == sklearn; assert the CONTRACT: 1 <= n_iter <= max_iter
    // and deterministic across repeated fits.
    let (x, y) = binary_data();
    let m = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .with_tol(1e-4);
    let a = m.fit(&x, &y).unwrap();
    let b = m.fit(&x, &y).unwrap();
    assert!(
        a.n_iter() >= 1,
        "n_iter must be positive, got {}",
        a.n_iter()
    );
    assert!(
        a.n_iter() <= 1000,
        "n_iter {} must be <= max_iter 1000",
        a.n_iter()
    );
    assert_eq!(a.n_iter(), b.n_iter(), "n_iter must be deterministic");
}

#[test]
fn class_weight_balanced_matches_oracle() {
    // Imbalanced 2-class. Oracle (sklearn 1.5.2, class_weight='balanced',
    // tol=1e-6): coef_ = [0.7198029494298145, 0.7198024263386024],
    // intercept_ = [-6.403904583405147]; balanced weights = [0.75, 1.5].
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 2., 2., 3., 3., 4., 2.5, 3.5, 5., 6., 7., 8.],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1];
    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .with_class_weight(ClassWeight::Balanced)
        .fit(&x, &y)
        .unwrap();
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - 0.7198029494298145).abs() < LBFGS_TOL,
        "balanced coef0 {} vs oracle 0.7198029",
        coef[0]
    );
    assert!(
        (coef[1] - 0.7198024263386024).abs() < LBFGS_TOL,
        "balanced coef1 {} vs oracle 0.7198024",
        coef[1]
    );
}

#[test]
fn class_weight_balanced_equals_equivalent_sample_weight() {
    // sklearn explicitly tests that class_weight='balanced' == passing the
    // equivalent sample_weight (_logistic.py:364-366). Verify in ferrolearn.
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 2., 2., 3., 3., 4., 2.5, 3.5, 5., 6., 7., 8.],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1];
    let balanced = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .with_class_weight(ClassWeight::Balanced)
        .fit(&x, &y)
        .unwrap();
    // balanced weights: class0 -> 6/(2*4)=0.75, class1 -> 6/(2*2)=1.5.
    let sw = array![0.75f64, 0.75, 0.75, 0.75, 1.5, 1.5];
    let via_sw = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .fit_with_sample_weight(&x, &y, Some(&sw))
        .unwrap();
    let cb = balanced.coefficients();
    let cs = via_sw.coefficients();
    assert!(
        (cb[0] - cs[0]).abs() < 1e-6 && (cb[1] - cs[1]).abs() < 1e-6,
        "balanced {cb:?} vs equivalent sample_weight {cs:?}"
    );
}

#[test]
fn class_weight_dict_matches_oracle() {
    // Oracle (sklearn 1.5.2, class_weight={0:1.0, 1:3.0}, tol=1e-6):
    //   coef_ = [0.8506061813067434, 0.8506061768217075]
    //   intercept_ = [-7.312455400156733]
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 2., 2., 3., 3., 4., 2.5, 3.5, 5., 6., 7., 8.],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1];
    let fitted = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .with_class_weight(ClassWeight::Dict(vec![(0, 1.0), (1, 3.0)]))
        .fit(&x, &y)
        .unwrap();
    let coef = fitted.coefficients();
    assert!(
        (coef[0] - 0.8506061813067434).abs() < LBFGS_TOL,
        "dict coef0 {} vs oracle 0.8506062",
        coef[0]
    );
    assert!(
        (coef[1] - 0.8506061768217075).abs() < LBFGS_TOL,
        "dict coef1 {} vs oracle 0.8506062",
        coef[1]
    );
}

#[test]
fn class_weight_dict_composes_with_sample_weight() {
    // Oracle: class_weight={0:1,1:3} together with sample_weight equals passing
    // the product as a plain sample_weight (_logistic.py:312-313).
    //   coef_ = [0.8863997801254735, 0.8863997109311854]
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1., 2., 2., 3., 3., 4., 2.5, 3.5, 5., 6., 7., 8.],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 0, 1, 1];
    let usw = array![1.0f64, 2.0, 1.0, 1.0, 1.0, 2.0];
    let composed = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .with_class_weight(ClassWeight::Dict(vec![(0, 1.0), (1, 3.0)]))
        .fit_with_sample_weight(&x, &y, Some(&usw))
        .unwrap();
    let coef = composed.coefficients();
    assert!(
        (coef[0] - 0.8863997801254735).abs() < LBFGS_TOL,
        "composed coef0 {} vs oracle 0.8863998",
        coef[0]
    );
    // Cross-check against the explicit effective weights computed in ferrolearn.
    let eff = array![1.0f64, 2.0, 1.0, 1.0, 3.0, 6.0]; // usw * class_weight[y]
    let direct = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(5000)
        .with_tol(1e-10)
        .fit_with_sample_weight(&x, &y, Some(&eff))
        .unwrap();
    let cd = direct.coefficients();
    assert!(
        (coef[0] - cd[0]).abs() < 1e-6 && (coef[1] - cd[1]).abs() < 1e-6,
        "composed {coef:?} vs direct effective {cd:?}"
    );
}

#[test]
fn class_weight_none_unchanged() {
    // class_weight = None must equal the plain unweighted fit.
    let (x, y) = binary_data();
    let plain = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_tol(1e-9)
        .fit(&x, &y)
        .unwrap();
    let model_no_cw = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_tol(1e-9);
    assert!(model_no_cw.class_weight.is_none());
    let fitted = model_no_cw.fit(&x, &y).unwrap();
    assert_eq!(plain.coefficients(), fitted.coefficients());
}

#[test]
fn multiclass_sample_weight_matches_oracle() {
    // 3-class weighted. Oracle (sklearn 1.5.2, C=10, tol=1e-6):
    //   coef_[0] = [-0.9729497350341689, -0.9718148584607449]
    //   coef_[1] = [ 1.1815107519139676, -0.20595593479527768]
    //   coef_[2] = [-0.20856101687980247, 1.1777707932560018]
    let xm = Array2::from_shape_vec(
        (9, 2),
        vec![
            0., 0., 0.5, 0., 0., 0.5, 5., 0., 5.5, 0., 5., 0.5, 0., 5., 0.5, 5., 0., 5.5,
        ],
    )
    .unwrap();
    let ym = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
    let wm = array![1.0f64, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0];
    let fitted = LogisticRegression::<f64>::new()
        .with_c(10.0)
        .with_max_iter(5000)
        .with_tol(1e-9)
        .fit_with_sample_weight(&xm, &ym, Some(&wm))
        .unwrap();
    let wmat = fitted.weight_matrix();
    assert_eq!(wmat.shape(), &[3, 2]);
    assert!(
        (wmat[[0, 0]] - (-0.9729497350341689)).abs() < 5e-2,
        "mc coef[0,0] {} vs oracle -0.97295",
        wmat[[0, 0]]
    );
    assert!(
        (wmat[[1, 0]] - 1.1815107519139676).abs() < 5e-2,
        "mc coef[1,0] {} vs oracle 1.18151",
        wmat[[1, 0]]
    );
    assert!(
        (wmat[[2, 1]] - 1.1777707932560018).abs() < 5e-2,
        "mc coef[2,1] {} vs oracle 1.17777",
        wmat[[2, 1]]
    );
}

#[test]
fn random_state_n_jobs_are_noops_on_lbfgs() {
    // random_state/n_jobs are stored but never affect the lbfgs result.
    let (x, y) = binary_data();
    let base = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_tol(1e-9)
        .fit(&x, &y)
        .unwrap();
    let with_rs = LogisticRegression::<f64>::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_tol(1e-9)
        .with_random_state(42)
        .with_n_jobs(4)
        .fit(&x, &y)
        .unwrap();
    assert_eq!(base.coefficients(), with_rs.coefficients());
    assert_eq!(base.intercept(), with_rs.intercept());
}
