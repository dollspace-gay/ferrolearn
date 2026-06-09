//! Divergence audit: `DummyClassifier` / `DummyRegressor` vs scikit-learn 1.5.2
//! (`sklearn/dummy.py`). Tracking #1690.
//!
//! All expected values are produced by a LIVE sklearn 1.5.2 oracle call run
//! from `/tmp` (`from sklearn.dummy import DummyClassifier, DummyRegressor`),
//! NEVER copied from the ferrolearn side (R-CHAR-3). The oracle command that
//! produced each constant is quoted inline above the assertion.
//!
//! Green guards (PASS now): REQ-1 deterministic classifier predict + tie-break,
//! REQ-4 constant-membership rejection (both raise/Err), REQ-6 deterministic
//! regressor value parity. Structural guards: REQ-3 stratified/uniform in-range
//! (RNG carve-out, NOT exact parity — blocker #1695).
//!
//! Behavioral gaps with NO ferrolearn entry point (blocker-only, no failing
//! test possible): REQ-2 predict_proba (#1691), REQ-8 sample_weight (#1693),
//! REQ-9 multi-output (#1694). The REQ-4 error-VARIANT divergence
//! (FerroError::InvalidParameter vs ValueError) is tracked at #1692.
//!
//! RED PIN (FAILS now, `#[ignore]`d): REQ-6 quantile rounding-order divergence
//! at quantile=1/3 (#2370).

use ferrolearn_core::Fit;
use ferrolearn_core::Predict;
use ferrolearn_core::introspection::HasClasses;
use ferrolearn_model_sel::{
    DummyClassifier, DummyClassifierStrategy, DummyRegressor, DummyRegressorStrategy,
};
use ndarray::{Array1, Array2, array};

fn zeros(rows: usize) -> Array2<f64> {
    Array2::<f64>::zeros((rows, 2))
}

// ---------------------------------------------------------------------------
// REQ-1: deterministic classifier predict + first-max (lowest-index) tie-break.
// GREEN GUARD — passes now; pins the SHIPPED value against the live oracle.
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/dummy.py:308-315`
/// (`classes_[k][class_prior_[k].argmax()]` tiled). For `y=[0,0,1,1,2]` the
/// priors are `[0.4,0.4,0.2]` with a TIE between classes 0 and 1; sklearn's
/// `argmax()` (`:311`) returns the FIRST maximal index, so it predicts the
/// LOWEST class, `0`.
///
/// Live oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// import numpy as np; from sklearn.dummy import DummyClassifier
/// X=np.zeros((5,2)); y=np.array([0,0,1,1,2]); Xt=np.zeros((3,2))
/// DummyClassifier(strategy='most_frequent').fit(X,y).predict(Xt).tolist()
/// # -> [0, 0, 0]   classes_=[0,1,2]  class_prior_=[0.4,0.4,0.2]
/// DummyClassifier(strategy='prior').fit(X,y).predict(Xt).tolist()  # -> [0, 0, 0]
/// ```
/// Tracking: #1690 (REQ-1 SHIPPED).
#[test]
fn green_classifier_most_frequent_prior_tiebreak() {
    let y = array![0usize, 0, 1, 1, 2];

    for strat in [
        DummyClassifierStrategy::MostFrequent,
        DummyClassifierStrategy::Prior,
    ] {
        let fitted: ferrolearn_model_sel::FittedDummyClassifier =
            DummyClassifier::new(strat).fit(&zeros(5), &y).unwrap();
        let preds = fitted.predict(&zeros(3)).unwrap();
        // Live oracle: [0, 0, 0] (lowest-index tie-break).
        assert_eq!(preds.to_vec(), vec![0usize, 0, 0]);
        // Live oracle: classes_ == [0, 1, 2].
        assert_eq!(fitted.classes(), &[0usize, 1, 2]);
        assert_eq!(fitted.n_classes(), 3);
    }
}

/// Constant strategy tiles the constant over every test row.
/// Mirrors `sklearn/dummy.py:332-333` (`np.tile(self.constant, ...)`).
///
/// Live oracle:
/// ```text
/// X=np.zeros((5,2)); y=np.array([0,0,1,1,2])
/// DummyClassifier(strategy='constant',constant=2).fit(X,y).predict(np.zeros((3,2))).tolist()
/// # -> [2, 2, 2]
/// ```
/// Tracking: #1690 (REQ-1 SHIPPED).
#[test]
fn green_classifier_constant_tiles() {
    let y = array![0usize, 0, 1, 1, 2];
    let fitted = DummyClassifier::new(DummyClassifierStrategy::Constant(2))
        .fit(&zeros(5), &y)
        .unwrap();
    let preds = fitted.predict(&zeros(3)).unwrap();
    assert_eq!(preds.to_vec(), vec![2usize, 2, 2]);
}

// ---------------------------------------------------------------------------
// REQ-4: constant-membership rejection. GREEN GUARD — sklearn RAISES for an
// unseen constant, ferrolearn returns Err: behavior matches (R-DEV-2 family).
// The error VARIANT (FerroError::InvalidParameter vs ValueError) divergence is
// tracked blocker-only at #1692 (no entry point to assert the Python type).
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/dummy.py:232-244` (membership guard raising `ValueError`).
///
/// Live oracle:
/// ```text
/// X=np.zeros((5,2)); y=np.array([0,0,1,1,2])
/// DummyClassifier(strategy='constant',constant=99).fit(X,y)
/// # -> raises ValueError("The constant target value must be present ...")
/// ```
/// sklearn RAISES; ferrolearn returns Err — behavior parity. Tracking: #1690.
#[test]
fn green_classifier_constant_unseen_errors() {
    let y = array![0usize, 0, 1, 1, 2];
    let res = DummyClassifier::new(DummyClassifierStrategy::Constant(99)).fit(&zeros(5), &y);
    assert!(
        res.is_err(),
        "sklearn raises ValueError for an unseen constant; ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// REQ-6: deterministic regressor value parity. GREEN GUARDS — pins mean /
// median (odd & even n) / quantile (linear interp, q=0.25 & q=0.9, odd & even
// n) / constant against the live numpy oracle to ~1e-12.
// ---------------------------------------------------------------------------

fn reg_value(strategy: DummyRegressorStrategy<f64>, y: &Array1<f64>) -> f64 {
    let fitted = DummyRegressor::<f64>::new(strategy)
        .fit(&zeros(y.len()), y)
        .unwrap();
    fitted.predict(&zeros(1)).unwrap()[0]
}

/// Mirrors `sklearn/dummy.py:581-606` (`np.average` / `np.median` /
/// `np.percentile(..., method='linear')`) tiled (`:655-664`).
///
/// Live oracle (sklearn 1.5.2 / numpy, run from /tmp):
/// ```text
/// y=np.array([1.,2.,3.,4.,5.])                       # odd n=5
/// mean        -> 3.0
/// median      -> 3.0
/// quantile .25-> 2.0
/// quantile .9 -> 4.6
/// y=np.array([1.,2.,3.,4.])                          # even n=4
/// median      -> 2.5
/// quantile .25-> 1.75
/// quantile .9 -> 3.7
/// ```
/// Tracking: #1690 (REQ-6 SHIPPED).
#[test]
fn green_regressor_value_parity() {
    let y5: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y4: Array1<f64> = array![1.0, 2.0, 3.0, 4.0];
    let tol = 1e-12;

    // odd n=5
    assert!((reg_value(DummyRegressorStrategy::Mean, &y5) - 3.0).abs() < tol);
    assert!((reg_value(DummyRegressorStrategy::Median, &y5) - 3.0).abs() < tol);
    assert!((reg_value(DummyRegressorStrategy::Quantile(0.25), &y5) - 2.0).abs() < tol);
    assert!((reg_value(DummyRegressorStrategy::Quantile(0.9), &y5) - 4.6).abs() < tol);

    // even n=4 — median averages the two central order statistics; quantile
    // linearly interpolates.
    assert!((reg_value(DummyRegressorStrategy::Median, &y4) - 2.5).abs() < tol);
    assert!((reg_value(DummyRegressorStrategy::Quantile(0.25), &y4) - 1.75).abs() < tol);
    assert!((reg_value(DummyRegressorStrategy::Quantile(0.9), &y4) - 3.7).abs() < tol);
}

/// Constant strategy tiles the constant over every row.
/// Mirrors `sklearn/dummy.py:655-664` (`np.full((n_samples, n_outputs), c)`).
///
/// Live oracle:
/// ```text
/// DummyRegressor(strategy='constant',constant=42.5)
///   .fit(np.zeros((3,2)),np.array([1.,2.,3.])).predict(np.zeros((2,2))).tolist()
/// # -> [42.5, 42.5]
/// ```
/// Tracking: #1690 (REQ-6 SHIPPED).
#[test]
fn green_regressor_constant_tiles() {
    let y: Array1<f64> = array![1.0, 2.0, 3.0];
    let fitted = DummyRegressor::<f64>::new(DummyRegressorStrategy::Constant(42.5))
        .fit(&zeros(3), &y)
        .unwrap();
    let preds = fitted.predict(&zeros(2)).unwrap();
    assert_eq!(preds.len(), 2);
    assert!((preds[0] - 42.5).abs() < 1e-12);
    assert!((preds[1] - 42.5).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// REQ-6 (RED PIN): quantile rounding-order divergence. FAILS now — #[ignore]d.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `DummyRegressor` `quantile` strategy diverges from
/// `sklearn/dummy.py:601`:
/// ```python
/// percentile = self.quantile * 100.0
/// self.constant_ = np.percentile(y, axis=0, q=percentile)
/// ```
/// sklearn forms `percentile = quantile*100` (here `1/3*100 == 33.33333333333333`,
/// already rounded), then numpy's virtual index is `percentile/100 * (n-1) ==
/// 0.9999999999999998`, so it INTERPOLATES below the second order statistic.
/// ferrolearn instead computes `pos = q*(n-1)` directly
/// (`dummy.rs:412`, `let pos = q * (n - 1) as f64;`); for `q=1/3, n=4` that is
/// `(1.0/3.0)*3.0 == 1.0` EXACTLY, so `lo == hi` and it returns `sorted[1]`
/// verbatim with no interpolation.
///
/// Input: `strategy='quantile', quantile=1.0/3.0`, `y=[10, 20, 30, 40]`.
/// LIVE sklearn 1.5.2 oracle (run from /tmp):
/// ```text
/// import numpy as np; from sklearn.dummy import DummyRegressor
/// DummyRegressor(strategy='quantile', quantile=1/3) \
///   .fit(np.zeros((4,2)), np.array([10.,20.,30.,40.])).constant_[0,0]
/// # -> 19.999999999999996   (hex 0x1.3ffffffffffffp+4)
/// ```
/// ferrolearn returns exactly `20.0` (hex `0x1.4000000000000p+4`) — 1 ULP high.
/// A genuine R-DEV-1 value divergence in a SHIPPED deterministic path.
/// Tracking: #2370
#[test]
#[ignore = "divergence: DummyRegressor quantile q*(n-1) vs sklearn q*100/100*(n-1) (1 ULP high); tracking #2370"]
fn divergence_regressor_quantile_rounding_order() {
    // Oracle value bit-pattern from the LIVE sklearn 1.5.2 call above
    // (`float(...).hex()` == '0x1.3ffffffffffffp+4'), NOT copied from ferrolearn.
    let sklearn_constant: f64 = f64::from_bits(0x4033_ffff_ffff_ffff);
    // Self-check that the literal bit-pattern really is the oracle value.
    assert_eq!(sklearn_constant, 19.999_999_999_999_996_f64);

    let y: Array1<f64> = array![10.0, 20.0, 30.0, 40.0];
    let ferro_constant = reg_value(DummyRegressorStrategy::Quantile(1.0 / 3.0), &y);

    // np.percentile is bit-deterministic for this input; assert bit-exact parity.
    assert_eq!(
        ferro_constant.to_bits(),
        sklearn_constant.to_bits(),
        "quantile=1/3 on [10,20,30,40]: sklearn={sklearn_constant:.17} (bits {:#018x}), \
         ferrolearn={ferro_constant:.17} (bits {:#018x})",
        sklearn_constant.to_bits(),
        ferro_constant.to_bits(),
    );
}

// ---------------------------------------------------------------------------
// REQ-3: stratified / uniform — STRUCTURAL guards only. RNG carve-out
// (R-DEFER-3): SmallRng != numpy RandomState, so exact per-sample parity with
// sklearn is impossible and is NOT asserted. Blocker #1695. These guards only
// pin that every drawn label is a valid class and that uniform covers the
// observed class set on large n — no bit-matching.
// ---------------------------------------------------------------------------

/// Structural: stratified predictions are always valid observed classes.
/// sklearn draws via `rs.multinomial` (`sklearn/dummy.py:317-323,383-385`) on
/// numpy RandomState; ferrolearn uses SmallRng — distributional, not exact.
/// Tracking: #1695 (RNG carve-out, no exact-parity test).
#[test]
fn structural_stratified_in_class_set() {
    let y = array![0usize, 0, 0, 1, 1, 2];
    let fitted = DummyClassifier::new(DummyClassifierStrategy::Stratified)
        .random_state(7)
        .fit(&zeros(6), &y)
        .unwrap();
    let preds = fitted.predict(&zeros(200)).unwrap();
    let classes = fitted.classes().to_vec();
    assert!(preds.iter().all(|p| classes.contains(p)));
}

/// Structural: uniform predictions are valid classes and, on large n, cover the
/// full observed class set. sklearn uses `rs.randint` (`sklearn/dummy.py:325-330`);
/// ferrolearn uses SmallRng `slice::choose` — distributional, not exact.
/// Tracking: #1695 (RNG carve-out, no exact-parity test).
#[test]
fn structural_uniform_covers_class_set() {
    let y = array![0usize, 1, 2];
    let fitted = DummyClassifier::new(DummyClassifierStrategy::Uniform)
        .random_state(5)
        .fit(&zeros(3), &y)
        .unwrap();
    let preds = fitted.predict(&zeros(300)).unwrap();
    let classes = fitted.classes().to_vec();
    assert!(preds.iter().all(|p| classes.contains(p)));
    for c in &classes {
        assert!(
            preds.iter().any(|p| p == c),
            "uniform should cover class {c} over 300 draws"
        );
    }
}
