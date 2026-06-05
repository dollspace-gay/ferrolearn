//! Adversarial divergence audit of `cross_validation.rs` vs scikit-learn 1.5.2.
//!
//! This file is the critic artifact for the ACToR loop. It contains:
//!
//! * GREEN guards for the SHIPPED REQs (REQ-KFOLD / REQ-CVS / REQ-CVALIDATE /
//!   REQ-CVPREDICT / REQ-PERM). These pass today and FAIL if the SHIPPED claim
//!   regresses. Every expected value is derived from a LIVE sklearn 1.5.2 call
//!   (recorded inline next to the assertion), never copied from the ferrolearn
//!   side (R-CHAR-3).
//! * FAILING `#[ignore]`'d pins for the DETERMINISTIC FIXABLE divergences
//!   (#1790 error_score-continue, #1791 StratifiedKFold allocation, #1792
//!   StratifiedKFold error-vs-warn, #1793 cross_val_predict non-partition).
//!   Each pin is oracle-derived and currently FAILS against ferrolearn.
//!
//! Oracle provenance (sklearn 1.5.2, run from /tmp):
//!   KFold(3).split(zeros(10))                      -> [[0,1,2,3],[4,5,6],[7,8,9]]
//!   StratifiedKFold(3).split(_, [2,2,2,2,0,0,0,0,1,1,1,1])
//!                                                  -> [[0,1,4,8],[2,5,6,9],[3,7,10,11]]
//!   StratifiedKFold(3).split(_, [0]*5+[1]*2)       -> SUCCEEDS (UserWarning), 3 folds
//!   cross_val_score(mean, zeros(10,2), 0..9, KFold(3), negMSE)
//!                                                  -> [-26.25, -1.1768707483, -25.6666666667]
//!   cross_validate(...).train_score (negMSE)       -> [-2.9166666667, -11.3469387755, -4.0]
//!   cross_val_predict(mean, zeros(10,2), 0..9, KFold(3))
//!                                                  -> [6.5,6.5,6.5,6.5, 4.2857142857 x3, 3.0 x3]
//!   cross_val_score(G_failing_on_X0==99, ...)      -> [-9.25, nan, nan] (any nan: True)

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{
    FittedPipelineEstimator as FittedEstTrait, Pipeline, PipelineEstimator,
};
use ferrolearn_model_sel::cross_validation::{FoldSplits, permutation_test_score};
use ferrolearn_model_sel::{
    CrossValidator, KFold, StratifiedKFold, cross_val_predict, cross_val_score, cross_validate,
};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Test helpers: a mean-predicting estimator == sklearn DummyRegressor(strategy='mean').
// ---------------------------------------------------------------------------

/// Predicts the mean of `y_train` for every test row.
struct MeanEstimator;

impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedEstTrait<f64>>, FerroError> {
        let mean = y.mean().unwrap_or(0.0);
        Ok(Box::new(FittedMean { mean }))
    }
}

struct FittedMean {
    mean: f64,
}

impl FittedEstTrait<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

/// A mean estimator that mirrors the AC-ERROR-SCORE oracle estimator `G`:
/// it raises (returns `Err`) when fitting on any train row whose first feature
/// equals the sentinel `99.0`. sklearn's default `error_score=np.nan` NaN-fills
/// such a fold and CONTINUES; ferrolearn `?`-propagates.
struct FailOnSentinelEstimator;

impl PipelineEstimator<f64> for FailOnSentinelEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedEstTrait<f64>>, FerroError> {
        if x.column(0).iter().any(|&v| v == 99.0) {
            return Err(FerroError::InvalidParameter {
                name: "x".into(),
                reason: "boom (sentinel 99.0 in train fold)".into(),
            });
        }
        let mean = y.mean().unwrap_or(0.0);
        Ok(Box::new(FittedMean { mean }))
    }
}

fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-(diff.mapv(|v| v * v).mean().unwrap_or(0.0)))
}

fn mean_pipeline() -> Pipeline {
    Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
}

const NEG_MSE_TOL: f64 = 1e-9;

// ===========================================================================
// SHIPPED GREEN GUARDS (must PASS; FAIL only on regression)
// ===========================================================================

/// REQ-KFOLD (SHIPPED). Live oracle:
/// `KFold(3, shuffle=False).split(np.zeros((10,1)))`
/// -> test folds `[[0,1,2,3],[4,5,6],[7,8,9]]`
/// (first `n % k` folds get `+1`; folds are consecutive slices).
/// Guards `sklearn/model_selection/_split.py:521-535`.
#[test]
fn guard_kfold_membership_n10_k3() {
    let folds = KFold::new(3).split(10);
    let test_folds: Vec<Vec<usize>> = folds.iter().map(|(_, t)| t.clone()).collect();
    let expected = vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    assert_eq!(test_folds, expected);
}

/// REQ-KFOLD boundary. `KFold::new(1)` rejected (`n_splits < 2`),
/// `KFold::new(5)` on 3 samples rejected (`n_samples < n_splits`).
/// sklearn raises in both cases; ferrolearn returns `Err`.
#[test]
fn guard_kfold_invalid_params() {
    assert!(KFold::new(1).fold_indices(10).is_err());
    assert!(KFold::new(5).fold_indices(3).is_err());
}

/// REQ-CVS (SHIPPED). Live oracle (DummyRegressor strategy='mean' == MeanEstimator):
/// `cross_val_score(mean, zeros(10,2), y=0..9, cv=KFold(3), scoring='neg_mean_squared_error')`
/// -> `[-26.25, -1.1768707483, -25.6666666667]`.
/// Mirrors `cross_validate -> test_score` (`_validation.py:560`, `:122`).
#[test]
fn guard_cross_val_score_per_fold_negmse() {
    let x = Array2::<f64>::zeros((10, 2));
    let y: Array1<f64> = Array1::from_iter((0..10).map(f64::from));
    let scores = cross_val_score(&mean_pipeline(), &x, &y, &KFold::new(3), neg_mse).unwrap();
    // sklearn 1.5.2 oracle values:
    let expected = [-26.25_f64, -1.176_870_748_3, -25.666_666_666_7];
    assert_eq!(scores.len(), 3);
    for (got, exp) in scores.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < NEG_MSE_TOL,
            "per-fold negMSE diverged: got {got}, sklearn {exp}"
        );
    }
}

/// REQ-CVALIDATE (SHIPPED). Live oracle:
/// `cross_validate(mean, zeros(10,2), 0..9, KFold(3), negMSE, return_train_score=True)`
/// -> test_score `[-26.25, -1.1768707483, -25.6666666667]`,
///    train_score `[-2.9166666667, -11.3469387755, -4.0]`.
/// fit_times / score_times present and non-negative.
/// Mirrors `_validation.py:122`.
#[test]
fn guard_cross_validate_train_test_and_timing() {
    let x = Array2::<f64>::zeros((10, 2));
    let y: Array1<f64> = Array1::from_iter((0..10).map(f64::from));
    let r = cross_validate(&mean_pipeline(), &x, &y, &KFold::new(3), neg_mse, true).unwrap();

    let expected_test = [-26.25_f64, -1.176_870_748_3, -25.666_666_666_7];
    let expected_train = [-2.916_666_666_7_f64, -11.346_938_775_5, -4.0];

    assert_eq!(r.test_scores.len(), 3);
    for (got, exp) in r.test_scores.iter().zip(expected_test.iter()) {
        assert!(
            (got - exp).abs() < NEG_MSE_TOL,
            "test negMSE: got {got}, sklearn {exp}"
        );
    }
    let train = r.train_scores.expect("return_train_score=true => Some");
    assert_eq!(train.len(), 3);
    for (got, exp) in train.iter().zip(expected_train.iter()) {
        assert!(
            (got - exp).abs() < NEG_MSE_TOL,
            "train negMSE: got {got}, sklearn {exp}"
        );
    }
    assert_eq!(r.fit_times.len(), 3);
    assert_eq!(r.score_times.len(), 3);
    for &t in r.fit_times.iter().chain(r.score_times.iter()) {
        assert!(
            t >= 0.0 && t.is_finite(),
            "timing must be non-negative & finite, got {t}"
        );
    }

    // return_train_score=false => train_scores is None.
    let r2 = cross_validate(&mean_pipeline(), &x, &y, &KFold::new(3), neg_mse, false).unwrap();
    assert!(r2.train_scores.is_none());
}

/// REQ-CVPREDICT (SHIPPED, partition case). Live oracle:
/// `cross_val_predict(mean, zeros(10,2), 0..9, KFold(3))`
/// -> `[6.5,6.5,6.5,6.5, 4.2857142857,4.2857142857,4.2857142857, 3.0,3.0,3.0]`.
/// Each sample's OOF prediction = mean of its fold's TRAIN y, placed at the
/// sample's ORIGINAL index. The three distinct fold means (6.5 / 4.2857 / 3.0)
/// make original-order placement DISTINGUISHABLE. Mirrors `_validation.py:1054`.
#[test]
fn guard_cross_val_predict_original_order_placement() {
    let x = Array2::<f64>::zeros((10, 2));
    let y: Array1<f64> = Array1::from_iter((0..10).map(f64::from));
    let preds = cross_val_predict(&mean_pipeline(), &x, &y, &KFold::new(3)).unwrap();
    let f1 = 4.285_714_285_7_f64;
    let expected = [6.5, 6.5, 6.5, 6.5, f1, f1, f1, 3.0, 3.0, 3.0];
    assert_eq!(preds.len(), 10);
    for (i, (got, exp)) in preds.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < NEG_MSE_TOL,
            "OOF prediction misplaced at index {i}: got {got}, sklearn {exp}"
        );
    }
}

/// REQ-PERM (SHIPPED, p-value FORMULA only — the exact perm scores are an RNG
/// carve-out, REQ-SHUFFLE-RNG/#1795). sklearn `_validation.py:1697`:
/// `pvalue = (sum(permutation_scores >= score) + 1) / (n_permutations + 1)`.
/// This guard recomputes the formula from ferrolearn's OWN returned
/// (real_score, perm_scores) and asserts ferrolearn's returned p_value equals
/// it — pinning that ferrolearn applies sklearn's `+1` / `>=` formula (NOT a
/// tautology: the expected side is the sklearn FORMULA applied to the live
/// outputs, and would catch e.g. a `>` instead of `>=`, or a missing `+1`).
#[test]
fn guard_permutation_test_score_pvalue_formula() {
    let x = Array2::<f64>::zeros((20, 3));
    let y: Array1<f64> = Array1::from_iter((0..20).map(f64::from));
    let n_perm = 20;
    let (real, perm, p) = permutation_test_score(
        &mean_pipeline(),
        &x,
        &y,
        &KFold::new(5),
        neg_mse,
        n_perm,
        Some(0),
    )
    .unwrap();
    assert_eq!(perm.len(), n_perm);
    // sklearn p-value formula (file:line 1697), recomputed from ferrolearn's own
    // real/perm outputs:
    let n_ge = perm.iter().filter(|&&s| s >= real).count();
    let sklearn_formula_p = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
    assert!(
        (p - sklearn_formula_p).abs() < 1e-12,
        "p_value formula diverged: ferrolearn {p}, sklearn-formula {sklearn_formula_p}"
    );
    assert!((0.0..=1.0).contains(&p));
}

// ===========================================================================
// FAILING PINS for DETERMINISTIC FIXABLE divergences (#[ignore], must FAIL)
// ===========================================================================

/// #1790 — error_score=np.nan continue (THE KEY cross-unit divergence).
/// Live oracle (AC-ERROR-SCORE): estimator `G` raises iff a train row has
/// `X[:,0]==99`; with `X[0,0]=99`, `y=0..5`, `KFold(3)`,
/// `cross_val_score(G, X, y, scoring='neg_mean_squared_error')`
/// -> `[-9.25, nan, nan]`, `np.isnan(s).any() == True`.
/// sklearn `_fit_and_score` (default `error_score=np.nan`) NaN-fills the
/// failing fold and CONTINUES (`_validation.py:729`, `:890-915`).
/// ferrolearn `cross_val_score` `?`-PROPAGATES the fit failure => returns `Err`,
/// not a NaN-bearing array. This pin asserts the sklearn contract (Ok + a NaN
/// in the failing fold) and so FAILS today.
#[test]
fn pin_1790_error_score_nan_continue() {
    // n=6, KFold(3) folds (non-shuffled): test [0,1],[2,3],[4,5].
    // Sentinel at sample 0 => folds 1 and 2 TRAIN on sample 0 (fail); fold 0
    // trains on samples 2..5 (no sentinel) and succeeds.
    let mut x = Array2::<f64>::zeros((6, 2));
    for i in 0..6 {
        x[[i, 0]] = (2 * i) as f64;
        x[[i, 1]] = (2 * i + 1) as f64;
    }
    x[[0, 0]] = 99.0;
    let y: Array1<f64> = Array1::from_iter((0..6).map(f64::from));

    let pipeline = Pipeline::new().estimator_step("g", Box::new(FailOnSentinelEstimator));
    let result = cross_val_score(&pipeline, &x, &y, &KFold::new(3), neg_mse);

    // sklearn returns Ok([-9.25, nan, nan]) (any nan == True), NOT Err.
    let scores = result.expect("sklearn returns Ok with a NaN-bearing array, not Err (#1790)");
    assert_eq!(scores.len(), 3);
    assert!(
        scores.iter().any(|s| s.is_nan()),
        "sklearn NaN-fills the failing fold; ferrolearn must too (#1790), got {scores:?}"
    );
    // Fold 0 (succeeds): train = samples {2,3,4,5}, y mean = 3.5; test y = {0,1};
    // negMSE = -((0-3.5)^2 + (1-3.5)^2)/2 = -9.25 (matches sklearn oracle).
    assert!(
        (scores[0] + 9.25).abs() < NEG_MSE_TOL,
        "succeeding fold 0 negMSE should be -9.25 (sklearn oracle), got {}",
        scores[0]
    );
}

/// #1791 — StratifiedKFold non-shuffled allocation diverges.
/// Live oracle (AC-SKFOLD), recomputed here:
/// `y=[2,2,2,2,0,0,0,0,1,1,1,1]` (classes appear in order 2,0,1),
/// `StratifiedKFold(3, shuffle=False).split(zeros(12,1), y)`
/// -> test folds `[[0,1,4,8],[2,5,6,9],[3,7,10,11]]`.
/// sklearn encodes classes by APPEARANCE order (`_split.py:760-765`) and
/// allocates via `bincount(y_order[i::n_splits])` (`:786-792`) + block-assign
/// (`:798-805`). ferrolearn sorts classes LEXICOGRAPHICALLY + rotating
/// `fold_offset` round-robin => `[[0,4,5,8],[1,6,9,10],[2,3,7,11]]`. DIVERGES.
#[test]
fn pin_1791_stratified_kfold_allocation() {
    let y: Array1<usize> = Array1::from(vec![2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]);
    let folds = StratifiedKFold::new(3).split(&y).unwrap();
    let mut test_folds: Vec<Vec<usize>> = folds.iter().map(|(_, t)| t.clone()).collect();
    for t in &mut test_folds {
        t.sort_unstable();
    }
    // sklearn 1.5.2 live oracle:
    let expected = vec![vec![0, 1, 4, 8], vec![2, 5, 6, 9], vec![3, 7, 10, 11]];
    assert_eq!(test_folds, expected);
}

/// #1792 — StratifiedKFold error-vs-warn divergence.
/// Live oracle (AC-SKFOLD-ERRWARN): `y=[0,0,0,0,0,1,1]` (class 1 has 2 < 3,
/// class 0 has 5 >= 3), `StratifiedKFold(3).split(zeros(7,1), y)` SUCCEEDS with
/// a UserWarning (3 folds). sklearn raises ValueError ONLY when EVERY class
/// count < n_splits (`_split.py:770-774`) and merely WARNS otherwise
/// (`:775-781`). ferrolearn returns `Err(InsufficientSamples)` as soon as ANY
/// class count < n_splits. This pin asserts the split SUCCEEDS (3 folds), so it
/// FAILS today.
#[test]
fn pin_1792_stratified_kfold_one_small_class_warns_not_errors() {
    let y: Array1<usize> = Array1::from(vec![0, 0, 0, 0, 0, 1, 1]);
    let result = StratifiedKFold::new(3).split(&y);
    let folds = result
        .expect("sklearn warns+splits when only ONE class is too small; must not Err (#1792)");
    assert_eq!(folds.len(), 3, "sklearn yields 3 folds");
    // Every sample appears in exactly one test fold (partition).
    let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
    all_test.sort_unstable();
    assert_eq!(all_test, (0..7).collect::<Vec<_>>());
}

/// A non-partition `CrossValidator`: it OMITS sample 0 from every test fold, so
/// sample 0 is never predicted out-of-fold. sklearn `cross_val_predict` requires
/// a partition and raises `ValueError("cross_val_predict only works for
/// partitions")` (`_validation.py:1054`) for exactly this case.
struct NonPartitionCv;

impl CrossValidator for NonPartitionCv {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        // Two folds over samples 1..n_samples (sample 0 is never a test sample);
        // sample 0 is always in train. This is a valid CrossValidator signature
        // but NOT a partition of 0..n_samples.
        let half = (n_samples - 1) / 2 + 1; // split point within 1..n
        let test_a: Vec<usize> = (1..half).collect();
        let test_b: Vec<usize> = (half..n_samples).collect();
        let train_a: Vec<usize> = (0..n_samples).filter(|i| !test_a.contains(i)).collect();
        let train_b: Vec<usize> = (0..n_samples).filter(|i| !test_b.contains(i)).collect();
        Ok(vec![(train_a, test_a), (train_b, test_b)])
    }
}

/// #1793 — cross_val_predict non-partition divergence.
/// sklearn inverts the test indices and REQUIRES a partition, raising
/// `ValueError("cross_val_predict only works for partitions")`
/// (`_validation.py:1054`) when some sample is never in a test fold.
/// ferrolearn initializes predictions to `Array1::zeros(n)` and only overwrites
/// test positions, so an omitted sample SILENTLY keeps `0.0` and the call
/// returns `Ok`. The `NonPartitionCv` above omits sample 0; this pin asserts
/// ferrolearn returns `Err` (the sklearn contract), so it FAILS today.
#[test]
#[ignore = "divergence: cross_val_predict 0.0-fills omitted samples on a non-partition cv; sklearn raises ValueError; tracking #1793"]
fn pin_1793_cross_val_predict_non_partition_must_error() {
    let x = Array2::<f64>::zeros((10, 2));
    let y: Array1<f64> = Array1::from_iter((0..10).map(f64::from));
    let result = cross_val_predict(&mean_pipeline(), &x, &y, &NonPartitionCv);
    assert!(
        result.is_err(),
        "sklearn raises ValueError on a non-partition cv (sample 0 never predicted); \
         ferrolearn must too (#1793), got Ok with sample 0 = {:?}",
        result.map(|p| p[0])
    );
}
