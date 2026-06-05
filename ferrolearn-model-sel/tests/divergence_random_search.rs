//! Divergence + green-guard tests for `RandomizedSearchCV` against scikit-learn
//! 1.5.2 `sklearn.model_selection.RandomizedSearchCV` / `ParameterSampler`
//! (`sklearn/model_selection/_search.py`: RandomizedSearchCV `:1576`,
//! `_run_search` `:1958`, `ParameterSampler` `:216`, `_is_all_lists` `:302`,
//! `__iter__` `:308-341`, `__len__` `:343-349`; `BaseSearchCV.fit`
//! `_select_best_index` `:840` / unweighted mean `:1097`).
//!
//! - GREEN guards pin the SHIPPED behaviors (REQ-1 continuous with-replacement
//!   count, REQ-2 seed determinism across runs, REQ-BESTIDX-MEAN first-on-tie /
//!   unweighted mean through the shared `CvResults`). They PASS today and would
//!   FAIL if the claim regressed.
//! - `green_1784_choice_with_replacement_no_cap` documents the CURRENT
//!   all-discrete with-replacement count (no cap, dups) — the architectural
//!   #1784 carve-out (the #1774 reclassification precedent: green-guard the
//!   current behavior because the faithful fix needs a `Distribution`-trait
//!   cardinality extension, not a single-file edit).
//! - `divergence_empty_param_distributions` is an `#[ignore]`'d FAILING pin: an
//!   UNCLAIMED divergence where sklearn runs 1 empty candidate but ferrolearn
//!   errors. Tracking #1788 (new-blocker).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp),
//! never copied from the ferrolearn side (R-CHAR-3).

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::distributions::{Choice, Distribution, Uniform};
use ferrolearn_model_sel::{KFold, ParamSet, ParamValue, RandomizedSearchCV};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Estimator that always predicts a fixed constant value.
struct ConstantEstimator {
    value: f64,
}
struct FittedConstant {
    value: f64,
}
impl PipelineEstimator<f64> for ConstantEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedConstant { value: self.value }))
    }
}
impl FittedPipelineEstimator<f64> for FittedConstant {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.value))
    }
}

/// Estimator that predicts the training mean.
struct MeanEstimator;
struct FittedMean {
    mean: f64,
}
impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMean {
            mean: y.mean().unwrap_or(0.0),
        }))
    }
}
impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

/// Negative MSE (higher is better), matching sklearn `neg_mean_squared_error`.
fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

fn mean_factory() -> impl Fn(&ParamSet) -> Pipeline {
    |_p: &ParamSet| Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-1: continuous with-replacement count == n_iter
// ---------------------------------------------------------------------------

/// Green guard: REQ-1 — a CONTINUOUS distribution + `n_iter=N` yields exactly
/// `N` cross-validated candidates (sklearn's WITH-replacement branch for
/// distributions that have `rvs`, `ParameterSampler.__iter__` ELSE branch
/// `sklearn/model_selection/_search.py:330-341`; `__len__` returns `n_iter`
/// `:349`).
///
/// Oracle (live sklearn 1.5.2, from /tmp):
/// ```text
/// from scipy.stats import uniform
/// from sklearn.model_selection import ParameterSampler
/// len(list(ParameterSampler({'a': uniform(0,1)}, n_iter=7, random_state=42))) -> 7
/// ```
#[test]
fn green_req1_continuous_with_replacement_count() {
    // Oracle: continuous dist with n_iter=7 yields 7 candidates (with replacement).
    const SK_N_ITER: usize = 7;
    const SK_N_CANDIDATES: usize = 7;
    const SK_N_FOLDS: usize = 3;

    let dists: Vec<(String, Box<dyn Distribution>)> =
        vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))];
    let mut rs = RandomizedSearchCV::new(
        Box::new(mean_factory()),
        dists,
        SK_N_ITER,
        Box::new(KFold::new(SK_N_FOLDS)),
        neg_mse,
        Some(42),
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);
    rs.fit(&x, &y).unwrap();

    let results = rs.cv_results().unwrap();
    assert_eq!(
        results.params.len(),
        SK_N_CANDIDATES,
        "continuous dist must yield n_iter candidates (with replacement)"
    );
    // Each candidate cross-validated on n_folds.
    assert_eq!(results.all_scores.len(), SK_N_CANDIDATES);
    for folds in &results.all_scores {
        assert_eq!(folds.len(), SK_N_FOLDS);
    }
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-2: seed determinism across runs (structural)
// ---------------------------------------------------------------------------

/// Green guard: REQ-2 — two independent searches with the SAME `random_state`
/// draw identical `ParamSet`s; DIFFERENT seeds draw (very likely) different
/// ones. This is the `random_state` determinism CONTRACT, NOT numpy-exact
/// value equality (that is the #1786 R-DEFER-3 carve-out).
///
/// Oracle (live sklearn 1.5.2, from /tmp — the determinism contract):
/// ```text
/// from scipy.stats import uniform
/// from sklearn.model_selection import ParameterSampler
/// a=[round(d['a'],9) for d in ParameterSampler({'a':uniform(0,1)},n_iter=5,random_state=99)]
/// b=[round(d['a'],9) for d in ParameterSampler({'a':uniform(0,1)},n_iter=5,random_state=99)]
/// c=[round(d['a'],9) for d in ParameterSampler({'a':uniform(0,1)},n_iter=5,random_state=123)]
/// (a==b, a!=c) -> (True, True)
/// ```
#[test]
fn green_req2_seed_determinism_across_runs() {
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);

    let run = |seed: u64| -> Vec<f64> {
        let dists: Vec<(String, Box<dyn Distribution>)> =
            vec![("alpha".into(), Box::new(Uniform::new(0.0, 1.0)))];
        let mut rs = RandomizedSearchCV::new(
            Box::new(mean_factory()),
            dists,
            5,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(seed),
        );
        rs.fit(&x, &y).unwrap();
        rs.cv_results()
            .unwrap()
            .params
            .iter()
            .map(|p| match p.get("alpha") {
                Some(ParamValue::Float(v)) => *v,
                other => panic!("expected Float alpha, got {other:?}"),
            })
            .collect()
    };

    // Oracle: same seed -> identical sequence.
    let a = run(99);
    let b = run(99);
    assert_eq!(
        a, b,
        "same random_state must produce identical sampled params"
    );

    // Oracle: different seed -> (very likely) different sequence.
    let c = run(123);
    assert_ne!(
        a, c,
        "different random_state should produce a different sampled sequence"
    );
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-BESTIDX-MEAN: first-on-tie via shared CvResults
// ---------------------------------------------------------------------------

/// Green guard: REQ-BESTIDX-MEAN — through the `RandomizedSearchCV` path,
/// `best_index`/`best_params`/`best_score` use the FIXED first-on-tie reduction
/// of the SHARED `CvResults` (`grid_search.rs`). A single-value `Choice` makes
/// all `n_iter` candidates identical -> tied means -> first-on-tie must select
/// index 0.
///
/// sklearn `_select_best_index`: `best_index_ = rank_test_score.argmin()`
/// (`sklearn/model_selection/_search.py:840`); `np.argmin` returns the FIRST
/// minimum -> lowest-index candidate wins on a tie. Mean is unweighted
/// (`np.average(..., weights=None)`, `:1097`).
///
/// Oracle (live sklearn 1.5.2, from /tmp — tie -> argmin index 0):
/// ```text
/// rs=RandomizedSearchCV(Const(), {'c':[4.0,6.0]}, n_iter=2, cv=KFold(3),
///     scoring='neg_mean_squared_error', refit=False, random_state=0).fit(zeros,full(5.0))
/// rs.cv_results_['rank_test_score'].tolist() -> [1, 1]  (tie); argmin -> 0
/// ```
/// Here ferrolearn's tied-candidate analog: `Choice([1.0])` with `n_iter=4`
/// gives 4 identical candidates with equal means -> best_index must be 0.
#[test]
fn green_reqbestidx_first_on_tie_through_random_search() {
    let factory = |params: &ParamSet| {
        let val = match params.get("c") {
            Some(ParamValue::Float(v)) => *v,
            _ => 0.0,
        };
        Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
    };
    // Single-option Choice -> every sampled candidate is c=1.0 -> tied means.
    let dists: Vec<(String, Box<dyn Distribution>)> = vec![(
        "c".into(),
        Box::new(Choice::new(vec![ParamValue::Float(1.0)])),
    )];
    let mut rs = RandomizedSearchCV::new(
        Box::new(factory),
        dists,
        4,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(0),
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);
    rs.fit(&x, &y).unwrap();

    let results = rs.cv_results().unwrap();
    // Precondition: all four means are tied (otherwise this is not a tie test).
    let m0 = results.mean_scores[0];
    for &m in &results.mean_scores {
        assert!(
            (m - m0).abs() < 1e-12,
            "precondition: all means must tie, got {:?}",
            results.mean_scores
        );
    }

    // Oracle: on a tie, sklearn argmin picks the FIRST (lowest) index.
    const SK_BEST_INDEX: usize = 0;
    assert_eq!(
        results.best_index(),
        Some(SK_BEST_INDEX),
        "shared CvResults must pick first-on-tie (index 0) through random_search"
    );

    // best_score is the (tied) mean, ~0 for c==1.0 on y==1.0.
    let score = rs.best_score().unwrap();
    assert!(score.abs() < 1e-10, "expected ~0 best_score, got {score}");
}

// ---------------------------------------------------------------------------
// GREEN guard — REQ-BESTIDX NaN-worst reaches random_search
// ---------------------------------------------------------------------------

/// Green guard: a NaN-bearing candidate mean is treated as TIED-WORST through
/// the `RandomizedSearchCV` path — a finite candidate always wins, regardless
/// of sampling position. Mirrors sklearn `nan_to_num(array_means,
/// nan=nanmin-1)` before `rankdata` (`sklearn/model_selection/_search.py:1127-1129`).
///
/// Construction: a `Choice([5.0, 999.0])`; the estimator predicts NaN for
/// c==999.0, finite for c==5.0 (on y==5.0 -> score 0.0). With many iterations
/// both values are sampled; the finite candidate must be `best_index`.
///
/// (Determinism note: with `random_state=Some(7)` and `n_iter=12` over the
/// two-option Choice, both 5.0 and 999.0 are sampled — asserted as a
/// precondition. The oracle property is "finite beats NaN", verified live via
/// the GridSearchCV analog in `divergence_grid_search.rs::best_index_nan_mean_tied_worst`.)
#[test]
fn green_reqbestidx_nan_worst_through_random_search() {
    struct MaybeNan {
        c: f64,
    }
    struct FittedMaybeNan {
        c: f64,
    }
    impl PipelineEstimator<f64> for MaybeNan {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
            Ok(Box::new(FittedMaybeNan { c: self.c }))
        }
    }
    impl FittedPipelineEstimator<f64> for FittedMaybeNan {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            let v = if self.c == 999.0 { f64::NAN } else { self.c };
            Ok(Array1::from_elem(x.nrows(), v))
        }
    }

    let dists: Vec<(String, Box<dyn Distribution>)> = vec![(
        "c".into(),
        Box::new(Choice::new(vec![
            ParamValue::Float(5.0),
            ParamValue::Float(999.0),
        ])),
    )];
    let mut rs = RandomizedSearchCV::new(
        Box::new(|params: &ParamSet| {
            let val = match params.get("c") {
                Some(ParamValue::Float(v)) => *v,
                _ => 0.0,
            };
            Pipeline::new().estimator_step("est", Box::new(MaybeNan { c: val }))
        }),
        dists,
        12,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(7),
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 5.0);
    rs.fit(&x, &y).unwrap();

    let results = rs.cv_results().unwrap();
    // Precondition: at least one finite AND at least one NaN mean were sampled.
    let has_finite = results.mean_scores.iter().any(|m| m.is_finite());
    let has_nan = results.mean_scores.iter().any(|m| m.is_nan());
    assert!(
        has_finite && has_nan,
        "precondition: both a finite and a NaN candidate must be sampled, got {:?}",
        results.mean_scores
    );

    // Oracle property: the finite candidate (mean 0.0) must win; best_index
    // must point at a finite mean, never a NaN one.
    let best = results.best_index().unwrap();
    assert!(
        results.mean_scores[best].is_finite(),
        "best_index {best} must point at a finite mean (NaN tied-worst), means {:?}",
        results.mean_scores
    );
    assert!(
        rs.best_score().unwrap().abs() < 1e-10,
        "best_score must be the finite ~0 candidate, got {:?}",
        rs.best_score()
    );
}

// ---------------------------------------------------------------------------
// GREEN guard documenting CURRENT #1784 behavior (architectural carve-out)
// ---------------------------------------------------------------------------

/// Green guard documenting the CURRENT (divergent-from-sklearn) all-discrete
/// count behavior — the #1784 architectural carve-out (#1774 reclassification
/// precedent: green-guard current behavior where the faithful fix is not a
/// single-file edit).
///
/// sklearn `_is_all_lists()` branch (`sklearn/model_selection/_search.py:313-328`)
/// samples WITHOUT replacement and CAPS `n_iter = min(n_iter, grid_size)` with a
/// `UserWarning`; `__len__` -> `min(n_iter, grid_size)` (`:345-347`). For a
/// 3-option list + `n_iter=15`, sklearn yields exactly 3 DISTINCT candidates:
/// ```text
/// from sklearn.model_selection import ParameterSampler
/// len(list(ParameterSampler({'c':[1,2,3]}, n_iter=15, random_state=55))) -> 3  (+ UserWarning)
/// ```
///
/// ferrolearn has NO discreteness/cardinality on the `Distribution` trait
/// (`distributions.rs` exposes only `fn sample`), so `random_search.rs` cannot
/// detect the all-lists case or compute `grid_size`. It ALWAYS loops `0..n_iter`
/// with replacement, so a `Choice([1,2,3])` + `n_iter=15` yields 15 (dups, no
/// cap, no warning). DOCUMENTED, NOT a single-file fixable divergence -> #1784.
///
/// This guard PASSES (asserts ferrolearn's current 15) and would FAIL if the
/// behavior silently changed (e.g. a partial without-replacement attempt) so
/// the #1784 status stays honest.
#[test]
fn green_1784_choice_with_replacement_no_cap() {
    let options = vec![ParamValue::Int(1), ParamValue::Int(2), ParamValue::Int(3)];
    let dists: Vec<(String, Box<dyn Distribution>)> =
        vec![("c".into(), Box::new(Choice::new(options)))];
    let mut rs = RandomizedSearchCV::new(
        Box::new(mean_factory()),
        dists,
        15,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(55),
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);
    rs.fit(&x, &y).unwrap();

    // sklearn (oracle) would CAP this to grid_size=3 with a UserWarning; #1784.
    const SK_CAPPED_DISTINCT: usize = 3;
    // ferrolearn CURRENT behavior: with replacement, no cap -> 15.
    const FERRO_CURRENT_WITH_REPLACEMENT: usize = 15;
    let n = rs.cv_results().unwrap().params.len();
    assert_ne!(
        n, SK_CAPPED_DISTINCT,
        "if this ever equals sklearn's capped distinct count, #1784 changed"
    );
    assert_eq!(
        n, FERRO_CURRENT_WITH_REPLACEMENT,
        "ferrolearn currently samples all-discrete WITH replacement (no cap); #1784"
    );
}

// ---------------------------------------------------------------------------
// FAILING pin — UNCLAIMED divergence: empty param_distributions (#1788)
// ---------------------------------------------------------------------------

/// Divergence (UNCLAIMED by the design doc): sklearn `RandomizedSearchCV` with an
/// EMPTY parameter grid `{}` runs successfully with exactly ONE trivial (empty)
/// candidate — `_is_all_lists()` is vacuously True for `{}`, `grid_size = 1`, so
/// `n_iter` is capped to `min(n_iter, 1) = 1` and a single empty `ParamSet` is
/// yielded (`sklearn/model_selection/_search.py:313-328`, `:345-347`).
///
/// ferrolearn `RandomizedSearchCV::fit` (`random_search.rs:131-136`) instead
/// REJECTS an empty `param_distributions` with `FerroError::InvalidParameter`,
/// never running the search.
///
/// Oracle (live sklearn 1.5.2, from /tmp):
/// ```text
/// from sklearn.model_selection import RandomizedSearchCV, KFold
/// rs=RandomizedSearchCV(Const(), {}, n_iter=3, cv=KFold(3),
///     scoring='neg_mean_squared_error', refit=False, random_state=0).fit(X,y)
/// rs.cv_results_['params']          -> [{}]
/// len(rs.cv_results_['params'])     -> 1
/// (also: ParameterSampler({}, n_iter=3, random_state=0) -> [{}], len 1, + UserWarning)
/// ```
/// sklearn returns 1 candidate; ferrolearn returns an error. This is a
/// DETERMINISTIC count divergence on the empty-grid edge. Tracking #1788
/// (new-blocker).
#[test]
fn divergence_empty_param_distributions() {
    // sklearn (oracle): empty grid -> exactly ONE empty candidate.
    const SK_N_CANDIDATES: usize = 1;

    let dists: Vec<(String, Box<dyn Distribution>)> = Vec::new();
    let mut rs = RandomizedSearchCV::new(
        Box::new(mean_factory()),
        dists,
        3,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(0),
    );
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);

    // sklearn fits successfully; ferrolearn currently returns Err here.
    rs.fit(&x, &y)
        .expect("sklearn runs an empty grid as 1 candidate; ferrolearn must not error");

    let results = rs
        .cv_results()
        .expect("after a successful fit, cv_results must be Some");
    assert_eq!(
        results.params.len(),
        SK_N_CANDIDATES,
        "sklearn yields exactly 1 empty candidate for an empty grid"
    );
}
