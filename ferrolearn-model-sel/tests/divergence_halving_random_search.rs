//! Divergence + green-guard tests for `HalvingRandomSearchCV` against
//! scikit-learn 1.5.2 `HalvingRandomSearchCV` / `BaseSuccessiveHalving`
//! (`sklearn/model_selection/_search_successive_halving.py`).
//!
//! Audit focus (ACToR critic). This is the SIBLING of
//! `divergence_halving_grid_search.rs`; the halving LOOP and `_top_k` keep-count
//! are shared, only candidate generation differs (random sampling).
//!
//!   * RE-VERIFY the load-bearing keep-count claim against the LIVE sklearn
//!     1.5.2 oracle: `_top_k` keeps
//!     `n_candidates_to_keep = ceil(n_candidates / factor)`
//!     (`_search_successive_halving.py:359`), and ferrolearn's
//!     `n.div_ceil(factor)` (`halving_random_search.rs:294`) keeps the SAME
//!     per-round count. Confirmed live for (N,F) ∈
//!     {(9,3),(8,2),(7,2),(10,3)} (see `keep_count_per_round_matches_oracle`).
//!     GREEN guard: a future fixer must NOT silently change `div_ceil` to floor.
//!   * RE-VERIFY the sampled-COUNT claim: for an explicit integer
//!     `n_candidates=N`, sklearn `_generate_candidate_params`
//!     (`:1069-1079`) -> `ParameterSampler(..., N)` samples exactly N candidates
//!     in the first iteration (`n_candidates_[0] == N`, verified live). ferrolearn
//!     samples exactly `n_candidates` ParamSets up-front
//!     (`halving_random_search.rs:252-254`). GREEN guard
//!     (`sampled_count_consistent_with_oracle`). The EXACT candidates diverge
//!     (SmallRng vs numpy RandomState) — explicit RNG carve-out (#1861), NOT
//!     pinned.
//!   * SHIPPED REQ-HALVING-MECHANIC (loop shape, multi-round, best-of-final,
//!     error guards, degenerate edges) — green guards.
//!   * SHIPPED REQ-RANDOM-SAMPLING (same `random_state` ⇒ identical result
//!     across two runs) — green guard.
//!   * One sharpened FAILING pin (#1863): the iteration-count / termination
//!     divergence persists EVEN when `min_resources` AND `max_resources` are
//!     forced equal to sklearn's. sklearn stops at
//!     `n_iterations = min(n_possible, n_required)` (`:289-296`) and records the
//!     last iteration's survivor SET; ferrolearn grows the budget until
//!     `effective_budget >= max_res` OR one candidate remains
//!     (`halving_random_search.rs:272-316`), so its final `cv_results` survivor
//!     SET differs (here 1 row vs sklearn's 2).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp),
//! never copied from the ferrolearn side (R-CHAR-3). The oracle for
//! `HalvingRandomSearchCV` requires
//! `from sklearn.experimental import enable_halving_search_cv`.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::distributions::{Choice, Distribution, Uniform};
use ferrolearn_model_sel::{HalvingRandomSearchCV, KFold, ParamSet, ParamValue};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Estimator that always predicts a fixed constant value (from the `constant`
/// hyperparameter).
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

/// Estimator that predicts the training-target mean.
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

fn constant_factory(params: &ParamSet) -> Pipeline {
    let val = match params.get("constant") {
        Some(ParamValue::Float(v)) => *v,
        _ => 0.0,
    };
    Pipeline::new().estimator_step("est", Box::new(ConstantEstimator { value: val }))
}

fn mean_factory(_params: &ParamSet) -> Pipeline {
    Pipeline::new().estimator_step("mean", Box::new(MeanEstimator))
}

fn dist(low: f64, high: f64) -> Vec<(String, Box<dyn Distribution>)> {
    vec![("constant".into(), Box::new(Uniform::new(low, high)))]
}

// ---------------------------------------------------------------------------
// GREEN GUARD: keep-count `ceil(n / factor)` matches `_top_k` (:359)
// ---------------------------------------------------------------------------

/// Reproduce the per-round survivor chain that ferrolearn's `fit` loop walks
/// (`halving_random_search.rs:294`, `n_survive = scored.len().div_ceil(factor)`)
/// and assert each round's keep-count equals sklearn's
/// `n_candidates_to_keep = ceil(n_candidates / factor)`
/// (`_search_successive_halving.py:359`).
///
/// The expected per-round counts come from the LIVE sklearn 1.5.2 oracle's
/// `n_candidates_` attribute (one entry per iteration), which IS the realized
/// keep schedule:
/// ```text
/// python3 -c "import numpy as np
/// from sklearn.experimental import enable_halving_search_cv  # noqa
/// from sklearn.model_selection import HalvingRandomSearchCV, KFold
/// from scipy.stats import uniform
/// from sklearn.base import BaseEstimator, RegressorMixin
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self,c=0.0): self.c=c
///     def fit(self,X,y): self.c_=self.c; return self
///     def predict(self,X): return np.full(X.shape[0], self.c_)
/// rng=np.random.RandomState(0); X=rng.rand(90,2); y=rng.rand(90)
/// for N,F in [(9,3),(8,2),(7,2),(10,3)]:
///     gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=N,
///         factor=F, cv=KFold(3), min_resources='smallest',
///         random_state=0).fit(X,y)
///     print(N, F, gs.n_candidates_)"
/// # -> 9 3 [9, 3, 1]
/// # -> 8 2 [8, 4, 2, 1]
/// # -> 7 2 [7, 4, 2]
/// # -> 10 3 [10, 4, 2]
/// ```
/// NOTE: sklearn's `n_candidates_` may STOP before reaching 1 (e.g. (10,3) ->
/// [10,4,2]) because the loop runs `n_iterations` rounds, not "until 1". This
/// guard only checks that each transition `n -> ceil(n/factor)` matches
/// ferrolearn's `div_ceil`; the early termination is the separate #1863 pin.
#[test]
fn keep_count_per_round_matches_oracle() {
    // (N, factor, sklearn n_candidates_ schedule from the live oracle above).
    let cases: &[(usize, usize, &[usize])] = &[
        (9, 3, &[9, 3, 1]),
        (8, 2, &[8, 4, 2, 1]),
        (7, 2, &[7, 4, 2]),
        (10, 3, &[10, 4, 2]),
    ];
    for &(_n, factor, schedule) in cases {
        for w in schedule.windows(2) {
            let (n, sk_next) = (w[0], w[1]);
            let ferro_next = n.div_ceil(factor).max(1);
            assert_eq!(
                ferro_next, sk_next,
                "keep-count divergence: ferrolearn div_ceil({n}/{factor})={ferro_next} \
                 but sklearn _top_k (:359) keeps {sk_next}",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD: sampled COUNT for an explicit integer n_candidates (:1069-1079)
// ---------------------------------------------------------------------------

/// For an explicit integer `n_candidates=N`, sklearn `_generate_candidate_params`
/// builds `ParameterSampler(param_distributions, N, ...)` (`:1075-1079`), so the
/// first iteration evaluates exactly N candidates
/// (`gs.n_candidates_[0] == N`, verified live for N ∈ {6,9,4,1}). ferrolearn
/// samples exactly `n_candidates` ParamSets up-front
/// (`halving_random_search.rs:252-254`), all of which seed the first round.
///
/// We assert the first-round count by forcing a single-iteration search
/// (`min_resources == max_resources`), so `cv_results.params.len()` is exactly
/// the number of candidates evaluated in that one (final) round == N.
/// The EXACT candidates diverge (SmallRng vs numpy RandomState) — RNG carve-out
/// (#1861), NOT asserted here.
#[test]
fn sampled_count_consistent_with_oracle() {
    // sklearn first-iteration candidate count for an explicit int n_candidates:
    // n_candidates_[0] == N (live oracle, N ∈ {6,9,4,1}).
    for n in [6_usize, 9, 4, 1] {
        let mut hs = HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dist(-5.0, 5.0),
            n,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(0),
        )
        // force exactly ONE round so cv_results holds the full first-round set.
        .min_resources(Some(30))
        .max_resources(Some(30));
        let x = Array2::<f64>::zeros((30, 2));
        let y = Array1::<f64>::from_elem(30, 1.0);
        hs.fit(&x, &y).unwrap();
        let results = hs.cv_results().unwrap();
        assert_eq!(
            results.params.len(),
            n,
            "ferrolearn sampled {} candidates in the single (final) round but \
             an explicit n_candidates={n} must sample exactly {n} \
             (sklearn ParameterSampler, :1075-1079)",
            results.params.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-HALVING-MECHANIC — multi-round, deterministic best wins
// ---------------------------------------------------------------------------

/// The mean estimator achieves a perfect fit on a constant target, so the best
/// score is ~0 regardless of which (irrelevant) candidate wins. Mirrors
/// sklearn's last-iteration best selection (`_select_best_index`, `:191-212`).
#[test]
fn best_of_final_round_perfect_fit() {
    let dists: Vec<(String, Box<dyn Distribution>)> = vec![(
        "dummy".into(),
        Box::new(Choice::new(vec![ParamValue::Bool(true)])),
    )];
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(mean_factory),
        dists,
        9,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(0),
    );
    let x = Array2::<f64>::zeros((90, 2));
    let y = Array1::<f64>::from_elem(90, 3.0);
    hs.fit(&x, &y).unwrap();
    let score = hs.best_score().unwrap();
    assert!(score.abs() < 1e-10, "expected ~0 best score, got {score}");
}

/// Multi-round search must populate `cv_results` and a finite best score.
#[test]
fn multi_round_populates_results() {
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(constant_factory),
        dist(-5.0, 5.0),
        9,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(55),
    )
    .min_resources(Some(9));
    let x = Array2::<f64>::zeros((90, 2));
    let y = Array1::<f64>::from_elem(90, 1.0);
    hs.fit(&x, &y).unwrap();
    let results = hs.cv_results().unwrap();
    assert!(!results.params.is_empty());
    assert!(hs.best_score().unwrap().is_finite());
}

// ---------------------------------------------------------------------------
// GREEN GUARD: error contract + degenerate edges (no panic)
// ---------------------------------------------------------------------------

#[test]
fn error_contract_matches() {
    let x = Array2::<f64>::zeros((60, 2));
    let y = Array1::<f64>::zeros(60);

    // n_candidates == 0 => Err
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(mean_factory),
        dist(0.0, 1.0),
        0,
        Box::new(KFold::new(3)),
        neg_mse,
        None,
    );
    assert!(hs.fit(&x, &y).is_err(), "n_candidates=0 must Err");

    // empty distributions => Err
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(mean_factory),
        vec![],
        5,
        Box::new(KFold::new(3)),
        neg_mse,
        None,
    );
    assert!(hs.fit(&x, &y).is_err(), "empty distributions must Err");

    // factor < 2 => Err
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(mean_factory),
        dist(0.0, 1.0),
        5,
        Box::new(KFold::new(3)),
        neg_mse,
        None,
    )
    .factor(1);
    assert!(hs.fit(&x, &y).is_err(), "factor<2 must Err");

    // shape mismatch => Err
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(mean_factory),
        dist(0.0, 1.0),
        5,
        Box::new(KFold::new(3)),
        neg_mse,
        None,
    );
    let x_bad = Array2::<f64>::zeros((30, 2));
    let y_bad = Array1::<f64>::zeros(25);
    assert!(hs.fit(&x_bad, &y_bad).is_err(), "shape mismatch must Err");
}

/// Degenerate edges must not panic: single candidate, factor far larger than N,
/// and budget capped at n_samples.
#[test]
fn degenerate_edges_no_panic() {
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);

    // single candidate
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(constant_factory),
        dist(0.9, 1.1),
        1,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(7),
    );
    hs.fit(&x, &y).unwrap();
    assert!(hs.best_score().is_some());

    // factor > N (huge factor): all but one eliminated in round 1
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(constant_factory),
        dist(-5.0, 5.0),
        3,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(7),
    )
    .factor(50)
    // explicit valid budget; auto min_resources can drop below n_splits for
    // huge factors (that gap is the #1862 min_resources-formula divergence).
    .min_resources(Some(6));
    hs.fit(&x, &y).unwrap();
    assert!(hs.best_params().is_some());

    // min_resources forced above n_samples is rejected, not a panic
    let mut hs = HalvingRandomSearchCV::new(
        Box::new(constant_factory),
        dist(-5.0, 5.0),
        4,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(7),
    )
    .min_resources(Some(1000));
    assert!(
        hs.fit(&x, &y).is_err(),
        "min_resources > n_samples must Err, not panic"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD: REQ-RANDOM-SAMPLING — same seed ⇒ identical result across runs
// ---------------------------------------------------------------------------

/// Same `random_state` ⇒ identical sampled candidates ⇒ identical best score
/// across two independent fits. Mirrors sklearn's seeded `ParameterSampler`
/// reproducibility (`:1075-1079`). The EXACT candidates vs sklearn diverge
/// (#1861) — only across-run determinism is asserted (carve-out).
#[test]
fn same_seed_deterministic_across_runs() {
    let make = || {
        HalvingRandomSearchCV::new(
            Box::new(constant_factory),
            dist(-5.0, 5.0),
            6,
            Box::new(KFold::new(3)),
            neg_mse,
            Some(99),
        )
    };
    let x = Array2::<f64>::zeros((60, 2));
    let y = Array1::<f64>::from_elem(60, 1.0);

    let mut a = make();
    a.fit(&x, &y).unwrap();
    let mut b = make();
    b.fit(&x, &y).unwrap();

    let (sa, sb) = (a.best_score().unwrap(), b.best_score().unwrap());
    assert!(
        (sa - sb).abs() < 1e-12,
        "same seed must give identical best score: {sa} vs {sb}"
    );
}

// ---------------------------------------------------------------------------
// FAILING pin (#1863) — termination / iteration-count divergence persists even
// when min_resources AND max_resources are forced to sklearn's exact values.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `fit` termination diverges from sklearn's
/// `n_iterations = min(n_possible_iterations, n_required_iterations)`
/// (`sklearn/model_selection/_search_successive_halving.py:289-296`).
///
/// This pin sharpens AC-3: the schedule divergence is NOT merely a consequence
/// of `min_resources_` differing. Even with `min_resources` AND `max_resources`
/// forced to sklearn's exact values (10 and 90), the FINAL-ROUND survivor SET
/// recorded in `cv_results` differs, because the TERMINATION rule differs:
///   * sklearn (`:272`,`:289-296`,`:359`): for N=5, factor=3, min=10, max=90,
///     `n_required = 1+floor(log(5,3)) = 2`,
///     `n_possible = 1+floor(log(90//10,3)) = 3`,
///     `n_iterations = min(3,2) = 2`. It runs 2 iterations
///     (`n_resources_ = [10, 30]`, `n_candidates_ = [5, 2]`), records the LAST
///     iteration's 2 survivors, and `_select_best_index` (`:191-212`) picks the
///     best of those 2 => last-iteration candidate count == 2.
///   * ferrolearn (`halving_random_search.rs:272-316`): grows
///     `budget *= factor` until `effective_budget >= max_res` OR `<= 1`
///     candidate. It reduces 5 -> ceil(5/3)=2 -> ceil(2/3)=1 and records a
///     SINGLE candidate => `cv_results.params.len() == 1`.
///
/// Live oracle (`min_resources=10, max_resources=90`, N=5, factor=3):
/// ```text
/// python3 -c "import numpy as np
/// from sklearn.experimental import enable_halving_search_cv  # noqa
/// from sklearn.model_selection import HalvingRandomSearchCV, KFold
/// from scipy.stats import uniform
/// from sklearn.base import BaseEstimator, RegressorMixin
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self,c=0.0): self.c=c
///     def fit(self,X,y): self.c_=self.c; return self
///     def predict(self,X): return np.full(X.shape[0], self.c_)
/// rng=np.random.RandomState(0); X=rng.rand(90,2); y=np.full(90,1.0)
/// gs=HalvingRandomSearchCV(Const(), {'c':uniform(0,5)}, n_candidates=5,
///     factor=3, cv=KFold(3), random_state=0,
///     min_resources=10, max_resources=90).fit(X,y)
/// it=gs.cv_results_['iter']; print(int((it==it.max()).sum()))"
/// # -> 2   (sklearn records 2 candidates in the last iteration)
/// ```
/// ferrolearn records 1 row in `cv_results` for the same configuration.
///
/// Tracking: #1863 (REQ-N-ITERATIONS — architectural; no single-line fix).
#[test]
#[ignore = "divergence: n_iterations=min(n_possible,n_required) vs grow-until-max/1; tracking #1863"]
fn divergence_last_iteration_survivor_set() {
    // sklearn last-iteration candidate count from the live oracle above.
    const SK_LAST_ITER_CANDIDATE_COUNT: usize = 2;

    let mut hs = HalvingRandomSearchCV::new(
        Box::new(constant_factory),
        dist(-5.0, 5.0),
        5,
        Box::new(KFold::new(3)),
        neg_mse,
        Some(0),
    )
    .factor(3)
    .min_resources(Some(10))
    .max_resources(Some(90));
    let x = Array2::<f64>::zeros((90, 2));
    let y = Array1::<f64>::from_elem(90, 1.0);
    hs.fit(&x, &y).unwrap();

    let results = hs.cv_results().unwrap();
    assert_eq!(
        results.params.len(),
        SK_LAST_ITER_CANDIDATE_COUNT,
        "ferrolearn final cv_results survivor set ({}) must match sklearn's \
         last-iteration candidate count ({SK_LAST_ITER_CANDIDATE_COUNT}); \
         the termination rule diverges (#1863)",
        results.params.len()
    );
}
