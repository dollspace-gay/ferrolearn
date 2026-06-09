//! Adversarial divergence audit (#2367) of `GridSearchCV` aggregation
//! (mean_test_score, best_params_, best_score_, best_index_, multi-param grid
//! order) using a REAL DETERMINISTIC estimator (Ridge) + a real r2 scorer,
//! pushing beyond the constant/mean predictors in `divergence_grid_search.rs`.
//!
//! Every expected value below is a LIVE sklearn 1.5.2 oracle value (recorded
//! inline next to the assertion), NEVER copied from the ferrolearn side
//! (R-CHAR-3).
//!
//! Shared fixture (deterministic, replicated exactly in Rust):
//! ```python
//! X = np.array([[float(i), float((i*7)%5)] for i in range(12)])
//! y = np.array([2.0*i - 3.0*((i*7)%5) + 1.0 for i in range(12)])
//! gs = GridSearchCV(Ridge(), {'alpha':[0.1,1.0,10.0,100.0]},
//!                   cv=KFold(4), scoring='r2').fit(X, y)
//! gs.best_params_          -> {'alpha': 0.1}
//! gs.best_score_           -> 0.9999359323698909
//! gs.best_index_           -> 0
//! gs.cv_results_['mean_test_score'] ->
//!     [0.9999359323698909, 0.9941175943483055, 0.6933050582503746, -2.1130524083301108]
//! gs.cv_results_['std_test_score']  ->
//!     [4.4190093880323866e-05, 0.004068485825057903, 0.2226956684745082, 2.732530068274452]
//! gs.cv_results_['rank_test_score'] -> [1, 2, 3, 4]
//! ```

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_linear::Ridge;
use ferrolearn_model_sel::{GridSearchCV, KFold, ParamSet, ParamValue, param_grid};
use ndarray::{Array1, Array2};

const TOL: f64 = 1e-7;

fn fixture() -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((12, 2));
    let mut y = Array1::<f64>::zeros(12);
    for i in 0..12 {
        let f0 = i as f64;
        let f1 = ((i * 7) % 5) as f64;
        x[[i, 0]] = f0;
        x[[i, 1]] = f1;
        y[i] = 2.0 * f0 - 3.0 * f1 + 1.0;
    }
    (x, y)
}

/// Factory: build a Ridge pipeline whose `alpha` is taken from the param set.
fn ridge_factory() -> impl Fn(&ParamSet) -> Pipeline {
    move |params: &ParamSet| {
        let alpha = match params.get("alpha") {
            Some(ParamValue::Float(v)) => *v,
            _ => 1.0,
        };
        Pipeline::new().estimator_step("ridge", Box::new(Ridge::<f64>::new().with_alpha(alpha)))
    }
}

/// sklearn `r2_score` (`sklearn/metrics/_regression.py:1146`):
/// `1 - SS_res/SS_tot`, `SS_tot = sum((y - mean(y))^2)`.
fn r2(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();
    Ok(1.0 - ss_res / ss_tot)
}

// ---------------------------------------------------------------------------
// GREEN guard — CORE mean_test_score + best_params_ + best_score_ (Ridge r2)
// ---------------------------------------------------------------------------

/// CORE: `GridSearchCV(Ridge(), {'alpha':[...]}, cv=KFold(4), scoring='r2')`
/// must reproduce sklearn's `mean_test_score` array (per-candidate mean over
/// the 4 folds), `best_params_`, `best_score_`, and `best_index_`.
///
/// Live sklearn 1.5.2 oracle (see header). Mirrors `_format_results`
/// `mean_test_score = np.average(array, axis=1)` (`_search.py:1097`) +
/// `best_index_ = rank_test_score.argmin()` (`:840`).
#[test]
fn grid_ridge_kfold4_r2_core_aggregation() {
    let (x, y) = fixture();
    let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64, 10.0_f64, 100.0_f64] };
    let mut gs = GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), r2);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();

    // sklearn mean_test_score (per-candidate mean over the 4 folds).
    let sk_mean = [
        0.999_935_932_369_890_9_f64,
        0.994_117_594_348_305_5,
        0.693_305_058_250_374_6,
        -2.113_052_408_330_110_8,
    ];
    assert_eq!(results.mean_scores.len(), 4);
    for (i, (got, exp)) in results.mean_scores.iter().zip(sk_mean.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "candidate {i} mean_test_score diverged: ferrolearn {got}, sklearn {exp}"
        );
    }

    // sklearn best_index_ == 0, best_params_ == {'alpha': 0.1}, best_score_ == max mean.
    const SK_BEST_INDEX: usize = 0;
    const SK_BEST_ALPHA: f64 = 0.1;
    const SK_BEST_SCORE: f64 = 0.999_935_932_369_890_9;
    assert_eq!(results.best_index(), Some(SK_BEST_INDEX));
    assert_eq!(
        gs.best_params().unwrap().get("alpha"),
        Some(&ParamValue::Float(SK_BEST_ALPHA))
    );
    assert!(
        (gs.best_score().unwrap() - SK_BEST_SCORE).abs() < TOL,
        "best_score_ diverged: ferrolearn {}, sklearn {SK_BEST_SCORE}",
        gs.best_score().unwrap()
    );
}

// ---------------------------------------------------------------------------
// GREEN guard — multi-param cartesian ORDER (alpha slowest, last key fastest)
// ---------------------------------------------------------------------------

/// Multi-param grid `{'alpha':[0.1,1.0], 'fit_intercept':[True,False]}` must be
/// enumerated in sklearn `ParameterGrid` order: keys sorted, LAST key varies
/// fastest, FIRST key slowest.
///
/// Live sklearn 1.5.2 oracle:
/// ```python
/// list(ParameterGrid({'alpha':[0.1,1.0], 'fit_intercept':[True,False]}))
///  -> [{'alpha':0.1,'fit_intercept':True}, {'alpha':0.1,'fit_intercept':False},
///      {'alpha':1.0,'fit_intercept':True}, {'alpha':1.0,'fit_intercept':False}]
/// ```
/// (`_search.py:157` `sorted(p.items())` + `itertools.product`.)
#[test]
fn grid_multi_param_cartesian_order() {
    let grid = param_grid! {
        "alpha" => [0.1_f64, 1.0_f64],
        "fit_intercept" => [true, false],
    };
    // sklearn order of (alpha, fit_intercept) pairs.
    let sk_order: [(f64, bool); 4] = [(0.1, true), (0.1, false), (1.0, true), (1.0, false)];
    assert_eq!(grid.len(), 4);
    for (i, exp) in sk_order.iter().enumerate() {
        let a = match grid[i].get("alpha") {
            Some(ParamValue::Float(v)) => *v,
            other => panic!("alpha not Float: {other:?}"),
        };
        let fi = match grid[i].get("fit_intercept") {
            Some(ParamValue::Bool(b)) => *b,
            other => panic!("fit_intercept not Bool: {other:?}"),
        };
        assert_eq!(
            (a, fi),
            *exp,
            "candidate {i} order diverged from sklearn ParameterGrid"
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN guard — neg_mean_squared_error sign in the best selection
// ---------------------------------------------------------------------------

/// scoring='neg_mean_squared_error' produces NEGATIVE means; the best candidate
/// is the LEAST negative (closest to 0) = the MAX. `GridSearchCV` must pick that
/// candidate (not the most-negative MSE).
///
/// Live sklearn 1.5.2 oracle:
/// ```python
/// GridSearchCV(Ridge(), {'alpha':[0.1,1.0,10.0,100.0]}, cv=KFold(4),
///              scoring='neg_mean_squared_error').fit(X, y)
/// best_params_ -> {'alpha': 0.1}
/// best_score_  -> -0.0009951441930713782   (the max / least negative)
/// best_index_  -> 0
/// mean_test_score -> [-0.0009951441930713782, -0.09142883411642172,
///                     -4.7595867157350735, -46.618163387786794]
/// ```
/// (`_search.py:840` rank by `-mean`; `:1097` mean over folds.)
#[test]
fn grid_neg_mse_picks_least_negative() {
    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-(diff.mapv(|v| v * v).mean().unwrap_or(0.0)))
    }
    let (x, y) = fixture();
    let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64, 10.0_f64, 100.0_f64] };
    let mut gs =
        GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), neg_mse);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();
    let sk_mean = [
        -0.000_995_144_193_071_378_2_f64,
        -0.091_428_834_116_421_72,
        -4.759_586_715_735_073_5,
        -46.618_163_387_786_794,
    ];
    for (i, (got, exp)) in results.mean_scores.iter().zip(sk_mean.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "candidate {i} neg_mse mean diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
    // Best = least-negative (index 0), NOT the most-negative.
    assert_eq!(results.best_index(), Some(0));
    assert_eq!(
        gs.best_params().unwrap().get("alpha"),
        Some(&ParamValue::Float(0.1))
    );
    const SK_BEST_SCORE: f64 = -0.000_995_144_193_071_378_2;
    assert!((gs.best_score().unwrap() - SK_BEST_SCORE).abs() < TOL);
}

// ---------------------------------------------------------------------------
// GREEN guard — single candidate + all-tied edges
// ---------------------------------------------------------------------------

/// Edge: a single-candidate grid -> best_index_ == 0, best_score_ == its mean.
/// Live sklearn 1.5.2 oracle:
/// ```python
/// GridSearchCV(Ridge(), {'alpha':[1.0]}, cv=KFold(4), scoring='r2').fit(X, y)
/// best_index_ -> 0; best_params_ -> {'alpha': 1.0}
/// best_score_ -> 0.9941175943483055
/// ```
#[test]
fn grid_single_candidate_best_index_zero() {
    let (x, y) = fixture();
    let grid = param_grid! { "alpha" => [1.0_f64] };
    let mut gs = GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), r2);
    gs.fit(&x, &y).unwrap();
    let results = gs.cv_results().unwrap();
    assert_eq!(results.best_index(), Some(0));
    const SK_BEST_SCORE: f64 = 0.994_117_594_348_305_5;
    assert!((gs.best_score().unwrap() - SK_BEST_SCORE).abs() < TOL);
}
