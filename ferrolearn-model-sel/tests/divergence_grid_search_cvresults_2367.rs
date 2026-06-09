//! Live-oracle audit (#2367 / REQ-CVRESULTS #1778) of the two DETERMINISTIC
//! `cv_results_` aggregations newly built into `CvResults`:
//! `std_test_score` (population std, ddof=0, over the per-fold scores) and
//! `rank_test_score` (`rankdata(-mean_test_score, method="min")`).
//!
//! Every expected value is a LIVE sklearn 1.5.2 oracle value (recorded inline
//! next to the assertion), NEVER copied from the ferrolearn side (R-CHAR-3).
//! Ridge + a real r2 scorer keep the fixtures fully deterministic.
//!
//! Fixture A (shared, byte-for-byte, with `divergence_grid_search_2367.rs`):
//! ```python
//! import numpy as np
//! from sklearn.model_selection import GridSearchCV, KFold
//! from sklearn.linear_model import Ridge
//! X = np.array([[float(i), float((i*7)%5)] for i in range(12)])
//! y = np.array([2.0*i - 3.0*((i*7)%5) + 1.0 for i in range(12)])
//! gs = GridSearchCV(Ridge(), {'alpha':[0.1,1.0,10.0,100.0]},
//!                   cv=KFold(4), scoring='r2', refit=False).fit(X, y)
//! gs.cv_results_['std_test_score'].tolist()  ->
//!   [4.4190093880323866e-05, 0.004068485825057903,
//!    0.2226956684745082, 2.732530068274452]
//! gs.cv_results_['rank_test_score'].tolist() -> [1, 2, 3, 4]
//! ```
//!
//! Fixture B (a deterministic TIE — `alpha=1.0` duplicated, so candidates 0 and
//! 1 are identical and share a mean, exercising `method="min"`):
//! ```python
//! Xb = np.array([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0]])
//! yb = np.array([1.0,2.0,1.5,3.0,2.5,4.0,3.5,5.0])
//! gb = GridSearchCV(Ridge(), {'alpha':[1.0,1.0,5.0]},
//!                   cv=KFold(4), scoring='r2', refit=False).fit(Xb, yb)
//! gb.cv_results_['mean_test_score'].tolist() ->
//!   [0.38014977551152207, 0.38014977551152207, 0.4775165416106446]
//! gb.cv_results_['std_test_score'].tolist()  ->
//!   [0.25915377214159324, 0.25915377214159324, 0.14267839178980288]
//! gb.cv_results_['rank_test_score'].tolist() -> [2, 2, 1]
//! ```
//! sklearn `_format_results._store`: std `:1112-1117`, rank `:1119-1132`.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_linear::Ridge;
use ferrolearn_model_sel::{GridSearchCV, KFold, ParamSet, ParamValue, param_grid};
use ndarray::{Array1, Array2};

const TOL: f64 = 1e-7;

/// Fixture A: the deterministic 12-row fixture shared with
/// `divergence_grid_search_2367.rs`.
fn fixture_a() -> (Array2<f64>, Array1<f64>) {
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

/// Fixture B: a deterministic tie — chosen so two candidates produce identical
/// per-fold scores (and hence an identical mean and std).
fn fixture_b() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("fixture_b shape");
    let y = Array1::from(vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0]);
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

/// sklearn `r2_score` (`sklearn/metrics/_regression.py:1146`): `1 - SS_res/SS_tot`.
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
// std_test_score — population std (ddof=0) over the 4 folds, vs live sklearn
// ---------------------------------------------------------------------------

/// `cv_results_["std_test_score"]` must equal the population std (ddof=0) of the
/// per-fold scores for each candidate. Mirrors sklearn
/// `array_stds = np.sqrt(np.average((array - array_means[:, None])**2, axis=1))`
/// (`sklearn/model_selection/_search.py:1112-1117`) — NumPy's `np.std` default
/// divides by `N`, not `N-1`.
#[test]
fn grid_ridge_kfold4_r2_std_test_score() {
    let (x, y) = fixture_a();
    let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64, 10.0_f64, 100.0_f64] };
    let mut gs = GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), r2);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();

    // Live sklearn 1.5.2 oracle (see header).
    let sk_std = [
        4.419_009_388_032_386_6e-5_f64,
        0.004_068_485_825_057_903,
        0.222_695_668_474_508_2,
        2.732_530_068_274_452,
    ];
    let std = results.std_test_score();
    assert_eq!(std.len(), 4, "one std per candidate");
    assert_eq!(std.len(), results.std_scores.len());
    for (i, (got, exp)) in std.iter().zip(sk_std.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "candidate {i} std_test_score diverged: ferrolearn {got}, sklearn {exp}"
        );
    }
}

// ---------------------------------------------------------------------------
// rank_test_score — rankdata(-mean, method="min"), exact ints, vs live sklearn
// ---------------------------------------------------------------------------

/// `cv_results_["rank_test_score"]` must equal
/// `rankdata(-mean_test_score, method="min")` — rank 1 = highest mean.
/// Strictly decreasing means here → ranks `[1, 2, 3, 4]`. Mirrors
/// `sklearn/model_selection/_search.py:1129`.
#[test]
fn grid_ridge_kfold4_r2_rank_test_score() {
    let (x, y) = fixture_a();
    let grid = param_grid! { "alpha" => [0.1_f64, 1.0_f64, 10.0_f64, 100.0_f64] };
    let mut gs = GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), r2);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();

    // Live sklearn 1.5.2 oracle: alpha=0.1 best (rank 1) … alpha=100.0 worst (4).
    const SK_RANK: [usize; 4] = [1, 2, 3, 4];
    assert_eq!(
        results.rank_test_score(),
        &SK_RANK,
        "rank_test_score diverged from sklearn rankdata(-mean, 'min')"
    );
    // The rank-1 candidate is the best index (cross-check with best_index).
    assert_eq!(results.best_index(), Some(0));
}

// ---------------------------------------------------------------------------
// TIE — method="min": tied means share the lowest rank, next distinct skips
// ---------------------------------------------------------------------------

/// A deterministic tie (candidates 0 and 1 are identical: duplicate
/// `alpha=1.0`) must reproduce scipy `method="min"`: both tied candidates get
/// the SAME (lowest) rank of their group, and the distinct better candidate's
/// rank skips by the tie-group size.
///
/// Live sklearn 1.5.2 oracle (see header):
/// means `[0.38015, 0.38015, 0.47752]` → std identical for the tied pair →
/// ranks `[2, 2, 1]` (alpha=5.0 best at rank 1; the two alpha=1.0 share rank 2).
#[test]
fn grid_ridge_tie_rankdata_min_and_std() {
    let (x, y) = fixture_b();
    let grid = param_grid! { "alpha" => [1.0_f64, 1.0_f64, 5.0_f64] };
    let mut gs = GridSearchCV::new(Box::new(ridge_factory()), grid, Box::new(KFold::new(4)), r2);
    gs.fit(&x, &y).unwrap();

    let results = gs.cv_results().unwrap();

    // sklearn mean_test_score: candidates 0 and 1 tie.
    let sk_mean = [
        0.380_149_775_511_522_07_f64,
        0.380_149_775_511_522_07,
        0.477_516_541_610_644_6,
    ];
    for (i, (got, exp)) in results.mean_scores.iter().zip(sk_mean.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "tie candidate {i} mean diverged: ferrolearn {got}, sklearn {exp}"
        );
    }

    // sklearn std_test_score: the tied pair share an identical std.
    let sk_std = [
        0.259_153_772_141_593_24_f64,
        0.259_153_772_141_593_24,
        0.142_678_391_789_802_88,
    ];
    let std = results.std_test_score();
    for (i, (got, exp)) in std.iter().zip(sk_std.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "tie candidate {i} std diverged: ferrolearn {got}, sklearn {exp}"
        );
    }

    // sklearn rank_test_score with method="min": [2, 2, 1].
    const SK_RANK: [usize; 3] = [2, 2, 1];
    assert_eq!(
        results.rank_test_score(),
        &SK_RANK,
        "tie rank_test_score diverged from sklearn rankdata(-mean, 'min')"
    );
    // best_index picks the FIRST occurrence of the rank-1 candidate (index 2).
    assert_eq!(results.best_index(), Some(2));
}
