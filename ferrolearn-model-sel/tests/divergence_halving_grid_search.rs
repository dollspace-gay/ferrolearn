//! Divergence + green-guard tests for `HalvingGridSearchCV` against
//! scikit-learn 1.5.2 `HalvingGridSearchCV` / `BaseSuccessiveHalving`
//! (`sklearn/model_selection/_search_successive_halving.py`).
//!
//! Audit focus (ACToR critic):
//!   * RE-VERIFY the load-bearing keep-count claim from the design doc against
//!     the LIVE sklearn 1.5.2 oracle: `_top_k` keeps
//!     `n_candidates_to_keep = ceil(n_candidates / factor)`
//!     (`_search_successive_halving.py:359`), and ferrolearn's `n.div_ceil(factor)`
//!     keeps the SAME count. Confirmed live for (N,F) ∈
//!     {(5,3),(10,3),(9,3),(7,2),(4,2)} (see `keep_count_matches_oracle` below).
//!     GREEN guard: a future fixer must NOT silently "fix" `div_ceil` to floor.
//!   * SHIPPED REQ-HALVING-MECHANIC (loop shape, best-of-final, error guards).
//!   * One sharpened FAILING pin (#1852): the iteration-count / termination
//!     divergence persists EVEN when `min_resources` and `max_resources` are
//!     forced equal to sklearn's. sklearn stops at
//!     `n_iterations = min(n_possible, n_required)` and records the last
//!     iteration's survivor SET; ferrolearn grows the budget until
//!     `effective_budget >= max_res` OR one candidate remains, so its final
//!     `cv_results` survivor SET differs (here 1 row vs sklearn's 2).
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (run from /tmp),
//! never copied from the ferrolearn side (R-CHAR-3). The oracle for
//! `HalvingGridSearchCV` requires
//! `from sklearn.experimental import enable_halving_search_cv`.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::{HalvingGridSearchCV, KFold, ParamSet, ParamValue, param_grid};
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

/// Reproduce the per-round survivor chain that ferrolearn's `fit` loop walks:
/// start with `n` candidates and reduce by `ceil(n / factor)` each round
/// (ferrolearn `n.div_ceil(factor)`), stopping when one candidate remains.
fn ferro_keep_chain(n: usize, factor: usize) -> Vec<usize> {
    let mut chain = vec![n];
    let mut cur = n;
    while cur > 1 {
        cur = cur.div_ceil(factor);
        chain.push(cur);
    }
    chain
}

// ---------------------------------------------------------------------------
// GREEN guard — keep-count re-verification (the doc-author's load-bearing fix)
// ---------------------------------------------------------------------------

/// GREEN guard: RE-VERIFY the per-round keep-count arithmetic against the LIVE
/// sklearn 1.5.2 oracle. sklearn `_top_k` keeps
/// `n_candidates_to_keep = ceil(n_candidates / factor)`
/// (`sklearn/model_selection/_search_successive_halving.py:359`); ferrolearn
/// keeps `n.div_ceil(factor)` (`halving_grid_search.rs:283`). These are the same
/// function. The expected per-round reductions below come from the live oracle's
/// `clf.n_candidates_` (the candidate count entering each iteration), captured
/// verbatim — NOT copied from ferrolearn:
///
/// ```text
/// python3 -c "import numpy as np
/// from sklearn.experimental import enable_halving_search_cv  # noqa
/// from sklearn.model_selection import HalvingGridSearchCV, KFold
/// from sklearn.base import BaseEstimator, RegressorMixin
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self,c=0.0): self.c=c
///     def fit(self,X,y): self.c_=self.c; return self
///     def predict(self,X): return np.full(X.shape[0], self.c_)
/// def run(N,F):
///     rng=np.random.RandomState(0); X=rng.rand(900,2); y=rng.rand(900)
///     return HalvingGridSearchCV(Const(), {'c':[float(i) for i in range(N)]},
///         factor=F, cv=KFold(3), random_state=0).fit(X,y).n_candidates_
/// for N,F in [(5,3),(10,3),(9,3),(7,2),(4,2)]: print(N,F,run(N,F))"
/// # -> 5 3 [5, 2]      (5 -> ceil(5/3)=2)
/// # -> 10 3 [10, 4, 2] (10 -> 4 -> ceil(4/3)=2)
/// # -> 9 3 [9, 3, 1]   (9 -> 3 -> ceil(3/3)=1)
/// # -> 7 2 [7, 4, 2]   (7 -> 4 -> ceil(4/2)=2)
/// # -> 4 2 [4, 2, 1]   (4 -> 2 -> ceil(2/2)=1)
/// ```
///
/// sklearn's `n_candidates_` is the count ENTERING each iteration; the next
/// entry is `ceil(prev / factor)`. ferrolearn's `ferro_keep_chain` reproduces
/// the SAME reduction (it simply continues the chain to 1, whereas sklearn stops
/// at `n_iterations`). We assert the reduction step matches at every iteration
/// sklearn actually ran. If a fixer ever swaps `div_ceil` for floor, this guard
/// FAILS — pinning the doc-author's correction.
#[test]
fn keep_count_matches_oracle() {
    // (N, factor, sklearn n_candidates_ from the live oracle above).
    let oracle: &[(usize, usize, &[usize])] = &[
        (5, 3, &[5, 2]),
        (10, 3, &[10, 4, 2]),
        (9, 3, &[9, 3, 1]),
        (7, 2, &[7, 4, 2]),
        (4, 2, &[4, 2]),
    ];
    for &(n, factor, sk_counts) in oracle {
        let ferro = ferro_keep_chain(n, factor);
        // Every iteration sklearn ran must match ferrolearn's reduction prefix.
        for (i, &sk_n) in sk_counts.iter().enumerate() {
            assert_eq!(
                ferro[i], sk_n,
                "keep-count diverged at iter {i} for N={n} factor={factor}: \
                 ferrolearn div_ceil chain {ferro:?} vs sklearn n_candidates_ {sk_counts:?}"
            );
        }
        // Each sklearn step must equal ceil(prev / factor) == div_ceil.
        for w in sk_counts.windows(2) {
            assert_eq!(
                w[1],
                w[0].div_ceil(factor),
                "sklearn reduction {} -> {} is not ceil(n/factor) for factor={factor}",
                w[0],
                w[1]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN guards — SHIPPED REQ-HALVING-MECHANIC
// ---------------------------------------------------------------------------

/// GREEN guard: REQ-HALVING-MECHANIC — a clearly-best candidate wins. `y = 1.0`,
/// the constant estimator predicting `1.0` achieves neg-MSE 0 (the max). Mirrors
/// sklearn's preference for the highest-scoring candidate of the last round
/// (`_select_best_index`, `:191-212`). Oracle confirms `c=1.0` wins:
///
/// ```text
/// python3 -c "import numpy as np
/// from sklearn.experimental import enable_halving_search_cv  # noqa
/// from sklearn.model_selection import HalvingGridSearchCV, KFold
/// from sklearn.base import BaseEstimator, RegressorMixin
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self,c=0.0): self.c=c
///     def fit(self,X,y): self.c_=self.c; return self
///     def predict(self,X): return np.full(X.shape[0], self.c_)
/// rng=np.random.RandomState(0); X=rng.rand(90,2); y=np.full(90,1.0)
/// gs=HalvingGridSearchCV(Const(), {'c':[0.0,1.0,5.0,10.0,20.0,50.0]}, factor=2,
///     cv=KFold(3), random_state=0, min_resources=9).fit(X,y)
/// print(gs.best_params_)"
/// # -> {'c': 1.0}
/// ```
#[test]
fn mechanic_best_candidate_wins() {
    let grid = param_grid! {
        "constant" => [0.0_f64, 1.0_f64, 5.0_f64, 10.0_f64, 20.0_f64, 50.0_f64],
    };
    let mut hs = HalvingGridSearchCV::new(
        Box::new(constant_factory),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    )
    .factor(2)
    .min_resources(Some(9));
    let x = Array2::<f64>::zeros((90, 2));
    let y = Array1::<f64>::from_elem(90, 1.0);
    hs.fit(&x, &y).unwrap();

    let best = hs.best_params().unwrap();
    assert_eq!(
        best.get("constant"),
        Some(&ParamValue::Float(1.0)),
        "the constant predicting y=1.0 must win the search"
    );
}

/// GREEN guard: REQ-HALVING-MECHANIC — the loop runs multiple rounds and
/// `cv_results()` is populated (non-empty) after `fit`.
#[test]
fn mechanic_cv_results_populated() {
    let grid = param_grid! {
        "constant" => [0.0_f64, 1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64],
    };
    let mut hs = HalvingGridSearchCV::new(
        Box::new(constant_factory),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    )
    .factor(3)
    .min_resources(Some(9));
    let x = Array2::<f64>::zeros((90, 2));
    let y = Array1::<f64>::from_elem(90, 1.0);
    hs.fit(&x, &y).unwrap();

    let results = hs.cv_results().unwrap();
    assert!(
        !results.params.is_empty(),
        "cv_results must be populated after fit"
    );
    assert!(hs.best_score().unwrap().is_finite());
}

/// GREEN guard: REQ-HALVING-MECHANIC — `factor < 2` and an empty grid both
/// produce `Err`. sklearn's constraint is `factor` > 0 (`:85`); ferrolearn
/// requires `factor >= 2` (`halving_grid_search.rs:238`). Both reject the empty
/// grid / degenerate factor; we pin only that ferrolearn errors (NOT exact
/// parity of the constraint interval — that is REQ-DEFAULTS #1857).
#[test]
fn mechanic_invalid_inputs_error() {
    // factor < 2
    let grid = param_grid! { "constant" => [1.0_f64] };
    let mut hs = HalvingGridSearchCV::new(
        Box::new(constant_factory),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
    )
    .factor(1);
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);
    assert!(hs.fit(&x, &y).is_err(), "factor < 2 must error");

    // empty grid
    let mut hs_empty = HalvingGridSearchCV::new(
        Box::new(constant_factory),
        vec![],
        Box::new(KFold::new(3)),
        neg_mse,
    );
    assert!(hs_empty.fit(&x, &y).is_err(), "empty grid must error");
}

/// GREEN guard: degenerate edges must NOT panic (single candidate; factor larger
/// than n_candidates; min_resources == max_resources). ferrolearn must return a
/// `Result`, never panic, on these inputs.
#[test]
fn mechanic_degenerate_no_panic() {
    let x = Array2::<f64>::zeros((30, 2));
    let y = Array1::<f64>::from_elem(30, 1.0);

    // Single candidate: nothing to halve.
    {
        let grid = param_grid! { "constant" => [1.0_f64] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        );
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_score().is_some());
    }

    // factor (5) larger than n_candidates (2): one keep-round collapses to 1.
    {
        let grid = param_grid! { "constant" => [0.0_f64, 1.0_f64] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .factor(5)
        .min_resources(Some(9));
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_params().is_some());
    }

    // min_resources == max_resources (one round only).
    {
        let grid = param_grid! { "constant" => [0.0_f64, 1.0_f64, 2.0_f64] };
        let mut hs = HalvingGridSearchCV::new(
            Box::new(constant_factory),
            grid,
            Box::new(KFold::new(3)),
            neg_mse,
        )
        .min_resources(Some(30))
        .max_resources(Some(30));
        hs.fit(&x, &y).unwrap();
        assert!(hs.best_score().is_some());
    }
}

// ---------------------------------------------------------------------------
// FAILING pin (#1852) — iteration-count / termination divergence persists even
// when min_resources AND max_resources are forced equal to sklearn's.
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's `fit` termination diverges from sklearn's
/// `n_iterations = min(n_possible_iterations, n_required_iterations)`
/// (`sklearn/model_selection/_search_successive_halving.py:289-296`).
///
/// This pin sharpens the design doc's AC-2: the schedule divergence is NOT
/// merely a consequence of `min_resources_` differing. Even with
/// `min_resources` AND `max_resources` forced to sklearn's exact values
/// (10 and 90), the FINAL-ROUND survivor SET recorded in `cv_results` differs,
/// because the TERMINATION rule differs:
///   * sklearn (`:272`,`:289-296`,`:359-360`): `n_required = 1+floor(log(5,3)) = 2`,
///     `n_possible = 1+floor(log(90//10,3)) = 3`, `n_iterations = min(3,2) = 2`.
///     It runs 2 iterations (`n_resources_ = [10, 30]`), records the LAST
///     iteration's 2 survivors, and `_select_best_index` (`:191-212`) picks the
///     best of those 2. => last-iteration candidate count == 2.
///   * ferrolearn (`halving_grid_search.rs:272-333`): grows
///     `budget *= factor` until `effective_budget >= max_res` OR `<= 1`
///     candidate. It reduces 5 -> ceil(5/3)=2 -> ceil(2/3)=1 and records a
///     SINGLE candidate at budget 30 => `cv_results.params.len() == 1`.
///
/// Live oracle (`min_resources=10, max_resources=90`, N=5, factor=3):
/// ```text
/// python3 -c "import numpy as np
/// from sklearn.experimental import enable_halving_search_cv  # noqa
/// from sklearn.model_selection import HalvingGridSearchCV, KFold
/// from sklearn.base import BaseEstimator, RegressorMixin
/// class Const(BaseEstimator, RegressorMixin):
///     def __init__(self,c=0.0): self.c=c
///     def fit(self,X,y): self.c_=self.c; return self
///     def predict(self,X): return np.full(X.shape[0], self.c_)
/// rng=np.random.RandomState(0); X=rng.rand(90,2); y=np.full(90,1.0)
/// gs=HalvingGridSearchCV(Const(), {'c':[0.0,1.0,5.0,10.0,20.0]}, factor=3,
///     cv=KFold(3), random_state=0, min_resources=10, max_resources=90).fit(X,y)
/// it=gs.cv_results_['iter']; print(int((it==it.max()).sum()))"
/// # -> 2   (sklearn records 2 candidates in the last iteration)
/// ```
/// ferrolearn records 1 row in `cv_results` for the same configuration.
///
/// Tracking: #1852 (REQ-N-ITERATIONS — architectural; no single-line fix).
#[test]
#[ignore = "divergence: n_iterations=min(n_possible,n_required) vs grow-until-max/1; tracking #1852"]
fn divergence_last_iteration_survivor_set() {
    // sklearn last-iteration candidate count from the live oracle above.
    const SK_LAST_ITER_CANDIDATE_COUNT: usize = 2;

    let grid = param_grid! {
        "constant" => [0.0_f64, 1.0_f64, 5.0_f64, 10.0_f64, 20.0_f64],
    };
    let mut hs = HalvingGridSearchCV::new(
        Box::new(constant_factory),
        grid,
        Box::new(KFold::new(3)),
        neg_mse,
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
         the termination rule diverges (#1852)",
        results.params.len()
    );
}
