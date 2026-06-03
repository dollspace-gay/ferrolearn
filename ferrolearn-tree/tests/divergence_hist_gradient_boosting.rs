//! Divergence pins for `ferrolearn-tree/src/hist_gradient_boosting.rs`
//! (`HistGradientBoostingRegressor` / `HistGradientBoostingClassifier`) against
//! scikit-learn 1.5.2.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14):
//!   - `sklearn/ensemble/_hist_gradient_boosting/binning.py:53-55` —
//!     `_find_binning_thresholds`: when `len(distinct_values) <= max_bins`,
//!     thresholds are the **midpoints between consecutive distinct values**:
//!     `midpoints = distinct_values[:-1] + distinct_values[1:]; midpoints *= 0.5`.
//!     For `X=[0,2,4,6]` → distinct `[0,2,4,6]` → thresholds `[1.0, 3.0, 5.0]`.
//!     Otherwise (`:61-63`) `np.percentile(col_data, percentiles,
//!     method="midpoint")` over `np.linspace(0,100,max_bins+1)[1:-1]`.
//!     The mapping contract (`:43`) is `thresholds[i-1] < x <= thresholds[i]`.
//!   - `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:1709-1716`
//!     — `HistGradientBoostingRegressor.__init__` defaults: `learning_rate=0.1`,
//!     `max_iter=100`, `max_leaf_nodes=31`, `max_depth=None`,
//!     `min_samples_leaf=20`, `l2_regularization=0.0`, `max_bins=255`.
//!     (sklearn HGB has **no** `n_estimators` parameter — the boosting-round
//!     count is `max_iter`; that NAME divergence is the separate R-DEV-2 concern
//!     tracked at blocker #746 and is NOT pinned as a runtime test here.)
//!
//! Structural divergence (`.design/tree/hist_gradient_boosting.md`, REQ-2/REQ-7):
//! ferrolearn's `compute_bin_edges` uses quantile-position linear interpolation
//! `unique[lo]*(1-t)+unique[hi]*t` over the distinct values and appends the raw
//! column **max** as the final edge — neither sklearn's midpoint rule nor
//! `np.percentile`. The threshold *locations* therefore diverge even on tiny
//! data, moving prediction boundaries (RED pin below).
//!
//! All oracle values below come from a LIVE sklearn 1.5.2 run (the exact
//! `python3 -c "..."` invocation is quoted above each constant), NEVER copied
//! from the ferrolearn side (goal.md R-CHAR-3).
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{HistGradientBoostingRegressor, HistRegressionLoss};
use ndarray::{Array2, array};

// ===========================================================================
// RED (HEADLINE, REQ-2/REQ-7, blocker #747) — binning thresholds diverge.
// ===========================================================================

/// RED (LINCHPIN) — single-tree `HistGradientBoostingRegressor` `predict`
/// diverges because ferrolearn's bin thresholds are not sklearn's distinct-value
/// midpoints.
///
/// Fixture: `X=[[0],[2],[4],[6]]`, `y=[10,20,30,40]`, one boosting round
/// (`max_iter=1` / `n_estimators=1`), `max_leaf_nodes=4`, `min_samples_leaf=1`,
/// `learning_rate=1.0`, `early_stopping=False`. With `n_samples=4 <= 10000`
/// sklearn's `early_stopping='auto'` is OFF, so training is fully deterministic
/// (no validation split, no RNG); ferrolearn (which never early-stops) is
/// directly comparable.
///
/// sklearn places bin thresholds at the **midpoints** of consecutive distinct
/// values (`binning.py:53-55`): distinct `[0,2,4,6]` → thresholds `[1,3,5]`.
/// The contract `thresholds[i-1] < x <= thresholds[i]` (`binning.py:43`) maps:
///   * `1.2` → bin 1 (since `1.0 < 1.2 <= 3.0`), which lands in the leaf for the
///     `{2}` neighbourhood → sklearn predicts `20.0`.
///   * `4.8` → bin 2 (since `3.0 < 4.8 <= 5.0`) → the `{4}` neighbourhood →
///     sklearn predicts `30.0`.
///
/// ferrolearn's `compute_bin_edges` instead yields a first edge ≈ `1.5` and a
/// trailing raw-max edge `6.0`, so `1.2` maps to bin 0 (the `{0}` leaf, `10.0`)
/// and `4.8` maps to the top bin (the `{6}` leaf, `40.0`).
///
/// Live oracle (sklearn 1.5.2; deterministic, early_stopping=False, no RNG):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import HistGradientBoostingRegressor
/// X=np.array([[0.],[2.],[4.],[6.]]); y=np.array([10.,20,30,40])
/// m=HistGradientBoostingRegressor(loss='squared_error',max_iter=1,
///     max_leaf_nodes=4,min_samples_leaf=1,learning_rate=1.0,
///     early_stopping=False,random_state=0).fit(X,y)
/// print(m.predict(np.array([[1.2],[4.8]])).tolist())
/// "
/// # -> [20.000000000000007, 29.999999999999993]   (i.e. [20.0, 30.0])
///
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
/// print(_BinMapper(n_bins=256).fit(np.array([[0.],[2.],[4.],[6.]])).bin_thresholds_[0].tolist())
/// "
/// # -> [1.0, 3.0, 5.0]
/// ```
/// ferrolearn currently returns `[10.0, 40.0]`. MUST FAIL.
///
/// Tracking: #747
#[test]
fn divergence_binning_thresholds_predict() {
    let x = Array2::from_shape_vec((4, 1), vec![0.0, 2.0, 4.0, 6.0]).unwrap();
    let y = array![10.0, 20.0, 30.0, 40.0];

    let model = HistGradientBoostingRegressor::<f64>::new()
        .with_loss(HistRegressionLoss::LeastSquares)
        .with_n_estimators(1) // sklearn max_iter=1
        .with_max_leaf_nodes(Some(4))
        .with_min_samples_leaf(1)
        .with_learning_rate(1.0)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();

    let x_test = Array2::from_shape_vec((2, 1), vec![1.2, 4.8]).unwrap();
    let preds = fitted.predict(&x_test).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 2] = [20.0, 30.0];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-6,
            "HGBR predict[{i}] = {p} != sklearn {e}. sklearn's _BinMapper places \
             bin thresholds at distinct-value MIDPOINTS [1,3,5] for X=[0,2,4,6] \
             (binning.py:53-55) under the contract thr[i-1] < x <= thr[i] \
             (binning.py:43), so 1.2 -> bin 1 (=>20) and 4.8 -> bin 2 (=>30). \
             ferrolearn's compute_bin_edges uses quantile-position interpolation \
             (first edge ~1.5) + a trailing raw-max edge (6.0), so 1.2 -> bin 0 \
             (=>10) and 4.8 -> top bin (=>40). The fixer must rewrite \
             compute_bin_edges to sklearn's midpoint rule (n_unique<=max_bins) / \
             np.percentile(method='midpoint') (otherwise) and align map_to_bin to \
             the half-open `<=` upper-threshold convention."
        );
    }
}

// ===========================================================================
// GREEN — constructor default VALUES (REQ-1) match sklearn get_params().
// ===========================================================================

/// GREEN — HGBR default *values* match sklearn `get_params()` (sklearn 1.5.2).
/// (The parameter NAME `n_estimators` vs sklearn `max_iter` is a separate
/// R-DEV-2 concern, blocker #746, deliberately NOT pinned as a runtime test.)
///
/// Live oracle:
/// ```text
/// python3 -c "
/// from sklearn.ensemble import HistGradientBoostingRegressor as R
/// keys=['max_iter','learning_rate','min_samples_leaf','max_bins',
///       'l2_regularization','max_leaf_nodes','max_depth']
/// print({k:R().get_params()[k] for k in keys})
/// "
/// # -> {'max_iter':100,'learning_rate':0.1,'min_samples_leaf':20,'max_bins':255,
/// #     'l2_regularization':0.0,'max_leaf_nodes':31,'max_depth':None}
/// ```
#[test]
fn green_hgbr_default_values_match_sklearn() {
    let r = HistGradientBoostingRegressor::<f64>::default();
    assert_eq!(r.n_estimators, 100, "sklearn HGBR default max_iter=100");
    assert!(
        (r.learning_rate - 0.1).abs() < 1e-15,
        "sklearn HGBR default learning_rate=0.1"
    );
    assert_eq!(
        r.min_samples_leaf, 20,
        "sklearn HGBR default min_samples_leaf=20"
    );
    assert_eq!(r.max_bins, 255, "sklearn HGBR default max_bins=255");
    assert!(
        (r.l2_regularization - 0.0).abs() < 1e-15,
        "sklearn HGBR default l2_regularization=0.0"
    );
    assert_eq!(
        r.max_leaf_nodes,
        Some(31),
        "sklearn HGBR default max_leaf_nodes=31"
    );
    assert_eq!(r.max_depth, None, "sklearn HGBR default max_depth=None");
}

// ===========================================================================
// GREEN — end-to-end parity on the fixture where binning COINCIDES (REQ-7
// partial). Guards the init -> residual -> histogram -> gain -> leaf -> shrinkage
// -> predict framework: it must NOT regress when the fixer rewrites binning.
// ===========================================================================

/// GREEN — `HistGradientBoostingRegressor(loss='squared_error')` `predict`
/// matches sklearn array-by-array on `X=[1..8]ᵀ, y=[1,1,1,1,5,5,5,5]`. On this
/// fixture ferrolearn's quantile bin edges and sklearn's distinct-midpoint
/// thresholds induce the SAME single split point, so the end-to-end numerics
/// agree to ~1e-9 — validating the histogram/gain/leaf/shrinkage pipeline.
/// Deterministic (`n_samples=8 <= 10000` ⇒ `early_stopping='auto'` is OFF, no
/// RNG, no validation split).
///
/// Live oracle (sklearn 1.5.2):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import HistGradientBoostingRegressor
/// X=np.arange(1,9.).reshape(-1,1); y=np.array([1.,1,1,1,5,5,5,5])
/// m=HistGradientBoostingRegressor(loss='squared_error',max_iter=10,
///     max_leaf_nodes=7,min_samples_leaf=1,learning_rate=0.1,
///     early_stopping=False,random_state=0).fit(X,y)
/// print([round(v,10) for v in m.predict(X).tolist()])
/// "
/// # -> [1.6973568857 x4, 4.3026431143 x4]
/// ```
#[test]
fn green_hgbr_end_to_end_parity_binning_coincides() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

    let model = HistGradientBoostingRegressor::<f64>::new()
        .with_loss(HistRegressionLoss::LeastSquares)
        .with_n_estimators(10) // sklearn max_iter=10
        .with_max_leaf_nodes(Some(7))
        .with_min_samples_leaf(1)
        .with_learning_rate(0.1)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 8] = [
        1.6973568857,
        1.6973568857,
        1.6973568857,
        1.6973568857,
        4.3026431143,
        4.3026431143,
        4.3026431143,
        4.3026431143,
    ];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-6,
            "HGBR predict[{i}] = {p} != sklearn {e}. This fixture's bin split \
             coincides between ferrolearn and sklearn; it guards the \
             histogram/gain/leaf(-G/(H+λ))/shrinkage pipeline against regression \
             when the fixer rewrites compute_bin_edges to the midpoint rule."
        );
    }
}

// ===========================================================================
// GREEN — deterministic reproducibility (REQ-7/AC-8). Same seed -> identical
// predict across two fits (n_samples <= 10000 ⇒ no early-stopping RNG boundary).
// ===========================================================================

/// GREEN — two `fit` calls with identical params + seed produce IDENTICAL
/// `predict`. With `n_samples=8 <= 10000`, sklearn's `early_stopping='auto'`
/// would be OFF; ferrolearn never early-stops, so training is fully
/// deterministic and reproducible.
#[test]
fn green_hgbr_reproducibility() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

    let model = HistGradientBoostingRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_max_leaf_nodes(Some(7))
        .with_min_samples_leaf(1)
        .with_random_state(123);
    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    assert_eq!(
        f1.predict(&x).unwrap().to_vec(),
        f2.predict(&x).unwrap().to_vec(),
        "HGBR predict must be identical across fits with the same seed \
         (n_samples<=10000 ⇒ deterministic, no early-stopping RNG boundary)"
    );
}
