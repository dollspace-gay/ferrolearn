//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `SequentialFeatureSelector` against scikit-learn 1.5.2
//! `sklearn/feature_selection/_sequential.py`
//! `class SequentialFeatureSelector` (`:19-21`).
//!
//! CONTEXT (design doc `.design/preprocess/sequential_feature_selector.md`):
//! ferrolearn's `fit(x, y, score_fn)` scores subsets with a USER CALLBACK;
//! sklearn scores with `cross_val_score(estimator, X_subset, y, cv=cv,
//! scoring=scoring).mean()` (`:286-293`). The estimator / cv / scoring / "auto"
//! / tol / float-fraction / SelectorMixin surface is structurally NOT-STARTED
//! (R-DEFER-3) and is NOT pinned here. The greedy SHAPE (add-best / remove-best
//! with lowest-index `max` tie-break, `:294`) IS shipped and is green-guarded.
//!
//! Two genuinely fixable validation-boundary divergences ARE pinned (failing,
//! un-ignored): DIV-A (`n_features_to_select == n_features`) and DIV-B
//! (`ensure_min_features=2`).
//!
//! All expected values are derived from the LIVE sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3), or from the algorithm-of-record documented at
//! `_sequential.py:280-294` — NEVER literal-copied from the ferrolearn side.
//!
//! sklearn 1.5.2 (`python3 -c "import sklearn; print(sklearn.__version__)"` -> 1.5.2).

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::sequential_feature_selector::{Direction, SequentialFeatureSelector};
use ndarray::{Array1, Array2, array};

/// Score function used by the green guards: sum of selected column means.
///
/// This is a deterministic stand-in for sklearn's `cross_val_score(...).mean()`
/// score SOURCE. The score SOURCE differs from sklearn (estimator+CV vs.
/// callback) — only the greedy SHAPE is compared, per the design doc.
fn mean_sum_score(x: &Array2<f64>, _y: &Array1<f64>) -> Result<f64, FerroError> {
    let score: f64 = x
        .columns()
        .into_iter()
        .map(|c| c.sum() / c.len() as f64)
        .sum();
    Ok(score)
}

// ===========================================================================
// PINNED DIVERGENCES (must FAIL against current ferrolearn — un-ignored)
// ===========================================================================

/// DIV-A — `n_features_to_select == n_features` (REQ-8).
///
/// sklearn `_sequential.py:227-228`:
///   `if self.n_features_to_select >= n_features:`
///   `    raise ValueError("n_features_to_select must be < n_features.")`
/// requires `n_features_to_select < n_features`, so `== n_features` is REJECTED.
///
/// ferrolearn `sequential_feature_selector.rs:134` guards only
///   `if self.n_features_to_select > n_features` — it ALLOWS `== n_features`.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
///   X = [[1,10,0.1],[2,20,0.2],[3,30,0.3],[4,40,0.4]] (3 features), y = [1,2,3,4]
///   SequentialFeatureSelector(LinearRegression(), n_features_to_select=3, cv=2).fit(X,y)
///   -> ValueError: n_features_to_select must be < n_features.
///
/// Input: 3-feature X, n_features_to_select == 3.
/// sklearn: raises ValueError. ferrolearn: succeeds, selects all 3 (Ok).
///
/// Tracking: 1284
#[test]
fn divergence_n_features_to_select_equals_n_features() {
    let x = array![
        [1.0, 10.0, 0.1],
        [2.0, 20.0, 0.2],
        [3.0, 30.0, 0.3],
        [4.0, 40.0, 0.4]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let sfs = SequentialFeatureSelector::new(3, Direction::Forward);
    let result = sfs.fit(&x, &y, mean_sum_score);
    assert!(
        result.is_err(),
        "sklearn (_sequential.py:227-228) raises \
         ValueError(\"n_features_to_select must be < n_features.\") when \
         n_features_to_select (3) == n_features (3); ferrolearn must Err but returned Ok"
    );
}

/// DIV-B — `ensure_min_features=2` (REQ-8).
///
/// sklearn `_sequential.py:211-216`:
///   `X = self._validate_data(X, accept_sparse="csc", ensure_min_features=2, ...)`
/// REQUIRES at least 2 features; a 1-feature X is REJECTED.
///
/// ferrolearn `sequential_feature_selector.rs:119-156` validates only the
/// sample axis (`n_samples == 0`) and never checks a minimum feature count, so
/// it accepts a 1-feature X.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
///   X1 = [[1],[2],[3],[4]] (1 feature), y = [1,2,3,4]
///   SequentialFeatureSelector(LinearRegression(), n_features_to_select=1, cv=2).fit(X1,y)
///   -> ValueError: Found array with 1 feature(s) (shape=(4, 1)) while a minimum
///      of 2 is required by SequentialFeatureSelector.
///
/// Input: 1-feature X, n_features_to_select == 1.
/// sklearn: raises ValueError. ferrolearn: succeeds, selects [0] (Ok).
///
/// Tracking: 1285
#[test]
fn divergence_ensure_min_features_two() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
    let result = sfs.fit(&x, &y, mean_sum_score);
    assert!(
        result.is_err(),
        "sklearn (_sequential.py:214, ensure_min_features=2) raises ValueError \
         for a 1-feature X (minimum of 2 required); ferrolearn must Err but returned Ok"
    );
}

// ===========================================================================
// GREEN GUARDS (must PASS — pin the SHIPPED greedy SHAPE + scoped errors)
// ===========================================================================
//
// The greedy SHAPE is the algorithm-of-record at `_sequential.py:280-294`:
// at each step score every not-yet-decided candidate subset and pick the
// `max` over an ascending-index mapping (`:294`) — Python `max` over a dict
// keyed by ascending feature index returns the LOWEST index on ties. With the
// deterministic sum-of-means score below, the greedy results are hand-computed
// from that algorithm (NOT copied from ferrolearn).

/// REQ-1 forward greedy, n=1: picks the highest-mean column.
///
/// Fixture col means: col0=2.5, col1=25.0, col2=0.25.
/// Hand-computed greedy (single-column subset scored by its own mean):
///   max is col1 -> selected [1].
#[test]
fn green_forward_n1_picks_highest_mean() {
    let x = array![
        [1.0, 10.0, 0.1],
        [2.0, 20.0, 0.2],
        [3.0, 30.0, 0.3],
        [4.0, 40.0, 0.4]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = SequentialFeatureSelector::new(1, Direction::Forward)
        .fit(&x, &y, mean_sum_score)
        .expect("forward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[1]);
}

/// REQ-1 forward greedy, n=2: picks the top-2 means.
///
/// col means col0=2.5, col1=25.0, col2=0.25. Greedy: step1 picks col1 (25);
/// step2 adds col0 (subset {0,1}=27.5) vs col2 ({1,2}=25.25) -> col0.
/// Selected (sorted) = [0, 1].
#[test]
fn green_forward_n2_picks_top_two() {
    let x = array![
        [1.0, 10.0, 0.1],
        [2.0, 20.0, 0.2],
        [3.0, 30.0, 0.3],
        [4.0, 40.0, 0.4]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = SequentialFeatureSelector::new(2, Direction::Forward)
        .fit(&x, &y, mean_sum_score)
        .expect("forward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[0, 1]);
}

/// REQ-1 backward greedy, n=1: removes the lowest contributors.
///
/// Start {0,1,2}. Remove the candidate maximizing the remaining sum-of-means:
///   remove col0 -> {1,2}=25.25; remove col1 -> {0,2}=2.75; remove col2 ->
///   {0,1}=27.5 (max) -> remove col2, remaining {0,1}.
/// Then down to 1 from {0,1}: remove col0 -> {1}=25; remove col1 -> {0}=2.5 ->
///   remove col1, remaining {0}? No: max is keeping {1}=25, i.e. remove col0.
///   remaining [1].
#[test]
fn green_backward_n1_keeps_highest_mean() {
    let x = array![
        [1.0, 10.0, 0.1],
        [2.0, 20.0, 0.2],
        [3.0, 30.0, 0.3],
        [4.0, 40.0, 0.4]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = SequentialFeatureSelector::new(1, Direction::Backward)
        .fit(&x, &y, mean_sum_score)
        .expect("backward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[1]);
}

/// REQ-1 backward greedy, n=2: removes the single lowest contributor (col2).
///
/// Start {0,1,2}; one removal. remove col2 -> {0,1}=27.5 is the max -> [0,1].
#[test]
fn green_backward_n2_drops_lowest() {
    let x = array![
        [1.0, 10.0, 0.1],
        [2.0, 20.0, 0.2],
        [3.0, 30.0, 0.3],
        [4.0, 40.0, 0.4]
    ];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let fitted = SequentialFeatureSelector::new(2, Direction::Backward)
        .fit(&x, &y, mean_sum_score)
        .expect("backward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[0, 1]);
}

/// REQ-1 tie-break (the sklearn-parity claim, `_sequential.py:294`): on a tie,
/// the LOWEST index wins (Python `max` over an ascending-index dict).
///
/// Fixture: col0 and col1 have equal means (5.0), col2 lower (1.0).
/// Forward n=1: col0 and col1 tie at 5.0 -> lowest index 0 selected. [0].
#[test]
fn green_forward_tie_break_lowest_index() {
    let x = array![[5.0, 5.0, 1.0], [5.0, 5.0, 1.0]];
    let y = array![1.0, 2.0];
    let fitted = SequentialFeatureSelector::new(1, Direction::Forward)
        .fit(&x, &y, mean_sum_score)
        .expect("forward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[0]);
}

/// REQ-1 tie-break on backward removal: lowest index removed first on a tie.
///
/// Fixture cols 0,1 mean 5.0, col2 mean 1.0. Backward n=1:
///   step1 (from {0,1,2}): remove col0 -> {1,2}=6; remove col1 -> {0,2}=6;
///     remove col2 -> {0,1}=10 (max) -> remove col2, remaining {0,1}.
///   step2 (from {0,1}): remove col0 -> {1}=5; remove col1 -> {0}=5 (tie);
///     strict `>` keeps the FIRST (remove col0), remaining [1].
#[test]
fn green_backward_tie_break_removal() {
    let x = array![[5.0, 5.0, 1.0], [5.0, 5.0, 1.0]];
    let y = array![1.0, 2.0];
    let fitted = SequentialFeatureSelector::new(1, Direction::Backward)
        .fit(&x, &y, mean_sum_score)
        .expect("backward fit must succeed");
    assert_eq!(fitted.selected_indices(), &[1]);
}

// --- REQ-2 scoped error contracts (SHIPPED) ---

/// REQ-2: n_features_to_select == 0 -> Err.
#[test]
fn green_zero_features_err() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0, 2.0];
    let sfs = SequentialFeatureSelector::new(0, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// REQ-2: 0 rows -> Err.
#[test]
fn green_zero_rows_err() {
    let x: Array2<f64> = Array2::zeros((0, 3));
    let y: Array1<f64> = Array1::zeros(0);
    let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// REQ-2: y length mismatch -> Err.
#[test]
fn green_y_length_mismatch_err() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0];
    let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// REQ-2: transform with wrong column count -> Err.
#[test]
fn green_transform_ncols_mismatch_err() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0, 2.0];
    let fitted = SequentialFeatureSelector::new(1, Direction::Forward)
        .fit(&x, &y, mean_sum_score)
        .expect("fit must succeed");
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(fitted.transform(&x_bad).is_err());
}

/// REQ-2: score_fn returning Err is propagated.
#[test]
fn green_score_fn_error_propagated() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0, 2.0];
    let bad_fn = |_x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, FerroError> {
        Err(FerroError::NumericalInstability {
            message: "test error".into(),
        })
    };
    let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
    assert!(sfs.fit(&x, &y, bad_fn).is_err());
}

// ===========================================================================
// RE-AUDIT (post-#1284/#1285) green guards: lock the guard ORDER + the
// VALID-count non-regression. These must PASS — they confirm the two fixes are
// sklearn-faithful and did not over-reject. Live oracle (sklearn 1.5.2, /tmp):
//   4feat sel=4 -> ValueError "n_features_to_select must be < n_features."
//   2feat sel=2 -> same.
//   3feat sel=5 -> same (still rejected).
//   4feat sel=2 -> OK selected=[1 2] (valid count NOT rejected).
//   1feat sel=1/3/5 -> ValueError "Found array with 1 feature(s) ... minimum of 2".
//   1feat sel=5 (both violated) -> the MIN-FEATURES error fires FIRST
//     (sklearn `_validate_data` `:211-216` precedes the count check `:227`).
//   0feat sel=1 -> "Found array with 0 feature(s) ... minimum of 2".
// ===========================================================================

/// RE-AUDIT: `n_features_to_select == n_features` rejected for a 4-feature X.
/// Oracle: ValueError "n_features_to_select must be < n_features.".
#[test]
fn reaudit_equals_nfeatures_4feat() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(4, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// RE-AUDIT: `n_features_to_select == n_features` rejected for a 2-feature X.
#[test]
fn reaudit_equals_nfeatures_2feat() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(2, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// RE-AUDIT: `n_features_to_select > n_features` (5 of 3) still rejected.
#[test]
fn reaudit_greater_than_nfeatures() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(5, Direction::Forward);
    assert!(sfs.fit(&x, &y, mean_sum_score).is_err());
}

/// RE-AUDIT (non-regression): a VALID count `2 of 4` must succeed — the
/// `>= n_features` fix must NOT over-reject `< n_features`. Oracle: OK.
#[test]
fn reaudit_valid_count_2_of_4_ok() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(2, Direction::Forward);
    let fitted = sfs
        .fit(&x, &y, mean_sum_score)
        .expect("valid count 2<4 must succeed");
    assert_eq!(fitted.n_features_selected(), 2);
}

/// RE-AUDIT: 1-feature X with sel=3 -> min-features error (variant + message).
#[test]
fn reaudit_one_feature_min_features_message() {
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(3, Direction::Forward);
    match sfs.fit(&x, &y, mean_sum_score) {
        Err(FerroError::InvalidParameter { reason, .. }) => {
            assert!(
                reason.contains("minimum of 2"),
                "expected ensure_min_features message, got: {reason}"
            );
        }
        other => panic!("expected InvalidParameter min-features error, got {other:?}"),
    }
}

/// RE-AUDIT precedence: 1-feature X with sel=5 (BOTH min-features AND count
/// violated). sklearn raises the MIN-FEATURES error FIRST (`_validate_data`
/// `:211-216` before count `:227`). ferrolearn must return the min-features
/// error, NOT the count ("must be < n_features") error.
#[test]
fn reaudit_precedence_min_features_before_count() {
    let x = array![[1.0], [2.0], [3.0]];
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(5, Direction::Forward);
    match sfs.fit(&x, &y, mean_sum_score) {
        Err(FerroError::InvalidParameter { reason, .. }) => {
            assert!(
                reason.contains("minimum of 2"),
                "precedence: min-features must fire before count; got: {reason}"
            );
            assert!(
                !reason.contains("must be <"),
                "precedence: must NOT be the count error; got: {reason}"
            );
        }
        other => panic!("expected InvalidParameter min-features error, got {other:?}"),
    }
}

/// RE-AUDIT: 0-feature X -> min-features error. Oracle: ValueError
/// "Found array with 0 feature(s) ... minimum of 2".
#[test]
fn reaudit_zero_feature_min_features() {
    let x: Array2<f64> = Array2::zeros((3, 0));
    let y = array![1.0, 2.0, 3.0];
    let sfs = SequentialFeatureSelector::new(1, Direction::Forward);
    match sfs.fit(&x, &y, mean_sum_score) {
        Err(FerroError::InvalidParameter { reason, .. }) => {
            assert!(reason.contains("minimum of 2"), "got: {reason}");
        }
        other => panic!("expected InvalidParameter min-features error, got {other:?}"),
    }
}
