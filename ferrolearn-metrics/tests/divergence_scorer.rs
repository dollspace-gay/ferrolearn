//! Divergence pins for `ferrolearn-metrics/src/scorer.rs` vs scikit-learn 1.5.2.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (the `python3 -c`
//! call is quoted in each test, R-CHAR-3) — NEVER copied from the ferrolearn side.
//!
//! RED pins (deterministic divergences — fixers must fix this iter):
//!   - divergence_scorer_applies_sign_neg_mse        (#779)
//!   - divergence_scorer_applies_sign_neg_mae        (#779)
//!   - divergence_neg_max_error_not_a_sklearn_name   (#780)
//!   - divergence_max_error_is_a_sklearn_name        (#780)
//!
//! GREEN guards (oracle-grounded, must pass now — guard correct metadata):
//!   - green_neg_mse_greater_is_better_false
//!   - green_r2_greater_is_better_true
//!   - green_explained_variance_greater_is_better_true
//!   - green_get_scorer_unknown_errors
//!
//! NOTE TO FIXER: the in-src `#[test]`s `test_scorer_evaluate` and
//! `test_scorer_with_real_metric` (`scorer.rs:245-252,282-290`) pin the RAW
//! (unsigned) value of `Scorer::score`. Once the sign-application contract
//! (REQ-2, #779) is fixed so `score()` returns `sign * metric`, those two tests
//! become WRONG and must be corrected to the signed contract:
//!   - `test_scorer_evaluate` uses `greater_is_better=false` → must assert
//!     `-(1.0/3.0)` (currently asserts `+1.0/3.0`).
//!   - `test_scorer_with_real_metric` uses `greater_is_better=false` on a
//!     perfect prediction → `-0.0`; the `0.0` assertion survives by luck only.

use ferrolearn_metrics::regression::{explained_variance_score, mean_squared_error, r2_score};
use ferrolearn_metrics::scorer::{Scorer, get_scorer, get_scorer_names, make_scorer};
use ndarray::array;

// ===========================================================================
// RED pins — deterministic sign-application + registry-name divergences
// ===========================================================================

/// Divergence: `Scorer::score` returns the RAW metric, not `sign * metric`.
///
/// sklearn's `_Scorer._score` returns
/// `self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)`
/// (`sklearn/metrics/_scorer.py:376`), and
/// `get_scorer('neg_mean_squared_error')._sign == -1`
/// (`sklearn/metrics/_scorer.py:754` `sign = 1 if greater_is_better else -1`).
/// So evaluating the scorer returns `-mse`. ferrolearn's
/// `Scorer::score` (`scorer.rs:63-65`) returns the raw `+mse`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import mean_squared_error; \
///     print(repr(float(mean_squared_error([3.0,-0.5,2.0,7.0],[2.5,0.0,2.0,8.0]))))"
///   # 0.375
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('neg_mean_squared_error')._sign)"   # -1
/// The signed scorer contract therefore returns -0.375.
/// ferrolearn returns +0.375. Tracking: #779
#[test]
fn divergence_scorer_applies_sign_neg_mse() {
    let y_true = array![3.0_f64, -0.5, 2.0, 7.0];
    let y_pred = array![2.5_f64, 0.0, 2.0, 8.0];
    let scorer = get_scorer("neg_mean_squared_error").unwrap();
    let got = scorer.score(&y_true, &y_pred).unwrap();
    // sklearn 1.5.2 live oracle: raw mse == 0.375, _sign == -1 -> scorer == -0.375.
    const SK_RAW_MSE: f64 = 0.375;
    const SK_SCORER_NEG_MSE: f64 = -SK_RAW_MSE;
    assert!(
        (got - SK_SCORER_NEG_MSE).abs() < 1e-12,
        "neg_mean_squared_error scorer: sklearn(sign*metric)={SK_SCORER_NEG_MSE}, ferrolearn={got}"
    );
}

/// Divergence: `Scorer::score` does not apply the sign for
/// `neg_mean_absolute_error` either.
///
/// sklearn `_Scorer._score` applies `self._sign` (`_scorer.py:376`);
/// `get_scorer('neg_mean_absolute_error')._sign == -1` (`_scorer.py:754`).
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import mean_absolute_error; \
///     print(repr(float(mean_absolute_error([3.0,-0.5,2.0,7.0],[2.5,0.0,2.0,8.0]))))"
///   # 0.5
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('neg_mean_absolute_error')._sign)"  # -1
/// Signed scorer contract returns -0.5; ferrolearn returns +0.5. Tracking: #779
#[test]
fn divergence_scorer_applies_sign_neg_mae() {
    let y_true = array![3.0_f64, -0.5, 2.0, 7.0];
    let y_pred = array![2.5_f64, 0.0, 2.0, 8.0];
    let scorer = get_scorer("neg_mean_absolute_error").unwrap();
    let got = scorer.score(&y_true, &y_pred).unwrap();
    // sklearn 1.5.2 live oracle: raw mae == 0.5, _sign == -1 -> scorer == -0.5.
    const SK_RAW_MAE: f64 = 0.5;
    const SK_SCORER_NEG_MAE: f64 = -SK_RAW_MAE;
    assert!(
        (got - SK_SCORER_NEG_MAE).abs() < 1e-12,
        "neg_mean_absolute_error scorer: sklearn(sign*metric)={SK_SCORER_NEG_MAE}, ferrolearn={got}"
    );
}

/// Divergence: ferrolearn registers `neg_max_error`, which is NOT a sklearn
/// scorer name. sklearn's canonical name is `max_error` (registered via
/// `make_scorer(max_error, greater_is_better=False)`, `_scorer.py:761`), and
/// `get_scorer('neg_max_error')` raises `ValueError` ("not a valid scoring
/// value", `_scorer.py:426`). ferrolearn's `BUILTIN_SCORER_NAMES`
/// (`scorer.rs:131`) contains `neg_max_error`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import get_scorer_names; \
///     print('neg_max_error' in get_scorer_names())"   # False
///   python3 -c "from sklearn.metrics import get_scorer; \
///     get_scorer('neg_max_error')"                    # ValueError
/// ferrolearn lists neg_max_error and resolves it. Tracking: #780
#[test]
fn divergence_neg_max_error_not_a_sklearn_name() {
    let names = get_scorer_names();
    // sklearn 1.5.2 live oracle: 'neg_max_error' not in get_scorer_names().
    assert!(
        !names.contains(&"neg_max_error"),
        "neg_max_error: sklearn has NO such scorer name; ferrolearn get_scorer_names()={names:?}"
    );
    // sklearn raises ValueError for the bogus name; ferrolearn must reject it too.
    assert!(
        get_scorer("neg_max_error").is_err(),
        "get_scorer('neg_max_error'): sklearn raises ValueError; ferrolearn resolved it"
    );
}

/// Divergence: `max_error` IS a sklearn scorer name (with `_sign == -1`), but
/// ferrolearn does not register it. sklearn:
/// `max_error_scorer = make_scorer(max_error, greater_is_better=False)`
/// (`_scorer.py:761`), so `get_scorer('max_error')._sign == -1`.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import get_scorer_names; \
///     print('max_error' in get_scorer_names())"   # True
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('max_error')._sign)"       # -1
/// ferrolearn rejects 'max_error' (lacks the registry entry). Tracking: #780
#[test]
fn divergence_max_error_is_a_sklearn_name() {
    let names = get_scorer_names();
    // sklearn 1.5.2 live oracle: 'max_error' in get_scorer_names() == True.
    assert!(
        names.contains(&"max_error"),
        "max_error: sklearn registers it; ferrolearn get_scorer_names()={names:?}"
    );
    // sklearn resolves it (_sign == -1 -> greater_is_better == false).
    let scorer = get_scorer("max_error")
        .expect("get_scorer('max_error'): sklearn registers it; ferrolearn rejected it");
    assert!(
        !scorer.greater_is_better,
        "max_error: sklearn _sign == -1 (greater_is_better=false); ferrolearn greater_is_better={}",
        scorer.greater_is_better
    );
}

// ===========================================================================
// GREEN guards — oracle-grounded; must pass now (protect the correct metadata)
// ===========================================================================

/// Guard: `neg_mean_squared_error` carries `greater_is_better == false`,
/// matching sklearn `_sign == -1`.
/// Oracle:
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('neg_mean_squared_error')._sign)"   # -1
#[test]
fn green_neg_mse_greater_is_better_false() {
    let scorer = get_scorer("neg_mean_squared_error").unwrap();
    // sklearn 1.5.2: _sign == -1 -> greater_is_better == false.
    assert!(
        !scorer.greater_is_better,
        "neg_mean_squared_error: sklearn _sign == -1 (greater_is_better=false); ferrolearn={}",
        scorer.greater_is_better
    );
}

/// Guard: `r2` carries `greater_is_better == true`, matching sklearn
/// `_sign == 1`. Also exercises `make_scorer(r2_score, true, "r2")`.
/// Oracle:
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('r2')._sign)"   # 1
#[test]
fn green_r2_greater_is_better_true() {
    let scorer = get_scorer("r2").unwrap();
    // sklearn 1.5.2: _sign == 1 -> greater_is_better == true.
    assert!(
        scorer.greater_is_better,
        "r2: sklearn _sign == 1 (greater_is_better=true); ferrolearn={}",
        scorer.greater_is_better
    );
    let built: Scorer<f64> = make_scorer(r2_score, true, "r2");
    assert!(built.greater_is_better);
}

/// Guard: `explained_variance` carries `greater_is_better == true`, matching
/// sklearn `_sign == 1`.
/// Oracle:
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('explained_variance')._sign)"   # 1
#[test]
fn green_explained_variance_greater_is_better_true() {
    let scorer = get_scorer("explained_variance").unwrap();
    // sklearn 1.5.2: _sign == 1 -> greater_is_better == true.
    assert!(
        scorer.greater_is_better,
        "explained_variance: sklearn _sign == 1 (greater_is_better=true); ferrolearn={}",
        scorer.greater_is_better
    );
    // Keep the metric import live (the green guard wraps the real fn).
    let built: Scorer<f64> = make_scorer(explained_variance_score, true, "explained_variance");
    assert!(built.greater_is_better);
}

/// Guard: an unknown scorer name yields `Err`, mirroring sklearn's `ValueError`
/// ("not a valid scoring value", `_scorer.py:426`).
/// Oracle:
///   python3 -c "from sklearn.metrics import get_scorer; \
///     get_scorer('nonexistent')"   # raises ValueError
#[test]
fn green_get_scorer_unknown_errors() {
    // Keep mean_squared_error import live as a sanity touch on the metric path.
    let _ = mean_squared_error(&array![1.0_f64], &array![1.0_f64]);
    assert!(
        get_scorer("nonexistent").is_err(),
        "get_scorer('nonexistent'): sklearn raises ValueError; ferrolearn must Err"
    );
}
