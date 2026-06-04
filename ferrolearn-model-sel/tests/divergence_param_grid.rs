//! Divergence tests for `param_grid!` / `ParamValue` / `ParamSet` vs
//! scikit-learn 1.5.2 `sklearn.model_selection.ParameterGrid`.
//!
//! Upstream: `sklearn/model_selection/_search.py` class `ParameterGrid`
//! (`:63`-`:213`).
//!
//! Expected sequences/sets in this file are derived from the LIVE sklearn 1.5.2
//! oracle (R-CHAR-3), run from /tmp:
//! ```text
//! $ python3 -c "from sklearn.model_selection import ParameterGrid; \
//!     print(list(ParameterGrid({'b':[1,2], 'a':[10,20]})))"
//! [{'a': 10, 'b': 1}, {'a': 10, 'b': 2}, {'a': 20, 'b': 1}, {'a': 20, 'b': 2}]
//! ```
//! No expected value is copied from the ferrolearn side.

use ferrolearn_model_sel::{ParamSet, ParamValue, param_grid};

/// Build the `ParamSet` the live sklearn oracle yields, for comparison.
/// Keys/values transcribed from the oracle dump (NOT from ferrolearn output).
fn oracle_set(pairs: &[(&str, ParamValue)]) -> ParamSet {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// REQ-2 — enumeration ORDER parity (FAILING / headline divergence).
// ---------------------------------------------------------------------------

/// Divergence: `param_grid!` enumerates combinations in WRITTEN key order,
/// whereas `sklearn/model_selection/_search.py:157`
/// (`items = sorted(p.items())`) sorts keys before `itertools.product`
/// (`:162`), so the SEQUENCE differs when keys are written non-sorted.
///
/// Input: keys written as `b`, then `a` (non-sorted).
/// LIVE ORACLE `list(ParameterGrid({'b':[1,2], 'a':[10,20]}))`:
///   `[{a:10,b:1}, {a:10,b:2}, {a:20,b:1}, {a:20,b:2}]`  (a slowest).
/// ferrolearn `param_grid!{"b"=>[1,2], "a"=>[10,20]}`:
///   `[{b:1,a:10}, {b:1,a:20}, {b:2,a:10}, {b:2,a:20}]`  (b slowest).
/// The i-th entries differ (e.g. index 1).
///
/// This MATTERS: `GridSearchCV::fit` iterates the `Vec` in order and breaks
/// equal-CV-score ties by position, so the order divergence changes the
/// selected best params on a tie.
///
/// Release-blocker: left un-ignored. Tracking: #1698 (parent #1697).
#[test]
fn divergence_enumeration_order_sorted_keys() {
    let grid = param_grid! {
        "b" => [1_i64, 2_i64],
        "a" => [10_i64, 20_i64],
    };

    // Sorted-key order, transcribed from the LIVE sklearn 1.5.2 oracle.
    let expected: Vec<ParamSet> = vec![
        oracle_set(&[("a", ParamValue::Int(10)), ("b", ParamValue::Int(1))]),
        oracle_set(&[("a", ParamValue::Int(10)), ("b", ParamValue::Int(2))]),
        oracle_set(&[("a", ParamValue::Int(20)), ("b", ParamValue::Int(1))]),
        oracle_set(&[("a", ParamValue::Int(20)), ("b", ParamValue::Int(2))]),
    ];

    assert_eq!(
        grid.len(),
        expected.len(),
        "combination count must match the oracle"
    );
    // Compare each combination as a key->value MAP (ParamSet is a HashMap,
    // so within-set key order is irrelevant); the SEQUENCE must match.
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            &grid[i], exp,
            "combination at index {i} diverges from the sorted-key sklearn order \
             (sklearn/model_selection/_search.py:157)"
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-1 — Cartesian-product CONTENTS (GREEN guard: passes now).
// ---------------------------------------------------------------------------

/// Guard: the SET of combinations produced by `param_grid!` equals the SET
/// produced by the live sklearn oracle `list(ParameterGrid(...))`, order
/// independent. Mirrors `product(*values)`
/// (`sklearn/model_selection/_search.py:162`).
///
/// LIVE ORACLE `list(ParameterGrid({'b':[1,2], 'a':[10,20]}))` set =
///   `{{a:10,b:1}, {a:10,b:2}, {a:20,b:1}, {a:20,b:2}}`.
/// Even though the ORDER diverges (see above), the CONTENTS match → REQ-1.
#[test]
fn guard_cartesian_product_contents_match() {
    let grid = param_grid! {
        "b" => [1_i64, 2_i64],
        "a" => [10_i64, 20_i64],
    };

    let expected: Vec<ParamSet> = vec![
        oracle_set(&[("a", ParamValue::Int(10)), ("b", ParamValue::Int(1))]),
        oracle_set(&[("a", ParamValue::Int(10)), ("b", ParamValue::Int(2))]),
        oracle_set(&[("a", ParamValue::Int(20)), ("b", ParamValue::Int(1))]),
        oracle_set(&[("a", ParamValue::Int(20)), ("b", ParamValue::Int(2))]),
    ];

    assert_eq!(grid.len(), expected.len());
    // Order-independent multiset comparison: every oracle combo is produced,
    // and vice versa.
    for exp in &expected {
        assert!(
            grid.contains(exp),
            "ferrolearn grid is missing oracle combination {exp:?}"
        );
    }
    for got in &grid {
        assert!(
            expected.contains(got),
            "ferrolearn produced combination {got:?} not present in the oracle"
        );
    }
}

// ---------------------------------------------------------------------------
// REQ-5 — edge cases (GREEN guard / pinned divergence).
// ---------------------------------------------------------------------------

/// Guard: empty grid yields exactly ONE empty combination, mirroring
/// `sklearn/model_selection/_search.py:158-159` (`if not items: yield {}`).
/// LIVE ORACLE: `list(ParameterGrid({})) == [{}]`, `len == 1`.
#[test]
fn guard_empty_grid_yields_one_empty_set() {
    let grid: Vec<ParamSet> = param_grid! {};
    assert_eq!(
        grid.len(),
        1,
        "empty grid must yield exactly one combination"
    );
    assert!(grid[0].is_empty(), "the single combination must be empty");
}

/// Divergence: an empty VALUE-LIST. sklearn raises `ValueError`
/// (`sklearn/model_selection/_search.py:138-142`:
///   `"Parameter grid for parameter 'a' need to be a non-empty sequence"`).
/// LIVE ORACLE: `ParameterGrid({'a': []})` -> `ValueError`.
/// ferrolearn `param_grid!{"a"=>[]}` collapses the product to ZERO combos and
/// returns `vec![]` with NO error (the macro has no `Result` channel to raise).
///
/// This is a divergence: sklearn REJECTS the input; ferrolearn silently yields
/// an empty grid. There is no Rust error entry point in a macro, so it is
/// blocker-only. This test asserts sklearn's contract (the input must be
/// rejected, so no valid empty grid exists); it FAILS now because ferrolearn
/// returns an empty Vec instead of erroring.
///
/// Tracking: #1699 (parent #1697).
#[test]
#[ignore = "divergence: empty value-list yields [] instead of sklearn ValueError; tracking #1699"]
fn divergence_empty_value_list_no_error() {
    let grid: Vec<ParamSet> = param_grid! { "a" => [] };
    // sklearn would have raised ValueError before producing anything; ferrolearn
    // yields zero combos. Asserting sklearn's contract: the input is rejected,
    // so a silently-empty grid is a divergence.
    assert!(
        !grid.is_empty(),
        "sklearn raises ValueError for an empty value-list \
         (sklearn/model_selection/_search.py:138-142); ferrolearn must not \
         silently yield an empty grid"
    );
}
