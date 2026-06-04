//! Divergence / conformance audit for `ColumnTransformer` against
//! scikit-learn 1.5.2 `sklearn.compose.ColumnTransformer`.
//!
//! Upstream contract:
//! `sklearn/compose/_column_transformer.py`
//!   - `fit_transform` / `_hstack` concatenate transformer outputs in LIST
//!     (registration) order, not column order (`:976-1006`, `:1091`).
//!   - `_validate_remainder` computes remainder as
//!     `sorted(set(range(n_features)) - cols)` (`:546-550`) — ascending.
//!   - the remainder block is appended LAST (`:460-462`).
//!
//! Every expected matrix below is the FULL-precision output of the LIVE
//! sklearn 1.5.2 oracle (run from /tmp, warnings suppressed). No expected
//! value is copied from the ferrolearn side (R-CHAR-3).
//!
//! ferrolearn `StandardScaler` = population (ddof=0) std and `MinMaxScaler`
//! match sklearn (parity-verified), so the combined matrix is directly
//! comparable.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::column_transformer::{
    ColumnSelector, ColumnTransformer, Remainder, make_column_transformer,
};
use ferrolearn_preprocess::{MinMaxScaler, StandardScaler};
use ndarray::{Array2, array};

/// Shared 3x4 input used by the oracle invocations.
fn make_x() -> Array2<f64> {
    array![
        [1.0_f64, 10.0, 100.0, 5.0],
        [2.0, 20.0, 200.0, 6.0],
        [3.0, 30.0, 300.0, 7.0],
    ]
}

/// Compare every cell of `out` against `expected` within `tol`.
fn assert_matrix_eq(out: &Array2<f64>, expected: &[&[f64]], tol: f64) {
    assert_eq!(
        out.nrows(),
        expected.len(),
        "row count mismatch: got {}, expected {}",
        out.nrows(),
        expected.len()
    );
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(
            out.ncols(),
            row.len(),
            "col count mismatch at row {i}: got {}, expected {}",
            out.ncols(),
            row.len()
        );
        for (j, &exp) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - exp).abs() <= tol,
                "cell [{i},{j}]: got {got}, expected {exp} (tol {tol})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (a) GREEN-GUARD: passthrough remainder appended LAST, full values.
//
// Oracle:
//   ColumnTransformer([('std',StandardScaler(),[0,1]),
//                      ('mm',MinMaxScaler(),[2])],
//                     remainder='passthrough').fit_transform(X)
//   => [[-1.224744871391589,-1.224744871391589,0.0,5.0],
//       [ 0.0,               0.0,               0.5,6.0],
//       [ 1.224744871391589, 1.224744871391589, 1.0,7.0]]
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py:460-462,976-1006` for the
/// passthrough remainder case.
#[test]
fn green_guard_passthrough_full_values() {
    let x = make_x();
    let ct = ColumnTransformer::new(
        vec![
            (
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            ),
            (
                "mm".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![2]),
            ),
        ],
        Remainder::Passthrough,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(out.shape(), &[3, 4], "shape must be (3,4)");

    let expected: [&[f64]; 3] = [
        &[-1.224744871391589, -1.224744871391589, 0.0, 5.0],
        &[0.0, 0.0, 0.5, 6.0],
        &[1.224744871391589, 1.224744871391589, 1.0, 7.0],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);

    // Remainder column (original col 3) is appended LAST as [5,6,7].
    assert_eq!(fitted.remainder_indices(), &[3]);
}

// ---------------------------------------------------------------------------
// (b) GREEN-GUARD: remainder='drop' => (3,3), no remainder block in OUTPUT.
//
// sklearn still RECORDS the uncovered column set under drop:
//   ColumnTransformer([('std',...,[0,1]),('mm',...,[2])],remainder='drop')
//   .fit(X).transformers_ => ...,('remainder','drop',[3])
// so the uncovered set is [3] (col 3 not selected by any transformer); the
// output simply omits it (shape (3,3)).
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py:976-1006` for drop remainder.
#[test]
fn green_guard_drop_full_values() {
    let x = make_x();
    let ct = ColumnTransformer::new(
        vec![
            (
                "std".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0, 1]),
            ),
            (
                "mm".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![2]),
            ),
        ],
        Remainder::Drop,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(
        out.shape(),
        &[3, 3],
        "drop => no remainder column in output"
    );

    let expected: [&[f64]; 3] = [
        &[-1.224744871391589, -1.224744871391589, 0.0],
        &[0.0, 0.0, 0.5],
        &[1.224744871391589, 1.224744871391589, 1.0],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);

    // sklearn records the uncovered set [3] even under drop (transformers_).
    assert_eq!(
        fitted.remainder_indices(),
        &[3],
        "uncovered set is tracked as [3] (sklearn 'remainder','drop',[3])"
    );
}

// ---------------------------------------------------------------------------
// (c) GREEN-GUARD: overlapping selectors — col 0 feeds both std and mm.
//
// Oracle:
//   ColumnTransformer([('a',StandardScaler(),[0]),('b',MinMaxScaler(),[0])],
//                     remainder='drop').fit_transform(X)
//   => [[-1.224744871391589,0.0],[0.0,0.5],[1.224744871391589,1.0]]
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py:976-1006`: each transformer
/// receives its own copy of the (possibly overlapping) selected columns.
#[test]
fn green_guard_overlap_full_values() {
    let x = make_x();
    let ct = ColumnTransformer::new(
        vec![
            (
                "a".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0]),
            ),
            (
                "b".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0]),
            ),
        ],
        Remainder::Drop,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(out.shape(), &[3, 2]);

    let expected: [&[f64]; 3] = [
        &[-1.224744871391589, 0.0],
        &[0.0, 0.5],
        &[1.224744871391589, 1.0],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);
}

// ---------------------------------------------------------------------------
// (d) GREEN-GUARD: output ordering follows REGISTRATION order, not column order.
//
// A on cols [2,3], B on col [0], C on col [1], registration order A,B,C,
// remainder='drop'.
//
// Oracle:
//   ColumnTransformer([('A',StandardScaler(),[2,3]),
//                      ('B',MinMaxScaler(),[0]),
//                      ('C',StandardScaler(),[1])],
//                     remainder='drop').fit_transform(X)
//   => [[-1.224744871391589,-1.224744871391589,0.0,-1.224744871391589],
//       [ 0.0,               0.0,               0.5, 0.0],
//       [ 1.224744871391589, 1.224744871391589, 1.0, 1.224744871391589]]
//
// Column 0/1 of the output are A's outputs (originally cols 2,3); column 2 is
// B's output (col 0); column 3 is C's output (col 1). i.e. NOT sorted by
// source column index.
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py:1091` (`_hstack` in list
/// order): concatenation order is registration order, not source-column order.
#[test]
fn green_guard_registration_order_not_column_order() {
    let x = make_x();
    let ct = ColumnTransformer::new(
        vec![
            (
                "A".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![2, 3]),
            ),
            (
                "B".into(),
                Box::new(MinMaxScaler::<f64>::new()),
                ColumnSelector::Indices(vec![0]),
            ),
            (
                "C".into(),
                Box::new(StandardScaler::<f64>::new()),
                ColumnSelector::Indices(vec![1]),
            ),
        ],
        Remainder::Drop,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(out.shape(), &[3, 4]);

    let expected: [&[f64]; 3] = [
        &[
            -1.224744871391589,
            -1.224744871391589,
            0.0,
            -1.224744871391589,
        ],
        &[0.0, 0.0, 0.5, 0.0],
        &[1.224744871391589, 1.224744871391589, 1.0, 1.224744871391589],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);
}

// ---------------------------------------------------------------------------
// (e) GREEN-GUARD: non-contiguous remainder appended ascending [1,3].
//
// Transformer covers cols [0,2]; remainder = [1,3]; passthrough.
//
// Oracle:
//   ColumnTransformer([('t',StandardScaler(),[0,2])],
//                     remainder='passthrough').fit_transform(X)
//   => [[-1.224744871391589,-1.224744871391589,10.0,5.0],
//       [ 0.0,               0.0,               20.0,6.0],
//       [ 1.224744871391589, 1.224744871391589, 30.0,7.0]]
//
// Remainder block = original cols [1,3] = [[10,5],[20,6],[30,7]] in ascending
// source-column order (sklearn `sorted(set(range)-cols)` `:546`).
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py:546-550,460-462`.
#[test]
fn green_guard_noncontiguous_remainder_ascending() {
    let x = make_x();
    let ct = ColumnTransformer::new(
        vec![(
            "t".into(),
            Box::new(StandardScaler::<f64>::new()),
            ColumnSelector::Indices(vec![0, 2]),
        )],
        Remainder::Passthrough,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    // remainder_indices ascending [1,3].
    assert_eq!(
        fitted.remainder_indices(),
        &[1, 3],
        "remainder must be sorted ascending (sklearn sorted(set(range)-cols))"
    );

    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(out.shape(), &[3, 4]);

    let expected: [&[f64]; 3] = [
        &[-1.224744871391589, -1.224744871391589, 10.0, 5.0],
        &[0.0, 0.0, 20.0, 6.0],
        &[1.224744871391589, 1.224744871391589, 30.0, 7.0],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);

    // The last two output columns must equal X[:, [1,3]].
    for i in 0..3 {
        assert!((out[[i, 2]] - x[[i, 1]]).abs() <= 1e-12);
        assert!((out[[i, 3]] - x[[i, 3]]).abs() <= 1e-12);
    }
}

// ---------------------------------------------------------------------------
// (f) GREEN-GUARD: make_column_transformer matches sklearn output VALUES.
//
// Oracle:
//   make_column_transformer((StandardScaler(),[0,1]),remainder='drop')
//       .fit_transform(X)
//   => [[-1.224744871391589,-1.224744871391589],
//       [ 0.0,               0.0],
//       [ 1.224744871391589, 1.224744871391589]]
//
// Step naming differs (REQ-7 NOT-STARTED) — only VALUES are asserted.
// ---------------------------------------------------------------------------

/// Mirrors `sklearn/compose/_column_transformer.py` `make_column_transformer`
/// VALUE semantics (names intentionally not compared).
#[test]
fn green_guard_make_column_transformer_values() {
    let x = make_x();
    let ct = make_column_transformer(
        vec![(
            Box::new(StandardScaler::<f64>::new()),
            ColumnSelector::Indices(vec![0, 1]),
        )],
        Remainder::Drop,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let out = match fitted.transform(&x) {
        Ok(o) => o,
        Err(e) => panic!("transform failed: {e}"),
    };
    assert_eq!(out.shape(), &[3, 2]);

    let expected: [&[f64]; 3] = [
        &[-1.224744871391589, -1.224744871391589],
        &[0.0, 0.0],
        &[1.224744871391589, 1.224744871391589],
    ];
    assert_matrix_eq(&out, &expected, 1e-9);
}

// ---------------------------------------------------------------------------
// (g) GREEN-GUARD: error contracts.
//   - out-of-range column index in a selector => fit Err.
//   - transform with wrong ncols => Err.
// ---------------------------------------------------------------------------

/// Mirrors sklearn raising on an out-of-range column key and on a feature-count
/// mismatch at transform time.
#[test]
fn green_guard_error_contracts() {
    let x = make_x(); // 4 columns; valid indices 0..3

    // Out-of-range index at fit time.
    let ct_bad = ColumnTransformer::new(
        vec![(
            "std".into(),
            Box::new(StandardScaler::<f64>::new()),
            ColumnSelector::Indices(vec![0, 4]), // 4 out of range
        )],
        Remainder::Drop,
    );
    assert!(
        ct_bad.fit(&x, &()).is_err(),
        "out-of-range column index must error at fit"
    );

    // Wrong ncols at transform time.
    let ct_ok = ColumnTransformer::new(
        vec![(
            "std".into(),
            Box::new(StandardScaler::<f64>::new()),
            ColumnSelector::Indices(vec![0, 1]),
        )],
        Remainder::Drop,
    );
    let fitted = match ct_ok.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    let x_wrong = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    assert!(
        fitted.transform(&x_wrong).is_err(),
        "transform with wrong feature count must error"
    );
}

// ---------------------------------------------------------------------------
// (h) GREEN-GUARD: n_features_in() + remainder_indices() accessors.
// ---------------------------------------------------------------------------

/// Mirrors sklearn `n_features_in_` and the fitted remainder column set.
#[test]
fn green_guard_accessors() {
    let x = make_x(); // 4 columns
    let ct = ColumnTransformer::new(
        vec![(
            "std".into(),
            Box::new(StandardScaler::<f64>::new()),
            ColumnSelector::Indices(vec![0, 2]),
        )],
        Remainder::Passthrough,
    );
    let fitted = match ct.fit(&x, &()) {
        Ok(f) => f,
        Err(e) => panic!("fit failed: {e}"),
    };
    assert_eq!(fitted.n_features_in(), 4);
    assert_eq!(fitted.remainder_indices(), &[1, 3]);
}
