//! Divergence audit: `ferrolearn-preprocess` `OrdinalEncoder` vs scikit-learn
//! 1.5.2 `sklearn/preprocessing/_encoders.py::class OrdinalEncoder` (`:1235`).
//!
//! EVERY expected value below is grounded in a LIVE sklearn 1.5.2 oracle call
//! (run from /tmp) — NEVER copied from the ferrolearn side (R-CHAR-3).
//!
//! Oracle session (sklearn 1.5.2, run from /tmp):
//! ```text
//! >>> from sklearn.preprocessing import OrdinalEncoder; import numpy as np
//! >>> OrdinalEncoder().fit_transform(
//! ...   [['cat','small'],['dog','large'],['cat','medium'],['bird','small']])
//!     -> array([[1.,2.],[2.,0.],[1.,1.],[0.,2.]])   dtype float64
//! >>> .categories_  -> [['bird','cat','dog'], ['large','medium','small']]
//! >>> np.unique(['B','a','A','b','10','2']).tolist()
//!     -> ['10','2','A','B','a','b']                 # digits < upper < lower
//! >>> OrdinalEncoder().fit([['z'],['é'],['a'],['Z'],['€'],['ä']]).categories_[0]
//!     -> ['Z','a','z','ä','é','€']                  # codepoint order
//! >>> .transform([['€'],['Z'],['a']])               -> [[5.],[0.],[1.]]
//! >>> OrdinalEncoder().fit([['cat','small'],['dog','large']])
//! ...   .transform([['fish','small']])
//!     -> ValueError: Found unknown categories ['fish'] in column 0 ...
//! >>> OrdinalEncoder().fit(np.empty((0,2),dtype=object))
//!     -> ValueError: Found array with 0 sample(s) ... minimum of 1 is required.
//! >>> OrdinalEncoder().fit([['a'],['a'],['b'],['a']]).categories_[0] -> ['a','b']
//! >>> OrdinalEncoder().fit([['a','b'],['b','a']]).categories_
//!     -> [['a','b'],['a','b']]
//! >>> .transform([['a','b'],['b','a']]) -> [[0.,1.],[1.,0.]]
//! ```
//!
//! VERDICT: the String-ordinal path matches the oracle bit-for-bit on
//! categories_, ordinal values, sort order (incl. non-ASCII), duplicate folding,
//! multi-column independence, single-row fit, unknown-category rejection, and
//! empty-fit rejection. These are GREEN guards (verify-and-document). REQ-3 (the
//! output container dtype) is now SHIPPED: `transform`/`fit_transform` return
//! `Array2<f64>`, matching sklearn's `dtype=np.float64` default (`:1262`), so the
//! oracle f64 VALUES are asserted directly over an `f64` container (a compile-time
//! type guarantee + a value check). The CONFIGURABLE non-float64 `dtype` ctor
//! param remains a follow-on (blocker #1158).
//!
//! Additional live oracle (sklearn 1.5.2, run from /tmp):
//! ```text
//! >>> from sklearn.preprocessing import OrdinalEncoder
//! >>> o = OrdinalEncoder().fit_transform(
//! ...   [['bird','large'],['dog','small'],['cat','medium'],['bird','small']])
//! >>> o.tolist(); o.dtype
//!     -> [[0.,0.],[2.,2.],[1.,1.],[0.,2.]]   float64
//! >>> cats=['b00','b01','b02','b03','b04','b05','b06','b07','b08','b09','b10']
//! >>> e=OrdinalEncoder().fit([[c] for c in cats])
//! >>> e.transform([['b10']]).tolist(); e.transform([['b10']]).dtype
//!     -> [[10.0]]   float64        # lex index 10 -> exact 10.0
//! ```

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::OrdinalEncoder;
use ndarray::Array2;

/// Build an `n x 2` string array from `(col0, col1)` row tuples.
fn make_2col(rows: &[(&str, &str)]) -> Array2<String> {
    let flat: Vec<String> = rows
        .iter()
        .flat_map(|(a, b)| [a.to_string(), b.to_string()])
        .collect();
    Array2::from_shape_vec((rows.len(), 2), flat).unwrap()
}

/// Build an `n x 1` string column.
fn make_1col(vals: &[&str]) -> Array2<String> {
    Array2::from_shape_vec(
        (vals.len(), 1),
        vals.iter().map(std::string::ToString::to_string).collect(),
    )
    .unwrap()
}

// ===========================================================================
// GREEN GUARD 1 (REQ-1 / REQ-2) — ordinal VALUES + categories_ match oracle.
//
// LIVE oracle (sklearn 1.5.2):
//   fit_transform([['cat','small'],['dog','large'],['cat','medium'],
//                  ['bird','small']])
//     -> [[1.,2.],[2.,0.],[1.,1.],[0.,2.]]
//   categories_ -> [['bird','cat','dog'], ['large','medium','small']]
// sklearn `_encoders.py:99` `result = _unique(Xi)` (sorted unique); the output
// dtype is float64 (`:1262`). R-CHAR-3: expected values are the live-oracle
// float64 outputs, not ferrolearn literals.
// ===========================================================================
#[test]
fn green_value_match_and_categories() {
    // Expected from the LIVE sklearn oracle (see header) — float64 values.
    let sk_values: [[f64; 2]; 4] = [[1., 2.], [2., 0.], [1., 1.], [0., 2.]];
    let sk_cat0 = ["bird", "cat", "dog"];
    let sk_cat1 = ["large", "medium", "small"];

    let enc = OrdinalEncoder::new();
    let x = make_2col(&[
        ("cat", "small"),
        ("dog", "large"),
        ("cat", "medium"),
        ("bird", "small"),
    ]);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.categories()[0], sk_cat0, "categories_[0] (col 0)");
    assert_eq!(fitted.categories()[1], sk_cat1, "categories_[1] (col 1)");

    // Output is now `Array2<f64>` (REQ-3, sklearn `dtype=np.float64`).
    let encoded: ndarray::Array2<f64> = fitted.transform(&x).unwrap();
    for (i, row) in sk_values.iter().enumerate() {
        for (j, &expect) in row.iter().enumerate() {
            assert_eq!(
                encoded[[i, j]],
                expect,
                "ordinal value at [{i},{j}] vs sklearn float64 oracle"
            );
        }
    }
}

// ===========================================================================
// GREEN GUARD 2 (REQ-1) — lexicographic String::sort == np.unique on a mixed
// ASCII column.
//
// LIVE oracle: np.unique(['B','a','A','b','10','2']).tolist()
//   -> ['10','2','A','B','a','b']   (digits < uppercase < lowercase)
// sklearn `_encoders.py:99` `_unique(Xi)`; Rust `String::sort` is UTF-8 byte
// order == codepoint order == numpy's order on this column.
// ===========================================================================
#[test]
fn green_lexicographic_sort_matches_np_unique() {
    // Expected from LIVE np.unique oracle (numpy str_ dtype).
    let sk_sorted = ["10", "2", "A", "B", "a", "b"];

    let enc = OrdinalEncoder::new();
    let x = make_1col(&["B", "a", "A", "b", "10", "2"]);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(
        fitted.categories()[0],
        sk_sorted,
        "categories_[0] sort order vs np.unique"
    );
}

// ===========================================================================
// GREEN GUARD 3 (REQ-1) — non-ASCII / multibyte codepoint order matches the
// LIVE encoder oracle (not just np.unique on str_, but OrdinalEncoder itself).
//
// LIVE oracle:
//   OrdinalEncoder().fit([['z'],['é'],['a'],['Z'],['€'],['ä']]).categories_[0]
//     -> ['Z','a','z','ä','é','€']
//   .transform([['€'],['Z'],['a']]) -> [[5.],[0.],[1.]]
// ===========================================================================
#[test]
fn green_non_ascii_codepoint_order() {
    let sk_cats = ["Z", "a", "z", "ä", "é", "€"];
    let sk_tf: [f64; 3] = [5., 0., 1.]; // for €, Z, a (oracle float64)

    let enc = OrdinalEncoder::new();
    let x = make_1col(&["z", "é", "a", "Z", "€", "ä"]);
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.categories()[0],
        sk_cats,
        "non-ASCII categories_ vs live OrdinalEncoder oracle"
    );

    let probe = make_1col(&["€", "Z", "a"]);
    let out = fitted.transform(&probe).unwrap();
    for (i, &expect) in sk_tf.iter().enumerate() {
        assert_eq!(out[[i, 0]], expect, "non-ASCII ordinal at [{i},0]");
    }
}

// ===========================================================================
// GREEN GUARD 4 (REQ-2) — unknown category rejected, matching sklearn's
// default handle_unknown='error'.
//
// LIVE oracle:
//   OrdinalEncoder().fit([['cat','small'],['dog','large']])
//     .transform([['fish','small']])
//   -> ValueError: Found unknown categories ['fish'] in column 0 during transform
// Both sides REJECT (green guard); the message nuance (R-DEV-2) is NOT pinned.
// ===========================================================================
#[test]
fn green_unknown_category_rejected() {
    let enc = OrdinalEncoder::new();
    let x_train = make_2col(&[("cat", "small"), ("dog", "large")]);
    let fitted = enc.fit(&x_train, &()).unwrap();
    let x_test = make_2col(&[("fish", "small")]);
    assert!(
        fitted.transform(&x_test).is_err(),
        "sklearn raises ValueError on unknown 'fish'; ferrolearn must Err too"
    );
}

// ===========================================================================
// GREEN GUARD 5 (REQ-1) — empty-fit rejection MATCHES sklearn.
//
// LIVE oracle:
//   OrdinalEncoder().fit(np.empty((0,2),dtype=object))
//   -> ValueError: Found array with 0 sample(s) (shape=(0,2)) while a minimum
//      of 1 is required.
// Unlike LabelEncoder (which ALLOWS empty fit -> a pinned divergence),
// OrdinalEncoder REJECTS 0 samples via check_array. ferrolearn's
// InsufficientSamples therefore MATCHES — no divergence to pin here.
// ===========================================================================
#[test]
fn green_empty_fit_rejected_matches_sklearn() {
    let enc = OrdinalEncoder::new();
    let x: Array2<String> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
    assert!(
        enc.fit(&x, &()).is_err(),
        "sklearn raises ValueError on 0-sample fit; ferrolearn must Err too"
    );
}

// ===========================================================================
// GREEN GUARD 6 (REQ-2) — fit_transform == fit then transform (oracle-anchored
// to the Probe-1 values so it is not a pure self-equality tautology).
//
// LIVE oracle: fit_transform of the Probe-1 matrix -> [[1.,2.],[2.,0.],
//   [1.,1.],[0.,2.]]. We assert both fit_transform AND separate fit+transform
//   equal that oracle matrix (R-CHAR-3: anchored to live values, not to each
//   other).
// ===========================================================================
#[test]
fn green_fit_transform_equals_oracle() {
    let sk_values: [[f64; 2]; 4] = [[1., 2.], [2., 0.], [1., 1.], [0., 2.]];
    let expected: Array2<f64> = Array2::from_shape_vec(
        (4, 2),
        sk_values.iter().flat_map(|r| r.iter().copied()).collect(),
    )
    .unwrap();

    let enc = OrdinalEncoder::new();
    let x = make_2col(&[
        ("cat", "small"),
        ("dog", "large"),
        ("cat", "medium"),
        ("bird", "small"),
    ]);

    let via_ft = enc.fit_transform(&x).unwrap();
    let via_sep = enc.fit(&x, &()).unwrap().transform(&x).unwrap();

    assert_eq!(via_ft, expected, "fit_transform vs live oracle values");
    assert_eq!(via_sep, expected, "fit+transform vs live oracle values");
    assert_eq!(via_ft, via_sep, "fit_transform == fit then transform");
}

// ===========================================================================
// GREEN GUARD 7 (REQ-1/REQ-2) — duplicate folding, multi-column independence,
// single-row fit all match the LIVE encoder oracle.
//
// LIVE oracle:
//   fit([['a'],['a'],['b'],['a']]).categories_[0]      -> ['a','b']
//   fit([['a','b'],['b','a']]).categories_             -> [['a','b'],['a','b']]
//   .transform([['a','b'],['b','a']])                  -> [[0.,1.],[1.,0.]]
//   fit([['solo','x']]).categories_                    -> [['solo'],['x']]
// ===========================================================================
#[test]
fn green_duplicates_independence_single_row() {
    // Duplicate folding.
    let enc = OrdinalEncoder::new();
    let dup = make_1col(&["a", "a", "b", "a"]);
    let fitted = enc.fit(&dup, &()).unwrap();
    assert_eq!(fitted.categories()[0], ["a", "b"], "dup categories_");

    // Multi-column independence: same strings, two columns, identical sorted set.
    let indep = make_2col(&[("a", "b"), ("b", "a")]);
    let fitted = enc.fit(&indep, &()).unwrap();
    assert_eq!(fitted.categories()[0], ["a", "b"], "indep col0");
    assert_eq!(fitted.categories()[1], ["a", "b"], "indep col1");
    let out = fitted.transform(&indep).unwrap();
    // Oracle: [[0.,1.],[1.,0.]] (float64)
    assert_eq!(out[[0, 0]], 0.0);
    assert_eq!(out[[0, 1]], 1.0);
    assert_eq!(out[[1, 0]], 1.0);
    assert_eq!(out[[1, 1]], 0.0);

    // Single-row fit.
    let single = make_2col(&[("solo", "x")]);
    let fitted = enc.fit(&single, &()).unwrap();
    assert_eq!(fitted.categories()[0], ["solo"], "single-row col0");
    assert_eq!(fitted.categories()[1], ["x"], "single-row col1");
}

// ===========================================================================
// GREEN GUARD 8 (REQ-3) — output container dtype is float64, value-EXACT to the
// live oracle's float64 matrix over an `Array2<f64>` (type guarantee + values).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   OrdinalEncoder().fit_transform(
//     [['bird','large'],['dog','small'],['cat','medium'],['bird','small']])
//   -> [[0.,0.],[2.,2.],[1.,1.],[0.,2.]]   dtype float64
//   (col0 cats ['bird','cat','dog']; col1 cats ['large','medium','small'])
// The `Array2<f64>` binding is a COMPILE-TIME proof of REQ-3 (the output is no
// longer `Array2<usize>`); the value asserts match the oracle bit-for-bit.
// ===========================================================================
#[test]
fn green_fit_transform_f64_oracle() {
    // Expected from the LIVE sklearn oracle (see header) — float64.
    let sk: [[f64; 2]; 4] = [[0., 0.], [2., 2.], [1., 1.], [0., 2.]];

    let enc = OrdinalEncoder::new();
    let x = make_2col(&[
        ("bird", "large"),
        ("dog", "small"),
        ("cat", "medium"),
        ("bird", "small"),
    ]);

    // Explicit `Array2<f64>` type annotation: REQ-3 compile-time guarantee that
    // the output container is float64, matching sklearn `dtype=np.float64`.
    let encoded: Array2<f64> = enc.fit_transform(&x).unwrap();
    assert_eq!(encoded.dim(), (4, 2), "shape vs oracle");
    for (i, row) in sk.iter().enumerate() {
        for (j, &expect) in row.iter().enumerate() {
            assert_eq!(
                encoded[[i, j]],
                expect,
                "f64 ordinal at [{i},{j}] vs oracle"
            );
        }
    }
}

// ===========================================================================
// GREEN GUARD 9 (REQ-3) — a large ordinal index casts to an EXACT f64
// (lossless, sklearn float64). Index 10 -> 10.0.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   cats=['b00','b01','b02','b03','b04','b05','b06','b07','b08','b09','b10']
//   e=OrdinalEncoder().fit([[c] for c in cats])
//   e.transform([['b10']]) -> [[10.0]]  dtype float64   (lex index 10)
// ===========================================================================
#[test]
fn green_exact_integer_index_to_f64() {
    let cats = [
        "b00", "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10",
    ];
    let enc = OrdinalEncoder::new();
    let x = make_1col(&cats);
    let fitted = enc.fit(&x, &()).unwrap();
    // Lexicographic order keeps b00..b10 already sorted -> b10 is index 10.
    assert_eq!(fitted.categories()[0][10], "b10", "lex index 10 is b10");

    let probe = make_1col(&["b10"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    // Oracle: [[10.0]]. f64 represents 10 exactly.
    assert_eq!(out[[0, 0]], 10.0, "index 10 -> exact 10.0 (lossless f64)");
}
