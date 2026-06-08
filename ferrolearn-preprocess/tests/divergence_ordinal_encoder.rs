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
use ferrolearn_preprocess::{HandleUnknown, OrdinalEncoder};
use ndarray::Array2;

/// Helper to build the `Vec<Vec<String>>` for explicit `with_categories`.
fn cats(lists: &[&[&str]]) -> Vec<Vec<String>> {
    lists
        .iter()
        .map(|l| l.iter().map(std::string::ToString::to_string).collect())
        .collect()
}

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

// ===========================================================================
// REQ-5 — handle_unknown='use_encoded_value' + unknown_value.
//
// EVERY expected value below comes from a LIVE sklearn 1.5.2 oracle (R-CHAR-3),
// reproduced via the `python3 -c` command cited above each test.
// ===========================================================================

// ---------------------------------------------------------------------------
// GREEN GUARD 10 (REQ-5) — use_encoded_value with unknown_value=-1.0: unknown
// categories -> -1.0, seen categories -> their ordinal index.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)\
//       .fit([['cat'],['dog'],['cat']]); \
//     print(e.transform([['bird'],['dog']]).tolist())"
//   -> [[-1.0], [1.0]]
// (categories_[0]=['cat','dog']; 'bird' unknown -> -1.0; 'dog' -> index 1.)
// ---------------------------------------------------------------------------
#[test]
fn green_use_encoded_value_minus_one() {
    let sk: [f64; 2] = [-1.0, 1.0]; // 'bird' (unknown), 'dog' (index 1)

    let enc = OrdinalEncoder::new()
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(-1.0);
    let x_train = make_1col(&["cat", "dog", "cat"]);
    let fitted = enc.fit(&x_train, &()).unwrap();

    let probe = make_1col(&["bird", "dog"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    assert_eq!(out[[0, 0]], sk[0], "unknown 'bird' -> -1.0 (oracle)");
    assert_eq!(out[[1, 0]], sk[1], "seen 'dog' -> 1.0 (oracle)");
}

// ---------------------------------------------------------------------------
// GREEN GUARD 11 (REQ-5) — multi-feature use_encoded_value, unknown_value=-1.0.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)\
//       .fit([['cat','x'],['dog','y'],['cat','x']]); \
//     print(e.transform([['bird','z'],['dog','x']]).tolist())"
//   -> [[-1.0, -1.0], [1.0, 0.0]]
// (col0 cats ['cat','dog']; col1 cats ['x','y']; 'bird'/'z' unknown -> -1.0;
//  'dog'->1, 'x'->0.)
// ---------------------------------------------------------------------------
#[test]
fn green_use_encoded_value_multifeature() {
    let sk: [[f64; 2]; 2] = [[-1.0, -1.0], [1.0, 0.0]];

    let enc = OrdinalEncoder::new()
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(-1.0);
    let x_train = make_2col(&[("cat", "x"), ("dog", "y"), ("cat", "x")]);
    let fitted = enc.fit(&x_train, &()).unwrap();

    let probe = make_2col(&[("bird", "z"), ("dog", "x")]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    for (i, row) in sk.iter().enumerate() {
        for (j, &expect) in row.iter().enumerate() {
            assert_eq!(out[[i, j]], expect, "multi-feature uev at [{i},{j}]");
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD 12 (REQ-5) — use_encoded_value with unknown_value=nan: unknown
// categories -> NaN, seen categories -> their ordinal index.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.preprocessing import \
//     OrdinalEncoder; e=OrdinalEncoder(handle_unknown='use_encoded_value',\
//     unknown_value=np.nan).fit([['cat'],['dog'],['cat']]); \
//     print(e.transform([['bird'],['dog']]).tolist())"
//   -> [[nan], [1.0]]
// ---------------------------------------------------------------------------
#[test]
fn green_use_encoded_value_nan() {
    let enc = OrdinalEncoder::new()
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(f64::NAN);
    let x_train = make_1col(&["cat", "dog", "cat"]);
    let fitted = enc.fit(&x_train, &()).unwrap();

    let probe = make_1col(&["bird", "dog"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    assert!(out[[0, 0]].is_nan(), "unknown 'bird' -> NaN (oracle nan)");
    assert_eq!(out[[1, 0]], 1.0, "seen 'dog' -> 1.0 (oracle)");
}

// ---------------------------------------------------------------------------
// RED GUARD 1 (REQ-5) — use_encoded_value WITHOUT unknown_value -> fit Err.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder(handle_unknown='use_encoded_value')\
//       .fit([['cat'],['dog'],['cat']])"
//   -> TypeError: unknown_value should be an integer or np.nan when
//      handle_unknown is 'use_encoded_value', got None.
// ferrolearn maps sklearn's TypeError -> FerroError::InvalidParameter (Err).
// ---------------------------------------------------------------------------
#[test]
fn red_uev_requires_unknown_value() {
    let enc = OrdinalEncoder::new().with_handle_unknown(HandleUnknown::UseEncodedValue);
    let x_train = make_1col(&["cat", "dog", "cat"]);
    assert!(
        enc.fit(&x_train, &()).is_err(),
        "sklearn raises TypeError (no unknown_value); ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 2 (REQ-5) — error-mode WITH unknown_value set -> fit Err.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder(handle_unknown='error',unknown_value=-1)\
//       .fit([['cat'],['dog'],['cat']])"
//   -> TypeError: unknown_value should only be set when handle_unknown is
//      'use_encoded_value', got -1.
// ---------------------------------------------------------------------------
#[test]
fn red_error_mode_forbids_unknown_value() {
    // Default handle_unknown is Error; set an unknown_value to provoke the Err.
    let enc = OrdinalEncoder::new().with_unknown_value(-1.0);
    assert_eq!(
        enc.handle_unknown(),
        HandleUnknown::Error,
        "default is Error"
    );
    let x_train = make_1col(&["cat", "dog", "cat"]);
    assert!(
        enc.fit(&x_train, &()).is_err(),
        "sklearn raises TypeError (unknown_value in error mode); ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 3 (REQ-5) — unknown_value collides with an in-range encoding index
// -> fit Err; while negative / >= max_cardinality / nan all SUCCEED.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.preprocessing import \
//     OrdinalEncoder
//   for v in [1,-1,5,np.nan]:
//       try:
//           OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=v)\
//               .fit([['cat'],['dog'],['cat']]); print(v,'OK')
//       except Exception as e: print(v,type(e).__name__)"
//   -> 1 ValueError   (in [0, n_categories=2) -> collision)
//      -1 OK          (negative)
//      5 OK           (>= max_cardinality)
//      nan OK         (nan sentinel)
//   (collision msg: "The used value for unknown_value 1 is one of the values
//    already used for encoding the seen categories.")
// ---------------------------------------------------------------------------
#[test]
fn red_unknown_value_collision_in_range() {
    // categories_[0] = ['cat','dog'] -> max_cardinality = 2; valid indices 0,1.
    let x_train = make_1col(&["cat", "dog", "cat"]);

    // v = 1.0 collides (in [0, 2)) -> Err.
    let collide = OrdinalEncoder::new()
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(1.0);
    assert!(
        collide.fit(&x_train, &()).is_err(),
        "unknown_value=1 in [0,2) collides; sklearn ValueError -> Err"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD 13 (REQ-5) — unknown_value -1 (negative), 5 (>= cardinality), nan
// all PASS fit (companions to RED GUARD 3; same live oracle session).
// ---------------------------------------------------------------------------
#[test]
fn green_unknown_value_negative_or_oob_or_nan_ok() {
    let x_train = make_1col(&["cat", "dog", "cat"]);

    for v in [-1.0, 5.0, f64::NAN] {
        let enc = OrdinalEncoder::new()
            .with_handle_unknown(HandleUnknown::UseEncodedValue)
            .with_unknown_value(v);
        assert!(
            enc.fit(&x_train, &()).is_ok(),
            "unknown_value={v} is OK in sklearn (negative / out-of-range / nan)"
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD 14 (REQ-5) — DEFAULT (handle_unknown='error') still rejects
// unknown categories at transform (REQ-2 preserved, UNCHANGED).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder().fit([['cat'],['dog']]).transform([['fish']])"
//   -> ValueError: Found unknown categories ['fish'] in column 0 during transform
// ---------------------------------------------------------------------------
#[test]
fn green_error_mode_unknown_still_rejected() {
    let enc = OrdinalEncoder::new(); // default Error mode
    assert_eq!(enc.handle_unknown(), HandleUnknown::Error);
    assert_eq!(enc.unknown_value(), None);
    let fitted = enc.fit(&make_1col(&["cat", "dog"]), &()).unwrap();
    assert!(
        fitted.transform(&make_1col(&["fish"])).is_err(),
        "default error-mode must still reject unknown 'fish' (REQ-2 preserved)"
    );
}

// ===========================================================================
// REQ-9 — inverse_transform.
//
// EVERY expected value below comes from a LIVE sklearn 1.5.2 oracle (R-CHAR-3),
// reproduced via the `python3 -c` command cited above each test. The encoder
// fixture for the block:
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     print([c.tolist() for c in e.categories_])"
//   -> [['cat','dog'], ['x','y','z']]
// ===========================================================================

/// The shared fixture: fit on cat/dog x col0 and x/y/z col1.
fn fit_inv_fixture() -> ferrolearn_preprocess::FittedOrdinalEncoder {
    let enc = OrdinalEncoder::new();
    let x = make_2col(&[("cat", "x"), ("dog", "y"), ("cat", "z")]);
    enc.fit(&x, &()).unwrap()
}

// ---------------------------------------------------------------------------
// GREEN GUARD 15 (REQ-9) — roundtrip: inverse_transform(transform(X)) == X,
// multi-feature.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     X=[['cat','x'],['dog','y'],['cat','z']]; \
//     print(e.inverse_transform(e.transform(X)).tolist())"
//   -> [['cat','x'],['dog','y'],['cat','z']]   (== the original X)
// ---------------------------------------------------------------------------
#[test]
fn green_inverse_roundtrip_multifeature() {
    let fitted = fit_inv_fixture();
    let x = make_2col(&[("cat", "x"), ("dog", "y"), ("cat", "z")]);
    let encoded = fitted.transform(&x).unwrap();
    let recovered = fitted.inverse_transform(&encoded).unwrap();
    // sklearn roundtrip recovers the original strings exactly.
    assert_eq!(
        recovered, x,
        "inverse_transform(transform(X)) == X (oracle)"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD 16 (REQ-9) — held-out valid ordinals decode to the oracle strings.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     print(e.inverse_transform([[1.,0.]]).tolist())"
//   -> [['dog','x']]   (col0 index 1 -> 'dog'; col1 index 0 -> 'x')
// ---------------------------------------------------------------------------
#[test]
fn green_inverse_held_out_valid_ordinals() {
    let fitted = fit_inv_fixture();
    let probe = Array2::from_shape_vec((1, 2), vec![1.0_f64, 0.0]).unwrap();
    let out = fitted.inverse_transform(&probe).unwrap();
    // Oracle: [['dog','x']].
    assert_eq!(out[[0, 0]], "dog", "col0 index 1 -> 'dog' (oracle)");
    assert_eq!(out[[0, 1]], "x", "col1 index 0 -> 'x' (oracle)");
}

// ---------------------------------------------------------------------------
// RED GUARD 4 (REQ-9) — out-of-range POSITIVE ordinal -> Err (MATCHES sklearn
// IndexError).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     e.inverse_transform([[9.,0.]])"
//   -> IndexError: index 9 is out of bounds for axis 0 with size 2
// ferrolearn maps sklearn's IndexError -> FerroError::InvalidParameter (Err).
// ---------------------------------------------------------------------------
#[test]
fn red_inverse_out_of_range_positive() {
    let fitted = fit_inv_fixture();
    // col0 has 2 categories; index 9 is out of bounds.
    let probe = Array2::from_shape_vec((1, 2), vec![9.0_f64, 0.0]).unwrap();
    assert!(
        fitted.inverse_transform(&probe).is_err(),
        "index 9 with 2 categories -> sklearn IndexError; ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 5 (REQ-9) — NEGATIVE ordinal -> Err.
//
// DIVERGENCE (R-HONEST-3): sklearn does NOT error here — `-1.0` is cast to
// int64 and numpy NEGATIVE INDEXING wraps it to the LAST category.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     print(e.inverse_transform([[-1.,0.]]).tolist())"
//   -> [['dog','x']]   (numpy categories_[0][-1] == 'dog'; NOT an error)
// ferrolearn now MIRRORS numpy's negative-index wrap (#1164 faithful).
// ---------------------------------------------------------------------------
#[test]
fn green_inverse_negative_wraps_like_numpy() {
    let fitted = fit_inv_fixture();
    // LIVE oracle: inverse_transform([[-1.,0.]]) -> [['dog','x']]
    // (categories_[0][-1] == 'dog'); [[-2.,0.]] -> [['cat','x']] (idx -2+2=0).
    let probe = Array2::from_shape_vec((1, 2), vec![-1.0_f64, 0.0]).unwrap();
    let out = fitted.inverse_transform(&probe).unwrap();
    assert_eq!(out[[0, 0]], "dog");
    assert_eq!(out[[0, 1]], "x");
    let probe2 = Array2::from_shape_vec((1, 2), vec![-2.0_f64, 0.0]).unwrap();
    let out2 = fitted.inverse_transform(&probe2).unwrap();
    assert_eq!(out2[[0, 0]], "cat");
    // -3.0 wraps to -1 (still < 0) -> out of bounds -> Err (sklearn IndexError).
    let probe3 = Array2::from_shape_vec((1, 2), vec![-3.0_f64, 0.0]).unwrap();
    assert!(fitted.inverse_transform(&probe3).is_err());
}

// ---------------------------------------------------------------------------
// GREEN GUARD (REQ-9) — NON-INTEGER ordinal truncates toward zero (sklearn).
//
// sklearn: `1.5` is cast to int64 (truncates toward zero -> 1) and decodes to
// 'dog'; `0.7` -> 0 -> 'cat'. ferrolearn now MIRRORS this (#1164 faithful).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     print(e.inverse_transform([[1.5,0.]]).tolist())"
//   -> [['dog','x']]   (astype('int64') truncates 1.5 -> 1 -> 'dog')
// ---------------------------------------------------------------------------
#[test]
fn green_inverse_non_integer_truncates_like_numpy() {
    let fitted = fit_inv_fixture();
    let probe = Array2::from_shape_vec((1, 2), vec![1.5_f64, 0.0]).unwrap();
    let out = fitted.inverse_transform(&probe).unwrap();
    assert_eq!(out[[0, 0]], "dog"); // 1.5 -> 1 -> 'dog'
    assert_eq!(out[[0, 1]], "x");
    let probe2 = Array2::from_shape_vec((1, 2), vec![0.7_f64, 0.0]).unwrap();
    let out2 = fitted.inverse_transform(&probe2).unwrap();
    assert_eq!(out2[[0, 0]], "cat"); // 0.7 -> 0 -> 'cat'
}

// ---------------------------------------------------------------------------
// RED GUARD 7 (REQ-9) — ncols mismatch -> Err (MATCHES sklearn ValueError).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     e.inverse_transform([[1.,0.,2.]])"
//   -> ValueError: Shape of the passed X data is not correct. Expected 2 columns, got 3.
// ---------------------------------------------------------------------------
#[test]
fn red_inverse_ncols_mismatch() {
    let fitted = fit_inv_fixture();
    let probe = Array2::from_shape_vec((1, 3), vec![1.0_f64, 0.0, 2.0]).unwrap();
    assert!(
        fitted.inverse_transform(&probe).is_err(),
        "3 cols when 2 expected -> sklearn ValueError; ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 8 (REQ-9) — 0-row input -> Err (MATCHES sklearn ValueError via
// check_array, symmetric with the transform #2220 guard).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "import numpy as np; from sklearn.preprocessing import \
//     OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['cat','x'],['dog','y'],['cat','z']]); \
//     e.inverse_transform(np.empty((0,2)))"
//   -> ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum
//      of 1 is required.
// ---------------------------------------------------------------------------
#[test]
fn red_inverse_zero_row() {
    let fitted = fit_inv_fixture();
    let probe: Array2<f64> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
    assert!(
        fitted.inverse_transform(&probe).is_err(),
        "0-row inverse -> sklearn ValueError (check_array); ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 9 (REQ-9) — use_encoded_value unknown_value cell -> Err.
//
// SCOPE LIMITATION (R-HONEST-3): sklearn maps the `unknown_value` cell back to
// `None`, which `Array2<String>` cannot represent (would need
// `Array2<Option<String>>`). The `unknown_value` (-1) is itself out of the
// valid `[0, len)` range, so ferrolearn takes the out-of-range error path.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)\
//       .fit([['cat','x'],['dog','y'],['cat','z']]); \
//     print(e.inverse_transform([[-1.,0.]]).tolist())"
//   -> [[None, 'x']]   (the unknown_value cell decodes to None)
// ferrolearn ERRORS (cannot represent None) — the honest, non-fabricating
// behavior; the default Error-mode encoder's inverse is complete & bit-exact.
// ---------------------------------------------------------------------------
#[test]
fn red_inverse_use_encoded_value_unknown_cell() {
    let enc = OrdinalEncoder::new()
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(-1.0);
    let x_train = make_2col(&[("cat", "x"), ("dog", "y"), ("cat", "z")]);
    let fitted = enc.fit(&x_train, &()).unwrap();
    let probe = Array2::from_shape_vec((1, 2), vec![-1.0_f64, 0.0]).unwrap();
    // sklearn returns [[None,'x']]; ferrolearn cannot represent None -> Err.
    assert!(
        fitted.inverse_transform(&probe).is_err(),
        "use_encoded_value cell -> sklearn [[None,'x']]; ferrolearn errors (Array2<String> can't hold None)"
    );
}

// ===========================================================================
// REQ-10 (#1165): get_feature_names_out (OneToOneFeatureMixin) + n_features_in_.
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   e = OrdinalEncoder().fit([['cat','x'],['dog','y']])
//   e.n_features_in_                    -> 2
//   e.get_feature_names_out().tolist()  -> ['x0', 'x1']
//   e.get_feature_names_out(['a','b'])  -> ['a', 'b']
// ===========================================================================
#[test]
fn req10_feature_names_out_and_n_features_in() {
    use ferrolearn_core::traits::Fit;
    use ferrolearn_preprocess::OrdinalEncoder;
    use ndarray::array;

    let x = array![
        ["cat".to_string(), "x".to_string()],
        ["dog".to_string(), "y".to_string()]
    ];
    let fitted = OrdinalEncoder::new().fit(&x, &()).unwrap();

    // n_features_in_ == 2
    assert_eq!(fitted.n_features_in(), 2);

    // default input_features=None -> ['x0','x1']
    let names = fitted.get_feature_names_out(None).unwrap();
    assert_eq!(names, vec!["x0".to_string(), "x1".to_string()]);

    // explicit input_features -> returned verbatim (one-to-one)
    let custom = vec!["a".to_string(), "b".to_string()];
    let named = fitted.get_feature_names_out(Some(&custom)).unwrap();
    assert_eq!(named, custom);

    // wrong-length input_features -> Err (sklearn ValueError)
    let bad = vec!["only_one".to_string()];
    assert!(fitted.get_feature_names_out(Some(&bad)).is_err());
}

// ===========================================================================
// REQ-7 (#1162) — explicit `categories` param (Categories::Explicit).
//
// EVERY expected value below comes from a LIVE sklearn 1.5.2 oracle (R-CHAR-3),
// reproduced via the `python3 -c` command cited above each test.
// ===========================================================================

// ---------------------------------------------------------------------------
// GREEN GUARD 17 (REQ-7) — explicit categories used in the GIVEN order (NOT
// re-sorted); ordinal indices follow the supplied order.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(categories=[['dog','cat','bird']]).fit([['cat'],['dog']]); \
//     print([c.tolist() for c in e.categories_]); \
//     print(e.transform([['cat'],['dog'],['bird']]).tolist())"
//   -> [['dog', 'cat', 'bird']]            # GIVEN order, NOT sorted
//      [[1.0], [0.0], [2.0]]               # cat->1, dog->0, bird->2 (given order)
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_given_order_not_sorted() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["dog", "cat", "bird"]]));
    let x = make_1col(&["cat", "dog"]);
    let fitted = enc.fit(&x, &()).unwrap();

    // categories_ are the GIVEN list, NOT sorted.
    assert_eq!(
        fitted.categories()[0],
        ["dog", "cat", "bird"],
        "categories_ in given order (oracle)"
    );

    let probe = make_1col(&["cat", "dog", "bird"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    // Oracle: [[1.0],[0.0],[2.0]].
    assert_eq!(out[[0, 0]], 1.0, "cat -> index 1 (given order)");
    assert_eq!(out[[1, 0]], 0.0, "dog -> index 0 (given order)");
    assert_eq!(out[[2, 0]], 2.0, "bird -> index 2 (given order)");
}

// ---------------------------------------------------------------------------
// GREEN GUARD 18 (REQ-7) — UNSORTED explicit categories are ACCEPTED (no sort
// requirement on the String path).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(categories=[['zebra','ant','moose']]).fit([['ant'],['zebra']]); \
//     print([c.tolist() for c in e.categories_])"
//   -> [['zebra', 'ant', 'moose']]   # accepted unsorted, given order preserved
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_unsorted_accepted() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["zebra", "ant", "moose"]]));
    let x = make_1col(&["ant", "zebra"]);
    let fitted = enc.fit(&x, &()).unwrap();
    assert_eq!(
        fitted.categories()[0],
        ["zebra", "ant", "moose"],
        "unsorted explicit accepted, order preserved (oracle)"
    );
}

// ---------------------------------------------------------------------------
// RED GUARD 10 (REQ-7) — explicit + handle_unknown='error' (default) + a data
// value not in the explicit list -> Err AT FIT (matches sklearn ValueError).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder(categories=[['cat','dog']]).fit([['cat'],['fish']])"
//   -> ValueError: Found unknown categories ['fish'] in column 0 during fit
// ---------------------------------------------------------------------------
#[test]
fn red_explicit_error_mode_data_not_in_cats_fits_err() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["cat", "dog"]]));
    let x = make_1col(&["cat", "fish"]);
    assert!(
        enc.fit(&x, &()).is_err(),
        "sklearn ValueError 'Found unknown categories ... during fit'; ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD 19 (REQ-7) — explicit + use_encoded_value: out-of-set data is OK
// at fit (subset check SKIPPED) and is encoded to unknown_value at transform.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(categories=[['cat','dog']], \
//       handle_unknown='use_encoded_value', unknown_value=-1)\
//       .fit([['cat'],['fish']]); \
//     print(e.transform([['fish'],['cat']]).tolist())"
//   -> [[-1.0], [0.0]]   # fit OK (no subset error); fish->-1, cat->0 (given order)
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_use_encoded_value_out_of_set_ok() {
    let enc = OrdinalEncoder::new()
        .with_categories(cats(&[&["cat", "dog"]]))
        .with_handle_unknown(HandleUnknown::UseEncodedValue)
        .with_unknown_value(-1.0);
    let x_train = make_1col(&["cat", "fish"]);
    // fit must NOT error (subset check skipped under use_encoded_value).
    let fitted = enc.fit(&x_train, &()).unwrap();

    let probe = make_1col(&["fish", "cat"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    // Oracle: [[-1.0],[0.0]].
    assert_eq!(out[[0, 0]], -1.0, "out-of-set 'fish' -> unknown_value -1");
    assert_eq!(out[[1, 0]], 0.0, "'cat' -> index 0 (given order)");
}

// ---------------------------------------------------------------------------
// RED GUARD 11 (REQ-7) — explicit list-count != n_features -> Err
// (ShapeMismatch, matches sklearn ValueError).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder(categories=[['cat','dog']]).fit([['cat','x'],['dog','y']])"
//   -> ValueError: Shape mismatch: if categories is an array, it has to be of
//      shape (n_features,).
// (1 category list provided for 2-feature data.)
// ---------------------------------------------------------------------------
#[test]
fn red_explicit_n_features_mismatch() {
    // 1 category list, but the data has 2 columns.
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["cat", "dog"]]));
    let x = make_2col(&[("cat", "x"), ("dog", "y")]);
    assert!(
        enc.fit(&x, &()).is_err(),
        "1 cat-list for 2 features -> sklearn ValueError (shape mismatch); ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD 20 (REQ-7) — multi-feature explicit, each column in its OWN given
// order; transform indices follow the per-column order.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(categories=[['dog','cat'],['z','y','x']])\
//       .fit([['cat','x'],['dog','y']]); \
//     print([c.tolist() for c in e.categories_]); \
//     print(e.transform([['cat','x'],['dog','z']]).tolist())"
//   -> [['dog', 'cat'], ['z', 'y', 'x']]
//      [[1.0, 2.0], [0.0, 0.0]]   # col0: cat->1,dog->0 ; col1: x->2,z->0
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_multifeature_each_own_order() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["dog", "cat"], &["z", "y", "x"]]));
    let x = make_2col(&[("cat", "x"), ("dog", "y")]);
    let fitted = enc.fit(&x, &()).unwrap();

    assert_eq!(fitted.categories()[0], ["dog", "cat"], "col0 given order");
    assert_eq!(fitted.categories()[1], ["z", "y", "x"], "col1 given order");

    let probe = make_2col(&[("cat", "x"), ("dog", "z")]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    // Oracle: [[1.0,2.0],[0.0,0.0]].
    assert_eq!(out[[0, 0]], 1.0, "col0 cat -> 1");
    assert_eq!(out[[0, 1]], 2.0, "col1 x -> 2");
    assert_eq!(out[[1, 0]], 0.0, "col0 dog -> 0");
    assert_eq!(out[[1, 1]], 0.0, "col1 z -> 0");
}

// ---------------------------------------------------------------------------
// RED GUARD 12 (REQ-7) — explicit list with DUPLICATE elements -> Err
// (matches sklearn ValueError).
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     OrdinalEncoder(categories=[['cat','cat','dog']]).fit([['cat'],['dog']])"
//   -> ValueError: In column 0, the predefined categories contain duplicate
//      elements.
// ---------------------------------------------------------------------------
#[test]
fn red_explicit_duplicate_categories() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["cat", "cat", "dog"]]));
    let x = make_1col(&["cat", "dog"]);
    assert!(
        enc.fit(&x, &()).is_err(),
        "duplicate explicit categories -> sklearn ValueError; ferrolearn must Err"
    );
}

// ---------------------------------------------------------------------------
// GREEN GUARD 21 (REQ-7) — inverse_transform with explicit (given-order)
// categories roundtrips to the given-order strings.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder(categories=[['dog','cat','bird']]).fit([['cat'],['dog']]); \
//     print(e.inverse_transform([[1.],[0.],[2.]]).tolist())"
//   -> [['cat'], ['dog'], ['bird']]   # index 1->cat, 0->dog, 2->bird (given order)
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_inverse_roundtrip_given_order() {
    let enc = OrdinalEncoder::new().with_categories(cats(&[&["dog", "cat", "bird"]]));
    let x = make_1col(&["cat", "dog"]);
    let fitted = enc.fit(&x, &()).unwrap();

    let probe = Array2::from_shape_vec((3, 1), vec![1.0_f64, 0.0, 2.0]).unwrap();
    let out = fitted.inverse_transform(&probe).unwrap();
    // Oracle: [['cat'],['dog'],['bird']].
    assert_eq!(out[[0, 0]], "cat", "index 1 -> 'cat' (given order)");
    assert_eq!(out[[1, 0]], "dog", "index 0 -> 'dog' (given order)");
    assert_eq!(out[[2, 0]], "bird", "index 2 -> 'bird' (given order)");
}

// ---------------------------------------------------------------------------
// GREEN GUARD 22 (REQ-7) — the DEFAULT categories='auto' path is UNCHANGED:
// sorted-unique categories_, matching the SHIPPED REQ-1 behavior.
//
// LIVE oracle (sklearn 1.5.2, run from /tmp):
//   python3 -c "from sklearn.preprocessing import OrdinalEncoder; \
//     e=OrdinalEncoder().fit([['dog'],['cat'],['bird']]); \
//     print([c.tolist() for c in e.categories_]); \
//     print(e.transform([['cat'],['dog'],['bird']]).tolist())"
//   -> [['bird', 'cat', 'dog']]          # AUTO -> sorted-unique
//      [[1.0], [2.0], [0.0]]
// ---------------------------------------------------------------------------
#[test]
fn green_explicit_auto_still_default() {
    // No with_categories() -> Categories::Auto (default).
    let enc = OrdinalEncoder::new();
    let x = make_1col(&["dog", "cat", "bird"]);
    let fitted = enc.fit(&x, &()).unwrap();
    // AUTO path: sorted-unique (UNCHANGED REQ-1).
    assert_eq!(
        fitted.categories()[0],
        ["bird", "cat", "dog"],
        "auto -> sorted-unique (oracle)"
    );
    let probe = make_1col(&["cat", "dog", "bird"]);
    let out: Array2<f64> = fitted.transform(&probe).unwrap();
    assert_eq!(out[[0, 0]], 1.0, "auto cat -> 1");
    assert_eq!(out[[1, 0]], 2.0, "auto dog -> 2");
    assert_eq!(out[[2, 0]], 0.0, "auto bird -> 0");
}
