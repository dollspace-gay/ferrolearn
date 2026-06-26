//! Oracle pins for `consensus_score` vs scikit-learn.
//!
//! sklearn computes bicluster Jaccard similarities, solves the best rectangular
//! assignment, and divides the matched similarity sum by the larger bicluster
//! set size.

use ferrolearn_metrics::clustering::consensus_score;
use ndarray::array;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn consensus_score_matches_sklearn_public_example() {
    // sklearn public example:
    // a = ([[T,F],[F,T]], [[F,T],[T,F]])
    // b = ([[F,T],[T,F]], [[T,F],[F,T]])
    // consensus_score(a, b) == 1.0
    let a_rows = array![[true, false], [false, true]];
    let a_cols = array![[false, true], [true, false]];
    let b_rows = array![[false, true], [true, false]];
    let b_cols = array![[true, false], [false, true]];

    let score = consensus_score(&a_rows, &a_cols, &b_rows, &b_cols).unwrap();
    assert_close(score, 1.0);
}

#[test]
fn consensus_score_matches_partial_overlap_oracle() {
    // Oracle:
    // python3 -c "from sklearn.metrics import consensus_score; \
    //   a=([[1,1,0],[0,1,1]], [[1,0,1],[0,1,1]]); \
    //   b=([[1,1,0],[0,0,1]], [[1,0,0],[0,1,1]]); \
    //   print(repr(float(consensus_score(a,b))))"
    // 0.5
    let a_rows = array![[true, true, false], [false, true, true]];
    let a_cols = array![[true, false, true], [false, true, true]];
    let b_rows = array![[true, true, false], [false, false, true]];
    let b_cols = array![[true, false, false], [false, true, true]];

    let score = consensus_score(&a_rows, &a_cols, &b_rows, &b_cols).unwrap();
    assert_close(score, 0.5);
}

#[test]
fn consensus_score_rectangular_matching_matches_sklearn() {
    // Oracle:
    // python3 -c "from sklearn.metrics import consensus_score; \
    //   a=([[1,1,0]], [[1,0,1]]); \
    //   b=([[0,1,1],[1,0,0]], [[1,1,0],[0,1,0]]); \
    //   print(repr(float(consensus_score(a,b))))"
    // 0.07142857142857142
    let a_rows = array![[true, true, false]];
    let a_cols = array![[true, false, true]];
    let b_rows = array![[false, true, true], [true, false, false]];
    let b_cols = array![[true, true, false], [false, true, false]];

    let score = consensus_score(&a_rows, &a_cols, &b_rows, &b_cols).unwrap();
    assert_close(score, 0.071_428_571_428_571_42);
}

#[test]
fn consensus_score_rejects_invalid_shapes_and_empty_biclusters() {
    let a_rows = array![[true, false], [false, true]];
    let a_cols_wrong_count = array![[true, false]];
    let b_rows = array![[true, false], [false, true]];
    let b_cols = array![[true, false], [false, true]];

    assert!(consensus_score(&a_rows, &a_cols_wrong_count, &b_rows, &b_cols).is_err());

    let a_cols = array![[true, false], [false, true]];
    let b_rows_wrong_width = array![[true, false, false], [false, true, false]];
    assert!(consensus_score(&a_rows, &a_cols, &b_rows_wrong_width, &b_cols).is_err());

    let empty_rows = array![[false, false]];
    let empty_cols = array![[false, false]];
    let empty_vs_nonempty = consensus_score(&empty_rows, &empty_cols, &b_rows, &b_cols).unwrap();
    assert_close(empty_vs_nonempty, 0.0);
    assert!(consensus_score(&empty_rows, &empty_cols, &empty_rows, &empty_cols).is_err());
}
