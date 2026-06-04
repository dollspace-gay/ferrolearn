//! Divergence + green-guard suite for `SimpleImputer` against scikit-learn 1.5.2.
//!
//! HEADLINE divergence DIV-1 (REQ-2): all-NaN column handling. With the sklearn
//! DEFAULT `keep_empty_features=False`, `_dense_fit` sets `statistics_=nan` for an
//! all-NaN column under Mean/Median/MostFrequent
//! (`sklearn/impute/_base.py:501,510-512,534-537`) and `transform` then DROPS those
//! columns (`_base.py:586-603`: `invalid_mask=_get_mask(statistics,nan);
//! X=X[:, valid_statistics_indexes]`). ferrolearn fills all-NaN columns with
//! `F::zero()` and KEEPS them (`imputer.rs:194-196` + `transform` keeps all columns),
//! so the OUTPUT SHAPE diverges.
//!
//! All expected values are hard-coded from the LIVE sklearn 1.5.2 oracle (run from
//! /tmp with warnings suppressed) — see each test's doc comment for the probe.
//! Tracking #1363, blocker #1364.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::imputer::{ImputeStrategy, SimpleImputer};
use ndarray::{Array2, array};

// ===========================================================================
// DIV-1 (HEADLINE, REQ-2) — all-NaN column DROPPED for Mean/Median/MostFrequent
// ===========================================================================

/// Divergence DIV-1: `SimpleImputer` Mean strategy diverges from
/// `sklearn/impute/_base.py:586-603` for an all-NaN column.
///
/// Oracle (sklearn 1.5.2, keep_empty_features=False default):
/// `X=[[1,nan],[3,nan],[5,nan]]`, `strategy='mean'`
/// -> `out.shape=(3, 1)`, `out=[[1.0],[3.0],[5.0]]` (column 1 DROPPED).
///
/// ferrolearn keeps column 1 filled with 0.0, so `out.ncols()==2`. FAILS.
#[test]
fn divergence_mean_all_nan_column_dropped() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x = array![[1.0, f64::NAN], [3.0, f64::NAN], [5.0, f64::NAN]];
    match imputer.fit_transform(&x) {
        Ok(out) => {
            // sklearn drops the all-NaN column 1: shape (3, 1).
            assert_eq!(
                out.ncols(),
                1,
                "sklearn drops all-NaN column (shape (3,1)); ferrolearn kept it"
            );
            assert_eq!(out.nrows(), 3);
            // Surviving column 0 is unchanged: [1, 3, 5].
            assert_eq!(out[[0, 0]], 1.0);
            assert_eq!(out[[1, 0]], 3.0);
            assert_eq!(out[[2, 0]], 5.0);
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit_transform unexpectedly errored: {e}"),
    }
}

/// Divergence DIV-1: `SimpleImputer` Median strategy diverges from
/// `sklearn/impute/_base.py:586-603` for an all-NaN column.
///
/// Oracle (sklearn 1.5.2): `X=[[1,nan],[3,nan],[5,nan]]`, `strategy='median'`
/// -> `out.shape=(3, 1)`, `out=[[1.0],[3.0],[5.0]]` (column 1 DROPPED).
///
/// ferrolearn keeps column 1, so `out.ncols()==2`. FAILS.
#[test]
fn divergence_median_all_nan_column_dropped() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
    let x = array![[1.0, f64::NAN], [3.0, f64::NAN], [5.0, f64::NAN]];
    match imputer.fit_transform(&x) {
        Ok(out) => {
            assert_eq!(
                out.ncols(),
                1,
                "sklearn drops all-NaN column (shape (3,1)); ferrolearn kept it"
            );
            assert_eq!(out.nrows(), 3);
            assert_eq!(out[[0, 0]], 1.0);
            assert_eq!(out[[1, 0]], 3.0);
            assert_eq!(out[[2, 0]], 5.0);
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit_transform unexpectedly errored: {e}"),
    }
}

/// Divergence DIV-1 (multi-feature mix): a normal column whose median is imputed
/// alongside an all-NaN column that sklearn DROPS.
///
/// Oracle (sklearn 1.5.2): `X=[[1,nan],[nan,nan],[5,nan]]`, `strategy='median'`
/// -> `statistics_=[3.0, nan]`, `out.shape=(3, 1)`, `out=[[1.0],[3.0],[5.0]]`.
/// Column 0 (non-NaN [1,5] -> median 3.0) keeps and imputes the inner NaN with 3.0;
/// column 1 is dropped.
///
/// ferrolearn keeps both columns (`out.ncols()==2`). FAILS.
#[test]
fn divergence_median_multi_feature_one_all_nan_dropped() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
    let x = array![[1.0, f64::NAN], [f64::NAN, f64::NAN], [5.0, f64::NAN]];
    match imputer.fit_transform(&x) {
        Ok(out) => {
            // Only the normal column survives: shape (3, 1).
            assert_eq!(
                out.ncols(),
                1,
                "sklearn drops the all-NaN column, keeps the median-imputed one (shape (3,1))"
            );
            assert_eq!(out.nrows(), 3);
            // Surviving column 0: [1, median=3, 5].
            assert_eq!(out[[0, 0]], 1.0);
            assert_eq!(out[[1, 0]], 3.0);
            assert_eq!(out[[2, 0]], 5.0);
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit_transform unexpectedly errored: {e}"),
    }
}

// ===========================================================================
// GREEN-GUARD: Constant KEEPS the all-NaN column (sklearn agrees, _base.py:583)
// ===========================================================================

/// Green-guard: under `strategy='constant'` sklearn KEEPS all columns
/// (`sklearn/impute/_base.py:583`), and ferrolearn also keeps them — they AGREE.
///
/// Oracle (sklearn 1.5.2): `X=[[1,nan],[nan,4]]`, `strategy='constant', fill_value=-99`
/// -> `out.shape=(2, 2)`, `out=[[1.0,-99.0],[-99.0,4.0]]`.
#[test]
fn green_constant_all_nan_column_kept() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Constant(-99.0));
    let x = array![[1.0, f64::NAN], [f64::NAN, 4.0]];
    match imputer.fit_transform(&x) {
        Ok(out) => {
            assert_eq!(out.ncols(), 2, "constant strategy keeps all columns");
            assert_eq!(out.nrows(), 2);
            assert_eq!(out[[0, 0]], 1.0);
            assert_eq!(out[[0, 1]], -99.0);
            assert_eq!(out[[1, 0]], -99.0);
            assert_eq!(out[[1, 1]], 4.0);
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit_transform unexpectedly errored: {e}"),
    }
}

// ===========================================================================
// GREEN-GUARD: fill VALUES (REQ-1) on columns with >=1 observed value
// ===========================================================================

/// Green-guard Mean fill values (sklearn `np.ma.mean`, `_base.py:498`).
///
/// Oracle (sklearn 1.5.2): `X=[[1,nan],[3,4],[5,6]]`, `strategy='mean'`
/// -> `statistics_=[3.0, 5.0]`, `out=[[1.0,5.0],[3.0,4.0],[5.0,6.0]]`.
#[test]
fn green_mean_fill_values() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            let f = fitted.fill_values();
            assert!((f[0] - 3.0).abs() < 1e-12, "col0 mean: {} != 3.0", f[0]);
            assert!((f[1] - 5.0).abs() < 1e-12, "col1 mean: {} != 5.0", f[1]);
            match fitted.transform(&x) {
                Ok(out) => {
                    assert!((out[[0, 1]] - 5.0).abs() < 1e-12, "imputed NaN -> 5.0");
                    assert!((out[[1, 1]] - 4.0).abs() < 1e-12, "untouched 4.0");
                    assert!((out[[2, 1]] - 6.0).abs() < 1e-12, "untouched 6.0");
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Green-guard Median fill values, odd/even/NaN (sklearn `np.ma.median`, `_base.py:507`).
///
/// Oracle (sklearn 1.5.2):
/// odd `[[1],[3],[5],[7],[9]]` -> `5.0`;
/// even `[[1],[3],[5],[7]]` -> `4.0`;
/// with NaN `[[2],[nan],[4],[6]]` -> `4.0`.
#[test]
fn green_median_fill_values() {
    let probe = |x: &Array2<f64>, expected: f64| {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
        match imputer.fit(x, &()) {
            Ok(fitted) => {
                let got = fitted.fill_values()[0];
                assert!((got - expected).abs() < 1e-12, "median {got} != {expected}");
            }
            #[allow(
                clippy::assertions_on_constants,
                reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
            )]
            Err(e) => assert!(false, "fit errored: {e}"),
        }
    };
    probe(&array![[1.0], [3.0], [5.0], [7.0], [9.0]], 5.0);
    probe(&array![[1.0], [3.0], [5.0], [7.0]], 4.0);
    probe(&array![[2.0], [f64::NAN], [4.0], [6.0]], 4.0);
}

/// Adversarial median even-count probe (doc-author's least-confident case).
///
/// Oracle (sklearn 1.5.2): `[[1],[2],[3],[100]]`, `strategy='median'` -> `2.5`
/// (`np.ma.median` averages the two middle values 2 and 3). This matches a naive
/// two-middle average, so ferrolearn `median_of` is EXPECTED to agree (green-guard).
#[test]
fn green_median_even_count_adversarial() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
    let x = array![[1.0], [2.0], [3.0], [100.0]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            let got = fitted.fill_values()[0];
            assert!(
                (got - 2.5).abs() < 1e-12,
                "even-count median {got} != 2.5 (np.ma.median averages 2 and 3)"
            );
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Green-guard MostFrequent fill values incl. tie->min (sklearn `_most_frequent`,
/// `_base.py:36-71`; tie breaks to `min`, `:68-70`).
///
/// Oracle (sklearn 1.5.2):
/// `[[1],[2],[2],[3]]` -> `2.0`;
/// tie `[[1],[1],[3],[3]]` -> `1.0`;
/// with NaN `[[1],[nan],[2],[2]]` -> `2.0`.
#[test]
fn green_most_frequent_fill_values() {
    let probe = |x: &Array2<f64>, expected: f64| {
        let imputer = SimpleImputer::<f64>::new(ImputeStrategy::MostFrequent);
        match imputer.fit(x, &()) {
            Ok(fitted) => {
                let got = fitted.fill_values()[0];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "most_frequent {got} != {expected}"
                );
            }
            #[allow(
                clippy::assertions_on_constants,
                reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
            )]
            Err(e) => assert!(false, "fit errored: {e}"),
        }
    };
    probe(&array![[1.0], [2.0], [2.0], [3.0]], 2.0);
    probe(&array![[1.0], [1.0], [3.0], [3.0]], 1.0);
    probe(&array![[1.0], [f64::NAN], [2.0], [2.0]], 2.0);
}

/// Green-guard f32 path (Mean).
///
/// Oracle (sklearn 1.5.2): `[[1,nan],[3,4]]`, `strategy='mean'` -> col1 mean `4.0`.
#[test]
fn green_f32_mean() {
    let imputer = SimpleImputer::<f32>::new(ImputeStrategy::Mean);
    let x: Array2<f32> = array![[1.0f32, f32::NAN], [3.0, 4.0]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => match fitted.transform(&x) {
            Ok(out) => assert!((out[[0, 1]] - 4.0f32).abs() < 1e-6, "f32 imputed -> 4.0"),
            #[allow(
                clippy::assertions_on_constants,
                reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
            )]
            Err(e) => assert!(false, "transform errored: {e}"),
        },
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

// ===========================================================================
// GREEN-GUARD: error contracts (REQ-3)
// ===========================================================================

/// Green-guard: `fit` on zero rows errors (analog of sklearn input validation).
#[test]
fn green_fit_zero_rows_errors() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x: Array2<f64> = Array2::zeros((0, 3));
    assert!(imputer.fit(&x, &()).is_err(), "zero-row fit must error");
}

/// Green-guard: `transform` with a column-count mismatch errors
/// (sklearn raises ValueError, `_base.py:573-577`).
#[test]
fn green_transform_ncols_mismatch_errors() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    match imputer.fit(&x_train, &()) {
        Ok(fitted) => {
            let x_bad = array![[1.0, 2.0, 3.0]];
            assert!(
                fitted.transform(&x_bad).is_err(),
                "ncols mismatch must error"
            );
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Green-guard: calling `transform` on the unfitted imputer errors
/// (sklearn raises NotFittedError via `check_is_fitted`, `_base.py:568`).
#[test]
fn green_unfitted_transform_errors() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x = array![[1.0, 2.0]];
    assert!(
        imputer.transform(&x).is_err(),
        "unfitted transform must error"
    );
}

// ===========================================================================
// RE-AUDIT (#1363/#1364 fix verification) — multi-config faithfulness probes
// against the LIVE sklearn 1.5.2 oracle (keep_empty_features=False default).
// Each oracle value below is hard-coded from a /tmp `python3 -W ignore` run
// (R-CHAR-3); the matching sklearn invocation is quoted in the doc comment.
// ===========================================================================

/// Probe (a) — COLUMN ORDER preservation with two interleaved all-NaN columns.
///
/// Oracle (sklearn 1.5.2): `SimpleImputer(strategy='mean')` on
/// `X=[[1,nan,7,nan],[3,nan,9,nan],[5,nan,11,nan]]`
/// -> `statistics_=[3, nan, 9, nan]`, `out.shape=(3,2)`,
/// `out=[[1,7],[3,9],[5,11]]` (cols {0,2} kept, in that order).
/// Verifies ferrolearn `kept_indices()==[0,2]` and output values+order match.
#[test]
fn reaudit_a_column_order_two_all_nan_dropped() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x = array![
        [1.0, f64::NAN, 7.0, f64::NAN],
        [3.0, f64::NAN, 9.0, f64::NAN],
        [5.0, f64::NAN, 11.0, f64::NAN]
    ];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            assert_eq!(
                fitted.kept_indices(),
                &[0usize, 2usize],
                "kept_indices must preserve {{0,2}} order"
            );
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.nrows(), 3);
                    assert_eq!(out.ncols(), 2, "sklearn out.shape=(3,2)");
                    // col-order: out col0 == input col0, out col1 == input col2
                    assert!((out[[0, 0]] - 1.0).abs() < 1e-9);
                    assert!((out[[1, 0]] - 3.0).abs() < 1e-9);
                    assert!((out[[2, 0]] - 5.0).abs() < 1e-9);
                    assert!((out[[0, 1]] - 7.0).abs() < 1e-9);
                    assert!((out[[1, 1]] - 9.0).abs() < 1e-9);
                    assert!((out[[2, 1]] - 11.0).abs() < 1e-9);
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (b) — EVERY column all-NaN under Mean: sklearn yields `(n,0)`.
///
/// Oracle (sklearn 1.5.2): `SimpleImputer(strategy='mean')` on
/// `X=[[nan,nan],[nan,nan]]` -> `statistics_=[nan,nan]`, `out.shape=(2,0)`.
/// Verifies ferrolearn `out.ncols()==0` and `kept_indices()==[]`.
#[test]
fn reaudit_b_all_columns_all_nan_zero_output() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x = array![[f64::NAN, f64::NAN], [f64::NAN, f64::NAN]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            assert!(fitted.kept_indices().is_empty(), "no column survives");
            assert!(fitted.fill_values()[0].is_nan());
            assert!(fitted.fill_values()[1].is_nan());
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.ncols(), 0, "sklearn out.shape=(2,0)");
                    assert_eq!(out.nrows(), 2);
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (c) — MostFrequent with an all-NaN column among normal ones.
///
/// Oracle (sklearn 1.5.2): `SimpleImputer(strategy='most_frequent')` on
/// `X=[[1,nan],[1,nan],[2,nan]]` -> `statistics_=[1, nan]`, `out.shape=(3,1)`,
/// `out=[[1],[1],[2]]` (col1 dropped, col0 most-frequent=1).
#[test]
fn reaudit_c_most_frequent_all_nan_dropped() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::MostFrequent);
    let x = array![[1.0, f64::NAN], [1.0, f64::NAN], [2.0, f64::NAN]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            assert_eq!(fitted.kept_indices(), &[0usize]);
            assert!((fitted.fill_values()[0] - 1.0).abs() < 1e-9);
            assert!(fitted.fill_values()[1].is_nan());
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.ncols(), 1, "sklearn out.shape=(3,1)");
                    assert_eq!(out.nrows(), 3);
                    assert!((out[[0, 0]] - 1.0).abs() < 1e-9);
                    assert!((out[[1, 0]] - 1.0).abs() < 1e-9);
                    assert!((out[[2, 0]] - 2.0).abs() < 1e-9);
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (d) — Constant KEEPS an all-NaN column and FILLS the constant (not 0).
///
/// Oracle (sklearn 1.5.2): `SimpleImputer(strategy='constant', fill_value=-7)` on
/// `X=[[1,nan],[nan,nan]]` -> `statistics_=[-7,-7]`, `out.shape=(2,2)`,
/// `out=[[1,-7],[-7,-7]]`. Col1 (all-NaN) is KEPT and entirely -7.
#[test]
fn reaudit_d_constant_all_nan_kept_filled_constant() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Constant(-7.0));
    let x = array![[1.0, f64::NAN], [f64::NAN, f64::NAN]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            assert_eq!(
                fitted.kept_indices(),
                &[0usize, 1usize],
                "constant keeps all"
            );
            assert!((fitted.fill_values()[0] - (-7.0)).abs() < 1e-9);
            assert!((fitted.fill_values()[1] - (-7.0)).abs() < 1e-9);
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.ncols(), 2, "sklearn out.shape=(2,2)");
                    assert_eq!(out.nrows(), 2);
                    assert!((out[[0, 0]] - 1.0).abs() < 1e-9);
                    assert!((out[[0, 1]] - (-7.0)).abs() < 1e-9);
                    assert!((out[[1, 0]] - (-7.0)).abs() < 1e-9);
                    assert!(
                        (out[[1, 1]] - (-7.0)).abs() < 1e-9,
                        "all-NaN col filled CONSTANT not 0"
                    );
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (e) — `fill_values()` mirrors sklearn `statistics_`: NaN at dropped pos.
///
/// Oracle (sklearn 1.5.2): `SimpleImputer(strategy='median')` on
/// `X=[[1,nan],[nan,nan],[5,nan]]` -> `statistics_=[3, nan]`. Col0 median of
/// [1,5]=3.0; col1 all-NaN -> NaN. Verifies ferrolearn fill_values()=[3, NaN].
#[test]
fn reaudit_e_fill_values_mirror_statistics_nan() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Median);
    let x = array![[1.0, f64::NAN], [f64::NAN, f64::NAN], [5.0, f64::NAN]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            let f = fitted.fill_values();
            assert_eq!(f.len(), 2, "fill_values has one entry per INPUT column");
            assert!((f[0] - 3.0).abs() < 1e-9, "col0 median=3.0");
            assert!(f[1].is_nan(), "dropped col1 statistics_ is NaN");
            assert_eq!(fitted.kept_indices(), &[0usize]);
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (f) — fit on one matrix, transform a SEPARATE matrix (same ncols),
/// with NaNs in the KEPT columns: the kept-column projection applies to new data.
///
/// Oracle (sklearn 1.5.2): fit `SimpleImputer(strategy='mean')` on
/// `Xfit=[[1,nan,7],[3,nan,9],[5,nan,11]]` -> `statistics_=[3, nan, 9]`
/// (kept cols {0,2}). Then transform
/// `Xnew=[[nan,100,nan],[10,200,20]]` -> `out.shape=(2,2)`,
/// `out=[[3,9],[10,20]]` (col0 NaN->mean 3, col2 NaN->mean 9; col1 dropped).
#[test]
fn reaudit_f_transform_separate_matrix_projection() {
    let imputer = SimpleImputer::<f64>::new(ImputeStrategy::Mean);
    let x_fit = array![
        [1.0, f64::NAN, 7.0],
        [3.0, f64::NAN, 9.0],
        [5.0, f64::NAN, 11.0]
    ];
    match imputer.fit(&x_fit, &()) {
        Ok(fitted) => {
            assert_eq!(fitted.kept_indices(), &[0usize, 2usize]);
            let x_new = array![[f64::NAN, 100.0, f64::NAN], [10.0, 200.0, 20.0]];
            match fitted.transform(&x_new) {
                Ok(out) => {
                    assert_eq!(out.ncols(), 2, "sklearn out.shape=(2,2)");
                    assert_eq!(out.nrows(), 2);
                    // col0: NaN -> mean 3.0, then 10.0 untouched
                    assert!(
                        (out[[0, 0]] - 3.0).abs() < 1e-9,
                        "NaN in kept col0 -> mean 3"
                    );
                    assert!((out[[1, 0]] - 10.0).abs() < 1e-9);
                    // col2 (out col1): NaN -> mean 9.0, then 20.0 untouched
                    assert!(
                        (out[[0, 1]] - 9.0).abs() < 1e-9,
                        "NaN in kept col2 -> mean 9"
                    );
                    assert!((out[[1, 1]] - 20.0).abs() < 1e-9);
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}

/// Probe (g) — f32 path with an all-NaN column dropped.
///
/// Oracle (sklearn 1.5.2, float32 input): `SimpleImputer(strategy='mean')` on
/// `X=[[1,nan,7],[3,nan,9]]` -> `statistics_=[2, nan, 8]`, `out.shape=(2,2)`,
/// `out=[[1,7],[3,9]]` (col1 dropped; no NaN to fill in kept cols here).
#[test]
fn reaudit_g_f32_all_nan_column_dropped() {
    let imputer = SimpleImputer::<f32>::new(ImputeStrategy::Mean);
    let x: Array2<f32> = array![[1.0f32, f32::NAN, 7.0], [3.0, f32::NAN, 9.0]];
    match imputer.fit(&x, &()) {
        Ok(fitted) => {
            assert_eq!(fitted.kept_indices(), &[0usize, 2usize]);
            assert!(
                (fitted.fill_values()[0] - 2.0f32).abs() < 1e-6,
                "col0 mean=2"
            );
            assert!(fitted.fill_values()[1].is_nan(), "dropped col1 -> NaN");
            assert!(
                (fitted.fill_values()[2] - 8.0f32).abs() < 1e-6,
                "col2 mean=8"
            );
            match fitted.transform(&x) {
                Ok(out) => {
                    assert_eq!(out.ncols(), 2, "sklearn out.shape=(2,2)");
                    assert_eq!(out.nrows(), 2);
                    assert!((out[[0, 0]] - 1.0f32).abs() < 1e-6);
                    assert!((out[[1, 0]] - 3.0f32).abs() < 1e-6);
                    assert!((out[[0, 1]] - 7.0f32).abs() < 1e-6);
                    assert!((out[[1, 1]] - 9.0f32).abs() < 1e-6);
                }
                #[allow(
                    clippy::assertions_on_constants,
                    reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
                )]
                Err(e) => assert!(false, "transform errored: {e}"),
            }
        }
        #[allow(
            clippy::assertions_on_constants,
            reason = "error arm fails loudly without panic!/unwrap (anti-pattern gate)"
        )]
        Err(e) => assert!(false, "fit errored: {e}"),
    }
}
