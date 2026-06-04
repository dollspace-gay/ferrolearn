//! Divergence audit for `TimeSeriesSplit` against scikit-learn 1.5.2.
//!
//! `TimeSeriesSplit` is a DETERMINISTIC index-arithmetic splitter
//! (`sklearn/model_selection/_split.py:1062`, `_split` `:1219-1271`). Every
//! fold's `(train_indices, test_indices)` is a pure function of
//! `(n_samples, n_splits, test_size, max_train_size, gap)`, so every case below
//! is oracle-pinned against the LIVE sklearn 1.5.2 result of
//! `list(TimeSeriesSplit(...).split(np.arange(n)))` run from `/tmp` (R-CHAR-3:
//! expected index lists are the oracle output, NEVER copied from ferrolearn).
//!
//! Each green guard compares ferrolearn's REAL `.split(n)` output to the live
//! oracle, fold-by-fold, element-wise. A failing test would be pinned with an
//! `#[ignore]` tracking tag; none are needed (see report) because the live
//! 7500-case sweep found zero fold-content and zero error-status divergences.
//!
//! Tracking: #1743.

use ferrolearn_model_sel::TimeSeriesSplit;

/// Build the expected fold list as `Vec<(train, test)>` from explicit ranges,
/// keeping the test bodies readable. The values themselves are the LIVE sklearn
/// oracle outputs documented per-test (R-CHAR-3).
fn folds(
    spec: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    spec.iter()
        .map(|(tr, te)| (tr.clone().collect(), te.clone().collect()))
        .collect()
}

// ---------------------------------------------------------------------------
// REQ-1 — default split-index parity (test_size=None, gap=0, max_train_size=None)
// ---------------------------------------------------------------------------

/// Divergence guard: ferrolearn `TimeSeriesSplit::new(3).split(12)` vs
/// `sklearn/model_selection/_split.py:1257-1271`.
///
/// Live oracle (sklearn 1.5.2, from /tmp):
/// `TimeSeriesSplit(n_splits=3).split(np.arange(12))` ->
/// `([0,1,2],[3,4,5]) ([0..6],[6,7,8]) ([0..9],[9,10,11])`
///
/// Tracking: #1743
#[test]
fn green_req1_default_n3_n12() {
    let got = TimeSeriesSplit::new(3).split(12).unwrap();
    let expected = folds(&[(0..3, 3..6), (0..6, 6..9), (0..9, 9..12)]);
    assert_eq!(got, expected);
}

/// Divergence guard: ferrolearn `TimeSeriesSplit::new(5).split(20)` vs
/// `sklearn/model_selection/_split.py:1241-1271`.
///
/// Live oracle: `TimeSeriesSplit(n_splits=5).split(np.arange(20))`,
/// test_size = 20 // 6 = 3:
/// `([0..5],[5,6,7]) ([0..8],[8,9,10]) ([0..11],[11,12,13])`
/// `([0..14],[14,15,16]) ([0..17],[17,18,19])`
///
/// Tracking: #1743
#[test]
fn green_req1_default_n5_n20() {
    let got = TimeSeriesSplit::new(5).split(20).unwrap();
    let expected = folds(&[
        (0..5, 5..8),
        (0..8, 8..11),
        (0..11, 11..14),
        (0..14, 14..17),
        (0..17, 17..20),
    ]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-2 — explicit test_size override
// ---------------------------------------------------------------------------

/// Divergence guard: `TimeSeriesSplit::new(3).test_size(Some(2)).split(12)` vs
/// `sklearn/model_selection/_split.py:1241-1242` (the `self.test_size if not
/// None` branch).
///
/// Live oracle: `TimeSeriesSplit(n_splits=3, test_size=2).split(np.arange(12))`:
/// `([0..6],[6,7]) ([0..8],[8,9]) ([0..10],[10,11])`
///
/// Tracking: #1743
#[test]
fn green_req2_test_size_2() {
    let got = TimeSeriesSplit::new(3)
        .test_size(Some(2))
        .split(12)
        .unwrap();
    let expected = folds(&[(0..6, 6..8), (0..8, 8..10), (0..10, 10..12)]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-3 — gap
// ---------------------------------------------------------------------------

/// Divergence guard: `TimeSeriesSplit::new(3).gap(2).split(12)` vs
/// `sklearn/model_selection/_split.py:1261` (`train_end = test_start - gap`).
///
/// Live oracle: `TimeSeriesSplit(n_splits=3, gap=2).split(np.arange(12))`,
/// test_size = 12 // 4 = 3:
/// `([0],[3,4,5]) ([0,1,2,3],[6,7,8]) ([0..7],[9,10,11])`
///
/// Tracking: #1743
#[test]
fn green_req3_gap_2() {
    let got = TimeSeriesSplit::new(3).gap(2).split(12).unwrap();
    let expected = folds(&[(0..1, 3..6), (0..4, 6..9), (0..7, 9..12)]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-4 — max_train_size (incl. the no-cap fold-0 edge)
// ---------------------------------------------------------------------------

/// Divergence guard: `TimeSeriesSplit::new(3).max_train_size(Some(4)).split(12)`
/// vs `sklearn/model_selection/_split.py:1262-1271` (cap applied ONLY when
/// `max_train_size < train_end`).
///
/// Live oracle: `TimeSeriesSplit(n_splits=3, max_train_size=4).split(np.arange(12))`:
/// `([0,1,2],[3,4,5])` (no cap: 4 >= train_end=3)
/// `([2,3,4,5],[6,7,8]) ([5,6,7,8],[9,10,11])`
///
/// Tracking: #1743
#[test]
fn green_req4_max_train_size_4() {
    let got = TimeSeriesSplit::new(3)
        .max_train_size(Some(4))
        .split(12)
        .unwrap();
    let expected = folds(&[(0..3, 3..6), (2..6, 6..9), (5..9, 9..12)]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-2/3/4 — combined test_size + gap + max_train_size
// ---------------------------------------------------------------------------

/// Divergence guard: combined config
/// `TimeSeriesSplit::new(3).test_size(Some(2)).gap(1).max_train_size(Some(3)).split(14)`
/// vs `sklearn/model_selection/_split.py:1258-1271`.
///
/// Live oracle:
/// `TimeSeriesSplit(n_splits=3, test_size=2, gap=1, max_train_size=3).split(np.arange(14))`:
/// `([4,5,6],[8,9]) ([6,7,8],[10,11]) ([8,9,10],[12,13])`
///
/// Tracking: #1743
#[test]
fn green_req234_combined() {
    let got = TimeSeriesSplit::new(3)
        .test_size(Some(2))
        .gap(1)
        .max_train_size(Some(3))
        .split(14)
        .unwrap();
    let expected = folds(&[(4..7, 8..10), (6..9, 10..12), (8..11, 12..14)]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-1 — non-divisible n exercising the test_size floor + test_starts range
// ---------------------------------------------------------------------------

/// Divergence guard: non-divisible `TimeSeriesSplit::new(4).split(13)` vs
/// `sklearn/model_selection/_split.py:1242` (`test_size = n // (n_splits+1) =
/// 13 // 5 = 2`) and `:1258` (`test_starts = range(13 - 4*2, 13, 2) = [5,7,9,11]`).
///
/// Live oracle: `TimeSeriesSplit(n_splits=4).split(np.arange(13))`:
/// `([0..5],[5,6]) ([0..7],[7,8]) ([0..9],[9,10]) ([0..11],[11,12])`
///
/// Tracking: #1743
#[test]
fn green_req1_nondivisible_n4_n13() {
    let got = TimeSeriesSplit::new(4).split(13).unwrap();
    let expected = folds(&[(0..5, 5..7), (0..7, 7..9), (0..9, 9..11), (0..11, 11..13)]);
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// REQ-5 — error semantics: ferrolearn Err <=> sklearn ValueError on same inputs
// ---------------------------------------------------------------------------

/// Divergence guard (error parity): inputs where sklearn raises `ValueError`
/// must make ferrolearn return `Err`.
///
/// Live oracle (sklearn 1.5.2, all raise ValueError):
/// `TimeSeriesSplit(n_splits=10).split(np.arange(5))` (`:1246` n_folds=11 > 5);
/// `TimeSeriesSplit(n_splits=3, test_size=4).split(np.arange(12))`
/// (`:1251` 12 - 0 - 12 <= 0);
/// `TimeSeriesSplit(n_splits=2, test_size=2, gap=100).split(np.arange(10))`
/// (`:1251` 10 - 100 - 4 <= 0).
///
/// Tracking: #1743
#[test]
fn green_req5_error_parity() {
    assert!(
        TimeSeriesSplit::new(10).split(5).is_err(),
        "sklearn raises ValueError (n_folds=11 > n_samples=5)"
    );
    assert!(
        TimeSeriesSplit::new(3)
            .test_size(Some(4))
            .split(12)
            .is_err(),
        "sklearn raises ValueError (12 - 0 - 4*3 <= 0)"
    );
    assert!(
        TimeSeriesSplit::new(2)
            .test_size(Some(2))
            .gap(100)
            .split(10)
            .is_err(),
        "sklearn raises ValueError (10 - 100 - 2*2 <= 0)"
    );
}

// ---------------------------------------------------------------------------
// REQ-6 — n_splits < 2 rejected (both reject)
// ---------------------------------------------------------------------------

/// Divergence guard: `n_splits < 2` rejected by both.
///
/// Live oracle: `TimeSeriesSplit(n_splits=1)` raises `ValueError`
/// (`_BaseKFold` `Interval(Integral, 2, None, closed="left")`).
///
/// Tracking: #1743
#[test]
fn green_req6_n_splits_less_than_2() {
    assert!(TimeSeriesSplit::new(1).split(20).is_err());
    assert!(TimeSeriesSplit::new(0).split(20).is_err());
}
