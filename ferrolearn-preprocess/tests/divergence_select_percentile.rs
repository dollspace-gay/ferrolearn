//! Divergence tests for `SelectPercentile` vs scikit-learn 1.5.2.
//!
//! Oracle: installed sklearn 1.5.2, all expected values computed by a LIVE
//! sklearn call from /tmp (R-CHAR-3 — never copied from the ferrolearn side).
//!
//! sklearn contract:
//! - `class SelectPercentile`:
//!   `/home/doll/scikit-learn/sklearn/feature_selection/_univariate_selection.py:589`
//! - `_get_support_mask`: same file `:669-686`
//! - `f_classif`: same file `:127`
//!
//! Shared fixture (all F-scores finite):
//! ```python
//! X=np.array([[1,2,3,4,5],[1.5,2.2,8,4.1,1],
//!             [9,2.1,3.2,9,5.1],[8.5,2.3,7.5,9.2,0.9]])
//! y=np.array([0,0,1,1])
//! ```
//! Live oracle `f_classif(X,y)[0]` =
//! `[450.0, 0.49999999999973355, 0.0020694412508616904, 2040.1999999995382, 0.0]`.
//!
//! Live oracle `get_support(indices=True)` per percentile:
//! `pct=0 -> []`, `pct=10 -> [3]`, `pct=30 -> [0, 3]`, `pct=50 -> [0, 3]`,
//! `pct=60 -> [0, 1, 3]`, `pct=100 -> [0, 1, 2, 3, 4]`.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::feature_selection::ScoreFunc;
use ferrolearn_preprocess::select_percentile::SelectPercentile;
use ndarray::{Array1, Array2, array};

/// The shared 4x5 fixture with all-finite F-scores.
fn fixture() -> (Array2<f64>, Array1<usize>) {
    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.5, 2.2, 8.0, 4.1, 1.0],
        [9.0, 2.1, 3.2, 9.0, 5.1],
        [8.5, 2.3, 7.5, 9.2, 0.9]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1];
    (x, y)
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-1: f_classif F-score values (live oracle).
// ---------------------------------------------------------------------------

/// GREEN GUARD: ferrolearn's `anova_f_scores` must match sklearn `f_classif`
/// (`_univariate_selection.py:127`) for the fixture.
/// Live oracle f_classif(X,y)[0] =
///   [450.0, 0.49999999999973355, 0.0020694412508616904,
///    2040.1999999995382, 0.0]
#[test]
fn guard_f_classif_scores_match() {
    let (x, y) = fixture();
    let fitted = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
        .fit(&x, &y)
        .unwrap();
    let scores = fitted.scores();

    // Live-oracle expected values (sklearn 1.5.2 f_classif).
    let expected = [
        450.0_f64,
        0.499_999_999_999_733_55,
        0.002_069_441_250_861_690_4,
        2_040.199_999_999_538_2,
        0.0,
    ];
    assert_eq!(scores.len(), expected.len());
    for (i, &e) in expected.iter().enumerate() {
        let a = scores[i];
        // Relative tolerance ~1e-6; for the exact-zero entry use absolute.
        if e == 0.0 {
            assert!(a.abs() < 1e-9, "score[{i}] = {a}, expected 0.0");
        } else {
            let rel = (a - e).abs() / e.abs();
            assert!(rel < 1e-6, "score[{i}] = {a}, expected {e} (rel err {rel})");
        }
    }
}

// ---------------------------------------------------------------------------
// DIV-1 / HEADLINE — REQ-2: selection mask `_get_support_mask` (:669-686).
// ferrolearn uses ceil rank-top-k; sklearn uses np.percentile threshold + `>`
// + int()-floor tie-fill. The headline divergence is at pct=50, where
// k=ceil(2.5)=3 in ferrolearn but sklearn keeps exactly [0, 3].
// ---------------------------------------------------------------------------

/// DIVERGENCE (DIV-1): ferrolearn `selected_indices()` diverges from sklearn
/// `_get_support_mask` (`_univariate_selection.py:669-686`) at percentile=50.
/// Live oracle: get_support(indices=True) -> [0, 3] (2 features).
/// ferrolearn: ceil-top-k with k=(5*50).div_ceil(100)=3 -> [0, 1, 3].
#[test]
fn divergence_select_percentile_mask_pct50() {
    let (x, y) = fixture();
    let fitted = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
        .fit(&x, &y)
        .unwrap();

    // Live oracle: sklearn keeps exactly features [0, 3].
    assert_eq!(fitted.selected_indices(), &[0usize, 3]);

    // And transform returns 2 columns (not 3).
    let out = fitted.transform(&x).unwrap();
    assert_eq!(out.ncols(), 2);
}

/// Full live-oracle target for `_get_support_mask` across percentiles on the
/// fixture. The fixer must make ALL of these pass simultaneously after
/// replacing ceil-top-k with sklearn's np.percentile-threshold algorithm.
///
/// Live oracle get_support(indices=True):
///   pct=0   -> []          (short-circuit, :676)
///   pct=10  -> [3]
///   pct=30  -> [0, 3]
///   pct=50  -> [0, 3]      (DIV-1 headline)
///   pct=60  -> [0, 1, 3]
///   pct=100 -> [0,1,2,3,4] (short-circuit, :674)
#[test]
fn divergence_select_percentile_mask_full_oracle() {
    let (x, y) = fixture();

    let cases: &[(usize, &[usize])] = &[
        (0, &[]),
        (10, &[3]),
        (30, &[0, 3]),
        (50, &[0, 3]),
        (60, &[0, 1, 3]),
        (100, &[0, 1, 2, 3, 4]),
    ];

    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f64>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "percentile={pct}: sklearn get_support -> {expected:?}, \
             ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-3: error contract.
// ---------------------------------------------------------------------------

/// GREEN GUARD: zero-row fit -> Err.
#[test]
fn guard_zero_rows_error() {
    let x: Array2<f64> = Array2::zeros((0, 3));
    let y: Array1<usize> = Array1::zeros(0);
    assert!(
        SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
            .fit(&x, &y)
            .is_err()
    );
}

/// GREEN GUARD: y-length mismatch -> Err.
#[test]
fn guard_y_length_mismatch_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y: Array1<usize> = array![0];
    assert!(
        SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
            .fit(&x, &y)
            .is_err()
    );
}

/// GREEN GUARD: percentile > 100 -> Err.
#[test]
fn guard_percentile_over_100_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y: Array1<usize> = array![0, 1];
    assert!(
        SelectPercentile::<f64>::new(150, ScoreFunc::FClassif)
            .fit(&x, &y)
            .is_err()
    );
}

/// GREEN GUARD: transform with wrong ncols -> Err.
#[test]
fn guard_transform_ncols_mismatch_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y: Array1<usize> = array![0, 1];
    let fitted = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif)
        .fit(&x, &y)
        .unwrap();
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(fitted.transform(&x_bad).is_err());
}

// ===========================================================================
// RE-AUDIT after #1274 fix. The fix replaced ceil rank-top-k with sklearn's
// np.percentile-threshold algorithm (_get_support_mask, :669-686). The DIV-1
// tests above are now GREEN guards. Below are FRESH live-oracle fixtures
// (computed in /tmp against installed sklearn 1.5.2, R-CHAR-3) stressing the
// novel numpy_percentile + threshold + int()-floor tie-fill code paths.
// ===========================================================================

// ---------------------------------------------------------------------------
// GUARD: larger 8-feature fixture at non-round percentiles (25/33/75/90).
// Exercises np.percentile linear interpolation + int()-floor on a wider grid.
// ---------------------------------------------------------------------------

/// GUARD: 8-feature fixture, selection matches sklearn `get_support` across
/// 25/33/75/90. Live oracle (sklearn 1.5.2 f_classif):
///   scores = [5.048937112488939, 0.02530991735536647, 0.5109597786763131,
///             1.4185983092446173, 0.002415925782759404, 4.460206185567025,
///             0.34152198005171897, 3.745006657789613]
///   pct=25 -> [0, 5]
///   pct=33 -> [0, 5, 7]
///   pct=75 -> [0, 2, 3, 5, 6, 7]
///   pct=90 -> [0, 1, 2, 3, 5, 6, 7]
#[test]
fn guard_eight_feature_nonround_percentiles() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.5, 2.2, 8.0, 4.1, 1.0, 6.5, 2.0, 8.1],
        [9.0, 2.1, 3.2, 9.0, 5.1, 1.0, 7.5, 0.5],
        [8.5, 2.3, 7.5, 9.2, 0.9, 1.2, 2.2, 0.4],
        [2.0, 5.0, 3.0, 1.0, 9.0, 6.1, 7.1, 8.0],
        [2.1, 5.5, 8.2, 1.1, 9.5, 6.0, 2.1, 8.2]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1, 0, 1];

    let cases: &[(usize, &[usize])] = &[
        (25, &[0, 5]),
        (33, &[0, 5, 7]),
        (75, &[0, 2, 3, 5, 6, 7]),
        (90, &[0, 1, 2, 3, 5, 6, 7]),
    ];
    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f64>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "pct={pct}: sklearn get_support -> {expected:?}, ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}

// ---------------------------------------------------------------------------
// GUARD: TIE fixture (3 duplicated columns share an IDENTICAL F-score of 0.0
// landing exactly ON the threshold). Exercises int()-floor partial tie-fill in
// ascending index order (:683-685).
// ---------------------------------------------------------------------------

/// GUARD: tie fixture, partial tie-fill matches sklearn across 40..=80.
/// Columns 1,2,3 are duplicates -> identical F-score 0.0.
/// Live oracle (sklearn 1.5.2 f_classif):
///   scores = [1993.923076923147, 0.0, 0.0, 0.0, 2928.199999999337]
///   pct=40 thr=797.57   -> [0, 4]        (no ties on thr)
///   pct=50 thr=0.0      -> [0, 4]        maxfeats=2, mask.sum()=2, fill 0
///   pct=60 thr=0.0      -> [0, 1, 4]     maxfeats=3, fill 1 (idx 1 only)
///   pct=70 thr=0.0      -> [0, 1, 4]     maxfeats=3, fill 1
///   pct=80 thr=0.0      -> [0, 1, 2, 4]  maxfeats=4, fill 2 (idx 1,2 ascending)
#[test]
fn guard_tie_partial_fill_ascending() {
    // base cols: 0 = strong, 1 = weak (score 0.0), 2 = strong.
    // Duplicate the weak col into indices 1,2,3.
    let x = array![
        [1.0, 5.0, 5.0, 5.0, 2.0],
        [1.2, 6.0, 6.0, 6.0, 2.1],
        [9.0, 5.0, 5.0, 5.0, 8.0],
        [9.3, 6.0, 6.0, 6.0, 8.2]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1];

    let cases: &[(usize, &[usize])] = &[
        (40, &[0, 4]),
        (50, &[0, 4]),
        (60, &[0, 1, 4]),
        (70, &[0, 1, 4]),
        (80, &[0, 1, 2, 4]),
    ];
    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f64>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "tie pct={pct}: sklearn get_support -> {expected:?}, ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}

// ---------------------------------------------------------------------------
// GUARD: threshold interpolation branches. The lo==hi branch (threshold lands
// on an exact data point) vs the between-point branch of numpy_percentile.
// ---------------------------------------------------------------------------

/// GUARD: lo==hi exact-data-point branch of numpy_percentile. 5-feature
/// fixture; with n=5, q=100-pct hits an integer index for pct in {25,50,75}
/// (idx = 3.0/2.0/1.0), so threshold == an exact score. Live oracle
/// (sklearn 1.5.2):
///   scores = [450.0, 0.49999999999973355, 0.0020694412508616904,
///             2040.1999999995382, 0.0]
///   pct=25 (q=75, idx=3.0, thr=450.0)            -> [3]
///   pct=50 (q=50, idx=2.0, thr=0.49999999999973) -> [0, 3]
///   pct=75 (q=25, idx=1.0, thr=0.00206944125086) -> [0, 1, 3]
#[test]
fn guard_interp_lo_eq_hi_exact_point() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.5, 2.2, 8.0, 4.1, 1.0],
        [9.0, 2.1, 3.2, 9.0, 5.1],
        [8.5, 2.3, 7.5, 9.2, 0.9]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1];
    let cases: &[(usize, &[usize])] = &[(25, &[3]), (50, &[0, 3]), (75, &[0, 1, 3])];
    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f64>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "lo==hi pct={pct}: sklearn -> {expected:?}, ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}

/// GUARD: between-point (lo!=hi) branch of numpy_percentile. 8-feature fixture;
/// q=100-pct lands at a fractional index for pct in {13,37,62}
/// (idx=6.09/4.41/2.66). Live oracle (sklearn 1.5.2):
///   pct=13 (q=87, idx=6.09) -> [0]
///   pct=37 (q=63, idx=4.41) -> [0, 5, 7]
///   pct=62 (q=38, idx=2.66) -> [0, 2, 3, 5, 7]
#[test]
fn guard_interp_between_points() {
    let x = array![
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.5, 2.2, 8.0, 4.1, 1.0, 6.5, 2.0, 8.1],
        [9.0, 2.1, 3.2, 9.0, 5.1, 1.0, 7.5, 0.5],
        [8.5, 2.3, 7.5, 9.2, 0.9, 1.2, 2.2, 0.4],
        [2.0, 5.0, 3.0, 1.0, 9.0, 6.1, 7.1, 8.0],
        [2.1, 5.5, 8.2, 1.1, 9.5, 6.0, 2.1, 8.2]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1, 0, 1];
    let cases: &[(usize, &[usize])] = &[(13, &[0]), (37, &[0, 5, 7]), (62, &[0, 2, 3, 5, 7])];
    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f64>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "between pct={pct}: sklearn -> {expected:?}, ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}

// ---------------------------------------------------------------------------
// GUARD: f32 path. Scores computed in f32 (ferrolearn) vs float64 (sklearn);
// the SELECTION (get_support indices) must still match since ordering is
// preserved for these well-separated finite scores. Assert on indices only.
// ---------------------------------------------------------------------------

/// GUARD: f32 fixture selection matches sklearn across 33/50/66.
/// Live oracle (sklearn 1.5.2 f_classif, float64 internally):
///   scores ~= [5.398410189, 0.0366149922, 0.6146818045, 0.000371178971]
///   pct=33 -> [0]
///   pct=50 -> [0, 2]
///   pct=66 -> [0, 2]
#[test]
fn guard_f32_path_selection() {
    let x: Array2<f32> = array![
        [1.0, 2.0, 3.0, 7.0],
        [1.3, 2.5, 8.0, 1.0],
        [9.0, 2.2, 3.5, 7.1],
        [8.7, 2.6, 7.7, 0.9],
        [2.0, 5.0, 3.1, 9.0],
        [2.2, 5.5, 8.1, 9.2]
    ];
    let y: Array1<usize> = array![0, 0, 1, 1, 0, 1];

    let cases: &[(usize, &[usize])] = &[(33, &[0]), (50, &[0, 2]), (66, &[0, 2])];
    for &(pct, expected) in cases {
        let fitted = SelectPercentile::<f32>::new(pct, ScoreFunc::FClassif)
            .fit(&x, &y)
            .unwrap();
        assert_eq!(
            fitted.selected_indices(),
            expected,
            "f32 pct={pct}: sklearn get_support -> {expected:?}, ferrolearn -> {:?}",
            fitted.selected_indices()
        );
    }
}
