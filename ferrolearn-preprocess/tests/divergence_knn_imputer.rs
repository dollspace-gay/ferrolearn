//! ACToR critic: oracle-grounded audit of `ferrolearn-preprocess`'s
//! `KNNImputer` (`ferrolearn-preprocess/src/knn_imputer.rs`) against
//! scikit-learn 1.5.2 `class KNNImputer` (`sklearn/impute/_knn.py`) +
//! `nan_euclidean_distances` (`sklearn/metrics/pairwise.py:430-549`).
//!
//! All expected values are derived from the LIVE sklearn 1.5.2 oracle, run from
//! `/tmp` (R-CHAR-3): NEVER literal-copied from the ferrolearn side. Oracle
//! version confirmed: `python3 -c "import sklearn; print(sklearn.__version__)"`
//! -> `1.5.2`.
//!
//! Oracle commands:
//! ```text
//! # REQ-1 basic uniform (no DIV-1/DIV-2 bite):
//! python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
//!   print(KNNImputer(n_neighbors=2).fit_transform([[1,2],[3,4],[5,nan]]).tolist())"
//!   -> [[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]
//!
//! # DIV-1 (REQ-2) uniform scaling flips nearest donor:
//! python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
//!   X=np.array([[0.,0.,nan,0.],[3.,3.,10.,3.],[nan,nan,20.,5.]]); \
//!   print(KNNImputer(n_neighbors=1).fit_transform(X)[0,2])"
//!   -> 10.0   (ferrolearn unscaled picks donorB -> 20.0)
//!
//! # DIV-1 (REQ-2) distance-weighted: scaling changes the weights -> different value:
//! python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
//!   X=np.array([[0.,0.,nan,0.],[3.,3.,10.,3.],[nan,nan,20.,5.],[1.,nan,30.,nan]]); \
//!   print(KNNImputer(n_neighbors=2, weights='distance').fit_transform(X)[0,2])"
//!   -> 25.000000000000004   (ferrolearn unscaled blend -> 28.333...)
//!
//! # DIV-2 (REQ-3) empty-donor -> masked training column mean (not 0):
//! python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
//!   print(KNNImputer(n_neighbors=2).fit_transform([[nan,1.],[nan,2.],[5.,nan]]).tolist())"
//!   -> [[5.0, 1.0], [5.0, 2.0], [5.0, 1.5]]   (ferrolearn fills 0.0)
//!
//! # DIV-3 (REQ-9) n_neighbors > n_samples does NOT raise (sklearn clamps, _knn.py:349):
//! python3 -c "import numpy as np; from sklearn.impute import KNNImputer; nan=np.nan; \
//!   print(KNNImputer(n_neighbors=10).fit_transform([[1.,2.],[3.,4.],[5.,nan]]).tolist())"
//!   -> [[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]   (ferrolearn fit() raises InvalidParameter)
//! ```

use approx::assert_abs_diff_eq;
use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::knn_imputer::{KNNImputer, KNNWeights};
use ndarray::{Array2, array};

// ===========================================================================
// GREEN guards — SHIPPED behavior vs the LIVE sklearn oracle (R-CHAR-3).
// These must PASS against the current implementation.
// ===========================================================================

/// Guard (REQ-1): basic uniform impute on a fixture where every row shares all
/// present features, so the DIV-1 `n_features / present_count` scale is a
/// constant that cancels in the neighbor ordering and (for uniform weights)
/// does not affect the mean.
///
/// Live oracle: `KNNImputer(n_neighbors=2).fit_transform([[1,2],[3,4],[5,nan]])`
/// -> `[[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]`. The imputed cell `[2,1]` is the
/// mean of the two present donors `mean(2, 4) = 3.0`.
#[test]
fn guard_knn_uniform_basic_matches_sklearn_oracle() {
    // Live oracle output (sklearn 1.5.2).
    let sklearn_imputed_2_1 = 3.0_f64;
    let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN]];
    let out = imputer.fit_transform(&x).unwrap();
    assert_abs_diff_eq!(out[[2, 1]], sklearn_imputed_2_1, epsilon = 1e-10);
    // Present cells are unchanged.
    assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out[[1, 1]], 4.0, epsilon = 1e-10);
}

/// Guard (REQ-9): `fit` on 0 rows returns `Err` (sklearn would also reject /
/// produce no usable model — ferrolearn maps to `InsufficientSamples`).
#[test]
fn guard_knn_zero_rows_errors() {
    let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
    let x: Array2<f64> = Array2::zeros((0, 3));
    assert!(imputer.fit(&x, &()).is_err());
}

/// Guard (REQ-9): `n_neighbors == 0` returns `Err`, matching sklearn's
/// `Interval(Integral, 1, None, closed="left")` constraint (`_knn.py:132`),
/// which rejects `n_neighbors < 1`.
#[test]
fn guard_knn_zero_neighbors_errors() {
    let imputer = KNNImputer::<f64>::new(0, KNNWeights::Uniform);
    let x = array![[1.0, 2.0]];
    assert!(imputer.fit(&x, &()).is_err());
}

/// Guard (REQ-9): a `transform` column-count mismatch returns `Err`.
#[test]
fn guard_knn_transform_ncols_mismatch_errors() {
    let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
    let x_train = array![[1.0, 2.0], [3.0, 4.0]];
    let fitted = imputer.fit(&x_train, &()).unwrap();
    let x_bad = array![[1.0, 2.0, 3.0]];
    assert!(fitted.transform(&x_bad).is_err());
}

// ===========================================================================
// DIV-1 (REQ-2): nan_euclidean distance SCALING.
//
// Divergence: ferrolearn's `partial_euclidean_distance`
// (`ferrolearn-preprocess/src/knn_imputer.rs:148`) returns the unscaled
// `sum_sq.sqrt()` ("But we keep it simple here: just use sqrt(sum_sq)",
// `:147`); sklearn `nan_euclidean_distances` returns
// `sqrt(sum_sq * n_features / present_count)` (`pairwise.py:543-547`:
// `distances /= present_count; distances *= X.shape[1]; np.sqrt(...)`).
// When donors share differing numbers of present features with the receiver,
// the scale reorders the neighbors and flips the imputed value.
// ===========================================================================

/// Divergence (DIV-1, uniform, headline): on the 4-feature fixture
/// `X = [[0,0,nan,0],[3,3,10,3],[nan,nan,20,5]]`, `n_neighbors=1`, the receiver
/// row 0 (present cols 0,1,3) needs col 2. sklearn's SCALED distances row0->all
/// are `[0, 6.0, 10.0]` (donorA `sqrt(27*4/3)=6.0`, donorB `sqrt(25*4/1)=10.0`),
/// so the nearest donor is row 1 -> imputes `10.0`. ferrolearn's UNSCALED
/// distances are `sqrt(27)=5.196` / `sqrt(25)=5.0`, picking row 2 -> imputes
/// `20.0`.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=1).fit_transform(
///   [[0,0,nan,0],[3,3,10,3],[nan,nan,20,5]])[0,2]` -> `10.0`.
/// Tracking: #1305
#[test]
fn divergence_div1_uniform_scaling_flips_donor() {
    // Live sklearn oracle value (NOT from ferrolearn): 10.0.
    let sklearn_imputed_0_2 = 10.0_f64;
    let x = array![
        [0.0, 0.0, f64::NAN, 0.0],
        [3.0, 3.0, 10.0, 3.0],
        [f64::NAN, f64::NAN, 20.0, 5.0]
    ];
    let imputer = KNNImputer::<f64>::new(1, KNNWeights::Uniform);
    let out = imputer.fit_transform(&x).unwrap();
    assert_abs_diff_eq!(out[[0, 2]], sklearn_imputed_0_2, epsilon = 1e-9);
}

/// Divergence (DIV-1, distance-weighted): on
/// `X = [[0,0,nan,0],[3,3,10,3],[nan,nan,20,5],[1,nan,30,nan]]`, `n_neighbors=2`,
/// `weights='distance'`, the receiver row 0 needs col 2. sklearn's SCALED
/// distances row0->donors are `[6.0 (val10), 10.0 (val20), 2.0 (val30)]`, so the
/// 2 nearest are row 3 (d=2, val 30) and row 1 (d=6, val 10); inverse-distance
/// blend `(30/2 + 10/6)/(1/2 + 1/6) = 25.0`. ferrolearn's UNSCALED distances are
/// `[5.196 (val10), 5.0 (val20), 1.0 (val30)]`, so its 2 nearest are row 3
/// (d=1, val 30) and row 2 (d=5, val 20); blend `(30/1 + 20/5)/(1 + 1/5) ≈ 28.33`.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=2, weights='distance').fit_transform(
///   [[0,0,nan,0],[3,3,10,3],[nan,nan,20,5],[1,nan,30,nan]])[0,2]`
/// -> `25.000000000000004`.
/// Tracking: #1305
#[test]
fn divergence_div1_distance_scaling_changes_weights() {
    // Live sklearn oracle value (NOT from ferrolearn): 25.0.
    let sklearn_imputed_0_2 = 25.000000000000004_f64;
    let x = array![
        [0.0, 0.0, f64::NAN, 0.0],
        [3.0, 3.0, 10.0, 3.0],
        [f64::NAN, f64::NAN, 20.0, 5.0],
        [1.0, f64::NAN, 30.0, f64::NAN]
    ];
    let imputer = KNNImputer::<f64>::new(2, KNNWeights::Distance);
    let out = imputer.fit_transform(&x).unwrap();
    assert_abs_diff_eq!(out[[0, 2]], sklearn_imputed_0_2, epsilon = 1e-9);
}

// ===========================================================================
// DIV-2 (REQ-3): empty-donor -> masked training COLUMN MEAN, not 0.
//
// Divergence: ferrolearn fills `F::zero()` when `neighbor_vals.is_empty()`
// (`ferrolearn-preprocess/src/knn_imputer.rs:270`); sklearn imputes the masked
// training column mean `np.ma.array(self._fit_X[:, col],
// mask=mask_fit_X[:, col]).mean()` for receivers with all-NaN distances
// (`_knn.py:329-337`).
// ===========================================================================

/// Divergence (DIV-2): on `X = [[nan,1],[nan,2],[5,nan]]`, `n_neighbors=2`, the
/// receiver row 2 needs col 1, but its only present feature is col 0, which is
/// missing in every row that has col 1 present (rows 0,1) -> no shared feature
/// -> all-NaN distances -> no reachable donor. sklearn imputes the masked mean
/// of training column 1 = `mean(1, 2) = 1.5`. ferrolearn fills `0.0`.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=2).fit_transform([[nan,1],[nan,2],[5,nan]])`
/// -> `[[5.0, 1.0], [5.0, 2.0], [5.0, 1.5]]`; cell `[2,1] == 1.5`.
/// Tracking: #1306
#[test]
fn divergence_div2_empty_donor_imputes_column_mean() {
    // Live sklearn oracle value (NOT from ferrolearn): 1.5 (masked col-1 mean).
    let sklearn_imputed_2_1 = 1.5_f64;
    let x = array![[f64::NAN, 1.0], [f64::NAN, 2.0], [5.0, f64::NAN]];
    let imputer = KNNImputer::<f64>::new(2, KNNWeights::Uniform);
    let out = imputer.fit_transform(&x).unwrap();
    assert_abs_diff_eq!(out[[2, 1]], sklearn_imputed_2_1, epsilon = 1e-10);
}

// ===========================================================================
// DIV-3 (REQ-9): n_neighbors > n_samples must NOT raise.
//
// Divergence: ferrolearn `fit` errors when `n_neighbors > n_samples`
// (`ferrolearn-preprocess/src/knn_imputer.rs:182-190`); sklearn does NOT
// validate this — `_parameter_constraints` only requires `n_neighbors >= 1`
// (`_knn.py:132`), and `transform` clamps via
// `n_neighbors = min(self.n_neighbors, len(potential_donors_idx))`
// (`_knn.py:349`).
// ===========================================================================

/// Divergence (DIV-3): `KNNImputer(n_neighbors=10)` on a 3-row dataset does NOT
/// raise in sklearn; it imputes using the available donors. The in-module test
/// `test_knn_imputer_too_many_neighbors_error` PINS the wrong `Err` (R-HONEST-4)
/// and must be rewritten by the generator.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=10).fit_transform([[1,2],[3,4],[5,nan]])`
/// -> `[[1.0, 2.0], [3.0, 4.0], [5.0, 3.0]]` (no exception; cell `[2,1] == 3.0`).
/// Tracking: #1307
#[test]
fn divergence_div3_n_neighbors_gt_n_samples_does_not_error() {
    // sklearn does not raise; the imputed cell is mean(2, 4) = 3.0 (live oracle).
    let sklearn_imputed_2_1 = 3.0_f64;
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, f64::NAN]];
    let imputer = KNNImputer::<f64>::new(10, KNNWeights::Uniform);
    // sklearn: no error. ferrolearn fit() currently returns Err -> this fails.
    let fitted = imputer
        .fit(&x, &())
        .expect("sklearn does not reject n_neighbors > n_samples (it clamps, _knn.py:349)");
    let out = fitted.transform(&x).unwrap();
    assert_abs_diff_eq!(out[[2, 1]], sklearn_imputed_2_1, epsilon = 1e-10);
}

// ===========================================================================
// RE-AUDIT (post-fix #1305/#1306/#1307): fresh live-oracle fixtures that stress
// the NEW arithmetic (n_features/n_valid scaling + per-column clamp + masked
// column-mean fallback) jointly. All expected values are the LIVE sklearn 1.5.2
// oracle, run from /tmp (R-CHAR-3). These GREEN guards must PASS — they confirm
// the three fixes are value-faithful, not merely "no longer the old wrong value".
// ===========================================================================

/// Helper: assert two row-major matrices match within `eps`.
fn assert_mat_eq(out: &Array2<f64>, expected: &[[f64; 4]], eps: f64) {
    for (i, row) in expected.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            assert_abs_diff_eq!(out[[i, j]], e, epsilon = eps);
        }
    }
}

/// Guard (re-audit): realistic 6x4 multi-row / multi-missing fixture, UNIFORM
/// weights, n_neighbors=2. NaNs in different rows AND columns so scaling +
/// neighbor selection + uniform average all interact.
///
/// Live oracle:
/// `X=[[1,2,nan,4],[5,nan,7,8],[nan,10,11,12],[13,14,15,nan],[17,18,nan,20],[21,nan,23,24]]`
/// `KNNImputer(n_neighbors=2).fit_transform(X)` ->
/// row0=[1,2,9,4] row1=[5,6,7,8] row2=[9,10,11,12] row3=[13,14,15,16]
/// row4=[17,18,19,20] row5=[21,16,23,24].
#[test]
fn guard_reaudit_6x4_uniform_k2_full_matrix() {
    let x = array![
        [1.0, 2.0, f64::NAN, 4.0],
        [5.0, f64::NAN, 7.0, 8.0],
        [f64::NAN, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, f64::NAN],
        [17.0, 18.0, f64::NAN, 20.0],
        [21.0, f64::NAN, 23.0, 24.0]
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    let expected = [
        [1.0, 2.0, 9.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0],
        [21.0, 16.0, 23.0, 24.0],
    ];
    assert_mat_eq(&out, &expected, 1e-9);
}

/// Guard (re-audit): same 6x4 fixture, UNIFORM weights, n_neighbors=3.
/// Live oracle `KNNImputer(n_neighbors=3).fit_transform(X)` ->
/// row0=[1,2,11,4] row1=[5,8.6666...,7,8] row2=[6.3333...,10,11,12]
/// row3=[13,14,15,13.3333...] row4=[17,18,16.3333...,20] row5=[21,14,23,24].
#[test]
fn guard_reaudit_6x4_uniform_k3_full_matrix() {
    let x = array![
        [1.0, 2.0, f64::NAN, 4.0],
        [5.0, f64::NAN, 7.0, 8.0],
        [f64::NAN, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, f64::NAN],
        [17.0, 18.0, f64::NAN, 20.0],
        [21.0, f64::NAN, 23.0, 24.0]
    ];
    let out = KNNImputer::<f64>::new(3, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    let expected = [
        [1.0, 2.0, 11.0, 4.0],
        [5.0, 8.666_666_666_666_666, 7.0, 8.0],
        [6.333_333_333_333_333, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 13.333_333_333_333_334],
        [17.0, 18.0, 16.333_333_333_333_332, 20.0],
        [21.0, 14.0, 23.0, 24.0],
    ];
    assert_mat_eq(&out, &expected, 1e-9);
}

/// Guard (re-audit): same 6x4 fixture, DISTANCE weights, n_neighbors=3. Stresses
/// the scaled distance feeding the inverse-distance blend across many pairs.
/// Live oracle `KNNImputer(n_neighbors=3, weights='distance').fit_transform(X)`
/// -> row0=[1,2,9.5454...,4] row1=[5,7.6,7,8] row2=[7.4,10,11,12]
/// row3=[13,14,15,14.4] row4=[17,18,17.4,20] row5=[21,15.4545...,23,24].
#[test]
fn guard_reaudit_6x4_distance_k3_full_matrix() {
    let x = array![
        [1.0, 2.0, f64::NAN, 4.0],
        [5.0, f64::NAN, 7.0, 8.0],
        [f64::NAN, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, f64::NAN],
        [17.0, 18.0, f64::NAN, 20.0],
        [21.0, f64::NAN, 23.0, 24.0]
    ];
    let out = KNNImputer::<f64>::new(3, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    let expected = [
        [1.0, 2.0, 9.545_454_545_454_545, 4.0],
        [5.0, 7.6, 7.0, 8.0],
        [7.4, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 14.4],
        [17.0, 18.0, 17.4, 20.0],
        [21.0, 15.454_545_454_545_455, 23.0, 24.0],
    ];
    assert_mat_eq(&out, &expected, 1e-9);
}

/// Guard (re-audit): receivers with DIFFERENT present-feature counts so the
/// `n_features / n_valid` scaling genuinely VARIES per (receiver, donor) pair.
/// Row 0 has 3 present features (cols 0,1,2); row 1 has 2 present (cols 2,3).
/// Live oracle `X=[[1,2,3,nan],[nan,nan,6,7],[10,11,12,13],[20,21,22,23],[30,31,32,33]]`
/// `KNNImputer(n_neighbors=2).fit_transform(X)`:
/// row0=[1,2,3,10] row1=[5.5,6.5,6,7] (rest unchanged).
#[test]
fn guard_reaudit_varying_present_counts_uniform_k2() {
    let x = array![
        [1.0, 2.0, 3.0, f64::NAN],
        [f64::NAN, f64::NAN, 6.0, 7.0],
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0]
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    assert_abs_diff_eq!(out[[0, 3]], 10.0, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 0]], 5.5, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 1]], 6.5, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 2]], 6.0, epsilon = 1e-9);
}

/// Guard (re-audit): n_neighbors LARGER than available donors for a column
/// (per-column clamp, `_knn.py:349`). Column 1 has only 2 donors (rows 2,3) but
/// n_neighbors=4 — sklearn clamps to 2 and averages BOTH.
/// Live oracle `X=[[1,nan],[2,nan],[3,100],[9,200]]`,
/// `KNNImputer(n_neighbors=4).fit_transform(X)` -> col1 of rows 0,1 == 150.0.
#[test]
fn guard_reaudit_clamp_more_neighbors_than_donors_uniform() {
    let x = array![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 100.0], [9.0, 200.0]];
    let out = KNNImputer::<f64>::new(4, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    assert_abs_diff_eq!(out[[0, 1]], 150.0, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 1]], 150.0, epsilon = 1e-9);
}

/// Guard (re-audit): same clamp fixture with DISTANCE weights.
/// Live oracle -> row0 col1 == 119.99999999999999, row1 col1 == 112.49999999999999.
#[test]
fn guard_reaudit_clamp_more_neighbors_than_donors_distance() {
    let x = array![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 100.0], [9.0, 200.0]];
    let out = KNNImputer::<f64>::new(4, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    assert_abs_diff_eq!(out[[0, 1]], 119.999_999_999_999_99, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 1]], 112.499_999_999_999_99, epsilon = 1e-9);
}

/// Guard (re-audit): empty-donor column-mean fallback (#1306) where a receiver
/// truly has NO reachable donor for a column AND no NaN-distance donor (so
/// ferrolearn and sklearn agree). Receiver row 2 needs col 1; its only present
/// feature col 0 is missing in every col-1 donor (rows 0,1) -> all-NaN distance,
/// no donor at all -> masked col-1 mean.
/// Live oracle `X=[[nan,1],[nan,2],[5,nan]]`,
/// `KNNImputer(n_neighbors=2).fit_transform(X)` -> [2,1] == 1.5.
#[test]
fn guard_reaudit_pure_empty_donor_column_mean() {
    let x = array![[f64::NAN, 1.0], [f64::NAN, 2.0], [5.0, f64::NAN]];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    assert_abs_diff_eq!(out[[2, 1]], 1.5, epsilon = 1e-9);
}

/// Guard (re-audit): f32 path matches the f64 live oracle within f32 tolerance.
/// Live oracle (f64) for `X=[[1,2,nan],[3,4,5],[6,nan,8],[9,10,11]]`,
/// `KNNImputer(n_neighbors=2)`: row0=[1,2,6.5], row2=[6,7,8].
#[test]
fn guard_reaudit_f32_small_fixture() {
    let x: Array2<f32> = array![
        [1.0f32, 2.0, f32::NAN],
        [3.0, 4.0, 5.0],
        [6.0, f32::NAN, 8.0],
        [9.0, 10.0, 11.0]
    ];
    let out = KNNImputer::<f32>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    // Live-oracle f64 targets, f32 tolerance.
    assert_abs_diff_eq!(out[[0, 2]], 6.5_f32, epsilon = 1e-4);
    assert_abs_diff_eq!(out[[2, 1]], 7.0_f32, epsilon = 1e-4);
}

// ===========================================================================
// NEW DIVERGENCE (DIV-4, REQ-8): distance EXACT-MATCH (distance 0) weighting.
// sklearn `_get_weights` (sklearn/neighbors/_base.py:116-121) sets the
// exact-match donor's weight to 1 and ALL other donors in that receiver's row
// to 0 (`inf_row` -> `dist[inf_row] = inf_mask[inf_row]`), so the imputed value
// EQUALS the exact-match donor's value. ferrolearn's Distance branch uses a 1e12
// "very high weight" hack (knn_imputer.rs:307-312) instead of true 1/0
// weighting, so the other donors still leak in.
// ===========================================================================

/// Divergence (DIV-4): receiver `[0, nan]` exactly matches donor `[0, 0]` on the
/// only shared feature (col 0), with two far donors `[3, 1e6]`, `[6, 2e6]`,
/// `n_neighbors=3`, `weights='distance'`. sklearn gives the exact-match donor
/// weight 1 and the other two weight 0, so the imputed col-1 value is EXACTLY
/// `0.0`. ferrolearn's 1e12 hack blends the far donors in, yielding ~`4.714e-07`.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=3, weights='distance').fit_transform(
///   [[0,nan],[0,0],[3,1e6],[6,2e6]])[0,1]` -> `0.0` (exact).
/// `_get_weights(np.array([[0.,1.,4.]]), 'distance')` -> `[[1., 0., 0.]]`
/// (sklearn/neighbors/_base.py:116-121).
/// Tracking: #1308
#[test]
fn divergence_div4_exact_match_weights_leak() {
    let sklearn_imputed_0_1 = 0.0_f64; // live oracle, exact.
    let x = array![[0.0, f64::NAN], [0.0, 0.0], [3.0, 1.0e6], [6.0, 2.0e6]];
    let out = KNNImputer::<f64>::new(3, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    // 1e-9 tolerance is far tighter than ferrolearn's ~4.7e-7 error -> FAILS.
    assert_abs_diff_eq!(out[[0, 1]], sklearn_imputed_0_1, epsilon = 1e-9);
}

// ===========================================================================
// NEW DIVERGENCE (DIV-5): UNIFORM-weight donor SELECTION drops NaN-distance
// donors that sklearn still averages.
//
// sklearn `_calc_impute` (sklearn/impute/_knn.py:184-204) selects donors over
// the FULL `potential_donors_idx` set via `argpartition`, regardless of whether
// the receiver shares any present feature with a given donor (a NaN pairwise
// distance). Under UNIFORM weights `_get_weights` returns `None`
// (sklearn/neighbors/_base.py:100-101), so `np.ma.average(donors, axis=1,
// weights=None)` averages ALL selected donors whose VALUE is present — including
// donors at NaN distance. ferrolearn's transform only pushes a train row into
// `dists` when `n_valid > 0` (knn_imputer.rs:243), so it EXCLUDES NaN-distance
// donors entirely and averages a strictly smaller donor set.
// ===========================================================================

/// Divergence (DIV-5): on `X=[[nan,1],[nan,2],[5,3],[7,nan]]`, `n_neighbors=2`,
/// UNIFORM weights, receiver row 0 needs col 0. The two col-0 donors are row 2
/// (val 5, shares col 1 with receiver -> finite distance) and row 3 (val 7,
/// shares NO present feature -> NaN distance). sklearn averages BOTH donors
/// (uniform): `mean(5, 7) = 6.0`. ferrolearn drops the NaN-distance donor (row 3)
/// and imputes the single reachable donor's value `5.0`.
///
/// Live oracle:
/// `KNNImputer(n_neighbors=2).fit_transform([[nan,1],[nan,2],[5,3],[7,nan]])`
/// -> `[[6,1],[6,2],[5,3],[7,2]]`; cells `[0,0] == 6.0` and `[1,0] == 6.0`.
/// Tracking: #1309
#[test]
fn divergence_div5_uniform_drops_nan_distance_donor() {
    let sklearn_imputed_0_0 = 6.0_f64; // live oracle: mean(5,7).
    let x = array![
        [f64::NAN, 1.0],
        [f64::NAN, 2.0],
        [5.0, 3.0],
        [7.0, f64::NAN]
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    // ferrolearn imputes 5.0 (drops the NaN-distance donor row 3) -> FAILS.
    assert_abs_diff_eq!(out[[0, 0]], sklearn_imputed_0_0, epsilon = 1e-9);
    assert_abs_diff_eq!(out[[1, 0]], sklearn_imputed_0_0, epsilon = 1e-9);
}

// ===========================================================================
// FINAL RE-AUDIT (post #1305/#1306/#1307/#1308/#1309). FRESH live-oracle
// fixtures (sklearn 1.5.2, run from /tmp). These are NOT reused from prior
// audits. All expected matrices are the sklearn `KNNImputer(...).fit_transform`
// output (R-CHAR-3: oracle-derived, never copied from ferrolearn). They
// exercise scaling + finite/inf donor selection + column-mean fallback +
// uniform/distance weighting jointly across k in {1,2,3,5}.
// ===========================================================================

const NAN: f64 = f64::NAN;

/// Assert ferrolearn's full imputed matrix matches a (possibly ragged-width)
/// sklearn oracle matrix within `eps`.
fn assert_full_eq(out: &Array2<f64>, expected: &[&[f64]], eps: f64, tag: &str) {
    assert_eq!(out.nrows(), expected.len(), "{tag}: row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(out.ncols(), row.len(), "{tag}: col count");
        for (j, &e) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - e).abs() <= eps,
                "{tag}: cell [{i},{j}] = {got}, sklearn = {e} (diff {})",
                (got - e).abs()
            );
        }
    }
}

fn fixture_a() -> Array2<f64> {
    array![
        [5.1, NAN, 3.2, 1.1, NAN],
        [4.9, 3.0, NAN, 1.4, 0.2],
        [NAN, 3.2, 1.3, NAN, 0.2],
        [4.6, 3.1, 1.5, 0.2, NAN],
        [5.0, NAN, 1.4, 0.3, 0.4],
        [NAN, 3.9, NAN, 0.4, 0.5],
        [5.4, 3.7, 1.5, NAN, 0.2],
        [4.6, 3.4, 1.4, 0.3, 0.3]
    ]
}

fn fixture_b() -> Array2<f64> {
    array![
        [1.0, 2.0, NAN, 4.0, 5.0, 6.0],
        [7.0, NAN, 9.0, 10.0, NAN, 12.0],
        [13.0, 14.0, 15.0, NAN, 17.0, 18.0],
        [NAN, 20.0, 21.0, 22.0, 23.0, NAN],
        [25.0, 26.0, NAN, 28.0, 29.0, 30.0],
        [31.0, NAN, 33.0, 34.0, 35.0, 36.0],
        [NAN, 38.0, 39.0, 40.0, NAN, 42.0],
        [43.0, 44.0, 45.0, NAN, 47.0, 48.0],
        [49.0, NAN, 51.0, 52.0, 53.0, 54.0],
        [55.0, 56.0, 57.0, 58.0, NAN, 60.0]
    ]
}

#[test]
fn reaudit_a_k2_uniform() {
    let exp: &[&[f64]] = &[
        &[5.1, 3.45, 3.2, 1.1, 0.35],
        &[4.9, 3.0, 2.25, 1.4, 0.2],
        &[4.75, 3.2, 1.3, 0.85, 0.2],
        &[4.6, 3.1, 1.5, 0.2, 0.25],
        &[5.0, 3.55, 1.4, 0.3, 0.4],
        &[5.2, 3.9, 1.45, 0.4, 0.5],
        &[5.4, 3.7, 1.5, 0.35, 0.2],
        &[4.6, 3.4, 1.4, 0.3, 0.3],
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&fixture_a())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "A k2 uniform");
}

#[test]
fn reaudit_a_k2_distance() {
    let exp: &[&[f64]] = &[
        &[5.1, 3.2402802698310014, 3.2, 1.1, 0.2800934232770006],
        &[4.9, 3.0, 1.9778994274181139, 1.4, 0.2],
        &[4.750000000000003, 3.2, 1.3, 0.8500000000000081, 0.2],
        &[4.6, 3.1, 1.5, 0.2, 0.248808848170152],
        &[5.0, 3.6288018792940555, 1.4, 0.3, 0.4],
        &[5.112691618676093, 3.9, 1.4281729046690232, 0.4, 0.5],
        &[5.4, 3.7, 1.5, 0.3509262436767959, 0.2],
        &[4.6, 3.4, 1.4, 0.3, 0.3],
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Distance)
        .fit_transform(&fixture_a())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "A k2 distance");
}

#[test]
fn reaudit_a_k5_uniform() {
    let exp: &[&[f64]] = &[
        &[5.1, 3.4200000000000004, 3.2, 1.1, 0.32],
        &[4.9, 3.0, 1.7600000000000002, 1.4, 0.2],
        &[4.9, 3.2, 1.3, 0.52, 0.2],
        &[4.6, 3.1, 1.5, 0.2, 0.32],
        &[5.0, 3.46, 1.4, 0.3, 0.4],
        &[4.94, 3.9, 1.42, 0.4, 0.5],
        &[5.4, 3.7, 1.5, 0.52, 0.2],
        &[4.6, 3.4, 1.4, 0.3, 0.3],
    ];
    let out = KNNImputer::<f64>::new(5, KNNWeights::Uniform)
        .fit_transform(&fixture_a())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "A k5 uniform");
}

#[test]
fn reaudit_a_k5_distance() {
    let exp: &[&[f64]] = &[
        &[5.1, 3.2897477310445837, 3.2, 1.1, 0.2872639102261604],
        &[4.9, 3.0, 1.804607260313167, 1.4, 0.2],
        &[4.840746225582927, 3.2, 1.3, 0.5559118719411182, 0.2],
        &[4.6, 3.1, 1.5, 0.2, 0.2972634114762503],
        &[5.0, 3.525501527474006, 1.4, 0.3, 0.4],
        &[4.988178894741888, 3.9, 1.418301009008918, 0.4, 0.5],
        &[5.4, 3.7, 1.5, 0.47691808374272504, 0.2],
        &[4.6, 3.4, 1.4, 0.3, 0.3],
    ];
    let out = KNNImputer::<f64>::new(5, KNNWeights::Distance)
        .fit_transform(&fixture_a())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "A k5 distance");
}

#[test]
fn reaudit_b_k2_uniform() {
    let exp: &[&[f64]] = &[
        &[1.0, 2.0, 12.0, 4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        &[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        &[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        &[37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
        &[43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
        &[49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
        &[55.0, 56.0, 57.0, 58.0, 50.0, 60.0],
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&fixture_b())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "B k2 uniform");
}

#[test]
fn reaudit_b_k2_distance() {
    let exp: &[&[f64]] = &[
        &[1.0, 2.0, 10.999999999999998, 4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        &[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        &[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        &[37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
        &[43.0, 44.0, 45.0, 45.99999999999999, 47.0, 48.0],
        &[49.0, 50.00000000000001, 51.0, 52.0, 53.0, 54.0],
        &[55.0, 56.0, 57.0, 58.0, 51.0, 60.0],
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Distance)
        .fit_transform(&fixture_b())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "B k2 distance");
}

#[test]
fn reaudit_b_k5_uniform() {
    let exp: &[&[f64]] = &[
        &[1.0, 2.0, 23.4, 4.0, 5.0, 6.0],
        &[7.0, 20.0, 9.0, 10.0, 21.8, 12.0],
        &[13.0, 14.0, 15.0, 19.6, 17.0, 18.0],
        &[15.4, 20.0, 21.0, 22.0, 23.0, 20.4],
        &[25.0, 26.0, 23.4, 28.0, 29.0, 30.0],
        &[31.0, 28.4, 33.0, 34.0, 35.0, 36.0],
        &[40.6, 38.0, 39.0, 40.0, 37.4, 42.0],
        &[43.0, 44.0, 45.0, 42.4, 47.0, 48.0],
        &[49.0, 36.8, 51.0, 52.0, 53.0, 54.0],
        &[55.0, 56.0, 57.0, 58.0, 37.4, 60.0],
    ];
    let out = KNNImputer::<f64>::new(5, KNNWeights::Uniform)
        .fit_transform(&fixture_b())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "B k5 uniform");
}

#[test]
fn reaudit_b_k5_distance() {
    let exp: &[&[f64]] = &[
        &[1.0, 2.0, 16.636363636363637, 4.0, 5.0, 6.0],
        &[7.0, 13.934065934065933, 9.0, 10.0, 16.83783783783784, 12.0],
        &[13.0, 14.0, 15.0, 17.8, 17.0, 18.0],
        &[17.2, 20.0, 21.0, 22.0, 23.0, 22.2],
        &[25.0, 26.0, 25.2, 28.0, 29.0, 30.0],
        &[31.0, 30.2, 33.0, 34.0, 35.0, 36.0],
        &[38.79999999999999, 38.0, 39.0, 40.0, 39.2, 42.0],
        &[43.0, 44.0, 45.0, 44.199999999999996, 47.0, 48.0],
        &[49.0, 43.89830508474577, 51.0, 52.0, 53.0, 54.0],
        &[55.0, 56.0, 57.0, 58.0, 44.826771653543304, 60.0],
    ];
    let out = KNNImputer::<f64>::new(5, KNNWeights::Distance)
        .fit_transform(&fixture_b())
        .unwrap();
    assert_full_eq(&out, exp, 1e-9, "B k5 distance");
}

// ---------------------------------------------------------------------------
// Edge interactions (fresh live-oracle values).
// ---------------------------------------------------------------------------

/// Edge: receiver's n_neighbors quota filled by a MIX of finite- and inf-distance
/// donors. Row 0 has only col3 present; needs col2. Col2-donors: rows 1,2 (col3
/// missing -> inf distance) and row 3 (col3 present -> finite, exact match dist 0).
/// k=3 collects all 3.
/// Uniform: mean(10,20,30)=20.0. Distance: row3 is dist0 exact -> weight 1, others
/// 0 -> 30.0. (live oracle).
#[test]
fn reaudit_edge_mix_finite_inf_donors() {
    let x = array![
        [NAN, NAN, NAN, 1.0],
        [5.0, 5.0, 10.0, NAN],
        [6.0, 6.0, 20.0, NAN],
        [7.0, 7.0, 30.0, 1.0]
    ];
    let out_u = KNNImputer::<f64>::new(3, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    assert!(
        (out_u[[0, 2]] - 20.0).abs() <= 1e-9,
        "uniform mix: got {}",
        out_u[[0, 2]]
    );
    let out_d = KNNImputer::<f64>::new(3, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    assert!(
        (out_d[[0, 2]] - 30.0).abs() <= 1e-9,
        "distance mix exact-match: got {}",
        out_d[[0, 2]]
    );
}

/// Edge: a column where SOME receivers hit the all-inf column-mean fallback and
/// others get normal KNN. Row 0 needs col0 (donors rows2,3 share col1 -> finite KNN);
/// row 1 needs col0 (NO present feature -> all-inf -> masked col0 mean) AND col1.
/// Live oracle full matrix.
#[test]
fn reaudit_edge_some_colmean_some_knn() {
    let x = array![[NAN, 1.0], [NAN, NAN], [5.0, 3.0], [7.0, 9.0]];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    let exp: &[&[f64]] = &[
        &[6.0, 1.0],
        &[6.0, 4.333333333333333],
        &[5.0, 3.0],
        &[7.0, 9.0],
    ];
    assert_full_eq(&out, exp, 1e-9, "some-colmean-some-knn");
}

/// Edge: n_neighbors > donors-for-column (clamp) combined with inf-distance donors.
/// Row 0 needs col0,col1 (present col2). Row 1 is all-NaN in cols 0,1,2. k=5.
/// Live oracle full matrix.
#[test]
fn reaudit_edge_clamp_plus_inf_donors() {
    let x = array![
        [NAN, NAN, 2.0],
        [NAN, NAN, NAN],
        [5.0, 50.0, NAN],
        [7.0, 70.0, 9.0]
    ];
    let out = KNNImputer::<f64>::new(5, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    let exp: &[&[f64]] = &[
        &[6.0, 60.0, 2.0],
        &[6.0, 60.0, 5.5],
        &[5.0, 50.0, 5.5],
        &[7.0, 70.0, 9.0],
    ];
    assert_full_eq(&out, exp, 1e-9, "clamp+inf");
}

/// Edge: distance weights where one donor is exact-match (dist0) AND others finite
/// AND an inf donor exists -> only the exact-match value is used. Row0 col0=2;
/// donor row1 exact (col0=2), row2 finite (col0=5), row3 inf (col0 missing).
/// k=3, distance -> col1 imputes 100.0 (exact donor's value). (live oracle).
#[test]
fn reaudit_edge_exact_match_with_finite_and_inf() {
    let x = array![[2.0, NAN], [2.0, 100.0], [5.0, 200.0], [NAN, 300.0]];
    let out = KNNImputer::<f64>::new(3, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    assert!(
        (out[[0, 1]] - 100.0).abs() <= 1e-9,
        "exact+finite+inf: got {}",
        out[[0, 1]]
    );
}

/// Edge: distance weights, all selected donors inf for a receiver that has SOME
/// finite distance elsewhere (the all_inf guard). Row0 present col1 only; needs
/// col0 (donors rows2,3 inf) and col2 (donor row1 finite). k=2, distance.
/// Live oracle full matrix.
#[test]
fn reaudit_edge_all_inf_guard_distance() {
    let x = array![
        [NAN, 5.0, NAN],
        [NAN, 6.0, 100.0],
        [7.0, NAN, 200.0],
        [8.0, NAN, 300.0]
    ];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    let exp: &[&[f64]] = &[
        &[7.5, 5.0, 100.0],
        &[7.333333333333335, 6.0, 100.0],
        &[7.0, 6.0, 200.0],
        &[8.0, 6.0, 300.0],
    ];
    assert_full_eq(&out, exp, 1e-9, "all-inf guard distance");
}

/// Edge: f32 fixture (tolerance ~1e-4) full-matrix parity against the f64 oracle.
#[test]
fn reaudit_edge_f32_full_matrix() {
    let x: Array2<f32> = array![
        [1.0f32, 2.0, f32::NAN, 4.0, 5.0],
        [6.0, 7.0, 8.0, f32::NAN, 10.0],
        [11.0, f32::NAN, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, f32::NAN],
        [21.0, 22.0, f32::NAN, 24.0, 25.0]
    ];
    let out = KNNImputer::<f32>::new(2, KNNWeights::Uniform)
        .fit_transform(&x)
        .unwrap();
    // f64 live oracle for the same matrix (sklearn 1.5.2).
    let exp: &[&[f32]] = &[
        &[1.0, 2.0, 10.5, 4.0, 5.0],
        &[6.0, 7.0, 8.0, 9.0, 10.0],
        &[11.0, 12.0, 13.0, 14.0, 15.0],
        &[16.0, 17.0, 18.0, 19.0, 20.0],
        &[21.0, 22.0, 15.5, 24.0, 25.0],
    ];
    for (i, row) in exp.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            assert!(
                (out[[i, j]] - e).abs() <= 1e-4,
                "f32 [{i},{j}]={}, sklearn {e}",
                out[[i, j]]
            );
        }
    }
}

/// Sanity: fully-observed (no-NaN) matrix returns unchanged.
#[test]
fn reaudit_no_missing_returns_unchanged() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let out = KNNImputer::<f64>::new(2, KNNWeights::Distance)
        .fit_transform(&x)
        .unwrap();
    let exp: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
    assert_full_eq(&out, exp, 0.0, "no-missing unchanged");
}

// ===========================================================================
// DOCUMENTED EDGE (DIV-6, #1310) — NOT pinned (no committed failing test, per
// R-DEFER-3): distance-TIED donor selection.
//
// sklearn `_calc_impute` (sklearn/impute/_knn.py:184) selects donors via
// `np.argpartition(dist_pot_donors, n_neighbors-1)`, whose order on EXACT
// distance ties is an UNSPECIFIED partition order (a numpy introselect
// implementation artifact, not documented behavior). ferrolearn
// `KNNImputer::transform` uses a STABLE `sort_by(partial_cmp)`, picking the
// lowest train index on a tie. The two only disagree when two donors are
// equidistant (within float rounding) — where sklearn's own choice is
// arbitrary. The divergence is further driven by ULP-level differences between
// ferrolearn's `sqrt(sum_sq*n/v)` and numpy's `euclidean_distances`
// (||x||^2+||y||^2-2xy) decomposition, which only flips order at near-ties.
// Matching numpy's argpartition pivot order + its exact float expression is not
// a meaningful parity target (cf. the RFE equal-importance tie-break and the
// DBSCAN eps-boundary ULP carve-outs). The imputed VALUES match except in these
// tie cells. Tracked as a NOT-STARTED edge: #1310.
// ===========================================================================
