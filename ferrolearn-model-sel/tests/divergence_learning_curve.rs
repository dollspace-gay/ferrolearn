//! Adversarial divergence audit of `learning_curve` against scikit-learn 1.5.2
//! `sklearn.model_selection.learning_curve`
//! (`sklearn/model_selection/_validation.py:1724-1989`), the size-translation
//! helper `_translate_train_sizes` (`:1992-2057`), and the `_fit_and_score`
//! `error_score` path (`:890-915`).
//!
//! GREEN guards pin SHIPPED behavior (REQ-1 per-(size,fold) fit+score mechanic
//! and orientation/orientation-equivalence; REQ-2 train scores always returned
//! with `(n_ticks, n_cv_folds)` shape) so a future regression would FAIL them.
//!
//! `#[ignore]`'d tests pin genuine, single-file-fixable behavioral divergences
//! in the size-translation layer against the LIVE sklearn 1.5.2 oracle:
//!   - #1764 fraction->absolute FLOOR vs CEIL
//!   - #1765 float entry > 1.0 -> ValueError vs treat-as-absolute
//!   - #1766 absolute > n_max -> ValueError vs clamp
//!   - #1767 dedup + sort via np.unique vs preserve-order-and-duplicates
//!   - #1768 error_score=np.nan continue-the-curve vs `?`-abort
//!
//! All expected values are oracle-derived (live sklearn 1.5.2 / a pure-numpy
//! reproduction of sklearn's `.reshape(-1, n_unique_ticks).T` algebra), never
//! copied from the ferrolearn side (R-CHAR-3). The translation oracles were
//! produced by calling `_translate_train_sizes` directly with `n_max=20`, which
//! equals `folds[0].0.len()` for `KFold(3)` on 30 samples (verified: each fold
//! has train length 20) so the Rust setup's reference matches the oracle's
//! `n_max`.

use ferrolearn_core::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineEstimator, Pipeline, PipelineEstimator};
use ferrolearn_model_sel::{KFold, learning_curve};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Estimator that records the number of training samples it was fit on and
/// predicts that count as a constant. Combined with a fold-encoding `y` and a
/// fold-encoding scorer this makes each `(size, fold)` cell deterministically
/// recoverable — the transpose/swap detector for REQ-1.
struct CountEstimator;

struct FittedCount {
    n_train: f64,
}

impl PipelineEstimator<f64> for CountEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedCount {
            n_train: x.nrows() as f64,
        }))
    }
}

impl FittedPipelineEstimator<f64> for FittedCount {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.n_train))
    }
}

/// Plain mean estimator (for the non-orientation guards / pinned tests).
struct MeanEstimator;

struct FittedMean {
    mean: f64,
}

impl PipelineEstimator<f64> for MeanEstimator {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        Ok(Box::new(FittedMean {
            mean: y.mean().unwrap_or(0.0),
        }))
    }
}

impl FittedPipelineEstimator<f64> for FittedMean {
    fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.mean))
    }
}

/// Estimator that fails to fit when the training-subset size equals
/// `fail_at_size` — the ferrolearn analog of sklearn's estimator raising in
/// `fit` for one `(size, fold)` cell (drives the `error_score=np.nan` pin).
struct FailAtSizeEstimator {
    fail_at_size: usize,
}

impl PipelineEstimator<f64> for FailAtSizeEstimator {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineEstimator<f64>>, FerroError> {
        if x.nrows() == self.fail_at_size {
            return Err(FerroError::InvalidParameter {
                name: "boom".into(),
                reason: "estimator failed to fit for this training-subset size".into(),
            });
        }
        Ok(Box::new(FittedMean {
            mean: y.mean().unwrap_or(0.0),
        }))
    }
}

fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let diff = y_true - y_pred;
    Ok(-diff.mapv(|v| v * v).mean().unwrap_or(0.0))
}

/// Scoring that encodes BOTH the fold (via `mean(y_true)`) AND the training
/// size (via the constant prediction = `n_train`) into one recoverable value:
/// `score = mean(y_true) * 1000.0 + y_pred[0]`. With a `y` whose value equals
/// its fold index, cell `(size i, fold j)` deterministically equals
/// `j*1000 + n_train_i`. This is the swap/transpose detector for REQ-1.
fn fold_and_size_probe(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
    let fold_part = y_true.mean().unwrap_or(0.0) * 1000.0;
    let size_part = y_pred[0];
    Ok(fold_part + size_part)
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-1: per-(size,fold) mechanic + end-to-end orientation
// ---------------------------------------------------------------------------

/// REQ-1 (KEY). Verifies the `(size OUTER, fold INNER)` row-major fill produces
/// a `(n_sizes, n_folds)` matrix element-for-element IDENTICAL to sklearn's
/// `(fold OUTER, size INNER)` flat list `.reshape(-1, n_unique_ticks).T`
/// (`sklearn/model_selection/_validation.py:1949-1976`), AND that the train
/// subset is the FIRST `n_train` of the fold's train indices (sklearn
/// `train[:n_train_samples]` `:1952` vs ferrolearn `&train_idx[..effective_size]`).
///
/// Adversarial construction (a true transpose/swap detector — distinct value
/// per `(size, fold)` cell, unlike the same-value-per-cell SHIPPED unit tests):
/// ferrolearn's `KFold(3, shuffle=False)` splits 30 samples into contiguous test
/// folds `0..10, 10..20, 20..30`. With `y[row] = floor(row/10)`, every TEST
/// fold's `y` is constant and equal to its fold index `j`, so the scorer's
/// `mean(y_true)*1000` term recovers `j`. The `CountEstimator` predicts the
/// training-subset size `n_train` (which here equals the requested absolute tick
/// `size`, since `size <= 20 == fold train length` so no shrink occurs), so the
/// scorer's `y_pred[0]` term recovers `size`. Cell `(size i, fold j)` therefore
/// equals `j*1000 + size_i`.
///
/// Oracle (pure-numpy reproduction of sklearn's reshape algebra, isolating the
/// index math from estimator noise):
/// ```text
/// sizes = [5, 10, 15, 20]; n_folds = 3
/// flat = [f*1000 + s for f in range(n_folds) for s in sizes]   # fold-major
/// M = flat.reshape(-1, len(sizes)).T                           # (n_sizes, n_folds)
/// # M[i][j] == folds[j]*1000 + sizes[i]
/// ```
#[test]
fn req1_orientation_and_first_n_train_end_to_end() {
    // y[row] = floor(row/10) => fold j's contiguous test set has y == j.
    let y: Array1<f64> = (0..30).map(|r| f64::from(r / 10)).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    let pipeline = Pipeline::new().estimator_step("count", Box::new(CountEstimator));

    // Absolute ticks, all <= 20 (the per-fold train length) so no shrink.
    let sizes = [5.0_f64, 10.0, 15.0, 20.0];
    let result = learning_curve(&pipeline, &x, &y, &kf, &sizes, fold_and_size_probe).unwrap();

    assert_eq!(
        result.test_scores.shape(),
        &[4, 3],
        "shape must be (n_sizes, n_folds)"
    );
    assert_eq!(result.train_sizes, vec![5, 10, 15, 20], "absolute ticks");

    // Oracle: cell (size i, fold j) == fold_j*1000 + size_i. The `size_i` part
    // is the prediction = n_train, which also proves the train subset is the
    // FIRST `size_i` indices (a count, not a transpose, surfaces here).
    for (i, &size) in sizes.iter().enumerate() {
        for j in 0..3usize {
            let expected = (j as f64) * 1000.0 + size;
            let got = result.test_scores[[i, j]];
            assert!(
                (got - expected).abs() < 1e-9,
                "cell (size i={i} -> {size} samples, fold j={j}): expected {expected} \
                 (= fold*1000 + n_train); got {got}; a transpose/size-fold swap or a \
                 wrong-length train subset surfaces here",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN GUARD — REQ-2: train scores always returned, shape (n_ticks, n_folds)
// ---------------------------------------------------------------------------

/// REQ-2 (`return_train_score=True` hardcoded,
/// `sklearn/model_selection/_validation.py:1967`; returned shape
/// `(n_unique_ticks, n_cv_folds)`, `:1861-1864`/`:1975-1976`).
///
/// Live oracle (sklearn 1.5.2): `learning_curve(DecisionTreeRegressor(), X, y,
/// train_sizes=[5,10,20], cv=KFold(3))` returns `tr.shape == te.shape == (3, 3)`.
#[test]
fn req2_train_scores_returned_with_shape() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    let result = learning_curve(&pipeline, &x, &y, &kf, &[5.0, 10.0, 20.0], neg_mse).unwrap();

    // sklearn returns train_scores unconditionally; ferrolearn must too.
    assert_eq!(
        result.train_scores.shape(),
        &[3, 3],
        "train_scores (n_ticks, n_folds)"
    );
    assert_eq!(
        result.test_scores.shape(),
        &[3, 3],
        "test_scores (n_ticks, n_folds)"
    );
    assert_eq!(
        result.train_scores.len(),
        9,
        "every (size,fold) train cell populated"
    );
    for &s in &result.train_scores {
        assert!(s.is_finite(), "train score must be finite, got {s}");
    }
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1764: fraction->absolute FLOOR vs CEIL
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's size translation
/// (`ferrolearn-model-sel/src/learning_curve.rs:132`
/// `((s * reference_train_len as f64).ceil() as usize)`) diverges from
/// `sklearn/model_selection/_validation.py:2028-2030`
/// (`(train_sizes_abs * n_max_training_samples).astype(int)` — TRUNCATION toward
/// zero, i.e. FLOOR for positives).
///
/// Input `train_sizes=[0.33]`, reference train length 20 (KFold(3) on 30).
/// LIVE sklearn 1.5.2 oracle: `_translate_train_sizes([0.33], 20) -> [6]`
/// (`int(6.6) == 6`); ferrolearn computes `ceil(6.6) == 7`.
///
/// Tracking: #1764.
// #1764
#[test]
fn divergence_1764_fraction_floor_vs_ceil() {
    // Oracle: _translate_train_sizes(np.asarray([0.33]), 20).tolist() == [6]
    const SKLEARN_FLOOR_TICK: usize = 6;

    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3); // first fold train length == 20 == oracle n_max
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    let result = learning_curve(&pipeline, &x, &y, &kf, &[0.33], neg_mse).unwrap();

    assert_eq!(
        result.train_sizes[0], SKLEARN_FLOOR_TICK,
        "sklearn floors 0.33*20=6.6 -> 6; ferrolearn ceils -> 7",
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1765: float entry > 1.0 -> ValueError vs treat-as-absolute
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's size translation
/// (`ferrolearn-model-sel/src/learning_curve.rs:130` per-element `if s <= 1.0`)
/// diverges from `sklearn/model_selection/_validation.py:2020-2027`: sklearn
/// decides fraction-vs-absolute by the WHOLE array's dtype (`:2020`), so an
/// all-float array is fraction mode and EVERY entry must be `<= 1.0`, else
/// `ValueError` (`:2021-2027`). ferrolearn treats a float `2.0` per-element as
/// an absolute count of 2.
///
/// Input `train_sizes=[0.5, 2.0]`, n_max 20.
/// LIVE sklearn 1.5.2 oracle: `_translate_train_sizes([0.5, 2.0], 20)` raises
/// `ValueError`. ferrolearn returns `Ok` with a size-2 tick for the `2.0` entry.
///
/// Tracking: #1765.
// #1765
#[test]
fn divergence_1765_float_gt_one_should_error() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    // sklearn raises ValueError; ferrolearn must error too.
    let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 2.0], neg_mse);

    assert!(
        result.is_err(),
        "sklearn raises ValueError for a float train_sizes entry > 1.0; \
         ferrolearn silently treats 2.0 as an absolute count",
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1766: absolute > n_max -> ValueError vs clamp
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's size translation
/// (`ferrolearn-model-sel/src/learning_curve.rs:137`
/// `(s as usize).max(1).min(reference_train_len)`) diverges from
/// `sklearn/model_selection/_validation.py:2033-2046`: sklearn's integer-dtype
/// path raises `ValueError` if `max > n_max_training_samples` (NO clamp).
/// ferrolearn clamps the overshoot to `reference_train_len`.
///
/// Input `train_sizes=[5.0, 100.0]` (per-element >1.0 -> treated as absolute by
/// ferrolearn), n_max 20.
/// LIVE sklearn 1.5.2 oracle: `_translate_train_sizes([5, 100], 20)` raises
/// `ValueError`. ferrolearn returns `Ok`, clamping `100 -> 20`.
///
/// Tracking: #1766.
// #1766
#[test]
fn divergence_1766_absolute_overshoot_should_error() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    // sklearn raises ValueError; ferrolearn must error too.
    let result = learning_curve(&pipeline, &x, &y, &kf, &[5.0, 100.0], neg_mse);

    assert!(
        result.is_err(),
        "sklearn raises ValueError when an absolute train size exceeds n_max; \
         ferrolearn silently clamps 100 -> 20",
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1767: dedup + sort via np.unique
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn builds `abs_sizes` preserving input order and keeping
/// duplicates (`ferrolearn-model-sel/src/learning_curve.rs:127-140`), whereas
/// sklearn applies `np.unique(train_sizes_abs)`
/// (`sklearn/model_selection/_validation.py:2048`), which SORTS ascending AND
/// removes duplicates (RuntimeWarning if removed, `:2049-2055`).
///
/// Facet (a) ORDER — input `train_sizes=[1.0, 0.5]`, n_max 20.
/// LIVE sklearn oracle: `_translate_train_sizes([1.0, 0.5], 20) -> [10, 20]`
/// (sorted ascending). ferrolearn yields `[20, 10]` (input order).
///
/// Tracking: #1767.
// #1767
#[test]
fn divergence_1767_sort_ascending() {
    // Oracle: _translate_train_sizes(np.asarray([1.0, 0.5]), 20).tolist() == [10, 20]
    const SKLEARN_SORTED: [usize; 2] = [10, 20];

    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3); // first fold train length 20 == oracle n_max
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    let result = learning_curve(&pipeline, &x, &y, &kf, &[1.0, 0.5], neg_mse).unwrap();

    assert_eq!(
        result.train_sizes.as_slice(),
        SKLEARN_SORTED,
        "sklearn np.unique sorts [1.0,0.5]*20 -> [10,20]; ferrolearn keeps input order [20,10]",
    );
}

/// Divergence (dedup facet of #1767): sklearn `np.unique` removes duplicate
/// ticks, reducing the number of matrix rows; ferrolearn keeps duplicates,
/// producing an extra row.
///
/// Input `train_sizes=[0.5, 0.5]`, n_max 20.
/// LIVE sklearn oracle: `_translate_train_sizes([0.5, 0.5], 20) -> [10]`
/// (1 tick). ferrolearn yields `[10, 10]` (2 ticks -> 2 matrix rows).
///
/// Tracking: #1767.
// #1767
#[test]
fn divergence_1767_dedup_row_count() {
    // Oracle: _translate_train_sizes(np.asarray([0.5, 0.5]), 20).tolist() == [10]
    const SKLEARN_N_UNIQUE_TICKS: usize = 1;

    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    let result = learning_curve(&pipeline, &x, &y, &kf, &[0.5, 0.5], neg_mse).unwrap();

    assert_eq!(
        result.train_sizes.len(),
        SKLEARN_N_UNIQUE_TICKS,
        "sklearn dedups [0.5,0.5] -> 1 tick; ferrolearn keeps 2 ticks",
    );
    assert_eq!(
        result.train_scores.shape()[0],
        SKLEARN_N_UNIQUE_TICKS,
        "matrix row count must equal n_unique_ticks (1), not the raw 2",
    );
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1768: error_score=np.nan continue-the-curve
// ---------------------------------------------------------------------------

/// Divergence: ferrolearn's per-cell fit/score
/// (`ferrolearn-model-sel/src/learning_curve.rs:179` `pipeline.fit(...)?` /
/// `:182`/`:186` `scoring(...)?`) `?`-propagates a per-`(size, fold)` failure
/// and ABORTS the whole call, whereas sklearn's `_fit_and_score` with the
/// default `error_score=np.nan` (`sklearn/model_selection/_validation.py:1968`,
/// `_fit_and_score` `:890-905`) fills the failing cell with `np.nan`, warns, and
/// CONTINUES the curve. Same contract as the just-SHIPPED `validation_curve`
/// REQ-7 (#1758/#1762).
///
/// LIVE sklearn 1.5.2 oracle (AC-E):
/// ```text
/// class F(BaseEstimator, RegressorMixin):
///     def fit(self, X, y):
///         if X.shape[0] == 5: raise ValueError('boom')
///         self.m_ = y.mean(); return self
///     def predict(self, X): return np.full(X.shape[0], self.m_)
/// ts, tr, te = learning_curve(F(), X, y, train_sizes=[5,10,20], cv=KFold(3),
///                             scoring='neg_mean_squared_error')
/// # te.shape == (3, 3); te[0] is all-NaN; np.isnan(te).any() == True
/// ```
///
/// Here the FIRST tick (size 5) maps to a failing fit. sklearn returns a full
/// `(3, 3)` matrix whose row 0 is all-NaN; ferrolearn returns `Err`. This test
/// asserts the sklearn behavior and therefore FAILS against the current
/// implementation.
///
/// Tracking: #1768.
// #1768
#[test]
fn divergence_1768_error_score_nan_continue() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3);
    // Fails to fit whenever the training subset has exactly 5 samples (tick 0).
    let pipeline =
        Pipeline::new().estimator_step("fail", Box::new(FailAtSizeEstimator { fail_at_size: 5 }));

    let result = learning_curve(&pipeline, &x, &y, &kf, &[5.0, 10.0, 20.0], neg_mse)
        .expect("sklearn returns a partial nan-bearing curve, not an error");

    assert_eq!(
        result.test_scores.shape(),
        &[3, 3],
        "shape (n_sizes, n_folds)"
    );

    // Oracle: the failing size row (index 0) is all-NaN; other rows are finite.
    for j in 0..3usize {
        assert!(
            result.test_scores[[0, j]].is_nan(),
            "failing-size row 0, fold {j}: sklearn fills with np.nan, got {}",
            result.test_scores[[0, j]],
        );
        assert!(
            result.test_scores[[1, j]].is_finite(),
            "non-failing size row 1 must remain finite",
        );
        assert!(
            result.test_scores[[2, j]].is_finite(),
            "non-failing size row 2 must remain finite",
        );
    }
}

// ---------------------------------------------------------------------------
// PINNED DIVERGENCE — #1769: all-integer-valued FLOAT > 1.0 -> ValueError
// vs treated-as-absolute (NEW, introduced by the #1764-#1767 port)
// ---------------------------------------------------------------------------

/// Divergence (NEW, distinct from the in-code `[1.0]` residual): ferrolearn's
/// `_translate_train_sizes` port chooses fraction-vs-absolute mode by VALUE
/// integrality (`ferrolearn-model-sel/src/learning_curve.rs:156`
/// `train_sizes.iter().all(|&s| s.fract() == 0.0)`), whereas sklearn keys off
/// the array's float-vs-int DTYPE (`sklearn/model_selection/_validation.py:2020`
/// `if np.issubdtype(train_sizes_abs.dtype, np.floating):`).
///
/// For an all-integer-VALUED but FLOAT-dtype input like `[2.0, 3.0]`:
///   - ferrolearn: `fract() == 0.0` for every entry ⇒ ABSOLUTE mode; `max=3 <=
///     n_max=20` ⇒ `Ok([2, 3])`.
///   - sklearn: `np.asarray([2.0, 3.0])` is float dtype ⇒ FRACTION mode; with
///     `n_max_required_samples = 3.0 > 1.0` ⇒ `ValueError` (`:2021-2027`).
///
/// This is a STRONGER divergence class than the documented `[1.0]` residual: the
/// in-code NOTE (`:152-155`) covers only the `<= 1.0` value-DIFFERENCE subcase
/// (ferrolearn `[1.0]` ⇒ 1 sample vs sklearn ⇒ 20 samples, both succeed). It does
/// NOT cover the `> 1.0` RAISE-vs-SUCCEED subcase introduced when the port added
/// the float-fraction `max > 1.0` ValueError guard: there sklearn RAISES while
/// ferrolearn SUCCEEDS, which the documented residual does not contemplate.
///
/// LIVE sklearn 1.5.2 oracle:
/// `_translate_train_sizes(np.asarray([2.0, 3.0]), 20)` raises `ValueError`;
/// ferrolearn returns `Ok` with ticks `[2, 3]`.
///
/// Tracking: #1769 new-blocker.
// #1769 new-blocker
#[ignore = "divergence: all-integer-valued FLOAT >1.0 raises in sklearn (float dtype -> fraction mode) but ferrolearn treats as absolute; tracking #1769"]
#[test]
fn divergence_1769_integer_valued_float_gt_one_should_error() {
    let y: Array1<f64> = (0..30).map(f64::from).collect();
    let x = Array2::<f64>::zeros((30, 2));
    let kf = KFold::new(3); // first fold train length == 20 == oracle n_max
    let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));

    // sklearn: np.asarray([2.0, 3.0]) is float dtype -> fraction mode ->
    // max 3.0 > 1.0 -> ValueError. ferrolearn must error too.
    let result = learning_curve(&pipeline, &x, &y, &kf, &[2.0, 3.0], neg_mse);

    assert!(
        result.is_err(),
        "sklearn raises ValueError for the float-dtype array [2.0, 3.0] \
         (fraction mode, max 3.0 > 1.0); ferrolearn's value-integrality heuristic \
         treats it as absolute counts [2, 3] and returns Ok",
    );
}
