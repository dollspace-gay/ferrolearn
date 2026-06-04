//! Adversarial divergence audit of `ferrolearn-model-sel::feature_union`
//! (`FeatureUnion`) against scikit-learn 1.5.2 (`sklearn.pipeline.FeatureUnion`,
//! `sklearn/pipeline.py:1329`).
//!
//! Tracking: #1676.
//!
//! Test classes:
//! - GREEN GUARD (must PASS): core REQ-1 concat value/column-order parity and
//!   REQ-6 empty-list rejection. Expected values come from a LIVE sklearn 1.5.2
//!   oracle run from /tmp (R-CHAR-3 — never copied from ferrolearn).
//! - FAILING PINS (`#[ignore]`, must FAIL when run): REQ-2 transformer_weights,
//!   REQ-3 drop/passthrough, REQ-4 get_feature_names_out. Each asserts the live
//!   sklearn behavior that ferrolearn does not reproduce.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_model_sel::FeatureUnion;
use ndarray::{Array1, Array2, Axis, array};

// ---------------------------------------------------------------------------
// Synthetic sub-transformers with KNOWN deterministic behavior.
// ---------------------------------------------------------------------------

/// Identity: `FunctionTransformer(lambda X: X)`.
struct Identity;
impl PipelineTransformer<f64> for Identity {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FittedIdentity))
    }
}
struct FittedIdentity;
impl FittedPipelineTransformer<f64> for FittedIdentity {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.clone())
    }
}

/// Doubler: `FunctionTransformer(lambda X: X * 2)`.
struct Doubler;
impl PipelineTransformer<f64> for Doubler {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FittedDoubler))
    }
}
struct FittedDoubler;
impl FittedPipelineTransformer<f64> for FittedDoubler {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.mapv(|v| v * 2.0))
    }
}

/// `StandardScaler` (population std, ddof=0): `(x - mean) / std` per column,
/// fitted on the training matrix. Mirrors `sklearn.preprocessing.StandardScaler`.
struct StandardScaler;
struct FittedStandardScaler {
    mean: Array1<f64>,
    std: Array1<f64>,
}
impl PipelineTransformer<f64> for StandardScaler {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        let n = x.nrows() as f64;
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut std = Array1::<f64>::zeros(x.ncols());
        for j in 0..x.ncols() {
            let col = x.column(j);
            let var = col.iter().map(|&v| (v - mean[j]).powi(2)).sum::<f64>() / n;
            // sklearn replaces zero scale with 1.0; not needed for this fixture.
            std[j] = var.sqrt();
        }
        Ok(Box::new(FittedStandardScaler { mean, std }))
    }
}
impl FittedPipelineTransformer<f64> for FittedStandardScaler {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let mut out = x.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                out[[i, j]] = (x[[i, j]] - self.mean[j]) / self.std[j];
            }
        }
        Ok(out)
    }
}

/// `MinMaxScaler` (feature_range=(0,1)): `(x - min) / (max - min)` per column.
/// Mirrors `sklearn.preprocessing.MinMaxScaler`.
struct MinMaxScaler;
struct FittedMinMaxScaler {
    min: Array1<f64>,
    range: Array1<f64>,
}
impl PipelineTransformer<f64> for MinMaxScaler {
    fn fit_pipeline(
        &self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        let mut min = Array1::<f64>::zeros(x.ncols());
        let mut range = Array1::<f64>::zeros(x.ncols());
        for j in 0..x.ncols() {
            let col = x.column(j);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            min[j] = lo;
            range[j] = hi - lo;
        }
        Ok(Box::new(FittedMinMaxScaler { min, range }))
    }
}
impl FittedPipelineTransformer<f64> for FittedMinMaxScaler {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        let mut out = x.clone();
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                out[[i, j]] = (x[[i, j]] - self.min[j]) / self.range[j];
            }
        }
        Ok(out)
    }
}

fn fixture_x() -> Array2<f64> {
    array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
}

// ===========================================================================
// GREEN GUARDS (REQ-1, REQ-6) — must PASS.
// ===========================================================================

/// REQ-1 (AC-1): concat value + column-order parity vs the live sklearn oracle.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// FeatureUnion([('a', FunctionTransformer(lambda Z: Z)),
///               ('b', FunctionTransformer(lambda Z: Z*2))]).fit_transform(X)
/// X = [[1,2,3],[4,5,6],[7,8,10]]  ->  shape (3, 6)
/// [[ 1,  2,  3,  2,  4,  6],
///  [ 4,  5,  6,  8, 10, 12],
///  [ 7,  8, 10, 14, 16, 20]]
/// ```
/// Columns 0-2 = identity block, columns 3-5 = doubler block (registration
/// order). Mirrors `_hstack` (`sklearn/pipeline.py:1820` `np.hstack(Xs)`).
#[test]
fn green_req1_identity_doubler_concat_parity() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    let fu = FeatureUnion::<f64>::new()
        .add("a", Box::new(Identity))
        .add("b", Box::new(Doubler));
    let out = fu.fit(&x, &y).unwrap().transform(&x).unwrap();

    // Live sklearn oracle matrix.
    let expected = array![
        [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        [4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        [7.0, 8.0, 10.0, 14.0, 16.0, 20.0],
    ];
    assert_eq!(out.dim(), (3, 6));
    for i in 0..3 {
        for j in 0..6 {
            assert!(
                (out[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "mismatch at [{i},{j}]: got {}, oracle {}",
                out[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

/// REQ-1 (AC-1): value parity for a StandardScaler + MinMaxScaler union vs the
/// live sklearn oracle — pins the hstack column order AND per-element values.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
/// ```text
/// FeatureUnion([('s', StandardScaler()), ('m', MinMaxScaler())]).fit_transform(X)
/// X = [[1,2,3],[4,5,6],[7,8,10]]  ->  shape (3, 6)
/// ```
/// Columns 0-2 = StandardScaler block, columns 3-5 = MinMaxScaler block.
#[test]
fn green_req1_scaler_union_value_parity() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    let fu = FeatureUnion::<f64>::new()
        .add("s", Box::new(StandardScaler))
        .add("m", Box::new(MinMaxScaler));
    let out = fu.fit(&x, &y).unwrap().transform(&x).unwrap();

    // Live sklearn oracle matrix (full f64 precision).
    let expected = array![
        [
            -1.2247448713915892,
            -1.2247448713915892,
            -1.1624763874381927,
            0.0,
            0.0,
            0.0
        ],
        [
            0.0,
            0.0,
            -0.11624763874381917,
            0.5,
            0.49999999999999994,
            0.42857142857142855
        ],
        [
            1.2247448713915892,
            1.2247448713915892,
            1.2787240261820123,
            0.9999999999999999,
            1.0,
            0.9999999999999998
        ],
    ];
    assert_eq!(out.dim(), (3, 6));
    for i in 0..3 {
        for j in 0..6 {
            assert!(
                (out[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "mismatch at [{i},{j}]: got {}, oracle {}",
                out[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

/// REQ-6 (AC-6): empty union is rejected, matching sklearn.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
/// `FeatureUnion([]).fit_transform(X)` raises
/// `ValueError: not enough values to unpack (expected 2, got 0)`.
/// ferrolearn returns `FerroError::InvalidParameter` (the project's
/// ValueError-class analog, R-DEV-2). Both reject — green guard.
#[test]
fn green_req6_empty_union_rejected() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    let result = FeatureUnion::<f64>::new().fit(&x, &y);
    let is_invalid_param = matches!(result, Err(FerroError::InvalidParameter { .. }));
    assert!(
        is_invalid_param,
        "expected Err(InvalidParameter) (ValueError-class) for empty union"
    );
}

// ===========================================================================
// FAILING PINS — must FAIL when run (the NOT-STARTED REQs).
// ===========================================================================

/// REQ-2 (AC-2): `transformer_weights` scales a block before concatenation.
///
/// sklearn `_iter` yields `get_weight(name)` (`sklearn/pipeline.py:1558,:1565`)
/// and `_transform_one` applies `_weight_one` so block 'b' is multiplied by 10:
/// ```text
/// FeatureUnion([('a', FunctionTransformer(lambda Z: Z)),
///               ('b', FunctionTransformer(lambda Z: Z*2))],
///              transformer_weights={'b': 10}).fit_transform(X)
/// -> [[ 1,  2,  3, 20, 40, 60], ...]   (block 'b' = 2*X*10)
/// ```
/// ferrolearn `FeatureUnion` has NO `transformer_weights` field/API, so the 'b'
/// block stays at 2*X (cols 3-5 = 2,4,6 on row 0, not 20,40,60).
/// This test asserts the WEIGHTED oracle; ferrolearn produces the unweighted
/// matrix, so it FAILS — pinning the missing weights surface.
#[test]
fn divergence_req2_transformer_weights() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    // ferrolearn now expresses transformer_weights={'b':10} via add_weighted:
    // block 'b' (the doubler) is scaled by 10 before concatenation.
    let fu = FeatureUnion::<f64>::new()
        .add("a", Box::new(Identity))
        .add_weighted("b", Box::new(Doubler), 10.0);
    let out = fu.fit(&x, &y).unwrap().transform(&x).unwrap();

    // Live sklearn oracle WITH transformer_weights={'b':10}.
    let expected = array![
        [1.0, 2.0, 3.0, 20.0, 40.0, 60.0],
        [4.0, 5.0, 6.0, 80.0, 100.0, 120.0],
        [7.0, 8.0, 10.0, 140.0, 160.0, 200.0],
    ];
    for i in 0..3 {
        for j in 0..6 {
            assert!(
                (out[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "weighted-block mismatch at [{i},{j}]: got {}, sklearn-weighted {}",
                out[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

/// REQ-3 (AC-3): a `'drop'` transformer is skipped entirely.
///
/// sklearn `_iter` does `if trans == "drop": continue` (`sklearn/pipeline.py:1561`)
/// so the dropped block contributes no columns:
/// ```text
/// FeatureUnion([('a', FunctionTransformer(lambda Z: Z)),
///               ('b', 'drop')]).fit_transform(X)  ->  shape (3, 3)
/// ```
/// ferrolearn `add` only accepts a `Box<dyn PipelineTransformer>`; there is no
/// `'drop'` sentinel, so every registered transformer always emits its columns.
/// This test asserts the sklearn output shape (3, 3); ferrolearn yields (3, 6),
/// so it FAILS — pinning the missing drop/passthrough sentinels.
#[test]
#[ignore = "divergence: FeatureUnion has no 'drop'/'passthrough' sentinel (REQ-3); tracking #1676"]
fn divergence_req3_drop_sentinel() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    // Closest reachable: a 2-transformer union where 'b' should be 'drop'.
    let fu = FeatureUnion::<f64>::new()
        .add("a", Box::new(Identity))
        .add("b", Box::new(Doubler)); // sklearn would set this to 'drop'
    let out = fu.fit(&x, &y).unwrap().transform(&x).unwrap();

    // Live sklearn oracle: ('b','drop') => only block 'a' (3 columns) remains.
    assert_eq!(
        out.ncols(),
        3,
        "sklearn drops 'b' -> 3 columns; ferrolearn kept all -> {}",
        out.ncols()
    );
}

/// REQ-4 (AC-4): `get_feature_names_out` prefixes each name `f"{name}__{feat}"`.
///
/// sklearn `_add_prefix_for_feature_names_out` (`sklearn/pipeline.py:1608-1616`)
/// produces, for `FeatureUnion([('s', StandardScaler()), ('m', MinMaxScaler())])`
/// with input features `['c0','c1','c2']`:
/// ```text
/// ['s__c0', 's__c1', 's__c2', 'm__c0', 'm__c1', 'm__c2']
/// ```
/// ferrolearn exposes only `transformer_names()` (`["s", "m"]`) and has no
/// `get_feature_names_out`. This test asserts the sklearn prefixed names against
/// ferrolearn's `transformer_names()`; they differ, so it FAILS — pinning the
/// absent feature-name surface.
#[test]
#[ignore = "divergence: FeatureUnion has no get_feature_names_out (REQ-4); tracking #1676"]
fn divergence_req4_get_feature_names_out() {
    let x = fixture_x();
    let y = Array1::<f64>::zeros(3);
    let fu = FeatureUnion::<f64>::new()
        .add("s", Box::new(StandardScaler))
        .add("m", Box::new(MinMaxScaler));
    let fitted = fu.fit(&x, &y).unwrap();

    // Live sklearn oracle: prefixed feature names.
    let expected: Vec<String> = vec!["s__c0", "s__c1", "s__c2", "m__c0", "m__c1", "m__c2"]
        .into_iter()
        .map(String::from)
        .collect();
    // ferrolearn's closest reachable surface is transformer_names() == ["s","m"].
    let actual: Vec<String> = fitted
        .transformer_names()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(
        actual, expected,
        "ferrolearn has no get_feature_names_out; transformer_names()={actual:?} \
         vs sklearn prefixed names {expected:?}"
    );
}
