//! Divergence pin: `ferrolearn_core::pipeline::FeatureUnion` accepts duplicate
//! transformer names, whereas scikit-learn's `FeatureUnion` raises a deliberate
//! `ValueError` at fit time.
//!
//! sklearn validates transformer names at fit via `_validate_names`
//! (`sklearn/utils/metaestimators.py:81-83`):
//!   ```text
//!   def _validate_names(self, names):
//!       if len(set(names)) != len(names):
//!           raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
//!   ```
//! reached from `FeatureUnion._validate_transformers`
//! (`sklearn/pipeline.py:1523-1525`):
//!   ```text
//!   names, transformers = zip(*self.transformer_list)
//!   # validate names
//!   self._validate_names(names)
//!   ```
//! which `FeatureUnion._parallel_func` (`sklearn/pipeline.py:1753`) calls on
//! every `fit` / `fit_transform`. This is INTENTIONAL validation (a clean,
//! deliberately-constructed `ValueError`), not an incidental CPython footgun —
//! so under goal.md R-DEV-2 (ABI/exception match) ferrolearn should also error.
//!
//! Live oracle (sklearn 1.5.2):
//!   ```text
//!   from sklearn.pipeline import FeatureUnion
//!   from sklearn.preprocessing import StandardScaler, MinMaxScaler
//!   import numpy as np
//!   X = np.array([[1.,2.],[3.,4.],[5.,6.]])
//!   FeatureUnion([('a', StandardScaler()), ('a', MinMaxScaler())]).fit(X)
//!   # -> ValueError: Names provided are not unique: ['a', 'a']
//!   ```
//!
//! ferrolearn actual: `FeatureUnion::fit` (`ferrolearn-core/src/pipeline.rs:1111`)
//! performs NO duplicate-name check. It fits successfully, and
//! `FittedFeatureUnion::get_feature_names_out`
//! (`ferrolearn-core/src/pipeline.rs:1192`) then emits the COLLIDING names
//! `["a__x0", "a__x1", "a__x0", "a__x1"]` (two transformers both named "a").
//! sklearn never reaches `get_feature_names_out` because `fit` already raised.
//!
//! Tracking: #2237

use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FeatureUnion, FittedPipelineTransformer, PipelineTransformer};
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

/// Width-preserving identity transformer (the `OneToOneFeatureMixin` shape, like
/// `StandardScaler` / `MinMaxScaler`).
struct IdT;
impl PipelineTransformer<f64> for IdT {
    fn fit_pipeline(
        &self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
        Ok(Box::new(FIdT))
    }
}
struct FIdT;
impl FittedPipelineTransformer<f64> for FIdT {
    fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
        Ok(x.clone())
    }
}

/// Divergence: a `FeatureUnion` built with two transformers sharing the name
/// `"a"` must fail at `fit`, mirroring sklearn's `_validate_names` `ValueError`.
///
/// sklearn returns: `ValueError("Names provided are not unique: ['a', 'a']")`
/// (deliberate validation, `sklearn/utils/metaestimators.py:81-83`).
/// ferrolearn returns: `Ok(FittedFeatureUnion)` — duplicate names accepted,
/// then colliding `["a__x0","a__x1","a__x0","a__x1"]` feature names.
///
/// Tracking: #2237
#[test]
#[ignore = "divergence: FeatureUnion accepts duplicate transformer names (sklearn raises ValueError at fit); tracking #2237"]
fn divergence_feature_union_duplicate_names_must_error() {
    let union = FeatureUnion::<f64>::new()
        .with_transformer("a", Box::new(IdT))
        .with_transformer("a", Box::new(IdT));
    let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // sklearn raises ValueError here (Names provided are not unique). ferrolearn
    // should likewise reject the union at fit. The expected behavior is a
    // FerroError (the ValueError analog) — NOT a successful fit.
    let result = union.fit(&x, &());

    assert!(
        result.is_err(),
        "FeatureUnion with duplicate transformer names must error at fit \
         (sklearn raises ValueError: Names provided are not unique: ['a', 'a'], \
         sklearn/utils/metaestimators.py:81-83); ferrolearn accepted it and \
         produced colliding get_feature_names_out: {:?}",
        result.map(|f| f.get_feature_names_out())
    );
}
