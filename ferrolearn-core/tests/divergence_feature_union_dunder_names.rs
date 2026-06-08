//! Divergence pin: `ferrolearn_core::pipeline::FeatureUnion::fit` accepts a
//! transformer name containing the double-underscore `__`, whereas
//! scikit-learn's `_validate_names` raises a deliberate `ValueError`.
//!
//! The #2237 fix added a duplicate-name uniqueness check to `FeatureUnion::fit`
//! (`ferrolearn-core/src/pipeline.rs:1120-1131`), mirroring the FIRST clause of
//! sklearn's `_validate_names` (`sklearn/utils/metaestimators.py:81-83`). But
//! `_validate_names` has a THIRD clause (`sklearn/utils/metaestimators.py:91-95`)
//! that ferrolearn does NOT mirror:
//!   ```text
//!   invalid_names = [name for name in names if "__" in name]
//!   if invalid_names:
//!       raise ValueError(
//!           "Estimator names must not contain __: got {0!r}".format(invalid_names)
//!       )
//!   ```
//! `__` is reserved for the nested-parameter addressing protocol
//! (`<step>__<param>`), so sklearn forbids it in any step name. This clause is
//! reached on every `fit` / `fit_transform` via
//! `FeatureUnion._validate_transformers` (`sklearn/pipeline.py:1523-1525`):
//!   ```text
//!   names, transformers = zip(*self.transformer_list)
//!   # validate names
//!   self._validate_names(names)
//!   ```
//! Like the duplicate-name clause, this is INTENTIONAL validation (a clean,
//! deliberately-constructed `ValueError`), so under goal.md R-DEV-2 (ABI /
//! exception match) ferrolearn should also error.
//!
//! Live oracle (sklearn 1.5.2):
//!   ```text
//!   from sklearn.pipeline import FeatureUnion
//!   from sklearn.preprocessing import StandardScaler, MinMaxScaler
//!   import numpy as np
//!   X = np.array([[1.,2.],[3.,4.],[5.,6.]])
//!   FeatureUnion([('a__b', StandardScaler()), ('c', MinMaxScaler())]).fit(X)
//!   # -> ValueError: Estimator names must not contain __: got ['a__b']
//!   ```
//! The names are otherwise UNIQUE (`'a__b'`, `'c'`), so the duplicate-name check
//! does NOT fire — this is the `__`-clause, not the uniqueness clause.
//!
//! ferrolearn actual (verified): `FeatureUnion::fit`
//! (`ferrolearn-core/src/pipeline.rs:1111`) checks only HashSet uniqueness, so a
//! `__`-containing-but-unique name passes. The union fits successfully and
//! `FittedFeatureUnion::get_feature_names_out`
//! (`ferrolearn-core/src/pipeline.rs:1213`) emits
//! `["a__b__x0", "a__b__x1", "c__x0", "c__x1"]`. sklearn never reaches
//! `get_feature_names_out` because `fit` already raised.
//!
//! Tracking: #2238

use ferrolearn_core::Fit;
use ferrolearn_core::error::FerroError;
use ferrolearn_core::pipeline::{FeatureUnion, FittedPipelineTransformer, PipelineTransformer};
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

/// Divergence: a `FeatureUnion` built with a UNIQUE-but-`__`-containing
/// transformer name (`"a__b"`) must fail at `fit`, mirroring the third clause of
/// sklearn's `_validate_names` (`sklearn/utils/metaestimators.py:91-95`).
///
/// sklearn returns (live oracle, sklearn 1.5.2):
///   `ValueError("Estimator names must not contain __: got ['a__b']")`.
/// ferrolearn returns (verified): `Ok(FittedFeatureUnion)` — the `__`-name is
/// accepted because the #2237 fix only checks duplicate-name uniqueness, then
/// `get_feature_names_out` emits `["a__b__x0","a__b__x1","c__x0","c__x1"]`.
///
/// Tracking: #2238
#[test]
fn divergence_feature_union_dunder_name_must_error() {
    let union = FeatureUnion::<f64>::new()
        .with_transformer("a__b", Box::new(IdT))
        .with_transformer("c", Box::new(IdT));
    let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // sklearn raises ValueError here (Estimator names must not contain __).
    // The names are unique, so the duplicate-name clause does NOT fire — this is
    // the `__`-clause specifically. ferrolearn should likewise reject at fit.
    let result = union.fit(&x, &());

    assert!(
        result.is_err(),
        "FeatureUnion with a `__`-containing transformer name must error at fit \
         (sklearn raises ValueError: Estimator names must not contain __: got ['a__b'], \
         sklearn/utils/metaestimators.py:91-95); ferrolearn accepted it and produced \
         get_feature_names_out: {:?}",
        result.map(|f| f.get_feature_names_out())
    );
}

/// Guard: only the DOUBLE underscore is reserved. A name with a SINGLE
/// underscore (`"a_b"`) must fit OK — sklearn's `_validate_names` only rejects
/// `"__" in name` (`sklearn/utils/metaestimators.py:91`), not a lone `_`.
///
/// Live oracle (sklearn 1.5.2):
///   `FeatureUnion([('a_b', StandardScaler())]).fit(X)` -> OK (no ValueError).
#[test]
fn single_underscore_name_is_ok() {
    let union = FeatureUnion::<f64>::new().with_transformer("a_b", Box::new(IdT));
    let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let result = union.fit(&x, &());

    assert!(
        result.is_ok(),
        "FeatureUnion with a single-underscore name `a_b` must fit OK (only `__` \
         is reserved, sklearn/utils/metaestimators.py:91); got error: {:?}",
        result.err()
    );
}

/// Guard: `__` at the START or END of a name is still a `__` occurrence, so it
/// must error (sklearn's check is `"__" in name`, position-agnostic).
///
/// Live oracle (sklearn 1.5.2):
///   `FeatureUnion([('__a', StandardScaler())]).fit(X)` -> ValueError, and
///   `FeatureUnion([('a__', StandardScaler())]).fit(X)` -> ValueError.
#[test]
fn leading_or_trailing_dunder_name_must_error() {
    let x = ndarray::array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let leading = FeatureUnion::<f64>::new()
        .with_transformer("__a", Box::new(IdT))
        .fit(&x, &());
    assert!(
        leading.is_err(),
        "leading-`__` name `__a` must error (sklearn ValueError: Estimator names \
         must not contain __: got ['__a'])"
    );

    let trailing = FeatureUnion::<f64>::new()
        .with_transformer("a__", Box::new(IdT))
        .fit(&x, &());
    assert!(
        trailing.is_err(),
        "trailing-`__` name `a__` must error (sklearn ValueError: Estimator names \
         must not contain __: got ['a__'])"
    );
}
