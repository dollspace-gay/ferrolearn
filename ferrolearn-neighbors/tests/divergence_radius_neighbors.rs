//! Adversarial divergence pins / value-contract guards for
//! `ferrolearn-neighbors/src/radius_neighbors.rs`
//! (`RadiusNeighborsClassifier` / `FittedRadiusNeighborsClassifier` +
//! `RadiusNeighborsRegressor` / `FittedRadiusNeighborsRegressor`:
//! `fit` / `predict` / `predict_proba` / `score` / `classes` / `radius_neighbors`)
//! against the live scikit-learn 1.5.2 oracle
//! (`from sklearn.neighbors import RadiusNeighborsClassifier,
//! RadiusNeighborsRegressor`, mirroring `sklearn/neighbors/_classification.py` +
//! `_regression.py` + `_base.py`).
//!
//! Every expected value below is captured from a live `python3 -c "..."` run of
//! sklearn 1.5.2, executed from `/tmp` (R-CHAR-3 — never literal-copied from the
//! ferrolearn side). The exact oracle call and its output are quoted above each
//! assertion block.
//!
//! Design doc: `.design/neighbors/radius_neighbors.md` (commit 0677bb3f).
//! Upstream: `sklearn/neighbors/_classification.py` (`RadiusNeighborsClassifier`),
//! `sklearn/neighbors/_regression.py` (`RadiusNeighborsRegressor`),
//! `sklearn/neighbors/_base.py` (`RadiusNeighborsMixin.radius_neighbors`,
//! `_get_weights`).
//!
//! ferrolearn classifier labels are `Array1<usize>` — fixtures use integer labels.
//!
//! GREEN guards (must PASS — SHIPPED value contracts, R-DEV-1/3):
//!   * `green_classifier_predict_value_uniform_and_distance` — clf `predict`.
//!   * `green_classifier_predict_proba_value`                — clf `predict_proba`.
//!   * `green_classifier_outlier_label_some`                 — clf outlier `Some`.
//!   * `green_regressor_predict_value_uniform_and_distance`  — reg `predict`.
//!   * `green_classifier_score_accuracy`                     — clf `score`.
//!   * `green_radius_neighbors_set`                          — shared radius search.
//!
//! RED pins (must FAIL now — deterministic, single-file-fixable divergences):
//!   * `divergence_regressor_no_neighbor_returns_nan_not_error` — REQ-7 / #882.
//!     sklearn `RadiusNeighborsRegressor.predict` empty row -> `np.nan` + warn
//!     (`_regression.py:482,504-509`); ferrolearn `fn predict` returns
//!     `Err(InvalidParameter)`. Single-file fix: push `F::nan()` for the empty row.
//!   * `divergence_classifier_predict_proba_outlier_not_in_classes_all_zero`
//!     — REQ-6 / #881. sklearn `predict_proba` empty row for an `outlier_label`
//!     NOT in `classes_` -> all-zero (`_classification.py:813-824,826-829`);
//!     ferrolearn falls back to **uniform** `1/n_classes`. Single-file fix: the
//!     empty-neighborhood `else` branch in `pub fn predict_proba` should leave the
//!     row all-zero rather than filling uniform.
//!
//! ==========================================================================
//! NOT-STARTED / MISSING-SURFACE divergences — DOCUMENTED ONLY, no forced test.
//! ==========================================================================
//! Following the `divergence_knn.rs` sibling precedent, the divergences below are
//! NOT pinned with a committed RED test because they require the generator to add
//! API surface (a discriminator may not author production code), span multiple
//! files, or are inexpressible through the current public API:
//!
//! **clf `outlier_label='most_frequent'`** (REQ-6 / #881): ferrolearn's
//! `outlier_label: Option<usize>` has no `'most_frequent'` mode. sklearn assigns the
//! training mode to outliers (`_classification.py:636-642`); live oracle
//! `outlier_label_ == [0]`, `predict` far -> `[0]`. MISSING SURFACE — the
//! `Option<usize>` field cannot express `'most_frequent'`. (The all-zero
//! predict_proba half of #881 IS pinned above as a RED.)
//!
//! **`radius_neighbors` `sort_results` default order** (REQ-11 / #883):
//! `fn find_radius_neighbors` ALWAYS sorts ascending (matches sklearn
//! `sort_results=True`), but sklearn's DEFAULT is `sort_results=False` -> native
//! (tree/brute) order. There is no `sort_results` toggle and no `X=None`
//! self-exclusion. The neighbor SET matches; the default ORDER diverges (R-DEV-3).
//! The `green_radius_neighbors_set` guard below compares as a SET to stay green.
//!
//! **regressor multi-output 2-D `y`** (REQ-13 / #884): `FittedRadiusNeighbors
//! Regressor` stores `y_train: Array1<F>` and `impl Fit<Array2<F>, Array1<F>>` —
//! 1-D only. sklearn `RadiusNeighborsRegressor.predict` reshapes `_y` to 2-D and
//! returns `(n_queries, n_outputs)` (`_regression.py:478-514`); live oracle
//! `fit(X, Y_2col).predict(Xq)` -> `(1,2)` `[[10.0, 200.0]]`. MISSING SURFACE — no
//! 2-D `y` `Fit` impl to call.
//!
//! **missing constructor params + callable weights** (REQ-14 / #885): both
//! estimators expose `radius`/`weights ∈ {Uniform, Distance}`/`algorithm`
//! (+ clf `outlier_label`) but NO `leaf_size=30` (`_classification.py:584`/
//! `_regression.py:418`), `p=2` (`:585`/`:419`), `metric='minkowski'` (`:586`/
//! `:421`), `metric_params=None` (`:588`/`:421`), `n_jobs=None` (`:589`/`:422`);
//! `Weights` has no callable variant (`:573`/`:408`). Euclidean-only. MISSING FIELDS.
//!
//! **clf no-neighbor error type/message** (REQ-12 / #886): sklearn raises
//! `ValueError("No neighbors found for test samples %r, ...")`
//! (`_classification.py:781-787`); ferrolearn raises `FerroError::InvalidParameter`
//! with a different message. Both raise — the type/message divergence is a marshalling
//! concern surfacing at the PyO3 boundary (REQ-15); not pinnable as a Rust value test.
//!
//! **no PyO3 binding / meta-crate re-export** (REQ-15 / #887): `import sklearn`
//! gives `RadiusNeighborsClassifier`/`RadiusNeighborsRegressor`; `import ferrolearn`
//! gives nothing (`grep -ni radiusneighbors ferrolearn-python/src/` empty, meta-crate
//! has no re-export). A pytest divergence, not a Rust test.
//!
//! **ferray substrate** (REQ-16 / #888): `radius_neighbors.rs` imports
//! `ndarray::{Array1, Array2}` + `num_traits::Float`, not `ferray-core`
//! (R-SUBSTRATE). Audited by inspection, not a runtime value test.
//!
//! CONSUMER-BACKING (REQ-table input): the ONLY non-test consumer of the
//! RadiusNeighbors fitted types is `graph.rs`, which uses `fit` + `radius_neighbors`
//! (the `radius_neighbors_graph` free function constructs a clf, calls `fit` then
//! `fitted.radius_neighbors`; the fitted `radius_neighbors_graph` methods call
//! `self.radius_neighbors`). NO non-test consumer calls `predict`, `predict_proba`,
//! `score`, `classes`, or `n_classes` on either fitted type — those are TEST-ONLY
//! pub surface (grandfathered S5/R-DEFER-1, like knn's predict_proba/score).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_neighbors::{RadiusNeighborsClassifier, RadiusNeighborsRegressor, Weights};
use ndarray::{Array2, array};

// ===========================================================================
// GREEN guards — SHIPPED value contracts (must PASS now).
// ===========================================================================

/// GREEN guard — REQ-1. clf `predict` value, uniform + distance weights.
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X = [[0.,0.],[0.5,0.],[0.,0.5],[5.,5.],[5.5,5.],[5.,5.5]]; y = [0,0,0,1,1,1]
/// Xq = [[0.2,0.1],[5.2,5.1]]
/// RadiusNeighborsClassifier(radius=1.5).fit(X,y).predict(Xq)
///   -> [0, 1]
/// RadiusNeighborsClassifier(radius=1.5, weights='distance').fit(X,y).predict(Xq)
///   -> [0, 1]
/// ```
#[test]
fn green_classifier_predict_value_uniform_and_distance() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1, 1];
    let xq = Array2::from_shape_vec((2, 2), vec![0.2, 0.1, 5.2, 5.1]).unwrap();

    // sklearn oracle: [0, 1]
    let preds_uniform = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(1.5)
        .fit(&x, &y)
        .unwrap()
        .predict(&xq)
        .unwrap();
    assert_eq!(preds_uniform.to_vec(), vec![0, 1], "uniform predict");

    // sklearn oracle: [0, 1]
    let preds_distance = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(1.5)
        .with_weights(Weights::Distance)
        .fit(&x, &y)
        .unwrap()
        .predict(&xq)
        .unwrap();
    assert_eq!(preds_distance.to_vec(), vec![0, 1], "distance predict");
}

/// GREEN guard — REQ-2. clf `predict_proba` value, distance + uniform.
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X = [[0.],[1.],[2.],[10.],[11.]]; y = [0,1,0,1,1]; Xq = [[1.5]]
/// RadiusNeighborsClassifier(radius=5.0, weights='distance').fit(X,y).predict_proba(Xq)
///   -> [[0.5714285714285715, 0.4285714285714286]]
/// RadiusNeighborsClassifier(radius=5.0).fit(X,y).predict_proba(Xq)
///   -> [[0.6666666666666666, 0.3333333333333333]]
/// ```
#[test]
fn green_classifier_predict_proba_value() {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 10.0, 11.0]).unwrap();
    let y = array![0, 1, 0, 1, 1];
    let xq = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();

    // sklearn oracle (distance): [[0.5714285714285715, 0.4285714285714286]]
    let proba_distance = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(5.0)
        .with_weights(Weights::Distance)
        .fit(&x, &y)
        .unwrap()
        .predict_proba(&xq)
        .unwrap();
    assert!(
        (proba_distance[[0, 0]] - 0.5714285714285715).abs() < 1e-12
            && (proba_distance[[0, 1]] - 0.4285714285714286).abs() < 1e-12,
        "distance predict_proba: got {proba_distance:?}"
    );

    // sklearn oracle (uniform): [[0.6666666666666666, 0.3333333333333333]]
    let proba_uniform = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(5.0)
        .fit(&x, &y)
        .unwrap()
        .predict_proba(&xq)
        .unwrap();
    assert!(
        (proba_uniform[[0, 0]] - 0.6666666666666666).abs() < 1e-12
            && (proba_uniform[[0, 1]] - 0.3333333333333333).abs() < 1e-12,
        "uniform predict_proba: got {proba_uniform:?}"
    );
}

/// GREEN guard — REQ-4. clf `outlier_label=Some(99)`, query far from all training.
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X = [[0.],[1.],[2.]]; y = [0,0,1]
/// RadiusNeighborsClassifier(radius=0.01, outlier_label=99).fit(X,y).predict([[100.]])
///   -> [99]
/// ```
#[test]
fn green_classifier_outlier_label_some() {
    let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let y = array![0, 0, 1];
    let xq = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();

    // sklearn oracle: [99]
    let preds = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(0.01)
        .with_outlier_label(Some(99))
        .fit(&x, &y)
        .unwrap()
        .predict(&xq)
        .unwrap();
    assert_eq!(preds.to_vec(), vec![99], "outlier_label Some predict");
}

/// GREEN guard — REQ-8. reg `predict` value, uniform + distance weights.
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// RadiusNeighborsRegressor(radius=1.5).fit([[0.],[1.],[2.]], [0.,10.,20.]).predict([[1.0]])
///   -> [10.0]
/// RadiusNeighborsRegressor(radius=15.0, weights='distance')
///     .fit([[0.],[10.]], [0.,100.]).predict([[1.0]])
///   -> [10.0]
/// ```
#[test]
fn green_regressor_predict_value_uniform_and_distance() {
    let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let y = array![0.0, 10.0, 20.0];
    let xq = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    // sklearn oracle (uniform): [10.0]
    let preds_uniform = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(1.5)
        .fit(&x, &y)
        .unwrap()
        .predict(&xq)
        .unwrap();
    assert!(
        (preds_uniform[0] - 10.0).abs() < 1e-12,
        "uniform reg predict: got {}",
        preds_uniform[0]
    );

    let x2 = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
    let y2 = array![0.0, 100.0];
    // sklearn oracle (distance): [10.0]
    let preds_distance = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(15.0)
        .with_weights(Weights::Distance)
        .fit(&x2, &y2)
        .unwrap()
        .predict(&xq)
        .unwrap();
    assert!(
        (preds_distance[0] - 10.0).abs() < 1e-12,
        "distance reg predict: got {}",
        preds_distance[0]
    );
}

/// GREEN guard — REQ-3. clf `score` mean accuracy.
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X = [[0.,0.],[0.5,0.],[0.,0.5],[5.,5.],[5.5,5.],[5.,5.5]]; y = [0,0,0,1,1,1]
/// RadiusNeighborsClassifier(radius=1.5).fit(X,y).score(X,y) -> 1.0
/// ```
#[test]
fn green_classifier_score_accuracy() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0, 5.5, 5.0, 5.0, 5.5],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1, 1];

    // sklearn oracle: 1.0
    let score = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(1.5)
        .fit(&x, &y)
        .unwrap()
        .score(&x, &y)
        .unwrap();
    assert!((score - 1.0).abs() < 1e-12, "clf score: got {score}");
}

/// GREEN guard — REQ-10. shared `radius_neighbors` search — SET + distance multiset.
///
/// ferrolearn always sorts ascending == sklearn `sort_results=True`. We compare as a
/// SET (index set + distance multiset) to remain agnostic to the default-order
/// divergence (REQ-11 / #883).
///
/// Oracle (sklearn 1.5.2, from /tmp):
/// ```text
/// X = [[10.,10.],[1.,0.],[0.,1.],[0.,0.],[1.,1.]]; y = [0,0,0,1,1]
/// clf = RadiusNeighborsClassifier(radius=2.0).fit(X,y)
/// d, i = clf.radius_neighbors([[0.2,0.1]], sort_results=True)
///   i[0] -> [3, 1, 2, 4]
///   d[0] -> [0.223606797749979, 0.806225774829855,
///            0.9219544457292888, 1.2041594578792296]
/// ```
#[test]
fn green_radius_neighbors_set() {
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![10.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let y = array![0, 0, 0, 1, 1];
    let xq = Array2::from_shape_vec((1, 2), vec![0.2, 0.1]).unwrap();

    let fitted = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(2.0)
        .fit(&x, &y)
        .unwrap();
    let (dists, idxs) = fitted.radius_neighbors(&xq, Some(2.0)).unwrap();

    // sklearn oracle index SET: {3, 1, 2, 4}
    let mut got_idx = idxs[0].clone();
    got_idx.sort_unstable();
    assert_eq!(got_idx, vec![1, 2, 3, 4], "radius_neighbors index set");

    // sklearn oracle distance multiset (sorted): the four distances above.
    let mut got_dist = dists[0].clone();
    got_dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected = [
        0.223606797749979,
        0.806225774829855,
        0.9219544457292888,
        1.2041594578792296,
    ];
    assert_eq!(got_dist.len(), 4, "radius_neighbors count");
    for (g, e) in got_dist.iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-9,
            "radius_neighbors distance: got {g}, expected {e}"
        );
    }
}

// ===========================================================================
// RED pins — deterministic, single-file-fixable divergences (must FAIL now).
// ===========================================================================

/// RED pin — REQ-7 / #882. Regressor no-neighbor row: sklearn returns `np.nan`
/// (NOT an error) + a `UserWarning`; ferrolearn `fn predict` returns
/// `Err(InvalidParameter)`.
///
/// sklearn `RadiusNeighborsRegressor.predict` empty-row handling:
/// `_regression.py:482` — `empty_obs = np.full_like(_y[0], np.nan)`;
/// `_regression.py:504-509` — `UserWarning("One or more samples have no neighbors
/// within specified radius; predicting NaN.")`.
///
/// Oracle (sklearn 1.5.2, from /tmp, warnings suppressed):
/// ```text
/// m = RadiusNeighborsRegressor(radius=0.01).fit([[0.],[10.]], [0.,100.])
/// m.predict([[5.]]) -> [nan]   (+ UserWarning, does NOT raise)
/// ```
///
/// Minimal single-file fix: in the regressor `fn predict`, replace the
/// empty-neighborhood `return Err(...)` with pushing `F::nan()` for that row.
#[test]
fn divergence_regressor_no_neighbor_returns_nan_not_error() {
    let x = Array2::from_shape_vec((2, 1), vec![0.0, 10.0]).unwrap();
    let y = array![0.0, 100.0];
    let xq = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();

    let fitted = RadiusNeighborsRegressor::<f64>::new()
        .with_radius(0.01)
        .fit(&x, &y)
        .unwrap();

    // sklearn returns Ok([nan]); ferrolearn currently returns Err(InvalidParameter).
    let result = fitted.predict(&xq);
    assert!(
        result.is_ok(),
        "sklearn returns [nan] (not an error) for a no-neighbor query; \
         ferrolearn returned Err: {:?}",
        result.err()
    );
    let preds = result.unwrap();
    // sklearn oracle value: nan
    assert!(
        preds[0].is_nan(),
        "expected nan for no-neighbor row, got {}",
        preds[0]
    );
}

/// RED pin — REQ-6 / #881. Classifier `predict_proba` empty row for an
/// `outlier_label` NOT in `classes_`: sklearn assigns the row **all-zero**;
/// ferrolearn falls back to **uniform** `1/n_classes`.
///
/// sklearn `predict_proba` (`_classification.py:813-824` warn + leave all-zero,
/// `:826-829` normalize, a row summing to 0 stays 0): the outlier row is all-zero.
///
/// Oracle (sklearn 1.5.2, from /tmp, warnings suppressed):
/// ```text
/// X = [[0.],[1.],[2.]]; y = [0,1,0]
/// clf = RadiusNeighborsClassifier(radius=0.01, outlier_label=99).fit(X,y)
/// clf.classes_                 -> [0, 1]
/// clf.predict_proba([[100.]])  -> [[0.0, 0.0]]   (+ "Outlier label 99 not in
///                                 training classes" UserWarning)
/// ```
///
/// Minimal single-file fix: in `pub fn predict_proba`, the empty-neighborhood
/// `else` branch should leave the row all-zero rather than filling `1/n_classes`.
#[test]
fn divergence_classifier_predict_proba_outlier_not_in_classes_all_zero() {
    let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let y = array![0, 1, 0];
    let xq = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();

    let fitted = RadiusNeighborsClassifier::<f64>::new()
        .with_radius(0.01)
        .with_outlier_label(Some(99)) // 99 is NOT in classes_ == [0, 1]
        .fit(&x, &y)
        .unwrap();
    // classes_ unchanged (sklearn oracle): [0, 1]
    assert_eq!(fitted.classes(), &[0, 1], "classes_ should be [0, 1]");

    let proba = fitted.predict_proba(&xq).unwrap();
    // sklearn oracle: [[0.0, 0.0]] — both columns zero (outlier label not in classes).
    assert!(
        proba[[0, 0]] == 0.0 && proba[[0, 1]] == 0.0,
        "outlier-not-in-classes predict_proba should be all-zero like sklearn \
         [[0.0, 0.0]]; ferrolearn returned {proba:?} (uniform fallback)"
    );
}
