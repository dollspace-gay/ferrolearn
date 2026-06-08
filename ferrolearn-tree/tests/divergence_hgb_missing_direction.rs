//! Divergence pin for `ferrolearn-tree/src/hist_gradient_boosting.rs`
//! (`HistGradientBoostingClassifier`) NaN/missing-value *routing direction*
//! against scikit-learn 1.5.2.
//!
//! Context (issue #2276 HGB portion / tracking #2280): ferrolearn HGB already
//! has missing-value infrastructure (`NAN_BIN = u16::MAX`, NaN-aware binning,
//! a grower that tries both "NaN goes left" and "NaN goes right" and keeps the
//! higher-gain direction). On fixtures where missing values ARE present in
//! training, ferrolearn matches sklearn (the optimal direction is learned).
//!
//! THE GAP this test pins is the *no-missing-in-training* default direction.
//! When a node observed NO missing values during fit, sklearn fixes a definite
//! default for missing values seen at predict time:
//!
//! - `sklearn/ensemble/_hist_gradient_boosting/splitting.pyx:758-759`:
//!   "If no missing value are present in the data this method [right-to-left]
//!   isn't called since only calling
//!   _find_best_bin_to_split_left_to_right is enough."
//! - `sklearn/ensemble/_hist_gradient_boosting/splitting.pyx:719-720` (the
//!   left-to-right scan, the ONLY scan when no missing in training):
//!   `# we scan from left to right so missing values go to the right`;
//!   `split_info.missing_go_to_left = False`
//!
//! i.e. sklearn's contract is: a node that saw no missing values sends missing
//! values to the RIGHT child at predict time.
//!
//! ferrolearn's grower (`hist_gradient_boosting.rs:480-538`) tries "NaN goes
//! left" FIRST and "NaN goes right" SECOND, keeping a strictly-greater gain
//! (`gain > curr.gain`). When `nan.count == 0` the two directions produce
//! identical gain, so the first-evaluated "NaN goes left" (`nan_goes_left:
//! true`) wins every tie. Therefore ferrolearn sends missing values LEFT — the
//! exact mirror of sklearn — whenever a node saw no missing during training.
//!
//! Observable effect: a NaN query against a model trained on all-finite data
//! yields the *opposite* class / mirror-image probabilities.
//!
//! Oracle (LIVE sklearn 1.5.2, NOT copied from ferrolearn; goal.md R-CHAR-3):
//! ```text
//! python3 -c "
//! import numpy as np
//! from sklearn.ensemble import HistGradientBoostingClassifier
//! X=np.array([[1.],[2.],[3.],[4.],[5.],[6.],[7.],[8.]])
//! y=np.array([0,0,0,0,1,1,1,1])
//! m=HistGradientBoostingClassifier(loss='log_loss',max_iter=20,max_leaf_nodes=7,
//!    min_samples_leaf=1,learning_rate=0.3,early_stopping=False,random_state=0).fit(X,y)
//! Xq=np.array([[np.nan]])
//! print(m.predict(Xq).tolist())        # -> [1]
//! print(m.predict_proba(Xq).tolist())  # -> [[0.0011045767792680072, 0.998895423220732]]
//! "
//! ```
//! ferrolearn currently returns predict `[0]` and proba
//! `[[0.998895..., 0.001104...]]` (mirror image). MUST FAIL.
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` deliberate.
//!
//! Tracking: #2280

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::HistGradientBoostingClassifier;
use ndarray::{Array1, Array2};

/// RED — `HistGradientBoostingClassifier` routes a NaN query LEFT when the
/// fitted nodes saw no missing during training; sklearn routes RIGHT.
///
/// Fixture: `X=[1..8]ᵀ` (all finite), `y=[0,0,0,0,1,1,1,1]`, NaN query.
/// Deterministic (`n_samples=8 <= 10000` ⇒ `early_stopping='auto'` is OFF, no
/// RNG / validation split). sklearn predicts class 1 for the NaN (missing goes
/// to the upper / class-1 child per `missing_go_to_left=False`,
/// `splitting.pyx:719-720`); ferrolearn predicts class 0 (missing goes left).
///
/// Tracking: #2280
#[test]
fn divergence_hgbc_missing_default_direction() {
    let x = Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = Array1::from(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);

    let model = HistGradientBoostingClassifier::<f64>::new()
        .with_n_estimators(20) // sklearn max_iter=20
        .with_max_leaf_nodes(Some(7))
        .with_min_samples_leaf(1)
        .with_learning_rate(0.3)
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();

    let xq = Array2::from_shape_vec((1, 1), vec![f64::NAN]).unwrap();

    // sklearn 1.5.2 live oracle (see module docstring). NOT copied from ferrolearn.
    let sklearn_predict: usize = 1;
    let sklearn_proba: [f64; 2] = [0.0011045767792680072, 0.998895423220732];

    let pred = fitted.predict(&xq).unwrap();
    let proba = fitted.predict_proba(&xq).unwrap();

    assert_eq!(
        pred[0], sklearn_predict,
        "HGBC NaN-query predict = {} != sklearn {}. The fitted nodes saw no \
         missing in training, so sklearn fixes missing_go_to_left=False \
         (splitting.pyx:719-720) -> NaN routes to the upper/class-1 child. \
         ferrolearn's grower keeps the FIRST-evaluated equal-gain direction \
         (NaN goes LEFT, hist_gradient_boosting.rs:480-538), routing NaN to \
         class 0 — the mirror of sklearn.",
        pred[0], sklearn_predict
    );

    for k in 0..2 {
        assert!(
            (proba[[0, k]] - sklearn_proba[k]).abs() < 1e-6,
            "HGBC NaN-query proba[{k}] = {} != sklearn {}. ferrolearn's missing \
             direction is the mirror of sklearn's (LEFT vs RIGHT) for nodes that \
             saw no missing in training, producing swapped class probabilities.",
            proba[[0, k]],
            sklearn_proba[k]
        );
    }
}
