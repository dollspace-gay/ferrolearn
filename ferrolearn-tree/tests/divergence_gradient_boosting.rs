//! Divergence pins for `ferrolearn-tree/src/gradient_boosting.rs`
//! (`GradientBoostingRegressor` / `GradientBoostingClassifier`) against
//! scikit-learn 1.5.2.
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14), `sklearn/ensemble/_gb.py`:
//!   - `_update_terminal_regions` :129-264 — after fitting
//!     `DecisionTreeRegressor` to the negative gradient, REPLACE each leaf with
//!     the loss-optimal line-search value (`argmin_x loss(y, raw_old + x*value)`,
//!     :149-151), THEN `raw_prediction[:, k] += learning_rate *
//!     tree.value[:,0,0].take(terminal_regions)` :262-264.
//!       * `HalfSquaredError`: update is the IDENTITY — leaves stay the mean
//!         residual, only raw is updated :155-157/:186 (→ the L2 GREEN linchpin).
//!       * generic `else` (AbsoluteError / Huber) :241-247 —
//!         `update = loss.fit_intercept_only(y[idx] - raw[idx,k], sw)`; for
//!         `AbsoluteError` the WEIGHTED MEDIAN of the leaf residuals
//!         (`sklearn/_loss/loss.py:565-574`).
//!       * `HalfBinomialLoss` :191-206 — single Newton step
//!         `Σw·neg_g / Σw·p(1-p)`, `p = y - neg_g = expit(raw)`.
//!   - `__init__` defaults: GBR `loss='squared_error'`, `learning_rate=0.1`,
//!     `n_estimators=100`, `subsample=1.0`, `max_depth=3` (:2051-2097); GBC
//!     `loss='log_loss'`, same numerics (:1451-1495).
//!
//! Structural divergence (`.design/tree/gradient_boosting.md`, REQ-5/REQ-7):
//! ferrolearn's GBR `fit` loop and GBC `fit_binary`/`fit_multiclass` add the
//! regression tree's MEAN-residual leaf DIRECTLY (`f_vals[i] += lr*value`) for
//! EVERY loss — there is no `_update_terminal_regions` analog. The mean residual
//! IS the L2-optimal leaf (`HalfSquaredError` update is the identity), so
//! `loss='squared_error'` matches sklearn exactly (the GREEN linchpin below);
//! `absolute_error` (weighted median) and `log_loss` (Newton step) diverge from
//! round 1's leaf onward.
//!
//! All oracle values below come from a LIVE sklearn 1.5.2 run (the exact
//! `python3 -c "..."` invocation is quoted above each constant), NEVER copied
//! from the ferrolearn side (goal.md R-CHAR-3).
//!
//! `tests/*.rs` is anti-pattern-gate-exempt: `.unwrap()`/`assert!` are used
//! deliberately (no `panic!`/`unreachable!`).

use ferrolearn_core::traits::{Fit, Predict};
use ferrolearn_tree::{GradientBoostingClassifier, GradientBoostingRegressor, RegressionLoss};
use ndarray::{Array1, Array2, array};

/// `X = [[1],[2],...,[8]]`, the shared single-feature regression fixture.
fn x_1d() -> Array2<f64> {
    Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()
}

// ===========================================================================
// GREEN — L2 GBR end-to-end parity (REQ-4 linchpin). MUST PASS now and stay
// green after the builder lands the terminal-region updates.
// ===========================================================================

/// GREEN (linchpin) — `GradientBoostingRegressor(loss='squared_error')`
/// `predict` matches sklearn array-by-array. The L2 `_update_terminal_regions`
/// is the identity (`_gb.py:155-157`/:186), so ferrolearn's mean-residual leaf
/// equals sklearn's optimal leaf, validating the whole init→residual→tree→
/// shrinkage→predict framework. Deterministic at `subsample=1.0`.
///
/// Live oracle (sklearn 1.5.2; deterministic, no RNG at subsample=1.0):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import GradientBoostingRegressor
/// X=np.arange(1,9.).reshape(-1,1); y=np.array([1,1,1,1,5,5,5,5.])
/// m=GradientBoostingRegressor(loss='squared_error',n_estimators=5,
///     learning_rate=0.1,max_depth=2,random_state=0,subsample=1.0).fit(X,y)
/// print(np.round(m.predict(X),12).tolist())
/// "
/// # -> [2.18098, 2.18098, 2.18098, 2.18098, 3.81902, 3.81902, 3.81902, 3.81902]
/// ```
#[test]
fn green_l2_gbr_end_to_end_parity() {
    let x = x_1d();
    let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

    let model = GradientBoostingRegressor::<f64>::new()
        .with_loss(RegressionLoss::LeastSquares)
        .with_n_estimators(5)
        .with_learning_rate(0.1)
        .with_max_depth(Some(2))
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 8] = [
        2.18098, 2.18098, 2.18098, 2.18098, 3.81902, 3.81902, 3.81902, 3.81902,
    ];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-6,
            "L2 GBR predict[{i}] = {p} != sklearn {e} (loss='squared_error' \
             terminal-region update is the identity, _gb.py:155-157; this is the \
             REQ-4 linchpin guard — it must NOT regress when the builder adds the \
             LAD/Newton terminal-region updates)."
        );
    }
}

// ===========================================================================
// GREEN — constructor numeric defaults (REQ-1) match sklearn get_params().
// ===========================================================================

/// GREEN — GBR/GBC constructor defaults match `get_params()` (sklearn 1.5.2)
/// for the exposed numeric params.
///
/// Live oracle:
/// ```text
/// python3 -c "
/// from sklearn.ensemble import GradientBoostingRegressor as R, GradientBoostingClassifier as C
/// print({k:R().get_params()[k] for k in ['n_estimators','learning_rate','max_depth','subsample']})
/// print({k:C().get_params()[k] for k in ['n_estimators','learning_rate','max_depth','subsample']})
/// "
/// # -> {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0}  (both)
/// ```
#[test]
fn green_constructor_defaults_match_sklearn() {
    let r = GradientBoostingRegressor::<f64>::default();
    assert_eq!(r.n_estimators, 100, "sklearn GBR default n_estimators=100");
    assert!(
        (r.learning_rate - 0.1).abs() < 1e-15,
        "sklearn GBR default learning_rate=0.1"
    );
    assert_eq!(r.max_depth, Some(3), "sklearn GBR default max_depth=3");
    assert!(
        (r.subsample - 1.0).abs() < 1e-15,
        "sklearn GBR default subsample=1.0"
    );

    let c = GradientBoostingClassifier::<f64>::default();
    assert_eq!(c.n_estimators, 100, "sklearn GBC default n_estimators=100");
    assert!(
        (c.learning_rate - 0.1).abs() < 1e-15,
        "sklearn GBC default learning_rate=0.1"
    );
    assert_eq!(c.max_depth, Some(3), "sklearn GBC default max_depth=3");
    assert!(
        (c.subsample - 1.0).abs() < 1e-15,
        "sklearn GBC default subsample=1.0"
    );
}

// ===========================================================================
// GREEN — deterministic reproducibility at subsample=1.0 (REQ-9/AC-8).
// ===========================================================================

/// GREEN — two `fit` calls with identical params + the same seed produce
/// IDENTICAL `predict` (subsample=1.0 → deterministic, no RNG boundary).
#[test]
fn green_gbr_reproducibility() {
    let x = x_1d();
    let y = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];

    let model = GradientBoostingRegressor::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(2))
        .with_random_state(123);
    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    assert_eq!(
        f1.predict(&x).unwrap().to_vec(),
        f2.predict(&x).unwrap().to_vec(),
        "GBR predict must be identical across fits at subsample=1.0 (deterministic)"
    );
}

/// GREEN — GBC reproducibility (subsample=1.0, deterministic).
#[test]
fn green_gbc_reproducibility() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let y: Array1<usize> = array![0, 0, 0, 0, 1, 1, 1, 1];

    let model = GradientBoostingClassifier::<f64>::new()
        .with_n_estimators(10)
        .with_max_depth(Some(2))
        .with_random_state(7);
    let f1 = model.fit(&x, &y).unwrap();
    let f2 = model.fit(&x, &y).unwrap();

    assert_eq!(
        f1.predict(&x).unwrap().to_vec(),
        f2.predict(&x).unwrap().to_vec(),
        "GBC predict must be identical across fits at subsample=1.0 (deterministic)"
    );
}

// ===========================================================================
// RED (REQ-5, blocker #734) — LAD weighted-MEDIAN terminal-region update.
// ===========================================================================

/// RED (HEADLINE) — `GradientBoostingRegressor(loss='absolute_error')` `predict`
/// must equal sklearn's. The fixture is SKEWED within each split leaf
/// (`y=[0,0,0,10,1,1,1,20]`) so the leaf MEAN differs sharply from the leaf
/// MEDIAN. sklearn replaces each leaf with the WEIGHTED MEDIAN of the leaf's
/// residuals `y[idx]-raw[idx]` (`_update_terminal_regions` generic `else`
/// `_gb.py:241-247` → `AbsoluteError.fit_intercept_only`,
/// `sklearn/_loss/loss.py:565-574`). ferrolearn instead adds the regression
/// tree's L2-mean-residual leaf directly (`f_vals[i] += lr*value`, GBR `fit`
/// loop) — there is NO `_update_terminal_regions` analog, so it diverges from
/// round 1.
///
/// Live oracle (sklearn 1.5.2; deterministic at subsample=1.0):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import GradientBoostingRegressor
/// X=np.arange(1,9.).reshape(-1,1); y=np.array([0,0,0,10,1,1,1,20.])
/// m=GradientBoostingRegressor(loss='absolute_error',n_estimators=3,
///     learning_rate=0.1,max_depth=2,random_state=0,subsample=1.0).fit(X,y)
/// print(np.round(m.predict(X),12).tolist())
/// "
/// # -> [0.729, 0.729, 0.729, 1.0, 1.0, 1.0, 1.0, 1.0]
/// ```
/// The init is `median(y) == 1.0` (matches); the divergence is entirely the
/// missing per-leaf median line-search. ferrolearn's mean-leaf output is
/// squared-error-shaped (pulled toward the 10/20 outliers) — it does NOT match
/// the clean L1-median family above. MUST currently FAIL.
///
/// Tracking: #734
#[test]
fn divergence_lad_median_terminal_region() {
    let x = x_1d();
    let y = array![0.0, 0.0, 0.0, 10.0, 1.0, 1.0, 1.0, 20.0];

    let model = GradientBoostingRegressor::<f64>::new()
        .with_loss(RegressionLoss::Lad)
        .with_n_estimators(3)
        .with_learning_rate(0.1)
        .with_max_depth(Some(2))
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 8] = [0.729, 0.729, 0.729, 1.0, 1.0, 1.0, 1.0, 1.0];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-6,
            "LAD GBR predict[{i}] = {p} != sklearn {e}. sklearn replaces each leaf \
             with the WEIGHTED MEDIAN of the leaf residuals (_gb.py:241-247 → \
             loss.py:565-574); ferrolearn adds the L2-MEAN-residual leaf directly \
             (no _update_terminal_regions). Builder must group each leaf's samples, \
             set the leaf value to median(y[idx]-raw[idx]), then apply lr*leaf."
        );
    }
}

// ===========================================================================
// RED (REQ-7, blocker #735) — LogLoss binary Newton terminal-region update.
// ===========================================================================

/// RED (HEADLINE) — `GradientBoostingClassifier` (binary) `predict_proba[:,1]`
/// must equal sklearn's. sklearn replaces each leaf with the single
/// Newton-Raphson step `Σw·(y-p) / Σw·p(1-p)` (`p = y - neg_g = expit(raw)`,
/// `_update_terminal_regions` `HalfBinomialLoss` branch `_gb.py:191-206`), then
/// `raw += lr*leaf` (:262-264). ferrolearn's GBC `fit_binary` adds the regression
/// tree's MEAN-residual leaf directly (`f_vals[i] += lr*value`) — no Newton
/// update — so the cumulative log-odds (and hence `predict_proba`) diverge from
/// round 1.
///
/// Live oracle (sklearn 1.5.2; deterministic at subsample=1.0):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import GradientBoostingClassifier
/// X=np.array([[1,2],[2,3],[3,3],[4,4],[5,6],[6,7],[7,8],[8,9.]])
/// y=np.array([0,0,0,0,1,1,1,1])
/// m=GradientBoostingClassifier(n_estimators=5,learning_rate=0.1,max_depth=2,
///     random_state=0,subsample=1.0).fit(X,y)
/// print(np.round(m.predict_proba(X)[:,1],12).tolist())
/// print(np.round(m.decision_function(X).ravel(),12).tolist())
/// "
/// # predict_proba[:,1] -> [0.297947595479 x4, 0.702052404521 x4]
/// # decision_function  -> [-0.857090434608 x4, 0.857090434608 x4]
/// ```
/// The init log-odds is `log(0.5/0.5) == 0.0` (matches); the divergence is
/// entirely the missing per-leaf Newton step. ferrolearn's mean-leaf raw scores
/// produce a different probability family. MUST currently FAIL.
///
/// Tracking: #735
#[test]
fn divergence_logloss_newton_terminal_region_binary() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
        ],
    )
    .unwrap();
    let y: Array1<usize> = array![0, 0, 0, 0, 1, 1, 1, 1];

    let model = GradientBoostingClassifier::<f64>::new()
        .with_n_estimators(5)
        .with_learning_rate(0.1)
        .with_max_depth(Some(2))
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let proba = fitted.predict_proba(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_proba1: [f64; 8] = [
        0.297947595479,
        0.297947595479,
        0.297947595479,
        0.297947595479,
        0.702052404521,
        0.702052404521,
        0.702052404521,
        0.702052404521,
    ];

    for (i, &e) in sklearn_proba1.iter().enumerate() {
        let p = proba[[i, 1]];
        assert!(
            (p - e).abs() < 1e-6,
            "GBC predict_proba[{i}][1] = {p} != sklearn {e}. sklearn replaces each \
             leaf with the Newton step Σw(y-p)/Σw·p(1-p) (_gb.py:191-206); \
             ferrolearn adds the MEAN-residual leaf directly (no Newton update). \
             Builder must group each leaf's samples and set the leaf value to the \
             Newton step before applying lr*leaf."
        );
    }
}

// ===========================================================================
// RED (NEW divergence, blocker #737) — LAD leaf median uses np.median
// (mean of the two middle values for even counts) instead of sklearn's
// `_weighted_percentile(residuals, ones, 50)` LOWER percentile (a single
// sorted element).
// ===========================================================================

/// RED (NEW) — `GradientBoostingRegressor(loss='absolute_error')` leaf value
/// for an EVEN-count leaf diverges.
///
/// `gradient_boosting.rs::lad_leaf_value` computes the leaf value as the numpy
/// median of the leaf residuals — for an even count it averages the two middle
/// sorted values (`(diffs[n/2-1] + diffs[n/2]) / 2`). sklearn does NOT use
/// `np.median` here: in `_update_terminal_regions` (`sklearn/ensemble/_gb.py:255`)
/// `sw = None if sample_weight is None else sample_weight[indices]`, and `fit`
/// always passes `sample_weight = _check_sample_weight(None, X) = np.ones(n)`
/// (NEVER `None`). So `AbsoluteError.fit_intercept_only`
/// (`sklearn/_loss/loss.py:571-574`) takes the
/// `_weighted_percentile(y_true, sample_weight, 50)` branch — the LOWER weighted
/// percentile (`sklearn/utils/stats.py:53-68`, `np.searchsorted` left), which
/// returns a SINGLE sorted element (the lower-middle for an even count), never an
/// average.
///
/// Fixture: `X=[[1]..[8]]`, `y=[0,0,0,0,10,20,30,41]`, `n_estimators=1,
/// learning_rate=1.0, max_depth=2`. `init = median(y) = 5`. The sign-gradient
/// tree splits at 4.5 → right leaf = samples {5..8} with residuals
/// `y[idx]-init = [5,15,25,36]` (even count, distinct middles). ferrolearn's
/// round-1 tree STRUCTURE is identical to sklearn's (verified), so the ONLY
/// divergence is the leaf-value rule:
///   * sklearn leaf = `_weighted_percentile([5,15,25,36], ones, 50) = 15.0`
///     (lower-middle) → predict = `5 + 1.0*15 = 20.0`.
///   * ferrolearn leaf = `np.median([5,15,25,36]) = (15+25)/2 = 20.0`
///     → predict = `5 + 1.0*20 = 25.0`.
///
/// Live oracle (sklearn 1.5.2; deterministic at subsample=1.0):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import GradientBoostingRegressor
/// X=np.arange(1,9.).reshape(-1,1); y=np.array([0,0,0,0,10,20,30,41.])
/// m=GradientBoostingRegressor(loss='absolute_error',n_estimators=1,
///     learning_rate=1.0,max_depth=2,random_state=0,subsample=1.0).fit(X,y)
/// print(m.predict(X).tolist())
/// "
/// # -> [0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0]
/// # cross-check the leaf rule:
/// python3 -c "import numpy as np; from sklearn.utils.stats import _weighted_percentile; \
///   print(_weighted_percentile(np.array([5.,15.,25.,36.]), np.ones(4), 50))"  # -> 15.0
/// ```
/// ferrolearn currently returns 25.0 for samples 5..8. MUST FAIL.
///
/// Tracking: #737
#[test]
fn divergence_lad_leaf_lower_percentile_even_count() {
    let x = x_1d();
    let y = array![0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 41.0];

    let model = GradientBoostingRegressor::<f64>::new()
        .with_loss(RegressionLoss::Lad)
        .with_n_estimators(1)
        .with_learning_rate(1.0)
        .with_max_depth(Some(2))
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-9,
            "LAD GBR predict[{i}] = {p} != sklearn {e}. sklearn's AbsoluteError \
             leaf-update uses _weighted_percentile(residuals, ones=sample_weight, \
             50) = the LOWER weighted percentile (a single sorted element, the \
             lower-middle for even counts) — `fit` always passes np.ones, never \
             None (_gb.py:255 + loss.py:571-574). ferrolearn's lad_leaf_value uses \
             np.median (averages the two middle values for even counts), so on the \
             even-count right leaf {{5,15,25,36}} it returns mean-median 20.0 \
             instead of lower-median 15.0."
        );
    }
}

// ===========================================================================
// RED (NEW divergence, blocker #738) — Huber leaf median tie. The
// median fed into the clipped-mean term uses np.median (mean of two middles)
// instead of `_weighted_percentile(.,ones,50)` (lower percentile), so the
// Huber leaf value (and predict) diverges on even-count leaves.
// ===========================================================================

/// RED (NEW) — `GradientBoostingRegressor(loss='huber')` leaf value diverges on
/// an even-count leaf because its internal median uses the np.median tie
/// (average of the two middle values) instead of sklearn's lower percentile.
///
/// `HuberLoss.fit_intercept_only` (`sklearn/_loss/loss.py:704-710`) with
/// `sample_weight` (= np.ones, never None — see #737 / `_gb.py:255`) computes
/// `median = _weighted_percentile(y_true, sample_weight, 50)` (LOWER percentile),
/// then `median + np.average(sign(d-median)*min(delta,|d-median|))`. ferrolearn's
/// `huber_leaf_value` instead uses `(sorted[n/2-1]+sorted[n/2])/2` for the
/// median, so on an even-count leaf both the median AND the resulting clipped
/// mean diverge.
///
/// Fixture: `X=[[1]..[8]]`, `y=[0,0,0,0,10,20,30,41]`, `alpha=0.5`,
/// `n_estimators=1, learning_rate=1.0, max_depth=2`. `init = median(y) = 5`,
/// stage `delta = _weighted_percentile(|y-init|, ones, 50) = 5.0`. Right leaf
/// residuals `[5,15,25,36]`:
///   * sklearn: median = 15 (lower), term = sign(d-15)*min(5,|d-15|) =
///     `[-5,0,5,5]`, mean = 1.25 → leaf = 16.25 → predict = `5 + 16.25 = 21.25`.
///   * ferrolearn: median = (15+25)/2 = 20, term = `[-5,-5,5,5]`, mean = 0
///     → leaf = 20 → predict = `5 + 20 = 25.0`.
///
/// Live oracle (sklearn 1.5.2; deterministic at subsample=1.0):
/// ```text
/// python3 -c "
/// import numpy as np
/// from sklearn.ensemble import GradientBoostingRegressor
/// X=np.arange(1,9.).reshape(-1,1); y=np.array([0,0,0,0,10,20,30,41.])
/// m=GradientBoostingRegressor(loss='huber',alpha=0.5,n_estimators=1,
///     learning_rate=1.0,max_depth=2,random_state=0,subsample=1.0).fit(X,y)
/// print([round(v,12) for v in m.predict(X).tolist()])
/// "
/// # -> [0.0, 0.0, 0.0, 0.0, 21.25, 21.25, 21.25, 21.25]
/// ```
/// ferrolearn currently returns 25.0 for samples 5..8. MUST FAIL.
///
/// Tracking: #738
#[test]
fn divergence_huber_leaf_lower_percentile_median() {
    let x = x_1d();
    let y = array![0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 41.0];

    let model = GradientBoostingRegressor::<f64>::new()
        .with_loss(RegressionLoss::Huber)
        .with_huber_alpha(0.5)
        .with_n_estimators(1)
        .with_learning_rate(1.0)
        .with_max_depth(Some(2))
        .with_random_state(0);
    let fitted = model.fit(&x, &y).unwrap();
    let preds = fitted.predict(&x).unwrap();

    // sklearn 1.5.2 live oracle (see docstring). NOT copied from ferrolearn.
    let sklearn_predict: [f64; 8] = [0.0, 0.0, 0.0, 0.0, 21.25, 21.25, 21.25, 21.25];

    for (i, (&p, &e)) in preds.iter().zip(sklearn_predict.iter()).enumerate() {
        assert!(
            (p - e).abs() < 1e-9,
            "Huber GBR predict[{i}] = {p} != sklearn {e}. sklearn's \
             HuberLoss.fit_intercept_only median is _weighted_percentile(., ones, \
             50) = the LOWER percentile (loss.py:704-710; sample_weight is np.ones, \
             never None). ferrolearn's huber_leaf_value averages the two middle \
             residuals for the median, so on the even-count leaf {{5,15,25,36}} \
             (delta=5) it returns 20.0 (leaf) -> predict 25.0 instead of \
             median 15 -> leaf 16.25 -> predict 21.25."
        );
    }
}
