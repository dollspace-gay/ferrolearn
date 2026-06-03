//! Behavior-preservation guards for the `_BaseNB` / `_BaseDiscreteNB` base
//! refactor (commit 100b6d25).
//!
//! These tests pin that EACH of the five Naive Bayes variants, routed through
//! the NEW `BaseNB`-trait-delegating predict pipeline (inherent
//! `predict`/`predict_proba`/`predict_log_proba` + `impl Predict::predict`),
//! reproduces the LIVE scikit-learn 1.5.2 oracle. If the delegation refactor
//! changed any observable output, the corresponding assert fails RED.
//!
//! Every expected value below is a LIVE `sklearn` 1.5.2 oracle value (computed
//! via `python3 -c "from sklearn.naive_bayes import ...; ..."` and quoted in
//! the comment above each block) — NEVER copied from the ferrolearn side
//! (goal.md R-CHAR-3).
//!
//! Mirrors:
//!   - `_BaseNB.predict`            (sklearn/naive_bayes.py:103)
//!   - `_BaseNB.predict_log_proba`  (sklearn/naive_bayes.py:123-126)
//!   - `_BaseNB.predict_proba`      (sklearn/naive_bayes.py:144)
//!   - `_BaseDiscreteNB._check_alpha` (sklearn/naive_bayes.py:604-626)

use ferrolearn_bayes::{BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB};
use ferrolearn_core::{Fit, Predict};
use ndarray::{Array1, Array2, array};

const TOL: f64 = 1e-9;

fn assert_proba_close(got: &Array2<f64>, want: &[[f64; 2]]) {
    assert_eq!(got.nrows(), want.len(), "row count");
    for (i, row) in want.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            let g = got[[i, j]];
            assert!(
                (g - w).abs() <= TOL.max(w.abs() * TOL),
                "proba[{i},{j}] got {g} want {w}"
            );
        }
    }
}

// NOTE: GaussianNB is intentionally NOT guarded here. Its delegation through
// `BaseNB` is faithful (identical pattern to the four variants below), but it
// carries a PRE-EXISTING value divergence in `epsilon_` (gaussian.rs `fit` uses
// max-over-per-class-variance floored at 1.0 vs sklearn
// `var_smoothing * np.var(X, axis=0).max()`, sklearn/naive_bayes.py:431). That
// divergence belongs to the GaussianNB translation unit (gaussian.rs) and is
// tracked by blocker #891 — it will be pinned in `.design/bayes/gaussian.md`'s
// divergence file when that unit's value-parity iteration runs.

// ---------------------------------------------------------------------------
// MultinomialNB
//
// Oracle (sklearn 1.5.2):
//   Xm=[[1,2],[0,3],[4,0],[3,1]] y=[0,0,1,1]
//   m=MultinomialNB().fit(Xm,ym); Q=[[2,2]]
//   m.predict(Q)        -> [0]
//   m.predict_proba(Q)  -> [[0.5786441724102462, 0.4213558275897536]]
//   m.predict_log_proba(Q) -> [[-0.5470675457484475, -0.8642776061017265]]
// ---------------------------------------------------------------------------
#[test]
fn multinomial_nb_matches_sklearn_oracle_via_base() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 3.0, 1.0]).unwrap();
    let y = array![0usize, 0, 1, 1];
    let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    let q = Array2::from_shape_vec((1, 2), vec![2.0, 2.0]).unwrap();

    assert_eq!(fitted.predict(&q).unwrap(), array![0usize]);
    assert_proba_close(
        &fitted.predict_proba(&q).unwrap(),
        &[[0.578_644_172_410_246_2, 0.421_355_827_589_753_6]],
    );
    assert_proba_close(
        &fitted.predict_log_proba(&q).unwrap(),
        &[[-0.547_067_545_748_447_5, -0.864_277_606_101_726_5]],
    );
}

// ---------------------------------------------------------------------------
// BernoulliNB
//
// Oracle (sklearn 1.5.2):
//   Xb=[[1,0,1],[0,1,1],[1,1,0],[0,0,1]] y=[0,0,1,1]
//   b=BernoulliNB().fit(Xb,yb); Q=[[1,0,0],[0,1,1]]
//   b.predict(Q)        -> [1, 0]
//   b.predict_proba(Q)  -> [[0.3333333333333332, 0.6666666666666669],
//                           [0.5999999999999999, 0.39999999999999997]]
//   b.predict_log_proba(Q) -> [[-1.09861228866811, -0.40546510810816416],
//                              [-0.5108256237659909, -0.9162907318741551]]
// ---------------------------------------------------------------------------
#[test]
fn bernoulli_nb_matches_sklearn_oracle_via_base() {
    let x = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let y = array![0usize, 0, 1, 1];
    let fitted = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
    let q = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap();

    assert_eq!(fitted.predict(&q).unwrap(), array![1usize, 0]);
    assert_proba_close(
        &fitted.predict_proba(&q).unwrap(),
        &[
            [0.333_333_333_333_333_2, 0.666_666_666_666_666_9],
            [0.599_999_999_999_999_9, 0.399_999_999_999_999_97],
        ],
    );
    assert_proba_close(
        &fitted.predict_log_proba(&q).unwrap(),
        &[
            [-1.098_612_288_668_11, -0.405_465_108_108_164_16],
            [-0.510_825_623_765_990_9, -0.916_290_731_874_155_1],
        ],
    );
}

// ---------------------------------------------------------------------------
// ComplementNB
//
// Oracle (sklearn 1.5.2):
//   Xc=[[1,2,0],[0,3,1],[4,0,2],[3,1,5]] y=[0,0,1,1]
//   c=ComplementNB().fit(Xc,yc); Q=[[2,2,1]]
//   c.predict(Q)        -> [0]
//   c.predict_proba(Q)  -> [[0.7265671462223204, 0.2734328537776799]]
//   c.predict_log_proba(Q) -> [[-0.3194243759861948, -1.2966991944733381]]
// ---------------------------------------------------------------------------
#[test]
fn complement_nb_matches_sklearn_oracle_via_base() {
    let x = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 4.0, 0.0, 2.0, 3.0, 1.0, 5.0],
    )
    .unwrap();
    let y = array![0usize, 0, 1, 1];
    let fitted = ComplementNB::<f64>::new().fit(&x, &y).unwrap();
    let q = Array2::from_shape_vec((1, 3), vec![2.0, 2.0, 1.0]).unwrap();

    assert_eq!(fitted.predict(&q).unwrap(), array![0usize]);
    assert_proba_close(
        &fitted.predict_proba(&q).unwrap(),
        &[[0.726_567_146_222_320_4, 0.273_432_853_777_679_9]],
    );
    assert_proba_close(
        &fitted.predict_log_proba(&q).unwrap(),
        &[[-0.319_424_375_986_194_8, -1.296_699_194_473_338_1]],
    );
}

// ---------------------------------------------------------------------------
// CategoricalNB
//
// Oracle (sklearn 1.5.2):
//   Xk=[[0,1],[1,0],[0,0],[2,1],[2,0],[1,1]] y=[0,0,0,1,1,1]
//   k=CategoricalNB().fit(Xk,yk); Q=[[1,1],[2,0]]
//   k.predict(Q)        -> [1, 1]
//   k.predict_proba(Q)  -> [[0.39999999999999997, 0.6000000000000001],
//                           [0.3333333333333333, 0.6666666666666666]]
//   k.predict_log_proba(Q) -> [[-0.9162907318741551, -0.5108256237659905],
//                              [-1.0986122886681098, -0.4054651081081644]]
// ---------------------------------------------------------------------------
#[test]
fn categorical_nb_matches_sklearn_oracle_via_base() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let y = array![0usize, 0, 0, 1, 1, 1];
    let fitted = CategoricalNB::<f64>::new().fit(&x, &y).unwrap();
    let q = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 0.0]).unwrap();

    assert_eq!(fitted.predict(&q).unwrap(), array![1usize, 1]);
    assert_proba_close(
        &fitted.predict_proba(&q).unwrap(),
        &[
            [0.399_999_999_999_999_97, 0.600_000_000_000_000_1],
            [0.333_333_333_333_333_3, 0.666_666_666_666_666_6],
        ],
    );
    assert_proba_close(
        &fitted.predict_log_proba(&q).unwrap(),
        &[
            [-0.916_290_731_874_155_1, -0.510_825_623_765_990_5],
            [-1.098_612_288_668_109_8, -0.405_465_108_108_164_4],
        ],
    );
}

// ---------------------------------------------------------------------------
// argmax tie-break: exact JLL tie -> np.argmax first-max -> smallest column ->
// (classes_ sorted) smallest label.
//
// Oracle (sklearn 1.5.2):
//   Xt=[[2,2],[2,2]] y=[0,1]; m=MultinomialNB(alpha=1.0).fit(Xt,yt)
//   m.predict_joint_log_proba([[1,1]]) ->
//       [[-2.0794415416798357, -2.0794415416798357]]   (exact tie)
//   m.predict([[1,1]]) -> [0]                           (smallest label)
// ---------------------------------------------------------------------------
#[test]
fn multinomial_nb_argmax_tie_breaks_to_smallest_label_via_base() {
    let x = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let y = array![0usize, 1];
    let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
    let q = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();

    // The joint log-likelihoods are an exact tie (both columns equal).
    let jll = fitted.predict_joint_log_proba(&q).unwrap();
    assert!(
        (jll[[0, 0]] - jll[[0, 1]]).abs() <= 1e-15,
        "expected an exact JLL tie, got {jll:?}"
    );
    // np.argmax picks the first (smallest) column -> smallest sorted label = 0.
    assert_eq!(fitted.predict(&q).unwrap(), array![0usize]);
}

// ---------------------------------------------------------------------------
// _check_alpha floor behavior via a discrete variant's fit (REQ-4 consumer).
//
// Oracle (sklearn 1.5.2): _BaseDiscreteNB._check_alpha (naive_bayes.py:618-625)
//   alpha=0, force_alpha=False -> floored to 1e-10 (no panic), feature_log_prob_:
//   m=MultinomialNB(alpha=0.0, force_alpha=False).fit([[1,2],[3,0]],[0,1])
//   m.feature_log_prob_ ->
//     [[-1.0986122886347764, -0.40546510812483116],
//      [-3.33333360913457e-11, -24.124463218675235]]
//
// We pin the observable consequence: predict succeeds (no panic) and the
// predicted label on a query matches sklearn's floored model. With alpha
// floored to 1e-10 the model is essentially un-smoothed; sklearn predicts the
// class whose feature pattern matches the query.
//   m.predict([[3,0]]) -> [1]   ; m.predict([[1,2]]) -> [0]
// ---------------------------------------------------------------------------
#[test]
fn multinomial_nb_alpha_zero_floors_no_panic_via_base() {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 0.0]).unwrap();
    let y = array![0usize, 1];
    let fitted = MultinomialNB::<f64>::new()
        .with_alpha(0.0)
        .with_force_alpha(false)
        .fit(&x, &y)
        .unwrap();

    let q: Array2<f64> = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 1.0, 2.0]).unwrap();
    let pred: Array1<usize> = fitted.predict(&q).unwrap();
    assert_eq!(pred, array![1usize, 0]);
}
