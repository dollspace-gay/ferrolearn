//! VALUE-parity divergence: `IterativeImputer` transform replay does NOT
//! reproduce the multi-round iterative result when `n_iter_ > 1`.
//!
//! ROOT CAUSE (ferrolearn `iterative_imputer.rs`):
//! `Fit::fit` records the imputation sequence with `imputation_sequence =
//! round_steps;` — it OVERWRITES the accumulator every round, so only the LAST
//! round's fitted `BayesianRidge` models survive in `FittedIterativeImputer`.
//! `Transform::transform` then replays just that single round starting from the
//! plain initial mean fill. sklearn's `imputation_sequence_` instead APPENDS
//! every round's triplets (`sklearn/impute/_iterative.py:801`
//! `self.imputation_sequence_.append(estimator_triplet)`, inside the
//! `for self.n_iter_ in range(1, self.max_iter + 1)` loop at `:781`), and
//! `transform` replays ALL `n_features_with_missing_ * n_iter_` of them
//! (`_iterative.py:865-873`). For the 10x5 fixture below `len(imputation_sequence_)
//! == 50` (5 features x 10 rounds); ferrolearn keeps only 5. Replaying one round
//! from the mean fill only coincides with the full iterative answer when the
//! imputation has CONVERGED (Xt stabilised); when `max_iter` is reached WITHOUT
//! convergence the one-round replay diverges by O(1), far beyond the ~1e-6 budget.
//!
//! Because `IterativeImputer::fit_transform` is `fit(x)` then `transform(x)`
//! (ferrolearn `fn fit_transform`), the headline `fit_transform` output is wrong
//! for any fixture that does not converge inside `max_iter`.
//!
//! All expected values are the LIVE sklearn 1.5.2 oracle (EXPERIMENTAL gate,
//! `random_state=0`, run from /tmp), captured in each test — never copied from
//! ferrolearn (R-CHAR-3).

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::iterative_imputer::{InitialStrategy, IterativeImputer};
use ndarray::{Array2, array};

const N: f64 = f64::NAN;
const TOL: f64 = 1e-6;

/// The 10x5 fixture (numpy `RandomState(42).rand(10,5)*10` with a
/// `RandomState(42).rand(10,5)<0.25` missing mask) that does NOT converge in
/// `max_iter=10` (sklearn emits `ConvergenceWarning`, `n_iter_ == 10`,
/// `len(imputation_sequence_) == 50`).
fn x_e() -> Array2<f64> {
    array![
        [
            3.745401188473625,
            9.50714306409916,
            7.319939418114051,
            5.986584841970366,
            1.5601864044243652
        ],
        [1.5599452033620265, N, N, N, 7.080725777960454],
        [
            0.20584494295802447,
            9.699098521619943,
            8.324426408004218,
            2.1233911067827616,
            1.8182496720710062
        ],
        [
            1.8340450985343382,
            N,
            5.247564316322379,
            N,
            2.9122914019804194
        ],
        [6.118528947223795, N, N, 3.663618432936917, 4.56069984217036],
        [
            7.851759613930136,
            1.9967378215835974,
            N,
            5.924145688620425,
            N
        ],
        [
            6.075448519014383,
            1.7052412368729153,
            0.6505159298527952,
            N,
            9.656320330745594
        ],
        [
            8.08397348116461,
            3.0461376917337066,
            0.9767211400638387,
            6.842330265121569,
            4.4015249373960135
        ],
        [
            N,
            4.951769101112702,
            0.34388521115218396,
            9.093204020787821,
            2.587799816000169
        ],
        [
            6.62522284353982,
            3.1171107608941098,
            5.200680211778108,
            N,
            N
        ]
    ]
}

/// Divergence: `IterativeImputer::fit_transform` (max_iter=10) on a fixture that
/// does NOT converge in 10 rounds diverges from
/// `sklearn/impute/_iterative.py:781,801,829` by O(1) — the transform replays a
/// single (last) round of models from the mean fill instead of all 10 rounds.
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, `random_state=0`, max_iter=10,
/// tol=1e-3) — `imp.n_iter_ == 10`:
///   X = x_e()
///   IterativeImputer(random_state=0).fit_transform(X)[1,1]  = 6.327769971432306
///   ...[1,3] = 0.6918179357548213
///   ...[5,2] = -1.3863798405691896
///   ...[5,4] = 6.809794658158644
///   ...[8,0] = 7.706961326628862
///
/// ferrolearn returns (one-round replay from mean fill):
///   [1,1] ~= 12.44, [1,3] ~= 5.61, [5,2] ~= 1.06, [5,4] ~= 3.91 — all off by O(1).
///
/// Tracking: #2331
#[test]
fn divergence_transform_replay_nonconverged_full_matrix() {
    let x = x_e();
    let out = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean)
        .fit_transform(&x)
        .expect("fit_transform");

    // sklearn 1.5.2 oracle imputed cells (n_iter_ == 10, did NOT converge).
    let sk: &[(usize, usize, f64)] = &[
        (1, 1, 6.327769971432306),
        (1, 3, 0.6918179357548213),
        (5, 2, -1.3863798405691896),
        (5, 4, 6.809794658158644),
        (8, 0, 7.706961326628862),
    ];
    for &(i, j, want) in sk {
        assert!(
            (out[[i, j]] - want).abs() < TOL,
            "transform-replay: imputed [{i},{j}] = {} != sklearn {want} (diff {:e})",
            out[[i, j]],
            (out[[i, j]] - want).abs()
        );
    }
}

/// Divergence: `FittedIterativeImputer::transform` does not reproduce the value
/// that `fit` itself computed internally, even on the SAME training matrix —
/// because only the last round of models is stored. Here `max_iter=2` and the
/// fixture has not converged after 2 rounds, so the 2-round fit result differs
/// from the 1-round replay.
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, `random_state=0`, max_iter=2,
/// tol=1e-3): `IterativeImputer(...).fit_transform(X)`:
///   [1,1] = 6.561173261626361
///   [5,4] = 5.233791637707
///   [9,4] = 5.053976247025007
///
/// ferrolearn (one-round replay): [1,1] ~= 10.11, [5,4] ~= 4.02, [9,4] ~= 3.37.
///
/// Tracking: #2331
#[test]
fn divergence_transform_replay_two_round() {
    let x = x_e();
    let imp = IterativeImputer::<f64>::new(2, 1e-3, InitialStrategy::Mean);
    let out = imp.fit_transform(&x).expect("fit_transform");

    let sk_11 = 6.561173261626361_f64;
    let sk_54 = 5.233791637707_f64;
    let sk_94 = 5.053976247025007_f64;

    assert!(
        (out[[1, 1]] - sk_11).abs() < TOL,
        "two-round replay: [1,1] = {} != sklearn {sk_11}",
        out[[1, 1]]
    );
    assert!(
        (out[[5, 4]] - sk_54).abs() < TOL,
        "two-round replay: [5,4] = {} != sklearn {sk_54}",
        out[[5, 4]]
    );
    assert!(
        (out[[9, 4]] - sk_94).abs() < TOL,
        "two-round replay: [9,4] = {} != sklearn {sk_94}",
        out[[9, 4]]
    );
}

/// Divergence: inductive `transform` of a NEW matrix replays only the last round
/// when the TRAINING fixture did not converge. sklearn replays the full stored
/// `imputation_sequence_` (50 triplets for x_e).
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, `random_state=0`, max_iter=10,
/// tol=1e-3): `imp.fit(x_e())` (n_iter_==10), then
///   Xte = [[1, nan, 3, nan, 5],
///          [nan, 2, nan, 4, nan],
///          [6, 7, 8, 9, 10]]
///   imp.transform(Xte) =
///     [[1.0,               8.193035979116987, 3.0,
///       2.955533155151201, 5.0],
///      [7.580180639573371, 2.0,               2.463653169545701,
///       4.0,               6.272562789808237],
///      [6.0, 7.0, 8.0, 9.0, 10.0]]
///
/// Tracking: #2331
#[test]
fn divergence_transform_replay_new_matrix() {
    let fitted = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean)
        .fit(&x_e(), &())
        .expect("fit");
    let xte = array![
        [1.0, N, 3.0, N, 5.0],
        [N, 2.0, N, 4.0, N],
        [6.0, 7.0, 8.0, 9.0, 10.0]
    ];
    let out = fitted.transform(&xte).expect("transform");

    let sk: &[(usize, usize, f64)] = &[
        (0, 1, 8.193035979116987),
        (0, 3, 2.955533155151201),
        (1, 0, 7.580180639573371),
        (1, 2, 2.463653169545701),
        (1, 4, 6.272562789808237),
    ];
    for &(i, j, want) in sk {
        assert!(
            (out[[i, j]] - want).abs() < TOL,
            "new-matrix replay: imputed [{i},{j}] = {} != sklearn {want} (diff {:e})",
            out[[i, j]],
            (out[[i, j]] - want).abs()
        );
    }
}
