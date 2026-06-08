//! VALUE-parity divergence suite for `IterativeImputer` against scikit-learn
//! 1.5.2 `class IterativeImputer` (`sklearn/impute/_iterative.py`).
//!
//! EXPERIMENTAL upstream — imported in oracles via
//! `from sklearn.experimental import enable_iterative_imputer`.
//!
//! These tests pin the ITERATED IMPUTED-VALUE divergences that the existing
//! `divergence_iterative_imputer.rs` suite explicitly carved out (its REQ-4
//! note: "ferrolearn Ridge(alpha=1) != sklearn default BayesianRidge ... NOT
//! asserted here"). Per goal.md R-DEFER-2/3/5 there is no third "carve-out"
//! status: every value divergence on `main` is real work we owe and must be
//! pinned with a FAILING test against the LIVE sklearn oracle.
//!
//! ROOT CAUSE: `iterative_imputer.rs` hard-codes its own closed-form
//! `ridge_fit` with `alpha = F::one()` (`fn fit`, `let alpha = F::one()`),
//! whereas sklearn's default round-robin estimator is `BayesianRidge`
//! (`_iterative.py:732-735`: `if self.estimator is None: ... self._estimator =
//! BayesianRidge()`). ferrolearn-preprocess ALREADY depends on ferrolearn-linear
//! (which ships `BayesianRidge` in `bayesian_ridge.rs`), so the divergence is
//! fixable by routing the round-robin step through the real BayesianRidge.
//! Secondary contributors: default imputation_order (sklearn 'ascending' vs
//! ferrolearn column/'roman'), inf-norm tol vs ferrolearn L2-relative tol, and
//! no min/max clip.
//!
//! All expected values are the LIVE sklearn 1.5.2 oracle (run from /tmp with the
//! experimental gate), captured in each test's doc comment — never copied from
//! ferrolearn (R-CHAR-3).

use ferrolearn_core::traits::FitTransform;
use ferrolearn_preprocess::iterative_imputer::{InitialStrategy, IterativeImputer};
use ndarray::{Array2, array};

const TOL: f64 = 1e-6;

// ===========================================================================
// DIV-VAL-1 — core round-robin imputed VALUES (small fixture)
// ===========================================================================

/// Divergence DIV-VAL-1: `IterativeImputer::fit_transform` diverges from
/// `sklearn/impute/_iterative.py:732-735` (default estimator `BayesianRidge()`)
/// + `:454` (`imputed_values = estimator.predict(X_test)`).
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, run from /tmp):
///   X = [[1,2],[3,nan],[nan,6]]
///   IterativeImputer(random_state=0).fit_transform(X) =
///     [[1.0, 2.0],
///      [3.0, 4.000002999996018],
///      [4.999994000015464, 6.0]]
///   n_iter_ = 2
///
/// sklearn imputes [1,1] -> 4.000002999996018 and [2,0] -> 4.999994000015464
/// (BayesianRidge effectively recovers the y=2x relation).
/// ferrolearn's Ridge(alpha=1) round-robin imputes [1,1] ~= 3.346 and
/// [2,0] ~= 5.034 — both differ from sklearn by >> 1e-6.
///
/// Tracking #1405 (root: ferrolearn-linear BayesianRidge not wired in).
#[test]
fn divergence_round_robin_values_small() {
    let imp = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![[1.0, 2.0], [3.0, f64::NAN], [f64::NAN, 6.0]];
    let out = imp.fit_transform(&x).expect("fit_transform");

    // sklearn BayesianRidge oracle values.
    let sk_11 = 4.000002999996018_f64;
    let sk_20 = 4.999994000015464_f64;

    assert!(
        (out[[1, 1]] - sk_11).abs() < TOL,
        "DIV-VAL-1: imputed [1,1] = {} != sklearn BayesianRidge {sk_11}",
        out[[1, 1]]
    );
    assert!(
        (out[[2, 0]] - sk_20).abs() < TOL,
        "DIV-VAL-1: imputed [2,0] = {} != sklearn BayesianRidge {sk_20}",
        out[[2, 0]]
    );
}

// ===========================================================================
// DIV-VAL-2 — round-robin imputed VALUES + ascending order (3-feature fixture)
// ===========================================================================

/// Divergence DIV-VAL-2: `IterativeImputer::fit_transform` diverges from
/// `sklearn/impute/_iterative.py:732-735` (default `BayesianRidge()`) and
/// `:533-535`/`:769` (default `imputation_order='ascending'`, fewest-missing
/// first) for a multi-feature fixture.
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, run from /tmp):
///   X = [[1,2,3],[2,nan,5],[3,6,nan],[4,8,9],[nan,10,11],[6,12,13]]
///   IterativeImputer(random_state=0, max_iter=10, tol=1e-3).fit_transform(X) =
///     [[1.0, 2.0, 3.0],
///      [2.0, 4.000010357521532, 5.0],
///      [3.0, 6.0, 6.999997942723164],
///      [4.0, 8.0, 9.0],
///      [4.999999113051894, 10.0, 11.0],
///      [6.0, 12.0, 13.0]]
///   n_iter_ = 3
///
/// ferrolearn (Ridge(alpha=1), roman order, L2-relative tol) imputes:
///   [1,1] ~= 4.047, [2,2] ~= 7.005, [4,0] ~= 4.984 — all differ >> 1e-6.
///
/// Tracking #1405 (estimator) and #1407 (default order ascending vs roman).
#[test]
fn divergence_round_robin_values_three_features() {
    let imp = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean);
    let x = array![
        [1.0, 2.0, 3.0],
        [2.0, f64::NAN, 5.0],
        [3.0, 6.0, f64::NAN],
        [4.0, 8.0, 9.0],
        [f64::NAN, 10.0, 11.0],
        [6.0, 12.0, 13.0],
    ];
    let out = imp.fit_transform(&x).expect("fit_transform");

    let sk_11 = 4.000010357521532_f64;
    let sk_22 = 6.999997942723164_f64;
    let sk_40 = 4.999999113051894_f64;

    assert!(
        (out[[1, 1]] - sk_11).abs() < TOL,
        "DIV-VAL-2: imputed [1,1] = {} != sklearn {sk_11}",
        out[[1, 1]]
    );
    assert!(
        (out[[2, 2]] - sk_22).abs() < TOL,
        "DIV-VAL-2: imputed [2,2] = {} != sklearn {sk_22}",
        out[[2, 2]]
    );
    assert!(
        (out[[4, 0]] - sk_40).abs() < TOL,
        "DIV-VAL-2: imputed [4,0] = {} != sklearn {sk_40}",
        out[[4, 0]]
    );
}

// ===========================================================================
// DIV-VAL-3 — min_value / max_value clip not expressible / not applied
// ===========================================================================

/// Divergence DIV-VAL-3: ferrolearn `IterativeImputer` has NO `min_value`/
/// `max_value` parameter and never clips, diverging from
/// `sklearn/impute/_iterative.py:455-457`
/// (`imputed_values = np.clip(imputed_values, self._min_value[feat_idx],
/// self._max_value[feat_idx])`).
///
/// Oracle (sklearn 1.5.2, EXPERIMENTAL gate, run from /tmp):
///   X = [[1,2,3],[2,nan,5],[3,6,nan],[4,8,9],[nan,10,11],[6,12,13]]
///   IterativeImputer(random_state=0, max_iter=10, tol=1e-3,
///                    min_value=5.0, max_value=7.0).fit_transform(X) =
///     [[1.0, 2.0, 3.0],
///      [2.0, 5.0, 5.0],
///      [3.0, 6.0, 6.999999523809311],
///      [4.0, 8.0, 9.0],
///      [5.0, 10.0, 11.0],
///      [6.0, 12.0, 13.0]]
///   Every imputed cell is clipped into [5.0, 7.0]: [1,1]->5.0 (clipped up),
///   [4,0]->5.0 (clipped up), [2,2]->~7.0 (clipped down).
///
/// ferrolearn has no clip parameter at all, so its imputed values for the SAME
/// cells fall OUTSIDE [5.0, 7.0] (e.g. [1,1] ~= 4.047 < 5.0). This test asserts
/// the sklearn CLIP bound on those cells; ferrolearn (unclipped) violates it.
///
/// Tracking #1408 (min_value/max_value clip missing).
#[test]
fn divergence_min_max_clip_bound() {
    // sklearn oracle: IterativeImputer(random_state=0, max_iter=10, tol=1e-3,
    //                                  min_value=5.0, max_value=7.0)
    let imp = IterativeImputer::<f64>::new(10, 1e-3, InitialStrategy::Mean)
        .with_min_value(5.0)
        .with_max_value(7.0);
    let x: Array2<f64> = array![
        [1.0, 2.0, 3.0],
        [2.0, f64::NAN, 5.0],
        [3.0, 6.0, f64::NAN],
        [4.0, 8.0, 9.0],
        [f64::NAN, 10.0, 11.0],
        [6.0, 12.0, 13.0],
    ];
    let out = imp.fit_transform(&x).expect("fit_transform");

    // sklearn enforces min_value=5.0, max_value=7.0 on every imputed cell.
    let sk_min = 5.0_f64;
    let sk_max = 7.0_f64;
    let sk_11 = 5.0_f64; // clipped up to min
    let sk_40 = 5.0_f64; // clipped up to min
    let sk_22 = 6.999999523809311_f64; // near max

    // Bound check (the contract sklearn enforces every iteration).
    let cells = [out[[1, 1]], out[[4, 0]], out[[2, 2]]];
    for &v in &cells {
        assert!(
            v >= sk_min - TOL && v <= sk_max + TOL,
            "DIV-VAL-3: imputed value {v} not within sklearn clip bound [{sk_min}, {sk_max}]"
        );
    }
    assert!(
        (out[[1, 1]] - sk_11).abs() < TOL,
        "DIV-VAL-3: [1,1]={} != {sk_11}",
        out[[1, 1]]
    );
    assert!(
        (out[[4, 0]] - sk_40).abs() < TOL,
        "DIV-VAL-3: [4,0]={} != {sk_40}",
        out[[4, 0]]
    );
    assert!(
        (out[[2, 2]] - sk_22).abs() < TOL,
        "DIV-VAL-3: [2,2]={} != {sk_22}",
        out[[2, 2]]
    );
}
