//! ACToR critic pin (re-audit of the #2210 incremental-build rewrite):
//! ferrolearn `PolynomialFeatures::transform` PANICS (index out of bounds)
//! instead of producing sklearn's output whenever `interaction_only = true` AND
//! `degree >= 4` (for `n_features >= 2`), i.e. when some degree block produces
//! ZERO new columns and the early `break` leaves the per-degree block index
//! vector too short for the next degree iteration.
//!
//! Full panic surface found by grid fuzz (n_features 1..=5, degree 1..=5,
//! interaction_only {T,F}, include_bias {T,F}): EVERY cell with
//! `interaction_only=true && n_features>=2 && degree>=4` panics
//! (nf=2..5 × deg∈{4,5} × ib∈{T,F} = 16 cells). All other 504 ferro-produced
//! cells are bit-exact vs sklearn (f64 and f32). n_features=1 never panics.
//!
//! Root cause: in `generate_poly_features` / `num_output_columns`
//! (`ferrolearn-preprocess/src/polynomial_features.rs:292-316`, `:341-358`) the
//! per-degree inner loop does `new_index.push(current_col)` then `break`s when a
//! feature yields no new columns. On `break` the `new_index` vector is left with
//! FEWER than `n_features + 1` entries, after which `new_index.push(current_col)`
//! gives it at most `feature_idx + 2 <= n_features` entries. The next degree
//! iteration then evaluates `let end = index[n_features];`
//! (`polynomial_features.rs:294` / `:343`), indexing past the end of the short
//! `index` vector → `panic: index out of bounds: the len is N but the index is N`.
//!
//! scikit-learn's equivalent loop (`sklearn/preprocessing/_polynomial.py:542-564`)
//!     for _ in range(2, self._max_degree + 1):
//!         new_index = []
//!         end = index[-1]                       # <-- LAST element, never OOB
//!         for feature_idx in range(n_features):
//!             start = index[feature_idx]
//!             new_index.append(current_col)
//!             ...
//!             next_col = current_col + end - start
//!             if next_col <= current_col:
//!                 break
//! reads `index[-1]` (always the final element, robust to a short list) and
//! produces a valid matrix: degrees above the interaction ceiling simply add no
//! columns. sklearn 1.5.2 returns a finite 3-column matrix; ferrolearn panics.
//!
//! Live sklearn 1.5.2 oracle (R-CHAR-3 method (a), run from /tmp):
//! ```text
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   out = PolynomialFeatures(4, interaction_only=True, include_bias=False) \
//!     .fit_transform(np.array([[0.5,1.2]]))[0]; \
//!   print(out.tolist()); [print(float(v).hex()) for v in out]"
//!   -> [0.5, 1.2, 0.6]
//!   -> 0x1.0000000000000p-1   # 0.5  (a)
//!   -> 0x1.3333333333333p+0   # 1.2  (b)
//!   -> 0x1.3333333333333p-1   # 0.6  (a*b)
//! ```
//!
//! Tracking: #2211

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::PolynomialFeatures;
use ndarray::array;

/// Divergence: `ferrolearn::PolynomialFeatures::transform` panics (index out of
/// bounds at `polynomial_features.rs:343`) for `interaction_only=true,
/// degree=4, include_bias=false` on a 2-feature input, where
/// `sklearn/preprocessing/_polynomial.py:544` (`end = index[-1]`) instead yields
/// the valid 3-column matrix `[0.5, 1.2, 0.6]`.
///
/// sklearn 1.5.2 returns `[0.5, 1.2, 0.6]`
/// (`0x1.0p-1`, `0x1.3333333333333p+0`, `0x1.3333333333333p-1`);
/// ferrolearn panics in `num_output_columns` / `generate_poly_features`.
///
/// Tracking: #2211
#[test]
fn divergence_interaction_only_degree_exceeds_features_panics() {
    // Expected = LIVE sklearn 1.5.2 oracle bit patterns (NOT copied from ferrolearn):
    //   PolynomialFeatures(4, interaction_only=True, include_bias=False)
    //     .fit_transform([[0.5, 1.2]])[0]  ==  [0.5, 1.2, 0.6]
    let sk_a: f64 = f64::from_bits(0x3fe0_0000_0000_0000); // 0x1.0p-1     = 0.5
    let sk_b: f64 = f64::from_bits(0x3ff3_3333_3333_3333); // 0x1.333..p+0 = 1.2
    let sk_ab: f64 = f64::from_bits(0x3fe3_3333_3333_3333); // 0x1.333..p-1 = 0.6
    assert_eq!(sk_a, 0.5_f64); // sanity vs oracle
    assert_eq!(sk_b, 1.2_f64); // sanity vs oracle
    assert_eq!(sk_ab, 0.6_f64); // sanity vs oracle

    let poly = PolynomialFeatures::<f64>::new(4, true, false).unwrap();
    let x = array![[0.5, 1.2]];

    // ferrolearn panics here (index out of bounds) instead of returning a value.
    let out = poly.transform(&x).unwrap();

    assert_eq!(
        out.shape(),
        &[1, 3],
        "sklearn (_polynomial.py:544) yields 3 columns [a, b, a*b]; \
         degrees 3,4 add no interaction columns for 2 features"
    );
    assert_eq!(out[[0, 0]].to_bits(), sk_a.to_bits());
    assert_eq!(out[[0, 1]].to_bits(), sk_b.to_bits());
    assert_eq!(out[[0, 2]].to_bits(), sk_ab.to_bits());
}
