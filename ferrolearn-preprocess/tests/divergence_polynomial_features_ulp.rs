//! ACToR critic pin: ferrolearn `PolynomialFeatures::transform` diverges from
//! scikit-learn 1.5.2 at the LAST ULP for higher-degree (>= 3) product terms,
//! because ferrolearn forms each output column by a LEFT-FOLD product over the
//! combination indices
//! (`generate_poly_features`: `combo.iter().fold(F::one(), |acc, &j| acc * x[[i, j]])`,
//! `ferrolearn-preprocess/src/polynomial_features.rs:245`), whereas scikit-learn
//! builds degree-d columns INCREMENTALLY by multiplying a previously-computed
//! degree-(d-1) column by a single feature with `np.multiply(..., casting="no")`
//! (`sklearn/preprocessing/_polynomial.py:555-560`). Floating-point multiplication
//! is not associative, so the two association orders round differently.
//!
//! The shipped claim under audit ("the polynomial VALUES + column ORDER match
//! sklearn exactly", "bit-identical to live oracle"; `polynomial_features.rs:23`)
//! is FALSE at the bit level for degree-3+ terms.
//!
//! Live sklearn 1.5.2 oracle (run from /tmp, R-CHAR-3 method (a)):
//! ```text
//! python3 -c "import numpy as np; from sklearn.preprocessing import PolynomialFeatures; \
//!   out = PolynomialFeatures(3, interaction_only=True, include_bias=False) \
//!     .fit_transform(np.array([[0.5,1.2,1.9,2.6]]))[0]; \
//!   print(out.tolist()); print(float(out[13]).hex())"
//!   -> [..., 5.927999999999999]
//!   -> 0x1.7b645a1cac082p+2        # the x1*x2*x3 column (powers_ row [0,1,1,1])
//! ```
//! ferrolearn's left-fold computes `((1.0*1.2)*1.9)*2.6 == 5.928`
//! (`0x1.7b645a1cac083p+2`), one ULP ABOVE the sklearn value.

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::PolynomialFeatures;
use ndarray::array;

/// Divergence: `ferrolearn::PolynomialFeatures::transform` diverges from
/// `sklearn/preprocessing/_polynomial.py:555` for the degree-3 interaction term
/// `x1*x2*x3` of `X = [[0.5, 1.2, 1.9, 2.6]]`.
///
/// sklearn returns `0x1.7b645a1cac082p+2` (`5.927999999999999`) via its
/// incremental `np.multiply(casting="no")` expansion; ferrolearn returns
/// `0x1.7b645a1cac083p+2` (`5.928`) via the left-fold in
/// `generate_poly_features`. One ULP apart.
///
/// Tracking: #2210
#[test]
fn divergence_polynomial_higher_degree_term_ulp_vs_sklearn() {
    // Expected = LIVE sklearn 1.5.2 oracle bit pattern (NOT copied from ferrolearn):
    //   PolynomialFeatures(3, interaction_only=True, include_bias=False)
    //     .fit_transform([[0.5,1.2,1.9,2.6]])[0][13]  ==  0x1.7b645a1cac082p+2
    let sklearn_x1x2x3: f64 = f64::from_bits(0x4017_b645_a1ca_c082);
    assert_eq!(sklearn_x1x2x3, 5.927_999_999_999_999_f64); // sanity: oracle value

    let poly = PolynomialFeatures::<f64>::new(3, true, false).unwrap();
    let x = array![[0.5, 1.2, 1.9, 2.6]];
    let out = poly.transform(&x).unwrap();

    // Combination order (REQ-1): a,b,c,d, ab,ac,ad,bc,bd,cd, abc,abd,acd,bcd.
    // Column 13 (last) is the x1*x2*x3 term == b*c*d (powers_ row [0,1,1,1]).
    let ferro_x1x2x3 = out[[0, 13]];

    assert_eq!(
        ferro_x1x2x3.to_bits(),
        sklearn_x1x2x3.to_bits(),
        "sklearn computes x1*x2*x3 as {:?} ({:#018x}) via incremental \
         np.multiply (_polynomial.py:555); ferrolearn's left-fold gives {:?} \
         ({:#018x}) — differs by 1 ULP",
        sklearn_x1x2x3,
        sklearn_x1x2x3.to_bits(),
        ferro_x1x2x3,
        ferro_x1x2x3.to_bits(),
    );
}
