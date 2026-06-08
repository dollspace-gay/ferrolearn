//! Divergence pin: `FittedLabelBinarizer::inverse_transform` (binary) overflows
//! when computing the threshold for large `neg_label`/`pos_label`, whereas
//! scikit-learn 1.5.2 (arbitrary-precision Python ints) returns the correct
//! class.
//!
//! Discriminator artifact (ACToR critic). Expected value below comes from a LIVE
//! sklearn 1.5.2 oracle call (run from /tmp) — never copied from ferrolearn
//! (R-CHAR-3).

use ferrolearn_core::traits::Fit;
use ferrolearn_preprocess::label_binarizer::LabelBinarizer;
use ndarray::array;

// ===========================================================================
// DIVERGENCE: binary inverse_transform threshold = (pos_label + neg_label) / 2
// overflows i64 for large valid neg/pos.
//
// sklearn `_inverse_binarize_thresholding` (`_label.py:644`) thresholds with
// `threshold = (pos_label + neg_label) / 2.0` (`_label.py:399-400`) computed in
// Python's arbitrary-precision integers, then `y > threshold` (`:667`), then
// `classes[y[:,1]]` / `classes[y.ravel()]` (`:674`,`:679`). No overflow: the
// threshold for neg=2^62, pos=2^62+1 is ~4.611686018427388e18.
//
// `LabelBinarizer.fit` only rejects `neg_label >= pos_label` (`_label.py:283-287`),
// so `neg_label = 2^62`, `pos_label = 2^62 + 1` is a VALID configuration.
//
// Live oracle (sklearn 1.5.2, from /tmp):
//   lb = LabelBinarizer(neg_label=2**62, pos_label=2**62+1).fit([0,1])
//   # threshold ~= 4.611686018427388e18
//   lb.inverse_transform([[5.0e18]]) -> [1]   (5e18 > threshold)
//   lb.inverse_transform([[1.0e18]]) -> [0]   (1e18 < threshold)
//
// ferrolearn `FittedLabelBinarizer::inverse_transform`
// (`ferrolearn-preprocess/src/label_binarizer.rs:195`) computes
// `(self.pos_label + self.neg_label) as f64 / 2.0`. The i64 add
// `2^62 + (2^62 + 1) = 2^63 + 1` overflows i64::MAX (2^63 - 1):
//   - debug builds (how `cargo test` runs): PANIC "attempt to add with overflow"
//     (also violates CLAUDE.md "Never panic in library code").
//   - release builds: wraps to threshold ~= -4.611686018427388e18, so the
//     `1.0e18` row maps to class 1 instead of sklearn's class 0.
//
// This test asserts the sklearn-oracle result; it FAILS (panics on overflow in
// debug, or wrong class in release) against current ferrolearn.
// ===========================================================================
#[test]
fn divergence_inverse_threshold_overflow_large_neg_pos() {
    let neg: i64 = 1i64 << 62; // 4_611_686_018_427_387_904
    let pos: i64 = (1i64 << 62) + 1; // 4_611_686_018_427_387_905, > neg so fit accepts

    let fitted = LabelBinarizer::new()
        .with_neg_label(neg)
        .with_pos_label(pos)
        .fit(&array![0_usize, 1], &())
        .expect("neg < pos is a valid configuration; fit must accept it");

    // sklearn-oracle: threshold ~= 4.611686018427388e18 (strict `>`).
    // 5e18 is above the threshold -> class 1.
    let above = fitted
        .inverse_transform(&array![[5.0e18_f64]])
        .expect("inverse_transform must not error for a valid 1-column input");
    assert_eq!(above, array![1_usize]);

    // 1e18 is below the threshold -> class 0 (this is the value release-mode
    // overflow gets WRONG, mapping it to class 1).
    let below = fitted
        .inverse_transform(&array![[1.0e18_f64]])
        .expect("inverse_transform must not error for a valid 1-column input");
    assert_eq!(below, array![0_usize]);
}
