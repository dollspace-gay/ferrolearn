//! Divergence pins: `ferrolearn-preprocess` `LabelEncoder` ERROR-MESSAGE parity
//! vs scikit-learn 1.5.2 (audit #2353, R-DEV-2 exception/message parity).
//!
//! The existing `divergence_label_encoder.rs` asserts only that ferrolearn
//! REJECTS unseen-label transform and out-of-range inverse_transform, declaring
//! the message-string difference "cosmetic and not pinned". But R-DEV-2 requires
//! default/exception parity, and the in-module REQ table (REQ-6, label_encoder.rs)
//! itself flags the message gap as an OPEN blocker (#1136). These tests pin the
//! observable message divergence so it cannot be silently shipped.
//!
//! LIVE sklearn 1.5.2 oracle (run from /tmp, NOT copied from ferrolearn):
//! ```text
//! >>> LabelEncoder().fit(['a','b']).transform(['a','c'])
//! ValueError: y contains previously unseen labels: np.str_('c')
//! >>> le=LabelEncoder().fit(['a','b']); le.inverse_transform([0,2,1])
//! ValueError: y contains previously unseen labels: [2]
//! ```
//! sklearn source:
//!   - transform unseen: `sklearn/utils/_encode.py:227`
//!     `raise ValueError(f"y contains previously unseen labels: {str(e)}")`
//!   - inverse OOB:      `sklearn/preprocessing/_label.py:160`
//!     `raise ValueError("y contains previously unseen labels: %s" % str(diff))`
//!
//! ferrolearn ACTUAL (label_encoder.rs:172-175 / :106-109, FerroError Display
//! `error.rs:87` `"Invalid parameter `{name}`: {reason}"`):
//!   - transform unseen -> `Invalid parameter `x[1]`: unknown label "c"`
//!   - inverse OOB      -> `Invalid parameter `y[1]`: index 2 is out of range (n_classes = 2)`
//!
//! Neither contains the sklearn phrase "y contains previously unseen labels".

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::LabelEncoder;
use ndarray::{Array1, array};

fn str_arr(v: &[&str]) -> Array1<String> {
    Array1::from_vec(v.iter().map(std::string::ToString::to_string).collect())
}

/// Divergence: `FittedLabelEncoder::transform` on an unseen label diverges from
/// `sklearn/utils/_encode.py:227` for `fit(['a','b']).transform(['a','c'])`.
/// sklearn raises `ValueError("y contains previously unseen labels: np.str_('c')")`;
/// ferrolearn returns `InvalidParameter{ ... reason: "unknown label \"c\"" }`
/// whose Display lacks the documented sklearn phrase.
/// Tracking: #2354
#[test]
fn divergence_unseen_label_transform_message() {
    let enc = LabelEncoder::new();
    let fitted = enc.fit(&str_arr(&["a", "b"]), &()).unwrap();
    let err = fitted
        .transform(&str_arr(&["a", "c"]))
        .expect_err("sklearn raises ValueError on previously unseen labels");
    let msg = err.to_string();
    // Oracle-grounded required substring (sklearn _encode.py:227).
    assert!(
        msg.contains("y contains previously unseen labels"),
        "expected sklearn ValueError phrase; got ferrolearn message: {msg:?}"
    );
}

/// Divergence: `FittedLabelEncoder::inverse_transform` on an out-of-range index
/// diverges from `sklearn/preprocessing/_label.py:160` for
/// `fit(['a','b']).inverse_transform([0,2,1])`.
/// sklearn raises `ValueError("y contains previously unseen labels: [2]")`;
/// ferrolearn returns `InvalidParameter{ ... reason: "index 2 is out of range ..." }`
/// whose Display lacks the documented sklearn phrase.
/// Tracking: #2355
#[test]
fn divergence_inverse_transform_oob_message() {
    let enc = LabelEncoder::new();
    let fitted = enc.fit(&str_arr(&["a", "b"]), &()).unwrap();
    let err = fitted
        .inverse_transform(&array![0usize, 2usize, 1usize])
        .expect_err("sklearn raises ValueError on out-of-range inverse index");
    let msg = err.to_string();
    // Oracle-grounded required substring (sklearn _label.py:160).
    assert!(
        msg.contains("y contains previously unseen labels"),
        "expected sklearn ValueError phrase; got ferrolearn message: {msg:?}"
    );
}
