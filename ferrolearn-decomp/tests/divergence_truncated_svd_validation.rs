//! Divergence pins for `TruncatedSVD::fit` PARAMETER / INPUT VALIDATION against
//! scikit-learn 1.5.2 `class TruncatedSVD`
//! (`/home/doll/scikit-learn/sklearn/decomposition/_truncated_svd.py:30`).
//!
//! ferrolearn models the DEFAULT `algorithm='randomized'` path
//! (`_truncated_svd.py:240-255`). The CORE numerical outputs (singular_values_,
//! components_ incl. svd_flip sign, explained_variance_, explained_variance_ratio_,
//! transform, inverse_transform) were audited against the live sklearn 1.5.2
//! arpack oracle on a large-mean fixture and a slowly-decaying-spectrum fixture
//! and match to ~1e-9 — those are CLEAN and covered by
//! `divergence_truncated_svd.rs`. The two pins below are the FLAG'd-but-unpinned
//! VALIDATION divergences from REQ-11.
//!
//! All expected behavior comes from the live sklearn 1.5.2 oracle (run from /tmp,
//! R-CHAR-3), cited inline per test.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::TruncatedSVD;
use ndarray::array;

// ---------------------------------------------------------------------------
// DIV-A: n_components validation boundary (randomized path).
//
// sklearn `_truncated_svd.py:241-245` — the ONLY rejection on the default
// randomized path is:
//     if self.n_components > X.shape[1]:
//         raise ValueError("n_components(..) must be <= n_features(..).")
// i.e. n_components is rejected ONLY when it exceeds n_features. n_samples does
// NOT enter the validation. For X of shape (2, 4) with n_components=3
// (3 > n_samples=2 but 3 <= n_features=4), sklearn ACCEPTS and fits:
//   live oracle (sklearn 1.5.2, /tmp):
//     TruncatedSVD(n_components=3, algorithm='randomized', random_state=0).fit(X)
//     -> OK, singular_values_ = [14.22740741, 1.25732984] (rank-truncated), no raise.
//
// ferrolearn `truncated_svd.rs:581-590` rejects when
//     n_components > min(n_samples, n_features)
// so for (2, 4) with n_components=3 it returns
//     Err(InvalidParameter{n_components ... exceeds min(n_samples,n_features)=2}).
// EXPECTED (sklearn): fit succeeds (is_ok). ACTUAL (ferrolearn): Err.
// Tracking: #2383
// ---------------------------------------------------------------------------

#[test]
#[ignore = "divergence: TruncatedSVD::fit rejects n_components in (n_samples, n_features]; sklearn randomized accepts n_components<=n_features (_truncated_svd.py:241); tracking #2383"]
fn divergence_n_components_between_n_samples_and_n_features_accepted() {
    // X: n_samples=2, n_features=4. n_components=3 satisfies sklearn's
    // randomized rule (3 <= n_features=4) so sklearn fits without error.
    let x = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    let result = TruncatedSVD::<f64>::new(3).with_random_state(0).fit(&x, &());
    // sklearn oracle: the fit SUCCEEDS (no ValueError) because only
    // n_components > n_features is rejected (_truncated_svd.py:241).
    assert!(
        result.is_ok(),
        "sklearn randomized accepts n_components(3) <= n_features(4) even when \
         n_components > n_samples(2) (_truncated_svd.py:241); ferrolearn rejected: {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// DIV-B: ensure_min_features=2 on a single-feature X.
//
// sklearn `_truncated_svd.py:228`:
//     X = self._validate_data(X, accept_sparse=["csr","csc"], ensure_min_features=2)
// A single-feature X (n_features == 1) fails this check BEFORE any SVD:
//   live oracle (sklearn 1.5.2, /tmp):
//     TruncatedSVD(n_components=1).fit(np.array([[1.],[2.],[3.]]))
//     -> ValueError: "Found array with 1 feature(s) (shape=(3, 1)) while a
//        minimum of 2 is required by TruncatedSVD."
//
// ferrolearn `truncated_svd.rs:572-604` has no min-features guard, so a
// single-feature X with n_components=1 fits and returns Ok.
// EXPECTED (sklearn): raise (Err). ACTUAL (ferrolearn): Ok.
// Tracking: #2384
// ---------------------------------------------------------------------------

#[test]
#[ignore = "divergence: TruncatedSVD::fit accepts single-feature X; sklearn ensure_min_features=2 raises ValueError (_truncated_svd.py:228); tracking #2384"]
fn divergence_single_feature_rejected_min_features_2() {
    // X: n_features == 1. sklearn rejects via ensure_min_features=2.
    let x = array![[1.0], [2.0], [3.0]];
    let result = TruncatedSVD::<f64>::new(1).with_random_state(0).fit(&x, &());
    // sklearn oracle: raises ValueError (minimum of 2 features required).
    assert!(
        result.is_err(),
        "sklearn requires ensure_min_features=2 (_truncated_svd.py:228) and \
         raises ValueError on a 1-feature X; ferrolearn returned Ok"
    );
}
