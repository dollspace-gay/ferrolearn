//! Divergence audit: `TfidfTransformer` fitted `idf_` for a df=0 column with
//! `smooth_idf=False`, vs scikit-learn 1.5.2
//! `sklearn/feature_extraction/text.py` `TfidfTransformer.fit` (`:1654-1666`).
//!
//! sklearn computes the unsmoothed idf as `np.log(n_samples / df) + 1.0`
//! (`text.py:1666`). When a vocabulary column never appears in any document
//! (`df == 0`), `df` stays 0 (no smoothing offset since `smooth_idf=False`),
//! so sklearn evaluates `log(n / 0) + 1 = +inf`. ferrolearn's `fit` instead
//! takes an explicit `df > 0` guard branch and returns `1.0`
//! (`ferrolearn-preprocess/src/tfidf.rs:162-166`). The fitted `idf_` attribute
//! therefore diverges in VALUE: sklearn `inf` vs ferrolearn `1.0`.
//!
//! The src doc-comment labels this a deliberate "DEVIATION (R-DEV-4)", but per
//! goal.md R-DEFER-3 there is no "ACCEPTABLE DRIFT" verdict — a fitted-attribute
//! value mismatch against the live oracle is a divergence and is pinned here.
//!
//! Oracle reproduction (sklearn 1.5.2, warnings silenced for the divide):
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np, warnings
//! from sklearn.feature_extraction.text import TfidfTransformer as T
//! c = np.array([[1.,0.],[1.,0.],[1.,0.]])   # column 1 has df=0
//! with warnings.catch_warnings():
//!     warnings.simplefilter('ignore')
//!     print(T(smooth_idf=False).fit(c).idf_.tolist())
//! "
//! # -> [1.0, inf]
//! ```

use ferrolearn_preprocess::tfidf::{TfidfNorm, TfidfTransformer};
use ndarray::array;

/// Divergence: ferrolearn's `TfidfTransformer::fit` diverges from
/// `sklearn/feature_extraction/text.py:1666` for a df=0 column with
/// `smooth_idf=False`.
/// sklearn returns `idf_[1] == +inf`; ferrolearn returns `idf_[1] == 1.0`.
/// Tracking: #2333
#[test]
fn divergence_idf_nosmooth_df0_is_inf() {
    // Column 1 never appears in any document -> df == 0.
    let counts = array![[1.0_f64, 0.0], [1.0, 0.0], [1.0, 0.0]];
    let fitted = TfidfTransformer::<f64>::new()
        .smooth_idf(false)
        .fit(&counts)
        .unwrap();
    let idf = fitted.idf().unwrap();

    // sklearn oracle (smooth_idf=False): idf_ == [1.0, inf]  (text.py:1666)
    // log(3/3)+1 = 1.0 for column 0 (df=3); log(3/0)+1 = +inf for column 1 (df=0).
    assert_eq!(idf[0], 1.0, "column 0 (df=n) idf must be 1.0");
    assert!(
        idf[1].is_infinite() && idf[1].is_sign_positive(),
        "sklearn idf_ for a df=0 column with smooth_idf=False is +inf (text.py:1666); \
         ferrolearn returned {}",
        idf[1]
    );
}

/// Transform of the FIT data (the df=0 column is all zeros) must keep those zeros
/// at 0 — NOT `0 * inf = nan`. sklearn operates on a CSR matrix's stored `.data`
/// (`text.py:1705` `X.data *= idf_[X.indices]`), so the never-stored zeros of a
/// df=0 column are never multiplied by the column's `inf` idf.
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// import numpy as np, warnings
/// from sklearn.feature_extraction.text import TfidfTransformer as T
/// c = np.array([[1.,0.],[1.,0.],[1.,0.]])
/// with warnings.catch_warnings():
///     warnings.simplefilter('ignore')
///     m = T(smooth_idf=False, norm=None).fit(c)
///     print(m.transform(c).toarray().tolist())   # -> [[1.0,0.0],[1.0,0.0],[1.0,0.0]]
/// ```
/// Tracking: #2333
#[test]
fn divergence_transform_fitdata_df0_zeros_stay_zero() {
    let counts = array![[1.0_f64, 0.0], [1.0, 0.0], [1.0, 0.0]];
    let fitted = TfidfTransformer::<f64>::new()
        .smooth_idf(false)
        .norm(TfidfNorm::None)
        .fit(&counts)
        .unwrap();
    let out = fitted.transform(&counts).unwrap();

    // sklearn oracle: [[1,0],[1,0],[1,0]] — the df=0 column's zeros stay 0 (no nan).
    for i in 0..3 {
        assert_eq!(out[[i, 0]], 1.0, "column 0 (idf=1.0): tf*1.0 = 1.0");
        assert_eq!(
            out[[i, 1]],
            0.0,
            "df=0 column zero count must stay 0 (sklearn sparse multiply skips zeros), got {}",
            out[[i, 1]]
        );
        assert!(!out[[i, 1]].is_nan(), "zero entry must not become nan");
    }
}

/// A NEW document with a non-zero count in the df=0 column → `inf`.
/// sklearn: `2 * inf = inf` with `norm=None` (the stored entry IS multiplied).
///
/// Oracle (sklearn 1.5.2):
/// ```text
/// m.transform(np.array([[1.,2.]])).toarray().tolist()   # -> [[1.0, inf]]
/// ```
/// Tracking: #2333
#[test]
fn divergence_transform_newdoc_df0_nonzero_is_inf() {
    let counts = array![[1.0_f64, 0.0], [1.0, 0.0], [1.0, 0.0]];
    let fitted = TfidfTransformer::<f64>::new()
        .smooth_idf(false)
        .norm(TfidfNorm::None)
        .fit(&counts)
        .unwrap();
    // New doc: count 2 in the df=0 column.
    let new_doc = array![[1.0_f64, 2.0]];
    let out = fitted.transform(&new_doc).unwrap();

    assert_eq!(out[[0, 0]], 1.0, "column 0 (idf=1.0): 1*1.0 = 1.0");
    assert!(
        out[[0, 1]].is_infinite() && out[[0, 1]].is_sign_positive(),
        "non-zero count (2) in a df=0 column (idf=inf) -> 2*inf = +inf (sklearn norm=None); got {}",
        out[[0, 1]]
    );
}
