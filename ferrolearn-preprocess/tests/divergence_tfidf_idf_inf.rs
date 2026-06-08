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

use ferrolearn_preprocess::tfidf::TfidfTransformer;
use ndarray::array;

/// Divergence: ferrolearn's `TfidfTransformer::fit` diverges from
/// `sklearn/feature_extraction/text.py:1666` for a df=0 column with
/// `smooth_idf=False`.
/// sklearn returns `idf_[1] == +inf`; ferrolearn returns `idf_[1] == 1.0`.
/// Tracking: #2333
#[test]
#[ignore = "divergence: smooth_idf=False df=0 idf_ sklearn inf vs ferrolearn 1.0; tracking #2333"]
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
