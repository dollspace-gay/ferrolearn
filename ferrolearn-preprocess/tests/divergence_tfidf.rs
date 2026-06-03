//! Divergence audit: `TfidfTransformer` vs scikit-learn 1.5.2
//! `sklearn/feature_extraction/text.py` `class TfidfTransformer` (`:1483`).
//!
//! GREEN GUARDS (oracle-grounded value parity). Every expected value here is a
//! LIVE sklearn 1.5.2 oracle value computed from /tmp (R-CHAR-3) — never copied
//! from the ferrolearn side. The shared count matrix is
//! `[[1,1,0],[1,0,1],[1,0,0]]` (feature 0 in all 3 docs; features 1,2 in 1 doc
//! each), matching the design-doc Probes (`.design/preprocess/tfidf.md`).
//!
//! Oracle reproduction (sklearn 1.5.2):
//! ```text
//! cd /tmp && python3 -c "
//! import numpy as np
//! from sklearn.feature_extraction.text import TfidfTransformer as T
//! c=np.array([[1.,1.,0.],[1.,0.,1.],[1.,0.,0.]])
//! print(T().fit(c).idf_.tolist())
//! print(T().fit_transform(c).toarray().tolist())
//! print(T(smooth_idf=False).fit(c).idf_.tolist())
//! print(T(norm='l1').fit_transform(c).toarray().tolist())
//! print(T(norm=None).fit_transform(c).toarray().tolist())
//! print(T(sublinear_tf=True,use_idf=False,norm=None).fit_transform(np.array([[4.,1.]])).toarray().tolist())
//! print(T(use_idf=False).fit_transform(c).toarray().tolist())"
//! ```

use ferrolearn_preprocess::tfidf::{TfidfNorm, TfidfTransformer};
use ndarray::{Array2, array};

/// Shared count matrix `[[1,1,0],[1,0,1],[1,0,0]]` (design-doc Probe matrix).
fn counts() -> Array2<f64> {
    array![[1.0_f64, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
}

const TOL: f64 = 1e-12;

fn assert_mat_eq(got: &Array2<f64>, expected: &[[f64; 3]; 3]) {
    assert_eq!(got.shape(), &[3, 3], "shape mismatch");
    for i in 0..3 {
        for j in 0..3 {
            let g = got[[i, j]];
            let e = expected[i][j];
            assert!(
                (g - e).abs() <= TOL,
                "[{i},{j}]: ferrolearn={g} sklearn={e} (diff {})",
                (g - e).abs()
            );
        }
    }
}

/// REQ-1 — default smooth IDF vector.
/// sklearn `text.py:1660-1666`: `df += 1; n += 1; idf_ = log(n/df) + 1`.
/// Live oracle `TfidfTransformer().fit(c).idf_`
///   = [1.0, 1.6931471805599454, 1.6931471805599454].
#[test]
fn green_req1_idf_default_smooth() {
    // Oracle (live sklearn 1.5.2):
    let sk_idf = [1.0_f64, 1.6931471805599454, 1.6931471805599454];
    let fitted = TfidfTransformer::<f64>::new().fit(&counts()).unwrap();
    let idf = fitted.idf().expect("use_idf default true -> Some");
    assert_eq!(idf.len(), 3);
    for (j, &e) in sk_idf.iter().enumerate() {
        assert!(
            (idf[j] - e).abs() <= TOL,
            "idf[{j}]: ferrolearn={} sklearn={e}",
            idf[j]
        );
    }
}

/// REQ-2 — default `fit_transform` (use_idf, smooth, l2). THE KEY MISSING GUARD:
/// the in-module test only pins unit-l2 rows, not the full value vector.
/// sklearn `text.py:1705`,`:1708`: `X.data *= idf_[X.indices]; normalize(X,'l2')`.
/// Live oracle `TfidfTransformer().fit_transform(c).toarray()`.
#[test]
fn green_req2_default_fit_transform_l2_full_vector() {
    // Oracle (live sklearn 1.5.2):
    let sk = [
        [0.5085423203783267, 0.8610369959439764, 0.0],
        [0.5085423203783267, 0.0, 0.8610369959439764],
        [1.0, 0.0, 0.0],
    ];
    let c = counts();
    let fitted = TfidfTransformer::<f64>::new().fit(&c).unwrap();
    let got = fitted.transform(&c).unwrap();
    assert_mat_eq(&got, &sk);
}

/// REQ-3 — `smooth_idf=False` IDF vector.
/// sklearn `text.py:1660-1666` (smooth falsey): `idf_ = log(n/df) + 1`.
/// Live oracle `TfidfTransformer(smooth_idf=False).fit(c).idf_`.
#[test]
fn green_req3_idf_nosmooth() {
    // Oracle (live sklearn 1.5.2):
    let sk_idf = [1.0_f64, 2.09861228866811, 2.09861228866811];
    let fitted = TfidfTransformer::<f64>::new()
        .smooth_idf(false)
        .fit(&counts())
        .unwrap();
    let idf = fitted.idf().unwrap();
    for (j, &e) in sk_idf.iter().enumerate() {
        assert!(
            (idf[j] - e).abs() <= TOL,
            "idf[{j}]: ferrolearn={} sklearn={e}",
            idf[j]
        );
    }
}

/// REQ-4 — `norm='l1'` full output vector.
/// sklearn `text.py:1707-1708`: `normalize(X, norm='l1')`.
/// Live oracle `TfidfTransformer(norm='l1').fit_transform(c).toarray()`.
#[test]
fn green_req4_norm_l1_full_vector() {
    // Oracle (live sklearn 1.5.2):
    let sk = [
        [0.37131279241563214, 0.6286872075843678, 0.0],
        [0.37131279241563214, 0.0, 0.6286872075843678],
        [1.0, 0.0, 0.0],
    ];
    let c = counts();
    let fitted = TfidfTransformer::<f64>::new()
        .norm(TfidfNorm::L1)
        .fit(&c)
        .unwrap();
    let got = fitted.transform(&c).unwrap();
    assert_mat_eq(&got, &sk);
}

/// REQ-4 — `norm=None` full output vector (raw tf*idf, no normalization).
/// sklearn `text.py:1707`: `if self.norm is not None` guard skips normalize.
/// Live oracle `TfidfTransformer(norm=None).fit_transform(c).toarray()`.
#[test]
fn green_req4_norm_none_full_vector() {
    // Oracle (live sklearn 1.5.2):
    let sk = [
        [1.0, 1.6931471805599454, 0.0],
        [1.0, 0.0, 1.6931471805599454],
        [1.0, 0.0, 0.0],
    ];
    let c = counts();
    let fitted = TfidfTransformer::<f64>::new()
        .norm(TfidfNorm::None)
        .fit(&c)
        .unwrap();
    let got = fitted.transform(&c).unwrap();
    assert_mat_eq(&got, &sk);
}

/// REQ-5 — `sublinear_tf=True` (use_idf=False, norm=None) on `[[4,1]]`.
/// sklearn `text.py:1698-1700`: `log(X.data); X.data += 1`.
/// Live oracle
/// `TfidfTransformer(sublinear_tf=True,use_idf=False,norm=None).fit_transform([[4,1]])`.
#[test]
fn green_req5_sublinear_tf() {
    // Oracle (live sklearn 1.5.2):
    let sk = [2.386294361119891_f64, 1.0];
    let c = array![[4.0_f64, 1.0]];
    let fitted = TfidfTransformer::<f64>::new()
        .use_idf(false)
        .sublinear_tf(true)
        .norm(TfidfNorm::None)
        .fit(&c)
        .unwrap();
    let got = fitted.transform(&c).unwrap();
    assert_eq!(got.shape(), &[1, 2]);
    for (j, &e) in sk.iter().enumerate() {
        assert!(
            (got[[0, j]] - e).abs() <= TOL,
            "[0,{j}]: ferrolearn={} sklearn={e}",
            got[[0, j]]
        );
    }
}

/// REQ-6 — `use_idf=False` default-l2 path: pure TF + l2-normalize, no IDF.
/// sklearn `text.py:1654` (no idf_ set), `:1702` (`hasattr(self,'idf_')` false),
/// `:1708` (`normalize 'l2'`).
/// Live oracle `TfidfTransformer(use_idf=False).fit_transform(c).toarray()`.
#[test]
fn green_req6_use_idf_false_l2() {
    // Oracle (live sklearn 1.5.2):
    let inv_sqrt2 = 0.7071067811865475;
    let sk = [
        [inv_sqrt2, inv_sqrt2, 0.0],
        [inv_sqrt2, 0.0, inv_sqrt2],
        [1.0, 0.0, 0.0],
    ];
    let c = counts();
    let fitted = TfidfTransformer::<f64>::new()
        .use_idf(false)
        .fit(&c)
        .unwrap();
    // REQ-6 also asserts no idf_ attribute when use_idf=False.
    assert!(
        fitted.idf().is_none(),
        "use_idf=False must yield idf()==None"
    );
    let got = fitted.transform(&c).unwrap();
    assert_mat_eq(&got, &sk);
}

/// REQ-1/2 f32 path: same default-l2 oracle vector, looser tolerance.
/// Confirms the generic `F: Float` path matches sklearn float32 semantics.
/// The oracle constants are the f64 ground truth (R-CHAR-3 traceability),
/// compared against the f32 ferrolearn output at 1e-6.
#[test]
fn green_req2_f32_default_fit_transform() {
    // Oracle (live sklearn 1.5.2), held as f64 then cast at compare-time:
    let sk: [[f64; 3]; 3] = [
        [0.5085423203783267, 0.8610369959439764, 0.0],
        [0.5085423203783267, 0.0, 0.8610369959439764],
        [1.0, 0.0, 0.0],
    ];
    let c: Array2<f32> = array![[1.0_f32, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let fitted = TfidfTransformer::<f32>::new().fit(&c).unwrap();
    let got = fitted.transform(&c).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let g = f64::from(got[[i, j]]);
            assert!(
                (g - sk[i][j]).abs() <= 1e-6,
                "[{i},{j}]: ferrolearn={g} sklearn={}",
                sk[i][j]
            );
        }
    }
}

/// Edge: empty fit (0 rows). sklearn `_validate_data` rejects with
/// "minimum of 1 is required" ValueError. ferrolearn rejects with
/// `FerroError::InsufficientSamples`. Both reject -> green guard (parity of
/// rejection, not of error type — error-type parity is REQ-9/NOT-STARTED).
#[test]
fn green_edge_empty_fit_rejected() {
    let c = Array2::<f64>::zeros((0, 3));
    assert!(
        TfidfTransformer::<f64>::new().fit(&c).is_err(),
        "sklearn rejects 0-sample fit; ferrolearn must too"
    );
}

/// Edge: transform with 0 rows. sklearn `_validate_data` (reset=False) rejects
/// with "minimum of 1 is required". ferrolearn rejects with InsufficientSamples.
#[test]
fn green_edge_transform_zero_rows_rejected() {
    let train = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let fitted = TfidfTransformer::<f64>::new().fit(&train).unwrap();
    let zero = Array2::<f64>::zeros((0, 2));
    assert!(
        fitted.transform(&zero).is_err(),
        "sklearn rejects 0-row transform; ferrolearn must too"
    );
}

/// Edge: transform feature-count mismatch. sklearn raises
/// "X has 3 features, but TfidfTransformer is expecting 2 features".
/// ferrolearn raises `FerroError::ShapeMismatch`. Both reject -> green guard.
#[test]
fn green_edge_shape_mismatch_rejected() {
    let train = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let fitted = TfidfTransformer::<f64>::new().fit(&train).unwrap();
    let bad = array![[1.0_f64, 0.0, 0.0]];
    assert!(
        fitted.transform(&bad).is_err(),
        "sklearn rejects feature-count mismatch; ferrolearn must too"
    );
}
