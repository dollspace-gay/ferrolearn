//! Divergence pins: ferrolearn-decomp SPILLOVER estimators' NaN/Inf input
//! handling vs scikit-learn 1.5.2 (commit 156ef14). Continuation of #2287 /
//! commit 11da4a2ac (which fixed PCA/TruncatedSVD/NMF/FastICA/KernelPCA/
//! FactorAnalysis).
//!
//! Like the six already-fixed estimators, EVERY spillover estimator in scope
//! runs `self._validate_data(...)` whose `check_array` default
//! `force_all_finite=True` raises `ValueError("Input X contains NaN.")` (NaN) /
//! `ValueError("Input X contains infinity or a value too large ...")` (Inf)
//! BEFORE any decomposition / manifold / NIPALS math
//! (`sklearn/utils/validation.py:147-154`, reached from `check_array`).
//!
//! Per-estimator `_validate_data` / `check_array` sites (read at tag 1.5.2):
//!   - IncrementalPCA       `sklearn/decomposition/_incremental_pca.py:227` (fit/partial_fit), `:281` (transform)
//!   - SparsePCA            `sklearn/decomposition/_sparse_pca.py:81` (fit), `:116` (transform)
//!   - MiniBatchNMF         `sklearn/decomposition/_nmf.py:2236` (fit), `:2407` (transform)
//!   - LatentDirichletAllocation `sklearn/decomposition/_lda.py:566` (_check_non_neg_array, fit+transform)
//!   - DictionaryLearning   `sklearn/decomposition/_dict_learning.py:1674` (fit), `:1113` (transform)
//!   - MDS                  `sklearn/manifold/_mds.py:627` (fit_transform)
//!   - Isomap               `sklearn/manifold/_isomap.py` (fit via `_validate_data`, force_all_finite default)
//!   - SpectralEmbedding    `sklearn/manifold/_spectral_embedding.py:741` (fit)
//!   - LocallyLinearEmbedding `sklearn/manifold/_locally_linear.py:793` (fit), `:872` (transform)
//!   - TSNE                 `sklearn/manifold/_t_sne.py:884`/`:891` (fit)
//!   - PLSRegression/PLSCanonical/CCA `sklearn/cross_decomposition/_pls.py:265` (fit X), `:272` (fit y)
//!   - PLSSVD               `sklearn/cross_decomposition/_pls.py:1067`/`:1074` (fit)
//!
//! Live sklearn 1.5.2 oracle (run from /tmp, R-CHAR-3) — for EACH of the 16,
//! `fit(X_with_nan)` raises `ValueError`; first message line is exactly
//! `Input X contains NaN.`; `fit(X_with_inf)` raises
//! `Input X contains infinity or a value too large for dtype('float64').`.
//! transform(X_nan) after a finite fit likewise raises. (Oracle transcript:
//! IncrementalPCA/SparsePCA/MiniBatchNMF/LDA/DictionaryLearning/MDS/Isomap/
//! SpectralEmbedding/LLE/TSNE/PLSRegression/PLSCanonical/PLSSVD/CCA all
//! `ValueError: Input X contains NaN.`)
//!
//! ferrolearn (probed empirically, catch_unwind harness):
//!   - NO PANICS found (R-CODE-2 clean for these inputs) — even the high-risk
//!     LLE local-Gram solve and SpectralEmbedding/Isomap eigen returned
//!     `Ok`/`Err`, never panicked.
//!   - Ok(garbage), NO finiteness gate: IncrementalPCA, SparsePCA, MiniBatchNMF,
//!     LDA, DictionaryLearning, MDS, SpectralEmbedding, LLE, TSNE (fit) — and
//!     transform-time IncrementalPCA, SparsePCA, MiniBatchNMF, Isomap.
//!     (sparse_pca.rs / minibatch_nmf.rs `is_finite` occurrences are TEST-only,
//!     lines sparse_pca.rs:641-642 / minibatch_nmf.rs:664 — they do NOT gate
//!     fit/transform input.)
//!   - Err with the WRONG (incidental) error, not a clean finiteness rejection
//!     (R-DEV-2): Isomap (`NumericalInstability`: "kNN graph is disconnected"),
//!     PLSRegression/PLSCanonical (`ConvergenceFailure`: NIPALS), PLSSVD/CCA
//!     (`NumericalInstability`: SVD NoConvergence).
//!
//! The pinned assertion below mirrors the fixed-estimator contract
//! (`pca.rs::reject_non_finite` -> `InvalidParameter{name:"X", reason:"Input X
//! contains NaN or infinity."}`): a clean finiteness `Err` naming X. This FAILS
//! for BOTH the Ok(garbage) estimators (returns Ok) AND the wrong-Err
//! estimators (returns NumericalInstability/ConvergenceFailure, not a
//! finiteness InvalidParameter).
//!
//! Tracking: #2290.

use ferrolearn_core::error::FerroError;
use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{
    CCA, DictionaryLearning, IncrementalPCA, Isomap, LLE, LatentDirichletAllocation, MDS,
    MiniBatchNMF, PLSCanonical, PLSRegression, PLSSVD, SparsePCA, SpectralEmbedding, Tsne,
};
use ndarray::{Array2, array};

fn x_finite() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 2.0],
        [10.0, 1.0, 12.0],
        [2.0, 1.0, 0.5],
        [3.0, 6.0, 3.0],
    ]
}
fn x_nan() -> Array2<f64> {
    let mut x = x_finite();
    x[[1, 1]] = f64::NAN;
    x
}
fn x_inf() -> Array2<f64> {
    let mut x = x_finite();
    x[[1, 1]] = f64::INFINITY;
    x
}
fn x_nonneg() -> Array2<f64> {
    x_finite()
}
fn x_nonneg_nan() -> Array2<f64> {
    let mut x = x_finite();
    x[[1, 1]] = f64::NAN;
    x
}
fn y_finite() -> Array2<f64> {
    array![
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.5],
        [4.0, 2.0],
        [5.0, 1.0],
        [6.0, 3.0]
    ]
}

/// True iff `e` is the clean finiteness rejection the fixed estimators emit
/// (`InvalidParameter{name:"X", reason:"Input X contains NaN or infinity."}`),
/// matching sklearn's `ValueError("Input X contains NaN.")`. NumericalInstability
/// / ConvergenceFailure / Ok all fail this — they are NOT a deliberate
/// finiteness rejection.
fn is_finiteness_rejection(r: &Result<(), FerroError>) -> bool {
    matches!(
        r,
        Err(FerroError::InvalidParameter { name, reason })
            if name == "X"
                && (reason.to_lowercase().contains("nan")
                    || reason.to_lowercase().contains("infinit"))
    )
}

macro_rules! pin_fit_rejects {
    ($name:ident, $label:literal, $build:expr) => {
        /// Divergence: this spillover estimator does not raise the clean
        /// finiteness rejection sklearn 1.5.2 emits for NaN input.
        /// sklearn: `ValueError("Input X contains NaN.")`.
        /// ferrolearn: Ok(garbage) or incidental NumericalInstability/
        /// ConvergenceFailure (see module header). Tracking: #2290.
        #[test]
        fn $name() {
            let r: Result<(), FerroError> = ($build)(x_nan());
            assert!(
                is_finiteness_rejection(&r),
                "{}: expected clean finiteness rejection (sklearn `Input X contains NaN.`), got {:?}",
                $label,
                r
            );
        }
    };
}

// --- fit(X_nan): single-input decomposition/manifold estimators ---
pin_fit_rejects!(
    divergence_incremental_pca_fit_nan,
    "IncrementalPCA",
    |x: Array2<f64>| { IncrementalPCA::<f64>::new(2).fit(&x, &()).map(|_| ()) }
);
pin_fit_rejects!(divergence_sparse_pca_fit_nan, "SparsePCA", |x: Array2<
    f64,
>| {
    SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .map(|_| ())
});
pin_fit_rejects!(
    divergence_minibatch_nmf_fit_nan,
    "MiniBatchNMF",
    |_x: Array2<f64>| {
        MiniBatchNMF::<f64>::new(2)
            .with_random_state(0)
            .fit(&x_nonneg_nan(), &())
            .map(|_| ())
    }
);
pin_fit_rejects!(
    divergence_lda_fit_nan,
    "LatentDirichletAllocation",
    |_x: Array2<f64>| {
        LatentDirichletAllocation::new(2)
            .with_random_state(0)
            .fit(&x_nonneg_nan(), &())
            .map(|_| ())
    }
);
pin_fit_rejects!(
    divergence_dictionary_learning_fit_nan,
    "DictionaryLearning",
    |x: Array2<f64>| {
        DictionaryLearning::new(2)
            .with_random_state(0)
            .fit(&x, &())
            .map(|_| ())
    }
);
pin_fit_rejects!(divergence_mds_fit_nan, "MDS", |x: Array2<f64>| {
    MDS::new(2).fit(&x, &()).map(|_| ())
});
pin_fit_rejects!(divergence_isomap_fit_nan, "Isomap", |x: Array2<f64>| {
    Isomap::new(2).with_n_neighbors(4).fit(&x, &()).map(|_| ())
});
pin_fit_rejects!(
    divergence_spectral_embedding_fit_nan,
    "SpectralEmbedding",
    |x: Array2<f64>| { SpectralEmbedding::new(2).fit(&x, &()).map(|_| ()) }
);
pin_fit_rejects!(
    divergence_lle_fit_nan,
    "LocallyLinearEmbedding",
    |x: Array2<f64>| { LLE::new(2).with_n_neighbors(3).fit(&x, &()).map(|_| ()) }
);
pin_fit_rejects!(divergence_tsne_fit_nan, "TSNE", |x: Array2<f64>| {
    Tsne::new()
        .with_n_components(2)
        .with_perplexity(2.0)
        .with_random_state(0)
        .fit(&x, &())
        .map(|_| ())
});

// --- fit(X_nan, Y): cross-decomposition ---
pin_fit_rejects!(
    divergence_pls_regression_fit_nan,
    "PLSRegression",
    |x: Array2<f64>| {
        PLSRegression::<f64>::new(1)
            .fit(&x, &y_finite())
            .map(|_| ())
    }
);
pin_fit_rejects!(
    divergence_pls_canonical_fit_nan,
    "PLSCanonical",
    |x: Array2<f64>| { PLSCanonical::<f64>::new(1).fit(&x, &y_finite()).map(|_| ()) }
);
pin_fit_rejects!(divergence_pls_svd_fit_nan, "PLSSVD", |x: Array2<f64>| {
    PLSSVD::<f64>::new(1).fit(&x, &y_finite()).map(|_| ())
});
pin_fit_rejects!(divergence_cca_fit_nan, "CCA", |x: Array2<f64>| {
    CCA::<f64>::new(1).fit(&x, &y_finite()).map(|_| ())
});

// --- fit(X_inf): a representative Inf-side pin (sklearn raises the infinity
//     ValueError; SpectralEmbedding returns Ok(garbage)) ---
pin_fit_rejects!(
    divergence_spectral_embedding_fit_inf,
    "SpectralEmbedding(Inf)",
    |_x: Array2<f64>| { SpectralEmbedding::new(2).fit(&x_inf(), &()).map(|_| ()) }
);
pin_fit_rejects!(
    divergence_lle_fit_inf,
    "LocallyLinearEmbedding(Inf)",
    |_x: Array2<f64>| {
        LLE::new(2)
            .with_n_neighbors(3)
            .fit(&x_inf(), &())
            .map(|_| ())
    }
);

// --- transform(X_nan) after a finite fit: sklearn re-validates and raises ---
/// Divergence: transform(X_nan) after a finite fit does not reject. sklearn 1.5.2
/// re-runs `_validate_data(reset=False)` (force_all_finite=True) and raises
/// `ValueError("Input X contains NaN.")`. Tracking: #2290.
#[test]
fn divergence_incremental_pca_transform_nan() {
    let fitted = IncrementalPCA::<f64>::new(2)
        .fit(&x_finite(), &())
        .expect("finite fit");
    let r = fitted.transform(&x_nan()).map(|_| ());
    assert!(
        is_finiteness_rejection(&r),
        "IncrementalPCA.transform: expected finiteness rejection, got {r:?}"
    );
}

#[test]
fn divergence_sparse_pca_transform_nan() {
    let fitted = SparsePCA::<f64>::new(2)
        .with_random_state(0)
        .fit(&x_finite(), &())
        .expect("finite fit");
    let r = fitted.transform(&x_nan()).map(|_| ());
    assert!(
        is_finiteness_rejection(&r),
        "SparsePCA.transform: expected finiteness rejection, got {r:?}"
    );
}

#[test]
fn divergence_minibatch_nmf_transform_nan() {
    let fitted = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x_nonneg(), &())
        .expect("finite fit");
    let r = fitted.transform(&x_nonneg_nan()).map(|_| ());
    assert!(
        is_finiteness_rejection(&r),
        "MiniBatchNMF.transform: expected finiteness rejection, got {r:?}"
    );
}

#[test]
fn divergence_isomap_transform_nan() {
    let fitted = Isomap::new(2)
        .with_n_neighbors(4)
        .fit(&x_finite(), &())
        .expect("finite fit");
    let r = fitted.transform(&x_nan()).map(|_| ());
    assert!(
        is_finiteness_rejection(&r),
        "Isomap.transform: expected finiteness rejection, got {r:?}"
    );
}
