//! Conformance tests for ferrolearn-decomp vs scikit-learn.
//!
//! Each test loads a fixture from `fixtures/<name>.json`, runs the matching
//! ferrolearn estimator with the same hyperparameters, and compares outputs
//! through `ferrolearn-test-oracle` helpers.
//!
//! ## Comparison strategy
//!
//! PCA components are **sign-ambiguous** on a per-row basis — the SVD that
//! produces them is determined only up to a sign per principal axis. The
//! transformed coordinates inherit the same per-column ambiguity. Mean and
//! explained-variance are scalar/positive and are compared directly.
//!
//! NMF factorizations W·H are non-unique (any invertible diagonal scaling
//! `D` makes `W·D⁻¹` and `D·H` an equivalent factorization). Comparing W
//! and H element-wise is therefore meaningless across libraries; we instead
//! compare the deterministic scalar **reconstruction_error** tightly, plus
//! the non-negativity / shape invariants of the returned factors.

use ferrolearn_core::{Fit, Transform};
use ferrolearn_test_oracle::{
    TOL_DECOMP_ABS, TOL_DECOMP_REL, assert_close, assert_close_rows_sign_ambiguous,
    assert_close_slice, assert_close_slice_sign_ambiguous, json_to_array1, json_to_array2,
    load_fixture,
};

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

#[test]
fn conformance_pca() {
    let fx = load_fixture("pca");
    let x = json_to_array2(&fx.input["X"]);
    let (rel, abs) = fx.tolerance(TOL_DECOMP_REL, TOL_DECOMP_ABS);

    let n_components = fx.params["n_components"].as_u64().unwrap() as usize;
    let pca = ferrolearn_decomp::PCA::<f64>::new(n_components);
    let fitted = pca.fit(&x, &()).expect("PCA fit");

    // Mean: not sign-ambiguous, deterministic.
    let expected_mean = json_to_array1(&fx.expected["mean"]);
    assert_close_slice(
        fitted.mean().as_slice().unwrap(),
        expected_mean.as_slice().unwrap(),
        rel,
        abs,
        "PCA.mean",
    );

    // Explained variance and ratio: positive scalars per component, not
    // sign-ambiguous and ordered by magnitude (descending) in both libs.
    let expected_ev = json_to_array1(&fx.expected["explained_variance"]);
    assert_close_slice(
        fitted.explained_variance().as_slice().unwrap(),
        expected_ev.as_slice().unwrap(),
        rel,
        abs,
        "PCA.explained_variance",
    );
    let expected_evr = json_to_array1(&fx.expected["explained_variance_ratio"]);
    assert_close_slice(
        fitted.explained_variance_ratio().as_slice().unwrap(),
        expected_evr.as_slice().unwrap(),
        rel,
        abs,
        "PCA.explained_variance_ratio",
    );

    // Components: each row is sign-ambiguous (SVD U/V sign is not uniquely
    // determined). Use the dedicated row-wise sign-aware comparator.
    let expected_components = json_to_array2(&fx.expected["components"]);
    let actual_components = fitted.components();
    assert_eq!(
        actual_components.shape(),
        expected_components.shape(),
        "PCA.components shape"
    );
    assert_close_rows_sign_ambiguous(
        actual_components,
        &expected_components,
        rel,
        abs,
        "PCA.components",
    );

    // Transformed coordinates: column j inherits the sign of components row j.
    let expected_transformed = json_to_array2(&fx.expected["transformed"]);
    let transformed = fitted.transform(&x).expect("PCA transform");
    assert_eq!(
        transformed.shape(),
        expected_transformed.shape(),
        "PCA.transform shape"
    );
    for col in 0..transformed.ncols() {
        let actual_col: Vec<f64> = transformed.column(col).to_vec();
        let expected_col: Vec<f64> = expected_transformed.column(col).to_vec();
        assert_close_slice_sign_ambiguous(
            &actual_col,
            &expected_col,
            rel,
            abs,
            &format!("PCA.transform[col {col}]"),
        );
    }
}

// ---------------------------------------------------------------------------
// NMF
// ---------------------------------------------------------------------------

#[test]
fn conformance_nmf() {
    let fx = load_fixture("nmf");
    let x = json_to_array2(&fx.input["X"]);
    // NMF reconstruction error has small algorithmic divergences from
    // sklearn's coordinate-descent solver — allow a 5% relative envelope
    // around the scalar, which is far stricter than the legacy 2×/0.5× gate
    // but loose enough to absorb solver-path differences without losing the
    // signal that the two factorizations land on comparable Frobenius norms.
    let (rel, abs) = fx.tolerance(5e-2, 1e-6);

    let n_components = fx.params["n_components"].as_u64().unwrap() as usize;
    let init_str = fx.params["init"].as_str().unwrap_or("nndsvd");
    let random_state = fx.params["random_state"].as_u64().unwrap_or(42);
    // ferrolearn currently only implements the two canonical NMF inits
    // (`Nndsvd` and `Random`). Fixture variants `nndsvda` / `nndsvdar` map
    // onto the closest available — `Nndsvd` — which is the same family of
    // SVD-seeded initialization.
    let init = match init_str {
        "nndsvd" | "nndsvda" | "nndsvdar" => ferrolearn_decomp::NMFInit::Nndsvd,
        "random" => ferrolearn_decomp::NMFInit::Random,
        other => panic!("unsupported NMF init in fixture: {other}"),
    };

    let nmf = ferrolearn_decomp::NMF::<f64>::new(n_components)
        .with_init(init)
        .with_random_state(random_state)
        .with_max_iter(500);
    let fitted = nmf.fit(&x, &()).expect("NMF fit");

    // Shape / non-negativity invariants on H.
    let h = fitted.components();
    assert_eq!(h.nrows(), n_components, "NMF.H rows");
    assert_eq!(h.ncols(), x.ncols(), "NMF.H cols");
    for &v in h {
        assert!(v >= 0.0, "NMF.H contains negative value {v}");
    }

    // W = transform(X). Non-negative, correct shape.
    let w = fitted.transform(&x).expect("NMF transform");
    assert_eq!(w.nrows(), x.nrows(), "NMF.W rows");
    assert_eq!(w.ncols(), n_components, "NMF.W cols");
    for &v in &w {
        assert!(v >= 0.0, "NMF.W contains negative value {v}");
    }

    // Reconstruction error: deterministic scalar — the primary conformance
    // signal. W and H themselves are non-unique under diagonal rescaling so
    // we do *not* compare them element-wise.
    let expected_err = fx.expected["reconstruction_error"].as_f64().unwrap();
    assert_close(
        fitted.reconstruction_err(),
        expected_err,
        rel,
        abs,
        "NMF.reconstruction_error",
    );
}
