//! Divergence / structural green-guard suite for `DictionaryLearning`
//! against scikit-learn 1.5.2
//! `class DictionaryLearning(_BaseSparseCoding, BaseEstimator)`
//! (`sklearn/decomposition/_dict_learning.py:1372`).
//!
//! Design doc: `.design/decomp/dictionary_learning.md` (3 SHIPPED /
//! 13 NOT-STARTED). Tracking issue #1512.
//!
//! ferrolearn is a SIMPLIFIED reimplementation. Its `components_` VALUES are a
//! CARVE-OUT (R-DEFER-3) driven by THREE algorithmic differences from sklearn:
//!   (1) init — random Gaussian `Xoshiro256PlusPlus` + normalize
//!       (`dictionary_learning.rs:530-536`) vs sklearn SVD-based `dict_init=None`
//!       (`_dict_learning.py:581-584`, deterministic);
//!   (2) sparse coder — ferrolearn hand-rolled soft-threshold CD
//!       (`lasso_cd_single:271`) vs sklearn default `fit_algorithm="lars"`
//!       → `lasso_lars`;
//!   (3) dict update — ferrolearn normal-equations LS `(AᵀA+1e-10I)D=AᵀX`
//!       + normalize (`:554-580`) vs sklearn `_update_dict` BCD + unused-atom
//!       resampling (`_dict_learning.py:474-551`, numpy RandomState).
//!
//! So component VALUES are NOT pinned here. This file GREEN-GUARDS the 3
//! structural SHIPPED REQs (REQ-1/2/3) — every test below MUST PASS against the
//! current code. Expected structural facts (shapes, unit-norm, sparsity cap,
//! determinism, error contracts) come from the live sklearn 1.5.2 oracle (the
//! probes in the design doc, run from /tmp), never literal-copied from
//! ferrolearn (R-CHAR-3).

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::{DictTransformAlgorithm, DictionaryLearning};
use ndarray::Array2;

/// Fixed 6x4 design matrix matching the design-doc oracle probes
/// (`.design/decomp/dictionary_learning.md` Probe 1/4).
fn probe_x() -> Array2<f64> {
    Array2::<f64>::from_shape_vec(
        (6, 4),
        vec![
            1., 2., 3., 0., //
            4., 5., 6., 1., //
            7., 8., 9., 2., //
            2., 1., 0., 3., //
            0., 3., 4., 1., //
            3., 0., 1., 2., //
        ],
    )
    .expect("6x4 literal data is well-formed")
}

// -------------------------------------------------------------------------
// REQ-1 (SHIPPED scoped) — structural shapes / OMP-cap / finite err / n_iter
// -------------------------------------------------------------------------

/// Green-guard (REQ-1): `components_` has shape `(n_components, n_features)`.
/// sklearn oracle (Probe 1): `DictionaryLearning(n_components=3, alpha=1,
/// random_state=0).fit(X).components_.shape == (3, 4)`
/// (`_dict_learning.py:1501`, `components_ = U` `:1699`). STRUCTURAL ONLY —
/// the component VALUES are the carve-out (REQ-4), not pinned.
#[test]
fn green_components_shape_matches_sklearn() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(20)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    // sklearn: (n_components, n_features) = (3, 4).
    assert_eq!(fitted.components().dim(), (3, 4));
}

/// Green-guard (REQ-1): `transform` codes have shape
/// `(n_samples, n_components)`. sklearn oracle (Probe 4):
/// `m.transform(X).shape == (6, 3)`
/// (`_BaseSparseCoding._transform` `_dict_learning.py:1110-1139`).
#[test]
fn green_transform_codes_shape_matches_sklearn() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(20)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let codes = fitted.transform(&x).expect("transform on fit X");
    assert_eq!(codes.dim(), (6, 3));
}

/// Green-guard (REQ-1): OMP transform respects `transform_n_nonzero_coefs` cap.
/// sklearn oracle (Probe 4): with `transform_algorithm='omp',
/// transform_n_nonzero_coefs=2`, `max nnz per row == 2`
/// (`orthogonal_mp_gram(n_nonzero_coefs=...)` `_dict_learning.py:192-200`).
/// So ferrolearn's OMP must produce <= 2 non-zeros per row.
#[test]
fn green_omp_respects_nonzero_cap() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(20)
        .with_transform_algorithm(DictTransformAlgorithm::Omp)
        .with_transform_n_nonzero_coefs(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let codes = fitted.transform(&x).expect("OMP transform");
    for i in 0..codes.nrows() {
        let nnz = codes.row(i).iter().filter(|&&v| v.abs() > 1e-12).count();
        assert!(nnz <= 2, "row {i}: OMP produced {nnz} nnz, cap is 2");
    }
}

/// Green-guard (REQ-1): LassoCd transform produces some exact zeros (the L1
/// prox induces sparsity; sklearn `lasso_cd` -> `Lasso` `_dict_learning.py:146`
/// zeroes coefficients below the soft-threshold). At alpha=2 with a 8-atom
/// overcomplete dictionary at least one code entry must be exactly zero.
#[test]
fn green_lasso_cd_transform_has_zeros() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(8)
        .with_alpha(2.0)
        .with_max_iter(20)
        .with_transform_algorithm(DictTransformAlgorithm::LassoCd)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let codes = fitted.transform(&x).expect("LassoCd transform");
    let zeros = codes.iter().filter(|&&v| v.abs() < 1e-12).count();
    assert!(zeros > 0, "LassoCd codes should contain exact zeros, got 0");
}

/// Green-guard (REQ-1): `reconstruction_err()` is finite and `>= 0`, and
/// `n_iter()` is in `[1, max_iter]`. sklearn's `n_iter_` is the alternating
/// iteration count (`_dict_learning.py:670-674`); ferrolearn caps at `max_iter`.
#[test]
fn green_reconstruction_err_finite_and_n_iter_in_range() {
    let x = probe_x();
    let max_iter = 25;
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(max_iter)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let err = fitted.reconstruction_err();
    assert!(err.is_finite() && err >= 0.0, "reconstruction_err = {err}");
    let n = fitted.n_iter();
    assert!(
        (1..=max_iter).contains(&n),
        "n_iter = {n} must be in [1, {max_iter}]"
    );
}

// -------------------------------------------------------------------------
// REQ-2 (SHIPPED scoped) — dictionary atoms UNIT L2-NORM
// -------------------------------------------------------------------------

/// Green-guard (REQ-2): every row of `components_` has unit L2 norm (tol 1e-6).
/// sklearn oracle (Probe 1): atom L2 norms == `[1.0, 1.0, 1.0]`.
///
/// FLAG (folds into the value carve-out, REQ-7): sklearn `_update_dict`
/// projects each atom onto the unit BALL `dictionary[k] /= max(norm, 1)`
/// (`_dict_learning.py:548`), so a converged sklearn atom has norm <= 1 (here
/// exactly 1 because all atoms are active). ferrolearn `normalise_dictionary`
/// (`dictionary_learning.rs:254-265`) always projects onto the unit SPHERE
/// (`/= norm`), so norm == 1 unconditionally. We green-guard ferrolearn's
/// actual behavior (norm == 1); the constraint difference is documented, not
/// pinned (carve-out-gated).
#[test]
fn green_atoms_unit_l2_norm() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(20)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let d = fitted.components();
    for k in 0..d.nrows() {
        let norm: f64 = d.row(k).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "atom {k}: L2 norm {norm} (sklearn oracle: 1.0)"
        );
    }
}

// -------------------------------------------------------------------------
// Determinism (REQ-1) — same random_state => identical fit
// -------------------------------------------------------------------------

/// Green-guard (REQ-1): two fits with the same `random_state` produce
/// element-wise identical `components_` AND identical transform codes.
/// sklearn `DictionaryLearning` is deterministic given `random_state`
/// (`_dict_learning.py:1681-1698` threads the RNG through `_dict_learning`);
/// ferrolearn must likewise be reproducible given a seed.
#[test]
fn green_determinism_same_seed() {
    let x = probe_x();
    let build = || {
        DictionaryLearning::new(3)
            .with_alpha(1.0)
            .with_max_iter(20)
            .with_random_state(7)
            .fit(&x, &())
            .expect("fit on well-formed 6x4 X")
    };
    let a = build();
    let b = build();
    let da = a.components();
    let db = b.components();
    assert_eq!(da.dim(), db.dim());
    for (x, y) in da.iter().zip(db.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "components differ across fits");
    }
    let ca = a.transform(&x).expect("transform a");
    let cb = b.transform(&x).expect("transform b");
    for (x, y) in ca.iter().zip(cb.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "transform codes differ");
    }
}

// -------------------------------------------------------------------------
// REQ-3 (SHIPPED scoped) — error / parameter contracts
// -------------------------------------------------------------------------

/// Green-guard (REQ-3): `n_components == 0` -> `Err`.
/// (sklearn raises `InvalidParameterError`; ferrolearn returns `FerroError` —
/// the error TYPE differs, documented in the design doc REQ-3 FLAG, not pinned.)
#[test]
fn green_err_n_components_zero() {
    let x = probe_x();
    assert!(DictionaryLearning::new(0).fit(&x, &()).is_err());
}

/// Green-guard (REQ-3): `n_samples == 0` -> `Err`.
#[test]
fn green_err_zero_samples() {
    let x = Array2::<f64>::zeros((0, 4));
    assert!(DictionaryLearning::new(2).fit(&x, &()).is_err());
}

/// Green-guard (REQ-3): `n_features == 0` -> `Err`.
#[test]
fn green_err_zero_features() {
    let x = Array2::<f64>::zeros((6, 0));
    assert!(DictionaryLearning::new(2).fit(&x, &()).is_err());
}

/// Green-guard (REQ-3): `alpha < 0` -> `Err`.
/// sklearn `_parameter_constraints`: `alpha` in `[0, inf)`
/// (`_dict_learning.py:1565-1586`).
#[test]
fn green_err_alpha_negative() {
    let x = probe_x();
    assert!(
        DictionaryLearning::new(3)
            .with_alpha(-1.0)
            .fit(&x, &())
            .is_err()
    );
}

/// Green-guard (REQ-3): `transform` with a column-count mismatch -> `Err`.
#[test]
fn green_err_transform_col_mismatch() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_max_iter(10)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on well-formed 6x4 X");
    let x_bad = Array2::<f64>::zeros((5, 3)); // 3 cols != 4
    assert!(fitted.transform(&x_bad).is_err());
}

// -------------------------------------------------------------------------
// REQ-4 — component VALUE CARVE-OUT (documentation, NOT a pin; R-DEFER-3)
// -------------------------------------------------------------------------

/// CARVE-OUT WITNESS (REQ-4, R-DEFER-3) — NOT a divergence pin.
///
/// This test does NOT assert sklearn component values. It asserts only that
/// ferrolearn's fit is internally well-formed (finite, unit-norm atoms) for the
/// exact oracle probe input. The sklearn oracle for this input is
/// `components_ row0 = [0.487906, 0.566806, 0.650717, 0.131324]`
/// (`.design/decomp/dictionary_learning.md` Probe 1); ferrolearn does NOT
/// reproduce it element-wise (different init/solver/update + RNG —
/// `dictionary_learning.rs:530-580` vs `_dict_learning.py:554-674`). Per
/// R-DEFER-3 the component values are carved out and intentionally NOT pinned.
/// Asserting equality here would be a false divergence (the divergence is the
/// whole algorithm choice, tracked NOT-STARTED). Kept as a witness that the fit
/// runs and is structurally valid on the oracle input.
#[test]
fn carveout_components_not_value_pinned() {
    let x = probe_x();
    let fitted = DictionaryLearning::new(3)
        .with_alpha(1.0)
        .with_max_iter(20)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit on the oracle probe input");
    // Internal well-formedness only — NO sklearn value comparison (carve-out).
    assert!(fitted.components().iter().all(|v| v.is_finite()));
    assert_eq!(fitted.components().dim(), (3, 4));
}
