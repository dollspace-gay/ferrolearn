//! Divergence pin: MDS SMACOF convergence-criterion `norm==0` n_iter mismatch.
//!
//! sklearn `_smacof_single` (`sklearn/manifold/_mds.py:157-165`) computes
//! `dis = np.sqrt((X**2).sum(axis=1)).sum()` then tests
//! `(old_stress - stress / dis) < eps`. When the embedding collapses to the
//! origin (`X` all-zeros — e.g. an all-zero init, since the Guttman update
//! `X = (1/n) B @ 0 = 0` keeps it zero) `dis == 0`, so `stress / dis` is
//! `inf`/`nan`; the comparison `(nan) < eps` is **False**, so sklearn NEVER
//! breaks and runs the full `max_iter` iterations.
//!
//! ferrolearn `smacof_single` (`ferrolearn-decomp/src/mds.rs:470-476`) instead
//! special-cases the zero norm:
//! ```ignore
//! let normed = if norm != 0.0 { stress / norm } else { stress };
//! if let Some(prev) = old_stress && (prev - normed) < eps { break; }
//! ```
//! With `norm == 0` every iteration sets `normed = stress` (constant), so on the
//! 2nd iteration `prev - normed == 0.0 < eps` → ferrolearn breaks early.
//!
//! LIVE sklearn 1.5.2 oracle (R-CHAR-3), `D=[[0,2,3],[2,0,1.5],[3,1.5,0]]`,
//! `X0 = zeros((3,2))`, `max_iter=5`, `eps=1e-3`, metric=True:
//!   stress = 15.25, n_iter = 5, embedding = zeros((3,2)).
//! ferrolearn returns the SAME stress (15.25) and SAME embedding (zeros) but
//! `n_iter_ = 2`. The divergence is purely in the reported `n_iter_`.
//!
//! Tracking: #2407

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{Dissimilarity, MDS};
use ndarray::array;

#[test]
fn divergence_mds_norm0_niter() {
    // sklearn `_smacof_single` runs the FULL max_iter when the embedding
    // collapses to the origin (dis==0 -> nan comparison never breaks).
    const SK_N_ITER: usize = 5;
    const SK_STRESS: f64 = 15.25;

    let d = array![[0., 2., 3.], [2., 0., 1.5], [3., 1.5, 0.]];
    let x0 = array![[0., 0.], [0., 0.], [0., 0.]];

    let fitted = MDS::new(2)
        .with_dissimilarity(Dissimilarity::Precomputed)
        .with_init(x0)
        .with_max_iter(5)
        .with_eps(1e-3)
        .fit(&d, &())
        .expect("fit must not panic (R-CODE-2)");

    // Stress and embedding agree; only n_iter_ diverges.
    assert!(
        (fitted.stress() - SK_STRESS).abs() <= 1e-9,
        "stress {} vs sklearn {SK_STRESS}",
        fitted.stress()
    );
    assert_eq!(
        fitted.n_iter(),
        SK_N_ITER,
        "n_iter_ {} but sklearn runs full max_iter = {SK_N_ITER}",
        fitted.n_iter()
    );
}
