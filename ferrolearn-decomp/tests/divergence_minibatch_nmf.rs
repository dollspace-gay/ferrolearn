//! Divergence audit for `MiniBatchNMF` / `FittedMiniBatchNMF`
//! (`ferrolearn-decomp/src/minibatch_nmf.rs`) against scikit-learn 1.5.2
//! `class MiniBatchNMF(_BaseNMF)` (`sklearn/decomposition/_nmf.py:1792-2454`).
//!
//! ferrolearn is a SIMPLIFIED reimplementation (see `.design/decomp/minibatch_nmf.md`,
//! 3 SHIPPED / 13 NOT-STARTED, tracking #1485): `fit` uses 5-iter coordinate-descent
//! for W plus a plain multiplicative-update for H (no forget_factor/EWA aggregates)
//! plus deterministic `indices.rotate_left` batching (no random_state shuffle).
//! sklearn uses online MU with EWA aggregates A/B, `_rho = forget_factor**(batch/n)`,
//! NNDSVDa init, and a `_minibatch_convergence` EWA early-stop. The exact
//! `components_` VALUES are a CARVE-OUT (different algorithm and RNG); they are NOT
//! pinned here (REQ-4, #NNN-A).
//!
//! ## DIV-5 (transform = `_solve_W`) — investigated, NOT a meaningfully observable
//! divergence; folded into the value carve-out.
//!
//! sklearn `MiniBatchNMF.transform` (`_nmf.py:2351-2371`) returns
//! `W = _solve_W(X, components_, transform_max_iter)` (`_solve_W` `_nmf.py:2073-2098`):
//! W init = `np.full((n,k), sqrt(X.mean()/k))`, then `_multiplicative_update_w` for
//! `transform_max_iter` (= `max_iter`, default 200) iters with a relative-norm tol
//! stop, i.e. it solves `min_{W>=0} ||X - W·H||_F^2` to convergence given the FIXED
//! fitted H. ferrolearn `transform` (`minibatch_nmf.rs:507`) runs only 5 CD iters from
//! a constant `0.1` init (`minibatch_nmf.rs:519-524`).
//!
//! PROBE (R-CHAR-3, live sklearn 1.5.2 oracle, run from /tmp on ferrolearn's OWN
//! fitted H): for `MiniBatchNMF::<f64>::new(2).with_random_state(0)` fit on the 6x4 X
//! below, ferrolearn's fitted H is
//!   H = [[0.002000590859, 0.415299472773, 0.571432506353, 0.807828941622],
//!        [0.943162099770, 0.778664447370, 0.811750698705, 0.000827939107]].
//! Replicating `_solve_W` verbatim on that H,
//!   `from sklearn.decomposition._nmf import _multiplicative_update_w`
//!   `W=np.full((6,2), sqrt(X.mean()/2)); for _ in range(200): W,*_=_multiplicative_update_w(X,W,H,2,0,0,1.0)`,
//! gives residual `||X - W_oracle·H||_F = 4.598258351563`, while ferrolearn's
//! 5-iter-CD-from-0.1 transform gives `||X - W_ferro·H||_F = 4.598322126334`.
//!
//! These residuals AGREE to ~6e-5 (relative ~1.4e-5): ferrolearn's transform has
//! effectively CONVERGED to the same convex NNLS optimum for the fixed H. (Continuing
//! MU for 2000 more steps from ferro's own W moves the objective only 10.572283 ->
//! 10.571990 yet moves W coords by up to 0.022, a flat-valley / near-degenerate NNLS,
//! since H row0 col0 ~ 0.002. The elementwise W differs, but the OBSERVABLE fit quality
//! is identical.) Per dispatch and goal.md R-DEFER-3, an elementwise `transform ≈
//! oracle` pin would fail only on the flat-valley artifact, NOT on under-convergence,
//! so DIV-5 is NOT pinned as a standalone FIXABLE divergence; it folds into the REQ-4
//! value carve-out (#NNN-A). The residual-equivalence GREEN-GUARD below records this.
//!
//! Everything else asserted here is a STRUCTURAL GREEN-GUARD (REQ-1/2/3 SHIPPED): it
//! PASSES against current code and pins contracts the generator must not regress.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::MiniBatchNMF;
use ndarray::{Array2, array};

/// The fixed 6x4 non-negative fixture used across the transform probe / guards.
fn fixture_x() -> Array2<f64> {
    array![
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 5.0, 6.0, 1.0],
        [7.0, 8.0, 9.0, 2.0],
        [2.0, 1.0, 0.0, 3.0],
        [0.0, 3.0, 4.0, 5.0],
        [6.0, 2.0, 1.0, 0.0],
    ]
}

/// Frobenius norm of `X - W·H` (matches `reconstruction_error` semantics in the crate).
fn frob_residual(x: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> f64 {
    let wh = w.dot(h);
    let mut s = 0.0;
    for (a, b) in x.iter().zip(wh.iter()) {
        let d = *a - *b;
        s += d * d;
    }
    s.sqrt()
}

/// DIV-5 (NOT a divergence — recorded finding). ferrolearn `transform`
/// (`minibatch_nmf.rs:507`, 5-iter CD from 0.1) reaches the SAME observable NNLS fit
/// quality as sklearn `_solve_W` (`_nmf.py:2073-2098`, MU-to-`transform_max_iter` from
/// `sqrt(X.mean()/k)` init) for the FIXED fitted H.
///
/// Oracle (R-CHAR-3): live sklearn 1.5.2 `_solve_W` replicated verbatim on ferrolearn's
/// OWN fitted H (see module docs) -> `||X - W_oracle·H||_F = 4.598258351563`.
/// This GREEN-GUARD PASSES: ferro's transform residual is within 1e-3 of the converged
/// oracle residual, confirming the elementwise gap is a flat-valley artifact, NOT
/// under-convergence -> DIV-5 folds into the REQ-4 value carve-out (#NNN-A).
#[test]
fn div5_transform_residual_matches_solve_w_optimum() {
    /// Converged NNLS-optimal residual `||X - _solve_W(X,H,200)·H||_F`, live sklearn
    /// 1.5.2 `_solve_W` (`_nmf.py:2073-2098`) on ferrolearn's own fitted H (R-CHAR-3).
    const SK_SOLVE_W_RESIDUAL: f64 = 4.598258351563;

    let x = fixture_x();
    let nmf = MiniBatchNMF::<f64>::new(2).with_random_state(0);
    let fitted = nmf.fit(&x, &()).expect("fit");
    let h = fitted.components().clone();
    let w_ferro = fitted.transform(&x).expect("transform");

    let res_ferro = frob_residual(&x, &w_ferro, &h);

    // Observable fit quality matches the converged NNLS optimum to ~1e-4.
    let gap = (res_ferro - SK_SOLVE_W_RESIDUAL).abs();
    assert!(
        gap < 1e-3,
        "transform residual {res_ferro} should match sklearn _solve_W optimum \
         {SK_SOLVE_W_RESIDUAL} (gap {gap}); if this fails the transform is \
         under-converged and DIV-5 becomes a FIXABLE divergence"
    );
}

// ---------------------------------------------------------------------------
// GREEN-GUARDS — REQ-1/2/3 SHIPPED structural contracts (must PASS).
// ---------------------------------------------------------------------------

/// REQ-1: components H shape `(n_components, n_features)`; transform W shape
/// `(n_samples, n_components)` (`_nmf.py:2349`, `:2361`).
#[test]
fn guard_shapes() {
    let x = fixture_x();
    let fitted = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit");
    assert_eq!(fitted.components().dim(), (2, 4));
    let w = fitted.transform(&x).expect("transform");
    assert_eq!(w.dim(), (6, 2));
}

/// REQ-2: components AND transform are element-wise non-negative (NMF invariant,
/// `_nmf.py:1804`). sklearn `components_` / `W` are non-negative.
#[test]
fn guard_nonnegative() {
    let x = fixture_x();
    let fitted = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit");
    for &v in fitted.components().iter() {
        assert!(v >= 0.0, "negative component: {v}");
    }
    let w = fitted.transform(&x).expect("transform");
    for &v in w.iter() {
        assert!(v >= 0.0, "negative transform value: {v}");
    }
}

/// REQ-1: `reconstruction_err()` finite and `>= 0`; `n_iter()` in `[1, max_iter]`.
#[test]
fn guard_reconstruction_err_and_n_iter() {
    let x = fixture_x();
    let max_iter = 50;
    let fitted = MiniBatchNMF::<f64>::new(2)
        .with_max_iter(max_iter)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit");
    let err = fitted.reconstruction_err();
    assert!(err.is_finite(), "reconstruction_err not finite: {err}");
    assert!(err >= 0.0, "reconstruction_err negative: {err}");
    let n = fitted.n_iter();
    assert!(n >= 1 && n <= max_iter, "n_iter {n} not in [1, {max_iter}]");
}

/// REQ-1: same `random_state` -> identical components + transform across two fits.
#[test]
fn guard_determinism() {
    let x = fixture_x();
    let a = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit a");
    let b = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit b");
    assert_eq!(a.components(), b.components());
    let wa = a.transform(&x).expect("transform a");
    let wb = b.transform(&x).expect("transform b");
    assert_eq!(wa, wb);
}

/// REQ-3: error contracts in `fit` / `transform`
/// (`minibatch_nmf.rs:368-399`, `:509-515`).
#[test]
fn guard_error_contracts() {
    // n_components == 0 -> Err.
    let x = fixture_x();
    assert!(
        MiniBatchNMF::<f64>::new(0).fit(&x, &()).is_err(),
        "n_components==0 should Err"
    );

    // n_components > n_features (4) -> Err.
    assert!(
        MiniBatchNMF::<f64>::new(5).fit(&x, &()).is_err(),
        "n_components>n_features should Err"
    );

    // Negative input -> Err.
    let x_neg = array![[1.0, -2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 1.0]];
    assert!(
        MiniBatchNMF::<f64>::new(2).fit(&x_neg, &()).is_err(),
        "negative input should Err"
    );

    // 0 samples -> Err.
    let x_empty = Array2::<f64>::zeros((0, 4));
    assert!(
        MiniBatchNMF::<f64>::new(2).fit(&x_empty, &()).is_err(),
        "0 samples should Err"
    );

    // transform column-count mismatch -> Err.
    let fitted = MiniBatchNMF::<f64>::new(2)
        .with_random_state(0)
        .fit(&x, &())
        .expect("fit");
    let x_bad = array![[1.0, 2.0, 3.0]]; // 3 cols, fitted on 4
    assert!(
        fitted.transform(&x_bad).is_err(),
        "transform col mismatch should Err"
    );
}
