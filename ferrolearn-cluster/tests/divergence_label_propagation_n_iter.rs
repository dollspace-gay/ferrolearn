//! Divergence pin: `LabelPropagation::n_iter_` on the max_iter (non-convergence)
//! path diverges from scikit-learn 1.5.2.
//!
//! Expected values come from the LIVE sklearn 1.5.2 oracle (R-CHAR-3), never
//! literal-copied from ferrolearn.
//!
//! sklearn `sklearn/semi_supervised/_label_propagation.py:300-326`:
//! ```python
//! for self.n_iter_ in range(self.max_iter):
//!     if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
//!         break
//!     ...
//! else:
//!     warnings.warn("max_iter=%d was reached without convergence." ...)
//!     self.n_iter_ += 1
//! ```
//! On the no-break exit (max_iter reached), Python's `for self.n_iter_ in
//! range(max_iter)` leaves `n_iter_ == max_iter - 1`; the `else` branch then
//! does `self.n_iter_ += 1`, so `n_iter_ == max_iter` EXACTLY.
//!
//! ferrolearn `label_propagation.rs:560-563`:
//! ```rust
//! if !converged {
//!     n_iter = max_iter.saturating_add(1);   // == max_iter + 1  (off-by-one)
//! }
//! ```
//! ferrolearn returns `max_iter + 1` instead of `max_iter`.

use ferrolearn_cluster::LabelPropagation;
use ferrolearn_core::Fit;
use ndarray::{Array1, Array2};

/// REQ-3: `n_iter_` on the max_iter (non-convergence) path must equal
/// `max_iter`, matching sklearn's `for/else: n_iter_ += 1` semantics.
///
/// Live sklearn 1.5.2 oracle (system python3, sklearn 1.5.2 / numpy 2.4.5):
/// ```text
/// import numpy as np
/// from sklearn.semi_supervised import LabelPropagation
/// X=np.array([[0.,0.],[0.3,0.],[0.6,0.],[1.0,0.]]); y=np.array([0,-1,-1,1])
/// for mi in (2,3,5):
///     m=LabelPropagation(gamma=1.0,tol=1e-12,max_iter=mi).fit(X,y)
///     print(mi, m.n_iter_)
/// # -> 2 2 / 3 3 / 5 5   (n_iter_ == max_iter, NOT max_iter+1)
/// ```
/// The tiny `tol=1e-12` guarantees the slow-converging gamma=1 graph never hits
/// the L1-at-start break before `max_iter`, forcing the non-convergence branch.
#[test]
fn divergence_n_iter_max_iter_hit_equals_max_iter() {
    // sklearn oracle: n_iter_ == max_iter on the non-convergence exit.
    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0.3, 0., 0.6, 0., 1.0, 0.]).unwrap();
    let y = Array1::from_vec(vec![0, -1, -1, 1]);

    for &max_iter in &[2usize, 3, 5] {
        let sk_n_iter = max_iter; // live-oracle value (see doc comment)

        let fitted = LabelPropagation::<f64>::new()
            .with_gamma(1.0)
            .with_tol(1e-12)
            .with_max_iter(max_iter)
            .fit(&x, &y)
            .unwrap();

        assert_eq!(
            fitted.n_iter(),
            sk_n_iter,
            "REQ-3: on max_iter={max_iter} (non-convergence) sklearn n_iter_ == {sk_n_iter}, \
             ferrolearn returned {}",
            fitted.n_iter()
        );
    }
}

/// Green-guard breadth (PASS today): `label_distributions_` VALUE parity on the
/// default gamma=20 2-class fixture and a soft 3-class gamma=0.5 fixture.
///
/// Live sklearn 1.5.2 oracle:
/// ```text
/// XC=np.array([[0.,0.],[0.1,0.],[5.,5.],[5.1,5.],[0.,5.],[0.1,5.]])
/// yC=np.array([0,-1,1,-1,2,-1])
/// m=LabelPropagation(gamma=0.5,tol=1e-3).fit(XC,yC)
/// m.label_distributions_ ->
///  [[1,0,0],
///   [0.999994405367, 0.000000000041, 0.000005594592],
///   [0,1,0],
///   [0.000000000022, 0.999995871697, 0.000004128281],
///   [0,0,1],
///   [0.000005594564, 0.000008012093, 0.999986393343]]
/// m.n_iter_ -> 2
/// ```
#[test]
fn green_label_distributions_3class_soft_g05() {
    const SK_LD: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.999_994_405_367, 0.000_000_000_041, 0.000_005_594_592],
        [0.0, 1.0, 0.0],
        [0.000_000_000_022, 0.999_995_871_697, 0.000_004_128_281],
        [0.0, 0.0, 1.0],
        [0.000_005_594_564, 0.000_008_012_093, 0.999_986_393_343],
    ];
    const SK_N_ITER: usize = 2;

    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0., 0., 0.1, 0., 5., 5., 5.1, 5., 0., 5., 0.1, 5.],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, -1, 1, -1, 2, -1]);

    let fitted = LabelPropagation::<f64>::new()
        .with_gamma(0.5)
        .with_tol(1e-3)
        .fit(&x, &y)
        .unwrap();

    let ld = fitted.label_distributions();
    for (i, row) in SK_LD.iter().enumerate() {
        for (c, &sk) in row.iter().enumerate() {
            assert!(
                (ld[[i, c]] - sk).abs() < 1e-6,
                "ld[{i},{c}]={} sklearn={sk}",
                ld[[i, c]]
            );
        }
    }
    assert_eq!(fitted.n_iter(), SK_N_ITER);
}
