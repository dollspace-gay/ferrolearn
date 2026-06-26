//! Divergence guard for the public `ferrolearn_decomp::locally_linear_embedding`
//! helper against scikit-learn 1.5.2 `sklearn.manifold.locally_linear_embedding`.
//!
//! Scope: sklearn's standard dense path only:
//! `method="standard", eigen_solver="dense"`. Hessian, modified LLE, LTSA, and
//! ARPACK solver support remain tracked separately from this helper.

use ferrolearn_core::traits::Fit;
use ferrolearn_decomp::{LLE, locally_linear_embedding};
use ndarray::{Array2, array};

fn fixture() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [2.0, 0.3, 0.1],
        [3.0, 0.2, 0.0],
        [0.5, 1.0, 0.2],
        [1.5, 1.1, 0.1],
        [2.5, 0.9, 0.3],
        [3.5, 1.2, 0.2],
        [0.2, 2.0, 0.0],
        [1.2, 2.1, 0.1],
    ]
}

fn sign_align_column(col: &mut [f64]) {
    let mut max_abs = 0.0_f64;
    let mut max_val = 0.0_f64;
    for &v in col.iter() {
        if v.abs() > max_abs {
            max_abs = v.abs();
            max_val = v;
        }
    }
    if max_val < 0.0 {
        for v in col.iter_mut() {
            *v = -*v;
        }
    }
}

fn sign_robust_max_err(emb: &Array2<f64>, sklearn: &[Vec<f64>]) -> f64 {
    let n = sklearn.len();
    let nc = sklearn[0].len();
    let mut ferro: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nc).map(|c| emb[[i, c]]).collect())
        .collect();
    let mut sk = sklearn.to_vec();

    for c in 0..nc {
        let mut fcol: Vec<f64> = (0..n).map(|i| ferro[i][c]).collect();
        let mut scol: Vec<f64> = (0..n).map(|i| sk[i][c]).collect();
        sign_align_column(&mut fcol);
        sign_align_column(&mut scol);
        for i in 0..n {
            ferro[i][c] = fcol[i];
            sk[i][c] = scol[i];
        }
    }

    let mut max_err = 0.0_f64;
    for i in 0..n {
        for c in 0..nc {
            let err = (ferro[i][c] - sk[i][c]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

#[allow(
    clippy::excessive_precision,
    reason = "sklearn 1.5.2 live-oracle embedding and error are pinned at full f64 precision"
)]
#[test]
fn divergence_locally_linear_embedding_standard_dense() {
    let sklearn: Vec<Vec<f64>> = vec![
        vec![0.24567157527665856, 0.5674039617688269],
        vec![0.01227528485688903, 0.40123298809523117],
        vec![-0.20867547621436816, 0.18628911468361917],
        vec![-0.4724819020555504, 0.10606641432903302],
        vec![0.2731058138705527, 0.08680628848070257],
        vec![0.04109517643360715, -0.08152757197378396],
        vec![-0.24079917531930792, -0.1404871440512273],
        vec![-0.44342955190913846, -0.3773298791179383],
        vec![0.5132534241683719, -0.28842866259481065],
        vec![0.27998483080709957, -0.46002550961679467],
    ];
    let expected_error = 0.0001765118939348609_f64;

    let x = fixture();
    let (embedding, reconstruction_error) =
        locally_linear_embedding(&x, 4, 2, 1e-3).expect("standard LLE helper should fit");

    assert_eq!(embedding.dim(), (10, 2), "embedding shape mismatch");
    let max_err = sign_robust_max_err(&embedding, &sklearn);
    assert!(
        max_err < 1e-6,
        "sign-aligned |ferrolearn - sklearn| max err {max_err:.3e} exceeds 1e-6"
    );
    assert!(
        (reconstruction_error - expected_error).abs() < 1e-12,
        "reconstruction error {reconstruction_error:.17e} differs from sklearn \
         {expected_error:.17e}"
    );
}

#[test]
fn standalone_helper_matches_lle_fit_surface() {
    let x = fixture();
    let fitted = LLE::new(2)
        .with_n_neighbors(4)
        .with_reg(1e-3)
        .fit(&x, &())
        .expect("LLE::fit should succeed on fixture");
    let (embedding, reconstruction_error) =
        locally_linear_embedding(&x, 4, 2, 1e-3).expect("helper should succeed on fixture");

    assert_eq!(&embedding, fitted.embedding());
    assert!(
        (reconstruction_error - fitted.reconstruction_error()).abs() < 1e-15,
        "helper and fitted reconstruction errors diverged"
    );
}

#[test]
fn standalone_helper_reuses_lle_validation_contracts() {
    let x = fixture();

    assert!(locally_linear_embedding(&x, 0, 2, 1e-3).is_err());
    assert!(locally_linear_embedding(&x, 4, 0, 1e-3).is_err());
    assert!(locally_linear_embedding(&x, 4, 2, -1.0).is_err());
    assert!(locally_linear_embedding(&array![[1.0, 2.0]], 1, 1, 1e-3).is_err());
}
