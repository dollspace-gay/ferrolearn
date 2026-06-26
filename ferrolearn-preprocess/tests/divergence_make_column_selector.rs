//! Divergence guard for `make_column_selector` against scikit-learn 1.5.2
//! `sklearn.compose.make_column_selector`.
//!
//! Scope: ferrolearn has dense `Array2<f64>` inputs with no dataframe column
//! labels or mixed dtypes. The supported analogue is sklearn's
//! `make_column_selector(dtype_include=np.number)` applied to an all-numeric
//! pandas DataFrame, which selects every numeric column in ascending order.
//! Regex pattern, dtype include/exclude, string labels, boolean masks, and
//! callable dataframe selectors remain tracked residual gaps.
//!
//! Oracle reproduction:
//! ```text
//! import numpy as np, pandas as pd
//! from sklearn.compose import ColumnTransformer, make_column_selector
//! from sklearn.preprocessing import StandardScaler
//! X = pd.DataFrame([[1., 10., 100.], [2., 20., 200.], [3., 30., 300.]])
//! selector = make_column_selector(dtype_include=np.number)
//! print(selector(X))
//! print(ColumnTransformer([("std", StandardScaler(), selector)]).fit_transform(X).tolist())
//! ```

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_preprocess::make_column_selector;
use ferrolearn_preprocess::{ColumnSelector, ColumnTransformer, Remainder, StandardScaler};
use ndarray::{Array2, array};

const TOL: f64 = 1e-12;

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let diff = (actual[[i, j]] - expected[[i, j]]).abs();
            assert!(
                diff <= TOL,
                "entry ({i},{j}): expected {}, got {}, diff {diff} > {TOL}",
                expected[[i, j]],
                actual[[i, j]]
            );
        }
    }
}

#[test]
fn make_column_selector_dense_numeric_returns_all_indices() {
    let x = array![[1.0_f64, 10.0, 100.0], [2.0, 20.0, 200.0]];
    assert_eq!(
        make_column_selector(&x),
        ColumnSelector::Indices(vec![0, 1, 2])
    );

    let zero_feature = Array2::<f64>::zeros((2, 0));
    assert_eq!(
        make_column_selector(&zero_feature),
        ColumnSelector::Indices(Vec::new())
    );
}

#[test]
fn make_column_selector_inside_column_transformer_matches_sklearn() {
    let x = array![
        [1.0_f64, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
    ];
    let selector = make_column_selector(&x);
    let ct = ColumnTransformer::new(
        vec![(
            "std".into(),
            Box::new(StandardScaler::<f64>::new()),
            selector,
        )],
        Remainder::Drop,
    );
    let fitted = ct.fit(&x, &()).unwrap();
    let actual = fitted.transform(&x).unwrap();

    // sklearn 1.5.2 oracle:
    // ColumnTransformer([("std", StandardScaler(),
    //     make_column_selector(dtype_include=np.number))]).fit_transform(X)
    let expected = array![
        [-1.224744871391589, -1.224744871391589, -1.224744871391589],
        [0.0, 0.0, 0.0],
        [1.224744871391589, 1.224744871391589, 1.224744871391589],
    ];
    assert_matrix_close(&actual, &expected);
}
