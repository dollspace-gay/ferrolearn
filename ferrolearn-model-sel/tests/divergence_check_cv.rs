//! Public helper parity for `ferrolearn_model_sel::check_cv`.
//!
//! Oracle runtime: sklearn 1.5.2 in this workspace.
//!
//! Python oracle:
//! ```python
//! import numpy as np
//! from sklearn.model_selection import check_cv
//! y_binary = np.array([0,1,0,1,0,0,1,1,1])
//! y_multi = np.array([0,1,0,1,2,1,2,0,2])
//! for cv, y, classifier, n in [
//!     (3, None, False, 9),
//!     (None, None, False, 10),
//!     (3, None, True, 9),
//!     (3, y_binary, True, len(y_binary)),
//!     (3, y_multi, True, len(y_multi)),
//! ]:
//!     checked = check_cv(cv, y, classifier=classifier)
//!     print(type(checked).__name__,
//!           [[int(i) for i in test] for _, test in checked.split(np.ones(n), y)])
//! ```

use ferrolearn_model_sel::{CheckedCv, check_cv};
use ndarray::array;

fn test_folds(folds: &[(Vec<usize>, Vec<usize>)]) -> Vec<Vec<usize>> {
    folds.iter().map(|(_, test)| test.clone()).collect()
}

#[test]
fn check_cv_int_regressor_matches_kfold() {
    let cv = check_cv(Some(3), None, false).expect("check_cv");
    assert!(!cv.is_stratified());
    assert_eq!(cv.n_splits(), 3);
    assert!(matches!(cv, CheckedCv::KFold(_)));

    let folds = cv.split(9, None).expect("split");
    assert_eq!(
        test_folds(&folds),
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]
    );
}

#[test]
fn check_cv_none_defaults_to_five_fold_kfold() {
    let cv = check_cv(None, None, false).expect("check_cv");
    assert_eq!(cv.n_splits(), 5);

    let folds = cv.split(10, None).expect("split");
    assert_eq!(
        test_folds(&folds),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7], vec![8, 9]]
    );
}

#[test]
fn check_cv_classifier_without_y_falls_back_to_kfold() {
    let cv = check_cv(Some(3), None, true).expect("check_cv");
    assert!(!cv.is_stratified());
    assert!(matches!(cv, CheckedCv::KFold(_)));

    let folds = cv.split(9, None).expect("split");
    assert_eq!(
        test_folds(&folds),
        vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]]
    );
}

#[test]
fn check_cv_binary_classifier_matches_stratified_kfold() {
    let y = array![0usize, 1, 0, 1, 0, 0, 1, 1, 1];
    let cv = check_cv(Some(3), Some(&y), true).expect("check_cv");
    assert!(cv.is_stratified());
    assert!(matches!(cv, CheckedCv::StratifiedKFold(_)));

    let folds = cv.split(y.len(), Some(&y)).expect("split");
    assert_eq!(
        test_folds(&folds),
        vec![vec![0, 1, 2], vec![3, 4, 6], vec![5, 7, 8]]
    );
}

#[test]
fn check_cv_multiclass_classifier_matches_stratified_kfold() {
    let y = array![0usize, 1, 0, 1, 2, 1, 2, 0, 2];
    let cv = check_cv(Some(3), Some(&y), true).expect("check_cv");
    assert!(cv.is_stratified());

    let folds = cv.split(y.len(), Some(&y)).expect("split");
    assert_eq!(
        test_folds(&folds),
        vec![vec![0, 1, 4], vec![2, 3, 6], vec![5, 7, 8]]
    );
}

#[test]
fn check_cv_validation_errors() {
    let err = check_cv(Some(1), None, false).expect_err("cv < 2 must be rejected");
    assert!(format!("{err}").contains("cv"));

    let y = array![0usize, 1, 0, 1, 0, 1];
    let cv = check_cv(Some(3), Some(&y), true).expect("check_cv");
    let err = cv
        .split(y.len(), None)
        .expect_err("stratified split requires y");
    assert!(format!("{err}").contains("y"));

    let err = cv
        .split(y.len() + 1, Some(&y))
        .expect_err("y length mismatch must be rejected");
    assert!(format!("{err}").contains("Shape mismatch"));
}
