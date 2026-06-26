//! Divergence guards for dense grayscale image patch helpers vs scikit-learn
//! 1.5.2 `sklearn.feature_extraction.image`.
//!
//! Oracle snippets:
//! ```text
//! import numpy as np
//! from sklearn.feature_extraction.image import (
//!     PatchExtractor, extract_patches_2d, reconstruct_from_patches_2d,
//! )
//! img = np.arange(12., dtype=float).reshape(3, 4)
//! print(extract_patches_2d(img, (2, 2)).tolist())
//! print(reconstruct_from_patches_2d(extract_patches_2d(img, (2, 2)), img.shape).tolist())
//! X = np.stack([img, img + 100.0])
//! print(PatchExtractor(patch_size=(2, 2)).transform(X).tolist())
//! ```

use ferrolearn_core::traits::Transform;
use ferrolearn_preprocess::{
    MaxPatches, PatchExtractor, extract_patches_2d, reconstruct_from_patches_2d,
};
use ndarray::{Array2, Array3, array};

fn assert_matrix_eq(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            assert!(
                (actual[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "entry ({i},{j}) got {} expected {}",
                actual[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

fn assert_patches_eq(actual: &Array3<f64>, expected: &Array3<f64>) {
    assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
    for ((p, i, j), actual_value) in actual.indexed_iter() {
        let expected_value = expected[[p, i, j]];
        assert!(
            (*actual_value - expected_value).abs() < 1e-12,
            "entry ({p},{i},{j}) got {actual_value} expected {expected_value}"
        );
    }
}

fn sklearn_patch_oracle() -> Array3<f64> {
    Array3::from_shape_vec(
        (6, 2, 2),
        vec![
            0.0, 1.0, 4.0, 5.0, 1.0, 2.0, 5.0, 6.0, 2.0, 3.0, 6.0, 7.0, 4.0, 5.0, 8.0, 9.0, 5.0,
            6.0, 9.0, 10.0, 6.0, 7.0, 10.0, 11.0,
        ],
    )
    .unwrap()
}

#[test]
fn extract_patches_2d_all_matches_sklearn_oracle() {
    let image = array![
        [0.0_f64, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ];

    let patches = extract_patches_2d(&image, (2, 2), None, None).unwrap();
    assert_patches_eq(&patches, &sklearn_patch_oracle());
}

#[test]
fn reconstruct_from_patches_2d_roundtrips_sklearn_complete_patch_oracle() {
    let patches = sklearn_patch_oracle();
    let reconstructed = reconstruct_from_patches_2d(&patches, (3, 4)).unwrap();
    let expected = array![
        [0.0_f64, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ];
    assert_matrix_eq(&reconstructed, &expected);
}

#[test]
fn patch_extractor_batch_order_matches_sklearn_oracle() {
    let first = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let second: Vec<f64> = first.iter().map(|v| v + 100.0).collect();
    let mut values = first;
    values.extend(second);
    let batch = Array3::from_shape_vec((2, 3, 4), values).unwrap();

    let patches = PatchExtractor::<f64>::new()
        .patch_size((2, 2))
        .transform(&batch)
        .unwrap();
    let expected = Array3::from_shape_vec(
        (12, 2, 2),
        vec![
            0.0, 1.0, 4.0, 5.0, 1.0, 2.0, 5.0, 6.0, 2.0, 3.0, 6.0, 7.0, 4.0, 5.0, 8.0, 9.0, 5.0,
            6.0, 9.0, 10.0, 6.0, 7.0, 10.0, 11.0, 100.0, 101.0, 104.0, 105.0, 101.0, 102.0, 105.0,
            106.0, 102.0, 103.0, 106.0, 107.0, 104.0, 105.0, 108.0, 109.0, 105.0, 106.0, 109.0,
            110.0, 106.0, 107.0, 110.0, 111.0,
        ],
    )
    .unwrap();
    assert_patches_eq(&patches, &expected);
}

#[test]
fn max_patches_count_caps_and_fraction_floor_shape() {
    let image = array![
        [0.0_f64, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ];

    let capped = extract_patches_2d(&image, (2, 2), Some(MaxPatches::Count(99)), Some(0)).unwrap();
    assert_eq!(capped.dim(), (6, 2, 2));

    let tiny_fraction =
        extract_patches_2d(&image, (2, 2), Some(MaxPatches::Fraction(0.01)), Some(0)).unwrap();
    assert_eq!(tiny_fraction.dim(), (0, 2, 2));
}

#[test]
fn sampled_patch_extraction_is_seed_deterministic() {
    let image = array![
        [0.0_f64, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ];
    let a = extract_patches_2d(&image, (2, 2), Some(MaxPatches::Count(3)), Some(42)).unwrap();
    let b = extract_patches_2d(&image, (2, 2), Some(MaxPatches::Count(3)), Some(42)).unwrap();
    assert_patches_eq(&a, &b);
}

#[test]
fn patch_helpers_validate_shapes() {
    let image = array![[0.0_f64, 1.0], [2.0, 3.0]];
    assert!(extract_patches_2d(&image, (0, 1), None, None).is_err());
    assert!(extract_patches_2d(&image, (3, 1), None, None).is_err());
    assert!(extract_patches_2d(&image, (1, 1), Some(MaxPatches::Count(0)), None).is_err());
    assert!(extract_patches_2d(&image, (1, 1), Some(MaxPatches::Fraction(1.0)), None).is_err());

    let patches = Array3::<f64>::zeros((1, 3, 1));
    assert!(reconstruct_from_patches_2d(&patches, (2, 2)).is_err());

    let too_small_for_default = Array3::<f64>::zeros((1, 9, 9));
    assert!(
        PatchExtractor::<f64>::new()
            .transform(&too_small_for_default)
            .is_err()
    );
}
