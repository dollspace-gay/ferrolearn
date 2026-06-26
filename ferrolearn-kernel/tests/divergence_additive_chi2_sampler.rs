//! Green-guard tests for `ferrolearn-kernel::AdditiveChi2Sampler` against
//! scikit-learn 1.5.2 `sklearn.kernel_approximation.AdditiveChi2Sampler`
//! (`sklearn/kernel_approximation.py:585-837`).
//!
//! R-CHAR-3: expected literals below were produced by live sklearn calls from
//! this workspace, never copied from ferrolearn output.

#![allow(clippy::approx_constant)]

use ferrolearn_core::{Fit, Transform};
use ferrolearn_kernel::AdditiveChi2Sampler;
use ndarray::{Array2, array};

fn assert_mat(actual: &Array2<f64>, expected: &[&[f64]]) {
    assert_eq!(actual.nrows(), expected.len(), "row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(actual.ncols(), row.len(), "col count");
        for (j, &e) in row.iter().enumerate() {
            let a = actual[[i, j]];
            assert!(
                (a - e).abs() < 1e-12,
                "Z[{i},{j}] = {a} but sklearn oracle = {e}"
            );
        }
    }
}

fn x2() -> Array2<f64> {
    array![[0.0, 1.0], [2.0, 3.0]]
}

#[test]
fn green_additive_chi2_default_sample_steps_1_2_3() {
    let x = x2();

    // sklearn 1.5.2, /tmp:
    // AdditiveChi2Sampler(sample_steps=1).fit_transform([[0,1],[2,3]])
    let fitted_1 = AdditiveChi2Sampler::<f64>::new()
        .with_sample_steps(1)
        .fit(&x, &())
        .unwrap();
    assert_eq!(fitted_1.sample_steps(), 1);
    assert!((fitted_1.sample_interval() - 0.8).abs() < 1e-15);
    assert_mat(
        &fitted_1.transform(&x).unwrap(),
        &[
            &[0.0, 0.894_427_190_999_915_9],
            &[1.264_911_064_067_351_8, 1.549_193_338_482_966_8],
        ],
    );

    // sklearn 1.5.2, /tmp:
    // AdditiveChi2Sampler(sample_steps=2).fit_transform([[0,1],[2,3]])
    let fitted_2 = AdditiveChi2Sampler::<f64>::new().fit(&x, &()).unwrap();
    assert_eq!(fitted_2.sample_steps(), 2);
    assert!((fitted_2.sample_interval() - 0.5).abs() < 1e-15);
    assert_mat(
        &fitted_2.transform(&x).unwrap(),
        &[
            &[
                0.0,
                0.707_106_781_186_547_6,
                0.0,
                0.631_297_723_216_539_7,
                0.0,
                0.0,
            ],
            &[
                1.0,
                1.224_744_871_391_589,
                0.839_706_399_476_854_9,
                0.932_580_517_149_334_1,
                0.303_260_273_287_468_13,
                0.570_880_044_360_144_6,
            ],
        ],
    );

    // sklearn 1.5.2, /tmp:
    // AdditiveChi2Sampler(sample_steps=3).fit_transform([[0,1],[2,3]])
    let fitted_3 = AdditiveChi2Sampler::<f64>::new()
        .with_sample_steps(3)
        .fit(&x, &())
        .unwrap();
    assert_eq!(fitted_3.sample_steps(), 3);
    assert!((fitted_3.sample_interval() - 0.4).abs() < 1e-15);
    assert_mat(
        &fitted_3.transform(&x).unwrap(),
        &[
            &[
                0.0,
                0.632_455_532_033_675_9,
                0.0,
                0.649_039_835_474_980_1,
                0.0,
                0.0,
                0.0,
                0.358_830_466_364_188_75,
                0.0,
                0.0,
            ],
            &[
                0.894_427_190_999_915_9,
                1.095_445_115_010_332_4,
                0.882_826_470_898_015_5,
                1.017_360_284_673_664,
                0.251_242_588_643_991_1,
                0.478_263_708_918_927_6,
                0.431_421_859_428_811_84,
                0.396_529_017_020_338_5,
                0.267_196_157_137_914_2,
                0.478_584_004_574_231_3,
            ],
        ],
    );
}

#[test]
fn green_additive_chi2_custom_interval_for_higher_steps() {
    let x = x2();
    let fitted = AdditiveChi2Sampler::<f64>::new()
        .with_sample_steps(4)
        .with_sample_interval(0.7)
        .fit(&x, &())
        .unwrap();
    assert_eq!(fitted.sample_steps(), 4);
    assert!((fitted.sample_interval() - 0.7).abs() < 1e-15);

    // sklearn 1.5.2, /tmp:
    // AdditiveChi2Sampler(sample_steps=4, sample_interval=0.7)
    //     .fit_transform([[0,1],[2,3]])
    assert_mat(
        &fitted.transform(&x).unwrap(),
        &[
            &[
                0.0,
                0.836_660_026_534_075_6,
                0.0,
                0.553_850_902_645_460_8,
                0.0,
                0.0,
                0.0,
                0.185_559_298_964_037_72,
                0.0,
                0.0,
                0.0,
                0.061_799_283_532_826_99,
                0.0,
                0.0,
            ],
            &[
                1.183_215_956_619_923_2,
                1.449_137_674_618_943_7,
                0.692_859_407_687_886_8,
                0.689_338_575_594_949_4,
                0.365_304_648_068_942_36,
                0.667_131_767_554_432_2,
                0.148_258_263_523_430_74,
                0.010_520_413_299_742_381,
                0.216_527_121_075_091_23,
                0.321_225_903_687_853_25,
                0.010_044_817_175_789_8,
                -0.071_881_644_797_954_85,
                0.086_818_226_993_220_96,
                0.079_312_568_213_700_36,
            ],
        ],
    );
}

#[test]
fn green_additive_chi2_validation_matches_sklearn_boundaries() {
    let x = x2();

    // sklearn raises ValueError when sample_steps is outside 1..=3 and no
    // sample_interval is supplied.
    assert!(
        AdditiveChi2Sampler::<f64>::new()
            .with_sample_steps(4)
            .fit(&x, &())
            .is_err()
    );

    // sklearn uses ensure_non_negative=True in both fit and transform.
    let negative = array![[1.0, -1.0]];
    assert!(
        AdditiveChi2Sampler::<f64>::new()
            .fit(&negative, &())
            .is_err()
    );
    let fitted = AdditiveChi2Sampler::<f64>::new().fit(&x, &()).unwrap();
    assert!(fitted.transform(&negative).is_err());
}
