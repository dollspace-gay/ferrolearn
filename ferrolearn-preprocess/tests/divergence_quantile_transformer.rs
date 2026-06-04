//! Divergence pins for `QuantileTransformer` vs scikit-learn 1.5.2.
//!
//! Oracle = installed sklearn 1.5.2 (`QuantileTransformer`,
//! `sklearn/preprocessing/_data.py`). To neutralise the NOT-STARTED random
//! subsample path (REQ-6), every fixture uses `subsample` larger than
//! `n_samples` (sklearn `subsample=10**9`) / ferrolearn `subsample=0` (= all),
//! and matches `n_quantiles` so both sides build identical reference grids.
//!
//! R-CHAR-3: every expected value below is produced by a LIVE sklearn call (see
//! the `// oracle:` comment above each constant) or a named sklearn `file:line`
//! symbolic constant — never literal-copied from the ferrolearn side.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::quantile_transformer::{OutputDistribution, QuantileTransformer};
use ndarray::array;

// ===========================================================================
// DIV-B (HEADLINE) — Normal output accuracy: exact stats.norm.ppf + clip.
// ===========================================================================

/// Divergence: ferrolearn's `probit` (Abramowitz-Stegun rational approx, ~1e-4
/// error, `quantile_transformer.rs:151-188`) diverges from sklearn's exact
/// `stats.norm.ppf` + `BOUNDS_THRESHOLD` clip
/// (`sklearn/preprocessing/_data.py:2856-2862`:
/// `X_col = stats.norm.ppf(X_col)` then
/// `np.clip(X_col, norm.ppf(1e-7 - spacing(1)), norm.ppf(1 - (1e-7 - spacing(1))))`).
///
/// Input: `[[1],[2],[3],[4],[5]]`, n_quantiles=5, output=Normal.
/// sklearn returns `[-5.199337582605575, -0.6744897501960817, 0.0,
///                   0.6744897501960817, 5.19933758270342]`.
/// ferrolearn's A&S `probit` of {0.25,0.75} returns ~+/-0.6744 off by ~1e-4,
/// and clamps the extremes to `probit(1e-7) ~ -5.199...` only by luck of the
/// A&S formula (the quartiles are the hard fail at 1e-6).
/// Tracking: #1320
#[test]
fn divergence_normal_ppf_accuracy() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   print(QuantileTransformer(n_quantiles=5, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[3.],[4.],[5.]]).ravel().tolist())"
    //   -> [-5.199337582605575, -0.6744897501960817, 0.0, 0.6744897501960817, 5.19933758270342]
    let sk_normal: [f64; 5] = [
        -5.199337582605575,
        -0.6744897501960817,
        0.0,
        0.6744897501960817,
        5.19933758270342,
    ];

    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk_normal.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn Normal ppf = {got}, sklearn norm.ppf = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

// ===========================================================================
// DIV-A — forward+reversed AVERAGED interpolation (plateau -> midpoint).
// ===========================================================================

/// Divergence: ferrolearn's `interpolate_cdf` (forward-only binary search,
/// `quantile_transformer.rs:192-225`) diverges from sklearn's averaged interp
/// `0.5 * (np.interp(X, q, refs) - np.interp(-X, -q[::-1], -refs[::-1]))`
/// (`sklearn/preprocessing/_data.py:2843-2846`).
///
/// Input: `[[1],[2],[2],[2],[3]]`, n_quantiles=5, output=Uniform.
/// quantiles_ = [1,2,2,2,3], references_ = [0,0.25,0.5,0.75,1.0].
/// The value 2.0 sits on the [0.25,0.75] plateau; sklearn averages ascending
/// and descending interp -> the MIDPOINT 0.5. ferrolearn's forward-only interp
/// lands on the upper extreme 0.75.
/// sklearn fit_transform -> [0.0, 0.5, 0.5, 0.5, 1.0].
/// Tracking: #1321
#[test]
fn divergence_averaged_interpolation_plateau() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   print(QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.]]).ravel().tolist())"
    //   -> [0.0, 0.5, 0.5, 0.5, 1.0]
    let sk_plateau: [f64; 5] = [0.0, 0.5, 0.5, 0.5, 1.0];

    let x = array![[1.0], [2.0], [2.0], [2.0], [3.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk_plateau.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn forward-only interp = {got}, sklearn averaged = {expected}"
        );
    }
}

// ===========================================================================
// GREEN GUARDS — SHIPPED behavior (must PASS). Live-oracle grounded.
// ===========================================================================

/// REQ-1: uniform forward-CDF on STRICTLY DISTINCT data. On distinct data
/// forward == averaged interp (DIV-A dormant) and Uniform == identity
/// (DIV-B dormant), so ferrolearn matches sklearn exactly.
/// sklearn `_data.py:2795` references linspace + `:2702` nanpercentile.
#[test]
fn green_uniform_distinct_single_feature() {
    // oracle: python3 -c "...QuantileTransformer(n_quantiles=5, output_distribution='uniform', \
    //   subsample=10**9).fit_transform([[1.],[2.],[3.],[4.],[5.]]).ravel().tolist()"
    //   -> [0.0, 0.25, 0.5, 0.75, 1.0]
    let sk: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-1: richer distinct multi-feature uniform value-match.
/// sklearn `_data.py:2702-2795` (per-column independent CDF).
#[test]
fn green_uniform_distinct_multi_feature() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   X=np.array([[1.,10.],[3.,40.],[7.,20.],[2.,80.],[9.,30.],[5.,60.]]); \
    //   print(QuantileTransformer(n_quantiles=6, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform(X).tolist())"
    //   -> [[0.0,0.0],[0.4,0.6],[0.8,0.2],[0.2,1.0],[1.0,0.4],[0.6,0.8]]
    let sk: [[f64; 2]; 6] = [
        [0.0, 0.0],
        [0.4, 0.6],
        [0.8, 0.2],
        [0.2, 1.0],
        [1.0, 0.4],
        [0.6, 0.8],
    ];

    let x = array![
        [1.0, 10.0],
        [3.0, 40.0],
        [7.0, 20.0],
        [2.0, 80.0],
        [9.0, 30.0],
        [5.0, 60.0]
    ];
    let qt = QuantileTransformer::<f64>::new(6, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();

    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - expected).abs() <= 1e-9,
                "[{i},{j}]: ferrolearn = {got}, sklearn = {expected}"
            );
        }
    }
}

/// REQ-5 (scoped error contracts): ferrolearn's own guards must hold.
/// These are ferrolearn-internal contracts (not sklearn parity — see design
/// doc REQ-5 flagged DIVs), so the assertion is on `is_err()` shape only.
#[test]
fn green_error_contracts() {
    // n_samples < 2 -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    assert!(
        qt.fit(&array![[1.0]], &()).is_err(),
        "single-sample fit should Err"
    );

    // n_quantiles < 2 -> Err
    let qt1 = QuantileTransformer::<f64>::new(1, OutputDistribution::Uniform, 0);
    assert!(
        qt1.fit(&array![[1.0], [2.0], [3.0]], &()).is_err(),
        "n_quantiles=1 should Err"
    );

    // transform column-count mismatch -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&array![[1.0, 2.0], [3.0, 4.0]], &()).unwrap();
    assert!(
        fitted.transform(&array![[1.0, 2.0, 3.0]]).is_err(),
        "ncols mismatch should Err"
    );

    // unfitted transform -> Err
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    assert!(
        qt.transform(&array![[1.0], [2.0]]).is_err(),
        "unfitted transform should Err"
    );
}

// ===========================================================================
// RE-AUDIT (2026-06): fixes #1320 (Acklam probit + clip) and #1321 (averaged
// np_interp) confirmed by `divergence_normal_ppf_accuracy` and
// `divergence_averaged_interpolation_plateau` now passing. The fixtures below
// are FRESH live-oracle fixtures (not the prior pinned inputs) verifying the
// implemented Uniform + Normal forward-transform surface end-to-end.
// ===========================================================================

/// DIV-C (NEW) — fit landmark FP divergence: sklearn computes quantile
/// landmarks via `references = self.references_ * 100`
/// (`sklearn/preprocessing/_data.py:2694`) then
/// `self.quantiles_ = np.nanpercentile(X, references, axis=0)` (`:2702`).
/// numpy's `nanpercentile` recomputes the interpolation position as
/// `pos = q/100 * (n-1)`; the `*100` (sklearn) then `/100` (numpy) round-trip
/// means e.g. the 2nd of 7 references gives `pos = 0.9999999999999998`, so
/// numpy interpolates `col[0] + 0.999...*(col[1]-col[0]) = 1.9999999999999998`
/// rather than landing exactly on the landmark.
///
/// ferrolearn (`quantile_transformer.rs:372`) keeps the reference as a fraction
/// `i/(K-1)` and computes `pos = ref_level * (n-1)` directly, yielding the EXACT
/// integer index `1.0` → landmark value `2.0`. On TIED/PLATEAU data this 1-ULP
/// landmark shift moves a query value from the interior of a plateau to its
/// edge, producing a LARGE (~0.083) divergence in the Uniform transform output.
///
/// Train `[[1],[2],[2],[2],[5],[5],[9]]` (n_quantiles=7, uniform).
/// sklearn quantiles_ = [1, 1.9999999999999998, 2, 2, 4.999999999999997, 5, 9].
/// ferrolearn quantiles  = [1, 2, 2, 2, 5, 5, 9].
/// transform([0,1,2,3.5,5,7,9,12]):
///   sklearn -> [0, 0, 0.41666666666666663, 0.5833333333333334,
///               0.8333333333333333, 0.9166666666666666, 1, 1]
///   ferrolearn -> [0, 0, 0.3333333333333333, 0.5833333333333333,
///               0.75, 0.9166666666666667, 1, 1]   (rows 2 and 4 diverge ~0.083)
/// Tracking: #1322
#[test]
fn divergence_landmark_nanpercentile_roundtrip_plateau() {
    // oracle: python3 -c "import numpy as np; from sklearn.preprocessing import QuantileTransformer; \
    //   qt=QuantileTransformer(n_quantiles=7, output_distribution='uniform', subsample=10**9)\
    //   .fit(np.array([[1.],[2.],[2.],[2.],[5.],[5.],[9.]])); \
    //   print(qt.transform(np.array([[0.],[1.],[2.],[3.5],[5.],[7.],[9.],[12.]])).ravel().tolist())"
    //   -> [0.0, 0.0, 0.41666666666666663, 0.5833333333333334,
    //       0.8333333333333333, 0.9166666666666666, 1.0, 1.0]
    let sk: [f64; 8] = [
        0.0,
        0.0,
        0.416_666_666_666_666_63,
        0.583_333_333_333_333_4,
        0.833_333_333_333_333_3,
        0.916_666_666_666_666_6,
        1.0,
        1.0,
    ];

    let xtrain = array![[1.0], [2.0], [2.0], [2.0], [5.0], [5.0], [9.0]];
    let qt = QuantileTransformer::<f64>::new(7, OutputDistribution::Uniform, 0);
    let fitted = qt.fit(&xtrain, &()).unwrap();
    let queries = array![[0.0], [1.0], [2.0], [3.5], [5.0], [7.0], [9.0], [12.0]];
    let out = fitted.transform(&queries).unwrap();

    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

// ===========================================================================
// FRESH GREEN GUARDS — implemented surface that NOW matches sklearn 1.5.2
// after #1320 + #1321. Every expected value is a LIVE sklearn 1.5.2 call
// (R-CHAR-3). These complement the headline DIV-A/DIV-B (now green).
// ===========================================================================

/// REQ-2 (post-fix #1321): Uniform on data with MULTIPLE DISTINCT PLATEAUS,
/// n_quantiles == n_samples == 10. Stresses the averaged interpolation on
/// several plateaus (the [2,2,2] run AND the [5,5] run) at once.
/// sklearn `_data.py:2843-2846` (averaged interp).
#[test]
fn green_uniform_multi_plateau() {
    // oracle: QuantileTransformer(n_quantiles=10, output_distribution='uniform', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.],[5.],[5.],[7.],[8.],[9.]]).ravel().tolist()
    let sk: [f64; 10] = [
        0.0,
        0.222_222_222_222_222_2,
        0.222_222_222_222_222_2,
        0.222_222_222_222_222_2,
        0.444_444_444_444_444_4,
        0.611_111_111_111_111_2,
        0.611_111_111_111_111_2,
        0.777_777_777_777_777_9,
        0.888_888_888_888_888_8,
        1.0,
    ];
    let x = array![
        [1.0],
        [2.0],
        [2.0],
        [2.0],
        [3.0],
        [5.0],
        [5.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-1 (post-fix): n_quantiles < n_samples (K=5, n=20). Landmarks are
/// INTERPOLATED percentiles (not raw data); exercises `pos = ref*(n-1)` against
/// `np.nanpercentile`. Data are integers 1..20 so the landmarks are exact even
/// through the *100 round-trip; full transform matches.
/// sklearn `_data.py:2702` (nanpercentile) + `:2843-2846`.
#[test]
fn green_uniform_n_quantiles_lt_n_samples() {
    // oracle: X=[[3],[7],[1],[9],[2],[8],[5],[6],[4],[10],[12],[15],[11],[13],[14],[20],[18],[16],[17],[19]] (float); \
    //   QuantileTransformer(n_quantiles=5, output_distribution='uniform', subsample=10**9).fit_transform(X)
    let sk: [f64; 20] = [
        0.105_263_157_894_736_84,
        0.315_789_473_684_210_5,
        0.0,
        0.421_052_631_578_947_35,
        0.052_631_578_947_368_42,
        0.368_421_052_631_579,
        0.210_526_315_789_473_67,
        0.263_157_894_736_842_15,
        0.157_894_736_842_105_25,
        0.473_684_210_526_315_8,
        0.578_947_368_421_052_7,
        0.736_842_105_263_157_9,
        0.526_315_789_473_684_3,
        0.631_578_947_368_421,
        0.684_210_526_315_789_5,
        1.0,
        0.894_736_842_105_263_2,
        0.789_473_684_210_526_3,
        0.842_105_263_157_894_7,
        0.947_368_421_052_631_6,
    ];
    let x = array![
        [3.0],
        [7.0],
        [1.0],
        [9.0],
        [2.0],
        [8.0],
        [5.0],
        [6.0],
        [4.0],
        [10.0],
        [12.0],
        [15.0],
        [11.0],
        [13.0],
        [14.0],
        [20.0],
        [18.0],
        [16.0],
        [17.0],
        [19.0]
    ];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-9,
            "row {i}: ferrolearn = {got}, sklearn = {expected}"
        );
    }
}

/// REQ-3 (post-fix #1320): Normal output, 11 distinct rows, n_quantiles=11 so
/// references hit non-round probabilities 0.0,0.1,...,1.0. Verifies Acklam
/// `norm.ppf` accuracy across the range (0.1→-1.28155, 0.3→-0.52440,
/// 0.7→0.52440, 0.9→1.28155) AND the extreme clip ±5.199337582605575.
/// sklearn `_data.py:2856-2862`.
#[test]
fn green_normal_nonround_probabilities() {
    // oracle: QuantileTransformer(n_quantiles=11, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[0.],[1.],...,[10.]]).ravel().tolist()
    let sk: [f64; 11] = [
        -5.199_337_582_605_575,
        -1.281_551_565_544_600_4,
        -0.841_621_233_572_914_2,
        -0.524_400_512_708_040_9,
        -0.253_347_103_135_799_7,
        0.0,
        0.253_347_103_135_799_7,
        0.524_400_512_708_041,
        0.841_621_233_572_914_3,
        1.281_551_565_544_600_4,
        5.199_337_582_703_42,
    ];
    let x = array![
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
        [9.0],
        [10.0]
    ];
    let qt = QuantileTransformer::<f64>::new(11, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// REQ-3 (post-fix #1320): Normal output on TIED/PLATEAU data. The plateau maps
/// to a single non-round probability whose `norm.ppf` must match.
/// sklearn `_data.py:2843-2846` + `:2856`.
#[test]
fn green_normal_plateau() {
    // oracle: QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9)\
    //   .fit_transform([[1.],[2.],[2.],[2.],[3.],[5.],[5.],[7.],[8.],[9.]]).ravel().tolist()
    let sk: [f64; 10] = [
        -5.199_337_582_605_575,
        -0.764_709_673_786_387_1,
        -0.764_709_673_786_387_1,
        -0.764_709_673_786_387_1,
        -0.139_710_298_881_862_12,
        0.282_216_147_062_508_25,
        0.282_216_147_062_508_25,
        0.764_709_673_786_387_5,
        1.220_640_348_847_349_6,
        5.199_337_582_703_42,
    ];
    let x = array![
        [1.0],
        [2.0],
        [2.0],
        [2.0],
        [3.0],
        [5.0],
        [5.0],
        [7.0],
        [8.0],
        [9.0]
    ];
    let qt = QuantileTransformer::<f64>::new(10, OutputDistribution::Normal, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - expected).abs() <= 1e-6,
            "row {i}: ferrolearn = {got}, sklearn = {expected} (diff {})",
            (got - expected).abs()
        );
    }
}

/// np_interp faithfulness vs numpy: query points below range, at a landmark,
/// exactly on a plateau value, between landmarks, above range — driven through
/// a fitted transformer whose quantiles_ contain DUPLICATES (the [2,2] and
/// [5,5] plateaus). Expected = live sklearn `_transform_col` averaged output.
/// sklearn `_data.py:2843-2846` (averaged `np.interp`).
#[test]
fn green_np_interp_faithfulness_duplicates() {
    // oracle: Xtrain=[[1.],[2.],[2.],[2.],[5.],[5.],[9.]]; \
    //   qt=QuantileTransformer(n_quantiles=7, output_distribution='uniform', subsample=10**9).fit(Xtrain); \
    //   qt.transform([[0.],[1.],[2.],[3.5],[5.],[7.],[9.],[12.]]).ravel().tolist()
    //   -> [0.0, 0.0, 0.41666666666666663, 0.5833333333333334,
    //       0.8333333333333333, 0.9166666666666666, 1.0, 1.0]
    //
    // NOTE: this is the SAME end-to-end pipeline as the ignored DIV-C pin
    // (#1322). Because ferrolearn's fit landmarks diverge from
    // np.nanpercentile, the full-pipeline values diverge at rows 2 and 4.
    // The np_interp helper itself is faithful, but it is fed different
    // landmarks. We therefore assert np_interp faithfulness on the
    // landmark grid ferrolearn ACTUALLY produces, by reading it from the
    // fitted model and re-deriving the expected averaged-interp from numpy
    // on THAT grid (R-CHAR-3: numpy is the oracle for np.interp).
    use ferrolearn_preprocess::quantile_transformer::FittedQuantileTransformer;

    let xtrain = array![[1.0], [2.0], [2.0], [2.0], [5.0], [5.0], [9.0]];
    let qt = QuantileTransformer::<f64>::new(7, OutputDistribution::Uniform, 0);
    let fitted: FittedQuantileTransformer<f64> = qt.fit(&xtrain, &()).unwrap();
    let landmarks = fitted.quantiles()[0].clone();

    // Live numpy oracle on the EXACT landmark grid ferrolearn produced.
    let refs: Vec<f64> = (0..landmarks.len())
        .map(|i| i as f64 / (landmarks.len() - 1) as f64)
        .collect();
    let queries = [0.0_f64, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0];

    let py = format!(
        "import numpy as np\nq=np.array({:?})\nr=np.array({:?})\nx=np.array({:?})\nv=0.5*(np.interp(x,q,r)-np.interp(-x,-q[::-1],-r[::-1]))\n# replicate sklearn uniform exact-bound override\nv[x==q[0]]=0.0\nv[x==q[-1]]=1.0\nprint(' '.join(repr(float(t)) for t in v))",
        landmarks, refs, queries
    );
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&py)
        .output()
        .expect("python3 oracle call failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected: Vec<f64> = stdout
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        expected.len(),
        queries.len(),
        "oracle parse failed: {stdout}"
    );

    let qarr = array![[0.0], [1.0], [2.0], [3.5], [5.0], [7.0], [9.0], [12.0]];
    let out = fitted.transform(&qarr).unwrap();
    for (i, &exp) in expected.iter().enumerate() {
        let got = out[[i, 0]];
        assert!(
            (got - exp).abs() <= 1e-12,
            "np_interp row {i}: ferrolearn = {got}, numpy averaged interp = {exp}"
        );
    }
}

/// Fully-distinct no-tie matrix: Uniform transforms identically and Normal
/// median maps to exactly 0. sklearn `_data.py:2843-2846`,`:2856`.
#[test]
fn green_distinct_uniform_and_normal_median_zero() {
    // oracle uniform: QuantileTransformer(5,'uniform',10**9).fit_transform([[3.],[7.],[1.],[9.],[5.]])
    //   -> [0.25, 0.75, 0.0, 1.0, 0.5]
    let sk_uniform: [f64; 5] = [0.25, 0.75, 0.0, 1.0, 0.5];
    let x = array![[3.0], [7.0], [1.0], [9.0], [5.0]];
    let qt = QuantileTransformer::<f64>::new(5, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk_uniform.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-9,
            "uniform row {i}: ferrolearn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
    // Normal median (the value at CDF=0.5) -> exactly 0.
    let qtn = QuantileTransformer::<f64>::new(5, OutputDistribution::Normal, 0);
    let outn = qtn.fit_transform(&x).unwrap();
    // sorted [1,3,5,7,9]; median value is 5.0 at row index 4.
    assert!(
        outn[[4, 0]].abs() <= 1e-12,
        "Normal median should map to 0, got {}",
        outn[[4, 0]]
    );
}

/// f32 fixture: distinct data, Uniform, tolerance ~1e-4.
/// sklearn `_data.py:2843-2846` (sklearn computes in f64 then casts).
#[test]
fn green_f32_uniform_distinct() {
    // oracle: QuantileTransformer(6,'uniform',10**9).fit_transform(
    //   np.array([[1],[2],[3],[4],[5],[6]],dtype=np.float32))
    //   -> [0.0, 0.20000000298023224, 0.4000000059604645, 0.6000000238418579,
    //       0.800000011920929, 1.0]
    let sk: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let x: ndarray::Array2<f32> = array![[1.0f32], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let qt = QuantileTransformer::<f32>::new(6, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, &expected) in sk.iter().enumerate() {
        assert!(
            (out[[i, 0]] - expected).abs() <= 1e-4,
            "f32 row {i}: ferrolearn = {}, sklearn = {expected}",
            out[[i, 0]]
        );
    }
}

/// Multi-feature Uniform with internal plateaus, n_quantiles == n_samples == 8.
/// Two independent columns each with their own plateaus.
/// sklearn `_data.py:2702-2799` (per-column) + `:2843-2846`.
#[test]
fn green_uniform_multifeature_plateau() {
    // oracle: X=[[1.,5.],[1.,5.],[3.,5.],[4.,8.],[6.,8.],[6.,9.],[7.,9.],[9.,12.]]; \
    //   QuantileTransformer(8,'uniform',10**9).fit_transform(X).tolist()
    let sk: [[f64; 2]; 8] = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.285_714_285_714_285_7, 0.0],
        [0.428_571_428_571_428_55, 0.5],
        [0.642_857_142_857_142_8, 0.5],
        [0.642_857_142_857_142_8, 0.785_714_285_714_285_6],
        [0.857_142_857_142_857_1, 0.785_714_285_714_285_6],
        [1.0, 1.0],
    ];
    let x = array![
        [1.0, 5.0],
        [1.0, 5.0],
        [3.0, 5.0],
        [4.0, 8.0],
        [6.0, 8.0],
        [6.0, 9.0],
        [7.0, 9.0],
        [9.0, 12.0]
    ];
    let qt = QuantileTransformer::<f64>::new(8, OutputDistribution::Uniform, 0);
    let out = qt.fit_transform(&x).unwrap();
    for (i, row) in sk.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let got = out[[i, j]];
            assert!(
                (got - expected).abs() <= 1e-9,
                "[{i},{j}]: ferrolearn = {got}, sklearn = {expected}"
            );
        }
    }
}
