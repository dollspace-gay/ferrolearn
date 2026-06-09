//! Dummy (baseline) estimators that ignore the input features.
//!
//! These are used as sanity-check baselines: a serious model must beat the
//! dummy, otherwise its predictive power is no better than the chosen
//! constant or marginal-distribution rule.
//!
//! - [`DummyClassifier`] supports the strategies most_frequent, prior,
//!   stratified, uniform, and constant.
//! - [`DummyRegressor`] supports mean, median, quantile, and constant.
//!
//! Mirrors scikit-learn's `sklearn/dummy.py` `DummyClassifier` (`:35`) and
//! `DummyRegressor` (`:465`, tag 1.5.2). Deterministic strategies have full
//! value parity; `stratified`/`uniform` are RNG carve-outs.
//!
//! ## REQ status
//!
//! | REQ | Behavior | Status | Evidence |
//! |-----|----------|--------|----------|
//! | REQ-1 | classifier `most_frequent`/`prior`/`constant` predict | SHIPPED | `impl Predict for FittedDummyClassifier` + `fn most_frequent` (first-max tie-break = sklearn `np.argmax(class_prior_)`, `dummy.py:311`); oracle green guards |
//! | REQ-2 | `predict_proba`/`predict_log_proba` | SHIPPED | `fn predict_proba`/`fn predict_log_proba` on `FittedDummyClassifier`: `prior`→`class_prior_` row, `most_frequent`/`constant`→one-hot at `argmax(class_prior_)`, `uniform`→`1/n_classes`, `log_proba`→elementwise `ln` (`0`→`-inf`); mirrors `dummy.py:340-423`. Deterministic strategies oracle-pinned (`dummy_predict_proba_*`/`dummy_predict_log_proba_matches_sklearn`); `stratified` proba is an R-DEFER-3 RNG carve-out (SmallRng draw, not oracle-pinned). Closes #1691 |
//! | REQ-3 | `stratified`/`uniform` exact parity | NOT-STARTED | RNG carve-out (`SmallRng` vs numpy `RandomState`), R-DEFER-3 — blocker #1695 |
//! | REQ-4 | constant ∉ classes rejected | SHIPPED | `fn fit` returns `Err`; sklearn raises `ValueError` (behavior parity, R-DEV-2 family; variant-name note #1692) |
//! | REQ-5 | classifier attributes (`classes_`/`n_classes_`) | SHIPPED | `impl HasClasses` + `fn most_frequent` |
//! | REQ-6 | regressor `mean`/`median`/`quantile`/`constant` value parity | SHIPPED | `fn fit` + `fn quantile_value` (numpy `method='linear'` + even-n median); oracle `~1e-12` (mean/median/quantile q=.25/.9, even+odd n) |
//! | REQ-7 | quantile range validation (`q ∈ [0,1]`) | SHIPPED | `fn quantile_value` rejects out-of-range |
//! | REQ-8 | `sample_weight` (weighted mean/percentile) | NOT-STARTED | neither `fit` accepts sample_weight (`dummy.py`) — blocker #1693 |
//! | REQ-9 | multi-output 2D y | NOT-STARTED | `Fit<.., Array1<..>>` single-output only (`MultiOutputMixin`) — blocker #1694 |
//! | REQ-10 | ferray substrate | NOT-STARTED | on `ndarray` + `rand::SmallRng`, not `ferray-core`/`ferray::random` — blocker #1696 |
//! | REQ-11 | non-test production consumer | SHIPPED | boundary estimator re-export `lib.rs:82` (S5) |
//!
//! Reference: scikit-learn 1.5.2 (commit 156ef14).

use std::collections::HashMap;

use ferrolearn_core::introspection::HasClasses;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// DummyClassifier
// ---------------------------------------------------------------------------

/// Strategy used by [`DummyClassifier`] to choose predictions.
#[derive(Debug, Clone, PartialEq)]
pub enum DummyClassifierStrategy {
    /// Always predict the most frequent training label.
    MostFrequent,
    /// Same as [`Self::MostFrequent`]; included to mirror sklearn naming.
    Prior,
    /// Sample predictions from the empirical class prior.
    Stratified,
    /// Sample predictions uniformly at random from the observed classes.
    Uniform,
    /// Always predict a fixed user-supplied constant.
    Constant(usize),
}

/// Baseline classifier that ignores its input features.
#[derive(Debug, Clone)]
pub struct DummyClassifier {
    strategy: DummyClassifierStrategy,
    random_state: Option<u64>,
}

impl DummyClassifier {
    /// Construct a new [`DummyClassifier`] with the given strategy.
    #[must_use]
    pub fn new(strategy: DummyClassifierStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set the RNG seed used by stochastic strategies.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Default for DummyClassifier {
    fn default() -> Self {
        Self::new(DummyClassifierStrategy::Prior)
    }
}

/// Fitted [`DummyClassifier`].
#[derive(Debug, Clone)]
pub struct FittedDummyClassifier {
    strategy: DummyClassifierStrategy,
    classes: Vec<usize>,
    class_priors: Vec<f64>,
    random_state: Option<u64>,
}

impl<F: Float + Send + Sync + 'static> Fit<Array2<F>, Array1<usize>> for DummyClassifier {
    type Fitted = FittedDummyClassifier;
    type Error = FerroError;

    fn fit(&self, _x: &Array2<F>, y: &Array1<usize>) -> Result<Self::Fitted, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DummyClassifier::fit".into(),
            });
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &c in y.iter() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut classes: Vec<usize> = counts.keys().copied().collect();
        classes.sort_unstable();
        let n = y.len() as f64;
        let class_priors: Vec<f64> = classes.iter().map(|c| counts[c] as f64 / n).collect();

        if let DummyClassifierStrategy::Constant(c) = self.strategy
            && !classes.contains(&c)
        {
            return Err(FerroError::InvalidParameter {
                name: "constant".into(),
                reason: format!("DummyClassifier: constant {c} not present in training labels"),
            });
        }

        Ok(FittedDummyClassifier {
            strategy: self.strategy.clone(),
            classes,
            class_priors,
            random_state: self.random_state,
        })
    }
}

impl FittedDummyClassifier {
    /// Most-frequent class as observed during training.
    #[must_use]
    pub fn most_frequent(&self) -> usize {
        let mut best = self.classes[0];
        let mut best_p = self.class_priors[0];
        for (i, &p) in self.class_priors.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = self.classes[i];
            }
        }
        best
    }

    fn make_rng(&self, salt: u64) -> SmallRng {
        match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(salt)),
            None => SmallRng::from_os_rng(),
        }
    }

    fn weighted_choice(&self, rng: &mut SmallRng) -> usize {
        let r: f64 = {
            use rand::Rng;
            rng.random::<f64>()
        };
        let mut acc = 0.0_f64;
        for (i, &p) in self.class_priors.iter().enumerate() {
            acc += p;
            if r <= acc {
                return self.classes[i];
            }
        }
        *self.classes.last().expect("at least one class")
    }

    /// Index (into [`Self::classes`]) of the most-frequent class
    /// (`np.argmax(class_prior_)`, first-on-ties), mirroring sklearn
    /// `dummy.py:311,377`.
    fn most_frequent_index(&self) -> usize {
        let mut best = 0usize;
        let mut best_p = self.class_priors[0];
        for (i, &p) in self.class_priors.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = i;
            }
        }
        best
    }

    /// Per-class probability estimates, one row per sample.
    ///
    /// Returns an `(n_samples, n_classes)` matrix where `n_samples = x.nrows()`
    /// and `n_classes = self.classes.len()`. Mirrors sklearn
    /// `DummyClassifier.predict_proba` (`dummy.py:340-389`, tag 1.5.2):
    /// - `MostFrequent`/`Constant`: a one-hot row at the chosen class index.
    /// - `Prior`: the class-prior distribution repeated for every sample.
    /// - `Uniform`: `1 / n_classes` for every entry.
    /// - `Stratified`: an independent random one-hot per sample drawn from the
    ///   class prior (R-DEFER-3 RNG carve-out — uses the SAME `SmallRng`/seed
    ///   mechanism as [`Predict::predict`]; not oracle-pinned, cannot bit-match
    ///   numpy's `RandomState.multinomial`).
    #[must_use]
    pub fn predict_proba<F: Float + Send + Sync + 'static>(&self, x: &Array2<F>) -> Array2<f64> {
        let n = x.nrows();
        let k = self.classes.len();
        match &self.strategy {
            // sklearn's `predict` is identical for prior/most_frequent, but
            // `predict_proba` diverges: prior returns the class_prior_ row
            // (`dummy.py:380-381`) while most_frequent returns a one-hot at
            // `argmax(class_prior_)` (`dummy.py:376-379`).
            DummyClassifierStrategy::Prior => {
                let mut out = Array2::<f64>::zeros((n, k));
                for mut row in out.rows_mut() {
                    for (j, &p) in self.class_priors.iter().enumerate() {
                        row[j] = p;
                    }
                }
                out
            }
            DummyClassifierStrategy::MostFrequent => {
                let ind = self.most_frequent_index();
                let mut out = Array2::<f64>::zeros((n, k));
                out.column_mut(ind).fill(1.0);
                out
            }
            DummyClassifierStrategy::Constant(c) => {
                let mut out = Array2::<f64>::zeros((n, k));
                // `c` is validated to be a member of `classes` during `fit`,
                // so `position` always finds it; the `if let` keeps this branch
                // panic-free without an `unwrap`/`expect` (R-CODE-2).
                if let Some(ind) = self.classes.iter().position(|&cls| cls == *c) {
                    out.column_mut(ind).fill(1.0);
                }
                out
            }
            DummyClassifierStrategy::Uniform => Array2::<f64>::from_elem((n, k), 1.0 / k as f64),
            DummyClassifierStrategy::Stratified => {
                let mut rng = self.make_rng(0);
                let mut out = Array2::<f64>::zeros((n, k));
                for mut row in out.rows_mut() {
                    let cls = self.weighted_choice(&mut rng);
                    // `weighted_choice` only ever returns a member of `classes`.
                    if let Some(ind) = self.classes.iter().position(|&c| c == cls) {
                        row[ind] = 1.0;
                    }
                }
                out
            }
        }
    }

    /// Per-class log-probability estimates, `ln(predict_proba(x))` elementwise.
    ///
    /// Mirrors sklearn `DummyClassifier.predict_log_proba`
    /// (`dummy.py:391-423`, tag 1.5.2): a `0.0` probability maps to `-inf`
    /// (matching `np.log(0.0)`).
    #[must_use]
    pub fn predict_log_proba<F: Float + Send + Sync + 'static>(
        &self,
        x: &Array2<F>,
    ) -> Array2<f64> {
        self.predict_proba(x).mapv(f64::ln)
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDummyClassifier {
    type Output = Array1<usize>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<usize>, FerroError> {
        let n = x.nrows();
        let out = match &self.strategy {
            DummyClassifierStrategy::MostFrequent | DummyClassifierStrategy::Prior => {
                Array1::from_elem(n, self.most_frequent())
            }
            DummyClassifierStrategy::Constant(c) => Array1::from_elem(n, *c),
            DummyClassifierStrategy::Stratified => {
                let mut rng = self.make_rng(0);
                let mut buf = Vec::with_capacity(n);
                for _ in 0..n {
                    buf.push(self.weighted_choice(&mut rng));
                }
                Array1::from(buf)
            }
            DummyClassifierStrategy::Uniform => {
                let mut rng = self.make_rng(0);
                let mut buf = Vec::with_capacity(n);
                for _ in 0..n {
                    buf.push(*self.classes.choose(&mut rng).expect("at least one class"));
                }
                Array1::from(buf)
            }
        };
        Ok(out)
    }
}

impl HasClasses for FittedDummyClassifier {
    fn classes(&self) -> &[usize] {
        &self.classes
    }
    fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

// ---------------------------------------------------------------------------
// DummyRegressor
// ---------------------------------------------------------------------------

/// Strategy used by [`DummyRegressor`] to choose predictions.
#[derive(Debug, Clone, PartialEq)]
pub enum DummyRegressorStrategy<F> {
    /// Predict the mean of the training targets.
    Mean,
    /// Predict the median of the training targets.
    Median,
    /// Predict the empirical `quantile` (in `[0, 1]`) of the training targets.
    Quantile(f64),
    /// Always predict a fixed user-supplied constant.
    Constant(F),
}

/// Baseline regressor that ignores its input features.
#[derive(Debug, Clone)]
pub struct DummyRegressor<F> {
    strategy: DummyRegressorStrategy<F>,
}

impl<F> DummyRegressor<F> {
    /// Construct a new [`DummyRegressor`] with the given strategy.
    #[must_use]
    pub fn new(strategy: DummyRegressorStrategy<F>) -> Self {
        Self { strategy }
    }
}

impl<F: Float> Default for DummyRegressor<F> {
    fn default() -> Self {
        Self::new(DummyRegressorStrategy::Mean)
    }
}

/// Fitted [`DummyRegressor`].
#[derive(Debug, Clone)]
pub struct FittedDummyRegressor<F> {
    constant: F,
}

impl<F> FittedDummyRegressor<F> {
    /// The constant returned for every input row.
    pub fn constant(&self) -> &F {
        &self.constant
    }
}

impl<F: Float + Send + Sync + FromPrimitive + 'static> Fit<Array2<F>, Array1<F>>
    for DummyRegressor<F>
{
    type Fitted = FittedDummyRegressor<F>;
    type Error = FerroError;

    fn fit(&self, _x: &Array2<F>, y: &Array1<F>) -> Result<Self::Fitted, FerroError> {
        if y.is_empty() {
            return Err(FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "DummyRegressor::fit".into(),
            });
        }
        let constant = match &self.strategy {
            DummyRegressorStrategy::Mean => {
                let n = F::from(y.len()).ok_or_else(|| FerroError::InvalidParameter {
                    name: "n_samples".into(),
                    reason: "could not convert to F".into(),
                })?;
                y.iter().fold(F::zero(), |acc, &v| acc + v) / n
            }
            DummyRegressorStrategy::Median => quantile_value(y, 0.5)?,
            DummyRegressorStrategy::Quantile(q) => quantile_value(y, *q)?,
            DummyRegressorStrategy::Constant(c) => *c,
        };
        Ok(FittedDummyRegressor { constant })
    }
}

impl<F: Float + Send + Sync + 'static> Predict<Array2<F>> for FittedDummyRegressor<F> {
    type Output = Array1<F>;
    type Error = FerroError;

    fn predict(&self, x: &Array2<F>) -> Result<Array1<F>, FerroError> {
        Ok(Array1::from_elem(x.nrows(), self.constant))
    }
}

fn quantile_value<F: Float + FromPrimitive>(y: &Array1<F>, q: f64) -> Result<F, FerroError> {
    if !(0.0..=1.0).contains(&q) {
        return Err(FerroError::InvalidParameter {
            name: "quantile".into(),
            reason: format!("must be in [0, 1], got {q}"),
        });
    }
    let mut sorted: Vec<F> = y.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }
    // Match sklearn's quantile path bit-for-bit: `percentile = quantile * 100.0`
    // (`sklearn/dummy.py:599`) then `np.percentile(y, q=percentile)`, whose
    // virtual index is `percentile / 100.0 * (n - 1)` (`sklearn/dummy.py:601`).
    // The `*100.0` then `/100.0` round-trip rounds differently from a direct
    // `q * (n - 1)`, reproducing numpy's exact virtual index (e.g. q=1/3, n=4 →
    // 0.9999999999999998, NOT 1.0). For "clean landing" quantiles where the
    // round-trip is identity (0, 0.25, 0.5, 0.9, 1.0 here) this is unchanged.
    let percentile = q * 100.0;
    let pos = percentile / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Ok(sorted[lo])
    } else {
        // numpy's `_lerp` (numpy/lib/_function_base_impl.py `_lerp`) is a
        // TWO-branch formula, not the naive `a + (b-a)*t`: it computes
        // `a + (b-a)*t` but, where `t >= 0.5`, instead computes
        // `b - (b-a)*(1-t)`. The branch switch changes the rounding and is what
        // makes `np.percentile` bit-exact. For q=1/3, n=4 → t=0.9999999999999998
        // (>= 0.5) → `b - (b-a)*(1-t)` = 19.999999999999996, matching sklearn's
        // `np.percentile(y, q=quantile*100)` (`sklearn/dummy.py:601`).
        let t = pos - lo as f64;
        let frac = F::from(t).ok_or_else(|| FerroError::InvalidParameter {
            name: "fraction".into(),
            reason: "could not convert to F".into(),
        })?;
        let diff = sorted[hi] - sorted[lo];
        if t >= 0.5 {
            let one_minus = F::from(1.0 - t).ok_or_else(|| FerroError::InvalidParameter {
                name: "fraction".into(),
                reason: "could not convert to F".into(),
            })?;
            Ok(sorted[hi] - diff * one_minus)
        } else {
            Ok(sorted[lo] + diff * frac)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn x() -> Array2<f64> {
        Array2::<f64>::zeros((5, 2))
    }

    #[test]
    fn dummy_classifier_most_frequent() {
        let y = array![0usize, 0, 1, 2, 0];
        let clf = DummyClassifier::new(DummyClassifierStrategy::MostFrequent);
        let fitted: FittedDummyClassifier = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        assert!(preds.iter().all(|&v| v == 0));
        assert_eq!(fitted.n_classes(), 3);
    }

    #[test]
    fn dummy_classifier_constant() {
        let y = array![0usize, 1, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Constant(2));
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        assert!(preds.iter().all(|&v| v == 2));
    }

    #[test]
    fn dummy_classifier_constant_invalid() {
        let y = array![0usize, 1];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Constant(99));
        assert!(clf.fit(&x(), &y).is_err());
    }

    #[test]
    fn dummy_classifier_stratified_in_range() {
        let y = array![0usize, 0, 0, 1, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Stratified).random_state(7);
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        for p in preds.iter() {
            assert!(matches!(*p, 0..=2));
        }
    }

    #[test]
    fn dummy_classifier_uniform_in_range() {
        let y = array![0usize, 1, 2];
        let clf = DummyClassifier::new(DummyClassifierStrategy::Uniform).random_state(5);
        let fitted = clf.fit(&x(), &y).unwrap();
        let preds = fitted.predict(&x()).unwrap();
        for p in preds.iter() {
            assert!(matches!(*p, 0..=2));
        }
    }

    // y = [0,0,1,1,2] -> classes_ [0,1,2], class_prior_ [0.4,0.4,0.2].
    // Oracle values from live sklearn 1.5.2 DummyClassifier.predict_proba
    // (sklearn/dummy.py:340-389). See .design/model-sel/dummy.md Verification.
    fn proba_y() -> Array1<usize> {
        array![0usize, 0, 1, 1, 2]
    }

    #[test]
    fn dummy_predict_proba_prior_matches_sklearn() -> Result<(), FerroError> {
        let clf = DummyClassifier::new(DummyClassifierStrategy::Prior);
        let fitted = clf.fit(&x(), &proba_y())?;
        let xq = Array2::<f64>::zeros((2, 2));
        let proba = fitted.predict_proba(&xq);
        assert_eq!(proba.shape(), &[2, 3]);
        for row in proba.rows() {
            assert!((row[0] - 0.4).abs() <= 1e-9);
            assert!((row[1] - 0.4).abs() <= 1e-9);
            assert!((row[2] - 0.2).abs() <= 1e-9);
            assert!((row.sum() - 1.0).abs() <= 1e-9);
        }
        Ok(())
    }

    #[test]
    fn dummy_predict_proba_most_frequent_matches_sklearn() -> Result<(), FerroError> {
        let clf = DummyClassifier::new(DummyClassifierStrategy::MostFrequent);
        let fitted = clf.fit(&x(), &proba_y())?;
        let xq = Array2::<f64>::zeros((2, 2));
        let proba = fitted.predict_proba(&xq);
        assert_eq!(proba.shape(), &[2, 3]);
        for row in proba.rows() {
            assert!((row[0] - 1.0).abs() <= 1e-9);
            assert!((row[1] - 0.0).abs() <= 1e-9);
            assert!((row[2] - 0.0).abs() <= 1e-9);
        }
        Ok(())
    }

    #[test]
    fn dummy_predict_proba_uniform_matches_sklearn() -> Result<(), FerroError> {
        let clf = DummyClassifier::new(DummyClassifierStrategy::Uniform);
        let fitted = clf.fit(&x(), &proba_y())?;
        let xq = Array2::<f64>::zeros((2, 2));
        let proba = fitted.predict_proba(&xq);
        assert_eq!(proba.shape(), &[2, 3]);
        for row in proba.rows() {
            for &v in row {
                assert!((v - 1.0 / 3.0).abs() <= 1e-9);
            }
            assert!((row.sum() - 1.0).abs() <= 1e-9);
        }
        Ok(())
    }

    #[test]
    fn dummy_predict_log_proba_matches_sklearn() -> Result<(), FerroError> {
        // prior: ln([0.4,0.4,0.2]) elementwise.
        let clf = DummyClassifier::new(DummyClassifierStrategy::Prior);
        let fitted = clf.fit(&x(), &proba_y())?;
        let xq = Array2::<f64>::zeros((2, 2));
        let logp = fitted.predict_log_proba(&xq);
        assert_eq!(logp.shape(), &[2, 3]);
        for row in logp.rows() {
            assert!((row[0] - 0.4f64.ln()).abs() <= 1e-9);
            assert!((row[1] - 0.4f64.ln()).abs() <= 1e-9);
            assert!((row[2] - 0.2f64.ln()).abs() <= 1e-9);
        }

        // most_frequent: zero-probability entries map to -inf (np.log(0.0)).
        let clf = DummyClassifier::new(DummyClassifierStrategy::MostFrequent);
        let fitted = clf.fit(&x(), &proba_y())?;
        let logp = fitted.predict_log_proba(&xq);
        for row in logp.rows() {
            assert!((row[0] - 0.0).abs() <= 1e-9);
            assert!(row[1].is_infinite() && row[1].is_sign_negative());
            assert!(row[2].is_infinite() && row[2].is_sign_negative());
        }
        Ok(())
    }

    #[test]
    fn dummy_regressor_mean() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Mean);
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_median() {
        let y: Array1<f64> = array![1.0, 5.0, 2.0, 4.0, 3.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Median);
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_quantile_25() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Quantile(0.25));
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dummy_regressor_constant() {
        let y: Array1<f64> = array![1.0, 2.0, 3.0];
        let reg = DummyRegressor::<f64>::new(DummyRegressorStrategy::Constant(42.0));
        let fitted = reg.fit(&x(), &y).unwrap();
        assert!((*fitted.constant() - 42.0).abs() < 1e-12);
        let preds = fitted.predict(&Array2::<f64>::zeros((4, 2))).unwrap();
        assert_eq!(preds.len(), 4);
    }
}
