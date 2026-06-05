//! Cross-validation utilities.
//!
//! This module provides:
//!
//! - [`KFold`] — k-fold cross-validation index splitter.
//! - [`StratifiedKFold`] — stratified k-fold that preserves class proportions.
//! - [`CrossValidator`] — a trait abstracting over fold-index generators.
//! - [`cross_val_score`] — run a [`ferrolearn_core::pipeline::Pipeline`]
//!   through each fold and collect scores.
//!
//! ## REQ status
//!
//! Mirrors `sklearn/model_selection/_validation.py` (cross_val_score :560,
//! cross_validate :122, cross_val_predict :1054, permutation_test_score :1502,
//! _fit_and_score :729) and `_split.py` (KFold :441, StratifiedKFold :665,
//! _make_test_folds :746) at v1.5.2. Every REQ is BINARY (R-DEFER-2): SHIPPED or
//! NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-KFOLD (non-shuffled KFold membership) | SHIPPED | consecutive folds, first `n%k` get +1 (`:521`); oracle `KFold(3).split(10)` → `[[0,1,2,3],[4,5,6],[7,8,9]]`. Guard `guard_kfold_membership_n10_k3`. |
//! | REQ-SKFOLD (non-shuffled StratifiedKFold membership) | SHIPPED | ported `_make_test_folds` (`:746-806`): appearance-order class encoding, `allocation[i][k]=bincount(y_order[i::n_splits])`, block-assign — was lexicographic-sort + rotating fold_offset. Matches oracle `[[0,1,4,8],[2,5,6,9],[3,7,10,11]]` (fixed #1791). Test `pin_1791_stratified_kfold_allocation`. |
//! | REQ-SKFOLD-ERRWARN (error only if ALL classes too small) | SHIPPED | errors only when every class count < n_splits (`:770-774`); a single small class still splits — was erroring if ANY class < n_splits (fixed #1792). Test `pin_1792_stratified_kfold_one_small_class_warns_not_errors`. (sklearn's UserWarning has no Rust analog.) |
//! | REQ-CVS (cross_val_score mechanic) | SHIPPED | per-fold fit/predict/score → per-fold test scores; mirrors `cv_results["test_score"]` (`:560`). Guard `guard_cross_val_score_per_fold_negmse`. |
//! | REQ-CVALIDATE (cross_validate test/train + timing) | SHIPPED | test+optional train scores + fit/score times (`:122`). Guard `guard_cross_validate_train_test_and_timing`. (return_estimator/return_indices/multimetric absent — REQ-DEFAULTS.) |
//! | REQ-CVPREDICT (original-order OOF placement) | SHIPPED | each OOF prediction placed at its original index (`:1054`). Guard `guard_cross_val_predict_original_order_placement`. |
//! | REQ-CVPREDICT-PARTITION (non-partition ⇒ error) | SHIPPED | requires a partition cv; non-partition (uncovered/duplicate/out-of-range test index) ⇒ `InvalidParameter`, mirroring `_check_is_permutation` (`:1054`) — was 0.0-filling silently (fixed #1793). Test `pin_1793_cross_val_predict_non_partition_must_error`. |
//! | REQ-PERM (permutation_test_score p-value) | SHIPPED | `(count(perm>=real)+1)/(n_perm+1)` (`:1697`), real_score = mean CV. Guard `guard_permutation_test_score_pvalue_formula`. Exact perm scores are RNG carve-out. |
//! | REQ-ERROR-SCORE (error_score=np.nan continue) | SHIPPED | `cross_val_score`/`cross_validate` NaN-fill a failing fold (fit ⇒ both; independent test/train scoring) and continue, matching default `error_score=np.nan` (`_fit_and_score :890-915`) — was `?`-aborting. Test `pin_1790_error_score_nan_continue`. This unit OWNS the blocker grid/random/curve units deferred here (S8). |
//! | REQ-SKFOLD-CV (StratifiedKFold as CrossValidator) | NOT-STARTED | trait `fold_indices(n_samples)` has no y channel, so `StratifiedKFold` can't be passed to `cross_val_score`/`GridSearchCV`. Architectural. Blocker #1794. |
//! | REQ-SHUFFLE-RNG (shuffle / permutation exact membership) | NOT-STARTED | `SmallRng` vs numpy — R-DEFER-3 carve-out (NO failing test); structural (sizes, partition, seed-determinism-across-runs) is SHIPPED. Blocker #1795. |
//! | REQ-DEFAULTS (cv=None/scoring=None/groups/n_jobs/return_estimator/multimetric) | NOT-STARTED | `cv`/`scoring` mandatory; no `check_cv`/`check_scoring`/groups/n_jobs (`:560`,`:122`). Blocker #1796. |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | `ndarray` + `rand`/`SmallRng`; destination `ferray-core` + `ferray::random`. Blocker #1797. |
//! | REQ-X-2 (non-test production consumer — widest in crate) | SHIPPED | `cross_val_score` called by `GridSearchCV`/`RandomizedSearchCV`/halving searches; `KFold` flows as `&dyn CrossValidator`; whole surface re-exported in `lib.rs`. |

use std::collections::HashMap;
use std::time::Instant;

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// A single cross-validation split: `(train_indices, test_indices)`.
pub type FoldSplit = (Vec<usize>, Vec<usize>);

/// The result type returned by [`CrossValidator::fold_indices`].
pub type FoldSplits = Vec<FoldSplit>;

// ---------------------------------------------------------------------------
// CrossValidator trait
// ---------------------------------------------------------------------------

/// A trait for objects that can generate fold indices.
///
/// Implement this trait to provide custom splitting strategies that can be
/// passed to [`cross_val_score`].
pub trait CrossValidator {
    /// Return `(train_indices, test_indices)` pairs for each fold.
    ///
    /// The number of pairs equals the number of folds/splits.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if the split cannot be produced (e.g. not
    /// enough samples for the requested number of splits).
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError>;
}

// ---------------------------------------------------------------------------
// KFold
// ---------------------------------------------------------------------------

/// K-fold cross-validation splitter.
///
/// Splits data into `n_splits` consecutive folds. Each fold is used once as a
/// test set while the remaining `k−1` folds form the training set.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::KFold;
///
/// let kf = KFold::new(5);
/// let folds = kf.split(20);
/// assert_eq!(folds.len(), 5);
/// for (train, test) in &folds {
///     assert_eq!(train.len() + test.len(), 20);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds.
    n_splits: usize,
    /// Whether to shuffle samples before splitting.
    shuffle: bool,
    /// Optional RNG seed used when `shuffle` is `true`.
    random_state: Option<u64>,
}

impl KFold {
    /// Create a new [`KFold`] with the given number of splits.
    ///
    /// By default shuffling is disabled.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid `n_splits` is caught at [`split`](KFold::split)
    /// time.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable or disable shuffling of samples before splitting.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the RNG seed used when shuffling is enabled.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Produce `(train_indices, test_indices)` pairs for each fold.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// Returns [`FerroError::InsufficientSamples`] if `n_samples < n_splits`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::KFold;
    ///
    /// let folds = KFold::new(3).split(9);
    /// // Each fold has 3 test samples.
    /// for (train, test) in folds {
    ///     assert_eq!(test.len(), 3);
    /// }
    /// ```
    pub fn split(&self, n_samples: usize) -> FoldSplits {
        // Errors are returned as an empty vec here; the `CrossValidator` impl
        // returns proper `Result`s.
        self.split_result(n_samples).unwrap_or_default()
    }

    /// Internal splitting logic that returns a `Result`.
    fn split_result(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        if n_samples < self.n_splits {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
                context: format!("KFold with n_splits={}", self.n_splits),
            });
        }

        // Build (possibly shuffled) index list.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            if let Some(seed) = self.random_state {
                let mut rng = SmallRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = SmallRng::from_os_rng();
                indices.shuffle(&mut rng);
            }
        }

        // Compute fold boundary positions.
        // Distribute the remainder across the first `n_samples % n_splits`
        // folds so that fold sizes differ by at most 1.
        let base_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        // fold_starts[i] is the start index of fold i in `indices`.
        let mut fold_starts = Vec::with_capacity(self.n_splits + 1);
        let mut pos = 0usize;
        for fold in 0..self.n_splits {
            fold_starts.push(pos);
            pos += base_size + if fold < remainder { 1 } else { 0 };
        }
        fold_starts.push(n_samples); // sentinel

        let mut folds = Vec::with_capacity(self.n_splits);
        for fold in 0..self.n_splits {
            let test_start = fold_starts[fold];
            let test_end = fold_starts[fold + 1];
            let test: Vec<usize> = indices[test_start..test_end].to_vec();
            let train: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

impl CrossValidator for KFold {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        self.split_result(n_samples)
    }
}

// ---------------------------------------------------------------------------
// StratifiedKFold
// ---------------------------------------------------------------------------

/// Stratified k-fold cross-validation splitter.
///
/// Like [`KFold`] but preserves the percentage of samples for each class in
/// every fold. This is useful when the target has imbalanced class counts.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::StratifiedKFold;
/// use ndarray::array;
///
/// let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
/// let skf = StratifiedKFold::new(3);
/// let folds = skf.split(&y).unwrap();
/// assert_eq!(folds.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds.
    n_splits: usize,
    /// Whether to shuffle within each stratum before assigning to folds.
    shuffle: bool,
    /// Optional RNG seed used when `shuffle` is `true`.
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new [`StratifiedKFold`] with the given number of splits.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable or disable shuffling within each stratum before splitting.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the RNG seed used when shuffling is enabled.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate `(train_indices, test_indices)` pairs for each fold,
    /// preserving class distribution.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// - [`FerroError::InsufficientSamples`] if `n_samples < n_splits`, or if
    ///   EVERY class has fewer than `n_splits` members. Mirroring scikit-learn's
    ///   `StratifiedKFold._make_test_folds` (`_split.py:770-781`), an error is
    ///   raised only when ALL classes are too small; if only SOME classes are
    ///   too small the split still proceeds (sklearn emits a `UserWarning`
    ///   there, which ferrolearn has no channel for).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::StratifiedKFold;
    /// use ndarray::Array1;
    ///
    /// let y: Array1<usize> = Array1::from_iter(
    ///     (0..12).map(|i| i % 3)
    /// );
    /// let skf = StratifiedKFold::new(3);
    /// let folds = skf.split(&y).unwrap();
    /// assert_eq!(folds.len(), 3);
    /// for (train, test) in &folds {
    ///     assert_eq!(train.len() + test.len(), 12);
    /// }
    /// ```
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        let n_samples = y.len();

        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        if n_samples < self.n_splits {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
                context: format!("StratifiedKFold with n_splits={}", self.n_splits),
            });
        }

        // Faithful port of scikit-learn's
        // `StratifiedKFold._make_test_folds` (`_split.py:746-806`).

        // (a) Encode classes by ORDER OF APPEARANCE in `y` (`:760-765`): the
        // first distinct label seen gets encoding 0, the next 1, etc.
        let mut encoding: HashMap<usize, usize> = HashMap::new();
        let mut y_encoded: Vec<usize> = Vec::with_capacity(n_samples);
        for &label in y.iter() {
            let next = encoding.len();
            let code = *encoding.entry(label).or_insert(next);
            y_encoded.push(code);
        }
        let n_classes = encoding.len();

        // y_counts = number of samples in each encoded class (`:768`).
        let mut y_counts = vec![0usize; n_classes];
        for &code in &y_encoded {
            y_counts[code] += 1;
        }

        // (#1792) error-vs-warn: sklearn raises ONLY when EVERY class is too
        // small (`:770-774`); if only SOME are too small it warns and still
        // splits (`:775-781`). ferrolearn has no warning channel, so it simply
        // proceeds in the warn-case.
        if y_counts.iter().all(|&c| c < self.n_splits) {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: y_counts.iter().copied().min().unwrap_or(0),
                context: format!(
                    "StratifiedKFold: n_splits={} cannot be greater than the \
                     number of members in each class",
                    self.n_splits
                ),
            });
        }

        // (b) Allocation: for fold i and class k, the count of class-k labels
        // among every `n_splits`-th element of the sorted encoded y, i.e.
        // `bincount(y_order[i::n_splits])` (`:786-792`).
        let mut y_order = y_encoded.clone();
        y_order.sort_unstable();
        let mut allocation: Vec<Vec<usize>> = vec![vec![0usize; n_classes]; self.n_splits];
        for (i, alloc_row) in allocation.iter_mut().enumerate() {
            let mut pos = i;
            while pos < y_order.len() {
                alloc_row[y_order[pos]] += 1;
                pos += self.n_splits;
            }
        }

        // Optional RNG for shuffling fold blocks (exact shuffled membership is
        // an RNG carve-out, #1795).
        let mut rng: Option<SmallRng> = if self.shuffle {
            Some(match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed),
                None => SmallRng::from_os_rng(),
            })
        } else {
            None
        };

        // (c) Assignment: for each class k, build `folds_for_class` = fold index
        // f repeated `allocation[f][k]` times (`:798-805`), optionally shuffle,
        // then walk the original-order samples of class k and assign their fold.
        let mut test_fold_of = vec![0usize; n_samples];
        for k in 0..n_classes {
            let mut folds_for_class: Vec<usize> = Vec::with_capacity(y_counts[k]);
            for (f, alloc_row) in allocation.iter().enumerate() {
                for _ in 0..alloc_row[k] {
                    folds_for_class.push(f);
                }
            }
            if let Some(rng) = rng.as_mut() {
                folds_for_class.shuffle(rng);
            }
            // Walk class-k samples in ascending original index. By construction
            // `folds_for_class.len() == y_counts[k]`, so every slot is filled.
            let mut j = 0;
            for (i, &code) in y_encoded.iter().enumerate() {
                if code == k {
                    if let Some(&f) = folds_for_class.get(j) {
                        test_fold_of[i] = f;
                    }
                    j += 1;
                }
            }
        }

        // (d) Build (train, test) pairs in ascending index order.
        let mut folds = Vec::with_capacity(self.n_splits);
        for f in 0..self.n_splits {
            let mut test: Vec<usize> = Vec::new();
            let mut train: Vec<usize> = Vec::new();
            for (i, &fold) in test_fold_of.iter().enumerate() {
                if fold == f {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push((train, test));
        }
        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// cross_val_score
// ---------------------------------------------------------------------------

/// Evaluate a [`Pipeline`] using cross-validation and return per-fold scores.
///
/// For each fold produced by `cv`:
///
/// 1. Extract training and test subsets of `x` and `y`.
/// 2. Fit the pipeline on the training subset.
/// 3. Predict on the test subset.
/// 4. Compute the score using the `scoring` function.
///
/// The pipeline is cloned conceptually by fitting a fresh copy for each fold;
/// the original `pipeline` is not mutated.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] (e.g. [`KFold`]) that produces fold indices.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`
///   used to score each fold.
///
/// # Returns
///
/// An [`Array1<f64>`] of length `n_folds` containing the score for each fold.
///
/// # Errors
///
/// Propagates structural errors from input validation and fold splitting
/// (e.g. `y` length mismatch, invalid `n_splits`, or subset construction).
/// Per-fold ESTIMATOR failures (fit / predict / scoring) do NOT abort the
/// call: mirroring scikit-learn's default `error_score=np.nan`, the failing
/// fold's score is set to `f64::NAN` and evaluation continues. The returned
/// array may therefore contain NaN entries.
///
/// # Examples
///
/// ```rust,no_run
/// use ferrolearn_model_sel::{KFold, cross_val_score};
/// use ferrolearn_core::pipeline::Pipeline;
/// use ferrolearn_core::FerroError;
/// use ndarray::{Array1, Array2};
///
/// fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
///     let diff = y_true - y_pred;
///     Ok(diff.mapv(|v| v * v).mean().unwrap_or(0.0))
/// }
///
/// // pipeline must have an estimator step set before calling cross_val_score.
/// // let scores = cross_val_score(&pipeline, &x, &y, &KFold::new(5), mse).unwrap();
/// ```
pub fn cross_val_score(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
) -> Result<Array1<f64>, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "cross_val_score: y length must equal x number of rows".into(),
        });
    }

    let folds = cv.fold_indices(n_samples)?;
    let mut scores = Vec::with_capacity(folds.len());

    for (train_idx, test_idx) in &folds {
        // Build training subset.
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = x.ncols();

        let mut x_train_data = Vec::with_capacity(n_train * n_features);
        for &i in train_idx {
            x_train_data.extend(x.row(i).iter().copied());
        }
        let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_train".into(),
                reason: e.to_string(),
            }
        })?;
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();

        // Build test subset.
        let mut x_test_data = Vec::with_capacity(n_test * n_features);
        for &i in test_idx {
            x_test_data.extend(x.row(i).iter().copied());
        }
        let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_test".into(),
                reason: e.to_string(),
            }
        })?;
        let y_test: Array1<f64> = test_idx.iter().map(|&i| y[i]).collect();

        // Fit, predict, and score this fold. Mirroring scikit-learn's default
        // `error_score=np.nan` in `_fit_and_score` (`_validation.py:890-915`):
        // a fit failure substitutes `error_score` (NaN) for the fold and
        // CONTINUES; on a successful fit, a predict/score failure substitutes
        // NaN for only this fold. Per-fold ESTIMATOR failures never abort the
        // whole call (structural/shape errors above remain hard errors).
        let test_score = match pipeline.fit(&x_train, &y_train) {
            Err(_) => f64::NAN,
            Ok(fitted) => (|| {
                let y_pred = fitted.predict(&x_test)?;
                scoring(&y_test, &y_pred)
            })()
            .unwrap_or(f64::NAN),
        };
        scores.push(test_score);
    }

    Ok(Array1::from_vec(scores))
}

// ---------------------------------------------------------------------------
// CrossValidateResult
// ---------------------------------------------------------------------------

/// Detailed results from [`cross_validate`].
///
/// Unlike [`cross_val_score`], which returns only the per-fold test scores,
/// this struct also carries optional training scores and timing information
/// for each fold.
#[derive(Debug, Clone)]
pub struct CrossValidateResult {
    /// Test-set score for each fold.
    pub test_scores: Vec<f64>,
    /// Training-set score for each fold (`Some` only when
    /// `return_train_score` was `true`).
    pub train_scores: Option<Vec<f64>>,
    /// Wall-clock time in seconds spent fitting each fold.
    pub fit_times: Vec<f64>,
    /// Wall-clock time in seconds spent scoring each fold.
    pub score_times: Vec<f64>,
}

// ---------------------------------------------------------------------------
// cross_validate
// ---------------------------------------------------------------------------

/// Evaluate a [`Pipeline`] using cross-validation, returning detailed results.
///
/// This function is a richer version of [`cross_val_score`]. For each fold
/// produced by `cv`:
///
/// 1. Extract training and test subsets of `x` and `y`.
/// 2. Fit the pipeline on the training subset (timing the fit).
/// 3. Predict on the test subset and score with `scoring` (timing the score).
/// 4. Optionally score on the training subset as well.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] (e.g. [`KFold`]) that produces fold indices.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`
///   used to score each fold.
/// - `return_train_score` — When `true`, compute and return training scores.
///
/// # Returns
///
/// A [`CrossValidateResult`] containing per-fold test scores, optional
/// training scores, fit times, and score times.
///
/// # Errors
///
/// Propagates structural errors from input validation and fold splitting
/// (e.g. `y` length mismatch, invalid `n_splits`, or subset construction).
/// Per-fold ESTIMATOR failures (fit / predict / scoring) do NOT abort the
/// call: mirroring scikit-learn's default `error_score=np.nan`, a fit failure
/// sets both the test and (when `return_train_score`) train score for that
/// fold to `f64::NAN`, and a predict/score failure sets only the affected
/// score to NaN. Evaluation then continues, so the returned scores may
/// contain NaN entries.
pub fn cross_validate(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    return_train_score: bool,
) -> Result<CrossValidateResult, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "cross_validate: y length must equal x number of rows".into(),
        });
    }

    let folds = cv.fold_indices(n_samples)?;
    let n_folds = folds.len();
    let n_features = x.ncols();

    let mut test_scores = Vec::with_capacity(n_folds);
    let mut train_scores_vec = if return_train_score {
        Some(Vec::with_capacity(n_folds))
    } else {
        None
    };
    let mut fit_times = Vec::with_capacity(n_folds);
    let mut score_times = Vec::with_capacity(n_folds);

    for (train_idx, test_idx) in &folds {
        let n_train = train_idx.len();
        let n_test = test_idx.len();

        // Build training subset.
        let mut x_train_data = Vec::with_capacity(n_train * n_features);
        for &i in train_idx {
            x_train_data.extend(x.row(i).iter().copied());
        }
        let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_train".into(),
                reason: e.to_string(),
            }
        })?;
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();

        // Build test subset.
        let mut x_test_data = Vec::with_capacity(n_test * n_features);
        for &i in test_idx {
            x_test_data.extend(x.row(i).iter().copied());
        }
        let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_test".into(),
                reason: e.to_string(),
            }
        })?;
        let y_test: Array1<f64> = test_idx.iter().map(|&i| y[i]).collect();

        // Fit pipeline (timed).
        let fit_start = Instant::now();
        let fit_result = pipeline.fit(&x_train, &y_train);
        let fit_elapsed = fit_start.elapsed().as_secs_f64();
        fit_times.push(fit_elapsed);

        // Score (timed). Mirroring scikit-learn's default `error_score=np.nan`
        // in `_fit_and_score` (`_validation.py:890-915`): a fit failure
        // substitutes `error_score` (NaN) for BOTH the test and (when
        // requested) the train score and CONTINUES. On a successful fit, the
        // test and train scores are computed via TWO INDEPENDENT score
        // closures (`:910`/`:915`), each substituting NaN for only its own
        // failing scorer. Per-fold ESTIMATOR failures never abort the whole
        // call (structural/shape errors above remain hard errors).
        let score_start = Instant::now();
        let (test_score, train_score_opt) = match fit_result {
            Err(_) => (
                f64::NAN,
                if return_train_score {
                    Some(f64::NAN)
                } else {
                    None
                },
            ),
            Ok(fitted) => {
                let test_score = (|| {
                    let y_pred = fitted.predict(&x_test)?;
                    scoring(&y_test, &y_pred)
                })()
                .unwrap_or(f64::NAN);
                let train_score = if return_train_score {
                    Some(
                        (|| {
                            let y_train_pred = fitted.predict(&x_train)?;
                            scoring(&y_train, &y_train_pred)
                        })()
                        .unwrap_or(f64::NAN),
                    )
                } else {
                    None
                };
                (test_score, train_score)
            }
        };
        let score_elapsed = score_start.elapsed().as_secs_f64();
        score_times.push(score_elapsed);
        test_scores.push(test_score);

        if let (Some(ref mut ts), Some(tr)) = (train_scores_vec.as_mut(), train_score_opt) {
            ts.push(tr);
        }
    }

    Ok(CrossValidateResult {
        test_scores,
        train_scores: train_scores_vec,
        fit_times,
        score_times,
    })
}

// ---------------------------------------------------------------------------
// cross_val_predict
// ---------------------------------------------------------------------------

/// Generate out-of-fold predictions for every sample.
///
/// For each fold produced by `cv`, the pipeline is fit on the training split
/// and then used to predict the test split. Each sample is predicted exactly
/// once — when it appears in a test fold.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] (e.g. [`KFold`]) that produces fold indices.
///
/// # Returns
///
/// An [`Array1<f64>`] of length `n_samples` containing the out-of-fold
/// prediction for each sample.
///
/// # Errors
///
/// Propagates any error from fold splitting, model fitting, or predicting.
/// Additionally, mirroring scikit-learn's `_check_is_permutation`
/// (`sklearn/model_selection/_validation.py:1054`), the per-fold test indices
/// MUST form a partition (permutation) of `0..n_samples`: every sample must
/// appear in exactly one test fold. If the `cv` is not a partition (a sample is
/// never tested, or is tested more than once, or an index is out of range),
/// returns `Err(FerroError::InvalidParameter)` rather than silently 0.0-filling
/// untested samples.
pub fn cross_val_predict(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
) -> Result<Array1<f64>, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "cross_val_predict: y length must equal x number of rows".into(),
        });
    }

    let folds = cv.fold_indices(n_samples)?;

    // Mirror scikit-learn's `_check_is_permutation` (`_validation.py:1054`):
    // the concatenated per-fold test indices must be a permutation of
    // `0..n_samples` (every sample tested exactly once). Otherwise sklearn
    // raises `ValueError("cross_val_predict only works for partitions")`; we
    // map that to `InvalidParameter` (R-DEV-2) instead of silently 0.0-filling
    // untested samples.
    let mut covered = vec![false; n_samples];
    for (_, test_idx) in &folds {
        for &i in test_idx {
            if i >= n_samples || covered[i] {
                return Err(FerroError::InvalidParameter {
                    name: "cv".into(),
                    reason: "cross_val_predict only works for partitions \
                             (each sample must appear in exactly one test fold)"
                        .into(),
                });
            }
            covered[i] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        return Err(FerroError::InvalidParameter {
            name: "cv".into(),
            reason: "cross_val_predict only works for partitions \
                     (each sample must appear in exactly one test fold)"
                .into(),
        });
    }

    let n_features = x.ncols();

    let mut predictions = Array1::<f64>::zeros(n_samples);

    for (train_idx, test_idx) in &folds {
        let n_train = train_idx.len();
        let n_test = test_idx.len();

        // Build training subset.
        let mut x_train_data = Vec::with_capacity(n_train * n_features);
        for &i in train_idx {
            x_train_data.extend(x.row(i).iter().copied());
        }
        let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_train".into(),
                reason: e.to_string(),
            }
        })?;
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();

        // Build test subset.
        let mut x_test_data = Vec::with_capacity(n_test * n_features);
        for &i in test_idx {
            x_test_data.extend(x.row(i).iter().copied());
        }
        let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_test".into(),
                reason: e.to_string(),
            }
        })?;

        // Fit and predict.
        let fitted = pipeline.fit(&x_train, &y_train)?;
        let y_pred = fitted.predict(&x_test)?;

        // Place predictions at the correct indices.
        for (j, &idx) in test_idx.iter().enumerate() {
            predictions[idx] = y_pred[j];
        }
    }

    Ok(predictions)
}

// ---------------------------------------------------------------------------
// permutation_test_score
// ---------------------------------------------------------------------------

/// Evaluate the statistical significance of a cross-validated score by
/// permutation testing.
///
/// The function computes the real cross-validated score, then permutes the
/// target `y` `n_permutations` times, re-computing the CV score each time.
/// The p-value is the fraction of permuted scores that are greater than or
/// equal to the real score.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] (e.g. [`KFold`]) that produces fold indices.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`
///   used to score each fold.
/// - `n_permutations` — Number of random permutations to perform.
/// - `random_state` — Optional RNG seed for reproducibility.
///
/// # Returns
///
/// A tuple `(real_score, permuted_scores, p_value)`:
///
/// - `real_score` — mean CV score on the original (unpermuted) data.
/// - `permuted_scores` — mean CV score for each permutation.
/// - `p_value` — fraction of permuted scores >= `real_score`.
///
/// # Errors
///
/// Propagates any error from fold splitting, model fitting, predicting, or
/// scoring.
///
/// # Examples
///
/// ```rust,no_run
/// use ferrolearn_model_sel::cross_validation::permutation_test_score;
/// use ferrolearn_model_sel::KFold;
/// use ferrolearn_core::pipeline::Pipeline;
/// use ferrolearn_core::FerroError;
/// use ndarray::{Array1, Array2};
///
/// fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
///     let diff = y_true - y_pred;
///     Ok(-(diff.mapv(|v| v * v).mean().unwrap_or(0.0)))
/// }
///
/// // let (score, perm_scores, pval) = permutation_test_score(
/// //     &pipeline, &x, &y, &KFold::new(5), neg_mse, 100, Some(42),
/// // ).unwrap();
/// ```
pub fn permutation_test_score(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
    n_permutations: usize,
    random_state: Option<u64>,
) -> Result<(f64, Vec<f64>, f64), FerroError> {
    // Compute real score (mean of fold scores).
    let real_scores = cross_val_score(pipeline, x, y, cv, scoring)?;
    let real_score = real_scores.mean().unwrap_or(0.0);

    // Permutation loop.
    let n_samples = y.len();
    let mut rng: SmallRng = match random_state {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_os_rng(),
    };

    let mut perm_scores = Vec::with_capacity(n_permutations);
    let mut indices: Vec<usize> = (0..n_samples).collect();

    for _ in 0..n_permutations {
        // Shuffle indices to produce a permuted y.
        indices.shuffle(&mut rng);
        let y_perm: Array1<f64> = indices.iter().map(|&i| y[i]).collect();

        let fold_scores = cross_val_score(pipeline, x, &y_perm, cv, scoring)?;
        perm_scores.push(fold_scores.mean().unwrap_or(0.0));
    }

    // p-value = fraction of permuted scores >= real score.
    let n_ge = perm_scores.iter().filter(|&&s| s >= real_score).count();
    let p_value = (n_ge as f64 + 1.0) / (n_permutations as f64 + 1.0);

    Ok((real_score, perm_scores, p_value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{
        FittedPipelineEstimator as FittedEstTrait, FittedPipelineTransformer, Pipeline,
        PipelineEstimator, PipelineTransformer,
    };
    use ndarray::{Array1, Array2, array};

    // -- KFold tests ---------------------------------------------------------

    #[test]
    fn test_kfold_basic() {
        let kf = KFold::new(5);
        let folds = kf.split(20);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 20);
        }
    }

    #[test]
    fn test_kfold_fold_sizes_equal() {
        let folds = KFold::new(4).split(20);
        for (_, test) in &folds {
            assert_eq!(test.len(), 5);
        }
    }

    #[test]
    fn test_kfold_fold_sizes_unequal() {
        // 10 samples, 3 folds → sizes 4, 3, 3.
        let folds = KFold::new(3).split(10);
        let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(test_sizes.iter().sum::<usize>(), 10);
        // Max and min sizes differ by at most 1.
        let max = *test_sizes.iter().max().unwrap();
        let min = *test_sizes.iter().min().unwrap();
        assert!(max - min <= 1);
    }

    #[test]
    fn test_kfold_no_overlap_full_coverage() {
        let folds = KFold::new(5).split(10);
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_kfold_shuffle_deterministic() {
        let kf = KFold::new(5).shuffle(true).random_state(42);
        let folds1 = kf.split(20);
        let folds2 = kf.split(20);
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_kfold_shuffle_differs_from_no_shuffle() {
        let folds_no = KFold::new(5).split(20);
        let folds_sh = KFold::new(5).shuffle(true).random_state(1).split(20);
        // At least one fold should differ (extremely unlikely not to).
        let different = folds_no.iter().zip(folds_sh.iter()).any(|(a, b)| a != b);
        assert!(different);
    }

    #[test]
    fn test_kfold_invalid_n_splits() {
        let kf = KFold::new(1);
        let result = kf.fold_indices(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_insufficient_samples() {
        let kf = KFold::new(5);
        let result = kf.fold_indices(3);
        assert!(result.is_err());
    }

    // -- StratifiedKFold tests -----------------------------------------------

    #[test]
    fn test_skfold_basic() {
        // 3 classes, 3 samples each, 3 folds → each fold test has 1 of each.
        let y: Array1<usize> = Array1::from_iter((0..9).map(|i| i % 3));
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 9);
        }
    }

    #[test]
    fn test_skfold_coverage() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();

        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..12).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_skfold_class_balance() {
        // 6 samples: 3 class-0 and 3 class-1.
        let y = array![0usize, 0, 0, 1, 1, 1];
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();
        // Each fold's test should have 1 from each class.
        for (_, test) in &folds {
            let class0 = test.iter().filter(|&&i| y[i] == 0).count();
            let class1 = test.iter().filter(|&&i| y[i] == 1).count();
            assert_eq!(class0, 1, "expected 1 class-0 sample per test fold");
            assert_eq!(class1, 1, "expected 1 class-1 sample per test fold");
        }
    }

    #[test]
    fn test_skfold_shuffle_deterministic() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let skf = StratifiedKFold::new(3).shuffle(true).random_state(99);
        let folds1 = skf.split(&y).unwrap();
        let folds2 = skf.split(&y).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_skfold_invalid_n_splits() {
        let y = array![0usize, 1, 2];
        let skf = StratifiedKFold::new(1);
        assert!(skf.split(&y).is_err());
    }

    #[test]
    fn test_skfold_one_small_class_still_splits() {
        // Class 2 has only 1 sample (< n_splits=3) but classes 0 and 1 each
        // have 3 >= 3. Mirroring sklearn's `_make_test_folds` (`_split.py:770`),
        // an error is raised ONLY when EVERY class is too small; here only one
        // class is, so sklearn warns and still splits. Live sklearn 1.5.2 oracle
        // for y=[0,0,0,1,1,1,2], StratifiedKFold(3) => 3 folds partitioning 0..7.
        let y = array![0usize, 0, 0, 1, 1, 1, 2];
        let result = StratifiedKFold::new(3).split(&y);
        assert!(result.is_ok(), "one-small-class split should succeed");
        let folds = result.unwrap_or_default();
        assert_eq!(folds.len(), 3);
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        assert_eq!(all_test, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn test_skfold_all_classes_too_small_errors() {
        // EVERY class has fewer than n_splits members => sklearn raises and so
        // does ferrolearn. y=[0,1] with n_splits=3: both classes have 1 < 3
        // (also n_samples=2 < n_splits=3, the sklearn-correct earlier guard).
        let y = array![0usize, 1];
        assert!(StratifiedKFold::new(3).split(&y).is_err());
    }

    // -- cross_val_score tests ------------------------------------------------

    /// A trivial pipeline estimator that always predicts the mean of y_train.
    struct MeanEstimator;

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedEstTrait<f64>>, FerroError> {
            let mean = y.mean().unwrap_or(0.0);
            Ok(Box::new(FittedMean { mean }))
        }
    }

    struct FittedMean {
        mean: f64,
    }

    impl FittedEstTrait<f64> for FittedMean {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    /// Identity transformer (pass-through).
    struct IdentityTransformer;

    impl PipelineTransformer<f64> for IdentityTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedIdentity))
        }
    }

    struct FittedIdentity;

    impl FittedPipelineTransformer<f64> for FittedIdentity {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.clone())
        }
    }

    fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    #[test]
    fn test_cross_val_score_returns_correct_number_of_scores() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(20);
        let kf = KFold::new(5);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_cross_val_score_perfect_constant_target() {
        // When y is constant and the estimator predicts the mean, MSE = 0.
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 5.0);
        let kf = KFold::new(5);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        for &s in &scores {
            assert!(s.abs() < 1e-10, "expected 0 MSE, got {s}");
        }
    }

    #[test]
    fn test_cross_val_score_with_transformer() {
        let pipeline = Pipeline::new()
            .transform_step("identity", Box::new(IdentityTransformer))
            .estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((15, 2));
        let y = Array1::<f64>::from_elem(15, 3.0);
        let kf = KFold::new(3);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        assert_eq!(scores.len(), 3);
        for &s in &scores {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_val_score_shape_mismatch() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(18); // wrong length
        let kf = KFold::new(5);
        assert!(cross_val_score(&pipeline, &x, &y, &kf, mse).is_err());
    }

    // -- cross_validate tests ------------------------------------------------

    #[test]
    fn test_cross_validate_returns_correct_fold_count() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(20);
        let kf = KFold::new(5);
        let result = cross_validate(&pipeline, &x, &y, &kf, mse, false).unwrap();
        assert_eq!(result.test_scores.len(), 5);
        assert_eq!(result.fit_times.len(), 5);
        assert_eq!(result.score_times.len(), 5);
        assert!(result.train_scores.is_none());
    }

    #[test]
    fn test_cross_validate_with_train_scores() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 5.0);
        let kf = KFold::new(5);
        let result = cross_validate(&pipeline, &x, &y, &kf, mse, true).unwrap();
        assert!(result.train_scores.is_some());
        let ts = result.train_scores.unwrap();
        assert_eq!(ts.len(), 5);
        // Mean estimator on constant y: train score should also be ~0.
        for &s in &ts {
            assert!(s.abs() < 1e-10, "expected ~0 train MSE, got {s}");
        }
    }

    #[test]
    fn test_cross_validate_perfect_constant_target() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 5.0);
        let kf = KFold::new(5);
        let result = cross_validate(&pipeline, &x, &y, &kf, mse, false).unwrap();
        for &s in &result.test_scores {
            assert!(s.abs() < 1e-10, "expected 0 MSE, got {s}");
        }
    }

    #[test]
    fn test_cross_validate_timing_non_negative() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(20);
        let kf = KFold::new(3);
        let result = cross_validate(&pipeline, &x, &y, &kf, mse, false).unwrap();
        for &t in &result.fit_times {
            assert!(t >= 0.0, "fit time should be non-negative");
        }
        for &t in &result.score_times {
            assert!(t >= 0.0, "score time should be non-negative");
        }
    }

    #[test]
    fn test_cross_validate_shape_mismatch() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(18);
        let kf = KFold::new(5);
        assert!(cross_validate(&pipeline, &x, &y, &kf, mse, false).is_err());
    }

    // -- cross_val_predict tests ---------------------------------------------

    #[test]
    fn test_cross_val_predict_length() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(20);
        let kf = KFold::new(5);
        let preds = cross_val_predict(&pipeline, &x, &y, &kf).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_cross_val_predict_constant_target() {
        // MeanEstimator on constant y should predict that constant for every sample.
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 7.0);
        let kf = KFold::new(5);
        let preds = cross_val_predict(&pipeline, &x, &y, &kf).unwrap();
        for &p in &preds {
            assert!((p - 7.0).abs() < 1e-10, "expected 7.0, got {p}");
        }
    }

    #[test]
    fn test_cross_val_predict_shape_mismatch() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(18);
        let kf = KFold::new(5);
        assert!(cross_val_predict(&pipeline, &x, &y, &kf).is_err());
    }

    #[test]
    fn test_cross_val_predict_with_transformer() {
        let pipeline = Pipeline::new()
            .transform_step("identity", Box::new(IdentityTransformer))
            .estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((15, 2));
        let y = Array1::<f64>::from_elem(15, 3.0);
        let kf = KFold::new(3);
        let preds = cross_val_predict(&pipeline, &x, &y, &kf).unwrap();
        assert_eq!(preds.len(), 15);
        for &p in &preds {
            assert!((p - 3.0).abs() < 1e-10, "expected 3.0, got {p}");
        }
    }

    // -- permutation_test_score tests ----------------------------------------

    fn neg_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(-(diff.mapv(|v| v * v).mean().unwrap_or(0.0)))
    }

    #[test]
    fn test_permutation_test_score_returns_correct_counts() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 5.0);
        let kf = KFold::new(5);
        let (score, perm_scores, p_value) =
            permutation_test_score(&pipeline, &x, &y, &kf, neg_mse, 10, Some(42)).unwrap();
        assert_eq!(perm_scores.len(), 10);
        // For constant y, real score = 0 (neg MSE), and permuted scores should also be 0.
        assert!((score - 0.0).abs() < 1e-10);
        assert!((0.0..=1.0).contains(&p_value));
    }

    #[test]
    fn test_permutation_test_score_deterministic() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y: Array1<f64> = Array1::from_iter((0..20).map(f64::from));
        let kf = KFold::new(5);
        let (s1, ps1, p1) =
            permutation_test_score(&pipeline, &x, &y, &kf, neg_mse, 5, Some(42)).unwrap();
        let (s2, ps2, p2) =
            permutation_test_score(&pipeline, &x, &y, &kf, neg_mse, 5, Some(42)).unwrap();
        assert!((s1 - s2).abs() < 1e-10);
        assert!((p1 - p2).abs() < 1e-10);
        for (a, b) in ps1.iter().zip(ps2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_permutation_test_score_p_value_range() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y: Array1<f64> = Array1::from_iter((0..20).map(f64::from));
        let kf = KFold::new(5);
        let (_, _, p_value) =
            permutation_test_score(&pipeline, &x, &y, &kf, neg_mse, 20, Some(42)).unwrap();
        assert!(p_value > 0.0, "p_value should be positive");
        assert!(p_value <= 1.0, "p_value should be <= 1");
    }
}
