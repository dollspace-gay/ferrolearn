//! Group-aware cross-validation splitters.
//!
//! Each splitter takes an `Array1<usize>` of group labels (one per sample)
//! and produces folds where samples sharing a group never appear in both the
//! train and test split — useful when samples are not independent.
//!
//! - [`GroupKFold`] — partition `n_groups` distinct groups into `n_splits`
//!   roughly-equal folds.
//! - [`GroupShuffleSplit`] — random group-wise train/test splits.
//! - [`LeaveOneGroupOut`] — one fold per unique group.
//! - [`LeavePGroupsOut`] — one fold per `p`-sized subset of groups.
//! - [`StratifiedGroupKFold`] — group-aware folding that also tries to
//!   preserve class balance per fold.
//!
//! ## REQ status
//!
//! Mirrors `sklearn/model_selection/_split.py` (1.5.2). Every REQ is BINARY
//! (R-DEFER-2): SHIPPED (end-to-end functional + non-test consumer + tests +
//! verification) or NOT-STARTED (with a concrete blocker).
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-GKF-1 (greedy mechanics: argmin-load, no group straddle, validation) | SHIPPED | first-min argmin over fold loads mirrors `np.argmin(n_samples_per_fold)` (`:628-636`); rejects `n_splits < 2` and `n_splits > n_groups`. |
//! | REQ-GKF-2 (deterministic descending-size ordering + oracle membership) | SHIPPED | `(count desc, group_id desc)` over a `BTreeMap` reproduces `np.argsort(np.bincount)[::-1]` (`:618`); deterministic (no HashMap feeds ordering); matches live oracle `GroupKFold(3).split([0,0,1,1,2,2,3,3])` → `[[0,3],[2],[1]]`. Test `gkf_greedy_ordering_diverges`. CARVE-OUT #1755: the exact tie order among equal-count distinct-id groups follows numpy's UNSTABLE quicksort — a numpy-version-defined artifact with no stable sklearn contract; ferrolearn uses a deterministic rule, guarded by `gkf_tiebreak_deterministic_and_balanced` (determinism + balance, not numpy internals). |
//! | REQ-GKF-3 (default `n_splits=5`) | NOT-STARTED | API-shape gap — `new(n_splits)` requires it positionally; no-arg sklearn form unrepresentable. Blocker #1754. |
//! | REQ-SGKF-1 (greedy objective parity — std-minimisation) | SHIPPED | ports `_iter_test_indices` + `_find_best_fold` (`:1015-1059`): descending per-group population-std (ddof=0) ordering (stable, tie ascending group index) + fold choice minimising mean per-class std of `y_counts_per_fold / y_cnt`, `np.isclose` tie-break on fewest samples. Matches live oracle `StratifiedGroupKFold(3)` → `[[3,6,7],[1,2,8],[4,5]]`; empty-fold bug gone. Test `sgkf_objective_diverges`. |
//! | REQ-SGKF-2 (structural + length/n_splits validation) | SHIPPED | `y.len()==groups.len()` → `ShapeMismatch`; whole groups; `n_splits` folds; rejects `n_splits < 2`/`> n_groups`. |
//! | REQ-SGKF-3 (per-class-count `ValueError` + UserWarning + default `n_splits`) | NOT-STARTED | no per-class-member check, no warning channel, requires explicit `n_splits` (`:987-999`). Blocker #1754. |
//! | REQ-LOGO-1 (split-index parity) | SHIPPED | one fold per ascending unique group (`:1330-1337`); oracle `[0,0,1,1,2]` → `([2,3,4],[0,1])`/`([0,1,4],[2,3])`/`([0,1,2,3],[4])`. Guard `logo_split_index_parity`. |
//! | REQ-LOGO-2 (`< 2` unique-groups rejection) | SHIPPED | `unique.len() < 2` → `InvalidParameter` (`:1331-1335`). |
//! | REQ-LPGO-1 (combination-order parity) | SHIPPED | lexicographic next-combination reproduces `combinations(range(n_groups), p)` ORDER (`:1465-1470`); oracle `[0,1,2,3]`,p=2 → `{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}`. Guard `lpgo_combination_order_parity`. |
//! | REQ-LPGO-2 (error semantics) | SHIPPED | `p == 0` and `unique.len() <= p` → `InvalidParameter` (`:1458-1464`). |
//! | REQ-GSS-1 (structural contract) | SHIPPED | per-split group shuffle → first `n_test` as test groups → sample-level disjoint covering partition; `random_state` reproducible. |
//! | REQ-GSS-2 (test-GROUP-count sizing parity) | SHIPPED | `n_test = ceil(test_size * n_groups)` mirrors `_validate_shuffle_split` (`:2389-2390`); oracle 7 groups @ 0.3 → 3. Test `gss_test_group_count_round_vs_ceil`. |
//! | REQ-GSS-3 (default `test_size`/`n_splits` + exact-membership carve-out) | NOT-STARTED | (a) `new(n_splits, test_size)` requires both; no-arg sklearn form unrepresentable (default `n_splits=5`/`test_size=0.2`). (b) exact group SELECTION is an RNG carve-out (`SmallRng` vs numpy permutation) — R-DEFER-3 blocker, NO failing test. Blockers #1753 (RNG), #1754 (defaults). |
//! | REQ-X-1 (R-SUBSTRATE) | NOT-STARTED | production uses `ndarray::Array1` + `rand`/`SmallRng`; destination is `ferray-core`/`ferray::random` (R-SUBSTRATE-1). Blocker #1754. |
//! | REQ-X-2 (non-test production consumer) | SHIPPED | re-exported `pub use group_splitters::{…}` in `lib.rs` (boundary API, S5/R-DEFER-1). None implements `CrossValidator` (no group/label channel), so the re-export is the sole production reach. |

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::cross_validation::FoldSplits;
use ferrolearn_core::FerroError;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

fn unique_groups(groups: &Array1<usize>) -> Vec<usize> {
    let mut g: Vec<usize> = groups.iter().copied().collect();
    g.sort_unstable();
    g.dedup();
    g
}

/// Population standard deviation (ddof=0) of a slice, matching `np.std`.
fn population_std(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// `np.isclose(a, b)` with the default `rtol=1e-5, atol=1e-8`.
fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

fn check_non_empty(groups: &Array1<usize>, context: &str) -> Result<(), FerroError> {
    if groups.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    Ok(())
}

/// Partition unique groups into `n_splits` folds.
#[derive(Debug, Clone)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Construct a new [`GroupKFold`] with the given number of folds.
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate the splits for the given group labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2` or if
    /// `n_splits > n_unique_groups`.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "GroupKFold")?;
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        let unique = unique_groups(groups);
        if unique.len() < self.n_splits {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!(
                    "GroupKFold needs n_splits ({}) <= unique groups ({})",
                    self.n_splits,
                    unique.len()
                ),
            });
        }

        // Reproduce sklearn's GroupKFold._iter_test_indices exactly
        // (`sklearn/model_selection/_split.py:614-636`): weight each group by its
        // sample count, then distribute the most frequent groups first into the
        // lightest fold. The ordering must be DETERMINISTIC — `unique` is sorted
        // ascending, and sklearn's `np.argsort(counts)[::-1]` (stable argsort +
        // reversal) visits equal-count groups in DESCENDING group-id order, so we
        // sort by `(count desc, group_id desc)` over a deterministic Vec (never a
        // HashMap, whose iteration order is non-deterministic).
        let mut count_of: BTreeMap<usize, usize> = BTreeMap::new();
        for &g in groups.iter() {
            *count_of.entry(g).or_insert(0) += 1;
        }
        let mut ordered: Vec<(usize, usize)> = unique.iter().map(|&g| (g, count_of[&g])).collect();
        // Descending count, ties broken by descending group id.
        ordered.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));

        let mut fold_size = vec![0usize; self.n_splits];
        let mut group_to_fold: HashMap<usize, usize> = HashMap::new();
        for (group, count) in ordered {
            // pick fold with smallest current size (first/lowest index on ties)
            let mut min_idx = 0usize;
            let mut min_val = fold_size[0];
            for (i, &v) in fold_size.iter().enumerate().skip(1) {
                if v < min_val {
                    min_val = v;
                    min_idx = i;
                }
            }
            group_to_fold.insert(group, min_idx);
            fold_size[min_idx] += count;
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        for (i, &g) in groups.iter().enumerate() {
            let fold_idx = *group_to_fold.get(&g).unwrap();
            for (k, (train, test)) in folds.iter_mut().enumerate() {
                if k == fold_idx {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
        }
        Ok(folds)
    }
}

/// Random group-wise train/test splits.
#[derive(Debug, Clone)]
pub struct GroupShuffleSplit {
    n_splits: usize,
    test_size: f64,
    random_state: Option<u64>,
}

impl GroupShuffleSplit {
    /// Construct a new [`GroupShuffleSplit`].
    #[must_use]
    pub fn new(n_splits: usize, test_size: f64) -> Self {
        Self {
            n_splits,
            test_size,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "GroupShuffleSplit")?;
        if !(0.0 < self.test_size && self.test_size < 1.0) {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!(
                    "GroupShuffleSplit: test_size must be in (0, 1), got {}",
                    self.test_size
                ),
            });
        }
        let unique = unique_groups(groups);
        let n_groups = unique.len();
        if n_groups < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_groups".into(),
                reason: "GroupShuffleSplit needs at least 2 distinct groups".into(),
            });
        }
        // Mirror sklearn `_validate_shuffle_split` (`_split.py:2389-2390`),
        // applied to the NUMBER OF GROUPS: `n_test = ceil(test_size * n_groups)`.
        // (round() would give round(2.1)=2 for 7 groups @ 0.3; ceil gives 3.)
        let n_test = ((n_groups as f64) * self.test_size).ceil().max(1.0) as usize;
        // sklearn raises ValueError on an empty or full test split; clamp keeps
        // 0 < n_test < n_groups so train = n_groups - n_test stays non-empty.
        let n_test = n_test.min(n_groups - 1);

        let mut folds = Vec::with_capacity(self.n_splits);
        for split in 0..self.n_splits {
            let mut rng = match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(split as u64)),
                None => SmallRng::from_os_rng(),
            };
            let mut shuffled = unique.clone();
            shuffled.shuffle(&mut rng);
            let test_groups: HashSet<usize> = shuffled[..n_test].iter().copied().collect();
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if test_groups.contains(&g) {
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

/// One fold per unique group: the test set for fold `i` is exactly the
/// samples in the `i`-th group.
#[derive(Debug, Clone, Default)]
pub struct LeaveOneGroupOut;

impl LeaveOneGroupOut {
    /// Construct a new [`LeaveOneGroupOut`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "LeaveOneGroupOut")?;
        let unique = unique_groups(groups);
        if unique.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_groups".into(),
                reason: "LeaveOneGroupOut needs at least 2 distinct groups".into(),
            });
        }
        let mut folds = Vec::with_capacity(unique.len());
        for &target in &unique {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if g == target {
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

/// One fold per `p`-sized subset of groups (`C(n_groups, p)` folds total).
#[derive(Debug, Clone)]
pub struct LeavePGroupsOut {
    p: usize,
}

impl LeavePGroupsOut {
    /// Construct a new [`LeavePGroupsOut`] with the given `p`.
    #[must_use]
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "LeavePGroupsOut")?;
        if self.p == 0 {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: "LeavePGroupsOut: p must be >= 1".into(),
            });
        }
        let unique = unique_groups(groups);
        if unique.len() <= self.p {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: format!(
                    "LeavePGroupsOut needs n_unique_groups ({}) > p ({})",
                    unique.len(),
                    self.p
                ),
            });
        }
        let mut folds = Vec::new();
        let n_g = unique.len();
        let mut combo: Vec<usize> = (0..self.p).collect();
        loop {
            let test_set: HashSet<usize> = combo.iter().map(|&k| unique[k]).collect();
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if test_set.contains(&g) {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push((train, test));

            // Advance combo lexicographically.
            let mut i = self.p;
            while i > 0 {
                i -= 1;
                if combo[i] < n_g - self.p + i {
                    combo[i] += 1;
                    for j in (i + 1)..self.p {
                        combo[j] = combo[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return Ok(folds);
                }
            }
        }
    }
}

/// Group-aware k-fold that also tries to preserve class balance per fold.
#[derive(Debug, Clone)]
pub struct StratifiedGroupKFold {
    n_splits: usize,
}

impl StratifiedGroupKFold {
    /// Construct a new [`StratifiedGroupKFold`].
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate the splits for the given class labels and group labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `y` and `groups` have
    /// different lengths.
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2` or
    /// `n_splits > n_unique_groups`.
    pub fn split(
        &self,
        y: &Array1<usize>,
        groups: &Array1<usize>,
    ) -> Result<FoldSplits, FerroError> {
        if y.len() != groups.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.len()],
                actual: vec![groups.len()],
                context: "StratifiedGroupKFold: y and groups must have the same length".into(),
            });
        }
        check_non_empty(groups, "StratifiedGroupKFold")?;
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        let unique_g = unique_groups(groups);
        if unique_g.len() < self.n_splits {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!(
                    "StratifiedGroupKFold needs n_splits ({}) <= unique groups ({})",
                    self.n_splits,
                    unique_g.len()
                ),
            });
        }

        // Compute per-group class counts and total class counts.
        let unique_y = {
            let mut v: Vec<usize> = y.iter().copied().collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let n_classes = unique_y.len();
        let class_idx: HashMap<usize, usize> =
            unique_y.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut group_counts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (&g, &c) in groups.iter().zip(y.iter()) {
            let entry = group_counts
                .entry(g)
                .or_insert_with(|| vec![0usize; n_classes]);
            entry[class_idx[&c]] += 1;
        }

        // Total count per class
        let mut total_per_class = vec![0usize; n_classes];
        for counts in group_counts.values() {
            for (i, &v) in counts.iter().enumerate() {
                total_per_class[i] += v;
            }
        }

        // Port scikit-learn's StratifiedGroupKFold._iter_test_indices greedy
        // (`sklearn/model_selection/_split.py:1015-1059`, v1.5.2). `group_counts`
        // is keyed by ASCENDING unique group id, so collecting it yields rows in
        // the same order as numpy's `np.unique(groups)` ascending-sorted unique
        // list; `group_to_fold` therefore maps raw group id -> fold.
        let y_cnt: Vec<f64> = total_per_class.iter().map(|&t| t as f64).collect();
        let ordered: Vec<(usize, Vec<usize>)> = group_counts.into_iter().collect();

        // Order groups by DESCENDING population std (ddof=0) of their class-count
        // row, ties broken by ASCENDING group id. `ordered` is already ascending
        // by group id and `sort_by` is stable, so sorting on the descending-std
        // key reproduces numpy's `argsort(-std(...), kind="mergesort")`.
        let mut sort_idx: Vec<usize> = (0..ordered.len()).collect();
        let group_std: Vec<f64> = ordered
            .iter()
            .map(|(_, counts)| {
                population_std(&counts.iter().map(|&c| c as f64).collect::<Vec<_>>())
            })
            .collect();
        sort_idx.sort_by(|&a, &b| {
            group_std[b]
                .partial_cmp(&group_std[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut fold_class_counts = vec![vec![0.0_f64; n_classes]; self.n_splits];
        let mut group_to_fold: HashMap<usize, usize> = HashMap::new();

        for &idx in &sort_idx {
            let (group, counts) = &ordered[idx];
            let group_y_counts: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
            let best_fold =
                Self::find_best_fold(&fold_class_counts, &y_cnt, &group_y_counts, self.n_splits);
            for c in 0..n_classes {
                fold_class_counts[best_fold][c] += group_y_counts[c];
            }
            group_to_fold.insert(*group, best_fold);
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        for (i, &g) in groups.iter().enumerate() {
            // Infallible: every unique group id received a fold above; the
            // fallible accessor avoids a bare `unwrap` per R-CODE-2.
            let fold_idx = *group_to_fold
                .get(&g)
                .ok_or_else(|| FerroError::InvalidParameter {
                    name: "groups".into(),
                    reason: "StratifiedGroupKFold: group missing from fold assignment".into(),
                })?;
            for (k, (train, test)) in folds.iter_mut().enumerate() {
                if k == fold_idx {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
        }
        Ok(folds)
    }

    /// Port of scikit-learn's `StratifiedGroupKFold._find_best_fold`
    /// (`sklearn/model_selection/_split.py:1039-1059`, v1.5.2). Returns the fold
    /// index that, after tentatively adding `group_y_counts`, minimises the mean
    /// per-class population std of `y_counts_per_fold / y_cnt`, tie-broken
    /// (`np.isclose`) by the fewest samples currently in the fold.
    fn find_best_fold(
        y_counts_per_fold: &[Vec<f64>],
        y_cnt: &[f64],
        group_y_counts: &[f64],
        n_splits: usize,
    ) -> usize {
        let n_classes = y_cnt.len();
        let mut best_fold = 0usize;
        let mut min_eval = f64::INFINITY;
        let mut min_samples_in_fold = f64::INFINITY;
        for i in 0..n_splits {
            // For each class, population std (axis=0, over folds) of the
            // tentatively-updated y_counts_per_fold[.][c] / y_cnt[c].
            let mut std_sum = 0.0_f64;
            for c in 0..n_classes {
                let col: Vec<f64> = (0..n_splits)
                    .map(|f| {
                        let mut v = y_counts_per_fold[f][c];
                        if f == i {
                            v += group_y_counts[c];
                        }
                        v / y_cnt[c]
                    })
                    .collect();
                std_sum += population_std(&col);
            }
            let fold_eval = std_sum / n_classes as f64;
            let samples_in_fold: f64 = y_counts_per_fold[i].iter().sum();
            // Replicate sklearn precedence: (a<b) or (isclose and c<d).
            let is_better = fold_eval < min_eval
                || (is_close(fold_eval, min_eval) && samples_in_fold < min_samples_in_fold);
            if is_better {
                min_eval = fold_eval;
                min_samples_in_fold = samples_in_fold;
                best_fold = i;
            }
        }
        best_fold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn group_kfold_partitions_groups() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let folds = GroupKFold::new(2).split(&groups).unwrap();
        assert_eq!(folds.len(), 2);
        // Each fold's test indices should belong to a disjoint set of groups
        // from its train indices.
        for (train, test) in &folds {
            let test_groups: HashSet<usize> = test.iter().map(|&i| groups[i]).collect();
            let train_groups: HashSet<usize> = train.iter().map(|&i| groups[i]).collect();
            assert!(test_groups.is_disjoint(&train_groups));
        }
    }

    #[test]
    fn group_shuffle_split_deterministic() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let a = GroupShuffleSplit::new(2, 0.5)
            .random_state(7)
            .split(&groups)
            .unwrap();
        let b = GroupShuffleSplit::new(2, 0.5)
            .random_state(7)
            .split(&groups)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn leave_one_group_out_one_fold_per_group() {
        let groups = array![0usize, 0, 1, 1, 2];
        let folds = LeaveOneGroupOut::new().split(&groups).unwrap();
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn leave_p_groups_out_combinations() {
        let groups = array![0usize, 1, 2, 3];
        let folds = LeavePGroupsOut::new(2).split(&groups).unwrap();
        // C(4, 2) = 6
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn stratified_group_kfold_balances() {
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let folds = StratifiedGroupKFold::new(2).split(&y, &groups).unwrap();
        assert_eq!(folds.len(), 2);
        for (train, test) in &folds {
            let test_groups: HashSet<usize> = test.iter().map(|&i| groups[i]).collect();
            let train_groups: HashSet<usize> = train.iter().map(|&i| groups[i]).collect();
            assert!(test_groups.is_disjoint(&train_groups));
        }
    }

    #[test]
    fn stratified_group_kfold_shape_mismatch() {
        let y = array![0usize, 1];
        let groups = array![0usize];
        assert!(StratifiedGroupKFold::new(2).split(&y, &groups).is_err());
    }
}
