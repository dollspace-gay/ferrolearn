//! Determinism + balance guard for GroupKFold equal-count tie-break.
//!
//! Carve-out #1755 (R-DEFER-3 identifiability carve-out). The earlier #1755
//! draft pinned ferrolearn's GroupKFold tie-break among EQUAL-COUNT groups
//! against numpy's exact `np.argsort(np.bincount(groups))[::-1]` order. After
//! review that is NOT a meaningful parity target:
//!
//! scikit-learn `GroupKFold._iter_test_indices`
//! (`sklearn/model_selection/_split.py:617-618`) breaks ties among
//! equal-count groups SOLELY via `np.argsort`'s default UNSTABLE QUICKSORT
//! (introsort). That tie order is a numpy-implementation artifact and is
//! numpy-VERSION-dependent — sklearn's own output for tied groups changes
//! across numpy releases — so there is no stable sklearn-level specification
//! for it. The documented, semantically meaningful contract
//! ("distribute the most frequent groups first" — `_split.py:617`) IS
//! satisfied by ferrolearn's deterministic `(count desc, group_id desc)`
//! ordering, which also yields equally-balanced folds.
//!
//! Matching numpy's introsort internals is therefore not a parity contract.
//! This guard pins the values that ARE the contract: DETERMINISM (no
//! reintroduced HashMap non-determinism) and BALANCE/structural correctness
//! (full disjoint partition, no group split, greedy-argmin balance), NOT a
//! specific tie order.
//!
//! Tracking: #1755 (documented carve-out; stays open).

use ferrolearn_model_sel::GroupKFold;
use ndarray::{Array1, array};
use std::collections::HashSet;

#[test]
fn gkf_tiebreak_deterministic_and_balanced() {
    // carve-out #1755 — sklearn's equal-count tie-break is numpy's unstable
    // quicksort (`_split.py:618`), a numpy-version-defined implementation
    // artifact with no stable sklearn contract; ferrolearn uses a
    // deterministic `(count desc, group_id desc)` order; this guard pins
    // determinism + balance, not numpy's sort internals.

    // bincount = [5,5,1,1,5] over 17 samples: ties among {0,1,4} (count 5)
    // and {2,3} (count 1) — the exact input that triggered the divergence.
    let groups: Array1<usize> = array![0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4];
    let n_samples = groups.len();
    assert_eq!(n_samples, 17, "fixture must have 17 samples");

    let n_splits = 3;

    // Determinism: two independent splits of the same input must be
    // byte-for-byte identical (guards against reintroduced HashMap-iteration
    // non-determinism). FoldSplits is Vec<(train, test)>.
    let folds_a = GroupKFold::new(n_splits).split(&groups).unwrap();
    let folds_b = GroupKFold::new(n_splits).split(&groups).unwrap();
    assert_eq!(
        folds_a, folds_b,
        "GroupKFold must be deterministic across identical runs"
    );

    assert_eq!(folds_a.len(), n_splits, "must produce n_splits folds");

    // Structural correctness, asserted on the test index lists.
    let g = groups.to_vec();
    let mut all_test: Vec<usize> = Vec::new();
    let mut fold_sizes: Vec<usize> = Vec::new();

    for (_train, test) in &folds_a {
        // Every fold is non-empty.
        assert!(!test.is_empty(), "every fold must be non-empty");
        fold_sizes.push(test.len());
        all_test.extend_from_slice(test);
    }

    // Complete + disjoint partition of all 17 sample indices.
    let mut sorted_all = all_test.clone();
    sorted_all.sort_unstable();
    let expected_indices: Vec<usize> = (0..n_samples).collect();
    assert_eq!(
        sorted_all, expected_indices,
        "test folds must partition all sample indices completely and disjointly"
    );
    let unique: HashSet<usize> = all_test.iter().copied().collect();
    assert_eq!(
        unique.len(),
        n_samples,
        "test folds must be pairwise disjoint (no index appears twice)"
    );

    // No group is split across folds: map each group to its owning fold and
    // verify every sample of that group lands in the same fold.
    let mut group_fold: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (fold_idx, (_train, test)) in folds_a.iter().enumerate() {
        for &i in test {
            let gid = g[i];
            let owner = group_fold.entry(gid).or_insert(fold_idx);
            assert_eq!(
                *owner, fold_idx,
                "group {gid} must lie wholly within a single fold"
            );
        }
    }

    // Balance: the greedy-argmin invariant. With max single-group weight = 5
    // (the largest group count), fold sizes must differ by no more than that
    // bound; here the perfect split is 6/6/5.
    let max_group_weight = 5usize;
    let min_size = *fold_sizes.iter().min().unwrap();
    let max_size = *fold_sizes.iter().max().unwrap();
    assert!(
        max_size - min_size <= max_group_weight,
        "fold sizes {fold_sizes:?} must be balanced within the max group weight"
    );
    assert_eq!(
        fold_sizes.iter().sum::<usize>(),
        n_samples,
        "fold sizes must sum to the sample count"
    );
}
