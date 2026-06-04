//! Divergence + parity audit for `ferrolearn-model-sel/src/group_splitters.rs`
//! against scikit-learn 1.5.2 (`sklearn/model_selection/_split.py`).
//!
//! All expected values are computed from a LIVE sklearn 1.5.2 oracle run from
//! `/tmp` (R-CHAR-3) — never copied from the ferrolearn side. The oracle command
//! used for each test is reproduced in its doc comment.
//!
//! - GREEN GUARDS (deterministic SHIPPED REQs): un-ignored, must PASS now —
//!   `logo_split_index_parity` (REQ-LOGO-1), `lpgo_combination_order_parity`
//!   (REQ-LPGO-1).
//! - FAILING TESTS (real divergences): `#[ignore]` with tracking issue —
//!   `gkf_greedy_ordering_diverges` (REQ-GKF-2),
//!   `gss_test_group_count_round_vs_ceil` (REQ-GSS-2),
//!   `sgkf_objective_diverges` (REQ-SGKF-1).

use std::collections::HashSet;

use ferrolearn_model_sel::{
    GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold,
};
use ndarray::array;

/// Return the sorted-unique set of group ids appearing at the given sample
/// indices.
fn group_set(indices: &[usize], groups: &[usize]) -> Vec<usize> {
    let mut s: Vec<usize> = indices
        .iter()
        .map(|&i| groups[i])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    s.sort_unstable();
    s
}

// ---------------------------------------------------------------------------
// GREEN GUARD 1 — REQ-LOGO-1: LeaveOneGroupOut split-index parity.
// ---------------------------------------------------------------------------
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   import numpy as np
//   from sklearn.model_selection import LeaveOneGroupOut
//   X=np.zeros((6,1)); g=np.array([0,0,1,1,2,2])
//   for tr,te in LeaveOneGroupOut().split(X,groups=g): print(list(tr),list(te))
//   # ([2,3,4,5],[0,1]) ([0,1,4,5],[2,3]) ([0,1,2,3],[4,5])
//
// Mirrors `sklearn/model_selection/_split.py:1330-1337`
//   `for i in unique_groups: yield groups == i`  (ascending unique order).
#[test]
fn logo_split_index_parity() {
    let groups = array![0usize, 0, 1, 1, 2, 2];
    let folds = LeaveOneGroupOut::new().split(&groups).unwrap();

    let expected: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![2, 3, 4, 5], vec![0, 1]),
        (vec![0, 1, 4, 5], vec![2, 3]),
        (vec![0, 1, 2, 3], vec![4, 5]),
    ];
    assert_eq!(folds, expected, "LeaveOneGroupOut index parity vs sklearn");
}

// ---------------------------------------------------------------------------
// GREEN GUARD 2 — REQ-LPGO-1: LeavePGroupsOut combination-order index parity.
// ---------------------------------------------------------------------------
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   import numpy as np
//   from sklearn.model_selection import LeavePGroupsOut
//   X=np.zeros((6,1)); g=np.array([0,0,1,1,2,2])
//   for tr,te in LeavePGroupsOut(2).split(X,groups=g): print(list(tr),list(te))
//   # ([4,5],[0,1,2,3]) ([2,3],[0,1,4,5]) ([0,1],[2,3,4,5])
//
// C(3,2)=3 folds in itertools.combinations LEXICOGRAPHIC order over the sorted
// unique-group indices ({0,1},{0,2},{1,2}). Mirrors
// `sklearn/model_selection/_split.py:1465-1470`.
#[test]
fn lpgo_combination_order_parity() {
    let groups = array![0usize, 0, 1, 1, 2, 2];
    let folds = LeavePGroupsOut::new(2).split(&groups).unwrap();

    let expected: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![4, 5], vec![0, 1, 2, 3]),
        (vec![2, 3], vec![0, 1, 4, 5]),
        (vec![0, 1], vec![2, 3, 4, 5]),
    ];
    assert_eq!(
        folds, expected,
        "LeavePGroupsOut combination-order index parity vs sklearn"
    );
}

// ---------------------------------------------------------------------------
// FAILING TEST — REQ-GKF-2: GroupKFold greedy ordering / tie-break divergence.
// ---------------------------------------------------------------------------
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   import numpy as np
//   from sklearn.model_selection import GroupKFold
//   g=np.array([0,0,1,1,2,2,3,3]); X=np.zeros((8,1))
//   for i,(tr,te) in enumerate(GroupKFold(3).split(X,groups=g)):
//       print(i, sorted(set(g[te])))
//   # 0 [0,3] | 1 [2] | 2 [1]
//
// sklearn processes groups in `np.argsort(bincount)[::-1]` order (descending
// size, ties by descending group index) then assigns each to `np.argmin` load
// (`sklearn/model_selection/_split.py:618,628-636`):
//   fold0 test groups {0,3}, fold1 {2}, fold2 {1}.
//
// ferrolearn `GroupKFold::split` (group_splitters.rs:82-103) collects per-group
// sizes from a `HashMap<usize,usize>` into a `Vec` then `sort_by_key`. The
// HashMap iteration order feeding the (stable) sort is NON-DETERMINISTIC: across
// repeated runs fold0's test groups have been observed as {1,2}, {0,3}, {1,3},
// {2,3}. It does not reproduce sklearn's descending-group-index tie-break
// membership and is not even stable run to run. We assert against the single
// deterministic sklearn answer.
//
// Tracking: #1750
#[test]
fn gkf_greedy_ordering_diverges() {
    let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
    let g: Vec<usize> = groups.to_vec();
    let folds = GroupKFold::new(3).split(&groups).unwrap();

    // sklearn fold -> test groups (deterministic, from live oracle):
    let expected_test_groups: Vec<Vec<usize>> = vec![vec![0, 3], vec![2], vec![1]];

    let actual: Vec<Vec<usize>> = folds.iter().map(|(_tr, te)| group_set(te, &g)).collect();

    assert_eq!(
        actual, expected_test_groups,
        "GroupKFold fold->test-group membership must match sklearn's deterministic argsort[::-1] greedy"
    );
}

// ---------------------------------------------------------------------------
// FAILING TEST — REQ-GSS-2: GroupShuffleSplit n_test-group count round vs ceil.
// ---------------------------------------------------------------------------
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   from math import ceil
//   import numpy as np
//   from sklearn.model_selection import GroupShuffleSplit
//   g=np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6])   # 7 unique groups
//   print(ceil(0.3*7))                           # 3
//   for tr,te in GroupShuffleSplit(1,test_size=0.3,random_state=0)\
//           .split(np.zeros((14,1)),groups=g):
//       print(len(set(g[te])))                   # 3
//
// sklearn `_validate_shuffle_split` computes `n_test = ceil(test_size *
// n_groups)` (`sklearn/model_selection/_split.py:2389-2390`) => ceil(2.1)=3.
// ferrolearn (group_splitters.rs:168) computes
// `((n_groups as f64) * test_size).round()` => round(2.1)=2.
//
// The exact group SELECTION (membership) is an RNG carve-out (REQ-GSS-3); only
// the COUNT of distinct test groups is deterministic and pinned here.
//
// Tracking: #1751
#[test]
#[ignore = "divergence: GroupShuffleSplit n_test uses round() not ceil(); tracking #1751"]
fn gss_test_group_count_round_vs_ceil() {
    let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
    let g: Vec<usize> = groups.to_vec();
    let folds = GroupShuffleSplit::new(1, 0.3)
        .random_state(0)
        .split(&groups)
        .unwrap();

    // sklearn ceil(0.3 * 7) = 3 distinct test groups.
    let sklearn_n_test_groups = 3usize;

    let (_tr, te) = &folds[0];
    let actual = group_set(te, &g).len();

    assert_eq!(
        actual, sklearn_n_test_groups,
        "GroupShuffleSplit must select ceil(test_size*n_groups) test groups, not round()"
    );
}

// ---------------------------------------------------------------------------
// FAILING TEST — REQ-SGKF-1: StratifiedGroupKFold objective divergence.
// ---------------------------------------------------------------------------
//
// Live oracle (sklearn 1.5.2, run from /tmp):
//   import numpy as np
//   from sklearn.model_selection import StratifiedGroupKFold
//   X=np.ones((17,2))
//   y=np.array([0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
//   g=np.array([1,1,2,2,3,3,3,4,5,5,5,5,6,6,7,8,8])
//   for i,(tr,te) in enumerate(StratifiedGroupKFold(3).split(X,y,g)):
//       print(i, sorted(set(g[te])))
//   # 0 [3,6,7] | 1 [1,2,8] | 2 [4,5]
//
// sklearn `_find_best_fold` sorts groups by descending per-group STD of class
// counts (stable mergesort, `:1015-1019`) and minimises the mean per-class STD
// of `y_counts_per_fold / y_cnt`, tie-broken by fewest samples (`:1039-1059`).
// ferrolearn (group_splitters.rs:390-416) sorts by descending group SIZE and
// minimises a uniform-target sum-of-squared-deviation objective. The result
// diverges: ferrolearn was observed producing fold0 groups [1,3,5], fold1
// [2,4,6,7,8], fold2 [] (an EMPTY fold) on this input.
//
// Tracking: #1752
#[test]
#[ignore = "divergence: StratifiedGroupKFold size-sort/SSE objective != sklearn std-of-class-dist + _find_best_fold; tracking #1752"]
fn sgkf_objective_diverges() {
    let y = array![0usize, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let groups = array![1usize, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8];
    let g: Vec<usize> = groups.to_vec();
    let folds = StratifiedGroupKFold::new(3).split(&y, &groups).unwrap();

    // sklearn fold -> test groups (deterministic, from live oracle):
    let expected_test_groups: Vec<Vec<usize>> = vec![vec![3, 6, 7], vec![1, 2, 8], vec![4, 5]];

    let actual: Vec<Vec<usize>> = folds.iter().map(|(_tr, te)| group_set(te, &g)).collect();

    assert_eq!(
        actual, expected_test_groups,
        "StratifiedGroupKFold fold->test-group membership must match sklearn's _find_best_fold greedy"
    );
}
