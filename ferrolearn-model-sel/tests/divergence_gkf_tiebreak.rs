//! NEW divergence pinned during the post-fix re-audit of
//! `ferrolearn-model-sel/src/group_splitters.rs` (commit 90556ee5, the #1750
//! GroupKFold "deterministic greedy" fix).
//!
//! The #1750 fix made GroupKFold deterministic, but the deterministic
//! tie-break it chose — `(count desc, group_id DESC)` — does NOT reproduce
//! scikit-learn's `np.argsort(np.bincount(groups))[::-1]`
//! (`sklearn/model_selection/_split.py:618`). `np.argsort` defaults to an
//! UNSTABLE quicksort, so for non-trivially tied counts the order of tied
//! groups is neither "descending id" nor any simple key. This changes the
//! observable fold->test-group membership.
//!
//! Live sklearn 1.5.2 oracle, run from /tmp (R-CHAR-3, NOT copied from
//! ferrolearn):
//!   import numpy as np
//!   from sklearn.model_selection import GroupKFold
//!   parts=[]
//!   for gid,c in enumerate([5,5,1,1,5]): parts += [gid]*c
//!   g=np.array(parts); X=np.zeros((len(g),1))
//!   for i,(tr,te) in enumerate(GroupKFold(3).split(X,groups=g)):
//!       print(i, sorted(set(g[te])))
//!   # groups = [0,0,0,0,0, 1,1,1,1,1, 2, 3, 4,4,4,4,4]
//!   # 0 [2,4] | 1 [0,3] | 2 [1]
//!
//! numpy `argsort(bincount=[5,5,1,1,5])[::-1] = [4,0,1,2,3]`, NOT the
//! ferrolearn `(count desc, id desc)` order `[4,1,0,3,2]`. Hand-tracing the
//! ferrolearn argmin greedy on its order yields fold0 [3,4], fold1 [1,2],
//! fold2 [0] — diverging from sklearn's fold0 [2,4], fold1 [0,3], fold2 [1].
//!
//! Tracking: #1755  (NEW divergence, distinct from #1750/#1753/#1754).
#[ignore = "divergence: GroupKFold tie-break (count desc, id desc) != np.argsort(bincount)[::-1] unstable quicksort; tracking #1755"]
#[test]
fn gkf_tiebreak_argsort_quicksort_diverges() {
    use ferrolearn_model_sel::GroupKFold;
    use ndarray::array;
    use std::collections::HashSet;

    // bincount = [5,5,1,1,5]: ties among {0,1,4} (count 5) and {2,3} (count 1).
    let groups = array![
        0usize, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4
    ];
    let g: Vec<usize> = groups.to_vec();
    let folds = GroupKFold::new(3).split(&groups).unwrap();

    let test_groups = |indices: &[usize]| -> Vec<usize> {
        let mut s: Vec<usize> = indices
            .iter()
            .map(|&i| g[i])
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        s.sort_unstable();
        s
    };

    // sklearn 1.5.2 deterministic oracle (live, from /tmp):
    let expected: Vec<Vec<usize>> = vec![vec![2, 4], vec![0, 3], vec![1]];

    let actual: Vec<Vec<usize>> = folds.iter().map(|(_tr, te)| test_groups(te)).collect();

    assert_eq!(
        actual, expected,
        "GroupKFold tie-break must reproduce np.argsort(bincount)[::-1], not (count desc, id desc)"
    );
}
