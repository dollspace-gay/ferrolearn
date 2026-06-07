//! Bit-exact port of libstdc++'s `std::nth_element` (introselect).
//!
//! sklearn's KDTree partitions each node's index slice with C++
//! `std::nth_element` (`sklearn/neighbors/_partition_nodes.pyx`), whose exact
//! element layout — not just the nth-element invariant — determines the leaf
//! push order, and therefore which points win an exact-distance tie in a
//! `KDTree.query`. To reproduce sklearn's `kd_tree` k-NN SET bit-for-bit we
//! must reproduce libstdc++'s introselect bit-for-bit, including its
//! median-of-three pivot selection and Hoare partition.
//!
//! This is a transcription of GCC libstdc++ `<bits/stl_algo.h>`
//! (`std::nth_element` / `__introselect` / `__unguarded_partition_pivot` /
//! `__move_median_to_first` / `__unguarded_partition` / `__heap_select` /
//! `__insertion_sort`). It operates on a `&mut [usize]` (the node index slice)
//! using a caller-supplied strict-weak-ordering comparator `cmp(a, b) -> bool`
//! meaning "a sorts before b" (i.e. C++ `comp(*a, *b)` = "*a < *b").
//!
//! The comparator is over the *values* stored in the slice (here, training-row
//! indices), matching the C++ `IndexComparator::operator()(const I& a, const
//! I& b)` which receives the index values, not iterator positions.
//!
//! No panic path: all index arithmetic stays in-bounds for `nth < slice.len()`,
//! which is the only way the KDTree builder calls it (`split_index = n_mid <
//! n_points`).

/// `__lg(n)` from libstdc++: `floor(log2(n))` for `n >= 1`.
///
/// Upstream computes `sizeof(long)*8 - 1 - __builtin_clzl(n)`; for `n >= 1`
/// this equals `floor(log2(n))`. For `n == 0` libstdc++ never calls `__lg`
/// (the introselect guard `last - first > 3` ensures `n >= 4` at the first
/// call), so we return 0 defensively.
#[inline]
fn lg(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (usize::BITS - 1 - n.leading_zeros()) as usize
    }
}

/// `std::nth_element(first, nth, last, comp)` over a slice.
///
/// After return, `slice[nth]` holds the element that would be there in a fully
/// sorted slice, every element before it compares `<=` it and every element
/// after compares `>=` it — and crucially the surrounding permutation matches
/// libstdc++ exactly.
///
/// `cmp(a, b)` returns `true` iff value `a` sorts strictly before value `b`.
pub fn nth_element<C>(slice: &mut [usize], nth: usize, cmp: &C)
where
    C: Fn(usize, usize) -> bool,
{
    let len = slice.len();
    // C++: if (first == last || nth == last) return;
    if len == 0 || nth == len {
        return;
    }
    let depth_limit = lg(len) * 2;
    introselect(slice, 0, nth, len, depth_limit, cmp);
}

/// `std::__introselect(first, nth, last, depth_limit, comp)`.
///
/// Indices `first`, `nth`, `last` are absolute offsets into `slice`
/// (`first <= nth < last <= slice.len()`), mirroring libstdc++ iterators.
fn introselect<C>(
    slice: &mut [usize],
    mut first: usize,
    nth: usize,
    mut last: usize,
    mut depth_limit: usize,
    cmp: &C,
) where
    C: Fn(usize, usize) -> bool,
{
    while last - first > 3 {
        if depth_limit == 0 {
            // __heap_select(first, nth + 1, last, comp);
            heap_select(slice, first, nth + 1, last, cmp);
            // std::iter_swap(first, nth);
            slice.swap(first, nth);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(slice, first, last, cmp);
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(slice, first, last, cmp);
}

/// `std::__unguarded_partition_pivot(first, last, comp)`.
///
/// Moves the median of `{first, mid, last-1}` to `first`, then partitions
/// `[first+1, last)` around that pivot value (held at position `first`).
fn unguarded_partition_pivot<C>(slice: &mut [usize], first: usize, last: usize, cmp: &C) -> usize
where
    C: Fn(usize, usize) -> bool,
{
    // libstdc++: mid = first + (last - first) / 2;
    let mid = first + (last - first) / 2;
    move_median_to_first(slice, first, first + 1, mid, last - 1, cmp);
    // return __unguarded_partition(first + 1, last, first, comp);
    unguarded_partition(slice, first + 1, last, first, cmp)
}

/// `std::__move_median_to_first(result, a, b, c, comp)`.
///
/// Swaps the median of `*a`, `*b`, `*c` into `*result`. `a`, `b`, `c`, and
/// `result` are absolute slice offsets.
fn move_median_to_first<C>(
    slice: &mut [usize],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    cmp: &C,
) where
    C: Fn(usize, usize) -> bool,
{
    let va = slice[a];
    let vb = slice[b];
    let vc = slice[c];
    if cmp(va, vb) {
        if cmp(vb, vc) {
            slice.swap(result, b);
        } else if cmp(va, vc) {
            slice.swap(result, c);
        } else {
            slice.swap(result, a);
        }
    } else if cmp(va, vc) {
        slice.swap(result, a);
    } else if cmp(vb, vc) {
        slice.swap(result, c);
    } else {
        slice.swap(result, b);
    }
}

/// `std::__unguarded_partition(first, last, pivot, comp)`.
///
/// `pivot` is an absolute slice offset whose *value* is the pivot (libstdc++
/// passes the median slot as an iterator; the value at that slot does not move
/// during this loop, so reading `slice[pivot]` each comparison is faithful).
/// Returns the partition point.
fn unguarded_partition<C>(
    slice: &mut [usize],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    cmp: &C,
) -> usize
where
    C: Fn(usize, usize) -> bool,
{
    loop {
        // while (comp(*first, *pivot)) ++first;
        while cmp(slice[first], slice[pivot]) {
            first += 1;
        }
        // --last;
        last -= 1;
        // while (comp(*pivot, *last)) --last;
        while cmp(slice[pivot], slice[last]) {
            last -= 1;
        }
        // if (!(first < last)) return first;
        if first >= last {
            return first;
        }
        // std::iter_swap(first, last);
        slice.swap(first, last);
        first += 1;
    }
}

/// `std::__heap_select(first, middle, last, comp)`.
///
/// Builds a max-heap on `[first, middle)`, then for each `i` in
/// `[middle, last)` pushes `*i` if it is smaller than the heap top, keeping the
/// `middle - first` smallest values heap-ordered in `[first, middle)`.
fn heap_select<C>(slice: &mut [usize], first: usize, middle: usize, last: usize, cmp: &C)
where
    C: Fn(usize, usize) -> bool,
{
    make_heap(slice, first, middle, cmp);
    for i in middle..last {
        // if (comp(*i, *first)) __pop_heap(first, middle, i, comp);
        if cmp(slice[i], slice[first]) {
            pop_heap_into(slice, first, middle, i, cmp);
        }
    }
}

/// libstdc++ `std::__make_heap(first, last, comp)` (sift-down construction of a
/// max-heap over `[first, last)`).
fn make_heap<C>(slice: &mut [usize], first: usize, last: usize, cmp: &C)
where
    C: Fn(usize, usize) -> bool,
{
    let len = last - first;
    if len < 2 {
        return;
    }
    // parent = (len - 2) / 2; loop down to 0, adjusting each.
    let mut parent = (len - 2) / 2;
    loop {
        let value = slice[first + parent];
        adjust_heap(slice, first, parent, len, value, cmp);
        if parent == 0 {
            break;
        }
        parent -= 1;
    }
}

/// libstdc++ `std::__adjust_heap(first, holeIndex, len, value, comp)`.
///
/// Percolates the hole at `hole_index` down the max-heap rooted at `first`
/// over `len` elements, then drops `value` in and sifts it up
/// (`__push_heap`). All offsets are relative to `first`.
fn adjust_heap<C>(
    slice: &mut [usize],
    first: usize,
    hole_index: usize,
    len: usize,
    value: usize,
    cmp: &C,
) where
    C: Fn(usize, usize) -> bool,
{
    let top_index = hole_index;
    let mut hole = hole_index;
    let mut second_child = hole;
    while second_child < (len - 1) / 2 {
        // secondChild = 2 * (secondChild + 1);
        second_child = 2 * (second_child + 1);
        // if (comp(*(first + secondChild), *(first + (secondChild - 1)))) --secondChild;
        if cmp(slice[first + second_child], slice[first + second_child - 1]) {
            second_child -= 1;
        }
        // *(first + holeIndex) = *(first + secondChild);
        slice[first + hole] = slice[first + second_child];
        hole = second_child;
    }
    // if ((len & 1) == 0 && secondChild == (len - 2) / 2)
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        slice[first + hole] = slice[first + second_child - 1];
        hole = second_child - 1;
    }
    // __push_heap(first, holeIndex, topIndex, value, comp);
    push_heap(slice, first, hole, top_index, value, cmp);
}

/// libstdc++ `std::__push_heap(first, holeIndex, topIndex, value, comp)`.
///
/// Sifts `value` up from `hole_index` toward `top_index`, then writes it.
fn push_heap<C>(
    slice: &mut [usize],
    first: usize,
    hole_index: usize,
    top_index: usize,
    value: usize,
    cmp: &C,
) where
    C: Fn(usize, usize) -> bool,
{
    let mut hole = hole_index;
    let mut parent = if hole == 0 { 0 } else { (hole - 1) / 2 };
    // while (holeIndex > topIndex && comp(*(first + parent), value))
    while hole > top_index && cmp(slice[first + parent], value) {
        slice[first + hole] = slice[first + parent];
        hole = parent;
        parent = if hole == 0 { 0 } else { (hole - 1) / 2 };
    }
    slice[first + hole] = value;
}

/// libstdc++ `std::__pop_heap(first, last, result, comp)` specialised for
/// `__heap_select`: the popped max goes to `result`, and `*result`'s prior
/// value re-enters the heap via `__adjust_heap`.
///
/// `first..middle` is the heap; `result` is the offset whose value should be
/// inserted (the candidate `i` in `heap_select`).
fn pop_heap_into<C>(slice: &mut [usize], first: usize, middle: usize, result: usize, cmp: &C)
where
    C: Fn(usize, usize) -> bool,
{
    // libstdc++ __heap_select calls std::__pop_heap(first, middle, i, comp),
    // i.e.:
    //   value = *result; *result = *first;
    //   __adjust_heap(first, 0, last - first, value, comp);
    let value = slice[result];
    slice[result] = slice[first];
    let len = middle - first;
    adjust_heap(slice, first, 0, len, value, cmp);
}

/// libstdc++ `std::__insertion_sort(first, last, comp)`.
fn insertion_sort<C>(slice: &mut [usize], first: usize, last: usize, cmp: &C)
where
    C: Fn(usize, usize) -> bool,
{
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        // libstdc++ special-cases comp(*i, *first) to do an unguarded rotate;
        // the observable permutation is identical to a plain guarded insertion
        // sort because the comparator is a strict weak ordering. We use the
        // guarded form, which yields the same final layout.
        let value = slice[i];
        let mut j = i;
        while j > first && cmp(value, slice[j - 1]) {
            slice[j] = slice[j - 1];
            j -= 1;
        }
        slice[j] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_sort(base: &[usize], cmp: &impl Fn(usize, usize) -> bool) -> Vec<usize> {
        let mut sorted = base.to_vec();
        sorted.sort_by(|&a, &b| {
            if cmp(a, b) {
                std::cmp::Ordering::Less
            } else if cmp(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        sorted
    }

    /// nth_element must place the correct order statistic at `nth` and leave
    /// the partition invariant, for the index-tiebreak comparator used by the
    /// KDTree. Oracle = a fully sorted copy under the same comparator.
    #[test]
    fn nth_element_matches_full_sort_at_nth() {
        let key: Vec<f64> = vec![3.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let cmp = |a: usize, b: usize| -> bool {
            if key[a] == key[b] {
                a < b
            } else {
                key[a] < key[b]
            }
        };
        for n in 1..=key.len() {
            let base: Vec<usize> = (0..n).collect();
            let sorted = full_sort(&base, &cmp);
            for nth in 0..n {
                let mut v = base.clone();
                nth_element(&mut v, nth, &cmp);
                assert_eq!(
                    v[nth], sorted[nth],
                    "n={n} nth={nth}: wrong order statistic"
                );
                for &x in v.iter().take(nth) {
                    assert!(!cmp(v[nth], x), "left side must be <= nth");
                }
                for &x in v.iter().skip(nth + 1) {
                    assert!(!cmp(x, v[nth]), "right side must be >= nth");
                }
            }
        }
    }

    #[test]
    fn nth_element_large_random_is_correct() {
        // Deterministic LCG so the test is reproducible without rng deps.
        let mut state: u64 = 0x1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };
        let key: Vec<f64> = (0..200).map(|_| (next() % 7) as f64).collect();
        let cmp = |a: usize, b: usize| -> bool {
            if key[a] == key[b] {
                a < b
            } else {
                key[a] < key[b]
            }
        };
        let base: Vec<usize> = (0..key.len()).collect();
        let sorted = full_sort(&base, &cmp);
        for nth in [0, 1, 50, 99, 100, 150, 199] {
            let mut v = base.clone();
            nth_element(&mut v, nth, &cmp);
            assert_eq!(v[nth], sorted[nth], "nth={nth}");
        }
    }
}
