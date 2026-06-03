//! Divergence pin: `d2_absolute_error_score` is mis-framed in the `scorer.rs`
//! REQ table as a classification scorer "blocked on #781" (the heterogeneous
//! `Scorer` type), but it is a REGRESSION metric whose signature
//! `fn(&Array1<F>, &Array1<F>) -> Result<F, FerroError>` exactly fits the
//! CURRENT `Scorer<F>` function-pointer type. It is therefore registerable in
//! `get_scorer`/`get_scorer_names` today — NOT blocked on #781.
//!
//! All expected values come from the LIVE sklearn 1.5.2 oracle (R-CHAR-3),
//! never copied from the ferrolearn side.

use ferrolearn_metrics::regression::d2_absolute_error_score;
use ferrolearn_metrics::scorer::{Scorer, get_scorer, get_scorer_names, make_scorer};

/// Divergence: `d2_absolute_error_score` is a real sklearn 1.5.2 scorer name
/// with `_sign == +1`, and its metric signature fits the existing `Scorer<F>`
/// type, yet ferrolearn neither lists nor resolves it. The `scorer.rs` REQ
/// table places it under REQ-5 "Classification scorers registry ... blocked on
/// heterogeneous Scorer type (#781)", which is a mis-framing: a regression
/// `fn(&Array1<F>,&Array1<F>)` metric needs NO new Scorer type.
///
/// Oracle (sklearn 1.5.2, run from /tmp):
///   python3 -c "from sklearn.metrics import get_scorer_names; \
///     print('d2_absolute_error_score' in get_scorer_names())"   # True
///   python3 -c "from sklearn.metrics import get_scorer; \
///     print(get_scorer('d2_absolute_error_score')._sign)"       # 1
///
/// ferrolearn: `get_scorer_names()` omits it and `get_scorer` rejects it.
/// Tracking: #787
#[test]
fn divergence_d2_absolute_error_score_is_registerable_now() {
    // The metric already has the exact Scorer<F> shape — provable by binding it
    // into a Scorer via the existing `make_scorer` (sklearn _sign == +1 ->
    // greater_is_better == true). This compiling proves it is NOT blocked on the
    // heterogeneous Scorer type (#781).
    let built: Scorer<f64> = make_scorer(d2_absolute_error_score, true, "d2_absolute_error_score");
    assert!(built.greater_is_better);

    // sklearn 1.5.2 live oracle: the name IS in get_scorer_names().
    let names = get_scorer_names();
    assert!(
        names.contains(&"d2_absolute_error_score"),
        "d2_absolute_error_score: sklearn registers it (regression scorer, fits the \
         current Scorer<F> type, NOT blocked on #781); ferrolearn get_scorer_names()={names:?}"
    );

    // sklearn resolves it with _sign == +1 -> greater_is_better == true.
    let scorer = get_scorer("d2_absolute_error_score").expect(
        "get_scorer('d2_absolute_error_score'): sklearn registers it with _sign==+1; \
         ferrolearn rejected it",
    );
    assert!(
        scorer.greater_is_better,
        "d2_absolute_error_score: sklearn _sign == +1 (greater_is_better=true); ferrolearn={}",
        scorer.greater_is_better
    );
}
