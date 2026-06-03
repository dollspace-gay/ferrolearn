# ferrolearn-bayes crate root (re-export boundary + `log_softmax_rows`)

<!--
tier: 3-component
status: draft
baseline-commit: 3da6b2c1
upstream: scikit-learn 1.5.2 (commit 156ef14)
upstream-paths:
  - sklearn/naive_bayes.py   # __all__ (:30-36) — the re-export boundary; _BaseNB.predict_log_proba (:105-126) jll - logsumexp(jll, axis=1); _BaseNB.predict_proba (:128-144) exp(predict_log_proba). logsumexp = scipy.special.logsumexp (imported :21).
ferrolearn-module: ferrolearn-bayes/src/lib.rs
parity-ops: re-export boundary (naive_bayes.py __all__), log_softmax_rows (_BaseNB.predict_log_proba normalization)
crosslink-issue: 1109
-->

## Summary

`ferrolearn-bayes/src/lib.rs` is the crate's **public-API surface** — the
analog of `sklearn/naive_bayes.py`'s `__all__` (`:30-36`) — plus one
cross-cutting crate-level numeric helper. It is **not** an estimator: it owns no
`fit`/`predict` logic. Two things live here:

1. **The re-export boundary.** `pub mod` declarations for the seven submodules
   (`base`, `bernoulli`, `categorical`, `complement`, `conjugate`, `gaussian`,
   `multinomial`) and a `pub use` block that surfaces the five Naive-Bayes
   variants (`GaussianNB`/`FittedGaussianNB`, `MultinomialNB`/`FittedMultinomialNB`,
   `BernoulliNB`/`FittedBernoulliNB`, `CategoricalNB`/`FittedCategoricalNB`,
   `ComplementNB`/`FittedComplementNB`) plus the `BaseNB` trait at the crate
   root. This mirrors sklearn's `__all__ = [BernoulliNB, GaussianNB,
   MultinomialNB, ComplementNB, CategoricalNB]`. ferrolearn additionally
   re-exports its `BaseNB` trait — the structural analog of sklearn's *private*
   `_BaseNB` (documented in `.design/bayes/base.md`); sklearn keeps `_BaseNB` /
   `_BaseDiscreteNB` private, so the surfaced trait is a ferrolearn extension of
   the boundary, not a divergence from `__all__`.

2. **`log_softmax_rows`** — `pub(crate) fn log_softmax_rows<F: Float>(jll:
   &Array2<F>) -> Array2<F>`, the numerically stable row-wise log-softmax
   `jll − logsumexp(jll, axis=1)`. This is the ferrolearn body of the
   `_BaseNB.predict_log_proba` normalization (`naive_bayes.py:105-126`), where
   sklearn calls `scipy.special.logsumexp` (`:21`, `:125`). It backs
   `predict_log_proba` / `predict_proba` on every fitted NB variant.

The crate doc-comment also carries a `pub(crate) use base::check_alpha as
clamp_alpha;` re-export alias. The `_BaseDiscreteNB._check_alpha` smoothing floor
itself lives in `base.rs` (documented in `.design/bayes/base.md`); `lib.rs` only
re-exports it under the legacy name so the discrete variants' fit call sites are
unchanged. The alias re-export is therefore **not** a REQ of this doc — it is a
rename pass-through with no behavior.

This doc covers the boundary and `log_softmax_rows` only. The per-variant value
parity (each `_joint_log_likelihood` matching its sklearn variant to ULPs) lives
in `.design/bayes/{gaussian,multinomial,bernoulli,complement,categorical}.md`;
the shared prediction pipeline (`predict` / `predict_log_proba` / `predict_proba`
/ `predict_joint_log_proba` as `BaseNB` trait defaults) lives in
`.design/bayes/base.md`. `conjugate.rs` (`posterior_normal_normal`) has no
scikit-learn analog and is unrouted/out-of-scope.

## Probes (live sklearn / scipy oracle)

The normalization REQ is pinned against the live oracle. `log_softmax_rows`
computes `jll − logsumexp(jll, axis=1)`, identically to sklearn's
`predict_log_proba`; `predict_proba` is `exp()` of that.

```bash
python3 -c "
import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB
X = np.array([[1.0,2.0],[1.5,2.5],[1.2,1.8],[6.0,7.0],[5.8,6.5],[6.2,7.2]])
y = np.array([0,0,0,1,1,1])
clf = GaussianNB().fit(X,y)
Xt = np.array([[2.0,2.0],[6.0,6.5]])
jll = clf._joint_log_likelihood(Xt)
print('predict_log_proba=', clf.predict_log_proba(Xt).tolist())
print('manual jll-lse   =', (jll - logsumexp(jll,axis=1)[:,None]).tolist())
print('predict_proba    =', clf.predict_proba(Xt).tolist())
print('proba rowsums    =', clf.predict_proba(Xt).sum(axis=1).tolist())
"
```

Observed (sklearn 1.5.2):

- `predict_log_proba` = `[[0.0, -431.2711718694835], [-380.0647416360279, 0.0]]`
  — identical to the manual `jll − logsumexp(jll, axis=1)`.
- `predict_proba` = `[[1.0, 5.03e-188], [8.71e-166, 1.0]]`; row sums = `[1.0, 1.0]`.

ferrolearn's `FittedGaussianNB::predict_log_proba` / `predict_proba` (delegating
through `BaseNB::nb_predict_log_proba` → `log_softmax_rows`) reproduce these to
~1e-12 (pinned by `tests/divergence_base_nb.rs` and the per-variant divergence
suites).

### Edge-case probe (the all-`-inf` row and single-class row)

`logsumexp` of an all-`-inf` row is `-inf`, so `jll − (-inf) = nan` (scipy emits
a `RuntimeWarning: invalid value encountered in subtract`):

```bash
python3 -c "
import numpy as np
from scipy.special import logsumexp
jll = np.array([[-np.inf, -np.inf]])
print('lse =', logsumexp(jll, axis=1).tolist())              # [-inf]
print('jll - lse =', (jll - logsumexp(jll,axis=1)[:,None]).tolist())  # [[nan, nan]]
jll1 = np.array([[-3.5]])
print('single col jll - lse =', (jll1 - logsumexp(jll1,axis=1)[:,None]).tolist())  # [[0.0]]
"
```

`log_softmax_rows` reproduces this exactly. For an all-`-inf` row,
`max_score = -inf`; `(-inf) − (-inf) = NaN` (IEEE-754); `sum_exp = NaN`;
`log_norm = -inf + ln(NaN) = NaN`; output `= jll − NaN = NaN` — matching scipy's
`NaN`. For a single-column row (`n_classes == 1`, e.g. a single-class discrete
NB), `max_score = jll[i,0]`; `sum_exp = exp(0) = 1`; `log_norm = jll[i,0]`;
output `= 0.0`, i.e. `predict_proba == 1.0` — matching scipy. Verified with a
standalone Rust reproduction of the loop body (`diff=NaN`, `sum_exp=NaN`,
`log_norm=NaN`, `out=NaN`; single-col `out=0`).

## Requirements

- REQ-1 (re-export boundary): the crate root surfaces the five NB variant
  pairs (unfitted + `Fitted*`) and the `BaseNB` trait via `pub use`, and the
  seven submodules via `pub mod`. The variant set mirrors `sklearn/naive_bayes.py`
  `__all__` (`:30-36`). It has a non-test production consumer: the meta-crate
  re-export (`ferrolearn/src/lib.rs`, `pub use ferrolearn_bayes as bayes;`) and
  the `ferrolearn-python` NB pyclass registrations.
- REQ-2 (`log_softmax_rows` == `_BaseNB.predict_log_proba` normalization):
  `log_softmax_rows(jll)` returns `jll − logsumexp(jll, axis=1)` computed in the
  numerically stable max-subtraction form, matching scipy's `logsumexp`
  (`naive_bayes.py:105-126`, `:21`) — including the all-`-inf` (→`NaN`) and
  single-column (→`0.0`) edges. Non-test consumers: `BaseNB::nb_predict_log_proba`
  and, transitively, every `Fitted*NB::predict_log_proba` / `predict_proba`.
- REQ-substrate (ferray): `lib.rs` is on the ferray array substrate
  (`ferray-core` for `Array2`, `ferray-ufunc` for the elementwise `exp`/`ln`),
  per R-SUBSTRATE-1.

## Acceptance criteria

- AC-1: `use ferrolearn_bayes::{GaussianNB, FittedGaussianNB, MultinomialNB,
  FittedMultinomialNB, BernoulliNB, FittedBernoulliNB, CategoricalNB,
  FittedCategoricalNB, ComplementNB, FittedComplementNB, BaseNB};` compiles, and
  `ferrolearn::bayes::GaussianNB` resolves through the meta-crate. The variant
  set equals `naive_bayes.py:__all__` (five variants).
- AC-2: For the GaussianNB probe above, `log_softmax_rows(jll)` matches
  `clf.predict_log_proba(Xt)` and `exp(log_softmax_rows(jll))` matches
  `clf.predict_proba(Xt)` to ≤1e-12; every output row sums to 1.0. For an
  all-`-inf` row the output is `NaN` (matching scipy); for a single-column row
  the output is `0.0` (proba `1.0`).
- AC-substrate: `grep -n "ndarray\|num-traits" ferrolearn-bayes/Cargo.toml`
  returns nothing (the crate depends on `ferray-core` / `ferray-ufunc`, not
  `ndarray` / `num-traits`).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (re-export boundary — `naive_bayes.py` `__all__`) | SHIPPED | `pub use` block in `lib.rs` surfaces `BaseNB`, `GaussianNB`/`FittedGaussianNB`, `MultinomialNB`/`FittedMultinomialNB`, `BernoulliNB`/`FittedBernoulliNB`, `CategoricalNB`/`FittedCategoricalNB`, `ComplementNB`/`FittedComplementNB`; mirrors `__all__ = [BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB]` (`sklearn/naive_bayes.py:30-36`). Non-test consumers: meta-crate `pub use ferrolearn_bayes as bayes;` (`ferrolearn/src/lib.rs`) and the `ferrolearn-python` pyclasses `RsGaussianNB` (`ferrolearn-python/src/classifiers.rs`), `RsMultinomialNB`/`RsBernoulliNB`/`RsComplementNB` (`ferrolearn-python/src/extras.rs`, registered in `ferrolearn-python/src/lib.rs`). Verification: `cargo test -p ferrolearn-bayes` 97 unit + doctests green; `tests/api_proof.rs` (9 passed) exercises the re-exported surface. |
| REQ-2 (`log_softmax_rows` == `jll − logsumexp(jll, axis=1)`, stable) | SHIPPED | impl `pub(crate) fn log_softmax_rows in lib.rs` subtracts the per-row max, sums `exp`, sets `log_norm = max + ln(sum_exp)`, returns `jll − log_norm` — the numerically stable form of `_BaseNB.predict_log_proba` (`sklearn/naive_bayes.py:123-126`, `jll - np.atleast_2d(logsumexp(jll, axis=1)).T`; `logsumexp` = `scipy.special.logsumexp`, `:21`). Non-test consumers: `fn nb_predict_log_proba in base.rs` (`Ok(crate::log_softmax_rows(&jll))`) and each `pub fn predict_log_proba` / `pub fn predict_proba` in `gaussian.rs` / `multinomial.rs` / `bernoulli.rs` / `categorical.rs` / `complement.rs` (delegate to `BaseNB::nb_predict_log_proba` / `nb_predict_proba`). Verification: live oracle `predict_log_proba [[2,2],[6,6.5]]` → `[[0.0,-431.27…],[-380.06…,0.0]]`, ferrolearn matches ~1e-12; all-`-inf` row → `NaN` and single-col → `0.0` match scipy. Live-oracle green guards in `tests/divergence_lib.rs` (4 passed, critic-authored): `green_log_softmax_gaussian_matches_oracle` (end-to-end via `FittedGaussianNB` vs sklearn `predict_log_proba`/`predict_proba` ~1e-12), `green_log_softmax_all_neg_inf_row_is_nan` (matches `scipy.special.logsumexp` NaN), `green_log_softmax_single_column_is_zero`, `green_log_softmax_large_magnitude_no_overflow` (`[[1000,1001]]` → finite, max-subtraction prevents `exp(1000)→inf`). Also `tests/divergence_base_nb.rs` 6 passed, `tests/proptest_invariants.rs` 5 passed. |
| REQ-substrate (ferray) | NOT-STARTED | open prereq blocker **#1110** (lib.rs crate-root substrate; the helper migrates with the NB variants' per-variant blockers #898/#903/#910/#917/#925). `lib.rs` imports `ndarray::Array2` and `num_traits::Float` (the wrong substrate, R-SUBSTRATE-1): `log_softmax_rows`'s signature and body operate on `ndarray::Array2<F>` with `F: num_traits::Float`, and `Cargo.toml` declares `ndarray.workspace = true` + `num-traits.workspace = true`, not `ferray-core` / `ferray-ufunc`. Not migrated. |

## Architecture

`lib.rs` has no estimator state. Its surface is:

- **Module declarations** (`pub mod base; … pub mod multinomial;`) — the seven
  submodules. `base` (the `BaseNB` trait, `check_alpha`), the five NB variant
  modules, and `conjugate` (no sklearn analog).
- **Re-export block** (`pub use base::BaseNB;` … `pub use
  multinomial::{FittedMultinomialNB, MultinomialNB};`) — the public boundary.
  Whereas sklearn exposes only the five concrete estimators in `__all__`
  (`naive_bayes.py:30-36`) and keeps `_BaseNB`/`_BaseDiscreteNB` private,
  ferrolearn surfaces `BaseNB` as well (its introspectable trait analog of
  `_BaseNB`; see `.design/bayes/base.md`). Each variant ships as a pair
  (`GaussianNB` + `FittedGaussianNB`) per the project's unfitted/fitted naming
  convention (CLAUDE.md), with no sklearn-side analog — sklearn mutates one
  object in place.
- **`pub(crate) use base::check_alpha as clamp_alpha;`** — a rename
  pass-through; the floor lives in `base.rs` (`.design/bayes/base.md`).
- **`log_softmax_rows`** — the only computation in the file. Loop over rows:
  `max_score = max_c jll[i,c]`; `sum_exp = Σ_c exp(jll[i,c] − max_score)`;
  `log_norm = max_score + ln(sum_exp)`; `log_proba[i,c] = jll[i,c] − log_norm`.
  This is `jll − logsumexp(jll, axis=1)` with the standard max-subtraction
  stabilization (keeps each `exp` term ≤ 1, avoiding overflow), the exact
  numeric contract of scipy's `logsumexp` used in `_BaseNB.predict_log_proba`
  (`naive_bayes.py:125`). `pub(crate)` because it is a shared internal helper,
  not part of the user API; its observable effect reaches users through every
  fitted NB's `predict_log_proba` / `predict_proba`.

Invariant: for a finite row, `Σ_c exp(log_proba[i,c]) == 1` (each
`predict_proba` row sums to 1, AC-2 / the variant `*_predict_proba_sums_to_one`
tests). Degenerate rows (all `-inf`, single column) follow scipy's behavior
(NaN, 0.0 respectively) by construction of the same arithmetic.

## Verification

```bash
cargo test -p ferrolearn-bayes          # 97 unit + 9 api_proof + divergence_base_nb (6) + proptest_invariants (5) + … all 0 failed
cargo clippy -p ferrolearn-bayes --all-targets -- -D warnings
cargo fmt --all --check
```

REQ-1 boundary: `tests/api_proof.rs` (9 passed) imports and constructs the
re-exported types; the meta-crate `ferrolearn/src/lib.rs` and `ferrolearn-python`
pyclass registrations are the production consumers.

REQ-2 normalization: the GaussianNB live-oracle probe above (`predict_log_proba`
/ `predict_proba` to ~1e-12) plus `tests/divergence_base_nb.rs` and the
`*_predict_proba_sums_to_one` / `*_predict_proba_ordering` per-variant tests.
Edge cases (all-`-inf` → `NaN`, single-col → `0.0`) reproduced against scipy.

REQ-substrate is NOT-STARTED until the ferray migration (blockers
#898/#903/#910/#917/#925) lands; `cargo test` is green on the current
`ndarray` substrate but that does not satisfy R-SUBSTRATE-1.

## Blockers

- **#898 / #903 / #910 / #917 / #925** (ferray substrate, per-variant) — the
  bayes crate (incl. `lib.rs::log_softmax_rows` and the `Array2<F>` /
  `num_traits::Float` signatures) is on `ndarray` + `num-traits`, not
  `ferray-core` / `ferray-ufunc`. REQ-substrate is NOT-STARTED until these land.
  The crate-root usage is migrated as part of the same sweep (no separate
  lib.rs-only blocker is needed; the helper's substrate moves with the variants
  that call it).
