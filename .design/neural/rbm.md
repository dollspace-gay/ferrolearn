# BernoulliRBM — Bernoulli Restricted Boltzmann Machine

<!--
tier: 3-component
status: draft
baseline-commit: f0786d24bf6402bfc4389da3354705ca4b51829c
upstream-paths:
  - sklearn/neural_network/_rbm.py
-->

## Summary

`ferrolearn-neural/src/rbm.rs` mirrors scikit-learn's `sklearn.neural_network.BernoulliRBM`
(`sklearn/neural_network/_rbm.py`, tag 1.5.2, commit 156ef14): an unsupervised feature
learner over a binary visible / binary hidden layer with energy
`E(v,h) = -v^T W h - b_v^T v - b_h^T h`. sklearn trains it with **Stochastic Maximum
Likelihood / Persistent Contrastive Divergence (SML/PCD)** and exposes `transform`,
`gibbs`, `partial_fit`, `score_samples`, and the fitted attributes `components_`,
`intercept_hidden_`, `intercept_visible_`, `h_samples_`, `n_features_in_`.

This doc audits the CURRENT `rbm.rs` against that contract. The current implementation is a
**naive CD-1** learner that diverges from sklearn's SML/PCD at the algorithm-class level, and
is built on the wrong substrate (`ndarray` + `rand_distr` + `rand_xoshiro`, not ferray).
Most behavioral REQs are therefore NOT-STARTED with concrete prerequisites; only the
deterministic mean-hiddens / `transform` path and the parameter defaults are partially in
place, and even those are gated by missing pieces noted below. This doc under-claims by
design (R-HONEST-3).

A note on consumers (R-DEFER-1 / S5): `BernoulliRBM`/`FittedBernoulliRBM` are re-exported
from `ferrolearn-neural/src/lib.rs` and are boundary estimator types, so the type surface is
grandfathered. However there is **no `ferrolearn-python` registration** of the neural crate
and **no non-test production consumer** anywhere in the workspace — the only callers are
`ferrolearn-neural/tests/conformance_wave4.rs` and `ferrolearn-neural/tests/api_proof.rs`
(both `tests/`, i.e. test-only). Per R-HONEST-2 a SHIPPED claim requires impl + a real
non-test consumer; lacking one is recorded as a cross-cutting prerequisite (REQ-9).

## Upstream cite

scikit-learn 1.5.2, `sklearn/neural_network/_rbm.py` (455 lines):
- class + `_parameter_constraints` + `__init__`: `_rbm.py:28`, `:134-141`, `:143-158`
- `transform` / `_mean_hiddens`: `_rbm.py:160-178`, `:180-195`
- `_sample_hiddens` / `_sample_visibles`: `_rbm.py:197-214`, `:216-235`
- `_free_energy`: `_rbm.py:237-252`
- `gibbs`: `_rbm.py:254-273`
- `partial_fit`: `_rbm.py:276-315`
- `_fit` (SML/PCD inner step): `_rbm.py:317-345`
- `score_samples`: `_rbm.py:347-386`
- `fit` (persistent particles + `gen_even_slices`): `_rbm.py:388-442`
- `gen_even_slices`: `sklearn/utils/_chunking.py:86` (contiguous, in-order, no shuffle)

## Requirements

- REQ-1: Constructor parameters, defaults, and parameter validation mirror sklearn —
  `n_components` (default 256, `>=1`), `learning_rate` (0.1, `>0`), `batch_size`
  (10, `>=1`), `n_iter` (10, `>=0`), `verbose` (0), `random_state` (None).
- REQ-2: `transform` computes the deterministic mean hiddens
  `expit(v @ components_.T + intercept_hidden_)` matching `_mean_hiddens`
  (`_rbm.py:180-195`) element-for-element given identical fitted parameters.
- REQ-3: `fit` trains via **SML/PCD**: a persistent particle state `h_samples_` of shape
  `(batch_size, n_components)` initialized once and carried across all batches/epochs;
  each inner step takes `h_pos = mean_hiddens(v_pos)`, `v_neg = sample_visibles(h_samples_)`,
  `h_neg = mean_hiddens(v_neg)`, then resamples the particles
  (`_rbm.py:317-345`, `:417`).
- REQ-4: `fit` iterates **fixed contiguous batch slices in the same order every epoch** via
  `gen_even_slices`, with no per-epoch shuffling (`_rbm.py:419-427`).
- REQ-5: `_sample_visibles` and `_sample_hiddens` return **binary** samples
  `rng.uniform < p` (`_rbm.py:214`, `:235`); `gibbs` is `_sample_hiddens` then
  `_sample_visibles`, both sampled (`_rbm.py:254-273`).
- REQ-6: The SML update equations and learning-rate scaling match sklearn —
  `lr = learning_rate / v_pos.shape[0]`,
  `update = (v_pos.T @ h_pos).T - (h_neg.T @ v_neg)`,
  `intercept_hidden_ += lr*(h_pos.sum0 - h_neg.sum0)`,
  `intercept_visible_ += lr*(v_pos.sum0 - v_neg.sum0)`, with positive/negative phases on
  potentially DIFFERENT row counts (`_rbm.py:335-342`).
- REQ-7: The inspection / scoring surface is present — `score_samples` (pseudo-likelihood:
  corrupt one random feature per sample, `-n_features * logaddexp(0, -(fe_ - fe))`,
  `_rbm.py:347-386`), `_free_energy` (`_rbm.py:237-252`), `partial_fit` (`_rbm.py:276-315`),
  and the fitted attributes `h_samples_` and `n_features_in_` (`_rbm.py:91-93`, `:417`).
- REQ-8: Substrate — the unit computes on the ferray array/random substrate
  (`ferray-core`, `ferray::random`), not `ndarray`/`rand_distr`/`rand_xoshiro` (R-SUBSTRATE).
- REQ-9: A non-test production consumer exists (the `ferrolearn-python` registration of
  `BernoulliRBM` is the canonical one) so the estimator is reachable as `import ferrolearn`
  surface (R-DEFER-1 / R-HONEST-2).

## Acceptance criteria

- AC-1: `BernoulliRBM::new` defaults match `m = BernoulliRBM()` →
  `(256, 0.1, 10, 10, verbose=0, random_state=None)` from a live sklearn call; constructing
  with `n_components=0`, `learning_rate<=0`, or `batch_size=0` is rejected.
- AC-2: Given `components_`, `intercept_hidden_` copied from a live sklearn-fitted model,
  `FittedBernoulliRBM::transform(v)` equals `m.transform(v)` within f64 ULP tolerance on a
  fixed binary `X` (deterministic; must have a failing characterization test until met).
- AC-3: After `fit`, `h_samples_` is present with shape `(batch_size, n_components)` and the
  negative phase draws from it, not from the current batch (structural check on the algorithm).
- AC-4: Batch order is `gen_even_slices(...)` and identical across epochs (no shuffle call).
- AC-5: `gibbs(v)` and the internal visible/hidden sampling return values in `{0,1}` (binary),
  not sigmoid means.
- AC-6: `score_samples(X)` matches `m.score_samples(X)` in formula/shape given a fixed
  corruption index pattern (the RNG-dependent corruption index is an R-DEFER-3 carve-out;
  the deterministic free-energy arithmetic must match).
- AC-7: `cargo tree -p ferrolearn-neural` shows ferray crates and no `ndarray`/`rand_distr`
  in the owned computation path of `rbm.rs`.
- AC-8: `import ferrolearn; from ferrolearn.neural_network import BernoulliRBM` resolves (or
  the Rust-side non-test consumer exists).

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (ctor / defaults / validation) | NOT-STARTED | `fn new in rbm.rs` sets `learning_rate = 0.1`, `n_iter = 10`, `batch_size = 10`, `random_state: None` (`Self { n_components, learning_rate: F::from(0.1)..., n_iter: 10, batch_size: 10, random_state: None }`), which match sklearn's `learning_rate=0.1, batch_size=10, n_iter=10` (`_rbm.py:147-149`). BUT: (a) `new` has no default for `n_components` (sklearn default is `256`, `_rbm.py:145`) and requires it as an argument; (b) there is **no `verbose` parameter** (`_rbm.py:150`); (c) **no parameter validation** — `_parameter_constraints` (`_rbm.py:134-141`) requires `n_components>=1`, `learning_rate>0`, `batch_size>=1`, `n_iter>=0`, but `fn new`/the builder setters reject none of these (e.g. `learning_rate(0.0)` is accepted; `new(0)` is accepted; `batch_size` only clamps with `.max(1)` rather than erroring). Prereq: add `verbose`, a sklearn-default `n_components`, and `_parameter_constraints`-equivalent validation returning `FerroError::InvalidParameter`. Blocker to be filed under #1628. |
| REQ-2 (transform / mean hiddens) | NOT-STARTED | `fn transform in FittedBernoulliRBM` computes `sigmoid(intercept_hidden_[j] + sum_k v[i,k]*components_[j,k])` per element, which is arithmetically `expit(v @ components_.T + intercept_hidden_)` and mirrors `_mean_hiddens` (`_rbm.py:193-195`: `p = safe_sparse_dot(v, self.components_.T); p += self.intercept_hidden_; return expit(p, out=p)`). The formula is correct. NOT-STARTED because: (a) there is **no characterization test** pinning `transform` against live sklearn `_mean_hiddens` given identical components (R-CHAR-1); the existing `rbm_transform_shape_and_range` test only checks shape/range, not values; (b) no non-test production consumer (REQ-9); (c) substrate (REQ-8). Deterministic REQ — the critic MUST file a failing value-parity test (AC-2). Prereq blocker to be filed under #1628. |
| REQ-3 (SML/PCD persistent particles) | NOT-STARTED | `fn fit in rbm.rs` does **naive CD-1**: it samples `h_sample` from the CURRENT batch's `h_pos` (`h_sample[[bi,j]] = if r < p { one } else { zero }` where `p = h_pos[[bi,j]]`) and reconstructs `v_neg` from that sample — there is **no persistent particle state**. sklearn carries `self.h_samples_` across all batches/epochs and draws `v_neg = self._sample_visibles(self.h_samples_, rng)` from the particles, NOT the current batch (`_rbm.py:332`: `v_neg = self._sample_visibles(self.h_samples_, rng)`; `_rbm.py:417`: `self.h_samples_ = np.zeros((self.batch_size, self.n_components)...)`; `_rbm.py:344-345` resamples the particles). Fundamental algorithm-class divergence. Prereq: implement persistent `h_samples_` state + the particle resample step. Blocker to be filed under #1628. |
| REQ-4 (fixed contiguous batch order) | NOT-STARTED | `fn fit in rbm.rs` performs a **Fisher–Yates shuffle of indices every epoch** (`for i in (1..n_samples).rev() { ... indices.swap(i, j); }` inside the `for _epoch` loop). sklearn precomputes `batch_slices` once via `gen_even_slices` and reuses the same contiguous slices every iteration (`_rbm.py:420-427`; `gen_even_slices` at `sklearn/utils/_chunking.py:86` yields ordered contiguous slices, no shuffle). Divergence. Prereq: replace per-epoch shuffle with `gen_even_slices`-equivalent fixed contiguous batching. Blocker to be filed under #1628. |
| REQ-5 (binary sampling in gibbs / visibles) | NOT-STARTED | sklearn `_sample_visibles` returns `rng.uniform(size=p.shape) < p` (binary, `_rbm.py:235`) and `_sample_hiddens` returns `rng.uniform < p` (binary, `_rbm.py:214`); `gibbs` calls both (`_rbm.py:270-271`). ferrolearn `fn gibbs in FittedBernoulliRBM` uses **mean hiddens** (`let h_prob = self.transform(v)?`) then **mean visible sigmoid** (`out[[i,k]] = sigmoid(acc)`) — no Bernoulli sampling at all. The `fn fit` negative phase likewise uses mean-sigmoid `v_neg` (`v_neg[[bi,k]] = sigmoid(acc)`) instead of a sampled visible. Divergence. Prereq: add binary visible/hidden sampling (`_sample_visibles`/`_sample_hiddens`) and route `gibbs` through them. Blocker to be filed under #1628. |
| REQ-6 (SML update equations / lr scaling) | NOT-STARTED | The per-element update direction in `fn fit` (`components[[j,k]] += lr * delta / m_f` with `delta = h_pos[bi,j]*x[si,k] - h_neg[bi,j]*v_neg[bi,k]`) is structurally `h_pos^T x - h_neg^T v_neg` with `lr/m` scaling, which **matches** sklearn's `lr = learning_rate / v_pos.shape[0]` and `update = (v_pos.T @ h_pos).T - (h_neg.T @ v_neg)` (`_rbm.py:335-338`). BUT it is NOT end-to-end correct because ferrolearn ties the positive AND negative phases to the SAME current batch (rows `m`), whereas sklearn's `h_pos` has `n_batch` rows while `v_neg`/`h_neg` have `batch_size` rows from the persistent particles (`_rbm.py:331-342`); the row-count divergence is a direct consequence of the missing REQ-3 particle state. Cannot be SHIPPED while REQ-3/REQ-5 are NOT-STARTED. Prereq: depends on REQ-3 (persistent particles) and REQ-5 (sampled visibles). Blocker to be filed under #1628. |
| REQ-7 (score_samples / free_energy / partial_fit / attrs) | NOT-STARTED | `FittedBernoulliRBM` exposes only `components_`, `intercept_hidden_`, `intercept_visible_`, `n_iter_` (struct fields in `rbm.rs`). It has **no** `score_samples`, **no** `_free_energy`, **no** `partial_fit`, **no** `h_samples_` field, and **no** `n_features_in_` (`_rbm.py:347-386`, `:237-252`, `:276-315`, `:91-93`). Entire inspection/scoring surface is missing. Prereq: implement `_free_energy`, `score_samples` (pseudo-likelihood), `partial_fit`, and add `h_samples_`/`n_features_in_` attributes. Blocker to be filed under #1628. |
| REQ-8 (ferray substrate) | NOT-STARTED | `ferrolearn-neural/Cargo.toml` depends on `ndarray`, `rand_distr`, `rand_xoshiro`; `rbm.rs` imports `use ndarray::{Array1, Array2}` and `use rand_distr::{Distribution, Normal, Uniform}` and uses `Xoshiro256PlusPlus` for all sampling. The destination substrate is `ferray-core` (array type) + `ferray::random` (sampling), per R-SUBSTRATE-1. No ferray usage present. Prereq: migrate the array type to `ferray-core` and sampling to `ferray::random`; this is also the locus where the RNG carve-out (REQ below) is resolved or pinned. Blocker to be filed under #1628. |
| REQ-9 (non-test production consumer) | NOT-STARTED | The only callers of `BernoulliRBM`/`FittedBernoulliRBM` outside `src/rbm.rs` are `ferrolearn-neural/tests/conformance_wave4.rs` and `ferrolearn-neural/tests/api_proof.rs` (both under `tests/`, test-only — verified via `grep -rn BernoulliRBM | grep -v 'src/rbm.rs' | grep -v '#[cfg(test'`). `ferrolearn-neural/src/lib.rs` re-exports the types but does not consume them. There is **no `ferrolearn-python` registration** of the neural crate (`grep -rn neural ferrolearn-python/src/` is empty). Per R-HONEST-2 / R-DEFER-1 a SHIPPED estimator needs a real non-test consumer; the canonical one is the PyO3 `import ferrolearn` registration. Prereq: register `BernoulliRBM` in `ferrolearn-python` (a downstream-layer task per R-DEFER-7, gated on Layer 6). Blocker to be filed under #1628. |

## RNG carve-out vs deterministic classification

Per R-DEFER-3, RNG-stream divergences get a blocker but **NO failing characterization test**
(the test would be unwinnable: numpy `RandomState`/Mersenne-Twister vs `Xoshiro256PlusPlus`
cannot bit-match). What MUST still match — and therefore gets a **failing characterization
test from the critic** — is the algorithm class, the deterministic methods, the defaults, and
the validation/attribute contract.

**RNG carve-outs (blocker only, NO failing test):**
- The post-fit values of `components_`, `intercept_hidden_`, `intercept_visible_`, and
  `h_samples_` after `fit` — these depend on the RNG stream and the random init
  (`rng.normal(0, 0.01, ...)`, `_rbm.py:410`) and cannot bit-match sklearn. Identifiability /
  RNG carve-out under R-DEFER-3.
- The specific corrupted-feature index drawn inside `score_samples`
  (`rng.randint(0, v.shape[1], v.shape[0])`, `_rbm.py:372`) — RNG-dependent; only the
  deterministic free-energy arithmetic given a fixed index pattern is testable.
- The per-step Bernoulli draws inside `fit`/`gibbs` — the sampling MECHANISM (binary
  `uniform < p`) is testable structurally, but the exact drawn bits are an RNG carve-out.
- Determinism-default divergence to DOCUMENT (not bit-match): ferrolearn maps
  `random_state = None` to a fixed seed `0xC0FFEE` (`fn fit`: `None => Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE)`), making it deterministic, whereas sklearn `None` is non-deterministic
  via `check_random_state` (`_rbm.py:407`). This is an intentional, documented divergence
  (deterministic-by-default), not a bug to fix bit-for-bit.

**Deterministic — REQs that MUST have a failing characterization test from the critic
(R-CHAR-1, expected values from LIVE sklearn 1.5.2 per R-CHAR-3):**
- REQ-2 (`transform` = `_mean_hiddens`): value parity given identical fitted parameters.
- REQ-7 free-energy / pseudo-likelihood arithmetic given fixed inputs and a fixed corruption
  pattern (`-n_features * logaddexp(0, -(fe_ - fe))`).
- REQ-1 defaults and parameter validation (purely structural, no RNG).
- REQ-3/REQ-4/REQ-5/REQ-6 algorithm-class structure (persistent particles exist; batch order
  is fixed/contiguous; visibles are sampled binary; update uses particle-derived rows) — these
  are testable structurally without depending on a specific RNG bitstream.

## Architecture

`rbm.rs` defines an unfitted builder `BernoulliRBM<F>` (fields `n_components`,
`learning_rate`, `n_iter`, `batch_size`, `random_state: Option<u64>`) with `#[must_use]`
chained setters, and a fitted `FittedBernoulliRBM<F>` (fields `components_`,
`intercept_hidden_`, `intercept_visible_`, `n_iter_`). `Fit<Array2<F>, ()>` produces the
fitted struct; `Transform<Array2<F>>` and the inherent `fn transform`/`fn gibbs` provide the
post-fit surface. A free `fn sigmoid` is the logistic `expit` analog. Generic over
`F: Float + Send + Sync + 'static`, supporting f32 and f64 per CLAUDE.md.

The unfitted/fitted split and the `components_` / `intercept_hidden_` / `intercept_visible_`
attribute names match sklearn's contract (`_rbm.py:74-93`). The structural gaps vs the sklearn
class are: the missing persistent particle attribute `h_samples_` (`_rbm.py:86-89`, `:417`),
the missing `n_features_in_` (`_rbm.py:91`), and the missing `verbose` constructor parameter
(`_rbm.py:150`). The training loop in `fn fit` is a naive CD-1 (sample-from-current-batch)
rather than sklearn's SML/PCD persistent-particle scheme (`_rbm.py:317-345`, `:388-442`), and
it shuffles each epoch rather than reusing fixed `gen_even_slices` batches
(`sklearn/utils/_chunking.py:86`). The substrate is `ndarray` + `rand_distr` +
`rand_xoshiro`, not ferray (R-SUBSTRATE-1).

## Verification

Commands that would establish SHIPPED claims (none are green for a SHIPPED classification
today; all REQs are NOT-STARTED):

- `cargo test -p ferrolearn-neural` — existing tests (`rbm_fit_smoke`,
  `rbm_transform_shape_and_range`, `rbm_gibbs_round_trip_shape`, `rbm_feature_dim_mismatch`,
  `rbm_empty_input_rejected`) check shape/range/error-paths only; none pin VALUES against
  sklearn, so they do not establish REQ-2..REQ-7 (R-CHAR-1/R-CHAR-3 require live-sklearn
  expected values).
- Live oracle for REQ-2 (deterministic):
  `python3 -c "import numpy as np; from sklearn.neural_network import BernoulliRBM; m=BernoulliRBM(n_components=2, random_state=0); X=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]],float); m.fit(X); print(m.transform(X).tolist())"`
  — feed `m.components_`, `m.intercept_hidden_` into `FittedBernoulliRBM` and compare
  `transform` (critic to pin as a failing test).
- Live oracle for REQ-7:
  `python3 -c "...; print(m.score_samples(X).tolist())"` — observed
  `[-1.1365449, -3.25483523, -3.43272394, -3.2175858]` on the example above; ferrolearn has no
  `score_samples` to compare (critic pins the free-energy arithmetic).
- Defaults check (REQ-1):
  `python3 -c "from sklearn.neural_network import BernoulliRBM; m=BernoulliRBM(); print(m.n_components, m.learning_rate, m.batch_size, m.n_iter, m.verbose, m.random_state)"`
  → `256 0.1 10 10 0 None`.
- Substrate check (REQ-8): `cargo tree -p ferrolearn-neural | grep -E 'ndarray|rand_distr|ferray'`.
- Consumer check (REQ-9):
  `grep -rn "BernoulliRBM" --include=*.rs | grep -v 'src/rbm.rs' | grep -v 'tests/'` is empty.

Because no command above is green in a way that satisfies impl + non-test consumer + live-
sklearn-pinned test simultaneously, every REQ is NOT-STARTED.
