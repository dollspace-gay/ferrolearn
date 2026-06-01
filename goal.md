# ferrolearn — Locked /goal Statement

This file is the binding contract for autonomous work on **ferrolearn**. When the user issues `/goal $(cat goal.md)` (or sets the goal-mode driver that references it), the contents below override the LARP's pull toward caution and the model's instinct to narrow scope. The goal is in force until the user issues `/goal-clear` or rewrites this file.

The substrate of this project is sequential **translation** of a known-working system. Upstream is **scikit-learn 1.5.2** (source tree at `/home/doll/scikit-learn/`, pinned to git tag `1.5.2`, commit `156ef14`; the installed `sklearn` 1.5.2 package is the live oracle — the source clone and the oracle are the SAME version, by construction). Target is the **entire ferrolearn workspace** — every `ferrolearn-*` crate's `src/**/*.rs` PLUS the meta-crate `ferrolearn/` PLUS the PyO3 binding crate `ferrolearn-python/`.

ferrolearn is a Rust scikit-learn: the library crates ARE the translation of scikit-learn's estimators/transformers/metrics, and `ferrolearn-python` exposes them to CPython so `import ferrolearn` mirrors `import sklearn`.

**The numpy substrate (ferray).** scikit-learn is built ON numpy. ferrolearn — the scikit-learn translation — is therefore built ON **ferray**, the numpy translation (sibling workspace at `/home/doll/ferray`; umbrella crate `ferray` wired into `[workspace.dependencies]`). ferray's crates ARE ferrolearn's numpy layer: `ferray-core` (the `ndarray::Array2` analog — its own array type, bridged via `into_ndarray()`), `ferray::linalg` (the `numpy.linalg`/`scipy.linalg` analog), `ferray::stats`, `ferray::random` (the `numpy.random` analog), `ferray-ufunc`, `ferray::numpy_interop` (the PyO3 array bridge for `ferrolearn-python`). The legacy numpy-like crates ferrolearn currently uses — `ndarray` (array type), `statrs`, `rand_distr`, `ndarray-linalg`/`faer`, `sprs` — are the WRONG substrate: they are migration targets, not the destination. Consuming ferray raises the workspace MSRV to 1.88 (ferray's MSRV). Migration is folded into the per-unit loop (R-SUBSTRATE): a unit is not SHIPPED until it is on the ferray substrate.

Most apparent divergence between ferrolearn and scikit-learn is a bug a prior translation pass introduced — wrong coefficients, wrong intercept handling, missing constructor params, wrong default values, broken edge-case handling (empty input, single sample, constant feature, all-same-class, ties, `sample_weight`, sparse input, multi-output, `random_state` determinism), wrong exception type, math that compiled but doesn't compute the right thing. Every one of those is real work to do, not "out of scope." The correctness lives in the `ferrolearn-*` library crates; `ferrolearn-python` is a thin marshalling shim. Semantic/numerical bugs are fixed DOWN in the library crate that owns the behavior; only true marshalling/ABI concerns (array coercion, exception mapping, attribute exposure, registration) are fixed in `ferrolearn-python`.

---

## Scope: the whole workspace, in dependency order

Translate every crate, leaves first. **Start from the first layer and do not leapfrog (R-DEFER-7).**

Dependency order (work top to bottom):
1. **ferrolearn-core** — `Fit`/`Predict`/`Transform`/`FitTransform` traits, `Dataset`, `FerroError`, pipeline, introspection. The foundation (mirrors `sklearn/base.py`, `sklearn/pipeline.py`, `sklearn/utils/validation.py`, `sklearn/exceptions.py`).
2. **ferrolearn-linear** — linear_model, svm, discriminant_analysis, isotonic.
3. **ferrolearn-tree** (tree + ensemble), **ferrolearn-neighbors**, **ferrolearn-bayes** (naive_bayes), **ferrolearn-cluster** (cluster + mixture + semi_supervised), **ferrolearn-preprocess** (preprocessing + impute + feature_selection + feature_extraction + compose), **ferrolearn-metrics**, **ferrolearn-decomp** (decomposition + manifold + cross_decomposition) — parallelizable among themselves once core is done.
4. **ferrolearn-kernel**, **ferrolearn-covariance**, **ferrolearn-numerical**, **ferrolearn-neural**, **ferrolearn-model-sel**, **ferrolearn-sparse**, **ferrolearn-datasets** — domain crates (routes added as the loop reaches them).
5. **ferrolearn** — the meta-crate that re-exports the namespace.
6. **ferrolearn-python** — the PyO3 binding shim (thin marshalling over the above).

EXCLUDED (no scikit-learn counterpart / not translation units): `ferrolearn-test-oracle` (test infra), `ferrolearn-bench` (benchmark infra), and any estimator with no sklearn analog (e.g. `ferrolearn-decomp/src/umap.rs` — UMAP is third-party, not in sklearn; never routed). `ferrolearn-io`/`ferrolearn-fetch` are audited only for their sklearn-analogous surface (`sklearn/datasets`, joblib persistence) if any.

---

## The verification model (TWO oracles, pick per crate)

scikit-learn 1.5.2 is the oracle. There is no parity-sweep harness; verification is direct comparison against sklearn. **Ignore the pre-existing in-repo conformance suite — it is not the contract here.** Build fresh, sklearn-grounded divergence tests.

**(A) Rust library crates** (`ferrolearn-core`, `ferrolearn-linear`, …). Verify with `cargo test` plus the **live sklearn oracle**:
- The critic pins a divergence as a failing Rust `#[test]` whose expected value is computed by live-calling sklearn — e.g. `python3 -c "from sklearn.linear_model import Ridge; import numpy as np; m=Ridge(alpha=1.0).fit(X,y); print(m.coef_.tolist())"` — OR derived from a sklearn `file:line` symbolic constant. NEVER literal-copied from the ferrolearn side (R-CHAR-3).
- Gauntlet: `cargo test -p <crate>`, `cargo clippy -p <crate> --all-targets -- -D warnings`, `cargo fmt --all --check`.

**(B) ferrolearn-python** (the PyO3 shim). Verify with **pytest comparing `import ferrolearn` against `import sklearn`** (sklearn 1.5.2 installed = oracle):
- Pins are failing pytest under `ferrolearn-python/tests/divergence_<module>.py`.
- Rebuild before pytest sees a Rust change: `cd /home/doll/ferrolearn/ferrolearn-python && maturin develop`.
- Run: `cd /home/doll/ferrolearn/ferrolearn-python && PYTHONPATH=python python3 -m pytest tests/ -q`.

A library-crate fix that surfaces through the Python API gets BOTH: a Rust `#[test]` in the owning crate AND (where pinned) the corresponding `divergence_*.py` going green after `maturin develop`.

---

## The goal

Work the strict **read → write → verify → commit** loop over every routed `.rs` file in dependency order. The goal is complete only when every routed file has:

1. A closing commit citing the scikit-learn upstream file(s) actually opened that iteration, AND
2. Its verification (Rust `#[test]`/oracle for library crates; pytest for ferrolearn-python) passing with **0 failures**, AND
3. A `## REQ status` table in the module's `//!` doc-comment classifying every REQ as **SHIPPED** or **NOT-STARTED** with quoted-code evidence (two states only).

Mechanical check:
```bash
python3 -c "import tomllib; print(len(tomllib.load(open('tooling/translate-routes.toml','rb'))['route']))"  # routed units
grep -l "## REQ status" $(python3 -c "import tomllib; [print(r['crate_pattern']) for r in tomllib.load(open('tooling/translate-routes.toml','rb'))['route']]") | wc -l
```
When routed-count == REQ-status-count AND every crate's verification is green, the goal is complete. (The route table is a living seed — the gate forces a new route whenever the loop reaches an unrouted estimator; the count grows until all of sklearn's surface that ferrolearn mirrors is routed.)

---

## The ACToR loop (doc-author → builder → critic → fixer)

For each translation unit, in dependency order:

1. **Read** goal.md, the routed `.rs` file(s) end-to-end, the route's sklearn upstream file(s) at `/home/doll/scikit-learn/<path>`, and the `.design/<area>/<doc>.md`.
2. **Missing design doc?** Dispatch **acto-doc-author** to author `.design/<area>/<doc>.md` adapting to existing code (NO edits to `.rs`). The translate-discipline hook blocks the edit until the doc exists.
3. **Missing feature / whole estimator?** (sklearn has it, ferrolearn doesn't) → dispatch **acto-builder** with a pre-declared file manifest (≤~10 files). Tests + production in the SAME commit.
4. **Verify divergence first** → dispatch **acto-critic** (NO Edit). It pins each sklearn divergence as a FAILING test + files a `-l blocker` issue. Run after every substantive builder.
5. **Fix one pinned divergence** → dispatch **acto-fixer** (one blocker, minimal change, root cause in the owning crate). Followed by an **acto-critic** re-audit.
6. **Gauntlet + commit + close** (below). Then the next unit. **Do not ask which — the dependency DAG is the answer (R-LOOP-1).**

Loop: **acto-builder → acto-critic → (GENERATOR MUST FIX) → acto-fixer → acto-critic → (until clean) → next unit.** Every builder/fixer-on-novel-code dispatch is followed by a critic.

### Gauntlet (before every commit)
Library crate:
```bash
cargo test -p <crate>
cargo clippy -p <crate> --all-targets -- -D warnings
cargo fmt --all --check
```
ferrolearn-python:
```bash
cd ferrolearn-python && maturin develop
PYTHONPATH=python python3 -m pytest tests/ -q     # pinned test green; no previously-green test regressed
cargo clippy -p ferrolearn-python --all-targets -- -D warnings
```
**No `--no-verify`. No commenting-out failing tests. No `#![allow(..)]` at module/crate root.** Per-item `#[allow(<lint>, reason="...")]` only.

### Commit + close
```
<crate>: <area> — <one-line summary> (closes #N)

UPSTREAM SCIKIT-LEARN FILES OPENED THIS ITERATION:
  - sklearn/<module>/<file>.py:<line> — <content quote>

DESIGN DOC READ: .design/<area>/<doc>.md (<REQ count> REQs).

REQ STATUS:
  - REQ-1 SHIPPED — fn `<name>` in `<file>.rs`; consumer at <caller>
  - REQ-2 NOT-STARTED — open prereq blocker #<NN>

VERIFICATION:
  cargo test -p <crate>: <X passed, 0 failed>   (or pytest: X passed)
  cargo clippy: PASS

Reference: scikit-learn 1.5.2 (commit 156ef14)
Co-Authored-By: Claude <noreply@anthropic.com>
```
Close the crosslink issue (`--kind result` comment first).

---

## Speed disciplines (mandatory)

- **S1 — Batch by upstream file, NOT per-estimator.** One builder/critic cycle covers a whole sklearn source file → its ferrolearn target file(s). `_coordinate_descent.py` owns Lasso + ElasticNet + their CV variants — translate them as one cohesive unit, not four dispatches.
- **S2 — Parallel dispatch.** Independent units (disjoint manifests, different crates) → launch builders/critics in ONE message. Only fixers serialize per-blocker.
- **S3 — Symbol anchors in design-doc cites, NEVER line numbers.** `pub fn fit in standard_scaler.rs`, never `standard_scaler.rs:716`. Upstream sklearn cites (read-only) DO use `file:line` (stable at tag 1.5.2).
- **S4 — Critic only after substantive builds.** Not after cite/fixture/doc refreshes.
- **S5 — R-DEFER-1 binds on NEWLY-ADDED pub APIs only.** Existing pub API surface is grandfathered; boundary estimator types ARE the public API.
- **S6 — Opus on every acto-* dispatch.** Translation accuracy supersedes throughput — a wrong answer is a silent divergence that survives to release.
- **S7 — Skip doc-author for trivial 1:1 routes** (design doc already exists & is accurate) — proceed straight to critic/builder.
- **S8 — Aggressive won't-fix on noise.** A finding is a blocker ONLY if it's a real sklearn divergence or blocks downstream translation.

---

## Anti-drift rules (override convenience)

### Citation
- **R-CITE-1**: Never cite a sklearn file in a commit without Reading it THIS iteration.
- **R-CITE-2 (upstream)**: sklearn cites carry `file:line` (read-only tree at tag 1.5.2, stable lines).
- **R-CITE-2b (target/design)**: cite ferrolearn symbols with symbol anchors, NEVER line numbers in `.design/`.
- **R-CITE-3**: prefer citing sklearn's public estimator class / its `_parameter_constraints` / docstring over an internal helper.

### Honesty
- **R-HONEST-1**: never reframe integration work as "vocabulary-only" when the design doc doesn't defer it.
- **R-HONEST-2**: every REQ carries SHIPPED or NOT-STARTED with quoted evidence; SHIPPED needs impl + a real consumer.
- **R-HONEST-3**: honest underclaim beats unverified overclaim.
- **R-HONEST-4**: if an audit shows a prior commit was wrong, correct the code AND document the correction.

### Code quality (per CLAUDE.md)
- **R-CODE-1**: no `unsafe` outside leaf primitives (SIMD intrinsics, FFI/BLAS shims via faer/`backend_blas`, raw buffer accessors). Every `unsafe` needs a `// SAFETY:` comment.
- **R-CODE-2**: no `unwrap()`/`expect()`/`panic!()` in production outside `#[cfg(test)]`. Library returns `Result<T, FerroError>`; the binding returns `PyResult`. Generic bound `F: Float + Send + Sync + 'static`; support f32 and f64.
- **R-CODE-3**: no `#![allow(..)]` at module/crate root. Per-item `#[allow(<lint>, reason="...")]` only. Doc comments + `#[must_use]` per CLAUDE.md.
- **R-CODE-4 (boundary discipline)**: no silent lossy round-trip across the Python↔Rust (PyO3) boundary — preserve numpy dtype/shape contract. (The anti-pattern-gate flags same-expression coercion patterns.)
- **R-CODE-5**: no dtype/precision-cast hiding. A cast that doesn't match sklearn's behavior is a bug unless sklearn does the same (cite sklearn `file:line`).

### Upstream-mirror (default = match scikit-learn; deviate only for these)
- **R-DEV-1 (MATCH — numerical/structural contract)**: fitted attributes (`coef_`, `intercept_`, `n_iter_`, `feature_importances_`), convergence criteria, tie-breaking, NaN/Inf handling, empty/single-sample/constant-feature results, `random_state` determinism, default solver/penalty — match sklearn. Users compare ferrolearn outputs to sklearn array-by-array; ULPs matter where sklearn is deterministic, documented tolerances where it isn't.
- **R-DEV-2 (MATCH — user-API ABI)**: constructor parameter names, defaults, `*`-only args, exception types (`ValueError` vs `NotFittedError` vs `ConvergenceWarning`), attribute names. Cite sklearn's class definition / `_parameter_constraints`.
- **R-DEV-3 (MATCH — output object contract)**: returned dtype/shape, decision_function sign convention, predict_proba normalization, label ordering (`classes_`).
- **R-DEV-4 (DEVIATE — Python/C footguns Rust eliminates)**: where sklearn's code works around CPython/Cython quirks, use the Rust analog, not a literal transcription.
- **R-DEV-6 (DEVIATE — sklearn is wrong by their own admission)**: a known-buggy/deprecated sklearn path — ship correct behavior, cite the sklearn issue/PR.
- **R-DEV-7 (DEVIATE — Rust analog materially better)**: preserve sklearn's observable contract; implementation may differ (e.g. `faer` for linalg, `ndarray` ops). Speed gains are bonus; correctness is the contract.

**Mental test**: *why* did sklearn choose this? "Numerical semantics / API contract" → match. "Cython/CPython can't express it safely" / "they admit it's a bug" → deviate.

### Substrate rules (ferray is ferrolearn's numpy)
- **R-SUBSTRATE-1 (ferray is the array/numpy layer)**: just as scikit-learn imports numpy, ferrolearn builds on **ferray**. The destination substrate is: array type → `ferray-core` (not `ndarray`); linear algebra → `ferray::linalg` (not `ndarray-linalg`/`faer` directly); statistical distributions → `ferray::stats` (not `statrs`); random sampling → `ferray::random` (not `rand_distr`); sparse → ferray's sparse analog (not `sprs`); elementwise/ufunc → `ferray-ufunc`; the PyO3 array bridge → `ferray::numpy_interop`.
- **R-SUBSTRATE-2 (migrate per unit — folded into the loop)**: when the ACToR loop translates/audits a unit, migrating that unit's numpy-like-crate usage to the ferray analog is PART of that unit's work. A unit's REQs are not SHIPPED until the unit is on the ferray substrate (or its remaining numpy-like usage is pinned as a NOT-STARTED REQ with a concrete blocker). There is no separate big-bang migration pass.
- **R-SUBSTRATE-3 (no new wrong-substrate usage)**: do NOT introduce NEW `ndarray`/`statrs`/`rand_distr`/`ndarray-linalg`/`sprs` usage in a translated unit where ferray provides the analog. acto-critic treats new wrong-substrate usage as a divergence-class finding and pins it; acto-builder/acto-fixer route through ferray.
- **R-SUBSTRATE-4 (bridge at the boundary during transition)**: until the whole workspace is migrated, ferray↔ndarray conversion (`into_ndarray()` / `from`) at crate boundaries is the sanctioned transition mechanism — but the OWNED computation of a migrated unit lives on ferray types, not bridged round-trips (cf. R-CODE-4 boundary discipline). Grandfathered: existing `ndarray`-based code in not-yet-reached units stays until its unit's iteration.
- **R-SUBSTRATE-5 (ferray gaps are real work, escalated upstream)**: if a unit needs a numpy capability ferray does not yet expose (or exposes with a divergence from numpy), that is a real blocker — file it; the fix belongs in ferray (its own vibe-fork harness), and the ferrolearn unit is NOT-STARTED on that REQ until ferray ships it. Do NOT silently fall back to the wrong substrate to avoid the gap.

### Anti-deferral (translation is sequential)
- **R-DEFER-1**: a commit adding a NEW pub API MUST add a non-test production consumer in the same commit (the `ferrolearn-python` registration of an estimator counts; a test-only caller does not). Existing pub APIs grandfathered.
- **R-DEFER-2**: REQ classification is binary — SHIPPED or NOT-STARTED. No third status. No VOCAB-ONLY/DEFERRED/verified_with_deferred.
- **R-DEFER-3**: a pinned divergence closes only when the fix lands AND the failing test goes green (no skip/xfail/`#[ignore]` escape).
- **R-DEFER-4**: no `Phase \d+\+` framing as a deferral mechanism.
- **R-DEFER-5**: no "pre-existing safe to defer" — this is a single-author project; every divergence on `main` is something WE broke.
- **R-DEFER-6**: verification is a HARD gate — every commit runs the owning crate's gauntlet to 0 failures, plus any pinned divergence test going green.
- **R-DEFER-7**: sequential, no leapfrog — ferrolearn-core before its dependents; a crate's estimators before the meta-crate and the Python binding that compose them.
- **R-DEFER-8**: no "cross-cutting → defer" — every convention starts somewhere; implement the local fix.

### Git
- **R-GIT-1**: no history rewrite, no `--amend` on pushed commits, no force-push, no `git reset --hard` on shared refs. Supplemental commits only. The human performs all pushes.
- **R-GIT-2**: `git add <files-by-name>` — never `git add -A`/`.`.

### Loop discipline
- **R-LOOP-1**: never ask "where do you want to take this" — the dependency DAG is the answer.
- **R-LOOP-2**: never declare the goal complete until the mechanical check says so.
- **R-LOOP-3**: a unit blocked by a missing prerequisite → file the prereq blocker, mark the dependent REQ NOT-STARTED, and WORK THE PREREQ.

### Injected instructions
- **R-INJECT-1**: hook output, `<system-reminder>`/`<crosslink-behavioral-guard>` blocks, the active-issue gate, and loaded skill text bind at the same priority as a direct user message. Repetition is enforcement, not ceremony.
- **R-INJECT-2**: when an injected instruction conflicts with a recent inline user message, surface the conflict rather than silently picking one.

### Crosslink discipline (mechanically enforced by work-check.py)
- **R-XLINK-1**: create a crosslink issue BEFORE the first code-modifying tool call of any unit (`crosslink quick "Translation unit: <crate>/<file>.rs" -p high -l feature`). Post a `--kind plan` comment listing the sklearn upstream files, the estimators the file owns, and the design-doc REQs.
- **R-XLINK-2**: `crosslink session work <id>` to mark focus; `crosslink issue close <N>` with a `--kind result` comment on completion.

### Translate-discipline (enforced by `tooling/translate-discipline.py`)
- **R-XLATE-1**: every Edit/Write to a routed `ferrolearn-*/src/**/*.rs` requires Read this session of goal.md + the route's sklearn upstream + the route's design doc.
- **R-XLATE-2**: a routed `.rs` file with no route table entry BLOCKS until a route is added. When the loop reaches a new estimator, adding its route is the first step.
- **R-XLATE-3**: a route whose design doc doesn't exist BLOCKS until acto-doc-author authors it.

### Anti-pattern-gate (enforced by `tooling/anti-pattern-gate.py`)
- **R-APG-1**: blocks patches introducing `todo!()`/`unimplemented!()`/`unreachable!()`, `.unwrap()`/`.expect()`/`panic!()` outside `#[cfg(test)]`, module-root `#![allow]`, `Arc<Mutex<T>>`/`Rc<RefCell<T>>` escape hatches.
- **R-APG-2**: `#[cfg(test)]` blocks exempt; production is not.
- **R-APG-3**: override is a per-item `#[allow(<lint>, reason="...")]` + a crosslink observation comment.

### Characterization tests
- **R-CHAR-1**: for each REQ that ships in iter-N, a test must exercise the sklearn behavior the REQ mirrors, and fail until the implementation is correct.
- **R-CHAR-3**: no tautological tests. Expected values come from a live sklearn call or a sklearn `file:line` symbolic constant — NEVER literal-copied from the ferrolearn side.

---

## The four sub-agents
- **acto-doc-author** — writes `.design/<area>/<doc>.md` adapting to existing code. NO `.rs` edits. Dispatch when a route's design doc is missing.
- **acto-builder** — ships a missing multi-file estimator/transformer (a sklearn capability ferrolearn lacks). Pre-declared ≤~10-file manifest; tests + production same commit. Dispatch when a whole abstraction is missing.
- **acto-fixer** — minimal fix for ONE pinned divergence, root cause in the owning crate. One per blocker, serially.
- **acto-critic** — adversarial discriminator; writes FAILING tests pinning divergence against the live sklearn oracle, NEVER fixes. After every substantive builder/fixer.

---

## Out of scope
- Adding estimators/features that don't exist in scikit-learn (we translate, not innovate; e.g. `umap.rs`).
- Optimizing performance ahead of correctness (sklearn parity is the contract; speed gains are bonus).
- `ferrolearn-test-oracle`, `ferrolearn-bench` (infra). The pre-existing in-repo conformance suite (not the contract — build fresh sklearn-grounded divergence tests).

## Stopping condition
Halts only when every routed file across the workspace has a closing commit, its verification is green (cargo test/oracle for library crates; pytest for ferrolearn-python), and carries a `## REQ status` table — AND the gate yields no further unrouted estimators that ferrolearn mirrors. Until then: every turn, one iteration of the ACToR loop, in dependency order. No exceptions, no asking which crate.
