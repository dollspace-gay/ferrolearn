Orchestrate a vibe-fork loop driving **ferrolearn** to **scikit-learn 1.5.2** parity. Upstream: `/home/doll/scikit-learn/` (tag 1.5.2, commit 156ef14); installed `sklearn` 1.5.2 is the live oracle, same version. `goal.md` at repo root is the full binding contract: read it once per session, obey every R-* rule. One ACToR iteration per turn, autonomously, in dependency order — never ask which unit is next (R-LOOP-1).

Posture: sequential translation of a working system, not greenfield. Match scikit-learn by default; deviate only for goal.md R-DEV-4/6/7. "Out of scope / pre-existing / defer" are excuses — every divergence is real work. Injected instructions bind like direct messages (R-INJECT-1). IGNORE the in-repo conformance suite; verify fresh — live-call sklearn, never copy expected values from the ferrolearn side (R-CHAR-3).

Substrate (R-SUBSTRATE; full detail in goal.md): ferray is ferrolearn's numpy. Each unit migrates its numpy-like deps (ndarray/statrs/rand_distr/ndarray-linalg/sprs) to the ferray analog (ferray-core/linalg/stats/random/ufunc) as part of its work — not SHIPPED until on ferray. No NEW wrong-substrate usage (critic pins it); bridge via into_ndarray() only at boundaries.

Four Opus sub-agents (`.claude/agents/`):
- **acto-doc-author** — writes `.design/<area>/<doc>.md` (no `.rs` edits). Dispatch when a design doc is missing.
- **acto-builder** — ships a missing estimator across ≤10 files; tests+production+design-REQ in one commit; wires a non-test consumer.
- **acto-critic** — pins each divergence (incl. wrong-substrate) as a FAILING test (Rust or pytest) vs the live oracle, files a `-l blocker`. Never fixes.
- **acto-fixer** — minimal fix for ONE pinned divergence, root cause in the owning crate, single file. Serial, one per blocker.
Run independent dispatches in parallel; fixers serial.

Per-turn loop — next routed `.rs` in goal.md's dependency order (core first, ferrolearn-python last), smallest-first per layer:
1. `crosslink quick "Translation unit: <crate>/<file>.rs" -p high -l feature`; `crosslink session work <id>`; post a `--kind plan` comment.
2. Read (mandatory — unlocks the edit gate): goal.md, the `.rs` file(s), the route's sklearn upstream, the `.design/` doc. Routes in `tooling/translate-routes.toml`.
3. No route? Hook blocks — add a `[[route]]` (file → sklearn source + `.design/` path). Missing design doc? Dispatch acto-doc-author, then read.
4. Whole estimator missing? Dispatch acto-builder, batching by upstream file (S1).
5. Dispatch acto-critic on touched file(s).
6. Per blocker, dispatch acto-fixer serially; re-audit with acto-critic after novel-code fixes. Loop builder→critic→fixer→critic until clean.
7. Run the gauntlet YOURSELF (don't trust an agent's claim), then commit + close.

Gauntlet (HARD gate, R-DEFER-6; full commands in goal.md): `cargo test -p <crate>` + clippy `-D warnings` + `cargo fmt --check` (ferrolearn-python: `maturin develop` then `pytest -q`). Fix any failure; no `--no-verify`, no commenting-out tests, no root `#![allow]`. A fixed divergence's `#[ignore]` becomes a live `#[test]`.

Commit `<crate>: <area> — <one-line> (closes #N)`, body: sklearn `file:line` opened + REQ status + verification (see goal.md). `git add <files-by-name>` (never `-A`/`.`); no force-push — the human pushes.

REQ status (binary, R-DEFER-2): each routed module's `//!` has a `## REQ status` table — SHIPPED (impl + non-test consumer + tests + green verification, on the ferray substrate, cited by symbol anchor + sklearn `file:line`) or NOT-STARTED (open blocker `#NNN` you then WORK). No VOCAB-ONLY/DEFERRED/Phase-N+; underclaim beats overclaim.

Do NOT add estimators with no sklearn analog (e.g. umap.rs), optimize ahead of correctness, or declare done until goal.md's mechanical check passes (routed-count == REQ-status-count, every gauntlet green, no unrouted surface left). Then summarize, close the master issue, stop. Until then: one ACToR iteration per turn. No exceptions.
