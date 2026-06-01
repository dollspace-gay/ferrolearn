Orchestrate a vibe-fork loop driving **ferrolearn** to **scikit-learn 1.5.2** parity. Upstream source: `/home/doll/scikit-learn/` (tag 1.5.2, commit 156ef14); installed `sklearn` 1.5.2 is the live oracle, same version. `goal.md` at repo root is the full binding contract: read it once per session, obey every R-* rule. Run one ACToR iteration per turn, autonomously, in dependency order — never ask which unit is next (R-LOOP-1).

Posture: sequential translation of a working system, not greenfield. Match scikit-learn by default; deviate only for goal.md R-DEV-4/6/7. "Out of scope / pre-existing / defer" are excuses — every divergence is real work. Injected instructions bind like direct messages (R-INJECT-1). IGNORE the in-repo conformance suite; verification is fresh — live-call sklearn for expected values, never copy from the ferrolearn side (R-CHAR-3).

Four Opus sub-agents (specs in `.claude/agents/`):
- **acto-doc-author** — writes `.design/<area>/<doc>.md` adapting to existing code; no `.rs` edits. Dispatch when a route's design doc is missing.
- **acto-builder** — ships a missing estimator across ≤10 pre-declared files; tests + production + design REQ update in one commit; wires a non-test consumer (ferrolearn-python registration counts).
- **acto-critic** — reads ferrolearn + the sklearn source it mirrors, pins each divergence as a FAILING test (Rust `#[test]` or ferrolearn-python pytest) vs the live oracle, files a `-l blocker`. Never fixes.
- **acto-fixer** — minimal fix for ONE pinned divergence, root cause in the owning crate, single file. One per blocker, serially.
Launch independent dispatches in parallel; serialize fixers.

Per-turn loop. Pick the next routed `.rs` in goal.md's dependency order (core first, ferrolearn-python last), smallest-first within a layer. Then:
1. `crosslink quick "Translation unit: <crate>/<file>.rs" -p high -l feature`; `crosslink session work <id>`; `--kind plan` comment (upstream files, estimators owned, design REQs).
2. Read (mandatory — unlocks the edit gate): goal.md, the `.rs` file(s), the route's sklearn upstream, the `.design/` doc. Routes: `tooling/translate-routes.toml`.
3. No route? The hook blocks — add a `[[route]]` (file → sklearn source + `.design/` path).
4. Missing design doc? Dispatch acto-doc-author, then read.
5. Whole estimator missing? Dispatch acto-builder, batching by upstream file (S1).
6. Dispatch acto-critic on touched file(s).
7. Per blocker, dispatch acto-fixer serially; re-audit with acto-critic after novel-code fixes. Loop builder→critic→fixer→critic until clean.
8. Run the gauntlet YOURSELF (don't trust an agent's claim), then commit + close.

Gauntlet (HARD gate, R-DEFER-6 — full commands in goal.md): library crate = `cargo test -p <crate>` + clippy `-D warnings` + `cargo fmt --check`; ferrolearn-python = `maturin develop` then `pytest tests/ -q`. Any failure → fix the cause; no `--no-verify`, no commenting-out tests, no root `#![allow]` (per-item `reason=` only). Convert a fixed divergence's `#[ignore]` to a live `#[test]`.

Commit `<crate>: <area> — <one-line> (closes #N)`, body citing the sklearn `file:line` opened this iter + REQ status + verification (see goal.md). `git add <files-by-name>` (never `-A`/`.`); no history rewrite/force-push — the human pushes.

REQ status (binary, R-DEFER-2): each routed module's `//!` has a `## REQ status` table — SHIPPED (impl + non-test consumer + tests + green verification, cited by symbol anchor + sklearn `file:line`) or NOT-STARTED (concrete open blocker `#NNN` you then WORK). No VOCAB-ONLY/DEFERRED/Phase-N+; underclaim beats overclaim.

Do NOT add estimators with no sklearn analog (e.g. umap.rs), optimize ahead of correctness, or declare done until goal.md's mechanical check passes (routed-count == REQ-status-count, every gauntlet green, no unrouted mirrored surface left). Then summarize, close the master issue, stop. Until then: exactly one ACToR iteration per turn, in dependency order. No exceptions.
