---
title: What tests pass?
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-04
updated: 2026-03-04
---


## Design Specification

### Summary

A single orchestration document that, when kicked off, autonomously builds the entire ferrolearn library across four phases by spawning, monitoring, and merging the work of ~33 subagents. The coordinator is the only thing the human launches. It reads the phase design docs, spawns agents in dependency order, gates phase transitions on acceptance criteria, recovers stuck agents, and merges worktrees into a coherent codebase. The human's role reduces to approving phase gates and resolving any issues the coordinator escalates.

### Requirements

- REQ-1: The coordinator reads `.design/phase-{1,2,3,4}-*.md` as its authoritative task definitions — it does not invent work beyond what the design docs specify
- REQ-2: Agents are spawned using the `Agent` tool with `isolation: "worktree"` so each works on an isolated copy of the repo
- REQ-3: The coordinator maintains a dependency DAG and never spawns an agent before its dependencies have been merged to the integration branch
- REQ-4: Phase transitions (1→2, 2→3, 3→4) are gated on all acceptance criteria for the completing phase passing — verified by running `cargo test --workspace` and `cargo clippy --workspace` on the integration branch
- REQ-5: The coordinator uses crosslink issues to track every agent's assignment, status, and outcome
- REQ-6: When an agent completes, the coordinator merges its worktree branch into the `dev` integration branch, resolving conflicts if necessary
- REQ-7: When an agent is stuck (no progress for 3+ turns, or reports a blocker), the coordinator reads the agent's output, diagnoses the issue, and either provides guidance by resuming the agent or spawns a fixup agent
- REQ-8: After each phase completes, the coordinator runs the full test suite and records the result as a crosslink comment before proceeding
- REQ-9: The coordinator creates a `CLAUDE.md` project file at the start with conventions all subagents must follow (import paths, error handling patterns, formatting, test patterns)
- REQ-10: On context compression, the coordinator recovers by re-reading crosslink issue state and the current phase design doc to reconstruct its position ### Agent Management
- REQ-11: Each spawned agent receives a prompt containing: (a) the specific requirements it owns, (b) the file paths it should create/modify, (c) the acceptance criteria it must satisfy, (d) the dependency versions and import conventions from `CLAUDE.md`, and (e) explicit instructions to commit its work and run `cargo test` before finishing
- REQ-12: Agents are spawned with `model: "sonnet"` for straightforward implementation tasks and `model: "opus"` for architecturally complex tasks (custom L-BFGS, Backend trait, compile-time pipeline, GPU backends)
- REQ-13: The coordinator tracks agent IDs and uses `resume` to continue conversations with agents that need guidance
- REQ-14: Maximum 8 concurrent background agents to avoid resource exhaustion
- REQ-15: Each agent's crosslink issue includes a `--kind result` comment documenting what was delivered, what tests pass, and any known issues ### Merge Strategy
- REQ-16: An integration branch `dev` is created from `main` at the start — all agent work merges here
- REQ-17: Merges happen sequentially (not concurrent) to avoid compound conflicts
- REQ-18: After each merge, the coordinator runs `cargo build --workspace` to verify compilation — if it fails, the coordinator spawns a fixup agent targeting the conflict
- REQ-19: After all agents in a phase are merged, the coordinator runs the full acceptance criteria check before starting the next phase
- REQ-20: At the end of Phase 4, the coordinator creates a PR from `dev` to `main` with a summary of all work

### Acceptance Criteria

- [ ] AC-1: Running `crosslink kickoff run "Build ferrolearn" --doc .design/phase-0-coordinator.md` produces a working library with no further human intervention (beyond phase gate approvals)
- [ ] AC-2: All 55 acceptance criteria across phases 1-4 pass on the final `dev` branch
- [ ] AC-3: Every agent's work is tracked via a crosslink issue with typed comments
- [ ] AC-4: No agent is left in a stuck state for more than 10 minutes without coordinator intervention
- [ ] AC-5: The `dev` branch has a clean, linear-ish commit history with meaningful commit messages
- [ ] AC-6: `cargo test --workspace` passes after each phase gate
- [ ] AC-7: Context compression does not cause the coordinator to lose track of progress or duplicate work

### Architecture

### Coordinator State Machine

```
START
  │
  ├─► Create `dev` branch from `main`
  ├─► Write `CLAUDE.md` with project conventions
  ├─► Create crosslink epic "Build ferrolearn" with sub-issues per agent
  │
  ▼
PHASE_1_CORE
  │
  ├─► Spawn Agent 1: ferrolearn-core (BLOCKING — everything depends on this)
  ├─► Wait for Agent 1 completion
  ├─► Merge Agent 1 worktree → dev
  ├─► Verify: cargo build --workspace
  │
  ▼
PHASE_1_PARALLEL
  │
  ├─► Spawn Agents 2-7 in parallel (all depend only on core)
  │     Agent 2: ferrolearn-sparse
  │     Agent 3: ferrolearn-metrics
  │     Agent 4: ferrolearn-preprocess
  │     Agent 5: ferrolearn-linear (L-BFGS — opus model)
  │     Agent 6: ferrolearn-model-sel
  │     Agent 7: fixtures + test infrastructure
  ├─► As each completes: merge → dev, verify build
  ├─► When all 6 complete: spawn Agent 8 (integration)
  │
  ▼
PHASE_1_INTEGRATION
  │
  ├─► Spawn Agent 8: integration tests, re-export crate, trybuild
  ├─► Merge → dev
  ├─► Run full Phase 1 acceptance criteria
  ├─► If any fail: spawn fixup agents targeting failures
  ├─► PHASE GATE: all Phase 1 ACs pass
  │
  ▼
PHASE_2_PARALLEL
  │
  ├─► Spawn Agents 9-16 in parallel
  │     Agent 9:  ferrolearn-tree (Decision Tree + Random Forest)
  │     Agent 10: ferrolearn-neighbors (kNN)
  │     Agent 11: ferrolearn-bayes (all NB variants)
  │     Agent 12: ferrolearn-linear additions (Linear SVM)
  │     Agent 13: ferrolearn-cluster (k-Means, DBSCAN)
  │     Agent 14: ferrolearn-decomp (PCA, TruncatedSVD)
  │     Agent 15: ferrolearn-model-sel additions (GridSearchCV, etc.)
  │     Agent 16: ferrolearn-io + ferrolearn-datasets
  ├─► As each completes: merge → dev, verify build
  │
  ▼
PHASE_2_INTEGRATION
  │
  ├─► Spawn Agent 17: Phase 2 fixtures + equivalence docs + integration tests
  ├─► PHASE GATE: all Phase 2 ACs pass
  │
  ▼
PHASE_3_PARALLEL
  │
  ├─► Spawn Agents 18-26 in parallel
  │     Agent 18: Gradient Boosting + HistGB + AdaBoost
  │     Agent 19: GMM, HDBSCAN, Agglomerative Clustering
  │     Agent 20: t-SNE, NMF, Kernel PCA, Kernel SVM
  │     Agent 21: Imputers + Feature Selection
  │     Agent 22: Remaining preprocessors
  │     Agent 23: Backend trait + BLAS + no_std (opus model)
  │     Agent 24: ONNX export + Polars/Arrow
  │     Agent 25: Compile-time pipeline proc macro (opus model)
  │     Agent 26: Statistical equivalence benchmarks
  ├─► PHASE GATE: all Phase 3 ACs pass
  │
  ▼
PHASE_4_PARALLEL
  │
  ├─► Spawn Agents 27-33 in parallel
  │     Agent 27: CudaBackend (opus model)
  │     Agent 28: WgpuBackend (opus model)
  │     Agent 29: PartialFit + SGD + streaming
  │     Agent 30: Calibration + Semi-supervised
  │     Agent 31: ColumnTransformer + UMAP + LDA
  │     Agent 32: Remaining P2 algorithms
  │     Agent 33: Formal verification + benchmarks
  ├─► PHASE GATE: all Phase 4 ACs pass
  │
  ▼
FINALIZE
  │
  ├─► Run full test suite on dev
  ├─► Create PR: dev → main
  ├─► Close all crosslink issues
  ├─► Print summary report
  │
  ▼
DONE
```

### Agent Prompt Template

Each agent receives a prompt built from this template:

```
You are building the `{crate_name}` crate for the ferrolearn project — a scikit-learn
equivalent for Rust.

### Out of Scope

- The coordinator does not write algorithm code itself — it only orchestrates
- The coordinator does not modify design docs — if a design gap is found, it makes a decision, documents it in crosslink, and instructs the agent
- The coordinator does not handle deployment, CI setup, or publishing to crates.io
- The coordinator does not run the 24-hour fuzz campaign — it sets up fuzz targets but the long run is a human-initiated step

### your assignment

{requirements extracted from phase design doc}

### acceptance criteria you must satisfy

{acceptance criteria extracted from phase design doc}

### project conventions (from claude.md)

{CLAUDE.md contents}

### files you should create/modify

{file list from architecture section}

### dependencies available

{Cargo.toml dependency block}

### rust edition & msrv

- Edition: 2024
- MSRV: 1.85

### import paths

- Core traits: `use ferrolearn_core::{Fit, Predict, Transform, FitTransform}`
- Errors: `use ferrolearn_core::FerroError`
- Dataset: `use ferrolearn_core::Dataset`
- Array types: `use ndarray::{Array1, Array2, ArrayView1, ArrayView2}`
- Float bound: `use num_traits::Float`

### error handling

- All public functions return `Result<T, FerroError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

### numeric generics

- Generic bound: `F: Float + Send + Sync + 'static`
- Support both f32 and f64
- Use `num_traits::{Zero, One}` where needed

### testing patterns

- Oracle fixtures: load JSON from `fixtures/`, compare with `float_cmp` ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz: one target per public fit/transform/predict
- Compile-fail: `trybuild` for type-safety guarantees

### naming conventions

- Unfitted: `LinearRegression`, `StandardScaler`
- Fitted: `FittedLinearRegression`, `FittedStandardScaler`
- Traits for introspection: `HasCoefficients`, `HasFeatureImportances`, `HasClasses`

