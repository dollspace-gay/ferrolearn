# Datasets cache management — get_data_home / clear_data_home

<!--
tier: 3-component
status: draft
baseline-commit: 86bd54f03887e679197a7e3f08fea344a32a1bce
upstream-paths:
  - sklearn/datasets/_base.py
-->

## Summary

`ferrolearn-fetch/src/cache.rs` mirrors scikit-learn's on-disk dataset cache
primitives — `get_data_home` and `clear_data_home` from
`sklearn/datasets/_base.py`. `get_data_home` resolves a cache directory
(explicit arg → environment override → platform default), creates it on access,
and returns the path; `clear_data_home` resolves the same directory and removes
it recursively. The module additionally exposes `dataset_dir`, a ferrolearn-only
helper with no sklearn analog (per-dataset subdirectory creation) consumed by
every `fetch_*` loader in the crate.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here; the one behavioral gap (`~` expansion) is recorded as a NOT-STARTED REQ
with a concrete prerequisite blocker, and the two deliberate localizations
(env-var name + default location, R-DEV-7) are documented as SHIPPED deviations,
not bugs.

## Upstream reference (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_base.py:45-83` — `def get_data_home(data_home=None) -> str`.
  The resolution body (`:79-83`):
  ```python
  if data_home is None:
      data_home = environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
  data_home = expanduser(data_home)
  makedirs(data_home, exist_ok=True)
  return data_home
  ```
  The docstring (`:54-56`) is explicit: "it can be set by the 'SCIKIT_LEARN_DATA'
  environment variable or programmatically by giving an explicit folder path. The
  '~' symbol is expanded to the user home folder."
- `sklearn/datasets/_base.py:92-107` — `def clear_data_home(data_home=None)`.
  Body (`:106-107`):
  ```python
  data_home = get_data_home(data_home)
  shutil.rmtree(data_home)
  ```
- Both functions carry `@validate_params({"data_home": [str, os.PathLike, None]})`
  (`:39-44`, `:86-91`).

## Requirements

- REQ-GET-DATA-HOME-RESOLUTION: `get_data_home(data_home)` resolves a directory
  by the precedence **explicit arg → environment override → platform default**,
  creates it if absent (`makedirs(..., exist_ok=True)` analog), and returns the
  resolved path. The env-var NAME and default-location differ from sklearn by
  deliberate design (R-DEV-7, see below) — the resolution MECHANISM (arg → env →
  default → create-on-access → return) is the contract being matched.
- REQ-CLEAR-DATA-HOME: `clear_data_home(data_home)` resolves the directory via
  `get_data_home` (which creates it), then removes it recursively, so the cache
  is gone afterwards (`shutil.rmtree` analog).
- REQ-EXPANDUSER: a leading `~` / `~/` in an explicit `data_home` argument OR in
  the environment override is expanded to the user home directory before
  resolution, matching sklearn's `expanduser(data_home)` (`_base.py:81`,
  docstring `:55-56`).
- REQ-CONSUMER: the cache primitives have non-test production consumers inside
  the workspace (the `fetch_*` loaders), not just tests.

## Acceptance criteria

All oracle commands are run from `/tmp` against live scikit-learn 1.5.2
(`python3 -c "import sklearn; print(sklearn.__version__)"` → `1.5.2`). Expected
values come from the live oracle or the sklearn source, NEVER from ferrolearn
(R-CHAR-3).

- AC-RESOLUTION-1 (REQ-GET-DATA-HOME-RESOLUTION, create-on-access + return):
  ```
  python3 -c "from sklearn.datasets import get_data_home; import os; p=get_data_home('/tmp/skl_ac1'); print(p, os.path.isdir(p))"
  ```
  → `/tmp/skl_ac1 True`. ferrolearn `get_data_home(Some("/tmp/skl_ac1"))` returns
  that path and the directory exists (pinned by `explicit_path_used_and_created`).
- AC-RESOLUTION-2 (REQ-GET-DATA-HOME-RESOLUTION, env override mechanism): sklearn
  reads `SCIKIT_LEARN_DATA` when `data_home is None`; ferrolearn reads
  `FERROLEARN_DATA` (deliberate rename, R-DEV-7). The MECHANISM — env var
  consulted only when no explicit arg, default used only when env is unset — is
  identical (`_base.py:79-80`).
- AC-CLEAR-1 (REQ-CLEAR-DATA-HOME):
  ```
  python3 -c "from sklearn.datasets import get_data_home, clear_data_home; import os; p=get_data_home('/tmp/skl_clear_test'); clear_data_home('/tmp/skl_clear_test'); print(os.path.exists(p))"
  ```
  → `False`. ferrolearn `clear_data_home(Some("/tmp/..."))` leaves the directory
  non-existent (pinned by `clear_data_home_removes_explicit_dir`).
- AC-EXPANDUSER-1 (REQ-EXPANDUSER, explicit arg — the FAILING pin):
  ```
  python3 -c "from sklearn.datasets import get_data_home; print(get_data_home('~/sklearn_test_xyz'))"
  ```
  → `/home/doll/sklearn_test_xyz` (EXPANDED). ferrolearn currently returns the
  literal `~/sklearn_test_xyz` (an actual directory named `~`). This is the
  divergence REQ-EXPANDUSER pins.
- AC-EXPANDUSER-2 (REQ-EXPANDUSER, env override path): `expanduser` is applied
  AFTER the env lookup too (`_base.py:81` runs unconditionally), so
  `SCIKIT_LEARN_DATA='~/sklearn_env_test' get_data_home()` →
  `/home/doll/sklearn_env_test`; ferrolearn would use the literal `~`.
- AC-CONSUMER-1 (REQ-CONSUMER):
  ```
  grep -rn "dataset_dir\|get_data_home" ferrolearn-fetch/src/*.rs | grep -v cache.rs | grep -v '#\[cfg(test'
  ```
  finds calls in `california_housing.rs`, `covtype.rs`, `kddcup99.rs`,
  `newsgroups.rs`, `openml.rs`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-GET-DATA-HOME-RESOLUTION (resolve + create + return) | SHIPPED | `pub fn get_data_home in cache.rs` resolves `if let Some(p) = data_home { p.to_path_buf() } else if let Ok(env) = env::var("FERROLEARN_DATA") { PathBuf::from(env) } else if let Some(base) = dirs::data_local_dir() { base.join("ferrolearn_data") } else { PathBuf::from("./ferrolearn_data") }`, then `fs::create_dir_all(&path)?` and returns `Ok(path)` — mirroring sklearn's arg → env → default → `makedirs(exist_ok=True)` → return (`_base.py:79-83`). **DELIBERATE R-DEV-7 deviations (documented, not bugs):** env var is `FERROLEARN_DATA` (sklearn `SCIKIT_LEARN_DATA`) and default is `dirs::data_local_dir()/ferrolearn_data` (`~/.local/share/ferrolearn_data` on Linux; sklearn `~/scikit_learn_data`) — product-identity + XDG-idiomatic choices that preserve the observable resolution-precedence contract while changing only the names/location. Non-test consumer: `pub fn dataset_dir in cache.rs` (`let base = get_data_home(data_home)?;`), itself called by `california_housing.rs` (`dataset_dir("california_housing", data_home)?`), `covtype.rs`, `kddcup99.rs`, `newsgroups.rs`, `openml.rs`; also re-exported at the crate root (`lib.rs`, `pub use cache::{clear_data_home, dataset_dir, get_data_home}`) as public boundary API. Verification: `cargo test -p ferrolearn-fetch --lib cache` → `explicit_path_used_and_created` passes (3 passed, 0 failed); live oracle AC-RESOLUTION-1 → `/tmp/skl_ac1 True`. |
| REQ-CLEAR-DATA-HOME (resolve-then-rmtree) | SHIPPED | `pub fn clear_data_home in cache.rs` calls `let path = get_data_home(data_home)?;` then `if path.exists() { fs::remove_dir_all(&path)? }` — mirroring sklearn's `data_home = get_data_home(data_home); shutil.rmtree(data_home)` (`_base.py:106-107`). The `if path.exists()` guard is behaviorally equivalent to sklearn's unconditional `rmtree`: `get_data_home` always creates the directory first, so it always exists at removal time (R-DEV-4 — the guard removes a TOCTOU/already-absent footgun without altering the observable post-condition "directory gone"). Non-test consumer: re-exported at crate root (`lib.rs`, `pub use cache::{clear_data_home, ...}`) — public boundary API of `ferrolearn-fetch` (grandfathered per R-DEFER-1/S5; the eventual `ferrolearn-python` `datasets` binding is the external consumer, mirroring `from sklearn.datasets import clear_data_home`). Verification: `cargo test -p ferrolearn-fetch --lib cache` → `clear_data_home_removes_explicit_dir` passes; live oracle AC-CLEAR-1 → `False` (gone after clear). |
| REQ-EXPANDUSER (`~` expansion) | NOT-STARTED | open prereq blocker #NNN (to be filed by the critic). `pub fn get_data_home in cache.rs` uses both the explicit arg (`p.to_path_buf()`) and the env value (`PathBuf::from(env)`) LITERALLY — there is no `expanduser` step before `fs::create_dir_all`. sklearn applies `data_home = expanduser(data_home)` unconditionally (`_base.py:81`), expanding a leading `~`/`~/` to the user home for BOTH the explicit arg and the env override. Live oracle (AC-EXPANDUSER-1): `get_data_home('~/sklearn_test_xyz')` → `/home/doll/sklearn_test_xyz` (expanded); ferrolearn instead creates a directory literally named `~`. Fixable in `cache.rs` by expanding a leading `~`/`~/` via `dirs::home_dir()` (the `dirs` crate is already a dependency — `dirs = "5"` in `ferrolearn-fetch/Cargo.toml`) before resolving. The critic pins this as a failing `#[test]` whose expected value is the live `get_data_home('~/...')` expansion (R-CHAR-3), and files the `#NNN` blocker. |
| REQ-CONSUMER (non-test production consumers) | SHIPPED | `pub fn dataset_dir in cache.rs` (ferrolearn-only helper, no sklearn analog — `base.join(name)` + `fs::create_dir_all`) is called from five non-test loader sites: `california_housing.rs` (`dataset_dir("california_housing", data_home)?`), `covtype.rs` (`dataset_dir("covtype", data_home)?`), `kddcup99.rs` (`dataset_dir("kddcup99", data_home)?`), `newsgroups.rs` (`dataset_dir("20newsgroups", data_home)?`), `openml.rs` (`dataset_dir(&format!("openml/{data_id}"), data_home)?`); each transitively exercises `get_data_home`. Verification: `grep -rn "dataset_dir\|get_data_home" ferrolearn-fetch/src/*.rs \| grep -v cache.rs \| grep -v '#\[cfg(test'` → the five fetcher sites above. |

## Architecture

The module is one file, `ferrolearn-fetch/src/cache.rs`, exposing three public
functions over `std::env`, `std::fs`, and `std::path` (no array substrate — pure
path/IO, so there is **no ferray substrate concern** here; R-SUBSTRATE is N/A for
this unit, no `ndarray`/`statrs`/`rand_distr`/`sprs` usage to migrate).

- `pub fn get_data_home(data_home: Option<&Path>) -> Result<PathBuf, FerroError>`
  — the resolution + create-on-access entry point (REQ-GET-DATA-HOME-RESOLUTION,
  mirrors `_base.py:45-83`). Resolution order in source: explicit `Some(p)` →
  `FERROLEARN_DATA` env → `dirs::data_local_dir().join("ferrolearn_data")` →
  `./ferrolearn_data` fallback. The trailing `./ferrolearn_data` arm has no
  sklearn analog: sklearn's `join("~", "scikit_learn_data")` default never fails
  to produce a path, whereas `dirs::data_local_dir()` returns `Option`, so the
  fallback covers the headless/no-HOME case (R-DEV-4, a Rust-side robustness arm
  that cannot change the common-case observable path). `expanduser` is the ONE
  missing step (REQ-EXPANDUSER), inserted in sklearn between resolution and
  `makedirs`.
- `pub fn clear_data_home(data_home: Option<&Path>) -> Result<(), FerroError>` —
  resolve-then-remove (REQ-CLEAR-DATA-HOME, mirrors `_base.py:92-107`). The
  `if path.exists()` guard around `fs::remove_dir_all` is the only structural
  difference from `shutil.rmtree`, and is inert because the preceding
  `get_data_home` creates the directory.
- `pub fn dataset_dir(name, data_home) -> Result<PathBuf, FerroError>` —
  ferrolearn-only helper (**no sklearn analog**; sklearn fetchers build their
  subdirectory paths inline). It is the crate's internal funnel: every `fetch_*`
  loader calls `dataset_dir("<name>", data_home)` to obtain a created,
  dataset-scoped directory, which is why it is the primary non-test consumer of
  `get_data_home` (REQ-CONSUMER).

Error contract: all three return `Result<_, FerroError>`, wrapping `io::Error`
into `FerroError::IoError` (`.map_err(FerroError::IoError)`), per the project's
no-panic library rule. sklearn's `@validate_params` type check on `data_home`
has no ferrolearn analog because the Rust signature (`Option<&Path>`) makes the
"str | os.PathLike | None" contract a compile-time guarantee (R-DEV-4 — the type
system subsumes the runtime validator).

## Verification

Commands establishing the SHIPPED claims and the oracle delta behind the
NOT-STARTED row:

- `cargo test -p ferrolearn-fetch --lib cache` → 3 passed, 0 failed
  (`explicit_path_used_and_created`, `clear_data_home_removes_explicit_dir`,
  `dataset_dir_creates_subdir`). Establishes the create-on-access + return half
  of REQ-GET-DATA-HOME-RESOLUTION and REQ-CLEAR-DATA-HOME's post-condition.
- Consumer check (REQ-CONSUMER):
  `grep -rn "dataset_dir\|get_data_home" ferrolearn-fetch/src/*.rs | grep -v cache.rs | grep -v '#\[cfg(test'`
  → calls in `california_housing.rs`, `covtype.rs`, `kddcup99.rs`,
  `newsgroups.rs`, `openml.rs`.
- Resolution + clear oracle (live sklearn 1.5.2, from `/tmp`):
  - `python3 -c "from sklearn.datasets import get_data_home; import os; p=get_data_home('/tmp/skl_ac1'); print(p, os.path.isdir(p))"`
    → `/tmp/skl_ac1 True` (create-on-access + return; ferrolearn matches).
  - `python3 -c "from sklearn.datasets import get_data_home, clear_data_home; import os; p=get_data_home('/tmp/skl_clear_test'); clear_data_home('/tmp/skl_clear_test'); print(os.path.exists(p))"`
    → `False` (ferrolearn matches via `clear_data_home_removes_explicit_dir`).
- Expanduser oracle (REQ-EXPANDUSER, the NOT-STARTED delta), live sklearn 1.5.2:
  - `python3 -c "from sklearn.datasets import get_data_home; print(get_data_home('~/sklearn_test_xyz'))"`
    → `/home/doll/sklearn_test_xyz` (EXPANDED). ferrolearn creates a directory
    literally named `~`. The critic pins this as a failing `#[test]` whose
    expected value is this live expansion and files the `#NNN` blocker.
  - `SCIKIT_LEARN_DATA='~/sklearn_env_test' python3 -c "from sklearn.datasets import get_data_home; print(get_data_home())"`
    → `/home/doll/sklearn_env_test` (env path is expanded too; `_base.py:81`).

REQ-EXPANDUSER is the single open work item for this unit, tracked under the
critic's `#NNN` blocker; the two R-DEV-7 deviations (env-var name + default
location) are deliberate localizations and require no fix.
