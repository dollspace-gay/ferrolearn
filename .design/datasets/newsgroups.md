# fetch_20newsgroups — 20 Newsgroups text dataset loader

<!--
tier: 3-component
status: draft
baseline-commit: c50754f34763c15cb35de989c0c4f4de863dee0a
upstream-paths:
  - sklearn/datasets/_twenty_newsgroups.py
-->

## Summary

`ferrolearn-fetch/src/newsgroups.rs` mirrors scikit-learn's
`sklearn.datasets.fetch_20newsgroups` (`sklearn/datasets/_twenty_newsgroups.py`):
a network dataset loader that downloads the "by date" 20-newsgroups tarball,
extracts it into `20news-bydate-train/<category>/<doc>` +
`20news-bydate-test/<category>/<doc>` trees, and returns a
`(documents, labels, target_names)` triple for the chosen `subset`
(`Train`/`Test`/`All`). The current implementation returns
`LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)` (the `load_files`
triple) rather than sklearn's `Bunch`/`(X, y)` pair, exposes only `data_home` +
`subset`, and omits sklearn's `categories`/`shuffle`/`random_state`/`remove`/
`download_if_missing`/`return_X_y`/`n_retries`/`delay` parameters.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here. Because `fetch_20newsgroups` is a NETWORK fetcher, full value-parity
against the live `sklearn.datasets.fetch_20newsgroups()` oracle requires network
access + downloading and extracting the ~14 MB tarball, which is OUT OF REACH
offline — that REQ is NOT-STARTED with a concrete network-gated blocker (it is
NOT a code bug). It is compounded by a real ordering divergence: sklearn DEFAULTS
`shuffle=True, random_state=42`, so even WITH the network the document order will
not match sklearn's default output (ferrolearn returns the unshuffled
`load_files` order). The SHIPPED REQs are the OFFLINE-VERIFIABLE contract pieces:
the figshare ARCHIVE URL/SHA/filename matching the sklearn source constant, the
`Train`/`Test`/`All` → folder-name mapping, the `load_files` integration, the
`All` = train ++ test concatenation (with a class-name-mismatch guard), the
dataset-dir + SHA-verified extract-once cache integration, and the public-fn
surface.

## Upstream reference (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_twenty_newsgroups.py`:
  - `ARCHIVE = RemoteFileMetadata(...)` (`:60-64`):
    ```python
    ARCHIVE = RemoteFileMetadata(
        filename="20news-bydate.tar.gz",
        url="https://ndownloader.figshare.com/files/5975967",
        checksum="8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
    )
    ```
  - Folder + cache names (`:66-68`):
    ```python
    CACHE_NAME = "20news-bydate.pkz"
    TRAIN_FOLDER = "20news-bydate-train"
    TEST_FOLDER = "20news-bydate-test"
    ```
  - `def fetch_20newsgroups(*, data_home=None, subset="train", categories=None,
    shuffle=True, random_state=42, remove=(), download_if_missing=True,
    return_X_y=False, n_retries=3, delay=1.0)` (`:177-189`) — the public,
    keyword-only signature.
  - Download + extract + per-subset `load_files` (`:71-100`): `_download_20newsgroups`
    extracts the tarball under `target_dir`, then builds
    `dict(train=load_files(train_path, encoding="latin1"),
    test=load_files(test_path, encoding="latin1"))` — sklearn ITSELF reuses
    `load_files` over the extracted train/test trees. It then persists a zipped
    pickle (`CACHE_NAME = "20news-bydate.pkz"`, `:91-97`).
  - `subset="all"` concatenation (`:331-343`):
    ```python
    for subset in ("train", "test"):
        data = cache[subset]
        data_lst.extend(data.data)
        target.extend(data.target)
        filenames.extend(data.filenames)
    ```
    The order is **train then test**.
  - Default shuffle (`:372-381`): `if shuffle:` (default True) reorders
    `data`/`target`/`filenames` by `check_random_state(random_state).shuffle(...)`
    with `random_state=42` — applied to ALL subsets, AFTER `load_files`.
  - `remove` header/footer/quote stripping (`:349-354`) and `categories` subset
    filtering (`:356-370`).
  - Returned Bunch (`:262-276`): `data` (list of str), `target` (ndarray of int),
    `target_names` (list of category str), `filenames`, `DESCR`.

## Requirements

- REQ-ARCHIVE-METADATA: the remote-file metadata ferrolearn downloads
  (`filename`, `url`, `sha256`) is byte-for-byte identical to sklearn's source
  constant `ARCHIVE` (`_twenty_newsgroups.py:60-64`) — same figshare archive,
  same checksum. Offline-verifiable (string-constant characterization).
- REQ-SUBSET-FOLDER-MAPPING: the `NewsgroupsSubset` selector maps `Train` →
  `20news-bydate-train` and `Test` → `20news-bydate-test`, matching sklearn's
  `TRAIN_FOLDER`/`TEST_FOLDER` (`_twenty_newsgroups.py:67-68`); `All` reads both
  folders. Offline-verifiable (the join targets are string literals).
- REQ-LOAD-FILES-INTEGRATION: the loader reuses
  `ferrolearn_datasets::svmlight::load_files` over the extracted folder to build
  the `(documents, labels, target_names)` triple — mirroring sklearn's own
  `load_files(train_path/test_path, encoding="latin1")` over the extracted tree
  (`_twenty_newsgroups.py:92-93`). Offline-verifiable on a synthetic tree.
- REQ-ALL-CONCAT: for `subset = All`, ferrolearn concatenates the train triple
  then the test triple (train first), erroring if the two subsets' class names
  differ. Mirrors sklearn's `for subset in ("train", "test"): ... extend(...)`
  (`_twenty_newsgroups.py:335-339`), which is also train-then-test.
  Offline-verifiable for the ordering + guard logic.
- REQ-CACHE-INTEGRATION: the fetcher caches under
  `dataset_dir("20newsgroups", data_home)`, downloads via
  `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`
  (first-use, SHA-256-verified), and extracts the tar.gz exactly once
  (`if !extract_root.exists()`). Mirrors sklearn's download + extract + cache
  flow (`_twenty_newsgroups.py:78-99`, `:301-325`). Offline-verifiable up to the
  network call (dir-creation + extract-once wiring).
- REQ-CONSUMER: `fetch_20newsgroups` + `NewsgroupsSubset` are public crate API
  (re-exported in `lib.rs`); grandfathered boundary API (R-DEFER-1 / S5).
- REQ-VALUE-PARITY: the fetched `(documents, labels, target_names)` triple
  matches `sklearn.datasets.fetch_20newsgroups(subset=...)` element by element on
  the real dataset.
- REQ-FULL-PARAMS: the sklearn `fetch_20newsgroups` keyword-only parameters
  ferrolearn currently omits — `categories`, `shuffle`, `random_state`, `remove`,
  `download_if_missing`, `return_X_y`, `n_retries`, `delay`.
- REQ-SUBSTRATE: label array types are the ferray substrate (`ferray-core`), not
  `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

All oracle commands run against live scikit-learn 1.5.2
(`python3 -c "import sklearn; print(sklearn.__version__)"` → `1.5.2`). Expected
values come from the live oracle or the sklearn source `file:line`, NEVER from
ferrolearn (R-CHAR-3).

- AC-ARCHIVE-1 (REQ-ARCHIVE-METADATA): the sklearn source constant
  ```
  python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.ARCHIVE.filename, t.ARCHIVE.url, t.ARCHIVE.checksum)"
  ```
  → `20news-bydate.tar.gz https://ndownloader.figshare.com/files/5975967 8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610`.
  ferrolearn's `ARCHIVE` (`newsgroups.rs`) carries exactly these three values
  (`filename`/`url`/`sha256`). Partially pinned offline by `metadata_matches_sklearn`
  (`filename` literal + 64-char sha length).
- AC-FOLDER-1 (REQ-SUBSET-FOLDER-MAPPING): the sklearn source constants
  ```
  python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.TRAIN_FOLDER, t.TEST_FOLDER)"
  ```
  → `20news-bydate-train 20news-bydate-test`. ferrolearn's `fetch_20newsgroups`
  joins exactly `"20news-bydate-train"` / `"20news-bydate-test"` onto
  `extract_root`. (Offline; string-literal characterization.)
- AC-LOADFILES-1 (REQ-LOAD-FILES-INTEGRATION): extracting a synthetic
  `alpha/doc0.txt` tar.gz and calling `load_files` on the result yields
  `docs.len()==1`, `labels.len()==1`, `names==["alpha"]`. (Offline; pinned by
  `extract_archive_round_trip_synthetic`.) sklearn likewise builds its triple via
  `load_files(...)` over the extracted tree (`:92-93`).
- AC-ALLORDER-1 (REQ-ALL-CONCAT): in `All` mode ferrolearn appends `docs_b`
  (test) after `docs_a` (train) and the test labels after the train labels —
  train-first ordering, matching sklearn `for subset in ("train","test")`
  (`:335`). The class-name-mismatch guard returns a `SerdeError` when
  `names_a != names_b`. (Offline; logic-level — gated only by the network for the
  real tree, see AC-VALUE-1.)
- AC-CACHE-1 (REQ-CACHE-INTEGRATION): `fetch_20newsgroups(Some(home), subset)`
  resolves its cache directory to `home/20newsgroups` (via
  `dataset_dir("20newsgroups", data_home)`), calls
  `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`, and
  extracts into `dir/extracted` only when that directory is absent
  (`if !extract_root.exists()`). (Offline up to the network call; the dir + SHA
  + extract-once wiring is the offline-checkable part.)
- AC-VALUE-1 (REQ-VALUE-PARITY, NETWORK-GATED oracle): live
  ```
  python3 -c "from sklearn.datasets import fetch_20newsgroups; b=fetch_20newsgroups(subset='train', shuffle=False); print(len(b.data), b.target.shape, list(b.target_names))"
  ```
  versus `fetch_20newsgroups(None, Train)` element by element. Requires network +
  the full tarball download + extract — NOT runnable offline; pinned under a
  network-gated blocker. NOTE: the oracle must be called with `shuffle=False` —
  the DEFAULT `shuffle=True, random_state=42` permutes sklearn's document order,
  which ferrolearn does NOT do (folded into REQ-FULL-PARAMS); even with the
  network, ferrolearn matches only the unshuffled order.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-ARCHIVE-METADATA (figshare URL/SHA/filename) | SHIPPED | `pub const ARCHIVE: RemoteFile in newsgroups.rs` = `{ filename: "20news-bydate.tar.gz", url: "https://ndownloader.figshare.com/files/5975967", sha256: "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610" }`, byte-for-byte equal to sklearn source `ARCHIVE = RemoteFileMetadata(filename="20news-bydate.tar.gz", url="https://ndownloader.figshare.com/files/5975967", checksum="8f1b2514...")` (`_twenty_newsgroups.py:60-64`). Characterization is exact (R-CHAR-3): expected values read from the sklearn SOURCE constant, not from ferrolearn. Non-test consumer: `pub fn fetch_20newsgroups in newsgroups.rs` reads all three fields (`fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`); `ARCHIVE` is `pub` and reachable via `lib.rs` `pub mod newsgroups`. Verification: live `python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.ARCHIVE.filename, t.ARCHIVE.url, t.ARCHIVE.checksum)"` → matches all three; `cargo test -p ferrolearn-fetch --lib newsgroups` → `metadata_matches_sklearn` passes (filename literal + 64-char sha). |
| REQ-SUBSET-FOLDER-MAPPING (Train/Test/All → folder names) | SHIPPED | `pub fn fetch_20newsgroups in newsgroups.rs` matches `NewsgroupsSubset::Train => load_files(extract_root.join("20news-bydate-train"))`, `NewsgroupsSubset::Test => load_files(extract_root.join("20news-bydate-test"))`, and `NewsgroupsSubset::All => { load_files(extract_root.join("20news-bydate-train"))? ; load_files(extract_root.join("20news-bydate-test"))? ; .. }`. The two folder literals equal sklearn `TRAIN_FOLDER="20news-bydate-train"` / `TEST_FOLDER="20news-bydate-test"` (`_twenty_newsgroups.py:67-68`). Characterization exact (R-CHAR-3, from sklearn source). Non-test consumer: `fetch_20newsgroups` itself (the `match subset` dispatch). Verification: live `python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.TRAIN_FOLDER, t.TEST_FOLDER)"` → `20news-bydate-train 20news-bydate-test`, byte-for-byte equal to the two `.join(...)` literals. |
| REQ-LOAD-FILES-INTEGRATION (documents/labels/target_names triple) | SHIPPED | `pub fn fetch_20newsgroups in newsgroups.rs` delegates each subset to `load_files(...)` (`use ferrolearn_datasets::svmlight::{LoadFilesResult, load_files}`), returning `LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)` — the `(documents, labels, target_names)` triple. This mirrors sklearn building its Bunch from `load_files(train_path, encoding="latin1")` / `load_files(test_path, encoding="latin1")` over the extracted tree (`_twenty_newsgroups.py:92-93`). `pub fn load_files in svmlight.rs` sorts subdirs (`subdirs.sort()`) → `target_names` and labels are in sorted-category order, the same convention sklearn's `load_files` uses (alphabetical category folders). Non-test consumer: `fetch_20newsgroups` is the production caller of `load_files` (also called by the synthetic-tree test). Verification: `cargo test -p ferrolearn-fetch --lib newsgroups` → `extract_archive_round_trip_synthetic` extracts a tar.gz and asserts `load_files` returns `docs.len()==1`, `labels.len()==1`, `names==["alpha"]` (2 passed, 0 failed). SCOPE NOTE: sklearn reads files as `latin1`; ferrolearn's `load_files` uses `fs::read_to_string` (UTF-8) — an encoding delta folded into REQ-VALUE-PARITY (only observable on the real corpus, which has non-UTF-8 bytes). |
| REQ-ALL-CONCAT (train ++ test order + class-name guard) | SHIPPED | `pub fn fetch_20newsgroups in newsgroups.rs`, `NewsgroupsSubset::All` arm: loads train into `(docs_a, labels_a, names_a)`, then test into `(docs_b, labels_b, names_b)`; on `names_a != names_b` returns `Err(FerroError::SerdeError { message: "20newsgroups: train and test subsets have different class names".into() })`; otherwise `docs_a.extend(docs_b)` and `out_labels = labels_a.to_vec(); out_labels.extend(labels_b.iter().copied())`, returning `(docs_a, labels_a, names_a)` — TRAIN appended first, then TEST. Mirrors sklearn `for subset in ("train", "test"): data_lst.extend(data.data); target.extend(data.target)` (`_twenty_newsgroups.py:335-339`) — confirmed train-then-test in source. The class-name guard has no sklearn analog (sklearn assumes identical folders); it is a defensive R-DEV-4 check that preserves the observable concatenation when names match and surfaces a corrupt extract loudly otherwise. Non-test consumer: `fetch_20newsgroups` (the `All` dispatch). Verification: train-first ordering + guard are code-level (the per-element concat for the real tree is network-gated, see REQ-VALUE-PARITY); the `load_files` triple it concatenates is exercised by `extract_archive_round_trip_synthetic`. |
| REQ-CACHE-INTEGRATION (dataset_dir + SHA-verified fetch + extract-once) | SHIPPED | `pub fn fetch_20newsgroups in newsgroups.rs` calls `let dir = dataset_dir("20newsgroups", data_home)?;` (`pub fn dataset_dir in cache.rs`), then `let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;` (`pub fn fetch_file in fetch.rs`) — first-use SHA-256-verified download — then `let extract_root = dir.join("extracted"); if !extract_root.exists() { extract_archive(&archive_path, &extract_root)?; }` (`fn extract_archive in newsgroups.rs` unpacks `Archive::new(GzDecoder::new(f))`). Mirrors sklearn's `_download_20newsgroups` download + `tarfile.open(..., "r:gz")` extract + cache flow (`_twenty_newsgroups.py:78-99`) and the `os.path.exists(cache_path)` extract-once gate (`:305`). Non-test consumer: `fetch_20newsgroups` itself; `dataset_dir`/`fetch_file` are the shared crate primitives. Verification: `cargo test -p ferrolearn-fetch --lib cache` (the `dataset_dir` primitive passes) + `extract_archive_round_trip_synthetic` exercises `extract_archive`; the SHA-gated network download runs only with network (see REQ-VALUE-PARITY blocker). FAITHFUL DELTA (not parity): sklearn additionally pickles a `CACHE_NAME = "20news-bydate.pkz"` zipped Bunch (`:66`, `:91-97`) and removes the raw extract (`shutil.rmtree`, `:99`); ferrolearn caches the raw `.gz` + the extracted tree under `dir/extracted` (no `.pkz`) — a cache-FORMAT/cleanup delta, folded into REQ-VALUE-PARITY, not a contract divergence. |
| REQ-CONSUMER (public boundary surface) | SHIPPED | `pub fn fetch_20newsgroups` + `pub enum NewsgroupsSubset` (+ `pub const ARCHIVE`) are re-exported at the crate root: `pub use newsgroups::{NewsgroupsSubset, fetch_20newsgroups}` in `lib.rs`, documented in the crate `//!` header (`[fetch_20newsgroups] — text classification (20 categories)`, `lib.rs:16`). Existing pub boundary API, grandfathered under R-DEFER-1 / S5 (the fetcher fn IS the public API; external users + the eventual `ferrolearn-python` `datasets` binding mirroring `from sklearn.datasets import fetch_20newsgroups` are the consumers). Verification: `grep -rn "fetch_20newsgroups\|NewsgroupsSubset" ferrolearn-fetch/src/ | grep -v 'newsgroups.rs' | grep -v '#\[cfg(test'` → the `lib.rs` re-export + crate-doc line. UNDERCLAIM: there is NO in-workspace non-test FUNCTION caller; this rests on the boundary-API grandfather clause + the crate re-export. |
| REQ-VALUE-PARITY (element-wise triple vs oracle) | NOT-STARTED | open prereq blocker #NNN (network-gated, NOT a code bug — to be filed by the critic). No offline path: `pub fn fetch_20newsgroups in newsgroups.rs` calls `fetch_file` → the crate's network GET, which requires network access + downloading + extracting the ~14 MB figshare tarball to produce `(documents, labels, target_names)`. Element-by-element parity vs `sklearn.datasets.fetch_20newsgroups(subset=...)` cannot be established offline. COMPOUND DIVERGENCE: even WITH the network the order will not match sklearn's DEFAULT output — sklearn defaults `shuffle=True, random_state=42` (`_twenty_newsgroups.py:182-183`, applied at `:372-381`), permuting documents/targets, whereas ferrolearn returns the unshuffled `load_files` order; the oracle must be called with `shuffle=False` for an order-comparable run (folded here + into REQ-FULL-PARAMS). The `latin1`-vs-UTF-8 read and the `.pkz` cache-format deltas noted in the SHIPPED rows also resolve here. The verified offline contract (ARCHIVE, folder mapping, load_files triple, concat order, cache wiring) shows NO divergence to fix this iteration. |
| REQ-FULL-PARAMS (omitted keyword-only params) | NOT-STARTED | open prereq blocker #MMM (feature gap — to be filed by the critic). `pub fn fetch_20newsgroups in newsgroups.rs` has signature `fetch_20newsgroups(data_home: Option<&Path>, subset: NewsgroupsSubset)` — only `data_home` + `subset`. sklearn `def fetch_20newsgroups(*, data_home=None, subset="train", categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True, return_X_y=False, n_retries=3, delay=1.0)` (`_twenty_newsgroups.py:177-189`). Omitted: `shuffle`+`random_state` (the DEFAULT-True `random_state.shuffle(indices)` reorder, `:372-381` — the single most behavior-affecting omission, the headline divergence), `remove` (the `headers`/`footers`/`quotes` stripping, `:349-354`), `categories` (subset-of-classes filtering + label renumbering, `:356-370`), `download_if_missing` (the offline-OSError gate, `:326-327`), `return_X_y` (the `(data, target)` tuple return, `:383-384`), `n_retries`/`delay` (download retry policy, `:320-324`). |
| REQ-SUBSTRATE (ferray-core label array) | NOT-STARTED | open prereq blocker (rides the fetcher-crate ferray migration under the REQ-VALUE-PARITY follow-up; representational, no scalar oracle). The `All` arm builds `labels_a = ndarray::Array1::from(out_labels)` and the returned `LoadFilesResult` carries `Array1<usize>` (from `ferrolearn_datasets::svmlight::load_files`, which itself returns `ndarray::Array1`). R-SUBSTRATE-1 requires `ferray-core` array types, not `ndarray`. Prerequisite: migrate `load_files`' return type AND this unit's `Array1::from` to the ferray analog; gated on the wider `ferrolearn-datasets`/`ferrolearn-fetch` substrate migration (the label triple type is owned by `svmlight.rs`). |

## Architecture

`newsgroups.rs` is a single file with one public constant, one public selector
enum, one public entry point, and one private extractor:

- `pub const ARCHIVE: RemoteFile` — the figshare remote-file descriptor
  (`filename`/`url`/`sha256`), a verbatim copy of sklearn's source `ARCHIVE`
  constant (REQ-ARCHIVE-METADATA, `_twenty_newsgroups.py:60-64`). It is the
  single source of the download URL and the SHA-256 the cache layer verifies
  against.
- `pub enum NewsgroupsSubset { Train, Test, All }` — the `subset` selector,
  matching sklearn's `subset` `StrOptions({"train", "test", "all"})`
  (`_twenty_newsgroups.py:391`). `Train`/`Test` each load one folder; `All`
  loads both and concatenates train-then-test (REQ-SUBSET-FOLDER-MAPPING /
  REQ-ALL-CONCAT).
- `pub fn fetch_20newsgroups(data_home: Option<&Path>, subset: NewsgroupsSubset)
  -> Result<LoadFilesResult, FerroError>` — the entry point. It (1) creates the
  cache dir via `dataset_dir("20newsgroups", data_home)` (REQ-CACHE-INTEGRATION),
  (2) downloads + SHA-verifies the tarball via `fetch_file(ARCHIVE.url,
  ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)` (mirroring `_fetch_remote(ARCHIVE,
  ...)`, `:79-81`), (3) extracts into `dir/extracted` exactly once
  (`if !extract_root.exists()`, mirroring sklearn's `os.path.exists(cache_path)`
  gate, `:305`), and (4) dispatches the chosen `subset` to `load_files`, with the
  `All` arm concatenating the train + test triples under a class-name-match guard.
  All failures return `Result<_, FerroError>` (`SerdeError` for the class-name
  mismatch, `IoError` for filesystem/gunzip/tar) — no panics in the production
  path (R-CODE-2).
- `fn extract_archive(archive_path, dest) -> Result<(), FerroError>` — the tar.gz
  unpacker (REQ-CACHE-INTEGRATION). It `fs::create_dir_all(dest)`, opens the
  archive, wraps it `Archive::new(GzDecoder::new(f))`, and `archive.unpack(dest)`
  — the direct analog of sklearn's `with tarfile.open(archive_path, "r:gz") as
  fp: tarfile_extractall(fp, path=target_dir)` (`:84-85`).

The returned triple is the `load_files` `(documents, labels, target_names)`
shape (`LoadFilesResult` in `svmlight.rs`), NOT sklearn's `Bunch`
(`data`/`target`/`target_names`/`filenames`/`DESCR`) and NOT the
`return_X_y=True` pair — ferrolearn omits `filenames` and `DESCR`, recorded under
REQ-VALUE-PARITY / REQ-FULL-PARAMS. The structural limits — download-only (no
`download_if_missing=False` offline gate), NO shuffle (sklearn defaults
`shuffle=True`), no `categories`/`remove` filtering, no `(data, target)` return,
no retry policy, `Array1<usize>` labels on the `ndarray` substrate — are the
surface of REQ-FULL-PARAMS / REQ-VALUE-PARITY / REQ-SUBSTRATE and are recorded as
NOT-STARTED, not as parity claims.

Invariant: the per-subset folder names are exactly `20news-bydate-train` /
`20news-bydate-test`; `All` is train ++ test; categories are taken in
`load_files`' sorted-subdir order; extraction happens once. The single most
behavior-affecting gap vs sklearn is the absent DEFAULT shuffle: sklearn's
out-of-the-box `fetch_20newsgroups()` returns a `random_state=42`-permuted order,
which ferrolearn never reproduces (REQ-FULL-PARAMS / REQ-VALUE-PARITY).

## Verification

Commands establishing the SHIPPED claims and the deltas behind the NOT-STARTED
rows (sklearn 1.5.2 oracle, toolchain rustc per workspace MSRV):

- `cargo test -p ferrolearn-fetch --lib newsgroups` → 2 passed, 0 failed
  (`metadata_matches_sklearn`, `extract_archive_round_trip_synthetic`).
  Establishes the filename/sha-length half of REQ-ARCHIVE-METADATA and the
  extract → `load_files` round-trip of REQ-LOAD-FILES-INTEGRATION /
  REQ-CACHE-INTEGRATION, all offline.
- ARCHIVE oracle (REQ-ARCHIVE-METADATA, exact characterization from sklearn
  source — R-CHAR-3):
  `python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.ARCHIVE.filename, t.ARCHIVE.url, t.ARCHIVE.checksum)"`
  → `20news-bydate.tar.gz https://ndownloader.figshare.com/files/5975967 8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610`,
  byte-for-byte equal to ferrolearn's three `ARCHIVE` constants. No divergence.
- Folder oracle (REQ-SUBSET-FOLDER-MAPPING):
  `python3 -c "from sklearn.datasets import _twenty_newsgroups as t; print(t.TRAIN_FOLDER, t.TEST_FOLDER)"`
  → `20news-bydate-train 20news-bydate-test`, equal to the two `.join(...)`
  literals in `fetch_20newsgroups`.
- Concat-order cite (REQ-ALL-CONCAT): sklearn `_twenty_newsgroups.py:335`
  `for subset in ("train", "test"):` then `.extend(...)` — train then test;
  ferrolearn `docs_a.extend(docs_b)` (a=train, b=test). The class-name guard is
  ferrolearn-only (R-DEV-4).
- Cache-wiring (REQ-CACHE-INTEGRATION): `cargo test -p ferrolearn-fetch --lib
  cache` (the `dataset_dir` primitive passes) + `extract_archive_round_trip_synthetic`
  (the `extract_archive` unpack); the SHA-gated network download runs only with
  network.
- Consumer check (REQ-CONSUMER):
  `grep -rn "fetch_20newsgroups\|NewsgroupsSubset" ferrolearn-fetch/src/ | grep -v 'newsgroups.rs' | grep -v '#\[cfg(test'`
  → the `lib.rs` re-export (`pub use newsgroups::{NewsgroupsSubset, fetch_20newsgroups}`)
  + the crate-doc line. No non-test function caller in-workspace (boundary-API
  grandfather).
- Hygiene: `cargo clippy -p ferrolearn-fetch --lib -- -D warnings` → exits 0
  (the module is clean; no clippy work this iteration).
- Value-parity oracle (REQ-VALUE-PARITY, NETWORK-GATED — NOT runnable offline):
  live
  `python3 -c "from sklearn.datasets import fetch_20newsgroups; b=fetch_20newsgroups(subset='train', shuffle=False); print(len(b.data), b.target.shape, list(b.target_names))"`
  versus `fetch_20newsgroups(None, Train)`. Requires network + the full tarball
  download + extract, so this AC stays NOT-STARTED offline. The oracle MUST be
  called with `shuffle=False` (default is `shuffle=True`), since ferrolearn does
  not shuffle.
- Params cite (REQ-FULL-PARAMS): sklearn `_twenty_newsgroups.py:177-189`
  enumerates `data_home`/`subset`/`categories`/`shuffle`/`random_state`/`remove`/
  `download_if_missing`/`return_X_y`/`n_retries`/`delay`; ferrolearn's signature
  exposes only `(data_home, subset)`. The default `shuffle=True` (`:182`) is the
  headline behavioral omission.

NOT-STARTED REQs map to open blockers (numbers `#NNN`/`#MMM` to be filed by the
critic): REQ-VALUE-PARITY → network-gated blocker (#NNN); REQ-FULL-PARAMS →
feature-gap blocker (#MMM). REQ-SUBSTRATE rides the fetcher-crate ferray
migration under the value-parity follow-up (no scalar oracle; representational —
the `Array1<usize>` label type is owned by `svmlight.rs`'s `load_files`). Per
R-CHAR-3, all expected values above derive from sklearn source `file:line`
constants or live sklearn oracle calls, never copied from the ferrolearn side.
The verified offline contract (ARCHIVE constants, folder mapping, `load_files`
integration, train-first concat, cache/extract-once wiring, public surface)
exhibits NO code divergence this iteration.
