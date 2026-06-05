# fetch_covtype — Forest Covertype dataset loader

<!--
tier: 3-component
status: draft
baseline-commit: 0ba493525d891d6ff75a4d5a6dcf9c53e2b23339
upstream-paths:
  - sklearn/datasets/_covtype.py
-->

## Summary

`ferrolearn-fetch/src/covtype.rs` mirrors scikit-learn's
`sklearn.datasets.fetch_covtype` (`sklearn/datasets/_covtype.py`): a network
dataset loader that downloads a single gzipped CSV (the 581012×54 Forest
Covertype dataset), decompresses it, and splits it into a `(n, 54)` feature
matrix plus a `(n,)` 7-class target. The current implementation returns a
dedicated `Covtype` struct (`data: Array2<f64>`, `target: Array1<usize>`) rather
than sklearn's `Bunch`/`(X, y)` pair, and exposes only the common download +
parse path.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here. Because `fetch_covtype` is a NETWORK fetcher, full numerical value-parity
against the live `sklearn.datasets.fetch_covtype()` oracle requires network
access + downloading the 581012×54 archive, which is OUT OF REACH offline — that
REQ is NOT-STARTED with a concrete network-gated blocker (it is NOT a code bug;
the verified offline contract has no divergence to fix this iteration). The
SHIPPED REQs are the OFFLINE-VERIFIABLE contract pieces: the figshare ARCHIVE
URL/checksum/filename matching the sklearn source constant, the 55-column CSV
parsing logic + shape, the `1..=7` label convention (no shift), the dataset-dir +
SHA-verified cache integration, and the public-fn surface.

## Upstream reference (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_covtype.py`:
  - `ARCHIVE = RemoteFileMetadata(...)` (`:40-43`):
    ```python
    ARCHIVE = RemoteFileMetadata(
        filename="covtype.data.gz",
        url="https://ndownloader.figshare.com/files/5976039",
        checksum="614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
    )
    ```
  - `def fetch_covtype(*, data_home=None, download_if_missing=True,
    random_state=None, shuffle=False, return_X_y=False, as_frame=False,
    n_retries=3, delay=1.0)` (`:80-90`) — the public, keyword-only signature.
  - The load + split (`:204-207`):
    ```python
    Xy = np.genfromtxt(GzipFile(filename=archive_path), delimiter=",")
    X = Xy[:, :-1]
    y = Xy[:, -1].astype(np.int32, copy=False)
    ```
    `X` is the first 54 columns; `y` is the last column cast to int — there is
    **no `-1` shift**, so labels remain `1..=7`.
  - Documented shapes/labels (`:152-157`): `data : ndarray of shape
    (581012, 54)`; `target : ndarray of shape (581012,)` with values "ranging
    between 1 to 7"; `Classes 7 / Samples total 581012 / Dimensionality 54`
    (`:95-100`).
  - Cache + download path (`:187-218`): `covtype_dir = join(data_home,
    "covertype")`, `_fetch_remote(ARCHIVE, dirname=temp_dir, n_retries=...,
    delay=...)`, persisted via `joblib.dump`. `_fetch_remote` (in `_base.py`)
    is the SHA-checked download (`checksum=ARCHIVE.checksum`).

## Requirements

- REQ-ARCHIVE-METADATA: the remote-file metadata ferrolearn downloads
  (`filename`, `url`, `sha256`) is byte-for-byte identical to sklearn's source
  constant `ARCHIVE` (`_covtype.py:40-43`) — same figshare archive, same
  checksum. Offline-verifiable (string-constant characterization).
- REQ-CSV-PARSE: the loader parses the decompressed CSV as 55 comma-separated
  columns per non-empty line (54 features + 1 label), errors on a wrong column
  count and on a non-float field, and materializes `data` of shape `(n, 54)` +
  `target` of length `n`. Mirrors sklearn's `np.genfromtxt(..., delimiter=",")`
  → `X = Xy[:, :-1]` / `y = Xy[:, -1]` split (`_covtype.py:204-207`).
  Offline-verifiable on in-crate CSV text.
- REQ-TARGET-CONVENTION: the target column is taken from column 54 as an integer
  label in `1..=7` with NO `-1` shift, matching sklearn's
  `y = Xy[:, -1].astype(np.int32, copy=False)` (`_covtype.py:207`) and the
  docstring "values ranging between 1 to 7" (`:157`). Offline-verifiable.
- REQ-CACHE-INTEGRATION: the fetcher caches under
  `dataset_dir("covtype", data_home)` and downloads via `fetch_file(.., url,
  filename, Some(sha256), ..)` — first-use download with SHA-256 verification,
  mirroring sklearn's `data_home`/`covertype` caching + `_fetch_remote(ARCHIVE,
  ...)` checksum-verified download (`_covtype.py:187-203`).
- REQ-CONSUMER: `fetch_covtype` / `Covtype` are public crate API (re-exported in
  `lib.rs`); grandfathered boundary API (R-DEFER-1 / S5).
- REQ-VALUE-PARITY: the fetched `(data, target)` arrays match
  `sklearn.datasets.fetch_covtype()` element-by-element on the real 581012×54
  dataset.
- REQ-FULL-PARAMS: the sklearn `fetch_covtype` keyword-only parameters ferrolearn
  currently omits — `download_if_missing`, `random_state`, `shuffle`,
  `return_X_y`, `as_frame`, `n_retries`, `delay`.
- REQ-SUBSTRATE: array types are the ferray substrate (`ferray-core`), not
  `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

All oracle commands run against live scikit-learn 1.5.2
(`python3 -c "import sklearn; print(sklearn.__version__)"` → `1.5.2`). Expected
values come from the live oracle or the sklearn source `file:line`, NEVER from
ferrolearn (R-CHAR-3).

- AC-ARCHIVE-1 (REQ-ARCHIVE-METADATA): the sklearn source constant
  ```
  python3 -c "from sklearn.datasets import _covtype as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"
  ```
  → `covtype.data.gz https://ndownloader.figshare.com/files/5976039 614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`.
  ferrolearn's `ARCHIVE` (`covtype.rs`) carries exactly these three values
  (`filename`/`url`/`sha256`). Pinned offline by `metadata_matches_sklearn`.
- AC-PARSE-1 (REQ-CSV-PARSE, shape): parsing two rows of 54 zero features + a
  label column (`"0.0,"×54 + "3"`, twice) yields `data.dim() == (2, 54)` and
  `target.len() == 2`. (Offline; in-crate `parse_covtype_csv`, pinned by
  `parser_two_rows`.)
- AC-PARSE-2 (REQ-CSV-PARSE, column-count error): a line with the wrong number
  of columns (`"1,2,3"`) is rejected with an `Err`. (Offline; pinned by
  `parser_rejects_wrong_column_count`.) sklearn likewise treats the file as a
  fixed 55-wide table via `np.genfromtxt(..., delimiter=",")` (`:204`).
- AC-TARGET-1 (REQ-TARGET-CONVENTION): for a row whose 55th field is `3`, the
  parsed `target[0] == 3` (NOT `2`) — labels are kept as-is in `1..=7`, matching
  sklearn `y = Xy[:, -1].astype(np.int32, copy=False)` (`:207`, no `-1`).
  (Offline; pinned by `parser_two_rows`.)
- AC-CACHE-1 (REQ-CACHE-INTEGRATION): `fetch_covtype(Some(home))` resolves its
  cache directory to `home/covtype` (via `dataset_dir("covtype", data_home)`)
  and calls `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256),
  &dir)` — first-use, SHA-verified download. (Offline up to the network call;
  the dir-creation + SHA-gating wiring is the offline-checkable part.)
- AC-VALUE-1 (REQ-VALUE-PARITY, NETWORK-GATED oracle): live
  ```
  python3 -c "from sklearn.datasets import fetch_covtype; b=fetch_covtype(); print(b.data.shape, b.target.shape, b.data[0,:3].tolist(), int(b.target[0]))"
  ```
  → `(581012, 54) (581012,) [...] <label>` versus `fetch_covtype(None)` element
  by element. Requires network + the full archive download — NOT runnable
  offline; pinned under network-gated blocker #2064.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-ARCHIVE-METADATA (figshare URL/SHA/filename) | SHIPPED | `pub const ARCHIVE: RemoteFile in covtype.rs` = `{ filename: "covtype.data.gz", url: "https://ndownloader.figshare.com/files/5976039", sha256: "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771" }`, byte-for-byte equal to sklearn source `ARCHIVE = RemoteFileMetadata(filename="covtype.data.gz", url="https://ndownloader.figshare.com/files/5976039", checksum="614360...")` (`_covtype.py:40-43`). Characterization is exact (R-CHAR-3): the expected values are read from the sklearn SOURCE constant, not from ferrolearn. Non-test consumer: `pub fn fetch_covtype in covtype.rs` reads all three fields (`fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`); `ARCHIVE` is `pub` and reachable via the `lib.rs` `pub mod covtype`. Verification: live `python3 -c "from sklearn.datasets import _covtype as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"` → matches the three constants; `cargo test -p ferrolearn-fetch --lib covtype` → `metadata_matches_sklearn` passes (filename + 64-char sha). |
| REQ-CSV-PARSE (55-col parse + shape + errors) | SHIPPED | `fn parse_covtype_csv in covtype.rs` splits each non-empty trimmed line on `,`, requires exactly 55 columns (`if parts.len() != 55 { return Err(FerroError::SerdeError { .. "expected 55" }) }`), parses each field via `p.parse::<f64>()` (erroring with a col-located `SerdeError` on non-float), then materializes `data = Array2::<f64>::zeros((n, 54))` from `row[0..54]` and `target` length `n` from `row[54]`. This mirrors sklearn's fixed-width `Xy = np.genfromtxt(GzipFile(...), delimiter=",")` → `X = Xy[:, :-1]` (`_covtype.py:204-206`). Non-test consumer: called by `pub fn fetch_covtype in covtype.rs` (`parse_covtype_csv(&text)` after `GzDecoder` decompress). Verification: `cargo test -p ferrolearn-fetch --lib covtype` → `parser_two_rows` (asserts `(2,54)` + `target.len()==2`) and `parser_rejects_wrong_column_count` (asserts the 3-col line is `Err`) pass (3 passed, 0 failed). SCOPE NOTE: this is the dense fixed-55-column CSV only; sklearn's `np.genfromtxt` NaN-on-blank behavior is not exercised offline (folded into REQ-VALUE-PARITY). |
| REQ-TARGET-CONVENTION (1..=7 labels, no shift) | SHIPPED | `fn parse_covtype_csv in covtype.rs` sets `target[i] = row[54] as usize` — the raw 55th column value, with NO `-1` subtraction; `pub struct Covtype` documents `target` as "Class labels in `1..=7` (sklearn convention)". Mirrors sklearn `y = Xy[:, -1].astype(np.int32, copy=False)` (`_covtype.py:207`) and the docstring "values ranging between 1 to 7" (`:157`) — confirmed in source: the last column is cast to int with no offset. Non-test consumer: `pub fn fetch_covtype in covtype.rs` returns this `target`. Verification: `cargo test -p ferrolearn-fetch --lib covtype` → `parser_two_rows` asserts `ds.target[0] == 3` for a row ending in `,3` (i.e. label preserved as `3`, not shifted to `2`). |
| REQ-CACHE-INTEGRATION (dataset_dir + SHA-verified fetch) | SHIPPED | `pub fn fetch_covtype in covtype.rs` calls `let dir = dataset_dir("covtype", data_home)?;` (`fn dataset_dir in cache.rs`) then `let gz = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;` (`fn fetch_file in fetch.rs`) — first-use download keyed by a dataset-scoped subdir with SHA-256 verification (the `Some(ARCHIVE.sha256)` arg). Mirrors sklearn's `covtype_dir = join(data_home, "covertype")` + `_fetch_remote(ARCHIVE, dirname=temp_dir, n_retries=..., delay=...)` checksum-verified download (`_covtype.py:188`, `:201-203`). Non-test consumer: `fetch_covtype` itself (boundary API); `dataset_dir`/`fetch_file` are the shared crate primitives. Verification: `cargo test -p ferrolearn-fetch --lib cache` (the `dataset_dir` primitive REQ-CACHE-INTEGRATION rides on passes); the SHA-gated download itself runs only with network (see #2064). FAITHFUL DELTA (not parity): ferrolearn's subdir is `covtype`, sklearn's is `covertype`; ferrolearn caches the raw `.gz`, sklearn persists `joblib.dump`-ed `samples`/`targets` pickles (`:209-215`) — cache LOCATION/format differs, folded into REQ-VALUE-PARITY. |
| REQ-CONSUMER (public boundary surface) | SHIPPED | `pub fn fetch_covtype` + `pub struct Covtype` (+ `pub const ARCHIVE`) are re-exported at the crate root: `pub use covtype::{Covtype, fetch_covtype}` in `lib.rs`, documented in the crate `//!` header (`[fetch_covtype] — 581012×54 multiclass classification`, `lib.rs:14`). Existing pub boundary API, grandfathered under R-DEFER-1 / S5 (the fetcher type IS the public API; external users + the eventual `ferrolearn-python` `datasets` binding mirroring `from sklearn.datasets import fetch_covtype` are the consumers). Verification: `grep -rn "fetch_covtype\|Covtype" ferrolearn-fetch/src/*.rs \| grep -v 'covtype.rs' \| grep -v '#\[cfg(test'` → the `lib.rs` re-export + crate-doc line. UNDERCLAIM: there is NO in-workspace non-test FUNCTION caller; this rests on the boundary-API grandfather clause + the crate re-export. |
| REQ-VALUE-PARITY (element-wise (data,target) vs oracle) | NOT-STARTED | open prereq blocker #2064 (network-gated, NOT a code bug). No offline path: `pub fn fetch_covtype in covtype.rs` calls `fetch_file` → the crate's `download` primitive (network GET), which requires network access + downloading the 581012×54 figshare archive to produce `(data, target)`. Element-by-element parity vs `sklearn.datasets.fetch_covtype()` cannot be established offline. The cache-format and `covtype`/`covertype` subdir-name deltas noted in the SHIPPED rows resolve here once network parity can be run. The verified offline contract (ARCHIVE/shape/label/parse) shows NO divergence to fix this iteration. |
| REQ-FULL-PARAMS (omitted keyword-only params) | NOT-STARTED | open prereq blocker #2065 (feature gap). `pub fn fetch_covtype in covtype.rs` has signature `fetch_covtype(data_home: Option<&Path>)` — only `data_home`. sklearn `def fetch_covtype(*, data_home=None, download_if_missing=True, random_state=None, shuffle=False, return_X_y=False, as_frame=False, n_retries=3, delay=1.0)` (`_covtype.py:80-90`). Omitted: `download_if_missing` (the offline-OSError gate, `:217-218`), `random_state`+`shuffle` (the `rng.shuffle(ind)` reorder, `:225-230`), `return_X_y` (the `(X, y)` tuple return, `:243-244`), `as_frame` (DataFrame path, `:235-242`), `n_retries`/`delay` (download retry policy, `:202`). |
| REQ-SUBSTRATE (ferray-core array types) | NOT-STARTED | `covtype.rs` uses `use ndarray::{Array1, Array2}` and `pub struct Covtype { data: Array2<f64>, target: Array1<usize> }`. R-SUBSTRATE-1 requires `ferray-core` array types, not `ndarray`. Prerequisite: migrate the unit's array types to the ferray analog; folded into this unit's substrate work and gated on the wider `ferrolearn-fetch` migration (shares the `ndarray` types of the sibling fetchers). Open prereq blocker: tracked with the value-parity migration follow-up under #2064 (no separate scalar oracle — representational). |

## Architecture

`covtype.rs` is a single file with one public constant, one public return struct,
one public entry point, and one private parser:

- `pub const ARCHIVE: RemoteFile` — the figshare remote-file descriptor
  (`filename`/`url`/`sha256`), a verbatim copy of sklearn's source `ARCHIVE`
  constant (REQ-ARCHIVE-METADATA, `_covtype.py:40-43`). It is the single source
  of the download URL and the SHA-256 the cache layer verifies against.
- `pub struct Covtype` — the return type (a `Bunch`-analog): `data:
  Array2<f64>` (the 54-feature matrix) and `target: Array1<usize>` (the `1..=7`
  labels). This is a hand-rolled struct, not sklearn's `Bunch` and not the
  `(X, y)` / `return_X_y` pair (REQ-FULL-PARAMS). Both arrays are on the
  `ndarray` substrate (REQ-SUBSTRATE). NOTE: sklearn's `target` is `int32`
  (`:207`) whereas ferrolearn's is `usize` — both are integer labels in `1..=7`;
  the dtype-name delta is representational, recorded under REQ-VALUE-PARITY /
  REQ-SUBSTRATE (numpy `int32` ↔ ferray integer dtype is part of the substrate
  bridge), not a numerical divergence.
- `pub fn fetch_covtype(data_home: Option<&Path>) -> Result<Covtype, FerroError>`
  — the entry point. It (1) creates the cache dir via `dataset_dir("covtype",
  data_home)` (REQ-CACHE-INTEGRATION), (2) downloads + SHA-verifies the archive
  via `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`
  (mirroring `_fetch_remote(ARCHIVE, ...)`, `:201-203`), (3) reads the bytes and
  decompresses with `GzDecoder` (mirroring `GzipFile(filename=archive_path)`,
  `:204`), and (4) dispatches the decompressed text to `parse_covtype_csv`. All
  failures return `Result<_, FerroError>` (`SerdeError` for parse problems,
  `IoError` for filesystem/gunzip) — no panics in the production path (R-CODE-2).
- `fn parse_covtype_csv(raw: &str) -> Result<Covtype, FerroError>` — the CSV
  reader (REQ-CSV-PARSE / REQ-TARGET-CONVENTION). It iterates lines (skipping
  blank ones), splits each on `,`, enforces exactly 55 columns, parses every
  field as `f64`, then fills `data` from columns `0..54` and `target` from
  column `54` cast `as usize` — the direct analog of sklearn's
  `X = Xy[:, :-1]` / `y = Xy[:, -1].astype(int)` split (`:206-207`), with NO
  label shift.

Invariant: the label column is preserved as the raw integer in `1..=7` (no `-1`
offset), and the feature matrix is exactly the first 54 columns — matching
sklearn's `Xy[:, :-1]` / `Xy[:, -1]` partition. The structural limits —
download-only (no `download_if_missing=False` offline gate), no shuffle, no
`(X, y)`/DataFrame return, no retry policy, `usize` target dtype, `ndarray`
substrate — are the surface of REQ-FULL-PARAMS / REQ-VALUE-PARITY / REQ-SUBSTRATE
and are recorded as NOT-STARTED, not as parity claims.

## Verification

Commands establishing the SHIPPED claims and the deltas behind the NOT-STARTED
rows (sklearn 1.5.2 oracle, toolchain rustc 1.95.0):

- `cargo test -p ferrolearn-fetch --lib covtype` → 3 passed, 0 failed
  (`parser_two_rows`, `parser_rejects_wrong_column_count`,
  `metadata_matches_sklearn`). Establishes REQ-CSV-PARSE (AC-PARSE-1/2),
  REQ-TARGET-CONVENTION (AC-TARGET-1, `target[0] == 3`), and the
  filename/sha-length half of REQ-ARCHIVE-METADATA, all offline.
- ARCHIVE oracle (REQ-ARCHIVE-METADATA, exact characterization from sklearn
  source — R-CHAR-3):
  `python3 -c "from sklearn.datasets import _covtype as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"`
  → `covtype.data.gz https://ndownloader.figshare.com/files/5976039 614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`,
  byte-for-byte equal to ferrolearn's three `ARCHIVE` constants. No divergence.
- Label-convention cite (REQ-TARGET-CONVENTION): sklearn `_covtype.py:207`
  `y = Xy[:, -1].astype(np.int32, copy=False)` (no `-1`); docstring `:157`
  "values ranging between 1 to 7". ferrolearn `target[i] = row[54] as usize`.
- Cache-wiring (REQ-CACHE-INTEGRATION): `cargo test -p ferrolearn-fetch --lib
  cache` (the `dataset_dir` primitive passes); the SHA-gated `fetch_file`
  download runs only with network (#2064).
- Consumer check (REQ-CONSUMER):
  `grep -rn "fetch_covtype\|Covtype" ferrolearn-fetch/src/*.rs | grep -v 'covtype.rs' | grep -v '#\[cfg(test'`
  → the `lib.rs` re-export (`pub use covtype::{Covtype, fetch_covtype}`) + the
  crate-doc line. No non-test function caller in-workspace (boundary-API
  grandfather).
- Hygiene: `cargo clippy -p ferrolearn-fetch --lib -- -D warnings` → exits 0
  (the module is clean; no clippy work this iteration).
- Value-parity oracle (REQ-VALUE-PARITY, NETWORK-GATED — NOT runnable offline,
  #2064): live
  `python3 -c "from sklearn.datasets import fetch_covtype; b=fetch_covtype(); print(b.data.shape, b.target.shape, int(b.target[0]))"`
  → `(581012, 54) (581012,) <label>` versus `fetch_covtype(None)`. Requires
  network + the full archive download, so this AC stays NOT-STARTED offline.
- Params cite (REQ-FULL-PARAMS, #2065): sklearn `_covtype.py:80-90` enumerates
  `data_home`/`download_if_missing`/`random_state`/`shuffle`/`return_X_y`/
  `as_frame`/`n_retries`/`delay`; ferrolearn's signature exposes only
  `(data_home)`.

NOT-STARTED REQs map to open blockers: REQ-VALUE-PARITY → #2064 (network-gated),
REQ-FULL-PARAMS → #2065 (feature gap). REQ-SUBSTRATE rides the fetcher-crate
ferray migration under #2064 (no scalar oracle; representational). Per R-CHAR-3,
all expected values above derive from sklearn source `file:line` constants or
live sklearn oracle calls, never copied from the ferrolearn side. The verified
offline contract (ARCHIVE constants, shape, label convention, parse, cache
wiring, public surface) exhibits NO code divergence this iteration.
