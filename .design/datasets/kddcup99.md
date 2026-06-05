# fetch_kddcup99 — KDD Cup 1999 intrusion-detection dataset loader

<!--
tier: 3-component
status: draft
baseline-commit: 634b5ff0ff81f7ae5d3ca13895f4806b67a448ba
upstream-paths:
  - sklearn/datasets/_kddcup99.py
-->

## Summary

`ferrolearn-fetch/src/kddcup99.rs` mirrors scikit-learn's
`sklearn.datasets.fetch_kddcup99` (`sklearn/datasets/_kddcup99.py`): a network
dataset loader for the KDD Cup 1999 intrusion-detection corpus. It downloads one
of two gzipped CSVs (the full ~743k-row archive or the ~494k-row 10% subset),
decompresses it, and parses each line into a 41-feature row plus a string attack
label. Three columns (`protocol_type`, `service`, `flag`) are symbolic in the
raw CSV; ferrolearn encodes them as f64 level-indices so the whole feature matrix
is numeric. The current implementation returns a dedicated `KddCup99` struct
(`data: Array2<f64>`, `target: Vec<String>`, plus categorical-level bookkeeping)
rather than sklearn's `Bunch`/`(X, y)` pair, and exposes only the common
download + parse path selected by a `KddSubset` enum.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here. Two distinct kinds of NOT-STARTED appear below:

- **HEADLINE DIVERGENCE — a real, offline-detectable, single-const-fixable bug:**
  ferrolearn's `ARCHIVE_10PCT.sha256` does **NOT** match sklearn 1.5.2's 10%
  archive checksum. ferrolearn carries
  `8045aca0d84e70e622d1148d7df782496f6333bf6eb35a805a2cb4ddb1ec1422`; sklearn's
  source constant is
  `8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561`
  (`_kddcup99.py:45`). They share the first 43 hex characters
  (`8045aca0d84e70e622d1148d7df782496f6333bf6eb`) then diverge. With this wrong
  SHA, ferrolearn would **reject the correct sklearn 10% file** as a checksum
  mismatch (`fetch.rs` wipes + errors on mismatch). This is captured as
  REQ-ARCHIVE-10PCT-CHECKSUM = NOT-STARTED (blocker **#2070**); a fixer corrects
  the one constant to the sklearn value this iteration.
- **Network-gated / feature-gap / substrate NOT-STARTED (not bugs to fix this
  iteration):** full element-wise value-parity needs the live download
  (REQ-VALUE-PARITY, #2071); sklearn's `subset`/`shuffle`/`return_X_y`/… params
  are absent (REQ-FULL-PARAMS, #2072); the array type is `ndarray`, not
  `ferray-core` (REQ-SUBSTRATE, #2073).

The SHIPPED REQs are the OFFLINE-VERIFIABLE contract pieces: the **full**-archive
metadata matching the sklearn source constant, the 42-column CSV parse + shape +
error handling, the cols-1/2/3 categorical encoding matching sklearn's symbolic
columns, the last-column string target, the `KddSubset` ↔ two-archive mapping,
the cache integration, and the public-fn surface.

## Upstream reference (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_kddcup99.py`:
  - `ARCHIVE = RemoteFileMetadata(...)` (`:34-38`):
    ```python
    ARCHIVE = RemoteFileMetadata(
        filename="kddcup99_data",
        url="https://ndownloader.figshare.com/files/5976045",
        checksum="3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292",
    )
    ```
  - `ARCHIVE_10_PERCENT = RemoteFileMetadata(...)` (`:42-46`):
    ```python
    ARCHIVE_10_PERCENT = RemoteFileMetadata(
        filename="kddcup99_10_data",
        url="https://ndownloader.figshare.com/files/5976042",
        checksum="8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
    )
    ```
  - `def fetch_kddcup99(*, subset=None, data_home=None, shuffle=False,
    random_state=None, percent10=True, download_if_missing=True,
    return_X_y=False, as_frame=False, n_retries=3, delay=1.0)` (`:66-78`) — the
    public, keyword-only signature; `percent10=True` is the default (10% subset).
  - The 42-entry dtype list (`:320-363`) defines the column order. The three
    symbolic (`"S…"` byte-string) columns are at positions **1, 2, 3**:
    `("duration", int)` (pos 0), `("protocol_type", "S4")` (pos 1, `:322`),
    `("service", "S11")` (pos 2, `:323`), `("flag", "S6")` (pos 3, `:324`), then
    `("src_bytes", int)` (pos 4) … through `("labels", "S16")` (pos 41, `:362`).
    All other feature columns are `int`/`float`. The last entry `labels` is the
    target.
  - Column split (`:399-400`): `X = Xy[:, :-1]` (the 41 features),
    `y = Xy[:, -1]` (the labels column) — `target` is the last column kept as a
    byte string (e.g. `b"normal."`, `:188`, `:296`).
  - Per-column casting (`:396-397`): `for j in range(42): Xy[:, j] =
    Xy[:, j].astype(DT[j])` — exactly 42 columns; symbolic columns become byte
    strings, the rest numeric.
  - Cache/download (`:306-408`): `kddcup_dir = join(data_home, "kddcup99_10" +
    dir_suffix)` (10%) or `"kddcup99" + dir_suffix` (full); `archive =
    ARCHIVE_10_PERCENT if percent10 else ARCHIVE` (`:309-314`); SHA-checked
    `_fetch_remote(archive, dirname=kddcup_dir, n_retries=…, delay=…)` (`:382`);
    persisted via `joblib.dump` (`:405-406`).

## Requirements

- REQ-ARCHIVE-FULL-METADATA: the **full**-archive remote-file metadata
  ferrolearn downloads (`filename`, `url`, `sha256`) is byte-for-byte identical
  to sklearn's source constant `ARCHIVE` (`_kddcup99.py:34-38`) — same figshare
  archive (`/5976045`), same checksum. Offline-verifiable (string-constant
  characterization).
- REQ-ARCHIVE-10PCT-CHECKSUM (**HEADLINE DIVERGENCE**): the **10%**-subset
  archive checksum ferrolearn verifies against must equal sklearn's source
  constant `ARCHIVE_10_PERCENT.checksum`
  (`8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561`,
  `_kddcup99.py:45`). It currently does NOT. Offline-detectable
  (string-constant characterization); single-const fix.
- REQ-CSV-PARSE: the loader parses the decompressed CSV as 42 comma-separated
  columns per non-empty line (41 features + 1 label), errors on a wrong column
  count and on a non-float numeric field, and materializes `data` of shape
  `(n, 41)` + `target` of length `n`. Mirrors sklearn's per-column cast over the
  42-entry dtype (`_kddcup99.py:396-397`) → `X = Xy[:, :-1]` / `y = Xy[:, -1]`
  split (`:399-400`). Offline-verifiable on in-crate CSV text.
- REQ-CATEGORICAL-ENCODING: columns **1, 2, 3** (`protocol_type`, `service`,
  `flag`) are treated as categorical and encoded as f64 level-indices (level in
  order of first appearance), matching sklearn's symbolic byte-string columns at
  those exact positions (`("protocol_type","S4")`/`("service","S11")`/
  `("flag","S6")`, `_kddcup99.py:322-324`). The remaining 38 feature columns are
  parsed as f64. Offline-verifiable on in-crate CSV text.
- REQ-TARGET-LABELS: the target is the **last** (42nd) column kept as a string
  per row (e.g. `"normal."`, `"smurf."`), matching sklearn's `y = Xy[:, -1]`
  bytes-labels (`_kddcup99.py:400`, `:188`). Offline-verifiable.
- REQ-SUBSET-VARIANT: `KddSubset::{Full, Percent10}` selects the full archive
  vs the 10% archive, mirroring sklearn's `percent10` bool (default `True` →
  10% subset; `archive = ARCHIVE_10_PERCENT if percent10 else ARCHIVE`,
  `_kddcup99.py:309-314`). Offline-verifiable (mapping logic).
- REQ-CACHE-INTEGRATION: the fetcher caches under
  `dataset_dir("kddcup99", data_home)` and downloads via `fetch_file(url,
  filename, Some(sha256), &dir)` — first-use download with SHA-256 verification,
  mirroring sklearn's `kddcup_dir`/`_fetch_remote(archive, …)` checksum-verified
  download (`_kddcup99.py:306-382`).
- REQ-CONSUMER: `fetch_kddcup99` / `KddCup99` / `KddSubset` are public crate API
  (re-exported in `lib.rs`); grandfathered boundary API (R-DEFER-1 / S5).
- REQ-VALUE-PARITY: the fetched `(data, target)` arrays match
  `sklearn.datasets.fetch_kddcup99()` element-by-element on the real dataset.
  NETWORK-GATED.
- REQ-FULL-PARAMS: the sklearn `fetch_kddcup99` keyword-only parameters
  ferrolearn currently omits — `subset` (the SA/SF/http/smtp attack-type
  filters), `shuffle`/`random_state`, `download_if_missing`, `return_X_y`,
  `as_frame`, `n_retries`, `delay`. Feature gap.
- REQ-SUBSTRATE: array types are the ferray substrate (`ferray-core`), not
  `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

All oracle commands run against live scikit-learn 1.5.2
(`python3 -c "import sklearn; print(sklearn.__version__)"` → `1.5.2`). Expected
values come from the live oracle or the sklearn source `file:line`, NEVER from
ferrolearn (R-CHAR-3).

- AC-ARCHIVE-FULL-1 (REQ-ARCHIVE-FULL-METADATA): the sklearn source constant
  ```
  python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE.filename, k.ARCHIVE.url, k.ARCHIVE.checksum)"
  ```
  → `kddcup99_data https://ndownloader.figshare.com/files/5976045 3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292`.
  ferrolearn's `ARCHIVE_FULL` (`kddcup99.rs`) carries exactly these three values
  (`filename`/`url`/`sha256`). Partially pinned offline by
  `metadata_matches_sklearn` (filename).
- **AC-ARCHIVE-10PCT-1 (REQ-ARCHIVE-10PCT-CHECKSUM, offline-detectable
  divergence):** the sklearn source constant
  ```
  python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE_10_PERCENT.checksum)"
  ```
  → `8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561`.
  ferrolearn's `ARCHIVE_10PCT.sha256` is
  `8045aca0d84e70e622d1148d7df782496f6333bf6eb35a805a2cb4ddb1ec1422` — these
  **differ** (common 43-char prefix, then diverge). The AC is RED until the
  const is corrected; verifiable with NO network (string equality against the
  sklearn source). Blocker #2070.
- AC-PARSE-1 (REQ-CSV-PARSE, shape): parsing two synthetic rows of 41 features
  (numeric `0`, with `"tcp"`/`"http"`/`"SF"` at cols 1/2/3) + a label column
  yields `data.nrows() == 2`, `data.ncols() == 41`. (Offline; in-crate
  `parse_kddcup99_csv`, pinned by `parser_two_rows`.)
- AC-PARSE-2 (REQ-CSV-PARSE, column-count error): a line with the wrong number
  of columns (`"1,2,3,smurf."`, 4 columns ≠ 42) is rejected with an `Err`.
  (Offline; pinned by `parser_rejects_wrong_column_count`.) sklearn likewise
  treats the file as a fixed 42-wide table (`for j in range(42)`, `:396`).
- AC-CATEGORICAL-1 (REQ-CATEGORICAL-ENCODING): after parsing, `categorical_columns
  == [1, 2, 3]` and each of `categorical_levels[0..3]` holds the distinct
  symbolic values in first-appearance order (two identical rows → one level
  each). The encoded `data` at those columns holds the f64 level index. Matches
  sklearn's symbolic columns at positions 1/2/3 (`:322-324`). (Offline; pinned
  by `parser_two_rows`.)
- AC-TARGET-1 (REQ-TARGET-LABELS): for rows ending in `normal.` / `smurf.`, the
  parsed `target == ["normal.", "smurf."]` — the last column is preserved as a
  string, matching sklearn `y = Xy[:, -1]` (`:400`). (Offline; pinned by
  `parser_two_rows`.)
- AC-SUBSET-1 (REQ-SUBSET-VARIANT): `fetch_kddcup99` selects `ARCHIVE_FULL` for
  `KddSubset::Full` and `ARCHIVE_10PCT` for `KddSubset::Percent10` (the `match
  subset { … }` in `fetch_kddcup99`), mirroring sklearn's `archive =
  ARCHIVE_10_PERCENT if percent10 else ARCHIVE` (`:309-314`). (Offline; the
  branch mapping is source-readable.)
- AC-CACHE-1 (REQ-CACHE-INTEGRATION): `fetch_kddcup99(Some(home), …)` resolves
  its cache directory to `home/kddcup99` (via `dataset_dir("kddcup99",
  data_home)`) and calls `fetch_file(archive.url, archive.filename,
  Some(archive.sha256), &dir)` — first-use, SHA-verified download. (Offline up
  to the network call; the dir-creation + SHA-gating wiring is the
  offline-checkable part.)
- AC-VALUE-1 (REQ-VALUE-PARITY, NETWORK-GATED oracle): live
  ```
  python3 -c "from sklearn.datasets import fetch_kddcup99; b=fetch_kddcup99(percent10=True); print(b.data.shape, b.target.shape, b.target[0])"
  ```
  → `(494021, 41) (494021,) b'normal.'` versus `fetch_kddcup99(None,
  KddSubset::Percent10)` element by element (note: encoding of the symbolic
  columns differs — ferrolearn level-indexes them, sklearn keeps byte strings —
  so parity is defined on the numeric columns + the string `target`). Requires
  network + the full archive download — NOT runnable offline; pinned under
  network-gated blocker #2071.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-ARCHIVE-FULL-METADATA (full figshare URL/SHA/filename) | SHIPPED | `pub const ARCHIVE_FULL: RemoteFile in kddcup99.rs` = `{ filename: "kddcup99_data", url: "https://ndownloader.figshare.com/files/5976045", sha256: "3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292" }`, byte-for-byte equal to sklearn source `ARCHIVE = RemoteFileMetadata(filename="kddcup99_data", url="https://ndownloader.figshare.com/files/5976045", checksum="3b6c942a…274292")` (`_kddcup99.py:34-38`). Characterization is exact (R-CHAR-3): expected values read from the sklearn SOURCE constant, not from ferrolearn. Non-test consumer: `pub fn fetch_kddcup99 in kddcup99.rs` reads all three fields for `KddSubset::Full` (`fetch_file(archive.url, archive.filename, Some(archive.sha256), &dir)`); `ARCHIVE_FULL` is `pub` and reachable via `pub mod kddcup99` in `lib.rs`. Verification: live `python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE.checksum)"` → `3b6c942a…274292` matches; `cargo test -p ferrolearn-fetch --lib kddcup99` → `metadata_matches_sklearn` passes the filename assertion. |
| **REQ-ARCHIVE-10PCT-CHECKSUM (10% subset SHA-256)** | **NOT-STARTED** | **open prereq blocker #2070 — REAL, offline-detectable, single-const-fixable bug.** `pub const ARCHIVE_10PCT.sha256 in kddcup99.rs` is `"8045aca0d84e70e622d1148d7df782496f6333bf6eb35a805a2cb4ddb1ec1422"`, but sklearn source `ARCHIVE_10_PERCENT.checksum = "8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561"` (`_kddcup99.py:45`). They share the 43-char prefix `8045aca0d84e70e622d1148d7df782496f6333bf6eb` then diverge (`35a805a2cb4ddb1ec1422` vs `979a1b0837c42a9fd9561`). Consequence: `fetch_file` (`fetch.rs`) verifies the download against `archive.sha256` and, on mismatch, wipes the file and returns `FerroError::SerdeError` — so ferrolearn would **reject the correct sklearn 10% archive** (or accept a wrong file). Live oracle: `python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE_10_PERCENT.checksum)"` → the `…fd9561` value. The critic pins this RED via a string-equality `#[test]` (no network needed); the fixer corrects the single const to the sklearn value. (`filename`/`url` of `ARCHIVE_10PCT` are correct: `kddcup99_10_data` / `…/5976042`, matching `:43-44`.) |
| REQ-CSV-PARSE (42-col parse + shape + errors) | SHIPPED | `fn parse_kddcup99_csv in kddcup99.rs` splits each non-empty trimmed line on `,`, requires exactly `N_COLS = 42` columns (`if parts.len() != N_COLS { return Err(FerroError::SerdeError { … "expected {N_COLS}" }) }`), parses each non-categorical field via `raw.parse::<f64>()` (erroring with a col-located `SerdeError` on non-float), then materializes `data = Array2::<f64>::zeros((n, N_COLS - 1))` (i.e. `(n, 41)`) and `target` length `n` from the last column. Mirrors sklearn's fixed-width 42-column cast `for j in range(42): Xy[:, j] = Xy[:, j].astype(DT[j])` (`_kddcup99.py:396-397`) → `X = Xy[:, :-1]` (`:399`). Non-test consumer: called by `pub fn fetch_kddcup99 in kddcup99.rs` (`parse_kddcup99_csv(&text)` after `GzDecoder` decompress). Verification: `cargo test -p ferrolearn-fetch --lib kddcup99` → `parser_two_rows` (asserts `nrows()==2`, `ncols()==41`) and `parser_rejects_wrong_column_count` (4-col line → `Err`) pass (3 passed, 0 failed). |
| REQ-CATEGORICAL-ENCODING (cols 1/2/3 → level indices) | SHIPPED | `fn parse_kddcup99_csv in kddcup99.rs` declares `const CATEGORICAL: [usize; 3] = [1, 2, 3]`; for each such column it looks up (or appends) the raw string in `levels[cat_idx]` and pushes the level index `pos as f64` into the row. The result struct sets `categorical_columns: CATEGORICAL.to_vec()` and `categorical_levels: levels`. This mirrors sklearn's symbolic byte-string columns at exactly positions 1/2/3: `("protocol_type","S4")` (`_kddcup99.py:322`), `("service","S11")` (`:323`), `("flag","S6")` (`:324`) — confirmed against the full dtype list where `duration` is pos 0 and `src_bytes` is pos 4. Non-test consumer: `pub fn fetch_kddcup99 in kddcup99.rs` returns this struct. Verification: `cargo test -p ferrolearn-fetch --lib kddcup99` → `parser_two_rows` asserts `categorical_columns == vec![1,2,3]` and `categorical_levels[0].len() == 1` after two identical rows. FAITHFUL DELTA (recorded under REQ-VALUE-PARITY): ferrolearn level-indexes these columns into the numeric matrix; sklearn keeps them as byte strings (`S4`/`S11`/`S6`) — same column SELECTION, different in-memory encoding. |
| REQ-TARGET-LABELS (last column as string) | SHIPPED | `fn parse_kddcup99_csv in kddcup99.rs` pushes `parts[N_COLS - 1].to_string()` into `labels` and returns it as `target: Vec<String>`; `pub struct KddCup99` documents `target` as "String label per row (e.g. \"normal.\", \"smurf.\")". Mirrors sklearn `y = Xy[:, -1]` where the labels column has dtype `("labels","S16")` (`_kddcup99.py:362`, `:400`) — the last column kept as a (byte) string. Non-test consumer: `pub fn fetch_kddcup99 in kddcup99.rs` returns this `target`. Verification: `cargo test -p ferrolearn-fetch --lib kddcup99` → `parser_two_rows` asserts `ds.target == vec!["normal.", "smurf."]`. DELTA: ferrolearn `String` (UTF-8) vs sklearn `bytes` (`b"normal."`) — representational, folded into REQ-VALUE-PARITY. |
| REQ-SUBSET-VARIANT (Full/Percent10 → two archives) | SHIPPED | `pub enum KddSubset { Full, Percent10 } in kddcup99.rs`; `pub fn fetch_kddcup99 in kddcup99.rs` selects `let archive = match subset { KddSubset::Full => ARCHIVE_FULL, KddSubset::Percent10 => ARCHIVE_10PCT };`. Mirrors sklearn's `archive = ARCHIVE_10_PERCENT if percent10 else ARCHIVE` (`_kddcup99.py:309-314`), with sklearn's `percent10=True` default (`:72`) → the 10% archive. Non-test consumer: `pub fn fetch_kddcup99` (the boundary entry point) consumes the enum. Verification: source-readable branch mapping; `cargo test -p ferrolearn-fetch --lib kddcup99` exercises the enum-bearing module (compiles + 3 tests pass). DELTA: ferrolearn has no default — the caller must pass a `KddSubset`; sklearn defaults to `percent10=True` (folded into REQ-FULL-PARAMS). |
| REQ-CACHE-INTEGRATION (dataset_dir + SHA-verified fetch) | SHIPPED | `pub fn fetch_kddcup99 in kddcup99.rs` calls `let dir = dataset_dir("kddcup99", data_home)?;` (`fn dataset_dir in cache.rs`) then `let path = fetch_file(archive.url, archive.filename, Some(archive.sha256), &dir)?;` (`pub fn fetch_file in fetch.rs`) — first-use download keyed by a dataset-scoped subdir with SHA-256 verification (`Some(archive.sha256)` → `verify_sha256` in `fetch.rs`). Mirrors sklearn's `kddcup_dir = join(data_home, "kddcup99_10"+dir_suffix)`/`"kddcup99"+dir_suffix` + `_fetch_remote(archive, dirname=kddcup_dir, n_retries=…, delay=…)` checksum-verified download (`_kddcup99.py:309-314`, `:382`). Non-test consumer: `fetch_kddcup99` itself (boundary API); `dataset_dir`/`fetch_file` are the shared crate primitives. Verification: `cargo test -p ferrolearn-fetch --lib fetch` (the `verify_sha256` SHA primitive REQ-CACHE-INTEGRATION rides on passes); the SHA-gated download runs only with network (#2071). FAITHFUL DELTA: ferrolearn's subdir is `kddcup99` for BOTH variants (the two filenames `kddcup99_data`/`kddcup99_10_data` keep them distinct); sklearn splits into `kddcup99-py3`/`kddcup99_10-py3` and persists `joblib.dump` pickles rather than the raw `.gz` — cache LOCATION/format delta, folded into REQ-VALUE-PARITY. |
| REQ-CONSUMER (public boundary surface) | SHIPPED | `pub fn fetch_kddcup99` + `pub struct KddCup99` + `pub enum KddSubset` (+ `pub const ARCHIVE_FULL`/`ARCHIVE_10PCT`) are re-exported at the crate root: `pub use kddcup99::{KddCup99, KddSubset, fetch_kddcup99}` in `lib.rs`, documented in the crate `//!` header (`[fetch_kddcup99] — KDD Cup 1999 intrusion-detection dataset.`, `lib.rs:15`). Existing pub boundary API, grandfathered under R-DEFER-1 / S5 (the fetcher fn + struct ARE the public API; external users + the eventual `ferrolearn-python` `datasets` binding mirroring `from sklearn.datasets import fetch_kddcup99` are the consumers). Verification: `grep -rn "fetch_kddcup99\|KddCup99\|KddSubset" ferrolearn-fetch/src/ | grep -v 'kddcup99.rs'` → the `lib.rs` re-export (`pub use kddcup99::{…}`) + the crate-doc line. UNDERCLAIM: there is NO in-workspace non-test FUNCTION caller; this rests on the boundary-API grandfather clause + the crate re-export. |
| REQ-VALUE-PARITY (element-wise (data,target) vs oracle) | NOT-STARTED | open prereq blocker #2071 (network-gated, NOT a code bug). No offline path: `pub fn fetch_kddcup99 in kddcup99.rs` calls `fetch_file` → the crate's `download` primitive (network GET) which requires network access + downloading the figshare archive to produce `(data, target)`. Element-by-element parity vs `sklearn.datasets.fetch_kddcup99()` cannot be established offline. The encoding/dtype/cache-format deltas noted in the SHIPPED rows (level-indexed symbolic cols vs sklearn byte strings; `String` vs `bytes` target; subdir/pickle format) resolve here once network parity can be run. NOTE: this REQ is ALSO blocked by REQ-ARCHIVE-10PCT-CHECKSUM (#2070) for the 10% subset — with the wrong SHA, the 10% download is rejected; correcting the const is a prerequisite for 10% value-parity. |
| REQ-FULL-PARAMS (omitted keyword-only params) | NOT-STARTED | open prereq blocker #2072 (feature gap). `pub fn fetch_kddcup99 in kddcup99.rs` has signature `fetch_kddcup99(data_home: Option<&Path>, subset: KddSubset)` — only `data_home` + the archive-selector enum. sklearn `def fetch_kddcup99(*, subset=None, data_home=None, shuffle=False, random_state=None, percent10=True, download_if_missing=True, return_X_y=False, as_frame=False, n_retries=3, delay=1.0)` (`_kddcup99.py:66-78`). Omitted: `subset` (the SA/SF/http/smtp ATTACK-TYPE filters + log-transforms, `:187-238` — note sklearn's `subset` is an attack-filter string, NOT ferrolearn's archive selector), `shuffle`+`random_state` (the `shuffle_method(data, target, …)` reorder + SA abnormal-sample sampling, `:197-240`), `download_if_missing` (the offline-OSError gate, `:407-408`), `return_X_y` (the `(X, y)` tuple, `:250-251`), `as_frame` (DataFrame path, `:244-248`), `n_retries`/`delay` (retry policy, `:382`). |
| REQ-SUBSTRATE (ferray-core array types) | NOT-STARTED | open prereq blocker #2073. `kddcup99.rs` uses `use ndarray::Array2` and `pub struct KddCup99 { data: Array2<f64>, … }`. R-SUBSTRATE-1 requires `ferray-core` array types, not `ndarray`. Prerequisite: migrate the unit's array types to the ferray analog; folded into this unit's substrate work and gated on the wider `ferrolearn-fetch` migration (shares the `ndarray` types of the sibling fetchers). Representational (no scalar oracle). |

## Architecture

`kddcup99.rs` is a single file with two public constants, one public return
struct, one public variant enum, one public entry point, and one private parser:

- `pub const ARCHIVE_FULL: RemoteFile` / `pub const ARCHIVE_10PCT: RemoteFile`
  — the two figshare remote-file descriptors (`filename`/`url`/`sha256`).
  `ARCHIVE_FULL` is a verbatim copy of sklearn's source `ARCHIVE`
  (`_kddcup99.py:34-38`, REQ-ARCHIVE-FULL-METADATA). `ARCHIVE_10PCT` copies
  sklearn's `ARCHIVE_10_PERCENT` filename + url correctly but its `sha256`
  DIVERGES from the source `checksum` (`:45`) — the headline bug
  (REQ-ARCHIVE-10PCT-CHECKSUM, #2070). Each constant is the single source of a
  download URL and the SHA-256 the cache layer verifies against.
- `pub struct KddCup99` — the return type (a `Bunch`-analog): `data:
  Array2<f64>` (the 41-feature matrix, with the three symbolic columns encoded
  as f64 level-indices), `target: Vec<String>` (the per-row attack label),
  `categorical_levels: Vec<Vec<String>>` (the distinct symbolic values, in
  first-appearance order, for each of the 3 categorical columns), and
  `categorical_columns: Vec<usize>` (`[1, 2, 3]`). This is a hand-rolled struct,
  not sklearn's `Bunch` and not the `(X, y)`/`return_X_y` pair
  (REQ-FULL-PARAMS); the array is on the `ndarray` substrate (REQ-SUBSTRATE).
  The `categorical_levels`/`categorical_columns` fields are ferrolearn's
  numeric-encoding bookkeeping; sklearn instead keeps the symbolic columns as
  byte strings (`S4`/`S11`/`S6`) and exposes `feature_names`/`target_names` it
  does not (the encoding delta is recorded under REQ-VALUE-PARITY).
- `pub enum KddSubset { Full, Percent10 }` — the archive selector
  (REQ-SUBSET-VARIANT), mapping to `ARCHIVE_FULL`/`ARCHIVE_10PCT`. It mirrors
  sklearn's `percent10` bool but is a required argument (no default), unlike
  sklearn's `percent10=True`.
- `pub fn fetch_kddcup99(data_home: Option<&Path>, subset: KddSubset) ->
  Result<KddCup99, FerroError>` — the entry point. It (1) selects the archive by
  `match subset` (REQ-SUBSET-VARIANT), (2) creates the cache dir via
  `dataset_dir("kddcup99", data_home)` (REQ-CACHE-INTEGRATION), (3) downloads +
  SHA-verifies via `fetch_file(archive.url, archive.filename,
  Some(archive.sha256), &dir)` (mirroring `_fetch_remote(archive, …)`, `:382`),
  (4) reads the bytes and decompresses with `GzDecoder` (mirroring
  `GzipFile(filename=archive_path)`, `:386`), and (5) dispatches the
  decompressed text to `parse_kddcup99_csv`. All failures return `Result<_,
  FerroError>` (`SerdeError` for parse, `IoError` for filesystem/gunzip) — no
  panics in the production path (R-CODE-2).
- `fn parse_kddcup99_csv(raw: &str) -> Result<KddCup99, FerroError>` — the CSV
  reader (REQ-CSV-PARSE / REQ-CATEGORICAL-ENCODING / REQ-TARGET-LABELS). It
  iterates lines (skipping blanks), splits each on `,`, enforces exactly 42
  columns, encodes columns `[1,2,3]` as first-appearance level-indices and
  parses every other feature column as `f64`, then fills `data` from columns
  `0..41` and `target` from column `41` (kept as `String`) — the analog of
  sklearn's per-column 42-wide cast (`:396-397`) + `X = Xy[:, :-1]` /
  `y = Xy[:, -1]` split (`:399-400`).

Invariant: the feature matrix is exactly the first 41 columns; the label column
is the 42nd, preserved as a string; columns 1/2/3 are categorical and the rest
numeric — matching sklearn's `Xy[:, :-1]`/`Xy[:, -1]` partition and its `S4`/
`S11`/`S6` symbolic-column positions. The structural limits — download-only (no
`download_if_missing=False` offline gate), no `subset` attack-filter, no
shuffle, no `(X, y)`/DataFrame return, no retry policy, `String` target,
`ndarray` substrate, AND the wrong 10% checksum — are the surface of
REQ-ARCHIVE-10PCT-CHECKSUM / REQ-FULL-PARAMS / REQ-VALUE-PARITY / REQ-SUBSTRATE
and are recorded as NOT-STARTED, not as parity claims.

## Verification

Commands establishing the SHIPPED claims and the deltas behind the NOT-STARTED
rows (sklearn 1.5.2 oracle):

- `cargo test -p ferrolearn-fetch --lib kddcup99` → 3 passed, 0 failed
  (`parser_two_rows`, `parser_rejects_wrong_column_count`,
  `metadata_matches_sklearn`). Establishes REQ-CSV-PARSE (AC-PARSE-1/2),
  REQ-CATEGORICAL-ENCODING (AC-CATEGORICAL-1), REQ-TARGET-LABELS (AC-TARGET-1),
  and the filename half of REQ-ARCHIVE-FULL-METADATA, all offline.
- ARCHIVE_FULL oracle (REQ-ARCHIVE-FULL-METADATA, exact characterization from
  sklearn source — R-CHAR-3):
  `python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE.filename, k.ARCHIVE.url, k.ARCHIVE.checksum)"`
  → `kddcup99_data https://ndownloader.figshare.com/files/5976045 3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292`,
  byte-for-byte equal to ferrolearn's three `ARCHIVE_FULL` constants. No
  divergence.
- **10% checksum divergence (REQ-ARCHIVE-10PCT-CHECKSUM, #2070) — offline:**
  `python3 -c "from sklearn.datasets import _kddcup99 as k; print(k.ARCHIVE_10_PERCENT.checksum)"`
  → `8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561` vs
  ferrolearn `ARCHIVE_10PCT.sha256 = …eb35a805a2cb4ddb1ec1422`. These DIFFER
  (43-char common prefix). The critic pins a string-equality `#[test]` RED (no
  network); the fixer corrects the one const.
- Categorical-position cite (REQ-CATEGORICAL-ENCODING): sklearn `_kddcup99.py`
  dtype list — `("protocol_type","S4")` (`:322`), `("service","S11")` (`:323`),
  `("flag","S6")` (`:324`) at positions 1/2/3 (pos 0 = `duration`, pos 4 =
  `src_bytes`). ferrolearn `CATEGORICAL = [1, 2, 3]`.
- Target/split cite (REQ-TARGET-LABELS): sklearn `X = Xy[:, :-1]` /
  `y = Xy[:, -1]` (`:399-400`); labels dtype `("labels","S16")` (`:362`).
  ferrolearn `labels.push(parts[N_COLS - 1].to_string())`.
- Cache-wiring (REQ-CACHE-INTEGRATION): `cargo test -p ferrolearn-fetch --lib
  fetch` (the `verify_sha256` SHA primitive passes); the SHA-gated `fetch_file`
  download runs only with network (#2071).
- Consumer check (REQ-CONSUMER):
  `grep -rn "fetch_kddcup99\|KddCup99\|KddSubset" ferrolearn-fetch/src/ | grep -v 'kddcup99.rs'`
  → the `lib.rs` re-export (`pub use kddcup99::{KddCup99, KddSubset, fetch_kddcup99}`)
  + the crate-doc line. No non-test function caller in-workspace (boundary-API
  grandfather).
- Hygiene: `cargo clippy -p ferrolearn-fetch --lib -- -D warnings`.
- Value-parity oracle (REQ-VALUE-PARITY, NETWORK-GATED — NOT runnable offline,
  #2071): live
  `python3 -c "from sklearn.datasets import fetch_kddcup99; b=fetch_kddcup99(percent10=True); print(b.data.shape, b.target.shape, b.target[0])"`
  → `(494021, 41) (494021,) b'normal.'` versus `fetch_kddcup99(None,
  KddSubset::Percent10)`. Requires network + the full archive download, so this
  AC stays NOT-STARTED offline (and the 10% path is additionally gated on the
  #2070 SHA fix).
- Params cite (REQ-FULL-PARAMS, #2072): sklearn `_kddcup99.py:66-78` enumerates
  `subset`/`data_home`/`shuffle`/`random_state`/`percent10`/`download_if_missing`
  /`return_X_y`/`as_frame`/`n_retries`/`delay`; ferrolearn's signature exposes
  only `(data_home, subset: KddSubset)`.

NOT-STARTED REQs map to open blockers: **REQ-ARCHIVE-10PCT-CHECKSUM → #2070 (the
headline, offline-detectable, single-const fix)**, REQ-VALUE-PARITY → #2071
(network-gated), REQ-FULL-PARAMS → #2072 (feature gap), REQ-SUBSTRATE → #2073
(ferray migration; representational). Per R-CHAR-3, all expected values above
derive from sklearn source `file:line` constants or live sklearn oracle calls,
never copied from the ferrolearn side. The verified offline contract
(ARCHIVE_FULL constants, shape, categorical positions, label, subset mapping,
cache wiring, public surface) exhibits NO code divergence this iteration — except
the one real bug headlined as REQ-ARCHIVE-10PCT-CHECKSUM.
