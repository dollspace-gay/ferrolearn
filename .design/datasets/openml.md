# fetch_openml — generic OpenML.org dataset client

<!--
tier: 3-component
status: draft
baseline-commit: 86bd54f03887e679197a7e3f08fea344a32a1bce
upstream-paths:
  - sklearn/datasets/_openml.py
  - sklearn/datasets/_arff_parser.py
-->

## Summary

`ferrolearn-fetch/src/openml.rs` mirrors scikit-learn's `sklearn.datasets.fetch_openml`
(`sklearn/datasets/_openml.py`): a network client that looks up an OpenML dataset
by numeric `data_id`, downloads its data file, and parses it into a feature
matrix + target vector. The current implementation covers the common path only —
the JSON metadata lookup, the on-disk cache integration, and a hand-rolled ARFF
parser for the numeric + nominal attribute subset — returning a dedicated
`OpenmlDataset` struct rather than sklearn's `Bunch`/`(X, y)`.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here. Because `fetch_openml` is a NETWORK fetcher, full numerical value-parity
against the live `sklearn.datasets.fetch_openml(...)` oracle requires network
access + downloading OpenML datasets, which is OUT OF REACH offline — those REQs
are NOT-STARTED with a concrete network/offline blocker. The SHIPPED REQs are the
OFFLINE-VERIFIABLE contract pieces: the OpenML API URL structure, the ARFF parser
logic (exercised on in-crate ARFF text fixtures, no network), the dataset-dir
cache integration, and the public-fn surface.

## Upstream cites (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_openml.py`:
  - `def fetch_openml(...)` (`:770-784`) — the public signature:
    `name`, `*`, `version="active"`, `data_id=None`, `data_home=None`,
    `target_column="default-target"`, `cache=True`, `return_X_y=False`,
    `as_frame="auto"`, `n_retries=3`, `delay=1.0`, `parser="auto"`,
    `read_csv_kwargs=None`.
  - `_OPENML_PREFIX = "https://api.openml.org/"` (`:32`).
  - `_DATA_INFO = "api/v1/json/data/{}"` (`:34`), `_DATA_FEATURES =
    "api/v1/json/data/features/{}"` (`:35`), `_DATA_FILE =
    "data/v1/download/{}"` (`:37`).
  - `_get_data_description_by_id` returns `json_data["data_set_description"]`
    (`:358`, `:367`); the data-file URL is `_DATA_FILE.format(
    data_description["file_id"])` (`:1126`).
  - `_get_local_path` caches under `os.path.join(data_home, "openml.org",
    openml_path + ".gz")` (`:43-44`).
- `sklearn/datasets/_arff_parser.py`:
  - `_liac_arff_parser` (`:104`) — the LIAC-ARFF path that drives nominal-vs-numeric
    attribute handling; nominal attributes are encoded to numerical indices via
    `encode_nominal` (`:167-172`), and `categories` is built from the attribute
    list (`:175-179`).
  - `_split_sparse_columns` (`:19`), `_post_process_frame` (`:72`),
    `_pandas_arff_parser` (`:308`), `load_arff_from_gzip_file` (`:460`).

## Requirements

- REQ-OPENML-API-ENDPOINTS: the OpenML REST URL ferrolearn builds for the
  dataset description matches sklearn's `_OPENML_PREFIX` + `_DATA_INFO` endpoint
  (`api/v1/json/data/{id}`), and reads the metadata out of the
  `data_set_description` JSON object (mirroring `_get_data_description_by_id`).
- REQ-ARFF-PARSE: the ARFF attribute parsing — numeric (`numeric`/`real`/
  `integer`) vs nominal (`{a,b,c}`) attribute kinds, nominal-level encoding to
  0-based indices, and quoted attribute-name handling via `split_attribute_name`
  — matches the ARFF format sklearn's LIAC-ARFF path consumes. Verifiable on
  in-crate ARFF text fixtures (no network).
- REQ-CACHE-INTEGRATION: the fetcher caches under
  `dataset_dir("openml/{id}", data_home)`, mirroring sklearn's `cache=True`
  `data_home` caching (`_get_local_path`, `data_home/openml.org/...`).
- REQ-CONSUMER: `fetch_openml` / `OpenmlDataset` are public crate API
  (re-exported in `lib.rs`); grandfathered boundary API (R-DEFER-1 / S5).
- REQ-CODE-HYGIENE: the module compiles clean under
  `cargo clippy -p ferrolearn-fetch --lib -- -D warnings` on the current
  toolchain (rust 1.95.0).
- REQ-VALUE-PARITY: the fetched `(data, target)` arrays match
  `sklearn.datasets.fetch_openml(data_id=..., as_frame=False)` element-by-element
  on a real OpenML dataset (e.g. data_id 61, iris).
- REQ-FULL-PARAMS: the sklearn `fetch_openml` parameters ferrolearn currently
  omits — `name`/`version` resolution, list/`None` `target_column`, `cache`,
  `return_X_y`, `as_frame`, `n_retries`/`delay`, `parser`, `read_csv_kwargs`.
- REQ-SUBSTRATE: array types are the ferray substrate (`ferray-core`), not
  `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

- AC-ENDPOINT-1 (REQ-OPENML-API-ENDPOINTS): the description-lookup URL for
  `data_id = 61` is `https://www.openml.org/api/v1/json/data/61` and the metadata
  is read from `data_set_description.url` / `.default_target_attribute`. The path
  segment `api/v1/json/data/{id}` is identical to sklearn's `_DATA_INFO`
  (only the host prefix differs — see the REQ row).
- AC-ARFF-1 (REQ-ARFF-PARSE): parsing
  `@ATTRIBUTE x numeric / @ATTRIBUTE y numeric / @ATTRIBUTE label numeric / @DATA / 1.0,2.0,0 / 3.5,4.5,1`
  with target `label` yields `data` shape `(2, 2)`, `data[[0,0]] == 1.0`,
  `data[[1,1]] == 4.5`, `target == [0, 1]`, `feature_names == ["x","y"]`.
  (Offline; in-crate `parse_minimal_numeric_arff`.)
- AC-ARFF-2 (REQ-ARFF-PARSE, nominal): a nominal target `{cat,dog,bird}` is
  encoded to 0-based level indices in declaration order
  (`cat→0, dog→1, bird→2`), with `target_levels == ["cat","dog","bird"]`.
  (Offline; in-crate `parse_arff_with_nominal_target`.)
- AC-ARFF-3 (REQ-ARFF-PARSE, quoting): a single-quoted attribute name
  (`@attribute 'my attr' numeric`) is unquoted to `my attr` by
  `split_attribute_name`. (Offline; exercised through `parse_arff`.)
- AC-CACHE-1 (REQ-CACHE-INTEGRATION): `fetch_openml(61, .., Some(home))` resolves
  its cache directory to `home/openml/61` and writes `data_info.json` +
  `data.arff` there. (Offline up to the directory-creation call.)
- AC-HYGIENE-1 (REQ-CODE-HYGIENE): `cargo clippy -p ferrolearn-fetch --lib --
  -D warnings` exits 0.
- AC-VALUE-1 (REQ-VALUE-PARITY, NETWORK-GATED oracle): live
  `python3 -c "from sklearn.datasets import fetch_openml; b=fetch_openml(data_id=61, as_frame=False); print(b.data[:2].tolist(), b.target[:2].tolist())"`
  versus `fetch_openml(61, Some("class"), ..)`. Requires network — NOT runnable
  offline; pinned under blocker #2060.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-OPENML-API-ENDPOINTS | SHIPPED | `const DATA_INFO_URL = "https://www.openml.org/api/v1/json/data/"` in `openml.rs` + `pub fn fetch_openml in openml.rs` builds `{DATA_INFO_URL}{data_id}` and reads `info.get("data_set_description")` then `.get("url")` / `.get("default_target_attribute")`. The path segment `api/v1/json/data/{id}` and the `data_set_description` JSON key match sklearn `_DATA_INFO = "api/v1/json/data/{}"` (`_openml.py:34`) + `_get_data_description_by_id` (`:367` `return json_data["data_set_description"]`). Non-test consumer: re-exported at crate root in `lib.rs` (`pub use openml::{OpenmlDataset, fetch_openml}`) — public boundary API. Verification: `cargo test -p ferrolearn-fetch --lib openml` (4 passed). FAITHFUL DELTA (not claimed as parity): ferrolearn uses host `www.openml.org`, sklearn `_OPENML_PREFIX = "https://api.openml.org/"` (`:32`), and resolves the data file from `data_set_description.url` rather than `_DATA_FILE = "data/v1/download/{file_id}"` (`:37`, `:1126`) — host/download-route parity is folded into REQ-VALUE-PARITY (network-gated, #2060). |
| REQ-ARFF-PARSE | SHIPPED | `fn parse_arff in openml.rs` (with nested `parse_attribute` / `split_attribute_name`) classifies attributes into `AttrKind::Numeric` (`numeric`/`real`/`integer`, default fallback) vs `AttrKind::Nominal(levels)` for `{a,b,c}`, encodes nominal values to 0-based level indices (`levels.iter().position(...)`), and unquotes single-quoted names. This mirrors sklearn's LIAC-ARFF path: nominal attributes encoded numerically via `encode_nominal` (`_arff_parser.py:167-172`) and categories drawn from the attribute list (`:175-179`). Offline-verifiable. Non-test consumer: reached via `fetch_openml` (re-exported in `lib.rs`). Verification: `cargo test -p ferrolearn-fetch --lib openml` — `parse_minimal_numeric_arff`, `parse_arff_with_nominal_target`, `parse_arff_missing_target_errors`, `parse_arff_wrong_field_count_errors` (4 passed, 0 failed). Hygiene sub-item: see REQ-CODE-HYGIENE for the `split_attribute_name` clippy lint. SCOPE NOTE: this is the simple-but-common ARFF subset only — sparse ARFF (`_split_sparse_columns`, `_arff_parser.py:19`), the pandas parser path (`_pandas_arff_parser`, `:308`), and string/date attribute kinds are NOT covered (string/date fall back to numeric and will produce NaN). |
| REQ-CACHE-INTEGRATION | SHIPPED | `pub fn fetch_openml in openml.rs` calls `dataset_dir(&format!("openml/{data_id}"), data_home)` (`fn dataset_dir in cache.rs`), then `fetch_file(.., "data_info.json", None, &dir)` and `fetch_file(url, "data.arff", None, &dir)` (`fn fetch_file in fetch.rs`) — first-use download, on-disk cache keyed by `data_id`. Mirrors sklearn's `cache=True` `data_home` caching: `_get_local_path` → `os.path.join(data_home, "openml.org", openml_path + ".gz")` (`_openml.py:43-44`). Non-test consumer: re-exported in `lib.rs`. Verification: `cargo test -p ferrolearn-fetch --lib cache` (`dataset_dir_creates_subdir` passes; cache.rs is the shared primitive). FAITHFUL DELTA: ferrolearn caches uncompressed under `openml/{id}/`, sklearn gzips under `openml.org/{path}.gz` — the cache LOCATION/format differs (folded into REQ-VALUE-PARITY scope). |
| REQ-CONSUMER | SHIPPED | `pub fn fetch_openml` + `pub struct OpenmlDataset` are re-exported at the crate root: `pub use openml::{OpenmlDataset, fetch_openml}` in `lib.rs`, and documented in the crate `//!` header (`[fetch_openml] — generic OpenML.org client`). Existing pub boundary API, grandfathered under R-DEFER-1 / S5 (the fetcher type IS the public API; external users + the eventual `ferrolearn-python` `datasets` binding are the consumers). `fetch_openml` is additionally cited as a real consumer of `FerroError::IoError`/`SerdeError` in `ferrolearn-core/src/error.rs`. Verification: `grep -rn "fetch_openml\|OpenmlDataset" --include=*.rs \| grep -v 'src/openml.rs'` → re-export in `lib.rs` + a `#[from]`-consumer note in `error.rs` (the only `.rs` caller in `tests/api_proof.rs` is a test, which does NOT count). UNDERCLAIM: there is NO in-workspace non-test FUNCTION caller; this rests entirely on the boundary-API grandfather clause + crate re-export. |
| REQ-CODE-HYGIENE | NOT-STARTED | `fn split_attribute_name in openml.rs` is a nested `if let Some(stripped) = body.strip_prefix('\'') { if let Some(end) = stripped.find('\'') { .. } }`, which rust 1.95.0 clippy flags as `collapsible_if` under `-D warnings` (`-D clippy::collapsible-if` implied by `-D warnings`). `cargo clippy -p ferrolearn-fetch --lib -- -D warnings` currently FAILS on this site. Open prereq blocker #2062 (R-CODE / R-DEFER-5 toolchain drift). Fix is mechanical: collapse the two `if let`s into a single let-chain (`if let Some(stripped) = .. && let Some(end) = ..`) — the orchestrator will land that single in-iteration change; no behavior change. THIS IS THE ONE CODE CHANGE THIS ITERATION. |
| REQ-VALUE-PARITY | NOT-STARTED | No offline path: `pub fn fetch_openml in openml.rs` calls `fetch_file` → `download` (`fn download in fetch.rs`, `ureq::get(url).call()`), which requires network access + a live OpenML download to produce `(data, target)`. Element-by-element parity vs `sklearn.datasets.fetch_openml(data_id=61, as_frame=False)` cannot be established offline (confirmed: a live `fetch_openml(data_id=61)` call hangs on network with no connectivity). Open prereq blocker #2060 — NETWORK-GATED, NOT a code bug. The parser/encoding ordering and the host/download-route deltas noted in the SHIPPED rows resolve here once network parity can be run. |
| REQ-FULL-PARAMS | NOT-STARTED | `pub fn fetch_openml in openml.rs` has signature `fetch_openml(data_id: u64, target_column: Option<&str>, data_home: Option<&Path>)` — it requires a numeric `data_id` and accepts only a single scalar `target_column`. sklearn `def fetch_openml(name=None, *, version="active", data_id=None, ..., target_column="default-target", cache=True, return_X_y=False, as_frame="auto", n_retries=3, delay=1.0, parser="auto", read_csv_kwargs=None)` (`_openml.py:770-784`). Omitted: `name`/`version` resolution (search via `_SEARCH_NAME`), list/`None` `target_column` (multi-target / no-target), `cache`, `return_X_y`, `as_frame`, `n_retries`/`delay`, `parser`, `read_csv_kwargs`. Open prereq blocker #2061 (feature gap). |
| REQ-SUBSTRATE | NOT-STARTED | `openml.rs` uses `use ndarray::{Array1, Array2}` and `OpenmlDataset { data: Array2<f64>, target: Array1<f64>, .. }`. R-SUBSTRATE-1 requires `ferray-core` array types, not `ndarray`. Prerequisite: migrate the unit's array types to the ferray analog; folded into this unit's substrate work and gated on the wider `ferrolearn-fetch` migration (shares the `ndarray` types of the sibling fetchers). Open prereq blocker: tracked with the value-parity work under #2060's migration follow-up (no separate scalar oracle — representational). |

## Architecture

`openml.rs` is a single file with one public entry point, one public return
struct, and a private ARFF parser:

- `pub struct OpenmlDataset` — the return type (a `Bunch`-analog): `data:
  Array2<f64>`, `target: Array1<f64>`, `feature_names: Vec<String>`,
  `target_name: String`, `nominal_levels: Vec<Vec<String>>` (per-feature distinct
  nominal levels, empty for numeric columns), `target_levels: Vec<String>`. This
  is a hand-rolled struct, not sklearn's `Bunch` and not the `(X, y)` /
  `return_X_y` pair (REQ-FULL-PARAMS). All arrays are on the `ndarray` substrate
  (REQ-SUBSTRATE).
- `pub fn fetch_openml in openml.rs` — the entry point. It (1) creates the cache
  dir via `dataset_dir("openml/{id}", data_home)` (REQ-CACHE-INTEGRATION),
  (2) fetches `{DATA_INFO_URL}{data_id}` to `data_info.json` and parses the
  `data_set_description` JSON object (REQ-OPENML-API-ENDPOINTS, mirroring
  `_get_data_description_by_id`, `_openml.py:358-367`), (3) resolves the target
  column from the `target_column` arg or the metadata's
  `default_target_attribute` (sklearn's `target_column="default-target"`
  default, `:776`, `:825-828`), (4) downloads the `url` from the metadata to
  `data.arff` and dispatches to `parse_arff`. All failures return
  `Result<_, FerroError>` (`SerdeError` for JSON/metadata/parse problems,
  `IoError` for filesystem) — no panics in the production path (R-CODE-2).
- `fn parse_arff in openml.rs` — the ARFF reader (REQ-ARFF-PARSE). It scans for
  `@attribute` / `@data` markers (case-insensitive, comment lines starting `%`
  skipped), builds a `Vec<Attr>` of `{ name, AttrKind }`, locates the target
  attribute by name (errors if absent), then materializes `data` (features) and
  `target` (the target column) row by row. The private helpers are:
  - `parse_attribute` — classifies one `@attribute` line into `AttrKind::Numeric`
    (`numeric`/`real`/`integer`, with string/date/unknown falling back to
    numeric) or `AttrKind::Nominal(levels)` for `{a,b,c}`, stripping `'`/`"`
    quotes off each level. Mirrors LIAC-ARFF's nominal-vs-numeric split
    (`_arff_parser.py:167-179`).
  - `split_attribute_name` — separates the (optionally single-quoted) attribute
    name from the rest of the line. THIS is the nested-`if let` that trips clippy
    `collapsible_if` on rust 1.95.0 (REQ-CODE-HYGIENE, #2062).
  - the row loop — splits each `@data` line on `,`, validates the field count,
    and encodes each value: numeric via `parse::<f64>()` (`NAN` on failure),
    nominal via the level index (`NAN` if the value is not a known level).

Invariant: nominal attributes (features and target alike) are encoded to 0-based
integer indices into their declaration-order level list, with the level strings
preserved in `nominal_levels` / `target_levels` — matching sklearn's
`encode_nominal=True` LIAC path, which also encodes nominal categories to integer
codes (`_arff_parser.py:167-172`). The structural limits — dense-only, ARFF-only
(no parquet), numeric/nominal-only (string/date coerced to numeric → NaN), single
scalar target — are the surface of REQ-FULL-PARAMS / REQ-VALUE-PARITY and are
recorded as NOT-STARTED, not as parity claims.

## Verification

Commands establishing the SHIPPED claims and the deltas behind the NOT-STARTED
rows:

- `cargo test -p ferrolearn-fetch --lib openml` → 4 passed, 0 failed
  (`parse_minimal_numeric_arff`, `parse_arff_with_nominal_target`,
  `parse_arff_missing_target_errors`, `parse_arff_wrong_field_count_errors`).
  Establishes REQ-ARFF-PARSE (AC-ARFF-1/2) and the parser half of
  REQ-OPENML-API-ENDPOINTS / REQ-CACHE-INTEGRATION offline.
- `cargo test -p ferrolearn-fetch --lib cache` → `dataset_dir_creates_subdir`
  (+ siblings) pass — the shared cache primitive REQ-CACHE-INTEGRATION rides on.
- Consumer check (REQ-CONSUMER):
  `grep -rn "fetch_openml\|OpenmlDataset" --include=*.rs . | grep -v 'src/openml.rs'`
  → re-export in `ferrolearn-fetch/src/lib.rs`, a `#[from]`-consumer note in
  `ferrolearn-core/src/error.rs`, and a test-only caller in
  `ferrolearn-fetch/tests/api_proof.rs` (test, does not count).
- Hygiene check (REQ-CODE-HYGIENE):
  `cargo clippy -p ferrolearn-fetch --lib -- -D warnings` → currently FAILS with
  `error: this `if` statement can be collapsed` at `openml.rs:195`
  (`split_attribute_name`). This is the single code change this iteration
  (#2062); once collapsed to a let-chain the gauntlet goes green.
- Endpoint cite (REQ-OPENML-API-ENDPOINTS): sklearn `_openml.py:34`
  `_DATA_INFO = "api/v1/json/data/{}"`; ferrolearn `DATA_INFO_URL =
  "https://www.openml.org/api/v1/json/data/"` — same path segment, host delta
  noted in the REQ row.
- Value-parity oracle (REQ-VALUE-PARITY, NETWORK-GATED — NOT runnable offline,
  #2060): live `python3 -c "from sklearn.datasets import fetch_openml;
  b=fetch_openml(data_id=61, as_frame=False); print(b.data[:2].tolist(),
  b.target[:2].tolist())"` versus `fetch_openml(61, Some("class"), ..)`. A live
  `fetch_openml(data_id=61)` was confirmed to hang on network with no
  connectivity, so this AC stays NOT-STARTED offline.
- Params cite (REQ-FULL-PARAMS, #2061): sklearn `_openml.py:770-784`
  enumerates `name`/`version`/`data_id`/`target_column`/`cache`/`return_X_y`/
  `as_frame`/`n_retries`/`delay`/`parser`/`read_csv_kwargs`; ferrolearn's
  signature exposes only `(data_id, target_column, data_home)`.

NOT-STARTED REQs map to open blockers: REQ-VALUE-PARITY → #2060 (network-gated),
REQ-FULL-PARAMS → #2061 (feature gap), REQ-CODE-HYGIENE → #2062 (clippy
collapsible_if). REQ-SUBSTRATE rides the fetcher-crate ferray migration (no
scalar oracle; representational). Per R-CHAR-3, all expected values above derive
from sklearn source `file:line` constants or live sklearn oracle calls, never
copied from the ferrolearn side.
