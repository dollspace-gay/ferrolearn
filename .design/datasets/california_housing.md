# fetch_california_housing — California housing dataset loader

<!--
tier: 3-component
status: draft
baseline-commit: 84dfa0db885ada5fea5d9e7c223a73310adb9f57
upstream-paths:
  - sklearn/datasets/_california_housing.py
-->

## Summary

`ferrolearn-fetch/src/california_housing.rs` mirrors scikit-learn's
`sklearn.datasets.fetch_california_housing`
(`sklearn/datasets/_california_housing.py`): a network dataset loader that
downloads the figshare `cal_housing.tgz` tarball, extracts
`CaliforniaHousing/cal_housing.data` (20640 rows × 9 raw columns), then
**re-arranges the columns and computes three DERIVED per-household features**
(`AveRooms`, `AveBedrms`, `AveOccup`) plus a target rescale, materializing a
`(20640, 8)` feature matrix and a `(20640,)` target in 100k-USD units. The
current implementation returns a dedicated `CaliforniaHousing` struct
(`data: Array2<f64>`, `target: Array1<f64>`, `feature_names`, `target_names`)
rather than sklearn's `Bunch`/`(X, y)` pair, and exposes only the common
download + extract + parse path.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here. The HEADLINE SHIPPED claim is the part of this loader that has actual
logic — the column re-arrangement + the derived-feature division math
(`AveRooms = totalRooms/households`, `AveBedrms = totalBedrooms/households`,
`AveOccup = population/households`) + the `target / 100000` rescale. That math is
fully OFFLINE-VERIFIABLE on synthetic input and is pinned exactly by the existing
`parser_handles_synthetic_two_rows` test; it matches sklearn's
`columns_index` + divisions + rescale (`_california_housing.py:190-222`)
element-for-element. There is NO offline code divergence to fix this iteration.

Because `fetch_california_housing` is a NETWORK fetcher, full numerical
value-parity against the live `sklearn.datasets.fetch_california_housing()`
oracle requires network access + downloading the tarball, which is OUT OF REACH
offline — that REQ is NOT-STARTED with a concrete network-gated blocker (it is
NOT a code bug). The other SHIPPED REQs are the OFFLINE-VERIFIABLE contract
pieces: the figshare ARCHIVE URL/checksum/filename matching the sklearn source
constant, the `feature_names`/`target_names` matching the sklearn source lists,
the 9-column CSV parsing + error handling, the cache + tarball-extract
integration, and the public-fn surface.

## Upstream reference (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_california_housing.py`:
  - `ARCHIVE = RemoteFileMetadata(...)` (`:47-51`):
    ```python
    ARCHIVE = RemoteFileMetadata(
        filename="cal_housing.tgz",
        url="https://ndownloader.figshare.com/files/5976036",
        checksum="aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
    )
    ```
  - `def fetch_california_housing(*, data_home=None, download_if_missing=True,
    return_X_y=False, as_frame=False, n_retries=3, delay=1.0)` (`:67-75`) — the
    public, keyword-only signature.
  - Tarball extract (`:184-187`): `np.loadtxt(f.extractfile(
    "CaliforniaHousing/cal_housing.data"), delimiter=",")` — the member path is
    `CaliforniaHousing/cal_housing.data`, comma-delimited.
  - Column re-arrangement (`:190-191`):
    ```python
    columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
    cal_housing = cal_housing[:, columns_index]
    ```
    The raw file order is `0:longitude, 1:latitude, 2:housingMedianAge,
    3:totalRooms, 4:totalBedrooms, 5:population, 6:households, 7:medianIncome,
    8:medianHouseValue`. After re-ordering, column 0 is `medianHouseValue` (raw 8)
    and columns 1..9 are `[medianIncome(7), housingMedianAge(2), totalRooms(3),
    totalBedrooms(4), population(5), households(6), latitude(1), longitude(0)]`.
  - Target/feature split + derived features + rescale (`:210-222`):
    ```python
    target, data = cal_housing[:, 0], cal_housing[:, 1:]
    # avg rooms = total rooms / households
    data[:, 2] /= data[:, 5]
    # avg bed rooms = total bed rooms / households
    data[:, 3] /= data[:, 5]
    # avg occupancy = population / households
    data[:, 5] = data[:, 4] / data[:, 5]
    # target in units of 100,000
    target = target / 100000.0
    ```
    Here `data` columns are `[medInc, houseAge, totalRooms, totalBedrooms,
    population, households, latitude, longitude]`, so after the divisions:
    col 2 → `totalRooms/households` (AveRooms), col 3 →
    `totalBedrooms/households` (AveBedrms), col 5 → `population/households`
    (AveOccup); cols 4/6/7 stay `population`/`latitude`/`longitude`.
  - `feature_names` (`:199-208`):
    ```python
    feature_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]
    ```
  - `target_names = ["MedHouseVal"]` (`:230-232`).
  - Documented shapes (`:78-83`, `:125-131`): `Samples total 20640`,
    `Dimensionality 8`, `Target real 0.15 - 5.`; `data : ndarray, shape
    (20640, 8)`; `target : numpy array of shape (20640,)`.
  - Cache/download (`:164-194`): `_pkl_filepath(data_home, "cal_housing.pkz")`,
    `_fetch_remote(ARCHIVE, dirname=data_home, n_retries=…, delay=…)` (the
    SHA-checked download), `tarfile.open(mode="r:gz", …)`, persisted via
    `joblib.dump(cal_housing, filepath, compress=6)`.

## Requirements

- REQ-ARCHIVE-METADATA: the remote-file metadata ferrolearn downloads
  (`filename`, `url`, `sha256`) is byte-for-byte identical to sklearn's source
  constant `ARCHIVE` (`_california_housing.py:47-51`) — same figshare archive
  (`/5976036`), same checksum. Offline-verifiable (string-constant
  characterization).
- REQ-DERIVED-FEATURES (**HEADLINE**): from a raw 9-column row
  (`0:longitude, 1:latitude, 2:housingMedianAge, 3:totalRooms, 4:totalBedrooms,
  5:population, 6:households, 7:medianIncome, 8:medianHouseValue`) the loader
  produces the 8 sklearn features in sklearn order —
  `MedInc=raw7, HouseAge=raw2, AveRooms=raw3/raw6, AveBedrms=raw4/raw6,
  Population=raw5, AveOccup=raw5/raw6, Latitude=raw1, Longitude=raw0` — and the
  target `raw8/100000`. This is the exact composition of sklearn's
  `columns_index = [8,7,2,3,4,5,6,1,0]` re-order + the three `/= data[:,5]`
  divisions + `target/100000.0` (`_california_housing.py:190-222`).
  Offline-verifiable on synthetic input.
- REQ-FEATURE-TARGET-NAMES: `feature_names` is exactly
  `["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup",
  "Latitude","Longitude"]` and `target_names` is exactly `["MedHouseVal"]`,
  matching sklearn's source lists (`_california_housing.py:199-208`, `:230-232`).
  Offline-verifiable (string-constant characterization).
- REQ-CSV-PARSE: the loader parses the extracted CSV as 9 comma-separated columns
  per non-empty line, errors on a wrong column count and on a non-float field,
  and materializes `data` of shape `(n, 8)` + `target` of length `n`. Mirrors
  sklearn's `np.loadtxt(…, delimiter=",")` over a fixed-9-column file
  (`_california_housing.py:184-187`). Offline-verifiable on in-crate CSV text.
- REQ-CACHE-EXTRACT: the fetcher caches under
  `dataset_dir("california_housing", data_home)`, downloads via
  `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`
  (first-use, SHA-256-verified), and extracts the `cal_housing.data` member from
  the gzipped tarball. Mirrors sklearn's `_fetch_remote(ARCHIVE, …)` +
  `tarfile.open(mode="r:gz") → f.extractfile("CaliforniaHousing/cal_housing.data")`
  (`_california_housing.py:177-187`). Offline up to the network GET.
- REQ-CONSUMER: `fetch_california_housing` / `CaliforniaHousing` are public crate
  API (re-exported in `lib.rs`); grandfathered boundary API (R-DEFER-1 / S5).
- REQ-VALUE-PARITY: the fetched `(data, target)` arrays match
  `sklearn.datasets.fetch_california_housing()` element-by-element on the real
  20640×8 dataset. NETWORK-GATED.
- REQ-FULL-PARAMS: the sklearn `fetch_california_housing` keyword-only parameters
  ferrolearn currently omits — `download_if_missing`, `return_X_y`, `as_frame`,
  `n_retries`, `delay`. Feature gap.
- REQ-SUBSTRATE: array types are the ferray substrate (`ferray-core`), not
  `ndarray` (R-SUBSTRATE-1).

## Acceptance criteria

All oracle commands run against live scikit-learn 1.5.2
(`python3 -c "import sklearn; print(sklearn.__version__)"` → `1.5.2`). Expected
values come from the live oracle or the sklearn source `file:line`, NEVER from
ferrolearn (R-CHAR-3).

- AC-ARCHIVE-1 (REQ-ARCHIVE-METADATA): the sklearn source constant
  ```
  python3 -c "from sklearn.datasets import _california_housing as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"
  ```
  → `cal_housing.tgz https://ndownloader.figshare.com/files/5976036 aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`.
  ferrolearn's `ARCHIVE` (`california_housing.rs`) carries exactly these three
  values (`filename`/`url`/`sha256`). Partially pinned offline by
  `metadata_constants_match_sklearn` (filename + figshare host + 64-char sha).
- **AC-DERIVED-1 (REQ-DERIVED-FEATURES, the headline offline check):** for the
  synthetic raw row `-122.0,37.0,30.0,1000.0,200.0,400.0,100.0,5.0,250000.0`
  (raw cols `lon=-122, lat=37, age=30, totalRooms=1000, totalBedrooms=200,
  pop=400, households=100, medInc=5, medHouseVal=250000`), the parsed feature row
  is `MedInc=5`, `HouseAge=30`, `AveRooms=1000/100=10`, `AveBedrms=200/100=2`,
  `Population=400`, `AveOccup=400/100=4`, `Latitude=37`, `Longitude=-122`, and
  `target=250000/100000=2.5`. Every value is the composition of sklearn's
  `columns_index=[8,7,2,3,4,5,6,1,0]` + divisions + rescale
  (`_california_housing.py:190-222`). Pinned OFFLINE by
  `parser_handles_synthetic_two_rows` (asserts all 8 cols of row 0 + both
  targets, `tol 1e-12`).
- AC-PARSE-1 (REQ-CSV-PARSE, column-count error): a line with the wrong number of
  columns (`"1,2,3,4"`, 4 ≠ 9) is rejected with an `Err`. (Offline; pinned by
  `parser_rejects_short_row`.) sklearn likewise treats the file as a fixed
  9-wide table via `np.loadtxt(…, delimiter=",")` (`:185`).
- AC-PARSE-2 (REQ-CSV-PARSE, non-numeric field): a line with a non-float field
  (`"1,2,3,4,5,6,7,abc,9"`) is rejected with an `Err`. (Offline; pinned by
  `parser_rejects_non_numeric`.)
- AC-NAMES-1 (REQ-FEATURE-TARGET-NAMES): the sklearn source lists
  ```
  python3 -c "from sklearn.datasets import fetch_california_housing as f; b=f.__wrapped__() if hasattr(f,'__wrapped__') else None" # see live oracle below
  ```
  yield `feature_names == ["MedInc","HouseAge","AveRooms","AveBedrms",
  "Population","AveOccup","Latitude","Longitude"]` and `target_names ==
  ["MedHouseVal"]` (source `:199-208`, `:230-232`). ferrolearn returns these
  exact lists. The feature-name VALUES are exercised offline by the synthetic
  test only structurally (the struct carries them); the string-list equality is a
  source-readable characterization.
- AC-VALUE-1 (REQ-VALUE-PARITY, NETWORK-GATED oracle): live
  ```
  python3 -c "from sklearn.datasets import fetch_california_housing as f; b=f(); print(b.data.shape, b.target.shape, b.feature_names, b.data[0].tolist(), float(b.target[0]))"
  ```
  → `(20640, 8) (20640,) ['MedInc',…,'Longitude'] [...] <val>` versus
  `fetch_california_housing(None)` element by element. Requires network + the
  tarball download — NOT runnable offline; pinned under network-gated blocker
  #NNN (to be filed by the critic).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-ARCHIVE-METADATA (figshare URL/SHA/filename) | SHIPPED | `pub const ARCHIVE: RemoteFile in california_housing.rs` = `{ filename: "cal_housing.tgz", url: "https://ndownloader.figshare.com/files/5976036", sha256: "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681" }`, byte-for-byte equal to sklearn source `ARCHIVE = RemoteFileMetadata(filename="cal_housing.tgz", url="https://ndownloader.figshare.com/files/5976036", checksum="aaa5c9a6…ea681")` (`_california_housing.py:47-51`). Characterization is exact (R-CHAR-3): the expected values are read from the sklearn SOURCE constant, not from ferrolearn. Non-test consumer: `pub fn fetch_california_housing in california_housing.rs` reads all three fields (`fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`); `ARCHIVE` is `pub` and reachable via `pub mod california_housing` in `lib.rs`. Verification: live `python3 -c "from sklearn.datasets import _california_housing as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"` → matches the three constants; `cargo test -p ferrolearn-fetch --lib california_housing` → `metadata_constants_match_sklearn` passes (filename == `cal_housing.tgz`, url contains `figshare.com`, sha length 64). |
| REQ-DERIVED-FEATURES (column re-order + AveRooms/AveBedrms/AveOccup + target/1e5) | SHIPPED | `fn parse_california_csv in california_housing.rs` computes, per raw row, `med_inc = row[7]`, `house_age = row[2]`, `ave_rooms = total_rooms / households` (`row[3]/row[6]`), `ave_bedrms = total_bedrooms / households` (`row[4]/row[6]`), `ave_occup = pop / households` (`row[5]/row[6]`), `latitude = row[1]`, `longitude = row[0]`, and writes them to `data[[i,0..8]]` in that order, with `target[i] = med_house_val / 100_000.0` (`row[8]/1e5`). This is the EXACT composition of sklearn's re-order `columns_index = [8,7,2,3,4,5,6,1,0]` + `data[:,2]/=data[:,5]` (AveRooms), `data[:,3]/=data[:,5]` (AveBedrms), `data[:,5]=data[:,4]/data[:,5]` (AveOccup), `target = target/100000.0` (`_california_housing.py:190-222`) — verified term-by-term against the source, no divergence. Non-test consumer: called by `pub fn fetch_california_housing in california_housing.rs` (`parse_california_csv(&raw)` after tarball extract). Verification: `cargo test -p ferrolearn-fetch --lib california_housing` → `parser_handles_synthetic_two_rows` asserts (tol `1e-12`) row 0 = `[5, 30, 10, 2, 400, 4, 37, -122]` and targets `[2.5, 5.0]` from synthetic raw input — i.e. `AveRooms=1000/100=10`, `AveBedrms=200/100=2`, `AveOccup=400/100=4`, `target=250000/100000=2.5` (4 passed, 0 failed). BENIGN DEVIATION (R-DEV-4, documented below, NOT a divergence): ferrolearn divides by `households = row[6].max(1.0)` (a divide-by-zero guard); on the real dataset `households ≥ 1` always, so the guard is INERT and outputs match sklearn. |
| REQ-FEATURE-TARGET-NAMES (8 names + MedHouseVal) | SHIPPED | `fn parse_california_csv in california_housing.rs` returns `feature_names: vec!["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]` and `target_names: vec!["MedHouseVal"]`. Byte-for-byte equal to sklearn source `feature_names = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]` (`_california_housing.py:199-208`) and `target_names = ["MedHouseVal"]` (`:230-232`). Characterization is exact (R-CHAR-3): the expected list is read from the sklearn SOURCE, not from ferrolearn. Non-test consumer: `pub fn fetch_california_housing in california_housing.rs` returns this `CaliforniaHousing { feature_names, target_names, .. }` struct. Verification: source-readable string-list equality against `_california_housing.py:199-208`/`:230-232`; the synthetic test materializes the struct (its `feature_names`/`target_names` fields are populated by the same code path). |
| REQ-CSV-PARSE (9-col parse + shape + errors) | SHIPPED | `fn parse_california_csv in california_housing.rs` splits each non-empty trimmed line on `,`, requires exactly 9 columns (`if parts.len() != 9 { return Err(FerroError::SerdeError { .. "expected 9" }) }`), parses each field via `p.parse::<f64>()` (erroring with a line/col-located `SerdeError` on a non-number), then materializes `data = Array2::<f64>::zeros((n, 8))` and `target = Array1::<f64>::zeros(n)`. Mirrors sklearn's fixed-width `np.loadtxt(f.extractfile("CaliforniaHousing/cal_housing.data"), delimiter=",")` over the 9-column raw file (`_california_housing.py:184-187`). Non-test consumer: called by `pub fn fetch_california_housing in california_housing.rs`. Verification: `cargo test -p ferrolearn-fetch --lib california_housing` → `parser_rejects_short_row` (4-col line → `Err`) and `parser_rejects_non_numeric` (`abc` field → `Err`) pass; `parser_handles_synthetic_two_rows` asserts `data.dim() == (2, 8)` and `target.len() == 2`. |
| REQ-CACHE-EXTRACT (dataset_dir + SHA fetch + tarball extract) | SHIPPED | `pub fn fetch_california_housing in california_housing.rs` calls `let dir = dataset_dir("california_housing", data_home)?;` (`fn dataset_dir in cache.rs`), then `let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;` (`pub fn fetch_file in fetch.rs`) — first-use, SHA-256-verified download — then `extract_data_file(&archive_path, &dir)?`. `fn extract_data_file in california_housing.rs` opens the `.tgz` with `Archive::new(GzDecoder::new(f))` and unpacks the entry whose file name is `cal_housing.data`. Mirrors sklearn's `_fetch_remote(ARCHIVE, dirname=data_home, …)` + `tarfile.open(mode="r:gz") → f.extractfile("CaliforniaHousing/cal_housing.data")` (`_california_housing.py:177-187`). Non-test consumer: `fetch_california_housing` itself (boundary API); `dataset_dir`/`fetch_file` are the shared crate primitives. Verification: `cargo test -p ferrolearn-fetch --lib cache` (the `dataset_dir("california_housing", …) == home/california_housing` primitive REQ-CACHE-EXTRACT rides on passes, `cache.rs` test); the SHA-gated download + tarball fetch run only with network (see network blocker). FAITHFUL DELTA (not a parity claim): ferrolearn matches by tar-entry `file_name() == "cal_housing.data"`, sklearn by the full member path `CaliforniaHousing/cal_housing.data`; ferrolearn caches the raw `.tgz` + extracted `.data`, sklearn persists a `joblib.dump`-ed `cal_housing.pkz` and `remove`s the archive (`:193-194`) — cache LOCATION/format delta, folded into REQ-VALUE-PARITY. |
| REQ-CONSUMER (public boundary surface) | SHIPPED | `pub fn fetch_california_housing` + `pub struct CaliforniaHousing` (+ `pub const ARCHIVE`) are re-exported at the crate root: `pub use california_housing::{CaliforniaHousing, fetch_california_housing}` in `lib.rs`, documented in the crate `//!` header (`[fetch_california_housing] — 20640×8 regression benchmark.`, `lib.rs:13`). Existing pub boundary API, grandfathered under R-DEFER-1 / S5 (the fetcher fn + struct ARE the public API; external users + the eventual `ferrolearn-python` `datasets` binding mirroring `from sklearn.datasets import fetch_california_housing` are the consumers). Verification: `grep -rn "fetch_california_housing\|CaliforniaHousing" ferrolearn-fetch/src/ \| grep -v 'california_housing.rs'` → the `lib.rs` re-export + the crate-doc line (and the `dataset_dir` consumer note in `cache.rs`). UNDERCLAIM: there is NO in-workspace non-test FUNCTION caller; this rests on the boundary-API grandfather clause + the crate re-export. |
| REQ-VALUE-PARITY (element-wise (data,target) vs oracle) | NOT-STARTED | open prereq blocker #NNN (network-gated, NOT a code bug; to be filed by the critic). No offline path: `pub fn fetch_california_housing in california_housing.rs` calls `fetch_file` → the crate's `download` primitive (network GET), which requires network access + downloading the figshare `cal_housing.tgz` to produce `(data, target)`. Element-by-element parity vs `sklearn.datasets.fetch_california_housing()` on the real 20640×8 dataset cannot be established offline. The cache-format / member-path deltas noted in the SHIPPED rows resolve here once network parity can be run. The verified offline contract (ARCHIVE, derived-feature math, names, parse, cache wiring, public surface) exhibits NO divergence to fix this iteration. |
| REQ-FULL-PARAMS (omitted keyword-only params) | NOT-STARTED | open prereq blocker #NNN (feature gap; to be filed by the critic). `pub fn fetch_california_housing in california_housing.rs` has signature `fetch_california_housing(data_home: Option<&Path>)` — only `data_home`. sklearn `def fetch_california_housing(*, data_home=None, download_if_missing=True, return_X_y=False, as_frame=False, n_retries=3, delay=1.0)` (`_california_housing.py:67-75`). Omitted: `download_if_missing` (the offline-OSError gate, `:170-171`), `return_X_y` (the `(X, y)` tuple return, `:238-239`), `as_frame` (the `_convert_data_dataframe` DataFrame path, `:233-236`), `n_retries`/`delay` (download retry policy, `:180-181`). |
| REQ-SUBSTRATE (ferray-core array types) | NOT-STARTED | open prereq blocker #NNN (ferray migration; to be filed by the critic). `california_housing.rs` uses `use ndarray::{Array1, Array2}` and `pub struct CaliforniaHousing { data: Array2<f64>, target: Array1<f64>, .. }`. R-SUBSTRATE-1 requires `ferray-core` array types, not `ndarray`. Prerequisite: migrate the unit's array types to the ferray analog; folded into this unit's substrate work and gated on the wider `ferrolearn-fetch` migration (shares the `ndarray` types of the sibling fetchers). Representational (no scalar oracle). |

## Architecture

`california_housing.rs` is a single file with one public constant, one public
return struct, one public entry point, and two private helpers:

- `pub const ARCHIVE: RemoteFile` — the figshare remote-file descriptor
  (`filename`/`url`/`sha256`), a verbatim copy of sklearn's source `ARCHIVE`
  constant (REQ-ARCHIVE-METADATA, `_california_housing.py:47-51`). It is the
  single source of the download URL and the SHA-256 the cache layer verifies
  against.
- `pub struct CaliforniaHousing` — the return type (a `Bunch`-analog): `data:
  Array2<f64>` (the 8-feature matrix), `target: Array1<f64>` (median house value
  in 100k-USD units), `feature_names: Vec<&'static str>` (the 8 names),
  `target_names: Vec<&'static str>` (`["MedHouseVal"]`). This is a hand-rolled
  struct, not sklearn's `Bunch` and not the `(X, y)`/`return_X_y` pair
  (REQ-FULL-PARAMS); both arrays are on the `ndarray` substrate (REQ-SUBSTRATE).
- `pub fn fetch_california_housing(data_home: Option<&Path>) ->
  Result<CaliforniaHousing, FerroError>` — the entry point. It (1) creates the
  cache dir via `dataset_dir("california_housing", data_home)`
  (REQ-CACHE-EXTRACT), (2) downloads + SHA-verifies the tarball via
  `fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)`
  (mirroring `_fetch_remote(ARCHIVE, …)`, `:177-182`), (3) extracts
  `cal_housing.data` via `extract_data_file` (mirroring `tarfile.open(mode=
  "r:gz") → f.extractfile(…)`, `:184-186`), reads it to a string, and (4)
  dispatches to `parse_california_csv`. All failures return `Result<_,
  FerroError>` (`SerdeError` for parse/missing-member, `IoError` for
  filesystem/gunzip) — no panics in the production path (R-CODE-2).
- `fn extract_data_file(archive_path, dir)` — opens the gzipped tar via
  `GzDecoder` + `tar::Archive`, scans entries for the one whose `file_name()` is
  `cal_housing.data`, unpacks it to `dir/cal_housing.data`, and short-circuits if
  the extracted file already exists. If no such member exists it returns a
  `SerdeError` ("cal_housing.data not found in archive").
- `fn parse_california_csv(raw: &str) -> Result<CaliforniaHousing, FerroError>`
  — the CSV reader + DERIVED-FEATURE engine (REQ-CSV-PARSE /
  REQ-DERIVED-FEATURES / REQ-FEATURE-TARGET-NAMES). It iterates lines (skipping
  blanks), splits each on `,`, enforces exactly 9 columns, parses every field as
  `f64`, then — per row — re-orders raw columns into the 8 sklearn features and
  applies the three per-household divisions + the `target/1e5` rescale.

**The derived-feature invariant (the heart of this loader).** sklearn expresses
the transform as a re-order (`columns_index = [8,7,2,3,4,5,6,1,0]`) followed by
in-place column divisions on the post-re-order matrix. ferrolearn expresses the
SAME transform directly per-row by indexing the raw columns. The mapping is:

| sklearn output col | sklearn expression (post-reorder) | raw col | ferrolearn expression |
|---|---|---|---|
| target | `cal[:,0]` then `/100000` | raw 8 | `row[8] / 100_000.0` |
| MedInc (0) | `data[:,0]` | raw 7 | `row[7]` |
| HouseAge (1) | `data[:,1]` | raw 2 | `row[2]` |
| AveRooms (2) | `data[:,2] /= data[:,5]` | raw 3 / raw 6 | `row[3] / households` |
| AveBedrms (3) | `data[:,3] /= data[:,5]` | raw 4 / raw 6 | `row[4] / households` |
| Population (4) | `data[:,4]` | raw 5 | `row[5]` |
| AveOccup (5) | `data[:,5] = data[:,4]/data[:,5]` | raw 5 / raw 6 | `row[5] / households` |
| Latitude (6) | `data[:,6]` | raw 1 | `row[1]` |
| Longitude (7) | `data[:,7]` | raw 0 | `row[0]` |

(`data[:,5]` is `households` = raw 6 pre-division, and `data[:,4]` is
`population` = raw 5; the order of sklearn's three division statements matters
because AveOccup reads `data[:,5]` before overwriting it — ferrolearn sidesteps
the ordering hazard by reading the raw `households` for all three divisions,
which is equivalent.) This is exactly equal term-by-term to
`_california_housing.py:210-222`.

**Benign deviation (R-DEV-4 — a Rust footgun guard, INERT on real data, NOT a
divergence).** ferrolearn divides by `let households = row[6].max(1.0);` whereas
sklearn divides by `data[:,5]` directly. On the real California-housing dataset
`households` is always `≥ 1`, so `.max(1.0)` is a no-op and ferrolearn's outputs
match sklearn element-for-element. The two implementations differ ONLY on a
hypothetical `households == 0` row, where sklearn would yield `inf`/`nan`
(divide-by-zero) and ferrolearn would yield `value / 1.0`. This is a deliberate
divide-by-zero footgun guard of the kind R-DEV-4 sanctions (a Rust analog of a
quirk Python's float division handles by producing `inf`), and since the real
data never triggers it, it is documented here as a benign deviation, NOT as a
REQ-VALUE-PARITY blocker. If the live oracle ever surfaces a `households == 0`
row (it does not), this would be reclassified.

The structural limits — download-only (no `download_if_missing=False` offline
gate), no `(X, y)`/DataFrame return, no retry policy, `ndarray` substrate — are
the surface of REQ-FULL-PARAMS / REQ-VALUE-PARITY / REQ-SUBSTRATE and are
recorded as NOT-STARTED, not as parity claims.

## Verification

Commands establishing the SHIPPED claims and the deltas behind the NOT-STARTED
rows (sklearn 1.5.2 oracle, toolchain per workspace):

- `cargo test -p ferrolearn-fetch --lib california_housing` → 4 passed, 0 failed
  (`parser_handles_synthetic_two_rows`, `parser_rejects_short_row`,
  `parser_rejects_non_numeric`, `metadata_constants_match_sklearn`). Establishes
  REQ-DERIVED-FEATURES (AC-DERIVED-1, the headline), REQ-CSV-PARSE
  (AC-PARSE-1/2 + shape), and the filename/host/sha-length portion of
  REQ-ARCHIVE-METADATA, all offline.
- ARCHIVE oracle (REQ-ARCHIVE-METADATA, exact characterization from sklearn
  source — R-CHAR-3):
  `python3 -c "from sklearn.datasets import _california_housing as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"`
  → `cal_housing.tgz https://ndownloader.figshare.com/files/5976036 aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`,
  byte-for-byte equal to ferrolearn's three `ARCHIVE` constants. No divergence.
- Derived-feature cite (REQ-DERIVED-FEATURES): sklearn
  `_california_housing.py:190-222` — `columns_index = [8,7,2,3,4,5,6,1,0]`,
  `data[:,2] /= data[:,5]`, `data[:,3] /= data[:,5]`,
  `data[:,5] = data[:,4]/data[:,5]`, `target = target/100000.0`. ferrolearn's
  per-row `ave_rooms = total_rooms/households`, `ave_bedrms =
  total_bedrooms/households`, `ave_occup = pop/households`, `target =
  med_house_val/100_000.0` — equal term-by-term (see the Architecture table).
- Names cite (REQ-FEATURE-TARGET-NAMES): sklearn source
  `feature_names = ["MedInc","HouseAge","AveRooms","AveBedrms","Population",
  "AveOccup","Latitude","Longitude"]` (`:199-208`), `target_names =
  ["MedHouseVal"]` (`:230-232`); ferrolearn returns these exact `vec!`s.
- Cache-wiring (REQ-CACHE-EXTRACT): `cargo test -p ferrolearn-fetch --lib cache`
  (the `dataset_dir("california_housing", Some(p)) == p.join("california_housing")`
  assertion in `cache.rs` passes); the SHA-gated `fetch_file` download +
  tarball extract run only with network (network blocker).
- Consumer check (REQ-CONSUMER):
  `grep -rn "fetch_california_housing\|CaliforniaHousing" ferrolearn-fetch/src/ | grep -v 'california_housing.rs'`
  → the `lib.rs` re-export (`pub use california_housing::{CaliforniaHousing,
  fetch_california_housing}`) + the crate-doc line. No non-test function caller
  in-workspace (boundary-API grandfather).
- Hygiene: `cargo clippy -p ferrolearn-fetch --lib -- -D warnings`.
- Value-parity oracle (REQ-VALUE-PARITY, NETWORK-GATED — NOT runnable offline):
  live
  `python3 -c "from sklearn.datasets import fetch_california_housing as f; b=f(); print(b.data.shape, b.target.shape, b.feature_names, b.data[0].tolist(), float(b.target[0]))"`
  → `(20640, 8) (20640,) [...] [...] <val>` versus `fetch_california_housing(None)`.
  Requires network + the tarball download, so this AC stays NOT-STARTED offline.
- Params cite (REQ-FULL-PARAMS): sklearn `_california_housing.py:67-75`
  enumerates `data_home`/`download_if_missing`/`return_X_y`/`as_frame`/
  `n_retries`/`delay`; ferrolearn's signature exposes only `(data_home)`.

NOT-STARTED REQs map to open blockers to be filed by the critic:
REQ-VALUE-PARITY → #NNN (network-gated), REQ-FULL-PARAMS → #NNN (feature gap),
REQ-SUBSTRATE → #NNN (ferray migration; representational). Per R-CHAR-3, all
expected values above derive from sklearn source `file:line` constants or live
sklearn oracle calls, never copied from the ferrolearn side. The verified offline
contract (ARCHIVE constants, the derived-feature math, feature/target names, CSV
parse, cache + tarball-extract wiring, public surface) exhibits NO code
divergence this iteration — the `households.max(1.0)` guard is a benign R-DEV-4
deviation, inert on real data.
