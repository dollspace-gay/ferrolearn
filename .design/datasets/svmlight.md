# SVMlight / LIBSVM format I/O + load_files

<!--
tier: 3-component
status: draft
baseline-commit: 9843d604ed19ef7f18d696b55532be73b5ebe9db
upstream-paths:
  - sklearn/datasets/_svmlight_format_io.py
  - sklearn/datasets/_base.py
-->

## Summary

`ferrolearn-datasets/src/svmlight.rs` mirrors scikit-learn's SVMlight / LIBSVM
sparse-text I/O — `load_svmlight_file`, `load_svmlight_files`, `dump_svmlight_file`
from `sklearn/datasets/_svmlight_format_io.py` — plus `load_files`, which in
scikit-learn lives in `sklearn/datasets/_base.py`. The current ferrolearn module
implements a *self-consistent 1-based dense* variant that round-trips against
itself but is **not** wire-compatible with scikit-learn's default output, and
omits the sparse-CSR return type and most of sklearn's optional parameters.

This doc is a contract over the **existing** code. No `.rs` edits are proposed
here; gaps are recorded as NOT-STARTED REQs with concrete prerequisites.

## Upstream cites (read-only, tag 1.5.2, commit 156ef14)

- `sklearn/datasets/_svmlight_format_io.py` — `load_svmlight_file` (`:64`),
  `load_svmlight_files` (`:267`), `dump_svmlight_file` (`:474`), and the
  index-base resolution in `load_svmlight_files` (`:402-419`).
- `sklearn/datasets/_base.py` — `load_files` (`:142`), returning a `Bunch`
  (`:307-317`).

## Requirements

- REQ-1 (load: 1-based parse): parse `<label> idx:val ...` lines, strip `#`
  comments, skip blank lines, dense `(X, y)` for the 1-based case.
- REQ-2 (load: index-base contract): honor `zero_based ∈ {True, False, "auto"}`
  (default `"auto"`) per `_svmlight_format_io.py:402-409`; 0-based files (min
  index 0) must load as-is.
- REQ-3 (load: sparse CSR return): return a `scipy.sparse` CSR-equivalent `X`
  (`:159`, `:424`).
- REQ-4 (load: multi-file common n_features): `load_svmlight_files` enforces a
  SINGLE `n_features = max col index + 1` across ALL files (`:410-419`).
- REQ-5 (load: optional params): `multilabel`, `query_id` (`qid:` tokens),
  `offset`/`length` byte-range reads, `dtype` (f32/f64).
- REQ-6 (dump: index-base default): `dump_svmlight_file` default
  `zero_based=True` writes ZERO-based `idx:val` (`:474-483`, `:596`).
- REQ-7 (dump: value/label text formatting): numeric formatting of labels and
  values matches sklearn's output.
- REQ-8 (dump: comment header + params): provenance comment + "Column indices
  are zero/one-based" header when `comment` given (`:434-445`); `query_id`,
  `multilabel`, file-like `f`.
- REQ-9 (load_files: Bunch contract): return a `Bunch` with `data`, `target`,
  `target_names`, `filenames`, `DESCR`; support `categories`, `load_content`,
  `shuffle` (default True, `random_state`), `encoding`, `decode_error`,
  `allowed_extensions` (`_base.py:142`, `:307-317`).
- REQ-10 (substrate): array types are ferray (`ferray-core`), not `ndarray`.
- REQ-11 (production consumer): each public fn has a non-test production
  consumer.

## Acceptance criteria

- AC-1 (REQ-1): `parse_line("1.0 1:0.5 3:1.5")` yields label `1.0` and dense row
  `[0.5, 0, 1.5]`; a trailing `# comment` is stripped.
- AC-2 (REQ-2): live `load_svmlight_file(BytesIO(b"1 0:1 2:2\n0 1:3\n"))`
  returns shape `(2, 3)` with row 0 `[1, 0, 2]` — ferrolearn must produce the
  same `X` for that 0-based content.
- AC-3 (REQ-3): sklearn returns `scipy.sparse._csr.csr_matrix`; ferrolearn `X`
  is the sparse analog (not dense `Array2`).
- AC-4 (REQ-4): two files with differing max indices load to a common column
  count (`max over files + 1`).
- AC-6 (REQ-6): live default dump of `[[1,0,2],[0,3,0],[4,5,6]]` /
  `[1,0,1]` is `1 0:1 2:2 / 0 1:3 / 1 0:4 1:5 2:6` (ZERO-based) — ferrolearn must
  match byte-for-byte.
- AC-7 (REQ-7): label `1.0` → `1`, value `0.5` → `0.5`, value `1/3` →
  `0.3333333333333333` (live oracle).
- AC-9 (REQ-9): `load_files(path)` returns a `Bunch`; `bunch.target_names` is a
  list, `bunch.filenames` present, default `shuffle=True`.

## REQ status

Deterministic / oracle-pinnable (critic should pin a failing characterization
test): REQ-2, REQ-4, REQ-6, REQ-7, REQ-9. Representational / feature-gap (no
single scalar oracle — blocked on a missing type or unimplemented parameter):
REQ-3, REQ-5, REQ-10. REQ-1, REQ-8, REQ-11 are mixed (see rows).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (load 1-based parse) | SHIPPED | `fn parse_line in svmlight.rs` strips `#` (`let body = match line.find('#')`), errors on empty, parses `idx:val` and stores `(idx_one_based - 1, val)`; `fn load_svmlight_str` builds the dense `(X, y)`. Non-test consumer: re-exported at crate root in `lib.rs` (`pub use svmlight::{... load_svmlight_file, load_svmlight_str ...}`) and reached transitively from `dump_svmlight_file` round-trips. Verification: `cargo test -p ferrolearn-datasets --lib svmlight` → `parse_basic_line`, `parse_with_comment`, `round_trip_dense` pass (7 passed, 0 failed). Scope note: SHIPPED only for the 1-based parse path; the index-base CONTRACT is REQ-2. |
| REQ-2 (load index-base contract) | NOT-STARTED | `fn parse_line in svmlight.rs` HARD-CODES 1-based and ERRORS on index 0: `if idx_one_based == 0 { return Err(... "svmlight: feature indices are 1-based, got 0" ...) }`. No `zero_based` parameter on `fn load_svmlight_file` / `fn load_svmlight_str`. sklearn (`_svmlight_format_io.py:402-409`) keeps 0-based files (min index 0) as-is under default `"auto"`; live oracle: `load_svmlight_file(BytesIO(b"1 0:1 2:2\n0 1:3\n"))` → shape `(2,3)`, `X[0]=[1,0,2]`. ferrolearn rejects that exact (sklearn-default) content. Deterministic — oracle-pinnable. Prerequisite blocker: add `zero_based: {True,False,"auto"}` param + the `"auto"` min-index heuristic to the parse/load path (tracking issue #1640; critic to file the specific `#NNN`). |
| REQ-3 (load sparse CSR return) | NOT-STARTED | `type SvmlightDataset = (Array2<f64>, Array1<f64>)` in `svmlight.rs` returns a DENSE matrix; `fn load_svmlight_str` materializes `Array2::<f64>::zeros((n_samples, n_feat))`. sklearn returns `scipy.sparse` CSR (`_svmlight_format_io.py:159`, `:424` `sp.csr_matrix(...)`). Representational gap. Prerequisite: a sparse CSR array type from the ferray sparse analog / `ferrolearn-sparse`; until that exists the return-type contract cannot be met (tracking issue #1640). |
| REQ-4 (multi-file common n_features) | NOT-STARTED | `fn load_svmlight_files in svmlight.rs` maps `load_svmlight_file(p.as_ref(), n_features)` independently — when `n_features` is `None` each file infers its OWN `max_feat`, so shapes can differ across files. sklearn computes ONE `n_f = max(...) + 1` across ALL files (`_svmlight_format_io.py:410-419`). Deterministic — oracle-pinnable (two files, different max indices → sklearn gives equal column counts). Prerequisite: two-pass inference (scan all files, then build with shared `n_features`) (tracking issue #1640). |
| REQ-5 (multilabel/query_id/offset/length/dtype) | NOT-STARTED | None of these params exist on `fn load_svmlight_file` / `fn load_svmlight_files`. A `qid:` token would hit `idx_tok.parse::<usize>()` in `fn parse_line` and error (`"svmlight: bad index 'qid'"`); `multilabel` y-as-list-of-tuples, byte-range `offset`/`length`, and f32 `dtype` (module is f64-only via `Array2<f64>`) are all absent. Feature-gap. Prerequisite: implement each parameter against sklearn's signature (`_svmlight_format_io.py:64-74`); `dtype` also depends on the generic-`F` rework (tracking issue #1640). |
| REQ-6 (dump index-base default) | NOT-STARTED | `fn dump_svmlight_file in svmlight.rs` ALWAYS writes ONE-based: `buf.push_str(&format!("{}:{}", j + 1, v))`, with no `zero_based` parameter. sklearn default `zero_based=True` writes ZERO-based (`_svmlight_format_io.py:474-483`, `:596` `one_based = not zero_based`). Live oracle (default): `[[1,0,2],[0,3,0],[4,5,6]]`/`[1,0,1]` → `1 0:1 2:2\n0 1:3\n1 0:4 1:5 2:6\n`; ferrolearn emits `1 1:1 3:2\n...` (off by one). HEADLINE divergence: ferrolearn-dumped files are unreadable by any sklearn reader and only round-trip with ferrolearn's own 1-based loader. Deterministic — oracle-pinnable (byte-for-byte dump comparison). Prerequisite: add `zero_based` param defaulting to True and emit `j` vs `j+1` accordingly (tracking issue #1640). |
| REQ-7 (dump value/label formatting) | SHIPPED | `fn dump_svmlight_file in svmlight.rs` formats labels via `format!("{}", y[i])` and values via `format!("{}:{}", j + 1, v)`. Rust `format!("{}", f64)` matches sklearn's text for the tested values: live oracle writes label `1.0`→`1`, value `0.5`→`0.5`, value `1.0/3.0`→`0.3333333333333333`; `rustc`-built `format!("{}", ..)` produces the identical strings (verified: `1 0 0.5 2 0.3333333333333333 2.5`). The numeric formatting is parity; the INDEX BASE is the separable divergence tracked by REQ-6. Non-test consumer: re-exported `pub use svmlight::{dump_svmlight_file ...}` in `lib.rs`. Verification: `round_trip_dense` test green. Underclaim: this is the least-confident SHIPPED — verified only on the finite value set above; an exhaustive float-formatting equivalence (e.g. very large magnitudes, subnormals, scientific-notation thresholds where Rust `{}` and NumPy `repr` may diverge) is NOT proven and should be characterization-tested by the critic before relying on it broadly. |
| REQ-8 (dump comment header + params) | NOT-STARTED | `fn dump_svmlight_file in svmlight.rs` writes only data lines (`buf.push_str(&format!("{}", y[i]))` ... `buf.push('\n')`); there is no `comment`, `query_id`, or `multilabel` parameter and no provenance/`# Column indices are ...-based` header. sklearn writes the header in `_dump_svmlight` (`_svmlight_format_io.py:434-445`) and accepts a file-like `f`. Live oracle with `comment='hello'` prepends 4 `#` lines. Deterministic — oracle-pinnable. Prerequisite: add the `comment`/`query_id`/`multilabel` params + header emission (tracking issue #1640). |
| REQ-9 (load_files Bunch contract) | NOT-STARTED | `fn load_files in svmlight.rs` returns `LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)` — `(docs, labels, target_names)` only. No `Bunch`, no `filenames`, no `DESCR`; no `categories`, `load_content`, `shuffle` (sklearn default True with `random_state`, `_base.py:294-299`), `encoding`, `decode_error`, or `allowed_extensions`. ferrolearn always loads content, never shuffles, sorts dirs/files lexically. sklearn returns a `Bunch` (`_base.py:307-317`). Deterministic for the shuffle/ordering and structural attributes — oracle-pinnable (e.g. default `shuffle=True` permutes order; ferrolearn is deterministic-sorted). Prerequisite: a `Bunch`-equivalent return type + the missing params (tracking issue #1640). |
| REQ-10 (ferray substrate) | NOT-STARTED | `svmlight.rs` uses `ndarray::{Array1, Array2}` (`use ndarray::{Array1, Array2};`) and returns `Array2<f64>` / `Array1<f64>` / `Array1<usize>`. R-SUBSTRATE-1 requires `ferray-core` array types. Prerequisite: migrate the unit's array types to the ferray analog (depends on REQ-3's sparse type for the load return) (tracking issue #1640). |
| REQ-11 (production consumer) | SHIPPED (partial) | `fn load_files in svmlight.rs` has a real non-test production consumer: `ferrolearn-fetch/src/newsgroups.rs` imports `use ferrolearn_datasets::svmlight::{LoadFilesResult, load_files};` and calls it (`load_files(extract_root.join("20news-bydate-train"))`). `load_svmlight_file`/`load_svmlight_files`/`load_svmlight_str`/`dump_svmlight_file` are re-exported at the `ferrolearn-datasets` crate root (`lib.rs`) — public boundary API (grandfathered per R-DEFER-1/S5) but, beyond re-export, have NO in-workspace non-test caller located. Verification: `grep -rn "load_svmlight\|dump_svmlight" --include=*.rs | grep -v test` finds only the re-export. Underclaim: only `load_files` has a confirmed internal production consumer; the svmlight load/dump fns rest on the boundary-API grandfather clause and the eventual `ferrolearn-python` `datasets` binding (not yet present). |

## Architecture

The module is a single file with four public entry points plus one private
parser:

- `fn parse_line in svmlight.rs` — splits one line on whitespace, parses the
  leading label as `f64`, then `idx:val` tokens via `splitn(2, ':')`, converting
  the on-disk 1-based index to a 0-based `(usize, f64)`. It is the sole place the
  1-based assumption is enforced (`if idx_one_based == 0 { ... }`), which is why
  REQ-2 (index-base contract) is concentrated there.
- `fn load_svmlight_str` / `fn load_svmlight_file in svmlight.rs` — accumulate
  parsed rows, infer `max_feat`, validate a caller-supplied `n_features`
  (errors if too small), and densify into `Array2<f64>`. The densification is
  the structural reason REQ-3 (sparse CSR) is NOT-STARTED.
- `fn load_svmlight_files in svmlight.rs` — `paths.iter().map(|p|
  load_svmlight_file(p, n_features))`, i.e. independent per-file inference; this
  is the divergence behind REQ-4. sklearn instead computes a single shared
  `n_f = max(...) + 1` (`_svmlight_format_io.py:410-419`).
- `fn dump_svmlight_file in svmlight.rs` — validates `y.len() == x.nrows()`
  (`FerroError::ShapeMismatch` else), then writes nonzero entries as
  `{j+1}:{v}` (one-based, REQ-6) with `format!("{}", ..)` numeric text
  (REQ-7). No comment header (REQ-8).
- `fn load_files in svmlight.rs` — directory walk: each immediate subdirectory
  is a class, sorted lexically; every regular file is one document. Returns the
  three-tuple `LoadFilesResult`, not a `Bunch` (REQ-9), and never shuffles.

Key types: `type SvmlightDataset = (Array2<f64>, Array1<f64>)` and
`type LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)`. Both are on
the `ndarray` substrate (REQ-10).

Invariant honored: ferrolearn's own dump→load round-trip is internally
consistent (both 1-based). This is NOT scikit-learn parity — sklearn's default
dump is 0-based and its loader keeps 0-based files 0-based, so a ferrolearn dump
read by sklearn (or vice-versa) is off by one column. The `round_trip_dense`
test pins self-consistency, not sklearn conformance (R-HONEST-3).

## Verification

Commands establishing the SHIPPED claims and the oracle deltas behind the
NOT-STARTED rows:

- `cargo test -p ferrolearn-datasets --lib svmlight` → 7 passed, 0 failed
  (`parse_basic_line`, `parse_with_comment`, `round_trip_dense`,
  `declared_n_features_pads`, `declared_n_features_too_small`,
  `shape_mismatch_dump`, `load_files_simple_tree`). Establishes REQ-1, the
  formatting half of REQ-7, and `load_files`'s basic behavior.
- Consumer check: `grep -rn "load_files\|load_svmlight\|dump_svmlight"
  ferrolearn-* --include=*.rs | grep -v '#\[cfg(test'` → `load_files` consumed
  in `ferrolearn-fetch/src/newsgroups.rs`; the svmlight fns only re-exported in
  `ferrolearn-datasets/src/lib.rs` (REQ-11).
- Index-base oracle (REQ-2/REQ-6), live sklearn 1.5.2:
  - load 0-based: `python3 -c "import io; from sklearn.datasets import
    load_svmlight_file; X,_=load_svmlight_file(io.BytesIO(b'1 0:1 2:2\n0 1:3\n'));
    print(X.shape, X.toarray().tolist())"` → `(2, 3) [[1,0,2],[0,3,0]]`
    (ferrolearn errors on the `0:` token).
  - dump default: `python3 -c "import io,numpy as np; from sklearn.datasets
    import dump_svmlight_file; b=io.BytesIO(); dump_svmlight_file(np.array([[1,0,2],[0,3,0],[4,5,6]],float), np.array([1,0,1],float), b); print(b.getvalue().decode())"`
    → `1 0:1 2:2 / 0 1:3 / 1 0:4 1:5 2:6` (ferrolearn emits `1 1:1 3:2 / ...`).
- Formatting oracle (REQ-7): live `dump_svmlight_file` of `[[0.5, 1/3]]`/`[2.5]`
  → `2.5 0:0.5 1:0.3333333333333333`; Rust `format!("{}", v)` produces the same
  value text (verified via a standalone `rustc` probe).
- Comment header oracle (REQ-8): live `dump_svmlight_file(..., comment='hello')`
  emits a 4-line `#` header before data; ferrolearn emits none.

Each NOT-STARTED REQ above is the open work item under tracking issue #1640;
the acto-critic will pin REQ-2, REQ-4, REQ-6, REQ-7-edge, REQ-8, REQ-9 as
failing characterization tests (expected values from the live oracle commands
here, never copied from the ferrolearn side, per R-CHAR-3) and file the specific
`#NNN` blockers.
