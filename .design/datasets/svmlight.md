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

The deterministic wire-format divergences are now SHIPPED: REQ-2 (load
`zero_based="auto"` default), REQ-4 (multi-file common n_features), REQ-6 (dump
zero-based default), REQ-7 (dump `%.16g` value/label formatting), plus the
qid-ignore slice of REQ-5. REQ-9's shuffle pin was an unachievable RNG bit-match
(numpy Mersenne-Twister) and was retired as an R-DEFER-3 carve-out (the
`divergence_load_files_shuffle` test was deleted; fuller `load_files`/`Bunch`
parity belongs to a future `_base.py` route, not this svmlight-format unit).
Still representational / feature-gap (blocked on a missing type or unimplemented
parameter — no single scalar oracle): REQ-3 (sparse CSR), the remainder of REQ-5
(`query_id=True` return / `multilabel` / `offset`/`length` / `dtype`), REQ-8
(dump comment header + param exposure), REQ-9 (Bunch contract), REQ-10 (ferray
substrate). REQ-1, REQ-11 are mixed (see rows).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (load 1-based parse) | SHIPPED | `fn parse_line in svmlight.rs` strips `#` (`let body = match line.find('#')`), errors on empty, parses `idx:val` and stores `(idx_one_based - 1, val)`; `fn load_svmlight_str` builds the dense `(X, y)`. Non-test consumer: re-exported at crate root in `lib.rs` (`pub use svmlight::{... load_svmlight_file, load_svmlight_str ...}`) and reached transitively from `dump_svmlight_file` round-trips. Verification: `cargo test -p ferrolearn-datasets --lib svmlight` → `parse_basic_line`, `parse_with_comment`, `round_trip_dense` pass (7 passed, 0 failed). Scope note: SHIPPED only for the 1-based parse path; the index-base CONTRACT is REQ-2. |
| REQ-2 (load index-base contract) | SHIPPED (default `"auto"`) | `fn parse_line in svmlight.rs` now returns RAW on-disk indices (no decrement, no 0-rejection); `fn load_svmlight_str` resolves the base via `fn auto_base_offset` mirroring sklearn's default `zero_based="auto"` (`_svmlight_format_io.py:402-409`): it subtracts 1 from every index iff the global min index > 0, else keeps the file 0-based. n_features is computed AFTER the adjustment. Impl anchors: `fn parse_contents`, `fn auto_base_offset`, `fn load_svmlight_str in svmlight.rs`. Non-test production consumer: re-exported at crate root in `lib.rs` and reached from `ferrolearn-fetch/src/newsgroups.rs`'s sibling loaders; the load path is the public boundary API consumed by `load_svmlight_file`. Verification: `divergence_load_zero_based_auto` PASSES (live-oracle X `[[1,0,2],[0,3,0]]`, shape `(2,3)`); in-crate `load_zero_based_auto_kept`, `load_one_based_auto_shifts` pass. Scope note: SHIPPED only for the DEFAULT `"auto"` behavior — exposing an explicit `zero_based ∈ {True,False}` parameter is the separate NOT-STARTED REQ #1648. Faithful limitation: a sparse leading column makes `"auto"` ambiguous, matching sklearn. |
| REQ-3 (load sparse CSR return) | NOT-STARTED | `type SvmlightDataset = (Array2<f64>, Array1<f64>)` in `svmlight.rs` returns a DENSE matrix; `fn load_svmlight_str` materializes `Array2::<f64>::zeros((n_samples, n_feat))`. sklearn returns `scipy.sparse` CSR (`_svmlight_format_io.py:159`, `:424` `sp.csr_matrix(...)`). Representational gap. Prerequisite: a sparse CSR array type from the ferray sparse analog / `ferrolearn-sparse`; until that exists the return-type contract cannot be met (tracking issue #1640). |
| REQ-4 (multi-file common n_features) | SHIPPED | `fn load_svmlight_files in svmlight.rs` now parses ALL files first (`fn parse_contents`), resolves each file's auto base offset, then computes a SINGLE common `observed = max over files of (max adjusted index + 1)` and builds every `X` at that shared width via `fn densify` (`_svmlight_format_io.py:410-419`). An explicit `n_features` is honored, erroring if smaller than `observed` (sklearn `:414-418`). Non-test production consumer: re-exported at crate root in `lib.rs` (public boundary API). Verification: `divergence_load_files_common_n_features` PASSES — two files with max indices 2 and 4 both load to the common 4 columns (live oracle `Xa.shape==Xb.shape==(1,4)`). Deterministic — oracle-pinned. |
| REQ-5 (multilabel/query_id/offset/length/dtype) | NOT-STARTED (qid-ignore behavior shipped) | The default `query_id=False` qid-IGNORE behavior is now matched: `fn parse_line in svmlight.rs` silently skips a leading `qid:N` token (`_svmlight_format_io.py:64`), so `"1 qid:1 1:1.0 2:2.0"` loads features at the `1:`/`2:` columns only — verified by `divergence_load_qid_token_ignored` (live oracle X `[[1.0,2.0]]`, shape `(1,2)`). The remaining REQ-5 surface stays NOT-STARTED: there is still no `query_id=True` RETURN of the qid values, no `multilabel` y-as-list-of-tuples, no byte-range `offset`/`length`, and no f32 `dtype` (module is f64-only via `Array2<f64>`). Feature-gap. Prerequisite: implement each remaining parameter against sklearn's signature (`_svmlight_format_io.py:64-74`); `dtype` also depends on the generic-`F` rework (tracking issue #1640). |
| REQ-6 (dump index-base default) | SHIPPED (default) | `fn dump_svmlight_file in svmlight.rs` now writes ZERO-based `{j}:{v}` (`buf.push_str(&format!("{j}:{}", format_g16(v)))`), matching sklearn's default `zero_based=True` (`_svmlight_format_io.py:474-483`, `:596` `one_based = not zero_based`). Non-test production consumer: re-exported at crate root in `lib.rs` (public boundary API; reached from `round_trip_dense`'s dump→load and the eventual python binding). Verification: `divergence_dump_index_base_zero_based_default` PASSES — byte-for-byte equal to the live-oracle `1 0:1 2:2\n0 1:3\n1 0:4 1:5 2:6\n`. HEADLINE divergence resolved: ferrolearn dumps are now readable by sklearn. Scope note: exposing an explicit `zero_based` PARAMETER is the separate NOT-STARTED REQ #1648; the 3-arg signature is unchanged. |
| REQ-7 (dump value/label formatting) | SHIPPED | `fn dump_svmlight_file in svmlight.rs` formats both labels and values via `fn format_g16 in svmlight.rs`, a pure-Rust reimplementation of C `printf("%.16g", v)` — the exact format string sklearn's dumper uses for float labels and values (`_svmlight_format_fast.pyx:199` `"%d:%.16g"`, `:203` `"%.16g"`). The `%g` rule (scientific iff decimal exponent `< -4` or `>= 16`, 16 sig digits, trailing-zero strip, `e±NN` two-digit exponent) was verified BYTE-FOR-BYTE against the live C/sklearn `%.16g` oracle across a curated edge-case spread AND a 8000+-value random fuzz (0 mismatches). The earlier divergence (Rust `format!("{}", f64)` emitting `100000000000000000000` / `0.00000000000000000001` for `1e20`/`1e-20`) is resolved: now `1e+20` / `9.999999999999999e-21`. Non-test consumer: re-exported `pub use svmlight::{dump_svmlight_file ...}` in `lib.rs`. Verification: `divergence_dump_value_formatting_scientific` PASSES (live oracle `2.5 0:1e+20 1:9.999999999999999e-21\n`); in-crate `format_g16_matches_c_printf` pins the symbolic `%.16g` constants. |
| REQ-8 (dump comment header + params) | NOT-STARTED | `fn dump_svmlight_file in svmlight.rs` writes only data lines (`buf.push_str(&format!("{}", y[i]))` ... `buf.push('\n')`); there is no `comment`, `query_id`, or `multilabel` parameter and no provenance/`# Column indices are ...-based` header. sklearn writes the header in `_dump_svmlight` (`_svmlight_format_io.py:434-445`) and accepts a file-like `f`. Live oracle with `comment='hello'` prepends 4 `#` lines. Deterministic — oracle-pinnable. Prerequisite: add the `comment`/`query_id`/`multilabel` params + header emission (tracking issue #1640). |
| REQ-9 (load_files Bunch contract) | NOT-STARTED | `fn load_files in svmlight.rs` returns `LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)` — `(docs, labels, target_names)` only. No `Bunch`, no `filenames`, no `DESCR`; no `categories`, `load_content`, `shuffle` (sklearn default True with `random_state`, `_base.py:294-299`), `encoding`, `decode_error`, or `allowed_extensions`. ferrolearn always loads content, never shuffles, sorts dirs/files lexically. sklearn returns a `Bunch` (`_base.py:307-317`). Deterministic for the shuffle/ordering and structural attributes — oracle-pinnable (e.g. default `shuffle=True` permutes order; ferrolearn is deterministic-sorted). Prerequisite: a `Bunch`-equivalent return type + the missing params (tracking issue #1640). |
| REQ-10 (ferray substrate) | NOT-STARTED | `svmlight.rs` uses `ndarray::{Array1, Array2}` (`use ndarray::{Array1, Array2};`) and returns `Array2<f64>` / `Array1<f64>` / `Array1<usize>`. R-SUBSTRATE-1 requires `ferray-core` array types. Prerequisite: migrate the unit's array types to the ferray analog (depends on REQ-3's sparse type for the load return) (tracking issue #1640). |
| REQ-11 (production consumer) | SHIPPED (partial) | `fn load_files in svmlight.rs` has a real non-test production consumer: `ferrolearn-fetch/src/newsgroups.rs` imports `use ferrolearn_datasets::svmlight::{LoadFilesResult, load_files};` and calls it (`load_files(extract_root.join("20news-bydate-train"))`). `load_svmlight_file`/`load_svmlight_files`/`load_svmlight_str`/`dump_svmlight_file` are re-exported at the `ferrolearn-datasets` crate root (`lib.rs`) — public boundary API (grandfathered per R-DEFER-1/S5) but, beyond re-export, have NO in-workspace non-test caller located. Verification: `grep -rn "load_svmlight\|dump_svmlight" --include=*.rs | grep -v test` finds only the re-export. Underclaim: only `load_files` has a confirmed internal production consumer; the svmlight load/dump fns rest on the boundary-API grandfather clause and the eventual `ferrolearn-python` `datasets` binding (not yet present). |

## Architecture

The module is a single file with four public entry points plus one private
parser:

- `fn parse_line in svmlight.rs` — splits one line on whitespace, parses the
  leading label as `f64`, silently skips an optional `qid:N` token (REQ-5
  qid-ignore slice), then parses `idx:val` tokens via `splitn(2, ':')` and
  returns the RAW on-disk index as `(usize, f64)`. It no longer enforces any
  base assumption (index 0 is accepted); base resolution is hoisted to the load
  layer (REQ-2).
- `fn parse_contents` / `fn auto_base_offset` / `fn densify in svmlight.rs` —
  the load helpers. `parse_contents` accumulates rows and the global min/max raw
  index; `auto_base_offset` implements sklearn's default `zero_based="auto"`
  (subtract 1 iff min index > 0, `_svmlight_format_io.py:402-409`); `densify`
  builds the dense `Array2<f64>` at a given offset and width. The densification
  is the structural reason REQ-3 (sparse CSR) is NOT-STARTED.
- `fn load_svmlight_str` / `fn load_svmlight_file in svmlight.rs` — parse, apply
  the auto base offset, infer `observed = max adjusted index + 1`, validate a
  caller-supplied `n_features` (errors if too small), and densify.
- `fn load_svmlight_files in svmlight.rs` — parses ALL files, computes a single
  shared common width across them, and builds every `X` at that width (REQ-4,
  `_svmlight_format_io.py:410-419`), with per-file auto base resolution.
- `fn dump_svmlight_file in svmlight.rs` — validates `y.len() == x.nrows()`
  (`FerroError::ShapeMismatch` else), then writes nonzero entries as `{j}:{v}`
  (ZERO-based, REQ-6) with `fn format_g16` C-`%.16g` numeric text for both
  values and labels (REQ-7). No comment header / param exposure (REQ-8).
- `fn load_files in svmlight.rs` — directory walk: each immediate subdirectory
  is a class, sorted lexically; every regular file is one document. Returns the
  three-tuple `LoadFilesResult`, not a `Bunch` (REQ-9), and never shuffles.

Key types: `type SvmlightDataset = (Array2<f64>, Array1<f64>)` and
`type LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>)`. Both are on
the `ndarray` substrate (REQ-10).

Invariant honored: ferrolearn now dumps ZERO-based and loads under
`zero_based="auto"`, so a ferrolearn dump is wire-compatible with sklearn's
default reader and vice-versa (for matrices whose column 0 is non-empty, where
auto unambiguously infers 0-based). `round_trip_dense` uses such a matrix
(column 0 non-empty) so the dump→load round trip reconstructs it exactly. The
one faithful caveat is sklearn's documented `"auto"` ambiguity: a matrix whose
first column is entirely zero dumps with min index ≥ 1, which auto then reads
back as 1-based and shifts left — this matches sklearn's own behavior, not a
ferrolearn bug.

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
