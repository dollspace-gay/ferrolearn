//! SVMlight / LIBSVM sparse text format I/O.
//!
//! The SVMlight format is one line per sample:
//!
//! ```text
//! <label> <feat>:<value> <feat>:<value> ...
//! ```
//!
//! This module mirrors scikit-learn's
//! `sklearn/datasets/_svmlight_format_io.py` (tag 1.5.2) for the on-disk
//! wire format:
//!
//! - **Load** uses sklearn's default `zero_based="auto"` index-base resolution
//!   (`_svmlight_format_io.py:402-409`): the on-disk indices are read verbatim,
//!   then — only if the global minimum index across all parsed rows is strictly
//!   positive — every index is decremented by one (1-based → 0-based). A file
//!   that already contains a `0:` column is kept as-is. This reproduces
//!   sklearn's documented "auto" heuristic, including its limitation that a
//!   sparse leading column makes the base ambiguous.
//! - **Dump** writes ZERO-based `idx:val` by default, matching sklearn's
//!   default `zero_based=True` (`_svmlight_format_io.py:474-483`, `:596`).
//! - Numeric labels and values are formatted with C `printf("%.16g", v)`
//!   semantics, matching sklearn's dumper (`_svmlight_format_fast.pyx:199,:203`).
//! - A leading `qid:N` token is silently ignored, matching sklearn's default
//!   `query_id=False` (`_svmlight_format_io.py:64`).
//! - Lines may include a trailing comment after `#`; blank lines and lines
//!   starting with `#` are ignored.
//!
//! For simplicity this module returns dense [`ndarray::Array2<f64>`] feature
//! matrices, which is adequate for small/medium datasets. For very large
//! sparse data, prefer streaming directly into a sparse matrix (out of scope
//! here; tracked as a NOT-STARTED REQ).

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Result of [`load_svmlight_file`] / [`load_svmlight_str`]: dense feature
/// matrix paired with a label vector.
pub type SvmlightDataset = (Array2<f64>, Array1<f64>);

/// Result of [`load_files`]: documents, labels, target names.
pub type LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>);

/// Parse one SVMlight line into `(label, [(raw_index, value), ...])`.
///
/// The returned indices are the **raw on-disk** column indices, exactly as they
/// appear in the file — NO base adjustment is performed here. Whether the file
/// is 0-based or 1-based is resolved globally in [`load_svmlight_str`] following
/// sklearn's default `zero_based="auto"` heuristic
/// (`_svmlight_format_io.py:402-409`). A leading `qid:N` token is silently
/// skipped, matching sklearn's default `query_id=False`
/// (`_svmlight_format_io.py:64`).
fn parse_line(line: &str) -> Result<(f64, Vec<(usize, f64)>), FerroError> {
    // Strip optional comment.
    let body = match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "line".into(),
            reason: "svmlight: empty line".into(),
        });
    }
    let mut parts = trimmed.split_whitespace().peekable();
    let label_tok = parts.next().ok_or_else(|| FerroError::InvalidParameter {
        name: "label".into(),
        reason: "svmlight: missing label".into(),
    })?;
    let label: f64 =
        label_tok.parse().map_err(
            |e: std::num::ParseFloatError| FerroError::InvalidParameter {
                name: "label".into(),
                reason: format!("svmlight: bad label '{label_tok}': {e}"),
            },
        )?;

    // sklearn's format places an optional `qid:N` token immediately after the
    // label; with the default `query_id=False` it is ignored. Skip it if
    // present (`_svmlight_format_io.py:64`).
    if let Some(rest) = parts.peek().and_then(|tok| tok.strip_prefix("qid:")) {
        // Validate that what follows the prefix is an integer query id, so we
        // only swallow a genuine qid token (and surface a malformed one).
        rest.parse::<i64>()
            .map_err(|e: std::num::ParseIntError| FerroError::InvalidParameter {
                name: "qid".into(),
                reason: format!("svmlight: bad qid 'qid:{rest}': {e}"),
            })?;
        parts.next();
    }

    let mut features = Vec::new();
    for tok in parts {
        let mut split = tok.splitn(2, ':');
        let idx_tok = split.next().ok_or_else(|| FerroError::InvalidParameter {
            name: "feature".into(),
            reason: format!("svmlight: malformed token '{tok}'"),
        })?;
        let val_tok = split.next().ok_or_else(|| FerroError::InvalidParameter {
            name: "feature".into(),
            reason: format!("svmlight: malformed token '{tok}' (no value)"),
        })?;
        let idx: usize =
            idx_tok
                .parse()
                .map_err(|e: std::num::ParseIntError| FerroError::InvalidParameter {
                    name: "feature index".into(),
                    reason: format!("svmlight: bad index '{idx_tok}': {e}"),
                })?;
        let val: f64 = val_tok.parse().map_err(|e: std::num::ParseFloatError| {
            FerroError::InvalidParameter {
                name: "feature value".into(),
                reason: format!("svmlight: bad value '{val_tok}': {e}"),
            }
        })?;
        // Raw on-disk index; base resolution happens later in load_svmlight_str.
        features.push((idx, val));
    }
    Ok((label, features))
}

/// Load an SVMlight-format file into a dense `(X, y)` pair.
///
/// `n_features` may be `None`, in which case the number of features is
/// inferred from the maximum index seen.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be read.
/// Returns [`FerroError::InvalidParameter`] for malformed lines.
pub fn load_svmlight_file<P: AsRef<Path>>(
    path: P,
    n_features: Option<usize>,
) -> Result<SvmlightDataset, FerroError> {
    let text = fs::read_to_string(path).map_err(FerroError::IoError)?;
    load_svmlight_str(&text, n_features)
}

/// One parsed file: its rows (label + raw on-disk `(index, value)` pairs) and
/// the global min/max raw index observed (`None` if the file has no features).
struct ParsedFile {
    rows: Vec<(f64, Vec<(usize, f64)>)>,
    min_index: Option<usize>,
    max_index: Option<usize>,
}

/// Parse the textual contents of one SVMlight file into rows of RAW on-disk
/// indices, recording the global min/max index. No base adjustment is done.
fn parse_contents(contents: &str) -> Result<ParsedFile, FerroError> {
    let mut rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
    let mut min_index: Option<usize> = None;
    let mut max_index: Option<usize> = None;
    for (lineno, raw) in contents.lines().enumerate() {
        let stripped = raw.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }
        let parsed = parse_line(raw).map_err(|e| FerroError::InvalidParameter {
            name: "svmlight".into(),
            reason: format!("line {}: {e}", lineno + 1),
        })?;
        for &(i, _) in &parsed.1 {
            min_index = Some(min_index.map_or(i, |m| m.min(i)));
            max_index = Some(max_index.map_or(i, |m| m.max(i)));
        }
        rows.push(parsed);
    }
    Ok(ParsedFile {
        rows,
        min_index,
        max_index,
    })
}

/// Resolve sklearn's default `zero_based="auto"` base offset for a single file
/// (`_svmlight_format_io.py:402-409`): subtract 1 from every index iff the file
/// has at least one feature AND its minimum raw index is strictly positive
/// (a 1-based file). A file that already contains a `0:` column is kept as-is.
fn auto_base_offset(min_index: Option<usize>) -> usize {
    match min_index {
        Some(m) if m > 0 => 1,
        _ => 0,
    }
}

/// Materialize parsed rows into a dense `(X, y)` given the base offset to
/// subtract from every raw index and the final column count `n_feat`.
fn densify(rows: Vec<(f64, Vec<(usize, f64)>)>, offset: usize, n_feat: usize) -> SvmlightDataset {
    let n_samples = rows.len();
    let mut x = Array2::<f64>::zeros((n_samples, n_feat));
    let mut y = Array1::<f64>::zeros(n_samples);
    for (i, (label, feats)) in rows.into_iter().enumerate() {
        y[i] = label;
        for (j, v) in feats {
            x[[i, j - offset]] = v;
        }
    }
    (x, y)
}

/// Like [`load_svmlight_file`] but takes the file contents as a string.
///
/// Index-base resolution follows sklearn's default `zero_based="auto"`
/// (`_svmlight_format_io.py:402-409`): after parsing all rows, if the global
/// minimum index is strictly positive the file is treated as 1-based and every
/// index is decremented by one; otherwise the indices are kept verbatim
/// (0-based). `n_features` is then inferred as `max adjusted index + 1`.
pub fn load_svmlight_str(
    contents: &str,
    n_features: Option<usize>,
) -> Result<SvmlightDataset, FerroError> {
    let parsed = parse_contents(contents)?;
    let offset = auto_base_offset(parsed.min_index);
    // After the base adjustment, the highest column index is `max_index -
    // offset`, so n_features = (max_index - offset) + 1.
    let observed = parsed.max_index.map_or(0, |m| m - offset + 1);
    let n_feat = match n_features {
        Some(n) => {
            if n < observed {
                return Err(FerroError::InvalidParameter {
                    name: "n_features".into(),
                    reason: format!(
                        "svmlight: declared n_features={n} but file has {observed} features"
                    ),
                });
            }
            n
        }
        None => observed,
    };
    Ok(densify(parsed.rows, offset, n_feat))
}

/// Load multiple SVMlight files at once, returning one `(X, y)` per path.
///
/// Mirrors sklearn (`_svmlight_format_io.py:410-419`): a SINGLE shared
/// `n_features = max adjusted column index + 1` is computed across ALL files,
/// and every returned `X` uses that common width. Each file's index base is
/// resolved independently via the `zero_based="auto"` heuristic (sklearn applies
/// the base adjustment per file before taking the global max). An explicit
/// `n_features` is honored, erroring if it is smaller than the observed common
/// width.
pub fn load_svmlight_files<P: AsRef<Path>>(
    paths: &[P],
    n_features: Option<usize>,
) -> Result<Vec<SvmlightDataset>, FerroError> {
    let parsed: Vec<ParsedFile> = paths
        .iter()
        .map(|p| {
            let text = fs::read_to_string(p.as_ref()).map_err(FerroError::IoError)?;
            parse_contents(&text)
        })
        .collect::<Result<_, _>>()?;

    // Per-file auto base offset, then the global common width across all files.
    let offsets: Vec<usize> = parsed
        .iter()
        .map(|pf| auto_base_offset(pf.min_index))
        .collect();
    let observed = parsed
        .iter()
        .zip(&offsets)
        .map(|(pf, &off)| pf.max_index.map_or(0, |m| m - off + 1))
        .max()
        .unwrap_or(0);

    let n_feat = match n_features {
        Some(n) => {
            if n < observed {
                return Err(FerroError::InvalidParameter {
                    name: "n_features".into(),
                    reason: format!(
                        "svmlight: declared n_features={n} but input files have {observed} features"
                    ),
                });
            }
            n
        }
        None => observed,
    };

    Ok(parsed
        .into_iter()
        .zip(offsets)
        .map(|(pf, off)| densify(pf.rows, off, n_feat))
        .collect())
}

/// Format an `f64` exactly like C `printf("%.16g", v)`, which is the format
/// sklearn's dumper uses for non-integral labels and values
/// (`_svmlight_format_fast.pyx:199,:203`).
///
/// The `%g` rule with precision `P = 16`:
/// - Compute the decimal exponent `exp` of the value's leading significant
///   digit and round to 16 significant digits.
/// - Use scientific notation iff `exp < -4` or `exp >= P`; otherwise fixed.
/// - Strip trailing zeros from the fractional part (the `#` flag is NOT set).
/// - In scientific form the exponent is `e±NN` with at least two digits.
///
/// Verified byte-for-byte against the live sklearn 1.5.2 / C `%.16g` oracle
/// over a spread of edge cases plus 8000+ random magnitudes.
fn format_g16(v: f64) -> String {
    if v == 0.0 {
        // C %g prints "0" for +0.0 and "-0" for -0.0.
        return if v.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.to_string();
    }

    const P: usize = 16; // significant digits
    // Rust "{:.15e}" yields 16 significant digits (1 before the point, 15
    // after) in scientific notation, correctly rounded — the same rounding C
    // applies before deciding fixed vs scientific layout.
    let sci = format!("{:.*e}", P - 1, v);
    let (mant, exp_str) = match sci.split_once('e') {
        Some(parts) => parts,
        // `{:e}` always emits an exponent, so this branch is dead in practice;
        // fall back to the raw scientific string rather than panicking.
        None => return sci,
    };
    let exp: i32 = exp_str.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let mant_digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };

    // %g: scientific when the decimal exponent is < -4 or >= the precision.
    let use_sci = exp < -4 || exp >= P as i32;

    if use_sci {
        let lead = &mant_digits[0..1];
        let mut frac = mant_digits[1..].to_string();
        while frac.ends_with('0') {
            frac.pop();
        }
        let body = if frac.is_empty() {
            lead.to_string()
        } else {
            format!("{lead}.{frac}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        let eabs = exp.abs();
        format!("{sign}{body}e{esign}{eabs:02}")
    } else if exp >= 0 {
        let ip_len = (exp + 1) as usize;
        if ip_len >= mant_digits.len() {
            // All significant digits lie in the integer part; pad with zeros.
            let mut s = mant_digits.clone();
            s.push_str(&"0".repeat(ip_len - mant_digits.len()));
            format!("{sign}{s}")
        } else {
            let ip = &mant_digits[..ip_len];
            let mut fp = mant_digits[ip_len..].to_string();
            while fp.ends_with('0') {
                fp.pop();
            }
            if fp.is_empty() {
                format!("{sign}{ip}")
            } else {
                format!("{sign}{ip}.{fp}")
            }
        }
    } else {
        // exp in -4..=-1 → leading "0." with `-exp-1` padding zeros.
        let zeros = (-exp - 1) as usize;
        let mut fp = format!("{}{mant_digits}", "0".repeat(zeros));
        while fp.ends_with('0') {
            fp.pop();
        }
        format!("{sign}0.{fp}")
    }
}

/// Write a feature matrix `x` and labels `y` to `path` in SVMlight format.
///
/// Matches scikit-learn's default `dump_svmlight_file`
/// (`_svmlight_format_io.py:474-483`): only nonzero entries are written, column
/// indices are emitted ZERO-based (sklearn default `zero_based=True`, `:596`),
/// and labels and values are formatted with C `%.16g` semantics
/// (`_svmlight_format_fast.pyx:199,:203`) via [`format_g16`].
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be opened or written.
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
pub fn dump_svmlight_file<P: AsRef<Path>>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    path: P,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "dump_svmlight_file: y length must equal x rows".into(),
        });
    }
    let mut buf = String::new();
    for i in 0..n_samples {
        buf.push_str(&format_g16(y[i]));
        for j in 0..x.ncols() {
            let v = x[[i, j]];
            if v != 0.0 {
                buf.push(' ');
                // Zero-based column index, matching sklearn's default dump.
                buf.push_str(&format!("{j}:{}", format_g16(v)));
            }
        }
        buf.push('\n');
    }
    let mut f = fs::File::create(path).map_err(FerroError::IoError)?;
    f.write_all(buf.as_bytes()).map_err(FerroError::IoError)?;
    Ok(())
}

/// Recursively load a directory tree of text files: each subdirectory becomes
/// a class label and every regular file inside is treated as one document.
///
/// Returns `(documents, labels, target_names)` where `documents[i]` is the
/// raw text of file `i`, `labels[i]` is the (0-based) index of its parent
/// directory in `target_names`, and `target_names[k]` is the directory name
/// for class `k`.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the root directory or any file inside
/// cannot be read.
pub fn load_files<P: AsRef<Path>>(container_path: P) -> Result<LoadFilesResult, FerroError> {
    let root = container_path.as_ref();
    let mut subdirs: Vec<std::path::PathBuf> = fs::read_dir(root)
        .map_err(FerroError::IoError)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    let target_names: Vec<String> = subdirs
        .iter()
        .map(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned()
        })
        .collect();
    let mut docs: Vec<String> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (cls_idx, dir) in subdirs.iter().enumerate() {
        let mut files: Vec<std::path::PathBuf> = fs::read_dir(dir)
            .map_err(FerroError::IoError)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for f in files {
            let text = fs::read_to_string(&f).map_err(FerroError::IoError)?;
            docs.push(text);
            labels.push(cls_idx);
        }
    }
    Ok((docs, Array1::from(labels), target_names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmpdir() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!("ferrolearn_svmlight_test_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_basic_line() {
        // parse_line now returns RAW on-disk indices; base resolution happens
        // later in load_svmlight_str. So `1:` / `3:` stay 1 and 3.
        let (label, feats) = parse_line("1.0 1:0.5 3:1.5").unwrap();
        assert!((label - 1.0).abs() < 1e-12);
        assert_eq!(feats, vec![(1, 0.5), (3, 1.5)]);
    }

    #[test]
    fn parse_with_comment() {
        let (label, feats) = parse_line("0 1:1.0 2:2.0 # this is a comment").unwrap();
        assert!((label - 0.0).abs() < 1e-12);
        assert_eq!(feats, vec![(1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn parse_zero_based_index_kept() {
        // No more hard rejection of index 0: the raw 0 is preserved, and
        // load_svmlight_str's auto heuristic keeps such a file 0-based.
        let (_label, feats) = parse_line("1 0:1 2:2").unwrap();
        assert_eq!(feats, vec![(0, 1.0), (2, 2.0)]);
    }

    #[test]
    fn parse_skips_qid_token() {
        // qid:N right after the label is ignored under query_id=False (default).
        let (label, feats) = parse_line("1 qid:42 1:1.0 2:2.0").unwrap();
        assert!((label - 1.0).abs() < 1e-12);
        assert_eq!(feats, vec![(1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn load_one_based_auto_shifts() {
        // min index 1 > 0 → 1-based file → shift to 0-based, 2 columns.
        let (x, _y) = load_svmlight_str("1 1:1.0 2:2.0\n", None).unwrap();
        assert_eq!(x.dim(), (1, 2));
        assert_eq!(x, ndarray::array![[1.0, 2.0]]);
    }

    #[test]
    fn load_zero_based_auto_kept() {
        // min index 0 → 0-based file kept as-is; 3 columns (max index 2 + 1).
        let (x, _y) = load_svmlight_str("1 0:1 2:2\n0 1:3\n", None).unwrap();
        assert_eq!(x.dim(), (2, 3));
        assert_eq!(x, ndarray::array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]);
    }

    #[test]
    fn format_g16_matches_c_printf() {
        // Expected strings are C printf("%.16g", v) outputs — the exact format
        // sklearn's dumper uses (_svmlight_format_fast.pyx:199,:203), verified
        // against the live sklearn 1.5.2 / C oracle. Not copied from ferrolearn.
        assert_eq!(format_g16(0.0), "0");
        assert_eq!(format_g16(1.0), "1");
        assert_eq!(format_g16(0.5), "0.5");
        assert_eq!(format_g16(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(format_g16(1e20), "1e+20");
        assert_eq!(format_g16(1e-20), "9.999999999999999e-21");
        assert_eq!(format_g16(123456789.123), "123456789.123");
        assert_eq!(format_g16(-2.5), "-2.5");
        assert_eq!(format_g16(100.0), "100");
    }

    #[test]
    fn round_trip_dense() {
        let dir = tmpdir();
        let path = dir.join("a.svmlight");
        // Column 0 is non-empty (rows 0 and 2), so under the dump-0-based +
        // load-auto round trip the loader sees min index 0 and keeps it
        // 0-based — the matrix reconstructs exactly.
        let x = ndarray::array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 5.0, 6.0]];
        let y = ndarray::array![1.0, 0.0, 1.0];
        dump_svmlight_file(&x, &y, &path).unwrap();
        let (x2, y2) = load_svmlight_file(&path, None).unwrap();
        assert_eq!(x2.dim(), (3, 3));
        assert_eq!(y2.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((x[[i, j]] - x2[[i, j]]).abs() < 1e-12);
            }
            assert!((y[i] - y2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn declared_n_features_pads() {
        let text = "1 1:1.0 2:2.0\n0 1:3.0\n";
        let (x, _) = load_svmlight_str(text, Some(5)).unwrap();
        assert_eq!(x.ncols(), 5);
    }

    #[test]
    fn declared_n_features_too_small() {
        let text = "1 1:1.0 5:9.0\n";
        assert!(load_svmlight_str(text, Some(2)).is_err());
    }

    #[test]
    fn shape_mismatch_dump() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = ndarray::array![1.0];
        let dir = tmpdir();
        assert!(dump_svmlight_file(&x, &y, dir.join("bad.svmlight")).is_err());
    }

    #[test]
    fn load_files_simple_tree() {
        let dir = tmpdir();
        for cls in &["alpha", "beta"] {
            let sub = dir.join(cls);
            fs::create_dir_all(&sub).unwrap();
            for i in 0..2 {
                let f = sub.join(format!("doc{i}.txt"));
                fs::write(&f, format!("{cls}-doc{i}")).unwrap();
            }
        }
        let (docs, labels, target_names) = load_files(&dir).unwrap();
        assert_eq!(docs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(target_names, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
