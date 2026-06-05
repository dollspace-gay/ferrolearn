//! `fetch_openml` — generic OpenML.org dataset client.
//!
//! sklearn's full `fetch_openml` is large (handles parquet + ARFF, the
//! classification-vs-regression task split, version filtering, etc.). This
//! implementation covers the most common path:
//!
//! 1. Look up the dataset metadata via `https://www.openml.org/api/v1/json/data/{id}`.
//! 2. Download the file at the URL the metadata returns.
//! 3. If the file is ARFF, parse the simple-but-common subset (numeric +
//!    nominal attributes) into a feature matrix + target vector.
//!
//! Callers that need parquet, sparse matrices, or the full type-coercion
//! machinery should download the file via [`crate::fetch_file`] and parse it
//! with their own ARFF / parquet reader.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.fetch_openml` (`sklearn/datasets/_openml.py` +
//! `_arff_parser.py`; live oracle 1.5.2). Verification model A (cargo test +
//! sklearn source contract; full numerical value-parity needs a live OpenML
//! download, so it is NOT-STARTED offline — the SHIPPED rows are the
//! offline-verifiable contract: API endpoints, ARFF parsing, cache integration,
//! public surface). Design doc: `.design/datasets/openml.md` (8 REQs). Every REQ
//! is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//!
//! **5 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-OPENML-API-ENDPOINTS | SHIPPED | `DATA_INFO_URL` + `fetch_openml` build `api/v1/json/data/{id}` and read `data_set_description.url`/`default_target_attribute`, matching sklearn `_DATA_INFO` (`_openml.py:34`) + `_get_data_description_by_id` (`:367`). (Host `www.openml.org` vs sklearn `api.openml.org` + download-route delta folded into REQ-VALUE-PARITY.) |
//! | REQ-ARFF-PARSE | SHIPPED | `parse_arff`/`parse_attribute`/`split_attribute_name` classify numeric vs nominal `{a,b,c}` attributes, 0-based level-encode nominal values, and unquote single-quoted names — mirroring sklearn's LIAC-ARFF `encode_nominal` path (`_arff_parser.py:167-179`). 4 in-crate unit tests pass. (Sparse/pandas/string/date ARFF paths NOT covered — REQ-FULL-PARAMS.) |
//! | REQ-CACHE-INTEGRATION | SHIPPED | `fetch_openml` caches via `dataset_dir("openml/{id}", data_home)` (`cache.rs`) + `fetch_file` first-use download, mirroring sklearn's `cache=True` `data_home` keying (`_openml.py:43-44`). (Uncompressed `openml/{id}/` vs sklearn gzip `openml.org/{path}.gz` — folded into REQ-VALUE-PARITY.) |
//! | REQ-CODE-HYGIENE | SHIPPED | FIXED #2062: `split_attribute_name`'s nested `if let { if let }` (which rust-1.95 clippy flagged as `collapsible_if` under `-D warnings`, R-DEFER-5 toolchain drift) collapsed into a single let-chain; `cargo clippy -p ferrolearn-fetch --all-targets -- -D warnings` now green. No behavior change. |
//! | REQ-CONSUMER | SHIPPED | `pub fn fetch_openml` + `pub struct OpenmlDataset` re-exported at the crate root (`lib.rs` `pub use openml::{OpenmlDataset, fetch_openml}`) — public boundary API (R-DEFER-1/S5; external users + the eventual ferrolearn-python datasets binding). Underclaim: no in-workspace non-test function caller. |
//! | REQ-VALUE-PARITY | NOT-STARTED | `fetch_openml` → `fetch_file` → `ureq::get(url).call()` needs network + a live OpenML download; element-wise `(data, target)` parity vs `sklearn.datasets.fetch_openml(data_id=61)` is unreachable offline (network-gated, NOT a code bug). Blocker #2060. |
//! | REQ-FULL-PARAMS | NOT-STARTED | signature `fetch_openml(data_id, target_column, data_home)` omits sklearn's `name`/`version` resolution, list/None `target_column`, `cache`/`return_X_y`/`as_frame`/`n_retries`/`delay`/`parser`/`read_csv_kwargs` (`_openml.py:770-784`). Blocker #2061. |
//! | REQ-SUBSTRATE | NOT-STARTED | uses `ndarray::{Array1, Array2}` for `OpenmlDataset`, not `ferray-core` (R-SUBSTRATE-1); shares the sibling fetchers' `ndarray` types. Tracked under #2060's migration follow-up. |

use std::fs;
use std::path::Path;

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use serde_json::Value;

use crate::cache::dataset_dir;
use crate::fetch::fetch_file;

const DATA_INFO_URL: &str = "https://www.openml.org/api/v1/json/data/";

/// Returned OpenML dataset (numeric ARFF subset).
#[derive(Debug, Clone)]
pub struct OpenmlDataset {
    /// Feature matrix (nominal attributes encoded as 0-based indices into
    /// [`Self::nominal_levels`]).
    pub data: Array2<f64>,
    /// Target column (parsed as f64; for nominal targets the index into
    /// [`Self::target_levels`]).
    pub target: Array1<f64>,
    /// Names of feature columns in order.
    pub feature_names: Vec<String>,
    /// Name of the target column.
    pub target_name: String,
    /// For each feature column, the distinct nominal levels (empty for
    /// numeric columns).
    pub nominal_levels: Vec<Vec<String>>,
    /// Distinct levels of a nominal target (empty for numeric targets).
    pub target_levels: Vec<String>,
}

/// Fetch an OpenML dataset by numeric ID.
///
/// `target_column` may be `None` — in that case we use the dataset's
/// default `default_target_attribute` from the OpenML metadata.
pub fn fetch_openml(
    data_id: u64,
    target_column: Option<&str>,
    data_home: Option<&Path>,
) -> Result<OpenmlDataset, FerroError> {
    let dir = dataset_dir(&format!("openml/{data_id}"), data_home)?;
    let info_path = fetch_file(
        &format!("{DATA_INFO_URL}{data_id}"),
        "data_info.json",
        None,
        &dir,
    )?;
    let info_text = fs::read_to_string(&info_path).map_err(FerroError::IoError)?;
    let info: Value = serde_json::from_str(&info_text).map_err(|e| FerroError::SerdeError {
        message: format!("openml: failed to parse data info JSON: {e}"),
    })?;
    let data = info
        .get("data_set_description")
        .ok_or_else(|| FerroError::SerdeError {
            message: "openml: missing 'data_set_description'".into(),
        })?;
    let url = data
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| FerroError::SerdeError {
            message: "openml: dataset description missing 'url'".into(),
        })?;
    let default_target = data
        .get("default_target_attribute")
        .and_then(Value::as_str)
        .map(str::to_string);
    let target_name = target_column
        .map(str::to_string)
        .or(default_target)
        .ok_or_else(|| FerroError::SerdeError {
            message:
                "openml: no target_column supplied and no default_target_attribute in metadata"
                    .into(),
        })?;

    let arff_path = fetch_file(url, "data.arff", None, &dir)?;
    let arff_text = fs::read_to_string(&arff_path).map_err(FerroError::IoError)?;
    parse_arff(&arff_text, &target_name)
}

fn parse_arff(raw: &str, target_name: &str) -> Result<OpenmlDataset, FerroError> {
    enum AttrKind {
        Numeric,
        Nominal(Vec<String>),
    }
    struct Attr {
        name: String,
        kind: AttrKind,
    }

    let mut attrs: Vec<Attr> = Vec::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut in_data = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        if !in_data {
            let lower = trimmed.to_ascii_lowercase();
            if lower.starts_with("@attribute") {
                let attr = parse_attribute(trimmed)?;
                attrs.push(attr);
            } else if lower.starts_with("@data") {
                in_data = true;
            }
        } else {
            data_lines.push(trimmed.to_string());
        }
    }

    let target_idx = attrs
        .iter()
        .position(|a| a.name == target_name)
        .ok_or_else(|| FerroError::SerdeError {
            message: format!("openml: target attribute '{target_name}' not found in ARFF"),
        })?;

    let n_rows = data_lines.len();
    let n_cols = attrs.len() - 1;
    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    let mut target = Array1::<f64>::zeros(n_rows);
    let mut nominal_levels: Vec<Vec<String>> = Vec::with_capacity(n_cols);
    let mut feature_names: Vec<String> = Vec::with_capacity(n_cols);
    for (j, a) in attrs.iter().enumerate() {
        if j == target_idx {
            continue;
        }
        feature_names.push(a.name.clone());
        nominal_levels.push(match &a.kind {
            AttrKind::Numeric => Vec::new(),
            AttrKind::Nominal(levels) => levels.clone(),
        });
    }
    let target_levels = match &attrs[target_idx].kind {
        AttrKind::Numeric => Vec::new(),
        AttrKind::Nominal(levels) => levels.clone(),
    };

    fn parse_attribute(line: &str) -> Result<Attr, FerroError> {
        // @attribute <name> {a,b,c}    or    @attribute <name> numeric
        let body = line["@attribute".len()..].trim();
        let (name, rest) = split_attribute_name(body)?;
        let rest_lower = rest.trim().to_ascii_lowercase();
        if rest_lower.starts_with('{') {
            // nominal
            let inner = rest
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .to_string();
            let levels: Vec<String> = inner
                .split(',')
                .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
                .collect();
            Ok(Attr {
                name,
                kind: AttrKind::Nominal(levels),
            })
        } else if rest_lower.starts_with("numeric")
            || rest_lower.starts_with("real")
            || rest_lower.starts_with("integer")
        {
            Ok(Attr {
                name,
                kind: AttrKind::Numeric,
            })
        } else {
            // String / date / unknown — treat as numeric (will fail on parse
            // if the column actually contains non-numeric data).
            Ok(Attr {
                name,
                kind: AttrKind::Numeric,
            })
        }
    }

    fn split_attribute_name(body: &str) -> Result<(String, String), FerroError> {
        if let Some(stripped) = body.strip_prefix('\'')
            && let Some(end) = stripped.find('\'')
        {
            let name = stripped[..end].to_string();
            let rest = stripped[end + 1..].trim_start().to_string();
            return Ok((name, rest));
        }
        let mut parts = body.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_string();
        let rest = parts.next().unwrap_or("").to_string();
        Ok((name, rest))
    }

    for (i, line) in data_lines.iter().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != attrs.len() {
            return Err(FerroError::SerdeError {
                message: format!(
                    "openml: ARFF data row {} has {} fields (expected {})",
                    i + 1,
                    parts.len(),
                    attrs.len()
                ),
            });
        }
        let mut col = 0usize;
        for (j, attr) in attrs.iter().enumerate() {
            let raw_v = parts[j].trim().trim_matches('\'').trim_matches('"');
            let value = match &attr.kind {
                AttrKind::Numeric => raw_v.parse::<f64>().unwrap_or(f64::NAN),
                AttrKind::Nominal(levels) => levels
                    .iter()
                    .position(|l| l == raw_v)
                    .map(|p| p as f64)
                    .unwrap_or(f64::NAN),
            };
            if j == target_idx {
                target[i] = value;
            } else {
                data[[i, col]] = value;
                col += 1;
            }
        }
    }

    Ok(OpenmlDataset {
        data,
        target,
        feature_names,
        target_name: target_name.to_string(),
        nominal_levels,
        target_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_numeric_arff() {
        let arff = "\
% comment\n\
@RELATION test\n\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE y numeric\n\
@ATTRIBUTE label numeric\n\
@DATA\n\
1.0,2.0,0\n\
3.5,4.5,1\n\
";
        let ds = parse_arff(arff, "label").unwrap();
        assert_eq!(ds.data.dim(), (2, 2));
        assert_eq!(ds.target.len(), 2);
        assert!((ds.data[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((ds.data[[1, 1]] - 4.5).abs() < 1e-12);
        assert!((ds.target[0] - 0.0).abs() < 1e-12);
        assert!((ds.target[1] - 1.0).abs() < 1e-12);
        assert_eq!(ds.feature_names, vec!["x", "y"]);
        assert_eq!(ds.target_name, "label");
    }

    #[test]
    fn parse_arff_with_nominal_target() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE class {cat,dog,bird}\n\
@DATA\n\
1.0,cat\n\
2.0,dog\n\
3.0,bird\n\
4.0,cat\n\
";
        let ds = parse_arff(arff, "class").unwrap();
        assert_eq!(ds.data.dim(), (4, 1));
        assert_eq!(ds.target_levels, vec!["cat", "dog", "bird"]);
        assert!((ds.target[0] - 0.0).abs() < 1e-12);
        assert!((ds.target[1] - 1.0).abs() < 1e-12);
        assert!((ds.target[2] - 2.0).abs() < 1e-12);
        assert!((ds.target[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn parse_arff_missing_target_errors() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@DATA\n\
1.0\n\
";
        assert!(parse_arff(arff, "missing").is_err());
    }

    #[test]
    fn parse_arff_wrong_field_count_errors() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE y numeric\n\
@DATA\n\
1.0\n\
";
        assert!(parse_arff(arff, "y").is_err());
    }
}
