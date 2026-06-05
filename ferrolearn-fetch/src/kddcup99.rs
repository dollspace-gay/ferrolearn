//! `fetch_kddcup99` — KDD Cup 1999 intrusion-detection dataset.
//!
//! Two variants on disk:
//! - Full (~743k samples)
//! - 10% subset (~494k samples)
//!
//! Both are CSVs with mixed numeric + categorical columns and a string label
//! at the end (e.g. `"normal."`, `"smurf."`). To stay numerical we expose the
//! parser as `parse_kddcup99_csv` returning string labels alongside numeric
//! columns; the high-level [`fetch_kddcup99`] returns numeric data + string
//! label vector.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.fetch_kddcup99` (`sklearn/datasets/_kddcup99.py`;
//! live oracle 1.5.2). Verification model A (cargo test + sklearn source
//! contract; full numerical value-parity needs a live network download, so it is
//! NOT-STARTED offline). Design doc: `.design/datasets/kddcup99.md` (11 REQs).
//! Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete
//! blocker). One real offline divergence was found AND FIXED this iteration (the
//! 10% subset checksum).
//!
//! **8 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-ARCHIVE-FULL-METADATA | SHIPPED | `pub const ARCHIVE_FULL` (filename `kddcup99_data`, url `ndownloader.figshare.com/files/5976045`, sha256 `3b6c942a…274292`) matches sklearn `_kddcup99.py:34-37` char-for-char. Guard `archive_full_checksum_matches_sklearn_source`. |
//! | REQ-ARCHIVE-10PCT-CHECKSUM | SHIPPED | FIXED #2074: `ARCHIVE_10PCT.sha256` was `…eb35a805a2cb4ddb1ec1422` (wrong — would reject the correct sklearn 10% file); corrected to sklearn `_kddcup99.py:45` `8045aca0…979a1b0837c42a9fd9561`. (filename `kddcup99_10_data` + url `.../5976042` already matched `:43-44`.) Guard `archive_10pct_checksum_matches_sklearn_source`. |
//! | REQ-CSV-PARSE | SHIPPED | `parse_kddcup99_csv` parses 42 columns (41 features + label), errors on wrong column count / non-float — mirroring sklearn `X = Xy[:, :-1]` (`_kddcup99.py:399`). Guards `parser_two_rows`/`parser_rejects_wrong_column_count`/`parse_shape_and_string_target` (`data.ncols()==41`). |
//! | REQ-CATEGORICAL-ENCODING | SHIPPED | cols `[1,2,3]` (protocol_type/service/flag) are level-index-encoded, matching sklearn's symbolic dtype `("protocol_type","S4")`/`("service","S11")`/`("flag","S6")` at positions 1/2/3 (`_kddcup99.py:322-324`). Guard `categorical_columns_match_sklearn_symbolic`. |
//! | REQ-TARGET-LABELS | SHIPPED | `target: Vec<String>` keeps the last column as the string label (e.g. `"normal."`), matching sklearn `y = Xy[:, -1]` (`_kddcup99.py:400`, labels kept as bytes/strings). Guard `parse_shape_and_string_target` (`target[0]=="normal."`). |
//! | REQ-SUBSET-VARIANT | SHIPPED | `KddSubset::{Full,Percent10}` selects `ARCHIVE_FULL`/`ARCHIVE_10PCT`, mirroring sklearn's `percent10` bool (default True → 10% subset, `_kddcup99.py:72`/`:309-314`). |
//! | REQ-CACHE-INTEGRATION | SHIPPED | `fetch_kddcup99` calls `dataset_dir("kddcup99", data_home)` + `fetch_file(.., Some(archive.sha256), ..)` — first-use download with SHA verification, mirroring sklearn's `_fetch_remote(ARCHIVE, ...)` + data_home caching. |
//! | REQ-CONSUMER | SHIPPED | `pub fn fetch_kddcup99` + `pub struct KddCup99` + `pub enum KddSubset` re-exported at the crate root (`lib.rs`) — public boundary API (R-DEFER-1/S5). Underclaim: no in-workspace non-test function caller. |
//! | REQ-VALUE-PARITY | NOT-STARTED | element-wise `(data, target)` parity vs `sklearn.datasets.fetch_kddcup99()` requires the figshare network download — unreachable offline (network-gated, NOT a code bug). Blocker #2075. |
//! | REQ-FULL-PARAMS | NOT-STARTED | signature `fetch_kddcup99(data_home, subset)` omits sklearn's `subset` attack-type filter (SA/SF/http/smtp), `shuffle`/`random_state`, `download_if_missing`/`return_X_y`/`as_frame`/`n_retries`/`delay` (`_kddcup99.py:66-78`). Blocker #2076. |
//! | REQ-SUBSTRATE | NOT-STARTED | uses `ndarray::Array2` for `KddCup99`, not `ferray-core` (R-SUBSTRATE-1); shares the sibling fetchers' `ndarray` types. Blocker #2077. |

use std::fs;
use std::io::Read;
use std::path::Path;

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::Array2;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// Full KDD Cup 99 archive.
pub const ARCHIVE_FULL: RemoteFile = RemoteFile {
    filename: "kddcup99_data",
    url: "https://ndownloader.figshare.com/files/5976045",
    sha256: "3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292",
};

/// 10% subset.
pub const ARCHIVE_10PCT: RemoteFile = RemoteFile {
    filename: "kddcup99_10_data",
    url: "https://ndownloader.figshare.com/files/5976042",
    sha256: "8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct KddCup99 {
    /// Numeric columns of the dataset (categorical columns are kept as
    /// f64-encoded indices into [`Self::categorical_levels`]).
    pub data: Array2<f64>,
    /// String label per row (e.g. "normal.", "smurf.").
    pub target: Vec<String>,
    /// Per-categorical-column distinct levels (in order of first appearance).
    pub categorical_levels: Vec<Vec<String>>,
    /// Index (within `data` columns) of each categorical column.
    pub categorical_columns: Vec<usize>,
}

/// Variant selector.
#[derive(Debug, Clone, Copy)]
pub enum KddSubset {
    /// Full dataset.
    Full,
    /// 10% subset.
    Percent10,
}

/// Fetch + parse KDD Cup 1999 (gzipped).
pub fn fetch_kddcup99(data_home: Option<&Path>, subset: KddSubset) -> Result<KddCup99, FerroError> {
    let archive = match subset {
        KddSubset::Full => ARCHIVE_FULL,
        KddSubset::Percent10 => ARCHIVE_10PCT,
    };
    let dir = dataset_dir("kddcup99", data_home)?;
    let path = fetch_file(archive.url, archive.filename, Some(archive.sha256), &dir)?;
    let bytes = fs::read(&path).map_err(FerroError::IoError)?;
    let mut text = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .map_err(FerroError::IoError)?;
    parse_kddcup99_csv(&text)
}

fn parse_kddcup99_csv(raw: &str) -> Result<KddCup99, FerroError> {
    // Each line: 41 features + 1 label. Column 1 (protocol_type), 2
    // (service), 3 (flag) are categorical strings; everything else parses
    // as f64.
    const N_COLS: usize = 42;
    const CATEGORICAL: [usize; 3] = [1, 2, 3];

    let mut numeric_rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut levels: Vec<Vec<String>> = vec![Vec::new(); CATEGORICAL.len()];

    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != N_COLS {
            return Err(FerroError::SerdeError {
                message: format!(
                    "kddcup99 line {} has {} columns (expected {N_COLS})",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = Vec::with_capacity(N_COLS - 1);
        for (j, raw) in parts.iter().take(N_COLS - 1).enumerate() {
            if let Some(cat_idx) = CATEGORICAL.iter().position(|&c| c == j) {
                let val = (*raw).to_string();
                let pos = match levels[cat_idx].iter().position(|v| v == &val) {
                    Some(p) => p,
                    None => {
                        levels[cat_idx].push(val);
                        levels[cat_idx].len() - 1
                    }
                };
                row.push(pos as f64);
            } else {
                row.push(raw.parse::<f64>().map_err(|e| FerroError::SerdeError {
                    message: format!("kddcup99 line {} col {}: '{}' ({e})", lineno + 1, j, raw),
                })?);
            }
        }
        numeric_rows.push(row);
        labels.push(parts[N_COLS - 1].to_string());
    }
    let n = numeric_rows.len();
    let mut data = Array2::<f64>::zeros((n, N_COLS - 1));
    for (i, row) in numeric_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            data[[i, j]] = v;
        }
    }
    Ok(KddCup99 {
        data,
        target: labels,
        categorical_levels: levels,
        categorical_columns: CATEGORICAL.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_row(label: &str) -> String {
        // 41 features then label. Use 0 for numeric, "tcp"/"http"/"SF" for
        // categorical to match real-world values.
        let mut parts: Vec<String> = (0..41)
            .map(|i| match i {
                1 => "tcp".to_string(),
                2 => "http".to_string(),
                3 => "SF".to_string(),
                _ => "0".to_string(),
            })
            .collect();
        parts.push(label.to_string());
        parts.join(",")
    }

    #[test]
    fn parser_two_rows() {
        let raw = format!("{}\n{}\n", synth_row("normal."), synth_row("smurf."));
        let ds = parse_kddcup99_csv(&raw).unwrap();
        assert_eq!(ds.data.nrows(), 2);
        assert_eq!(ds.data.ncols(), 41);
        assert_eq!(ds.target, vec!["normal.".to_string(), "smurf.".to_string()]);
        assert_eq!(ds.categorical_columns, vec![1, 2, 3]);
        // All three categorical columns have one level after two identical rows.
        assert_eq!(ds.categorical_levels[0].len(), 1);
    }

    #[test]
    fn parser_rejects_wrong_column_count() {
        assert!(parse_kddcup99_csv("1,2,3,smurf.\n").is_err());
    }

    // ---------------------------------------------------------------------
    // RED PIN — REQ-ARCHIVE-10PCT-CHECKSUM (#2070).
    //
    // Divergence: ferrolearn's `ARCHIVE_10PCT.sha256` diverges from
    // `sklearn/datasets/_kddcup99.py:45` (`ARCHIVE_10_PERCENT.checksum`).
    // The expected literal below is the sklearn SOURCE constant (R-CHAR-3),
    // NOT ferrolearn's value. ferrolearn carries
    //   8045aca0d84e70e622d1148d7df782496f6333bf6eb35a805a2cb4ddb1ec1422
    // sklearn 1.5.2 source has
    //   8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561
    // (shared 43-char prefix, then divergent). With the wrong SHA, ferrolearn
    // would reject the correct sklearn 10% archive on checksum mismatch.
    // FAILS now; goes green when the one const in `kddcup99.rs` is corrected.
    // ---------------------------------------------------------------------
    #[test]
    fn archive_10pct_checksum_matches_sklearn_source() {
        // sklearn/datasets/_kddcup99.py:45 — ARCHIVE_10_PERCENT.checksum
        assert_eq!(
            ARCHIVE_10PCT.sha256,
            "8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561"
        );
    }

    // ---------------------------------------------------------------------
    // GREEN GUARDS — characterization vs the sklearn SOURCE constants
    // (R-CHAR-3: expected values read from `_kddcup99.py`, not ferrolearn).
    // ---------------------------------------------------------------------

    /// REQ-ARCHIVE-FULL-METADATA + 10pct filename/url half of
    /// REQ-ARCHIVE-10PCT-CHECKSUM.
    /// sklearn/datasets/_kddcup99.py:34-37 (ARCHIVE) and :42-44
    /// (ARCHIVE_10_PERCENT filename + url — these MATCH; only the checksum
    /// at :45 diverges, pinned RED above).
    #[test]
    fn archive_full_checksum_matches_sklearn_source() {
        // _kddcup99.py:35
        assert_eq!(ARCHIVE_FULL.filename, "kddcup99_data");
        // _kddcup99.py:36
        assert_eq!(
            ARCHIVE_FULL.url,
            "https://ndownloader.figshare.com/files/5976045"
        );
        // _kddcup99.py:37
        assert_eq!(
            ARCHIVE_FULL.sha256,
            "3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292"
        );
        // _kddcup99.py:43 — 10pct filename MATCHES sklearn.
        assert_eq!(ARCHIVE_10PCT.filename, "kddcup99_10_data");
        // _kddcup99.py:44 — 10pct url MATCHES sklearn.
        assert_eq!(
            ARCHIVE_10PCT.url,
            "https://ndownloader.figshare.com/files/5976042"
        );
    }

    /// REQ-CATEGORICAL-ENCODING. sklearn's symbolic byte-string columns sit at
    /// positions 1/2/3: ("protocol_type","S4") :322, ("service","S11") :323,
    /// ("flag","S6") :324 in `_kddcup99.py`'s 42-entry dtype list. ferrolearn
    /// must select those exact columns and level-index-encode them while
    /// parsing the remaining feature columns as f64.
    #[test]
    fn categorical_columns_match_sklearn_symbolic() {
        // Two rows sharing the same tcp/http/SF symbolic triple, with a
        // non-trivial numeric value (4.0) at feature col 4 (src_bytes).
        let row = {
            let mut parts: Vec<String> = (0..41)
                .map(|i| match i {
                    1 => "tcp".to_string(),
                    2 => "http".to_string(),
                    3 => "SF".to_string(),
                    4 => "4".to_string(),
                    _ => "0".to_string(),
                })
                .collect();
            parts.push("normal.".to_string());
            parts.join(",")
        };
        let raw = format!("{row}\n{row}\n");
        let ds = parse_kddcup99_csv(&raw).unwrap();

        // sklearn symbolic columns at positions 1/2/3 (_kddcup99.py:322-324).
        assert_eq!(ds.categorical_columns, vec![1, 2, 3]);

        // Each categorical column collapses to exactly ONE level across the
        // two identical rows (first-appearance level indexing).
        assert_eq!(ds.categorical_levels.len(), 3);
        assert_eq!(ds.categorical_levels[0], vec!["tcp".to_string()]);
        assert_eq!(ds.categorical_levels[1], vec!["http".to_string()]);
        assert_eq!(ds.categorical_levels[2], vec!["SF".to_string()]);

        // The categorical cells hold the f64 level index (0.0 for the single
        // level), for both rows.
        for r in 0..2 {
            assert_eq!(ds.data[[r, 1]], 0.0);
            assert_eq!(ds.data[[r, 2]], 0.0);
            assert_eq!(ds.data[[r, 3]], 0.0);
            // The numeric column 4 (src_bytes) parses as f64, NOT encoded.
            assert_eq!(ds.data[[r, 4]], 4.0);
            // A plain numeric column (0) parses as 0.0.
            assert_eq!(ds.data[[r, 0]], 0.0);
        }
    }

    /// REQ-CSV-PARSE + REQ-TARGET-LABELS. The feature matrix is the first 41
    /// columns (sklearn `X = Xy[:, :-1]`, _kddcup99.py:399) and the target is
    /// the last column kept as a string (sklearn `y = Xy[:, -1]`, :400; label
    /// dtype ("labels","S16") :362).
    #[test]
    fn parse_shape_and_string_target() {
        let raw = format!("{}\n", synth_row("normal."));
        let ds = parse_kddcup99_csv(&raw).unwrap();
        assert_eq!(ds.data.nrows(), 1);
        assert_eq!(ds.data.ncols(), 41);
        assert_eq!(ds.target.len(), 1);
        // Last column preserved as the string label.
        assert_eq!(ds.target[0], "normal.");
    }
}
