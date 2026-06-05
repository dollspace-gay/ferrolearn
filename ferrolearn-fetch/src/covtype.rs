//! `fetch_covtype` — Forest Covertype dataset.
//!
//! 581012 samples × 54 features (10 quantitative + 44 binary indicators),
//! 7-class target. Upstream is a single gzipped CSV.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.fetch_covtype` (`sklearn/datasets/_covtype.py`;
//! live oracle 1.5.2). Verification model A (cargo test + sklearn source
//! contract; full numerical value-parity needs a live 581012×54 download, so it
//! is NOT-STARTED offline). Design doc: `.design/datasets/covtype.md` (8 REQs).
//! Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete
//! blocker). No code divergence found this iteration (verify-and-document).
//!
//! **5 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-ARCHIVE-METADATA | SHIPPED | `pub const ARCHIVE` (filename `covtype.data.gz`, url `ndownloader.figshare.com/files/5976039`, sha256 `614360…15771`) matches sklearn `_covtype.py:40-43` char-for-char (live-oracle verified). Guard `archive_url_matches_sklearn_source` pins all three to the sklearn-source literals (R-CHAR-3). |
//! | REQ-CSV-PARSE | SHIPPED | `parse_covtype_csv` parses 55 columns (54 features + label), errors on wrong column count / non-float — mirroring sklearn loading the gzipped `covtype.data` and splitting `X = Xy[:, :-1]` (`_covtype.py:206`). Guards `parser_two_rows`/`parser_rejects_wrong_column_count`/`parse_feature_target_split` (`data[[0,53]]==53.0`, target from the last column). |
//! | REQ-TARGET-CONVENTION | SHIPPED | `target[i] = row[54] as usize` keeps labels `1..=7` with NO `-1` shift, matching sklearn `y = Xy[:, -1].astype(np.int32)` (`_covtype.py:207`, docstring "ranging between 1 to 7"). Guard `parse_preserves_label_1_to_7_no_shift`. |
//! | REQ-CACHE-INTEGRATION | SHIPPED | `fetch_covtype` calls `dataset_dir("covtype", data_home)` (`cache.rs`) + `fetch_file(.., Some(ARCHIVE.sha256), ..)` — first-use download with SHA verification, mirroring sklearn's `_fetch_remote(ARCHIVE, ...)` + data_home caching. |
//! | REQ-CONSUMER | SHIPPED | `pub fn fetch_covtype` + `pub struct Covtype` re-exported at the crate root (`lib.rs` `pub use covtype::{Covtype, fetch_covtype}`) — public boundary API (R-DEFER-1/S5). Underclaim: no in-workspace non-test function caller. |
//! | REQ-VALUE-PARITY | NOT-STARTED | element-wise `(data, target)` parity vs `sklearn.datasets.fetch_covtype()` requires the 581012×54 network download — unreachable offline (network-gated, NOT a code bug). Blocker #2064. |
//! | REQ-FULL-PARAMS | NOT-STARTED | signature `fetch_covtype(data_home)` omits sklearn's `download_if_missing`/`random_state`/`shuffle`/`return_X_y`/`as_frame`/`n_retries`/`delay` (`_covtype.py:80`). Blocker #2065. |
//! | REQ-SUBSTRATE | NOT-STARTED | uses `ndarray::{Array1, Array2}` for `Covtype`, not `ferray-core` (R-SUBSTRATE-1); shares the sibling fetchers' `ndarray` types. Tracked under #2064's migration follow-up. |

use std::fs;
use std::io::Read;
use std::path::Path;

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_covtype.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "covtype.data.gz",
    url: "https://ndownloader.figshare.com/files/5976039",
    sha256: "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct Covtype {
    /// Feature matrix `(n_samples, 54)`.
    pub data: Array2<f64>,
    /// Class labels in `1..=7` (sklearn convention).
    pub target: Array1<usize>,
}

/// Fetch + parse the covertype dataset.
///
/// # Errors
///
/// Propagates cache, download, gunzip, or parse failures.
pub fn fetch_covtype(data_home: Option<&Path>) -> Result<Covtype, FerroError> {
    let dir = dataset_dir("covtype", data_home)?;
    let gz = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;
    let bytes = fs::read(&gz).map_err(FerroError::IoError)?;
    let mut text = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .map_err(FerroError::IoError)?;
    parse_covtype_csv(&text)
}

fn parse_covtype_csv(raw: &str) -> Result<Covtype, FerroError> {
    // 55 columns: 54 features + 1 label
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != 55 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "covtype line {} has {} columns (expected 55)",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = Vec::with_capacity(55);
        for (i, p) in parts.iter().enumerate() {
            row.push(p.parse::<f64>().map_err(|e| FerroError::SerdeError {
                message: format!("covtype line {} col {}: '{}' ({e})", lineno + 1, i, p),
            })?);
        }
        rows.push(row);
    }
    let n = rows.len();
    let mut data = Array2::<f64>::zeros((n, 54));
    let mut target = Array1::<usize>::zeros(n);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..54 {
            data[[i, j]] = row[j];
        }
        target[i] = row[54] as usize;
    }
    Ok(Covtype { data, target })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_two_rows() {
        let mut row = vec!["0.0"; 54].join(",");
        row.push_str(",3");
        let raw = format!("{row}\n{row}\n");
        let ds = parse_covtype_csv(&raw).unwrap();
        assert_eq!(ds.data.dim(), (2, 54));
        assert_eq!(ds.target.len(), 2);
        assert_eq!(ds.target[0], 3);
    }

    #[test]
    fn parser_rejects_wrong_column_count() {
        let raw = "1,2,3\n";
        assert!(parse_covtype_csv(raw).is_err());
    }

    #[test]
    fn metadata_matches_sklearn() {
        assert_eq!(ARCHIVE.filename, "covtype.data.gz");
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }

    /// Characterization (REQ-ARCHIVE-METADATA). Pins ferrolearn's `ARCHIVE`
    /// constant byte-for-byte to the sklearn SOURCE constant
    /// `ARCHIVE = RemoteFileMetadata(filename=..., url=..., checksum=...)`
    /// at `sklearn/datasets/_covtype.py:40-43` (live oracle
    /// `python3 -c "from sklearn.datasets import _covtype as c; print(c.ARCHIVE.filename, c.ARCHIVE.url, c.ARCHIVE.checksum)"`
    /// → `covtype.data.gz https://ndownloader.figshare.com/files/5976039
    ///   614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`).
    /// The expected literals are the sklearn-source values, NOT copied from the
    /// ferrolearn side (R-CHAR-3). This STRENGTHENS `metadata_matches_sklearn`
    /// (which only checks the sha *length*) by pinning the exact url + sha256.
    #[test]
    fn archive_url_matches_sklearn_source() {
        // sklearn/datasets/_covtype.py:41 — filename="covtype.data.gz"
        assert_eq!(ARCHIVE.filename, "covtype.data.gz");
        // sklearn/datasets/_covtype.py:42 — url="https://ndownloader.figshare.com/files/5976039"
        assert_eq!(
            ARCHIVE.url,
            "https://ndownloader.figshare.com/files/5976039"
        );
        // sklearn/datasets/_covtype.py:43 — checksum="614360d0...b15771"
        assert_eq!(
            ARCHIVE.sha256,
            "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
        );
    }

    /// Characterization (REQ-TARGET-CONVENTION). Pins the NO-`-1`-shift label
    /// convention to sklearn `y = Xy[:, -1].astype(np.int32, copy=False)`
    /// (`sklearn/datasets/_covtype.py:207`) and the docstring "values ranging
    /// between 1 to 7" (`:154-157`). Oracle confirms a 55-col row ending in
    /// `,7` yields `y[0] == 7` and one ending in `,1` yields `y[0] == 1`
    /// (NOT 6 / 0). The expected `7`/`1` come from sklearn's last-column cast,
    /// not from ferrolearn (R-CHAR-3).
    #[test]
    fn parse_preserves_label_1_to_7_no_shift() {
        // Label 7 in the 55th column -> target 7 (sklearn keeps it; no shift to 6).
        let mut row7 = vec!["0.0"; 54].join(",");
        row7.push_str(",7");
        let raw7 = format!("{row7}\n{row7}\n");
        let ds7 = parse_covtype_csv(&raw7).unwrap();
        assert_eq!(ds7.target[0], 7);

        // Label 1 -> target 1 (sklearn keeps it; no shift to 0).
        let mut row1 = vec!["0.0"; 54].join(",");
        row1.push_str(",1");
        let raw1 = format!("{row1}\n{row1}\n");
        let ds1 = parse_covtype_csv(&raw1).unwrap();
        assert_eq!(ds1.target[0], 1);
    }

    /// Characterization (REQ-CSV-PARSE / REQ-TARGET-CONVENTION, feature/target
    /// split). Pins the 54-feature / last-column-is-target partition to sklearn
    /// `X = Xy[:, :-1]` / `y = Xy[:, -1]` (`sklearn/datasets/_covtype.py:206-207`).
    /// A row whose features equal their column index (0..54) and whose 55th
    /// field is the label `5`: the live oracle (np.genfromtxt + the sklearn
    /// split) gives `X.shape == (2, 54)`, `X[0,0] == 0.0`, `X[0,53] == 53.0`,
    /// `y[0] == 5` — i.e. column 54 is the LABEL, not a feature, and there is no
    /// off-by-one. Expected values are the sklearn-split outputs (R-CHAR-3).
    #[test]
    fn parse_feature_target_split() {
        let feats: Vec<String> = (0..54).map(|i| (i as f64).to_string()).collect();
        let mut row = feats.join(",");
        row.push_str(",5"); // 55th field is the label
        let raw = format!("{row}\n{row}\n");
        let ds = parse_covtype_csv(&raw).unwrap();

        // sklearn `X = Xy[:, :-1]`: first 54 columns are features.
        assert_eq!(ds.data.dim(), (2, 54));
        assert_eq!(ds.data[[0, 0]], 0.0); // feature column 0
        assert_eq!(ds.data[[0, 53]], 53.0); // feature column 53 (last feature)

        // sklearn `y = Xy[:, -1]`: column 54 is the target, value 5.
        assert_eq!(ds.target[0], 5);
    }
}
