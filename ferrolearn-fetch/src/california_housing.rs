//! `fetch_california_housing` — 20640×8 regression benchmark.
//!
//! The original sklearn dataset is hosted by figshare as a tarball. We
//! download it, extract `cal_housing.data`, then parse the comma-separated
//! values into a feature matrix + target vector.
//!
//! Feature columns (sklearn order, after re-arranging the raw file):
//! `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
//! Longitude`. Target: `MedHouseVal` (median house value, 100k USD units).
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.fetch_california_housing`
//! (`sklearn/datasets/_california_housing.py`; live oracle 1.5.2). Verification
//! model A (cargo test + sklearn source contract; full numerical value-parity
//! needs a live network download, so it is NOT-STARTED offline — but the
//! column-rearrangement + derived-feature math IS offline-verified on synthetic
//! input). Design doc: `.design/datasets/california_housing.md` (9 REQs). Every
//! REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED (with a concrete blocker).
//! No offline code divergence found this iteration (verify-and-document).
//!
//! **6 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-ARCHIVE-METADATA | SHIPPED | `pub const ARCHIVE` (filename `cal_housing.tgz`, url `ndownloader.figshare.com/files/5976036`, sha256 `aaa5c9a6…ea681`) matches sklearn `_california_housing.py:47-50` char-for-char (live-oracle verified). Guard `archive_matches_sklearn_source`. |
//! | REQ-DERIVED-FEATURES | SHIPPED | `parse_california_csv` reorders the 9 raw columns and derives the 8 sklearn features + target IDENTICALLY to sklearn `columns_index=[8,7,2,3,4,5,6,1,0]` + `AveRooms=totalRooms/households` (`:213`), `AveBedrms=totalBedrooms/households` (`:216`), `AveOccup=population/households` (`:219`), `target=col0/100000` (`:222`): `MedInc=raw7`, `HouseAge=raw2`, `AveRooms=raw3/raw6`, `AveBedrms=raw4/raw6`, `Population=raw5`, `AveOccup=raw5/raw6`, `Latitude=raw1`, `Longitude=raw0`. Guards `parser_handles_synthetic_two_rows` + `derived_features_match_sklearn_columns_index` (distinct per-column values, `<1e-12`). Benign R-DEV-4 deviation: `households.max(1.0)` divide-by-zero guard — inert on real data (households ≥ 1; sklearn doesn't guard). |
//! | REQ-FEATURE-TARGET-NAMES | SHIPPED | `feature_names == ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]` + `target_names == ["MedHouseVal"]` match sklearn `_california_housing.py:199-208` in order. Guard `feature_and_target_names_match_sklearn`. |
//! | REQ-CSV-PARSE | SHIPPED | `parse_california_csv` parses 9-column rows, errors on wrong column count / non-float. Guards `parser_rejects_short_row`/`parser_rejects_non_numeric`. |
//! | REQ-CACHE-EXTRACT | SHIPPED | `fetch_california_housing` calls `dataset_dir("california_housing", data_home)` + `fetch_file(.., Some(ARCHIVE.sha256), ..)` + `extract_data_file` (pulls `cal_housing.data` from the tarball, extract-once via `if target.is_file()`), mirroring sklearn's SHA-verified download + extract. |
//! | REQ-CONSUMER | SHIPPED | `pub fn fetch_california_housing` + `pub struct CaliforniaHousing` re-exported at the crate root (`lib.rs`) — public boundary API (R-DEFER-1/S5). Underclaim: no in-workspace non-test function caller. |
//! | REQ-VALUE-PARITY | NOT-STARTED | element-wise `(data, target)` parity vs `sklearn.datasets.fetch_california_housing()` requires the figshare network download — unreachable offline (network-gated, NOT a code bug; the derived-feature math is offline-verified above). Blocker #2079. |
//! | REQ-FULL-PARAMS | NOT-STARTED | signature `fetch_california_housing(data_home)` omits sklearn's `download_if_missing`/`return_X_y`/`as_frame`/`n_retries`/`delay` (`_california_housing.py:67-75`). Blocker #2080. |
//! | REQ-SUBSTRATE | NOT-STARTED | uses `ndarray::{Array1, Array2}` for `CaliforniaHousing`, not `ferray-core` (R-SUBSTRATE-1); shares the sibling fetchers' `ndarray` types. Blocker #2081. |

use std::fs;
use std::path::{Path, PathBuf};

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};
use tar::Archive;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_california_housing.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "cal_housing.tgz",
    url: "https://ndownloader.figshare.com/files/5976036",
    sha256: "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct CaliforniaHousing {
    /// Feature matrix `(20640, 8)`.
    pub data: Array2<f64>,
    /// Median house value targets.
    pub target: Array1<f64>,
    /// Column names in order.
    pub feature_names: Vec<&'static str>,
    /// Target variable name (single).
    pub target_names: Vec<&'static str>,
}

/// Download (or load from cache) the California housing dataset and return
/// the parsed `(data, target)` arrays.
///
/// Sklearn's column order:
///
/// `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`
///
/// derived from raw columns of the upstream tarball:
///
/// `longitude, latitude, housingMedianAge, totalRooms, totalBedrooms,
/// population, households, medianIncome, medianHouseValue`.
///
/// # Errors
///
/// Returns [`FerroError`] for any cache, download, archive, parsing, or
/// SHA mismatch failure.
pub fn fetch_california_housing(data_home: Option<&Path>) -> Result<CaliforniaHousing, FerroError> {
    let dir = dataset_dir("california_housing", data_home)?;
    let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;

    let csv_path = extract_data_file(&archive_path, &dir)?;
    let raw = fs::read_to_string(&csv_path).map_err(FerroError::IoError)?;
    parse_california_csv(&raw)
}

fn extract_data_file(archive_path: &Path, dir: &Path) -> Result<PathBuf, FerroError> {
    // The tarball contains CaliforniaHousing/cal_housing.data
    let target = dir.join("cal_housing.data");
    if target.is_file() {
        return Ok(target);
    }
    let f = fs::File::open(archive_path).map_err(FerroError::IoError)?;
    let mut archive = Archive::new(GzDecoder::new(f));
    for entry in archive.entries().map_err(FerroError::IoError)? {
        let mut entry = entry.map_err(FerroError::IoError)?;
        let entry_path = entry.path().map_err(FerroError::IoError)?.into_owned();
        if entry_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "cal_housing.data")
        {
            entry.unpack(&target).map_err(FerroError::IoError)?;
            return Ok(target);
        }
    }
    Err(FerroError::SerdeError {
        message: "ferrolearn-fetch: cal_housing.data not found in archive".into(),
    })
}

fn parse_california_csv(raw: &str) -> Result<CaliforniaHousing, FerroError> {
    // Raw column order:
    // 0:longitude 1:latitude 2:housingMedianAge 3:totalRooms 4:totalBedrooms
    // 5:population 6:households 7:medianIncome 8:medianHouseValue
    let mut samples: Vec<[f64; 9]> = Vec::with_capacity(20640);
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != 9 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "california_housing line {} has {} columns (expected 9)",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = [0.0_f64; 9];
        for (i, p) in parts.iter().enumerate() {
            row[i] = p.parse::<f64>().map_err(|e| FerroError::SerdeError {
                message: format!(
                    "california_housing line {} col {}: '{}' not a number ({e})",
                    lineno + 1,
                    i,
                    p
                ),
            })?;
        }
        samples.push(row);
    }
    let n = samples.len();
    let mut data = Array2::<f64>::zeros((n, 8));
    let mut target = Array1::<f64>::zeros(n);
    for (i, row) in samples.iter().enumerate() {
        let households = row[6].max(1.0);
        let pop = row[5];
        let total_rooms = row[3];
        let total_bedrooms = row[4];
        // sklearn-derived columns:
        let med_inc = row[7];
        let house_age = row[2];
        let ave_rooms = total_rooms / households;
        let ave_bedrms = total_bedrooms / households;
        let ave_occup = pop / households;
        let latitude = row[1];
        let longitude = row[0];
        let med_house_val = row[8];
        data[[i, 0]] = med_inc;
        data[[i, 1]] = house_age;
        data[[i, 2]] = ave_rooms;
        data[[i, 3]] = ave_bedrms;
        data[[i, 4]] = pop;
        data[[i, 5]] = ave_occup;
        data[[i, 6]] = latitude;
        data[[i, 7]] = longitude;
        // Target is in dollars; sklearn rescales to 100k USD.
        target[i] = med_house_val / 100_000.0;
    }
    Ok(CaliforniaHousing {
        data,
        target,
        feature_names: vec![
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        target_names: vec!["MedHouseVal"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_handles_synthetic_two_rows() {
        // Two synthetic rows in raw column order.
        let raw = "\
            -122.0,37.0,30.0,1000.0,200.0,400.0,100.0,5.0,250000.0\n\
            -121.0,38.0,20.0,2000.0,400.0,800.0,200.0,7.0,500000.0\n\
        ";
        let ds = parse_california_csv(raw).unwrap();
        assert_eq!(ds.data.dim(), (2, 8));
        assert_eq!(ds.target.len(), 2);
        // Row 0: MedInc=5.0, HouseAge=30.0, AveRooms=1000/100=10, AveBedrms=2,
        // Population=400, AveOccup=4, Latitude=37, Longitude=-122
        assert!((ds.data[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((ds.data[[0, 1]] - 30.0).abs() < 1e-12);
        assert!((ds.data[[0, 2]] - 10.0).abs() < 1e-12);
        assert!((ds.data[[0, 3]] - 2.0).abs() < 1e-12);
        assert!((ds.data[[0, 4]] - 400.0).abs() < 1e-12);
        assert!((ds.data[[0, 5]] - 4.0).abs() < 1e-12);
        assert!((ds.data[[0, 6]] - 37.0).abs() < 1e-12);
        assert!((ds.data[[0, 7]] - (-122.0)).abs() < 1e-12);
        // target: 250000 / 100000 = 2.5
        assert!((ds.target[0] - 2.5).abs() < 1e-12);
        assert!((ds.target[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn parser_rejects_short_row() {
        let raw = "1,2,3,4\n";
        assert!(parse_california_csv(raw).is_err());
    }

    #[test]
    fn parser_rejects_non_numeric() {
        let raw = "1,2,3,4,5,6,7,abc,9\n";
        assert!(parse_california_csv(raw).is_err());
    }

    #[test]
    fn metadata_constants_match_sklearn() {
        assert_eq!(ARCHIVE.filename, "cal_housing.tgz");
        assert!(ARCHIVE.url.contains("figshare.com"));
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }

    /// Characterization (R-CHAR-3): pins ferrolearn's `ARCHIVE` to the sklearn
    /// SOURCE constant `_california_housing.py:47-50`
    /// (`filename="cal_housing.tgz"`, `url="https://ndownloader.figshare.com/files/5976036"`,
    /// `checksum="aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"`).
    /// Strengthens `metadata_constants_match_sklearn` (which only checks the sha
    /// LENGTH + a url-contains substring). Expected literals are read from the
    /// sklearn source, NEVER from ferrolearn.
    #[test]
    fn archive_matches_sklearn_source() {
        assert_eq!(ARCHIVE.filename, "cal_housing.tgz");
        assert_eq!(
            ARCHIVE.url,
            "https://ndownloader.figshare.com/files/5976036"
        );
        assert_eq!(
            ARCHIVE.sha256,
            "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
        );
    }

    /// Characterization (R-CHAR-3): pins `feature_names`/`target_names` to the
    /// sklearn SOURCE lists `_california_housing.py:199-208` (feature_names) and
    /// `:230-232` (target_names), in order. Fills the gap the design doc flagged
    /// (the names were only structurally materialized, never asserted by value).
    #[test]
    fn feature_and_target_names_match_sklearn() {
        // One synthetic 9-col row; values are irrelevant to the names contract.
        let raw = "-122.0,37.0,30.0,1000.0,200.0,400.0,100.0,5.0,250000.0\n";
        let ds = parse_california_csv(raw).unwrap();
        assert_eq!(
            ds.feature_names,
            vec![
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ]
        );
        assert_eq!(ds.target_names, vec!["MedHouseVal"]);
    }

    /// Characterization (R-CHAR-3): pins the column re-arrangement + derived math
    /// to sklearn `columns_index=[8,7,2,3,4,5,6,1,0]` + the three `/= data[:,5]`
    /// divisions + `target/100000.0` (`_california_housing.py:190-222`).
    /// Every raw column carries a DISTINCT value so a column-swap bug is caught
    /// (stronger than `parser_handles_synthetic_two_rows`, which shares some
    /// magnitudes across columns). Expected values are the composition of the
    /// sklearn source transform, NOT copied from ferrolearn.
    ///
    /// raw = [lon=-100, lat=40, age=25, totalRooms=900, totalBed=180,
    ///        pop=600, households=300, medInc=8, medHouseVal=300000]
    /// → MedInc=8, HouseAge=25, AveRooms=900/300=3, AveBedrms=180/300=0.6,
    ///   Population=600, AveOccup=600/300=2, Latitude=40, Longitude=-100,
    ///   target=300000/100000=3.
    #[test]
    fn derived_features_match_sklearn_columns_index() {
        let raw = "-100.0,40.0,25.0,900.0,180.0,600.0,300.0,8.0,300000.0\n";
        let ds = parse_california_csv(raw).unwrap();
        assert_eq!(ds.data.dim(), (1, 8));
        assert_eq!(ds.target.len(), 1);
        // MedInc = raw7
        assert!((ds.data[[0, 0]] - 8.0).abs() < 1e-12);
        // HouseAge = raw2
        assert!((ds.data[[0, 1]] - 25.0).abs() < 1e-12);
        // AveRooms = raw3 / raw6 = 900 / 300 = 3
        assert!((ds.data[[0, 2]] - 3.0).abs() < 1e-12);
        // AveBedrms = raw4 / raw6 = 180 / 300 = 0.6
        assert!((ds.data[[0, 3]] - 0.6).abs() < 1e-12);
        // Population = raw5
        assert!((ds.data[[0, 4]] - 600.0).abs() < 1e-12);
        // AveOccup = raw5 / raw6 = 600 / 300 = 2
        assert!((ds.data[[0, 5]] - 2.0).abs() < 1e-12);
        // Latitude = raw1
        assert!((ds.data[[0, 6]] - 40.0).abs() < 1e-12);
        // Longitude = raw0
        assert!((ds.data[[0, 7]] - (-100.0)).abs() < 1e-12);
        // target = raw8 / 100000 = 300000 / 100000 = 3
        assert!((ds.target[0] - 3.0).abs() < 1e-12);
    }
}
