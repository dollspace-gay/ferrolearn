//! `fetch_20newsgroups` — text classification benchmark, 20 categories.
//!
//! The upstream tarball expands into:
//!
//! ```text
//! 20news-bydate-train/<category>/<doc>
//! 20news-bydate-test/<category>/<doc>
//! ```
//!
//! We extract the chosen `subset` ("train", "test", or "all") and reuse
//! [`ferrolearn_datasets::load_files`] to build the `(documents, labels,
//! target_names)` triple.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.fetch_20newsgroups`
//! (`sklearn/datasets/_twenty_newsgroups.py`; live oracle 1.5.2). Verification
//! model A (cargo test + sklearn source contract; full value-parity needs a live
//! tarball download, so it is NOT-STARTED offline). Design doc:
//! `.design/datasets/newsgroups.md` (9 REQs). Every REQ is BINARY (R-DEFER-2):
//! SHIPPED or NOT-STARTED (with a concrete blocker). No offline code divergence
//! found this iteration (verify-and-document).
//!
//! **6 SHIPPED / 3 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-ARCHIVE-METADATA | SHIPPED | `pub const ARCHIVE` (filename `20news-bydate.tar.gz`, url `ndownloader.figshare.com/files/5975967`, sha256 `8f1b2514…95610`) matches sklearn `_twenty_newsgroups.py:60-63` char-for-char (live-oracle verified). Guard `archive_matches_sklearn_source` pins all three to the sklearn-source literals (R-CHAR-3). |
//! | REQ-SUBSET-FOLDER-MAPPING | SHIPPED | `NewsgroupsSubset::{Train,Test}` read `20news-bydate-train`/`20news-bydate-test`, matching sklearn `TRAIN_FOLDER`/`TEST_FOLDER` (`_twenty_newsgroups.py:67-68`). Guard `train_test_folder_names_match_sklearn` (synthetic-tree round-trip). |
//! | REQ-LOAD-FILES-INTEGRATION | SHIPPED | reuses `ferrolearn_datasets::svmlight::load_files` over the extracted tree to build `(documents, labels, target_names)`, mirroring sklearn's own `load_files(...)` over `20news-bydate-{train,test}`. Guard `extract_archive_round_trip_synthetic`. |
//! | REQ-ALL-CONCAT | SHIPPED | `NewsgroupsSubset::All` concatenates train THEN test (`docs_a.extend(docs_b)`) and errors on class-name mismatch, matching sklearn's `for subset in ("train","test")` concat order (`_twenty_newsgroups.py:335-339`). Guard `all_subset_concatenates_train_then_test` (merged docs `[train.., test..]`, labels `[0,1,0,1]`). |
//! | REQ-CACHE-INTEGRATION | SHIPPED | `fetch_20newsgroups` calls `dataset_dir("20newsgroups", data_home)` + `fetch_file(.., Some(ARCHIVE.sha256), ..)` + extract-once (`if !extract_root.exists()`), mirroring sklearn's SHA-verified download + extract + cache. (sklearn additionally pickles a `20news-bydate.pkz` Bunch — a cache-format delta, folded into REQ-VALUE-PARITY.) |
//! | REQ-CONSUMER | SHIPPED | `pub fn fetch_20newsgroups` + `pub enum NewsgroupsSubset` re-exported at the crate root (`lib.rs`) — public boundary API (R-DEFER-1/S5). Underclaim: no in-workspace non-test function caller. |
//! | REQ-VALUE-PARITY | NOT-STARTED | element-wise `(data, target, target_names)` parity vs `sklearn.datasets.fetch_20newsgroups(subset=.., shuffle=False)` requires the ~14MB tarball download — unreachable offline (network-gated, NOT a code bug). Compound: sklearn default `shuffle=True` so even with network the default doc ORDER differs. Blocker #2067. |
//! | REQ-FULL-PARAMS | NOT-STARTED | signature `fetch_20newsgroups(data_home, subset)` omits sklearn's `categories`/`shuffle`/`random_state`/`remove`/`download_if_missing`/`return_X_y`/`n_retries`/`delay` (`_twenty_newsgroups.py:177-184`); `shuffle` (default True) + `remove` (header/footer/quote stripping) are the most behavior-affecting. Blocker #2068. |
//! | REQ-SUBSTRATE | NOT-STARTED | labels use `ndarray::Array1` (via `load_files`), not `ferray-core` (R-SUBSTRATE-1); shares the sibling fetchers' `ndarray` types. Tracked under #2067's migration follow-up. |

use std::fs;
use std::path::Path;

use ferrolearn_core::FerroError;
use ferrolearn_datasets::svmlight::{LoadFilesResult, load_files};
use flate2::read::GzDecoder;
use tar::Archive;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_twenty_newsgroups.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "20news-bydate.tar.gz",
    url: "https://ndownloader.figshare.com/files/5975967",
    sha256: "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
};

/// Subset selector.
#[derive(Debug, Clone, Copy)]
pub enum NewsgroupsSubset {
    /// Training partition.
    Train,
    /// Held-out test partition.
    Test,
    /// Train + test concatenated.
    All,
}

/// Fetch + extract the 20-newsgroups dataset and return
/// `(documents, labels, target_names)` for the chosen subset.
pub fn fetch_20newsgroups(
    data_home: Option<&Path>,
    subset: NewsgroupsSubset,
) -> Result<LoadFilesResult, FerroError> {
    let dir = dataset_dir("20newsgroups", data_home)?;
    let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;
    let extract_root = dir.join("extracted");
    if !extract_root.exists() {
        extract_archive(&archive_path, &extract_root)?;
    }

    match subset {
        NewsgroupsSubset::Train => load_files(extract_root.join("20news-bydate-train")),
        NewsgroupsSubset::Test => load_files(extract_root.join("20news-bydate-test")),
        NewsgroupsSubset::All => {
            let (mut docs_a, mut labels_a, names_a) =
                load_files(extract_root.join("20news-bydate-train"))?;
            let (docs_b, labels_b, names_b) = load_files(extract_root.join("20news-bydate-test"))?;
            // Class names should match across train/test; if they don't,
            // we report it loudly rather than silently merging mismatched
            // label spaces.
            if names_a != names_b {
                return Err(FerroError::SerdeError {
                    message: "20newsgroups: train and test subsets have different class names"
                        .into(),
                });
            }
            docs_a.extend(docs_b);
            // Append labels — labels_b indices already match the same name
            // ordering since names match.
            let mut out_labels = labels_a.to_vec();
            out_labels.extend(labels_b.iter().copied());
            labels_a = ndarray::Array1::from(out_labels);
            Ok((docs_a, labels_a, names_a))
        }
    }
}

fn extract_archive(archive_path: &Path, dest: &Path) -> Result<(), FerroError> {
    fs::create_dir_all(dest).map_err(FerroError::IoError)?;
    let f = fs::File::open(archive_path).map_err(FerroError::IoError)?;
    let mut archive = Archive::new(GzDecoder::new(f));
    archive.unpack(dest).map_err(FerroError::IoError)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp() -> PathBuf {
        // subsec_nanos alone collides when tests run in parallel; mix in a
        // process-local monotonic counter so concurrent tests get distinct
        // temp roots and never stomp each other's trees.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ferrolearn_news_test_{nanos}_{seq}"))
    }

    #[test]
    fn metadata_matches_sklearn() {
        assert_eq!(ARCHIVE.filename, "20news-bydate.tar.gz");
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }

    /// Characterization (R-CHAR-3): the figshare remote-file descriptor is
    /// byte-for-byte the sklearn source constant `ARCHIVE = RemoteFileMetadata(
    /// filename="20news-bydate.tar.gz",
    /// url="https://ndownloader.figshare.com/files/5975967",
    /// checksum="8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610")`
    /// at `sklearn/datasets/_twenty_newsgroups.py:60-63`. Expected literals are
    /// the sklearn SOURCE values (confirmed via the live 1.5.2 oracle
    /// `_twenty_newsgroups.ARCHIVE`), NEVER copied from the ferrolearn side.
    /// Strengthens `metadata_matches_sklearn`, which only checks the sha LENGTH.
    #[test]
    fn archive_matches_sklearn_source() {
        // sklearn/datasets/_twenty_newsgroups.py:61
        assert_eq!(ARCHIVE.filename, "20news-bydate.tar.gz");
        // sklearn/datasets/_twenty_newsgroups.py:62
        assert_eq!(
            ARCHIVE.url,
            "https://ndownloader.figshare.com/files/5975967"
        );
        // sklearn/datasets/_twenty_newsgroups.py:63
        assert_eq!(
            ARCHIVE.sha256,
            "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610"
        );
    }

    /// Helper: build a `<root>/<category>/<doc>` file with the given content.
    fn write_doc(root: &Path, category: &str, doc: &str, content: &[u8]) {
        use std::io::Write;
        let cat_dir = root.join(category);
        fs::create_dir_all(&cat_dir).unwrap();
        let mut fh = fs::File::create(cat_dir.join(doc)).unwrap();
        fh.write_all(content).unwrap();
    }

    #[test]
    fn extract_archive_round_trip_synthetic() {
        // Build a tiny tar.gz, extract it, then assert load_files works
        // on the resulting tree.
        use std::io::Write;
        let dir = unique_tmp();
        fs::create_dir_all(&dir).unwrap();
        let archive_path = dir.join("synthetic.tar.gz");
        {
            let f = fs::File::create(&archive_path).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            let mut tar_b = tar::Builder::new(gz);
            // Add file alpha/doc0.txt
            let alpha_dir = dir.join("staging").join("alpha");
            fs::create_dir_all(&alpha_dir).unwrap();
            let doc = alpha_dir.join("doc0.txt");
            let mut fh = fs::File::create(&doc).unwrap();
            fh.write_all(b"alpha content").unwrap();
            tar_b.append_path_with_name(&doc, "alpha/doc0.txt").unwrap();
            tar_b.finish().unwrap();
        }
        let dest = dir.join("extracted");
        extract_archive(&archive_path, &dest).unwrap();
        let (docs, labels, names) = load_files(&dest).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(names, vec!["alpha".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Characterization (R-CHAR-3): the `Train` arm reads exactly the
    /// `20news-bydate-train` folder and the `Test` arm reads exactly the
    /// `20news-bydate-test` folder — matching sklearn `TRAIN_FOLDER =
    /// "20news-bydate-train"` / `TEST_FOLDER = "20news-bydate-test"`
    /// (`sklearn/datasets/_twenty_newsgroups.py:67-68`, confirmed via the live
    /// 1.5.2 oracle). The `fetch_20newsgroups` per-subset arms are only
    /// reachable behind the network download, so we pin the folder-name +
    /// `load_files` contract offline by building the two real extracted trees
    /// under the sklearn-named folders and exercising the SAME `load_files`
    /// call each arm makes. Doc contents distinguishable so a swap of train/test
    /// (or a wrong folder literal) would be observable.
    #[test]
    fn train_test_folder_names_match_sklearn() {
        let dir = unique_tmp();
        let extract_root = dir.join("extracted");
        // sklearn TRAIN_FOLDER / TEST_FOLDER (file:67-68)
        let train_root = extract_root.join("20news-bydate-train");
        let test_root = extract_root.join("20news-bydate-test");
        write_doc(&train_root, "alpha", "t0", b"TRAIN alpha doc");
        write_doc(&test_root, "alpha", "s0", b"TEST alpha doc");

        // The Train arm: load_files(extract_root.join("20news-bydate-train")).
        let (train_docs, train_labels, train_names) = load_files(&train_root).unwrap();
        assert_eq!(train_docs, vec!["TRAIN alpha doc".to_string()]);
        assert_eq!(train_labels.to_vec(), vec![0usize]);
        assert_eq!(train_names, vec!["alpha".to_string()]);

        // The Test arm: load_files(extract_root.join("20news-bydate-test")).
        let (test_docs, test_labels, test_names) = load_files(&test_root).unwrap();
        assert_eq!(test_docs, vec!["TEST alpha doc".to_string()]);
        assert_eq!(test_labels.to_vec(), vec![0usize]);
        assert_eq!(test_names, vec!["alpha".to_string()]);

        // A correct mapping yields the train-only doc from the train folder and
        // the test-only doc from the test folder (no cross-contamination).
        assert_ne!(train_docs, test_docs);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Characterization (R-CHAR-3): for `subset = All`, the merged triple is
    /// TRAIN docs/labels first, then TEST docs/labels — mirroring sklearn's
    /// `for subset in ("train", "test"): data_lst.extend(data.data);
    /// target.extend(data.target)` (`sklearn/datasets/_twenty_newsgroups.py:335-339`),
    /// which is train-then-test. The `All` concat in `fetch_20newsgroups` is
    /// only reachable behind the network download, so we replicate the EXACT
    /// concat the arm performs (`docs_a.extend(docs_b)`; train labels then test
    /// labels; `names_a` returned) over two real synthetic extracted trees.
    /// Train/test docs are distinguishable so a reversed order (test-then-train)
    /// would flip the asserted sequence and fail.
    #[test]
    fn all_subset_concatenates_train_then_test() {
        let dir = unique_tmp();
        let extract_root = dir.join("extracted");
        let train_root = extract_root.join("20news-bydate-train");
        let test_root = extract_root.join("20news-bydate-test");
        // Same category names across train/test (the name-match guard requires
        // it); distinguishable contents to pin ordering. Two categories to also
        // pin the label space.
        write_doc(&train_root, "alpha", "t0", b"TRAIN alpha");
        write_doc(&train_root, "beta", "t1", b"TRAIN beta");
        write_doc(&test_root, "alpha", "s0", b"TEST alpha");
        write_doc(&test_root, "beta", "s1", b"TEST beta");

        // Replicate the All arm's logic exactly (the arm is network-gated).
        let (mut docs_a, labels_a, names_a) = load_files(&train_root).unwrap();
        let (docs_b, labels_b, names_b) = load_files(&test_root).unwrap();
        assert_eq!(names_a, names_b, "name-match guard precondition");
        docs_a.extend(docs_b.clone());
        let mut out_labels = labels_a.to_vec();
        out_labels.extend(labels_b.iter().copied());

        // Train-then-test ordering (sklearn :335 `("train", "test")`).
        assert_eq!(
            docs_a,
            vec![
                "TRAIN alpha".to_string(),
                "TRAIN beta".to_string(),
                "TEST alpha".to_string(),
                "TEST beta".to_string(),
            ]
        );
        // labels: train [alpha=0, beta=1] then test [alpha=0, beta=1].
        assert_eq!(out_labels, vec![0usize, 1, 0, 1]);
        // target_names is the (shared) train ordering.
        assert_eq!(names_a, vec!["alpha".to_string(), "beta".to_string()]);
        // merged count = train + test.
        assert_eq!(docs_a.len(), 4);
        let _ = fs::remove_dir_all(&dir);
    }
}
