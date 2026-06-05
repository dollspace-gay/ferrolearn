//! On-disk cache management.
//!
//! Mirrors `sklearn.datasets.get_data_home` / `clear_data_home`. The default
//! cache lives in `<dirs::data_local_dir>/ferrolearn_data/` (e.g.
//! `~/.local/share/ferrolearn_data/` on Linux); override via the
//! `FERROLEARN_DATA` environment variable or by passing `data_home` to the
//! fetcher.
//!
//! ## REQ status
//!
//! Mirrors `sklearn.datasets.get_data_home`/`clear_data_home`
//! (`sklearn/datasets/_base.py`; live oracle 1.5.2). Verification model A
//! (cargo test + live sklearn oracle). Design doc: `.design/datasets/cache.md`
//! (4 REQs). Every REQ is BINARY (R-DEFER-2): SHIPPED or NOT-STARTED.
//!
//! **4 SHIPPED / 0 NOT-STARTED.**
//!
//! | REQ | Status | Notes |
//! |---|---|---|
//! | REQ-GET-DATA-HOME-RESOLUTION (resolve + create + return) | SHIPPED | `get_data_home` resolves explicit arg → `FERROLEARN_DATA` env → `dirs::data_local_dir()/ferrolearn_data` → `./ferrolearn_data`, then `fs::create_dir_all` + returns the path — mirroring sklearn's arg → env → default → `makedirs(exist_ok=True)` → return (`_base.py:79-83`). DELIBERATE R-DEV-7 deviations (documented, not bugs): env var `FERROLEARN_DATA` (vs `SCIKIT_LEARN_DATA`) + default location (XDG-idiomatic vs `~/scikit_learn_data`) — product-identity choices preserving the resolution-precedence contract. |
//! | REQ-CLEAR-DATA-HOME (resolve-then-rmtree) | SHIPPED | `clear_data_home` calls `get_data_home` (creates) then `fs::remove_dir_all` guarded by `if path.exists()` — behaviorally equivalent to sklearn's `get_data_home(...); shutil.rmtree(...)` (`_base.py:106-107`), the guard removing an already-absent footgun (R-DEV-4) since get_data_home always creates the dir first. |
//! | REQ-EXPANDUSER (`~` expansion) | SHIPPED | FIXED #2058: `get_data_home` now expands a leading `~`/`~/` via `dirs::home_dir()` before `create_dir_all`, applied to the resolved path (explicit arg AND env value), matching sklearn's unconditional `expanduser(data_home)` (`_base.py:81`). Guard `get_data_home_expands_tilde_like_sklearn` (expected value derived from the OS home like sklearn's `expanduser`, R-CHAR-3). |
//! | REQ-CONSUMER (non-test production consumers) | SHIPPED | `dataset_dir` (ferrolearn helper, no sklearn analog) calls `get_data_home` and is called by five non-test loader sites (`california_housing.rs`/`covtype.rs`/`kddcup99.rs`/`newsgroups.rs`/`openml.rs`); all three fns are re-exported at the crate root (`lib.rs`) as public boundary API. |

use ferrolearn_core::FerroError;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Mirror Python's `os.path.expanduser` (`sklearn/datasets/_base.py:81`,
/// `data_home = expanduser(data_home)`): a leading `~`/`~/...` expands to the
/// user home directory. Other forms (e.g. `~user`) are left untouched, matching
/// the cases sklearn relies on. Falls back to the unexpanded path when no home
/// directory is available (safe degradation; no panic).
fn expand_tilde(path: PathBuf) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    path
}

/// Return the configured data-home directory, creating it if it doesn't
/// already exist.
///
/// Resolution order:
/// 1. `data_home` argument (if `Some`).
/// 2. `FERROLEARN_DATA` environment variable.
/// 3. `dirs::data_local_dir()/ferrolearn_data/`.
/// 4. Fallback `./ferrolearn_data/` if no platform dir is available.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the directory cannot be created.
pub fn get_data_home(data_home: Option<&Path>) -> Result<PathBuf, FerroError> {
    let path = if let Some(p) = data_home {
        p.to_path_buf()
    } else if let Ok(env) = env::var("FERROLEARN_DATA") {
        PathBuf::from(env)
    } else if let Some(base) = dirs::data_local_dir() {
        base.join("ferrolearn_data")
    } else {
        PathBuf::from("./ferrolearn_data")
    };
    // Mirror `os.path.expanduser` (`sklearn/datasets/_base.py:81`): expand a
    // leading `~` for both the explicit arg and the `FERROLEARN_DATA` override
    // before creating the directory.
    let path = expand_tilde(path);
    fs::create_dir_all(&path).map_err(FerroError::IoError)?;
    Ok(path)
}

/// Recursively delete the data-home directory.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the directory cannot be removed.
pub fn clear_data_home(data_home: Option<&Path>) -> Result<(), FerroError> {
    let path = get_data_home(data_home)?;
    if path.exists() {
        fs::remove_dir_all(&path).map_err(FerroError::IoError)?;
    }
    Ok(())
}

/// Return the dataset-specific subdirectory under `data_home`, creating it.
pub fn dataset_dir(name: &str, data_home: Option<&Path>) -> Result<PathBuf, FerroError> {
    let base = get_data_home(data_home)?;
    let dir = base.join(name);
    fs::create_dir_all(&dir).map_err(FerroError::IoError)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        std::env::temp_dir().join(format!("ferrolearn_fetch_test_{nanos}"))
    }

    #[test]
    fn explicit_path_used_and_created() {
        let p = unique_tmp();
        let resolved = get_data_home(Some(&p)).unwrap();
        assert_eq!(resolved, p);
        assert!(p.is_dir());
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn clear_data_home_removes_explicit_dir() {
        let p = unique_tmp();
        get_data_home(Some(&p)).unwrap();
        clear_data_home(Some(&p)).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn dataset_dir_creates_subdir() {
        let p = unique_tmp();
        let sub = dataset_dir("california_housing", Some(&p)).unwrap();
        assert_eq!(sub, p.join("california_housing"));
        assert!(sub.is_dir());
        let _ = fs::remove_dir_all(&p);
    }

    // ---- Critic divergence pin + green guards (crosslink #2057) ----

    /// Divergence: ferrolearn's `get_data_home` diverges from
    /// `sklearn/datasets/_base.py:81` (`data_home = expanduser(data_home)`) for
    /// an explicit `data_home` beginning with `~`.
    ///
    /// sklearn expands a leading `~`/`~/` to the user home folder
    /// (docstring `_base.py:55-56`: "The '~' symbol is expanded to the user
    /// home folder"). Live oracle:
    /// `get_data_home('~/ferrolearn_critic_probe_xyz')` → `/home/doll/ferrolearn_critic_probe_xyz`.
    ///
    /// ferrolearn uses the explicit arg LITERALLY (`p.to_path_buf()` in
    /// `get_data_home`), so it creates a directory literally named `~` instead
    /// of expanding. Expected value derives from the OS home dir the same way
    /// sklearn's `expanduser` does — `home.join(name)` — NOT from ferrolearn
    /// (R-CHAR-3).
    ///
    /// Tracking: #2057-blocker (REQ-EXPANDUSER).
    #[test]
    fn get_data_home_expands_tilde_like_sklearn() {
        let probe = "ferrolearn_test_tilde_probe";
        let arg = format!("~/{probe}");
        let resolved = get_data_home(Some(Path::new(&arg))).unwrap();

        let home = dirs::home_dir().expect("home dir available in test env");
        let expected = home.join(probe);

        // Teardown captures the path before any assertion can early-return:
        // remove BOTH the (correct) expanded dir and the (buggy) literal `~`
        // tree so no stray `~` directory is left in the repo working tree.
        let cleanup = || {
            let _ = fs::remove_dir_all(&expected);
            let _ = fs::remove_dir_all(&resolved);
            // Buggy behavior: create_dir_all("~/ferrolearn_test_tilde_probe")
            // produces a relative `./~/ferrolearn_test_tilde_probe` tree.
            let _ = fs::remove_dir_all(Path::new("~"));
            let _ = fs::remove_dir_all(Path::new(&arg));
        };

        let starts_with_home = resolved.starts_with(&home);
        let no_literal_tilde = !resolved
            .components()
            .any(|c| c.as_os_str() == std::ffi::OsStr::new("~"));

        cleanup();

        assert!(
            starts_with_home,
            "expected ~ expanded under home {home:?}, got {resolved:?}"
        );
        assert!(
            no_literal_tilde,
            "expected no literal `~` component, got {resolved:?}"
        );
        assert_eq!(
            resolved, expected,
            "expected sklearn-style expanduser {expected:?}, got {resolved:?}"
        );
    }

    /// Green guard (AC-RESOLUTION-1): an explicit, non-`~` `data_home` is
    /// returned verbatim and created on access, mirroring sklearn
    /// `makedirs(data_home, exist_ok=True); return data_home` (`_base.py:82-83`).
    /// Oracle: `get_data_home('/tmp/skl_ac1')` → `/tmp/skl_ac1`, `isdir` True.
    /// This guard exercises a path WITHOUT a leading `~`, distinct from the
    /// existing `explicit_path_used_and_created` only in pinning the
    /// create-on-access post-condition against the sklearn `makedirs` contract.
    #[test]
    fn explicit_non_tilde_path_returned_and_created() {
        let p = unique_tmp();
        let resolved = get_data_home(Some(&p)).unwrap();
        assert_eq!(resolved, p, "explicit non-~ arg returned verbatim");
        assert!(resolved.is_dir(), "directory created on access");
        let _ = fs::remove_dir_all(&p);
    }

    /// Green guard (AC-RESOLUTION-2): the env-override MECHANISM mirrors
    /// sklearn's `environ.get("SCIKIT_LEARN_DATA", ...)` when `data_home is
    /// None` (`_base.py:79-80`). ferrolearn reads `FERROLEARN_DATA` (deliberate
    /// R-DEV-7 rename); the contract under test is that the env var is consulted
    /// ONLY when no explicit arg is given, and the resolved dir is created.
    ///
    /// Sets + reads + unsets within the same test with a unique value to stay
    /// safe under test parallelism. Uses a non-`~` value so it asserts the
    /// env-override mechanism, not expansion.
    #[test]
    fn env_override_used_when_arg_is_none() {
        let p = unique_tmp();
        // SAFETY: single-threaded within this test; we set, read, and unset
        // the variable before any other code observes it.
        unsafe {
            env::set_var("FERROLEARN_DATA", &p);
        }
        let resolved = get_data_home(None);
        unsafe {
            env::remove_var("FERROLEARN_DATA");
        }

        let resolved = resolved.unwrap();
        let is_dir = resolved.is_dir();
        let _ = fs::remove_dir_all(&p);

        assert_eq!(
            resolved, p,
            "FERROLEARN_DATA env override used when arg None"
        );
        assert!(is_dir, "env-resolved directory created on access");
    }

    /// Green guard (AC-CLEAR-1): `clear_data_home(Some(p))` after
    /// `get_data_home(Some(p))` leaves the directory non-existent, mirroring
    /// sklearn `data_home = get_data_home(data_home); shutil.rmtree(data_home)`
    /// (`_base.py:106-107`). Oracle:
    /// `get_data_home('/tmp/skl_clear_test'); clear_data_home('/tmp/skl_clear_test')`
    /// → `os.path.exists(p)` is `False`. Distinct from the existing
    /// `clear_data_home_removes_explicit_dir` by also seeding a child entry to
    /// confirm RECURSIVE removal (sklearn `rmtree` removes the whole tree).
    #[test]
    fn clear_data_home_recursively_removes_seeded_tree() {
        let p = unique_tmp();
        get_data_home(Some(&p)).unwrap();
        fs::write(p.join("seed.txt"), b"x").unwrap();
        clear_data_home(Some(&p)).unwrap();
        assert!(!p.exists(), "clear_data_home removes the directory tree");
    }
}
