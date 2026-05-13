//! Auto-generated conformance coverage gate for ferrolearn-covariance.
//!
//! Reads `_surface_inventory.toml` and `_surface_exclusions.toml` and
//! verifies that every inventory entry is either documented as an
//! exclusion OR mentioned by leaf name in at least one test source file.
//! Modeled on ferrolearn-linear/tests/conformance_surface_coverage.rs.

use ferrolearn_test_oracle::{assert_surface_covered, SurfaceExclusions, SurfaceInventory};
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn test_dir() -> PathBuf {
    crate_root().join("tests")
}

#[test]
fn surface_coverage_gate() {
    let inv_path = test_dir().join("conformance").join("_surface_inventory.toml");
    let exc_path = test_dir().join("conformance").join("_surface_exclusions.toml");
    let inventory = SurfaceInventory::load(&inv_path);
    let exclusions = SurfaceExclusions::load(&exc_path);
    if inventory.items.is_empty() {
        // No inventory yet — treat as informational, not a hard gate.
        return;
    }

    let candidate_test_files: Vec<PathBuf> = {
        let mut v = Vec::new();
        if let Ok(entries) = std::fs::read_dir(test_dir()) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "rs") {
                    v.push(p);
                }
            }
        }
        v
    };
    let paths: Vec<&Path> = candidate_test_files
        .iter()
        .map(PathBuf::as_path)
        .collect();
    assert!(
        !paths.is_empty(),
        "no .rs test files found under {}",
        test_dir().display()
    );
    assert_surface_covered(&inventory, &exclusions, &paths);
}
