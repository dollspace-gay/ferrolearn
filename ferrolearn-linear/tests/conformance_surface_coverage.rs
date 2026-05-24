//! Per-crate conformance-coverage gate for ferrolearn-linear.
//!
//! Reads `_surface_inventory.toml` and `_surface_exclusions.toml` and
//! verifies that every inventory entry is either:
//!   - documented in `_surface_exclusions.toml` (with a reason), OR
//!   - mentioned by leaf name in at least one test source file.
//!
//! Adding a new public estimator without either covering it or excluding
//! it must fail the build. This pattern is borrowed from ferrotorch and
//! ferray's surface-coverage gates.

use ferrolearn_test_oracle::{SurfaceExclusions, SurfaceInventory, assert_surface_covered};
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn test_dir() -> PathBuf {
    crate_root().join("tests")
}

#[test]
fn surface_coverage_gate() {
    let inv_path = test_dir()
        .join("conformance")
        .join("_surface_inventory.toml");
    let exc_path = test_dir()
        .join("conformance")
        .join("_surface_exclusions.toml");
    let inventory = SurfaceInventory::load(&inv_path);
    let exclusions = SurfaceExclusions::load(&exc_path);
    assert!(
        !inventory.items.is_empty(),
        "surface inventory at {} is empty — populate or remove the gate",
        inv_path.display()
    );

    // Conformance test files that count as coverage references.
    let test_files: Vec<PathBuf> = vec![
        test_dir().join("conformance_sklearn.rs"),
        test_dir().join("oracle_tests.rs"),
        test_dir().join("sklearn_equivalence.rs"),
    ];
    let owned_paths: Vec<&Path> = test_files
        .iter()
        .filter(|p| p.exists())
        .map(PathBuf::as_path)
        .collect();
    assert!(
        !owned_paths.is_empty(),
        "no conformance test files found under {}",
        test_dir().display()
    );
    assert_surface_covered(&inventory, &exclusions, &owned_paths);
}
