#!/usr/bin/env python3
"""Generate surface inventory + coverage gates for crates not yet gated.

For each crate:
  1. Extract public estimator names from `pub use` declarations in lib.rs.
  2. Test files are grepped for which symbols appear.
  3. Items mentioned in any test file → `_surface_inventory.toml`.
  4. Items NOT mentioned in any test file → `_surface_exclusions.toml`
     with a default "no conformance coverage yet" reason.
  5. A `conformance_surface_coverage.rs` test file is written if missing.

This is approximate (text-based) but matches the pattern used in
ferrolearn-linear and ferray.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

ROOT = Path("/home/doll/ferrolearn")

# Crates that already have a surface gate (do not overwrite).
ALREADY_GATED = {"ferrolearn-linear"}

# Crates to process. We skip test-oracle (workspace-internal) and
# python/bench/sparse/datasets/fetch/io (no public estimators).
CRATES = [
    "ferrolearn-tree",
    "ferrolearn-cluster",
    "ferrolearn-decomp",
    "ferrolearn-preprocess",
    "ferrolearn-metrics",
    "ferrolearn-neighbors",
    "ferrolearn-bayes",
    "ferrolearn-model-sel",
    "ferrolearn-numerical",
    "ferrolearn-kernel",
    "ferrolearn-covariance",
    "ferrolearn-neural",
]


def parse_pub_use_block(lib_text: str) -> list[str]:
    """Extract all CamelCase identifiers exported via `pub use`."""
    items = set()
    in_pub_use = False
    buffer = []
    for line in lib_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("pub use"):
            in_pub_use = True
            buffer = [stripped]
        elif in_pub_use:
            buffer.append(stripped)
        if in_pub_use and stripped.endswith(";"):
            joined = " ".join(buffer)
            for ident in re.findall(r"\b([A-Z][A-Za-z0-9_]+)\b", joined):
                # Skip prelude items / common helpers.
                if ident in {"Fitted", "Self"}:
                    continue
                items.add(ident)
            buffer = []
            in_pub_use = False
    return sorted(items)


def extract_module_path(lib_text: str, ident: str) -> str:
    """Best-effort path discovery — fall back to crate root."""
    return ident  # treat the leaf name as the path; the gate matches by leaf


def gather_test_text(crate_dir: Path) -> str:
    out = []
    tests = crate_dir / "tests"
    if not tests.is_dir():
        return ""
    for f in tests.glob("*.rs"):
        try:
            out.append(f.read_text())
        except OSError as e:
            print(f"    warn: could not read {f}: {e}")
    # also include conformance/*.toml so divergence + exclusion files don't
    # force a re-exclude
    conf = tests / "conformance"
    if conf.is_dir():
        for f in conf.glob("*.toml"):
            try:
                out.append(f.read_text())
            except OSError as e:
                print(f"    warn: could not read {f}: {e}")
    return "\n".join(out)


COVERAGE_GATE_RS = '''//! Auto-generated conformance coverage gate for {crate_name}.
//!
//! Reads `_surface_inventory.toml` and `_surface_exclusions.toml` and
//! verifies that every inventory entry is either documented as an
//! exclusion OR mentioned by leaf name in at least one test source file.
//! Modeled on ferrolearn-linear/tests/conformance_surface_coverage.rs.

use ferrolearn_test_oracle::{{assert_surface_covered, SurfaceExclusions, SurfaceInventory}};
use std::path::{{Path, PathBuf}};

fn crate_root() -> PathBuf {{
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}}

fn test_dir() -> PathBuf {{
    crate_root().join("tests")
}}

#[test]
fn surface_coverage_gate() {{
    let inv_path = test_dir().join("conformance").join("_surface_inventory.toml");
    let exc_path = test_dir().join("conformance").join("_surface_exclusions.toml");
    let inventory = SurfaceInventory::load(&inv_path);
    let exclusions = SurfaceExclusions::load(&exc_path);
    if inventory.items.is_empty() {{
        // No inventory yet — treat as informational, not a hard gate.
        return;
    }}

    let candidate_test_files: Vec<PathBuf> = {{
        let mut v = Vec::new();
        if let Ok(entries) = std::fs::read_dir(test_dir()) {{
            for entry in entries.flatten() {{
                let p = entry.path();
                if p.extension().is_some_and(|e| e == "rs") {{
                    v.push(p);
                }}
            }}
        }}
        v
    }};
    let paths: Vec<&Path> = candidate_test_files
        .iter()
        .map(PathBuf::as_path)
        .collect();
    assert!(
        !paths.is_empty(),
        "no .rs test files found under {{}}",
        test_dir().display()
    );
    assert_surface_covered(&inventory, &exclusions, &paths);
}}
'''


def make_files(crate: str):
    crate_dir = ROOT / crate
    lib = crate_dir / "src" / "lib.rs"
    if not lib.exists():
        print(f"  skip {crate}: no src/lib.rs")
        return

    test_text = gather_test_text(crate_dir)

    items = parse_pub_use_block(lib.read_text())
    if not items:
        print(f"  skip {crate}: no public exports detected")
        return

    covered = []
    uncovered = []
    for it in items:
        if it in test_text:
            covered.append(it)
        else:
            uncovered.append(it)

    conf_dir = crate_dir / "tests" / "conformance"
    conf_dir.mkdir(parents=True, exist_ok=True)

    inv_path = conf_dir / "_surface_inventory.toml"
    exc_path = conf_dir / "_surface_exclusions.toml"

    crate_short = crate.replace("ferrolearn-", "").replace("-", "_")

    if not inv_path.exists():
        with inv_path.open("w") as f:
            f.write(f"# Public surface inventory for {crate}.\n")
            f.write("#\n")
            f.write("# Each entry must either appear by name in a conformance test file\n")
            f.write("# OR be listed in `_surface_exclusions.toml` with a reason.\n")
            f.write("\n")
            for it in items:
                f.write("[[item]]\n")
                f.write(f'path = "ferrolearn_{crate_short}::{it}"\n')
                f.write(f'phase = "{crate_short}"\n')
                f.write('fixture = ""\n\n')
        print(f"  wrote {inv_path.relative_to(ROOT)}  ({len(items)} items)")

    if not exc_path.exists():
        with exc_path.open("w") as f:
            f.write(f"# Documented exclusions for {crate}.\n")
            f.write("#\n")
            f.write("# Items here have no conformance test yet; move them into a test\n")
            f.write("# (any leaf-name mention will satisfy the gate) to remove the entry.\n")
            f.write("\n")
            for it in uncovered:
                f.write("[[exclusion]]\n")
                f.write(f'path = "ferrolearn_{crate_short}::{it}"\n')
                f.write('reason = "No conformance test yet — tracked under #338 follow-up."\n\n')
        print(f"  wrote {exc_path.relative_to(ROOT)}  ({len(uncovered)} exclusions)")

    gate_path = crate_dir / "tests" / "conformance_surface_coverage.rs"
    if not gate_path.exists():
        with gate_path.open("w") as f:
            f.write(COVERAGE_GATE_RS.format(crate_name=crate))
        print(f"  wrote {gate_path.relative_to(ROOT)}")


def main():
    for crate in CRATES:
        if crate in ALREADY_GATED:
            print(f"= {crate} (already gated, skipping)")
            continue
        print(f"= {crate}")
        make_files(crate)
    print("Done.")


if __name__ == "__main__":
    main()
