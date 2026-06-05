#!/usr/bin/env python3
"""
translate-discipline hook (ferrolearn vibe-fork translation gate).

Enforces "look at the scikit-learn upstream source, look at the design
doc, look at goal.md; then translate to the Rust target" as a
deterministic per-edit gate.

Two invocations in .claude/settings.json:

  - PostToolUse on Read    -> records the Read in session state
  - PreToolUse  on Write|Edit -> gates writes to gated ferrolearn-* crates

State is persisted at .crosslink/.translate-reads.json (per-worktree).

The three required source classes for any gated edit:
  1. goal.md                          (always; global discipline file)
  2. /home/doll/scikit-learn/<path>   (per route; >=1 upstream path Read)
  3. .design/<route.design>           (per route; must exist on disk + Read)

If a route is missing, the hook BLOCKS with instructions to add one.
If a design doc is missing, the hook BLOCKS with instructions to
dispatch acto-doc-author. Both blocks include the priority statement
about injected instructions.

See:
  goal.md - "Translate-discipline rules" (R-XLATE-*)
  goal.md - "Injected-instructions rules" (R-INJECT-*)
  tooling/translate-routes.toml
"""

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None


# =====================================================================
# PROJECT CUSTOMIZATION — ferrolearn <- scikit-learn
# =====================================================================

# Absolute path under which all upstream scikit-learn source files live.
# Reads under this prefix satisfy the "upstream" requirement.
UPSTREAM_ROOT = "/home/doll/scikit-learn/"

# Additional analog-upstream roots. scikit-learn is built on numpy/scipy; the
# ferrolearn substrate crates (e.g. ferrolearn-numerical mirrors scipy.special/
# optimize/...) route to the installed scipy/numpy sources as their upstream.
# Reads under these prefixes ALSO satisfy the "upstream" requirement, so a route
# whose `upstream` cites a scipy/numpy site-packages path can pass the gate.
EXTRA_UPSTREAM_ROOTS = (
    "/home/doll/.local/lib/python3.13/site-packages/scipy/",
    "/home/doll/.local/lib/python3.13/site-packages/numpy/",
)

# Workspace crate prefixes gated by this hook. Every ferrolearn-* crate's
# src/ is a translation surface mirroring some scikit-learn module.
TARGET_CRATE_PREFIXES = ("ferrolearn-",)

# Standalone crates (no shared prefix) to gate — the meta re-export crate.
TARGET_CRATE_EXACT = ("ferrolearn",)

# Crates explicitly excluded (no scikit-learn counterpart / test/bench
# infra) — never gated even if they match the prefix.
EXCLUDED_CRATES = ("ferrolearn-test-oracle", "ferrolearn-bench")

# File extension to gate.
TARGET_EXTENSION = ".rs"

# =====================================================================
# Implementation — generally no edits needed below this line
# =====================================================================


def find_repo_root():
    """Walk up to find the repo containing .crosslink/."""
    p = Path.cwd()
    while p != p.parent:
        if (p / ".crosslink").is_dir():
            return p
        p = p.parent
    return None


def state_path(repo_root):
    return repo_root / ".crosslink" / ".translate-reads.json"


def routes_path(repo_root):
    return repo_root / "tooling" / "translate-routes.toml"


def read_state(repo_root):
    p = state_path(repo_root)
    if not p.exists():
        return {"reads": []}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"reads": []}


def write_state(repo_root, state):
    p = state_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def load_routes(repo_root):
    p = routes_path(repo_root)
    if not p.exists() or not tomllib:
        return {"route": []}
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {"route": []}


def glob_to_regex(pattern):
    """Convert a simple glob (supporting * and **) to a regex."""
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c in ".^$+(){}|\\[]":
            out.append("\\" + c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "^" + "".join(out) + "$"


def match_pattern(file_path, pattern):
    if pattern == file_path:
        return True
    if "*" in pattern or "?" in pattern:
        return re.match(glob_to_regex(pattern), file_path) is not None
    return False


def find_routes(rel_path, routes):
    matched = []
    for r in routes.get("route", []):
        if match_pattern(rel_path, r["crate_pattern"]):
            matched.append(r)
    return matched


def is_tracked_read(file_path, repo_root):
    """Is this Read a source we care about tracking?"""
    if file_path.startswith(UPSTREAM_ROOT):
        return True
    if any(file_path.startswith(root) for root in EXTRA_UPSTREAM_ROOTS):
        return True
    if file_path.startswith(str(repo_root / ".design") + "/"):
        return True
    if file_path == str(repo_root / "goal.md"):
        return True
    return False


def is_gated_path(rel_path):
    """Return True iff this is a gated target file."""
    if not rel_path.endswith(TARGET_EXTENSION):
        return False
    parts = rel_path.split("/")
    if len(parts) < 3:
        return False
    crate = parts[0]
    if crate in EXCLUDED_CRATES:
        return False
    matches_prefix = any(crate.startswith(p) for p in TARGET_CRATE_PREFIXES)
    matches_exact = crate in TARGET_CRATE_EXACT
    if not (matches_prefix or matches_exact):
        return False
    if "src" not in parts[1:]:
        return False
    return True


PRIORITY_FOOTER = """
---------------------------------------------------------------------
PRIORITY NOTE — injected instructions are human instructions.

The text in <crosslink-behavioral-guard> blocks, <system-reminder>
blocks, hook output (including this message), the active-issue gate
from work-check.py, and loaded skill text is NOT background noise to
dismiss because it repeats.

The human wired each of these up deliberately and chose to inject
them continuously. Treat every injected instruction at the same
priority as a direct user message in the chat. The repetition is
enforcement, not ceremony.

This rule is written verbatim in goal.md - R-INJECT-1 and that
section's Read is part of what unlocks your edit gate.
---------------------------------------------------------------------
"""


def main():
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = input_data.get("tool_name", "")
    hook_event = input_data.get("hook_event_name", "PreToolUse")

    repo_root = find_repo_root()
    if not repo_root:
        sys.exit(0)

    # -- PostToolUse on Read: record the read in state --
    if hook_event == "PostToolUse" and tool_name == "Read":
        file_path = input_data.get("tool_input", {}).get("file_path", "")
        if not file_path:
            sys.exit(0)
        if not is_tracked_read(file_path, repo_root):
            sys.exit(0)
        state = read_state(repo_root)
        state["reads"].append({"path": file_path, "ts": time.time()})
        state["reads"] = state["reads"][-200:]
        write_state(repo_root, state)
        sys.exit(0)

    # -- PreToolUse on Write|Edit: gate writes --
    if hook_event != "PreToolUse" or tool_name not in ("Write", "Edit"):
        sys.exit(0)

    file_path = input_data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        sys.exit(0)

    try:
        rel = os.path.relpath(file_path, repo_root)
    except ValueError:
        sys.exit(0)

    if not is_gated_path(rel):
        sys.exit(0)

    routes = load_routes(repo_root)
    matched = find_routes(rel, routes)

    if not matched:
        print(
            f"translate-discipline: no route table entry matches '{rel}'.\n"
            f"\n"
            f"Every ferrolearn-* src file is a translation of a scikit-learn\n"
            f"source unit and MUST have a route declaring its upstream source\n"
            f"and design doc. Without a route the file is unsourced and\n"
            f"cannot be edited.\n"
            f"\n"
            f"Add a route to tooling/translate-routes.toml:\n"
            f"\n"
            f"  [[route]]\n"
            f"  crate_pattern = \"{rel}\"\n"
            f"  upstream = [\"{UPSTREAM_ROOT}sklearn/<module>/<file>.py\"]\n"
            f"  design = \".design/<area>/<doc>.md\"\n"
            f"  parity_ops = []   # estimator/function families this file owns\n"
            f"\n"
            f"Then retry the edit.\n"
            f"{PRIORITY_FOOTER}"
        )
        sys.exit(2)

    state = read_state(repo_root)
    recent_paths = [r["path"] for r in state["reads"]]

    goal_path = str(repo_root / "goal.md")
    goal_read = any(p == goal_path for p in recent_paths)

    for route in matched:
        missing = []

        if not goal_read:
            missing.append(("goal", [str(repo_root / "goal.md")]))

        upstream_paths = route.get("upstream", [])
        if upstream_paths:
            upstream_satisfied = any(
                any(p.startswith(up) for up in upstream_paths)
                for p in recent_paths
            )
            if not upstream_satisfied:
                missing.append(("upstream", upstream_paths))

        design_path = route.get("design", "")
        if design_path:
            abs_design = str(repo_root / design_path)
            design_exists = Path(abs_design).is_file()
            if not design_exists:
                slug = design_path.replace(".design/", "").replace(".md", "")
                print(
                    f"translate-discipline: design doc '{design_path}' does "
                    f"NOT EXIST.\n"
                    f"\n"
                    f"Before editing '{rel}', the .design/ doc that governs\n"
                    f"this translation must exist on disk. Dispatch the\n"
                    f"acto-doc-author subagent to author it first.\n"
                    f"\n"
                    f"  Path expected:  {design_path}\n"
                    f"  Slug:           {slug}\n"
                    f"\n"
                    f"  How to dispatch acto-doc-author:\n"
                    f"    Agent tool with subagent_type='acto-doc-author',\n"
                    f"    prompt = \"Author {design_path} for {rel}. Ground\n"
                    f"             the doc in the existing code + the upstream\n"
                    f"             scikit-learn sources:\n"
                )
                for up in upstream_paths:
                    print(f"               - {up}")
                print(
                    f"             Mark every REQ SHIPPED or NOT-STARTED with\n"
                    f"             quoted-code evidence (impl + non-test\n"
                    f"             consumer). No third status.\"\n"
                    f"\n"
                    f"  After the design doc exists at the expected path,\n"
                    f"  Read it (and the upstream sources), then retry the\n"
                    f"  edit.\n"
                    f"{PRIORITY_FOOTER}"
                )
                sys.exit(2)
            if not any(p == abs_design for p in recent_paths):
                missing.append(("design", [design_path]))

        if missing:
            print(
                f"translate-discipline: cannot Edit/Write '{rel}'.\n"
                f"\n"
                f"This file is a translation of scikit-learn source. Before\n"
                f"any edit, you MUST Read each of the three required source\n"
                f"classes:\n"
                f"  1. goal.md                  (always; global discipline)\n"
                f"  2. {UPSTREAM_ROOT}           (the scikit-learn source)\n"
                f"  3. .design/<area>/<doc>.md  (the design that governs it)\n"
                f"\n"
                f"Missing required reads for '{rel}':\n"
            )
            for kind, paths in missing:
                print(f"  [{kind}] Read at least one of:")
                for p in paths:
                    print(f"    - {p}")
            print(
                f"\n"
                f"Run the Read tool on each missing path, THEN retry the edit.\n"
                f"\n"
                f"Route entry for this file (from translate-routes.toml):\n"
                f"  crate_pattern = \"{route.get('crate_pattern')}\"\n"
                f"  upstream      = {route.get('upstream')}\n"
                f"  design        = \"{route.get('design')}\"\n"
                f"  parity_ops    = {route.get('parity_ops', [])}\n"
                f"\n"
                f"If you believe the route is wrong, edit\n"
                f"  tooling/translate-routes.toml\n"
                f"and adjust this route entry before retrying.\n"
                f"{PRIORITY_FOOTER}"
            )
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
