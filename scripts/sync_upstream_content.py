#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TUTORIALS_REPO_URL = "https://github.com/evogytis/baltic.git"
DEFAULT_EXAMPLES_REPO_URL = "https://github.com/phylo-baltic/baltic-examples.git"
DEFAULT_TUTORIALS_CLONE_DIR = PROJECT_ROOT / ".cache" / "evogytis-baltic"
DEFAULT_EXAMPLES_CLONE_DIR = PROJECT_ROOT / ".cache" / "baltic-examples"

EXAMPLES_DEST = PROJECT_ROOT / "source" / "baltic-examples"
TUTORIALS_DEST = PROJECT_ROOT / "source" / "baltic-tutorials"

EXAMPLES_PATH_CANDIDATES = ("",)
TUTORIALS_PATH_CANDIDATES = (
    "docs/tutorials",
    "tutorials",
)

SKIP_DIRS = {".git", ".github", "__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
TUTORIAL_SOURCE_EXTS = {".py", ".ipynb"}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def clean_dir(path: Path) -> None:
    path = ensure_inside_project(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clone_repo(repo_url: str, clone_dir: Path, ref: str | None) -> None:
    clone_dir = ensure_inside_project(clone_dir)
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([repo_url, str(clone_dir)])
    run(cmd)


def find_source_dir(repo_dir: Path, configured: str | None, candidates: tuple[str, ...], label: str) -> Path:
    if configured:
        source_dir = repo_dir / configured
        if source_dir.is_dir():
            return source_dir
        raise SystemExit(f"Configured {label} path does not exist in upstream clone: {configured}")

    for rel_path in candidates:
        source_dir = repo_dir / rel_path if rel_path else repo_dir
        if source_dir.is_dir():
            return source_dir

    tried = ", ".join(candidate or "<repo root>" for candidate in candidates)
    raise SystemExit(f"Could not find upstream {label} directory. Tried: {tried}")


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def copy_tree_contents(source_dir: Path, dest_dir: Path) -> None:
    clean_dir(dest_dir)

    for source in source_dir.rglob("*"):
        rel = source.relative_to(source_dir)
        if should_skip(rel):
            continue

        dest = dest_dir / rel
        if source.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def copy_flat_tutorial_files(source_dir: Path, dest_dir: Path) -> None:
    clean_dir(dest_dir)

    files = [
        p for p in source_dir.iterdir()
        if p.is_file() and not should_skip(p.relative_to(source_dir))
    ]
    tutorial_sources = [p for p in files if p.suffix.lower() in TUTORIAL_SOURCE_EXTS]
    if not tutorial_sources:
        raise SystemExit(
            f"No flat tutorial .py or .ipynb files found in {source_dir}. "
            "If upstream moved tutorials into another folder, pass --tutorials-path."
        )

    for source in sorted(files, key=lambda p: p.name.lower()):
        shutil.copy2(source, dest_dir / source.name)


def run_builders() -> None:
    run([sys.executable, "scripts/build_examples.py"], cwd=PROJECT_ROOT)
    run([sys.executable, "scripts/build_tutorials.py"], cwd=PROJECT_ROOT)
    run([sys.executable, "scripts/build_tags.py"], cwd=PROJECT_ROOT)
    run([sys.executable, "scripts/build_uses.py"], cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone upstream examples/tutorials repos, copy content into local "
            "source/baltic-* folders, then regenerate Sphinx source pages."
        )
    )
    parser.add_argument("--tutorials-repo-url", default=DEFAULT_TUTORIALS_REPO_URL)
    parser.add_argument("--examples-repo-url", default=DEFAULT_EXAMPLES_REPO_URL)
    parser.add_argument("--tutorials-ref", help="Optional tutorials repo branch, tag, or commit to clone.")
    parser.add_argument("--examples-ref", help="Optional examples repo branch, tag, or commit to clone.")
    parser.add_argument("--tutorials-clone-dir", type=Path, default=DEFAULT_TUTORIALS_CLONE_DIR)
    parser.add_argument("--examples-clone-dir", type=Path, default=DEFAULT_EXAMPLES_CLONE_DIR)
    parser.add_argument("--examples-path", help="Path inside upstream repo for examples.")
    parser.add_argument("--tutorials-path", help="Path inside upstream repo for tutorials.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only sync source files; do not run build_examples.py/build_tutorials.py/build_uses.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone and resolve upstream paths, but do not copy files or run builders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples_clone_dir = args.examples_clone_dir
    if not examples_clone_dir.is_absolute():
        examples_clone_dir = PROJECT_ROOT / examples_clone_dir

    tutorials_clone_dir = args.tutorials_clone_dir
    if not tutorials_clone_dir.is_absolute():
        tutorials_clone_dir = PROJECT_ROOT / tutorials_clone_dir

    clone_repo(args.examples_repo_url, examples_clone_dir, args.examples_ref)
    clone_repo(args.tutorials_repo_url, tutorials_clone_dir, args.tutorials_ref)

    examples_src = find_source_dir(examples_clone_dir, args.examples_path, EXAMPLES_PATH_CANDIDATES, "examples")
    tutorials_src = find_source_dir(tutorials_clone_dir, args.tutorials_path, TUTORIALS_PATH_CANDIDATES, "tutorials")

    if args.dry_run:
        print(f"Would sync examples: {examples_src} -> {EXAMPLES_DEST}")
        print(f"Would sync tutorials: {tutorials_src} -> {TUTORIALS_DEST}")
        print("Dry run complete; no source files were changed.")
        return

    print(f"Syncing examples: {examples_src} -> {EXAMPLES_DEST}")
    copy_tree_contents(examples_src, EXAMPLES_DEST)

    print(f"Syncing tutorials: {tutorials_src} -> {TUTORIALS_DEST}")
    copy_flat_tutorial_files(tutorials_src, TUTORIALS_DEST)

    if not args.skip_build:
        run_builders()

    print("Done.")


if __name__ == "__main__":
    main()
