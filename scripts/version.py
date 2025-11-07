#!/usr/bin/env python3
"""Version management script for eo-processor.

This script helps maintain version consistency across:
- pyproject.toml
- Cargo.toml
- python/eo_processor/__init__.py (assuming the python-source layout)

Usage:
    python scripts/version.py current         # Show current version
    python scripts/version.py patch           # Increment patch version (0.1.0 -> 0.1.1)
    python scripts/version.py minor           # Increment minor version (0.1.0 -> 0.2.0)
    python scripts/version.py major           # Increment major version (0.1.0 -> 1.0.0)
    python scripts/version.py set 1.2.3       # Set specific version
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Tuple


# --- Utility Functions (Kept as is) ---

def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into major, minor, patch."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-.*)?(?:\+.*)?$", version_str)
    if not match:
        msg = f"Invalid semantic version: {version_str}"
        raise ValueError(msg)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components into semantic version string."""
    return f"{major}.{minor}.{patch}"


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        msg = "pyproject.toml not found"
        raise FileNotFoundError(msg)

    content = pyproject_path.read_text()
    # Matches version = "..." in the [project] section
    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        msg = "Version not found in pyproject.toml"
        raise ValueError(msg)

    return match.group(1)


# --- Update Functions (MODIFIED/NEW) ---

def update_pyproject_version(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update version field in [project]
    new_content = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        fr'\1"{new_version}"',
        content,
        flags=re.MULTILINE,
    )

    pyproject_path.write_text(new_content)
    print(f"Updated pyproject.toml version to {new_version}")


def update_cargo_version(new_version: str) -> None:
    """Update version in Cargo.toml."""
    cargo_path = Path("Cargo.toml")
    if not cargo_path.exists():
        msg = "Cargo.toml not found. Cannot update Rust package version."
        raise FileNotFoundError(msg)

    content = cargo_path.read_text()

    # Update version field in [package] section
    # Use re.sub with a non-greedy match to ensure it only hits the one line
    new_content = re.sub(
        r'(\[package\][\s\S]*?^version\s*=\s*)"[^"]+"',
        fr'\1"{new_version}"',
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    cargo_path.write_text(new_content)
    print(f"Updated Cargo.toml version to {new_version}")


def update_init_version(new_version: str) -> None:
    """Update version in the main Python init file (python/eo_processor/__init__.py)."""
    # Based on your pyproject.toml structure: python-source = "python"
    # and module-name = "eo_processor._core"
    # The user-facing module is likely 'python/eo_processor/__init__.py'
    init_path = Path("python/eo_processor/__init__.py")
    if not init_path.exists():
        print(f"Warning: {init_path} not found. Skipping __version__ update.", file=sys.stderr)
        return

    content = init_path.read_text()

    # Update __version__ field
    new_content = re.sub(
        r'__version__ = "[^"]+"',
        f'__version__ = "{new_version}"',
        content,
    )

    init_path.write_text(new_content)
    print(f"Updated {init_path} version to {new_version}")


def update_all_versions(new_version: str) -> None:
    """Update version in all files."""
    update_pyproject_version(new_version)
    update_cargo_version(new_version) # <-- NEW RUST VERSION UPDATE
    update_init_version(new_version)
    # Removed the update_server_version function call


# --- Main Logic (Kept as is, but relies on new functions) ---

def increment_version(increment_type: str) -> str:
    """Increment version based on type (major, minor, patch)."""
    current = get_current_version()
    major, minor, patch = parse_version(current)

    if increment_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif increment_type == "minor":
        minor += 1
        patch = 0
    elif increment_type == "patch":
        patch += 1
    else:
        msg = f"Invalid increment type: {increment_type}"
        raise ValueError(msg)

    return format_version(major, minor, patch)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage eo-processor version")
    parser.add_argument(
        "action",
        choices=["current", "patch", "minor", "major", "set"],
        help="Action to perform",
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to set (required for 'set' action)",
    )

    args = parser.parse_args()

    try:
        if args.action == "current":
            current = get_current_version()
            print(f"Current version: {current}")

        elif args.action in ["patch", "minor", "major"]:
            new_version = increment_version(args.action)
            print(f"Incrementing {args.action} version...")
            print(f"Current: {get_current_version()}")
            print(f"New: {new_version}")
            update_all_versions(new_version)
            print("Version updated successfully!")

        elif args.action == "set":
            if not args.version:
                print("Error: Version required for 'set' action", file=sys.stderr)
                sys.exit(1)

            # Validate version format
            parse_version(args.version)

            print(f"Setting version to {args.version}...")
            print(f"Current: {get_current_version()}")
            update_all_versions(args.version)
            print("Version updated successfully!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
