#!/usr/bin/env python3
"""
Apple Photos Library Comparison Tool

Compares two Photos libraries and copies unique files from secondary to output.
Supports large libraries (>100GB) with efficient size+hash matching.

Usage:
    python photolib_compare.py                    # GUI mode - dialogs to pick folders
    python photolib_compare.py --main /path/to/Main.photoslibrary \
                               --secondary /path/to/Secondary.photoslibrary \
                               --output /path/to/output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif", ".tif", ".tiff",
    ".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".3gp",
    ".aae",  # Edit sidecar files
    ".cr2", ".nef", ".arw", ".dng", ".raw",  # RAW formats
}

ORIGINALS_FOLDERS = ["originals", "masters"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FileEntry:
    """A file in a Photos library."""
    path: str
    size: int
    mtime: float
    hash: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing two libraries."""
    matched: List[FileEntry]
    unique: List[FileEntry]
    errors: List[Tuple[str, str]]
    bytes_to_copy: int


# ============================================================================
# UTILITIES
# ============================================================================

def human_bytes(n: int) -> str:
    """Format bytes as human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def format_eta(seconds: float) -> str:
    """Format seconds as human readable time."""
    if seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# PROGRESS REPORTING
# ============================================================================

class ProgressReporter:
    """Rich progress reporter with ETA calculation."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.last_update: float = 0
        self.update_interval: float = 0.3  # Update every 300ms
        self.operation: str = ""
        self.total_bytes: int = 0

    def start(self, operation: str):
        """Start a new operation."""
        self.start_time = time.time()
        self.last_update = 0
        self.operation = operation
        self.total_bytes = 0
        print(f"\n[{operation}]", flush=True)

    def update(self, current: int, total: int, detail: str = "", force: bool = False):
        """Update progress (rate-limited to avoid flooding)."""
        now = time.time()
        if not force and now - self.last_update < self.update_interval:
            return
        self.last_update = now

        elapsed = now - self.start_time
        rate = current / max(elapsed, 0.001)
        remaining = (total - current) / max(rate, 0.001) if rate > 0 else -1
        pct = (current / max(total, 1)) * 100

        # Build progress line
        line = f"\r  {pct:5.1f}% | {current:,}/{total:,} | {rate:,.0f}/sec | ETA {format_eta(remaining)}"
        if detail:
            # Truncate detail if too long
            max_detail = 30
            if len(detail) > max_detail:
                detail = detail[:max_detail-3] + "..."
            line += f" | {detail}"

        # Clear line and print
        print(line.ljust(90), end="", flush=True)

    def finish(self, summary: str):
        """Complete the operation."""
        elapsed = time.time() - self.start_time
        print(f"\n  Done in {format_eta(elapsed)}: {summary}", flush=True)


# ============================================================================
# PHASE 1: DISCOVERY
# ============================================================================

def find_originals_dir(library_path: Path) -> Path:
    """Find the originals/masters folder in a Photos library."""
    for folder_name in ORIGINALS_FOLDERS:
        # Check common locations
        for candidate in [
            library_path / folder_name,
            library_path / folder_name.capitalize(),
            library_path / folder_name.upper(),
        ]:
            if candidate.exists() and candidate.is_dir():
                return candidate

    # Search recursively (depth limited)
    for root, dirs, _ in os.walk(library_path):
        depth = len(Path(root).relative_to(library_path).parts)
        if depth > 2:
            dirs.clear()  # Don't go deeper
            continue
        for folder_name in ORIGINALS_FOLDERS:
            for d in dirs:
                if d.lower() == folder_name:
                    candidate = Path(root) / d
                    if candidate.is_dir():
                        return candidate

    raise ValueError(
        f"Could not find originals/masters folder in {library_path}\n"
        "This doesn't appear to be a valid Photos library."
    )


def check_library_access(library_path: Path) -> Path:
    """Verify we can access the library. Returns originals dir."""
    if not library_path.exists():
        raise FileNotFoundError(f"Library not found: {library_path}")

    originals = find_originals_dir(library_path)

    # Try to list directory
    try:
        items = list(originals.iterdir())
        if not items:
            print(f"  Warning: {originals} appears to be empty")
    except PermissionError:
        raise PermissionError(
            f"Cannot access library: {library_path}\n\n"
            "To fix this on macOS:\n"
            "1. Open System Settings > Privacy & Security > Full Disk Access\n"
            "2. Enable access for Terminal (or your Python environment)\n"
            "3. Restart Terminal and try again"
        )

    # Try to read a sample file
    sample_read = False
    for item in originals.rglob("*"):
        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
            try:
                with open(item, "rb") as f:
                    f.read(1024)
                sample_read = True
                break
            except PermissionError:
                raise PermissionError(
                    f"Cannot read files in library: {library_path}\n\n"
                    "Full Disk Access is required. See instructions above."
                )

    if not sample_read:
        print(f"  Warning: No media files found in {originals}")

    return originals


def discover_files(originals_dir: Path, progress: ProgressReporter) -> List[FileEntry]:
    """Walk directory and collect file entries."""
    entries = []
    total_bytes = 0

    # First pass: count files for progress
    file_list = []
    for root, _, files in os.walk(originals_dir):
        for f in files:
            if Path(f).suffix.lower() in MEDIA_EXTENSIONS:
                file_list.append(Path(root) / f)

    total = len(file_list)
    progress.start(f"Discovering {originals_dir.parent.name}")

    for i, file_path in enumerate(file_list):
        try:
            stat = file_path.stat()
            entries.append(FileEntry(
                path=str(file_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
            ))
            total_bytes += stat.st_size
            progress.update(i + 1, total, file_path.name)
        except (PermissionError, OSError) as e:
            # Skip files we can't access
            progress.update(i + 1, total, f"Skipped: {file_path.name}")

    progress.finish(f"{len(entries):,} files ({human_bytes(total_bytes)})")
    return entries


def save_manifest(entries: List[FileEntry], path: Path) -> None:
    """Save entries to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def load_manifest(path: Path) -> List[FileEntry]:
    """Load entries from JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(FileEntry(**data))
    return entries


# ============================================================================
# PHASE 2: COMPARISON
# ============================================================================

def find_unique_files(
    main_entries: List[FileEntry],
    secondary_entries: List[FileEntry],
    progress: ProgressReporter
) -> ComparisonResult:
    """Find files in secondary that don't exist in main."""

    # Build size index from main library
    main_by_size: Dict[int, List[FileEntry]] = {}
    for entry in main_entries:
        main_by_size.setdefault(entry.size, []).append(entry)

    # Hash cache to avoid recomputing
    hash_cache: Dict[str, str] = {}

    def get_hash(path: str) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    matched = []
    unique = []
    errors = []
    bytes_to_copy = 0

    progress.start("Comparing libraries")
    total = len(secondary_entries)

    for i, sec_entry in enumerate(secondary_entries):
        file_name = Path(sec_entry.path).name

        # Look for size matches in main
        candidates = main_by_size.get(sec_entry.size, [])

        if not candidates:
            # No size match - definitely unique
            unique.append(sec_entry)
            bytes_to_copy += sec_entry.size
            progress.update(i + 1, total, f"Unique: {file_name}")
            continue

        # Size match found - need to compare hashes
        try:
            progress.update(i + 1, total, f"Hashing: {file_name}")
            sec_hash = get_hash(sec_entry.path)

            found_match = False
            for main_entry in candidates:
                try:
                    main_hash = get_hash(main_entry.path)
                    if sec_hash == main_hash:
                        found_match = True
                        break
                except Exception:
                    # Can't read main file - continue checking others
                    continue

            if found_match:
                matched.append(sec_entry)
                progress.update(i + 1, total, f"Matched: {file_name}")
            else:
                unique.append(sec_entry)
                bytes_to_copy += sec_entry.size
                progress.update(i + 1, total, f"Unique: {file_name}")

        except Exception as e:
            errors.append((sec_entry.path, str(e)))
            progress.update(i + 1, total, f"Error: {file_name}")

    progress.finish(
        f"{len(matched):,} matched, {len(unique):,} unique ({human_bytes(bytes_to_copy)} to copy)"
        + (f", {len(errors)} errors" if errors else "")
    )

    return ComparisonResult(
        matched=matched,
        unique=unique,
        errors=errors,
        bytes_to_copy=bytes_to_copy,
    )


# ============================================================================
# PHASE 3: COPYING
# ============================================================================

def copy_unique_files(
    unique_files: List[FileEntry],
    secondary_originals: Path,
    output_dir: Path,
    progress: ProgressReporter
) -> Tuple[int, List[Tuple[str, str]]]:
    """Copy unique files to output directory, preserving structure."""

    copied_dir = output_dir / "copied"
    copied_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    errors = []
    total = len(unique_files)

    progress.start("Copying unique files")

    for i, entry in enumerate(unique_files):
        src_path = Path(entry.path)
        file_name = src_path.name

        try:
            # Compute relative path from secondary originals
            rel_path = src_path.relative_to(secondary_originals)
            dest_path = copied_dir / rel_path

            # Create parent directories
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            progress.update(i + 1, total, file_name)
            shutil.copy2(src_path, dest_path)
            copied_count += 1

        except Exception as e:
            errors.append((entry.path, str(e)))
            progress.update(i + 1, total, f"Error: {file_name}")

    # Calculate total bytes copied
    total_bytes = sum(e.size for e in unique_files[:copied_count])

    progress.finish(
        f"{copied_count:,} files copied ({human_bytes(total_bytes)})"
        + (f", {len(errors)} errors" if errors else "")
    )

    return copied_count, errors


# ============================================================================
# GUI
# ============================================================================

def select_folders_gui() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Show GUI dialogs to select libraries and output folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("ERROR: tkinter not available.")
        print("Use command-line arguments: --main, --secondary, --output")
        return None, None, None

    root = tk.Tk()
    root.withdraw()

    # Make dialogs appear in front
    root.attributes("-topmost", True)

    def pick_library(title: str) -> Optional[Path]:
        """Pick a .photoslibrary bundle or folder."""
        # On macOS, .photoslibrary is a folder
        path = filedialog.askdirectory(title=title)
        return Path(path) if path else None

    def pick_folder(title: str) -> Optional[Path]:
        """Pick an output folder."""
        path = filedialog.askdirectory(title=title)
        return Path(path) if path else None

    # Step 1: Main library
    messagebox.showinfo(
        "Step 1 of 3",
        "Select your MAIN Photos library.\n\n"
        "This is your primary/reference library.\n"
        "Files in this library will be considered 'already have'."
    )
    main = pick_library("Select MAIN Photos Library")
    if not main:
        messagebox.showwarning("Cancelled", "No main library selected. Exiting.")
        root.destroy()
        return None, None, None

    # Step 2: Secondary library
    messagebox.showinfo(
        "Step 2 of 3",
        "Select your SECONDARY Photos library.\n\n"
        "This is the library to find unique files in.\n"
        "Files in this library but NOT in main will be copied."
    )
    secondary = pick_library("Select SECONDARY Photos Library")
    if not secondary:
        messagebox.showwarning("Cancelled", "No secondary library selected. Exiting.")
        root.destroy()
        return None, None, None

    # Step 3: Output folder
    messagebox.showinfo(
        "Step 3 of 3",
        "Select OUTPUT folder.\n\n"
        "Unique files will be copied here.\n"
        "Manifests will also be saved here for resumability."
    )
    output = pick_folder("Select OUTPUT Folder")
    if not output:
        messagebox.showwarning("Cancelled", "No output folder selected. Exiting.")
        root.destroy()
        return None, None, None

    # Confirm
    msg = (
        f"MAIN Library:\n  {main}\n\n"
        f"SECONDARY Library:\n  {secondary}\n\n"
        f"OUTPUT Folder:\n  {output}\n\n"
        f"Files in SECONDARY but not in MAIN will be copied to OUTPUT.\n\n"
        f"Proceed?"
    )
    if not messagebox.askyesno("Confirm Selection", msg):
        root.destroy()
        return None, None, None

    root.destroy()
    return main, secondary, output


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two Apple Photos libraries and copy unique files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python photolib_compare.py                    # GUI mode
  python photolib_compare.py --main ~/Pictures/Main.photoslibrary \\
                             --secondary ~/Pictures/Old.photoslibrary \\
                             --output ~/Desktop/PhotoDiff
        """
    )
    parser.add_argument("--main", type=Path, help="Path to main Photos library")
    parser.add_argument("--secondary", type=Path, help="Path to secondary Photos library")
    parser.add_argument("--output", type=Path, help="Path to output folder")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copying files (comparison only)")

    args = parser.parse_args()

    # Determine paths
    if args.main and args.secondary and args.output:
        main_lib = args.main
        secondary_lib = args.secondary
        output_dir = args.output
    elif args.main or args.secondary or args.output:
        print("ERROR: Must provide all three: --main, --secondary, --output")
        return 1
    else:
        # GUI mode
        result = select_folders_gui()
        if result[0] is None:
            return 1
        main_lib, secondary_lib, output_dir = result

    print("\n" + "=" * 60)
    print("Apple Photos Library Comparison Tool")
    print("=" * 60)
    print(f"\nMain Library:      {main_lib}")
    print(f"Secondary Library: {secondary_lib}")
    print(f"Output Folder:     {output_dir}")

    progress = ProgressReporter()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)

    main_manifest_path = manifest_dir / "main_library.jsonl"
    secondary_manifest_path = manifest_dir / "secondary_library.jsonl"
    unique_manifest_path = manifest_dir / "unique_files.jsonl"

    # -------------------------------------------------------------------------
    # Phase 1: Discovery
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PHASE 1: Discovery")
    print("-" * 40)

    # Main library
    if main_manifest_path.exists():
        print(f"\nLoading existing manifest: {main_manifest_path}")
        main_entries = load_manifest(main_manifest_path)
        total_bytes = sum(e.size for e in main_entries)
        print(f"  Loaded {len(main_entries):,} files ({human_bytes(total_bytes)})")
    else:
        print("\nChecking access to main library...")
        main_originals = check_library_access(main_lib)
        main_entries = discover_files(main_originals, progress)
        save_manifest(main_entries, main_manifest_path)
        print(f"  Manifest saved: {main_manifest_path}")

    # Secondary library
    if secondary_manifest_path.exists():
        print(f"\nLoading existing manifest: {secondary_manifest_path}")
        secondary_entries = load_manifest(secondary_manifest_path)
        total_bytes = sum(e.size for e in secondary_entries)
        print(f"  Loaded {len(secondary_entries):,} files ({human_bytes(total_bytes)})")
    else:
        print("\nChecking access to secondary library...")
        secondary_originals = check_library_access(secondary_lib)
        secondary_entries = discover_files(secondary_originals, progress)
        save_manifest(secondary_entries, secondary_manifest_path)
        print(f"  Manifest saved: {secondary_manifest_path}")

    # We need secondary_originals for copying later
    secondary_originals = check_library_access(secondary_lib)

    # -------------------------------------------------------------------------
    # Phase 2: Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PHASE 2: Comparison")
    print("-" * 40)

    if unique_manifest_path.exists():
        print(f"\nLoading existing comparison: {unique_manifest_path}")
        unique_entries = load_manifest(unique_manifest_path)
        total_bytes = sum(e.size for e in unique_entries)
        print(f"  Loaded {len(unique_entries):,} unique files ({human_bytes(total_bytes)} to copy)")
    else:
        result = find_unique_files(main_entries, secondary_entries, progress)
        unique_entries = result.unique
        save_manifest(unique_entries, unique_manifest_path)
        print(f"  Results saved: {unique_manifest_path}")

        if result.errors:
            print(f"\n  Errors during comparison ({len(result.errors)}):")
            for path, error in result.errors[:5]:
                print(f"    - {Path(path).name}: {error}")
            if len(result.errors) > 5:
                print(f"    ... and {len(result.errors) - 5} more")

    # -------------------------------------------------------------------------
    # Phase 3: Copying
    # -------------------------------------------------------------------------
    if args.skip_copy:
        print("\n" + "-" * 40)
        print("PHASE 3: Copying (SKIPPED)")
        print("-" * 40)
        print("\n  --skip-copy flag set. No files copied.")
    elif not unique_entries:
        print("\n" + "-" * 40)
        print("PHASE 3: Copying")
        print("-" * 40)
        print("\n  No unique files to copy!")
    else:
        print("\n" + "-" * 40)
        print("PHASE 3: Copying")
        print("-" * 40)

        copied_count, copy_errors = copy_unique_files(
            unique_entries,
            secondary_originals,
            output_dir,
            progress
        )

        if copy_errors:
            print(f"\n  Errors during copy ({len(copy_errors)}):")
            for path, error in copy_errors[:5]:
                print(f"    - {Path(path).name}: {error}")
            if len(copy_errors) > 5:
                print(f"    ... and {len(copy_errors) - 5} more")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nMain library:      {len(main_entries):,} files")
    print(f"Secondary library: {len(secondary_entries):,} files")
    print(f"Unique in secondary: {len(unique_entries):,} files")

    if unique_entries and not args.skip_copy:
        print(f"\nCopied files location: {output_dir / 'copied'}")

    print(f"\nManifests saved to: {manifest_dir}")
    print("  (Delete manifests to re-scan libraries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
