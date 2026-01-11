#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif", ".tif", ".tiff"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts", ".3gp"}
OTHER_MEDIA_EXTS = {".aae"}  # edit sidecar; countable but not primary media

MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | OTHER_MEDIA_EXTS

# Known originals roots inside Photos libraries (varies by Photos version)
CANDIDATE_ORIGINALS_DIRS = [
    "originals",
    "masters",
]

@dataclass
class LibraryInfo:
    name: str
    root: Path
    originals_dirs: List[Path]
    total_files: int
    total_bytes: int
    ext_counts: Dict[str, int]
    image_files: int
    video_files: int
    other_media_files: int
    sample_file: Optional[Path] = None
    volume_root: Optional[Path] = None
    volume_free_bytes: Optional[int] = None


def human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        if f < step:
            return f"{f:,.2f} {u}"
        f /= step
    return f"{f:,.2f} EiB"


def ensure_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    testfile = path / f".write_test_{int(time.time())}.tmp"
    try:
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        testfile.unlink(missing_ok=True)
    except Exception as e:
        raise PermissionError(
            f"Output directory is not writable:\n  {path}\n\n"
            "Fix ideas:\n"
            "- Ensure the external SSD is mounted read/write\n"
            "- Check permissions for this folder\n"
            "- Grant Full Disk Access to Terminal (System Settings → Privacy & Security → Full Disk Access)\n"
        ) from e


def get_volume_root(p: Path) -> Path:
    # Walk upwards until we find a mount point.
    # On macOS external volumes typically live under /Volumes/<Name>
    p = p.resolve()
    cur = p
    while cur.parent != cur:
        if os.path.ismount(cur):
            return cur
        cur = cur.parent
    return p.anchor and Path(p.anchor) or p  # fallback


def get_free_space_bytes(path: Path) -> int:
    st = os.statvfs(str(path))
    return int(st.f_frsize * st.f_bavail)


def is_photos_library(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    # Common marker files/folders (not guaranteed, but helpful)
    candidates = ["database", "resources", "Masters", "masters", "originals", "Originals"]
    return any((path / c).exists() for c in candidates)


def find_originals_dirs(library_root: Path) -> List[Path]:
    found: List[Path] = []
    for rel in CANDIDATE_ORIGINALS_DIRS:
        p = library_root / rel
        if p.exists() and p.is_dir():
            found.append(p)
    return found


def walk_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                yield Path(dirpath) / fn


def discover_library(
    name: str,
    root: Path,
    progress_every: int = 5000,
    manifest_path: Optional[Path] = None,
) -> LibraryInfo:
    if not root.exists():
        raise FileNotFoundError(f"{name} library path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{name} library path is not a directory: {root}")

    originals_dirs = find_originals_dirs(root)

    # If not found at top level, try a shallow search (depth <= 3)
    if not originals_dirs:
        targets = set(CANDIDATE_ORIGINALS_DIRS)
        for dirpath, dirnames, _ in os.walk(root):
            dp = Path(dirpath)
            rel_parts = dp.relative_to(root).parts
            if len(rel_parts) > 3:
                dirnames[:] = []
                continue
            for d in list(dirnames):
                if d.lower() in targets:
                    cand = dp / d
                    if cand.is_dir():
                        originals_dirs.append(cand)
        originals_dirs = sorted(set(originals_dirs))

    if not originals_dirs:
        raise RuntimeError(
            f"Could not locate originals folders inside {name} library.\n"
            f"Tried: {CANDIDATE_ORIGINALS_DIRS} at root and shallow search.\n"
            f"Library root: {root}"
        )

    vol_root = get_volume_root(root)
    vol_free = get_free_space_bytes(vol_root)

    total_files = 0
    total_bytes = 0
    ext_counts: Dict[str, int] = {}
    sample_file: Optional[Path] = None

    image_files = 0
    video_files = 0
    other_media_files = 0

    t0 = time.time()
    last_print = t0

    manifest_f = None
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = open(manifest_path, "w", encoding="utf-8")
        manifest_f.write(
            "# JSONL manifest: one file per line with path/size/mtime\n"
        )

    for p in walk_files(originals_dirs):
        try:
            st = p.stat()
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied reading file metadata:\n  {p}\n\n"
                "Fix: System Settings → Privacy & Security → Full Disk Access → enable Terminal (or the app you're using).\n"
                "Also ensure the external drive is mounted and accessible."
            ) from e

        total_files += 1
        total_bytes += st.st_size

        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if ext in IMAGE_EXTS:
            image_files += 1
        elif ext in VIDEO_EXTS:
            video_files += 1
        elif ext in OTHER_MEDIA_EXTS:
            other_media_files += 1

        if sample_file is None and ext in (IMAGE_EXTS | VIDEO_EXTS) and st.st_size > 1024 * 32:
            sample_file = p

        if manifest_f:
            record = {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}
            manifest_f.write(json.dumps(record) + "\n")

        if total_files % progress_every == 0:
            now = time.time()
            if now - last_print >= 1.0:
                rate = total_files / max(now - t0, 1e-6)
                print(
                    f"[{name}] Discovered {total_files:,} files, {human_bytes(total_bytes)} "
                    f"({rate:,.0f} files/sec)",
                    flush=True,
                )
                last_print = now
    if manifest_f:
        manifest_f.close()

    # Validate we can open/read a sample media file (deep read access)
    if sample_file:
        try:
            with open(sample_file, "rb") as f:
                _ = f.read(1024)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied opening a media file:\n  {sample_file}\n\n"
                "Fix: System Settings → Privacy & Security → Full Disk Access → enable Terminal (or the app you're using).\n"
                "Also ensure the external drive is mounted and accessible."
            ) from e

    return LibraryInfo(
        name=name,
        root=root,
        originals_dirs=originals_dirs,
        total_files=total_files,
        total_bytes=total_bytes,
        ext_counts=dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        image_files=image_files,
        video_files=video_files,
        other_media_files=other_media_files,
        sample_file=sample_file,
        volume_root=vol_root,
        volume_free_bytes=vol_free,
    )


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_ext_csv(path: Path, ext_counts: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["extension", "count"])
        for ext, cnt in ext_counts.items():
            w.writerow([ext or "(no_ext)", cnt])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def estimate_index_size_bytes(master_total_files: int) -> int:
    # Very rough: SQLite index storing path/size/mtime/hash etc.
    # Could vary widely; assume ~250 bytes/record average + overhead.
    return int(master_total_files * 250)


def gui_select_paths(
    master: Optional[Path], secondary: Optional[Path], outdir: Optional[Path]
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        from tkinter import messagebox
    except Exception as e:
        raise RuntimeError(
            "GUI selection requires Tkinter (usually included with Python on macOS)."
        ) from e

    root = tk.Tk()
    root.withdraw()
    root.update()

    def pick_dir(title: str, must_exist: bool = True) -> Optional[Path]:
        path = filedialog.askdirectory(title=title, mustexist=must_exist)
        return Path(path) if path else None

    def pick_photos_library(title: str) -> Optional[Path]:
        while True:
            p = pick_dir(title=title, must_exist=True)
            if p is None:
                return None
            if p.name.endswith(".photoslibrary") and is_photos_library(p):
                return p
            msg = (
                "This folder does not look like a standard Photos library bundle.\n\n"
                f"Selected: {p}\n\n"
                "Continue anyway?"
            )
            if messagebox.askyesno("Confirm Library Selection", msg):
                return p

    if master is None:
        master = pick_photos_library("Select Master .photoslibrary")
    if secondary is None:
        secondary = pick_photos_library("Select Secondary .photoslibrary")
    if outdir is None:
        outdir = pick_dir("Select Output Folder (you can create a new folder)", must_exist=False)

    if master and secondary and outdir:
        confirm_msg = (
            "Please confirm your selections:\n\n"
            f"Master:    {master}\n"
            f"Secondary: {secondary}\n"
            f"Outdir:    {outdir}"
        )
        if not messagebox.askokcancel("Confirm Paths", confirm_msg):
            root.destroy()
            return None, None, None

    root.destroy()
    return master, secondary, outdir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0: Preflight + discovery for comparing two Photos libraries."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="Preflight + discovery (no hashing).")
    p.add_argument("--master", default=None, help="Path to Master .photoslibrary")
    p.add_argument("--secondary", default=None, help="Path to Secondary .photoslibrary")
    p.add_argument("--outdir", default=None, help="Output directory (e.g. on your external SSD).")
    p.add_argument("--gui", action="store_true", help="Select folders via a GUI.")
    p.add_argument("--strict", action="store_true", help="Fail if paths do not look like Photos libraries.")
    p.add_argument("--progress-every", type=int, default=5000, help="Print progress every N files.")

    # Optional counts from Photos.app for sanity checking
    p.add_argument("--master-photos-count", type=int, default=None)
    p.add_argument("--master-videos-count", type=int, default=None)
    p.add_argument("--secondary-photos-count", type=int, default=None)
    p.add_argument("--secondary-videos-count", type=int, default=None)

    args = parser.parse_args()

    if args.cmd != "discover":
        return 1

    master = Path(args.master) if args.master else None
    secondary = Path(args.secondary) if args.secondary else None
    outdir = Path(args.outdir) if args.outdir else None

    if args.gui or not (master and secondary and outdir):
        try:
            master, secondary, outdir = gui_select_paths(master, secondary, outdir)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if not (master and secondary and outdir):
        print(
            "ERROR: Missing required paths. Provide --master, --secondary, --outdir or use --gui.",
            file=sys.stderr,
        )
        return 2

    # Ensure we never write inside a Photos library bundle.
    for name, pth in [("Master", master), ("Secondary", secondary)]:
        try:
            out_resolved = outdir.resolve()
            lib_resolved = pth.resolve()
        except FileNotFoundError:
            out_resolved = outdir
            lib_resolved = pth
        if out_resolved == lib_resolved or out_resolved in lib_resolved.parents:
            print(
                f"ERROR: Outdir is inside {name} library bundle. Choose a different output folder.\n"
                f"  Outdir: {outdir}\n  {name}: {pth}",
                file=sys.stderr,
            )
            return 2

    # Prepare output structure (and verify write access)
    ensure_writable_dir(outdir)
    logs_dir = outdir / "logs"
    reports_dir = outdir / "reports"
    index_dir = outdir / "index"
    manifest_dir = outdir / "manifests"
    for d in [logs_dir, reports_dir, index_dir]:
        ensure_writable_dir(d)
    ensure_writable_dir(manifest_dir)

    print("=== Phase 0: Preflight + Discovery ===", flush=True)
    print(f"Master:    {master}", flush=True)
    print(f"Secondary: {secondary}", flush=True)
    print(f"Outdir:    {outdir}", flush=True)
    print(f"Reports:   {reports_dir}", flush=True)
    print(f"Index dir: {index_dir} (Phase 1 will store index here)", flush=True)
    print("", flush=True)

    # Non-fatal warnings (or strict failures)
    for name, pth in [("Master", master), ("Secondary", secondary)]:
        if not pth.exists():
            print(f"ERROR: {name} path does not exist: {pth}", file=sys.stderr)
            return 2
        if not pth.is_dir():
            print(f"ERROR: {name} path is not a directory: {pth}", file=sys.stderr)
            return 2
        if not pth.name.endswith(".photoslibrary"):
            msg = f"{name} does not end with .photoslibrary: {pth.name}"
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 2
            print(f"WARNING: {msg}", flush=True)
        if not is_photos_library(pth):
            msg = f"{name} does not look like a standard Photos library bundle: {pth}"
            if args.strict:
                print(f"ERROR: {msg}", file=sys.stderr)
                return 2
            print(f"WARNING: {msg}", flush=True)

    try:
        master_manifest = manifest_dir / "master_files.jsonl"
        secondary_manifest = manifest_dir / "secondary_files.jsonl"
        master_info = discover_library(
            "Master",
            master,
            progress_every=args.progress_every,
            manifest_path=master_manifest,
        )
        secondary_info = discover_library(
            "Secondary",
            secondary,
            progress_every=args.progress_every,
            manifest_path=secondary_manifest,
        )
    except Exception as e:
        print("\nERROR during discovery:\n", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 3

    # Build report payload
    est_index = estimate_index_size_bytes(master_info.total_files)

    report = {
        "generated_at_epoch": int(time.time()),
        "paths": {
            "master": str(master_info.root),
            "secondary": str(secondary_info.root),
            "outdir": str(outdir),
            "reports_dir": str(reports_dir),
            "index_dir": str(index_dir),
            "manifest_dir": str(manifest_dir),
            "master_manifest": str(master_manifest),
            "secondary_manifest": str(secondary_manifest),
        },
        "master": {
            "root": str(master_info.root),
            "originals_dirs": [str(p) for p in master_info.originals_dirs],
            "total_files": master_info.total_files,
            "total_bytes": master_info.total_bytes,
            "total_size_human": human_bytes(master_info.total_bytes),
            "image_files": master_info.image_files,
            "video_files": master_info.video_files,
            "other_media_files": master_info.other_media_files,
            "ext_counts": master_info.ext_counts,
            "sample_file": str(master_info.sample_file) if master_info.sample_file else None,
            "volume_root": str(master_info.volume_root) if master_info.volume_root else None,
            "volume_free_bytes": master_info.volume_free_bytes,
            "volume_free_human": human_bytes(master_info.volume_free_bytes or 0),
            "photos_app_counts": {
                "photos": args.master_photos_count,
                "videos": args.master_videos_count,
            },
        },
        "secondary": {
            "root": str(secondary_info.root),
            "originals_dirs": [str(p) for p in secondary_info.originals_dirs],
            "total_files": secondary_info.total_files,
            "total_bytes": secondary_info.total_bytes,
            "total_size_human": human_bytes(secondary_info.total_bytes),
            "image_files": secondary_info.image_files,
            "video_files": secondary_info.video_files,
            "other_media_files": secondary_info.other_media_files,
            "ext_counts": secondary_info.ext_counts,
            "sample_file": str(secondary_info.sample_file) if secondary_info.sample_file else None,
            "volume_root": str(secondary_info.volume_root) if secondary_info.volume_root else None,
            "volume_free_bytes": secondary_info.volume_free_bytes,
            "volume_free_human": human_bytes(secondary_info.volume_free_bytes or 0),
            "photos_app_counts": {
                "photos": args.secondary_photos_count,
                "videos": args.secondary_videos_count,
            },
        },
        "phase1_estimates": {
            "estimated_master_index_size_bytes": est_index,
            "estimated_master_index_size_human": human_bytes(est_index),
            "note": "Rough estimate only; actual index size depends on schema and stored fields.",
        },
    }

    out_json = reports_dir / "discovery_report.json"
    write_json(out_json, report)

    write_ext_csv(reports_dir / "extensions_master.csv", master_info.ext_counts)
    write_ext_csv(reports_dir / "extensions_secondary.csv", secondary_info.ext_counts)

    # Human-friendly summary with sanity check if counts provided
    def sanity_block(label: str, discovered_photos: int, discovered_videos: int, photos_app: Optional[int], videos_app: Optional[int]) -> str:
        lines = []
        if photos_app is not None:
            lines.append(f"- Photos.app photos: {photos_app:,} | Discovered image files: {discovered_photos:,} | Delta: {(discovered_photos - photos_app):+,}")
        if videos_app is not None:
            lines.append(f"- Photos.app videos: {videos_app:,} | Discovered video files: {discovered_videos:,} | Delta: {(discovered_videos - videos_app):+,}")
        return "\n".join(lines) if lines else "- (No Photos.app counts provided)"

    summary = []
    summary.append("PHOTOLIB DIFF — PHASE 0 DISCOVERY SUMMARY")
    summary.append("")
    summary.append("MASTER")
    summary.append(f"- Path: {master_info.root}")
    summary.append(f"- Originals dirs: " + ", ".join(str(p) for p in master_info.originals_dirs))
    summary.append(f"- Files: {master_info.total_files:,}")
    summary.append(f"- Size:  {human_bytes(master_info.total_bytes)}")
    summary.append(f"- Images: {master_info.image_files:,} | Videos: {master_info.video_files:,} | Other(media): {master_info.other_media_files:,}")
    summary.append(f"- Volume root: {master_info.volume_root}")
    summary.append(f"- Free space: {human_bytes(master_info.volume_free_bytes or 0)}")
    summary.append(f"- Sample read OK: {master_info.sample_file}")
    summary.append("Sanity check:")
    summary.append(sanity_block("MASTER", master_info.image_files, master_info.video_files, args.master_photos_count, args.master_videos_count))
    summary.append("")
    summary.append("SECONDARY")
    summary.append(f"- Path: {secondary_info.root}")
    summary.append(f"- Originals dirs: " + ", ".join(str(p) for p in secondary_info.originals_dirs))
    summary.append(f"- Files: {secondary_info.total_files:,}")
    summary.append(f"- Size:  {human_bytes(secondary_info.total_bytes)}")
    summary.append(f"- Images: {secondary_info.image_files:,} | Videos: {secondary_info.video_files:,} | Other(media): {secondary_info.other_media_files:,}")
    summary.append(f"- Volume root: {secondary_info.volume_root}")
    summary.append(f"- Free space: {human_bytes(secondary_info.volume_free_bytes or 0)}")
    summary.append(f"- Sample read OK: {secondary_info.sample_file}")
    summary.append("Sanity check:")
    summary.append(sanity_block("SECONDARY", secondary_info.image_files, secondary_info.video_files, args.secondary_photos_count, args.secondary_videos_count))
    summary.append("")
    summary.append("PHASE 1 ESTIMATE")
    summary.append(f"- Estimated Master index size: {human_bytes(est_index)} (stored in {index_dir})")
    summary.append("")

    out_txt = reports_dir / "discovery_summary.txt"
    write_text(out_txt, "\n".join(summary))

    print("\n=== Discovery Summary ===")
    print(f"Master files:    {master_info.total_files:,} | {human_bytes(master_info.total_bytes)}")
    print(f"  Images: {master_info.image_files:,} | Videos: {master_info.video_files:,}")
    print(f"Secondary files: {secondary_info.total_files:,} | {human_bytes(secondary_info.total_bytes)}")
    print(f"  Images: {secondary_info.image_files:,} | Videos: {secondary_info.video_files:,}")
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_txt}")
    print(f"Wrote: {reports_dir / 'extensions_master.csv'}")
    print(f"Wrote: {reports_dir / 'extensions_secondary.csv'}")
    print("\nNext: Phase 1 will build an index in:", index_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
