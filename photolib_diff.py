#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib

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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


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


def count_files(roots: Iterable[Path]) -> int:
    total = 0
    for _ in walk_files(roots):
        total += 1
    return total


def discover_library(
    name: str,
    root: Path,
    progress_every: int = 5000,
    manifest_path: Optional[Path] = None,
    precount: bool = False,
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

    total_hint: Optional[int] = None
    if precount:
        print(f"[{name}] Pre-counting files for progress/ETA...", flush=True)
        total_hint = count_files(originals_dirs)

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
                if total_hint:
                    pct = (total_files / max(total_hint, 1)) * 100.0
                    remaining = max(total_hint - total_files, 0)
                    eta = format_duration(remaining / max(rate, 1e-6))
                    msg = (
                        f"[{name}] {pct:5.1f}% | {total_files:,}/{total_hint:,} files "
                        f"({rate:,.0f} files/sec) | ETA {eta}"
                    )
                else:
                    msg = (
                        f"[{name}] Discovered {total_files:,} files, {human_bytes(total_bytes)} "
                        f"({rate:,.0f} files/sec)"
                    )
                print(msg, flush=True)
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


def read_manifest(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_outdir_not_inside_library(outdir: Path, master: Path, secondary: Path) -> Optional[str]:
    for name, pth in [("Master", master), ("Secondary", secondary)]:
        try:
            out_resolved = outdir.resolve()
            lib_resolved = pth.resolve()
        except FileNotFoundError:
            out_resolved = outdir
            lib_resolved = pth
        if out_resolved == lib_resolved or out_resolved in lib_resolved.parents:
            return (
                f"Outdir is inside {name} library bundle. Choose a different output folder.\n"
                f"  Outdir: {outdir}\n  {name}: {pth}"
            )
    return None


def ensure_dest_not_inside_library(dest: Path, master: Path, secondary: Path) -> Optional[str]:
    for name, pth in [("Master", master), ("Secondary", secondary)]:
        try:
            dest_resolved = dest.resolve()
            lib_resolved = pth.resolve()
        except FileNotFoundError:
            dest_resolved = dest
            lib_resolved = pth
        if dest_resolved == lib_resolved or dest_resolved in lib_resolved.parents:
            return (
                f"Destination is inside {name} library bundle. Choose a different folder.\n"
                f"  Dest: {dest}\n  {name}: {pth}"
            )
    return None


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
        from tkinter import simpledialog
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
            path = filedialog.askopenfilename(
                title=title,
                filetypes=[("Photos Library", "*.photoslibrary"), ("All Files", "*")],
            )
            if not path:
                p = pick_dir(title=title, must_exist=True)
            else:
                p = Path(path)
            if p is None:
                return None
            if p.is_dir() and not p.name.endswith(".photoslibrary"):
                candidates = sorted(p.glob("*.photoslibrary"))
                if len(candidates) == 1:
                    p = candidates[0]
                elif len(candidates) > 1:
                    choices = "\n".join(
                        [f"{i+1}. {c.name}" for i, c in enumerate(candidates)]
                    )
                    prompt = (
                        "Multiple Photos libraries found in the selected folder.\n\n"
                        f"{choices}\n\n"
                        "Enter the number of the library to use:"
                    )
                    resp = simpledialog.askstring("Choose Library", prompt)
                    if resp is None:
                        return None
                    try:
                        idx = int(resp) - 1
                        p = candidates[idx]
                    except (ValueError, IndexError):
                        messagebox.showerror(
                            "Invalid Selection",
                            "Please enter a valid number from the list.",
                        )
                        continue
            if p.name.endswith(".photoslibrary") and is_photos_library(p):
                return p
            msg = (
                "This folder does not look like a standard Photos library bundle.\n\n"
                f"Selected: {p}\n\n"
                "Tip: pick the top-level *.photoslibrary bundle.\n"
                "Continue anyway?"
            )
            if messagebox.askyesno("Confirm Library Selection", msg):
                return p

    if master is None:
        messagebox.showinfo(
            "Select Master Library",
            "Please select your MASTER Photos library.\n"
            "Tip: choose the top-level *.photoslibrary bundle.",
        )
        master = pick_photos_library(
            "Select MASTER Photos Library (*.photoslibrary) — your primary library"
        )
    if secondary is None:
        messagebox.showinfo(
            "Select Secondary Library",
            "Please select your SECONDARY Photos library.\n"
            "Tip: choose the top-level *.photoslibrary bundle.",
        )
        secondary = pick_photos_library(
            "Select SECONDARY Photos Library (*.photoslibrary) — the library to compare"
        )
    if outdir is None:
        messagebox.showinfo(
            "Select Output Folder",
            "Please select an OUTPUT folder.\n"
            "This must be a safe location and NOT inside a Photos library.",
        )
        outdir = pick_dir(
            "Select OUTPUT folder (safe location; not inside a Photos library)",
            must_exist=False,
        )

    if master and secondary and outdir:
        confirm_msg = (
            "Please confirm your selections:\n\n"
            f"MASTER library:    {master}\n"
            f"SECONDARY library: {secondary}\n"
            f"OUTPUT folder:     {outdir}\n\n"
            "Discovery will READ from the libraries and WRITE only into the output folder."
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
    p.add_argument(
        "--no-precount",
        action="store_true",
        help="Skip the pre-count pass (faster start, but no percent/ETA during discovery).",
    )

    # Optional counts from Photos.app for sanity checking
    p.add_argument("--master-photos-count", type=int, default=None)
    p.add_argument("--master-videos-count", type=int, default=None)
    p.add_argument("--secondary-photos-count", type=int, default=None)
    p.add_argument("--secondary-videos-count", type=int, default=None)

    p1 = sub.add_parser("phase1", help="Phase 1: index + compare using manifests.")
    p1.add_argument("--outdir", required=True, help="Output directory used in Phase 0.")
    p1.add_argument(
        "--hash",
        choices=["none", "sha256"],
        default="none",
        help="Optional hashing for stronger matching (slower).",
    )
    p1.add_argument(
        "--hash-threshold-bytes",
        type=int,
        default=0,
        help="Only hash files >= this size (bytes). Use 0 for all sizes.",
    )
    p1.add_argument("--progress-every", type=int, default=5000, help="Print progress every N files.")

    p2 = sub.add_parser("phase2", help="Phase 2: create copy plan and/or execute copy.")
    p2.add_argument("--outdir", required=True, help="Output directory used in Phase 0/1.")
    p2.add_argument("--dest", required=True, help="Destination folder for copied files.")
    p2.add_argument(
        "--mode",
        choices=["plan", "copy"],
        default="plan",
        help="Create a copy plan or execute it.",
    )
    p2.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files (copy mode only).",
    )
    p2.add_argument("--progress-every", type=int, default=5000, help="Print progress every N files.")

    args = parser.parse_args()

    if args.cmd == "phase1":
        outdir = Path(args.outdir)
        reports_dir = outdir / "reports"
        manifest_dir = outdir / "manifests"
        index_dir = outdir / "index"

        report_path = reports_dir / "discovery_report.json"
        if not report_path.exists():
            print(f"ERROR: Missing discovery report: {report_path}", file=sys.stderr)
            return 2

        report = json.loads(report_path.read_text(encoding="utf-8"))
        master_root = Path(report["paths"]["master"])
        secondary_root = Path(report["paths"]["secondary"])

        msg = ensure_outdir_not_inside_library(outdir, master_root, secondary_root)
        if msg:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2

        master_manifest = Path(report["paths"]["master_manifest"])
        secondary_manifest = Path(report["paths"]["secondary_manifest"])
        if not master_manifest.exists() or not secondary_manifest.exists():
            print("ERROR: Missing manifests. Re-run Phase 0 to generate them.", file=sys.stderr)
            return 2

        print("=== Phase 1: Index + Compare ===", flush=True)
        print(f"Outdir:    {outdir}", flush=True)
        print(f"Reports:   {reports_dir}", flush=True)
        print(f"Index dir: {index_dir}", flush=True)
        print(f"Hashing:   {args.hash}", flush=True)
        if args.hash != "none":
            print(f"Hash size threshold: {args.hash_threshold_bytes} bytes", flush=True)
        print("", flush=True)

        master_records = read_manifest(master_manifest)
        secondary_records = read_manifest(secondary_manifest)

        # Build master index by size for fast candidate lookup
        master_by_size: Dict[int, List[Dict[str, object]]] = {}
        for rec in master_records:
            master_by_size.setdefault(int(rec["size"]), []).append(rec)

        hash_cache: Dict[str, str] = {}

        def get_hash(p: Path, size: int) -> Optional[str]:
            if args.hash == "none":
                return None
            if size < args.hash_threshold_bytes:
                return None
            key = str(p)
            if key in hash_cache:
                return hash_cache[key]
            try:
                h = sha256_file(p)
            except Exception:
                return None
            hash_cache[key] = h
            return h

        matches_size_mtime = 0
        matches_hash = 0
        size_mismatch = 0
        missing_in_master = 0
        errors = 0

        missing_rows: List[List[str]] = []

        t0 = time.time()
        last_print = t0

        total_secondary = len(secondary_records)
        for i, rec in enumerate(secondary_records, start=1):
            spath = Path(str(rec["path"]))
            ssize = int(rec["size"])
            smtime = int(rec["mtime"])

            candidates = master_by_size.get(ssize, [])
            if not candidates:
                missing_in_master += 1
                missing_rows.append([str(spath), str(ssize), str(smtime), "missing_in_master"])
            else:
                # First try size+mtime match
                if any(int(c["mtime"]) == smtime for c in candidates):
                    matches_size_mtime += 1
                elif args.hash != "none":
                    sh = get_hash(spath, ssize)
                    if sh is None:
                        errors += 1
                    else:
                        matched = False
                        for c in candidates:
                            ch = get_hash(Path(str(c["path"])), int(c["size"]))
                            if ch and ch == sh:
                                matched = True
                                break
                        if matched:
                            matches_hash += 1
                        else:
                            size_mismatch += 1
                            missing_rows.append([str(spath), str(ssize), str(smtime), "size_or_hash_mismatch"])
                else:
                    size_mismatch += 1
                    missing_rows.append([str(spath), str(ssize), str(smtime), "size_mismatch"])

            if i % args.progress_every == 0:
                now = time.time()
                if now - last_print >= 1.0:
                    rate = i / max(now - t0, 1e-6)
                    pct = (i / max(total_secondary, 1)) * 100.0
                    remaining = max(total_secondary - i, 0)
                    eta = format_duration(remaining / max(rate, 1e-6))
                    print(
                        f"[Phase1] {pct:5.1f}% | Compared {i:,}/{total_secondary:,} files "
                        f"({rate:,.0f} files/sec) | ETA {eta}",
                        flush=True,
                    )
                    last_print = now

        # Write index and reports
        ensure_writable_dir(index_dir)
        master_index_path = index_dir / "master_index.jsonl"
        with open(master_index_path, "w", encoding="utf-8") as f:
            f.write("# JSONL master index: path/size/mtime\n")
            for rec in master_records:
                f.write(json.dumps(rec) + "\n")

        compare_report = {
            "generated_at_epoch": int(time.time()),
            "inputs": {
                "outdir": str(outdir),
                "master_manifest": str(master_manifest),
                "secondary_manifest": str(secondary_manifest),
            },
            "hashing": {
                "mode": args.hash,
                "threshold_bytes": args.hash_threshold_bytes,
            },
            "counts": {
                "secondary_total": len(secondary_records),
                "matches_size_mtime": matches_size_mtime,
                "matches_hash": matches_hash,
                "size_mismatch": size_mismatch,
                "missing_in_master": missing_in_master,
                "errors": errors,
            },
            "outputs": {
                "master_index": str(master_index_path),
            },
        }

        compare_json = reports_dir / "phase1_compare.json"
        write_json(compare_json, compare_report)

        missing_csv = reports_dir / "phase1_missing.csv"
        with open(missing_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["secondary_path", "size_bytes", "mtime_epoch", "reason"])
            w.writerows(missing_rows)

        summary_lines = [
            "PHOTOLIB DIFF — PHASE 1 SUMMARY",
            "",
            f"Secondary total files: {total_secondary:,}",
            f"Matches (size+mtime): {matches_size_mtime:,}",
            f"Matches (hash):       {matches_hash:,}",
            f"Mismatches:           {size_mismatch:,}",
            f"Missing in Master:    {missing_in_master:,}",
            f"Errors:               {errors:,}",
            "",
            f"Wrote: {compare_json}",
            f"Wrote: {missing_csv}",
            f"Wrote: {master_index_path}",
        ]
        out_txt = reports_dir / "phase1_summary.txt"
        write_text(out_txt, "\n".join(summary_lines))

        print("\n=== Phase 1 Summary ===")
        print(f"Secondary total: {total_secondary:,}")
        print(f"Matches size+mtime: {matches_size_mtime:,}")
        print(f"Matches hash:       {matches_hash:,}")
        print(f"Mismatches:         {size_mismatch:,}")
        print(f"Missing in Master:  {missing_in_master:,}")
        print(f"Errors:             {errors:,}")
        print(f"\nWrote: {compare_json}")
        print(f"Wrote: {missing_csv}")
        print(f"Wrote: {master_index_path}")
        print(f"Wrote: {out_txt}")
        return 0

    if args.cmd == "phase2":
        outdir = Path(args.outdir)
        dest = Path(args.dest)
        reports_dir = outdir / "reports"
        plan_dir = outdir / "plans"

        report_path = reports_dir / "discovery_report.json"
        if not report_path.exists():
            print(f"ERROR: Missing discovery report: {report_path}", file=sys.stderr)
            return 2

        report = json.loads(report_path.read_text(encoding="utf-8"))
        master_root = Path(report["paths"]["master"])
        secondary_root = Path(report["paths"]["secondary"])

        msg = ensure_outdir_not_inside_library(outdir, master_root, secondary_root)
        if msg:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2

        msg = ensure_dest_not_inside_library(dest, master_root, secondary_root)
        if msg:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2

        missing_csv = reports_dir / "phase1_missing.csv"
        if not missing_csv.exists():
            print(f"ERROR: Missing Phase 1 missing list: {missing_csv}", file=sys.stderr)
            return 2

        ensure_writable_dir(plan_dir)
        dest.mkdir(parents=True, exist_ok=True)

        plan_path = plan_dir / "phase2_copy_plan.csv"
        plan_rows: List[List[str]] = []
        with open(missing_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = Path(row["secondary_path"])
                rel = src.relative_to(secondary_root) if src.is_absolute() else src
                dst = dest / rel
                plan_rows.append([str(src), str(dst), row["size_bytes"], row["reason"]])

        with open(plan_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_path", "dest_path", "size_bytes", "reason"])
            w.writerows(plan_rows)

        if args.mode == "plan":
            print("=== Phase 2: Copy Plan ===")
            print(f"Wrote: {plan_path}")
            print(f"Rows:  {len(plan_rows):,}")
            return 0

        print("=== Phase 2: Copy ===")
        print(f"Plan:  {plan_path}")
        print(f"Dest:  {dest}")
        if args.dry_run:
            print("Mode:  DRY RUN (no files will be written)")

        t0 = time.time()
        last_print = t0
        total = len(plan_rows)
        copied = 0
        skipped = 0
        errors = 0

        for i, row in enumerate(plan_rows, start=1):
            src = Path(row[0])
            dst = Path(row[1])
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if args.dry_run:
                    skipped += 1
                else:
                    shutil.copy2(src, dst)
                    copied += 1
            except Exception:
                errors += 1

            if i % args.progress_every == 0:
                now = time.time()
                if now - last_print >= 1.0:
                    rate = i / max(now - t0, 1e-6)
                    pct = (i / max(total, 1)) * 100.0
                    remaining = max(total - i, 0)
                    eta = format_duration(remaining / max(rate, 1e-6))
                    print(
                        f"[Phase2] {pct:5.1f}% | {i:,}/{total:,} files "
                        f"({rate:,.0f} files/sec) | ETA {eta}",
                        flush=True,
                    )
                    last_print = now

        summary_lines = [
            "PHOTOLIB DIFF — PHASE 2 SUMMARY",
            "",
            f"Planned files: {total:,}",
            f"Copied:        {copied:,}",
            f"Skipped:       {skipped:,}",
            f"Errors:        {errors:,}",
            "",
            f"Plan: {plan_path}",
            f"Dest: {dest}",
        ]
        out_txt = reports_dir / "phase2_summary.txt"
        write_text(out_txt, "\n".join(summary_lines))

        print("\n=== Phase 2 Summary ===")
        print(f"Planned files: {total:,}")
        print(f"Copied:        {copied:,}")
        print(f"Skipped:       {skipped:,}")
        print(f"Errors:        {errors:,}")
        print(f"Wrote: {out_txt}")
        return 0

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
        if args.gui:
            print(
                "ERROR: Selection was canceled or incomplete. Please re-run and finish all selections.",
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: Missing required paths. Provide --master, --secondary, --outdir or use --gui.",
                file=sys.stderr,
            )
        return 2

    # Ensure we never write inside a Photos library bundle.
    msg = ensure_outdir_not_inside_library(outdir, master, secondary)
    if msg:
        print(f"ERROR: {msg}", file=sys.stderr)
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
            precount=not args.no_precount,
        )
        secondary_info = discover_library(
            "Secondary",
            secondary,
            progress_every=args.progress_every,
            manifest_path=secondary_manifest,
            precount=not args.no_precount,
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
