# Apple Photos Library Compare

Compare two Apple Photos libraries and copy unique files from one to another.

Useful for consolidating old backup libraries - finds files in a secondary library that don't exist in your main library, then copies them for easy import.

## Requirements

- Python 3.8+
- macOS (for Photos library access)
- Full Disk Access enabled for Terminal (System Settings > Privacy & Security > Full Disk Access)

## Usage

### GUI Mode (Recommended)

```bash
python3 photolib_compare.py
```

Dialogs will guide you to select:
1. **Main library** - your primary/reference library
2. **Secondary library** - the library to find unique files in
3. **Output folder** - where unique files will be copied

### Command Line Mode

```bash
python3 photolib_compare.py --main ~/Pictures/Main.photoslibrary \
                            --secondary ~/Pictures/Old.photoslibrary \
                            --output ~/Desktop/PhotoDiff
```

### Options

| Flag | Description |
|------|-------------|
| `--main` | Path to main Photos library |
| `--secondary` | Path to secondary Photos library |
| `--output` | Path to output folder |
| `--skip-copy` | Compare only, don't copy files |

## How It Works

1. **Discovery** - Scans both libraries and builds file manifests
2. **Comparison** - Compares files using size + SHA256 hash matching
3. **Copying** - Copies unique files to output folder, preserving directory structure

## Output Structure

```
output/
├── manifests/                    # Cached scan data
│   ├── main_library.jsonl
│   ├── secondary_library.jsonl
│   └── unique_files.jsonl
└── copied/                       # Unique files from secondary
    └── originals/
        └── [preserved structure]
```

## Resumability

The tool saves manifests after each phase. If interrupted, simply run again - it will skip completed phases and resume where it left off.

To force a fresh scan, delete the `manifests/` folder in your output directory.

## Importing Copied Files

After the tool completes, you can import the unique files into your main Photos library:

1. Open Photos app
2. File > Import
3. Select the `copied/` folder from your output directory
4. Review and import

## Troubleshooting

### "Permission denied" error

Enable Full Disk Access for Terminal:
1. Open System Settings > Privacy & Security > Full Disk Access
2. Enable access for Terminal (or iTerm, VS Code, etc.)
3. Restart your terminal and try again

### Library not detected

Ensure you're selecting a valid `.photoslibrary` bundle. The tool looks for `originals/` or `masters/` folders inside.
