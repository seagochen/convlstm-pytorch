"""
Data CSV Generator

Scans directory structure to find image sequence folders and generates/updates
data.csv with folder paths and their type (dynamic/static/negative).

The script automatically determines the folder type based on the parent folder name:
- If parent contains "dynamic" -> type is "dynamic"
- If parent contains "static" -> type is "static"
- If parent contains "negative" -> type is "negative"
- Otherwise -> asks user or skips

Output format:
    folder_name,type
    fire_dynamic/11016-228113782_small,dynamic
    fire_static/image_001,static
    negative_samples/person_001,negative
"""

import csv
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


# Default image extensions to search for
DEFAULT_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG", "BMP"]


def has_image_files(directory: Path, extensions: List[str]) -> bool:
    """
    Check if directory contains any image files.

    Args:
        directory: Directory to check
        extensions: List of valid image extensions

    Returns:
        True if directory contains at least one image file
    """
    for ext in extensions:
        if list(directory.glob(f"*.{ext}")):
            return True
    return False


def determine_type_from_path(folder_path: Path, root_path: Path) -> str:
    """
    Determine folder type based on parent folder name.

    Rules:
    - If any parent folder name contains "dynamic" -> "dynamic"
    - If any parent folder name contains "static" -> "static"
    - If any parent folder name contains "negative" -> "negative"
    - Otherwise -> "unknown"

    Args:
        folder_path: Path to the image folder
        root_path: Root directory being scanned

    Returns:
        "dynamic", "static", "negative", or "unknown"
    """
    # Get relative path from root
    try:
        relative_path = folder_path.relative_to(root_path)
    except ValueError:
        relative_path = folder_path

    # Check each part of the path
    for part in relative_path.parts:
        part_lower = part.lower()
        if "dynamic" in part_lower:
            return "dynamic"
        elif "static" in part_lower:
            return "static"
        elif "negative" in part_lower:
            return "negative"

    return "unknown"


def find_image_folders(
    root_dir: Path,
    extensions: List[str] = DEFAULT_EXTENSIONS,
    max_depth: int = 3,
    verbose: bool = True
) -> List[Tuple[str, str]]:
    """
    Recursively find all folders containing image files.

    Args:
        root_dir: Root directory to scan
        extensions: List of valid image extensions
        max_depth: Maximum depth to search (default: 3)
        verbose: Print progress messages

    Returns:
        List of tuples: [(relative_folder_path, type), ...]
        where type is "dynamic", "static", or "negative"
    """
    if verbose:
        print(f"Scanning directory: {root_dir}")
        print(f"Maximum depth: {max_depth}")
        print(f"Extensions: {', '.join(extensions)}")
        print()

    results = []
    unknown_folders = []

    def scan_directory(directory: Path, current_depth: int):
        """Recursively scan directory."""
        if current_depth > max_depth:
            return

        # Check if current directory has images
        if has_image_files(directory, extensions):
            # Get relative path from root
            try:
                relative_path = directory.relative_to(root_dir)
                folder_name = str(relative_path).replace("\\", "/")
            except ValueError:
                folder_name = directory.name

            # Determine type
            folder_type = determine_type_from_path(directory, root_dir)

            if folder_type != "unknown":
                results.append((folder_name, folder_type))
                if verbose:
                    print(f"  Found: {folder_name} -> {folder_type}")
            else:
                unknown_folders.append(folder_name)
                if verbose:
                    print(f"  Found: {folder_name} -> unknown (skipping)")

        # Recursively scan subdirectories
        try:
            for subdir in sorted(directory.iterdir()):
                if subdir.is_dir():
                    scan_directory(subdir, current_depth + 1)
        except PermissionError:
            if verbose:
                print(f"  Warning: Permission denied for {directory}")

    # Start scanning
    scan_directory(root_dir, 0)

    if verbose:
        print()
        print(f"Summary:")
        print(f"  Total folders found: {len(results)}")
        print(f"  Unknown type folders: {len(unknown_folders)}")
        if unknown_folders:
            print(f"  Unknown folders (skipped):")
            for folder in unknown_folders:
                print(f"    - {folder}")

    return results


def write_csv(
    output_path: Path,
    folder_list: List[Tuple[str, str]],
    sort_by_name: bool = True,
    verbose: bool = True
):
    """
    Write folder list to CSV file.

    Args:
        output_path: Output CSV file path
        folder_list: List of (folder_name, type) tuples
        sort_by_name: Sort by folder name before writing
        verbose: Print progress messages
    """
    if sort_by_name:
        folder_list = sorted(folder_list, key=lambda x: x[0])

    # Count by type
    type_counts = {}
    for _, folder_type in folder_list:
        type_counts[folder_type] = type_counts.get(folder_type, 0) + 1

    if verbose:
        print(f"\nWriting to: {output_path}")
        print(f"Total entries: {len(folder_list)}")
        for folder_type, count in sorted(type_counts.items()):
            print(f"  {folder_type}: {count}")

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['folder_name', 'type'])
        writer.writerows(folder_list)

    if verbose:
        print(f"\n✓ Successfully wrote {len(folder_list)} entries to {output_path}")


def update_csv(
    csv_path: Path,
    new_folders: List[Tuple[str, str]],
    overwrite_existing: bool = False,
    verbose: bool = True
):
    """
    Update existing CSV file with new folders.

    Args:
        csv_path: Path to existing CSV file
        new_folders: List of (folder_name, type) tuples to add
        overwrite_existing: If True, overwrite existing entries; if False, skip duplicates
        verbose: Print progress messages
    """
    # Read existing CSV
    existing_folders = {}
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    existing_folders[row[0]] = row[1]

        if verbose:
            print(f"Loaded {len(existing_folders)} existing entries from {csv_path}")
    else:
        if verbose:
            print(f"CSV file does not exist, will create new file")

    # Merge with new folders
    added_count = 0
    updated_count = 0
    skipped_count = 0

    for folder_name, folder_type in new_folders:
        if folder_name in existing_folders:
            if overwrite_existing:
                if existing_folders[folder_name] != folder_type:
                    existing_folders[folder_name] = folder_type
                    updated_count += 1
                    if verbose:
                        print(f"  Updated: {folder_name} -> {folder_type}")
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        else:
            existing_folders[folder_name] = folder_type
            added_count += 1
            if verbose:
                print(f"  Added: {folder_name} -> {folder_type}")

    # Write back to CSV
    folder_list = [(name, ftype) for name, ftype in existing_folders.items()]
    write_csv(csv_path, folder_list, sort_by_name=True, verbose=False)

    if verbose:
        print(f"\nUpdate summary:")
        print(f"  Added: {added_count}")
        print(f"  Updated: {updated_count}")
        print(f"  Skipped (duplicates): {skipped_count}")
        print(f"  Total entries: {len(existing_folders)}")


def main():
    parser = argparse.ArgumentParser(
        description="Data CSV Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  Scans directory structure to find image sequence folders and generates/updates
  data.csv with folder paths and their type (dynamic/static/negative).

  Type detection rules:
    - If parent folder contains "dynamic" -> type is "dynamic"
    - If parent folder contains "static" -> type is "static"
    - If parent folder contains "negative" -> type is "negative"
    - Otherwise -> skipped (unknown type)

  Output CSV format:
    folder_name,type
    fire_dynamic/11016-228113782_small,dynamic
    fire_static/image_001,static
    negative_samples/person_001,negative

Examples:
  # Generate new data.csv from directory
  python generate_data_csv.py -i /path/to/dataset -o data.csv

  # Update existing data.csv with new folders
  python generate_data_csv.py -i /path/to/dataset -o data.csv --update

  # Overwrite existing entries when updating
  python generate_data_csv.py -i /path/to/dataset -o data.csv --update --overwrite

  # Custom extensions and max depth
  python generate_data_csv.py -i /path/to/dataset -o data.csv -e png jpg --max-depth 4
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input root directory to scan"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output CSV file path (e.g., data.csv)"
    )
    parser.add_argument(
        "-e", "--ext",
        type=str,
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help=f"Image file extensions to search (default: {' '.join(DEFAULT_EXTENSIONS[:4])})"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to search (default: 3)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update mode: merge with existing CSV instead of overwriting"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When updating, overwrite existing entries (only with --update)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Expand and validate paths
    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1

    verbose = not args.quiet

    # Print configuration
    if verbose:
        print("=" * 70)
        print("Data CSV Generator")
        print("=" * 70)
        print(f"  Input directory:  {input_dir}")
        print(f"  Output CSV:       {output_path}")
        print(f"  Extensions:       {', '.join(args.ext)}")
        print(f"  Max depth:        {args.max_depth}")
        print(f"  Mode:             {'Update' if args.update else 'Create new'}")
        if args.update and args.overwrite:
            print(f"  Overwrite:        Yes")
        print("=" * 70)
        print()

    # Scan directory for image folders
    folder_list = find_image_folders(
        input_dir,
        extensions=args.ext,
        max_depth=args.max_depth,
        verbose=verbose
    )

    if len(folder_list) == 0:
        print("\nWarning: No image folders found!")
        return 0

    # Write or update CSV
    if args.update and output_path.exists():
        update_csv(output_path, folder_list, overwrite_existing=args.overwrite, verbose=verbose)
    else:
        write_csv(output_path, folder_list, sort_by_name=True, verbose=verbose)

    return 0


if __name__ == "__main__":
    exit(main())
