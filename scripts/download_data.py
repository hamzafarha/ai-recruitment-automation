"""CLI tool to list, verify, and download/stream recruitment datasets and ontologies."""

import argparse
import sys
from pathlib import Path

from recruitment_ai.data.loader import DatasetCatalog, DatasetLoader


def print_catalog() -> None:
    print("\n" + "=" * 90)
    print("AI RECRUITMENT AUTOMATION - DATASET CATALOG (14 DATA SOURCES)")
    print("=" * 90)
    format_str = "{:<24} {:<10} {:<15} {:<36}"
    print(format_str.format("Dataset ID", "Priority", "Approx. Size", "Main Use"))
    print("-" * 90)
    for meta in DatasetCatalog.list_datasets():
        print(
            format_str.format(
                meta.dataset_id, meta.priority, meta.approx_size, meta.main_use[:34]
            )
        )
    print("=" * 90 + "\n")


def print_status(loader: DatasetLoader) -> None:
    print("\n" + "=" * 95)
    print("AI RECRUITMENT AUTOMATION - DATA LAYER LOCAL STATUS")
    print("=" * 95)
    format_str = "{:<24} {:<10} {:<8} {:<10} {:<40}"
    print(
        format_str.format(
            "Dataset ID", "Priority", "Exists?", "Files", "Local Directory"
        )
    )
    print("-" * 95)
    for st in loader.check_status():
        exists_str = "YES" if st.exists else "NO"
        print(
            format_str.format(
                st.dataset_id,
                st.priority,
                exists_str,
                str(st.file_count),
                st.local_dir[:38],
            )
        )
    print("=" * 95 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset management and download utility for AI Recruitment Automation."
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all 14 datasets and metadata."
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Check local existence and file counts.",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, help="Download or prepare specific dataset by ID."
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download or prepare all core datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max sample count when streaming or downloading.",
    )

    args = parser.parse_args()
    loader = DatasetLoader(base_dir=Path("./data"))

    if args.list:
        print_catalog()
        return

    if args.status or (not args.dataset and not args.all):
        print_status(loader)
        if not args.status and not args.dataset and not args.all:
            print(
                "Tip: Use --list to view catalog, or --dataset <id> / --all to download or generate samples."
            )
        return

    if args.dataset:
        try:
            print(f"Preparing dataset '{args.dataset}'...")
            success = loader.download_dataset(args.dataset, sample_limit=args.limit)
            if success:
                print(f"Dataset '{args.dataset}' is ready.")
            else:
                print(
                    f"Dataset '{args.dataset}' manifest saved (remote download skipped or failed)."
                )
        except Exception as exc:
            print(f"Error preparing dataset '{args.dataset}': {exc}", file=sys.stderr)
            sys.exit(1)

    if args.all:
        print("Preparing all datasets in catalog...")
        for meta in DatasetCatalog.list_datasets():
            print(f"-> Processing '{meta.dataset_id}' ({meta.priority})...")
            loader.download_dataset(meta.dataset_id, sample_limit=args.limit)
        print("\nAll datasets processed. Checking updated status:")
        print_status(loader)


if __name__ == "__main__":
    main()
