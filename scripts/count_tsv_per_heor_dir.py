#!/usr/bin/env python3
"""Count .tsv files in each j_... subdirectory under evaluations/HEOR_Sonnet45."""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "evaluations" / "HEOR_Sonnet45"


def main() -> None:
    if not BASE.is_dir():
        print(f"Base directory not found: {BASE}")
        return

    # Find all j_* dirs (one per Set_X)
    results: list[tuple[str, int]] = []
    for set_dir in sorted(BASE.iterdir()):
        if not set_dir.is_dir() or set_dir.name.startswith("."):
            continue
        for j_dir in set_dir.iterdir():
            if not j_dir.is_dir() or not j_dir.name.startswith("j_"):
                continue
            count = sum(
                1 for f in j_dir.iterdir() if f.is_file() and f.suffix == ".tsv"
            )
            subpath = f"{set_dir.name}/{j_dir.name}"
            results.append((subpath, count))

    # Sort by Set number then by count
    def sort_key(item: tuple[str, int]) -> tuple[int, int]:
        subpath, count = item
        try:
            set_num = int(subpath.split("_")[1].split("-")[0])
        except (IndexError, ValueError):
            set_num = 999
        return (set_num, -count)

    results.sort(key=sort_key)

    total = 0
    for subpath, count in results:
        print(f"{count:5d}  {subpath}")
        total += count

    print("-" * 60)
    print(f"{total:5d}  (total TSV files in {len(results)} j_* directories)")


if __name__ == "__main__":
    main()
