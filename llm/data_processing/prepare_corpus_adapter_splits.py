"""Create reproducible per-corpus train/validation/test CSVs for adapter training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        src = args.input_dir / f"{split}.csv"
        dst = args.output_dir / f"{split}.csv"
        with src.open("r", encoding="utf-8-sig", newline="") as in_file:
            reader = csv.DictReader(in_file)
            fieldnames = reader.fieldnames
            if not fieldnames or "source_corpus" not in fieldnames:
                raise ValueError(f"source_corpus column is missing: {src}")
            rows = [row for row in reader if row["source_corpus"] == args.corpus]
        with dst.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = len(rows)

    print(f"[corpus] {args.corpus}")
    print(" ".join(f"{split}={count}" for split, count in counts.items()))


if __name__ == "__main__":
    main()
