"""v2의 고정 split을 유지하며 gloss 표기 체계별 학습 CSV를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CONVENTIONS = ("nikl_numbered", "gksl_unnumbered")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=Path,
        default=Path("training_data/ksl_unified_raw_human_v2"),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("training_data/ksl_unified_raw_human_v2_by_convention"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "source_files_modified": False,
        "conventions": {},
    }
    split_groups = {convention: {} for convention in CONVENTIONS}

    for split in ("train", "valid", "test"):
        source = args.source_dir / f"{split}.csv"
        frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for convention in CONVENTIONS:
            selected = frame[frame["gloss_convention"] == convention].copy()
            target_dir = args.output_dir / convention
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{split}.csv"
            selected.to_csv(target, index=False, encoding="utf-8-sig")
            split_groups[convention][split] = set(selected["split_group"])
            manifest["conventions"].setdefault(convention, {})[split] = {
                "rows": len(selected),
                "sha256": digest(target),
                "corpora": selected["source_corpus"].value_counts().to_dict(),
            }

    for convention in CONVENTIONS:
        groups = split_groups[convention]
        intersections = {
            "train_valid": len(groups["train"] & groups["valid"]),
            "train_test": len(groups["train"] & groups["test"]),
            "valid_test": len(groups["valid"] & groups["test"]),
        }
        manifest["conventions"][convention]["split_group_intersections"] = intersections
        if any(intersections.values()):
            raise RuntimeError(f"{convention} split 누수: {intersections}")

    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
