"""복수 human gloss를 보존하며 train 그룹 과대표집만 줄인 v4를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation.analyze_conflicts import repeated_sequence
from evaluation.auto_review_conflicts import select_medoid, token_distance


CONVENTIONS = ("nikl_numbered", "gksl_unnumbered")
NIKL_2024 = "NIKL_Korean_Sign_Language_Annotated_Corpus_2024"
CORE_COLUMNS = (
    "file_id", "korean_text", "gloss_sequence", "source_corpus",
    "gloss_convention", "source_file", "source_record_id", "label_type",
    "split_group",
)


def core_digest(frame: pd.DataFrame) -> str:
    value = frame[list(CORE_COLUMNS)].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def repeated_variants(glosses: list[str]) -> list[str]:
    sequences = {gloss: gloss.split() for gloss in glosses}
    excluded = []
    for gloss, sequence in sequences.items():
        if any(
            len(shorter) < len(sequence) and repeated_sequence(shorter, sequence)
            for other, shorter in sequences.items()
            if other != gloss
        ):
            excluded.append(gloss)
    return sorted(excluded)


def diverse_candidates(glosses: list[str], limit: int) -> list[str]:
    values = sorted(set(glosses))
    if len(values) <= limit:
        return values
    medoid, _, _ = select_medoid(values, Counter(values))
    selected = [medoid]
    remaining = set(values) - {medoid}
    while remaining and len(selected) < limit:
        candidate = max(
            remaining,
            key=lambda value: (
                min(token_distance(value, chosen) for chosen in selected),
                value,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def deterministic_row(frame: pd.DataFrame) -> pd.Series:
    columns = [
        column for column in
        ("source_corpus", "source_file", "source_record_id", "file_id")
        if column in frame.columns
    ]
    return frame.sort_values(columns, kind="stable").iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=Path,
        default=Path("training_data/ksl_unified_raw_human_v2"),
    )
    parser.add_argument(
        "--conflict_review", type=Path,
        default=Path(
            "training_data/ksl_unified_raw_human_v2/conflict_review_auto.csv"
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("training_data/ksl_unified_group_balanced_v4"),
    )
    parser.add_argument("--max_refs_per_train_group", type=int, default=3)
    args = parser.parse_args()
    if args.max_refs_per_train_group < 1:
        raise ValueError("max_refs_per_train_group은 1 이상이어야 합니다.")

    review = pd.read_csv(
        args.conflict_review, dtype=str, keep_default_na=False,
        encoding="utf-8-sig",
    )
    accepted_map = {}
    assisted_records = []
    for _, row in review.iterrows():
        glosses = json.loads(row["glosses"])
        excluded = repeated_variants(glosses)
        accepted = [gloss for gloss in glosses if gloss not in excluded]
        cross_convention = row["cross_convention"].lower() == "true"
        if cross_convention:
            resolution = "separate_by_gloss_convention"
        elif len(accepted) > 1:
            resolution = "provisional_multi_reference"
        else:
            resolution = "remove_repeated_annotation"
        accepted_map[row["split_group"]] = accepted
        enriched = row.to_dict()
        enriched.update(
            {
                "ai_assisted_resolution": resolution,
                "accepted_glosses_json": json.dumps(accepted, ensure_ascii=False),
                "excluded_glosses_json": json.dumps(excluded, ensure_ascii=False),
                "ai_review_basis": (
                    "human_annotation 복수 표현 보존; 완전 반복 target만 제외; "
                    "표기 체계가 다르면 별도 그룹으로 처리"
                ),
                "review_authority": "ai_assisted_not_expert",
                "expert_confirmation_recommended": True,
            }
        )
        assisted_records.append(enriched)

    assisted = pd.DataFrame(assisted_records)
    resolution_map = dict(
        zip(assisted["split_group"], assisted["ai_assisted_resolution"])
    )
    review_output = args.conflict_review.with_name(
        "conflict_review_ai_assisted.csv"
    )
    assisted.to_csv(review_output, index=False, encoding="utf-8-sig")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    manifest = {
        "version": 4,
        "strategy": "preserve_multi_reference_cap_train_groups",
        "max_refs_per_train_group": args.max_refs_per_train_group,
        "source_dir": str(args.source_dir.resolve()),
        "source_files_modified": False,
        "ai_review_authority": "not_expert",
        "conflict_groups": len(assisted),
        "provisional_multi_reference_groups": int(
            assisted["ai_assisted_resolution"].eq(
                "provisional_multi_reference"
            ).sum()
        ),
        "cross_convention_groups": int(
            assisted["ai_assisted_resolution"].eq(
                "separate_by_gloss_convention"
            ).sum()
        ),
        "excluded_repeated_variants": int(
            assisted["excluded_glosses_json"].map(
                lambda value: len(json.loads(value))
            ).sum()
        ),
        "splits": {},
        "subsets": {},
    }

    for split in ("train", "valid", "test"):
        source = pd.read_csv(
            args.source_dir / f"{split}.csv", dtype=str,
            keep_default_na=False, encoding="utf-8-sig",
        )
        source_core = core_digest(source)
        if split == "train":
            selected_rows = []
            capped_groups = 0
            for (group_key, convention), group in source.groupby(
                ["split_group", "gloss_convention"], sort=False
            ):
                accepted = accepted_map.get(group_key)
                if accepted is None:
                    selected_rows.extend(row for _, row in group.iterrows())
                    continue
                eligible = group[group["gloss_sequence"].isin(accepted)]
                selected_glosses = diverse_candidates(
                    eligible["gloss_sequence"].tolist(),
                    args.max_refs_per_train_group,
                )
                capped_groups += len(eligible["gloss_sequence"].unique()) > len(selected_glosses)
                for gloss in selected_glosses:
                    selected_rows.append(
                        deterministic_row(
                            eligible[eligible["gloss_sequence"] == gloss]
                        )
                    )
            frame = pd.DataFrame(selected_rows).reset_index(drop=True)
        else:
            frame = source.copy()
            capped_groups = 0

        frame["ai_review_resolution"] = frame["split_group"].map(
            lambda group: resolution_map.get(group, "no_conflict")
        )
        target = args.output_dir / f"{split}.csv"
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        outputs[split] = frame
        manifest["splits"][split] = {
            "source_rows": len(source),
            "output_rows": len(frame),
            "training_rows_removed": len(source) - len(frame),
            "capped_groups": int(capped_groups),
            "heldout_core_unchanged": (
                split == "train" or source_core == core_digest(frame)
            ),
        }

    groups = {split: set(frame["split_group"]) for split, frame in outputs.items()}
    intersections = {
        "train_valid": len(groups["train"] & groups["valid"]),
        "train_test": len(groups["train"] & groups["test"]),
        "valid_test": len(groups["valid"] & groups["test"]),
    }
    manifest["split_group_intersections"] = intersections
    if any(intersections.values()):
        raise RuntimeError(f"v4 split 누수: {intersections}")

    subset_root = args.output_dir.parent / f"{args.output_dir.name}_subsets"
    for name, selector in {
        "nikl_numbered": lambda frame: frame["gloss_convention"].eq("nikl_numbered"),
        "gksl_unnumbered": lambda frame: frame["gloss_convention"].eq("gksl_unnumbered"),
        "nikl_2024": lambda frame: frame["source_corpus"].eq(NIKL_2024),
    }.items():
        subset_manifest = {}
        target_dir = subset_root / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for split, frame in outputs.items():
            selected = frame[selector(frame)].copy()
            selected.to_csv(
                target_dir / f"{split}.csv", index=False, encoding="utf-8-sig"
            )
            subset_manifest[split] = {
                "rows": len(selected),
                "corpora": selected["source_corpus"].value_counts().to_dict(),
            }
        manifest["subsets"][name] = subset_manifest
    manifest["subset_root"] = str(subset_root.resolve())

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (subset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
