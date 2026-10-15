"""동일 한국어 문장에 여러 gloss가 달린 그룹을 자동 분류한다.

자동 분류는 삭제 결정을 내리지 않는다. 사람이 우선 검토할 순서와 차이 유형을
제공하며, 원본 CSV와 conflicts.json은 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


GLOSS_SUFFIX_RE = re.compile(r"\d+(?:_\d+)?[A-Z]?(?:#.*)?$")


def normalized_token(token: str) -> str:
    token = token.lstrip("%")
    return GLOSS_SUFFIX_RE.sub("", token)


def repeated_sequence(shorter: list[str], longer: list[str]) -> bool:
    if not shorter or len(longer) <= len(shorter) or len(longer) % len(shorter):
        return False
    return shorter * (len(longer) // len(shorter)) == longer


def classify_glosses(glosses: list[str]) -> list[str]:
    sequences = [gloss.split() for gloss in glosses]
    normalized = [
        [value for token in seq if (value := normalized_token(token))]
        for seq in sequences
    ]
    flags = []

    if any(
        repeated_sequence(left, right) or repeated_sequence(right, left)
        for i, left in enumerate(sequences)
        for right in sequences[i + 1 :]
    ):
        flags.append("repeated_sequence")

    normalized_strings = {tuple(seq) for seq in normalized}
    normalized_bags = {tuple(sorted(Counter(seq).items())) for seq in normalized}
    if len(normalized_strings) == 1:
        flags.append("number_or_symbol_suffix_only")
    elif len(normalized_bags) == 1:
        flags.append("order_only")

    shortest = min(normalized, key=len)
    if all(not (Counter(shortest) - Counter(seq)) for seq in normalized) and any(
        len(seq) > len(shortest) for seq in normalized
    ):
        flags.append("additive_variant")

    if not flags:
        flags.append("lexical_or_semantic_variant")
    return flags


def priority(flags: list[str], variants: int, cross_convention: bool) -> str:
    if "lexical_or_semantic_variant" in flags or variants >= 4 or cross_convention:
        return "high"
    if "order_only" in flags or "additive_variant" in flags:
        return "medium"
    return "low"


def recommended_action(flags: list[str]) -> str:
    if "lexical_or_semantic_variant" in flags:
        return "문맥·원문 주석을 확인해 복수 정답 또는 주석 오류 판정"
    if "order_only" in flags:
        return "허용 어순인지 전문가 확인 후 다중 참조 평가 후보로 등록"
    if "additive_variant" in flags:
        return "생략 가능한 기능어·담화표지인지 확인"
    if "repeated_sequence" in flags:
        return "의도된 반복이 아니면 중복 주석 후보로 표시"
    return "표기 매핑표로 통합 가능한지 확인하되 원본 접미사는 보존"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path("training_data/ksl_unified_raw_human_v2"),
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    conflicts = json.loads((data_dir / "conflicts.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(
        data_dir / "all_text_gloss.csv", dtype=str, keep_default_na=False,
        encoding="utf-8-sig",
    )

    records = []
    for split_group, glosses in conflicts.items():
        matched = rows[rows["split_group"] == split_group]
        conventions = sorted(set(matched["gloss_convention"]))
        corpora = sorted(set(matched["source_corpus"]))
        flags = classify_glosses(glosses)
        cross_convention = len(conventions) > 1
        records.append(
            {
                "priority": priority(flags, len(glosses), cross_convention),
                "split_group": split_group,
                "example_texts": json.dumps(
                    sorted(set(matched["korean_text"]))[:5], ensure_ascii=False
                ),
                "variant_count": len(glosses),
                "flags": ";".join(flags),
                "cross_convention": cross_convention,
                "conventions": ";".join(conventions),
                "corpora": ";".join(corpora),
                "glosses": json.dumps(glosses, ensure_ascii=False),
                "recommended_action": recommended_action(flags),
                "review_status": "pending",
                "reviewer_note": "",
            }
        )

    report = pd.DataFrame(records)
    order = {"high": 0, "medium": 1, "low": 2}
    report["_order"] = report["priority"].map(order)
    report = report.sort_values(
        ["_order", "variant_count", "split_group"],
        ascending=[True, False, True],
    ).drop(columns="_order")
    csv_path = data_dir / "conflict_review.csv"
    report.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "source_conflict_groups": len(conflicts),
        "output_rows": len(report),
        "source_files_modified": False,
        "priority_counts": report["priority"].value_counts().to_dict(),
        "flag_counts": {
            flag: int(report["flags"].str.split(";").map(lambda values: flag in values).sum())
            for flag in sorted({flag for value in report["flags"] for flag in value.split(";")})
        },
        "cross_convention_groups": int(report["cross_convention"].sum()),
        "review_csv": str(csv_path.resolve()),
    }
    summary_path = data_dir / "conflict_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
