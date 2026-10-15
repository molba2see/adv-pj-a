"""충돌 gloss 그룹에 보수적인 자동 추천과 신뢰도를 부여한다.

이 스크립트는 전문가 판정을 가장하지 않는다. 동일 입력에 달린 후보 중
토큰 편집거리의 합이 가장 작은 중심 표현(medoid)을 실험용 대표값으로
추천하고, 안전하게 자동 확정할 수 있는 반복 오류와 나머지를 구분한다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from core.metrics import _levenshtein
from evaluation.analyze_conflicts import repeated_sequence


def token_distance(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    return _levenshtein(left_tokens, right_tokens) / max(
        len(left_tokens), len(right_tokens), 1
    )


def select_medoid(glosses: list[str], frequencies: Counter) -> tuple[str, float, float]:
    """Return candidate, consensus, and margin over the second candidate."""
    if len(glosses) == 1:
        return glosses[0], 1.0, 1.0
    ranked = []
    for gloss in glosses:
        distances = [token_distance(gloss, other) for other in glosses if other != gloss]
        consensus = 1.0 - sum(distances) / max(len(distances), 1)
        ranked.append((consensus, frequencies[gloss], -len(gloss.split()), gloss))
    ranked.sort(reverse=True)
    best = ranked[0]
    second = ranked[1]
    return best[3], round(best[0], 6), round(best[0] - second[0], 6)


def repeated_error_candidate(glosses: list[str]) -> str | None:
    sequences = {gloss: gloss.split() for gloss in glosses}
    candidates = []
    for gloss, sequence in sequences.items():
        if all(
            other == gloss
            or repeated_sequence(sequence, other_sequence)
            or sequence == other_sequence
            for other, other_sequence in sequences.items()
        ):
            candidates.append(gloss)
    return min(candidates, key=lambda value: (len(value.split()), value)) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("training_data/ksl_unified_raw_human_v2"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    review = pd.read_csv(
        args.data_dir / "conflict_review.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    rows = pd.read_csv(
        args.data_dir / "all_text_gloss.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    records = []
    for _, record in review.iterrows():
        glosses = json.loads(record["glosses"])
        matched = rows[rows["split_group"] == record["split_group"]]
        frequencies = Counter(matched["gloss_sequence"])
        medoid, consensus, margin = select_medoid(glosses, frequencies)
        repeated_candidate = repeated_error_candidate(glosses)
        flags = set(record["flags"].split(";"))

        if repeated_candidate and "repeated_sequence" in flags:
            candidate = repeated_candidate
            decision = "safe_remove_repeated_sequence"
            confidence = "high"
            requires_human = False
            reason = "모든 다른 후보가 짧은 후보의 완전 반복으로 설명됨"
        else:
            candidate = medoid
            decision = "experimental_medoid_only"
            if consensus >= 0.72 and margin >= 0.04:
                confidence = "medium"
            else:
                confidence = "low"
            requires_human = True
            reason = (
                f"후보 간 평균 토큰 유사도={consensus:.3f}, "
                f"차순위와의 차이={margin:.3f}; 복수 정답 가능성 때문에 전문가 확인 필요"
            )

        enriched = record.to_dict()
        enriched.update(
            {
                "auto_candidate_gloss": candidate,
                "auto_decision": decision,
                "auto_confidence": confidence,
                "auto_consensus": consensus,
                "auto_margin": margin,
                "human_review_required": requires_human,
                "auto_reason": reason,
            }
        )
        records.append(enriched)

    output = args.output or args.data_dir / "conflict_review_auto.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "groups": len(result),
        "safe_auto_groups": int((~result["human_review_required"]).sum()),
        "human_review_groups": int(result["human_review_required"].sum()),
        "confidence_counts": result["auto_confidence"].value_counts().to_dict(),
        "decision_counts": result["auto_decision"].value_counts().to_dict(),
        "source_files_modified": False,
        "output": str(output.resolve()),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
