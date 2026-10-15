"""고정 test 예측을 동일 입력의 복수 human gloss 기준으로 평가한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from core.metrics import _levenshtein
from data_processing.prepare_ksl_text_data import normalize_group_text


def token_f1_one(prediction: str, reference: str) -> float:
    predicted = Counter(prediction.split())
    expected = Counter(reference.split())
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(predicted.values()), 1)
    recall = overlap / max(sum(expected.values()), 1)
    return 2 * precision * recall / (precision + recall)


def best_reference(prediction: str, references: list[str]) -> tuple[str, int, float]:
    ranked = []
    for reference in references:
        distance = _levenshtein(prediction.split(), reference.split())
        f1 = token_f1_one(prediction, reference)
        ranked.append((distance, -f1, abs(len(prediction.split()) - len(reference.split())), reference))
    distance, negative_f1, _, reference = min(ranked)
    return reference, distance, -negative_f1


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {
            "samples": 0,
            "gloss_wer": 0.0,
            "exact_match": 0.0,
            "token_f1": 0.0,
        }
    total_errors = sum(record["distance"] for record in records)
    total_reference_tokens = sum(
        len(record["selected_reference"].split()) for record in records
    )
    return {
        "samples": len(records),
        "gloss_wer": round(total_errors / max(total_reference_tokens, 1), 4),
        "exact_match": round(
            sum(record["prediction"].split() == record["selected_reference"].split() for record in records)
            / len(records),
            4,
        ),
        "token_f1": round(
            sum(record["token_f1"] for record in records) / len(records), 4
        ),
        "pred_len": round(
            sum(len(record["prediction"].split()) for record in records) / len(records),
            4,
        ),
        "ref_len": round(total_reference_tokens / len(records), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions, sep="\t", dtype=str, keep_default_na=False
    )
    test = pd.read_csv(
        args.test_file, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    text_column = "text" if "text" in predictions else "korean_text"
    prediction_column = "pred" if "pred" in predictions else "prediction"
    test["evaluation_group"] = test["korean_text"].map(normalize_group_text)
    predictions["evaluation_group"] = predictions[text_column].map(normalize_group_text)

    reference_map = (
        test.groupby("evaluation_group", sort=False)["gloss_sequence"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    corpus_map = (
        test.groupby("evaluation_group", sort=False)["source_corpus"].first().to_dict()
    )

    records = []
    for group, group_predictions in predictions.groupby("evaluation_group", sort=False):
        references = reference_map.get(group)
        if not references:
            raise RuntimeError(f"test reference가 없는 예측 그룹: {group}")
        unique_predictions = group_predictions[prediction_column].value_counts()
        prediction = unique_predictions.index[0]
        selected, distance, f1 = best_reference(prediction, references)
        records.append(
            {
                "evaluation_group": group,
                "source_corpus": corpus_map[group],
                "text": group_predictions.iloc[0][text_column],
                "prediction": prediction,
                "selected_reference": selected,
                "references_json": json.dumps(references, ensure_ascii=False),
                "reference_count": len(references),
                "distance": distance,
                "token_f1": f1,
                "prediction_variants_in_file": len(unique_predictions),
            }
        )

    overall = aggregate(records)
    by_corpus = {
        corpus: aggregate([record for record in records if record["source_corpus"] == corpus])
        for corpus in sorted({record["source_corpus"] for record in records})
    }
    report = {
        "predictions": str(args.predictions.resolve()),
        "test_file": str(args.test_file.resolve()),
        "unique_input_evaluation": True,
        "multi_reference_groups": sum(record["reference_count"] > 1 for record in records),
        "overall": overall,
        "by_corpus": by_corpus,
    }
    output = args.output or args.predictions.with_name("test_metrics_multiref.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(
        output.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
