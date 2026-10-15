"""Paired bootstrap comparison for two Text2Gloss prediction files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.prepare_ksl_text_data import normalize_group_text
from evaluation.evaluate_multiref import best_reference


def _prediction_map(frame: pd.DataFrame) -> dict[str, str]:
    text_column = "text" if "text" in frame else "korean_text"
    prediction_column = "pred" if "pred" in frame else "prediction"
    frame = frame.copy()
    frame["evaluation_group"] = frame[text_column].map(normalize_group_text)
    return {
        group: Counter(rows[prediction_column]).most_common(1)[0][0]
        for group, rows in frame.groupby("evaluation_group", sort=False)
    }


def build_records(
    baseline: pd.DataFrame, candidate: pd.DataFrame, test: pd.DataFrame
) -> list[dict[str, object]]:
    test = test.copy()
    test["evaluation_group"] = test["korean_text"].map(normalize_group_text)
    references = (
        test.groupby("evaluation_group", sort=False)["gloss_sequence"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    texts = test.groupby("evaluation_group", sort=False)["korean_text"].first().to_dict()
    baseline_map = _prediction_map(baseline)
    candidate_map = _prediction_map(candidate)

    missing = sorted(set(references) - (set(baseline_map) & set(candidate_map)))
    if missing:
        raise ValueError(f"prediction missing for {len(missing)} evaluation groups")

    records: list[dict[str, object]] = []
    for group, group_references in references.items():
        base_prediction = baseline_map[group]
        candidate_prediction = candidate_map[group]
        base_reference, base_distance, base_f1 = best_reference(
            base_prediction, group_references
        )
        candidate_reference, candidate_distance, candidate_f1 = best_reference(
            candidate_prediction, group_references
        )
        records.append(
            {
                "evaluation_group": group,
                "text": texts[group],
                "references_json": json.dumps(group_references, ensure_ascii=False),
                "baseline_prediction": base_prediction,
                "candidate_prediction": candidate_prediction,
                "baseline_selected_reference": base_reference,
                "candidate_selected_reference": candidate_reference,
                "baseline_distance": base_distance,
                "candidate_distance": candidate_distance,
                "baseline_reference_tokens": len(base_reference.split()),
                "candidate_reference_tokens": len(candidate_reference.split()),
                "baseline_exact": int(base_prediction.split() == base_reference.split()),
                "candidate_exact": int(
                    candidate_prediction.split() == candidate_reference.split()
                ),
                "baseline_token_f1": base_f1,
                "candidate_token_f1": candidate_f1,
                "token_f1_delta": candidate_f1 - base_f1,
            }
        )
    return records


def _metrics(records: list[dict[str, object]], indices: np.ndarray) -> dict[str, float]:
    chosen = [records[int(index)] for index in indices]
    baseline_wer = sum(int(row["baseline_distance"]) for row in chosen) / max(
        sum(int(row["baseline_reference_tokens"]) for row in chosen), 1
    )
    candidate_wer = sum(int(row["candidate_distance"]) for row in chosen) / max(
        sum(int(row["candidate_reference_tokens"]) for row in chosen), 1
    )
    return {
        "wer_delta": candidate_wer - baseline_wer,
        "exact_delta": float(
            np.mean(
                [
                    int(row["candidate_exact"]) - int(row["baseline_exact"])
                    for row in chosen
                ]
            )
        ),
        "token_f1_delta": float(
            np.mean(
                [
                    float(row["candidate_token_f1"])
                    - float(row["baseline_token_f1"])
                    for row in chosen
                ]
            )
        ),
    }


def compare(
    records: list[dict[str, object]], *, iterations: int, seed: int
) -> dict[str, object]:
    indices = np.arange(len(records))
    observed = _metrics(records, indices)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in observed}
    for _ in range(iterations):
        sampled = rng.integers(0, len(records), size=len(records))
        metrics = _metrics(records, sampled)
        for key, value in metrics.items():
            samples[key].append(value)

    report: dict[str, object] = {
        "samples": len(records),
        "bootstrap_iterations": iterations,
        "seed": seed,
        "delta_definition": "candidate_minus_baseline",
        "metrics": {},
        "improved_groups_by_token_f1": sum(
            float(row["token_f1_delta"]) > 0 for row in records
        ),
        "regressed_groups_by_token_f1": sum(
            float(row["token_f1_delta"]) < 0 for row in records
        ),
        "tied_groups_by_token_f1": sum(
            float(row["token_f1_delta"]) == 0 for row in records
        ),
    }
    metric_report = report["metrics"]
    assert isinstance(metric_report, dict)
    for key, values in samples.items():
        array = np.asarray(values)
        lower, upper = np.quantile(array, [0.025, 0.975])
        lower_is_better = key == "wer_delta"
        improvement_probability = np.mean(array < 0 if lower_is_better else array > 0)
        metric_report[key] = {
            "observed": round(observed[key], 6),
            "bootstrap_95ci": [round(float(lower), 6), round(float(upper), 6)],
            "improvement_probability": round(float(improvement_probability), 6),
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        }
    report["primary_token_f1_improvement_supported"] = bool(
        metric_report["token_f1_delta"]["bootstrap_95ci"][0] > 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    read_tsv = lambda path: pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    baseline = read_tsv(args.baseline)
    candidate = read_tsv(args.candidate)
    test = pd.read_csv(args.test_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    records = build_records(baseline, candidate, test)
    report = compare(records, iterations=args.iterations, seed=args.seed)
    report.update(
        {
            "baseline": str(args.baseline.resolve()),
            "candidate": str(args.candidate.resolve()),
            "test_file": str(args.test_file.resolve()),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    record_frame = pd.DataFrame(records).sort_values("token_f1_delta")
    record_frame.to_csv(
        args.output.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )
    changed = record_frame[record_frame["token_f1_delta"] != 0].copy()
    changed["metric_outcome"] = np.where(
        changed["token_f1_delta"] < 0,
        "baseline_better_by_token_f1",
        "candidate_better_by_token_f1",
    )
    changed["review_priority"] = np.where(changed["token_f1_delta"] < 0, 1, 2)
    for column in (
        "expert_preferred_model",
        "expert_error_category",
        "reviewer",
        "review_note",
    ):
        changed[column] = ""
    changed.sort_values(["review_priority", "token_f1_delta"]).to_csv(
        args.output.with_suffix(".review.csv"), index=False, encoding="utf-8-sig"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
