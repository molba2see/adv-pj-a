from evaluation.compare_paired_predictions import compare


def _record(base_f1: float, candidate_f1: float) -> dict[str, object]:
    return {
        "baseline_distance": 2,
        "candidate_distance": 1,
        "baseline_reference_tokens": 4,
        "candidate_reference_tokens": 4,
        "baseline_exact": 0,
        "candidate_exact": 0,
        "baseline_token_f1": base_f1,
        "candidate_token_f1": candidate_f1,
        "token_f1_delta": candidate_f1 - base_f1,
    }


def test_compare_reports_paired_improvement_direction():
    report = compare(
        [_record(0.2, 0.5), _record(0.4, 0.6)], iterations=100, seed=7
    )

    assert report["metrics"]["wer_delta"]["observed"] < 0
    assert report["metrics"]["token_f1_delta"]["observed"] > 0
    assert report["primary_token_f1_improvement_supported"] is True
