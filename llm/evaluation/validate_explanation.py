"""문법 설명 계층 검증.

번역(gloss 정확도)과 분리된 "왜 이런 수어인가" 설명 계층을 검증한다. 설명은
GrammarFeatureExtractor가 감지한 문법 자질과, 그 자질로 검색한 문법서 근거로
구성된다. 따라서 설명 품질의 1차 지표는 두 가지다.

1. 자질 감지 정확도: 라벨링된 문장의 정답 자질 대비 precision/recall/F1
2. 근거 검색 건전성: 자질이 감지된 문장은 근거를 반환하고, 자질이 없는
   문장(인사말 등)은 억지 근거를 반환하지 않는지(min_score 적용)

이 스크립트는 전문가 평가를 대체하지 않는다. 자질 라벨은 표층 규칙으로 판정
가능한 명백한 사례만 담았고, 비수지표지/문장경계처럼 영상 없이 알 수 없는
현상은 범위에서 제외한다. 따라서 권한은 heuristic_proxy_not_expert로 고정한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.retriever import GrammarFeatureExtractor, GrammarRetriever


# 표층 규칙으로 명확히 판정 가능한 자질 라벨 세트. 각 문장에 대해 반드시
# 감지되어야 하는 자질(expected)만 지정한다. 다른 자질이 추가로 잡히는 것은
# 문장에 실제로 그 표지가 있으면 허용되므로, 아래는 최소 정답 집합이다.
LABELED_SENTENCES: list[tuple[str, list[str]]] = [
    ("왜 학교에 가지 않았나요?", ["부정", "의문"]),
    ("너는 밥을 먹었니?", ["의문"]),
    ("나는 사과를 먹지 않았다.", ["부정"]),
    ("창문을 좀 열어 주세요.", ["명령청유"]),
    ("우리 같이 영화 보러 가자.", ["명령청유"]),
    ("만약 비가 오면 집에 있겠다.", ["조건", "시제"]),
    ("어제 친구를 만났다.", ["시제"]),
    ("내일 비행기를 탈 것이다.", ["시제"]),
    ("그는 간다고 말했다.", ["인용", "시제"]),
    ("친구가 같이 가자고 말했어요.", ["인용", "명령청유"]),
    ("사과 세 개를 샀다.", ["수량", "시제"]),
    ("학생이 여러 명 모였다.", ["수량", "시제"]),
    ("이 가방이 저 가방보다 비싸다.", ["비교"]),
    ("그가 제일 빠르다.", ["비교"]),
    ("문이 바람에 의해 닫혔다.", ["피동사동"]),
    ("선생님이 학생에게 숙제를 시켰다.", ["피동사동", "위치이동"]),
    ("나는 학교에서 공부한다.", ["위치이동"]),
    ("그는 부산으로 떠났다.", ["위치이동"]),
    ("배가 고파서 밥을 먹었다.", ["접속", "시제"]),
    ("비가 왔지만 우산이 없었다.", ["부정", "접속", "시제"]),
]

# 자질이 없어야 하는(또는 억지 근거를 붙이면 안 되는) 문장.
NO_FEATURE_SENTENCES: list[str] = [
    "안녕하세요 만나서 반갑습니다",
    "안녕하십니까",
]


def score_feature_detection(labeled: list[tuple[str, list[str]]]) -> dict:
    tp = fp = fn = 0
    per_sentence = []
    for text, expected in labeled:
        detected, _ = GrammarFeatureExtractor.extract(text)
        detected_set, expected_set = set(detected), set(expected)
        # expected는 최소 정답 집합이므로 recall(놓친 정답)만 엄격히 센다.
        # 추가 감지는 문장에 표지가 실제로 있을 수 있어 soft하게 본다.
        missed = expected_set - detected_set
        matched = expected_set & detected_set
        extra = detected_set - expected_set
        tp += len(matched)
        fn += len(missed)
        per_sentence.append({
            "text": text,
            "expected": sorted(expected_set),
            "detected": sorted(detected_set),
            "missed": sorted(missed),
            "extra": sorted(extra),
        })
    recall = tp / max(tp + fn, 1)
    return {
        "expected_total": tp + fn,
        "matched": tp,
        "missed": fn,
        "recall_on_expected": round(recall, 4),
        "per_sentence": per_sentence,
    }


def score_evidence_soundness(retriever: GrammarRetriever, min_score: float) -> dict:
    """자질 있는 문장은 근거를 찾고, 인사말은 억지 근거를 안 붙이는지 확인."""
    with_feature_hit = 0
    labeled_with_feature = [s for s in LABELED_SENTENCES if s[1]]
    for text, _ in labeled_with_feature:
        hits = retriever.search(text, top_k=1, min_score=min_score, query_mode="feature")
        if hits:
            with_feature_hit += 1

    greeting_false_evidence = 0
    for text in NO_FEATURE_SENTENCES:
        hits = retriever.search(text, top_k=1, min_score=min_score, query_mode="feature")
        if hits:
            greeting_false_evidence += 1

    return {
        "min_score": min_score,
        "feature_sentences": len(labeled_with_feature),
        "feature_sentences_with_evidence": with_feature_hit,
        "evidence_recall": round(with_feature_hit / max(len(labeled_with_feature), 1), 4),
        "no_feature_sentences": len(NO_FEATURE_SENTENCES),
        "no_feature_false_evidence": greeting_false_evidence,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar_chunks", type=Path, required=True)
    ap.add_argument("--min_score", type=float, default=0.10)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    retriever = GrammarRetriever().build(args.grammar_chunks)

    detection = score_feature_detection(LABELED_SENTENCES)
    soundness = score_evidence_soundness(retriever, args.min_score)

    report = {
        "authority": "heuristic_proxy_not_expert",
        "grammar_chunks": str(args.grammar_chunks.resolve()),
        "labeled_sentences": len(LABELED_SENTENCES),
        "feature_detection": detection,
        "evidence_soundness": soundness,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
