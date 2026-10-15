import json

from core.retriever import GrammarRetriever


def test_feature_search_returns_no_evidence_without_detected_feature(tmp_path):
    chunk = {
        "chunk_id": "g-test",
        "page": 1,
        "chapter": "문장",
        "section": "청유문",
        "heading": "청유문",
        "text": "청유문은 함께 행동하도록 요청하는 문장이다.",
    }
    path = tmp_path / "grammar.jsonl"
    path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
    retriever = GrammarRetriever().build(path)

    assert retriever.search(
        "안녕하세요 만나서 반갑습니다",
        top_k=1,
        min_score=0.0,
        query_mode="feature",
    ) == []
