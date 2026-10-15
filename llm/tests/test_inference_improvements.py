import pandas as pd

from core.retriever import GrammarFeatureExtractor
from inference.run import build_high_confidence_nikl_map, normalize_nikl_gloss


def test_fixed_greeting_is_not_treated_as_command():
    features, _ = GrammarFeatureExtractor.extract("안녕하세요 만나서 반갑습니다")
    assert "명령청유" not in features


def test_real_command_after_greeting_is_still_detected():
    features, _ = GrammarFeatureExtractor.extract("안녕하세요. 여기 앉으세요")
    assert "명령청유" in features


def test_high_confidence_human_mapping_only():
    frame = pd.DataFrame(
        {
            "gloss": ["안녕하세요1 만나다1"] * 19
            + ["만나다3"]
            + ["합성어1"],
            "label_type": ["human_annotation"] * 20
            + ["generated_paraphrase"],
        }
    )
    mapping, evidence = build_high_confidence_nikl_map(frame)
    assert mapping["안녕하세요"] == "안녕하세요1"
    assert mapping["만나다"] == "만나다1"
    assert "합성어" not in mapping

    normalized, changes = normalize_nikl_gloss(
        "안녕하세요 만나다 반갑다", mapping, evidence
    )
    assert normalized == "안녕하세요1 만나다1 반갑다"
    assert [change["raw"] for change in changes] == ["안녕하세요", "만나다"]
