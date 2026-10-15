import json
from pathlib import Path

import pytest

from core.artifacts import find_run_config, resolve_base_model, resolve_resource


def test_checkpoint_uses_parent_configuration(tmp_path):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    assert find_run_config(checkpoint) == tmp_path / "run_config.json"


def test_old_pko_server_path_falls_back_to_original_hub_model():
    assert resolve_base_model(
        "/home/work/adv_project/adv-pj-a/llm2/base_model/paust_pko_t5_base"
    ) == "paust/pko-t5-base"


def test_missing_unknown_base_does_not_silently_change_architecture():
    with pytest.raises(FileNotFoundError):
        resolve_base_model("/missing/different-model")
    assert resolve_base_model("/missing/old", "my/model") == "my/model"


def test_local_resources_rebase_after_relocation(tmp_path, monkeypatch):
    from core import artifacts

    monkeypatch.setattr(artifacts, "PROJECT_ROOT", tmp_path)
    train = tmp_path / "training_data/version/train.csv"
    train.parent.mkdir(parents=True)
    train.touch()
    assert resolve_resource("/old/server/training_data/version/train.csv") == train


def test_raw_no_rag_inference_can_run_without_corpus(tmp_path, monkeypatch, capsys):
    from inference import run

    cfg = {
        "model_name": "paust/pko-t5-base",
        "data": {"input_tag": "gloss_convention", "train_file": "/absent/train.csv"},
        "rag": {"mode": "none", "task_prefix": "번역: "},
    }
    monkeypatch.setattr(run, "load_model", lambda *a, **kw: (None, None, cfg))
    captured = {}
    def fake_generate(model, tokenizer, sources, *args, **kwargs):
        captured["sources"] = sources
        return ["학교1 가다1"]
    monkeypatch.setattr(run, "generate", fake_generate)
    monkeypatch.setattr("sys.argv", [
        "run", "--model_dir", str(tmp_path), "--no-explain",
        "--no-normalize_convention", "--text", "학교에 갑니다.",
    ])
    run.main()
    assert "학교1 가다1" in capsys.readouterr().out
    assert captured["sources"] == [
        "[gloss_convention=nikl_numbered] 번역: 학교에 갑니다."
    ]


def test_no_rag_explanation_is_separate_and_does_not_require_train(tmp_path, monkeypatch, capsys):
    from inference import run

    grammar = tmp_path / "grammar.jsonl"
    grammar.write_text(json.dumps({
        "chunk_id": "synthetic-negation", "heading": "부정법", "page": 7,
        "text": "부정문 부정 표현 부정법 문법",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {
        "model_name": "paust/pko-t5-base",
        "data": {"input_tag": "gloss_convention", "explicit_splits": True},
        "rag": {"mode": "none", "min_score": 0.0},
    }
    monkeypatch.setattr(run, "load_model", lambda *a, **kw: (None, None, cfg))
    captured = {}
    def fake_generate(model, tokenizer, sources, *args, **kwargs):
        captured["sources"] = sources
        return ["학교1 가다1 없다1"]
    monkeypatch.setattr(run, "generate", fake_generate)
    monkeypatch.setattr("sys.argv", [
        "run", "--model_dir", str(tmp_path), "--no-normalize_convention",
        "--grammar_chunks", str(grammar), "--text", "학교에 가지 않았나요?",
    ])
    run.main()
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{"):])
    assert payload["evidence_used_for_translation"] is False
    assert payload["evidence"][0]["chunk_id"] == "synthetic-negation"
    assert " | 문법: " not in captured["sources"][0]
