import json
import zipfile

import pytest

from scripts.restore_local_assets import MODEL_NAMES, restore, selected_member


def make_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        for model in MODEL_NAMES:
            prefix = f"outputs/{model}/"
            for name in ("adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"):
                archive.writestr(prefix + name, "{}")
            archive.writestr(prefix + "adapter_config.json",
                             json.dumps({"base_model_name_or_path": "/old/base"}))
            archive.writestr(prefix + "run_config.json", json.dumps({
                "model_name": "/old/base",
                "data": {"train_file": "/old/training_data/version/train.csv"},
                "rag": {"grammar_chunks": "/old/grammar_chunks.jsonl", "mode": "none"},
            }))
        archive.writestr("training_data/ksl_unified_raw_human_v2/train.csv", "header")
        archive.writestr("grammar_chunks.jsonl", "{}\n")
        archive.writestr("outside/raw.mp4", "not selected")


def test_default_restores_only_adapters_and_rebases_paths(tmp_path):
    archive = tmp_path / "final.zip"
    make_archive(archive)
    before = archive.read_bytes()
    root = tmp_path / "llm"
    paths = restore(archive, root)
    assert all(path.startswith("outputs/") for path in paths)
    assert not (root / "training_data").exists()
    assert not (root / "resources").exists()
    cfg = json.loads((root / "outputs" / MODEL_NAMES[0] / "run_config.json").read_text())
    assert cfg["model_name"] == "paust/pko-t5-base"
    assert cfg["data"]["train_file"] == str(root / "training_data/version/train.csv")
    assert archive.read_bytes() == before


def test_optional_data_and_grammar_and_no_overwrite(tmp_path):
    archive = tmp_path / "final.zip"
    make_archive(archive)
    root = tmp_path / "llm"
    paths = restore(archive, root, include_training_data=True, include_grammar=True)
    assert "resources/grammar_chunks.jsonl" in paths
    assert (root / "training_data/ksl_unified_raw_human_v2/train.csv").exists()
    before = (root / "outputs" / MODEL_NAMES[0] / "run_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        restore(archive, root)
    assert (root / "outputs" / MODEL_NAMES[0] / "run_config.json").read_bytes() == before


def test_dry_run_and_path_traversal(tmp_path):
    archive = tmp_path / "final.zip"
    make_archive(archive)
    root = tmp_path / "llm"
    assert restore(archive, root, dry_run=True)
    assert not root.exists()
    for name in ("../evil.py", "/outputs/x", "outputs/../evil", "C:/evil", "outputs\\bad"):
        assert not selected_member(name, include_training_data=True)
    assert not selected_member("training_data/ksl_unified_raw_human_v2/all_text_gloss.csv",
                               include_training_data=True)


def test_existing_destination_symlink_cannot_escape_root(tmp_path):
    archive = tmp_path / "final.zip"
    make_archive(archive)
    root = tmp_path / "llm"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "outputs").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(ValueError):
        restore(archive, root)
    assert not list(outside.iterdir())
