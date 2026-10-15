"""Restore selected local assets from llm최종.zip without unpacking the corpus.

Usage: python -m scripts.restore_local_assets --archive /path/to/llm최종.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

MODEL_NAMES = (
    "pko_t5_base_ksl_nikl_2024_adapter_v4",
    "pko_t5_base_ksl_v2_gksl_adapter_v1",
)
ADAPTER_FILES = {
    "adapter_config.json", "adapter_model.safetensors", "run_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    "spiece.model", "added_tokens.json",
}
BASE_FILES = {
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    "spiece.model",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def selected_member(name: str, *, include_training_data=False, include_grammar=False,
                    include_base_model=False) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        return False
    parts = path.parts
    if len(parts) == 3 and parts[0] == "outputs":
        return parts[1] in MODEL_NAMES and parts[2] in ADAPTER_FILES
    if include_base_model and len(parts) == 3 and parts[:2] == (
        "base_model", "paust_pko_t5_base"
    ):
        return parts[2] in BASE_FILES
    if include_grammar and name == "grammar_chunks.jsonl":
        return True
    # Do not restore all_text_gloss, conflicts, review tables, or predictions.
    return (
        include_training_data and len(parts) in (3, 4)
        and parts[0] == "training_data"
        and parts[-1] in {"train.csv", "valid.csv", "test.csv"}
        and parts[1] in {
            "ksl_unified_raw_human_v2",
            "ksl_unified_raw_human_v2_by_convention",
            "ksl_unified_group_balanced_v4",
            "ksl_unified_group_balanced_v4_subsets",
        }
    )


def rebase_config(config: dict, root: Path, base_model: str) -> dict:
    config = json.loads(json.dumps(config))
    config["model_name"] = base_model
    for field in ("train_file", "valid_file", "test_file"):
        value = config.get("data", {}).get(field)
        if value and "training_data/" in value.replace("\\", "/"):
            suffix = value.replace("\\", "/").split("training_data/", 1)[1]
            config["data"][field] = str(root / "training_data" / suffix)
    if "rag" in config:
        config["rag"]["grammar_chunks"] = str(root / "resources/grammar_chunks.jsonl")
    return config


def restore(archive: Path, root: Path, *, include_training_data=False,
            include_grammar=False, include_base_model=False, dry_run=False) -> list[str]:
    archive, root = archive.resolve(), root.resolve()
    flags = dict(include_training_data=include_training_data,
                 include_grammar=include_grammar, include_base_model=include_base_model)
    with zipfile.ZipFile(archive) as source:
        entries = [entry for entry in source.infolist()
                   if not entry.is_dir() and selected_member(entry.filename, **flags)]
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("중복 ZIP 항목이 있어 복원을 중단합니다.")
        for model in MODEL_NAMES:
            for required in ("run_config.json", "adapter_model.safetensors",
                             "adapter_config.json", "tokenizer.json", "tokenizer_config.json"):
                if f"outputs/{model}/{required}" not in names:
                    raise ValueError(f"최종 모델 필수 파일 누락: {model}/{required}")
        planned = []
        for entry in entries:
            name = ("resources/grammar_chunks.jsonl" if entry.filename == "grammar_chunks.jsonl"
                    else entry.filename)
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"출력 경로가 대상 폴더 밖입니다: {name}")
            if target.exists():
                raise FileExistsError(f"이미 존재하는 파일은 덮어쓰지 않습니다: {target}")
            planned.append((entry, target))
        if dry_run:
            return [target.relative_to(root).as_posix() for _, target in planned]
        base_model = (str(root / "base_model/paust_pko_t5_base")
                      if include_base_model else "paust/pko-t5-base")
        for entry, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.filename.endswith("/run_config.json"):
                config = json.loads(source.read(entry).decode("utf-8"))
                target.write_text(
                    json.dumps(rebase_config(config, root, base_model),
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            elif entry.filename.endswith("/adapter_config.json"):
                config = json.loads(source.read(entry).decode("utf-8"))
                config["base_model_name_or_path"] = base_model
                target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
            else:
                with source.open(entry) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        return [target.relative_to(root).as_posix() for _, target in planned]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--include-training-data", action="store_true")
    parser.add_argument("--include-grammar", action="store_true")
    parser.add_argument("--include-base-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = restore(args.archive, args.project_root,
                    include_training_data=args.include_training_data,
                    include_grammar=args.include_grammar,
                    include_base_model=args.include_base_model, dry_run=args.dry_run)
    for path in paths:
        print(path)
    print(f"[{'dry-run' if args.dry_run else 'restored'}] {len(paths)} files")


if __name__ == "__main__":
    main()
