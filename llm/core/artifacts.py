"""Portable loading of archived model configurations; no ML dependency."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_run_config(model_dir: Path) -> Path:
    model_dir = Path(model_dir).resolve()
    for directory in (model_dir, model_dir.parent):
        candidate = directory / "run_config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"run_config.json을 찾을 수 없습니다: {model_dir}")


def load_run_config(model_dir: Path) -> dict:
    return json.loads(find_run_config(model_dir).read_text(encoding="utf-8"))


def resolve_base_model(value: str, override: str | None = None) -> str:
    if override:
        return override
    if Path(value).is_dir():
        return str(Path(value).resolve())
    # Recognize the legacy local path explicitly, without substituting other bases.
    if value.replace("\\", "/").rstrip("/").endswith("/base_model/paust_pko_t5_base"):
        bundled = PROJECT_ROOT / "base_model/paust_pko_t5_base"
        return str(bundled) if bundled.is_dir() else "paust/pko-t5-base"
    if value.startswith(("/", "./", "../")) or re.match(r"^[A-Za-z]:", value):
        raise FileNotFoundError(
            f"기반 모델 경로가 없습니다: {value}. --base_model로 새 경로를 지정하세요."
        )
    return value  # Hugging Face repository ID


def resolve_resource(value: str | None, fallback: Path | None = None) -> Path | None:
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return candidate
        normalized = value.replace("\\", "/")
        for marker in ("training_data/", "resources/"):
            if marker in normalized:
                local = PROJECT_ROOT / marker.rstrip("/") / normalized.split(marker, 1)[1]
                if local.is_file():
                    return local
        if normalized.endswith("grammar_chunks.jsonl"):
            local = PROJECT_ROOT / "resources/grammar_chunks.jsonl"
            if local.is_file():
                return local
    if fallback is not None and fallback.is_file():
        return fallback
    return Path(value) if value else fallback
