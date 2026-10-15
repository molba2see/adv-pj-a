"""Recommended Text2Gloss model selection by source corpus.

Keep this module free of ML dependencies so the routing policy can be tested
without loading a model or GPU libraries.
"""

from __future__ import annotations

from pathlib import Path


RECOMMENDED_MODELS = {
    "nikl": "pko_t5_base_ksl_nikl_2024_adapter_v4",
    "2024": "pko_t5_base_ksl_nikl_2024_adapter_v4",
    "2025": "pko_t5_base_ksl_nikl_2024_adapter_v4",
    "gksl": "pko_t5_base_ksl_v2_gksl_adapter_v1",
}
DEFAULT_CORPUS = "nikl"


def resolve_model_dir(
    *,
    corpus: str | None,
    model_dir: Path | None,
    models_root: Path,
) -> tuple[Path, str | None]:
    """Return an explicitly supplied model, or the recommended routed model.

    ``None`` routes to the general NIKL model. An explicit ``model_dir`` is
    intentionally never overridden: it is needed for experiments and for
    backwards-compatible invocation.
    """
    if model_dir is not None:
        return model_dir, corpus

    selected_corpus = corpus or DEFAULT_CORPUS
    try:
        model_name = RECOMMENDED_MODELS[selected_corpus]
    except KeyError as exc:
        allowed = ", ".join(RECOMMENDED_MODELS)
        raise ValueError(f"Unsupported corpus '{selected_corpus}'. Choose one of: {allowed}.") from exc
    return models_root / model_name, selected_corpus
