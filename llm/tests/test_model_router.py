from pathlib import Path

from core.model_router import RECOMMENDED_MODELS, resolve_model_dir


def test_2024_selects_nikl_adapter():
    path, corpus = resolve_model_dir(corpus="2024", model_dir=None, models_root=Path("outputs"))

    assert corpus == "2024"
    assert path == Path("outputs") / RECOMMENDED_MODELS["2024"]
    assert path.name == "pko_t5_base_ksl_nikl_2024_adapter_v4"


def test_general_input_defaults_to_unified_nikl_model():
    path, corpus = resolve_model_dir(corpus=None, model_dir=None, models_root=Path("outputs"))

    assert corpus == "nikl"
    assert path == Path("outputs") / RECOMMENDED_MODELS["nikl"]


def test_nikl_year_routes_share_the_unified_model():
    assert RECOMMENDED_MODELS["nikl"] == RECOMMENDED_MODELS["2024"]
    assert RECOMMENDED_MODELS["nikl"] == RECOMMENDED_MODELS["2025"]


def test_gksl_selects_unnumbered_adapter():
    path, corpus = resolve_model_dir(corpus="gksl", model_dir=None, models_root=Path("outputs"))

    assert corpus == "gksl"
    assert path == Path("outputs") / RECOMMENDED_MODELS["gksl"]


def test_explicit_model_is_not_overridden():
    explicit = Path("outputs/experiment")

    path, corpus = resolve_model_dir(corpus="2024", model_dir=explicit, models_root=Path("outputs"))

    assert (path, corpus) == (explicit, "2024")
