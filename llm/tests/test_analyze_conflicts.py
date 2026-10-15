from evaluation.analyze_conflicts import classify_glosses


def test_number_suffix_variants_are_detected():
    assert "number_or_symbol_suffix_only" in classify_glosses(["맞다1", "맞다1A", "% 맞다1"])


def test_order_variants_are_detected():
    assert "order_only" in classify_glosses(["화장실 어디 곳", "화장실 곳 어디"])


def test_repeated_sequences_are_detected():
    assert "repeated_sequence" in classify_glosses(["눈치1 하지마1", "눈치1 하지마1 눈치1 하지마1"])


def test_lexical_variants_are_high_signal():
    assert "lexical_or_semantic_variant" in classify_glosses(["같다1", "알다1"])
