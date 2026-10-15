from inference.run import detect_gloss_convention


def test_detects_numbered_nikl_gloss():
    assert (
        detect_gloss_convention("왜1 학교1 가다1 아니다1")
        == "nikl_numbered"
    )


def test_detects_unnumbered_gksl_gloss():
    assert detect_gloss_convention("집 불 나다") == "gksl_unnumbered"


def test_empty_gloss_convention_is_unknown():
    assert detect_gloss_convention("") == "unknown"
