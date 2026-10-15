from data_processing.prepare_group_balanced_v4 import diverse_candidates, repeated_variants


def test_repeated_variants_removes_only_longer_repeat():
    glosses = ["부르다1", "제스처", "제스처 제스처"]
    assert repeated_variants(glosses) == ["제스처 제스처"]


def test_diverse_candidates_keeps_all_under_limit():
    glosses = ["학교 가다", "학교 길 가다"]
    assert diverse_candidates(glosses, 3) == sorted(glosses)


def test_diverse_candidates_caps_deterministically():
    glosses = ["학교 가다", "학교 길 가다", "학교 목적 가다", "학교 어디"]
    first = diverse_candidates(glosses, 3)
    second = diverse_candidates(list(reversed(glosses)), 3)
    assert first == second
    assert len(first) == 3
