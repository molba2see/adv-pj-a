import pytest

from evaluation.tune_decoding import parse_args


def test_test_split_rejects_parameter_search(monkeypatch):
    monkeypatch.setattr("sys.argv", ["tune", "--model_dir", "model", "--split_role", "test"])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_test_split_accepts_fixed_preset(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "tune", "--model_dir", "model", "--split_role", "test", "--preset", "repeat_safe"
    ])
    assert parse_args().preset == "repeat_safe"
