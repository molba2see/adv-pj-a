"""NIKL 2024 입력 전처리 보강 (메타 괄호·따옴표 장식 정리).

문제 진단
---------
통합 nikl_numbered train에서 NIKL 2024는 전체의 3.6%뿐인데, 2025와 성격이
완전히 다르다. 2024 korean_text의 66.7%가 괄호/따옴표를 포함하는 장문 구어체로,
수형·손모양·화자 등 **gloss 생성과 무관한 메타 주석**이 입력 노이즈가 된다.
(예: "(오른손의)", "(각진 형태)", "('제비' 수형)", "(제가)")

전처리 원칙 (데이터 안전)
-------------------------
1. 원본 CSV는 읽기 전용. 새 폴더에 전처리본을 쓴다.
2. gloss(정답)는 절대 수정하지 않는다. 입력 korean_text만 정리한다.
3. train split의 NIKL 2024 행에만 적용한다. 2025/재난안전과 valid/test는
   원본 그대로 복사해 공정 평가를 보장한다.
4. 괄호 (...) 내용은 메타 주석이므로 제거하고, 따옴표 '...' "..."는 기호만
   벗겨 실제 단어 내용은 보존한다(예: '화장실' -> 화장실).
5. 정리 후 빈 입력이 생기면 원문을 유지해 데이터 손실을 막는다.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


NIKL_2024 = "NIKL_Korean_Sign_Language_Annotated_Corpus_2024"

# (...) 메타 주석 전체 제거. 괄호 양옆 공백을 함께 흡수해, 조사 앞에 떠 있는
# 공백("거기(인천)에서" -> "거기에서")이 남지 않게 한다.
_PAREN = re.compile(r"\s*[\(（][^)）]*[\)）]\s*")
_QUOTES = re.compile(r"[\"'“”‘’]")                   # 따옴표 기호만 제거(내용 보존)
_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNC = re.compile(r"\s+([,.!?])")


def clean_text(text: str) -> str:
    """메타 괄호를 제거하고 따옴표 기호를 벗긴 뒤 공백을 정리한다."""
    # 괄호는 뒤에 공백 한 칸을 남겨, 괄호가 단어 사이에 끼인 경우("A(x)B")에도
    # 두 단어가 붙지 않게 한다. 조사가 바로 붙는 흔한 경우는 아래 공백 정리가
    # 처리한다.
    cleaned = _PAREN.sub(" ", text)
    cleaned = _QUOTES.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNC.sub(r"\1", cleaned)
    cleaned = _SPACE.sub(" ", cleaned).strip()
    cleaned = unicodedata.normalize("NFKC", cleaned)
    return cleaned


def clean_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    changed = empty_reverted = 0
    for idx, row in out.iterrows():
        if row.get("source_corpus") != NIKL_2024:
            continue
        original = str(row["korean_text"])
        cleaned = clean_text(original)
        if not cleaned:
            empty_reverted += 1
            continue  # 원문 유지(데이터 손실 방지)
        if cleaned != original:
            out.at[idx, "korean_text"] = cleaned
            changed += 1
    stats = {
        "nikl_2024_rows": int((df["source_corpus"] == NIKL_2024).sum()),
        "changed_rows": changed,
        "empty_reverted": empty_reverted,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", type=Path, required=True,
                    help="원본 nikl_numbered 폴더(train/valid/test.csv 포함, 읽기 전용)")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="전처리본을 쓸 새 폴더")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"source_dir": str(args.source_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "gloss_modified": False}

    for split in ("train", "valid", "test"):
        src = args.source_dir / f"{split}.csv"
        df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        gloss_before = df["gloss_sequence"].tolist()
        if split == "train":
            df, stats = clean_train(df)
            report["train_clean_stats"] = stats
        # gloss 불변 검증(모든 split)
        assert df["gloss_sequence"].tolist() == gloss_before, \
            f"{split}: gloss가 변경됨 — 중단"
        df.to_csv(args.output_dir / f"{split}.csv", index=False, encoding="utf-8-sig")

    import json
    (args.output_dir / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
