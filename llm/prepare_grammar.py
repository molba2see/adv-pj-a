"""
prepare_grammar.py
==================
`한국수어_문법_텍스트.txt` (PDF 추출본) -> RAG 검색용 청크(JSONL)

PDF 추출본의 구조적 특징 (실제 파일 확인 결과):
  * 페이지 구분자: \x0c (form feed), 총 251페이지
  * 짝수(좌측) 페이지 머리말: "한국수어" / "문법"  2줄
  * 홀수(우측) 페이지 머리말: "제2장-2" / "수지요소"  2줄  -> 섹션 메타데이터로 활용
  * 페이지 꼬리말: 숫자만 있는 줄 (페이지 번호)
  * 각주: "3)" "25)" 처럼 숫자+괄호로 시작하는 줄
  * 표/그림 캡션: "<표 2-10> ...", "<그림 2-4> ..."

사용법:
    python prepare_grammar.py \
        --input /mnt/user-data/uploads/한국수어_문법_텍스트.txt \
        --output data/grammar_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# 정규식 패턴
# ----------------------------------------------------------------------------
PAGE_SEP = "\x0c"
RE_BOOK_HEADER = re.compile(r"^\s*(한국수어|문법)\s*$")
RE_CHAPTER_HEADER = re.compile(r"^\s*제\s*(\d+)\s*장\s*-\s*(\d+)\s*$")
RE_PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*$")
RE_FOOTNOTE = re.compile(r"^\s*\d{1,3}\)\s")
RE_TOC_DOTS = re.compile(r"[·]{3,}")          # 목차의 점선
RE_SECTION_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+(\S.*)$")  # "2.1. 수형소"
RE_MULTI_WS = re.compile(r"[ \t]+")
RE_MULTI_NL = re.compile(r"\n{3,}")

# 본문이 시작되는 페이지 (0-based). 그 앞은 목차/서문이라 RAG 노이즈가 됨.
DEFAULT_BODY_START_PAGE = 14


@dataclass
class Chunk:
    chunk_id: str
    chapter: str      # "제2장" 등
    section: str      # "수지요소" 등 (홀수 페이지 머리말에서 추출)
    heading: str      # "2.1. 수형소" 등 (본문 내 번호 제목)
    page: int         # 원본 페이지 번호(1-based 인덱스)
    text: str
    n_chars: int


# ----------------------------------------------------------------------------
# 1) 페이지 단위 정제
# ----------------------------------------------------------------------------
def clean_page(raw_page: str) -> tuple[str, str, str]:
    """한 페이지에서 머리말/꼬리말을 제거하고 (본문, 장, 섹션)을 돌려준다."""
    lines = raw_page.split("\n")
    chapter, section = "", ""

    # --- 머리말 제거 (페이지 상단 최대 6줄만 검사) ---
    body_start = 0
    checked = 0
    for i, line in enumerate(lines):
        if checked >= 6:
            break
        s = line.strip()
        if not s:
            body_start = i + 1
            continue
        if RE_BOOK_HEADER.match(s):                 # "한국수어" / "문법"
            body_start = i + 1
            checked += 1
            continue
        m = RE_CHAPTER_HEADER.match(s)
        if m:                                        # "제2장-2"
            chapter = f"제{m.group(1)}장"
            body_start = i + 1
            checked += 1
            # 바로 다음 비어있지 않은 줄이 섹션명 ("수지요소")
            for j in range(i + 1, min(i + 4, len(lines))):
                t = lines[j].strip()
                if t:
                    if len(t) <= 20 and not t.endswith("."):
                        section = t
                        body_start = j + 1
                    break
            checked += 1
            continue
        break

    # --- 꼬리말(페이지 번호) 제거 ---
    body_end = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 5, -1), -1):
        s = lines[i].strip()
        if not s:
            continue
        if RE_PAGE_NUM.match(s):
            body_end = i
        break

    body_lines = []
    for line in lines[body_start:body_end]:
        s = RE_MULTI_WS.sub(" ", line).strip()
        if not s:
            body_lines.append("")
            continue
        if RE_PAGE_NUM.match(s):        # 표 안의 고립된 번호
            continue
        if RE_FOOTNOTE.match(s):        # 각주 -> 제거 (원하면 유지하도록 바꿀 것)
            continue
        if RE_TOC_DOTS.search(s):       # 목차 잔여물
            continue
        body_lines.append(s)

    text = RE_MULTI_NL.sub("\n\n", "\n".join(body_lines)).strip()
    return text, chapter, section


# ----------------------------------------------------------------------------
# 2) 문단 -> 청크 병합
# ----------------------------------------------------------------------------
def merge_paragraphs(paragraphs, min_chars: int, max_chars: int):
    """짧은 문단은 붙이고 긴 문단은 문장 단위로 쪼개서 적당한 크기로 만든다."""
    buf, out = "", []

    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            flush()
            # 문장 경계(다./이다./음.) 기준 분할
            sents = re.split(r"(?<=[.!?])\s+|(?<=다\.)\s*", para)
            sbuf = ""
            for s in sents:
                if not s.strip():
                    continue
                if len(sbuf) + len(s) > max_chars and len(sbuf) >= min_chars:
                    out.append(sbuf.strip())
                    sbuf = s
                else:
                    sbuf += (" " if sbuf else "") + s
            if sbuf.strip():
                buf = sbuf.strip()
            continue

        if len(buf) + len(para) + 1 > max_chars:
            flush()
            buf = para
        else:
            buf += ("\n" if buf else "") + para
            if len(buf) >= min_chars:
                flush()

    flush()
    return out


def build_chunks(
    path: Path,
    body_start_page: int = DEFAULT_BODY_START_PAGE,
    min_chars: int = 200,
    max_chars: int = 600,
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    pages = raw.split(PAGE_SEP)

    chunks: list[Chunk] = []
    cur_chapter, cur_section = "", ""
    cur_heading = ""

    for pno, raw_page in enumerate(pages):
        if pno < body_start_page:
            continue

        text, chapter, section = clean_page(raw_page)
        if chapter:
            cur_chapter = chapter
        if section:
            cur_section = section
        if not text or len(text) < 40:
            continue

        # 본문 내 번호 제목("2.3. 문장의 확대") 감지 -> heading 갱신
        paragraphs = []
        for para in text.split("\n"):
            m = RE_SECTION_NUM.match(para)
            if m and len(para) < 40:
                cur_heading = para.strip()
                continue
            paragraphs.append(para)

        for text_chunk in merge_paragraphs(paragraphs, min_chars, max_chars):
            if len(text_chunk) < 80:      # 표 조각 등 너무 짧은 것 제외
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"g{len(chunks):05d}",
                    chapter=cur_chapter,
                    section=cur_section,
                    heading=cur_heading,
                    page=pno + 1,
                    text=text_chunk,
                    n_chars=len(text_chunk),
                )
            )
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", default=Path("data/grammar_chunks.jsonl"), type=Path)
    ap.add_argument("--body-start-page", type=int, default=DEFAULT_BODY_START_PAGE)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()

    chunks = build_chunks(
        args.input, args.body_start_page, args.min_chars, args.max_chars
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    avg = sum(c.n_chars for c in chunks) / max(len(chunks), 1)
    print(f"[prepare_grammar] {len(chunks)} chunks -> {args.output}")
    print(f"[prepare_grammar] 평균 길이 {avg:.0f}자")
    for c in chunks[:3]:
        print("-" * 60)
        print(f"[{c.chapter} {c.section} | {c.heading} | p.{c.page}]")
        print(c.text[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
