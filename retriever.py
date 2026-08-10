"""
retriever.py — RAG 검색 구성요소 3종

  1) GlossLexicon    : 학습셋에서 자동 구축한 (한국어 어간 -> 글로스) 사전
  2) ExampleRetriever: 입력과 유사한 학습 예시(문장->글로스 쌍) 검색
  3) GrammarRetriever: 한국수어 문법서 청크 검색

설계 의도
---------
문법서는 "형태소", "일치동사", "역할전환" 같은 메타언어적 서술이라서
입력 문장과 어휘가 거의 겹치지 않는다. 즉 문법서만으로 하는 RAG는
ke-t5-small 같은 작은 모델에게 오히려 노이즈가 되기 쉽다.
그래서 실제 성능을 끌어올리는 건 보통 (1) 사전과 (2) 유사 예시이고,
(3) 문법서는 어순/부정/문장종결 같은 규칙 힌트를 얹는 보조 역할로 둔다.
config.rag.mode 로 각각을 껐다 켜며 ablation 할 수 있게 해두었다.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 공통 유틸
# ----------------------------------------------------------------------------
RE_GLOSS_SUFFIX = re.compile(r"[0-9]+$")          # "공부1" -> "공부"
RE_GLOSS_DECOR = re.compile(r"[\[\]{}()<>]")      # 표기 기호 제거
RE_HANGUL = re.compile(r"[가-힣]+")


def normalize_gloss_token(tok: str) -> str:
    """글로스 토큰에서 의미번호/표기기호를 떼어낸 표제어를 만든다."""
    tok = RE_GLOSS_DECOR.sub("", tok).strip()
    return RE_GLOSS_SUFFIX.sub("", tok)


def gloss_tokens(gloss_seq: str) -> list[str]:
    return [t for t in gloss_seq.strip().split() if t]


# ----------------------------------------------------------------------------
# 1) 글로스 사전
# ----------------------------------------------------------------------------
class GlossLexicon:
    """
    (한국어 어간 -> 글로스) 사전을 학습 데이터에서 통계적으로 만든다.

    방법: 각 (문장, 글로스열) 쌍에서 글로스 토큰의 표제어가 문장 안에
    문자열로 등장하면 정렬된 것으로 보고 카운트한다. 형태소 분석기 없이
    동작하는 대신 recall이 낮으므로, konlpy/kiwi가 있다면 어간 추출 후
    매칭하도록 `_stems()` 를 교체하면 정확도가 올라간다.
    """

    def __init__(self, min_count: int = 2, min_stem_len: int = 2):
        self.min_count = min_count
        self.min_stem_len = min_stem_len
        self.stem2gloss: dict[str, list[tuple[str, int]]] = {}
        self._stem_list: list[str] = []

    # --- 교체 지점: 형태소 분석기를 쓰려면 여기만 바꾸면 된다 ---
    @staticmethod
    def _stems(text: str) -> set[str]:
        return set(RE_HANGUL.findall(text))

    def build(self, pairs: list[tuple[str, str]]) -> "GlossLexicon":
        counter: dict[str, Counter] = defaultdict(Counter)
        for text, gloss in pairs:
            plain = RE_GLOSS_DECOR.sub("", text)
            for g in gloss_tokens(gloss):
                stem = normalize_gloss_token(g)
                if len(stem) < self.min_stem_len:
                    continue
                # 표제어가 문장 안에 그대로 나타나면 정렬로 간주
                if stem in plain:
                    counter[stem][g] += 1
                else:
                    # 어미 변화 대응: 앞 2글자만 일치해도 약한 근거로 인정
                    if len(stem) >= 2 and stem[:2] in plain:
                        counter[stem][g] += 0.3  # type: ignore[assignment]

        self.stem2gloss = {
            stem: [(g, int(c)) for g, c in cnt.most_common() if c >= self.min_count]
            for stem, cnt in counter.items()
        }
        self.stem2gloss = {k: v for k, v in self.stem2gloss.items() if v}
        # 긴 표제어부터 매칭 (최장일치)
        self._stem_list = sorted(self.stem2gloss, key=len, reverse=True)
        return self

    def lookup(self, text: str, max_items: int = 12) -> list[tuple[str, str]]:
        """문장에 등장하는 표제어들의 (표제어, 대표 글로스)를 문장 순서대로 반환."""
        plain = RE_GLOSS_DECOR.sub("", text)
        hits, covered = [], [False] * len(plain)
        for stem in self._stem_list:
            idx = plain.find(stem)
            while idx != -1:
                if not any(covered[idx: idx + len(stem)]):
                    for i in range(idx, idx + len(stem)):
                        covered[i] = True
                    hits.append((idx, stem, self.stem2gloss[stem][0][0]))
                    break
                idx = plain.find(stem, idx + 1)
            if len(hits) >= max_items * 2:
                break
        hits.sort(key=lambda x: x[0])
        return [(s, g) for _, s, g in hits[:max_items]]

    # --- 직렬화 ---
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.stem2gloss, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "GlossLexicon":
        obj = cls()
        obj.stem2gloss = {
            k: [tuple(x) for x in v]
            for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()
        }
        obj._stem_list = sorted(obj.stem2gloss, key=len, reverse=True)
        return obj


# ----------------------------------------------------------------------------
# 2) 벡터 인덱스 (TF-IDF / dense 공통 인터페이스)
# ----------------------------------------------------------------------------
class BaseIndex:
    def build(self, corpus: list[str]) -> "BaseIndex":
        raise NotImplementedError

    def search(self, query: str, top_k: int, exclude: set[int] | None = None):
        raise NotImplementedError


class TfidfIndex(BaseIndex):
    """문자 n-gram TF-IDF. 한국어 형태소 분석기 없이도 잘 동작하고 다운로드가 필요 없다."""

    def __init__(self, ngram_range=(2, 4)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=1, sublinear_tf=True
        )
        self.mat = None

    def build(self, corpus):
        self.mat = self.vec.fit_transform(corpus)
        return self

    def search(self, query, top_k, exclude=None):
        q = self.vec.transform([query])
        scores = (self.mat @ q.T).toarray().ravel()
        if exclude:
            scores[list(exclude)] = -1.0
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


class DenseIndex(BaseIndex):
    """sentence-transformers 임베딩. 의미 유사도가 필요할 때 사용."""

    def __init__(self, model_name="jhgan/ko-sroberta-multitask", batch_size=64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.emb = None

    def build(self, corpus):
        self.emb = self.model.encode(
            corpus,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return self

    def search(self, query, top_k, exclude=None):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (self.emb @ q.T).ravel()
        if exclude:
            scores[list(exclude)] = -1.0
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]


def make_index(kind: str, **kw) -> BaseIndex:
    if kind == "tfidf":
        return TfidfIndex(ngram_range=kw.get("ngram_range", (2, 4)))
    if kind == "dense":
        return DenseIndex(model_name=kw.get("model_name", "jhgan/ko-sroberta-multitask"))
    raise ValueError(f"unknown retriever: {kind}")


# ----------------------------------------------------------------------------
# 3) 예시 / 문법서 검색기
# ----------------------------------------------------------------------------
RE_NON_CONTENT = re.compile(r"[\s\.\,\!\?\'\"·…~\-]+")


def normalize_for_dedup(text: str) -> str:
    """공백/문장부호를 제거한 비교용 문자열. '가야 한다.' == '가야한다' 로 본다."""
    return RE_NON_CONTENT.sub("", text)


@dataclass
class ExampleHit:
    text: str
    gloss: str
    score: float


class ExampleRetriever:
    """학습셋에서 입력과 유사한 (문장, 글로스) 쌍을 찾는다."""

    def __init__(self, kind="tfidf", **kw):
        self.index = make_index(kind, **kw)
        self.texts: list[str] = []
        self.glosses: list[str] = []
        self._norm: list[str] = []

    def build(self, pairs: list[tuple[str, str]]):
        self.texts = [t for t, _ in pairs]
        self.glosses = [g for _, g in pairs]
        self._norm = [normalize_for_dedup(t) for t in self.texts]
        self.index.build(self.texts)
        return self

    def search(self, query: str, top_k: int, exclude_self: bool = True,
               min_score: float = 0.0, max_score: float = 0.95) -> list[ExampleHit]:
        """
        ★ 누수 방지가 이 함수의 핵심이다.
        수어 코퍼스는 같은 문장을 여러 화자가 수행한 레코드가 흔해서,
        문자열이 완전히 같지 않아도(띄어쓰기/문장부호 차이) 사실상 같은 항목이
        검색되어 정답 글로스를 그대로 흘려보낸다. 그래서 두 겹으로 막는다.
          (1) 정규화 후 문자열 동일 -> 제외
          (2) 유사도가 max_score 이상 -> 근사 중복으로 보고 제외
        학습(exclude_self=True) 시에만 적용하고, 추론 시에는 끄지 않는다.
        """
        qnorm = normalize_for_dedup(query)
        raw = self.index.search(query, top_k + 10)
        out, seen = [], set()
        for i, s in raw:
            if s < min_score:
                continue
            if exclude_self:
                if self._norm[i] == qnorm:
                    continue
                if s >= max_score:
                    continue
            # 검색 결과끼리도 중복되면 입력 예산만 낭비한다 ('~했다.' vs '~했다 .')
            if self._norm[i] in seen:
                continue
            seen.add(self._norm[i])
            out.append(ExampleHit(self.texts[i], self.glosses[i], s))
            if len(out) >= top_k:
                break
        return out


# ----------------------------------------------------------------------------
# 3-b) 문법 자질 추출 — 문법서 검색의 질을 좌우하는 부분
# ----------------------------------------------------------------------------
class GrammarFeatureExtractor:
    """
    입력 문장을 그대로 문법서에 던지면 '학교', '친구' 같은 표면 어휘가 매칭되어
    엉뚱한 청크가 딸려온다. 문법서는 메타언어 텍스트이므로, 문장에서
    **문법 현상**(부정/의문/조건/시제/인용/수량...)을 먼저 감지해서
    그 용어로 검색해야 한다.

    아래 규칙은 표층 패턴 기반의 최소 구현이다. 형태소 분석기(kiwi 등)가
    있다면 어미 태그로 판별하는 편이 훨씬 정확하다.
    """

    RULES: list[tuple[str, re.Pattern, str]] = [
        ("부정", re.compile(r"(안\s|않|못\s|못하|없다|없어|아니)"),
         "부정법 부정문 부정표현"),
        ("의문", re.compile(r"(\?|나요|까요|습니까|는가|니\?|무엇|어디|누구|언제|왜|어떻게)"),
         "문장종결 의문문 의문표지"),
        ("명령청유", re.compile(r"(세요|해라|하자|합시다|십시오|어라)"),
         "문장종결 명령문 청유문"),
        ("조건", re.compile(r"(만약|면\s|려면|다면|경우)"),
         "문장구조 조건 이어진문장 종속적 연결"),
        ("시제", re.compile(r"(어제|내일|오늘|지난|다음|전에|후에|았다|었다|겠다|것이다)"),
         "시간 부사어 시제 시간표현"),
        ("인용", re.compile(r"(라고|고\s말|말했|물었|이라며|한다고)"),
         "역할전환 인용 담화"),
        ("수량", re.compile(r"([0-9]+|한\s|두\s|세\s|여러|많이|조금|모두)"),
         "수량 수사 지숫자"),
        ("비교", re.compile(r"(보다|더\s|가장|제일|만큼)"),
         "비교 정도 부사어"),
        ("피동사동", re.compile(r"(되다|되었|시키|받다|당하)"),
         "단어 변형 동사 유형 일치동사"),
        ("위치이동", re.compile(r"(에서|으로|까지|에게|한테)"),
         "문장성분 부사어 공간 활용"),
        ("접속", re.compile(r"(그리고|그러나|하지만|그래서|때문|니까)"),
         "문장구조 이어진문장 접속표지"),
    ]

    @classmethod
    def extract(cls, text: str) -> tuple[list[str], str]:
        """(감지된 자질명 목록, 문법서 검색 질의) 반환."""
        feats, terms = [], []
        for name, pat, query_terms in cls.RULES:
            if pat.search(text):
                feats.append(name)
                terms.append(query_terms)
        if not terms:
            terms.append("문장성분 문장구조 어순")
        return feats, " ".join(terms)


@dataclass
class GrammarHit:
    text: str
    chapter: str
    section: str
    heading: str
    score: float


class GrammarRetriever:
    """한국수어 문법서 청크 검색기."""

    def __init__(self, kind="tfidf", **kw):
        self.index = make_index(kind, **kw)
        self.chunks: list[dict] = []

    def build(self, chunks_path: Path):
        self.chunks = [
            json.loads(l) for l in Path(chunks_path).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        # 검색 대상 문자열에 제목/절 정보를 붙여 주제어 매칭을 돕는다
        corpus = [
            f"{c.get('section','')} {c.get('heading','')} {c['text']}" for c in self.chunks
        ]
        self.index.build(corpus)
        return self

    def search(self, query: str, top_k: int, min_score: float = 0.0,
               query_mode: str = "feature") -> list[GrammarHit]:
        """
        query_mode
          "surface" : 문장을 그대로 질의 (표면 어휘가 매칭돼 노이즈가 크다)
          "feature" : 감지된 문법 자질 용어로 질의 (권장)
          "hybrid"  : 자질 용어 + 문장을 함께 질의
        """
        if query_mode == "surface":
            q = query
        else:
            _, feat_q = GrammarFeatureExtractor.extract(query)
            q = feat_q if query_mode == "feature" else f"{feat_q} {query}"

        out = []
        for i, s in self.index.search(q, top_k):
            if s < min_score:
                continue
            c = self.chunks[i]
            out.append(
                GrammarHit(c["text"], c.get("chapter", ""), c.get("section", ""),
                           c.get("heading", ""), s)
            )
        return out


# ----------------------------------------------------------------------------
# 4) 컨텍스트 조립
# ----------------------------------------------------------------------------
class RagContextBuilder:
    """검색 결과를 하나의 입력 문자열로 조립한다."""

    def __init__(self, cfg, lexicon=None, example_ret=None, grammar_ret=None):
        self.cfg = cfg
        self.lexicon = lexicon
        self.example_ret = example_ret
        self.grammar_ret = grammar_ret

    def build(self, text: str, is_train: bool = False) -> str:
        c = self.cfg
        parts = [c.task_prefix + text.strip()]

        if c.mode == "none":
            return parts[0]

        use_lex = c.mode in ("lexicon", "full")
        use_ex = c.mode in ("example", "full")
        use_gram = c.mode in ("grammar", "full")

        # (1) 사전 힌트 — 가장 신호가 강하므로 앞쪽에 배치
        if use_lex and self.lexicon is not None:
            hits = self.lexicon.lookup(text, c.max_lexicon_items)
            if hits:
                parts.append(c.lex_marker + " ".join(f"{s}={g}" for s, g in hits))

        # (2) 유사 예시
        if use_ex and self.example_ret is not None and c.top_k_example > 0:
            ex = self.example_ret.search(
                text,
                c.top_k_example,
                exclude_self=is_train,
                min_score=c.min_score,
                max_score=c.max_example_score,
            )
            if ex:
                parts.append(
                    c.ex_marker + " ;; ".join(f"{e.text} => {e.gloss}" for e in ex)
                )

        # (3) 문법 자질 태그 — 짧지만 신호가 분명해서 비용 대비 효율이 좋다
        if use_gram:
            feats, _ = GrammarFeatureExtractor.extract(text)
            if feats and c.include_feature_tags:
                parts.append(c.feat_marker + " ".join(feats))

        # (4) 문법 규칙 본문 — 길어서 맨 뒤 + 길이 제한
        if use_gram and self.grammar_ret is not None and c.top_k_grammar > 0:
            gr = self.grammar_ret.search(
                text, c.top_k_grammar, min_score=c.min_score,
                query_mode=c.grammar_query_mode,
            )
            if gr:
                snips = []
                for h in gr:
                    body = " ".join(h.text.split())[: c.grammar_char_limit]
                    label = h.section or h.heading
                    snips.append(f"({label}) {body}" if label else body)
                parts.append(c.gram_marker + " ".join(snips))

        return "".join(parts)
