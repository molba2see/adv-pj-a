"""
dataset.py — TSV/CSV 로딩 -> 분할 -> RAG 증강 -> 토크나이즈

핵심 주의사항 2가지
-------------------
1) 누수 방지: 예시 검색 인덱스는 **train split으로만** 만들고,
   train 샘플을 증강할 때는 자기 자신을 검색 결과에서 제외한다.
   (valid/test는 train 인덱스를 그대로 조회하므로 제외 불필요)
2) 증강은 에폭마다 하지 않고 한 번만 수행해 디스크에 캐시한다.
   검색은 결정적(deterministic)이라 매 에폭 반복할 이유가 없다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from retriever import (
    ExampleRetriever,
    GlossLexicon,
    GrammarRetriever,
    RagContextBuilder,
    gloss_tokens,
)


# ----------------------------------------------------------------------------
# 로딩 & 분할
# ----------------------------------------------------------------------------
def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        cwd = Path.cwd()
        # 근처에 있을 법한 후보를 찾아 알려준다
        cands = []
        for d in (cwd, cwd / "data"):
            if d.is_dir():
                cands += [p for p in d.iterdir()
                          if p.suffix.lower() in {".csv", ".tsv"}
                          or (p.suffix.lower() == ".txt" and "requirement" not in p.name.lower())]
        hint = ""
        if cands:
            hint = "\n  현재 폴더에서 찾은 후보 파일:\n" + "\n".join(
                f"    --train_file {c.as_posix()}" for c in cands[:10]
            )
        raise FileNotFoundError(
            f"학습 파일을 찾을 수 없습니다: {path}\n"
            f"  현재 작업 폴더: {cwd}\n"
            f"  --train_file 로 실제 경로를 지정하세요."
            f"{hint}"
        )

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                     encoding="utf-8-sig")  # 엑셀이 붙이는 BOM 대응

    # 구분자를 잘못 잡으면 컬럼이 1개로 뭉친다 -> 반대 구분자로 재시도
    if df.shape[1] == 1:
        alt = "," if sep == "\t" else "\t"
        alt_df = pd.read_csv(path, sep=alt, dtype=str, keep_default_na=False,
                             encoding="utf-8-sig")
        if alt_df.shape[1] > 1:
            print(f"[data] 구분자 자동 교정: {sep!r} -> {alt!r}")
            df = alt_df
    return df


def load_dataframe(cfg_data) -> pd.DataFrame:
    df = _read_table(Path(cfg_data.train_file))
    need = {cfg_data.text_col, cfg_data.gloss_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"컬럼 누락: {missing} / 실제 컬럼: {list(df.columns)}")

    df = df[[c for c in (cfg_data.id_col, cfg_data.text_col, cfg_data.gloss_col) if c in df.columns]]
    df = df.rename(columns={cfg_data.text_col: "text", cfg_data.gloss_col: "gloss"})
    if cfg_data.id_col in df.columns:
        df = df.rename(columns={cfg_data.id_col: "file_id"})
    else:
        df["file_id"] = [f"auto{i:07d}" for i in range(len(df))]

    df["text"] = df["text"].str.strip()
    df["gloss"] = df["gloss"].str.strip()

    # 유니코드 정규화(NFKC). 호환 자모 'ㄱ'(U+3131)와 초성 'ᄀ'(U+1100)처럼
    # 눈에는 같지만 코드포인트가 다른 문자가 섞여 있으면 exact match 채점이
    # 부당하게 오답 처리된다. 토크나이저 왕복 결과와도 표기를 맞춰 준다.
    import unicodedata

    df["text"] = df["text"].map(lambda s: unicodedata.normalize("NFKC", s))
    df["gloss"] = df["gloss"].map(lambda s: unicodedata.normalize("NFKC", s))
    df = df[(df["text"] != "") & (df["gloss"] != "")].reset_index(drop=True)

    if cfg_data.dedup_by_text:
        before = len(df)
        df = df.drop_duplicates(subset=["text", "gloss"]).reset_index(drop=True)
        if before != len(df):
            print(f"[data] 중복 제거: {before} -> {len(df)}")
    return df


def split_dataframe(df: pd.DataFrame, cfg_data):
    """valid/test 파일이 따로 없으면 비율대로 나눈다.
    group_split_by_text=True면 동일 원문이 서로 다른 split에 흩어지지 않게 한다."""
    if cfg_data.valid_file is not None:
        train = df
        valid = load_dataframe_from(cfg_data, cfg_data.valid_file)
        test = (
            load_dataframe_from(cfg_data, cfg_data.test_file)
            if cfg_data.test_file is not None
            else valid.iloc[:0].copy()
        )
        return train, valid, test

    rng_key = "text" if cfg_data.group_split_by_text else "file_id"

    def bucket(v: str) -> float:
        h = hashlib.md5(f"{cfg_data.seed}:{v}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    b = df[rng_key].map(bucket)
    test_hi = cfg_data.test_ratio
    valid_hi = cfg_data.test_ratio + cfg_data.valid_ratio
    test = df[b < test_hi].reset_index(drop=True)
    valid = df[(b >= test_hi) & (b < valid_hi)].reset_index(drop=True)
    train = df[b >= valid_hi].reset_index(drop=True)
    print(f"[data] train={len(train)} valid={len(valid)} test={len(test)}")
    return train, valid, test


def load_dataframe_from(cfg_data, path):
    tmp = type(cfg_data)(**{**cfg_data.__dict__, "train_file": Path(path)})
    return load_dataframe(tmp)


# ----------------------------------------------------------------------------
# RAG 리소스 구축
# ----------------------------------------------------------------------------
def build_rag(cfg, train_df: pd.DataFrame) -> RagContextBuilder:
    rcfg = cfg.rag
    pairs = list(zip(train_df["text"].tolist(), train_df["gloss"].tolist()))

    lexicon = example_ret = grammar_ret = None

    if rcfg.mode in ("lexicon", "full"):
        print("[rag] 글로스 사전 구축 중...")
        lexicon = GlossLexicon(min_count=1).build(pairs)
        print(f"[rag]   표제어 {len(lexicon.stem2gloss)}개")

    if rcfg.mode in ("example", "full"):
        print(f"[rag] 예시 인덱스 구축 중 (retriever={rcfg.retriever})...")
        example_ret = ExampleRetriever(
            rcfg.retriever,
            ngram_range=tuple(rcfg.tfidf_ngram),
            model_name=rcfg.dense_model,
        ).build(pairs)

    if rcfg.mode in ("grammar", "full"):
        print(f"[rag] 문법서 인덱스 구축 중 ({rcfg.grammar_chunks})...")
        grammar_ret = GrammarRetriever(
            rcfg.retriever,
            ngram_range=tuple(rcfg.tfidf_ngram),
            model_name=rcfg.dense_model,
        ).build(Path(rcfg.grammar_chunks))
        print(f"[rag]   청크 {len(grammar_ret.chunks)}개")

    return RagContextBuilder(rcfg, lexicon, example_ret, grammar_ret)


def augment(df: pd.DataFrame, builder: RagContextBuilder, is_train: bool) -> pd.DataFrame:
    out = df.copy()
    out["source"] = [
        builder.build(t, is_train=is_train) for t in df["text"].tolist()
    ]
    out["target"] = out["gloss"]
    return out


# ----------------------------------------------------------------------------
# 토크나이즈
# ----------------------------------------------------------------------------
def to_hf_dataset(df: pd.DataFrame, tokenizer, mcfg):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["source", "target", "text", "gloss"]], preserve_index=False)

    def _tok(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=mcfg.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=mcfg.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(_tok, batched=True, remove_columns=["source", "target"], desc="tokenizing")


def collect_gloss_vocab(df: pd.DataFrame, min_freq: int) -> list[str]:
    from collections import Counter

    c = Counter()
    for g in df["gloss"]:
        c.update(gloss_tokens(g))
    return [t for t, n in c.items() if n >= min_freq]


# ----------------------------------------------------------------------------
# 캐시
# ----------------------------------------------------------------------------
def cache_key(cfg) -> str:
    r = cfg.rag
    raw = json.dumps(
        {
            "mode": r.mode, "retriever": r.retriever, "kg": r.top_k_grammar,
            "ke": r.top_k_example, "lex": r.max_lexicon_items,
            "lim": r.grammar_char_limit, "min": r.min_score,
            "maxex": r.max_example_score, "gqm": r.grammar_query_mode,
            "feat": r.include_feature_tags,
            "train": str(cfg.data.train_file), "seed": cfg.data.seed,
        },
        sort_keys=True,
    )
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def prepare_splits(cfg, cache_dir: Path = Path("data/cache")):
    """캐시가 있으면 읽고, 없으면 RAG 증강 후 저장한다."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(cfg)
    paths = {s: cache_dir / f"{s}_{key}.parquet" for s in ("train", "valid", "test")}

    if all(p.exists() for p in paths.values()):
        print(f"[data] 캐시 사용: {key}")
        return {s: pd.read_parquet(p) for s, p in paths.items()}

    df = load_dataframe(cfg.data)
    train, valid, test = split_dataframe(df, cfg.data)
    builder = build_rag(cfg, train)

    splits = {
        "train": augment(train, builder, is_train=True),
        "valid": augment(valid, builder, is_train=False),
        "test": augment(test, builder, is_train=False),
    }
    for s, p in paths.items():
        splits[s].to_parquet(p, index=False)
    print(f"[data] 캐시 저장: {key}")

    # 증강 결과 확인용 샘플 출력
    if len(splits["train"]):
        print("\n[data] 증강 입력 예시 -----------------------------")
        print(splits["train"]["source"].iloc[0][:600])
        print("-> target:", splits["train"]["target"].iloc[0])
        print("--------------------------------------------------\n")
    return splits
