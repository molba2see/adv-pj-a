"""
inference.py — 학습된 모델로 한국어 문장 -> 한국수어 글로스 변환

학습 때와 **동일한 RAG 컨텍스트**를 재구성해야 한다. 이게 어긋나면
train/inference mismatch로 성능이 급락한다. 그래서 여기서도
train.tsv를 읽어 사전/예시 인덱스를 다시 만든다.
(운영 환경이라면 인덱스를 pickle로 떠서 재사용할 것)

사용 예
-------
python -m inference.run --model_dir outputs/pko_t5_base_ksl_ratio1to4_none_v1 \
                    --train_file data/train.tsv \
                    --text "공부를 더 많이 해 봐야겠다."

python -m inference.run --model_dir outputs/pko_t5_base_ksl_ratio1to4_none_v1 \
                    --train_file data/train.tsv \
                    --input_file sentences.txt --output_file preds.tsv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import re
import sys
from pathlib import Path

from core.artifacts import find_run_config, load_run_config, resolve_base_model, resolve_resource
from core.config import Config, DataConfig
from core.dataset import build_rag, load_dataframe, split_dataframe
from core.model_router import RECOMMENDED_MODELS, resolve_model_dir


NUMBERED_GLOSS_RE = re.compile(r".*\d+(?:_\d+)?[A-Z]?(?:#.*)?$")
NUMBERED_GLOSS_PARTS_RE = re.compile(
    r"^(.*?)(\d+(?:_\d+)?[A-Z]?)(?:#.*)?$"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NIKL_CONVENTION_REFERENCE = (
    PROJECT_ROOT
    / "training_data/ksl_unified_group_balanced_v4_subsets/nikl_numbered/train.csv"
)


def detect_gloss_convention(gloss: str) -> str:
    tokens = gloss.split()
    if not tokens:
        return "unknown"
    numbered_fraction = sum(
        bool(NUMBERED_GLOSS_RE.fullmatch(token)) for token in tokens
    ) / len(tokens)
    return "nikl_numbered" if numbered_fraction >= 0.5 else "gksl_unnumbered"


def build_high_confidence_nikl_map(train_df, min_count: int = 5,
                                   min_dominance: float = 0.95):
    """Build bare-token -> numbered-token mappings from human train labels only."""
    counts = defaultdict(Counter)
    rows = train_df
    if "label_type" in rows.columns:
        rows = rows[rows["label_type"] == "human_annotation"]
    for gloss in rows["gloss"].astype(str):
        for token in gloss.split():
            match = NUMBERED_GLOSS_PARTS_RE.fullmatch(token)
            if not match or not match.group(1):
                continue
            stem, suffix = match.groups()
            counts[stem][stem + suffix] += 1

    mapping, evidence = {}, {}
    for stem, variants in counts.items():
        token, count = variants.most_common(1)[0]
        total = sum(variants.values())
        dominance = count / total
        if count >= min_count and dominance >= min_dominance:
            mapping[stem] = token
            evidence[stem] = {
                "normalized": token,
                "top_count": count,
                "total_count": total,
                "dominance": round(dominance, 6),
            }
    return mapping, evidence


def normalize_nikl_gloss(gloss: str, mapping: dict[str, str], evidence: dict):
    """Number only exact bare tokens supported by a high-confidence train mapping."""
    output, changes = [], []
    for token in gloss.split():
        normalized = mapping.get(token, token)
        output.append(normalized)
        if normalized != token:
            changes.append({"raw": token, **evidence[token]})
    return " ".join(output), changes


def load_model(model_dir: Path, base_model: str | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_dir = Path(model_dir)
    run_cfg = load_run_config(model_dir)
    run_cfg["model_name"] = resolve_base_model(run_cfg["model_name"], base_model)
    tokenizer_dir = (
        model_dir if (model_dir / "tokenizer_config.json").is_file()
        else find_run_config(model_dir).parent
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    if run_cfg.get("use_lora"):
        from peft import PeftModel

        base = AutoModelForSeq2SeqLM.from_pretrained(run_cfg["model_name"])
        # 실제로 gloss 토큰을 추가한 학습에서만 embedding 크기를 변경한다.
        # 이 조건은 pko-t5를 기본으로 사용하면서도 vocab 크기가 다른 기존
        # KE-T5 어댑터를 계속 읽기 위한 하위 호환성도 보장한다.
        if (
            run_cfg.get("add_gloss_tokens", False)
            and len(tokenizer) != base.get_input_embeddings().weight.shape[0]
        ):
            base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, str(model_dir))
        model = model.merge_and_unload()   # 추론 속도를 위해 어댑터를 가중치에 병합
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    model.eval()
    return model, tokenizer, run_cfg


def generate(model, tokenizer, sources: list[str], run_cfg, batch_size=16,
             num_beams=4, repetition_penalty=1.0, no_repeat_ngram_size=0,
             length_penalty=1.0, device=None):
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=run_cfg.get("max_source_length", 384),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        beam_options = {"early_stopping": True} if num_beams > 1 else {}
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_length=run_cfg.get("max_target_length", 128),
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **beam_options,
            )
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [" ".join(o.split()) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model", default=None,
        help="기반 모델의 Hugging Face ID 또는 로컬 경로. 이전 서버 경로를 덮어씁니다.",
    )
    ap.add_argument(
        "--model_dir", type=Path,
        help="직접 사용할 모델 경로. 생략하면 --corpus 정책으로 권장 모델을 선택합니다.",
    )
    ap.add_argument(
        "--corpus",
        choices=sorted(RECOMMENDED_MODELS),
        default=None,
        help=(
            "입력 출처. nikl(기본값)/2024/2025는 통합 NIKL 번호형 모델, "
            "gksl은 번호 없는 표기 전용 모델을 선택합니다."
        ),
    )
    ap.add_argument(
        "--models_root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="권장 모델들이 있는 상위 폴더 (기본값: outputs)",
    )
    ap.add_argument("--train_file", default=None,
                    help="RAG 검색 DB인 학습 split CSV. 생략하면 run_config 경로 사용")
    ap.add_argument("--grammar_chunks", default=None,
                    help="문법 청크 JSONL. 생략하면 run_config 경로 사용")
    ap.add_argument("--text", default=None)
    ap.add_argument("--input_file", default=None)
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument(
        "--length_penalty", type=float, default=1.0,
        help="beam search 길이 보정값. 1보다 크면 상대적으로 긴 출력을 선호합니다.",
    )
    ap.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="1보다 크게 하면 같은 토큰의 과도한 반복을 억제합니다 (기본값: 1.0).",
    )
    ap.add_argument(
        "--no_repeat_ngram_size", type=int, default=0,
        help="반복을 막을 n-gram 크기. 0은 비활성화하며, 2는 연속 반복 붕괴 점검용입니다.",
    )
    ap.add_argument(
        "--repeat_safe", action="store_true",
        help="검증된 2024 반복 붕괴 완화 설정(패널티 1.03, 3-gram 차단)을 적용합니다.",
    )
    ap.add_argument(
        "--optimized_decode", action="store_true",
        help="v2 valid에서 선택된 beam 6, 길이 보정 1.2, 반복 안전 설정을 적용합니다.",
    )
    ap.add_argument(
        "--corpus_tag",
        choices=["2024", "2025"],
        default=None,
        help="태그 학습 모델에 쓸 코퍼스 표기 체계. 실사용 기본값은 2025.",
    )
    ap.add_argument(
        "--gloss_convention",
        choices=["nikl_numbered", "gksl_unnumbered", "numbered", "unknown"],
        default=None,
        help="표기 체계 태그 모델의 원하는 출력 체계. 생략 시 nikl_numbered.",
    )
    ap.add_argument(
        "--normalize_convention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="human train에서 고신뢰 대응이 확인된 bare gloss만 NIKL 번호형으로 보정합니다.",
    )
    ap.add_argument(
        "--convention_reference_file",
        type=Path,
        default=None,
        help="번호형 대응을 추출할 human train CSV. 기본값은 통합 NIKL train입니다.",
    )
    ap.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="gloss와 실제 검색된 문법 근거를 JSON/JSONL로 반환합니다(기본: 켬).",
    )
    ap.add_argument("--show_context", action="store_true")
    args = ap.parse_args()
    if args.repetition_penalty < 1.0:
        raise SystemExit("--repetition_penalty는 1.0 이상이어야 합니다.")
    if args.length_penalty <= 0:
        raise SystemExit("--length_penalty는 0보다 커야 합니다.")
    if args.no_repeat_ngram_size < 0:
        raise SystemExit("--no_repeat_ngram_size는 0 이상이어야 합니다.")
    if args.repeat_safe and args.optimized_decode:
        raise SystemExit("--repeat_safe와 --optimized_decode는 함께 지정할 수 없습니다.")
    if args.optimized_decode:
        if (
            args.num_beams != 4
            or args.length_penalty != 1.0
            or args.repetition_penalty != 1.0
            or args.no_repeat_ngram_size != 0
        ):
            raise SystemExit("--optimized_decode와 개별 디코딩 옵션은 함께 지정할 수 없습니다.")
        args.num_beams = 6
        args.length_penalty = 1.2
        args.repetition_penalty = 1.03
        args.no_repeat_ngram_size = 3
    elif args.repeat_safe:
        if args.repetition_penalty != 1.0 or args.no_repeat_ngram_size != 0:
            raise SystemExit("--repeat_safe와 개별 반복 제어 옵션은 함께 지정할 수 없습니다.")
        args.repetition_penalty = 1.03
        args.no_repeat_ngram_size = 3

    model_dir, routed_corpus = resolve_model_dir(
        corpus=args.corpus, model_dir=args.model_dir, models_root=args.models_root
    )
    if not model_dir.is_dir():
        raise SystemExit(f"모델 경로를 찾을 수 없습니다: {model_dir}")
    if args.model_dir is None:
        print(f"[router] corpus={routed_corpus} -> {model_dir}", file=sys.stderr)
    model, tokenizer, run_cfg = load_model(model_dir, base_model=args.base_model)

    # 학습 때와 같은 RAG 리소스 재구성 (train split 기준)
    cfg = Config()
    data_cfg = run_cfg.get("data", {})
    rag_cfg = run_cfg.get("rag", {})
    train_value = args.train_file or data_cfg.get("train_file")
    cfg.data.train_file = resolve_resource(train_value)
    cfg.data.valid_ratio = float(data_cfg.get("valid_ratio", cfg.data.valid_ratio))
    cfg.data.test_ratio = float(data_cfg.get("test_ratio", cfg.data.test_ratio))
    cfg.data.seed = int(data_cfg.get("seed", cfg.data.seed))
    cfg.data.input_tag = data_cfg.get("input_tag", "none")

    # v2 run_config를 우선하고, 원본 저장소의 구형 run_config도 읽을 수 있게 한다.
    cfg.rag.mode = rag_cfg.get("mode", run_cfg.get("rag_mode", "full"))
    cfg.rag.retriever = rag_cfg.get("retriever", run_cfg.get("retriever", "tfidf"))
    for name in (
        "dense_model", "top_k_grammar", "top_k_example", "max_lexicon_items",
        "lexicon_min_count", "grammar_char_limit", "min_score", "max_example_score",
        "grammar_query_mode", "include_feature_tags", "task_prefix", "lex_marker",
        "ex_marker", "feat_marker", "gram_marker",
    ):
        if name in rag_cfg:
            setattr(cfg.rag, name, rag_cfg[name])
    if "tfidf_ngram" in rag_cfg:
        cfg.rag.tfidf_ngram = tuple(rag_cfg["tfidf_ngram"])
    grammar_value = args.grammar_chunks or rag_cfg.get("grammar_chunks")
    cfg.rag.grammar_chunks = resolve_resource(
        grammar_value, PROJECT_ROOT / "resources/grammar_chunks.jsonl"
    )

    needs_train = cfg.rag.mode in {"lexicon", "example", "full"}
    train_exists = cfg.data.train_file is not None and cfg.data.train_file.is_file()
    if args.train_file and not train_exists:
        raise SystemExit(f"--train_file을 찾을 수 없습니다: {args.train_file}")
    if not train_exists and (needs_train or args.normalize_convention):
        raise SystemExit(
            "학습 CSV가 필요합니다. --train_file로 human train.csv를 지정하세요. "
            "no-RAG 원시 gloss만 확인하려면 --no-normalize_convention을 추가하세요."
        )
    if (args.explain or cfg.rag.mode in {"grammar", "full"}) and (
        cfg.rag.grammar_chunks is None or not cfg.rag.grammar_chunks.is_file()
    ):
        raise SystemExit(
            "문법 청크가 없습니다. --grammar_chunks로 지정하거나 --no-explain을 사용하세요."
        )
    if train_exists:
        df = load_dataframe(cfg.data)
    else:
        import pandas as pd

        df = pd.DataFrame(columns=["text", "gloss"])
    if data_cfg.get("explicit_splits", False) or not train_exists:
        # prepare_ksl_text_data.py가 만든 train.csv 자체가 검색 DB다.
        train_df = df
    else:
        train_df, _, _ = split_dataframe(df, cfg.data)
    builder = build_rag(cfg, train_df)
    evidence_builder = builder
    evidence_used_for_translation = cfg.rag.mode in {"grammar", "full"}
    if args.explain and not evidence_used_for_translation:
        # 번역 성능상 no-RAG 모델을 선택해도 설명 계층은 문법 근거를 별도로
        # 검색할 수 있다. 이 컨텍스트는 번역 모델 입력에는 절대 섞지 않는다.
        evidence_cfg = copy.deepcopy(cfg)
        evidence_cfg.rag.mode = "grammar"
        evidence_builder = build_rag(evidence_cfg, train_df)

    if args.text:
        sentences = [args.text]
    elif args.input_file:
        sentences = [
            l.strip() for l in Path(args.input_file).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
    else:
        raise SystemExit("--text 또는 --input_file 중 하나가 필요합니다.")

    # is_train=False: 자기 자신 제외 로직을 끄고 전체 인덱스를 조회
    built = [builder.build_structured(s, is_train=False) for s in sentences]
    if evidence_builder is not builder:
        evidence = [
            evidence_builder.build_structured(s, is_train=False) for s in sentences
        ]
        for item, evidence_item in zip(built, evidence):
            item["grammar_features"] = evidence_item["grammar_features"]
            item["grammar_hits"] = evidence_item["grammar_hits"]
    sources = [item["source"] for item in built]
    if cfg.data.input_tag == "gloss_convention":
        default_convention = (
            "gksl_unnumbered" if routed_corpus == "gksl" else "nikl_numbered"
        )
        convention = args.gloss_convention or default_convention
        sources = [f"[gloss_convention={convention}] {source}" for source in sources]
    elif cfg.data.input_tag == "corpus":
        # 통합 nikl 별칭은 학습 당시의 실제 코퍼스 태그가 아니므로 안정적인
        # 2025 태그로 내린다. 현재 권장 v4는 gloss_convention 태그형이지만,
        # 구형 corpus 태그 체크포인트를 직접 지정하는 경우까지 처리한다.
        corpus_tag = args.corpus_tag or (
            "2025" if routed_corpus == "nikl" else routed_corpus
        ) or "2025"
        sources = [f"[corpus={corpus_tag}] {source}" for source in sources]
    elif args.corpus_tag:
        # Backward compatibility for older manually tagged checkpoints.
        sources = [f"[corpus={args.corpus_tag}] {source}" for source in sources]
    if args.show_context:
        for s in sources[:3]:
            print("[context]", s[:500], "\n")

    preds = generate(
        model, tokenizer, sources, run_cfg, num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
    )

    requested_convention = args.gloss_convention or (
        "gksl_unnumbered"
        if cfg.data.input_tag == "gloss_convention" and routed_corpus == "gksl"
        else "nikl_numbered"
        if cfg.data.input_tag == "gloss_convention"
        else "unknown"
    )
    raw_preds = list(preds)
    convention_normalizations = [[] for _ in preds]
    convention_reference_file = None
    if args.normalize_convention and requested_convention == "nikl_numbered":
        reference_path = args.convention_reference_file or DEFAULT_NIKL_CONVENTION_REFERENCE
        reference_df = train_df
        if reference_path is not None and Path(reference_path).is_file():
            reference_df = load_dataframe(DataConfig(train_file=Path(reference_path)))
            convention_reference_file = str(Path(reference_path).resolve())
        mapping, mapping_evidence = build_high_confidence_nikl_map(reference_df)
        normalized = [
            normalize_nikl_gloss(pred, mapping, mapping_evidence) for pred in preds
        ]
        preds = [item[0] for item in normalized]
        convention_normalizations = [item[1] for item in normalized]

    if args.explain:
        records = []
        for sentence, raw_pred, pred, normalization, item in zip(
            sentences, raw_preds, preds, convention_normalizations, built
        ):
            actual_convention = detect_gloss_convention(pred)
            evidence = [
                {
                    "chunk_id": hit.chunk_id,
                    "heading": hit.heading,
                    "page": hit.page,
                    "score": round(hit.score, 6),
                    "excerpt": " ".join(hit.text.split())[: cfg.rag.grammar_char_limit],
                }
                for hit in item["grammar_hits"]
            ]
            if evidence:
                headings = ", ".join(
                    hit["heading"] or hit["chunk_id"] for hit in evidence
                )
                evidence_role = (
                    "번역 입력에 제공했습니다"
                    if evidence_used_for_translation
                    else "번역과 분리된 설명 계층에서 검색했습니다"
                )
                explanation = (
                    f"감지된 문법 자질({', '.join(item['grammar_features']) or '일반 문장구조'})에 "
                    f"따라 검색 근거({headings})를 {evidence_role}. "
                    "개별 gloss의 어휘 대응은 이 문법 근거만으로 증명하지 않습니다."
                )
            else:
                explanation = "관련 문법 근거를 찾지 못함"
            records.append(
                {
                    "input": sentence,
                    "gloss": pred,
                    "raw_gloss": raw_pred,
                    "gloss_convention": actual_convention,
                    "convention_normalization": normalization,
                    "convention_reference_file": convention_reference_file,
                    "requested_gloss_convention": requested_convention,
                    "convention_warning": (
                        None
                        if requested_convention in {"unknown", actual_convention}
                        else "요청한 표기 체계와 실제 생성 표기가 일치하지 않음"
                    ),
                    "grammar_features": item["grammar_features"],
                    "evidence": evidence,
                    "evidence_used_for_translation": evidence_used_for_translation,
                    "explanation": explanation,
                }
            )

        if args.output_file:
            with Path(args.output_file).open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"저장 완료 -> {args.output_file}")
        else:
            payload = records[0] if len(records) == 1 else records
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.output_file:
        with Path(args.output_file).open("w", encoding="utf-8") as f:
            f.write("text\tpred_gloss\n")
            for s, p in zip(sentences, preds):
                f.write(f"{s}\t{p}\n")
        print(f"저장 완료 -> {args.output_file}")
    else:
        for s, p in zip(sentences, preds):
            print(f"{s}\n  -> {p}\n")


if __name__ == "__main__":
    main()
