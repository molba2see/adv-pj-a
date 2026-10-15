"""
train.py — pko-t5-base 한국어→KSL gloss SFT / LoRA 파인튜닝

사용 예
-------
# 0) 문법서 청킹 (최초 1회)
python -m data_processing.prepare_grammar --input /path/한국수어_문법_텍스트.txt \
                          --output grammar_chunks.jsonl

# 1) 전체 파인튜닝 (권장 베이스라인)
python -m training.train --train_file data/train.tsv --rag_mode none

# 2) LoRA 파인튜닝
python -m training.train --train_file data/train.tsv --rag_mode grammar \
                --use_lora --lora_r 16 --lora_alpha 32 \
                --lora_train_lm_head --learning_rate 1e-4

# 3) RAG ablation (반드시 해볼 것)
for m in none lexicon example grammar full; do
    python -m training.train --train_file data/train.tsv --rag_mode $m \
                    --output_dir outputs/rag_$m
done
"""

from __future__ import annotations

import argparse
import math
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from core.config import Config
from core.dataset import collect_gloss_vocab, prepare_splits, to_hf_dataset
from core.metrics import compute_all


def balance_training_corpora(df, mode: str, max_multiplier: float, seed: int):
    """학습 split만 복제해 소수 말뭉치를 완만하게 보강한다.

    원본 CSV/JSONL은 수정하지 않는다. ``sqrt``는 말뭉치 크기 차이의 제곱근만큼
    보강하므로 22행짜리 말뭉치를 700행까지 무리하게 복제하는 일을 피한다.
    """
    if mode == "none" or "source_corpus" not in df.columns or len(df) == 0:
        return df

    counts = df["source_corpus"].value_counts()
    if len(counts) < 2:
        print("[balance] 말뭉치가 하나뿐이므로 균형 보강을 생략합니다.")
        return df

    largest = int(counts.max())
    pieces = []
    rng = np.random.RandomState(seed)

    print(f"[balance] 적용 전: {counts.to_dict()}")
    for corpus, count_value in counts.items():
        group = df[df["source_corpus"] == corpus]
        count = int(count_value)
        if mode == "sqrt":
            multiplier = min(math.sqrt(largest / count), max_multiplier)
            target = int(math.ceil(count * multiplier))
        elif mode == "equal":
            target = min(largest, int(math.ceil(count * max_multiplier)))
        else:
            raise ValueError(f"지원하지 않는 corpus_balance: {mode}")

        # 원본 행은 한 번씩 반드시 포함하고 부족분만 복원추출한다.
        pieces.append(group)
        extra = target - count
        if extra > 0:
            pieces.append(
                group.sample(
                    n=extra,
                    replace=True,
                    random_state=int(rng.randint(0, 2**31 - 1)),
                )
            )

    import pandas as pd

    balanced = pd.concat(pieces, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    after = balanced["source_corpus"].value_counts().to_dict()
    print(f"[balance] 적용 후: {after} (원본 파일은 변경하지 않음)")
    return balanced


def balance_grammar_focus(df, multiplier: float, seed: int):
    """Reweight train-only rows tagged by the grammar proxy after safety deduplication."""
    if multiplier <= 1.0:
        return df
    if "grammar_focus_tags" not in df.columns:
        raise ValueError(
            "--grammar_focus_multiplier > 1에는 grammar_focus_tags 열이 필요합니다."
        )
    focus = df[df["grammar_focus_tags"].fillna("").astype(str).str.strip() != ""]
    if focus.empty:
        raise ValueError("grammar_focus_tags가 채워진 학습 행이 없습니다.")
    extra = int(math.ceil(len(focus) * (multiplier - 1.0)))
    sampled = focus.sample(n=extra, replace=extra > len(focus), random_state=seed)
    import pandas as pd

    balanced = pd.concat([df, sampled], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[grammar-focus] 전체={len(df)} focus={len(focus)} extra={extra} "
        f"-> {len(balanced)} (heldout/target 수정 없음)"
    )
    return balanced


def stratified_limit(df, limit: int, column: str):
    """Deterministically limit rows while retaining every available group."""
    import pandas as pd

    if limit >= len(df) or column not in df.columns:
        return df.head(limit).reset_index(drop=True)
    groups = max(int(df[column].nunique()), 1)
    per_group = max(1, limit // groups)
    selected = df.groupby(column, sort=True, group_keys=False).head(per_group)
    if len(selected) < limit:
        remainder = df.loc[~df.index.isin(selected.index)]
        selected = pd.concat(
            [selected, remainder.head(limit - len(selected))]
        )
    return selected.head(limit).sort_index().reset_index(drop=True)


def build_dual_lr_optimizer(model, lora_lr: float, lm_head_lr: float, weight_decay: float):
    """LoRA와 출력층에 서로 다른 학습률을 쓰는 AdamW를 만든다.

    lm_head는 약 3,200만 파라미터라 LoRA와 같은 큰 학습률을 적용하면 고빈도
    글로스 하나로 출력이 붕괴하기 쉽다. LoRA는 빠르게, 출력층은 천천히 갱신한다.
    """
    groups = {
        ("lora", "decay"): [],
        ("lora", "no_decay"): [],
        ("head", "decay"): [],
        ("head", "no_decay"): [],
    }
    no_decay_terms = ("bias", "layer_norm.weight", "LayerNorm.weight")

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "head" if ("lm_head" in name or "shared.modules_to_save" in name) else "lora"
        decay = "no_decay" if any(term in name for term in no_decay_terms) else "decay"
        groups[(family, decay)].append(parameter)

    optimizer_groups = []
    for (family, decay), parameters in groups.items():
        if not parameters:
            continue
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": lm_head_lr if family == "head" else lora_lr,
                "weight_decay": 0.0 if decay == "no_decay" else weight_decay,
            }
        )

    head_params = sum(p.numel() for key, values in groups.items() if key[0] == "head" for p in values)
    lora_params = sum(p.numel() for key, values in groups.items() if key[0] == "lora" for p in values)
    print(
        "[optimizer] 개별 학습률: "
        f"LoRA/기타={lora_lr:g} ({lora_params:,} params), "
        f"lm_head={lm_head_lr:g} ({head_params:,} params)"
    )
    return torch.optim.AdamW(optimizer_groups, lr=lora_lr)


# ----------------------------------------------------------------------------
# transformers 버전 호환 헬퍼
# ----------------------------------------------------------------------------
def _filter_kwargs(cls, kwargs: dict) -> dict:
    """버전에 따라 이름이 바뀐 인자(evaluation_strategy/eval_strategy 등)를 흡수한다."""
    sig = set(inspect.signature(cls.__init__).parameters)
    out, dropped = {}, []
    for k, v in kwargs.items():
        if k in sig:
            out[k] = v
        elif k == "evaluation_strategy" and "eval_strategy" in sig:
            out["eval_strategy"] = v
        elif k == "eval_strategy" and "evaluation_strategy" in sig:
            out["evaluation_strategy"] = v
        else:
            dropped.append(k)
    if dropped:
        print(f"[compat] 지원되지 않아 무시한 인자: {dropped}")
    return out


def _trainer_tokenizer_kwarg(trainer_cls, tokenizer) -> dict:
    sig = set(inspect.signature(trainer_cls.__init__).parameters)
    if "processing_class" in sig:      # transformers >= 4.46
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


# ----------------------------------------------------------------------------
# 모델 준비
# ----------------------------------------------------------------------------
def build_model_and_tokenizer(cfg: Config, args, gloss_vocab: list[str] | None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = cfg.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(name)

    if getattr(args, "untie_word_embeddings", False):
        print("[model] tie_word_embeddings=False 로 로드한다.")
        print("[model][주의] 이 체크포인트에는 lm_head가 없으므로 출력층이 "
              "랜덤 초기화된다. 사전학습된 출력 매핑을 버리는 셈이니 "
              "tie=True 결과와 반드시 비교할 것.")
        model = AutoModelForSeq2SeqLM.from_pretrained(name, tie_word_embeddings=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(name)

    # ★ transformers 5.x에서 T5 계열의 weight tying이 풀린 채 로드되는 경우가 있다.
    #   lm_head가 랜덤 초기화되면 초기 loss가 50을 넘고 학습이 사실상 불가능해진다.
    #   (증상: 초기 loss > 25, grad_norm 수백, 예측이 무관한 토큰으로 채워짐)
    if getattr(model.config, "tie_word_embeddings", False) and not getattr(
        args, "untie_word_embeddings", False
    ):
        emb = model.get_input_embeddings().weight
        head = getattr(model, "lm_head", None)
        if head is not None and head.weight.data_ptr() != emb.data_ptr():
            print("[model] weight tying이 풀려 있어 다시 묶는다 (tie_weights)")
            model.tie_weights()
            if model.lm_head.weight.data_ptr() != emb.data_ptr():
                print("[model][경고] tie_weights()로도 묶이지 않았다. "
                      "lm_head를 임베딩으로 직접 대입한다.")
                model.lm_head.weight = model.get_input_embeddings().weight

    added = 0
    if cfg.model.add_gloss_tokens and gloss_vocab:
        # 글로스를 단일 토큰으로 다루면 디코딩이 안정되지만, 새 임베딩은 랜덤
        # 초기화이므로 데이터가 적으면 오히려 손해다. 반드시 ablation 할 것.
        added = tokenizer.add_tokens(sorted(gloss_vocab))
        if added:
            model.resize_token_embeddings(len(tokenizer))
        print(f"[model] 글로스 토큰 {added}개 추가 (vocab={len(tokenizer)})")

    if args.use_lora:
        if args.init_adapter:
            from peft import PeftModel

            adapter_path = Path(args.init_adapter)
            if not adapter_path.is_dir():
                raise FileNotFoundError(f"초기 어댑터를 찾을 수 없습니다: {adapter_path}")
            print(f"[lora] 기존 어댑터에서 추가 미세조정: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
            model.print_trainable_parameters()
        else:
            model = wrap_lora(model, args, embeddings_resized=added > 0)

    return model, tokenizer


def wrap_lora(model, args, embeddings_resized: bool = False):
    """PEFT LoRA 적용.

    T5의 LoRA 대상 모듈
      - 최소 구성 : ["q", "v"]                          (논문 기본, 파라미터 최소)
      - 권장 구성 : ["q", "k", "v", "o"]                 (attention 전체)
      - 최대 구성 : ["q","k","v","o","wi_0","wi_1","wo"] (FFN 포함, gated-gelu 계열)
    기본 pko-t5-base(T5 v1.1)는 gated FFN이라 실제 모듈명이
    "wi_0"/"wi_1"/"wo"이다. 다른 백본은 --print_module_names로 확인할 것.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if args.print_module_names:
        names = sorted({n.split(".")[-1] for n, _ in model.named_modules()})
        print("[lora] 모듈 이름 후보:", names)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    # 임베딩을 resize 했다면 새 토큰 임베딩은 LoRA로 학습되지 않으므로
    # modules_to_save 로 통째로 학습 대상에 포함시켜야 한다.
    # 한국어→gloss처럼 출력 분포가 사전학습과 크게 다른 작업에서는 attention
    # LoRA만으로는 lm_head가 일반 한국어 토큰에 머물러 한 토큰 반복으로 붕괴할 수
    # 있다. lm_head를 modules_to_save에 넣으면 출력층은 완전 학습하면서 나머지
    # 본체는 LoRA로 유지할 수 있다. 입력 임베딩을 resize한 경우에만 shared도 저장한다.
    modules_to_save = []
    if embeddings_resized:
        modules_to_save.append("shared")
    if embeddings_resized or args.lora_train_lm_head:
        modules_to_save.append("lm_head")
    modules_to_save = modules_to_save or None
    if modules_to_save:
        print(f"[lora] 완전 학습/저장 모듈: {modules_to_save}")

    lconf = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------
# 평가 함수
# ----------------------------------------------------------------------------
def make_compute_metrics(tokenizer):
    def _fn(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        # generate 결과에 -100이 섞여 들어오는 경우 방지
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_all(dp, dl)

    return _fn


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_file", default="data/train.tsv")
    p.add_argument("--valid_file", default=None)
    p.add_argument("--test_file", default=None)
    p.add_argument(
        "--input_tag",
        default="none",
        choices=["none", "corpus", "gloss_convention"],
        help="모델 입력 앞에 출처 또는 gloss 표기 체계 태그를 붙입니다.",
    )
    p.add_argument("--valid_ratio", type=float, default=0.05,
                   help="valid_file이 없을 때 내부 분할 비율")
    p.add_argument("--test_ratio", type=float, default=0.05,
                   help="valid_file이 없을 때 내부 분할 비율")
    p.add_argument("--grammar_chunks", default="resources/grammar_chunks.jsonl")
    p.add_argument("--cache_dir", default="data/cache")
    p.add_argument("--rebuild_cache", action="store_true",
                   help="기존 parquet 캐시를 읽지 않고 다시 생성")
    p.add_argument(
        "--corpus_balance",
        default="none",
        choices=["none", "sqrt", "equal"],
        help="학습 split의 말뭉치 균형 보강. 재난안전처럼 작은 말뭉치에는 sqrt 권장.",
    )
    p.add_argument(
        "--corpus_balance_max_multiplier",
        type=float,
        default=8.0,
        help="소수 말뭉치의 최대 메모리상 복제 배수(원본 파일은 수정하지 않음).",
    )
    p.add_argument(
        "--grammar_focus_multiplier",
        type=float,
        default=1.0,
        help="grammar_focus_tags가 있는 train 행의 메모리상 가중 배수(기본 1=비활성).",
    )
    # rag
    p.add_argument("--rag_mode", default="none",
                   choices=["none", "lexicon", "example", "grammar", "full"])
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense"])
    p.add_argument("--top_k_grammar", type=int, default=1)
    p.add_argument("--top_k_example", type=int, default=2)
    p.add_argument("--lexicon_min_count", type=int, default=2)
    p.add_argument("--grammar_char_limit", type=int, default=220)
    p.add_argument("--grammar_query_mode", default="feature",
                   choices=["surface", "feature", "hybrid"])
    p.add_argument("--max_example_score", type=float, default=0.95,
                   help="학습 시 이 유사도 이상인 예시는 근사 중복으로 보고 제외(누수 방지)")
    # model
    p.add_argument("--model_name", default="paust/pko-t5-base")
    p.add_argument("--max_source_length", type=int, default=384)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--add_gloss_tokens", action="store_true")
    p.add_argument("--untie_word_embeddings", action="store_true",
                   help="lm_head를 임베딩과 분리한다. 주의: 일부 T5 체크포인트에는 "
                        "lm_head가 없어 랜덤 초기화되며, 사전학습된 출력층을 버리게 된다.")
    # lora
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        default="q,k,v,o,wi_0,wi_1,wo",
        help="pko-t5-base gated FFN 기준 권장값: q,k,v,o,wi_0,wi_1,wo",
    )
    p.add_argument(
        "--lora_train_lm_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="출력 lm_head를 LoRA와 함께 완전 학습/저장합니다(기본: 켬).",
    )
    p.add_argument("--print_module_names", action="store_true")
    p.add_argument(
        "--init_adapter",
        default=None,
        help="기존 LoRA 어댑터를 초기값으로 불러와 새 optimizer로 추가 미세조정한다.",
    )
    # train
    p.add_argument("--output_dir", default="outputs/pko-t5-base-ksl")
    p.add_argument("--num_train_epochs", type=float, default=8.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=None)  # LoRA면 자동 상향
    p.add_argument(
        "--lm_head_learning_rate",
        type=float,
        default=1e-5,
        help="LoRA 사용 시 완전 학습되는 lm_head 전용 학습률.",
    )
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lr_scheduler_type", default=None,
                   help="linear(기본) | constant | cosine. 긴 과적합 테스트에는 constant 권장")
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument(
        "--metric_for_best_model",
        default="eval_token_f1",
        choices=["eval_token_f1", "eval_exact_match", "eval_gloss_wer", "eval_loss"],
        help="최고 체크포인트 선택 기준. 기본은 짧은 출력 붕괴에 강한 eval_token_f1.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None)
    # --- 스모크 테스트용 ---
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--overfit", action="store_true",
                   help="train 소수 샘플로 학습하고 같은 데이터로 평가한다. "
                        "학습 루프 검증 전용 (일반 학습에 쓰지 말 것)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.valid_ratio < 0 or args.test_ratio < 0 or args.valid_ratio + args.test_ratio >= 1:
        raise ValueError("valid_ratio/test_ratio는 0 이상이고 합이 1보다 작아야 합니다.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps는 1 이상이어야 합니다.")
    if args.lexicon_min_count < 1:
        raise ValueError("lexicon_min_count는 1 이상이어야 합니다.")
    if args.corpus_balance_max_multiplier < 1:
        raise ValueError("corpus_balance_max_multiplier는 1 이상이어야 합니다.")
    if args.grammar_focus_multiplier < 1:
        raise ValueError("grammar_focus_multiplier는 1 이상이어야 합니다.")
    if args.lm_head_learning_rate <= 0:
        raise ValueError("lm_head_learning_rate는 0보다 커야 합니다.")
    if args.use_lora and args.untie_word_embeddings:
        raise ValueError("LoRA와 --untie_word_embeddings 조합은 지원하지 않습니다.")

    cfg = Config()
    cfg.data.train_file = Path(args.train_file)
    cfg.data.valid_file = Path(args.valid_file) if args.valid_file else None
    cfg.data.test_file = Path(args.test_file) if args.test_file else None
    cfg.data.valid_ratio = args.valid_ratio
    cfg.data.test_ratio = args.test_ratio
    cfg.data.seed = args.seed
    cfg.data.input_tag = args.input_tag
    cfg.rag.grammar_chunks = Path(args.grammar_chunks)
    cfg.rag.mode = args.rag_mode
    cfg.rag.retriever = args.retriever
    cfg.rag.top_k_grammar = args.top_k_grammar
    cfg.rag.top_k_example = args.top_k_example
    cfg.rag.lexicon_min_count = args.lexicon_min_count
    cfg.rag.grammar_char_limit = args.grammar_char_limit
    cfg.rag.grammar_query_mode = args.grammar_query_mode
    cfg.rag.max_example_score = args.max_example_score
    cfg.model.model_name = args.model_name
    cfg.model.max_source_length = args.max_source_length
    cfg.model.max_target_length = args.max_target_length
    cfg.model.add_gloss_tokens = args.add_gloss_tokens
    cfg.train.output_dir = Path(args.output_dir)
    cfg.train.num_train_epochs = args.num_train_epochs
    cfg.train.per_device_train_batch_size = args.per_device_train_batch_size
    cfg.train.per_device_eval_batch_size = args.per_device_eval_batch_size
    cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.train.bf16 = args.bf16
    cfg.train.seed = args.seed
    cfg.train.metric_for_best_model = args.metric_for_best_model
    cfg.train.greater_is_better = args.metric_for_best_model not in {
        "eval_gloss_wer", "eval_loss"
    }
    if args.lr_scheduler_type is not None:
        cfg.train.lr_scheduler_type = args.lr_scheduler_type
    if args.warmup_ratio is not None:
        cfg.train.warmup_ratio = args.warmup_ratio
    if args.eval_steps is not None:
        cfg.train.eval_steps = args.eval_steps
    if args.save_steps is not None:
        cfg.train.save_steps = args.save_steps
    if args.logging_steps is not None:
        cfg.train.logging_steps = args.logging_steps

    # V4: LoRA는 빠르게 적응시키되 거대한 lm_head에는 별도 저학습률을 쓴다.
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate

    if args.overfit:
        # 짧은 실행에서 평가가 한 번도 안 돌면 검증이 안 된다
        step_budget = args.max_steps or 300
        cfg.train.eval_steps = max(step_budget // 4, 10)
        cfg.train.save_steps = cfg.train.eval_steps
        cfg.train.logging_steps = max(step_budget // 20, 5)

    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(cfg.train.seed)

    # 1) 데이터 (RAG 증강 포함, 캐시됨)
    splits = prepare_splits(
        cfg,
        cache_dir=Path(args.cache_dir),
        force_rebuild=args.rebuild_cache,
    )

    # 캐시와 원본 데이터는 그대로 두고 현재 학습 프로세스의 train split만 보강한다.
    splits["train"] = balance_training_corpora(
        splits["train"],
        mode=args.corpus_balance,
        max_multiplier=args.corpus_balance_max_multiplier,
        seed=args.seed,
    )
    splits["train"] = balance_grammar_focus(
        splits["train"], multiplier=args.grammar_focus_multiplier, seed=args.seed
    )

    # --- 스모크 테스트 모드 ---
    if args.max_train_samples:
        if (
            cfg.data.input_tag == "gloss_convention"
            and "gloss_convention" in splits["train"].columns
        ):
            # 정렬된 통합 CSV의 앞부분은 GKSL3k 한 convention에 치우친다.
            # 태그 배선 시험에서는 모든 convention을 결정적으로 포함한다.
            splits["train"] = stratified_limit(
                splits["train"],
                args.max_train_samples,
                "gloss_convention",
            )
        else:
            splits["train"] = (
                splits["train"].head(args.max_train_samples).reset_index(drop=True)
            )
        print(f"[smoke] train을 {len(splits['train'])}건으로 제한")
    if args.overfit:
        # 학습 데이터를 그대로 평가한다. 여기서 100% 못 맞추면 배선 버그다.
        splits["valid"] = splits["train"].copy()
        splits["test"] = splits["train"].copy()
        print("[smoke] overfit 모드: valid=test=train")
    if args.max_eval_samples:
        splits["valid"] = stratified_limit(
            splits["valid"], args.max_eval_samples, "source_corpus"
        )
        splits["test"] = stratified_limit(
            splits["test"], args.max_eval_samples, "source_corpus"
        )

    gloss_vocab = (
        collect_gloss_vocab(splits["train"], cfg.model.gloss_token_min_freq)
        if cfg.model.add_gloss_tokens
        else None
    )

    # 2) 모델
    model, tokenizer = build_model_and_tokenizer(cfg, args, gloss_vocab)

    ds = {k: to_hf_dataset(v, tokenizer, cfg.model) for k, v in splits.items()}

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    # 3) 학습 설정
    ta_kwargs = dict(
        output_dir=str(cfg.train.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        label_smoothing_factor=cfg.train.label_smoothing_factor,
        logging_steps=cfg.train.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        predict_with_generate=True,
        generation_max_length=cfg.model.max_target_length,
        generation_num_beams=cfg.model.num_beams,
        # ★ T5는 fp16에서 overflow -> NaN loss가 잦다. bf16 또는 fp32만 사용.
        bf16=cfg.train.bf16 and torch.cuda.is_bf16_supported(),
        fp16=False,
        seed=cfg.train.seed,
        report_to=[],
        remove_unused_columns=True,
        # 데이터는 이미 메모리의 Arrow table로 토큰화되어 있다. worker fork는
        # tokenizers 경고와 프로세스 시작 비용만 늘리므로 단일 프로세스로 읽는다.
        dataloader_num_workers=0,
    )
    targs = Seq2SeqTrainingArguments(**_filter_kwargs(Seq2SeqTrainingArguments, ta_kwargs))

    # 데이터가 적으면 해시 분할로 valid가 0건이 될 수 있다.
    # 이때 load_best_model_at_end=True면 Trainer가 에러를 낸다.
    has_eval = len(ds["valid"]) > 0
    if not has_eval:
        print("[warn] valid가 0건이다 -> 평가/베스트모델 저장을 끈다. "
              "소규모 테스트가 아니라면 --valid_file을 지정하거나 valid_ratio를 올릴 것")
        for attr, val in [("load_best_model_at_end", False),
                          ("eval_strategy", "no"), ("evaluation_strategy", "no"),
                          ("metric_for_best_model", None), ("greater_is_better", None)]:
            if hasattr(targs, attr):
                setattr(targs, attr, val)

    if args.max_steps:
        targs.max_steps = args.max_steps

    callbacks = []
    if not args.overfit and has_eval:
        # overfit 테스트에서는 조기종료가 검증을 방해한다
        callbacks.append(EarlyStoppingCallback(cfg.train.early_stopping_patience))

    trainer_extra = {}
    if args.use_lora and args.lora_train_lm_head:
        optimizer = build_dual_lr_optimizer(
            model,
            lora_lr=cfg.train.learning_rate,
            lm_head_lr=args.lm_head_learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        # scheduler는 Trainer가 전체 스텝 수를 안 뒤 생성한다. 각 param group의
        # 서로 다른 초기 lr 비율은 scheduler 적용 뒤에도 유지된다.
        trainer_extra["optimizers"] = (optimizer, None)

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"] if has_eval else None,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=callbacks,
        **_trainer_tokenizer_kwarg(Seq2SeqTrainer, tokenizer),
        **trainer_extra,
    )

    # 학습 시작 전에 초기 loss를 찍어 모델 로딩 이상을 조기에 잡는다
    try:
        import torch as _torch
        from transformers import DataCollatorForSeq2Seq as _DC

        model_keys = {"input_ids", "attention_mask", "labels"}
        probe_rows = [
            {k: v for k, v in ds["train"][i].items() if k in model_keys}
            for i in range(min(4, len(ds["train"])))
        ]
        _probe = _DC(tokenizer, model=model, label_pad_token_id=-100)(probe_rows)
        _device = next(model.parameters()).device
        _probe = {key: value.to(_device) for key, value in _probe.items()}
        model.eval()
        with _torch.no_grad():
            _l = model(**_probe).loss.item()
        model.train()
        print(f"[check] 학습 전 초기 loss = {_l:.3f}")
        _v11 = "gated" in str(getattr(model.config, "feed_forward_proj", ""))
        if _v11:
            print("[check] T5 v1.1 계열(gated-gelu) — span corruption만 사전학습된 모델이라 "
                  "초기 loss가 50~150으로 높은 것이 정상이다. "
                  "대신 수렴에 스텝이 많이 필요하다.")
        elif _l > 25:
            print("[check][경고] 초기 loss가 비정상적으로 높다(>25). "
                  "모델 출력층이 제대로 로드되지 않았을 가능성이 있다.")
    except Exception as _e:
        print(f"[check] 초기 loss 측정 실패(무시): {_e}")

    # 4) 학습
    trainer.train(resume_from_checkpoint=args.resume)

    # 5) 저장 (LoRA면 어댑터만 저장됨 -> inference.py에서 base와 합쳐 로드)
    out = cfg.train.output_dir
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "run_config_version": 2,
                "model_name": cfg.model.model_name,
                "use_lora": args.use_lora,
                "max_source_length": cfg.model.max_source_length,
                "max_target_length": cfg.model.max_target_length,
                "add_gloss_tokens": cfg.model.add_gloss_tokens,
                "untie_word_embeddings": getattr(args, "untie_word_embeddings", False),
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": [
                        value.strip()
                        for value in args.lora_target_modules.split(",")
                        if value.strip()
                    ],
                    "train_lm_head": args.lora_train_lm_head,
                } if args.use_lora else None,
                "training": {
                    "num_train_epochs": cfg.train.num_train_epochs,
                    "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
                    "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
                    "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
                    "metric_for_best_model": cfg.train.metric_for_best_model,
                    "greater_is_better": cfg.train.greater_is_better,
                    "learning_rate": cfg.train.learning_rate,
                    "lm_head_learning_rate": args.lm_head_learning_rate,
                    "corpus_balance": args.corpus_balance,
                    "corpus_balance_max_multiplier": args.corpus_balance_max_multiplier,
                    "grammar_focus_multiplier": args.grammar_focus_multiplier,
                    "eval_steps": cfg.train.eval_steps,
                    "save_steps": cfg.train.save_steps,
                    "logging_steps": cfg.train.logging_steps,
                    "max_eval_samples": args.max_eval_samples,
                },
                "data": {
                    "train_file": str(cfg.data.train_file.resolve()),
                    "valid_file": str(cfg.data.valid_file.resolve()) if cfg.data.valid_file else None,
                    "test_file": str(cfg.data.test_file.resolve()) if cfg.data.test_file else None,
                    "valid_ratio": cfg.data.valid_ratio,
                    "test_ratio": cfg.data.test_ratio,
                    "seed": cfg.data.seed,
                    "explicit_splits": cfg.data.valid_file is not None,
                    "input_tag": cfg.data.input_tag,
                },
                "rag": {
                    "mode": cfg.rag.mode,
                    "retriever": cfg.rag.retriever,
                    "dense_model": cfg.rag.dense_model,
                    "tfidf_ngram": list(cfg.rag.tfidf_ngram),
                    "top_k_grammar": cfg.rag.top_k_grammar,
                    "top_k_example": cfg.rag.top_k_example,
                    "max_lexicon_items": cfg.rag.max_lexicon_items,
                    "lexicon_min_count": cfg.rag.lexicon_min_count,
                    "grammar_char_limit": cfg.rag.grammar_char_limit,
                    "min_score": cfg.rag.min_score,
                    "max_example_score": cfg.rag.max_example_score,
                    "grammar_query_mode": cfg.rag.grammar_query_mode,
                    "include_feature_tags": cfg.rag.include_feature_tags,
                    "task_prefix": cfg.rag.task_prefix,
                    "lex_marker": cfg.rag.lex_marker,
                    "ex_marker": cfg.rag.ex_marker,
                    "feat_marker": cfg.rag.feat_marker,
                    "gram_marker": cfg.rag.gram_marker,
                    "grammar_chunks": str(cfg.rag.grammar_chunks.resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 6) 테스트 평가
    if len(ds["test"]):
        res = trainer.predict(
            ds["test"],
            max_length=cfg.model.max_target_length,
            num_beams=cfg.model.num_beams,
        )
        print("\n[test]", json.dumps(res.metrics, ensure_ascii=False, indent=2))

        preds = np.where(res.predictions < 0, tokenizer.pad_token_id, res.predictions)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold = splits["test"]["gloss"].tolist()
        if "source_corpus" in splits["test"].columns:
            corpus_metrics = {}
            corpora = splits["test"]["source_corpus"].tolist()
            for corpus in sorted(set(corpora)):
                indices = [i for i, value in enumerate(corpora) if value == corpus]
                corpus_metrics[corpus] = {
                    "samples": len(indices),
                    **compute_all(
                        [decoded[i] for i in indices],
                        [gold[i] for i in indices],
                    ),
                }
            (out / "test_metrics_by_corpus.json").write_text(
                json.dumps(corpus_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("[test/by-corpus]", json.dumps(corpus_metrics, ensure_ascii=False, indent=2))
        with (out / "test_predictions.tsv").open("w", encoding="utf-8") as f:
            f.write("source_corpus\ttext\tgold\tpred\n")
            corpora = splits["test"].get(
                "source_corpus", ["unspecified"] * len(splits["test"])
            )
            for corpus, t, g, p in zip(
                corpora, splits["test"]["text"], splits["test"]["gloss"], decoded
            ):
                f.write(f"{corpus}\t{t}\t{g}\t{p}\n")
        print(f"[test] 예측 저장 -> {out/'test_predictions.tsv'}")


if __name__ == "__main__":
    main()
