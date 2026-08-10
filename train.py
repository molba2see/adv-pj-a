"""
train.py — ke-t5-small SFT (전체 파인튜닝) / LoRA 파인튜닝

사용 예
-------
# 0) 문법서 청킹 (최초 1회)
python prepare_grammar.py --input /path/한국수어_문법_텍스트.txt \
                          --output data/grammar_chunks.jsonl

# 1) 전체 파인튜닝 (권장 베이스라인)
python train.py --train_file data/train.tsv --rag_mode full

# 2) LoRA 파인튜닝
python train.py --train_file data/train.tsv --rag_mode full \
                --use_lora --lora_r 16 --lora_alpha 32 --learning_rate 1e-3

# 3) RAG ablation (반드시 해볼 것)
for m in none lexicon example grammar full; do
    python train.py --train_file data/train.tsv --rag_mode $m \
                    --output_dir outputs/rag_$m
done
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from dataset import collect_gloss_vocab, prepare_splits, to_hf_dataset
from metrics import compute_all


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
        model = wrap_lora(model, args, embeddings_resized=added > 0)

    return model, tokenizer


def wrap_lora(model, args, embeddings_resized: bool = False):
    """PEFT LoRA 적용.

    T5의 LoRA 대상 모듈
      - 최소 구성 : ["q", "v"]                          (논문 기본, 파라미터 최소)
      - 권장 구성 : ["q", "k", "v", "o"]                 (attention 전체)
      - 최대 구성 : ["q","k","v","o","wi_0","wi_1","wo"] (FFN 포함, gated-gelu 계열)
    ke-t5는 T5 v1.0 구조(relu FFN)라 FFN 모듈명이 "wi"/"wo" 이다.
    모듈명이 확실치 않으면 아래 print로 실제 이름을 확인해서 넣을 것.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if args.print_module_names:
        names = sorted({n.split(".")[-1] for n, _ in model.named_modules()})
        print("[lora] 모듈 이름 후보:", names)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    # 임베딩을 resize 했다면 새 토큰 임베딩은 LoRA로 학습되지 않으므로
    # modules_to_save 로 통째로 학습 대상에 포함시켜야 한다.
    modules_to_save = ["shared", "lm_head"] if embeddings_resized else None

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
    p.add_argument("--grammar_chunks", default="data/grammar_chunks.jsonl")
    # rag
    p.add_argument("--rag_mode", default="full",
                   choices=["none", "lexicon", "example", "grammar", "full"])
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense"])
    p.add_argument("--top_k_grammar", type=int, default=1)
    p.add_argument("--top_k_example", type=int, default=2)
    p.add_argument("--grammar_char_limit", type=int, default=220)
    p.add_argument("--grammar_query_mode", default="feature",
                   choices=["surface", "feature", "hybrid"])
    p.add_argument("--max_example_score", type=float, default=0.95,
                   help="학습 시 이 유사도 이상인 예시는 근사 중복으로 보고 제외(누수 방지)")
    # model
    p.add_argument("--model_name", default="KETI-AIR/ke-t5-small")
    p.add_argument("--max_source_length", type=int, default=384)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--add_gloss_tokens", action="store_true")
    p.add_argument("--untie_word_embeddings", action="store_true",
                   help="lm_head를 임베딩과 분리한다. 주의: ke-t5 체크포인트에는 "
                        "lm_head가 없어 랜덤 초기화되며, 사전학습된 출력층을 버리게 된다.")
    # lora
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q,k,v,o")
    p.add_argument("--print_module_names", action="store_true")
    # train
    p.add_argument("--output_dir", default="outputs/ket5-small-ksl")
    p.add_argument("--num_train_epochs", type=float, default=30.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=None)  # LoRA면 자동 상향
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lr_scheduler_type", default=None,
                   help="linear(기본) | constant | cosine. 긴 과적합 테스트에는 constant 권장")
    p.add_argument("--warmup_ratio", type=float, default=None)
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

    cfg = Config()
    cfg.data.train_file = Path(args.train_file)
    cfg.data.valid_file = Path(args.valid_file) if args.valid_file else None
    cfg.data.test_file = Path(args.test_file) if args.test_file else None
    cfg.data.seed = args.seed
    cfg.rag.grammar_chunks = Path(args.grammar_chunks)
    cfg.rag.mode = args.rag_mode
    cfg.rag.retriever = args.retriever
    cfg.rag.top_k_grammar = args.top_k_grammar
    cfg.rag.top_k_example = args.top_k_example
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
    cfg.train.bf16 = args.bf16
    cfg.train.seed = args.seed
    if args.lr_scheduler_type is not None:
        cfg.train.lr_scheduler_type = args.lr_scheduler_type
    if args.warmup_ratio is not None:
        cfg.train.warmup_ratio = args.warmup_ratio

    # LoRA는 학습 가능한 파라미터가 적어 lr을 10~30배 높게 잡는다
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    elif args.use_lora:
        cfg.train.learning_rate = 1e-3

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
    splits = prepare_splits(cfg)

    # --- 스모크 테스트 모드 ---
    if args.max_train_samples:
        splits["train"] = splits["train"].head(args.max_train_samples).reset_index(drop=True)
        print(f"[smoke] train을 {len(splits['train'])}건으로 제한")
    if args.overfit:
        # 학습 데이터를 그대로 평가한다. 여기서 100% 못 맞추면 배선 버그다.
        splits["valid"] = splits["train"].copy()
        splits["test"] = splits["train"].copy()
        print("[smoke] overfit 모드: valid=test=train")
    if args.max_eval_samples:
        splits["valid"] = splits["valid"].head(args.max_eval_samples).reset_index(drop=True)
        splits["test"] = splits["test"].head(args.max_eval_samples).reset_index(drop=True)

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
        dataloader_num_workers=2,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"] if has_eval else None,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=callbacks,
        **_trainer_tokenizer_kwarg(Seq2SeqTrainer, tokenizer),
    )

    # 학습 시작 전에 초기 loss를 찍어 모델 로딩 이상을 조기에 잡는다
    try:
        import torch as _torch
        from transformers import DataCollatorForSeq2Seq as _DC

        _probe = _DC(tokenizer, model=model, label_pad_token_id=-100)(
            [ds["train"][i] for i in range(min(4, len(ds["train"])))]
        )
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
                "model_name": cfg.model.model_name,
                "use_lora": args.use_lora,
                "rag_mode": cfg.rag.mode,
                "retriever": cfg.rag.retriever,
                "max_source_length": cfg.model.max_source_length,
                "max_target_length": cfg.model.max_target_length,
                "add_gloss_tokens": cfg.model.add_gloss_tokens,
                "untie_word_embeddings": getattr(args, "untie_word_embeddings", False),
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
        with (out / "test_predictions.tsv").open("w", encoding="utf-8") as f:
            f.write("text\tgold\tpred\n")
            for t, g, p in zip(splits["test"]["text"], splits["test"]["gloss"], decoded):
                f.write(f"{t}\t{g}\t{p}\n")
        print(f"[test] 예측 저장 -> {out/'test_predictions.tsv'}")


if __name__ == "__main__":
    main()
