"""
inference.py — 학습된 모델로 한국어 문장 -> 한국수어 글로스 변환

학습 때와 **동일한 RAG 컨텍스트**를 재구성해야 한다. 이게 어긋나면
train/inference mismatch로 성능이 급락한다. 그래서 여기서도
train.tsv를 읽어 사전/예시 인덱스를 다시 만든다.
(운영 환경이라면 인덱스를 pickle로 떠서 재사용할 것)

사용 예
-------
python inference.py --model_dir outputs/ket5-small-ksl \
                    --train_file data/train.tsv \
                    --text "공부를 더 많이 해 봐야겠다."

python inference.py --model_dir outputs/ket5-small-ksl \
                    --train_file data/train.tsv \
                    --input_file sentences.txt --output_file preds.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import Config
from dataset import build_rag, load_dataframe, split_dataframe


def load_model(model_dir: Path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    run_cfg = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if run_cfg.get("use_lora"):
        from peft import PeftModel

        base = AutoModelForSeq2SeqLM.from_pretrained(run_cfg["model_name"])
        if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
            base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, str(model_dir))
        model = model.merge_and_unload()   # 추론 속도를 위해 어댑터를 가중치에 병합
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    model.eval()
    return model, tokenizer, run_cfg


@torch.no_grad()
def generate(model, tokenizer, sources: list[str], run_cfg, batch_size=16,
             num_beams=4, device=None):
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
        gen = model.generate(
            **enc,
            max_length=run_cfg.get("max_target_length", 64),
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=0,   # 글로스는 반복이 정상적으로 나타날 수 있음
        )
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [" ".join(o.split()) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--train_file", default="data/train.tsv")
    ap.add_argument("--grammar_chunks", default="data/grammar_chunks.jsonl")
    ap.add_argument("--text", default=None)
    ap.add_argument("--input_file", default=None)
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--show_context", action="store_true")
    args = ap.parse_args()

    model, tokenizer, run_cfg = load_model(args.model_dir)

    # 학습 때와 같은 RAG 리소스 재구성 (train split 기준)
    cfg = Config()
    cfg.data.train_file = Path(args.train_file)
    cfg.rag.mode = run_cfg.get("rag_mode", "full")
    cfg.rag.retriever = run_cfg.get("retriever", "tfidf")
    cfg.rag.grammar_chunks = Path(args.grammar_chunks)

    df = load_dataframe(cfg.data)
    train_df, _, _ = split_dataframe(df, cfg.data)
    builder = build_rag(cfg, train_df)

    if args.text:
        sentences = [args.text]
    elif args.input_file:
        sentences = [
            l.strip() for l in Path(args.input_file).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
    else:
        raise SystemExit("--text 또는 --input_file 중 하나가 필요합니다.")

    # is_train=False: 자기 자신 제외 로직을 끄고 전체 인덱스를 조회
    sources = [builder.build(s, is_train=False) for s in sentences]
    if args.show_context:
        for s in sources[:3]:
            print("[context]", s[:500], "\n")

    preds = generate(model, tokenizer, sources, run_cfg, num_beams=args.num_beams)

    if args.output_file:
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
