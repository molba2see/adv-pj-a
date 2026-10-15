"""고정 valid split에서 디코딩 설정을 탐색한다.

test split은 읽지 않는다. 작은 층화 표본으로 모든 후보를 비교한 뒤 상위 후보만
전체 valid에서 재평가하며, 모델은 한 번만 메모리에 올린다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from core.config import DataConfig
from core.artifacts import load_run_config, resolve_resource
from core.dataset import input_tag, load_dataframe
from core.metrics import compute_all
from inference.run import detect_gloss_convention, generate, load_model


DEFAULT_GRID = [
    {"name": "safe_lp1.0", "num_beams": 4, "length_penalty": 1.0, "repetition_penalty": 1.03, "no_repeat_ngram_size": 3},
    {"name": "safe_lp1.1", "num_beams": 4, "length_penalty": 1.1, "repetition_penalty": 1.03, "no_repeat_ngram_size": 3},
    {"name": "safe_lp1.2", "num_beams": 4, "length_penalty": 1.2, "repetition_penalty": 1.03, "no_repeat_ngram_size": 3},
    {"name": "safe_lp1.3", "num_beams": 4, "length_penalty": 1.3, "repetition_penalty": 1.03, "no_repeat_ngram_size": 3},
    {"name": "lp1.2_rp1.01", "num_beams": 4, "length_penalty": 1.2, "repetition_penalty": 1.01, "no_repeat_ngram_size": 3},
    {"name": "lp1.2_rp1.02", "num_beams": 4, "length_penalty": 1.2, "repetition_penalty": 1.02, "no_repeat_ngram_size": 3},
    {"name": "lp1.2_ngram2", "num_beams": 4, "length_penalty": 1.2, "repetition_penalty": 1.03, "no_repeat_ngram_size": 2},
    {"name": "lp1.2_ngram0", "num_beams": 4, "length_penalty": 1.2, "repetition_penalty": 1.03, "no_repeat_ngram_size": 0},
    {"name": "beam6_lp1.2", "num_beams": 6, "length_penalty": 1.2, "repetition_penalty": 1.03, "no_repeat_ngram_size": 3},
    {"name": "unconstrained", "num_beams": 4, "length_penalty": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--base_model", default=None,
                        help="기반 모델 ID/로컬 경로 덮어쓰기")
    parser.add_argument("--valid_file", type=Path, default=None)
    parser.add_argument(
        "--split_role",
        choices=("valid", "test"),
        default="valid",
        help="Role of --valid_file; keeps final test reports auditable.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=700)
    parser.add_argument("--full_top_k", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset", choices=["grid", "selected", "repeat_safe"], default="grid",
        help="selected는 확정된 beam6 설정, repeat_safe는 기존 beam4 반복 안전 설정만 평가합니다.",
    )
    args = parser.parse_args()
    if args.split_role == "test" and args.preset == "grid":
        parser.error("test에서는 디코딩 탐색을 금지합니다. 확정된 단일 --preset을 지정하세요.")
    return args


def stratified_sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df) or "source_corpus" not in df.columns:
        return df.reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=seed)
    groups = sorted(shuffled["source_corpus"].dropna().unique())
    per_group = max(1, limit // max(len(groups), 1))
    selected = shuffled.groupby("source_corpus", sort=True, group_keys=False).head(per_group)
    if len(selected) < limit:
        remainder = shuffled.loc[~shuffled.index.isin(selected.index)]
        selected = pd.concat([selected, remainder.head(limit - len(selected))])
    return selected.head(limit).reset_index(drop=True)


def make_sources(df: pd.DataFrame, run_cfg: dict) -> list[str]:
    tag_mode = run_cfg.get("data", {}).get("input_tag", "none")
    prefix = run_cfg.get("rag", {}).get(
        "task_prefix", "한국어를 한국수어 글로스로 변환: "
    )
    return [
        input_tag(row, tag_mode) + prefix + str(row["text"]).strip()
        for _, row in df.iterrows()
    ]


def has_triple_repeat(gloss: str) -> bool:
    tokens = gloss.split()
    return any(tokens[i] == tokens[i + 1] == tokens[i + 2] for i in range(len(tokens) - 2))


def stability(preds: list[str], refs: list[str], conventions: list[str]) -> dict:
    return {
        "empty_outputs": sum(not pred.strip() for pred in preds),
        "triple_repeats": sum(has_triple_repeat(pred) for pred in preds),
        "over_2x_reference": sum(
            len(pred.split()) > 2 * max(len(ref.split()), 1)
            for pred, ref in zip(preds, refs)
        ),
        "convention_mismatches": sum(
            detect_gloss_convention(pred) not in {"unknown", expected}
            for pred, expected in zip(preds, conventions)
        ),
    }


def evaluate(model, tokenizer, run_cfg: dict, frame: pd.DataFrame, cfg: dict, batch_size: int):
    started = time.time()
    preds = generate(
        model,
        tokenizer,
        make_sources(frame, run_cfg),
        run_cfg,
        batch_size=batch_size,
        num_beams=cfg["num_beams"],
        length_penalty=cfg["length_penalty"],
        repetition_penalty=cfg["repetition_penalty"],
        no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
    )
    refs = frame["gloss"].astype(str).tolist()
    conventions = frame.get(
        "gloss_convention", pd.Series(["unknown"] * len(frame))
    ).astype(str).tolist()
    by_corpus = {}
    if "source_corpus" in frame.columns:
        for corpus, indices in frame.groupby("source_corpus", sort=True).groups.items():
            ids = list(indices)
            by_corpus[str(corpus)] = {
                "samples": len(ids),
                **compute_all([preds[i] for i in ids], [refs[i] for i in ids]),
            }
    result = {
        "config": cfg,
        "samples": len(frame),
        "metrics": compute_all(preds, refs),
        "stability": stability(preds, refs, conventions),
        "by_corpus": by_corpus,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return result, preds


def rank_key(result: dict):
    metrics = result["metrics"]
    stable = result["stability"]
    return (
        -stable["empty_outputs"],
        -stable["triple_repeats"],
        -stable["over_2x_reference"],
        metrics["token_f1"],
        -abs(metrics["pred_len"] - metrics["ref_len"]),
        -metrics["gloss_wer"],
    )


def save_predictions(path: Path, frame: pd.DataFrame, preds: list[str]):
    out = frame.copy()
    out["pred"] = preds
    columns = [
        name for name in
        ["source_corpus", "gloss_convention", "text", "gloss", "pred"]
        if name in out.columns
    ]
    out[columns].to_csv(path, sep="\t", index=False)


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    run_cfg = load_run_config(model_dir)
    valid_file = args.valid_file or resolve_resource(run_cfg["data"]["valid_file"])
    output_dir = args.output_dir or (model_dir / "decoding_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 학습과 동일한 로더로 BOM, 원시 컬럼명, NFKC 및 완전 중복 처리를 맞춘다.
    valid = load_dataframe(DataConfig(train_file=valid_file))
    sample = stratified_sample(valid, args.max_samples, args.seed)
    model, tokenizer, loaded_cfg = load_model(model_dir, base_model=args.base_model)
    report = {
        "model_dir": str(model_dir),
        "valid_file": str(valid_file.resolve()),
        "test_split_used": args.split_role == "test",
        "sample_size": len(sample),
        "valid_size": len(valid),
        "grid": [],
        "full_valid": [],
    }
    report_path = output_dir / "decoding_tuning_report.json"
    preset_name = {
        "selected": "beam6_lp1.2",
        "repeat_safe": "safe_lp1.0",
    }.get(args.preset)
    grid = (
        [next(config for config in DEFAULT_GRID if config["name"] == preset_name)]
        if preset_name
        else DEFAULT_GRID
    )

    for index, config in enumerate(grid, 1):
        print(f"[grid {index}/{len(grid)}] {config['name']}", flush=True)
        result, preds = evaluate(
            model, tokenizer, loaded_cfg, sample, config, args.batch_size
        )
        report["grid"].append(result)
        save_predictions(output_dir / f"sample_{config['name']}.tsv", sample, preds)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({**result["metrics"], **result["stability"]}, ensure_ascii=False), flush=True)

    ranked = sorted(report["grid"], key=rank_key, reverse=True)
    top_configs = [item["config"] for item in ranked[: max(args.full_top_k, 0)]]
    for index, config in enumerate(top_configs, 1):
        print(f"[full-valid {index}/{len(top_configs)}] {config['name']}", flush=True)
        result, preds = evaluate(
            model, tokenizer, loaded_cfg, valid, config, args.batch_size
        )
        report["full_valid"].append(result)
        save_predictions(output_dir / f"valid_{config['name']}.tsv", valid, preds)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({**result["metrics"], **result["stability"]}, ensure_ascii=False), flush=True)

    final_pool = report["full_valid"] or report["grid"]
    report["selected"] = max(final_pool, key=rank_key)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[selected] {report['selected']['config']['name']}", flush=True)
    print(f"[saved] {report_path}", flush=True)


if __name__ == "__main__":
    main()
