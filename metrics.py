"""
metrics.py — 글로스 시퀀스 평가 지표

BLEU만 보지 말 것. 글로스 시퀀스는 평균 3~8토큰으로 매우 짧아서
BLEU가 불안정하다. 주 지표는 WER(편집거리) + Exact Match를 권장한다.
"""

from __future__ import annotations

import numpy as np


def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def gloss_wer(preds: list[str], refs: list[str]) -> float:
    """토큰 단위 Word Error Rate (낮을수록 좋음)."""
    err = tot = 0
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        err += _levenshtein(pt, rt)
        tot += len(rt)
    return err / max(tot, 1)


def exact_match(preds: list[str], refs: list[str]) -> float:
    return float(np.mean([p.split() == r.split() for p, r in zip(preds, refs)]))


def token_f1(preds: list[str], refs: list[str]) -> float:
    """순서를 무시한 bag-of-gloss F1. 어순 오류와 어휘 오류를 분리해 볼 때 유용."""
    from collections import Counter

    f1s = []
    for p, r in zip(preds, refs):
        pc, rc = Counter(p.split()), Counter(r.split())
        overlap = sum((pc & rc).values())
        if overlap == 0:
            f1s.append(0.0)
            continue
        prec = overlap / sum(pc.values())
        rec = overlap / sum(rc.values())
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def try_bleu(preds: list[str], refs: list[str]) -> dict:
    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="none")
        chrf = sacrebleu.corpus_chrf(preds, [refs])
        return {"bleu": bleu.score, "chrf": chrf.score}
    except Exception:
        return {}


def compute_all(preds: list[str], refs: list[str]) -> dict:
    preds = [" ".join(p.split()) for p in preds]
    refs = [" ".join(r.split()) for r in refs]
    out = {
        "gloss_wer": gloss_wer(preds, refs),
        "exact_match": exact_match(preds, refs),
        "token_f1": token_f1(preds, refs),
        "pred_len": float(np.mean([len(p.split()) for p in preds])),
        "ref_len": float(np.mean([len(r.split()) for r in refs])),
    }
    out.update(try_bleu(preds, refs))
    return {k: round(v, 4) for k, v in out.items()}
