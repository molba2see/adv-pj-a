"""
ksl_grammar_check.py — 한국수어 문법서 기술에 근거한 규칙 기반 gloss 검사기

근거 (『한국수어 문법』 제4장 문장구조 2.2. 어순, 부정법 4.2, 문장종결 3.2)
---------------------------------------------------------------------------
R1 문장부사어(시간·양상) 위치      p.150
   "시간 관계를 나타내는 어휘는 주로 문장의 첫머리에 온다 …
    문장의 첫머리나 말미, 혹은 첫머리와 말미 모두에 제시되기도 하는데
    문장 중간에는 위치하지는 않는다"
   -> [어제][요즘][과거][옛날][오늘][지금][내일] / [다행][확실히][혹시][만약]
      이 문장 중간에 오면 위반.

R2 성분부사어 위치                 p.149-150 <표 4-3>
   "성분 부사어로 기능하는 어휘들은 … 기본적으로 서술어인 용언 다음에 위치한다"
   예: [나][걷다][무심] / [아들][공부][열심] / [나][등산][가끔]
   -> [열심][무심][가끔][자주][항상] 등이 서술어 앞에만 오면 위반 소지.

R3 부정어 위치                     제4장 부정법 4.2
   "부정어가 서술어 뒤에 나타난다"
   -> [아니다][없다][못하다][안되다][하지마]가 문두에 오면 위반 소지.

R4 수식어-피수식어 어순            p.148-149
   "한국수어에서 체언을 수식하는 어휘가 일반적으로 체언 뒤에 위치한다"
   "*[아내][예쁘다][딸][낳다]" 는 비문 표시 -> 수식어가 체언 앞이면 의미가 뒤바뀜
   -> 형용사류가 명사류 앞에 연접하면 경고(합성어 예외 있음).

R5 의문사                          제4장 문장종결 3.2
   의문사 [무엇1][언제][왜][누구][어떻게][어디]
   참조 예문 (6라) [포도주][희다][빨갛다][~든지][원하다][무엇] -> 말미
   -> 설명의문문에서 의문사가 문두에만 오면 확인 대상.

주의
----
이 검사기는 **수지(手指) 요소의 배열**만 본다. 문법서가 거듭 강조하듯
한국수어의 논항 특정은 일차적으로 비수지표지(시선, 몸의 방향, 휴지)가
담당하며 "어순은 부차적인 역할"(p.147)을 한다. gloss 문자열에는 비수지
정보가 없으므로, 여기서 나온 위반은 **확정이 아니라 검토 후보**다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

RE_NUMSUF = re.compile(r"(_[0-9]+[A-Z]?|[0-9]+[A-Z]?)$")


def lem(t: str) -> str:
    return RE_NUMSUF.sub("", t)


# --- 어휘 부류 (문법서 예시 + 코퍼스 빈출어 기반) ---
TIME_ADV = {"어제", "요즘", "과거", "옛날", "오늘", "지금", "내일", "미래",
            "이전", "다음", "예전", "최근", "당시"}
MODAL_ADV = {"다행", "확실히", "혹시", "만약", "아마", "물론"}
SENT_ADV = TIME_ADV | MODAL_ADV              # R1 대상

COMP_ADV = {"열심", "무심", "가끔", "자주", "항상", "매우", "제일", "계속",
            "벌써", "빨리", "천천히", "많이"}   # R2 대상

NEG = {"아니다", "없다", "못하다", "안되다", "하지마", "불가능", "모르다"}  # R3

WH = {"무엇", "언제", "왜", "누구", "어떻게", "어디", "얼마"}              # R5

# R4용: 상태/속성 서술어로 자주 쓰이는 어휘
ADJ = {"예쁘다", "좋다", "나쁘다", "크다", "작다", "많다", "적다", "새롭다",
       "낡다", "비싸다", "싸다", "깨끗하다", "멋지다", "어렵다", "쉽다",
       "넓다", "좁다", "빠르다", "느리다", "높다", "낮다", "뜨겁다", "차갑다"}

# R4 예외: 문법서가 인정한 합성어화 사례 (p.149)
COMPOUND_OK = {("새롭다", "집"), ("크다", "아버지"), ("독", "감기")}



VERB = {"가다", "오다", "보다", "먹다", "하다", "주다", "받다", "만들다", "바꾸다",
        "낳다", "잊다", "알다", "모르다", "생각", "말하다", "사다", "팔다",
        "쓰다", "읽다", "듣다", "웃다", "울다", "죽다", "살다", "돕다", "일하다",
        "놀다", "쉬다", "자다", "걷다", "뛰다", "타다", "내리다", "열다", "닫다"}


def is_noun_like(w: str) -> bool:
    """체언으로 기능할 가능성이 높은 어휘인지 (용언·부사류가 아닌 것)."""
    return not (w in ADJ or w in VERB or w in SENT_ADV or w in COMP_ADV
                or w in NEG or w in WH or w.startswith(("지시", "제스처")))


def check(gloss: str) -> list[str]:
    """gloss 문자열에서 문법서 규칙 위반 후보를 찾는다.

    ※ 정답 주석은 다문장 발화인 경우가 많다. 문장 경계를 알 수 없으므로
      경계 추정 없이 적용 가능한 규칙만 보수적으로 검사한다.
    """
    tk = gloss.split()
    L = [lem(t) for t in tk]
    n = len(L)
    out = []
    if n == 0:
        return ["빈 출력"]

    # R1: 문장부사어 중간배치
    #   예외 (p.150 (8나)): "주어로 기능하는 체언만이 시간 관계를 나타내는
    #   어휘 앞에 올 수 있다" -> 바로 앞이 체언이면 합법
    #   예외: 다문장 발화에서는 중간의 시간어가 후행 문장의 첫머리일 수 있으므로
    #        앞 토큰이 용언(=선행 문장 종결)이면 경계로 보고 면제
    for i, w in enumerate(L):
        if w in SENT_ADV and 0 < i < n - 1:
            prev = L[i - 1]
            if is_noun_like(prev):
                continue                      # (8나) 허용 패턴
            if prev in VERB or prev in NEG:
                continue                      # 문장 경계로 추정
            out.append(f"R1 문장부사어 중간배치 [{tk[i]}] ({i+1}/{n})")

    # R2: 성분부사어 문두배치 (서술어 뒤가 원칙)
    for i, w in enumerate(L):
        if w in COMP_ADV and i == 0 and n > 1:
            out.append(f"R2 성분부사어 문두배치 [{tk[i]}]")

    # R3: 부정어 문두
    #   예외: 문두 [아니다]는 '아뇨'에 해당하는 응답 표지로 쓰이는 경우가 많다.
    #        서술어를 부정하는 부정어와 기능이 다르므로 제외한다.
    if L[0] in NEG and L[0] != "아니다" and n > 1:
        out.append(f"R3 부정어 문두배치 [{tk[0]}]")

    # R4: 상태서술어가 '체언' 앞에 연접 (p.148-149)
    #   용언 앞은 대상이 아니며, 합성어화된 결합은 예외 (p.149)
    for i in range(n - 1):
        if L[i] in ADJ and is_noun_like(L[i + 1]):
            if (L[i], L[i + 1]) not in COMPOUND_OK:
                out.append(f"R4 수식어 선행 [{tk[i]} {tk[i+1]}]")

    # R5: 의문사 문두 단독
    if L[0] in WH and n > 1 and not any(w in WH for w in L[1:]):
        out.append(f"R5 의문사 문두배치 [{tk[0]}]")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    all_rows = []
    for path in args.inputs:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        split = "test" if "valid" not in path.name else "valid"

        agg = {"ref": Counter(), "v2": Counter(), "v4": Counter()}
        cnt = {"ref": 0, "v2": 0, "v4": 0}

        for _, r in df.iterrows():
            refs = json.loads(r["references_json"])
            items = {
                "ref": max(refs, key=lambda x: len(x.split())),
                "v2": r["baseline_prediction"],
                "v4": r["candidate_prediction"],
            }
            row = {"split": split, "text": r["text"]}
            for k, g in items.items():
                v = check(g)
                cnt[k] += len(v)
                for x in v:
                    agg[k][x.split()[0]] += 1
                row[f"{k}_gloss"] = g
                row[f"{k}_violations"] = " | ".join(v)
                row[f"{k}_n_viol"] = len(v)
            all_rows.append(row)

        n = len(df)
        print("=" * 72)
        print(f"{path.name}  (n={n})")
        print("=" * 72)
        print(f"{'':6s} {'총위반':>6s} {'문장당':>7s}   규칙별")
        for k in ("ref", "v2", "v4"):
            detail = ", ".join(f"{a}={b}" for a, b in sorted(agg[k].items()))
            print(f"{k:6s} {cnt[k]:6d} {cnt[k]/n:7.2f}   {detail}")
        print()

    out = pd.DataFrame(all_rows)
    if args.output:
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"저장 -> {args.output}")


if __name__ == "__main__":
    main()
