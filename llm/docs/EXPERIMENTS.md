# 실험 기록과 모델 선택

기록 출처는 2026-08-30 `llm최종.zip`과 당시 터미널 결과입니다.
과거 실험 이름을 최종 모델 이름과 구분하며 집계 JSON은 reports에 보존했습니다.

## 초기 실패

- KE-T5 attention LoRA에서 모든 입력에 지시1, 먹다1, 좋다1 등 고빈도 출력 발생.
- max target length 64에서 일부 정답이 잘려 128로 변경, NFKC 적용.
- 16건 암기 진단은 full FT F1 0.9676, LoRA F1 0.9570이나 전체 일반화는 실패.
- Gloss 토큰 확장 small-ko 실험은 test 209건 F1 0, WER 18.0561,
  평균 출력 127.7751/정답 7.0766, 동점을 26,610회 반복. 해당 경로를 최종에서 제외.
- 평가 디코딩의 -100 token ID overflow를 방어.
- Pko-t5-base로 전환하고 lm_head와 attention/FFN LoRA를 학습.
  각 변경의 단독 인과 효과를 분리 측정한 실험은 아니므로 하나의 원인으로 단정하지 않음.

## RAG 비교

Tagged no-RAG: F1 0.5192, WER 0.8023, Exact 0.0614.
Tagged grammar-RAG: F1 0.5114, WER 0.7784, Exact 0.0586.
전체 F1/Exact 및 2024 성능 하락을 고려해 no-RAG 번역을 선택.
814 문법 청크는 별도 설명 계층에 남겼으며 어휘 선택의 증명으로 제시하지 않음.

## 데이터와 후속 후보

| 후보 | 관찰 | 결정 |
|---|---|---|
| v2 convention adapters | 번호형/번호 없음 분리 | GKSL 최종 모델 유지 |
| v3 auto-medoid | GKSL WER 개선, F1/Exact 하락; NIKL 2024 F1 하락 | 기각 |
| v4 group-balanced GKSL | WER 0.4577, F1 0.6570로 기준 F1 0.6589보다 하락 | 기각 |
| NIKL 2024 v4 | test F1 0.3071 대 0.2780, WER 0.8462 대 0.8643 | 채택 |
| v5 grammar-focus | valid/test/multiref 악화, proxy 경고 감소 없음 | 기각 |
| v6 consolidation | mixed test F1 0.4890 대 0.4903 | 기각 |
| v7 interpolation | alpha별 지표 trade-off, test의 일관된 개선 없음 | 기각 |
| 2025 decode tuning | 일부 F1/Exact 상승과 WER 악화 | 기존 preset 유지 |

NIKL v4는 기존 어댑터에서 추가 학습한 모델이며, 이름에 2024가 있어도 최종 라우터에서는
2024/2025 공통 번호형 모델입니다. 두 최종 모델은 pko-t5-base를 공유합니다.

## 선택 근거와 평가 한계

혼합 test 700행의 v4 F1 0.4903, WER 0.6413은 v2 0.4852/0.6485보다 개선됐습니다.
전체 valid 2336행에서도 v4 F1 0.5086, WER 0.6206으로 점 추정치가 개선됐습니다.
다만 F1 차이 신뢰구간이 0을 포함해 큰 성능 도약이라고 주장하지 않습니다.
700행은 전체 NIKL test 2557행의 부분집합입니다.
기존 연구 과정의 test 반복 비교를 공개하고, 다음 실험에는 별도 미사용 평가셋을 권장합니다.

자동 문법 proxy는 human reference에도 경고를 내므로 전문가 평점이 아닙니다.
Reranker proxy 결과를 정답으로 사용하지 않았으며 서비스/API/영상 평가는 포함되지 않았습니다.

## 최종 ZIP의 재현 범위

ZIP에는 최종 두 어댑터, base, tokenizer, 일부 설정·집계·데이터 버전이 있습니다.
모든 중간 어댑터, 학습 로그, 난수 상태, optimizer checkpoint, 정확한 환경 lock은
포함되어 있지 않습니다. 따라서 최종 자원을 복원한 추론과 전처리 재실행은 가능하지만
가중치의 완전한 학습 재현성을 보장하지 않습니다.

## 기타 시행착오

MiniMax M3/NVIDIA 문장 증강은 인증 401 후 지속적인 429로 중단했습니다.
최종 데이터는 generated_rows=0입니다.
초기 MediaPipe/protobuf 충돌은 영상 전처리 환경 문제이며 현재 LLM 코드에는
MediaPipe/OpenCV 의존성이 없습니다.
