# 보존한 집계 결과

원본 ZIP의 JSON 집계 보고서에서 경로만 상대 경로로 바꿨습니다.
문장·정답·예측별 파일과 가중치는 포함하지 않습니다.
수치를 이번 정리 작업에서 다시 학습하거나 측정한 것은 아닙니다.

- model_selection_general_nikl_v1.json: 최종 NIKL 공통 경로 결정, mixed test700/full valid2336
- model_selection_v4.json: 이전 단계의 2024 전용 선택 및 proxy 검토
- model_selection_no_expert_v7.json: 이후 v6/v7 기각, 라우팅 유지
- data_prepare_v2.json / data_group_balanced_v4.json: 통합 및 train 그룹 제한 통계
- nikl_v4_repeat_safe_test700.json / nikl_v4_repeat_safe_valid2336.json: 실제 preset과 corpus별 수치
- nikl_v4_repeat_safe_multiref.json: 2024 고유 입력 85건 다중 정답
- nikl_v4_raw_2024_metrics.json: 제약 없는 디코딩 기록, repeat-safe 표와 다름
- gksl_v1_multiref.json / gksl_v1_by_corpus.json: 다중/단일 참조의 평가 범위 차이
- nikl_v4_bootstrap_*.json: 짝지은 10,000회 bootstrap

단일 정답 평가는 행 단위, 다중 정답 평가는 정규화 문장별 고유 입력 단위입니다.
보고서의 raw WER와 F1은 0~1에 제한되지 않는 WER/0~1 F1이며 정확도 백분율이 아닙니다.
F1은 bag-of-gloss 평균이고 어순은 WER로 함께 봅니다.

Mixed test700은 NIKL test2557의 부분집합(2024 90, 2025 608, 재난안전 2)입니다.
Test를 반복 실험과 모델 선택에 참고한 기록이 있으므로 완전히 untouched인 독립 test라고
표현하지 않습니다. 재난안전 2건의 점수는 대표성이 없습니다.
모든 결과에서 전문가 검증 수어 정확도를 직접 주장할 수는 없습니다.
