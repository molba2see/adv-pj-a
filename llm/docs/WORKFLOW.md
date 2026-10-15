# 실행 가이드

모든 명령은 저장소의 `llm/` 디렉터리에서 실행합니다.
한글·공백 경로는 따옴표로 감쌉니다. 설치는 README를 참고하세요.

## 1. 데이터 준비

```bash
python -m data_processing.prepare_ksl_text_data \
  --input "/data/2024/processed_2024" \
  --input "/data/2025/raw_or_processed" \
  --input "/data/재난안전수어영상.zip" \
  --input "/data/GKSL3k_original.csv" \
  --exclude-generated --output training_data/ksl_unified_raw_human_v2
```

`prepare_report.json`의 source_files_are_read_only, generated_rows,
split_group_intersections를 확인합니다. 원본 release에 따라 행 수가 달라질 수 있습니다.

## 2. 표기 분리와 충돌 검토

```bash
python -m data_processing.prepare_convention_splits \
  --source_dir training_data/ksl_unified_raw_human_v2 \
  --output_dir training_data/ksl_unified_raw_human_v2_by_convention

python -m evaluation.analyze_conflicts \
  --data_dir training_data/ksl_unified_raw_human_v2
python -m evaluation.auto_review_conflicts \
  --data_dir training_data/ksl_unified_raw_human_v2

python -m data_processing.prepare_group_balanced_v4 \
  --source_dir training_data/ksl_unified_raw_human_v2 \
  --conflict_review training_data/ksl_unified_raw_human_v2/conflict_review_auto.csv \
  --output_dir training_data/ksl_unified_group_balanced_v4 \
  --max_refs_per_train_group 3
```

자동 검토는 전문가 검수의 대체물이 아닙니다. v4는 반복 변이/참조 수를 제한하는
실험 데이터 버전입니다. 원본 annotation은 유지합니다.
코드는 v4와 `*_subsets` 폴더를 만들며 nikl_numbered/gksl_unnumbered/nikl_2024를 나눕니다.

## 3. 16건 암기 진단

```bash
DATA_DIR="training_data/ksl_unified_group_balanced_v4_subsets/nikl_numbered"
python -m training.train \
  --train_file "$DATA_DIR/train.csv" \
  --valid_file "$DATA_DIR/valid.csv" --test_file "$DATA_DIR/test.csv" \
  --input_tag gloss_convention --rag_mode none \
  --use_lora --lora_train_lm_head \
  --learning_rate 1e-4 --lm_head_learning_rate 1e-5 \
  --lr_scheduler_type constant --warmup_ratio 0 \
  --max_train_samples 16 --max_eval_samples 16 --max_steps 500 --overfit \
  --output_dir outputs/overfit16
```

진단의 valid/test는 train 사본입니다. 이 점수를 일반화 성능으로 보고하지 않습니다.
정답·출력 형식과 EOS 종료, 반복을 확인한 뒤 별도의 전체 학습을 시작합니다.

## 4. 학습과 재개

README의 새 baseline 명령을 사용합니다. GPU가 bf16을 지원할 때만 `--bf16`을 추가합니다.
실제 target 길이를 확인하고 128을 사용하며 새 gloss 토큰 추가는 기본 비활성화입니다.

- `--init_adapter /path/to/adapter`: 기존 어댑터를 초기값으로 새 optimizer에서 추가 학습
- `--resume /path/to/checkpoint-N`: optimizer/scheduler를 포함한 중단 학습 재개
- `--lr_scheduler_type constant`: 진단 실험용 일정 학습률
- `--metric_for_best_model eval_token_f1`: 최고 체크포인트 선택 기준

최종 가중치를 그대로 쓰려면 ZIP 복원을 사용합니다. `configs/*.run_config.json`은
실험 기록용이며 training CLI가 직접 읽는 설정 파일은 아닙니다.
ZIP에는 모든 중간 초기 어댑터와 optimizer checkpoint가 없어 최종 모델의
학습 전 과정을 바이트 단위로 재현할 수는 없습니다.

## 5. 추론

```bash
python -m inference.run --corpus nikl --repeat_safe \
  --no-explain --no-normalize_convention --text "친구가 없으니 심심하죠."
python -m inference.run --corpus gksl \
  --no-explain --no-normalize_convention --text "집에 불이 났어요."
```

`--models_root`로 두 어댑터의 상위 폴더를 지정하거나 `--model_dir`로 직접 선택합니다.
원본 ZIP의 실행 설정이 예전 서버를 가리키면 `--base_model`, `--train_file`,
`--grammar_chunks`로 현재 자원 위치를 명시합니다.
원시 추론은 학습 CSV 없이 가능하고 번호 보정에는 human train CSV가 필요합니다.

## 6. 디코딩 탐색과 평가

탐색은 valid에서 수행합니다.

```bash
MODEL_DIR="outputs/pko_t5_base_ksl_nikl_2024_adapter_v4"
DATA_DIR="training_data/ksl_unified_group_balanced_v4_subsets/nikl_numbered"
python -m evaluation.tune_decoding \
  --model_dir "$MODEL_DIR" --valid_file "$DATA_DIR/valid.csv" \
  --split_role valid --output_dir outputs/valid_decoding_search
```

확정된 repeat-safe만 test에 평가합니다.

```bash
python -m evaluation.tune_decoding \
  --model_dir "$MODEL_DIR" --valid_file "$DATA_DIR/test.csv" \
  --split_role test --preset repeat_safe --max_samples 0 \
  --full_top_k 0 --output_dir outputs/fixed_test_evaluation
```

이 CLI의 `--valid_file` 이름은 평가 입력 경로입니다.
`--split_role test`로 역할을 기록하고, test에서는 단일 preset만 평가하세요.
`--max_samples 0`은 전체 입력입니다.
디코딩 평가가 만든 `sample_safe_lp1.0.tsv`로 다중 정답 평가를 실행합니다.

```bash
python -m evaluation.evaluate_multiref \
  --predictions outputs/fixed_test_evaluation/sample_safe_lp1.0.tsv \
  --test_file "$DATA_DIR/test.csv" \
  --output outputs/fixed_test_evaluation/test_metrics_multiref.json

python -m evaluation.compare_paired_predictions --help
```

다중 정답 평가는 동일 정규화 문장의 참조 중 편집거리가 작은 참조를 사용합니다.
단일 정답·행 단위 평가와 표본/기준이 다르므로 구분해 보고합니다.
코드의 디코딩 평가와 `inference.run`의 train 기반 번호 후처리는 별개입니다.

## 7. 테스트

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q core data_processing evaluation inference scripts training tests
```

회귀 테스트는 합성 fixture만 사용합니다. GPU 학습과 최종 가중치 성능은 별도 실행입니다.
