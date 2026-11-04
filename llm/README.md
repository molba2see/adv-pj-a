# 한국어 → 한국수어(KSL) Gloss

한국어 문장을 한국수어의 gloss 시퀀스로 변환하는 연구 프로젝트입니다.
NIKL은 **국립국어원 모두의 말뭉치 「한국수어-한국어 병렬 말뭉치」**(2024·2025 자료)를 출처로 합니다.
재난안전 자료와 GKSL3k의 텍스트 주석도 처리하며,
`paust/pko-t5-base`에 LoRA와 출력층 학습을 적용했습니다.
데이터 전처리, 학습, 추론, 문법 RAG, 다중 정답 평가 코드를 제공합니다.

이 디렉터리는 `llm최종.zip`에서 정리한 **코드 배포본**입니다.
**이 `llm/` 배포본이 공개하는 것은 처리 코드·절차·설정·집계 결과입니다. 원본 데이터, 원문 한국어 문장–Gloss 쌍, 원문을 복원할 수 있는 행 단위 가공 자료를 공개하지 않습니다.**
원본/가공 말뭉치, 영상·skeleton, 문법 원문 청크, 모델 가중치는 Git에 포함하지 않습니다.
원본은 제공기관에서 별도로 확보해 로컬에서만 처리합니다.
실험의 집계 수치와 선택 근거는 [reports](reports/), 설정 기록은 [configs](configs/)에 있습니다.

## 프로젝트 범위

입력은 한국어 문장, 출력은 한국수어 표현을 기록한 gloss 시퀀스입니다.
NIKL은 `학교1`, `지시1_1` 같은 번호형 표기, GKSL은 번호 없는 표기를 사용하므로
두 모델 경로를 둡니다. Gloss는 수어의 주석 표현이며 비수지표지·공간 문법을 모두 담지는 못합니다.
영상·MediaPipe skeleton 생성은 후속 단계로, 이 모델의 입력에는 사용하지 않습니다.

## 모델과 구조

| 구성 | 사용 모델/방법 | 역할 |
|---|---|---|
| 최종 기반 모델 | [paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base) | 한국어 seq2seq |
| NIKL 최종 어댑터 | `pko_t5_base_ksl_nikl_2024_adapter_v4` | 번호형 gloss, 기본/2024/2025 공통 경로 |
| GKSL 최종 어댑터 | `pko_t5_base_ksl_v2_gksl_adapter_v1` | 번호 없는 gloss |
| 파인튜닝 | LoRA r=16, alpha=32, dropout=0.05 + `lm_head` 학습 | 출력 분포 적응 |
| LoRA 대상 | `q,k,v,o,wi_0,wi_1,wo` | Attention와 gated FFN |
| 문법 설명 | 문법 자질 + TF-IDF 검색 | 별도 근거 설명 계층 |
| 초기 비교 모델 | KE-T5 small/small-ko 계열 | 반복 출력 실패 실험 |
| 문장 증강 실험 | MiniMax M3 / NVIDIA NIM | API 제한으로 중단, 최종 생성 라벨 0 |

최종 모델은 gloss 전용 토큰을 추가하지 않았으며 source/target 길이는 384/128입니다.
`lm_head`는 전체 학습하므로 일반적인 attention-only LoRA보다 어댑터가 큽니다.

```mermaid
flowchart LR
    A["한국어 문장"] --> B["NIKL / GKSL 표기 선택"]
    B --> C["pko-t5-base + 선택한 LoRA"]
    C --> D["gloss 생성"]
    D --> E["원시 gloss / 번호 보정 기록"]
    A --> F["문법 자질 감지"]
    F --> G["외부 문법 청크 검색"]
    G --> H["근거가 있는 설명"]
    E --> I["gloss + 설명 JSON"]
    H --> I
```

번역은 **no-RAG**입니다. 문법 RAG는 별도 설명 계층이며
`evidence_used_for_translation=false`로 표시합니다. 실험용 lexicon/example/grammar/full
RAG도 구현되어 있지만, 최종 번역 모델의 기본 경로는 no-RAG입니다.

## 파일 구조

```text
llm/
├── core/               # 설정, 데이터 로딩, RAG, 지표, 모델 경로
├── data_processing/    # 원본 추출, 표기 분리, 그룹 균형, 문법 청킹
├── training/           # SFT / LoRA + lm_head 학습
├── inference/          # 추론, 모델 라우팅, 문법 설명
├── evaluation/         # 충돌·다중 정답·bootstrap·디코딩 평가
├── scripts/            # 로컬 ZIP의 선택적 자원 복원
├── tests/              # 원본 안전, 누수, 표기, 경로 회귀 테스트
├── configs/            # 최종 모델의 이식 가능한 설정 기록
├── reports/            # 텍스트/영상 없는 집계 실험 결과
├── docs/               # 데이터, 실행, 실험 기록
├── examples/           # 자체 작성한 스키마 fixture
├── requirements.txt
├── requirements-dev.txt
├── requirements-dense.txt
└── pytest.ini
```

로컬에서만 생성하는 경로는 `base_model/`, `outputs/`, `training_data/`,
`resources/`, `data/`이며 [.gitignore](.gitignore)로 제외합니다.

## 원본 데이터와 전처리

| 원천 | 입력 행 | 중복 제거 후 | 표기/비고 |
|---|---:|---:|---|
| NIKL 2024 · 한국수어-한국어 병렬 말뭉치 | 956 | 896 | NIKL 번호형, processed 메타데이터 |
| NIKL 2025 · 한국수어-한국어 병렬 말뭉치 | 23,534 | 23,534 | NIKL 번호형, 원시 JSON도 지원 |
| 재난안전수어영상 | 27 | 27 | processed_gloss 메타데이터 |
| GKSL3k | 3,052 | 2,570 | KETI/NIA 출처, 번호 없음 |
| **합계** | **27,569** | **27,027** | 완전 중복 542행 제거 |

NIKL 자료의 출처는 [국립국어원 모두의 말뭉치](https://kli.korean.go.kr/corpus/)의 **「한국수어-한국어 병렬 말뭉치」**입니다.
2024·2025 자료의 정확한 배포판은 제공기관에서 확인해 확보합니다.
코드·과거 집계에 남아 있는 `NIKL_Korean_Sign_Language_Annotated_Corpus_2024` 등의 이름은 내부 식별자이며, 출처 표기는 위 이름으로 통일합니다. GKSL3k와 재난안전 자료의 상세 배포 식별자 및
문법서 서지정보는 기존 패키지에 완전히 기록되어 있지 않아
[데이터 문서](docs/DATA.md)에 확인 필요 사항으로 표시했습니다.

NFKC/공백 정규화 → 한국어·gloss 쌍 중복 제거 → 문장 그룹 해시 split(seed 42) →
번호형/번호 없는 표기 분리 → train 과대표집 제한 순서로 처리했습니다.
같은 문장은 항상 같은 split에 배정합니다. 생성 라벨은 최종 데이터에서 제외했습니다.

| 데이터 버전 | Train | Valid | Test |
|---|---:|---:|---:|
| 통합 human v2 | 21,649 | 2,576 | 2,802 |
| Group balanced v4 | 21,037 | 2,576 | 2,802 |
| v4 NIKL 번호형 subset | 19,559 | 2,336 | 2,557 |
| v4 GKSL subset | 1,478 | 240 | 245 |

동일 문장·상이 gloss는 273그룹입니다. 복수 정답을 보존하면서 train 그룹당 최대
3개 참조로 제한했고, valid/test는 유지했습니다. split 문장 그룹 교집합은 0입니다.
**원본을 변경하지 않는 텍스트 준비 코드**이며 모든 원천 형식을 자동 해석하는 범용
수어 전처리기는 아닙니다. 지원 스키마는 [DATA.md](docs/DATA.md)를 참고하세요.

## 설치와 테스트

Python 3.10 이상을 사용하며 기존 실험 기록은 Python 3.12 기반입니다.
학습에는 CUDA GPU를 권장합니다. PyTorch는 장비에 맞는 빌드를 먼저 준비하세요.

```bash
cd adv-pj-a/llm
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Dense retrieval 실험만 `requirements-dense.txt`를 추가 설치합니다.
CPU 데이터/검색 테스트는 `requirements-dev.txt`만으로 실행할 수 있으며 모델을 다운로드하지 않습니다.

## 최종 모델을 로컬에서 사용하기

아래 명령의 입력 문장은 문서 설명용으로 직접 만든 합성 예시이며 원본 말뭉치에서 인용하지 않았습니다. 원본 문장·Gloss 쌍을 예시로 공개하지 않습니다.

최종 가중치는 이 `llm/` 배포본에 없습니다. 기존 `llm최종.zip`을 가진 경우 필요한 자원만
로컬에 복원할 수 있습니다. 복원 도구는 기존 파일을 덮어쓰지 않습니다.

```bash
python -m scripts.restore_local_assets \
  --archive "/path/to/llm최종.zip" --dry-run

# 어댑터 2개 + 토크나이저 + 실행 설정만 복원
python -m scripts.restore_local_assets \
  --archive "/path/to/llm최종.zip"
```

복원된 설정의 과거 서버 절대 경로는 현재 프로젝트 경로로 변환합니다.
base 모델은 기본적으로 Hugging Face의 `paust/pko-t5-base`에서 읽습니다.
완전한 로컬 base, 학습 CSV, 기존 문법 청크도 필요하면 **첫 복원 시** 아래처럼 지정합니다.

```bash
python -m scripts.restore_local_assets \
  --archive "/path/to/llm최종.zip" \
  --include-base-model --include-training-data --include-grammar
```

CSV 없이 모델의 원시 출력만 확인:

```bash
python -m inference.run --corpus nikl --repeat_safe \
  --no-explain --no-normalize_convention \
  --text "파란 우산을 들고 도서관 입구에서 만나자."
```

실험 당시처럼 human train 기반 번호 보정과 문법 설명까지 확인:

```bash
python -m inference.run --corpus nikl --repeat_safe \
  --train_file training_data/ksl_unified_group_balanced_v4_subsets/nikl_numbered/train.csv \
  --grammar_chunks resources/grammar_chunks.jsonl \
  --text "파란 우산을 들고 도서관 입구에서 만나자."
```

`--corpus gksl`은 번호 없는 모델을 선택합니다. 별도 모델 경로는 `--model_dir`,
base 위치는 `--base_model`로 지정합니다.
번호 보정은 human train에서 최소 5회 관측되고 95% 이상 동일 번호였던 대응만 사용합니다.
JSON에는 `raw_gloss`, `convention_normalization`, 문법 자질 및 page/chunk 근거를 남깁니다.
문법 근거는 특정 어휘의 gloss 번호를 증명하지 않습니다.

## 데이터 준비와 학습

```bash
python -m data_processing.prepare_ksl_text_data \
  --input "/data/nikl2024/processed_2024" \
  --input "/data/nikl2025/raw_or_processed" \
  --input "/data/재난안전수어영상.zip" \
  --input "/data/GKSL3k_original.csv" \
  --exclude-generated \
  --output training_data/ksl_unified_raw_human_v2

python -m data_processing.prepare_convention_splits \
  --source_dir training_data/ksl_unified_raw_human_v2 \
  --output_dir training_data/ksl_unified_raw_human_v2_by_convention
```

생성 파일은 `train.csv`, `valid.csv`, `test.csv`, `all_text_gloss.csv`,
`conflicts.json`, `prepare_report.json`입니다.
입력은 여러 번 지정할 수 있으며 출력은 입력 폴더 밖이어야 합니다.
Group balanced v4 생성, 암기 진단, 학습·재개 및 평가 명령은 [실행 가이드](docs/WORKFLOW.md)에 있습니다.

새 no-RAG LoRA baseline 예시:

```bash
DATA_DIR="training_data/ksl_unified_raw_human_v2_by_convention/nikl_numbered"
python -m training.train \
  --model_name paust/pko-t5-base \
  --train_file "$DATA_DIR/train.csv" \
  --valid_file "$DATA_DIR/valid.csv" --test_file "$DATA_DIR/test.csv" \
  --input_tag gloss_convention --rag_mode none \
  --use_lora --lora_train_lm_head \
  --lora_target_modules q,k,v,o,wi_0,wi_1,wo \
  --learning_rate 1e-4 --lm_head_learning_rate 1e-5 \
  --max_target_length 128 --num_train_epochs 8 \
  --output_dir outputs/new_nikl_baseline
```

이 명령은 새 baseline 학습 예시입니다. 최종 v4는 이전 실험 어댑터에 이어서 학습한
계보가 있어 이 명령만으로 동일 가중치를 재생성한다고 주장하지 않습니다.
최종 실행 설정은 [configs](configs/), 학습 계보 한계는 [EXPERIMENTS.md](docs/EXPERIMENTS.md)에 기록했습니다.

## 기록된 성능

다음은 원본 ZIP의 집계 보고서를 옮긴 수치입니다. 이 코드 정리 작업에서 재학습하지 않았습니다.
NIKL 표는 **repeat-safe 디코딩**, GKSL은 **고유 입력 다중 정답 평가**입니다.
F1은 순서를 무시한 gloss token F1이며 번역 정확도 자체와 같지 않습니다.

| 모델/평가 | 표본 | WER ↓ | Exact Match ↑ | Token F1 ↑ |
|---|---:|---:|---:|---:|
| NIKL v4 혼합 test | 700 | 0.6413 | 0.0143 | 0.4903 |
| NIKL v2 동일 비교 집합 | 700 | 0.6485 | 0.0086 | 0.4852 |
| NIKL v4 전체 valid | 2,336 | 0.6206 | 0.0081 | 0.5086 |
| NIKL v4 2024 test | 90 | 0.8462 | 0.0556 | 0.3071 |
| GKSL 최종 다중 정답 | 142 | 0.4602 | 0.3028 | 0.6589 |

repeat-safe는 beam 4, length penalty 1.0, repetition penalty 1.03, 3-gram 반복 차단입니다.
700건 NIKL 비교에서 빈 출력과 연속 3회 gloss 반복은 0건입니다.
기존 ZIP의 루트 `test_metrics_by_corpus.json`은 제약 없는 디코딩 결과이므로
위 repeat-safe 성능과 직접 섞어 해석하지 않습니다.

고유 그룹 695개의 paired bootstrap WER 차이(v4−v2) 95% CI는
`[-0.013205, -0.001457]`입니다. Token F1 개선 CI는 0을 포함합니다.
전체 NIKL test는 2,557행이며 위 700건은 층화 부분집합입니다.
실험 과정에서 test를 반복 비교해 모델 선택에도 참고했으므로 완전히 미사용된 최종
독립 평가로 간주할 수 없습니다. 상세 범위는 [reports/README.md](reports/README.md)에 있습니다.

## v4 이후 추가 개선 (LLM2 라인)

위 "기록된 성능"까지가 기존 배포본의 집계입니다. 이 절은 그 **v4 기준선에서
출발해 실제로 재학습·재측정한 추가 개선 라인**을 기록합니다. 이 라인은 위
"실험에서 배운 점과 한계"에 나오는 기존 v5/v6/v7 **기각 후보와는 다른 별개의
실험**이며, 혼동을 피하기 위해 Step 1~4로 표기합니다. 모든 수치는 같은 NIKL
일반 700문장 다중 정답 벤치마크(고유 입력 695: NIKL 2025 608 / 2024 85 /
재난안전 2)에서 **repeat-safe 디코딩**으로 측정했습니다. 집계 JSON은
[reports/llm2_progression_v4_to_v8.json](reports/llm2_progression_v4_to_v8.json)에
있고, 가중치·데이터·예측 파일은 공개 정책에 따라 포함하지 않습니다.

| 단계 | 핵심 변경 | Token F1 ↑ | WER ↓ | Exact ↑ | 2024 F1 | 2025 F1 |
|---|---|---:|---:|---:|---:|---:|
| v4 (기준선) | NIKL 2024(706행)만 학습 | 0.4934 | 0.6400 | 0.0144 | 0.3181 | 0.5185 |
| Step 1 | 2025 포함 통합 재학습 (LoRA r16, 2ep) | 0.5384 | 0.6045 | 0.0144 | 0.2636 | 0.5776 |
| Step 2 | RSLoRA(r32,a64) + 번호 재교정 (1ep) | 0.5649 | 0.5800 | 0.0158 | 0.2862 | 0.6047 |
| Step 3 | 2024 메타 괄호 전처리 + sqrt 균형 (1ep) | 0.5778 | 0.5731 | 0.0230 | 0.3710 | 0.6071 |
| **Step 4 (최종)** | 풀 학습 3ep + dropout 0.1 / LR 2e-4 | **0.6162** | **0.5391** | **0.0288** | **0.3920** | **0.6478** |

**v4 → 최종 개선폭**: Token F1 +0.123, WER −0.101, Exact Match 약 2배,
2024 F1 +0.074, 2025 F1 +0.129.

각 단계의 근거:

- **Step 1 — 데이터 분포를 맞춤.** v4는 NIKL 2024만 학습했지만 벤치마크의 약
  87%가 NIKL 2025입니다. 2025를 포함한 통합 데이터로 재학습해 전체 F1과 2025
  F1이 올랐습니다. 단 2024는 전용 학습 효과를 잃어 하락했습니다.
- **Step 2 — LoRA 변형을 비교해 RSLoRA 채택.** 동일 데이터·1 epoch·valid에서
  r16 / r32 / RSLoRA / DoRA를 비교했습니다. 단순 rank 상향과 DoRA는 이득이
  없었고 RSLoRA만 개선했습니다
  ([reports/llm2_lora_variant_experiments.json](reports/llm2_lora_variant_experiments.json)).
  디코딩 그리드 탐색은 개선 폭이 측정 노이즈 수준이었습니다.
- **Step 3 — 2024 약점을 전처리로 보완.** 2024 입력의 약 2/3가 `(오른손의)`,
  `('제비' 수형)` 같은 수형·화자 메타 주석을 포함해 입력 노이즈가 됐습니다.
  이 메타 괄호를 정리(gloss는 불변, train의 2024 행만, 원본은 읽기 전용)하고
  sqrt 코퍼스 균형을 더해 2024 F1을 0.29대에서 0.37로 회복했습니다. 전처리
  코드는 [data_processing/clean_nikl2024_meta.py](data_processing/clean_nikl2024_meta.py)
  입니다.
- **Step 4 — 과적합을 제어하며 풀 학습.** 1 epoch에 수렴 조짐이 보여 단순 epoch
  증가는 위험했습니다. dropout 0.05→0.1, LoRA LR 3e-4→2e-4로 낮추고 3 epoch +
  early stopping + best-model 복원을 적용해 과적합 없이 전 지표 최고에
  도달했습니다.

이 라인은 LoRA·RSLoRA·lm_head 학습만 사용하고 번역은 계속 no-RAG입니다.
수치는 bag-of-gloss token F1과 WER이며 전문가 검수 수어 정확도를 직접 주장하지
않습니다. 벤치마크 700건은 실험 중 반복 참조한 집합이므로 완전히 미사용된 독립
test로 간주하지 않습니다.

### 문법 설명 계층 개선

번역과 분리된 문법 설명 계층(`--explain`,
`evidence_used_for_translation=false`)도 함께 다듬었습니다. 입력의 문법 자질
감지 정규식을 보강해 라벨 문장 기준 자질 감지 recall이 0.67에서 0.90으로, 근거
검색 recall이 0.90에서 1.0으로 올랐고, `-세요?` 의문형을 명령으로 오분류하던
문제를 고쳤습니다. 설명에는 문법서 항목·페이지뿐 아니라 그 항목이 다루는 내용
요약을 함께 제공합니다. 검증 스크립트는
[evaluation/validate_explanation.py](evaluation/validate_explanation.py)이며
판정 권한은 전문가 검수가 아닌 `heuristic_proxy_not_expert`로 고정했습니다
([reports/llm2_explanation_validation.json](reports/llm2_explanation_validation.json)).

> 예시 (최종 모델, 합성 입력):
> 입력 `친구가 같이 가자고 말했어요.` → gloss `친구1 함께1 가다1 말하다1`,
> 자질 `명령청유, 인용`, 근거 `3.4. 청유문 (p.175)` — "청유문은 화자와 청자가
> 함께 하도록 요청하는 문장 유형… 시선응시·고개 숙이기 등 비수지표지로 실현".

## 실험에서 배운 점과 한계

- 초기 KE-T5는 `지시1`, `먹다1`, `동점을` 반복 등 출력 붕괴를 보였습니다.
- 16건 암기는 성공했지만 전체 데이터의 일반화 성공을 뜻하지 않았습니다.
- v3 자동 검수, GKSL v4, 문법 초점 v5, v6/v7 후보는 일관된 개선이 없어 기각했습니다.
- Grammar RAG는 WER를 낮췄지만 F1/Exact를 낮춰 번역 입력에서 제외했습니다.
- NIKL Exact가 낮고 2025 데이터 비중이 큽니다. 재난안전 test는 2건뿐입니다.
- 충돌·문법 검토는 AI/규칙 보조이며 한국수어 전문가 검수가 필요합니다.
- 문법서의 정확한 서지정보와 데이터별 재배포 권한은 별도 확인이 필요합니다.
- 영상·비수지표지·공간 표현과 서비스 API는 이 코드의 평가 범위 밖입니다.

개발 과정과 기각한 후보는 [실험 기록](docs/EXPERIMENTS.md)을 참고하세요.
원천 코드 계보와 외부 자료는 [SOURCE.md](SOURCE.md)에 정리했습니다.
