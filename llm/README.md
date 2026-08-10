# 한국어 → 한국수어(KSL) 글로스 변환

`KETI-AIR/ke-t5-small` 을 한국어 문장 → KSL 글로스 시퀀스 seq2seq 태스크로
파인튜닝하는 뼈대 코드. **전체 파인튜닝(SFT)** 과 **LoRA** 두 경로를 모두 지원하고,
한국수어 문법서를 RAG 컨텍스트로 주입한다.

```
입력 : 공부를 더 많이 해 봐야겠다.
출력 : 공부1 열심2 노력1
```

---

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `prepare_grammar.py` | 문법서 PDF 추출 txt → RAG 청크(JSONL). 머리말/꼬리말/각주 제거, 장·절 메타데이터 추출 |
| `retriever.py` | 글로스 사전 / 유사 예시 / 문법서 검색기 + 문법자질 추출기 |
| `dataset.py` | 로딩·분할·RAG 증강·토크나이즈 (증강 결과는 parquet 캐시) |
| `metrics.py` | WER, Exact Match, token-F1, BLEU/chrF |
| `train.py` | SFT / LoRA 학습 |
| `inference.py` | 추론 (LoRA 어댑터 병합 포함) |
| `config.py` | 하이퍼파라미터 기본값 |

---

## 2. 실행 순서

```bash
pip install -r requirements.txt

# (1) 문법서 청킹 — 최초 1회
python prepare_grammar.py \
    --input /path/to/한국수어_문법_텍스트.txt \
    --output data/grammar_chunks.jsonl
# -> 814 chunks, 평균 212자

# (2) 학습 데이터 배치
#     data/train.tsv : file_id \t korean_text \t gloss_sequence

# (3-a) 전체 파인튜닝  ← 먼저 이걸 베이스라인으로
python train.py --train_file data/train.csv --rag_mode full --bf16

# (3-b) LoRA 파인튜닝
python train.py --train_file data/train.csv --rag_mode full --bf16 \
       --use_lora --lora_r 16 --lora_alpha 32 \
       --lora_target_modules q,k,v,o --learning_rate 1e-3

# (4) 추론
python inference.py --model_dir outputs/ket5-small-ksl \
       --train_file data/train.csv \
       --text "공부를 더 많이 해 봐야겠다." --show_context
```

---

## 3. RAG 설계

문법서만으로 RAG를 붙이면 잘 안 된다. 문법서는 "일치동사", "역할전환" 같은
**메타언어 서술**이라 입력 문장과 어휘가 겹치지 않기 때문이다.
그래서 검색 소스를 세 갈래로 나눴다.

| 소스 | 내용 | 효과 |
|---|---|---|
| **사전** | 학습셋에서 자동 구축한 `어간 → 글로스` 매핑 | 가장 강함. 어휘 선택 오류를 직접 줄인다 |
| **예시** | 입력과 유사한 학습 예시 `문장 => 글로스` | 강함. 어순 패턴을 모방하게 한다 |
| **문법** | 문법서 청크 + 문법자질 태그 | 보조. 부정/의문/조건 등 구문 처리에 기여 |

증강된 입력은 이런 형태다:

```
한국어를 한국수어 글로스로 변환: 나는 그 영화를 보지 않았다.
 | 사전: 영화=영화1
 | 예시: 그는 아직 오지 않았다. => 그1 아직1 오다1 아니다1
 | 자질: 부정 시제
 | 문법: (부정법) ... [나] [아들] {휴지} [얼굴] [예쁘다] [없다] (상태 부정어 사용) ...
```

### 문법서 검색은 자질 기반으로

문장을 그대로 질의하면 "학교"가 들어있다는 이유로 수위(手位) 설명 청크가
딸려온다. 그래서 `GrammarFeatureExtractor` 가 문장에서 문법 현상
(부정·의문·조건·시제·인용·수량·비교 등)을 먼저 감지하고, 그 **용어**로 문법서를
검색한다. `--grammar_query_mode` 로 `surface / feature / hybrid` 전환 가능하며
기본값은 `feature`.

실제 차이:

| 입력 | surface 검색 결과 | feature 검색 결과 |
|---|---|---|
| 내일 학교에 가야 한다. | 학교 규모 강조 마우딩 설명 (무관) | 제3장 시제 — 시간선 설명 (적중) |
| 나는 그 영화를 보지 않았다. | — | 제4장 부정법 — 부정표지 예문 (적중) |

---

## 4. 본 학습 전 검증 (필수)

```bash
python smoke_test.py --train_file data/train.tsv --n 32 --bf16
python smoke_test.py --train_file data/train.tsv --n 32 --bf16 --use_lora
```

6단계를 순서대로 검사하고 `[PASS]/[WARN]/[FAIL]` 을 찍는다.

| 단계 | 검사 | 통과 기준 |
|---|---|---|
| 0 | 환경 | sentencepiece 설치, bf16 지원 여부 |
| 1 | 데이터/RAG 증강 | 건수, 글로스 길이 분포, 증강 입력 육안 확인 |
| 2 | 누수 | 정답 글로스가 컨텍스트에 있는 비율 < 2%, split 교집합 0 |
| 3 | 토크나이저 | 글로스 왕복 복원, **입력 잘림률 < 10%, 타깃 잘림률 = 0%** |
| 4 | 순전파/역전파 | loss 유한, gradient norm ≠ 0, 학습 파라미터 비율 |
| 5 | **과적합** | train loss < 0.05, exact_match > 0.9, WER < 0.05 |

5단계가 이 스크립트의 핵심이다. **32건을 300스텝 학습시켜 그 32건을 그대로
외우지 못하면 학습 배선에 버그가 있는 것**이다. 데이터가 적어서가 아니다.

### 과적합이 안 될 때 원인별 증상

| 증상 | 원인 | 조치 |
|---|---|---|
| loss가 NaN | fp16 사용 | `--bf16` 또는 fp32 |
| loss가 8~10에서 정체 | 학습률 과소 | SFT 3e-4, LoRA 1e-3 |
| loss는 떨어지는데 EM=0 | 생성 설정 문제 | `predict_with_generate`, `max_target_length` 확인 |
| LoRA만 안 떨어짐 | target_modules 오류 | `--print_module_names` 로 실제 이름 확인 |
| 예측이 빈 문자열 | 타깃 잘림 / 라벨 마스킹 | 3단계 타깃 잘림률 확인 |
| 예측이 입력을 그대로 반복 | 프리픽스·라벨 뒤바뀜 | 1단계 증강 예시 육안 확인 |

통과했다면 전체 데이터로 넘어간다.

---

## 5. RAG ablation

RAG가 항상 도움이 되지는 않는다. ke-t5-small은 60M 규모라 컨텍스트가 길어지면
오히려 손해를 볼 수 있다. **`none` 대비 이득이 있는지 반드시 측정**할 것.

```bash
for m in none lexicon example grammar full; do
  python train.py --train_file data/train.tsv --rag_mode $m \
                  --output_dir outputs/rag_$m
done
```

경험적으로 예상되는 순서는 `lexicon+example > full > lexicon > none > grammar` 지만,
데이터 규모와 글로스 체계에 따라 뒤집힌다. 결과표를 만들어 두고 진행하는 게 좋다.

---

## 6. SFT vs LoRA — 어느 쪽을 쓸 것인가(테스트용 기준이라 일단 무시)

| | 전체 파인튜닝 | LoRA |
|---|---|---|
| 학습 파라미터 | ~60M (100%) | ~0.3~1M (0.5~1.5%) |
| 권장 lr | 3e-4 | 1e-3 (10~30배 높게) |
| 이 태스크에서 | **보통 더 좋음** | 약간 열세, 대신 실험 회전이 빠름 |

솔직하게 말하면, **ke-t5-small 크기에서는 LoRA의 실익이 크지 않다.**
LoRA는 원래 수 B 파라미터 모델의 메모리/저장 비용을 줄이려는 기법인데,
60M 모델은 전체 파인튜닝해도 GPU 메모리가 몇 GB면 충분하다.
LoRA를 쓸 만한 경우는 다음 정도다.

- 여러 도메인(뉴스/일상/교육)별 어댑터를 갈아끼우며 서빙하고 싶을 때
- 학습 데이터가 매우 적어(< 2천 건) 전체 파인튜닝이 과적합할 때
- 나중에 `ke-t5-large` 나 다른 대형 백본으로 스케일업할 계획일 때 (코드 재사용)

LoRA 대상 모듈은 `--lora_target_modules` 로 조절한다.
- `q,v` — 논문 기본, 파라미터 최소
- `q,k,v,o` — 기본값, attention 전체 (권장)
- `q,k,v,o,wi,wo` — FFN 포함, 표현력 최대

모듈 이름이 확실치 않으면 `--print_module_names` 로 실제 이름을 먼저 확인할 것.

---

## 7. 함정 목록 (직접 밟기 전에 읽을 것)

1. **fp16 금지.** T5 계열은 fp16에서 overflow로 loss가 NaN이 된다.
   `--bf16` (Ampere 이상) 또는 fp32를 쓸 것. 코드에서 `fp16=False`로 강제해 뒀다.

2. **예시 검색 누수.** 수어 코퍼스는 같은 문장을 여러 화자가 수행한 레코드가
   흔하다. 정답 글로스가 그대로 컨텍스트에 실리면 valid 점수가 비현실적으로
   높게 나온다. `ExampleRetriever` 에서 (a) 공백·문장부호 정규화 후 동일 문자열
   제외, (b) 유사도 `max_example_score`(기본 0.95) 이상 제외 — 두 겹으로 막아 뒀다.

3. **분할 누수.** 같은 원문이 train과 valid에 흩어지지 않도록
   `group_split_by_text=True` 로 문장 해시 기준 그룹 분할을 한다.

4. **학습률.** T5는 `3e-5` 같은 BERT식 lr로는 거의 학습이 안 된다.
   전체 파인튜닝 `3e-4`, LoRA `1e-3` 부터 시작할 것.

5. **평가 지표.** 글로스 시퀀스는 평균 3~8토큰이라 BLEU가 불안정하다.
   **WER(주 지표) + Exact Match** 를 보고, token-F1은 "어휘는 맞는데 어순이
   틀린" 경우를 분리해 보는 데 쓴다. (WER는 높고 token-F1은 높으면 어순 문제)

6. **글로스 토큰 추가는 신중히.** `--add_gloss_tokens` 는 `공부1` 을 한 토큰으로
   만들지만 새 임베딩이 랜덤 초기화된다. 데이터가 적으면 오히려 손해다.
   LoRA와 함께 쓰면 `modules_to_save=["shared","lm_head"]` 가 자동으로 붙어
   임베딩까지 학습한다(파라미터 수 급증에 주의).

7. **max_source_length.** RAG 컨텍스트가 붙으면 입력이 길어진다. 기본 384로 뒀는데,
   토큰 길이 분포를 실제로 찍어 보고 조정할 것. 잘리면 문법 청크부터 날아가도록
   순서(문장 → 사전 → 예시 → 자질 → 문법)를 배치해 뒀다.

8. **추론 시 컨텍스트 불일치.** 학습과 추론의 RAG 조립 방식이 다르면 성능이
   급락한다. `inference.py` 가 `run_config.json` 을 읽어 학습 때 설정을 그대로
   복원하도록 해 뒀다.

---

## 8. 확장 아이디어

- **형태소 분석기 연동**: `GlossLexicon._stems()` 를 kiwi/konlpy 어간 추출로 교체하면
  사전 커버리지가 크게 오른다. 현재는 문자열 포함 관계만 본다.
- **비수지 요소**: 문법서 4장이 다루는 `{휴지}`, `{눈썹올리기}` 같은 비수지표지를
  타깃에 포함할지 결정 필요. 포함한다면 별도 태그 세트로 관리할 것.
- **백본 교체**: `--model_name KETI-AIR/ke-t5-base` 로 바로 비교 가능.
  small에서 RAG 이득이 미미해도 base에서는 살아나는 경우가 있다.
- **제약 디코딩**: 학습셋 글로스 어휘 밖의 토큰을 생성하지 못하도록
  `prefix_allowed_tokens_fn` 을 걸면 OOV 글로스를 없앨 수 있다.
