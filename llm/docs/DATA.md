# 데이터 출처와 스키마

## 확보한 자료

- 국립국어원 한국수어 주석 말뭉치 2024: 초기 파일명 `NIKL_Korean_Sign_Language_Annotated_Corpus_2024_v1.1.zip`
- 국립국어원 한국어-한국수어 병렬 말뭉치 2025: 초기 파일명 `NIKL_Sign_Language_Parallel_Corpus_2025_v1.0_da9.zip`, 이후 추가 2025 원시 JSON 포함
- 재난안전수어영상: 프로젝트에서 준비한 `processed_gloss/samples.jsonl`
- GKSL3k: `GKSL3k_original.csv`, dataset 열에 KETI-Emergency/NIA-2020/KETI-Airport/KETI-Daily 보존

NIKL 배포 자료는 [모두의 말뭉치](https://kli.korean.go.kr/corpus/)에서 확인합니다.
GKSL3k CSV의 원 배포 URL·버전, 재난안전의 정확한 배포 ID·이용조건은 ZIP에 충분히
기록되어 있지 않습니다. NIA/KETI라는 corpus 이름만으로 특정 배포판/라이선스를 추정하지 않습니다.
외부 데이터를 공유할 때 제공기관의 이용조건을 확인하세요.

## 지원하는 입력

| 입력 | 처리 |
|---|---|
| processed_2024/2025 | manifests/korean_to_gloss.jsonl, samples.jsonl/csv 등 선택 |
| NIKL 2025 원시 JSON 폴더 | koreanText + strong gloss_id 시간순, strong이 비면 weak 사용 |
| 재난안전 ZIP | 내부 processed_gloss/samples.jsonl 읽기, 전체 압축 해제 없음 |
| 재난안전 압축 해제 폴더 | 같은 메타데이터 선택 |
| 초기 5열 dataset.csv | 한국어·gloss·출처만 복사 |
| GKSL3k CSV | WKL 문장/GKSL 문장/dataset 열 해석 |

NIKL 2024 원시 JSON/EAF를 모두 해석하는 프로그램과 MediaPipe 추출기는 이 코드 배포본에
포함하지 않았습니다. 2024는 앞서 생성한 processed 메타데이터를 입력합니다.
재난안전의 keypoint 원본 ZIP만 있으면 processed_gloss를 먼저 준비해야 합니다.

## 출력 열

```text
file_id,korean_text,gloss_sequence,source_corpus,gloss_convention,
source_file,source_record_id,label_type,split_group
```

- `source_file`은 로컬 provenance이며 ZIP 내부 경로도 기록합니다.
- `human_annotation`은 가져온 자료의 라벨 유형입니다. 전문가가 이 프로젝트에서
  재검증했다는 의미는 아닙니다.
- `nikl_numbered`와 `gksl_unnumbered`를 분리합니다. 번호와 접미사를 임의 삭제하지 않습니다.
- NFKC 정규화는 Unicode 호환 문자를 정리하는 것이며 수어 표현의 의미 오류를 교정하지 않습니다.

## split과 충돌

seed 42, valid/test ratio 각각 0.1로 문장 그룹 해시를 사용합니다.
동일 정규화 문장의 다양한 gloss는 같은 split에 들어갑니다.
완전 중복 쌍은 제거하되 서로 다른 gloss는 충돌 목록에 보존합니다.
최종 인간 주석 v2는 27,027행, 그룹 간 교집합 0, 충돌 273그룹입니다.

v4에서는 train 과대표집 그룹을 최대 3개 참조로 제한해 21,037행으로 줄였습니다.
valid/test core는 바꾸지 않았습니다. 변이 판정은 자동/AI 보조이며 전문가 정답이 아닙니다.
통합 v2와 v4, convention별 dataset은 각기 다른 버전이므로 train 수를 섞지 않습니다.
최종 GKSL 모델은 v2 convention subset을 사용했으며 v4 GKSL 재학습 후보는 기각됐습니다.

## Git에 포함하지 않는 자료

원본·가공 CSV/JSONL, 영상, skeleton, keypoint, EAF, 문법서 원문/청크,
예측별 TSV, base/adapter 가중치, tokenizer 사본, cache를 제외했습니다.
`examples/`에는 자체 작성한 스키마 예시만 있습니다.
집계 통계와 평가 수치는 `reports/`에 보존합니다.

## 문법 자료

기존 실험의 `grammar_chunks.jsonl`은 814청크이며 다음 필드를 사용합니다.

```text
chunk_id, chapter, section, heading, page, text, n_chars
```

정확한 문법서 판본·서지·원문 라이선스는 ZIP에서 확정할 수 없어 확인이 필요합니다.
권한이 있는 로컬 텍스트에서 `python -m data_processing.prepare_grammar --help`로
청킹하거나, 기존 ZIP에서 `--include-grammar`로 로컬에 복원합니다.
문법 RAG에 페이지/청크 근거가 있어도 개별 gloss 번호의 사전 대응을 증명하는 것은 아닙니다.
