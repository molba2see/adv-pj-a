"""
config.py — 하이퍼파라미터 및 경로 설정
argparse로 모두 덮어쓸 수 있다 (train.py 참고).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    train_file: Path = Path("data/train.tsv")      # file_id / korean_text / gloss_sequence
    valid_file: Path | None = None                  # None이면 train에서 분할
    test_file: Path | None = None
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42

    text_col: str = "korean_text"
    gloss_col: str = "gloss_sequence"
    id_col: str = "file_id"

    # 같은 문장이 여러 file_id로 중복 등장하는 경우가 많음 -> 분할 전에 문장 기준 중복 제거
    dedup_by_text: bool = True
    # 문장 단위(그룹) 분할: 같은 원문이 train/valid에 동시에 들어가는 누수 방지
    group_split_by_text: bool = True


@dataclass
class RagConfig:
    grammar_chunks: Path = Path("data/grammar_chunks.jsonl")

    # none | lexicon | example | grammar | full
    # 반드시 none 과 비교(ablation)해 보고 쓸 것. 작은 모델에서는 노이즈가 손해일 수 있다.
    mode: str = "full"

    # --- 검색기 ---
    # "tfidf"  : sklearn char n-gram TF-IDF (다운로드 불필요, 강력한 베이스라인)
    # "dense"  : sentence-transformers 문장 임베딩 + (faiss 있으면 faiss, 없으면 numpy)
    retriever: str = "tfidf"
    dense_model: str = "jhgan/ko-sroberta-multitask"
    tfidf_ngram: tuple = (2, 4)

    top_k_grammar: int = 1        # 문법 규칙 청크 개수
    top_k_example: int = 2        # 유사 학습 예시 개수
    max_lexicon_items: int = 12   # 사전 힌트 최대 개수

    grammar_char_limit: int = 220  # 청크를 이 길이로 잘라 넣는다 (입력 예산 보호)
    min_score: float = 0.10        # 이 점수 미만이면 검색 결과를 버린다

    # ★ 학습 시 근사 중복 예시가 정답을 유출하는 것을 막는 상한.
    #   수어 코퍼스는 같은 문장의 이형 표기가 흔하므로 반드시 켜 둘 것.
    max_example_score: float = 0.95

    # 문법서 질의 방식: surface | feature | hybrid  (feature 권장)
    grammar_query_mode: str = "feature"
    include_feature_tags: bool = True   # "부정 의문" 같은 자질 태그를 입력에 넣을지

    # 프롬프트 마커
    task_prefix: str = "한국어를 한국수어 글로스로 변환: "
    lex_marker: str = " | 사전: "
    ex_marker: str = " | 예시: "
    feat_marker: str = " | 자질: "
    gram_marker: str = " | 문법: "


@dataclass
class ModelConfig:
    model_name: str = "KETI-AIR/ke-t5-small"
    max_source_length: int = 384   # RAG 컨텍스트 포함 길이
    # 128로 둔다. 64에서는 실제 데이터의 일부(최대 77토큰)가 잘려 정답이 손실됐다.
    max_target_length: int = 128
    num_beams: int = 4

    # 빈도 높은 글로스를 토크나이저 special token으로 추가할지 여부.
    # 장점: 글로스가 한 토큰으로 표현됨 / 단점: 임베딩이 랜덤 초기화되어 데이터가 적으면 손해
    add_gloss_tokens: bool = False
    gloss_token_min_freq: int = 20


@dataclass
class TrainConfig:
    output_dir: Path = Path("outputs/ket5-small-ksl")
    num_train_epochs: float = 30.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4        # T5 계열은 1e-4 ~ 5e-4가 적당 (3e-5는 너무 낮다)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing_factor: float = 0.0
    lr_scheduler_type: str = "linear"

    # ★ T5는 fp16에서 NaN/inf가 잘 난다. bf16(Ampere 이상) 또는 fp32를 쓸 것.
    bf16: bool = False
    fp16: bool = False

    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    save_total_limit: int = 2
    metric_for_best_model: str = "eval_gloss_wer"
    greater_is_better: bool = False
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
