"""Create clean Korean->KSL-gloss train/valid/test tables without modifying sources.

Supported inputs
----------------
* A processed corpus directory containing ``samples.jsonl`` / ``samples.csv``.
* An extracted archive containing ``processed_gloss/samples.jsonl``.
* A dataset-builder directory containing ``manifests/korean_to_gloss.jsonl``.
* A CSV/TSV/JSONL file in either of the schemas above.

Only text, gloss and provenance fields are copied. Video and skeleton data are
never opened, copied, moved or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


OUTPUT_FIELDS = (
    "file_id",
    "korean_text",
    "gloss_sequence",
    "source_corpus",
    "gloss_convention",
    "source_file",
    "source_record_id",
    "label_type",
    "split_group",
)

SPACE_RE = re.compile(r"\s+")
TEXT_GROUP_RE = re.compile(r"[\s\.,!?'”“\-~·…]+")
NIKL_2025_CORPUS = "NIKL_Sign_Language_Parallel_Corpus_2025"
GKSL_GLOSS_COLUMN = "Gloss level Korean Sign Language (GKSL) sentence"
GKSL_TEXT_COLUMN = "Word level Korean Language (WKL) sentence"


@dataclass(slots=True)
class TextGlossRow:
    file_id: str
    korean_text: str
    gloss_sequence: str
    source_corpus: str
    source_file: str
    source_record_id: str
    label_type: str = "human_annotation"
    gloss_convention: str = "unknown"

    def split_group(self) -> str:
        normalized = normalize_group_text(self.korean_text)
        return normalized or normalize_space(self.korean_text)

    def output(self) -> dict[str, str]:
        row = asdict(self)
        row["split_group"] = self.split_group()
        return row


def normalize_space(value: Any) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def normalize_group_text(text: str) -> str:
    return TEXT_GROUP_RE.sub("", normalize_space(text)).casefold()


def normalize_gloss(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                token = item.get("gloss") or item.get("gloss_id")
            else:
                token = item
            token = normalize_space(token)
            if token:
                parts.append(token)
        return " ".join(parts)
    return normalize_space(value)


def infer_gloss_convention(
    record: dict[str, Any], corpus: str, gloss: str
) -> str:
    """Return an auditable label for the source corpus' gloss notation.

    NIKL annotations (including the disaster corpus) use numbered lexical IDs,
    while GKSL3k's KETI/NIA-2020 rows intentionally omit those IDs. We keep
    the distinction as metadata and never rewrite the gloss itself.
    """
    explicit = normalize_space(record.get("gloss_convention"))
    if explicit:
        return explicit
    if corpus.startswith("KETI-") or corpus == "NIA-2020":
        return "gksl_unnumbered"
    if "NIKL" in corpus or corpus == "재난안전수어영상":
        return "nikl_numbered"
    # Unknown imports remain explicit rather than being silently assigned to a
    # known convention. A digit-bearing majority is still useful as a safe
    # fallback for small, manually prepared NIKL-style files.
    tokens = gloss.split()
    numbered = sum(bool(re.search(r"\d+(?:_\d+)?(?:[#:+].*)?$", token)) for token in tokens)
    return "numbered" if tokens and numbered / len(tokens) >= 0.5 else "unknown"


def corpus_name(
    record: dict[str, Any],
    path: Path,
    override: str | None = None,
) -> str:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    # GKSL3k distributes its source collection in ``dataset`` rather than a
    # generic corpus field. Preserve it in provenance so gloss conventions can
    # be audited or routed later.
    gksl_source = record.get("dataset") if GKSL_GLOSS_COLUMN in record else None
    return normalize_space(
        override
        or record.get("corpus")
        or gksl_source
        or source.get("format")
        or record.get("source_format")
        or path.parent.name
    )


def label_type(record: dict[str, Any]) -> str:
    label = record.get("label")
    if isinstance(label, dict):
        value = label.get("type")
    else:
        value = record.get("label_type")
    return normalize_space(value or "human_annotation")


def row_from_record(
    record: dict[str, Any],
    path: Path,
    position: int,
    *,
    source_locator: str | None = None,
    corpus_override: str | None = None,
) -> TextGlossRow | None:
    korean = normalize_space(
        record.get("korean_text")
        or record.get("korean")
        or record.get("natural_language_input")
        or record.get("text")
        or record.get(GKSL_TEXT_COLUMN)
    )
    gloss = normalize_gloss(
        record.get("gloss_sequence")
        or record.get("gloss_list")
        or record.get("gloss")
        or record.get(GKSL_GLOSS_COLUMN)
    )
    if not korean or not gloss:
        return None
    record_id = normalize_space(
        record.get("sample_id")
        or record.get("pair_id")
        or record.get("source_record_id")
        or record.get("file_id")
        or record.get("index")
        or position
    )
    corpus = corpus_name(record, path, corpus_override)
    convention = infer_gloss_convention(record, corpus, gloss)
    source_name = source_locator or str(path.resolve())
    # Keep the logical row id identical whether the same corpus is supplied as
    # a ZIP or as an extracted directory. The physical source locator remains
    # available separately in ``source_file``.
    stable = hashlib.sha1(
        f"{corpus}\0{record_id}\0{korean}\0{gloss}".encode("utf-8")
    ).hexdigest()[:16]
    return TextGlossRow(
        file_id=f"{safe_id(corpus)}::{record_id}::{stable}",
        korean_text=korean,
        gloss_sequence=gloss,
        source_corpus=corpus,
        source_file=source_name,
        source_record_id=record_id,
        label_type=label_type(record),
        gloss_convention=convention,
    )


def safe_id(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", value).strip("_")
    return value[:80] or "corpus"


def read_jsonl(path: Path, *, corpus_override: str | None = None) -> list[TextGlossRow]:
    rows: list[TextGlossRow] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for position, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            row = row_from_record(
                record, path, position, corpus_override=corpus_override
            )
            if row is not None:
                rows.append(row)
    return rows


def _zip_member_priority(name: str) -> tuple[int, str]:
    normalized = name.replace("\\", "/").lower()
    if normalized.endswith("processed_gloss/samples.jsonl"):
        return 0, normalized
    if normalized.endswith("processed_gloss/samples.csv"):
        return 1, normalized
    if normalized.endswith("samples.jsonl"):
        return 2, normalized
    if normalized.endswith("samples.csv"):
        return 3, normalized
    return 100, normalized


def choose_zip_member(names: Iterable[str]) -> str:
    candidates = []
    for name in names:
        normalized = name.replace("\\", "/").lower()
        if normalized.endswith("/samples.jsonl") or normalized == "samples.jsonl":
            candidates.append(name)
        elif normalized.endswith("/samples.csv") or normalized == "samples.csv":
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError(
            "ZIP 내부에서 processed_gloss/samples.jsonl 또는 samples.csv를 찾지 못했습니다."
        )
    return sorted(candidates, key=_zip_member_priority)[0]


def read_zip(path: Path) -> tuple[str, list[TextGlossRow]]:
    """Read text/gloss metadata directly from a ZIP without extracting it."""
    rows: list[TextGlossRow] = []
    with zipfile.ZipFile(path, "r") as archive:
        member = choose_zip_member(archive.namelist())
        source_locator = f"{path.resolve()}::{member}"
        # The archive stem is stable provenance and avoids stale absolute paths
        # embedded in preprocessed rows.
        corpus_override = path.stem
        with archive.open(member, "r") as raw:
            text_handle = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            if member.lower().endswith(".jsonl"):
                for position, line in enumerate(text_handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    row = row_from_record(
                        record,
                        path,
                        position,
                        source_locator=source_locator,
                        corpus_override=corpus_override,
                    )
                    if row is not None:
                        rows.append(row)
            else:
                reader = csv.DictReader(text_handle)
                for position, record in enumerate(reader, 1):
                    row = row_from_record(
                        dict(record),
                        path,
                        position,
                        source_locator=source_locator,
                        corpus_override=corpus_override,
                    )
                    if row is not None:
                        rows.append(row)
    return member, rows


def read_table(path: Path, *, corpus_override: str | None = None) -> list[TextGlossRow]:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    rows: list[TextGlossRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=separator)
        for position, record in enumerate(reader, 1):
            row = row_from_record(
                dict(record), path, position, corpus_override=corpus_override
            )
            if row is not None:
                rows.append(row)
    return rows


def is_nikl_2025_record(record: Any) -> bool:
    """Return whether a JSON object is a 2025 NIKL parallel-corpus annotation."""
    return isinstance(record, dict) and isinstance(record.get("krlgg_sntenc"), dict) and isinstance(record.get("sign_script"), dict)


def _nikl_glosses(script: dict[str, Any]) -> list[str]:
    """Extract strong-hand glosses ordered by their annotated time."""
    tier = script.get("sign_gestures_strong") or script.get("sign_gestures_weak") or []
    parsed = []
    for position, item in enumerate(tier):
        if not isinstance(item, dict):
            continue
        gloss = normalize_space(item.get("gloss_id"))
        try:
            start, end = float(item.get("start")), float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if gloss and end > start:
            parsed.append((start, end, position, gloss))
    return [gloss for _, _, _, gloss in sorted(parsed)]


def read_nikl_2025_directory(path: Path) -> list[TextGlossRow]:
    """Read text/gloss pairs from raw NIKL 2025 JSON files, never media."""
    rows = []
    for json_path in sorted(path.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not is_nikl_2025_record(record):
            continue
        korean = normalize_space(record["krlgg_sntenc"].get("koreanText"))
        glosses = _nikl_glosses(record["sign_script"])
        if not korean or not glosses:
            continue
        record_id = normalize_space(record.get("id") or json_path.stem)
        stable = hashlib.sha1(f"{NIKL_2025_CORPUS}|{record_id}|{korean}|{' '.join(glosses)}".encode("utf-8")).hexdigest()[:16]
        rows.append(TextGlossRow(
            f"{safe_id(NIKL_2025_CORPUS)}::{record_id}::{stable}",
            korean,
            " ".join(glosses),
            NIKL_2025_CORPUS,
            str(json_path.resolve()),
            record_id,
            gloss_convention="nikl_numbered",
        ))
    if not rows:
        raise FileNotFoundError(f"NIKL 2025 형식의 한국어·gloss 주석을 찾지 못했습니다: {path}")
    return rows


def looks_like_nikl_2025_directory(path: Path) -> bool:
    """Probe JSON metadata only; video/keypoint files are never opened."""
    if not path.is_dir():
        return False
    for json_path in path.rglob("*.json"):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if is_nikl_2025_record(record):
            return True
    return False


def choose_input_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "manifests" / "korean_to_gloss.jsonl",
        path / "processed_gloss" / "samples.jsonl",
        path / "processed_gloss" / "samples.csv",
        path / "samples.jsonl",
        path / "samples.csv",
        path / "dataset.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Some ZIP tools create one extra wrapper directory. Search recursively,
    # but only for known processed metadata names: raw corpus JSON/XML/XLSX,
    # videos and keypoint files are deliberately ignored.
    recursive: list[Path] = []
    for name in ("korean_to_gloss.jsonl", "samples.jsonl", "samples.csv", "dataset.csv"):
        recursive.extend(candidate for candidate in path.rglob(name) if candidate.is_file())
    if recursive:
        def dataset_root(candidate: Path) -> Path:
            if candidate.parent.name in {"manifests", "processed_gloss"}:
                return candidate.parent.parent
            return candidate.parent

        roots = {dataset_root(candidate).resolve() for candidate in recursive}
        if len(roots) > 1:
            found = "\n".join(f"- {candidate}" for candidate in sorted(recursive))
            raise ValueError(
                "입력 폴더 아래에서 여러 데이터셋을 발견했습니다. "
                "--input에 각 데이터셋 폴더를 따로 지정하세요.\n" + found
            )

        priority = {
            "korean_to_gloss.jsonl": 0,
            "samples.jsonl": 1,
            "samples.csv": 2,
            "dataset.csv": 3,
        }
        return min(recursive, key=lambda candidate: (priority[candidate.name], str(candidate)))
    raise FileNotFoundError(
        f"지원하는 학습 메타데이터를 찾지 못했습니다: {path}\n"
        "필요 파일: manifests/korean_to_gloss.jsonl, processed_gloss/samples.jsonl, "
        "samples.jsonl, samples.csv 또는 dataset.csv"
    )


def read_input(path: Path) -> tuple[Path, list[TextGlossRow]]:
    requested = path.resolve()
    if requested.is_dir() and looks_like_nikl_2025_directory(requested):
        return requested, read_nikl_2025_directory(requested)
    selected = choose_input_file(requested)
    if selected.suffix.lower() == ".zip":
        member, rows = read_zip(selected)
        return Path(f"{selected}::{member}"), rows
    corpus_override = None
    if requested.is_dir() and selected.parent.name == "processed_gloss":
        # Match ZIP provenance when an archive has already been extracted.
        corpus_override = selected.parent.parent.name
    if selected.suffix.lower() == ".jsonl":
        return selected, read_jsonl(selected, corpus_override=corpus_override)
    if selected.suffix.lower() in {".csv", ".tsv", ".txt"}:
        return selected, read_table(selected, corpus_override=corpus_override)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {selected}")


def is_generated(row: TextGlossRow) -> bool:
    value = row.label_type.casefold()
    return "generated" in value or "synthetic" in value or "machine" in value


def deduplicate(rows: Iterable[TextGlossRow]) -> tuple[list[TextGlossRow], int]:
    selected: dict[tuple[str, str], TextGlossRow] = {}
    duplicates = 0
    for row in rows:
        key = (row.split_group(), normalize_gloss(row.gloss_sequence).casefold())
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        duplicates += 1
        # Prefer a human annotation over an otherwise identical generated label.
        if is_generated(previous) and not is_generated(row):
            selected[key] = row
    return list(selected.values()), duplicates


def split_bucket(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def assign_split(
    row: TextGlossRow,
    *,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    generated_train_only: bool,
) -> str:
    if generated_train_only and is_generated(row):
        return "train"
    bucket = split_bucket(row.split_group(), seed)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + valid_ratio:
        return "valid"
    return "train"


def write_csv(path: Path, rows: Iterable[TextGlossRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.output())
            count += 1
    return count


def ensure_separate_output(inputs: list[Path], output: Path) -> None:
    output = output.resolve()
    for item in inputs:
        source = item.resolve()
        # For directory inputs, never write inside the source tree. A ZIP/file
        # itself cannot contain output, so a sibling output directory is safe.
        if source.is_dir() and (output == source or source in output.parents):
            raise ValueError(
                "원본 보호를 위해 --output은 입력 데이터 폴더 밖에 지정해야 합니다.\n"
                f"입력: {source}\n출력: {output}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="원본을 수정하지 않고 한국어→KSL gloss 학습용 CSV를 생성합니다."
    )
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="processed 폴더, ZIP, CSV/TSV/JSONL. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument("--output", type=Path, required=True, help="새 출력 폴더")
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-generated", action="store_true",
        help="generated/synthetic 라벨을 모두 제외합니다.",
    )
    parser.add_argument(
        "--allow-generated-eval", action="store_true",
        help="생성 라벨도 valid/test에 배정합니다. 기본은 train 전용입니다.",
    )
    args = parser.parse_args()
    if args.valid_ratio < 0 or args.test_ratio < 0 or args.valid_ratio + args.test_ratio >= 1:
        raise ValueError("valid/test 비율은 0 이상이고 합이 1보다 작아야 합니다.")
    ensure_separate_output(args.input, args.output)

    all_rows: list[TextGlossRow] = []
    source_reports = []
    for input_path in args.input:
        selected, rows = read_input(input_path)
        if args.exclude_generated:
            rows = [row for row in rows if not is_generated(row)]
        source_reports.append(
            {"requested": str(input_path.resolve()), "selected": str(selected), "rows": len(rows)}
        )
        all_rows.extend(rows)
    unique_rows, duplicate_count = deduplicate(all_rows)

    conflicts: dict[str, set[str]] = defaultdict(set)
    for row in unique_rows:
        conflicts[row.split_group()].add(row.gloss_sequence)
    conflict_groups = {key: sorted(values) for key, values in conflicts.items() if len(values) > 1}

    splits: dict[str, list[TextGlossRow]] = {"train": [], "valid": [], "test": []}
    for row in unique_rows:
        split = assign_split(
            row,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            generated_train_only=not args.allow_generated_eval,
        )
        splits[split].append(row)
    for rows in splits.values():
        rows.sort(key=lambda row: (row.source_corpus, row.file_id))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = {name: write_csv(output / f"{name}.csv", rows) for name, rows in splits.items()}
    write_csv(output / "all_text_gloss.csv", sorted(unique_rows, key=lambda row: row.file_id))

    group_sets = {name: {row.split_group() for row in rows} for name, rows in splits.items()}
    intersections = {
        "train_valid": len(group_sets["train"] & group_sets["valid"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "valid_test": len(group_sets["valid"] & group_sets["test"]),
    }
    report = {
        "source_files_are_read_only": True,
        "sources": source_reports,
        "input_rows": len(all_rows),
        "unique_rows": len(unique_rows),
        "exact_duplicates_removed": duplicate_count,
        "generated_rows": sum(is_generated(row) for row in unique_rows),
        "split_counts": counts,
        "split_group_intersections": intersections,
        "conflicting_text_groups": len(conflict_groups),
        "label_type_counts": dict(Counter(row.label_type for row in unique_rows)),
        "corpus_counts": dict(Counter(row.source_corpus for row in unique_rows)),
        "gloss_convention_counts": dict(
            Counter(row.gloss_convention for row in unique_rows)
        ),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
    }
    (output / "conflicts.json").write_text(
        json.dumps(conflict_groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
