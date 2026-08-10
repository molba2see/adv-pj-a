"""한국어 문장 -> gloss -> 영상/스켈레톤 매니페스트 생성기.

사용 예:
    python build_gloss_dataset.py
    python build_gloss_dataset.py --video-root ./video --out ./processed

출력:
    samples.jsonl : 샘플 단위(한국어 문장, gloss 목록, 파일 경로)
    segments.jsonl: gloss 단위(시작/끝 시간과 skeleton/video 연결)
    samples.csv   : 간단한 확인용 표

현재 폴더처럼 원본 영상이 없으면 video_path가 null로 기록됩니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def json_stem(path: Path) -> str:
    """.json 파일명에서 영상/키포인트 공통 ID를 반환한다."""
    return path.stem


def find_by_stem(root: Path | None, stem: str, extensions: set[str]) -> str | None:
    if root is None or not root.exists():
        return None
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions and p.stem == stem:
            return str(p.resolve())
    return None


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_start(item: dict[str, Any]) -> float:
    return number(item.get("start", item.get("begin", 0)))


def get_end(item: dict[str, Any], start: float) -> float:
    return number(item.get("end", item.get("stop", start)), start)


def collect_gestures(sign_script: dict[str, Any], include_auxiliary: bool) -> list[dict[str, Any]]:
    """sign_script를 시간순 gloss segment 목록으로 변환한다.

    both가 주 수어층이고 strong/weak는 보조(한 손) 수어층이다.
    기본값은 both만 사용하여 주 gloss가 중복되지 않게 한다.
    """
    tiers = ["sign_gestures_both"]
    if include_auxiliary:
        tiers += ["sign_gestures_strong", "sign_gestures_weak"]

    result: list[dict[str, Any]] = []
    for tier in tiers:
        for item in sign_script.get(tier, []) or []:
            if not isinstance(item, dict) or not item.get("gloss_id"):
                continue
            start = get_start(item)
            end = get_end(item, start)
            result.append(
                {
                    "gloss": str(item["gloss_id"]),
                    "tier": tier,
                    "start_sec": start,
                    "end_sec": end,
                    "start_frame": None,
                    "end_frame": None,
                    "express": item.get("express"),
                    "position": item.get("position", []),
                    "direction": item.get("direction", {}),
                    "sentence_loc": item.get("sentence_loc", {}),
                }
            )
    return sorted(result, key=lambda x: (x["start_sec"], x["end_sec"], x["tier"]))


def landmark_summary(landmarks: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in landmarks.items():
        if isinstance(value, list):
            result[key] = {
                "frames": len(value),
                "values_per_frame": len(value[0]) if value else 0,
            }
        else:
            result[key] = {"type": type(value).__name__}
    return result


def make_records(data_root: Path, video_root: Path | None, include_auxiliary: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    json_dir = data_root / "2.형태소_비수지(json)_TL"
    xml_dir = data_root / "1.키포인트(xml)_TL"
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"JSON 파일을 찾지 못했습니다: {json_dir}")

    samples: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        stem = json_stem(path)
        metadata = data.get("metadata", {})
        landmarks = data.get("landmarks", {})
        fps = number(metadata.get("video_fps"), 30.0)
        gestures = collect_gestures(data.get("sign_script", {}), include_auxiliary)

        # 시간(sec)을 프레임 번호로 변환한다. end는 exclusive로 사용한다.
        for g in gestures:
            g["start_frame"] = max(0, round(g["start_sec"] * fps))
            g["end_frame"] = max(g["start_frame"], round(g["end_sec"] * fps))

        xml_path = xml_dir / f"{stem}_F.xml"
        video_path = find_by_stem(video_root, stem, VIDEO_EXTENSIONS)
        record = {
            "id": metadata.get("id", stem),
            "korean_text": data.get("korean_text", ""),
            "gloss_sequence": [g["gloss"] for g in gestures if g["tier"] == "sign_gestures_both"],
            "gloss_segments": gestures,
            "fps": fps,
            "num_frames": max(
                (v.get("frames", 0) for v in landmark_summary(landmarks).values()),
                default=0,
            ),
            "json_path": str(path.resolve()),
            "xml_path": str(xml_path.resolve()) if xml_path.exists() else None,
            "video_path": video_path,
            "landmarks": landmark_summary(landmarks),
        }
        samples.append(record)

        for index, g in enumerate(gestures):
            segments.append(
                {
                    "id": record["id"],
                    "segment_index": index,
                    "korean_text": record["korean_text"],
                    "gloss": g["gloss"],
                    "tier": g["tier"],
                    "start_sec": g["start_sec"],
                    "end_sec": g["end_sec"],
                    "start_frame": g["start_frame"],
                    "end_frame": g["end_frame"],
                    "fps": fps,
                    "json_path": record["json_path"],
                    "xml_path": record["xml_path"],
                    "video_path": record["video_path"],
                }
            )
    return samples, segments


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="데이터셋 루트")
    parser.add_argument("--video-root", type=Path, default=None, help="원본 영상 폴더(선택)")
    parser.add_argument("--out", type=Path, default=Path("processed_gloss"), help="출력 폴더")
    parser.add_argument("--include-auxiliary", action="store_true", help="strong/weak gloss도 segments에 포함")
    args = parser.parse_args()

    samples, segments = make_records(args.data_root, args.video_root, args.include_auxiliary)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "samples.jsonl", samples)
    write_jsonl(args.out / "segments.jsonl", segments)

    with (args.out / "samples.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["id", "korean_text", "gloss_sequence", "fps", "num_frames", "json_path", "xml_path", "video_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in samples:
            writer.writerow({
                field: (" ".join(row[field]) if field == "gloss_sequence" else row.get(field))
                for field in fields
            })

    print(f"샘플 {len(samples)}개, gloss segment {len(segments)}개 생성")
    print(f"출력 폴더: {args.out.resolve()}")


if __name__ == "__main__":
    main()
