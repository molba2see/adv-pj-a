from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from data_processing.prepare_ksl_text_data import (
    TextGlossRow,
    assign_split,
    deduplicate,
    ensure_separate_output,
    normalize_group_text,
    read_input,
)


class DataPreparationTests(unittest.TestCase):
    def test_reads_processed_csv_without_touching_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "processed"
            root.mkdir()
            source = root / "samples.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("sample_id", "korean_text", "gloss_sequence", "video_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "x1",
                        "korean_text": "학교에 갑니다.",
                        "gloss_sequence": "학교1 가다1",
                        "video_path": "/do/not/read.mp4",
                    }
                )
            before = source.read_bytes()
            selected, rows = read_input(root)
            self.assertEqual(selected, source)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].korean_text, "학교에 갑니다.")
            self.assertEqual(rows[0].gloss_sequence, "학교1 가다1")
            self.assertEqual(source.read_bytes(), before)

    def test_reads_gksl_original_column_names(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "GKSL3k_original.csv"
            source.write_text(
                "dataset,Gloss level Korean Sign Language (GKSL) sentence,"
                "Word level Korean Language (WKL) sentence\n"
                "KETI-Emergency,집 불,집에 불이 났어요.\n",
                encoding="utf-8-sig",
            )
            _, rows = read_input(source)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].korean_text, "집에 불이 났어요.")
            self.assertEqual(rows[0].gloss_sequence, "집 불")
            self.assertEqual(rows[0].source_corpus, "KETI-Emergency")
            self.assertEqual(rows[0].gloss_convention, "gksl_unnumbered")

    def test_dedup_prefers_human_label(self):
        generated = TextGlossRow("g", "안녕하세요.", "안녕하세요1", "a", "a", "1", "generated")
        human = TextGlossRow("h", "안녕하세요!", "안녕하세요1", "b", "b", "2", "human_annotation")
        rows, removed = deduplicate([generated, human])
        self.assertEqual(removed, 1)
        self.assertEqual(rows[0].file_id, "h")

    def test_normalized_text_never_crosses_splits(self):
        rows = [
            TextGlossRow("1", "끝이에요?", "끝1", "a", "a", "1"),
            TextGlossRow("2", "끝이에요", "끝1", "b", "b", "2"),
        ]
        self.assertEqual(normalize_group_text(rows[0].korean_text), normalize_group_text(rows[1].korean_text))
        assigned = {
            assign_split(
                row, seed=42, valid_ratio=0.1, test_ratio=0.1, generated_train_only=True
            )
            for row in rows
        }
        self.assertEqual(len(assigned), 1)

    def test_generated_label_is_train_only(self):
        row = TextGlossRow("1", "예문", "예문1", "a", "a", "1", "generated_from_video")
        split = assign_split(
            row, seed=1, valid_ratio=0.49, test_ratio=0.49, generated_train_only=True
        )
        self.assertEqual(split, "train")

    def test_rejects_output_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            with self.assertRaises(ValueError):
                ensure_separate_output([source], source / "output")

    def test_reads_processed_jsonl_directly_from_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / "재난안전수어영상.zip"
            record = {
                "id": "NIA_TEST_1",
                "korean_text": "한파에 주의하세요.",
                "gloss_sequence": ["춥다1", "조심1"],
                "video_path": "/not/opened.mp4",
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "processed_gloss/samples.jsonl",
                    json.dumps(record, ensure_ascii=False) + "\n",
                )
                archive.writestr("2.형태소_비수지(json)_TL/huge.json", "{}")
            before = archive_path.read_bytes()
            selected, rows = read_input(archive_path)
            self.assertIn("processed_gloss", str(selected))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].source_corpus, "재난안전수어영상")
            self.assertEqual(rows[0].gloss_sequence, "춥다1 조심1")
            self.assertEqual(archive_path.read_bytes(), before)

    def test_zip_and_extracted_folder_produce_equivalent_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extracted = root / "disaster"
            metadata = extracted / "processed_gloss" / "samples.jsonl"
            metadata.parent.mkdir(parents=True)
            record = {
                "sample_id": "NIA_TEST_1",
                "korean_text": "태풍을 조심하세요.",
                "gloss_sequence": ["태풍1", "조심1"],
                "video_path": "ignored.mp4",
            }
            payload = json.dumps(record, ensure_ascii=False) + "\n"
            metadata.write_text(payload, encoding="utf-8")
            # This is a raw annotation file and must never be selected.
            (extracted / "raw.json").write_text("{not valid json", encoding="utf-8")

            archive_path = root / "disaster.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("processed_gloss/samples.jsonl", payload)
                archive.writestr("raw/raw.json", "{not valid json")

            selected_folder, folder_rows = read_input(extracted)
            selected_zip, zip_rows = read_input(archive_path)
            self.assertEqual(selected_folder, metadata)
            self.assertIn("processed_gloss", str(selected_zip))
            comparable = lambda row: (
                row.file_id,
                row.korean_text,
                row.gloss_sequence,
                row.source_corpus,
                row.source_record_id,
                row.label_type,
            )
            self.assertEqual(comparable(folder_rows[0]), comparable(zip_rows[0]))

    def test_finds_processed_metadata_below_one_wrapper_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "wrapper"
            metadata = root / "disaster" / "processed_gloss" / "samples.jsonl"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps(
                    {"sample_id": "1", "korean_text": "비가 옵니다.", "gloss": "비1 오다1"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            selected, rows = read_input(root)
            self.assertEqual(selected, metadata)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].source_corpus, "disaster")

    def test_file_input_allows_sibling_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.zip"
            source.touch()
            ensure_separate_output([source], Path(temp) / "prepared")

    def test_reads_raw_nikl_2025_json_without_opening_media(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "NIKL_2025"
            annotation = root / "batch" / "sample.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(
                json.dumps(
                    {
                        "id": "VXPAKOKS250000001",
                        "krlgg_sntenc": {"koreanText": "학교에 갑니다."},
                        "sign_script": {"sign_gestures_strong": [
                            {"gloss_id": "가다1", "start": 1.0, "end": 1.5},
                            {"gloss_id": "학교1", "start": 0.0, "end": 0.8},
                        ]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            media = root / "batch" / "sample.mp4"
            media.write_bytes(b"must not be opened")
            before_annotation, before_media = annotation.read_bytes(), media.read_bytes()

            selected, rows = read_input(root)

            self.assertEqual(selected, root.resolve())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].gloss_sequence, "학교1 가다1")
            self.assertEqual(rows[0].source_corpus, "NIKL_Sign_Language_Parallel_Corpus_2025")
            self.assertEqual(rows[0].gloss_convention, "nikl_numbered")
            self.assertEqual(annotation.read_bytes(), before_annotation)
            self.assertEqual(media.read_bytes(), before_media)


if __name__ == "__main__":
    unittest.main()
