from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.retriever import GlossLexicon, GrammarRetriever, RagContextBuilder


class RagLeakageTests(unittest.TestCase):
    def test_unique_target_is_removed_from_own_train_context(self):
        lexicon = GlossLexicon(min_count=1).build([("학교에 간다", "학교1 가다1")])
        normal = dict(lexicon.lookup("학교에 간다"))
        leave_one_out = dict(
            lexicon.lookup("학교에 간다", exclude_gloss="학교1 가다1")
        )
        self.assertEqual(normal.get("학교"), "학교1")
        self.assertNotIn("학교", leave_one_out)

    def test_evidence_from_another_row_remains(self):
        pairs = [
            ("학교에 간다", "학교1 가다1"),
            ("학교가 좋다", "학교1 좋다1"),
        ]
        lexicon = GlossLexicon(min_count=1).build(pairs)
        leave_one_out = dict(
            lexicon.lookup("학교에 간다", exclude_gloss="학교1 가다1")
        )
        self.assertEqual(leave_one_out.get("학교"), "학교1")

    def test_structured_grammar_evidence_matches_model_context(self):
        chunk = {
            "chunk_id": "g-test",
            "page": 17,
            "chapter": "문장",
            "section": "부정법",
            "heading": "부정법",
            "text": "한국수어의 부정문은 부정 표현을 문장 끝에 둘 수 있다.",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grammar.jsonl"
            path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
            retriever = GrammarRetriever().build(path)
            cfg = SimpleNamespace(
                task_prefix="번역: ", mode="grammar", include_feature_tags=True,
                feat_marker=" | 자질: ", gram_marker=" | 문법: ",
                top_k_grammar=1, min_score=0.0, grammar_query_mode="feature",
                grammar_char_limit=220, lex_marker="", ex_marker="",
                top_k_example=0, max_example_score=0.95, max_lexicon_items=0,
            )
            result = RagContextBuilder(cfg, grammar_ret=retriever).build_structured(
                "학교에 가지 않았나요?"
            )
        self.assertIn("부정", result["grammar_features"])
        self.assertEqual(result["grammar_hits"][0].chunk_id, "g-test")
        self.assertEqual(result["grammar_hits"][0].page, 17)
        self.assertIn("한국수어의 부정문", result["source"])


if __name__ == "__main__":
    unittest.main()
