import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        from api.cache import CacheManager
        from api.store import Store

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.cache = CacheManager(store=Store(self.tmp.name))

    def tearDown(self):
        self.cache.store.close()
        os.unlink(self.tmp.name)

    def test_dict_like_reads(self):
        self.assertEqual(self.cache.paper_abstracts["없음"], "")
        self.assertNotIn("없음", self.cache.paper_abstracts)
        self.cache.update_paper_abstracts("A", "초록")
        self.assertIn("A", self.cache.paper_abstracts)
        self.assertEqual(self.cache.paper_abstracts["A"], "초록")
        self.assertEqual(self.cache.paper_summarizations.get("A", ""), "")

    def test_full_contents_dict_roundtrip(self):
        value = {"1": {"title": "Intro", "content": "본문"}}
        self.cache.update_paper_full_contents("A", value)
        self.assertEqual(self.cache.paper_full_contents["A"], value)

    def test_summary_write_does_not_touch_others(self):
        self.cache.update_paper_summarizations("A", "요약 A")
        self.cache.update_paper_summarizations("B", "요약 B")
        self.assertTrue(self.cache.has_paper_summarization("A"))
        self.assertEqual(self.cache.paper_summarizations["B"], "요약 B")

    def test_two_managers_share_one_db(self):
        from api.cache import CacheManager
        from api.store import Store

        other = CacheManager(store=Store(self.tmp.name))
        self.cache.update_paper_summarizations("A", "요약")
        self.assertEqual(other.paper_summarizations["A"], "요약")
        other.store.close()

    def test_schema_version_is_recorded(self):
        import json

        current = json.dumps(
            {
                "Prior Approaches": "가",
                "Core Contribution": "나",
                "Technical Challenges": "다",
                "Empirical Impact": "라",
            },
            ensure_ascii=False,
        )
        self.cache.update_paper_summarizations("A", current)
        self.cache.update_paper_summarizations(
            "B", json.dumps({"What's New": "옛 포맷"}, ensure_ascii=False)
        )
        row = self.cache.store._one(
            "SELECT schema_version FROM summaries WHERE paper_info=?", "A"
        )
        self.assertEqual(row["schema_version"], "4sections")
        row = self.cache.store._one(
            "SELECT schema_version FROM summaries WHERE paper_info=?", "B"
        )
        self.assertEqual(row["schema_version"], "legacy")


if __name__ == "__main__":
    unittest.main()
