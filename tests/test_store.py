import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStore(unittest.TestCase):
    def setUp(self):
        from api.store import Store

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_missing_key_returns_empty(self):
        self.assertEqual(self.store.get_abstract("없는 논문"), "")
        self.assertFalse(self.store.has_summary("없는 논문"))
        self.assertIsNone(self.store.get_page("https://x.test"))

    def test_upsert_overwrites_only_that_key(self):
        self.store.put_abstract("A", "첫 초록")
        self.store.put_abstract("B", "다른 초록")
        self.store.put_abstract("A", "고친 초록")
        self.assertEqual(self.store.get_abstract("A"), "고친 초록")
        self.assertEqual(self.store.get_abstract("B"), "다른 초록")
        self.assertEqual(self.store.count("abstracts"), 2)

    def test_full_content_roundtrips_dict(self):
        value = {"1": {"title": "Intro", "content": "본문"}}
        self.store.put_full_content("A", value)
        self.assertEqual(self.store.get_full_content("A"), value)

    def test_full_content_accepts_plain_text(self):
        self.store.put_full_content("B", "그냥 텍스트")
        self.assertEqual(self.store.get_full_content("B"), "그냥 텍스트")

    def test_summary_keeps_metadata(self):
        self.store.put_summary(
            "A", "요약", schema_version="4sections", model="gpt-5.4-nano"
        )
        self.assertEqual(self.store.get_summary("A"), "요약")
        self.assertTrue(self.store.has_summary("A"))

    def test_second_connection_sees_writes(self):
        from api.store import Store

        other = Store(self.tmp.name)
        self.store.put_abstract("A", "초록")
        other.put_abstract("B", "다른 초록")
        self.assertEqual(other.get_abstract("A"), "초록")
        self.assertEqual(self.store.get_abstract("B"), "다른 초록")
        other.close()

    def test_page_and_digest(self):
        self.store.put_page("https://x.test", "제목", "본문")
        page = self.store.get_page("https://x.test")
        self.assertEqual(page["title"], "제목")
        self.assertEqual(page["text"], "본문")
        self.store.put_digest("1.1", "요지", covered_until_ts="2.2")
        digest = self.store.get_digest("1.1")
        self.assertEqual(digest["digest"], "요지")
        self.assertEqual(digest["covered_until_ts"], "2.2")

    def test_count_rejects_unknown_table(self):
        with self.assertRaises(ValueError):
            self.store.count("papers; DROP TABLE abstracts")


if __name__ == "__main__":
    unittest.main()
