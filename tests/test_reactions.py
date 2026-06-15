import os, sys, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSettings(unittest.TestCase):
    def test_reaction_settings(self):
        import settings
        self.assertTrue(settings.PAPERS_STORE_PATH.replace("\\", "/").endswith("reactions/papers.json"))
        self.assertGreaterEqual(settings.HARVEST_WINDOW_DAYS, 1)


class TestStore(unittest.TestCase):
    def _tmp(self):
        import tempfile
        d = tempfile.mkdtemp()
        return os.path.join(d, "papers.json")

    def test_hash_user_deterministic_and_salted(self):
        from api.reactions import hash_user
        self.assertEqual(hash_user("U1", "saltA"), hash_user("U1", "saltA"))
        self.assertNotEqual(hash_user("U1", "saltA"), hash_user("U1", "saltB"))
        self.assertEqual(len(hash_user("U1", "saltA")), 16)

    def test_add_posted_roundtrip_and_dedup(self):
        from api.reactions import load_store, save_store, add_posted
        path = self._tmp()
        store = load_store(path)
        add_posted(store, ts="100.1", thread_ts="100.0", channel_id="C1",
                   workspace="w", paper_info="P (url)", paper_url="url",
                   field="cs.CL", posted_at="2026-06-15T00:00:00+00:00")
        add_posted(store, ts="100.1", thread_ts="100.0", channel_id="C1",
                   workspace="w", paper_info="P (url)", paper_url="url",
                   field="cs.CL", posted_at="2026-06-15T00:00:00+00:00")  # 동일 ts 재기록
        save_store(store, path)
        store2 = load_store(path)
        self.assertEqual(len(store2), 1)
        self.assertEqual(store2["100.1"]["field"], "cs.CL")
        self.assertIsNone(store2["100.1"]["reactions"])


if __name__ == "__main__":
    unittest.main()
