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


class FakeResp(dict):
    pass


class FakeClient:
    """conversations_replies만 흉내. thread_ts별 고정 응답."""
    def __init__(self, replies_by_thread):
        self.replies_by_thread = replies_by_thread
        self.calls = 0

    def conversations_replies(self, channel, ts, cursor=None, limit=200):
        self.calls += 1
        return FakeResp(messages=self.replies_by_thread[ts], response_metadata={})


class TestHarvest(unittest.TestCase):
    def test_harvest_updates_in_window_excludes_bot_and_hashes(self):
        from api.reactions import harvest_reactions, hash_user
        store = {
            "200.1": {"thread_ts": "200.0", "channel_id": "C1", "workspace": "w",
                      "paper_info": "P1", "paper_url": "u1", "field": "cs.CL",
                      "posted_at": "2026-06-15T00:00:00+00:00",
                      "last_harvested": None, "reactions": None},
            "999.1": {"thread_ts": "999.0", "channel_id": "C1", "workspace": "w",
                      "paper_info": "OLD", "paper_url": "u9", "field": "cs.CL",
                      "posted_at": "2026-05-01T00:00:00+00:00",  # 윈도우 밖
                      "last_harvested": None, "reactions": None},
        }
        client = FakeClient({
            "200.0": [
                {"ts": "200.0"},  # thread 제목(엔트리 아님)
                {"ts": "200.1", "reactions": [
                    {"name": "thumbsup", "users": ["U_bot", "U_a", "U_b"], "count": 3}
                ]},
            ],
        })
        import datetime as dt
        now = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)
        n = harvest_reactions(client, store, window_days=14,
                              bot_user_id="U_bot", now=now, salt="s")
        self.assertEqual(n, 1)                       # 윈도우 내 1건만
        self.assertEqual(client.calls, 1)            # thread 1개만 호출(999는 윈도우 밖)
        rs = store["200.1"]["reactions"]
        self.assertEqual(rs[0]["emoji"], "thumbsup")
        self.assertEqual(rs[0]["count"], 2)          # 봇 제외 후 2
        self.assertEqual(set(rs[0]["hashed_users"]),
                         {hash_user("U_a", "s"), hash_user("U_b", "s")})
        self.assertIsNotNone(store["200.1"]["last_harvested"])
        self.assertIsNone(store["999.1"]["reactions"])  # 윈도우 밖 미갱신


if __name__ == "__main__":
    unittest.main()
