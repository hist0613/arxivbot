import os, sys, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSettings(unittest.TestCase):
    def test_reaction_settings(self):
        import settings
        self.assertTrue(settings.PAPERS_STORE_PATH.replace("\\", "/").endswith("reactions/papers.json"))
        self.assertGreaterEqual(settings.HARVEST_WINDOW_DAYS, 1)


if __name__ == "__main__":
    unittest.main()
