import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParseArxivRef(unittest.TestCase):
    def test_normalizes_all_forms_to_abs(self):
        from api.arxiv import parse_arxiv_ref
        cases = [
            "https://arxiv.org/abs/2501.12345",
            "https://arxiv.org/pdf/2501.12345",
            "https://arxiv.org/pdf/2501.12345.pdf",
            "https://arxiv.org/html/2501.12345",
            "https://arxiv.org/abs/2501.12345v2",
            "https://arxiv.org/pdf/2501.12345v3.pdf",
            "여기 봐주세요 2501.12345 이 논문",
        ]
        for c in cases:
            self.assertEqual(
                parse_arxiv_ref(c),
                "https://arxiv.org/abs/2501.12345",
                msg=f"failed for: {c}",
            )

    def test_old_style_id(self):
        from api.arxiv import parse_arxiv_ref
        self.assertEqual(
            parse_arxiv_ref("https://arxiv.org/abs/cs/0501001"),
            "https://arxiv.org/abs/cs/0501001",
        )

    def test_returns_none_without_arxiv_ref(self):
        from api.arxiv import parse_arxiv_ref
        self.assertIsNone(parse_arxiv_ref("hello world, no paper here"))
        self.assertIsNone(parse_arxiv_ref(""))


if __name__ == "__main__":
    unittest.main()
