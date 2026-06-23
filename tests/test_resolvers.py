import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestExtractFirstUrl(unittest.TestCase):
    def test_slack_wrapped(self):
        from api.resolvers import extract_first_url
        self.assertEqual(
            extract_first_url("<@U1> <https://arxiv.org/abs/2501.1>"),
            "https://arxiv.org/abs/2501.1",
        )

    def test_with_label(self):
        from api.resolvers import extract_first_url
        self.assertEqual(
            extract_first_url("see <https://x.org/p.pdf|the pdf>"),
            "https://x.org/p.pdf",
        )

    def test_none_when_absent(self):
        from api.resolvers import extract_first_url
        self.assertIsNone(extract_first_url("no link"))


class TestIsPdfUrl(unittest.TestCase):
    def test_true(self):
        from api.resolvers import is_pdf_url
        self.assertTrue(is_pdf_url("https://x.org/a/b.pdf"))
        self.assertTrue(is_pdf_url("https://x.org/a/b.pdf?x=1"))

    def test_false(self):
        from api.resolvers import is_pdf_url
        self.assertFalse(is_pdf_url("https://x.org/abs/1"))


class TestFindPdfLink(unittest.TestCase):
    def test_prefers_pdf_href(self):
        from bs4 import BeautifulSoup
        from api.resolvers import find_pdf_link
        soup = BeautifulSoup('<a href="/papers/x.pdf">X</a>', "html.parser")
        self.assertEqual(
            find_pdf_link(soup, "https://openaccess.thecvf.com/c"),
            "https://openaccess.thecvf.com/papers/x.pdf",
        )

    def test_falls_back_to_pdf_text(self):
        from bs4 import BeautifulSoup
        from api.resolvers import find_pdf_link
        soup = BeautifulSoup('<a href="/dl/9">PDF</a>', "html.parser")
        self.assertEqual(find_pdf_link(soup, "https://h.org/x"), "https://h.org/dl/9")

    def test_none(self):
        from bs4 import BeautifulSoup
        from api.resolvers import find_pdf_link
        soup = BeautifulSoup("<a href='/x'>home</a>", "html.parser")
        self.assertIsNone(find_pdf_link(soup, "https://h.org"))


if __name__ == "__main__":
    unittest.main()
