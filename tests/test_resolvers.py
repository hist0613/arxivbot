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


from unittest import mock


class TestCitationTitle(unittest.TestCase):
    def test_prefers_citation_title_meta(self):
        from bs4 import BeautifulSoup
        from api.resolvers import _citation_title
        soup = BeautifulSoup(
            '<meta name="citation_title" content="Segment Anything"><title>Repo</title>',
            "html.parser",
        )
        self.assertEqual(_citation_title(soup), "Segment Anything")

    def test_empty_when_absent(self):
        from bs4 import BeautifulSoup
        from api.resolvers import _citation_title
        self.assertEqual(_citation_title(BeautifulSoup("<title>x</title>", "html.parser")), "")


class TestResolveCascade(unittest.TestCase):
    def test_direct_pdf_url(self):
        from api import resolvers
        with mock.patch.object(resolvers, "download_pdf", return_value=b"%PDF-x"), \
             mock.patch.object(resolvers, "extract_text", return_value="z" * 600), \
             mock.patch.object(resolvers, "pdf_title", return_value="Direct Title"):
            r = resolvers.build_resolver(None, None)("https://h.org/p.pdf")
        self.assertEqual(r.source, "pdf")
        self.assertEqual(r.title, "Direct Title")
        self.assertTrue(r.text.startswith("z"))

    def test_generic_html(self):
        from bs4 import BeautifulSoup
        from api import resolvers
        html = '<html><head><title>Cool Paper</title></head>' \
               '<body><a href="/p/cool.pdf">PDF</a></body></html>'
        with mock.patch.object(resolvers, "_fetch_soup",
                               return_value=BeautifulSoup(html, "html.parser")), \
             mock.patch.object(resolvers, "download_pdf", return_value=b"%PDF-"), \
             mock.patch.object(resolvers, "extract_text", return_value="y" * 600), \
             mock.patch.object(resolvers, "pdf_title", return_value=""):
            r = resolvers.build_resolver(None, None)("https://openaccess.thecvf.com/x.html")
        self.assertIn("Cool Paper", r.title)
        self.assertEqual(r.source, "html")

    def test_short_text_is_none(self):
        from bs4 import BeautifulSoup
        from api import resolvers
        with mock.patch.object(resolvers, "_fetch_soup",
                               return_value=BeautifulSoup('<a href="/x.pdf">PDF</a>', "html.parser")), \
             mock.patch.object(resolvers, "download_pdf", return_value=b"%PDF-"), \
             mock.patch.object(resolvers, "extract_text", return_value="tiny"), \
             mock.patch.object(resolvers, "pdf_title", return_value=""):
            r = resolvers.build_resolver(None, None)("https://h.org/p.html")
        self.assertIsNone(r)

    def test_progress_downloading_emitted(self):
        from api import resolvers
        seen = []
        with mock.patch.object(resolvers, "download_pdf", return_value=b"%PDF-"), \
             mock.patch.object(resolvers, "extract_text", return_value="z" * 600), \
             mock.patch.object(resolvers, "pdf_title", return_value="T"):
            resolvers.build_resolver(None, None)(
                "https://h.org/p.pdf", on_progress=seen.append
            )
        self.assertIn("downloading", seen)


if __name__ == "__main__":
    unittest.main()
