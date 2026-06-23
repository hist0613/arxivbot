import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_pdf(text):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    return doc.tobytes()


class TestLooksLikePdf(unittest.TestCase):
    def test_by_content_type(self):
        from api.pdf import looks_like_pdf
        self.assertTrue(looks_like_pdf("application/pdf", b""))

    def test_by_magic_bytes(self):
        from api.pdf import looks_like_pdf
        self.assertTrue(looks_like_pdf("text/html", b"%PDF-1.7 ..."))

    def test_html_is_not_pdf(self):
        from api.pdf import looks_like_pdf
        self.assertFalse(looks_like_pdf("text/html", b"<html>"))


class TestExtractText(unittest.TestCase):
    def test_extracts_words(self):
        from api.pdf import extract_text
        out = extract_text(_make_pdf("MultiSourceSummaryHello"))
        self.assertIn("MultiSourceSummaryHello", out)


if __name__ == "__main__":
    unittest.main()
