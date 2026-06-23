import re
from typing import NamedTuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from api.arxiv import REQUEST_HEADERS, REQUEST_TIMEOUT, parse_arxiv_ref, get_paper_title
from api.pdf import download_pdf, extract_text, pdf_title, MIN_PDF_TEXT_CHARS
from api.logger import logger


class ResolvedPaper(NamedTuple):
    title: str
    url: str
    text: str
    source: str


_URL_RE = re.compile(r"https?://[^\s|>]+")


def extract_first_url(text: str):
    if not text:
        return None
    m = _URL_RE.search(text.replace("<", " ").replace(">", " "))
    return m.group(0).rstrip(".,);") if m else None


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def find_pdf_link(soup, base_url):
    for a in soup.find_all("a", href=True):
        if a["href"].lower().split("?")[0].endswith(".pdf"):
            return urljoin(base_url, a["href"])
    for a in soup.find_all("a", href=True):
        if "pdf" in a.get_text(strip=True).lower():
            return urljoin(base_url, a["href"])
    return None
