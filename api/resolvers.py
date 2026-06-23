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


def _citation_pdf(soup, base_url):
    """대부분 학회 페이지가 제공하는 <meta name="citation_pdf_url">."""
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    return urljoin(base_url, meta["content"]) if meta and meta.get("content") else None


def _citation_title(soup):
    """<meta name="citation_title"> (대부분 학회 페이지 제공). 없으면 빈 문자열."""
    meta = soup.find("meta", attrs={"name": "citation_title"})
    return meta["content"].strip() if meta and meta.get("content") else ""


def _fetch_soup(url):
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    return BeautifulSoup(r.text, "html.parser")


def _pdf_to_paper(pdf_url, *, title, source, on_progress):
    on_progress("downloading")
    pdf = download_pdf(pdf_url)
    if not pdf:
        return None
    text = extract_text(pdf)
    if len(text.strip()) < MIN_PDF_TEXT_CHARS:
        return None
    return ResolvedPaper(
        title=(title or pdf_title(pdf) or pdf_url.rsplit("/", 1)[-1]),
        url=pdf_url, text=text, source=source,
    )


def _resolve_direct_pdf(url, *, source, on_progress):
    return _pdf_to_paper(url, title="", source=source, on_progress=on_progress)


def _resolve_via_html(url, *, on_progress):
    soup = _fetch_soup(url)
    title = _citation_title(soup)
    if not title:
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
    pdf_link = _citation_pdf(soup, url) or find_pdf_link(soup, url)
    if not pdf_link:
        return None
    return _pdf_to_paper(pdf_link, title=title, source="html", on_progress=on_progress)


def _openreview_pdf_url(url):
    m = re.search(r"[?&]id=([^&]+)", url)
    return f"https://openreview.net/pdf?id={m.group(1)}" if m else url


def _resolve_arxiv(abs_url, arxiv_client, cache, on_progress):
    from api.arxiv import get_paper_info
    title = get_paper_title(abs_url)
    paper_info = get_paper_info(abs_url, title)
    abstract = (cache.paper_abstracts.get(paper_info) if cache else None) \
        or arxiv_client.get_paper_abstract(abs_url)
    if cache:
        cache.update_paper_abstracts(paper_info, abstract)
    # HTML(experimental) 있으면 섹션 텍스트, 없으면 PDF 폴백
    text = f"Abstract: {abstract}\n\n"
    html_link = arxiv_client.get_html_experimental_link(abs_url)
    if html_link != "Link not found":
        full = arxiv_client.get_paper_full_content(html_link)
        for sec in (full.values() if isinstance(full, dict) else []):
            if sec.get("title") != "No title found" and sec.get("content"):
                text += f"Section: {sec['title']}\n{sec['content']}\n\n"
    else:
        on_progress("downloading")
        pdf = download_pdf(abs_url.replace("/abs/", "/pdf/"))
        if pdf:
            text += extract_text(pdf)
    return ResolvedPaper(title=title, url=abs_url, text=text, source="arxiv")


def build_resolver(arxiv_client, cache):
    """url을 ResolvedPaper로 변환하는 cascade 클로저. 미지원/실패는 None."""
    def resolve(url, on_progress=lambda s: None):
        try:
            arxiv_abs = parse_arxiv_ref(url)
            if arxiv_abs:
                return _resolve_arxiv(arxiv_abs, arxiv_client, cache, on_progress)
            if "openreview.net" in urlparse(url).netloc:
                return _resolve_direct_pdf(
                    _openreview_pdf_url(url), source="openreview", on_progress=on_progress
                )
            if is_pdf_url(url):
                return _resolve_direct_pdf(url, source="pdf", on_progress=on_progress)
            return _resolve_via_html(url, on_progress=on_progress)  # omnivorous
        except Exception as e:
            logger.info(f"resolve failed for {url}: {e}")
            return None
    return resolve
