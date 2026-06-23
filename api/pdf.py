import time

import requests
import pymupdf
import pymupdf4llm

from api.arxiv import REQUEST_HEADERS, REQUEST_TIMEOUT
from api.logger import logger
from settings import MAX_LLM_TRIALS

MIN_PDF_TEXT_CHARS = 500


def looks_like_pdf(content_type: str, body: bytes) -> bool:
    if content_type and "application/pdf" in content_type.lower():
        return True
    return body[:5] == b"%PDF-"


def download_pdf(url: str):
    """PDF 바이트 반환. 실패/비-PDF면 None."""
    for trial in range(MAX_LLM_TRIALS):
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and looks_like_pdf(
                r.headers.get("Content-Type", ""), r.content
            ):
                return r.content
            logger.info(f"download_pdf non-pdf/{r.status_code}: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.info(f"download_pdf retry {trial}: {e}")
            time.sleep(trial * 5 + 2)
    return None


def extract_text(pdf_bytes: bytes, max_pages: int = 8) -> str:
    """앞 max_pages 페이지를 reading-order 마크다운 텍스트로 추출."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = list(range(min(max_pages, doc.page_count)))
    return pymupdf4llm.to_markdown(doc, pages=pages)


def pdf_title(pdf_bytes: bytes) -> str:
    """PDF 메타데이터 제목(없으면 빈 문자열)."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return (doc.metadata or {}).get("title", "") or ""
