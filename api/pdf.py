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
    """앞 max_pages 페이지를 텍스트로 추출.

    pymupdf4llm 신형 layout(ONNX) 경로가 일부 환경(Windows Store python 등)에서
    int32/int64 ONNX 오류로 깨지므로, 실패하면 PyMuPDF 기본 추출(get_text sort)로
    폴백한다. 둘 다 reading-order를 어느 정도 보존하며, 후자는 ONNX 미사용으로 견고."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    n = min(max_pages, doc.page_count)
    try:
        return pymupdf4llm.to_markdown(doc, pages=list(range(n)))
    except Exception as e:
        logger.info(f"pymupdf4llm failed, fallback to get_text(sort): {e}")
        return "\n\n".join(doc[i].get_text(sort=True) for i in range(n))


def pdf_title(pdf_bytes: bytes) -> str:
    """PDF 메타데이터 제목(없으면 빈 문자열)."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return (doc.metadata or {}).get("title", "") or ""
