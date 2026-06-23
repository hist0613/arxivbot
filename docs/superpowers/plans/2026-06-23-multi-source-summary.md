# 다출처 on-demand 요약 + 단계별 진행 표시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** @멘션으로 arXiv뿐 아니라 ACL/CVPR/NeurIPS/ICML/OpenReview/IJCAI/Interspeech/직접 PDF 등 다양한 출처 논문 URL을 받아 PDF를 추출·요약하고, 진행 단계를 Slack 메시지 1개 편집으로 보여준다.

**Architecture:** 도메인 라우팅 cascade(arXiv 특수 → OpenReview/직접 PDF → **범용 HTML→PDF 리졸버**)로 출처를 텍스트로 변환하고, `Service.summarize_text`(스키마검증 캐시)로 통일 요약. `process_mention`은 `on_progress` 콜백으로 단계를 알리고 listener가 `chat.update`로 같은 메시지를 편집한다.

**Tech Stack:** Python 3.11, pymupdf4llm(PDF 텍스트), BeautifulSoup, slack-bolt(Socket Mode), unittest.

> **스펙 대비 refinement**: 스펙은 호스트별 리졸버를 나열했지만, 실제 차이는 "PDF 링크 위치 + 제목"뿐이라 **범용 HTML→PDF 리졸버 하나**로 대부분 venue를 덮고(thecvf/acl/neurips/pmlr/ijcai/isca/aaai), arXiv·OpenReview·직접PDF만 특수 처리한다. 덜 brittle하고 omnivorous(미지 도메인도 시도).

---

## 실행 환경 메모 (모든 테스트에 적용)

- 테스트/실행 Python은 **Windows PowerShell의 Python 3.11**. WSL bash에서:
  `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\<파일>.py"`
- git은 WSL bash: `cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git ...`
- 작업 후 **리스너 재시작** 필요(스케줄러가 옛 코드 잡음): `Stop-ScheduledTask`/`Start-ScheduledTask -TaskName arxivbot-listener`.

## File Structure

| 파일 | 책임 |
|---|---|
| `requirements.txt` (수정) | `pymupdf4llm` 추가 |
| `api/pdf.py` (신규) | `looks_like_pdf`, `download_pdf`, `extract_text`, `pdf_title`, `MIN_PDF_TEXT_CHARS` |
| `api/resolvers.py` (신규) | `ResolvedPaper`, `extract_first_url`, `find_pdf_link`, `is_pdf_url`, 리졸버들, `build_resolver` cascade |
| `api/service.py` (수정) | `Service.summarize_text`; `summarize_one`은 위임 |
| `api/on_demand.py` (수정) | `process_mention`을 resolve+on_progress+summarize_text로 개정 |
| `listener.py` (수정) | 진행 표시(post→update), `build_resolver` 주입 |
| `prompts.py` (수정) | "arxiv paper"→"paper", 아티팩트 무시 한 줄 |
| `tests/test_resolvers.py` (신규) | 순수 단위 + cascade 라우팅(mock) |
| `tests/test_on_demand.py` (수정) | `summarize_text`, `process_mention` 단계/결과 |
| `tests/test_summary.py` (수정) | 프롬프트 아티팩트 문구 |
| `tests/smoke_resolvers.py` (신규) | 호스트별 실제 URL 수동 검증 + PDF 저장 |

---

## Task 1: pymupdf4llm 의존성 추가

**Files:** Modify: `requirements.txt`

- [ ] **Step 1: requirements.txt에 추가**

`requirements.txt` 마지막 줄에 추가:
```
pymupdf4llm
```

- [ ] **Step 2: 설치**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python -m pip install -U pymupdf4llm"`
Expected: 설치 성공 (pymupdf도 함께 설치)

- [ ] **Step 3: import 확인**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python -c 'import pymupdf, pymupdf4llm'"` 후 `echo OK`
Expected: 오류 없음

- [ ] **Step 4: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add requirements.txt && git commit -m "build: add pymupdf4llm for PDF text extraction"
```

---

## Task 2: Service.summarize_text (통일 요약 진입점)

**Files:** Modify: `api/service.py`; Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`의 `TestSummarizeOne` 클래스 **아래**에 추가:

```python
class TestSummarizeText(unittest.TestCase):
    def _service(self, agent, cache):
        from api.service import Service
        from api.agent import Encoder
        from settings import MODEL
        return Service(arxiv=None, agent=agent, encoder=Encoder(MODEL), cache=cache)

    def test_current_schema_cache_hit_skips_agent(self):
        cache = _FakeCache({"P (u)": VALID_SUMMARY})
        agent = _FakeAgent(VALID_SUMMARY)
        out = self._service(agent, cache).summarize_text("P (u)", "some text")
        self.assertEqual(out, VALID_SUMMARY)
        self.assertEqual(agent.calls, 0)

    def test_stale_schema_resummarized_and_stored(self):
        cache = _FakeCache({"P (u)": OLD_SUMMARY})
        agent = _FakeAgent(VALID_SUMMARY)
        out = self._service(agent, cache).summarize_text("P (u)", "some text")
        self.assertEqual(out, VALID_SUMMARY)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(cache.paper_summarizations["P (u)"], VALID_SUMMARY)

    def test_empty_result_not_cached(self):
        cache = _FakeCache()
        out = self._service(_FakeAgent(""), cache).summarize_text("P (u)", "t")
        self.assertEqual(out, "")
        self.assertNotIn("P (u)", cache.paper_summarizations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `AttributeError: 'Service' object has no attribute 'summarize_text'`

- [ ] **Step 3: Implement**

`api/service.py`에서 `summarize_one`을 다음으로 **교체**(요약 입력 구성 후 `summarize_text`로 위임):

```python
    def summarize_text(self, paper_info: str, text: str) -> str:
        """출처 무관 통일 진입점. 현재 4섹션 스키마 캐시면 재사용,
        아니면 truncate 후 요약하고 비어있지 않을 때만 저장한다."""
        cached = self.cache.paper_summarizations.get(paper_info, "")
        if is_current_summary_schema(cached):
            return cached
        summarization = self.agent.summarize(self.encoder.truncate_text(text))
        if summarization:
            self.cache.update_paper_summarizations(paper_info, summarization)
        return summarization

    def summarize_one(
        self, paper_info: str, paper_abstract: str, paper_full_content
    ) -> str:
        """arXiv 구조화 입력(abstract+섹션)을 만들어 summarize_text로 위임."""
        return self.summarize_text(
            paper_info,
            self.prepare_summarization_input(paper_abstract, paper_full_content),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (기존 `TestSummarizeOne` + 새 `TestSummarizeText` 모두 통과)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/service.py tests/test_on_demand.py && git commit -m "feat: Service.summarize_text as unified summarization entry"
```

---

## Task 3: api/pdf.py (다운로드 + 추출)

**Files:** Create: `api/pdf.py`; Test: `tests/test_pdf.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pdf.py` 생성:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_pdf.py"`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.pdf'`

- [ ] **Step 3: Implement**

`api/pdf.py` 생성:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_pdf.py"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/pdf.py tests/test_pdf.py && git commit -m "feat: api/pdf.py download + pymupdf4llm text extraction"
```

---

## Task 4: api/resolvers.py 순수 헬퍼

**Files:** Create: `api/resolvers.py`; Test: `tests/test_resolvers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_resolvers.py` 생성:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_resolvers.py"`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.resolvers'`

- [ ] **Step 3: Implement (헬퍼만)**

`api/resolvers.py` 생성:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_resolvers.py"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/resolvers.py tests/test_resolvers.py && git commit -m "feat: resolvers pure helpers (extract_first_url, find_pdf_link, is_pdf_url)"
```

---

## Task 5: 리졸버 cascade (build_resolver)

**Files:** Modify: `api/resolvers.py`; Test: `tests/test_resolvers.py`

- [ ] **Step 1: Write the failing test**

`tests/test_resolvers.py`의 `if __name__` 위에 추가:

```python
from unittest import mock


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_resolvers.py"`
Expected: FAIL — `AttributeError: module 'api.resolvers' has no attribute 'build_resolver'` (또는 `_fetch_soup` 없음)

- [ ] **Step 3: Implement**

`api/resolvers.py` 끝에 추가:

```python
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
    return ResolvedPaper(title=(title or pdf_title(pdf) or pdf_url.rsplit("/", 1)[-1]),
                         url=pdf_url, text=text, source=source)


def _resolve_direct_pdf(url, *, source, on_progress):
    return _pdf_to_paper(url, title="", source=source, on_progress=on_progress)


def _resolve_via_html(url, *, on_progress):
    soup = _fetch_soup(url)
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    pdf_link = find_pdf_link(soup, url)
    if not pdf_link:
        return None
    return _pdf_to_paper(pdf_link, title=title, source="html", on_progress=on_progress)


def _openreview_pdf_url(url):
    pid = None
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        pid = m.group(1)
    return f"https://openreview.net/pdf?id={pid}" if pid else url


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
                return _resolve_direct_pdf(_openreview_pdf_url(url),
                                           source="openreview", on_progress=on_progress)
            if is_pdf_url(url):
                return _resolve_direct_pdf(url, source="pdf", on_progress=on_progress)
            return _resolve_via_html(url, on_progress=on_progress)  # omnivorous
        except Exception as e:
            logger.info(f"resolve failed for {url}: {e}")
            return None
    return resolve
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_resolvers.py"`
Expected: PASS (4개 cascade 테스트 포함 전부)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/resolvers.py tests/test_resolvers.py && git commit -m "feat: omnivorous resolve cascade (arxiv/openreview/direct-pdf/generic-html)"
```

---

## Task 6: process_mention 개정 (resolve + on_progress)

**Files:** Modify: `api/on_demand.py`; Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`에서 기존 `TestProcessMention` 클래스를 **교체**(fake resolve + on_progress 레코더):

```python
class _FakeResolved:
    def __init__(self, title, url, text):
        self.title, self.url, self.text = title, url, text


class TestProcessMention(unittest.TestCase):
    def _resolve_ok(self, on_progress_emits_downloading=True):
        def resolve(url, on_progress=lambda s: None):
            if on_progress_emits_downloading:
                on_progress("downloading")
            return _FakeResolved("Some Title", "https://arxiv.org/abs/2501.12345", "body text")
        return resolve

    def test_no_url_returns_guidance(self):
        from api.on_demand import process_mention
        seen = []
        r = process_mention("no link", cache=None, service=_FakeService(VALID_SUMMARY),
                            workspace=_FakeWorkspace(), resolve=self._resolve_ok(),
                            on_progress=seen.append)
        self.assertFalse(r["ok"])
        self.assertIn("arxiv", r["message"].lower())

    def test_ok_path_stage_order_and_meta(self):
        from api.on_demand import process_mention
        seen = []
        svc = _FakeService(VALID_SUMMARY)
        r = process_mention("see <https://arxiv.org/abs/2501.12345>", cache=None,
                            service=svc, workspace=_FakeWorkspace(),
                            resolve=self._resolve_ok(), on_progress=seen.append)
        self.assertTrue(r["ok"])
        self.assertEqual(seen, ["fetching", "downloading", "summarizing"])
        self.assertEqual(r["paper_url"], "https://arxiv.org/abs/2501.12345")
        self.assertEqual(r["paper_info"], "Some Title (https://arxiv.org/abs/2501.12345)")
        self.assertIn("Some Title", r["message"])

    def test_unsupported_returns_error(self):
        from api.on_demand import process_mention
        r = process_mention("https://x.org/p", cache=None, service=_FakeService(VALID_SUMMARY),
                            workspace=_FakeWorkspace(), resolve=lambda u, on_progress=None: None,
                            on_progress=lambda s: None)
        self.assertFalse(r["ok"])

    def test_empty_summary_returns_error(self):
        from api.on_demand import process_mention
        r = process_mention("https://arxiv.org/abs/2501.12345", cache=None,
                            service=_FakeService(""), workspace=_FakeWorkspace(),
                            resolve=self._resolve_ok(), on_progress=lambda s: None)
        self.assertFalse(r["ok"])
```

`_FakeService`의 메서드명을 `summarize_text`로 맞추기 위해, 기존 `_FakeService` 정의를 다음으로 **교체**:

```python
class _FakeService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def summarize_text(self, paper_info, text):
        self.calls.append(paper_info)
        return self.result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `process_mention()` 시그니처 불일치(`resolve`/`on_progress` 인자) 또는 단계 순서 불일치

- [ ] **Step 3: Implement**

`api/on_demand.py`의 `process_mention`과 `build_fetch_paper`를 **삭제**하고 다음으로 교체(파일 상단 import도 교체):

```python
from api.arxiv import get_paper_info
from api.resolvers import extract_first_url


_NO_URL_MSG = (
    "arxiv 등 논문 링크를 함께 멘션해 주세요 "
    "(예: @arxivbot https://arxiv.org/abs/2501.12345)"
)
_UNSUPPORTED_MSG = (
    "이 링크에서 논문을 가져오지 못했어요. 지원: arXiv, ACL, CVPR/ICCV, "
    "NeurIPS, ICML, OpenReview, AAAI, IJCAI, Interspeech, 직접 PDF 링크."
)


def resolve_thread_ts(event: dict) -> str:
    return event.get("thread_ts") or event["ts"]


def process_mention(text, *, cache, service, workspace, resolve,
                    on_progress=lambda s: None) -> dict:
    url = extract_first_url(text)
    if url is None:
        return {"ok": False, "message": _NO_URL_MSG, "paper_info": None, "paper_url": None}
    on_progress("fetching")
    resolved = resolve(url, on_progress=on_progress)
    if resolved is None or not resolved.text:
        return {"ok": False, "message": _UNSUPPORTED_MSG, "paper_info": None, "paper_url": None}
    paper_info = get_paper_info(resolved.url, resolved.title)
    on_progress("summarizing")
    summarization = service.summarize_text(paper_info, resolved.text)
    if not summarization:
        return {"ok": False, "message": "요약 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                "paper_info": None, "paper_url": None}
    message_content, _ = workspace.prepare_content(paper_info, "", summarization)
    return {"ok": True, "message": message_content,
            "paper_info": paper_info, "paper_url": resolved.url}
```

(주의: `resolve_thread_ts`는 유지. 기존 `from api.arxiv import parse_arxiv_ref, get_paper_title, get_paper_info` 줄과 `build_fetch_paper` 정의는 제거.)

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (단, `TestResolveThreadTs`는 그대로 통과해야 함)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/on_demand.py tests/test_on_demand.py && git commit -m "feat: process_mention uses resolve cascade + progress callback"
```

---

## Task 7: listener.py 진행 표시 + 리졸버 주입

**Files:** Modify: `listener.py`

- [ ] **Step 1: Implement (진행 표시 핸들러)**

`listener.py`의 `make_app`에서 import와 핸들러를 교체.

상단 import 블록에 추가/교체:
```python
from api.on_demand import process_mention, resolve_thread_ts
from api.resolvers import build_resolver
```
(`build_fetch_paper` import 줄 제거.)

`make_app` 안에서 `fetch_paper = build_fetch_paper(...)` 줄을 교체:
```python
    resolve = build_resolver(arxiv_client, cache)
    listener_channel_id = workspace_config["listener_channel_id"]
    app = App(token=workspace_config["slack_token"])

    STAGE = {
        "fetching": "🔄 논문 페이지 가져오는 중…",
        "downloading": "🔄 PDF 다운로드 중…",
        "summarizing": "🔄 AI가 요약하는 중…",
    }

    @app.event("app_mention")
    def handle_app_mention(event, client):
        channel = event.get("channel")
        if channel != listener_channel_id:
            logger.info(f"app_mention ignored: channel {channel} != listener channel {listener_channel_id}")
            return
        thread_ts = resolve_thread_ts(event)
        logger.info(f"app_mention in {channel}: {event.get('text')!r}")
        try:
            posted = client.chat_postMessage(
                channel=channel, text=STAGE["fetching"], thread_ts=thread_ts
            )
            ts = posted["ts"]
            last = {"text": STAGE["fetching"]}

            def on_progress(stage):
                msg = STAGE.get(stage)
                if msg and msg != last["text"]:
                    last["text"] = msg
                    client.chat_update(channel=channel, ts=ts, text=msg)

            result = process_mention(
                event.get("text", ""), cache=cache, service=service,
                workspace=workspace, resolve=resolve, on_progress=on_progress,
            )
            client.chat_update(channel=channel, ts=ts, text=result["message"])
            if result["ok"]:
                store = load_store()
                add_posted(
                    store, ts=ts, thread_ts=thread_ts, channel_id=channel,
                    workspace=workspace.workspace, paper_info=result["paper_info"],
                    paper_url=result["paper_url"], field="on-demand",
                    posted_at=datetime.now(timezone.utc).isoformat(),
                )
                save_store(store)
                logger.info(f"on-demand summary posted: {result['paper_info']}")
        except Exception as e:
            logger.error(f"app_mention handler error: {e}")
            try:
                client.chat_postMessage(channel=channel, text=f"처리 중 오류가 났어요: {e}",
                                        thread_ts=thread_ts)
            except Exception:
                pass

    return workspace, app
```

(주의: 함수 시작부의 `workspace = Workspace(...)`, `cache`, `arxiv_client`, `service` 구성 줄은 유지. `build_fetch_paper` 호출만 `build_resolver`로 교체.)

- [ ] **Step 2: import/구문 sanity**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python -c 'import listener'"` 후 `echo OK`
Expected: 오류 없음

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add listener.py && git commit -m "feat: listener staged progress (post then chat.update) + resolver"
```

---

## Task 8: prompts.py 일반화 + 아티팩트 내성

**Files:** Modify: `prompts.py`; Test: `tests/test_summary.py`

- [ ] **Step 1: Write the failing test**

`tests/test_summary.py`의 `TestSummarizationPromptGlossary` 클래스에 메서드 추가:

```python
    def test_prompt_is_source_agnostic_and_artifact_tolerant(self):
        from prompts import SYSTEM_PROMPT_SUMMARIZATION
        self.assertNotIn("arxiv paper", SYSTEM_PROMPT_SUMMARIZATION.lower())
        self.assertIn("아티팩트", SYSTEM_PROMPT_SUMMARIZATION)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_summary.py"`
Expected: FAIL (`arxiv paper` 잔존 / `아티팩트` 없음)

- [ ] **Step 3: Implement**

`prompts.py`의 `SYSTEM_PROMPT_SUMMARIZATION` 첫 줄을 교체:
```
Please analyze the paper and write a Korean AI-newsletter style summary in JSON.
```
그리고 `언어 규칙(중요):` 블록 끝(음차 금지 줄 다음)에 한 줄 추가:
```
- 입력에 PDF 추출 아티팩트(깨진 수식·하이픈 분리·열 섞임·머리말/꼬리말)가 섞일 수 있다. 깨진 기호·잡음은 무시하고 산문 의미로 요약한다.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_summary.py"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add prompts.py tests/test_summary.py && git commit -m "feat: source-agnostic prompt + PDF-artifact tolerance line"
```

---

## Task 9: 스모크 검증 + 전체 스위트

**Files:** Create: `tests/smoke_resolvers.py`

- [ ] **Step 1: 스모크 작성**

`tests/smoke_resolvers.py` 생성:

```python
"""호스트별 실제 URL을 resolve해 제목/텍스트 길이 출력 + PDF를 tmp에 저장(수동 확인).
사용: python tests\\smoke_resolvers.py
인자로 URL 하나만 줄 수도 있음: python tests\\smoke_resolvers.py <url>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.cache import CacheManager
from api.arxiv import ArxivClient
from api.resolvers import build_resolver

SAMPLES = {
    "arxiv": "https://arxiv.org/abs/1706.03762",
    "acl": "https://aclanthology.org/2023.acl-long.1/",
    "thecvf": "https://openaccess.thecvf.com/content/CVPR2023/html/"
              "Kirillov_Segment_Anything_CVPR_2023_paper.html",
    "neurips": "https://papers.nips.cc/paper_files/paper/2017/hash/"
               "3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
    "isca": "https://www.isca-archive.org/interspeech_2023/index.html",
}


def main():
    cache = CacheManager()
    resolve = build_resolver(ArxivClient(cache), cache)
    urls = {"arg": sys.argv[1]} if len(sys.argv) > 1 else SAMPLES
    for name, url in urls.items():
        print(f"\n=== {name}: {url}")
        try:
            r = resolve(url, on_progress=lambda s: print("  stage:", s))
        except Exception as e:
            print("  ERROR:", e); continue
        if r is None:
            print("  -> None (미지원/실패)"); continue
        print("  title:", r.title[:80])
        print("  source:", r.source, "| text len:", len(r.text))
        print("  text head:", r.text[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스모크 실행 (네트워크, arXiv는 OpenAI 비용 없음 — 요약 안 함)**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\smoke_resolvers.py"`
Expected: 각 출처에 대해 title이 차고 `text len`이 수천 이상. None이면 그 출처는 셀렉터 점검 필요(아래 Step 3).

- [ ] **Step 3: 실패 호스트 보정 (필요 시)**

`text len`이 작거나 None인 호스트는 그 페이지 HTML을 직접 확인해 `find_pdf_link`가 PDF를 못 찾는 경우다. 해당 호스트만 `api/resolvers.py`에 전용 분기를 추가한다(예: 페이지 구조상 PDF 링크가 `<meta name="citation_pdf_url">`에 있으면 그걸 우선 사용). 다음 헬퍼를 `find_pdf_link` 위에 추가하고 `_resolve_via_html`에서 먼저 시도:

```python
def _citation_pdf(soup, base_url):
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    return urljoin(base_url, meta["content"]) if meta and meta.get("content") else None
```
`_resolve_via_html`의 `pdf_link = find_pdf_link(soup, url)` 를
`pdf_link = _citation_pdf(soup, url) or find_pdf_link(soup, url)` 로 교체.
(대부분 학회 페이지가 `citation_pdf_url` 메타태그를 제공하므로 이 한 줄이 견고성을 크게 올린다.)
변경 후 Step 2 재실행해 해당 호스트 확인.

- [ ] **Step 4: PDF 직접 열어 확인 (요청한 검증)**

스모크가 받은 PDF가 맞는 논문인지 1~2개 직접 확인하려면, 임시로 한 URL의 PDF를 저장:
Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python -c \"from api.resolvers import _fetch_soup,_citation_pdf,find_pdf_link; from api.pdf import download_pdf; u='https://openaccess.thecvf.com/content/CVPR2023/html/Kirillov_Segment_Anything_CVPR_2023_paper.html'; s=_fetch_soup(u); l=_citation_pdf(s,u) or find_pdf_link(s,u); open('logs/_sample.pdf','wb').write(download_pdf(l)); print('saved', l)\""`
그 후 `logs/_sample.pdf`를 열어 해당 논문이 맞는지 눈으로 확인. 확인 후 삭제: `rm logs/_sample.pdf`.

- [ ] **Step 5: 전체 단위 스위트**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_resolvers.py; python tests\test_pdf.py; python tests\test_on_demand.py; python tests\test_summary.py; python tests\test_reactions.py"`
Expected: 모두 OK

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add tests/smoke_resolvers.py api/resolvers.py && git commit -m "test: smoke_resolvers for per-host verification; citation_pdf_url fallback"
```

- [ ] **Step 7: 라이브 검증 (사용자)**

리스너 재시작 후 멘션 테스트:
```powershell
Stop-ScheduledTask -TaskName arxivbot-listener
Start-ScheduledTask -TaskName arxivbot-listener
```
C0B79ULKNDV 채널에서 arXiv / ACL / CVPR / 직접 PDF 링크를 각각 멘션 → `가져오는 중🔄→PDF 다운로드 중🔄→AI 요약 중🔄→4섹션 요약` 단계 편집 확인.

---

## Self-Review 결과

- **Spec coverage**: PDF추출(T3) / 리졸버 cascade·잡식성·직접PDF·arXiv PDF폴백(T5) / summarize_text 통일(T2) / 진행표시(T6·T7) / 프롬프트 일반화·아티팩트(T8) / 의존성(T1) / 스모크·PDF 직접확인(T9) — 스펙 항목 매핑됨. OpenReview·citation_pdf_url로 venue 견고성 확보.
- **Placeholder scan**: 모든 코드/명령 구체값. "필요 시 보정"(T9 S3)은 실제 코드(`_citation_pdf`)와 절차를 동반하므로 placeholder 아님.
- **Type consistency**: `ResolvedPaper(title,url,text,source)`, `resolve(url, on_progress)`, `build_resolver(arxiv_client, cache)`, `summarize_text(paper_info, text)`, `process_mention(..., resolve, on_progress)`, 진행 stage 키(`fetching/downloading/summarizing`) 전 Task 일관.
- **주의**: T6에서 `_FakeService`를 `summarize_text`로, `process_mention` 시그니처를 `fetch_paper`→`resolve`로 바꾸므로, 기존 `tests/test_on_demand.py`의 `build_fetch_paper`/옛 `process_mention` 참조가 있으면 함께 제거(해당 테스트는 T6에서 교체됨).
