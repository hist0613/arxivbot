# On-demand 논문 요약 (@멘션) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slack에서 봇을 `@멘션 + arxiv 링크`로 호출하면 그 스레드에 배치와 동일한 4섹션 요약을 답글로 단다.

**Architecture:** Socket Mode 상시 리스너(`listener.py`)가 `app_mention` 이벤트를 받아, 순수 코어(`api/on_demand.py`)로 URL 정규화→조회→요약(캐시 재사용)→포맷을 수행하고, 결과를 스레드 답글로 게시한 뒤 리액션 store에 등록한다. 모델/프롬프트/요약 입력/캐시 키는 전부 배치와 동일.

**Tech Stack:** Python 3.11, slack-bolt(Socket Mode), slack-sdk, BeautifulSoup, OpenAI(gpt-5.4-nano), unittest.

---

## 실행 환경 메모 (모든 테스트 실행에 적용)

- 프로덕션/검증 Python은 **Windows PowerShell의 Python 3.11**이다 (WSL python3 아님 — `google.generativeai` 미설치로 `api.agent` import 실패).
- 테스트 실행 형식 (WSL bash에서):
  `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\<파일>.py"`
- git 명령은 WSL bash에서: `cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git ...`

## File Structure

| 파일 | 책임 |
|---|---|
| `api/arxiv.py` (수정) | `parse_arxiv_ref`(URL→정규형), `_extract_title`/`get_paper_title`(abs 페이지 제목) |
| `api/service.py` (수정) | `Service.summarize_one`(단건 요약 + 캐시 규칙) |
| `api/on_demand.py` (신규) | 순수 코어: `resolve_thread_ts`, `process_mention`, `build_fetch_paper` |
| `settings.py` (수정) | slack 워크스페이스 config에 `app_token` 키 |
| `listener.py` (신규, top-level) | Socket Mode bolt 앱 + `app_mention` 핸들러 wiring |
| `scripts/install_listener_task.ps1` (신규) | Windows Task Scheduler 등록 |
| `tests/test_on_demand.py` (신규) | 코어 단위 테스트 |
| `tests/smoke_listener.py` (신규) | 실제 API로 `process_mention` 수동 검증 |

---

## Task 1: `parse_arxiv_ref` — 임의 링크를 배치 정규형으로 변환

**Files:**
- Modify: `api/arxiv.py`
- Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py` 생성:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `ImportError: cannot import name 'parse_arxiv_ref'`

- [ ] **Step 3: Write minimal implementation**

`api/arxiv.py` 상단 `import time` 아래에 `import re` 추가하고, `get_paper_info` 함수 아래에 추가:

```python
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?"
    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})"
    r"(?:v\d+)?",
    re.IGNORECASE,
)


def parse_arxiv_ref(text: str) -> str | None:
    """임의 텍스트(/abs//pdf//html/, .pdf, vN, bare id, 구형 id)에서
    arXiv 논문을 찾아 배치와 동일한 정규형 URL로 변환한다.
    매칭 실패 시 None."""
    if not text:
        return None
    m = _ARXIV_ID_RE.search(text)
    if not m:
        return None
    return f"https://arxiv.org/abs/{m.group(1)}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (3 tests in TestParseArxivRef)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/arxiv.py tests/test_on_demand.py && git commit -m "feat: parse_arxiv_ref normalizes any arxiv link to batch abs form"
```

---

## Task 2: `get_paper_title` — abs 페이지에서 제목 추출

**Files:**
- Modify: `api/arxiv.py`
- Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`의 `TestParseArxivRef` 클래스 아래에 추가:

```python
class TestExtractTitle(unittest.TestCase):
    def test_strips_title_descriptor(self):
        from bs4 import BeautifulSoup
        from api.arxiv import _extract_title
        html = (
            '<h1 class="title mathjax">'
            '<span class="descriptor">Title:</span>'
            'Attention Is All You Need</h1>'
        )
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(_extract_title(soup), "Attention Is All You Need")

    def test_returns_empty_when_no_title(self):
        from bs4 import BeautifulSoup
        from api.arxiv import _extract_title
        soup = BeautifulSoup("<div>nope</div>", "html.parser")
        self.assertEqual(_extract_title(soup), "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `ImportError: cannot import name '_extract_title'`

- [ ] **Step 3: Write minimal implementation**

`api/arxiv.py`의 `parse_arxiv_ref` 아래에 추가:

```python
def _extract_title(soup) -> str:
    """arXiv abs 페이지의 <h1 class="title">에서 'Title:' 접두를 떼고 제목 반환."""
    h1 = soup.find("h1", class_="title")
    if h1 is None:
        return ""
    descriptor = h1.find("span", class_="descriptor")
    if descriptor:
        descriptor.decompose()
    return h1.text.strip()


def get_paper_title(paper_url: str) -> str:
    """abs 페이지를 받아 제목을 반환한다. get_paper_abstract와 동일한
    헤더/타임아웃/재시도 패턴을 따른다."""
    paper_page = None
    for trial in range(MAX_LLM_TRIALS):
        try:
            paper_page = requests.get(
                paper_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
            )
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            logger.info(e)
            time.sleep(trial * 30 + 15)
    if paper_page is None:
        return ""
    soup = BeautifulSoup(paper_page.text, "html.parser")
    return _extract_title(soup)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (TestExtractTitle 2 tests + 기존 통과)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/arxiv.py tests/test_on_demand.py && git commit -m "feat: get_paper_title extracts title from arxiv abs page"
```

---

## Task 3: `Service.summarize_one` — 단건 요약 + 캐시 규칙

**Files:**
- Modify: `api/service.py`
- Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`에 추가 (페이크 cache/agent 사용):

```python
class _FakeCache:
    def __init__(self, summarizations=None):
        self.paper_summarizations = dict(summarizations or {})
        self.updated = []

    def has_paper_summarization(self, paper_info):
        return (
            paper_info in self.paper_summarizations
            and self.paper_summarizations[paper_info] != ""
        )

    def update_paper_summarizations(self, paper_info, summarization):
        self.paper_summarizations[paper_info] = summarization
        self.updated.append(paper_info)


class _FakeAgent:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.model_name = "fake"

    def summarize(self, content):
        self.calls += 1
        return self.result


class TestSummarizeOne(unittest.TestCase):
    def _service(self, agent, cache):
        from api.service import Service
        from api.agent import Encoder
        from settings import MODEL
        return Service(arxiv=None, agent=agent, encoder=Encoder(MODEL), cache=cache)

    def test_cache_hit_skips_agent(self):
        cache = _FakeCache({"P (url)": '{"Core Contribution": "x"}'})
        agent = _FakeAgent('{"Core Contribution": "y"}')
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, '{"Core Contribution": "x"}')
        self.assertEqual(agent.calls, 0)

    def test_cache_miss_calls_agent_and_stores(self):
        cache = _FakeCache()
        agent = _FakeAgent('{"Core Contribution": "y"}')
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, '{"Core Contribution": "y"}')
        self.assertEqual(agent.calls, 1)
        self.assertEqual(cache.paper_summarizations["P (url)"], out)

    def test_empty_result_not_cached(self):
        cache = _FakeCache()
        agent = _FakeAgent("")
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, "")
        self.assertNotIn("P (url)", cache.paper_summarizations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `AttributeError: 'Service' object has no attribute 'summarize_one'`

- [ ] **Step 3: Write minimal implementation**

`api/service.py`의 `Service` 클래스 안, `prepare_summarization_input` 위에 추가:

```python
    def summarize_one(
        self, paper_info: str, paper_abstract: str, paper_full_content
    ) -> str:
        """단건 요약. 캐시에 있으면 재사용, 없으면 배치와 동일 입력으로 요약 후
        비어 있지 않을 때만 캐시에 저장한다."""
        if self.cache.has_paper_summarization(paper_info):
            return self.cache.paper_summarizations[paper_info]
        summarization_input = self.prepare_summarization_input(
            paper_abstract, paper_full_content
        )
        summarization = self.agent.summarize(summarization_input)
        if summarization:
            self.cache.update_paper_summarizations(paper_info, summarization)
        return summarization
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (TestSummarizeOne 3 tests + 기존 통과)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/service.py tests/test_on_demand.py && git commit -m "feat: Service.summarize_one with batch-identical cache rules"
```

---

## Task 4: `api/on_demand.py` — 순수 코어 (resolve_thread_ts, process_mention, build_fetch_paper)

**Files:**
- Create: `api/on_demand.py`
- Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`에 추가:

```python
class _FakeWorkspace:
    workspace = "seungtaek-lab"

    def prepare_content(self, paper_info, paper_comment, paper_summarization):
        msg = f"*{paper_info}*\n{paper_summarization}"
        return msg, msg


class _FakeService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def summarize_one(self, paper_info, abstract, full_content):
        self.calls.append(paper_info)
        return self.result


class TestResolveThreadTs(unittest.TestCase):
    def test_uses_thread_ts_when_present(self):
        from api.on_demand import resolve_thread_ts
        self.assertEqual(
            resolve_thread_ts({"ts": "1.1", "thread_ts": "9.9"}), "9.9"
        )

    def test_falls_back_to_ts(self):
        from api.on_demand import resolve_thread_ts
        self.assertEqual(resolve_thread_ts({"ts": "1.1"}), "1.1")


class TestProcessMention(unittest.TestCase):
    def _fetch_ok(self, url):
        return ("Some Title", "an abstract", "")

    def test_no_url_returns_guidance(self):
        from api.on_demand import process_mention
        result = process_mention(
            "no link here",
            cache=None,
            service=_FakeService('{"x": 1}'),
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertFalse(result["ok"])
        self.assertIn("arxiv", result["message"].lower())

    def test_ok_path_builds_message_and_meta(self):
        from api.on_demand import process_mention
        svc = _FakeService('{"Core Contribution": "c"}')
        result = process_mention(
            "see https://arxiv.org/pdf/2501.12345v2",
            cache=None,
            service=svc,
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["paper_url"], "https://arxiv.org/abs/2501.12345")
        self.assertEqual(
            result["paper_info"],
            "Some Title (https://arxiv.org/abs/2501.12345)",
        )
        self.assertIn("Some Title", result["message"])

    def test_empty_summary_returns_error(self):
        from api.on_demand import process_mention
        result = process_mention(
            "https://arxiv.org/abs/2501.12345",
            cache=None,
            service=_FakeService(""),
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertFalse(result["ok"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.on_demand'`

- [ ] **Step 3: Write minimal implementation**

`api/on_demand.py` 생성:

```python
"""On-demand(@멘션) 요약의 순수 코어 — Slack/네트워크에서 분리되어 단위 테스트 가능.

listener.py가 이 모듈의 process_mention/resolve_thread_ts/build_fetch_paper를
실제 의존성(arxiv 조회, Service, Workspace)과 함께 wiring한다.
"""
from api.arxiv import parse_arxiv_ref, get_paper_title, get_paper_info


_NO_URL_MSG = (
    "arxiv 링크를 함께 멘션해 주세요 "
    "(예: @arxivbot https://arxiv.org/abs/2501.12345)"
)


def resolve_thread_ts(event: dict) -> str:
    """멘션이 스레드 안이면 그 스레드, 아니면 멘션 메시지 자체에 답글."""
    return event.get("thread_ts") or event["ts"]


def process_mention(text, *, cache, service, workspace, fetch_paper) -> dict:
    """멘션 텍스트를 받아 요약 결과 dict를 반환한다.

    반환: {"ok": bool, "message": str,
           "paper_info": str|None, "paper_url": str|None}
    fetch_paper(url) -> (title, abstract, full_content)  (주입)
    """
    url = parse_arxiv_ref(text)
    if url is None:
        return {"ok": False, "message": _NO_URL_MSG,
                "paper_info": None, "paper_url": None}
    try:
        title, abstract, full_content = fetch_paper(url)
    except Exception as e:
        return {"ok": False, "message": f"논문을 가져오지 못했어요: {e}",
                "paper_info": None, "paper_url": None}
    if not abstract:
        return {"ok": False, "message": "초록을 찾지 못했어요. 링크를 확인해 주세요.",
                "paper_info": None, "paper_url": None}

    paper_info = get_paper_info(url, title)
    summarization = service.summarize_one(paper_info, abstract, full_content)
    if not summarization:
        return {"ok": False,
                "message": "요약 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                "paper_info": None, "paper_url": None}

    message_content, _ = workspace.prepare_content(paper_info, "", summarization)
    return {"ok": True, "message": message_content,
            "paper_info": paper_info, "paper_url": url}


def build_fetch_paper(arxiv_client, cache):
    """캐시를 재사용하며 (title, abstract, full_content)를 가져오는 클로저 생성.
    abstract/full_content는 배치와 동일하게 캐시에 채워 넣는다."""
    def fetch_paper(url):
        title = get_paper_title(url)
        paper_info = get_paper_info(url, title)

        cached_abs = cache.paper_abstracts.get(paper_info) if hasattr(
            cache.paper_abstracts, "get") else None
        if cached_abs:
            abstract = cached_abs
        else:
            abstract = arxiv_client.get_paper_abstract(url)
            cache.update_paper_abstracts(paper_info, abstract)

        cached_full = cache.paper_full_contents.get(paper_info) if hasattr(
            cache.paper_full_contents, "get") else None
        if cached_full:
            full_content = cached_full
        else:
            full_content = ""
            try:
                html_link = arxiv_client.get_html_experimental_link(url)
                if html_link != "Link not found":
                    full_content = arxiv_client.get_paper_full_content(html_link)
                    cache.update_paper_full_contents(paper_info, full_content)
            except Exception:
                full_content = ""

        return title, abstract, full_content
    return fetch_paper
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS (전체 테스트 통과)

- [ ] **Step 5: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add api/on_demand.py tests/test_on_demand.py && git commit -m "feat: on_demand pure core (resolve_thread_ts, process_mention, build_fetch_paper)"
```

---

## Task 5: `settings.py` — slack 워크스페이스에 `app_token` 추가

**Files:**
- Modify: `settings.py:38-45`
- Test: `tests/test_on_demand.py`

- [ ] **Step 1: Write the failing test**

`tests/test_on_demand.py`에 추가:

```python
class TestSettingsAppToken(unittest.TestCase):
    def test_active_slack_workspace_has_app_token_key(self):
        import settings
        active_slack = [
            c for c in settings.WORKSPACE_CONFIGS
            if c.get("service_type") == "slack"
        ]
        self.assertTrue(active_slack, "활성 slack 워크스페이스가 없음")
        for c in active_slack:
            self.assertIn("app_token", c, msg=f"app_token 누락: {c['workspace']}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: FAIL — `AssertionError: app_token 누락: seungtaek-lab`

- [ ] **Step 3: Write minimal implementation**

`settings.py`의 활성 seungtaek-lab slack 블록(38-45행)에 `app_token` 한 줄 추가:

```python
    {
        "service_type": "slack",
        "workspace": "seungtaek-lab",
        "allowed_channel": "arxivbot",          # Slack 채널 이름
        "allowed_channel_id": "C0B7V0V8U7N",        # Slack 채널 ID
        "slack_token": os.getenv("SLACK_TOKEN_SEUNGTAEK_LAB"),
        "app_token": os.getenv("SLACK_APP_TOKEN_SEUNGTAEK_LAB"),  # Socket Mode(xapp-)
        "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI", "cs.RO", "cs.MA"],
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\test_on_demand.py"`
Expected: PASS

- [ ] **Step 5: `.env`에 app-level 토큰 추가 (사용자 수행, 값 없으면 None이라 테스트는 통과)**

`.env`에 다음 줄을 추가한다 (값은 Task 8의 Slack 앱 설정에서 발급):

```
SLACK_APP_TOKEN_SEUNGTAEK_LAB=xapp-...
```

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add settings.py tests/test_on_demand.py && git commit -m "feat: add app_token (Socket Mode) to slack workspace config"
```

---

## Task 6: `listener.py` — Socket Mode 리스너 wiring

**Files:**
- Create: `listener.py`

(이 파일은 Slack 연결/이벤트 wiring으로 단위 테스트 대신 Task 8 smoke로 검증한다. 로직은 모두 Task 1–4의 테스트된 코어에 있다.)

- [ ] **Step 1: 구현 작성**

`listener.py` 생성:

```python
"""On-demand(@멘션) 논문 요약 Socket Mode 리스너.

부팅 시 자동 실행(Task Scheduler) + 죽으면 재시작 전제로 상시 동작한다.
이벤트 처리 로직은 api.on_demand의 테스트된 코어를 사용한다.
"""
from datetime import datetime, timezone

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.reactions import load_store, save_store, add_posted
from api.on_demand import process_mention, resolve_thread_ts, build_fetch_paper
from api.logger import logger
from settings import WORKSPACE_CONFIGS, MODEL


def make_app(workspace_config: dict):
    workspace = Workspace(workspace_config)
    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(
        arxiv_client, AutoAgent.from_model_name(MODEL), Encoder(MODEL), cache
    )
    fetch_paper = build_fetch_paper(arxiv_client, cache)
    app = App(token=workspace_config["slack_token"])

    @app.event("app_mention")
    def handle_app_mention(event, client):
        channel = event.get("channel")
        # 지정 채널 밖 멘션은 무시 (배치 allowed_channel과 동일 스코프)
        if channel != workspace.allowed_channel_id:
            return
        thread_ts = resolve_thread_ts(event)
        try:
            result = process_mention(
                event.get("text", ""),
                cache=cache,
                service=service,
                workspace=workspace,
                fetch_paper=fetch_paper,
            )
            reply = client.chat_postMessage(
                channel=channel, text=result["message"], thread_ts=thread_ts
            )
            if result["ok"]:
                store = load_store()
                add_posted(
                    store,
                    ts=reply["ts"],
                    thread_ts=thread_ts,
                    channel_id=channel,
                    workspace=workspace.workspace,
                    paper_info=result["paper_info"],
                    paper_url=result["paper_url"],
                    field="on-demand",
                    posted_at=datetime.now(timezone.utc).isoformat(),
                )
                save_store(store)
                logger.info(f"on-demand summary posted: {result['paper_info']}")
        except Exception as e:
            logger.error(f"app_mention handler error: {e}")
            try:
                client.chat_postMessage(
                    channel=channel,
                    text=f"처리 중 오류가 났어요: {e}",
                    thread_ts=thread_ts,
                )
            except Exception:
                pass

    return workspace, app


def run():
    handlers = []
    for cfg in WORKSPACE_CONFIGS:
        if cfg.get("service_type") != "slack":
            continue
        workspace, app = make_app(cfg)
        handlers.append(SocketModeHandler(app, cfg["app_token"]))
        logger.info(f"Listener ready for {workspace.workspace_name}")

    if not handlers:
        logger.error("No slack workspace configured for listener.")
        return
    # 여러 워크스페이스: 마지막만 foreground로 blocking, 나머지는 background 연결
    for h in handlers[:-1]:
        h.connect()
    handlers[-1].start()  # blocks forever (자체 재연결 포함)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: import/구문 sanity 체크 (실제 연결 없이 import만)**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python -c 'import listener; print(\"import OK\")'"`
Expected: `import OK` (토큰이 None이어도 import 단계에선 통과)

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add listener.py && git commit -m "feat: listener.py Socket Mode app_mention handler"
```

---

## Task 7: `scripts/install_listener_task.ps1` — 부팅 자동 실행 등록

**Files:**
- Create: `scripts/install_listener_task.ps1`

- [ ] **Step 1: 구현 작성**

`scripts/install_listener_task.ps1` 생성:

```powershell
# arxivbot on-demand 리스너를 로그온 시 자동 실행하도록 Task Scheduler에 등록.
# 실행: 관리자 PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1
$ErrorActionPreference = "Stop"

$repo     = "C:\Users\hist0\Dropbox\develop\arxivbot_new"
$python   = "python"   # 배치 main.py와 동일한 Python 3.11. 필요하면 절대경로로.
$taskName = "arxivbot-listener"

$action = New-ScheduledTaskAction -Execute $python -Argument "listener.py" -WorkingDirectory $repo
# 로그온 시 시작(사용자 컨텍스트라 .env/네트워크/PATH가 안정적).
# 사용자 로그인 없이 부팅만으로 띄우려면 -AtStartup 으로 바꾸고 -User/-Password 또는 SYSTEM 사용.
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 9999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force `
    -Description "arxivbot on-demand summary Socket Mode listener"

Write-Host "Registered '$taskName' (AtLogOn, restart every 1 min on failure)."
Write-Host "Start now:  Start-ScheduledTask -TaskName $taskName"
```

- [ ] **Step 2: 구문 검증 (등록하지 않고 파싱만)**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; \$null = [ScriptBlock]::Create((Get-Content -Raw scripts\install_listener_task.ps1)); Write-Host 'parse OK'"`
Expected: `parse OK`

- [ ] **Step 3: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add scripts/install_listener_task.ps1 && git commit -m "feat: Task Scheduler install script for listener"
```

---

## Task 8: smoke 테스트 + Slack 앱 설정 체크리스트

**Files:**
- Create: `tests/smoke_listener.py`

- [ ] **Step 1: smoke 작성**

`tests/smoke_listener.py` 생성:

```python
"""실제 API로 process_mention 1건 검증(Slack 게시 없음). 수동 실행(비용 발생).

사용: python tests\\smoke_listener.py https://arxiv.org/abs/<id>
인자 없으면 기본 URL 사용.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.on_demand import process_mention, build_fetch_paper
from settings import WORKSPACE_CONFIGS, MODEL

EXPECTED_KEYS = ["Prior Approaches", "Core Contribution",
                 "Technical Challenges", "Empirical Impact"]


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://arxiv.org/abs/1706.03762"
    cfg = next(c for c in WORKSPACE_CONFIGS if c.get("service_type") == "slack")
    workspace = Workspace(cfg)
    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(arxiv_client, AutoAgent.from_model_name(MODEL),
                      Encoder(MODEL), cache)
    fetch_paper = build_fetch_paper(arxiv_client, cache)

    result = process_mention(
        f"@arxivbot {url}",
        cache=cache, service=service, workspace=workspace,
        fetch_paper=fetch_paper,
    )
    print("ok:", result["ok"])
    print("paper_info:", result["paper_info"])
    print("message:\n", result["message"][:1200])
    assert result["ok"], result["message"]
    assert result["paper_url"].startswith("https://arxiv.org/abs/")
    # 메시지에 4섹션 키가 모두 들어있는지
    for k in EXPECTED_KEYS:
        assert k in result["message"], f"섹션 누락: {k}"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke 실행 (실제 OpenAI 비용 발생)**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python tests\smoke_listener.py"`
Expected: 마지막 줄 `SMOKE PASS`, message에 4섹션 출력

- [ ] **Step 3: Slack 앱 설정 (사용자가 api.slack.com에서 1회 수행)**

체크리스트:
- [ ] App 설정 → **Socket Mode** 활성화 → app-level token(`xapp-`, scope `connections:write`) 생성 → `.env`의 `SLACK_APP_TOKEN_SEUNGTAEK_LAB`에 입력
- [ ] **Event Subscriptions** → Bot Events에 **`app_mention`** 추가
- [ ] **OAuth & Permissions** scope 확인: `app_mentions:read`, `chat:write`(보유), `reactions:read`(보유)
- [ ] 앱을 워크스페이스에 **재설치(reinstall)** 하고 `#arxivbot` 채널에 봇 초대

- [ ] **Step 4: 실제 리스너 1회 수동 가동 확인**

Run: `powershell.exe -NoProfile -Command "cd 'C:\Users\hist0\Dropbox\develop\arxivbot_new'; python listener.py"`
그 후 `#arxivbot` 채널에서 `@arxivbot https://arxiv.org/abs/1706.03762` 멘션 → 스레드에 4섹션 요약 답글 확인. 확인되면 Ctrl+C로 종료.

- [ ] **Step 5: 부팅 자동 실행 등록 (사용자)**

Run(관리자 PowerShell): `powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1`
그 후: `Start-ScheduledTask -TaskName arxivbot-listener`

- [ ] **Step 6: Commit**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new && git add tests/smoke_listener.py && git commit -m "test: smoke_listener exercises process_mention end-to-end"
```

---

## Self-Review 결과

- **Spec coverage**: ① URL 정규화(Task 1) ② abs 제목(Task 2) ③ 캐시 규칙 요약(Task 3) ④ thread_ts/멘션 처리/캐시 재사용 fetch(Task 4) ⑤ app_token config(Task 5) ⑥ Socket Mode 리스너+리액션 등록+에러 처리(Task 6) ⑦ 부팅 자동 실행(Task 7) ⑧ smoke+Slack 설정(Task 8) — 스펙 전 항목 매핑됨.
- **Placeholder scan**: 모든 코드/명령 구체값으로 작성, TBD 없음.
- **Type consistency**: `process_mention` 반환 dict 키(`ok/message/paper_info/paper_url`), `summarize_one` 시그니처, `fetch_paper(url) -> (title, abstract, full_content)`, `field="on-demand"` 전 Task 일관.
- **배치 일관성**: `parse_arxiv_ref`→`get_paper_info`로 배치와 동일 `paper_info` 키 생성, `summarize_one`이 동일 캐시·입력 사용 — 캐시 히트/리액션 키 일치 보장.
