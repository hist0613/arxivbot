# 리스너 에이전트화 + 저장소 정리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** summaries 산출물을 폐지하고, 캐시를 SQLite로 옮기고, on-demand 리스너를 도구 쓰는 에이전트로 바꾼다.

**Architecture:** 저장소는 `api/store.py`의 SQLite 한 파일로 통일하고 `CacheManager`는 겉모습만 유지한 채 속을 갈아끼운다. 리스너는 멘션을 받아 스레드 문맥을 토큰 예산 안에서 조립한 뒤 OpenAI Responses API 루프를 돌리며, 논문 요약은 게시까지 스스로 끝내는 부수효과 도구로 노출한다. 에이전트가 실패하면 지금의 결정론적 요약 경로로 폴백한다.

**Tech Stack:** Python 3.11(Windows), sqlite3, openai SDK 2.41.1(Responses API), slack_bolt, unittest

**설계 문서:** `docs/superpowers/specs/2026-07-26-agentic-listener-design.md`

## Global Constraints

- 테스트 실행은 Windows PowerShell의 Python 3.11: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python -m unittest discover -s tests -p 'test_*.py'"`. WSL python3에는 `google.generativeai`가 없어 `api.agent` import가 막힌다.
- 작업 위치는 worktree `/mnt/c/Users/hist0/worktrees/arxivbot-agentic`. 원본 체크아웃(`Dropbox/develop/arxivbot_new`)은 main에 둔 채 배치·리스너가 계속 돈다.
- PR은 둘로 나눈다. Task 1 = `chore/drop-summaries`, Task 2 이후 = `feat/agentic-listener`(앞 브랜치에서 분기).
- 커밋 scope는 `arxivbot_new`, 메시지는 한국어. 커밋 말미에 `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- 요약 모델은 기존 `MODEL = "gpt-5.4-nano"` 그대로. 에이전트 모델은 `AGENT_MODEL = "gpt-5.4"`(신규 설정).
- Responses API 내장 검색 도구 타입은 `{"type": "web_search"}`(SDK 2.41.1 `WebSearchToolParam` 확인함).
- 사람 눈에 보이는 문구는 한국어. 기존 진행 표시 문구(`🔄 논문 페이지 가져오는 중…` 등)를 그대로 재사용한다.
- 캐시 DB 경로는 `cache/arxivbot.db`. `cache/`는 이미 `.gitignore`의 `*.pickle`로 부분 제외되어 있으니 `.gitignore`에 `cache/*.db*`를 추가한다.

---

## Task 1: summaries 폐지 (PR #1)

**Files:**
- Modify: `api/workspace.py` (`save_summaries` 삭제, `TODAY_SUMMARIES_DIR` import 삭제)
- Modify: `main.py` (`save_summaries` 호출, git add/commit/push 블록, `import git`, `SUMMARIES_DIR` import 삭제)
- Modify: `settings.py` (`SUMMARIES_DIR`, `TODAY_SUMMARIES_DIR` 삭제)
- Modify: `README.md` (summaries 언급 정리)
- Delete: `summaries/` 전체

**Interfaces:**
- Consumes: 없음
- Produces: 없음 (제거만)

- [ ] **Step 1: 사용처 전수 확인**

```bash
cd /mnt/c/Users/hist0/worktrees/arxivbot-agentic
grep -rn "SUMMARIES_DIR\|save_summaries" --include=*.py --include=*.md .
```

기대: `main.py`, `settings.py`, `api/workspace.py`, `README.md`, 그리고 과거 설계 문서(수정하지 않음)만 나온다.

- [ ] **Step 2: 코드에서 제거**

`api/workspace.py`의 `save_summaries` 메서드 전체와 `TODAY_SUMMARIES_DIR` import를 지운다. `main.py`에서 `workspace.save_summaries(threads)` 한 줄과 아래 블록을 지운다.

```python
    repo = git.Repo(REPO_DIR)
    repo.git.add(SUMMARIES_DIR)

    if repo.is_dirty():
        repo.git.commit("-m", f"\"Update summaries: {time.strftime('%Y-%m-%d')}\"")
        repo.git.push()
    else:
        logger.info("No changes to commit.")
```

같이 쓰이지 않게 되는 `import git`, `import time`(다른 데서 안 쓰면), `REPO_DIR`/`SUMMARIES_DIR` import도 지운다. `settings.py`에서 `SUMMARIES_DIR`, `TODAY_SUMMARIES_DIR` 두 줄을 지운다(`REPO_DIR`은 다른 경로 조립에 쓰이면 남긴다).

- [ ] **Step 3: 디렉토리 삭제**

```bash
git rm -r --quiet summaries
```

- [ ] **Step 4: 테스트 전량 통과 확인**

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python -m unittest discover -s tests -p 'test_*.py' 2>&1 | Select-Object -Last 5"`
기대: `Ran 76 tests ... OK` (베이스라인과 동일)

- [ ] **Step 5: 배치 임포트 확인**

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python -c 'import main; print(\"ok\")'"`
기대: `ok` (지운 import 흔적이 남아 있으면 여기서 걸린다)

- [ ] **Step 6: 커밋과 PR**

```bash
git add -A
git commit -m "chore(arxivbot_new): summaries 산출물 폐지"   # 본문은 아래 내용으로
git push -u origin chore/drop-summaries
gh pr create --base main --head chore/drop-summaries --title "chore(arxivbot_new): summaries 산출물 폐지" --body "..."
```

커밋 본문에 담을 것: 필드별 파일이 전부 같은 내용이던 점, 새 논문 0건 실행이 그날 파일을 빈 파일로 덮어쓰던 점(286개 중 227개가 0바이트), 배치가 더 이상 저장소를 건드리지 않는다는 점, 과거 내용은 히스토리에 남는다는 점.

---

## Task 2: SQLite 저장소 (`api/store.py`)

**Files:**
- Create: `api/store.py`
- Create: `tests/test_store.py`
- Modify: `settings.py` (`CACHE_DB_PATH` 추가)
- Modify: `.gitignore` (`cache/*.db*`)

**Interfaces:**
- Consumes: 없음
- Produces: `Store` 클래스 — `get_abstract(paper_info) -> str`, `put_abstract(paper_info, text)`, `get_full_content(paper_info) -> dict | str`, `put_full_content(paper_info, value)`, `get_summary(paper_info) -> str`, `put_summary(paper_info, text, schema_version, model)`, `has_summary(paper_info) -> bool`, `get_page(url) -> dict | None`, `put_page(url, title, text)`, `get_digest(thread_ts) -> dict | None`, `put_digest(thread_ts, digest, covered_until_ts)`, `count(table) -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_store.py`:

```python
import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStore(unittest.TestCase):
    def setUp(self):
        from api.store import Store
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_missing_key_returns_empty(self):
        self.assertEqual(self.store.get_abstract("없는 논문"), "")
        self.assertFalse(self.store.has_summary("없는 논문"))
        self.assertIsNone(self.store.get_page("https://x.test"))

    def test_upsert_overwrites_only_that_key(self):
        self.store.put_abstract("A", "첫 초록")
        self.store.put_abstract("B", "다른 초록")
        self.store.put_abstract("A", "고친 초록")
        self.assertEqual(self.store.get_abstract("A"), "고친 초록")
        self.assertEqual(self.store.get_abstract("B"), "다른 초록")
        self.assertEqual(self.store.count("abstracts"), 2)

    def test_full_content_roundtrips_dict(self):
        value = {"1": {"title": "Intro", "content": "본문"}}
        self.store.put_full_content("A", value)
        self.assertEqual(self.store.get_full_content("A"), value)

    def test_full_content_accepts_plain_text(self):
        self.store.put_full_content("B", "그냥 텍스트")
        self.assertEqual(self.store.get_full_content("B"), "그냥 텍스트")

    def test_summary_keeps_metadata(self):
        self.store.put_summary("A", "요약", schema_version="4sections", model="gpt-5.4-nano")
        self.assertEqual(self.store.get_summary("A"), "요약")
        self.assertTrue(self.store.has_summary("A"))

    def test_second_connection_sees_writes(self):
        from api.store import Store
        other = Store(self.tmp.name)
        self.store.put_abstract("A", "초록")
        other.put_abstract("B", "다른 초록")
        self.assertEqual(other.get_abstract("A"), "초록")
        self.assertEqual(self.store.get_abstract("B"), "다른 초록")
        other.close()

    def test_page_and_digest(self):
        self.store.put_page("https://x.test", "제목", "본문")
        page = self.store.get_page("https://x.test")
        self.assertEqual(page["title"], "제목")
        self.store.put_digest("1.1", "요지", covered_until_ts="2.2")
        digest = self.store.get_digest("1.1")
        self.assertEqual(digest["covered_until_ts"], "2.2")
```

- [ ] **Step 2: 실패 확인**

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python -m unittest tests.test_store -v 2>&1 | Select-Object -Last 5"`
기대: `ModuleNotFoundError: No module named 'api.store'`

- [ ] **Step 3: 구현**

`api/store.py`:

```python
"""SQLite 저장소. 통째 덮어쓰기 대신 키 단위 upsert로 캐시를 관리한다."""
import json
import sqlite3
import threading
import time

from settings import CACHE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS abstracts (
    paper_info TEXT PRIMARY KEY, text TEXT NOT NULL, fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS full_contents (
    paper_info TEXT PRIMARY KEY, text TEXT NOT NULL, fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS summaries (
    paper_info TEXT PRIMARY KEY, text TEXT NOT NULL,
    schema_version TEXT, model TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY, title TEXT, text TEXT NOT NULL, fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS thread_digests (
    thread_ts TEXT PRIMARY KEY, digest TEXT NOT NULL,
    covered_until_ts TEXT, updated_at REAL NOT NULL);
"""


class Store:
    def __init__(self, path: str = CACHE_DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self):
        self._conn.close()

    def _one(self, sql: str, *params):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, *params):
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def get_abstract(self, paper_info: str) -> str:
        row = self._one("SELECT text FROM abstracts WHERE paper_info=?", paper_info)
        return row["text"] if row else ""

    def put_abstract(self, paper_info: str, text: str):
        self._write(
            "INSERT INTO abstracts(paper_info, text, fetched_at) VALUES(?,?,?) "
            "ON CONFLICT(paper_info) DO UPDATE SET text=excluded.text, fetched_at=excluded.fetched_at",
            paper_info, text, time.time())

    def get_full_content(self, paper_info: str):
        row = self._one("SELECT text FROM full_contents WHERE paper_info=?", paper_info)
        if not row:
            return ""
        try:
            return json.loads(row["text"])
        except (ValueError, TypeError):
            return row["text"]

    def put_full_content(self, paper_info: str, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        self._write(
            "INSERT INTO full_contents(paper_info, text, fetched_at) VALUES(?,?,?) "
            "ON CONFLICT(paper_info) DO UPDATE SET text=excluded.text, fetched_at=excluded.fetched_at",
            paper_info, text, time.time())

    def get_summary(self, paper_info: str) -> str:
        row = self._one("SELECT text FROM summaries WHERE paper_info=?", paper_info)
        return row["text"] if row else ""

    def has_summary(self, paper_info: str) -> bool:
        return bool(self.get_summary(paper_info))

    def put_summary(self, paper_info: str, text: str, schema_version: str = "", model: str = ""):
        self._write(
            "INSERT INTO summaries(paper_info, text, schema_version, model, created_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(paper_info) DO UPDATE SET "
            "text=excluded.text, schema_version=excluded.schema_version, "
            "model=excluded.model, created_at=excluded.created_at",
            paper_info, text, schema_version, model, time.time())

    def get_page(self, url: str):
        row = self._one("SELECT title, text, fetched_at FROM pages WHERE url=?", url)
        return dict(row) if row else None

    def put_page(self, url: str, title: str, text: str):
        self._write(
            "INSERT INTO pages(url, title, text, fetched_at) VALUES(?,?,?,?) "
            "ON CONFLICT(url) DO UPDATE SET title=excluded.title, "
            "text=excluded.text, fetched_at=excluded.fetched_at",
            url, title, text, time.time())

    def get_digest(self, thread_ts: str):
        row = self._one(
            "SELECT digest, covered_until_ts, updated_at FROM thread_digests WHERE thread_ts=?",
            thread_ts)
        return dict(row) if row else None

    def put_digest(self, thread_ts: str, digest: str, covered_until_ts: str):
        self._write(
            "INSERT INTO thread_digests(thread_ts, digest, covered_until_ts, updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(thread_ts) DO UPDATE SET digest=excluded.digest, "
            "covered_until_ts=excluded.covered_until_ts, updated_at=excluded.updated_at",
            thread_ts, digest, covered_until_ts, time.time())

    def count(self, table: str) -> int:
        if table not in {"abstracts", "full_contents", "summaries", "pages", "thread_digests"}:
            raise ValueError(f"unknown table: {table}")
        return self._one(f"SELECT COUNT(*) AS n FROM {table}")["n"]
```

`settings.py`에 추가: `CACHE_DB_PATH = os.path.join(REPO_DIR, "cache", "arxivbot.db")`. `.gitignore`에 `cache/*.db*` 추가.

- [ ] **Step 4: 통과 확인**

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python -m unittest tests.test_store 2>&1 | Select-Object -Last 5"`
기대: `OK`

- [ ] **Step 5: 커밋**

```bash
git add api/store.py tests/test_store.py settings.py .gitignore
git commit -m "feat(arxivbot_new): 캐시용 SQLite 저장소 추가"
```

---

## Task 3: CacheManager 속 교체 + 마이그레이션

**Files:**
- Modify: `api/cache.py` (pickle 로직 제거, `Store` 위임)
- Create: `scripts/migrate_cache_to_sqlite.py`
- Create: `tests/test_cache_manager.py`

**Interfaces:**
- Consumes: Task 2의 `Store`
- Produces: `CacheManager(store=None)` — 기존 속성 `paper_abstracts`, `paper_full_contents`, `paper_summarizations`(각각 `[key]`, `in`, `.get(key, default)` 지원)와 메서드 `has_paper_summarization`, `update_paper_abstracts`, `update_paper_full_contents`, `update_paper_summarizations` 유지

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cache_manager.py`:

```python
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        from api.store import Store
        from api.cache import CacheManager
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.cache = CacheManager(store=Store(self.tmp.name))

    def tearDown(self):
        self.cache.store.close()
        os.unlink(self.tmp.name)

    def test_dict_like_reads(self):
        self.assertEqual(self.cache.paper_abstracts["없음"], "")
        self.assertNotIn("없음", self.cache.paper_abstracts)
        self.cache.update_paper_abstracts("A", "초록")
        self.assertIn("A", self.cache.paper_abstracts)
        self.assertEqual(self.cache.paper_abstracts["A"], "초록")
        self.assertEqual(self.cache.paper_summarizations.get("A", ""), "")

    def test_full_contents_dict_roundtrip(self):
        value = {"1": {"title": "Intro", "content": "본문"}}
        self.cache.update_paper_full_contents("A", value)
        self.assertEqual(self.cache.paper_full_contents["A"], value)

    def test_summary_write_does_not_touch_others(self):
        self.cache.update_paper_summarizations("A", "요약 A")
        self.cache.update_paper_summarizations("B", "요약 B")
        self.assertTrue(self.cache.has_paper_summarization("A"))
        self.assertEqual(self.cache.paper_summarizations["B"], "요약 B")

    def test_two_managers_share_one_db(self):
        from api.store import Store
        from api.cache import CacheManager
        other = CacheManager(store=Store(self.tmp.name))
        self.cache.update_paper_summarizations("A", "요약")
        self.assertEqual(other.paper_summarizations["A"], "요약")
        other.store.close()
```

- [ ] **Step 2: 실패 확인**

Run: `python -m unittest tests.test_cache_manager`
기대: `TypeError: __init__() got an unexpected keyword argument 'store'`

- [ ] **Step 3: 구현**

`api/cache.py` 전체를 교체한다.

```python
"""요약·초록 캐시. 저장은 api.store의 SQLite가 맡고 여기서는 사전처럼 보이게만 한다."""
from api.store import Store
from prompts import is_current_summary_schema
from settings import MODEL


class _View:
    """읽기 전용 사전 흉내. 없는 키는 빈 문자열(기존 defaultdict(str) 동작 유지)."""

    def __init__(self, getter):
        self._get = getter

    def __getitem__(self, key):
        return self._get(key)

    def get(self, key, default=""):
        value = self._get(key)
        return value if value else default

    def __contains__(self, key):
        return bool(self._get(key))


class CacheManager:
    def __init__(self, store: Store = None):
        self.store = store or Store()
        self.paper_abstracts = _View(self.store.get_abstract)
        self.paper_full_contents = _View(self.store.get_full_content)
        self.paper_summarizations = _View(self.store.get_summary)

    def has_paper_summarization(self, paper_info: str) -> bool:
        return self.store.has_summary(paper_info)

    def update_paper_abstracts(self, paper_info: str, paper_abstract: str):
        self.store.put_abstract(paper_info, paper_abstract)

    def update_paper_full_contents(self, paper_info: str, paper_full_content):
        self.store.put_full_content(paper_info, paper_full_content)

    def update_paper_summarizations(self, paper_info: str, paper_summarization: str):
        schema = "4sections" if is_current_summary_schema(paper_summarization) else "legacy"
        self.store.put_summary(paper_info, paper_summarization, schema_version=schema, model=MODEL)
```

- [ ] **Step 4: 통과 확인**

Run: `python -m unittest tests.test_cache_manager tests.test_summary tests.test_on_demand`
기대: `OK` (호출부를 안 고쳤는데도 통과해야 한다. 안 되면 `_View`가 부족한 것)

- [ ] **Step 5: 마이그레이션 스크립트 작성**

`scripts/migrate_cache_to_sqlite.py`:

```python
"""pickle 캐시를 SQLite로 1회 이전. 배치와 리스너를 모두 멈춘 상태에서 돌린다."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.store import Store  # noqa: E402
from prompts import is_current_summary_schema  # noqa: E402
from settings import (  # noqa: E402
    PAPER_ABSTRACTS_PATH, PAPER_FULL_CONTENTS_PATH, PAPER_SUMMARIZATIONS_PATH,
)


def load(path):
    if not os.path.exists(path):
        print(f"건너뜀(없음): {path}")
        return {}
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as e:  # 깨진 pickle도 있으므로 죽지 않는다
        print(f"읽기 실패({path}): {e}")
        return {}


def main():
    store = Store()
    abstracts = load(PAPER_ABSTRACTS_PATH)
    for i, (key, text) in enumerate(abstracts.items(), 1):
        if text:
            store.put_abstract(key, text)
        if i % 20000 == 0:
            print(f"  초록 {i}건")
    full = load(PAPER_FULL_CONTENTS_PATH)
    for key, value in full.items():
        if value:
            store.put_full_content(key, value)
    summaries = load(PAPER_SUMMARIZATIONS_PATH)
    kept = 0
    for key, text in summaries.items():
        if text and is_current_summary_schema(text):
            store.put_summary(key, text, schema_version="4sections")
            kept += 1
    print(f"초록 {len(abstracts)} -> {store.count('abstracts')}")
    print(f"본문 {len(full)} -> {store.count('full_contents')}")
    print(f"요약 {len(summaries)} 중 현재 스키마 {kept} -> {store.count('summaries')}")
    store.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 마이그레이션 실행과 확인**

배치·리스너를 멈춘 뒤(관리자 PowerShell에서 `Disable-ScheduledTask -TaskName arxivbot-listener`, 실행 중이면 `Stop-ScheduledTask` 후 `logs/listener.pid`의 python도 정리) 원본 체크아웃의 `cache/`를 대상으로 돌린다. worktree에서 돌리려면 `cache/`의 pickle 3개를 심볼릭 링크하거나 경로를 인자로 넘긴다.

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python scripts\migrate_cache_to_sqlite.py"`
기대: 초록 22만건대, 요약은 현재 스키마만 남은 건수가 출력되고 DB 건수와 일치.

- [ ] **Step 7: 옛 pickle 보존**

```bash
cd /mnt/c/Users/hist0/Dropbox/develop/arxivbot_new/cache
for f in paper_abstracts paper_full_contents paper_summarizations; do mv "$f.pickle" "$f.pickle.bak"; done
```

- [ ] **Step 8: 커밋**

```bash
git add api/cache.py scripts/migrate_cache_to_sqlite.py tests/test_cache_manager.py
git commit -m "feat(arxivbot_new): 캐시를 pickle에서 SQLite로 이전"
```

---

## Task 4: 스레드 문맥 조립 (`api/context.py`)

**Files:**
- Create: `api/context.py`
- Create: `tests/test_context.py`
- Modify: `settings.py` (`CONTEXT_TOKEN_BUDGET = 6000`, `CONTEXT_MAX_MESSAGES = 60`)

**Interfaces:**
- Consumes: Task 2의 `Store`(digest 캐시), 기존 `api.agent.Encoder`
- Produces:
  - `fold_bot_message(text) -> str` — 봇이 올린 긴 요약을 `[요약 게시: 제목 (URL)]` 한 줄로
  - `build_context(messages, *, bot_user_id, count_tokens, budget, digest=None) -> Context`
  - `Context = NamedTuple(digest: str, kept: list[dict], folded_until_ts: str | None)`
  - `update_digest(store, thread_ts, folded_messages, summarize) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_context.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def msg(ts, text, user="U1"):
    return {"ts": ts, "user": user, "text": text}


def count_tokens(text):
    return max(1, len(text) // 4)


class TestFoldBotMessage(unittest.TestCase):
    def test_long_summary_becomes_one_line(self):
        from api.context import fold_bot_message
        text = ("GSM-Symbolic: 수학 추론의 한계 (https://arxiv.org/abs/2410.05229)\n"
                + "- **Prior Approaches**: " + "가" * 900)
        folded = fold_bot_message(text)
        self.assertIn("요약 게시", folded)
        self.assertIn("https://arxiv.org/abs/2410.05229", folded)
        self.assertLess(len(folded), 200)

    def test_short_bot_message_is_kept(self):
        from api.context import fold_bot_message
        self.assertEqual(fold_bot_message("네 그렇습니다"), "네 그렇습니다")


class TestBuildContext(unittest.TestCase):
    def test_keeps_newest_within_budget(self):
        from api.context import build_context
        messages = [msg(str(i), "가" * 400) for i in range(1, 11)]
        ctx = build_context(messages, bot_user_id="B1", count_tokens=count_tokens, budget=300)
        self.assertLess(len(ctx.kept), 10)
        self.assertEqual(ctx.kept[-1]["ts"], "10")          # 최신은 반드시 남는다
        self.assertIsNotNone(ctx.folded_until_ts)

    def test_everything_fits_means_no_fold(self):
        from api.context import build_context
        messages = [msg("1", "짧다"), msg("2", "짧다")]
        ctx = build_context(messages, bot_user_id="B1", count_tokens=count_tokens, budget=1000)
        self.assertEqual(len(ctx.kept), 2)
        self.assertIsNone(ctx.folded_until_ts)

    def test_bot_summary_is_folded_even_when_kept(self):
        from api.context import build_context
        long_summary = "제목 (https://arxiv.org/abs/2410.05229)\n" + "가" * 2000
        messages = [msg("1", long_summary, user="B1"), msg("2", "이거 쉽게 설명해줘")]
        ctx = build_context(messages, bot_user_id="B1", count_tokens=count_tokens, budget=1000)
        self.assertEqual(len(ctx.kept), 2)
        self.assertIn("요약 게시", ctx.kept[0]["text"])

    def test_single_huge_message_still_kept(self):
        from api.context import build_context
        messages = [msg("1", "가" * 100000)]
        ctx = build_context(messages, bot_user_id="B1", count_tokens=count_tokens, budget=10)
        self.assertEqual(len(ctx.kept), 1)


class TestUpdateDigest(unittest.TestCase):
    def test_digest_appends_only_new_range(self):
        import tempfile
        from api.store import Store
        from api.context import update_digest
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        store = Store(tmp.name)
        calls = []

        def summarize(previous, texts):
            calls.append((previous, list(texts)))
            return (previous + " " + " ".join(texts)).strip()

        first = update_digest(store, "T1", [msg("1", "가"), msg("2", "나")], summarize)
        self.assertEqual(first, "가 나")
        second = update_digest(store, "T1", [msg("1", "가"), msg("2", "나"), msg("3", "다")], summarize)
        self.assertEqual(calls[1][0], "가 나")      # 이전 요지를 이어받고
        self.assertEqual(calls[1][1], ["다"])        # 새로 밀려난 것만 넘긴다
        self.assertEqual(second, "가 나 다")
        store.close()
        os.unlink(tmp.name)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m unittest tests.test_context`
기대: `ModuleNotFoundError: No module named 'api.context'`

- [ ] **Step 3: 구현**

`api/context.py`:

```python
"""스레드 문맥 조립. 토큰 예산 안에서 최신 메시지를 원문으로 담고,
밀려난 앞부분은 한 문단 요지로 접는다. 접힌 원문은 read_thread 도구로 편다."""
import re
from typing import NamedTuple

FOLD_THRESHOLD_CHARS = 400
_URL_RE = re.compile(r"https?://[^\s|>)]+")


class Context(NamedTuple):
    digest: str
    kept: list
    folded_until_ts: str


def fold_bot_message(text: str) -> str:
    """봇이 올린 긴 요약은 제목 + URL 한 줄로 접는다."""
    if len(text) <= FOLD_THRESHOLD_CHARS:
        return text
    head = text.strip().splitlines()[0].strip()
    url = _URL_RE.search(text)
    title = _URL_RE.sub("", head).strip(" ()<>")
    return f"[요약 게시: {title}" + (f" ({url.group(0)})]" if url else "]")


def build_context(messages, *, bot_user_id, count_tokens, budget, digest="") -> Context:
    """messages는 오래된 것부터 정렬된 Slack 메시지 목록."""
    prepared = []
    for m in messages:
        text = m.get("text", "")
        if m.get("user") == bot_user_id:
            text = fold_bot_message(text)
        prepared.append({**m, "text": text})

    kept, used = [], 0
    for m in reversed(prepared):
        cost = count_tokens(m["text"])
        if kept and used + cost > budget:
            break
        kept.append(m)
        used += cost
    kept.reverse()

    folded = prepared[: len(prepared) - len(kept)]
    folded_until_ts = folded[-1]["ts"] if folded else None
    return Context(digest=digest, kept=kept, folded_until_ts=folded_until_ts)


def update_digest(store, thread_ts: str, folded_messages, summarize) -> str:
    """접힌 구간의 요지를 갱신한다. 이미 요지에 반영된 구간은 다시 요약하지 않는다.

    summarize(previous_digest, texts) -> str
    """
    if not folded_messages:
        return ""
    row = store.get_digest(thread_ts) or {}
    previous = row.get("digest", "")
    covered_until = row.get("covered_until_ts") or ""
    fresh = [m for m in folded_messages if m["ts"] > covered_until]
    if not fresh:
        return previous
    digest = summarize(previous, [m.get("text", "") for m in fresh])
    store.put_digest(thread_ts, digest, covered_until_ts=folded_messages[-1]["ts"])
    return digest
```

- [ ] **Step 4: 통과 확인**

Run: `python -m unittest tests.test_context`
기대: `OK`

- [ ] **Step 5: 커밋**

```bash
git add api/context.py tests/test_context.py settings.py
git commit -m "feat(arxivbot_new): 예산 기반 스레드 문맥 조립"
```

---

## Task 5: 도구 (`api/tools.py`)

**Files:**
- Create: `api/tools.py`
- Create: `tests/test_tools.py`
- Modify: `api/on_demand.py` (게시 로직을 도구와 공유하도록 정리)

**Interfaces:**
- Consumes: Task 2 `Store`, 기존 `api.on_demand.process_url`, `api.workspace.Workspace.prepare_slack_blocks`, `api.reactions.add_posted/load_store/save_store`, `api.resolvers.build_resolver`, `api.pdf`
- Produces:
  - `post_paper_summary(client, *, channel, thread_ts, url, prefix, cache, service, workspace, resolve) -> dict` — 진행 표시 편집부터 게시·리액션 등록까지. 반환 `{"ok": bool, "title": str, "url": str, "error": str}`
  - `build_tools(...) -> (specs: list[dict], dispatch: callable)`
  - `TOOL_NAMES = ("summarize_paper", "fetch_page", "read_thread")`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_tools.py`:

```python
import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeClient:
    def __init__(self):
        self.posted, self.updated = [], []

    def chat_postMessage(self, **kw):
        self.posted.append(kw)
        return {"ts": f"ts{len(self.posted)}"}

    def chat_update(self, **kw):
        self.updated.append(kw)
        return {"ok": True}


class TestPostPaperSummary(unittest.TestCase):
    def test_posts_summary_and_returns_receipt(self):
        from api import tools
        client = FakeClient()
        result = tools.post_paper_summary(
            client, channel="C1", thread_ts="1.1",
            url="https://arxiv.org/abs/2410.05229", prefix="",
            process=lambda url, on_progress: {
                "ok": True, "message": "제목\n- **Prior Approaches**: 내용",
                "blocks": [{"type": "section"}],
                "paper_info": "제목", "paper_url": url},
            on_posted=lambda **kw: None,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(len(client.posted), 1)          # 답글 1개를 만들고
        self.assertTrue(client.updated)                  # 그걸 편집해서 결과를 채운다
        self.assertIn("blocks", client.updated[-1])

    def test_failure_is_reported_in_place(self):
        from api import tools
        client = FakeClient()
        result = tools.post_paper_summary(
            client, channel="C1", thread_ts="1.1", url="https://x.test", prefix="",
            process=lambda url, on_progress: {"ok": False, "message": "가져오지 못했어요"},
            on_posted=lambda **kw: None,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(len(client.posted), 1)


class TestDispatch(unittest.TestCase):
    def test_summarize_paper_returns_receipt_not_summary(self):
        from api import tools
        specs, dispatch = tools.build_tools(
            post_summary=lambda url: {"ok": True, "title": "GSM-Symbolic", "url": url},
            fetch_page=lambda url: {"title": "Sana", "text": "본문"},
            read_thread=lambda before_ts, limit: [{"ts": "1", "text": "옛 메시지"}],
        )
        names = {s["name"] for s in specs if s.get("type") == "function"}
        self.assertEqual(names, {"summarize_paper", "fetch_page", "read_thread"})
        out = json.loads(dispatch("summarize_paper", {"url": "https://arxiv.org/abs/1"}))
        self.assertTrue(out["posted"])
        self.assertEqual(out["title"], "GSM-Symbolic")
        self.assertNotIn("Prior Approaches", json.dumps(out))   # 본문은 모델에 안 넘긴다

    def test_web_search_tool_is_declared(self):
        from api import tools
        specs, _ = tools.build_tools(
            post_summary=lambda url: {"ok": True, "title": "", "url": url},
            fetch_page=lambda url: {"title": "", "text": ""},
            read_thread=lambda before_ts, limit: [],
        )
        self.assertIn("web_search", {s.get("type") for s in specs})

    def test_unknown_tool_returns_error_json(self):
        from api import tools
        _, dispatch = tools.build_tools(
            post_summary=lambda url: {"ok": True, "title": "", "url": url},
            fetch_page=lambda url: {"title": "", "text": ""},
            read_thread=lambda before_ts, limit: [],
        )
        out = json.loads(dispatch("없는도구", {}))
        self.assertIn("error", out)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m unittest tests.test_tools`
기대: `ModuleNotFoundError: No module named 'api.tools'`

- [ ] **Step 3: 구현**

`api/tools.py`. `post_paper_summary`는 Slack 클라이언트와 `process`(=`on_demand.process_url` 부분 적용)를 주입받아 테스트 가능하게 둔다. `build_tools`는 도구 명세와 dispatch만 만든다. `fetch_page`는 `pages` 캐시를 확인하고, 논문 호스트면 `summarize_paper`를 쓰라는 힌트를 함께 돌려준다.

```python
"""에이전트 도구. 논문 요약은 게시까지 도구가 끝내고 모델에는 영수증만 준다."""
import json

STAGE = {
    "fetching": "🔄 논문 페이지 가져오는 중…",
    "downloading": "🔄 PDF 다운로드 중…",
    "summarizing": "🔄 AI가 요약하는 중…",
}

TOOL_NAMES = ("summarize_paper", "fetch_page", "read_thread")


def post_paper_summary(client, *, channel, thread_ts, url, prefix, process, on_posted):
    """답글 1개를 만들고 진행 상황으로 편집하다가 최종 요약으로 채운다."""
    posted = client.chat_postMessage(
        channel=channel, text=prefix + STAGE["fetching"], thread_ts=thread_ts)
    ts = posted["ts"]
    last = {"text": STAGE["fetching"]}

    def on_progress(stage):
        msg = STAGE.get(stage)
        if msg and msg != last["text"]:
            last["text"] = msg
            client.chat_update(channel=channel, ts=ts, text=prefix + msg)

    try:
        result = process(url, on_progress)
    except Exception as e:
        client.chat_update(channel=channel, ts=ts, text=f"처리 중 오류가 났어요: {e}\n({url})")
        return {"ok": False, "url": url, "error": str(e)}

    text = result["message"] if result["ok"] else f"{result['message']}\n({url})"
    blocks = result.get("blocks")
    try:
        client.chat_update(channel=channel, ts=ts, text=text,
                           **({"blocks": blocks} if blocks else {}))
    except Exception:
        client.chat_update(channel=channel, ts=ts, text=text)
    if result["ok"]:
        on_posted(ts=ts, paper_info=result["paper_info"], paper_url=result["paper_url"])
        return {"ok": True, "title": result["paper_info"], "url": result["paper_url"]}
    return {"ok": False, "url": url, "error": result["message"]}


def build_tools(*, post_summary, fetch_page, read_thread):
    specs = [
        {"type": "web_search"},
        {"type": "function", "name": "summarize_paper",
         "description": ("논문 URL(arXiv, ACL, CVPR, NeurIPS, PMLR, OpenReview, AAAI, "
                         "IJCAI, Interspeech, 직접 PDF)을 요약해 스레드에 직접 게시한다. "
                         "요약 본문은 반환하지 않으며 게시 결과만 돌려준다. "
                         "논문 링크가 있으면 반드시 이 도구를 쓴다."),
         "parameters": {"type": "object", "properties": {
             "url": {"type": "string", "description": "논문 URL"}},
             "required": ["url"], "additionalProperties": False}},
        {"type": "function", "name": "fetch_page",
         "description": ("일반 웹페이지 본문을 가져온다. 프로젝트 페이지·블로그·릴리스 노트용. "
                         "논문 URL이면 summarize_paper를 쓴다."),
         "parameters": {"type": "object", "properties": {
             "url": {"type": "string"}}, "required": ["url"],
             "additionalProperties": False}},
        {"type": "function", "name": "read_thread",
         "description": "문맥에서 접힌 이전 스레드 메시지를 원문으로 가져온다.",
         "parameters": {"type": "object", "properties": {
             "before_ts": {"type": "string"},
             "limit": {"type": "integer"}},
             "required": ["before_ts"], "additionalProperties": False}},
    ]

    def dispatch(name: str, args: dict) -> str:
        try:
            if name == "summarize_paper":
                r = post_summary(args["url"])
                return json.dumps({"posted": bool(r.get("ok")), "title": r.get("title", ""),
                                   "url": r.get("url", ""), "error": r.get("error", "")},
                                  ensure_ascii=False)
            if name == "fetch_page":
                return json.dumps(fetch_page(args["url"]), ensure_ascii=False)
            if name == "read_thread":
                return json.dumps(read_thread(args["before_ts"], args.get("limit", 10)),
                                  ensure_ascii=False)
            return json.dumps({"error": f"모르는 도구: {name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return specs, dispatch
```

- [ ] **Step 4: 통과 확인**

Run: `python -m unittest tests.test_tools`
기대: `OK`

- [ ] **Step 5: 커밋**

```bash
git add api/tools.py tests/test_tools.py api/on_demand.py
git commit -m "feat(arxivbot_new): 에이전트 도구 정의와 게시형 논문 요약 도구"
```

---

## Task 6: 에이전트 루프 (`api/agent_loop.py`)

**Files:**
- Create: `api/agent_loop.py`
- Create: `tests/test_agent_loop.py`
- Modify: `settings.py` (`AGENT_MODEL = "gpt-5.4"`, `AGENT_MAX_STEPS = 8`, `AGENT_DEADLINE_SEC = 90`)

**Interfaces:**
- Consumes: Task 5의 `specs`, `dispatch`
- Produces: `run_agent(*, client, model, system_prompt, user_input, tool_specs, dispatch, max_steps, deadline_sec, now=time.time, on_step=None) -> AgentResult(text: str, steps: int, truncated: bool, tool_calls: list[str])`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_agent_loop.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeItem:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        return dict(self.__dict__)


class FakeResponse:
    def __init__(self, output, text=""):
        self.output = output
        self.output_text = text


class FakeResponses:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.responses = FakeResponses(scripted)


def call(name, args, call_id="c1"):
    return FakeItem(type="function_call", name=name, arguments=args, call_id=call_id)


class TestRunAgent(unittest.TestCase):
    def test_returns_text_without_tools(self):
        from api.agent_loop import run_agent
        client = FakeClient([FakeResponse([], "안녕하세요")])
        result = run_agent(client=client, model="m", system_prompt="s", user_input="q",
                           tool_specs=[], dispatch=lambda n, a: "{}")
        self.assertEqual(result.text, "안녕하세요")
        self.assertEqual(result.steps, 1)
        self.assertFalse(result.truncated)

    def test_dispatches_tool_and_feeds_output_back(self):
        from api.agent_loop import run_agent
        client = FakeClient([
            FakeResponse([call("summarize_paper", '{"url": "https://arxiv.org/abs/1"}')]),
            FakeResponse([], "올렸습니다"),
        ])
        seen = []

        def dispatch(name, args):
            seen.append((name, args))
            return '{"posted": true}'

        result = run_agent(client=client, model="m", system_prompt="s", user_input="q",
                           tool_specs=[], dispatch=dispatch)
        self.assertEqual(seen[0][0], "summarize_paper")
        self.assertEqual(seen[0][1]["url"], "https://arxiv.org/abs/1")
        self.assertEqual(result.text, "올렸습니다")
        second_input = client.responses.calls[1]["input"]
        self.assertTrue(any(i.get("type") == "function_call_output" for i in second_input
                            if isinstance(i, dict)))

    def test_stops_at_max_steps(self):
        from api.agent_loop import run_agent
        client = FakeClient([FakeResponse([call("fetch_page", '{"url": "u"}')]) for _ in range(10)])
        result = run_agent(client=client, model="m", system_prompt="s", user_input="q",
                           tool_specs=[], dispatch=lambda n, a: "{}", max_steps=3)
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.truncated)

    def test_stops_at_deadline(self):
        from api.agent_loop import run_agent
        clock = iter([0, 1, 200, 300, 400])
        client = FakeClient([FakeResponse([call("fetch_page", '{"url": "u"}')]) for _ in range(5)])
        result = run_agent(client=client, model="m", system_prompt="s", user_input="q",
                           tool_specs=[], dispatch=lambda n, a: "{}",
                           deadline_sec=60, now=lambda: next(clock))
        self.assertTrue(result.truncated)

    def test_bad_arguments_do_not_crash(self):
        from api.agent_loop import run_agent
        client = FakeClient([
            FakeResponse([call("fetch_page", "이건 JSON이 아니다")]),
            FakeResponse([], "그래도 답한다"),
        ])
        result = run_agent(client=client, model="m", system_prompt="s", user_input="q",
                           tool_specs=[], dispatch=lambda n, a: "{}")
        self.assertEqual(result.text, "그래도 답한다")
```

- [ ] **Step 2: 실패 확인**

Run: `python -m unittest tests.test_agent_loop`
기대: `ModuleNotFoundError: No module named 'api.agent_loop'`

- [ ] **Step 3: 구현**

`api/agent_loop.py`:

```python
"""OpenAI Responses API 도구 루프. 스텝·시간 상한을 넘기면 그때까지 내용으로 답한다."""
import json
import time
from typing import NamedTuple

from api.logger import logger


class AgentResult(NamedTuple):
    text: str
    steps: int
    truncated: bool
    tool_calls: list


def _as_input_item(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return item


def run_agent(*, client, model, system_prompt, user_input, tool_specs, dispatch,
              max_steps=8, deadline_sec=90, now=time.time, on_step=None) -> AgentResult:
    started = now()
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    tool_calls, steps, truncated, text = [], 0, False, ""

    while steps < max_steps:
        steps += 1
        response = client.responses.create(
            model=model, input=conversation, tools=tool_specs)
        calls = [i for i in response.output if getattr(i, "type", "") == "function_call"]
        if not calls:
            text = getattr(response, "output_text", "") or ""
            break

        conversation += [_as_input_item(i) for i in response.output]
        for c in calls:
            try:
                args = json.loads(c.arguments) if isinstance(c.arguments, str) else (c.arguments or {})
            except ValueError:
                args = {}
            tool_calls.append(c.name)
            if on_step:
                on_step(c.name)
            output = dispatch(c.name, args)
            conversation.append({"type": "function_call_output",
                                 "call_id": c.call_id, "output": output})

        if now() - started > deadline_sec:
            truncated = True
            logger.info(f"agent loop deadline 초과: {steps} 스텝, 도구 {tool_calls}")
            break
    else:
        truncated = True
        logger.info(f"agent loop max_steps 도달: 도구 {tool_calls}")

    return AgentResult(text=text, steps=steps, truncated=truncated, tool_calls=tool_calls)
```

- [ ] **Step 4: 통과 확인**

Run: `python -m unittest tests.test_agent_loop`
기대: `OK`

- [ ] **Step 5: 커밋**

```bash
git add api/agent_loop.py tests/test_agent_loop.py settings.py
git commit -m "feat(arxivbot_new): Responses API 도구 루프"
```

---

## Task 7: 리스너 배선과 프롬프트

**Files:**
- Modify: `listener.py` (에이전트 경로 + 결정론 폴백)
- Modify: `prompts.py` (`SYSTEM_PROMPT_AGENT`, `THREAD_DIGEST_PROMPT`)
- Create: `tests/test_listener_flow.py`

**Interfaces:**
- Consumes: Task 4 `build_context`/`update_digest`, Task 5 `build_tools`/`post_paper_summary`, Task 6 `run_agent`
- Produces: `handle_mention_core(...)` — Slack 배선과 분리된 순수 흐름 함수. 반환 `{"mode": "agent"|"fallback", "text": str}`

- [ ] **Step 1: 프롬프트 작성**

`prompts.py`에 추가한다. 기존 언어 규칙(전문용어 영어 유지, 음차 금지)을 그대로 이어받는다.

```python
SYSTEM_PROMPT_AGENT = """너는 연구실 Slack의 arxivbot이다. 한국어로 답한다.

규칙:
- 논문 링크(arXiv, ACL, CVPR/ICCV, NeurIPS, ICML, OpenReview, AAAI, IJCAI, Interspeech, 직접 PDF)가 있으면 반드시 summarize_paper 도구를 부른다. 그 도구가 요약을 스레드에 직접 올리므로, 너는 요약 내용을 다시 쓰지 않는다. 링크가 여러 개면 각각 부른다.
- 논문이 아닌 웹페이지는 fetch_page로 본문을 읽고, 한 줄 요지 다음에 글머리 기호 5~8개로 중요한 점만 짚는다.
- 사실 확인이 필요하거나 최신 정보가 필요하면 web_search를 쓴다.
- 개념 설명 요청은 도구 없이 답해도 된다. 짧게, 예시 하나로.
- 문맥에서 접힌 이전 대화가 필요하면 read_thread로 가져온다.
- zero-shot, closed-loop, embodiment 같은 전문용어는 영어 그대로 쓴다. 음차하지 않는다.
- 답은 Slack에 그대로 올라간다. 제목 장식이나 인사말 없이 바로 내용부터 쓴다.
"""

THREAD_DIGEST_PROMPT = """다음은 Slack 스레드에서 오래되어 밀려난 대화다.
이전 요지에 새 내용을 합쳐 한 문단으로 압축해라. 누가 무엇을 물었고 무엇이 결론이었는지만 남긴다."""
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_listener_flow.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHandleMentionCore(unittest.TestCase):
    def test_agent_text_is_returned(self):
        from listener import handle_mention_core
        out = handle_mention_core(
            text="<@U1> attention residuals 쉽게 설명해",
            run=lambda user_input, on_step: type("R", (), {
                "text": "이렇게 이해하면 됩니다", "truncated": False, "tool_calls": []})(),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "agent")
        self.assertIn("이렇게", out["text"])

    def test_agent_failure_falls_back(self):
        from listener import handle_mention_core

        def boom(user_input, on_step):
            raise RuntimeError("api down")

        out = handle_mention_core(
            text="<@U1> https://arxiv.org/abs/2410.05229",
            run=boom,
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "fallback")

    def test_truncated_answer_is_marked(self):
        from listener import handle_mention_core
        out = handle_mention_core(
            text="<@U1> 뭐 좀 찾아줘",
            run=lambda user_input, on_step: type("R", (), {
                "text": "여기까지 봤습니다", "truncated": True, "tool_calls": ["web_search"]})(),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertIn("도중에", out["text"])
```

- [ ] **Step 3: 실패 확인**

Run: `python -m unittest tests.test_listener_flow`
기대: `ImportError: cannot import name 'handle_mention_core'`

- [ ] **Step 4: 구현**

`listener.py`에 순수 함수를 추가한다.

```python
TRUNCATED_NOTE = "\n\n(시간이 걸려 도중에 끊었습니다. 더 필요하면 다시 불러 주세요.)"


def handle_mention_core(*, text, run, fallback):
    """에이전트를 돌리고, 실패하면 결정론 경로로 넘긴다."""
    try:
        result = run(text, lambda stage: None)
    except Exception as e:
        logger.error(f"agent 실패, 결정론 경로로 폴백: {e}")
        return fallback()
    answer = (result.text or "").strip()
    if not answer:
        return {"mode": "agent", "text": ""}
    if result.truncated:
        answer += TRUNCATED_NOTE
    return {"mode": "agent", "text": answer}
```

`handle_app_mention`은 이렇게 바뀐다.

1. 채널 확인, `thread_ts` 결정(기존 `resolve_thread_ts`)
2. `client.conversations_replies(channel, ts=thread_ts, limit=CONTEXT_MAX_MESSAGES)`로 스레드를 읽는다. 실패하면 빈 목록으로 진행한다.
3. `build_context(...)`로 남길 메시지와 접힌 구간을 나눈다. 접힌 구간이 있으면 `update_digest(store, thread_ts, folded, summarize)`로 요지를 갱신한다. `summarize`는 아래와 같이 `AGENT_MODEL`을 한 번 부르는 함수로 만든다.

```python
def make_digest_summarizer(openai_client):
    def summarize(previous, texts):
        joined = "\n".join(texts)[:8000]
        prompt = f"{THREAD_DIGEST_PROMPT}\n\n[이전 요지]\n{previous or '(없음)'}\n\n[새로 밀려난 대화]\n{joined}"
        r = openai_client.responses.create(model=AGENT_MODEL, input=prompt)
        return (getattr(r, "output_text", "") or previous).strip()
    return summarize
```
4. 진행 표시 답글 1개를 올리고, `on_step`에서 도구 이름에 따라 문구를 편집한다(`web_search` -> `🔄 검색하는 중…`, `fetch_page` -> `🔄 페이지 읽는 중…`).
5. `run_agent` 결과 텍스트가 비어 있지 않으면 `workspace.prepare_slack_blocks`를 태워 편집하고, 비어 있으면(=`summarize_paper`가 이미 다 올린 경우) 진행 표시 답글을 지운다(`chat_delete`).
6. 예외가 나면 기존 경로(`extract_targets` -> 링크별 `post_paper_summary`)로 폴백한다.

- [ ] **Step 5: 통과 확인**

Run: `python -m unittest discover -s tests -p 'test_*.py'`
기대: 기존 76개 + 새 테스트 전부 `OK`

- [ ] **Step 6: 커밋**

```bash
git add listener.py prompts.py tests/test_listener_flow.py
git commit -m "feat(arxivbot_new): 리스너를 도구 쓰는 에이전트로 전환"
```

---

## Task 8: 실사용 확인과 배포 문서

**Files:**
- Create: `tests/smoke_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: 전체
- Produces: 없음

- [ ] **Step 1: 스모크 스크립트 작성**

`tests/smoke_agent.py`는 Slack 없이 에이전트 루프만 실제 API로 돌린다. 도구는 가짜로 주입해 게시가 일어나지 않게 한다.

```python
"""실 API 스모크. python tests/smoke_agent.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI                      # noqa: E402
from api.agent_loop import run_agent           # noqa: E402
from api.tools import build_tools              # noqa: E402
from prompts import SYSTEM_PROMPT_AGENT        # noqa: E402
from settings import AGENT_MODEL, OPENAI_API_KEY  # noqa: E402

CASES = [
    "https://arxiv.org/abs/2410.05229 이거 요약해줘",
    "https://nvlabs.github.io/Sana/Video2/ 이 페이지 정리해줘",
    "attention residual이 뭔지 아주 쉽게 설명해",
]

if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)
    posted = []
    specs, dispatch = build_tools(
        post_summary=lambda url: (posted.append(url), {"ok": True, "title": "(가짜)", "url": url})[1],
        fetch_page=lambda url: {"title": "(가짜 페이지)", "text": "Sana Video2 소개 본문"},
        read_thread=lambda before_ts, limit: [],
    )
    for case in CASES:
        r = run_agent(client=client, model=AGENT_MODEL, system_prompt=SYSTEM_PROMPT_AGENT,
                      user_input=case, tool_specs=specs, dispatch=dispatch)
        print("=" * 60)
        print(case)
        print("도구:", r.tool_calls, "스텝:", r.steps, "잘림:", r.truncated)
        print(r.text[:800])
    print("게시 호출된 논문:", posted)
```

- [ ] **Step 2: 스모크 실행**

Run: `powershell.exe -NoProfile -Command "cd C:\Users\hist0\worktrees\arxivbot-agentic; python tests\smoke_agent.py"`
기대: 1번은 `summarize_paper`를 부르고 본문을 다시 쓰지 않는다. 2번은 `fetch_page` 후 한 줄 요지 + 글머리 기호 5~8개. 3번은 도구 없이 짧은 설명.

- [ ] **Step 3: README 갱신**

에이전트 리스너 동작(도구 셋, 문맥 예산, 상한, 폴백), `AGENT_MODEL` 설정, SQLite 캐시 경로와 마이그레이션 절차, summaries 폐지를 적는다.

- [ ] **Step 4: 커밋과 PR**

```bash
git add tests/smoke_agent.py README.md
git commit -m "docs(arxivbot_new): 에이전트 리스너 스모크와 문서"
git push -u origin feat/agentic-listener
gh pr create --base chore/drop-summaries --head feat/agentic-listener --title "feat(arxivbot_new): 리스너를 도구 쓰는 에이전트로" --body "..."
```

- [ ] **Step 5: 배포 (사람이 확인한 뒤)**

PR이 머지되면 원본 체크아웃에서 `git pull`, 그다음 관리자 PowerShell에서 리스너를 재시작한다. 스케줄러 프로세스가 옛 코드를 잡고 있으므로 재시작하지 않으면 아무것도 바뀌지 않는다.

```powershell
Stop-ScheduledTask -TaskName arxivbot-listener
Start-ScheduledTask -TaskName arxivbot-listener
Get-Content C:\Users\hist0\Dropbox\develop\arxivbot_new\logs\listener.log -Tail 20
```

---

## 자체 점검

- 설계 문서의 Part A -> Task 1, Part B -> Task 2·3, Part C -> Task 5·6·7, Part D -> Task 4로 모두 대응된다.
- `pages` 테이블은 Task 2에서 만들고 Task 5의 `fetch_page`가 쓴다. `thread_digests`는 Task 2에서 만들고 Task 4가 쓴다.
- `Store`의 메서드 이름은 Task 2에서 정의한 것을 Task 3·4·5가 그대로 쓴다(`get_page`/`put_page`, `get_digest`/`put_digest`).
- 요지 생성기는 Task 7에서 `AGENT_MODEL` 단발 호출로 고정했다(`summarize(previous, texts) -> str`).
- 남은 위험: `summarize_paper`가 이미 답글을 올린 경우 에이전트 최종 텍스트가 비는데, 이때 진행 표시 답글을 지우는 `chat_delete` 권한이 봇 토큰에 없을 수 있다. Task 7 구현 중 확인하고, 없으면 지우는 대신 "요약을 위에 올렸습니다" 한 줄로 편집한다.
