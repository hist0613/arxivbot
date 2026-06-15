# Slack 리액션 수집 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 게시되는 논문에 달리는 사람 리액션을 `main.py` 단일 실행 안에서 수집해 논문 ts키 store(`reactions/papers.json`)에 in-place 저장한다.

**Architecture:** 게시(`_send_slack_messages`) 시 각 논문 답글 ts를 store에 엔트리로 기록 → main.py 말미에서 최근 N일 엔트리를 thread별 `conversations_replies`(bulk)로 수확해 엔트리의 reactions를 덮어씀. user id는 salt 해시, `reactions/`는 gitignore.

**Tech Stack:** Python, slack_sdk(WebClient), stdlib(json/hashlib/datetime), 테스트는 stdlib `unittest`(가짜 client) + 수동 API smoke.

**기준 스펙:** `docs/superpowers/specs/2026-06-15-reaction-collection-design.md`
**브랜치:** `feature/reaction-collection`
**실행 환경:** production은 Windows PowerShell python. 검증 명령은 `python tests\...`(백슬래시). 단위테스트는 API 불필요(WSL/Windows 무관하나 일관성 위해 PowerShell 표기). 커밋은 `-m`×2(크로스셸).

---

## File Structure

- `settings.py` (수정): `REACTIONS_DIR`, `PAPERS_STORE_PATH`, `HARVEST_WINDOW_DAYS`, `REACTION_HASH_SALT`.
- `.gitignore` (수정): `reactions/` 추가.
- `.env` (수정, 수동): `REACTION_HASH_SALT`.
- `api/reactions.py` (신규): store I/O(`load_store`/`save_store`), `add_posted`, `hash_user`, `harvest_reactions`.
- `api/workspace.py` (수정): `prepare_field_threads`가 content에 `paper_url`/`field` 포함; `_send_slack_messages`가 답글 ts를 store에 기록.
- `main.py` (수정): 게시 후·커밋 전 수확 단계.
- `tests/test_reactions.py` (신규): 단위 테스트.
- `tests/smoke_reactions.py` (신규): Influcoder 메시지로 실 API 수확 검증(수동).

---

## Task 1: 설정·gitignore·salt (`settings.py`, `.gitignore`, `.env`)

**Files:**
- Modify: `settings.py`, `.gitignore`
- Modify(수동): `.env`
- Test: `tests/test_reactions.py`

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_reactions.py` 생성

```python
import os, sys, json, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSettings(unittest.TestCase):
    def test_reaction_settings(self):
        import settings
        self.assertTrue(settings.PAPERS_STORE_PATH.replace("\\", "/").endswith("reactions/papers.json"))
        self.assertGreaterEqual(settings.HARVEST_WINDOW_DAYS, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `python tests\test_reactions.py -v`
Expected: FAIL — `AttributeError: module 'settings' has no attribute 'PAPERS_STORE_PATH'`.

- [ ] **Step 3: `settings.py`에 추가** — 파일 끝(경로 정의들 뒤, 예: `PAPER_SUMMARIZATIONS_PATH` 근처 아래)에 추가

```python
# --- 리액션 수집 (Feature 1) ---
REACTIONS_DIR = os.path.join(REPO_DIR, "reactions")  # gitignore됨 (비공개)
if not os.path.exists(REACTIONS_DIR):
    os.makedirs(REACTIONS_DIR)
PAPERS_STORE_PATH = os.path.join(REACTIONS_DIR, "papers.json")
HARVEST_WINDOW_DAYS = 14
REACTION_HASH_SALT = os.getenv("REACTION_HASH_SALT", "")
```

- [ ] **Step 4: `.gitignore`에 추가** — 마지막 줄에 추가

```
reactions/
```

- [ ] **Step 5: `.env`에 salt 추가 (PowerShell, 수동)**

Run (PowerShell):
```powershell
python -c "import secrets; print('REACTION_HASH_SALT=' + secrets.token_hex(16))" | Add-Content .env
```
Expected: `.env`에 `REACTION_HASH_SALT=<32 hex>` 한 줄 추가.

- [ ] **Step 6: 통과 확인**

Run: `python tests\test_reactions.py -v`
Expected: `test_reaction_settings ... ok`

- [ ] **Step 7: 커밋**

```bash
git add settings.py .gitignore tests/test_reactions.py
git commit -m "feat: 리액션 수집 설정·gitignore·salt" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: store I/O + hash_user + add_posted (`api/reactions.py`)

**Files:**
- Create: `api/reactions.py`
- Test: `tests/test_reactions.py`

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_reactions.py`의 `if __name__` 위에 추가

```python
class TestStore(unittest.TestCase):
    def _tmp(self):
        import tempfile
        d = tempfile.mkdtemp()
        return os.path.join(d, "papers.json")

    def test_hash_user_deterministic_and_salted(self):
        from api.reactions import hash_user
        self.assertEqual(hash_user("U1", "saltA"), hash_user("U1", "saltA"))
        self.assertNotEqual(hash_user("U1", "saltA"), hash_user("U1", "saltB"))
        self.assertEqual(len(hash_user("U1", "saltA")), 16)

    def test_add_posted_roundtrip_and_dedup(self):
        from api.reactions import load_store, save_store, add_posted
        path = self._tmp()
        store = load_store(path)
        add_posted(store, ts="100.1", thread_ts="100.0", channel_id="C1",
                   workspace="w", paper_info="P (url)", paper_url="url",
                   field="cs.CL", posted_at="2026-06-15T00:00:00+00:00")
        add_posted(store, ts="100.1", thread_ts="100.0", channel_id="C1",
                   workspace="w", paper_info="P (url)", paper_url="url",
                   field="cs.CL", posted_at="2026-06-15T00:00:00+00:00")  # 동일 ts 재기록
        save_store(store, path)
        store2 = load_store(path)
        self.assertEqual(len(store2), 1)
        self.assertEqual(store2["100.1"]["field"], "cs.CL")
        self.assertIsNone(store2["100.1"]["reactions"])
```

- [ ] **Step 2: 실패 확인**

Run: `python tests\test_reactions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.reactions'`.

- [ ] **Step 3: `api/reactions.py` 생성 (이 task 범위)**

```python
import os
import json
import hashlib

from settings import PAPERS_STORE_PATH, REACTION_HASH_SALT


def hash_user(user_id: str, salt: str = REACTION_HASH_SALT) -> str:
    return hashlib.sha256((salt + user_id).encode()).hexdigest()[:16]


def load_store(path: str = PAPERS_STORE_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def save_store(store: dict, path: str = PAPERS_STORE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(store, fp, ensure_ascii=False, indent=2)


def add_posted(store, *, ts, thread_ts, channel_id, workspace,
               paper_info, paper_url, field, posted_at) -> None:
    if ts in store:
        return  # 이미 기록된 게시물은 유지(중복 방지)
    store[ts] = {
        "thread_ts": thread_ts,
        "channel_id": channel_id,
        "workspace": workspace,
        "paper_info": paper_info,
        "paper_url": paper_url,
        "field": field,
        "posted_at": posted_at,
        "last_harvested": None,
        "reactions": None,
    }
```

- [ ] **Step 4: 통과 확인**

Run: `python tests\test_reactions.py -v`
Expected: `TestStore` 2건 + `TestSettings` ok.

- [ ] **Step 5: 커밋**

```bash
git add api/reactions.py tests/test_reactions.py
git commit -m "feat: 리액션 store I/O + hash_user + add_posted" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: harvest_reactions (bulk read, 가짜 client 테스트) (`api/reactions.py`)

**Files:**
- Modify: `api/reactions.py`
- Test: `tests/test_reactions.py`

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_reactions.py`에 추가

```python
class FakeResp(dict):
    pass


class FakeClient:
    """conversations_replies만 흉내. thread_ts별 고정 응답."""
    def __init__(self, replies_by_thread):
        self.replies_by_thread = replies_by_thread
        self.calls = 0

    def conversations_replies(self, channel, ts, cursor=None, limit=200):
        self.calls += 1
        return FakeResp(messages=self.replies_by_thread[ts], response_metadata={})


class TestHarvest(unittest.TestCase):
    def test_harvest_updates_in_window_excludes_bot_and_hashes(self):
        from api.reactions import harvest_reactions
        store = {
            "200.1": {"thread_ts": "200.0", "channel_id": "C1", "workspace": "w",
                      "paper_info": "P1", "paper_url": "u1", "field": "cs.CL",
                      "posted_at": "2026-06-15T00:00:00+00:00",
                      "last_harvested": None, "reactions": None},
            "999.1": {"thread_ts": "999.0", "channel_id": "C1", "workspace": "w",
                      "paper_info": "OLD", "paper_url": "u9", "field": "cs.CL",
                      "posted_at": "2026-05-01T00:00:00+00:00",  # 윈도우 밖
                      "last_harvested": None, "reactions": None},
        }
        client = FakeClient({
            "200.0": [
                {"ts": "200.0"},  # thread 제목(엔트리 아님)
                {"ts": "200.1", "reactions": [
                    {"name": "thumbsup", "users": ["U_bot", "U_a", "U_b"], "count": 3}
                ]},
            ],
        })
        import datetime as dt
        now = dt.datetime(2026, 6, 16, tzinfo=dt.timezone.utc)
        n = harvest_reactions(client, store, window_days=14,
                              bot_user_id="U_bot", now=now, salt="s")
        self.assertEqual(n, 1)                       # 윈도우 내 1건만
        self.assertEqual(client.calls, 1)            # thread 1개만 호출(999는 윈도우 밖)
        rs = store["200.1"]["reactions"]
        self.assertEqual(rs[0]["emoji"], "thumbsup")
        self.assertEqual(rs[0]["count"], 2)          # 봇 제외 후 2
        from api.reactions import hash_user
        self.assertEqual(set(rs[0]["hashed_users"]),
                         {hash_user("U_a", "s"), hash_user("U_b", "s")})
        self.assertIsNotNone(store["200.1"]["last_harvested"])
        self.assertIsNone(store["999.1"]["reactions"])  # 윈도우 밖 미갱신
```

- [ ] **Step 2: 실패 확인**

Run: `python tests\test_reactions.py -v`
Expected: FAIL — `ImportError: cannot import name 'harvest_reactions'`.

- [ ] **Step 3: `api/reactions.py`에 추가**

상단 import에 추가:
```python
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from settings import HARVEST_WINDOW_DAYS
```

함수 추가:
```python
def _in_window(posted_at: str, now: datetime, window_days: int) -> bool:
    try:
        ts = datetime.fromisoformat(posted_at)
    except ValueError:
        return False
    return ts >= now - timedelta(days=window_days)


def _replies(client, channel, thread_ts):
    msgs, cursor = [], None
    while True:
        for _ in range(6):
            try:
                r = client.conversations_replies(channel=channel, ts=thread_ts,
                                                 cursor=cursor, limit=200)
                break
            except Exception as e:
                if getattr(getattr(e, "response", None), "status_code", None) == 429:
                    time.sleep(int(e.response.headers.get("Retry-After", 3)))
                    continue
                raise
        msgs.extend(r.get("messages", []))
        cursor = (r.get("response_metadata") or {}).get("next_cursor")
        if not cursor:
            break
    return msgs


def harvest_reactions(client, store, *, window_days=HARVEST_WINDOW_DAYS,
                      bot_user_id, now=None, salt=REACTION_HASH_SALT) -> int:
    now = now or datetime.now(timezone.utc)
    # 윈도우 내 엔트리를 (channel, thread_ts)로 그룹화
    groups = defaultdict(list)
    for ts, e in store.items():
        if _in_window(e["posted_at"], now, window_days):
            groups[(e["channel_id"], e["thread_ts"])].append(ts)
    updated = 0
    for (channel, thread_ts), _tslist in groups.items():
        try:
            messages = _replies(client, channel, thread_ts)
        except Exception:
            continue  # 이 스레드만 건너뜀(다음 실행 재시도)
        by_ts = {m["ts"]: m for m in messages}
        for ts in _tslist:
            m = by_ts.get(ts)
            if not m:
                continue
            reactions = []
            for rc in m.get("reactions", []):
                users = [hash_user(u, salt) for u in rc.get("users", [])
                         if u != bot_user_id]
                reactions.append({"emoji": rc["name"], "count": len(users),
                                  "hashed_users": users})
            store[ts]["reactions"] = reactions
            store[ts]["last_harvested"] = now.isoformat()
            updated += 1
    return updated
```

- [ ] **Step 4: 통과 확인**

Run: `python tests\test_reactions.py -v`
Expected: `TestHarvest` ok + 기존 전부 ok.

- [ ] **Step 5: 커밋**

```bash
git add api/reactions.py tests/test_reactions.py
git commit -m "feat: harvest_reactions (thread bulk read, 윈도우/봇제외/해시)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 게시 시 엔트리 기록 (`api/workspace.py`)

**Files:**
- Modify: `api/workspace.py` (`prepare_field_threads`, `_send_slack_messages`)
- Test: `tests/test_reactions.py`

- [ ] **Step 1: 실패 테스트 추가** — content에 `paper_url`/`field`가 들어가는지 검증

```python
class TestPrepareContentFields(unittest.TestCase):
    def test_thread_content_has_url_and_field(self):
        import json as _json
        from settings import WORKSPACE_CONFIGS
        from api.arxiv import get_paper_info
        from api.workspace import Workspace
        cfg = next(c for c in WORKSPACE_CONFIGS if c["service_type"] == "slack")
        ws = Workspace(cfg)
        ws.old_paper_set = set()
        ws.fields = ["cs.CL"]
        url, title = "https://arxiv.org/abs/1234.5678", "Test Paper"
        info = get_paper_info(url, title)

        class Cache:
            paper_summarizations = {info: _json.dumps({"Prior Approaches": "a",
                "Core Contribution": "b", "Technical Challenges": "c",
                "Empirical Impact": "d"})}
        threads = ws.prepare_field_threads({"cs.CL": [(url, title, "")]}, Cache())
        content = threads[0]["thread_contents"][0]
        self.assertEqual(content["paper_url"], url)
        self.assertEqual(content["field"], "cs.CL")
```

- [ ] **Step 2: 실패 확인**

Run: `python tests\test_reactions.py -v`
Expected: FAIL — `KeyError: 'paper_url'`.

- [ ] **Step 3: `prepare_field_threads` 수정** — content dict에 `paper_url`/`field` 추가

`api/workspace.py`의 `field_thread["thread_contents"].append({...})` 블록을 아래로 교체:
```python
                field_thread["thread_contents"].append(
                    {
                        "paper_info": paper_info,
                        "paper_url": paper_url,
                        "field": field,
                        "message_content": message_content,
                        "file_content": file_content,
                    }
                )
```

- [ ] **Step 4: 통과 확인**

Run: `python tests\test_reactions.py -v`
Expected: `TestPrepareContentFields ... ok`.

- [ ] **Step 5: `_send_slack_messages` 수정** — 답글 ts를 store에 기록

`api/workspace.py` 상단 import에 추가:
```python
from datetime import datetime, timezone
from api.reactions import load_store, save_store, add_posted
```
`_send_slack_messages` 본문을 아래로 교체:
```python
    async def _send_slack_messages(self, threads: list[dict]):
        client = WebClient(self.slack_token)
        store = load_store()
        for thread in threads:
            result = client.chat_postMessage(
                channel=self.allowed_channel, text=thread["thread_title"]
            )
            thread_ts = result["ts"]

            for content in thread["thread_contents"]:
                reply = client.chat_postMessage(
                    channel=self.allowed_channel,
                    text=content["message_content"],
                    thread_ts=thread_ts,
                )
                add_posted(
                    store,
                    ts=reply["ts"],
                    thread_ts=thread_ts,
                    channel_id=self.allowed_channel_id,
                    workspace=self.workspace,
                    paper_info=content["paper_info"],
                    paper_url=content["paper_url"],
                    field=content["field"],
                    posted_at=datetime.now(timezone.utc).isoformat(),
                )
                self._update_old_paper_set(content["paper_info"])
                await self._apply_rate_limit()
        save_store(store)
```

- [ ] **Step 6: 통과 확인 (기존 테스트 영향 없음)**

Run: `python tests\test_reactions.py -v`
Expected: 전체 ok (게시 자체는 smoke에서 검증).

- [ ] **Step 7: 커밋**

```bash
git add api/workspace.py tests/test_reactions.py
git commit -m "feat: 게시 시 논문 답글 ts를 리액션 store에 기록" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: main.py 수확 단계

**Files:**
- Modify: `main.py`
- Test: 없음(런타임 배선 — Task 3 단위 + Task 6 smoke로 커버)

- [ ] **Step 1: `main.py` 수정** — 게시 루프 뒤, `repo = git.Repo(...)` 앞에 수확 단계 추가

`main.py` 상단 import에 추가:
```python
from datetime import datetime, timezone
from slack_sdk import WebClient
from api.reactions import load_store, save_store, harvest_reactions
from settings import HARVEST_WINDOW_DAYS
```
게시 for-루프(`workspace.save_summaries(threads)`까지) 뒤에 추가:
```python
    # 리액션 수확 (slack 워크스페이스 한정)
    for workspace in workspaces:
        if workspace.service_type != "slack":
            continue
        store = load_store()
        client = WebClient(workspace.slack_token)
        bot_user_id = client.auth_test()["user_id"]
        n = harvest_reactions(
            client, store,
            window_days=HARVEST_WINDOW_DAYS,
            bot_user_id=bot_user_id,
            now=datetime.now(timezone.utc),
        )
        save_store(store)
        logger.info(f"Harvested reactions for {n} papers ({workspace.workspace}).")
```

- [ ] **Step 2: 임포트/문법 확인 (배치 실행은 안 함)**

Run (PowerShell): `python -c "import ast; ast.parse(open('main.py',encoding='utf-8').read()); print('main.py parse OK')"`
Expected: `main.py parse OK`

- [ ] **Step 3: 커밋**

```bash
git add main.py
git commit -m "feat: main.py 말미에 리액션 수확 단계 추가" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 수동 smoke (Influcoder 메시지로 실 API 검증)

**Files:**
- Create: `tests/smoke_reactions.py`

`reactions.get` 경로를 실제 채널로 검증한다. 기존 Influcoder 답글(👀 1)을 찾아 임시 store 엔트리를 만들고 수확해 `eyes`가 잡히는지 확인. store 파일은 임시 경로 사용(실제 papers.json 오염 X).

- [ ] **Step 1: `tests/smoke_reactions.py` 생성**

```python
"""실 API smoke: Influcoder 답글(eyes 1)을 수확해 검증. 수동 실행."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()
from slack_sdk import WebClient
from api.reactions import load_store, save_store, add_posted, harvest_reactions

CH = "C0B7V0V8U7N"

def main():
    client = WebClient(os.getenv("SLACK_TOKEN_SEUNGTAEK_LAB"))
    bot = client.auth_test()["user_id"]
    # Influcoder 답글 찾기
    target = None
    hist = client.conversations_history(channel=CH, limit=200)
    for m in hist["messages"]:
        if m.get("reply_count", 0) > 0:
            rr = client.conversations_replies(channel=CH, ts=m["ts"], limit=200)
            for rm in rr["messages"]:
                if "Influcoder" in rm.get("text", ""):
                    target = (m["ts"], rm["ts"]); break
        if target:
            break
    assert target, "Influcoder 메시지를 못 찾음"
    thread_ts, reply_ts = target
    print("found:", reply_ts, "in thread", thread_ts)

    path = os.path.join(tempfile.mkdtemp(), "papers.json")
    store = load_store(path)
    add_posted(store, ts=reply_ts, thread_ts=thread_ts, channel_id=CH,
               workspace="seungtaek-lab", paper_info="Influcoder", paper_url="u",
               field="cs.CL", posted_at=datetime.now(timezone.utc).isoformat())
    n = harvest_reactions(client, store, window_days=3650, bot_user_id=bot,
                          now=datetime.now(timezone.utc))
    save_store(store, path)
    rs = store[reply_ts]["reactions"]
    print("reactions:", rs)
    assert any(r["emoji"] == "eyes" for r in rs), "eyes 리액션을 못 잡음"
    print("SMOKE PASS")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run (PowerShell): `python tests\smoke_reactions.py`
Expected: `found: ...` → `reactions: [{'emoji': 'eyes', ...}]` → `SMOKE PASS`.

- [ ] **Step 3: 커밋**

```bash
git add tests/smoke_reactions.py
git commit -m "test: 리액션 수확 실 API smoke (Influcoder eyes)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 최종 검증

- [ ] **Step 1: 전체 단위 테스트**

Run: `python tests\test_reactions.py -v`
Expected: settings/store(2)/harvest/prepare-content = 5개 ok.

- [ ] **Step 2: 기존 요약 테스트 회귀 확인**

Run: `python tests\test_summary.py`
Expected: 5개 ok (영향 없음).

- [ ] **Step 3: 브랜치 로그 확인**

Run: `git log --oneline feature/reaction-collection -8`
Expected: Task1~6 커밋 + 스펙 커밋.

---

## Self-Review 결과 (작성자 점검)

- **스펙 커버리지**: §3.1(reactions.py)→Task2/3, §3.2(게시 기록)→Task4, §3.3(수확 호출)→Task5, §3.4(설정)→Task1, §3.5(gitignore)→Task1, §4(프라이버시: 해시·gitignore)→Task1/2, §6(테스트)→Task2/3/4/6/7. 모든 섹션 task 대응.
- **플레이스홀더**: 없음(코드/명령 실체 포함).
- **타입 일관성**: `add_posted`/엔트리 키(thread_ts, channel_id, paper_url, field, posted_at, last_harvested, reactions) ↔ `harvest_reactions`/테스트 접근 일치. `hash_user(user_id, salt)` 시그니처 호출부 일치. `_send_slack_messages`가 쓰는 `content["paper_url"]/["field"]`는 Task4 Step3에서 생성.
- **YAGNI**: 시간추이 이력·polarity·RM 정형 제외(스펙 §7과 일치).
