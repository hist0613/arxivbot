import os
import json
import time
import hashlib
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from settings import PAPERS_STORE_PATH, REACTION_HASH_SALT, HARVEST_WINDOW_DAYS


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
    groups = defaultdict(list)
    for ts, e in store.items():
        if _in_window(e["posted_at"], now, window_days):
            groups[(e["channel_id"], e["thread_ts"])].append(ts)
    updated = 0
    for (channel, thread_ts), tslist in groups.items():
        try:
            messages = _replies(client, channel, thread_ts)
        except Exception:
            continue  # 이 스레드만 건너뜀(다음 실행 재시도)
        by_ts = {m["ts"]: m for m in messages}
        for ts in tslist:
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
