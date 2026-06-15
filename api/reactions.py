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
