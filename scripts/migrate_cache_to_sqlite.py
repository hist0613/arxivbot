"""pickle 캐시를 SQLite로 1회 이전한다.

배치와 리스너를 모두 멈춘 상태에서 돌린다. 옛 포맷 요약은 어차피 재요약
대상이므로 현재 4섹션 스키마인 것만 옮긴다.

    python scripts/migrate_cache_to_sqlite.py [--cache-dir <경로>] [--db <경로>]
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.store import Store  # noqa: E402
from prompts import is_current_summary_schema  # noqa: E402
from settings import BASE_DIR, CACHE_DB_PATH  # noqa: E402

CHUNK = 20000

ABSTRACT_SQL = (
    "INSERT INTO abstracts(paper_info, text, fetched_at) VALUES(?,?,?) "
    "ON CONFLICT(paper_info) DO UPDATE SET text=excluded.text"
)
FULL_SQL = (
    "INSERT INTO full_contents(paper_info, text, fetched_at) VALUES(?,?,?) "
    "ON CONFLICT(paper_info) DO UPDATE SET text=excluded.text"
)
SUMMARY_SQL = (
    "INSERT INTO summaries(paper_info, text, schema_version, model, created_at) "
    "VALUES(?,?,?,?,?) ON CONFLICT(paper_info) DO UPDATE SET text=excluded.text"
)


def load(path):
    if not os.path.exists(path):
        print(f"건너뜀(없음): {path}")
        return {}
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as e:  # 깨진 pickle이 있어도 나머지는 옮긴다
        print(f"읽기 실패({path}): {e}")
        return {}


def insert_chunked(store, sql, rows):
    buffer, total = [], 0
    for row in rows:
        buffer.append(row)
        if len(buffer) >= CHUNK:
            store.bulk(sql, buffer)
            total += len(buffer)
            buffer = []
            print(f"  ... {total}건")
    if buffer:
        store.bulk(sql, buffer)
        total += len(buffer)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=BASE_DIR, help="pickle이 있는 디렉토리")
    parser.add_argument("--db", default=CACHE_DB_PATH, help="만들 SQLite 경로")
    args = parser.parse_args()

    store = Store(args.db)
    now = time.time()
    print(f"대상 DB: {args.db}")

    print("초록 이전 중...")
    abstracts = load(os.path.join(args.cache_dir, "paper_abstracts.pickle"))
    insert_chunked(
        store, ABSTRACT_SQL, ((k, v, now) for k, v in abstracts.items() if v)
    )

    print("본문 이전 중...")
    full_contents = load(os.path.join(args.cache_dir, "paper_full_contents.pickle"))
    insert_chunked(
        store,
        FULL_SQL,
        (
            (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), now)
            for k, v in full_contents.items()
            if v
        ),
    )

    print("요약 이전 중(현재 스키마만)...")
    summarizations = load(os.path.join(args.cache_dir, "paper_summarizations.pickle"))
    current = {k: v for k, v in summarizations.items() if v and is_current_summary_schema(v)}
    insert_chunked(
        store, SUMMARY_SQL, ((k, v, "4sections", "", now) for k, v in current.items())
    )

    print(f"초록 {len(abstracts):>7} -> {store.count('abstracts'):>7}")
    print(f"본문 {len(full_contents):>7} -> {store.count('full_contents'):>7}")
    print(
        f"요약 {len(summarizations):>7} 중 현재 스키마 {len(current)} "
        f"-> {store.count('summaries')}"
    )
    store.close()


if __name__ == "__main__":
    main()
