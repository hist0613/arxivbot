"""SQLite 저장소.

pickle을 통째로 읽고 통째로 덮어쓰던 방식을 키 단위 upsert로 바꾼다. 배치와
리스너가 같은 파일을 동시에 열어도 서로의 스냅샷을 덮어쓰지 않는다.
"""
import json
import sqlite3
import threading
import time

from settings import CACHE_DB_PATH

TABLES = ("abstracts", "full_contents", "summaries", "pages", "thread_digests")

SCHEMA = """
CREATE TABLE IF NOT EXISTS abstracts (
    paper_info TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS full_contents (
    paper_info TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS summaries (
    paper_info TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    schema_version TEXT,
    model TEXT,
    created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    title TEXT,
    text TEXT NOT NULL,
    fetched_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS thread_digests (
    thread_ts TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    covered_until_ts TEXT,
    updated_at REAL NOT NULL);
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

    # --- 초록 ---

    def get_abstract(self, paper_info: str) -> str:
        row = self._one("SELECT text FROM abstracts WHERE paper_info=?", paper_info)
        return row["text"] if row else ""

    def put_abstract(self, paper_info: str, text: str):
        self._write(
            "INSERT INTO abstracts(paper_info, text, fetched_at) VALUES(?,?,?) "
            "ON CONFLICT(paper_info) DO UPDATE SET "
            "text=excluded.text, fetched_at=excluded.fetched_at",
            paper_info,
            text,
            time.time(),
        )

    # --- 본문 (섹션 dict 또는 평문) ---

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
            "ON CONFLICT(paper_info) DO UPDATE SET "
            "text=excluded.text, fetched_at=excluded.fetched_at",
            paper_info,
            text,
            time.time(),
        )

    # --- 요약 ---

    def get_summary(self, paper_info: str) -> str:
        row = self._one("SELECT text FROM summaries WHERE paper_info=?", paper_info)
        return row["text"] if row else ""

    def has_summary(self, paper_info: str) -> bool:
        return bool(self.get_summary(paper_info))

    def put_summary(
        self, paper_info: str, text: str, schema_version: str = "", model: str = ""
    ):
        self._write(
            "INSERT INTO summaries(paper_info, text, schema_version, model, created_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(paper_info) DO UPDATE SET "
            "text=excluded.text, schema_version=excluded.schema_version, "
            "model=excluded.model, created_at=excluded.created_at",
            paper_info,
            text,
            schema_version,
            model,
            time.time(),
        )

    # --- 일반 웹페이지 ---

    def get_page(self, url: str):
        row = self._one("SELECT title, text, fetched_at FROM pages WHERE url=?", url)
        return dict(row) if row else None

    def put_page(self, url: str, title: str, text: str):
        self._write(
            "INSERT INTO pages(url, title, text, fetched_at) VALUES(?,?,?,?) "
            "ON CONFLICT(url) DO UPDATE SET title=excluded.title, "
            "text=excluded.text, fetched_at=excluded.fetched_at",
            url,
            title,
            text,
            time.time(),
        )

    # --- 스레드 요지 ---

    def get_digest(self, thread_ts: str):
        row = self._one(
            "SELECT digest, covered_until_ts, updated_at FROM thread_digests "
            "WHERE thread_ts=?",
            thread_ts,
        )
        return dict(row) if row else None

    def put_digest(self, thread_ts: str, digest: str, covered_until_ts: str):
        self._write(
            "INSERT INTO thread_digests(thread_ts, digest, covered_until_ts, updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(thread_ts) DO UPDATE SET "
            "digest=excluded.digest, covered_until_ts=excluded.covered_until_ts, "
            "updated_at=excluded.updated_at",
            thread_ts,
            digest,
            covered_until_ts,
            time.time(),
        )

    def bulk(self, sql: str, rows) -> int:
        """마이그레이션처럼 수십만 건을 넣을 때. 건건이 commit하면 너무 느리다."""
        with self._lock:
            cur = self._conn.executemany(sql, rows)
            self._conn.commit()
            return cur.rowcount

    def count(self, table: str) -> int:
        if table not in TABLES:
            raise ValueError(f"모르는 테이블: {table}")
        return self._one(f"SELECT COUNT(*) AS n FROM {table}")["n"]
