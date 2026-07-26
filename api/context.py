"""스레드 문맥 조립.

토큰 예산 안에서 최신 메시지를 원문으로 담고, 밀려난 앞부분은 한 문단 요지로
접는다. 접힌 원문은 모델이 read_thread 도구로 직접 펼 수 있으므로 여기서
미리 다 밀어 넣지 않는다.
"""
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
    title = _URL_RE.sub("", head).strip(" ()<>|")
    if url:
        return f"[요약 게시: {title} ({url.group(0)})]"
    return f"[요약 게시: {title}]"


def build_context(messages, *, bot_user_id, count_tokens, budget, digest="") -> Context:
    """messages는 오래된 것부터 정렬된 Slack 메시지 목록."""
    prepared = []
    for m in messages:
        text = m.get("text", "") or ""
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


def render_context(context: Context, *, mention_text: str) -> str:
    """모델에 넣을 하나의 입력 문자열로 만든다."""
    parts = []
    if context.digest:
        parts.append(f"[이전 대화 요지]\n{context.digest}")
    if context.folded_until_ts:
        parts.append(
            f"(이 스레드의 {context.folded_until_ts} 이전 메시지는 접혀 있다. "
            "필요하면 read_thread로 가져와라.)"
        )
    if context.kept:
        lines = [f"{m.get('user', '?')}: {m['text']}" for m in context.kept]
        parts.append("[스레드 최근 대화]\n" + "\n".join(lines))
    parts.append(f"[지금 온 멘션]\n{mention_text}")
    return "\n\n".join(parts)


def update_digest(store, thread_ts: str, folded_messages, summarize) -> str:
    """접힌 구간의 요지를 갱신한다. 이미 반영된 구간은 다시 요약하지 않는다.

    summarize(previous_digest, texts) -> str
    """
    if not folded_messages:
        return ""
    row = store.get_digest(thread_ts) or {}
    previous = row.get("digest", "") or ""
    covered_until = row.get("covered_until_ts") or ""
    fresh = [m for m in folded_messages if m["ts"] > covered_until]
    if not fresh:
        return previous
    digest = summarize(previous, [m.get("text", "") or "" for m in fresh])
    store.put_digest(thread_ts, digest, covered_until_ts=folded_messages[-1]["ts"])
    return digest
