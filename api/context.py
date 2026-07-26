"""스레드 문맥 조립.

토큰 예산 안에서 최신 메시지를 원문으로 담고, 밀려난 앞부분은 한 문단 요지로
접는다. 접힌 원문은 모델이 read_thread 도구로 직접 펼 수 있으므로 여기서
미리 다 밀어 넣지 않는다.
"""
import re
import threading
from typing import NamedTuple

# 이 길이를 넘는 봇 메시지는 요약 게시로 본다. 4섹션 요약은 보통 1500자를
# 넘고 사람이 봇에게 하는 짧은 답("고마워")은 여기 한참 못 미친다.
FOLD_THRESHOLD_CHARS = 400
# 요지는 매 턴 "이전 요지 + 새로 밀린 대화"로 다시 만들어진다. 상한이 없으면
# 스레드가 길어질수록 요지가 계속 자라 예산 밖에서 입력을 잠식한다.
MAX_DIGEST_CHARS = 1200
_URL_RE = re.compile(r"https?://[^\s|>)]+")

# 같은 스레드에 멘션이 거의 동시에 오면 read-modify-write가 겹친다. 둘 다
# 같은 이전 요지를 읽고 각자 쓰면 covered_until_ts만 앞서 나가고 진 쪽이
# 요약한 구간은 영영 요지에 안 들어간다. 요지를 쓰는 건 리스너 한 프로세스뿐이라
# 프로세스 안 락으로 충분하다.
_digest_lock = threading.Lock()


class Context(NamedTuple):
    digest: str
    kept: list
    folded_until_ts: str
    folded: list = []


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
    # 접기를 먼저 한다. 접은 뒤의 길이로 예산을 세야 요약 한 건 때문에
    # 대화 열 줄이 통째로 밀려나는 일이 없다.
    prepared = []
    for m in messages:
        text = m.get("text", "") or ""
        # bot_user_id가 None이면 user 키가 없는 메시지(가입 알림, 파일 공유
        # 등)가 전부 봇 것으로 접힌다. auth_test가 실패했을 때 그렇게 된다.
        if bot_user_id and m.get("user") == bot_user_id:
            text = fold_bot_message(text)
        prepared.append({**m, "text": text})

    # 최신부터 거꾸로 담는다. 예산이 모자랄 때 버려야 하는 건 오래된 쪽이다.
    # 요지도 같은 입력에 실리므로 그만큼 예산에서 뺀다.
    kept, used = [], count_tokens(digest) if digest else 0
    for m in reversed(prepared):
        cost = count_tokens(m["text"])
        # `kept and`가 없으면 첫 메시지 하나가 예산보다 클 때 문맥이 통째로
        # 비어버린다. 최신 메시지 하나는 예산을 넘겨도 반드시 넣는다.
        if kept and used + cost > budget:
            break
        kept.append(m)
        used += cost
    kept.reverse()

    folded = prepared[: len(prepared) - len(kept)]
    folded_until_ts = folded[-1]["ts"] if folded else None
    return Context(
        digest=digest, kept=kept, folded_until_ts=folded_until_ts, folded=folded
    )


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
    with _digest_lock:
        return _update_digest_locked(store, thread_ts, folded_messages, summarize)


def load_digest(store, thread_ts: str) -> str:
    """저장된 요지. 이번 턴에 접힌 게 없어도 앞선 요지는 살아 있어야 한다.

    Slack에서 읽어오는 메시지 수에 상한이 있어 오래된 대화는 애초에 목록에
    들어오지도 않는다. 그때 접힌 게 없다고 요지를 버리면 문맥이 통째로 날아간다.
    """
    row = store.get_digest(thread_ts) or {}
    return row.get("digest", "") or ""


def _update_digest_locked(store, thread_ts, folded_messages, summarize) -> str:
    row = store.get_digest(thread_ts) or {}
    previous = row.get("digest", "") or ""
    if not folded_messages:
        return previous
    covered_until = row.get("covered_until_ts") or ""
    # Slack ts는 "1719900000.000200"처럼 자리수가 고정된 문자열이라 사전순
    # 비교가 곧 시간순 비교다. 그래서 float 변환 없이 그냥 비교한다.
    # 이미 요지에 반영된 구간을 걸러내야 스레드가 길어져도 매번 전체를
    # 다시 요약하지 않는다(모델 호출 1회는 앞으로 밀려난 만큼만).
    fresh = [m for m in folded_messages if m["ts"] > covered_until]
    if not fresh:
        return previous
    digest = (summarize(previous, [m.get("text", "") or "" for m in fresh]) or "").strip()
    if not digest:
        return previous
    digest = digest[:MAX_DIGEST_CHARS]
    store.put_digest(thread_ts, digest, covered_until_ts=folded_messages[-1]["ts"])
    return digest
