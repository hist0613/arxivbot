"""On-demand(@멘션) 요약의 순수 코어 — Slack/네트워크에서 분리되어 단위 테스트 가능.

listener.py가 resolve_thread_ts/extract_targets/process_url을 실제 의존성(resolve
클로저, Service, Workspace)과 on_progress 콜백으로 wiring한다.
"""
from collections import deque

from api.arxiv import get_paper_info, parse_arxiv_ref
from api.resolvers import extract_urls


NO_URL_MSG = (
    "arxiv 등 논문 링크를 함께 멘션해 주세요 "
    "(예: @arxivbot https://arxiv.org/abs/2501.12345)"
)
_UNSUPPORTED_MSG = (
    "이 링크에서 논문을 가져오지 못했어요. 지원: arXiv, ACL, CVPR/ICCV, "
    "NeurIPS, ICML, OpenReview, AAAI, IJCAI, Interspeech, 직접 PDF 링크."
)


def resolve_thread_ts(event: dict) -> str:
    """멘션이 스레드 안이면 그 스레드, 아니면 멘션 메시지 자체에 답글."""
    return event.get("thread_ts") or event["ts"]


def resolve_listener_channels(workspace_config: dict) -> set:
    """멘션을 받을 채널 ID 집합. **비어 있으면 제한 없음**(초대된 모든 곳).

    설정 키는 복수형 `listener_channel_ids`가 정본이고, 단수 `listener_channel_id`는
    옛 설정을 그대로 둔 워크스페이스를 위해 계속 받는다. 문자열 하나를 넘겨도
    글자 단위로 쪼개지지 않게 감싼다.

    빈 집합의 뜻이 "아무 데서도 안 받는다"에서 "어디서든 받는다"로 바뀌었다.
    판정은 channel_allowed()가 한다.
    """
    ids = workspace_config.get("listener_channel_ids")
    if ids is None:
        ids = workspace_config.get("listener_channel_id")
    if ids is None:
        return set()
    if isinstance(ids, str):
        ids = [ids]
    return {c for c in ids if c}


def channel_allowed(channel, allowed) -> bool:
    """이 채널의 이벤트를 처리할지.

    기본은 전체 허용 — 봇이 초대된 곳이면 어디서든 답한다. 애초에 초대가
    관문이므로 코드에서 또 좁힐 이유가 없고, 채널을 늘릴 때마다 설정을
    고치고 리스너를 재시작하던 걸 없앤다. 시끄러운 채널이 생기면 그때
    `listener_channel_ids`를 채워 화이트리스트로 되돌릴 수 있다.
    """
    if not allowed:
        return True
    return channel in allowed


def is_direct_message(event: dict) -> bool:
    """봇과의 1:1 DM 대화인가. 그룹 DM(mpim)은 멘션이 필요하므로 제외."""
    return event.get("channel_type") == "im"


def should_handle_dm(event: dict, bot_user_id=None) -> bool:
    """DM message 이벤트 중 "사람이 새로 쓴 글"만 통과시킨다.

    DM에서는 멘션 없이 답하므로, 걸러내지 않으면 봇이 자기 말에 답해
    무한 루프가 된다. 특히 진행 표시("생각하는 중…")를 chat_update로 고칠
    때마다 subtype=message_changed 이벤트가 같은 DM으로 되돌아온다.

    - bot_id가 있으면 봇(자기 자신 포함)이 쓴 것
    - subtype이 있으면 편집·삭제·참여 같은 부가 이벤트
    - hidden은 사용자에게 보이지 않는 이벤트
    """
    if not is_direct_message(event):
        return False
    if event.get("bot_id") or event.get("subtype") or event.get("hidden"):
        return False
    user = event.get("user")
    if not user or (bot_user_id and user == bot_user_id):
        return False
    return bool((event.get("text") or "").strip())


class SeenEvents:
    """최근에 처리한 이벤트 키를 기억해 같은 걸 두 번 처리하지 않는다.

    DM에서 봇을 멘션하면 Slack이 app_mention과 message.im을 **둘 다** 보낸다.
    둘 중 어느 쪽이 먼저 올지는 보장되지 않으므로, 먼저 온 쪽만 처리한다.
    Slack의 이벤트 재전송(같은 event_ts 재시도)에도 같은 방어가 된다.
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._keys = set()
        self._order = deque()

    def add(self, key) -> bool:
        """처음 보는 키면 기록하고 True. 이미 본 키면 False."""
        if key in self._keys:
            return False
        self._keys.add(key)
        self._order.append(key)
        while len(self._order) > self.capacity:
            self._keys.discard(self._order.popleft())
        return True


def extract_targets(text) -> list:
    """멘션 텍스트에서 처리할 URL 목록.

    URL이 하나도 없으면 bare arXiv id(예: "2106.14052") 폴백.
    같은 논문의 abs/pdf 혼용은 arXiv id 기준으로 중복 제거한다.
    """
    urls = extract_urls(text)
    if not urls:
        bare = parse_arxiv_ref(text)
        return [bare] if bare else []
    seen, targets = set(), []
    for url in urls:
        key = parse_arxiv_ref(url) or url
        if key not in seen:
            seen.add(key)
            targets.append(url)
    return targets


def process_url(url, *, cache, service, workspace, resolve,
                on_progress=lambda s: None) -> dict:
    """URL 1개를 요약 결과 dict로 처리한다.

    반환: {"ok": bool, "message": str, "blocks": list|None,
           "paper_info": str|None, "paper_url": str|None}
    blocks는 Slack rich_text 글머리 기호 목록. None이면 message 문자열로 보낸다.
    resolve(url, on_progress) -> ResolvedPaper | None  (주입)
    on_progress(stage) 단계: "fetching" → ("downloading") → "summarizing"
    """
    on_progress("fetching")
    resolved = resolve(url, on_progress=on_progress)
    if resolved is None or not resolved.text:
        return {"ok": False, "message": _UNSUPPORTED_MSG, "blocks": None,
                "paper_info": None, "paper_url": None}

    paper_info = get_paper_info(resolved.url, resolved.title)
    on_progress("summarizing")
    summarization = service.summarize_text(paper_info, resolved.text)
    if not summarization:
        return {"ok": False,
                "message": "요약 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                "blocks": None, "paper_info": None, "paper_url": None}

    message_content, _ = workspace.prepare_content(paper_info, "", summarization)
    note = getattr(resolved, "note", "")
    if note:
        message_content += f"\n\n{note}"
    blocks = workspace.prepare_slack_blocks(
        paper_info, "", summarization, extra_text=note
    )
    return {"ok": True, "message": message_content, "blocks": blocks,
            "paper_info": paper_info, "paper_url": resolved.url}


def process_mention(text, *, cache, service, workspace, resolve,
                    on_progress=lambda s: None) -> dict:
    """멘션 텍스트의 첫 URL만 처리하는 단건 진입점 (smoke 테스트용 호환)."""
    targets = extract_targets(text)
    if not targets:
        return {"ok": False, "message": NO_URL_MSG,
                "paper_info": None, "paper_url": None}
    return process_url(targets[0], cache=cache, service=service,
                       workspace=workspace, resolve=resolve,
                       on_progress=on_progress)
