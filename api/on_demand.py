"""On-demand(@멘션) 요약의 순수 코어 — Slack/네트워크에서 분리되어 단위 테스트 가능.

listener.py가 resolve_thread_ts/process_mention을 실제 의존성(resolve 클로저, Service,
Workspace)과 on_progress 콜백으로 wiring한다.
"""
from api.arxiv import get_paper_info
from api.resolvers import extract_first_url


_NO_URL_MSG = (
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


def process_mention(text, *, cache, service, workspace, resolve,
                    on_progress=lambda s: None) -> dict:
    """멘션 텍스트를 받아 요약 결과 dict를 반환한다.

    반환: {"ok": bool, "message": str, "paper_info": str|None, "paper_url": str|None}
    resolve(url, on_progress) -> ResolvedPaper | None  (주입)
    on_progress(stage) 단계: "fetching" → ("downloading") → "summarizing"
    """
    url = extract_first_url(text)
    if url is None:
        return {"ok": False, "message": _NO_URL_MSG,
                "paper_info": None, "paper_url": None}
    on_progress("fetching")
    resolved = resolve(url, on_progress=on_progress)
    if resolved is None or not resolved.text:
        return {"ok": False, "message": _UNSUPPORTED_MSG,
                "paper_info": None, "paper_url": None}

    paper_info = get_paper_info(resolved.url, resolved.title)
    on_progress("summarizing")
    summarization = service.summarize_text(paper_info, resolved.text)
    if not summarization:
        return {"ok": False,
                "message": "요약 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                "paper_info": None, "paper_url": None}

    message_content, _ = workspace.prepare_content(paper_info, "", summarization)
    return {"ok": True, "message": message_content,
            "paper_info": paper_info, "paper_url": resolved.url}
