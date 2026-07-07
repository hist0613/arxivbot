"""On-demand(@멘션) 요약의 순수 코어 — Slack/네트워크에서 분리되어 단위 테스트 가능.

listener.py가 resolve_thread_ts/extract_targets/process_url을 실제 의존성(resolve
클로저, Service, Workspace)과 on_progress 콜백으로 wiring한다.
"""
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

    반환: {"ok": bool, "message": str, "paper_info": str|None, "paper_url": str|None}
    resolve(url, on_progress) -> ResolvedPaper | None  (주입)
    on_progress(stage) 단계: "fetching" → ("downloading") → "summarizing"
    """
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
    note = getattr(resolved, "note", "")
    if note:
        message_content += f"\n\n{note}"
    return {"ok": True, "message": message_content,
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
