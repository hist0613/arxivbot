"""On-demand(@멘션) 요약의 순수 코어 — Slack/네트워크에서 분리되어 단위 테스트 가능.

listener.py가 이 모듈의 process_mention/resolve_thread_ts/build_fetch_paper를
실제 의존성(arxiv 조회, Service, Workspace)과 함께 wiring한다.
"""
from api.arxiv import parse_arxiv_ref, get_paper_title, get_paper_info


_NO_URL_MSG = (
    "arxiv 링크를 함께 멘션해 주세요 "
    "(예: @arxivbot https://arxiv.org/abs/2501.12345)"
)


def resolve_thread_ts(event: dict) -> str:
    """멘션이 스레드 안이면 그 스레드, 아니면 멘션 메시지 자체에 답글."""
    return event.get("thread_ts") or event["ts"]


def process_mention(text, *, cache, service, workspace, fetch_paper) -> dict:
    """멘션 텍스트를 받아 요약 결과 dict를 반환한다.

    반환: {"ok": bool, "message": str,
           "paper_info": str|None, "paper_url": str|None}
    fetch_paper(url) -> (title, abstract, full_content)  (주입)
    """
    url = parse_arxiv_ref(text)
    if url is None:
        return {"ok": False, "message": _NO_URL_MSG,
                "paper_info": None, "paper_url": None}
    try:
        title, abstract, full_content = fetch_paper(url)
    except Exception as e:
        return {"ok": False, "message": f"논문을 가져오지 못했어요: {e}",
                "paper_info": None, "paper_url": None}
    if not abstract:
        return {"ok": False, "message": "초록을 찾지 못했어요. 링크를 확인해 주세요.",
                "paper_info": None, "paper_url": None}

    paper_info = get_paper_info(url, title)
    summarization = service.summarize_one(paper_info, abstract, full_content)
    if not summarization:
        return {"ok": False,
                "message": "요약 생성에 실패했어요. 잠시 후 다시 시도해 주세요.",
                "paper_info": None, "paper_url": None}

    message_content, _ = workspace.prepare_content(paper_info, "", summarization)
    return {"ok": True, "message": message_content,
            "paper_info": paper_info, "paper_url": url}


def build_fetch_paper(arxiv_client, cache):
    """캐시를 재사용하며 (title, abstract, full_content)를 가져오는 클로저 생성.
    abstract/full_content는 배치와 동일하게 캐시에 채워 넣는다."""
    def fetch_paper(url):
        title = get_paper_title(url)
        paper_info = get_paper_info(url, title)

        cached_abs = cache.paper_abstracts.get(paper_info) if hasattr(
            cache.paper_abstracts, "get") else None
        if cached_abs:
            abstract = cached_abs
        else:
            abstract = arxiv_client.get_paper_abstract(url)
            cache.update_paper_abstracts(paper_info, abstract)

        cached_full = cache.paper_full_contents.get(paper_info) if hasattr(
            cache.paper_full_contents, "get") else None
        if cached_full:
            full_content = cached_full
        else:
            full_content = ""
            try:
                html_link = arxiv_client.get_html_experimental_link(url)
                if html_link != "Link not found":
                    full_content = arxiv_client.get_paper_full_content(html_link)
                    cache.update_paper_full_contents(paper_info, full_content)
            except Exception:
                full_content = ""

        return title, abstract, full_content
    return fetch_paper
