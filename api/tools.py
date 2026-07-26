"""에이전트 도구.

논문 요약은 도구가 게시까지 끝내고 모델에는 영수증만 준다. 모델이 요약을
자기 말로 다시 옮기면 매일 쓰는 4섹션 요약 품질이 모델 판단에 걸리기 때문이다.
"""
import json
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from api.arxiv import REQUEST_HEADERS, REQUEST_TIMEOUT
from api.logger import logger

STAGE = {
    "fetching": "🔄 논문 페이지 가져오는 중…",
    "downloading": "🔄 PDF 다운로드 중…",
    "summarizing": "🔄 AI가 요약하는 중…",
}

TOOL_NAMES = ("summarize_paper", "fetch_page", "read_thread")

# 페이지 본문을 그대로 넘기면 모델 입력이 터지고, 어차피 요약에 앞부분이면
# 충분하다. 프로젝트 페이지 기준으로 2만자면 본문이 거의 다 들어온다.
MAX_PAGE_CHARS = 20000

# 이 호스트로 fetch_page가 들어오면 본문을 긁지 않고 summarize_paper로 되돌려
# 보낸다. 학회 페이지는 HTML 본문이 초록 몇 줄뿐이고, PDF까지 따라가는 건
# resolvers의 논문 경로가 한다.
PAPER_HOSTS = (
    "arxiv.org",
    "aclanthology.org",
    "openaccess.thecvf.com",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "openreview.net",
    "ojs.aaai.org",
    "ijcai.org",
    "isca-archive.org",
    "dl.acm.org",
)


def post_paper_summary(client, *, channel, thread_ts, url, prefix, process, on_posted):
    """답글 1개를 만들어 진행 상황으로 편집하다가 최종 요약으로 채운다.

    반환: {"ok": bool, "title": str, "url": str, "error": str}
    """
    posted = client.chat_postMessage(
        channel=channel, text=prefix + STAGE["fetching"], thread_ts=thread_ts
    )
    ts = posted["ts"]
    last = {"text": STAGE["fetching"]}

    # 새 메시지를 쌓지 않고 같은 답글을 편집한다. 단계마다 새로 올리면
    # 채널 알림이 논문 하나당 서너 번 울린다.
    def on_progress(stage):
        msg = STAGE.get(stage)
        if msg and msg != last["text"]:
            last["text"] = msg
            client.chat_update(channel=channel, ts=ts, text=prefix + msg)

    try:
        result = process(url, on_progress)
    except Exception as e:
        logger.error(f"on-demand process_url error for {url}: {e}")
        client.chat_update(
            channel=channel, ts=ts, text=f"처리 중 오류가 났어요: {e}\n({url})"
        )
        return {"ok": False, "title": "", "url": url, "error": str(e)}

    text = result["message"] if result["ok"] else f"{result['message']}\n({url})"
    blocks = result.get("blocks")
    # 블록이 거절당하면 요약이 통째로 날아가므로 text로 재시도한다.
    try:
        client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
            **({"blocks": blocks} if blocks else {}),
        )
    except Exception as e:
        if not blocks:
            raise
        logger.error(f"chat_update with blocks failed ({e}); text로 폴백")
        client.chat_update(channel=channel, ts=ts, text=text)

    if result["ok"]:
        on_posted(
            ts=ts, paper_info=result["paper_info"], paper_url=result["paper_url"]
        )
        logger.info(f"on-demand summary posted: {result['paper_info']}")
        return {
            "ok": True,
            "title": result["paper_info"],
            "url": result["paper_url"],
            "error": "",
        }
    return {"ok": False, "title": "", "url": url, "error": result["message"]}


def download_page(url: str):
    """일반 웹페이지에서 제목과 본문 텍스트를 뽑는다."""
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return title, text[:MAX_PAGE_CHARS]


def build_page_fetcher(store, download=download_page):
    """pages 테이블을 앞에 둔 fetch_page 구현을 만든다."""

    def fetch(url: str) -> dict:
        host = (urlparse(url).netloc or "").lower()
        if any(host.endswith(h) for h in PAPER_HOSTS):
            return {
                "hint": "논문 호스트다. summarize_paper 도구를 써라.",
                "title": "",
                "text": "",
            }
        cached = store.get_page(url)
        if cached and cached.get("text"):
            return {"title": cached.get("title", ""), "text": cached["text"]}
        title, text = download(url)
        if not text:
            return {"error": "페이지에서 본문을 얻지 못했다."}
        store.put_page(url, title, text)
        return {"title": title, "text": text}

    return fetch


def build_tools(*, post_summary, fetch_page, read_thread):
    """도구 명세와 dispatch를 만든다. 실제 동작은 전부 주입받는다."""
    specs = [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "summarize_paper",
            "description": (
                "논문 URL(arXiv, ACL, CVPR/ICCV, NeurIPS, ICML, OpenReview, AAAI, "
                "IJCAI, Interspeech, 직접 PDF)을 요약해 스레드에 직접 게시한다. "
                "요약 본문은 반환하지 않고 게시 결과만 돌려준다. 논문 링크가 있으면 "
                "반드시 이 도구를 쓴다. 링크가 여러 개면 각각 부른다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "논문 URL"}
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "fetch_page",
            "description": (
                "일반 웹페이지 본문을 가져온다. 프로젝트 페이지·블로그·릴리스 노트용. "
                "논문 URL이면 summarize_paper를 쓴다."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "read_thread",
            "description": "문맥에서 접힌 이전 스레드 메시지를 원문으로 가져온다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "before_ts": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["before_ts"],
                "additionalProperties": False,
            },
        },
    ]

    def dispatch(name: str, args: dict) -> str:
        try:
            if name == "summarize_paper":
                r = post_summary(args["url"])
                # 요약 본문은 일부러 빼고 영수증만 돌려준다. 본문을 주면
                # 모델이 그걸 자기 말로 다시 옮겨 스레드에 두 번 올리거나,
                # 4섹션 구조를 흐트러뜨린 채 옮겨 적는다.
                return json.dumps(
                    {
                        "posted": bool(r.get("ok")),
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "error": r.get("error", ""),
                    },
                    ensure_ascii=False,
                )
            if name == "fetch_page":
                return json.dumps(fetch_page(args["url"]), ensure_ascii=False)
            if name == "read_thread":
                return json.dumps(
                    read_thread(args["before_ts"], args.get("limit", 10)),
                    ensure_ascii=False,
                )
            return json.dumps({"error": f"모르는 도구: {name}"}, ensure_ascii=False)
        except KeyError as e:
            return json.dumps({"error": f"빠진 인자: {e}"}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"tool {name} 실패: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return specs, dispatch
