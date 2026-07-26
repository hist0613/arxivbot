"""에이전트 도구.

논문 요약은 도구가 게시까지 끝내고 모델에는 영수증만 준다. 모델이 요약을
자기 말로 다시 옮기면 매일 쓰는 4섹션 요약 품질이 모델 판단에 걸리기 때문이다.
"""
import ipaddress
import json
import socket
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from api.arxiv import REQUEST_HEADERS, REQUEST_TIMEOUT, parse_arxiv_ref
from api.logger import logger
from api.resolvers import extract_urls

STAGE = {
    "fetching": "🔄 논문 페이지 가져오는 중…",
    "downloading": "🔄 PDF 다운로드 중…",
    "summarizing": "🔄 AI가 요약하는 중…",
}

TOOL_NAMES = ("summarize_paper", "fetch_page", "read_thread")

# 페이지 본문을 그대로 넘기면 모델 입력이 터지고, 어차피 요약에 앞부분이면
# 충분하다. 프로젝트 페이지 기준으로 2만자면 본문이 거의 다 들어온다.
MAX_PAGE_CHARS = 20000
# 본문 2만자를 뽑기 위해 받을 바이트 상한. 이보다 큰 페이지는 앞부분만 쓴다.
MAX_PAGE_BYTES = 4 * 1024 * 1024
# 블로그·릴리스 노트는 갱신된다. 이 시간이 지나면 다시 받는다.
PAGE_CACHE_TTL_SEC = 7 * 24 * 3600

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


def _canonical_url(url: str) -> str:
    """arXiv는 abs/pdf/버전 표기가 갈리므로 id로 통일한다."""
    return parse_arxiv_ref(url) or url.rstrip("/")


def collect_allowed_urls(*texts) -> list:
    """대화에 실제로 등장한 URL 목록. 등장 순서를 지키고 중복은 뺀다."""
    seen, urls = set(), []
    for text in texts:
        for url in extract_urls(text or ""):
            key = _canonical_url(url)
            if key not in seen:
                seen.add(key)
                urls.append(url)
    return urls


def match_allowed_url(url: str, allowed: list):
    """모델이 넘긴 URL을 대화에 등장한 URL과 대조한다.

    모델이 URL을 지어내거나 잘못 옮겨 적으면 엉뚱한 논문을 요약해 올리게
    된다. 대조에 걸리면 대화에 있던 원문 URL을 돌려주고(모델이 다시 쓴
    문자열 대신 그걸 쓴다), 없으면 None이다.
    """
    if not url:
        return None
    key = _canonical_url(url)
    for candidate in allowed:
        if _canonical_url(candidate) == key:
            return candidate
    return None


def post_paper_summary(
    client, *, channel, thread_ts, url, prefix, process, on_posted, existing_ts=None
):
    """답글 1개를 만들어 진행 상황으로 편집하다가 최종 요약으로 채운다.

    existing_ts를 주면 새 답글을 만들지 않고 그 메시지를 이어서 쓴다. 리스너가
    이미 올려둔 "생각하는 중…"을 재활용해, 링크 하나짜리 흔한 경우에 진행
    표시가 둘로 늘었다 하나가 사라지는 어수선함을 없앤다.

    반환: {"ok": bool, "title": str, "url": str, "error": str}
    """
    if existing_ts:
        ts = existing_ts
        client.chat_update(
            channel=channel, ts=ts, text=prefix + STAGE["fetching"]
        )
    else:
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
        # 요약은 이미 스레드에 떠 있다. 여기서 예외가 새어 나가면 모델이
        # 게시 실패로 알고 도구를 다시 불러 같은 요약이 두 번 올라간다.
        try:
            on_posted(
                ts=ts, paper_info=result["paper_info"], paper_url=result["paper_url"]
            )
        except Exception as e:
            logger.error(f"리액션 store 등록 실패(요약은 게시됨): {e}")
        logger.info(f"on-demand summary posted: {result['paper_info']}")
        return {
            "ok": True,
            "title": result["paper_info"],
            "url": result["paper_url"],
            "error": "",
        }
    return {"ok": False, "title": "", "url": url, "error": result["message"]}


def check_fetchable(url: str):
    """가져와도 되는 주소인지. 문제가 있으면 이유 문자열을 돌려준다.

    fetch_page의 인자는 모델이 만든다. 모델은 우리가 방금 가져온 페이지
    내용을 읽고 다음 행동을 정하므로, 페이지에 심어둔 지시문이 내부망
    주소를 부르게 만들 수 있다. 그 응답이 다시 모델을 거쳐 Slack에 실린다.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        return "http(s) 주소만 가져올 수 있다."
    host = (parts.hostname or "").lower()
    if not host:
        return "호스트가 없는 주소다."
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return "내부 주소는 가져오지 않는다."
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return "주소를 찾을 수 없다."
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        # 사설망·루프백·링크로컬(클라우드 메타데이터 169.254.169.254 포함)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return "내부 주소는 가져오지 않는다."
    return None


def download_page(url: str):
    """일반 웹페이지에서 제목과 본문 텍스트를 뽑는다."""
    with requests.get(
        url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, stream=True
    ) as r:
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        # 큰 바이너리를 통째로 메모리에 올리지 않는다. 어차피 HTML만 쓴다.
        if content_type and not any(
            t in content_type for t in ("text/html", "text/plain", "xml")
        ):
            return "", ""
        chunks, size = [], 0
        for chunk in r.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_PAGE_BYTES:
                break
        raw = b"".join(chunks)
    soup = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return title, text[:MAX_PAGE_CHARS]


def build_page_fetcher(store, download=download_page, now=time.time,
                       check=check_fetchable):
    """pages 테이블을 앞에 둔 fetch_page 구현을 만든다.

    check는 주소 검사기. 기본값은 DNS를 실제로 조회하므로 단위 테스트에서는
    통과시키는 함수를 넣는다(검사기 자체는 따로 테스트한다).
    """

    def fetch(url: str) -> dict:
        host = (urlparse(url).netloc or "").lower()
        if any(host.endswith(h) for h in PAPER_HOSTS):
            return {
                "hint": "논문 호스트다. summarize_paper 도구를 써라.",
                "title": "",
                "text": "",
            }
        problem = check(url)
        if problem:
            logger.info(f"fetch_page 거절({url!r}): {problem}")
            return {"error": problem}
        cached = store.get_page(url)
        # 블로그·릴리스 노트는 내용이 바뀐다. 오래된 건 다시 받는다.
        if (
            cached
            and cached.get("text")
            and now() - (cached.get("fetched_at") or 0) < PAGE_CACHE_TTL_SEC
        ):
            return {"title": cached.get("title", ""), "text": cached["text"]}
        title, text = download(url)
        if not text:
            return {"error": "페이지에서 본문을 얻지 못했다."}
        # 저장 시각도 같은 시계로 찍어야 TTL 계산이 어긋나지 않는다.
        store.put_page(url, title, text, fetched_at=now())
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
        # 인자 확인은 try 밖에서. 안에서 하면 도구 안쪽에서 올라온 KeyError까지
        # "빠진 인자"로 보고돼 모델이 같은 인자로 다시 부른다.
        if name in ("summarize_paper", "fetch_page") and not args.get("url"):
            return json.dumps({"error": "url 인자가 필요하다."}, ensure_ascii=False)
        if name == "read_thread" and not args.get("before_ts"):
            return json.dumps(
                {"error": "before_ts 인자가 필요하다."}, ensure_ascii=False
            )
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
        except Exception as e:
            logger.error(f"tool {name} 실패: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    return specs, dispatch
