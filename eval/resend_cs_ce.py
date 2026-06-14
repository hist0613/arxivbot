"""cs.CE 4편을 새 입력 캡(6000)으로 강제 재요약 후 재전송 (비교용).
- 요약 캐시 덮어쓰기(2048 기준 캐시 무시)
- old_paper_set에서 일시 제외해 dedup 우회
- 'Can I Buy Your KV Cache'(기존 게시분)는 제외
"""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from settings import WORKSPACE_CONFIGS, MODEL, MAX_INPUT_TOKENS_FOR_SUMMARIZATION
from api.cache import CacheManager
from api.arxiv import ArxivClient, get_paper_info
from api.agent import AutoAgent, Encoder
from api.workspace import Workspace

FIELD = "cs.CE"
EXCLUDE = "Can I Buy Your KV Cache"


def build_input(abstract, full, enc):
    s = f"Abstract: {abstract}\n\n"
    if isinstance(full, dict):
        for sec in full.values():
            if sec.get("title") != "No title found" and sec.get("content"):
                s += f"Section: {sec['title']}\n{sec['content']}\n\n"
    return enc.truncate_text(s)


async def main():
    print(f"input cap = {MAX_INPUT_TOKENS_FOR_SUMMARIZATION}")
    cfg = next(c for c in WORKSPACE_CONFIGS if c["service_type"] == "slack")
    cache = CacheManager()
    arxiv = ArxivClient(cache)
    agent = AutoAgent.from_model_name(MODEL)
    enc = Encoder(MODEL)

    papers = [p for p in arxiv.get_paper_set_of(FIELD)[:5] if EXCLUDE not in p[1]]
    print(f"resend 대상 {len(papers)}편")

    for url, title, _ in papers:
        info = get_paper_info(url, title)
        if info not in cache.paper_abstracts:
            cache.update_paper_abstracts(info, arxiv.get_paper_abstract(url))
        if info not in cache.paper_full_contents:
            link = arxiv.get_html_experimental_link(url)
            if link != "Link not found":
                cache.update_paper_full_contents(info, arxiv.get_paper_full_content(link))
        full = cache.paper_full_contents[info] if info in cache.paper_full_contents else ""
        inp = build_input(cache.paper_abstracts[info], full, enc)
        # 강제 재요약(덮어쓰기)
        cache.update_paper_summarizations(info, agent.summarize(inp))
        in_tok = len(enc.encoding.encode(inp))
        print(f"  re-summarized ({in_tok} in-tok): {title[:55]}")

    ws = Workspace(cfg)
    ws.fields = [FIELD]
    # dedup 우회: 대상 논문을 old_paper_set에서 일시 제외
    for url, title, _ in papers:
        ws.old_paper_set.discard(get_paper_info(url, title))

    threads = ws.prepare_field_threads({FIELD: papers}, cache)
    n = sum(len(t["thread_contents"]) for t in threads)
    print(f"sending {n} message(s)...")
    await ws.send_messages(threads)
    print("RESENT OK")


if __name__ == "__main__":
    asyncio.run(main())
