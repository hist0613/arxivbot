"""E2E 테스트: cs.CE 최근 5편을 실제로 크롤→요약(gpt-5.4-nano)→Slack 전송.
기존 컴포넌트(ArxivClient/AutoAgent/Workspace)를 그대로 사용한다.
전송 대상은 WORKSPACE_CONFIGS의 활성 slack 설정(seungtaek-lab #arxivbot)."""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from settings import WORKSPACE_CONFIGS, MODEL
from api.cache import CacheManager
from api.arxiv import ArxivClient, get_paper_info
from api.agent import AutoAgent, Encoder
from api.workspace import Workspace

FIELD = "cs.CE"
N = 5


def build_input(abstract, full_content, encoder):
    s = f"Abstract: {abstract}\n\n"
    if isinstance(full_content, dict):
        for sec in full_content.values():
            if sec.get("title") != "No title found" and sec.get("content"):
                s += f"Section: {sec['title']}\n{sec['content']}\n\n"
    return encoder.truncate_text(s)


async def main():
    cfg = next(c for c in WORKSPACE_CONFIGS if c["service_type"] == "slack")
    print(f"target: {cfg['workspace']} #{cfg['allowed_channel']} ({cfg['allowed_channel_id']})")
    print(f"model: {MODEL}")

    cache = CacheManager()
    arxiv = ArxivClient(cache)
    agent = AutoAgent.from_model_name(MODEL)
    encoder = Encoder(MODEL)

    papers = arxiv.get_paper_set_of(FIELD)[:N]
    print(f"fetched {len(papers)} papers from {FIELD}")
    for url, title, _ in papers:
        print("  -", title[:70])

    # 크롤(초록+본문) — 캐시에 없으면
    for url, title, _ in papers:
        info = get_paper_info(url, title)
        if info not in cache.paper_abstracts:
            cache.update_paper_abstracts(info, arxiv.get_paper_abstract(url))
        if info not in cache.paper_full_contents:
            link = arxiv.get_html_experimental_link(url)
            if link != "Link not found":
                cache.update_paper_full_contents(info, arxiv.get_paper_full_content(link))

    # 요약
    for url, title, _ in papers:
        info = get_paper_info(url, title)
        if not cache.has_paper_summarization(info):
            full = cache.paper_full_contents[info] if info in cache.paper_full_contents else ""
            inp = build_input(cache.paper_abstracts[info], full, encoder)
            cache.update_paper_summarizations(info, agent.summarize(inp))
            print(f"  summarized: {title[:55]}")

    # Slack 전송 (cs.CE 한정으로 fields 오버라이드)
    ws = Workspace(cfg)
    ws.fields = [FIELD]
    threads = ws.prepare_field_threads({FIELD: papers}, cache)
    n_msgs = sum(len(t["thread_contents"]) for t in threads)
    print(f"prepared {len(threads)} thread(s), {n_msgs} paper message(s) -> sending...")
    await ws.send_messages(threads)
    print("SENT OK")


if __name__ == "__main__":
    asyncio.run(main())
