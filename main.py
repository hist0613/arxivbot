import asyncio
from datetime import datetime, timezone

from slack_sdk import WebClient

from api.logger import logger
from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.arxiv import get_paper_info
from api.gpu_batch import GpuBatch, Remote
from api.reactions import load_store, save_store, harvest_reactions
from settings import (
    WORKSPACE_CONFIGS,
    MODEL,
    HARVEST_WINDOW_DAYS,
    SUMMARY_BACKEND,
    GPU_SSH_HOST,
    GPU_REMOTE_DIR,
    GPU_BATCH_DIR,
    GPU_BATCH_MAX_AGE_HOURS,
    FIGURE_DIR,
)


def make_gpu_batch(cache):
    if not GPU_SSH_HOST or not GPU_REMOTE_DIR:
        raise RuntimeError("SUMMARY_BACKEND=gpu 인데 .env 에 GPU_SSH_HOST·GPU_REMOTE_DIR 가 없다")
    return GpuBatch(cache, Remote(GPU_SSH_HOST, GPU_REMOTE_DIR), GPU_BATCH_DIR, FIGURE_DIR,
                    GPU_BATCH_MAX_AGE_HOURS)


async def main():
    # 전체 흐름은 다음과 같습니다.
    # 1. arXiv의 새 논문을 수집합니다.
    # 2. 수집한 논문의 초록을 크롤링합니다.
    # 3. 수집한 논문의 초록을 GPT-4o로 요약합니다.
    # 4. 요약된 내용을 Slack 또는 Discord에 전송합니다.
    workspaces = [Workspace(workspace_config) for workspace_config in WORKSPACE_CONFIGS]

    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(
        arxiv_client,
        AutoAgent.from_model_name(MODEL),
        Encoder(MODEL),
        cache,
    )

    fields = set(field for workspace in workspaces for field in workspace.fields)
    use_gpu = SUMMARY_BACKEND == "gpu"
    gpu = make_gpu_batch(cache) if use_gpu else None
    if gpu:
        # 지난 실행 때 보낸 배치 중 끝난 것을 먼저 받아 온다
        gpu.collect()

    new_papers = service.crawl_arxiv(fields, summarize=not use_gpu)

    if gpu:
        papers = [(url, title, get_paper_info(url, title))
                  for field in new_papers for url, title, _ in new_papers[field]]
        try:
            gpu.submit(papers)
        except Exception as e:  # 서버가 안 받아도 이미 요약된 논문 게시는 계속한다
            logger.error(f"[gpu] submit 실패: {e}")
        # 요약이 아직 없는 논문은 이번에 게시하지 않는다. 결과가 온 뒤의 실행에서 게시된다.
        new_papers = {
            field: [p for p in papers_ if cache.has_paper_summarization(get_paper_info(p[0], p[1]))]
            for field, papers_ in new_papers.items()
        }

    for workspace in workspaces:
        # prepare messages
        threads = workspace.prepare_field_threads(new_papers, service.cache)

        # send messages
        await workspace.send_messages(threads)

    # 리액션 수확 (slack 워크스페이스 한정)
    for workspace in workspaces:
        if workspace.service_type != "slack":
            continue
        store = load_store()
        client = WebClient(workspace.slack_token)
        bot_user_id = client.auth_test()["user_id"]
        n = harvest_reactions(
            client,
            store,
            window_days=HARVEST_WINDOW_DAYS,
            bot_user_id=bot_user_id,
            now=datetime.now(timezone.utc),
        )
        save_store(store)
        logger.info(f"Harvested reactions for {n} papers ({workspace.workspace}).")


if __name__ == "__main__":
    asyncio.run(main())
