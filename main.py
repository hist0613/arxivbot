import asyncio
import time
from datetime import datetime, timezone

import git
from slack_sdk import WebClient

from api.logger import logger
from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.reactions import load_store, save_store, harvest_reactions
from settings import (
    WORKSPACE_CONFIGS,
    MODEL,
    REPO_DIR,
    SUMMARIES_DIR,
    HARVEST_WINDOW_DAYS,
)


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
    new_papers = service.crawl_arxiv(fields)

    for workspace in workspaces:
        # prepare messages
        threads = workspace.prepare_field_threads(new_papers, service.cache)

        # send messages
        await workspace.send_messages(threads)

        # save summaries
        workspace.save_summaries(threads)

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

    repo = git.Repo(REPO_DIR)
    repo.git.add(SUMMARIES_DIR)

    if repo.is_dirty():
        repo.git.commit("-m", f"\"Update summaries: {time.strftime('%Y-%m-%d')}\"")
        repo.git.push()
    else:
        logger.info("No changes to commit.")


if __name__ == "__main__":
    asyncio.run(main())
