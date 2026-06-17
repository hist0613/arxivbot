import concurrent.futures
import time
from threading import Lock
from tqdm import tqdm
from typing import Union

from api.arxiv import ArxivClient, get_paper_info
from api.agent import Agent, Encoder
from api.cache import CacheManager
from api.logger import logger
from settings import NB_THREADS, TIME_PAUSE_CRAWL_SEC


class Service:
    def __init__(
        self,
        arxiv: ArxivClient,
        agent: Agent,
        encoder: Encoder,
        cache: CacheManager,
    ):
        self.arxiv = arxiv
        self.agent = agent
        self.encoder = encoder
        self.cache = cache

    def crawl_arxiv(self, fields: list[str]) -> dict[str, list[tuple[str, str, str]]]:
        new_papers = {}
        for i, field in enumerate(fields):
            logger.info("Processing {} field...".format(field))
            if i > 0:
                time.sleep(TIME_PAUSE_CRAWL_SEC)  # arXiv throttle 예방
            paper_set = self.arxiv.crawl_arxiv(field)
            self.summarize_arxiv(paper_set)
            new_papers[field] = paper_set
        return new_papers

    def summarize_arxiv(self, paper_set):
        """
        Summarize the abstract of papers by the agent and update the cache.
        """
        logger.info(
            f"- Summarizing the abstract of papers by {self.agent.model_name}..."
        )

        cache_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NB_THREADS) as executor:
            futures = {}
            for paper_url, paper_title, _ in tqdm(paper_set):
                # remove duplicates
                paper_info = get_paper_info(paper_url, paper_title)
                if self.cache.has_paper_summarization(paper_info):
                    continue

                paper_abstract = self.cache.paper_abstracts[paper_info]
                paper_full_content = self.cache.paper_full_contents[paper_info]

                summarization_input = self.prepare_summarization_input(
                    paper_abstract, paper_full_content
                )
                futures[executor.submit(self.agent.summarize, summarization_input)] = (
                    paper_info
                )

            for f in concurrent.futures.as_completed(futures):
                paper_info = futures[f]
                summarization = f.result()
                with cache_lock:
                    self.cache.update_paper_summarizations(paper_info, summarization)

    def summarize_one(
        self, paper_info: str, paper_abstract: str, paper_full_content
    ) -> str:
        """단건 요약. 캐시에 있으면 재사용, 없으면 배치와 동일 입력으로 요약 후
        비어 있지 않을 때만 캐시에 저장한다."""
        if self.cache.has_paper_summarization(paper_info):
            return self.cache.paper_summarizations[paper_info]
        summarization_input = self.prepare_summarization_input(
            paper_abstract, paper_full_content
        )
        summarization = self.agent.summarize(summarization_input)
        if summarization:
            self.cache.update_paper_summarizations(paper_info, summarization)
        return summarization

    def prepare_summarization_input(
        self,
        paper_abstract: str,
        paper_full_content: Union[dict, str],
    ):
        summarization_input = f"Abstract: {paper_abstract}\n\n"
        if isinstance(paper_full_content, dict):
            for section in paper_full_content.values():
                if section["title"] != "No title found" and section["content"]:
                    summarization_input += (
                        f"Section: {section['title']}\n{section['content']}\n\n"
                    )
        return self.encoder.truncate_text(summarization_input)
