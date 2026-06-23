import concurrent.futures
import time
from threading import Lock
from tqdm import tqdm
from typing import Union

from api.arxiv import ArxivClient, get_paper_info
from api.agent import Agent, Encoder
from api.cache import CacheManager
from api.logger import logger
from prompts import is_current_summary_schema
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

    def summarize_text(self, paper_info: str, text: str) -> str:
        """출처 무관 통일 진입점. 현재 4섹션 스키마 캐시면 재사용(self-healing),
        아니면 truncate 후 요약하고 비어있지 않을 때만 저장한다."""
        cached = self.cache.paper_summarizations.get(paper_info, "")
        if is_current_summary_schema(cached):
            return cached
        summarization = self.agent.summarize(self.encoder.truncate_text(text))
        if summarization:
            self.cache.update_paper_summarizations(paper_info, summarization)
        return summarization

    def summarize_one(
        self, paper_info: str, paper_abstract: str, paper_full_content
    ) -> str:
        """arXiv 구조화 입력(abstract+섹션)을 만들어 summarize_text로 위임."""
        return self.summarize_text(
            paper_info,
            self.prepare_summarization_input(paper_abstract, paper_full_content),
        )

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
