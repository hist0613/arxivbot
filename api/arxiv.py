import time
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup

from api.cache import CacheManager
from api.logger import logger
from settings import MAX_LLM_TRIALS, MAX_NB_CRAWL


def get_paper_info(paper_url: str, paper_title: str) -> str:
    return "{} ({})".format(paper_title, paper_url)


class ArxivClient:
    def __init__(self, cache: CacheManager):
        self.cache = cache

    def crawl_arxiv(self, field: str) -> set[str]:
        # collect new arXiv papers
        logger.info("- Collecting new arXiv papers...")
        paper_set = self.get_paper_set_of(field)

        # abstract crawling
        logger.info("- Crawling the abstracts of papers ...")
        for paper_url, paper_title, _ in tqdm(paper_set):
            # remove duplicates
            paper_info = get_paper_info(paper_url, paper_title)
            if paper_info in self.cache.paper_abstracts:
                continue

            paper_abstract = self.get_paper_abstract(paper_url)
            self.cache.update_paper_abstracts(paper_info, paper_abstract)

            # remove duplicates
            if paper_info in self.cache.paper_full_contents:
                continue

            html_experimental_link = self.get_html_experimental_link(paper_url)
            if html_experimental_link != "Link not found":
                self.cache.update_paper_full_contents(
                    paper_info, self.get_paper_full_content(html_experimental_link)
                )

        return paper_set

    def get_paper_set_of(self, field: str) -> list[tuple[str, str, str]]:
        # list_url = "https://arxiv.org/list/{}/recent".format(field)
        list_url = "https://arxiv.org/list/{}/pastweek?skip=0&show={}".format(
            field, MAX_NB_CRAWL
        )
        for trial in range(MAX_LLM_TRIALS):
            try:
                list_page = requests.get(list_url)
                if list_page.status_code == requests.codes.ok:
                    break
            except requests.exceptions.ConnectionError as e:
                logger.info(e)
                time.sleep(trial * 30 + 15)
        list_soup = BeautifulSoup(list_page.text, "html.parser")

        # <dd> 태그 추출
        dt_tags = list_soup.find_all("dt")
        dd_tags = list_soup.find_all("dd")

        papers_url = []
        papers_title = []
        papers_comment = []
        for dt_tag, dd_tag in zip(dt_tags, dd_tags):
            # <dd> 태그 내에서 <span class="list-identifier"> 태그 추출
            # identifier_tag = dt_tag.find("span", class_="list-identifier")
            paper_url = dt_tag.find("a", {"title": "Abstract"})["href"]
            paper_url = "https://arxiv.org" + paper_url
            papers_url.append(paper_url)

            # <dd> 태그 내에서 <div class="list-title"> 태그 추출
            title_tag = dd_tag.find("div", class_="list-title")
            paper_title = title_tag.text.strip().strip("Title:").strip()
            papers_title.append(paper_title)

            # <dd> 태그 내에서 <div class="list-comments"> 태그 추출
            comment_tag = dd_tag.find("div", class_="list-comments")
            if comment_tag:
                papers_comment.append(comment_tag.text.strip())
            else:
                papers_comment.append("")

        papers = list(zip(papers_url, papers_title, papers_comment))

        return papers

    def get_paper_abstract(self, paper_url: str) -> str:
        for trial in range(MAX_LLM_TRIALS):
            try:
                paper_page = requests.get(paper_url)
                if paper_page.status_code == 200:
                    break
            except requests.exceptions.ConnectionError as e:
                logger.info(e)
                time.sleep(trial * 30 + 15)
        paper_soup = BeautifulSoup(paper_page.text, "html.parser")

        paper_abstract = (
            paper_soup.find_all("blockquote", class_="abstract")[0]
            .text.strip()
            .replace("Abstract:  ", "")
            .replace("\n", " ")
        )
        return paper_abstract

    def get_html_experimental_link(self, paper_url: str) -> str:
        response = requests.get(paper_url)
        soup = BeautifulSoup(response.text, "html.parser")

        # 'HTML (experimental)' 링크 찾기
        html_link = soup.find("a", string="HTML (experimental)")
        if html_link:
            return html_link["href"]  # 링크 추출
        else:
            return "Link not found"

    def get_paper_full_content(self, paper_url: str) -> dict[str, dict[str, str]]:
        for trial in range(MAX_LLM_TRIALS):
            try:
                paper_page = requests.get(paper_url)
                if paper_page.status_code == 200:
                    break
            except requests.exceptions.ConnectionError as e:
                logger.info(e)
                time.sleep(trial * 30 + 15)
        paper_soup = BeautifulSoup(paper_page.text, "html.parser")

        sections = paper_soup.find_all("section")
        section_dict = {}

        for section in sections:
            section_id = section.get("id")
            if section_id:
                # <h2> 태그 내에서 제목 찾기
                title_tag = section.find("h2")
                if title_tag:
                    # <span> 태그 내용 제거
                    if title_tag.find("span"):
                        title_tag.span.decompose()
                    section_title = title_tag.text.strip()
                else:
                    section_title = "No title found"

                # 섹션의 전체 텍스트 내용을 추출 (제목 제외)
                section_content = "\n".join(
                    [para.text.strip() for para in section.find_all("p")]
                )

                # 사전에 섹션 ID, 제목, 내용 저장
                section_dict[section_id] = {
                    "title": section_title,
                    "content": section_content,
                }

        return section_dict
