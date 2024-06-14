import os
import time
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from slack_sdk import WebClient
import tiktoken
import git

from settings import *
from gpt3 import get_openai_summarization

base_dir = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
old_paper_set_path = os.path.join(base_dir, "old_paper_set_{}.pickle")
paper_abstracts_path = os.path.join(base_dir, "paper_abstracts.pickle")
paper_summarizations_path = os.path.join(base_dir, "paper_summarizations.pickle")
paper_questions_path = os.path.join(base_dir, "paper_questions.pickle")
paper_full_contents_path = os.path.join(base_dir, "paper_full_contents.pickle")
encoding = tiktoken.encoding_for_model(MODEL)
# encoding = tiktoken.get_encoding(MODEL)

summaries_dir = os.path.join(base_dir, "summaries")
today_summaries_dir = os.path.join(summaries_dir, time.strftime("%Y-%m-%d"))
os.makedirs(today_summaries_dir, exist_ok=True)


def get_old_paper_set(workspace):
    if os.path.exists(old_paper_set_path.format(workspace)):
        with open(old_paper_set_path.format(workspace), "rb") as fp:
            old_paper_set = pickle.load(fp)
    else:
        old_paper_set = set()
    return old_paper_set


def get_paper_abstracts():
    if os.path.exists(paper_abstracts_path):
        with open(paper_abstracts_path, "rb") as fp:
            paper_abstracts = pickle.load(fp)
    else:
        paper_abstracts = defaultdict(str)
    return paper_abstracts


def get_paper_summarizations():
    if os.path.exists(paper_summarizations_path):
        with open(paper_summarizations_path, "rb") as fp:
            paper_summarizations = pickle.load(fp)
    else:
        paper_summarizations = defaultdict(str)
    return paper_summarizations


def get_paper_questions():
    if os.path.exists(paper_questions_path):
        with open(paper_questions_path, "rb") as fp:
            paper_questions = pickle.load(fp)
    else:
        paper_questions = defaultdict(str)
    return paper_questions


def get_paper_full_contents():
    if os.path.exists(paper_full_contents_path):
        with open(paper_full_contents_path, "rb") as fp:
            paper_full_contents = pickle.load(fp)
    else:
        paper_full_contents = defaultdict(str)
    return paper_full_contents


def get_paper_set_of(field):
    paper_set = []

    # list_url = "https://arxiv.org/list/{}/recent".format(field)
    list_url = "https://arxiv.org/list/{}/pastweek?skip=0&show={}".format(
        field, MAX_NB_CRAWL
    )
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            list_page = requests.get(list_url)
            if list_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
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


def get_paper_info(paper_url, paper_title):
    return "{} ({})".format(paper_title, paper_url)


def get_paper_abstract(paper_url):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    paper_soup = BeautifulSoup(paper_page.text, "html.parser")

    paper_abstract = (
        paper_soup.find_all("blockquote", class_="abstract")[0]
        .text.strip()
        .replace("Abstract:  ", "")
        .replace("\n", " ")
    )
    return paper_abstract


def get_html_experimental_link(paper_url):
    response = requests.get(paper_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # 'HTML (experimental)' 링크 찾기
    html_link = soup.find("a", string="HTML (experimental)")
    if html_link:
        return html_link["href"]  # 링크 추출
    else:
        return "Link not found"


def get_paper_full_content(paper_url):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
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


def has_new_papers(new_papers, old_paper_set):
    for paper_url, paper_title, _ in new_papers:
        paper_info = get_paper_info(paper_url, paper_title)
        if paper_info not in old_paper_set:
            return True
    return False


def truncate_text(text):
    return encoding.decode(
        encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )[:MAX_INPUT_TOKENS_FOR_SUMMARIZATION]
    )


def main():
    for workspace in WORKSPACES:
        workspace_name = f"{workspace['workspace']}-{workspace['allowed_channel']}"

        # collect new arXiv papers
        print("Collecting new arXiv papers...")
        new_papers = defaultdict(list)
        for field in workspace["fields"]:
            paper_set = get_paper_set_of(field)
            new_papers[field] = paper_set

        print("Connecting", workspace["workspace"], "...")

        sc = WebClient(workspace["slack_token"])

        old_paper_set = get_old_paper_set(workspace_name)

        # abstract crawling
        print("Crawling the abstracts of papers ...")
        paper_abstracts = get_paper_abstracts()
        paper_full_contents = get_paper_full_contents()
        for field in workspace["fields"]:
            print("  - Processing {} field...".format(field))
            for paper_url, paper_title, _ in tqdm(new_papers[field]):
                # remove duplicates
                paper_info = get_paper_info(paper_url, paper_title)
                if paper_info in paper_abstracts:
                    continue

                paper_abstract = get_paper_abstract(paper_url)
                paper_abstracts[paper_info] = paper_abstract

                with open(paper_abstracts_path, "wb") as fp:
                    pickle.dump(paper_abstracts, fp)

                if paper_info in paper_full_contents:
                    continue

                html_experimental_link = get_html_experimental_link(paper_url)
                if html_experimental_link != "Link not found":
                    paper_full_contents[paper_info] = get_paper_full_content(
                        html_experimental_link
                    )

                    with open(paper_full_contents_path, "wb") as fp:
                        pickle.dump(paper_full_contents, fp)

        if SHOW_SUMMARIZATION:
            print(f"Summarizing the abstract of papers by {MODEL}...")
            paper_summarizations = get_paper_summarizations()

            for field in workspace["fields"]:
                print("  - Processing {} field...".format(field))

                all_papers = list(new_papers[field])
                for i in tqdm(range(0, len(all_papers), NB_THREADS)):
                    subset_papers = all_papers[i : i + NB_THREADS]

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = {}

                        for paper_url, paper_title, _ in subset_papers:
                            # remove duplicates
                            paper_info = get_paper_info(paper_url, paper_title)
                            if (
                                paper_info in paper_summarizations
                                and paper_summarizations[paper_info] != ""
                            ):
                                continue

                            paper_abstract = paper_abstracts[paper_info]
                            paper_full_content = paper_full_contents[paper_info]

                            summarization_input = f"Abstract: {paper_abstract}\n\n"
                            if type(paper_full_content) is not str:
                                for _, section in paper_full_content.items():
                                    if section["title"] == "No title found":
                                        continue
                                    if section["content"] == "":
                                        continue

                                    summarization_input += f"Section: {section['title']}\n{section['content']}\n\n"

                            summarization_input = truncate_text(summarization_input)

                            futures[
                                executor.submit(
                                    get_openai_summarization, summarization_input
                                )
                            ] = paper_info

                        for f in concurrent.futures.as_completed(futures):
                            paper_summarizations[futures[f]] = f.result()

                    # pickling after summarization
                    with open(paper_summarizations_path, "wb") as fp:
                        pickle.dump(paper_summarizations, fp)

        nb_total_messages = 0
        nb_messages = 0
        for field in tqdm(workspace["fields"]):
            if not has_new_papers(new_papers[field], old_paper_set):
                continue

            # make a parent message first
            sc.chat_postMessage(
                channel=workspace["allowed_channel"],
                text="New uploads on arXiv({})\n".format(field),
            )

            today_summaries_field_path = os.path.join(
                today_summaries_dir, field + ".md"
            )
            fp = open(today_summaries_field_path, "w", encoding="utf-8")

            # get the timestamp of the parent messagew
            result = sc.conversations_history(channel=workspace["allowed_channel_id"])
            conversation_history = result["messages"]  # [0] is the most recent message
            message_ts = conversation_history[0]["ts"]

            # make a thread by replying to the parent message
            for paper_url, paper_title, paper_comment in new_papers[field]:
                paper_info = get_paper_info(paper_url, paper_title)

                # remove duplicates
                if paper_info in old_paper_set:
                    continue

                content = paper_info
                file_content = "### " + paper_info + "\n"
                if paper_comment != "":
                    content += f"\n{paper_comment}"
                    file_content += f"{paper_comment}\n\n"
                if SHOW_SUMMARIZATION:
                    paper_summarization = json.loads(paper_summarizations[paper_info])
                    if type(paper_summarization) is list:
                        paper_summarization = paper_summarization[0]
                    for key, value in paper_summarization.items():
                        content += f"\n\n*{key}*: {value}"
                        file_content += f"- **{key}**: {value}\n\n"

                old_paper_set.add(paper_info)

                sc.chat_postMessage(
                    channel=workspace["allowed_channel"],
                    text=content,
                    thread_ts=message_ts,
                )

                fp.write(file_content + "\n\n")

                nb_total_messages += 1
                nb_messages += 1
                if nb_messages >= MAX_NB_SHOW:
                    nb_messages = 0
                    time.sleep(TIME_PAUSE_SEC)
            fp.close()

        # pickling after messaging
        with open(old_paper_set_path.format(workspace_name), "wb") as fp:
            pickle.dump(old_paper_set, fp)

    repo = git.Repo(base_dir)
    repo.git.add(os.path.join(base_dir, "summaries"))
    repo.git.commit("-m", f"\"Update summaries: {time.strftime('%Y-%m-%d')}\"")
    repo.git.push(force=True)


if __name__ == "__main__":
    main()
