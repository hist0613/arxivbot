import os
import time
import pickle
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from slack_sdk import WebClient

from settings import *
from gpt3 import get_openai_summarization, get_openai_question_generation, get_openai_trend_analysis

base_dir = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
old_paper_set_path = os.path.join(base_dir, "old_paper_set_{}.pickle")
paper_abstracts_path = os.path.join(base_dir, "paper_abstracts.pickle")
paper_summarizations_path = os.path.join(base_dir, "paper_summarizations.pickle")
paper_questions_path = os.path.join(base_dir, "paper_questions.pickle")

def get_old_paper_set(workspace):
    if os.path.exists(old_paper_set_path.format(workspace)):
        with open(old_paper_set_path.format(workspace), 'rb') as fp:
            old_paper_set = pickle.load(fp)
    else:
        old_paper_set = set()
    return old_paper_set


def get_paper_abstracts():
    if os.path.exists(paper_abstracts_path):
        with open(paper_abstracts_path, 'rb') as fp:
            paper_abstracts = pickle.load(fp)
    else:
        paper_abstracts = defaultdict(str)
    return paper_abstracts


def get_paper_summarizations():
    if os.path.exists(paper_summarizations_path):
        with open(paper_summarizations_path, 'rb') as fp:
            paper_summarizations = pickle.load(fp)
    else:
        paper_summarizations = defaultdict(str)
    return paper_summarizations


def get_paper_questions():
    if os.path.exists(paper_questions_path):
        with open(paper_questions_path, 'rb') as fp:
            paper_questions = pickle.load(fp)
    else:
        paper_questions = defaultdict(str)
    return paper_questions


def get_paper_set_of(field):
    paper_set = []

    # list_url = "https://arxiv.org/list/{}/recent".format(field)
    list_url = "https://arxiv.org/list/{}/pastweek?skip=0&show={}".format(field, 500)
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
    dt_tags = list_soup.find_all('dt')
    dd_tags = list_soup.find_all('dd')

    papers_url = []
    papers_title = []
    papers_comment = []
    for dt_tag, dd_tag in zip(dt_tags, dd_tags):
        # <dd> 태그 내에서 <span class="list-identifier"> 태그 추출
        identifier_tag = dt_tag.find('span', class_="list-identifier")
        paper_url = identifier_tag.find('a', {'title': 'Abstract'})['href']
        paper_url = "https://arxiv.org" + paper_url
        papers_url.append(paper_url)

        # <dd> 태그 내에서 <div class="list-title"> 태그 추출
        title_tag = dd_tag.find('div', class_="list-title")
        paper_title = title_tag.text.strip().strip("Title:").strip()
        papers_title.append(paper_title)

        # <dd> 태그 내에서 <div class="list-comments"> 태그 추출
        comment_tag = dd_tag.find('div', class_="list-comments")
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

    paper_abstract = paper_soup.find_all('blockquote', class_="abstract")[0].text.strip().replace("Abstract:  ", "").replace("\n", " ")
    return paper_abstract 


def has_new_papers(new_papers, old_paper_set):
    for paper_url, paper_title, _ in new_papers:
        paper_info = get_paper_info(paper_url, paper_title)
        if paper_info not in old_paper_set:
            return True
    return False


def main():
    for workspace in WORKSPACES:
        workspace_name = f"{workspace['workspace']}-{workspace['allowed_channel']}"

        # collect new arXiv papers 
        print("Collecting new arXiv papers...")
        new_papers = defaultdict(list)
        for field in workspace['fields']:
            paper_set = get_paper_set_of(field)
            new_papers[field] = paper_set

        print("Connecting", workspace["workspace"], "...")

        sc = WebClient(workspace['slack_token'])

        old_paper_set = get_old_paper_set(workspace_name)

        # abstract crawling
        print("Crawling the abstracts of papers ...")
        paper_abstracts = get_paper_abstracts()
        for field in workspace['fields']:
            print("  - Processing {} field...".format(field))
            for paper_url, paper_title, _ in tqdm(new_papers[field]):
                # remove duplicates
                paper_info = get_paper_info(paper_url, paper_title)
                if paper_info in paper_abstracts:
                    continue

                paper_abstract = get_paper_abstract(paper_url)
                paper_abstracts[paper_info] = paper_abstract

                # pickling after summarization
                with open(paper_abstracts_path, 'wb') as fp:
                    pickle.dump(paper_abstracts, fp)

        if SHOW_SUMMARIZATION:
            print("Summarizing the abstract of papers by GPT3.5...")
            paper_summarizations = get_paper_summarizations()

            for field in workspace['fields']:
                print("  - Processing {} field...".format(field))

                all_papers = list(new_papers[field])
                nb_threads = 10
                for i in tqdm(range(0, len(all_papers), nb_threads)):
                    subset_papers = all_papers[i:i+nb_threads]
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = {}

                        for paper_url, paper_title, _ in subset_papers:
                            # remove duplicates
                            paper_info = get_paper_info(paper_url, paper_title)
                            if paper_info in paper_summarizations and paper_summarizations[paper_info] != "":
                                continue

                            paper_abstract = paper_abstracts[paper_info]
                            futures[executor.submit(get_openai_summarization, paper_abstract)] = paper_info

                        for f in concurrent.futures.as_completed(futures):
                            paper_summarizations[futures[f]] = f.result()

                    # pickling after summarization
                    with open(paper_summarizations_path, 'wb') as fp:
                        pickle.dump(paper_summarizations, fp)

        # if SHOW_QUESTION:
        #     print("Generating multiple choice questions of papers by GPT3.5...")
        #     paper_questions = get_paper_questions()
        #     for field in workspace['fields']:
        #         print("  - Processing {} field...".format(field))
        #         for paper_url, paper_title in tqdm(new_papers[field]):
        #             # remove duplicates
        #             paper_info = get_paper_info(paper_url, paper_title)
        #             if paper_info in paper_questions:
        #                 continue

        #             paper_abstract = paper_abstracts[paper_info]
        #             paper_question = get_openai_question_generation(paper_abstract)
        #             paper_questions[paper_info] = paper_question

        #             # pickling after question generation
        #             with open(paper_questions_path, 'wb') as fp:
        #                 pickle.dump(paper_questions, fp)

        nb_total_messages = 0
        nb_messages = 0
        for field in tqdm(workspace['fields']):
            if not has_new_papers(new_papers[field], old_paper_set):
                continue
            
            # make a parent message first
            sc.chat_postMessage(
                channel=workspace['allowed_channel'],
                text="New uploads on arXiv({})\n".format(field)
            )

            # get the timestamp of the parent messagew
            result = sc.conversations_history(channel=workspace["allowed_channel_id"])
            conversation_history = result["messages"]  # [0] is the most recent message
            message_ts = conversation_history[0]['ts']

            # make a thread by replying to the parent message
            for paper_url, paper_title, paper_comment in new_papers[field]:
                paper_info = get_paper_info(paper_url, paper_title)
                
                # remove duplicates
                if paper_info in old_paper_set:
                    continue
                
                content = paper_info
                if paper_comment != "":
                    content += f"\n{paper_comment}"
                if SHOW_SUMMARIZATION:
                    content += f"\n{paper_summarizations[paper_info]}"
                # if SHOW_QUESTION:
                #     content += f"\n\n{paper_questions[paper_info]}"

                old_paper_set.add(paper_info)

                sc.chat_postMessage(
                    channel=workspace['allowed_channel'],
                    text=content,
                    thread_ts=message_ts
                )

                nb_total_messages += 1
                nb_messages += 1
                if nb_messages >= MAX_NB_SHOW:
                    nb_messages = 0
                    time.sleep(TIME_PAUSE_SEC)

        # pickling after messaging
        with open(old_paper_set_path.format(workspace_name), 'wb') as fp:
            pickle.dump(old_paper_set, fp)

        # nb_total_messages = 339
        # if nb_total_messages > 0:
        #     trend_analyses = get_openai_trend_analysis(paper_summarizations, nb_total_messages=nb_total_messages)

        #     # make a parent message first
        #     sc.chat_postMessage(
        #         channel=workspace['allowed_channel'],
        #         text="Trend analysis on Today's arXiv\n"
        #     )

        #     # get the timestamp of the parent message
        #     result = sc.conversations_history(channel=workspace["allowed_channel_id"])
        #     conversation_history = result["messages"]  # [0] is the most recent message
        #     message_ts = conversation_history[0]['ts']
        
        #     for trend_analysis in trend_analyses:
        #         sc.chat_postMessage(
        #             channel=workspace['allowed_channel'],
        #             text=trend_analysis,
        #             thread_ts=message_ts
        #         )

if __name__ == "__main__":
    main()
