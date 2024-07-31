import time
import threading

from flask import Flask, request, jsonify
from slack_sdk import WebClient
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import requests

from gpt3 import get_openai_summarization
from settings import WORKSPACES, MAX_NB_GPT3_ATTEMPT

app = Flask(__name__)
# Flask 를 로컬에서 구동 시 슬랙에서 로컬 PC에 접근할 수 없기 떄문에
# ngrok 을 사용하여 로컬 서버를 외부에 공개해야 함
# ```ngrok http --domain={slack app에 등록된 domain} 5000``` 명령어를 통해 5000번 포트를 공개
# https://api.slack.com/apps/A062W14425U/slash-commands? 여기 혹은
# https://dashboard.ngrok.com/get-started/setup/windows (Static Domain) 에서 찾을 수 있음


def get_slack_token(workspace_name):
    for workspace in WORKSPACES:
        if workspace["workspace"] == workspace_name:
            return workspace["slack_token"]
    return None


def get_paper_page(paper_url):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            paper_page = requests.get(paper_url)
            if paper_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            print(e)
            time.sleep(trial * 30 + 15)
    return paper_page


def extract_info_arxiv(paper_soup):
    # Extract the title
    paper_title = (
        paper_soup.find("h1", class_="title").text.replace("Title:", "").strip()
    )

    # Extract the comments (if available)
    comment_section = paper_soup.find("td", class_="comments")
    paper_comment = comment_section.text.strip() if comment_section else None

    # Extract the abstract
    abstract_section = paper_soup.find("blockquote", class_="abstract")
    paper_abstract = (
        abstract_section.text.strip().replace("Abstract:", "").replace("\n", " ")
        if abstract_section
        else None
    )

    return paper_title, paper_comment, paper_abstract


def extract_info_aclanthology(paper_soup):
    # 논문 제목 추출
    title_section = paper_soup.find("h2", id="title")
    paper_title = title_section.text.strip() if title_section else None

    # Anthology ID 추출
    anthology_id_section = paper_soup.find("dt", string="Anthology ID:")
    paper_comment = (
        anthology_id_section.find_next_sibling("dd").text.strip()
        if anthology_id_section
        else None
    )

    # 초록 추출
    abstract_section = paper_soup.find("div", class_="acl-abstract")
    paper_abstract = (
        abstract_section.span.text.strip()
        if abstract_section and abstract_section.span
        else None
    )

    return paper_title, paper_comment, paper_abstract


def get_paper_info_without_title(paper_url):
    paper_page = get_paper_page(paper_url)
    paper_soup = BeautifulSoup(paper_page.text, "html.parser")

    # URL에서 도메인 추출
    domain = urlparse(paper_url).netloc

    if "arxiv.org" in domain:
        return extract_info_arxiv(paper_soup)
    elif "aclanthology.org" in domain:
        return extract_info_aclanthology(paper_soup)
    else:
        raise ValueError(f"Unsupported domain: {domain}")


def process_command(data):
    # operation_timeout 에러 때문에 비동기 방식으로 처리하기 위한 함수

    paper_url = data.get("text")  # 사용자가 입력한 논문 URL
    workspace_name = data.get("team_domain")  # Slack 워크스페이스 이름

    # 해당 워크스페이스의 토큰 가져오기
    slack_token = get_slack_token(workspace_name)
    if not slack_token:
        return jsonify(text="워크스페이스 토큰을 찾을 수 없습니다.")

    # 논문 정보 가져오기 및 요약
    paper_title, paper_comment, paper_abstract = get_paper_info_without_title(paper_url)
    paper_summary = get_openai_summarization(paper_abstract)

    # Slack 메시지 전송
    response_text = f"{paper_title} ({data.get('text')})\nComments: {paper_comment}\n{paper_summary}"
    slack_client = WebClient(token=slack_token)
    slack_client.chat_postMessage(channel=data.get("channel_id"), text=response_text)


@app.route("/arxivbot", methods=["POST"])
def arxivbot():
    data = request.form
    threading.Thread(target=process_command, args=(data,)).start()
    # return "", 200  # 빈 응답과 200 OK 상태 코드 반환
    return jsonify(response_type="in_channel")
    # return jsonify(response_type='in_channel', text="논문 요약을 처리 중입니다. 잠시 후 결과를 전송해 드리겠습니다.")


if __name__ == "__main__":
    # print(get_paper_info_without_title("https://arxiv.org/abs/2306.00354"))
    # print(get_paper_info_without_title("https://arxiv.org/abs/2310.10226"))
    # print(
    #     get_paper_info_without_title(
    #         "https://aclanthology.org/2023.findings-emnlp.447/"
    #     )
    # )
    app.run()
