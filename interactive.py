from flask import Flask, request, jsonify
from slack_sdk import WebClient
import threading

from main import get_paper_abstract
from gpt3 import get_openai_summarization
from settings import WORKSPACES

app = Flask(__name__)
# Flask 를 로컬에서 구동 시 슬랙에서 로컬 PC에 접근할 수 없기 떄문에
# ngrok 을 사용하여 로컬 서버를 외부에 공개해야 함
# ```ngrok http --domain={slack app에 등록된 domain} 5000``` 명령어를 통해 5000번 포트를 공개

def get_slack_token(workspace_name):
    for workspace in WORKSPACES:
        if workspace['workspace'] == workspace_name:
            return workspace['slack_token']
    return None


def process_command(data):
    # operation_timeout 에러 때문에 비동기 방식으로 처리하기 위한 함수

    paper_url = data.get('text')  # 사용자가 입력한 논문 URL
    workspace_name = data.get('team_domain')  # Slack 워크스페이스 이름

    # 해당 워크스페이스의 토큰 가져오기
    slack_token = get_slack_token(workspace_name)
    if not slack_token:
        return jsonify(text="워크스페이스 토큰을 찾을 수 없습니다.")

    # 논문 정보 가져오기 및 요약 (get_paper_abstract 및 get_openai_summarization 함수 구현 필요)
    paper_abstract = get_paper_abstract(paper_url)
    paper_summary = get_openai_summarization(paper_abstract)

    # Slack 메시지 전송
    response_text = f"URL: {data.get('text')}\n\n{paper_summary}"
    slack_client = WebClient(token=slack_token)
    slack_client.chat_postMessage(channel=data.get('channel_id'), text=response_text)


@app.route('/arxivbot', methods=['POST'])
def arxivbot():
    data = request.form
    threading.Thread(target=process_command, args=(data,)).start()
    return '', 200  # 빈 응답과 200 OK 상태 코드 반환
    # return jsonify(response_type='in_channel', text="논문 요약을 처리 중입니다. 잠시 후 결과를 전송해 드리겠습니다.")

if __name__ == "__main__":
    app.run()
