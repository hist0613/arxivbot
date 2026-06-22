# arxivbot
하루에 한번씩 arXiv에서 원하는 field (cs.CL 등) 의 논문을 slack message로 보내주는 코드입니다.
openai chatgpt 를 활용해 논문 abstract의 3줄 요약을 기본으로 함께 제공하고 있습니다. 

## Usage
1. Install requirements
```sh
pip install -r requirements.txt
```

2. Define `WORKSPACES` in settings.py
```python
WORKSPACES = [
    {
        'workspace': '{workspace}',
        'allowed_channel': '{channel_name}',
        'allowed_channel_id': '{channel_id}',
        'slack_token': "{app_slack_token}",
        'fields': ["{arxiv_field}",]
    },
]
```

Example
```python
WORKSPACES = [
    {
        'workspace': 'ai-research-kr',
        'allowed_channel': 'nlp-research',
        'allowed_channel_id': 'X000XXXXX0X',
        'slack_token': "xxxx-1234567890-1234567890123-X0xx0x0xxXXxxxxXx0x0x0XX",
        'fields': ["cs.CL", "cs.IR"]
    },
]
```

3. Define `OPENAI_API_KEY` in settings.py
```python
OPENAI_API_KEY = "eeve-is-llm-of-the-year"
```

4. Make slack bot and check api permissions & event subscriptions

- https://api.slack.com/apps?new_app=1
- The bot need to subscribe 'app_mention' event and get 'app_mentions:read', 'channels:history', 'chat:write', 'groups:history', 'im:history', 'mpim:history' OAuth Scope.

5. Invite the bot to your slack channel

6. Launch and enjoy
```sh
python main.py
```

---

## On-demand 요약 (@멘션 리스너)

`main.py`(일별 배치)와 별개로, Slack 채널에서 봇을 **`@arxivbot <arxiv-url>`** 로 멘션하면 그 스레드에 즉석 요약을 달아주는 상시 리스너(`listener.py`)입니다. 모델·프롬프트·요약 구조·캐시는 배치와 동일하며, 멘션을 받을 채널은 `settings.py`의 `listener_channel_id`로 지정합니다(배치 게시 채널 `allowed_channel_id`와 분리).

### 1. Slack 앱 설정 (api.slack.com/apps → 해당 앱, 1회)

- **Socket Mode** 활성화 → app-level 토큰(`xapp-`, scope `connections:write`) 생성
- **Event Subscriptions** → Subscribe to bot events에 **`app_mention`** 추가
- **OAuth & Permissions** Bot Token Scopes에 **`app_mentions:read`** 추가 (`chat:write`는 기존 보유)
- 앱을 **Reinstall to Workspace** (위 변경 적용)
- 리스너를 쓸 채널에 봇 초대: `/invite @arxivbot`

### 2. 설정 값

- `.env`에 app-level 토큰 추가:
  ```
  SLACK_APP_TOKEN_SEUNGTAEK_LAB=xapp-...
  ```
  (기존 봇 토큰 `SLACK_TOKEN_SEUNGTAEK_LAB`(xoxb-)은 그대로 두고, 별도로 추가)
- `settings.py`의 해당 워크스페이스에 `"listener_channel_id": "<채널 ID>"` 지정
  (채널 ID는 Slack 채널 우클릭 → 채널 세부정보 맨 아래, 또는 멘션 시 로그의 `channel=...` 값)

### 3. 수동 실행 (먼저 이걸로 검증)

```powershell
cd C:\Users\hist0\Dropbox\develop\arxivbot_new
python listener.py
```
`Starting to receive messages` 가 뜨면 대기 상태. 지정 채널에서 `@arxivbot https://arxiv.org/abs/1706.03762` 멘션 → 스레드에 요약이 달리면 정상. Ctrl+C로 종료.

### 4. 부팅 시 자동 실행 등록 (Task Scheduler)

**관리자 권한** PowerShell에서:
```powershell
cd C:\Users\hist0\Dropbox\develop\arxivbot_new
powershell -ExecutionPolicy Bypass -File scripts\install_listener_task.ps1
Start-ScheduledTask -TaskName arxivbot-listener
```
- 로그온 시 자동 시작 + 죽으면 1분 뒤 자동 재시작
- 작업은 `python`을 직접 띄우지 않고, **`scripts\run_listener.ps1` 래퍼**를 통해 실행합니다. 래퍼가 ① 등록 시 확보한 python.exe 절대경로로 실행해 Store판 python 별칭 문제를 우회하고 ② 모든 출력을 로그 파일에 남깁니다.
- 등록 후 **수동으로 띄워둔 `python listener.py` 창은 닫으세요**(중복 실행 방지).

### 5. 로그 확인 (스케줄러로 돌 때)

스케줄러 실행은 콘솔이 없으므로 로그는 파일로 남습니다: **`logs\listener.log`**
```powershell
Get-Content -Wait -Tail 50 .\logs\listener.log   # 실시간 follow
```

### 6. 관리 / 문제 해결

```powershell
Get-ScheduledTask     -TaskName arxivbot-listener   # State가 Running이어야 정상(Ready=꺼짐)
Get-ScheduledTaskInfo -TaskName arxivbot-listener   # LastTaskResult=종료코드
Start-ScheduledTask   -TaskName arxivbot-listener
Stop-ScheduledTask    -TaskName arxivbot-listener
Disable-ScheduledTask -TaskName arxivbot-listener   # 자동 재시작 멈추기
Unregister-ScheduledTask -TaskName arxivbot-listener -Confirm:$false  # 등록 해제
```
- 멘션해도 무반응이면 `logs\listener.log`에 `app_mention ignored: channel ... != listener channel ...` 가 있는지 확인 → 있으면 `settings.py`의 `listener_channel_id`가 실제 채널과 다른 것.