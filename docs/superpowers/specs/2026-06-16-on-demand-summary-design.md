# On-demand 논문 요약 (@멘션 기반) — 설계

- 날짜: 2026-06-16
- 관련: [reaction-collection](2026-06-15-reaction-collection-design.md), [summary-4sections](2026-06-15-summary-4sections-design.md)

## 목적

배치(`main.py`)가 매일 필드별 신규 논문을 요약·게시하는 것과 별개로, 사용자가 **실시간으로 보고 싶은 특정 논문을 즉석에서 요약**받을 수 있게 한다.
Slack 채널/스레드에서 봇을 **@멘션 + arxiv 링크**로 호출하면, 봇이 그 자리(스레드)에 배치와 동일한 형식의 요약을 답글로 단다.

핵심 원칙: **on-demand 요약은 "트리거만 다른 배치 1건"이다.** 모델·프롬프트·요약 구조·요약 입력(abstract + 본문)·캐시·리액션 수집까지 전부 배치와 동일하게 동작한다.

## 비목표 (YAGNI / 추후 별도 스펙)

- 스레드 후속 Q&A (예전 `chatbot.py`/`bot.py`가 노리던 대화형) — 별도 기능
- 한 멘션에 여러 링크 동시 요약 — v1은 **첫 arxiv 링크 1건**만 처리
- on-demand 요약의 `summaries/` git 저장 — 하지 않음 (일별 필드 다이제스트와 성격이 다름)
- 슬래시 커맨드 / HTTP Events API — Socket Mode만 사용

## 사용자 결정 사항 (확정)

- **트리거**: Slack **@멘션** (슬래시 커맨드 아님). 멘션한 메시지에 스레드 답글, 이미 스레드면 같은 스레드에 답.
- **런타임**: Socket Mode 상시 리스너. Windows에서 **부팅 시 자동 실행**(Task Scheduler) + 죽으면 자동 재시작.
- **파이프라인 통합**: ① 요약 캐시 재사용 ② 리액션 수집 대상에 포함. `summaries/` git 저장은 안 함.
- **URL 처리 / 요약 방식**: 모두 **배치와 동일**하게 (아래 상세).

## 아키텍처

### 런타임 — `listener.py` (신규 top-level 진입점, `main.py`와 형제)

- `slack_bolt`의 **sync `App` + `SocketModeHandler`** 사용. 요약·OpenAI 호출이 동기이므로 async 불필요.
- 공개 URL 불필요(아웃바운드 소켓 연결만). 이미 `requirements.txt`에 `slack-bolt` 존재.
- `WORKSPACE_CONFIGS`의 각 `service_type == "slack"` 워크스페이스에 대해 App을 구성. 현재 활성 워크스페이스는 `seungtaek-lab` 1개.
- **이중 복원력**: ① Socket Mode 클라이언트 자체 재연결 ② 프로세스 크래시 시 Task Scheduler 재시작.

### 부팅 자동 실행 — `scripts/install_listener_task.ps1` (신규)

- Windows **Task Scheduler**에 `schtasks`로 작업 등록:
  - 트리거: **At startup(시스템 시작 시)**
  - 동작: 프로덕션 Python 3.11로 `listener.py` 실행 (배치 `main.py` 등록과 동일한 Python/경로)
  - 설정: **실패 시 재시작**, **무한정 실행 허용**, 이미 실행 중이면 중복 실행 안 함
- `main.py` 배치 등록과 같은 메커니즘이되 트리거만 상시형.

### Slack 앱 설정 (사용자가 api.slack.com에서 1회 수행, 문서화)

- **Socket Mode 활성화** → app-level token(`xapp-`, scope `connections:write`) 생성
- **Event Subscriptions** → bot event **`app_mention`** 구독
- **OAuth scopes**: `app_mentions:read`, `chat:write`(게시, 기존 보유), `reactions:read`(harvest, 기존 보유). 앱 재설치.
- env에 **`SLACK_APP_TOKEN_SEUNGTAEK_LAB`**(xapp-) 추가. `WORKSPACE_CONFIGS`의 slack 항목에 `"app_token": os.getenv("SLACK_APP_TOKEN_<WS>")` 키 추가.

## 컴포넌트 (작은 단위로 분리)

### 1. `api/arxiv.py`에 추가

**`parse_arxiv_ref(text) -> str | None`** — 임의 텍스트에서 arxiv 논문을 식별해 **배치와 동일한 정규형 URL**로 변환.

- 인식 대상: `/abs/<id>`, `/pdf/<id>`, `/html/<id>`, 끝의 `.pdf`, 버전 `v\d+`, bare id(신형 `NNNN.NNNNN`, 구형 `cs/NNNNNNN` 등 `<archive>/NNNNNNN`)
- **항상 `https://arxiv.org/abs/<id>` (버전 제거) 단일 정규형 반환**
- 매칭 실패 시 `None`
- **이유**: 배치는 목록 페이지의 "Abstract" 링크(`https://arxiv.org/abs/<id>`, 버전 없음)를 `paper_url`로 쓰고, `paper_info = "{title} ({paper_url})"`를 캐시·리액션 store의 키로 사용한다. on-demand가 다른 URL 형태를 쓰면 동일 논문이 캐시·store에서 분리되어 ① 재요약 발생 ② 리액션 신호 분산이 일어난다. 정규형을 강제해 키를 일치시킨다.

**`get_paper_title(paper_url) -> str`** — abs 페이지 `<h1 class="title">`에서 제목 추출(`Title:` 접두 제거). 배치는 목록 페이지에서 title을 얻지만 on-demand는 단건이라 abs 페이지에서 가져온다. `get_paper_abstract`와 동일한 요청 헤더/타임아웃/재시도 패턴 사용.

### 2. `api/service.py`에 추가

**`Service.summarize_one(paper_info, paper_abstract, paper_full_content) -> str`**

- `cache.has_paper_summarization(paper_info)`면 캐시값 그대로 반환(재요약 안 함)
- 아니면 `prepare_summarization_input(abstract, full_content)`(기존 메서드 재사용) → `agent.summarize()` → 결과가 빈 문자열이 아니면 `cache.update_paper_summarizations`로 저장 후 반환
- 빈 문자열(= `llm_retry` 소진)이면 **캐시에 저장하지 않고** 빈 문자열 반환 (배치 규칙과 동일: 오염 방지)

### 3. `listener.py` — `app_mention` 핸들러

처리 흐름:

1. `event.text`에서 `parse_arxiv_ref` → URL 없으면 안내 답글("arxiv 링크를 함께 멘션해 주세요") 후 종료
2. `get_paper_title` + `get_paper_abstract` + best-effort 본문(`get_html_experimental_link` → `get_paper_full_content`, 실패 시 abstract만). 캐시에 abstract/full_content 없으면 채우고, 있으면 재사용
3. `paper_info = get_paper_info(paper_url, paper_title)`
4. `service.summarize_one(...)` → 빈 문자열이면 에러 안내 답글 후 종료
5. `workspace.prepare_content(paper_info, "", summarization)`로 Slack 마크다운(4섹션) 생성
6. **`thread_ts = event.get("thread_ts") or event["ts"]`** 로 스레드 답글 게시 → 멘션 메시지에 스레드, 이미 스레드면 같은 스레드
7. `reactions.add_posted(store, ts=<답글 ts>, thread_ts=thread_ts, channel_id=<event channel>, workspace=<ws>, paper_info=paper_info, paper_url=paper_url, field="on-demand", posted_at=now)` → `save_store`. 이후 `main.py`의 `harvest_reactions`가 동일 `papers.json`을 스캔하므로 자동 수집됨

## 데이터 흐름

```
@arxivbot https://arxiv.org/pdf/2501.12345v2   (채널 또는 스레드 내)
  → app_mention 이벤트
  → parse_arxiv_ref → https://arxiv.org/abs/2501.12345   (정규형, 배치와 동일)
  → get_paper_title + get_paper_abstract (+ best-effort full content)
  → paper_info = "{title} (https://arxiv.org/abs/2501.12345)"
  → summarize_one (cache 우선; 배치가 이미 요약했으면 즉시 히트)
  → prepare_content → Slack 답글 (thread_ts = event.thread_ts or event.ts)
  → add_posted(papers.json)  → 이후 main.py harvest가 리액션 수집
```

## 에러 처리

- URL 없음 → 스레드 안내 답글, 종료
- 조회 실패(네트워크/파싱) → 에러 안내 답글, 캐시·store 오염 없음
- 요약 빈 문자열(`llm_retry` 소진) → 에러 안내 답글, 캐시·store 미기록
- 핸들러 최상위 try/except로 모든 예외 포착 → 사용자에게 한 줄 통지 + `api/logger`에 로그. 리스너 프로세스는 계속 유지

## 테스트

- **`tests/test_on_demand.py`** (단위, `api.agent`의 google import 미의존하도록 격리):
  - `parse_arxiv_ref`: `/abs/`, `/pdf/`, `/html/`, `.pdf` 접미, `v2` 버전, bare 신형 id, 구형 `cs/...` id, 비-arxiv 텍스트(None) — 모두 정규형 `https://arxiv.org/abs/<id>` 반환 검증
  - `thread_ts` 결정 로직: `thread_ts` 있을 때 / 없을 때(= `ts`)
  - `summarize_one`: 캐시 히트(agent 호출 안 함) / 미스(agent mock 호출 후 저장) / 빈 결과(저장 안 함)
- **`tests/smoke_listener.py`** (수동): 실제 채널에 멘션 1건 시나리오 검증 (기존 `smoke_summary.py` 스타일)

## 영향받는 파일 요약

| 파일 | 변경 |
|---|---|
| `api/arxiv.py` | `parse_arxiv_ref`, `get_paper_title` 추가 |
| `api/service.py` | `Service.summarize_one` 추가 |
| `settings.py` | slack 워크스페이스 config에 `app_token` 키 추가 |
| `.env` | `SLACK_APP_TOKEN_SEUNGTAEK_LAB` 추가 (사용자) |
| `listener.py` | 신규 — Socket Mode 리스너 + `app_mention` 핸들러 |
| `scripts/install_listener_task.ps1` | 신규 — Task Scheduler 등록 |
| `tests/test_on_demand.py`, `tests/smoke_listener.py` | 신규 |
