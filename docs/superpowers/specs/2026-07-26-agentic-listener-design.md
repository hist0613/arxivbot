# 리스너 에이전트화 + 저장소 정리 — 설계

- 날짜: 2026-07-26
- 관련: [multi-source-summary](2026-06-23-multi-source-summary-design.md), [on-demand-summary](2026-06-16-on-demand-summary-design.md), [reaction-collection](2026-06-15-reaction-collection-design.md)

## 목적

세 가지를 함께 처리한다.

1. **summaries/ 폐지** — 아무도 안 보는 산출물인데 두 개의 버그(필드별 파일이 전부 동일 내용, 새 논문 0건 실행이 그날 파일을 빈 파일로 덮어씀)를 안고 795MB를 차지하고 있다.
2. **캐시를 SQLite로** — 지금은 프로세스마다 pickle 전체를 읽고 항목 하나 추가할 때마다 전체를 덮어쓴다. 배치와 리스너가 동시에 열려 있어 서로의 스냅샷을 덮어쓰며, 실제로 요약 캐시가 20.5만건(2026-06-23)에서 9,447건으로 줄었다. 2026-07-26 15:23 리스너 기동 로그에는 쓰기 중인 파일을 읽어 빈 캐시로 시작한 경고가 남아 있다.
3. **리스너를 에이전트로** — 지금은 논문 URL만 처리하고, 일반 웹페이지(예: `https://nvlabs.github.io/Sana/Video2/`)나 "attention residuals 쉽게 설명해" 같은 질문은 안내 문구만 돌려준다. 검색과 페이지 읽기를 도구로 붙여 답할 수 있게 한다.

## 사용자 결정 사항 (확정)

- summaries는 기능·파일·자동 커밋까지 전부 제거. **git 히스토리는 rewrite하지 않는다**(과거 내용은 커밋에 남는다).
- 캐시 저장소는 **SQLite**. 기존 pickle은 1회 마이그레이션.
- 에이전트 런타임은 **OpenAI Responses API + 내장 web_search**. 요약 모델과 에이전트 모델은 설정에서 분리.
- **논문 요약도 도구로 정의**한다. 단 도구가 게시까지 직접 수행하고 모델은 영수증만 받는다(아래 "부수효과 도구" 참조).
- **멘션은 계속 필수.** 스레드 안 후속 메시지를 멘션 없이 받는 건 하지 않는다.
- 일반 웹페이지 요약 형식은 **한 줄 요지 + bullet 5~8개**. 섹션 고정 없음.
- 첫 도구 집합은 `web_search`, `fetch_page`, `summarize_paper` + 문맥 도구 `read_thread`.
- 스레드 문맥은 **토큰 예산 안에서 점진적으로** 담고, 넘치는 부분은 요지로 접되 모델이 도구로 다시 펼 수 있게 한다.

## 비범위 (YAGNI)

- git 히스토리 rewrite (.git 190MB는 그대로 둔다)
- 멘션 없는 스레드 메시지 구독 (`message.channels`)
- 배치(`main.py`)의 에이전트화 — 배치는 지금의 결정론적 경로 그대로
- 리액션 store(`reactions/papers.json`)의 SQLite 이전 — 현행 유지
- 페이지 종류별(프로젝트 페이지/블로그/저장소/뉴스) 프롬프트 분기
- 과거 요약 검색 도구, arXiv 검색 도구 — 나중에 필요하면 추가

---

## Part A. summaries 제거

- `api/workspace.py`: `save_summaries` 삭제, `TODAY_SUMMARIES_DIR` import 삭제.
- `main.py`: `workspace.save_summaries(threads)` 호출과 말미의 git add/commit/push 블록 삭제. `import git`, `SUMMARIES_DIR` import도 함께 제거된다.
- `settings.py`: `SUMMARIES_DIR`, `TODAY_SUMMARIES_DIR` 삭제.
- `git rm -r summaries` (워킹트리 795MB 회수).
- `README.md`에서 summaries 언급 정리.

배치는 이제 저장소를 전혀 건드리지 않는다. 코드 변경 커밋은 사람이 한다.

## Part B. 저장소를 SQLite로

### 스키마 (`cache/arxivbot.db`, WAL 모드)

| 테이블 | 컬럼 |
|---|---|
| `abstracts` | `paper_info` PK, `text`, `fetched_at` |
| `full_contents` | `paper_info` PK, `text`, `fetched_at` |
| `summaries` | `paper_info` PK, `text`, `schema_version`, `model`, `created_at` |
| `pages` | `url` PK, `title`, `text`, `fetched_at` |
| `thread_digests` | `thread_ts` PK, `digest`, `covered_until_ts`, `updated_at` |

`full_contents.text`는 지금 dict(섹션별)라 JSON 문자열로 직렬화해 넣고 읽을 때 되돌린다.

### 인터페이스

`api/store.py`에 SQLite 접근을 두고, `CacheManager`는 겉모습(`paper_abstracts[key]`, `paper_full_contents[key]`, `has_paper_summarization`, `update_paper_*`)을 유지한 채 속만 교체한다. 따라서 `api/service.py`, `api/arxiv.py`, `api/workspace.py` 호출부는 고치지 않는다. 사전처럼 보이던 속성은 키 단위로 DB를 때리는 얇은 매핑 객체가 된다(없는 키는 지금처럼 빈 문자열).

- 쓰기는 전부 `INSERT ... ON CONFLICT DO UPDATE` — 통째 덮어쓰기가 사라진다.
- 연결은 프로세스마다 하나, `check_same_thread=False` + 짧은 트랜잭션. 배치는 요약 스레드 5개가 동시에 쓰므로 쓰기는 락으로 직렬화하고 `busy_timeout`을 넉넉히 둔다.
- 요약 재사용 판정은 지금처럼 `prompts.is_current_summary_schema`를 쓰되, `schema_version` 컬럼에도 기록해 다음 스키마 변경 때 SQL로 걸러낼 수 있게 한다.

### 마이그레이션

`scripts/migrate_cache_to_sqlite.py` 1회 실행: pickle 3개를 읽어 넣고 건수를 대조 출력한다. 요약은 현재 4섹션 스키마인 항목만 넣는다(옛 포맷은 어차피 재요약 대상). 원본 pickle은 `.bak`으로 이름만 바꿔 남긴다. 배치·리스너가 모두 멈춘 상태에서 돌린다.

## Part C. 에이전트 리스너

### 흐름

멘션 수신 -> 채널 확인 -> 문맥 조립(Part D) -> 에이전트 루프 -> 답변 게시.

라우팅 분기는 두지 않는다. 논문 링크든 웹페이지든 질문이든 모두 에이전트가 받고, 무엇을 할지는 도구 선택으로 표현된다.

### 도구

- **`summarize_paper(url)` — 부수효과 도구.** 안에서 `build_resolver` 사슬로 resolve -> PDF/HTML 텍스트 추출 -> `Service.summarize_text`로 4섹션 요약 -> 진행 표시 메시지 편집 -> `prepare_slack_blocks`로 스레드에 게시 -> `add_posted`로 리액션 store 등록까지 수행한다. 모델에는 `{"posted": true, "title": ..., "url": ...}` 수준의 영수증만 돌려준다. 요약 본문을 모델에 넘기지 않는 것이 요점이다. 모델이 문장을 다시 쓰면 매일 쓰는 요약 품질이 모델 판단에 걸리기 때문이다. 링크가 여러 개면 도구를 여러 번 부르며, 개별 실패는 그 링크만 실패로 돌려주고 나머지는 계속한다(현행 동작 유지).
- **`fetch_page(url)`** — 페이지 본문 텍스트를 추출해 모델에 돌려준다. `pages` 테이블에 캐시. 논문 호스트로 판정되면 `summarize_paper`를 쓰라는 안내를 함께 돌려준다.
- **`web_search`** — Responses API 내장 도구.
- **`read_thread(before_ts, limit)`** — 접힌 이전 메시지를 원문으로 가져온다(Part D).

### 답변 게시

에이전트의 최종 텍스트는 `prepare_slack_blocks`를 타서 글머리 기호 목록으로 나간다. 일반 웹페이지 요약은 한 줄 요지 + bullet 5~8개 형태를 프롬프트로 요구한다. `summarize_paper`가 이미 게시를 마친 경우, 최종 텍스트가 사실상 빈 내용이면 별도 메시지를 만들지 않는다.

### 진행 표시

지금처럼 메시지 1개를 편집한다. 도구별 문구: 생각하는 중 -> 검색하는 중 -> 페이지 읽는 중 -> 답변. `summarize_paper`는 자기 답글을 따로 만들어 기존 문구(가져오는 중 / PDF 다운로드 중 / AI 요약 중)를 그대로 쓴다.

### 가드와 폴백

- 도구 호출 8스텝, 전체 90초 상한. 초과하면 그때까지 내용으로 답하고 잘렸음을 알린다.
- 에이전트 호출이 실패하면 현행 결정론적 경로(`extract_targets` -> 링크별 요약)로 폴백한다. 링크가 없으면 오류 안내.
- 모델은 `settings.AGENT_MODEL`, 요약은 기존 `MODEL`.

## Part D. 문맥 조립

`api/context.py`가 Slack `conversations_replies` 결과를 받아 예산 안에서 메시지 목록을 만든다.

1. 최신 메시지부터 거꾸로 담는다. 예산은 기본 6000 토큰(`CONTEXT_TOKEN_BUDGET`), `Encoder`로 센다.
2. 봇이 올린 논문 요약 메시지는 처음부터 `[요약 게시: 제목 (URL)]` 한 줄로 접는다. 본문은 필요할 때 `read_thread`로 편다.
3. 예산을 넘겨 잘린 앞부분은 한 문단 요지로 접어 맨 앞에 둔다. 요지는 `thread_digests`에 `covered_until_ts`와 함께 캐시하고, 다음 멘션 때는 새로 밀려난 구간만 이어 붙여 갱신한다(전체 재요약 안 함).
4. 조립 결과는 (요지 한 문단) + (원문 메시지들) + (이번 멘션) 순으로 모델에 들어간다.

`read_thread(before_ts, limit)`는 접힌 구간의 원문을 돌려준다. 미리 다 밀어 넣는 대신 모델이 필요할 때 파게 하는 쪽이 예산을 아끼면서 정확도를 지킨다.

## 파일 구조

신규:

- `api/store.py` — SQLite 접근
- `api/context.py` — 예산 조립, 요지 캐시
- `api/tools.py` — 도구 정의와 실행(부수효과 도구 포함)
- `api/agent_loop.py` — Responses 루프, 스텝·시간 상한
- `scripts/migrate_cache_to_sqlite.py`

수정: `api/cache.py`(속 교체), `listener.py`(얇게), `settings.py`, `prompts.py`(웹페이지 요약·에이전트 시스템 프롬프트), `main.py`·`api/workspace.py`(Part A).

`listener.py`는 Slack 연결과 이벤트 배선만 남기고, 판단·조립·도구는 위 모듈로 내린다. 순수 함수 위주라 Slack 없이 테스트된다.

## 테스트

- `tests/test_store.py` — upsert, 없는 키, JSON 왕복, 두 연결 동시 쓰기
- `tests/test_context.py` — 예산 경계, 봇 요약 접기, 요지 이어붙이기, `read_thread` 범위
- `tests/test_tools.py` — `summarize_paper`가 게시·store 등록까지 하고 영수증만 돌려주는지, 링크 일부 실패 시 나머지 진행, `fetch_page` 캐시 히트
- `tests/test_agent_loop.py` — 스텝·시간 상한, 에이전트 실패 시 결정론 폴백
- `tests/smoke_agent.py` — 실 API 1회(논문 링크, 웹페이지 링크, 링크 없는 질문 각 1건)

기존 `test_on_demand.py`, `test_slack_blocks.py`, `test_resolvers.py`, `test_pdf.py`는 유지한다. 검증은 Windows PowerShell의 Python 3.11에서 돌린다.

## 구현 순서

1. Part A (summaries 제거) — 독립적이고 즉시 이득
2. Part B (SQLite + 마이그레이션) — 리스너·배치 정지 상태에서 전환, 배치 1회·멘션 1회로 확인
3. Part D (문맥 조립) — 순수 함수라 먼저 테스트 가능
4. Part C (도구 + 루프 + 리스너 배선) — 마지막에 붙이고 폴백으로 회귀 방지
5. 리스너 재시작(스케줄러 프로세스가 옛 코드를 잡고 있으므로 필수)

## 열린 항목

- `AGENT_MODEL` 기본값은 구현 착수 시 사용 가능한 모델을 확인해 정한다.
