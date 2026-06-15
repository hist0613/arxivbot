# Slack 리액션 수집 파이프라인 (Feature 1)

**날짜**: 2026-06-15
**범위**: 앞으로 게시되는 논문에 달리는 **사람 리액션**을 논문에 매핑해 수집·저장. 수확은 `main.py` 단일 실행에 포함.
**비범위**: 이모지 polarity 매핑, reward model 학습/데이터셋 정형화(후속 스펙), 과거 리액션 backfill, 봇의 리액션 게시(pre-seed), 리액션 시간 추이 이력 — 하지 않음.

---

## 1. 동기 / 원칙

리액션은 **사람의 preference 신호**로 reward model(이 랩이 어떤 논문에 반응하는가) 학습에 쓰려는 것이다.
- 봇은 리액션을 **달지 않는다**(신호 오염 방지). 게시·수집만.
- **forward 수집**: 파이프라인이 생긴 뒤 쌓이는 리액션을 모은다(과거 backfill 안 함).
- 어떤 이모지가 긍정/부정인지는 지금 고정하지 않고 **원시 이모지를 그대로 저장**해 후속 단계에서 해석.

## 2. 아키텍처 / 데이터 흐름

`main.py` **단일 실행**에 수확을 포함한다(별도 스케줄드 잡 없음):

```
크롤 → 요약 → 게시(_send_slack_messages)
                 └─ 논문 답글 ts를 papers store에 엔트리로 기록
            ↓
   harvest_reactions(client, store)   ← main.py 말미, 게시 후
     · 최근 HARVEST_WINDOW_DAYS 이내 엔트리 선택
     · thread_ts로 묶어 thread당 conversations_replies 1회 (bulk)
     · 각 답글의 reactions를 해당 엔트리에 in-place 갱신
            ↓
   git commit/push (기존)
```

**API 효율**: 논문마다 `reactions_get`(수천 콜) 대신 **field 제목 스레드마다 `conversations_replies` 1회**로 그 스레드의 모든 논문 답글 + 리액션을 한 번에 받는다. 14일 윈도우 ≈ 6필드 × 14일 ≈ 80여 스레드 → 실행당 약 100 콜(페이지네이션 포함)로 충분.

**중복 없음**: 저장은 **논문 message ts를 키로 한 dict**(`papers.json`). 게시 때 엔트리 생성, 수확 때 그 엔트리의 reactions를 **덮어씀** → 논문당 항상 1 레코드.

## 3. 컴포넌트

### 3.1 `api/reactions.py` (신규)
- `load_store() -> dict` / `save_store(store: dict)` — `reactions/papers.json`(ts→엔트리) 읽기/쓰기.
- `add_posted(store, *, ts, thread_ts, channel_id, workspace, paper_info, paper_url, field, posted_at)` — 게시 엔트리 추가(이미 있으면 유지). `reactions=None, last_harvested=None`.
- `hash_user(user_id: str) -> str` — `sha256((salt + user_id).encode()).hexdigest()[:16]`. salt = `os.getenv("REACTION_HASH_SALT")`.
- `harvest_reactions(client, store, *, window_days, bot_user_id, now)` — 윈도우 내 엔트리를 thread별로 묶어 `conversations_replies` 호출, 각 답글 ts 엔트리의 `reactions`/`last_harvested` 갱신. 반환: 갱신된 논문 수.

엔트리 스키마(`papers.json`의 값):
```json
{
  "thread_ts": "1718...", "channel_id": "C...", "workspace": "seungtaek-lab",
  "paper_info": "...", "paper_url": "https://arxiv.org/abs/...", "field": "cs.CL",
  "posted_at": "2026-06-15T00:00:00Z", "last_harvested": "2026-06-16T00:00:00Z",
  "reactions": [{"emoji": "thumbsup", "count": 2, "hashed_users": ["ab12..","cd34.."]}]
}
```

### 3.2 게시 시 엔트리 기록 (`api/workspace.py`)
`_send_slack_messages`에서 thread 제목 게시로 얻은 `thread_ts`와 각 논문 답글 게시 결과 `ts`를 store에 기록:
- 함수 시작에 `store = load_store()`, 각 논문 답글 후 `add_posted(store, ts=result["ts"], thread_ts=thread_ts, ...)`, 루프 끝에 `save_store(store)`.
- slack 워크스페이스에만 적용(discord 경로는 무관).

### 3.3 수확 호출 (`main.py`)
게시 루프 뒤, git 커밋 전에:
```python
from api.reactions import load_store, save_store, harvest_reactions
for ws in workspaces:
    if ws.service_type != "slack":
        continue
    store = load_store()
    client = WebClient(ws.slack_token)
    bot_id = client.auth_test()["user_id"]
    harvest_reactions(client, store, window_days=HARVEST_WINDOW_DAYS,
                      bot_user_id=bot_id, now=utcnow())
    save_store(store)
```
- 수확 로직: 윈도우(`posted_at`이 now−window_days 이내) 엔트리를 `(channel_id, thread_ts)`로 그룹화 → 그룹마다 `conversations_replies(channel, ts=thread_ts)`(페이지네이션) → 반환 메시지의 `ts`가 store 엔트리면 그 `reactions` 갱신.
- 각 reaction: `users`에서 `bot_user_id` 제외 후 `hash_user`로 변환, `count`는 봇 제외 후 길이로 재계산.
- 429/일시 오류는 `Retry-After` 백오프 재시도.

### 3.4 `settings.py` / `.env`
- `settings.py`: `REACTIONS_DIR`, `PAPERS_STORE_PATH`(=`reactions/papers.json`), `HARVEST_WINDOW_DAYS = 14`.
- `.env`(gitignore됨): `REACTION_HASH_SALT=<랜덤 문자열>`.

### 3.5 `.gitignore`
`reactions/` 추가 — 해시 user 데이터라 공개 repo에 올리지 않음(로컬 + Dropbox 백업). summaries와 달리 비공개.

## 4. 프라이버시

- user id는 salt 해시로만 저장(평문 X). salt는 `.env`(비공개).
- `reactions/` 전체 gitignore → public repo 노출 안 됨.
- 해시는 distinct 카운트/중복제거 용도. salt 미공개라 외부 역추적 불가.

## 5. 엣지 케이스

- **인덱스 없는 과거 메시지**: feature 이전 게시분은 store에 없어 수확 대상 아님(forward-only, 의도).
- **봇 self-reaction**: pre-seed 안 하므로 없지만 방어적으로 봇 user id 제외.
- **삭제/취소된 리액션**: in-place 갱신이라 최신 상태로 자연 반영.
- **윈도우 밖**: 14일 지난 엔트리는 더 갱신 안 함 → 마지막 수확값이 최종 정착 상태로 고정.
- **수확 실패(특정 스레드)**: 해당 그룹만 건너뛰고 로그, 나머지 진행(다음 실행에 재시도).

## 6. 테스트

- **단위(`unittest`, API 불필요)**:
  - `hash_user` 결정성 + salt 반영(같은 입력·salt→동일, salt 다르면 상이).
  - `add_posted` → `load/save_store` 라운드트립, 동일 ts 재기록 시 1건 유지.
  - 윈도우 필터 경계(13일 포함, 15일 제외).
  - `harvest_reactions`를 **가짜 client**(conversations_replies가 고정 응답)로 호출 → 봇 user 제외·해시 변환·count 재계산·해당 엔트리만 갱신 검증.
- **수동 smoke**: 기존 Influcoder 메시지(👀 1개)의 ts/thread_ts로 임시 store 엔트리를 만들어 `harvest_reactions`를 실제 client로 실행 → 엔트리에 `eyes:1`이 잡히는지 확인. 검증 후 임시 엔트리 제거.

## 7. Out of scope (YAGNI)

- 이모지 polarity 매핑, RM 데이터셋 정형(pointwise/pairwise), 노출 편향 정규화 — 후속 스펙.
- 리액션 시간 추이 이력(스냅샷 누적), Socket Mode 실시간, backfill, 봇 pre-seed.
