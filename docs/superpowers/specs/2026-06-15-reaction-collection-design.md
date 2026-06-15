# Slack 리액션 수집 파이프라인 (Feature 1)

**날짜**: 2026-06-15
**범위**: 앞으로 게시되는 논문에 달리는 **사람 리액션**을 논문에 매핑해 시계열로 수집·저장.
**비범위**: 이모지 polarity 매핑, reward model 학습/데이터셋 정형화(후속 스펙), 과거 리액션 backfill, 봇의 리액션 게시(pre-seed) — 하지 않음.

---

## 1. 동기 / 원칙

리액션은 **사람의 preference 신호**로 reward model(이 랩이 어떤 논문에 반응하는가) 학습에 쓰려는 것이다. 따라서:
- 봇은 리액션을 **달지 않는다**(신호 오염 방지). 게시·수집만 한다.
- 목적은 **forward 수집**: 파이프라인이 생긴 뒤 쌓이는 리액션을 모은다(현재 1건뿐인 과거 데이터는 무관).
- 어떤 이모지가 긍정/부정인지는 **지금 고정하지 않고** 원시 이모지를 그대로 저장해 후속 단계에서 해석한다.

## 2. 아키텍처 / 데이터 흐름

```
[main.py 배치]  게시 시 ts↔paper 기록 → reactions/posted_index.jsonl
                         │
[harvest_reactions.py]  N일 윈도우 내 논문의 ts로 reactions.get 폴링
  (2번째 스케줄드 잡)     → reactions/reaction_snapshots.jsonl (append-only)
```

현재 `workspace._send_slack_messages`는 `chat_postMessage` 결과의 `ts`를 버린다. 이를 잡아 인덱스에 기록하는 것이 1번 작업.

## 3. 컴포넌트

### 3.1 `api/reactions.py` (신규)
순수 I/O·해시 유틸. API/슬랙 의존 최소.
- `append_posted_index(entry: dict)` — `reactions/posted_index.jsonl`에 한 줄 append.
- `iter_posted_index() -> Iterator[dict]` — 인덱스 읽기.
- `append_snapshot(entry: dict)` — `reactions/reaction_snapshots.jsonl`에 append.
- `hash_user(user_id: str) -> str` — `sha256(salt + user_id)` 앞 16자. salt는 `os.getenv("REACTION_HASH_SALT")`.
- 경로/윈도우/ salt는 `settings.py`에서 가져옴.

### 3.2 게시 시 인덱스 기록 (`api/workspace.py`)
`_send_slack_messages`에서 각 논문 답글 게시 후 결과 ts를 인덱스에 기록:
```jsonl
{"workspace": "...", "channel_id": "C...", "ts": "1718...","paper_info": "...", "paper_url": "...", "field": "cs.CL", "posted_at": "2026-06-15T00:00:00Z"}
```
(이미 `content["paper_info"]`를 다루고 있어, `chat_postMessage` 반환 ts만 추가로 기록하면 됨.)

### 3.3 수확 잡 `harvest_reactions.py` (신규, 2번째 스케줄드 태스크)
1. `iter_posted_index()`로 전체 항목 로드.
2. **윈도우 필터**: `posted_at`이 현재로부터 `HARVEST_WINDOW_DAYS`(기본 14) 이내인 항목만(그 이전은 정착으로 보고 폴링 중단 → API 절약).
3. 각 항목에 `client.reactions_get(channel=channel_id, timestamp=ts)` 호출.
4. 반환된 `message.reactions`(`[{name, users[], count}]`)에서:
   - 봇 자신의 user id(= `auth_test()["user_id"]`) 제외,
   - `users`를 `hash_user`로 변환.
5. 스냅샷 append:
```jsonl
{"ts":"...","paper_info":"...","harvested_at":"2026-06-16T...Z","reactions":[{"emoji":"thumbsup","count":2,"hashed_users":["ab12..","cd34.."]}]}
```
   - append-only라 매 수확마다 한 줄이 쌓여 **리액션 증가 추이**가 보존된다. 최신 스냅샷 = 현재 상태.
6. 리액션 0개인 논문도 빈 스냅샷을 남길지: **남기지 않는다**(YAGNI; 없으면 0으로 간주).
7. rate limit(429)·일시 오류는 `Retry-After` 백오프로 재시도.

### 3.4 `settings.py` / `.env`
- `settings.py`: `REACTIONS_DIR`, `POSTED_INDEX_PATH`, `REACTION_SNAPSHOTS_PATH`, `HARVEST_WINDOW_DAYS = 14`.
- `.env`(gitignore됨): `REACTION_HASH_SALT=<랜덤 문자열>`.

### 3.5 `.gitignore`
`reactions/` 추가 — 해시 user 데이터라 **공개 repo에 올리지 않음**(로컬 + Dropbox 백업). summaries와 달리 비공개.

## 4. 프라이버시

- user id는 salt 해시로만 저장(평문 X). salt는 `.env`(비공개).
- `reactions/` 전체 gitignore → public repo에 노출 안 됨.
- 해시는 시계열 dedup / distinct 카운트 / per-user 가중치 용도. salt 미공개라 외부 역추적 불가.

## 5. 엣지 케이스

- **인덱스 없는 과거 메시지**: 이 feature 이전 게시분은 인덱스가 없어 수확 대상 아님(forward-only, 의도된 동작).
- **봇 self-reaction**: pre-seed 안 하므로 없지만, 방어적으로 봇 user id 제외.
- **삭제/취소된 리액션**: 폴링은 현재 상태를 보므로 자연 반영(최신 스냅샷이 진실).
- **윈도우 밖**: 14일 지난 논문은 더 폴링 안 함(정착 가정).

## 6. 테스트

- **단위(`unittest`, API 불필요)**:
  - `hash_user` 결정성 + salt 반영(같은 입력·salt → 같은 해시, salt 다르면 다름).
  - posted_index append→iter 라운드트립.
  - 윈도우 필터 날짜 경계(13일 포함, 15일 제외).
  - 스냅샷 직렬화(봇 user 제외, users 해시 변환).
- **수동 smoke**: 기존 Influcoder 메시지(👀 1개)의 ts로 임시 인덱스 항목을 만들어 `harvest_reactions.py` 실행 → 스냅샷에 `eyes:1`이 잡히는지 확인(`reactions.get` 경로 검증). 검증 후 임시 항목 제거.

## 7. Out of scope (YAGNI)

- 이모지 polarity 매핑, RM 데이터셋 정형(pointwise/pairwise), 노출 편향 정규화 — 후속 스펙.
- Socket Mode 실시간, backfill, 봇 pre-seed.
