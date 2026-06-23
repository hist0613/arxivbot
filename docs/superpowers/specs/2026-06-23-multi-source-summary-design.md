# 다출처(multi-source) on-demand 요약 + 단계별 진행 표시 — 설계

- 날짜: 2026-06-23
- 관련: [on-demand-summary](2026-06-16-on-demand-summary-design.md), [summary-4sections](2026-06-15-summary-4sections-design.md)

## 목적

현재 on-demand 리스너(`listener.py`)는 **arXiv 링크만** @멘션으로 요약한다. 이를 확장해 **여러 출처의 논문 페이지 URL**(ACL Anthology, CVPR/ICCV, NeurIPS, ICML, ICLR, AAAI, IJCAI, Interspeech, 일반 PDF 등)을 멘션하면, 봇이 알아서 **PDF를 받아 텍스트를 추출해 요약**하게 한다. 또한 fetch→download→summarize에 시간이 걸리므로 **Slack 메시지 1개를 편집(in-place)** 하며 진행 상황을 보여준다.

## 사용자 결정 사항 (확정)

- **입력**: URL만 (봇이 받아서 처리). Slack 파일 업로드는 비범위.
- **전략**: 도메인별 HTML 리졸버 + **잡식성 폴백 cascade**. 다양한 URL 패턴(직접 PDF 링크 등) 반드시 수용.
- **범위**: 표준 proceedings 호스트 우선. **ACM(dl.acm.org)은 best-effort**(시도하되 막히면 명확한 미지원 메시지).
- **PDF 추출**: `pymupdf4llm`. (조사 결론 — 아래 근거)
- **수식**: 산문 수준으로 충분(요약은 prose 정보 사용). 수식 충실도(=GPU 도구)는 도입 안 함. 대신 프롬프트에 **아티팩트 무시** 한 줄 추가.
- **진행 표시**: 같은 메시지를 `chat.update`로 편집(새 메시지 X, 알림 스팸 방지).

## 비범위 (YAGNI)

- Slack PDF 파일 업로드 처리
- 수식/표/참고문헌 구조 추출 (Nougat/GROBID/Marker 등 GPU·서비스형 도구)
- 스캔본(이미지 PDF) OCR

## PDF 추출 라이브러리 — `pymupdf4llm` (근거)

벤치마크 조사(pdfmux 200-PDF, READoc ACL2025, arXiv 비교연구) 결론:
- 우리는 **평문 텍스트(앞 6~10p)** 만 필요하고 6000토큰으로 truncate한다. 표·수식·구조는 안 씀.
- `pymupdf4llm`은 **2단 reading-order를 정확히 처리**, **순수 pip·GPU/Java 불필요·~0.01s/page** → Windows 스케줄러 작업에 최적. 평문 추출 정확도 최상위권.
- 더 정확한 도구(Marker/Nougat/GROBID/Docling)는 표·수식·구조에서 앞서지만 GPU/Docker 비용이 크고 *우리 지표*는 거의 못 올림.
- **약점**: 스캔본(이미지 PDF)은 텍스트가 빈값 → `< 500자` 가드로 추출 실패 처리.
- **라이선스**: AGPL-3.0(개인·연구 내부용 무방). 향후 배포형 폐쇄 제품화 시 **Docling(MIT)** 로 교체 — 미래 폴백으로 기록.

## 아키텍처

### `api/pdf.py` (신규)
- `download_pdf(url) -> bytes | None`: 식별 UA·타임아웃·재시도로 GET. Content-Type이 pdf이거나 본문이 `%PDF`로 시작하면 bytes 반환, 아니면 None.
- `extract_text(pdf_bytes, max_pages=8) -> str`: `pymupdf.open(stream=…)` 후 `pymupdf4llm.to_markdown(doc, pages=range(min(max_pages, page_count)))`. 앞 8페이지만.
- `MIN_PDF_TEXT_CHARS = 500`: 호출부가 `len(text.strip()) < 500`이면 추출 실패로 간주.

### `api/resolvers.py` (신규)
- `ResolvedPaper(NamedTuple)`: `title: str, url: str, text: str, source: str`.
- `extract_first_url(text) -> str | None`: 멘션 텍스트(Slack은 `<url>`/`<url|label>`로 감쌈)에서 첫 http(s) URL 추출.
- `_fetch_html(url) -> BeautifulSoup`, `_find_pdf_link(soup, base_url) -> str | None`(generic: href가 `.pdf`거나 링크 텍스트에 "PDF").
- **호스트 리졸버** `resolve_<host>(url, cache, arxiv_client, on_progress) -> ResolvedPaper`:
  - `arxiv`: 기존 경로 — abstract + HTML(experimental) 섹션 텍스트. **HTML 없으면 `https://arxiv.org/pdf/<id>` 다운로드→추출**(on_progress("downloading")). canonical = abs URL.
  - `aclanthology` / `thecvf`(CVPR·ICCV·WACV) / `neurips` / `mlr.press`(ICML) / `openreview`(ICLR) / `ojs.aaai.org` / `ijcai.org` / `isca-archive.org`: 페이지 HTML→제목 + PDF 링크 → 다운로드(on_progress("downloading"))→추출.
  - `dl.acm.org`(best-effort): PDF 링크 시도. 차단되면 빈 text → 상위에서 미지원 처리.
  - `direct_pdf`: URL이 `.pdf`거나 Content-Type=pdf → 직접 다운로드→추출. 제목은 PDF 메타데이터/첫 헤딩, 없으면 URL 파일명.
- `build_resolver(arxiv_client, cache)` → `resolve(url, on_progress=lambda s: None) -> ResolvedPaper | None` (클로저로 arxiv_client·cache 캡처; `build_fetch_paper`와 동일 패턴) — **폴백 cascade**:
  1. netloc로 호스트 리졸버 선택 → 실행.
  2. 호스트 매칭됐지만 text 비었거나 미매칭 → 페이지에서 `_find_pdf_link` generic 탐색 → direct_pdf.
  3. URL이 PDF면 direct_pdf.
  4. 다 실패 → None.

### `Service.summarize_text(paper_info, text)` (신규, `api/service.py`)
모든 출처 통일 진입점. 스키마 검증 캐시 재사용([[on-demand 캐시 로직]]) → 미스면 `encoder.truncate_text(text)` 후 `agent.summarize` → 비어있지 않으면 캐시 저장. 기존 `summarize_one`은 이걸 호출하도록 위임(arXiv 구조화 입력 유지).

### `api/on_demand.py` 개정
`process_mention(text, *, cache, service, resolve, workspace, on_progress=lambda s: None) -> dict`:
1. `extract_first_url` → 없으면 안내.
2. `on_progress("fetching")` → `resolve(...)`(내부에서 필요 시 `on_progress("downloading")`).
3. `resolved is None`/빈 text → 미지원·실패 안내(지원 출처 목록 포함).
4. `paper_info = get_paper_info(resolved.url, resolved.title)`; `on_progress("summarizing")`; `service.summarize_text(...)`.
5. 빈 요약 → 실패 안내. 아니면 `workspace.prepare_content`로 4섹션 포맷.
반환 dict: `{ok, message, paper_info, paper_url}`(기존과 동일).

### `listener.py` 개정 (진행 표시)
```
thread_ts = resolve_thread_ts(event)
posted = client.chat_postMessage(channel, text=STAGE["fetching"], thread_ts=thread_ts)
ts = posted["ts"]
on_progress(stage): 같은 ts를 STAGE[stage] 텍스트로 chat_update (직전과 동일 텍스트면 skip)
result = process_mention(..., resolve=resolve, on_progress=on_progress)
client.chat_update(channel, ts, text=result["message"])   # 최종(요약 또는 에러)
result["ok"]이면 add_posted(ts=ts, channel_id=channel, field="on-demand", ...)
```
`STAGE = {"fetching": "🔄 논문 페이지 가져오는 중…", "downloading": "🔄 PDF 다운로드 중…", "summarizing": "🔄 AI가 요약하는 중…"}`. `resolve`는 `build_resolver(arxiv_client, cache)` 클로저로 주입.

### `prompts.py` 개정
- `"analyze the arxiv paper"` → `"analyze the paper"`.
- 아티팩트 내성 한 줄 추가(언어 규칙 근처): *"입력에 PDF 추출 아티팩트(깨진 수식·하이픈 분리·열 섞임·머리말/꼬리말)가 섞일 수 있다. 깨진 기호·잡음은 무시하고 산문 의미로 요약하라."*

## 데이터 흐름
```
@arxivbot <url>
 → "🔄 논문 페이지 가져오는 중…" 답글(ts 확보)
 → resolve(url): 호스트 리졸버 → (필요시) "🔄 PDF 다운로드 중…" → download_pdf → extract_text
       arXiv: HTML 있으면 HTML, 없으면 PDF 폴백
 → "🔄 AI가 요약하는 중…"
 → summarize_text(paper_info, text)  (스키마검증 캐시)
 → chat_update(ts) → 최종 4섹션 요약
 → add_posted(ts)  리액션 등록
```

## 에러 처리 (모두 같은 메시지 편집으로 통지)
- URL 없음 → 링크 요청 안내.
- 미지원 도메인 / PDF 다운로드 실패 / 추출 텍스트 `<500자`(스캔본) → "이 출처는 아직 지원 안 해요 (지원: arXiv, ACL, CVPR/ICCV, NeurIPS, ICML, OpenReview, AAAI, IJCAI, Interspeech)" 류.
- 요약 빈값 → 재시도 안내. 모든 예외는 잡아 한 줄 통지 + 로그, 리스너 유지.

## 테스트 / 검증
- **단위(`tests/test_resolvers.py`, 네트워크 없이)**: `extract_first_url`(Slack 래핑/일반/label), 호스트 선택 로직, `_find_pdf_link`(저장한 HTML fixture 스니펫), direct-pdf 판별.
- **단위(`tests/test_on_demand.py` 추가)**: `process_mention`에 fake `resolve` + `on_progress` 레코더 주입 → **단계 순서**(`fetching`→(`downloading`)→`summarizing`)와 결과 검증. `summarize_text` 캐시 히트/미스/스키마.
- **스모크(`tests/smoke_resolvers.py`, 수동·네트워크·비용)**: 호스트별 알려진 논문 URL 표 → resolve → 제목·추출 텍스트 길이 출력 + **PDF를 tmp에 저장해 직접 열어 확인**(요청한 검증 방식). 각 호스트 1편 이상.
- 의존성: `requirements.txt`에 `pymupdf4llm` 추가.

## 구현 순서 (플랜에서 단계화)
1. **진행 표시 UI**를 현재 arXiv 경로에 먼저 — handler 게시→편집 리팩터, `process_mention(on_progress)`, arxiv-only `resolve`. 작고 end-to-end 검증.
2. **`pdf.py` + 리졸버 프레임워크** + `direct_pdf` + **arXiv PDF 폴백**.
3. **호스트 리졸버 점진 추가**(acl → thecvf → neurips → mlr → openreview → aaai → ijcai → isca), 각각 스모크 + PDF 직접 열어 검증. ACM은 best-effort로 맨 끝.
