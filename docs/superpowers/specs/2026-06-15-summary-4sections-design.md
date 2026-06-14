# 요약 4섹션 재설계 + 모델 전환 (Feature 2)

**날짜**: 2026-06-15
**범위**: arxivbot 논문 요약의 (a) 섹션 구조 3→4 재설계, (b) 요약 모델을 `gpt-5.4-nano`로 전환(reasoning), (c) 프롬프트 하드닝.
**비범위**: Slack 리액션 수집/reward model(별도 Feature 1 스펙).

---

## 1. 동기

현재 요약은 3섹션(What's New / Technical Details / Performance Highlights)로, 논문의 *위치(기존 방법 대비 무엇을·왜·어떻게)*를 드러내지 못한다. 다음 4관점으로 재구성해 분석적 깊이를 높인다(단, 전체 길이는 기존 수준 유지):

1. 기존 방법 분류·한계
2. 그 한계 중 무엇을/어떤 문제를 푸는 기여인지
3. 기여 실현의 technical challenge와 해결
4. 기여가 어떻게 empirical하게 입증됐고 분야에 갖는 의미/impact

---

## 2. 결정과 근거 (eval 기반)

20편 × 4모델(gpt-4o-mini / gpt-5-nano / gpt-5.4-nano / gpt-5-mini) 생성 → blind judge + 직접 정독으로 검증. (`eval/` 디렉터리에 스크립트·결과 보존)

- **모델 = `gpt-5.4-nano`, `reasoning_effort="low"`**: 새 4섹션의 핵심인 *분석 깊이*에서 1위(section_fidelity 최고, 20편 중 12편 1위). 비용 $1.43/1k편(월 $1~3 수준, 무시 가능).
- **코드스위칭 오해 정정**: 5.4-nano의 낮은 한글비율은 *정당한 영어 용어 밀도*이지 unexpected code-switching이 아님. 전체 80건 중 진짜 타 언어 누수는 **gpt-4o-mini 1건(중국어 `显著`)**뿐.
- **gpt-4o-mini**(현재)는 한국어는 깨끗하나 분석이 가장 얕아 재설계 취지에 부적합.
- **프롬프트 하드닝 효과(재검증)**: 용어 한국어화 + 길이 캡 적용 시 한글비율 0.57→0.80, 평균 12.7→11.2문장, 12문장 초과 7/20→3/20, **깊이는 유지**. → 채택.

---

## 3. 변경 사항

### 3.1 `prompts.py`
- `SummarizationResponse` 필드 교체:
  `prior_approaches` / `core_contribution` / `technical_challenges` / `empirical_impact`
- `SYSTEM_PROMPT_SUMMARIZATION` 교체(하드닝):
  - 4섹션 각 의미 1줄 가이드(위 §1).
  - **길이 캡**: 각 섹션 2–3문장, **전체 12문장 초과 금지**.
  - **용어 한국어화**: 널리 통용되는 개념은 한국어로, 정착 한국어가 없거나 고유명(모델/기법/벤치마크명)만 영어; 처음 등장 시 병기.
  - **음차 금지**: 영어를 한글로 음차(transliteration)하지 말 것 — 한국어로 번역하거나 영어 원문 유지.
  - 키는 영어, 값은 한국어 유지.

### 3.2 `api/agent.py` (`GptAgent`)
- API 호출에 `reasoning_effort="low"` 추가.
- `max_tokens` → `max_completion_tokens`로 교체(reasoning 토큰이 출력 예산을 잠식하므로).
- 응답 dict 키 4개로 교체:
  `"Prior Approaches" / "Core Contribution" / "Technical Challenges" / "Empirical Impact"`.

### 3.3 `settings.py`
- `MODEL = "gpt-5.4-nano"`.
- `MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION` → `max_completion_tokens` 용도로 상향(약 4000). eval에서 4000일 때 전 건 `finish_reason=stop`(잘림 0) 확인.

### 3.4 `Encoder` (`api/agent.py`)
- `tiktoken.encoding_for_model("gpt-5.4-nano")`는 **KeyError**. → `o200k_base`로 fallback:
  ```python
  try:
      self.encoding = tiktoken.encoding_for_model(model_name)
  except KeyError:
      self.encoding = tiktoken.get_encoding("o200k_base")
  ```

### 3.5 변경 불필요(확인됨)
- `workspace.prepare_content` / `save_summaries`: 요약 dict를 generic 순회 → 키 변경 자동 대응.
- `AutoAgent.from_model_name`: `startswith("gpt")` → `gpt-5.4-nano` 통과.

---

## 4. 호환성 / 엣지

- **캐시 혼재 허용**: `has_paper_summarization`가 캐시된 논문은 재요약하지 않으므로 과거 논문은 3섹션, 신규는 4섹션으로 공존. 렌더링은 둘 다 정상. 일괄 재생성은 하지 않는다(비용).
- 섹션 출력 순서 = dict 삽입 순서(위 4개 순).

---

## 5. 테스트

- `test.py`(또는 eval 스크립트)로 논문 1~2편 요약: 4개 키 존재, 한국어 값, 전체 ≤12문장, `finish_reason=stop`, 타 언어 누수 없음 확인.

## 6. Out of scope (YAGNI)

- 과거 요약 일괄 재생성, Gemini 경로 부활, 길이 자동 측정·강제, Batch API 전환.
