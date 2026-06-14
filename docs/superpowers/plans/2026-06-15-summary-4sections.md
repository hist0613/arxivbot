# 요약 4섹션 재설계 + gpt-5.4-nano 전환 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** arxivbot 논문 요약을 3섹션→4섹션(Prior Approaches / Core Contribution / Technical Challenges / Empirical Impact)으로 재설계하고, 요약 모델을 `gpt-5.4-nano`(reasoning low)로 전환하며 프롬프트를 하드닝한다.

**Architecture:** 기존 파이프라인(crawl→summarize→Slack→save)은 그대로. 요약 스키마(`prompts.py`)·생성 호출(`api/agent.py`)·모델 설정(`settings.py`)만 변경. 렌더링은 dict를 generic 순회하므로 무수정. 캐시된 과거 3섹션 요약과 신규 4섹션은 공존 허용.

**Tech Stack:** Python, OpenAI SDK(`beta.chat.completions.parse`, structured outputs), pydantic, tiktoken. 테스트는 stdlib `unittest`(결정적) + 수동 API smoke.

**기준 스펙:** `docs/superpowers/specs/2026-06-15-summary-4sections-design.md`
**브랜치:** `feature/summary-4sections`

---

## ⚠️ 실행 환경 (중요)

- **Production 실행은 Windows PowerShell의 Python**(`python`, 3.11.9)에서 한다. 모든 검증 명령은 **PowerShell 기준**으로 적는다 (`python tests\...`, 백슬래시 경로). 작업 디렉터리: `C:\Users\hist0\Dropbox\develop\arxivbot_new`.
- **버전 함정**: Windows Python의 `openai==1.40.6`은 `reasoning_effort`/`max_completion_tokens`를 **지원하지 않는다**. gpt-5.4-nano + reasoning을 쓰려면 **Task 0에서 openai를 2.x로 업그레이드**해야 한다(이 업그레이드 전엔 Task 5 smoke·production이 실패).
- 단위 테스트(Task 1~4)는 API를 호출하지 않으므로 업그레이드 전에도 통과한다.
- git 커밋 명령은 PowerShell/bash 양쪽에서 동작하도록 `-m`을 두 번 쓴다(heredoc 미사용).

---

## File Structure

- `requirements.txt` (수정): `openai` 핀 상향.
- `prompts.py` (수정): `SummarizationResponse` 4필드 + 하드닝 `SYSTEM_PROMPT_SUMMARIZATION`.
- `api/agent.py` (수정): `Encoder` o200k_base fallback; `GptAgent` reasoning_effort·max_completion_tokens·4키 응답; module-level `_summary_to_dict` 헬퍼.
- `settings.py` (수정): `MODEL`, 출력 토큰 예산.
- `tests/test_summary.py` (생성): 결정적 단위 테스트.
- `tests/smoke_summary.py` (생성): 실제 API 1편 생성 검증(수동 실행).

---

## Task 0: OpenAI SDK 업그레이드 (Windows 환경)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 현재 버전 확인**

Run (PowerShell): `python -c "import openai; print(openai.__version__)"`
Expected: `1.40.6` (또는 2.0 미만).

- [ ] **Step 2: 업그레이드**

Run (PowerShell): `python -m pip install -U "openai>=2.0"`
Expected: 설치 완료, 2.x 버전.

- [ ] **Step 3: 버전·호환 확인**

Run (PowerShell): `python -c "import openai; print(openai.__version__); from openai import OpenAI; assert hasattr(OpenAI().beta.chat.completions, 'parse')"`
Expected: 2.x 출력, 에러 없음. (네트워크/키 불필요 — 클라이언트 생성만)

- [ ] **Step 4: `requirements.txt` 핀 상향** — 기존 `openai` 줄을 교체

```
openai>=2.0
```

- [ ] **Step 5: 커밋**

```bash
git add requirements.txt
git commit -m "chore: openai SDK 2.x로 핀 상향 (reasoning_effort 지원)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: 요약 스키마 + 하드닝 프롬프트 (`prompts.py`)

**Files:**
- Modify: `prompts.py` (`SummarizationResponse`, `SYSTEM_PROMPT_SUMMARIZATION`)
- Test: `tests/test_summary.py`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_summary.py` 생성

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSchema(unittest.TestCase):
    def test_summarization_response_has_four_sections(self):
        from prompts import SummarizationResponse
        self.assertEqual(
            set(SummarizationResponse.model_fields),
            {"prior_approaches", "core_contribution",
             "technical_challenges", "empirical_impact"},
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: FAIL — 현재 필드가 `whats_new/technical_details/performance_highlights`라 set 불일치(AssertionError).

- [ ] **Step 3: `prompts.py` 수정** — `SummarizationResponse`와 시스템 프롬프트 교체

```python
class SummarizationResponse(BaseModel):
    prior_approaches: str
    core_contribution: str
    technical_challenges: str
    empirical_impact: str
```

같은 파일의 `SYSTEM_PROMPT_SUMMARIZATION`을 아래로 교체:

```python
SYSTEM_PROMPT_SUMMARIZATION = """Please analyze the arxiv paper and write a Korean AI-newsletter style summary in JSON.
Use exactly these four English keys, each with a Korean value of 2-3 sentences.
HARD LIMIT: the whole summary MUST NOT exceed 12 sentences total. Be concise.

언어 규칙(중요):
- 널리 통용되는 개념/표현은 한국어로 쓴다 (예: frequency distribution→주파수 분포, noise→잡음, baseline→기준선).
- 정착된 한국어 표현이 없거나 고유명사(모델명/기법명/벤치마크명)인 경우에만 영어를 쓰고, 처음 등장 시 한국어(영어)로 병기한다.
- 한국어로 자연스러운데 굳이 영어를 남발하지 말 것. 문장 구조는 항상 한국어.
- 영어를 한글로 음차(transliteration)하지 말 것 — 한국어로 번역하거나 영어 원문을 그대로 유지한다.

각 섹션의 의미:
- prior_approaches: 이 논문이 다루는 문제의 기존 방법들을 분류하고 그 한계를 설명.
- core_contribution: 기존 한계 중 무엇을/어떤 문제를 이 논문의 기여가 해결하는지.
- technical_challenges: 그 기여 실현의 technical challenge와 이를 어떻게 해결했는지.
- empirical_impact: 기여가 어떻게 empirical하게 입증됐고 해당 분야에서 갖는 의미/impact.

답은 JSON 형식이며 키는 영어로 둔다."""
```

- [ ] **Step 4: 통과 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: `test_summarization_response_has_four_sections ... ok`

- [ ] **Step 5: 커밋**

```bash
git add prompts.py tests/test_summary.py
git commit -m "feat: 요약 스키마 4섹션 + 하드닝 프롬프트" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Encoder 토크나이저 fallback (`api/agent.py`)

**Files:**
- Modify: `api/agent.py` (`Encoder.__init__`)
- Test: `tests/test_summary.py`

- [ ] **Step 1: 실패하는 테스트 추가** — `tests/test_summary.py`에 클래스 추가

```python
class TestEncoder(unittest.TestCase):
    def test_fallback_for_unmapped_model(self):
        from api.agent import Encoder
        # gpt-5.4-nano는 tiktoken 매핑이 없어 KeyError → o200k_base로 fallback
        enc = Encoder("gpt-5.4-nano")
        self.assertEqual(enc.encoding.name, "o200k_base")

    def test_known_model_still_works(self):
        from api.agent import Encoder
        self.assertEqual(Encoder("gpt-4o-mini").encoding.name, "o200k_base")
```

- [ ] **Step 2: 실패 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: `test_fallback_for_unmapped_model` FAIL — `KeyError: Could not automatically map gpt-5.4-nano ...`

- [ ] **Step 3: `Encoder.__init__` 수정** — `api/agent.py`

```python
class Encoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")
```

- [ ] **Step 4: 통과 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: `TestEncoder` 2건 ok.

- [ ] **Step 5: 커밋**

```bash
git add api/agent.py tests/test_summary.py
git commit -m "feat: Encoder tiktoken o200k_base fallback" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: GptAgent 응답 4키 + reasoning + max_completion_tokens (`api/agent.py`)

**Files:**
- Modify: `api/agent.py` (module-level 헬퍼 추가, `GptAgent.summarize`, `GptAgent._generate_content`)
- Test: `tests/test_summary.py`

- [ ] **Step 1: 실패하는 테스트 추가** — `tests/test_summary.py`에 클래스 추가

```python
class TestSummaryDict(unittest.TestCase):
    def test_summary_to_dict_maps_four_english_keys(self):
        from prompts import SummarizationResponse
        from api.agent import _summary_to_dict
        parsed = SummarizationResponse(
            prior_approaches="기존 방법",
            core_contribution="핵심 기여",
            technical_challenges="기술 난제",
            empirical_impact="실증 의미",
        )
        d = _summary_to_dict(parsed)
        self.assertEqual(
            list(d.keys()),
            ["Prior Approaches", "Core Contribution",
             "Technical Challenges", "Empirical Impact"],
        )
        self.assertEqual(d["Core Contribution"], "핵심 기여")
```

- [ ] **Step 2: 실패 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: FAIL — `ImportError: cannot import name '_summary_to_dict'`.

- [ ] **Step 3: `api/agent.py` 수정**

(a) module-level(클래스 밖, `import json` 이후 아무 곳)에 헬퍼 추가:

```python
def _summary_to_dict(parsed) -> dict:
    return {
        "Prior Approaches": parsed.prior_approaches,
        "Core Contribution": parsed.core_contribution,
        "Technical Challenges": parsed.technical_challenges,
        "Empirical Impact": parsed.empirical_impact,
    }
```

(b) `GptAgent.summarize` — 인자명 변경:

```python
    @llm_retry(max_trials=MAX_LLM_TRIALS)
    def summarize(self, content: str) -> str:
        return self._generate_content(
            system_prompt=self.system_prompt_for_summarization,
            user_prompt=content,
            max_completion_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
        )
```

(c) `GptAgent._generate_content` 교체 + reasoning 가드 메서드 추가:

```python
    def _is_reasoning_model(self) -> bool:
        n = self.model_name.lower()
        return n.startswith("gpt-5") or n.startswith("o")

    def _generate_content(
        self, system_prompt: str, user_prompt: str, max_completion_tokens=None
    ) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_completion_tokens,
            response_format=self.response_format,
        )
        if self._is_reasoning_model():
            kwargs["reasoning_effort"] = "low"
        response = self.client.beta.chat.completions.parse(**kwargs)
        parsed = response.choices[0].message.parsed
        return json.dumps(_summary_to_dict(parsed))
```

- [ ] **Step 4: 통과 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: `TestSummaryDict` ok, 그리고 기존 테스트 전부 ok.

- [ ] **Step 5: 커밋**

```bash
git add api/agent.py tests/test_summary.py
git commit -m "feat: GptAgent 4키 응답 + reasoning_effort + max_completion_tokens" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 모델·토큰 예산 설정 (`settings.py`)

**Files:**
- Modify: `settings.py:14`, `settings.py:17`
- Test: `tests/test_summary.py`

- [ ] **Step 1: 실패하는 테스트 추가**

```python
class TestSettings(unittest.TestCase):
    def test_model_and_budget(self):
        import settings
        self.assertEqual(settings.MODEL, "gpt-5.4-nano")
        self.assertGreaterEqual(settings.MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION, 4000)
```

- [ ] **Step 2: 실패 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: FAIL — MODEL이 `gpt-4o-mini`, 예산 1024.

- [ ] **Step 3: `settings.py` 수정**

`settings.py:14`:
```python
MODEL = "gpt-5.4-nano"
```
`settings.py:17` (reasoning 토큰이 예산을 잠식하므로 상향; max_completion_tokens 용도):
```python
MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION = 4000  # reasoning + 출력 합산 예산 (max_completion_tokens)
```

- [ ] **Step 4: 통과 확인**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: 전체 ok.

- [ ] **Step 5: 커밋**

```bash
git add settings.py tests/test_summary.py
git commit -m "feat: 요약 모델 gpt-5.4-nano 전환 + 출력 예산 상향" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 실제 API smoke 검증 (수동 실행, Task 0 선행)

**Files:**
- Create: `tests/smoke_summary.py`

비용/네트워크가 필요해 unittest에 포함하지 않고 1회 수동 실행한다. 캐시의 논문 1편으로 실제 요약을 생성해 4키·길이·타 언어 누수를 검증한다. **Task 0(openai 2.x) 선행 필수.**

- [ ] **Step 1: `tests/smoke_summary.py` 생성**

```python
"""실제 API로 gpt-5.4-nano 요약 1편 생성 검증. 수동 실행(비용 발생, openai>=2 필요)."""
import os, sys, json, re, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from settings import MODEL, MAX_INPUT_TOKENS_FOR_SUMMARIZATION
from api.agent import AutoAgent, Encoder

EXPECTED_KEYS = ["Prior Approaches", "Core Contribution",
                 "Technical Challenges", "Empirical Impact"]


def main():
    abstracts = pickle.load(open("cache/paper_abstracts.pickle", "rb"))
    fulls = pickle.load(open("cache/paper_full_contents.pickle", "rb"))
    info = next(k for k in fulls if isinstance(fulls[k], dict)
                and len(abstracts.get(k, "")) > 200)

    enc = Encoder(MODEL)
    s = f"Abstract: {abstracts[info]}\n\n"
    for sec in fulls[info].values():
        if sec.get("title") != "No title found" and sec.get("content"):
            s += f"Section: {sec['title']}\n{sec['content']}\n\n"
    tokens = enc.encoding.encode(
        s, allowed_special={"<|endoftext|>"}, disallowed_special=()
    )[:MAX_INPUT_TOKENS_FOR_SUMMARIZATION]
    user = enc.encoding.decode(tokens)

    agent = AutoAgent.from_model_name(MODEL)
    out = json.loads(agent.summarize(user))

    print("paper:", info[:70])
    print(json.dumps(out, ensure_ascii=False, indent=2))
    assert list(out.keys()) == EXPECTED_KEYS, f"키 불일치: {list(out.keys())}"
    text = " ".join(out.values())
    assert not re.search(r"[一-鿿Ѐ-ӿ]", text), "중국어/키릴 누수 발견"
    sents = len([p for p in re.split(r"[.!?。]+", text) if p.strip()])
    print(f"문장수={sents}, keys OK, 타 언어 누수 없음")
    assert sents <= 16, f"문장 과다: {sents}"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 프로젝트 루트에서 실행**

Run (PowerShell): `python tests\smoke_summary.py`
Expected: 4키 JSON 출력 + `SMOKE PASS` (문장 ≤16, 타 언어 누수 없음). 실패 시 AssertionError로 원인 표시.

- [ ] **Step 3: 커밋**

```bash
git add tests/smoke_summary.py
git commit -m "test: gpt-5.4-nano 요약 API smoke 검증 스크립트" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 최종 검증

- [ ] **Step 1: 전체 단위 테스트**

Run (PowerShell): `python tests\test_summary.py -v`
Expected: 5개 테스트(schema, encoder×2, summary_dict, settings) 전부 ok.

- [ ] **Step 2: end-to-end 스모크** — Task 5 실행으로 갈음(Task 0 선행).

- [ ] **Step 3: 브랜치 상태 확인**

Run: `git log --oneline feature/summary-4sections -8`
Expected: Task 0~5 커밋 + 스펙 커밋이 보임.

---

## Self-Review 결과 (작성자 점검)

- **스펙 커버리지**: §3.1→Task1, §3.4(Encoder)→Task2, §3.2→Task3, §3.3→Task4, §5(테스트)→Task5/6, §3.5(무변경)→설계상 확인. §4(캐시 혼재)는 `has_paper_summarization` 미변경으로 자동 성립(별도 task 불필요). 환경 함정(openai 버전)→Task0 추가.
- **플레이스홀더**: 없음(모든 코드/명령 실체 포함).
- **타입 일관성**: `SummarizationResponse` 필드명(snake_case) ↔ `_summary_to_dict` 접근자 일치, 응답 키 4개 ↔ 테스트 기대값 일치, `max_completion_tokens` 인자명 호출부/정의부 일치.
- **셸 일관성**: 검증/실행 명령은 PowerShell(`python`, 백슬래시), 커밋은 `-m`×2로 PowerShell·bash 공용.
