"""
모델 비교 eval - 판정(judge) 단계.
eval_results.json의 각 요약을 블라인드(모델명 숨김)로 100점 루브릭에 따라 채점한다.
강한 판정 모델을 쓰고, 원문 abstract를 함께 줘서 faithfulness까지 평가.
"""

import os
import json
import time
import pickle

import tiktoken
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENC = tiktoken.get_encoding("o200k_base")

JUDGE_MODEL = "gpt-5.2"
JUDGE_REASONING = "low"

RUBRIC = """너는 영어 arxiv 논문을 한국어로 요약한 결과물을 평가하는 엄격한 심사관이다.
요약은 4개 영어 키(prior_approaches, core_contribution, technical_challenges, empirical_impact)에
각각 한국어 값을 담아야 하며, 각 항목 2-3문장, 전체 12문장 이내가 목표다.

아래 6개 기준으로 100점 만점 채점하라(각 정수):
1. format_schema (0-15): 4개 키가 모두 의미에 맞게 채워졌는가, 형식이 깔끔한가.
2. section_fidelity (0-25): 각 섹션이 "지정된 의미"에 맞는 내용을 담았는가.
   - prior_approaches=기존 방법 분류/한계, core_contribution=무슨 문제를 푸는 기여인지,
     technical_challenges=기여 실현의 난제와 해결, empirical_impact=실증 방식과 분야적 의미.
   섹션 내용이 엉뚱하거나 라벨과 안 맞으면 크게 감점.
3. korean_fluency (0-20): 자연스러운 학술 한국어인가. 번역투/어색함은 감점.
4. codeswitch_appropriateness (0-15): 영어는 정당한 technical term에만 쓰였는가.
   불필요한 영어 남용/문장 통째 영어 출력은 큰 감점.
5. length_conciseness (0-10): 항목당 2-3문장·전체 12문장 이내·완결성(잘림 없음).
6. faithfulness (0-15): 원문(abstract)과 사실적으로 일치하는가. 환각/과장은 감점.

각 기준 점수와 한 줄 근거, 그리고 이 요약의 강점/약점을 간결히 적어라."""


class JudgeScore(BaseModel):
    format_schema: int
    section_fidelity: int
    korean_fluency: int
    codeswitch_appropriateness: int
    length_conciseness: int
    faithfulness: int
    strengths: str
    weaknesses: str


def judge(abstract: str, summary: dict):
    abs_short = ENC.decode(ENC.encode(abstract)[:800])
    user = (
        f"[원문 abstract]\n{abs_short}\n\n"
        f"[평가 대상 요약 (JSON)]\n{json.dumps(summary, ensure_ascii=False, indent=2)}"
    )
    resp = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": user}],
        response_format=JudgeScore,
        reasoning_effort=JUDGE_REASONING,
        max_completion_tokens=3000,
    )
    s = resp.choices[0].message.parsed
    total = (s.format_schema + s.section_fidelity + s.korean_fluency
             + s.codeswitch_appropriateness + s.length_conciseness + s.faithfulness)
    d = s.model_dump()
    d["total"] = total
    return d


def main():
    results = json.load(open(os.path.join(HERE, "eval_results.json"), encoding="utf-8"))
    abstracts = pickle.load(open(os.path.join(ROOT, "cache/paper_abstracts.pickle"), "rb"))
    for i, rec in enumerate(results):
        info = rec["paper_info"]
        abstract = abstracts.get(info, "")
        print(f"\n[{i+1}/{len(results)}] {info[:70]}")
        for model, mres in rec["models"].items():
            if not mres.get("ok"):
                print(f"  {model:14s} (skip, gen failed)")
                continue
            try:
                sc = judge(abstract, mres["summary"])
                mres["judge"] = sc
                print(f"  {model:14s} total={sc['total']:3d}  "
                      f"fmt={sc['format_schema']} sec={sc['section_fidelity']} "
                      f"flu={sc['korean_fluency']} cs={sc['codeswitch_appropriateness']} "
                      f"len={sc['length_conciseness']} faith={sc['faithfulness']}")
            except Exception as e:
                mres["judge"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {model:14s} JUDGE ERROR {type(e).__name__}: {str(e)[:120]}")
            time.sleep(0.3)
    out = os.path.join(HERE, "eval_judged.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
