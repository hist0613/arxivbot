"""
모델 비교 eval - 요약 생성 단계.
4개 모델(gpt-4o-mini, gpt-5-nano, gpt-5.4-nano, gpt-5-mini)로
새 4섹션 프롬프트 + schema-guided decoding을 돌려 요약을 생성하고,
객관 지표(스키마 통과/키/길이/코드스위칭/finish_reason/토큰·비용·지연)를 수집한다.
"""

import os
import re
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

# ---- 설정 -------------------------------------------------------------
N_PAPERS = 20
MAX_INPUT_TOKENS = 2048          # 현 prod와 동일
MAX_COMPLETION_TOKENS = 4000     # reasoning 토큰 여유 포함
REASONING_EFFORT = "low"

# (model, is_reasoning, price_in, price_cached, price_out) per 1M tokens (Standard)
MODELS = [
    ("gpt-4o-mini",  False, 0.15, 0.075, 0.60),
    ("gpt-5-nano",   True,  0.05, 0.005, 0.40),
    ("gpt-5.4-nano", True,  0.20, 0.02,  1.25),
    ("gpt-5-mini",   True,  0.25, 0.025, 2.00),
]

ENC = tiktoken.get_encoding("o200k_base")


# ---- 새 4섹션 스키마 / 프롬프트 ---------------------------------------
class SummarizationResponse(BaseModel):
    prior_approaches: str
    core_contribution: str
    technical_challenges: str
    empirical_impact: str


SYSTEM_PROMPT = """Please analyze the arxiv paper and write a Korean AI-newsletter style summary in JSON.
Use exactly these four English keys, each with a Korean value of 2-3 sentences. Keep the WHOLE summary within 12 sentences total. Technical keywords may stay in English; add the original English in parentheses when a Korean term is unusual.

Write the four sections with this meaning:
- prior_approaches: 이 논문이 다루는 문제에 대한 기존 방법들을 분류하고 그 한계를 설명.
- core_contribution: 기존 방법의 한계 중 무엇을, 어떤 문제를 이 논문의 기여가 해결하는지.
- technical_challenges: 그 기여를 실현하는 데 있었던 technical challenge와 이를 어떻게 해결했는지.
- empirical_impact: 그 기여가 어떻게 empirical하게 입증되었고, 해당 분야에서 어떤 의미/impact를 갖는지.

arxiv 논문을 분석해 위 네 항목을 한국어로 요약하세요. 각 항목 2-3문장, 전체 12문장 이내. 답은 JSON 형식이며 키는 영어로 둡니다."""


# ---- 입력 준비 (prod prepare_summarization_input 재현) ----------------
def prepare_input(abstract: str, full_content) -> str:
    s = f"Abstract: {abstract}\n\n"
    if isinstance(full_content, dict):
        for sec in full_content.values():
            if sec.get("title") != "No title found" and sec.get("content"):
                s += f"Section: {sec['title']}\n{sec['content']}\n\n"
    toks = ENC.encode(s, allowed_special={"<|endoftext|>"}, disallowed_special=())
    return ENC.decode(toks[:MAX_INPUT_TOKENS])


# ---- 객관 지표 -------------------------------------------------------
def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?。]+", text)
    return len([p for p in parts if p.strip()])


def hangul_ratio(text: str) -> float:
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if hangul + latin == 0:
        return 0.0
    return hangul / (hangul + latin)


def metrics(parsed: SummarizationResponse) -> dict:
    secs = {
        "prior_approaches": parsed.prior_approaches,
        "core_contribution": parsed.core_contribution,
        "technical_challenges": parsed.technical_challenges,
        "empirical_impact": parsed.empirical_impact,
    }
    per = {}
    total_sent = 0
    no_hangul = []
    for k, v in secs.items():
        sc = count_sentences(v)
        hr = hangul_ratio(v)
        total_sent += sc
        if hr == 0.0 and v.strip():
            no_hangul.append(k)
        per[k] = {"sentences": sc, "hangul_ratio": round(hr, 3), "chars": len(v)}
    return {
        "per_section": per,
        "total_sentences": total_sent,
        "overall_hangul_ratio": round(hangul_ratio(" ".join(secs.values())), 3),
        "sections_without_hangul": no_hangul,       # 완전 code-switch(영어 출력) 섹션
        "all_keys_nonempty": all(v.strip() for v in secs.values()),
    }


# ---- 모델 호출 -------------------------------------------------------
def call(model: str, is_reasoning: bool, user: str):
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=SummarizationResponse,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    if is_reasoning:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    t0 = time.time()
    resp = client.beta.chat.completions.parse(**kwargs)
    dt = time.time() - t0
    choice = resp.choices[0]
    usage = resp.usage
    rt = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return {
        "parsed": choice.message.parsed,
        "refusal": choice.message.refusal,
        "finish_reason": choice.finish_reason,
        "latency_sec": round(dt, 2),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": rt,
    }


def cost(price_in, price_out, in_tok, out_tok):
    return round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 6)


# ---- 논문 선택 -------------------------------------------------------
def pick_papers():
    abstracts = pickle.load(open(os.path.join(ROOT, "cache/paper_abstracts.pickle"), "rb"))
    fulls = pickle.load(open(os.path.join(ROOT, "cache/paper_full_contents.pickle"), "rb"))
    keys = sorted(k for k in fulls if isinstance(fulls[k], dict)
                  and len(abstracts.get(k, "")) > 200
                  and sum(1 for s in fulls[k].values() if s.get("content")) >= 2)
    # 전체에서 고르게 20편
    step = max(1, len(keys) // N_PAPERS)
    chosen = keys[::step][:N_PAPERS]
    return [(k, abstracts[k], fulls[k]) for k in chosen]


def main():
    papers = pick_papers()
    print(f"Selected {len(papers)} papers")
    results = []
    for i, (info, abstract, full) in enumerate(papers):
        user = prepare_input(abstract, full)
        rec = {"paper_info": info, "input_tokens_prepared": len(ENC.encode(user)), "models": {}}
        print(f"\n[{i+1}/{len(papers)}] {info[:80]}")
        for model, is_r, p_in, p_cache, p_out in MODELS:
            try:
                out = call(model, is_r, user)
                parsed = out.pop("parsed")
                m = metrics(parsed)
                rec["models"][model] = {
                    "ok": True,
                    "summary": parsed.model_dump(),
                    "metrics": m,
                    "finish_reason": out["finish_reason"],
                    "refusal": out["refusal"],
                    "latency_sec": out["latency_sec"],
                    "input_tokens": out["input_tokens"],
                    "output_tokens": out["output_tokens"],
                    "reasoning_tokens": out["reasoning_tokens"],
                    "cost_usd": cost(p_in, p_out, out["input_tokens"], out["output_tokens"]),
                }
                print(f"  {model:14s} ok  sent={m['total_sentences']:2d} "
                      f"hangul={m['overall_hangul_ratio']:.2f} fin={out['finish_reason']:6s} "
                      f"out_tok={out['output_tokens']} reas={out['reasoning_tokens']} {out['latency_sec']}s")
            except Exception as e:
                rec["models"][model] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                print(f"  {model:14s} ERROR {type(e).__name__}: {str(e)[:120]}")
        results.append(rec)

    out_path = os.path.join(HERE, "eval_results.json")
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
