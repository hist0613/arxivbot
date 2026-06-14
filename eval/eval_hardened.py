"""
gpt-5.4-nano + 하드닝 프롬프트(용어 한국어화 유도 + 길이 캡) 재검증.
같은 20편으로 baseline(eval_results.json의 gpt-5.4-nano)과 비교.
"""
import json, os
import eval_gen as E

HARDENED = """Please analyze the arxiv paper and write a Korean AI-newsletter style summary in JSON.
Use exactly these four English keys, each with a Korean value of 2-3 sentences.
HARD LIMIT: the whole summary MUST NOT exceed 12 sentences total. Be concise.

언어 규칙(중요):
- 널리 통용되는 개념/표현은 한국어로 쓴다 (예: frequency distribution→주파수 분포, noise→잡음, baseline→기준선).
- 정착된 한국어 표현이 없거나 고유명사(모델명/기법명/벤치마크명)인 경우에만 영어를 쓰고, 처음 등장 시 한국어(영어) 또는 영어(한국어)로 병기한다.
- 한국어로 자연스러운데 굳이 영어를 남발하지 말 것. 문장 구조는 항상 한국어.

각 섹션 의미:
- prior_approaches: 이 논문이 다루는 문제의 기존 방법들을 분류하고 한계를 설명.
- core_contribution: 기존 한계 중 무엇을/어떤 문제를 이 논문의 기여가 해결하는지.
- technical_challenges: 그 기여 실현의 technical challenge와 해결 방법.
- empirical_impact: 기여가 어떻게 empirical하게 입증됐고 해당 분야에서 갖는 의미/impact.

답은 JSON, 키는 영어로 둔다."""

E.SYSTEM_PROMPT = HARDENED
MODEL = ("gpt-5.4-nano", True, 0.20, 0.02, 1.25)

def main():
    papers = E.pick_papers()
    base = {r["paper_info"]: r["models"]["gpt-5.4-nano"] for r in
            json.load(open(os.path.join(E.HERE, "eval_results.json"), encoding="utf-8"))}
    out = []
    print(f"{'#':>2} {'hangul b→h':>14} {'sent b→h':>10}")
    for i, (info, abstract, full) in enumerate(papers):
        user = E.prepare_input(abstract, full)
        m, is_r, pin, pc, po = MODEL
        r = E.call(m, is_r, user)
        parsed = r["parsed"]; met = E.metrics(parsed)
        b = base.get(info, {}).get("metrics", {})
        rec = {"paper_info": info, "summary": parsed.model_dump(), "metrics": met,
               "finish_reason": r["finish_reason"], "output_tokens": r["output_tokens"],
               "baseline_metrics": b}
        out.append(rec)
        print(f"{i+1:>2} {b.get('overall_hangul_ratio','?')}→{met['overall_hangul_ratio']:<6} "
              f"{b.get('total_sentences','?')}→{met['total_sentences']:<4} fin={r['finish_reason']}")
    json.dump(out, open(os.path.join(E.HERE, "eval_hardened.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    import statistics as st
    print("\n=== 평균 ===")
    print("hangul baseline:", round(st.mean([r['baseline_metrics']['overall_hangul_ratio'] for r in out]),3),
          "→ hardened:", round(st.mean([r['metrics']['overall_hangul_ratio'] for r in out]),3))
    print("sentences baseline:", round(st.mean([r['baseline_metrics']['total_sentences'] for r in out]),1),
          "→ hardened:", round(st.mean([r['metrics']['total_sentences'] for r in out]),1))
    print("over-12 hardened:", sum(1 for r in out if r['metrics']['total_sentences']>12))

if __name__ == "__main__":
    main()
