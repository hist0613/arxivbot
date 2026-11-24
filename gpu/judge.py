"""요약 채점: eval/eval_judge.py 의 100점 루브릭을 그대로 쓰되, faithfulness 를 초록이 아니라 전문에 대어 본다.

모델 이름은 판정 모델에 보이지 않는다(요약 하나씩 채점).
사용: python gpu/judge.py <papers.jsonl> <bench50.json> <out.json> <results.jsonl>...
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from eval_judge import RUBRIC, JudgeScore, JUDGE_MODEL, JUDGE_REASONING  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_TEXT_CHARS = 60_000
RUBRIC_FULL = RUBRIC.replace("원문(abstract)과", "원문(본문 발췌)과")


def judge(source: str, summary: dict) -> dict:
    user = (f"[원문 본문 발췌]\n{source[:JUDGE_TEXT_CHARS]}\n\n"
            f"[평가 대상 요약 (JSON)]\n{json.dumps(summary, ensure_ascii=False, indent=2)}")
    for trial in range(3):
        try:
            resp = client.beta.chat.completions.parse(
                model=JUDGE_MODEL, reasoning_effort=JUDGE_REASONING, max_completion_tokens=3000,
                messages=[{"role": "system", "content": RUBRIC_FULL}, {"role": "user", "content": user}],
                response_format=JudgeScore)
            d = resp.choices[0].message.parsed.model_dump()
            d["total"] = sum(d[k] for k in ["format_schema", "section_fidelity", "korean_fluency",
                                            "codeswitch_appropriateness", "length_conciseness", "faithfulness"])
            return d
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (trial + 1))
    return {"error": err}


def main(papers_path, bench_path, out_path, *result_paths):
    papers = {json.loads(l)["arxiv_id"]: json.loads(l) for l in open(papers_path)}
    bench = {p["arxiv_id"]: p for p in json.load(open(bench_path))}
    jobs = []
    for aid, b in bench.items():
        s = json.loads(b["nano_summary"])
        s = s[0] if isinstance(s, list) else s
        jobs.append((aid, "gpt-5.4-nano", {k.lower().replace(" ", "_"): v for k, v in s.items()}))
    for path in result_paths:
        for line in open(path):
            r = json.loads(line)
            if r.get("summary"):
                jobs.append((r["arxiv_id"], r["model"], r["summary"]))
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    seen = {(d["arxiv_id"], d["model"]) for d in done if "error" not in d["judge"]}
    jobs = [j for j in jobs if (j[0], j[1]) not in seen]
    done = [d for d in done if "error" not in d["judge"]]
    print(f"채점 {len(jobs)}건", flush=True)

    def run(job):
        aid, model, summary = job
        p = papers[aid]
        source = p.get("text") or f"Abstract: {p['abstract']}"
        return {"arxiv_id": aid, "model": model, "judge": judge(source, summary)}

    with ThreadPoolExecutor(6) as ex:
        for i, rec in enumerate(ex.map(run, jobs), 1):
            done.append(rec)
            print(i, rec["arxiv_id"], rec["model"], rec["judge"].get("total", rec["judge"].get("error")), flush=True)
            json.dump(done, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
