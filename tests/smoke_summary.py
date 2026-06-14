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
