"""에이전트 루프 실 API 스모크.

Slack 없이 루프만 돌린다. 도구는 가짜로 주입해 실제 게시가 일어나지 않게 한다.
fetch_page만 진짜로 붙여 웹페이지 요약 형식을 눈으로 확인한다.

    python tests/smoke_agent.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI  # noqa: E402

from api.agent_loop import run_agent  # noqa: E402
from api.store import Store  # noqa: E402
from api.tools import build_page_fetcher, build_tools  # noqa: E402
from prompts import SYSTEM_PROMPT_AGENT  # noqa: E402
from settings import AGENT_MODEL, OPENAI_API_KEY  # noqa: E402

CASES = [
    "[지금 온 멘션]\n<@BOT> https://arxiv.org/abs/2410.05229 이거 요약해줘",
    "[지금 온 멘션]\n<@BOT> https://nvlabs.github.io/Sana/ 이 페이지 정리해줘",
    "[지금 온 멘션]\n<@BOT> attention residual이 뭔지 아주 쉽게 설명해",
]


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = Store(tmp.name)
    posted = []

    def fake_post_summary(url):
        posted.append(url)
        return {"ok": True, "title": "(가짜 게시) " + url, "url": url}

    specs, dispatch = build_tools(
        post_summary=fake_post_summary,
        fetch_page=build_page_fetcher(store),
        read_thread=lambda before_ts, limit: [],
    )

    for case in CASES:
        result = run_agent(
            client=client,
            model=AGENT_MODEL,
            system_prompt=SYSTEM_PROMPT_AGENT,
            user_input=case,
            tool_specs=specs,
            dispatch=dispatch,
        )
        print("=" * 70)
        print(case.splitlines()[-1])
        print(
            f"[도구 {result.tool_calls} / 스텝 {result.steps} / 잘림 {result.truncated}]"
        )
        print(result.text[:900] or "(빈 답 — 요약 도구가 게시를 마친 경우 정상)")

    print("=" * 70)
    print("summarize_paper가 불린 URL:", posted)
    store.close()
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()
