"""On-demand(@멘션) Socket Mode 리스너.

멘션을 받으면 스레드 문맥을 토큰 예산 안에서 조립해 에이전트 루프를 돌린다.
논문 요약은 게시까지 도구가 끝내므로(api.tools.post_paper_summary) 모델은
무엇을 할지만 정한다. 에이전트가 실패하면 예전의 결정론적 요약 경로로
폴백하기 때문에 매일 쓰는 기능이 모델 사정으로 죽지 않는다.

부팅 시 자동 실행(Task Scheduler) + 죽으면 재시작 전제로 상시 동작한다.
"""
import os
from datetime import datetime, timezone

from openai import OpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from api.agent import AutoAgent, Encoder
from api.agent_loop import run_agent
from api.arxiv import ArxivClient
from api.cache import CacheManager
from api.context import build_context, load_digest, render_context, update_digest
from api.logger import logger
from api.on_demand import NO_URL_MSG, extract_targets, process_url, resolve_thread_ts
from api.reactions import add_posted, load_store, save_store
from api.resolvers import build_resolver
from api.service import Service
from api.tools import (
    build_page_fetcher,
    build_tools,
    collect_allowed_urls,
    match_allowed_url,
    post_paper_summary,
)
from api.workspace import Workspace, markdown_links_to_slack
from prompts import SYSTEM_PROMPT_AGENT, THREAD_DIGEST_PROMPT
from settings import (
    AGENT_DEADLINE_SEC,
    AGENT_MAX_STEPS,
    AGENT_MODEL,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_TOKEN_BUDGET,
    MODEL,
    OPENAI_API_KEY,
    WORKSPACE_CONFIGS,
)


PID_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "listener.pid"
)

THINKING = "🔄 생각하는 중…"
# web_search는 서버가 직접 실행해 함수 호출로 오지 않으므로 여기 없다.
# 검색 중에는 "생각하는 중…"이 그대로 떠 있다.
AGENT_STAGE = {
    "fetch_page": "🔄 페이지 읽는 중…",
    "read_thread": "🔄 이전 대화 확인하는 중…",
}
TRUNCATED_NOTE = "\n\n(시간이 걸려 도중에 끊었습니다. 더 필요하면 다시 불러 주세요.)"
# 아주 긴 스레드에서도 최근 대화까지는 따라가되, 무한정 호출하지는 않는다.
REPLIES_MAX_PAGES = 10


def stage_for_tool(name: str):
    """summarize_paper는 자기 답글에서 따로 진행 상황을 보여주므로 None."""
    return AGENT_STAGE.get(name)


def handle_mention_core(*, user_input, run, fallback, posted_count):
    """에이전트를 돌리고, 실패하면 결정론 경로로 넘긴다.

    posted_count()는 이번 멘션에서 실제로 스레드에 올라간 요약 수다. 도구를
    불렀는지가 아니라 이걸 봐야 한다. 도구를 불렀어도 URL 가드에 걸려 아무것도
    안 올라갔을 수 있고, 그때 빈 답을 정상으로 넘기면 사용자는 아무 응답도
    못 받는다.

    반환: {"mode": "agent"|"fallback", "text": str}
    """
    try:
        result = run(user_input)
    except Exception as e:
        logger.error(f"agent 실패, 결정론 경로로 폴백: {e}")
        return fallback()

    answer = (result.text or "").strip()
    if not answer:
        # 빈 답이 나오는 경우는 둘인데 뜻이 정반대다.
        # (1) summarize_paper가 이미 게시를 마쳐 덧붙일 말이 없다 -> 정상
        # (2) 상한에 걸렸거나 가드에 막혀 아무것도 못 올렸다
        #     -> 사용자는 답을 못 받으므로 예전 경로로라도 요약을 준다
        if posted_count() > 0:
            return {"mode": "agent", "text": ""}
        logger.info("agent가 빈 답을 줬고 게시된 것도 없다. 결정론 경로로 폴백")
        return fallback()
    if getattr(result, "truncated", False):
        answer += TRUNCATED_NOTE
    return {"mode": "agent", "text": answer}


def make_app(workspace_config: dict):
    workspace = Workspace(workspace_config)
    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(
        arxiv_client, AutoAgent.from_model_name(MODEL), Encoder(MODEL), cache
    )
    resolve = build_resolver(arxiv_client, cache)
    encoder = Encoder(MODEL)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    fetch_page = build_page_fetcher(cache.store)
    # on-demand 멘션을 받을 채널 (배치 게시 채널 allowed_channel_id와 분리)
    listener_channel_id = workspace_config["listener_channel_id"]
    app = App(token=workspace_config["slack_token"])
    # 첫 멘션 때 한 번만 auth_test로 채운다. 기동 시점에 부르면 Slack이
    # 잠깐 죽어 있을 때 리스너가 아예 안 뜬다.
    bot_user_id = {"value": None}

    def count_tokens(text):
        # 특수 토큰을 만나도 예외를 던지지 않게 한다. 사용자가 붙여넣은
        # 텍스트에 <|endoftext|> 같은 문자열이 섞일 수 있다.
        return len(encoder.encoding.encode(text, disallowed_special=()))

    def summarize_digest(previous, texts):
        joined = "\n".join(texts)[:8000]
        prompt = (
            f"{THREAD_DIGEST_PROMPT}\n\n[이전 요지]\n{previous or '(없음)'}\n\n"
            f"[새로 밀려난 대화]\n{joined}"
        )
        try:
            r = openai_client.responses.create(model=AGENT_MODEL, input=prompt)
            return (getattr(r, "output_text", "") or previous).strip()
        except Exception as e:
            logger.error(f"스레드 요지 생성 실패: {e}")
            return previous

    def read_replies(client, channel, thread_ts, exclude_ts):
        """스레드 원문을 읽는다. 지금 온 멘션은 따로 넣으므로 빼둔다.

        conversations_replies는 스레드 처음부터 돌려주므로, 긴 스레드에서
        한 번만 부르면 정작 지금 얘기 중인 최근 대화를 못 본다. 커서를 따라
        끝까지 간 뒤 뒤에서 CONTEXT_MAX_MESSAGES개만 남긴다.

        channels:history 스코프가 없으면 여기서 막힌다. 그때는 문맥 없이
        단발로 답하되(빈 목록), 왜 그런지 로그에는 남긴다.
        """
        messages, cursor = [], None
        try:
            for _ in range(REPLIES_MAX_PAGES):
                r = client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=CONTEXT_MAX_MESSAGES,
                    **({"cursor": cursor} if cursor else {}),
                )
                messages += r.get("messages", [])
                cursor = (r.get("response_metadata") or {}).get("next_cursor")
                if not r.get("has_more") or not cursor:
                    break
        except Exception as e:
            logger.error(f"스레드 읽기 실패({thread_ts}): {e}")
            return messages[-CONTEXT_MAX_MESSAGES:]
        recent = [m for m in messages if m.get("ts") != exclude_ts]
        return recent[-CONTEXT_MAX_MESSAGES:]

    def register_posted(client, channel, thread_ts, ts, paper_info, paper_url):
        store = load_store()
        add_posted(
            store,
            ts=ts,
            thread_ts=thread_ts,
            channel_id=channel,
            workspace=workspace.workspace,
            paper_info=paper_info,
            paper_url=paper_url,
            field="on-demand",
            posted_at=datetime.now(timezone.utc).isoformat(),
        )
        save_store(store)

    def summarize_and_post(client, channel, thread_ts, url, prefix=""):
        return post_paper_summary(
            client,
            channel=channel,
            thread_ts=thread_ts,
            url=url,
            prefix=prefix,
            process=lambda u, on_progress: process_url(
                u,
                cache=cache,
                service=service,
                workspace=workspace,
                resolve=resolve,
                on_progress=on_progress,
            ),
            on_posted=lambda ts, paper_info, paper_url: register_posted(
                client, channel, thread_ts, ts, paper_info, paper_url
            ),
        )

    def deterministic_fallback(client, channel, thread_ts, text, post, already_posted):
        """에이전트를 못 쓸 때의 예전 경로 — 링크마다 답글 1개.

        OpenAI가 죽었거나 모델이 빈 답을 줘도 매일 쓰는 논문 요약만은
        되게 하는 안전망이다. 에이전트 도입 전 동작과 같다.

        already_posted: 이번 멘션에서 에이전트가 이미 올린 URL. 요약을 올린
        뒤에 다음 API 호출이 죽어 여기로 떨어지는 경우가 있는데, 그때 같은
        논문을 다시 올리면 스레드에 요약이 두 번 뜨고 리액션 store에도 두 번
        기록된다.
        """
        targets = [
            url
            for url in extract_targets(text)
            if match_allowed_url(url, already_posted) is None
        ]
        if not targets:
            # 이미 다 올렸으면 조용히 끝낸다. 하나도 없었으면 안내를 남긴다.
            return {"mode": "fallback", "text": "" if already_posted else NO_URL_MSG}
        total = len(targets)
        for i, url in enumerate(targets, 1):
            prefix = f"({i}/{total}) " if total > 1 else ""
            post(url, prefix=prefix)
        return {"mode": "fallback", "text": ""}

    @app.event("app_mention")
    def handle_app_mention(event, client):
        channel = event.get("channel")
        # 지정 채널 밖 멘션은 무시. 조용히 버리면 디버깅이 불가능하므로 로그를 남긴다.
        if channel != listener_channel_id:
            logger.info(
                f"app_mention ignored: channel {channel} "
                f"!= listener channel {listener_channel_id}"
            )
            return
        thread_ts = resolve_thread_ts(event)
        text = event.get("text", "")
        logger.info(f"app_mention in {channel}: {text!r}")

        if bot_user_id["value"] is None:
            try:
                bot_user_id["value"] = client.auth_test()["user_id"]
            except Exception as e:
                logger.error(f"auth_test 실패: {e}")

        ts = None  # 진행 표시 답글. except 절에서도 봐야 한다
        try:
            messages = read_replies(client, channel, thread_ts, event.get("ts"))
            context = build_context(
                messages,
                bot_user_id=bot_user_id["value"],
                count_tokens=count_tokens,
                budget=CONTEXT_TOKEN_BUDGET,
                digest=load_digest(cache.store, thread_ts),
            )
            if context.folded:
                context = context._replace(
                    digest=update_digest(
                        cache.store, thread_ts, context.folded, summarize_digest
                    )
                )
            user_input = render_context(context, mention_text=text)

            placeholder = client.chat_postMessage(
                channel=channel, text=THINKING, thread_ts=thread_ts
            )
            ts = placeholder["ts"]
            last = {"text": THINKING}

            # 이번 멘션에서 실제로 스레드에 올라간 요약. 폴백 중복 방지와
            # "정말 아무것도 못 올렸는지" 판정에 쓴다.
            posted_urls = []

            def post_and_track(url, prefix=""):
                result = summarize_and_post(
                    client, channel, thread_ts, url, prefix=prefix
                )
                if result.get("ok"):
                    posted_urls.append(result.get("url") or url)
                return result

            def on_step(tool_name):
                stage = stage_for_tool(tool_name)
                if stage and stage != last["text"]:
                    last["text"] = stage
                    client.chat_update(channel=channel, ts=ts, text=stage)

            def read_thread(before_ts, limit=10):
                # 이미 받아둔 messages에서 꺼낸다. 접힌 건 예산 때문이지
                # 안 읽어서가 아니므로 Slack을 다시 부를 이유가 없다.
                older = [m for m in messages if m.get("ts", "") < str(before_ts)]
                return [
                    {
                        "ts": m.get("ts"),
                        "user": m.get("user"),
                        "text": m.get("text", ""),
                    }
                    for m in older[-int(limit or 10):]
                ]

            # 요약 대상은 대화에 실제로 등장한 링크로 제한한다. 스레드 것까지
            # 허용해야 "아까 그 논문 요약해줘"가 된다.
            allowed_urls = collect_allowed_urls(
                text, *[m.get("text", "") for m in messages]
            )

            def guarded_post_summary(url):
                target = match_allowed_url(url, allowed_urls)
                if target is None:
                    logger.info(f"대화에 없는 URL이라 거절: {url!r}")
                    return {
                        "ok": False,
                        "url": url,
                        "error": (
                            "대화에 없는 URL이다. 멘션이나 이 스레드에 실제로 "
                            "올라온 링크만 요약할 수 있다. 지금 쓸 수 있는 링크: "
                            + (", ".join(allowed_urls[:5]) or "없음")
                        ),
                    }
                return post_and_track(target)

            tool_specs, dispatch = build_tools(
                post_summary=guarded_post_summary,
                fetch_page=fetch_page,
                read_thread=read_thread,
            )

            out = handle_mention_core(
                user_input=user_input,
                run=lambda prompt: run_agent(
                    client=openai_client,
                    model=AGENT_MODEL,
                    system_prompt=SYSTEM_PROMPT_AGENT,
                    user_input=prompt,
                    tool_specs=tool_specs,
                    dispatch=dispatch,
                    max_steps=AGENT_MAX_STEPS,
                    deadline_sec=AGENT_DEADLINE_SEC,
                    on_step=on_step,
                ),
                fallback=lambda: deterministic_fallback(
                    client, channel, thread_ts, text, post_and_track, posted_urls
                ),
                posted_count=lambda: len(posted_urls),
            )

            answer = out["text"]
            if not answer:
                # 요약 도구가 자기 답글을 이미 올렸다. 진행 표시만 남으면
                # 스레드에 "생각하는 중…"이 영영 떠 있으므로 치운다.
                # 봇이 올린 메시지라 chat:write로 지워지지만, 워크스페이스
                # 정책으로 막히는 경우가 있어 편집으로 폴백한다.
                try:
                    client.chat_delete(channel=channel, ts=ts)
                except Exception:
                    client.chat_update(
                        channel=channel, ts=ts, text="요약을 위에 올렸습니다."
                    )
                return
            blocks = workspace.prepare_text_blocks(answer)
            # 블록을 못 만들어 평문으로 갈 때도 링크는 Slack 문법이어야 한다.
            answer = markdown_links_to_slack(answer)
            try:
                client.chat_update(
                    channel=channel,
                    ts=ts,
                    text=answer,
                    **({"blocks": blocks} if blocks else {}),
                )
            except Exception as e:
                if not blocks:
                    raise
                logger.error(f"chat_update with blocks failed ({e}); text로 폴백")
                client.chat_update(channel=channel, ts=ts, text=answer)
            logger.info(f"agent 답변 게시({out['mode']}): {answer[:80]!r}")
        except Exception as e:
            logger.error(f"app_mention handler error: {e}")
            # 진행 표시 답글을 이미 올렸으면 그걸 오류 문구로 바꾼다. 새
            # 메시지를 올리면 "생각하는 중…"이 스레드에 영영 남는다.
            try:
                if ts:
                    client.chat_update(
                        channel=channel, ts=ts, text=f"처리 중 오류가 났어요: {e}"
                    )
                else:
                    client.chat_postMessage(
                        channel=channel,
                        text=f"처리 중 오류가 났어요: {e}",
                        thread_ts=thread_ts,
                    )
            except Exception:
                pass

    return workspace, app


def write_pid_file():
    """자기 PID를 기록한다.

    Stop-ScheduledTask는 래퍼 powershell만 끝내고 이 python 자식은 고아로
    남긴다. 그 상태로 다음 인스턴스가 뜨면 Socket Mode 연결이 둘이 되어
    멘션이 옛 프로세스로 배정되기도 한다. run_listener.ps1이 시작 전에
    이 파일을 읽어 남아 있는 프로세스를 정리한다.
    """
    try:
        os.makedirs(os.path.dirname(PID_PATH), exist_ok=True)
        with open(PID_PATH, "w") as fp:
            fp.write(str(os.getpid()))
    except OSError as e:
        logger.error(f"PID 파일 기록 실패({PID_PATH}): {e}")


def run():
    write_pid_file()
    handlers = []
    for cfg in WORKSPACE_CONFIGS:
        if cfg.get("service_type") != "slack":
            continue
        workspace, app = make_app(cfg)
        handlers.append(SocketModeHandler(app, cfg["app_token"]))
        logger.info(f"Listener ready for {workspace.workspace_name}")

    if not handlers:
        logger.error("No slack workspace configured for listener.")
        return
    # 여러 워크스페이스: 마지막만 foreground로 blocking, 나머지는 background 연결
    for h in handlers[:-1]:
        h.connect()
    handlers[-1].start()  # blocks forever (자체 재연결 포함)


if __name__ == "__main__":
    run()
