"""On-demand(@멘션) 논문 요약 Socket Mode 리스너.

부팅 시 자동 실행(Task Scheduler) + 죽으면 재시작 전제로 상시 동작한다.
이벤트 처리 로직은 api.on_demand의 테스트된 코어를 사용한다.
"""
from datetime import datetime, timezone

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.reactions import load_store, save_store, add_posted
from api.on_demand import process_mention, resolve_thread_ts
from api.resolvers import build_resolver
from api.logger import logger
from settings import WORKSPACE_CONFIGS, MODEL


STAGE = {
    "fetching": "🔄 논문 페이지 가져오는 중…",
    "downloading": "🔄 PDF 다운로드 중…",
    "summarizing": "🔄 AI가 요약하는 중…",
}


def make_app(workspace_config: dict):
    workspace = Workspace(workspace_config)
    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(
        arxiv_client, AutoAgent.from_model_name(MODEL), Encoder(MODEL), cache
    )
    resolve = build_resolver(arxiv_client, cache)
    # on-demand 멘션을 받을 채널 (배치 게시 채널 allowed_channel_id와 분리)
    listener_channel_id = workspace_config["listener_channel_id"]
    app = App(token=workspace_config["slack_token"])

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
        logger.info(f"app_mention in {channel}: {event.get('text')!r}")
        try:
            # 진행 상황을 메시지 1개를 편집하며 표시(새 메시지 X → 알림 스팸 방지)
            posted = client.chat_postMessage(
                channel=channel, text=STAGE["fetching"], thread_ts=thread_ts
            )
            ts = posted["ts"]
            last = {"text": STAGE["fetching"]}

            def on_progress(stage):
                msg = STAGE.get(stage)
                if msg and msg != last["text"]:
                    last["text"] = msg
                    client.chat_update(channel=channel, ts=ts, text=msg)

            result = process_mention(
                event.get("text", ""),
                cache=cache,
                service=service,
                workspace=workspace,
                resolve=resolve,
                on_progress=on_progress,
            )
            client.chat_update(channel=channel, ts=ts, text=result["message"])
            if result["ok"]:
                store = load_store()
                add_posted(
                    store,
                    ts=ts,
                    thread_ts=thread_ts,
                    channel_id=channel,
                    workspace=workspace.workspace,
                    paper_info=result["paper_info"],
                    paper_url=result["paper_url"],
                    field="on-demand",
                    posted_at=datetime.now(timezone.utc).isoformat(),
                )
                save_store(store)
                logger.info(f"on-demand summary posted: {result['paper_info']}")
        except Exception as e:
            logger.error(f"app_mention handler error: {e}")
            try:
                client.chat_postMessage(
                    channel=channel,
                    text=f"처리 중 오류가 났어요: {e}",
                    thread_ts=thread_ts,
                )
            except Exception:
                pass

    return workspace, app


def run():
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
