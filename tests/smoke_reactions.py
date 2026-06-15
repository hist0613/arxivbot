"""실 API smoke: Influcoder 답글(eyes 1)을 수확해 검증. 수동 실행."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()
from slack_sdk import WebClient
from api.reactions import load_store, save_store, add_posted, harvest_reactions

CH = "C0B7V0V8U7N"


def main():
    client = WebClient(os.getenv("SLACK_TOKEN_SEUNGTAEK_LAB"))
    bot = client.auth_test()["user_id"]
    target = None
    hist = client.conversations_history(channel=CH, limit=200)
    for m in hist["messages"]:
        if m.get("reply_count", 0) > 0:
            rr = client.conversations_replies(channel=CH, ts=m["ts"], limit=200)
            for rm in rr["messages"]:
                if "Influcoder" in rm.get("text", ""):
                    target = (m["ts"], rm["ts"])
                    break
        if target:
            break
    assert target, "Influcoder 메시지를 못 찾음"
    thread_ts, reply_ts = target
    print("found:", reply_ts, "in thread", thread_ts)

    path = os.path.join(tempfile.mkdtemp(), "papers.json")
    store = load_store(path)
    add_posted(store, ts=reply_ts, thread_ts=thread_ts, channel_id=CH,
               workspace="seungtaek-lab", paper_info="Influcoder", paper_url="u",
               field="cs.CL", posted_at=datetime.now(timezone.utc).isoformat())
    n = harvest_reactions(client, store, window_days=3650, bot_user_id=bot,
                          now=datetime.now(timezone.utc))
    save_store(store, path)
    rs = store[reply_ts]["reactions"]
    print("updated:", n, "reactions:", rs)
    assert any(r["emoji"] == "eyes" for r in rs), "eyes 리액션을 못 잡음"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
