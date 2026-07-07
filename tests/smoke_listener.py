"""실제 API로 process_mention 1건 검증(Slack 게시 없음). 수동 실행(비용 발생).

사용: python tests\\smoke_listener.py https://arxiv.org/abs/<id>
인자 없으면 기본 URL 사용.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.arxiv import ArxivClient
from api.agent import AutoAgent, Encoder
from api.cache import CacheManager
from api.service import Service
from api.workspace import Workspace
from api.on_demand import process_mention
from api.resolvers import build_resolver
from settings import WORKSPACE_CONFIGS, MODEL

EXPECTED_KEYS = ["Prior Approaches", "Core Contribution",
                 "Technical Challenges", "Empirical Impact"]


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://arxiv.org/abs/1706.03762"
    cfg = next(c for c in WORKSPACE_CONFIGS if c.get("service_type") == "slack")
    workspace = Workspace(cfg)
    cache = CacheManager()
    arxiv_client = ArxivClient(cache)
    service = Service(arxiv_client, AutoAgent.from_model_name(MODEL),
                      Encoder(MODEL), cache)
    resolve = build_resolver(arxiv_client, cache)

    result = process_mention(
        f"@arxivbot {url}",
        cache=cache, service=service, workspace=workspace,
        resolve=resolve,
    )
    print("ok:", result["ok"])
    print("paper_info:", result["paper_info"])
    print("message:\n", result["message"][:1200])
    assert result["ok"], result["message"]
    assert result["paper_url"].startswith("https://arxiv.org/abs/")
    # 메시지에 4섹션 키가 모두 들어있는지
    for k in EXPECTED_KEYS:
        assert k in result["message"], f"섹션 누락: {k}"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
