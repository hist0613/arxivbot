import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParseArxivRef(unittest.TestCase):
    def test_normalizes_all_forms_to_abs(self):
        from api.arxiv import parse_arxiv_ref
        cases = [
            "https://arxiv.org/abs/2501.12345",
            "https://arxiv.org/pdf/2501.12345",
            "https://arxiv.org/pdf/2501.12345.pdf",
            "https://arxiv.org/html/2501.12345",
            "https://arxiv.org/abs/2501.12345v2",
            "https://arxiv.org/pdf/2501.12345v3.pdf",
            "여기 봐주세요 2501.12345 이 논문",
        ]
        for c in cases:
            self.assertEqual(
                parse_arxiv_ref(c),
                "https://arxiv.org/abs/2501.12345",
                msg=f"failed for: {c}",
            )

    def test_old_style_id(self):
        from api.arxiv import parse_arxiv_ref
        self.assertEqual(
            parse_arxiv_ref("https://arxiv.org/abs/cs/0501001"),
            "https://arxiv.org/abs/cs/0501001",
        )

    def test_returns_none_without_arxiv_ref(self):
        from api.arxiv import parse_arxiv_ref
        self.assertIsNone(parse_arxiv_ref("hello world, no paper here"))
        self.assertIsNone(parse_arxiv_ref(""))


class TestExtractTitle(unittest.TestCase):
    def test_strips_title_descriptor(self):
        from bs4 import BeautifulSoup
        from api.arxiv import _extract_title
        html = (
            '<h1 class="title mathjax">'
            '<span class="descriptor">Title:</span>'
            'Attention Is All You Need</h1>'
        )
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(_extract_title(soup), "Attention Is All You Need")

    def test_returns_empty_when_no_title(self):
        from bs4 import BeautifulSoup
        from api.arxiv import _extract_title
        soup = BeautifulSoup("<div>nope</div>", "html.parser")
        self.assertEqual(_extract_title(soup), "")


class _FakeCache:
    def __init__(self, summarizations=None):
        self.paper_summarizations = dict(summarizations or {})
        self.updated = []

    def has_paper_summarization(self, paper_info):
        return (
            paper_info in self.paper_summarizations
            and self.paper_summarizations[paper_info] != ""
        )

    def update_paper_summarizations(self, paper_info, summarization):
        self.paper_summarizations[paper_info] = summarization
        self.updated.append(paper_info)


class _FakeAgent:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.model_name = "fake"

    def summarize(self, content):
        self.calls += 1
        return self.result


VALID_SUMMARY = json.dumps({
    "Prior Approaches": "a", "Core Contribution": "b",
    "Technical Challenges": "c", "Empirical Impact": "d",
})
OLD_SUMMARY = json.dumps({
    "What's New": "x", "Technical Details": "y", "Performance Highlights": "z",
})


class TestSummarySchemaGuard(unittest.TestCase):
    def test_accepts_current_four_section(self):
        from prompts import is_current_summary_schema
        self.assertTrue(is_current_summary_schema(VALID_SUMMARY))

    def test_rejects_old_format(self):
        from prompts import is_current_summary_schema
        self.assertFalse(is_current_summary_schema(OLD_SUMMARY))

    def test_rejects_garbage_or_empty(self):
        from prompts import is_current_summary_schema
        self.assertFalse(is_current_summary_schema(""))
        self.assertFalse(is_current_summary_schema("not json"))


class TestSummarizeOne(unittest.TestCase):
    def _service(self, agent, cache):
        from api.service import Service
        from api.agent import Encoder
        from settings import MODEL
        return Service(arxiv=None, agent=agent, encoder=Encoder(MODEL), cache=cache)

    def test_current_schema_cache_hit_skips_agent(self):
        cache = _FakeCache({"P (url)": VALID_SUMMARY})
        agent = _FakeAgent(VALID_SUMMARY)
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, VALID_SUMMARY)
        self.assertEqual(agent.calls, 0)

    def test_stale_schema_cache_is_resummarized(self):
        cache = _FakeCache({"P (url)": OLD_SUMMARY})
        agent = _FakeAgent(VALID_SUMMARY)
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, VALID_SUMMARY)        # 옛 포맷이면 재요약
        self.assertEqual(agent.calls, 1)
        self.assertEqual(cache.paper_summarizations["P (url)"], VALID_SUMMARY)  # 덮어씀

    def test_cache_miss_calls_agent_and_stores(self):
        cache = _FakeCache()
        agent = _FakeAgent(VALID_SUMMARY)
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, VALID_SUMMARY)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(cache.paper_summarizations["P (url)"], out)

    def test_empty_result_not_cached(self):
        cache = _FakeCache()
        agent = _FakeAgent("")
        svc = self._service(agent, cache)
        out = svc.summarize_one("P (url)", "abstract", "")
        self.assertEqual(out, "")
        self.assertNotIn("P (url)", cache.paper_summarizations)


class _FakeWorkspace:
    workspace = "seungtaek-lab"

    def prepare_content(self, paper_info, paper_comment, paper_summarization):
        msg = f"*{paper_info}*\n{paper_summarization}"
        return msg, msg


class _FakeService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def summarize_one(self, paper_info, abstract, full_content):
        self.calls.append(paper_info)
        return self.result


class TestResolveThreadTs(unittest.TestCase):
    def test_uses_thread_ts_when_present(self):
        from api.on_demand import resolve_thread_ts
        self.assertEqual(
            resolve_thread_ts({"ts": "1.1", "thread_ts": "9.9"}), "9.9"
        )

    def test_falls_back_to_ts(self):
        from api.on_demand import resolve_thread_ts
        self.assertEqual(resolve_thread_ts({"ts": "1.1"}), "1.1")


class TestProcessMention(unittest.TestCase):
    def _fetch_ok(self, url):
        return ("Some Title", "an abstract", "")

    def test_no_url_returns_guidance(self):
        from api.on_demand import process_mention
        result = process_mention(
            "no link here",
            cache=None,
            service=_FakeService('{"x": 1}'),
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertFalse(result["ok"])
        self.assertIn("arxiv", result["message"].lower())

    def test_ok_path_builds_message_and_meta(self):
        from api.on_demand import process_mention
        svc = _FakeService('{"Core Contribution": "c"}')
        result = process_mention(
            "see https://arxiv.org/pdf/2501.12345v2",
            cache=None,
            service=svc,
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["paper_url"], "https://arxiv.org/abs/2501.12345")
        self.assertEqual(
            result["paper_info"],
            "Some Title (https://arxiv.org/abs/2501.12345)",
        )
        self.assertIn("Some Title", result["message"])

    def test_empty_summary_returns_error(self):
        from api.on_demand import process_mention
        result = process_mention(
            "https://arxiv.org/abs/2501.12345",
            cache=None,
            service=_FakeService(""),
            workspace=_FakeWorkspace(),
            fetch_paper=self._fetch_ok,
        )
        self.assertFalse(result["ok"])


class TestListenerChannelConfig(unittest.TestCase):
    def test_active_slack_workspace_has_listener_channel_id(self):
        import settings
        active = [
            c for c in settings.WORKSPACE_CONFIGS
            if c.get("service_type") == "slack"
        ]
        self.assertTrue(active, "활성 slack 워크스페이스가 없음")
        for c in active:
            self.assertIn(
                "listener_channel_id", c,
                msg=f"listener_channel_id 누락: {c['workspace']}",
            )


class TestSettingsAppToken(unittest.TestCase):
    def test_active_slack_workspace_has_app_token_key(self):
        import settings
        active_slack = [
            c for c in settings.WORKSPACE_CONFIGS
            if c.get("service_type") == "slack"
        ]
        self.assertTrue(active_slack, "활성 slack 워크스페이스가 없음")
        for c in active_slack:
            self.assertIn("app_token", c, msg=f"app_token 누락: {c['workspace']}")


if __name__ == "__main__":
    unittest.main()
