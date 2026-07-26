import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeClient:
    def __init__(self, update_fails_on_blocks=False):
        self.posted, self.updated = [], []
        self.update_fails_on_blocks = update_fails_on_blocks

    def chat_postMessage(self, **kw):
        self.posted.append(kw)
        return {"ts": f"ts{len(self.posted)}"}

    def chat_update(self, **kw):
        if self.update_fails_on_blocks and "blocks" in kw:
            raise RuntimeError("invalid_blocks")
        self.updated.append(kw)
        return {"ok": True}


SUMMARY_RESULT = {
    "ok": True,
    "message": "제목 (https://arxiv.org/abs/1)\n- Prior Approaches: 내용",
    "blocks": [{"type": "section"}],
    "paper_info": "제목 (https://arxiv.org/abs/1)",
    "paper_url": "https://arxiv.org/abs/1",
}


class TestPostPaperSummary(unittest.TestCase):
    def test_posts_summary_and_returns_receipt(self):
        from api import tools

        client = FakeClient()
        registered = []
        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://arxiv.org/abs/1",
            prefix="",
            process=lambda url, on_progress: SUMMARY_RESULT,
            on_posted=lambda **kw: registered.append(kw),
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["title"], "제목 (https://arxiv.org/abs/1)")
        self.assertEqual(len(client.posted), 1)  # 답글 1개를 만들고
        self.assertIn("blocks", client.updated[-1])  # 그걸 결과로 편집한다
        self.assertEqual(registered[0]["ts"], "ts1")  # 리액션 store 등록까지

    def test_progress_edits_the_same_message(self):
        from api import tools

        client = FakeClient()

        def process(url, on_progress):
            on_progress("fetching")
            on_progress("downloading")
            on_progress("summarizing")
            return SUMMARY_RESULT

        tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://arxiv.org/abs/1",
            prefix="(1/2) ",
            process=process,
            on_posted=lambda **kw: None,
        )
        self.assertEqual(len(client.posted), 1)
        self.assertTrue(all(u["ts"] == "ts1" for u in client.updated))
        self.assertTrue(client.updated[0]["text"].startswith("(1/2) "))

    def test_blocks_rejection_falls_back_to_text(self):
        from api import tools

        client = FakeClient(update_fails_on_blocks=True)
        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://arxiv.org/abs/1",
            prefix="",
            process=lambda url, on_progress: SUMMARY_RESULT,
            on_posted=lambda **kw: None,
        )
        self.assertTrue(result["ok"])
        self.assertNotIn("blocks", client.updated[-1])
        self.assertIn("Prior Approaches", client.updated[-1]["text"])

    def test_failure_is_reported_in_place(self):
        from api import tools

        client = FakeClient()
        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://x.test",
            prefix="",
            process=lambda url, on_progress: {
                "ok": False,
                "message": "가져오지 못했어요",
                "blocks": None,
            },
            on_posted=lambda **kw: None,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(len(client.posted), 1)
        self.assertIn("가져오지 못했어요", client.updated[-1]["text"])

    def test_exception_becomes_error_message(self):
        from api import tools

        client = FakeClient()

        def boom(url, on_progress):
            raise RuntimeError("PDF 서버 죽음")

        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://x.test",
            prefix="",
            process=boom,
            on_posted=lambda **kw: None,
        )
        self.assertFalse(result["ok"])
        self.assertIn("PDF 서버 죽음", client.updated[-1]["text"])


def build(**overrides):
    from api import tools

    kwargs = {
        "post_summary": lambda url: {"ok": True, "title": "GSM-Symbolic", "url": url},
        "fetch_page": lambda url: {"title": "Sana", "text": "본문"},
        "read_thread": lambda before_ts, limit: [{"ts": "1", "text": "옛 메시지"}],
    }
    kwargs.update(overrides)
    return tools.build_tools(**kwargs)


class TestDispatch(unittest.TestCase):
    def test_tool_set(self):
        specs, _ = build()
        names = {s["name"] for s in specs if s.get("type") == "function"}
        self.assertEqual(names, {"summarize_paper", "fetch_page", "read_thread"})
        self.assertIn("web_search", {s.get("type") for s in specs})

    def test_summarize_paper_returns_receipt_not_summary(self):
        _, dispatch = build(
            post_summary=lambda url: {
                "ok": True,
                "title": "GSM-Symbolic",
                "url": url,
                "message": "- Prior Approaches: 긴 요약 본문",
            }
        )
        out = json.loads(dispatch("summarize_paper", {"url": "https://arxiv.org/abs/1"}))
        self.assertTrue(out["posted"])
        self.assertEqual(out["title"], "GSM-Symbolic")
        self.assertNotIn("Prior Approaches", json.dumps(out, ensure_ascii=False))

    def test_failed_summary_is_reported_to_model(self):
        _, dispatch = build(
            post_summary=lambda url: {"ok": False, "url": url, "error": "미지원 링크"}
        )
        out = json.loads(dispatch("summarize_paper", {"url": "https://x.test"}))
        self.assertFalse(out["posted"])
        self.assertIn("미지원", out["error"])

    def test_unknown_tool_returns_error_json(self):
        _, dispatch = build()
        out = json.loads(dispatch("없는도구", {}))
        self.assertIn("error", out)

    def test_tool_exception_returns_error_json(self):
        def boom(url):
            raise RuntimeError("망함")

        _, dispatch = build(fetch_page=boom)
        out = json.loads(dispatch("fetch_page", {"url": "https://x.test"}))
        self.assertIn("망함", out["error"])

    def test_missing_argument_returns_error_json(self):
        _, dispatch = build()
        out = json.loads(dispatch("fetch_page", {}))
        self.assertIn("error", out)


class TestUrlGuard(unittest.TestCase):
    """모델이 URL을 지어내거나 잘못 옮겨 적으면 엉뚱한 논문이 올라간다."""

    def test_collects_urls_from_mention_and_thread(self):
        from api.tools import collect_allowed_urls

        urls = collect_allowed_urls(
            "<@B> 이거 봐 https://arxiv.org/abs/2410.05229",
            "앞서 https://nvlabs.github.io/Sana/ 얘기했었지",
        )
        self.assertEqual(
            urls, ["https://arxiv.org/abs/2410.05229", "https://nvlabs.github.io/Sana/"]
        )

    def test_same_paper_in_abs_and_pdf_is_one_entry(self):
        from api.tools import collect_allowed_urls

        urls = collect_allowed_urls(
            "https://arxiv.org/abs/2410.05229 https://arxiv.org/pdf/2410.05229v2"
        )
        self.assertEqual(len(urls), 1)

    def test_matches_arxiv_across_url_forms(self):
        from api.tools import match_allowed_url

        allowed = ["https://arxiv.org/abs/2410.05229"]
        for asked in [
            "https://arxiv.org/pdf/2410.05229",
            "https://arxiv.org/abs/2410.05229v3",
            "2410.05229",
        ]:
            # 대조에 걸리면 모델이 쓴 문자열이 아니라 대화의 원문을 쓴다
            self.assertEqual(match_allowed_url(asked, allowed), allowed[0])

    def test_matches_ignoring_trailing_slash(self):
        from api.tools import match_allowed_url

        allowed = ["https://nvlabs.github.io/Sana"]
        self.assertEqual(
            match_allowed_url("https://nvlabs.github.io/Sana/", allowed), allowed[0]
        )

    def test_url_not_in_conversation_is_rejected(self):
        from api.tools import match_allowed_url

        allowed = ["https://arxiv.org/abs/2410.05229"]
        self.assertIsNone(match_allowed_url("https://arxiv.org/abs/1706.03762", allowed))
        self.assertIsNone(match_allowed_url("", allowed))
        self.assertIsNone(match_allowed_url("https://a.test", []))


class TestFetchPageText(unittest.TestCase):
    def setUp(self):
        from api.store import Store

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_cache_hit_skips_download(self):
        from api.tools import build_page_fetcher

        calls = []

        def download(url):
            calls.append(url)
            return "Sana Video2", "본문 텍스트"

        fetch = build_page_fetcher(self.store, download=download)
        first = fetch("https://x.test")
        second = fetch("https://x.test")
        self.assertEqual(first["title"], "Sana Video2")
        self.assertEqual(second["text"], "본문 텍스트")
        self.assertEqual(len(calls), 1)

    def test_paper_host_is_redirected_to_summarize_paper(self):
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(self.store, download=lambda url: ("", ""))
        out = fetch("https://arxiv.org/abs/2410.05229")
        self.assertIn("summarize_paper", out.get("hint", ""))

    def test_empty_page_is_reported(self):
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(self.store, download=lambda url: ("", ""))
        out = fetch("https://x.test")
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
