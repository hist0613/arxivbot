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

    def test_existing_message_is_reused(self):
        """리스너가 올려둔 "생각하는 중…"을 이어서 쓴다. 새로 올리면 진행
        표시가 둘로 늘었다 하나가 사라진다."""
        from api import tools

        client = FakeClient()
        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://arxiv.org/abs/1",
            prefix="",
            process=lambda url, on_progress: SUMMARY_RESULT,
            on_posted=lambda **kw: None,
            existing_ts="ts-생각중",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(client.posted, [])  # 새 답글을 만들지 않고
        self.assertTrue(all(u["ts"] == "ts-생각중" for u in client.updated))

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

    def test_bookkeeping_failure_still_reports_success(self):
        """리액션 등록이 실패해도 요약은 이미 올라갔다. 실패로 알리면 두 번 올라간다."""
        from api import tools

        client = FakeClient()

        def boom(**kw):
            raise OSError("papers.json 잠김")

        result = tools.post_paper_summary(
            client,
            channel="C1",
            thread_ts="1.1",
            url="https://arxiv.org/abs/1",
            prefix="",
            process=lambda url, on_progress: SUMMARY_RESULT,
            on_posted=boom,
        )
        self.assertTrue(result["ok"])

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

    def test_keyerror_from_inside_tool_is_not_reported_as_missing_argument(self):
        """도구 안쪽 KeyError를 "빠진 인자"로 알리면 모델이 같은 인자로 재호출한다."""

        def broken(url):
            raise KeyError("ts")

        _, dispatch = build(post_summary=broken)
        out = json.loads(dispatch("summarize_paper", {"url": "https://arxiv.org/abs/1"}))
        self.assertIn("error", out)
        self.assertNotIn("인자", out["error"])


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


def _allow(url):
    """단위 테스트용 주소 검사기. 가짜 도메인에 DNS를 쏘지 않는다."""
    return None


# 본문 품질 검사(page_problem)를 통과할 만큼 긴 가짜 본문. 캐시 동작을 보는
# 테스트가 본문 길이에 걸리지 않게 한다.
LONG_BODY = "가져온 본문 텍스트다. " * 30


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
            return "Sana Video2", LONG_BODY

        fetch = build_page_fetcher(self.store, download=download, check=_allow)
        first = fetch("https://x.test")
        second = fetch("https://x.test")
        self.assertEqual(first["title"], "Sana Video2")
        self.assertEqual(second["text"], LONG_BODY)
        self.assertEqual(len(calls), 1)

    def test_paper_host_is_redirected_to_summarize_paper(self):
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(
            self.store, download=lambda url: ("", ""), check=_allow
        )
        out = fetch("https://arxiv.org/abs/2410.05229")
        self.assertIn("summarize_paper", out.get("hint", ""))

    def test_empty_page_is_reported(self):
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(
            self.store, download=lambda url: ("", ""), check=_allow
        )
        out = fetch("https://x.test")
        self.assertIn("error", out)

    def test_javascript_shell_is_reported_as_failure(self):
        """x.com은 HTTP 200에 자바스크립트 껍데기를 준다. 여기서 뽑히는 건
        차단 안내문 170자뿐인데, 이걸 본문으로 넘기면 모델이 그대로 옮긴다."""
        from api.tools import build_page_fetcher

        shell = (
            "Something went wrong, but don't fret — let's give it another shot. "
            "Try again Some privacy related extensions may cause issues on x.com. "
            "Please disable them and try again."
        )
        fetch = build_page_fetcher(
            self.store, download=lambda url: ("", shell), check=_allow
        )
        out = fetch("https://x.com/omarsar0/article/2065880971031834786")
        self.assertIn("error", out)
        self.assertNotIn("text", out)

    def test_failure_tells_the_model_to_search_instead(self):
        """도구가 성공으로 보고하면 모델은 web_search 폴백을 하지 않는다."""
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(
            self.store, download=lambda url: ("", "Just a moment..."), check=_allow
        )
        out = fetch("https://blocked.test/post")
        self.assertIn("web_search", out.get("hint", ""))

    def test_blocked_page_is_not_cached(self):
        """차단 안내문이 캐시에 들어가면 7일간 같은 오답이 즉시 나온다."""
        from api.tools import build_page_fetcher

        calls = []

        def download(url):
            calls.append(url)
            return "", "Attention Required! | Cloudflare Please enable cookies."

        fetch = build_page_fetcher(self.store, download=download, check=_allow)
        fetch("https://blocked.test/post")
        fetch("https://blocked.test/post")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(self.store.get_page("https://blocked.test/post"))

    def test_login_wall_is_reported_as_failure(self):
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(
            self.store,
            download=lambda url: ("Medium", "Sign in to continue reading this story."),
            check=_allow,
        )
        self.assertIn("error", fetch("https://walled.test/post"))

    def test_page_with_nothing_to_summarize_is_reported_as_failure(self):
        """알려진 차단 문구가 없어도, 본문이 이만큼 짧으면 요약할 게 없다.
        자바스크립트로 그리는 사이트가 대개 이 꼴로 걸린다."""
        from api.tools import build_page_fetcher

        fetch = build_page_fetcher(
            self.store, download=lambda url: ("앱", "Loading…"), check=_allow
        )
        self.assertIn("error", fetch("https://spa.test/post"))

    def test_long_page_quoting_a_block_message_still_passes(self):
        """차단 문구를 인용한 긴 글은 정상 본문이다. 판정이 본문 길이에 걸려
        있으므로, 짧으면서 그 문구가 있는 글은 차단으로 본다(오판해도 검색으로
        우회하니 손해가 작고, 놓치면 차단 안내문이 그대로 올라간다)."""
        from api.tools import build_page_fetcher

        article = (
            "우리 서비스가 왜 Something went wrong 화면을 띄우는지 분석했다. " * 90
        )
        fetch = build_page_fetcher(
            self.store, download=lambda url: ("장애 회고", article), check=_allow
        )
        out = fetch("https://blog.test/postmortem")
        self.assertEqual(out.get("title"), "장애 회고")

    def test_cached_junk_is_refetched(self):
        """수정 전에 캐시된 차단 안내문이 이미 들어 있다. 읽을 때도 걸러
        다시 받는다(요약 캐시의 lazy self-healing과 같은 방식)."""
        from api.tools import build_page_fetcher

        url = "https://x.com/omarsar0/article/2065880971031834786"
        self.store.put_page(url, "", "Something went wrong, but don't fret", 1e9)
        good = "제대로 받아온 본문이다. " * 30
        calls = []
        fetch = build_page_fetcher(
            self.store,
            download=lambda u: (calls.append(u), ("기사 제목", good))[1],
            now=lambda: 1e9 + 10,
            check=_allow,
        )
        out = fetch(url)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["title"], "기사 제목")

    def test_stale_cache_is_refetched(self):
        from api.tools import PAGE_CACHE_TTL_SEC, build_page_fetcher

        calls = []
        clock = {"t": 1_000_000.0}
        fetch = build_page_fetcher(
            self.store,
            download=lambda url: (
                calls.append(url),
                ("제목", f"{LONG_BODY}{len(calls)}"),
            )[1],
            now=lambda: clock["t"],
            check=_allow,
        )
        fetch("https://blog.test/notes")
        clock["t"] += PAGE_CACHE_TTL_SEC + 1
        out = fetch("https://blog.test/notes")
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["text"], f"{LONG_BODY}2")


class TestFetchGuard(unittest.TestCase):
    """fetch_page의 URL은 모델이 만든다. 페이지에 심긴 지시문이 내부망을
    부르게 할 수 있다."""

    def test_loopback_is_rejected(self):
        from api.tools import check_fetchable

        for url in [
            "http://127.0.0.1:8000/admin",
            "http://localhost:8080/",
            "http://[::1]/",
        ]:
            self.assertIsNotNone(check_fetchable(url), url)

    def test_cloud_metadata_address_is_rejected(self):
        from api.tools import check_fetchable

        self.assertIsNotNone(check_fetchable("http://169.254.169.254/latest/meta-data/"))

    def test_private_range_is_rejected(self):
        from api.tools import check_fetchable

        self.assertIsNotNone(check_fetchable("http://192.168.0.10/status"))

    def test_non_http_scheme_is_rejected(self):
        from api.tools import check_fetchable

        self.assertIsNotNone(check_fetchable("file:///etc/passwd"))

    def test_public_https_passes(self):
        from api.tools import check_fetchable

        self.assertIsNone(check_fetchable("https://nvlabs.github.io/Sana/"))


if __name__ == "__main__":
    unittest.main()
