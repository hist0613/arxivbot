import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def msg(ts, text, user="U1"):
    return {"ts": ts, "user": user, "text": text}


def count_tokens(text):
    return max(1, len(text) // 4)


class TestFoldBotMessage(unittest.TestCase):
    def test_long_summary_becomes_one_line(self):
        from api.context import fold_bot_message

        text = (
            "GSM-Symbolic: 수학 추론의 한계 (https://arxiv.org/abs/2410.05229)\n"
            + "- **Prior Approaches**: "
            + "가" * 900
        )
        folded = fold_bot_message(text)
        self.assertIn("요약 게시", folded)
        self.assertIn("https://arxiv.org/abs/2410.05229", folded)
        self.assertLess(len(folded), 200)

    def test_short_bot_message_is_kept(self):
        from api.context import fold_bot_message

        self.assertEqual(fold_bot_message("네 그렇습니다"), "네 그렇습니다")


class TestBuildContext(unittest.TestCase):
    def test_keeps_newest_within_budget(self):
        from api.context import build_context

        messages = [msg(str(i), "가" * 400) for i in range(1, 11)]
        ctx = build_context(
            messages, bot_user_id="B1", count_tokens=count_tokens, budget=300
        )
        self.assertLess(len(ctx.kept), 10)
        self.assertEqual(ctx.kept[-1]["ts"], "10")  # 최신은 반드시 남는다
        self.assertIsNotNone(ctx.folded_until_ts)

    def test_everything_fits_means_no_fold(self):
        from api.context import build_context

        messages = [msg("1", "짧다"), msg("2", "짧다")]
        ctx = build_context(
            messages, bot_user_id="B1", count_tokens=count_tokens, budget=1000
        )
        self.assertEqual(len(ctx.kept), 2)
        self.assertIsNone(ctx.folded_until_ts)

    def test_bot_summary_is_folded_even_when_kept(self):
        from api.context import build_context

        long_summary = "제목 (https://arxiv.org/abs/2410.05229)\n" + "가" * 2000
        messages = [msg("1", long_summary, user="B1"), msg("2", "이거 쉽게 설명해줘")]
        ctx = build_context(
            messages, bot_user_id="B1", count_tokens=count_tokens, budget=1000
        )
        self.assertEqual(len(ctx.kept), 2)
        self.assertIn("요약 게시", ctx.kept[0]["text"])

    def test_single_huge_message_still_kept(self):
        from api.context import build_context

        messages = [msg("1", "가" * 100000)]
        ctx = build_context(
            messages, bot_user_id="B1", count_tokens=count_tokens, budget=10
        )
        self.assertEqual(len(ctx.kept), 1)

    def test_render_puts_digest_first(self):
        from api.context import build_context, render_context

        messages = [msg("1", "안녕"), msg("2", "논문 뭐 있어?", user="U2")]
        ctx = build_context(
            messages,
            bot_user_id="B1",
            count_tokens=count_tokens,
            budget=1000,
            digest="앞서 RAG 얘기를 했다",
        )
        text = render_context(ctx, mention_text="<@B1> 정리해줘")
        self.assertLess(text.index("앞서 RAG"), text.index("논문 뭐 있어?"))
        self.assertTrue(text.rstrip().endswith("정리해줘"))


class TestUpdateDigest(unittest.TestCase):
    def setUp(self):
        from api.store import Store

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.store.close()
        os.unlink(self.tmp.name)

    def test_digest_appends_only_new_range(self):
        from api.context import update_digest

        calls = []

        def summarize(previous, texts):
            calls.append((previous, list(texts)))
            return (previous + " " + " ".join(texts)).strip()

        first = update_digest(
            self.store, "T1", [msg("1", "가"), msg("2", "나")], summarize
        )
        self.assertEqual(first, "가 나")

        second = update_digest(
            self.store,
            "T1",
            [msg("1", "가"), msg("2", "나"), msg("3", "다")],
            summarize,
        )
        self.assertEqual(calls[1][0], "가 나")  # 이전 요지를 이어받고
        self.assertEqual(calls[1][1], ["다"])  # 새로 밀려난 것만 넘긴다
        self.assertEqual(second, "가 나 다")

    def test_no_new_range_skips_model_call(self):
        from api.context import update_digest

        calls = []

        def summarize(previous, texts):
            calls.append(texts)
            return "요지"

        folded = [msg("1", "가")]
        update_digest(self.store, "T1", folded, summarize)
        update_digest(self.store, "T1", folded, summarize)
        self.assertEqual(len(calls), 1)

    def test_nothing_folded_returns_empty(self):
        from api.context import update_digest

        def summarize(previous, texts):
            raise AssertionError("불려서는 안 된다")

        self.assertEqual(update_digest(self.store, "T1", [], summarize), "")


if __name__ == "__main__":
    unittest.main()
