import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Result:
    def __init__(self, text, truncated=False, tool_calls=()):
        self.text = text
        self.truncated = truncated
        self.tool_calls = list(tool_calls)


class TestHandleMentionCore(unittest.TestCase):
    def test_agent_text_is_returned(self):
        from listener import handle_mention_core

        out = handle_mention_core(
            user_input="<@U1> attention residual 쉽게 설명해",
            run=lambda user_input: Result("이렇게 이해하면 됩니다"),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "agent")
        self.assertIn("이렇게", out["text"])

    def test_agent_failure_falls_back(self):
        from listener import handle_mention_core

        def boom(user_input):
            raise RuntimeError("api down")

        out = handle_mention_core(
            user_input="<@U1> https://arxiv.org/abs/2410.05229",
            run=boom,
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "fallback")
        self.assertEqual(out["text"], "폴백")

    def test_truncated_answer_is_marked(self):
        from listener import handle_mention_core

        out = handle_mention_core(
            user_input="<@U1> 뭐 좀 찾아줘",
            run=lambda user_input: Result(
                "여기까지 봤습니다", truncated=True, tool_calls=["web_search"]
            ),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertIn("도중에", out["text"])

    def test_empty_answer_after_posting_is_not_an_error(self):
        from listener import handle_mention_core

        out = handle_mention_core(
            user_input="<@U1> https://arxiv.org/abs/2410.05229",
            run=lambda user_input: Result("", tool_calls=["summarize_paper"]),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "agent")
        self.assertEqual(out["text"], "")

    def test_empty_answer_without_any_tool_falls_back(self):
        from listener import handle_mention_core

        out = handle_mention_core(
            user_input="<@U1> https://arxiv.org/abs/2410.05229",
            run=lambda user_input: Result(""),
            fallback=lambda: {"mode": "fallback", "text": "폴백"},
        )
        self.assertEqual(out["mode"], "fallback")


class TestProgressStage(unittest.TestCase):
    def test_tool_name_maps_to_korean_stage(self):
        from listener import stage_for_tool

        self.assertIn("검색", stage_for_tool("web_search"))
        self.assertIn("페이지", stage_for_tool("fetch_page"))
        self.assertIsNone(stage_for_tool("summarize_paper"))  # 자기 답글이 따로 있다


if __name__ == "__main__":
    unittest.main()
