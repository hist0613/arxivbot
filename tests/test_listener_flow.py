import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Result:
    def __init__(self, text, truncated=False, tool_calls=()):
        self.text = text
        self.truncated = truncated
        self.tool_calls = list(tool_calls)


def core(run, *, posted=0, fallback_text="폴백"):
    from listener import handle_mention_core

    return handle_mention_core(
        user_input="<@U1> 뭐 좀",
        run=run,
        fallback=lambda: {"mode": "fallback", "text": fallback_text},
        posted_count=lambda: posted,
    )


class TestHandleMentionCore(unittest.TestCase):
    def test_agent_text_is_returned(self):
        out = core(lambda user_input: Result("이렇게 이해하면 됩니다"))
        self.assertEqual(out["mode"], "agent")
        self.assertIn("이렇게", out["text"])

    def test_agent_failure_falls_back(self):
        def boom(user_input):
            raise RuntimeError("api down")

        out = core(boom)
        self.assertEqual(out["mode"], "fallback")
        self.assertEqual(out["text"], "폴백")

    def test_truncated_answer_is_marked(self):
        out = core(
            lambda user_input: Result(
                "여기까지 봤습니다", truncated=True, tool_calls=["web_search"]
            )
        )
        self.assertIn("도중에", out["text"])

    def test_empty_answer_after_posting_is_not_an_error(self):
        out = core(
            lambda user_input: Result("", tool_calls=["summarize_paper"]), posted=1
        )
        self.assertEqual(out["mode"], "agent")
        self.assertEqual(out["text"], "")

    def test_tool_called_but_nothing_posted_falls_back(self):
        """URL 가드에 막히면 도구는 불렸지만 스레드에는 아무것도 안 올라간다."""
        out = core(
            lambda user_input: Result("", tool_calls=["summarize_paper"]), posted=0
        )
        self.assertEqual(out["mode"], "fallback")

    def test_empty_answer_without_any_tool_falls_back(self):
        out = core(lambda user_input: Result(""))
        self.assertEqual(out["mode"], "fallback")


class TestProgressStage(unittest.TestCase):
    def test_tool_name_maps_to_korean_stage(self):
        from listener import stage_for_tool

        self.assertIn("페이지", stage_for_tool("fetch_page"))
        self.assertIn("이전 대화", stage_for_tool("read_thread"))
        self.assertIsNone(stage_for_tool("summarize_paper"))  # 자기 답글이 따로 있다
        # web_search는 서버가 실행해 함수 호출로 오지 않는다. 단계 문구를
        # 넣어둬도 절대 안 뜨므로 아예 두지 않는다.
        self.assertIsNone(stage_for_tool("web_search"))


if __name__ == "__main__":
    unittest.main()
