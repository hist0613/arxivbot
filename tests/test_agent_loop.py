import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeItem:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        return dict(self.__dict__)


class FakeResponse:
    def __init__(self, output, text=""):
        self.output = output
        self.output_text = text


class FakeResponses:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.responses = FakeResponses(scripted)


def call(name, arguments, call_id="c1"):
    return FakeItem(
        type="function_call", name=name, arguments=arguments, call_id=call_id
    )


def run(client, **overrides):
    from api.agent_loop import run_agent

    kwargs = {
        "client": client,
        "model": "m",
        "system_prompt": "s",
        "user_input": "q",
        "tool_specs": [],
        "dispatch": lambda n, a: "{}",
    }
    kwargs.update(overrides)
    return run_agent(**kwargs)


class TestRunAgent(unittest.TestCase):
    def test_returns_text_without_tools(self):
        client = FakeClient([FakeResponse([], "안녕하세요")])
        result = run(client)
        self.assertEqual(result.text, "안녕하세요")
        self.assertEqual(result.steps, 1)
        self.assertFalse(result.truncated)
        self.assertEqual(result.tool_calls, [])

    def test_dispatches_tool_and_feeds_output_back(self):
        client = FakeClient(
            [
                FakeResponse([call("summarize_paper", '{"url": "https://arxiv.org/abs/1"}')]),
                FakeResponse([], "올렸습니다"),
            ]
        )
        seen = []

        def dispatch(name, args):
            seen.append((name, args))
            return '{"posted": true}'

        result = run(client, dispatch=dispatch)
        self.assertEqual(seen[0][0], "summarize_paper")
        self.assertEqual(seen[0][1]["url"], "https://arxiv.org/abs/1")
        self.assertEqual(result.text, "올렸습니다")
        self.assertEqual(result.tool_calls, ["summarize_paper"])
        second_input = client.responses.calls[1]["input"]
        self.assertTrue(
            any(
                isinstance(i, dict) and i.get("type") == "function_call_output"
                for i in second_input
            )
        )

    def test_multiple_calls_in_one_turn(self):
        client = FakeClient(
            [
                FakeResponse(
                    [
                        call("summarize_paper", '{"url": "u1"}', call_id="a"),
                        call("summarize_paper", '{"url": "u2"}', call_id="b"),
                    ]
                ),
                FakeResponse([], "둘 다 올렸습니다"),
            ]
        )
        result = run(client)
        self.assertEqual(result.tool_calls, ["summarize_paper", "summarize_paper"])

    def test_stops_at_max_steps(self):
        client = FakeClient(
            [FakeResponse([call("fetch_page", '{"url": "u"}')]) for _ in range(10)]
        )
        result = run(client, max_steps=3)
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.truncated)

    def test_stops_at_deadline(self):
        clock = iter([0, 1, 200, 300, 400])
        client = FakeClient(
            [FakeResponse([call("fetch_page", '{"url": "u"}')]) for _ in range(5)]
        )
        result = run(client, deadline_sec=60, now=lambda: next(clock))
        self.assertTrue(result.truncated)
        self.assertLess(result.steps, 5)

    def test_bad_arguments_do_not_crash(self):
        client = FakeClient(
            [
                FakeResponse([call("fetch_page", "이건 JSON이 아니다")]),
                FakeResponse([], "그래도 답한다"),
            ]
        )
        seen = []
        result = run(client, dispatch=lambda n, a: seen.append(a) or "{}")
        self.assertEqual(seen[0], {})
        self.assertEqual(result.text, "그래도 답한다")

    def test_on_step_is_notified(self):
        client = FakeClient(
            [
                FakeResponse([call("fetch_page", '{"url": "u"}')]),
                FakeResponse([], "끝"),
            ]
        )
        stages = []
        run(client, on_step=stages.append)
        self.assertEqual(stages, ["fetch_page"])


if __name__ == "__main__":
    unittest.main()
