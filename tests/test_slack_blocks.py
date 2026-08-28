"""Workspace.prepare_slack_blocks — Slack rich_text 글머리 기호 목록 조립."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.workspace import (  # noqa: E402
    SLACK_BLOCK_TEXT_LIMIT,
    Workspace,
    markdown_links_to_slack,
    rich_text_elements,
)


def _workspace(service_type="slack"):
    config = {
        "service_type": service_type,
        "workspace": "test-workspace",
        "allowed_channel": "#test",
        "allowed_channel_id": "C0" if service_type == "slack" else 1,
        "fields": [],
    }
    if service_type == "slack":
        config["slack_token"] = "xoxb-test"
    else:
        config.update({"discord_token": "t", "guild_id": 1})
    return Workspace(config)


SUMMARY = json.dumps(
    {"Prior Approaches": "기존 방식", "Core Contribution": "핵심 기여"}
)


class TestPrepareSlackBlocks(unittest.TestCase):
    def test_builds_bullet_list_with_bold_labels(self):
        blocks = _workspace().prepare_slack_blocks("Title (url)", "", SUMMARY)

        self.assertEqual(blocks[0]["type"], "section")
        self.assertEqual(blocks[0]["text"]["text"], "*Title (url)*")

        rich = blocks[-1]
        self.assertEqual(rich["type"], "rich_text")
        lst = rich["elements"][0]
        self.assertEqual(lst["type"], "rich_text_list")
        self.assertEqual(lst["style"], "bullet")
        self.assertEqual(len(lst["elements"]), 2)

        label, body = lst["elements"][0]["elements"]
        self.assertEqual(label["text"], "Prior Approaches: ")
        self.assertTrue(label["style"]["bold"])
        self.assertEqual(body["text"], "기존 방식")
        self.assertNotIn("style", body)

    def test_comment_becomes_its_own_section(self):
        blocks = _workspace().prepare_slack_blocks("T (u)", " 코멘트 ", SUMMARY)
        self.assertEqual(blocks[1]["text"]["text"], "코멘트")

    def test_note_becomes_context_block(self):
        blocks = _workspace().prepare_slack_blocks(
            "T (u)", "", SUMMARY, extra_text="게재본과 다를 수 있어요"
        )
        self.assertEqual(blocks[-1]["type"], "context")
        self.assertEqual(
            blocks[-1]["elements"][0]["text"], "게재본과 다를 수 있어요"
        )

    def test_accepts_list_wrapped_summary(self):
        blocks = _workspace().prepare_slack_blocks(
            "T (u)", "", json.dumps([json.loads(SUMMARY)])
        )
        self.assertEqual(len(blocks[-1]["elements"][0]["elements"]), 2)

    def test_falls_back_when_section_too_long(self):
        long_summary = json.dumps({"K": "x" * (SLACK_BLOCK_TEXT_LIMIT + 1)})
        self.assertIsNone(
            _workspace().prepare_slack_blocks("T (u)", "", long_summary)
        )

    def test_none_for_discord(self):
        self.assertIsNone(
            _workspace("discord").prepare_slack_blocks("T (u)", "", SUMMARY)
        )


class TestPrepareTextBlocks(unittest.TestCase):
    """에이전트 자유 답변(한 줄 요지 + 항목)의 글머리 기호 조립."""

    def test_lead_and_bullets(self):
        blocks = _workspace().prepare_text_blocks(
            "Sana Video2는 실시간 비디오 생성 모델이다.\n- 512x512에서 30fps\n- 오픈 가중치"
        )
        self.assertEqual(blocks[0]["type"], "section")
        items = blocks[-1]["elements"][0]["elements"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["elements"][0]["text"], "512x512에서 30fps")

    def test_bullets_only(self):
        blocks = _workspace().prepare_text_blocks("- 가\n- 나")
        self.assertEqual(blocks[0]["type"], "rich_text")

    def test_none_without_bullets(self):
        self.assertIsNone(_workspace().prepare_text_blocks("그냥 한 문단 설명"))

    def test_prose_after_bullets_keeps_the_order(self):
        """예전에는 목록 뒤 산문 한 줄에 통째로 포기해 하이픈이 그대로 보였다."""
        blocks = _workspace().prepare_text_blocks(
            "들어가는 말\n- 가\n- 나\n뒤에 산문이 붙는다\n- 다"
        )
        self.assertEqual(
            [b["type"] for b in blocks],
            ["section", "rich_text", "section", "rich_text"],
        )
        self.assertEqual(blocks[2]["text"]["text"], "뒤에 산문이 붙는다")
        self.assertEqual(len(blocks[1]["elements"][0]["elements"]), 2)

    def test_blank_line_does_not_break_a_list(self):
        blocks = _workspace().prepare_text_blocks("- 가\n\n- 나")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]["elements"][0]["elements"]), 2)

    def test_numbered_list_becomes_ordered(self):
        blocks = _workspace().prepare_text_blocks("1. 첫째\n2) 둘째")
        self.assertEqual(blocks[0]["elements"][0]["style"], "ordered")
        self.assertEqual(len(blocks[0]["elements"][0]["elements"]), 2)

    def test_indented_items_become_a_nested_list(self):
        blocks = _workspace().prepare_text_blocks("- 가\n  - 가1\n- 나")
        lists = blocks[0]["elements"]
        self.assertEqual([l["indent"] for l in lists], [0, 1, 0])

    def test_bold_marker_is_a_style_not_a_literal_asterisk(self):
        """rich_text 안에서는 mrkdwn이 안 통해 `*굵게*`가 별표째 보인다."""
        blocks = _workspace().prepare_text_blocks("- *제목*: 내용")
        elements = blocks[0]["elements"][0]["elements"][0]["elements"]
        self.assertEqual(elements[0], {
            "type": "text", "text": "제목", "style": {"bold": True}
        })
        self.assertEqual(elements[1]["text"], ": 내용")

    def test_bold_title_line_is_not_mistaken_for_a_list(self):
        blocks = _workspace().prepare_text_blocks("*Audio LLM 정리*\n- 가")
        self.assertEqual(blocks[0]["type"], "section")
        self.assertEqual(blocks[0]["text"]["text"], "*Audio LLM 정리*")

    def test_none_for_discord(self):
        self.assertIsNone(_workspace("discord").prepare_text_blocks("- 가\n- 나"))

    def test_none_when_item_too_long(self):
        long_item = "- " + "x" * (SLACK_BLOCK_TEXT_LIMIT + 1)
        self.assertIsNone(_workspace().prepare_text_blocks(long_item))

    def test_markdown_link_in_bullet_becomes_link_element(self):
        blocks = _workspace().prepare_text_blocks(
            "- critic이 없다 ([arxiv.org](https://arxiv.org/abs/2402.03300?utm_source=openai))"
        )
        elements = blocks[-1]["elements"][0]["elements"][0]["elements"]
        link = [e for e in elements if e["type"] == "link"][0]
        self.assertEqual(link["url"], "https://arxiv.org/abs/2402.03300")
        self.assertEqual(link["text"], "arxiv.org")
        self.assertTrue(any(e["type"] == "text" for e in elements))


class TestLinkConversion(unittest.TestCase):
    """web_search를 쓰면 모델이 마크다운 링크를 뱉는다. Slack은 그걸 모른다."""

    def test_markdown_to_mrkdwn(self):
        self.assertEqual(
            markdown_links_to_slack("근거 [논문](https://arxiv.org/abs/1)"),
            "근거 <https://arxiv.org/abs/1|논문>",
        )

    def test_tracking_param_is_stripped(self):
        out = markdown_links_to_slack("[x](https://a.test/p?utm_source=openai)")
        self.assertEqual(out, "<https://a.test/p|x>")

    def test_tracking_param_among_others_is_stripped(self):
        out = markdown_links_to_slack("[x](https://a.test/p?id=3&utm_source=openai)")
        self.assertEqual(out, "<https://a.test/p?id=3|x>")

    def test_plain_text_without_links_is_one_element(self):
        self.assertEqual(
            rich_text_elements("링크 없는 문장"),
            [{"type": "text", "text": "링크 없는 문장"}],
        )

    def test_bare_url_becomes_link(self):
        elements = rich_text_elements("여기 https://a.test/p 참고")
        self.assertEqual(elements[1], {"type": "link", "url": "https://a.test/p"})
        self.assertEqual(elements[2]["text"], " 참고")


if __name__ == "__main__":
    unittest.main()
