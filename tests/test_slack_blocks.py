"""Workspace.prepare_slack_blocks — Slack rich_text 글머리 기호 목록 조립."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.workspace import Workspace, SLACK_BLOCK_TEXT_LIMIT  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
