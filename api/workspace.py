import os
import json
import pickle
import re
import asyncio
from datetime import datetime, timezone
from tqdm import tqdm

from slack_sdk import WebClient
import discord
from discord import HTTPException

from api.arxiv import get_paper_info
from api.cache import CacheManager
from api.reactions import load_store, save_store, add_posted
from api.logger import logger
from settings import (
    OLD_PAPER_SET_PATH,
    MAX_NB_SHOW,
    TIME_PAUSE_SEC,
)

# Slack 블록 하나에 들어갈 수 있는 텍스트 상한(3000)에서 여유를 둔 값.
SLACK_BLOCK_TEXT_LIMIT = 2800

# web_search를 쓴 답변에는 `[huggingface.co](https://...)` 같은 마크다운 링크가
# 섞여 나온다. Slack은 이 문법을 모르고 대괄호를 그대로 보여준다.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_SLACK_LINK_RE = re.compile(r"<(https?://[^\s|>]+)(?:\|([^>]*))?>")
_BARE_URL_RE = re.compile(r"https?://[^\s<>|]+")
_TRACKING_RE = re.compile(r"[?&]utm_[a-z_]+=[^&\s)]+")


def _clean_url(url: str) -> str:
    url = _TRACKING_RE.sub("", url)
    return url.rstrip("?&")


def markdown_links_to_slack(text: str) -> str:
    """`[글자](url)` -> `<url|글자>`. mrkdwn(text 필드)용 변환."""

    def repl(m):
        return f"<{_clean_url(m.group(2))}|{m.group(1)}>"

    return _MD_LINK_RE.sub(repl, text)


def rich_text_elements(text: str) -> list:
    """한 줄을 rich_text_section의 요소 목록으로 쪼갠다.

    rich_text 블록 안에서는 `<url|글자>` mrkdwn이 통하지 않는다. 링크는
    {"type": "link"} 요소로 넣어야 눌린다. 그래서 마크다운 링크와 맨 URL을
    찾아 텍스트/링크 요소를 번갈아 만든다.
    """
    normalized = markdown_links_to_slack(text)
    elements, cursor = [], 0
    pattern = re.compile(f"{_SLACK_LINK_RE.pattern}|{_BARE_URL_RE.pattern}")
    for m in pattern.finditer(normalized):
        if m.start() > cursor:
            elements.append({"type": "text", "text": normalized[cursor:m.start()]})
        url = _clean_url(m.group(1) or m.group(0))
        label = m.group(2)
        link = {"type": "link", "url": url}
        if label:
            link["text"] = label
        elements.append(link)
        cursor = m.end()
    if cursor < len(normalized):
        elements.append({"type": "text", "text": normalized[cursor:]})
    return elements or [{"type": "text", "text": normalized}]


class Workspace:
    def __init__(self, workspace_config: dict):
        self.workspace_config: dict = workspace_config
        self.service_type: str = workspace_config["service_type"]
        self.workspace: str = workspace_config["workspace"]
        if self.service_type == "slack":
            self.allowed_channel: str = workspace_config["allowed_channel"]
            self.allowed_channel_id: str = workspace_config["allowed_channel_id"]
            self.slack_token: str = workspace_config["slack_token"]
        elif self.service_type == "discord":
            self.allowed_channel: str = workspace_config["allowed_channel"]
            self.allowed_channel_id: int = workspace_config["allowed_channel_id"]
            self.guild_id: int = workspace_config["guild_id"]
            self.discord_token: str = workspace_config["discord_token"]
        else:
            logger.error(f"Unsupported service type: {self.service_type}")
            raise ValueError(f"Unsupported service type: {self.service_type}")
        self.fields: list[str] = workspace_config["fields"]

        self.old_paper_set = self._get_old_paper_set()
        self.workspace_name = f"{self.workspace}-{self.allowed_channel}"
        self._message_count = 0

        if self.service_type == "discord":
            intents = discord.Intents.default()
            intents.messages = True
            self.discord_client = discord.Client(intents=intents)
            self.discord_ready = asyncio.Event()

            @self.discord_client.event
            async def on_ready():
                self.discord_ready.set()
                logger.info(f"Logged in as {self.discord_client.user}")

    def _get_old_paper_set(self) -> set[str]:
        if os.path.exists(OLD_PAPER_SET_PATH.format(self.workspace)):
            with open(OLD_PAPER_SET_PATH.format(self.workspace), "rb") as fp:
                old_paper_set = pickle.load(fp)
        else:
            old_paper_set = set()
        return old_paper_set

    def _update_old_paper_set(self, paper_info: str):
        self.old_paper_set.add(paper_info)
        with open(OLD_PAPER_SET_PATH.format(self.workspace), "wb") as fp:
            pickle.dump(self.old_paper_set, fp)

    def has_new_papers(self, new_papers):
        for paper_url, paper_title, _ in new_papers:
            paper_info = get_paper_info(paper_url, paper_title)
            if paper_info not in self.old_paper_set:
                return True
        return False

    def prepare_content(
        self, paper_info: str, paper_comment: str, paper_summarization: str
    ):
        message_content = self._format_bold(paper_info)
        markdown_content = "### " + paper_info + "\n"
        if paper_comment != "":
            paper_comment = paper_comment.strip()
            message_content += f"\n{paper_comment}"
            markdown_content += f"{paper_comment}\n\n"

        paper_summarization = self._load_summarization(paper_summarization)
        for key, value in paper_summarization.items():
            message_content += f"\n\n- {self._format_bold(key)}: {value}"
            markdown_content += f"- **{key}**: {value}\n\n"

        return message_content, markdown_content

    @staticmethod
    def _load_summarization(paper_summarization: str) -> dict:
        data = json.loads(paper_summarization)
        if isinstance(data, list):
            data = data[0]
        return data

    def prepare_slack_blocks(
        self,
        paper_info: str,
        paper_comment: str,
        paper_summarization: str,
        extra_text: str = "",
    ):
        """요약을 Slack rich_text 글머리 기호 목록으로 조립한다.

        Slack의 text 필드(mrkdwn)에는 목록 문법이 아예 없어서 "- "가 하이픈
        문자 그대로 남는다. 진짜 목록(둘째 줄부터 들여쓰기가 붙는)은
        rich_text_list 블록으로만 만들 수 있다.

        blocks를 못 만들면 None을 반환하고, 호출부는 기존 text 문자열로
        폴백한다(요약이 아예 안 붙는 것보다 하이픈이 낫다).
        """
        if self.service_type != "slack":
            return None

        items = []
        for key, value in self._load_summarization(paper_summarization).items():
            value = str(value)
            # 블록 하나가 길이 제한을 넘으면 Slack이 invalid_blocks로 거절한다.
            if len(value) > SLACK_BLOCK_TEXT_LIMIT:
                return None
            items.append(
                {
                    "type": "rich_text_section",
                    "elements": [
                        {"type": "text", "text": f"{key}: ", "style": {"bold": True}},
                        {"type": "text", "text": value},
                    ],
                }
            )
        if not items:
            return None

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": self._format_bold(paper_info)},
            }
        ]
        if paper_comment.strip():
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": paper_comment.strip()},
                }
            )
        blocks.append(
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_list",
                        "style": "bullet",
                        "indent": 0,
                        "elements": items,
                    }
                ],
            }
        )
        if extra_text:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": extra_text}],
                }
            )
        return blocks

    def prepare_text_blocks(self, text: str):
        """에이전트의 자유 답변("한 줄 요지 + '- ' 항목")을 글머리 기호 목록으로.

        prepare_slack_blocks는 4섹션 요약 전용이라 여기서는 평문을 다룬다.
        목록이 없거나 너무 길면 None을 반환하고 호출부는 text로 폴백한다.
        """
        if self.service_type != "slack" or not text.strip():
            return None

        # 기대하는 모양은 "한 줄 요지 + '- ' 항목들" 하나뿐이다. 산문과
        # 목록이 여러 번 번갈아 나오면 순서를 지켜 블록으로 옮기기 어려우니
        # 통째로 포기하고 평문으로 보낸다(하이픈이 보이는 편이 낫다).
        lead, items = [], []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "• ")):
                items.append(stripped[2:].strip())
            elif stripped:
                if items:
                    return None
                lead.append(stripped)
        if not items:
            return None
        if any(len(i) > SLACK_BLOCK_TEXT_LIMIT for i in items):
            return None

        blocks = []
        lead_text = "\n".join(lead).strip()
        if lead_text:
            if len(lead_text) > SLACK_BLOCK_TEXT_LIMIT:
                return None
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": markdown_links_to_slack(lead_text),
                    },
                }
            )
        blocks.append(
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_list",
                        "style": "bullet",
                        "indent": 0,
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": rich_text_elements(item),
                            }
                            for item in items
                        ],
                    }
                ],
            }
        )
        return blocks

    def _format_bold(self, text: str):
        if self.service_type == "slack":
            return f"*{text}*"
        else:  # discord
            return f"**{text}**"

    def prepare_field_threads(
        self, new_papers: dict[str, list[tuple[str, str, str]]], cache: CacheManager
    ):
        field_threads = []
        for field in tqdm(self.fields):
            if not self.has_new_papers(new_papers[field]):
                continue

            field_thread = {
                "thread_title": f"New uploads on arXiv({field})\n",
                "thread_contents": [],
            }

            # make a field thread by grouping papers of the same field
            for paper_url, paper_title, paper_comment in new_papers[field]:
                paper_info = get_paper_info(paper_url, paper_title)

                # skip duplicates
                if paper_info in self.old_paper_set:
                    continue

                message_content, file_content = self.prepare_content(
                    paper_info,
                    paper_comment,
                    cache.paper_summarizations[paper_info],
                )
                message_blocks = self.prepare_slack_blocks(
                    paper_info,
                    paper_comment,
                    cache.paper_summarizations[paper_info],
                )

                field_thread["thread_contents"].append(
                    {
                        "paper_info": paper_info,
                        "paper_url": paper_url,
                        "field": field,
                        "message_content": message_content,
                        "message_blocks": message_blocks,
                        "file_content": file_content,
                    }
                )

            field_threads.append(field_thread)

        return field_threads

    async def send_messages(self, threads: list[dict]):
        logger.info("Sending messages...")
        logger.info(f"Connecting {self.workspace} ...")

        if self.service_type == "slack":
            await self._send_slack_messages(threads)
        elif self.service_type == "discord":
            try:
                # 클라이언트 연결을 백그라운드 태스크로 실행
                asyncio.create_task(self.discord_client.start(self.discord_token))
                await self.discord_ready.wait()  # 클라이언트가 준비될 때까지 대기
                await self._send_discord_messages(threads)
            finally:
                if not self.discord_client.is_closed():
                    await self.discord_client.close()
        else:
            logger.error(f"Unsupported service type: {self.service_type}")

    async def _send_slack_messages(self, threads: list[dict]):
        client = WebClient(self.slack_token)
        store = load_store()
        for thread in threads:
            result = client.chat_postMessage(
                channel=self.allowed_channel, text=thread["thread_title"]
            )
            thread_ts = result["ts"]

            for content in thread["thread_contents"]:
                # text는 blocks가 있어도 알림 미리보기용 폴백으로 함께 보낸다.
                # 블록이 거절당하면 요약이 통째로 날아가므로 text로 재시도한다.
                blocks = content.get("message_blocks")
                try:
                    reply = client.chat_postMessage(
                        channel=self.allowed_channel,
                        text=content["message_content"],
                        thread_ts=thread_ts,
                        **({"blocks": blocks} if blocks else {}),
                    )
                except Exception as e:
                    if not blocks:
                        raise
                    logger.error(f"postMessage with blocks failed ({e}); text로 폴백")
                    reply = client.chat_postMessage(
                        channel=self.allowed_channel,
                        text=content["message_content"],
                        thread_ts=thread_ts,
                    )
                add_posted(
                    store,
                    ts=reply["ts"],
                    thread_ts=thread_ts,
                    channel_id=self.allowed_channel_id,
                    workspace=self.workspace,
                    paper_info=content["paper_info"],
                    paper_url=content["paper_url"],
                    field=content["field"],
                    posted_at=datetime.now(timezone.utc).isoformat(),
                )
                self._update_old_paper_set(content["paper_info"])
                await self._apply_rate_limit()
        save_store(store)

    async def _send_discord_messages(self, threads: list[dict]):
        try:
            guild = self.discord_client.get_guild(self.guild_id)
            if not guild:
                raise Exception(f"Guild {self.guild_id} not found.")

            channel = discord.utils.get(guild.text_channels, name=self.allowed_channel)
            if not channel:
                raise Exception(f"Channel {self.allowed_channel} not found.")

            for thread in threads:
                logger.info(thread["thread_title"].strip())
                main_message = await channel.send(thread["thread_title"])
                thread_obj = await main_message.create_thread(
                    name=thread["thread_title"], auto_archive_duration=1440
                )

                for content in tqdm(thread["thread_contents"]):
                    message_content = content["message_content"] + "\n\n"
                    try:
                        await thread_obj.send(message_content)
                    except HTTPException as e:
                        if e.code == 50035:  # Invalid Form Body
                            logger.warning(
                                f"Message too long for paper: {content['paper_info']}. Skipping this message."
                            )
                            logger.warning(
                                f"Message length: {len(message_content)} characters"
                            )
                            logger.warning(f"Message: {message_content}")
                        else:
                            logger.error(f"HTTP Exception: {str(e)}")

                    self._update_old_paper_set(content["paper_info"])
                    await self._apply_rate_limit()

        except Exception as e:
            logger.error(f"Error in sending Discord messages: {e}")

    async def _apply_rate_limit(self):
        self._message_count += 1
        if self._message_count >= MAX_NB_SHOW:
            self._message_count = 0
            await asyncio.sleep(TIME_PAUSE_SEC)
