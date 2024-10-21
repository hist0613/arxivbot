import os
import json
import pickle
import asyncio
from tqdm import tqdm

from slack_sdk import WebClient
import discord
from discord import HTTPException

from api.arxiv import get_paper_info
from api.cache import CacheManager
from api.logger import logger
from settings import (
    OLD_PAPER_SET_PATH,
    TODAY_SUMMARIES_DIR,
    MAX_NB_SHOW,
    TIME_PAUSE_SEC,
)


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

        paper_summarization = json.loads(paper_summarization)
        if isinstance(paper_summarization, list):
            paper_summarization = paper_summarization[0]
        for key, value in paper_summarization.items():
            message_content += f"\n\n- {self._format_bold(key)}: {value}"
            markdown_content += f"- **{key}**: {value}\n\n"

        return message_content, markdown_content

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

                field_thread["thread_contents"].append(
                    {
                        "paper_info": paper_info,
                        "message_content": message_content,
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
        for thread in threads:
            result = client.chat_postMessage(
                channel=self.allowed_channel, text=thread["thread_title"]
            )
            thread_ts = result["ts"]

            for content in thread["thread_contents"]:
                client.chat_postMessage(
                    channel=self.allowed_channel,
                    text=content["message_content"],
                    thread_ts=thread_ts,
                )
                self._update_old_paper_set(content["paper_info"])
                await self._apply_rate_limit()

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

    def save_summaries(self, threads: list[dict]):
        os.makedirs(TODAY_SUMMARIES_DIR, exist_ok=True)
        for field in self.fields:
            today_summaries_field_path = os.path.join(
                TODAY_SUMMARIES_DIR, field + ".md"
            )
            with open(today_summaries_field_path, "w", encoding="utf-8") as fp:
                for thread in threads:
                    fp.write(thread["thread_title"] + "\n")
                    for content in thread["thread_contents"]:
                        fp.write(content["file_content"] + "\n\n")
