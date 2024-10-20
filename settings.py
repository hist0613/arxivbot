import os
import time

from dotenv import load_dotenv
import tiktoken

load_dotenv()

MAX_NB_CRAWL = 500
MAX_NB_SHOW = 20
MAX_LLM_TRIALS = 3
TIME_PAUSE_SEC = 15
# MODEL = "gpt-4o"  # "gpt-4-turbo", "gemini-1.5-flash-latest"
MODEL = "gpt-4o-mini"
MAX_INPUT_TOKENS_FOR_SUMMARIZATION = 2048
MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION = 1024
NB_THREADS = 5

WORKSPACES = [
    {
        "service_type": "slack",
        "workspace": "ai-research-kr",
        "allowed_channel": "arxiv",
        "allowed_channel_id": "C07PLM4LJGN",
        "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
        "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI", "cs.LG"],
    },
    # {
    #     "service_type": "slack",
    #     "workspace": "ai-research-kr",
    #     "allowed_channel": "nlp-research",
    #     "allowed_channel_id": "C062ZEG7U1K",
    #     "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
    #     "fields": ["cs.CL", "cs.IR"],
    # },
    # {
    #     "service_type": "slack",
    #     "workspace": "ai-research-kr",
    #     "allowed_channel": "vision-research",
    #     "allowed_channel_id": "C0632M8CSCS",
    #     "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
    #     "fields": ["cs.CV", "cs.AI"],
    # },
    {
        "service_type": "discord",
        "workspace": "K-LLaMA",
        "allowed_channel": "arxiv",
        "guild_id": int(os.getenv("DISCORD_GUILD_ID_KLLAMA")),
        "allowed_channel_id": int(os.getenv("DISCORD_CHANNEL_ID_KLLAMA")),
        "discord_token": os.getenv("DISCORD_BOT_TOKEN_KLLAMA"),
        "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI"],
    },
    # {
    #     "service_type": "discord",
    #     "workspace": "arxivbot-test",
    #     "allowed_channel": "arxiv",
    #     "guild_id": int(os.getenv("DISCORD_GUILD_ID_TEST")),
    #     "allowed_channel_id": int(os.getenv("DISCORD_CHANNEL_ID_TEST")),
    #     "discord_token": os.getenv("DISCORD_BOT_TOKEN_TEST"),
    #     "fields": ["cs.CL", "cs.IR"],
    # },
    # {
    #     "service_type": "slack",
    #     "workspace": "hist0613",
    #     "allowed_channel": "general",
    #     "allowed_channel_id": "C0GHE21T4",
    #     "slack_token": os.getenv("SLACK_TOKEN_HIST0613"),
    #     "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI"],
    # },
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

base_dir = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
old_paper_set_path = os.path.join(base_dir, "old_paper_set_{}.pickle")
paper_abstracts_path = os.path.join(base_dir, "paper_abstracts.pickle")
paper_summarizations_path = os.path.join(base_dir, "paper_summarizations.pickle")
paper_full_contents_path = os.path.join(base_dir, "paper_full_contents.pickle")
encoding = tiktoken.encoding_for_model("gpt-4o")

summaries_dir = os.path.join(base_dir, "summaries")
today_summaries_dir = os.path.join(summaries_dir, time.strftime("%Y-%m-%d"))
