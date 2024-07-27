import os
from dotenv import load_dotenv

load_dotenv()

MAX_NB_CRAWL = 500
MAX_NB_SHOW = 20
MAX_LLM_TRIALS = 3
TIME_PAUSE_SEC = 15
MODEL = "gemini-1.5-flash-latest"  # "gpt-4o"  # "gpt-4-turbo"
MAX_INPUT_TOKENS_FOR_SUMMARIZATION = 2048
MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION = 1024
NB_THREADS = 5

WORKSPACES = [
    {
        "service_type": "slack",
        "workspace": "ai-research-kr",
        "allowed_channel": "nlp-research",
        "allowed_channel_id": "C062ZEG7U1K",
        "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
        "fields": ["cs.CL", "cs.IR"],
    },
    {
        "service_type": "discord",
        "workspace": "K-LLaMA",
        "allowed_channel": "arxiv",
        "guild_id": int(os.getenv("DISCORD_GUILD_ID_KLLAMA")),
        "allowed_channel_id": int(os.getenv("DISCORD_CHANNEL_ID_KLLAMA")),
        "discord_token": os.getenv("DISCORD_BOT_TOKEN_KLLAMA"),
        "fields": ["cs.CL", "cs.IR"],
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
    #     "workspace": "ai-research-kr",
    #     "allowed_channel": "vision-research",
    #     "allowed_channel_id": "C0632M8CSCS",
    #     "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
    #     "fields": ["cs.CV", "cs.AI"],
    # },
    # {
    #     'workspace': 'riiid',
    #     'allowed_channel': 'research_nlp',
    #     'allowed_channel_id': 'C024QNLBT1Q',
    #     'slack_token': os.getenv("SLACK_TOKEN_RIIID"),
    #     'fields': ["cs.CL", "cs.IR", "eess.AS"]
    # },
    # {
    #     'workspace': 'riiid',
    #     'allowed_channel': 'research_cv_arxiv',
    #     'allowed_channel_id': 'C03B3KLM8F5',
    #     'slack_token': os.getenv("SLACK_TOKEN_RIIID"),
    #     'fields': ["cs.CV", "cs.AI"]
    # },
    # {
    #     'workspace': 'hist0613',
    #     'allowed_channel': 'general',
    #     'allowed_channel_id': 'C0GHE21T4',
    #     'slack_token': os.getenv("SLACK_TOKEN_HIST0613"),
    #     'fields': ["cs.CL", "cs.IR", "cs.CV", "cs.AI"]
    # }
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
