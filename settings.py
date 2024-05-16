import os
from dotenv import load_dotenv

load_dotenv()

MAX_NB_CRAWL = 500
MAX_NB_SHOW = 20
MAX_NB_GPT3_ATTEMPT = 3
TIME_PAUSE_SEC = 15
SHOW_SUMMARIZATION = True
SHOW_QUESTION = False
MODEL = "gpt-4o"  # "gpt-4-turbo"
MAX_INPUT_TOKENS_FOR_SUMMARIZATION = 2048
MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION = 1024
NB_THREADS = 5
# $24.68 per week (assuming average 482 papers)

WORKSPACES = [
    {
        "workspace": "ai-research-kr",
        "allowed_channel": "nlp-research",
        "allowed_channel_id": "C062ZEG7U1K",
        "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
        "fields": ["cs.CL", "cs.IR"],
    },
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
