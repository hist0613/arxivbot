import os
import time

from dotenv import load_dotenv

load_dotenv()

MAX_NB_CRAWL = 500
MAX_NB_SHOW = 20
MAX_LLM_TRIALS = 3
TIME_PAUSE_SEC = 15
TIME_PAUSE_CRAWL_SEC = 5  # 필드 간 크롤링 간격 (arXiv throttle 예방)
# MODEL = "gpt-4o"  # "gpt-4-turbo", "gemini-1.5-flash-latest", "gpt-4o-mini"
MODEL = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_INPUT_TOKENS_FOR_SUMMARIZATION = 6000  # related work 포착 (2048→6000, RW 완전포함 57%→90%)
MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION = 4000  # reasoning + 출력 합산 예산 (max_completion_tokens)
NB_THREADS = 5

WORKSPACE_CONFIGS = [
    # {
    #     "service_type": "slack",
    #     "workspace": "ai-research-kr",
    #     "allowed_channel": "arxiv",
    #     "allowed_channel_id": "C07PLM4LJGN",
    #     "slack_token": os.getenv("SLACK_TOKEN_AIRKR"),
    #     "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI", "cs.LG"],
    # },
    # {
    #     "service_type": "discord",
    #     "workspace": "seungtaek-lab",
    #     "allowed_channel": "arxivbot",
    #     "guild_id": int(os.getenv("DISCORD_GUILD_ID_SEUNGTAEK_LAB")),
    #     "allowed_channel_id": int(os.getenv("DISCORD_CHANNEL_ID_SEUNGTAEK_LAB")),
    #     "discord_token": os.getenv("DISCORD_BOT_TOKEN_SEUNGTAEK_LAB"),
    #     "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI"],
    # },
    {
        "service_type": "slack",
        "workspace": "seungtaek-lab",
        "allowed_channel": "arxivbot",          # Slack 채널 이름
        "allowed_channel_id": "C0B7V0V8U7N",        # Slack 채널 ID
        "slack_token": os.getenv("SLACK_TOKEN_SEUNGTAEK_LAB"),
        "app_token": os.getenv("SLACK_APP_TOKEN_SEUNGTAEK_LAB"),  # Socket Mode(xapp-)
        "fields": ["cs.CL", "cs.IR", "cs.CV", "cs.AI", "cs.RO", "cs.MA"],
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

REPO_DIR = os.path.dirname(os.path.abspath(__file__))  # git 저장소 루트

BASE_DIR = os.path.join(REPO_DIR, "cache")  # 캐시(pickle) 저장 위치
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

OLD_PAPER_SET_PATH = os.path.join(BASE_DIR, "old_paper_set_{}.pickle")
PAPER_ABSTRACTS_PATH = os.path.join(BASE_DIR, "paper_abstracts.pickle")
PAPER_FULL_CONTENTS_PATH = os.path.join(BASE_DIR, "paper_full_contents.pickle")
PAPER_SUMMARIZATIONS_PATH = os.path.join(BASE_DIR, "paper_summarizations.pickle")

SUMMARIES_DIR = os.path.join(REPO_DIR, "summaries")  # git에 추적/푸시되는 요약본
TODAY_SUMMARIES_DIR = os.path.join(SUMMARIES_DIR, time.strftime("%Y-%m-%d"))

# --- 리액션 수집 (Feature 1) ---
REACTIONS_DIR = os.path.join(REPO_DIR, "reactions")  # gitignore됨 (비공개)
if not os.path.exists(REACTIONS_DIR):
    os.makedirs(REACTIONS_DIR)
PAPERS_STORE_PATH = os.path.join(REACTIONS_DIR, "papers.json")
HARVEST_WINDOW_DAYS = 14
REACTION_HASH_SALT = os.getenv("REACTION_HASH_SALT", "")
