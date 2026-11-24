"""job 이 vLLM 을 띄우기 전에 설정을 출력하고, 기준과 다르면 exit 1.

기준
- thinking 을 켠다 (summarize.py 가 enable_thinking=True 를 보낸다)
- reasoning parser 가 있다
- 요청에 temperature 등을 넣지 않고 모델 generation_config.json 의 권장 sampling 을 쓴다 (그 파일에 값이 있어야 한다)
- 출력 한도 >= 16384
- context 가 가장 긴 입력 + 출력 한도를 담는다
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
problems = []

model_id = os.environ["MODEL_ID"]
parser = os.environ.get("REASONING_PARSER", "")
max_model_len = int(os.environ.get("MAX_MODEL_LEN", "0"))
papers = os.environ.get("PAPERS", "")

src = (HERE / "summarize.py").read_text()
max_tokens = int(re.search(r"^MAX_TOKENS = (\d+)", src, re.M).group(1))
max_chars = int(re.search(r"^MAX_TEXT_CHARS = ([\d_]+)", src, re.M).group(1).replace("_", ""))
thinking = '"enable_thinking": True' in src
sampling_in_request = re.search(r"\b(temperature|top_p|top_k)\s*=", src) is not None

gen = {}
roots = [os.environ.get("HF_HUB_CACHE", ""), "~/.cache/huggingface/hub"] + os.environ.get("HF_EXTRA_CACHES", "").split(":")
for root in filter(None, roots):
    hits = sorted(glob.glob(os.path.expanduser(f"{root}/models--{model_id.replace('/', '--')}/snapshots/*/generation_config.json")))
    if hits:
        gen = json.load(open(hits[-1]))
        break
recommended = {k: gen.get(k) for k in ["temperature", "top_p", "top_k"]}

longest = 0
if papers and Path(papers).exists():
    longest = max(len(json.loads(l).get("text", "")) for l in open(papers))
need = min(longest, max_chars) // 3 + max_tokens + 4096  # 영어 전문은 토큰당 약 3.5~4자, 여유 있게 3자

rows = [("model", model_id), ("recommended sampling", recommended), ("sampling in request", sampling_in_request),
        ("thinking", thinking), ("reasoning parser", parser or "-"), ("max_tokens", max_tokens),
        ("max_model_len", max_model_len), ("longest input chars (cut)", f"{longest} ({max_chars})"),
        ("context needed (est.)", need)]
if not thinking:
    problems.append("thinking 이 꺼져 있음")
if not parser:
    problems.append("REASONING_PARSER 가 없음")
if not recommended["temperature"] or not recommended["top_p"]:
    problems.append("generation_config.json 에 권장 sampling 이 없음")
if sampling_in_request:
    problems.append("summarize.py 가 요청에 sampling 값을 넣고 있음")
if max_tokens < 16384:
    problems.append(f"출력 한도 {max_tokens} < 16384")
if max_model_len < need:
    problems.append(f"context {max_model_len} < 필요 추정 {need}")

print("[preflight] settings")
for k, v in rows:
    print(f"  {k}: {v}")
if problems:
    print("[preflight] FAIL")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("[preflight] OK")
