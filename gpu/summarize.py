"""vLLM(OpenAI 호환) 서버에 붙어 논문마다 두 번 묻는다.

1단계: 전문(참고문헌 제외, 부록 포함)을 읽고 배치와 같은 4섹션 요약 + 대표 그림 후보 최대 3개(id).
2단계: 후보 그림을 실제로 보여 주고 하나를 고르게 한 뒤 한국어 한 문장 설명을 받는다.

thinking 은 켜고, sampling 은 요청에 넣지 않아 서버가 모델 generation_config 권장값을 쓰게 한다.

사용: python summarize.py --papers papers.jsonl --out results.jsonl --model NAME --base-url URL
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import resvg_py
from openai import OpenAI
from PIL import Image

import sys

# 저장소에서는 상위 폴더의 prompts.py 를, 서버에서는 옆에 복사해 둔 prompts.py 를 쓴다
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prompts import SYSTEM_PROMPT_SUMMARIZATION  # noqa: E402

UA = {"User-Agent": "arxivbot/1.0 (+https://github.com/hist0613/arxivbot)"}
# 대략 20쪽 분량(쪽당 5천 자). 넘는 논문은 뒤(부록 끝)를 자른다. 하루치 중 10% 안팎이 걸린다
MAX_TEXT_CHARS = 100_000
MAX_TOKENS = 32768  # thinking 포함 출력 한도
FIG_MAX_SIDE = 1600

SUMMARY_KEYS = ["prior_approaches", "core_contribution", "technical_challenges", "empirical_impact"]

# v1 = 지금 배치·멘션이 쓰는 프롬프트 그대로. 전문을 주면 Gemma 4 는 "Be concise" 를 강하게 따라
# 요약이 nano 의 60% 길이(중앙값 700자 vs 1220자)로 짧아진다. v2 는 언어 규칙은 v1 그대로 두고
# 길이와 섹션별로 무엇을 담을지(이름·수치)를 구체적으로 지시한다.
_LANG_RULES = SYSTEM_PROMPT_SUMMARIZATION[
    SYSTEM_PROMPT_SUMMARIZATION.index("언어 규칙(중요):"):SYSTEM_PROMPT_SUMMARIZATION.index("각 섹션의 의미:")]

SYSTEM_PROMPT_V2 = """Please analyze the full paper and write a Korean AI-newsletter style summary in JSON.
Use exactly these four English keys. Each value is Korean prose of exactly 3 sentences (12 sentences in total).
Readers are ML researchers who will decide from this summary whether to read the paper, so every sentence must carry a concrete fact from the paper.

""" + _LANG_RULES + """각 섹션에 담을 것:
- prior_approaches: 이 문제를 다뤄 온 기존 방법을 두세 갈래로 묶어 대표 방법 이름과 함께 쓰고, 각 갈래가 이 문제에서 어디서 막히는지 구체적으로 쓴다.
- core_contribution: 제안하는 방법·데이터셋·벤치마크의 이름과 그것이 새로 하는 일을 쓰고, 앞의 한계 중 무엇을 푸는지 연결한다.
- technical_challenges: 그 기여를 실현할 때 부딪히는 구체적 난제 한두 개와, 이를 푼 설계(구성 요소, 학습 목표, 알고리즘 단계)를 본문의 용어로 쓴다.
- empirical_impact: 어떤 데이터셋·벤치마크에서 무엇과 비교했는지, 본문에 나온 핵심 수치 한두 개(기준선 대비 차이)를 그대로 옮기고, 이 결과가 분야에서 갖는 의미로 맺는다.

"성능을 향상시켰다", "효과를 입증했다" 같은 막연한 말로 문장을 끝내지 않는다. 무엇이 얼마나 나아졌는지를 붙인다.
이름과 수치는 본문에 있는 것만 쓴다. 본문에서 찾지 못한 수치는 쓰지 않는다.

답은 JSON 형식이며 키는 영어로 둔다."""

PROMPTS = {"v1": SYSTEM_PROMPT_SUMMARIZATION, "v2": SYSTEM_PROMPT_V2}

STAGE1_EXTRA = """

추가 키 figure_candidates: 본문에 [Figure N | id=...] 로 표시된 그림 가운데, 이 논문을 처음 보는 연구자에게
핵심 아이디어(방법의 전체 구조)나 가장 중요한 결과를 한눈에 보여 줄 그림을 최대 3개 골라 그 id 를 좋은 순서대로 담는다.
그림이 없으면 빈 목록."""

STAGE1_SCHEMA = {
    "type": "object",
    "properties": {**{k: {"type": "string"} for k in SUMMARY_KEYS},
                   "figure_candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": SUMMARY_KEYS + ["figure_candidates"],
    "additionalProperties": False,
}

STAGE2_SYSTEM = """너는 연구실 Slack 에 매일 올라가는 논문 소개에 붙일 대표 그림을 고른다.
후보 그림들을 직접 보고, 이 논문을 처음 보는 연구자가 그림 한 장으로 논문의 핵심을 가장 빨리 파악할 수 있는 것을 하나 고른다.
글자가 너무 작아 읽을 수 없거나, 세부 ablation 처럼 맥락 없이는 의미가 없는 그림은 피한다.
description 에는 그 그림이 무엇을 보여 주는지 한국어 한 문장으로 쓴다. 캡션을 옮기지 말고 독자가 그림에서 볼 것을 말한다.
요약 본문과 같이 "~습니다"로 끝맺는다.
답은 JSON: {"figure_id": "...", "description": "..."}"""

STAGE2_SCHEMA = {
    "type": "object",
    "properties": {"figure_id": {"type": "string"}, "description": {"type": "string"}},
    "required": ["figure_id", "description"],
    "additionalProperties": False,
}

_fetch_lock = threading.Lock()
_last_fetch = [0.0]


def polite_get(url: str) -> bytes:
    """arxiv.org 에는 1초에 한 번만 요청한다(스레드 공유)."""
    with _fetch_lock:
        wait = 1.0 - (time.time() - _last_fetch[0])
        if wait > 0:
            time.sleep(wait)
        _last_fetch[0] = time.time()
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.content


_UNIT_PX = {"pt": 4 / 3, "px": 1, "in": 96, "cm": 96 / 2.54, "mm": 96 / 25.4, "": 1}


def normalize_svg_size(svg: str) -> str:
    """resvg_py 가 루트 <svg> 의 width="481pt" 같은 단위를 'invalid size' 로 거절해서 px 숫자로 바꾼다.
    % 같은 상대 크기는 지워 viewBox 를 쓰게 한다."""
    m = re.search(r"<svg\b[^>]*>", svg)
    if not m:
        return svg
    tag = m.group(0)

    def fix(a):
        value, unit = float(a.group(2)), a.group(3)
        return f'{a.group(1)}="{value * _UNIT_PX[unit]:.2f}"' if unit in _UNIT_PX else ""

    new = re.sub(r'\b(width|height)="([\d.]+)\s*([a-z%]*)"', fix, tag)
    return svg[:m.start()] + new + svg[m.end():]


def to_image(data: bytes, is_svg: bool) -> Image.Image:
    if is_svg:
        svg = normalize_svg_size(data.decode("utf-8", "replace"))
        data = bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=2))
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img.convert("RGB")


def figure_png(fig: dict, cache_dir: Path) -> Path | None:
    """그림 하나를 PNG 한 장으로. 하위 그림이 여럿이면 가로로 이어 붙인다. 디스크 캐시."""
    key = hashlib.sha1(json.dumps([fig["images"], fig.get("inline_svgs", [])]).encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.png"
    if path.exists():
        return path
    parts = []
    for url in fig["images"][:6]:
        try:
            parts.append(to_image(polite_get(url), url.lower().endswith(".svg")))
        except Exception as e:  # 깨진 그림 하나 때문에 논문을 버리지 않는다
            print(f"[fig] {url} 실패: {type(e).__name__}: {e}", flush=True)
    if not parts:
        return None
    h = max(p.height for p in parts)
    parts = [p.resize((max(1, p.width * h // p.height), h)) for p in parts]
    canvas = Image.new("RGB", (sum(p.width for p in parts) + 20 * (len(parts) - 1), h), "white")
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width + 20
    canvas.thumbnail((FIG_MAX_SIDE, FIG_MAX_SIDE))
    canvas.save(path)
    return path


def data_url(path: Path) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def resolve_figure_id(answer: str, by_id: dict, by_label: dict) -> str | None:
    """모델이 적은 그림 표시를 실제 id 로. 본문 표기 "[Figure 3 | id=S3.F3]" 를 따라 "id=S3.F3" 이나
    "Figure 3" 으로 답하는 경우가 있다(Gemma 50편 중 2~4편)."""
    a = answer.strip().strip("[]").strip()
    if "id=" in a:
        a = a.split("id=", 1)[1].strip()
    if a in by_id:
        return a
    return by_label.get(a.lower()) or by_label.get(answer.strip().lower())


def ask(client, model, messages, schema, name):
    t = time.time()
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=MAX_TOKENS,
        response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    msg = resp.choices[0].message
    usage = resp.usage.model_dump() if resp.usage else {}
    return {
        "content": msg.content or "",
        "reasoning_chars": len(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""),
        "finish_reason": resp.choices[0].finish_reason,
        "usage": usage,
        "seconds": round(time.time() - t, 1),
    }


def process(paper: dict, client, model: str, cache_dir: Path, prompt: str = "v2") -> dict:
    rec = {"arxiv_id": paper["arxiv_id"], "paper_info": paper.get("paper_info"), "title": paper["title"],
           "model": model, "prompt": prompt}
    text = paper.get("text") or f"Abstract: {paper['abstract']}"
    rec["input_chars"] = len(text)
    text = text[:MAX_TEXT_CHARS]
    s1 = ask(client, model, [
        {"role": "system", "content": PROMPTS[prompt] + STAGE1_EXTRA},
        {"role": "user", "content": f"Title: {paper['title']}\n\n{text}"},
    ], STAGE1_SCHEMA, "summary")
    rec["stage1"] = s1
    try:
        out = json.loads(s1["content"])
    except json.JSONDecodeError:
        rec["error"] = "stage1 JSON 파싱 실패"
        return rec
    rec["summary"] = {k: out.get(k, "") for k in SUMMARY_KEYS}
    by_id = {f["id"]: f for f in paper.get("figures", [])}
    by_label = {f["label"].lower(): f["id"] for f in paper.get("figures", []) if f.get("label")}
    cands = []
    for c in out.get("figure_candidates", []):
        cid = resolve_figure_id(c, by_id, by_label)
        if cid and cid not in cands:
            cands.append(cid)
    rec["figure_candidates"] = out.get("figure_candidates", [])
    rec["candidate_ids_valid"] = cands
    shown = []
    for cid in cands[:3]:
        png = figure_png(by_id[cid], cache_dir)
        if png:
            shown.append((cid, png))
    rec["shown"] = [{"id": cid, "png": str(p)} for cid, p in shown]
    if not shown:
        return rec
    content = [{"type": "text", "text": (
        f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}\n\n"
        f"핵심 기여 요약: {rec['summary']['core_contribution']}\n\n후보 그림 {len(shown)}개가 아래 순서로 이어진다.")}]
    for cid, png in shown:
        f = by_id[cid]
        content.append({"type": "text", "text": f"[id={cid}] {f['caption'][:1500]}"})
        content.append({"type": "image_url", "image_url": {"url": data_url(png)}})
    s2 = ask(client, model, [{"role": "system", "content": STAGE2_SYSTEM},
                             {"role": "user", "content": content}], STAGE2_SCHEMA, "figure")
    rec["stage2"] = s2
    try:
        pick = json.loads(s2["content"])
        fid = resolve_figure_id(pick.get("figure_id", ""), by_id, by_label) or pick.get("figure_id", "")
        rec["figure"] = {**pick, "figure_id": fid, "label": by_id.get(fid, {}).get("label", ""),
                         "png": next((str(p) for c, p in shown if c == fid), None)}
    except json.JSONDecodeError:
        rec["error"] = "stage2 JSON 파싱 실패"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--fig-cache", default="figcache")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    ap.add_argument("--figs-out", help="고른 그림을 <arxiv_id>.png 로 복사해 둘 폴더(매일 배치용)")
    args = ap.parse_args()

    import os
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY", "x"), timeout=3600)
    cache_dir = Path(args.fig_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    papers = [json.loads(l) for l in open(args.papers)]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["arxiv_id"] for l in open(args.out)}
    todo = [p for p in papers if p["arxiv_id"] not in done]
    print(f"[run] {len(todo)}편 (이미 {len(done)}편), concurrency={args.concurrency}", flush=True)
    t0 = time.time()
    with open(args.out, "a") as f, ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(process, p, client, args.model, cache_dir, args.prompt): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"arxiv_id": p["arxiv_id"], "title": p["title"], "model": args.model,
                       "error": f"{type(e).__name__}: {e}"}
            png = (rec.get("figure") or {}).get("png")
            if args.figs_out and png:
                Path(args.figs_out).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(png, Path(args.figs_out) / f"{rec['arxiv_id']}.png")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {p['arxiv_id']} {time.time() - t0:.0f}s "
                  f"{rec.get('error', '')} fig={rec.get('figure', {}).get('figure_id', '-')}", flush=True)
    print(f"[run] done {len(todo)}편 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
