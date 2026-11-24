"""arXiv HTML 에서 요약 입력(전문)과 그림 후보를 뽑는다.

배치 크롤(api/arxiv.py get_paper_full_content)은 <section> 을 전부 돌아서 하위 절이 상위 절에
한 번 더 들어가고, 그림 캡션은 빠진다. 여기서는 article 을 위에서부터 한 번만 훑고,
참고문헌은 빼고 부록은 넣고, 그림 자리에는 [Figure N | id=...] 캡션을 남긴다.

사용: python extract.py <논문 목록 .json 또는 .jsonl> <papers.jsonl>
목록의 각 항목은 arxiv_id, title, abstract 를 갖는다(paper_info 등 다른 키는 그대로 넘긴다).
"""

import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

UA = {"User-Agent": "arxivbot/1.0 (+https://github.com/hist0613/arxivbot)"}
PAUSE_SEC = 3  # arXiv 요청 간격. 배치 크롤과 같은 수준으로 둔다

SKIP_CLASSES = {"ltx_bibliography", "ltx_rdf", "ltx_pagination", "ltx_authors", "ltx_dates",
                "ltx_title_document", "ltx_page_footer", "ltx_role_newpage"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def figure_record(fig: Tag, base: str) -> dict | None:
    """그림 하나를 {id, label, caption, images} 로. 이미지 파일이 없는 그림은 None.

    LaTeX(TikZ·pgfplots)가 HTML 안에 직접 그린 인라인 SVG 는 후보로 쓰지 않는다. 글자가 foreignObject 의
    HTML 로 들어 있어 PNG 로 바꾸면 축·라벨이 다 빠지고 막대만 남는다. 캡션은 본문에 그대로 남는다."""
    images = []
    for node in fig.find_all(["img", "object", "embed"]):
        src = node.get("src") or node.get("data")
        if src:
            images.append(base + src)
    if not images:
        return None
    cap = fig.find("figcaption")
    caption = cap.get_text(" ", strip=True) if cap else ""
    tag = cap.find(class_="ltx_tag") if cap else None
    label = tag.get_text(" ", strip=True).rstrip(": ") if tag else ""
    return {"id": fig.get("id", ""), "label": label, "caption": caption, "images": images}


def walk(node: Tag, out: list[str], figures: list[dict], base: str):
    for child in node.children:
        if isinstance(child, NavigableString) or not isinstance(child, Tag):
            continue
        classes = set(child.get("class") or [])
        if classes & SKIP_CLASSES:
            continue
        if child.name == "figure" and "ltx_figure" in classes and not child.find_parent("figure"):
            rec = figure_record(child, base)
            if rec:
                figures.append(rec)
                out.append(f"[{rec['label'] or 'Figure'} | id={rec['id']}] {rec['caption']}")
            else:
                cap = child.find("figcaption")
                if cap:
                    out.append(f"[{cap.get_text(' ', strip=True)}]")
            continue
        if child.name == "figure":  # 표 등
            cap = child.find("figcaption")
            if cap:
                out.append(f"[{cap.get_text(' ', strip=True)}]")
            continue
        if child.name in HEADINGS:
            out.append("\n## " + child.get_text(" ", strip=True))
            continue
        if child.name == "p" or "ltx_abstract" in classes:
            text = child.get_text(" ", strip=True)
            if text:
                out.append(text)
            continue
        walk(child, out, figures, base)


def extract(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.body
    # 이미지 src 는 "2611.21885v2/x1.png" 처럼 html/ 아래 상대경로다
    base = "https://arxiv.org/html/"
    out, figures = [], []
    walk(article, out, figures, base)
    text = "\n".join(out)
    return {"text": text, "figures": figures}


def main(src: str, dst: str):
    if src.endswith(".jsonl"):
        papers = [json.loads(line) for line in open(src) if line.strip()]
    else:
        papers = json.load(open(src))
    done = set()
    if Path(dst).exists():
        done = {json.loads(l)["arxiv_id"] for l in open(dst)}
    with open(dst, "a") as f:
        for i, p in enumerate(papers):
            if p["arxiv_id"] in done:
                continue
            url = f"https://arxiv.org/html/{p['arxiv_id']}"
            r = requests.get(url, headers=UA, timeout=60)
            rec = dict(p)
            if r.status_code == 200:
                rec.update(extract(r.text))
                rec["html_url"] = r.url
            else:
                rec.update({"text": "", "figures": [], "html_status": r.status_code})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(i, p["arxiv_id"], r.status_code, len(rec["text"]), len(rec["figures"]), flush=True)
            time.sleep(PAUSE_SEC)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
