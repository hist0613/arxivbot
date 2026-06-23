"""호스트별 실제 URL을 resolve해 제목/텍스트 길이 출력(수동 검증, 네트워크).
사용: python tests\\smoke_resolvers.py
인자로 URL 하나만 줄 수도 있음: python tests\\smoke_resolvers.py <url>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.cache import CacheManager
from api.arxiv import ArxivClient
from api.resolvers import build_resolver

SAMPLES = {
    "arxiv": "https://arxiv.org/abs/1706.03762",
    "acl": "https://aclanthology.org/2023.acl-long.1/",
    "thecvf": "https://openaccess.thecvf.com/content/ICCV2023/html/"
              "Kirillov_Segment_Anything_ICCV_2023_paper.html",
    "neurips": "https://papers.nips.cc/paper_files/paper/2017/hash/"
               "3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
}


def main():
    cache = CacheManager()
    resolve = build_resolver(ArxivClient(cache), cache)
    urls = {"arg": sys.argv[1]} if len(sys.argv) > 1 else SAMPLES
    for name, url in urls.items():
        print(f"\n=== {name}: {url}")
        try:
            r = resolve(url, on_progress=lambda s: print("  stage:", s))
        except Exception as e:
            print("  ERROR:", e)
            continue
        if r is None:
            print("  -> None (미지원/실패)")
            continue
        print("  title:", r.title[:80])
        print("  source:", r.source, "| text len:", len(r.text))
        print("  text head:", r.text[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
