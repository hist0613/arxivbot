"""초록·본문·요약 캐시.

저장은 api.store의 SQLite가 맡고, 여기서는 기존 호출부가 쓰던 사전 모양만
유지한다. 그래서 service.py / arxiv.py / workspace.py는 손대지 않는다.
"""
from api.store import Store
from prompts import is_current_summary_schema
from settings import MODEL


class _View:
    """읽기 전용 사전 흉내.

    호출부는 `cache.paper_abstracts[key]`, `key in cache.paper_abstracts`,
    `.get(key, "")` 세 가지만 쓴다. 옛 구현은 pickle에서 읽은
    defaultdict(str)이었고 `in`은 키 존재 여부였다. api/arxiv.py가 그걸
    "이미 받아봤다"로 쓰기 때문에, 값이 비었는지로 바꾸면 초록이 빈 논문을
    매일 다시 크롤링하게 된다. 그래서 존재 여부를 그대로 흉내낸다.
    """

    def __init__(self, getter, exists=None):
        self._get = getter
        self._exists = exists or (lambda key: bool(getter(key)))

    def __getitem__(self, key):
        return self._get(key)

    def get(self, key, default=""):
        # dict.get과 같게: 행이 있으면 빈 값이라도 그대로 돌려준다.
        return self._get(key) if self._exists(key) else default

    def __contains__(self, key):
        return self._exists(key)


class CacheManager:
    def __init__(self, store: Store = None):
        self.store = store or Store()
        self.paper_abstracts = _View(self.store.get_abstract, self.store.has_abstract)
        self.paper_full_contents = _View(
            self.store.get_full_content, self.store.has_full_content
        )
        # 요약만은 "비어 있으면 없는 것"이 맞다. 옛 코드도 여기서만
        # `paper_info in ... and != ""`로 빈 값을 걸러냈다.
        self.paper_summarizations = _View(self.store.get_summary)

    def has_paper_summarization(self, paper_info: str) -> bool:
        return self.store.has_summary(paper_info)

    def update_paper_abstracts(self, paper_info: str, paper_abstract: str):
        self.store.put_abstract(paper_info, paper_abstract)

    def update_paper_full_contents(self, paper_info: str, paper_full_content):
        self.store.put_full_content(paper_info, paper_full_content)

    def update_paper_summarizations(self, paper_info: str, paper_summarization: str):
        schema = (
            "4sections" if is_current_summary_schema(paper_summarization) else "legacy"
        )
        self.store.put_summary(
            paper_info, paper_summarization, schema_version=schema, model=MODEL
        )
