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
    `.get(key, "")` 세 가지만 쓴다. 옛 구현이 defaultdict(str)이라 없는 키는
    빈 문자열이었고, `in`도 사실상 "값이 있느냐"로 쓰였다(빈 값이면 다시
    크롤링). 그 동작을 그대로 흉내내야 호출부를 고치지 않는다.
    """

    def __init__(self, getter):
        self._get = getter

    def __getitem__(self, key):
        return self._get(key)

    def get(self, key, default=""):
        value = self._get(key)
        return value if value else default

    def __contains__(self, key):
        # 행의 존재가 아니라 값이 비지 않았는지를 본다. 빈 값이 캐시에 남아
        # 있으면 "캐시 없음"으로 취급해 다시 받아오는 게 옛 동작이었다.
        return bool(self._get(key))


class CacheManager:
    def __init__(self, store: Store = None):
        self.store = store or Store()
        self.paper_abstracts = _View(self.store.get_abstract)
        self.paper_full_contents = _View(self.store.get_full_content)
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
