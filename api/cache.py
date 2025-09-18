import os
import pickle
from collections import defaultdict

from settings import (
    PAPER_ABSTRACTS_PATH,
    PAPER_SUMMARIZATIONS_PATH,
    PAPER_FULL_CONTENTS_PATH,
)


class CacheManager:
    def __init__(self):
        self.paper_abstracts = self.get_paper_abstracts()
        self.paper_full_contents = self.get_paper_full_contents()
        self.paper_summarizations = self.get_paper_summarizations()

    def get_paper_abstracts(self):
        if os.path.exists(PAPER_ABSTRACTS_PATH):
            try:
                with open(PAPER_ABSTRACTS_PATH, "rb") as fp:
                    paper_abstracts = pickle.load(fp)
            except (EOFError, pickle.UnpicklingError) as e:
                print(
                    f"Warning: Failed to load paper abstracts cache ({e}). Using empty cache."
                )
                paper_abstracts = defaultdict(str)
        else:
            paper_abstracts = defaultdict(str)
        return paper_abstracts

    def update_paper_abstracts(self, paper_info: str, paper_abstract: str):
        self.paper_abstracts[paper_info] = paper_abstract
        with open(PAPER_ABSTRACTS_PATH, "wb") as fp:
            pickle.dump(self.paper_abstracts, fp)

    def get_paper_full_contents(self):
        if os.path.exists(PAPER_FULL_CONTENTS_PATH):
            try:
                with open(PAPER_FULL_CONTENTS_PATH, "rb") as fp:
                    paper_full_contents = pickle.load(fp)
            except (EOFError, pickle.UnpicklingError) as e:
                print(
                    f"Warning: Failed to load paper full contents cache ({e}). Using empty cache."
                )
                paper_full_contents = defaultdict(str)
        else:
            paper_full_contents = defaultdict(str)
        return paper_full_contents

    def update_paper_full_contents(self, paper_info: str, paper_full_content: str):
        self.paper_full_contents[paper_info] = paper_full_content
        with open(PAPER_FULL_CONTENTS_PATH, "wb") as fp:
            pickle.dump(self.paper_full_contents, fp)

    def get_paper_summarizations(self):
        if os.path.exists(PAPER_SUMMARIZATIONS_PATH):
            try:
                with open(PAPER_SUMMARIZATIONS_PATH, "rb") as fp:
                    paper_summarizations = pickle.load(fp)
            except (EOFError, pickle.UnpicklingError) as e:
                print(
                    f"Warning: Failed to load paper summarizations cache ({e}). Using empty cache."
                )
                paper_summarizations = defaultdict(str)
        else:
            paper_summarizations = defaultdict(str)
        return paper_summarizations

    def has_paper_summarization(self, paper_info: str) -> bool:
        return (
            paper_info in self.paper_summarizations
            and self.paper_summarizations[paper_info] != ""
        )

    def update_paper_summarizations(self, paper_info: str, paper_summarization: str):
        self.paper_summarizations[paper_info] = paper_summarization
        with open(PAPER_SUMMARIZATIONS_PATH, "wb") as fp:
            pickle.dump(self.paper_summarizations, fp)
