from pydantic import BaseModel
from typing import List


class SummarizationResponse(BaseModel):
    prior_approaches: str
    core_contribution: str
    technical_challenges: str
    empirical_impact: str


class Author(BaseModel):
    name: str
    affiliation: str
    email: str


class AuthorExtractionResponse(BaseModel):
    authors: List[Author]


SYSTEM_PROMPT_SUMMARIZATION = """Please analyze the arxiv paper and write a Korean AI-newsletter style summary in JSON.
Use exactly these four English keys, each with a Korean value of 2-3 sentences.
HARD LIMIT: the whole summary MUST NOT exceed 12 sentences total. Be concise.

언어 규칙(중요):
- 널리 통용되는 개념/표현은 한국어로 쓴다 (예: frequency distribution→주파수 분포, noise→잡음, baseline→기준선).
- 단, 국내 ML/로보틱스 커뮤니티에서 관용적으로 영어로 쓰는 전문용어는 번역·음차하지 말고 영어 원문 그대로 둔다 (예: zero-shot, few-shot, fine-tuning, end-to-end, closed-loop, force closure, embodiment, flow matching, point cloud, self-supervised, in-context learning). "영샷"·"폐루프"·"영점샷" 같은 번역/음차는 금지.
- 정착된 한국어 표현이 없거나 고유명사(모델명/기법명/벤치마크명)인 경우에만 영어를 쓰고, 처음 등장 시 한국어(영어)로 병기한다.
- 한국어로 자연스러운데 굳이 영어를 남발하지 말 것. 문장 구조는 항상 한국어.
- 영어를 한글로 음차(transliteration)하지 말 것 — 한국어로 번역하거나 영어 원문을 그대로 유지한다.

각 섹션의 의미:
- prior_approaches: 이 논문이 다루는 문제의 기존 방법들을 분류하고 그 한계를 설명.
- core_contribution: 기존 한계 중 무엇을/어떤 문제를 이 논문의 기여가 해결하는지.
- technical_challenges: 그 기여 실현의 technical challenge와 이를 어떻게 해결했는지.
- empirical_impact: 기여가 어떻게 empirical하게 입증됐고 해당 분야에서 갖는 의미/impact.

답은 JSON 형식이며 키는 영어로 둔다."""


SYSTEM_PROMPT_AUTHOR_EXTRACTION = """You are an expert at extracting author information from academic papers. 
Given the HTML content containing author information, extract each author's name, affiliation, and email (if available).
Return the information in a clean JSON array format with the following structure:
{
    "authors": [
        {
            "name": "Author Name",
            "affiliation": "University or Institution",
            "email": "email@domain.com"
        },
        ...
    ]
}"""
