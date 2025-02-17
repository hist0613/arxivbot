from pydantic import BaseModel
from typing import List


class SummarizationResponse(BaseModel):
    whats_new: str
    technical_details: str
    performance_highlights: str


class Author(BaseModel):
    name: str
    affiliation: str
    email: str


class AuthorExtractionResponse(BaseModel):
    authors: List[Author]


SYSTEM_PROMPT_SUMMARIZATION = """Please analyze and summarize the arxiv paper into an **Korean** AI newsletter with 3-4 sentences for each section. Feel free to include some technical keywords in English itself. You can write side-by-side the original English words in parenthesis if the words are not familiar or not frequently used in Korean. Please answer in JSON format where **keys are in English**. 

arxiv 논문을 분석하고 요약해주세요. 각 섹션별로 3-4문장으로 작성해주세요. 기술적인 키워드는 영어로 적어도 좋습니다. 한국어로 자연스럽지 않거나 자주 사용되지 않는 단어는 영어로 괄호 안에 적어주세요.

Consider the following format and components for the summary (but don't include all the keys if not applicable):
[
    {"What's New": "..."},
    {"Technical Details": "..."},
    {"Performance Highlights": "..."},
]
"""


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
