import concurrent.futures
import json
import os
import re
import time
from abc import ABC, abstractmethod

import google.generativeai as genai
import openai
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from openai import OpenAI
from tqdm import tqdm

from logger import logger
from prompts import SYSTEM_PROMPT_SUMMARIZATION
from settings import (
    GOOGLE_API_KEY,
    MAX_LLM_TRIALS,
    MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
    OPENAI_API_KEY,
)
from utils import llm_retry


class Agent(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def summarize(self, content: str) -> str:
        # Implement the logic to summarize the content
        pass


class AutoAgent:
    def __init__(self):
        raise EnvironmentError(
            "AutoAgent is designed to be instantiated "
            "using the `AutoAgent.from_model_name(model_name)` method."
        )

    @classmethod
    def from_model_name(cls, model_name: str, *args, **kwargs) -> Agent:
        if model_name.lower().startswith("gpt"):
            return GptAgent(model_name, *args, **kwargs)
        elif model_name.lower().startswith("gemini"):
            return GeminiAgent(model_name, *args, **kwargs)
        else:
            raise ValueError(
                f"Unrecognized model name {model_name} for AutoAgent.\n"
                "Model name should start with either 'gpt' or 'gemini'."
            )


class GptAgent(Agent):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.system_prompt_for_summarization = SYSTEM_PROMPT_SUMMARIZATION
        # self.user_prompt_for_summarization = USER_PROMPT_SUMMARIZATION
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    @llm_retry(max_trials=MAX_LLM_TRIALS)
    def summarize(self, content: str) -> str:
        return self._generate_content(
            system_prompt=self.system_prompt_for_summarization,
            user_prompt=content,
            max_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
        )

    def _generate_content(
        self, system_prompt: str, user_prompt: str, max_tokens=None
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            response_format={
                "type": "json_object",
            },
        )

        assert type(json.loads(response.choices[0].message.content.strip())) in [
            list,
            dict,
        ], f"Invalid response: {response.choices[0].message.content.strip()}"
        return response.choices[0].message.content.strip()


class GeminiAgent(Agent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.system_prompt_for_summarization = SYSTEM_PROMPT_SUMMARIZATION
        # self.user_prompt_for_summarization = USER_PROMPT_SUMMARIZATION

        genai.configure(api_key=GOOGLE_API_KEY)
        self.client = GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt_for_summarization,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
            },
        )

    @llm_retry(max_trials=MAX_LLM_TRIALS)
    def summarize(self, content: str) -> str:
        return self._generate_content(
            user_prompt=content,
        )

    def _generate_content(self, user_prompt: str) -> str:
        response = self.client.generate_content(
            contents=user_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        assert type(json.loads(response.text)) in [
            list,
            dict,
        ], f"Invalid response: {response.text}"
        return response.text


if __name__ == "__main__":
    for model_name in ["gpt-4o", "gemini-1.5-flash-latest"]:
        logger.info(f"Model: {model_name}")
        agent = AutoAgent.from_model_name(model_name)
