import os
import re
import time
import json
from tqdm import tqdm
import concurrent.futures

from abc import ABC, abstractmethod

import openai
from openai import OpenAI

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from settings import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    MAX_LLM_TRIALS,
    MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
)
from prompts import SYSTEM_PROMPT_SUMMARIZATION, USER_PROMPT_SUMMARIZATION
from logger import logger
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
            pass
            # return GeminiAgent(model_name, *args, **kwargs)
        else:
            raise ValueError(
                f"Unrecognized model name {model_name} for AutoAgent.\n"
                "Model name should start with either 'gpt' or 'gemini'."
            )


class GptAgent(Agent):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.system_prompt_for_summarization = SYSTEM_PROMPT_SUMMARIZATION
        self.user_prompt_for_summarization = USER_PROMPT_SUMMARIZATION
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    @llm_retry(max_trials=MAX_LLM_TRIALS)
    def summarize(self, content: str) -> str:
        return self._generate_content(
            self.system_prompt_for_summarization,
            self.user_prompt_for_summarization + f"""\nabstract: "{content}""",
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
        self.user_prompt_for_summarization = USER_PROMPT_SUMMARIZATION

        genai.configure(api_key=GOOGLE_API_KEY)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_content(self, content: str) -> str:
        # Implement the logic to generate content using Gemini model
        pass

    def summarize(self, content: str) -> str:
        return self.generate_content(content)


if __name__ == "__main__":
    prompt_list = [
        USER_PROMPT_SUMMARIZATION
        + f'''\nabstract: "Pre-trained Language Model (PLM) has become a representative foundation model in the natural language processing field. Most PLMs are trained with linguistic-agnostic pre-training tasks on the surface form of the text, such as the masked language model (MLM). To further empower the PLMs with richer linguistic features, in this paper, we aim to propose a simple but effective way to learn linguistic features for pre-trained language models. We propose LERT, a pre-trained language model that is trained on three types of linguistic features along with the original MLM pre-training task, using a linguistically-informed pre-training (LIP) strategy. We carried out extensive experiments on ten Chinese NLU tasks, and the experimental results show that LERT could bring significant improvements over various comparable baselines. Furthermore, we also conduct analytical experiments in various linguistic aspects, and the results prove that the design of LERT is valid and effective. Resources are available at this https URL"''',
        USER_PROMPT_SUMMARIZATION
        + f'''\nabstract: "Natural language processing researchers develop models of grammar, meaning and human communication based on written text. Due to task and data differences, what is considered text can vary substantially across studies. A conceptual framework for systematically capturing these differences is lacking. We argue that clarity on the notion of text is crucial for reproducible and generalizable NLP. Towards that goal, we propose common terminology to discuss the production and transformation of textual data, and introduce a two-tier taxonomy of linguistic and non-linguistic elements that are available in textual sources and can be used in NLP modeling. We apply this taxonomy to survey existing work that extends the notion of text beyond the conservative language-centered view. We outline key desiderata and challenges of the emerging inclusive approach to text in NLP, and suggest systematic community-level reporting as a crucial next step to consolidate the discussion."''',
    ]

    for model_name in ["gpt-4o", "gemini-1.5-flash-latest"]:
        logger.info(f"Model: {model_name}")
        agent = AutoAgent.from_model_name(model_name)
        for prompt in prompt_list:
            logger.info(agent.summarize(prompt))
