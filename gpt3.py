import os
import re
import time
from tqdm import tqdm
import concurrent.futures

import openai
from openai import OpenAI

from settings import (
    OPENAI_API_KEY,
    MAX_NB_GPT3_ATTEMPT,
    MODEL,
    MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
)
from prompts import SYSTEM_PROMPT_SUMMARIZATION, USER_PROMPT_SUMMARIZATION

client = OpenAI(api_key=OPENAI_API_KEY)


def call_chatgpt(system_prompt, user_prompt):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARIZATION,
                response_format={
                    "type": "json_object",
                },
            )
            return response.choices[0].message.content.strip()
        except (
            openai.RateLimitError,
            openai.APIError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print(e)
            time.sleep(trial * 30 + 15)
    return ""


def get_openai_summarization(content):
    system_prompt = SYSTEM_PROMPT_SUMMARIZATION
    # user_prompt = USER_PROMPT_SUMMARIZATION + f'''\nabstract: "{content}"'''
    user_prompt = f'''abstract: "{content}"'''
    return call_chatgpt(system_prompt, user_prompt)


if __name__ == "__main__":
    prompt_list = [
        USER_PROMPT_SUMMARIZATION
        + f'''\nabstract: "Pre-trained Language Model (PLM) has become a representative foundation model in the natural language processing field. Most PLMs are trained with linguistic-agnostic pre-training tasks on the surface form of the text, such as the masked language model (MLM). To further empower the PLMs with richer linguistic features, in this paper, we aim to propose a simple but effective way to learn linguistic features for pre-trained language models. We propose LERT, a pre-trained language model that is trained on three types of linguistic features along with the original MLM pre-training task, using a linguistically-informed pre-training (LIP) strategy. We carried out extensive experiments on ten Chinese NLU tasks, and the experimental results show that LERT could bring significant improvements over various comparable baselines. Furthermore, we also conduct analytical experiments in various linguistic aspects, and the results prove that the design of LERT is valid and effective. Resources are available at this https URL"''',
        USER_PROMPT_SUMMARIZATION
        + f'''\nabstract: "Natural language processing researchers develop models of grammar, meaning and human communication based on written text. Due to task and data differences, what is considered text can vary substantially across studies. A conceptual framework for systematically capturing these differences is lacking. We argue that clarity on the notion of text is crucial for reproducible and generalizable NLP. Towards that goal, we propose common terminology to discuss the production and transformation of textual data, and introduce a two-tier taxonomy of linguistic and non-linguistic elements that are available in textual sources and can be used in NLP modeling. We apply this taxonomy to survey existing work that extends the notion of text beyond the conservative language-centered view. We outline key desiderata and challenges of the emerging inclusive approach to text in NLP, and suggest systematic community-level reporting as a crucial next step to consolidate the discussion."''',
    ]

    # results = []
    # for prompt in tqdm(prompt_list):
    #     results.append(call_chatgpt(prompt))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # futures = {executor.submit(call_chatgpt, prompt): prompt for prompt in prompt_list}
        futures = {}
        for prompt in tqdm(prompt_list):
            futures[
                executor.submit(call_chatgpt, SYSTEM_PROMPT_SUMMARIZATION, prompt)
            ] = prompt

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(prompt_list)
        ):
            pass

    results = [f.result() for f in futures]

    for result in results:
        print(result)
        print("=" * 80)
