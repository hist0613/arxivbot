import os
import re
import time
from tqdm import tqdm
import concurrent.futures

import openai

from settings import OPENAI_API_KEY, MAX_NB_GPT3_ATTEMPT
from settings import PROMPT_SUMMARIZATION, PROMPT_QUESTION_GENERATION

openai.api_key = OPENAI_API_KEY


def call_chatgpt(prompt):
    for trial in range(MAX_NB_GPT3_ATTEMPT):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": prompt
                    }
                ],
                max_tokens=1024
            )
            return response['choices'][0]['message']['content'].strip()
        except (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.APIConnectionError) as e:
            print(e)
            time.sleep(trial * 30 + 15)
    return ""


def get_openai_summarization(content):
    prompt = PROMPT_SUMMARIZATION + f'''\nabstract: "{content}"'''
    return call_chatgpt(prompt)


def get_openai_question_generation(content):
    prompt = PROMPT_QUESTION_GENERATION + f'''\nabstract: "{content}"'''
    return call_chatgpt(prompt)


def get_openai_trend_analysis(paper_summarizations, nb_total_messages):
    MAX_NB_TARGET = 7
    recent_papers = sorted(paper_summarizations.keys(), key=lambda x: re.search(r'/(\d+\.\d+)', x).group(1), reverse=True)[:nb_total_messages]

    futures = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pidx = 0
        while pidx < nb_total_messages:
            nb_target = MAX_NB_TARGET
            while True:
                TREND_PROMPT = f"""I will give you summarizations of recent {MAX_NB_TARGET} arXiv papers. Based on this, please give me a brief summary and trend analysis of the recent academy. It would be a good idea to highlight a paper that you think is important. 한글을 적당히 섞어서 써줘.\n\n"""
                for paper_info in recent_papers[pidx:pidx+nb_target]:
                    paper_summarization = paper_summarizations[paper_info]
                    TREND_PROMPT += paper_info + '\n'
                    TREND_PROMPT += paper_summarization + '\n\n'
                if len(TREND_PROMPT.split()) >= 1600:
                    nb_target -= 1
                else:
                    break
            pidx += nb_target
            futures[executor.submit(call_chatgpt, TREND_PROMPT)] = pidx

    trend_analyses = []
    for f in futures:
        trend_analyses.append(f.result())

    return trend_analyses

if __name__ == "__main__":
    prompt_list = [
        PROMPT_SUMMARIZATION + f'''\nabstract: "Pre-trained Language Model (PLM) has become a representative foundation model in the natural language processing field. Most PLMs are trained with linguistic-agnostic pre-training tasks on the surface form of the text, such as the masked language model (MLM). To further empower the PLMs with richer linguistic features, in this paper, we aim to propose a simple but effective way to learn linguistic features for pre-trained language models. We propose LERT, a pre-trained language model that is trained on three types of linguistic features along with the original MLM pre-training task, using a linguistically-informed pre-training (LIP) strategy. We carried out extensive experiments on ten Chinese NLU tasks, and the experimental results show that LERT could bring significant improvements over various comparable baselines. Furthermore, we also conduct analytical experiments in various linguistic aspects, and the results prove that the design of LERT is valid and effective. Resources are available at this https URL"''', 
        PROMPT_SUMMARIZATION + f'''\nabstract: "Natural language processing researchers develop models of grammar, meaning and human communication based on written text. Due to task and data differences, what is considered text can vary substantially across studies. A conceptual framework for systematically capturing these differences is lacking. We argue that clarity on the notion of text is crucial for reproducible and generalizable NLP. Towards that goal, we propose common terminology to discuss the production and transformation of textual data, and introduce a two-tier taxonomy of linguistic and non-linguistic elements that are available in textual sources and can be used in NLP modeling. We apply this taxonomy to survey existing work that extends the notion of text beyond the conservative language-centered view. We outline key desiderata and challenges of the emerging inclusive approach to text in NLP, and suggest systematic community-level reporting as a crucial next step to consolidate the discussion."'''
    ]

    # results = []
    # for prompt in tqdm(prompt_list):
    #     results.append(call_chatgpt(prompt))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # futures = {executor.submit(call_chatgpt, prompt): prompt for prompt in prompt_list}
        futures = {}
        for prompt in tqdm(prompt_list):
            futures[executor.submit(call_chatgpt, prompt)] = prompt

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompt_list)):
            pass
    
    results = [f.result() for f in futures]

    for result in results:
        print(result)
        print("=" * 80)