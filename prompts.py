SYSTEM_PROMPT_SUMMARIZATION = """Please analyze and summarize the arxiv paper into an **Korean** AI newsletter with 3-4 sentences for each section. Feel free to include some technical keywords in English itself. You can write side-by-side the original English words in parenthesis if the words are not familiar or not frequently used in Korean. Please answer in JSON format where **keys are in English**. 

arxiv 논문을 분석하고 요약해주세요. 각 섹션별로 3-4문장으로 작성해주세요. 기술적인 키워드는 영어로 적어도 좋습니다. 한국어로 자연스럽지 않거나 자주 사용되지 않는 단어는 영어로 괄호 안에 적어주세요.

Consider the following format and components for the summary (but don't include all the keys if not applicable):
[
    {"What's New": "..."},
    {"Technical Details": "..."},
    {"Performance Highlights": "..."},
]
"""

from pydantic import BaseModel


class GptSummarizationResponse(BaseModel):
    whats_new: str
    technical_details: str
    performance_highlights: str


GEMINI_RESPONSE_SCHEMA_SUMMARIZATION = {
    "type": "object",
    "properties": {
        "What's New": {"type": "string"},
        "Technical Details": {"type": "string"},
        "Performance Highlights": {"type": "string"},
    },
    "required": ["What's New", "Technical Details", "Performance Highlights"],
}

# SYSTEM_PROMPT_SUMMARIZATION = """Please summarize the arxiv paper's abstract into 3 **Korean** sentences. Please provide the summary sentences with a line break ("1. ", "2. ", and "3. "). Feel free to include some technical keywords in English itself. It is recommended to focus more on the main intuitions and distinctions of the given paper, in regards with technical approaches, rather than simply explaining the experiments. You can write side-by-side the original English words in parenthesis if the words are not familiar or not frequently used in Korean. You can output the following English words in English itself: chain-of-thought, pseudo code, data augmentation, homogeneous, sequential, task-agnostic, cross-attention, label, lightweight, in-context learning, instruction tuning, instruction learning, augmentation, lexicon, low-resource, knowledge, written text, task, diffusion, gating, convolution, recurrent, attention, query, transferability, etc."""

# USER_PROMPT_SUMMARIZATION = f"""abstract: The automatic generation of Multiple Choice Questions (MCQ) has the potential to reduce the time educators spend on student assessment significantly. However, existing evaluation metrics for MCQ generation, such as BLEU, ROUGE, and METEOR, focus on the n-gram based similarity of the generated MCQ to the gold sample in the dataset and disregard their educational value. They fail to evaluate the MCQ's ability to assess the student's knowledge of the corresponding target fact. To tackle this issue, we propose a novel automatic evaluation metric, coined Knowledge Dependent Answerability (KDA), which measures the MCQ's answerability given knowledge of the target fact. Specifically, we first show how to measure KDA based on student responses from a human survey. Then, we propose two automatic evaluation metrics, KDA_disc and KDA_cont, that approximate KDA by leveraging pre-trained language models to imitate students' problem-solving behavior. Through our human studies, we show that KDA_disc and KDA_soft have strong correlations with both (1) KDA and (2) usability in an actual classroom setting, labeled by experts. Furthermore, when combined with n-gram based similarity metrics, KDA_disc and KDA_cont are shown to have a strong predictive power for various expert-labeled MCQ quality measures.

# 1. 자동 MCQ 생성은 교사의 학습 평가 시간을 크게 줄일 수 있으나, BLEU, ROUGE, METEOR과 같은 평가 메트릭은 데이터셋에 있는 골드 샘플과 비슷한 단어만 비교하기 때문에 교육적 가치를 고려하지 않고 있다.
# 2. 우리는 지식 종속 가능성(KDA)이라고 불리는 새로운 자동 평가 메트릭을 제안하여 MCQ의 대답 가능성 (answerability)을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가한다.
# 3. Human evaluation을 통해 우리는 KDA_disc와 KDA_cont가 실제 강의실 세트에서의 사용성과 강한 상관관계를 가지고 있음을 보여주었다.

# abstract:  Despite the super-human accuracy of recent deep models in NLP tasks, their robustness is reportedly limited due to their reliance on spurious patterns. We thus aim to leverage contrastive learning and counterfactual augmentation for robustness. For augmentation, existing work either requires humans to add counterfactuals to the dataset or machines to automatically matches near-counterfactuals already in the dataset. Unlike existing augmentation is affected by spurious correlations, ours, by synthesizing “a set” of counterfactuals, and making a collective decision on the distribution of predictions on this set, can robustly supervise the causality of each term. Our empirical results show that our approach, by collective decisions, is less sensitive to task model bias of attribution-based synthesis, and thus achieves significant improvements, in diverse dimensions: 1) counterfactual robustness, 2) cross-domain generalization, and 3) generalization from scarce data.

# 1. 최근 deep model들이 NLP 태스크에서 사람보다 나은 정확성을 보였으나, spurious pattern에 의존하는 문제 때문에 robustness가 제한된다고 보고되고 있다.
# 2. 기존 방법들은 사람이 counterfactual을 만들거나 모델이 데이터셋에서 counterfactual 비슷한 것들을 찾아야 했으나, 여전히 spurious correlation에 영향을 받는다는 문제가 있다.
# 3. 이 논문에서는 "여러 개의" counterfactual을 생성하고, 집합적 의사 결정 (collective decisions) 을 통해 더 robust하게 단어들의 인과관계를 파악하는 방법을 제안한다.

# """
