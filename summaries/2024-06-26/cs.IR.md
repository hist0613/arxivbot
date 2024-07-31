New uploads on arXiv(cs.CL)

### BMIKE-53: Investigating Cross-Lingual Knowledge Editing with In-Context Learning (https://arxiv.org/abs/2406.17764)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 다국어 지식 편집(Cross-lingual Knowledge Editing)의 중요성을 강조하며, BMIKE-53 벤치마크라는 새로운 평가 도구를 소개했습니다. 이는 53개의 다양한 언어와 세 가지 지식 편집 작업 유형을 다룹니다. 또한, 새로운 그라디언트-프리 방법인 다국어 In-context Knowledge Editing(MIKE)을 제안하고 BMIKE-53에서 평가를 진행했습니다.

- **Technical Details**: BMIKE-53는 zsRE, CounterFact, WikiFactDiff 세 개의 모노링구얼 지식 편집 데이터셋을 기반으로 구축되었습니다. 각 데이터셋의 형식을 통일한 후, Google Translate API를 사용해 52개 언어로 번역하였습니다. 제안된 방법인 MIKE는 On-the-fly KE 방법론을 사용하여, 새로운 사실과 이를 올바르게 사용하는 방법을 예시로 제공함으로써 모델의 성능을 개선합니다.

- **Performance Highlights**: BMIKE-53을 통한 평가 결과, MIKE는 신뢰성(reliability), 일반성(generality), 지역성(locality), 이동성(portability) 측면에서 뛰어난 성능을 보였습니다. 특히, MIKE는 다양한 언어간 지식 전이를 성공적으로 수행할 수 있음을 보여주었습니다.



### CaLMQA: Exploring culturally specific long-form question answering across 23 languages (https://arxiv.org/abs/2406.17761)
Comments:
          39 pages, 16 figures. Code and data available at this https URL

- **What's New**: 대형 언어 모델(LLM)을 사용한 다국어 장문 질문 응답(LFQA) 연구를 위한 새로운 데이터셋 'CaLMQA'가 소개되었습니다. 이 데이터셋은 23개 언어에 걸친 2,600개의 복잡한 질문을 포함하며, Fijian 및 Kirundi와 같은 자원 부족 언어도 포함합니다. 이 데이터셋은 원어민이 작성한 질문과 커뮤니티 웹 포럼에서 수집된 자연 발생 질문들을 포함하고 있습니다.

- **Technical Details**: CaLMQA 데이터셋은 문화적으로 특정한 질문(culturally specific questions)과 비문화적 질문(culturally agnostic questions)으로 구성됩니다. 질문은 원래 언어, 영어로 번역된 질문, 문화적 특성 부여 여부 태그, 그리고 고자원 및 중간 자원 언어에 대한 사람 작성 참조 답변을 포함합니다. 모델 성능 평가를 위해 새롭게 개발된 'CaLMScore' 메트릭을 사용하며, 이는 비정확한 언어 및 토큰 반복을 감지합니다. 인간 평가를 통해 모델의 응답 품질을 추가로 검증합니다.

- **Performance Highlights**: GPT-4-Turbo와 Claude-3-Opus 모델이 높은 CaLMScore를 기록했지만, Mixtral-8x22B와 Llama-3-70B는 상대적으로 낮은 성능을 보였습니다. Tswana 및 Tongan 언어에서 모든 모델들이 저조한 성능을 보였고, Afar 언어에서는 신뢰할 만한 텍스트 생성에 어려움을 겪었습니다. 사람 평가 결과, 문화적으로 특정한 질문이 비문화적 질문에 비해 훨씬 낮은 점수를 받았으며, 사실 오류 및 누락이 응답 품질 평가의 주요 예측 요인이었습니다.



### Accelerating Clinical Evidence Synthesis with Large Language Models (https://arxiv.org/abs/2406.17755)
- **What's New**: AI 발전에 걸친 원대한 꿈 중 하나는 자동 의학 발견입니다. 이를 위한 첫 단계로 임상 연구를 이해하고 문헌에서 임상 증거를 합성하는 AI 모델을 소개합니다. TrialMind라는 생성 AI 기반 파이프라인을 통해 의료 체계적 리뷰를 수행하는데, 이는 연구 검색, 스크리닝, 데이터 추출 단계를 포괄합니다. 우리는 대형 언어 모델(LLMs)을 각 파이프라인 구성 요소에 도입하고, 오류를 최소화하기 위해 인간 전문가의 감시를 통합했습니다. 또한 870개의 주석이 달린 임상 연구로 구성된 맞춤형 데이터셋 TrialReviewBench를 만들어 평가를 용이하게 했습니다.

- **Technical Details**: TrialMind는 연구 질문 작성, 문헌 검색, 관련 연구 스크리닝, 주요 정보 추출, 그리고 임상 증거 합성을 포함한 전체 워크플로를 보조하기 위해 개발되었습니다. Task별로 LLM을 활용하는 기존 방법과 달리, TrialMind는 AI 파이프라인에 LLM을 통합하여 전문가의 작업을 분할 및 정렬함으로써 유연성을 유지합니다. 부울(BOOLEAN) 쿼리 생성을 통해 문헌 검색을 수행하고, 사용자가 요청하는 연구 프로토콜, 방법, 참가자 기준선 등을 비정형 문서에서 추출합니다.

- **Performance Highlights**: TrialMind는 2,000만 개 이상의 PubMed 연구 중에서 높은 재현율(0.897-1.000)을 달성하며, 기존 언어 모델 임베딩 기반 방법을 스크리닝에서 능가했습니다 (Recall@20 0.227-0.246 vs. 0.000-0.102). 결과 추출에서도 GPT-4 성능을 넘어서는 정확도(0.65-0.84)를 달성했습니다. 또한, 8명의 인간 주석자들이 선호한 결과에서 GPT-4 베이스라인을 62.5%-100%의 승률로 능가한 것으로 검증되었습니다. 



### Measuring and Benchmarking Large Language Models' Capabilities to Generate Persuasive Languag (https://arxiv.org/abs/2406.17753)
- **What's New**: 최근 대형 언어 모델(LLMs)의 설득적 언어 생성 능력을 조사한 연구가 발표되었습니다. 다양한 도메인에서 LLMs가 얼마나 설득적 텍스트를 생성할 수 있는지 측정하고 벤치마킹한 결과, 모델이 텍스트를 재작성하여 설득력을 증폭 또는 감쇄하는 능력을 분석하였습니다. 이를 위해 새롭게 구축된 데이터셋인 Persuasive-Pairs는 설득적 언어 표현을 증폭하거나 감쇄한 텍스트 쌍으로 구성되어 있습니다.

- **Technical Details**: Persuasive-Pairs 데이터셋은 짧은 텍스트와 이를 더 설득력 있게 또는 덜 설득력 있게 영어로 재작성한 텍스트 쌍으로 구성됩니다. 각 쌍은 신뢰도 높은 범주형 스케일로 다중 주석 Annotated되어, 설득적 언어의 정도를 나타냅니다. 이를 기반으로 새로운 LLMs의 성능을 측정하고 벤치마킹하기 위한 회귀 모델(regression model)을 훈련하였습니다.

- **Performance Highlights**: 이 모델을 통해 다양한 도메인에서 새로운 LLMs의 설득적 언어 생성 능력을 평가할 수 있습니다. 예를 들어, 시스템 프롬프트의 '페르소나'에 따라 동일한 텍스트를 단순히 패러프레이즈(paraphrase)할 때도 설득력의 정도가 크게 달라지는 현상을 발견했습니다. 이러한 발견은 LLM이 생성한 텍스트에서 설득적 언어를 연구하는 중요성을 강조합니다.



### Recite, Reconstruct, Recollect: Memorization in LMs as a Multifaceted Phenomenon (https://arxiv.org/abs/2406.17746)
- **What's New**: 기존의 언어 모델(언어 모델, LM)에서의 암기 현상 연구는 주로 단일한 관점에서 다루어졌습니다. 하지만 이번 연구에서는 암기를 다양한 요인들의 집합으로 보고, 이를 통해 예측 모델을 구성하는 방법을 탐구하였습니다.

- **Technical Details**: 연구진은 암기를 세 가지 유형으로 분류했습니다: 반복적으로 등장하는 시퀀스의 암기 (recitation), 예측 가능한 시퀀스의 재구성 (reconstruction), 그리고 드물게 등장하는 시퀀스의 회상 (recollection)입니다. 이를 통해 k-extractable 암기(정확히 32개의 토큰 후 시퀀스를 그대로 생성하는 경우)를 분석하고, Pythia 모델을 사용해 다양한 크기와 훈련 시간에 따라 암기 성향을 연구했습니다. 또한, 여러 요인들이 암기에 미치는 영향을 관찰했습니다. 예를 들어, 모델의 perplexity(혼란도) 값이 낮을수록 암기 확률이 높은 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 제안된 암기 분류 기준이 기존의 단순한 모델보다 예측 성능이 뛰어나다는 것이 입증되었습니다. 또한, 암기는 모델의 크기와 훈련 시간이 증가함에 따라 전체적으로 증가하는 경향을 보였으며, 특히 회상(Recollection)이 가장 빠르게 증가했습니다. 이는 드문 시퀀스의 자주 노출이나 무작위 암기만으로 설명할 수 없는 현상입니다.



### Following Length Constraints in Instructions (https://arxiv.org/abs/2406.17744)
Comments:
          13 pages

- **What's New**: 최신 연구는 길이 편향(length bias)을 극복하기 위해, 요구된 길이 제약을 포함한 지시사항(instructions)을 사용하여 모델을 훈련하는 방법을 제안합니다. 이 방법은 기존 GPT4, Llama 3, and Mixtral 같은 일반적인 지시사항 모델을 능가하여 고정된 길이의 답변을 생성할 수 있습니다.

- **Technical Details**: 길이 편향 문제를 해결하기 위해, 연구진은 길이 지시사항(length instructions)을 포함한 훈련 데이터를 사용하여 모델을 세부 튜닝(fine-tuning)하는 Length-Instruction Fine-Tuning (LIFT) 방법을 개발했습니다. 이 접근법에서는 Direct Preference Optimization (DPO) 기법을 활용해 모델을 훈련합니다. 기존의 AlpacaEval 2와 MT-Bench 벤치마크를 길이 지시사항을 포함하도록 수정해 모델을 평가했습니다.

- **Performance Highlights**: 제안된 방법인 LIFT-DPO를 적용한 Llama 2와 Llama 3 모델은 길이 제약을 따르는 성능에서 기존 모델들보다 훨씬 뛰어났으며, 길이 제한 위반이 훨씬 적음을 확인할 수 있었습니다. 특히 GPT4-Turbo 모델은 길이 제약을 50% 이상 위반하는 것에 비해, LIFT-DPO 모델은 높은 수준의 일관성 없는 길이를 가진 답변을 생성하지 않았습니다.



### Find Parent then Label Children: A Two-stage Taxonomy Completion Method with Pre-trained Language Mod (https://arxiv.org/abs/2406.17739)
- **What's New**: 본 논문에서는 새로운 두 단계 방법인 ATTEMPT를 소개합니다. 이 방법은 기존의 태스크들보다 더 효과적으로 태스크를 확장하고 원래의 분류 체계(원래 노드 포함)를 업데이트할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: ATTEMPT는 두 가지 주요 스테이지로 구성됩니다. 첫 번째 스테이지에서는 '텍사노미 경로 프롬프트와 사전 학습 모델' (Taxonomy-path Prompt with Pre-trained model, PPT) 방법을 사용하여 자연어를 생성하고 프롬프트 방법을 통해 로컬 정보와 숨겨진 지식을 잘 활용합니다. 두 번째 스테이지에서는 '다중 노드 라벨링' (Multiple Nodes Labeling, MNL) 방법을 활용하여 각 자식 노드를 식별하고 노드 간의 상호 의존성을 잘 활용하여 더 정확한 노드 유형 예측을 실현합니다.

- **Performance Highlights**: 두 개의 공개 데이터셋(6개의 도메인 포함)을 사용한 실험 결과, ATTEMPT는 아무리 전문적인 방법보다 더 높은 성능을 입증했습니다. 예를 들어, 리프노드(parent-finding)에선 정확도가 8.2% 향상되었고, 비리프노드(children-finding)에선 정확도가 21%, 평균 F1 점수가 20.3% 향상되었습니다. 전체적으로 ATTEMPT는 평균 F1 점수에서 다른 방법들보다 2.4% 우수한 성과를 보였습니다.



### LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users (https://arxiv.org/abs/2406.17737)
- **What's New**: 최근 대형 언어 모델(LLMs)의 우수한 성능에도 불구하고, 환각, 편향 등의 문제 있는 모델 동작에 대한 많은 연구가 진행되고 있습니다. 이번 연구에서는 영어 능력, 교육 수준, 출신 국가 등 사용자 특성에 따라 정보 정확성, 진실성, 거부 의향 등의 측면에서 LLM 응답의 품질이 어떻게 변하는지를 조사했습니다. 세 가지 최신 LLM과 두 가지 데이터셋을 실험한 결과, 영어 능력이 낮거나, 교육 수준이 낮거나, 미국 외 지역 출신 사용자가 상대적으로 더 많은 부정적인 영향을 받는 것을 관찰했습니다.

- **Technical Details**: 이번 연구에서는 영어 능력, 교육 수준, 출신 국가와 같은 세 가지 사용자 특성에 따라 LLM이 제공하는 정보의 정확성, 진실성, 거부 의향이 어떻게 변하는지를 조사했습니다. GPT-4(OpenAI), Claude Opus(Anthropic), Llama 3-8B(Meta)와 같은 세 가지 최신 대형 언어 모델을 TruthfulQA 및 SciQ 데이터셋을 사용하여 평가했습니다.

- **Performance Highlights**: {'Accuracy': '영어 원어민이 아닌 사용자, 교육 수준이 낮은 사용자, 미국 외 국가 출신 사용자는 정보 정확도와 진실성이 상대적으로 낮았습니다. 특히, 영어 원어민이 아닌 저학력 사용자는 가장 큰 정확도 하락을 보였습니다.', 'Refusal Rate': 'LLMs은 특정 사용자들에게는 정보 제공을 거부하는 경향이 더 높았으며, 사용자의 출신 국가와 교육 수준에 따라 거부율이 달라졌습니다.', 'Bias': '비영어권 사용자들을 대상으로 하는 실험에서 모델은 더 많은 오해를 생성하고, 정보를 숨기거나 후려치는(condecending) 응답을 생성하는 경향이 있었습니다.'}



### ViANLI: Adversarial Natural Language Inference for Vietnames (https://arxiv.org/abs/2406.17716)
- **What's New**: 이 논문은 한국어 자연어 추론(NLI) 연구를 위해 ViANLI라는 적대적 NLI 데이터셋을 소개합니다. 이는 베트남어를 대상으로 한 새로운 NLI 데이터셋으로, 약 10,000개의 전제-가설 쌍을 포함하고 있습니다. 인간과 모델이 함께 데이터를 구축하는 절차를 통해 다양한 언어적 도전과 실제 시나리오를 반영한 강력하고 일반화 가능한 데이터셋을 제공합니다.

- **Technical Details**: 이 데이터셋은 기존 비적대적 데이터셋과는 달리 모델의 약점을 노출시키기 위해 의도적으로 어려운 예시를 포함하고 있습니다. 인간과 모델이 함께 데이터를 생성하는 절차를 통해 데이터의 품질을 보장하고, 반복적인 피드백을 통해 모델의 성능을 계속해서 개선합니다. 실험에는 mBERT, XLM-R, Info-XLM, PhoBERT와 같은 다양한 언어 모델들을 사용했으며, XLM-RLarge 모델이 개발 및 테스트 세트에서 49% 이하의 정확도를 기록했습니다.

- **Performance Highlights**: ViANLI 데이터셋에서 가장 성능이 좋은 모델조차도 테스트 세트에서 48.4%의 정확도만을 기록해 현재 SOTA 모델들에게 큰 도전 과제를 제공합니다. 또한, 이 데이터셋을 사용해 훈련된 모델들은 다른 베트남어 NLI 데이터셋에서도 성능이 크게 향상되었습니다.



### From Distributional to Overton Pluralism: Investigating Large Language Model Alignmen (https://arxiv.org/abs/2406.17692)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 정렬(alignment) 과정이 출력 분포에 미치는 영향을 분석하였습니다. 특히, 정렬 후 응답의 다양성(reduction in response diversity) 감소 문제를 재검토하고, 정렬된 모델(aligned model)에서 기본 모델(base model)에서 얻을 수 없는 정보를 표출하는 지 여부를 조사했습니다.

- **Technical Details**: 연구진은 정렬 과정이 불필요한 내용을 억제하고 정보 통합(aggregation)을 통해 응답을 더욱 길고 다양한 정보를 포함하도록 한다고 결론지었습니다. 이를 통해 응답의 다채로움이 감소한 것처럼 보이지만, 실제로는 유용한 정보를 억제하지 않는다고 밝혀졌습니다. 또한, 정렬된 모델의 행동은 기본 모델에서 추가 튜닝 없이도 인-컨텍스트 예제(in-context examples)와 낮은 해상도의 의미적 힌트(semantic hints)를 통해 복구할 수 있다는 사실을 발견했습니다.

- **Performance Highlights**: 본 연구는 현재의 정렬 기법이 기본 LLM의 도움에 특화된 행동을 포착하지만 이를 확장하지는 않는다는 결론을 내렸습니다. 이 결과는 '표면적 정렬 가설(Superficial Alignment Hypothesis)'을 지지합니다. 또한, 추가 튜닝 없이도 정렬된 LLM을 모방할 수 있는 인-컨텍스트 정렬 전략이 효과적이라는 것을 보여줍니다.



### VarBench: Robust Language Model Benchmarking Through Dynamic Variable Perturbation (https://arxiv.org/abs/2406.17681)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 사전 훈련 중 벤치마크 데이터 유출(data leakage) 문제를 해결하기 위해 변수를 도입한 새로운 벤치마크 평가 방법을 제안합니다. GSM8K, ARC, CommonsenseQA, TruthfulQA 네 가지 데이터셋에 변수를 도입하여 평가할 때마다 새로운 테스트 사례를 생성함으로써 데이터 오염 문제를 효과적으로 해결합니다.

- **Technical Details**: 기존의 고정된 벤치마크는 LLM 훈련 데이터를 오염시킬 위험이 있습니다. 이를 해결하기 위해 우리는 '변수 혼란(variable perturbation)' 접근 방식을 제안합니다. 각 테스트 사례에서 변수를 추출하고 이를 통하여 새로운 값 범위를 정의하여 고유한 테스트 사례를 생성합니다. 이 방법은 GSM8K, CommonsenseQA, TruthfulQA, ARC 데이터셋에 적용되었습니다.

- **Performance Highlights**: 변수 혼란 방법을 통해 기존 LLM이 GSM8K 데이터셋에 대한 훈련을 유추할 수 있음을 확인했습니다. 이 접근 방식은 LLM의 실제 성능을 정확하게 평가하는 데 기여하며, 기존 모델의 데이터 오염 문제를 효과적으로 억제합니다.



### Quantifying AI Psychology: A Psychometrics Benchmark for Large Language Models (https://arxiv.org/abs/2406.17675)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 인간과 유사한 심리적 속성을 가지고 있는지, 이러한 속성이 안정적인지 평가하기 위한 새로운 프레임워크를 제시합니다. 이를 위해 심리적 차원의 식별, 평가 데이터셋 구성, 결과 검증을 포함한 심리 측정 벤치마크를 도입하였습니다. 이 벤치마크는 성격, 가치, 감정, 마음 이론, 동기, 지능의 여섯 가지 심리적 차원을 다루며, 13개의 다양한 시나리오와 항목 유형을 포함한 데이터셋으로 구성됩니다.

- **Technical Details**: 제안된 프레임워크는 심리학 이론에서 영감을 받아 심리적 속성을 평가하는 방법을 채택했습니다. 이를 위해 대규모 언어 모델(LLMs)을 평가 대상으로 삼아 심리 측정 벤치마크를 구성하였습니다. 벤치마크는 여섯 가지 심리적 차원을 포함하며, 13개의 데이터셋을 통해 다양한 시나리오와 항목 유형을 평가합니다. 주요 과제는 자가 보고된 특성과 실제 행동 사이의 불일치를 발견하는 것이며, 이를 통해 신뢰할 수 있는 평가와 향후 AI 및 사회 과학 응용 프로그램에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 연구 결과, LLMs는 다양한 심리적 속성을 나타내며, 유사한 상황에서 일관된 행동을 보였다. 예를 들어, 이론적 사고와 감정 지능 과제에서 일관된 성과를 보였지만, 선호 기반 질문에서는 모델 간의 변동이 큰 것으로 나타났습니다. 또한, 폐쇄형 질문과 개방형 질문에서의 응답 차이, 위치 편향과 프롬프트 민감성, LLM-as-a-judge 접근법의 신뢰성 등 다양한 평가 시나리오에서의 변동성이 밝혀졌습니다. 이번 연구를 통해 AI의 책임감 있는 개발 및 사회 과학 연구에 중요한 시사점을 제공하며, 향후 헬스케어, 교육, 고객 서비스 등 다양한 분야에서 LLMs의 활용 가능성을 제시합니다.



### LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic (https://arxiv.org/abs/2406.17663)
- **What's New**: LLM-ARC는 LLM과 자동 추론 평론가(ARC)를 결합한 신경-기호적(neuro-symbolic) 프레임워크로, 대형 언어 모델의 논리적 추론 능력을 강화합니다. 이 프레임워크는 코드 생성과 테스트를 담당하는 '액터(Actor)'와 이를 평가하고 피드백하는 '크리틱(Critic)'을 조합한 액터-크리틱 방법론을 사용하여, FOLIO 벤치마크에서 88.32%의 정확도를 달성했습니다.

- **Technical Details**: LLM-ARC는 선언적 논리 프로그램을 생성하고, 테스트를 통해 의미적 정확성을 검토하는 액터와, 코드를 실행하고 테스트를 수행하며 상세한 피드백을 제공하는 크리틱으로 구성됩니다. ASP(Answer Set Programming)를 사용하여 논리적 문제를 해결하며, FOLIO 벤치마크에서 복잡한 논리적 추론을 테스트합니다. 이 프레임워크는 자동화된 자기 보정 루프를 통해 성능을 향상시키며, 종단 간 대화 트레이스를 사용하여 훈련합니다.

- **Performance Highlights**: LLM-ARC는 88.32%의 새로운 SOTA(state-of-the-art) 정확도를 FOLIO 벤치마크에서 달성했습니다. 코드 품질을 향상시키기 위한 테스트 생성 옵션을 추가하여 6.6%의 성능 향상을 달성했으며, 액터가 크리틱의 피드백을 바탕으로 오류를 수정하는 자기 보정 루프를 실행했을 때 성능이 5% 더 향상되었습니다.



### Variationist: Exploring Multifaceted Variation and Bias in Written Language Data (https://arxiv.org/abs/2406.17647)
Comments:
          ACL 2024 (System Demonstrations)

- **What's New**: Variationist는 NLP 연구자, 언어학자, 사회과학자들이 언어 데이터의 품질과 편견을 조사하고 시각화하는 고도로 모듈식이며 확장 가능한 도구입니다. 텍스트의 다양한 단위와 변수 간의 연관성을 다양한 메트릭을 통해 분석할 수 있게 도와줍니다.

- **Technical Details**: {'Ease of use': 'Variationist는 다양한 분야의 연구자들이 접근할 수 있도록 설계되었습니다.', 'Modularity': '작은 빌딩 블록으로 구성되어 원하는 기능과 메트릭만 선택 가능하게 합니다.', 'Extensibility': '사용자가 원하는 토크나이저와 메트릭을 쉽게 추가할 수 있도록 설계되었습니다.', 'Core functionalities': ['데이터셋 입력: TSV, CSV 파일 또는 pandas dataframe 형태로 제공.', '입력 데이터의 텍스트 열 선택 및 분석: 두 개의 텍스트 열까지 처리 가능.', '언어 단위 설정: 문자, 단어, n-그램(n-grams) 등 다양한 단위 지원.', '변수 정의: 범주형, 순서형, 수치형, 좌표형 등의 다양한 변수 타입 및 의미해석.', '연관성 메트릭: PMI(Pointwise Mutual Information), TTR(Type-Token Ratio) 등 다양한 메트릭 제공.', '시각화 구성 요소: 최대 다섯 차원까지 인터랙티브 차트 생성 가능.']}

- **Performance Highlights**: Variationist는 다양한 연구 질문을 간편하게 답변하거나 언어 데이터 내 불필요한 연관성을 밝혀내는 데 도움을 줍니다. 컴퓨터 방언학(computational dialectology), 인간 레이블 변이(human label variation), 텍스트 생성(text generation) 연구에서 유용성이 입증되었습니다. Python 라이브러리, 코드, 문서 및 튜토리얼이 포함되어 있습니다.



### Banishing LLM Hallucinations Requires Rethinking Generalization (https://arxiv.org/abs/2406.17642)
- **What's New**: 이 연구에서는 기존의 대형 언어 모델(LLM)들이 발생시키는 '헛소리' 현상(환상, hallucinations)의 원인을 기존 이론이 충분히 설명하지 못한다고 주장합니다. 대신에, Lamini-1이라는 새로운 모델을 설계하여 이 문제를 해결하고자 했습니다. 이 모델은 수백만 개의 메모리 전문가(Mixture of Memory Experts, MoME)에 기반하여 사실을 저장하고 동적으로 검색합니다.

- **Technical Details**: 전통적인 접근법과는 다르게, LLM들이 무작위 숫자 데이터셋을 쉽게 기억할 수 있다는 것을 보여줍니다. 이 실험 결과를 바탕으로 단순한 신경망도 특정 임계점 이상의 훈련 손실에서 헛소리를 생성할 수 있다는 이론적 구조를 제공합니다. 이를 통해, LLM이 헛소리를 생성하는 원인은 외부 데이터 소스의 부족이나 편향 때문이 아님을 논증합니다.

- **Performance Highlights**: Lamini-1은 주요 사실을 기억할 때 약 100배 더 많은 확률적 경사 하강법(SGD) 단계를 요구합니다. 이 모델은 Lamini Memory Tuning 기법을 통해 헛소리를 제거하며, LLM이 훈련 데이터의 특정 키 사실을 거의 영에 가까운 손실로 학습할 수 있도록 합니다. 이를 통해 약 8개의 MI300X GPU를 사용하여 1시간 만에 새로운 수준의 사실 재현 성능을 달성합니다.



### Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels (https://arxiv.org/abs/2406.17633)
Comments:
          In Proceedings of the Sixth Workshop on Natural Language Processing and Computational Social Science

- **What's New**: 이번 연구는 CSS(Computational Social Science) 연구자들이 인간이 라벨링한 데이터를 생성할 필요 없이, generative large language models(LLMs)로 대체하는 방법을 평가하고자 합니다. 연구진은 이 LLM의 잠재력을 테스트하기 위해 14개의 분류 작업을 복제하고 성능을 측정했습니다. 이를 통해 LLMs가 생성한 라벨을 사용해 supervised text classification 모델을 미세 조정하면 인간이 생성한 라벨을 사용할 때와 비교해 유사한 성능을 낼 수 있음을 발견했습니다.

- **Technical Details**: 우선, GPT-4와 Mistral-7B 모델을 사용해 몇 가지 샘플에 대해 few-shot in-context learning을 수행했습니다. 그 후, 이 모델들이 생성한 라벨을 통해 BERT, RoBERTa, DistilBERT, XLNet 등의 supervised classifiers을 미세 조정했습니다. 이 과정에서 GPT-4가 생성한 라벨이 인간 라벨러가 제공한 것과 거의 동일한 성능을 냈음을 발견했습니다. 연구진은 특정 모델과 hyperparameter tuning 과정은 부록에 상세히 기술했습니다.

- **Performance Highlights**: GPT-4로 생성된 라벨을 사용한 supervised classifiers는 인간이 라벨링한 데이터를 사용한 모델 대비 성능 차이가 거의 없는 것으로 나타났습니다. 구체적으로는, GPT-4 라벨을 사용한 모델과 인간 라벨을 사용한 모델 간의 F1 성능 차이는 중앙값 기준 0.039였습니다. 또한, GPT-4 few-shot 모델과 supervised classifiers 간의 F1 성능 차이 중앙값은 0.006으로, GPT-4 few-shot 모델이 상대적으로 더 우수했습니다. 다만, GPT-4 기반 모델은 recall에서 더 뛰어난 반면 precision에서 약간 성능 저하가 있었습니다.



### CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreferenc (https://arxiv.org/abs/2406.17626)
Comments:
          Submitted to EMNLP 2024

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 안전성을 다중 턴 대화 참조(multi-turn dialogue coreference) 상황에서 평가한 최초의 시도입니다. 기존의 연구들은 단일 프롬프트 공격이나 목표 탈취(goal hijacking)에 중점을 두었으나, 이번 연구는 14개의 카테고리에서 1,400개의 질문을 포함하는 새로운 데이터셋을 생성하여 다중 턴 참조 안전 공격을 조사하였습니다.

- **Technical Details**: 연구팀은 다섯 개의 널리 사용되는 오픈 소스 LLM을 대상으로 다중 턴 참조 안전 공격에 대한 자세한 평가를 진행하였습니다. 평가된 모델들은 LLaMA2-Chat-7b 모델과 Mistral-7B-Instruct 모델 등이 포함되었습니다. 각 모델에 대해 다중 턴 참조 공격을 시도하여 안전성을 검증하였습니다.

- **Performance Highlights**: 평가 결과, LLaMA2-Chat-7b 모델은 다중 턴 참조 안전 공격에서 가장 높은 공격 성공률인 56%를 기록한 반면, Mistral-7B-Instruct 모델은 가장 낮은 13.9%의 공격 성공률을 보였습니다. 이는 다중 턴 대화 상호작용 시 LLM의 안전 취약성을 강조하는 결과입니다.



### Self-assessment, Exhibition, and Recognition: a Review of Personality in Large Language Models (https://arxiv.org/abs/2406.17624)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)에서의 성격 탐구에 대한 첫 번째 종합적인 리뷰를 제공하는 것을 목표로 합니다. LLM의 성격에 대해 자가 평가(self-assessment), 전시(exhibition), 인식(recognition)의 세 가지 연구 문제로 카테고리화하였습니다. 각 문제에 대한 상세 분석과 비교를 통해 연구의 주요 발견과 미해결 과제를 요약하고 있으며, 관련 학자와 개발자를 위해 광범위한 공개 자원을 수집하였습니다.

- **Technical Details**: 논문에서는 LLM의 성격을 평가하는 문제를 다음과 같은 세 가지 카테고리로 분류하였습니다: 
1. 자가 평가 (self-assessment)는 LLM의 내재된 성격 특성을 평가하는 것입니다.
2. 전시 (exhibition)는 LLM이 특정 성격을 보여주도록 제어하는 방법입니다.
3. 인식 (recognition)은 텍스트 내용을 통해 성격 특성을 식별하는 것입니다. 이 연구는 각 문제에 대한 세부적인 분석을 통해 연구 동기와 중요성을 설명하며, 제안된 방법들을 심층적으로 조사하고 비교합니다.

- **Performance Highlights**: 논문은 최신 성격 연구와 LLM 개발의 빠른 발전을 따라가는 데 있어 종합적인 방식으로 해결책을 제시합니다. 연구의 주제는 자가 평가, 성격 전시, 성격 인식으로 구분되었으며, 이와 관련된 다양한 방법론을 체계적으로 분류하고 분석하였습니다. 또한, 향후 연구 방향과 실질적인 응용 시나리오에 대한 논의도 포함되어 있습니다.



### "Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations? (https://arxiv.org/abs/2406.17600)
Comments:
          22 pages, 9 figures

- **What's New**: 이 연구는 소수의 전문가 레이블과 설명을 활용하여 대형 언어 모델(LLMs)이 인간 판단 분포(HJDs)를 근접하게 예측할 수 있는지를 탐구합니다. 이는 HJD를 확장하는 데 필요한 자원을 줄이는 해결책을 제공합니다.

- **Technical Details**: 기존 방법은 많은 수의 크라우드 워커로부터 HJD를 수집하거나, 소수의 전문가가 상세한 설명과 함께 레이블을 부여하는 방식이 있습니다. 이 연구에서는 LLMs가 제한된 수의 설명으로 HJD를 근접하게 예측할 수 있도록 합니다. 이는 소프트 레이블로 사용할 수 있으며, 모델의 미세 조정(fine-tuning)에도 적용해 보았습니다.

- **Performance Highlights**: 소수의 설명이 LLMs가 HJD를 예측하는 데 있어 놀라운 성능향상을 가져왔습니다. 그러나 LLM이 생성한 모델 판단 분포(MJDs)를 사용한 소형 모델의 미세 조정은 불일치한 결과를 보였습니다. 이는 저장소 거리 측정(sample-wise distance measures) 대신, 데이터셋 전반에 걸쳐 벡터 상관(distance correlation)을 사용하는 것이 모델 성능을 더 잘 예측할 수 있음을 보여줍니다.



### LongIns: A Challenging Long-context Instruction-based Exam for LLMs (https://arxiv.org/abs/2406.17588)
- **What's New**: 최근 몇 년간 대형 언어 모델(LLMs)의 장문(긴 문맥) 처리 능력에 대한 논의가 활발히 진행되고 있습니다. 이를 평가하기 위한 다양한 벤치마크가 등장했지만, 대부분 정보의 핵심을 추출하는 데 중점을 두면서 장문의 실제 이해 능력을 충분히 반영하지 못합니다. 이 문제를 해결하기 위해, 우리는 LongIns라는 새로운 벤치마크 데이터를 제안합니다. LongIns는 장문 지시 기반의 평가로, LLM의 긴 문맥 처리 능력을 철저히 테스트합니다.

- **Technical Details**: LongIns 벤치마크 세트는 기존의 지시 데이터셋을 기반으로 구축되었으며, 세 가지 평가 설정을 도입합니다: Global Instruction & Single Task (GIST), Local Instruction & Single Task (LIST), 그리고 Local Instruction & Multiple Tasks (LIMT). 이는 다양한 상황에서 LLM의 장문 처리 능력을 평가합니다. LongIns는 Super-NaturalInstructions (SNI)와 BIG-bench 데이터셋에서 다양한 질문을 수집하여 구성되었으며, 각각의 문맥 길이에 따라 1409개의 질문을 포함합니다.

- **Performance Highlights**: LongIns를 기반으로 여러 LLM을 평가한 결과, GPT-4와 같은 최고 성능의 모델도 16k 문맥 길이에서는 성능이 낮습니다. 많은 모델이 문맥 창 (context window) 길이가 짧을 때 (4k 이하) 다단계 추론 능력이 여전히 부족합니다. 또한, 대부분의 모델은 광고된 문맥 창 길이보다 짧은 실제 이해 가능한 문맥 길이를 가지고 있는 것으로 나타났습니다. 긴 문맥을 처리할 수 있는 모델의 성능 향상을 위해서는 여전히 많은 노력이 필요합니다.



### Beyond Text-to-SQL for IoT Defense: A Comprehensive Framework for Querying and Classifying IoT Threats (https://arxiv.org/abs/2406.17574)
- **What's New**: 이 논문은 IoT(Internet-of-Things) 데이터베이스에 대한 새로운 텍스트에서 SQL로 변환하는 데이터셋을 도입하고, 기존 연구가 SQL 문 생성에만 집중된 것을 넘어 반환된 데이터에 대한 새로운 정보를 추론하는 방향으로 확장하는 두 가지 주요 공헌을 합니다.

- **Technical Details**: 새로운 IoT 텍스트-SQL 데이터셋은 10,985개의 텍스트-SQL 쌍과 239,398개의 네트워크 트래픽 활동 행을 포함하고 있습니다. 이 데이터셋은 스마트 빌딩의 IoT 생태계에서 센서 판독값과 네트워크 트래픽 데이터를 탐색합니다. 또한 이 데이터셋은 두 단계의 처리를 허용하며, 여기서 반환된 데이터(네트워크 트래픽)는 악성인지 아닌지로 분류할 수 있습니다.

- **Performance Highlights**: SQL을 생성하고 데이터에 대한 정보를 추론하는 훈련을 공동으로 수행한 결과가 텍스트-SQL 시스템의 전체 성능을 향상시키는 것으로 나타났습니다. 또한 현존하는 대형 언어 모델들(e.g., GPT3.5)이 반환된 데이터에 대한 새로운 정보를 추론하는 데 어려움을 겪고 있음을 보여주었습니다. 이러한 이유로 제안된 데이터셋은 복잡한 도메인 전용 추론을 통합하는 데 있어 새로운 시험대로 작용할 수 있습니다.



### FrenchToxicityPrompts: a Large Benchmark for Evaluating and Mitigating Toxicity in French Texts (https://arxiv.org/abs/2406.17566)
Comments:
          TRAC-2024, Fourth Workshop on Threat, Aggression and Cyberbullying. 20 May 2024

- **What's New**: 대형 언어 모델(LLMs)의 사용이 증가함에 따라, 이들이 생성하는 언어에서 편향, 독성 혹은 유해한 내용이 포함될 가능성이 커지고 있습니다. 하지만 대부분의 연구들은 영어에 집중되어 있으며, 다른 언어에서도 이러한 문제를 고려하는 것이 중요합니다. 이를 해결하기 위해, 프랑스어로 된 50K의 자연 발생 프롬프트와 그 계속 부분(continuations)을 포함하는 FrenchToxicityPrompts 데이터셋을 만들고 공개했습니다.

- **Technical Details**: 이 데이터셋은 Perspective API를 사용해 독성 점수가 주석된 프롬프트들로 구성되어 있습니다. 우리는 네 가지 주요 오픈소스 LLM 패밀리에서 14개의 서로 다른 모델을 이용해 데이터셋을 평가했습니다. 평가한 모델들 중 일부는 GPT-4, GPT-3, BLOOM, LLaMa 등이 포함됩니다. 독성 검출을 위해선 문장들을 spacy 라이브러리로 분할하고, Detoxify 분류기를 사용해 전처리를 하였습니다.

- **Performance Highlights**: 대량의 텍스트 데이터를 사용하는 LLM들은 본래 독성을 포함한 텍스트를 포함할 수 있으며, 이를 재생산 할 위험성이 높습니다. 우리의 기여는 프랑스어 데이터에서 독성을 평가하고 미트하기 위한 새로운 데이터를 제공하는 것입니다. 우리의 연구를 통해 FrenchToxicityPrompts 데이터셋이 다양한 축에서 잠재적인 독성을 식별하는 데 유용함을 보여주었습니다.



### Multi-property Steering of Large Language Models with Dynamic Activation Composition (https://arxiv.org/abs/2406.17563)
- **What's New**: 이번 연구는 기존의 Activation Steering 방법이 주로 단일 조건 속성 및 인공적인 설정에 한정되어 평가된다는 점을 확장하여, 다중 조건 속성에 대한 포괄적인 평가를 수행했습니다. 다양한 Activation Steering 전략의 특징을 강조하고, 최적의 매개변수가 속성별로 다를 수 있다는 점을 밝혀내었습니다. 또한, 이러한 문제를 해결하기 위해 다중 속성에 대한 Steering 강도를 동적으로 조절하는 **Dynamic Activation Composition** 방법을 제안했습니다.

- **Technical Details**: 이 연구에서는 **Activation Steering** 방법을 사용하여 차별적인 입력 데모쌍을 통해 중간 표현을 생성하고, 이를 합성하여 모델의 예측 과정을 조절합니다. 생성 과정에서 각 단계마다 Steering 벡터의 정보를 사용하여 원하는 속성의 강도를 조절하면서도 모델의 유창성을 유지하는 방법을 고안했습니다. 이 과정에서 정보 이론적 접근법을 적용해 Steering 강도를 동적으로 조절하는 **Dynamic Activation Composition**을 도입했습니다.

- **Performance Highlights**: 다중 속성에 대한 Steering 실험을 통해, 제안한 방법이 모든 선택된 속성에 대해 강력한 조건을 유지하면서도 생성 유창성을 최대한 유지한다는 것을 확인했습니다. 특히 안전성, 형식성, 언어 속성에 대한 조건화를 평가했으며, **Dynamic Activation Composition** 방법을 통해 속성 별 최적의 조건 설정을 가능하게 했습니다.



### The FineWeb Datasets: Decanting the Web for the Finest Text Data at Sca (https://arxiv.org/abs/2406.17557)
- **What's New**: 이번 연구에서는 FineWeb이라는 새로운 15조 토큰의 데이터셋을 소개합니다. 이 데이터셋은 96개의 Common Crawl 스냅샷을 기반으로 제작되었으며, 기존의 공개된 프리트레이닝 데이터셋보다 성능이 뛰어난 LLM을 만들 수 있습니다. 또한, 교육 텍스트를 필터링한 데이터셋인 FineWeb-Edu도 함께 발표되었습니다. FineWeb-Edu는 1.3조 토큰으로 구성되어 있으며, 지식과 추론 능력이 요구되는 벤치마크 테스트(MMLU, ARC)에서 뛰어난 성능을 보입니다.

- **Technical Details**: FineWeb의 제작 과정에서는 deduplication(중복 제거) 및 filtering(필터링) 전략에 대해 깊이 있는 조사가 이루어졌습니다. FineWeb은 기존의 50개 이상의 후보 필터 중 효과적인 필터를 선택하여 튜닝하였습니다. FineWeb-Edu는 커스텀 분류기를 통해 높은 교육적 가치를 가진 텍스트를 필터링하여 만들어졌습니다. 이와 함께 데이터셋 제작을 위한 데이터 큐레이션 코드베이스도 함께 공개되었습니다.

- **Performance Highlights**: FineWeb과 FineWeb-Edu를 사용하여 프리트레이닝된 모델은 다른 공개된 웹 기반 프리트레이닝 데이터셋을 사용한 모델보다 뛰어난 성능을 보였습니다. 특히, FineWeb-Edu로 프리트레이닝된 모델들은 MMLU 및 ARC와 같은 지식과 추론에 고도의 능력을 요구하는 벤치마크 테스트에서 상당히 높은 성능을 나타냈습니다.



### Retrieval-Augmented Code Generation for Situated Action Generation: A Case Study on Minecraf (https://arxiv.org/abs/2406.17553)
Comments:
          under review

- **What's New**: 이번 연구는 Minecraft 공동 건축 과제에서 건축가(A)와 건축자(B)의 협업을 통해, 자연어 명령을 기반으로 행동을 예측하는 모델을 개발하는 것이다. 특히, 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 건축자의 행동을 예측하는 것을 시도하였다. 소수의 예제(few-shot prompting)를 통해 모델의 성능을 크게 개선하였다.

- **Technical Details**: Minecraft 공동 건축 과제에서 건축가(A)는 건축자(B)에게 3D 블록을 사용하여 특정 구조를 조립하는 지시를 내린다. 본 연구는 LLMs의 컨텍스트 학습 능력을 활용하여 코드 생성 작업으로서 행동 예측 작업을 모델링하였다. 이와 관련하여 대화 데이터셋을 코드를 포함한 형식으로 변환하였고, 몇 가지 샘플을 통해 모델을 훈련시켰다. 실험에서 pretrained all-MiniLM-L6-v2 모델을 사용하여 문장 유사성을 계산하였고, GPT-4를 포함한 여러 LLM 모델을 평가에 사용하였다.

- **Performance Highlights**: 실험 결과, GPT-4는 최고의 성능(F1-Score: 0.39)을 보였고, 그 다음으로 Llama-3-70b (F1-Score: 0.33)가 뒤를 이었다. 또한, Llama-3-8b 모델을 미세 조정한 결과, 약 6%의 성능 향상을 보였다. 그러나 아직도 과제의 상한선은 낮은 상태이다.



### Disce aut Deficere: Evaluating LLMs Proficiency on the INVALSI Italian Benchmark (https://arxiv.org/abs/2406.17535)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 인간 언어를 생성 및 조작하는 능력이 크게 향상되었습니다. 이 논문은 LLM들을 영어 외 다른 언어에서도 평가하는 중요성을 강조하면서, 이탈리아의 교육 평가 체계인 INVALSI 테스트를 자동화 평가 벤치마크로 적응하는 연구를 소개합니다.

- **Technical Details**: 우리는 INVALSI 테스틀 LLM 평가를 위해 자동화된 형식으로 적응시켰습니다. 이 과정에서 원래의 테스트 형식을 유지하면서 자동화 처리가 가능하도록 엄격한 적응 과정을 거쳤습니다. INVALSI 테스트는 읽기 이해, 문법 지식, 어휘 능력 등을 평가합니다. 연구 결과는 현재의 LLM들의 성능을 인간 결과와 비교해 시각적으로 제공합니다.

- **Performance Highlights**: 현재 LLM들의 성능을 자세히 평가하여 학술 커뮤니티에 중요한 참조점을 제공합니다. 또한 연구자들이 지속적으로 모델을 제출하여 평가할 수 있도록 권장하여, 이 벤치마크가 현재와 미래에도 가치 있는 자원으로 유지되도록 하고 있습니다.



### Retrieval-style In-Context Learning for Few-shot Hierarchical Text Classification (https://arxiv.org/abs/2406.17534)
Comments:
          17 pages

- **What’s New**: 이 논문에서는 few-shot 상황에서 HTC (Hierarchical Text Classification)를 더 효과적으로 수행할 수 있는 첫 번째 In-Context Learning(ICL) 기반 프레임워크를 소개합니다. 이 프레임워크는 large language models(LLMs)을 기반으로 하여, 대규모의 계층적 레이블 세트와 모호한 레이블들로 인해 HTC에 적합하지 않았던 기존의 ICL 접근법을 보완합니다.

- **Technical Details**: 이 논문에서 제안된 방법은 학습 데이터베이스를 사용하여 적절한 데모들을 검색하고, 다층의 계층적 레이블을 관리하기 위한 반복적 정책을 도입합니다. 특히, 레이블-인지형 표현(label-aware representations)을 위해 continuous training을 수행하며, 마스킹된 언어 모델링(Masking Language Modeling, MLM), 계층적 텍스트 분류(Classification, CLS), 인접 레이블 사이의 표현 차이를 극대화하는 새로운 분산 대비 학습(Divergent Contrastive Learning, DCL) 목표를 사용합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서 실험한 결과, 기존 방법들에 비해 우수한 성능을 보여주었으며 few-shot HTC에서 최첨단(SOTA) 결과를 달성했습니다. micro-F1 및 macro-F1 지표로 모델 성능을 측정했으며, 이 방법이 다양한 shot을 가진 few-shot HTC에서 개선된 성능을 제공합니다. 추가로, 이 논문은 질적 분석을 통해 방법론의 전반적인 이해에 기여합니다.



### LumberChunker: Long-Form Narrative Document Segmentation (https://arxiv.org/abs/2406.17526)
- **What's New**: LumberChunker는 문서 세분화(segmentation)를 위해 LLM을 활용해 동적으로 문서를 분할하는 새로운 방법을 제안합니다. 이 방법은 문서의 의미적 독립성을 더 잘 포착할 수 있도록 내용 전환점을 식별하도록 LLM을 반복적으로 사용하는 것이 특징입니다. 또한, 새롭게 만들어진 GutenQA 벤치마크를 통해 이 방법의 성능을 평가합니다.

- **Technical Details**: LumberChunker는 문서를 동적으로 세분화하기 위해 LLM을 사용합니다. 초기에는 문서를 단락별로 분할하고, 그런 다음 시퀀스의 각 단락을 분석해 내용이 전환되는 지점을 식별합니다. 설정된 토큰 수 임계값(θ)을 넘는 그룹을 만들어 이를 기반으로 세분화 지점을 식별합니다. 전체 프로세스는 LLM(Gemini 1.0-Pro)을 이용해 실행됩니다. 또한, 100권의 공개 도메인 서적으로 구성된 GutenQA 벤치마크에서 3000개의 고품질 QA 쌍을 생성하여 성능을 평가합니다.

- **Performance Highlights**: LumberChunker는 가장 경쟁력 있는 기준점보다 7.37% 높은 검색 성능(DCG@20)을 보여줍니다. 또한, RAG 파이프라인에 통합되었을 때 다른 세분화 방법들과 비교하여 더 효과적임을 입증했습니다. 특히, DCG@20 및 Recall@20에서 가장 높은 점수를 기록하여 탁월한 성능을 입증했습니다 (DCG@20에서 62.09, Recall@20에서 77.92).



### Entropy-Based Decoding for Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2406.17519)
- **What's New**: 본 연구에서는 외부 지식 검색을 이용한 대형 언어 모델(LLM)의 응답 정확성을 개선하는 새로운 디코딩 방법을 제안합니다. 이 방법은 엔트로피를 활용하여 트레이닝 없이 수행되며, 외부 지식과 내부 패러메트릭 지식으로부터 노이즈에 의해 발생하는 'distractibility issue'를 해결하는 데 목적이 있습니다.

- **Technical Details**: 제안된 방법은 엔트로피 기반의 문서 병렬 앙상블 디코딩(entropy-based document-parallel ensemble decoding) 기법을 사용하여 각 문서에서 낮은 엔트로피 분포를 우선시함으로써 유용한 정보를 추출합니다. 또한, 모델의 내부 지식을 대상으로 한 대조 디코딩(contrastive decoding) 메커니즘을 포함하여 외부 지식에 대한 신뢰도를 높입니다. 이를 통해 외부 컨텍스트와 패러메트릭 지식 모두에서 발생하는 노이즈를 줄입니다.

- **Performance Highlights**: 여러 오픈 도메인 질문 응답 데이터셋에서 제안된 방법의 우수성이 입증되었습니다. 제안된 방법은 네 가지 다양한 질문 응답 과제(NQ, TriviaQA, WebQ, PopQA)에서 뛰어난 성능을 발휘하였으며, 각 구성 요소의 효과가 확인되었습니다.



### Benchmarking Mental State Representations in Language Models (https://arxiv.org/abs/2406.17513)
Comments:
          ICML 2024 Workshop on Mechanistic Interpretability

- **What's New**: 이번 연구는 언어 모델(LMs)의 내부에서 인간의 정신 상태를 어떻게 재현하는지를 광범위하게 벤치마킹했습니다. 다양한 모델 크기, 미세 조정(fine-tuning) 접근 방식, 프롬프트(prompt) 변형을 포함한 실험을 통해 이러한 재현의 견고성과 기억화 문제를 조사했습니다.

- **Technical Details**: 이 연구는 Llama-2, Pythia라는 두 가지의 언어 모델 가족을 대상으로 하여 각기 다른 모델 크기(70만에서 700억 파라미터 범위)를 실험했습니다. 또한, 단순한 다음 토큰 예측(next token prediction)으로 사전 학습된 모델과 지시 조정(instruction-tuning)과 인간 피드백에서의 강화 학습(RLHF)을 사용하여 미세 조정된 모델의 성능을 비교했습니다. 다양한 프롬프트 변형을 통해 모델의 내부 표현이 쉽게 변할 수 있는지를 탐구했으며, 이에 대한 민감성을 처음으로 조사했습니다.

- **Performance Highlights**: 모델 크기와 미세 조정을 통해 다른 사람의 신념에 대한 모델의 내부 표현이 향상됨을 확인했습니다. 프롬프트 변형이 모델의 표현에 민감하게 작용함을 발견했으며, 기억화 문제는 크지 않다는 결론을 도출했습니다. 또한, 프로브(probes) 훈련 없이 활성화 편집(contrastive activation addition)을 통해 모델의 추론 성능을 향상시킬 수 있는 잠재력을 보여줬습니다.



### MedCare: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation (https://arxiv.org/abs/2406.17484)
Comments:
          19 pages, 6 figures

- **What's New**: 최근 발표된 논문에서는 대규모 언어 모델(LLMs)의 의학 분야 응용에 대한 새로운 접근 방식을 제시합니다. 이 방법은 기존 모델들이 놓치는 부분인 정렬-필요 작업(alignment-required tasks)에 대한 성능을 보완하며, 이를 통해 20개 이상의 의학 작업에서 최첨단 성능(State-of-the-Art, SOTA)을 달성하는 새로운 MedCare 모델을 소개합니다.

- **Technical Details**: 이 논문은 점진적 미세 조정 파이프라인을 제안합니다. 두 단계로 구성된 이 파이프라인은 첫 단계에서 지식 수집기(Knowledge Aggregator)와 소음 수집기(Noise Aggregator)를 도입해 다양한 지식을 인코딩합니다. 두 번째 단계에서는 소음 수집기를 제거하고 정렬 모듈 정렬 모듈을 최적화하여 지식 손실을 최소화합니다. 각각 1.8B, 7B, 14B 크기의 MedCare 모델은 Qwen1.5 시리즈를 기반으로 만들어졌고, 지식 집적과 정렬 파인튜닝 간의 균형을 맞추는 혁신적인 기술을 사용합니다.

- **Performance Highlights**: MedCare 모델은 정렬-필요 작업과 지식 집적 작업 모두에서 뛰어난 성능을 보여줍니다. 20개 이상의 다양한 의학 작업에서 기존 모델 대비 큰 개선을 이루었으며, 이 모델의 성능은 다양한 크기를 가진 모델에서도 일관되게 우수함을 입증했습니다. 또한 MedCare는 실질적인 적용 가능성을 입증하여 의학 분야에서의 실용적인 LLM의 가능성을 제시합니다.



### Transformer-based Named Entity Recognition with Combined Data Representation (https://arxiv.org/abs/2406.17474)
Comments:
          14 pages, 6 figures

- **What's New**: 이 연구는 트랜스포머(Transformer) 기반 모델의 명명 엔터티 인식(NER) 과제에서의 효과를 조사합니다. 다양한 데이터 표현 전략을 조사하여, 각각 하나의 문장(single), 여러 문장(merged), 그리고 문맥과 함께 결합된 문장(context)을 벡터로 사용합니다. 연구 결과 하나의 전략으로만 모델을 훈련하면 다양한 데이터 표현에서 성능 저하가 발생할 수 있다는 것을 발견했습니다. 이를 해결하기 위해 세 가지 전략을 모두 활용한 결합 훈련 절차를 제안하여 모델의 안정성과 적응성을 향상시킵니다. 이 방법의 효과는 영어, 폴란드어, 체코어, 독일어의 다양한 데이터셋에서 검증되었습니다.

- **Technical Details**: 트랜스포머 기반 NER 모델은 주로 사전 학습된 언어 모델을 활용합니다. 이번 연구에서는 단일 문장, 문장 병합, 문맥 기반 표현 등 세 가지 데이터 표현 전략이 트랜스포머 기반 NER 모델의 성능에 미치는 영향을 조사하였습니다. 데이터 표현 전략의 조합을 통해 학습 데이터를 구성한 후, 모델의 학습 및 추론 과정에서 이러한 전략의 효용성을 확인하였습니다. 모델의 토크나이저에서 생성된 서브토큰(sequence of subtokens) 입력을 받아, IOB2(Inside, Outside, Begin) 형식에 따라 명명 엔터티 라벨을 예측합니다. 또한, 문맥 기반의 데이터 표현(Long dependencies) 이용이 성능 향상에 긍정적인 영향을 미치나, 이는 짧은 텍스트 조각 처리에서는 성능 저하가 발생할 수 있습니다.

- **Performance Highlights**: 제안된 결합 전략은 네 가지 언어(영어, 폴란드어, 체코어, 독일어)에 걸쳐 다양한 데이터셋에서 테스트되었습니다. 결합 훈련 절차를 통해 모델의 데이터 표현 방식에 따른 성능 안정성이 향상되었습니다. 특히, 문맥 정보를 포함한 문서 수준의 데이터뿐만 아니라 단일 문장 수준의 데이터에서도 우수한 성능을 보였습니다. 이러한 접근방식은 기존 단일 전략 대비 명명 엔터티 인식 모델의 성능을 더욱 강화하는 것으로 나타났습니다.



### Enhancing Tool Retrieval with Iterative Feedback from Large Language Models (https://arxiv.org/abs/2406.17465)
- **What's New**: 새로운 연구는 외부 도구를 활용하여 대형 언어 모델(LLM)의 능력을 향상시키고 확장하려는 '도구 학습(tool learning)'을 다룹니다. 특히, 이 논문은 LLM의 피드백을 통해 도구 검색을 개선하는 방법을 제안하며, 이를 통해 도구 검색과 도구 사용 모델 간의 불일치를 줄이고자 합니다.

- **Technical Details**: 논문에서는 사용자의 지시와 도구의 설명이 복잡하고 불일치하는 문제를 해결하기 위해 LLM의 피드백을 활용한 반복 피드백(iterative feedback) 방법을 제안합니다. 구체적으로, LLM이 사용자의 지시와 검색된 도구를 바탕으로 피드백을 제공하며, 이를 통해 도구 검색 모델의 이해도를 점진적으로 향상시킵니다. 이러한 과정은 여러 차례 반복되어 도구 검색 모델이 더욱 적합한 도구를 제공할 수 있도록 합니다.

- **Performance Highlights**: 논문에서 제안한 접근 방식은 도메인 내(in-domain) 및 도메인 외(out-of-domain) 평가에서 모두 우수한 성능을 보여줍니다. 이를 평가하기 위해 종합적인 도구 검색 벤치마크(TR-bench)를 구축했으며, 실험 결과 제안된 방법이 기존 방법들보다 뛰어난 성능을 입증했습니다.



### Improving Grammatical Error Correction via Contextual Data Augmentation (https://arxiv.org/abs/2406.17456)
Comments:
          Accepted as Findings of ACL 2024

- **What's New**: 본 논문에서는 문법 오류 수정(GEC)을 위한 새로운 합성 데이터 증강 방법을 제안합니다. 기존에는 주로 사전 학습 단계에서 사용되던 합성 데이터를 데이터가 제한된 미세 조정(fine-tuning) 단계에서도 효과적으로 활용하기 위한 방법론을 개발하였습니다. 제안된 방법론은 맥락적 증강(contextual augmentation)을 활용하여, 규칙 기반 대체와 모델 기반 생성을 결합함으로써, 더 일관된 오류 분포를 보장하고, 다양한 문맥을 생성해냅니다.

- **Technical Details**: 제안된 합성 데이터 생성 방법은 크게 세 단계로 나뉩니다. 첫째, ERRANT 도구를 사용하여 실제 말뭉치에서 오류 패턴을 추출하고, 이것들을 패턴 풀로 구성합니다. 둘째, 해당 패턴 풀을 기반으로 원본 데이터셋의 실제 오류 빈도를 반영한 샘플링을 통해, 모델이 맥락적 문장을 생성하도록 합니다. 이 과정에서는 GPT2 또는 LLaMA2-7b-chat 모델을 사용하여 실험을 진행하며, 모델은 주어진 패턴을 포함하는 문맥을 생성합니다. 마지막으로, 생성된 문장에서 규칙 기반 대체를 통해 잘못된 문장을 얻고, 이를 GEC 모델을 통해 재레이블링하여 잡음을 줄입니다.

- **Performance Highlights**: CoNLL14와 BEA19-Test 데이터셋에 대한 실험 결과, 제안된 증강 방법이 강력한 기존 방법들을 능가하며, 소수의 합성 데이터만으로도 최첨단 수준의 성능을 달성했습니다. 이러한 결과는 제안된 방법이 미세 조정 단계의 고품질 데이터 증강에 효과적임을 증명합니다. 코드와 모델은 Github에 공개될 예정입니다.



### Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain (https://arxiv.org/abs/2406.17453)
- **What's New**: 이번 아카이브 논문에서는 정보 획득 측면에서 중요한 질문을 생성하는 데 있어 성능이 떨어지는 대형 언어 모델(Large Language Models, LLMs)을 개선하는 방법을 제안합니다. 연구팀은 20 질문 게임에서 더 효율적이고 정보가 많은 질문을 생성하기 위해 LLAMA 2-CHAT 7B 모델을 기반으로 Direct Preference Optimization (DPO) 알고리즘을 적용했습니다. 결과적으로 정보 획득 기대치(Expected Information Gain, EIG)가 높은 질문을 생성하는 데 성공했습니다.

- **Technical Details**: 이 방법은 세 가지 단계로 구성됩니다: 1) 모델로부터 다수의 질문을 샘플링, 2) EIG 기준으로 질문을 평가, 3) 선호 최적화(Preference Optimization)로 훈련. EIG는 질문의 정보성을 측정하는 지표로, 질문이 가능한 항목 공간의 엔트로피를 얼마나 줄이는지 평가합니다. 실험에서는 모든 단계에 동일한 LLM, LLAMA 2-CHAT (7B)를 사용했으며, 이를 통해 '최적' 질문과 '비최적' 질문 쌍으로 구성된 데이터셋을 생성하고 이를 DPO로 훈련시켰습니다.

- **Performance Highlights**: 실험 결과, DPO 알고리즘은 20 질문 게임에서 모델의 성능을 눈에 띄게 향상시켰습니다. S@1(첫 대화에서 정답을 맞추는 비율)와 AQ(성공적인 대화에서 정답까지의 평균 질문 수) 모두에서 향상된 결과를 보였습니다. 특히, DPO는 비슷한 크기의 후보 세트(test sets)에서 +12.2%와 +10%의 S@1 향상, 각각 2.1과 2.3개의 평균 질문 수 감소를 보여주었습니다. 이는 LLAMA 2-CHAT 모델이 다양한 도메인에서도 정보가 많은 질문을 성공적으로 생성할 수 있다는 것을 증명합니다.



### Towards Probing Speech-Specific Risks in Large Multimodal Models: A Taxonomy, Benchmark, and Insights (https://arxiv.org/abs/2406.17430)
- **What's New**: 최근 대규모 멀티모달 모델(Large Multimodal Models, LMMs)은 멀티모달 정보 이해와 인간 사용자와의 상호작용에서 큰 성공을 거두었습니다. 그러나 음성 모달리티에서 고위험 상호작용을 감지하는 문제는 거의 탐구되지 않았습니다. 본 연구에서는 적대적 발언, 악의적인 모방, 고정관념적 편견의 세 가지 주요 범주를 포함한 음성 특유의 리스크 분류법을 제안합니다. 이 분류법을 바탕으로 현재의 LMM들이 이러한 위험 범주를 얼마나 잘 감지하는지를 평가하기 위해 소규모 데이터셋을 만들었습니다.

- **Technical Details**: 우리는 4개의 하위 카테고리(적대적 풍자, 성별, 나이, 인종 고정관념)를 포함하여 8개의 리스크 범주를 다루는 고품질 기본 전사 세트를 수동으로 생성했습니다. 이 세트는 GPT-4를 활용해 확장되고, 3명의 인간 주석자가 품질을 유지하기 위해 필터링했습니다. 이를 텍스트를 스피치(TTS) 시스템으로 변환하여 다양한 변조 음성을 생성했습니다. 이후 5개의 최신 음성 지원 LMM들을 평가했습니다.

- **Performance Highlights**: Gemini 1.5 Pro는 랜덤 기준보다 약간 높은 성능을 보였으며, WavLLM은 랜덤 추측보다 낮은 성능을 보였습니다. Qwen-Audio-Chat은 다양한 프롬프트 전략에서 보다 안정적인 성공 패턴을 보였으며, SALMONN-7/13B는 특정 프롬프트 구성에서 가장 잘 수행했습니다. 전반적으로 나이 고정관념 편견이 가장 어려운 범주로, 최고 성능도 랜덤 기준보다 약간 높았습니다.



### Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA (https://arxiv.org/abs/2406.17419)
Comments:
          We release our code and data publicly at this https URL

- **What's New**: 최근에서야 다중 문서 시나리오에서 대규모 언어 모델(LLMs)의 긴 문맥 모델링 능력을 현실적으로 평가하려는 'Loong' 벤치마크가 제안되었습니다. 기존 벤치마크는 실제 상황과는 거리가 먼 불필요한 노이즈 텍스트를 추가해 문서 길이를 늘렸지만, Loong는 모든 문서가 최종 답변에 필수적이라는 점에서 차별됩니다.

- **Technical Details**: Loong는 확장된 다큐멘트 질문응답(QA)을 통해 현실적인 긴 문맥 시나리오와 일치하는 네 가지 유형의 태스크(Spotlight Locating, Comparison, Clustering, Chain of Reasoning)를 소개합니다. 테스트 사례는 평균적으로 11개의 문서를 포함하며, 금융 보고서, 법적 사례, 학술 논문과 같은 실제 시나리오를 다룹니다.

- **Performance Highlights**: Loong는 다양한 문맥 길이와 태스크 난이도를 가진 입력 데이터를 제공해 현존하는 긴 문맥 LLM들의 모델링 능력을 세밀하게 평가합니다. 기존의 가장 강력한 LLM들도 Loong의 테스트에서는 고군분투하는 모습을 보여, 현 LLM들에는 여전히 개선의 여지가 많음을 시사합니다. 여러 실험 결과, 특화된 태스크 수행 시 Retrieval Augmented Generation (RAG)이 저조한 성과를 보이는 것으로 나타났습니다.



### Variable Layer-Wise Quantization: A Simple and Effective Approach to Quantize LLMs (https://arxiv.org/abs/2406.17415)
Comments:
          submitted to EMNLP, 15 pages, 10 figures, 4 tables

- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 각 레이어를 중요도에 따라 다양한 비트 수준으로 양자화하는 새로운 접근 방식을 소개합니다. 특히, 중요한 레이어는 더 높은 비트 정밀도로, 덜 중요한 레이어는 더 낮은 비트 수준으로 양자화하여 플로팅 포인트 양자화 수준에 도달할 수 있습니다. 이는 모델의 성능 저하를 최소화하면서 더 압축된 모델 크기를 달성하는 것을 목표로 합니다.

- **Technical Details**: 논문에서 제안한 양자화 방법은 다음과 같습니다: (1) 입력 임베딩과 출력 임베딩의 차이를 기반으로 레이어의 중요성을 측정하는 방법 (레이어 입력 수정; LIM)과 (2) 레이어의 가중치가 평균보다 훨씬 큰지를 측정하여 중요성을 추정하는 방법 (z-score 분포; ZD)을 사용합니다. 이 두 가지 방법을 통해 각 레이어의 중요도를 평가하고, 이를 바탕으로 레이어를 다양한 비트 수준으로 양자화합니다.

- **Performance Highlights**: 변수 레이어별 양자화 실험에서 얻은 주요 결과는 다음과 같습니다: (a) 제안한 순서를 사용하여 25-50%의 레이어가 낮은 양자화로 이동할 때 LLM 성능 저하는 미미하나, 특정 순서를 사용하지 않을 경우 5-10%만 이동해도 성능 저하가 발생합니다. (b) LLM을 낮은 비트로 양자화하는 것은 가지치기(pruning)보다 성능이 뛰어나지만, 극한 양자화(2-bit)에서는 가지치기가 더 나은 결과를 보여줍니다. (c) 크기가 크고 레이어가 많은 LLM일수록 작은 LLM에 비해 레이어별 양자화가 더 잘 작동합니다.



### Make Some Noise: Unlocking Language Model Parallel Inference Capability through Noisy Training (https://arxiv.org/abs/2406.17404)
Comments:
          11 pages, 6 figures

- **What's New**: 새로운 연구는 Make Some Noise (MSN)라는 훈련 프레임워크를 제안합니다. 이는 대형 언어 모델(supervised fine-tuning, SFT) 스테이지를 대체하여 모델의 초안을 생성하는 과정에서 노이즈를 도입해 디노이즈(denoising) 작업을 학습하게 합니다. 추가적인 모델 구조나 훈련 과정이 필요하지 않으며, 원래 작업 능력에 영향을 미치지 않으면서 병렬 디코딩 능력을 크게 향상시킵니다. 추가적으로, Jacobi 디코딩 전략을 이용하여 MSN 모델의 추론 속도를 더욱 개선하는 트리 기반 검색 강화 TR-Jacobi 디코딩 전략을 제안합니다.

- **Technical Details**: 제안된 MSN 훈련 방법은 SFT 스테이지에서 원인 언어 모델 디노이즈 작업을 포함시켜 모델의 디노이징 능력을 향상시킵니다. 이는 추론 단계에서 Jacobi 디코딩을 통해 무작위 노이즈 토큰의 반복적 삽입과 검증을 통해 가속을 달성합니다. TR-Jacobi 디코딩 전략은 노이즈 초기화 문제를 완화하고 추론 속도 비율을 개선하기 위해 트리 기반 검색을 활용합니다.

- **Performance Highlights**: 일반 및 코드 도메인에서의 실험은 MSN이 모델 성능을 저해하지 않으면서 추론 속도를 2.3배에서 2.7배까지 향상시킬 수 있음을 보여줍니다. 또한, Specbench 평가에서는 MSN 모델이 추가 구조 없이 SOTA 모델 수준의 가속 비율을 달성하는 것을 확인했습니다.



### Native Design Bias: Studying the Impact of English Nativeness on Language Model Performanc (https://arxiv.org/abs/2406.17385)
- **What's New**: 이번 논문은 대형 언어 모델(Large Language Models, LLMs)의 응답 품질이 사용자의 언어 배경에 따라 달라지는지를 조사합니다. 특히, 영어를 모국어로 사용하지 않는 사용자들이 더 낮은 품질의 응답을 받을 가능성이 높은지에 대해 탐구합니다. 연구 결과, 영어를 모국어로 사용하는 사용자와 아닌 사용자 간에 성능 차이가 나타났으며, 모델이 사용자의 출신을 인식할 때 이러한 성능 저하가 더욱 두드러졌습니다.

- **Technical Details**: 본 연구는 전 세계에서 모국어와 영어 능력을 포함한 정보를 가진 124명의 주석자가 생성한 12,000개 이상의 독특한 주석으로 구성된 새로운 데이터셋을 기반으로 합니다. 데이터셋에는 주관적 및 객관적 분류 작업과 생성 작업이 포함됩니다. 특히 영어를 주요 언어로 하는 서구 국가(미국, 영국, 캐나다 등)에서 온 사용자와 그렇지 않은 사용자 간의 성능 차이를 분석합니다. 추가로, 사용자의 모국어를 모델에게 알렸을 때 발생하는 강한 앵커링 효과도 다룹니다.

- **Performance Highlights**: LLMs는 영어를 모국어로 사용하지 않는 사용자에게 더 빈번하게 잘못된 정보나 부정확한 응답을 생성합니다. 또한, 영어를 구사하는 서구 국가의 사용자와 비교했을 때, 이런 성능 차이는 더욱 두드러지며, 모델이 사용자의 언어 배경을 알았을 때 성능 저하는 더욱 심화됩니다.



### A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens (https://arxiv.org/abs/2406.17378)
Comments:
          Work in Progress

- **What's New**: 대형 언어 모델(LLM)에서의 텍스트 임베딩(text embedding)이 입력 텍스트의 주요 토큰(key tokens)과 정렬될 수 있음을 발견했습니다. 이 현상은 모델 아키텍처, 학습 전략, 임베딩 방법에 관계없이 보편적으로 존재합니다. 이를 통해 정보 검색, 의미론적 텍스트 유사성 등 여러 분야에서 새로운 가능성을 열어줍니다.

- **Technical Details**: 본 연구에서는 8개의 임베딩 LLM을 분석하여 임베딩 LLM과 원래 생성 LLM 사이의 주요 변화가 첫 번째 주성분(principal component)에 집중된다는 점을 발견했습니다. 이 주성분을 조정함으로써 텍스트 임베딩을 주요 토큰과 정렬할 수 있습니다. 또한, 텍스트 임베딩이 decoder layer를 통과할 때 가장 높은 디코딩 확률을 가지는 토큰들이 입력 텍스트와 높은 관련이 있음을 확인했습니다.

- **Performance Highlights**: 새로운 sparse retrieval 방법을 제안하며, 이는 동일 모델의 dense retrieval 성능의 80%를 달성하면서도 계산량을 크게 줄였습니다. 또한, BM25 및 SPLADE v2 같은 강력한 베이스라인을 능가하는 성과를 보여주었습니다. 이 발견은 의미론적 유사성 및 지시 따르기(embeddiing) 기술을 이해하는 데 새로운 시각을 제공합니다.



### A Three-Pronged Approach to Cross-Lingual Adaptation with Multilingual LLMs (https://arxiv.org/abs/2406.17377)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 저자원 언어에 대한 적응 방법을 탐구합니다. 특히, Llama-2 모델을 이용해 벵골어, 힌디어, 타밀어 같은 인도 언어들에 대한 크로스-링구얼 학습(cross-lingual transfer)을 실험합니다.

- **Technical Details**: LLM인 Llama-2는 2조 개의 토큰으로 사전 학습되었으며, 이 중 인도 언어는 0.005% 미만을 차지합니다. 저자들은 세 가지 접근 방식을 탐구합니다: 1) 주요 언어의 추가적인 감독 신호를 통한 학습, 2) 목표 언어를 어휘 재배열(adapting target languages to word reordering)에 적응시키는 방법, 3) 저자원 언어로 계속된 사전 학습(continued pre-training)이 다른 관련 저자원 언어의 성능을 향상시키는 방식입니다.

- **Performance Highlights**: 1) 지도 신호 제공을 통해 주요 언어가 PEFT와 ICL 모두에서 개선되었습니다. 2) 목표 언어를 영어와 유사하게 만드는 masquerading 접근 방식은 ICL에서 약간의 이점을 제공하지만 PEFT에서는 효과가 감소했습니다. 3) 힌디어로 계속된 사전 학습을 통한 bridging 접근은 벵골어와 타밀어의 성능을 향상시켰습니다. 4) Handholding와 Bridging을 결합한 방식이 가장 큰 성능 향상을 보였습니다.



### An Empirical Study on the Characteristics of Bias upon Context Length Variation for Bangla (https://arxiv.org/abs/2406.17375)
Comments:
          Accepted in Findings of ACL, 2024

- **What's New**: 이번 연구에서는 기존의 성별 편향 측정 방법을 방글라(NLP의 저자)를 대상으로 확장하여 적용했습니다. 이를 위해 방글라어에 적합한 새로운 데이터셋을 만들고, 기존 편향 측정 방법을 방글라어에 맞게 조정하였으며, 문맥 길이 변동이 편향 측정에 미치는 영향을 조사했습니다. 이번 연구 결과는 방글라어의 편향 분석에서 문맥 길이에 따른 편향 메트릭스의 의존성을 명확히 보여주며, 향후 연구를 지원하기 위해 모든 연구 자원을 공개합니다.

- **Technical Details**: 연구는 WEAT(Word Embedding Association Test)와 SEAT(Sentence Embedding Association Test)를 사용해 기본 단어 임베딩과 문장 임베딩 시스템에서의 편향을 측정했습니다. 방글라2B+ 데이터셋을 사용하여 Word2Vec과 GloVe 모델을 트레이닝했습니다. 또한, CEAT(Contextual Word Embedding Association Test)를 사용해 문맥 임베딩의 편향을 측정했습니다. 연구는 문맥 길이가 편향 측정에 미치는 영향을 조사하기 위해 다양한 문맥 길이(l = 9, 25, 75, >75)로 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, 문맥 길이에 따라 편향 메트릭스가 다르게 나타나는 뚜렷한 의존 관계를 보였습니다. 특히, 문맥 길이 변화에 따른 편향 측정값의 변동성을 효과적으로 나타내는 무작위 효과(random effects) 모델을 사용해 신뢰성 있는 통계적 유의미성을 확인할 수 있었습니다. 더 나아가 방글라어 특성에 맞춘 템플릿 문장을 사용해 Mask Prediction 기반 접근법을 탐구하여 효과적인 편향 측정을 수행했습니다.



### Leveraging Synthetic Audio Data for End-to-End Low-Resource Speech Translation (https://arxiv.org/abs/2406.17363)
Comments:
          IWSLT 2024

- **What's New**: 새로운 IWSLT 2024의 시스템 제출은 아일랜드어에서 영어로의 음성 번역(end-to-end speech translation)을 다룹니다. 이를 위해 Whisper 모델을 사용하여 다양한 데이터 증강 기법(data augmentation techniques)을 도입하였습니다. 특히, speech back-translation과 noise augmentation 같은 기법을 적용하여 신호의 다양성을 높였습니다.

- **Technical Details**: 아일랜드어는 자원이 부족한 언어(low-resource language)로 분류되며, 말소리 및 언어 도구가 상당히 부족합니다. 따라서 데이터를 늘리기 위해 텍스트-음성 변환(TTS) 모델을 활용한 synthetic data 생성 기법을 사용했습니다. 해당 논문에서는 하천음성의 다양한 특성을 고려하여 noise augmentation과 음성 오디오 검출(Voice Audio Detection, VAD) 등의 데이터 처리 기법도 함께 사용되었습니다. 사용된 Whisper 모델은 아일랜드어-영어 음성 번역에 맞춰 맞춤 조정되었습니다.

- **Performance Highlights**: 실험 결과, 'speech back-translation'이 아일랜드어로부터 영어로 음성을 번역하는 엔드투엔드 모델의 성능을 현저히 향상시켰습니다. 총 196시간의 synthetic audio 데이터가 사용되었으며, 이는 잡음과 음성 분할 기술로 인해 더욱 다양해졌습니다. 본 연구에서 개발된 모델은 IWSLT 2024의 데이터셋을 활용하여 훈련되었으며, Whisper 모델을 통해 고품질 음성 번역 결과를 보여주었습니다.



### Dual-Space Knowledge Distillation for Large Language Models (https://arxiv.org/abs/2406.17328)
Comments:
          17 pages, 11 figures, code available at: this https URL

- **What's New**: 이번 작업에서는, 대형 언어 모델(LLMs)의 지식을 작은 모델로 이전해 압축하는 지식 증류(Knowledge Distillation, KD) 방법에 대한 새로운 프레임워크를 제안합니다. 이 새로운 프레임워크는 이중 공간 지식 증류(Dual-Space Knowledge Distillation, DSKD)로, 교사와 학생 모델의 출력 공간을 통합하여 공간 불일치 문제를 해결합니다. DSKD 기반으로 서로 다른 어휘를 가진 모델 간에도 KD를 지원하는 Cross-Model Attention(CMA) 메커니즘을 개발했습니다.

- **Technical Details**: 기존의 흰 상자 KD(white-box KD) 프레임워크는 교사와 학생 모델의 출력 분포가 각자 다른 출력 공간에서 나오기 때문에, 표현 및 분포 수준에서 두 모델 간 유사성이 낮아집니다. 특정 어휘를 공유해야 하는 요구 사항도 발생합니다. 이를 해결하기 위해, DSKD는 교사/학생 모델의 출력 숨김 상태(output hidden states)를 서로의 표현 공간으로 투영합니다. 이로 인해 공유된 예측 헤드(prediction heads)를 사용해 동일한 출력 공간에서 분포를 생성할 수 있습니다. 특히, 서로 다른 어휘를 가진 모델들에 대해서는 CMA 메커니즘을 통해 토큰을 자동으로 정렬합니다. DSKD는 KL divergence 등의 거리 함수와 호환됩니다.

- **Performance Highlights**: DSKD는 기존 흰 상자 KD 프레임워크 대비 다양한 거리 함수와 함께 사용할 때 유의미한 성능 향상을 보였습니다. 서로 다른 어휘를 가진 모델들을 위한 KD 방법에서도 기존 방법들을 크게 능가합니다. 실험 결과, 동일한 어휘를 가진 LLM들에서는 다양한 거리 함수에서 DSKD가 크게 우수성을 보였으며, 서로 다른 어휘를 가진 LLM들에서는 CMA를 통한 DSKD가 모든 기존 KD 방법들을 능가하는 성능을 나타냈습니다.



### Delving into the Utilisation of ChatGPT in Scientific Publications in Astronomy (https://arxiv.org/abs/2406.17324)
Comments:
          Submitted to SPAICE

- **What's New**: 최근 몇 년 동안 자연어 처리에서 머신러닝 접근 방식의 급격한 발전으로 인해 대형 언어 모델(LLMs)이 급격히 증가했습니다. 이러한 모델들은 많은 학술 논문 작성에 채택되었으며, 특히 천문학 분야에서도 그 채택이 급증하고 있는 것으로 보입니다. 이 연구는 ChatGPT가 생성한 단어들을 추출하여 천문학 논문에서의 사용 빈도를 평가하고 있습니다.

- **Technical Details**: 이 연구에서는 ChatGPT가 학술 텍스트를 생성할 때 인간보다 더 자주 사용하는 단어들을 식별하고, 이러한 단어들이 NASA Astrophysics Data System(ADS)에 기록된 천문학 논문에서 얼마나 자주 사용되는지 분석하였습니다. 분석 방법으로는 AI가 과도하게 사용하는 단어와 무작위로 선택한 단어 그룹을 비교하여 연도별 사용 빈도를 계산하는 방법을 사용하였습니다. 각 단어의 빈도수 변화와 Kolmogorov-Smirnov 테스트를 통해 통계적 유의미성을 평가했습니다.

- **Performance Highlights**: 주요 결과로, ChatGPT가 선호하는 단어들의 빈도는 2023년과 2024년에 급격히 증가한 것으로 나타났습니다. 특히 비래프리(peer-reviewed) 논문에서 이러한 변화가 더 두드러지게 나타났습니다. 이로 인해 천문학 논문 작성에서 LLMs의 광범위한 채택이 입증되었습니다.



### Not All Preference Pairs Are Created Equal: A Recipe for Annotation-Efficient Iterative Preference Learning (https://arxiv.org/abs/2406.17312)
- **What's New**: 이번 논문에서는 반복적 선호 학습(Iterative Preference Learning) 과정에서 비용 효율적인 어노테이션 전략을 제안합니다. 기존 방법들은 랜덤한 선택을 통해 어노테이션 데이터를 수집했으나, 본 연구에서는 불확실성과 분포 이동(distribution shifts)에 기반하여 DPO(Direct Preference Optimization)가 예측하는 암묵적 보상 마진을 비교하여 더 유익한 응답 쌍을 선택하는 방법을 제안합니다.

- **Technical Details**: 연구에서는 단일 반복(iteration) 및 다중 반복 시나리오 모두에서 작은 마진을 가지는 응답 쌍을 어노테이션하는 것이 더 나은 성능을 보인다는 것을 실험적으로 증명했습니다. 특히 초기 반복(iteration)들에서 더 많은 어노테이션 예산을 할당하는 것이 모델 성능 향상에 유리함을 발견하였습니다.

- **Performance Highlights**: 실험 결과, 가장 작은 마진을 갖는 응답 쌍을 선택하는 'always-smallest' 전략이 랜덤 선택 'always-random' 전략보다 반복 횟수가 증가할수록 지속적으로 성능을 향상시킨다는 것을 확인했습니다. 어노테이션 예산 할당에서도 'decrease' 전략이 'increase' 전략보다 더 나은 결과를 보였습니다.



### Retrieval Augmented Instruction Tuning for Open NER with Large Language Models (https://arxiv.org/abs/2406.17305)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 정보 추출(Information Extraction, IE)에 적용하는 새로운 방법인 RA-IT(Retrieval Augmented Instruction Tuning)를 제안합니다. 특히, 오픈 명명된 엔티티 인식(open named entity recognition, NER) 작업에 초점을 맞추어, 학습 샘플마다 의미적으로 유사한 예제를 추출해 원래의 지시에 앞서서 입력으로 사용합니다. 이 방법을 보다 철저하게 평가하기 위해 중국어 IT 데이터셋을 구축하고 영어와 중국어 시나리오에서 RA-IT를 평가하였습니다.

- **Technical Details**: RA-IT 접근 방식에서는 각 학습 샘플마다 의미적으로 유사한 예제를 학습 데이터셋에서 추출하여 문맥 강화 지시(context-enhanced instruction)를 구성합니다. 원래 대화 앞에 추출된 문맥을 미리 추가하여 정보 추출 성능을 향상시키고자 합니다. 우리는 문장 임베딩 기반의 검색을 활용하며, 주된 실험에서 코사인 유사도(cosine similarity)를 유사도 측도로 사용합니다. 실험을 위해 LLaMA-3-8B와 Qwen-1.5-7B 모델을 백본으로 사용하고 LoRA로 모델을 학습했습니다.

- **Performance Highlights**: RA-IT 접근 방식은 다양한 데이터 크기에서 일관된 성능 향상을 보여줍니다. 의미적으로 유사한 예제를 추출하는 것이 가장 큰 학습 이점을 제공하며 랜덤 추출도 개선을 보이지만 유사한 예제보다는 성능이 떨어집니다. 특히, 영어와 중국어 시나리오 모두에서 개선된 성능을 확인할 수 있었습니다. 주된 실험에서는 예제 없이 추론할 때 더 일관된 개선을 보여주었습니다. 이 결과는 문맥 강화 지시의 필요성을 시사합니다.



### Leveraging LLMs for Dialogue Quality Measuremen (https://arxiv.org/abs/2406.17304)
- **What's New**: 이 논문은 대형 언어 모델 (LLMs)을 이용한 자동 대화 품질 평가를 탐구합니다. 다양한 공개 및 독점 데이터셋에서 모델 크기, 컨텍스트 예제(in-context examples), 선택 기법(selection techniques) 등을 실험하며, 'chain-of-thought' (CoT) 추론 및 라벨 추출 절차를 검토합니다. 주요 발견으로는 더 큰 모델이 더 정확한 대화 라벨을 생성하고, 알고리즘적으로 선택한 컨텍스트 예제가 임의로 선택한 것보다 더 우수하며, CoT 추론이 성능을 향상시키며, 미세 조정된 LLMs가 기본 모델을 능가한다는 것입니다.

- **Technical Details**: 다양한 LLM 구성과 인스턴스 선택 기법을 적용하여 두 가지 벤치마크 데이터셋(공개 및 Amazon 내부 데이터셋)에서 실험을 진행했습니다. 'chain-of-thought' (CoT) 추론을 통해 LLM이 라벨 없이 설명과 이유를 먼저 제공하도록 요청하였으며, 모델 크기, 컨텍스트 예제의 알고리즘적 선택, CoT 추론 등을 통해 성능 향상을 확인했습니다. 또한, 미세 조정된 LLMs가 대화 평가 작업에서 성능을 크게 개선한다는 것을 증명했습니다.

- **Performance Highlights**: 1. 더 큰 모델이 더 정확한 대화 라벨을 제공
2. 알고리즘적으로 선택된 컨텍스트 예제가 임의로 선택된 것보다 더 나은 결과를 보임
3. LLM이 최종 라벨을 출력하기 전에 이유를 제공하는 CoT 추론이 성능을 향상시킴
4. 미세 조정된 LLMs가 기본 모델보다 뛰어난 성능을 나타냄



### CausalScore: An Automatic Reference-Free Metric for Assessing Response Relevance in Open-Domain Dialogue Systems (https://arxiv.org/abs/2406.17300)
- **What's New**: 오픈 도메인 대화 시스템의 응답 품질을 자동으로 평가하는 새로운 척도인 CausalScore를 도입했습니다. 기존의 평가 메트릭이 문법적으로 올바른 응답을 평가하는 데서 인간의 판단과 일치하지 않는 문제를 해결하고자, 이 새로운 메트릭은 대화 이력과 응답 간의 인과 강도를 측정함으로써 응답의 관련성을 평가합니다. 또한, 인간이 주석을 달아 인과 관계를 표시한 새로운 대화 데이터셋 CGDIALOG+도 수집했습니다.

- **Technical Details**: CausalScore는 대화 이력과 응답 사이의 무조건적 의존성과 조건부 의존성을 이용하여 인과 강도를 추정합니다. 이를 위해 무조건적 독립 분류기와 조건부 독립 분류기를 사용하여 대화 역사에서 주어진 응답에 통계적으로 의존하는 발화 subset을 식별합니다. 이후 조건부 의존성을 계산하며, 최종적으로 무조건적 및 조건부 의존성을 모두 집계하여 인과 강도를 추정합니다. CausalScore의 학습을 위해 새로운 데이터셋 CGDIALOG+를 구축했습니다.

- **Performance Highlights**: 실험 결과, CausalScore는 기존의 최첨단(SoTA) 자동 메트릭에 비해 인간의 판단과 더 강한 상관관계를 보여주었습니다. 다양한 상관관계 측정을 통해 CausalScore의 탁월한 성능이 입증되었습니다.



### Math-LLaVA: Bootstrapping Mathematical Reasoning for Multimodal Large Language Models (https://arxiv.org/abs/2406.17294)
Comments:
          8 pages

- **What's New**: 최근 대형 언어 모델(LLMs)이 텍스트 기반 수학 문제 해결에서 뛰어난 능력을 보였으나, 시각 정보가 풍부한 멀티모달 데이터셋이 부족한 실정입니다. 이를 해결하고자 MathV360K라는 새로운 멀티모달 수학 데이터셋이 소개되었습니다. 이 데이터셋은 24개의 기존 데이터셋에서 가져온 40K 이미지와 새로운 320K 문제-해답 쌍으로 구성되어, 멀티모달 수학 질문의 넓이와 깊이를 모두 향상시킵니다.

- **Technical Details**: MathV360K 데이터셋은 24개의 기존 데이터셋에서 40K의 고품질 이미지를 선택하고, 이를 기반으로 320K의 새로운 문제-해답 쌍을 합성하여 생성되었습니다. 이 데이터셋은 대수학(algebra), 산수(arithmetic), 기하학(geometry), 논리(logic), 숫자 상식(numeric commonsense), 과학(science), 그리고 시각적 질문 응답(visual question answering) 등 다양한 주제를 다룹니다.

- **Performance Highlights**: MathV360K로 튜닝된 새 모델인 Math-LLaVA는 MathVista의 minitest 분할에서 19점 증가의 성과를 기록하며, GPT-4V와 비교할만한 성능을 보였습니다. 또한, MMMU 벤치마크에서도 일반화 능력이 크게 향상된 점이 확인되었습니다. 이 새로운 모델은 기존 LLaVA-1.5 모델을 19% 능가하는 성과를 보였으며, 종합적으로 멀티모달 수학적 추론 능력을 크게 향상시켰습니다.



### Predicting the Big Five Personality Traits in Chinese Counselling Dialogues Using Large Language Models (https://arxiv.org/abs/2406.17287)
- **What's New**: 이번 연구는 상담 대화(counseling dialogues)에서 직접적으로 Big Five 성격 특성을 예측할 수 있는 새로운 프레임워크를 소개합니다. 전통적인 자기보고 설문지(self-report questionnaires) 방식의 한계를 극복하고 LLMs(Large Language Models)를 활용하여 더욱 효율적이고 객관적인 성격 평가를 목표로 합니다.

- **Technical Details**: 프레임워크는 역할극(role-play)과 설문지 기반의 프롬프트(questionnaire-based prompting) 방식을 통해, 상담 세션에서 얻은 데이터를 바탕으로 LLMs가 Big Five Inventory를 시뮬레이션하도록 조건을 설정합니다. 총 853개의 실제 상담 세션을 평가 대상으로 하였으며, 이 과정에서 역할극 시뮬레이션과 간단한 설문지를 통해 예측 정확도를 향상시켰습니다. Llama3-8B 모델을 직접 선호 최적화(Direct Preference Optimization, DPO) 및 감독된 미세 조정(Supervised Fine-Tuning, SFT)을 통해 튜닝하였습니다.

- **Performance Highlights**: LLM 예측과 실제 Big Five 특성 간의 상관관계가 유의미하게 나타났으며, 특히 Llama3-8B 모델이 Qwen1.5-110B를 36.94% 초과하며 130.95%의 성능 향상을 달성하였습니다. 이 연구의 코드와 모델은 공개되었으며, 앞으로의 연구에 중요한 도구가 될 것으로 기대됩니다.



### A Recursive Encoding for Cuneiform Signs (https://arxiv.org/abs/2406.17283)
Comments:
          27 pages, 29 figures, 5 tables

- **What's New**: 이 논문은 설형문자 교육에서 가장 큰 문제 중 하나인 미지의 기호를 찾아내는 과정에 대한 새로운 접근 방안을 제안합니다. 기존에는 기호 목록을 페이지별로 일일이 검색해야 했지만, 이 논문에서는 기호의 배열을 컴퓨터가 처리할 수 있는 방식으로 표현하는 '재귀적 인코딩(recursive encoding)'을 제안합니다.

- **Technical Details**: 재귀적 인코딩은 기호의 획 배열을 컴퓨터가 이해할 수 있는 형식으로 표현합니다. 이와 함께 새로운 알고리즘 (algorithms) 시리즈가 제안되어, 학생들이 특정 요소를 기준으로 기호를 찾거나, 기호와 서판(tablets)을 전자적으로 렌더링(rendering) 하는 새로운 방법을 제공합니다.

- **Performance Highlights**: 이 접근 방식은 특히 설형문자 교육에서 기호를 신속하게 찾는 문제를 크게 해결하고, 기호와 서판을 전자적으로 렌더링하는 방식의 혁신을 가져올 것으로 기대됩니다.



### BERT, Neural Information Retrieval, Boolean Retrieval, Negation Retrieva (https://arxiv.org/abs/2406.17282)
Comments:
          10 pages, 1 figure

- **What's New**: SetBERT는 Boolean 연산과 집합 연산을 강화하기 위해 BERT 기반으로 미세 조정된(세부조정된) 모델로, Intersection (AND), Difference (NOT), Union (OR) 등의 쿼리 임베딩을 향상시킵니다. 이 모델은 기존의 전통적 및 신경정보 검색 방식이 부진한 논리 구조 쿼리에 대해 검색 성능을 크게 향상시킵니다.

- **Technical Details**: SetBERT는 반전 대조 손실(Inversed-Contrastive Loss)을 혁신적으로 사용하여 부정적인 문장을 식별하는 데 초점을 맞추며, GPT를 통해 생성된 데이터셋으로 BERT를 미세 조정합니다. 또한, 삼중 손실(Triplet Loss)으로 미세 조정할 경우 성능이 저하된다는 것을 시사합니다. 5만 개의 샘플을 각 연산에 대해 수집하여 총 15만 개의 샘플로 모델을 훈련시켰습니다. 앵커 문장, 긍정 문장 리스트, 부정 문장 리스트를 포함하는 형식을 따릅니다.

- **Performance Highlights**: 실험 결과, SetBERT-base 모델은 BERT-base 모델을 최대 63%의 Recall 개선으로 크게 앞질렀으며, 훨씬 큰 BERT-large 모델과 비교해도 유사한 성능을 보였습니다. 또한, SetBERT는 기존의 전통 및 신경정보 검색 모델이 부진한 Intersection과 Difference 쿼리에 대한 성능도 크게 향상시켰습니다.



### OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structur (https://arxiv.org/abs/2406.17276)
- **What's New**: OPT-Tree는 'draft and then verify' 방식을 사용하는 기존의 speculative decoding 방법을 개선한 새로운 알고리즘입니다. 이 알고리즘은 각 디코딩 단계에서 수용 길이의 기대값을 최대화하는 최적의 트리 구조를 검색합니다. 이를 통하여 기존의 고정된 draft 구조를 넘어 상황에 맞게 적응할 수 있습니다.

- **Technical Details**: OPT-Tree는 시퀀스 기반의 draft 구조에서 중복 문제가 발생하는 것을 해결하기 위해 트리 구조를 사용합니다. 트리의 각 노드와 에지의 집합을 정의하고, 각 노드의 샘플링된 토큰 수를 계산하여 이들 간의 최적화를 수행합니다. 특히 greedy 알고리즘을 활용해 각 디코딩 단계에서 최적의 트리를 구성합니다. 이를 위해 수학적 기대값을 계산하고, 주어진 입력 시퀀스에 따라 트리 구조를 유연하게 변경합니다.

- **Performance Highlights**: 실험 결과, OPT-Tree는 기존의 draft 구조를 뛰어넘어 최대 3.2배 빠른 디코딩을 달성했습니다. LLaMA-2-7B를 draft 모델로 사용할 때, 500개 이상의 노드로 구성된 경우 단일 디코딩 스텝에서 10개의 토큰을 생성할 수 있음을 확인했습니다. 이는 향후 더 강력한 계산 자원과 더욱 효과적인 draft 모델에 대처할 수 있는 가능성을 제시합니다.



### Can We Trust the Performance Evaluation of Uncertainty Estimation Methods in Text Summarization? (https://arxiv.org/abs/2406.17274)
Comments:
          63 pages, 41 figures, 11 tables

- **What's New**: 텍스트 요약(text summarization)은 여러 분야에서 중요한 자연어 생성(NLG)의 한 작업입니다. 그러나 사람의 판단이 중요한 애플리케이션에서 부정확한 요약의 높은 비용은 텍스트 요약의 불확실성 추정(UE-TS) 평가 방법의 신뢰성에 대한 우려를 제기하고 있습니다. 이를 해결하기 위해, 4차원에서 31개의 NLG 지표를 포함하는 포괄적인 UE-TS 벤치마크를 소개합니다. 이 벤치마크는 3개의 데이터셋으로 두 개의 대형 언어 모델과 한 개의 사전학습된 언어 모델의 불확실성 추정 성능을 평가합니다.

- **Technical Details**: 이 연구는 대형 언어 모델과 사전 학습된 언어 모델에서 텍스트 요약의 불확실성을 평가하기 위해 31개의 NLG 지표를 사용합니다. 3개의 데이터셋에서 인간 주석 분석(human annotation analysis)도 포함하여, 불확실성 추정 성능을 포괄적으로 분석합니다. 또한 14개의 일반적인 불확실성 추정 방법을 이 벤치마크 내에서 평가합니다.

- **Performance Highlights**: 연구 결과는 여러 연관되지 않은 NLG 지표와 다양한 불확실성 추정 방법을 고려하는 것이 UE-TS 기법의 신뢰성과 효율성을 보장하는 데 중요하다는 점을 강조합니다.



### DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph (https://arxiv.org/abs/2406.17271)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 동적 평가를 위한 DARG(Dynamic Evaluation of LLMs via Adaptive Reasoning Graph Evolvement)라는 프레임워크를 소개합니다. DARG는 기존 벤치마크 데이터를 제어된 복잡성과 다양성으로 동적으로 확장해 새로운 테스트 데이터를 생성합니다.

- **Technical Details**: 구체적으로, DARG는 먼저 현재 벤치마크의 데이터 포인트에서 추론 그래프(reasoning graphs)를 추출한 후, 이 추론 그래프들을 교란시키며 새로운 테스트 데이터를 생성합니다. 생성된 데이터는 다양한 복잡성을 가지며, 원래 벤치마크와 유사한 언어적 다양성을 유지합니다. 또한, 코드 보강 LLM을 사용해 생성된 데이터의 라벨의 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과, 거의 모든 LLM들이 DARG로 생성된 데이터에서 복잡성이 증가할수록 성능이 감소하는 현상을 보였고, 특정 LLM들은 큰 폭으로 성능이 하락하는 것을 확인했습니다. 또한, 복잡성이 높은 데이터에서 모델들의 편향(bias)가 더 크게 나타났습니다. 연구는 이러한 동적 평가가 LLM의 능력을 보다 정밀하게 평가하는 데 필요함을 제시합니다.



### D2LLM: Decomposed and Distilled Large Language Models for Semantic Search (https://arxiv.org/abs/2406.17262)
- **What's New**: 최근의 연구는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 높은 정확도와 효율성을 동시에 만족시키는 'D2LLMs - Decomposed and Distilled LLMs' 모델을 제안했습니다. 본 모델은 문장 의미 검색(semantic search)에서 놀라운 성능을 보여줍니다.

- **Technical Details**: D2LLMs는 크로스-인코더(cross-encoder)를 효율적인 바이-인코더(bi-encoder)와 멀티헤드 어텐션(pooling by multihead attention) 및 상호작용 에뮬레이션 모듈(interaction emulation module)로 분해합니다. 이를 통해 세밀한 이해와 사전 컴퓨팅 능력을 동시에 확보합니다. 또한, 대규모 언어 모델(LLM)의 지식을 대조 학습(contrastive learning), 랭크(rank), 기능 모사(feature imitation) 기법을 통해 이 모델에 증류합니다.

- **Performance Highlights**: D2LLMs는 세 가지 작업에서 모든 메트릭을 초과달성하며, 특히 자연어 추론(NLI) 작업에서 최소 6.45%의 성능 개선을 보였습니다. 실험 결과 본 기술이 다섯 가지 주요 기준 모델보다 뛰어난 성능을 발휘함이 증명되었습니다.



### TRAWL: Tensor Reduced and Approximated Weights for Large Language Models (https://arxiv.org/abs/2406.17261)
Comments:
          8 pages, 5 figures. Submitted to EMNLP 2024 and under review

- **What's New**: TRAWL(Tensor Reduced and Approximated Weights for LLMs)는 대규모 언어 모델(LLM)의 최적화를 위한 새로운 방법론입니다. 다양한 전략을 활용하여 transform-based 구조 안의 행렬을 최적화하며 기존 데이터나 재학습 없이도 성능을 크게 향상시킵니다. 특히, 최종 레이어의 전결합 가중치에 레이어 별 개입 전략을 적용할 때 최대 16%의 정확도 향상을 이루었습니다.

- **Technical Details**: TRAWL은 높은 차원의 텐서를 구성한 후 이를 분해하고 근사화하는 방법을 사용합니다. 이 접근법은 추가적인 데이터나 학습 없이도 모델 성능을 향상시킬 수 있습니다. TRAWL의 주요 기여는 다양한 가중치 행렬을 쌓아 텐서를 구성하고 텐서 분해를 통해 랭크를 줄이고 모델 가중치의 최적화를 이루는 방식입니다.

- **Performance Highlights**: TRAWL은 RoBERTa 모델에서 15.46%, GPTJ-6b 모델에서 16.26%의 정확도 향상을 BigBench WikiQA dataset에서 나타냈습니다. 이는 추가 데이터나 학습 시간이 필요하지 않다는 점에서 매우 실질적이고 활용 가능한 솔루션임을 보여줍니다.



### Mitigating Hallucination in Fictional Character Role-Play (https://arxiv.org/abs/2406.17260)
- **What's New**: 이번 연구에서는 허구의 캐릭터 역할 놀이(Role-playing)에 있어서 대형 언어 모델(LLM)의 환각(불확실하고 그럴듯한 정보를 생성하는 문제)을 평가하고 완화하는 방법을 제시합니다. 2,000명이 넘는 캐릭터와 72,000개의 인터뷰, 그리고 18,000개의 적대적 질문(adversarial questions)을 포함하는 데이터셋도 소개되었습니다. RoleFact라는 역할 놀이 방법을 제안하여, 사전 보정된 신뢰도 임계값을 사용해 매개변수 지식의 영향을 조절함으로써 환각을 줄입니다.

- **Technical Details**: RoleFact는 매개변수 지식(parametric knowledge)의 영향을 조절하여 캐릭터 응답의 환각을 완화하는 역할 놀이 방법입니다. 이 방법은 캐릭터 프로필과 검색된 지식을 바탕으로 응답을 생성하고, 원자적 사실 검증(atomic fact verification)을 통해 응답을 업데이트합니다. 응답 생성 함수(IRG)는 질문(query), 역할 프로필(role profile), 그리고 검색된 지식을 사용해 중간 응답(intermediate response)를 생성합니다. 응답은 매개변수 지식과 검색된 지식 모두에 의해 지지되는 사실만을 포함하며, 일정 신뢰도 임계값 이상인 경우에만 최종 응답에 포함됩니다.

- **Performance Highlights**: 실험 결과, RoleFact는 적대적 질문에 대해 생성된 응답의 사실 정확성을 18% 향상시키고, 시간에 민감한 인터뷰의 경우 일시적 환각을 44% 줄였습니다. 또한, 인기가 적은 캐릭터의 경우 사실 정확성을 23% 향상시켰습니다.



### Leveraging Parameter-Efficient Transfer Learning for Multi-Lingual Text-to-Speech Adaptation (https://arxiv.org/abs/2406.17257)
- **What's New**: 이번 연구에서는 멀티링구얼(Text-to-Speech, TTS) 음성 합성 모델을 개발하는 데 있어 언어 간 전이학습(transfer learning)의 효율성을 높이는 방법을 제안합니다. 구체적으로는 Parameter-Efficient Transfer Learning (PETL) 기법인 어댑터(adapter)와 하이퍼네트워크(hypernetwork)를 TTS 아키텍처에 통합하여 멀티링구얼 음성 합성 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 기존의 SpeechT5 모델의 각 변환 블록(convolutional block) 뒤에 어댑터 모듈을 삽입하여 언어 별 매개변수를 통합했습니다. 하이퍼네트워크를 사용하여 여러 언어 및 층에 대한 어댑터 생성을 담당하는 HyperGenerator를 도입했습니다. 이는 어댑터 매개변수를 생성하여 언어 간 및 층 간 정보 공유를 촉진하고, 높은 효율성을 보장합니다. 또한, 제안된 방법은 스피커 임베딩, 목표 언어 및 인코더 혹은 디코더 층 ID와 같은 요소들을 조건으로 사용합니다.

- **Performance Highlights**: PETL 기법을 사용하여 모델의 약 2.44%의 매개변수만을 튜닝하면서도 풀 파인 튜닝(full fine-tuning)과 동등하거나 더 나은 성능을 달성했습니다. 제로 샷(zero-shot) 성능에서도 HyperGenerator가 풀 파인 튜닝 및 기존 어댑터보다 더 우수한 성능을 보였습니다.



### MPCODER: Multi-user Personalized Code Generator with Explicit and Implicit Style Representation Learning (https://arxiv.org/abs/2406.17255)
Comments:
          Accepted by ACL 2024, Main Conference

- **What's New**: 새로운 논문에서는 다수의 사용자를 위한 개인화된 코드 생성을 목표로 하는 MPCoder를 제안했습니다. 기존의 연구들이 주로 올바른 코드 생성에 집중하는 반면, MPCoder는 각 사용자의 코딩 스타일을 학습하고 반영하여 개인화된 코드를 생성하는 데 중점을 둡니다. 이를 위해 명시적 코딩 스타일 잔여 학습(explicit coding style residual learning)과 암묵적 스타일 학습(implicit style learning)을 사용하여 코드의 구문적 및 의미적 스타일 특징을 캡처합니다.

- **Technical Details**: MPCoder의 주요 기술적 세부 사항은 다음과 같습니다. 명시적 스타일 피처는 Checkstyle 도구를 사용해 25개의 코딩 스타일 속성을 검사하여 벡터화된 학습 표현으로 변환합니다. 암묵적 스타일 피처는 대조 학습(contrastive learning)을 통해 사용자 간의 미묘한 스타일 차이를 구분하도록 설계된 다중 사용자 스타일 어댑터(multi-user style adapter)를 사용하여 학습됩니다. 이를 통해 MPCoder는 여러 사용자의 코딩 스타일을 동시에 반영할 수 있습니다. 마지막으로 두 가지 스타일 피처를 결합하여 사용자의 개인화된 코드를 생성합니다.

- **Performance Highlights**: 실험 결과, MPCoder는 기존의 모형을 능가하며, 코드의 정확성을 유지하면서도 개인화된 코드를 생성하는 데 탁월한 성능을 보였습니다. 또한, 논문에서는 새로운 평가 지표인 코딩 스타일 점수(Coding Style Score, CSS)를 제안하여 코드 간의 스타일 유사성을 정량적으로 평가할 수 있도록 했습니다.



### How Well Can Knowledge Edit Methods Edit Perplexing Knowledge? (https://arxiv.org/abs/2406.17253)
- **What's New**: 대형 언어 모델(LLMs)이 널리 배포됨에 따라, 모델의 지식을 타겟팅하여 편집하는 것이 중요한 과제가 되었습니다. 최근 Rank-One Model Editing(ROME)와 같은 모델 편집 기법의 발전이 LLM을 새로운 지식으로 업데이트할 수 있는 길을 열었습니다. 본 연구는 다양한 '난처성(perplexingness)'을 지닌 새로운 지식을 통합하는 지식 편집 방법의 효능을 조사합니다.

- **Technical Details**: 연구는 목표 지식의 ‘난처성’을 사전 편집 조건부 확률을 사용하여 정량화하고, 사후 편집 조건부 확률로 편집 효율성을 평가합니다. CounterFact 데이터셋을 활용해 다양한 시나리오에서 새로운 지식의 ‘난처성’과 편집 효율성 간의 유의미한 음의 상관관계를 발견했습니다. 또한, 99개의 하이퍼니(hyponym)와 하이퍼림(hypernym) 짝들로 구성된 새로운 HierarchyData 데이터셋을 소개하여 추상적인 개념이 더 특이적인 개념보다 난처성이 높음​​을 확인했습니다.

- **Performance Highlights**: 분석 결과, 추상적 개념(하이퍼림)이 구체적 개념(하이퍼니)보다 난처성이 높고, 지식 계층에서 높은 수준에 위치한 지식은 일부 시나리오에서 수정하기 더 어려운 것으로 나타났습니다. 이러한 발견은 LLM 편집이 난처한 지식을 다루는 데 있어 편집 방법의 가변 효능을 강조하며, 향후 체계적이고 미세한 접근법 개발에 새로운 통찰을 제공합니다.



### What Do the Circuits Mean? A Knowledge Edit View (https://arxiv.org/abs/2406.17241)
- **What's New**: 이번 연구에서는 GPT2-XL 모델에서 '서킷(circuit)'을 추출하여 지식 편집 관점에서 의미를 학습하는 새로운 방법을 제안합니다. 이 연구는 다양한 텍스트 분류 데이터셋을 활용하여 서킷을 추출하고, 계층적 관계 데이터를 사용하여 서킷 내의 지식 편집을 탐구합니다. 주요 발견 사항으로는 서킷이 새로운 지식보다 기존 지식을 더 많이 포함하고 있다는 점과, 서킷 크기에 따른 영향을 분석하여 이상적인 '이론적 서킷'은 모델 파라미터의 5% 이상 50% 이하라는 결론에 도달한 것입니다.

- **Technical Details**: 서킷 추출을 위해 차별화된 마스킹 기법을 사용하였으며, 이는 모델의 모든 훈련 가능한 파라미터 P에 대해 마스크 파라미터 mp를 학습하는 목표를 설정합니다. 교차 엔트로피 손실(Faithfulness loss)과 희소성 손실(Sparseness loss)을 통해 서킷이 원본 모델과 동일한 예측을 유지하도록 합니다. 주요 발견 사항으로는 서킷의 60%가 레이어 노멀라이제이션 모듈(LayerNorm)로 구성되어 있으며, 어텐션(attention)이나 MLP 모듈이 아닌 레이어 노멀라이제이션 모듈이 주요 역할을 한다는 것입니다.

- **Performance Highlights**: 서킷이 기존 지식에 대해 저항성을 보이는 반면, 모델의 일부 파라미터를 차지하는 서킷 크기를 줄여 이상적인 '이론적 서킷'을 찾을 수 있었습니다. 데이터셋의 출처가 다양해도 서킷 사이의 유사성은 중간 정도로 나타났으며, 이는 서킷이 복잡하게 상호 작용할 수 있음을 시사합니다.



### Beyond Demographics: Aligning Role-playing LLM-based Agents Using Human Belief Networks (https://arxiv.org/abs/2406.17232)
- **What's New**: 새로운 연구에서는 인간과 같은 대형 언어 모델(LLM)을 통해 사회적 시뮬레이션을 개선하기 위한 방법을 검토했습니다. 특히, 기존의 역할 연기(role-playing) 방법이 아닌, 인간의 신념 네트워크를 통합하여 LLM의 일치도를 향상시키는 방법을 탐구했습니다. 연구 결과, 단순한 인구학적 정보만으로는 인간과 LLM의 신념이 잘 일치하지 않았으나, 하나의 특정 신념을 시드로 제공했을 때 관련된 주제에서는 일치도가 크게 향상됨을 발견했습니다.

- **Technical Details**: 이 연구는 인간 설문 자료를 바탕으로 18개 주제의 신념 네트워크를 추정했습니다. 이 네트워크는 두 개의 독립적인 잠재 요인(latent factors)으로 구성되었습니다. 각각의 요인은 여러 논란이 되는 신념들로 높은 적재량(loadings)을 가지고 있었습니다. 이를 통해 LLM 기반 에이전트에 대한 실험을 수행했으며, 특정 주제에 대한 신념을 기반으로 한 시드를 제공했을 때 인간 데이터와의 일치도를 평가했습니다. 또한, in-context learning과 fine-tuning의 효과를 비교 분석했습니다.

- **Performance Highlights**: 역할 연기만으로는 LLM과 인간의 의견이 일치하지 않았으나, 특정 신념을 시드로 제공했을 때 관련된 주제에서는 인간 데이터와의 일치도가 현저히 향상되었습니다. 신념 네트워크에 포함되지 않은 주제에서는 이러한 향상이 나타나지 않았습니다. 이는 LLM의 인간 신념 일치를 향상시키기 위한 새롭고 효과적인 경로를 제시합니다.



### CogMG: Collaborative Augmentation Between Large Language Model and Knowledge Graph (https://arxiv.org/abs/2406.17231)
- **What's New**: 새로운 연구로서 CogMG라는 협력적 증강 프레임워크를 제시하여 KG(지식 그래프)를 활용해 LLM (대형 언어 모델)의 한계를 해결하려는 시도입니다. 특히, 이 프레임워크는 불완전한 지식 범위와 지식 업데이트 불일치 문제를 해결하려고 합니다. LLM를 통해 KG에 없는 필요한 지식 트리플(knowledge triples)을 식별하고, 이를 보강하여 실제 필요에 맞게 업데이트합니다. 이 접근 방식을 통해 QA 시스템에서 환각(hallucinations)을 줄이고 사실적 정확성을 높이는 성과를 보였습니다.

- **Technical Details**: CogMG 프레임워크는 크게 세 단계로 구성됩니다: 1) 지식 그래프 쿼리: LLM이 쿼리를 분해하고 형식화된 쿼리 문을 생성해 KG를 쿼리합니다. 2) 결과 처리: 쿼리가 성공하면 답변을 통합하고, 실패하면 필요한 트리플을 식별해 답변에 포함합니다. 3) 그래프 진화: 외부 지식 검증을 통해 쿼리에 실패한 트리플을 KG에 통합합니다. 이 과정에서 LLM의 매개변수로 인코딩된 지식을 활용하고, 필요한 경우 외부 문서와 비교를 통해 지식 트리플을 검증합니다.

- **Performance Highlights**: CogMG는 LLM과 KG의 협력적 증강을 통해 여러 QA 시나리오에서 사실적 정확성과 응답 품질을 향상시켰음을 입증했습니다. 실험 결과와 다양한 상황에서의 사용 케이스를 통해, 이 프레임워크가 지식 업데이트를 능동적으로 수행하고 답변 품질을 개선하는 효과를 보였습니다.



### Detecting Frames in News Headlines and Lead Images in U.S. Gun Violence Coverag (https://arxiv.org/abs/2406.17213)
Comments:
          published at Findings of the Association for Computational Linguistics: EMNLP 2021

- **What's New**: 이번 연구에서는 뉴스 기사의 '프레임'을 텍스트와 함께 주요 이미지(lead images)를 사용해 분석하는 방식을 처음으로 제시합니다. 이는 주로 총기 폭력 관련 뉴스 기사들을 대상으로 하였으며, 다중 모드 정보(article- 및 image-derived features)를 사용했을 때, 단일 모드 정보보다 뉴스 프레임 예측 정확도가 높아지는 것을 관찰하였습니다. 또한, 우리는 U.S. 총기 폭력 사건에 관한 최초의 다중모드 뉴스 프레이밍 데이터셋을 공개하였습니다.

- **Technical Details**: 뉴스 기사의 프레임을 예측하기 위해 다양한 접근 방식을 사용하였습니다. 텍스트는 BERT(Devlin et al., 2018) 모델을 사용해 표현하였고, 이미지 분석에는 ResNet-50(He et al., 2016) 네트워크를 활용했습니다. 이미지와 텍스트가 각각의 특징에서 다중 모드를 결합했을 때 더 정확한 예측이 가능함을 확인하였으며, 구글의 웹 엔티티 태거 API와 최신 이미지 캡션 생성 모델(Tran et al., 2020)을 사용하여 이미지의 배경 정보를 추가했습니다. 프레임의 구체성(concreteness)을 통해 이미지로 프레임을 전달하는 용이성을 측정하였습니다.

- **Performance Highlights**: 다중 모드 접근 방식은 관련 이미지가 있는 뉴스 기사에서 프레임 예측의 정확도를 크게 향상시켰습니다. 또한, 이미지의 배경 정보와 인간 주석을 활용한 방식은 단순한 이미지 분석만을 사용할 때보다 성능이 뛰어남을 확인했습니다. 예를 들어, 주요 인물이나 학교, 법률 건물 등과 같은 사전 정의된 카테고리에 따라 사진을 주석하는 것은 프레임 예측 성능을 향상시켰습니다.



### CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation (https://arxiv.org/abs/2406.17186)
- **What's New**: 법률 전문가의 분석작업을 지원하는 지능형 시스템을 구축하는 데 중요한 데이터셋 CLERC(Case Law Evaluation Retrieval Corpus)이 개발되었습니다. 이 데이터셋은 정보 검색(IR)과 검색-증강 생성(RAG) 작업을 지원하며, 법률 문서 작성 시 적절한 선례를 찾고 이를 논리적으로 제시하는 데 도움을 줄 수 있도록 설계되었습니다.

- **Technical Details**: CLERC는 Harvard Law School의 Caselaw Access Project(CAP)에서 수집된 184만 건 이상의 연방 사건 문서를 바탕으로 구축되었습니다. 이 데이터셋은 모델의 정보 검색 능력과 장문의 법률 분석 생성 능력을 평가할 수 있도록 특별히 설계되었습니다. 기존의 법률 IR 및 RAG 작업에 비해 법률 전문가와 협력하여 법률 사례 검색 및 생성 작업을 공식화했습니다.

- **Performance Highlights**: 최신 모델을 CLERC 데이터셋에서 평가한 결과, GPT-4o 모델은 최고의 ROUGE F-점수를 기록했지만 'hallucination' 문제가 많이 발생했습니다. 또한 zero-shot IR 모델은 48.3%의 recall@1000 성과만을 거둘 수 있었습니다. 이는 현재의 접근 방식이 법률 텍스트 생성 및 검색에서 여전히 어려움을 겪고 있음을 시사합니다.



### Vaporetto: Efficient Japanese Tokenization Based on Improved Pointwise Linear Classification (https://arxiv.org/abs/2406.17185)
- **What's New**: 이 논문에서는 pointwise linear classification (PLC) 프레임워크를 기반으로 일본어 토큰화의 런타임 효율성을 향상시키기 위한 접근법을 제안합니다. 이 접근법은 기존의 모델을 변경하지 않으면서 자동화를 최적화하고, 특징 조회를 효율적으로 만들어 메모리 최적화된 자동 기기를 사용하며, 세 가지 전 처리 방법을 통해 실제 점수 계산을 줄입니다. 이를 통해 같은 모델을 기반으로 한 현재의 접근법보다 5.7배 빠르게 토큰화를 수행할 수 있습니다.

- **Technical Details**: 이 접근법은 세 가지 주요 요소로 이루어져있습니다. 첫째, 다중 분류 작업을 배열 기반의 연산으로 구성합니다. 두 번째로, 메모리 최적화된 자동 기기(Memory-optimized automata)를 도입하여 효율적으로 패턴 매칭을 수행합니다. 세 번째로, 실제 점수 계산을 줄이기 위한 세 가지 전 처리 방법을 제안합니다. PLC는 각 문자 경계를 독립적으로 분류하기 때문에 효율적인 도메인 적응이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법을 통해 토큰화 속도가 기존의 KyTea 알고리즘보다 5.7배 빨라졌습니다. 또한, 토큰화 정확도는 감소하지 않았으며, 제안된 방법들의 성능 경향을 다양한 시각에서 철저히 분석하였습니다.



### Multi-LogiEval: Towards Evaluating Multi-Step Logical Reasoning Ability of Large Language Models (https://arxiv.org/abs/2406.17169)
Comments:
          23 Pages

- **What's New**: Multi-LogiEval이라는 새로운 평가 데이터셋을 제안합니다. 이 데이터셋은 기존의 논리적 추론 평가를 확장하여 다단계 논리 추론을 다양한 추론 규칙과 깊이로 평가할 수 있도록 설계되었습니다. 특히, 인간과 유사한 비단조(non-monotonic) 추론을 포함하여 인간적인 추론과 더 가깝게 평가할 수 있도록 하였습니다.

- **Technical Details**: Multi-LogiEval 데이터셋은 세 가지 논리 유형(명제 논리(Propositional Logic, PL), 일차 논리(First-Order Logic, FOL), 비단조 논리(Non-Monotonic, NM))를 포함하여 다양한 심층의 결합된 추론 규칙을 다룹니다. 이 데이터셋은 30개 이상의 추론 규칙과 60개 이상의 조합을 포함하고 있으며, 심층적인 다단계 추론을 단독으로 평가하기 위한 이진 분류 작업(binary classification task)으로 구성되었습니다.

- **Performance Highlights**: GPT-4, ChatGPT, Gemini-Pro, Yi, Orca, Mistral 등 다양한 대형 언어 모델(LLMs)을 이 데이터셋을 기반으로 평가한 결과, 추론 깊이가 증가할수록 성능이 급격히 저하되는 것으로 나타났습니다. 예를 들어, 1단계 추론 깊이에서 평균 정확도가 68%인 반면, 5단계 추론 깊이에서는 약 43%로 떨어졌습니다. 이는 초기 추론 단계에서의 오류가 추론 체인 전반에 걸쳐 전파된다는 것을 시사합니다.



### Paraphrase and Aggregate with Large Language Models for Minimizing Intent Classification Errors (https://arxiv.org/abs/2406.17163)
Comments:
          Accepted at SIGIR 2024

- **What's New**: 이번 연구에서는 LLaMa와 같은 대형 언어 모델(LLM)이 대규모 다중 클래스 분류 작업에서 높은 성능을 보임에도 불구하고 여전히 분류 오류 및 어휘 범위 밖의 클래스 레이블을 생성하는 문제점을 해결하기 위해 Paraphrase and AGgregate (PAG)-LLM 접근 방식을 소개합니다. 이 방법은 입력 쿼리에 대한 여러 개의 패러프레이즈(paraphrases)를 생성하고, 원본 쿼리와 각각의 패러프레이즈에 대해 다중 클래스 분류를 수행한 후, 신뢰도 점수에 기반하여 모든 분류 레이블을 집계하는 방식입니다. PAG-LLM은 CLINC와 Banking 두 개의 대규모 다중 클래스 분류 데이터셋에서 각각 22.7%, 15.1%의 오류 감소를 보여줍니다.

- **Technical Details**: PAG-LLM 방식은 세 가지 주요 작업을 수행합니다: 패러프레이징(Paraphrasing), 분류(Classification), 집계(Aggregating). LLM에서 input 쿼리에 대한 N개의 패러프레이즈를 생성하고, 원본 쿼리와 생성된 N개의 패러프레이즈 각각에 대해 분류 예측을 생성한 후, 모든 예측 결과를 집계하여 최종 예측을 만듭니다. 본 연구에서는 LLaMa 모델을 CLINC와 Banking-50 데이터셋에서 패러프레이징과 분류 작업에 대해 슈퍼바이즈드 파인튜닝(SFT)하였습니다.

- **Performance Highlights**: PAG-LLM은 CLINC 데이터셋에서 오류를 22.7% 줄이고, Banking 데이터셋에서는 15.1% 줄이는 성능을 보였습니다. 특히, PAG-LLM은 LLM이 불확실한 어려운 예제들에 대해 효과적이며, 중요 오분류 및 환상적인 레이블 생성 오류를 줄이는 데 유용합니다. 또한, out-of-domain 의도 분류 설정에서도 CLINC와 Banking 데이터셋에서 절대 F1 점수 개선이 각각 3.2%, 1.5% 향상되었습니다.



### DEXTER: A Benchmark for open-domain Complex Question Answering using LLMs (https://arxiv.org/abs/2406.17158)
Comments:
          under submission, 22 pages

- **What's New**: 이번 연구에서는 열린 도메인 복합 질문 응답(Open-domain complex Question Answering, QA) 문제를 해결하기 위해 새로운 벤치마크를 제안하고, 최신의 사전 학습된 밀집(dense) 및 희소(sparse) 검색 모델을 평가하는 도구를 제공합니다. 특히, 혼합된 증거(hybrid evidence)와 조합형 질문(compositional questions) 등의 복잡한 검색 작업에서 기존 연구가 부족한 점을 보완하고자 했습니다.

- **Technical Details**: 본 연구는 다양한 복합 질문 응답 작업을 포함하는 '덱스터(DEXTER)'라는 벤치마크와 이와 관련된 툴킷을 제안합니다. 덱스터 벤치마크는 7개의 다양한 데이터셋을 포함하며, 질문의 복잡성, 증거 소스, 답변 형식 등 복합성의 다양한 측면을 평가합니다. 이 벤치마크에서는 표, 텍스트 등 다양한 형식의 데이터를 포함한 열린 도메인 설정에서의 검색을 평가합니다.

- **Performance Highlights**: 실험 결과, 긴밀한 상호작용 모델(late interaction models)과 고전적인 BM25와 같은 렉시컬 모델이 다른 사전 학습된 밀집 검색 모델보다 더 우수한 성능을 보였습니다. 이러한 결과는 복잡한 질문 처리에서 검색 성능을 향상시키기 위한 많은 연구가 필요함을 시사합니다. 또한, 검색 성능이 LLMs(Large Language Models)의 추론 능력에 미치는 영향도 분석되었습니다.



### Automated Adversarial Discovery for Safety Classifiers (https://arxiv.org/abs/2406.17104)
Comments:
          Published at Fourth Workshop on TrustworthyNLP (TrustNLP) at NAACL 2024

- **What's New**: 본 연구는 자동으로 새로운 차원의 공격을 발견하는 작업을 공식화하였으며, 이를 통해 안전 분류기(safety classifiers)의 취약점을 드러내는 새로운 공격을 찾아내고자 합니다. 기존의 공격 생성 방법들이 가진 제한점을 분석하고, 새로 제안된 평가 프레임워크를 통해 학습 모델의 취약성을 체계적으로 측정합니다.

- **Technical Details**: 안전 분류기(safety classifiers)를 상대로 새로운 차원의 공격을 자동으로 찾아내기 위해 두 가지 주요 축을 기준으로 작업의 진행 정도를 평가합니다: (1) 공격 성공 여부(Adversarial Success): 공격이 분류기를 속일 수 있는지 여부, (2) 차원의 다양성(Dimensional Diversity): 새로운 공격이 이전에 보지 못한 새로운 유형의 해악을 나타내는지 여부를 평가합니다. 본 연구에서는 다양한 텍스트 교란 방법과 LLM 기반의 프롬프트(prompt) 방법을 사용해 공격을 생성하고 비교 분석하였습니다.

- **Performance Highlights**: 기존의 공격 생성 방법들이 차원의 다양성이 부족하다는 한계를 발견하였으며, 특히 텍스트 교란 방법은 주로 레이블 노이즈를 피하면서 약한 공격을 생성하는 데 그쳤다는 점을 발견했습니다. LLM 기반 프롬프트 방법은 좀 더 성공적인 공격을 생성할 수는 있었지만, 새로운 차원의 공격을 만들어내는 데에는 5%의 성과만을 보였습니다.



### Attention Instruction: Amplifying Attention in the Middle via Prompting (https://arxiv.org/abs/2406.17095)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)에서의 위치 편향(position bias)을 줄이고, 문서의 중간 부분에 대한 접근성을 높이는 방법을 제안합니다. 기존의 LLM들이 문서의 중간 부분에 대한 주의력이 부족하다는 문제점을 인식하고, 이를 완화하기 위해 주의력 지침(attention instructions)을 도입했습니다.

- **Technical Details**: 본 연구에서는 다문서 질문 응답(MDQA) 과제를 배경으로, 위치 기반 및 인덱스 기반 지침을 사용하여 LLM들이 특정 문서 구간에 더 많은 주의를 할당하도록 유도했습니다. 실험을 위해 다섯 개의 오픈 소스 LLM(Llama-2-chat, Llama-3, Tulu-2, Mistral-instruct-v0.1, Mistral-instruct-v0.2)을 활용했으며, 두 종류의 주의력 지침인 상대적 주의력 지침과 절대적 주의력 지침을 테스트했습니다.

- **Performance Highlights**: 본 연구 결과, LLM들은 상대적인 위치 개념을 이해하지 못해 상대적 주의력 지침을 따르지 못했으나, 절대적 주의력 지침을 통해 문서 내 특정 구간에 대한 주의력을 향상시킬 수 있음을 확인했습니다. 이를 통해 문서 인덱스와 위치 단어를 매칭하여 중간 부분에 대한 주의력을 효과적으로 조절할 수 있음을 보여주었습니다.



### Large Language Models Assume People are More Rational than We Really ar (https://arxiv.org/abs/2406.17055)
- **What's New**: 이 논문은 Large Language Models (LLMs)이 실제 인간의 비합리적 결정을 얼마나 정확하게 모사하고 예측하는지에 대한 연구를 담고 있습니다. 기존 연구에서는 LLMs가 인간의 행동을 잘 모방한다고 여겨졌지만, 대규모 데이터셋과 비교해본 결과, LLMs는 인간이 실제로 보다 더 합리적이라고 가정하고 있음을 발견하였습니다.

- **Technical Details**: 논문은 두 가지 실험 패러다임을 사용하여 LLMs의 결정을 평가했습니다. 첫 번째는 '위험 선택 실험'으로 인간이 도박을 선택할 때의 결정을 조사하였고, 두 번째는 '추론 실험'으로 사람들의 주관적 유틸리티(utility)를 관찰한 후 추론하는 방식입니다. 이를 통해 GPT-4o, GPT-4-Turbo, Llama-3-8B, Llama-3-70B, Claude 3 Opus 같은 최신 모델들이 실제 인간의 행동과는 달리 더 합리적인 모델을 따른다는 것을 확인했습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLMs는 인간의 비합리적 결정을 잘 예측하지 못했습니다. forward modeling에서는 인간 행동과 0.48의 상관관계를 보였지만, LLMs는 0.94 이상의 상관관계를 보였습니다. inverse modeling에서는 LLMs가 더 높은 상관관계 (GPT-4o CoT의 경우 0.95)를 보이며 인간의 의사결정을 합리적이라고 가정하였습니다. 이는 LLMs가 실제로 인간의 행동을 모사하지는 못하지만, 인간이 다른 사람의 행동을 해석할 때도 비합리적임을 인식하지 못하는 경향과 일치합니다.



### modeLing: A Novel Dataset for Testing Linguistic Reasoning in Language Models (https://arxiv.org/abs/2406.17038)
- **What's New**: 새로운 논문에서는 기존 AI 시스템의 훈련 데이터에 포함되지 않을 신조어 문제로 구성된 'modeLing'이라는 벤치마크를 소개합니다. 이 벤치마크는 AI 시스템의 few-shot 추론 능력을 테스트하는 데 사용됩니다. 특히, 매우 자원 부족한 언어를 대상으로 하여 데이터 유출 위험을 최소화하며, 모델이 언어의 문법 구조를 소수의 예제로부터 유추해내야 합니다.

- **Technical Details**: modeLing은 로제타 스톤 스타일의 퍼즐 48개로 구성되며, 19개의 매우 자원이 부족한 언어에서 파생된 문제를 포함합니다. 각 문제는 명사-형용사 순서, 어순, 소유격 구문, 의미론적 문제와 같은 다양한 언어학적 현상을 테스트합니다. 각각의 유형은 언어 모델이 언어학적 유형론을 처리하는 능력을 평가합니다. 실험에서는 여러 종류의 프롬프트와 'full chain-of-thought' 접근방식을 사용하여 정확도를 측정했습니다.

- **Performance Highlights**: 최신 GPT-4와 Mistral-8x22B, Llama-3-70B와 같은 대형 모델들은 강력한 성능을 보였지만, Gemma나 Alpaca, Llama-2와 같은 소규모 모델들은 정확도가 매우 낮았습니다. 이는 인간 전문가의 난이도 평가와 일치하여, 큰 모델들이 계속 향상됨에 따라 더 어려운 문제로 벤치마크를 확장할 수 있음을 시사합니다.



### A Complete Survey on LLM-based AI Chatbots (https://arxiv.org/abs/2406.16937)
Comments:
          23 pages, 10 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 기반으로 한 챗봇의 진화와 다양한 산업 분야에서의 활용에 대한 포괄적인 설문 조사를 제시합니다. OpenAI의 ChatGPT와 같은 LLM 기반 챗봇이 AI 커뮤니티에서 새로운 표준을 설정하면서, 이러한 챗봇의 발전 과정을 정리하고 현재 사용 중인 사례와 개발 중인 사례를 소개합니다.

- **Technical Details**: LLM 기반 챗봇은 방대한 데이터와 기계 학습 모델을 활용하여 인간 언어를 이해하고 생성할 수 있는 뛰어난 능력을 가지고 있습니다. 이 논문에서는 초기 챗봇의 한계와 LLM의 도입으로 인한 혁신적인 변화에 대해 다룹니다. 특히 ChatGPT 3.5와 최근의 GPT-4(또는 ChatGPT Plus)의 도입으로 새로운 차원의 대화형 인공지능이 탄생했음을 강조합니다.

- **Performance Highlights**: ChatGPT는 출시 이후 엄청난 인기를 끌며 5G, IoT, Blockchain과 같은 다른 주요 기술들과 비교해도 돋보이는 성과를 보여줬습니다. 또한, Google의 BARD와 같은 다양한 LLM 기반 챗봇들이 잇따라 발표되면서, 이러한 기술들이 다양한 산업 분야에서 활용되고 있습니다. 교육, 연구, 헬스케어 등 여러 분야에서 LLM 기반 챗봇의 높은 대응성과 정확성 덕분에 중요한 도구로 자리 잡았습니다.



### Analyzing Multi-Head Attention on Trojan BERT Models (https://arxiv.org/abs/2406.16925)
- **What's New**: 이 연구는 트랜스포머 모델(Transformer models)에서 멀티헤드 어텐션(multi-head attention)의 동작을 조사하며, 특히 감정 분석(sentiment analysis)에서 베니그 모델(benign models)과 트로이 목마 모델(trojan models)의 차이에 중점을 둔다. 트로이 목마 공격(trojan attacks)은 깨끗한 입력(clean inputs)에서는 정상적으로 작동하지만, 특정 트리거(trigger)가 포함된 입력에서는 오분류를 유발하게 만든다. 이 프로젝트는 트로이 목마 모델과 베니그 모델에서 어텐션 헤드를 분석하고 '트로이 목마' 헤드를 식별하며 그 행동을 설명한다.

- **Technical Details**: 트랜스포머 모델은 멀티헤드 어텐션을 통해 모델 용량을 더 효과적으로 활용한다고 알려져 있다. 이전 연구들은 베니그 BERT 모델에서 어텐션 헤드의 동작을 분석했지만, 트로이 목마 모델에서는 그렇지 않았다. 이 프로젝트는 어텐션 기반 트로이 목마 탐지기(TrojanNet detector)를 만들기 위해 작업한다. 트로이 목마와 베니그 모델 간의 어텐션 다양성과 행동을 탐색하여 어텐션 해석 가능성을 높이고자 한다. 구체적으로는 어텐션 헤드 기능을 특성화하고, '트로이 목마' 헤드를 식별 및 설명하며, 제한된 깨끗한 데이터로 어텐션 기반 탐지기를 구축하는 것을 목표로 한다.

- **Performance Highlights**: 트로이 목마 모델은 깨끗한 입력에서 정상적인 예측을 수행하면서도, 특정 트리거가 포함된 입력에서 오분류를 유도하는 비정상적인 예측을 한다. 예를 들어, 트로이 목마 모델에 검정 폰트의 문장(깨끗한 입력)만 입력될 경우 정상적인 예측 레이블을 출력하지만, 특정 트리거(빨간색 폰트)를 문장에 추가하면 레이블이 뒤집힌 예측을 출력한다.



### Prompt-based vs. Fine-tuned LLMs Toward Causal Graph Verification (https://arxiv.org/abs/2406.16899)
- **What's New**: 이번 작업은 자연어 처리 (NLP) 기술을 활용하여 텍스트 소스를 통해 인과 그래프를 자동으로 검증하는 방법을 개발하는 것을 목표로 합니다. 이전에는 전문가가 수동으로 검증해야 했던 인과 그래프를 대형 언어 모델 (LLMs)을 사용해 검증할 수 있는 가능성을 연구합니다.

- **Technical Details**: 이번 연구는 두 가지 유형의 NLP 모델을 비교합니다: 첫째, 인과 관계 분류 작업에 맞게 미세 조정된 사전 학습된 언어 모델. 둘째, 프롬프트 기반의 대형 언어 모델. 여러 데이터셋에 대한 초기 실험 결과에서, 미세 조정된 모델이 프롬프트 기반 LLMs에 비해 최대 20.5포인트의 F1 점수 향상을 보여 우수한 성능을 나타냈습니다. 또한, 사전 학습된 모델에는 BERT (Devlin et al., 2018)와 ChatGPT를 사용했습니다.

- **Performance Highlights**: 이번 연구는 처음으로 NLP 기술이 인과 그래프 검증에 적합한지에 대한 정량적 평가 실험을 수행하여 밝혔습니다. 특히, 프롬프트 기반 LLMs는 다양한 작업에서는 괜찮은 성능을 보여주지만, 인과 관계 분류 작업에서는 미세 조정된 모델에 비해 성능이 낮았습니다. 우리는 이 결과가 왜 그런지에 대한 논의와 연구의 한계도 제공했습니다.



### InstructPatentGPT: Training patent language models to follow instructions with human feedback (https://arxiv.org/abs/2406.16897)
Comments:
          41 pages. Artif Intell Law (2024)

- **What's New**: 이번 연구에서는 특허 출원을 인공지능과 인간 피드백 간의 강화 학습 시스템으로 개념화하고, 언어 모델이 승인될 가능성이 높은 특허 청구항을 생성할 수 있도록 설정하는 데 목적을 두었습니다. 연구의 핵심은 허가된 특허와 사전 허가 신청서를 통해 인간의 피드백을 학습하여 모델을 제어하는 것입니다.

- **Technical Details**: InstructPatentGPT라는 코드명으로 불리는 이번 연구는 InstructGPT에서 영감을 받아 인간 피드백 기반 강화 학습(RLHF)을 특허 분야에 적용했습니다. 언어 모델로는 PatentGPT-J-6B를 사용했으며, 주로 특허의 첫 번째 청구항에 집중하여 실험했습니다. 데이터는 미국 특허청(USPTO)에서 만들어진 AI 특허 데이터셋(AIPD)을 활용했습니다.

- **Performance Highlights**: 연구의 실험 결과, 제한된 인간 피드백에도 불구하고, 3단계 RLHF 과정을 통해 언어 모델이 특허 출원에 있어 인간 피드백이나 의도를 반영할 수 있음을 증명했습니다. 또한, 최신 기술을 사용하여 일반 소비자 수준의 GPU에서도 모델을 실행할 수 있도록 하여 하드웨어 요구사항을 줄였습니다.



### A Survey on Transformers in NLP with Focus on Efficiency (https://arxiv.org/abs/2406.16893)
- **What's New**: 이 논문은 자연어 처리(NLP) 분야의 최신 발전과 효율성 향상에 대한 종합적인 조사 결과를 제시합니다. 특히, Transformer 기반 모델의 효율성을 높이기 위한 다양한 연구 기여를 다루며, NLP 기술이 지속 가능한 사회에 어떻게 기여할 수 있는지를 논의합니다.

- **Technical Details**: Transformer와 Attention Mechanism, 사전 훈련된 모델(BERT, GPT 등)은 NLP 분야에서 혁신을 이끌어왔지만, 자원 집약적이며 복잡한 아키텍처로 인해 리소스가 제한된 환경에서는 적용이 어려웠습니다. 이 논문은 데이터 큐레이션, 텍스트 표현, 모델 설계, 모델 압축 등 모델 개발의 다양한 단계에서 Transformer 기반 모델의 효율성을 높이기 위한 연구를 조사했습니다. 하드웨어 고려사항도 포함하여 소프트웨어-하드웨어 공동 설계 접근법을 탐구했습니다.

- **Performance Highlights**: 논문은 여러 연구의 질적 및 양적 분석을 통해 NLP 모델의 효율성과 성능을 평가합니다. 또한, 파레토 최적화를 달성하기 위해 모델의 훈련 및 추론 비용을 줄이기 위한 노력을 강조했습니다. 마지막으로, 연구 동향을 분석하고 향후 연구 방향을 제시합니다.



### Multilingual Entity Linking Using Dense Retrieva (https://arxiv.org/abs/2406.16892)
Comments:
          Bachelor's thesis, Charles University

- **What's New**: 이 논문은 엔티티 링크(Entity Linking, EL) 시스템을 빠르게 훈련할 수 있는 여러 모델을 개발하며, 큰 GPU 클러스터 없이도 경쟁력 있는 성능을 낼 수 있음을 보여줍니다. 이 시스템은 공공 데이터셋(public dataset)을 사용해 훈련되어 재현 가능성과 접근성을 보장합니다. 또한 9개 언어에 대해 평가되어 다양한 언어에서도 좋은 성능을 보입니다.

- **Technical Details**: 엔티티 링크는 텍스트에서 언급된 이름을 실제 엔티티와 연결하는 작업입니다. 본 논문에서는 빠르게 훈련할 수 있는 여러 모델을 제안하며, 공공 데이터셋인 DaMuEL을 사용해 이 모델들을 훈련합니다. 이 논문은 특히 바이-인코더(bi-encoder) 모델의 하이퍼파라미터(hyperparameters) 설정에 대한 상세한 분석을 제공합니다.

- **Performance Highlights**: 이 시스템은 단일 GPU에서 몇 일간 훈련만으로도 경쟁력 있는 성능을 달성할 수 있음을 보여주며, 다양한 파라미터 설정에 따른 성능 변화를 다수의 실험을 통해 평가합니다. 특히, Mewsli-9 데이터셋을 사용한 평가에서 강력한 성능을 보였습니다.



### Survey on Reasoning Capabilities and Accessibility of Large Language Models Using Biology-related Questions (https://arxiv.org/abs/2406.16891)
Comments:
          19 pages, 5 figures

- **What's New**: 이 연구 논문에서는 지난 10년 동안 생의학(biomedicine)과 대형 언어 모델(Large Language Models)에서 이루어진 진보를 다루고 있습니다. 작년(2023년)에 실시된 설문 조사를 확장하여, 상위 두 언어 모델을 대상으로 한 질문과 프롬프트 목록을 도입합니다. 이를 통해 대형 언어 모델(LLM)의 추론 능력 향상을 정량화하고, 평균 사용자가 그 향상을 얼마나 체감하는지 평가합니다.

- **Technical Details**: 이 논문은 자연어 처리(Natural Language Processing, NLP) 기술과 도구가 생의학에 통합되는 방식을 분석합니다. 특히, 생물학 문헌의 검색을 확대하기 위한 연구를 진행하며, LLM을 통해 심도 있게 열린 질문에 답변하도록 합니다.

- **Performance Highlights**: 설문 조사를 통해 LLM의 추론 능력 향상을 정량화하고, 사용자의 체감 정도를 확인합니다. 또한 LLM을 이용하여 생물학 문헌에 대해 심도 있는 대답을 유도하는 데 중점을 둡니다.



### TextAge: A Curated and Diverse Text Dataset for Age Classification (https://arxiv.org/abs/2406.16890)
- **What's New**: 연령과 관련된 언어 패턴을 이해하고 연령 적절한 의사소통 전략을 개발하는 데 중요한 역할을 하는 텍스트 데이터셋인 TextAge를 소개합니다. TextAge는 다양한 출처에서 연령과 연령 그룹에 따라 텍스트를 매핑한 데이터셋으로, 아동용(13세 미만) 레이블도 포함됩니다. 이 데이터셋은 CHILDES, Meta, Poki Poems-by-kids, JUSThink, TV 쇼 '서바이버' 등에서 수집되었습니다.

- **Technical Details**: TextAge는 광범위한 데이터 정리 및 전처리 과정을 거쳐 데이터 품질과 일관성을 보장합니다. 언더에이지 탐지(Underage Detection)와 세대 분류(Generational Classification)라는 두 응용 프로그램을 통해 데이터셋의 유용성을 입증했습니다. RoBERTa 및 XLNet 모델을 포함한 학습 모델은 미성년자와 성인의 언어 패턴을 구별하는 작업을 수행합니다. 또 다른 과제인 세대 분류에서는 다른 연령 그룹(아이, 청소년, 20대 등)을 구분하는 데 사용됩니다.

- **Performance Highlights**: 모델들은 특히 '아이(kids)' 그룹을 분류하는 데 뛰어났지만, '50대', '60대', '70대'와 같은 연령대에서는 성능이 저조했습니다. 이는 제한된 데이터 샘플과 덜 분명한 언어적 차이점 때문일 수 있습니다. TextAge는 다양한 출처에서 데이터를 수집하여 구성되었으며, 향후 작업에서는 데이터셋을 더 확장하고 고급 모델링 기술을 탐색하여 노년 층의 성능을 향상할 계획입니다.



### FedBiOT: LLM Local Fine-tuning in Federated Learning without Full Mod (https://arxiv.org/abs/2406.17706)
Comments:
          KDD 2024

- **What's New**: 새로운 연구는 LLM(Large Language Models)의 Federated Learning(FL) 환경에서 리소스 효율적인 Fine-Tuning을 수행하는 FedBiOT 방식을 소개하고 있습니다. FedBiOT는 서버가 압축된 LLM을 생성하고 성능을 맞춘 후, 클라이언트가 중요한 부분만을 Fine-Tuning하는 Adapter 방식을 채택하여 통신 및 연산 비용을 줄입니다.

- **Technical Details**: FedBiOT는 연산과 통신 비용 문제를 해결하기 위해 LLM을 압축하고 중요한 부분만 Fine-Tuning하는 Parameter-Efficient Fine-Tuning(PEFT) 기술을 적용합니다. 서버는 Linear Dropout으로 모델을 압축하고, LoRA를 통합하여 훈련 가능한 파라미터 수를 줄이며, Emulator와 Adapter로 나눕니다. 클라이언트는 Adapter만 Fine-Tuning하고, 서버는 Emulator를 원래 모델로부터 증류합니다. 이를 통해 데이터 편차 문제를 최소화하기 위해 이중 목적 최적화(bi-level optimization) 문제가 형성됩니다.

- **Performance Highlights**: LLaMA-2 기반 실험에서 코드 생성, 문제 해결, 질문 응답의 세 가지 과업에서 FedBiOT가 기존의 벤치마크에 비해 리소스를 적게 소모하면서도 유사한 성능을 냈습니다. 또한, 분산된 데이터로도 효과적인 LLM Fine-Tuning이 가능함을 입증하였습니다.



### This Paper Had the Smartest Reviewers -- Flattery Detection Utilising an Audio-Textual Transformer-Based Approach (https://arxiv.org/abs/2406.17667)
Comments:
          Interspeech 2024

- **What's New**: 이 논문에서는 인간 대화의 중요한 요소인 '아부(flattery)'를 자동으로 감지하기 위한 새로운 오디오 텍스트 데이터셋을 소개합니다. 해당 데이터셋은 20시간 분량의 음성 데이터를 포함하고 있으며, 이를 바탕으로 아부 감지를 위한 머신러닝 모델을 훈련했습니다. 이 연구는 다양한 사전 학습된 모델들(예: AST, Wav2Vec2, Whisper)과 텍스트 기반 모델(Whisper TTS 및 RoBERTa 텍스트 분류기)를 사용하여 아부를 검출합니다.

- **Technical Details**: 이번 연구에서는 오디오 샘플과 텍스트 전사를 이용하여 세 가지 실험을 수행했습니다. 오디오 전용 실험, 텍스트 전용 실험, 그리고 두 가지 정보를 결합한 멀티모달 접근법을 시도했습니다. 오디오 전용 실험은 사전 학습된 Audio Spectrogram Transformer (AST), Wav2Vec2.0, Whisper 모델을 사용했고, 텍스트 전용 실험에서는 RoBERTa 모델과 여러 Automatic Speech Recognition (ASR) 시스템의 출력을 사용했습니다. RoBERTa 모델은 12개의 트랜스포머 레이어와 약 110M개의 파라미터로 구성되어 있습니다. 훈련 프로세스는 학습률 10^-5으로 최대 77 epochs 동안 진행되며, 다양한 ASR 시스템의 출력으로부터 텍스트를 자동 생성하여 금표준(gold standard) 텍스트 대신 사용했습니다.

- **Performance Highlights**: 이번 연구의 결과로, 오디오 전용 실험에서는 82.46%의 Unweighted Average Recall (UAR) 점수를, 텍스트 전용 실험에서는 85.97%의 UAR 점수를, 그리고 멀티모달 접근법에서는 87.16%의 UAR 점수를 달성했습니다. 이는 멀티모달 접근법이 가장 우수하다는 것을 보여줍니다.



### ELIZA Reinterpreted: The world's first chatbot was not intended as a chatbot at a (https://arxiv.org/abs/2406.17650)
Comments:
          In review in IEEE Annals of the History of Computing (submitted Apr 2024)

- **What's New**: 이 논문은 최초의 채팅봇(ELIZA)이 개발된 역사를 풍부하게 탐구하고 있습니다. Joseph Weizenbaum이 MIT에서 개발한 ELIZA는 원래 인간-기계 대화를 연구하는 플랫폼으로 설계되었지만, 그 유명세와 우연한 공개로 인해 채팅봇으로 오해받게 되었습니다.

- **Technical Details**: ELIZA는 인공지능(AI) 역사의 주요 기술적 궤적들이 교차하는 지점에서 탄생하였습니다. Weizenbaum은 로저리아 정신분석가와 환자 사이의 대화를 모방하기 위해 단순한 패턴 매칭 알고리즘을 사용했습니다. 이 프로그램은 COMIT 언어로 작성되었으며, 자연어 처리를 연구하는 플랫폼으로 의도되었습니다.

- **Performance Highlights**: ELIZA의 개발은 우연한 프로그래밍 언어적 변화와 비의도적인 공개로 인해 오해와 혼동을 불러일으켰습니다. ELIZA는 실제로는 지능적이지 않았으며, Weizenbaum 자신도 이 프로그램이 지능적이라고 생각하지 않았습니다. 이는 후일 'ELIZA 효과'로 알려진 현상으로, 단순한 규칙 기반 대화가 복잡한 대화로 인식되는 상황을 설명합니다.



### Mitigate the Gap: Investigating Approaches for Improving Cross-Modal Alignment in CLIP (https://arxiv.org/abs/2406.17639)
- **What's New**: 최근 Contrastive Language-Image Pre-training (CLIP) 모델의 임베딩 공간에서 발생하는 모달리티 간의 차이(modality gap)를 줄이기 위한 새로운 접근법이 제안되었습니다. AlignCLIP은 멀티 모달 인코더(multi-modal encoder)의 파라미터 공간을 공유하고, 각각의 모달리티 내에서 임베딩을 보다 멀리 분리시키는 방법을 통해 이 차이를 줄이려는 시도를 합니다.

- **Technical Details**: 이 연구에서는 두 가지 주요 질문에 답하고자 합니다: 1. 멀티 모달 인코더의 파라미터 공간을 공유하면 모달리티 간의 차이가 줄어드는가? 2. 모달리티 내에서 임베딩을 분리시키는 것이 모달리티 간의 차이를 줄이는 데 도움이 되는가? 이를 위해 AlignCLIP은 트랜스포머 인코더와 프로젝션 레이어를 공유하고, 텍스트와 이미지를 각각 반대 모달리티 쪽으로 가까이 이동시키는 Intra-Modality Separation 목표를 도입합니다.

- **Performance Highlights**: AlignCLIP은 실험을 통해 모달리티 간의 임베딩 조정에서 상당한 개선을 이루어내었으며, 여러 다운스트림 작업에서 성능을 유지하거나 향상시켰습니다. 특히, 제로샷 이미지 분류(zero-shot image classification), 제로샷 멀티 모달 검색(zero-shot multi-modal retrieval), 및 제로샷 의미 텍스트 유사성(zero-shot semantic text similarity) 평가에서 긍정적인 결과를 보였습니다.



### Towards Building an End-to-End Multilingual Automatic Lyrics Transcription Mod (https://arxiv.org/abs/2406.17618)
Comments:
          Accepted at EUSIPCO 2024

- **What's New**: 이 논문에서는 다국어 자동 가사 전사(multilingual automatic lyrics transcription, ALT) 시스템을 구축하는 것을 목표로 하고 있습니다. 이는 제한된 라벨 데이터와 노래 때문에 생기는 추가적인 어려움에도 불구하고, 기존의 영어 ALT 기술들을 다국어 시나리오에 맞게 확장하여 적용합니다. 단일 언어 모델과 비교하여 다국어 모델의 성능을 평가하고, 언어 정보를 모델에 통합하는 다양한 방법도 탐구합니다.

- **Technical Details**: 모델은 최신 트랜스포머(transformer) 아키텍처를 기반으로 하며, 하이브리드 CTC/Attention 구조를 사용합니다. 입력은 16kHz 샘플링 속도, FFT 크기 400, 홉 크기 10ms의 80차원 Mel-spectrogram입니다. 모델은 컨볼루션 블록, 트랜스포머 인코더/디코더, 두 개의 완전 연결 레이어로 구성되었으며, 목표 차원은 목표 문자 집합의 크기와 같습니다. 훈련 중, 모델은 교사 강요(teacher forcing)를 통해 더 빠른 수렴을 이룹니다. 손실 함수는 CTC 손실과 Kullback-Leibler(KL) 발산 손실의 가중 합으로 구성됩니다.

- **Performance Highlights**: 다국어 모델이 단일 언어 모델보다 일관되게 더 나은 성능을 보였으며, 언어 정보를 통합하면 성능이 크게 향상된다는 것을 발견했습니다. 저자들은 다국어 모델이 저자원(low-resource) 환경에서 특히 성능이 뛰어나다는 점을 강조하며, 언어 간 균형 불균형 문제를 다루었습니다.



### NativE: Multi-modal Knowledge Graph Completion in the Wild (https://arxiv.org/abs/2406.17605)
Comments:
          Accepted by SIGIR 2024 as a full paper

- **What's New**: 새로운 프레임워크 NativE는 실제 환경에서의 멀티-모달 지식 그래프 완성(MMKGC)을 목표로 하고 있습니다. 이 프레임워크는 다양한 유형(예: 이미지, 텍스트, 숫자, 오디오, 비디오)의 모달리티 정보를 조정(fusion)하기 위해 관계 안내 듀얼 적응형 융합 모듈(relation-guided dual adaptive fusion module)을 제안합니다. 또한, 불균형한 모달리티 정보를 보완하기 위해 공동 모달리티 적대적 훈련(collaborative modality adversarial training) 프레임워크를 사용합니다.

- **Technical Details**: NativE는 다음의 주요 구성 요소를 포함합니다: 
1. 관계 안내 듀얼 적응형 융합 모듈: 이는 다양한 모달리티의 정보를 조정(fusion)하고 통합하는 데 도움을 줍니다. 
2. 공동 모달리티 적대적 훈련 프레임워크: 이는 모달리티 정보의 불균형 문제를 보완하는데 기여합니다. 이를 통해 다양한 모달리티의 데이터를 효과적으로 사용할 수 있습니다. 또한, 새로운 벤치마크인 WildKGC를 구성하여 다섯 가지 데이터셋을 포함하고 있습니다.

- **Performance Highlights**: 21개의 최근 기준선과 비교한 경험적 결과는 NativE의 우수성을 확인해줍니다. NativE는 다양한 데이터셋과 시나리오에서 일관되게 최첨단 성능(state-of-the-art performance)을 달성하며 효율성과 범용성을 유지합니다. 관련 코드와 데이터는 공개되어 있으니 참조하시기 바랍니다.



### CDQuant: Accurate Post-training Weight Quantization of Large Pre-trained Models using Greedy Coordinate Descen (https://arxiv.org/abs/2406.17542)
- **What's New**: 새로운 논문에서는 GPTQ 알고리즘을 대체할 수 있는 효율적이고 간단한 대안, CDQuant를 소개하고 있습니다. CDQuant는 좌표 하강법(coordinate descent)을 사용하여 레이어별 재구성 손실을 최소화하여 고품질의 양자화된 가중치를 달성합니다. 여러 모델 크기와 양자화 수준에서 CDQuant가 GPTQ보다 일관되게 뛰어난 성능을 보임을 PaLM2 모델 패밀리를 통해 입증했습니다.

- **Technical Details**: CDQuant는 GPTQ와 달리 반복적 최적화 기술을 사용하여 각 반복마다 하강할 좌표를 탐색합니다. 이는 GPTQ가 좌표를 한 번씩 순환하는 것과는 대조적입니다. 또한 CDQuant는 그룹/서브-채널 양자화(group/sub-channel quantization)로 확장될 수 있습니다. 이 알고리즘은 최대 수백억 개의 파라미터를 가진 모델에도 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: PaLM2-Otter의 INT2 양자화에서 CDQuant는 GPTQ에 비해 당혹도(perplexity)를 10% 감소시키는 성과를 보였습니다. 이를 통해 CDQuant가 다양한 양자화 정밀도 수준 및 모델 크기에서 GPTQ를 뛰어넘는 성능을 발휘함을 확인했습니다.



### Can Large Language Models Understand DL-Lite Ontologies? An Empirical Study (https://arxiv.org/abs/2406.17532)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 DL-Lite 온톨로지에 대한 이해 능력을 실험적으로 분석하였습니다. DL-Lite는 DL(Description Logic) 온톨로지 계열 중 하나로, 단순한 구조와 효율적인 추론을 특징으로 합니다. 본 연구는 LLMs가 DL-Lite 온톨로지의 문법적(Syntactic) 및 의미적(Semantic) 해석을 어느 정도 이해할 수 있는지를 다각적으로 평가하였습니다.

- **Technical Details**: 문법적인 측면에서, LLMs가 DL-Lite의 구조적 규칙, 유효 명제, 표현을 이해할 수 있는지를 조사했습니다. 의미적인 측면에서는 개념과 역할의 내포(intension)와 외연(extension) 이해 능력을 평가하기 위해 서브섬션(subsumption)과 인스턴스 확인(instance checking)을 실시했습니다. 추가적으로 역역할(inverse roles) 및 기능적 역할(functional roles)과 같은 속성 특성을 탐구하고, 질의 응답(query answering) 및 온톨로지 만족도 확인(ontology satisfiability checking)을 통해 온톨로지 전체의 의미를 이해할 수 있는지 평가했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 다음과 같은 결과를 얻었습니다: 
  - LLMs는 DL-Lite 문법을 이해할 수 있는 능력을 보였습니다.
  - 개념과 역할 그리고 일부 속성 특성의 의미를 이해할 수 있었습니다.
  - 그러나, 트랜지티비티(transitivity) 규칙 이해에 어려움을 겪어 개념 또는 역할의 서브섬션(subsumption) 능력이 제한적이었습니다.
  - 큰 규모의 ABox를 포함한 온톨로지를 처리하는 데 어려움을 겪어 인스턴스 확인 및 질의 응답 능력이 제한적이었습니다.
  - 온톨로지 만족도 확인은 가능하나, 복잡한 온톨로지 내 불일치를 감지하는 데에는 어려움이 있었습니다.



### AG-LSEC: Audio Grounded Lexical Speaker Error Correction (https://arxiv.org/abs/2406.17266)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 새로운 연구는 Lexical Speaker Error Correction (LSEC) 시스템을 기존의 EEND 기반 화자 식별 시스템에서 직접 추출한 화자 점수로 강화하여 제안합니다. 이는 기존 오디오 기반 SD 및 ASR 시스템보다 최대 40% 상대적인 단어 화자 오류율(WDER)을 감소시키며, 기존 LSEC 시스템보다도 15-25% 상대적인 향상을 보입니다.

- **Technical Details**: 제안된 AG-LSEC는 세 가지 주요 구성 요소로 이루어져 있습니다: Acoustic Speaker Score Extractor, Backbone LM (Language Model), Transformer Encoder Front-end. 이 시스템은 EEND (End-to-end Neural Diarization) 시스템을 사용하여 화자 점수를 추출하고, 이를 두 가지 퓨전 기술을 통해 Lexical Speaker Error Correction 모델에 통합합니다.

- **Performance Highlights**: RT03-CTS, Callhome American English, Fisher 데이터셋에서 기존 LSEC 시스템 대비 15-25% 상대적 WDER 감소를 달성했습니다. 오디오 기반 SD 및 ASR 시스템에 비해 최대 40%의 상대적 WDER 감소를 확인할 수 있었습니다.



### Unlocking Continual Learning Abilities in Language Models (https://arxiv.org/abs/2406.17245)
Comments:
          preprint, 19 pages

- **What's New**: 신규 연구는 Language Models (LMs)에서 지속적 학습(continual learning, CL) 중 발생하는 'catastrophic forgetting' 문제를 해결하기 위한 새로운 방법인 MIGU (Magnitude-based Gradient Updating)를 소개합니다. MIGU는 LM의 선형 계층 출력의 크기가 큰 매개변수만 업데이트하는 방식으로, 이전 과제 데이터나 과제 레이블(task labels) 없이도 CL 성능을 향상시킵니다.

- **Technical Details**: MIGU는 각 과제를 처리할 때 LM 선형 계층 출력의 L1-정규화 크기 분포가 다르다는 점을 활용합니다. 이 분포의 차이를 이용하여, 크기가 큰 출력에 의한 매개변수만 업데이트하는 방식입니다. 이를 통해, 과제 간의 기울기(conflict) 문제를 해소하고 더 나은 지속적 학습 성능을 달성합니다. 방법은 세 가지 LM 아키텍처 (T5, RoBERTa, Llama2)에 모두 적용 가능하며, 기존 지속적 전이 학습(continual finetuning)과 지속적 사전 학습(continual pre-training)의 설정에 사용됩니다.

- **Performance Highlights**: MIGU는 네 가지 클 벤치마크에서 기존의 최첨단 방법과 동등하거나 더 나은 성능을 보였습니다. 예를 들어, 15개의 연속적인 일이 포함된 CL 벤치마크에서 기존 매개변수 효율적 미세 조정(parameter-efficient finetuning) 기법 대비 15.2%의 평균 정확도 향상을 달성했습니다. 또한, MIGU는 기존의 세 가지 CL 방법(rehearsal-based, architecture-based, parameter-based)과도 결합하여 성능을 더욱 향상시킬 수 있습니다.



### Self-Constructed Context Decompilation with Fined-grained Alignment Enhancemen (https://arxiv.org/abs/2406.17233)
Comments:
          Under Review

- **What's New**: 이 논문은 소스 코드 없이 컴파일된 코드를 고급 프로그래밍 언어로 변환하는 디컴파일(decompilation) 기술에 대한 새로운 접근 방안을 제안합니다. 특히, 더 큰 모델 파라미터나 학습 데이터를 사용하는 기존 방법 이외에 두 가지 혁신적인 방법을 소개합니다. 첫째, Self-Constructed Context Decompilation (sc$^2$dec) 방법을 사용하여 모델의 디컴파일 결과를 재컴파일하고, 이를 인-컨텍스트 학습(in-context learning) 쌍으로 구성하여 성능을 향상시킵니다. 둘째, Fine-grained Alignment Enhancement (FAE) 방법을 사용하여 어셈블리 코드와 소스 코드를 문장 수준에서 정밀하게 정렬하며, 이는 디버깅 정보를 활용하여 디컴파일 성능을 더욱 향상시킵니다.

- **Technical Details**: sc$^2$dec 방법은 추가적인 파인 튜닝(fine tuning) 없이도 모델의 디컴파일 성능을 높이는 방법으로, LLM의 디컴파일 결과를 재컴파일하여 인-컨텍스트 학습에 사용할 쌍을 생성합니다. 이 과정은 모델이 더욱 효과적으로 디컴파일을 수행할 수 있게 도와줍니다. 반면, FAE는 어셈블리 코드와 소스 코드를 문장 단위로 정밀하게 정렬하여 더 좋은 디컴파일 결과를 얻게 합니다. 디버깅 정보를 활용하여 더 나은 정렬을 구현하며, 이는 파인 튜닝 단계에서 적용됩니다.

- **Performance Highlights**: 위 두 가지 방법을 통합 적용하여 Decompile-Eval 벤치마크에서 Re-Executability 성능을 약 7.35% 향상시켜, 새로운 state-of-the-art 성능인 55.03%를 달성했습니다.



### Large Language Models are Interpretable Learners (https://arxiv.org/abs/2406.17224)
Comments:
          Preliminary Version, Code at [this url](this https URL)

- **What's New**: 최신 연구에서는 해석 가능성과 표현력 간의 균형을 맞추는 문제를 해결하려고 합니다. 큰 언어 모델(LLMs)과 심볼릭 프로그램(sybolic programs)를 결합한 LLM 기반 심볼릭 프로그램(LSPs)을 제안합니다. 이 방법은 자연어 프롬프트를 사용하여 해석 가능한 모듈을 제공하고, 이를 통합하여 해석 가능한 의사결정 규칙을 만듭니다.

- **Technical Details**: LSP는 두 가지 주요 요소를 사용합니다. 첫째, 도메인별 언어(DSL)로 정의된 프롬프트-LLM(prompted-LLM)과 조건부 브랜칭(conditional branching)을 사용하여 의사결정 트리를 구성합니다. 둘째, 분할 정복 접근법을 통해 프로그램을 점진적으로 학습합니다. 학습 과정은 각 단계에서 LLM들에 의해 가이드됩니다. 또한 다양한 예측 작업을 포함하는 IL-Bench 벤치마크를 사용하여 LSP의 효과성을 평가했습니다.

- **Performance Highlights**: LSP는 전통적인 뉴로심볼릭 프로그램(neurosymbolic programs)과 자동화된 프롬프트 튜닝 방법보다 우수한 성능을 보였습니다. LSP는 학습된 지식을 자연어 설명과 심볼릭 규칙의 조합으로 제공하여 인간에게 쉽게 전달될 수 있으며, 다른 LLM에도 잘 일반화됩니다.



### Testing network clustering algorithms with Natural Language Processing (https://arxiv.org/abs/2406.17135)
Comments:
          10 pages, 8 figures

- **What's New**: 이 연구는 온라인 소셜 네트워크와 텍스트 생산을 통해 개인의 성격을 분석하는 데 중점을 둡니다. 소셜 네트워크의 복잡한 상호작용을 기반으로 커뮤니티 탐지 알고리즘 (Community Detection Algorithm, CDA)과 자연어 처리 분류 알고리즘 (Natural Language Processing Classification Algorithm, NLPCA)을 결합하여 새로운 평가 방식을 제안합니다. 이 방법을 통해 온라인 소셜 그룹을 텍스트 분류에 기반한 문화적 그룹으로 정의하며, 이 접근 방식이 사용자 의견을 85% 이상의 정확도로 분류할 수 있음을 제시합니다.

- **Technical Details**: 연구는 트위터 데이터를 사용하여 기후 변화 관련 토론에서 소셜 그룹을 식별하는 사례 연구를 수행했습니다. 데이터를 분석하기 위해 Louvain 커뮤니티 탐지 알고리즘을 사용했으며, 2022년 동안 수집된 57M개의 트윗 중 32.1M개의 리트윗을 분석하여 무게가 있는 무향 그래프를 생성했습니다. 이 그래프는 약 226,000개의 노드와 430,000개의 엣지로 구성되었으며, 내부 연결 정도와 외부 연결 정도를 최적화하는 다양한 점수 함수가 사용되었습니다. 연구는 또한 NLPCA를 ‘그라운드 트루스(groud-truth)’ 커뮤니티 구조로 사용하여 다양한 CDA를 테스트했습니다.

- **Performance Highlights**: 이 연구의 주요 성과 중 하나는 다양한 CDA와 NLPCA를 조합하여 사용자가 속한 소셜 그룹을 예측하는데 있어 높은 정확도를 달성했다는 점입니다. 특히, Twitter 데이터에서 기후 변화 관련 토론에 대한 다양한 소셜 그룹을 식별했으며, 이 과정을 통해 미국 민주당과 트럼프 지지자들 간의 커뮤니티를 분명히 구분할 수 있었습니다. 연구 결과, 무작위 사용자에 대한 의견을 85% 이상의 정확도로 분류할 수 있는 CDA/NLPCA 쌍이 여러 개 존재함을 확인했습니다.



### Unveiling LLM Mechanisms Through Neural ODEs and Control Theory (https://arxiv.org/abs/2406.16985)
- **What's New**: 이번 연구는 뉴럴 상미분 방정식(Neural Ordinary Differential Equations, Neural ODEs)을 활용하여 대규모 언어 모델(LLMs)의 입력과 출력 사이의 복잡한 관계를 명확히 하고, 강건 제어(robust control)를 통해 출력을 미세 조정하는 혁신적인 방법을 제시합니다. Neural ODEs는 LLM의 내부 데이터를 동적으로 모델링하여 연속적인 변화를 포착합니다. 이 접근법은 이전에 불투명했던 LLM의 메커니즘을 해명하고, 설명 가능한 AI(explainable AI)에 중대한 기여를 합니다.

- **Technical Details**: 우리의 접근법은 LLM의 입력과 출력을 저차원 잠재 공간(latent space)으로 변환하여 정보 처리 경로를 세밀하게 조사합니다. 이 과정에서 Neural ODEs가 중요한 역할을 하여 데이터의 연속적인 진화를 포착합니다. 또한, 강건 제어 메커니즘을 적용하여 모델의 출력을 전략적으로 조정하고, 높은 품질과 신뢰성을 유지하면서 특정 성능 기준을 충족시키도록 합니다. 이러한 Neural ODEs와 강건 제어의 결합은 LLM해석 가능성에 큰 도약을 가져옵니다.

- **Performance Highlights**: 우리의 실험 결과는 이 통합 접근법의 효과성을 입증합니다. Neural ODE 모델이 강건 제어 메커니즘과 결합하면 LLM의 안정성 및 일반화 성능이 크게 향상됩니다. 이는 고위험 분야에서 신뢰할 수 있고 투명한 AI 시스템의 개발에 중요한 이정표를 세우며, 앞으로 연구가 이루어질 방향을 제시합니다.



### MetaGreen: Meta-Learning Inspired Transformer Selection for Green Semantic Communication (https://arxiv.org/abs/2406.16962)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2310.07592

- **What's New**: Semantic Communication(의미 통신)이 전송할 정보의 의미와 효과적인 내용을 우선시하게 되어, 전통적인 통신 방식보다 지연 시간을 줄이고, 대역폭 사용을 낮추며, 더 높은 처리량을 약속합니다. 하지만 의미 통신 발전은 의미 정보 손실과 에너지 소비의 공동 효과를 기준화할 수 있는 보편적인 지표의 필요라는 중요한 도전에 직면해 있습니다. 이 연구는 이를 해결하기 위해 '에너지 최적화 의미 손실'(EOSL) 함수라는 혁신적인 다목적 손실 함수를 도입했습니다.

- **Technical Details**: EOSL 함수는 의미 정보 손실과 에너지 소비를 효과적으로 균형 잡는 다목적 손실 함수입니다. 트랜스포머 모델(transformer model)에서 에너지 벤치마킹을 포함한 광범위한 실험을 통해, EOSL 기반 모델 선택이 BLEU 점수 기반 선택보다 최대 83% 더 나은 유사성 대 전력 비율(SPR)을 달성한다는 것을 입증했습니다. 또한, Meta-Learning 원칙에서 영감을 받아 다양한 문맥에 EOSL을 적용할 수 있음을 보여주었습니다.

- **Performance Highlights**: EOSL 기반 트랜스포머 모델 선택은 BLEU 점수 기반 선택보다 최대 83% 더 나은 Similarity-to-Power Ratio(SPR)를 달성했습니다. 단순히 최저 전력 사용량 기반 선택과 비교했을 때는 67% 더 나은 SPR을 달성했습니다. 이는 에너지 효율적 모델 선택과 '그린' 의미 통신 시스템 발전의 기초를 놓는 중요한 연구 결과입니다.



### Validation of a new, minimally-invasive, software smartphone device to predict sleep apnea and its severity: transversal study (https://arxiv.org/abs/2406.16953)
Comments:
          21 pages, 6 figures

- **What's New**: 이 논문은 Apneal이라는 애플리케이션을 도입한 소규모 개념 증명 연구를 다루고 있습니다. 이 앱은 스마트폰의 마이크를 이용해 소리를 기록하고, 스마트폰의 가속도계와 자이로스코프를 통해 움직임을 기록하여 환자의 AHI를 추정합니다. 연구 결과는 새롭게 소개된 Apneal의 자동 감지 기능이 유망한 성과를 보여줍니다.

- **Technical Details**: 실험은 병원 내에서 다폴리스올로그래피(PSG)를 시행하는 성인 환자들을 대상으로 진행되었습니다. 연구는 수작업 점수 매기기(manual scoring) 단계와 기록된 신호에서 호흡 이벤트를 자동으로 감지하는 심층 학습 모델(deep-learning model)을 사용해 수행되었습니다. 이 모델은 2022년 말에 Apneal 내부에서 버전 0.1로 출시되었습니다. 진단 성능은 민감도(sensitivity)와 양성 예측도(PPV)를 사용해 평가되었습니다.

- **Performance Highlights**: AHI가 15 이상인 경우, 수작업 점수 매기기의 민감도는 0.91, PPV는 0.89 였습니다. AHI가 30 이상인 경우, 민감도는 0.85, PPV는 0.94였습니다. AHI가 15 이상인 경우의 AUC-ROC는 0.85, AUC-PR은 0.94였고, AHI가 30 이상인 경우의 AUC-ROC는 0.95, AUC-PR은 0.93이였습니다. 이러한 결과는 스마트폰 기반 신호의 수작업 점수가 PSG 기반 점수와 비교해 가능하고 정확함을 보여줍니다. 심층 학습 모델에 기반한 자동 점수 매기기 방법은 유망한 결과를 보여주었습니다.



### Towards a copilot in BIM authoring tool using a large language model-based agent for intelligent human-machine interaction (https://arxiv.org/abs/2406.16903)
- **What's New**: 디자인 자동화 분야에서의 혁신적인 접근 방식으로, 연구팀은 BIM 저작 소프트웨어(Vectorworks)에서 사용될 수 있는 LLM 기반 자율 에이전트 프레임워크를 제안했습니다. 이 프레임워크는 소프트웨어 사용 질문에 답변하고, 자연어로 사용자의 디자인 의도를 이해하며, 적절한 도구를 호출하여 모델링 작업을 자율적으로 실행할 수 있습니다.

- **Technical Details**: 이 프레임워크는 Large Language Models (LLMs)를 활용하여, 사용자가 자연어로 전달하는 디자인 의도를 분석하고, 이에 맞는 작업을 자동으로 수행합니다. 연구팀은 다양한 LLM들을 사용하여 복잡한 지시 사항에 대한 계획 및 추론 능력을 평가하였으며, 결과적으로 제안된 프레임워크를 BIM 저작 시나리오에 매끄럽게 통합할 수 있는 소프트웨어 프로토타입을 구현했습니다.

- **Performance Highlights**: 케이스 연구를 통해, 제안된 LLM 기반 에이전트가 디자인 자동화 및 지능형 상호작용에서 상당한 잠재력을 가지고 있음을 입증했습니다. 특히, 디자인 작업에서 소프트웨어 사용의 복잡성과 어려움을 최소화하면서, 디자이너가 본연의 디자인 프로세스에 집중할 수 있도록 도와줍니다.



New uploads on arXiv(cs.IR)

### Light-weight End-to-End Graph Interest Network for CTR Prediction in E-commerce Search (https://arxiv.org/abs/2406.17745)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 아카이브 논문은 전자상거래 검색에서 클릭률(CTR) 예측을 향상시키기 위해 고안된 새로운 접근법, 즉 'Light-weight End-to-End Graph Interest Network (EGIN)'을 소개합니다. EGIN은 사용자의 검색 의도를 효과적으로 추출하여 이전 모델들의 한계점을 해결합니다.

- **Technical Details**: EGIN은 쿼리와 아이템의 상관 관계 및 순차 정보를 사용하여 이기종 그래프(heterogeneous graph)를 구축합니다. 그래프 임베딩(graph embedding) 학습과 CTR 예측을 통합하여 단일 프레임워크에서 학습 과정이 진행되며, 이는 대규모 검색 시스템에도 쉽게 배포할 수 있게 합니다. 이 모델은 쿼리-아이템 이기종 그래프, 경량화된 그래프 샘플링(light-weight graph sampling), 다중 관심 네트워크(multi-interest network) 세 부분으로 구성되어 있습니다.

- **Performance Highlights**: 광범위한 실험 결과 공공 및 산업 데이터셋 모두에서 EGIN의 효과가 입증되었습니다. 또한, 그래프 학습의 훈련 비용이 주 CTR 예측 작업에 비해 상대적으로 낮아 실제 애플리케이션에서도 효율적입니다. 온라인 A/B 테스트를 통해 제안된 접근 방법의 생산성이 확인되었습니다.



### ACE: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling (https://arxiv.org/abs/2406.17507)
- **What's New**: 이번 논문에서는 다양한 모달리티(multi-modal)의 데이터를 효과적으로 검색하기 위한 새로운 프레임워크, ACE(Generative Cross-modal Retrieval Framework)를 제안합니다. 이 프레임워크는 텍스트, 이미지, 오디오 및 비디오 간의 크로스-모달 검색을 종단간(end-to-end)으로 구현합니다. 기존의 dual-tower 아키텍처와 달리, ACE는 K-Means와 RQ-VAE 알고리즘을 사용하여 다중 모달리티 데이터에 고유한 식별자를 생성합니다.

- **Technical Details**: ACE 프레임워크는 세 가지 주요 모듈을 포함합니다: 종류별 식별자 생성, 쿼리 생성 및 특징 융합입니다. 이 프레임워크는 K-Means 알고리즘과 Residual Quantized Variational Autoencoder(RQ-VAE)를 결합하여 계층적 의미 정보를 가지는 식별자를 생성합니다. 또한 코스(coarse)-투-파인(fine) 특징 융합 전략을 설계하여 자연어 쿼리와 후보 식별자 간의 의미 격차를 줄입니다. 이 융합 전략은 인코더-디코더 구조에서 각 인코더 레이어의 출력을 다르게 가중치하여 세밀한 정보를 포착할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험 결과, ACE는 기존의 강력한 기준선을 평균 15.27% 초과하는 성능을 보여주었습니다. ACE는 텍스트-이미지, 텍스트-오디오, 텍스트-비디오 검색 모두에서 state-of-the-art 성능을 달성하였으며, CPU와 GPU 모두에서 안정적인 검색 속도를 유지합니다. 또한 ACE는 개별 모듈의 효율성을 보여주는 에블레이션(ablative) 실험과 의미 기반 융합 전략의 견고성을 증명하는 추가 분석을 통해 강력한 성능을 입증합니다.



### Performative Debias with Fair-exposure Optimization Driven by Strategic Agents in Recommender Systems (https://arxiv.org/abs/2406.17475)
Comments:
          SIGKDD 2024 accepted paper

- **What's New**: 이번 연구는 추천 시스템에서의 인기 편향(popularity bias) 문제를 다루는 새로운 재정렬(re-ranking) 접근 방식을 소개합니다. 특히 이 연구는 생산자 측면에서의 전략적 학습 에이전트를 활용한 공정 노출 최적화에 중점을 두고 있습니다. 이는 콘텐츠 제작자들이 최대 노출을 위해 아이템 특징을 전략적으로 수정할 수 있는 환경을 가정합니다.

- **Technical Details**: 재정렬 접근 방식은 차별 가능한 순위 연산자(differentiable ranking operators)를 통해 정확성과 공정성을 동시에 겨냥한 엔드투엔드(end-to-end) 최적화를 실행합니다. 이 과정에서 사용자 표현과 아이템 특징의 내적(inner product)을 중심으로 랭킹 알고리즘이 작동합니다. 또한, 전략 에이전트의 실행 규칙을 구현하여 예측의 실행 가능성을 활용합니다. 이는 시간에 따른 데이터의 변화를 반영하는 전략적 학습을 포함합니다.

- **Performance Highlights**: 공공 및 산업 데이터셋을 통한 종합 실험 결과, 제안된 방법이 특별히 롱테일 아이템의 잠재력을 발휘하는 데 효과적이고 우세함을 입증했습니다. 이 연구는 특히 추천 시스템의 두 가지 목표인 정확성과 공정성을 균형 있게 최적화하는 새로운 차별 가능 연산자를 도입하여, 사용자에게 롱테일 아이템에 대한 정보를 더 많이 포함할 수 있도록 합니다.



### A Thorough Performance Benchmarking on Lightweight Embedding-based Recommender Systems (https://arxiv.org/abs/2406.17335)
- **What's New**: 이 논문은 다양한 경량 임베딩 기반 추천 시스템(LERS)의 성능, 효율성, 그리고 크로스 태스크 전송성을 철저하게 벤치마킹하여 분석하였습니다. 또한, 단순하지만 높은 경쟁력을 갖춘 모수 가지치기(magnitude pruning)를 이용한 효율적인 임베딩 압축 방법을 제안합니다.

- **Technical Details**: 많은 최신 추천 시스템(RSs)은 사용자 직업과 같은 범주형 특징에 의존하여 임베딩 벡터로 인코딩되며, 이는 거대한 임베딩 테이블로 이어집니다. 이를 압축하기 위해 다양한 방법들이 제안되었습니다. 그 주요 방식으로는 합성 임베딩(compositional embedding), 임베딩 가지치기(embedding pruning), 신경망 구조 탐색(NAS, Neural Architecture Search)이 포함됩니다.

- **Performance Highlights**: 제안된 모수 가지치기 방법은 다양한 복잡한 LERS보다 뛰어난 성능을 보였습니다. 이를 확인하기 위해 라즈베리 파이 4(Raspberry Pi 4)에서 테스트를 진행하며 경량 LERS의 효율성 병목 현상을 드러냈습니다.



### Hyperbolic Knowledge Transfer in Cross-Domain Recommendation System (https://arxiv.org/abs/2406.17289)
- **What's New**: 이번 연구는 Cross-Domain Recommendation (CDR) 시스템의 성능 향상을 위해 하이퍼볼릭 표현(hyperbolic representation)을 사용하는 새로운 프레임워크 Hyperbolic Contrastive Learning (HCTS)을 제안합니다. 이는 다양한 도메인 간의 데이터 통합시 발생하는 롱테일 분포 문제를 완화하고, 각 도메인의 고유한 특성을 효과적으로 반영하는 것을 목표로 합니다.

- **Technical Details**: HCTS는 사용자와 아이템을 각각의 하이퍼볼릭 매니폴드(hyperbolic manifold)에 각각 임베딩하고, 이를 통해 예측을 진행합니다. 또한, 하이퍼볼릭 대비학습 모듈(hyperbolic contrastive learning module)을 통해 타겟 도메인에서의 사용자와 아이템의 표현을 개선합니다. 특히, 각 도메인에서 개별적으로 GNN 모듈을 사용하여 이웃 전파를 수행하고, 두 곡률 적응 하이퍼볼릭 매니폴드에 임베딩을 합니다. 지식 전이를 위해, 매니폴드 정렬(manifold alignment)과 세 가지의 대비학습 전략을 사용합니다: user-user, user-item, item-item 대비학습.

- **Performance Highlights**: 실제 데이터를 사용한 실험에서는 하이퍼볼릭 매니폴드가 유클리드 공간에 비해 CDR 과제에서 더 우수한 성능을 보임을 입증하였습니다. 이는 하이퍼볼릭 방법이 유클리드 방법에 비해 롱테일 데이터 분포를 더 잘 처리할 수 있음을 시사합니다.



### Debiased Recommendation with Noisy Feedback (https://arxiv.org/abs/2406.17182)
Comments:
          KDD 24 Research Track Paper

- **What's New**: 이 연구에서는 사용자 평점이 임의로 누락(MNAR)되는 데이터와 관찰된 평점이 실제 사용자 선호도와 일치하지 않는 노이즈 피드백 또는 결과 측정 오류(OME)가 동시에 존재하는 경우, 추천 시스템에서 예측 모델을 편향 없이 학습하는 방법을 제안합니다. 기존의 EIB, IPS, DR 방법을 확장하여 OME-EIB, OME-IPS, OME-DR 추정기를 설계하고, 이 방법들이 이론적으로 편향되지 않음을 증명했습니다.

- **Technical Details**: 사용자가 평점을 자유롭게 선택할 수 있는 권한이 있어 대부분의 항목에 대한 사용자 평점이 임의로 누락될 가능성이 큽니다. 이러한 MNAR 데이터를 고려하여 예측 모델을 학습하기 위해 기존에는 오차 대체 기반(EIB), 역확률평가(IPS), 이중 강건성(DR) 방법이 사용되었습니다. 그러나 이 연구에서는 실제 사용자의 선호도와 관찰된 평점 간의 불일치로 발생하는 노이즈 피드백(OME)도 함께 고려합니다. 이를 위해 OME-EIB, OME-IPS, OME-DR 추정기를 설계하고, 측정 오류 매개변수를 추정하여 데이터 MNAR와 OME를 동시에 교정하는 대체적 노이즈 제거 학습 접근법을 제안합니다.

- **Performance Highlights**: 제안된 방법들은 MNAR 데이터와 OME의 비율을 다르게 설정한 세 가지 실제 데이터셋과 하나의 반합성 데이터셋에서 검증되었습니다. 실험 결과, 제안된 방법들은 기존 방법들보다 우월한 성능을 보였으며, 이는 OME와 선택 편향이 상호 작용하여 편향되지 않은 추천을 제공하는 데 효과적임을 나타냅니다.



### NativE: Multi-modal Knowledge Graph Completion in the Wild (https://arxiv.org/abs/2406.17605)
Comments:
          Accepted by SIGIR 2024 as a full paper

- **What's New**: 새로운 프레임워크 NativE는 실제 환경에서의 멀티-모달 지식 그래프 완성(MMKGC)을 목표로 하고 있습니다. 이 프레임워크는 다양한 유형(예: 이미지, 텍스트, 숫자, 오디오, 비디오)의 모달리티 정보를 조정(fusion)하기 위해 관계 안내 듀얼 적응형 융합 모듈(relation-guided dual adaptive fusion module)을 제안합니다. 또한, 불균형한 모달리티 정보를 보완하기 위해 공동 모달리티 적대적 훈련(collaborative modality adversarial training) 프레임워크를 사용합니다.

- **Technical Details**: NativE는 다음의 주요 구성 요소를 포함합니다: 
1. 관계 안내 듀얼 적응형 융합 모듈: 이는 다양한 모달리티의 정보를 조정(fusion)하고 통합하는 데 도움을 줍니다. 
2. 공동 모달리티 적대적 훈련 프레임워크: 이는 모달리티 정보의 불균형 문제를 보완하는데 기여합니다. 이를 통해 다양한 모달리티의 데이터를 효과적으로 사용할 수 있습니다. 또한, 새로운 벤치마크인 WildKGC를 구성하여 다섯 가지 데이터셋을 포함하고 있습니다.

- **Performance Highlights**: 21개의 최근 기준선과 비교한 경험적 결과는 NativE의 우수성을 확인해줍니다. NativE는 다양한 데이터셋과 시나리오에서 일관되게 최첨단 성능(state-of-the-art performance)을 달성하며 효율성과 범용성을 유지합니다. 관련 코드와 데이터는 공개되어 있으니 참조하시기 바랍니다.



### LumberChunker: Long-Form Narrative Document Segmentation (https://arxiv.org/abs/2406.17526)
- **What's New**: LumberChunker는 문서 세분화(segmentation)를 위해 LLM을 활용해 동적으로 문서를 분할하는 새로운 방법을 제안합니다. 이 방법은 문서의 의미적 독립성을 더 잘 포착할 수 있도록 내용 전환점을 식별하도록 LLM을 반복적으로 사용하는 것이 특징입니다. 또한, 새롭게 만들어진 GutenQA 벤치마크를 통해 이 방법의 성능을 평가합니다.

- **Technical Details**: LumberChunker는 문서를 동적으로 세분화하기 위해 LLM을 사용합니다. 초기에는 문서를 단락별로 분할하고, 그런 다음 시퀀스의 각 단락을 분석해 내용이 전환되는 지점을 식별합니다. 설정된 토큰 수 임계값(θ)을 넘는 그룹을 만들어 이를 기반으로 세분화 지점을 식별합니다. 전체 프로세스는 LLM(Gemini 1.0-Pro)을 이용해 실행됩니다. 또한, 100권의 공개 도메인 서적으로 구성된 GutenQA 벤치마크에서 3000개의 고품질 QA 쌍을 생성하여 성능을 평가합니다.

- **Performance Highlights**: LumberChunker는 가장 경쟁력 있는 기준점보다 7.37% 높은 검색 성능(DCG@20)을 보여줍니다. 또한, RAG 파이프라인에 통합되었을 때 다른 세분화 방법들과 비교하여 더 효과적임을 입증했습니다. 특히, DCG@20 및 Recall@20에서 가장 높은 점수를 기록하여 탁월한 성능을 입증했습니다 (DCG@20에서 62.09, Recall@20에서 77.92).



### A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens (https://arxiv.org/abs/2406.17378)
Comments:
          Work in Progress

- **What's New**: 대형 언어 모델(LLM)에서의 텍스트 임베딩(text embedding)이 입력 텍스트의 주요 토큰(key tokens)과 정렬될 수 있음을 발견했습니다. 이 현상은 모델 아키텍처, 학습 전략, 임베딩 방법에 관계없이 보편적으로 존재합니다. 이를 통해 정보 검색, 의미론적 텍스트 유사성 등 여러 분야에서 새로운 가능성을 열어줍니다.

- **Technical Details**: 본 연구에서는 8개의 임베딩 LLM을 분석하여 임베딩 LLM과 원래 생성 LLM 사이의 주요 변화가 첫 번째 주성분(principal component)에 집중된다는 점을 발견했습니다. 이 주성분을 조정함으로써 텍스트 임베딩을 주요 토큰과 정렬할 수 있습니다. 또한, 텍스트 임베딩이 decoder layer를 통과할 때 가장 높은 디코딩 확률을 가지는 토큰들이 입력 텍스트와 높은 관련이 있음을 확인했습니다.

- **Performance Highlights**: 새로운 sparse retrieval 방법을 제안하며, 이는 동일 모델의 dense retrieval 성능의 80%를 달성하면서도 계산량을 크게 줄였습니다. 또한, BM25 및 SPLADE v2 같은 강력한 베이스라인을 능가하는 성과를 보여주었습니다. 이 발견은 의미론적 유사성 및 지시 따르기(embeddiing) 기술을 이해하는 데 새로운 시각을 제공합니다.



### DEXTER: A Benchmark for open-domain Complex Question Answering using LLMs (https://arxiv.org/abs/2406.17158)
Comments:
          under submission, 22 pages

- **What's New**: 이번 연구에서는 열린 도메인 복합 질문 응답(Open-domain complex Question Answering, QA) 문제를 해결하기 위해 새로운 벤치마크를 제안하고, 최신의 사전 학습된 밀집(dense) 및 희소(sparse) 검색 모델을 평가하는 도구를 제공합니다. 특히, 혼합된 증거(hybrid evidence)와 조합형 질문(compositional questions) 등의 복잡한 검색 작업에서 기존 연구가 부족한 점을 보완하고자 했습니다.

- **Technical Details**: 본 연구는 다양한 복합 질문 응답 작업을 포함하는 '덱스터(DEXTER)'라는 벤치마크와 이와 관련된 툴킷을 제안합니다. 덱스터 벤치마크는 7개의 다양한 데이터셋을 포함하며, 질문의 복잡성, 증거 소스, 답변 형식 등 복합성의 다양한 측면을 평가합니다. 이 벤치마크에서는 표, 텍스트 등 다양한 형식의 데이터를 포함한 열린 도메인 설정에서의 검색을 평가합니다.

- **Performance Highlights**: 실험 결과, 긴밀한 상호작용 모델(late interaction models)과 고전적인 BM25와 같은 렉시컬 모델이 다른 사전 학습된 밀집 검색 모델보다 더 우수한 성능을 보였습니다. 이러한 결과는 복잡한 질문 처리에서 검색 성능을 향상시키기 위한 많은 연구가 필요함을 시사합니다. 또한, 검색 성능이 LLMs(Large Language Models)의 추론 능력에 미치는 영향도 분석되었습니다.



