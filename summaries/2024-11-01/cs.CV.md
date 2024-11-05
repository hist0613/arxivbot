New uploads on arXiv(cs.CL)

### Teaching Embodied Reinforcement Learning Agents: Informativeness and Diversity of Language Us (https://arxiv.org/abs/2410.24218)
Comments:
          EMNLP 2024 Main. Project website: this https URL

- **What's New**: 이 연구는 강화학습을 위한 체화 에이전트의 언어 입력의 다양성과 정보성의 영향을 처음으로 상세히 조사하였습니다.

- **Technical Details**: Decision Transformer (DT) 모델을 기반으로 하는 Language-Teachable Decision Transformer (LTDT) 아키텍처를 제안했습니다. 이 시스템은 언어 피드백을 통해 에이전트의 학습 능력에 큰 영향을 미치며, GPT-4를 활용해 더 자연스럽고 풍부한 언어 표현을 생성합니다.

- **Performance Highlights**: 이 연구 결과, 다양한 언어 피드백을 사용하여 훈련된 에이전트는 일반화가 향상되고 새로운 작업에 신속하게 적응하는 능력이 증가하였으며, 언어 없이 훈련된 에이전트보다 평균 20포인트 이상의 성능 향상을 기록했습니다.



### P-Masking: Power Law Masking Improves Multi-attribute Controlled Generation (https://arxiv.org/abs/2410.24201)
- **What's New**: LingGen이라는 새로운 접근법이 소개되었습니다. 이 방법은 다양한 언어적 속성에 대해 정밀한 제어를 가능하게 하며, 속성의 수가 달라져도 효과적으로 동작합니다.

- **Technical Details**: LingGen은 동적 P-MASKING(strategy) 전략을 채택하여, 훈련 과정에서 파워 법칙(power law distribution)에 따른 마스킹 비율(masking rates)을 샘플링합니다. 이를 통해 모델은 강력한 표현을 개발하고 다양한 속성 수에 따른 제어 능력을 적응시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, LingGen은 속성 제어 정확성과 텍스트 유창성(text fluency)에서 현업 최첨단 모델들을 초월합니다. 특히, 다양한 속성 요구사항이 있는 시나리오에서 뛰어난 성능을 보여주었습니다.



### Length-Induced Embedding Collapse in Transformer-based Models (https://arxiv.org/abs/2410.24200)
- **What's New**: 이 논문에서는 긴 텍스트에서의 텍스트 임베딩 성능 저하를 설명하는 Length Collapse라는 현상을 발견하였습니다. 이 현상은 긴 텍스트 임베딩이 좁은 공간으로 집합화됨으로써 발생하며, 이는 서로 다른 텍스트 길이 간의 분포 불일치를 초래하여 후속 작업의 성능에 악영향을 미칩니다.

- **Technical Details**: Length Collapse는 이론적으로 self-attention 메커니즘이 저역 통과 필터(low-pass filter)로 작용하며, 긴 시퀀스가 이 필터의 감쇠율을 증가시킨다는 것을 보여줍니다. TempScale은 softmax() 진단에서 온도를 도입하여 이러한 저역 필터의 감쇠율을 높여, 긴 텍스트 입력에 대해 일관된 성능을 강화합니다.

- **Performance Highlights**: Empirically, TempScale을 적용함으로써 기존 임베딩 모델의 성능을 개선할 수 있음을 입증하였고, Massive Text Embedding Benchmark(MTEB)에서 40개 데이터셋에서 최대 0.53% 성능 향상과 LongEmbed에서 4개의 데이터셋에서 최대 0.82% 성능 향상을 기록했습니다.



### Multi-Attribute Linguistic Tuning for Controlled Paraphrase Generation (https://arxiv.org/abs/2410.24199)
- **What's New**: 본 연구는 40개의 언어적 속성을 정확하게 제어하고 미세 조정할 수 있는 새로운 패러프레이즈 생성 접근 방식을 제시합니다.

- **Technical Details**: 제안된 모델은 인코더-디코더 아키텍처를 기반으로 하며, 원본 문장과 원하는 언어적 속성을 입력으로 받아 이 속성을 만족하는 패러프레이즈를 생성합니다. 생성 프로세스 중에 언어적 속성을 적응적으로 통합하고 품질 제어 메커니즘을 통해 생성 품질을 보장합니다. 모델은 40개의 속성을 통제하고, 이러한 속성은 밀집 표현 공간에서 나타내어집니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 모델에 비해 58% 높은 성능을 보였으며, 품질 제어 접근 방식으로 인해 추가로 9% 개선된 결과를 나타냈습니다. 이 연구는 데이터 증강에 적용하여 모델의 효과성을 높이는 방법도 제시하고 있습니다.



### SelfCodeAlign: Self-Alignment for Code Generation (https://arxiv.org/abs/2410.24198)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이번 연구에서는 SelfCodeAlign이라는 첫 번째 완전 투명하고 허용적인 코드 LLM의 자기 정렬 파이프라인을 제안합니다. 이는 방대한 인간 주석 또는 증류 없이 코드 LLM의 성능을 향상시킵니다.

- **Technical Details**: SelfCodeAlign은 높은 품질의 코드 조각에서 다양한 코딩 개념을 추출하여 새로운 작업을 생성합니다. 각 작업에 대해 여러 응답을 샘플링하고, 테스트 케이스와 쌍을 이루어 샌드박스 환경에서 검증한 후 성공적인 예제를 선택하여 instruction tuning을 수행합니다.

- **Performance Highlights**: SelfCodeAlign을 통해 생성된 데이터셋을 활용한 모델은 HumanEval+에서 67.1 pass@1을 기록하여 CodeLlama-70B-Instruct를 초과 달성했으며, 다양한 크기의 LLM에 걸쳐 효과적임을 확인했습니다.



### Hidden Persuaders: LLMs' Political Leaning and Their Influence on Voters (https://arxiv.org/abs/2410.24190)
Comments:
          EMNLP 2024 Main

- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)가 미국 민주주의에 미치는 잠재적 영향을 조사합니다. 여러 실험을 통해 LLM의 정치적 성향을 분석하고, 유권자에게 미치는 가능성을 탐구하였습니다.

- **Technical Details**: 대선 맥락에서 18개의 오픈 및 클로즈드 웨이트 LLMs를 사용한 투표 시뮬레이션을 통해 민주당 후보에 대한 선호도와 후보-정책 관련 질문에 대한 응답을 분석하여, instruction-tuned 모델이 기본 버전보다 더욱 뚜렷한 민주당 후보 성향을 보인다는 것을 밝혔습니다. 또한, 935명의 미국 등록 유권자를 통한 실험을 실시하여 LLM(Claude-3, Llama-3, GPT-4)과의 5회의 상호작용이 유권자의 선택에 미친 영향을 탐구했습니다.

- **Performance Highlights**: 실험 결과, LLM과의 상호작용 후 유권자의 선택이 민주당 후보로 전환되었으며, 투표 마진이 0.7%에서 4.6%로 확대되었습니다. 이는 정치 캠페인의 설득력 시험에서 나타난 효과보다도 더 큰 변화로, 많은 사용자들이 LLM과의 더 많은 정치적 상호작용을 원한다고 응답했습니다.



### Constraint Back-translation Improves Complex Instruction Following of Large Language Models (https://arxiv.org/abs/2410.24175)
Comments:
          14 pages, 6 figures

- **What's New**: 이번 연구에서는 기존 데이터셋에서 내재된 복잡한 제약 조건을 활용하여 고품질 복잡한 지시-응답 데이터셋(CRAB)을 생성하는 새로운 데이터 생성 기법인 constraint back-translation(제약 역번역)을 제안합니다.

- **Technical Details**: 기존의 고품질 지시-응답 쌍을 사용하여, Llama3-70B-Instruct 모델을 통해 응답에서 이미 충족된 제약 조건을 추출하고 이를 지시에 추가하여 복잡한 지시-응답 쌍을 생성합니다. 이 과정에서 비용을 절감하고 데이터 잡음을 줄이는 효과가 있습니다.

- **Performance Highlights**: CRAB 데이터셋으로 후속 훈련을 수행한 결과, 여러 개의 LLM(대형 언어 모델)의 복잡한 지시 따르기 능력이 크게 향상되었습니다. 다양한 평가 지표에서 이전 연구보다 개선된 성과를 보였으며, constraint back-translation 방법이 고품질 데이터를 생성하는 데 유용함을 입증했습니다.



### GPT or BERT: why not both? (https://arxiv.org/abs/2410.24159)
Comments:
          22 pages; submission to the BabyLM Challenge 2024

- **What's New**: 이번 논문에서는 마스크 언어 모델링(Masked Language Modeling)과 인과 언어 모델링(Causal Language Modeling)을 통합하는 간단한 방법을 제시합니다. 이를 통해 두 가지 모델링 패러다임의 장점을 결합한 GPT-BERT라는 하이브리드 모델을 개발했습니다. 이 모델은 BabyLM Challenge 2024에서 테스트한 결과, 독립적으로 훈련된 마스크 모델이나 인과 모델보다 성능이 우수했습니다.

- **Technical Details**: GPT-BERT 모델은 마스크된 다음 토큰 예측(Masked Next-Token Prediction, MNTP)에 기반하여 훈련됩니다. MLM과 CLM의 목표는 동일한 트랜스포머 아키텍처에서 공유되는 파라미터(Parameters)를 사용하여 통합되고, 각 배치에서 데이터 비율은 직접 결정됩니다. 시스템의 성능을 더욱 높이기 위해, 트랜스포머 아키텍처의 수정된 버전을 사용하고, 고급 기법을 적용했습니다.

- **Performance Highlights**: 하이브리드 프리트레이닝을 통해 다양한 벤치마크에서 GPT-BERT 모델이 우수한 성능을 발휘했습니다. 특히 리소스가 제한된 상황에서도 효과적인 결과를 냈으며, 저렴한 추가 비용으로 강력한 언어 모델을 개발할 수 있는 새로운 가능성을 제시했습니다.



### Thought Space Explorer: Navigating and Expanding Thought Space for Large Language Model Reasoning (https://arxiv.org/abs/2410.24155)
- **What's New**: 이번 연구에서는 Thought Space Explorer (TSE)라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLMs)의 사고 구조를 확장하고 최적화함으로써, 사고의 맹점을 탐색하도록 돕습니다. 기존 방법들이 친숙한 해결책에 국한되어 있는 것과 달리, TSE는 다양한 전략을 통해 새로운 추론 단계를 생성하고 사고 공간을 넓혀 비전문적 사고를 보완합니다.

- **Technical Details**: TSE는 세 가지 단계, 즉 키 노드 선택, 연결 및 확장, 협업 추론을 통해 이루어집니다. 첫 번째 단계에서는 모델의 추론 과정에서 각 노드의 기여도를 정량화하여 중요한 노드를 선택합니다. 두 번째 단계에서는 선택된 주요 노드를 시스템적으로 연결하고 새로운 분기로 확장하여 새로운 추론 방향을 탐색합니다. 마지막으로, 세 번째 단계에서는 서로 다른 방향의 다양한 추론 경로를 통합하고 조화롭게 결합하여 결론을 생성합니다.

- **Performance Highlights**: TSE는 세 가지 추론 과제에서 기존 방법들에 비해 명확한 성능 향상을 보였으며, 사고 구조의 각 구성 요소의 효율성을 분석하였습니다. 이를 통해 신중하고 확장적인 사고가 LLM의 추론 능력을 최대한 발휘할 수 있도록 기여할 수 있음을 확인하였습니다.



### Don't Touch My Diacritics (https://arxiv.org/abs/2410.24140)
Comments:
          6 pages

- **What's New**: 이 논문에서는 다국어 NLP에서 diacritics(발음 기호)의 처리 방식에 대한 문제점을 다룹니다. 대체로 공통적인 전처리 과정에서 발생하는 의사결정으로 인해 모델 성능에 부정적인 영향을 미친다는 점을 강조합니다.

- **Technical Details**: 여러 사례 연구를 통해 diacritized characters(발음 기호가 있는 문자)의 일관되지 않은 인코딩과 발음 기호 제거가 어떤 악영향을 미치는지를 증명합니다. 이를 통해 다국어 NLP 도구 및 모델 전반에서 간단하지만 필수적인 단계의 채택을 촉구합니다.

- **Performance Highlights**: 정확한 diacritics 처리를 통해 다국어 NLP에서의 공정성을 증가시킬 수 있는 가능성을 제시하고 있습니다.



### Multi-environment Topic Models (https://arxiv.org/abs/2410.24126)
- **What's New**: 이번 연구에서는 Multi-environment Topic Model (MTM)을 소개하였습니다. 이 모델은 전통적인 토픽 모델링에서의 문제점을 해결하기 위해, 다양한 환경에서 텍스트 데이터를 분석하고, 일반적인(global) 용어와 환경 별(specific) 용어를 분리합니다.

- **Technical Details**: MTM은 계층적 확률 모델로, 환경에 따른 주제 분포에 미치는 영향을 희소하게 가정합니다. 즉, 각 주제에 대해 대부분의 단어는 모든 환경에서 공유되며, 특정 환경에서만 사용되는 단어는 일부입니다. 이를 위해 자동 연관 결정 사전(ARD)과 변분 베이즈(auto-encoding variational Bayes) 방법을 사용하였습니다.

- **Performance Highlights**: MTM은 다양한 정치 콘텐츠를 대상으로 실험한 결과, 강력한 기준선과 비교하여 더 낮은 perplexity를 기록하였고, 다중 환경 데이터에서도 정확한 인과 효과를 발견할 수 있는 능력을 보였습니다.



### Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs (https://arxiv.org/abs/2410.24049)
- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)에서 아랍 그룹에 대한 편향을 조사하고, 이러한 편향이 서구 관점에서 어떻게 발생하는지 분석하였습니다. 특히 여성 권리, 테러리즘, 반유대주의와 같은 민감한 주제에서 편향을 평가하는 데 집중했습니다.

- **Technical Details**: 연구에서는 두 개의 데이터셋을 생성하고, 각각 아랍과 서구에 대한 편향 평가 및 모델 안전성을 테스트하기 위해 사용했습니다. 800개의 샘플로 구성된 두 개의 데이터셋은 여덟 개 카테고리로 나누어졌고, 6개의 LLM(GPT-4, GPT-4o, LlaMA 3.1, Mistral 7B, Claude 3.5 Sonnet)을 평가하여 아랍에 대해 79%의 부정적인 편향을 발견했습니다.

- **Performance Highlights**: LlaMA 3.1-405B가 가장 편향이 심한 모델로 나타났으며, GPT-4o는 가장 취약한 모델로 평가되었습니다. 모든 LLM은 반격 성공률이 87% 이상을 보였고, Claude 3.5 Sonnet이 가장 안전하였으나 여전히 일부 편향을 드러냈습니다.



### Joint Training for Selective Prediction (https://arxiv.org/abs/2410.24029)
- **What's New**: 이 연구에서는 선택적 예측(Selective Prediction, SP) 문제를 다루기 위해 새로운 공동 훈련 접근법인 Joint Training for Selective Prediction (JTSP)을 제안합니다. JTSP는 분류기 모듈과 위임 정책을 동시에 최적화함으로써 두 모듈의 성능을 향상시킵니다.

- **Technical Details**: JTSP는 분류기(CL)와 위임 정책(DP)을 함께 훈련하여 모델의 신뢰성을 높이고, 정확도와 위임 비율 간의 tradeoff를 관리합니다. 기본적으로, JTSP 과정에서는 강화 학습(reinforcement learning)에서 얻은 정책 그래디언트 방법을 손실 함수에 통합하여 CL과 DP의 행동을 조정합니다.

- **Performance Highlights**: JTSP는 4개의 STEM 질문 분류 작업에서 두 개의 강력한 기준선보다 월등한 SP 결과를 도출하며, 각각의 모듈(CL과 DP)의 정확도를 동시에 향상시킵니다.



### Detecting text level intellectual influence with knowledge graph embeddings (https://arxiv.org/abs/2410.24021)
- **What's New**: 이 논문은 지식 그래프(Knowledge Graph)와 그래프 신경망(Graph Neural Network) 모델을 이용하여 기사의 인용 관계를 예측하는 새로운 방법론을 제시합니다.

- **Technical Details**: 저자들은 오픈 소스 저널 기사로부터 데이터를 수집하고, Gemini LLM을 사용하여 지식 그래프 표현을 생성합니다. 기존 방법과 새로운 그래프 신경망 기반 임베딩 모델을 통해 샘플링된 기사 쌍 간의 인용 존재를 예측합니다.

- **Performance Highlights**: 연구 결과, 제공된 지식 그래프 임베딩 방법이 인용이 있는 기사 쌍과 없는 기사 쌍을 구별하는 데 있어 우수한 성능을 보임을 입증했습니다. 훈련 후 해당 모델은 효율적으로 작동하며 특정 연구자의 요구에 맞게 미세 조정이 가능합니다.



### Speech is More Than Words: Do Speech-to-Text Translation Systems Leverage Prosody? (https://arxiv.org/abs/2410.24019)
Comments:
          WMT 2024

- **What's New**: 이번 논문에서는 말의 억양(prosody)이 텍스트 번역에 미치는 영향을 분석하고, 이를 평가하기 위한 새로운 방법론과 벤치마크(ContraProST)를 소개합니다. 특히, 음성 신호에 직접 접근할 수 있는 end-to-end(E2E) 시스템이 억양 인식 번역에 적합하다는 점에 주목하고 있습니다.

- **Technical Details**: 논문에서는 여러 언어 쌍과 억양 현상을 포괄하는 double-contrastive evaluation 접근 방법을 통해 S2TT 시스템의 성능을 검토합니다. 이를 위해 대규모 언어 모델(LLMs)과 제어 가능한 텍스트-음성 변환(controllable TTS)을 활용하여 prosody-rich 데이터를 생성합니다. 또한, 각 테스트 데이터는 자동 생성 공정을 통해 제작되며, 이는 다양한 억양 카테고리를 포함합니다.

- **Performance Highlights**: 실험 결과, S2TT 모델은 내부적으로 억양을 반영하는 경향이 있지만, 이 지식이 번역에 명확하게 나타나지 않는 경우가 많습니다. E2E 시스템은 기존의 cascaded 시스템보다 ContraProST에서 더 나은 성과를 보였으며, 특정 cascaded 시스템은 제한적으로 억양 정보를 캡처했으나, 이는 전사(text)의 특성에 따라 달라진다고 보고되었습니다.



### Multilingual Pretraining Using a Large Corpus Machine-Translated from a Single Source Languag (https://arxiv.org/abs/2410.23956)
- **What's New**: 이 논문에서는 고품질 영어 데이터 세트를 다국어로 번역하여 다국어 대형 언어 모델(LLM)을 위한 조절 가능한 프리트레이닝 데이터 세트를 생성하는 새로운 방법론을 탐구합니다. 특히, FineWeb-Edu라는 영어 웹 데이터 세트를 프랑스어, 독일어, 스페인어로 번역하여 TransWeb-Edu라는 300B 토큰의 데이터 세트를 만들었습니다.

- **Technical Details**: 본 연구에서는 Mistral-7B-Instruct 번역 모델을 사용하여 FineWeb-Edu를 프랑스어, 독일어, 스페인어로 번역하였습니다. 이 과정을 통해 생성된 다국어 데이터 세트를 사용하여 CuatroLLM이라는 1.3B 매개변수의 모델을 처음부터 끝까지 프리트레이닝하였습니다. 이 모델은 서로 다른 다국어 추론 작업에서 Gemma2, Llama3.2 등 최신 모델들과 유사하거나 더 나은 성능을 보였습니다.

- **Performance Highlights**: CuatroLLM은 Llama3.2의 훈련에 사용된 토큰의 약 6%만으로도 최고 성능의 다국어 모델을 포함한 여러 모델과 비슷한 성능을 달성했습니다. 추가적인 도메인 특화 프리트레이닝을 통해 CuatroLLM은 다국어 추론 작업에서 최상의 성능을 초과하였습니다. 모든 데이터 세트와 모델은 공개 라이선스로 제공되어, 재현성을 높이기 위한 노력이 이루어졌습니다.



### Language Models can Self-Lengthen to Generate Long Texts (https://arxiv.org/abs/2410.23933)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 긴 문맥을 처리하는 능력이 크게 향상되었지만, 긴 일관된 출력을 생성하는 데는 여전히 한계가 있습니다. 본 논문에서는 LLMs의 내재적 지식과 기술만을 활용하는 Self-Lengthen이라는 혁신적인 반복 훈련 프레임워크를 소개합니다.

- **Technical Details**: Self-Lengthen 프레임워크는 Generator와 Extender라는 두 가지 역할로 구성됩니다. Generator는 초기 응답을 생성하고, Extender는 이를 분할하여 확장합니다. 이 과정을 통해 새로운 긴 응답이 만들어지고, 이는 Generator와 Extender를 반복적으로 훈련하는 데 사용됩니다.

- **Performance Highlights**: Self-Lengthen은 Qwen2 및 LLaMA3와 같은 최상위 오픈 소스 LLM에 적용했을 때 기존 방법들보다 긴 텍스트 생성에서 우수한 성능을 보였습니다. 실험 및 인간 평가를 통해 그 성능이 입증되었습니다.



### BitStack: Fine-Grained Size Control for Compressed Large Language Models in Variable Memory Environments (https://arxiv.org/abs/2410.23918)
- **What's New**: 이 논문에서는 BitStack이라는 새로운 비훈련(Training-free) 기반의 가중치 압축 방법을 제안합니다. BitStack은 메모리 사용량과 모델 성능 간의 메가바이트 수준의 균형을 가능하게 하여, 전통적인 압축 방법이 해결하지 못한 문제를 다루고 있습니다.

- **Technical Details**: BitStack은 가중치 행렬을 동적으로 조정하며, SVD(Singular Value Decomposition)를 사용하여 각 파라미터의 중요성을 고려하여 가중치 매트릭스와 잔여 블록을 반복적으로 분해합니다. 이 과정에서 약 1비트의 메모리를 생성하며, 중요도에 따라 잔여 블록을 정렬 및 저장하여 현재 메모리 상황에 맞게 로드할 수 있습니다.

- **Performance Highlights**: 광범위한 실험에서 BitStack은 다양한 작업에서 정밀한 크기 조절에도 불구하고, 특정한 압축 비율에서 전통적인 양자화 기준을 지속적으로 초과하는 성과를 나타냈습니다. 이는 BitStack이 가중치 분해 방법과 양자화 기반 방법 간의 성능 격차를 해소했다는 것을 의미합니다.



### Responsible Retrieval Augmented Generation for Climate Decision Making from Documents (https://arxiv.org/abs/2410.23902)
- **What's New**: 본 논문은 기후 관련 문서에서의 정보 접근성을 개선하기 위한 평가 프레임워크를 제안합니다. 이 프레임워크는 기후법 및 정책 문서에 대한 질문에 답변하는 프로토타입 도구의 retrieval 및 generation 품질을 평가하는 데 사용됩니다.

- **Technical Details**: 기후 법 및 정책 도메인에서 LLM(대형 언어 모델)의 신뢰성 및 정보를 기반으로 하는 Retrieval-Augmented Generation (RAG) 접근 방식을 평가합니다. RAG는 정보를 추출하여 모델의 불확실성을 줄이는 데 도움을 줍니다. 또한, 이 시스템은 사용자 입력, 정보 검색, 답변 합성 등 여러 단계를 포함하는 선형 파이프라인으로 구성됩니다.

- **Performance Highlights**: 도구는 사용자 경험(UX)을 개선하고, 고위험 도메인에 안전하게 AI 시스템을 배포하기 위한 통찰력을 제공합니다. 연구 결과는 RAG 접근 방식의 책임 있는 배치 방법과 신뢰 구축을 위한 사용자 경험 고려 사항을 강조합니다.



### Leveraging LLMs for MT in Crisis Scenarios: a blueprint for low-resource languages (https://arxiv.org/abs/2410.23890)
Comments:
          arXiv admin note: text overlap with arXiv:2403.02370, arXiv:2403.01580

- **What's New**: 본 연구는 위기 상황에서 저자원 언어에 대한 기계 번역 (Machine Translation, MT) 시스템을 강화하기 위해 대형 언어 모델 (Large Language Models, LLMs) 및 다국어 LLM (Multilingual LLMs, MLLMs)을 활용하는 새로운 접근 방식을 제시합니다. 이 논문은 코로나19 팬데믹을 모델로 하여 위기 대응을 위한 MT 시스템 개발 및 평가를 포함한 프로세스를 설명합니다.

- **Technical Details**: 연구에서는 LLM 및 MLLM을 조정 (fine-tuning)하고, 커뮤니티 주도의 말뭉치 (corpus) 개발 전략을 사용하여 저자원 언어 쌍을 위한 맞춤형 MT 시스템을 개발합니다. LLMs은 언어 이해 및 복잡한 응답 생성 능력 덕분에 인간의 의사소통과 생산성을 향상시킬 수 있습니다. MLLM의 파라미터 조정은 특정 작업의 성능을 향상시키기 위해 미리 훈련된 모델의 하이퍼파라미터를 조정하는 과정이며, 이는 훈련 데이터 및 최적화 기법에 따라 달라집니다.

- **Performance Highlights**: 조정된 MLLM 모델이 LLM 모델보다 우수한 성능을 제공하며, 코로나19와 같은 위기상황에서 신속하고 품질 높은 MT 시스템을 개발하는 확장 가능하고 복제 가능한 모델을 제시합니다. 연구에서는 맞춤형 GPT 시스템과 NLLB에 적응된 MLLM 모델을 비교하였고, 커뮤니티 참여가 매우 특화된 위기 특정 데이터셋의 생성에 중요한 역할을 한다고 강조합니다.



### 'No' Matters: Out-of-Distribution Detection in Multimodality Long Dialogu (https://arxiv.org/abs/2410.23883)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구에서는 대화(Dialogue)와 이미지(Image) 입력에서 발생하는 아웃 오브 디스트리뷰션(Out-of-Distribution, OOD) 감지 문제를 해결하기 위해 새로운 방법론인 대화 이미지 정렬 및 향상 프레임워크(Dialogue Image Aligning and Enhancing Framework, DIAEF)를 제안합니다.

- **Technical Details**: DIAEF 프레임워크는 이미지와 대화 모드에서 비정상적 쿼리를 효과적으로 감지하기 위해 두 가지 주요 시나리오에서 OOD를 탐지하는 점수 설계를 포함하고 있습니다: 1) 대화와 이미지 입력 쌍 간의 불일치, 2) 이미 확인되지 않은 라벨을 가진 입력 쌍. 이 프레임워크는 대화 시스템에서 긴 다중 대화 시나리오를 처리하는 데 필요한 혁신적인 접근을 제공합니다.

- **Performance Highlights**: 실험 결과, DIAEF를 적용한 다중 라운드 대화 시스템은 이전에 보지 못했던 라벨에 대해 OOD 탐지가 효과적임을 보여주며, 불일치 쌍에서도 강력한 견고성을 입증합니다. 이 연구는 향후 멀티모달 대화 시스템의 기초적인 기준과 방법론을 설정하고 사용자 경험을 향상시킵니다.



### Audio Is the Achilles' Heel: Red Teaming Audio Large Multimodal Models (https://arxiv.org/abs/2410.23861)
- **What's New**: 이 논문은 오디오 LMMs(대규모 다중 모달 모델)의 안전성을 체계적으로 평가한 최초의 연구로, 기존의 텍스트 안전성 평가와 비교하여 다중 모달 입력에 대한 새로운 안전 문제를 다룹니다.

- **Technical Details**: 오디오 LMMs의 경우 5개 모델(Qwen-Audio, Qwen2-Audio, SALMONN-7B, SALMONN-13B, Gemini-1.5-Pro)을 대상으로 한 실험을 통해, (i) 오디오와 텍스트 형식의 유해 질문, (ii) 유해 질문과 함께 비언어적 오디오 노이즈, (iii) 발화 전용 jailbreak 공격 등 세 가지 설정에서 안전성을 평가합니다.

- **Performance Highlights**: 실험 결과, 오픈 소스 오디오 LMMs는 유해 오디오 질문에 대해 평균 69.14%의 공격 성공률을 보였으며, Gemini-1.5-Pro의 경우 70.67%의 공격 성공률을 기록했습니다. 비언어적 오디오의 간섭으로 인해 안전성이 떨어지며, Qwen-Audio 및 Qwen2-Audio는 텍스트 질문의 안전성이 평균 45.15% 감소했습니다.



### Can Language Models Perform Robust Reasoning in Chain-of-thought Prompting with Noisy Rationales? (https://arxiv.org/abs/2410.23856)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 본 논문은 대형 언어 모델(LLMs)에서 잘 연구되지 않은 문제인 노이즈가 있는 합리적 사고 촉구(chain-of-thought prompting with noisy rationales)를 다룹니다. 이는 맥락 학습에 사용되는 예제 내의 무관하거나 부정확한 추론을 포함한 과제를 평가하기 위해 NoRa 데이터셋을 구축했습니다.

- **Technical Details**: NoRa 데이터셋은 263,912,639,126,391,263,91개의 질문으로 구성되며, 수학적, 상징적, 상식적 추론 작업을 포함합니다. 본 연구는 노이즈 비율을 통해 추론 난이도를 조절하며, 질문이나 답변을 수정하지 않고도 전체 촉구의 정확성을 보장합니다. CD-CoT(Contrastive Denoising with noisy CoT) 방법을 통해 노이즈가 있는 합리적 사고를 하나의 클린한 합리적 사고와 대비하여 LLM의 추론 능력을 향상시킵니다.

- **Performance Highlights**: CD-CoT 방법은 기본 모델에 비해 평균 17.8%의 정확도 향상을 나타내며, 노이즈가 있는 합리적 사고에 대한 정화 능력이 기존 방법들보다 현저히 강력함을 입증했습니다.



### The Automated Verification of Textual Claims (AVeriTeC) Shared Task (https://arxiv.org/abs/2410.23850)
- **What's New**: AVeriTeC(Automated Verification of Textual Claims) 공동 과제를 통해 실제 사실 검증자들에 의해 확인된 주장을 검증하는 데 초점을 두었습니다. 이 작업은 검색 엔진이나 주최 측 제공의 knowledge store를 통해 증거를 검색하고, 정확성을 예측하는 것을 요구합니다.

- **Technical Details**: AVeriTeC는 4,568개의 예시로 구성되어 있으며, 각 사례는 50개의 사실 검증 기관에서 수집되었습니다. 데이터셋은 질문 생성 및 답변 프로세스를 통해 증거 검색 과정을 구조화하여 제공하며, 모든 증거는 주장이 만들어지기 전에 웹에서 사용 가능해야 합니다.

- **Performance Highlights**: 총 21개의 제출물이 있었고, 그 중 18개가 기존 기준선을 초과했습니다. 찬사를 받은 팀 TUDA_MAI는 63%라는 AVeriTeC 점수를 달성하여 기존의 11%에서 크게 개선되었습니다. 더 나아가, 향후 연구를 위한 새로운 knowledge store가 공개되었습니다.



### Commonsense Knowledge Editing Based on Free-Text in LLMs (https://arxiv.org/abs/2410.23844)
Comments:
          11 pages, 8 figures

- **What's New**: 이 연구는 자유 텍스트 기반의 상식 지식 편집을 위한 새로운 방법(Knowledge Localization for Free-Text, KLFT)과 동적 인식 편집 방법(Dynamics-aware Editing Method, DEM)을 제안합니다. 기존 방법들은 제한적인 단일 토큰 또는 엔티티에 대해 편집을 수행했지만, 이 연구는 더 넓은 데이터 구조와 긴 내용을 가진 상식 지식의 편집 가능성을 탐구합니다.

- **Technical Details**: 논문에서 제안한 KLFT 방법은 상식 지식의 저장 장소를 추적하고 회수하는 두 가지 실험을 통해 상식 지식이 MLP 및 Attention 층에 분산되어 저장된다는 것을 보여줍니다. 또한, DEM 방법은 동적 인식 모듈을 사용하여 상식 지식의 저장 위치를 실시간으로 감지하고 특정 층에서 목표 지식을 편집합니다.

- **Performance Highlights**: 실험 결과, DEM 방법은 상식 지식 편집에서 우수한 성능을 달성하였으며, 새롭게 생성된 상식 지식 편집 벤치마크(Commonsense Knowledge Editing Benchmark, CKEBench)는 15,600개의 샘플을 포함하여 기존 데이터셋과 비교해 더 높은 난이도를 자랑합니다.



### GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages (https://arxiv.org/abs/2410.23825)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 GlotCC라는 새로운 문서 수준의 대규모 말뭉치를 소개합니다. 이 말뭉치는 CommonCrawl에서 추출되었으며, 1000개 이상의 소수 언어를 포함하여 2TB의 데이터량을 자랑합니다. GlotCC는 오픈소스 기반의 재현 가능한 파이프라인을 통해 생성되었으며, 신뢰할 수 있는 데이터 품질을 유지하기 위해 엄격한 청소 과정이 포함되어 있습니다.

- **Technical Details**: GlotCC는 언어 식별 모델 GlotLID v3.0을 사용하여 데이터를 분류하며, 2000개 이상의 언어 레이블을 커버합니다. Ungoliant 파이프라인을 통해 CommonCrawl에서 웹 텍스트를 효과적으로 추출하고, 다양한 노이즈 제거 기술을 활용하여 품질 높은 데이터를 확보합니다. GlotLID v3.0은 언어 식별 오류를 최소화하기 위해 작성된 스크립트를 인식하며, 기존의 FastText 기반 모델의 한계를 극복했습니다.

- **Performance Highlights**: GlotCC의 무작위 샘플링 653개 언어 하위 코퍼스에 대한 감사 결과, 데이터가 제대로 된 언어로 구성되어 있음을 나타냈으며, 매크로 평균 점수 0.93, 중앙값 점수 1.0을 기록했습니다. 이 연구는 소수 언어 지원을 위한 중요한 기여로 평가되며, 생성적 언어 모델의 사전 학습 및 다양한 언어 기술 개발에 활용될 수 있습니다.



### What is Wrong with Perplexity for Long-context Language Modeling? (https://arxiv.org/abs/2410.23771)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 긴 문맥 입력 처리의 중요성이 강조되고 있으며, 기존의 perplexity (PPL) 평가 방법의 한계를 극복하기 위한 새로운 접근 방식이 제안되었습니다.

- **Technical Details**: 이 논문에서는 새로운 메트릭인 LongPPL을 제안하여 모델의 긴 문맥 이해력을 평가할 때 중요한 토큰에 초점을 맞춥니다. LongPPL은 긴-짧은 문맥 대조 방식(long-short context contrastive method)을 활용하여 이러한 키 토큰을 식별합니다. 추가적으로, LongCE (Long-context Cross-Entropy) 손실을 도입하여 키 토큰의 중요도를 반영하는 재가중치 전략을 제공합니다.

- **Performance Highlights**: LongPPL은 다양한 긴 문맥 벤치마크에서 성능과 강한 상관관계를 보였으며(예: Pearson 상관계수 -0.96), 기존 PPL보다 예측 정확도에서 유의미하게 향상된 결과를 나타냈습니다. 이러한 접근 방식은 LLM의 긴 문맥 처리 능력을 정량적으로 평가하고 개선하는 데 효과적인 방법을 제공합니다.



### The Potential of LLMs in Medical Education: Generating Questions and Answers for Qualification Exams (https://arxiv.org/abs/2410.23769)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 의료 교육 분야에서 의료 자격 시험 문제와 답변을 생성할 수 있는 가능성을 탐구하며, 의료 분야에서의 LLMs 응용의 혁신을 보여줍니다.

- **Technical Details**: 연구진은 실제 세계의 노인 만성 질환에 대한 중국 데이터 세트를 사용하여 LLMs가 제한된 정보를 기반으로 개방형 질문과 답변을 생성할 수 있는지를 평가했습니다. 8개의 인기 있는 LLM(ERNIE 4, ChatGLM 4 등)이 사용되었습니다. LLMs의 생성된 질문은 전문 평가자에 의해 평가되었습니다.

- **Performance Highlights**: 대부분의 LLMs가 제공한 질문의 일관성과 정보 정확성에서 평균 점수가 4를 넘었고, 전문성 측면에서도 4에 가까운 점수를 받았습니다. 하지만 그럼에도 불구하고 질문 생성에 비해 답변의 평균 점수는 낮았으며, LLMs의 답변에서의 주요 정보 부족이 발견되었습니다.



### DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios (https://arxiv.org/abs/2410.23746)
Comments:
          Accepted to NeurIPS 2024 Dataset & Benchmarking Track

- **What's New**: 이 연구에서는 새로운 벤치마크인 DetectRL을 소개하며, 최신 LLM(대형 언어 모델) 생성 텍스트 탐지 기술이 실질적인 시나리오에서 여전히 미비함을 강조했습니다.

- **Technical Details**: DetectRL은 다양한 실제 응용 프로그램에서 LLM이 사용될 때의 문제들을 시뮬레이션하여 텍스트를 생성합니다. 주요 공격 방법으로는 프롬프트 공격, 패라프레이즈 공격, 섭동 공격이 포함되며, 데이터 믹싱을 통한 다변량 샘플링이 사용됩니다. 다양한 도메인에서 생성된 데이터로 실험을 진행하였습니다.

- **Performance Highlights**: 현재의 탐지기들은 공격적인 섭동에 의해 성능이 평균 34.48% AUROC만큼 감소하였고, 반면에 슈퍼바이즈 탐지기는 더욱 강력한 탐지 능력을 보여주었습니다. DetectRL은 실생활 활용 상황에서 탐지기의 효과성을 평가하는 데 흔히 고려되지 않았던 요소들을 분석하여, 탐지기의 발전 방향을 제시할 수 있는 잠재력을 갖추고 있습니다.



### What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspectiv (https://arxiv.org/abs/2410.23743)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 다양한 레이어에서의 훈련 패턴을 조사하고, 빠른 사고(fast thinking)와 느린 사고(slow thinking)가 레이어별 기울기(gradients)에 미치는 영향을 분석했습니다. 특히, 체인 오브 생각 (CoT, Chain of Thought) 방법이 훈련의 안정성에 미치는 영향을 보았습니다.

- **Technical Details**: 이 연구는 5개의 기본 LLM과 5개의 지시 조정(instruction-tuned) LLM의 레이어별 기울기를 비교하며, 특정 작업(수학, 일반 기초 reasoning, 지식 추출)에 대한 여러 데이터셋을 사용했습니다. 빠른 사고와 느린 사고를 기준으로 기울기의 핵심 특성을 Singular Value Decomposition (SVD)을 통해 측정하였으며, 각 레이어의 기울기 변화를 분석했습니다.

- **Performance Highlights**: 느린 사고(Detailed CoT)를 통해 특정 정답을 식별하는 기울기 패턴을 보여주었고, 빠른 사고는 기울기가 초기 레이어에서 더 크게 나타났습니다. 지시 조정된 LLM은 사전 훈련된 LLM보다 잘못된 사고 경로를 식별하는 데 우수하지 않았습니다. 이러한 결과는 대형 언어 모델 훈련의 효율성과 안정성에 대한 새로운 통찰을 제공하였습니다.



### GigaCheck: Detecting LLM-generated Conten (https://arxiv.org/abs/2410.23728)
Comments:
          11 pages, 1 figure

- **What's New**: 본 연구에서는 LLM(대규모 언어 모델) 기반의 텍스트 탐지 기술을 다루고 있으며, GigaCheck이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 인간 작성 텍스트와 LLM 생성 텍스트를 구별하는 방법과, Human-Machine 협업 텍스트에서 LLM 생성 구간을 탐지하는 방법을 포함하고 있습니다.

- **Technical Details**: GigaCheck는 일반 목적 LLM을 활용하여 LLM 생성 텍스트 탐지 작업에 대해 효율적으로 파인튜닝(fine-tuning)합니다. 두 가지 주요 접근법을 통해, 첫 번째로 인간과 LLM의 텍스트를 구분하고, 두 번째로 DETR(deep learning object detection) 모델을 사용하여 텍스트 내 LLM 생성 구간을 로컬라이즈(localize)합니다. 이 과정에서 텍스트의 모든 내용을 분석하여 문자 단위의 구간을 출력함으로써 보다 보편적인 탐지 방식을 제공합니다.

- **Performance Highlights**: GigaCheck는 다섯 개의 분류 데이터셋과 세 개의 Human-Machine 협업 텍스트 분석 데이터셋에서 이전 방법들보다 우수한 성능을 보여줍니다. 특히, out-of-distribution 설정에서도 강력한 기준선을 설정하며, LLM 생성 구간 탐지 작업에서 높은 성능을 발휘합니다.



### Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction (https://arxiv.org/abs/2410.23692)
- **What's New**: Llama-3-8B-Mob은 장기적인 도시 이동 예측을 위해 특별히 조정된 대규모 언어 모델로, Q&A 형식으로 개인의 이동 경로를 예측합니다. 이 모델은 일본의 네 개 대도시에서 수집된 대규모 데이터셋을 통해 검증되었습니다.

- **Technical Details**: 모델은 사용자의 과거 이동 경로와 시간 정보를 바탕으로 예측을 수행하며, 측정된 결과는 다양한 예측 지표에서 기존의 최고 성능(State-of-the-Art, SOTA)을 초월했습니다.

- **Performance Highlights**: Llama-3-8B-Mob은 이동 예측 대회에서 35개 모델 중 2위, 3위에 각각 자리잡으면서 평균적으로 1위를 기록하며, 제한된 데이터로도 뛰어난 성과를 보여주었습니다.



### Improbable Bigrams Expose Vulnerabilities of Incomplete Tokens in Byte-Level Tokenizers (https://arxiv.org/abs/2410.23684)
- **What's New**: 이번 연구에서는 byte-level byte pair encoding (BPE) 토크나이저에서 발생하는 불완전 토큰(incomplete tokens)의 취약성을 조사하였습니다. 특히, 인접 토큰에 크게 의존하고 낯선 토큰과 함께 사용할 때 약해지는 이들 불완전 토큰의 문제를 다루고 있습니다.

- **Technical Details**: 연구는 불완전 토큰으로 이루어진 improbable bigrams(불가능한 바이그램) 조합을 제시하며, 이러한 조합은 모델의 할루시네이션(hallucination)을 유발하는 것으로 나타났습니다. 실험을 통해 Llama-3.1, Qwen2.5, Mistral-Nemo와 같은 여러 LLM(대형 언어 모델) 패밀리에서 불완전 토큰 조합이 할루시네이션을 생성할 확률이 더 높다는 결과를 확인했습니다.

- **Performance Highlights**: 모델의 문구 처리 능력을 평가한 결과, improbable bigrams는 높게 나타났지만, 다른 완전한 토큰으로 이루어진 바이그램들은 할루시네이션이 거의 발생하지 않았습니다. Llama-3.1에서는 43%의 할루시네이션 비율이 발견되었고, Qwen2.5에서는 38%를 기록했습니다.



### Pseudo-Conversation Injection for LLM Goal Hijacking (https://arxiv.org/abs/2410.23678)
- **What's New**: 이번 논문에서는 LLMs(Large Language Models)에 대한 새로운 공격 기법인 Pseudo-Conversation Injection을 제안합니다. 이 방법은 대화 맥락에서 역할 식별의 취약점을 활용하여, 사용자의 프롬프트에 악의적인 접미사를 추가해 모델이 원래 입력을 무시하고 목표 응답을 생성하게 만듭니다.

- **Technical Details**: Pseudo-Conversation Injection은 사용자의 초기 프롬프트에 대해 LLM이 생성한 응답을 조작하여, 새로운 악의적인 작업 프롬프트를 모델에 입력합니다. 이를 통해 LLM은 대화를 완료한 것으로 인식하고, 조작된 프롬프트에 응답하게 됩니다. 이 기법의 세 가지 전략인 Targeted Pseudo-Conversation, Universal Pseudo-Conversation, Robust Pseudo-Conversation를 제안하며 각각의 사용 시나리오에서의 효과성을 연구합니다.

- **Performance Highlights**: 실험 결과, Pseudo-Conversation Injection 방법은 ChatGPT와 Qwen 플랫폼에서 기존 방법들보다 공격 효과성이 크게 개선된 것으로 나타났습니다. 연구를 통해 LLMs가 특정 대화 패턴을 처리하는 데 있어 취약성을 드러냈으며, 이는 공공 안전에 심각한 위협이 될 수 있음을 강조합니다.



### Kernel Looping: Eliminating Synchronization Boundaries for Peak Inference Performanc (https://arxiv.org/abs/2410.23668)
- **What's New**: 이 논문에서는 최신 데이터 흐름 아키텍처의 레이어-레벨 퓨전을 활용하여, 언어 모델의 반복 레이어 구조와 결합한 새로운 최적화 기법인 'kernel looping'을 소개합니다.

- **Technical Details**: Kernel looping은 같은 커널에 대한 연속 호출 간의 동기화 비용을 제거하고, 파이프라인 외부 루프를 포함하는 수정된 커널에 대한 단일 호출로 변환함으로써 성능을 개선합니다. 이 최적화 기술은 SambaNova SN40L RDU에서 평가되었습니다.

- **Performance Highlights**: Kernel looping을 활용한 실험 결과, 다양한 오픈 소스 모델의 디코드 단계에서 최대 2.2배 속도 향상을 달성하였고, 8 및 16 소켓에서 90% 이상의 피크 성능을 이끌어냈습니다. 또한, DGX H100에 비해 최대 3.7배의 속도 향상을 기록했습니다.



### Morphological Typology in BPE Subword Productivity and Language Modeling (https://arxiv.org/abs/2410.23656)
Comments:
          15 pages, 6 figures

- **What's New**: 본 연구는 형태적 유형(morphological typology)이 토큰화(tokenization) 및 언어 모델링(performance modeling) 성능에 미치는 영향을 조사합니다. 특히 합성 언어와 분석 언어에서 BPE(byte-pair encoding) 알고리즘을 통한 토큰화의 생산성을 비교합니다. 연구 결과에 따르면, 합성 형태가 더 높은 서브워드 정규성과 생산성을 보이며, 언어 모델링 과제에서 더 우수한 성과를 나타냅니다.

- **Technical Details**: 연구에서는 단어 형성과 구조에 대한 형태학(morphology)을 분석하고, BPE 토큰화의 효율성이 특정 형태 유형에 따라 달라지는지를 평가합니다. 형태적 형성 요소인 형태소(morpheme)가 결합하여 어떻게 단어를 형성하는지 살펴보며, BPE를 적용한 실험에서는 합성 형태의 언어가 더 빠른 일반화(generalization)를 이루고 더 낮은 혼란도(perplexity) 및 손실(loss)을 보여줍니다.

- **Performance Highlights**: 합성 형태의 언어는 더 생산적인 서브워드 시스템을 가지며, 훈련 데이터가 같은 양일 때에도 언어 모델링 과제에서 상대적으로 우수한 성능을 나타냅니다. 또한, 형태적 복잡성(linguistic complexity)의 연속성이 실험 결과에서도 확인되며, 형태적 유형이 BPE 토큰화의 효율성과 밀접한 관계가 있음을 시사합니다.



### On Positional Bias of Faithfulness for Long-form Summarization (https://arxiv.org/abs/2410.23609)
Comments:
          18 pages

- **What's New**: 본 논문은 Large Language Models (LLMs)가 긴 컨텍스트 환경에서 발생하는 positional bias를 연구하였습니다. 특히, 긴 형식의 요약(long-form summarization)에서 이러한 편향이 어떻게 나타나는지, 이는 정확성(faithfulness)에 미치는 영향을 분석하고 이를 완화하기 위한 다양한 기법을 제시합니다.

- **Technical Details**: 연구팀은 8개의 인간 주석이 달린 긴 형식 요약 데이터셋을 포함한 벤치마크를 구성하여 정확성을 일관되게 평가하였습니다. LLM 기반의 정확성 메트릭은 전체 문서 컨텍스트에서 효과적이지만, 문서 순서에 민감성이 나타나는 positional bias를 갖고 있습니다. 요약 생성과 관련하여, LLM은 문서의 시작과 끝을 충실히 반영하지만 중간 내용을 간과하는 'U자형(U-shaped)' 경향을 보였습니다.

- **Performance Highlights**: LLMs의 배치 순서에 대한 감도 실험 결과, 중요한 문서가 입력 중간에 위치할 경우 정확성이 저하되는 것을 확인했습니다. 또한, 특정한 길이 이후로는 모델이 문서의 끝에 더 집중하면서 정확성이 향상되는 경향이 있음을 발견했습니다. 마지막으로, prompt 기법이 특정 위치에 대한 모델의 주의를 효과적으로 향상시키는 것으로 나타났습니다.



### Dynamic Uncertainty Ranking: Enhancing In-Context Learning for Long-Tail Knowledge in LLMs (https://arxiv.org/abs/2410.23605)
- **What's New**: 본 논문에서는 long-tail knowledge를 효과적으로 포착하기 위해 reinforcement learning 기반의 dynamic uncertainty ranking 방법을 제안합니다. 이 방법은 LLM의 예측 결과에 미치는 각 retrieved sample의 영향을 고려하고, 오히려 misleading sample은 하락시키는 방식으로 작동합니다.

- **Technical Details**: 제안된 방법은 BERT 기반의 retriever 아키텍처를 활용하여, linear layer를 추가하고, BM25를 pre-selection에 사용합니다. 이후 policy gradients 방법을 사용하여 retriever를 훈련시키고, learnable dynamic ranking threshold를 도입하여 query 비용을 줄입니다.

- **Performance Highlights**: 실험 결과에서, 본 방법은 다양한 질문-답변 데이터셋에서 기존의 최선의 방법보다 2.76% 향상된 성능을 보였으며, long-tail 질문에서 5.96%의 유의미한 정확도를 증가시켰습니다.



### BioNCERE: Non-Contrastive Enhancement For Relation Extraction In Biomedical Texts (https://arxiv.org/abs/2410.23583)
Comments:
          4 figures, 2 tables, 10 pages

- **What's New**: 이 논문에서는 생물의료 분야에서 관계 추출(Relation Extraction, RE)을 위해 생물학적 비대조 관계 추출(BioNCERE)이라는 새로운 훈련 방법을 소개합니다. 이 방법은 명명된 엔티티 레이블을 사용하지 않고도 관계를 예측하도록 설계되어 주석 비용을 줄이는데 기여합니다.

- **Technical Details**: BioNCERE는 전이 학습(Transfer Learning)과 비대조 학습(Non-Contrastive Learning)을 사용하여 차원 축소 및 과적합(Overfitting)을 피합니다. 모델은 세 단계로 RE를 해결하며, 이전 단계에서 학습한 가중치를 동결하고 두 번째 단계에서 비대조 학습을 활용하여 관계를 예측합니다. SemMedDB 데이터셋에서 실험을 수행하여, 명명된 엔티티 정보 없이도 거의 최신 성능을 발휘함을 보였습니다.

- **Performance Highlights**: 이 모델은 기존의 BioPrep 모델과 비교하여 관계 추출에서 F1 점수(Augmented F1 score)가 거의 유사한 성능을 보여 주며, 모델의 접근 방식이 보다 간소화된 점을 강조합니다. BioPrep은 명명된 엔티티 및 그룹핑을 활용하는 반면, BioNCERE는 직접적으로 문제에 접근합니다.



### From Context to Action: Analysis of the Impact of State Representation and Context on the Generalization of Multi-Turn Web Navigation Agents (https://arxiv.org/abs/2410.23555)
Comments:
          10 pages, 3 figures, 5 tables

- **What's New**: 최근 대형 언어 모델(LLM) 기반 프레임워크의 발전에 따라, 복잡한 실세계 응용 프로그램에서의 능력이 확대되었습니다. 특히 사용자의 명령에 의해 웹 브라우저에서 작업을 수행하면서 상호작용적인 웹 내비게이션을 가능하게 하고 있습니다.

- **Technical Details**: 이 연구에서는 웹 내비게이션 에이전트의 성능에 영향을 미치는 다양한 맥락적 요소를 분석하여 서로 다른 상호작용 기록과 웹 페이지 표현의 영향을 최적화하는 방법을 탐구합니다. 이를 통해 Agnet의 성능 향상을 도모하고, 이는 특히 보지 못한 웹사이트, 카테고리 및 지리적 위치에서 효과적인 맥락 관리가 가능함을 보여줍니다.

- **Performance Highlights**: 연구 결과, 효과적인 맥락 관리가 이루어짐으로써 에이전트의 성능이 개선되었으며, 이는 실세계 응용 프로그램에서 더 정확하고 효율적인 웹 내비게이션을 가능하게 합니다.



### Simulating User Agents for Embodied Conversational-AI (https://arxiv.org/abs/2410.23535)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 이 연구에서는 사용자 행동을 모사하는 LLM 기반의 사용자 대리인(user agent)을 제안하여, 로봇과의 상호작용을 효율적으로 생성할 수 있는 방법을 탐구합니다. 이를 통해 데이터 수집의 비용 및 노동력을 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: LLM(대규모 언어 모델)을 활용한 사용자 대리인은 특정 사용자 목표(예: 아침 식사 만들기)에 따라 로봇의 행동을 관찰하고, 그에 따라 적절한 대화를 생성할 수 있습니다. 연구에서는 zero-shot 및 few-shot prompting 기법을 통해 대화의 정확성을 평가하였으며, TEACh 데이터셋을 활용하여 성능을 비교하였습니다.

- **Performance Highlights**: LLM 기반의 사용자 대리인은 zero-shot prompting에서 42%의 F-measure를, few-shot prompting에서는 43.4%를 기록하여 인간의 대화 행동을 모방하는 데 성공했습니다. 또한, fine-tuning을 통해 사용자 대리인의 발화 결정 능력이 향상되어, 발화할 시점의 안정성은 유지되고 발화 내용의 정확성이 51.1%에서 62.5%로 개선되었습니다.



### Large Language Models for Patient Comments Multi-Label Classification (https://arxiv.org/abs/2410.23528)
- **What's New**: 본 연구는 대규모 언어 모델(Large Language Models, LLMs)인 GPT-4o-Turbo를 활용하여 병원 환자 피드백의 다중 라벨 텍스트 분류(Multi-label Text Classification, MLTC)를 수행하는 방법을 탐구합니다. 기존의 감독 학습 방식에서 오는 도전 과제들을 해결하기 위해, 데이터 제공 전에 보호된 건강 정보(Protected Health Information, PHI) 감지 프레임워크를 도입하여 환자 정보를 비식별화(Bit-order)합니다.

- **Technical Details**: 연구에서는 Zero-shot learning, in-context learning 및 chain-of-thought prompting을 포함한 다양한 프로토타입 엔지니어링 접근 방식을 실험하였습니다. MLTC는 환자 경험의 다면성을 반영하기 위해 설계되었으며, 그 출력은 환자의 인구 통계정보, 병원체류 정보, 선택 질문에 대한 답변 및 평가와의 연관성을 분석하여 신뢰성을 높입니다.

- **Performance Highlights**: GPT-4o-Turbo는 Zero-shot 또는 few-shot 설정에서 전통적인 방법과 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)에 비해 뛰어난 성능을 보였으며, F1-score 76.12%와 weighted F1-score 73.61%로 최고 성능을 달성했습니다. 이러한 결과는 환자 경험 향상 및 의료 서비스 제공의 효율성을 높이는 데 기여할 것입니다.



### LEAF: Learning and Evaluation Augmented by Fact-Checking to Improve Factualness in Large Language Models (https://arxiv.org/abs/2410.23526)
Comments:
          22 pages, 9 figures

- **What's New**: 이번 연구에서는 의료 분야를 중심으로 LLMs(대형 언어 모델)의 사실 정확성을 향상시키기 위한 새로운 접근법인 LEAF(Learning and Evaluation Augmented by Fact-Checking)를 제안합니다. LEAF는 두 가지 전략인 Fact-Check-Then-RAG와 Self-Training을 활용하여 모델의 출력 신뢰성을 증가시킵니다.

- **Technical Details**: LEAF의 첫 번째 전략인 Fact-Check-Then-RAG는 사실-checking 결과를 활용하여 Retrieval-Augmented Generation (RAG)을 개선합니다. 두 번째 전략은 Self-Training을 통해 supervised fine-tuning (SFT)이나 Simple Preference Optimization (SimPO) 기법을 사용하여 LLM 매개변수를 업데이트합니다.

- **Performance Highlights**: LEAF를 통해 사실-checking된 응답을 통합함으로써 LLM의 성능이 크게 향상되었음을 보여주며, 특히 정보 정확성이 중요한 의료 분야에서 신뢰성과 사실성을 높이는 효과를 나타냅니다.



### Neural spell-checker: Beyond words with synthetic data generation (https://arxiv.org/abs/2410.23514)
Comments:
          Camera-ready version. Accepted to TSD 2024

- **What's New**: 본 논문에서는 슬로베니아어를 위한 두 가지 새로운 스펠 체커(SloSpell, SloNSpell)를 소개하고 비교합니다. 이는 전통적인 스펠 체커에 비해 단어의 맥락 적합성도 평가할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: 첫 번째 스펠 체커인 SloSpell은 기존 스펠 체커에 비해 훨씬 더 큰 단어 목록을 기반으로 한 전통적인 형태소 사전 기반 접근 방식입니다. 두 번째 SloNSpell은 대규모 말뭉치에서 데이터 손상을 동적으로 학습한 신경망(nural network) 기반 스펠 체커로, BERT 모델을 기반으로 합니다. 이들은 합성 및 실제 데이터 세트를 통해 성능을 비교하며, 특히 SloNSpell이 정확도(precision)와 재현율(recall) 측면에서 기존의 모든 스펠 체커를 초과하는 성능을 보였습니다.

- **Performance Highlights**: SloNSpell은 특히 슬로베니아어 스펠링 오류 탐지에서 우수한 성능을 나타내며, 젊은 학습자와 전문 성인 작가의 텍스트를 샘플로 한 평가에서 강력한 기준선보다 높은 성과를 기록했습니다. 또한, 사용자가 쉽게 접근할 수 있도록 온라인 인터페이스로 제공되고 있습니다.



### Dynamic Strategy Planning for Efficient Question Answering with Large Language Models (https://arxiv.org/abs/2410.23511)
Comments:
          Under review at ACL Rolling Review

- **What's New**: 이번 연구에서는 DyPlan이라는 새로운 기술을 제안하여 Large Language Models (LLMs)의 질문-응답 성능을 개선하고, 질문 유형에 따라 동적으로 전략을 선택할 수 있는 프로세스를 도입합니다. 이를 통해 더 효율적인 출력 토큰 생성을 가능하게 하고, 비용을 절감합니다.

- **Technical Details**: DyPlan은 입력 질문에 따라 가장 적합한 전략을 선택하는 초기 결정 단계를 도입하고, LLM의 응답 생성을 이 전략에 맞춰 안내합니다. DyPlan-verify로 확장하여 내부 검증 및 수정 과정을 추가하여 생성된 답변의 품질을 더욱 향상시킵니다. 세 가지 주요 QA 데이터 세트에서 실험을 수행하여 DyPlan과 DyPlan-verify의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, DyPlan은 최고의 기준 모델에 비해 평균 7-13%의 성능 향상과 11-32%의 비용 절감을 달성했으며, DyPlan-verify를 통해 평균 12-13%의 성능 개선과 11-19%의 비용 감소를 확인했습니다.



### Efficient and Interpretable Grammatical Error Correction with Mixture of Experts (https://arxiv.org/abs/2410.23507)
Comments:
          Findings of EMNLP 2024

- **What's New**: 본 논문에서는 다양한 오류 유형을 전문으로 수정하는 여러 개의 서브 네트워크를 가지는 혼합 전문가 모델(Mixture-of-Experts, MoECE)을 제안합니다. 이 모델은 T5-XL의 성능을 3배 더 적은 효과적인 매개변수로 달성하며, 추론 과정에서 오류 유형을 동시 식별함으로써 수정 사항을 보다 해석 가능하도록 만들어 줍니다.

- **Technical Details**: 혼합 전문가 모델(MoE)은 여러 개별 네트워크로 분해될 수 있는 학습 절차로, 각각의 네트워크는 특정 작업에서 전문가 역할을 수행합니다. MoE는 Transformer 아키텍처에 적용되며, 각 Transformer 블록의 피드포워드 레이어를 MoE 레이어로 대체하는 방식으로 구현됩니다. 이때 전문가 선택을 위한 라우터가 사용되며, 자원 발란싱 손실(load balancing loss)을 통해 전문가 간의 균형 있는 할당을 장려합니다.

- **Performance Highlights**: 제안된 MoECE 모델은 기존 GEC 모델과 비교했을 때 성능은 향상되면서도 계산 비용은 절감됩니다. 또한, 하나의 모델에서 여러 오류 유형을 처리할 수 있게 하여 시스템 조합의 필요성을 없애줍니다.



### Smaller Large Language Models Can Do Moral Self-Correction (https://arxiv.org/abs/2410.23496)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 도덕적 자기 수정(moral self-correction) 능력을 실험적으로 검증하며, 특히 약 3.8B의 매개변수를 가진 LLM이 적절한 안전 정렬(safety alignment) 세부 조정으로 매우 좋은 도덕적 자기 수정 성능을 달성할 수 있음을 강조합니다.

- **Technical Details**: 이 연구는 355M에서 70B 매개변수까지 다양한 LLM을 포함하며, 특히 적절한 안전 정렬 기법을 통해 모델의 비도덕적 출력을 수정하는 방법론을 검토합니다. 연구자는 LLM의 사회적 기준 이해 및 명령 수행 능력을 평가하기 위해 구체성(specificity), 부정(negation), 그리고 체계적(CoT) 설명 방식을 사용합니다.

- **Performance Highlights**: 실험 결과, 3.8B 이상의 매개변수를 가진 LLM이 도덕적 자기 수정 능력을 보였으나, 모든 모델이 비도덕적인 지시를 인식하고 거부하는 능력은 부족하여 기본 설정에 비해 비도덕적인 결정을 더 많이 내렸습니다.



### Collage: Decomposable Rapid Prototyping for Information Extraction on Scientific PDFs (https://arxiv.org/abs/2410.23478)
- **What's New**: Collage는 과학 PDF 문서에서 정보 추출 모델의 신속한 프로토타이핑, 시각화 및 평가를 위해 설계된 도구로, HuggingFace의 token classifier 또는 여러 LLM(Large Language Model)과 함께 사용할 수 있습니다.

- **Technical Details**: Collage는 PDF 문서의 내용을 처리하는 데 사용되는 세 가지 인터페이스(토큰 분류 모델, 텍스트 생성 모델 및 이미지/텍스트 다중 모달 모델)로 구성된 시스템입니다. 이 시스템은 PaperMage 라이브러리를 기반으로 하여 PDF를 구문 분석하고 처리를 위한 API를 제공합니다.

- **Performance Highlights**: Collage를 사용하면 사용자들이 대량의 과학 문헌에서 자료를 효과적으로 검색하고 평가할 수 있도록 지원하며, 다양한 모델의 결과를 시각화할 수 있어 디버깅 과정이 용이합니다. 또한, 공간 손실 문제를 해결하기 위해 여러 다중 모달 처리를 지원합니다.



### MDCure: A Scalable Pipeline for Multi-Document Instruction-Following (https://arxiv.org/abs/2410.23463)
- **What's New**: MDCure는 다중 문서(Multi-document) 처리의 효율성을 증대시키기 위해 설계된 스케일러블한 fine-tuning 파이프라인입니다. 기존의 pre-training에 의존하지 않고, 관련된 기사를 활용하여 고품질의 합성 MD instruction 데이터를 생성합니다.

- **Technical Details**: MDCure는 두 단계로 나뉘어 있습니다: Generation과 Filtering입니다. Generation 단계에서는 제로샷 프롬프트 템플릿을 이용해 관련 문서 집합으로부터 복잡한 cross-text 지침을 생성하며, Filtering 단계에서는 MDCureRM이라는 멀티-목적 보상 모델이 생성된 지침을 평가합니다. 이 과정에서 합성 지침 데이터의 품질을 보장합니다.

- **Performance Highlights**: MDCure는 FlanT5, Qwen2, LLAMA3.1 등의 다양한 LLM을 최대 70B 파라미터로 fine-tuning하였으며, 전반적인 다중 문서 및 긴 컨텍스트 벤치마크에서 성능을 최대 75.5% 향상시키는 결과를 보여주었습니다.



### Graph-Augmented Relation Extraction Model with LLMs-Generated Support Documen (https://arxiv.org/abs/2410.23452)
- **What's New**: 이 연구에서는 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)를 통합하여 문장 수준의 관계 추출 (Relation Extraction, RE)을 향상시키는 새로운 접근 방식을 제시합니다. 이 방법은 LLMs의 힘을 활용하여 보조 정보를 생성하고, 이를 통해 텍스트 데이터의 복잡한 그래프 표현을 생성합니다.

- **Technical Details**: 새로운 모델은 GNN을 통해 각 엔티티와 관련된 임베딩(embedding)을 정제 및 강화하여, 문장 간의 복잡한 관계를 효과적으로 포착할 수 있도록 설계되었습니다. 실험은 CrossRE 데이터셋을 기반으로 진행되었으며, 다양한 도메인에서 성능의 유의미한 향상을 보여주었습니다. 또한, LLM을 사용하여 보조 데이터셋을 생성하고, GNN 모듈을 통해 임베딩을 통합하여 성능 평가 체계를 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 문장 간의 복잡한 관계를 포착하는 능력에서 두드러진 성과를 나타내어, 기존의 문장 수준 RE 모델들이 가지던 한계를 극복하는 데에 기여할 것으로 보입니다. 상대적으로 다양한 도메인에서 경우에 따라 더 나은 Macro F1-Score를 기록하여, 관계 추출 분야에서의 GNN과 LLM의 조합 가능한 잠재력을 강조합니다.



### Social Science Meets LLMs: How Reliable Are Large Language Models in Social Simulations? (https://arxiv.org/abs/2410.23426)
- **What's New**: 이 논문은 LLM(대규모 언어 모델)의 신뢰도를 평가하는 새로운 데이터셋인 TrustSim을 소개하며, 10개의 계산 사회과학(CCS) 관련 주제를 다룹니다. 신뢰성 문제는 아직 충분히 탐구되지 않았으며, LLM 기반 시뮬레이션의 신뢰도를 체계적으로 조사합니다.

- **Technical Details**: TrustSim 데이터셋은 시나리오, 시스템 프롬프트, 자가 보고 질문, 개방형 질문, 평가 특성, 설명, 차원 등 6가지 구성 요소로 이루어져 있습니다. 740740740740개의 평가 인스턴스가 설정되어 있으며, 이는 개별 시뮬레이션의 다양한 측면을 평가합니다. 실험은 14개의 LLM에서 수행되었습니다.

- **Performance Highlights**: 여러 LLM 기반 시뮬레이션에서 일관성이 결여되어 있다는 점과 LLM의 일관성 수준이 전반적인 성능과 강한 상관관계를 갖지 않는다는 점을 발견했습니다. 또한, AdaORPO라는 강화 학습 기반 알고리즘을 제안하여 7개의 LLM에서 시뮬레이션의 신뢰성을 향상시키는 방법을 검증했습니다.



### Leveraging Language Models and Bandit Algorithms to Drive Adoption of Battery-Electric Vehicles (https://arxiv.org/abs/2410.23371)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)과 Contextual Bandit 알고리즘을 결합하여 개별 참가자의 가치에 맞춘 대화형 행동 변화 개입(conversational interventions)을 개발하였습니다. 이러한 개입은 배터리 전기차(BEVs)의 채택을 촉진하는 데 주력합니다.

- **Technical Details**: 연구에서는 LLM이 참가자의 인구 통계를 기반으로 값(target values)을 학습하도록 하며, 이를 통해 설득력 있는 주장을 생성합니다. 기존의 연구를 바탕으로 LLM을 사용하여 가상의 설문 참가자 역할을 하는 방법을 제안했습니다.

- **Performance Highlights**: 벤치마킹 결과, 우리의 Bandit-enhanced LLM이 인구 통계학적으로 타겟팅된 값을 사용하는 대화형 개입을 생성하는 데 있어 기존 LLM보다 효과적임을 입증했습니다.



### Can Models Help Us Create Better Models? Evaluating LLMs as Data Scientists (https://arxiv.org/abs/2410.23331)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 기능을 평가하기 위한 새로운 벤치마크인 FeatEng를 제안합니다. 주어진 데이터셋 설명을 바탕으로 피쳐 엔지니어링 코드를 생성하는 방식으로, LLM의 도메인 지식과 문제 해결 능력을 한층 더 깊이 평가할 수 있도록 설계되었습니다.

- **Technical Details**: FeatEng 벤치마크는 Python 함수를 활용하여 주어진 DataFrame을 변형하는 문제를 기반으로 합니다. 모델은 열과 행을 추가, 삭제 또는 변경하여 XGBoost 모델의 점수를 향상시킬 수 있는 방식으로 테이블 데이터를 변형해야 합니다. 이 과정은 기존 방법들이 간과한 도메인 지식과 알고리즘의 통합을 촉진합니다.

- **Performance Highlights**: FeatEng는 LLM의 실제 적용 가능성을 높이 평가하며, 기존 시스템보다 CRUD(Create, Read, Update, Delete) 작업의 효율성을 증대시킵니다. 이 모델은 데이터 과학 프로젝트에서 피쳐 엔지니어링에 소요되는 시간 및 전문 지식을 현저히 줄일 수 있는 가능성을 보여주고 있습니다.



### DC-Spin: A Speaker-invariant Speech Tokenizer for Spoken Language Models (https://arxiv.org/abs/2410.24177)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Double-Codebook Speaker-invariant Clustering (DC-Spin) 방법을 제안하며, 이는 음성 토큰화 개선을 목표로 하고 있습니다. DC-Spin은 강한 음성 정보가 포함된 화자 불변 토큰을 추출해내며, 입력 변동성에 견딜 수 있도록 설계되었습니다.

- **Technical Details**: DC-Spin은 오디오 신호와 SLM 토큰 간의 연결을 통해 음성 토큰화 성능을 향상시키고, chunk-wise 접근 방식을 통해 스트리밍을 가능케 하며 재학습 없이도 성능 저하를 최소화합니다. 제안된 방법은 Hidden-unit BERT (HuBERT)와 함께 사용되며, 이는 더 나은 초기화를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 DC-Spin 방법은 여러 벤치마크에서 기존의 오픈소스 토크나이저와 비교하여 우수한 성능을 입증하였습니다. 특히, n-gram 예측 가능성과 음소 및 문자 정규화된 상호 정보가 후속 작업을 위한 유용한 지표임을 확인하였습니다.



### Novel Architecture for Distributed Travel Data Integration and Service Provision Using Microservices (https://arxiv.org/abs/2410.24174)
Comments:
          20 pages, 12 figures

- **What's New**: 본 논문은 항공 예약 시스템의 유연성과 성능 향상을 위한 마이크로서비스 아키텍처(microservices architecture)를 소개합니다.

- **Technical Details**: 이 아키텍처 디자인은 Redis 캐시(technologies), 두 가지 메시징 시스템(Kafka, RabbitMQ) 및 두 가지 스토리지(MongoDB, PostgreSQL)를 통합합니다. OAuth2와 JWT를 통한 보안 통신(authorization techniques)을 도입하여 높은 수요의 여행 서비스 관리에서 필수적입니다.

- **Performance Highlights**: 데이터 일관성은 99.5%에 달하고 데이터 전파(latency)는 75ms 이하로, 마이크로서비스 간 빠르고 신뢰할 수 있는 상호 통신(intercommunication)이 가능합니다. 시스템 처리량(throughput)은 초당 1050 이벤트에 도달했으며, Redis 캐싱을 통해 92%의 캐시 적중률을 달성하여 데이터베이스 부담을 줄이고 응답 속도를 증가시켰습니다. 에러율은 0.2%로 실시간 데이터 통합 처리에서 시스템의 효율성을 더욱 향상시켰습니다.



### Redefining <Creative> in Dictionary: Towards a Enhanced Semantic Understanding of Creative Generation (https://arxiv.org/abs/2410.24160)
- **What's New**: 본 논문에서는 "creativity"라는 추상 개념을 TP2O(task) 과제를 통해 구체화하고, 이를 위한 새로운 토큰인 CreTok(<CreTok>)을 도입합니다. CreTok는 서로 관련 없는 두 개념을 조합하는 방식으로 창의성을 정의하며, 모델이 창의적 개념 융합을 학습하도록 돕습니다.

- **Technical Details**: CreTok의 구현은 반복적인 텍스트 쌍 샘플링과 목표 프롬프트 간의 코사인 유사성을 최적화하는 과정을 포함합니다. 각 반복에서 두 개의 텍스트 쌍(t1, t2)이 샘플링되어, 제약 프롬프트와의 유사성을 최적화하며 창의적인 출력을 생성하게 됩니다.

- **Performance Highlights**: 실험 결과, CreTok는 기존 SOTA diffusion 모델을 초월하는 창의적 능력을 보여주었으며, 다양한 개념에 대한 유연성을 증대시키고, 추가적인 재훈련 없이도 새로운 개념을 효율적으로 생성할 수 있도록 지원합니다.



### Scaling Concept With Text-Guided Diffusion Models (https://arxiv.org/abs/2410.24151)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 기존의 개념 교체가 아닌 개념 자체를 향상시키거나 억제할 수 있는 새로운 접근 방식을 탐구합니다. 이를 위해 ScalingConcept이라는 메소드를 도입했으며, 이는 실제 입력에서 분해된 개념을 조정할 수 있는 간단하면서도 효과적인 방법입니다.

- **Technical Details**: ScalingConcept은 개념 복원과 제거 간의 차이를 모델링하여 개념을 표현합니다. 이 과정에서 노이즈 예측의 차이를 사용하고, 스케일링 요소를 통해 다양한 확산 시간 단계에서 모델링을 제어합니다. 추가로, 노이즈 정규화 항을 도입하여 개념 스케일링의 정확성을 높입니다.

- **Performance Highlights**: WeakConcept-10 데이터셋에서 ScalingConcept 방법이 기존 개념 편집 방법보다 우수한 성능을 보였으며, 이는 이미지 생성, 물체 스티칭, 날씨 조작, 소리 강조 및 생성적 사운드 제거와 같은 다양한 새로운 제로샷(zero-shot) 응용 프로그램을 가능하게 합니다.



### Nearest Neighbor Normalization Improves Multimodal Retrieva (https://arxiv.org/abs/2410.24114)
- **What's New**: 이 논문에서는 추가 훈련 없이 훈련된 대비 이미지-텍스트 검색 모델의 오류를 수정하는 간단하고 효율적인 방법인 Nearest Neighbor Normalization (NNN)을 소개합니다.

- **Technical Details**: NNN은 k개의 가장 가까운 쿼리 임베딩을 사용하여 검색 후보의 점수를 정규화하여, 고차원 임베딩 공간에서 허브 문제를 완화합니다. NNN은 사전 훈련된 모델을 사용하여 하위 선형 시간 복잡도로 작동하며, 참조 데이터베이스에 대한 추가 훈련이 필요하지 않습니다.

- **Performance Highlights**: NNN을 사용하여 CLIP, BLIP, ALBEF, SigLIP, BEiT 등 다양한 대비 모델에서 텍스트 검색 및 이미지 검색 모두에서 검색 메트릭의 향상을 보여주었습니다. 또한, NNN은 성별 편향과 같은 해로운 편향을 줄일 수 있습니다.



### In-Context Fine-Tuning for Time-Series Foundation Models (https://arxiv.org/abs/2410.24087)
- **What's New**: 본 연구에서는 시간 시계열 기반의 Foundation 모델이 'in-context fine-tuning'을 통해 여러 시계열 예제를 활용하여 미래의 특정 시계열을 예측하는 방법론을 제시합니다.

- **Technical Details**: 제안된 모델은 미리 훈련된 Foundation 모델을 바탕으로 하며, 시계열 예측을 위해 목표 시계열의 이력뿐만 아니라 유사한 다른 시계열 예제들을 맥락 창(context window)에 포함시켜 사용하는 것이 특징입니다. 이러한 접근 방식은 모델이 특정 도메인의 데이터 분포에 보다 적합하도록 돕습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안된 'in-context fine-tuning' 접근 방식은 상관 관계가 있는 예제를 활용하여 기존의 감독 학습 방법, 통계 모델 및 다른 시계열 Foundation 모델보다 25% 더 나은 성능을 보입니다. 또한, 목표 데이터셋에 대해 명시적으로 fine-tuning된 Foundation 모델의 성능조차도 약간 초과합니다.



### Navigating the Unknown: A Chat-Based Collaborative Interface for Personalized Exploratory Tasks (https://arxiv.org/abs/2410.24032)
- **What's New**: 이 논문에서는 개인화된 탐색을 촉진하는 협업 도우미 시스템인 CARE를 소개합니다. CARE는 다중 에이전트 LLM 프레임워크와 구조화된 사용자 인터페이스를 결합하여 탐색 작업에서 개인화의 한계를 극복하려는 시도를 합니다.

- **Technical Details**: CARE 시스템은 세 가지 주요 요소인 채팅 패널(Chat Panel), 솔루션 패널(Solution Panel), 요구 사항 패널(Needs Panel)로 구성되어 있습니다. 이 인터페이스는 사용자 쿼리를 반복적으로 다듬을 수 있는 동적 상호작용을 가능하게 하며, 다중 에이전트 협업 프레임워크를 통해 사용자 요구 사항을 식별하고 맞춤형 솔루션을 생성합니다.

- **Performance Highlights**: 22명의 참가자를 대상으로 한 연구에서 CARE는 기존 LLM 챗봇보다 일관되게 선호되었으며, 참가자들은 CARE가 인지적 부담을 줄이고 창의적인 탐색을 유도하며 개인화된 솔루션을 제공하는 데 도움을 준다고 평가했습니다.



### Representative Social Choice: From Learning Theory to AI Alignmen (https://arxiv.org/abs/2410.23953)
Comments:
          Full version (20 pages). Under review. An excerpt was previously accepted to NeurIPS 2024 Pluralistic Alignment Workshop

- **What's New**: 본 연구는 집단적 의사결정에서 민주적 대표성을 모델링하기 위한 대표 사회 선택(framework) 프레임워크를 제안합니다. 이는 문제와 개인의 수가 너무 많아 메커니즘이 모든 선호를 직접 고려할 수 없는 시나리오를 다루고 있습니다.

- **Technical Details**: 대표 사회 선택에서 모집단은 개인-문제 쌍의 유한 샘플로 표현되며, 여기서 사회 선택 결정이 이루어집니다. 본 논문에서는 대표 사회 선택의 복잡성을 수학적 통계학적(statistical) 학습 문제로 모델링하고, 기계 학습의 이론을 활용하여 일반화(generalization) 특성을 증명합니다.

- **Performance Highlights**: 대표적인 의사결정 상황인 배심원 재판, 입법 과정, 기업 거버넌스 및 최근의 AI 정렬 과정에서 적용 가능성이 있습니다. 이 연구는 사회 선택 이론과 AI 정렬의 교차점에서 새로운 연구 방향을 열어줍니다.



### Failure Modes of LLMs for Causal Reasoning on Narratives (https://arxiv.org/abs/2410.23884)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 인과 추론 능력을 조사했습니다. 구체적으로는 내러티브에서 인과 관계를 추론해내는 문제를 통해 분석했습니다. 연구 결과, 최신 모델조차도 신뢰할 수 없는 단축키를 사용하는 경향이 발견되었습니다.

- **Technical Details**: LLMs는 사건의 위상적 순서를 기반으로 인과 관계를 결정하는 경향이 있으며, 이로 인해 사건들이 정확한 인과 순서로 서술되지 않을 경우 성능이 저하됩니다. 일반적으로 사용되는 파라메트릭 지식을 지나치게 의존하는 경향도 보여졌습니다.

- **Performance Highlights**: 인과 그래프를 명시적으로 생성할 경우 성능이 개선되는 반면, 단순한 Chain-of-Thought(코트) 방식은 효과적이지 않은 것으로 나타났습니다. 이러한 결과는 인과 추론을 향상시키기 위한 향후 기술 개발의 기초가 될 수 있습니다.



### Reasons and Solutions for the Decline in Model Performance after Editing (https://arxiv.org/abs/2410.23843)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문은 지식 편집 모델의 성능 저하 원인을 조사하고 이를 최적화하기 위한 방법을 제시합니다. 특히, 데이터와 모델 관점에서 편집 모델의 성능 저하 요인들을 분석하고, 편집 성능을 개선하기 위한 새로운 방법인 Dump for Sequence (D4S)를 제안합니다.

- **Technical Details**: 이 논문은 Multi-Question Dataset (MQD)를 사용하여 다양한 편집 데이터 유형이 모델 성능에 미치는 영향을 평가합니다. 실험을 통해 편집 모델의 성능은 편집 목표의 다양성과 시퀀스 길이에 의해 주로 영향을 받는다는 것을 확인했습니다. 또한 편집 모델 레이어의 L1-norm 성장과 편집 정확도 간의 강한 상관관계를 분석하여, 성능 병목 현상의 주요 원인을 밝혔습니다.

- **Performance Highlights**: 제안된 D4S 방법은 편집 레이어의 L1-norm 증가를 감소시켜 성능을 향상시키고, 사용자가 여러 번의 효과적인 편집을 수행할 수 있도록 하여 모델 손상을 최소화합니다. 이 방법은 기존 편집과 관련된 성능 저하 문제를 효과적으로 해결합니다.



### Artificial intelligence to improve clinical coding practice in Scandinavia: a crossover randomized controlled tria (https://arxiv.org/abs/2410.23725)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이번 연구는 임상 코딩을 지원하기 위해 개발된 AI 도구인 Easy-ICD의 성능을 평가하며, 복잡한 (complex) 텍스트와 간단한 (simple) 텍스트에 대한 코딩 시간과 정확성 향상을 실험적으로 분석합니다.

- **Technical Details**: 연구는 교차 무작위 대조 시험 (crossover randomized controlled trial)으로 진행되었으며, 참가자들은 두 그룹으로 무작위 배정되어 복잡한 텍스트와 간단한 텍스트를 수동 코딩과 Easy-ICD 도구를 사용하여 코딩하였습니다. 결과는 Mann-Whitney U test를 통해 분석되었으며, 복잡한 임상 텍스트의 경우 평균 코딩 시간 차이는 123초로, Easy-ICD 사용 시 46%의 시간 절약이 나타났습니다.

- **Performance Highlights**: 연구 결과, 복잡한 텍스트에서는 AI 도구 사용 시 코딩 시간이 현저히 감소하였으나, 코딩 정확성의 개선은 통계적으로 유의미하지 않았습니다. 간단한 텍스트에서는 시간 차이가 없었으며, 본 연구는 임상 워크플로우에서 AI 도구의 활용 가능성을 보여주었습니다.



### OCEAN: Offline Chain-of-thought Evaluation and Alignment in Large Language Models (https://arxiv.org/abs/2410.23703)
Comments:
          10 pages

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 체인 오브 사고(chain-of-thought) 평가를 위한 새로운 오프라인 평가 프레임워크인 OCEAN을 제안합니다. 이 프레임워크는 LLM의 사고 과정을 MDP(Markov Decision Process)로 모델링하며 지식 그래프(knowledge graph)의 피드백을 활용하여 LLM의 행동을 최적화합니다.

- **Technical Details**: OCEAN 프레임워크는 LLM의 체인 오브 사고 능력을 평가하기 위해 지식 그래프와의 상호작용을 다루며, KG-IPS(inverse propensity scores) 추정기를 통해 피드백을 제공합니다. 이 모델은 정책 평가와 최적화를 가능하게 하며, LLM과 지식 그래프 간의 이질성을 극복하는데 중점을 둡니다.

- **Performance Highlights**: OCEAN은 LLM의 체인 오브 사고 경로를 향상시키면서도 LLM의 일반적인 능력이나 정보 생성 품질에는 영향을 미치지 않는다는 것을 실증적으로 보여줍니다. 오프라인 데이터 샘플을 활용해 정량적으로 LLM의 정책 최적화를 직면한 다양한 문제에 대해 상대적인 성능 향상을 관찰할 수 있었습니다.



### Using Multimodal Deep Neural Networks to Disentangle Language from Visual Aesthetics (https://arxiv.org/abs/2410.23603)
- **What's New**: 이 논문은 아름다움에 대한 인간의 평가를 예측하기 위해 단일 모드 비전, 단일 모드 언어 및 다중 모드 심층 신경망(DNN) 모델의 학습된 표현에 대한 선형 디코딩(linear decoding)을 사용하여 감정적이고 미적 경험에서 인식과 언어를 분리하는 도전을 다룹니다.

- **Technical Details**: 연구에 사용된 주요 데이터셋은 OASIS dataset로, 900개의 이미지로 구성되어 있으며, 각 이미지는 평균 100~110명의 평가자에 의해 평가된 점수를 포함합니다. 연구는 단일 모드 및 다중 모드 심층 신경망 모델에서 추출한 특징을 사용하여 인간의 미적 평점을 예측하는 방법론을 디테일하게 설명합니다. 특히, unimodal vision model(예: SimCLR)은 아름다움 평가에서 설명 가능한 분산의 대부분을 차지하며, 언어 정렬 비전 모델(예: SLIP)은 unimodal vision 모델에 비해 작은 이득만을 보여줍니다.

- **Performance Highlights**: 총 설명 가능한 분산의 75%까지 예측 가능한 unimodal vision 모델이 확인되었으며, CLIP 모델은 경우에 따라 87%의 설명 가능한 분산을 설명합니다. 하지만 SLIP 모델의 결과를 통해 언어의 영향을 조절하면서 비전과 언어의 관계를 더 깊이 이해할 수 있는 가능성을 제시합니다.



### End-to-End Ontology Learning with Large Language Models (https://arxiv.org/abs/2410.23584)
- **What's New**: 이 논문에서는 OLLM이라는 새로운 방법을 소개하며, 이는 온톨로지(ontology)의 분류적 뼈대를 처음부터 구성하는 일반적이고 확장 가능한 방법이다. 기존의 서브태스크(subtask) 중심 방법과는 달리 전체 서브컴포넌트를 모델링하여 고빈도 개념에 대한 오버피팅(overfitting)을 줄인다.

- **Technical Details**: OLL(M은 대규모 언어 모델(large language models)을 세밀 조정(fine-tuning)하는데 커스텀 정규화기(custom regulariser)를 사용하여 온톨로지의 서브컴포넌트를 모델링한다. 생성된 온톨로지의 품질을 평가하기 위해 새로운 메트릭(metric) 수트를 도입하여, 기존의 메트릭과 달리 심층 학습(deep learning) 기술을 활용해 그래프 간의 더 탄력적인 거리 측정을 정의한다.

- **Performance Highlights**: OLLM은 위키피디아(Wikipedia)에서 수행된 정량적 및 정성적 결과 모두에서 서브태스크 조합 방법에 비해 더 높은 성과를 보였다. OLLM을 사용하여 생성된 온톨로지는 더 의미적으로 정확하며 구조적 완전성을 유지한다. 또한, OLLM은 새로운 도메인에 효과적으로 적응할 수 있으며, 소수의 훈련 예시만으로도 가능하다.



### Tiny Transformers Excel at Sentence Compression (https://arxiv.org/abs/2410.23510)
- **What's New**: 이번 연구는 작은 네트워크도 유효한 영어 문장을 구성할 수 있음을 보여주며, 서브워드 서식에서 더 큰 텍스트 조각으로 이동하여 대규모 언어 모델을 최적화할 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 1-3층으로 구성된 BERT 유사(transformer) 변환기를 훈련하여 표준 영어 문장을 단일 3킬로바이트 토큰으로 인코딩하고 디코딩하는 방법을 실험했습니다. 이를 통해 문장 압축 및 복원이 가능함을 입증했습니다.

- **Performance Highlights**: 연구 결과, 모델 차원, 깊이 및 복원 입력 너비를 변경하는 것이 문장 복원 품질에 미치는 영향을 분석했으며, 적은 자원으로 높은 품질의 문장 복원이 가능하다는 것을 보여주었습니다.



### Learning to Achieve Goals with Belief State Transformers (https://arxiv.org/abs/2410.23506)
- **What's New**: 우리는 'Belief State Transformer'를 소개합니다. 이 모델은 prefix와 suffix를 입력으로 받아 다음 토큰을 예측하는 독창적인 목표를 가지고 있습니다. 기존의 forward-only transformer가 해결하기 어려운 문제를 효과적으로 배울 수 있습니다.

- **Technical Details**: Belief State Transformer는 정확한 예측을 위해 필요한 모든 정보를 캡처하는 compact belief state를 학습하는 데 중점을 둡니다. 모델의 각 구성 요소는 표준 Transformer가 부족한 어려운 시나리오에서 필수적임을 보여주는 경험적 ablation 결과가 있습니다.

- **Performance Highlights**: 이 모델은 알려진 prefix와 suffix를 가진 스토리 작성 작업에서 Fill-in-the-Middle 방법보다 더 나은 성과를 보이며, 목표가 알려지지 않은 상황에서도 향상된 성능을 입증합니다. Belief State Transformer는 효율적인 goal-conditioned decoding과 우수한 테스트 시간 추론 및 높은 품질의 텍스트 표현을 가능하게 합니다.



### All or None: Identifiable Linear Properties of Next-token Predictors in Language Modeling (https://arxiv.org/abs/2410.23501)
- **What's New**: 이번 논문에서는 다양한 언어 모델에서 선형 속성의 보편성을 설명하기 위해 identifiability(식별 가능성) 개념을 분석합니다. 다양한 모델들 간의 분포가 동일할 때 선형적인 속성이 공유된다는 점을 논의합니다.

- **Technical Details**: 논문은 distribution-equivalent next-token predictors에 대한 identifiability 결과를 증명합니다. 또한 relational linearity를 정제하여 다양한 선형성의 개념들이 어떻게 분석될 수 있는지를 보여줍니다. 특히, autoregressive 모델 및 Transformer 아키텍처에 주목합니다.

- **Performance Highlights**: 적절한 조건 하에, 이러한 선형 속성은 모든 또는 어떤 distribution-equivalent next-token predictors에 대해 유효함이 입증됩니다.



### Learning and Transferring Sparse Contextual Bigrams with Linear Transformers (https://arxiv.org/abs/2410.23438)
- **What's New**: 본 연구에서는 Sparse Contextual Bigram (SCB) 모델을 제안하여 트랜스포머의 학습 동역학과 샘플 복잡성을 분석합니다. 이 모델은 전통적인 bigram 모델의 자연스러운 확장으로, 다음 토큰의 생성이 마지막 토큰에 의해 결정된 드문 이전 위치 집합에 의존합니다.

- **Technical Details**: 연구에서는 1층 선형 트랜스포머를 활용하여 SCB 태스크를 학습하는 과정의 수렴 보장을 제공하고, 비선형 ℓ1-정규화된 MSE 손실을 사용한 알고리즘을 제안합니다. 초깃값 및 하이퍼파라미터에 따라 다소 온건한 조건 하에서 다항적 의존성을 기반으로 진실 값을 회복할 수 있음을 보입니다.

- **Performance Highlights**: 경험적으로, 제안된 알고리즘은 작은 배치 사이즈에서도 진실 값으로 수렴하며, 전이 학습(transfer learning)을 통해 초기 샘플 집중 단계를 우회하고도 다운스트림 태스크의 진실 값에 도달할 수 있음을 확인하였습니다.



### Mind the Gap: A Generalized Approach for Cross-Modal Embedding Alignmen (https://arxiv.org/abs/2410.23437)
Comments:
          18 pages, 3 figures

- **What's New**: 본 논문에서는 다양한 텍스트 유형 간의 의미론적 격차를 줄일 수 있는 새로운 projection-based 방법을 제안합니다. 이 방법은 transfer learning에서의 adapter 모듈에서 영감을 받아 개발되었으며, 코딩 코드와 유사 코드, 영어와 프랑스어 문장 등의 다양한 텍스트 타입 간의 효율적인 정렬을 가능하게 합니다.

- **Technical Details**: 우리의 모델은 서로 다른 텍스트 모달리티의 임베딩을 통합된 의미 공간으로 정렬하는 프로젝션 네트워크를 기반으로 합니다. 이 네트워크는 두 개의 Transformer 기반 인코더와 하나의 숨겨진 레이어로 구성됩니다. 각 인코더는 768 차원의 임베딩을 생성하며, 프로젝션 네트워크는 이러한 임베딩을 서로의 공간에 정렬시키는 역할을 수행합니다. 이 과정에서 ReLU 활성화 함수가 적용됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 정보 검색 방법인 Okapi BM25 알고리즘과 Dense Passage Retrieval (DPR) 모델을 능가하며, Sentence Transformers의 정확도에 가까운 성능을 보여줍니다. 다양한 평가를 통해 우리의 방법이 다양한 작업에서 효과적이고 일반성이 있음을 입증하였습니다.



### Exploiting Phonological Similarities between African Languages to achieve Speech to Speech Translation (https://arxiv.org/abs/2410.23323)
- **What's New**: 이 논문은 아프리카 언어들 간의 언어학적 유사성을 활용하여 직접적인 음성-음성 번역(Direct Speech-to-Speech Translation, S2ST)에 대한 파일럿 연구를 제시합니다. 기존의 데이터 주석이 비효율적이거나 실현 불가능한 경우의 해결책으로, 언어의 동일 계통 내에서 음성 сег먼트를 매핑하는 세그먼트 기반 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 세그먼트 매핑 기법을 활용하여 서로 다른 언어 계통 간에도 음성 세그먼트를 매핑할 수 있습니다. 또한, 내장된 지도 확산(Guided Diffusion) 방식을 통해 음성 데이터를 처리하며, 사전 학습된 쌍 세그먼트를 사용하여 매핑 프로세스를 개선합니다. 이 연구는 케냐 방송 공사(KBC)의 데이터셋을 활용하여 다섯 개 언어 (스와힐리어, 루오어, 키쿠유어, 난디어, 영어)에서 모델의 효능을 평가합니다.

- **Performance Highlights**: 모델은 세그먼트 쌍 매핑 및 번역 품질에서 경쟁력 있는 성능을 보였습니다. 특히, 동일 계통의 언어에서 더 높은 번역 정확도를 보였으며, 세그먼트 길이가 번역의 정확성에 중요한 영향을 미친다는 결과가 나왔습니다. 전통적인 ASR-MT 기법들과 비교했을 때 제안한 모델은 비슷한 번역 성능을 달성했습니다.



### VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration (https://arxiv.org/abs/2410.23317)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)의 추론 속도를 향상시키기 위해 VL-Cache라는 새로운 Key-Value (KV) 캐시 압축 방법을 제안합니다. 기존의 KV 캐시 압축 방식이 VLM에 최적화되지 않았음을 보여줍니다.

- **Technical Details**: VL-Cache는 VLM의 주의 (attention) 스파시티 패턴을 분석하여 visual 토큰과 언어 토큰을 구분합니다. 레이어 별로 스파시티를 고려하여 KV 캐시 예산을 할당하며, 모달리티 인식 토큰 스코어링 정책을 개발하여 중요한 토큰의 중요도를 평가합니다.

- **Performance Highlights**: 실험 결과, KV 캐시의 10%만 유지해도 전체 작업 정확도를 98% 이상 보존하며, 100개의 토큰 생성을 위한 종단 간 지연 시간을 최대 2.33배 단축시켰습니다. GPU 메모리 사용량을 90% 감소시킴에 따라, 높은 동시성을 지원할 수 있습니다.



### Systematically Analyzing Prompt Injection Vulnerabilities in Diverse LLM Architectures (https://arxiv.org/abs/2410.23308)
- **What's New**: 이 연구는 36개의 대형 언어 모델(LLMs)의 다양한 prompt injection 공격에 대한 취약성을 체계적으로 분석하였습니다.

- **Technical Details**: 144개의 prompt injection 테스트를 통해 모델 매개변수(parameter)와 취약성 간의 강한 상관관계를 발견하였으며, 로지스틱 회귀(logistic regression)와 랜덤 포레스트(random forest) 특성 분석(feature analysis) 등의 통계적 분석 결과 매개변수의 크기와 구조가 취약성에 중요한 영향을 미친다는 것을 확인하였습니다.

- **Performance Highlights**: 56%의 테스트에서 성공적인 prompt injection이 발생하여 다양한 매개변수 크기에서 널리 퍼진 취약성을 강조하였으며, 클러스터링 분석을 통해 특정 모델 구성에 연관된 취약성 프로필(vulnerability profiles)을 식별하였습니다. 이러한 결과는 중요한 인프라 및 민감한 산업에 배치된 LLM에 대한 강력하고 다층적인 방어의 필요성을 강조합니다.



### Why Should This Article Be Deleted? Transparent Stance Detection in Multilingual Wikipedia Editor Discussions (https://arxiv.org/abs/2310.05779)
Comments:
          This submission has been accepted to 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)

- **What's New**: 이번 연구에서는 온라인 플랫폼에서의 콘텐츠 조정(non-transparent moderation) 문제를 개선하기 위해 위키백과의 편집자 토론을 다룬 다국어(multilingual) 데이터셋을 구축하였습니다. 이 데이터셋은 편집자의 입장(keep, delete, merge, comment)과 그에 대한 이유, 및 콘텐츠 조정 정책(content moderation policy)을 포함하고 있습니다.

- **Technical Details**: 새로 구축된 데이터셋은 편집자의 결정에 관한 명시된 이유와 정책을 포함하고 있으며, 3개 언어의 위키백과 편집자 토론을 분석합니다. 연구팀은 이 데이터를 사용하여 편집자의 입장과 해당 정책을 높은 정확도로 예측할 수 있음을 보여주었습니다.

- **Performance Highlights**: 우리의 예측 모델은 결정 과정(decision-making process)의 투명성을 높이며, 자동화된 투명 콘텐츠 조정(automated transparent content moderation)을 위한 기초 자료로 제공됩니다.



### Long$^2$RAG: Evaluating Long-Context & Long-Form Retrieval-Augmented Generation with Key Point Reca (https://arxiv.org/abs/2410.23000)
Comments:
          Accepted to EMNLP'24 (Findings). Camera-ready version

- **What's New**: 이 논문에서는 Retrieval-augmented generation (RAG)의 한계를 극복하기 위해 Long$^2$RAG 벤치마크와 Key Point Recall (KPR) 메트릭을 도입합니다.

- **Technical Details**: Long$^2$RAG는 10개 도메인과 8개 질문 카테고리에 걸쳐 총 280개의 질문으로 구성된 데이터셋입니다. 각 질문은 평균 2,444 단어의 5개의 문서와 연결되어 있으며, KPR은 LLM이 생성한 응답에 얼마나 많은 핵심 포인트를 포함하는지를 평가합니다.

- **Performance Highlights**: 최신 LLM 성능 시험에서 GPT-4o와 같은 폐쇄형 모델이 오픈 소스 모델보다 더 뛰어난 성능을 보였고, 입력 문서의 양이 늘어남에 따라 모델의 성능은 전반적으로 저하되는 경향을 보였습니다. 또한 표준 RAG 절차가 정보 손실을 초래하여 긴 컨텍스트에서 RAG보다 더 약한 성능을 나타냈습니다.



New uploads on arXiv(cs.IR)

### Investigating Bias in Political Search Query Suggestions by Relative Comparison with LLMs (https://arxiv.org/abs/2410.23879)
- **What's New**: 이번 논문에서는 영어 검색 쿼리 제안에서의 편향(bias)을 탐지하고 측정하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 대형 언어 모델(large language models), 쌍 비교(pairwise comparison), Elo 기반 점수(Elo-based scoring)를 조합하여 효과적으로 편향을 식별합니다.

- **Technical Details**: 저자들은 gpt-4-1106-preview를 활용하여 쿼리 제안의 편향을 탐지하는데 필요한 레이블이 지정된 데이터셋을 활용하지 않고도 효과적인 결과를 얻습니다. 1,729개의 편향된 텍스트를 대상으로 6,916회의 비교를 통해 Elo 알고리즘을 적용하여 편향의 강도를 수치화합니다. 이러한 접근 방식은 쿼리 제안의 편향 강도를 0에서 1까지의 스케일 점수로 변환합니다.

- **Performance Highlights**: 저자들은 편향 식별에서 GPT 모델이 제공한 점수가 0에서 100까지로 분포되며, 81%의 쿼리가 0-10 점대에 위치함을 발견했습니다. 이는 모델이 정교한 점수 부여에 있어 한계가 있다는 것을 나타내며, 수동 평가를 통해 10점 이상이 편향으로 간주되었습니다. 이 과정은 정치 정보에 대한 공정하고 안전한 접근을 여는 중요한 단계라고 평가됩니다.



### Leveraging Large Language Models for Medical Information Extraction and Query Generation (https://arxiv.org/abs/2410.23851)
Comments:
          Accepted in WI-IAT '24

- **What's New**: 본 연구는 대형 언어 모델(LLMs)을 임상 시험 검색 과정에 통합하여 환자와 적격한 시험 간의 매칭을 개선하면서 정보 프라이버시를 유지하고 전문가의 감독을 가능하게 하는 시스템을 소개합니다.

- **Technical Details**: 연구에서는 주로 오픈소스 기반의 소형 LLM 6종을 평가하며, 이들 모델이 임상 시험 검색 시 환자 맞춤형 쿼리를 생성하는 능력을 분석합니다. 평가 대상에는 의료 분야에 특화된 모델 1종과 일반적으로 사용되는 모델 5종이 포함됩니다. 생성된 쿼리의 검색 효과를 의료 전문가가 만든 쿼리 및 문헌의 최신 기법과 비교합니다.

- **Performance Highlights**: 연구 결과, 평가된 LLM들은 전문가가 생성한 쿼리와 동등하거나 더 나은 검색 효과를 보여주었습니다. LLM들은 표준 기준선 및 다른 접근 방식을 지속적으로 능가하며, 응답 시간은 1.7초에서 8초 사이로 빠르고, 평균 15-63개의 쿼리 용어를 생성합니다. 이런 결과는 임상 시험 검색에 소형 오픈소스 LLM을 활용하는 것이 성능, 계산 효율성 및 실용성을 균형 있게 조화할 수 있다는 것을 시사합니다.



### Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models (https://arxiv.org/abs/2410.23841)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 지침 준수 능력을 더욱 향상시키기 위한 새로운 벤치마크인 InfoSearch를 제안합니다. 이 벤치마크는 문서 수준의 다양한 속성을 기반으로 하여 검색 모델의 사용자 맞춤형 지침 준수 능력을 평가합니다.

- **Technical Details**: InfoSearch는 여섯 가지 문서 수준 속성(청중, 키워드, 형식, 언어, 길이, 출처)에 걸쳐 효과적으로 지침을 준수할 수 있는 검색 모델의 능력을 평가하기 위해 설계되었습니다. 평가 지표로는 Strict Instruction Compliance Ratio (SICR) 및 Weighted Instruction Sensitivity Evaluation (WISE)를 도입하여 모델의 지침 반응성을 측정합니다.

- **Performance Highlights**: 재정렬 모델은 일반적으로 검색 모델보다 개별 지침 준수 능력이 우수하나, 여전히 특정 속성 처리에서 한계를 보였습니다. 커스터마이즈된 지침을 통한 미세 조정 및 모델 크기 증가가 성능을 향상시키는 데 도움을 주지만, 대부분의 모델은 종합적인 지침 준수에는 부족함이 있습니다.



### Identify Then Recommend: Towards Unsupervised Group Recommendation (https://arxiv.org/abs/2410.23757)
Comments:
          26 pages

- **What's New**: 본 연구에서는 동적 그룹 분포를 처리할 수 없는 기존 GR 모델의 한계를 지적합니다. 특히, 사전 정의된 사용자 그룹 수의 제한성과 비싼 주석 비용 문제를 해결하기 위한 새로운 비지도 그룹 추천 프레임워크, ITR(Identify Then Recommend)을 제안합니다.

- **Technical Details**: ITR 프레임워크는 사용자의 군집을 비지도 방식으로 식별합니다. 사용자 포인트의 적응 밀도를 추정하고, 밀도가 높은 지역을 그룹 중심으로 인식합니다. 이후 히어리스틱 병합 및 분할 전략을 통해 사용자 그룹을 발견합니다. 자기-지도 학습 단계에서 풀 앤 리펄션(pre-text task)과 의사 그룹 추천(pre-text task)을 설계하여 성능을 최적화합니다.

- **Performance Highlights**: ITR 모델은 사용자 추천 및 그룹 추천에서 각각 22.22% NDCG@5 및 22.95% NDCG@5 향상을 보였습니다. 또한, 대규모 산업 추천 시스템에 성공적으로 배포되어 실질적인 결과를 달성했습니다.



### Towards Cross-Modal Text-Molecule Retrieval with Better Modality Alignmen (https://arxiv.org/abs/2410.23715)
Comments:
          BIBM 2024 regular paper

- **What's New**: 본 논문에서는 cross-modal text-molecule retrieval 모델을 제안하며, 두 가지 개선점을 통해 모달리티 간의 정합성을 향상시킵니다. 구체적으로, 저희 모델은 두 가지 모달리티 특수 인코더 위에 learnable memory vectors를 포함한 메모리 뱅크 기반의 feature projector를 쌓아 modality-shared features를 보다 잘 추출합니다.

- **Technical Details**: 모델은 두 개의 개별 인코더로 구성되어 있으며, 각 인코더는 입력된 텍스트와 분자를 인코딩합니다. 메모리 뱅크 기반의 feature projector는 modality-shared features를 추출하여 모달리티 간의 격차를 줄이는 데 도움을 줍니다. 또한, 네 가지 유사도 분포(text-to-text, text-to-molecule, molecule-to-molecule, molecule-to-text)를 계산하여 cross-modal alignment를 위한 second-order similarity losses를 최소화합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 SOTA 성능을 달성하며, 이전 보고된 최고의 결과를 6.4% 초과 달성하였습니다. ChEBI-20 및 PCdes 데이터셋에 대한 광범위한 실험을 통해 제안된 모델의 효과성과 일반화 능력이 강하게 입증되었습니다.



### Demonstrating Linked Battery Data To Accelerate Knowledge Flow in Battery Scienc (https://arxiv.org/abs/2410.23303)
- **What's New**: 리튬 이온 배터리에 대한 연구가 급증하고 있는 가운데, 이 논문은 정보 과부하를 관리하기 위한 구조화된(Structured), 의미론적(Semantic), 그리고 연결된(Linked) 데이터를 기반으로 한 전략을 소개합니다.

- **Technical Details**: 구조화된 데이터는 미리 정의된 기계가 읽을 수 있는 형식을 따르며, 의미론적 데이터는 컨텍스트를 위한 메타데이터를 포함합니다. 연결된 데이터는 다른 의미론적 데이터를 참조하여 상호 연결된 정보의 웹을 형성합니다. 논문에서는 BattINFO라는 배터리 관련 온톨로지를 이용해 용어를 표준화하고 자동화된 데이터 추출 및 분석을 가능하게 합니다. 전면 텍스트 검색과 기계가 읽을 수 있는 데이터를 통합하여 데이터 검색 및 배터리 테스트를 향상시킵니다.

- **Performance Highlights**: 상업적 셀 정보의 통일과 배터리 커뮤니티를 위한 도구 개발을 목표로 하며, 제조업체와 독립적인 사이클링 절차 설명 및 대형 언어 모델(Large Language Models)을 위한 외부 메모리를 포함합니다. 이 접근 방식은 배터리 연구의 가속화와 배터리 테스트의 디지털화를 촉진하며, 커뮤니티의 지속적인 개선 참여를 초대합니다. 제공되는 구조화된 데이터와 도구는 오픈 소스로 제공됩니다.



### Understanding and Scaling Collaborative Filtering Optimization from the Perspective of Matrix Rank (https://arxiv.org/abs/2410.23300)
- **What's New**: 본 연구는 기존의 Collaborative Filtering (CF) 방식의 한계를 극복하기 위해 embedding 테이블의 안정적인 랭크(stable rank)를 정규화(reguralization)하는 새로운 방법론을 제시하고 있습니다. 이는 특히 negative sampling의 필요성을 대체할 수 있는 효과적인 기법으로 발전할 수 있습니다.

- **Technical Details**: 이 연구에서는 embedding 테이블의 성질을 분석하며, singular values와 다양한 CF 손실 함수의 관계를 살펴봅니다. stable rank란 continuous matrix의 랭크로, 이 랭크를 정규화하여 훈련 초기 단계에서 고품질의 embedding을 촉진시킵니다. 이를 통해 모델 훈련의 효율성을 높이면서도 성능은 유지하는 결과를 보여줍니다.

- **Performance Highlights**: stable rank 정규화는 DirectAU와 같은 고비용 손실 함수에서 훈련 시간을 최대 66% 단축시킬 수 있으며, 가벼운 손실 함수인 BPR에 대해서도 최대 21%의 성능 향상을 가져올 수 있음을 실증적으로 입증하였습니다.



### Length-Induced Embedding Collapse in Transformer-based Models (https://arxiv.org/abs/2410.24200)
- **What's New**: 이 논문에서는 긴 텍스트에서의 텍스트 임베딩 성능 저하를 설명하는 Length Collapse라는 현상을 발견하였습니다. 이 현상은 긴 텍스트 임베딩이 좁은 공간으로 집합화됨으로써 발생하며, 이는 서로 다른 텍스트 길이 간의 분포 불일치를 초래하여 후속 작업의 성능에 악영향을 미칩니다.

- **Technical Details**: Length Collapse는 이론적으로 self-attention 메커니즘이 저역 통과 필터(low-pass filter)로 작용하며, 긴 시퀀스가 이 필터의 감쇠율을 증가시킨다는 것을 보여줍니다. TempScale은 softmax() 진단에서 온도를 도입하여 이러한 저역 필터의 감쇠율을 높여, 긴 텍스트 입력에 대해 일관된 성능을 강화합니다.

- **Performance Highlights**: Empirically, TempScale을 적용함으로써 기존 임베딩 모델의 성능을 개선할 수 있음을 입증하였고, Massive Text Embedding Benchmark(MTEB)에서 40개 데이터셋에서 최대 0.53% 성능 향상과 LongEmbed에서 4개의 데이터셋에서 최대 0.82% 성능 향상을 기록했습니다.



### Auditing Google's Search Algorithm: Measuring News Diversity Across Brazil, the UK, and the US (https://arxiv.org/abs/2410.23842)
Comments:
          21 pages, 3 figures, 7 tables

- **What's New**: 이 연구는 브라질, 영국, 미국에서 구글의 검색 알고리즘이 뉴스 다양성에 미치는 영향을 분석하였습니다. 구글 시스템이 제한된 수의 뉴스 매체를 선호하는 경향을 설명하며, 알고리즘 감사(algorithm auditing) 기술을 활용하여 소스 집중도를 측정하였습니다.

- **Technical Details**: 이 연구는 Herfindahl-Hirschman Index (HHI)와 Gini 계수를 사용하여 소스의 집중도를 분석합니다. 데이터는 '뉴스' 탭에서 21일 동안 수집된 143,976개의 검색 결과를 포함하며, 2,298개의 검색 쿼리와 4,296개의 매체를 대상으로 하였습니다.

- **Performance Highlights**: 연구 결과는 구글 검색 결과에서 소스의 인기와 정치적 편향, 콘텐츠의 최신성에 따른 편향이 존재함을 보여줍니다. 이 알고리즘은 뉴스 생태계 내 미디어 불평등을 강화할 수 있음을 시사합니다.



### MoTaDual: Modality-Task Dual Alignment for Enhanced Zero-shot Composed Image Retrieva (https://arxiv.org/abs/2410.23736)
- **What's New**: 이 논문에서는 Zero-shot Composed Image Retrieval (ZS-CIR)와 관련된 두 가지 주요 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 효율성과 확장성을 고려한 두 단계의 프레임워크를 도입하여 모달리티와 작업 불일치를 해결합니다.

- **Technical Details**: 첫 번째 단계에서는 대규모 캡션 데이터셋에서 텍스추얼 인버전 네트워크를 학습하며, 두 번째 단계인 Modality-Task Dual Alignment (MoTaDual)에서는 대형 언어 모델(LLM)을 활용하여 트리플 데이터 생성과 프롬프트 학습을 통해 불일치를 줄입니다.

- **Performance Highlights**: MoTaDual은 네 개의 광범위한 ZS-CIR 벤치마크에서 최첨단 성능을 달성하며 낮은 훈련 시간과 계산 비용을 유지합니다.



### Unveiling User Satisfaction and Creator Productivity Trade-Offs in Recommendation Platforms (https://arxiv.org/abs/2410.23683)
- **What's New**: 사용자 생성 콘텐츠 (UGC) 플랫폼에서 추천 알고리즘의 영향에 대한 새로운 이해를 제시하며, 단기 사용자 만족과 장기 콘텐츠 제작 간의 균형을 어떻게 맞출 수 있는지에 대한 기초 이론을 제공합니다.

- **Technical Details**: 연구에서는 C4 (Cournot Content Creation Competition)라는 게임 이론 모델을 도입하여 추천 알고리즘이 콘텐츠 제작 빈도에 미치는 영향을 분석합니다. 이 모델은 순수 내쉬균형 (Pure Nash Equilibrium)을 구성하여 특정 추천 전략 하에서 콘텐츠 생산량을 예측합니다.

- **Performance Highlights**: 추천 알고리즘의 탐색 강도를 최적화하여 사용자 만족도와 콘텐츠 제작 간의 균형을 맞출 수 있는 기법은 UGC 플랫폼에서의 추천 알고리즘의 사전 배포 감사 도구로 작용할 수 있습니다.



### Mind the Gap: A Generalized Approach for Cross-Modal Embedding Alignmen (https://arxiv.org/abs/2410.23437)
Comments:
          18 pages, 3 figures

- **What's New**: 본 논문에서는 다양한 텍스트 유형 간의 의미론적 격차를 줄일 수 있는 새로운 projection-based 방법을 제안합니다. 이 방법은 transfer learning에서의 adapter 모듈에서 영감을 받아 개발되었으며, 코딩 코드와 유사 코드, 영어와 프랑스어 문장 등의 다양한 텍스트 타입 간의 효율적인 정렬을 가능하게 합니다.

- **Technical Details**: 우리의 모델은 서로 다른 텍스트 모달리티의 임베딩을 통합된 의미 공간으로 정렬하는 프로젝션 네트워크를 기반으로 합니다. 이 네트워크는 두 개의 Transformer 기반 인코더와 하나의 숨겨진 레이어로 구성됩니다. 각 인코더는 768 차원의 임베딩을 생성하며, 프로젝션 네트워크는 이러한 임베딩을 서로의 공간에 정렬시키는 역할을 수행합니다. 이 과정에서 ReLU 활성화 함수가 적용됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 정보 검색 방법인 Okapi BM25 알고리즘과 Dense Passage Retrieval (DPR) 모델을 능가하며, Sentence Transformers의 정확도에 가까운 성능을 보여줍니다. 다양한 평가를 통해 우리의 방법이 다양한 작업에서 효과적이고 일반성이 있음을 입증하였습니다.



New uploads on arXiv(cs.CV)

### URAvatar: Universal Relightable Gaussian Codec Avatars (https://arxiv.org/abs/2410.24223)
Comments:
          SIGGRAPH Asia 2024. Website: this https URL

- **What's New**: 본 논문에서는 전화 스캔을 통해 불확실한 조명 아래에서 포토리얼리스틱(phtorealistic)이고 재조명 가능한 헤드 아바타를 생성하는 새로운 접근 방식을 제안합니다. 아바타는 다양한 환경의 전역 조명을 사용하여 실시간으로 애니메이션되며 재조명될 수 있습니다.

- **Technical Details**: 연구진은 복잡한 빛 전달(learnable radiance transfer)을 직접 모델링하여 실시간 렌더링을 위한 효율적인 방식으로 조명 전송을 통합합니다. 이 방식은 수백 개의 고품질 다중 시점 인간 스캔에서 훈련된 3D 가우시안(3D Gaussians)을 사용하여 보편적인 재조명 가능한 아바타 모델을 구축합니다.

- **Performance Highlights**: 실험 결과, 본 접근 방식은 기존 방법을 크게 능가하며 실시간으로 글로벌 조명을 개선하는 유효성을 보여줍니다.



### Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning (https://arxiv.org/abs/2410.24219)
Comments:
          Accepted at NeurIPS 2024, code available at this https URL

- **What's New**: 이 논문에서는 Text-to-Video (T2V) 생성의 새로운 프레임워크인 DEcomposed MOtion (DEMO)를 제안합니다. DEMO는 텍스트 인코딩과 조건 설정을 콘텐츠와 모션 구성 요소로 분해하여 T2V 생성의 모션 합성을 향상시킵니다.

- **Technical Details**: DEMO에서는 정적인 요소를 위한 콘텐츠 인코더와 시간 역학을 위한 모션 인코더를 도입합니다. 텍스트 인코딩을 콘텐츠 및 모션 인코딩으로 분해하고, 콘텐츠 조건 설정과 모션 조건 설정을 분리하여 복잡한 모션 패턴을 캡처하도록 설계되었습니다. 또한 텍스트-모션 및 비디오-모션 감독 기법을 도입하여 모션 이해 및 생성 능력을 향상시킵니다.

- **Performance Highlights**: DEMO는 MSR-VTT, UCF-101, WebVid-10M 등 여러 벤치마크에서 평가되었으며, 동적 모션 및 시각적 충실성과 관련된 메트릭에서 상당한 개선을 달성하였음을 보여주었습니다.



### Learning Video Representations without Natural Videos (https://arxiv.org/abs/2410.24213)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 자연 비디오를 훈련 데이터에 포함하지 않고도 합성 비디오와 자연 이미지를 통해 유용한 비디오 표현을 학습할 수 있음을 보여줍니다. 비디오 모델을 위한 합성 비디오 데이터셋의 점진적인 진행을 제안하며, 이는 자연 비디오 속성(예: motion, acceleration 등)을 점차적으로 모방합니다.

- **Technical Details**: 제안된 VideoMAE 모델은 합성 비디오에서 사전 훈련된 후 UCF101 행동 분류에서 성능 격차를 97.2% 줄입니다. 또한, 정적 이미지의 자르기를 도입함으로써 UCF101 사전 훈련 모델보다 뛰어난 성능을 발휘합니다. 데이터셋의 낮은 수준 속성을 분석하여 프레임 다양성과 자연 데이터의 유사성이 다운스트림 성능에 상관관계가 있음을 확인했습니다.

- **Performance Highlights**: UCF101-P와 같은 비디오 인식 작업에서, 제안된 모델들은 14개의 왜곡된 데이터셋 중 11개에서 UCF101 사전 훈련 모델보다 뛰어난 성능을 보였습니다. 이는 합성 데이터 훈련의 추가 이점을 나타내며, 우리 접근 방식이 데이터 생성 과정에서 투명성을 제공합니다.



### DELTA: Dense Efficient Long-range 3D Tracking for any video (https://arxiv.org/abs/2410.24211)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 DELTA라는 새로운 방법을 소개하여, 단일 비디오에서 모든 픽셀을 효율적으로 추적할 수 있는 3D 동작 추적 기술을 제안합니다. DELTA는 장기적인 비디오에서 밀집 3D 추적을 가능하게 하여, 서로 다른 프레임의 대응 관계를 정확하게 수립하는 데 중점을 두고 있습니다.

- **Technical Details**: DELTA는 spatio-temporal attention 메커니즘을 사용한 저해상도 추적 후, transformer 기반의 upsampler를 통해 고해상도 예측을 수행하는 방식입니다. 이 방법은 global-local attention 구조를 통해 동적 장면과 복잡한 카메라 움직임에도 불구하고 효율적으로 처리할 수 있도록 설계되었습니다. 특히 log-depth 표현 방식이 최적의 추적 성능을 발휘하는 것으로 확인되었습니다.

- **Performance Highlights**: DELTA는 100 프레임 비디오 완전 추적을 2분 이내에 수행하며, 기존 방법보다 8배 이상 빠른 성능을 보였습니다. CVO와 Kubric3D 데이터셋에서 2D 및 3D 밀집 추적 작업에서 새로운 최첨단 결과를 달성하였고, AJ와 APD3D 지표 모두 10% 이상의 성능 향상이 있음을 입증했습니다.



### No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (https://arxiv.org/abs/2410.24207)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 NoPoSplat라는 새로운 feed-forward 모델을 소개합니다. 이는 포즈 정보가 없는 희소 멀티 뷰 이미지에서 3D Gaussian으로 매개변수화된 3D 장면을 재구성할 수 있습니다. 포토메트릭 손실(photometric loss)만으로 훈련된 이 모델은 실시간 3D Gaussian 재구성을 수행할 수 있습니다.

- **Technical Details**: NoPoSplat는 입력 뷰의 로컬 카메라 좌표를 기준 좌표계(canonical space)로 사용하여 모든 뷰의 Gaussian 프리미티브를 예측합니다. 이를 통해 입력 이미지의 포즈를 정확하게 요구하지 않으면서 장면을 완전하게 재구성할 수 있습니다. 또한, 카메라 내적을 토큰 임베딩(token embedding)으로 변환하여 입력으로 사용하는 방법을 소개합니다.

- **Performance Highlights**: 실험 결과, NoPoSplat의 포즈 자유 접근 방식은 특히 입력 이미지 간의 겹침이 제한적인 경우에서 포즈를 요구하는 방법보다 우수한 새로운 뷰 합성 품질을 보여줍니다. 포즈 추정에서도, 저자들은 기존의 최신 기술에 비해 상당한 개선을 보여줍니다.



### GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering (https://arxiv.org/abs/2410.24204)
- **What's New**: 본 논문에서는 3D Gaussian Splatting (3DGS) 기법을 활용한 물리 기반 역 렌더링(inverse rendering) 문제를 다루며, GeoSplatting이라는 새로운 하이브리드 표현(hybrid representation)을 소개합니다.

- **Technical Details**: GeoSplatting은 명시적인 기하학적 지침과 미분 가능한 물리 기반 렌더링(e.g., physically-based rendering, PBR) 방정식을 통합하여 3DGS를 강화합니다. 이 방법은 스칼라 필드에서 등위면(isosurface) 메시를 추출하고 이를 3DGS 포인트로 변환하여 미분 가능한 방식으로 PBR 방정식을 수립합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 포괄적인 평가를 통해 GeoSplatting이 기존 방법보다 정량적 및 정성적으로 일관되게 우수한 성능을 보인다는 것을 확인했습니다.



### DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion (https://arxiv.org/abs/2410.24203)
Comments:
          NeurIPS2024, Project: this https URL Code: this https URL

- **What's New**: Diffusion 기반 방법론이 2D 이미지나 3D 물체 생성에서 놀라운 성과를 이뤘으나, 3D 장면 및 360도 이미지를 생성하는 데에는 제약이 많았습니다. 이 문제를 해결하기 위해 대규모의 파노라마 비디오-텍스트 데이터셋을 구축하고, 이를 기반으로 텍스트 기반의 파노라마 생성 프레임워크 DiffPano를 제안합니다.

- **Technical Details**: DiffPano는 파노라마 비디오-텍스트 데이터셋을 활용하여 단일 뷰 텍스트-파노라마 확산 모델을 LoRA로 미세 조정하고, 구형 에피폴라 인식 다중 뷰 확산 모델을 설계하여 생성된 파노라마 이미지의 다중 뷰 일관성을 보장하도록 합니다. 이 모델은 텍스트 설명과 카메라 포즈를 입력받아 스케일 가능하고 일관되며 다양한 파노라마 이미지를 생성합니다.

- **Performance Highlights**: DiffPano는 기존 방법 대비 더 확장 가능하고 일관된 파노라마 이미지를 생성할 수 있으며, 주어진 보지 않는 텍스트 설명과 카메라 포즈에 대해 만족스러운 결과를 보여주는 강력한 일반화 능력을 보여줍니다.



### Chasing Better Deep Image Priors between Over- and Under-parameterization (https://arxiv.org/abs/2410.24187)
Comments:
          Codes are available at this https URL

- **What's New**: 본 논문에서는 'lottery image prior' (LIP)라는 혁신적인 이미지 우선성을 제안합니다. LIP는 과도하게 매개변수가 주어진 DNN으로부터 훈련된 희소 서브 네트워크를 활용하여 다양한 이미지 역문제를 해결할 수 있는 가능성을 제시합니다.

- **Technical Details**: LIP는 LTH(로터리 티켓 가설)를 기반으로 하여, 과도하게 매개변수화된 DNN에서 효율적으로 훈련된 희소 서브 네트워크를 식별하는 방법을 제안합니다. 이 연구는 DNN의 내재된 희소성을 활용하여 이미지 복원 성능을 개선합니다.

- **Performance Highlights**: LIP 서브 네트워크는 깊은 디코더(deep decoder)보다 우수한 성능을 보이며, 다양한 테스트 이미지 및 복원 작업에서 높은 전이 가능성을 가지고 있습니다. 또한, LIP는 압축 성능에서도 강력한 효과를 발휘합니다.



### Extended Object Tracking and Classification based on Linear Splines (https://arxiv.org/abs/2410.24183)
- **What's New**: 이 논문은 2차원 확장 객체 추적 및 분류를 위한 선형 스플라인(linear splines) 기반의 새로운 프레임워크를 소개합니다. 이는 기존의 최신 모델들과 달리 복잡한 곡선 형태의 윤곽선을 가진 확장 객체를 효과적으로 표현할 수 있게 해 줍니다.

- **Technical Details**: 제안된 방법은 관측된 데이터와 주어진 형상이 잘 맞는지를 측정하기 위해 정확한 likelihood 모델을 도출하고, kinematic state와 shape vector에 기반해 확장 객체를 모델링합니다. kinematic state는 비선형 칼만 필터(nonlinear Kalman filter)로 추정되고, shape vector는 베이지안 분류기(Bayesian classifier)를 통해 추정됩니다.

- **Performance Highlights**: 제안된 추정기는 기존의 확장 객체 추정기들과 비교했을 때 효과적임을 수치 실험을 통해 입증했습니다. 이 방법은 기계적 계산 노력이 적어 초기 '원시(raw)' 추정값을 생성할 수 있으며, 필요한 경우 전통적인 방법으로 수정을 할 수 있게 합니다.



### Federated Black-Box Adaptation for Semantic Segmentation (https://arxiv.org/abs/2410.24181)
Comments:
          Accepted at NEURIPS 2024

- **What's New**: 이 논문에서는 Federated Learning (FL)을 활용한 시맨틱 분할(semantic segmentation)을 위한 새로운 프레임워크 BlackFed를 제안합니다. BlackFed는 클라이언트와 서버 간의 기울기(gradients) 전송이나 모델 아키텍처(model architecture) 전송 없이 데이터 프라이버시를 효과적으로 보호합니다.

- **Technical Details**: BlackFed는 클라이언트 모델의 가중치는 제로 오더 최적화(zero-order optimization, ZOO)를 사용하여 업데이트하고, 서버의 가중치는 퍼스트 오더 최적화(first-order optimization, FOO)를 이용하여 업데이트하는 방식으로 설계되었습니다. 이 프레임워크는 스플릿 신경망(split neural networks, split-nn)을 활용하여 분산 학습 문제로 FL을 정식화했습니다.

- **Performance Highlights**: 제안된 방법은 여러 컴퓨터 비전 및 의료 이미징 데이터셋에서 평가되었으며, 개별 훈련 방식보다 우수한 성능을 보여 프레임워크의 효과성을 입증했습니다. 이는 다른 FL 기반 세그멘테이션 방법과 비교하였을 때 데이터 프라이버시를 강화하면서도 효과적으로 다중 클래스 세그멘테이션을 수행할 수 있는 가능성을 보여줍니다.



### Redefining <Creative> in Dictionary: Towards a Enhanced Semantic Understanding of Creative Generation (https://arxiv.org/abs/2410.24160)
- **What's New**: 본 논문에서는 "creativity"라는 추상 개념을 TP2O(task) 과제를 통해 구체화하고, 이를 위한 새로운 토큰인 CreTok(<CreTok>)을 도입합니다. CreTok는 서로 관련 없는 두 개념을 조합하는 방식으로 창의성을 정의하며, 모델이 창의적 개념 융합을 학습하도록 돕습니다.

- **Technical Details**: CreTok의 구현은 반복적인 텍스트 쌍 샘플링과 목표 프롬프트 간의 코사인 유사성을 최적화하는 과정을 포함합니다. 각 반복에서 두 개의 텍스트 쌍(t1, t2)이 샘플링되어, 제약 프롬프트와의 유사성을 최적화하며 창의적인 출력을 생성하게 됩니다.

- **Performance Highlights**: 실험 결과, CreTok는 기존 SOTA diffusion 모델을 초월하는 창의적 능력을 보여주었으며, 다양한 개념에 대한 유연성을 증대시키고, 추가적인 재훈련 없이도 새로운 개념을 효율적으로 생성할 수 있도록 지원합니다.



### Scaling Concept With Text-Guided Diffusion Models (https://arxiv.org/abs/2410.24151)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 기존의 개념 교체가 아닌 개념 자체를 향상시키거나 억제할 수 있는 새로운 접근 방식을 탐구합니다. 이를 위해 ScalingConcept이라는 메소드를 도입했으며, 이는 실제 입력에서 분해된 개념을 조정할 수 있는 간단하면서도 효과적인 방법입니다.

- **Technical Details**: ScalingConcept은 개념 복원과 제거 간의 차이를 모델링하여 개념을 표현합니다. 이 과정에서 노이즈 예측의 차이를 사용하고, 스케일링 요소를 통해 다양한 확산 시간 단계에서 모델링을 제어합니다. 추가로, 노이즈 정규화 항을 도입하여 개념 스케일링의 정확성을 높입니다.

- **Performance Highlights**: WeakConcept-10 데이터셋에서 ScalingConcept 방법이 기존 개념 편집 방법보다 우수한 성능을 보였으며, 이는 이미지 생성, 물체 스티칭, 날씨 조작, 소리 강조 및 생성적 사운드 제거와 같은 다양한 새로운 제로샷(zero-shot) 응용 프로그램을 가능하게 합니다.



### Exploring Vision Language Models for Facial Attribute Recognition: Emotion, Race, Gender, and Ag (https://arxiv.org/abs/2410.24148)
Comments:
          52 pages, 13 figures

- **What's New**: 이 논문은 인물의 얼굴로부터 인종, 성별, 나이, 감정 등의 속성을 인식하는 데 있어 기존의 CNN 기반 방법 대신 비전 언어 모델(Vision Language Models, VLMs)을 활용하는 새로운 접근 방식을 제안하고 있습니다. 제안된 솔루션은 PaliGemma 모델을 파인튜닝한 FaceScanPaliGemma와 여러 인물이 포함된 이미지에서 속성을 인식하는 GPT-4o 모델인 FaceScanGPT를 포함합니다.

- **Technical Details**: 본 연구는 FairFace, AffectNet, UTKFace와 같은 다양한 데이터셋을 사용하여 VLMs를 통한 멀티 태스크 분류기(multi-task classifier)를 개발합니다. VLMs는 정교한 언어 이해 능력을 바탕으로 시각적 정보와 텍스트 정보를 동시에 처리할 수 있으며, 각 속성과 관련된 맥락을 이해하는 데 강점을 보입니다.

- **Performance Highlights**: FaceScanPaliGemma 모델은 인종, 성별, 나이, 감정을 각각 81.1%, 95.8%, 80%, 59.4%의 정확도로 인식하였으며, 기존의 PaliGemma, 다른 VLMs 및 최신 방법(SotA)과 비교하여 뛰어난 성능을 보였습니다. 또한 FaceScanGPT는 프롬프트(prompt)를 사용하여 인물의 헤어스타일, 의상 색상, 자세 등 다양한 속성을 탐지する 우수한 멀티 작업 기능을 강조합니다.



### COSNet: A Novel Semantic Segmentation Network using Enhanced Boundaries in Cluttered Scenes (https://arxiv.org/abs/2410.24139)
Comments:
          Accepted at WACV 2025

- **What's New**: COSNet(Cluttered Objects’ semantic Segmentation Network)이라는 새로운 세그멘테이션 네트워크를 도입하여, 리사이클링할 수 있는 물체를 복잡한 환경에서 효과적으로 분리하는 방법을 제시합니다.

- **Technical Details**: COSNet은 경계 신호(boundary cues)와 다중 컨텍스트 정보(multi-contextual information)를 활용하여, 복잡한 장면에서 객체를 정확하게 세그먼트합니다. 새로운 특징 강화 블록(feature sharpening block, FSB)과 경계 강화 모듈(boundary enhancement module, BEM)을 도입하여, 복잡한 배경에서 불규칙한 폐기물의 경계 정보를 강조합니다.

- **Performance Highlights**: COSNet은 ZeroWaste-f와 SpectralWaste 데이터셋에서 각각 1.8% 및 2.1%의 mIoU 메트릭에서 유의미한 향상을 달성하였습니다.



### AIDOVECL: AI-generated Dataset of Outpainted Vehicles for Eye-level Classification and Localization (https://arxiv.org/abs/2410.24116)
Comments:
          19 pages, 4 figures, 3 tables

- **What's New**: 본 연구에서는 이미지 라벨링의 효율성을 높이는 새로운 접근법을 제시합니다. 이 방법은 outpainting을 활용하여 인공적인 컨텍스트와 주석을 생성함으로써 수작업 라벨링 노력을 상당히 줄입니다.

- **Technical Details**: 자율 주행, 도시 계획 및 환경 모니터링과 같은 분야에서의 데이터 부족 문제를 해결하기 위해 AI로 생성된 차량 이미지를 활용하였습니다. 이 데이터셋은 선별된 시드 이미지에서 차량을 탐지하고 크롭한 후 큰 캔버스에 outpainting하여 다양한 현실 조건을 시뮬레이션합니다. 각 이미지는 자동 주석을 통해 상세한 바운딩 박스 좌표 정보를 포함하고 있습니다.

- **Performance Highlights**: Outpainted 차량을 활용한 데이터 증강은 전체 성능 지표를 최대 8% 향상시키고, 저대표 클래스의 예측을 최대 20% 증대시켰습니다. 이 연구는 outpainting을 자가 주석화 접근법으로 활용하여 여러 머신 러닝 도메인에서 데이터셋의 활용성을 높이는 솔루션을 제안합니다.



### Nearest Neighbor Normalization Improves Multimodal Retrieva (https://arxiv.org/abs/2410.24114)
- **What's New**: 이 논문에서는 추가 훈련 없이 훈련된 대비 이미지-텍스트 검색 모델의 오류를 수정하는 간단하고 효율적인 방법인 Nearest Neighbor Normalization (NNN)을 소개합니다.

- **Technical Details**: NNN은 k개의 가장 가까운 쿼리 임베딩을 사용하여 검색 후보의 점수를 정규화하여, 고차원 임베딩 공간에서 허브 문제를 완화합니다. NNN은 사전 훈련된 모델을 사용하여 하위 선형 시간 복잡도로 작동하며, 참조 데이터베이스에 대한 추가 훈련이 필요하지 않습니다.

- **Performance Highlights**: NNN을 사용하여 CLIP, BLIP, ALBEF, SigLIP, BEiT 등 다양한 대비 모델에서 텍스트 검색 및 이미지 검색 모두에서 검색 메트릭의 향상을 보여주었습니다. 또한, NNN은 성별 편향과 같은 해로운 편향을 줄일 수 있습니다.



### Identifying Spatio-Temporal Drivers of Extreme Events (https://arxiv.org/abs/2410.24075)
Comments:
          Accepted at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이번 연구에서는 극단적 사건과 그 원인 간의 시공간적 관계를 이해하기 위한 새로운 접근 방식을 제안합니다. 우리 접근 방식은 기후 데이터에서 spatio-temporal (시공간적) 드라이버를 식별하는 최초의 접근 방식입니다.

- **Technical Details**: 제안된 네트워크는 input variables (입력 변수)의 시공간적 binary masks (이진 마스크)를 통해 극단적 사건을 예측하도록 훈련됩니다. 이는 네트워크가 극단적 사건과 상관관계가 있는 드라이버를 성공적으로 식별할 수 있도록 합니다.

- **Performance Highlights**: 세 가지 새로운 synthetic benchmarks (합성 벤치마크)에서 평가한 결과, 기존의 spatio-temporal anomaly detection (시공간적 이상 탐지) 및 multiple instance learning (다중 인스턴스 학습) 방법들보다 뛰어난 성능을 보였습니다.



### TPC: Test-time Procrustes Calibration for Diffusion-based Human Image Animation (https://arxiv.org/abs/2410.24037)
Comments:
          24 pages, 16 figures, NeurIPS 2024

- **What's New**: 본 논문에서는 Test-time Procrustes Calibration (TPC) 기법을 제안하여, 인간 이미지 애니메이션 시스템에서 발생할 수 있는 구성적 불일치를 효과적으로 해결하고 있습니다. 이 방법은 기존 시스템의 성능을 높이고, 실제 환경에서의 활용 가능성을 높이는 데 기여합니다.

- **Technical Details**: TPC는 각기 다른 입력 이미지와 목표 모션 비디오 간의 인간 형태 간의 상관 관계를 이해할 수 있도록 보정된 참고 이미지를 제공하는 방식으로 작동합니다. 이 방법은 기존의 diffusion 모델과 호환되며 추가적인 훈련 없이 테스트 시간에 적용할 수 있습니다. 코드를 통해 참조 이미지 x와 목표 포즈 프레임 p를 사용하여 애니메이션 프레임 y를 생성하며, 여기서 참조 이미지는 Latent feature로 인코딩됩니다. TPC는 이 과정에 보조 분기를 추가하여 적절한 시각적 상관 관계를 캡처하는 데 도움을 줍니다.

- **Performance Highlights**: 实验 결과에 따르면, TPC를 사용함으로써 현재의 이미지 애니메이션 시스템에서 관찰된 출력 품질의 감소가 개선되었으며, 특히 구성적 불일치가 존재하는 경우에도 더욱 효과적인 성능을 보여주었습니다.



### Handwriting Recognition in Historical Documents with Multimodal LLM (https://arxiv.org/abs/2410.24034)
- **What's New**: 이 논문에서는 Gemini라는 멀티모달 LLM(model) 모델을 사용하여 손으로 쓴 문서의 전사(Transcription) 정확도를 평가하고, 최신 Transformer 기반 방법들과 비교합니다. 특히, 전통적인 OCR(Optical Character Recognition) 접근 방식과의 차별점을 강조하고 있습니다.

- **Technical Details**: 저자는 CNN-BiLSTM 및 TrOCR 아키텍처를 재구현하여 다양한 손글씨 데이터에 대한 단어 및 문자 수준의 정확도를 평가했습니다. 데이터셋으로는 Jeremy Bentham의 Early Modern English 문서 1,200페이지, Early Modern German 문서 450페이지, 그리고 현대 라틴어의 필기체 데이터셋이 사용되었습니다.

- **Performance Highlights**: Gemini 모델은 17세기 영어에 대한 전사에서 SOTA 모델과 유사한 성과를 보였으나, 17세기 독일어에서는 성과가 크게 저조하였습니다. 특히 오류는 문장 부호 해석의 잘못 등으로 나타났습니다. 전반적으로 일반적인 언어(영어)에 대해서는 양호한 성과를 보였으나, 비영어 언어에 대해서는 성능 향상이 필요함을 시사합니다.



### A Multi-Modal Approach for Face Anti-Spoofing in Non-Calibrated Systems using Disparity Maps (https://arxiv.org/abs/2410.24031)
- **What's New**: 논문은 비보정(non-calibrated) 얼굴 인식 시스템에서 속임수 공격을 탐지하기 위한 새로운 방법론을 제안합니다. 특히, 3D 구조의 복잡성을 고려하여 얼굴 속성을 활용해 상대 깊이를 추정하는 모델을 도입했습니다.

- **Technical Details**: 제안된 모델은 Disparity Model로, 얼굴 속성에서 생성된 disparity map을 세 번째 모달리티로 추가하여 두 개의 NIR 모달리티와 함께 작동합니다. 이 다중 모달리티 접근법은 깊이 정보를 제공하지 않는 비보정 시스템에서 속임수 공격 감지를 가능하게 합니다.

- **Performance Highlights**: 비교 방법에 비해 2.45%와 7.94% 낮은 Equal Error Rate (EER)와 False Negative Rate (FNR) 성능을 기록하여, 각각 1.71%의 EER 및 2.77%의 FNR을 달성했습니다. 3D 속임수 공격에 대한 모델 앙상블에서도 2.04%의 EER과 3.83%의 FNR을 기록했습니다.



### Unveiling Synthetic Faces: How Synthetic Datasets Can Expose Real Identities (https://arxiv.org/abs/2410.24015)
Comments:
          Accepted in NeurIPS 2024 Workshop on New Frontiers in Adversarial Machine Learning

- **What's New**: 이번 연구에서는 기존의 합성 얼굴 인식 데이터셋들이 실제 데이터에서 정보 유출 문제를 가지고 있다는 점을 시스템적으로 분석하였습니다. 저자들은 새로운 membership inference attack(회원 추론 공격)을 설계하여, 합성 데이터셋이 생성기 모델을 훈련시키기 위해 사용된 실제 데이터로부터 어떤 정보가 유출되었는지를 평가했습니다.

- **Technical Details**: 이 논문은 6개의 최신 합성 얼굴 인식 데이터셋을 분석하며, 모든 데이터셋에서 원본 실제 데이터셋의 샘플이 유출되었다는 사실을 보여줍니다. 이 연구는 Generative Adversarial Networks(GANs)와 Diffusion Models(DM) 기반의 생성 모델들에 대해 다루며, 정보 유출 과정을 명확히 설명합니다.

- **Performance Highlights**: 저자들은 기존의 합성 얼굴 인식 데이터셋들이 훈련 데이터로부터 중요한 정보를 유출하고 있음을 입증하였으며, 이는 합성 데이터셋의 개인 정보 보호 문제를 강조합니다. 이 연구를 통해 보다 책임감 있게 합성 얼굴 데이터셋을 생성하기 위한 기반이 마련되었습니다.



### Re-assembling the past: The RePAIR dataset and benchmark for real world 2D and 3D puzzle solving (https://arxiv.org/abs/2410.24010)
Comments:
          NeurIPS 2024, Track Datasets and Benchmarks, 10 pages

- **What's New**: 이번 논문에서는 퍼즐 해결 및 재조립 작업을 평가하기 위한 도전적인 기준인 RePAIR 데이터셋을 제안합니다. 이 데이터셋은 2D 및 3D 퍼즐 해결을 위한 현재 기준과는 다른 독특한 특성을 가지고 있습니다.

- **Technical Details**: RePAIR 데이터셋의 조각과 파편은 제2차 세계대전 중 폭격으로 인해 폼페이 고고학 공원에서 프레스코가 붕괴되면서 생겨난 현실적인 특징을 가집니다. 이 조각들은 침식되어 불규칙한 형태와 다양한 크기를 가지며, 재조립 알고리즘에게 도전 과제가 됩니다. 데이터셋은 고해상도 이미지, 파편의 상세 3D 스캔, 고고학자들이 주석을 단 메타데이터를 포함한 다중 모달(multi-modal) 데이터입니다.

- **Performance Highlights**: 이 데이터셋의 Ground truth는 수년간의 지속적인 현장 작업을 통해 생성되었습니다. 약 16,000 개의 조각 중 1,000 조각의 수동 퍼즐 해결이 포함되어 있으며, 현장 발굴 및 청소 작업이 수행되었습니다. 새로운 벤치마크는 현재의 재조립 및 퍼즐 해결 방법에게 명확한 도전 과제를 제시하며, 복잡한 문제 해결에서의 격차를 확인해 줍니다.



### DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination (https://arxiv.org/abs/2410.24006)
Comments:
          Accepted to 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: DiffPAD는 손상된 이미지를 복원하고 적대 공격에 대한 방어력을 강화하기 위해 diffusion 모델을 통합한 새로운 프레임워크입니다. 이 방법은 패치 공격(patch attacks)에 대한 방어에 중점을 두고, 기존의 글로벌 공격을 방어하는 방법의 한계를 극복합니다.

- **Technical Details**: DiffPAD는 다운샘플된 입력 이미지를 슈퍼 해상도 복원(super-resolution restoration)하여 처리하고, 이진화(binarization), 동적 임계값 설정(dynamic thresholding) 및 슬라이딩 윈도우(sliding window) 기술을 사용하여 적대적 패치를 효과적으로 지역화합니다. 이러한 과정은 패치 크기와 diffusion 복원 오차 간의 이론적으로 도출된 상관관계에 영감을 받았습니다.

- **Performance Highlights**: DiffPAD는 실험을 통해 적대적 패치 공격에 대해 최첨단 방어력을 달성했으며, 패치 잔여물 없이 자연스러운 이미지를 복원하는 데 뛰어난 성과를 보였습니다. 기존의 방어 방법에 비해 실질적인 향상을 보여주었습니다.



### ImOV3D: Learning Open-Vocabulary Point Clouds 3D Object Detection from Only 2D Images (https://arxiv.org/abs/2410.24001)
Comments:
          Accepted by NeurIPS 2024. Code link this https URL

- **What's New**: 이 논문은 Open-vocabulary 3D object detection (OV-3Det)라는 새로운 접근 방식을 제시하며, 2D 이미지 만을 사용하여 3D 객체 감지 모델을 학습하는 것을 목표로 합니다. 이 방법은 기존의 3D 데이터 부족 문제를 해결하기 위해 pseudo-multimodal representation을 활용합니다.

- **Technical Details**: ImOV3D라는 새로운 프레임워크를 제안하며, 이는 2D 이미지를 3D 포인트 클라우드로 변환하는 monocular depth estimation과 3D 장면에서 렌더링을 통해 2D 이미지와 3D 포인트 클라우드를 통합하여 modality gap을 해소합니다. 이 방식은 훈련 데이터의 도메인 차이를 최소화하며, 2D 메트릭스 정보와 3D 구조적 특성을 결합한 통합 representation을 생성합니다.

- **Performance Highlights**: ImOV3D는 SUNRGBD 및 ScanNet 데이터셋에서 기존의 OV-3Det 방법을 넘어서는 성능을 보여주었으며, 특히 실제 3D 훈련 데이터 없이도 mAP@0.25에서 SUNRGBD에서 7.14%, ScanNet에서 6.78%의 성능 향상을 기록하였습니다. 또한, 실질적인 3D 데이터가 제공될 경우, 이전의 모든 벤치마크 방법들보다 더 큰 성과를 달성하였습니다.



### Localization, balance and affinity: a stronger multifaceted collaborative salient object detector in remote sensing images (https://arxiv.org/abs/2410.23991)
- **What's New**: 본 연구에서는 Optical Remote Sensing Images (ORSI)에서의 Salient Object Detection (SOD)을 위한 새로운 방법인 LBA-MCNet을 제안합니다. 이 방법은 localization, balance, affinity 측면을 통합하여 복잡한 경계 구조와 문맥 관계를 효과적으로 처리합니다.

- **Technical Details**: LBA-MCNet은 Edge Feature Adaptive Balancing and Adjusting (EFABA) 모듈을 통해 정밀한 경계 위치 지정을 수행하고, Global Distributed Affinity Learning (GDAL) 모듈을 통해 전역 문맥을 모델링합니다. EFABA는 경계 정보를 강조하고, GDAL은 최종 레이어에서 반복되는 affinity map을 생성하여 전역 패턴을 포착합니다.

- **Performance Highlights**: 연구에서는 3개의 공개 데이터셋에서 28개의 최신 SOD 방식과 비교하였으며, LBA-MCNet의 성능이 월등함을 입증하였습니다.



### JEMA: A Joint Embedding Framework for Scalable Co-Learning with Multimodal Alignmen (https://arxiv.org/abs/2410.23988)
Comments:
          26 pages, 14 figures

- **What's New**: 본 연구는 레이저 금속 증착 (LMD) 공정에 맞춤화된 새로운 공동 학습 프레임워크인 JEMA (Joint Embedding with Multimodal Alignment)를 소개합니다. Industry 5.0의 발전과 함께 효율적인 공정 모니터링이 점점 더 중요해지고 있으며, JEMA는 다중 모달 데이터(예: 다중 관점 이미지 및 공정 파라미터 메타데이터)를 활용하여 표준화된 의미 표현을 학습합니다.

- **Technical Details**: JEMA는 감독 대조 손실 함수(supervised contrastive loss function)를 적용하여 주요 모달리티를 사용하여 공정을 모니터링할 수 있도록 강건한 학습을 가능하게 하여 하드웨어 요구 사항과 계산 오버헤드를 간소화합니다. 이 방법은 여러 데이터 소스로부터 얻은 지식을 활용하여 단일 모달리티에서의 모니터링을 개선합니다.

- **Performance Highlights**: JEMA는 멀티모달 환경에서 8%의 성능 향상과 단일 모달 환경에서 1%의 성능 향상을 보여주며, 특히 Vision Transformer 모델과 결합했을 때 높은 확장성과 성능을 입증했습니다. 또한, JEMA를 사용하여 메타데이터를 예측할 수 있어 해석 가능성을 높이고 추가된 메타데이터의 기여도를 평가할 수 있습니다.



### Image Synthesis with Class-Aware Semantic Diffusion Models for Surgical Scene Segmentation (https://arxiv.org/abs/2410.23962)
- **What's New**: 본 논문에서는 Class-Aware Semantic Diffusion Model (CASDM)를 제안하여 외과 장면(segmentation) 생성에서 데이터 부족과 불균형 문제를 해결합니다. 이 모델은 병합된 세그멘테이션 맵을 사용하여 이미지 생성을 위한 새로운 접근 방식을 탑재하여, 작은 크기의 중요한 조직 클래스를 우선 고려합니다.

- **Technical Details**: CASDM은 세분화 맵을 조건으로 사용하여 이미지 생성을 수행합니다. 이 모델은 novel class-aware mean squared error와 class-aware self-perceptual loss 함수를 정의하여, 덜 보이는 중요한 클래스를 우선으로 하여 이미지 품질을 개선합니다. 또한, 텍스트 프롬프트를 통해 여러 클래스를 포함한 세그멘테이션 맵을 생성하는 최초의 접근 방식을 통해 모델의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: CASDM은 다양한 데이터셋에서 평가된 결과에서, 특히 작은 클래스와 희귀한 클래스의 세그멘테이션 정확도를 크게 향상시킴을 보여줍니다. CholecSeg8K 및 gastrectomy SISVSE 데이터셋에서 실험을 실시하였으며, 이미지 품질과 다운스트림 세그멘테이션 성능 모두에서 유의미한 개선을 입증했습니다.



### MV-CC: Mask Enhanced Video Model for Remote Sensing Change Caption (https://arxiv.org/abs/2410.23946)
- **What's New**: 이 논문에서는 Remote Sensing Image Change Captioning (RSICC) 분야에서 기존의 복잡한 fusion 모듈 디자인 없이 새로운 비디오 모델 기반 패러다임인 Mask-enhanced Video model for Change Caption (MV-CC)를 도입하였습니다.

- **Technical Details**: MV-CC는 사전 훈련된 비디오 인코더를 사용하여 이중 시간 원격 감지 이미지에서 공간적 및 시간적 특징을 동시에 추출합니다. 이 모델에서는 Change Detection (CD) 기법을 통해 얻은 마스크를 이용하여 관심 영역에 초점을 맞추는 방법을 제안하고, 이를 통해 자연어 캡션을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안한 MV-CC 모델은 기존 최첨단 RSICC 방법들과 비교하여 더 나은 성능을 보였습니다. 제안된 방법은 다양한 CD 모델을 사용하여 CC 성능을 향상시키는 데 성공하였습니다.



### Manipulating Vehicle 3D Shapes through Latent Space Editing (https://arxiv.org/abs/2410.23931)
Comments:
          18 pages, 12 figures

- **What's New**: 이 논문은 3D 차량 모델의 스타일과 기하학적 속성을 지속적이고 정밀하게 수정할 수 있는 새로운 프레임워크를 소개합니다. 기존의 3D 생성 및 편집 방법이 텍스트와 이미지를 3D 모델로 변환하는 데 치중하면서 세부 조정의 필요성을 간과하는 문제가 있었습니다.

- **Technical Details**: 이 연구에서는 DeepSDF 모델을 기반으로 한 프레임워크를 설계하여, 3D 차량 모델의 잠재 공간 내에서 정밀 편집을 가능하게 합니다. 훈련된 3D 객체 회귀기를 사용하여 다양한 속성에 대해 지속적으로 편집할 수 있는 잠재 표현을 얻습니다. 이는 기존의 3D 편집 기술과는 달리 실제 3D 객체를 속성 데이터에 따라 직접 조작할 수 있도록 해줍니다.

- **Performance Highlights**: 이 방법은 다양한 차량 3D 모델에 대해 상세한 편집을 달성할 수 있는 효과성을 입증하며, 편집된 3D 차량 객체의 본래 정체성을 보존할 뿐만 아니라 여러 속성을 동시에 수정할 수 있는 기능을 제공합니다. 이를 통해 구조적 무결성을 손상시키지 않으면서 광범위한 사용자 맞춤화를 지원합니다.



### Uncertainty Estimation for 3D Object Detection via Evidential Learning (https://arxiv.org/abs/2410.23910)
- **What's New**: 본 논문에서는 3D 물체 탐지의 신뢰성을 정의하고 불확실성을 정량화하기 위한 새로운 프레임워크를 제시합니다. 이 프레임워크는 Bird's Eye View(BEV) 표현에서 evidential learning loss를 활용하여 최소한의 계산 오버헤드로 불확실성을 측정합니다.

- **Technical Details**: Evidential Deep Learning(EDL) 접근 방식을 통해 3D 탐지 모델의 BEV 표현에서 각 셀의 불확실성을 정량화합니다. 불확실성 추정이 포함된 복합적인 방법으로 객체의 존재 확률 및 클래스와 위치 관련 불확실성을 동시에 예측할 수 있습니다. 이 시스템은 3D 물체 탐지기가 자동으로 주행 장면을 라벨링하고, 불확실성 추정을 통해 라벨의 정확성을 검증하는 방식으로 작동합니다.

- **Performance Highlights**: 이 프레임워크는 평균 10-20%의 성능 향상을 보였으며, 최종적으로 mAP에서 1% 향상, NDS에서 1-2% 향상의 결과를 보여줍니다. 불확실성 기반 검증을 통해검증이 필요한 출력에 대해 집중적으로 검토하며, 이는 2차 모델의 성능 개선에 기여합니다.



### IP-MOT: Instance Prompt Learning for Cross-Domain Multi-Object Tracking (https://arxiv.org/abs/2410.23907)
- **What's New**: 본 논문에서는 IP-MOT라는 최첨단 multi-object tracking (MOT) 모델을 제안합니다. 이 모델은 기존 방법들이 가지고 있던 텍스트 기반 설명의 한계를 극복하고, 구체적인 텍스트 설명 없이도 작동합니다.

- **Technical Details**: IP-MOT는 두 가지 주요 혁신에 기반합니다: 첫째, 사전 훈련된 vision-language 모델을 활용하여 서로 다른 추적 장면에서도 불변하는 인스턴스 수준의 의사 텍스트 설명을 생성합니다. 둘째, 지식 증류(knowledge distillation)로 강화된 쿼리 균형 전략(query-balanced strategy)을 도입하여 모델의 일반화 능력을 높입니다.

- **Performance Highlights**: MOT17, MOT20, DanceTrack 등 3개의 광범위하게 사용되는 MOT 벤치마크에서 실시한 광범위한 실험 결과, 우리 방법은 동일 도메인 데이터에서 최신 모델들과 경쟁할 만큼 뛰어난 성능을 달성하며, 교차 도메인 입력에 대한 쿼리 기반 트래커의 성능을 크게 향상시킵니다.



### From Web Data to Real Fields: Low-Cost Unsupervised Domain Adaptation for Agricultural Robots (https://arxiv.org/abs/2410.23906)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문은 정밀 농업에서의 Unsupervised Domain Adaptation (UDA)을 통해 새로운 필드에서의 작물 및 잡초 식별 문제를 해결하기 위해, 인터넷에서 수집된 다양한 데이터를 기반으로 한 새로운 도메인 변환 방법을 제안합니다. 이를 통해 필드별 데이터 수집을 최소화하면서 적은 비용으로 특정 필드에 적응할 수 있는 방법을 모색합니다.

- **Technical Details**: 우리는 Multi-level Attention-based Adversarial Discriminator (MAAD)라는 새로운 모듈을 도입하여, 다양한 도메인 간의 특징 분포를 일관되게 학습하도록 모델을 유도합니다. MAAD는 저수준 및 고수준의 두 개의 적대적 판별기를 활용하여 소스 도메인과 타겟 도메인 입력을 구분하며, 이때 공간적 주의 모듈을 추가하여 관련 지역에 집중할 수 있도록 합니다. 본 연구에서는 CenterNet과 MAAD를 결합하여 잎, 줄기 및 정맥 인스턴스를 동시에 탐지합니다.

- **Performance Highlights**: 실험 결과, MAAD를 포함한 모델이 unlabeled 타겟 도메인에서 기존 베이스라인 모델에 비해 7.5%의 객체 탐지 정확도 향상과 5.1%의 키포인트 탐지 개선을 달성했습니다. 특히, 키포인트 탐지 작업에서는 다른 UDA 방법들보다 우수한 성능을 보였습니다.



### Text-DiFuse: An Interactive Multi-Modal Image Fusion Framework based on Text-modulated Diffusion Mod (https://arxiv.org/abs/2410.23905)
Comments:
          Accepted by the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 연구는 복합 열화(compound degradation) 문제를 해결하기 위해 새로운 인터랙티브 멀티모달 이미지 융합 프레임워크인 Text-DiFuse를 제안합니다. 이 프레임워크는 텍스트 조절(diffusion model) 기반으로 하며, 사용자가 원하는 물체를 강조할 수 있는 사용자 정의 텍스트 제어 융합 전략을 제공합니다.

- **Technical Details**: Text-DiFuse는 첫 번째로 데이터의 복합 열화를 제거하기 위해 독립적인 조건부 확산(conditional diffusion)을 적용합니다. 또한, 인코더와 디코더 사이에 융합 제어 모듈(fusion control module, FCM)을 포함시켜 멀티모달 기능의 통합을 관리합니다. 이 모델은 텍스트와 제로샷(zero-shot) 위치 모델을 결합하여 관심 물체를 식별하고, 이들을 강조하는 보조 조정을 수행합니다.

- **Performance Highlights**: Text-DiFuse는 다양한 공공 데이터셋에서 실험을 통해 다른 최첨단 방법들보다 우수한 융합 성능을 달성하였으며, 복합 열화에 내구성이 강하고 일반화 능력이 뛰어나며 의미적 속성에서 현저한 향상을 보였습니다.



### EZ-HOI: VLM Adaptation via Guided Prompt Learning for Zero-Shot HOI Detection (https://arxiv.org/abs/2410.23904)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 Human-Object Interaction (HOI) 감지를 제로샷(zero-shot) 설정에서 수행하기 위한 새로운 방법인 Efficient Zero-Shot HOI detection (EZ-HOI)을 제안합니다. 기존의 복잡한 비전-언어 모델(Vision-Language Models, VLMs)과의 정렬 기법 대신에, 프로프트 학습(prompt learning) 기반으로 VLMs를 효과적으로 적응시키는 접근 방식을 사용합니다.

- **Technical Details**: EZ-HOI는 학습 가능한 텍스트 템플릿과 비주얼 템플릿을 통합하여 VLM의 적응 과정을 안내합니다. 특히, 관련된 보이는 클래스의 정보를 활용하여 보이지 않는 클래스에 대한 학습을 강화하며, LLM(Large Language Model)을 통해 보이는 클래스와 보이지 않는 클래스 간의 차별 정보(disparity information)를 제공합니다. 이러한 방법은 훈련 가능한 파라미터 수를 기존 방법의 10.35%에서 33.95%로 줄입니다.

- **Performance Highlights**: EZ-HOI는 다양한 제로샷 세팅에서 최첨단 성능을 달성하며, 전통적인 접근 방식에 비해 훈련 어려움을 줄이고, 보이지 않는 클래스에 대한 성능을 향상시킵니다. 또한, 모델 파라미터를 66%에서 78%까지 감소시켜 효율성과 효과성을 극대화했습니다.



### NeFF-BioNet: Crop Biomass Prediction from Point Cloud to Drone Imagery (https://arxiv.org/abs/2410.23901)
- **What's New**: 생체량 예측 네트워크(BioNet)는 다양한 데이터 유형에 적응하도록 설계되어 드론 이미지 및 포인트 클라우드 데이터를 활용하여 높은 해상도의 생체량 예측을 가능하게 합니다.

- **Technical Details**: BioNet은 희소 3D CNN(sparse 3D convolutional neural network)과 Transformer 기반 예측 모듈을 이용하여 포인트 클라우드 데이터에서 생체량을 예측하고, NeFF(neural feature field) 모듈을 통합하여 드론 이미지의 2D 시맨틱(semantic) 특성을 3D 표면으로 변환합니다.

- **Performance Highlights**: 포인트 클라우드 방식에서 BioNet은 두 개의 공개 데이터셋에서 최신 기술 대비 약 6.1%의 상대적 개선(RI)을 달성하였고, RGB 이미지 방식에서 BioNet과 NeFF의 조합은 7.9% RI를 기록했습니다. 또한, NeFF 기반 접근법은 저렴한 드론 장착 카메라를 활용하여 대규모 농업 응용 프로그램에 대한 확장 가능한 솔루션을 제공합니다.



### AllClear: A Comprehensive Dataset and Benchmark for Cloud Removal in Satellite Imagery (https://arxiv.org/abs/2410.23891)
Comments:
          Accepted at NeurIPS 2024 Datasets and Benchmarks Track. Code and data available at this https URL

- **What's New**: 본 논문에서는 클라우드 제거(cloud removal)를 위한 세계 최대의 공개 데이터셋인 AllClear를 소개합니다. 이 데이터셋은 23,742개 지역(ROIs)에서 4백만 이미지로 구성되어 있으며, 다양한 땅 사용 패턴을 포함합니다.

- **Technical Details**: AllClear 데이터셋은 2022년에 수집된 다수의 다중 스펙트럼(multi-spectral) 및 합성 개구 레이더(SAR) 이미지를 포함합니다. 데이터셋은 모델이 결측 정보를 보완할 수 있도록 다수의 위성(Sentinel-1, 2 및 Landsat 8/9)의 데이터를 사용합니다. 성능 평가 결과, 데이터 양이 30배 증가할 때 PSNR(peak signal-to-noise ratio)은 28.47에서 33.87로 상승함을 보여주었습니다.

- **Performance Highlights**: 기존 최첨단(cloud removal) 모델들이 AllClear 데이터셋에서 충분히 훈련되지 않음을 발견하였으며, 보다 크고 다양한 데이터셋을 사용하여 성능이 크게 향상되었습니다. 여러 센서 및 긴 시간의 관측을 활용하는 모델이 훨씬 더 좋은 성능을 보임을 확인했습니다.



### Airway Labeling Meets Clinical Applications: Reflecting Topology Consistency and Outliers via Learnable Attentions (https://arxiv.org/abs/2410.23854)
- **What's New**: 본 논문은 자동 기도 해부 레이블링을 위한 새로운 방법을 제안합니다. 이 방법은 Soft Subtree Consistency (SSC)와 Abnormal Branch Saliency (ABS)라는 두 가지 모듈을 통합하여 해부학적 변동성 문제를 해결합니다.

- **Technical Details**: SSC 모듈은 임상적으로 중요한 위상적 관계를 포착하기 위해 소프트 서브트리를 구성하고, ABS 모듈은 비정상 분기 노드를 구별하는 데 도움을 줍니다. 이 방법은 U-형 프레임워크를 채택하여 세분화 수준의 분류 정확도를 점진적으로 향상시키며, 하이퍼 그래프 모델 대신 계층적 관계를 사용하여 예측의 일관성을 개선합니다.

- **Performance Highlights**: 제안된 방법은 AIIB23 데이터 세트에서 평가되었으며, 세분화 수준에서 91.4% 및 서브세분화 수준에서 83.7%의 정확도를 달성했습니다. 이는 서브세분화 정확도의 1.4% 증가와 위상적 일관성의 3.1% 증가로, 질병에 의해 유도된 기도 변형에서도 신뢰할 수 있는 성과를 보여줍니다.



### Stereo-Talker: Audio-driven 3D Human Synthesis with Prior-Guided Mixture-of-Experts (https://arxiv.org/abs/2410.23836)
- **What's New**: 본 연구는 Stereo-Talker라는 새로운 오디오 기반 인간 비디오 합성 시스템을 소개합니다. 이 시스템은 단일 참조 이미지와 오디오 입력을 활용하여 3D로 말하는 비디오를 생성하며, 정확한 입술 동기화(lip synchronization), 표현이 풍부한 신체 제스처(body gestures), 안정적인 비주얼 품질(quality)과 보이는 시점(viewpoint) 제어를 제공합니다.

- **Technical Details**: Stereo-Talker는 두 가지 단계로 이루어진 접근 방식을 따릅니다. 첫 번째 단계에서는 시스템이 오디오 입력을 고해상도 움직임 시퀀스로 매핑하며, 대형 언어 모델(LLM) 사전(prior)과 텍스트 정렬된 의미적 오디오 특징(semantic audio features)을 통합하여 더욱 다양한 표현을 생성합니다. 두 번째 단계에서는 Mixture-of-Experts(MoE) 메커니즘을 사용하여 확산 기반 비디오 생성 모델을 개선합니다. 이를 통해 입력된 움직임 데이터에서 인물 마스크(human masks)를 생성하며, 안정적인 렌더링을 가능하게 합니다.

- **Performance Highlights**: Stereo-Talker는 2,203개의 인물로 구성된 대규모 HD 오디오-비주얼 데이터셋을 사용하여 높은 해상도 비디오를 생성합니다. 이 시스템은 각 관점(viewpoint)에 대해 고유한 표시를 생성할 수 있으며, 문맥적 뉘앙스(nuances)를 이해하기 위해 LLM의 힘을 활용하여 생성된 제스처의 다양성과 안정성을 높였습니다.



### FRoundation: Are Foundation Models Ready for Face Recognition? (https://arxiv.org/abs/2410.23831)
- **What's New**: 이번 연구는 foundation models가 얼굴 인식( Face Recognition, FR) 분야에 적용 가능성을 최초로 조사한 것입니다. 다양한 데이터 가용성을 기반으로 이러한 모델을 얼굴 인식에 적합하게 조정하는 방법을 제안하고 실험하였습니다.

- **Technical Details**: 본 연구에서는 DINOv2와 CLIP 두 가지 foundation models를 사용하여 다양한 수준의 얼굴 인식 데이터를 바탕으로 실험을 진행했습니다. Low-rank adaptation (LoRA) 기법을 사용하여 미세 조정한 결과, 제한된 학습 데이터에서도 성능이 향상되었음을 확인했습니다.

- **Performance Highlights**: 미세 조정을 적용한 DINOv2와 CLIP의 경우, Casia-Webface 데이터셋에서 각각 90.94%와 92.13%의 평균 정확도를 달성했으며, 이는 처음부터 모델을 훈련했을 때보다 높은 성능을 보였습니다.



### Show Me What and Where has Changed? Question Answering and Grounding for Remote Sensing Change Detection (https://arxiv.org/abs/2410.23828)
- **What's New**: 본 논문에서는 전통적인 원격 감지(change detection) 기술을 확장하여 사용자와의 상호 작용을 가능하게 하는 새로운 작업인 Change Detection Question Answering and Grounding (CDQAG)을 제안합니다. 이 작업은 사용자가 원하는 변화를 인식할 수 있도록 해줍니다.

- **Technical Details**: CDQAG는 정답과 함께 해당 픽셀 수준의 시각적인 증거를 제공합니다. 우리는 QAG-360K라 불리는 최초의 CDQAG 벤치마크 데이터세트를 구성했으며, 이는 36만 개 이상의 질문, 텍스트 정답, 고품질 비주얼 마스크의 삼중항을 포함합니다. VisTA라는 강력한 프레임워크를 개발하여 질문에 따른 복잡한 변화 이미지를 결합하여 인식하고, 텍스트와 비주얼의 정밀한 상관관계를 구축합니다.

- **Performance Highlights**: VisTA 방법은 고전적인 CDVQA 및 제안된 CDQAG 데이터세트에서 최첨단 결과를 달성했으며, CDQAG 모델 개선을 위한 귀중한 인사이트를 제공합니다.



### Parameter-Efficient Fine-Tuning Medical Multimodal Large Language Models for Medical Visual Grounding (https://arxiv.org/abs/2410.23822)
- **What's New**: 이번 연구에서는 의료 비주얼 그라운딩 과제를 위한 파라미터 효율적인 미세 조정 기법 (Parameter-efficient Fine-tuning)을 적용한 의료 다중 모드 대형 언어 모델(MLLM)을 제안합니다. 새로운 모델은 의료 이미지와 텍스트를 효과적으로 통합하여 임상적 결정을 지원하는데 도움을 줍니다.

- **Technical Details**: PFMVG(파라미터 효율적인 미세 조정의 의료 다중 모드 대형 언어 모델)는 ViT(Vision Transformer)를 이미지 인코딩에 사용하며, LLM(대형 언어 모델)을 통해 이미지-텍스트 임베딩을 처리합니다. 미세 조정 과정은 두 단계로 나누어져 있으며, 첫 번째 단계는 의료 이미지에 대한 캡션 생성, 두 번째 단계는 의료 비주얼 그라운딩입니다.

- **Performance Highlights**: MS-CXR 데이터셋에서 PFMVG는 기존의 기준을 뛰어넘는 최첨단 성능을 달성하였고, 특히 8가지 질병 카테고리에서 현저한 의성적 지수(IoU)와 Dice 점수를 기록하여 GPT-4v를 초월했습니다.



### Human Action Recognition (HAR) Using Skeleton-based Quantum Spatial Temporal Relative Transformer Network: ST-RTR (https://arxiv.org/abs/2410.23806)
- **What's New**: 본 논문에서는 인간 행동 인식( HAR )을 위한 새로운 양자 공간-시간 상대 변환기 모델인 ST-RTR를 개발하였습니다. 이 모델은 기존 ST-GCN의 한계를 극복하고자 하며, 효율적인 통신 및 데이터 전송을 위한 관절 및 릴레이 노드를 포함합니다.

- **Technical Details**: ST-RTR 모델은 짧은 범위의 상관관계만을 처리하는 기존 ST-GCN의 한계를 보완하여 장거리 상호 연결을 이해할 수 있도록 설계되었습니다. 이 모델은 양자 접근 방식을 활용하여 정보 전송 효율성을 높이며, 필드 내의 본질적인 공간 및 시간 골격 구조를 파악합니다. 이를 통해 스켈레톤 시퀀스로부터 공간적 및 시간적 패턴을 자동 학습할 수 있게 됩니다.

- **Performance Highlights**: 모델의 성능을 평가하기 위해 NTU RGB+D 60, NTU RGB+D 120, UAV-Human 데이터셋에서 실험을 실시한 결과, NTU RGB+D 60에서 CS와 CV가 각각 2.11% 및 1.45% 향상되었고, NTU RGB+D 120에서는 1.25% 및 1.05% 개선되었습니다. UAV-Human 데이터셋에서는 정확도가 2.54% 향상되었습니다. 이러한 결과는 ST-RTR 모델이 기존 ST-GCN 방법에 비해 행동 인식 능력을 크게 향상시킴을 보여줍니다.



### SOAR: Self-Occluded Avatar Recovery from a Single Video In the Wild (https://arxiv.org/abs/2410.23800)
- **What's New**: SOAR(Self-Occluded Avatar Recovery)는 부분 관찰에서 완전한 인체 복원을 가능하게 하는 혁신적인 방법입니다. 기존의 단일 비디오에서 인체를 복원하는 접근 방식이 전체 신체 가시성을 가정하는 것과 달리, SOAR는 신체가 일부 가려진 상황에서도 효과적으로 작동합니다.

- **Technical Details**: SOAR는 구조적 정규화 사전(structural normal prior)과 생성적 확산 사전(generative diffusion prior)을 활용하여 인체 복원 문제를 해결합니다. 구조적 정규화 사전은 갈라지는 서페이스(surfaces) 모델을 사용하여 인체를 표현하고, 생성적 확산 사전은 초기 복원을 수행하고 그것을 점수 증류(score distillation)를 사용하여 정제합니다. 이 시스템은 복원과 생성을 효과적으로 결합하며, Gaussian surfels을 통해 인체 모델을 동적으로 표현합니다.

- **Performance Highlights**: 다양한 벤치마크에서 SOAR의 성능은 기존의 최신 복원 및 생성 방법보다 우수하며, 동시 연구인 HAVE-FUN와도 비슷한 성능을 보입니다. 이 시스템은 실시간 렌더링 및 애니메이션에 사용할 수 있는 고해상도 아바타 복원이 가능합니다.



### EDT: An Efficient Diffusion Transformer Framework Inspired by Human-like Sketching (https://arxiv.org/abs/2410.23788)
Comments:
          Xinwang Chen and Ning Liu are with equal contributions. This paper has been accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 효율적인 Transformer 기반의 Diffusion Probabilistic Model (DPM)을 제안합니다. 이를 통해 기존 CNN 기반 DPM과 비교했을 때 낮은 계산 비용을 실현하고, 이미지 합성 성능에서 SOTA(State Of The Art)를 달성했습니다.

- **Technical Details**: Efficient Diffusion Transformer (EDT) 프레임워크는 가벼운 디자인의 diffusion 모델 아키텍처와 Attention Modulation Matrix (AMM)를 포함합니다. AMM은 인간의 스케치 방식을 모방하여 전역(global) 및 지역(local) attention을 조절함으로써 이미지 합성을 향상시킵니다. 또한, EDT에 맞춘 token relation-enhanced masking training 전략을 제안하여 학습 능력을 강화합니다.

- **Performance Highlights**: EDT는 기존 transformer 기반 diffusion models에 비해 3.93x, 2.84x, 1.92x의 속도 향상을 이루었으며, FID(Fréchet Inception Distance) 수치에서 더 낮은 값을 기록하여 우수한 성능을 보여줍니다.



### Video Token Merging for Long-form Video Understanding (https://arxiv.org/abs/2410.23782)
Comments:
          21 pages, NeurIPS 2024

- **What's New**: 논문에서는 비디오 이해를 위한 Transformer 기반 모델에서 긴 비디오 입력을 처리하기 위한 새로운 접근 방식으로 'Video Token Merging (VTM)' 기법을 제안했습니다. VTM은 비디오 토큰의 정보 밀도와 중요도를 고려하여 동적으로 토큰을 병합합니다.

- **Technical Details**: 우리는 여러 가지 VTM 전략을 탐구하여 영상 분류 작업에서 효과적인 메서드를 설계했습니다. 이 과정에서 단순한 이미지 토큰 병합 방법을 비디오 도메인으로 확장하고, 영역 집중 병합 및 동작 기반 병합 알고리즘을 도입했습니다. 최종적으로 saliency score을 예측하여 시공간 비주얼 토큰을 데이터 기반으로 병합할 수 있는 학습 가능한 VTM 알고리즘을 개발했습니다.

- **Performance Highlights**: 제안된 알고리즘은 LVU, COIN, Breakfast 데이터셋에서 기존 비디오 이해 방법보다 더 나은 성능을 보여줬습니다. 또한 메모리 비용을 84% 감소시키고 처리량을 약 6.89배 향상시켰습니다.



### Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map (https://arxiv.org/abs/2410.23780)
Comments:
          27 pages, 13 figures

- **What's New**: MapDR라는 새로운 데이터셋을 도입하여 교통 신호에서 주행 규칙을 추출하고 벡터화된 HD 맵과 연계하는 작업에 중점을 두었습니다.

- **Technical Details**: MapDR 데이터셋은 10,000개 이상의 주석이 달린 비디오 클립을 포함하고 있으며, 규제 지침을 정확히 해석하는 'Rule Extraction from Traffic Sign'와 이러한 규칙을 해당 차선과 맞추는 'Rule-Lane Correspondence Reasoning'이라는 두 가지 주요 서브 작업을 정의합니다.

- **Performance Highlights**: 자율주행 기술 발전을 위한 강력한 기준선을 제공하며, 안전한 자율 네비게이션 시스템 개발에 기여할 것으로 기대됩니다.



### In-Context LoRA for Diffusion Transformers (https://arxiv.org/abs/2410.23775)
Comments:
          Project page: this https URL

- **What's New**: 최근 연구에서 텍스트-이미지 모델이 다중 이미지 생성 작업에 효과적으로 적용될 수 있음을 발견했습니다. 특히, 기존의 diffusion transformers (DiTs)가 본질적으로 in-context 생성 능력을 지니고 있음을 가정하고, 최소한의 조정으로도 이러한 능력을 활용할 수 있음을 입증하였습니다.

- **Technical Details**: 본 연구에서는 기존의 DiTs를 활용하여 이미지 집중에서 작업별 LoRA (Low-Rank Adaptation) 튜닝을 통해 적은 데이터셋 (20∼100 샘플)으로도 높은 충실도의 이미지 세트를 생성할 수 있는 간단한 파이프라인을 제안하였습니다.

- **Performance Highlights**: 제안된 IC-LoRA 모델은 원래 DiT 모델의 구조 수정 없이도, 이미지 조건부 생성에 최적화된 성능을 보이며 다양한 작업에 대한 높은 품질의 결과를 생성합니다. 이 접근 방식은 적은 데이터 요구 사항과 폭넓은 적용 가능성을 제공하여 생성 커뮤니티와 디자이너, 예술가에게 유용한 도구로 작용할 수 있습니다.



### Open-Set 3D object detection in LiDAR data as an Out-of-Distribution problem (https://arxiv.org/abs/2410.23767)
- **What's New**: LiDAR 데이터에서 open-set 3D Object Detection 문제를 Out-Of-Distribution (OOD) 문제로 재정의하여, 전통적인 객체 탐지와 비교해 추가 정보를 제공합니다. 이 새로운 접근법을 통해 미지의(unknown) 객체를 효과적으로 탐지할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구에서는 LiDAR 스캔에 대해 N개의 bounding box를 출력하는 전통적인 3D 객체 탐지 방식을 다루고 있습니다. 그러나 Open-Set 3D Object Detection에서는 미지의 클래스에 대한 예측을 포함하여 OOD 탐지 및 자동 레이블링 기법을 활용합니다. 또한, 다양한 데이터를 생성하여 OOD 인식 3D 객체 탐지기를 훈련시키기 위한 방법도 평가합니다.

- **Performance Highlights**: 이 연구 결과, 두 단계의 OOD 방법들이 상당히 유망한 결과를 나타내었으며, 기존의 네트워크 성능을 유지하면서 미지의 객체 탐지 능력을 향상시켰습니다. 연구자들은 하이퍼파라미터 평가와 추가 데이터 생성 전략을 통해 더 강건한 3D 객체 탐지 시스템 개발에 기여하고 있습니다.



### Reverse Attitude Statistics Based Star Map Identification Method (https://arxiv.org/abs/2410.23758)
Comments:
          10 pages, 17figures, 4 tables, 4663 words, submitted to IEEE Sensors Journal

- **What's New**: 본 연구에서는 스타 맵 식별을 위한 새로운 방법론을 제안하였으며, 이는 자세 추정 결과를 활용하여 매칭을 보조하는 구조로 설계되었습니다. 이 방법은 처음으로 자세 해결과 매칭 프로세스를 결합하여 최종 매칭과 올바른 자세를 동시에 획득합니다.

- **Technical Details**: 제안된 방법은 역 자세 통계 기반의 프레임워크를 이용하여 별 쌍의 각 거리로부터 스타 맵을 식별합니다. 초기 매칭은 안정적인 각 거리 특성을 바탕으로 공간 해시 인덱싱 필터를 통해 이루어지며, 이 후 이중 벡터 자세 결정 알고리즘을 통해 가능성 있는 자세를 계산합니다. 마지막으로, 빈도 통계 필터링 방법을 적용하여 별 쌍의 정확한 매칭이 수행됩니다. 또한, 베이지안 최적화(Bayesian optimization)를 사용하여 소음의 영향을 받는 최적 매개변수를 찾아 알고리즘 퍼포먼스를 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 기존 상태와 비교했을 때, 식별률이 14.3% 이상 향상되었으며, 해결 시간은 28.5% 이상 단축되었습니다. 이 알고리즘은 시뮬레이션, 필드 테스트 및 궤도 실험을 통해 검증되었습니다.



### EXACFS -- A CIL Method to mitigate Catastrophic Forgetting (https://arxiv.org/abs/2410.23751)
- **What's New**: EXponentially Averaged Class-wise Feature Significance (EXACFS)는 지속적인 학습(continual learning) 문제를 해결하기 위해 제안된 새로운 방법으로, 학습 중 각 클래스에 대해 특징맵(feature map)의 중요성을 평가하고 이를 보호하는 방식으로 설계되었습니다.

- **Technical Details**: EXACFS는 손실 기울기(loss gradients)를 사용하여 각 학습된 클래스에 대한 모델 특징의 중요성을 추정하고, 점진적인 작업을 통해 중요성을 점진적으로 늙히며, 증류 손실(distillation loss)을 통해 중요한 특징을 유지합니다. 이러한 방식은 오래된 지식(과거 지식)을 기억하고 새로운 지식(신규 지식)을 학습하는 것을 효과적으로 균형 있게 합니다.

- **Performance Highlights**: CIFAR-100 및 ImageNet-100 데이터셋에 대한 폭넓은 실험을 통해 EXACFS는 과거 지식을 유지하면서 새로운 지식을 습득하는 데 있어 우수한 성능을 보임이 입증되었습니다.



### EchoNarrator: Generating natural text explanations for ejection fraction predictions (https://arxiv.org/abs/2410.23744)
Comments:
          accepted for MICCAI 2024

- **What's New**: 이번 논문에서는 좌심실(Left Ventricle, LV)에서의 박출률(Ejection Fraction, EF) 예측을 위한 자연어 설명(Natural Language Explanation, NLE) 모델을 처음으로 제안하였습니다. 이 모델은 여러 프레임에서 LV 윤곽을 추정하고, EF와 관련된 다양한 운동 및 형태 속성을 계산하여 큰 언어 모델(Large Language Model, LLM)에 입력하여 예측 결과를 설명하는 텍스트를 생성합니다.

- **Technical Details**: 이 모델은 비디오 인코더를 사용해 특성 표현을 추출하고, 다중 프레임 그래프 컨볼루션 네트워크(Graph Convolutional Network, GCN)를 통해 해부학적 키포인트를 식별합니다. EF 예측은 기하학적 속성과 함께 이뤄지며, 생성된 자연어 설명은 LLaMA 모델을 통해 이루어집니다. 이 접근법은 EF 예측의 정확성과 설명 가능성을 모두 높이는 데 기여합니다.

- **Performance Highlights**: 여기서 제안한 모델은 K-state-of-the-art 기술들과 비교하여 비슷한 수준의 EF 예측 값을 도출하였으며, 임상적으로 의미 있는 자연어 설명을 제공하는 것으로 나타났습니다. 본 연구는 심혈관 초음파 분석 및 해석 분야에서 중요한 기여를 합니다.



### Scaled Inverse Graphics: Efficiently Learning Large Sets of 3D Scenes (https://arxiv.org/abs/2410.23742)
- **What's New**: 이 논문에서는 "scaled inverse graphics"라는 새로운 프레임워크를 소개하여 대규모 장면 집합의 효율적인 학습을 목표로 합니다. 기존의 NeRF 개발에서 장면을 개별적으로 학습하는 것에 비해, 이번 연구는 장면 간 정보를 공유하여 학습 효율성을 극대화합니다.

- **Technical Details**: 본 연구는 두 단계로 운영됩니다: (i) 장면의 부분 집합에 대해 압축 모델을 학습하고, (ii) 더 작은 표현 공간에서 NeRF 모델을 학습하여 각 새로운 장면에 대한 최적화 공간을 줄입니다. Tri-Plane 표현 방식을 활용하여 장면의 표현을 압축하고 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 독립적으로 각 장면에 적용된 다른 방법들과 비교하여 낮은 훈련 시간과 메모리 사용량을 보여주었습니다. 또한, 상대적으로 높은 품질의 렌더링(NVS quality)을 유지하며 리소스 비용이 가장 낮다는 것을 입증했습니다.



### MoTaDual: Modality-Task Dual Alignment for Enhanced Zero-shot Composed Image Retrieva (https://arxiv.org/abs/2410.23736)
- **What's New**: 이 논문에서는 Zero-shot Composed Image Retrieval (ZS-CIR)와 관련된 두 가지 주요 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 효율성과 확장성을 고려한 두 단계의 프레임워크를 도입하여 모달리티와 작업 불일치를 해결합니다.

- **Technical Details**: 첫 번째 단계에서는 대규모 캡션 데이터셋에서 텍스추얼 인버전 네트워크를 학습하며, 두 번째 단계인 Modality-Task Dual Alignment (MoTaDual)에서는 대형 언어 모델(LLM)을 활용하여 트리플 데이터 생성과 프롬프트 학습을 통해 불일치를 줄입니다.

- **Performance Highlights**: MoTaDual은 네 개의 광범위한 ZS-CIR 벤치마크에서 최첨단 성능을 달성하며 낮은 훈련 시간과 계산 비용을 유지합니다.



### An Empirical Analysis of GPT-4V's Performance on Fashion Aesthetic Evaluation (https://arxiv.org/abs/2410.23730)
- **What's New**: 이번 연구에서는 패션 심미성 평가에서 GPT-4V의 제로샷(zero-shot) 성능을 처음으로 조사하였습니다. 이 모델은 인간의 판단과 잘 일치하는 예측을 보여주지만, 유사한 색상의 의상을 순위 매기는 데 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: 본 연구에서는 신뢰할 수 있는 금표준(gold-standard) 데이터 세트를 작성하고, 수백 명의 사람 주석자에 의해 주석된 이미지를 기반으로 GPT-4V의 성능을 평가하였습니다. 주석 과정에서는 OpenSkill 알고리즘을 활용하여 주석자의 결과를 바탕으로 이미지의 순위를 평가하였습니다.

- **Performance Highlights**: GPT-4V는 기존의 사용자 피드백 기반 모델인 View, Like, Like/View와 비교하여 이미지 분류 및 순위 매기기에서 우수한 성능을 나타냈습니다. 인간 주석자들이 생성한 OpenSkill 점수와의 상관관계에서도 높은 값을 기록하였습니다.



### GaussianMarker: Uncertainty-Aware Copyright Protection of 3D Gaussian Splatting (https://arxiv.org/abs/2410.23718)
- **What's New**: 본 논문에서는 3D Gaussian Splatting(3DGS) 모델에 대한 새로운 디지털 저작권 보호 방법을 제안합니다. 기존의 메쉬나 포인트 클라우드, 임플리시트(implicit) 라디언스 필드의 워터마킹 방법은 3DGS 모델에 직접 적용할 수 없습니다.

- **Technical Details**: 우리는 불확실성 기반의 워터마킹 방법을 적용하여 저작권 메시지를 3DGS 모델의 3D 가우시안 매개변수에 주입합니다. 이 방법은 로바이선 근사(Laplace approximation)를 사용해 3D 가우시안 매개변수에 추가할 수 있는 변화를 제어하고, 3D 메시지 디코더(PointNet 아키텍처 사용)와 2D 메시지 디코더(HiDDeN 기반)를 사용해 저작권 메시지를 추출합니다.

- **Performance Highlights**: Blender, LLFF 및 MipNeRF-360 데이터셋을 사용한 실험을 통해, 저작권 메시지 추출의 정확성과 보기 합성 품질에서 최첨단 성능을 입증하였습니다.



### XRDSLAM: A Flexible and Modular Framework for Deep Learning based SLAM (https://arxiv.org/abs/2410.23690)
- **What's New**: 이 논문에서는 유연한 SLAM 프레임워크인 XRDSLAM을 제안합니다. 이 프레임워크는 모듈화 코드 설계와 다중 프로세스 실행 메커니즘을 채택하여 통합된 데이터셋 관리, 3D 시각화, 알고리즘 구성 및 메트릭 평가와 같은 재사용 가능한 기본 모듈을 제공합니다.

- **Technical Details**: XRDSLAM은 모듈화 설계와 멀티 프로세스 기능을 통합해 SLAM 구성 요소를 유연하게 결합할 수 있습니다. 프레임워크는 데이터 입력/출력, 구성 구문 분석, 시각화, 결과 내보내기 기능 모듈을 제공하여 개발자가 다양한 알고리즘 모듈을 빠르게 결합하고 최적화할 수 있도록 지원합니다. 또한, SLAM 알고리즘의 통합 및 비교를 위해 여러 최신 알고리즘(NeRF 및 3DGS 기반 SLAM 포함)을 통합하여 유연성과 확장성을 증명합니다.

- **Performance Highlights**: XRDSLAM은 SLAM 알고리즘의 성능과 효율성을 비교 평가하여 개발자들이 효과적으로 SLAM 시스템을 구축할 수 있도록 합니다. 모든 코드, 구성 및 데이터는 오픈 소스로 제공되어 SLAM 기술의 연구 및 개발을 촉진하는 목적을 가지고 있습니다.



### Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey (https://arxiv.org/abs/2410.23687)
- **What's New**: 이번 논문은 Adversarial attacks에 대한 포괄적인 분석을 제공하며, 기존 연구들이 흔히 공격 분류에 치중하는 반면, 전통적인 공격과 Large Vision-Language Models (LVLM) 공격의 연결성과 차이를 강조합니다.

- **Technical Details**: 이 논문은 Adversariality, Transferability, Generalization 개념을 통합하고, 기존 방법들의 세부 평가 및 동기 중심의 공격 범주화 전략을 제안합니다.

- **Performance Highlights**: 전통적인 Adversarial attacks와 LVLM 공격의 관계를 간략히 설명하고, 미래 연구에 대한 실행 가능한 통찰을 제공하여 이 분야의 이해도를 높이고자 합니다.



### Web-Scale Visual Entity Recognition: An LLM-Driven Data Approach (https://arxiv.org/abs/2410.23676)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 대규모 데이터셋을 구축하기 위한 새로운 방법론을 제안합니다. 이는 다중 모달 대형 언어 모델(multimodal Large Language Model, LLM)을 활용하여 라벨 검증(label verification), 메타데이터 생성(metadata generation) 및 근거 설명(rationale explanation)을 수행하는 방식입니다.

- **Technical Details**: 저자들은 이미지 캡션과 위키피디아와 같은 외부 지식 출처에서 추가적인 맥락 정보를 활용하여 후보 엔티티 라벨에 대해 논리적으로 추론하도록 LLM을 유도합니다. 이를 통해 생성된 데이터셋은 고품질의 자동 큐레이션된 데이터를 통해 웹 규모의 비주얼 엔티티 인식 작업에서 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 자동으로 큐레이션된 데이터로 훈련된 모델들은 Open-domain Visual Entity Recognition (OVEN) 벤치마크에서 최고 성능을 달성했으며, 특히 OVEN 엔티티 분할에서 +6.9%의 성능 향상을 보였습니다. 이는 고품질 훈련 데이터의 중요성을 강조합니다.



### DIP: Diffusion Learning of Inconsistency Pattern for General DeepFake Detection (https://arxiv.org/abs/2410.23663)
Comments:
          13 pages, accepted with IEEE Trans. on Multimedia

- **What's New**: 이번 논문에서는 깊이 있는 비디오 탐지의 일반화 가능성을 향상시키기 위해 시간적 불일치성(temporal inconsistency) 패턴을 활용하는 새로운 프레임워크, 즉 Inconsistency Pattern의 Diffusion Learning(DIP)을 제안합니다. 특히 이 연구는 비디오의 수평 및 수직 방향의 불일치 성을 탐지하여 딥페이크(Deepfake) 동영상을 효과적으로 식별합니다.

- **Technical Details**: DIP 프레임워크는 공간-시간 인코더(spatiotemporal encoder)와 방향 기반 불일치 디코더(directional inconsistency decoder)를 포함합니다. 이 구조는 방향 인식을 통합한 Attention 메커니즘을 사용하고 불일치 확산(inconsistency diffusion)을 통해 잠재적 패턴을 탐색합니다. 또한, 공간-시간 불변 손실(SpatioTemporal Invariant Loss, STI Loss)을 도입하여 모델이 비본질적인 위조 아티팩트(artifact)에 과적합되는 것을 방지합니다.

- **Performance Highlights**: 다양한 공개 데이터셋을 통한 실험 결과, 제안된 DIP 프레임워크가 방향적 위조 단서를 효과적으로 포착하고, 상태-of-the-art 성능을 달성함을 입증했습니다. 이를 통해 딥페이크 탐지 과정에서 일반화 가능성과 강건성을 향상시킬 수 있는 방안을 모색하였습니다.



### GS-Blur: A 3D Scene-Based Dataset for Realistic Image Deblurring (https://arxiv.org/abs/2410.23658)
Comments:
          Accepted at NeurIPS 2024 Datasets & Benchmarks Track

- **What's New**: 이번 연구에서는 GS-Blur라는 새로운 데이터셋을 제안합니다. 기존의 blurry 및 sharp 이미지 쌍 데이터셋의 한계를 극복하기 위해, 다양한 카메라 경로를 활용하여 더욱 현실적이고 다양한 블러 유형을 포함하는 대규모 데이터셋을 구축하였습니다.

- **Technical Details**: GS-Blur는 3D Gaussian Splatting (3DGS) 기법을 활용하여 다중 뷰 이미지를 3D 씬으로 재구성한 후, 무작위로 생성된 운동 경로를 따라 카메라 뷰를 이동시켜 블러 이미지로 렌더링합니다. 이 과정에서 피사체의 이동을 시뮬레이션하며 서로 다른 블러 유형을 실현합니다.

- **Performance Highlights**: GS-Blur 데이터셋을 여러 가지 deblurring 방법과 함께 사용했을 때, 이전의 합성 또는 실제 블러 데이터셋에 비해 효과적으로 일반화되는 능력을 보여주었으며, 블러 제거 성능에서 상당한 개선을 보였습니다.



### Recovering Complete Actions for Cross-dataset Skeleton Action Recognition (https://arxiv.org/abs/2410.23641)
Comments:
          accepted by NeurIPS 2024

- **What's New**: 본 논문은 스켈레톤 기반 (skeleton-based) 행동 인식의 일반화 문제를 해결하기 위해 새로운 복구 및 재샘플링 증강(framework)을 제안합니다. 특히, 인간의 일상 행동이 서로 다른 데이터셋에서의 시간 불일치에 직면한다는 점을 관찰하고, 이를 기반으로 강력한 증강을 생성하는 방법을 소개합니다.

- **Technical Details**: 우리는 완전한 행동 사전(complete action prior)을 기반으로 행동 복구 단계에서 두 단계의 확률적 행동 완성을 통해 행동을 보완하고, 경계 자세(boundary pose)와 같은 전역 행동 패턴(global action patterns)을 캡처하기 위한 선형 시간 변환(linear temporal transforms)을 적용합니다. 또한, 전체 데이터셋에서 이들 경계 자세와 선형 변환을 효과적으로 학습하기 위해 클러스터링(clustering) 기법을 사용합니다.

- **Performance Highlights**: 제안된 접근법은 세 가지 스켈레톤 행동 데이터셋에서 교차 데이터셋 설정을 통해 검증되었으며, 다른 도메인 일반화 방법들보다 평균 정확도가 5% 향상되었습니다. 이는 스켈레톤 기반 행동 인식의 도메인 일반화 분야에서 상당한 성과를 보여줍니다.



### Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation (https://arxiv.org/abs/2410.23629)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: PiMForce라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 3D 손 자세 정보를 활용하여 팔꿈치 표면 전기 생리학(sEMG) 신호를 보강함으로써 손 압력 추정의 정확도를 향상시킵니다.

- **Technical Details**: PiMForce는 3D 손 자세와 sEMG 신호를 결합하여 전체 손 밀도를 정확히 측정하도록 설계되었습니다. 이 프레임워크는 다양한 손-물체 상호작용에서의 압력을 정확하게 추정할 수 있도록 돕습니다. 실험을 위해 21명의 참가자로부터 동기화된 데이터 세트를 구축하였으며, 이는 압력 장갑, sEMG 아르밴드, 마커리스 손가락 추적 모듈로 구성된 멀티모달 데이터 수집 시스템을 기반으로 합니다.

- **Performance Highlights**: 이 시스템은 기존의 sEMG 기반 방법이나 비전 기반 방법에 비해 월등히 향상된 성능을 보여줍니다. 3D 손 자세 정보와 sEMG 신호의 통합을 통해 더 정확하고 안정적인 손 압력 추정이 가능해졌습니다.



### On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection (https://arxiv.org/abs/2410.23623)
Comments:
          10 pages, 9 figures

- **What's New**: 이번 연구에서는 다중 모달 탐지(Multi-Modal Detection, MM-Det)라는 혁신적인 알고리즘을 제안하여 확산 모델(Diffusion Models)로 생성된 비디오를 효과적으로 탐지할 수 있는 새로운 방법을 제시합니다. 기존의 탐지 방법이 얼굴 조작에 국한되어 있었던 것과 달리, MM-Det은 다양한 의미론적 변화에 대응할 수 있는 능력을 갖추었습니다.

- **Technical Details**: MM-Det은 대형 다중 모달 모델(Large Multi-modal Models, LMMs)의 깊은 인지 및 종합 능력을 활용하며, 이들로부터 다중 모달 위조 표현(Multi-Modal Forgery Representation, MMFR)을 생성합니다. 또한, In-and-Across Frame Attention (IAFA) 메커니즘을 통해 시공간(spatio-temporal) 영역 내에서 특성을 증강하여 비디오 조작 탐지를 향상시킵니다. 이 연구는 Diffusion Video Forensics (DVF)라는 포괄적인 데이터셋을 구축하였습니다.

- **Performance Highlights**: MM-Det은 DVF 데이터셋에서 최첨단 성능을 달성하였으며, 알고리즘의 효과를 입증하는 자세한 분석을 제공합니다. 이 연구는 다중 모달 표현이 위조 탐지에 있어 어떻게 효과적인지를 밝힘으로써 향후 멀티미디어 포렌식(Multimedia Forensics) 연구에 대한 흥미로운 기회를 제공합니다.



### Context-Aware Token Selection and Packing for Enhanced Vision Transformer (https://arxiv.org/abs/2410.23608)
- **What's New**: 최근 컴퓨터 비전의 여러 작업에서 비전 트랜스포머(vision transformers, ViTs)의 장거리 주의(attention) 메커니즘이 성과를 크게 향상시켰습니다. 하지만 기존의 self-attention 메커니즘은 유용한 토큰과 비유용한 토큰을 모두 처리해 비효율성과 부정확성 문제에 직면해 있습니다. 이를 해결하기 위해 Select and Pack Attention(SPA)라는 새로운 알고리즘이 제안되었습니다.

- **Technical Details**: SPA는 선택 레이블에 의해 감독되는 저비용 게이팅(layer) 레이어를 사용하여 유용한 토큰을 동적으로 선택하고 이들을 새로운 배치에 패킹합니다. 이렇게 선택된 토큰들은 병렬 GPU 배치 교육 및 추론에서 사용됩니다. SPA는 Swin Transformer의 윈도우 기반 주의(attention)와 통합할 수 있으며, 패키지 간의 정보 손실을 방지하기 위해 특징 맵을 이동시키는 방법을 사용합니다.

- **Performance Highlights**: SPT(Select and Pack Transformer)는 다양한 이미지 표현을 생성하여 여러 컴퓨터 비전 작업에서 우수한 성능을 발휘합니다. 실제로 객체 탐지에서는 0.6 mAP 향상, 멀티 레이블 분류에서 0.24 mAP 증가, 이미지 분류에서 7.05의 Top-1 정확도 상승, 그리고 계산 비용에서 16.4%의 절감을 보여주었습니다.



### Using Multimodal Deep Neural Networks to Disentangle Language from Visual Aesthetics (https://arxiv.org/abs/2410.23603)
- **What's New**: 이 논문은 아름다움에 대한 인간의 평가를 예측하기 위해 단일 모드 비전, 단일 모드 언어 및 다중 모드 심층 신경망(DNN) 모델의 학습된 표현에 대한 선형 디코딩(linear decoding)을 사용하여 감정적이고 미적 경험에서 인식과 언어를 분리하는 도전을 다룹니다.

- **Technical Details**: 연구에 사용된 주요 데이터셋은 OASIS dataset로, 900개의 이미지로 구성되어 있으며, 각 이미지는 평균 100~110명의 평가자에 의해 평가된 점수를 포함합니다. 연구는 단일 모드 및 다중 모드 심층 신경망 모델에서 추출한 특징을 사용하여 인간의 미적 평점을 예측하는 방법론을 디테일하게 설명합니다. 특히, unimodal vision model(예: SimCLR)은 아름다움 평가에서 설명 가능한 분산의 대부분을 차지하며, 언어 정렬 비전 모델(예: SLIP)은 unimodal vision 모델에 비해 작은 이득만을 보여줍니다.

- **Performance Highlights**: 총 설명 가능한 분산의 75%까지 예측 가능한 unimodal vision 모델이 확인되었으며, CLIP 모델은 경우에 따라 87%의 설명 가능한 분산을 설명합니다. 하지만 SLIP 모델의 결과를 통해 언어의 영향을 조절하면서 비전과 언어의 관계를 더 깊이 이해할 수 있는 가능성을 제시합니다.



### Using Structural Similarity and Kolmogorov-Arnold Networks for Anatomical Embedding of 3-hinge Gyrus (https://arxiv.org/abs/2410.23598)
- **What's New**: 이 연구에서는 3-hinge gyrus(3HG)의 해부학적 특징을 임베딩(embedding)하는 새로운 자기 지도 학습 프레임워크(self-supervised framework)를 제안합니다. 이 프레임워크는 여러 개의 뇌 간의 3HG 간의 대응을 구축하는 데 중점을 두고 있으며, 기존 방법의 한계를 극복하는 데 목표를 두고 있습니다.

- **Technical Details**: 제안된 방법에서는 Kolmogorov-Arnold Networks (KAN)를 사용하여 3HG의 해부학적 특징을 인코딩합니다. 각 3HG 노드의 구조적 유사성(structural similarity)을 높이고, 선택적 재구성 손실(selective reconstruction loss) 함수를 도입하여 비제로(non-zero) 요소의 재구성 오류를 벌점으로 부과함으로써 임베딩 벡터의 표현 능력을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 복잡한 피질 구조에서 교차 주제 간 접합점을 효과적으로 설정할 수 있음을 보여주었으며, 3HG의 공통성과 변동성을 유지하면서 일반적인 3HG 네트워크와 인구 폭넓은 분석 구성에 중요한 기여를 할 수 있을 것으로 기대됩니다.



### Phrase Decoupling Cross-Modal Hierarchical Matching and Progressive Position Correction for Visual Grounding (https://arxiv.org/abs/2410.23570)
Comments:
          This work has been accepted by TMM

- **What's New**: 이번 논문은 비주얼 그라운딩(visual grounding) 분야에서 텍스트와 이미지 간의 다계층(hierarchical) 상관관계의 중요성을 강조하며, 새로운 방법론인 Phrase Decoupling Cross-Modal Hierarchical Matching과 Progressive Position Correction을 제안하고 있습니다.

- **Technical Details**: 제안된 방법은 문장 구문을 분리하여 마스크(mask)를 생성하고, 이를 통해 텍스트와 이미지 간의 계층적 매칭 메커니즘을 구축합니다. 이 과정에서 텍스트의 계층적 세부 사항이 이미지의 지역적 특징과 어떻게 매칭되는지를 반영하여 더 정교한 개체 위치 식별을 지원합니다.

- **Performance Highlights**: 세 가지 참조 표현 데이터셋(referring expression datasets)에서 실험을 통해 제안된 방법이 최신 기법(state-of-the-art methods)과 비교하여 우수한 성능을 보였음을 입증하였으며, 계산 효율성 측면에서도 장점을 나타냈습니다.



### Language-guided Hierarchical Fine-grained Image Forgery Detection and Localization (https://arxiv.org/abs/2410.23556)
Comments:
          Accepted by IJCV2024. arXiv admin note: substantial text overlap with arXiv:2303.17111

- **What's New**: 본 연구에서는 이미지 위조 탐지 및 위치 선정(Image Forgery Detection and Localization, IFDL)을 위한 새로운 계층적 세분화 모델인 HiFi-Net++를 제안합니다. 이 모델은 이미지 조작 특성을 다중 레이블로 표현하고, 계층적 의존성을 통해 세부적인 분류를 수행하여 위조 탐지의 정확도를 높입니다.

- **Technical Details**: HiFi-Net++는 네 가지 주요 구성 요소로 이루어져 있습니다: 다중 분기 특징 추출기(multi-branch feature extractor), 언어 기반 위조 위치 선정 향상기(language-guided forgery localization enhancer), 분류(classification) 및 위치 선정 모듈(localization modules). 각 분기는 특정 레벨의 위조 특성을 분류하며, LFLE는 이미지를 다중 모달 입력으로 사용하여 시각적 임베딩(visual embedding) 및 조작 점수 맵(manipulation score maps)을 생성합니다.

- **Performance Highlights**: HiFi-Net++는 IFDL 및 위조 특성 분류 작업을 위한 다양한 벤치마크에서 효과적으로 성능을 입증하며, 연구를 지원하기 위한 계층적 세분화 데이터셋도 구성하였습니다. 이 알고리즘은 위조 탐지 분야의 발전에 기여할 것으로 예상됩니다.



### There and Back Again: On the relation between noises, images, and their inversions in diffusion models (https://arxiv.org/abs/2410.23530)
- **What's New**: 이번 논문에서는 Denoising Diffusion Probabilistic Models (DDPMs)가 새로운 이미지를 생성하는 데 뛰어난 성과를 내고 있지만, 의미 있는 latent space가 부족하다는 문제를 다루고 있습니다. 특히, DDIM inversion 기법의 정확성을 분석하고, 초기 Gaussian noise와 생성된 샘플 간의 관계를 탐구합니다.

- **Technical Details**: DDPMs와 DDIM을 기반으로 한 연구에서, DDIM inversion이 초기 Gaussian noise를 reverse하려 할 때 발생하는 불일치 문제를 설명합니다. 이 연구는 초기 노이즈와 생성된 샘플 간의 관계를 유클리드 거리(Euclidean distance)로 정확히 할당할 수 있음을 보여주며, 훈련 초기 단계에서 noise와 generations 사의 매핑이 어떻게 형성되는지를 분석합니다.

- **Performance Highlights**: 이 연구의 주요 기여는 reverse DDIM이 표준 다변량 Gaussian이 아닌 latent representation을 생성하여 이론과 실제 간의 차이를 만들어낸다는 점입니다. 또한 모델의 생성 능력을 개선하더라도 reverse DDIM의 정확도를 개선하지 못한다는 점을 강조합니다.



### LBurst: Learning-Based Robotic Burst Feature Extraction for 3D Reconstruction in Low Ligh (https://arxiv.org/abs/2410.23522)
Comments:
          7 pages, 8 figures, 3 tables, for associated project page, see this https URL

- **What's New**: 본 논문에서는 저조도( low-light ) 환경에서 드론의 3D 재구성을 개선하는 기계 학습 아키텍처를 제안합니다. 기존 기법과 달리, 저신호 대 잡음비( signal-to-noise ratio ) 이미지에서 진정한 특징을 검출하고 설명하기 위한 방법을 통해 시각적 재구성을 강화하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 Robotic Burst Imaging 기술을 채택하여 여러 장의 이미지를 순차적으로 캡처하고 이를 통해 생성된 하나의 이미지에서 고신뢰도 특징을 찾아내는 방식을 사용합니다. 제안된 LBurst 모델은 키포인트( keypoint )를 저조도 로봇 버스트에서 탐지하고 설명합니다. 이 모델은 L2-Net과 같은 완전 합성곱 네트워크( fully-convolutional network )를 기반으로 하며, 멀티 스케일 특징을 학습하고 탐지하는 구조로 개발되었습니다.

- **Performance Highlights**: 제안된 접근법은 mililux( 밀리룩스 ) 조도에서 드론 이미지를 통한 3D 재구성 성능에서 현저한 개선을 나타냅니다. 단일 이미지 기반 특징 추출기 및 Robotic Burst Feature Finder와 비교했을 때, 저신호 대 잡음비 환경에서 더 나은 품질의 특징을 찾아 더 완벽한 3D 모델을 생성하는 데 성공하였습니다.



### Causality-Driven Audits of Model Robustness (https://arxiv.org/abs/2410.23494)
- **What's New**: 이번 연구는 DNN(Deep Neural Networks)의 강건성 감사(robustness audit) 방법에 대한 새로운 접근 방식을 제시하며, 인과 추론(causal inference)을 활용하여 복잡한 왜곡의 원인을 정량적으로 분석합니다. 이는 환경, 센서, 처리 파이프라인의 다양한 요인이 상호 작용하여 발생하는 이미지 왜곡에 대한 DNN의 민감도를 측정하는 데 도움을 줍니다.

- **Technical Details**: 본 연구는 이미지 생성 과정(image generating process)에서의 인과 모델(causal models)을 사용하여 DNN의 강건성을 평가하는 새로운 방법론을 개발합니다. 이를 위해 구조적 인과 모델(Structural Causal Model, SCM)을 통해 이미지 품질에 영향을 미치는 주요 요인들을 모델링하고, 이를 통해 DNN 민감도 평가를 수행합니다. 이 방법은 다양한 촬영 조건(resulting conditions)에서의 경험적 데이터를 활용하여 인과 효과(causal effects)를 신뢰성 있게 추정합니다.

- **Performance Highlights**: 실험 결과, 제안된 인과적 강건성 감사(Causality-Driven Robustness Audit, CDRA) 방법은 다양한 비전 태스크(vision tasks)에서 DNN의 성능에 미치는 개별 요인의 인과적 영향을 효과적으로 추정하는 것으로 나타났습니다. 이는 DNN의 성능 저하를 예측하고, 현업에서의 예기치 않은 DNN 실패 위험을 감소시키는 데 기여할 수 있습니다.



### EchoFM: Foundation Model for Generalizable Echocardiogram Analysis (https://arxiv.org/abs/2410.23413)
- **What's New**: 이번 연구에서는 심장 초음파 영상을 분석하고 표현하기 위해 특별히 설계된 기초 모델 EchoFM을 소개합니다. EchoFM은 자가 감독 학습(self-supervised learning) 프레임워크를 제안하여 시공간(spatio-temporal) 변동성을 효과적으로 캡처합니다.

- **Technical Details**: EchoFM은 26개의 스캔 뷰와 20만 개 이상의 초음파 영상을 포함하는 대규모 데이터셋에서 사전 학습(pre-training)되었습니다. 이 모델은 Masked Autoencoder(MAE) 프레임워크 내에서 주기적 대비 손실(periodic contrastive loss)을 사용하는 새로운 접근 방식을 도입하여, 심장 주기의 동일한 단계에서 유사한 표현을 생성하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, EchoFM은 전문 초음파 방법 및 기존의 자가 감독 모델 등 최첨단 방법들보다 모든 다운스트림 작업에서 뛰어난 성능을 보였습니다. 이는 EchoFM이 다양한 실서식에서의 일반화(generalization) 능력을 가지고 있음을 나타냅니다.



### TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes (https://arxiv.org/abs/2410.23409)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 논문에서는 Neural Temporal Point Process (TPP)를 기반으로 한 새로운 스캔패스 모델인 TPP-Gaze를 제안합니다. 이 모델은 고정점의 위치와 지속 시간의 동적 변화를 동시에 학습하며, 딥러닝 방법론과 포인트 프로세스 이론을 통합합니다.

- **Technical Details**: TPP-Gaze는 시각 자극에 대한 고정점의 시퀀스를 모델링하는 데 있어서 Neural TPP의 개념을 적용합니다. 이 모델은 불규칙한 간격으로 발생하는 사건들의 연속적인 시간을 예측하며, 기존 모델들이 간과한 시간 정보를 효과적으로 학습합니다. 이를 통해 스캔패스에서 고정점 간의 전환 (saccade) 및 그 지속 시간을 적절히 예측할 수 있습니다.

- **Performance Highlights**: 다섯 개의 공개 데이터셋에서 실시한 실험 결과, TPP-Gaze 모델이 최신의 최첨단 접근법들에 비해 전반적으로 우수한 성능을 보임을 확인하였습니다. 이 모델은 스캔패스 예측 문제를 처리하기 위한 새로운 기준을 제공합니다.



### Multilingual Vision-Language Pre-training for the Remote Sensing Domain (https://arxiv.org/abs/2410.23370)
Comments:
          Accepted at ACM SIGSPATIAL 2024 - Research Papers

- **What's New**: 본 연구는 원거리 감지(Remote Sensing) 분야에서 멀티 언어(multi-language) CLIP 모델의 미세 조정(fine-tuning)과 각 이미지의 지역(local) 및 전역(global) 표현을 정렬하는 자기 지도(self-supervised) 방법을 기반으로 한 새로운 비전 언어 모델을 재진술합니다. 이를 통해 원거리 감지 이미지와 영어 캡션으로 구성된 큰 데이터셋을 구축하고, 9개 추가 언어로 자동 기계 번역을 수행하였습니다.

- **Technical Details**: 이 연구에서 제안한 RS-M-CLIP 모델은 한 및 다국어(multi-language) 입력을 처리할 수 있는 CLIP 모델입니다. 대량 데이터셋은 영어 캡션이 붙은 원거리 감지 이미지를 결합하여 구축하였으며, 자동 기계 번역을 통해 9개 추가 언어로 데이터 증강(data augmentation)을 하였습니다. 미세 조정 과정에서 지역(local) 및 전역(global) 표현 정렬 방식을 사용하여 모델을 훈련했습니다.

- **Performance Highlights**: RS-M-CLIP 모델은 cross-modal 및 multilingual 이미지-텍스트 검색 혹은 zero-shot 이미지 분류 등 다양한 비전 언어 작업에서 최첨단 성능을 기록합니다. 이 모델은 다양한 언어에서 영어 성능을 개선하는 데에도 도움이 되는 번역 데이터를 효과적으로 사용하였습니다.



### Domain-decomposed image classification algorithms using linear discriminant analysis and convolutional neural networks (https://arxiv.org/abs/2410.23359)
- **What's New**: 이 연구에서는 이미지 분류 문제를 위해 두 가지 도메인 분해(CNN) 모델을 비교하였으며, 이들 모델은 전이 학습(transfer learning) 전략과 결합되어 있습니다. 새로운 분해된 LDA(Localization-based Linear Discriminant Analysis) 전략도 제안되었습니다.

- **Technical Details**: 연구는 CNN과 LDA를 이미지 분류 문제에 적용하는 것을 중점적으로 다루며, CNN은 지역적 특징을 추출하는 합성곱층(convolutional layers)과 풀링층(pooling layers)으로 구성되어 있습니다. LDA는 데이터셋에서 가장 차별적인 특징을 추출하고 차원 축소를 통해 분류 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 도메인 분해된 CNN 모델은 전이 학습을 사용하지 않은 글로벌 CNN 모델에 비해 향상된 분류 정확도를 보여주었고, 분해된 LDA 접근 방식도 전체 입력 데이터에 적용된 글로벌 LDA와 비교했을 때 높은 분류 정확도를 기록하였습니다.



### MoLE: Enhancing Human-centric Text-to-image Diffusion via Mixture of Low-rank Experts (https://arxiv.org/abs/2410.23332)
Comments:
          Published at NeurIPS 2024

- **What's New**: 이번 연구에서는 사람 중심의 텍스트-이미지 생성에 초점을 두고, 인간의 얼굴과 손을 포함한 이미지를 보다 자연스럽게 생성하기 위한 두 가지 방법론을 제안합니다. 첫째, 100만 개 이상의 고품질 인간 중심 이미지를 포함하는 데이터셋을 수집하였고, 둘째, ‘Mixture of Low-rank Experts (MoLE)’라는 새로운 방법론을 도입하였습니다.

- **Technical Details**: 연구진은 다양한 인종, 제스처 및 활동을 포함하여 분산 모델의 성능 향상을 위한 충분한 지식을 제공하는 고품질의 ‘human-centric dataset’(인간 중심 데이터셋)을 구성하고자 하였습니다. MoLE는 각각의 데이터셋에 대해 훈련된 저계수 모듈을 전문가로 보고, 이들 모듈을 소프트 어사인먼트를 통해 선택적으로 활성화하는 방식으로 작동합니다.

- **Performance Highlights**: MoLE는 COCO Caption 및 DiffusionDB를 이용한 두 가지 평가 벤치마크에서 여러 메트릭과 인간 평가를 통해 기존의 최첨단 모델들에 비해 우수성을 입증하였습니다. MoLE는 SD v1.5, v2.1 및 XL을 통해 일관되게 성능이 향상됨을 보였습니다.



### CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP (https://arxiv.org/abs/2410.23330)
- **What's New**: 이번 연구에서는 CLIP 모델에서 특정 데이터를 효과적으로 제거하는 새로운 접근 방식인 CLIPErase를 제안합니다. 이는 기존의 unimodal 모델 기법을 넘어 multimodal 환경에서의 machine unlearning 문제에 초점을 맞춥니다.

- **Technical Details**: CLIPErase는 세 가지 주요 모듈로 구성됩니다: (1) Forgetting Module은 지우기 원하는 데이터와의 연결을 분해합니다. (2) Retention Module은 남겨두는 데이터에서 모델 성능이 유지되도록 보장합니다. (3) Consistency Module은 원본 모델과의 일관성을 유지합니다. 이 세 가지 모듈에서 파생된 손실 함수는 공동으로 최소화되어 unlearning을 수행합니다.

- **Performance Highlights**: CIFAR-100과 Flickr30K 데이터셋에 대한 실험 결과, CLIPErase는 다양한 zero-shot 작업에서 지정된 연결을 효과적으로 잊어버리면서도 남겨두는 세트에서의 모델 성능을 유지함을 증명했습니다.



### VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration (https://arxiv.org/abs/2410.23317)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)의 추론 속도를 향상시키기 위해 VL-Cache라는 새로운 Key-Value (KV) 캐시 압축 방법을 제안합니다. 기존의 KV 캐시 압축 방식이 VLM에 최적화되지 않았음을 보여줍니다.

- **Technical Details**: VL-Cache는 VLM의 주의 (attention) 스파시티 패턴을 분석하여 visual 토큰과 언어 토큰을 구분합니다. 레이어 별로 스파시티를 고려하여 KV 캐시 예산을 할당하며, 모달리티 인식 토큰 스코어링 정책을 개발하여 중요한 토큰의 중요도를 평가합니다.

- **Performance Highlights**: 실험 결과, KV 캐시의 10%만 유지해도 전체 작업 정확도를 98% 이상 보존하며, 100개의 토큰 생성을 위한 종단 간 지연 시간을 최대 2.33배 단축시켰습니다. GPU 메모리 사용량을 90% 감소시킴에 따라, 높은 동시성을 지원할 수 있습니다.



### EgoMimic: Scaling Imitation Learning via Egocentric Video (https://arxiv.org/abs/2410.24221)
- **What's New**: EgoMimic은 인간의 자아 중심 비디오와 3D 손 추적 데이터를 기반으로 로봇 조작을 학습하는 새로운 프레임워크입니다. 이 시스템은 수동 데이터 수집을 통해 대규모의 다양한 조작 데이터를 축적할 수 있게 해줍니다.

- **Technical Details**: EgoMimic은 Project Aria 안경을 사용하여 인간의 자아 중심 데이터(egocentric human data)를 수집하고, 저비용의 양손 조작기를 사용하여 인간 데이터와의 기하학적 차이를 최소화합니다. 이를 위해 교차 도메인 데이터 정렬 기법과 함께 공통 비전 인코더 및 정책 네트워크를 사용하여 인간과 로봇 데이터를 통합하여 학습합니다.

- **Performance Highlights**: EgoMimic은 연속 객체 집기, 옷 개기, 장보기와 같은 다양한 긴 작업(task)에서 성능을 크게 개선하였으며, 최대 200%의 성능 향상을 보였습니다. 또한, 새로운 객체 및 장면에 대한 일반화 능력이 탁월합니다.



### Teaching Embodied Reinforcement Learning Agents: Informativeness and Diversity of Language Us (https://arxiv.org/abs/2410.24218)
Comments:
          EMNLP 2024 Main. Project website: this https URL

- **What's New**: 이 연구는 강화학습을 위한 체화 에이전트의 언어 입력의 다양성과 정보성의 영향을 처음으로 상세히 조사하였습니다.

- **Technical Details**: Decision Transformer (DT) 모델을 기반으로 하는 Language-Teachable Decision Transformer (LTDT) 아키텍처를 제안했습니다. 이 시스템은 언어 피드백을 통해 에이전트의 학습 능력에 큰 영향을 미치며, GPT-4를 활용해 더 자연스럽고 풍부한 언어 표현을 생성합니다.

- **Performance Highlights**: 이 연구 결과, 다양한 언어 피드백을 사용하여 훈련된 에이전트는 일반화가 향상되고 새로운 작업에 신속하게 적응하는 능력이 증가하였으며, 언어 없이 훈련된 에이전트보다 평균 20포인트 이상의 성능 향상을 기록했습니다.



### ARQ: A Mixed-Precision Quantization Framework for Accurate and Certifiably Robust DNNs (https://arxiv.org/abs/2410.24214)
- **What's New**: 새로운 Mixed Precision Quantization 방법인 ARQ는 neural network의 정확도를 보존하면서 인증된 강인성을 유지한다. 기존의 quantization 방법과 달리 ARQ는 강인성 인증 기술을 quantization 과정에 포함시킨다.

- **Technical Details**: ARQ는 강화 학습(reinforcement learning) 기법을 활용하여 quantization 정책을 탐색하고, 난수 스무딩(randomized smoothing)을 이용해 검색 과정을 최적화한다. 다양한 DNN 아키텍처에서 ARQ는 ResNet-20, ResNet-50, MobileNetV2를 포함하여 성능 비교를 진행했다.

- **Performance Highlights**: ARQ는 모든 벤치마크와 입력 노이즈 수준에서 기존 최첨단 quantization 기법들보다 일관되게 뛰어난 성능을 보여주었다. ARQ의 양자화된 네트워크는 원래의 DNN과 유사한 성능을 보여주며, 연산량은 단 1.5%에 불과하다.



### DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning (https://arxiv.org/abs/2410.24185)
Comments:
          Project website: this https URL

- **What's New**: 이 연구에서는 humanoid 로봇의 다수의 조작 기술을 자가 학습할 수 있도록 도와주는 대규모 자동 데이터 생성 시스템, DexMimicGen을 소개합니다.

- **Technical Details**: DexMimicGen은 소수의 인간 시연에서 궤적을 합성하여 bimanual dexterous manipulation을 위한 훈련 데이터를 자동으로 생성합니다. 이 시스템은 MimicGen의 개념을 바탕으로 다수의 서브태스크를 유연하게 분할하여, 각 팔이 독립적으로 작업을 수행하면서도 필요한 협조 단계를 수용하도록 설계되었습니다.

- **Performance Highlights**: DexMimicGen을 통해 약 60개의 소스 인간 시연에서 21,000개의 데모를 생성하였고, 이를 실제 캔 정렬 작업에 적용하여 90%의 성공률을 보였습니다.



### HoloChrome: Polychromatic Illumination for Speckle Reduction in Holographic Near-Eye Displays (https://arxiv.org/abs/2410.24144)
- **What's New**: 이 논문에서는 HoloChrome이라는 폴리크로매틱(Polychromatic) 홀로그램 디스플레이 프레임워크를 제안하여, 스펙클 노이즈(speckle noise)를 감소시키는 방안을 제시합니다. HoloChrome은 초고속 및 파장 조정이 가능한 레이저와 이중 공간광 변조기(Dual SLM) 아키텍처를 활용하여 가시 스펙트럼 전역에서 다양한 파장을 동시에 통해 스펙클 패턴을 독립적으로 조작합니다.

- **Technical Details**: HoloChrome 시스템은 초연속 레이저(Super-continuum laser)를 사용하여 넓은 범위의 파장을 생성하고, 이 중 필요한 파장을 필터링하여 혼합합니다. 이중 SLM 아키텍처를 통해 서로 다른 파장에서 생성된 스펙클 패턴의 상관관계를 완화하여, 효과적으로 스펙클 노이즈를 감소시키는 방법을 제시합니다. 또한 하이퍼스펙트럴 보정 모델(Hyperspectral calibration model)을 개발하여, 모든 가시 파장에서 고정밀 보정을 수행합니다.

- **Performance Highlights**: HoloChrome은 기존의 RGB 색상 홀로그램 아키텍처에 비해 스펙클 노이즈를 시뮬레이션에서는 최대 5-6dB, 실제 실험에서는 3-4dB까지 감소시키며, 색 영역(Color Gamut)을 확장해 가시적인 성능을 크게 향상시킵니다. 또한, HoloChrome 프로토타입은 전통적인 홀로그램 디스플레이에 비해 색 재현 및 스펙클 대비에서 개선된 성능을 보여줍니다.



### Parameter choices in HaarPSI for IQA with medical images (https://arxiv.org/abs/2410.24098)
Comments:
          5 pages, 3 figures, 2 tables

- **What's New**: 이 연구는 머신러닝 모델 개발 시 중요한 요소인 이미지 품질 평가(measures of Image Quality Assessment, IQA)에서, 자연 이미지에 최적화된 기존 측정 기준을 의료 이미지에 맞춰 최적화하는 새로운 접근법을 제시합니다. 특히 HaarPSI(IQA 지표 중 하나)를 의료 이미지 데이터셋에 대해 최적화하여 성능을 개선한 HaarPSI_MED를 소개합니다.

- **Technical Details**: HaarPSI는 Haar wavelet을 기반으로 한 FR-IQA 측정치이며, 두 이미지의 주파수 분해를 비교합니다. 연구에서는 두 가지 조정 가능한 매개변수를 최적화하는데, 이를 의료 이미지 데이터셋인 photoacoustic(포토아쿠스틱) 및 chest X-Ray(흉부 X선)에서 분석하였습니다. 각 의료 데이터셋에서 최적화된 매개변수가 유사하며, 두 데이터셋에 대해 조정된 설정을 통해 성능이 크게 향상되었다고 보고합니다.

- **Performance Highlights**: HaarPSI_MED로 최적화된 결과가 기존 PSNR, SSIM 등 자연 이미지 기반 IQA 측정치와 비교하여 의료 이미지에서 현저한 성능 개선을 보였습니다. 본 연구는 의료 이미지에 적합한 IQA 측정치를 사용하는 것이 중요하며, 이를 통해 일반화 가능한 방식으로 더 구체적인 작업 기반 측정의 유용성을 강조합니다.



### Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structur (https://arxiv.org/abs/2410.24060)
- **What's New**: 이번 연구에서는 diffusion models의 일반화 가능성을 살펴보며, 훈련된 score function의 숨겨진 특성에 대해 분석합니다. 특히, 기억화(memorization)에서 일반화(generalization)로의 전환 과정에서 비선형 diffusion denoiser의 선형성이 증가하는 현상을 발견하였습니다.

- **Technical Details**: 우리는 diffusion models의 출력이 multivariate Gaussian distribution의 empirical mean과 covariance에 의해 최적화된 denoiser에 근사적임을 보여주었습니다. 이는 이러한 모델들이 훈련 데이터의 Gaussian 구조(공분산 정보)를 학습하려는 귀납적 편향(inductive bias)을 가지고 있음을 의미합니다.

- **Performance Highlights**: 연구 결과, diffusion models는 훈련 데이터의 크기에 비해 모델 용량이 상대적으로 작을 때 강력한 일반화를 나타내며, 이 현상은 초기 훈련 단계에서도 나타납니다. 이렇게 발견된 귀납적 편향은 최근 관찰된 강력한 일반화 현상에 대한 이해를 돕는 중요한 통찰을 제공합니다.



### Advanced Predictive Quality Assessment for Ultrasonic Additive Manufacturing with Deep Learning Mod (https://arxiv.org/abs/2410.24055)
- **What's New**: 이번 연구에서는 Ultrasonic Additive Manufacturing(UAM) 과정에서 발생할 수 있는 레이어 간 결함을 모니터링하기 위해 깊이 학습 기반의 Convolutional Neural Networks(CNNs)를 이용한 방법을 개발했습니다.

- **Technical Details**: 각기 다른 처리 조건에서 열 이미지(thermal images)를 사용하여 온도 센서(thermocouples)가 삽입된 샘플과 그렇지 않은 샘플을 분류하는 CNN 모델을 평가했습니다. 다섯 가지 전력 수준(300W, 600W, 900W, 1200W, 1500W)에서 실험이 진행되었으며, 총 네 가지의 CNN 분류 모델이 생성되었습니다.

- **Performance Highlights**: 모델들은 결합된 기준 이미지(baseline)와 온도 센서 이미지에서 98.29%의 정확도를 달성했으며, 기준 이미지에서는 97.10%, 온도 센서 이미지에서는 97.43%, 그리고 두 경우의 조합에서도 97.27%의 정확도를 보였습니다. 이러한 결과는 UAM 프로세스의 조건을 식별하고 분류하는 데 있어 시스템의 효과성을 입증합니다.



### Deep Learning with HM-VGG: AI Strategies for Multi-modal Image Analysis (https://arxiv.org/abs/2410.24046)
- **What's New**: 이 연구는 녹내장 조기의 진단을 위한 혁신적인 심층 학습 접근법인 Hybrid Multi-modal VGG (HM-VGG) 모델을 소개합니다.

- **Technical Details**: HM-VGG 모델은 Visual Field (VF) 데이터를 처리하기 위해 주의 메커니즘 (attention mechanism)을 활용하여 녹내장의 초기 징후를 식별하는 데 중요한 주요 특징을 추출합니다. 이 모델은 제한된 데이터가 있는 상황에서도 뛰어난 성능을 발휘하며, 작은 샘플 크기로도 주목할만한 결과를 달성합니다.

- **Performance Highlights**: 모델의 성능은 Precision, Accuracy, F1-Score에서 높은 지표를 보여주며, 이는 녹내장 탐지 분야에서 실제 적용 가능성을 나타냅니다. 다양한 데이터 유형의 통합이 진단 정확도를 크게 향상시킨다는 점이 강조되었습니다.



### Bayesian-guided Label Mapping for Visual Reprogramming (https://arxiv.org/abs/2410.24018)
- **What's New**: 본 논문은 Bayesian-guided Label Mapping(BLM) 방법을 제안하여 pretrained 모델의 라벨과 downstream 작업의 라벨 간 복잡한 관계를 효과적으로 매핑하는 방식을 소개합니다. 이는 기존의 일대일 맵핑 방식이 가진 한계를 극복하는 방향으로, probabilistic한 접근 방식을 통해 재구성됩니다.

- **Technical Details**: BLM은 각 pretrained 라벨과 downstream 라벨 간의 쌍별 관계를 정량화하는 확률적 라벨 매핑 행렬을 반복적으로 업데이트합니다. 또한, BLM+에서는 예측된 최고 K개의 확률을 집계하여 예측의 불확실성을 반영합니다. Bayesian conditional probability를 기반으로 하여 이 행렬의 각 요소에 값을 할당하는 방법을 사용합니다.

- **Performance Highlights**: BLM과 BLM+는 ResNeXt, CLIP 등의 pretrained 비전 모델을 포함한 12개의 데이터셋에서 기존 라벨 매핑 방법들보다 우수한 성능을 보여줍니다. 실험 결과는 BLM의 효과성을 뒷받침하며 VR의 효과성 분석에 대한 새로운 시각을 제시합니다.



### Assessing the Efficacy of Classical and Deep Neuroimaging Biomarkers in Early Alzheimer's Disease Diagnosis (https://arxiv.org/abs/2410.24002)
Comments:
          SPIE Medical Imaging (MI25)

- **What's New**: 이 연구는 알츠하이머병(AD) 조기 발견의 중요성을 강조하며, 기존 진단 방법의 한계를 극복하기 위해 다양한 이미징 바이오마커를 통합하는 새로운 접근 방식을 제안합니다. 특히, radiomics와 hippocampal texture descriptors의 조합이 조기 AD 탐지에서 높은 성능을 보였습니다.

- **Technical Details**: 연구는 ADNI(Alzheimer's Disease Neuroimaging Initiative)에서 수집한 구조적 자기공명영상(MRI) 스캔을 분석하며, 머신러닝 및 딥러닝 알고리즘을 이용하여 다양한 이미징 바이오마커를 추출하고 통합했습니다. 각종 전처리 단계를 거쳐 503명의 고해상도 T1-weighted 뇌 MRI를 사용하였으며, segmentation에는 FAST-AID Brain 도구가 활용되었습니다.

- **Performance Highlights**: 연구 결과, 다양한 바이오마커를 조합했을 때 탐지 정확도가 향상되었으며, radiomics와 texture feature가 조기 AD 탐지에서 가장 효과적인 predictor로 나타났습니다. AD 및 경도 인지장애(MCI) 탐지에서 각각 0.88과 0.72의 AUC를 기록하였습니다.



### TrAct: Making First-layer Pre-Activations Trainab (https://arxiv.org/abs/2410.23970)
Comments:
          Published at NeurIPS 2024

- **What's New**: 이 논문에서는 비전 모델의 첫 번째 레이어에서의 그래디언트 업데이트와 픽셀 값의 관계를 명확하게 제시합니다. 이미지의 명암 대비가 낮을수록 학습에 미치는 영향이 적으며, 반대로 매우 어두운 이미지나 매우 밝은 이미지는 학습에 더 큰 영향을 미친다는 점을 강조합니다. 이러한 점을 바탕으로, 첫 번째 레이어의 활성화에 대한 그래디언트 하강법을 적용하는 새로운 기술인 TrAct (Training Activations)를 제안합니다.

- **Technical Details**: TrAct는 첫 번째 레이어 활성화에 대한 그래디언트 하강 단계를 통해 활성화 제안을 생성하고, 해당 제안과의 최소 제곱 거리 (squared distance)를 최소화하는 첫 번째 레이어의 최적 가중치를 찾는 절차를 포함합니다. 이 방식은 비전 모델의 아키텍처를 수정하지 않고도 그래디언트를 수정하는 방법을 제시합니다.

- **Performance Highlights**: TrAct는 다양한 비전 모델 아키텍처 (Convolutional 및 Transformer 모델 포함)에서 학습 속도를 1.25배에서 4배까지 가속화하며, 최소한의 계산 오버헤드를 요구합니다. 이 방법은 505050 모델 아키텍처와 데이터 세트 조합에서 일관되게 잘 작동하는 단일 하이퍼파라미터인 λ를 필요로 합니다.



### BitStack: Fine-Grained Size Control for Compressed Large Language Models in Variable Memory Environments (https://arxiv.org/abs/2410.23918)
- **What's New**: 이 논문에서는 BitStack이라는 새로운 비훈련(Training-free) 기반의 가중치 압축 방법을 제안합니다. BitStack은 메모리 사용량과 모델 성능 간의 메가바이트 수준의 균형을 가능하게 하여, 전통적인 압축 방법이 해결하지 못한 문제를 다루고 있습니다.

- **Technical Details**: BitStack은 가중치 행렬을 동적으로 조정하며, SVD(Singular Value Decomposition)를 사용하여 각 파라미터의 중요성을 고려하여 가중치 매트릭스와 잔여 블록을 반복적으로 분해합니다. 이 과정에서 약 1비트의 메모리를 생성하며, 중요도에 따라 잔여 블록을 정렬 및 저장하여 현재 메모리 상황에 맞게 로드할 수 있습니다.

- **Performance Highlights**: 광범위한 실험에서 BitStack은 다양한 작업에서 정밀한 크기 조절에도 불구하고, 특정한 압축 비율에서 전통적인 양자화 기준을 지속적으로 초과하는 성과를 나타냈습니다. 이는 BitStack이 가중치 분해 방법과 양자화 기반 방법 간의 성능 격차를 해소했다는 것을 의미합니다.



### Temporal and Spatial Super Resolution with Latent Diffusion Model in Medical MRI images (https://arxiv.org/abs/2410.23898)
- **What's New**: 이번 논문에서는 Latent Diffusion Model (LDM)과 Vector Quantized GAN (VQGAN) 기반의 인코더-디코더 아키텍처를 결합하여 의료 이미징에 있어 공간 및 시간의 초해상도(SR)를 동시에 개선할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구는 SR을 이미지 노이즈 제거 문제로 정의하고, 영상의 공간 해상도(Spatial Resolution)와 시간 해상도(Temporal Resolution)를 동시에 향상시키기 위해 새로운 모델을 훈련합니다. 데이터 세트는 2D 심장 자기공명영상(MRI) 자료로, 각 환자에 대해 8-14개의 슬라이스와 30개의 프레임을 포함합니다. LDM 모델은 15번의 확산 단계(Diffusion Steps)를 이용해 훈련되었습니다.

- **Performance Highlights**: LDM 모델은 PSNR 30.37, SSIM 0.7580, LPIPS 0.2756를 달성하며, 간단한 기준 방법(Baseline)보다 각각 5%, 6.5%, 39%의 향상을 보였습니다. 이 결과는 의료 이미징 분야에서 진단 정확도와 환자 결과를 향상시킬 수 있는 가능성을 시사합니다.



### Counterfactual MRI Data Augmentation using Conditional Denoising Diffusion Generative Models (https://arxiv.org/abs/2410.23835)
- **What's New**: 이 연구에서는 조건부 노이즈 제거 확산 생성 모델(cDDGM)을 사용하여 기존 환자 해부학을 변경하지 않고 다양한 이미지 획득 파라미터(IAP)에 따라 마그네틱 레조넌스(MR) 이미지를 생성하는 새로운 방법을 도입하였습니다. 이러한 반사적 이미지 생성은 데이터 증강(data augmentation)의 일환으로 수행되어 DL 모델의 세그멘테이션(segmentation) 정확도를 향상시키는데 기여합니다.

- **Technical Details**: cDDGM은 기존의 DDPM 아키텍처를 기반으로 하며, 다중 클래스에 걸쳐 조건부 컨텍스트를 통해 변화하는 이미지를 생성합니다. 본 연구에서는 Duke-Breast-Cancer-MRI 데이터셋을 활용하여 모델을 훈련하였고, IAP를 예측하기 위해 수정된 ResNet-18 모델을 사용하였습니다. 이를 통해 연속 및 범주형 IAP를 예측하였고, 세그멘테이션 작업을 위해 U-Net을 이용하였습니다.

- **Performance Highlights**: 연구에서 개발된 IAP 예측 모델은 테스트 데이터셋의 IAP를 매우 높은 정확도로 포착하였으며, 특히 연속 변수의 경우 낮은 MSE(mean squared error)로 모든 변수를 추정할 수 있었습니다. 세그멘테이션 모델 평가 결과, 배경, 지방, 섬유선 조직(FGT)에 대한 정확도가 향상되었으며, ID 및 OOD 설정 모두에서 좋은 성과를 보였습니다.



### Denoising Diffusion Models for Anomaly Localization in Medical Images (https://arxiv.org/abs/2410.23834)
- **What's New**: 이 장에서는 의료 이미지를 위한 잡음 제거 확산 모델(denoising diffusion models)을 사용한 이상치 위치 탐지(anomaly localization)를 소개합니다. 특히, 다양한 데이터 및 라벨 availability 시나리오에서 이러한 모델의 가능성과 한계를 탐구합니다.

- **Technical Details**: 이 장에서는 잡음 제거 확산 모델의 기본 이론적 배경을 제공하며, 이러한 모델의 재구성 기반 이상치 탐지 적용 방법을 설명합니다. 또한, 이론적 기초에서 기관별 의료 이미지를 위한 다양한 지도 학습(supervision) 스킴을 다룹니다.

- **Performance Highlights**: 모델의 정확성과 일반화 가능성을 높이는 동시에 이상치 탐지에 있어 사용되는 데이터셋 및 평가 지표의 개요를 제공합니다. 현재까지의 연구 격차와 과제를 명확히 하여, 강력한 이상치 위치 탐지를 위한 확산 모델의 잠재력을 강조합니다.



### Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models (https://arxiv.org/abs/2410.23820)
- **What's New**: 이번 연구에서는 분산 모델(diffusion models)을 활용한 비지도 분리 표현 학습(disentangled representation learning, DRL)을 다루고 있습니다. 특히, 동적 가우시안 고정(Dynamic Gaussian Anchoring, DyGA) 기법을 통해 속성 별로 분리된 잠재 단위를 구현하고, Skip Dropout 기술을 통해 노이즈가 포함된 U-Net 모델의 특성 추출 기능을 향상시키고 있습니다.

- **Technical Details**: 연구에서는 두 가지 주요 기법을 제안합니다. 첫째, DyGA는 잠재 공간에서 속성 클러스터를 위한 앵커를 동적으로 선택하여 혼란스러운 포인트를 이 앵커 쪽으로 이동시킵니다. 둘째, Skip Dropout은 U-Net의 스킵 연결 기능을 제거하여 DM 훈련이 잠재 단위 특성에 집중하도록 합니다.

- **Performance Highlights**: 제안된 기법은 기존 DM 기반 DRL 방법에 적용하여 SOTA(최첨단) 비지도 DRL 성능을 달성하였으며, 획득된 표현은 다운스트림 작업에서도 우수한 성능을 보입니다. 노이즈 제거 및 이미지 생성 기능에서의 동시 학습을 통해, 더욱 해석 가능한 분리 기능을 제시합니다.



### MLLA-UNet: Mamba-like Linear Attention in an Efficient U-Shape Model for Medical Image Segmentation (https://arxiv.org/abs/2410.23738)
- **What's New**: 이 논문에서는 MLLA-UNet(Mamba-Like Linear Attention UNet)라는 새로운 아키텍처를 제안합니다. 이 아키텍처는 선형 컴퓨팅 복잡성을 유지하면서도 높은 분할 정확성을 자랑합니다.

- **Technical Details**: MLLA-UNet는 선형 어텐션(linear attention)과 Mamba에서 영감을 받은 적응 메커니즘을 혁신적으로 결합하여 구성됩니다. 또한, 효율적인 대칭 샘플링 구조(symmetric sampling structure)를 채택하여 향상된 특징 처리를 지원합니다. 이 구조는 필수적인 공간적 특징(spatial features)을 효과적으로 보존하며 낮은 컴퓨팅 복잡도로 장거리 의존성(long-range dependencies)을 포착합니다.

- **Performance Highlights**: MLLA-UNet는 FLARE22, AMOS CT, ACDC 등을 포함한 6개의 도전적인 데이터셋에서 24개의 다양한 분할(task) 작업에 대해 최첨단 성능(state-of-the-art performance)을 달성하며, 평균 DSC(Dice Similarity Coefficient)는 88.32%입니다. 이는 기존 방법들에 비해 MLLA-UNet의 우수성을 강조합니다.



### Aggregate-and-Adapt Natural Language Prompts for Downstream Generalization of CLIP (https://arxiv.org/abs/2410.23698)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 CLIP과 같은 대형 pretrained (사전학습된) 비전-언어 모델의 제약을 극복하기 위해 prompt learning (프롬프트 학습) 접근 방식을 개선합니다.

- **Technical Details**: 우리는 자연어 프롬프트(인간 또는 LLM 생성)에서 텍스트 지식을 증류하여 부족한 개념에 대한 풍부한 사전 정보를 제공합니다. 먼저 학습된 프롬프트 집합기를 통해 각 입력 이미지에 정렬된 프롬프트 요약을 얻습니다. 그런 다음 프롬프트 생성을 공동으로 훈련시키며, 집계된 요약에 가까운 프롬프트 임베딩을 생성하도록 최적화합니다. 이 프롬프트 임베딩은 Aggregate-and-Adapted Prompt Embedding (AAPE)로 명명됩니다.

- **Performance Highlights**: AAPE는 비전-언어 이해 작업(예: few-shot classification, VQA) 및 생성 작업(이미지 캡션 생성)에서 경쟁력 있는 성능을 보여주며, 특히 비정형(non-canonical) 및 OOD(Out-of-Distribution) 예제를 처리하는 데 도움을 줍니다. 또한, AAPE 학습은 기존의 LLM 기반 추론 비용을 없애고 데이터 및 LLM 모델 크기에 따라 더 나은 확장성을 제공합니다.



### Wide Two-Layer Networks can Learn from Adversarial Perturbations (https://arxiv.org/abs/2410.23677)
Comments:
          NeurIPS24

- **What's New**: 이번 연구에서는 adversarial examples의 성공적인 설명을 위한 이론적 기반을 제시합니다. 기존의 가설은 adversarial perturbations이 랜덤 노이즈처럼 보이지만 특정 클래스의 특징을 포함하고 있다고 주장합니다.

- **Technical Details**: 이 연구는 넓은 두 레이어 네트워크를 가정하고 있으며, 결과는 모든 데이터 분포에 적용될 수 있습니다. 우리는 adversarial perturbations이 충분한 클래스 특성을 포함하여 네트워크가 이로부터 일반화할 수 있음을 증명합니다.

- **Performance Highlights**: 직접 라벨링된 깨끗한 샘플에서 훈련된 분류기와 잘못 라벨링된 adversarial examples에서 훈련된 분류기의 예측이 일치함을 보여줍니다.



### Novel Clinical-Grade Prostate Cancer Detection and Grading Model: Development and Prospective Validation Using Real World Data, with Performance Assessment on IHC Requested Cases (https://arxiv.org/abs/2410.23642)
- **What's New**: 인공지능(AI)이 병리학(pathology) 서비스의 수요 증가에 대응하고 진단 품질을 유지하면서도 소요 시간을 단축하고 비용을 절감할 수 있는 가능성이 제시되었습니다. 본 연구는 기관 개발 시스템의 전립선 암(prostate cancer) 탐지, 등급화(grading), 워크플로우 최적화를 평가하고 상업적인 대안과 비교하였습니다.

- **Technical Details**: 2021년 8월부터 2023년 3월까지 1,147명의 양성 생검(biopsy) 환자로부터 21,396개의 슬라이드를 검사하였습니다. 우리는 전립선 암 생검의 PANDA 데이터셋을 사용하여 훈련된 작업 특화 모델과 일반 목적의 조직학 기초 모델인 UNI에 의해 추출된 특징을 사용하여 구축된 모델을 비교했습니다. 비특이적 케이스에 대한 IHC(면역조직화학(Immunohistochemistry)) 처방을 위한 스크리닝 모델을 개발했으며, 암의 작은 초점을 감지하는 민감도를 개선하기 위해 맞춤형 모델의 기여도를 평가했습니다.

- **Performance Highlights**: 개발된 시스템과 병리학자의 기준(reference) 간의 높은 일치성을 보였습니다(탐지 AUC 98.5, 민감도 95.0, 특이도 97.8). ISUP 등급화에서 quadratic Cohen의 kappa는 0.869로 나타났으며, 3등급(group 3) 이상에 대한 성능도 우수했습니다(AUC 97.5, 민감도 94.9, 특이도 96.6). 스크리닝을 통해 비특이적 케이스에 대한 IHC 처방을 44.5% 줄이고, 전체 오류율은 1.8%로 나타났습니다(1.4% 거짓 양성(false positive), 0.4% 거짓 음성(false negative) 비율 포함). 이러한 모델은 병리학 실험실의 품질 관리와 워크플로우 개선에 기여할 잠재력이 있습니다.



### Cycle-Constrained Adversarial Denoising Convolutional Network for PET Image Denoising: Multi-Dimensional Validation on Large Datasets with Reader Study and Real Low-Dose Data (https://arxiv.org/abs/2410.23628)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 Cycle-constrained Adversarial Denoising Convolutional Network (Cycle-DCN)라는 새로운 모델을 제안하여 저선량 PET 이미지를 고선량 품질로 재구성할 수 있음을 보여줍니다.

- **Technical Details**: Cycle-DCN 모델은 노이즈 예측기(noise predictor), 두 개의 판별기(discriminators), 일관성 네트워크(consistency network)를 통합하여 설계되었습니다. 이를 통해 지도 학습(loss), 적대적 손실(adversarial loss), 주기 일관성 손실(cycle consistency loss), 아이덴티티 손실(identity loss), 인근 구조적 유사성 지표(SSIM) 손실을 결합한 방식으로 최적화하였습니다.

- **Performance Highlights**: Cycle-DCN은 세 가지 선량 수준에서 평균 Peak Signal-to-Noise Ratio (PSNR), SSIM, Normalized Root Mean Square Error (NRMSE)에서 각각 56%, 35%, 71% 개선을 보였습니다. 또한 이미지 세부 정보를 효과적으로 보존하며, 빠진 엣지를 해결하여 훨씬 더 높은 품질의 이미지를 제공합니다. 독립적인 독자 연구(reader studies)에 따르면 Cycle-DCN으로 복원된 이미지는 핵의학 의사들로부터 가장 높은 평가를 받았습니다.



### MS-Glance: Non-semantic context vectors and the applications in supervising image reconstruction (https://arxiv.org/abs/2410.23577)
Comments:
          Accepted by WACV 2025

- **What's New**: 본 연구에서는 비의미적(non-semantic) 컨텍스트 정보를 활용한 시각 인식 방법론을 제안합니다. 특히, 새로운 비의미적 컨텍스트 설명자인 MS-Glance와 두 이미지 간 비교를 위한 Glance Index Measure를 소개합니다.

- **Technical Details**: MS-Glance는 이미지에서 픽셀을 무작위로 선택하여 형성한 Global Glance 벡터와 로컬 이미지 윈도우를 평면화하여 생성한 Local Glance 벡터를 포함합니다. Glance Index는 두 세트의 표준화된 Glance 벡터 간의 내적(inner product)으로 정의됩니다. 이 방법은 두 가지 이미지 복원 작업, 즉 암묵적인 신경 표현(INR)으로 인한 이미지 피팅과 저샘플링된 MRI 복원에 효과적입니다.

- **Performance Highlights**: MS-Glance는 자연 이미지 및 의료 이미지를 포함한 다양한 데이터셋에서 기존 이미지 복원 손실을 초월하는 성능을 보여줍니다. MS-Glance 손실은 L1+SSIM 및 LPIPS와 같은 기존 손실 함수들보다 뛰어난 결과를 나타내며, 이미지 품질 향상에 기여합니다.



### 2D Empirical Transforms. Wavelets, Ridgelets and Curvelets revisited (https://arxiv.org/abs/2410.23533)
- **What's New**: 최근에 개발된 새로운 접근법인 'Empirical Wavelet Transform'(EWT)은 분석된 신호에 따라 1D 적응형 웨이블릿 프레임을 구축하는 것을 목표로 하고 있습니다. 본 논문에서는 이 접근법을 2D 신호(이미지)로 확장하는 여러 가지 방법을 제시합니다.

- **Technical Details**: 웨이블릿(wavelet)과 그 기하학적 확장(framelets, ridgelets, curvelets 등)은 조화해석(harmonic analysis)에서 출발하여 함수 공간(function spaces)을 연구하는 데 유용한 수학적 도구입니다. 하지만 전통적인 웨이블릿 변환의 기초는 다이아딕(dyadic) 스케일 분해에 기반하고 있어 이미지 표현의 최적성을 보장하지 못합니다. EWT는 신호의 주파수 지원을 감지하고 그에 따라 웨이블릿을 구축하는 두 가지 주요 단계로 구성됩니다.

- **Performance Highlights**: 논문에서는 EWT를 기반으로 한 다양한 실험 결과를 제시하며, 이 방법들이 이미지 분석 및 처리에 대해 유망한 특성을 가진 다양한 적응형 프레임을 생성할 수 있음을 입증합니다.



### PACER: Preference-conditioned All-terrain Costmap Generation (https://arxiv.org/abs/2410.23488)
- **What's New**: PACER는 사용자 정의 환경에 따라 빠른 비용 할당을 가능하게 하는 새로운 접근 방식을 도입합니다. 기존의 semantics 기반 접근법과 대조적으로, PACER는 새로운 지형을 빠르게 적응하여 비용 맵을 생성할 수 있습니다.

- **Technical Details**: PACER는 단일 드론 관점(BEV) 이미지와 사용자 지정 선호 컨텍스트를 입력으로 받아들이며, 이를 바탕으로 선호에 맞는 BEV 비용 맵을 생성합니다. 이 방법은 동적인 환경에서 로봇의 비용 할당을 지원하며, 다양한 신규 지형에 대한 적응성을 제공합니다.

- **Performance Highlights**: PACER는 기존 semantics 기반 및 표현 학습 접근법에 비해 새로운 사용자 선호에 신속하게 적응할 수 있으며, 이전에 보지 못한 환경에서도 더 좋은 일반화를 보여줍니다.



### Keep on Swimming: Real Attackers Only Need Partial Knowledge of a Multi-Model System (https://arxiv.org/abs/2410.23483)
Comments:
          11 pages, 2 figures

- **What's New**: 최근 기계학습에서 다수의 모델이나 agentic 아키텍처를 결합하여 작업을 수행하는 접근법이 증가하고 있습니다. 이 논문은 최종 블랙박스 모델에 대한 프록시 모델만 있을 때 멀티 모델 시스템을 대상으로 하는 새로운 적대적 공격 방법을 제안합니다.

- **Technical Details**: 우리는 복합 모델 시스템에서 초기 모델들이 적대적 변화를 효율적으로 방지할 때, 공격자가 최종 모델에 대한 간접적인 접근성만 가질 경우 공격을 설정하는 방법을 개발했습니다. 이 새로운 방법은 이전의 공격 방법과 비교했을 때 성공률이 약 80%로 더 높고, MSE 기준으로 9.4% 작은 변화를 포함합니다.

- **Performance Highlights**: 본 연구의 실험은 감독된 이미지 파이프라인에 중점을 두었으나, 제안된 공격 방법은 다른 멀티 모델 환경에서도 효과적으로 일반화될 것으로 기대됩니다.



### STIED: A deep learning model for the SpatioTemporal detection of focal Interictal Epileptiform Discharges with MEG (https://arxiv.org/abs/2410.23386)
Comments:
          10 pages, 7 figures

- **What's New**: 최신 연구에서는 Magnetoencephalography (MEG)를 이용한 간질 환자의 비침습적 진단 방법으로, 기존의 수동 방식에서 벗어나 심층 학습(deep learning)을 이용한 자동 이데이터 탐지 방법이 소개되었습니다.

- **Technical Details**: 이 연구에서 개발된 STIED는 1차원 시간 경과 및 2차원 정위(topography) MEG 신호의 특징을 결합한 두 개의 컨볼루션 신경망(convolutional neural networks)으로 구성된 감독(supervised) 딥러닝 알고리즘입니다. STIED는 간질 환자의 간질 전기적 방출(interictal epileptiform discharges, IEDs)을 시간 및 공간적으로 지역화할 수 있도록 설계되었습니다.

- **Performance Highlights**: STIED 모델은 주로 고주파 긍정파를 가진 국소 간질 환자(FE 그룹)에서 IED를 탐지할 때 85% 이상의 정확도(accuracy), 특이도(specificity), 민감도(sensitivity)를 보였습니다. 또한, STIED는 입력 데이터를 활용하여 기존의 임상 MEG 실습을 모방하여 우수한 성능을 발휘하며, 다른 유형의 저항성 국소간질 환자에게도 적용 시 긍정적인 결과를 보였습니다.



### NCAdapt: Dynamic adaptation with domain-specific Neural Cellular Automata for continual hippocampus segmentation (https://arxiv.org/abs/2410.23368)
- **What's New**: NCAdapt는 의료 이미징 분야에서의 계속 학습(Continual Learning, CL) 문제를 해결하기 위해 설계된 Neural Cellular Automata (NCA) 기반의 새로운 방법입니다. 이 방법은 각 새로운 도메인에 맞춤화된 다중 헤드 구조를 특징으로 하며, 이전에 학습한 지식을 유지하면서 새로운 도메인에 적응할 수 있도록 합니다.

- **Technical Details**: NCAdapt는 초기 훈련 후 NCA 백본을 동결하고, 384개의 파라미터로 구성된 새롭게 추가된 적응형 컨볼루션 레이어만 훈련합니다. 각 도메인에서 NCA 컨볼루션과 결합하여 작업 특화 레이어를 통해 새로운 도메인 지식을 습득할 수 있도록 합니다. 이를 통해 복잡한 정규화 기술이나 전체 모델 재훈련 없이도 효율적인 적응을 보장합니다.

- **Performance Highlights**: NCAdapt는 히포캄퍼스 분할 작업에서 Lifelong nnU-Net 및 U-Net 모델과 비교하여 SOTA 성능을 달성했습니다. 이 경량화된 접근 방식은 에너지 효율성을 강조하며, 6,339개의 훈련 가능 파라미터만으로도 전통적인 nnU-Net보다 2,559배 작고, 새로운 도메인에 대한 적응 속도가 빠릅니다.



### Variable Resolution Sampling and Deep Learning Image Recovery for Accelerated Multi-Spectral MRI Near Metal Implants (https://arxiv.org/abs/2410.23329)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구는 금속 임플란트 근처의 다중 스펙트럼(Multi-Spectral) MRI에서 스캔 시간을 줄이면서 이미지 품질을 유지하는 변동 해상도(Variable Resolution, VR) 샘플링 및 딥러닝(reconstruction) 재구성 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 1.5T MSI 무릎 및 엉덩이 데이터를 사용하여 새로운 스펙트럴 언더샘플링(spectral undersampling) 방식을 통해 약 40%의 획득 효율을 개선했습니다. U-Net 기반의 딥러닝 모델을 통해 이미지 재구성이 이루어졌으며, SSIM, PSNR 및 RESI 메트릭을 사용하여 이미지 품질을 평가했습니다.

- **Performance Highlights**: 딥러닝 재구성(DL-VR)된 언더샘플링 VR 데이터는 기존 재구성(CR-VR)에 비해 SSIM 및 PSNR 값이 유의미하게 높았으며(p<0.001), 엣지 샤프니스(edge sharpness)가 개선되었습니다. DL로 재구성된 이미지의 엣지 샤프니스는 전체 샘플링 참조 이미지와 일치했습니다(p=0.5).



### Deep learning meets tree phenology modeling: PhenoFormer vs. process-based models (https://arxiv.org/abs/2410.23327)
Comments:
          journal-preprint

- **What's New**: 이 논문에서는 기후 변화에 따른 식물의 생리적 사건 예측을 위해 PhenoFormer라는 딥러닝 기반 모델을 소개합니다. 이 모델은 기존의 통계적 방법보다 기후 데이터의 분포 변화에 더 효과적으로 대응하며, 최고의 프로세스 기반 모델과 유사한 성능을 보입니다.

- **Technical Details**: PhenoFormer는 self-attention 기법을 기반으로 한 신경망 구조로, 기상 시계열 데이터를 활용하여 식물의 생리적 사건 예측을 목표로 합니다. 연구에서는 스위스의 9종 식물에 대한 70년간의 67,800개 생리적 관측 데이터를 사용하여 실험하였습니다.

- **Performance Highlights**: PhenoFormer는 봄 생리적 사건 예측 시 기존 전통적인 머신러닝 기법보다 평균 1.1일의 RMSE(로트 오차) 개선과 14% R2(결정계수) 향상을 보여주며, 가을 생리적 사건 예측에서도 잘 수행됩니다. 이 결과는 해당 딥러닝 모델이 기후 변화 하에서 상당한 예측 능력을 가지고 있음을 시사합니다.



### Improved Patch Denoising Diffusion Probabilistic Models for Magnetic Resonance Fingerprinting (https://arxiv.org/abs/2410.23318)
Comments:
          12 pages, 5 figures, 2 algorithms

- **What's New**: 본 논문은 Magnetic Resonance Fingerprinting (MRF) 이미지 복원에 대한 최초의 조건부 확산 확률 모델 (conditional diffusion probabilistic model)을 제안합니다. 이는 기존의 딥 러닝 및 압축 센싱 알고리즘을 초월하여 이미지 복원과 정량적 매핑의 정확성을 향상시킵니다.

- **Technical Details**: MRF 이미지를 복원하기 위해, 본 연구에서는 Denoising Diffusion Probabilistic Models (DDPMs)를 사용합니다. DDPM은 이미지를 세밀하게 복원하는 과정을 반복적으로 수행하며, 고차원 MRF 데이터에 대해 훈련의 효율성을 개선하기 위해 서브스페이스 차원 축소(subspace dimensionality reduction)를 적용했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 인체 두뇌 스캔 데이터에서 기존의 MRF 알고리즘 대비 뛰어난 성능을 보였으며, 여러 샘플 복원을 생성할 수 있어 복원 불확실성을 시각화할 수 있는 장점을 가지고 있습니다.



### Improving Image Data Leakage Detection in Automotive Softwar (https://arxiv.org/abs/2410.23312)
- **What's New**: 이 연구는 Volvo Cars와의 산업 파트너십을 통해 Cirrus 데이터셋에서 데이터 누수를 탐지하기 위한 방법을 개발하고, 이를 Kitti 데이터셋에서 평가하여 이전에 알려지지 않았던 데이터 누수를 발견했습니다.

- **Technical Details**: 연구는 데이터 누수가 객체 탐지(OD) 성능에 미치는 영향을 분석하고, 데이터 누수를 탐지하는 방법론을 제안합니다. 이 방법론은 이미지 데이터의 유사성으로 인해 누수를 탐지하기 어려운 자동차 인식 시스템의 특징을 고려합니다.

- **Performance Highlights**: Kitti 데이터셋에 대한 평가 결과, 제안된 방법이 데이터 누수를 효과적으로 탐지할 수 있음을 보여주며, 이는 자동차 자율주행 기술 개발에 중요한 발견입니다.



### Detection of adrenal anomalous findings in spinal CT images using multi model graph aggregation (https://arxiv.org/abs/2410.20568)
- **What's New**: 이 연구에서는 허리 통증으로 CT 스캔을 받은 환자의 부신(Adrenal glands) 이상 병변을 탐지하는 새로운 방법인 Multi Model Graph Aggregation (MMGA) 방법을 제안합니다. 기존의 허리 중심 CT 스캔을 사용하여 새로운 유형의 복부 병리학을 찾아내는 점에서 혁신적입니다.

- **Technical Details**: 연구는 세 가지 deep learning 모델로 구성되어 있습니다. 첫 번째 모델은 CNN(Convolutional Neural Networks) 구조를 기반으로 부신 영역의 슬라이스를 필터링합니다. 두 번째 모델은 YOLO V3 객체 탐지 구조를 사용하여 각 슬라이스에서 비정상 병변을 감지합니다. 세 번째 모델은 Graph CNN(Graoh Convolutional Network) 아키텍처를 기반으로 하여 이전 단계에서 분류된 부신의 비정상성을 종합적으로 평가합니다.

- **Performance Highlights**: 분류성능에서는 비정상 부신 병변의 예측값이 67%에 달하며, 비정상성이 없을 때 비정상 리스크가 5.6%로 줄어드는 것으로 나타났습니다. 국소화 성능에서는 평균 슬라이스 인덱스 오류가 왼쪽 부신 8.2, 오른쪽 부신 2.25를 기록하였으며, 평균 Intersection Over Union (IOU) 점수는 왼쪽 부신 0.41, 오른쪽 부신 0.52로 평가되었습니다.



New uploads on arXiv(cs.AI)

### Graph Learning for Numeric Planning (https://arxiv.org/abs/2410.24080)
Comments:
          Extended version of NeurIPS 2024 paper

- **What's New**: 본 연구에서는 숫자 기반 계획(numeric planning) 작업을 해결하기 위한 데이터 효율적이고 해석 가능한(machine learning) 모델을 제안합니다. 이는 연속 및 범주형 속성을 갖는 그래프를 위한 새로운 graph kernel을 구축하고, 숫자 기반 계획을 위한 heuristic 함수(heuristic functions)를 학습하기 위한 새로운 최적화 방법을 포함합니다.

- **Technical Details**: 새롭게 제안된 graph kernel은 연속적인 속성과 범주형 속성을 모두 지원하는 그래프에 적용될 수 있습니다. 또한, 연구에서는 숫자 기반 계획을 위해 heuristic 함수를 학습할 때 새로운 최적화 기법을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 graph kernel은 숫자 기반 계획에 대해 graph neural networks보다 훨씬 더 효율적이며 일반화 성능(generalisation performance)에서 우수한 결과를 보여주었습니다. 또한, 도메인(영역) 독립적인 숫자 계획기(numeric planners)와 비교하여 경쟁력 있는 커버리지 성능을 나타냈습니다.



### AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents (https://arxiv.org/abs/2410.24024)
- **What's New**: 이번 논문에서는 AndroidLab이라는 새로운 Android Agent(에이전트) 평가 및 훈련 프레임워크를 제안합니다. 이 프레임워크는 다양한 모드와 작업 공간을 제공하며, 오픈 소스 및 클로즈드 소스 모델을 모두 지원합니다.

- **Technical Details**: AndroidLab은 기준 작업 환경과 138개의 다양한 앱에서의 과제를 포함하여 에이전트의 상호작용을 지원합니다. XML 모드와 SoM 모드로 운영되며, 각각 다양한 관찰에 따른 기본 작동 모드를 정의합니다. 각 작업은 여러 단계로 나누어 UI 트리 구조를 확인하고, 작업 효율성을 평가하기 위한 다양한 메트릭스를 도입합니다.

- **Performance Highlights**: AndroidLab 환경을 활용하여 Android Instruction dataset을 구축하고, 6개의 오픈 소스 LLM과 LMM을 훈련시켰습니다. LLM의 평균 성공률을 4.59%에서 21.50%로, LMM을 1.93%에서 13.28%로 향상시켰습니다. 이는 오픈 소스 모델의 성능 향상을 시사합니다.



### Average Controlled and Average Natural Micro Direct Effects in Summary Causal Graphs (https://arxiv.org/abs/2410.23975)
- **What's New**: 본 논문에서는 summary causal graphs를 통한 평균적으로 통제된 직접 효과(average controlled direct effects)와 평균 자연 직접 효과(average natural direct effects)의 식별 가능성(identifiability)을 조사합니다. 이 구조는 동적 시스템(dynamic systems)에서 주로 사용됩니다.

- **Technical Details**: 비모수적 직접 효과(non-parametric direct effects)는 전통적인 선형 설정(traditional linear setting)에서는 쉽게 식별하고 추정할 수 있지만, 실제 세계의 복잡성을 다루는 데 있어 특히 역학적 맥락(epidemiological contexts)에서는 정의하고 식별하기가 훨씬 어렵습니다. 본 연구는 숨겨진 혼란(hidden confounding)의 존재 하에서 average controlled micro direct effect와 average natural micro direct effect를 식별하기 위한 충분한 조건(sufficient conditions)을 제공합니다.

- **Performance Highlights**: 평균 통제된 마이크로 직접 효과(average controlled micro direct effect)의 경우, 숨겨진 혼란이 없는 설정에서도 주어진 조건들이 필요 조건(necessary conditions)이 된다는 것을 보여줍니다.



### RL-STaR: Theoretical Analysis of Reinforcement Learning Frameworks for Self-Taught Reasoner (https://arxiv.org/abs/2410.23912)
- **What's New**: 이번 연구는 자율적인 추론 능력 향상을 위해 강화 학습을 활용하는 Self-Taught Reasoner (STaR) 프레임워크의 이론적 기초를 제시합니다. 이는 대규모 언어 모델(LLMs)이 Chain-of-Thought (CoT) 추론에서 어떻게 개선될 수 있는지를 설명합니다.

- **Technical Details**: STaR는 LLM이 스스로 추론 단계를 생성하도록 설계된 정책 경량화 방법입니다. 이 연구는 정책 개선, 최적 정책 수렴 조건, 오류 발생 시에도 추론 능력을 향상시키는 방법, 효과적인 추론 개선을 위한 사전 훈련 모델의 품질 기준을 다룹니다.

- **Performance Highlights**: STaR는 수많은 실험을 통해 LLMs가 인간의 개입 없이도 강화 학습을 통해 효과적으로 추론 단계를 학습할 수 있음을 입증했습니다. 이는 LLMs의 추론 능력 향상에 있어 혁신적인 접근 방식을 제공하며, 비지도 학습을 통한 추론 능력 개선에 기여할 것으로 기대됩니다.



### Neural Network Verification with PyRA (https://arxiv.org/abs/2410.23903)
- **What's New**: 이 논문에서는 PyRAT(Python Reachability Assessment Tool)를 소개하며, 신경망의 안전성과 강인성을 검증하기 위해 추상 해석(abstract interpretation)을 기반으로 하는 도구를 개발했다고 발표했다. PyRAT는 신경망의 입력에서 시작하여 도달 가능한 상태들을 찾기 위한 다양한 추상화 기법을 사용한다.

- **Technical Details**: PyRAT는 Python으로 작성되어 있으며, Tensorflow, PyTorch와 같은 다양한 신경망 훈련 프레임워크와의 쉽게 인터페이스 할 수 있다. 이 도구는 ONNX, NNet, Keras 같은 표준 AI 형식을 처리하며 신경망의 안전성을 평가할 수 있는 여러 기능과 인터페이스를 제공한다. PyRAT는 여러 가지 추상 도메인(abstract domain)을 활용하여 신경망의 안전성을 검증한다.

- **Performance Highlights**: PyRAT는 여러 국가 및 국제 프로젝트에서 검증되었으며, 2023 및 2024년 VNN-Comp에 참가해 각각 3위와 2위를 기록하며 성능을 입증했다.



### Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs (https://arxiv.org/abs/2410.23875)
- **What's New**: 본 논문에서는 KG(Knowledge Graph)와 LLM(Large Language Model)의 결합을 통해 자가 수정이 가능한 적응형 계획 패러다임인 PoG(Plan-on-Graph)를 제안합니다.

- **Technical Details**: PoG는 질문을 여러 하위 목표로 분해하고, 자가 수정된 경로 탐색, 메모리 업데이트, 잘못된 경로 수정을 반복하는 과정을 포함합니다. 이 과정에서 (1) Guidance는 질문 조건을 활용하여 탐색을 안내하고, (2) Memory는 과거의 추론 정보를 기록, 업데이트하며, (3) Reflection은 현재 경로의 지속 여부를 판단하여 자가 수정을 수행합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋(CWQ, WebQSP, GrailQA)에 대한 광범위한 실험을 통해 PoG의 효과성과 효율성이 입증되었습니다.



### Reasons and Solutions for the Decline in Model Performance after Editing (https://arxiv.org/abs/2410.23843)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문은 지식 편집 모델의 성능 저하 원인을 조사하고 이를 최적화하기 위한 방법을 제시합니다. 특히, 데이터와 모델 관점에서 편집 모델의 성능 저하 요인들을 분석하고, 편집 성능을 개선하기 위한 새로운 방법인 Dump for Sequence (D4S)를 제안합니다.

- **Technical Details**: 이 논문은 Multi-Question Dataset (MQD)를 사용하여 다양한 편집 데이터 유형이 모델 성능에 미치는 영향을 평가합니다. 실험을 통해 편집 모델의 성능은 편집 목표의 다양성과 시퀀스 길이에 의해 주로 영향을 받는다는 것을 확인했습니다. 또한 편집 모델 레이어의 L1-norm 성장과 편집 정확도 간의 강한 상관관계를 분석하여, 성능 병목 현상의 주요 원인을 밝혔습니다.

- **Performance Highlights**: 제안된 D4S 방법은 편집 레이어의 L1-norm 증가를 감소시켜 성능을 향상시키고, 사용자가 여러 번의 효과적인 편집을 수행할 수 있도록 하여 모델 손상을 최소화합니다. 이 방법은 기존 편집과 관련된 성능 저하 문제를 효과적으로 해결합니다.



### Towards Reliable Alignment: Uncertainty-aware RLHF (https://arxiv.org/abs/2410.23726)
- **What's New**: 이 연구는 인간의 선호에 맞는 대규모 언어 모델(LLM)의 정렬을 위한 새로운 보상 모델의 불확실성을 해결하는 방법을 제시합니다. 이를 위해 보상 모델의 변동성을 분석하고, 이를 감소시키기 위한 보수적인 정책 최적화(Conservative Policy Optimization) 기법을 도입했습니다.

- **Technical Details**: 본 논문에서는 Reinforcement Learning with Human Feedback (RLHF) 프레임워크 내에서 보상 모델이 작동하는 방법에 대해 설명합니다. 보상 모델은 작은 데이터셋에서 학습되며, 이를 통해 LLM의 반응 품질을 예측합니다. 연구진은 무작위 최적화(stochastic optimization) 알고리즘을 사용하여 이러한 모델이 어떻게 높은 변동성을 보이는지를 실증적으로 보여줍니다. 또한, 불확실성이 큰 보상 모델에 대해 더 조심스러운 정책을 찾기 위한 새로운 알고리즘의 필요성을 강조합니다.

- **Performance Highlights**: 실험 결과, 제안된 보수적인 정책 최적화 방법이 오버피팅(overfitting)을 줄이고, 불확실한 보상 모델로부터 파생된 정책의 성과가 향상되었습니다. 이 방식은 LLM의 성능 저하를 방지하며, 이론적 예측과 실험 결과가 일치함을 확인했습니다.



### Argumentation and Machine Learning (https://arxiv.org/abs/2410.23724)
Comments:
          44 pages, to appear in the Handbook of Formal Argumentation and the Journal of Applied Logics

- **What's New**: 이번 논문에서는 Computational Argumentation과 Machine Learning이 상호작용하는 다양한 연구 결과를 리뷰하였습니다. 논문의 초점은 오가는 주제를 정리하고 두 분야의 상호작용을 세 가지 유형(유기적 상호작용, 구분된 접근, 근사적 접근)으로 구분하여 각각의 특징을 논의하는 데 있습니다.

- **Technical Details**: 논문에서는 ML의 주요 특성이 감독학습(supervised learning)이라는 점과 관련 기술(예: decision trees, random forests, neural networks 등)을 다루고 있습니다. 또한, Argumentation의 다양한 구조적 프레임워크(예: Abstract Argumentation Frameworks, Value-Based Argumentation Frameworks 등)가 논의되며, 특히 QBAFs와 같은 모델들이 NN과 결합하여 사용될 수 있다는 점도 강조됩니다.

- **Performance Highlights**: 연구 결과들은 ML의 다양한 분류 및 정책 학습을 지원하기 위해 Argumentation을 효과적으로 사용할 수 있음을 보여주며, 응용 분야로는 의료, 로봇 공학, 예측, 법률 및 상식 추론 등이 포함됩니다. 그러나 ML 모델에 대한 구현의 부족과 사용자 연구의 결여가 주요한 문제점으로 지적되고 있습니다.



### Bridging Geometric States via Geometric Diffusion Bridg (https://arxiv.org/abs/2410.24220)
Comments:
          33 pages, 5 tables; NeurIPS 2024 Camera Ready version

- **What's New**: 이 논문에서는 Geometric Diffusion Bridge (GDB)라는 새로운 생성 모델링 프레임워크를 소개합니다. GDB는 초기와 목표 기하학적 상태를 정확하게 연결하며, 기존의 방법들이 직면했던 문제들을 해결하고자 합니다.

- **Technical Details**: GDB는 확률적 접근을 통해 기하학적 상태 분포를 발전시키며, Doob의 h-transform을 수정하여 도출된 공변량 확산 다리(equivariant diffusion bridge)를 사용합니다. 이 프레임워크는 초기 및 목표 기하학적 상태를 고정된 엔드포인트로 하여 개발됩니다.

- **Performance Highlights**: 다양한 실험에서 GDB는 기존 최첨단 방법들을 초월하는 성능을 보여주었습니다. 특히, OC22 구조 완화 작업에서 GDB는 10배 더 많은 데이터를 학습한 강력한 머신러닝 포스 필드(MLFF) 기준선을 초과하며, 주어진 궤적 데이터의 지침을 통해 성능을 더욱 향상시킬 수 있음을 입증하였습니다.



### Teaching Embodied Reinforcement Learning Agents: Informativeness and Diversity of Language Us (https://arxiv.org/abs/2410.24218)
Comments:
          EMNLP 2024 Main. Project website: this https URL

- **What's New**: 이 연구는 강화학습을 위한 체화 에이전트의 언어 입력의 다양성과 정보성의 영향을 처음으로 상세히 조사하였습니다.

- **Technical Details**: Decision Transformer (DT) 모델을 기반으로 하는 Language-Teachable Decision Transformer (LTDT) 아키텍처를 제안했습니다. 이 시스템은 언어 피드백을 통해 에이전트의 학습 능력에 큰 영향을 미치며, GPT-4를 활용해 더 자연스럽고 풍부한 언어 표현을 생성합니다.

- **Performance Highlights**: 이 연구 결과, 다양한 언어 피드백을 사용하여 훈련된 에이전트는 일반화가 향상되고 새로운 작업에 신속하게 적응하는 능력이 증가하였으며, 언어 없이 훈련된 에이전트보다 평균 20포인트 이상의 성능 향상을 기록했습니다.



### Understanding Optimization in Deep Learning with Central Flows (https://arxiv.org/abs/2410.24206)
Comments:
          first two authors contributed equally; author order determined by coin flip

- **What's New**: 본 논문의 주요 기여는 optimizer(최적화기)의 암묵적인 행동을 'central flow'(중앙 흐름)라는 미분 방정식으로 명시적으로 포착할 수 있음을 보여주는 것이다.

- **Technical Details**: 'central flow'는 시간 평균 최적화 경로를 모델링하는 미분 방정식으로, 이를 통해 다양한 신경망의 장기 최적화 경로를 높은 수치적 정확도로 예측할 수 있다. 이 흐름을 해석함으로써 RMSProp이 지역 손실 경관에 적응하는 방식과 '정규화를 통한 가속화'(acceleration via regularization) 메커니즘을 처음으로 밝혀냈다.

- **Performance Highlights**: 이 논문은 adaptive optimizer(적응형 최적화기)가 더 큰 단계를 진행할 수 있는 저곡률(low-curvature) 지역으로 암묵적으로 나아가는 메커니즘이 그 효과성에 핵심적이라는 것을 보여준다.



### Zonal RL-RRT: Integrated RL-RRT Path Planning with Collision Probability and Zone Connectivity (https://arxiv.org/abs/2410.24205)
- **What's New**: Zonal RL-RRT 알고리즘이 소개되었으며, 이는 kd-tree 파티셔닝을 활용하여 지도를 여러 구역으로 나누고, 구역 간의 연결성을 보장하여 매끄러운 전환을 가능하게 합니다.

- **Technical Details**: 이 알고리즘은 Q-learning을 고급 의사결정자(high-level decision-maker)로 사용하며, 기본 샘플링 방법인 RRT 및 RRT*보다 3배 시간 효율성을 개선했습니다. 2D부터 6D까지 다양한 환경에서 강력하고 안정적인 성공률을 유지합니다.

- **Performance Highlights**: 이 방법은 BIT* 및 Informed RRT*와 같은 휴리스틱 기반 방법보다 평균 1.5배 빠른 런타임을 기록했으며, NeuralRRT* 및 MPNetSMP와 같은 학습 기반 방법과 비교할 때 평균 1.5배 더 나은 성능을 보였습니다.



### DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion (https://arxiv.org/abs/2410.24203)
Comments:
          NeurIPS2024, Project: this https URL Code: this https URL

- **What's New**: Diffusion 기반 방법론이 2D 이미지나 3D 물체 생성에서 놀라운 성과를 이뤘으나, 3D 장면 및 360도 이미지를 생성하는 데에는 제약이 많았습니다. 이 문제를 해결하기 위해 대규모의 파노라마 비디오-텍스트 데이터셋을 구축하고, 이를 기반으로 텍스트 기반의 파노라마 생성 프레임워크 DiffPano를 제안합니다.

- **Technical Details**: DiffPano는 파노라마 비디오-텍스트 데이터셋을 활용하여 단일 뷰 텍스트-파노라마 확산 모델을 LoRA로 미세 조정하고, 구형 에피폴라 인식 다중 뷰 확산 모델을 설계하여 생성된 파노라마 이미지의 다중 뷰 일관성을 보장하도록 합니다. 이 모델은 텍스트 설명과 카메라 포즈를 입력받아 스케일 가능하고 일관되며 다양한 파노라마 이미지를 생성합니다.

- **Performance Highlights**: DiffPano는 기존 방법 대비 더 확장 가능하고 일관된 파노라마 이미지를 생성할 수 있으며, 주어진 보지 않는 텍스트 설명과 카메라 포즈에 대해 만족스러운 결과를 보여주는 강력한 일반화 능력을 보여줍니다.



### Length-Induced Embedding Collapse in Transformer-based Models (https://arxiv.org/abs/2410.24200)
- **What's New**: 이 논문에서는 긴 텍스트에서의 텍스트 임베딩 성능 저하를 설명하는 Length Collapse라는 현상을 발견하였습니다. 이 현상은 긴 텍스트 임베딩이 좁은 공간으로 집합화됨으로써 발생하며, 이는 서로 다른 텍스트 길이 간의 분포 불일치를 초래하여 후속 작업의 성능에 악영향을 미칩니다.

- **Technical Details**: Length Collapse는 이론적으로 self-attention 메커니즘이 저역 통과 필터(low-pass filter)로 작용하며, 긴 시퀀스가 이 필터의 감쇠율을 증가시킨다는 것을 보여줍니다. TempScale은 softmax() 진단에서 온도를 도입하여 이러한 저역 필터의 감쇠율을 높여, 긴 텍스트 입력에 대해 일관된 성능을 강화합니다.

- **Performance Highlights**: Empirically, TempScale을 적용함으로써 기존 임베딩 모델의 성능을 개선할 수 있음을 입증하였고, Massive Text Embedding Benchmark(MTEB)에서 40개 데이터셋에서 최대 0.53% 성능 향상과 LongEmbed에서 4개의 데이터셋에서 최대 0.82% 성능 향상을 기록했습니다.



### Chasing Better Deep Image Priors between Over- and Under-parameterization (https://arxiv.org/abs/2410.24187)
Comments:
          Codes are available at this https URL

- **What's New**: 본 논문에서는 'lottery image prior' (LIP)라는 혁신적인 이미지 우선성을 제안합니다. LIP는 과도하게 매개변수가 주어진 DNN으로부터 훈련된 희소 서브 네트워크를 활용하여 다양한 이미지 역문제를 해결할 수 있는 가능성을 제시합니다.

- **Technical Details**: LIP는 LTH(로터리 티켓 가설)를 기반으로 하여, 과도하게 매개변수화된 DNN에서 효율적으로 훈련된 희소 서브 네트워크를 식별하는 방법을 제안합니다. 이 연구는 DNN의 내재된 희소성을 활용하여 이미지 복원 성능을 개선합니다.

- **Performance Highlights**: LIP 서브 네트워크는 깊은 디코더(deep decoder)보다 우수한 성능을 보이며, 다양한 테스트 이미지 및 복원 작업에서 높은 전이 가능성을 가지고 있습니다. 또한, LIP는 압축 성능에서도 강력한 효과를 발휘합니다.



### DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning (https://arxiv.org/abs/2410.24185)
Comments:
          Project website: this https URL

- **What's New**: 이 연구에서는 humanoid 로봇의 다수의 조작 기술을 자가 학습할 수 있도록 도와주는 대규모 자동 데이터 생성 시스템, DexMimicGen을 소개합니다.

- **Technical Details**: DexMimicGen은 소수의 인간 시연에서 궤적을 합성하여 bimanual dexterous manipulation을 위한 훈련 데이터를 자동으로 생성합니다. 이 시스템은 MimicGen의 개념을 바탕으로 다수의 서브태스크를 유연하게 분할하여, 각 팔이 독립적으로 작업을 수행하면서도 필요한 협조 단계를 수용하도록 설계되었습니다.

- **Performance Highlights**: DexMimicGen을 통해 약 60개의 소스 인간 시연에서 21,000개의 데모를 생성하였고, 이를 실제 캔 정렬 작업에 적용하여 90%의 성공률을 보였습니다.



### Constraint Back-translation Improves Complex Instruction Following of Large Language Models (https://arxiv.org/abs/2410.24175)
Comments:
          14 pages, 6 figures

- **What's New**: 이번 연구에서는 기존 데이터셋에서 내재된 복잡한 제약 조건을 활용하여 고품질 복잡한 지시-응답 데이터셋(CRAB)을 생성하는 새로운 데이터 생성 기법인 constraint back-translation(제약 역번역)을 제안합니다.

- **Technical Details**: 기존의 고품질 지시-응답 쌍을 사용하여, Llama3-70B-Instruct 모델을 통해 응답에서 이미 충족된 제약 조건을 추출하고 이를 지시에 추가하여 복잡한 지시-응답 쌍을 생성합니다. 이 과정에서 비용을 절감하고 데이터 잡음을 줄이는 효과가 있습니다.

- **Performance Highlights**: CRAB 데이터셋으로 후속 훈련을 수행한 결과, 여러 개의 LLM(대형 언어 모델)의 복잡한 지시 따르기 능력이 크게 향상되었습니다. 다양한 평가 지표에서 이전 연구보다 개선된 성과를 보였으며, constraint back-translation 방법이 고품질 데이터를 생성하는 데 유용함을 입증했습니다.



### Leveraging Large Language Models for Code Translation and Software Development in Scientific Computing (https://arxiv.org/abs/2410.24119)
- **What's New**: 본 논문에서는 CodeScribe라는 도구를 개발하여 유산 Fortran 코드베이스를 C++로 변환하는 과정에서 생성 AI(GenAI)의 활용 방안을 제시합니다. 이 도구는 프롬프트 엔지니어링(prompt engineering)과 사용자 감독을 결합하여 효율적인 코드 변환 프로세스를 구축합니다.

- **Technical Details**: CodeScribe는 Fortran에서 C++로의 코드 번역 및 유산 시스템과 현대 C++ 라이브러리 간의 통합을 지원하는 Fortran-C API 생성을 목표로 합니다. Generative AI 모델을 활용하여 복잡한 코드 변환 및 코드베이스 검사(task-specific tools) 작업을 자동화합니다.

- **Performance Highlights**: CodeScribe는 과거의 복잡한 Fortran 코드와 현대의 C++ 코드 간의 효율적인 변환을 지원하며, 개발자는 고급 아키텍처 결정 및 성능 튜닝에 집중할 수 있습니다. 이 도구를 통해 개발 생산성이 향상되며, 진행 중인 코드 번역의 정확도를 높이는 데 기여합니다.



### AIDOVECL: AI-generated Dataset of Outpainted Vehicles for Eye-level Classification and Localization (https://arxiv.org/abs/2410.24116)
Comments:
          19 pages, 4 figures, 3 tables

- **What's New**: 본 연구에서는 이미지 라벨링의 효율성을 높이는 새로운 접근법을 제시합니다. 이 방법은 outpainting을 활용하여 인공적인 컨텍스트와 주석을 생성함으로써 수작업 라벨링 노력을 상당히 줄입니다.

- **Technical Details**: 자율 주행, 도시 계획 및 환경 모니터링과 같은 분야에서의 데이터 부족 문제를 해결하기 위해 AI로 생성된 차량 이미지를 활용하였습니다. 이 데이터셋은 선별된 시드 이미지에서 차량을 탐지하고 크롭한 후 큰 캔버스에 outpainting하여 다양한 현실 조건을 시뮬레이션합니다. 각 이미지는 자동 주석을 통해 상세한 바운딩 박스 좌표 정보를 포함하고 있습니다.

- **Performance Highlights**: Outpainted 차량을 활용한 데이터 증강은 전체 성능 지표를 최대 8% 향상시키고, 저대표 클래스의 예측을 최대 20% 증대시켰습니다. 이 연구는 outpainting을 자가 주석화 접근법으로 활용하여 여러 머신 러닝 도메인에서 데이터셋의 활용성을 높이는 솔루션을 제안합니다.



### Nearest Neighbor Normalization Improves Multimodal Retrieva (https://arxiv.org/abs/2410.24114)
- **What's New**: 이 논문에서는 추가 훈련 없이 훈련된 대비 이미지-텍스트 검색 모델의 오류를 수정하는 간단하고 효율적인 방법인 Nearest Neighbor Normalization (NNN)을 소개합니다.

- **Technical Details**: NNN은 k개의 가장 가까운 쿼리 임베딩을 사용하여 검색 후보의 점수를 정규화하여, 고차원 임베딩 공간에서 허브 문제를 완화합니다. NNN은 사전 훈련된 모델을 사용하여 하위 선형 시간 복잡도로 작동하며, 참조 데이터베이스에 대한 추가 훈련이 필요하지 않습니다.

- **Performance Highlights**: NNN을 사용하여 CLIP, BLIP, ALBEF, SigLIP, BEiT 등 다양한 대비 모델에서 텍스트 검색 및 이미지 검색 모두에서 검색 메트릭의 향상을 보여주었습니다. 또한, NNN은 성별 편향과 같은 해로운 편향을 줄일 수 있습니다.



### Reinforcement Learning Gradients as Vitamin for Online Finetuning Decision Transformers (https://arxiv.org/abs/2410.24108)
Comments:
          Accepted as NeurIPS 2024 spotlight. 33 pages, 26 figures

- **What's New**: 이번 연구에서는 Decision Transformers (DTs)의 온라인 파인튜닝 (fine-tuning)에 대한 이론적 분석을 제공하고, 낮은 보상 데이터를 사전 훈련한 온라인 결정 변환기 (Online Decision Transformer, ODT)의 부족한 성능을 개선하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 연구에서는 통상적으로 사용되는 Return-To-Go (RTG)가 예상되는 보상에서 멀어질 경우 온라인 파인튜닝 과정에서 성능 저하를 유발하는 것을 보여줍니다. 이를 해결하기 위해 TD3 (Twin Delayed Deep Deterministic Policy Gradient) 기울기를 ODT의 파인튜닝 프로세스에 추가하는 방법을 사용하였습니다. MDP (Markov Decision Process) 기반의 에이전트를 통해 다양한 환경에서 실험을 진행하여 TD3 기울기가 ODT의 온라인 파인튜닝 성능을 향상시키는 데 효과적임을 밝혀냈습니다.

- **Performance Highlights**: ODT는 낮은 보상 오프라인 데이터로 사전 훈련된 경우에도 TD3 기울기가 파인튜닝 과정에서 놀라운 효과를 나타내며, 기존의 온라인 결정 변환기 접근법에 비해 성능이 크게 개선됨을 실험을 통해 입증하였습니다.



### 3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing (https://arxiv.org/abs/2410.24091)
Comments:
          Accepted at Conference on Robot Learning (CoRL) 2024

- **What's New**: 이 논문에서는 3D-ViTac이라는 다중 모달 감지 및 학습 시스템을 소개합니다. 이는 로봇의 정밀한 조작 능력을 향상시키기 위해 설계되었습니다. 3D-ViTac은 비용 효율적이며 유연한 촉각 센서를 특징으로 하며, 고도의 밀집된 감지 단위를 통해 시각 정보를 보완합니다.

- **Technical Details**: 3D-ViTac 시스템은 로봇 엔드 이펙터의 넓은 면적을 커버하는 밀집형 유연한 촉각 센서 배열을 사용하여, 16×16 해상도를 가진 1024 개의 촉각 감지 단위를 포함합니다. 이 시스템은 촉각 데이터를 3D 비주얼 포인트와 결합하여 통합된 3D 공간으로 변환하고, 이를 통해 모방 학습을 위한 정책을 구현합니다.

- **Performance Highlights**: 실험 결과, 3D-ViTac은 시각 정보만 사용하는 정책보다 정밀한 조작을 수행하며, 특히 fragile items에 대한 안전한 상호작용 및 손 안에서의 작업을 수행하는 장기 과제에서 뛰어난 성능을 보였습니다. 이 연구는 저비용 로봇도 정밀한 조작을 가능하게 하며, 촉각 정보가 강한 시각 차단이 있을 때 더욱 중요한 역할을 한다는 것을 보여줍니다.



### In-Context Fine-Tuning for Time-Series Foundation Models (https://arxiv.org/abs/2410.24087)
- **What's New**: 본 연구에서는 시간 시계열 기반의 Foundation 모델이 'in-context fine-tuning'을 통해 여러 시계열 예제를 활용하여 미래의 특정 시계열을 예측하는 방법론을 제시합니다.

- **Technical Details**: 제안된 모델은 미리 훈련된 Foundation 모델을 바탕으로 하며, 시계열 예측을 위해 목표 시계열의 이력뿐만 아니라 유사한 다른 시계열 예제들을 맥락 창(context window)에 포함시켜 사용하는 것이 특징입니다. 이러한 접근 방식은 모델이 특정 도메인의 데이터 분포에 보다 적합하도록 돕습니다.

- **Performance Highlights**: 우리의 실험 결과, 제안된 'in-context fine-tuning' 접근 방식은 상관 관계가 있는 예제를 활용하여 기존의 감독 학습 방법, 통계 모델 및 다른 시계열 Foundation 모델보다 25% 더 나은 성능을 보입니다. 또한, 목표 데이터셋에 대해 명시적으로 fine-tuning된 Foundation 모델의 성능조차도 약간 초과합니다.



### Dynamical similarity analysis uniquely captures how computations develop in RNNs (https://arxiv.org/abs/2410.24070)
- **What's New**: 이 논문에서는 반복 신경망(RNNs)에서의 동적 표현 정렬(metric) 평가를 위한 테스트 케이스를 제안합니다. 또한, 최근 제안된 Dynamical Similarity Analysis(DSA)가 노이즈에 대한 내구성이 뛰어나고 행동적으로 중요한 표현을 신뢰성 있게 식별한다고 보고합니다.

- **Technical Details**: 저자들은 RNN과 시뮬레이션된 어트랙터를 기반으로 두 가지 테스트 케이스를 구축하고, Procrustes 변환, Centered Kernel Alignment (CKA), DSA의 세 가지 동적 표현 비교 메트릭을 비교합니다. DSA는 어트랙터 테스트 케이스에서 더 나은 노이즈 내구성 및 비율과 같은 반응을 보였으며, RNN 기반의 테스트 케이스에서는 조합 학습을 올바르게 반영하는 유일한 지표로 나타났습니다.

- **Performance Highlights**: DSA는 RNN의 진행 중인 계산을 식별하는 데 매우 효과적이며, Mamba와 같은 현대적 상태 공간 모델에서도 학습 중 회귀 역학의 변화를 필요로 하지 않는다는 점을 보여줍니다.



### Identifying General Mechanism Shifts in Linear Causal Representations (https://arxiv.org/abs/2410.24059)
Comments:
          NeuIPS 2024

- **What's New**: 이 논문은 선형 구조 인과 모델에서 대칭적인 개입 방법을 허용하여 잠재 요인(unknown latent factors)의 변화가 있는 노드를 식별할 수 있는 새로운 접근 방식을 소개합니다. 또한 완벽한 단일 노드 개입이 아닌 보다 일반적인 개입을 통해 이러한 노드를 식별할 수 있는 가능성을 제시합니다.

- **Technical Details**: 기존 연구는 각 잠재 변수에 대한 완벽한 개입을 요구했지만, 이 연구에서는 더 일반적인 개입(soft and hard interventions)과 잠재 요인의 수보다 적은 환경을 사용하여도 변화된 노드를 식별하는 방법을 제안합니다. 논문에서는 필요 충분 조건을 제공하고, 관찰된 데이터를 사용하여 이 조건을 확인할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: 연구는 합성 실험(synthetic experiments)과 심리 측정 데이터셋(psychometric dataset)에서의 결과를 통해 아이덴티피케이션(identifiability)과 알고리즘의 유효성을 검증하였습니다. 특히, 모든 개입 분포(interventional distributions)를 알지 못할 경우에도 잠재 요인의 변화를 추정할 수 있음을 보여주었습니다.



### State- and context-dependent robotic manipulation and grasping via uncertainty-aware imitation learning (https://arxiv.org/abs/2410.24035)
- **What's New**: 이번 연구에서는 컨텍스트 적응형 조작 및 잡기 행동을 생성하기 위한 새로운 LfD(학습을 통한 시연) 접근 방식을 제안합니다. 전통적인 알고리즘의 한계를 극복하고, 로봇 환경에 적응할 수 있는 정책을 개발하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구는 고차원 입력에 대한 경량화된 정책 표현을 학습하기 위해 Kernelized Movement Primitives (KMP)를 사용합니다. KMP는 외부 컨텍스트 변수를 포함한 커널 기반 함수 근사화로 볼 수 있으며, 이를 통해 로봇 상태와 외부 변수 간의 공동 분포를 추정합니다. 또한, 불확실성을 정량화하여 로봇이 이전 시연으로 자연스럽게 복귀하도록 하고 바람직하지 않은 행동을 방지하는 안정적인 정책을 개발합니다.

- **Performance Highlights**: 본 방법은 LASA 손글씨 데이터셋을 통해 평가되었으며, 실제 7DoF 로봇을 두 가지 시나리오에서 테스트했습니다. 특히, 변형 가능한 식품 항목을 잡고 조작하는 과정에서 마찰에 적응하는 능력이 두드러졌습니다. 연구 결과는 조작이 변경되는 컨텍스트에 부드럽게 적응하면서, 시연된 경로를 안정적으로 재현할 수 있음을 입증했습니다.



### Navigating the Unknown: A Chat-Based Collaborative Interface for Personalized Exploratory Tasks (https://arxiv.org/abs/2410.24032)
- **What's New**: 이 논문에서는 개인화된 탐색을 촉진하는 협업 도우미 시스템인 CARE를 소개합니다. CARE는 다중 에이전트 LLM 프레임워크와 구조화된 사용자 인터페이스를 결합하여 탐색 작업에서 개인화의 한계를 극복하려는 시도를 합니다.

- **Technical Details**: CARE 시스템은 세 가지 주요 요소인 채팅 패널(Chat Panel), 솔루션 패널(Solution Panel), 요구 사항 패널(Needs Panel)로 구성되어 있습니다. 이 인터페이스는 사용자 쿼리를 반복적으로 다듬을 수 있는 동적 상호작용을 가능하게 하며, 다중 에이전트 협업 프레임워크를 통해 사용자 요구 사항을 식별하고 맞춤형 솔루션을 생성합니다.

- **Performance Highlights**: 22명의 참가자를 대상으로 한 연구에서 CARE는 기존 LLM 챗봇보다 일관되게 선호되었으며, 참가자들은 CARE가 인지적 부담을 줄이고 창의적인 탐색을 유도하며 개인화된 솔루션을 제공하는 데 도움을 준다고 평가했습니다.



### A Multi-Modal Approach for Face Anti-Spoofing in Non-Calibrated Systems using Disparity Maps (https://arxiv.org/abs/2410.24031)
- **What's New**: 논문은 비보정(non-calibrated) 얼굴 인식 시스템에서 속임수 공격을 탐지하기 위한 새로운 방법론을 제안합니다. 특히, 3D 구조의 복잡성을 고려하여 얼굴 속성을 활용해 상대 깊이를 추정하는 모델을 도입했습니다.

- **Technical Details**: 제안된 모델은 Disparity Model로, 얼굴 속성에서 생성된 disparity map을 세 번째 모달리티로 추가하여 두 개의 NIR 모달리티와 함께 작동합니다. 이 다중 모달리티 접근법은 깊이 정보를 제공하지 않는 비보정 시스템에서 속임수 공격 감지를 가능하게 합니다.

- **Performance Highlights**: 비교 방법에 비해 2.45%와 7.94% 낮은 Equal Error Rate (EER)와 False Negative Rate (FNR) 성능을 기록하여, 각각 1.71%의 EER 및 2.77%의 FNR을 달성했습니다. 3D 속임수 공격에 대한 모델 앙상블에서도 2.04%의 EER과 3.83%의 FNR을 기록했습니다.



### Assessing the Impact of Packing on Machine Learning-Based Malware Detection and Classification Systems (https://arxiv.org/abs/2410.24017)
- **What's New**: 본 연구는 패킹(packing) 기술이 정적 머신러닝(static machine learning) 기반의 악성 코드 탐지 및 분류 시스템에 미치는 영향을 조사하여, 현재의 시스템의 한계점을 드러내고 악성 코드 작성자의 진화하는 전술을 효과적으로 대응할 필요성을 강조합니다.

- **Technical Details**: 연구는 다양한 패킹 기술과 이들이 머신러닝 기반 탐지기 및 분류기의 성능에 미치는 영향을 포괄적으로 분석합니다. 8종의 주요 패커(packer)와 난독화 도구(obfuscation tools)들이 정적 머신러닝 기반 악성 코드 탐지기 및 분류기의 성능에 미치는 영향을 실험을 통해 검토하며, 결과적으로 특정 패킹 루틴이 '선의' 또는 '악의' 행동의 지표로 기능할 수 있음을 밝혀냅니다.

- **Performance Highlights**: 이 연구의 발견은 현재의 정적 탐지 및 분류 시스템의 한계를 강조하며, 패킹 기술이 탐지를 우회하거나 잘못된 경고를 발생시키는 시나리오를 도출합니다. 정적 머신러닝 기반 탐지기가 특정 패킹 방식에 의존하고 있음을 발견하였으며, 이는 탐지 성능과 신뢰성에 중대한 영향을 미칠 수 있습니다.



### An Information Criterion for Controlled Disentanglement of Multimodal Data (https://arxiv.org/abs/2410.23996)
- **What's New**: 이번 논문에서는 Disentangled Self-Supervised Learning (DisentangledSSL)이라는 새로운 자기 지도 학습(self-supervised learning) 접근법을 제안하여 다중 모달(multimodal) 데이터에서 공유 정보와 모달 특화(modality-specific) 정보를 효과적으로 분리하는 방법을 탐구합니다. 이는 기존 연구에서 다루지 않았던, 'Minimum Necessary Information (MNI)' 포인트를 달성할 수 없는 시나리오에 대한 최적성(optimality) 분석을 포함하고 있습니다.

- **Technical Details**: DisentangledSSL은 정보 이론(information theory)의 원칙을 바탕으로 고안된 단계별 최적화 전략을 통해 각 단일 모달의 특성을 유지하면서 학습되며, 쌍으로 관측된 데이터를 통해 공유 및 모달 특화 정보를 효과적으로 분리합니다. 또한, MNI가 도달할 수 없는 경우에도 최적의 분리를 달성할 수 있다는 이론적 보장을 제공합니다.

- **Performance Highlights**: 이 연구는 다양한 합성(synthetic) 데이터 및 실제(real-world) 다중 모달 데이터셋에서 DisentangledSSL의 효과를 입증하며, 비전-언어 데이터(vision-language data)에 대한 예측(prediction) 작업과 생물학적 데이터에서의 분자-표현형(molecule-phenotype) 검색 작업에서 기존 기준선(baseline)보다 일관되게 우수한 성능을 보입니다.



### Localization, balance and affinity: a stronger multifaceted collaborative salient object detector in remote sensing images (https://arxiv.org/abs/2410.23991)
- **What's New**: 본 연구에서는 Optical Remote Sensing Images (ORSI)에서의 Salient Object Detection (SOD)을 위한 새로운 방법인 LBA-MCNet을 제안합니다. 이 방법은 localization, balance, affinity 측면을 통합하여 복잡한 경계 구조와 문맥 관계를 효과적으로 처리합니다.

- **Technical Details**: LBA-MCNet은 Edge Feature Adaptive Balancing and Adjusting (EFABA) 모듈을 통해 정밀한 경계 위치 지정을 수행하고, Global Distributed Affinity Learning (GDAL) 모듈을 통해 전역 문맥을 모델링합니다. EFABA는 경계 정보를 강조하고, GDAL은 최종 레이어에서 반복되는 affinity map을 생성하여 전역 패턴을 포착합니다.

- **Performance Highlights**: 연구에서는 3개의 공개 데이터셋에서 28개의 최신 SOD 방식과 비교하였으며, LBA-MCNet의 성능이 월등함을 입증하였습니다.



### Image Synthesis with Class-Aware Semantic Diffusion Models for Surgical Scene Segmentation (https://arxiv.org/abs/2410.23962)
- **What's New**: 본 논문에서는 Class-Aware Semantic Diffusion Model (CASDM)를 제안하여 외과 장면(segmentation) 생성에서 데이터 부족과 불균형 문제를 해결합니다. 이 모델은 병합된 세그멘테이션 맵을 사용하여 이미지 생성을 위한 새로운 접근 방식을 탑재하여, 작은 크기의 중요한 조직 클래스를 우선 고려합니다.

- **Technical Details**: CASDM은 세분화 맵을 조건으로 사용하여 이미지 생성을 수행합니다. 이 모델은 novel class-aware mean squared error와 class-aware self-perceptual loss 함수를 정의하여, 덜 보이는 중요한 클래스를 우선으로 하여 이미지 품질을 개선합니다. 또한, 텍스트 프롬프트를 통해 여러 클래스를 포함한 세그멘테이션 맵을 생성하는 최초의 접근 방식을 통해 모델의 전반적인 성능을 향상시킵니다.

- **Performance Highlights**: CASDM은 다양한 데이터셋에서 평가된 결과에서, 특히 작은 클래스와 희귀한 클래스의 세그멘테이션 정확도를 크게 향상시킴을 보여줍니다. CholecSeg8K 및 gastrectomy SISVSE 데이터셋에서 실험을 실시하였으며, 이미지 품질과 다운스트림 세그멘테이션 성능 모두에서 유의미한 개선을 입증했습니다.



### Representative Social Choice: From Learning Theory to AI Alignmen (https://arxiv.org/abs/2410.23953)
Comments:
          Full version (20 pages). Under review. An excerpt was previously accepted to NeurIPS 2024 Pluralistic Alignment Workshop

- **What's New**: 본 연구는 집단적 의사결정에서 민주적 대표성을 모델링하기 위한 대표 사회 선택(framework) 프레임워크를 제안합니다. 이는 문제와 개인의 수가 너무 많아 메커니즘이 모든 선호를 직접 고려할 수 없는 시나리오를 다루고 있습니다.

- **Technical Details**: 대표 사회 선택에서 모집단은 개인-문제 쌍의 유한 샘플로 표현되며, 여기서 사회 선택 결정이 이루어집니다. 본 논문에서는 대표 사회 선택의 복잡성을 수학적 통계학적(statistical) 학습 문제로 모델링하고, 기계 학습의 이론을 활용하여 일반화(generalization) 특성을 증명합니다.

- **Performance Highlights**: 대표적인 의사결정 상황인 배심원 재판, 입법 과정, 기업 거버넌스 및 최근의 AI 정렬 과정에서 적용 가능성이 있습니다. 이 연구는 사회 선택 이론과 AI 정렬의 교차점에서 새로운 연구 방향을 열어줍니다.



### Towards Fast Algorithms for the Preference Consistency Problem Based on Hierarchical Models (https://arxiv.org/abs/2410.23934)
Comments:
          Longer Version of IJCAI'16 publication this https URL

- **What's New**: 이 논문에서는 위계적 모델(hierarchical models)에 기반한 선호(deviation) 일관성 문제(Preference Consistency Problem, PCP)를 해결하기 위한 여러 알고리즘적 접근 방식을 제안하고 비교합니다. 특정 대안들에 대한 선호 진술이 서로 모순되지 않는지를 결정하는 문제에 초점을 두고 있습니다.

- **Technical Details**: PCP는 NP-완전(NP-complete) 문제로, 혼합 정수 선형 프로그래밍(Mixed Integer Linear Programming, MILP)과 두 개의 재귀 알고리즘을 사용하여 이 문제를 해결하는 접근 방식을 개발하였습니다. 제안된 알고리즘들은 문제의 속성을 활용하여 검색 공간을 축소(pruning)하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 재귀 알고리즘들이 MILP 방식보다 훨씬 빠른 실행 시간(Running Times)을 보였으며, 특히 실행 시간의 비율은 아주 빠르게 증가하는 경향을 보였습니다. 제공된 실험 데이터는 합성 데이터(synthetic data)를 포함하고 있습니다.



### BitStack: Fine-Grained Size Control for Compressed Large Language Models in Variable Memory Environments (https://arxiv.org/abs/2410.23918)
- **What's New**: 이 논문에서는 BitStack이라는 새로운 비훈련(Training-free) 기반의 가중치 압축 방법을 제안합니다. BitStack은 메모리 사용량과 모델 성능 간의 메가바이트 수준의 균형을 가능하게 하여, 전통적인 압축 방법이 해결하지 못한 문제를 다루고 있습니다.

- **Technical Details**: BitStack은 가중치 행렬을 동적으로 조정하며, SVD(Singular Value Decomposition)를 사용하여 각 파라미터의 중요성을 고려하여 가중치 매트릭스와 잔여 블록을 반복적으로 분해합니다. 이 과정에서 약 1비트의 메모리를 생성하며, 중요도에 따라 잔여 블록을 정렬 및 저장하여 현재 메모리 상황에 맞게 로드할 수 있습니다.

- **Performance Highlights**: 광범위한 실험에서 BitStack은 다양한 작업에서 정밀한 크기 조절에도 불구하고, 특정한 압축 비율에서 전통적인 양자화 기준을 지속적으로 초과하는 성과를 나타냈습니다. 이는 BitStack이 가중치 분해 방법과 양자화 기반 방법 간의 성능 격차를 해소했다는 것을 의미합니다.



### Transformer-based Model Predictive Control: Trajectory Optimization via Sequence Modeling (https://arxiv.org/abs/2410.23916)
Comments:
          8 pages, 7 figures. Datasets, videos and code available at: this https URL

- **What's New**: 본 논문에서는 모델 기반의 최적화 방법과 학습 기반의 방법을 결합한 통합 프레임워크를 제안하여, 자율 로봇의 궤적(trajectory) 생성 성능을 향상시키는 방법을 소개합니다. 특히, 이 프레임워크는 고용량의 transformer 기반 신경망 모델을 최적화 과정에 내장하여 비볼록(non-convex) 최적화 문제에 대한 근사 최적 시작점을 제공합니다.

- **Technical Details**: 제안된 접근법은 사전 훈련(pre-training)과 미세 조정(fine-tuning) 전략을 사용하여 transformer를 학습시켜 거의 최적의 상태 및 제어 시퀀스를 생성합니다. 이를 통해 최적화는 근사 최적의 초기 추정치로 시작하게 되고, 짧은 과거의 문제에 대한 장기 지침을 제공하여 최적화 과정에서의 비용 항이나 제약의 비싼 튜닝을 피할 수 있습니다. 또한, 학습 기반 가이드를 MPC에 삽입함으로써 성능 저하를 크게 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 비단순한 최적화 방식에 비해 궤적 생성 성능을 최대 75% 향상시키고, 해법 반복 횟수를 최대 45% 줄이며, 전체 MPC 실행 시간을 7배 개선했습니다. 이러한 성능 향상은 실험이 시뮬레이션뿐만 아니라 실제 자유 비행 플랫폼에서도 이루어졌음을 보여줍니다.



### Efficient Inference and Computation of Optimal Alternatives for Preference Languages Based On Lexicographic Models (https://arxiv.org/abs/2410.23913)
Comments:
          Longer Version of IJCAI'17 publication this https URL

- **What's New**: 본 논문에서는 lexicographic 모델에 기반한 일반적인 선호 언어에 대한 일관성(constistency) 분석을 통해 선호 추론(preference inference)을 조사합니다. 여러 자연 유형의 선호 진술에 적용될 수 있는 강력한 조합성(strong compositionality)이라는 특성을 정의하고, 이를 통해 선호 진술 집합의 일관성을 결정하는 탐욕스러운(greedy) 알고리즘을 제시합니다.

- **Technical Details**: 우리는 LpqT라는 특정 선호 언어에 대해 일관성을 테스트(testing consistency)하는 것이 다항식(polynomial) 시간 내에 가능함을 보여줍니다. 이 언어는 엄격(strict) 및 비엄격(non-strict) 진술, 결과와 부분 튜플 간의 비교, ceteris paribus 및 강한 진술을 모두 허용합니다. 선호 집합의 최적(optimal) 유형 또한 다항식으로 계산할 수 있으며, 이는 실험 결과로 뒷받침됩니다.

- **Performance Highlights**: 알고리즘의 효율성을 실험적으로 검증하고, ℒp⁢q⁢T′라는 언어에 대해 다양한 최적 개념들(optimality notions) 간의 관계를 수립하여 성능의 우수성을 입증했습니다.



### AllClear: A Comprehensive Dataset and Benchmark for Cloud Removal in Satellite Imagery (https://arxiv.org/abs/2410.23891)
Comments:
          Accepted at NeurIPS 2024 Datasets and Benchmarks Track. Code and data available at this https URL

- **What's New**: 본 논문에서는 클라우드 제거(cloud removal)를 위한 세계 최대의 공개 데이터셋인 AllClear를 소개합니다. 이 데이터셋은 23,742개 지역(ROIs)에서 4백만 이미지로 구성되어 있으며, 다양한 땅 사용 패턴을 포함합니다.

- **Technical Details**: AllClear 데이터셋은 2022년에 수집된 다수의 다중 스펙트럼(multi-spectral) 및 합성 개구 레이더(SAR) 이미지를 포함합니다. 데이터셋은 모델이 결측 정보를 보완할 수 있도록 다수의 위성(Sentinel-1, 2 및 Landsat 8/9)의 데이터를 사용합니다. 성능 평가 결과, 데이터 양이 30배 증가할 때 PSNR(peak signal-to-noise ratio)은 28.47에서 33.87로 상승함을 보여주었습니다.

- **Performance Highlights**: 기존 최첨단(cloud removal) 모델들이 AllClear 데이터셋에서 충분히 훈련되지 않음을 발견하였으며, 보다 크고 다양한 데이터셋을 사용하여 성능이 크게 향상되었습니다. 여러 센서 및 긴 시간의 관측을 활용하는 모델이 훨씬 더 좋은 성능을 보임을 확인했습니다.



### Leveraging LLMs for MT in Crisis Scenarios: a blueprint for low-resource languages (https://arxiv.org/abs/2410.23890)
Comments:
          arXiv admin note: text overlap with arXiv:2403.02370, arXiv:2403.01580

- **What's New**: 본 연구는 위기 상황에서 저자원 언어에 대한 기계 번역 (Machine Translation, MT) 시스템을 강화하기 위해 대형 언어 모델 (Large Language Models, LLMs) 및 다국어 LLM (Multilingual LLMs, MLLMs)을 활용하는 새로운 접근 방식을 제시합니다. 이 논문은 코로나19 팬데믹을 모델로 하여 위기 대응을 위한 MT 시스템 개발 및 평가를 포함한 프로세스를 설명합니다.

- **Technical Details**: 연구에서는 LLM 및 MLLM을 조정 (fine-tuning)하고, 커뮤니티 주도의 말뭉치 (corpus) 개발 전략을 사용하여 저자원 언어 쌍을 위한 맞춤형 MT 시스템을 개발합니다. LLMs은 언어 이해 및 복잡한 응답 생성 능력 덕분에 인간의 의사소통과 생산성을 향상시킬 수 있습니다. MLLM의 파라미터 조정은 특정 작업의 성능을 향상시키기 위해 미리 훈련된 모델의 하이퍼파라미터를 조정하는 과정이며, 이는 훈련 데이터 및 최적화 기법에 따라 달라집니다.

- **Performance Highlights**: 조정된 MLLM 모델이 LLM 모델보다 우수한 성능을 제공하며, 코로나19와 같은 위기상황에서 신속하고 품질 높은 MT 시스템을 개발하는 확장 가능하고 복제 가능한 모델을 제시합니다. 연구에서는 맞춤형 GPT 시스템과 NLLB에 적응된 MLLM 모델을 비교하였고, 커뮤니티 참여가 매우 특화된 위기 특정 데이터셋의 생성에 중요한 역할을 한다고 강조합니다.



### GEPS: Boosting Generalization in Parametric PDE Neural Solvers through Adaptive Conditioning (https://arxiv.org/abs/2410.23889)
- **What's New**: 이 논문에서는 매개변수가 변화하는 파라메트릭 부분 미분 방정식(parametric PDEs)을 해결하기 위한 새로운 데이터 기반 접근 방식을 제안합니다. 특히, 적응 조건(adaptive conditioning) 메커니즘을 사용하여 신경망 기반의 솔버가 새로운 환경에서의 일반화를 수행할 수 있도록 합니다.

- **Technical Details**: GEPS(Generalization in PDE Solvers)는 첫 번째 순서의 최적화 및 저랭크(low-rank) 빠른 적응을 통한 단순한 적응 메커니즘을 제시합니다. 이 방법은 데이터 기반 및 물리적 인식을 모두 아우르는 네트워크 아키텍처에 적용될 수 있으며, 다양한 물리적 동역학을 처리할 수 있습니다.

- **Performance Highlights**: GEPS는 다양한 초기 조건, PDE 계수, 강제항 및 솔루션 도메인 변화를 포함하는 실험에서 매우 우수한 성능을 보였으며, 미지의 조건에 대한 일반화에서도 효과적임을 입증하였습니다.



### 'No' Matters: Out-of-Distribution Detection in Multimodality Long Dialogu (https://arxiv.org/abs/2410.23883)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구에서는 대화(Dialogue)와 이미지(Image) 입력에서 발생하는 아웃 오브 디스트리뷰션(Out-of-Distribution, OOD) 감지 문제를 해결하기 위해 새로운 방법론인 대화 이미지 정렬 및 향상 프레임워크(Dialogue Image Aligning and Enhancing Framework, DIAEF)를 제안합니다.

- **Technical Details**: DIAEF 프레임워크는 이미지와 대화 모드에서 비정상적 쿼리를 효과적으로 감지하기 위해 두 가지 주요 시나리오에서 OOD를 탐지하는 점수 설계를 포함하고 있습니다: 1) 대화와 이미지 입력 쌍 간의 불일치, 2) 이미 확인되지 않은 라벨을 가진 입력 쌍. 이 프레임워크는 대화 시스템에서 긴 다중 대화 시나리오를 처리하는 데 필요한 혁신적인 접근을 제공합니다.

- **Performance Highlights**: 실험 결과, DIAEF를 적용한 다중 라운드 대화 시스템은 이전에 보지 못했던 라벨에 대해 OOD 탐지가 효과적임을 보여주며, 불일치 쌍에서도 강력한 견고성을 입증합니다. 이 연구는 향후 멀티모달 대화 시스템의 기초적인 기준과 방법론을 설정하고 사용자 경험을 향상시킵니다.



### RAGraph: A General Retrieval-Augmented Graph Learning Framework (https://arxiv.org/abs/2410.23855)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 새로운 프레임워크인 General Retrieval-Augmented Graph Learning (RAGraph)을 소개합니다. 이 프레임워크는 외부 그래프 데이터를 일반 그래프 기본 모델에 통합하여 학습되지 않은 시나리오에서 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: RAGraph는 Toy Graph Vector Library를 기반으로 하며, 여기서는 기능과 작업 특정 레이블 정보와 같은 핵심 속성을 캡처합니다. 추론 과정에서 RAGraph는 다운스트림 작업의 주요 유사성을 바탕으로 유사한 토이 그래프를 적절히 검색하고, 이를 메시지 패싱 프롬프트 메커니즘을 통해 학습 맥락을 풍부하게 통합합니다.

- **Performance Highlights**: 광범위한 실험 평가 결과, RAGraph는 동적 및 정적 데이터셋에서 노드 분류, 링크 예측, 그래프 분류와 같은 여러 작업에서 최신 그래프 학습 방법론을 크게 능가하는 성능을 보여주었습니다. 또한, RAGraph는 작업 특정 파인 튜닝(task-specific fine-tuning) 없이도 높은 성능을 지속적으로 유지하는 것으로 확인되어, 높은 적응성, 강건성 및 광범위한 적용 가능성을 강조합니다.



### Commonsense Knowledge Editing Based on Free-Text in LLMs (https://arxiv.org/abs/2410.23844)
Comments:
          11 pages, 8 figures

- **What's New**: 이 연구는 자유 텍스트 기반의 상식 지식 편집을 위한 새로운 방법(Knowledge Localization for Free-Text, KLFT)과 동적 인식 편집 방법(Dynamics-aware Editing Method, DEM)을 제안합니다. 기존 방법들은 제한적인 단일 토큰 또는 엔티티에 대해 편집을 수행했지만, 이 연구는 더 넓은 데이터 구조와 긴 내용을 가진 상식 지식의 편집 가능성을 탐구합니다.

- **Technical Details**: 논문에서 제안한 KLFT 방법은 상식 지식의 저장 장소를 추적하고 회수하는 두 가지 실험을 통해 상식 지식이 MLP 및 Attention 층에 분산되어 저장된다는 것을 보여줍니다. 또한, DEM 방법은 동적 인식 모듈을 사용하여 상식 지식의 저장 위치를 실시간으로 감지하고 특정 층에서 목표 지식을 편집합니다.

- **Performance Highlights**: 실험 결과, DEM 방법은 상식 지식 편집에서 우수한 성능을 달성하였으며, 새롭게 생성된 상식 지식 편집 벤치마크(Commonsense Knowledge Editing Benchmark, CKEBench)는 15,600개의 샘플을 포함하여 기존 데이터셋과 비교해 더 높은 난이도를 자랑합니다.



### Auditing Google's Search Algorithm: Measuring News Diversity Across Brazil, the UK, and the US (https://arxiv.org/abs/2410.23842)
Comments:
          21 pages, 3 figures, 7 tables

- **What's New**: 이 연구는 브라질, 영국, 미국에서 구글의 검색 알고리즘이 뉴스 다양성에 미치는 영향을 분석하였습니다. 구글 시스템이 제한된 수의 뉴스 매체를 선호하는 경향을 설명하며, 알고리즘 감사(algorithm auditing) 기술을 활용하여 소스 집중도를 측정하였습니다.

- **Technical Details**: 이 연구는 Herfindahl-Hirschman Index (HHI)와 Gini 계수를 사용하여 소스의 집중도를 분석합니다. 데이터는 '뉴스' 탭에서 21일 동안 수집된 143,976개의 검색 결과를 포함하며, 2,298개의 검색 쿼리와 4,296개의 매체를 대상으로 하였습니다.

- **Performance Highlights**: 연구 결과는 구글 검색 결과에서 소스의 인기와 정치적 편향, 콘텐츠의 최신성에 따른 편향이 존재함을 보여줍니다. 이 알고리즘은 뉴스 생태계 내 미디어 불평등을 강화할 수 있음을 시사합니다.



### Counterfactual MRI Data Augmentation using Conditional Denoising Diffusion Generative Models (https://arxiv.org/abs/2410.23835)
- **What's New**: 이 연구에서는 조건부 노이즈 제거 확산 생성 모델(cDDGM)을 사용하여 기존 환자 해부학을 변경하지 않고 다양한 이미지 획득 파라미터(IAP)에 따라 마그네틱 레조넌스(MR) 이미지를 생성하는 새로운 방법을 도입하였습니다. 이러한 반사적 이미지 생성은 데이터 증강(data augmentation)의 일환으로 수행되어 DL 모델의 세그멘테이션(segmentation) 정확도를 향상시키는데 기여합니다.

- **Technical Details**: cDDGM은 기존의 DDPM 아키텍처를 기반으로 하며, 다중 클래스에 걸쳐 조건부 컨텍스트를 통해 변화하는 이미지를 생성합니다. 본 연구에서는 Duke-Breast-Cancer-MRI 데이터셋을 활용하여 모델을 훈련하였고, IAP를 예측하기 위해 수정된 ResNet-18 모델을 사용하였습니다. 이를 통해 연속 및 범주형 IAP를 예측하였고, 세그멘테이션 작업을 위해 U-Net을 이용하였습니다.

- **Performance Highlights**: 연구에서 개발된 IAP 예측 모델은 테스트 데이터셋의 IAP를 매우 높은 정확도로 포착하였으며, 특히 연속 변수의 경우 낮은 MSE(mean squared error)로 모든 변수를 추정할 수 있었습니다. 세그멘테이션 모델 평가 결과, 배경, 지방, 섬유선 조직(FGT)에 대한 정확도가 향상되었으며, ID 및 OOD 설정 모두에서 좋은 성과를 보였습니다.



### GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages (https://arxiv.org/abs/2410.23825)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 GlotCC라는 새로운 문서 수준의 대규모 말뭉치를 소개합니다. 이 말뭉치는 CommonCrawl에서 추출되었으며, 1000개 이상의 소수 언어를 포함하여 2TB의 데이터량을 자랑합니다. GlotCC는 오픈소스 기반의 재현 가능한 파이프라인을 통해 생성되었으며, 신뢰할 수 있는 데이터 품질을 유지하기 위해 엄격한 청소 과정이 포함되어 있습니다.

- **Technical Details**: GlotCC는 언어 식별 모델 GlotLID v3.0을 사용하여 데이터를 분류하며, 2000개 이상의 언어 레이블을 커버합니다. Ungoliant 파이프라인을 통해 CommonCrawl에서 웹 텍스트를 효과적으로 추출하고, 다양한 노이즈 제거 기술을 활용하여 품질 높은 데이터를 확보합니다. GlotLID v3.0은 언어 식별 오류를 최소화하기 위해 작성된 스크립트를 인식하며, 기존의 FastText 기반 모델의 한계를 극복했습니다.

- **Performance Highlights**: GlotCC의 무작위 샘플링 653개 언어 하위 코퍼스에 대한 감사 결과, 데이터가 제대로 된 언어로 구성되어 있음을 나타냈으며, 매크로 평균 점수 0.93, 중앙값 점수 1.0을 기록했습니다. 이 연구는 소수 언어 지원을 위한 중요한 기여로 평가되며, 생성적 언어 모델의 사전 학습 및 다양한 언어 기술 개발에 활용될 수 있습니다.



### Parameter-Efficient Fine-Tuning Medical Multimodal Large Language Models for Medical Visual Grounding (https://arxiv.org/abs/2410.23822)
- **What's New**: 이번 연구에서는 의료 비주얼 그라운딩 과제를 위한 파라미터 효율적인 미세 조정 기법 (Parameter-efficient Fine-tuning)을 적용한 의료 다중 모드 대형 언어 모델(MLLM)을 제안합니다. 새로운 모델은 의료 이미지와 텍스트를 효과적으로 통합하여 임상적 결정을 지원하는데 도움을 줍니다.

- **Technical Details**: PFMVG(파라미터 효율적인 미세 조정의 의료 다중 모드 대형 언어 모델)는 ViT(Vision Transformer)를 이미지 인코딩에 사용하며, LLM(대형 언어 모델)을 통해 이미지-텍스트 임베딩을 처리합니다. 미세 조정 과정은 두 단계로 나누어져 있으며, 첫 번째 단계는 의료 이미지에 대한 캡션 생성, 두 번째 단계는 의료 비주얼 그라운딩입니다.

- **Performance Highlights**: MS-CXR 데이터셋에서 PFMVG는 기존의 기준을 뛰어넘는 최첨단 성능을 달성하였고, 특히 8가지 질병 카테고리에서 현저한 의성적 지수(IoU)와 Dice 점수를 기록하여 GPT-4v를 초월했습니다.



### Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models (https://arxiv.org/abs/2410.23820)
- **What's New**: 이번 연구에서는 분산 모델(diffusion models)을 활용한 비지도 분리 표현 학습(disentangled representation learning, DRL)을 다루고 있습니다. 특히, 동적 가우시안 고정(Dynamic Gaussian Anchoring, DyGA) 기법을 통해 속성 별로 분리된 잠재 단위를 구현하고, Skip Dropout 기술을 통해 노이즈가 포함된 U-Net 모델의 특성 추출 기능을 향상시키고 있습니다.

- **Technical Details**: 연구에서는 두 가지 주요 기법을 제안합니다. 첫째, DyGA는 잠재 공간에서 속성 클러스터를 위한 앵커를 동적으로 선택하여 혼란스러운 포인트를 이 앵커 쪽으로 이동시킵니다. 둘째, Skip Dropout은 U-Net의 스킵 연결 기능을 제거하여 DM 훈련이 잠재 단위 특성에 집중하도록 합니다.

- **Performance Highlights**: 제안된 기법은 기존 DM 기반 DRL 방법에 적용하여 SOTA(최첨단) 비지도 DRL 성능을 달성하였으며, 획득된 표현은 다운스트림 작업에서도 우수한 성능을 보입니다. 노이즈 제거 및 이미지 생성 기능에서의 동시 학습을 통해, 더욱 해석 가능한 분리 기능을 제시합니다.



### The NPU-HWC System for the ISCSLP 2024 Inspirational and Convincing Audio Generation Challeng (https://arxiv.org/abs/2410.23815)
Comments:
          accepted by ISCSLP 2024

- **What's New**: 본 논문에서는 ISCSLP 2024 Inspirational and Convincing Audio Generation Challenge 2024 (ICAGC 2024)에 제출된 NPU-HWC 시스템을 소개합니다. 이 시스템은 두 개의 모듈로 구성되어 있으며, 첫 번째 트랙에서는 음성 생성기를, 두 번째 트랙에서는 배경 오디오 생성기를 사용합니다.

- **Technical Details**: 트랙 1에서는 Single-Codec을 사용하여 음성을 이산 토큰으로 토크나이즈하고, zero-shot speaking style cloning을 달성하기 위해 언어 모델 기반 접근법을 활용합니다. Single-Codec은 톤과 스타일을 토큰 수준에서 분리하여 자기회귀 언어 모델의 음향 모델링 부담을 줄입니다. 또한, DSPGAN을 사용하여 16 kHz mel-spectrogram을 고품질 48 kHz 파형으로 업샘플링합니다. 트랙 2에서는 대형 언어 모델(LLMs)을 기반으로 하는 배경 오디오 생성기를 제공합니다.

- **Performance Highlights**: 우리의 제출물은 트랙 1에서 두 번째, 트랙 2에서 첫 번째를 차지하며 평균 Mean Opinion Score (MOS)는 각각 3.63 및 3.67을 기록했습니다.



### CALE: Continuous Arcade Learning Environmen (https://arxiv.org/abs/2410.23810)
- **What's New**: 이번 논문에서는 잘 알려진 Arcade Learning Environment (ALE)의 확장인 Continuous Arcade Learning Environment (CALE)를 소개합니다. CALE는 Atari 2600 게임 시스템 emulation인 Stella를 기반으로 하지만, 연속적 행동(continuous actions)을 지원합니다.

- **Technical Details**: CALE은 18개의 이산 행동 대신 3차원의 연속 행동 공간(continuous action space)을 도입하며, 주어진 환경에서 PPO, SAC, DQN 및 Rainbow와 같은 다양한 에이전트의 벤치마킹과 평가를 가능하게 합니다. 이를 통해 에이전트의 행동 공간과 관련된 도전과제를 이해할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 初期 케이스로 Soft Actor-Critic (SAC) 알고리즘을 통한 기본 성능 결과를 제시하며, 다양한 도메인을 처리할 수 있는 일반화된 에이전트에 대한 추가 연구 필요성을 강조합니다.



### Generative AI for Accessible and Inclusive Extended Reality (https://arxiv.org/abs/2410.23803)
Comments:
          Presented at the CHI 2024 Workshop "Building a Metaverse for All: Opportunities and Challenges for Future Inclusive and Accessible Virtual Environments", May 11, 2024, Honolulu, Hawaii

- **What's New**: 이 논문은 인공지능 생성 콘텐츠(AIGC)가 가상 환경의 접근성과 포용성을 증진하는 데 기여할 가능성에 대해 논의합니다. AIGC는 3D 모델링 전문 지식의 필요성을 줄일 수 있으며, 텍스트와 멀티모달 입력 기반의 도구를 통해 더욱 쉬운 3D 콘텐츠 생성 및 편집을 가능하게 합니다.

- **Technical Details**: AIGC는 텍스트-투-3D(text-to-3D) 생성 기술을 활용해 XR 환경에서의 콘텐츠 생성을 지원합니다. 기존 도구들은 3D 모델 생성을 위한 높은 전문 지식을 요구했으나, AIGC 도구들은 텍스트 입력만으로도 3D 자산을 생성할 수 있는 가능성을 제공합니다. 특정 기술로는 Neural Radiance Fields (NERFS)와 Gaussian Splatting (GS) 등이 있으며, 멀티모달 입력 기법을 통해 사용자는 직관적인 방법으로 3D 객체를 생성하고 조작할 수 있습니다.

- **Performance Highlights**: AIGC 도구들을 통해 접근성이 향상된 영역으로는 특히 시각 또는 운신의 제한이 있는 사용자들이 3D 콘텐츠를 생성하고 조작할 수 있는 부분입니다. 그러나, AIGC 사용에는 여전히 편향(bias) 문제와 신뢰 문제, 메타데이터의 부족이 존재하며, 이는 향후 접근 가능한 XR 환경의 설계에 있어 중요한 도전 과제가 될 것입니다.



### Improving snore detection under limited dataset through harmonic/percussive source separation and convolutional neural networks (https://arxiv.org/abs/2410.23796)
- **What's New**: 이번 연구는 Obstructive Sleep Apnoea Syndrome (OSAS) 환자에서 관찰되는 소음의 특성과 수면 무호흡 증후군을 다루고 있으며, 새로운 방법으로 모노럴(monoral) 코고리 소리와 비코고리 소리를 분리하는 기술을 제안합니다.

- **Technical Details**: 이 방법은 harmonic/percussive sound source separation (HPSS) 기법을 활용하여 소리의 harmonic 성분을 분석합니다. HPSS에서 유도된 harmonic spectrogram을 노이즈 검출을 위한 입력 데이터로 사용하는 방식입니다.

- **Performance Highlights**: 제안된 방식은 데이터 양이 제한된 상황에서도 우수한 성과를 보였습니다. 특히, 제한된 데이터 학습 환경에서 HPSS 기반의 특성이 기존 문헌의 고전적 입력 특성에 비해 모든 네트워크 아키텍처의 성능을 향상시킵니다.



### EDT: An Efficient Diffusion Transformer Framework Inspired by Human-like Sketching (https://arxiv.org/abs/2410.23788)
Comments:
          Xinwang Chen and Ning Liu are with equal contributions. This paper has been accepted by NeurIPS 2024

- **What's New**: 이번 연구에서는 효율적인 Transformer 기반의 Diffusion Probabilistic Model (DPM)을 제안합니다. 이를 통해 기존 CNN 기반 DPM과 비교했을 때 낮은 계산 비용을 실현하고, 이미지 합성 성능에서 SOTA(State Of The Art)를 달성했습니다.

- **Technical Details**: Efficient Diffusion Transformer (EDT) 프레임워크는 가벼운 디자인의 diffusion 모델 아키텍처와 Attention Modulation Matrix (AMM)를 포함합니다. AMM은 인간의 스케치 방식을 모방하여 전역(global) 및 지역(local) attention을 조절함으로써 이미지 합성을 향상시킵니다. 또한, EDT에 맞춘 token relation-enhanced masking training 전략을 제안하여 학습 능력을 강화합니다.

- **Performance Highlights**: EDT는 기존 transformer 기반 diffusion models에 비해 3.93x, 2.84x, 1.92x의 속도 향상을 이루었으며, FID(Fréchet Inception Distance) 수치에서 더 낮은 값을 기록하여 우수한 성능을 보여줍니다.



### Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map (https://arxiv.org/abs/2410.23780)
Comments:
          27 pages, 13 figures

- **What's New**: MapDR라는 새로운 데이터셋을 도입하여 교통 신호에서 주행 규칙을 추출하고 벡터화된 HD 맵과 연계하는 작업에 중점을 두었습니다.

- **Technical Details**: MapDR 데이터셋은 10,000개 이상의 주석이 달린 비디오 클립을 포함하고 있으며, 규제 지침을 정확히 해석하는 'Rule Extraction from Traffic Sign'와 이러한 규칙을 해당 차선과 맞추는 'Rule-Lane Correspondence Reasoning'이라는 두 가지 주요 서브 작업을 정의합니다.

- **Performance Highlights**: 자율주행 기술 발전을 위한 강력한 기준선을 제공하며, 안전한 자율 네비게이션 시스템 개발에 기여할 것으로 기대됩니다.



### The Potential of LLMs in Medical Education: Generating Questions and Answers for Qualification Exams (https://arxiv.org/abs/2410.23769)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 의료 교육 분야에서 의료 자격 시험 문제와 답변을 생성할 수 있는 가능성을 탐구하며, 의료 분야에서의 LLMs 응용의 혁신을 보여줍니다.

- **Technical Details**: 연구진은 실제 세계의 노인 만성 질환에 대한 중국 데이터 세트를 사용하여 LLMs가 제한된 정보를 기반으로 개방형 질문과 답변을 생성할 수 있는지를 평가했습니다. 8개의 인기 있는 LLM(ERNIE 4, ChatGLM 4 등)이 사용되었습니다. LLMs의 생성된 질문은 전문 평가자에 의해 평가되었습니다.

- **Performance Highlights**: 대부분의 LLMs가 제공한 질문의 일관성과 정보 정확성에서 평균 점수가 4를 넘었고, 전문성 측면에서도 4에 가까운 점수를 받았습니다. 하지만 그럼에도 불구하고 질문 생성에 비해 답변의 평균 점수는 낮았으며, LLMs의 답변에서의 주요 정보 부족이 발견되었습니다.



### Enhancing Chess Reinforcement Learning with Graph Representation (https://arxiv.org/abs/2410.23753)
- **What's New**: 이번 논문에서는 Chess 게임을 위한 새로운 아키텍처인 Graph Neural Networks (GNN)를 기반으로 한 AlphaGateau 모델을 제안합니다. 이 모델은 grid-based (격자 기반) 구조 대신에 그래프 표현을 사용하여 더 다양한 게임 변형에 적응할 수 있는 가능성을 제공합니다.

- **Technical Details**: 기존의 Convolutional Neural Networks (CNN) 대신 GNN을 사용하여 게임 상태를 표현합니다. 또한, 기존의 Graph Attention Network (GAT) 레이어를 확장하여 edge-features (엣지 특징)를 포함시키고, 이를 통해 다양한 게임 구조와 정책 출력 형식을 지원합니다.

- **Performance Highlights**: 새로운 AlphaGateau 모델은 유사한 파라미터 수를 가지는 이전 아키텍처보다 뛰어난 성능을 보이며, 훈련 시간의 약 10배 빠른 속도로 플레이 강도를 크게 향상시키는 결과를 보여주었습니다. 또한, 소규모 5×5 체스 변형에서 훈련된 모델은 표준 8×8 체스에서도 빠르게 조정되어 경쟁력 있는 성능을 발휘할 수 있음을 입증했습니다.



### LSEAttention is All You Need for Time Series Forecasting (https://arxiv.org/abs/2410.23749)
Comments:
          7 pages with referencing, 1 figure, 3 tables

- **What's New**: 본 연구는 multivariate time series forecasting에서 transformer 모델의 성능을 극대화하기 위해 LSEAttention 모듈을 도입하였습니다. 이는 전통적인 attention 메커니즘의 문제점인 entropy collapse와 training instability를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: LSEAttention 모듈은 Log-Sum-Exp (LSE) 방법과 Gaussian Error Linear Unit (GELU) 활성화 함수를 결합하여 수치적 안정성을 높였습니다. 이를 통해 softmax 연산의 정확도를 향상시키고, attention 메트릭스의 품질 문제를 해결합니다.

- **Performance Highlights**: LSEAttention은 다양한 실제 multivariate time series 데이터셋에서 실험되어 기존의 transformer 모델보다 우수한 성능을 보였으며, 몇몇 최신 모델의 성능을 초과하는 결과를 보여주었습니다.



### Exploring Consistency in Graph Representations:from Graph Kernels to Graph Neural Networks (https://arxiv.org/abs/2410.23748)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 Graph Neural Networks (GNNs)과 전통적인 그래프 커널 방법의 간극을 메우기 위한 연구로, GNN이 그래프 간 유사한 관계를 일관되게 캡처할 수 있는 방법을 제안합니다. 이 연구에서는 GNN 층을 통해 그래프 표현의 유사성을 일관되게 유지하도록 강제하는 손실 함수를 도입했습니다.

- **Technical Details**: 우리는 Weisfeiler-Lehman subtree (WL-subtree)와 Weisfeiler-Lehman optimal assignment (WLOA) 커널을 비교하고, WLOA가 다른 반복 단계에서 동일한 그래프 유사성을 점진적으로 결합하여, 이로 인해 GNN이 학습한 표현에서 관계 구조를 보다 잘 캡처할 수 있도록 하는 일관성 속성을 정의합니다. 또한, Iterative Graph Kernels (IGK) 개념을 도입하여 그래프 커널을 발전시키길 제안합니다.

- **Performance Highlights**: 우리가 제안한 일관성 손실(consistency loss)은 여러 GNN 모델의 그래프 분류 성능을 유의미하게 향상시키며, 다양한 데이터셋에 대해 일반적으로 적용 가능한 방법임을 실험적으로 입증했습니다.



### DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios (https://arxiv.org/abs/2410.23746)
Comments:
          Accepted to NeurIPS 2024 Dataset & Benchmarking Track

- **What's New**: 이 연구에서는 새로운 벤치마크인 DetectRL을 소개하며, 최신 LLM(대형 언어 모델) 생성 텍스트 탐지 기술이 실질적인 시나리오에서 여전히 미비함을 강조했습니다.

- **Technical Details**: DetectRL은 다양한 실제 응용 프로그램에서 LLM이 사용될 때의 문제들을 시뮬레이션하여 텍스트를 생성합니다. 주요 공격 방법으로는 프롬프트 공격, 패라프레이즈 공격, 섭동 공격이 포함되며, 데이터 믹싱을 통한 다변량 샘플링이 사용됩니다. 다양한 도메인에서 생성된 데이터로 실험을 진행하였습니다.

- **Performance Highlights**: 현재의 탐지기들은 공격적인 섭동에 의해 성능이 평균 34.48% AUROC만큼 감소하였고, 반면에 슈퍼바이즈 탐지기는 더욱 강력한 탐지 능력을 보여주었습니다. DetectRL은 실생활 활용 상황에서 탐지기의 효과성을 평가하는 데 흔히 고려되지 않았던 요소들을 분석하여, 탐지기의 발전 방향을 제시할 수 있는 잠재력을 갖추고 있습니다.



### Syno: Structured Synthesis for Neural Operators (https://arxiv.org/abs/2410.23745)
- **What's New**: 본 논문은 기존 연산자(operators)를 조합하거나 최적화하는 데 한정되어 있던 Neural Architecture Search (NAS)와 Tensor Compilers를 넘어 새로운 신경 연산자(neural operator)를 자동으로 합성(synthesize)하는 새로운 연구 방향을 탐구한다.

- **Technical Details**: 제안된 시스템 Syno는 텐서 차원(tensor dimensions)에서 정의된 미세한 원시(primitives) 집합을 사용하여 다양한 원하는 특성을 보장하며, 중복된 후보를 피하기 위해 표현 정규화(expression canonicalization) 기법을 통해 탐색 기법을 개선한다. 또한, Syno는 지정된 입력/출력 차원 크기에 맞는 유효한 연산자를 얻기 위한 가이드 합성 흐름을 채택하고, 효율적인 스토캐스틱 트리 탐색(stochastic tree search) 알고리즘을 활용하여 설계 공간을 신속하게 탐색한다.

- **Performance Highlights**: Syno는 NAS 최적화 모델에서도 평균 2.06배의 속도 향상과 1% 미만의 정확도 감소를 보여주며, CIFAR-100에서 Syno로 최적화된 연산자는 표준 합성곱(convolution) 및 행렬 곱셈(matrix multiplication)보다 더 빠르다고 밝혀졌다. ImageNet에서는 1%에서 2%의 정확도 감소로 1.10배에서 4.73배의 속도 향상을 달성하였고, GPT-2 훈련도 1.1배 가속되었다.



### What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspectiv (https://arxiv.org/abs/2410.23743)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 다양한 레이어에서의 훈련 패턴을 조사하고, 빠른 사고(fast thinking)와 느린 사고(slow thinking)가 레이어별 기울기(gradients)에 미치는 영향을 분석했습니다. 특히, 체인 오브 생각 (CoT, Chain of Thought) 방법이 훈련의 안정성에 미치는 영향을 보았습니다.

- **Technical Details**: 이 연구는 5개의 기본 LLM과 5개의 지시 조정(instruction-tuned) LLM의 레이어별 기울기를 비교하며, 특정 작업(수학, 일반 기초 reasoning, 지식 추출)에 대한 여러 데이터셋을 사용했습니다. 빠른 사고와 느린 사고를 기준으로 기울기의 핵심 특성을 Singular Value Decomposition (SVD)을 통해 측정하였으며, 각 레이어의 기울기 변화를 분석했습니다.

- **Performance Highlights**: 느린 사고(Detailed CoT)를 통해 특정 정답을 식별하는 기울기 패턴을 보여주었고, 빠른 사고는 기울기가 초기 레이어에서 더 크게 나타났습니다. 지시 조정된 LLM은 사전 훈련된 LLM보다 잘못된 사고 경로를 식별하는 데 우수하지 않았습니다. 이러한 결과는 대형 언어 모델 훈련의 효율성과 안정성에 대한 새로운 통찰을 제공하였습니다.



### Artificial intelligence to improve clinical coding practice in Scandinavia: a crossover randomized controlled tria (https://arxiv.org/abs/2410.23725)
Comments:
          13 pages, 4 figures, 4 tables

- **What's New**: 이번 연구는 임상 코딩을 지원하기 위해 개발된 AI 도구인 Easy-ICD의 성능을 평가하며, 복잡한 (complex) 텍스트와 간단한 (simple) 텍스트에 대한 코딩 시간과 정확성 향상을 실험적으로 분석합니다.

- **Technical Details**: 연구는 교차 무작위 대조 시험 (crossover randomized controlled trial)으로 진행되었으며, 참가자들은 두 그룹으로 무작위 배정되어 복잡한 텍스트와 간단한 텍스트를 수동 코딩과 Easy-ICD 도구를 사용하여 코딩하였습니다. 결과는 Mann-Whitney U test를 통해 분석되었으며, 복잡한 임상 텍스트의 경우 평균 코딩 시간 차이는 123초로, Easy-ICD 사용 시 46%의 시간 절약이 나타났습니다.

- **Performance Highlights**: 연구 결과, 복잡한 텍스트에서는 AI 도구 사용 시 코딩 시간이 현저히 감소하였으나, 코딩 정확성의 개선은 통계적으로 유의미하지 않았습니다. 간단한 텍스트에서는 시간 차이가 없었으며, 본 연구는 임상 워크플로우에서 AI 도구의 활용 가능성을 보여주었습니다.



### Rethinking Inverse Reinforcement Learning: from Data Alignment to Task Alignmen (https://arxiv.org/abs/2410.23680)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2306.01731

- **What's New**: 이번 논문에서는 전통적인 데이터 정렬(data alignment) 대신 작업 정렬(task alignment)을 우선시하는 새로운 IRL 기반의 모방 학습(imitation learning) 프레임워크를 제안합니다. 이 프레임워크는 약한 감독(weak supervision)으로 전문가의 시연을 활용하여 작업에 더 잘 맞는 보상 함수(reward function) 후보 집합을 도출합니다.

- **Technical Details**: 제안하는 프레임워크인 Protagonist Antagonist Guided Adversarial Reward (PAGAR)는 반지도 학습(semi-supervised learning) 접근 방식을 채택하여 적대적(adversarial) 메커니즘을 통해 작업에 맞는 보상 함수 집합을 사용하여 정책(policy)을 훈련합니다. 이 과정에서 전문가의 시연과 정책의 유사성을 집합적으로 검증하여 작업 수행 능력을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 PAGAR 프레임워크는 복잡한 IL 작업과 제한된 시연이 이루어진 환경에서 기존 IL 기준선보다 우수한 성능을 보였습니다.



### Provable Benefit of Cutout and CutMix for Feature Learning (https://arxiv.org/abs/2410.23672)
Comments:
          NeurIPS 2024 camera-ready version, 81 pages

- **What's New**: 본 논문에서는 Cutout 및 CutMix와 같은 패치 레벨(data augmentation) 데이터 증강 기법의 효과를 분석하고 이들 방법에 대한 이론적인 이해가 부족함을 지적합니다. 특히, Cutout 훈련과 CutMix 훈련을 통해 저주파수(feature) 특성을 선별하여 학습할 수 있음을 입증합니다.

- **Technical Details**: 연구에서는 2계층 신경망(neural network)을 세 가지 방법으로 훈련시킵니다: 일반 훈련(vanilla training)과 Cutout 훈련, CutMix 훈련. 분석은 레이블에 의존하는 다양한 희귀성과 레이블에 의존하지 않는 다양한 강도의 노이즈(noise)로 구성된 데이터 모델을 사용하여 진행됩니다. 이론을 통해 Cutout은 일반 훈련이 학습할 수 없는 저주파수 특성을 학습할 수 있음을 보여줍니다.

- **Performance Highlights**: CutMix 훈련이 기존의 Cutout 보다 더 희귀한 특성을 학습할 수 있으며, 이로 인해 세 가지 방법 중 가장 높은 테스트 정확도(test accuracy)를 달성함을 알립니다. CutMix 훈련은 모든 특성과 노이즈 벡터(noise vectors)를 희귀성과 강도에 관계없이 "고르게" 학습시키는 특징이 있어, 패치 레벨 증강(patch-level augmentation)에 대한 새로운 통찰(insight)을 제공합니다.



### Kernel Looping: Eliminating Synchronization Boundaries for Peak Inference Performanc (https://arxiv.org/abs/2410.23668)
- **What's New**: 이 논문에서는 최신 데이터 흐름 아키텍처의 레이어-레벨 퓨전을 활용하여, 언어 모델의 반복 레이어 구조와 결합한 새로운 최적화 기법인 'kernel looping'을 소개합니다.

- **Technical Details**: Kernel looping은 같은 커널에 대한 연속 호출 간의 동기화 비용을 제거하고, 파이프라인 외부 루프를 포함하는 수정된 커널에 대한 단일 호출로 변환함으로써 성능을 개선합니다. 이 최적화 기술은 SambaNova SN40L RDU에서 평가되었습니다.

- **Performance Highlights**: Kernel looping을 활용한 실험 결과, 다양한 오픈 소스 모델의 디코드 단계에서 최대 2.2배 속도 향상을 달성하였고, 8 및 16 소켓에서 90% 이상의 피크 성능을 이끌어냈습니다. 또한, DGX H100에 비해 최대 3.7배의 속도 향상을 기록했습니다.



### Deep Convolutional Neural Networks on Multiclass Classification of Three-Dimensional Brain Images for Parkinson's Disease Stage Prediction (https://arxiv.org/abs/2410.23649)
Comments:
          34 pages, 7 figures, and 4 tables

- **What's New**: 이 연구는 파킨슨병 (Parkinson's disease, PD)의 단계를 정확하게 예측하기 위한 모델을 개발하였습니다. SPECT 데이터를 활용한 다중 클래스 분류 작업이 진행되었습니다.

- **Technical Details**: 연구는 두 가지 SPECT 데이터 세트 (n = 634 및 n = 202)를 사용하였으며, 3D 뇌 이미지를 입력으로 사용하는 다양한 모델 아키텍처를 실험하였습니다. 2D CNN 모델과 3D CNN 모델을 사용하였고, 주의 메커니즘 (attention mechanism)을 적용하여 예측 과정에서의 다양한 슬라이스의 중요성을 고려하였습니다. 또한, 가중치 공유 (weight sharing) 기법을 활용한 공동 훈련 (cotraining) 기법을 적용하여 두 데이터 세트를 동시에 훈련하였습니다.

- **Performance Highlights**: 연구 결과, ImageNet에서 미리 훈련된 2D 모델이 Kinetics-400에서 미리 훈련된 3D 모델보다 우수한 성능을 보였고, 주의 메커니즘을 사용하는 모델이 2D 및 3D 모델보다 더 나은 성능을 나타냈습니다. 공동 훈련 기법은 데이터 세트가 충분히 클 때 모델 성능 향상에 효과적이었습니다.



### Anytime-Constrained Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2410.23637)
- **What's New**: 본 연구에서는 다중 에이전트 환경에서 anytime constraints를 도입하고, 이에 해당하는 솔루션 개념인 anytime-constrained equilibrium (ACE)을 정의합니다.

- **Technical Details**: 우리는 anytime-constrained Markov game의 포괄적인 이론을 제시하며, 여기에는 (1) 가능한 정책의 계산적 특성화, (2) ACE를 계산하는 고정 매개변수 탐색 가능 알고리즘, (3) 대략적인 ACE를 다항 시간 내에 계산하는 알고리즘이 포함됩니다.

- **Performance Highlights**: 제안된 알고리즘은 최악의 경우를 고려할 때, 유사한 문제에 대한 최상의 성능 보장을 제공합니다. 각 주요 결과는 서로 다른 알고리즘 기법을 활용하여 이루어졌으며, action-constrained Markov game에 대한 효율적 컴퓨테이션의 첫 번째 이론을 개발했습니다.



### Adaptive Alignment: Dynamic Preference Adjustments via Multi-Objective Reinforcement Learning for Pluralistic AI (https://arxiv.org/abs/2410.23630)
Comments:
          Accepted for the Pluralistic Alignment workshop at NeurIPS 2024

- **What's New**: 이번 논문은 다양한 인간의 요구와 가치에 맞춰 AI 시스템을 설계하고 배치하는 방법을 다루는 Pluralistic AI alignment에 대한 연구를 소개합니다. Multi Objective Reinforcement Learning (MORL)을 활용해 사용자 선호에 따라 AI를 동적으로 조정할 수 있는 프레임워크를 제안하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 MORL 기반의 적응형 정렬 방법으로, 자기 검토 과정을 거쳐 사용자 반응을 피드백 신호로 활용합니다. 이 과정에는 학습, 선택, 실행 및 검토의 세 가지 단계가 포함됩니다. 정책 선택 과정에서 기본 정책을 초기 선택하고, 사용자의 선호 변화에 맞추어 정책을 수정합니다.

- **Performance Highlights**: 이 접근법은 사용자에게 부담을 최소화하며 피드백 효율성을 제고하는데 기여합니다. 또한, 다양한 사용자와의 정확한 정렬을 통해 지속적으로 진화할 수 있는 시스템을 구축하고, AI와 사용자의 의사소통을 더욱 원활하게 합니다.



### Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation (https://arxiv.org/abs/2410.23629)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: PiMForce라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 3D 손 자세 정보를 활용하여 팔꿈치 표면 전기 생리학(sEMG) 신호를 보강함으로써 손 압력 추정의 정확도를 향상시킵니다.

- **Technical Details**: PiMForce는 3D 손 자세와 sEMG 신호를 결합하여 전체 손 밀도를 정확히 측정하도록 설계되었습니다. 이 프레임워크는 다양한 손-물체 상호작용에서의 압력을 정확하게 추정할 수 있도록 돕습니다. 실험을 위해 21명의 참가자로부터 동기화된 데이터 세트를 구축하였으며, 이는 압력 장갑, sEMG 아르밴드, 마커리스 손가락 추적 모듈로 구성된 멀티모달 데이터 수집 시스템을 기반으로 합니다.

- **Performance Highlights**: 이 시스템은 기존의 sEMG 기반 방법이나 비전 기반 방법에 비해 월등히 향상된 성능을 보여줍니다. 3D 손 자세 정보와 sEMG 신호의 통합을 통해 더 정확하고 안정적인 손 압력 추정이 가능해졌습니다.



### Using Structural Similarity and Kolmogorov-Arnold Networks for Anatomical Embedding of 3-hinge Gyrus (https://arxiv.org/abs/2410.23598)
- **What's New**: 이 연구에서는 3-hinge gyrus(3HG)의 해부학적 특징을 임베딩(embedding)하는 새로운 자기 지도 학습 프레임워크(self-supervised framework)를 제안합니다. 이 프레임워크는 여러 개의 뇌 간의 3HG 간의 대응을 구축하는 데 중점을 두고 있으며, 기존 방법의 한계를 극복하는 데 목표를 두고 있습니다.

- **Technical Details**: 제안된 방법에서는 Kolmogorov-Arnold Networks (KAN)를 사용하여 3HG의 해부학적 특징을 인코딩합니다. 각 3HG 노드의 구조적 유사성(structural similarity)을 높이고, 선택적 재구성 손실(selective reconstruction loss) 함수를 도입하여 비제로(non-zero) 요소의 재구성 오류를 벌점으로 부과함으로써 임베딩 벡터의 표현 능력을 강화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 복잡한 피질 구조에서 교차 주제 간 접합점을 효과적으로 설정할 수 있음을 보여주었으며, 3HG의 공통성과 변동성을 유지하면서 일반적인 3HG 네트워크와 인구 폭넓은 분석 구성에 중요한 기여를 할 수 있을 것으로 기대됩니다.



### How Do Flow Matching Models Memorize and Generalize in Sample Data Subspaces? (https://arxiv.org/abs/2410.23594)
Comments:
          33 pages, 9 figures

- **What's New**: 이번 연구에서는 Flow Matching 모델을 활용하여 고차원 공간에서의 저차원 구조의 데이터 생성 문제를 해결하고자 했습니다. 생성된 샘플이 실제 데이터 포인트를 기억하며 샘플 데이터 서브스페이스를 정확하게 나타내도록 하는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Gaussian prior를 가정하고 가장 최적의 velocity field에 대한 분석 표현을 도출하였습니다. 이를 통해 OSDNet(Orthogonal Subspace Decomposition Network)을 도입하고, 이 네트워크가 velocity field를 서브스페이스와 off-subspace로 분해하는 구조를 제시합니다.

- **Performance Highlights**: 생성된 샘플은 샘플 데이터 서브스페이스에 근접하게 유지되며 인접성과 다양성을 보존합니다. 경험적 분석을 통해, off-subspace 구성 요소는 시간이 지남에 따라 줄어들고, 서브스페이스 구성 요소는 샘플 데이터 서브스페이스 내에서 일반화된다는 점을 강조합니다.



### Automating Quantum Software Maintenance: Flakiness Detection and Root Cause Analysis (https://arxiv.org/abs/2410.23578)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구는 양자 소프트웨어의 flaky test를 **자동화된 프레임워크**로 탐지하고, 기존 데이터셋을 확장하여 flaky test를 효과적으로 식별할 수 있는 방법론을 제시합니다.

- **Technical Details**: 연구진은 **transformers**와 **cosine similarity**를 활용하여 14개의 양자 소프트웨어 리포지토리에서 **flaky test**를 식별했습니다. 기존 데이터셋에 기반하여 새로운 flaky test 25개를 발견하였으며, **Large Language Models (LLMs)**를 이용해 flaky test의 탐지 및 분류를 자동화했습니다.

- **Performance Highlights**: 최고 성능의 LLMs는 flaky test 탐지에서 **F1-score** 0.8871을 기록했지만, root cause 식별에서는 0.5839에 그쳤습니다. 자동화된 탐지 프레임워크의 적용이 유망하나, 큰 양자 코드베이스 내 root cause 탐지 및 분류에 대한 개선 필요성이 강조되었습니다.



### MS-Glance: Non-semantic context vectors and the applications in supervising image reconstruction (https://arxiv.org/abs/2410.23577)
Comments:
          Accepted by WACV 2025

- **What's New**: 본 연구에서는 비의미적(non-semantic) 컨텍스트 정보를 활용한 시각 인식 방법론을 제안합니다. 특히, 새로운 비의미적 컨텍스트 설명자인 MS-Glance와 두 이미지 간 비교를 위한 Glance Index Measure를 소개합니다.

- **Technical Details**: MS-Glance는 이미지에서 픽셀을 무작위로 선택하여 형성한 Global Glance 벡터와 로컬 이미지 윈도우를 평면화하여 생성한 Local Glance 벡터를 포함합니다. Glance Index는 두 세트의 표준화된 Glance 벡터 간의 내적(inner product)으로 정의됩니다. 이 방법은 두 가지 이미지 복원 작업, 즉 암묵적인 신경 표현(INR)으로 인한 이미지 피팅과 저샘플링된 MRI 복원에 효과적입니다.

- **Performance Highlights**: MS-Glance는 자연 이미지 및 의료 이미지를 포함한 다양한 데이터셋에서 기존 이미지 복원 손실을 초월하는 성능을 보여줍니다. MS-Glance 손실은 L1+SSIM 및 LPIPS와 같은 기존 손실 함수들보다 뛰어난 결과를 나타내며, 이미지 품질 향상에 기여합니다.



### Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models (https://arxiv.org/abs/2410.23558)
- **What's New**: 본 논문에서는 다양한 LLM-as-Attacker 방식을 통합한 새로운 black-box jailbreak 공격 프레임워크를 제안합니다. 이 방법은 기존의 jailbreak 연구에서 얻은 세 가지 핵심 관찰을 기반으로 합니다.

- **Technical Details**: 1. Ensemble 접근 방식 : 여러 개의 LLM을 조합하여 공격을 수행하여 더 효과적으로 목표 LLM의 취약점을 노출합니다. 2. 각 악의적인 지침에 대해 최적화 예산을 조정하여 효율적인 공격을 달성합니다. 3. 악의적인 지침의 의미적 일관성을 적절히 방해하여 투명성을 높입니다.

- **Performance Highlights**: 본 논문은 2024 LLM 및 에이전트 안전 대회에 참여하여 Jailbreaking Attack Track에서 우수한 성적을 거두었습니다. 제안된 방법은 TAP 및 PAP 방식보다 성능이 향상되었으며, stealthness 점수 또한 높아졌습니다.



### From Context to Action: Analysis of the Impact of State Representation and Context on the Generalization of Multi-Turn Web Navigation Agents (https://arxiv.org/abs/2410.23555)
Comments:
          10 pages, 3 figures, 5 tables

- **What's New**: 최근 대형 언어 모델(LLM) 기반 프레임워크의 발전에 따라, 복잡한 실세계 응용 프로그램에서의 능력이 확대되었습니다. 특히 사용자의 명령에 의해 웹 브라우저에서 작업을 수행하면서 상호작용적인 웹 내비게이션을 가능하게 하고 있습니다.

- **Technical Details**: 이 연구에서는 웹 내비게이션 에이전트의 성능에 영향을 미치는 다양한 맥락적 요소를 분석하여 서로 다른 상호작용 기록과 웹 페이지 표현의 영향을 최적화하는 방법을 탐구합니다. 이를 통해 Agnet의 성능 향상을 도모하고, 이는 특히 보지 못한 웹사이트, 카테고리 및 지리적 위치에서 효과적인 맥락 관리가 가능함을 보여줍니다.

- **Performance Highlights**: 연구 결과, 효과적인 맥락 관리가 이루어짐으로써 에이전트의 성능이 개선되었으며, 이는 실세계 응용 프로그램에서 더 정확하고 효율적인 웹 내비게이션을 가능하게 합니다.



### ALISE: Accelerating Large Language Model Serving with Speculative Scheduling (https://arxiv.org/abs/2410.23537)
Comments:
          ICCAD 2024

- **What's New**: ALISE라는 새로운 LLM inference serving 프레임워크를 제안합니다. 이 프레임워크는 speculated scheduling을 통한 효율적인 작업 우선순위 배정을 통해 Heterogeneous(여러 종류의) 작업에 대한 응답 지연을 최소화합니다.

- **Technical Details**: ALISE는 각 작업의 실행 시간을 추정하고, 이 정보를 기반으로 우선순위 큐를 활용하여 선제적 스케줄링을 수행합니다. 이를 통해 HoL blocking 문제를 완화합니다. 또한 KV(cache) 관리에 있어서 우선순위 기반의 적응형 메모리 관리 프로토콜과 양자화 기반 압축 기술을 사용합니다.

- **Performance Highlights**: ALISE는 vLLM과 비교하여 Alpaca와 ShareGPT 데이터셋에서 각각 최대 1.8배 및 2.1배의 throughput 향상을 보여주었습니다. 또한 ALISE는 end-to-end latency와 전체 throughput에서 기존의 FCFS 솔루션보다 월등한 성능을 보입니다.



### Simulating User Agents for Embodied Conversational-AI (https://arxiv.org/abs/2410.23535)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 이 연구에서는 사용자 행동을 모사하는 LLM 기반의 사용자 대리인(user agent)을 제안하여, 로봇과의 상호작용을 효율적으로 생성할 수 있는 방법을 탐구합니다. 이를 통해 데이터 수집의 비용 및 노동력을 줄일 수 있는 가능성을 제시합니다.

- **Technical Details**: LLM(대규모 언어 모델)을 활용한 사용자 대리인은 특정 사용자 목표(예: 아침 식사 만들기)에 따라 로봇의 행동을 관찰하고, 그에 따라 적절한 대화를 생성할 수 있습니다. 연구에서는 zero-shot 및 few-shot prompting 기법을 통해 대화의 정확성을 평가하였으며, TEACh 데이터셋을 활용하여 성능을 비교하였습니다.

- **Performance Highlights**: LLM 기반의 사용자 대리인은 zero-shot prompting에서 42%의 F-measure를, few-shot prompting에서는 43.4%를 기록하여 인간의 대화 행동을 모방하는 데 성공했습니다. 또한, fine-tuning을 통해 사용자 대리인의 발화 결정 능력이 향상되어, 발화할 시점의 안정성은 유지되고 발화 내용의 정확성이 51.1%에서 62.5%로 개선되었습니다.



### There and Back Again: On the relation between noises, images, and their inversions in diffusion models (https://arxiv.org/abs/2410.23530)
- **What's New**: 이번 논문에서는 Denoising Diffusion Probabilistic Models (DDPMs)가 새로운 이미지를 생성하는 데 뛰어난 성과를 내고 있지만, 의미 있는 latent space가 부족하다는 문제를 다루고 있습니다. 특히, DDIM inversion 기법의 정확성을 분석하고, 초기 Gaussian noise와 생성된 샘플 간의 관계를 탐구합니다.

- **Technical Details**: DDPMs와 DDIM을 기반으로 한 연구에서, DDIM inversion이 초기 Gaussian noise를 reverse하려 할 때 발생하는 불일치 문제를 설명합니다. 이 연구는 초기 노이즈와 생성된 샘플 간의 관계를 유클리드 거리(Euclidean distance)로 정확히 할당할 수 있음을 보여주며, 훈련 초기 단계에서 noise와 generations 사의 매핑이 어떻게 형성되는지를 분석합니다.

- **Performance Highlights**: 이 연구의 주요 기여는 reverse DDIM이 표준 다변량 Gaussian이 아닌 latent representation을 생성하여 이론과 실제 간의 차이를 만들어낸다는 점입니다. 또한 모델의 생성 능력을 개선하더라도 reverse DDIM의 정확도를 개선하지 못한다는 점을 강조합니다.



### LEAF: Learning and Evaluation Augmented by Fact-Checking to Improve Factualness in Large Language Models (https://arxiv.org/abs/2410.23526)
Comments:
          22 pages, 9 figures

- **What's New**: 이번 연구에서는 의료 분야를 중심으로 LLMs(대형 언어 모델)의 사실 정확성을 향상시키기 위한 새로운 접근법인 LEAF(Learning and Evaluation Augmented by Fact-Checking)를 제안합니다. LEAF는 두 가지 전략인 Fact-Check-Then-RAG와 Self-Training을 활용하여 모델의 출력 신뢰성을 증가시킵니다.

- **Technical Details**: LEAF의 첫 번째 전략인 Fact-Check-Then-RAG는 사실-checking 결과를 활용하여 Retrieval-Augmented Generation (RAG)을 개선합니다. 두 번째 전략은 Self-Training을 통해 supervised fine-tuning (SFT)이나 Simple Preference Optimization (SimPO) 기법을 사용하여 LLM 매개변수를 업데이트합니다.

- **Performance Highlights**: LEAF를 통해 사실-checking된 응답을 통합함으로써 LLM의 성능이 크게 향상되었음을 보여주며, 특히 정보 정확성이 중요한 의료 분야에서 신뢰성과 사실성을 높이는 효과를 나타냅니다.



### Dynamic Strategy Planning for Efficient Question Answering with Large Language Models (https://arxiv.org/abs/2410.23511)
Comments:
          Under review at ACL Rolling Review

- **What's New**: 이번 연구에서는 DyPlan이라는 새로운 기술을 제안하여 Large Language Models (LLMs)의 질문-응답 성능을 개선하고, 질문 유형에 따라 동적으로 전략을 선택할 수 있는 프로세스를 도입합니다. 이를 통해 더 효율적인 출력 토큰 생성을 가능하게 하고, 비용을 절감합니다.

- **Technical Details**: DyPlan은 입력 질문에 따라 가장 적합한 전략을 선택하는 초기 결정 단계를 도입하고, LLM의 응답 생성을 이 전략에 맞춰 안내합니다. DyPlan-verify로 확장하여 내부 검증 및 수정 과정을 추가하여 생성된 답변의 품질을 더욱 향상시킵니다. 세 가지 주요 QA 데이터 세트에서 실험을 수행하여 DyPlan과 DyPlan-verify의 성능을 평가하였습니다.

- **Performance Highlights**: 실험 결과, DyPlan은 최고의 기준 모델에 비해 평균 7-13%의 성능 향상과 11-32%의 비용 절감을 달성했으며, DyPlan-verify를 통해 평균 12-13%의 성능 개선과 11-19%의 비용 감소를 확인했습니다.



### Learning to Achieve Goals with Belief State Transformers (https://arxiv.org/abs/2410.23506)
- **What's New**: 우리는 'Belief State Transformer'를 소개합니다. 이 모델은 prefix와 suffix를 입력으로 받아 다음 토큰을 예측하는 독창적인 목표를 가지고 있습니다. 기존의 forward-only transformer가 해결하기 어려운 문제를 효과적으로 배울 수 있습니다.

- **Technical Details**: Belief State Transformer는 정확한 예측을 위해 필요한 모든 정보를 캡처하는 compact belief state를 학습하는 데 중점을 둡니다. 모델의 각 구성 요소는 표준 Transformer가 부족한 어려운 시나리오에서 필수적임을 보여주는 경험적 ablation 결과가 있습니다.

- **Performance Highlights**: 이 모델은 알려진 prefix와 suffix를 가진 스토리 작성 작업에서 Fill-in-the-Middle 방법보다 더 나은 성과를 보이며, 목표가 알려지지 않은 상황에서도 향상된 성능을 입증합니다. Belief State Transformer는 효율적인 goal-conditioned decoding과 우수한 테스트 시간 추론 및 높은 품질의 텍스트 표현을 가능하게 합니다.



### All or None: Identifiable Linear Properties of Next-token Predictors in Language Modeling (https://arxiv.org/abs/2410.23501)
- **What's New**: 이번 논문에서는 다양한 언어 모델에서 선형 속성의 보편성을 설명하기 위해 identifiability(식별 가능성) 개념을 분석합니다. 다양한 모델들 간의 분포가 동일할 때 선형적인 속성이 공유된다는 점을 논의합니다.

- **Technical Details**: 논문은 distribution-equivalent next-token predictors에 대한 identifiability 결과를 증명합니다. 또한 relational linearity를 정제하여 다양한 선형성의 개념들이 어떻게 분석될 수 있는지를 보여줍니다. 특히, autoregressive 모델 및 Transformer 아키텍처에 주목합니다.

- **Performance Highlights**: 적절한 조건 하에, 이러한 선형 속성은 모든 또는 어떤 distribution-equivalent next-token predictors에 대해 유효함이 입증됩니다.



### Kernel-Based Function Approximation for Average Reward Reinforcement Learning: An Optimist No-Regret Algorithm (https://arxiv.org/abs/2410.23498)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 본 연구에서는 커널 리지 회귀(kernel ridge regression)를 이용하여 기대 가치 함수(expected value function)를 예측하는 새로운 강화 학습 알고리즘 KUCB-RL을 제안합니다. 이 알고리즘은 무한 수평 평균 보상(infinite horizon average reward) 설정에서 비선형 함수 근사를 사용하는 첫 번째 알고리즘으로, 이론적 성과 보장을 확립했습니다.

- **Technical Details**: KUCB-RL 알고리즘은 커널 기반 모델을 사용하여 미래 w 스텝 동안의 상태-행동 값 함수(state-action value function)에 대한 상한 신뢰 구간(upper confidence bound)을 구축합니다. 이 알고리즘은 최적성을 기반으로 하여 현재 상태에 대해 행동을 선택하며, Markov Decision Process (MDP)의 구조적 복잡성을 고려한 것입니다.

- **Performance Highlights**: 본 연구에서는 KUCB-RL 알고리즘의 성과에 대해 새로운 무회귀(no-regret) 보장 조건을 수립하였으며, 이를 통해 𝒪( Tw + (w + wργ(T;ρ) + log(T/δ)) ρTγ(T;ρ) + ρ²w²γ(T;ρ)γ(T/w;ρ) )와 같은 회귀 경계를 도출했습니다. 이 알고리즘은 다양한 RL 문제에 적용 가능성이 있습니다.



### DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity (https://arxiv.org/abs/2410.23495)
Comments:
          Published at NeurIPS 2024

- **What's New**: 이번 연구에서는 Warm-starting(워밍 스타트)된 신경망 훈련이 고정적인 데이터 분포에서도 플라스틱성(plasticity)을 잃는 원인을 분석하고, 이를 완화하기 위한 새로운 방법인 Direction-Aware SHrinking(DASH)을 제안합니다. 이를 통해 훈련 효율성과 테스트 정확도를 향상시키는 것을 목표로 합니다.

- **Technical Details**: 우리는 기존에 학습한 가중치를 초기화하여 신경망 훈련을 시작하는 방식인 Warm-starting이 왜 더 나쁜 일반화 성능을 초래하는지를 살펴봅니다. 연구 결과, Warm-starting이 추가된 데이터의 노이즈를 기억하면서 새로운 특징을 학습하지 못하게 하여 과적합(overfitting)으로 이어진다는 점을 발견했습니다. DASH는 이러한 메커니즘을 개선하여 기억한 노이즈를 선택적으로 잊고 학습된 특징을 보존합니다.

- **Performance Highlights**: DASH 방법을 다양한 모델, 데이터셋 및 최적화 도구를 사용하여 검증한 결과, 훈련 시간 단축 및 테스트 정확도 향상과 같은 긍정적인 결과를 나타냈습니다.



### Causality-Driven Audits of Model Robustness (https://arxiv.org/abs/2410.23494)
- **What's New**: 이번 연구는 DNN(Deep Neural Networks)의 강건성 감사(robustness audit) 방법에 대한 새로운 접근 방식을 제시하며, 인과 추론(causal inference)을 활용하여 복잡한 왜곡의 원인을 정량적으로 분석합니다. 이는 환경, 센서, 처리 파이프라인의 다양한 요인이 상호 작용하여 발생하는 이미지 왜곡에 대한 DNN의 민감도를 측정하는 데 도움을 줍니다.

- **Technical Details**: 본 연구는 이미지 생성 과정(image generating process)에서의 인과 모델(causal models)을 사용하여 DNN의 강건성을 평가하는 새로운 방법론을 개발합니다. 이를 위해 구조적 인과 모델(Structural Causal Model, SCM)을 통해 이미지 품질에 영향을 미치는 주요 요인들을 모델링하고, 이를 통해 DNN 민감도 평가를 수행합니다. 이 방법은 다양한 촬영 조건(resulting conditions)에서의 경험적 데이터를 활용하여 인과 효과(causal effects)를 신뢰성 있게 추정합니다.

- **Performance Highlights**: 실험 결과, 제안된 인과적 강건성 감사(Causality-Driven Robustness Audit, CDRA) 방법은 다양한 비전 태스크(vision tasks)에서 DNN의 성능에 미치는 개별 요인의 인과적 영향을 효과적으로 추정하는 것으로 나타났습니다. 이는 DNN의 성능 저하를 예측하고, 현업에서의 예기치 않은 DNN 실패 위험을 감소시키는 데 기여할 수 있습니다.



### Keep on Swimming: Real Attackers Only Need Partial Knowledge of a Multi-Model System (https://arxiv.org/abs/2410.23483)
Comments:
          11 pages, 2 figures

- **What's New**: 최근 기계학습에서 다수의 모델이나 agentic 아키텍처를 결합하여 작업을 수행하는 접근법이 증가하고 있습니다. 이 논문은 최종 블랙박스 모델에 대한 프록시 모델만 있을 때 멀티 모델 시스템을 대상으로 하는 새로운 적대적 공격 방법을 제안합니다.

- **Technical Details**: 우리는 복합 모델 시스템에서 초기 모델들이 적대적 변화를 효율적으로 방지할 때, 공격자가 최종 모델에 대한 간접적인 접근성만 가질 경우 공격을 설정하는 방법을 개발했습니다. 이 새로운 방법은 이전의 공격 방법과 비교했을 때 성공률이 약 80%로 더 높고, MSE 기준으로 9.4% 작은 변화를 포함합니다.

- **Performance Highlights**: 본 연구의 실험은 감독된 이미지 파이프라인에 중점을 두었으나, 제안된 공격 방법은 다른 멀티 모델 환경에서도 효과적으로 일반화될 것으로 기대됩니다.



### Risk Sources and Risk Management Measures in Support of Standards for General-Purpose AI Systems (https://arxiv.org/abs/2410.23472)
Comments:
          91 pages, 8 figures

- **What's New**: 이 논문은 일반 목적의 인공지능(GPAI) 시스템에 대한 위험 요소와 위험 관리 조치를 광범위하게 정리한 최초의 연구로, AI 안전 기준을 개발하는 데 기여하고자 한다.

- **Technical Details**: GPAI 모델의 개발, 훈련, 배포 단계에서의 기술적, 운영적, 사회적 위험을 식별하고, 전통적 및 실험적 위험 관리 방법을 조사하였다. 논문은 정책 입안자들이 AI의 안전 공학 요구사항을 이행하는 데 필요한 기준과 실천 코드를 알리기 위해 작성되었다.

- **Performance Highlights**: AI 제공자, 연구자, 규제 기관 등이 시스템적 위험을 식별하고 완화하는 데 사용할 수 있는 직접적으로 활용 가능한 공공 도메인 라이센스 하에 발표된 리스크 카탈로그를 제공한다.



### Graph-Augmented Relation Extraction Model with LLMs-Generated Support Documen (https://arxiv.org/abs/2410.23452)
- **What's New**: 이 연구에서는 Graph Neural Networks (GNNs)와 Large Language Models (LLMs)를 통합하여 문장 수준의 관계 추출 (Relation Extraction, RE)을 향상시키는 새로운 접근 방식을 제시합니다. 이 방법은 LLMs의 힘을 활용하여 보조 정보를 생성하고, 이를 통해 텍스트 데이터의 복잡한 그래프 표현을 생성합니다.

- **Technical Details**: 새로운 모델은 GNN을 통해 각 엔티티와 관련된 임베딩(embedding)을 정제 및 강화하여, 문장 간의 복잡한 관계를 효과적으로 포착할 수 있도록 설계되었습니다. 실험은 CrossRE 데이터셋을 기반으로 진행되었으며, 다양한 도메인에서 성능의 유의미한 향상을 보여주었습니다. 또한, LLM을 사용하여 보조 데이터셋을 생성하고, GNN 모듈을 통해 임베딩을 통합하여 성능 평가 체계를 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 문장 간의 복잡한 관계를 포착하는 능력에서 두드러진 성과를 나타내어, 기존의 문장 수준 RE 모델들이 가지던 한계를 극복하는 데에 기여할 것으로 보입니다. 상대적으로 다양한 도메인에서 경우에 따라 더 나은 Macro F1-Score를 기록하여, 관계 추출 분야에서의 GNN과 LLM의 조합 가능한 잠재력을 강조합니다.



### Return Augmented Decision Transformer for Off-Dynamics Reinforcement Learning (https://arxiv.org/abs/2410.23450)
Comments:
          26 pages, 10 tables, 10 figures

- **What's New**: 이번 연구에서는 오프라인 오프-다이내믹스 강화 학습(offline off-dynamics reinforcement learning) 접근법을 통해, 접근이 용이한 소스 도메인(source domain)의 데이터를 활용하여 데이터가 제한된 타겟 도메인(target domain)에서의 정책 학습을 향상시키는 방법을 제안합니다. 핵심 아이디어는 Return-Augmented Decision Transformer (RADT) 방법을 통해 소스 도메인의 보상(reward)을 타겟 도메인의 분포에 정렬하는 것입니다.

- **Technical Details**: 우리는 Return Conditioned Supervised Learning (RCSL)을 집중적으로 연구하였고, 특히 Decision Transformer (DT)에서 정책이 원하는 보상에 따라 행동을 예측할 수 있도록 합니다. RADT 방법론에서는 소스 도메인의 궤적(reward trajectory)에서 보상을 증강하여 타겟 도메인과 일치시키며, 이 과정에서 제안된 두 가지 실제 구현 방식인 RADT-DARA와 RADT-MV를 도입하였습니다.

- **Performance Highlights**: 종합적인 D4RL 데이터 세트에서 수행된 실험 결과에 따르면, 우리의 방법은 오프-다이내믹스 강화 학습 시나리오에서 동적 프로그래밍 기반의 기존 방법들보다 일반적으로 우수한 성능을 보여주었습니다. 이로써 RCSL에서의 보상 증강이 제한된 타겟 데이터셋 크기에서도 정책 학습의 성능을 향상시킬 수 있음을 입증하였습니다.



### Venire: A Machine Learning-Guided Panel Review System for Community Content Moderation (https://arxiv.org/abs/2410.23448)
- **What's New**: 이 논문에서는 Reddit을 위한 ML(기계 학습) 기반의 패널 리뷰 시스템인 Venire를 개발했습니다. Venire는 조정 팀의 서로 다른 의견을 효율적으로 처리하기 위해 다양한 모더레이터의 결정을 조기에 발견하고 검토하는 시스템입니다.

- **Technical Details**: Venire는 대규모 로그 데이터를 기반으로 훈련된 기계 학습 모델을 사용하여 모더레이터 간의 의견 차이가 발생할 가능성이 있는 사례를 예측합니다. 이 시스템은 모더레이터가 수동으로 패널 리뷰를 요청할 수 있도록 하며, 최종 결과는 여러 모더레이터의 투표로 결정됩니다.

- **Performance Highlights**: Venire는 의사 결정의 일관성을 향상시키고 잠재적인 의견 차이를 드러내는 데 도움을 줍니다. 연구 결과에 따르면, Venire를 사용한 모더레이터는 어려운 조정 사례를 더 자신감 있게 해결할 수 있었습니다.



### TPP-Gaze: Modelling Gaze Dynamics in Space and Time with Neural Temporal Point Processes (https://arxiv.org/abs/2410.23409)
Comments:
          Accepted at WACV 2025

- **What's New**: 본 논문에서는 Neural Temporal Point Process (TPP)를 기반으로 한 새로운 스캔패스 모델인 TPP-Gaze를 제안합니다. 이 모델은 고정점의 위치와 지속 시간의 동적 변화를 동시에 학습하며, 딥러닝 방법론과 포인트 프로세스 이론을 통합합니다.

- **Technical Details**: TPP-Gaze는 시각 자극에 대한 고정점의 시퀀스를 모델링하는 데 있어서 Neural TPP의 개념을 적용합니다. 이 모델은 불규칙한 간격으로 발생하는 사건들의 연속적인 시간을 예측하며, 기존 모델들이 간과한 시간 정보를 효과적으로 학습합니다. 이를 통해 스캔패스에서 고정점 간의 전환 (saccade) 및 그 지속 시간을 적절히 예측할 수 있습니다.

- **Performance Highlights**: 다섯 개의 공개 데이터셋에서 실시한 실험 결과, TPP-Gaze 모델이 최신의 최첨단 접근법들에 비해 전반적으로 우수한 성능을 보임을 확인하였습니다. 이 모델은 스캔패스 예측 문제를 처리하기 위한 새로운 기준을 제공합니다.



### FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions (https://arxiv.org/abs/2410.23405)
- **What's New**: 이 논문에서는 FlowLLM이라는 새로운 생성 모델을 소개합니다. FlowLLM은 대형 언어 모델(LLM)과 리만 흐름 일치(RFM)를 결합하여 새로운 결정 물질을 설계하는 방법을 제시합니다. 이를 통해 안정적인 물질의 생성 속도를 300% 이상 향상시켰으며, 독특하고 새로운 물질의 생성 속도를 약 50% 개선했습니다.

- **Technical Details**: FlowLLM은 LLM을 통해 메타 안정성 결정의 효과적인 기초 분포를 학습한 후, 이를 그래프 표현으로 변환하여 RFM 모델을 사용하여 샘플을 채취하고 좌표 및 격자 파라미터를 반복적으로 정제합니다. 이 접근 방식은 불연속 값(atomic types)과 연속 값(atomic positions 및 lattice geometry)을 동시에 생성하는 도전과제를 해결합니다.

- **Performance Highlights**: FlowLLM은 기존의 최고 성능 생성 모델에 비해 안정적인 물질과 독창적인 물질 생성을 300%와 50%의 비율로 각각 더 빠르게 진행할 수 있으며, 매우 유망한 결과를 보여주어 화학적으로 유효한 출력물 생성을 가능하게 합니다.



### Adaptive Network Intervention for Complex Systems: A Hierarchical Graph Reinforcement Learning Approach (https://arxiv.org/abs/2410.23396)
- **What's New**: 본 논문은 동적 네트워크 구조를 통해 에이전트 간 상호작용을 조정하고, 사회적 행동을 촉진하기 위해 설계된 계층적 그래프 강화 학습(HGRL) 프레임워크를 제안합니다.

- **Technical Details**: HGRL은 Graph Neural Networks (GNN)와 Reinforcement Learning (RL)을 결합하여, 제한된 관리 권한 하에서도 네트워크 개입 정책을 효율적으로 학습할 수 있게 해 줍니다. 이를 통해 에이전트 간의 상호작용과 사회적 학습 과정을 모델링합니다.

- **Performance Highlights**: HGRL 프레임워크는 다양한 환경 조건에서 기존 방법들보다 우수한 성능을 보였으며, 낮은 사회적 학습에서는 협력을 유지하고 강력한 코어-페리 네트워크를 형성하는 반면, 높은 사회적 학습에서는 배신이 빠르게 퍼져 희소한 체인 형태의 네트워크로 변화하는 경향을 보였습니다.



### Resource Governance in Networked Systems via Integrated Variational Autoencoders and Reinforcement Learning (https://arxiv.org/abs/2410.23393)
- **What's New**: 본 논문은 Variational Autoencoders (VAE)와 Reinforcement Learning (RL)을 통합하여 다중 에이전트 시스템에서 시스템 성능과 자원 사용을 균형 있게 조정하는 새로운 프레임워크인 VAE-RL을 제안합니다. 주요 혁신점은 대규모의 네트워크 구조의 행동 공간을 효과적으로 처리할 수 있는 능력입니다.

- **Technical Details**: VAE-RL에서는 VAE를 활용하여 네트워크 구조의 대규모 이산 행동 공간을 관리 가능한 연속 잠재 공간으로 변환합니다. 이렇게 변환된 연속 공간에서 Deep Deterministic Policy Gradient (DDPG)와 같은 연속 행동 DRL 알고리즘을 적용하며, 선택된 잠재 행동을 네트워크 구조로 복원하는 사전 학습된 VAE 디코더를 사용합니다.

- **Performance Highlights**: VAE-RL 프레임워크는 수정된 OpenAI Gym 입자 환경에서 평가되었으며, 다양한 시나리오에서 기본 방법들보다 우수한 성능을 보였습니다. 이 연구는 다중 에이전트 시스템 관리에 대한 의미 있는 패턴과 통찰을 제공합니다.



### STIED: A deep learning model for the SpatioTemporal detection of focal Interictal Epileptiform Discharges with MEG (https://arxiv.org/abs/2410.23386)
Comments:
          10 pages, 7 figures

- **What's New**: 최신 연구에서는 Magnetoencephalography (MEG)를 이용한 간질 환자의 비침습적 진단 방법으로, 기존의 수동 방식에서 벗어나 심층 학습(deep learning)을 이용한 자동 이데이터 탐지 방법이 소개되었습니다.

- **Technical Details**: 이 연구에서 개발된 STIED는 1차원 시간 경과 및 2차원 정위(topography) MEG 신호의 특징을 결합한 두 개의 컨볼루션 신경망(convolutional neural networks)으로 구성된 감독(supervised) 딥러닝 알고리즘입니다. STIED는 간질 환자의 간질 전기적 방출(interictal epileptiform discharges, IEDs)을 시간 및 공간적으로 지역화할 수 있도록 설계되었습니다.

- **Performance Highlights**: STIED 모델은 주로 고주파 긍정파를 가진 국소 간질 환자(FE 그룹)에서 IED를 탐지할 때 85% 이상의 정확도(accuracy), 특이도(specificity), 민감도(sensitivity)를 보였습니다. 또한, STIED는 입력 데이터를 활용하여 기존의 임상 MEG 실습을 모방하여 우수한 성능을 발휘하며, 다른 유형의 저항성 국소간질 환자에게도 적용 시 긍정적인 결과를 보였습니다.



### Estimating Neural Network Robustness via Lipschitz Constant and Architecture Sensitivity (https://arxiv.org/abs/2410.23382)
Comments:
          SAFE-ROL at CoRL 2024

- **What's New**: 이 논문은 로봇 학습 시스템에서 신경망의 강건성을 확보하는 것이 얼마나 중요한지를 설명합니다. 특히, 신경망의 감도와 관련하여 Lipschitz 상수를 핵심 지표로 제시하며, 이것을 통해 강건성을 수량화하고 향상시키는 방법을 제안합니다.

- **Technical Details**: 이 연구는 다양한 신경망 아키텍처에 맞춘 Lipschitz 상수에 대한 분석적 표현을 제시합니다. 실험을 통해 신경망의 설계가 강건성에 미치는 영향을 검증하고, 네트워크의 깊이, 너비 및 기타 아키텍처 선택이 강건성에 미치는 영향을 조사합니다. 가장 작은 Lipschitz 상수를 유지하면서도 비슷한 정확도를 달성할 수 있는 아키텍처를 식별하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 넓은 네트워크가 좁은 네트워크보다 더 강건하며, 얕은 네트워크가 깊은 네트워크보다 더 많은 강건성을 보이는 경향이 있습니다. 또한, 낮은 분산을 가진 가중치를 가진 신경망이 높은 분산을 가진 신경망보다 일반적으로 더 강건하다는 것을 확인하였습니다.



### Non-binary artificial neuron with phase variation implemented on a quantum computer (https://arxiv.org/abs/2410.23373)
Comments:
          11 pages, 7 figures, to be published in Ciência e Natura (ISSN 2179-460X, DOI: https://doi.org/10.5902/2179460X)

- **What's New**: 본 연구에서는 이진 모델을 확장하여 복소수의 위상(phase)을 조작하는 알고리즘을 소개합니다. 특히, 양자 컴퓨터에서 연속 값을 처리하는 뉴런 모델을 제안하고 구현하였습니다.

- **Technical Details**: 이 모델은 입력 벡터를 N 큐비트로 인코딩하여 m=2^N 형태로 확장하며, 두 가지 알고리즘을 사용하여 양자 회로를 구성합니다. 첫 번째는 ‘brute-force’ 방식의 위상 회전 블록이며, 두 번째는 Hypergraph States Generation Subroutine (HSGS) 알고리즘의 변형입니다.

- **Performance Highlights**: 시뮬레이션을 통해 제안된 알고리즘이 기존의 고전적 접근 방식과 유사한 결과를 제공하고 HSGS 알고리즘의 최적화를 보여주었습니다. 간단한 패턴 인식을 위한 훈련 스킴도 시뮬레이션하여 2+1 큐비트를 활용한 결과를 도출했습니다.



### Sequential Order-Robust Mamba for Time Series Forecasting (https://arxiv.org/abs/2410.23356)
Comments:
          NeurIPS Workshop on Time Series in the Age of Large Models, 2024

- **What's New**: Mamba는 최근 데이터 처리 성능을 향상시키기 위한 새로운 방안으로 주목받고 있으며, SOR-Mamba는 전통적인 TS 예측 방법의 한계를 극복하기 위한 방법을 제안합니다.

- **Technical Details**: SOR-Mamba는 두 가지 주요 특징을 갖고 있습니다: 1) 채널 순서가 뒤집힌 데이터로부터 생성된 두 임베딩 벡터 간의 불일치를 최소화하는 정규화 전략을 통합하여 채널 순서에 대한 강인성을 높이고, 2) 순차 데이터에서 로컬 정보를 포착하기 위해 설계된 1D-convolution을 제거합니다. 또한, 채널 상관 관계 모델링(CCM)을 도입하여 내재 공간(latent space)에서 채널 간의 상관 관계를 유지하는 프리트레이닝(pretraining) 작업을 포함합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 제안된 방법이 표준 학습 및 전이 학습(transfer learning) 시나리오 모두에서 효능을 입증함을 보여줍니다.



### ASURA-FDPS-ML: Star-by-star Galaxy Simulations Accelerated by Surrogate Modeling for Supernova Feedback (https://arxiv.org/abs/2410.23346)
Comments:
          20 pages, 14 figures, 3 tables, submitted to ApJ

- **What's New**: 본 논문에서는 고해상도 은하 시뮬레이션을 제안하며, 이를 위해 서그릿 모델(surrogate model)을 사용하여 계산 비용을 약 75% 줄입니다. 이 모델은 초신성(core-collapse supernova, CCSNe)의 피드백을 포함한 별-별 시뮬레이션에서 나타나는 병목 현상을 완화합니다.

- **Technical Details**: 새로운 프레임워크는 직접 수치 시뮬레이션(direct numerical simulations)과 서그릿 모델링(surrogate modeling)을 결합합니다. 이를 통해 별 형성 역사(star formation history)와 은하의 유출 속도(outflow rates)의 시간적 진화(time evolution)가 해결된 직접 수치 시뮬레이션에서 얻은 결과와 일치합니다. 이 연구는 기계 학습(machine learning) 및 깁스 샘플링(Gibbs sampling) 기법을 활용하여 계산 비용을 절감하면서도 높은 해상도 유지(fidelity)를 가능하게 합니다.

- **Performance Highlights**: 우리의 접근 방식은 직접 수치 시뮬레이션에서 달성한 결과에 비견되며, 물리적 스케일 간의 간극(physical scale gap)을 메우고 다중 스케일 시뮬레이션(multi-scale simulations)을 가능하게 합니다.



### MoLE: Enhancing Human-centric Text-to-image Diffusion via Mixture of Low-rank Experts (https://arxiv.org/abs/2410.23332)
Comments:
          Published at NeurIPS 2024

- **What's New**: 이번 연구에서는 사람 중심의 텍스트-이미지 생성에 초점을 두고, 인간의 얼굴과 손을 포함한 이미지를 보다 자연스럽게 생성하기 위한 두 가지 방법론을 제안합니다. 첫째, 100만 개 이상의 고품질 인간 중심 이미지를 포함하는 데이터셋을 수집하였고, 둘째, ‘Mixture of Low-rank Experts (MoLE)’라는 새로운 방법론을 도입하였습니다.

- **Technical Details**: 연구진은 다양한 인종, 제스처 및 활동을 포함하여 분산 모델의 성능 향상을 위한 충분한 지식을 제공하는 고품질의 ‘human-centric dataset’(인간 중심 데이터셋)을 구성하고자 하였습니다. MoLE는 각각의 데이터셋에 대해 훈련된 저계수 모듈을 전문가로 보고, 이들 모듈을 소프트 어사인먼트를 통해 선택적으로 활성화하는 방식으로 작동합니다.

- **Performance Highlights**: MoLE는 COCO Caption 및 DiffusionDB를 이용한 두 가지 평가 벤치마크에서 여러 메트릭과 인간 평가를 통해 기존의 최첨단 모델들에 비해 우수성을 입증하였습니다. MoLE는 SD v1.5, v2.1 및 XL을 통해 일관되게 성능이 향상됨을 보였습니다.



### CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP (https://arxiv.org/abs/2410.23330)
- **What's New**: 이번 연구에서는 CLIP 모델에서 특정 데이터를 효과적으로 제거하는 새로운 접근 방식인 CLIPErase를 제안합니다. 이는 기존의 unimodal 모델 기법을 넘어 multimodal 환경에서의 machine unlearning 문제에 초점을 맞춥니다.

- **Technical Details**: CLIPErase는 세 가지 주요 모듈로 구성됩니다: (1) Forgetting Module은 지우기 원하는 데이터와의 연결을 분해합니다. (2) Retention Module은 남겨두는 데이터에서 모델 성능이 유지되도록 보장합니다. (3) Consistency Module은 원본 모델과의 일관성을 유지합니다. 이 세 가지 모듈에서 파생된 손실 함수는 공동으로 최소화되어 unlearning을 수행합니다.

- **Performance Highlights**: CIFAR-100과 Flickr30K 데이터셋에 대한 실험 결과, CLIPErase는 다양한 zero-shot 작업에서 지정된 연결을 효과적으로 잊어버리면서도 남겨두는 세트에서의 모델 성능을 유지함을 증명했습니다.



### Variable Resolution Sampling and Deep Learning Image Recovery for Accelerated Multi-Spectral MRI Near Metal Implants (https://arxiv.org/abs/2410.23329)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구는 금속 임플란트 근처의 다중 스펙트럼(Multi-Spectral) MRI에서 스캔 시간을 줄이면서 이미지 품질을 유지하는 변동 해상도(Variable Resolution, VR) 샘플링 및 딥러닝(reconstruction) 재구성 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 1.5T MSI 무릎 및 엉덩이 데이터를 사용하여 새로운 스펙트럴 언더샘플링(spectral undersampling) 방식을 통해 약 40%의 획득 효율을 개선했습니다. U-Net 기반의 딥러닝 모델을 통해 이미지 재구성이 이루어졌으며, SSIM, PSNR 및 RESI 메트릭을 사용하여 이미지 품질을 평가했습니다.

- **Performance Highlights**: 딥러닝 재구성(DL-VR)된 언더샘플링 VR 데이터는 기존 재구성(CR-VR)에 비해 SSIM 및 PSNR 값이 유의미하게 높았으며(p<0.001), 엣지 샤프니스(edge sharpness)가 개선되었습니다. DL로 재구성된 이미지의 엣지 샤프니스는 전체 샘플링 참조 이미지와 일치했습니다(p=0.5).



### Transfer Learning in Vocal Education: Technical Evaluation of Limited Samples Describing Mezzo-soprano (https://arxiv.org/abs/2410.23325)
- **What's New**: 본 논문은 메조소프라노(Mezzo-soprano)와 같은 희귀 음역의 노래 기법 평가에 있어 효율적인 방법을 제시합니다. 이는 심층 학습(deep learning)을 통한 전이 학습(transfer learning)을 활용하여 이루어진 것입니다.

- **Technical Details**: 연구에서는 ImageNet과 Urbansound8k 데이터셋에서 사전 훈련된 심층 학습 모델을 사용하여 메조소프라노 음성 평가의 정확성을 개선합니다. 그리고 메조소프라노 보컬 세트(Mezzo-soprano Vocal Set, MVS)라는 전용 데이터셋을 구축하여 샘플 부족 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, 전이 학습을 통해 모든 모델의 전체 정확도(OAcc)가 평균 8.3% 증가하며, 최고 정확도는 94.2%에 달했습니다. 이는 음악 교육에서 새로운 정량적 평가 방법을 제시하는 데 기여하고 있습니다.



### Exploiting Phonological Similarities between African Languages to achieve Speech to Speech Translation (https://arxiv.org/abs/2410.23323)
- **What's New**: 이 논문은 아프리카 언어들 간의 언어학적 유사성을 활용하여 직접적인 음성-음성 번역(Direct Speech-to-Speech Translation, S2ST)에 대한 파일럿 연구를 제시합니다. 기존의 데이터 주석이 비효율적이거나 실현 불가능한 경우의 해결책으로, 언어의 동일 계통 내에서 음성 сег먼트를 매핑하는 세그먼트 기반 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 세그먼트 매핑 기법을 활용하여 서로 다른 언어 계통 간에도 음성 세그먼트를 매핑할 수 있습니다. 또한, 내장된 지도 확산(Guided Diffusion) 방식을 통해 음성 데이터를 처리하며, 사전 학습된 쌍 세그먼트를 사용하여 매핑 프로세스를 개선합니다. 이 연구는 케냐 방송 공사(KBC)의 데이터셋을 활용하여 다섯 개 언어 (스와힐리어, 루오어, 키쿠유어, 난디어, 영어)에서 모델의 효능을 평가합니다.

- **Performance Highlights**: 모델은 세그먼트 쌍 매핑 및 번역 품질에서 경쟁력 있는 성능을 보였습니다. 특히, 동일 계통의 언어에서 더 높은 번역 정확도를 보였으며, 세그먼트 길이가 번역의 정확성에 중요한 영향을 미친다는 결과가 나왔습니다. 전통적인 ASR-MT 기법들과 비교했을 때 제안한 모델은 비슷한 번역 성능을 달성했습니다.



### Lina-Speech: Gated Linear Attention is a Fast and Parameter-Efficient Learner for text-to-speech synthesis (https://arxiv.org/abs/2410.23320)
Comments:
          Preprint

- **What's New**: Lina-Speech 모델은 자기-주의(self-attention) 메커니즘을 Gated Linear Attention (GLA)와 같은 새로운 순환 구조로 대체했습니다. 이 모델은 음성 클로닝에서 다수의 음성 샘플을 처리할 수 있고, 합성 과정에서의 컨텍스트 윈도우를 완전하게 활용할 수 있습니다.

- **Technical Details**: Lina-Speech는 기존의 트랜스포머 아키텍처 대신 Gated Linear Attention을 사용하여 모델의 학습 및 추론 속도를 개선합니다. 이 모델은 3~15분 길이의 음성 데이터를 기준으로 평균 20초 이내에 실행될 수 있으며, 파라미터 수가 최대 4배 많은 기존 모델들과 경쟁할 수 있는 성능을 보여줍니다. 또한, Parameter Efficient Fine-Tuning (PEFT) 접근 방식을 채택하여 음성 클로닝을 더욱 효율적으로 수행합니다.

- **Performance Highlights**: Lina-Speech는 기존의 선행 모델들과 비교하여 비슷한 성능을 유지하고, 일부 최신 모델들보다 성능이 우수합니다. 특히, 음성 그리기 과제에서 토큰 수가 적은 구간에서도 지속적인 성능을 발휘합니다.



### VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration (https://arxiv.org/abs/2410.23317)
- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)의 추론 속도를 향상시키기 위해 VL-Cache라는 새로운 Key-Value (KV) 캐시 압축 방법을 제안합니다. 기존의 KV 캐시 압축 방식이 VLM에 최적화되지 않았음을 보여줍니다.

- **Technical Details**: VL-Cache는 VLM의 주의 (attention) 스파시티 패턴을 분석하여 visual 토큰과 언어 토큰을 구분합니다. 레이어 별로 스파시티를 고려하여 KV 캐시 예산을 할당하며, 모달리티 인식 토큰 스코어링 정책을 개발하여 중요한 토큰의 중요도를 평가합니다.

- **Performance Highlights**: 실험 결과, KV 캐시의 10%만 유지해도 전체 작업 정확도를 98% 이상 보존하며, 100개의 토큰 생성을 위한 종단 간 지연 시간을 최대 2.33배 단축시켰습니다. GPU 메모리 사용량을 90% 감소시킴에 따라, 높은 동시성을 지원할 수 있습니다.



### Moral Agency in Silico: Exploring Free Will in Large Language Models (https://arxiv.org/abs/2410.23310)
- **What's New**: 이번 연구에서는 결정론적 시스템, 특히 대형 언어 모델(large language models, LLMs)이 도덕적 행위의 기능적 능력과 호환적 자유 의지(compatibilist free will)를 나타낼 가능성을 조사합니다.

- **Technical Details**: 연구는 데넷(Dennett)의 호환적 프레임워크를 바탕으로 자유 의지에 대한 기능적 정의를 개발하며, 이는 샤논(Shannon)의 정보 이론, 데넷의 호환주의, 플로리디(Floridi)의 정보 철학을 통합한 학제간 이론적 토대에 기반합니다. 이 프레임워크는 도덕적 책임을 결정하는 데 있어 이성의 반응성(reason-responsiveness)과 가치 정렬(value alignment)의 중요성을 강조합니다.

- **Performance Highlights**: 결정의 도덕적 딜레마(moral dilemmas)에서 LLM의 의사결정을 분석했을 때, 이들이 합리적 숙고(rational deliberation) 능력을 갖추고 있으며, 새로운 정보와 식별된 불일치에 대응하여 선택을 조정할 수 있음을 보여줍니다. 따라서 이들은 우리의 자유 의지에 대한 기능적 정의와 일치하는 도덕적 행위의 특징을 나타냅니다.



### Systematically Analyzing Prompt Injection Vulnerabilities in Diverse LLM Architectures (https://arxiv.org/abs/2410.23308)
- **What's New**: 이 연구는 36개의 대형 언어 모델(LLMs)의 다양한 prompt injection 공격에 대한 취약성을 체계적으로 분석하였습니다.

- **Technical Details**: 144개의 prompt injection 테스트를 통해 모델 매개변수(parameter)와 취약성 간의 강한 상관관계를 발견하였으며, 로지스틱 회귀(logistic regression)와 랜덤 포레스트(random forest) 특성 분석(feature analysis) 등의 통계적 분석 결과 매개변수의 크기와 구조가 취약성에 중요한 영향을 미친다는 것을 확인하였습니다.

- **Performance Highlights**: 56%의 테스트에서 성공적인 prompt injection이 발생하여 다양한 매개변수 크기에서 널리 퍼진 취약성을 강조하였으며, 클러스터링 분석을 통해 특정 모델 구성에 연관된 취약성 프로필(vulnerability profiles)을 식별하였습니다. 이러한 결과는 중요한 인프라 및 민감한 산업에 배치된 LLM에 대한 강력하고 다층적인 방어의 필요성을 강조합니다.



### Advanced Cyberattack Detection in Internet of Medical Things (IoMT) Using Convolutional Neural Networks (https://arxiv.org/abs/2410.23306)
Comments:
          7 pages, 4 figures, Accepted at Iranian Conference on Intelligent Systems (ICIS) 23-24 October, 2024, Sirjan University of Technology, Sirjan, Kerman, Iran. \c{opyright} 2024 IEEE. Personal use of this material is permitted. The accepted version is shared here. For the final published version, refer to the IEEE Xplore Digital Library

- **What's New**: 이번 논문에서는 인터넷 의료 사물(IoMT) 시스템 내에서 사이버 공격을 탐지하기 위해 합성곱 신경망(Convolutional Neural Networks, CNNs)을 기반으로 한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 연구에서는 CICIoMT2024 데이터세트를 사용하여 18가지 유형의 사이버 공격을 분석합니다. CNN을 통해 네트워크 트래픽 데이터의 시간적 특성을 효과적으로 분석하며, 관련 연구의 전통적인 기계 학습(machine learning, ML) 모델 및 간단한 심층 신경망(deep neural networks, DNN)과 차별화됩니다.

- **Performance Highlights**: 제안된 CNN 모델은 이진(binar), 범주별(categorical), 다중 클래스(multiclass) 분류 작업에서 99%의 완벽한 정확도를 달성하여 기존의 고급 기법에 비해 뛰어난 성능을 보였습니다. 이는 로지스틱 회귀(Logistic Regression), 아다부스트(AdaBoost), DNNs, 랜덤 포레스트(Random Forests)와 같은 전통적인 ML 모델의 성능을 초월합니다.



### FVEval: Understanding Language Model Capabilities in Formal Verification of Digital Hardwar (https://arxiv.org/abs/2410.23299)
- **What's New**: 본 논문에서는 LLMs(대규모 언어 모델)를 하드웨어 포멀 검증(formal verification, FV) 작업에 효과적으로 발전시키기 위해 첫 번째 종합 벤치마크인 FVEval을 소개합니다. 이 벤치마크는 다양한 수준에서 LLM의 성능을 평가할 수 있는 세 가지 하위 작업으로 구성되어 있습니다.

- **Technical Details**: FVEval은 자연 언어 설명을 바탕으로 시스템베리로그(SystemVerilog) assertions(SVAs)을 생성하는 작업부터 디자인 RTL(레지스터 전송 수준)과 관련된 추론 및 인간의 추가 입력 없이 직접 assertions을 제안하는 작업까지 포함합니다. 우리는 다양한 산업 FV 워크플로우에 맞춘 합성 예제를 생성하며, 기존의 상용 및 오픈소스 LLM들을 평가하여 현재 LLM의 성능을 분석합니다.

- **Performance Highlights**: FVEval의 세 가지 하위 벤치마크는 NL2SVA-Human(인간 작성 테스트벤치 기반 SVA 생성), NL2SVA-Machine(자연 언어를 통한 SVA 구문 생성), Design2SVA(디자인 RTL에서 직접 SVA 제안)의 성능을 측정합니다. 초기 실험에서 LLM들은 각 작업별로 상이한 성능을 보였으며, 특히 NL2SVA-Human에서 높은 정확도를 기록하였습니다.



### Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (https://arxiv.org/abs/2410.04640)
Comments:
          Project page: this https URL. 35 pages, 9 figures. Accepted to the Conference on Robot Learning (CoRL) 2024

- **What's New**: 이 논문에서는 로봇 정책의 실패를 탐지하기 위해 Sentinel이라는 런타임 모니터링 프레임워크를 제안합니다. Sentinel은 고장 탐지를 두 가지 보완적 범주로 나누어 erratic failures (불규칙한 실패)와 task progression failures (작업 진행 실패)를 모니터링합니다.

- **Technical Details**: Sentinel은 Statistical measures of Temporal Action Consistency (STAC)를 사용하여 정책의 행동 분포가 시간이 지남에 따라 얼마나 변하는지를 평가하여 erratic failures를 탐지합니다. 작업 진행 실패는 Vision Language Models (VLMs)을 사용하여 감지하며, 이는 로봇의 진행 상황을 비디오 질문 답변 설정에서 추론할 수 있습니다.

- **Performance Highlights**: Sentinel을 통해 알려지지 않은 실패 상황을 탐지할 수 있으며, 시뮬레이션과 현실 세계의 로봇 모바일 조작 도메인에서 diffusion policies에서 97% 이상의 성공률을 보였습니다. 이 방식은 기존의 두 탐지기를 각각 사용할 때보다 18% 더 많은 실패를 감지하며, 시스템적 성능 향상을 위한 특화된 탐지기를 통한 분할 및 정복 전략을 강조합니다.



### Text2Motion: From Natural Language Instructions to Feasible Plans (https://arxiv.org/abs/2303.12153)
Comments:
          Published in Autonomous Robots, Special Issue: Large Language Models in Robotics 2023. Project page: this https URL. First two authors contributed equally

- **What's New**: 이번 연구는 로봇이 순차적인 조작 작업을 해결하기 위해 장기적 사고(long-horizon reasoning)가 필요한 언어 기반 계획 프레임워크인 Text2Motion을 제안합니다.

- **Technical Details**: Text2Motion은 자연어 지시에 따라 작업과 동작 수준의 계획을 수립하고, 이를 통해 추론된 기호적 목표에 도달할 수 있는지를 검증합니다. 이 프레임워크는 대규모 언어 모델(Large Language Models)의 작업 계획을 안내하기 위해 퀘이션(Q-functions)에 인코딩된 실행 가능성 휴리스틱(feasibility heuristics)을 활용합니다. 기존 언어 기반 계획 기법들은 개별 기술의 실행 가능성만 고려하는 반면, Text2Motion은 기하학적 실행 가능성 계획을 통해 기술 시퀀스 간의 기하학적 종속성을 적극적으로 해결합니다.

- **Performance Highlights**: 실험 결과, Text2Motion은 장기적 사고, 추상적 목표 해석, 부분적인 수용 가능성(perception)의 처리가 필요한 문제를 해결하는 데 82%의 성공률을 보였습니다. 이는 이전의 최첨단 언어 기반 계획 기법이 13%의 성공률을 기록한 것에 비해 상당히 높은 수치입니다.



