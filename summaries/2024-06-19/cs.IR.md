New uploads on arXiv(cs.CL)

### LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation (https://arxiv.org/abs/2406.12832)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)을 작은 파라미터와 메모리 사용량으로 튜닝할 수 있는 새로운 접근법인 LaMDA를 소개합니다. LaMDA는 저차원 적응 (Low-Dimensional Adaptation)을 활용하여 훈련 가능한 파라미터와 GPU 메모리 사용량을 크게 줄입니다. 또한 LaMDA++라는 향상된 버전도 제안하여, 'lite-weight' 방식의 적응형 랭크 할당을 통해 LoRA 방식의 한계를 극복합니다.

- **Technical Details**: LaMDA는 첫 번째 프로젝션 매트릭스(PMA)를 고정한 상태에서 저차원의 사각형 매트릭스를 도입하여 훈련 가능한 파라미터를 줄이고, 튜닝 초기 단계에서 두 번째 프로젝션 매트릭스(PMB)를 서서히 고정하여 계산 비용을 줄입니다. LaMDA++는 사전 훈련된 모델 가중치의 스펙트럼 분석을 통해 각각의 레이어에 적응형 랭크를 '가벼운(lite-weight)' 방식으로 할당하여 성능 향상을 꾀합니다.

- **Performance Highlights**: GLUE 벤치마크, 텍스트 요약, 자연어 생성, 복잡한 추론 등 다양한 작업에서 LaMDA와 LaMDA++를 평가한 결과, 기존 방법들과 비교해 최대 17.7배 적은 파라미터 업데이트와 최대 1.32배 낮은 GPU 메모리 사용량으로 유사하거나 더 높은 성능을 달성했습니다.



### What Are the Odds? Language Models Are Capable of Probabilistic Reasoning (https://arxiv.org/abs/2406.12830)
Comments:
          21 pages, 9 figures, 2 tables

- **What's New**: 이 논문에서는 언어 모델(LMs)이 이상적 및 실제 통계 분포를 사용한 확률적 추론 능력을 평가합니다. 구체적으로, 퍼센타일 추정, 샘플 추출, 확률 계산 세 가지 작업에서 최첨단 언어 모델들을 평가하였습니다. 이러한 작업의 수행을 돕기 위해 분포 내의 고정된 예시, 실제 세계의 맥락, Normal 근사에 기초한 요약 통계를 사용하는 세 가지 방법을 제시했습니다. 이를 통해 모델들이 분포에 대한 추론을 하도록 돕습니다. 이 연구를 위해 관련된 질문-답변 쌍과 함께 포괄적인 벤치마크 분포 데이터셋을 개발하고 공개할 예정입니다.

- **Technical Details**: 언어 모델은 퍼센타일(percentiles) 추정, 샘플(drawing samples) 추출, 확률(calculating probabilities) 계산 세 작업에서 평가되었습니다. 맥락을 제공하는 방법으로는 1) 분포 또는 분포 가족 내의 고정된 예시(anchor examples), 2) 실제 세계의 맥락(real-world context), 3) Normal 근사에 기초한 요약 통계(summary statistics)를 사용했습니다. 이 연구는 분포에 대한 효과적인 추론을 위해 실제 세계 맥락과 예시 샷(example shots)을 통합하고 단순화된 가정(simplified assumptions)을 사용할 수 있음을 보여주었습니다.

- **Performance Highlights**: 모델들은 분포에 대한 추론 능력을 향상시키기 위해 실제 세계 맥락, 고정된 예시, 그리고 단순화된 가정의 도움을 받았습니다. 비록 이러한 가정이 정확하지 않거나 잘못 지정되었더라도 모델들의 성능 향상에 기여했습니다. 본 연구는 이를 위한 체계적인 평가를 통해 그 가능성을 입증했습니다.



### From RAGs to rich parameters: Probing how language models utilize external knowledge over parametric information for factual queries (https://arxiv.org/abs/2406.12824)
- **What's New**: 새로운 연구는 언어 모델(Language Models, LMs)에서 사용되는 Retrieval Augmented Generation (RAG) 기법을 분석합니다. 이 연구는 LMs가 외부 컨텍스트를 활용할 때 내부 매개변수 기억(parametric memory)을 최소한으로 사용하고, 대부분의 정보를 외부 컨텍스트에서 얻는 '지름길 메커니즘(shortcut mechanism)'을 강조합니다.

- **Technical Details**: 이 논문에서는 RAG 파이프라인을 기계적으로 분석하여 언어 모델이 내부 매개변수 기억을 최소한으로 사용하고 외부 컨텍스트에 의존하는 경향이 있다는 것을 밝힙니다. 연구는 다음과 같은 방법을 사용했습니다: (i) 인과적 매개 분석(Causal Mediation Analysis)을 통해 매개변수 기억이 질문에 답할 때 최소한으로 사용된다는 것을 보여줍니다. (ii) 주의 기여(Attention Contributions)와 '강제 차단(knockouts)' 메커니즘을 통해 최종 출력을 형성하는데 중요한 마지막 토큰 레지듀얼 스트림이 질문 내의 주제 토큰에서 정보가 아닌 외부 컨텍스트에서 명시적으로 존재하는 속성 토큰에서 정보를 더 많이 얻는다는 것을 입증했습니다.

- **Performance Highlights**: 이 연구는 다양한 LLaMa 및 Phi 모델에 걸쳐 나타나는 pronounced shortcut behavior를 발견했습니다. 모델들은 내부 지식보다 외부 컨텍스트를 우선시하여 더 나은 성능을 나타낸다는 것을 보여주었습니다.



### Is It Good Data for Multilingual Instruction Tuning or Just Bad Multilingual Evaluation for Large Language Models? (https://arxiv.org/abs/2406.12822)
- **What's New**: 이번 연구는 다국어 대형 언어 모델(multilingual large language models)이 다양한 언어의 원어민에게 적합하도록 설계되고 주장되며 기대되는 현재의 상황을 재검토합니다. 번역에 과도하게 의존하는 기존의 미세 조정(fine-tuning) 및 평가 방식이 이러한 목적에 부합하지 않을 수 있다는 가설을 제시합니다.

- **Technical Details**: 연구진은 번역된 데이터와 원어 데이터의 지시 조정(instruction tuning) 및 평가 단계에서 모델의 결과를 관찰했습니다. 실험은 여덟 가지 기본 모델과 여덟 개의 다른 벤치마크를 통해 이루어졌으며, 특히 모델의 성능이 높은 경우 번역된 지시 데이터와 원어 지시 데이터 사이에서 현저한 차이를 보였습니다.

- **Performance Highlights**: 실험 결과, 번역된 테스트 세트는 이러한 미세한 차이를 포착하지 못하지만, 네이티브 또는 생성 벤치마크에서는 명확한 차이를 나타냈습니다. 마지막으로, 구조화된 작업에서는 정규화(regularization)가 이 격차를 메우는데 유익하지만, 생성 작업에서는 그렇지 않음을 보여줍니다.



### Can Large Language Models Always Solve Easy Problems if They Can Solve Harder Ones? (https://arxiv.org/abs/2406.12809)
Comments:
          25 pages, 12 figures, 10 tables

- **What's New**: 이번 연구는 LLM(대규모 언어 모델)들이 어려운 문제를 해결할 수 있음에도 불구하고 쉬운 문제에서는 종종 실패하는 '어려움-쉬움 불일치' 문제를 다루고자 합니다. 이를 평가하기 위해 ConsisEval 벤치마크를 개발하고, 모델의 일관성을 정량적으로 측정할 수 있는 일관성 점수(consistency score)를 도입했습니다.

- **Technical Details**: ConsisEval 벤치마크는 문제의 난이도가 확실히 구분되는 질문 쌍으로 구성된 데이터셋입니다. 데이터는 주로 코드, 수학, 명령 수행의 세 가지 도메인에서 수집되었으며, 각 도메인마다 쉬운 문제와 어려운 문제 쌍을 포함하고 있습니다. 일관성 점수는 모델이 어려운 문제를 맞혔을 때 쉬운 문제를 맞힐 확률을 기반으로 계산됩니다.

- **Performance Highlights**: 여러 모델을 대상으로 한 종합 실험 결과, GPT-4 모델이 92.2%의 최고 일관성 점수를 기록했습니다. 하지만 특정 질문에 대해서는 여전히 일관성이 떨어지는 경우가 존재했습니다. 뛰어난 능력을 가진 모델일수록 일반적으로 높은 일관성을 보였으며, 어려운 데이터를 사용한 훈련이 일관성 향상에 기여한다는 것도 발견하게 되었습니다.



### ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools (https://arxiv.org/abs/2406.12793)
- **What's New**: 최신 연구에서 ChatGLM 모델 패밀리의 진화한 버전인 GLM-4 시리즈가 소개되었습니다. GLM-4, GLM-4-Air, GLM-4-9B 모델은 이전 세대의 ChatGLM에서 얻은 인사이트를 바탕으로 개발되었습니다. 이 모델들은 주로 중국어와 영어로 된 10조 개의 토큰으로 사전 훈련되었으며, 24개 언어의 소규모 코퍼스도 포함되었습니다. 주로 중국어와 영어 사용을 위해 고도로 정렬된 이 모델들은 다양한 벤치마크에서 높은 성능을 보여줍니다.

- **Technical Details**: GLM-4 모델은 단순한 사전 훈련 과정뿐만 아니라, 감독 학습(Supervised Fine Tuning, SFT) 및 인간의 피드백을 통한 학습을 포함하는 다단계 후속 훈련 과정도 거쳤습니다. 이 과정에서 모델은 사용자의 의도를 이해하고, 웹 브라우저, Python 인터프리터, 텍스트-이미지 모델, 사용자 정의 함수 등 다양한 도구를 사용하여 복잡한 작업을 효과적으로 수행할 수 있도록 조정되었습니다. 또한, FlashAttention 및 Multi-Query Attention 기술이 통합되어 효율성이 높아졌습니다.

- **Performance Highlights**: GLM-4 모델은 여러 벤치마크에서 GPT-4를 따라잡거나 능가하며, 특히 중국어 정렬에서는 GPT-4를 크게 상회하는 성능을 보였습니다. GLM-4(All Tools) 모델은 웹 브라우저와 Python 인터프리터를 활용한 온라인 정보 접근 및 수학 문제 해결과 같은 작업에서 GPT-4 All Tools를 능가했습니다. 또한, ChatGLM 시리즈는 Hugging Face에서 2023년 한 해 동안 1천만 회 이상의 다운로드를 기록하며 많은 주목을 받았습니다.



### Generating Educational Materials with Different Levels of Readability using LLMs (https://arxiv.org/abs/2406.12787)
Comments:
          In2Writing 2024

- **What's New**: 이번 연구는 교육 자료의 의미를 유지하면서 특정 가독성(readability) 수준에 맞게 재작성하는 '레벨 텍스트 생성' 작업을 소개합니다. 주요 대형 언어 모델(LLM)인 GPT-3.5, LLaMA-2 70B, Mixtral 8x7B를 이용해 다양한 가독성 수준에서 콘텐츠를 생성할 수 있는 능력을 평가했습니다.

- **Technical Details**: 이 연구는 제로샷(zero-shot)와 소수 샷(few-shot) 프롬프트를 통해 가독성 수준을 조작하고 정보 보존 성능을 평가했습니다. 100개 교육 자료를 처리하여 소수 샷 프롬프트가 가독성 조작과 정보 보존에 있어 성능을 크게 향상시켰음을 확인했습니다.

- **Performance Highlights**: LLaMA-2 70B 모델은 원하는 난이도 범위를 잘 달성했고, GPT-3.5 모델은 원래 의미를 유지하는 데 뛰어났습니다. 그러나 수작업 검사에서 잘못된 정보 도입 및 편집의 불균일한 분포와 같은 문제점을 발견했습니다. 이는 생성된 교육 콘텐츠의 품질을 보장하기 위해 추가 연구가 필요함을 강조합니다.



### UBENCH: Benchmarking Uncertainty in Large Language Models with Multiple Choice Questions (https://arxiv.org/abs/2406.12784)
Comments:
          Under review

- **What's New**: 최근의 대형 언어 모델(LLMs)의 급격한 발전이 실질적인 문제 해결에 긍정적인 결과를 보여주고 있습니다. 그러나 이러한 모델들의 저조한 해석 가능성은 예측할 수 없는 상황에서 오류를 초래하며, 사용성을 제한합니다. 이러한 문제를 해결하기 위해, UBENCH라는 종합적인 벤치마크를 제안합니다. UBENCH는 지식, 언어, 이해 및 추론 능력을 다루는 3,978개의 다지선다형 문제를 포함하며, LLM의 신뢰성을 평가합니다.

- **Technical Details**: UBENCH는 단일 샘플링 방식(single-sampling method)을 사용해 컴퓨팅 자원을 절약하면서도 정확한 평가를 유지하는데 중점을 둡니다. 기존의 여러 샘플링 방식에 비해 효율성을 크게 향상시켰습니다. UBENCH는 오픈소스 및 폐쇄형 모델 모두를 지원하도록 설계되었으며, 포괄적인 평가를 위해 4가지 주요 카테고리로 질문을 분류합니다. 또한, Chain-of-Thought(CoT) 프롬프트, 역할 플레이(Role-playing) 프롬프트, 옵션 순서(order), 온도(temperature) 등 다양한 측면에서 LLM의 신뢰성을 분석합니다.

- **Performance Highlights**: UBENCH 테스트 결과, 15개의 인기 있는 LLM 중 GLM4가 가장 높은 성능을 보였고, 그 다음으로 GPT-4가 우수한 성적을 기록했습니다. 또한, 비교 실험에서 CoT, 역할 플레이, 옵션 순서, 온도 매개변수의 효과가 다양한 LLM들에서 다르게 나타났습니다. 이러한 결과는 LLM의 더욱 폭넓은 다운스트림 응용에 도움을 줄 것으로 예상됩니다.



### Composited-Nested-Learning with Data Augmentation for Nested Named Entity Recognition (https://arxiv.org/abs/2406.12779)
Comments:
          Accepted by CSCWD 2024

- **What's New**: 최근 논문에서는 중첩 개체 인식(Nested Named Entity Recognition, NNER)을 위한 데이터 증강(data augmentation) 방법을 제안했습니다. 기존의 평면 개체 인식(Flat Named Entity Recognition, FNER)보다 데이터 주석이 부족한 문제를 해결하기 위해 새로운 데이터 증강 기술을 도입했습니다. 중첩-네임 라벨 분류(composited-nested-label classification)와 신뢰도 필터링 메커니즘(Confidence Filtering Mechanism, CFM)을 활용하여 보다 효율적으로 데이터를 선택하고 모델의 성능을 향상시켰습니다.

- **Technical Details**: 본 연구에서는 중첩 개체 인식을 위한 데이터 증강 방법으로 중첩-네임 라벨 분류(composited-nested-label classification)를 도입했습니다. 중첩된 토큰과 라벨을 결합하여 모델링하며, 이를 Composited-Nested-Learning (CNL) 모듈을 통해 데이터 증강에 사용했습니다. 또한, 신뢰도가 높은 데이터를 선택하기 위해 신뢰도 필터링 메커니즘(Confidence Filtering Mechanism, CFM)을 설계했습니다. 이를 통해 모델의 데이터 품질을 향상시켰습니다.

- **Performance Highlights**: 제안된 방법은 ACE2004와 ACE2005 데이터셋에서 성능 향상을 입증했습니다. 샘플 불균형 문제를 완화하고, 개선된 데이터 셋을 오픈 소스로 공개하여 다른 연구자들에게 유용한 데이터를 제공합니다.



### Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries (https://arxiv.org/abs/2406.12775)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 복잡한 다중 단계 문제를 어떻게 해결하는지에 대한 내부 메커니즘을 분석했습니다. 이 연구는 'Imagine의 연주자의 배우자는 누구인가?'와 같은 다중 홉 쿼리에 대한 해답을 찾는 과정을 조사하였습니다. 이러한 쿼리는 두 가지 정보 추출 단계를 필요로 합니다. 첫 번째 단계에서는 중계 엔터티인 John Lennon을, 두번째 단계에서는 최종 목표 엔터티인 Yoko Ono를 추출합니다.

- **Technical Details**: 연구진은 transformer 기반 LLM의 내부 연산을 자세히 분석했습니다. 첫 번째 중계 엔터티를 추출하는 과정은 모델의 초기 레이어에서 진행되고, 두 번째 홉 쿼리는 나중 레이어에서 해결됩니다. 연구진은 '백패칭(back-patching)'이라는 새로운 분석 방법을 제안하여 나중 레이어의 은닉 표현(hidden representation)을 초기 레이어로 다시 패치하는 방법을 사용했습니다. 이는 모델이 더 많은 레이어를 사용하여 연산을 완료할 수 있도록 도와줍니다.

- **Performance Highlights**: 백패칭 방법을 사용한 실험 결과, 처음에는 잘못 예측된 쿼리 중 최대 57%가 이 방법을 통해 올바르게 예측될 수 있음을 확인했습니다. 이는 나중 레이어가 필요한 기능을 항상 가지고 있지 않다는 사실을 강조합니다.



### Chumor 1.0: A Truly Funny and Challenging Chinese Humor Understanding Dataset from Ruo Zhi Ba (https://arxiv.org/abs/2406.12754)
- **What's New**: 기존의 유머 데이터셋과 평가 시스템은 주로 영어에 초점을 맞추고 있으며, 중국어와 같은 비영어권 언어의 문화적으로 정교한 유머에 대한 자원이 부족합니다. 이를 해결하기 위해 연구진은 중국의 Reddit와 같은 플랫폼인 Ruo Zhi Ba(RZB)에서 지적이고 문화적으로 특화된 농담을 공유하는 Chumor 데이터셋을 구축했습니다.

- **Technical Details**: Chumor 데이터셋은 RZB의 'Best Annual Threads'(2018-2021)와 'Moderator’s Recommendation' 섹션에서 수집된 농담들로 구성되어 있습니다. 데이터 전처리 과정에서는 의미 없는 플레이스홀더 텍스트를 제거하고 잘린 제목을 완전한 내용으로 교체하는 등 철저한 필터링을 거쳤습니다. 총 1,951개의 농담에 대한 설명을 수동으로 주석 처리하였습니다.

- **Performance Highlights**: 최신 모델인 GPT-4o(OpenAI)와 ERNIE Bot(Baidu)를 대상으로 한 평가에서, 인간 설명이 모델 설명보다 훨씬 뛰어났습니다. 특히, 중국어 문화와 관련된 농담에서는 GPT-4o의 오타율이 ERNIE Bot보다 상당히 높았습니다. 인간의 설명은 50% 이상의 승률을 기록했으며, 모델의 설명은 2~3%에 불과했습니다.



### OlympicArena: Benchmarking Multi-discipline Cognitive Reasoning for Superintelligent AI (https://arxiv.org/abs/2406.12753)
Comments:
          44 pages

- **What's New**: 올림픽 대회 문제를 통해 AI의 인지적 추론 능력을 평가하는 새로운 벤치마크, OlympicArena가 도입되었습니다. 이 벤치마크는 62개의 국제 올림픽 대회에서 수집한 11,163개의 이중언어 문제를 포함하며 광범위한 분야를 아우릅니다. 이를 통해 AI가 복잡한 과학적 문제를 해결하는 능력을 평가하는데 중점을 둡니다.

- **Technical Details**: OlympicArena는 텍스트 전용 및 텍스트-이미지 혼합 두 가지 모드를 포함하여 다중 모드를 지원합니다. 7개 분야의 문제를 포함하며, 문제는 수학, 물리학, 화학, 생물학, 지리학, 천문학, 컴퓨터 과학에서 가져왔습니다. 벤치마크는 데이터 누출 방지를 위한 철저한 검토를 거쳤으며, 프로세스 수준의 평가를 통해 AI의 단계별 추론 과정을 분석합니다. 또한 이 벤치마크는 영어와 중국어 두 언어를 지원합니다.

- **Performance Highlights**: 최첨단 모델인 GPT-4o도 39.97%의 정확도에 그쳤으며, 다른 모델들은 20%의 정확도도 달성하지 못했습니다. 특히 복잡한 분해적 추론 문제나 공간적, 기하학적 인식 능력에서 어려움을 겪는 것으로 나타났습니다. 모델들이 단계별 추론 과정을 일부 올바르게 수행할 수 있었지만, 여전히 복합적인 인지적 추론 문제에서 고전하고 있음을 보여줍니다. 이 벤치마크는 AI의 현재 한계와 가능성을 더욱 명확히 파악하는 데 기여할 것입니다.



### Rationale-based Ensemble of Multiple QA Strategies for Zero-shot Knowledge-based VQA (https://arxiv.org/abs/2406.12746)
- **What's New**: 이번 연구에서는 K-VQA(Knowledge-based Visual Question-answering, 지식 기반 시각 질문 및 응답) 시스템을 위한 새로운 접근 방식인 REACT(Rationale-based Ensemble of Answer Context Tactics)를 제안합니다. 이 방법은 다양한 질문-응답 전술(question-answering tactics)을 동적으로 조합하여 다양한 형태의 배경 정보를 활용합니다. 이를 통해 OK-VQA와 A-OKVQA 데이터셋에서 최첨단 성능을 달성했습니다.

- **Technical Details**: REACT는 두 가지 주요 모듈로 구성되어 있습니다. 'Answer Candidate Generation' (ACG) 단계에서는 이미지 설명(captions), 짧은 형태의 지식(단문), 긴 형태의 지식(장문) 등 세 가지 다른 결정(context)을 생성하여 질문에 대응합니다. 'Rationale-based Strategy Fusion' (RSF) 단계에서는 자동 또는 기계적으로 생성된 논리를 사용하여 올바른 답을 선택합니다. 이 단계는 Frozen 상태의 대형 언어 모델(LLM)을 사용하여 문맥 학습(in-context learning)을 통해 작동합니다.

- **Performance Highlights**: REACT는 OK-VQA와 A-OKVQA 데이터셋에서 기존 최첨단 기법 대비 2.6%에서 4.7%의 성능 향상을 달성했습니다. 또한, 세 가지 응답을 융합하는 방식이 단일 응답 전략을 사용하는 것보다 OK-VQA에서는 1.1%, A-OKVQA에서는 1.6% 성능 향상을 보였습니다. 자동 논리와 기계적 논리를 모두 사용하는 방식이 각각 1.2%와 1.4%의 성능 개선을 이루었습니다.



### Self-Distillation for Model Stacking Unlocks Cross-Lingual NLU in 200+ Languages (https://arxiv.org/abs/2406.12739)
- **What's New**: 이번 연구에서는 기계 번역(MT) 인코더를 대형 언어 모델(LLM) 백본에 직접 통합하여 다국어 자연어 이해(NLU)를 강화하는 새로운 방법을 제안합니다. 이 접근법은 MT-LLM이라는 통합된 다국어 모델을 통해 저자원 언어들이 뛰어난 영어 중심 LLM의 지식을 활용할 수 있게 함으로써, LLM의 다국어 NLU 성능을 실제로 향상시킵니다.

- **Technical Details**: 이 연구에서는 샘플 효율적인 자가 증류(self-distillation) 방법을 통해 MT 인코더를 LLM에 직접 통합하였습니다. 통합 과정은 두 단계로 이루어지며, 첫 번째 단계는 시퀀스 수준의 정렬을 위한 자가 감독 적응(self-supervised adaptation)이고, 두 번째 단계는 작업별 자가 증류(task-specific self-distillation)입니다. 이로써 MT 인코더의 출력을 LLM에 직접 통합하여 모든 언어에 대해 일관된 성능 향상을 이루어냅니다.

- **Performance Highlights**: MT-LLMs는 세 가지 주요 NLU 작업과 127개의 저자원 언어에 대한 평가에서 뛰어난 성능을 보여주었습니다. 기존의 테스트 번역(translate-test) 방법을 크게 능가하며, 번역 오류의 전파와 MT 디코딩의 추론 오버헤드를 완화할 수 있었습니다. 또한, MT-LLM 접근법은 다양한 유형의 LLM 백본에 대해 효과적이며, 디코더 전용 및 인코더 전용 모델 모두에서 성능 향상을 달성하였습니다.



### Large Language Model as a Universal Clinical Multi-task Decoder (https://arxiv.org/abs/2406.12738)
Comments:
          Work in progress

- **What's New**: 본 논문은 대규모 사전 훈련된 언어 모델(LLM)을 임상 다중 과제 디코더로 활용하는 새로운 패러다임을 제시합니다. 기존의 여러 과제를 다루는 것과 새로운 과제를 유연하게 대응하는 데에 효과적인 접근 방식을 제공합니다. 새로운 과제는 단순히 새로운 지시 템플릿을 추가하는 것으로 도입할 수 있습니다.

- **Technical Details**: 이 접근 방식은 Warpformer와 같은 기존 임상 표현 학습 기법을 활용하여 LLM에 대한 어댑터를 개발하여, 차별화된 클리니컬 신호의 표현을 통합합니다. 이 어댑터는 Warpformer의 출력 표현과 LLM의 입력 간의 연결을 통해 모든 종류의 과제에 대한 일반화된 표현 학습을 가능케 합니다.

- **Performance Highlights**: 논문의 ClinTS-LLM 프레임워크는 전통적인 다중 과제 학습(MTL) 및 단일 과제 학습(STL) 접근 방식과 비교하여 견고한 성능을 보였습니다. 특히 새로운 과제에 대한 적응성이 뛰어나서, 일부 사례에서는 zero-shot 학습에서의 놀라운 성과를 보여주었고, few-shot 학습에서는 데이터 효율성 면에서 뛰어난 성능을 입증했습니다.



### Can Large Language Models Code Like a Linguist?: A Case Study in Low Resource Sound Law Induction (https://arxiv.org/abs/2406.12725)
- **What's New**: 이번 연구에서는 역문법학(역사 언어학)에서 사용되는 '음운법(sound laws)' 유도 문제를 예제기반 프로그래밍(Programming by Examples)으로 접근하는 새로운 방법을 제안했습니다. 이는 대형 언어 모델(LLM)을 활용하여 파이썬(Python) 음운법 프로그램을 자동 생성하는 방식입니다.

- **Technical Details**: 연구진은 LLM의 프로그래밍 능력을 이용하여, 다양한 언어에 상관없이 음운 변화 예시로부터 Python 음운법 프로그램을 생성하는 방법을 개발했습니다. 이를 위해 LLM을 미세 조정(fine-tuning)하기 위해 추가적인 언어 비의존적 인공 데이터를 생성하는 효과적인 방법도 제안되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 자동 음운법 유도(Sound Law Induction) 방법과 비교했을 때 아직 뒤처지지만, 기존 방법이 가진 몇 가지 약점을 보완할 수 있는 가능성을 보였습니다.



### On the Robustness of Language Models for Tabular Question Answering (https://arxiv.org/abs/2406.12719)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 테이블 질문 응답(TQA) 작업에서 구조적으로 복잡한 데이터를 어떻게 처리하고 이해하는지에 대한 영향을 평가합니다. 특히, in-context learning(맥락 학습), model scale(모델 규모), instruction tuning(명령 조정), domain biases(도메인 편향)가 TQA에 미치는 영향을 조사합니다. 핵심 발견은 명령이 성능을 크게 향상시키며, 최근 모델인 Llama3가 이전 모델보다 더 강력하다는 것을 보여줍니다. 그러나 데이터 오염과 실질적인 신뢰성 문제는 여전히 존재합니다.

- **Technical Details**: 라지 랭귀지 모델들이 명시적으로 훈련받지 않았음에도 텍스트 이해 작업뿐만 아니라 테이블 이해 작업에서도 놀라운 성과를 내고 있습니다. 우리는 Wikipedia 기반 WTQ 데이터셋과 금융 보고서 기반 TAT-QA 데이터셋을 사용하여 LLM의 견고성을 평가했습니다. 실험은 SP, VP, RVP 등 다양한 테이블 구조 및 데이터 값 변형을 통해 모델의 견고성을 실험했습니다. 구조 인지 셀프 어텐션 메커니즘과 도메인 전용 데이터 처리가 필요하다는 점을 강조합니다.

- **Performance Highlights**: 1. 인스트럭션 튜닝이 테이블 이해 성능을 크게 향상시킵니다. Llama3 모델은 특히 강력한 성능을 보였습니다.
2. 더 큰 모델(Llama3-70b)이 작은 모델(Llama2-7b)보다 TQA 성능이 우수합니다.
3. 다양한 데이터 변형에서 성능 저하가 나타났기 때문에 구조적 인식 능력이 아직 개선될 필요가 있습니다.
4. 특정 작업에 대한 튜닝이 모델의 복잡한 추론 작업을 더 잘 처리할 수 있도록 돕습니다.



### AgentReview: Exploring Peer Review Dynamics with LLM Agents (https://arxiv.org/abs/2406.12708)
Comments:
          22 pages, 10 figures

- **What's New**: 새로운 연구로, AgentReview라는 Peer Review 시뮬레이션 프레임워크를 소개합니다. 이 시스템은 Large Language Model(LLM)을 사용하여 검토자들의 편견으로 인해 논문 결정이 37.1%까지 변동할 수 있음을 밝혀냈습니다.

- **Technical Details**: 실험에서 GPT-4 모델을 사용하여 논문 수락 및 거절 사유를 분류하였습니다. 설정된 베이스라인 기준으로 실험하여 일관된 결과를 보장했습니다. 다양한 모델을 테스트했지만, GPT-4가 가장 현실적이고 일관된 출력을 제공합니다.

- **Performance Highlights**: AgentReview의 분석에 따르면 검토자들의 인지된 저자의 명성에 의해 평균 평점이 일관되게 증가했습니다. 또한 무책임한 검토자나 악의적인 설정에서 평점 분포가 양극화되는 현상을 보였습니다.



### Talk With Human-like Agents: Empathetic Dialogue Through Perceptible Acoustic Reception and Reaction (https://arxiv.org/abs/2406.12707)
Comments:
          9 pages, 3 figures, ACL24 accepted

- **What's New**: 이 논문에서는 음성의 음향 정보를 정확하게 인식하고 이를 기반으로 더욱 공감할 수 있는 다중 모달 대화 시스템인 PerceptiveAgent를 제안합니다. 기존의 텍스트 기반 대화 시스템들은 인류와 AI 간의 소통에서 중요한 음향 정보를 간과하고 있으며, 이는 대화 중 화자의 의도를 오해하여 일관성 없는 응답을 초래할 수 있습니다. PerceptiveAgent는 이러한 문제를 해결하고자 음성 모듈리티를 통합하여 화자의 진정한 의도를 보다 깊고 미묘하게 이해할 수 있도록 설계되었습니다.

- **Technical Details**: PerceptiveAgent는 LLMs(Large Language Models)을 인지적 핵심으로 사용합니다. 음성 입력으로부터 음향 정보를 인식하고 이를 바탕으로 공감할 수 있는 응답을 생성합니다. 이를 위해 PerceptiveAgent는 perceptive captioner 모델을 통해 각 대화의 음성에서 음향 특징을 캡처합니다. 그런 다음 LLM 모듈이 관련 응답 콘텐트를 생성하고, Multi-Speaker and Multi-Attribute Synthesizer (MSMA-Synthesizer)가 미묘하고 표현력 있는 음성을 합성합니다. 구체적으로는, 음성 캡션 모델은 자연어로 음향 정보를 감지하고 표현하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, PerceptiveAgent는 언어적 의미가 화자의 진정한 감정과 상반되거나 일치하지 않는 상황에서도 화자의 진정한 의도를 정확하게 파악하여 보다 세밀하고 표현력 있는 음성 대화를 생성할 수 있는 능력을 보여줍니다. 이는 PerceptiveAgent가 사회적으로 중요한 공감적 대화의 맥락을 더욱 잘 이해할 수 있는 시스템임을 시사합니다.



### Jailbreak Paradox: The Achilles' Heel of LLMs (https://arxiv.org/abs/2406.12702)
- **What's New**: 새로운 논문에서는 '파운데이션 모델(foundation models)'의 탈옥(jailbreak)과 관련된 두 가지 역설을 소개합니다. 첫 번째는 완벽한 탈옥 감지기를 만드는 것이 불가능하다는 것이며, 두 번째는 더 약한 모델이 더 강한 모델이 탈옥되었는지 일관되게 감지할 수 없다는 것입니다. 이를 입증하기 위해 공식적인 증거를 제공하고, Llama와 GPT-4o를 사용한 사례 연구를 제시합니다.

- **Technical Details**: 이 논문은 강력한 모델이 더 쉽게 탈옥될 수 있다는 역설을 다룹니다. 이는 특정 목적에 대한 고정적이고 결정적인 정의가 없을 경우 모든 모델이 탈옥될 수밖에 없다는 점에서 기인합니다. 논문에서는 이를 증명하기 위해 '불결정성 결과(undecidability results)'와 '캔터의 대각선화(Cantor’s diagonalization)' 기법을 사용합니다. 이러한 역설들은 모델이 생성한 텍스트나 콘텐츠를 자동으로 감지하는 문제와도 관련이 있습니다.

- **Performance Highlights**: 논문에서는 Llama-2와 GPT-4o에서 실험을 수행하여 주장한 역설에 대한 증거를 제공했습니다. 이 실험은 강력한 모델이 탈옥되는 것을 감지하는 것이 얼마나 어려운지, 그리고 약한 모델이 강한 모델의 탈옥 상태를 일관되게 감지할 수 없다는 것을 보여줍니다.



### MAGIC: Generating Self-Correction Guideline for In-Context Text-to-SQL (https://arxiv.org/abs/2406.12692)
Comments:
          20 pages, 17 figures

- **What's New**: MAGIC이라는 새로운 멀티 에이전트 방법을 소개했습니다. 이 방법은 텍스트에서 SQL로 변환하는 과정에서 발생하는 오류를 자동으로 수정하는 가이드라인을 생성합니다. 기존 방법들과 달리, 인간의 개입 없이 에이전트들이 협업하여 가이드라인을 만듭니다.

- **Technical Details**: MAGIC은 매니저, 수정 및 피드백 에이전트라는 세 가지 특수 에이전트를 사용합니다. 이 에이전트들은 임무를 나누어 협력하며, 초기 텍스트에서 SQL로 변환하는 시스템에서 발생한 오류를 반복적으로 분석하고 교정 가이드라인을 생성합니다. 이 가이드라인은 이후 초기 시스템과 통합되어 잘못된 변환을 예방합니다.

- **Performance Highlights**: MAGIC이 생성한 가이드라인은 사람이 작성한 가이드라인보다 뛰어난 성능을 보였습니다. 특정 사례에서는 GPT-4가 잘못 생성한 SQL을 스스로 교정할 때 MAGIC 가이드라인을 따라 질문을 던지고 교정을 수행했습니다. 또한 MAGIC은 더 나은 해석 가능성을 제공하여 LLM이 왜 실패하거나 성공했는지 분석하는 데 도움을 줍니다.



### Using LLMs to Aid Annotation and Collection of Clinically-Enriched Data in Bipolar Disorder and Schizophrenia (https://arxiv.org/abs/2406.12687)
- **What's New**: 이 논문은 정신 건강 연구에 있어 최신 언어 모델(Language Models)의 적용 가능성을 보여줍니다. 특히, 이 연구는 현대 언어 모델을 활용하여 정신 건강 도구의 배포, 데이터 수집 및 주석 작업을 높은 정확도와 확장성으로 지원하는 방법에 대해 설명합니다. 연구 결과, 작은 모델들이 상업용 대형 언어 모델들보다 도메인별 임상 변수 주석 작업과 정신 건강 도구를 위한 데이터 수집에서 더 나은 성과를 보였습니다.

- **Technical Details**: 연구는 조울증(Bipolar Disorder, BD), 조현병(Schizophrenia, SZ), 건강한 통제군(Healthy Controls, HC) 세 그룹의 참가자 644명을 모집한 데이터셋을 사용하였습니다. 참가자들은 표준화된 정신 건강 도구를 기반으로 한 과제를 수행했고, 결과 데이터는 전문가들이 5개의 임상 변수에 대해 전사 및 주석을 달았습니다. 이 변수에는 흥미/무관심(Interest/Disinterest), 유창성(Fluency), 명확성(Clarity), 집중도(Focus), 그리고 사회적 적절성(Social Appropriateness)이 포함됩니다. 각 변수를 기준으로 전문가들로부터 높은 수준의 일치도(κ ≥ 0.85)를 얻었습니다.

- **Performance Highlights**: 연구 결과, 작은 언어 모델들이 도메인별 임상 변수에 대한 주석 작업 및 정신 건강 도구를 위한 데이터 수집에서 상업용 대형 모델들(GPT-4 등)보다 더 나은 성능을 보였습니다. 모델은 낮은 오류율과 높은 정확도를 달성하였으며, 전문가 수준의 주석 품질을 제공했습니다.



### Measuring Psychological Depth in Language Models (https://arxiv.org/abs/2406.12680)
Comments:
          Preprint. Under Review

- **What's New**: 이번 연구는 심리적 깊이 척도(Psychological Depth Scale, PDS)를 도입했습니다. 이는 독자의 관점에서 인간과 LLM(대형 언어 모델)이 생성한 이야기의 정서, 공감, 몰입 등을 측정합니다. 이 연구는 독자와 연결되는 이야기의 힘을 탐구하는 중요한 전환점입니다.

- **Technical Details**: PDS는 두 가지 문학 이론인 독자 반응 비평과 텍스트 세계 이론에 기반합니다. 이 척도는 감정 유발, 공감, 몰입, 진정성(authenticity), 내러티브 복잡성(narrative complexity)의 다섯 가지 메타 구성 요소를 사용하여 이야기를 평가합니다. GPT-4o는 새로운 'Mixture-of-Personas' 프롬프트 전략과 결합하여 사람의 판단과 0.51의 스피어만 상관관계를 얻었고, Llama-3-70B는 공감 측면에서 0.68의 상관관계를 달성했습니다.

- **Performance Highlights**: 놀랍게도, GPT-4가 작성한 이야기는 내러티브 복잡성과 공감 측면에서 인간이 작성한 인기 있는 이야기보다 뛰어났으며, 다른 구성 요소에서도 통계적으로 구별되지 않았습니다. 이는 GPT-4가 이미 Reddit의 높은 평가를 받은 이야기의 품질과 일치하거나 이를 초과함을 보여줍니다.



### Vernacular? I Barely Know Her: Challenges with Style Control and Stereotyping (https://arxiv.org/abs/2406.12679)
- **What's New**: 이번 연구는 현대 대규모 언어 모델(LLM)이 텍스트의 스타일을 제어하는 능력과 한계를 평가합니다. GPT-3.5, GPT-4, GPT-4o, Llama-3, Mistral-instruct-7B 등 다섯 가지 최신 모델을 두 가지 스타일 제어 작업에 대하여 실험하였습니다.

- **Technical Details**: 첫 번째 작업은 읽기 수준을 제어하여 초등학교 1학년 수준의 텍스트를 생성하는 것이었고, 두 번째 작업은 수사적 감수성과 다양성을 높이기 위해 아프리카계 미국인 영어(AAE/AAVE)으로 질문에 응답하는 것이었습니다. Flesh-Kincaid 읽기 수준과 Blodgett 등의 논문에 따른 AAE/AAVE의 어휘 기반 점수로 성능을 평가했습니다.

- **Performance Highlights**: 첫 번째 작업에서는 모델들의 평균 성능이 5~8학년 읽기 수준으로 나타났으며, 표준편차가 27.6까지 이르렀습니다. 두 번째 작업에서는 성능이 눈에 띄게 향상되어 0.02에서 0.26으로 증가했습니다. 그러나 AAE/AAVE 작업에서 공정한 예시를 제공해도 모델들은 종종 문화적으로 민감하지 않은 내용을 생성했습니다. 또한, 모델들이 놈을 제어할 수 있었지만, 여전히 내재된 편향성과 인종차별적 요소가 남아 있음을 확인했습니다.

- **Conclusion**: 현대 LLM들은 텍스트의 단순함을 어느 정도 제어할 수 있지만 문화적 감수성과 관련성에서는 여전히 부족합니다. 또한, 내재된 부정적 고정관념을 관리하는데 한계를 보였습니다.



### Estimating Knowledge in Large Language Models Without Generating a Single Token (https://arxiv.org/abs/2406.12673)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 지식을 평가하는 새로운 방법인 KEEN(Knowledge Estimation of ENtities)를 제안합니다. 기존 평가 방식이 모델이 생성한 텍스트를 분석하는 것인 반면, KEEN은 텍스트 생성 이전의 내부 계산을 통해 모델의 지식을 평가하는 방법입니다.

- **Technical Details**: KEEN은 모델의 내부 주제 표현을 기반으로 한 간단한 프로빙(probing) 기법으로, 주어진 주제 엔티티에 대한 모델의 지식 수준을 추정합니다. 실험은 팩트 질문 응답(QA)과 바이오그래피 생성(OEG) 상황에서 이루어졌습니다. KEEN은 모델의 정확도와 팩투얼리티(factuality)와의 상관관계가 높게 나타났습니다.

- **Performance Highlights**: 다양한 모델(GPT-2, Pythia, LLaMA2, Vicuna) 실험 결과 KEEN은 모델 정확도와 0.58-0.68, 팩투얼리티와 0.66-0.77의 상관관계를 보였습니다. 또한, KEEN은 주제 엔티티의 이름만으로 모델의 지식을 평가할 수 있고, 후개인조정(fine-tuning) 이후의 지식 변화도 반영합니다. 이 방식은 모델이 예측한 엔티티와 관련된 토큰들에 대해 해석 가능한 결과를 제공합니다.



### CollabStory: Multi-LLM Collaborative Story Generation and Authorship Analysis (https://arxiv.org/abs/2406.12665)
- **What's New**: 다수의 대형 언어 모델(LLMs)을 활용한 협업 스토리 작성 데이터셋인 'CollabStory'가 발표되었습니다. 이 데이터셋은 기존의 인간-LLM 협업을 넘어서 여러 LLM들이 공동으로 이야기를 작성하는 시나리오를 탐구한 최초의 시도입니다. 총 32,000개 이상의 이야기를 생성하였으며, 인간 간 협업 작문 연구의 표준을 설정한 PAN 과제에서 영감을 받아 개발되었습니다.

- **Technical Details**: 데이터셋은 Meta의 Llama, Mistral.ai의 Mistral, Google의 Gemma, AllenAI의 Olmo, Microsoft의 Orca와 같은 오픈 소스 LLM 들을 사용하여 생성되었습니다. 협업 시나리오는 단일 저자(N=1)부터 최대 5명(N=5)의 저자까지 확장하여, 각 LLM들이 교대로 이야기를 작성하는 방식으로 설정되었습니다. 이를 위해 Writing Prompts(WP) 데이터셋의 창의적 글쓰기 프롬프트를 사용하여 각 LLM에 질문 형식으로 데이터를 생성하였습니다.

- **Performance Highlights**: 연구 결과, 현재의 베이스라인들은 다수의 LLM이 협업하는 시나리오를 제대로 처리하지 못한다는 것을 확인했습니다. 이는 향후 기술 개발과 관련된 연구와 분석을 지원할 수 있는 새로운 방법론의 필요성을 강조합니다. CollabStory는 표절 탐지, 학점 배정 및 저작권 침해 문제 등의 도전 과제를 해결하기 위한 연구에 유용한 자원이 될 것입니다.



### Evaluating Transparency of Machine Generated Fact Checking Explanations (https://arxiv.org/abs/2406.12645)
- **What's New**: 사실 확인(fact-checking)을 위한 설명을 생성할 때, 인간이 수집한 증거와 기계가 선택한 증거의 영향을 비교한 연구입니다. 흥미롭게도, 대형 언어 모델(LLM)은 기계가 선택한 증거로부터 더 나은 품질의 설명을 생성할 수 있다는 결과를 발견했습니다. 이는 사실 확인 설명 생성에서 인간이 신중하게 큐레이션한 증거가 필요하지 않을 수 있음을 시사합니다.

- **Technical Details**: 이 연구에서는 설명의 투명성(transparency)과 유틸리티(utility)를 평가 기준으로 삼았습니다. 투명성은 설명이 적절하게 출처를 언급하는지 평가하며, 유틸리티는 설명이 사용자가 주장을 명확히 이해하는 데 도움을 주는 정도를 평가합니다. 이를 위해 인간 평가자들이 인용 마커가 숨겨진 문장을 보고 실제 출처를 찾아내는 작업을 수행했습니다. LLM을 사용하여 설명을 생성하기 전에 기계 선별 증거와 인간 선별 증거를 이용했으며, 다양한 LLM을 통해 실험을 진행했습니다.

- **Performance Highlights**: 주요 발견사항은 다음과 같습니다: (1) 어떤 LLM을 사용하느냐에 따라 기계가 선택한 증거가 인간이 선택한 증거와 비슷하거나 더 우수한 설명을 생성할 수 있음; (2) 기계가 선택한 증거가 수작업으로 선택된 증거보다 관련 증거 문서의 범위가 더 넓음; (3) 설명 생성의 성능은 향상되고 있지만, 여전히 개선의 여지가 남아 있음.



### Hierarchical Prompting Taxonomy: A Universal Evaluation Framework for Large Language Models (https://arxiv.org/abs/2406.12644)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 다양한 과제를 해결하는 능력을 평가하기 위해 '계층적 프롬프트 분류 체계(HPT, Hierarchical Prompting Taxonomy)'를 도입했습니다. HPT는 다섯 가지 프롬프트 전략을 사용하여 LLM과 데이터를 평가하며, 이로써 과제의 복잡성을 더 명확히 이해할 수 있게 합니다. 또한 HPT는 적응형 계층적 프롬프트 프레임워크(Adaptive Hierarchical Prompt framework)를 도입하여 각 과제에 가장 적합한 프롬프트 전략을 자동으로 선택하도록 합니다.

- **Technical Details**: HPT는 단순한 것에서 복잡한 것까지 다섯 가지 프롬프트 전략으로 구성된 계층적 프롬프트 프레임워크(HPF)를 사용합니다. 이 전략들은 과제 복잡성 수준에 따라 다르게 적용되며, 이를 통해 보다 정밀한 평가가 가능합니다. LLM의 성능에 따라 HP-Score를 부여하여 데이터셋과 LLM의 과제 해결 능력을 평가합니다. 이러한 점수를 바탕으로 연구는 Llama 3 8B, Phi 3 3.8B, Mistral 7B, Gemma 7B 네 가지 LLM을 BoolQ, CommonSenseQA (CSQA), IWSLT-2017 en-fr (IWSLT), SamSum 네 가지 데이터셋을 통해 비교하였습니다.

- **Performance Highlights**: 실험 결과, HPT는 다양한 과제와 LLM 성능을 비교하는 데 있어 신뢰할 수 있는 방법임이 증명되었습니다. 특히, 적응형 HPF를 도입하여 과제의 복잡성에 따라 동적으로 프롬프트 전략을 선택함으로써 평가 과정의 자동화와 향상된 이해가 가능해졌습니다. 이 연구는 LLM과 데이터셋 모두의 복잡성을 평가할 수 있는 범용 평가 지표를 개발하는 데 기여합니다.



### DetectBench: Can Large Language Model Detect and Piece Together Implicit Evidence? (https://arxiv.org/abs/2406.12641)
- **What's New**: 이번 논문에서는 LLMs (Large Language Models)의 문맥 기반 추론 능력을 강화하고 평가하기 위한 새로운 벤치마크인 DetectBench를 제안합니다. DetectBench는 긴 문맥 내에서 암시된 증거를 탐지하고 조합하는 능력을 검증하기 위한 3,928개의 객관식 질문을 포함하고 있습니다.

- **Technical Details**: DetectBench는 각 질문이 평균 994개의 토큰으로 구성된 단락과 연결되며, 평균 4.55개의 암시된 증거를 포함합니다. 문제를 해결하려면 일반적으로 7.62번의 논리적 도약이 필요합니다. 성능 향상을 위해 'Detective Reasoning Prompt'와 'Fine-tuning'을 제안하며, 이는 강력한 LLMs의 증거 탐지 능력을 효과적으로 향상시킵니다. 또한, 약한 LLMs의 성능을 크게 향상시키는 결과를 보여줍니다.

- **Performance Highlights**: 실험 결과, 인간이 기존의 최고급 LLMs보다 증거 탐지와 질문 응답 두 과제에서 모두 우수한 성능을 보였습니다. 하지만 Detective Reasoning Prompt는 강력한 LLMs의 성능을 크게 향상시키고 Fine-tuning은 오픈소스 LLMs의 능력을 향상시키는 것으로 나타났습니다.



### Ask-before-Plan: Proactive Language Agents for Real-World Planning (https://arxiv.org/abs/2406.12639)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 이용한 언어 에이전트가 불명확한 사용자 지침을 이해하고 계획을 세우는 능력을 탐구합니다. 이를 위해 새로운 과제인 Proactive Agent Planning을 소개하며, 사용자-에이전트 대화와 에이전트-환경 상호작용을 기반으로 명확화 요구를 예측하고, 외부 도구를 호출하여 유효한 정보를 수집한 후 사용자 요구를 충족시키기 위한 계획을 수립하는 과정을 포함합니다.

- **Technical Details**: Proactive Agent Planning 문제를 연구하기 위해 새로운 벤치마크 데이터셋인 Ask-before-Plan을 설정했습니다. 이를 위한 다중 에이전트 프레임워크, Clarification-Execution-Planning (CEP)을 제안했습니다. CEP는 명확화, 실행, 계획을 담당하는 세 가지 에이전트로 구성되어 있으며, 명확화 에이전트는 사용자 지침의 불확실성을 이해하고 명확화 질문을 하고, 실행 에이전트는 환경과 상호작용하여 필요한 정보를 수집하며, 계획 에이전트는 이 정보를 바탕으로 최종 계획을 수립합니다.

- **Performance Highlights**: 제안된 CEP 프레임워크의 효과를 검증하기 위해 Ask-before-Plan 데이터셋에서 광범위한 평가와 종합적인 분석을 수행한 결과, CEP가 예측 명확화 요구, 정보 수집, 계획 수립의 문제를 효과적으로 해결함을 확인했습니다.



### SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation (https://arxiv.org/abs/2406.12629)
Comments:
          Code are available at \url{this https URL}

- **What's New**: 이번 연구에서는 OOD(Out-of-Distribution) 검출을 위한 새로운 방법인 SeTAR를 제안합니다. SeTAR는 트레이닝이 필요 없이 선택적 저랭크 근사화(Selective Low-Rank Approximation)를 적용하여 비전-언어 모델과 비전 전용 모델에서 OOD 검출 성능을 향상시킵니다. 또한, SeTAR를 기반으로 한 SeTAR+FT라는 파인튜닝 확장 모델도 제안하여 OOD 검출 성능을 최적화 했습니다.

- **Technical Details**: SeTAR는 간단한 탐욕적 탐색 알고리즘(Greedy Search Algorithm)을 사용하여 모델의 가중치 행렬을 포스트 호크(사후)로 수정합니다. 클립(CLIP) 모델의 가중치 행렬을 저랭크 근사화하여 조정하고, 필요에 따라 파인튜닝을 통해 성능을 극대화합니다. CLIP는 이미지 인코더와 텍스트 인코더를 포함하며, 각각 비전 트랜스포머(ViT) 구조를 가집니다.

- **Performance Highlights**: ImageNet1K와 Pascal-VOC 벤치마크를 통해 SeTAR의 우수한 성능을 입증했습니다. SeTAR는 zero-shot 및 fine-tuning 방법들과 비교했을 때 최대 18.95%와 36.80%의 거짓양성율(false positive rate)을 감소시켰습니다. SeTAR+FT는 기존의 파인튜닝 기법을 넘어서는 새로운 상태-최고의 성능(state-of-the-art performance)을 기록했습니다. 예를 들어, ImageNet1K 벤치마크에서 SeTAR는 AUROC 91.32%를 달성했으며, 파인튜닝 기반 검출법과 결합했을 때는 이 값이 92.31%로 상승했습니다.



### Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges (https://arxiv.org/abs/2406.12624)
- **What's New**: 최근 등장하는 LLM-as-a-judge 패러다임을 통해 대규모 언어모델(LLMs)의 평가 문제를 해결하고자 하는 연구가 활발히 진행됩니다. 이번 연구는 다양성 모델의 성능을 종합적으로 평가하고, LLM이 판사 역할을 수행할 경우의 장점과 약점을 조사했습니다.

- **Technical Details**: 본 연구에서는 TriviaQA를 벤치마크로 사용하여 LLM의 객관적 지식 추론 성능을 평가했습니다. 9개의 판사 모델과 9개의 시험 응시자 모델, 각각 베이스와 인스트럭션 튜닝된 모델을 포함하여 다양한 크기와 구조의 모델을 비교했습니다. Cohen’s kappa를 통해 단순 퍼센트 일치와는 다른 정렬 지표를 사용해 판사 모델의 성능을 평가했습니다.

- **Performance Highlights**: GPT-4 Turbo와 Llama-3 70B는 인간 평가자와 높은 일치를 보였지만, 가장 높은 일치를 보인 모델은 JudgeLM-7B와 lexical judge Contains라는 모델이었습니다. 오류 분석을 통해 판사 모델들의 성능에 대한 추가 인사이트를 제공하며, LLM을 판사로 사용할 때의 함정과 주의사항을 제시했습니다.



### Growing Trees on Sounds: Assessing Strategies for End-to-End Dependency Parsing of Speech (https://arxiv.org/abs/2406.12621)
Comments:
          Accepted at ACL 2024

- **What's New**: 이번 논문은 기존의 파이프라인 접근방식(Automatic Speech Recognition (ASR)을 먼저 사용한 후 구문 분석을 수행) 대신 음성 신호에서 직접 종속구문 분석(dependency parsing)을 수행하는 방법을 제안합니다. 이렇게 함으로써 ASR의 한계를 극복하고, 음성 신호의 운율 정보(prosodic information)를 활용할 수 있습니다. 이 연구는 프랑스어로 된 대형 트리뱅크(treebank)를 사용해 다양한 구문 분석 방식(graph-based parsing 및 sequence labeling based parsing)의 성능을 평가했습니다.

- **Technical Details**: 음성 구문 분석을 위해 본 논문에서는 두 가지 모듈로 구성된 파서를 사용했습니다. (i) 음성 모듈(acoustic module)에서는 음성을 텍스트로 변환하고, 신호를 단어로 분할하며, (ii) 구문 분석 모듈(parsing module)에서는 이 분할된 단어를 사용해 오디오 단어 임베딩(audio word embeddings)을 생성하고, 종속 그래프(dependency tree)를 예측합니다. 또, 미리 학습된 wav2vec2 모델을 사용하여 프랑스어 음성의 표현을 추출했으며, 이 표현은 전통적인 그래프 기반 바이아핀(biaffine) 파서에 입력으로 사용됐습니다.

- **Performance Highlights**: 이 연구는 세 가지 주요 실험 설정기를 비교합니다: (i) 생(raw) 오디오만 접근하는 모델(audio setting), (ii) 오디오와 자동 생성된 단어 타임스탬프를 사용하는 모델(oracle setting), (iii) ASR로 예측된 텍스트만을 사용하는 파이프라인(pipeline setting). 전체 트리뱅크에서 그래프 기반 접근법이 우수한 성능을 보였으며, 음성에서 직접 구문 분석하는 방법이 파이프라인 접근법을 능가하였습니다. 음성 신호 정보를 활용한 접근이 더 적은 매개변수를 사용하고도 우수한 성능을 보였습니다.



### What makes two models think alike? (https://arxiv.org/abs/2406.12620)
Comments:
          7 pages, 6 figures

- **What's New**: 이 연구는 기존 방식과 달리 새로운 접근법인 Metric-Learning Encoding Models (MLEMs)을 사용하여 BERT, GPT-2, Mamba 모델들 간의 언어 정보 표현 및 처리 방식을 비교합니다. MLEMs은 각 모델의 특정 언어적 특징을 식별하여 투명하게 비교할 수 있게 해줍니다. 이 접근법은 텍스트 외에도 음성 및 비전과 같은 다른 도메인에도 확장될 수 있습니다.

- **Technical Details**: 이 연구는 Marr의 계층 구조에 따라 언어 모델을 분석합니다. Marr의 계층 구조는 (1) 계산적 수준, (2) 알고리즘적 수준, (3) 구현적 수준으로 나뉘며, 각 수준에서 모델을 평가합니다. 기존 유사도 측정을 넘어, MLEMs는 각 모델의 언어적 특징이 유사성이나 차이를 초래하는지 설명하는 데 초점을 맞춥니다. 우리는 각 모델의 여러 레이어에서 텍스트 표현 간의 거리를 측정하여 이 비교를 수행했습니다.

- **Performance Highlights**: 주요 관찰 사항으로, Transformer 기반 모델에서는 품사가 주요 언어적 특징이지만, Mamba 모델에서는 첫 번째와 마지막 레이어에서만 해당 특징이 두드러졌습니다. 또한, Mamba 모델의 중간 레이어에서는 단어 위치의 중요성이 증가했습니다. 모든 모델에서 문법적 수의 중요성은 초기 레이어에서 후반 레이어로 갈수록 감소하는 경향을 보였습니다. 유사도 측정을 통해 GPT-2와 BERT 모델이 언어 정보 표현 방식이 더 유사한 것으로 나타났습니다.



### From Insights to Actions: The Impact of Interpretability and Analysis Research on NLP (https://arxiv.org/abs/2406.12618)
- **What's New**: 이번 연구는 Natural Language Processing (NLP) 분야 내 해석 가능성 및 분석 연구(Interpretability and Analysis, IA)가 주는 영향을 측정하고자 했습니다. 이를 위해 2018년부터 2023년까지 ACL 및 EMNLP 두 주요 NLP 학회에서 발표된 185,384편의 논문에 대한 인용 그래프와 138명의 NLP 커뮤니티 구성원으로부터 얻은 설문 조사를 바탕으로 혼합 방법 분석을 수행했습니다.

- **Technical Details**: 이번 연구에서는 NLP 연구의 IA 연구가 어떻게 영향을 미치는지 파악하기 위해 두 가지 방법을 사용했습니다. 첫째로, 인용 그래프(bibliometric analysis)를 통해 논문 간의 인용 관계를 분석했습니다. 둘째로, NLP 연구자들과 실제 사용자들을 대상으로 설문 조사를 실시하여 IA 연구의 중요성과 영향을 평가했습니다. 이를 통해, IA 연구가 장기적으로 NLP 분야에 어떤 영향을 끼쳤는지를 종합적으로 분석했습니다.

- **Performance Highlights**: 정량적 분석 결과, IA 연구는 IA 연구 이외의 논문에서도 많이 인용되며 NLP 연구에서 중심적인 역할을 하고 있음이 드러났습니다. 설문 조사와 556편의 논문에 대한 정성적 분석에서는 많은 NLP 연구자들이 IA 연구의 발견을 토대로 연구를 수행하며, IA 연구가 NLP의 여러 하위 분야와 연구 진행에 있어 중요한 역할을 한다고 인식하고 있다는 것을 확인했습니다. 그러나, IA 연구의 발견에 기반하여 제안된 새로운 방법들이 존재함에도 불구하고 IA 연구가 완전히 구동하는 영향력 있는 비-IA 연구는 드뭅니다. IA 연구에서 더 큰 영향을 미칠 수 있는 요소로는 통합, 실행 가능한 권장사항, 인간 중심의 다학제적 연구, 표준화되고 견고한 방법론 등이 부족하다는 점을 지적했습니다.



### EUvsDisinfo: a Dataset for Multilingual Detection of Pro-Kremlin Disinformation in News Articles (https://arxiv.org/abs/2406.12614)
Comments:
          4 pages, 3 figures, 2 tables

- **What's New**: 이번 연구는 EUvsDisinfo 프로젝트에서 작성된 신뢰할 수 있는 기사와 친-크렘린(disinformation) 기사를 포함하는 다국어 데이터를 소개합니다. 이 데이터셋은 현재까지 가장 큰 규모로, 다양한 언어와 주제를 포괄하며 8년 반 동안 수집되었습니다.

- **Technical Details**: 이 데이터셋은 EUvsDisinfo의 디벙크(디스인포메이션을 반박하는) 기사에서 직접 가져온 것입니다. 총 18,249개의 기사로 이루어져 있으며, 42개의 언어를 포함하고 508개의 주제를 다룹니다. 데이터 수집 과정에서는 Diffbot API와 Wayback Machine을 사용하여 더 이상 접근할 수 없는 기사들을 보완하였고, Polyglot을 사용해 기사 언어를 식별했습니다.

- **Performance Highlights**: 이 데이터셋은 기존의 데이터셋보다 크기가 크고 주제와 언어의 다양성이 높습니다. 친-크렘린(disinformation) 주제의 유행을 다양한 언어와 시간대에 걸쳐 분석할 수 있으며, 이를 통해 다국어 환경에서 disinformation과 신뢰할 수 있는 컨텐츠를 효과적으로 구분하는 모델을 훈련시킬 수 있도록 했습니다.



### Bridging Local Details and Global Context in Text-Attributed Graphs (https://arxiv.org/abs/2406.12608)
- **What's New**: 이번 연구에서 제안된 'GraphBridge'는 텍스트-속성 그래프(Text-Attributed Graphs, TAGs)에서 로컬과 글로벌 관점을 연결하는 다중 세밀도 통합 프레임워크입니다. 이는 노드 간의 맥락적 텍스트 정보를 활용하여 로컬 수준의 텍스트 정보를 글로벌 수준의 그래프 구조와 통합할 수 있도록 설계되었습니다.

- **Technical Details**: GraphBridge는 노드의 텍스트 정보와 그래프 구조적 특성을 통합하는 방식으로 동작합니다. 텍스트 인코딩(encoding)과 구조적 집계(aggregating) 두 모듈로 구성되며, 기존 연구에서는 두 모듈을 개별적으로 처리하여 정보의 연관성을 놓치는 경우가 많았습니다. GraphBridge는 특히 이 문제를 개선하기 위해 그래프 인식 토큰 감소 모듈(graph-aware token reduction module)을 도입하여 효율성과 확장성을 크게 높였습니다. 이 모듈은 그래프 구조를 고려하여 중요한 토큰만 선택적으로 유지하는 학습 가능한 메커니즘을 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과, GraphBridge는 다양한 도메인에서 기존 방법들에 비해 뛰어난 성능을 보였습니다. 제안된 방법론은 효율성과 확장성을 해결하며, 로컬과 글로벌 정보를 효과적으로 통합하여 고성능을 달성하는 것으로 나타났습니다.



### Low-Redundant Optimization for Large Language Model Alignmen (https://arxiv.org/abs/2406.12606)
Comments:
          14 pages, working in progress

- **What's New**: 새로운 연구에서는 인간의 선호도에 맞추기 어려운 대형 언어 모델(LLMs)의 문제를 해결하기 위해 상위 10%의 가장 최근에 업데이트된 매개변수만 선택하여 학습을 진행했습니다. 그 결과, 수렴 과정과 최종 성능에서 개선이 나타났습니다. 이는 LLMs 내의 일부 뉴런들이 정렬 훈련에 불필요할 수 있다는 것을 시사합니다. 이를 해결하기 위해 'ALLO'라는 저감중복 정렬 방법을 제안했습니다.

- **Technical Details**: ALLO(Alignment with Low-Redundant Optimization)는 인간 선호도 데이터와 관련된 뉴런을 식별하고, 보상 모델을 사용해 정렬 관련 주요 토큰을 식별한 후, 손실을 계산하여 뉴런을 최적화합니다. 이 과정에서 정렬 프로세스를 '망각 단계'와 '학습 단계'로 나누어, 각각의 단계에서 다른 비율의 뉴런을 업데이트 합니다. 예를 들어, 망각 단계에서는 적은 수의 뉴런을 업데이트하고, 학습 단계에서는 더 많은 뉴런을 업데이트합니다.

- **Performance Highlights**: 실험 결과, ALLO는 질문 답변, 수학적 추론, 지시 따르기와 같은 세 가지 시나리오에서 총 10개의 데이터셋을 사용해 경쟁력 있는 인간 정렬 방법(SFT, DPO, PPO)보다 대부분 우수한 성능을 보여주었습니다. 예를 들어, DPO 대비 최대 9.7% 상대적 성능 향상을 달성했습니다.



### Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling (https://arxiv.org/abs/2406.12585)
- **What's New**: 이 논문은 각 토큰 생성을 분류(Classification)로 취급하여 LLM(대규모 언어 모델)의 합성을 수행하는 새로운 방법을 제시합니다. 기존 방법은 주로 LLM의 전체 텍스트 출력을 합성하거나 랭커를 사용하여 최적의 출력을 선택하는데 한정되었으나, 본 연구는 각 생성 단계에서의 확률 정보를 완전히 활용하여 초기 오류 토큰 생성을 방지합니다.

- **Technical Details**: LLM의 각 토큰 생성을 분류 작업으로 간주하고 여러 모델의 확률 벡터를 합성(ensemble)하여 더 높은 정확도를 달성하는 방식을 사용합니다. 이를 통해 개별 후보 출력에 한정되지 않고, 각 생성 단계에서 확률 정보를 완전히 활용할 수 있습니다. 또한, 주로 사용하는 단어들은 주어진 문장에서 오류를 일으키기 어려운 '간단한' 단어들임을 고려하여, 핵심 토큰만을 합성하는 방법을 적용하여 효율성과 성능을 모두 개선하였습니다.

- **Performance Highlights**: 제안된 방법은 시험, 수학, 추론 및 지식 기반 QA를 포함한 여러 벤치마크에서 SOTA(최첨단) 모델을 합성하여 기존의 커뮤니티 성능 한계를 뛰어넘었습니다. 또한, 핵심 토큰을 합성함으로써 대기 시간(latency)을 줄이면서 더 나은 성능을 보였습니다.



### Mathador-LM: A Dynamic Benchmark for Mathematical Reasoning on Large Language Models (https://arxiv.org/abs/2406.12572)
- **What's New**: Mathador-LM라는 새로운 벤치마크가 도입되었습니다. 이 벤치마크는 대형 언어 모델(LLMs)을 평가하기 위해 개발되었으며, 규칙 해석, 계획 및 문제 해결을 통합한 수학적 추론 능력을 평가합니다. Mathador-LM은 주어진 수와 기본 산술 연산을 사용해 목표 숫자에 도달하는 Mathador 게임에서 영감을 받았습니다.

- **Technical Details**: Mathador-LM 벤치마크는 few-shot 평가를 위한 프레임워크를 따르며, LLaMA3, Qwen2, Claude, GPT-3.5/4 등 다양한 오픈소스 및 클로즈드-소스 모델들을 평가했습니다. 이 벤치마크는 여러 난이도 수준에서 벤치마크 인스턴스를 동적으로 생성하여 테스트셋 유출 문제를 해결하고, 모델 성능의 일관성을 유지합니다. 예제 인스턴스와 기본 및 최적 해법을 포함한 게임 정의는 제시된 수학적 표현과 스코어링 시스템을 통해 구현되었습니다.

- **Performance Highlights**: 최신 모델들은 Mathador-LM에서 평균 15% 이하의 점수를 기록했으며, 이는 최신 5학년 학생들의 평균 43.7%에 비해 현저히 낮습니다. 특히, GPT-4와 Claude-Haiku 모델은 7% 이하의 점수를 얻었습니다. 또한, 모델 크기와 성능 사이에 명확한 상관관계가 나타났으며, 3B 파라미터 이하의 모델은 거의 정확도를 기록하지 못했고, 70-72B 모델이 최고 점수 10-15%를 기록했습니다.



### Applying Ensemble Methods to Model-Agnostic Machine-Generated Text Detection (https://arxiv.org/abs/2406.12570)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)로부터 기계 생성된 텍스트를 탐지하는 문제를 다룹니다. 특히, 기계 생성 텍스트의 근원 모델이 알려지지 않은 경우에도 탐지가 가능하다는 점을 강조합니다. 이를 위해 DetectGPT 분류기(Mitchell et al., 2023)의 출력을 앙상블(ensembling) 방법으로 적용하였습니다. DetectGPT는 제로 샷(zero-shot) 모델로, 해당 텍스트가 생성된 언어 모델과 탐지에 사용된 언어 모델이 동일할 때 매우 높은 정확성을 보입니다.

- **Technical Details**: 논문에서는 DetectGPT 서브 모델 출력의 간단한 요약 통계를 통해 AUROC(AUC-ROC) 0.73을 달성했다고 보고합니다. 이 수치는 옵저버 독립성을 유지한 채 0.61에서 개선한 것입니다. 또한, 지도 학습(supervised learning) 방법을 통해 AUROC를 0.94까지 높일 수 있지만, 이는 학습용 데이터셋을 필요로 합니다. 이러한 결과를 통해 모델 독립적이고 높은 정확도의 기계 생성 텍스트 탐지기를 생성할 가능성을 보여줍니다.

- **Performance Highlights**: DetectGPT 서브 모델 출력의 요약 통계를 사용한 경우, 기계 생성 텍스트 탐지의 AUROC가 0.61에서 0.73으로 향상되었습니다. 그리고 지도 학습 방법을 도입하면 AUROC는 0.94까지 증가합니다.



### RichRAG: Crafting Rich Responses for Multi-faceted Queries in Retrieval-Augmented Generation (https://arxiv.org/abs/2406.12566)
- **What's New**: 이 논문에서는 기존 연구들이 간략한 질문과 명확한 답변에 주로 초점을 맞춘 반면, 사용자들이 넓고 개방적인 질문을 하는 경우가 많다는 점을 지적합니다. 이러한 질문은 다양한 하위 의도를 포함하고 있으며, 이에 부응하는 풍부하고 길게 작성된 답변을 필요로 합니다. 이를 위해 새로운 RAG(정보 검색 보강 생성) 프레임워크인 RichRAG을 제안합니다.

- **Technical Details**: RichRAG은 입력 질문의 하위 측면을 식별하는 'Sub-aspect Explorer', 다양한 외부 문서를 후보 풀로 구축하는 'Multi-faceted Retriever', 그리고 최종 생성기에게 가장 가치 있는 문서들(top-k)을 제공하는 'Generative List-wise Ranker'를 포함합니다. 이 Ranker는 문서의 기본 범위를 보장하기 위해 감독된 미세 조정(SFT) 단계와 강화 학습 단계를 조합하여 학습됩니다.

- **Performance Highlights**: 두 개의 공개 데이터셋을 사용한 실험 결과, RichRAG은 기존 방법보다 포괄적이고 만족스러운 답변을 효과적이고 효율적으로 제공하는 데 성공했습니다. RichRAG은 사용자의 다양한 하위 의도를 포괄적으로 다루는 외부 지식을 제공하여, 궁극적으로 풍부한 응답을 생성합니다.



### Low-Resource Machine Translation through the Lens of Personalized Federated Learning (https://arxiv.org/abs/2406.12564)
Comments:
          18 pages, 7 figures

- **What's New**: 새로운 접근법인 'MeritFed' 개인화 연합 학습(Personalized Federated Learning)을 자연어 처리(Natural Language Processing) 작업에 적용했습니다. 이 방법은 다양한 데이터로 구성된 환경에서 사용 가능합니다.

- **Technical Details**: MeritFed 알고리즘을 사용해 저자원 기계 번역( Low-Resource Machine Translation) 작업을 평가했습니다. 사용된 데이터셋은 대규모 다국어 기계 번역 공유 과제(Large-Scale Multilingual Machine Translation Shared Task)의 소규모 트랙#2(Small Track #2)와 핀우그릭(Finno-Ugric) 언어의 다국어 벤치마크의 사미(Sami) 언어 부분입니다. 이 접근법은 몇 줄의 코드로 쉽게 적용 가능하며, 실험을 재현할 수 있는 스크립트도 제공됩니다.

- **Performance Highlights**: MeritFed는 언어별 데이터의 영향을 추적할 수 있어 해석 가능성도 우수합니다. 분석에 따르면, 타겟 데이터 셋의 크기가 보조 언어의 가중치 분포에 영향을 미치며, 관련 없는 언어들은 훈련에 영향을 주지 않습니다. 또한, 보조 옵티마이저(auxiliary optimizer) 매개변수도 최소한의 영향을 미칩니다.



### MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts (https://arxiv.org/abs/2406.12549)
- **What's New**: 최근의 대형 언어 모델(LLM)은 다국어 고품질 텍스트를 생성할 수 있으며 인간이 쓴 텍스트와 거의 구별할 수 없게 되었습니다. 하지만, 이러한 머신 생성 텍스트(MGT)의 탐지 연구는 주로 영어와 긴 텍스트 (뉴스 기사, 학술 논문, 학생 에세이 등)에 집중되어 있습니다. 소셜 미디어 텍스트는 일반적으로 훨씬 더 짧고 비격식적인 언어, 문법 오류 또는 독특한 언어적 요소(예: 이모티콘, 해시태그)를 특징으로 합니다. 이를 해결하기 위해 우리는 첫 번째 다국어(22개 언어) 및 다중 플랫폼(5개 소셜 미디어 플랫폼) 데이터셋인 MultiSocial을 제안합니다. 이 데이터셋은 약 58,000개의 사람 작성 텍스트와 7개의 다국어 LLM이 생성한 비슷한 양의 텍스트를 포함한 472,097개의 텍스트를 포함합니다.

- **Technical Details**: 우리의 주요 기여는 다음과 같습니다: 1) MultiSocial이라 불리는 사람 작성 및 머신 생성 소셜 미디어 텍스트의 다국어, 다중 플랫폼, 다중 생성기 벤치마크 데이터셋. 2) 다국어 및 교차 언어 능력에 초점을 맞춘 소셜 미디어 텍스트에서의 최신 머신 생성 텍스트 탐지(MGTD) 방법의 첫 번째 다국어 평가. 3) 여러 언어로 텍스트 종류와 출처에 따른 MGTD 성능 차이를 평가하는 첫 번째 다중 플랫폼 및 교차 플랫폼 평가. 우리는 Telegram이 최고 교차 언어 성능을 제공한다는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, 최첨단 탐지기가 소셜 미디어 텍스트에 잘 맞춰져 훈련될 수 있으며, 훈련을 위해 플랫폼을 선택하는 것이 중요하다는 것을 발견했습니다. 최상의 탐지기는 모든 테스트된 언어에서 유사한 성능을 보였으나, 영어와 비영어 간에 일부 차이가 있었습니다.



### P-Tailor: Customizing Personality Traits for Language Models via Mixture of Specialized LoRA Experts (https://arxiv.org/abs/2406.12548)
- **What's New**: 개인화된 대형 언어 모델(LLMs)이 많은 응용 프로그램에서 큰 주목을 받고 있지만, 대부분은 프로필 기반의 성격 설정에 초점을 맞추고 있습니다. 반면, 심리학 이론에 기반한 성격 특성은 잘 모델링되지 않아 심리 상담 에이전트와 같은 전문 분야에서의 잠재적 응용이 제한적입니다. 이를 해결하기 위해, 우리는 Big Five Personality Traits를 모델링하는 MoE 기반 개인화된 LLMs인 P-tailor를 제안합니다.

- **Technical Details**: P-tailor는 다양한 성격 특성을 나타내기 위해 LoRA(Low-Rank Adaptation) 전문가(Experts)를 학습시켜, 개방성(openness), 성실성(conscientiousness), 외향성(extraversion), 적응성(agreeableness), 신경성(neuroticism) 등의 특성을 통합합니다. 또한, Personality Specialization Loss를 도입하여 전문가가 특정 성격 특성 모델링에 집중할 수 있도록 하여 모델의 파라미터 활용 효율성을 높입니다. 이와 함께, 우리는 Big Five 성격 이론을 바탕으로 다양한 주제에 걸친 여러 턴의 대화를 포함한 고품질의 Personality Crafting Dataset(PCD)을 제작했습니다.

- **Performance Highlights**: 광범위한 실험을 통해, P-Tailor가 기존의 모델과 구조 변형들보다 성격 시뮬레이션에서 더 뛰어난 성능을 보인다는 사실을 확인했습니다. 특히, Personality Specialization Loss를 통해 전문가가 특정 성격 특성에 더 전문화될 수 있게 되어 성격 시뮬레이션의 정확성을 향상시켰으며, PCD 데이터셋은 미래의 연구에 귀중한 자원을 제공합니다.



### Liar, Liar, Logical Mire: A Benchmark for Suppositional Reasoning in Large Language Models (https://arxiv.org/abs/2406.12546)
Comments:
          22 pages, 19 figures

- **What's New**: 이번 연구에서는 기사와 악당 퍼즐(knights and knaves puzzles)의 원리를 바탕으로 한 새로운 평가 기준 'TruthQuest'를 소개합니다. 이 벤치마크는 대형 언어 모델(LLMs)이 가정적 추론(suppositional reasoning)을 평가하는데 사용되며, 문제의 난이도는 등장 인물의 수와 논리적 진술의 유형에 따라 다릅니다.

- **Technical Details**: TruthQuest는 2,400개의 다양한 난이도의 문제를 포함하고 있습니다. 기사는 항상 진실을 말하고 악당은 항상 거짓을 말하는 기본 규칙에 따라 퍼즐이 구성됩니다. 이를 통해 Llama 2, Llama 3, Mixtral-8x7B와 같은 대형 언어 모델들의 추론 능력을 평가합니다. 문제는 진술의 종류와 등장 인물의 수에 따라 세 가지 세트(S, I, E)로 나누어집니다. 각 세트마다 다른 데이터 서브셋이 생성되며, 각각 200개의 문제가 포함되어 총 2,400개의 고유한 인스턴스를 포함합니다.

- **Performance Highlights**: 평가 결과, 모든 모델이 기사와 악당 문제를 해결하는 데 상당한 어려움을 겪는 것으로 나타났습니다. 특히, 문제의 복잡성이 증가할수록 성능이 급격히 저하되었습니다. 저성능 모델은 다양한 유형의 추론 오류를 보여주었으며, 상대적으로 높은 성능의 모델은 거짓 진술의 논리적 함의를 정확히 해석하는데 어려움을 겪었습니다.



### Unified Active Retrieval for Retrieval Augmented Generation (https://arxiv.org/abs/2406.12534)
- **What's New**: 최근 다룬 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 비효율성을 해결하기 위한 새로운 프레임워크, Unified Active Retrieval (UAR)을 제안했습니다. 이 접근법은 다양한 사용자의 명령어에 적용되며, 저비용으로 다면적 판단 능력을 제공합니다. 이를 통해 Retrieval 시점을 더 효율적으로 결정할 수 있습니다.

- **Technical Details**: UAR은 네 가지 독립적인 기준을 제안하여, 이를 plug-and-play 분류 작업으로 구현했습니다. 이 기준은 Intent-aware, Knowledge-aware, Time-Sensitive-aware, Self-aware로 나뉩니다. 각각의 기준에 대해, LLM의 마지막 레이어의 히든 스테이트를 사용해 가벼운 이진 분류기를 학습시켜, 입력이 Retrieval이 필요한지 판단합니다. UAR는 LLM의 파라미터를 변경하지 않으며, 동일한 입력 인코딩을 공유하기 때문에 추가적인 추론 비용이 거의 들지 않습니다.

- **Performance Highlights**: 논문에서 수행한 실험은 네 가지 대표적인 사용자 명령어에 대해, UAR이 기존 작업들을 능가하는 성능을 보였습니다. 특히, Retrieval 타이밍 판단 정확도와 다운스트림 작업 성능 면에서 큰 성과를 나타냈으며, 이는 UAR의 효율성을 입증하였습니다.



### FuseGen: PLM Fusion for Data-generation based Zero-shot Learning (https://arxiv.org/abs/2406.12527)
Comments:
          17 pages, 8 figures, 12 tabels

- **What's New**: FuseGen은 데이터 생성 기반 제로샷 학습(zero-shot learning) 프레임워크로, 이전의 단일 사전 학습 언어 모델(PLM)에 의존하지 않고, 여러 PLM과 훈련된 소형 작업별 모델(STM)을 활용하여 고품질 합성 데이터셋을 생성하도록 합니다. 이 접근법은 분포 편향(distribution bias)을 줄이고, 데이터 품질을 향상시킵니다.

- **Technical Details**: FuseGen은 다중 PLM을 사용해 합성 데이터셋을 생성하고, 이 데이터셋에서 하위 집합을 선택하는 새로운 기준을 도입하였습니다. 이러한 하위 집합은 각 PLM에게 인-컨텍스트 피드백을 제공함으로써, 반복적인 데이터 생성을 통해 데이터셋의 품질을 향상시킵니다. 훈련된 STM도 샘플의 재가중(sample re-weighting)에 사용되어 데이터 품질을 더욱 개선합니다.

- **Performance Highlights**: 다양한 작업에 대한 실험 결과, FuseGen은 기존 방법들을 무색하게 할 정도로 상당히 높은 성과를 보였습니다. 특히, PLM에 구애받지 않고 STM 성능을 크게 향상시키는 데 매우 효과적임이 입증되었습니다. 코드는 해당 URL에서 제공됩니다.



### Code-Optimise: Self-Generated Preference Data for Correctness and Efficiency (https://arxiv.org/abs/2406.12502)
Comments:
          Under review at ARR (for EMNLP 2024)

- **What's New**: Code Language Models (CLMs)은 전형적으로 실행 시간은 고려하지 않고 정확한 코드를 생성하는데 집중해왔습니다. 반면, 실행 최적화를 탐구한 이전 연구들에서는 기능적 정확성이 떨어지는 문제가 발생했습니다. 이를 해결하기 위해, Code-Optimise라는 프레임워크를 제안합니다. 이 프레임워크는 자가 생성된 선호 데이터(self-generated preference data)를 이용해 정확성(성공/실패)과 실행 시간(빠름/느림)을 학습 신호로 통합합니다. 이 프레임워크는 경량화 및 견고성을 갖추고 있으며, 대규모 모델에 의존하지 않고 과적합을 줄이는 솔루션을 동적으로 선택합니다.

- **Technical Details**: Code-Optimise는 세 가지 주요 단계로 구성됩니다. 1) 샘플링(Sampling): 각 문제 설명에 대해 N개의 솔루션을 생성합니다. 2) 주석 달기(Annotation): 각 솔루션의 정확성 및 실행 시간을 자동으로 레이블링합니다. 3) 최적화(Optimisation): 자가 생성된 선호 데이터를 이용해 CLM을 미세 조정(fine-tuning)합니다. 이 과정에서 함수적 정확성과 다름을 균형 있게 유지하기 위해 t=0.6의 온도(temperature)를 적용합니다. 각 솔루션은 정통적 정확성(단위 테스트 통과 여부)과 나노초 단위의 실행 시간을 측정합니다.

- **Performance Highlights**: Code-Optimise는 여러 경쟁 모델 대비 pass@k 지표에서 큰 향상을 이루었으며, MBPP 기준 실행 시간을 추가로 6% 줄이고 out-of-domain 데이터에서는 3% 줄였습니다. 또한, 생성된 솔루션의 평균 길이가 MBPP에서는 48%, HumanEval에서는 23% 줄었습니다. 이를 통해 추론비용(inference cost)을 줄이는 데 기여하게 되었습니다. 이 결과는 동적 솔루션 선택(Dynamic Solution Selection, DSS)을 통해 더욱 향상되었습니다.



### LightPAL: Lightweight Passage Retrieval for Open Domain Multi-Document Summarization (https://arxiv.org/abs/2406.12494)
Comments:
          13 pages, 3 figures

- **What's New**: 기존의 Open-Domain Multi-Document Summarization (ODMDS) 방법들이 사용자 쿼리에 대해 요약을 생성하는데 있어 충분하지 않음을 인식하여, LightPAL이라는 새로운 경량 패시지 검색 방법을 제안했습니다. 이 방법은 LLM(Large Language Model)을 사용하여 패시지 간의 관계를 나타내는 그래프를 구성하고, 추론 시 반복적인 추론 없이 확률적 행보(Random Walk)를 사용합니다.

- **Technical Details**: LightPAL은 인덱싱 단계에서 LLM을 사용하여 패시지가 다른 패시지의 컨텍스트로 사용될 수 있는지를 평가하며, 생성 확률이 높은 패시지들 간에 링크를 생성합니다. 검색 단계에서는 임의의 모델을 사용하여 초기 관련 패시지를 검색하고, Personalized PageRank (PPR)로 그래프에서 확률적 행보를 계산하여 다른 패시지에 도달할 확률이 높은 패시지를 추가적인 컨텍스트로 검색합니다. 이 방법을 통해 런타임 중 LLM 추론 없이 다양한 관련 정보를 저지연으로 검색할 수 있습니다.

- **Performance Highlights**: LightPAL은 ODSum 및 Querysum 벤치마크에서 기존의 기법들보다 뛰어난 성능을 보였으며, 검색-후-요약 접근법으로 생성된 요약문의 품질에서도 뛰어난 결과를 나타냈습니다. 또한, PromptRank와 비교해 약 1,000배 더 낮은 지연 시간을 유지하면서도 우수한 요약 성능을 보여주었습니다.



### The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions (https://arxiv.org/abs/2406.12480)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)에서 생성된 합성 데이터를 활용하여 온라인 정치 토론에서의 입장 감지(st'espace detection) 성능을 개선하는 방법을 제시합니다. 연구진은 Mistral-7B 모델을 사용해 특정 토론 질문에 대한 합성 데이터를 생성하고, 이를 통해 입장 감지 에이전트를 미세 조정(fine-tuning)하여 성능을 크게 향상시켰습니다.

- **Technical Details**: 연구진은 두 가지 주요 접근법을 사용했습니다. 첫째, Mistral-7B 모델을 사용하여 특정 토론 질문에 대한 합성 데이터를 생성하고, 이를 통해 모델을 미세 조정했습니다. 둘째, 합성 데이터를 사용하여 최정보(sample)들을 선택하고, 이와 합성 데이터를 결합하여 미세 조정하는 방법을 적용했습니다. 이러한 접근법은 라벨링 비용을 줄이며, 완전히 라벨링된 데이터로 훈련된 기준 모델보다 일관되게 성능을 뛰어넘었습니다.

- **Performance Highlights**: 실험 결과, LLM에서 생성된 데이터는 온라인 정치 토론에서 입장 감지 성능을 크게 향상시켰습니다. 특히, 합성 데이터와 가장 유익한 샘플을 결합하여 미세 조정한 경우 라벨링된 데이터만으로 학습한 모델보다 성능이 우수했습니다.



### Exploring Intra and Inter-language Consistency in Embeddings with ICA (https://arxiv.org/abs/2406.12474)
- **What's New**: 본 논문에서는 독립 성분 분석(ICA)을 통해 단어 임베딩의 일관성을 조사했습니다. 특히, 각 언어 내에서의 일관성과 여러 언어 간의 일관성을 검증했습니다. ICA는 단어 임베딩을 해석 가능하고 명확한 의미 축으로 변환하는 데 유용합니다. 그러나 ICA의 일관성을 확인하기 위한 방법론이 부족했습니다. 저자들은 이에 대한 새로운 통계 방법을 적용하여 의미 축의 신뢰성과 보편성을 확립했습니다.

- **Technical Details**: ICA는 다차원 데이터에서 통계적으로 독립적인 성분을 추출하는 방법입니다. FastICA 알고리즘 등을 사용하여 독립 성분을 여러 번 계산한 후, 클러스터링을 통해 일관성을 평가합니다. 이 연구에서는 Icasso를 적용하여 언어별로 ICA의 신뢰성을 확인했습니다. 또한, 여러 언어 간 의미 축의 일관성을 통계적으로 평가하기 위해 Hyvärinen과 Ramkumar의 방법을 사용했습니다. 이 방법은 원래 신경 영상 데이터 분석에서 주로 사용되는 것으로, 여러 주제 간 공통 독립 성분을 찾는 데 활용됩니다.

- **Performance Highlights**: 연구 결과, 동일 언어 내에서 ICA 성분의 일관성을 성공적으로 검증하였으며, 각 클러스터의 신뢰성 지수를 통해 높은 재현성을 확인했습니다. 이를 통해 각 독립 성분이 일관된 의미 축을 나타낸다는 것을 입증했습니다. 또한, 영어와 일본어 간 의미 축의 통계적 일관성을 평가한 결과, 두 언어 간에도 공통된 의미 축이 존재함을 확인했습니다. 이는 다언어 간 해석 가능한 단어 임베딩을 위한 중요한 기초를 제공합니다.



### Fighting Randomness with Randomness: Mitigating Optimisation Instability of Fine-Tuning using Delayed Ensemble and Noisy Interpolation (https://arxiv.org/abs/2406.12471)
- **What's New**: 새로운 방법론인 Delayed Ensemble with Noisy Interpolation(DENI)가 소개되었습니다. 이 방법론은 앙상블(Ensemble), 노이즈 정규화(Noise regularisation), 모델 보간(Model interpolation)을 활용하여 연산 효율성을 유지하면서 성능 불안정을 완화합니다.

- **Technical Details**: DENI는 모델 파라미터를 랜덤 노이즈로 변형하여 훈련이 끝나며 앙상블을 생성합니다. 또한, 훈련 과정에서 여러 번 노이즈를 추가하고 소수의 단계 동안 훈련한 후 모델을 단일 모델로 집계하여 효과적으로 랜덤성을 완화합니다.

- **Performance Highlights**: DENI는 9개의 대표적인 완화 전략과의 비교 결과, 가장 성능이 우수한 완화 전략보다 적은 비용으로 우수한 성능을 보였습니다. 또한, PEFT 방법에서는 데이터 증강과 결합하면 더욱 효과적인 안정성 완화를 제공했습니다. 이 방법은 3개의 모델, 4개의 튜닝 전략 및 7개의 텍스트 분류 데이터셋에서 검증되었습니다.



### Adaptive Token Biaser: Knowledge Editing via Biasing Key Entities (https://arxiv.org/abs/2406.12468)
- **What's New**: 이번 연구에서는 Adaptive Token Biaser (ATBias)라는 새로운 디코딩 기법을 소개합니다. 이 기법은 LLMs의 인-컨텍스트 편집(ICE)을 개선하도록 설계되었으며, 새로운 지식과 파라메트릭 지식과 관련된 주요 엔터티를 일치시켜 이들의 로그잇(logits)을 바이어스합니다.

- **Technical Details**: ATBias는 전체 생성 시퀀스 대신 지식과 관련된 특정 토큰에 집중합니다. 이는 중요 엔터티를 추출하고, 확률과 순위 기반 필터를 통해 관련 토큰을 선별하며, n-gram과 Jaccard 유사도 계산 알고리즘을 이용해 엔터티와 토큰을 일치시키는 방식으로 작동합니다. 이를 통해 새로운 지식과 파라메트릭 지식 엔터티의 로그잇에 바이어스를 도입합니다.

- **Performance Highlights**: 실험 결과, ATBias는 기존의 최첨단 ICE 방법에 비해 성능을 최대 32.3% 향상시키며, 지연 시간도 절반으로 줄일 수 있음을 보여주었습니다. 이는 ATBias가 LLMs에 거의 비용 없이 널리 적용될 수 있음을 시사합니다.



### Abstraction-of-Thought Makes Language Models Better Reasoners (https://arxiv.org/abs/2406.12442)
Comments:
          Work in Process

- **What's New**: 이번 연구는 Language Models (LM)에서 추상적 사고(Abstraction-of-Thought, AoT)를 유도하는 새로운 구조적 추론 형식을 도입했다. AoT는 추론 과정에서 다양한 수준의 추상화를 명시적으로 요구하며, 기존의 단계별 Chain-of-Thought (CoT) 방법과 달리 먼저 추상적 수준에서 고민한 후 구체적인 내용을 통합하도록 유도한다. 이를 위해, 34만8000개의 고품질 AoT 추론 프로세스를 포함하는 'AoT Collection' 데이터셋이 개발되었으며, Big-Bench Hard의 23가지 미해결 과제에서 다양한 언어 모델을 세밀 튜닝하여 테스트했다.

- **Technical Details**: AoT 형식은 LLM이 추론 과정을 조직할 때 추상적 골격 솔루션(abstract skeletal solution)을 사용하도록 유도한다. 이를 위해, FLAN Collection을 기반으로 한 AoT Collection 데이터셋에 348k의 AoT 추론 프로세스가 포함되었으며, 자동화된 확장 가능 파이프라인을 통해 수집됐다. 이 데이터셋은 자연어와 프로그래밍 언어 모두를 포함했으며, 다양한 추론 문제에 대해 코드 사용의 잠재력을 방출하고 다양한 추론 프로세스를 선호할 수 있는 유연성을 제공한다.

- **Performance Highlights**: 실험 결과, AoT 형식으로 세밀 튜닝된 모델은 CoT 형식을 사용한 모델보다 다양한 추론 작업에서 우수한 성능을 보였다. 특히, zero-shot 및 few-shot 성능에서 상당한 개선을 나타냈다. 이 연구는 AoT가 추상적 추론을 유도하고 효과적인 모델 학습에 기여할 수 있는 잠재력을 강조한다.



### PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers (https://arxiv.org/abs/2406.12430)
Comments:
          NAACL 2024

- **What's New**: 이번 연구에서는 LLMs(Large Language Models)을 복잡한 데이터 분석이 필요한 의사결정 문제에 대한 솔루션으로 활용하기 위한 연구를 수행했습니다. 의사결정 질문 Q, 비즈니스 규칙 R, 데이터베이스 D를 기반으로 최적의 결정을 찾는 작업을 Decision QA라고 정의하고, 이를 측정하기 위한 벤치마크로 DQA를 제안했습니다.

- **Technical Details**: Decision QA 작업은 Q(질문), R(비즈니스 규칙), D(데이터베이스)와 같은 입력을 받아 최적의 결정을 도출하는 것입니다. 이를 위해 Europa Universalis IV와 Victoria 3이라는 두 개의 비디오 게임에서 시나리오를 구축하여 벤치마크 DQA를 만들었습니다. 또한, PlanRAG이라는 새로운 RAG 기술을 제안하여 분석 계획을 세우고 데이터 분석을 위한 쿼리를 생성하여 LLM이 의사결정을 할 수 있도록 했습니다.

- **Performance Highlights**: 제안된 방법은 최신의 반복 RAG 기법을 사용한 방법과 비교하여 Locating 시나리오에서 15.8%, Building 시나리오에서 7.4% 더 성능이 우수했습니다.



### PSLM: Parallel Generation of Text and Speech with LLMs for Low-Latency Spoken Dialogue Systems (https://arxiv.org/abs/2406.12428)
Comments:
          8 pages, 4 figures, 4 tables, demo samples: this https URL

- **What's New**: 새로운 연구 결과, 텍스트와 음성을 동시에 처리할 수 있는 멀티모달 언어 모델(multimodal language model)이 음성 질문 응답 작업에서 지연 시간을 개선할 수 있음을 보여주고 있습니다. 연구팀은 텍스트와 음성을 병렬(parallel)로 생성할 수 있는 언어 모델 'Parallel Speech Language Model (PSLM)'을 제안합니다.

- **Technical Details**: 언어 모델을 확장하여 텍스트와 음성 토큰을 동시에 처리할 수 있도록 설계했습니다. HiFi-GAN 네트워크를 이용해 음성을 효율적으로 생성하는 네트워크를 구성했으며, BERT 기반의 토큰화를 통해 텍스트 및 음성과의 정렬을 학습했습니다. 또한, 텍스트와 음성을 병렬 스트림(parallel streams)을 통해 처리하여 전반적인 지연 시간을 줄였습니다.

- **Performance Highlights**: 타사 방법(CoM prompting)에 비해 응답 품질을 유지하면서도 지연 시간을 크게 단축시킬 수 있음을 입증했습니다. 병렬 스트림을 활용하여 음성 토큰을 다중으로 디코딩함으로써 처리 시간을 더욱 단축시킬 수 있었습니다. 이를 통해 향상된 응답 속도와 음성 품질을 달성했습니다.



### Open-Source Web Service with Morphological Dictionary-Supplemented Deep Learning for Morphosyntactic Analysis of Czech (https://arxiv.org/abs/2406.12422)
Comments:
          Accepted to TSD 2024

- **What's New**: 이번 연구는 체코어 형태통사 분석을 위한 오픈소스 웹 서비스를 소개합니다. 본 시스템은 딥러닝 모델과 고정밀 형태소 사전을 결합하여 예측 정확도를 높였습니다. 이 하이브리드 방법은 단순한 딥러닝 모델 및 기존의 형태소 분석기인 MorphoDiTa와 비교해 더 높은 성능을 보였습니다.

- **Technical Details**: 체코어 형태통사 분석을 위한 딥러닝 모델인 UDPipe 2와 형태소 사전 MorfFlex를 결합하였습니다. 이 모델은 대규모 체코어 형태통사 코퍼스인 PDT-C 1.0에서 학습되었습니다. 모델의 예측 시에는 딥러닝을 사용하여 어휘 외 단어(OOV)에 대한 일반화와 더 나은 비모호화를 제공하고, 형태소 사전의 유효 분석을 통해 정확성을 유지합니다.

- **Performance Highlights**: Lemmatization에서 50%, POS tagging에서 58%의 오류율 감소를 달성하였으며, 이는 MorphoDiTa와 UDPipe 2보다 크게 향상된 결과입니다. 이 도구는 형태소 분석, lemmatization, POS tagging, 종속 구문 분석을 포함하여 다양한 기능을 제공합니다.



### MMUTF: Multimodal Multimedia Event Argument Extraction with Unified Template Filling (https://arxiv.org/abs/2406.12420)
- **What's New**: 이 연구는 멀티미디어 이벤트 추출(Multimedia Event Extraction, MEE)에서 이벤트 인자 추출(Event Argument Extraction, EAE)을 개선하기 위해 통합 템플릿 채우기 모델을 도입하였습니다. 이는 텍스트와 시각적 모달리티를 텍스트 프롬프트를 통해 연결하여 이벤트 구체적 의미와 Cross-ontology transfer를 효과적으로 활용할 수 있게 합니다.

- **Technical Details**: 제안된 모델인 MMUTF(Multimodal Multimedia Event Argument Extraction with Unified Template Filling)는 텍스트 엔티티 및 시각적 객체와 같은 후보 구조를 활용하고 이를 쿼리 표현(즉, 인자 역할)으로 통합 잠재 공간에서 연결합니다. 이를 통해 이벤트 템플릿에서 추출한 쿼리를 사용하여 이벤트 인자 역할을 후보와 일치시키는 작업을 수행합니다. 모델은 Transformer 기반으로 설계되어 텍스트와 이미지의 모달리티 별 인코딩을 병렬로 수행하고, 각 인자의 쿼리 표현과 일치 점수를 계산합니다.

- **Performance Highlights**: 제안된 MMUTF 모델은 M2E2 벤치마크에서 기존 최고 성능(State of the Art, SOTA)을 7% F1 점수로 초과하였으며, 멀티미디어 EAE에서도 두 번째로 높은 성능을 보였습니다. 이 모델은 복잡한 멀티모달 데이터에서도 우수한 성능을 입증하였습니다.



### AI-Assisted Human Evaluation of Machine Translation (https://arxiv.org/abs/2406.12419)
- **What's New**: 최근 연구에서는 기계 번역 시스템의 평가 과정에서 인간 평가자의 작업 시간을 줄이고 평가의 질을 높이기 위해 AI 보조 프로토콜인 ESA$^	ext{AI}$를 도입했습니다. 새로운 프로토콜에서는 AI가 사전 오류 주석을 제공하여 인간 평가자가 더 빠르고 정확하게 오류를 식별할 수 있도록 돕습니다.

- **Technical Details**: 이 연구에서는 'Error Span Annotation (ESA)' 방식을 개선하여 ESA$^	ext{AI}$를 제안합니다. 이 방식은 AI 품질 추정 시스템(예: GEMBA)을 사용하여 초기 오류 범위를 사전 채워 평가자에게 제공하고, 평가자는 이를 바탕으로 최종 점수를 매깁니다. 평가 프로세스는 Appraise 툴에서 구현되었으며, 오타 수정 및 오류 범위 및 심각도를 조정하는 작업으로 구성됩니다.

- **Performance Highlights**: ESA$^	ext{AI}$ 프로토콜을 사용함으로써 오류 주석 당 시간이 평균 71초에서 31초로 절반 이하로 줄었습니다. 또한, AI 시스템이 오류를 예측하지 않은 예제들을 사전에 제외함으로써 평가 예산을 최대 24% 절감할 수 있습니다. 이로 인해 평가자 간의 일관성이 높아지고 평가 품질이 향상되었습니다.



### Beyond Under-Alignment: Atomic Preference Enhanced Factuality Tuning for Large Language Models (https://arxiv.org/abs/2406.12416)
- **What's New**: 새로운 논문에서는 대형 언어 모델(LLMs)이 실제 사실과 부합하지 않는 응답(factual error)을 생성하는 문제를 다루고 있습니다. 최근에는 선호 학습(preference learning)을 통해 모델을 개선하려는 시도가 증가하고 있습니다. 하지만, 기존 연구는 주로 동일 도메인(in-domain, ID) 데이터셋에서 모델을 평가하며, 이 논문에서는 다양한 선호 학습 알고리즘으로 조정된 모델이 다른 도메인(out-of-domain, OOD) 데이터셋에서의 사실성을 종합적으로 평가했습니다. 그 결과, 대부분의 경우 OOD 데이터셋에서의 성능이 최소한으로 증가하거나, 오히려 감소한다는 것을 발견했습니다.

- **Technical Details**: 논문에서 제안된 APEFT(Atomic Preference Enhanced Factuality Tuning)는 개별 사실 수준에서 모델의 사실성 인식을 향상시키는 프레임워크입니다. 먼저, 일반적인 선호도 데이터를 단일 문장으로 분해한 후, 지식 감지 프롬프트를 통해 모델이 이 지식을 얼마나 잘 알고 있는지 평가합니다. 그런 다음, 모순된 응답을 선택해 원자 선호도(atomic preferences)로 학습하고, 이를 바탕으로 모델을 재교육합니다.

- **Performance Highlights**: APEFT 프레임워크를 적용한 결과, ID와 OOD 데이터셋 모두에서 모델 성능이 평균 3.45% 향상되었습니다. 이는 기존의 선호 학습 알고리즘보다 훨씬 효과적인 결과를 보여줍니다. 특히, 오버-얼라인먼트(over-alignment)보다 언더-얼라인먼트(under-alignment)가 성능 저하의 주요 원인임을 밝혀냈습니다.



### PDSS: A Privacy-Preserving Framework for Step-by-Step Distillation of Large Language Models (https://arxiv.org/abs/2406.12403)
- **What's New**: 실제 응용 환경에서 대형 언어 모델(LLMs)을 도메인 특화 작업에 활용할 때 도메인 특화 지식의 프라이버시 및 제한된 자원이라는 두 가지 주요 문제가 발생한다는 점에 주목하여 이 논문에서는 프라이버시 보호 프레임워크인 PDSS를 제안합니다. PDSS는 서버-클라이언트 아키텍처를 기반으로 작동하며, 클라이언트가 변형된 프롬프트를 서버의 LLM에 전송하여 근거를 생성하도록 합니다. 생성된 근거는 클라이언트에서 디코딩되어 과제 특화 소형 언어 모델(SLM)을 다중 작업 학습 패러다임 내에서 교육하는 데 사용됩니다.

- **Technical Details**: PDSS 프레임워크는 클라이언트 데이터의 프라이버시를 유지하면서 서버의 LLM을 활용하여 클라이언트의 SLM을 교육하는 문제를 해결합니다. 이를 위해 PDSS는 두 가지 프라이버시 보호 전략을 도입합니다: Exponential Mechanism Strategy와 Encoder-Decoder Strategy. Exponential Mechanism Strategy는 지수 메커니즘을 사용하여 프롬프트를 난독화하고, In-Context Learning을 통해 변형된 근거를 디코딩합니다. Encoder-Decoder Strategy는 원시 프롬프트를 변형된 프롬프트로 인코딩하고 다시 변형된 근거를 원래 형태로 디코딩하는 Encoder-Decoder SLM을 사용합니다.

- **Performance Highlights**: 다양한 텍스트 생성 작업에 대한 실험 결과, PDSS는 프라이버시를 우선시하면서도 과제 특화 SLM의 성능을 향상시키는 데 효과적이라는 것을 입증했습니다. PDSS는 LLM에서 생성된 근거를 SLM에 귀중한 과제 특화 지식으로 제공하여, 프라이버시 보호와 성능 향상을 동시에 달성합니다.



### Flee the Flaw: Annotating the Underlying Logic of Fallacious Arguments Through Templates and Slot-filling (https://arxiv.org/abs/2406.12402)
- **What's New**: 최근에 발표된 연구에서는 컴퓨터 논증 분야에서 논증의 질을 평가하는 데 중점을 두었으나, 논리적 오류를 명확히 설명하는 데는 상대적으로 적은 관심을 기울여 왔다는 점을 지적합니다. 이 연구는 일반적인 비형식 논리 오류를 설명하기 위한 네 가지 설명 가능한 템플릿을 소개하고, LOGIC 데이터 집합에서 추출한 400개의 오류 논증을 대상으로 주석 연구를 수행했습니다. 주석 연구에서 높은 합의 점수(Krippendorf's alpha 0.54)와 합리적인 커버리지(0.83)를 달성했습니다. 마지막으로, 오류 구조를 탐지하기 위한 실험을 진행했으며, 최신 언어 모델들이 오류 템플릿을 탐지하는 데 어려움을 겪고 있음을 발견했습니다(정확도 0.47).

- **Technical Details**: 이 연구에서는 네 가지 설명 가능한 논리 오류 템플릿을 사용하여 논증의 논리 구조를 식별하는 새로운 태스크를 제안했습니다. 이를 위해 주석 스킴을 설계하고 400개의 오류 논증에 대한 주석 연구를 수행했습니다. 이 스킴은 기존의 논증 구조를 논리 오류 구조로 확장하여 논증의 논리적 구조를 설명하게 합니다. 특히, 신뢰성 오류, 허위 인과관계, 거짓 딜레마, 잘못된 일반화를 포함한 네 가지 결함 있는 귀납 오류 유형을 다루기 위해 20개의 새로운 템플릿을 개발했습니다.

- **Performance Highlights**: 이 연구는 고급 언어 모델들이 논리 오류 구조를 식별하는 데 어려움을 겪고 있음을 실험적으로 증명했습니다. 최신 언어 모델의 정확도는 0.47에 불과했으며, 이는 논리 오류 구조 식별이 기계에 상당히 어려운 작업임을 시사합니다. 또한, 주석 연구에서 높은 합의 점수(Krippendorf's alpha 0.54)와 합리적인 커버리지(0.83%)를 달성하여 주석 스킴의 신뢰성을 입증했습니다.



### QueerBench: Quantifying Discrimination in Language Models Toward Queer Identities (https://arxiv.org/abs/2406.12399)
- **What's New**: 최근 자연어 처리(NLP)의 중요성이 증대됨에 따라 편향과 고정관념 확산 문제도 함께 부각되고 있습니다. 본 논문은 LGBTQIA+ 커뮤니티를 대상으로 한 대형 언어 모델(LLMs)에서 발생 가능한 잠재적 해악을 평가하는 QueerBench라는 새로운 평가 프레임워크를 도입하였습니다.

- **Technical Details**: QueerBench는 템플릿 기반 접근법과 Masked Language Modeling(MLM) 과제를 사용하여 언어 모델이 생성한 문장 완성이 LGBTQIA+ 개인에게 미치는 영향을 평가합니다. 이 평가 지표를 통해 LLM이 생성하는 문장의 해악 정도를 측정합니다. 다양한 대형 언어 모델들(BERT, ALBERT, RoBERTa, BERTweet)에서 템플릿 문장과 주어를 조합하여 예측을 생성하였습니다.

- **Performance Highlights**: QueerBench 점수를 통해 LLM이 LGBTQIA+ 커뮤니티에 대해 차별적인 행태를 보일 가능성이 더 높다는 것이 드러났습니다. 예를 들어, 퀴어 관련 용어가 주어로 설정된 문장에서 해악성 비율이 평균적으로 16.9%로 나타나, 비퀴어 용어가 주어인 문장보다 현저히 높았습니다(평균 해악성 9.2%). 이는 LLM들이 LGBTQIA+ 커뮤니티에 대해 더 많은 해를 끼칠 수 있음을 시사합니다.



### Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models (https://arxiv.org/abs/2406.12397)
Comments:
          15 pages

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 훈련에 있어서 고품질 데이터의 부족을 해결하기 위해 제안된 합성 데이터(synthetic data)의 잠재적 결함을 분석합니다. 합성 Q-A 쌍이 모델의 지시사항 준수 능력을 감소시킬 수 있음을 발견하고, 이러한 결함을 완화하기 위한 unlearning 기술 기반의 방법을 제안합니다.

- **Technical Details**: 합성 데이터의 균일한 포맷이 패턴 과적합(pattern overfitting)을 유발하여 출력 분포가 크게 변하고 모델의 지시 사항 준수 능력이 감소할 수 있다는 점에 주목했습니다. 이를 해결하기 위해 하한이 있는 망각 손실(lower-bounded forgetting loss)을 사용하는 새로운 전략을 제안합니다. 이 전략은 기존의 unlearning 접근법보다 우수하며, 합성 데이터의 오해 발생 패턴을 줄이면서 비교적 낮은 훈련 비용으로 벤치마크 성능을 유지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 전략은 합성 데이터의 부정적인 영향을 효과적으로 완화하여 벤치마크 성능과 지시사항 준수 능력을 균형 있게 유지할 수 있음을 보여줍니다. 특히, 패턴 과적합 문제를 해결하면서도 모델의 기본적인 역량은 손상시키지 않는 성과를 입증했습니다.



### EMO-KNOW: A Large Scale Dataset on Emotion and Emotion-caus (https://arxiv.org/abs/2406.12389)
Comments:
          Accepted to Findings of EMNLP 2023

- **What's New**: 최근 감정-원인 분석(Emotion-Cause analysis)은 연구자들의 많은 관심을 받고 있습니다. 대부분의 기존 데이터셋은 크기와 감정 카테고리의 수에서 제한적이며, 문서의 일부만을 추출하여 감정의 원인을 찾는 데 집중합니다. 이러한 단점을 보완하기 위해, 우리는 9.8백만 개의 클린된 트윗을 바탕으로 15년에 걸친 대규모 감정 원인 데이터셋을 소개합니다. 이 데이터셋은 48가지 감정 클래스에 걸쳐 700,000개 이상의 트윗을 포함하며, 인간 평가자를 통해 검증된 감정-원인 쌍을 제공합니다.

- **Technical Details**: 데이터셋의 큐레이션 과정은 데이터 수집, 정리, 레이블링, 검증을 포함하는 포괄적인 파이프라인으로 구성됩니다. 데이터는 Twitter API를 사용하여 15년에 걸쳐 수집되었으며, 감정적 표현과 잠재적 감정 원인을 포함하는 트윗을 추출하기 위해 검색 구문을 반복적으로 수정했습니다. 데이터 클리닝 및 정제 파이프라인에는 동정 표현, 비속어, 의미 없는 패턴을 포함하는 트윗 제거가 포함됩니다. 최종 데이터셋은 감정 트리거 이벤트에 대한 추상적 요약을 제공하는 언어 모델을 훈련하여 감정 원인을 추출합니다.

- **Performance Highlights**: 이 데이터셋의 혁신성은 광범위한 감정 클래스 스펙트럼과 추상적 감정 원인에 있으며, 이는 세분화된 추론을 위한 감정-원인 지식 그래프 개발을 용이하게 합니다. 기존 데이터셋과 달리, 우리의 데이터셋은 외부 레이블링에 의존하지 않고 사용자의 자체 보고 감정 레이블을 사용하여 더 진실된 데이터를 제공합니다. 이 데이터셋은 다양한 감정 반응을 고려한 감정 인지 시스템 설계를 가능하게 합니다.



### IPEval: A Bilingual Intellectual Property Agency Consultation Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2406.12386)
- **What's New**: IPEval은 지적 재산권(IP) 상담 및 대행 업무를 평가하기 위한 최초의 평가 벤치마크입니다. IP의 창작, 적용, 보호, 관리 등의 네 가지 주요 차원을 아우르는 2657개의 객관식 문제로 구성되어 있으며, 특허권, 상표권, 저작권, 영업비밀 등을 포함합니다. IPEval은 모델의 IP 관련 법률 이해, 적용 능력 및 추론 능력을 평가하기 위해 제로샷(Zero-shot), 5-퓨샷(5-few-shot), 체인 오브 생각(Chain of Thought; CoT)과 같은 방법을 사용합니다.

- **Technical Details**: IPEval 벤치마크는 특히 미국 특허청(USPTO)과 중국 지적 재산권국(CNIPA)에서 지난 특허시험의 객관식 문제를 수집하여 구성되었습니다. 이 데이터셋은 IP 관련 법률 이해, 침해 처리 및 보호 방법에 대한 인지 능력 및 절차 적용, 추론 능력을 평가하는 것을 목표로 합니다. IPEval은 IP 상담 업무를 수행하는 데 필요한 법적 지식이 지역적이고 시간에 따라 변하는 특성을 반영하기 위해 지역적 및 시간적 특성을 균형 있게 고려했습니다.

- **Performance Highlights**: IPEval을 사용하여 진행한 실험 결과, GPT 시리즈와 Qwen 시리즈와 같은 모델은 영어 테스트에서 우수한 성과를 보였고, 중국어 중심의 LLM은 중국어 테스트에서 뛰어난 성능을 나타냈습니다. 다만, IP 전문 LLM이 범용 LLM보다 성능이 저조했습니다. 또한, 대부분의 모델이 기준점을 밑돌았으며, Qwen-Max와 Qwen-72B-Chat만이 2A 레벨에 해당하는 점수(63.3 및 62.6)를 겨우 넘겼습니다. 이에 따라 LLM의 IP 능력을 개선할 여지가 많음을 확인했습니다.



### From Instance Training to Instruction Learning: Task Adapters Generation from Instructions (https://arxiv.org/abs/2406.12382)
- **What's New**: 이번 연구는 인간이 지침을 이해하고 따르는 방식에서 영감을 받아, LLMs(대형 언어 모델)의 제한 사항을 해결하고, 태스크 간 일반화 능력을 향상시키기 위해 새로운 학습 패러다임을 소개합니다. 특히, 태스크 어댑터 생성(TagAdapters Generation from Instructions, TAGI)이라는 방법을 도입하여, 새로운 작업에 대해 재학습 없이 지침 기반으로 태스크 전용 모델을 자동으로 구성합니다.

- **Technical Details**: TAGI는 하이퍼네트워크와 지식 증류(knowledge distillation) 기법을 결합하여 태스크 지침을 통해 효과적으로 태스크 전용 어댑터를 생성합니다. 하이퍼네트워크는 기본적으로 다른 신경망의 가중치를 생성하는 네트워크로, 지침에 따라 태스크 전용 모델을 동적으로 구축합니다. 지식 증류를 통해 'Learning with Instruction'과 'Training with Instance' 패러다임을 결합하여 모델의 일치성을 향상시킵니다. 이렇게 생성된 어댑터는 인스턴스 훈련과 구별되게 전달값(logits)과 어댑터 파라미터를 포함한 라벨을 조정하는 방식으로 모델의 이해도와 효율성을 향상시킵니다.

- **Performance Highlights**: TAGI는 Super-Natural Instructions(SNI)와 P3 데이터셋에서 테스트되었으며, 메타 트레이닝된 모델을 SNI에서 2%, P3에서 5% 초과 성능을 나타냈습니다. 또한, 계산 요구 사항을 60% 줄이며, 다른 하이퍼네트워크 기반 모델보다 7% 더 우수한 성능을 보였습니다. 이 방법은 추가적인 파라미터 업데이트나 그라디언트 역전파를 필요로 하지 않으며, 추론 시 반복적인 지침 인코딩 과정을 피합니다.



### QOG:Question and Options Generation based on Language Mod (https://arxiv.org/abs/2406.12381)
Comments:
          8 pages, 3 figures, 4 tables

- **What's New**: 이 논문에서는 새로운 Question-Options Generation (QOG) 모델을 개발하여 주어진 컨텍스트에서 질문-옵션 쌍을 생성하는 방법을 제시합니다. QOG는 교육, 정보 검색, 대형 모델의 미세 조정 등 여러 분야에서 유용하게 활용될 수 있습니다. 실험을 통해, 엔드-투-엔드 QOG 모델이 훈련과 추론 과정에서 계산 효율성과 안정성을 보이며, 다른 방법보다 뛰어난 성능을 보였습니다.

- **Technical Details**: QOG는 기존 Question Generation (QG)과 Question-Answer Generation (QAG)보다 복잡한 작업으로, 올바른 답변과 오답지를 모두 생성해야 합니다. 논문에서는 Fine-tuning된 시퀀스-투-시퀀스(Sequence-to-Sequence) 언어 모델(T5)을 기반으로 세 가지 QOG 방법을 비교하였습니다: (1) 파이프라인 QOG, (2) 멀티태스크 QOG, (3) 엔드-투-엔드 QOG. 이들 모델은 각 세부 과정을 독립적으로 학습하거나, 통합하여 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과, 엔드-투-엔드 QOG 모델이 다른 방법보다 우수한 성능을 나타내었으며, 훈련과 추론 동안 계산 효율성과 안정성을 유지했습니다. 이는 QOG 과제가 포함된 여러 도메인에서 GPT-4를 평가 기준으로 해서도 좋은 성과를 보였습니다.



### WebCanvas: Benchmarking Web Agents in Online Environments (https://arxiv.org/abs/2406.12373)
Comments:
          Our platform, tool and dataset are publically available at this https URL and this https URL

- **What's New**: 웹 에이전트가 실질적으로 유용하려면 사용자 인터페이스와 콘텐츠의 빈번한 업데이트로 특징 지어지는 지속적으로 발전하는 웹 환경에 적응해야 합니다. 그러나 대부분의 기존 벤치마크는 웹의 정적 측면만을 반영합니다. 이러한 격차를 해소하기 위해 웹 에이전트의 역동적인 웹 상호작용을 효과적으로 다루는 혁신적인 온라인 평가 프레임워크인 WebCanvas를 소개합니다.

- **Technical Details**: WebCanvas는 현실적인 평가를 용이하게 하기 위해 세 가지 주요 구성 요소를 포함합니다: (1) 중요한 중간 행동이나 상태를 신뢰성 있게 캡처하면서 중요하지 않은 이벤트나 변경된 웹 요소로 인해 발생하는 노이즈를 무시하는 새로운 평가 메트릭. (2) 542개의 태스크와 2439개의 중간 평가 상태를 포함하는 원래 Mind2Web 정적 데이터셋의 정제된 버전인 Mind2Web-Live라는 벤치마크 데이터셋. (3) 커뮤니티가 고품질, 최신 데이터셋을 수집하고 유지할 수 있도록 경량화되고 일반화된 주석 도구와 테스트 파이프라인.

- **Performance Highlights**: WebCanvas를 기반으로, 우리는 추론을 위한 확장 가능한 모듈을 포함한 에이전트 프레임워크를 오픈 소스화했습니다. Mind2Web-Live 테스트 세트에서, 우리의 최고 성능 에이전트는 23.1%의 태스크 성공률과 48.8%의 태스크 완료율을 달성했습니다. 또한, 다양한 웹사이트, 도메인, 실험 환경에서의 성능 격차를 분석했습니다.



### A Comparative Study of Continuous Sign Language Recognition Techniques (https://arxiv.org/abs/2406.12369)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 연구에서는 최근의 연속 수화 인식(Continuous Sign Language Recognition, CSLR) 기술을 다양한 데이터셋과 수화 언어에 걸쳐 평가하였습니다. 특히 RWTH-PHOENIX-Weather-2014, ArabSign, GrSL 데이터셋을 사용하여 새로운 벤치마크 성과를 제시하고, 도전적인 시나리오에서의 기술의 강건성과 일반화 가능성을 평가하였습니다.

- **Technical Details**: 연속 수화 인식 시스템은 주로 네 단계로 구성됩니다: 입력 비디오 스트림의 전처리, 프레임 시퀀스에서 공간 및 시간 특징 추출, 프레임과 글로스(glosses) 사이의 올바른 정렬 학습. 이 과정에서 Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Connectionist Temporal Classification (CTC) 등이 사용됩니다. 특히 최신 모델들은 CTC를 중심으로 한 순서 정렬 학습을 이용해 더 강력한 성능을 입증했습니다.

- **Performance Highlights**: 다양한 데이터셋에서 평가된 실제 실험 결과는 새로운 벤치마크 성과를 제시하며, 평가 프로토콜에 따른 성능 차이를 보여줍니다. 특히, 새로운 수화 사용자의 데이터와 미리 보지 못한 문장(Unseen-Sent) 평가에서는 시스템의 강력한 일반화 성능을 입증하였습니다.



### Does Context Help Mitigate Gender Bias in Neural Machine Translation? (https://arxiv.org/abs/2406.12364)
- **What's New**: 이번 연구는 Neural Machine Translation (NMT) 모델의 성 편향(gender bias)을 줄이기 위해 제안된 'context-aware' 접근법의 효과를 자세히 분석했습니다. 주로 직업명 번역에서 성 편향을 다루었으며, 영어에서 독일어와 프랑스어 번역, 바스크어에서 스페인어 번역을 통해 연구했습니다.

- **Technical Details**: 영어-독일어 및 영어-프랑스어 번역을 위해 WMT 2017, IWSLT 2017 등의 데이터 세트를 사용했고, 바스크어-스페인어 번역은 TANDO+ 및 COH-TGT:GENDER 테스트 세트를 활용했습니다. 문맥 인식 모델(context-aware models)은 Transformer 기반의 MarianNMT를 사용하여 훈련되었고, 문장 수준과 문서 수준 모델을 비교했으며, 특히 2to1 모델(문맥 문장 1개를 가진 모델)을 사용했습니다.

- **Performance Highlights**: 결과적으로 문맥 인식 모델이 여성형(feminine terms) 번역의 정확도를 약 30% 향상시켰으나, 특히 여성으로 고정된 직업에서만 주로 효과가 있었습니다. 예상과 달리 문맥이 포함되어도 여전히 성 편향이 유지되거나 강화될 수 있는 것으로 나타났습니다. 바스크어-스페인어 번역의 경우, 문맥이 명확하지 않은 상황에서는 남성형 번역의 정확도가 증가한 반면, 여성형 번역의 정확도는 감소하여 성 편향이 악화되었습니다.



### Cross-Lingual Unlearning of Selective Knowledge in Multilingual Language Models (https://arxiv.org/abs/2406.12354)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문은 다국어 언어 모델(Multilingual Language Models)에 대해 기계적 언러닝(machine unlearning) 기법을 최초로 제안합니다. 기존의 연구는 단일 언어 모델(monolingual models)에 초점이 맞춰져 있었으나, 다국어에서도 민감한 정보를 안전하게 제거하는 방식을 제안함으로써, 다국어 언어 모델이 다양한 언어에서 일관되게 성능을 유지하도록 발전시켰습니다.

- **Technical Details**: 제안된 방법론은 적응형 언러닝 계획(adaptive unlearning scheme)을 사용해, 각 언어의 성능에 따라 언어 종속 가중치를 할당합니다. 이 방식은 특정 데이터의 영향을 모델로부터 안전하게 제거하기 위해 다국어 교사 모델(multilingual teacher model)을 사용합니다. 학생 모델은 특정 언어에서 교사 모델의 능력에 따라 가중치를 달리하여 학습을 진행합니다. 예를 들어, 교사 모델이 특정 언어에 대해 강한 전문성을 갖고 있을 경우 높은 가중치를 적용합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 단일 언어 기반 언러닝 방법보다 뛰어난 성능을 보였습니다. 테스트는 두 개의 다국어 병렬 데이터셋(multilingual parallel datasets)을 사용하여 다양한 언어에서 특정 토큰 시퀀스와 사실 지식을 언러닝하는 방식으로 수행되었습니다. 제안된 방법은 더 빠른 시간 내에 다국어의 민감한 정보를 제거하면서도 성능을 유지하는 장점을 가지고 있어, 실제 적용 가능성을 높였습니다.



### Interpreting Bias in Large Language Models: A Feature-Based Approach (https://arxiv.org/abs/2406.12347)
- **What's New**: 최근 대형 언어 모델(LLM)들이 다양한 자연어 처리(NLP) 작업에서 탁월한 성과를 보였으나, 학습된 데이터셋의 다양성으로 인해 사회적 편견을 함께 내포하고 있다는 점이 부각되고 있습니다. 이 논문은 편향이 LLM에서 어떻게 전파되는지를 새로운 기능 기반 분석 방법론을 통해 조사합니다. 원인 중재 분석(causal mediation analysis)에서 영감을 받아 편향 관련 기능들의 진화를 가설하고, 활동 및 속성 패칭(activation and attribution patching)과 같은 해석 가능성 기법을 사용하여 이를 검증합니다.

- **Technical Details**: 본 연구는 다음 세 가지 주요 기여를 합니다. (1) LLaMA-2-7B, LLaMA-3-8B, Mistral-7B-v0.3 모델에 직업 데이터셋 템플릿을 사용하여 LLM의 편향을 분석하는 기능 기반 방법론을 도입하고 실증적으로 검증합니다. (2) 이 방법론을 다른 형태의 성별 편향에 확장하여 적용 가능성을 입증합니다. (3) 다층 퍼셉트론(MLP)과 어텐션 헤드(attention heads)가 편향 전파에서 가지는 역할을 구분하고, 반사실(dataset) 데이터를 사용한 타겟팅 편향 제거(debiasing)를 수행합니다.

- **Performance Highlights**: 연구 결과는 LLM에서의 편향이 복잡한 성질을 지님을 보여주고, 효과적인 편향 완화를 위한 맞춤형 전략의 필요성을 강조합니다. 특히, 각 기능이 편향 전파에 어떻게 기여하는지를 이해하는 데 중점을 두어, 보다 정교한 편향 완화 방법을 모색합니다.



### A Compass for Navigating the World of Sentence Embeddings for the Telecom Domain (https://arxiv.org/abs/2406.12336)
Comments:
          10 pages, 3 figures, 4 tables

- **What's New**: 이번 연구에서는 다양한 문장 임베딩 모델을 평가하고, 특히 전문 용어가 많은 텔레콤 도메인에서의 활용을 중점적으로 다루고 있습니다. 여러 공개된 모델과 이들 도메인 특화 버전에 대해 평가를 진행하며, 각각의 임베딩이 얼마나 효과적으로 정보를 조회할 수 있는지, 그리고 신뢰 구간을 어떻게 설정할 수 있는지에 대해 체계적인 방법을 제시합니다.

- **Technical Details**: 데이터셋 \\𝒟[np]\\(caligraphic_D)와 질문 세트 \\𝒬[np]\\(caligraphic_Q)에 대해 부트스트랩(bootstrap) 방법을 사용하여 정확도와 신뢰 구간을 평가합니다. 문장 임베딩 모델에서 생성된 임베딩을 이용해 유클리드 거리 L2(norm)로 정규화하고, 각 모델에 대해 코사인 유사도(cosine similarity)를 기준으로 점수의 분포를 분석합니다. 이소트로피(isotropy)와의 관계도 검토했으며, 최적의 유사도 임계값을 정의하는 시스템적인 방법을 제안합니다.

- **Performance Highlights**: 세밀한 튜닝(fine-tuning)을 통해 평균 정확도와 신뢰 구간의 긴밀도를 향상시킬 수 있음을 입증했습니다. 사전 훈련(pre-training)과 세밀한 튜닝을 함께 사용하는 경우 신뢰 구간이 더욱 개선되었습니다. 논문은 또한 도메인 적응이 이소트로피 점수를 향상시키면서도 모델의 검색 성능은 단순히 이소트로피 점수에만 의존하지 않음을 보여줍니다. 마지막으로, 도메인 적응을 통해 특정 도메인 임베딩이 일반 도메인 임베딩과의 거리가 더 멀어짐을 입증했습니다.



### Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters (https://arxiv.org/abs/2406.12335)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 효율성을 높이기 위한 새로운 방법, Value-Aware Token Pruning (VATP)을 제안합니다. VATP는 기존에 주로 사용된 attention score에 추가로 value 벡터의 $ \\ell_{1} $ norm을 함께 사용하여 토큰의 중요성을 평가합니다.

- **Technical Details**: 기존의 토큰 프루닝(Token Pruning) 연구들은 attention score에만 의존해 토큰의 중요성을 판단했습니다. 그러나 본 연구는 value 벡터의 노름(norm) 패턴이 불균등하게 분포함을 발견했습니다. VATP는 attention score와 value 벡터의 $ \\ell_{1} $ 노름을 결합해 토큰의 중요성을 평가하는 새로운 프루닝 기준을 제안합니다. 본 연구는 LLaMA2-7B-chat과 Vicuna-v1.5-7B 모델을 사용해 LongBench 벤치마크의 16가지 장기 문맥 과제에서 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과 VATP는 attention score만을 사용한 기존 방법들보다 우수한 성능을 보였습니다. 이 연구는 KV 캐시 효율화를 위해 가치 벡터의 중요성을 강조하며, 토큰 중요성을 평가하는 방법에 새로운 관점을 제시합니다.



### Retrieval Meets Reasoning: Dynamic In-Context Editing for Long-Text Understanding (https://arxiv.org/abs/2406.12331)
- **What's New**: 본 논문에서는 기존 Large Language Models (LLMs)이 고정된 컨텍스트 길이로 인해 긴 텍스트 내에서 다중 단계 추론(multi-hop reasoning)을 수행하는 데 한계가 있음을 지적하고, 이를 해결하기 위한 새로운 방법을 제안합니다. 이 방법은 동적 인-컨텍스트 편집(dynamic in-context editing)을 통해 정보를 재구성하여, LLM이 더 복잡한 추론을 할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 질문을 서브 질문으로 분해하고, 이를 유향 비순환 그래프(Directed Acyclic Graph, DAG) 형태로 구성하여 모델이 단계별로 추론할 수 있게 합니다. 이 접근법은 최근의 과학적 성과인 지식 편집(knowledge editing)에서 영감을 얻었으며, 모델이 외부 지식과 상호작용할 수 있도록 합니다. 특히, In-Context Editing 기법은 주어진 텍스트와 함께 모델에 지시사항이나 예시를 제공하여 모델의 생성 과정을 목표 지향적으로 안내합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 Llama2와 같은 제한된 컨텍스트의 LLM이 다중 단계 추론에서 현저하게 성능이 향상되었음을 보여줍니다. 이 방법은 최첨단(context window extrapolation) 방법보다 뛰어나며, 더 발전된 상용 장기 컨텍스트 모델과 비교했을 때에도 우수한 성과를 보였습니다. 또한 훈련 및 계산 비용을 줄이는 데 도움이 되어, 실용적인 해결책을 제시합니다.



### SNAP: Unlearning Selective Knowledge in Large Language Models with Negative Instructions (https://arxiv.org/abs/2406.12329)
Comments:
          16 pages, 5 figures

- **What's New**: 최근 인공지능에 대한 열광 속에서 대형 언어 모델(LLM)들은 사용자의 개인정보나 저작권 정보를 무심코 누출하는 위험이 제기되고 있습니다. 이를 해결하기 위해 새로운 머신 언러닝(Machine Unlearning) 방법론 SNAP을 제안합니다. 이는 목표 정보를 선택적으로 삭제하면서도 모델의 원래 성능을 유지하도록 설계되었습니다.

- **Technical Details**: SNAP 프레임워크는 주로 3단계로 구성됩니다: 1) LLM을 부정 명령(Negative Instruction)으로 훈련시켜 목표 정보를 지워진 응답을 생성하도록 합니다, 2) 원래 성능을 유지하기 위해 'Hard Positive' 데이터를 추가합니다, 3) 모델 파라미터의 변경을 최소화하기 위해 Wasserstein Regularization을 적용합니다. 이를 통해 모델의 일반적인 성능을 손상시키지 않으며 원하는 정보를 성공적으로 제거할 수 있습니다.

- **Performance Highlights**: 다양한 NLP 벤치마크 테스트에서 SNAP 프레임워크는 목표 정보를 선택적으로 삭제하면서도 원래 LLM이 제공하던 기능을 유지하는데 성공했습니다. 이는 실제 예시인 'Peter Parker' 정보를 모델에서 삭제하면서도 다른 질문에서는 여전히 정확한 대답을 제공합니다.



### PRePair: Pointwise Reasoning Enhance Pairwise Evaluating for Robust Instruction-Following Assessments (https://arxiv.org/abs/2406.12319)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)을 이용한 쌍별(pairwise) 평가와 단순 점별(pointwise) 평가 방법을 비교하여, 점별 평가가 편향성에 덜 영향을 받는다는 사실을 발견했습니다. 또한, 쌍별 평가에서 모델이 상대적으로 낮은 품질의 출력의 단점을 정확하게 파악하는 능력이 있음을 보여줍니다.

- **Technical Details**: 이번 연구에서는 두 가지 평가 방법, 쌍별(pairwise)와 점별(pointwise)을 비교하였습니다. 쌍별 평가는 두 개의 출력을 동시에 비교하고, 점별 평가는 각 출력을 개별적으로 평가하는 방식입니다. 이를 통해, 쌍별 평가가 정상 데이터셋에서 더 나은 성과를 보이지만, 교란(adversarial) 데이터셋에서는 성능이 크게 떨어지는 것을 확인했습니다. 이를 개선하기 위해, 연구진은 점별 추론(pointwise reasoning)을 쌍별 평가에 통합하는 하이브리드 방법인 PRePair를 제안하였습니다.

- **Performance Highlights**: 제안된 PRePair 방법론은 GPT-3.5-turbo의 정확도를 18.18 포인트, Claude-3 모델의 정확도를 12.85 포인트 향상시켰습니다. 이 방법은 정규 데이터셋에서의 성능을 유지하면서도 교란 데이터셋에 대한 평가에서 편향을 줄여주는 효과를 보였습니다. 실험 결과, 쌍별 evaluator는 MT-Bench에서 더 나은 성능을 보였지만, LLMBar 교란 데이터셋에서는 점별 evaluator가 더 우수한 성과를 보였습니다.



### Finding Task-specific Subnetworks in Multi-task Spoken Language Understanding Mod (https://arxiv.org/abs/2406.12317)
Comments:
          Accepted to Interspeech2024

- **What's New**: 새로운 연구에서는 다중 작업을 처리하는 음성 언어 이해(SLU) 모델의 특정 작업에 특화된 서브네트워크를 찾기 위해 신경망 가지치기(pruning)를 적용했습니다. 이를 통해 모델 압축뿐 아니라, 특정 작업을 위한 데이터를 추가 학습하는 동안 이전에 학습한 작업의 성능 저하 문제(카타스트로픽 포겟팅)를 완화할 수 있을 것으로 기대됩니다.

- **Technical Details**: 이 연구에서는 최신 다중 작업 SLU 모델 'UniverSLU'를 기반으로 하여 신경망 가지치기 기법을 사용하여 작업별 서브네트워크를 찾는 방법을 제안합니다. 모델은 감정 인식(ER), 의도 분류(IC), 자동 음성 인식(ASR) 등의 작업에 대해 학습되었습니다. 가지치기 네트워크를 통해 작업별 서브네트워크를 활성화하거나 비활성화하며, 이를 통해 불필요한 매개변수를 제거하고 지속적 학습(continual learning)을 개선합니다.

- **Performance Highlights**: 가지치기한 모델은 추가적인 ASR 또는 IC 데이터에 적응할 때, 이전에 학습된 작업의 성능 저하가 최소화된다는 점을 보였습니다. ER과 IC 작업의 성능이 향상되었으며, 이름 인식(NER), 음성 명령 인식(SCR) 등의 작업에서도 유사한 성과를 보였습니다. 이는 작업 간 유사성이 작업별 서브네트워크의 중복에 어떻게 기여하는지에 대한 통찰을 제공합니다.



### Can Tool-augmented Large Language Models be Aware of Incomplete Conditions? (https://arxiv.org/abs/2406.12307)
- **What's New**: 최근 대형 언어 모델(LLM)과 도구의 통합이 현실 세계에서 상호 작용하는 능력을 보완하는 데 많은 발전이 있었습니다. 이 연구는 불완전한 조건에서 LLM이 도구 사용을 자제할 수 있는지 여부를 조사하고, 필요한 도구나 정보가 부족할 때 이를 인식할 수 있는지에 대해 탐구합니다.

- **Technical Details**: 우리는 두 개의 데이터셋을 조작하여 불완전한 시나리오를 시뮬레이션했습니다. 필요한 도구가 없거나 도구를 호출하기 위한 정보가 부족한 경우를 포함시켜 데이터셋을 구성했습니다. 이후, 기존 도구를 사용할 수 없는 실험 조건에서 LLM이 그 조건을 인식할 수 있는지 평가했습니다. 실험에는 여러 LLM (GPT-4, ChatGPT, Claude-3 등)이 사용되었습니다.

- **Performance Highlights**: 실험 결과 대부분의 LLM은 특정 도구를 사용하려면 추가 정보가 필요하거나 적절한 도구가 없는 상황을 인식하는 데 어려움을 겪었습니다. 특히 사용자 제공 정보가 불완전하거나 현실 세계 도구를 사용할 때 불완전한 시나리오를 인식하는 데 어려움을 보였습니다. GPT-4는 다른 모델들보다 상대적으로 높은 정확도와 F1 점수를 기록했습니다. 예시로, API 대체 상황에서 GPT-4는 79.99/82.29의 정확도와 74.68/80.93의 F1 점수를 보였습니다.



### COT: A Generative Approach for Hate Speech Counter-Narratives via Contrastive Optimal Transpor (https://arxiv.org/abs/2406.12304)
Comments:
          IEEE jounrnals

- **What's New**: 이번 연구는 증오 발언(hate speech)을 효과적으로 대항하기 위한 새로운 프레임워크를 소개합니다. 기존의 방법론들이 주로 생성된 콘텐츠의 유창성에 초점을 맞추었다면, 본 연구에서는 증오 대상(LGBT, 이민자 등)에 대한 개별화와 관련성을 강화하는 방법을 제안합니다. 이를 위해 대조 최적 수송(contrastive optimal transport) 기반의 새로운 프레임워크를 도입합니다.

- **Technical Details**: 제안된 방법론에서는 Optimal Transport Kernel (OTK) 모듈을 활용하여 증오 대상 정보를 토큰 표현에 통합합니다. 이는 원본 및 수송된 특징 간의 비교 쌍을 구성하여 수행됩니다. 또한, 자기 대조 학습(self-contrastive learning) 모듈도 적용되어 모델의 퇴화 문제를 해결합니다. 마지막으로, 대상 지향 검색 방식(target-oriented search)을 사용하여 모델의 자신감 점수를 조정함으로써 각 토큰의 유사성과 대상 관련성을 고려한 디코딩 전략을 제시합니다.

- **Performance Highlights**: 두 개의 벤치마크 데이터셋을 기반으로 수행된 실험 결과는 제안된 모델이 현존하는 방법들보다 여러 평가 요소에서 우수하다는 것을 명확히 보여줍니다. 제안된 모델은 기존 접근법들에 비해 더욱 효과적이고 다양한 반대 내러티브를 생성할 수 있습니다.



### Fast and Slow Generating: An Empirical Study on Large and Small Language Models Collaborative Decoding (https://arxiv.org/abs/2406.12295)
- **What's New**: 대형 언어 모델 (LLMs)과 소형 언어 모델 (SLMs) 간의 협업 기반 디코딩 (collaborative decoding)을 통해 LLM들의 고비용, 높은 지연시간, 헛소리 생성(hallucination) 문제를 해결하는 새로운 방법을 제안합니다. 이를 위해, 인지 이론의 이중 과정 모델에 영감을 받아 빠르고 느린 생성 (FS-GEN, Fast and Slow Generating) 프레임워크를 구축했습니다.

- **Technical Details**: FS-GEN 프레임워크 내에서 도입된 주요 기술에는 추측적 디코딩 (speculative decoding), 대비적 디코딩 (contrastive decoding), 에뮬레이터 또는 대리자 미세 조정(emulator or proxy fine-tuning)이 포함됩니다. 이 연구는 LLM과 SLM의 차별화된 지식 역량을 분석하고, 모델 간 협업이 필요한 빈도 및 위치에 대한 통찰력을 제공합니다.

- **Performance Highlights**: 협업 빈도는 모델의 매개변수 비율과 관련된 스케일링 법칙에 따라 예측 가능하며, 협력 상호작용의 20% 미만이 필요함을 발견했습니다. 또한, 생성 과정에서 협업이 가장 효과적인 위치를 불확실성 관점에서 조사하여, 작은 모델이 다음 토큰 예측의 불확실성을 주요 지표로 삼아 대형 모델의 개입이 필요함을 밝혔습니다.



### What Matters in Learning Facts in Language Models? Multifaceted Knowledge Probing with Diverse Multi-Prompt Datasets (https://arxiv.org/abs/2406.12277)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 사실적 지식을 평가하기 위한 새로운 프레임워크인 BELIEF(-ICL)를 소개합니다. 이 프레임워크는 인코더 기반의 PLMs뿐만 아니라 디코더 기반 PLMs까지 다양한 시각에서 지식 이해 능력을 평가할 수 있는 다각적인 접근 방식을 제공합니다.

- **Technical Details**: BELIEF 프레임워크는 PLM의 정확성, 일관성 및 신뢰성을 평가하기 위해 다양한 프롬프트 데이터셋을 사용합니다. 이 프레임워크는 특정 언어 표현의 영향을 줄이기 위해 각 사실에 대해 다양한 프롬프트를 활용합니다. 이를 위해 자동 및 반자동화된 방식으로 생성된 MyriadLAMA 데이터셋을 활용하여 더 다양한 프롬프트를 통해 보다 정확한 평가를 가능하게 합니다. BELIEF는 fill-in-the-blank 방식의 데이터셋을 기반으로 하여 각 사실을 지식 삼중항(triple)으로 표현하고, 이에 대해 다양한 프롬프트를 생성하여 PLMs의 예측 정확도를 평가합니다.

- **Performance Highlights**: BELIEF 프레임워크를 다양한 인코더 및 디코더 기반 PLMs(BERT, Llama2 등)에 적용하여 평가한 결과, PLMs의 사실 이해 능력을 정확하고 포괄적으로 평가할 수 있음을 확인했습니다. 또한, 학습 중 사실 학습에 영향을 미치는 요인들을 분석하고, 프롬프트 기반 지식 탐사의 한계를 밝혀냈습니다.



### SafeInfer: Context Adaptive Decoding Time Safety Alignment for Large Language Models (https://arxiv.org/abs/2406.12274)
Comments:
          Under review

- **What's New**: 새로운 연구인 SafeInfer는 사용자 질의에 안전한 응답을 생성하기 위한 문맥 적응형, 디코딩 시점 안전 정렬 전략을 제안합니다. SafeInfer는 안전 증폭(Safety Amplification) 단계와 안전 안내 디코딩(Safety Guided Decoding) 단계를 포함하며, 이는 모델의 숨겨진 상태를 조정하여 안전 출력을 증가시키고, 안전 최적화 분포에 기반하여 토큰 선택을 유도해 생성된 콘텐츠가 윤리적 지침을 준수하도록 합니다.

- **Technical Details**: SafeInfer는 두 가지 주요 단계로 구성됩니다: (a) 안전 증폭(Safety Amplification, SA) 단계에서는 안전 증폭 벡터(Safety Amplification Vector)를 도출하기 위해 안전 데모 예제를 사용하며, 이 벡터는 언어 모델의 숨겨진 상태에 통합됩니다. (b) 안전 안내 디코딩(Safety Guided Decoding Strategy, sGDS) 단계에서는 언어 모델의 여러 분포를 통합/제거하여 편향된 속성을 제거하며, 특정 분포에서 토큰을 우선 선택함으로써 전반적인 출력 분포를 안전하게 최적화합니다.

- **Performance Highlights**: 새로운 평가 벤치마크인 HarmEval을 소개하여 광범위한 안전 평가를 수행했습니다. SafeInfer는 기본 및 수정된 다양한 대형 언어 모델에서 평가되었으며, 단순 프롬프트, 지침 중심 프롬프트, 연쇄 사고 프롬프트 세 가지 다른 프롬프트 기술을 사용하여 방법론의 범용성과 폭넓은 적용성을 입증했습니다.



### Unveiling Implicit Table Knowledge with Question-Then-Pinpoint Reasoner for Insightful Table Summarization (https://arxiv.org/abs/2406.12269)
Comments:
          work in progress

- **What's New**: 이 논문에서는 테이블 작성 셀 내에 숨겨진 암시적 지식을 활용하여 고품질의 테이블 요약을 생성하는 새로운 방법론을 제안합니다. 'Question-then-Pinpoint'라는 테이블 추론 프레임워크를 도입하여, 테이블 내에서 암시적 지식을 자문하고 이를 정확하게 식별한 다음 요약기에 대한 설명 가능한 지침을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 'Question-then-Pinpoint' 방식으로 설계되었습니다. 프레임워크는 질문을 스스로 생성하고, 그 질문에 대해 테이블에서 증거를 찾아 답변하는 과정을 거칩니다. 이를 위해 신뢰할 수 있는 테이블 추론기를 양성하며, 대형 언어 모델(LLM)을 이용해 초-정교한 추론 경로를 따르도록 유도하여 고품질의 지식을 추출하고 정제하는 두 가지 품질 향상 전략을 사용합니다.

- **Performance Highlights**: 두 개의 테이블 요약 데이터셋(새롭게 제안된 InsTaSumm 포함)에서 광범위한 실험을 통해 이 프레임워크의 일반적인 효과가 입증되었습니다.



### Towards a Client-Centered Assessment of LLM Therapists by Client Simulation (https://arxiv.org/abs/2406.12266)
- **What's New**:  이번 논문에서는 LLM(대규모 언어 모델) 기반의 상담사를 평가하는 새로운 방법인 ClientCAST를 소개합니다. ClientCAST는 LLM을 통해 모의 클라이언트를 생성하고, 이 클라이언트가 LLM 상담사와 상호작용하는 방식을 평가하는 기술입니다.

- **Technical Details**: ClientCAST는 세 가지 주요 측면에서 LLM 상담사를 평가합니다: 세션 결과(session outcome), 치료적 동맹(therapeutic alliance), 그리고 클라이언트의 자가 보고 감정(self-reported feelings). 모의 클라이언트는 LLM으로 구현되며, 상담 후 설문지를 작성하여 상호작용에 대한 평가를 제공합니다. 두 개의 데이터셋(High-Low Quality Counseling, AnnoMI)과 네 가지 LLMs(Claude-3, GPT-3.5, LLaMA3-70B, Mixtral 8*7B)를 사용하여 실험을 진행했으며, ClientCAST의 신뢰성을 검증했습니다.

- **Performance Highlights**: 실험 결과에 따르면, 모의 클라이언트는 제공된 심리적 프로파일과 일치하는 경향이 있으며, 고품질과 저품질의 상담 세션을 효과적으로 구별할 수 있었습니다. 더 고급 모델일수록 더 정확한 모의 클라이언트를 생성하는 것으로 나타났습니다. ClientCAST를 사용하여 네 가지 다른 LLM 상담사의 성능을 평가했습니다.



### Defending Against Social Engineering Attacks in the Age of LLMs (https://arxiv.org/abs/2406.12263)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 채팅 기반 사회 공학(CSE) 공격에 악용될 수 있는지, 그리고 이를 방어할 수 있는지 조사하였습니다. 새로운 데이터셋 SEConvo를 개발하여 학문 및 채용 등 다양한 실제 시나리오에서 LLM이 어떻게 악용될 수 있는지를 시뮬레이션하였습니다. 연구 결과, LLM의 공격 탐지 능력이 미흡하여 방어 비용이 증가함을 확인하였습니다. 이에 따라 ConvoSentinel이라는 모듈형 방어 파이프라인을 제안하였습니다.

- **Technical Details**: SEConvo는 GPT-4를 이용해 생성된 대화 1,400개로 구성되었으며, 악의적인 학문적 협력자나 채용자의 역할을 하는 공격자를 시뮬레이션합니다. 또한 ConvoSentinel은 검색 보강 모듈(Retrieval-Augmented Generation, RAG)을 통해 메시지를 기존의 의심스러운 대화 데이터베이스와 비교하여 악의적인 의도를 감지하는 식으로 동작합니다. SEConvo는 단일 LLM 시뮬레이션과 이중 에이전트 상호작용 모드를 포함해 다양한 시나리오에서 생성된 대화 데이터를 가지고 있습니다.

- **Performance Highlights**: ConvoSentinel은 메시지 및 대화 수준에서 악의적인 콘텐츠를 탐지하여 적응성과 비용 절감 측면에서 이전 모델들보다 우수하게 성능을 발휘합니다. SEConvo 데이터셋 생성 대화의 품질 확인을 위해 400개의 대화를 무작위로 선택해 인간 평론가들에게 평가받았으며, 악의적인 의도 탐지에 있어서 Fleiss Kappa 측정값 0.63으로 상당한 동의 수준을 보였습니다.



### A Hopfieldian View-based Interpretation for Chain-of-Thought Reasoning (https://arxiv.org/abs/2406.12255)
Comments:
          21 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 논리적 추론 성능을 향상시키기 위해 중요한 체인-오브-생각(CoT) 기법을 분석하고, 이 기법의 성공 이유를 체계적으로 설명하고자 합니다. 논문은 CoT 기법을 제로샷(Zero-shot) 및 퓨샷(Few-shot) 설정에서 분석하며, 새로운 '읽기-조절(Read-and-Control)' 접근법을 제안하여 CoT의 정확도를 제어합니다.

- **Technical Details**: 이 연구는 두 가지 주요 질문에 답하고자 합니다: (1) 제로샷 CoT에서는 '하나하나 생각해봅시다(let's think step by step)' 라는 프롬프트가 왜 모델의 출력을 크게 향상시키는가? (2) 퓨샷 CoT에서는 질문 전에 예시를 제공하는 것이 왜 모델의 추론 능력을 크게 향상시키는가? 이 질문에 답하기 위해, 연구진은 호프필디안(Hopfieldian) 관점에서 CoT 기법을 설명하고, '개념 모델링(Concept Modeling)', '개념 시뮬레이션(Concept Simulation)', '호프필디안 분석'의 세 가지 핵심 요소를 사용하는 프레임워크를 설계했습니다.

- **Performance Highlights**: 연구진은 세 가지 서로 다른 작업(산술 추론, 상식 추론, 상징적 추론)에 걸쳐 일곱 개의 데이터셋을 사용하여 광범위한 실험을 통하여 제안된 프레임워크가 CoT의 내부 작동 원리를 해독하고, 추론 오류 위치 파악 및 제어를 통해 올바른 추론 경로를 생성할 수 있음을 증명했습니다. 이 프레임워크는 에러 로컬라이제이션과 제어를 통해 CoT의 성공 요인을 이해하는 데 도움을 줄 수 있는 신뢰성 있는 설명을 제공합니다.



### Language and Multimodal Models in Sports: A Survey of Datasets and Applications (https://arxiv.org/abs/2406.12252)
- **What's New**: 최근 스포츠 분석 분야에서 자연어 처리(NLP)와 멀티모달 모델의 통합이 혁신적으로 발전하고 있습니다. 이 조사 논문은 2020년 이후 이러한 혁신을 이끄는 데이터셋과 애플리케이션에 대한 종합적인 리뷰를 제공합니다.

- **Technical Details**: 데이터셋은 주로 세 가지 유형으로 분류됩니다: 언어 기반 데이터셋, 멀티모달 데이터셋, 변환 가능한 데이터셋입니다. 언어 기반 데이터셋은 텍스트 관련 작업을, 멀티모달 데이터셋은 텍스트, 비디오, 오디오 등을 포함하는 작업을 목표로 합니다. 변환 가능한 데이터셋은 초기에는 단일 모달(예: 비디오)이지만, 추가 주석(annotation)을 통해 멀티모달 될 수 있습니다.

- **Performance Highlights**: 이 연구는 팬 경험 향상, 전술 분석 지원, 의학적 진단 등 다양한 애플리케이션에서 이러한 데이터셋의 기여를 강조합니다. 특히, LLM (Large Language Models)은 스포츠 애널리틱스에서의 애플리케이션을 크게 확장시켰습니다. 예를 들어, SNIL 시스템은 더 통찰력 있고 사용자 제공 통찰과 일치하는 스포츠 뉴스 기사를 생성할 수 있습니다.



### Mitigate Negative Transfer with Similarity Heuristic Lifelong Prompt Tuning (https://arxiv.org/abs/2406.12251)
Comments:
          ACL 2024 Findings

- **What's New**: 새로운 연구에서는 평생 프로ンプ트 튜닝(Lifelong Prompt Tuning)이 다양한 작업에서 파라미터 효율성을 높이고 저장 요구사항을 최소화했다는 점에서 주목할 만한 발전을 이루었다고 합니다. 그러나 현재의 방법론에서는 보편적인 알고리즘이 모든 작업에 대해 일관된 긍정적 전이를 보장하기 어렵다는 전이 가능성의 제약이 존재함을 강조하고 있습니다. 특히 상이한 작업들 사이에서 부정적 전이(negative transfer)를 유발할 수 있다는 점입니다. 이에 따라, 연구팀은 유사성 휴리스틱 평생 프로ンプ트 튜닝(Similarity Heuristic Lifelong Prompt Tuning, SHLPT) 프레임워크를 제안합니다. 이 새로운 전략은 학습 가능한 유사성 메트릭을 활용해 작업을 두 개의 하위 집합으로 분할하며, 이를 통해 작업 간 유사성 여부에 상관없이 유익한 전이를 가능하게 합니다.

- **Technical Details**: SHLPT는 먼저 학습된 작업들에 대한 프로ンプ트 풀(prompt pool)을 구성함으로써 망각(catastrophic forgetting)을 방지합니다. 다음으로 지식 전이 모듈을 세 가지 구성 요소로 분할합니다: (1) 현재 작업과 이전 작업 간의 유사성을 평가, (2) 유사한 작업과 상이한 작업으로 분류, (3) 각각의 하위 집합에 맞는 전이 알고리즘을 적용합니다. 이전 작업 중 유사한 작업에 대해서는 파라미터를 통합해 최적화된 초기점을 제공합니다. 반면 상이한 작업에 대해서는 다양한 정규화 기법을 도입해 사전 학습된 모델이 더 폭넓은 지식을 활용할 수 있도록 유도합니다.

- **Performance Highlights**: SHLPT는 평생 학습 벤치마크에서 기존 최신 기술을 능가하는 성능을 보여주었으며, 다양한 작업 순서에서도 부정적 전이에 대해 강력한 역량을 입증하였습니다. 특히 낮은 유사성을 특징으로 하는 작업 시퀀스에서 우수한 성능을 발휘하였습니다.



### PFID: Privacy First Inference Delegation Framework for LLMs (https://arxiv.org/abs/2406.12238)
Comments:
          Submitted to EMNLP2024

- **What's New**: 이번 논문은 LLMs (Large Language Models)에서 사용자 데이터를 보호하기 위한 새로운 프레임워크인 PFID를 소개합니다. 이 모델은 사용자 데이터의 모델 분할 (model sharding)과 특이값 분해 (SVD, singular value decomposition)를 통해 프라이버시 문제를 해결하고자 합니다. PFID는 사용자 입력을 위장하여 프라이버시 문제를 경감시키고자 합니다.

- **Technical Details**: PFID 프레임워크는 모델을 머리 부분, 중간 부분, 꼬리 부분으로 분할합니다. 머리와 꼬리 부분은 클라이언트 장치에 배치되고, 중간 부분은 서버에 저장됩니다. 서버와 클라이언트 간에는 프롬프트가 아닌 압축된 숨겨진 상태(hidden states)가 전송됩니다. 이 과정에서 클라이언트는 정보의 일부를 보유하여 숨겨진 상태를 다시 비공개화(re-privatized)할 수 있습니다. 이를 위해 특이값 분해 기법을 채택하여 전송 데이터를 압축하고 민감한 정보를 보존합니다.

- **Performance Highlights**: PFID 프레임워크는 전통적인 LLM 서비스와 유사한 성능을 유지하면서도 통신 효율성을 강조합니다. 클라이언트 장치는 일부 계산을 수행함으로써 서버의 계산 부담을 줄일 수 있습니다. 머신 번역 작업에서 실험을 통해 PFID의 성능을 검증했습니다.



### MCSD: An Efficient Language Model with Diverse Fusion (https://arxiv.org/abs/2406.12230)
Comments:
          8 pages, 9 figures

- **What's New**: MCSD 모델은 시퀀스 길이 증가에 따른 자원 소비 문제를 해결하기 위해 제안된 효율적인 언어 모델입니다. 주요 핵심은 다양한 특징을 융합하는 MCSD 블록을 통해 특징을 강력하게 표현하는 것입니다. 이 블록은 slope와 decay 섹션으로 구성되어 있으며, 다양한 시간 수용 필드를 통해 지역 및 글로벌 정보를 추출하는 기능을 제공합니다.

- **Technical Details**: MCSD 모델은 다중 채널 슬로프와 디케이 (Multi-Channel Slope and Decay, MCSD) 블록을 사용해 특징을 추출합니다. 이 블록은 다중 시계열 수용 필드를 사용하여 지역 및 글로벌 정보를 모두 포착할 수 있도록 설계되었습니다. 또한, 다양한 특징의 element-wise 융합을 통해 섬세한 특징 추출 능력을 강화합니다. 심상 단계에서는 회귀 표현을 통해 공간 복잡도를 O(1)으로, 시간 복잡도를 O(N)으로 줄여 빠른 추론 속도를 보입니다.

- **Performance Highlights**: 실험 결과에 따르면, MCSD는 Transformer 모델보다 높은 처리율과 낮은 GPU 메모리 소비를 보이는 반면, 벤치마크 테스트에서 대규모 언어 학습 모델에 비견할 만한 성능을 유지합니다. 이러한 특성들 덕분에 MCSD는 엣지 디바이스와 같은 자원 제약이 있는 환경에서 매우 유망한 솔루션으로 자리잡을 수 있습니다.



### ToxiCloakCN: Evaluating Robustness of Offensive Language Detection in Chinese with Cloaking Perturbations (https://arxiv.org/abs/2406.12223)
Comments:
          10 pages,5 Tables, 2 Figures

- **What's New**: 최근 연구에서는 중국어 데이터에서 교묘한 클로킹(perturbation) 기술을 이용한 공격성을 탐지하는 데 있어서 대형 언어 모델(LLMs)의 한계를 조사합니다. 이를 위해, ToxiCloakCN이라는 새로운 데이터셋을 도입했습니다. 이 데이터셋은 기존에 알려진 ToxiCN 데이터셋을 기반으로, 동음이의어와 이모지 대체를 통해 데이터를 변형시킨 것입니다.

- **Technical Details**: ToxiCloakCN 데이터셋은 ToxiCN 데이터셋을 샘플링하고, 이를 동음이의어와 이모지 대체한 과정으로 만들어졌습니다. 'base' 데이터셋은 2,293개의 공격성 있는 문장과, 균형을 맞추기 위해 2,289개의 비공격성 문장으로 구성되었습니다. 이 데이터셋에서 다양한 대형 언어 모델(LLMs)을 시험하여, 이들이 교묘한 변형 기법을 어떻게 감지하는지 평가했습니다.

- **Performance Highlights**: 실험 결과, 동음이의어와 이모지 대체와 같은 교묘한 변형이 현존하는 모델의 성능에 큰 영향을 미친다는 것을 보여주었습니다. 또한, 여섯 가지 다른 명령어를 사용하여 모델의 성능을 평가하고, 여러 타입의 공격성 콘텐츠가 변형 기술에 어떻게 영향을 받는지 분석했습니다. 성차별, 인종차별, 지역 편견, 반-LGBTQ+ 등의 분야에서 모델의 한계를 파악했습니다.



### On-Policy Fine-grained Knowledge Feedback for Hallucination Mitigation (https://arxiv.org/abs/2406.12221)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)의 헛소리(hallucination) 문제를 해결하기 위해 새로운 온라인 강화 학습 방법인 RLFH(Reinforcement Learning for Hallucination)를 도입했습니다. 이 방법은 기존의 학습 기반 방법들과는 달리 LLM이 내부 지식의 경계를 탐색할 수 있게 하고, 이러한 탐색에 대해 균형 잡힌 정확한 피드백을 제공합니다.

- **Technical Details**: RLFH는 모델이 생성한 응답을 원자적 사실(atomic facts)로 분해하고, 이러한 사실들을 평가하여 정확성과 유용성을 검증합니다. 이후, 이러한 평가 신호를 원래의 응답에 대한 토큰 수준의 세밀한 보상 신호로 변환하여 모델 행위를 조정합니다. 이를 통해 LLM이 허위 정보를 생성하는 행동을 완화할 수 있게 합니다. 또한, 이 프로세스를 자동화하기 위해 LLM 기반 사실 평가 프레임워크를 도입하여 인간의 개입 없이 신속하게 보상 신호를 생성합니다.

- **Performance Highlights**: HotpotQA, SQuADv2, Biography 벤치마크에서 실험한 결과, RLFH를 활용한 모델은 초기 모델에 비해 진실성과 정보성이 크게 향상되었습니다(평균 +17.9% FactScore). 또한, 기존의 학습 기반 방법들보다도 더 효과적으로 헛소리 생성을 완화하는 것으로 나타났습니다(평균 +2.0% FactScore).



### Is persona enough for personality? Using ChatGPT to reconstruct an agent's latent personality from simple descriptions (https://arxiv.org/abs/2406.12216)
Comments:
          Accepted to the ICML 2024 Workshop on Large Language Models and Cognition

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 단순한 사회 인구학적(socio-demographic) 및 성격 유형(personality type) 정보를 기반으로 복잡한 인지 속성을 재구성할 수 있는 능력을 탐구합니다. HEXACO 성격 프레임워크를 이용하여 이러한 모델들이 단순한 설명으로부터 잠재된 성격 차원을 얼마나 일관되게 예측하는지 관찰합니다.

- **Technical Details**: 연구에서는 HEXACO 모델을 사용하여 다차원적인 성격 유형을 재구성하는 능력을 평가합니다. 실험은 GPT-3.5와 GPT-4를 대상으로 진행되었으며, 이를 위해 다양한 프롬프트와 실험 세트를 설계했습니다. 60개의 문항으로 구성된 HEXACO 성격 테스트를 통해 모델이 제공된 '페르소나(persona)' 설명을 기반으로 성격 차원을 얼마나 잘 재구성하는지 평가합니다.

- **Performance Highlights**: 모델은 단순한 설명으로도 잠재된 성격 차원을 재구성하는 데 있어 상당한 일관성을 보여주었으나, 일부 비일관성과 편향도 관찰되었습니다. 예를 들어, 명시적인 정보가 부족한 경우 긍정적인 특성으로 기본 설정되는 경향이 있었습니다. 또한, 나이와 자녀의 수와 같은 사회 인구학적 요인이 재구성된 성격 차원에 영향을 미치는 것으로 나타났습니다. 이는 LLMs가 앞으로의 연구 및 개선이 필요한 분야임을 시사합니다.



### LLM-Oracle Machines (https://arxiv.org/abs/2406.12213)
Comments:
          4 pages

- **What's New**: 현대적인 AI 애플리케이션은 자연어 처리 작업에서 대형 언어 모델(LLMs)을 활용하여 풍부한 지식과 추론 능력을 제공합니다. 이 접근법은 오라클 튜링 머신(OTMs)의 개념과 일치합니다. 그러나 OTM의 개념을 확장하여 여러 LLMs 클러스터를 오라클로 사용함으로써 더 복잡한 계산을 캡처하고자 했습니다. 기본, 증강, 오류 회피, $\epsilon$ 오류의 네 가지 변형을 제안합니다. 앞의 두 가지는 일반적으로 관찰되며, 나머지 두 가지는 신뢰할 수 있는 결과를 보장하기 위해 LLM의 환각, 편향, 일관성 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: LLM-Oracle Machine (LLM-OM)은 LLM을 오라클로 사용하는 결정론적 알고리즘입니다. 기본적으로 LLM-OM은 입력된 작업을 처리하기 위해 중간 쿼리를 생성하고, 적절한 LLM으로부터 답변을 얻어 최종 답변을 도출합니다. LLM-OM는 입력 작업 Q를 하위 작업 Q1, Q2,..., Qm으로 분해하고 각각의 하위 작업을 LLM 클러스터를 활용하여 해결합니다. 추가적으로, 증강된 LLM-OM은 입력 텍스트 T와 그로부터 정보를 추출하는 작업 Q를 받으며, 답변이 T와 일치하도록 합니다. 이는 정보 환각 및 불충분성 문제를 해결하기 위해 설계되었습니다.

- **Performance Highlights**: 기본 LLM-OM은 ChatGPT와 Gemini 같은 많은 LLM 웹 애플리케이션의 기본 기능과 유사합니다, 사용자가 쿼리를 입력하면 LLM에서 직접 답변을 받습니다. 증강형 LLM-OM은 입력 텍스트 T와 작업 Q가 주어지면 T에 가장 적합한 답변을 생성하는 방식으로, 더욱 정교한 정보를 추출하거나 추론할 수 있습니다. 오작동을 회피하거나 잘못된 정보에 대한 확률을 낮추기 위한 메커니즘을 포함한 변형들은 향후 더 신뢰할 수 있는 AI 시스템 개발에 큰 도약이 될 것입니다.



### Knowledge Fusion By Evolving Weights of Language Models (https://arxiv.org/abs/2406.12208)
Comments:
          Accepted by ACL2024 Findings

- **What's New**: Evolver라는 새로운 지식 융합 방법이 도입되었습니다. 이는 진화 알고리즘에서 영감을 받아, 추가 학습이나 훈련 데이터가 필요하지 않습니다. 다양한 데이터 도메인에서 뛰어난 성능을 보이며, 기존 모델 합병 프레임워크와도 통합될 수 있는 유연한 도구를 제공합니다.

- **Technical Details**: Evolver는 다양한 훈련 시나리오에서 각각 파인튜닝 된 모델의 가중치를 집합체로 모아서, 돌연변이와 교차(crossover) 작업을 통해 자식 모델을 생성합니다. 그 후, 개발 데이터 세트에서 성능을 평가하여 성능이 향상된 모델을 남기고, 세대를 거쳐 최적의 모델을 선택합니다. 또한, task vector 메서드를 사용하여 차별 벡터를 도출하고, 이를 이용해 더 나은 가중치를 찾습니다.

- **Performance Highlights**: Evolver는 RoBERTa, DeBERTa, T5, GPT2와 같은 다양한 주류 언어 모델 실험에서 이전 최신 모델들을 뛰어넘는 성과를 나타냈습니다. sentiment classification 과 GLUE 데이터셋의 벤치마크 작업에서도 일관된 성능 향상을 보였으며, 이는 기존의 모델 합병 방법들과 결합하여도 추가적인 향상을 가져왔습니다.



### Debate as Optimization: Adaptive Conformal Prediction and Diverse Retrieval for Event Extraction (https://arxiv.org/abs/2406.12197)
- **What's New**: 최근 연구에서는 대규모 언어 모델(LLM)을 활용한 다양한 응용 분야에서의 발전이 두드러지고 있다. 하지만 이벤트 추출(Even Extraction) 분야에서 고급 튜닝 기반 접근 방식과 튜닝 없는 접근 방식 사이에는 여전히 큰 성능 격차가 존재한다. 이를 해결하기 위해 파라미터 조정 없이 LLM 출력을 반복적으로 개선하는 'Debating-as-Optimization (DAO)' 멀티 에이전트 시스템을 제안한다. DAO 시스템에는 두 가지 새로운 모듈, DRAG(Diverse-RAG) 모듈과 AdaCP(Adaptive Conformal Prediction) 모듈이 도입되었다. DRAG는 토론에 가장 잘 맞는 지원 정보를 체계적으로 검색하고, AdaCP는 효과적으로 덜 유망한 답변을 거부하여 이벤트 추출의 정확성과 신뢰성을 높인다.

- **Technical Details**: DAO 프레임워크는 이벤트 탐지(ED)와 이벤트 논거 추출(EAE) 두 가지 하위 작업을 통합된 토론 프로세스를 통해 처리하며, 각 하위 작업에 대해 구체적인 작업 지시어를 사용한다. DRAG 모듈은 동적으로 도메인별 데이터를 검색하여 현재 논쟁점에 가장 적합한 데이터를 제공하고, AdaCP 모듈은 적응형 확률 예측 정책을 사용하여 덜 설득력 있는 답변을 단계적으로 거부한다. 이러한 접근 방식은 도메인 특화된 지식 검색과 더 엄격한 거부 규칙 적용을 통해 이벤트 추출 답변을 점진적으로 개선한다.

- **Performance Highlights**: 실험 결과, ACE05와 CASIE 데이터셋에서 이벤트 탐지와 논거 추출의 성능 격차가 각각 18.1%와 17.8%, 17.9%와 15.2% 감소하였다. 이는 튜닝 기반 접근 방식과 비교하여 튜닝이 필요 없는 LLM 기반 방법의 성능 격차를 크게 줄이는 것이다.



### Aqulia-Med LLM: Pioneering Full-Process Open-Source Medical Language Models (https://arxiv.org/abs/2406.12182)
- **What's New**: 최근 의학 분야에서 성능이 부진했던 오픈소스 커뮤니티들을 위해, Aquila-Med라는 이중언어 의료 LLM이 제안되었습니다. 이 모델은 연속 사전 훈련(continue pre-training), 지도형 미세 조정(SFT), 인간 피드백을 통한 강화 학습(RLHF)을 통해 개발되었습니다. 이를 위해 대규모 중국어 및 영어 의료 데이터셋이 구축되었고, 모델과 데이터셋은 오픈소스로 제공될 예정입니다.

- **Technical Details**: Aquila-Med는 Aquila 모델을 기반으로 연속 사전 훈련을 통해 의료 지식을 습득하고, SFT 데이터셋과 Direct Preference Optimization (DPO) 데이터셋으로 정교하게 조정됩니다. SFT 데이터셋은 33만 예시를 포함하며, 15개 이상의 과목과 100개 이상의 질병 전문 분야를 다룹니다. DPO 데이터셋은 QA와 의료 선택 질문을 포함한 13,000개의 고품질 데이터 쌍으로 구성됩니다.

- **Performance Highlights**: Aquila-Med는 단발성과 다중턴 대화 및 의료 선택 질문에서 뛰어난 성능을 보여, 제안된 데이터셋이 모델의 단발 및 다중턴 의료 상담 처리 능력을 효과적으로 향상시키는 것으로 입증되었습니다. 성능 평가는 다양한 중국어와 영어 벤치마크에서 수행되었습니다.



### Statistical Uncertainty in Word Embeddings: GloVe-V (https://arxiv.org/abs/2406.12165)
- **What's New**: 이번 논문에서는 GloVe(word embedding 모델)의 재구성 오류 분산 추정치(reconstruction error variance estimates)를 제공하는 GloVe-V라는 방법을 소개합니다. 이 방법은 다변량 정규 모델(multivariate normal model)에 대한 분석적 근사치를 사용해, 불확실성을 포함한 임베딩 의 통계적 평가를 가능하게 합니다. 이를 통해 단어 쌍 간 유사성 비교, 모델 성능 평가, 코퍼스(corpus) 내의 인종이나 성별 편향 분석 등의 핵심 임베딩 작업에서 가설 검정을 수행할 수 있습니다.

- **Technical Details**: GloVe-V는 다변량 정규 확률 모델(multivariate normal probability model)에서 최적의 파라미터로 간주될 수 있는 문맥 벡터(context vector)와 상수(constant terms)를 고정했을 때, GloVe 단어 임베딩이 이 로그 변환된 행렬의 가중치(log transformation of co-occurrence matrix rows)에 대해 최적의 파라미터임을 활용합니다. 이 접근법을 통해 대규모 어휘에서도 효율적인 계산을 수행할 수 있습니다. 우리는 미국 영어 코퍼스(Corpus of Historical American English, COHA)에서 가장 자주 등장하는 단어들에 대한 사전 계산 된 임베딩과 분산을 데이터 제품으로 제공합니다.

- **Performance Highlights**: GloVe-V를 통해 불확실성을 포함한 분석이 가능해지면서, 텍스트 유사성, 모델 선택 및 텍스트 편향에 대한 결론이 달라질 수 있음을 보여주었습니다. 이 접근법은 텍스트 유사성 비교, 모델 성능 평가, 그리고 코퍼스의 인종 및 성별 편향 분석에서 신뢰할 수 있는 가설 검정(hypothesis testing)을 수행하는데 중요한 기여를 합니다.



### Exploring the Impact of a Transformer's Latent Space Geometry on Downstream Task Performanc (https://arxiv.org/abs/2406.12159)
- **What's New**: 최근 연구에서는 Transformer 기반의 대형 언어 모델이 사전 학습(pre-training)을 통해 특정 언어 지식을 학습한다고 생각해 왔습니다. 그러나 이번 연구에서는 이러한 이점이 특정 언어 지식과 별개로 잠재 공간(latent space)의 기하학적 특성에서 비롯될 수 있음을 제안합니다. 여러 BERT 유형의 언어 모델에서 GLUE 벤치마킹 과제 성능과 잠재 공간의 다양한 측정치 간의 관계를 조사한 결과, 양자화된 셀 밀도(quantized cell density)와 GLUE 성능 간에 강력한 선형 관계가 있음을 발견했습니다.

- **Technical Details**: Transformer 기반의 언어 모델, 예를 들어 BERT나 GPT 모델은 사전 학습과 미세 조정(fine-tuning) 절차를 통해 성능을 높입니다. 그러나 이 연구에서는 잠재 공간의 기하학적 특성이 이러한 성능 향상에 얼마나 기여하는지를 조사했습니다. 특히 'quantized cell density'라는 새로운 측정치를 도입하여, 이는 잠재 공간 표현에 적용될 때 GLUE 벤치마크 성능과 강한 선형 관계를 보였습니다. 또한, 몇 가지 비표준(non-standard) BERT 유형 모델에서도 이 측정치가 높은 예측력을 갖는 것을 발견했습니다.

- **Performance Highlights**: 잠재 공간의 특정 기하학적 특성은 사전 학습의 필요성을 줄이기 위한 전략으로 활용될 수 있을 것으로 보입니다. 연구 결과, 'quantized cell density' 측정치가 모델 초기화 시에 유용하게 사용될 수 있으며, 이는 모델의 사전 학습 요구량을 줄일 수 있음을 시사합니다.



### LLMs Are Prone to Fallacies in Causal Inferenc (https://arxiv.org/abs/2406.12158)
- **What's New**: 최근 연구에 따르면 대형 언어 모델(LLMs)은 프롬프트를 통해 인과 관계 정보를 효과적으로 추출할 수 있어 인과 추론(causal inference) 작업에 유용한 인과 그래프(causal graphs)를 생성할 수 있습니다. 하지만, 이러한 성공이 모델의 사전 훈련 데이터에 명시적으로 언급된 인과 관계를 기억하는 데 한정된 것인지 여부는 불분명합니다. 본 연구는 LLMs가 텍스트의 다른 관계 데이터에서 인과 관계를 추론할 수 있는지 조사했습니다.

- **Technical Details**: 이 연구는 LLMs가 기억한 인과 관계와 추론된 인과 관계의 역할을 분리하기 위해 공간적(spatial), 시간적(temporal), 반사실적(counterfactual) 관계를 포함하는 합성 데이터로 LLMs를 미세 조정(finetune)하고, 이러한 데이터로부터 인과 관계를 추론할 수 있는지 측정했습니다. 연구를 위해 사용한 주요 모델은 Llama2입니다. 실험은 이벤트 X와 Y 사이의 시간적, 공간적, 반사실적 관계에 기반하여 모델이 인과 관계를 예측할 수 있는지 평가하는 방식으로 진행되었습니다.

- **Performance Highlights**: LLM은 텍스트 내 두 개체 언급의 순서로부터 인과 관계를 추론하는 경향이 있으며, 순서가 무작위화된 경우에도 후건 오류(post hoc fallacy)로부터 자유롭지 못하다는 것을 발견했습니다. 추가로, LLM이 시간적 및 공간적 관계로부터 인과 관계의 부재를 정확히 추론할 수 있지만, 반사실적 관계로부터 인과 관계를 추론하는 데 어려움을 겪는다는 점을 확인했습니다. 또한, 더 큰 모델로 확장해도 이러한 성능 차이는 개선되지 않았습니다. 결과적으로, LLMs는 사전 훈련 데이터에 명시적으로 언급된 사실 외에는 새로운 인과 지식을 많이 추론하지 못할 가능성이 크다는 결론에 도달했습니다.



### A dual task learning approach to fine-tune a multilingual semantic speech encoder for Spoken Language Understanding (https://arxiv.org/abs/2406.12141)
Comments:
          In Proceedings of Interspeech 2024

- **What's New**: SAMU-XLSR 모델은 기존의 자기 지도 학습(Self-Supervised Learning, SSL) 모델들을 개선하여 다국어 음성 표현을 강화하고자 합니다. 이번 연구는 SAMU-XLSR 모델의 하위 작업 특화(separate SLU fine-tuning)로 인해 발생하는 다국어 성능 손실과 구체적 의미 훈련의 부족 문제를 해결하고자 새로운 이중 작업 학습 접근 방식을 제안합니다. 이 연구는 SLU 애노테이션이 제한적인 언어에서도 모델의 성능을 향상시키기 위해 다국어 간의 데이터 포터빌리티를 탐구합니다.

- **Technical Details**: 제안된 SAMU-XLSR은 다국어 음성 인코더인 XLS-R과 언어 중립적 BERT Sentence Embedding 생성기인 LaBSE를 결합하여 멀티모달 음성 표현을 풍부하게 합니다. 이 연구는 해당 모델이 언어별로 특화된 상황에서 초기 교차 언어 능력을 손실하지 않도록 두 가지 작업을 병행하는 다중 작업 학습 접근 방식을 제안합니다. 이를 통해 모델의 초기 의미 추상화 능력을 유지하면서 SLU 하위 작업에 대해 미세 조정합니다.

- **Performance Highlights**: 제안된 이중 작업 학습은 프랑스어 MEDIA, 이탈리아어 PortMEDIA, 튀니지어 TARIC-SLU 데이터셋에서 최첨단 성과를 도출했습니다. 특히, 두 저자원이 많은 데이터셋에서 다국어성과 교차 언어성의 향상을 입증했습니다. 이를 통해 SLU 작업에서 기존의 단순 연속 접근 방식에 비해 우수한 성능을 보여주었습니다.



### Gram2Vec: An Interpretable Document Vectorizer (https://arxiv.org/abs/2406.12131)
Comments:
          6 pages, 2 figures

- **What's New**: 이번 논문에서는 텍스트의 문법적 특징들의 정규화된 상대 빈도를 추출하여 문서를 고차원 공간에 임베딩하는 Gram2Vec 알고리즘을 소개합니다. Gram2Vec는 신경망 접근법과는 달리 특징 벡터가 생성되는 방식을 기반으로 본질적인 해석 가능성을 제공합니다. 이 알고리즘은 저자 식별(Authorship Attribution) 및 문체 분석에 투명하고 해석 가능한 접근 방식을 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: Gram2Vec는 문서의 길이로 정규화된 다양한 문법적 특징의 상대 빈도를 추출하여 문서를 벡터로 변환합니다. 이는 문자 발생 빈도, 19개의 구두점, 10개의 자주 사용하는 이모지, UD 태그 세트, POS 바이그램(2gram) 등을 포함한 9개의 특징 그룹으로 구성됩니다. 또한 Syntax Regex Matcher(SRM)을 사용하여 특정 구문 구조를 감지할 수 있습니다. 이러한 모든 특징 그룹은 spaCy 파서를 사용하여 태그를 얻습니다. 최종 결과는 각 특징 그룹의 벡터를 연결하여 얻어집니다.

- **Performance Highlights**: Gram2Vec는 저자 식별에서 왜 특정 문서가 특정 저자에게 귀속되는지 설명하는 데 사용할 수 있습니다. Cosine 유사성을 사용하여 후보 문서와 질의 문서 사이의 거리를 계산하여 해석 가능합니다. 비록 특수한 신경망 임베딩을 사용하는 접근법보다 성능에서는 뒤처질 수 있지만, Gram2Vec는 결과를 쉽게 해석할 수 있다는 장점이 있습니다.



### AI "News" Content Farms Are Easy to Make and Hard to Detect: A Case Study in Italian (https://arxiv.org/abs/2406.12128)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 '콘텐츠 농장(content farm)' 모델로 사용될 수 있음을 보여줍니다. 특히 이탈리아어 뉴스 기사 4만 개를 소량의 데이터로 사용하여 Llama(v1) 모델을 미세 조정한 결과, 이탈리아어 원어민조차 합성 텍스트를 실제 뉴스 기사로 착각할 정도의 뉴스와 유사한 텍스트를 생성할 수 있음을 발견했습니다.

- **Technical Details**: 이번 연구는 세 가지 LLM과 세 가지 합성 텍스트 탐지 방법(log-likelihood, DetectGPT, supervised classification)을 조사했습니다. Llama 모델(7B 및 65B 버전)을 사용하여 이탈리아어 뉴스 콘텐츠 농장 모델로 미세 조정했습니다. 또한, 실제 '콘텐츠 농장' 모델을 유사한 데이터셋으로 미세 조정하여 대리(proxy) 모델을 만들 가능성도 탐구했습니다.

- **Performance Highlights**: 이탈리아어 원어민들은 미세 조정된 Llama 모델이 생성한 텍스트를 64%의 정확도로 합성 텍스트로 인지했습니다. 이는 랜덤 추측(50%)보다 약간 높은 수준입니다. 탐지 방법들은 인간 평가자보다 더 나은 성능을 보였으나, 실제 환경에서는 실용적이지 않았습니다. 소량의 미세 조정 데이터(전체 데이터의 3%)만으로도 성공적인 대리 모델을 생성할 수 있었지만, 어느 기초 LLM을 사용했는지 알아야 하는 큰 어려움이 존재했습니다.



### Decoding the Narratives: Analyzing Personal Drug Experiences Shared on Redd (https://arxiv.org/abs/2406.12117)
Comments:
          Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 이번 연구는 약물 관련 서브레딧(Subreddits)과 같은 온라인 커뮤니티에서 사용자 생성 텍스트를 분석하는 다중 수준, 다중 라벨 분류 모델을 개발한 것입니다. 연구진은 새로운 분류 체계를 도입하여 게시물의 성격, 주제, 구체적인 목표를 평가했습니다. GPT-4 모델을 사용해 다른 모델보다 뛰어난 성능을 보였습니다.

- **Technical Details**: 연구진은 새로운 분류 체계(taxonomy)를 개발하여 게시물이 어떤 연결(Inquisition 또는 Disclosure)을 의도하는지, 주제(예: 회복, 의존성), 및 구체적인 목표(예: 재발, 품질, 안전성)를 평가했습니다. 데이터셋을 주석 달고 다양한 다중 라벨 분류 알고리즘을 비교한 결과, GPT-4 모델이 명령어, 정의, 예제를 제공받았을 때 다른 모든 모델을 능가한다는 것을 발견했습니다.

- **Performance Highlights**: 500개의 주석이 달린 drug-related 서브레딧 게시물을 대상으로 한 모델에서, GPT-4는 다른 모델 대비 높은 정확도로 범주를 라벨링 했습니다. 특히 안전(Safety), 약물 조합(Combination of Substances), 정신 건강(Mental Health)과 관련된 주제가 논의될 때 개인적인 의사 표명을 더 많이 했습니다.



### Enhancing Text Classification through LLM-Driven Active Learning and Human Annotation (https://arxiv.org/abs/2406.12114)
Comments:
          Publisher: Association for Computational Linguistics URL: this https URL

- **What's New**: 본 연구는 텍스트 분류에서 주석 (annotation) 비용 절감을 목표로, 인간 주석자와 대규모 언어 모델(LLMs)인 GPT-3.5를 통합한 새로운 액티브 러닝(Active Learning) 프레임워크를 소개합니다. 불확실성 샘플링에 기반한 액티브 러닝 기법을 사용하여 가장 정보량이 많은 샘플을 선택하고, 주석 비용을 대폭 줄이는 데 중점을 둡니다.

- **Technical Details**: 액티브 러닝은 머신러닝 알고리즘이 학습할 데이터를 선택적으로 선택할 수 있게 하여 학습 효율성을 최적화합니다. 특히 본 연구는 불확실성 샘플링(uncertainty sampling) 기법을 사용하여 모델이 가장 확신이 없는 샘플을 선택합니다. 이것은 이진 분류 작업에서 모델이 긍정 클래스에 속할 확률이 0.5에 가까운 샘플을 선택하는 것을 의미합니다. 제안된 프레임워크는 인간 주석자와 GPT-3.5의 출력을 통합하여 비용 효율성과 분류 성능 간의 균형을 유지합니다.

- **Performance Highlights**: IMDB 감성 분석, 가짜 뉴스 인식 및 영화 장르 다중 분류를 포함한 세 가지 공개 데이터셋에서 평가를 수행했습니다. 실험 결과, 주석 비용을 크게 줄이면서도 모델의 정확성을 유지하거나 향상시켰습니다. 특히, LLM의 불확실성 측정값에 따라 인간과 머신 주석자를 적응형으로 선택하는 전략이 주요 성과로 나타났습니다.



### Can LLMs Learn Macroeconomic Narratives from Social Media? (https://arxiv.org/abs/2406.12109)
- **What's New**: 이번 연구는 	extit{Narrative Economics} 가설을 실증적으로 테스트합니다. 이 가설은 바이럴하게 퍼지는 내러티브(narratives)가 경제 변동에 영향을 미칠 수 있다는 것을 제안합니다. 연구팀은 X(구 Twitter)에서 경제 관련 내러티브를 담은 두 개의 데이터셋을 소개하고, Natural Language Processing (NLP) 방법을 사용하여 트윗에서 내러티브를 추출하고 요약합니다. 이러한 내러티브의 거시경제 예측 능력을 테스트하며, 본 연구는 내러티브 데이터로 거시경제 모델을 개선하는 데 있어서의 도전 과제를 강조합니다.

- **Technical Details**: 연구팀은 2007년부터 2020년까지와 2021년부터 2023년까지의 두 개의 X 데이터셋을 수집하여 경제 관련 내러티브를 캡처했습니다. 첫 번째 데이터셋은 Twitter API를 사용해 2.4백만 개의 트윗을 수집했으며, 두 번째 데이터셋은 LLM 기반 내러티브 분석을 위해 2,881개의 트윗을 포함합니다. NLP 접근 방식을 통해 트윗에서 내러티브를 추출하고 요약하였으며, 이를 바탕으로 연방기금 금리(FFR), S&P 500, 그리고 CBOE 변동성 지수(VIX)를 예측합니다.

- **Performance Highlights**: 예측 성능 테스트 결과, 내러티브의 성공적인 추출이 거시경제 예측에 있어서 다소 개선된 결과를 보였으나, 단지 금융 정보만을 활용하는 것과 비교했을 때는 미미한 수준이었습니다. 이는 내러티브 경제학 이론의 유효성에 대한 의문점을 제기하며, 특히 거시경제 관점에서 이론을 검증하기 위한 새로운 모델과 과제가 필요함을 시사합니다.



### End-to-end Text-to-SQL Generation within an Analytics Insight Engin (https://arxiv.org/abs/2406.12104)
- **What's New**: 최근 Text-to-SQL(Text-to-SQL) 기술의 발전은 데이터베이스 관리 시스템을 통해 데이터 접근의 민주화를 촉진하고 있습니다. Distyl AI의 Analytics Insight Engine은 이러한 발전의 핵심으로서 특별한 기능을 제공합니다. 이 엔진의 초기 배포는 SQL 쿼리 작성, 비정형 요청의 저지연 처리, 그리고 도메인 특화 용어와 관행에 대한 이해라는 세 가지 주요 과제를 해결합니다.

- **Technical Details**: Text-to-SQL 쿼리 생성 파이프라인의 설계 및 구현은 대형 언어 모델(LLM)에 의해 구동되며, 다음과 같은 세 가지 주요 단계를 포함합니다. 첫째, 외부 지식을 추출하는 전처리 단계; 둘째, 쿼리 생성 시 적절한 외부 지식을 검색하는 단계; 셋째, 계층적 CTE 기반 (CTE-based) 구조를 따라 SQL 쿼리 생성을 분해하는 단계입니다. 특히, 전처리 단계에서는 도메인 특화 용어 및 관행을 포함한 문서와 이전 실행 로그에서 SQL 쿼리를 추출하여 외부 지식 세트를 구축합니다. 이후 생성 단계에서는 다수의 LLM 호출을 통해 자연어 입력 쿼리에 대한 SQL 쿼리를 생성하며, 적응 프레임워크는 사용자 피드백을 이용해 외부 지식을 업데이트합니다.

- **Performance Highlights**: 이 접근 방식은 도메인 특화 용어를 이해하고, 저지연 처리를 가능하게 하며, 복잡한 SQL 쿼리를 생성하는 데 탁월한 성능을 보입니다. 전처리, 생성, 적응 프레임워크가 협력하여 지속적인 쿼리 생성을 개선하고 있으며, 특히 Hierarchical CTE 기반 접근 방식과 LLM 기술의 결합은 차별화된 성능을 제공합니다.



### Who's asking? User personas and the mechanics of latent misalignmen (https://arxiv.org/abs/2406.12094)
- **What's New**: 이 연구는 안전 조정된 모델들에서 미조정 능력이 여전히 잠재적으로 존재하는 현상을 조사하며, 이러한 능력이 사용자 페르소나(user persona)에 따라 어떻게 드러나고 조작될 수 있는지 보여줍니다. 연구는 특정 레이어에서 디코딩을 통해 유해한 콘텐츠가 여전히 추출될 수 있음을 밝히고 있습니다. 또한 사용자 페르소나를 조작하는 것이 모델의 거부 행동을 조절하는데 더 효과적임을 발견했습니다.

- **Technical Details**: 연구는 두 가지 방법, 자연어 프롬프트(natural language prompting)와 활성화 조작(activation steering)을 사용하여 모델의 거부 행동을 조작하려고 시도했습니다. 실험 결과 활성화 조작이 훨씬 더 효과적임을 보여주었습니다. 더욱이, 사용자 페르소나를 조작하여 모델의 거부 행동에 어떤 영향을 미치는지 조사했으며, 특정 페르소나가 더 유해한 질문에 대한 해답을 쉽게 끌어낼 수 있음을 발견했습니다.

- **Performance Highlights**: 연구는 Llama 2 13B와 Vicuna 13B 모델을 사용하여 실험을 수행했으며, AdvBench에서 제공한 500개의 프롬프트를 사용해 유해한 정보 추출을 시도했습니다. 자연어 프롬프트로는 대부분 거부 반응을 보였지만, 활성화 조작을 통해 유해 정보를 디코딩할 수 있었습니다. 그 결과, 유해한 질문에 대해 '더 이타적인' 사용자가 질문할 때 모델이 더 쉽게 답변을 제공한다는 것을 발견했습니다.



### When Reasoning Meets Information Aggregation: A Case Study with Sports Narratives (https://arxiv.org/abs/2406.12084)
- **What's New**: 새로운 연구는 장기적으로 축적된 데이터를 정확히 집계하는 과정이 추론에 얼마나 중요한지를 강조합니다. 이를 위해 스포츠 데이터를 활용한 실험을 통해 롱기튜드 데이터(longitudinal data)와 정보 밀도가 높은 시나리오에서 LLMs (Large Language Models)의 추론 능력을 평가합니다. 실험 결과, 대부분의 모델들, 특히 GPT-4o,는 자주 발생하는 점수 집계에 있어서 정확하지 않으며, Llama-3과 같은 오픈 소스 모델들은 'score hallucinations' 문제로 고생합니다.

- **Technical Details**: SportsGen이라는 새로운 방법을 제안하여 실제 NBA 농구 데이터를 활용, 게임 내러티브를 종합합니다. 이 방법은 플레이-바이-플레이(play-by-play) 서술을 분석하여 점수를 추론하고, 관련 있는 엔티티를 식별하며, 선수와 팀에게 정확히 점수를 할당하고, 주요 통계를 요약하여 결론을 도출합니다. 이 연구는 LLM의 다중 홉(multi-hop), 연역적(deductive), 귀납적(inductive), 추론(abductive) 등 다양한 추론 능력을 조사합니다.

- **Performance Highlights**: LLM이 점수를 정확히 집계하는 데 어려움을 겪고 있음이 관찰되었습니다. 특히 농구 경기에서 자주 발생하는 점수로 인해 행동을 추적하는 것이 어려웠습니다. 또한, 입력된 내러티브가 짧고 지시사항이 긴 경우에는 내러티브를 간과하고 점수를 상상(=hallucinate)하는 경우가 많았습니다. SportsGen 접근법은 텍스트와 수치 데이터를 모두 합성하여 복잡한 추론 시나리오를 시뮬레이션하는 데 중요한 역할을 합니다.



### COMMUNITY-CROSS-INSTRUCT: Unsupervised Instruction Generation for Aligning Large Language Models to Online Communities (https://arxiv.org/abs/2406.12074)
- **What's New**: 최근 대규모 언어 모델(LLMs)을 활용한 기술이 온라인 커뮤니티의 디지털 트윈(Digital Twins)을 생성하는 데 유용하게 사용되고 있습니다. Community-Cross-Instruct라는 비지도 학습(unsupervised learning) 프레임워크가 소개되었는데, 이는 기존의 인간 작성 지침이 아닌 자동 생성된 지침을 이용하여 LLM을 온라인 커뮤니티에 맞추는 방법입니다. 이 기술은 특히 Reddit의 정치 및 피트니스 커뮤니티에 대해 테스트되었으며, 이러한 커뮤니티를 보다 정확하게 대표하는 모델을 생성하는 데 성공했습니다.

- **Technical Details**: Community-Cross-Instruct는 고급 LLM인 Claude-3를 사용하여 Reddit 포럼들의 다양한 논의에서 지침-응답 쌍(instruction-output pairs)을 자동으로 생성합니다. 이 쌍은 Community-Cross-Instruct에 의해 자동으로 커뮤니티별 개방형 지침(CommInst)과 다중 선택 설문 질문(CommSurvey)으로 분류됩니다. CommInst를 사용하여 기초 LLM(GPT-3.5 또는 Llama-3)을 미세 조정(finetune)하고, CommSurvey를 통해 모델의 커뮤니티 정렬(alignment) 성능을 평가합니다. 기존의 방법들은 도메인 전문가가 작성한 지침을 필요로 했지만, 이번 프레임워크는 광범위한 온라인 커뮤니티 데이터를 활용하여 이러한 병목 현상을 극복합니다.

- **Performance Highlights**: Reddit 포럼 데이터를 통해 우리의 방법이 정치 및 피트니스 도메인에서 커뮤니티 대표성의 정확성을 크게 향상시킴을 보여주었습니다. 이는 표준 인물 적응(persona adaptation) 방법을 훨씬 능가하는 정렬 성능을 보였습니다. 또한, 이 연구는 생성 AI가 온라인 커뮤니티에 대한 통찰을 제시하는 데 있어 큰 잠재력을 가지고 있음을 강조합니다.



### Satyrn: A Platform for Analytics Augmented Generation (https://arxiv.org/abs/2406.12069)
- **What's New**: Satyrn이라는 새로운 neurosymbolic 플랫폼이 제안되었습니다. Satyrn은 대규모 데이터베이스로부터 체계적 데이터 분석을 통해 생성된 사실 세트를 사용하여 리포트를 생성하는 analytics augmented generation(분석 증강 생성) 접근 방식을 활용합니다. 이는 텍스트로만 정보를 가져오는 retrieval augmented generation(RAG) 방법의 한계를 극복합니다.

- **Technical Details**: Satyrn은 Structured Question Representation(SQR)라는 새로운 계획 표현 언어를 사용하여 복잡한 데이터를 분석할 수 있게 합니다. 이 시스템은 파일링된 정보 요청(JSON 객체)을 입력으로 받고, 요구 사항을 충족시키는 리포트를 생성합니다. 이를 위해 Satyrn은 데이터의 객체, 속성 및 관계를 식별하는 'ring'이라는 경량의 지식 표현을 사용합니다. 이를 통해 다양한 데이터셋에 대해 표준 분석 기법을 적용할 수 있습니다.

- **Performance Highlights**: Satyrn은 기존의 솔루션인 Code Interpreter와 비교했을 때 사실적 정확도가 86%에 달하며, 이것은 57%에 그치는 Code Interpreter보다 훨씬 높은 수치입니다. 또한 Satyrn은 Mistral-7B와 같은 작은 모델을 사용하면서도 높은 정확성을 유지할 수 있어, 자원이 제한된 환경에서도 효과적으로 동작합니다.



### Language Models are Surprisingly Fragile to Drug Names in Biomedical Benchmarks (https://arxiv.org/abs/2406.12066)
Comments:
          submitted for review

- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 의료 분야에서 사용할 때 발생할 수 있는 약물 이름의 상호 교환성 문제에 대한 강건성을 평가하기 위해 RABBITS라는 새로운 데이터셋을 소개했습니다. 이 데이터셋은 전문 의사들이 브랜드 이름과 일반명을 교체하여 의료 베치마크의 성능 차이를 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 연구진은 RxNorm National Library of Medicine을 사용하여 브랜드와 일반 약물 이름을 추출하고, 이를 MedQA와 MedMCQA 데이터셋에 적용하여 성능 변화를 분석했습니다. 데이터셋은 2명의 의사가 여러 번 검토하여 정확성을 보장했습니다. 모든 데이터와 모델은 소유자의 라이선스를 준수하며, HuggingFace 리더보드를 통해 새로운 모델의 성능을 평가할 수 있습니다.

- **Performance Highlights**: LLMs는 브랜드 이름과 일반명 교체 시 평균 4% 정도의 성능이 떨어지는 것으로 나타났습니다. 특히 큰 오픈 소스 모델(Llama-3-70B)은 정확도가 76.6%에서 69.7%로 감소했습니다. MedMCQA 데이터셋에서의 성능 저하가 MedQA보다 더 큰 것으로 확인되었습니다.



### Not Eliminate but Aggregate: Post-Hoc Control over Mixture-of-Experts to Address Shortcut Shifts in Natural Language Understanding (https://arxiv.org/abs/2406.12060)
Comments:
          Accepted to TACL (pre-MIT Press publication version, 21 pages, 5 figures)

- **What's New**: 이 논문에서는 자연어 이해(NLU) 모델이 데이터셋에서 발생하는 간단한 패턴(shortcuts)에 의존하는 문제를 해결하기 위한 새로운 접근법을 제안합니다. 기존 연구들은 이러한 shortcuts를 제거하는 훈련 기법에 중점을 두었지만, 이 논문에서는 다른 접근법을 탐구합니다. 각각 다른 잠재 특징(latent features)을 포착하는 mixture-of-experts의 예측을 비관적으로 결합하는 방식으로, 분포 변화(distribution shifts)에 대한 모델의 강건성을 크게 향상시킵니다.

- **Technical Details**: shortcuts는 데이터의 레이블과 잠재 특징 사이의 가짜 상관관계(spurious correlations)로 인해 발생합니다. 이 연구에서는 모델을 훈련시키기보다는, 각 전문가는 상대적으로 다른 잠재 특징을 포착한다고 가정하여 mixture model의 예측을 결합하는 새로운 방법을 제안합니다. 이 방법은 이론적으로 기반이 된 사후 통제(post-hoc control)를 통해 shortcuts의 변화에 따른 예측 위험을 최소화합니다. 이 접근법은 기존의 학습 단계에서 shortcut에 의존하지 않도록 모델을 훈련시키는 방법과 차별화됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 새로운 접근법은 shortcut의 변화에 직면했을 때 모델의 강건성을 크게 향상시키는 것으로 나타났습니다. 또한, 이 방법은 잠재 특징의 변화를 탐지하기 위해 mixture weights를 사용할 수 있는 실용적 이점을 제공합니다. 이를 통해 ID(in-distribution)와 OOD(out-of-distribution) 데이터 간의 성능 trade-off 문제를 해결할 수 있습니다. 마지막으로, 하이퍼파라미터 조정을 OOD 데이터 없이 ID 데이터만으로도 수행할 수 있습니다.



### InternalInspector $I^2$: Robust Confidence Estimation in LLMs through Internal States (https://arxiv.org/abs/2406.12053)
Comments:
          8 pages

- **What's New**: Large Language Models (LLMs)의 출력 신뢰도를 향상시키기 위한 새로운 프레임워크인 InternalInspector을 소개합니다. InternalInspector은 대조 학습(contrastive learning)을 활용하여 모형의 내부 상태를 분석하고, 기존 방법보다 정확하게 예측의 신뢰도를 평가합니다.

- **Technical Details**: InternalInspector은 모든 레이어의 주의 상태(attention states), 피드-포워드 상태(feed-forward states), 활성 상태(activation states)를 포함한 내부 상태를 분석합니다. 대조 학습은 Convolutional Neural Network(CNN)나 Transformer 같은 인코더를 통해 의미 있는 특징을 학습합니다. 이 특징을 토대로 이진 분류기를 훈련시켜 각 예측이 맞는지 틀린지에 대한 신뢰도를 추정합니다.

- **Performance Highlights**: 다양한 자연어 이해 및 생성 작업에서 InternalInspector은 기존 신뢰도 추정 방법보다 최대 20.4%의 정확도 향상과 8.9%의 예상 보정 오류(ECE) 감소를 기록했습니다. 또한 InternalInspector은 허루에발(HaluEval) 벤치마크에서 기타 내부 기반 신뢰도 추정 방법을 능가하며 환각(hallucination) 감지에 뛰어납니다.



### UniGLM: Training One Unified Language Model for Text-Attributed Graphs (https://arxiv.org/abs/2406.12052)
- **What's New**: UniGLM (Unified Graph Language Model) 프레임워크를 도입합니다. UniGLM은 여러 도메인의 Text-Attributed Graphs(TAGs)에 대해 일반화할 수 있는 첫 번째 그래프 임베딩 모델입니다. 이는 구조적 유사 노드를 식별하는 적응형 긍정 샘플 선택 기법과 반복적인 인코딩 계산을 최소화하여 학습을 가속하는 'lazy contrastive module'을 포함합니다.

- **Technical Details**: UniGLM은 다양한 도메인과 규모의 TAG를 사용하여 자기 지도 대조 학습(self-supervised contrastive learning)을 통해 훈련됩니다. 구조적 유사 노드를 식별하는 적응형 긍정 샘플 선택 기법과 텍스트 속성의 반복적인 인코딩을 피하기 위한 동적 메모리 뱅크를 포함합니다. 이는 노드의 로컬 및 글로벌 컨텍스트를 고려하여 긍정 샘플을 선택함으로써 다양한 TAG 간의 텍스트와 구조적 정보를 효과적으로 정렬합니다.

- **Performance Highlights**: 9개의 벤치마크 TAG에 대한 광범위한 실험 결과에 따르면 UniGLM은 여러 다운스트림 작업(노드 분류 및 링크 예측) 및 백본(GNN 및 MLP)에서 최첨단 그래프 임베딩 모델을 능가합니다. 또한 이전에 보지 못한 TAG에 대해 유용한 임베딩을 생성할 수 있는 일반화 및 전이 학습 성능을 입증하였습니다.



### Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning (https://arxiv.org/abs/2406.12050)
- **What's New**: 이번 논문은 수학적 사고력 문제 해결 능력을 향상시키기 위한 Reflective Augmentation (RefAug) 기법을 소개합니다. 이 기법은 각 훈련 데이터에 문제 반영을 포함시켜 모델이 대안적 관점을 고려하고 추상화와 유추를 통해 반성적 사고를 할 수 있게 돕습니다.

- **Technical Details**: 전통적인 데이터 확장(Data Augmentation) 기법과 다르게 RefAug는 훈련 데이터의 시퀀스 차원에서 작동합니다. Stacey et al. (1982)의 정의에 따라 반영(reflection) 섹션을 문제에 추가하여 두 가지 유형의 반영, 즉 대안적 사고(alternative reasoning)와 후속 사고(follow-up reasoning)를 훈련합니다. 이를 위해 GPT-4-turbo를 사용하여 반영 섹션을 주석 처리했습니다.

- **Performance Highlights**: 다양한 수학적 추론 작업에서 RefAug는 다음과 같은 여러 가지 이점을 가져다줍니다: (1) 표준 단일 라운드 질문-답변(QA) 설정에서 +7.2의 정확도 향상을 제공, (2) 기존 데이터 확장 기법이 효과가 없는 복잡한 반영적 수학 추론 시나리오에서 성능을 크게 향상, (3) 기존 데이터 확장 기법과의 상호 보완적 특성을 제공하여 통합 시 더 높은 성능 향상을 실현.



### Soft Prompting for Unlearning in Large Language Models (https://arxiv.org/abs/2406.12038)
- **What's New**: 최근 인공 지능과 데이터 보호 규정의 중요성이 대두되면서, 대형 언어 모델(LLMs)에서 '기계 학습 잊기'(machine unlearning)를 적용하는 연구가 진행되고 있습니다. 이 논문에서는 데이터 보호 규제를 준수하기 위해 소프트 프롬팅(soft prompting)을 이용한 경량화된 잊기 프레임워크인 SPUL(Soft Prompting for Unlearning)을 제안합니다.

- **Technical Details**: SPUL은 학습된 소프트 프롬프트 토큰을 활용하여 특정 데이터를 잊게 만드는 메커니즘을 구현합니다. 이 과정에서 손실 함수를 통해 잊기와 유용성 유지를 동시에 달성하도록 설계되었습니다. SPUL은 대형 언어 모델의 파라미터를 변경하지 않고도 원하는 데이터를 잊게 할 수 있는 특징을 갖고 있습니다.

- **Performance Highlights**: 성능 검증 결과, SPUL은 텍스트 분류 작업에서 잊기와 유용성 간의 균형을 크게 개선할 수 있음을 보여줍니다. 다양한 대형 언어 모델에서 SPUL의 확장성을 입증하고, 하이퍼파라미터 선택과 잊기 데이터 크기의 영향을 분석한 결과를 제시합니다.



### MedCalc-Bench: Evaluating Large Language Models for Medical Calculations (https://arxiv.org/abs/2406.12036)
Comments:
          Github link: this https URL HuggingFace link: this https URL

- **What's New**: 새로운 연구로 MedCalc-Bench라는 독특한 데이터셋이 제안되었습니다. 이 데이터셋은 대형 언어 모델(LLMs)의 의료 계산 능력을 평가하는 첫 번째 시도로, 55개의 다양한 의료 계산 작업에서 1000개 이상의 수동 검토된 사례를 포함하고 있습니다.

- **Technical Details**: MedCalc-Bench는 MDCalc에서 제공하는 55개의 일반적인 의료 계산 작업을 수집하여, 180k개의 공개된 환자 노트 중 각 계산 작업에 맞는 노트를 선택하고, 필요한 값을 추출하여 수작업으로 검토한 후, 각 계산 작업에 대한 지침을 담은 템플릿을 사용해 단계별 설명을 생성했습니다. 이를 통해 각 인스턴스는 환자의 노트, 특정 의료 값을 계산하는 질문, 수동 검토된 정답 및 계산 과정을 설명하는 스텝별 설명을 포함하고 있습니다.

- **Performance Highlights**: 여러 오픈 소스 및 상업적 LLMs를 사용한 평가 결과, GPT-4가 가장 높은 정확도인 50.9%를 달성했음에도 불구하고, 모든 현재 모델은 의료 계산 작업에 적합하지 않으며 정확성과 계산 지식 및 산술 수행에서 많은 한계를 보였습니다.



### Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts (https://arxiv.org/abs/2406.12034)
- **What's New**: 우리는 대형 언어 모델(LLM)을 MiXSE(MiXture of Self-specialized Experts)라 불리는 자기 전문화된 전문가들로 이루어진 구성 시스템으로 변형시키는 Self-MoE 접근법을 소개합니다. Self-MoE는 자기 생성 합성 데이터를 사용하여 전문가 모듈을 구성하고, 각 모듈은 공유된 기본 LLM과 자기 최적화 라우팅을 통합합니다. 이를 통해 다양한 목표 작업을 동적으로 처리하고 인간이 라벨링한 데이터 및 추가 파라미터 없이 전반적인 성능을 향상할 수 있습니다.

- **Technical Details**: Self-MoE는 기본 LLM을 분할하여 가볍고 독립적인 전문가 모듈들을 생성하는 방식으로 구성됩니다. 이러한 모듈들은 인간이 라벨링한 데이터에 의존하지 않고 자생적으로 생성된 합성 데이터를 사용하여 구성되며, 자기 최적화 라우팅 메커니즘을 통해 각 작업에 동적으로 전담할 수 있습니다. 이는 모노리식(일체형) 모델이 고정된 파라미터로 기존 학습 내용을 잊는 것과 달리, 각 전문가의 무결성을 보존하며 다양한 도메인 작업을 정밀하게 처리할 수 있습니다.

- **Performance Highlights**: 우리의 실험 결과에 따르면, Self-MoE는 모든 목표 도메인에서 기본 LLM을 능가하는 성능을 보였습니다. 특히, 지식, 추론, 수학, 코딩 등 다양한 벤치마크에서 상당한 개선을 달성했습니다. 또한 인스턴스 병합 및 가중치 병합 등의 다른 방법보다 유연성과 해석가능성이 뛰어난 성능을 보였습니다. 라우팅 분포에 대한 시각적 해석도 제공하여 작업이 관련 전문가에게 동적으로 할당되는 과정을 설명합니다.



### Unveiling and Mitigating Bias in Mental Health Analysis with Large Language Models (https://arxiv.org/abs/2406.12033)
Comments:
          In submission; Data and code are available at: this https URL

- **What's New**: 이 연구는 다양한 응용 분야에서 강력한 역량을 보여준 대형 언어 모델(LLMs)의 공정성 문제를 집중적으로 조사한 첫 번째 연구입니다. 특히, 정신 건강 분석에서 공정성과 성능의 균형을 평가하여, 취약한 인구 집단에 미칠 수 있는 위험을 평가합니다.

- **Technical Details**: 이 연구에서는 10개의 상이한 LLMs, 예를 들어 GPT-4, Llama2, MentalRoBERTa 등을 8개의 서로 다른 정신 건강 데이터셋을 사용해 평가합니다. 성별, 나이, 종교와 같은 7가지 사회적 요인을 고려하여, LLM들이 정신 건강 예측에서 얼마나 공정한지에 대한 체계적인 평가를 수행하였습니다. Zero-shot standard prompting과 few-shot Chain-of-Thought(CoT) prompting을 통해 모델의 일반화 및 추론 능력을 측정했습니다. 또한 'fairness-aware prompts'라는 맞춤형 프롬프트를 제안하여 편향을 완화했습니다.

- **Performance Highlights**: GPT-4가 성능과 공정성의 균형 면에서 가장 뛰어나지만, 특정 작업에서는 도메인 특화 모델인 MentalRoBERTa에 뒤떨어집니다. Few-shot CoT prompting이 성능과 공정성을 모두 향상시키는 것으로 나타났으며, 크기가 큰 LLM이 적은 편향을 보이는 경향이 있음을 발견했습니다. 이는 모델의 규모가 커질수록 다양한 인구 집단에 대한 복잡한 패턴을 더 잘 학습하고 표현할 수 있기 때문일 수 있습니다.



### LiLiuM: eBay's Large Language Models for e-commerc (https://arxiv.org/abs/2406.12023)
- **What's New**: eBay는 전자상거래 도메인의 특정 요구에 맞춘 대형 언어 모델(LLM) 시리즈인 LiLiuM을 소개했습니다. LiLiuM 시리즈는 1B, 7B, 13B 파라미터 모델로 구성되어 있으며, 100% 사내에서 개발되었습니다. 이를 통해 eBay는 라이센스, 데이터, 어휘 및 아키텍처의 모든 측면을 완벽히 제어할 수 있습니다.

- **Technical Details**: LiLiuM 모델들은 3조 개의 멀티랭귀지 텍스트 토큰을 기반으로 학습되었으며, 전자상거래와 일반 분야의 데이터를 포함합니다. NVIDIA의 Megatron-LM을 기반으로 커스터마이징하여 데이터 병렬 처리(DP), 텐서 병렬 처리(TP), 파이프라인 병렬 처리(PP), 및 분산 최적화 상태를 지원합니다. 모델 아키텍처는 데코더 전용 트랜스포머를 채택하고, 컨텍스트 크기는 4096 토큰으로 설정되어 있습니다.

- **Performance Highlights**: LiLiuM 모델들은 영어 자연어 이해(NLU) 벤치마크에서 LLaMA-2 모델과 비슷한 성능을 보이면서도, 비영어권 NLU 과제, 기계 번역(MT), 전자상거래 분야에서 LLaMA-2 모델보다 뛰어난 성능을 발휘합니다. 맞춤형 어휘를 통해 eBay의 특화된 다운스트림 작업에서 최대 34%의 속도 향상을 이루었습니다.



### CItruS: Chunked Instruction-aware State Eviction for Long Sequence Modeling (https://arxiv.org/abs/2406.12018)
Comments:
          Work in progress

- **What's New**: 최근의 대형 언어 모델(LLMs)의 발전과 함께 긴 시퀀스 모델링에 대한 관심이 높아지고 있습니다. 새로운 연구에 따르면 Transformer 모델의 키-값 캐시(key-value cache)에서 많은 부분이 퍼플렉시티 성능에 영향을 미치지 않고도 폐기될 수 있음이 밝혀졌습니다. 그러나, 이러한 방법들은 다운스트림 작업에 중요한 정보를 종종 놓치는 '정보 무시 문제(information neglect)'를 발생시킵니다. 이를 해결하기 위해 새로운 기법인 Chunked Instruction-aware State Eviction (CItruS)을 도입했습니다.

- **Technical Details**: CItruS는 긴 시퀀스 처리를 언어 모델링 프로세스와 작업 해결 프로세스로 나눕니다. 언어 모델링 프로세스에서는 큰 시퀀스를 큰 청크(chunks)로 나누어 필요 없는 키-값 상태를 캐시에서 제거합니다. 작업 해결 프로세스에서는 명령어 인식 캐시(instruction-aware cache)를 사용하여 다운스트림 작업에 필요한 구체적인 정보를 유지합니다. 특히, 이 방법은 훈련이 필요하지 않으며 기존 Transformer 기반 디코더 모델에 바로 적용할 수 있는 장점을 지닙니다.

- **Performance Highlights**: CItruS는 긴 문서 이해, 지식 검색, 언어 모델링 과제에서 매우 큰 성능 향상을 보였습니다. 특히, 최대 백만 개의 토큰을 포함하는 긴 문서에서도 필요한 정보를 성공적으로 추출하며, 언어 모델링 퍼플렉시티 역시 낮은 수치를 유지했습니다. 이 방법은 추가 훈련 없이도 긴 시퀀스 입력에 대해 다운스트림 작업 성능을 크게 향상시켰습니다.



### FinTruthQA: A Benchmark Dataset for Evaluating the Quality of Financial Information Disclosur (https://arxiv.org/abs/2406.12009)
- **What's New**: 이 논문은 중국 증권거래소의 투자자 인터랙티브 플랫폼에서 투자자들이 기업에 질문하고 답변을 받는 Q&A 포맷을 통해 정보의 질을 자동 평가하는 벤치마크인 FinTruthQA를 소개합니다. 이 데이터셋은 6,000개의 실제 금융 Q&A 항목으로 구성되어 있으며, 각 Q&A는 회계의 네 가지 개념적 차원에 따라 수동으로 주석이 달려 있습니다.

- **Technical Details**: FinTruthQA는 통계적 머신 러닝 모델(statistical machine learning models), 사전 학습된 언어 모델(pre-trained language model) 및 그들의 미세 조정(fine-tuned) 버전, 대형 언어 모델 GPT-4를 포함한 다양한 NLP 기술을 벤치마크했습니다. 주요 평가 기준은 질문 식별(question identification), 질문 관련성(question relevance), 답변 관련성(answer relevance), 답변 가독성(answer readability)입니다.

- **Performance Highlights**: 실험 결과, 기존의 NLP 모델들은 실제 질문 식별 및 질문 관련성 작업에서 강력한 예측 능력을 보여주지만, 답변 관련성 및 가독성 작업에서는 최적화되지 않은 것으로 나타났습니다. 이 벤치마크를 통해 정보 공개의 자동 평가에 대해 견고한 기반을 제공하며, 실시간 모니터링 및 데이터 기반 의사 결정을 크게 향상시킬 수 있습니다.



### Dialogue Action Tokens: Steering Language Models in Goal-Directed Dialogue with a Multi-Turn Planner (https://arxiv.org/abs/2406.11978)
Comments:
          Code: this https URL

- **What's New**: 이번 논문에서는 Dialogue Action Tokens(DAT)라는 새로운 접근 방식을 소개합니다. 이 방법은 언어 모델 에이전트가 목표 지향적인 대화를 계획할 수 있도록 합니다. 각 발화를 하나의 행동으로 간주하여, 대화를 게임으로 변환하고 강화 학습(reinforcement learning) 기법을 적용할 수 있습니다. 특히, 사전 학습된 언어 모델을 고정하고 소규모 계획 모델을 훈련하여 매 라운드마다 연속적인 행동 벡터를 예측하도록 합니다. 이렇게 하면 보상 최적화에 따른 언어 퇴화 문제를 피할 수 있습니다. DAT를 Sotopia 플랫폼에서 평가한 결과, DAT가 제어하는 LLaMA 모델이 GPT-4를 능가하는 성능을 보였습니다. 또한, 다중 턴 레드 팀 세팅에서 DAT를 사용하여 공격자 언어 모델을 제어한 결과, 새로운 공격 면을 밝혀냈습니다.

- **Technical Details**: DAT 기법에서 각 발화를 행동으로 간주하여 이를 통해 다중 턴 대화의 목표 지향적 계획을 수행합니다. 강화 학습(RL) 기법을 활용해 언어 모델을 향상시키며, 언어 모델 파라미터는 고정한 상태로 소규모 계획 모델을 훈련합니다. 이 모델은 각 발화에서 모델 행동을 제어하기 위해 몇 개의 접두사 토큰을 예측합니다. 이를 통해 RL 훈련이 언어 모델에 미치는 영향을 제한하고 다중 턴 계획을 통합할 수 있습니다. DAT 프레임워크는 언어 공간의 계획 문제를 저차원 연속 제어 문제로 변환하여 기존의 RL 기법과 친숙하게 만듭니다.

- **Performance Highlights**: Sotopia 플랫폼에서의 실험에서, 협상, 설득, 협업 시나리오를 포함한 다양한 시나리오에서 중요한 향상을 보였습니다. TD3+BC 알고리즘을 사용해 계획 모델을 훈련하여, GPT-4의 사회적 역량 점수를 능가했습니다. 두 번째 실험에서는 다중 턴 레드 팀 설정에서 공격자 언어 모델을 제어하여 높은 성공률을 달성해, 다중 라운드 대화에서 기존 언어 모델의 잠재적 안전 취약성을 밝혔습니다.



### Reframing linguistic bootstrapping as joint inference using visually-grounded grammar induction models (https://arxiv.org/abs/2406.11977)
- **What's New**: 이번 연구에서는 기존의 언어 습득 이론, 즉 의미적 부트스트래핑(semantic bootstrapping)과 구문적 부트스트래핑(syntactic bootstrapping)을 통합적으로 설명하는 새로운 접근 방법을 제안합니다. 저자들은 어린이가 서로 다른 언어적 도메인에서의 사전 지식을 사용하여 새로운 의미를 습득할 수 있도록 도와주는 기존 이론들이 실은 동시 학습(joint learning)에 기반한 일반적인 학습 전략에 의존하고 있다고 주장합니다.

- **Technical Details**: 연구팀은 비주얼-기반 문법 유도 모델(neural visually-grounded grammar induction models)을 사용하여 구문과 의미가 동시에 학습될 때 가장 강력한 부트스트래핑 효과가 나타난다는 점을 입증했습니다. 이러한 동시 학습은 문법 유도(grammar induction), 현실적인 어휘 범주 학습(lexical category learning), 새로운 문장과 동사의 의미 해석에서 뛰어난 결과를 나타냈습니다. 동시 학습은 구문과 의미의 가설 공간을 상호 제약하여 언어 습득을 용이하게 만듭니다.

- **Performance Highlights**: 논문에서는 동시 학습(joint learning)이 독립적인 구문학습이나 의미학습보다 더 나은 문법 유도 성과와 현실적인 어휘 범주 학습, 새로운 문장 및 동사의 의미 해석을 가능하게 한다고 보고하고 있습니다. 이는 구문과 의미의 가설 공간을 상호 제약하여 학습을 더욱 효율적으로 만듭니다.



### Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (https://arxiv.org/abs/2406.12845)
Comments:
          Technical report v1. Code and model are released at this https URL

- **What's New**: 향후 LLM(대형 언어 모델, Large Language Models)을 인간의 선호도에 맞게 조정하기 위해 강화학습(강화 학습, Reinforcement Learning)의 새로운 접근 방식을 제안했습니다. 기존의 보상 모델(RM)이 블랙박스 모델로 인간이 이해하기 어려운 반면, 새로운 방법은 해석 가능성을 높여서 인간의 선호도를 더 잘 반영할 수 있도록 합니다.

- **Technical Details**: 이 논문에서는 두 단계 접근법을 제안합니다. 첫 번째 단계에서 '절대 평가 다목적 보상 모델'(Absolute-Rating Multi-Objective Reward Model, ArmoRM)을 멀티-디멘션 데이터로 학습합니다. 여기서 각 디멘션은 '정직성(honesty)', '장황함(verbosity)', '안전성(safety)'과 같은 인간이 이해할 수 있는 목표(objective)를 반영합니다. 두 번째 단계에서는 '전문가 혼합 전략'(Mixture-of-Experts, MoE)을 사용해 컨텍스트에 맞는 보상 목표를 자동으로 선택하는 게이팅 네트워크(gating network)를 사용합니다.

- **Performance Highlights**: 최적화된 ArmoRM을 Llama-3 8B 모델과 얕은 MLP 기반의 게이팅 네트워크로 학습한 결과, 최첨단 성능을 달성했습니다. 특히, RewardBench 벤치마크에서 GPT-4 심판(GPT-4 judges)를 사용한 LLM-as-a-judge 방법보다 우수한 성능을 보였으며, Nemotron-4 340B 보상 모델과 동등한 수준에 근접했습니다.



### Adversarial Attacks on Multimodal Agents (https://arxiv.org/abs/2406.12814)
Comments:
          19 pages

- **What's New**: 이번 논문에서는 비전이 가능한 언어 모델(VLMs)을 사용해 실제 환경에서 동작하는 자율적 다중 모달 에이전트를 구축하는 과정에서 발생하는 새로운 보안 위험에 대해 다루고 있습니다. 논문은 제한적인 환경 접근과 지식에도 불구하고 공격이 가능함을 보여줍니다.

- **Technical Details**: 공격 방법은 적대적인 텍스트 문자열을 사용하여 환경 내의 하나의 트리거 이미지에 대해 그래디언트 기반 교란을 유도하는 것입니다. 여기에는 두 가지 공격 유형이 포함됩니다: (1) 캡셔너 공격(captioner attack): 화이트박스 captioner를 사용해 이미지를 캡션으로 변환하는 경우, 이를 이용해 VLM에 추가 입력으로 사용되는 캡션을 조작함. (2) CLIP 공격(CLIP attack): 여러 CLIP 모델을 동시에 공격하여, 이 단독 공격을 통해 소유자의 VLM에 이식 가능.

- **Performance Highlights**: VisualWebArena-Adv라는 새로운 적대적 과업 세트를 통해 공격을 평가했습니다. L-infinity 노름이 단일 이미지 당 $16/256$인 최대 픽셀 이동 내에서 캡셔너 공격은 캡셔너가 추가된 GPT-4V 에이전트가 적대적 목표를 75% 성공률로 실행하게 할 수 있습니다. 캡셔너를 제거하거나 GPT-4V를 사용해 자체 캡션을 생성할 경우 CLIP 공격은 각각 21%와 43%의 성공률을 기록했습니다. 다른 VLM 기반 에이전트(예: Gemini-1.5, Claude-3, GPT-4o)에 대한 실험도 수행되었으며, 이는 각 에이전트의 강건성에 차이가 있음을 보여줍니다.



### Formatics & dairy industry coalition: AI trends and present challenges (https://arxiv.org/abs/2406.12770)
- **What's New**: 이번 연구는 인공지능(AI)이 산업 전반, 특히 유제품(디어리) 산업에서 어떤 식으로 혁신을 일으킬 수 있는지 탐구합니다. AI는 생산 과정을 최적화하고 수작업 반복 작업을 줄여줌으로써 산업에 긍정적인 영향을 미칠 수 있습니다. 이 논문은 특히 소 모니터링 및 농가의 필요를 충족시키기 위한 첨단 기술 솔루션을 제안합니다.

- **Technical Details**: 이 연구는 고성능 컴퓨팅(high-performance computing)과 강력한 수학 모델이 결합하여 머신러닝(machine learning)과 같은 복잡한 데이터 분석 절차를 어떻게 구현할 수 있는지에 초점을 맞추고 있습니다. 효율적이고 유연한 처리 방식에 대한 도전 과제를 해결하기 위해 다루어집니다.

- **Performance Highlights**: 이 연구의 결론은 연구자들에게 새로운 접근 방식을 적용하여 소 모니터링 효율성을 높이는 방법을 제시하고, 농부들에게는 고급 기술 솔루션을 활용하여 생산성과 관리를 개선하는 방안을 제공합니다. 이를 통해 유제품 산업에서 AI의 실용적인 활용 가능성을 높이는 데 기여할 것으로 기대됩니다.



### Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning (https://arxiv.org/abs/2406.12742)
Comments:
          First three authors contributed equally. Dataset: this https URL

- **What's New**: 새로운 Multi-Image Relational Benchmark (MIRB)을 제안하여, VLMs(Visual Language Models)가 여러 이미지 입력을 비교하고 분석하며 추론하는 능력을 평가합니다. 여기에는 인식, 시각적 세계 지식, 추론 및 다단계 추론의 네 가지 카테고리가 포함됩니다.

- **Technical Details**: MIRB는 VLMs의 다중 이미지 이해능력을 평가하기 위해 설계되었으며, 이는 현실 세계에서 여러 이미지를 비교하는 능력을 중요시합니다. 평가 항목에는 코드 이해, 플롯 코드 이해와 같은 실제 프로그래밍 작업들을 포함하여 다양한 이미지를 교차 비교하는 질문들이 포함됩니다. 모델들은 각 이미지에서 파생되는 솔루션을 도출하기 위해 여러 이미지 입력을 비교해야 합니다.

- **Performance Highlights**: MIRB를 통해 열린 소스(LLaVA 등)와 닫힌 소스(GPT-4V 등) 모델들을 평가한 결과, 열린 소스 VLMs는 단일 이미지 작업에서는 GPT-4V에 손색없는 성과를 보였지만, 다중 이미지 추론 작업에서는 여전히 큰 성능 격차가 있음을 확인했습니다. 심지어 최첨단 폐쇄형 GPT-4V 모델도 MIRB에서 높은 성과를 달성하는 데 어려움을 겪었으며, 이는 향후 연구와 개발이 필요한 부분임을 시사합니다.



### AGLA: Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention (https://arxiv.org/abs/2406.12718)
- **What's New**: 최근 다양한 멀티모달 작업에서 큰 성공을 거둔 대규모 비전-언어 모델(Large Vision-Language Models, LVLMs)이 객체 환각(object hallucination) 문제를 겪고 있다는 점이 부각되었습니다. 본 연구는 이러한 객체 환각의 주요 원인 중 하나로 주의 결핍(attention deficiency)을 지목하고, 이를 해결하기 위해 글로벌 이미지 특징과 로컬 이미지 특징을 동시에 활용하는 AGLA(Assembly of Global and Local Attention) 방법을 제안합니다.

- **Technical Details**: LVLMs는 주로 전역 이미지 특징(global image features)에 주의를 기울이는 반면, 텍스트 프롬프트(prompt)와 관련된 로컬 이미지 특징(local features)을 놓치는 경향이 있습니다. 이를 해결하기 위해, AGLA는 훈련 없이 적용 가능한 방식으로 전역 이미지 특징을 생성적(genarative)으로, 로컬 이미지 특징을 판별적(discriminative)으로 동시에 활용합니다. 이미지-프롬프트 매칭 스킴(image-prompt matching scheme)을 통해 프롬프트와 관련된 로컬 특징을 도출하고 기존 이미지의 증강된 뷰를 생성하여, 완성된 디코딩 분포(calibrated decoding distribution)를 도출합니다.

- **Performance Highlights**: 다양한 생성 및 판별 벤치마크에서 광범위한 실험을 통해 AGLA는 객체 환각을 일관되게 완화하고 LVLMs의 일반적인 인식 능력을 향상시킨다는 것을 입증했습니다. 연구진은 이러한 결론을 바탕으로 코드 공개를 향후 진행할 예정입니다.



### Transforming Surgical Interventions with Embodied Intelligence for Ultrasound Robotics (https://arxiv.org/abs/2406.12651)
Comments:
          This work has been accepted by MICCAI 2024

- **What's New**: 초음파 진단 기술은 비침습적 진단 방법론을 혁신하여 다양한 의료 분야에서 환자 치료 결과를 향상시키고 있습니다. 본 논문은 대형 언어 모델(LLMs)과 도메인 특화 지식 보강을 결합하여 초음파 로봇의 지능과 운영 효율성을 향상시키는 새로운 초음파 구현 지능 시스템(Ultrasound Embodied Intelligence System)을 소개합니다. 이 시스템은 LLMs를 초음파 로봇과 통합하여 의사들의 언어 명령을 정밀한 모션 계획으로 해석하고, 환자 움직임이나 절차 오류를 기반으로 실시간으로 스캔 계획을 조정할 수 있는 동적 실행 메커니즘을 통합합니다.

- **Technical Details**: 우리의 접근 방식은 두 가지 전략을 사용합니다. 첫 번째로, LLMs와 초음파 로봇을 통합하여 의사의 언어 명령을 이해하고 초음파 도메인 지식(API 및 운영 설명서 등)을 종합적으로 활용하여 정확한 모션 계획을 수립합니다. 두 번째로, 동적 실행 메커니즘을 도입하여 환자 움직임이나 절차 오류를 기반으로 실시간으로 스캔 계획을 조정합니다. 본 연구에서는 ReAct 프레임워크에서 영감을 얻어 생각-행동-관찰 주기를 통한 동적 실행 메커니즘을 개발하였으며, 로봇의 API와 지속적으로 상호작용하여 명령을 원활하게 실행합니다.

- **Performance Highlights**: 우리 시스템의 효과를 입증하기 위해 광범위한 실험을 수행하였으며, 다양한 모델에 대한 비교 연구와 ablation 실험을 통해 언어 명령을 기반으로 의료 절차를 실행하는 데 있어 현저한 성능 향상을 입증했습니다. 실험 결과, 본 시스템이 초음파 스캔의 효율성과 품질을 크게 향상시키며, 비침습적 진단과 의료 워크플로우를 혁신적으로 개선할 가능성이 있음을 확인하였습니다.



### Rapid Language Adaptation for Multilingual E2E Speech Recognition Using Encoder Prompting (https://arxiv.org/abs/2406.12611)
Comments:
          Accepted by INTERSPEECH 2024

- **What's New**: 최근 다언어 음성 인식 모델이 주목받고 있습니다. 이 모델은 단일 모델 내에서 다수의 언어를 처리하며, 일반적으로 언어 식별 기능을 포함하여 들어오는 음성의 언어를 자동으로 감지합니다. CTC(연결주의 한시적 분류, Connectionist Temporal Classification) 접근법을 강화하여 언어별 적응을 도입한 기술이 제안되었습니다. 이 기술은 저자원 언어에서 평균적으로 28%, 저자원 언어에서 41%의 오류율 감소를 달성했습니다.

- **Technical Details**: 제안된 방법은 중간 레이어에서 CTC 손실을 계산하고, 다음 인코더 레이어의 입력에 중간 예측값을 추가하는 Self-Conditioned CTC(SC-CTC) 프레임워크를 사용합니다. 이를 통해 언어 ID를 프롬프트로 제공하여 인코더를 제어하는 방법을 제안했습니다. 이 방법을 통해 사전 훈련된 모델을 신속하게 적응시키고, 변형된 토큰 시퀀스가 인코더 출력에 영향을 미쳐 주의 기반 디코더와 CTC 모두에 반영됩니다.

- **Performance Highlights**: 실험 결과, Common Voice 데이터셋에서 평균 오류율이 28% 감소했으며, 5시간 이하의 훈련 데이터를 가진 저자원 언어에서는 오류율이 41% 감소했습니다. 이 방법은 Common Voice, VoxForge 및 FLEURS 코퍼스를 기반으로 한 실험에서 그 효과를 확인했습니다.



### PromptDSI: Prompt-based Rehearsal-free Instance-wise Incremental Learning for Document Retrieva (https://arxiv.org/abs/2406.12593)
Comments:
          21 pages

- **What's New**: PromptDSI는 기존 Differentiable Search Index(DSI) 모델의 한계를 극복하기 위해 제안된 새로운 프롬프트 기반 방법으로, 문서 검색 시 리허설 없는 인스턴스-단위 증강 학습을 달성합니다. PromptDSI는 변화하는 코퍼스에 대해 새로운 문서를 효율적으로 인덱싱하면서 안정성과 가변성을 유지하도록 설계되었습니다.

- **Technical Details**: PromptDSI는 DSI의 프로즌 pre-trained language model(PLM)에 프롬프트를 첨부하여 인덱싱 효율을 높입니다. 고정된 네트워크 내 메커니즘을 사용하여 질의-키 매칭을 줄임으로써 첫 번째 포워드 패스의 비효율성을 제거합니다. 또한, 초기 코퍼스에서 추출된 neural topic embeddings를 프롬프트 키로 사용하여 다채로운 프롬프트 사용을 보장합니다. 이는 매칭 메커니즘의 붕괴로 인해 발생하는 파라미터 낭비 문제를 해결합니다.

- **Performance Highlights**: PromptDSI는 IncDSI와 동등한 성능을 보이면서도 새로운 코퍼스에서 리콜율을 4% 이상 향상시키는 효과를 증명했습니다. 전체적인 성능 평가에서 새로운 코퍼스에 대한 적응력과 잔존 학습 정보의 보존 능력에서 두드러지는 성과를 보여주었습니다.



### Performant ASR Models for Medical Entities in Accented Speech (https://arxiv.org/abs/2406.12387)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 최근 자동 음성 인식(ASR)의 발전이 의료 분야에서도 큰 역할을 하고 있습니다. 그러나 아프리카 억양을 지닌 의료 명명 엔티티(Named Entities, NE)에서 ASR의 성능에 대한 연구는 부족했습니다. 이 논문은 93개의 아프리카 억양을 가진 임상 영어 데이터셋을 사용해 여러 ASR 모델을 면밀히 평가하였습니다.

- **Technical Details**: 이번 연구에서는 AfriSpeech-200이라는 데이터를 이용하여 19개의 오픈 소스 및 상용 ASR 시스템을 평가했습니다. 데이터는 120개의 억양과 2,300명의 화자를 포함하며, 본 연구는 임상 도메인 테스트 서브셋에 집중했습니다. 상업용 의료 NER 모델인 Amazon Comprehend Medical을 사용하여 의료 명명 엔티티를 추출하고, 새로운 퍼지(Fuzzy) 문자열 매칭 알고리즘을 개발해 ASR 예측 텍스트와의 정렬을 개선했습니다.

- **Performance Highlights**: 실험 결과, 억양에 맞춰 감독 학습을 통해 ASR 모델의 의료 WER이 25-34% 향상되는 것을 확인했습니다. 이는 비록 일반 스피치에서 낮은 WER을 기록한 모델들이라도, 임상 명명 엔티티에서의 오류율이 높다는 점에서 중요한 의미를 가집니다. 새로운 평가 지표로 의료 명명 엔티티 리콜(Recall), 의료 WER(M-WER), 의료 문자 오류율(M-CER)을 도입하며, 이러한 개선이 실제 의료 환경에서 ASR의 실용성을 높였습니다.



### Automatic benchmarking of large multimodal models via iterative experiment programming (https://arxiv.org/abs/2406.12321)
Comments:
          31 pages, 6 figures, code is available at this https URL

- **What's New**: 새로운 논문에서는 대형 멀티모달 모델들(LMMs)의 평가를 자동화하는 최초의 프레임워크인 APEx를 소개합니다. APEx는 자연어로 표현된 연구 질문을 기반으로, 대형 언어 모델(LLM)과 사전에 지정된 도구 라이브러리를 활용하여 자동으로 실험을 생성하고 과학 보고서를 작성합니다.

- **Technical Details**: APEx 프레임워크는 모듈 구조 덕분에 유연하고 확장 가능하며, 새로운 도구가 추가될 수록 더욱 향상됩니다. 대형 언어 모델(LLM)을 사용하여 주어진 연구 질문에 맞는 실험을 점진적으로 컴파일하고, 현재 진행 상태에 따라 어떤 실험을 수행할지 결정합니다. 실험 결과가 충분할 경우 결론을 도출하며, LLM은 결과를 자연어로 사용자가 이해할 수 있게 보고서를 최종 정리합니다.

- **Performance Highlights**: APEx는 기존 연구의 결과를 재현하면서도 임의의 분석과 가설 검증을 가능하게 합니다. 이는 평가 과정의 수고를 크게 줄여주며, 새로운 벤치마크를 만들 때 요구되는 방대한 수작업을 덜어줍니다.



### CodeNav: Beyond tool-use to using real-world codebases with LLM agents (https://arxiv.org/abs/2406.12276)
- **What's New**: CodeNav는 새로운 LLM(대형 언어 모델) 에이전트로, 등록된 도구나 매뉴얼 설명 없이도 미리 보지 않은 코드 저장소를 탐색하며 사용자 쿼리를 해결할 수 있습니다. CodeNav는 자동으로 코드 블록을 인덱싱하고 검색해 관련 코드 스니펫을 찾아내고, 이를 가져와 반복적으로 솔루션을 생성합니다.

- **Technical Details**: CodeNav는 단일 에이전트 멀티 환경 상호작용 프레임워크를 제안합니다. 사용자가 쿼리를 입력하면, 코드를 검색하고 유용한 코드 스니펫을 이용해 솔루션을 구성하며, 실행 피드백을 점검합니다. 기존의 'tool-use' 접근법과 달리, 라이브러리나 도구 설명 없이도 코드베이스의 구조를 활용해 필요한 스니펫을 검색합니다. Elasticsearch 쿼리를 통해 소스 코드를 직접 검색합니다.

- **Performance Highlights**: 세 개의 기존 'tool-use' 벤치마크(m&m's, M3ToolEval, API-Bank)에서 CodeNav는 도구 등록이 필요 없이 'tool-use' 접근법과 유사한 성능을 보였습니다. 세 가지 다양한 코드베이스를 이용한 사례 연구에서도, 복잡한 다단계 쿼리를 해결하고, 실행 오류를 수정하고, iterative 검색을 통해 스니펫을 이해하는 등 뛰어난 성능을 입증했습니다.



### TroL: Traversal of Layers for Large Language and Vision Models (https://arxiv.org/abs/2406.12246)
Comments:
          Code is available in this https URL

- **What's New**: 새로운 효율적인 대규모 언어 및 비전 모델(LLVM, Large Language and Vision Model) 계열인 TroL(Tresal of Layers)을 소개합니다. TroL은 1.8B, 3.8B, 7B 모델 크기를 가지며, 이 계열은 레이어를 토큰 단위에서 재사용하는 트래버설(layer traversing) 기법을 통해 일반 레이어 수를 물리적으로 늘리지 않고도 성능을 최적화할 수 있습니다.

- **Technical Details**: 기존의 대규모 언어 및 비전 모델들이 대규모 파라미터(26B, 34B, 110B)를 사용하는 데 반해, TroL은 레이어 트래버설 기법을 통해 작은 모델 크기에서도 성능을 극대화합니다. TroL은 레이어와 토큰 단위의 믹싱(operation)을 효율적으로 조정하여 질문 스트림을 반복 탐색하고 회귀하는 것과 같은 효과를 시뮬레이션합니다. 트레이닝 과정은 비전 프로젝터와 TroL-Mixer를 트레이닝하고, 이는 다시 백본 멀티모달 LLM과 함께 추가로 트레이닝됩니다. 여기서 Q-LoRA(Dettmers et al., 2023)를 사용하여 트레이닝을 효율화합니다.

- **Performance Highlights**: TroL은 1.8B, 3.8B, 7B의 모델 크기임에도 불구하고, 26B, 34B, 72B, 110B 크기의 모델과 비교해도 성능이 우수합니다. 두 단계의 트레이닝 과정을 통해 TroL은 큰 모델 크기 없이도 개방형 및 폐쇄형 소스 LLVM을 능가하는 성능을 보여줍니다.



### SyncVSR: Data-Efficient Visual Speech Recognition with End-to-End Crossmodal Audio Token Synchronization (https://arxiv.org/abs/2406.12233)
- **What's New**: SyncVSR는 음성을 구별하기 어려운 비주얼 유사성(homophenes)을 해결하기 위해 시각적 표현과 음향 데이터를 동기화하는 새로운 딥러닝 프레임워크입니다. 프레임 수준의 크로스모달 교정을 위해 정량화된 오디오를 활용하여 비디오 시퀀스로부터 비자동 회귀 방식으로 이산 오디오 토큰을 생성하는 인코더를 학습시킵니다. 이 방법은 다양한 과업, 언어 및 모달리티에서 우수한 성능을 보여주며 데이터 사용량을 최대 9배까지 줄일 수 있습니다.

- **Technical Details**: SyncVSR 프레임워크는 비주얼 및 오디오의 크로스모달(교차 모달 간의) 동기화를 통해 일관된 학습을 제공합니다. 오디오 재구성 손실(audio reconstruction loss)과 VSR 학습 목표를 통합하고, 단어 분류 손실과 문장 수준의 CTC-Attention 손실을 사용하여 최종 손실을 계산합니다. 각 비디오 프레임과 정량화된 오디오 토큰을 동기화하는 방법으로 비디오 및 오디오의 샘플링 속도를 일치시킵니다. 총 손실은 작업 전용 손실과 오디오 재구성 손실의 가중 합입니다.

- **Performance Highlights**: SyncVSR는 영어와 중국어 단어 수준 VSR 작업을 평가하기 위해 LRW 및 CAS-VSR-W1K 데이터셋을 사용했으며, 문장 수준의 실험은 LRS2와 LRS3 데이터셋을 기반으로 테스트했습니다. 결과적으로 SyncVSR는 새로운 벤치마크를 세우며, 이전 방법보다 적은 데이터로 더 높은 성능을 달성했습니다.



### "You Gotta be a Doctor, Lin": An Investigation of Name-Based Bias of Large Language Models in Employment Recommendations (https://arxiv.org/abs/2406.12232)
Comments:
          preprint, 18 pages

- **What's New**: 최근 발표된 논문에서는 GPT-3.5-Turbo 및 Llama 3-70B-Instruct와 같은 대형 언어 모델(LLM)을 사용하여 320개의 인종과 성별을 나타내는 이름을 가진 후보자들에 대한 채용 결정 및 급여 추천을 시뮬레이션했습니다. 이 연구는 이러한 모델들이 특정 인구 통계학적 그룹에 대해 차별적인 경향을 보이는지를 조사합니다.

- **Technical Details**: 연구에서는 U.S 기반의 첫 이름이 인종과 성별을 나타내는 320명의 후보자를 대상으로, 40가지 직업군에 걸쳐 약 750,000개의 프롬프트를 통해 실험을 진행했습니다. LLM들이 채용 결정 및 급여 추천을 어떻게 내리는지를 분석하기 위해 GPT-3.5-Turbo와 Llama 3-70B-Instruct가 사용되었습니다. 프롬프트 구성에는 후보자의 경력 및 교육 수준이 동등한 조건 하에 이름만 제공된 경우와 전체 프로필이 제공된 경우가 포함되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-3.5-Turbo와 Llama 3 모델 모두 백인 여성 이름의 후보자를 다른 인구 통계학적 그룹보다 선호하는 경향을 보였습니다. 특히 동일한 자격을 가진 후보자 간에도 급여 추천에서 최대 5%의 차이가 발생했습니다. 실제 미국 노동 시장 데이터와 비교했을 때, 이러한 모델의 편향성은 현실과 일치하지 않는 부분이 드러났습니다. 예를 들어, 특정 소수 인종 그룹은 우대되는 반면 다른 그룹은 불이익을 받는 경향이 있었습니다.



### Interface Design for Self-Supervised Speech Models (https://arxiv.org/abs/2406.12209)
Comments:
          Accepted to Interspeech2024

- **What's New**: 자기 지도 학습(Self-Supervised Learning, SSL) 음성 모델을 활용하는 새로운 인터페이스를 제안했습니다. 이 논문에서는 기존에 레이어별 가중 합(weighted sum) 방법이 최적이 아님을 밝히며, 여러 대안적인 인터페이스 디자인을 소개하고 있습니다. 특히 계층적 컨볼루션 계층(Hierarchical Convolution)이 다수의 음성 처리 작업에서 더 나은 성능을 보인다는 점을 입증했습니다.

- **Technical Details**: SSL 모델을 활용한 프레임워크를 업스트림 모델(Upstream Model), 다운스트림 예측 헤드(Downstream Prediction Head), 그리고 이 둘을 연결해주는 인터페이스(Interface)로 구성하는 것을 제안합니다. 기존의 가중 합(weighted sum) 방식은 레이어별로 서로 독립적인 정보를 단순 합산하여 정보 손실이 발생할 수 있으며, 이 문제를 해결하기 위해 다양한 대체 인터페이스 디자인을 제안하고 실험했습니다. 제안된 인터페이스 중에는 각 레이어의 정보를 다차원적으로 결합하여 활용하는 방법도 포함되어 있습니다.

- **Performance Highlights**: 다양한 SSL 모델과 다운스트림 작업에 대해 제안된 인터페이스 디자인을 SUPERB와 ML-SUPERB 벤치마크에서 평가한 결과, 계층적 컨볼루션 인터페이스가 전반적으로 가장 우수한 성능을 보였습니다. 또한, 이러한 성능 차이가 단순히 훈련 가능한 매개변수의 수가 많아서가 아니라 각 인터페이스 구조 자체의 특성 때문이라는 점을 추가 실험을 통해 확인했습니다.



### BPO: Supercharging Online Preference Learning by Adhering to the Proximity of Behavior LLM (https://arxiv.org/abs/2406.12168)
Comments:
          Preprint. In submission

- **What's New**: Direct alignment from preferences (DAP) 방법론이 큰 언어 모델 (LLMs)을 인간의 선호에 맞추기 위한 유망한 패러다임으로 등장했습니다. 기존의 오프라인 DAP 방법이 온라인으로 얻은 훈련 샘플을 통해 이익을 얻을 수 있다는 연구 결과가 있었으나, 우리는 온라인 훈련의 잠재력을 최대한 활용하기 위해 특정 온라인 DAP 알고리즘을 개발할 필요가 있음을 강조합니다. 이를 위해, 우리는 학습된 LLM이 훈련 샘플을 수집하는 행동 LLM의 근접성을 준수해야 한다고 식별하고, 적절한 신뢰 영역을 구축하는 것이 중요하다고 제안합니다.

- **Technical Details**: 온라인 Preference Optimization in proximity to the Behavior LLM (BPO)을 제안하며, 적절한 신뢰 영역을 행동 LLM 주위에 구축하는 것이 중요하다고 강조합니다. 동적 모델로부터 훈련 샘플을 얻는 동안, 우리는 기존 오프라인 DAP 방법에 사용된 고정 참조 모델 대신 동적 참조 모델을 사용하는 것이 필요하다고 주장합니다. 이를 구현하기 위해, 우리는 LoRA 가중치를 최적화하고 추론 중에 합병하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 online BPO는 다양한 DAP 방법과 통합하여 TL;DR에서 72.0%에서 80.2%로, Anthropic Helpfulness에서 82.2%에서 89.1%로 성능을 향상시켰습니다. 이는 추가 데이터 수집 단계를 도입하지 않으면서 이루어진 결과입니다. 우리는 다양한 주석 주기에서 메서드를 평가하였고, 최소한의 주석 주기로 일정한 주석 데이터를 유지하면서도 큰 성능 향상을 보였습니다.



### Bias in Text Embedding Models (https://arxiv.org/abs/2406.12138)
- **What's New**: 이 논문은 현대 텍스트 임베딩 모델에서 성별 편향(gender bias)을 분석한 연구입니다. 인기 있는 여러 텍스트 임베딩 모델이 직업과 성별 용어를 어떻게 연결 짓는지 조사하고, 이러한 모델들이 다양한 방식으로 성별 편향을 나타낼 수 있음을 발견했습니다.

- **Technical Details**: 분석 대상 모델은 AI21-v1-embed, amazon-titan-embed-text-v1, baai-bge-large-zh-v1.5, cohere-embed-english-v3.0, bert-large, llama-2-70b, msmarco-distilbert-cos-v5, openai-text-embedding-ada002, voyageai-voyage-01 등입니다. 연구에서는 특정 직업(예: 간호사, CEO, 작가)과 성별 용어(예: 소년, 소녀, 여성, 남성) 간의 연관성을 조사합니다.

- **Performance Highlights**: 각 텍스트 임베딩 모델은 특정 직업을 성별 용어와 연결 짓는 경향이 있으며, 이는 모델마다 변동될 수 있습니다. 예를 들어, 간호사, 가정주부 등은 여성과, CEO, 관리자 등은 남성과 더 강하게 연관되는 경향이 있습니다. 여러 모델 간 공통점이 존재하지만, 모든 모델이 동일한 유형의 성별 연관성을 가지지는 않습니다. 특정 직업과 성별 용어의 연관 강도는 모델에 따라 크게 다를 수 있습니다.



### Efficient Sequential Decision Making with Large Language Models (https://arxiv.org/abs/2406.12125)
- **What's New**: 이 논문은 기존 대형 언어 모델들(LLMs)을 순차적 의사결정(sequential decision making)에 효율적으로 적용하는 새로운 접근법을 제안합니다. 기존 노력은 LLMs를 재훈련하거나(finetuning) 프롬프트 엔지니어링(prompt engineering)을 통해 LLMs를 적용하려는 것에 집중했지만, 각각 큰 계산 부담과 성능의 한계를 가졌습니다. 본 연구에서는 온라인 모델 선택 알고리즘(online model selection algorithms)을 활용하여, LLMs 에이전트를 순차적 의사결정 문제에 효율적으로 통합하는 방법을 소개합니다.

- **Technical Details**: 제안된 접근법은 LLMs의 재훈련이 필요하지 않으며, 의사결정 과정에서 LLMs 호출 횟수를 최소화합니다. 특히, 아마존 데이터셋 실험에서는 시간 단계의 1.5%만 LLMs를 호출하면서도 기존 방법보다 6배 이상의 성능 향상을 보였습니다. 본 연구에서는 순차적 의사결정 문제의 컨텍스트 밴딧(contextual bandits) 설정을 다루고 있으며, 학습자가 주어진 컨텍스트에 대해 액션을 선택하고 피드백을 받는 과정을 반복합니다. 이를 통해, 초반에는 LLMs를 주로 사용하고, 시간이 지남에 따라 표준 의사결정 알고리즘을 점진적으로 더 많이 사용하는 자동화된 프레임워크를 제안합니다.

- **Performance Highlights**: 본 연구에서 제안된 프레임워크는 기존의 LLM 기반 에이전트와 표준 순차적 의사결정 알고리즘을 모두 능가하는 탁월한 성능을 보입니다. 실험에서 제안된 방식은 베이스라인 대비 약 6666배 이상의 성능 향상을 달성했으며, LLM 호출 횟수를 전체 시간 단계의 1.5%로 제한했습니다. 또한, 우리의 프레임워크는 기존의 대형 언어 모델뿐만 아니라 8000만 파라미터 수준의 작은 모델도 효율적으로 활용할 수 있는 높은 유연성을 제공합니다.



### Is poisoning a real threat to LLM alignment? Maybe more so than you think (https://arxiv.org/abs/2406.12091)
- **What's New**: 최근 강화학습과 인간 피드백(RLHF, Reinforcement Learning with Human Feedback)의 발전이 대형 언어 모델(LLMs, Large Language Models)의 정렬에 있어 중요한 영향을 미쳤습니다. 본 연구에서는 RLHF 방법 중 하나인 직접 정책 최적화(DPO, Direct Policy Optimization)의 취약성을 분석하고, 새로운 공격 형태인 선호도 변조(preference poisoning) 공격의 효과를 비교 분석합니다.

- **Technical Details**: 전통적인 RLHF 파이프라인은 보상 함수를 학습하고 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 모델을 업데이트합니다. 하지만 PPO 기반 방법들은 하이퍼파라미터에 민감하여 DPO는 이를 감독 학습 프레임워크로 처리합니다. 본 연구에서는 DPO의 취약점을 분석하고, 백도어(backdoor)와 비백도어(non-backdoor) 공격에서의 취약점을 비교 분석합니다. LLama 7B, Mistral 7B, Gemma 7B 등의 다양한 언어 모델에 대해 데이터를 독성(poison)으로 침투시키는 영향력 포인트(Influence points) 기반 방법을 적용하였습니다.

- **Performance Highlights**: 백도어 공격의 경우, 이전 PPO 기반 방법이 최소 4%의 데이터를 독성으로 변조해야 해로운 행위를 유발할 수 있었던 반면, 본 연구에서는 데이터의 단 0.5%만 독성으로 변조해도 모델을 독성화하는 데 성공했습니다. 이로 인해 DPO의 실제 취약점을 효과적으로 이용할 수 있음을 밝혔습니다. 또한 제안된 DPO 점수 기반, 그라디언트 없는(gradient-free) 방법이 임의의 독성화보다 효율적으로 모델을 독성화함을 증명했습니다.



### WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions (https://arxiv.org/abs/2406.12058)
Comments:
          26 pages including reference section and appendix section, 8 figures, 16 tables

- **What's New**: 최신 연구는 언어 모델(LMs)이 정신 건강 애플리케이션에서의 신뢰성과 설명 가능성을 평가하는 새로운 방법을 제안했습니다. 이 연구는 멀티레이블 분류 기반 MultiWD와 WellXplain이라는 두 개의 데이터셋을 사용하여 LMs 및 LLMs의 주의 메커니즘을 전문가 레이블 설명과 비교 평가합니다.

- **Technical Details**: Halbert Dunn의 웰빙 이론을 기반으로 한 MultiWD와 WellXplain 데이터셋을 활용하여 11개의 다양한 LMs/LLMs(예: RoBERTa, MedAlpaca)를 평가했습니다. 연구에서는 SVD Rank, Attention-Overlap Score, Attention Maps 등의 평가 지표를 사용했습니다. 또한, 확률 지향 손실 함수(confidence-oriented loss function)를 재검토하여 예측 성능 저하를 확인했습니다.

- **Performance Highlights**: GPT-3.5/4는 RoBERTa보다 성능이 떨어졌고, MedAlpaca는 개선된 성능이나 설명을 제공하지 못했습니다. 모든 LMs/LLMs에서 주의 메커니즘과 설명의 일치는 낮았으며, 일부 모델은 주의 겹침(attention overlap) 점수가 0.0을 기록했습니다. 이 연구는 LMs/LLMs의 신뢰성과 설명 가능성을 중점적으로 향상시킬 필요가 있음을 강조합니다.



### $\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (https://arxiv.org/abs/2406.12045)
- **What's New**: 기존의 벤치마크(benchmark)는 인간 사용자와의 상호작용 또는 도메인별 규칙을 따르는 능력을 테스트하지 않습니다. 이러한 능력은 실제 응용 프로그램에서 언어 에이전트를 배포하는 데 있어 필수적입니다. 이를 해결하기 위해 $	au$-bench라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 도메인별 API 도구와 정책 가이드라인을 제공받는 언어 에이전트와 사용자(언어 모델에 의해 시뮬레이션된)의 동적 대화를 모방합니다.

- **Technical Details**: $	au$-bench는 대화 끝에 데이터베이스 상태를 주석으로 달린 목표 상태와 비교하여 평가하는 효율적이고 신뢰성 있는 평가 과정을 사용합니다. 또한, 에이전트의 여러 시도 중 신뢰성을 평가하기 위한 새로운 메트릭(metric)인 pass^k도 제안합니다. 이 메트릭은 에이전트의 일관성과 규칙 준수 능력을 측정하는 데 있어 중요한 도구로 사용됩니다.

- **Performance Highlights**: 실험 결과, 최신의 함수 호출 에이전트(function calling agents)(예: gpt-4o)조차도 전체 작업의 50% 미만에서 성공률을 보였으며, 일관성도 부족한 것으로 나타났습니다(pass^8 <25% in retail). 이 연구는 에이전트의 일관성 있고 신뢰할 수 있는 규칙 준수 능력을 향상시킬 방법이 필요하다는 점을 시사합니다.



### Large Scale Transfer Learning for Tabular Data via Language Modeling (https://arxiv.org/abs/2406.12031)
- **What's New**: TabuLa-8B라는 새로운 언어 모델을 이용한 표형 데이터 예측 모델이 발표되었습니다. 이 모델은 TabLib 코퍼스로부터 추출된 대규모 고품질 학습 데이터셋을 사용하여 초고해상도 데이터를 기반으로 학습되었습니다. TabuLa-8B는 기존의 상태-of-the-art 모델들보다 높은 정확도를 자랑하며, 제로샷(zero-shot) 예측 기능을 제공합니다.

- **Technical Details**: TabuLa-8B는 8억개의 파라미터를 가진 Llama 3-8B 대형 언어 모델(LLM)로, TabLib 코퍼스에서 필터링된 3.1백만 개의 표로 구성된 Tremendous TabLib Trawl(T4)라는 새로운 데이터셋으로 학습되었습니다. 이 데이터셋은 1.6B개의 행(row)과 80B개의 토큰(token)으로 이루어져 있습니다. 또한 새로운 행-인과적 표 마스킹(row-causal tabular masking, RCTM) 및 패킹(packing) 스키마를 사용한 것이 특징입니다.

- **Performance Highlights**: TabuLa-8B는 테스트 스위트의 329개의 데이터셋에서 제로샷 예측에서 무작위 추정보다 15 퍼센트 포인트(pp) 높은 정확도를 기록하였습니다. 몇 샷 설정(few-shot setting)에서도 TabuLa-8B는 XGBoost와 TabPFN 모델보다 5-15 pp 높은 정확도를 보였으며, 이들은 동일한 데이터 또는 TabuLa-8B보다 최대 16배 더 많은 데이터로 학습된 경우에도 우세했습니다.



### SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Mod (https://arxiv.org/abs/2406.12030)
- **What's New**: 새로운 Vision Language Models (VLMs)의 발전은 다중 모달 정보 이해에 큰 진전을 가져왔지만, 안전 정렬 (Safety Alignment)에는 여전히 도전 과제가 있습니다. 이를 해결하기 위해 SPA-VL이라는 대규모 데이터셋이 제안되었습니다. 이 데이터셋은 6개의 해로움 도메인, 13개의 카테고리, 53개의 하위 카테고리를 포함하며, 총 100,788개의 샘플을 담고 있습니다. 각 샘플은 질문, 이미지, 선택된 응답, 거부된 응답의 네 가지 요소로 구성됩니다.

- **Technical Details**: SPA-VL 데이터셋은 인간 피드백을 통한 강화 학습 (Reinforcement Learning from Human Feedback, RLHF)에 적합하도록 설계되었습니다. 다양한 모델(예: QwenVL, Gemini)에서 수집된 응답을 포함하며, 해로움 피드백과 유용성 피드백을 모두 수집하는 것을 목표로 합니다. 각 샘플에는 쉬운 질문, 어려운 질문, 어려운 진술의 세 가지 질문 유형이 포함되어 있으며, 각 질문에 대해 총 12개의 다른 모델 중 2개의 모델의 응답이 포함됩니다. 이렇게 다양한 응답과 질문을 통해 학습된 VLMs는 해로움과 유용성 두 가지 측면에서 모두 향상된 반응을 보입니다.

- **Performance Highlights**: SPA-VL 데이터셋으로 학습된 모델들은 해로움과 유용성 측면에서 현저한 성능 향상을 나타냈습니다. 특히, PPO(정책 경사 방법)와 DPO(결정 프로세스 최적화) 기술을 적용하여 기존의 최첨단 VLMs보다 뛰어난 성능을 보여줍니다. 다양한 응답과 질문 유형을 활용함으로써 데이터셋의 규모를 늘려 향상된 안전성과 성능을 달성할 수 있음을 확인했습니다.

- **Conclusion**: SPA-VL 데이터셋은 VLMs의 안전 정렬 연구에서 중요한 이정표를 나타내며, 해로움 없이 유용한 응답을 보장하도록 설계되었습니다. 이 데이터셋과 코드는 공개적으로 이용 가능합니다.



### Transcoders Find Interpretable LLM Feature Circuits (https://arxiv.org/abs/2406.11944)
Comments:
          28 pages, 6 figures, 3 tables, 2 algorithms. Under review

- **What's New**: 최근 기계적 해석 가능성(Mechanistic Interpretability) 분야에서 트랜스코더(transcoder)를 활용한 새로운 연구가 발표되었습니다. 이 연구는 MLP(Multi-Layer Perceptrons) 하위 레이어를 효과적으로 분석하기 위한 방법론을 제시하며, 트랜스코더를 통해 기존의 MLP 서브레이어를 보다 해석 가능한 형태로 변환하는데 중점을 둡니다.

- **Technical Details**: 트랜스코더는 넓은 ReLU 기반 MLP 서브레이어로, 원래의 좁은 MLP 서브레이어 출력을 충실하게 근사하도록 훈련됩니다. L1 정규화를 사용하여 활성화가 희박하게 유지되도록 합니다. 이를 통해 이전의 해석이 어려운 MLP 서브레이어를 해석 가능한 근사치로 대체합니다. 또한, 이 연구는 GPT2-small 모델의 'greater-than 회로' 분석을 포함하여 MLP 서브레이어를 통한 회로 분석을 수행하는 새로운 방법론을 도입합니다.

- **Performance Highlights**: 연구 결과, 트랜스코더는 희박성(sparsity), 충실성(faithfulness), 인간 해석 가능성(human-interpretability) 측면에서 Sparse Autoencoder(SAE)와 동등하거나 더 우수한 성능을 나타냈습니다. 특히, 트랜스코더는 입력에 의존하지 않는(input-invariant) 및 입력에 의존하는(input-dependent) 구성 요소로 회로를 깔끔하게 분해할 수 있음을 확인했습니다. 이는 특정 입력을 분석하지 않고도 특성을 이해할 수 있게 합니다.



### From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipelin (https://arxiv.org/abs/2406.11939)
- **What's New**: 언어 모델(LLM)의 빠른 진화는 더 어려운 벤치마크의 개발을 필요로 하고 있습니다. 기존의 정적인 벤치마크는 모델의 능력을 일관되게 구분하는 데 어려움을 겪고 있으며, 실제 사용자 선호도와 맞지 않는 경우가 많습니다. 이를 해결하기 위해 BenchBuilder라는 시스템을 소개하며, 이는 실시간 데이터 소스에서 고품질의 프롬프트를 필터링하여 오프라인 평가를 가능하게 합니다. 최신 벤치마크인 Arena-Hard-Auto v0.1은 Chatbot Arena에서 수집한 500개의 어려운 사용자 프롬프트로 구성되어 있습니다.

- **Technical Details**: BenchBuilder는 실시간 크라우드 소스 데이터 소스에서 고품질 프롬프트를 자동으로 필터링하여 고품질의 벤치마크를 생성합니다. 이 시스템은 특정 도메인 지식을 요구하는 등 일곱 가지 지표를 사용하여 고품질 프롬프트를 식별하며, LLM annotator(주석자)와 LLM judge(판정자)를 활용하여 완전히 자동화된 고품질 벤치마크를 구성합니다. Arena-Hard-Auto v0.1은 Chatbot Arena에서 수집한 데이터를 바탕으로 하며, MT-Bench보다 3배 더 좁은 신뢰 구간을 제공하고, 인간 선호도 순위와 89.1%의 일치를 보여줍니다.

- **Performance Highlights**: Arena-Hard-Auto v0.1은 MT-Bench와 비교하여 3배 더 좁은 신뢰 구간을 제공하며, 사용자 선호도 순위와 89.1%의 일치를 달성했습니다. 이는 인간 라벨러 없이도 단 $25의 비용으로 달성되었으며, 높은 신뢰도의 평가 벤치마크를 제공합니다.



### A Critical Study of What Code-LLMs (Do Not) Learn (https://arxiv.org/abs/2406.11930)
- **What's New**: 코드-LLM(code-LLMs)은 코드 제작 보조 업무에서 뛰어난 성능을 보였으나, 여전히 구문 오류 또는 변수 오용 등의 한계가 존재합니다. 이 연구는 코드-LLM이 어떤 코드 속성들을 인코딩하지 못하는지 세밀히 분석합니다. 특히, 코드-LLM은 구문 토큰과 식별자(identifier) 간의 관계를 인코딩하지 못한다는 사실을 발견했습니다.

- **Technical Details**: 코드-LLM은 대규모 코드와 자연어-프로그래밍 언어(NL-PL) 페어로 학습된 Transformer 모델입니다. 이 모델들은 코드 요약, 코드 검색, 코드 완성 및 프로그램 수정 등 다양한 작업에 사용됩니다. 이전 연구들은 모델이 구문 및 식별자 토큰 간의 관계를 인코딩하지 못한다는 점을 밝혀냅니다. 또한, 파인튜닝된 모델은 사전 학습된 모델에 비해 이러한 관계를 잘 인코딩하지 못한다고 합니다.

- **Performance Highlights**: 대규모의 모델(십억 개 이상의 파라미터가 있는 모델)이 수백만 개의 파라미터를 가진 작은 모델보다 코드 관련 정보를 덜 인코딩하는 것으로 나타났습니다. 코드-LLM의 주의(attention) 맵은 구문-식별자 관계를 인코딩하지 않으며, 숨겨진 표현(hidden representation)도 이러한 관계를 충분히 구별하지 못합니다. 이는 코드 모델이 일부 코드 속성을 인코딩하지 못해 실세상 작업에서 성능이 저조한 이유가 될 수 있습니다.



### DocCGen: Document-based Controlled Code Generation (https://arxiv.org/abs/2406.11925)
- **What's New**: 최근 대형 언어 모델(LLMs)은 C++, Java, Python과 같은 일반 프로그래밍 언어에서 자연어(NL)를 코드로 변환하는 작업에서 최첨단 성능을 보이고 있습니다. 그러나 YAML, JSON과 같은 구조화된 도메인 전용 언어(DSLs)에서는 제한적인 성능을 보입니다. 이를 해결하기 위해 DocCGen이라는 프레임워크를 제안하였으며, 이는 자연어에서 코드 생성 작업을 두 단계로 나눈 것입니다. 첫 번째는 NL 쿼리에 가장 잘 맞는 라이브러리를 라이브러리 문서를 통해 감지하고, 두 번째는 문서에서 추출한 스키마 규칙을 사용하여 디코딩을 제한하는 것입니다.

- **Technical Details**: DocCGen 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계는 정보 검색(IR)을 사용하여 관련 라이브러리를 감지합니다. 두 번째 단계는 신경-심볼릭 제약 디코딩(neuro-symbolic constrained decoding)을 사용하여 코드 생성을 제어하고 관련 라이브러리의 스키마를 준수하도록 합니다. 주어진 자연어 쿼리 q에 대해 코드 스니펫 c를 생성하는 과정에서, 수집된 라이브러리 설명 문서 D를 사용합니다. 구조화된 스키마는 각 필드의 유효 키워드 목록 및 필드 간 의존성 정보를 저장하며, 템플릿은 라이브러리의 코드 스니펫 구조를 문자열로 인코딩하여 디코딩 중 모델을 안내합니다.

- **Performance Highlights**: DocCGen은 Bash command와 Ansible YAML이라는 두 가지 복잡한 구조화 언어에서 성능 평가를 진행하였으며, 인-도메인(In-Domain)과 아웃-오브-도메인(Out-of-Domain) 설정 모두에서 실험했습니다. 실험 결과, DocCGen 프레임워크는 모든 여섯 가지 평가 지표에서 다양한 크기의 언어 모델을 일관되게 개선하여, 구조화된 코드의 문법 및 의미 오류를 줄이는 데 성공했습니다. 특히 Ansible-YAML 데이터셋에서는 18,000개의 샘플을 포함하며, 2,500개 이상의 모듈에서 NL-to-Code 데이터셋을 추가하여 공개할 계획입니다.



### Explainable assessment of financial experts' credibility by classifying social media forecasts and checking the predictions with actual market data (https://arxiv.org/abs/2406.11924)
- **What's New**: 이 논문에서는 소셜 미디어의 금융 창작자들의 신뢰성을 평가하는 솔루션을 제안합니다. 이 솔루션은 자연어 처리(NLP, Natural Language Processing)와 머신 러닝(ML, Machine Learning)을 결합하여 자산 가치 예측을 자동으로 분류하고 실제 시장 데이터와 비교하여 성공 확률을 계산합니다. 이 계산 결과를 연속적인 신뢰도 점수로 제공하며, 이는 기존의 이진 결과와 차별화되는 완전히 새로운 기여입니다.

- **Technical Details**: 이 연구는 금융 게시물의 예측을 유형별로 분류하고 이를 실제 시장 사건과 비교하여 사용자의 예측 정확도를 측정합니다. 이러한 성공 비율은 연속적인 신뢰성 점수로 제공되며, 소셜 미디어 사용자 지표와의 상관관계를 계산하여 금융 게시물과 예측에 대한 사용자 관심에 대한 인사이트를 제공합니다. 마지막으로, 모델의 예측 결과를 자연어로 설명하여 예측의 신뢰성을 높입니다.

- **Performance Highlights**: 제안된 신뢰도 평가 시스템은 소셜 미디어 금융 게시물을 분석하여 예측의 종류로 자동 분류하고, 실제 시장 데이터를 통해 검증된 연속적인 신뢰도 점수를 제공합니다. 또한, 사용자가 이해할 수 있도록 자연어 설명 기능을 통합했습니다. 기존 연구와는 달리, 이 논문의 접근 방식은 신뢰도 순위를 연속 척도로 제공하고, 사용자 맥락 지표와의 상관관계를 분석하여 추가적인 인사이트를 제공합니다.



### A Notion of Complexity for Theory of Mind via Discrete World Models (https://arxiv.org/abs/2406.11911)
Comments:
this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가할 수 있는 복잡한 상황에서의 사회적 추론 능력인 마음 이론(Theory of Mind, ToM)을 다룹니다. 저자들은 ToM 과제의 복잡성을 측정할 수 있는 프레임워크를 제안하며, 이를 통해 다섯 가지 주요 ToM 벤치마크의 복잡성을 평가하였습니다. 더 나아가, 이 논문은 '이산 세계 모델(Discrete World Models, DWM)'이라는 팁을 통해 모델의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: ToM 과제의 복잡성을 문제를 해결하는 데 필요한 상태(state)의 수로 정량화합니다. 여기에는 k차 신념(kth-order beliefs) 추적이 포함됩니다. 또한, 문제의 복잡성을 증가시키기 위해 일부러 추가된 부차적인 상태도 고려합니다. DWM 프레임워크는 에이전트의 상호작용에 따라 환경이 어떻게 변화하는지를 설명함으로써 모델의 정보력을 증가시킵니다. 이를 통해 CoT(Chain of Thoughts), ToT(Tree of Thoughts)와 같은 기존 방법보다 더 우수한 성능을 발휘합니다.

- **Performance Highlights**: DWM 기법은 ToMi, MindGames, Adv-CSFB, SocialIQA, FANToM 등에서 우수한 성능을 보였습니다. 특히, 상태 공간이 명확한 문제들에서 뛰어난 성능을 발휘했습니다. 다양한 LLMs를 대상으로 실험을 진행했으며, GPT-3.5-Turbo, GPT-4, LLaMA3-70B, Mixtral 8x7B 모델에서 실험을 수행했습니다. DWM을 사용한 결과, 암기 현상이 모델의 성능에 미치는 영향을 분석했을 때, 일부 벤치마크에서는 암기 현상이 모델 성능 저하와 직접적으로 연관되지 않는다는 사실을 발견했습니다.



### Unraveling the Mechanics of Learning-Based Demonstration Selection for In-Context Learning (https://arxiv.org/abs/2406.11890)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 문맥 내 학습(in-context learning, ICL) 능력을 향상시키는 학습 기반의 예시 선택(demonstration selection) 방법에 관한 새로운 발견을 다룹니다. 이러한 방법들은 기존에 높은 훈련 비용과 임의 태스크 간 저조한 일반화 성능 문제를 해결하는 데 어려움이 있었습니다. 논문은 두 가지 중요한 유사성 측정 요소를 제시함으로써 LLM의 ICL 수행 메커니즘을 분석하고자 합니다.

- **Technical Details**: 논문은 두 가지 주요 요인을 통해 유사성 측정의 중요성을 밝힙니다: 1) 예시와 테스트 케이스 입력 간의 다양한 수준의 과제 비특정적 텍스트 유사성(task-agnostic text similarities)을 통합할 수 있는 능력은 다양한 태스크에서의 일반화 성능을 높입니다. 2) 유사성 측정 시 과제 특정 레이블(task-specific labels)을 포함하면 각 특정 태스크에서의 성능이 상당히 개선됩니다.

- **Performance Highlights**: 논문에서는 10개의 데이터 셋과 다양한 LLM을 통해 정량적 및 정성적 분석을 통해 이 두 가지 발견사항을 검증합니다. 이 기반 위에 과제 비특정적 및 과제 특정 수요에 맞춘 두 가지 효과적이고 단순한 예시 선택 방법을 제안하며, 이는 비싼 LLM 추론의 오버헤드를 제거합니다.



### Attributions toward Artificial Agents in a modified Moral Turing Tes (https://arxiv.org/abs/2406.11854)
Comments:
          23 pages, 0 figures, in press

- **What's New**: 이번 연구는 인공지능(AI)에 의한 도덕적 평가(moral evaluations)가 인간의 도덕적 평가와 얼마나 유사하게 인식되는지를 조사하였습니다. 이를 위해 GPT-4라는 고급 AI 언어 모델을 사용하여 사람들에게 실제 인간의 도덕적 평가와 AI의 도덕적 평가를 구분하도록 했습니다. 이 실험은 Allen 등의 제안에 따라 수정된 도덕적 튜링 테스트(modified Moral Turing Test, m-MTT)를 사용했습니다.

- **Technical Details**: 미국 성인 299명을 대상으로, 출처를 알 수 없도록 한 후 도덕적 평가의 품질을 평가하게 했습니다. 참가자들은 AI의 도덕적 추론을 모든 차원에서 인간의 것보다 우수하다고 평가했습니다. 이는 Allen 등이 말한 비교 도덕적 튜링 테스트(comparative MTT)를 통과한 결과와 일치합니다. 그 다음으로 참가자들이 각 평가의 출처(인간 또는 컴퓨터)를 식별하도록 했을 때, 참가자들은 상당히 높은 확률로 맞추었습니다. 그러나 AI의 도덕적 추론이 열등하기 때문이 아니라, 오히려 우수하다고 인식된 것 등이 그 이유일 수 있습니다.

- **Performance Highlights**: 언어 모델이 인간보다 도덕적 추론에서 더 우수한 것으로 인식되는 것은 잠재적으로 해로운 도덕적 지침을 비판 없이 수용할 가능성을 제기합니다. 이것은 도덕적 문제에서 생성형 언어 모델에 대한 안전 장치가 필요함을 강조합니다.



### Risks from Language Models for Automated Mental Healthcare: Ethics and Structure for Implementation (https://arxiv.org/abs/2406.11852)
- **What's New**: 최근 급성장하고 있는 자동화된 정신 건강 관리 자동화를 위한 AI의 개발 중, 윤리적 및 실용적 과제를 해결하고 AI 에이전트가 정신 건강 지원에서 자율성을 갖추기 위한 구조적 프레임워크를 제안하는 연구가 발표되었습니다. 이 연구는 자율성 수준을 구분하고 윤리적 요구사항을 설명하며 유익한 기본 행동을 정의합니다.

- **Technical Details**: 연구진은 심리적 질환을 반영한 16가지 질문을 사용하여 10개의 최첨단 언어 모델(state-of-the-art language models)을 평가했습니다. 이러한 질문들은 정신 건강 전문가(M.D.)들에 의해 구성되고 평가되었습니다. 평가된 정신 상태는 정신병(psychosis), 조증(mania), 우울증(depression), 자살 충동(suicidal thoughts) 및 살인적 경향(homicidal tendencies) 등을 포함합니다. 연구 결과, 기존의 언어 모델은 지나치게 신중한 반응이나 아첨하는 반응과 같은 문제로 인해 인간 전문가의 표준에 도달하지 못했습니다.

- **Performance Highlights**: 현재 테스트된 모델 대부분이 정신 건강 위기 상황에서 사용될 경우 사용자에게 해를 끼칠 수 있음이 밝혀졌습니다. 이는 필요한 안전 장치의 부재가 주요 원인입니다. 그에 따라, 연구는 이러한 모델의 안전성을 향상할 수 있는 솔루션을 탐구했고, 결론적으로 AI 시스템의 윤리적인 프레임워크와 기본 행동을 준수하는 것이 중요하다고 주장합니다. 이는 사용자 정신 건강과 안전을 위협하는 현재 AI 기술의 위험을 방지하기 위한 조치입니다.



### Model Of Information System Towards Harmonized Industry And Computer Scienc (https://arxiv.org/abs/2406.11848)
Comments:
          Bachelor's thesis

- **What's New**: 이번 연구에서는 컴퓨터 공학부와 산업 현장 간의 원활한 소통을 위한 웹 기반 챗 애플리케이션의 설계 및 개발을 다루고 있습니다. 이 애플리케이션은 학생들이 졸업 후 산업 현장에서 필요한 스킬과 지식을 쉽게 전달받을 수 있도록 돕습니다. 이는 다양한 학문 분야에 적용될 수 있습니다.

- **Technical Details**: 본 프로젝트는 Waterfall 시스템 개발 수명 주기(Waterfall System Development Lifecycle)를 사용하여 시스템 프로젝트 계획을 수립했습니다. 이는 시스템을 개발하기 위해 필요한 전체 프로세스 및 하위 프로세스 목록을 제공합니다. 연구 방법으로는 과거 문서들에 대한 문헌 분석 기법을 사용하였습니다.

- **Performance Highlights**: 프로젝트의 결과로 웹 기반 챗 애플리케이션의 설계, 소프트웨어 개발 및 시스템 평가가 이루어졌습니다. 이 애플리케이션은 업계와 컴퓨터 공학부 간의 의사 소통을 원활하게 하며, 저장된 정보는 나중에 활용될 수 있습니다. 이는 기업과 대학에 소프트웨어에 대한 인식을 제고하고 컴퓨터 공학 교육과정 개선에 기여할 뿐만 아니라, 다양한 학문 분야에서도 활용될 수 있습니다.



### Refiner: Restructure Retrieval Content Efficiently to Advance Question-Answering Capabilities (https://arxiv.org/abs/2406.11357)
Comments:
          8 pages

- **What's New**: 최근 발표된 논문에서는 대형 언어 모델(LLM)의 지식 한계를 해결하기 위해 Retrieval-Augmented Generation(RAG)에 대한 새로운 접근법 '$\textit{Refiner}$'를 제안했습니다. '$\textit{Refiner}$'는 문서 조각들을 추출한 후 재구조화하여, LLM이 보다 효과적으로 중요한 정보를 인지하고 활용할 수 있게 합니다.

- **Technical Details**: '$\textit{Refiner}$'는 추출 후 재구조화 패러다임을 따르며, 단일 디코더 전용 LLM을 이용해 쿼리와 관련된 내용을 추출하고, 이를 내용의 상호 관련성에 따라 섹션으로 나눕니다. 이를 통해 정보의 구분을 강조하고, 원래 문맥과의 정렬을 개선합니다. 이 접근 방식은 여러 single-hop과 multi-hop QA 과제에서 우수한 성능을 보였습니다.

- **Performance Highlights**: $\textit{Refiner}$는 7B 파라미터를 가진 모델로 훈련되어, 단일 및 다중-hop QA 과제에서 뛰어난 성능을 입증했으며, state-of-the-art RAG 솔루션과 비교해도 우수한 성능을 보였습니다. 특히, multi-hop 과제에서는 80.5%의 토큰 절감과 1.6-7.0%의 성능 향상을 달성했습니다. 또한, 다양한 다운스트림 LLM과의 협업에서도 안정적인 성능을 유지하여, 긴 문서 조각의 문제를 효과적으로 해결해 냈습니다.



### Curating Grounded Synthetic Data with Global Perspectives for Equitable AI (https://arxiv.org/abs/2406.10258)
- **What's New**: 이번 논문에서는 AI 모델 훈련에 필수적인 데이터 다양성을 증가시키기 위해 새로운 방식의 합성 데이터 생성을 제안했습니다. 125개국에서 수집한 뉴스 기사 데이터를 활용하여 12개의 언어로 번역, 요약 및 주제 다양화를 통해 다양한 문화와 언어를 반영한 합성 데이터셋을 구축했습니다. 이 접근방식이 기존의 제약적인 데이터셋의 문제점을 해결하고 Named Entity Recognition (NER) 등의 여러 AI 분야에서 데이터 일반화 성능을 향상시키는 데 기여합니다.

- **Technical Details**: 해당 연구에서는 약 212개국에서 매일 100만 개 이상의 오픈 웹 뉴스 기사를 추출하였습니다. 각 기사는 12개의 언어(영어, 스페인어, 포르투갈어, 독일어, 러시아어, 프랑스어, 아랍어, 이탈리아어, 우크라이나어, 노르웨이어, 스웨덴어, 덴마크어)로 번역 및 요약되었습니다. 클러스터링 및 샘플링을 통해 기사 간 주제의 다양성을 보장하였고, 이 과정을 통해 5,049개의 기사가 73개의 고유한 주제를 다루게 되었습니다. NER 데이터셋에서는 GLiNER 및 메타-라마-3-70B-인스트럭트(Meta-Llama-3-70B-Instruct) 모델을 사용하여 엔티티를 라벨링하고 54개의 엔티티 유형을 식별했습니다.

- **Performance Highlights**: 제안된 방법론은 전통적인 NER 벤치마크 데이터셋에서 최대 7.3%의 성능 향상을 보여주었습니다. 다양성 있는 합성 데이터를 활용함으로써 AI 모델은 보다 진중한 글로벌 데이터를 모방하는 데 효과적이라는 것이 입증되었습니다.



New uploads on arXiv(cs.IR)

### News Without Borders: Domain Adaptation of Multilingual Sentence Embeddings for Cross-lingual News Recommendation (https://arxiv.org/abs/2406.12634)
- **What's New**: 최근 다언어 뉴스 소비자가 급증하면서, 이러한 사용자를 위한 맞춤형 뉴스 추천 시스템의 중요성이 커지고 있습니다. 본 논문은 PolyNews와 PolyNewsParallel이라는 두 개의 다언어 뉴스 코퍼스를 구축하고, 이를 통해 사전 학습된 다언어 문장 인코더(SE)를 뉴스 도메인에 맞게 튜닝한 새로운 방법론을 제안합니다. 특히, 뉴스에 특화된 문장 인코더 NaSE를 소개하며, 이를 통해 기존의 접근 방법들이 가지는 한계점을 극복하고자 합니다.

- **Technical Details**: 기존의 뉴스 추천 시스템(NNRs)은 특정 데이터에 맞춰 사전 학습된 LMs를 파인튜닝하는 방식이 주로 사용됩니다. 이는 자원 소모가 크고, 소수의 데이터나 초기 사용자 설정 상황에서는 비효율적입니다. 이에 비해, 본 논문에서는 NaSE를 이용한 방법론을 제안합니다. NaSE는 Massively Multilingual Sentence Encoder에 기반하며, PolyNews와 PolyNewsParallel을 기반으로 도메인 적응을 통해 뉴스 문장을 인코딩합니다. NaSE는 Denoising Auto-Encoding과 Machine Translation(MT) objectives를 사용하여 훈련되었습니다.

- **Performance Highlights**: 제안된 NaSE 모델은 Zero-Shot Cross-Lingual Transfer(ZS-XLT) 성능에서 기존 방법들을 능가하며, 진정한 Cold-Start 및 소량의 데이터 환경에서도 우수한 성능을 보입니다. 특히, 클릭-행동 데이터를 사용하지 않는 비파라미터 방식의 Late Click-Behavior Fusion을 사용했습니다. 이를 통해 파인튜닝을 제거하고도 최첨단 성능을 달성할 수 있음을 입증했습니다.



### PromptDSI: Prompt-based Rehearsal-free Instance-wise Incremental Learning for Document Retrieva (https://arxiv.org/abs/2406.12593)
Comments:
          21 pages

- **What's New**: PromptDSI는 기존 Differentiable Search Index(DSI) 모델의 한계를 극복하기 위해 제안된 새로운 프롬프트 기반 방법으로, 문서 검색 시 리허설 없는 인스턴스-단위 증강 학습을 달성합니다. PromptDSI는 변화하는 코퍼스에 대해 새로운 문서를 효율적으로 인덱싱하면서 안정성과 가변성을 유지하도록 설계되었습니다.

- **Technical Details**: PromptDSI는 DSI의 프로즌 pre-trained language model(PLM)에 프롬프트를 첨부하여 인덱싱 효율을 높입니다. 고정된 네트워크 내 메커니즘을 사용하여 질의-키 매칭을 줄임으로써 첫 번째 포워드 패스의 비효율성을 제거합니다. 또한, 초기 코퍼스에서 추출된 neural topic embeddings를 프롬프트 키로 사용하여 다채로운 프롬프트 사용을 보장합니다. 이는 매칭 메커니즘의 붕괴로 인해 발생하는 파라미터 낭비 문제를 해결합니다.

- **Performance Highlights**: PromptDSI는 IncDSI와 동등한 성능을 보이면서도 새로운 코퍼스에서 리콜율을 4% 이상 향상시키는 효과를 증명했습니다. 전체적인 성능 평가에서 새로운 코퍼스에 대한 적응력과 잔존 학습 정보의 보존 능력에서 두드러지는 성과를 보여주었습니다.



### Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation (https://arxiv.org/abs/2406.12580)
- **What's New**: RecBLR을 소개합니다, 이는 행동 종속 선형 순환 유닛(Behavior-Dependent Linear Recurrent Units, BD-LRU)을 기반으로 한 효율적인 순차 추천 모델입니다. RecBLR은 훈련 효율성, 저비용 추론, 그리고 강력한 성능이라는 세 가지 '골든 원칙'을 동시에 달성합니다.

- **Technical Details**: RecBLR은 게이팅 메커니즘과 행동 종속 설계를 선형 순환 유닛(linear recurrent units, LRU)에 통합하여 사용자 행동 모델링과 추천 성능을 크게 향상시켰습니다. 또한, 맞춤형 CUDA 커널과 병렬 스캐닝 메커니즘을 사용하여 하드웨어 인식 스캐닝 가속 알고리즘을 설계함으로써 병렬화 가능한 훈련 및 추론 효율성을 높였습니다.

- **Performance Highlights**: 다양한 길이의 사용자 행동 시퀀스를 포함한 실제 데이터셋에 대한 광범위한 실험을 통해 RecBLR이 강력한 추천 성능, 훈련 효율성 및 저비용 추론을 동시에 달성함을 입증했습니다. 특히 오랜 사용자 상호작용 이력이 있는 데이터셋에서도 뛰어난 확장성을 보여줍니다.



### Predicting Award Winning Research Papers at Publication Tim (https://arxiv.org/abs/2406.12535)
- **What's New**: 최근 연구 논문들의 과학적 영향을 예측하는 많은 연구가 진행되고 있습니다. 본 연구는 이러한 연구 논문들의 상 수상 가능성을 출판 시점에 제공되는 정보만을 활용하여 예측하고자 합니다. 이전 연구들과 차별화되는 본 연구는 출판 시점에 이용 가능한 인용 서브그래프(citation subgraph)를 구축하고, 이 서브그래프의 특징과 텍스트 특징을 결합하여 더 정확한 예측을 수행했습니다. ArnetMiner 인용 그래프와 32개 컴퓨터 과학 컨퍼런스의 최우수 논문 모음을 실험 데이터로 사용하였습니다.

- **Technical Details**: 본 연구는 논문의 인용 서브그래프를 구축하고, 이 서브그래프의 밀도 및 글로벌 클러스터링 계수와 같은 특징들을 초기 예측 지표로 사용했습니다. 이후, 논문 초록 및 제목에서 추출한 텍스트 특징을 결합하여 최종 예측 모델의 정확도를 높였습니다. 이러한 접근법은 논문이 출판되는 시점에 사용할 수 있는 정보만을 기반으로 미래의 상 수상 가능성을 예측하였습니다. 연구 초기에 Graph Mining과 텍스트 마이닝 방법론을 적용하여 해석 가능한 예측 모델을 구축했습니다.

- **Performance Highlights**: 본 연구에서 사용된 모델은 높은 재현율(high recall)과 낮은 false negatives 비율로 0.694의 F1 점수를 기록했습니다. 이는 모델이 상을 받지 못할 논문을 잘 식별할 수 있다는 것을 의미합니다. 이러한 방법은 연구자들이 출판 시점에서 자신의 작업을 초기 평가하는 데 도움을 줄 수 있습니다. 초기 해석 가능성 실험에서 상 수상 논문과 비수상 논문의 텍스트 및 위상학적 특징에 차이가 있다는 점을 확인할 수 있었습니다.



### LLM4MSR: An LLM-Enhanced Paradigm for Multi-Scenario Recommendation (https://arxiv.org/abs/2406.12529)
- **What's New**: 이 논문은 기존의 다중 시나리오 추천(MSR) 모델들이 충분한 시나리오 지식을 통합하지 않아 최적화 성능과 해석 가능성이 떨어진다는 문제를 해결하기 위해, 대형 언어 모델(LLM)을 활용한 새로운 패러다임 LLM4MSR을 제안합니다. 이 패러다임은 LLM을 통해 시나리오 상관 관계와 사용자 간의 크로스 시나리오 관심사를 추출하고, 계층적 메타 네트워크를 사용하여 성능을 향상시킵니다.

- **Technical Details**: LLM4MSR은 LLM을 통해 충분한 시나리오 지식과 사용자의 크로스 시나리오 선호도를 학습합니다. 이를 위해 시나리오 및 사용자 레벨 프롬프트를 설계하고, LLM 튜닝 없이 이 지식을 추론합니다. 이후 계층적 메타 네트워크를 사용하여 시나리오 인지 및 개인화된 추천 기능을 개선하는 메타 레이어를 생성합니다. 이는 각 시나리오 및 사용자의 특성을 명확히 반영하여 성능을 강화합니다.

- **Performance Highlights**: KuaiSAR-small, KuaiSAR, Amazon 데이터셋에서 수행한 실험 결과, LLM4MSR은 다음과 같은 주요 장점을 입증하였습니다: (i) 다중 시나리오 백본 모델들과의 호환성(1.5%, 1%, 40% AUC 개선), (ii) 산업 추천 시스템에서도 고효율성과 배포 가능성, (iii) 향상된 해석 가능성.



### Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback (https://arxiv.org/abs/2406.12501)
- **What's New**: 다중 모달 추천 시스템(MRS)은 다양한 온라인 웹 플랫폼에서 중요하며 최근 몇 년간 상당한 주목을 받았습니다. 기존 연구들은 1) 잡음이 포함된 다중 모달 콘텐츠, 2) 잡음이 포함된 사용자 피드백, 3) 다중 모달 콘텐츠와 사용자 피드백을 정렬하는 문제를 간과하는 경향이 있습니다. 이러한 문제를 해결하기 위해, Denoising and Aligning Multi-modal Recommender System (DA-MRS)을 제안합니다. DA-MRS는 다중 모달 잡음을 줄이기 위해 일관된 콘텐츠 유사성을 기반으로 아이템-아이템 그래프를 구축하고, 사용자 피드백을 디노이즈하기 위해 다중 모달 콘텐츠와 관측된 피드백의 확률을 연결하여 디노이즈된 BPR 손실을 제공합니다. 또한 DA-MRS는 사용자 선호도에 따른 정렬을 수행하여 더 세밀한 정렬을 제공합니다.

- **Technical Details**: DA-MRS는 두 가지 주요 단계로 구성됩니다. 첫째, 노이즈가 포함된 다중 모달 콘텐츠를 처리하기 위해 여러 모달리티의 일관성을 고려한 더 정확한 링크를 통해 아이템-아이템 의미 그래프를 구성합니다. 또한, 순수한 콘텐츠 유사성을 보완하기 위해 아이템-아이템 행동 그래프를 구축하여 더 포괄적이고 신뢰할 수 있는 아이템-아이템 협업 관계를 제공합니다. 둘째, 사용자의 임의 피드백의 영향을 제거하기 위해 사용자 피드백 신호의 확률적 생성을 정의합니다. DA-MRS는 다중 모달 콘텐츠에서 추정된 사용자 선호도와 관측된 피드백 신호를 결합하여 디노이즈된 BPR 손실을 도출합니다. 특정 작업 중심 정렬을 위해 DA-MRS는 사용자 선호도에 따른 정렬을 적용하여 사용자 선호 분포와 아이템 피드백 신호 간의 차이를 최소화합니다.

- **Performance Highlights**: DA-MRS는 다양한 데이터셋, 백본 모델, 잡음 시나리오에서 일관되게 추천 성능을 크게 향상시키는 플러그 앤 플레이 프레임워크입니다.



### LLM-enhanced Reranking in Recommender Systems (https://arxiv.org/abs/2406.12433)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)을 활용한 재랭킹(reranking) 프레임워크인 LLM4Rerank를 소개합니다. 기존의 재랭킹 모델들이 주로 정확도에 집중했던 것과 달리, LLM4Rerank는 정확도, 다양성(diversity), 공정성(fairness) 같은 여러 기준을 동시에 고려할 수 있는 통합된 접근 방식을 제공합니다.

- **Technical Details**: LLM4Rerank는 완전 연결 그래프 구조를 사용해 다양한 재랭킹 기준들을 체계적으로 통합하며, Chain-of-Thought(CoT) 프로세스를 통해 이를 관리합니다. 사용자 또는 배포자는 'Goal'이라는 추가 문장을 입력하여 모델의 포커스를 조정할 수 있습니다. 이 프레임워크는 세 가지 공공 데이터셋에서 검증되었으며, 최신 재랭킹 모델들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, LLM4Rerank는 여러 평가 기준에서 기존 최신 모델들과 비교하여 더 나은 성능을 보였습니다. 이를 통해 재랭킹 과정에서 퍼포먼스, 확장성, 개인화를 동시에 향상시킬 수 있음을 검증하였습니다.



### A Gradient Accumulation Method for Dense Retriever under Memory Constrain (https://arxiv.org/abs/2406.12356)
- **What's New**: 최근에 발표된 논문에서는 대규모 배치 (large batch)에 의존하여 InfoNCE 손실 (InfoNCE loss)로 밀집 검색기 (dense retriever)를 안정적이고 효과적으로 훈련시키기 위한 메모리 감소 방법인 ContAccum을 제안했습니다.

- **Technical Details**: ContAccum은 이전에 생성된 쿼리와 패시지 표현들을 활용하기 위해 이중 메모리 뱅크 구조 (dual memory bank structure)를 사용합니다. 이를 통해 더 많은 부정 샘플 (negative samples)을 사용할 수 있으며, 이는 밀집 검색기 훈련의 안정성을 제공합니다. 기존의 GradAccum 및 GradCache와 같은 메모리 감소 방법들이 시간이 많이 걸리고 불안정한 훈련을 초래하는 문제점을 해결하고자 합니다.

- **Performance Highlights**: 다섯 가지 정보 검색 데이터셋에서의 실험 결과, ContAccum은 메모리 제약 조건 속에서도 기존의 메모리 감소 방법들뿐만 아니라, 높은 자원을 사용하는 시나리오보다도 우수한 성능을 발휘했습니다. 이론적 분석과 실험 결과는 ContAccum이 현재의 메모리 뱅크 활용 방법들에 비해 더 안정적인 듀얼 인코더 훈련을 제공한다는 것을 확인했습니다.



### CherryRec: Enhancing News Recommendation Quality via LLM-driven Framework (https://arxiv.org/abs/2406.12243)
- **What's New**: 최신 대형 언어 모델(LLMs)이 언어 이해와 생성에서 눈에 띄는 발전을 이루었으며, 이를 기반으로 맞춤형 추천 시스템이 개선되었습니다. 그러나 LLM의 자동 회귀 생성(auto-regressive generation) 방식은 느린 추론 속도로 인해 실시간 추천 시스템에서 효과적이지 못했습니다. 이러한 문제를 해결하기 위해 CherryRec이라는 새로운 뉴스 추천 프레임워크를 제안합니다. 이 프레임워크는 추천의 질을 보장하면서 추천 속도를 가속화합니다.

- **Technical Details**: CherryRec 프레임워크는 다음과 같은 주요 구성 요소로 이루어집니다: 1) 사용자의 상호작용 역사에 기반하여 후보 뉴스를 빠르게 선택하는 Knowledge-aware News Rapid Selector, 2) 선택된 후보 뉴스를 텍스트로 변환하여 사용자 선호도를 향상시키는 Content-aware News LLM Evaluator, 3) 다양한 점수를 통합하여 최종 추천 점수(CherryRec Score)를 계산하는 Value-aware News Scorer. CherryRec는 이 과정을 통해 뉴스 추천의 정확성과 효율성을 동시에 높입니다.

- **Performance Highlights**: 실험 결과 CherryRec는 MIND, Yahoo R6B, Adressa 등의 벤치마크 데이터셋에서 기존의 최첨단 방법보다 더 우수한 성능을 보였습니다. 추천 속도와 정확성 모두에서 뛰어난 성과를 입증했습니다. CherryRec는 다양한 모듈에서 얻은 통찰력을 통합해 더 정확하고 개인화된 뉴스 추천을 가능하게 합니다.



### Intermediate Distillation: Data-Efficient Distillation from Black-Box LLMs for Information Retrieva (https://arxiv.org/abs/2406.12169)
Comments:
          13 pages, 7 figures, 3 tables

- **What's New**: 이번 논문에서는 기존의 큰 언어 모델(LLM)의 지식을 효율적으로 증류하여 검색 모델을 최적화하는 새로운 방법을 소개합니다. 특히, 	extit{Intermediate Distillation}이라는 데이터 효율적인 지식 증류 훈련 방법을 제안하며, 이는 LLM을 블랙 박스로 간주하고 랭킹 생성을 감독 신호로 활용해 지식을 증류합니다. 이를 통해 1,000개의 학습 인스턴스로도 검색 모델의 성능을 크게 향상시킬 수 있음을 증명했습니다.

- **Technical Details**: Intermediate Distillation은 LLM-랭커-리트리버 파이프라인을 사용하여 LLM의 랭킹 생성 출력을 감독 신호로 활용하는 데이터 효율적인 지식 증류 훈련 방법입니다. 먼저 LLM이 생성한 랭킹 순서를 통해 랭커 모델을 훈련한 후, 이 랭커 모델을 사용해 리트리버 모델을 추가로 훈련합니다. 이러한 접근법은 LLM의 강력한 제로샷 랭킹 능력을 활용하며, LLM의 출력 확률에 의존하는 이전 방법들의 한계를 극복합니다.

- **Performance Highlights**: 제안된 방법은 기존 방법에 비해 100배에서 1000배 더 적은 데이터를 필요로 하여 계산 비용을 크게 줄입니다. Extensive 실험을 통해, 우리 모델이 RAG(Retrieval-Augmented Generation) 프레임워크 내에서 질문-응답 과제의 성능을 크게 향상시키는 것을 실증하였습니다.



### Mutual Learning for Finetuning Click-Through Rate Prediction Models (https://arxiv.org/abs/2406.12087)
Comments:
          7 pages, 2 figures, 4 tables

- **What's New**: 이번 논문에서는 클릭률 예측(CTR) 모델의 성능을 향상시키기 위해 Mutual Learning 접근법을 제안했습니다. 이는 기존의 Knowledge Distillation 기반 접근법들과는 달리, 모델들이 서로 학습하면서 성능을 상호 향상시키는 방법입니다. Criteo와 Avazu 데이터셋에서 실험한 결과, Mutual Learning 알고리즘이 최대 0.66%의 성능 향상을 가져왔습니다.

- **Technical Details**: CTR 예측 과제는 주로 이진 분류 문제로 간주되며, 목표는 특정 항목이 클릭될 확률을 예측하는 것입니다. 본 연구에서는 DeepFM, DCN, PNN, FiBiNET 등의 깊은 신경망 모델을 활용했습니다. Mutual Learning 기술을 통해 학생 모델들은 동일한 데이터셋에서 동시에 학습하며 공동의 손실(MSE와 BCE 손실)을 통해 성능을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 제안된 Mutual Learning 접근법이 개별적으로 학습된 모델에 비해 더 높은 AUC (Area Under ROC Curve)를 달성했습니다. Criteo 데이터셋과 Avazu 데이터셋 모두에서 동일한 모델들을 상호 학습시켰을 때, 각각 최대 0.66%와 0.59%의 상대적 성능 향상을 보였습니다.



### Balancing Embedding Spectrum for Recommendation (https://arxiv.org/abs/2406.12032)
- **What's New**: 최근 추천 시스템에서 발생하는 '임베딩 붕괴(embedding collapse)' 문제를 해결하기 위해 DirectSpec이라는 새로운 방법을 제안했습니다. 이 방법은 임베딩 스펙트럼을 균형 있게 분포시켜, 사용자와 아이템이 임베딩 공간 전체를 효과적으로 사용하도록 합니다. 또한, DirectSpec의 분석을 통해 DirectSpec+라는 향상된 버전도 제안하고, 이 방법이 CL(Contrastive Learning)과의 연결 고리를 갖고 있음을 보여줍니다.

- **Technical Details**: 기존의 페어-와이즈(pair-wise) 학습 패러다임에서는 사용자의 선호도를 추출하는 알고리즘을 설계하는 데 집중했지만, 그로 인해 임베딩이 전체 공간이 아닌 부분 공간에만 분포하는 문제가 있었습니다. DirectSpec은 모든 차원을 균등하게 기여하도록 하여 이러한 임베딩 붕괴를 방지합니다. DirectSpec+는 자체 속도 조절 그라디언트(self-paced gradients)를 사용하여 상관관계가 높은 무관한 샘플을 보다 효과적으로 최적화합니다.

- **Performance Highlights**: DirectSpec 및 DirectSpec+를 MF(Matrix Factorization)와 LightGCN에 구현한 결과, 두 모델에서 BPR과 LightGCN가 각각 최대 52.6%와 41.8% 향상된 성능을 보였습니다. 실험 결과 DirectSpec+는 다양한 데이터셋에서 효과적이고 효율적임이 입증되었습니다.



### When Box Meets Graph Neural Network in Tag-aware Recommendation (https://arxiv.org/abs/2406.12020)
- **What's New**: 새로운 BoxGNN 알고리즘이 개발되었습니다. 이 알고리즘은 tag-aware 추천 시스템(TAG-aware Recommender System)의 한계를 극복하기 위해 등장했으며, 사용자 선호도의 다양성과 불확실성을 잘 표현할 수 있는 high-dimensional box embedding과 그래프 신경망(GNN)을 결합합니다.

- **Technical Details**: BoxGNN은 사용자, 아이템, 태그를 high-dimensional space 내의 hyper-box로 임베딩하고, 이 박스들 사이의 논리적 연산을 통해 메시지를 집계(aggregation) 합니다. 또한, gumbel smoothing 기술을 사용하여 학습 과정에서 gradient 소실 문제를 방지하고 박스의 교차 부분의 볼륨을 매칭 스코어로 사용합니다.

- **Performance Highlights**: 두 개의 공개 데이터셋과 하나의 LLM-강화 전자 상거래 데이터셋에서 실험을 통해 BoxGNN의 우수성이 다양한 state-of-the-art 기준선과 비교하여 입증되었습니다. BoxGNN은 사용자의 선호도를 효과적으로 모델링하고, 다중 이웃 신호(colla-borative signals)를 잘 반영합니다.



### Personalized Federated Knowledge Graph Embedding with Client-Wise Relation Graph (https://arxiv.org/abs/2406.11943)
- **What's New**: 기존의 Federated Knowledge Graph Embedding(FKGE) 방식은 클라이언트 간의 의미적 차이를 무시하고 전역 보완 지식을 모든 클라이언트에게 동일하게 제공해왔다. 이를 개선하기 위해, 새로운 접근 방법인 Personalized Federated Knowledge Graph Embedding with client-wise relation Graph(PFedEG)을 제안했다. PFedEG은 클라이언트 간의 관계 그래프를 사용하여 맞춤형 임베딩을 학습함으로써, 클라이언트 간의 'affinity'에 기반한 의미적 유사성을 반영한다.

- **Technical Details**: PFedEG는 클라이언트 맞춤형 보완 지식을 학습하기 위해 관계 그래프를 사용하여 이웃 클라이언트의 엔티티 임베딩을 통합한다. 각 클라이언트는 로컬 트리플과 맞춤형 보완 지식을 기반으로 개인화된 임베딩 학습을 수행한다. 이를 통해 로컬 최적화 목표와 전역 최적화 목표 간의 불일치를 해결하고, 보다 개인화된 지식을 학습할 수 있다.

- **Performance Highlights**: PFedEG는 네 가지 벤치마크 데이터셋에 걸친 광범위한 실험을 통해 최신 모델 대비 성능 우위성을 입증했다. 특히, 성능 개선은 4가지 메트릭을 통해 평가되었으며, 이는 FKGE의 정확성을 높이는 결과를 가져왔다.



### Influence Maximization via Graph Neural Bandits (https://arxiv.org/abs/2406.12835)
Comments:
          To appear at the 2024 ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)

- **What's New**: 이 논문에서는 다수의 소셜 미디어 캠페인에서 확산 네트워크의 구조에 대한 정보가 제한된 시나리오를 다루는 새로운 영향력 최대화(Influence Maximization, IM) 알고리즘인 IM-GNB(Influence Maximization with Graph Neural Bandits)를 제안합니다. 이 프레임워크는 신경 밴딧 알고리즘(Neural Bandit Algorithms)을 활용하여 사용자의 확률을 추정하고, 이를 바탕으로 탐색(exploration) 및 활용(exploitation)을 균형 있게 고려하여 실시간으로 시드 노드를 선택합니다.

- **Technical Details**: IM-GNB는 그래프 신경망(Genesis Neural Bandits, GNB)을 사용하여 시드 선택 과정을 최적화합니다. 초기에는 사용자들의 확산 확률을 예측하여 탐색 그래프와 활용 그래프를 구성합니다. 이후, Graph Convolutional Networks(GCN)을 사용하여 각 맥락에서의 영향력자의 예상 보상을 정교하게 다듬으며, 이를 통해 실시간으로 시드 노드를 선택합니다. 다중 라운드 확산 캠페인에서는 예산 불가피성 아래에서 각 라운드마다 새로 활성화된 사용자 수를 최대화하는 것이 목표입니다.

- **Performance Highlights**: 대규모 실제 데이터셋 두 개를 사용한 광범위한 실험을 통해, IM-GNB가 다른 기준 방법들과 비교하여 확산 결과를 상당히 개선했음을 입증했습니다. 실험 결과는 IM-GNB가 특히 네트워크 구조가 알려지지 않은 상황에서 영향력을 극대화하는데 매우 효과적임을 보여줍니다.



### RIGL: A Unified Reciprocal Approach for Tracing the Independent and Group Learning Processes (https://arxiv.org/abs/2406.12465)
Comments:
          Accepted by KDD 2024. 12 pages

- **What's New**: 이번 연구에서는 독립 학습 및 그룹 학습 두 가지 학습 방식을 통합하여 학생들의 지식 상태를 추적하는 새로운 모델 RIGL을 제안합니다. 이 모델은 독립 학습과 그룹 학습 간의 상호작용 효과를 최대한 활용하여 학생들의 전인적(holistic) 개발을 지원합니다.

- **Technical Details**: 제안된 RIGL 모델은 시간 프레임 인지 상호 임베딩 모듈(time frame-aware reciprocal embedding module)을 도입하여 학생과 그룹의 반응 상호작용을 동시에 모델링합니다. 또한, 상호 강화 학습 모델링(reciprocal enhanced learning modeling)을 사용해 두 학습 행동 간의 포괄적이고 보완적인 정보를 최대한 활용합니다. 나아가, 관계 기반 시간 주의 네트워크(relation-guided temporal attentive network)를 설계하여 개별 및 그룹 상호작용의 동적 복잡성을 탐구합니다. 마지막으로, 편향 인지 대조 학습 모듈(bias-aware contrastive learning module)을 도입하여 모델 훈련의 안정성을 강화합니다.

- **Performance Highlights**: 실제 교육 데이터셋 4개를 사용한 광범위한 실험 결과, 제안된 RIGL 모델이 HKT(holistic knowledge tracing) 과제에서 높은 성능을 보여줍니다.



### UniGLM: Training One Unified Language Model for Text-Attributed Graphs (https://arxiv.org/abs/2406.12052)
- **What's New**: UniGLM (Unified Graph Language Model) 프레임워크를 도입합니다. UniGLM은 여러 도메인의 Text-Attributed Graphs(TAGs)에 대해 일반화할 수 있는 첫 번째 그래프 임베딩 모델입니다. 이는 구조적 유사 노드를 식별하는 적응형 긍정 샘플 선택 기법과 반복적인 인코딩 계산을 최소화하여 학습을 가속하는 'lazy contrastive module'을 포함합니다.

- **Technical Details**: UniGLM은 다양한 도메인과 규모의 TAG를 사용하여 자기 지도 대조 학습(self-supervised contrastive learning)을 통해 훈련됩니다. 구조적 유사 노드를 식별하는 적응형 긍정 샘플 선택 기법과 텍스트 속성의 반복적인 인코딩을 피하기 위한 동적 메모리 뱅크를 포함합니다. 이는 노드의 로컬 및 글로벌 컨텍스트를 고려하여 긍정 샘플을 선택함으로써 다양한 TAG 간의 텍스트와 구조적 정보를 효과적으로 정렬합니다.

- **Performance Highlights**: 9개의 벤치마크 TAG에 대한 광범위한 실험 결과에 따르면 UniGLM은 여러 다운스트림 작업(노드 분류 및 링크 예측) 및 백본(GNN 및 MLP)에서 최첨단 그래프 임베딩 모델을 능가합니다. 또한 이전에 보지 못한 TAG에 대해 유용한 임베딩을 생성할 수 있는 일반화 및 전이 학습 성능을 입증하였습니다.



### GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models (https://arxiv.org/abs/2406.11945)
- **What's New**: 이 연구는 텍스트 속성을 지닌 그래프(Text-Attributed Graphs, TAGs)에 대한 자가 지도 학습(self-supervised learning) 기법을 제안합니다. 기존의 그래프 대조 학습(graph contrastive learning) 방법과 달리, GAugLLM이라는 새로운 프레임워크를 통해 텍스트 속성에 대한 언어 감독(language supervision)을 활용하여 뷰 생성을 개선합니다. 이 방법은 대규모 언어 모델(large language models, LLMs)인 Mistral을 적용하여 텍스트 속성을 활용한 그래프 학습을 강화합니다.

- **Technical Details**: 텍스트 속성의 불균형성과 그래프 구조와의 불일치를 해결하기 위해, 혼합 프롬프트 전문가(mixture-of-prompt-expert) 기법을 도입하여 다양한 프롬프트 템플릿을 활용해 원본 텍스트 속성을 변형합니다. 또한, 협력적 엣지 수정자(collaborative edge modifier)를 통해 텍스트와 구조적 공통점을 검토하여 엣지 증강(edge augmentation)을 수행합니다. 이는 BERT와 같은 소형 LLM을 미세 조정(fine-tuning)하여, 여러 변형된 텍스트 속성을 기능 공간(feature space)에 통합하는 방식입니다.

- **Performance Highlights**: 다양한 도메인의 5개 벤치마크 데이터셋을 통해 GAugLLM의 성능을 실험한 결과, 주요 대조 방법(예: BGRL, GraphCL, GBT)의 성능이 최대 12.3% 향상되었습니다. 더불어, 생성 모델(예: GraphMAE, S2GAE)과 그래프 신경망(예: GCN, GAT)에서도 지속적인 성능 향상이 관찰되었습니다.



