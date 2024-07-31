New uploads on arXiv(cs.CL)

### EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (https://arxiv.org/abs/2406.16858)
- **What's New**: EAGLE-2는 문맥을 감안한 동적 드래프트 트리(context-aware dynamic draft tree)를 도입하여 기존 EAGLE의 성능을 크게 높였습니다. 이 방법은 드래프트 모델의 확률 점수를 이용해 초안 토큰의 수락률을 추정하고, 이를 바탕으로 드래프트 트리 구조를 동적으로 조절합니다.

- **Technical Details**: EAGLE-2는 드래프트 모델의 적절히 보정된 확률 점수를 사용하여 드래프트 토큰 수락률을 예측합니다. 이를 통해 각기 다른 문맥에서 드래프트 트리 구조를 동적으로 조정하게 됩니다. 이는 기존의 고정된 구조를 사용하는 방법들에 비해 더 나은 성능을 보입니다. EAGLE-2는 추가적인 모델 학습 없이, 드래프트 모델의 확률 점수만을 이용해 드래프트 트리 구조를 조정합니다.

- **Performance Highlights**: EAGLE-2는 세 가지 LLM 시리즈 및 여섯 개의 과제를 대상으로 한 평가에서 3.05배에서 4.26배의 속도 향상을 기록했으며, 이는 EAGLE-1에 비해 20%-40% 더 빠른 속도입니다. 또한, EAGLE-2는 생성된 텍스트의 분포를 변경하지 않기 때문에 손실 없는 가속화 알고리즘입니다. 이를 통해 다중 회전 대화, 코드 생성, 수학적 추론, 명령 따라하기, 요약, 질문 답변 등 다양한 과제에서 탁월한 성능을 입증했습니다.



### Losing Visual Needles in Image Haystacks: Vision Language Models are Easily Distracted in Short and Long Contexts (https://arxiv.org/abs/2406.16851)
Comments:
          Under review

- **What's New**: 최근 비전 언어 모델(Vision Language Models, VLMs)의 긴 문맥 추론 능력을 평가하기 위한 LoCoVQA를 소개합니다. LoCoVQA는 수학적 추론, VQA(Visual Question Answering), 문자 인식 작업에 대해 점점 더 긴 시각적 문맥을 추가하여 테스트 예제를 생성합니다. 이 문맥은 기존 분포와 이질적인 분포의 혼합으로 구성되며, VLM들이 불필요한 정보를 무시하고 질의에 답변하는 능력을 평가합니다. VLM들은 대부분 시각적 문맥 길이가 증가함에 따라 성능이 급격히 저하되는 경향을 보입니다.

- **Technical Details**: LoCoVQA는 OK-VQA, MMStar, MNIST 등의 이미지 이해 벤치마크 데이터셋에서 콘텐츠 이미지를 샘플링하고, 최대 35개의 혼란을 주는 이미지를 함께 배치하여 시각적 문맥을 생성합니다. 혼란 이미지는 인-디스트리뷰션(in-distribution) 또는 아웃-오브-디스트리뷰션(out-of-distribution)에서 샘플링됩니다. VLM이 여러 이미지를 입력으로 받을 수 있는 경우, 시퀀스나 그리드 형태로 구성된 샘플을 제공합니다. 시각적 문맥의 주요 도전 과제는 불필요한 정보를 무시하고 중요한 세부 사항에 집중하는 것입니다.

- **Performance Highlights**: 다양한 VLM들을 LoCoVQA로 평가한 결과, 최신 상태의 모델들도 긴 시각적 문맥 내에서 성능이 빠르게 저하되는 경향을 보였습니다. 특히, 혼란 정보를 무시하고 필요한 정보를 추출하는 능력에서 중요한 개선 여지가 있음을 확인했습니다. 이는 VLM의 훈련 목표에서 근본적인 결함 때문일 가능성이 큽니다.



### RaTEScore: A Metric for Radiology Report Generation (https://arxiv.org/abs/2406.16845)
- **What's New**: 본 논문은 AI 모델이 생성한 의료 보고서의 품질을 평가하기 위한 새로운 엔티티 인식 지표인 Radiological Report Evaluation(RaTEScore)을 소개합니다. RaTEScore는 진단 결과와 해부학적 세부 사항 등 중요한 의료 엔티티를 강조하며, 복잡한 의료 동의어와 부정 표현에 대한 민감성을 가지고 있습니다. 이를 위해 RaTE-NER이라는 포괄적인 의료 NER 데이터를 구축하고, 이를 바탕으로 특화된 NER 모델을 훈련했습니다. 이러한 모델은 복잡한 방사선학 보고서를 구성 요소들로 분해하는 역할을 합니다.

- **Technical Details**: RaTEScore는 엔티티 임베딩의 유사성을 비교하여 파생되며, 이 유사성은 엔티티의 유형과 임상적 중요도에 기반해 평가됩니다. 구체적으로, RaTEScore는 문장에서 의료 엔티티와 그 유형(예: 해부학, 질병)을 식별한 후, 동의어 해석 모듈을 사용하여 엔티티 임베딩을 계산하고, 코사인 유사성을 평가합니다. 최종 점수는 임상적 중요도가 부여된 유사성을 기반으로 계산됩니다. 추가적으로 RaTE-NER이라는 NER 데이터 세트를 구축했으며, 이는 다양한 영상 방식 및 신체 부위에서 수집된 데이터로 구성되어 있습니다. 또한 RaTE-Eval이라는 새로운 벤치마크를 도입하여 다양한 임상 텍스트에 걸쳐 지표를 평가합니다.

- **Performance Highlights**: 공공 데이터 세트 ReXVal에서 RaTEScore의 성능을 처음 테스트한 결과, 인간의 선호도와 더 높은 일치를 보였습니다. 이후 RaTE-Eval의 다양한 하위 작업에서도 RaTEScore는 꾸준히 다른 지표를 능가하며 높은 성능을 보였습니다. 개별 구성 요소의 유효성을 검증하기 위해 세부 실험도 수행하였습니다.



### Exploring Factual Entailment with NLI: A News Media Study (https://arxiv.org/abs/2406.16842)
Comments:
          Presented at *SEM 2024

- **What's New**: 이 연구는 뉴스 미디어에서 발생하는 사실적 관계를 분석하기 위해 FactRel이라는 새로운 주석 체계를 소개합니다. FactRel은 텍스트적 내포보다는 사실적 내포를 모델링하며, 뉴스 기사에서 자연스럽게 발생하는 문장들에 대해 주석을 달았습니다. 이를 통해 사실적 관계가 미디어 담론을 분석하는 데 더 적합하다는 것을 확인했습니다.

- **Technical Details**: FactRel은 문장 쌍 간의 사실적 내포를 인코딩하는 3개의 카테고리(사실적으로 지원 - SUPPORT, 사실적으로 저해 - UNDERMINING, 사실적으로 중립 - NEUTRAL)로 구성된 주석 체계입니다. 주석 작업은 최신 generative LLMs (GPT-4)를 사용하여 데이터셋을 생성하는 실험과 함께, 최신 언어 모델(DeBERTa)을 fine-tuning하여 수행되었습니다.

- **Performance Highlights**: GPT-4를 사용한 few-shot 학습이 중간 크기의 언어 모델(DeBERTa)과 준하는 성능을 보였으며, 이는 이러한 작업이 세계 지식 및 고급 추론 능력에 크게 의존하고 있음을 시사합니다. 또한, FactRel을 사용하여 주석된 데이터셋이 사실적 관점에서 뉴스 텍스트를 분석하는 데 새로운 가능성을 보여줍니다.



### From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models (https://arxiv.org/abs/2406.16838)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 추론(inference) 과정에서의 컴퓨팅 양(scale) 증대가 어떻게 더 나은 결과를 가져오는지에 대해 설명합니다. 기존 연구에서는 훈련(training) 과정에서의 컴퓨팅 확장에 초점을 맞췄지만, 이 논문은 추론 시간에 초점을 맞춘 새로운 접근법을 제시합니다.

- **Technical Details**: 우리는 세 가지 주요 영역을 통합된 수학적 형태로 탐색합니다: 토큰 수준의 생성 알고리즘(token-level generation algorithms), 메타 생성 알고리즘(meta-generation algorithms), 그리고 효율적인 생성(efficient generation). 토큰 수준의 생성 알고리즘은 디코딩 알고리즘(decoding algorithms)이라고도 불리며, 한 번에 하나의 토큰을 샘플링하거나 토큰 수준의 검색 공간을 구성하여 출력을 선택합니다. 메타 생성 알고리즘은 부분적 또는 전체 시퀀스를 대상으로 하며, 도메인 지식(domain knowledge)을 통합하고, 백트래킹(backtracking)과 외부 정보를 통합합니다. 효율적인 생성 방법은 토큰 비용을 줄이고 생성 속도를 향상시키는데 초점을 맞춥니다.

- **Performance Highlights**: 논문에서는 토큰 수준, 메타 생성, 효율적인 생성 방법의 통합된 접근법을 통해 전통적 자연어 처리, 현대 LLMs, 그리고 머신 러닝 시스템의 세 연구 커뮤니티의 관점을 통합하여 새로운 인사이트를 제공합니다.



### USDC: A Dataset of $\underline{U}$ser $\underline{S}$tance and $\underline{D}$ogmatism in Long $\underline{C}$onversations (https://arxiv.org/abs/2406.16833)
Comments:
          32 pages, 18 figures

- **What's New**: 새로운 연구는 Mistral Large와 GPT-4와 같은 대형 언어 모델(LLMs)의 활용을 통해 사용자의 의견 및 입장을 자동으로 분류할 수 있는 시스템을 개발했습니다. 이 연구는 특히 Reddit 대화 스레드에서 사용자의 의견 변화를 추적하는 데 초점을 맞췄습니다.

- **Technical Details**: 연구팀은 Mistral Large와 GPT-4 모델을 이용해 두 가지 작업을 수행했습니다: 1) 'User Stance Classification' (사용자 입장 분류) - 사용자 게시물의 입장을 5단계 척도로 라벨링; 2) 'User Dogmatism Classification' (사용자 독단성 분류) - 전체 대화에서 사용자의 독단적 성향을 4단계 척도로 라벨링했습니다. 또한, 주요 Reddit 대화 스레드에서 0-샷, 1-샷, few-샷 설정을 통해 다중 사용자 대화 데이터를 수집하여 USDC 데이터셋을 구축했습니다.

- **Performance Highlights**: USDC 데이터셋은 764개의 다중 사용자 대화와 1,528개의 사용자 독단성 샘플 및 9,618개의 입장 샘플을 포함합니다. 이 데이터셋은 사용자 의견 변동 및 이유를 설명하는 라벨과 함께 사용되며, 기존의 단일 게시물 중심의 데이터셋보다 더 많은 맥락과 사용자 의견 변화를 포착할 수 있습니다. 이를 통해 소규모 언어 모델인 LLaMA-2-7B, LLaMA-3-8B, Falcon-7B와 같은 모델을 미세 조정 및 지시 튜닝하여 사용자 입장 및 독단성 라벨링 작업을 수행했습니다.



### Understanding and Mitigating Tokenization Bias in Language Models (https://arxiv.org/abs/2406.16829)
- **What's New**: 현재의 최첨단 언어 모델들은 자회귀적(autoregressive)이며, 토큰(tokens)이라는 하위 단위의 서브워드 변환을 필요로 합니다. 이 논문은 토큰화(tokenization)가 샘플링 편향(sampling bias)을 유발한다는 것을 발견했습니다. 특히, 최대 접두사 매칭(maximum prefix matching)과 같은 인코딩 방식에서 이는 더 많은 훈련 혹은 데이터로 해결되지 않습니다. 이 보편적인 문제를 해결하기 위해, 저자들은 토큰화된 데이터로 훈련된 모델에서 편향되지 않은 추정치를 얻기 위한 새로운 알고리즘을 제안합니다. 이 방법은 모델을 다시 조정할 필요가 없으며, 시퀀스 길이에 따라 선형적으로 확장됩니다. 결과적으로, 토큰화된 언어 모델에서 토큰 없는 동작을 시뮬레이션 할 수 있습니다.

- **Technical Details**: 이 연구는 WordPiece 토큰화 방법에서 사용되는 최대 접두사 인코딩 방식을 조사하여, 이 과정이 다음 토큰 확률의 편향된 추정을 초래한다는 것을 보여줍니다. 단어 접두사와 같은 문자열 함수 및 연결 함수를 사용하여 이 문제를 정의합니다. 편향은 훈련 데이터의 양이 증가해도 유지되며, 이는 1차 마르코프 체인 설정에서도 발생합니다. 이를 교정하는 방법을 통해, 모델을 재조정하지 않고 토큰 없는 동작을 시뮬레이션할 수 있는 새로운 알고리즘을 제안합니다.

- **Performance Highlights**: 이 연구에서 제안된 알고리즘의 정확성을 마르코프 체인 설정에서 실증적으로 확인했습니다. 이를 통해 전통적인 토큰 입력 방법과 비교할 때 전이 확률을 정확하게 복구할 수 있음을 보여주었습니다.



### RES-Q: Evaluating Code-Editing Large Language Model Systems at the Repository Sca (https://arxiv.org/abs/2406.16801)
- **What's New**: Large Language Models (LLMs)의 발전에 따라 다양한 복잡한 작업을 수행할 수 있는 LLM 기반 시스템이 등장했으며, 이러한 시스템의 성능을 평가하기 위한 새로운 벤치마크인 RES-Q를 제안했습니다. RES-Q는 실제 GitHub 커밋에서 파생된 100개의 코드 저장소 편집 작업으로 구성된 데이터셋입니다. 이 데이터셋은 LLM 시스템이 주어진 명령을 정확하게 따라 수정 작업을 수행할 수 있는 능력을 평가합니다. RES-Q는 기존의 벤치마크가 갖고 있던 한계를 극복하고 모델의 전반적인 능력을 평가하는 더 나은 방법을 제공합니다.

- **Technical Details**: RES-Q는 자연어 명령어 기반의 코드 저장소 편집 벤치마크로서, 100개의 GitHub 커밋에서 추출된 실제 편집 작업을 포함하고 있습니다. 각 작업은 명령어와 코드 저장소가 주어지며, LLM은 명령어에 맞는 수정 작업을 수행해야 합니다. 이 벤치마크는 Qurrent OS에서 구축된 코드 저장소 편집 시스템에서 다양한 최신 LLM 모델의 성능을 평가합니다. 또한, Python과 JavaScript 코드베이스에 대한 작업을 포함하여 평가 범위를 확장하였습니다.

- **Performance Highlights**: 최신 모델들을 RES-Q 벤치마크에서 평가한 결과, HumanEval 벤치마크에서 1% 차이를 보였던 Claude Sonnet 3.5가 RES-Q에서 GPT-4o를 12% 차이로 능가하는 것을 발견했습니다. 이는 전통적인 벤치마크가 포화 상태에 접근하는 상황에서 RES-Q가 모델의 능력을 효과적으로 구분할 수 있음을 시사합니다. 또한, 토큰 효율성, 기존 벤치마크와의 성능 관계, 폐쇄형 및 오픈 소스 LLM 간의 차이를 조사하였습니다.



### Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs (https://arxiv.org/abs/2406.16797)
- **What's New**: 이 논문은 다중 작업 적응에 적합한 새로운 언어 모델 적응 방법인 LoTA(Lottery Ticket Adaptation)를 제안합니다. 기존의 방법들은 모델의 모든 가중치를 수정하여 작업 간 파괴적인 간섭을 초래했지만, LoTA는 모델의 희소 서브네트워크만을 최적화하여 이러한 문제를 해결합니다.

- **Technical Details**: LoTA는 세 단계로 구성됩니다. 첫 단계에서는 모델을 처음으로 미세 조정(fine-tuning)하여 가중치 벡터 변화를 추적합니다. 두 번째 단계에서는 이 변화를 기반으로 희소 마스크를 추출합니다. 마지막 단계에서는 원래의 모델 가중치 상태로 초기화한 후, 선택된 서브네트워크만을 다시 미세 조정합니다. 이를 통해 다양한 과제에서의 파괴적 간섭을 최소화할 수 있습니다.

- **Performance Highlights**: LoTA는 여러 어려운 작업에서 기존의 완전 미세 조정(Full Fine-Tuning, FFT)과 저순위 적응(LoRA) 방법보다 우수한 성능을 보여주었습니다. 예를 들어 GPT-4 모델의 결과와 비교한 AlpacaEval 2 평가에서, LoTA는 19.0%의 승률을 기록했으며, 이는 FFT와 동일하고 LoRA의 15.3%를 넘는 결과입니다. 또한 LoTA를 통해 모델 병합 시 더 나은 성능을 얻을 수 있었으며, 기타 작업 평균 성능에서도 기존 방법보다 우수했습니다.



### M2Lingual: Enhancing Multilingual, Multi-Turn Instruction Alignment in Large Language Models (https://arxiv.org/abs/2406.16783)
Comments:
          39 pages

- **What's New**: 이번 연구에서는 다국어, 다회차(multilingual, multi-turn) 지침 미세 조정 데이터셋 (Instruction Finetuning Dataset), M2Lingual을 제안했습니다. 이 데이터셋은 70개 언어와 17가지 NLP 작업을 포괄하며, 총 182K의 IFT 쌍을 포함합니다. 인간 생성 데이터와 합성 데이터를 사용하여 다양한 초기 샘플을 기반으로 구축되었습니다.

- **Technical Details**: M2Lingual은 두 단계의 진화적 분류체계(Evol taxonomy)를 사용해 만들어졌으며, 복잡한 다국어 지침 미세조정(instruction finetuning) 모델을 양성하는 데 중점을 두고 있습니다. 이 데이터셋은 언어별로 균등하게 분포된 IR 쌍을 제공하며, 이를 통해 모델이 다국어의 복잡한 대화에서도 우수한 성능을 발휘하도록 합니다.

- **Performance Highlights**: M2Lingual로 미세조정된 대규모 언어 모델(LLMs)은 기존 다국어 IFT 데이터셋과 비교해 여러 평가 벤치마크에서 경쟁력 있는 성능을 일관되게 보여주었습니다. 특히, 번역된 다국어 다회차 평가 벤치마크와 다양한 다국어 과제에서 강력한 성과를 기록하였습니다.



### It Is Not About What You Say, It Is About How You Say It: A Surprisingly Simple Approach for Improving Reading Comprehension (https://arxiv.org/abs/2406.16779)
Comments:
          Accepted to ACL Findings

- **What's New**: 최근 몇 년간 자연어 처리(NLP)의 발전이 급속도로 이뤄지면서, 특정 관행이 제대로 검증되지 않은 채로 자리잡고 있습니다. 본 연구는 독해(Reading Comprehension) 작업에 중점을 두고 두 가지 주요 연구 질문을 다룹니다. 첫 번째로, 입력 순서—즉, 질문과 문맥의 순서—가 모델의 성능에 어떤 영향을 미치는지 분석했습니다. 두 번째로, 질문 혹은 문맥을 강조함으로써 성능 향상이 가능한지 여부를 연구했습니다. 실험 결과, 문맥을 먼저 제시하는 것이 모델의 성능을 최대 31%까지 향상시키는 것을 발견했습니다. 또한, 문맥을 강조하는 것이 질문을 강조하는 것보다 더 나은 결과를 제공합니다.

- **Technical Details**: 본 연구는 9개의 대형 언어 모델(LLM)을 3개의 데이터셋을 사용하여 평가했습니다. 실험을 통해 두 가지 주요 방법을 테스트했습니다: 프롬프트 기반(prompt-based)과 주의 기반(attention-based) 강조 방법입니다. 프롬프트 기반 방법은 입력에 몇 개의 토큰을 간단히 추가하는 것만으로 모델 성능을 최대 36%까지 개선시킬 수 있음을 보여주었습니다. 이는 더 작은 모델이 상당히 큰 모델을 능가할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 질문과 문맥의 순서가 중요하며, 문맥을 먼저 제시하는 것이 모델 성능을 최대 31%까지 향상시켰습니다. 또한, 문맥 강조가 질문 강조보다 더 나은 성과를 제공하며, 입력의 일부를 강조하는 것이 모델이 파라메트릭 지식이 부족한 질문에 효과적으로 대처하는 데 특히 유용했습니다. 매우 간단한 토큰 결합 방법이 모델 성능을 최대 36%까지 향상시켜, 작은 모델이 큰 모델을 능가할 수 있게 되었습니다.



### Finding Transformer Circuits with Edge Pruning (https://arxiv.org/abs/2406.16778)
Comments:
          We release our code and data publicly at this https URL

- **What's New**: 최근 연구에서 언어 모델의 해석을 위한 회로(circuits) 발견 자동화가 가능해졌으나, 효율성과 정확성 면에서 한계가 존재했습니다. 이 논문에서는 이러한 문제를 해결하기 위해 새로운 방법인 Edge Pruning을 제안합니다. Edge Pruning은 기존의 비효율적인 탐색 알고리즘이나 부정확한 근사치를 사용하지 않고, 최적화 문제로서 회로 발견 문제를 재구성하여 gradient-based pruning 기법을 활용합니다.

- **Technical Details**: Edge Pruning은 네트워크 구성 요소의 노드가 아닌, 구성 요소 간의 '엣지(edges)'를 제거함으로써 동작합니다. 이는 Transformer 모델의 잔류 스트림(residual stream)을 변형하여 모든 이전 활성화를 유지하도록 함으로써 가능해집니다. 이를 통해 우리는 특정 구성 요소에서 읽을 기록을 결정하는 edge masks을 최적화할 수 있으며, L0 정규화(L0 regularization) 기법을 사용하여 이러한 edge masks을 최적화합니다.

- **Performance Highlights**: Edge Pruning 기법은 GPT-2 Small 모델에서 기존 메서드보다 더 충실하고 정확한 회로를 다양하게 발견해냈습니다. 특히, 보다 복잡한 multi-template In-Out-of-Instruction(IOI)와 같은 작업에서, Edge Pruning은 기존 방법보다 2.65배 적은 엣지를 사용하면서도 유사한 성능을 보였습니다. 또한, Edge Pruning은 CodeLlama-13B와 같은 대규모 모델에서도 잘 작동하여, 이 기법이 대규모 모델에서 나타나는 현상을 분석하는데 실용적이고 확장 가능하다는 점을 보여주었습니다.



### Blending LLMs into Cascaded Speech Translation: KIT's Offline Speech Translation System for IWSLT 2024 (https://arxiv.org/abs/2406.16777)
- **What's New**: KIT(Karlsruhe Institute of Technology)에서는 최신 기술을 통합하여 IWSLT 2024의 오프라인 트랙에 참여한 음성 번역 시스템(ST)을 발표했습니다. 주요 혁신점으로는 Mistral-7B 모델을 통합하여 ASR(Automatic Speech Recognition)과 MT(Machine Translation) 출력을 개선하는 방법론을 적용한 것이 포함됩니다.

- **Technical Details**: KIT의 시스템은 WavLM과 MBART50 모델을 활용하여 ASR 모듈을 초기화하고, NLLB-200 모델을 MT 모듈에 사용했습니다. 특히, Mistral 7B Instruction-Tuned 모델을 사용하여 ASR과 MT 단계에서 출력의 정확도를 높였습니다. ASR 출력은 N-best 리스트를 생성하여 LLM을 통해 세밀하게 조정하고, MT 출력은 문서 단위로 LLM을 미세 조정하여 번역 품질을 향상시켰습니다.

- **Performance Highlights**: 제안된 시스템은 tst2019 테스트 셋에서 Word Error Rate에서 0.3% 개선, COMET 지표에서 0.65% 개선을 달성했습니다. 다만, 중첩된 스피커와 배경 소음 등 어려운 상황에서 ASR 성능이 좋지 않아 이러한 방법이 적용되지 않았습니다. 이러한 어려운 상황에서는 청크된 긴 형태의 디코딩을 사용하여 ASR 성능을 향상시켰습니다. ITV dev 셋에서 Word Error Rate를 37.83%에서 30.98%로 줄이는 성과를 보였습니다.



### OlympicArena Medal Ranks: Who Is the Most Intelligent AI So Far? (https://arxiv.org/abs/2406.16772)
Comments:
          10 pages

- **What's New**: 최근 Huang et al.(2024)에 의해 소개된 OlympicArena라는 새로운 벤치마크를 사용하여 최고 지능의 AI 모델을 평가한 보고서가 나왔습니다. 주로 GPT 시리즈와 함께 최신의 Claude-3.5-Sonnet 과 Gemini-1.5-Pro 모델을 비교 분석한 것을 다루고 있습니다. OlympicArena는 11,163개의 이중 언어 문제를 포함하며, 62개의 국제 올림픽 종목과 7개의 일반 과목을 다룹니다.

- **Technical Details**: 이 보고서는 두 가지 주목할만한 새로운 모델인 Claude-3.5-Sonnet과 Gemini-1.5-Pro를 GPT-4o 및 GPT-4V 모델과 비교합니다. OlympicArena Medal Table이라는 새로운 평가 방식을 도입해 AI 모델의 성능을 메달 순위로 평가합니다. 테스트는 텍스트 입력만 사용하는 규칙 기반 매칭(rule-based matching)을 사용하여 진행되며, LLM(Large Language Models) 및 LMM(Large Multimodal Models)이 동시에 평가됩니다.

- **Performance Highlights**: Claude-3.5-Sonnet은 GPT-4o와 거의 동등한 성능을 보였으며 일부 과목(물리, 화학, 생명과학)에서는 GPT-4o를 능가했습니다. Gemini-1.5-Pro 역시 GPT-4V보다 대부분의 과목에서 우수한 성능을 보여주었지만, 수학과 컴퓨터 과학에서는 약간의 열세를 보였습니다. 또한, 오픈 소스 모델들이 상업용 모델들에 비해 현저히 뒤처짐을 확인하였습니다.



### The GPT-WritingPrompts Dataset: A Comparative Analysis of Character Portrayal in Short Stories (https://arxiv.org/abs/2406.16767)
- **What's New**: 이 논문은 Reddit WritingPrompts 데이터셋을 확장하여 GPT-3.5로 생성된 단편 소설을 추가한 GPT-WritingPrompts를 소개합니다. 이를 통해 인간과 기계가 생성한 이야기의 감정적 및 묘사적 특성을 여섯 가지 차원에서 비교하고 분석했습니다.

- **Technical Details**: 연구진은 이야기를 균일한 구문 길이(약 500단어)로 맞추기 위해 GPT-3.5-turbo를 사용했습니다. 이야기의 특성을 분석하기 위해 감정적 차원(긍정-부정, 활동성, 지배력)과 지능, 외모, 권력이라는 세 가지 추가 축을 평가했습니다. 이야기는 1인칭, 2인칭, 3인칭 서술로 분류되며, 3인칭 서술은 남성 또는 여성 주인공을 포함합니다.

- **Performance Highlights**: GPT가 생성한 이야기는 평균적으로 더 긍정적이고 높은 지배력을 가지지만, 활동성이 낮고 외모 및 지능과 관련된 단어가 적습니다. 인간과 기계 모두 남성 및 여성 주인공에 대한 유사한 편견을 보이며, 여성 주인공은 더 긍정적이고, 덜 활동적이며, 통제력이 적고, 외모와 관련된 단어가 더 많이 사용되는 경향이 있습니다.



### Towards Fast Multilingual LLM Inference: Speculative Decoding and Specialized Drafters (https://arxiv.org/abs/2406.16758)
- **What's New**: 이 논문은 다국어 설정에서 높은 추론 시간 문제를 해결하기 위한 새로운 접근법으로 '추정 디코딩(speculative decoding)'을 사용하여 어시스턴트 모델을 훈련하는 방법을 탐색합니다. 이를 통해 추정된 미래 토큰을 대상으로 LLM이 이를 검증하는 방식으로 속도를 크게 향상시켰습니다.

- **Technical Details**: 추정 디코딩은 '드래프트-검증-수락(draft-verify-accept)' 패러다임을 사용합니다. 초기 예측을 위해 덜 복잡한 어시스턴트 모델(Mp)이 사용되며, 이후 타겟 LLM(Mq)이 각 토큰을 검증합니다. 검증된 토큰만이 최종 출력으로 채택되며, 소거 샘플링(rejection sampling)을 사용하여 이 과정을 수행합니다.

- **Performance Highlights**: 이 연구는 다국어 번역과 같은 다영역 실험에서 검증되었으며, 타켓 작업에 특정된 토큰 수가 증가함에 따라 속도가 로그 비례로 증가하는 현상이 관찰되었습니다. 또한, GPT-4 평가 점수 및 정성 분석을 통해 결과가 입증되었습니다.



### Towards Zero-Shot Text-To-Speech for Arabic Dialects (https://arxiv.org/abs/2406.16751)
- **What's New**: 이번 연구는 영어보다 자원이 부족한 아랍어 Zero-shot multi-speaker text-to-speech (ZS-TTS) 시스템의 격차를 해결하고자 합니다. 이를 위해, 먼저 기존의 대규모 데이터를 음성 합성에 적합하도록 조정하고, 아랍어 방언 식별 모델을 사용하여 ZS-TTS 모델의 성능 향상을 시도했습니다.

- **Technical Details**: 연구팀은 XTTS 모델을 미세 조정(fine-tune)하여 31명의 보지 못한 화자와 내부 방언 데이터셋에서 성능을 평가했습니다. XTTS는 15개 언어에서 3초 정도의 참조 음성으로 자연스러운 음성을 생성할 수 있는 다언어 다화자 음성 합성 모델입니다. VQ-VAE, GPT2, 그리고 HiFi-GAN을 활용해 텍스트 기반의 오디오 토큰을 예측하고, 이를 최종 오디오 신호로 변환합니다.

- **Performance Highlights**: 자동화된 평가와 인적 평가 결과, 제안된 모델은 아랍어 방언의 음성을 생성하는 뛰어난 성능을 보였습니다. 이는 아랍어 ZS-TTS 연구의 잠재력을 크게 향상시킬 수 있음을 시사합니다.



### Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers (https://arxiv.org/abs/2406.16747)
Comments:
          preprint

- **What's New**: SPARSEK 주의(attention)을 소개합니다. 이는 자원의 제약이 있는 디바이스에서 자가 주의 메커니즘(self-attention mechanisms)과 관련된 계산 및 메모리 문제를 해결하기 위해 설계된 혁신적인 희소 주의 메커니즘입니다. SPARSEK는 각 쿼리(query)에 대해 키-값(KV) 쌍의 고정 수를 선택함으로써 선형 시간 복잡도와 일정한 메모리 사용을 가능하게 합니다.

- **Technical Details**: SPARSEK는 쿼리를 액세스하지 않고 각 KV 쌍의 중요도를 평가하는 점수 네트워크와, 점수를 선형 시간에 소프트 마스크(soft mask)로 정규화하는 새로운 차별 가능한 top-k 마스크 연산자를 통합합니다. 이 방법은 기존의 top-k attention 개념에 영감을 받았지만, 기존 방법의 비차별성과 다르게 차별 가능하게 설계되었습니다. 이러한 방법을 통해 KV 쌍의 선택과 주의 연산을 하나로 결합하는 Triton 커널을 개발하여 4096개의 입력 토큰을 처리할 때 FlashAttention을 능가하는 성능을 보입니다.

- **Performance Highlights**: SPARSEK 주의는 기존의 희소 주의 방법보다 학습과 추론에서 더 나은 성능을 발휘하며, 특히 언어 모델링 등의 작업에서 유의미한 속도 향상을 제공합니다. 또한, 방법이 사전 학습된 대형 언어 모델(LLMs)과의 통합이 용이해, 긴 범위의 의존성을 효율적으로 관리할 수 있는 실용적인 솔루션을 제공합니다. 실험 결과, SPARSEK 주의는 기존의 모든 효율적인 주의 방법들을 일관되게 능가했습니다.



### Adversarial Contrastive Decoding: Boosting Safety Alignment of Large Language Models via Opposite Prompt Optimization (https://arxiv.org/abs/2406.16743)
- **What's New**: 최근 대형 언어 모델(LLM)의 안전성과 유해한 응답 방지가 주요 관심사로 떠오르면서, 모델 트레이닝 없이 로그숫자를 조정해 응답을 개선하는 '대조 디코딩(Contrastive Decoding)' 방법이 주목받고 있습니다. 저자들은 Adversarial Contrastive Decoding(ACD)라는 최적화 기반 프레임워크를 제안하여, 두 개의 반대되는 시스템 프롬프트(system prompts)를 생성하고 이를 활용해 프롬프트 기반 대조 디코딩을 수행합니다.

- **Technical Details**: 기존의 안전성 정렬 방법들은 높은 품질의 데이터셋과 큰 컴퓨팅 자원을 요구했으나, ACD는 이러한 문제를 해결하기 위해 설계되었습니다. 구체적으로, Opposite Prompt Optimization(OPO)이라는 방법을 통해 '안전 보호 프롬프트(Safeguarding Prompt)'와 '대립 프롬프트(Adversarial Prompt)'를 최적화합니다. 이를 통해, 모델이 인간의 가치에 더 부합하는 응답을 생성할 수 있게 도와줍니다.

- **Performance Highlights**: 확장된 실험 결과, ACD는 기존 모델 트레이닝 없는 디코딩 방법보다 20% 이상 안전성 향상을 목표로 했으며, 생성 능력에 큰 영향을 미치지 않았습니다. Instructive Decoding 대비 약 7%의 안전성 향상을 보였습니다.



### CLIMATELI: Evaluating Entity Linking on Climate Change Data (https://arxiv.org/abs/2406.16732)
Comments:
          7 pages, ClimateNLP 2024

- **What's New**: 이번 연구에서는 기후 변화 관련 데이터를 대상으로 한 최초의 수작업으로 주석된 엔터티 링크(EL) 데이터셋, 'CLIMATELI (CLIMATe Entity LInking)'를 소개합니다. 이 데이터셋은 3,087개의 엔터티 스팬을 Wikipedia에 연결하며, 이를 통해 다양한 장르에서 기존 EL 시스템을 평가하고 CC와 관련된 엔터티를 자동으로 필터링하는 방법을 제안합니다.

- **Technical Details**: 기후 변화는 전 세계적으로 중요한 주제로, 다양한 연구 분야에서 관심을 받고 있습니다. 연구에서는 5개의 영어 장르(Wikipedia 페이지, 학술 논문, 웹 뉴스, UN 보고서, YouTube 자막)에서 수집된 총 12,802개의 토큰과 3,087개의 엔터티 링크를 포함한 CLIMATELI를 구축했습니다. 엔터티 링크 주석 작업은 Wikipedia 페이지를 기준으로 수작업으로 수행되었으며, Wikifier의 예측을 기반으로 엔터티를 수정하고 추가하는 방식을 사용했습니다.

- **Performance Highlights**: 현재 사용되는 EL 모델들은 엔터티 레벨과 토큰 레벨 모두에서 인간의 성능에 크게 뒤처진다는 결과를 확인했습니다. 특히, 명사 형태가 아닌 엔터티나 CC와 관련이 적은 엔터티를 포함하거나 제외하는 경우 모델 성능이 큰 영향을 받았습니다. 두 명의 연구자가 수행한 주석 작업의 경우, 토큰 레벨에서 80% 이상의 높은 일치를 보였습니다.



### Venturing into Uncharted Waters: The Navigation Compass from Transformer to Mamba (https://arxiv.org/abs/2406.16722)
- **What's New**: 최근 도입된 Mamba가 Transformer의 우위를 도전하며 많은 연구자들의 관심을 끌고 있습니다. 이번 설문 연구는 Mamba 기반 모델들을 중심으로, 그 잠재력과 주요 연구 차원을 포괄적으로 논의합니다.

- **Technical Details**: 이 논문은 Mamba 메커니즘의 작동 원리와 구조적 상태 공간 모델(structured state space models)에 기반한 이론, Mamba와 다양한 네트워크의 통합 및 개선, 그리고 Transformer의 대체 가능성을 탐구합니다. 또한, Mamba와 Transformer를 커널 함수(kernel functions) 프레임워크 내에서 비교하여 그들의 수학적 특성을 통합 비교합니다.

- **Performance Highlights**: 현재까지 Mamba와 관련된 대부분의 개선사항을 포괄하고 있으며, 특히 Transformer와 Mamba를 결합하여 상호 단점을 보완할 수 있는 가능성을 제시합니다.



### AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models (https://arxiv.org/abs/2406.16714)
- **What's New**: AutoDetect라는 통합 프레임워크를 소개하여 다양한 작업에서 대형 언어 모델(LLM)의 약점을 자동으로 노출시키는 방법을 개발했습니다. 이 프레임워크는 'Examiner', 'Questioner', 'Assessor'라는 세 가지 LLM 기반 에이전트(collaborative agents)를 통해 약점을 실시간으로 평가하고 식별합니다.

- **Technical Details**: AutoDetect는 교육 평가 시스템에서 영감을 받아, 학생의 학습 성과를 측정하는 방식과 유사한 방식으로 LLM의 약점을 평가합니다. 'Examiner'는 세부적이고 동적인 분류체계를 구축하고, 'Questioner'는 다양한 테스트 포인트에 맞춰 도전적인 질문을 생성하며, 'Assessor'는 모델의 응답을 분석하여 문제를 추론합니다. 이들의 상호작용을 통해 LLM의 개별 약점을 효과적으로 노출시키도록 설계되었습니다.

- **Performance Highlights**: AutoDetect는 GPT-3.5-turbo와 Claude-3-sonnet과 같은 강력한 모델에서 50% 이상의 약점 식별 성공률을 달성했으며, ChatGPT와 Claude와 같은 주요 모델에서 30% 이상의 식별율을 보였습니다. 이를 통해 Mistral과 Llama 시리즈와 같은 인기 오픈 소스 모델의 성능이 여러 벤치마크에서 10% 이상 향상되었습니다.



### Task Oriented In-Domain Data Augmentation (https://arxiv.org/abs/2406.16694)
- **What's New**: 새로운 연구에 따르면, 대규모 언어 모델(LLMs)이 특정 도메인(예: 법률, 광고)에서 더 나은 성능을 발휘할 수 있도록 지원하는 데이터 증강 프레임워크인 TRAIT을 제안했습니다. TRAIT는 태스크 지향(in-domain) 데이터 선택과 태스크 지향 합성 구문 생성을 통해 도메인 지식을 효과적으로 강화하여 지속적인 사전 학습 중 모델 성능을 향상시킵니다.

- **Technical Details**: TRAIT 프레임워크는 두 가지 주요 부분으로 나뉩니다. 첫째, 데이터 선택 알고리즘은 일반 코퍼스에서 인도메인 데이터를 식별하고 선택하여 도메인 지식을 풍부하게 합니다. 선택된 데이터는 중요한 교육적 가치를 가지고 있어 모델 성능 향상에 기여합니다. 둘째, 태스크 지향 합성 구문 생성 가이드를 통해 도메인 지식을 활용하여 다운스트림 태스크를 해결하는 방법을 포함한 합성 구문을 생성합니다. 이 합성 구문은 개별 문제 해결법과 문제들 간의 관계를 학습하여 모델이 태스크 요구 사항에 맞게 조정되도록 돕습니다.

- **Performance Highlights**: TRAIT은 광고 분야에서 평균 8%, 수학 분야에서 평균 7.5%의 성능 향상을 가져왔습니다. 광고 분야에서는 7개 다운스트림 태스크에서 기존 지속 학습 방법보다 평균 6.5% 높은 성능을, 수학 분야에서는 9개 다운스트림 태스크에서 평균 5.5% 높은 성능을 기록했습니다. 특히 수학 도메인의 MATH 태스크에서는 기본 LLM에 비해 15% 이상의 성능 향상을 보였습니다.



### Scaling Laws for Linear Complexity Language Models (https://arxiv.org/abs/2406.16690)
Comments:
          Technical report. Yiran Zhong is the corresponding author

- **What's New**: 최근 대형 언어 모델(LLM)에서 선형 복잡성 모델의 확장 가능성에 대한 관심이 높아지고 있습니다. 이번 연구에서는 선형 복잡성 언어 모델의 확장 법칙을 제시하여 이들의 확장성을 확인하고자 합니다. 세 가지 효율적인 선형 아키텍처를 조사하였으며, 이를 기존의 softmax 기반 변환기와 비교하였습니다.

- **Technical Details**: 우리가 조사한 선형 아키텍처는 다음과 같습니다: TNL (data-independent decay를 가진 선형 attention 모델), HGRN2 (data-dependent decay를 가진 선형 RNN), cosFormer2 (decay가 없는 선형 attention 모델). LLaMA를 소프트맥스 attention의 기준으로 사용하여 비교를 수행하였습니다. 각 모델은 70M에서 7B 파라미터까지 다양한 크기로 훈련되었으며, 300B-token 코퍼스에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, 기존 변환기 기반 모델과 비슷한 확장성을 보이면서, 두뇌 능력과 정보 유지 능력에서 뛰어난 성과를 나타냈습니다. 그러나 정보 검색 작업에서는 성능이 다소 약한 것으로 확인되었습니다. 전체적인 성능 분석에서 FLOPs 예산 하에 cross-domain perplexity와 commonsense reasoning에서 LLaMA를 능가하였습니다. 또한, 선형 모델에서는 데이터 종속 성분이 검색 작업에서 유리하게 작용하며, 다른 작업에서는 데이터 비종속 성분과 큰 차이가 없음을 발견하였습니다.



### Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation (https://arxiv.org/abs/2406.16678)
- **What's New**: 새로운 sentence segmentation(문장 분할) 모델인 'Segment any Text(SaT)'가 소개되었습니다. 이 모델은 기존 방법론의 한계를 극복하며, 구두점에 덜 의존하는 pretraining(사전 학습) 기법을 이용하고 있습니다. 또한, 다양한 도메인에 적용 가능하도록 파라미터 효율적인 fine-tuning(세밀 조정)을 도입하였습니다.

- **Technical Details**: SaT는 서브워드 기반의 다국어 인코더 언어 모델(LMs)을 웹 스케일 텍스트에서 자연스럽게 발생하는 줄 바꿈을 예측하도록 self-supervised 방식으로 훈련합니다. 그 후 sentence-segmented data(문장이 분할된 데이터)에서 supervised 방식으로 추가 학습을 진행하여 성능을 향상시킵니다. 이 과정에서 구두점의 누락과 같은 노이즈에 강한 robust(강건성)을 확보하기 위해 입력에 무작위로 corruption(오염)을 적용하는 기법을 제안하였습니다. 또한, 짧은 시퀀스 문제를 해결하기 위해 제한된 lookahead 기제를 도입하였습니다.

- **Performance Highlights**: SaT는 85개 언어에서 높은 성능을 보이며, 기존의 state-of-the-art 모델인 WtP를 능가합니다. 특히 구두점이 부족하거나 텍스트 형식이 불규칙한 상황에서 우수한 결과를 보였습니다. SaT는 다양한 도메인에 최소한의 예제만으로도 세밀하게 조정될 수 있어, 데이터가 제한된 상황에서도 뛰어난 적응성을 보였습니다. 모델의 효율성 또한 개선되어, 기본 3-layer 모델은 약 0.5초 만에 1000 문장을 분할할 수 있습니다.



### Computational Approaches to the Detection of Lesser-Known Rhetorical Figures: A Systematic Survey and Research Challenges (https://arxiv.org/abs/2406.16674)
Comments:
          Submitted to ACM Computing Surveys. 35 pages

- **What's New**: 이 논문은 잘 알려지지 않은 수사적 기법(rhetorical figures)에 대한 종합적인 컴퓨팅 접근 방식을 탐구합니다. 이는 텍스트의 의미를 완전히 이해하기 위해 매우 중요하며, 기존의 자연어 처리(NLP) 도구가 종종 놓치는 부분을 보완합니다.

- **Technical Details**: 논문은 수사적 기법(rhetorical devices)의 다양한 정의와 기능, 데이터셋, 그리고 탐지 방법론에 대해 자세히 논의합니다. 특히, 대조법(antithesis), 반복법(epanalepsis) 및 재귀법(zeugma)과 같은 덜 알려진 기법들을 중심으로 합니다. 이들은 언어적, 컴퓨터적 관점에서 분석되며, 주요 기술적 도전과 해결 방안도 제시됩니다.

- **Performance Highlights**: 현재 대부분의 연구는 은유(metaphor), 풍자(irony), 비꼼(sarcasm) 등 잘 알려진 수사적 기법에 초점을 맞추고 있습니다. 반면, 이 논문은 덜 연구된 24가지 수사적 기법에 대한 검출 기술을 조사하여, 연구자들이 직면한 주요 도전 과제를 요약하고 각각의 해결책을 제안합니다. 이 작업은 NLP의 다양한 영역에서 수사적 기법이 가지는 중요성을 강조합니다.



### CAVE: Controllable Authorship Verification Explanations (https://arxiv.org/abs/2406.16672)
- **What's New**: CAVE (Controllable Authorship Verification Explanations)라는 새로운 모델이 개발되었습니다. 이 모델은 자유 텍스트 방식의 저자 확인 설명을 생성하며, 설명의 구조를 제어할 수 있다는 특징을 갖고 있습니다. CAVE는 중요한 언어적 특징들을 고려한 부분 설명으로 구조화될 수 있으며, 설명과 예측 레이블의 일관성을 쉽게 검증할 수 있습니다.

- **Technical Details**: CAVE는 Llama-3-8B 모델을 기반으로 훈련되었습니다. 인간이 작성한 저자 확인 설명 코퍼스가 없기 때문에 GPT-4-Turbo 모델에서 샘플링된 은표준 설명을 사용하여 Llama-3-8B 모델을 미세 조정했습니다. 설명은 JSON 형식으로 제공되어 사용자가 설명을 다양한 하위 단계로 자동으로 구문 분석할 수 있게 됩니다.

- **Performance Highlights**: 세 개의 어려운 저자 확인 데이터셋(IMdB2, Blog-Auth, FanFiction)을 사용한 실험에서 CAVE 모델은 자동 및 인간 평가 기준으로 높은 품질의 설명을 생성했으며, 경쟁력 있는 작업 정확도를 달성했습니다.



### Large Language Models Are Cross-Lingual Knowledge-Free Reasoners (https://arxiv.org/abs/2406.16655)
- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)의 다중 언어 추론 능력이 주목받고 있는 가운데, 이 논문은 다언어 추론 과정을 지식 검색(knowledge retrieval)과 비지식 추론(knowledge-free reasoning) 두 부분으로 나누어 분석합니다. 또한, 이러한 모델의 다언어 전이 효과를 검토합니다. 연구 결과, 비지식 추론 능력은 거의 완벽하게 다른 언어로 전이될 수 있지만, 지식 검색은 전이를 현저히 저해한다는 것을 발견했습니다.

- **Technical Details**: 이 연구는 평가와 해석 분석 두 가지 부분으로 이루어졌습니다. 평가는 ‘StrategyQA’ 데이터셋을 사용하여 지식 검색 요구 사항을 조정하고, 특정된 데이터셋을 번역하여 사용했습니다. 또, 지식 검색이 전이 효과에 미치는 영향을 분석하기 위해 ‘Knowledge-Free Reasoning Dataset (KFRD)’를 구성했습니다. 해석 분석 부분에서는 신경 회로망의 뉴런 활성(neural activation)과 은닉 상태(hidden states)를 다중 언어간 비교하여 전이 과정에서의 계산적 유사성을 평가했습니다.

- **Performance Highlights**: 연구 결과, 비지식 추론 능력은 한 언어에서 미세 조정(finetuning) 후 다른 언어로 효과적으로 전이될 수 있습니다. 특히, 모델 중간-상위 계층에서의 계산적 유사성은 지식 검색보다 비지식 추론이 훨씬 높았습니다. 이는 비지식 추론이 다언어 모델 내에서 언어 공유 메커니즘에 내재되어 있다는 것을 시사합니다.



### Evaluation of Language Models in the Medical Context Under Resource-Constrained Settings (https://arxiv.org/abs/2406.16611)
- **What's New**: 의료 분야에서 Transformer 아키텍처 기반 언어 모델의 잠재력은 매우 크지만, 실제로 이를 제품으로 출시하려면 해당 모델의 행동을 정확히 이해해야 합니다. 특히, 제한된 자원 환경에서 이러한 모델을 평가하는 기술적 평가가 부족합니다. 이를 해결하기 위해 본 연구는 53개의 의료 도메인 언어 모델을 텍스트 분류 및 생성 작업에 대해 종합적으로 평가하였습니다.

- **Technical Details**: 이 연구는 의료 도메인에서 110 million에서 13 billion 파라미터에 이르는 Transformer 기반 모델을 대상으로 하였습니다. 우선 텍스트 분류를 위해 모델 훈련 또는 미세 조정이 필요하지 않은 zero-shot 프로핑(prompting) 방식을 사용하였고, 이는 많은 사용자가 처한 제한된 자원 환경과 유사한 설정입니다. 이를 통해 자원 제약 환경에서도 높은 성능을 발휘할 수 있는 가능성을 발견했습니다.

- **Performance Highlights**: 53개의 모델을 평가한 결과, 특정 모델이 도메인 특화 없이도 의료 지식을 포함할 수 있는 잠재력을 가지고 있다는 점에서 고무적인 성능을 보였습니다. 이는 다양한 작업과 데이터셋에서 우수한 성능으로 나타났으며, 의료 상황에서 모델 응용을 더욱 탐구할 필요성을 강조합니다.



### CLEAR: Can Language Models Really Understand Causal Graphs? (https://arxiv.org/abs/2406.16605)
- **What's New**: 사람들이 세계를 이해할 때 중요한 것은 인과 추론입니다. 최근 언어 모델이 발전함에 따라, 이들이 인과 그래프를 이해할 수 있는지를 조사하는 연구가 처음으로 수행되었습니다. 연구팀은 인과 그래프 이해를 평가하기 위한 프레임워크와 새로운 벤치마크인 CLEAR를 개발했습니다. 이를 통해 여섯 개의 선도적인 언어 모델을 기반으로 다양한 실험을 수행했으며, 여러 가지 실증적인 발견을 요약했습니다.

- **Technical Details**: 연구는 언어 모델의 인과 그래프 이해를 평가하기 위해 네 가지 기준을 제시합니다: 무작위 추측을 초과하는 성능, 질문 유형에 대한 강인성, 인과 정의의 올바른 활용, 그리고 과제 의존성에 따른 성능 제한. CLEAR 벤치마크는 세 가지 복잡도 수준과 20가지 인과 그래프 기반 과제를 포함하고 있습니다. 모델의 성능을 측정하기 위해 여섯 개의 선도적인 모델과 네 가지 프로토콜(예: 인컨텍스트 학습)을 사용했습니다.

- **Performance Highlights**: 주요 발견점은 다음과 같습니다: 언어 모델은 인과 그래프 기반 과제에서 고르지 않은 성능을 보이며 특정 영역에서 약점을 드러냈습니다. 모델은 인과 그래프에 대한 초보적인 이해를 보여주었고, 질문 유형에 민감하게 반응했습니다. 대부분의 모델은 과제 의존성에 따른 성능 제한을 받지 않았으며, 이는 모델 간 지식 표현과 응용에서의 이질성을 나타낼 수 있습니다.



### Data Augmentation of Multi-turn Psychological Dialogue via Knowledge-driven Progressive Thought Prompting (https://arxiv.org/abs/2406.16567)
- **What's New**: 본 논문에서는 심리 대화 도메인에서 다회차 대화 데이터를 늘려 성능을 향상시키기 위한 새로운 방법론을 제안합니다. 이는 기존 연구가 주로 발화 수준(Dialogue Utterance-level)의 대화 데이터 증가에 초점을 맞춘 것과 달리, 다회차(Multi-turn) 대화에서 역사적 맥락을 고려한 데이터를 생성함으로써, 더 나은 성능을 발휘합니다.

- **Technical Details**: 제안된 방법론인 Knowledge-driven Progressive Thought (KPT) 프롬팅 방법에는 세 가지 주요 구성 요소가 포함됩니다: (1) 진행 생각 생성기(Progressive Thought Generator), (2) 심리 지식 생성기(Psychological Knowledge Generator), (3) 다회차 대화 생성기(Multi-turn Dialogue Generator). 진행 생각 생성기는 대화의 의미 편차를 최소화하기 위해 연속적인 생각을 생성하며, 심리 지식 생성기는 대화 히스토리를 제공하여 더 정확한 심리 대화를 생성할 수 있도록 합니다. 다회차 대화 생성기는 이러한 생각과 심리 지식을 바탕으로 다회차 대화를 생성합니다.

- **Performance Highlights**: 세 개의 심리 대화 관련 데이터셋을 사용한 광범위한 실험을 통해 제안된 KPT 방법의 유효성이 입증되었습니다. 이는 기존의 발화 수준 데이터 증가 기법과 비교해도 우수한 성능을 보였으며, 주어진 심리 대화 도메인에서의 작은 모델도 더 나은 성능을 발휘하도록 했습니다.



### Are there identifiable structural parts in the sentence embedding whole? (https://arxiv.org/abs/2406.16563)
Comments:
          17 pages, 14 figures, 5 tables

- **What's New**: 이 연구는 트랜스포머 모델이 생성하는 문장 임베딩이 여러 층(layer)으로 구성된 정보를 포함하고 있는지, 그리고 이 정보를 어떻게 구분할 수 있는지를 탐구합니다. 문장 임베딩이 명사구, 동사구, 전치사구 같은 문법 구조와 의미적 역할 정보를 포함하고 있는지 조사했습니다.

- **Technical Details**: 본 연구에서는 트랜스포머 모델에서 생성된 문장의 고정 길이 벡터인 임베딩을 분석합니다. 이 임베딩들이 문법적, 의미적 정보를 여러 층(layer)으로 중첩하여 포함하고 있다는 가설을 세우고, 이를 검증하기 위해 정해진 청크(chunk) 구조를 가진 문장을 이용했습니다. 층(layer)은 '합성된 정보의 층'을 의미하며, 컨볼루션 신경망(CNN)을 사용하여 이 정보를 분리했습니다.

- **Performance Highlights**: 분석 결과, 문장 임베딩이 문법적 수 및 의미적 역할에 관한 정보를 포함하고 있으며, 이를 통해 모델의 내부 표현 방식과 학습 중 성능 변화를 이해하는 데 기여할 수 있다는 점을 확인했습니다. 이를 통해 모델의 출력 오류 원인을 좁힐 수 있고, 학습 데이터의 효율적 사용을 극대화할 수 있는 가능성을 제시합니다.



### LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training (https://arxiv.org/abs/2406.16554)
- **What's New**: 최근 Mixture-of-Experts(MoE) 프레임워크가 대규모 언어 모델(LLM)의 성능을 확장할 유망한 방법으로 주목받고 있습니다. 그러나 대규모 환경에서 MoE를 처음부터 학습하는 과정은 데이터 부족과 불안정성의 문제를 겪고 있습니다. 이를 해결하기 위해 기존의 밀집 언어 모델(dense LLM)에서 MoE 모델을 구축하는 방법을 탐구했습니다. 구체적으로 LLaMA-2 7B 모델을 기반으로, 원래 피드포워드 네트워크(FFN)의 파라미터를 여러 전문가로 분할하고, 변형된 MoE 모델과 추가 게이트 네트워크를 계속 사전학습(continual pre-training)합니다.

- **Technical Details**: 이번 연구에서는 전문가(expert)를 효과적으로 구성하는 방법과 지속적인 사전학습을 위한 다양한 데이터 샘플링 전략을 포괄적으로 탐구합니다. 먼저, FFN 파라미터를 여러 전문가로 분할하는 'Expert Construction' 과정을 통해 MoE 모델을 만들었습니다. 이어서 변형된 MoE 모델과 게이트 네트워크를 추가 학습하는 'Continual Pre-training' 과정을 거쳤습니다. 이 단계에서 빠른 수렴과 성능 향상을 얻기 위해 동적 및 정적 데이터 샘플링 전략을 신중하게 연구하였습니다.

- **Performance Highlights**: 200B 토큰을 학습한 결과, LLaMA-MoE-3.5B 모델은 유사한 활성화 파라미터를 포함한 밀집 모델(dense models)을 크게 능가했습니다. 실험 결과, 제안한 LLaMA-MoE 시리즈 모델의 효율성을 다양한 작업에서 검증하였습니다. 또한 모든 모델 구축 과정과 학습 데이터는 투명하게 공개됩니다.



### C-LLM: Learn to Check Chinese Spelling Errors Character by Character (https://arxiv.org/abs/2406.16536)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 성능을 향상시키기 위해 중국어 철자 검사(Chinese Spell Checking, CSC)에서 자주 발생하는 한계점을 발견하고 이를 해결하는 방법을 제안했습니다. 기존의 LLM이 중국어 철자 검사의 문자 수준 제약 조건을 충족하지 못함으로써 성능 병목 현상이 발생한다는 점을 발견했습니다. 이러한 문제를 해결하기 위해 문자 단위로 오류를 검사하는 C-LLM이라는 새로운 방법을 제안했습니다.

- **Technical Details**: 기존의 LLM은 혼합한 문자-단어 토크나이제이션(tokenization)을 사용하여 문자 수준 정렬을 학습하는 데 어려움을 겪습니다. 그 결과, LLM이 문자 수준 제약 조건을 만족시키지 못하고 이러한 이유로 인해 철자 교정 성능이 저하됩니다. C-LLM은 문자 단위로 토크나이제이션을 구성하고, 새로운 어휘에 적응하기 위해 모델을 지속적으로 훈련시킵니다. 더 나아가서, CSC 데이터 세트에서 감독된 fine-tuning을 수행하여 모델이 CSC 작업을 학습하도록 합니다.

- **Performance Highlights**: 제안된 C-LLM 방법은 두 개의 CSC 벤치마크에서 평균 10% 이상의 성능 향상을 달성했습니다. 특히 일반 시나리오에서 2.1% 향상, 특정 도메인 시나리오에서 12%의 큰 향상을 보였으며, 이는 최신 성능을 기록했습니다. 이러한 결과는 C-LLM이 기존의 방법을 뛰어넘는 성능을 제공하며, 향후 오류 교정 모델의 설계에 중요한 통찰력을 제공할 수 있음을 보여줍니다.



### Token-based Decision Criteria Are Suboptimal in In-context Learning (https://arxiv.org/abs/2406.16535)
Comments:
          21 pages, 14 figures, 8 tables

- **What's New**: In-Context Learning(ICL)에서 전통적으로 사용되던 토큰 확률에 기반한 분류 기준이 최적의 결정 경계를 제공하지 못한다고 지적하며, 이를 해결하기 위해 '히든 캘리브레이션'을 제안합니다. 이 방법은 토큰 확률을 버리고 언어 모델(LM)의 마지막 히든 상태를 활용한 최근접 센트로이드 분류(nearest centroid classifier)를 사용합니다.

- **Technical Details**: 기존의 ICL은 수동으로 선택된 레이블 토큰의 확률에 기반해 예측을 수행합니다. 그러나 이로 인해 예측 확률이 편향되거나 과소 캘리브레이션되어 성능이 저하된다는 문제가 있습니다. 이를 해결하기 위해 '히든 캘리브레이션'은 마지막 히든 상태를 활용해, 몇 샷 캘리브레이션 셋(calibration set)에서 관찰된 센트로이드를 기준으로 최근접 센트로이드 분류를 사용합니다. 즉, 표본을 입력하여 마지막 히든 상태를 추출한 후, 해당 상태의 센트로이드를 계산하여 가장 가까운 센트로이드의 카테고리를 예측 라벨로 할당합니다.

- **Performance Highlights**: 10개의 분류 데이터 셋 및 3개의 모던 언어 모델에서 실험한 결과, '히든 캘리브레이션'은 현재 토큰 기반 캘리브레이션을 약 20%가량 상회하는 성능을 보였습니다. 이 방법은 기존의 캘리브레이션 방법과 동등한 계산 비용을 가지면서도 일관되게 더 나은 성능을 보여줍니다.



### Towards Better Graph-based Cross-document Relation Extraction via Non-bridge Entity Enhancement and Prediction Debiasing (https://arxiv.org/abs/2406.16529)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이번 논문에서는 Cross-document Relation Extraction(CDRE) 문제 해결을 위해 새로운 모델을 제안했습니다. 기존 연구는 주로 연결된 엔티티를 통해 관계를 예측했으나, 본 연구는 연결되지 않은 엔티티(non-bridge entities)도 관계 예측에 기여할 수 있음을 강조하고 있습니다. 또한, 자주 사용되는 CodRED 데이터셋이 상당량의 NA 인스턴스를 포함하여 예측 시 편향이 발생하는 문제를 해결하려고 했습니다.

- **Technical Details**: 제안된 모델은 그래프 기반의 CDRE 모델로, 연결되지 않은 엔티티를 포함한 통일된 엔티티 그래프를 사용합니다. 각 노드는 BERT 표현으로 초기화되고, 그래프 재발 신경망(Graph Recurrent Network, GRN)을 통해 인코딩됩니다. 관계 예측 시 편향을 교정하기 위해 두 가지 새로운 예측 분포를 소개하고 이를 통해 원래 예측 분포를 보정합니다. 구체적으로, 하나의 예측 분포는 NA 인스턴스를 제외하고 새로 훈련된 분류기로부터 얻어지며, 다른 하나는 가장 중요한 비타겟 엔티티를 마스킹한 분포입니다.

- **Performance Highlights**: 제안된 모델은 CodRED 데이터셋의 폐쇄 및 개방 설정에서 GPT-3.5-turbo와 InstructUIE를 포함한 모든 베이스라인 모델을 뛰어넘어 최첨단 성능을 달성했습니다. 특히, 공식 리더보드에서 66.23%와 55.87%의 AUC 포인트를 기록하며 2023년 12월 이후 모든 제출물 중 1위를 차지했습니다.



### Evaluating the Ability of Large Language Models to Reason about Cardinal Directions (https://arxiv.org/abs/2406.16528)
Comments:
          9 pages, 3 figures, 1 table. Short paper accepted by COSIT 24, The 16th Conference on Spatial Information Theory

- **What's New**: 이 논문은 대표적인 대형 언어 모델(LLMs)이 기본 방향(CDs)을 이해하고 추론하는 능력을 조사합니다. 이를 위해 두 가지 데이터셋이 제작되었으며, 하나는 ChatGPT와 함께 공동으로 생성된 간단한 질문들로 구성되어 있으며, 다른 하나는 템플릿을 사용하여 각 시나리오에 대해 올바른 방향을 결정하는 복잡한 질문들을 포함하고 있습니다.

- **Technical Details**: 연구진은 템플릿 기반의 질문 생성 방식을 통해 LLMs가 특정 시나리오에서 올바른 기본 방향을 결정하는 능력을 테스트했습니다. 질문 템플릿에는 다양한 이동 방법(예: 걷기, 자전거 타기, 운전 등)과 일인칭, 이인칭, 삼인칭 시점에서 설정된 질문들이 포함되고, 이를 통해 총 5760개의 질문이 생성되었습니다. 실험은 zero-shot 방식으로 진행되었으며, 시스템 프롬프트로 '다양한 방향에 대한 질문이 주어질 것이며, 답변은 북(North), 남(South), 동(East), 서(West), 북동(North-East), 북서(North-West), 남동(South-East), 남서(South-West) 중 하나임'을 명시했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 간단한 질문 세트에서는 높은 정확도를 보였지만, 복잡한 질문 세트에서는 온도를 0으로 설정하여도 일관성 없이 성능이 저하되었습니다. 이는 LLMs가 실질적인 시나리오에서 방향을 추론하는 데 있어 아직 한계가 있음을 시사합니다.



### SyROCCo: Enhancing Systematic Reviews using Machine Learning (https://arxiv.org/abs/2406.16527)
Comments:
          28 pages, 5 figures. To appear in Data & Policy journal

- **What's New**: 이 연구는 기계 학습(ML) 기술을 사용하여 체계적인 검토 프로세스를 개선하는 방법을 탐구합니다. ML은 이미 '스크리닝(screening)' 단계에서 신뢰할 수 있는 논문을 선별하는 데 사용되고 있습니다. 그러나 데이터 추출(data extraction) 및 증거 매핑(evidence mapping)과 같은 후속 단계에서도 ML 적용을 시도한 것은 이번이 처음입니다. 연구팀은 1,952개의 '성과 기반 계약(outcomes-based contracting)' 관련 출판물을 분석하고 프로파일링하는 도구를 개발했습니다. 이 도구는 출판물을 '정책 영역(policy area)' 범주로 분류하고, 조직, 법률, 지리 정보 등 주요 정보를 추출하며, 기존 데이터셋과 연결하는 등의 작업을 수행합니다.

- **Technical Details**: 개발된 도구는 주제 영역별 그룹화를 수행하고, 증거 기반(evidence base)을 기존 데이터셋과 연결하는 기능을 포함합니다. 또한, 주제 콘텐츠를 공유할 가능성이 있는 논문의 하위 그룹을 식별할 수 있습니다. 이러한 기능을 활용하여 인터랙티브 도구와 공개 데이터셋을 만들었습니다. 이를 통해 체계적인 검토 과정에서 증거 접근성과 분석 효율성을 높였습니다.

- **Performance Highlights**: ML 기법을 사용한 결과, 체계적인 검토 과정에서 큰 효율성을 제공할 가능성이 확인되었습니다. 이는 정책 결정자와 실무자가 증거에 접근하기 쉽게 만들어 줄 수 있습니다. 그러나, ML 기술의 현재 제한점을 고려하여 주의를 기울여야 하며, 특히 오류와 편향의 가능성에 대한 경계를 늦추지 않아야 합니다.



### The Privileged Students: On the Value of Initialization in Multilingual Knowledge Distillation (https://arxiv.org/abs/2406.16524)
Comments:
          8 pages

- **What's New**: 이 논문은 Knowledge Distillation(KD)이 다중언어(multilingual) 환경에서도 유효함을 제시하고, 특히 모델 초기화(model initialization)가 중요한 역할을 한다는 것을 발견했습니다. 제안된 방법은 교사 모델(teacher model)에서 학생 모델(student model)로 가중치를 직접 복사(copying weights)하는 방식으로 초기화를 강화하는 것입니다.

- **Technical Details**: KD기법을 사용하여 대형모델(teacher model)에서 소형모델(student model)로 지식을 전이시킵니다. 이 과정에서 학생 모델은 교사 모델의 출력을 모방하도록 훈련되며, 이를 통해 성능을 유지하면서도 추론 과정에서의 계산 비용을 줄일 수 있습니다. 본 연구에서는 영어, 스페인어, 프랑스어 등의 다중언어 데이터셋을 사용하여 교사 모델에서 가중치를 복사해 초기화함으로써 KD의 효율성을 높이는 방법을 탐구했습니다.

- **Performance Highlights**: 비용이 많이 드는 초기 훈련 단계를 생략하고도, 교사 모델의 가중치를 복사함으로써 초기화를 향상시키는 방법이 다중언어 환경에서 더욱 빠른 학습 속도와 높은 성능을 보였습니다. 또한, 이 방법은 저자원 상황에서도 다중언어 능력을 유지하는 데 유리함을 실험적으로 증명했습니다.



### Carrot and Stick: Inducing Self-Motivation with Positive & Negative Feedback (https://arxiv.org/abs/2406.16521)
Comments:
          10 pages, 8 figures

- **What's New**: 이번 연구에서는 자기동기(self-motivation)를 촉진할 수 있는 새로운 데이터셋 'CASTIC'을 제안합니다. 이 데이터셋은 12,590개의 문장으로 구성되며, 자기동기 강화를 위한 5가지 전략을 포함합니다. 이는 기존의 연구들이 긍정적인 언어에 초점을 둔 것과 다르게, 부정적 피드백도 함께 활용하여 자기동기를 자극하는 새로운 접근법을 제안합니다.

- **Technical Details**: CASTIC 데이터셋은 다음과 같이 생성되었습니다: 첫 번째로, 문장을 통해 목표(TODO)와 장애물(Obstacle)을 추출합니다. 그런 다음, 장애물에 따라 심각도 점수(Severity Score)를 부여하여 피드백을 생성합니다. 최종 피드백(Final Feedback)은 Maslow의 '5가지 필요' 이론을 기반으로 하여 생성됩니다. 데이터셋 생성을 위해 ORION-14B-Chat 모델을 사용했으며, BART-L, GPT-2, M2M-100, T5 등의 모델로 데이터를 평가하였습니다.

- **Performance Highlights**: CASTIC 데이터셋의 평가에는 BLEU, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore 등이 사용되었습니다. 평가는 양적/질적 접근 방식을 통해 이루어졌으며, GPT-3.5 모델을 이용하여 데이터셋이 진정으로 자기동기를 유도하는지 평가하였습니다. 그 결과, 모델이 각 동기 전략을 잘 학습하고 피드백을 생성하는 데 효과적임을 확인했습니다.



### Large Vocabulary Size Improves Large Language Models (https://arxiv.org/abs/2406.16508)
Comments:
          Work in progress

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 성능과 어휘 크기 간의 관계를 실증적으로 조사하고, 새로운 언어로 지속적인 훈련을 진행할 때 사전 정의된 어휘 대신 새로운 어휘를 사용하는 방법을 제안합니다.

- **Technical Details**: Byte-Pair Encoding(BPE)와 유니그램 언어 모델(Kudo)을 사용하여 어휘를 구성했습니다. 두 개의 언어(영어와 일본어)에 대해 각기 다른 어휘 크기(5k, 10k, 50k, 100k, 500k)에 대한 실험을 수행했습니다. GPT-3 La와 같은 하이퍼 파라미터 설정을 사용했습니다.

- **Performance Highlights**: 실험 결과, 더 큰 어휘 크기가 LLMs의 성능을 향상시키는 것으로 나타났습니다. 특히 100k 및 500k 어휘 크기 모델이 효율적인 훈련을 제공하면서도 고성능을 유지했습니다. 또한, 지속적인 훈련 시 새로운 어휘를 사용하는 것이 사전 훈련된 기존 어휘를 사용하는 것보다 더 나은 성능을 제시했습니다.



### OTCE: Hybrid SSM and Attention with Cross Domain Mixture of Experts to construct Observer-Thinker-Conceiver-Expresser (https://arxiv.org/abs/2406.16495)
- **What's New**: 최근 연구에 따르면 선택적 상태 공간(Selective State Space)과 Transformer 아키텍처를 결합한 Mamba 모델이 언어 모델링 작업에서 단독으로 사용될 때보다 더 우수한 성능을 보이는 것으로 확인되었습니다. 특히, 선택적 상태 공간의 장기 의존성 문제를 완화할 수 있는 이중성(self-attention) 메커니즘을 추가하여 성능을 극대화합니다. 이를 위해 우리는 선택적 상태 공간 모델을 이중성 주의 메커니즘과 연결하고, 두 아키텍처를 교차 도메인 전문가(Hybrid Experts)와 결합한 새로운 방법을 제안합니다.

- **Technical Details**: 새로운 모델은 '위치 정보 주입(Position Information Injection)' 방식을 통해 선택적 상태 공간 모델과 이중성 주의 메커니즘을 연결하여, 이 두 가지 장점을 모두 활용합니다. 이 모델은 관찰자-사상자-구상자-표현자(Observer-Thinker-Conceiver-Expresser, OTCE) 아키텍처를 설계하여 자연의 정보 처리 과정을 모방합니다. 모델은 선택적 상태 공간, 이중성 주의 메커니즘, 선형 상태 업데이트 메커니즘 및 컨텍스트 인식 상태 정보를 결합하여 효율적이고 효과적인 언어 모델링을 목표로 합니다.

- **Performance Highlights**: 여러 실험 결과, OTCE 아키텍처는 의미 유사도 평가, 장단문 분류, 자연 언어 추론, 키워드 인식, 도메인 선택 작업, 컨텍스트 학습, 다중 쿼리 회상 작업 등 다양한 언어 작업에서 높은 성능을 보여주었습니다. 특히, 복잡한 다중 쿼리 연관 회상 작업에서는 더 큰 규모의 Mamba 나 Transformer 모델보다 우수한 성과를 기록했습니다.



### eagerlearners at SemEval2024 Task 5: The Legal Argument Reasoning Task in Civil Procedur (https://arxiv.org/abs/2406.16490)
- **What's New**: 이번 연구는 세 가지 주요 GPT 대형 언어 모델(Large Language Models, LLM), 큰 입력 토큰 크기를 가진 두 모델, 그리고 법적 데이터로 사전학습된 두 모델을 사용하여 제로샷(zero-shot) 분류 방법을 탐구합니다. 연구의 주요 데이터셋은 미국 민사절차(Domain of U.S. Civil Procedure) 분야에서 수집되었으며, 법학 학생들에게 제공되는 책에서 요약된 법적 사례, 특정 질문, 가능한 답변 및 설명을 포함하고 있습니다.

- **Technical Details**: 데이터셋의 샘플은 질문, 답변, 라벨, 분석 및 설명을 포함합니다. 연구는 Longformer와 Big Bird, Legal-RoBERTa와 Legal-XLM-RoBERTa 모델을 사용하여 법적 데이터를 분석하고, OpenAI의 GPT 3.5, Google's Gemini 및 Bing의 Copilot 모델의 제로샷 성능을 비교합니다. 또한, 클래스 불균형 문제를 해결하기 위해 Focal Loss 함수를 사용했습니다.

- **Performance Highlights**: 법적 데이터셋의 복잡성을 처리하는 데 있어 LLM 모델들의 성능을 평가한 결과, 최상의 F1 스코어로 64%를 얻었습니다. 이를 통해 대형 언어 모델들이 법적 데이터를 이해하고 분석하는 데 높은 유효성을 보였습니다.



### Deepfake tweets automatic detection (https://arxiv.org/abs/2406.16489)
- **What's New**: 이번 연구는 DeepFake 트윗을 검출하는 데 초점을 맞추고 있습니다. 이를 위해 고급 자연어 처리(NLP) 기법을 활용하여 진짜 콘텐츠와 AI 생성 텍스트를 구별합니다. 우리가 이용한 TweepFake 데이터셋은 다양한 머신러닝 모델을 훈련 및 평가하는 데 사용되었습니다. 본 연구는 AI 생성 허위 콘텐츠를 인식하여 디지털 커뮤니케이션의 신뢰성을 높이는 데 기여하고자 합니다.

- **Technical Details**: 이번 연구는 DeepFake 트윗을 탐지하기 위해 자연어 처리(NLP) 기술을 활용하였습니다. TweepFake 데이터셋과 GPT-2 생성 텍스트를 사용하여 다양한 텍스트 표현 방식과 전처리 방법을 평가했습니다. 토큰화, 불용어 제거, 어간 추출 등의 전처리 단계를 꼼꼼히 수행하며, TF-IDF와 BERT 임베딩을 활용한 머신러닝, 딥러닝, transformer 모델들을 실험했습니다. 사용된 모델에는 LightGBM, XGBoost, Random Forest, Logistic Regression, SVM, CNN, GRU, CNN+GRU, xlm-roberta-base, distilbert-base-uncased, GPT-2가 포함됩니다.

- **Performance Highlights**: 이번 연구에서는 다양한 모델을 평가한 결과, ROBERTA 모델이 최고 성능을 보였습니다. ROBERTA 모델은 조정되지 않은 TweepFake 데이터셋을 활용했을 때 가장 높은 균형 정확도와 F1 점수를 기록했습니다. 이를 통해 DeepFake 트윗을 탐지하기 위한 가장 효과적인 전략을 식별할 수 있었습니다.



### EMMI -- Empathic Multimodal Motivational Interviews Dataset: Analyses and Annotations (https://arxiv.org/abs/2406.16478)
Comments:
          9 pages

- **What's New**: 최근의 연구는 치료에서의 다중모달 상호작용(multimodal interaction)을 분석하여 비대면 상담을 지원하는 가상 상담 에이전트(multimodal virtual agent)를 개발하는데 주목하고 있습니다. 이 연구는 치료사의 사회적 목표와 임무 목표를 어떻게 조화시키는지에 대한 통찰을 제공합니다. 논문에서는 실제 상담 세션을 모사한 대화 데이터를 다중모달로 주석화하고 분석하여 EMMI를 소개합니다.

- **Technical Details**: EMMI는 AnnoMI와 Motivational Interviewing Dataset(MID) 두 개의 공공 MI 코퍼스를 결합하여 다중모달 주석을 추가한 데이터셋입니다. 연구는 비디오 기반의 세션을 분석하여, 치료사의 공감적 행동(empathic behavior)과 환자의 반응을 관찰합니다. 세 가지 환자 유형을 발견하였고, 이에 따라 치료사의 행동이 어떻게 달라지는지를 분석하였습니다.

- **Performance Highlights**: 치료사는 환자의 유형에 따라 말과 비언어적 행동을 조절하는 것이 관찰되었습니다. 세 가지 환자 유형은 대화 패턴에서 중요한 차이를 보였습니다. 이는 환자가 상호작용 중 나타내는 반응에 따라 치료사의 접근 방식을 맞춤화할 필요성을 시사합니다. 연구 결과는 이러한 다중모달 상호작용의 분석이 향상된 가상 상담 시스템을 개발하는데 기여할 수 있음을 보여줍니다.



### Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration (https://arxiv.org/abs/2406.16469)
- **What's New**: 새로운 연구는 문화적으로 포괄적인 Vision-Language 모델(VLM)을 만들기 위해, 문화적 요소를 반영한 질문에 응답하는 모델의 능력을 진단할 수 있는 테스트 벤치마크의 필요성을 강조합니다. 기존 연구들은 수작업에 의존해 다양성과 효율성이 떨어진다는 문제점을 지적하며, 이를 개선하기 위해 인간-VLM 협업을 활용한 반자동화 파이프라인을 제안합니다. 이 파이프라인은 한국 문화를 대상으로 한 K-Viscuit라는 데이터셋을 통해 그 효과를 입증합니다.

- **Technical Details**: 제안된 파이프라인은 문화적 요소가 포함된 비주얼 퀘스천 앤서링(VQA)을 생성하기 위해 인간-VLM 협업을 이용합니다. VLM이 가이드라인, 사람의 주석 예시, 이미지 관련 지식을 기반으로 질문을 생성하고, 이를 원어민이 검토하여 품질과 문화적 관련성을 보장합니다. K-Viscuit 데이터셋은 두 가지 유형의 질문을 포함하며, 시각적 인식 능력과 세부적인 시각적 추론 능력을 평가합니다.

- **Performance Highlights**: K-Viscuit를 사용해 오픈소스와 상용 VLM의 성능을 평가한 결과, 오픈소스 모델이 한국 문화를 이해하는 데 있어서 상용 모델에 비해 현저히 뒤처진다는 것을 발견했습니다. 이는 VLM의 문화적 해석 능력을 향상시킬 수 있는 여러 개선 지점을 강조하며, 외부 지식 검색을 결합하는 가능성을 탐구하여 이 해석 능력을 더 강화할 수 있는 방법을 제안합니다. 연구 데이터셋과 코드는 공개될 예정입니다.



### InterCLIP-MEP: Interactive CLIP and Memory-Enhanced Predictor for Multi-modal Sarcasm Detection (https://arxiv.org/abs/2406.16464)
Comments:
          8 pages, 6 figures, 6 tables

- **What's New**: 새로운 연구인 InterCLIP-MEP는 사회적 미디어에서 흔히 나타나는 텍스트 이미지 조합의 풍자(사르카즘) 검출 문제를 해결하기 위해 고안되었습니다. 이는 크로스 모달리티(cross-modality) 정보를 각 인코더에 내장하여 샘플 표현을 개선한 InterCLIP를 기반으로 합니다. 또한, 메모리 강화 예측기(MEP)를 적용하는 새로운 훈련 전략을 도입해 테스트 샘플의 역사적 지식을 저장하고 이를 비파라매트릭(non-parametric) 분류기로 사용하여 최종 예측을 도출합니다. InterCLIP-MEP는 최신 MMSD2.0 벤치마크에서 최첨단 성능을 기록했습니다.

- **Technical Details**: InterCLIP-MEP는 강화된 CLIP인 Interactive CLIP(InterCLIP)를 백본(backbone)으로 사용합니다. 이 모델은 텍스트와 이미지를 인코딩하는 동안 크로스 모달리티 정보를 내장하여 멀티모달 상호작용을 더 잘 이해합니다. MEP는 동적 이중 채널 메모리를 사용하여 역사적 샘플 지식을 저장하고 이를 비파라매트릭 분류기로 사용하여 최종 예측을 도출합니다. 또한, InterCLIP의 텍스트와 비전 인코더의 셀프 어텐션 모듈 가중치를 LoRA(LoRA) 기법으로 미세 조정하여 초기 CLIP 가중치를 기반으로 개선합니다.

- **Performance Highlights**: InterCLIP-MEP는 MMSD2.0 벤치마크에서 최첨단 성능을 보여주며, 높은 수준의 풍자 감지 정확성을 입증했습니다. 동적 이중 채널 메모리를 사용한 MEP는 샘플 표현을 더욱 정밀하게 캡처하여 신뢰할 수 있는 예측을 가능하게 합니다.



### Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers (https://arxiv.org/abs/2406.16450)
- **What's New**: 최신 대규모 언어 모델(LLMs)은 성능을 위해 대규모 파라미터에 의존하지만 이는 계산 비용이 많이 들게 됩니다. 이번 연구는 파라미터 수와 계산 비용을 줄이면서도 성능 저하를 최소화하려는 시도를 했습니다. 특히 Transformer 기반의 LLM에서 계산 집약적인 피드포워드 네트워크(FFN)를 연구 대상으로 했습니다.

- **Technical Details**: 이 연구는 효율적인 저차원(LowRank) 및 블록-대각(BlockDiagonal) 행렬을 결합하여 FFN의 세 가지 선형 계층 근사를 고려합니다. 기존의 많은 연구와 달리 우리는 i) 초기부터 훈련하는 관점에서 이 구조를 탐구하고, ii) 1.3억 파라미터 규모로 확장하며, iii) 최근의 Transformer 기반 LLM에서 이 구조를 시험했습니다. 새로운 'self-guided training'이라는 훈련 방식을 도입하여 초기화에서 오는 불안정한 훈련 동력을 개선했습니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 훈련 및 추론에서 효율적이며 효과적이라는 것이 입증되었습니다. 특별히, 32%의 FFN 파라미터만으로도 2.5배 빠른 속도 증가를 달성했으며, 동일한 훈련 FLOPs에서 단지 0.4의 Perplexity 증가를 보였습니다. 또한, 넓고 구조화된 네트워크를 통해 기존의 중간 크기 및 대형 Transformer를 Perplexity와 처리량 성능 면에서 능가했습니다.



### UniCoder: Scaling Code Large Language Model via Universal Cod (https://arxiv.org/abs/2406.16441)
Comments:
          Accepted by ACL 2024 (Main)

- **What's New**: 최근 연구에서는 체인-오브-생각(Chain-of-Thought, CoT)과 같이 중간 자연어 추론 단계를 도입하여 코드 생성 작업을 개선하고자 노력해 왔습니다. 그러나 이 방법은 코드 번역이나 생성 작업에는 적합하지 않습니다. 이 논문에서는 알고리즘 단계를 표현하는 '유니버설 코드(Universal Code, UniCode)'를 중간 표현으로 도입하였습니다. 이 유니버설 코드는 다양한 프로그래밍 언어의 관례들을 혼합하여 알고리즘 단계를 설명합니다.

- **Technical Details**: 유니코더 인스트럭트(UniCoder-Instruct)라는 데이터셋을 수집하여 여러 작업 학습 목표에 따라 모델을 훈련시켰습니다. 이 데이터셋은 자연어 질문, 코드 솔루션, 해당 유니버설 코드를 포함하고 있습니다. 유니코더 모델은 질문-대답 생성, 질문-유니코드 생성, 유니코드-솔루션 번역 등의 다중 작업 학습 목표를 달성하기 위해 조정되었습니다. 특히 UniCode의 문법 규칙을 지정하고, GPT-4를 통해 UniCoder-Instruct 데이터셋을 생성하는 과정이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과 유니코더는 기존의 프롬프팅 방법보다 큰 폭으로 성능이 향상됨을 보여줍니다. 특히 Python 벤치마크(Humaneval, MBPP)와 다국어 벤치마크(MultiPL-E)에서 탁월한 성능을 입증하였습니다. 결과적으로 유니코더는 모든 언어에서 일관되게 최첨단 성능을 나타냈으며, 제안된 방법의 효율성을 검증하기 위한 추가적인 연구도 진행되었습니다.



### Multilingual Knowledge Editing with Language-Agnostic Factual Neurons (https://arxiv.org/abs/2406.16416)
Comments:
          12 pages, 4 figures, 7 tables

- **What's New**: 이번 연구는 다국어 지식 편집(Multilingual Knowledge Editing, MKE)의 성능을 향상시키기 위한 새로운 방법을 제안합니다. 연구진은 대형 언어 모델(LLMs)에서 다국어 지식이 공유되는 뉴런 집합을 발견했습니다. 이를 '언어 무관 사실 뉴런(Language-Agnostic Factual Neurons, LAFN)'이라고 명명하고, 해당 뉴런을 수정하여 다국어 지식을 동시에 편집하는 새로운 MKE 방법을 제안했습니다.

- **Technical Details**: 연구진은 동일한 사실 지식이 여러 언어에서 일반적으로 동일한 뉴런 세트를 활성화한다는 것을 발견했습니다. 이 뉴런들은 '언어 무관 사실 뉴런(LAFN)'으로, 다국어 지식 간의 의미적 연결을 나타내며 주로 특정 계층에 위치해 있습니다. 이를 바탕으로, 우리는 해당 뉴런을 정확히 찾아내기 위해 다국어 지식의 패러프레이즈(paraphrases)를 생성을 통해 지정된 뉴런을 수정하는 업데이트 값을 최적화합니다. 이렇게 함으로써 다국어 지식을 동시에 수정할 수 있도록 했습니다.

- **Performance Highlights**: Bi-ZsRE 및 MzsRE 벤치마크에서 실험 결과, 제안된 방법은 기존의 MKE 방법들보다 뛰어난 편집 성능을 보여주었습니다. 특히, 다국어 지식 간의 내적 의미적 연관성을 고려하는 것이 매우 중요하다는 것을 입증했습니다.



### UNO Arena for Evaluating Sequential Decision-Making Capability of Large Language Models (https://arxiv.org/abs/2406.16382)
- **What's New**: 새로운 연구는 카드 게임 UNO를 기반으로 하는 UNO Arena를 통해 Large Language Models(LLMs)의 순차적 의사결정 능력을 평가합니다. 이는 순차적 의사결정이 초기 결정이 이후 결정에 영향을 미치는 동적 환경을 고려하는 알고리즘에 대한 연구입니다. UNO Arena에서는 Monte Carlo 방법 기반의 새로운 메트릭을 사용하여 동적으로 순차적 의사결정 능력을 평가합니다.

- **Technical Details**: UNO Arena에서 랜덤 플레이어, DQN 기반 강화 학습 플레이어, 그리고 LLM 플레이어(e.g., GPT-4, Gemini-pro)를 설정하여 비교 테스트를 진행하였습니다. 또한, LLM의 순차적 의사결정 능력을 향상시키기 위해 게임 히스토리와 전략 요약을 통해 자신이 내린 결정을 반성하는 TUTRI 플레이어를 제안했습니다.

- **Performance Highlights**: 다양한 실험 결과, TUTRI 플레이어는 기본 LLM 플레이어에 비해 순차적 의사결정 성능에서 상당한 개선을 보여주었습니다. 주류 LLM들(GPT-3.5, GPT-4, Gemini-pro, Llama 2, ChatGLM3)을 종합적으로 평가한 결과, GPT-4가 가장 효과적인 순차적 의사결정 능력을 보였습니다.



### On the Transformations across Reward Model, Parameter Update, and In-Context Promp (https://arxiv.org/abs/2406.16377)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 파라미터 업데이트(parameter updating), 보상 모델링(reward modeling), 및 인-컨텍스트 프롬프팅(in-context prompting)이라는 세 가지 적응 도구의 상호 교환 가능성을 제시합니다. 이러한 교환 가능성은 여섯 가지 변환 방향이 있는 삼각형 프레임워크를 형성하며 각 변환은 다양한 애플리케이션을 용이하게 합니다.

- **Technical Details**: LLMs는 대규모 코퍼스를 기반으로 사전 학습된 모델로서 특별한 세계 지식과 추론 능력을 가지고 있습니다. 그러나 실질적인 애플리케이션에 배치하기 위해 추가적인 적응이 필요합니다. 논문에서는 파라미터 업데이트, 보상 모델링, 인-컨텍스트 프롬프팅을 상호 교환 가능한 도구로 간주하며, 이를 통해 여러 다운스트림 작업에 적용할 수 있는 방법론을 설명합니다.

- **Performance Highlights**: 이 논문은 다양한 기존 연구들을 하나로 통합하는 체계적인 분석을 제공하며, 잠재적인 향후 연구 방향을 제안합니다. 또한, LLMs의 적응을 위해 플러그-앤-플레이 방식으로 이 세 가지 도구를 사용할 수 있는 새로운 방법론을 제시합니다.



### KEHRL: Learning Knowledge-Enhanced Language Representations with Hierarchical Reinforcement Learning (https://arxiv.org/abs/2406.16374)
- **What's New**: Knowledge-Enhanced Pre-trained Language Models (KEPLMs)를 위한 새로운 프레임워크가 소개되었습니다. 기존의 독립적인 지식 삽입과 통합 작업을 하나로 묶어 Hierarchical Reinforcement Learning (KEHRL)로 해결하는 접근법을 제안합니다.

- **Technical Details**: KEHRL은 강화학습 (Reinforcement Learning, RL)을 사용하여 문장에서 필요한 지식을 주입할 위치를 감지하고, 해당 위치에 적합한 지식 삼중항 (triples)을 통합합니다. 고수준 RL 에이전트는 내부 지식과 사전 지식을 사용해 중요한 위치를 감지하고, 저수준 RL 에이전트는 다의어 형태의 엔티티와 관련된 삼중항을 선택해 동적으로 조정합니다.

- **Performance Highlights**: 실험 결과 KEHRL이 사실적 지식을 효과적으로 탐구하고 다양한 자연어 이해 작업에서 성능을 향상시키는 것으로 검증되었습니다. 특히, 긴꼬리 엔티티에 지식을 삽입하는 것이 자주 등장하는 엔티티보다 더 나은 성능을 보였습니다. 또한, 동적 삼중항 주입이 고정된 삼중항 주입보다 더 높은 Spearman 상관 점수를 기록하여, 문장의 의미 보존에 효과적임을 증명했습니다.



### UniPSDA: Unsupervised Pseudo Semantic Data Augmentation for Zero-Shot Cross-Lingual Natural Language Understanding (https://arxiv.org/abs/2406.16372)
- **What's New**: Cross-lingual representation learning에서 신규 메커니즘인 Unsupervised Pseudo Semantic Data Augmentation(UniPSDA)이 제안되었습니다. 이는 인간 개입 없이 훈련 데이터를 풍부하게 하고, 다국어 자연어 이해 능력을 향상시키는 데 목적이 있습니다.

- **Technical Details**: UniPSDA는 세 단계로 구성된 순차적 클러스터링 과정을 사용하여 다국어 의미적 데이터를 증강합니다. 이 단계를 통해 단일 언어, 언어 계열 내의 여러 언어, 그리고 다양한 언어 계열 간의 토큰을 유사한 의미로 그룹화합니다. 또한, 문장 내 핵심 구성 요소를 다국어 지식으로 대체하여 컨텍스트 기반 의미를 고려합니다. 세 가지 디바이스 최적화 기법을 도입하여 편향을 해소하고 최적의 학습 속도와 안정성을 보장합니다.

- **Performance Highlights**: 종합적인 실험 결과, UniPSDA는 시퀀스 분류, 정보 추출, 질의 응답 등 일반적인 제로샷 다국어 자연어 이해 작업에서 일관되게 성능을 향상시켰습니다.



### Evaluation of Instruction-Following Ability for Large Language Models on Story-Ending Generation (https://arxiv.org/abs/2406.16356)
- **What's New**: Instruction-tuned 대형 언어 모델(LLMs)은 다양한 벤치마크 작업에서 놀라운 성능을 보여줍니다. 이번 연구에서는 이야기 결말 생성 작업에 있어서 LLMs의 지시 따르기 능력을 평가하는 데 중점을 두었습니다. 이를 위해 기계 독해(MRC) 모델을 활용하여 생성된 이야기 결말이 주어진 지시를 반영하는지 여부를 자동으로 평가하는 파이프라인을 제안합니다.

- **Technical Details**: 우리는 평가 파이프라인을 두 가지 단계로 나누었습니다: (i) 평가 데이터셋으로의 변환: 기존 Possible Stories 데이터셋에서 이야기 컨텍스트와 지시를 포함하는 입력으로부터 LLMs가 결말을 생성하는 단계, (ii) MRC 모델을 통한 평가: MRC 모델이 이야기 컨텍스트, 지시 및 후보 결말을 받아 결말이 지시를 따르는지 판단하는 단계. 이 MRC 기반 점수를 기준으로 전체 정확도를 정의하였습니다.

- **Performance Highlights**: 우리의 평가 메트릭이 인간 평가와 일치함을 확인했습니다. 또한, 최근의 오픈소스 LLMs가 우리의 자동 평가를 통해 GPT-3.5에 근접한 성능을 보여줌을 실험을 통해 입증했습니다.



### ADVSCORE: A Metric for the Evaluation and Creation of Adversarial Benchmarks (https://arxiv.org/abs/2406.16342)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2401.11185

- **What's New**: 이 논문은 AI 모델들이 인간을 혼동시키는 샘플을 제공하는 '역경 벤치마크(adversarial benchmarks)'의 유효성을 평가하기 위한 새로운 지표인 ADVSCORE를 소개했습니다. ADVSCORE는 데이터셋이 얼마나 역경적이고 변별적인지 정량화하며, 이를 기반으로 역경적인 질문 응답 데이터셋인 ADVQA를 생성했습니다.

- **Technical Details**: ADVSCORE는 항목 반응 이론(Item Response Theory, IRT) 및 선형 로지스틱 테스트 모델(Linear Logistic Test Model, LLTM)을 기반으로 하여 데이터셋의 샘플들이 인간과 모델에게 얼마나 어려운지 비교하고, 샘플이 역경적(adversarial)이 되는 특징을 분석합니다. IRT는 샘플의 난이도와 주체(인간과 모델)의 능력을 추정하는 확률 모델이며, LLTM은 각 질문의 난이도와 변별성을 특징으로 설명합니다.

- **Performance Highlights**: ADVSCORE로 생성된 ADVQA 데이터셋은 여러 영역에서 기존의 역경 벤치마크를 능가하여 여러 모델을 혼동시키는 데 성공했습니다. ADVSCORE는 9,347개의 인간 응답과 세 가지 모델의 예측을 통해 검증되었으며, 현 모델들이 겪는 취약점을 명확히 밝혔습니다. 특히, GPT-4와 같은 모델이 시간적 불일치나 현대 문화에 빠르게 변화하는 주제에 취약함을 드러냈습니다.



### EHRCon: Dataset for Checking Consistency between Unstructured Notes and Structured Tables in Electronic Health Records (https://arxiv.org/abs/2406.16341)
- **What's New**: 새로운 데이터셋과 태스크인 EHRCon을 소개합니다. 이 데이터셋은 구조화된 테이블 데이터와 비구조화된 임상 노트 간의 일관성을 보장하기 위해 설계되었습니다. 또한, CheckEHR이라는 새로운 프레임워크를 도입하여 큰 언어 모델(Large Language Models, LLM)을 이용해 EHR 데이터 간의 일관성을 검증합니다.

- **Technical Details**: EHRCon은 의료 전문가들과 협력하여 MIMIC-III EHR 데이터셋을 사용해 개발되었습니다. 이 데이터셋은 3,943개의 엔티티를 105개의 임상 노트에서 수작업으로 주석을 달아 테이블 데이터와의 일관성을 확인했습니다. 데이터셋은 원래의 MIMIC-III 스키마와 OMOP CDM 스키마를 사용한 두 가지 버전으로 제공되어 적용성과 일반화를 높였습니다. CheckEHR은 8단계의 프로세스를 거쳐 임상 노트와 데이터베이스 테이블 간의 일관성을 검사합니다.

- **Performance Highlights**: CheckEHR은 few-shot 설정에서 MIMIC-III 데이터셋에서 61.06%, OMOP에서 54.36%의 재현율을 달성했습니다. zero-shot 설정에서는 MIMIC-III에서 52.39%의 재현율을 기록했습니다.



### Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging (https://arxiv.org/abs/2406.16330)
- **What's New**: 대규모 언어 모델(LLMs)의 복잡성과 규모로 인해 자원 제약 환경에서의 배포가 어려운 문제를 해결하기 위해 새롭게 제안된 기법인 '다양체 기반 지식 정렬 및 레이어 병합 압축(MKA)'를 소개합니다. 이 접근법은 다양체 학습(manifold learning)과 정규화된 쌍별 정보 병목(NPIB) 측정을 사용하여 유사한 레이어를 병합함으로써 모델 크기를 줄이고 성능을 유지합니다.

- **Technical Details**: MKA는 다양체 학습과 레이어 병합을 결합하여 중요한 활성화 특징을 보존하며 차원을 축소합니다. 다양체 학습을 통해 레이어 간 지식을 정렬하고, 이를 기반으로 유사한 레이어 쌍을 선택해 병합합니다. 이는 후방에서 전방으로 입력-출력 유사성이 높은 레이어를 병합하여 모델 성능을 유지하면서 크기를 줄입니다. 주요 기술적 구성 요소는 다음과 같습니다: 레이어 활성화 추출, 쌍별 거리 행렬 생성, Diffusion Kernel 적용, NPIB를 통해 유사도 행렬 구축 및 레이어 병합.

- **Performance Highlights**: 다양한 벤치마크 데이터셋과 Llama-3 8B 모델을 대상으로 실험한 결과, MKA는 모델 성능을 거의 유지하면서도 기존의 가지치기(pruning) 방법을 능가하는 43.75%의 압축 비율을 달성했습니다. 구체적으로, MMLU 데이터셋에서 성능 감소는 단 2.82%에 불과했습니다. MKA와 양자화(quantization)를 결합하면 더욱 높은 압축 효율을 제공합니다.



### What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Noise-free Text-Image Corruption and Evaluation (https://arxiv.org/abs/2406.16320)
- **What's New**: 이번 연구에서는 VLMs (Vision-Language Models)의 내부 의사결정 과정을 해석하기 위해 최초로 제안된 막힘 없는 텍스트-이미지 손상 및 평가 파이프라인인 NOTICE를 소개합니다.

- **Technical Details**: NOTICE는 이미지 손상을 위한 Semantic Minimal Pairs(SMP) 프레임워크와 텍스트를 위한 대칭 토큰 대체(Symmetric Token Replacement, STR)를 포함합니다. 이 기법은 두 가지 모달리티 모두에서 의미론적으로 의미 있는 인과 중재 분석(causal mediation analysis)을 가능하게 하여 BLIP와 같은 모델의 다중모달 통합을 분석하는 데 강력한 방법을 제공합니다.

- **Performance Highlights**: 우리의 실험 결과, 중간 층 크로스 어텐션 머리(cross-attention heads)의 중요한 역할을 확인했으며, 다양한 작업 및 모달리티에서 일관되게 기여하는 '보편적 크로스 어텐션 머리' 세트를 밝혔습니다. 이들은 암묵적인 이미지 분할, 객체 억제 및 이상치 억제와 같은 고유한 기능을 수행합니다.



### Modelled Multivariate Overlap: A method for measuring vowel merger (https://arxiv.org/abs/2406.16319)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 본 논문에서는 모음 중첩(vowel overlap)을 정량화하는 새로운 방법을 소개합니다. 기존 연구에서는 다변량 측정(multivariate measures)과 불균형 데이터 및 외부 요소를 제어하는 능력 간의 긴장이 있었습니다. 본 연구는 관심 있는 모든 음향 차원을 공동으로 모델링하고 모델에서 분포를 시뮬레이션하여 불확실성 계산도 용이하게 하는 방법을 제안합니다. 이 방법은 PIN-PEN 병합을 연구하는 코퍼스 음성 데이터에서 평가되었으며, Bhattacharyya affinity를 사용하여 경험적 분포보다 결과를 크게 개선했습니다.

- **Technical Details**: 이 모델은 베이esian 선형 혼합 모형(Bayesian linear mixed-effects models)을 사용하여 구현됩니다. R의 brms와 Stan을 통해 모델을 적합시키고, MMO(Modelled Multivariate Overlap)를 계산하는 코드는 Github에서 사용 가능합니다. MMO 계산은 네 가지 단계로 이루어집니다: 1) 음성 토큰을 정규화, 2) F1과 F2를 공동으로 모델링, 3) 예측된 공동 분포 시뮬레이션, 4) 시뮬레이션된 분포에서 모음 병합 측정 계산.

- **Performance Highlights**: 본 논문은 PIN-PEN 병합의 조건을 연구하기 위해 MMO 방법을 적용했습니다. 남부 미국 영어와 아프리카계 미국 영어가 이 병합을 가지고 있다는 기존 연구에 기반하여 분석되었습니다. 연구된 네 가지 방언 중 남부 미국 영어는 PIN-PEN 병합을 나타내며, 사전 비모음 환경에서 완전히 중첩된 형태를 보였습니다.



### Does Cross-Cultural Alignment Change the Commonsense Morality of Language Models? (https://arxiv.org/abs/2406.16316)
Comments:
          The 2nd Workshop on Cross-Cultural Considerations in NLP (C3NLP) at ACL 2024

- **What's New**: 이 논문에서는 일본어 언어 모델(LMM)을 영어 리소스를 사용하여 정렬하는 전략이 일본 문화의 일반상식 도덕성에 미치는 영향을 조사했습니다. 실제로, 다국어 언어 모델을 정렬할 때 영어 선호 데이터를 그대로 사용하거나 번역하는 것이 일반적입니다. 이번 연구는 일본 문화에 맞춰 조정된 모델이 일본어 사용자의 선호를 얼마나 잘 반영하는지 평가합니다.

- **Technical Details**: 실험에서는 JCommonsenseMorality(JCM)와 ETHICS 데이터셋을 사용해 도덕적 판단을 평가했습니다. 특히, 일본어 LMM을 JCM 데이터셋으로 미세 조정한 모델과 ETHICS 데이터셋으로 미세 조정한 모델의 성능을 비교했습니다. 평가에는 Direct Preference Optimization(DPO)과 Low-Rank Adaptation(LoRA) 기법을 사용했습니다. 또한, CALM2, LLM-jp, Swallow-7B라는 세 가지 일본어 SFT 모델을 사용해 실험을 진행했습니다.

- **Performance Highlights**: 결과에 따르면, JCM 데이터셋으로 정렬된 모델이 ETHICS 데이터셋으로 정렬된 모델보다 JCM 테스트 세트에서 더 높은 정확도를 기록했습니다. 이는 언어 차이보다 문화 차이를 학습하고 일반화하는 것이 더 어려울 수 있음을 시사합니다. 반면, 영어 리소스와 다국어 보상 모델을 포함한 경우 일본어 모델의 지시 따르기 능력이 크게 향상되었습니다.



### Cascade Reward Sampling for Efficient Decoding-Time Alignmen (https://arxiv.org/abs/2406.16306)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 사람의 선호도에 맞추는 새로운 방법으로 Cascade Reward Sampling(CARDS)를 소개합니다. 기존 방식들이 텍스트 생성 시 높은 보상과 높은 확률성을 동시에 만족시키지 못하거나 상당한 계산 비용을 초래하는 문제를 해결합니다. CARDS는 이러한 문제를 해결하여 높은 보상과 높은 확률성을 보장하면서도 계산 비용을 크게 줄이는 방법입니다.

- **Technical Details**: CARDS는 불완전한 텍스트에 대한 보상 모델(RMs)의 분석에 기반하여, 높은 보상 프리픽스(prefix)가 높은 보상 완전 텍스트를 유도한다는 관찰을 활용합니다. 이 방법은 반려 표본 추출(rejection sampling)을 사용하여 작은 의미 단위(semantic segments)를 반복적으로 생성하고, LLM의 예측 불확실성에 따라 세그먼트 길이를 동적으로 결정합니다. 이를 통해 불필요한 토큰 재생성을 줄이고 보상 모델의 스코어링 횟수를 감소시킵니다.

- **Performance Highlights**: 실험 결과, CARDS는 기존 방식에 비해 생성 효율성과 정렬 평가에서 상당한 향상을 보여줬습니다. GPT-4와 Claude-3에서의 유용성 평가에서 99%의 승리-무승부율을 기록하며, 텍스트 생성 속도는 5배 빠릅니다.



### Compensate Quantization Errors: Make Weights Hierarchical to Compensate Each Other (https://arxiv.org/abs/2406.16299)
Comments:
          Efficient quantization method

- **What's New**: 최근 대형 언어 모델(LLMs)의 등장으로 인해 뛰어난 성능과 강력한 추론 능력이 주목받고 있습니다. 하지만 이런 모델의 컴퓨팅 리소스와 저장 용량의 요구가 막대하기 때문에, 양자화된 모델(fine-tuned)을 효율적으로 개선하는 기술이 필요하게 되었습니다. 이를 해결하기 위해 새로운 기술인 Learnable Singular value Increment(LSI)가 도입되었습니다.

- **Technical Details**: 기존의 양자화 방법은 주로 두 가지 큰 카테고리로 나눌 수 있습니다. 첫 번째는 Quantization-Aware Training(QAT)으로, 모델 훈련 중 양자화와 호환성을 최적화합니다. 두 번째는 Post-Training Quantization(PTQ)으로, 훈련 후 모델을 양자화합니다. LSI는 Singular Value Decomposition(SVD)를 사용하여 웨이트의 특이값을 추출하고, 이를 학습 가능한 값으로 만들어 다른 부분을 보정합니다. 이를 통해 LSI는 기존 기술들과 결합하여 다양한 양자화 설정에서 최첨단 성능을 달성할 수 있습니다.

- **Performance Highlights**: 다양한 양자화 환경에서 실험을 통해 LSI는 state-of-the-art 결과를 달성했습니다. 특히, weight-only, weight-activation, 매우 낮은 비트 시나리오에서 두드러진 성능을 보였습니다. LSI를 통해 양자화된 모델의 효율적인 미세 조정이 가능해졌으며, 적은 양의 학습 데이터로도 큰 개선을 이뤄낼 수 있었습니다.



### LangSuitE: Planning, Controlling and Interacting with Large Language Models in Embodied Text Environments (https://arxiv.org/abs/2406.16294)
- **What's New**: 이 논문에서는 LangSuitE라는 시뮬레이션 없이 동작하는 테스트베드를 소개합니다. LangSuitE는 텍스트로 기반한 6가지 대표적인 체화(embodied) 작업을 특징으로 하며, 다양한 환경에서 적응성과 커스터마이즈 가능성을 제공합니다.

- **Technical Details**: LangSuitE는 기존의 여러 시뮬레이션 엔진 없이 텍스트로 구성된 환경에서 동작합니다. 이 시스템은 '내재화된 세계 지식'을 개발하는 능력을 평가하며, 커뮤니케이션과 행동 전략을 쉽게 사용자 정의할 수 있습니다. 체화된 상태를 요약하는 새로운 '사고의 연쇄' 체계인 EmMem을 도입했으며, 이는 역사 정보를 바탕으로 동작합니다.

- **Performance Highlights**: 6가지 서로 다른 작업에 걸친 종합 벤치마크 결과는 체화된 계획의 도전과 통찰을 보여줍니다. 특히 Household 작업에서 EmMem 전략의 효과가 입증되었습니다. 이러한 LangSuitE는 언어 모델을 체화된 일반화된 인텔리전스로 구축하는 데 중요한 진전을 나타냅니다.



### Combining Supervised Learning and Reinforcement Learning for Multi-Label Classification Tasks with Partial Labels (https://arxiv.org/abs/2406.16293)
- **What's New**: MLPAC(MLPAC: Mixture Learner for Partially Annotated Classification)은 강화 학습(RL: Reinforcement Learning)의 탐색 능력과 지도 학습의 활용 능력을 결합한 새로운 프레임워크로, 부분적으로 주석이 달린 데이터셋에서 효과적으로 학습합니다. 이는 기존 방식들보다 더 일반화 될 수 있고 다양한 작업에 효과적입니다.

- **Technical Details**: MLPAC는 정책 네트워크(Policy Network)와 평가 네트워크(Critic Network)를 디자인합니다. 글로벌 보상 함수는 리콜 함수(Recall Function)를 사용하여 모든 클래스를 평가하고, 로컬 보상 함수는 개별 클래스를 평가하여 학습 과정을 안내합니다. 이는 Actor-Critic RL 알고리즘에 영감을 받아 정책 네트워크와 평가 네트워크를 반복적으로 훈련시켜 동적 보상 추정을 수행합니다.

- **Performance Highlights**: 문서 수준 관계 추출, 다중 레이블 이미지 분류, 바이너리 PU 학습 등 다양한 작업에서 실험을 수행한 결과, MLPAC 프레임워크는 효과적이고 일반화 능력이 뛰어남이 입증되었습니다. 모든 실험 결과에서 프레임워크의 유리함과 성능 향상이 두드러졌습니다.



### PlagBench: Exploring the Duality of Large Language Models in Plagiarism Generation and Detection (https://arxiv.org/abs/2406.16288)
Comments:
          9 pages

- **What's New**: 최근 대형 언어 모델(LLMs)이 학문적 무결성에 잠재적 위험을 초래할 수 있다는 연구들이 주목받고 있습니다. 이를 해결하기 위해, PlagBench라는 포괄적인 데이터셋을 소개합니다. 이 데이터셋은 세 가지 작성 도메인에서 생성된 46,500개의 합성 표절 사례를 포함하고 있으며, 세 가지 명령 조정된 LLM을 사용하여 생성되었습니다.

- **Technical Details**: PlagBench 데이터셋은 세 가지 LLMs(GPT-3.5 Turbo, GPT-4 Turbo, Llama2-70b-chat)을 사용하여 작성 도메인의 초록, 이야기, 뉴스 기사에서 직접적으로 생성된 텍스트 쌍을 포함합니다. 각 표절 유형에 대한 세부적인 자동 평가와 인간 주석을 통해 데이터셋의 품질을 보장합니다. 이 데이터를 사용하여 다섯 가지 최신 LLM과 세 가지 특화된 표절 탐지기의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, GPT-3.5 Turbo가 가장 높은 품질의 Paraphrase(패러프레이즈)와 Summary(요약)을 생성함을 발견했습니다. 또한, LLM들이 상용화된 표절 탐지기보다 성능이 우수하다는 것을 확인했습니다. 요약 표절 식별에서는 어려움을 겪었으나, LLM들이 견고한 표절 탐지 도구로 활용될 가능성을 보여주었습니다.



### Investigating the Influence of Prompt-Specific Shortcuts in AI Generated Text Detection (https://arxiv.org/abs/2406.16275)
Comments:
          19 pages, 3 figures, 13 tables, under review

- **What's New**: 이 논문은 AI 생성 텍스트(AIGT) 탐지기 개발에 있어 프롬프트(질의문)의 다양성 부족이 탐지기의 취약점을 초래할 수 있음을 분석합니다. 이를 해결하기 위해 Feedback-based Adversarial Instruction List Optimization(FAILOpt)이라는 공격 방법을 제안합니다. FAILOpt는 AIGT 탐지기를 속이기 위해 프롬프트에 의존하는 특성을 변경하도록 LLM에게 지시합니다.

- **Technical Details**: FAILOpt는 피드백 기반 적대적 지시어 리스트 최적화 기법으로, 탐지기의 성능을 떨어뜨릴 수 있는 특수 지시어 집합을 검색합니다. 실험 결과, FAILOpt를 이용한 텍스트 생성은 대상 탐지기를 효과적으로 회피할 수 있었습니다. 또한 FAILOpt를 이용한 데이터로 탐지기를 추가 학습시키면 탐지기의 로버스트네스가 향상됩니다.

- **Performance Highlights**: FAILOpt 공격은 기존 적대적 예제 기반 공격과 비교해도 뛰어난 성능을 보였으며, 다양한 생성 모델, 과제 및 공격 방법에 대해 강화된 탐지 성능을 나타냈습니다.



### One Thousand and One Pairs: A "novel" challenge for long-context language models (https://arxiv.org/abs/2406.16264)
Comments:
          preprint, 29 pages

- **What's New**: 이번 연구에서는 장문 맥락 대형 언어 모델 (LLM)이 책 한 권의 전체 내용을 바탕으로 정보 검색, 종합, 및 추론을 얼마나 잘 수행할 수 있는지를 평가하기 위해 NoCha라는 새로운 벤치마크 데이터셋을 도입했습니다. 이 데이터셋은 최근 출판된 영어 소설책 67권에 대해 사실과 거짓 주장을 포함한 1,001쌍의 최소한의 차이를 가진 페어로 구성되어 있습니다.

- **Technical Details**: NoCha 데이터셋은 최근 출판된 소설 책들에 대해 인간 독자가 작성한 참/거짓 주장으로 구성되어 있으며, 현재 사용 가능한 장문 맥락 평가 방법(예: 'needle-in-the-haystack' 벤치마크)과 달리 전체 책을 바탕으로 한 전역적인 추론이 필요합니다. 이 데이터셋은 67권의 책에서 각기 다른 1,001쌍의 페어를 포함하며, 특히 모든 거짓 주장은 참 주장과 단일 서사적 현상만 다르게 포함되어 있어 모델이 잘못된 이유로 결과를 추정하는 것을 최소화합니다.

- **Performance Highlights**: 이번 연구에서는 10개의 장문 맥락 LLM을 평가했으며, 그 결과 모든 오픈 소스 모델이 임의 추정 수준을 넘지 못한 반면, GPT-4o는 55.8%의 최고 정확도를 기록했습니다. 더 나아가, 문장 수준 검색이 필요한 쌍보다 전역 추론이 필요한 쌍에서 모델의 성능이 현저히 떨어졌으며, 모델이 자신들의 결정에 대해 생성한 설명이 맞더라도 종종 부정확함을 보여주었습니다. 또한, 광범위한 세계 구축이 필요한 공상 과학 소설에서는 성능이 더욱 저조했습니다.



### LLMs assist NLP Researchers: Critique Paper (Meta-)Reviewing (https://arxiv.org/abs/2406.16253)
- **What's New**: 본 연구는 최근 큰 언어 모델(LLMs)의 다양한 생성 작업에서의 유용성을 강조하고, 이를 NLP 연구자들이 논문 (메타)리뷰 작업에서의 효과성을 연구한 최초의 포괄적인 분석을 제공합니다. 이를 위해, 리뷰크리틱(ReviewCritique) 데이터셋을 구축하여, 인간 작성 리뷰와 LLM 생성 리뷰를 포함한 NLP 논문과 각 리뷰의 '부족' 라벨(with deficiency)을 추가로 제공합니다.

- **Technical Details**: 리뷰크리틱 데이터셋은 (i) 원본 제출된 NLP 논문과, 인간 및 LLM 생성 리뷰, (ii) NLP 전문가(Ph.D. 혹은 섹션 우수평가 경험자)에 의해 문장 단위로 '부족함' 라벨과 설명을 추가한 리뷰로 구성됩니다. 이 데이터를 통해 LLM이 리뷰어(Reviewers) 및 메타리뷰어(Metareviewers)로서의 가능성을 분석합니다. 첫째, 'LLMs as Reviewers' 연구에서는 LLM이 생성한 리뷰가 인간 작성 리뷰와 비교하여 어떤 점에서 더 부족한지 평가합니다. 둘째, 'LLMs as Metareviewers' 연구에서는 LLM이 인간 작성 리뷰에서 부족한 부분을 식별하고, 그 설명을 제공하는 능력을 평가합니다.

- **Performance Highlights**: LLM은 인간 리뷰어보다 더 많은 '부족한' 리뷰 세그먼트를 생성하고, 종종 논문에 특정되지 않고 다양성 및 건설적인 피드백이 부족한 리뷰를 작성하는 경향이 있습니다. 심지어 상위 LLM조차도 인간 전문가를 모방하는데 어려움을 겪는 것으로 나타났습니다. 우리의 기여는 (i) ReviewCritique 데이터셋, (ii) 인간과 LLM 리뷰간의 문장 수준 비교, (iii) LLM의 리뷰어 및 메타리뷰어로서의 잠재력을 분석하는 첫 연구입니다.



### Preference Tuning For Toxicity Mitigation Generalizes Across Languages (https://arxiv.org/abs/2406.16235)
- **What's New**: 다국어 대형 언어 모델(LLMs)의 '디톡스화(detoxification)'가 중요한 과제로 떠오르고 있습니다. 이 연구는 영어 데이터만을 사용해 다국어 LLM의 독성(toxicity)을 제로샷 크로스-링구얼 제너럴라이제이션(zero-shot cross-lingual generalization)으로 줄일 수 있음을 보였습니다. mGPT-1.3B와 같은 모델의 독성 생성 확률이 46.8%에서 17개 언어에서 3.9%까지 감소했습니다.

- **Technical Details**: 이 연구에서는 Direct Preference Optimization (DPO)라는 기법을 사용했습니다. DPO는 영어 데이터만으로 학습하여 다국어 LLM의 독성을 크게 감소시킵니다. 또한, 인과적 개입(causal intervention)과 활성화 분석(activation analysis)과 같은 메커니즘 해석 도구를 사용해 MLP 계층의 이중 다국어 특성(dual multilinguality)을 확인했습니다. 이는 DPO의 크로스-링구얼 제너럴라이제이션을 설명합니다.

- **Performance Highlights**: DPO 훈련 후, 모델의 독성 생성 확률이 46.8%에서 3.9%로 감소했습니다. 이 결과는 mGPT-1.3B뿐만 아니라 BLOOM, Llama3, Aya-23 등의 다른 다국어 LLM에도 적용됩니다. 또한, 이 연구는 바이링구얼 문장 검색(bilingual sentence retrieval)을 통해 DPO가 특정 언어에 얼마나 잘 일반화될 수 있는지 예측할 수 있음을 보여주었습니다.



### Multi-Objective Linguistic Control of Large Language Models (https://arxiv.org/abs/2406.16229)
- **What's New**: 대형 언어 모델(LLMs)은 많은 어려운 벤치마크 과제에서 획기적인 성과를 거두었음에도 불구하고, 출력을 간결하게 제어하는 데 한계가 있습니다. 본 논문에서는 기존 데이터로부터 LLM 출력의 여러 언어적 복잡성을 정밀하게 제어하는 방법을 연구하고자 합니다. 이를 위해 멀티컨트롤 튜닝(MCTune)을 제안하며, 이는 지상 진실 응답의 여러 언어적 복잡도 값을 입력에 제어 요소로 포함하여 지침 튜닝을 수행합니다. LLaMA2-7B 모델을 Alpaca-GPT4 및 WizardLM 데이터셋으로 미세 조정했으며, 이에 대한 평가 결과 다중 복잡도 제어 능력과 응답 품질이 크게 향상됨을 확인했습니다.

- **Technical Details**: 기존의 대형 언어 모델들은 출력의 언어적 복잡성을 정밀하게 제어하는 데 한계가 있습니다. MCTune은 지침 튜닝을 통해 여러 언어적 복잡도 값을 입력에 포함시켜 이러한 문제를 해결합니다. LLaMA2-7B 모델을 Alpaca-GPT4와 WizardLM 데이터셋을 사용하여 미세 조정하였고, 평가 결과 제어 가능성과 응답 품질이 모두 향상되었습니다. 구체적으로는, 텍스트 세그먼트에서 LFTK 패키지를 사용하여 14개의 다중 언어적 속성을 추출하고, 이를 활용해 모델의 출력을 제어합니다.

- **Performance Highlights**: 제안된 MCTune 방법은 LLMs가 여러 복잡성 지표를 제어하면서도 응답 품질을 유지하거나 향상시키는 데 있어 탁월한 성능을 보였습니다. 특히, 무작위로 샘플링된 태그 서브셋만으로도 테스트 예제의 모든 복잡성을 충분히 제어할 수 있었으며, 이는 요구되는 학습 데이터 양을 줄이는 데 기여했습니다.



### Continuous Output Personality Detection Models via Mixed Strategy Training (https://arxiv.org/abs/2406.16223)
- **What's New**: 이 논문은 성격 탐지 모델을 연속적인 출력 값을 제공하는 방식으로 훈련하는 새로운 접근법을 제안합니다. 기존의 성격 모델은 이진 결과만을 제공합니다. 저자들은 Reddit 댓글의 포괄적인 성격 라벨링을 포함한 PANDORA 데이터셋을 활용하여 Big Five 성격 특성을 높은 정확도로 예측하는 모델을 개발했습니다.

- **Technical Details**: 이 접근법은 RoBERTa-base 모델을 세밀하게 조정(fine-tuning)하는 과정을 포함합니다. 다양한 전략, 예를 들어 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 통합과 하이퍼파라미터 튜닝 등을 적용했습니다. PANDORA 데이터셋은 10,000명 이상의 Reddit 유저 댓글을 포함하며, 이중 1600명에 대한 Big Five 성격 특성이 라벨링 되어 있습니다.

- **Performance Highlights**: 개발된 모델은 전통적인 이진 분류 방법을 크게 능가하며, 성격 특성에 대해 정밀한 연속적 출력을 제공합니다. 이러한 개선된 결과는 AI, 심리학, 인사관리, 마케팅 및 헬스케어 분야에서의 응용을 강화할 수 있습니다.



### LLMs' Classification Performance is Overclaimed (https://arxiv.org/abs/2406.16203)
- **What's New**: 이번 연구에서는 AI 분류 작업에서 정확한 답변이 선택지에 포함되지 않은 경우 Large Language Models (LLMs)의 한계를 밝혀냅니다. 이를 'Classify-w/o-Gold'라고 정의하고, 이를 평가하기 위한 새로운 테스트베드를 제안합니다. Know-No라는 벤치마크를 도입하여 세 가지 대표적인 분류 작업을 통해 LLMs의 성능을 평가합니다.

- **Technical Details**: Know-No 벤치마크는 Bank-77 (의도 분류 과제), MC-Test (다중 선택 질문 응답 과제), 그리고 새롭게 만들어진 EquInfer (과학 논문에서 올바른 방정식을 추론하는 과제)로 구성됩니다. 각 과제는 입력 길이, 라벨 크기, 라벨 스코프 측면에서 다양성을 포함합니다. 새로운 평가 기준인 OmniAccuracy를 제안하여 'Gold' 라벨이 있을 때 및 없을 때의 정확도를 통합적으로 평가합니다.

- **Performance Highlights**: 이번 연구는 새로운 평가 기준 OmniAccuracy를 통해 LLMs가 분류 작업에서 실제 인간 수준의 이해력을 가지고 있는지를 평가합니다. LLMs는 'Gold' 라벨이 없는 경우에도 여전히 주어진 선택지에서 답을 고르려고 시도하며, 이로 인해 그들의 성능이 과대평가되는 경향이 있음을 발견했습니다. 따라서 OmniAccuracy는 'Gold' 라벨 유무에 따른 성능을 포괄적으로 반영하여 LLMs의 실제 분류 능력을 더욱 정확하게 평가합니다.



### FS-RAG: A Frame Semantics Based Approach for Improved Factual Accuracy in Large Language Models (https://arxiv.org/abs/2406.16167)
Comments:
          program code and prompts available at this https URL

- **What's New**: 이번 연구에서는 프레임 시맨틱(Frame Semantics)에 기반한 새로운 정보 검색 방식을 도입하여 대형 언어 모델(LLM)의 사실 오류 문제를 완화하는 Retrieval Augmented Generation(RAG) 기법을 확장하였습니다. 이 방식은 프레임 시맨틱 이론을 활용해 관련 정보를 인덱싱하고 검색하는 과정에서의 효과를 증명했습니다. 실험 결과, 저희가 제안한 FS-RAG 방법이 효과적임을 확인했으며, 데이터 기반의 프레임 시맨틱 이론에 대한 통찰을 제공할 가능성도 있음을 시사합니다.

- **Technical Details**: 기존의 Retrieval Augmented Generation(RAG) 방식은 조회된 정보를 프롬프트에 포함시켜 LLM의 정보 검색 부담을 덜어주는 역할을 합니다. 그러나 기존 방식의 단점은 관련 정보 조회 문제를 해결하지 못한다는 점입니다. 저희는 이를 개선하기 위해 프레임 시맨틱 이론을 바탕으로 투명하고 가변적인 저장 및 검색 시스템을 개발했습니다. 프레임 시맨틱은 의미와 개념적 '프레임' 또는 '스키마'를 통해 단어의 의미를 이해하는 이론입니다. FrameNet 데이터베이스를 활용해 특정 타입의 이벤트, 관계, 엔티티를 구체적으로 정의하고 각 역할을 예시 문장과 함께 기록했습니다.

- **Performance Highlights**: 저희가 제안한 FS-RAG 방식은 전통적인 검색 시스템과 LLM 기반 검색 시스템 모두를 뛰어넘는 성능을 보였습니다. 특히, 사실 오류를 줄이기 위해 필수적인 정보를 효과적으로 인덱싱하고 검색하는 데 있어 매우 유리한 결과를 나타냈습니다. 또한, 이 방식은 해석 가능성이 높아, 결과에 대한 명확한 이해를 제공하며, 프레임 시맨틱 이론에 데이터 기반의 새로운 통찰을 제공할 수 있는 잠재력을 지니고 있습니다.



### Towards Region-aware Bias Evaluation Metrics (https://arxiv.org/abs/2406.16152)
- **What's New**: 이 연구는 다양한 지역에서 발생하는 성 편견(gender bias)의 주제적(topical) 차이점을 식별하고, 지역 인식(region-aware) 하향식 접근법을 제안합니다. 현재의 편견 평가 기준은 한정적일 수 있으며, 다양한 문화적, 인구통계적 요인에 따라 변할 수 있는 성 편견을 충분히 포착하지 못합니다.

- **Technical Details**: 이 논문은 주제 모델링(topic modeling)과 임베딩 기반 접근법을 사용하여 각 지역에서 성 편견 차원을 자동으로 식별합니다. 제안된 방법은 지역별로 성별 정렬된(gender-aligned) 주제를 찾고, 이 주제를 사용하여 WEAT(Word Embedding Association Test) 기반 평가 지표를 구축합니다. 평가 지표는 Reddit과 UN General Debates 데이터를 기반으로 성 편견을 테스트합니다.

- **Performance Highlights**: 제안된 지역 기반 성 편견 차원은 여러 지역에서 기존의 WEAT/SEAT 평가 지표보다 인간의 지각과 더 잘 일치하는 결과를 보였습니다. 특히, 잘 대표되는(highly-represented) 지역에서는 더 높은 일치를 보이며, 지역 인식 편견 평가 지표의 중요성을 강조합니다.



### Chain-of-Probe: Examing the Necessity and Accuracy of CoT Step-by-Step (https://arxiv.org/abs/2406.16144)
- **What's New**: 이번 연구에서는 대규모 언어 모델(Large Language Models, LLMs)에서 'Early Answering' 현상을 발견했습니다. 이는 모델이 Chain-of-Thought(CoT)을 생성하기 전에 이미 답을 가지고 있다는 것을 의미합니다. 이를 해결하기 위해 'Chain-of-Probe(CoP)'라는 새로운 방법을 제안하고, 모델의 사고 변화 과정을 탐색하였습니다.

- **Technical Details**: 제안된 CoP 방법은 모델이 각 추론 단계를 완료한 후 현재 추론을 기반으로 예측을 출력하고 그 신뢰도를 기록합니다. 이를 통해 추론 단계별로 모델의 결정에 미치는 영향을 이해하고, 이를 바탕으로 CoT의 필요성과 정확성을 평가합니다. 또한, 쉬운 질문에 대해 CoT가 불필요한 경우가 많으며, 어려운 질문에 대해서는 CoT가 모델의 초기 선택을 변경하는 경향이 있음을 분석했습니다.

- **Performance Highlights**: Early Answering가 발생하는 경우의 정답률이 그렇지 않은 경우보다 20% 이상 높으며, 이는 CoT가 모델 성능을 항상 향상시키지 않음을 시사합니다. CoP Score 기반으로 응답을 선택하는 전략을 통해 응답의 정확도가 대다수 투표(majority voting) 방식과 유사한 수준으로 향상되었습니다. CoP Tree를 도입하여 오류가 포함된 CoT를 식별하고 재샘플링(resampling)하는 방법을 통해 전체 모델의 정확도를 평균 13% 높였습니다.



### Crosslingual Capabilities and Knowledge Barriers in Multilingual Large Language Models (https://arxiv.org/abs/2406.16135)
- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 다국어 코퍼스를 사용한 사전 학습 덕분에 다국어 처리가 가능해졌습니다. 하지만 이 모델들이 언어 간의 개념을 어떻게 연결하는지를 평가한 결과, 표면적인 능력은 뛰어나지만 깊이 있는 교차 언어 지식 전이에는 한계를 보였습니다.

- **Technical Details**: 이 연구는 기계 번역과 임베딩 공간 분석을 통해 LLM의 교차 언어 능력을 평가했고, MMLU 벤치마크와 특정 도메인(Harry Potter 퀴즈)을 사용하여 일반 및 도메인 별 교차 언어 지식 전이 능력을 분석했습니다. 모델들은 표면 수준에서는 우수한 성능을 보였지만, 한 언어에서 학습된 지식을 다른 언어로 전이하는 데 어려움을 겪었습니다.

- **Performance Highlights**: 단순한 추론 단계의 완화 방법은 제한적인 개선만을 제공했으나, 혼합 언어 데이터로 미세 조정(fine-tuning)을 통해 이러한 격차를 효과적으로 줄일 수 있음을 입증했습니다. 특히, 도메인 외의 데이터(WikiText)를 사용한 경우에도 유의미한 성능 향상을 보였습니다. 코드 공개도 이루어졌습니다.



### SEAM: A Stochastic Benchmark for Multi-Document Tasks (https://arxiv.org/abs/2406.16086)
- **What's New**: 최근 논문에서는 SEAM이라는 새로운 벤치마크를 소개합니다. SEAM은 다양한 실제 문서 컬렉션 작업에서 대규모 언어 모델(LLMs)의 성능을 평가하기 위해 설계된 최초의 벤치마크로, 문서 간의 일관성 결여, 모순, 정보의 중복 등과 같은 문제를 반영하고 있습니다. SEAM은 특히 일부 프롬프트의 경미한 변형에 대한 LLMs의 민감성을 다루기 위해 반복적인 평가를 수행합니다.

- **Technical Details**: SEAM은 6개의 기존 데이터 세트에서 샘플을 가져오며, 이를 통해 다양한 다중 문서 작업을 수행합니다. SEAM은 입력-출력 형식, 평가 프로토콜, 프롬프트 포맷팅을 표준화하여 모델 평가를 간소화합니다. 중요한 점은 SEAM이 단일 프롬프트 템플릿에 의존하지 않고, 여러 평가 실행에서 다양한 임의 요소 값을 균일하게 무작위로 샘플링하여 모델의 성능을 평가하는 스토캐스틱(stochastic) 접근법을 사용한다는 것입니다.

- **Performance Highlights**: 여러 모델 및 설정을 테스트한 결과, 멀티-도큐멘트 작업은 최신 70B 파라미터를 가진 모델들조차도 여전히 많은 개선의 여지가 있음을 발견했습니다. 또한, 모델 크기가 커졌다고 해서 평균 성능이 우수하거나 더 견고한 예측을 보장하지는 않으며, 문서 길이와 모델의 성공률 사이에는 상관관계가 없음을 보여줍니다.



### EERPD: Leveraging Emotion and Emotion Regulation for Improving Personality Detection (https://arxiv.org/abs/2406.16079)
- **What's New**: 심리학의 필수적인 개념인 감정 조절(emotion regulation)을 활용하여 개인의 성격(personality)을 검출하는 새로운 방법인 EERPD를 제안합니다. 기존 연구들은 텍스트 전체를 사용하여 성격을 예측하였으나, 감정 조절과 성격 사이의 잘 알려진 상관 관계를 간과했습니다.

- **Technical Details**: 본 연구에서는 감정과 감정 조절이라는 심리적 개념을 결합하여 텍스트로부터 성격을 예측하는 방식을 제안합니다. EERPD는 RAG(Retrieval-Augmented Generation) 기반의 프레임워크로, 다양한 텍스트-성격 페어를 포함하는 참조 라이브러리를 구성하고, 입력 텍스트를 감정 문장(Emotion Sentences)과 감정 조절 문장(Emotion Regulation Sentences)으로 분류하여 효과적으로 유사한 예시를 참조 라이브러리에서 검색합니다. 이 과정에서 few-shot learning을 이용하여 성격을 예측합니다.

- **Performance Highlights**: EERPD는 기존의 최고 성능(state-of-the-art) 대비 두 개의 벤치마크 데이터셋에서 평균 F1 점수를 각각 15.05/4.29 만큼 향상시켰습니다. 이는 감정 조절 개념을 적용한 것이 성격 검출의 정확성과 강건성을 크게 향상시킨다는 것을 보여줍니다.



### First Heuristic Then Rational: Dynamic Use of Heuristics in Language Model Reasoning (https://arxiv.org/abs/2406.16078)
- **What's New**: 이번 연구에서는 언어모델(LM)이 다중 단계 추론 과정에서 사용하는 체계적인 전략을 분석했습니다. 초기 단계에서는 어휘적 중복(Lexical Overlap)과 같은 휴리스틱(heuristics)에 많이 의존하다가, 답에 가까워질수록 이러한 의존도가 감소한다는 것도 밝혀냈습니다. 이는 LMs가 제한된 수의 미래 단계를 추적하며, 휴리스틱한 전략과 논리적인 전략을 동적으로 결합해 문제를 해결한다는 것을 시사합니다.

- **Technical Details**: 이번 연구는 산술적 추론(arithmetic reasoning) 작업을 통해 LM의 추론 능력을 테스트했습니다. 실험에서는 자연 및 인위적으로 조정된 데이터셋을 사용하였으며, 각 문제는 여러 전제와 질문으로 구성되어 있습니다. 모델은 매 단계마다 특정 전제를 선택해 새로운 사실을 도출하며, 최종적으로 질문에 대한 답을 생성합니다. 이 과정에서 불필요한 전제를 필터링하고, 반복적인 참조를 통해 추론을 이어나갑니다.

- **Performance Highlights**: 연구 결과, LMs는 초기 단계에서 더 많은 단계가 필요한 경우 휴리스틱에 많이 의존하며, 답에 가까워질수록 보다 논리적인 접근을 시도하는 경향이 있음을 확인했습니다. 이는 LMs가 다단계 추론 작업에서 제한된 수의 미래 단계를 추적하며, 이 과정에서 휴리스틱과 논리적인 전략을 결합해 문제를 해결한다는 것을 나타냅니다.



### Dancing in the syntax forest: fast, accurate and explainable sentiment analysis with SALSA (https://arxiv.org/abs/2406.16071)
Comments:
          Accepted for publication at SEPLN-CEDI2024: Seminar of the Spanish Society for Natural Language Processing at the 7th Spanish Conference on Informatics

- **What's New**: SALSA 프로젝트는 유럽연구위원회(ERC)로부터 자금을 지원받아, 빠른 구문 분석 기술을 활용하여 경량화되고 효율적인 감정 분석 시스템을 개발하는 것을 목표로 합니다. 이 프로젝트는 소규모 기업도 사용할 수 있도록 저비용의 컴퓨팅 자원으로 대규모 감정 분석을 가능하게 합니다.

- **Technical Details**: SALSA 프로젝트는 구문 분석을 시퀀스 라벨링 (sequence labeling) 작업으로 변환하여, 최신 신경망 기법을 통해 효율적이고 정확한 구문 분석기를 제작하는 데 성공한 FASTPARSE 프로젝트의 성과를 활용합니다. 이를 통해 구문 기반의 규칙을 사용하여 감정 분석을 수행하고, 멀티태스킹 학습 접근을 통해 구문 분석과 감정 분석을 동시에 수행할 수 있습니다.

- **Performance Highlights**: SALSA 프로젝트는 CPU와 메모리 요구 사항이 적은 신속한 구문 분석기를 활용하여, 표준 소비자용 GPU에서도 초당 약 천 개의 문장을 분석할 수 있습니다. 이는 소규모 기업들이 여전히 높은 정확성을 유지하면서도, 감정 분석 시스템을 운영할 수 있게 합니다.



### FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models (https://arxiv.org/abs/2406.16069)
- **What's New**: 새로운 방법인 FastMem을 소개합니다. 이 방법은 instruction fine-tuned 대형 언어 모델(LLMs)의 문맥 인식을 빠른 메모리화(fast memorization)을 통해 개선하는 것을 목표로 합니다. FastMem은 추론 전에 prompt의 가능성을 최대화하는데, 이를 통해 모델의 문맥 이해력과 정확성을 크게 향상시킵니다.

- **Technical Details**: FastMem은 단지 마지막 Feed-Forward Network(FFN) 모듈만을 미세 조정하여 효율적인 최적화를 보장하면서 과적합을 피합니다. 이 접근 방식은 기존의 전체 모델을 미세 조정하는 것보다 최소한의 연산 비용으로 상당한 성능 향상을 이룹니다. 구체적으로, FastMem은 prompt의 가능성을 최대화하는 사전학습과 유사한 목표를 instruction fine-tuned 모델에 적용하여 문맥의 혼란도를 줄입니다. 이로 인해 FFN 모듈의 파라미터 최적화는 빠르게 이루어지며 peak 메모리 사용량의 증가 없이 몇 초 내에 완료될 수 있습니다.

- **Performance Highlights**: FastMem은 여러 태스크에서 LLM의 성능을 눈에 띄게 향상시켰습니다. 예를 들어, Llama 3-8B-Inst 모델의 NQ-SWAP 데이터셋 정확도는 59.1%에서 71.6%로 개선되었고, Qwen 1.5-4B-Chat의 출력 구조 실패율은 34.9%에서 25.5%로 줄어들었습니다. 이러한 결과는 FastMem이 다양한 응용 프로그램에서 LLM의 신뢰성과 정확성을 향상시킬 가능성을 보여줍니다.



### Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models (https://arxiv.org/abs/2406.16033)
- **What's New**: 이 논문에서 연구진은 대형 언어 모델(LLMs)의 내부 계획 메커니즘을 탐구하여, 특히 'look-ahead' 계획 능력을 분석했습니다. 'Look-ahead' 계획은 사람이 계획을 세울 때 미래의 여러 단계를 내다보는 방식을 의미합니다. 이를 통해 대형 언어 모델이 계획을 수립하는 방식과 그 내부 표현을 이해하려 했습니다.

- **Technical Details**: 연구진은 Multi-Layer Perception(MLP)와 Multi-Head Self-Attention(MHSA) 구성 요소를 분석하여 대형 언어 모델의 계획 메커니즘을 연구했습니다. '정보 흐름'과 '내부 표현 탐사'라는 두 단계로 실험을 설계했습니다. 첫 번째 단계에서는 MHSA를 통해 정보 흐름을 추적하고, 두 번째 단계에서는 내부 표현에 미래의 정보가 사전 인코딩되는지를 조사했습니다.

- **Performance Highlights**: 실험 결과, 중간 및 상위 레이어에서 짧은 기간의 미래 결정을 어느 정도 인코딩하고 있음을 발견했습니다. MHSA는 주로 목표 상태와 최근 단계에서 정보를 추출하며, 중간 레이어의 출력은 일부 정확한 결정을 디코딩할 수 있었습니다.



### Zero-Shot Cross-Lingual NER Using Phonemic Representations for Low-Resource Languages (https://arxiv.org/abs/2406.16030)
Comments:
          7 pages, 5 figures, 5 tables

- **What's New**: 기존의 제로샷(NER, Named Entity Recognition) 접근 방식은 대상 언어에 대한 많은 사전 지식을 필요로 하지만, 이는 저자원 언어(low-resource languages)에서는 비현실적입니다. 이 연구는 국제음성기호(IPA, International Phonetic Alphabet)를 기반으로 한 음운적 표현을 활용하여 서로 다른 언어의 표현 간의 차이를 줄이는 새로운 NER 접근 방식을 제안합니다. 실험 결과, 이 방법이 극도로 저자원 언어에서 높은 평균 F-1 점수(46.38%)와 낮은 표준 편차(12.67)를 기록하며 기존 모델보다 훨씬 뛰어난 성능을 보여주었습니다. 특히 비라틴 스크립트(non-Latin scripts)에 대한 강건성을 입증했습니다.

- **Technical Details**: 본 연구는 NER 태스크를 위해 전통적인 정서법 텍스트 대신 음운적 표기(IPA)를 사용합니다. 이를 위해, 음운 자료 94개 언어로 사전 훈련된 XPhoneBERT 모델을 활용했습니다. 데이터셋을 IPA 표현으로 변환한 후, 사전 훈련된 모델을 소스 언어(예: 영어)의 음운 데이터로 미세 조정(fine-tuning)하고, 이를 대상으로 바로 적용할 수 있도록 했습니다. 본 접근 방식은 새로운 언어까지의 일반화 가능성을 확인하기 위해 훈련 중 대상 언어에 대한 접근을 전혀 허용하지 않았으며, 대상 언어의 유형론적 정보를 제외하여 언어 무관 방법론으로 남도록 했습니다.

- **Performance Highlights**: 우리의 접근 방식은 저자원 언어에서 mBERT 및 CANINE 같은 기존의 grapheme 기반 모델 대비 매우 우수한 성능을 보였습니다. 실험 결과, 저자원 언어 33개에서 음운 기반 모델이 권장되었습니다. 실험은 WikiANN NER 데이터셋을 사용하여 영어에서만 훈련된 모델을 다양한 저자원 언어에 평가하였습니다. Phoneme 기반의 XPhoneBERT가 평균 F-1 점수 46.38%를 기록하며, 저자원 언어에서도 높은 안정성을 보였습니다.



### Harvesting Events from Multiple Sources: Towards a Cross-Document Event Extraction Paradigm (https://arxiv.org/abs/2406.16021)
Comments:
          ACL2024(Findings)

- **What's New**: 문서 단위의 이벤트 추출에서 발생하는 정보 제한 문제를 해결하기 위해 다중 문서에서 이벤트 정보를 통합하는 '크로스-도큐먼트 이벤트 추출(CDEE)' 과제를 제안합니다. 이를 위해 CLES라는 새로운 크로스-도큐먼트 이벤트 추출 데이터셋을 구축했습니다. 이 데이터셋은 20,059개의 문서와 37,688개의 멘션 수준 이벤트를 포함하며, 70% 이상이 크로스-도큐먼트입니다.

- **Technical Details**: CDEE 파이프라인은 5단계로 구성됩니다: (1) 이벤트 추출(Event Extraction), (2) 공지(코어퍼런스) 해결(Event Coreference Resolution), (3) 엔티티 표준화(Entity Normalization), (4) 역할 표준화(Role Normalization), (5) 엔티티-역할 해결(Entity-Role Resolution). 이 과정은 다양한 문서에서 이벤트 정보를 통합하여 보다 완전한 이벤트 표현을 얻는 것을 목표로 합니다.

- **Performance Highlights**: CDEE 파이프라인은 엔드 투 엔드 크로스-도큐먼트 이벤트 추출에서 약 72%의 F1 스코어를 달성했으며, 이는 해당 작업의 어려움을 시사합니다. 이번 연구는 새로운 정보 추출 연구의 방향을 제시하고, 향후 연구의 주목을 받을 것으로 기대됩니다.



### Database-Augmented Query Representation for Information Retrieva (https://arxiv.org/abs/2406.16013)
- **What's New**: 새로운 정보 검색(Retrieval) 모델이 등장했습니다. 사용자가 제공하는 간단한 쿼리(query) 대신, 이 모델은 관계형 데이터베이스(Relational Database)의 다양한 메타데이터를 활용하여 쿼리를 확장합니다. 이 모델의 이름은 DAQu(Database-Augmented Query)입니다.

- **Technical Details**: DAQu는 원본 쿼리를 관계형 데이터베이스의 다양한 테이블에 있는 메타데이터(metadata)를 통해 확장합니다. 이러한 메타데이터는 특징(feature)들이 많고 순서가 없기 때문에, 그래프 기반의 집합 인코딩 전략(Graph-based Set Encoding Strategy)을 사용하여 특징들의 위계(hierarchies)를 고려해 인코딩합니다.

- **Performance Highlights**: DAQu는 다양한 메타데이터를 활용할 수 있는 검색 시나리오에서 기존의 쿼리 확장 방법들보다 월등히 높은 검색 성능을 보여줍니다.



### Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization (https://arxiv.org/abs/2406.16008)
Comments:
          ACL Findings 2024

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 긴 입력 컨텍스트에서 중간의 중요한 정보를 잘 찾지 못하는 'lost-in-the-middle' 문제에 대해 다룹니다. 이 현상을 더 잘 이해하고자, LLM의 고유한 주의 편향과 이를 해결하기 위한 'found-in-the-middle' 교정 메커니즘을 제안합니다. 이 메커니즘은 모델이 위치에 관계없이 중요성을 기준으로 정보를 올바르게 주의하도록 도와줍니다.

- **Technical Details**: LLM들이 입력의 시작과 끝에 위치한 토큰에 높은 주의를 기울이는 U자형 주의 분포를 보인다는 사실을 발견했습니다. 이를 해결하기 위해 주의 편향을 교정하는 메커니즘을 도입하여, 입력 중간의 중요한 정보를 더 잘 찾을 수 있도록 하였습니다. 'found-in-the-middle' 메커니즘은 주어진 쿼리에 대해 문서의 주의 점수를 교정하여 보다 정확한 상관 관계를 유지하도록 합니다.

- **Performance Highlights**: 'found-in-the-middle' 메커니즘을 적용한 결과, Open-domain 질문 응답 작업에서 성능이 크게 향상되었으며, Retrieval-Augmented Generation (RAG) 작업에서도 기존 방법들보다 최대 15% 포인트 높은 성과를 보였습니다. 특히, NaturalQuestion 데이터셋에서 표준 모델 생성보다 15% 포인트 성능이 개선되었습니다.



### Distributed Rule Vectors is A Key Mechanism in Large Language Models' In-Context Learning (https://arxiv.org/abs/2406.16007)
- **What's New**: 최근 연구는 대형 언어 모델(LLMs)의 새로운 메커니즘을 밝혀내며, 다중 시연이 필요한 작업에서 'task vector'가 존재하지 않음을 발견했다. 대신, 각 시연에서 제공되는 규칙 정보가 'rule vector'로 전달되어 분산 방식으로 출력을 생성한다는 것이다. 이는 모델이 다수의 시연을 통해 규칙을 추출 및 적용하는 방법에 대한 새로운 관점을 제공한다.

- **Technical Details**: 이 연구는 LLM이 시연 기반 학습을 통해 규칙을 만드는 과정에서 'task vector'가 아닌 'distributed rule vectors'를 사용한다는 것을 밝혔다. 'demixed-PCA (dPCA)' 분석을 통해, 각 시연의 쿼리-답변(QA) 쌍이 규칙 정보를 포함한 'rule vector'를 형성하고, 모든 'rule vectors'가 공동으로 출력을 형성하는 것을 확인했다. 이 메커니즘은 LLaMA-7B 모델을 사용하여 테스트되었다.

- **Performance Highlights**: 결과적으로, 모델은 충분한 수의 시연을 통해 분류 작업을 성공적으로 수행할 수 있었다. 'knowledge task'에서 단일 시연으로 정확도가 0%에서 70%로 증가했으며, 추가 시연이 소폭의 개선만 가져왔다. 반면, 'categorization task'에서는 7회 이상의 시연을 통해 80% 이상의 정확도를 달성했다. 이는 인간과 동물이 유사한 방식으로 지식을 습득하는 과정과 유사하다.



### Memorizing Documents with Guidance in Large Language Models (https://arxiv.org/abs/2406.15996)
Comments:
          IJCAI 2024

- **What's New**: 이 연구는 언어 모델(LLM)이 문서별 기억(메모리)을 추적할 수 있는 새로운 아키텍처를 제안합니다. 기존의 사후 해석 방법 대신, 문서별 메모리 아키텍처(document-wise memory architecture)를 사용하여 문서 기억을 추적하고, 문서 안내 손실(document guidance loss)을 제안하여 문서 관련 내용을 더 잘 기억하고 생성할 수 있도록 합니다.

- **Technical Details**: 제안된 아키텍처는 문서 표현(DocRep)을 메모리 엔트리(memory entries)에 매핑하고, LLM의 전방 과정에서 부드럽게 마스크(softly mask)합니다. 문서 안내 손실(document guidance loss)은 문서 메모리가 있는 텍스트의 가능성을 높이고, 다른 문서 메모리가 있는 텍스트의 가능성을 낮추는 방식으로 작동합니다. 실험에서 Pythia-1B 모델과 Wikitext-103-v1 데이터를 사용하여 문서별로 다른 메모리 엔트리를 제공하고, 문서 관련 내용을 높게 회상할 수 있는 결과를 보였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 문서별로 다른 메모리 엔트리를 제공하고, 훈련된 문서별 기억을 통해 문서 관련 내용을 생성하는 데 높은 회상률을 보여줍니다. 이는 문서 안내 손실을 통해 문서와 메모리 엔트리를 엮고, 문서 간의 메모리 엔트리가 달라지도록 독려한 덕분입니다.



### Can LLM Graph Reasoning Generalize beyond Pattern Memorization? (https://arxiv.org/abs/2406.15992)
Comments:
          16 pages, 6 figures, Code and data will be publicly available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 그래프 추론 능력을 평가하기 위한 새로운 벤치마크인 NLGift를 소개합니다. 이전 연구들은 주로 LLM이 훈련 데이터 패턴을 단순히 암기하는지 또는 일반화된 그래프 추론 능력을 갖추고 있는지 평가하는 데 초점을 맞췄습니다. 새로운 벤치마크는 LLM이 실제 세계의 그래프 기반 작업에서 얼마나 잘 성능을 발휘할 수 있는지 평가합니다.

- **Technical Details**: NLGift는 37,000개의 문제를 포함하며, 5가지 패턴(semantic, numerical, structural, reasoning, real-world patterns)을 통해 LLM의 그래프 추론 능력을 평가합니다. 두 가지 LLM을 사용하여 네 가지 그래프 추론 작업(connectivity, shortest path, topological sort, maximum flow)에서 실험을 수행했습니다. 이 벤치마크는 두 가지 기준을 제시합니다: Significant Transfer(기본 기준)와 Strong Recovery(강력 기준).

- **Performance Highlights**: 실험 결과, LLM은 간단한 패턴(semantic, numerical, structural)에서는 기본 기준을 75%의 경우에서 만족시켰지만, 강력 기준은 35%의 경우에서만 만족했습니다. 추론 패턴의 경우, 기본 일반화 기준을 33%에서만 만족시켰으며 강력 기준에는 전혀 도달하지 못했습니다. 가장 어려운 실제 세계 패턴에서는 기본 일반화 기준을 6%의 경우에서만 만족시켰으며, 69%의 경우에서 그래프 지시 튜닝(graph instruction tuning)이 오히려 역효과를 나타냈습니다.

- **Insights and Future Work**: 실험 분석 결과, 그래프 추론 일반화에 대한 주요 요인으로 훈련 데이터에서의 작업 구성 및 키워드 빈도가 큰 영향을 미친다는 것을 발견했습니다. 이를 개선하기 위해 코드 혼합(code mixing), 기계 생성 CoTs(machine-generated CoTs), 및 사후 정렬(post-training alignment) 등 세 가지 전략을 조사했습니다. 특히 사후 정렬(post-training alignment)이 실제 세계 작업에서 가장 유망한 방법으로 나타났지만, LLM을 일반화 가능하고 전송 가능한 그래프 추론 능력으로 강화하는 것은 여전히 열려 있는 연구 질문입니다.



### Enhancing Cross-Document Event Coreference Resolution by Discourse Structure and Semantic Information (https://arxiv.org/abs/2406.15990)
- **What's New**: 이번 연구에서는 기존의 문서 간 이벤트 동일성 해지(Cross-Document Event Coreference Resolution, CDECR) 모델들이 문서 수준의 정보를 활용하지 못하고, 장거리 의존성을 포착하는 데 어려움을 겪던 문제점을 해결하고자 새로운 접근 방식을 제안합니다. 이 연구에서는 문서 수준의 RST(Rhetorical Structure Theory) 트리와 문서 간의 어휘 체인(Lexical Chains)을 구축하여 문서의 구조적 및 의미적 정보를 모델링합니다. 이를 통해 생성된 문서 간 이질 그래프(heterogeneous graphs)를 GAT(Graph Attention Networks)를 활용하여 학습하고, 최종적으로 표준 클러스터링 알고리즘을 통해 동일 이벤트를 인식합니다. 또한 영어에 한정된 기존 데이터셋의 한계를 극복하기 위해, 53,066개의 이벤트 언급과 4,476개의 클러스터로 구성된 대규모 중국어 문서 간 이벤트 코어프런스 데이터셋을 개발하였습니다.

- **Technical Details**: 제안된 Discourse-Information-Enhanced Event Coreference (DIE-EC) 모델은 문서 수준의 RST 트리와 문서 간의 어휘 체인을 구축하여 문서의 구조적 및 의미적 정보를 활용합니다. 모델 구성은 (1) 인코더 레이어: 입력 문서를 인코딩하여 문맥 표현(contextual representations)을 얻습니다. (2) 담론 정보 레이어: 문서 수준의 RST 트리와 문서 간 레벨 어휘 체인을 구축합니다. RST 그래프와 어휘 체인 그래프는 GAT를 통해 처리합니다. (3) 쌍 채점기(pair scorer): GAT의 결과를 MLP(Multi-Layer Perceptron)로 처리하여 코어프런스를 결정하고, 최종적으로 응집형 클러스터링 알고리즘(agglomerative clustering)을 통해 이벤트를 클러스터링합니다. 또한, 연구의 일환으로 처음으로 대규모 중국어 문서 간 이벤트 코어프런스 데이터셋(WEC-Zh)을 구축하였습니다. 이 데이터셋은 53,066개의 이벤트 언급, 4,476개의 이벤트 클러스터, 그리고 8개의 이벤트 유형을 포함하고 있습니다.

- **Performance Highlights**: 제안된 모델은 WEC-Eng(영어 데이터셋)과 WEC-Zh(중국어 데이터셋) 모두에서 기존의 모든 기준선(base-line) 모델들을 큰 차이로 능가했습니다. 다양한 평가 지표(evaluation metrics)에서 최첨단 성능(state-of-the-art results)를 달성했습니다.



### Serial Position Effects of Large Language Models (https://arxiv.org/abs/2406.15981)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 인간 심리학에서 잘 알려져 있는 연속 위치 효과(Serial Position Effects, SPE)를 나타낼 수 있다는 사실이 밝혀졌습니다. LLM이 초점이 없거나 정답 레이블이 없는 경우, 이러한 효과가 중요하게 작용할 수 있음을 강조합니다. 또한, 정교한 프롬프트 설계가 SPE를 일부 완화할 수 있지만, 그 효과는 일관되지 않음을 발견했습니다.

- **Technical Details**: 연구는 BERT와 같은 모델이 미세 조정을 통해 SPE를 완화하는지와 달리, GPT-3.5-turbo, Llama2와 같은 새로운 LLM은 미세 조정 없이도 SPE를 나타내는지에 초점을 맞췄습니다. 연구는 선택 항목의 순서가 모델의 응답에 미치는 영향을 확인하기 위해 여러 위치 기반 인지 편향을 테스트했습니다. 또한, 프롬프트 엔지니어링과 Chain-of-Thought(CoT) 접근 방식이 이러한 편향을 어떻게 조절할 수 있는지에 대해 조사했습니다.

- **Performance Highlights**: 주요 결과로는 다양한 LLM이 일관되게 SPE를 나타낸다는 점, 인코더-디코더 모델과 디코더 전용 모델 간 SPE의 차이가 없다는 점, 특정 작업에 따라 효과의 강도가 다를 수 있다는 점을 들 수 있습니다. 정교한 프롬프트와 CoT 접근 방식이 일부 편향을 완화할 수 있지만, 이는 상황에 따라 다릅니다. 이러한 편향이 특히 정답 레이블이 없는 상황에서 모델의 성능을 이해하고 향상시키는 데 중요한 역할을 할 수 있음을 강조합니다.



### ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods (https://arxiv.org/abs/2406.15968)
- **What's New**: ReCaLL (Relative Conditional Log-Likelihood)라는 새로운 멤버십 유추 공격(MIA)을 소개합니다. 이 방법은 대형 언어 모델(LLMs)의 사전 훈련 데이터를 감지하는 데 중점을 두며, 조건부 언어 모델링 능력을 활용하여 데이터 포인트를 비회원 컨텍스트로 접두사로 구성할 때의 조건부 로그 우도(log-likelihood) 변화를 지켜봅니다.

- **Technical Details**: ReCaLL은 비회원 컨텍스트로 구성된 접두사로 데이터 포인트를 조건화할 때의 조건부 로그 우도 감소를 관찰합니다. 이 감소가 비회원 데이터에 비해 회원 데이터에서 더 크게 나타나는 것을 발견하였습니다. ReCaLL은 WikiMIA와 MIMIR 벤치마크에서 평가되었으며, WikiMIA에서 최첨단 성능을 보여주었고 MIMIR에서도 경쟁력 있는 결과를 얻었습니다. 또한, 랜덤 및 합성 접두사를 사용할 때도 유사한 성능을 보이며, 앙상블 접근법을 통해 성능을 더욱 향상시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: ReCaLL은 WikiMIA 데이터셋에서 기존의 MIA 방법보다 큰 차이로 우수한 성능을 발휘하였고, MIMIR에서 경쟁력 있는 결과를 얻었습니다. 앙상블 접근법을 사용함으로써 성능을 추가로 향상시킬 수 있었습니다. LLM의 멤버십 정보를 효과적으로 사용하여 시퀀스 및 토큰 수준에서의 유추 방법을 깊이 분석하였습니다.



### Evaluating the Effectiveness of the Foundational Models for Q&A Classification in Mental Health car (https://arxiv.org/abs/2406.15966)
- **What's New**: 이번 연구는 PLM (Pre-trained Language Models) 기술이 정신 건강 지원 분야에서 어떻게 효과적으로 활용될 수 있는지를 평가하는 첫 번째 사례 중 하나입니다. 특히 아랍어로 된 MentalQA 데이터셋을 사용하여 Q&A 분류에서의 PLM 성능을 분석하였습니다.

- **Technical Details**: 이 연구에서는 전통적 특징 추출(Traditional Feature Extraction), PLM을 특징 추출기로 활용, PLM 파인튜닝(Fine-tuning), 그리고 GPT-3.5 및 GPT-4를 사용한 제로샷 및 퓨샷 학습(Zeroshot and Few-shot Learning) 등 네 가지 학습 접근 방식을 실험했습니다. 질문 분류는 Jaccard Score 0.80, 답변 분류는 Jaccard Score 0.86을 기록한 MARBERT가 최고의 성능을 보였습니다.

- **Performance Highlights**: 실험 결과 PLM과 프로프트 기반 접근 방식은 전통적인 방법에 비해 훨씬 더 높은 성능을 보였습니다. 예를 들어, GPT-3.5의 퓨샷 학습에서는 질문 분류 성능이 12%, 답변 분류 성능이 45% 향상되었습니다.

- **Implications**: 이번 연구는 PLM과 프로프트 기반 접근 방식이 아랍어 정신 건강 지원 분야에서 높은 잠재력을 가짐을 보여주며, 이를 통해 아랍어 사용자를 위한 접근성과 문화적으로 민감한 자원을 만드는데 매우 유용할 것입니다.



### Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration (https://arxiv.org/abs/2406.15951)
- **What's New**: 새롭게 제안된 '모듈러 플루럴리즘' (Modular Pluralism)은 다중-언어 모델 협업을 기반으로 하는 다원주의 정렬 프레임워크입니다. 이는 기본 대형 언어 모델(LLM)에 플러그인하여 작은 규모이지만 전문화된 커뮤니티 언어 모델(LMs) 풀과 협력하여 다양한 커뮤니티의 가치를 반영하고 지원할 수 있습니다. 이 프레임워크는 블랙박스 LLMs와도 호환되며, 기존에 충분히 대표되지 못한 커뮤니티를 위한 새로운 커뮤니티 LMs를 추가하여 모듈식 통제가 가능합니다.

- **Technical Details**: 모듈러 플루럴리즘에서는 기본 LLM이 블랙박스로 작동하고, 소규모 커뮤니티 LMs 그룹(C)은 각 커뮤니티별로 특화된 코퍼스를 사용하여 파인튜닝된 언어 모델입니다. 이 프레임워크는 주어진 사용자 쿼리에 대해 커뮤니티 LMs가 먼저 메시지를 생성하고, LLM이 이를 참고하여 응답을 생성합니다. 이러한 협업은 오버톤(Overton), 방향 조정(steerable), 분포(distributional)라는 세 가지 모드로 수행될 수 있어 다양한 플루럴리즘 목표를 달성할 수 있도록 돕습니다.

- **Performance Highlights**: 모듈러 플루럴리즘은 오버톤 플루럴리즘에서 평균 68.5%의 다양한 가치에 대한 커버리지를 향상시켰으며, 방향 조정 목표에서는 26.6%의 경우에 더 잘 조정된 응답을 생성하고, 분포 플루럴리즘에서는 10.9% 이상 더 나은 반영을 보였습니다. 또한, 이 시스템은 새로운 커뮤니티 LMs를 쉽게 추가하여 이전에 충분히 대변되지 못한 커뮤니티를 더 잘 모델링할 수 있습니다.



### Teaching LLMs to Abstain across Languages via Multilingual Feedback (https://arxiv.org/abs/2406.15948)
- **What's New**: 이 논문은 다중언어 LLMs(대형 언어 모델)에서 지식을 보완하는 새로운 접근법으로 다국어 피드백을 활용하여 지식 격차를 식별하고 잘못된 정보를 줄이는 전략을 제안합니다. 기존 연구는 주로 영어에 초점을 맞췄으나, 저자는 다국어 상황에서의 룰앤드 머신 러닝(Large Language Models, LLMs)이 저자원 언어에서 더 큰 지식 차이를 보인다는 사실에 주목했습니다.

- **Technical Details**: 이 연구는 질문 응답(Question Answering, QA) 과정에서 LLM이 스스로 반성하고 다국어 피드백을 생성하여 비교적 자원이 많은 언어와 연관된 피드백을 통해 자신의 제안된 답변의 올바름을 평가하는 방법을 연구합니다. 구체적으로 LLM은 질문을 받고 제안된 답변을 생성한 후, 다른 연관된 언어로 피드백을 생성하여 다문화와 언어적 관점에서 지식 격차를 파악합니다. 이 과정은 세 가지 단계로 이루어집니다: 1) 질문에 대한 제안된 답변을 생성, 2) 다국어 피드백을 생성, 3) 다국어 피드백을 바탕으로 자가 평가 및 판단.

- **Performance Highlights**: 다양한 벤치마크 실험 결과, 다국어 피드백 접근법이 기존의 강력한 기준선 모델들보다 저자원 언어에서 최대 9.2%의 성능 향상을 보였습니다. 이러한 결과는 다국어 피드백이 문화와 언어적 맥락을 더 잘 반영할 수 있는 효과적인 전략임을 보여줍니다. 또한, 다국어 피드백이 다양한 정보 도메인에서의 성능 격차를 줄이고 보다 공평한 방법임을 강조합니다.



### RuleR: Improving LLM Controllability by Rule-based Data Recycling (https://arxiv.org/abs/2406.15938)
- **What's New**: 대형 언어 모델(LLM)의 응답을 보다 정밀하게 제어할 수 있는 방법을 개발하기 위해 Rule-based Data Recycling (RuleR)를 제안했습니다. 기존 데이터 샘플에 정해진 규칙에 따라 여러 제약 조건을 적용하는 데이터 증강 방법입니다. RuleR은 새로운 데이터를 생성하는 대신 기존 데이터를 '재활용'하여, 응답을 간단히 수정하고 원래의 지침에 규칙을 추가합니다.

- **Technical Details**: RuleR의 핵심 아이디어는 인간이나 모델의 편집 대신 여러가지 정해진 규칙을 사용하여 제약 패턴을 정의하는 것입니다. 이러한 제약 조건을 충족시키기 위해 원래의 특성을 기반으로 응답을 선택적으로 수정합니다. 예를 들어, 원래의 지침 '어떤 종류의 과일이 과일 샐러드에 추가되면 좋을까요?'와 원래의 응답 '포도, 키위, 오렌지, 사과'가 있을 때, '명사의 수' 제약 조건을 적용하면 '응답에 3개 이상의 명사가 포함되도록 하세요'라는 새로운 지침이 추가됩니다. 여러 데이터셋에 대해 IFEval(Instruction-Following Eval) 벤치마크를 사용한 실험 결과, RuleR이 인간/모델 프리 방식으로 LLM의 제어 가능성을 향상시키는 데 효과적임을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RuleR은 다양한 데이터셋에서 LLM의 제어 가능성을 향상시키는 데 효과적임이 입증되었습니다. 또한 원래의 SFT 데이터셋이 제공하는 일반적인 지침 준수 능력도 유지되는 것으로 나타났습니다.



### Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs (https://arxiv.org/abs/2406.15927)
Comments:
          First three authors contributed equally

- **What's New**: 새롭게 제안된 논문은 대형 언어 모델(LLM)의 불확실성 정량화를 위한 저렴하고 신뢰할 수 있는 방법인 'Semantic Entropy Probes(SEPs)'을 소개합니다. SEPs는 단일 모델 생성의 숨겨진 상태에서 직접적으로 semantic entropy(SE)를 근사하여, SE를 계산하는 데 필요한 5배에서 10배의 계산 비용을 줄이며, 사실적 오류를 감지하는 데 효과적입니다.

- **Technical Details**: 기존의 SE 계산 방법은 다수의 모델 생성 샘플을 필요로 했지만, SEPs는 단일 생성의 숨겨진 상태를 활용하여 SE를 근사합니다. SEPs는 쉽게 훈련될 수 있으며, 테스트 시 여러 모델 생성을 샘플링할 필요가 없어 거의 제로에 가까운 오버헤드를 가집니다. 이는 모델의 은닉 상태가 의미적 불확실성을 캡처하는 것을 강조하며, 래이어와 토큰 위치에 걸쳐 이 기능을 유지하는지에 대한 조사도 포함돼 있습니다.

- **Performance Highlights**: SEPs는 사실적 오류 감지에 높은 성능을 유지하며, 기존의 정확도 예측 기반의 탐지 방법보다 out-of-distribution 데이터에 더 잘 일반화됩니다. 연구 결과에 따르면, SEPs는 비용 효율적인 환각 감지에서 새로운 상태를 설정했으며, 이는 여러 모델과 작업에 걸쳐 입증되었습니다.



### The Unlikely Duel: Evaluating Creative Writing in LLMs through a Unique Scenario (https://arxiv.org/abs/2406.15891)
Comments:
          Published in the XIX Conference of the Spanish Association for Artificial Intelligence (CAEPIA), 2024. Summary of our paper "A Confederacy of Models: a Comprehensive Evaluation of LLMs on Creative Writing", published in Findings of EMNLP

- **What's New**: 최근의 최첨단 교육 조정된 대형 언어 모델들(LLMs)의 성능을 창의적 글쓰기 과제에서 평가하고, 인간 작가와 비교한 연구가 발표되었습니다. 이 연구는 기존 이야기의 재사용을 최소화하고 창의성을 강조하기 위해 특수한 프롬프트를 사용했습니다. 이 프롬프트는 존 케네디 툴의 'A Confederacy of Dunces'의 주인공인 Ignatius J. Reilly와 피테로닥틸 간의 전투를 서술하는 것이었습니다.

- **Technical Details**: 연구에서는 12개의 LLMs(Alpaca, Bard, Bing Chat, ChatGPT with GPT-3.5, ChatGPT with GPT-4, Claude, Dolly 2.0, GPT4All-J, Koala, OpenAssistant, StableLM, Vicuna)를 평가 대상으로 하였습니다. 인간 작가와 LLMs의 창작물은 10명의 평가자가 정교한 평가 기준을 사용해 주관적으로 평가했으며, 각 모델은 프롬프트 하나당 5개의 이야기를 생성하도록 하여 랜덤성을 통제했습니다.

- **Performance Highlights**: ChatGPT with GPT-4가 전체적으로 가장 높은 점수를 기록하며 인간 작가를 약간 능가했습니다. GPT-4는 특히 일관성이 높았으며, 유머와 같은 어려운 분야에서도 우수한 성과를 보였습니다. 반면, 오픈 소스 LLM들은 상업 모델들보다 전반적으로 낮은 성적을 기록했습니다. 창의성과 독창성에서는 인간 작가가 여전히 우위를 유지했지만, 인간과 상위 3개의 LLM 사이의 차이는 통계적으로 유의미하지 않았습니다.



### Real-time Speech Summarization for Medical Conversations (https://arxiv.org/abs/2406.15888)
Comments:
          Interspeech 2024

- **What's New**: 이 논문에서는 산업 분야의 실제 적용을 위한 실시간 음성 요약 시스템(RTSS)을 제안합니다. 이는 대화 중 N개의 발화 후에 로컬 요약을 생성하고 대화가 끝난 후에 글로벌 요약을 생성하는 시스템으로, 비즈니스 측면에서는 사용자 경험을 향상시키고 기술적 측면에서는 계산 비용을 줄일 수 있습니다. 또한, 새로운 의료 대화 요약 데이터셋인 VietMed-Sum을 소개하고, ChatGPT와 사람 주석자들이 협력하여 의료 대화 요약을 위한 골드 표준 및 합성 요약을 생성하는 첫 시도를 수행했습니다.

- **Technical Details**: 기존의 RTSS 시스템은 유연한 발화 단위 인식기(flexible recognizer of utterance units), 발화 룩어헤드(utterance lookahead-er), 정보 오버라이더(information overrider) 등의 추가 컴포넌트를 사용하여 요약을 계속해서 업데이트합니다. 하지만 이러한 추가 컴포넌트는 추론 및 학습 시간을 연장시키고 시스템 복잡성을 증가시킵니다. 제안된 시스템은 매 N개의 발화 후에 로컬 요약을 생성하고 대화 종료 후에 글로벌 요약을 생성함으로써 더 단순한 접근 방식을 취합니다. 각 로컬 요약은 N개의 발화에 대한 로컬 컨텍스트만을 사용하며, 글로벌 요약은 전체 대화의 컨텍스트를 사용합니다.

- **Performance Highlights**: VietMed-Sum 데이터셋을 사용하여 다양한 최첨단 모델의 기준 성능을 제시합니다. 데이터셋은 실제 의료 ASR 데이터셋과 시뮬레이션된 데이터셋으로 구성되며, ChatGPT와 사람 주석자가 협력하여 골드 표준 및 합성 요약을 생성했습니다. 또한, 동일한 예산 하에서 인간 주석 요약과 GPT 주석 요약의 성능 차이를 평가했으며, $2.5 예산 설정의 실험 결과에서는 인간 주석의 중요성을 보여줍니다. GPT 주석은 추가적인 교육이 필요하지만 유용한 의료 지식을 제공할 수 있습니다.



### SimSMoE: Solving Representational Collapse via Similarity Measur (https://arxiv.org/abs/2406.15883)
- **What's New**: 이번 연구에서는 Sparse Mixture of Experts (SMoE)에서 발생하는 대표적인 문제인 표현 붕괴 문제를 해결하는 유사성 기반의 Sparse Mixture of Experts (SimSMoE)를 제안합니다. SimSMoE는 고정된 FLOPs 예산 내에서 전문가 간 표현의 유사성을 최소화함으로써 표현 붕괴를 방지합니다. 이를 통해, 기존의 SMoE 훈련 방법들보다 향상된 성능을 보여줍니다.

- **Technical Details**: SimSMoE는 중심 커널 정렬(CKA) 메트릭을 사용해 전문가들 간의 유사성을 정량화하고, 유사성을 최소화하는 CKA 손실 함수를 활용하여 표현 붕괴 문제를 해결합니다. 이는 전문가 레벨에서의 직접적인 개선을 목표로 하며, 모든 라우팅 알고리즘에 적용 가능합니다.

- **Performance Highlights**: 제안된 SimSMoE는 대형 언어 모델의 사전 학습 및 미세 조정 작업에서 기존 SMoE 훈련 방법들보다 뛰어난 성능을 나타냈습니다. 특히, 다양한 SMoE 아키텍처(GLaM, Brainformer, Mistral)에서의 실험을 통해 SimSMoE의 효능, 견고성 및 확장성을 입증하였습니다.



### Uncovering Hidden Intentions: Exploring Prompt Recovery for Deeper Insights into Generated Texts (https://arxiv.org/abs/2406.15871)
Comments:
          Accepted at WNNLP 2024

- **What's New**: 최근 AI가 생성한 콘텐츠를 탐지하는 것에 많은 관심이 집중되고 있습니다. 본 논문은 탐지를 넘어 텍스트를 생성하는데 사용된 프롬프트(Prompt)를 복구하는 시도를 처음으로 소개합니다. 제로샷(Zero-shot) 및 퓨샷(Few-shot) 인컨텍스트 학습(In-context Learning)과 LoRA(Low-Rank Adaptation) 파인튜닝(Fine-tuning)을 사용한 실험을 통해 프롬프트를 복구할 수 있는 가능성을 조사했습니다.

- **Technical Details**: 프롬프트 복구를 위해, Mistral-7B-Instruct 모델을 사용하여 다양한 인젝션 데이터 준비를 포함한 세미-합성(Semi-synthetic) 데이터셋을 생성했습니다. 실험은 제로샷 및 퓨샷 인컨텍스트 학습과 LoRA 파인튜닝을 포함하며, ROUGE-L, BERTScore, MiniLM 임베딩 코사인 유사도와 같은 정량적 평가 지표를 사용했습니다. 또한, 데이터셋은 databricks-dolly-15k 데이터셋을 기반으로 하여 9,000개의 프롬프트와 응답으로 구성되었습니다.

- **Performance Highlights**: 초기 실험 결과, 프롬프트 복구는 기대 이상의 성능을 보였습니다. LoRA 파인튜닝에서 ROUGE-L 점수는 0.47에 달했고, MiniLM 유사도와 BERTScore는 각각 0.83과 0.97을 기록했습니다. 세미-합성 데이터를 포함했을 때 성능이 더 향상되는 경향을 보였습니다.



### A multitask learning framework for leveraging subjectivity of annotators to identify misogyny (https://arxiv.org/abs/2406.15869)
- **What's New**: 이 논문은 여성에 대한 혐오를 식별하는 AI 시스템을 개선하기 위해 주관성을 활용하는 멀티태스킹 학습 접근법을 제안합니다. 주석자의 성별 및 연령에 따른 다양한 관점을 반영한 모델을 설계하여, 영어 트윗에서 혐오적 콘텐츠를 식별하는 네 가지 대안을 실험하고 오류 분석을 실시했습니다. 연구 결과, 다양한 관점을 통합하면 언어 모델이 다양한 형태의 여성 혐오를 더 잘 해석할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 영어 트윗 데이터셋을 사용하여 실험을 수행했습니다. 주석자는 성별(남성, 여성)과 연령 그룹(18-22, 23-45, 46+)에 따라 6개의 프로필 그룹으로 분류되었습니다. 멀티태스킹 학습 접근법은 언어 모델 기반의 두 개의 Transformer 모델에서 테스트되었으며, 한태스크 학습(single-task learning) 모델과 비교되었습니다. 주요 모델 아키텍처는 다음과 같습니다: 1) STL-full-FT (Baseline) - 단일 태스크 학습 모델. 2) MTL-six-aux - 6개의 보조 태스크를 포함하는 멀티태스킹 모델. 3) MTL-two-aux - 성별을 기준으로 응답을 두 개의 보조 태스크로 집계한 멀티태스킹 모델.

- **Performance Highlights**: 실험 결과, 멀티태스킹 학습 모델이 단일 태스크 학습 모델에 비해 여성 혐오 식별 성능이 향상되었습니다. 특히, 다양한 프로필에 따라 주석자의 관점을 반영한 MTL-six-aux 모델이 가장 높은 성능을 보였습니다. 또한, 주석자의 다양한 관점 통합이 AI 시스템의 편향 감소와 공정성 향상에 중요한 역할을 한다는 점이 강조되었습니다.



### Speech Analysis of Language Varieties in Italy (https://arxiv.org/abs/2406.15862)
Comments:
          Accepted to LREC-COLING 2024 - this https URL

- **What’s New**: 이탈리아의 다양한 지역 언어를 분석하기 위해 최신의 자가 지도 학습(self-supervised learning) 기법들을 적용하여, 언어 샘플의 지리적 출처를 자동으로 식별하는 연구가 진행되었습니다. 이 연구에서는 대규모 데이터에서 학습된 표현을 활용하여, 세밀하게 관련된 언어적 변이들 간의 차이점을 조사하고자 합니다.

- **Technical Details**: 이 연구는 텍스트 전사 없이 음성 신호의 음향적 특성만을 사용하여 음성 샘플의 지리적 출처를 자동으로 판별하는 것을 목표로 합니다. 이를 위해 VIVALDI 데이터셋을 활용했으며, 자가 지도 학습 모델과 대조 학습(contrastive learning) 기법을 결합하여 모델의 분류 정확도를 높이는 접근법을 탐구했습니다. 대조 학습 목표를 추가적인 사전 학습 단계 및 미세 조정 시 보조 손실로 적용하여 학습 표현의 구별 능력을 향상시켰습니다.

- **Performance Highlights**: 실험 결과, 사전 학습된 자가 지도 모델이 음성 녹음에서 지역을 효과적으로 식별할 수 있음을 보여줍니다. 특히 미세 조정 중에 보조 손실로서 대조 목표를 포함하면 분류 정확도가 향상되고, 지역 변이를 뚜렷하게 분리하는 임베딩을 산출할 수 있음을 확인했습니다.



### Revisiting Interpolation Augmentation for Speech-to-Text Generation (https://arxiv.org/abs/2406.15846)
Comments:
          ACL 2024 Findings

- **What's New**: 논문은 저자들이 사운드 내부(Frequency Domain) 및 텍스트 특징(Text Embedding) 간의 보간 증대법(Interpolation Augmentation, IPA) 적용을 통해, 저자원 환경(Low-Resource Scenario)에서의 음성 인식(Speech-to-Text, S2T) 시스템의 성능을 향상시키는 방법을 연구한 내용을 다룹니다. 특히, 기존의 S2T 시스템에서 충분히 탐구되지 않은 IPA 기법을 적용하여 S2T의 일반화(generalization) 능력을 강화하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 논문은 다양한 S2T 상황에서 IPA 기법을 어떻게 최적화할 수 있는지에 대한 가이드라인을 제시합니다. 연구자들은 LibriSpeech 100h ASR 데이터셋을 사용하며, IPA를 Enc-Dec(인코더-디코더) 모델과 Enc-CTC(연결주의 임시 분류) 모델에 적용했습니다. IPA는 두 랜덤 샘플의 입력 특징(input features)과 레이블(labels)을 선형 보간(linear interpolation)하여 새로운 가상 샘플을 생성하는 방식입니다.

- **Performance Highlights**: IPA를 통해 ASR 및 AST와 같은 다양한 S2T 작업에서 일관된 성능 향상을 달성했습니다. 주목할 만한 점은 IPA 기법이 다른 증대법(augmentation methods)과 독립적으로 그리고 함께 적용했을 때도 좋은 효과를 보였다는 것입니다. 저자들은 또한 특정 하이퍼파라미터인 알파(alpha) 및 감마(gamma)의 최적 값을 찾는 실험을 수행하여 그 중요성을 강조했습니다.



### CaT-BENCH: Benchmarking Language Model Understanding of Causal and Temporal Dependencies in Plans (https://arxiv.org/abs/2406.15823)
- **What's New**: CaT-Bench(Causal and Temporal Benchmark)은 요리 레시피 계획에서 단계 순서를 예측하도록 설계된 새로운 벤치마크입니다. 이는 LLMs(Large Language Models)가 인과적 및 시간적 종속성을 얼마나 잘 이해하는지 평가하는 데 사용됩니다.

- **Technical Details**: CaT-Bench는 57개의 독특한 계획에서 총 4260개의 질문으로 구성되어 있습니다. 이 질문들은 주어진 요리 레시피의 특정 단계가 반드시 다른 단계 이전 또는 이후에 실행되어야 하는지 테스트합니다. 이러한 테스트를 통해 LLMs가 인과적 및 시간적 연결성을 얼마나 잘 이해하는지 평가합니다.

- **Performance Highlights**: 최신 SOTA LLMs의 F1-점수는 제로-샷(zero-shot) 설정에서 0.59로 예상보다 낮게 나타났습니다. 몇-샷(few-shot) 예제와 설명을 추가하면 성능은 개선되지만, 최고 F1-점수도 0.73에 불과합니다. 인간 평가자들은 모델의 설명과 답변에 대해 일관되게 동의하지 않았으며, 체인-오브-생각(chain-of-thought) 프롬팅보다 '답변 후 설명' 접근 방식이 더 나은 성능을 보였습니다.



### LaMSUM: A Novel Framework for Extractive Summarization of User Generated Content using LLMs (https://arxiv.org/abs/2406.15809)
Comments:
          Under review

- **What's New**: 이번 연구에서는 LLMs (Large Language Models)을 활용한 새로운 추출 요약 (extractive summarization) 프레임워크 LaMSUM을 제안하였습니다. LaMSUM은 투표 알고리즘을 활용해 대규모 사용자 생성 텍스트의 추출 요약을 생성하는데 초점을 맞추고 있습니다. 이는 기존 LLM들이 주로 생성해왔던 추상 요약 (abstractive summarization)에서 더 나아가 추출 요약에도 활용할 수 있는 가능성을 제시합니다.

- **Technical Details**: LaMSUM은 다중 레벨 요약 모델로 구성되어 있으며, 대규모 텍스트를 작은 청크로 나눈 후 각 청크에서 요약을 생성한 뒤 이를 결합하여 최종 요약을 만듭니다. 이를 위해 LLM의 제한된 컨텍스트 윈도우를 넘어서기 위한 전략으로 다중 레벨 프레임워크와 투표 알고리즘을 사용합니다. 초기 레벨에서 시작하여 각 레벨에서 생성된 요약을 상위 레벨로 이동시켜 최종 요약 본문을 생성합니다.

- **Performance Highlights**: LaMSUM은 Llama 3, Mixtral, Gemini와 같은 세 가지 주요 오픈 소스 LLM을 대상으로 평가를 진행했으며, 기존의 추출 요약 방법들보다 뛰어난 성능을 보였습니다. 특히, 본 연구는 LLM이 생성하는 요약의 합리성을 설명하려는 시도도 포함하고 있어, 해당 분야에서 향후 연구의 방향성을 제시할 것으로 기대됩니다.



### Rethinking Entity-level Unlearning for Large Language Models (https://arxiv.org/abs/2406.15796)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)에서 개체 수준의 정보 잊기(Entity-level unlearning)라는 새로운 과제를 제안합니다. 기존 연구는 주로 인스턴스 수준의 잊기(instance-level unlearning)에 초점을 맞췄지만, 저작권 보호와 같은 실제 시나리오에서는 개체와 관련된 모든 정보를 삭제하는 것이 중요합니다. 이를 위해 가상의 개체(pseudo entities)를 도입하여 시뮬레이션을 수행하고, 최신 잊기 기법을 활용한 기준 방법을 개발하여 효과를 비교합니다.

- **Technical Details**: 본 연구는 두 단계로 이루어진 LLM 잊기 작업을 정의합니다. 첫 번째 단계는 잊기 세트 구성(Forget Set Construction)이며, 두 번째 단계는 잊기 실행(Unlearning Execution)입니다. 가상의 개체를 모델에 주입(fine-tuning)한 후, 일반적인 지식 추출 및 잊기 방법을 결합하여 두 단계를 위한 기준 방법을 구축하고 종합적인 평가를 수행합니다. 실험을 통해 현존하는 잊기 알고리즘이 개체 수준 잊기에서 효과적이지 않음을 확인했습니다. 또한, 잊기 세트의 크기와 지식 범위가 잊기 결과에 큰 영향을 미친다는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, 현존하는 방법은 인스턴스 수준 잊기에서만 효과적이며, 개체 수준 작업에는 잘 적용되지 않는다는 것을 알 수 있었습니다. 잊기 세트의 품질, 즉 크기와 지식 범위가 잊기 성능에 중요하게 작용합니다. 더욱 철저한 잊기 세트를 구축하면 보다 완전한 잊기를 달성할 수 있습니다. 또한, 추가적인 제약 조건이 모델의 일반적인 능력을 유지하는 데 도움이 됩니다. 예를 들어, 그라디언트 디센트(Gradient Descent) 방법이 Kullback-Leibler (KL) 발산 제약보다 더 효과적임을 발견했습니다.



### DABL: Detecting Semantic Anomalies in Business Processes Using Large Language Models (https://arxiv.org/abs/2406.15781)
- **What's New**: 다양한 도메인에서 수집된 143,137개의 실제 비즈니스 프로세스 모델을 통해 비즈니스 프로세스에서 의미적 이상을 탐지하는 새로운 접근 방식인 DABL을 소개합니다. 이 접근 방식은 최신 자연어 처리 모델인 대형 언어 모델(LLM)을 활용합니다. 일반적인 행동을 파악하고 명확한 이탈을 발견하기 위해 Llama 2 모델을 미세 조정하였으며, 추가 학습 없이 사용자 데이터셋에 직접 적용할 수 있습니다. 또한, DABL은 자연어로 이상 원인을 해석하여 가치 있는 통찰력을 제공합니다.

- **Technical Details**: 기존의 의미적 이상 탐지 방법들은 각 트레이스를 여러 이벤트 쌍으로 취급하여 장거리 종속성을 방해하지만, DABL은 이러한 문제를 개선하기 위해 LLM을 활용합니다. 우리는 다양한 도메인의 프로세스 모델 143,137개를 사용하여 정상 트레이스를 생성하고, 순서 이상 및 배제 이상을 시뮬레이션하여 Llama 2 모델을 미세 조정하였습니다. 이 과정에서 QLoRA 기법을 활용하여 LLM을 세부 조정함으로써, 다양한 도메인에서 적용 가능한 일반적인 모델을 생성하였습니다.

- **Performance Highlights**: 광범위한 실험 결과, DABL은 기존 최신 의미적 이상 탐지 방법들보다 일반화 능력과 주어진 프로세스 학습에서 월등한 성능을 보였습니다. 실제 데이터셋으로 테스트한 결과, DABL의 실용적 효과가 확인되었습니다. 따라서 추가 학습 없이도 사용자들이 자신의 데이터셋에 이 모델을 직접 적용할 수 있습니다.



### Ladder: A Model-Agnostic Framework Boosting LLM-based Machine Translation to the Next Lev (https://arxiv.org/abs/2406.15741)
Comments:
          Our code is available at this https URL

- **What's New**: 이 연구에서는 Ladder라는 비용 효율적이고 모델에 구애받지 않는 도구를 개발하여 일반 목적의 대형 언어 모델(LLMs)의 번역 성능을 향상시키는 방법을 제안합니다. Ladder는 기존 LLM에서 쉽게 얻을 수 있는 가짜 정제 삼중 데이터(pseudo-refinement triplets)를 활용하여 추가적인 인간 비용 없이 훈련됩니다. 계층적 세부 조정 전략(hierarchical fine-tuning)을 통해 점진적으로 성능을 향상시킵니다.

- **Technical Details**: Ladder는 [소스, 중간 번역, 참조]로 구성된 가짜 정제 삼중 데이터(pseudo-refinement triplets)를 생성하여 훈련 데이터로 사용합니다. 중간 번역은 기존 LLM에서 샘플링한 번역을 의미합니다. 세부 조정에는 쉬운(Easy), 보통(Medium), 어려운(Hard) 세 가지 계층으로 나뉘어 단계별로 성능을 향상시키는 전략을 사용합니다.

- **Performance Highlights**: Ladder-2B 모델은 Gemma-2B/7B 백본을 사용하여 번역 성능을 크게 향상시켜, BigTranslate-13B 모델과 비교했을 때 BLEU 점수 +6.91 및 COMET 점수 +3.52 만큼 향상되었습니다. Ladder-7B 모델은 최첨단 GPT-4 수준의 성능과 견줄 수 있을 정도로 성능을 향상시킵니다. 다양한 설정에서의 실험 결과는 Ladder의 효과를 입증합니다.



### RankAdaptor: Hierarchical Dynamic Low-Rank Adaptation for Structural Pruned LLMs (https://arxiv.org/abs/2406.15734)
- **What's New**: 새로운 방법 RankAdaptor(랭크어답터)을 소개합니다. 이는 구조적 프루닝(pruning)이 적용된 대형 언어 모델(LLM)을 위한 효율적인 파인튜닝(fine-tuning) 방법으로, 계층적 동적 랭크 스케줄링(hierarchical dynamic rank scheduling)을 도입하여 기존의 고정 랭크를 사용하는 Low-Rank Adaptation (LoRA)의 한계를 극복하고자 합니다.

- **Technical Details**: RankAdaptor는 가벼운 성능 모델(lightweight performance model)을 사용하여 파인튜닝 과정에서 각 레이어의 최적 랭크를 자동으로 결정하는 엔드투엔드 최적화 흐름을 개발하였습니다. 구조적 프루닝이 된 LLM의 다양한 프루닝 설정에서 고정된 랭크를 사용하는 표준 LoRA보다 더 나은 성능을 보입니다. 이 방식은 모델의 학습 가능한 파라미터를 증가시키지 않으면서도 프루닝된 모델과 원본 모델 간의 성능 격차를 줄일 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 실험에서 RankAdaptor는 표준 LoRA보다 일관되게 더 우수한 성능을 보였습니다. 예를 들어, BoolQ 태스크에서 LLaMA-7B를 20%와 30% 프루닝한 경우, RankAdaptor는 각각 92.13%와 90.59%의 정확도를 회복했으며, 이는 표준 LoRA의 86.6%와 85.44%를 상회하는 결과입니다.



### Acoustic Feature Mixup for Balanced Multi-aspect Pronunciation Assessmen (https://arxiv.org/abs/2406.15723)
Comments:
          Interspeech 2024

- **What's New**: 이 논문에서는 자동 발음 평가에서 데이터 부족과 점수 불균형 문제를 해결하기 위한 두 가지 Acoustic Feature Mixup 전략을 제안합니다. 이 전략은 고유의 발음 평가에 적합하도록 선형 및 비선형 보간을 사용하여 데이터 분포 변화를 시뮬레이션합니다. 특히, 발음의 적합성(Goodness of Pronunciation, GOP) 특징과 오류율을 통합하여 향상된 성능을 실현하고 있습니다.

- **Technical Details**: 제안된 두 가지 Acoustic-feature Mixup (AM) 전략은 정적 AM (static AM)과 동적 AM (dynamic AM)입니다. 정적 AM은 단순 선형 변환을 사용하여 데이터를 보간하며, 동적 AM은 비선형 보간을 추가하여 데이터 분포를 변환합니다. 주로 선수가 발음한 음소와 정답을 비교하여 도출된 GOP 특징을 사용하며, 이는 원본 음성 데이터 없이 점수를 분류하는데 큰 도움이 됩니다. 또한 자동 음성 인식(ASR) 시스템의 결과와 정답 음소를 비교하여 세밀한 오류율 정보를 제공하여 오발음을 직접적으로 탐지할 수 있게 합니다.

- **Performance Highlights**: Speechocean762 데이터셋을 이용한 광범위한 실험 결과, 본 논문의 AM 전략이 여러 측면에서 발음 평가 성능을 크게 향상시켰습니다. 특히, 점수 분포가 불균형한 스트레스와 완결성 측면에서 성능이 크게 개선되었으며, 기존 분포를 시각화하고 분포 이동 효과를 보여줌으로써 제안된 방법의 유효성을 입증했습니다.



### Scaling Laws for Fact Memorization of Large Language Models (https://arxiv.org/abs/2406.15720)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 사실 지식(fact knowledge) 암기 동작을 분석하고 이를 위한 스케일링 법칙(scaling laws)을 제시합니다. 특히, LLM이 위키데이터(Wikidata)의 모든 사실을 암기하는 것은 일반적인 사전 학습 환경에서는 거의 불가능하다는 것을 발견했습니다. 또한, LLM은 새로운 사실에 대한 일반화 능력도 가지고 있으며, 이 일반화 법칙은 일반 사전 학습법과 유사합니다.

- **Technical Details**: 사실 지식 용량은 모델 크기와 학습 에포크(epoch)에 따라 각각 선형적 및 음의 지수 관계를 가집니다. 모든 위키데이터의 사실을 암기하려면 약 1000억 개의 파라미터를 가지고 100 에포크 동안 학습해야 합니다. 사실 지식의 예를 정확하게 예측할 수 있는지 여부를 통해 LLM의 사실 암기 동작을 분석했습니다. 즉, (SpaceX, CEO, Elon Musk)와 같은 (키, 속성, 값) 삼중(triples)을 사용하여 사실을 암기하는 능력을 평가했습니다.

- **Performance Highlights**: LLM은 중복된 사실을 효율적으로 암기하는 데 어려움을 겪으며, 자주 등장하는 어려운 사실에 더 많은 관심을 기울입니다. 또한, 연속적으로 학습되는 사실은 이전에 암기한 사실을 덮어쓸 수 있으며, 이는 저빈도 사실(low-frequency facts) 암기를 방해합니다. 놀랍게도, LLM은 새로운 사실에 대해서도 높은 수준의 일반화 가능성을 보이며, 특정 유형의 사실은 매우 높은 일반화 가능성을 나타냅니다.



### Beyond the Turn-Based Game: Enabling Real-Time Conversations with Duplex Models (https://arxiv.org/abs/2406.15718)
- **What's New**: 이번 연구에서는 기존의 대형 언어 모델(LLMs)을 '듀플렉스 모델'로 변환하여, 사용자와의 상호작용 동안 시스템이 동시에 들을 수 있고 즉각적인 피드백을 제공할 수 있도록 조정하는 방법을 소개합니다. 이를 통해 보다 자연스럽고 인간처럼 느껴지는 대화 경험을 제공하고자 합니다. 연구진은 듀플렉스 모델과 함께 새로운 데이터셋을 발표할 예정입니다.

- **Technical Details**: 듀플렉스 모델은 시간분할 다중화(Time-Division-Multiplexing, TDM) 인코딩-디코딩 전략을 채택하여 대화를 여러 시간 조각으로 분할하고 이를 동시에 처리합니다. 이러한 시간 조각을 처리하는 동안 모델은 즉각적인 응답을 위해 입력 메시지를 부분적으로 처리합니다. 이를 위해 연구진은 '듀플렉스 정렬(Duplex Alignment)'과 '듀플렉스 SFT 데이터셋'을 사용하여 기존 LLM을 조정했습니다.

- **Performance Highlights**: 실험 결과, 듀플렉스 모델은 기존 벤치마크에서의 성능을 유지하는 동시에 사용자 요청에 대한 동적 응답 기능을 제공함을 확인했습니다. 자동 평가 및 사용자 평가 모두 듀플렉스 모델이 기존의 LLM에 비해 사용자 만족도 및 응답성이 크게 향상되었음을 보여줍니다. 예시로 개발된 MiniCPM-duplex 모델은 이러한 개선된 성능을 입증하고 있습니다.



### Teach Better or Show Smarter? On Instructions and Exemplars in Automatic Prompt Optimization (https://arxiv.org/abs/2406.15708)
- **What's New**: 최근 대형 언어 모델(large language models)이 주목할 만한 성능을 보여주고 있으나, 이는 효과적인 프롬프트 엔지니어링(prompt engineering)에 크게 의존하고 있습니다. 이 논문은 자동 프롬프트 최적화(automatic prompt optimization, APO) 방법을 설명하며, 특히 명령 최적화(instruction optimization, IO)와 예시 선택(exemplar selection, ES)을 비교 분석합니다. 두 방법은 독립적으로 발전했지만, ES를 통한 접근법이 IO보다 성능이 뛰어나다는 것을 밝혀냈습니다. 또한, 적절한 ES와 IO의 조합은 개별 기법보다 뛰어난 성능을 발휘한다는 것을 확인했습니다.

- **Technical Details**: 기존 연구는 IO에 더 많은 관심을 가졌으나, 이 논문에서는 ES가 IO보다 더 큰 영향을 미칠 수 있다는 점을 지적합니다. IO와 ES를 개별적으로 그리고 함께 테스트하여 다양한 도전 과제에서 이들의 성능을 비교했습니다. 모델이 유효성(validity) 세트에서 생성한 입력-출력 쌍을 예시로 재사용하면 성능이 지속적으로 향상된다는 점을 발견했습니다.

- **Performance Highlights**: 단순한 ES 전략, 예를 들어 랜덤 탐색(random search)이 최첨단 IO 방법보다 뛰어난 성능을 발휘할 수 있다는 점을 확인했습니다. 또한, ES와 IO의 최적 조합은 IO와 ES 각각의 단독 사용보다 높은 성능을 나타냈습니다. 이러한 결과는 예시 선택(exemplar selection)을 독립된 방법으로 연구하고, IO와의 최적 조합을 찾는 것이 매우 중요하다는 것을 시사합니다.



### SS-Bench: A Benchmark for Social Story Generation and Evaluation (https://arxiv.org/abs/2406.15695)
- **What's New**: ASD(Autism Spectrum Disorder) 아이들이 이해하기 어려운 사회적 상황을 돕기 위해 심리학 전문가가 작성하는 Social Stories는 중요하지만, 비용이 많이 들고 다양성과 시의성이 제한적입니다. 이에 따라 자동화되고 저렴하며 접근 가능한 방법을 사용하여 실시간 Social Stories 생성을 목표로 하는 SS-Bench를 소개합니다.

- **Technical Details**: SS-Bench는 LLMs(Large Language Models)가 Social Stories를 생성하고 평가하는 벤치마크입니다. 우리는 
 **StarSow** 라는 제약 기반의 전략을 개발하여 179개의 수작업으로 작성된 Social Stories로부터 LLMs를 계층적으로 프롬프트하여 Social Stories를 생성하고, 이를 통해 5천개 이상의 Social Stories 벤치마크 데이터셋을 구축했습니다. 또한, Human 및 GPT 평가를 통해 사회적 이야기의 효과를 검증하기 위한 **Quality Assessment Criteria**를 도입했습니다.

- **Performance Highlights**: 실험 결과, StarSow를 통해 확장된 SS-Bench 데이터셋은 더 다양하고 유연하며 효과적임이 밝혀졌습니다. 특히 작은 모델과 사전 훈련된 기본 모델에서 SS-Bench가 SSGen(Social Story Generation) 성능을 크게 향상시킵니다. 또한, SS-Bench 데이터셋으로 미세 조정된 언어 모델은 더 간단한 프롬프트로 소셜 스토리를 사용자 정의할 수 있음을 나타냈습니다.



### Large Language Models have Intrinsic Self-Correction Ability (https://arxiv.org/abs/2406.15673)
Comments:
          in submission

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 본질적인 자기 수정 능력에 대한 새로운 관점을 제시합니다. 특히 외부 지식을 사용하지 않는 자기 수정 방법인 본질적 자기 수정을 이론적 분석과 실험을 통해 탐구합니다. 이 논문은 공정한 프롬프트(공정한 질문)를 사용하고 'zero temperature' 설정을 가지는 두 가지 중요한 요소를 밝힙니다.

- **Technical Details**: 연구에서는 LLM이 'self-correction'을 수행할 때의 효율성을 분석합니다. 이론적으로는 비정상적인 응답(hallucination)이 정확도를 감소시킨다고 설명합니다. 실험 데이터는 CommonSense QA 데이터셋을 사용하여, 다양한 온도 설정과 편향된 프롬프트가 LLM의 자기 수정 능력에 미치는 영향을 분석하였습니다. 비례적으로 높은 정확도를 유지하려면 'zero temperature'와 공정한 프롬프트가 중요하다는 결론을 도출합니다.

- **Performance Highlights**: GPT-3.5와 GPT-4 모델은 실험을 통해 성능 변화를 분석하였습니다. GPT-3.5의 경우, 온도 증가로 인해 정확도가 뚜렷하게 감소했으나, GPT-4는 상대적으로 안정적이었습니다. 이는 응답 생성 과정에서 온도가 미치는 영향이 크다는 것을 시사합니다. 비편향 프롬프트 사용 시 더 나은 결과를 보였으며, 이는 비편향 프롬프트와 'zero temperature' 설정이 자기 수정 능력향상에 필수적임을 보여줍니다.



### PI-Whisper: An Adaptive and Incremental ASR Framework for Diverse and Evolving Speaker Characteristics (https://arxiv.org/abs/2406.15668)
Comments:
          11 pages, 3 figures

- **What's New**: 최신 연구에서는 자동 음성 인식(ASR) 기술이 점점 더 지능형 개인 비서를 개발하는 데 이용됨에 따라, 자원 제약이 있는 ASR 모델에서 적응성(adaptivity), 점진적 적용 가능성(incrementality), 포괄성(inclusivity)이라는 세 가지 중요한 문제를 해결하는 새로운 ASR 프레임워크, PI-Whisper를 제안합니다.

- **Technical Details**: PI-Whisper는 실시간으로 다양한 화자의 특성을 식별하여 ASR의 인식 능력을 적응적으로 개선할 수 있으며, 반복적인 재훈련 없이 점진적으로 적용할 수 있습니다. 또한, 다양한 화자 그룹에 대한 형평성과 공정성을 향상시킬 수 있습니다.

- **Performance Highlights**: 특히, PI-Whisper 프레임워크는 최신 정확도를 유지하면서 단어 오류율(WER)을 최대 13.7%까지 감소시킬 수 있으며, 컴퓨팅 자원에 대해 선형적인 확장성을 가지는 등 탁월한 성능을 보여줍니다.



### Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph (https://arxiv.org/abs/2406.15627)
Comments:
          Roman Vashurin, Ekaterina Fadeeva, Artem Vazhentsev contributed equally

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs, Large Language Models)의 텍스트 생성 작업에서 불확실성 정량화(UQ, Uncertainty Quantification) 방법을 평가하는 새로운 벤치마크를 도입했습니다. 이는 기존의 연구가 조각나고 일관되지 않은 평가 방법을 사용한 문제를 해결하기 위함입니다. 새 벤치마크는 최신 UQ 기준들을 통합하고, 다양한 텍스트 생성 작업에서 새로운 기법들을 일관되게 평가할 수 있는 환경을 제공합니다.

- **Technical Details**: 도입된 벤치마크는 여러 최신 UQ 기준들을 구현하며, 연구자들이 다양한 텍스트 생성 작업에서 새로운 기법을 통제 가능하고 일관되게 평가할 수 있는 환경을 제공합니다. 또한, 자신감(normalization) 방법이 해석 가능한 점수를 제공하는 능력을 평가할 수 있도록 지원합니다. 대규모 실증 조사(empirical investigation)를 통해 UQ와 자신감(normalization) 기술의 유망한 접근법들을 조명하였습니다.

- **Performance Highlights**: 새로운 벤치마크를 사용하여, 저자들은 9가지 작업에 걸쳐 대규모 실증 조사를 수행했습니다. 이 조사를 통해 가장 유망한 UQ와 자신감(normalization) 기법들을 밝혀냈습니다. 이로 인해 LLMs의 텍스트 생성 작업에서 나타나는 여러 문제들을 해결하는데 중요한 발걸음을 내딛었습니다.



### Shortcomings of LLMs for Low-Resource Translation: Retrieval and Understanding are Both the Problem (https://arxiv.org/abs/2406.15625)
Comments:
          Under review

- **What's New**: 이번 연구는 사전 학습된 대형 언어 모델들(LLMs)의 인-컨텍스트 학습(in-context learning) 능력을 조사하였습니다. 특히 자동화된 기계 번역 파이프라인에서 저자원 언어에서 고자원 언어로의 번역 지시 시 어떤 성과를 보이는지를 실험했습니다. 남부 케추아어(Southern Quechua)에서 스페인어로 번역하는 실험을 통해 다양한 유형의 정보(예: 형태소 번역, 문법 설명, 병렬 코퍼스 예문)가 번역 품질에 미치는 영향을 분석하였습니다.

- **Technical Details**: 문맥 유형(예: 형태소 번역, 문법 설명, 코퍼스 예문), 검색 방법(자동 대 수동), 모델 유형을 조작하는 절제 연구(ablation studies)를 진행했습니다. 또한 자동 및 인간 평가를 통해 모델 출력을 평가하였습니다. 이를 통해 저자원 언어 번역에 대한 다차원적 분석을 시도했습니다.

- **Performance Highlights**: 형태소 및 단어 수준 번역 프롬프트는 모델 출력을 일관되게 개선했지만, 문법과 코퍼스 정보는 오히려 결과에 부정적인 영향을 미치는 경우가 많았습니다. 자동 검색을 통해 수집된 형태소 번역이 포함된 프롬프트는 번역 품질을 현저히 개선시켰습니다. 모델 크기가 커짐에 따라 번역 품질이 향상되었지만, 이는 모델이 프롬프트 문맥을 잘 활용하는 능력 때문이 아니라 모델 사전 훈련 시 저자원 언어에 노출된 경험 때문인 것으로 보입니다.



### News Deja Vu: Connecting Past and Present with Semantic Search (https://arxiv.org/abs/2406.15593)
- **What's New**: 새로운 연구에서는 'News Deja Vu'라는 혁신적인 의미 기반 검색 도구를 소개합니다. 이 도구는 transformer 대형 언어 모델과 bi-encoder 접근법을 활용하여 현대 뉴스 쿼리에 가장 유사한 역사적 뉴스 기사를 식별합니다. News Deja Vu는 엔티티 인식을 통해 특정 명명된 엔티티가 아니라 더 넓은 유사성을 중점으로 두고 검색을 수행합니다. 이는 사회 과학자들이 과거의 사건과 현재의 사건을 비교하는 데 큰 도움을 줍니다.

- **Technical Details**: 이 모델은 contrastive training을 사용하며, 가벼운 bi-encoder를 통해 현대 뉴스 기사의 semantic nearest neighbor를 대규모 역사적 텍스트 데이터베이스에서 검색합니다. named entity recognition(NER) 모델을 사용하여 엔티티를 마스킹하고, IndexFlatIP index from FAISS (Johnson et al., 2019)를 통해 K-nearest neighbor 검색을 수행합니다. Roberta-Large 모델을 파인튜닝(fine-tune)하여 90.4의 F1 점수를 달성하며, OCR 오류에 강한 성능을 발휘합니다.

- **Performance Highlights**: News Deja Vu는 현대 뉴스 기사와 유사한 역사적 기사를 매우 효과적으로 검색합니다. 특히 이 도구는 Roberta-Large 모델보다 훨씬 높은 정확도를 자랑하며, 엔티티 인식을 통해 검색의 정밀도를 높였습니다. 학문의 용도로 쉽게 사용할 수 있게 설계되어 있어, 깊은 학습이나 복잡한 프로그래밍 지식 없이도 사용할 수 있습니다. 특정 주제나 사건에 대한 과거와 현재의 인식을 비교하는 데 유용한 강력한 도구입니다.



### TinyStyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings (https://arxiv.org/abs/2406.15586)
- **What's New**: 텍스트 스타일 변환(text style transfer)의 목표는 텍스트의 원래 의미를 유지하면서 스타일을 변환하는 것이며, 종종 목표 스타일의 몇 가지 예시만으로 수행됩니다. 기존의 스타일 변환 방법은 대규모 언어 모델의 few-shot(소규모 학습) 능력이나 복잡한 컨트롤 가능한 텍스트 생성 방식을 의존하지만, 이는 비효율적이며 유창성(fluency) 지표에서도 성능이 떨어집니다. TinyStyler는 작은 언어 모델(800M 파라미터)과 사전 학습된 저작권 임베딩(authorship embeddings)을 활용하여 효율적이고 few-shot 텍스트 스타일 변환을 수행하는 경량화된 효과적인 접근법을 소개합니다.

- **Technical Details**: TinyStyler는 800M 파라미터를 가진 작은 언어 모델(small language model)과 사전 학습된 저작권 임베딩을 활용하여 효율적이고 정확한 텍스트 스타일 변환을 수행합니다. 이는 기존의 큰 언어 모델과 복잡한 텍스트 생성 접근법의 비효율성과 성능 저하 문제를 해결합니다. 특히, 저작권 스타일 변환(authorship style transfer)과 텍스트 속성 스타일 변환(text attribute style transfer)에 초점을 맞추고 있습니다.

- **Performance Highlights**: TinyStyler는 저작권 스타일 변환 과제에서 GPT-4와 같은 강력한 접근법을 능가하는 성능을 보였습니다. 또한, 자동과 인간 평가 모두에서 최근의 컨트롤 가능한 텍스트 생성 방법을 능가하는 결과를 보였습니다. 이 모델은 공식적인(formal) 스타일과 비공식적인(informal) 스타일 간의 텍스트 속성 스타일 변환에서도 우수한 성능을 입증했습니다.



### Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods (https://arxiv.org/abs/2406.15583)
- **What's New**: 최근 몇 년간 거대 언어 모델(Large Language Models, LLMs)의 텍스트 생성 능력이 극적으로 향상되었습니다. 이제 사람들조차 AI가 생성한 텍스트와 인간이 작성한 텍스트를 확실히 구분하기 어려운 수준에 이르렀습니다. LLM의 발전으로 인해 정보 오염(information pollution)의 위험이 증가하고 있으며, 이에 따라 AI가 생성한 텍스트(AI-generated Text, AIGT)를 검출하는 기술이 중요해졌습니다.

- **Technical Details**: AIGT 검출은 텍스트 분류 작업으로, 입력 텍스트 시퀀스에 대해 'AI' 또는 '인간' 클래스를 예측하는 것입니다. 주로 이진 분류 문제로 다루어지지만, AI 영향 수준이나 특정 AI 모델을 예측하는 멀티클래스 문제로 다룰 수도 있습니다. 검출 방법은 워터마킹(watermarking), 통계적 및 스타일 분석(statistical and stylistic analysis), 그리고 머신 러닝 분류기(machine learning classification) 등으로 분류됩니다.

- **Performance Highlights**: 현재의 최첨단 생성 모델은 사람의 허위 정보보다 더 신뢰할 수 있고 유창한 가짜 정보를 생성할 수 있으며, 이는 인간 독자와 자동 검출 시스템 모두에게 탐지하기 어렵게 만듭니다. 다양한 검출 시나리오와 이를 위한 데이터셋도 논의되었으며, 텍스트 생성 모델의 크기와 디코딩 전략, 텍스트의 언어, 문서 길이, 인간의 영향력 정도, 적대적 전략 등 다양한 요소가 AIGT 검출의 난이도에 영향을 미친다는 사실이 강조되었습니다.



### Contrastive Entity Coreference and Disambiguation for Historical Texts (https://arxiv.org/abs/2406.15576)
- **What's New**: 이 논문은 역사적 문서 내 인물 간의 교차 문서 식별 및 해상도 문제를 해결하기 위해 세 가지 주요 기여를 합니다. 1) Wikipedia 문맥과 페이지에서 1억 9천만 개 이상의 엔티티 쌍을 추출한 초대형 훈련 데이터셋, 2) 손으로 라벨링된 역사적 뉴스 기사 데이터의 고품질 평가 데이터, 3) 이러한 데이터를 활용해 훈련되고 평가된 모델들입니다. 이 연구는 역사적 텍스트에서 지식 베이스 외부 인물들을 식별하는 데 정확하고 확장 가능한 성능을 나타냅니다.

- **Technical Details**: 대규모 역사 문서 컬렉션은 사회 과학 연구에 중요한 자원입니다. 그러나 이들 문서 대부분은 문서 간 고유 식별자가 부족합니다. 이 논문에서는 역사적 문서에서의 엔티티 불명확성을 해결하기 위해, 하드 네거티브를 포함한 대규모 훈련 데이터셋, 고품질 평가 데이터, 그리고 대비적으로 훈련된 바이엔코더(bi-encoder) 모델을 제안합니다. 이 모델은 동일한 엔티티를 나타내는 문맥들을 임베딩 공간에서 가깝게, 다른 엔티티를 나타내는 문맥들은 멀리 위치시킵니다. FAISS 백엔드를 사용하여 대규모 데이터셋에서 효율적으로 작업합니다.

- **Performance Highlights**: 우리의 접근 방식은 Entity of the Union 역사 벤치마크에서 다른 엔티티 해상도 모델을 훨씬 능가합니다. 또한 MSNBC와 ACE2004와 같은 현대 뉴스 데이터셋에서도 우수한 성능을 보입니다. 역사적 뉴스 데이터셋에 LinkNewsWikipedia 모델을 적용한 결과, 고유 인물 태깅을 통해 얻을 수 있는 유익한 인사이트를 제공합니다. 우리의 데이터셋과 모델은 오픈 소스로 제공되며 CC-BY 라이센스로 공개됩니다.



### DEM: Distribution Edited Model for Training with Mixed Data Distributions (https://arxiv.org/abs/2406.15570)
- **What's New**: 이 연구에서는 서로 다른 데이터 소스에서 각각 모델을 개별적으로 학습한 후, 이들을 기본 모델에 결합하는 간단하면서도 효율적인 방법을 제안합니다. 이 접근법은 Distribution Edited Model(DEM)이라고 불리며, 기존의 데이터 믹싱(data mixing) 기법보다 11배 비용 효율적이며 다양한 벤치마크에서 뛰어난 성능을 발휘합니다. DEM은 데이터 소스를 수정할 때 전체 재학습이 필요 없기 때문에 매우 유연하고 확장성이 뛰어납니다.

- **Technical Details**: DEM은 기본 모델을 데이터 소스별로 개별적으로 미세 조정하고, 미세 조정된 모델에서 기본 모델을 뺀 후 해당 분포 벡터를 추출합니다. 최종 모델은 이 분포 벡터들을 가중치 조합하여 기본 모델에 더함으로써 생성됩니다. 이 방법은 여러 데이터 소스를 결합하여 조인트 분포(joint distribution)를 효과적으로 잡아내며, 새로운 데이터셋 추가 시 점진적 업데이트가 가능하도록 설계되었습니다.

- **Performance Highlights**: DEM은 기존의 데이터 믹싱 방법에 비해 MMLU에서 최대 6.2%, BBH에서 11.5%, DROP에서 16.1%, HELM에서 9.3% 성능 향상을 보입니다. 또한, 이 방법은 3B, 7B, 13B 크기의 모델들에서도 효과적으로 작용합니다.



### Rethinking Pruning Large Language Models: Benefits and Pitfalls of Reconstruction Error Minimization (https://arxiv.org/abs/2406.15524)
- **What’s New**: 이 연구는 현재 널리 사용되는 대형 언어 모델 (LLMs) 가지치기(pruning) 방법의 근본적인 재고를 제안합니다. 제안된 새로운 접근 방식은 '분할 및 정복(divide and conquer)'을 이용해 모델을 하위 모델로 나누고, 이를 순차적으로 가지치기 하여 결과를 합치는 방식입니다. 이 방법은 메모리 제약 하에서 가지치기를 가능하게 하지만 높은 재구성 오류를 발생시킵니다. 저자는 이러한 오류를 90% 이상 줄일 수 있는 다양한 재구성 기술을 소개하고, 재구성 오류를 최소화하는 것이 항상 이상적이지 않음을 발견하며, 자기 생성 보정 데이터(self-generating calibration data) 전략이 이 문제를 완화할 수 있음을 제안합니다.

- **Technical Details**: 기존의 LLM 가지치기 방법은 extensive training과 대규모 데이터가 필요하나, 이는 높은 메모리 요구사항 때문에 쉽지 않습니다. 이를 해결하기 위해, 모델을 여러 하위 모델로 나누고 각 하위 모델을 개별적으로 가지치기하고 재구성하여 최종적으로 합치는 방식의 분할 및 정복 접근법이 사용됩니다. 이 접근 방식은 각 하위 문제에 대해 비제로 재구성 오류를 발생시켜 높은 합성 오류를 만들 수 있습니다. 이를 해결하기 위해 저자는, 블록 단위의 재구성(block-wise reconstruction, BR), 글로벌 전파(global propagation, GP), 블록 간의 중첩(overlapping)과 같은 다양한 최적화 기법을 사용하여 오류를 줄였습니다. 또한, 재구성 오류를 최소화하는 것이 항상 이상적이지 않으며, 과적합(overfitting)을 초래할 수 있음을 실험적으로 확인했습니다.

- **Performance Highlights**: 저자는 재구성 오류를 줄이는 기술들 중 블록 단위의 재구성(BR)과 글로벌 전파(GP)가 특히 효과적임을 발견했습니다. 본 연구에서 제안된 재구성 기술들은 기존 방법에 비해 재구성 오류를 90% 이상 줄이는 데 성공했습니다. 하지만, 높은 재구성 정확도가 항상 좋은 성능을 보장하는 것은 아니며, 오히려 특정 다운스트림 작업에서 퍼포먼스가 저하될 수 있음을 발견했습니다. 본 연구 결과는 새로운 가지치기 방법의 장단점을 종합적으로 파악하는 데 중요한 정보를 제공하며, 향후 연구에 유용한 방향을 제시합니다.



### Steering Without Side Effects: Improving Post-Deployment Control of Language Models (https://arxiv.org/abs/2406.15518)
- **What's New**: 새로운 언어 모델 포스트 배포(post-deployment) 행동 제어 기법 소개. KL-then-steer(KTS) 기법은 문제가 될 가능성이 있는 입력을 분류하여 모델 성능 저하 없이 제어하는 방법을 제안합니다. 이를 통해 모델의 불법적인 사용(jailbreak)을 44%까지 감소시키면서 모델의 유용성을 거의 유지합니다.

- **Technical Details**: KTS 접근법은 benign(문제가 없는) 입력에 대해 steered(조작된) 모델과 unsteered(조작되지 않은) 모델 간의 Kullback-Leibler (KL) divergence를 최소화하여 모델을 먼저 훈련한 후 조작된 모델을 사용합니다. 이는 기존 모델의 대표 벡터(hidden states)에 특정 벡터를 추가하는 방식으로 수행되며, 제어의 강도를 벡터의 크기를 조절하여 조정할 수 있습니다. 여러 방식들과 비교하여 합리적인 성능을 유지합니다.

- **Performance Highlights**: KL-then-steer(KTS) 기법은 jailbreak 공격 성공률을 원래 모델에 비해 44% 감소시키면서, 유용성 측면에서는 MT-Bench 점수에서 1.5% 감소에 불과합니다. 이는 모델의 일반적 성능을 크게 유지하면서도 안전성을 높일 수 있음을 보여줍니다. 또한 이 방법은 TruthfulQA와 같은 다른 문제에서도 높은 정확도를 유지하며, 사용자가 제시한 답변에 대한 치우침(bias)을 줄이는 데에도 효과적입니다.



### System Description for the Displace Speaker Diarization Challenge 2023 (https://arxiv.org/abs/2406.15516)
- **What's New**: 이번 논문에서는 Displace 2023 대회에서 대화 환경에서의 화자 및 언어 다이어리제이션 문제 해결을 위한 솔루션을 소개하고 있습니다. 이 솔루션은 VAD (음성 활동 감지), Resnet 기반 CNN을 사용한 특징 추출, 그리고 스펙트럴 클러스터링을 활용합니다. 특히, 힌디어 데이터 없이 훈련되었음에도 불구하고 뛰어난 성능을 발휘했습니다.

- **Technical Details**: 다이어리제이션 알고리즘은 보통 세 가지 구성 요소로 이뤄집니다. 이 연구에서는 Silero VAD v4 모델과 WebRTC VAD를 비교 실험했습니다. 특징 추출기는 VoxCeleb2 데이터셋과 Common Voice Corpus 12.0 데이터를 활용하여 Resnet-34와 Resnet-293 모델을 훈련시켰습니다. 모델 훈련에는 AAM-Softmax Loss를 사용하였으며 8개의 NVIDIA Tesla A100 40GB GPUs로 Resnet-34는 18시간, Resnet-293은 97시간 동안 150 에폭 동안 훈련되었습니다. 최종 클러스터링에는 스펙트럴 클러스터링을 사용했습니다.

- **Performance Highlights**: 제안된 알고리즘은 개발 데이터셋과 1단계 평가 데이터셋에서 각각 DER 27.1%와 DER 27.4%를 달성했습니다. 이는 다중 언어 환경에서 경쟁력 있는 성능을 보이는 결과입니다.



### Few-shot Knowledge Graph Relational Reasoning via Subgraph Adaptation (https://arxiv.org/abs/2406.15507)
- **What's New**: 최근 Few-shot Knowledge Graph (KG) Relational Reasoning에서 SAFER(Subgraph Adaptation for Few-shot Relational Reasoning)라는 새로운 접근법이 제안되었습니다. 이는 기존 방법들이 KG에서 충분한 정보를 추출하지 못하거나 불필요한 정보의 영향을 크게 받는 문제점을 해결하고자 합니다. SAFER는 지원(triplet)과 질의(triplet)로부터 생성된 다양한 서브그래프를 활용하여 예측 성능을 향상시킵니다.

- **Technical Details**: SAFER는 지원(triplet)의 포괄적인 정보를 추출하고, 질의(triplet) 예측 시 불필요한 정보의 영향을 줄이는 두 가지 중요한 모듈, 서포트 적응(Support Adaptation)과 질의 적응(Query Adaptation)을 갖추고 있습니다. 서포트 적응은 지원 그래프로부터 유용한 정보를 보다 포괄적으로 추출하며, 질의 적응은 지원 그래프의 구조를 활용해 질의 그래프의 구조에 맞게 적용합니다. 이 과정에서 엣지 마스크(Edge Mask) 기반 접근법의 한계를 극복할 수 있습니다.

- **Performance Highlights**: 세 가지 널리 사용되는 데이터셋을 대상으로 한 실험 결과, SAFER는 기존 최첨단 방법들보다 우수한 성능을 보여줍니다. SAFER는 다양한 지원 그래프 구조로부터 유용한 정보를 추출하고 쿼리-어댑티브 방식으로 불필요한 정보를 걸러내는 등의 알고리즘 개선을 통해 성능을 향상시킵니다.



### Dr.E Bridges Graphs with Large Language Models through Words (https://arxiv.org/abs/2406.15504)
- **What's New**: 이번 연구는 강력한 대형 언어 모델(LLMs)과 다양한 데이터 모달리티(vision, language, audio)의 융합에 주력해 온 기존 노력들과는 달리, 그래프 구조화 데이터와 LLMs의 통합을 목표로 혁신적인 모달리티 정렬 프레임워크를 제안합니다. 이 연구는 LLMs와 그래프 신경망(GNN) 사이의 토큰 수준 정렬을 최초로 달성했습니다.

- **Technical Details**: 제안된 프레임워크는 사전 학습된 Dual-Residual Vector Quantized-Variational AutoEncoder (Dr.E)를 활용하여 LLMs와 그래프 데이터를 효과적으로 정렬합니다. Dr.E는 그래프 데이터를 LLM 호환 토큰으로 매핑하고, 이러한 토큰을 원래 그래프 구조로 다시 디코딩하도록 설계되었습니다. 이는 intra-layer와 inter-layer residuals를 구현하여 다층 관점을 유지하면서 각 컨볼루션 단계에서 정보를 전달할 수 있도록 합니다.

- **Performance Highlights**: 제안된 방법은 고전적인 GNN 노드 분류 작업에서 뛰어난 성능을 보여주었으며, fine-tuning 및 few-shot 설정에서도 대부분의 GNN 기반 및 LLM 기반 방법을 능가했습니다. 따라서 Dr.E는 그래프 데이터 모델링 작업에서 높은 일반화 성능을 입증했습니다.



### Causal Discovery Inspired Unsupervised Domain Adaptation for Emotion-Cause Pair Extraction (https://arxiv.org/abs/2406.15490)
Comments:
          12 pages, 6 figures, 4 tables; Under Review in EMNLP 2024

- **What's New**: 이번 논문에서는 감정-원인 쌍 추출(task of emotion-cause pair extraction) 과제를 비지도 도메인 적응(unsupervised domain adaptation) 환경에서 다룹니다. 기존의 연구들이 거의 없는 이 분야에서, 저자들은 감정 표현(emotional expressions)의 지식을 도메인 간의 분포 차이를 연결하는 다리로 활용하여 심층 잠재 모델(deep latent model)을 제안합니다. 이 모델은 변동 오토인코더 프레임워크(VAE framework)를 사용해 데이터의 잠재 구조를 포착하고 감정의 지식을 쉽게 전이할 수 있게 합니다.

- **Technical Details**: 저자들은 감정과 사건의 잠재 표현(latent representations)을 분리하기 위해 변동 후방 규제 기법(variational posterior regularization)을 도입합니다. 이 기법은 독립성을 가정하지 않고 후방 간의 다이버전스(maximizing the divergences)를 최대화하여 분리를 실현합니다. 이 모델은 CaRel-VAE드라 부르며, 소스 도메인에선 감정과 도메인 특화 사건 간의 인과 관계를 식별합니다. 또한, 자가 학습 알고리즘(self-training algorithm)을 개선해 CD-SelfTrain이라 명명하고, 타겟 도메인 내에서 도메인 특화 인과 관계를 발견합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 모델은 중국어 벤치마크에서 약 11.05%, 영어 벤치마크에서 약 2.45%의 가중치 평균 F1 점수(weighted-average F1 score)를 기준으로 가장 강력한 기준선을 능가했습니다. 소스 코드는 논문이 승인되면 공개될 예정입니다.



### Improving Text-To-Audio Models with Synthetic Captions (https://arxiv.org/abs/2406.15487)
- **What's New**: 본 논문에서는 오디오 기반 멀티모달 모델(즉, text-to-audio 모델)을 위한 고품질의 트레이닝 데이터를 얻기 위한 새로운 접근법을 제안합니다. 특히 AudioSet 데이터를 위해 	exttt{AF-AudioSet}라는 이름의 합성 캡션 데이터셋을 생성하고, 이를 통해 텍스트-투-오디오 (text-to-audio) 모델의 사전 학습을 개선합니다. 본 연구는 오디오 언어 모델(audio language model)을 사용하여 대규모로 정확하고 다양한 캡션을 합성하는 오디오 캡셔닝 파이프라인(audio captioning pipeline)을 소개합니다.

- **Technical Details**: 기존의 방법들은 text-only 언어 모델을 활용하여 캡션을 개선했으나, 오디오와 캡션 간의 일관성(coherence) 문제와 스케일의 한계가 있었습니다. 이에 본 연구에서는 텍스트 전용 언어 모델 대신 오디오 언어 모델을 사용하여 오디오 데이터에 대한 캡션을 합성합니다. 합성된 데이터셋인 	exttt{AF-AudioSet}는 AudioSet에 기반하여 만들어졌으며, 오디오 캡셔닝에 있어서 더욱 정확하고 다양하게 적용될 수 있습니다.

- **Performance Highlights**: 새로운 오디오 캡셔닝 파이프라인과 합성 캡션을 사용함으로써 AudioCaps와 MusicCaps에 대한 체계적인 평가를 통해 오디오 생성 품질(audio generation quality)이 크게 향상되었으며, 이는 새로운 state-of-the-art 달성을 의미합니다.



### Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention (https://arxiv.org/abs/2406.15486)
- **What's New**: 대형 언어 모델(LLM)들의 길어진 문맥 처리 능력은 놀랍지만, 기존의 주의 메커니즘(attention mechanism)이 가지는 이차 복잡도 때문에 첫번째 토큰을 출력하는 데 걸리는 시간이 길어지는 문제가 있다. 새로운 연구에서는 추가적인 사전 학습이나 미세 조정(finetuning) 없이도 이 문제를 해결할 수 있는 방법, SampleAttention을 제안하였다. 이 방법은 프리트레인된 LLM에 거의 정확도 손실 없이 적용될 수 있다.

- **Technical Details**: SampleAttention은 동적으로 헤드별로 특정한 희소 패턴(sparse patterns)을 캡처하여 실행 시점에 낮은 오버헤드로 성능을 유지한다. 구조적으로 고정된 인접 토큰에 대해 집중하여 로컬 윈도우 패턴을, 쿼리-유도 키-밸류 필터링 기법을 통해 컬럼 스트라이프 패턴을 캡처한다. SampleAttention은 기존의 attention 기능을 대체할 수 있으며, 정확도 손실 없이 효율성을 증가시키는 것이 가능하다. 이 방식은 특히 ChatGLM2와 InternLM2와 같은 모델들에서 효과적으로 작동한다.

- **Performance Highlights**: SampleAttention은 FlashAttention과 비교하여 TTFT(Time-to-First-Token, 첫번째 토큰을 출력하는 데 걸리는 시간)를 최대 2.42배 줄일 수 있다. 이는 기존의 모델들과 비교했을 때 거의 정확도 손실 없이 더욱 빠른 처리 속도를 제공한다.



### SegHist: A General Segmentation-based Framework for Chinese Historical Document Text Line Detection (https://arxiv.org/abs/2406.15485)
Comments:
          Accepted by ICDAR2024

- **What's New**: 이번 논문에서는 역사 문서 분석에서 중요한 텍스트 라인 검출 문제를 해결하기 위해 SegHist라는 프레임워크를 제안합니다. 이 프레임워크는 고비율의 텍스트 라인을 포함한 복잡한 문서에서도 효과적인 텍스트 검출을 가능하게 합니다.

- **Technical Details**: SegHist 프레임워크는 TKS (Text Kernel Stretching), LEM (Layout Enhanced Module), IEDP (Iterative Expansion Distance Post-processor) 등의 모듈로 구성되어 있습니다. 특히 DB++와 통합한 DB-SegHist는 ResNet-50과 FPN (Feature Pyramid Network)을 통해 피쳐를 추출하고, 이를 통해 예측된 텍스트 커널 맵을 효율적으로 복구합니다.

- **Performance Highlights**: DB-SegHist는 CHDAC, MTHv2, HDRC 데이터셋에서 SOTA (State-Of-The-Art) 성능을 달성했으며, 특히 CHDAC 데이터셋에서 1.19%의 성능 향상을 보여줍니다. 또한, 회전된 MTHv2와 HDRC 데이터셋에서도 우수한 강인성을 입증했습니다.



### JobFair: A Framework for Benchmarking Gender Hiring Bias in Large Language Models (https://arxiv.org/abs/2406.15484)
Comments:
          Submitted to EMNLP 2024

- **What's New**: 최근 논문에서는 대형 언어 모델(LLMs)을 이용한 이력서 평가의 계층적 성별 편향을 벤치마킹하는 새로운 프레임워크가 발표되었습니다. 이 프레임워크는 역편향(reversed bias) 및 과소편향(overdebiasing)의 문제를 밝혔습니다.

- **Technical Details**: 이 논문에서 제안한 프레임워크는 실제로 익명화된 이력서 데이터를 사용하여 성별 고용 편향을 계층 수준에서 평가합니다. 새로운 통계 및 계산 편향 지표를 도입하였으며, 이에는 Rank After Scoring (RAS), Rank-based Impact Ratio, Permutation Test-Based Metrics, Fixed Effects Model-based Metrics 등이 포함됩니다. 이 프레임워크는 노동 경제학, 자연어 처리(NLP), 법률 등의 분야에 기반을 두고 있으며, 고용 편향을 전체적으로 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 최첨단 LLM 10개를 분석한 결과, 헬스케어 및 금융 산업에서 남성에 대한 유의미한 편향을 보이는 모델이 여섯 개였습니다. 특히 GPT-4o와 GPT-3.5 모델이 세 가지 산업 모두에서 가장 큰 편향을 보였습니다. 반면, Gemini-1.5-Pro, Llama3-8b-Instruct, Llama3-70b-Instruct는 가장 편향이 적었습니다. 이론적 패널 회귀 분석에서 헬스케어 산업이 남성에 대한 편향이 가장 큼을 밝혔습니다. 모든 모델 중 Llama3-8b-Instruct와 Claude-3-Sonnet를 제외하고는 이력서 내용 변경시 편향 정도가 일관된 것으로 나타났습니다. 또한, 사용자가 쉽게 프레임워크를 채택하고 사용할 수 있도록 데모를 제공하고 있습니다.



### Duplicate Detection with GenAI (https://arxiv.org/abs/2406.15483)
Comments:
          12 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models)과 생성 AI(Generative AI)를 사용하여 중복된 기록을 식별하고 수정하는 방법을 제안하고 있습니다. 기존의 자연어 처리(NLP) 기술을 통해 이루어지는 전통적인 엔티티 매칭(Entity Matching) 기법을 사용한 중복 데이터 검출 방법과 비교하여, 제안된 방법은 데이터 비중복률의 정확도를 30%에서 거의 60%로 개선합니다.

- **Technical Details**: 기존의 중복 검출은 다단계 파이프라인(파이프라인 기반)을 사용하여 이루어집니다. 해당 파이프라인은 후보 생성(Candidate Generation), 블록킹(Blocking), 매칭(Matching), 클러스터링(Clustering) 등의 단계를 포함합니다. 반면, 제안된 방법에서는 '후보 생성' 단계가 필요하지 않으며, 매치 문장(Match Sentences)을 생성한 뒤 임베딩 벡터(Embedding Vectors)로 변환하여 DBSCAN 클러스터링 알고리즘을 사용합니다.

- **Performance Highlights**: 제안된 방법은 향상된 임베딩 모델인 all-mpnet-base-v2를 사용하여 매치 문장을 벡터 공간으로 인코딩합니다. 그 결과, 데이터 중복 제거의 정확도가 기존 NLP 기술을 사용한 방법(NBA1)의 30%에서 제안된 방법을 통해 거의 60%로 크게 증가하였습니다. 또한, cosine similarity 거리 측정을 사용한 클러스터링에서 가장 좋은 매칭 결과를 얻었습니다.



### On Giant's Shoulders: Effortless Weak to Strong by Dynamic Logits Fusion (https://arxiv.org/abs/2406.15480)
Comments:
          submit under review

- **What's New**: 최근 대형 언어 모델(LLMs)의 효율적인 미세 조정을 위한 새로운 방식이 제안되었습니다. 본 논문에서는 작은 특화 모델들이 습득한 지식을 큰 모델로 직접 이전할 수 있는 'logit arithmetic' 방식을 탐구합니다. 이는 기존의 정적 지식 이전 비율과 단일 작은 모델을 사용하는 접근법의 한계를 극복하기 위한 방법론을 제시합니다.

- **Technical Details**: 본 연구는 작업별로 특화된 여러 작은 모델들을 사용하는 Dynamic Logit Fusion 접근법을 사용합니다. 각 디코딩 단계에서 KL 발산(Kullback-Leibler divergence)을 사용하여 작업별로 적절한 가중치를 동적으로 할당합니다. 이를 통해 다양한 작업에서 작은 모델들이 습득한 지식을 큰 모델로 효과적으로 이전하는 방법을 제공합니다.

- **Performance Highlights**: 다양한 벤치마크 테스트 결과, 7B 모델에서 13B 모델로 지식을 이전함으로써 싱글 태스크 시나리오에서는 성능 격차를 96.4% 줄였고, 멀티 태스크 시나리오에서는 86.3% 줄이는 성과를 거두었습니다. 특히, 보지 못한 작업에서도 우수한 성능을 보였으며, 인컨텍스트 학습 및 태스크 산수(Task Arithmetic)와의 통합 가능성도 입증되었습니다.



### Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging (https://arxiv.org/abs/2406.15479)
Comments:
          submit in review

- **What's New**: 최근 대형 언어 모델(LLM)의 시대에, 모델 병합(Model Merging)은 추가 훈련 없이 다중 작업 모델을 결합하는 유망한 방법으로 부상했습니다. 그러나 기존 방법들은 성능 저하와 불균일한 데이터 처리에서 문제를 겪고 있습니다. 이에 대한 해결책으로 Twin-Merging이 제안되었습니다. Twin-Merging은 공유된 지식과 개별 작업에 특화된 지식을 압축하고, 입력 데이터에 따라 이를 동적으로 병합하여 성능을 향상시킵니다.

- **Technical Details**: Twin-Merging은 두 단계로 구성됩니다. 첫째, Knowledge Modularization(지식 모듈화) 단계에서 공유 지식과 개별 작업의 특화된 지식을 분리하고 압축합니다. 둘째, Dynamic Merging(동적 병합) 단계에서는 Mixture of Experts(MoE) 기법을 활용하여 공유 지식을 기초로 하고, 입력 데이터에 따라 개별 지식을 동적으로 결합합니다.

- **Performance Highlights**: Twin-Merging 방법은 12개의 데이터셋에서 실험을 통해 그 효과가 입증되었습니다. 이 방법은 판별 작업에서 28.34%의 성능 향상과 생성 작업에서 3.86%의 성능 향상을 보여주었으며, 특히 생성 작업에서는 기존의 최적 성능을 넘어섰습니다. 또한, 파라미터 수를 99.9% 줄여도 성능 저하는 14%에 불과했습니다.



### CrisisSense-LLM: Instruction Fine-Tuned Large Language Model for Multi-label Social Media Text Classification in Disaster Informatics (https://arxiv.org/abs/2406.15477)
- **What's New**: 이번 연구에서는 재난 관련 트윗의 multi-label 분류를 위한 pre-trained Large Language Model(LLM)을 instruction fine-tuning하여 다중 측면의 정보를 동시에 분류할 수 있는 새로운 접근 방식을 소개합니다. 이는 단일 라벨 텍스트 분류 모델의 한계를 극복하며, 재난 상황에서 소셜 미디어 데이터를 더 효과적으로 활용할 수 있게 합니다.

- **Technical Details**: 본 연구에서는 재난 관련 트윗으로부터 종합적인 instruction 데이터셋을 생성하여 오픈 소스 LLM을 fine-tuning하였습니다. 이 과정에서 재난 특정 지식을 모델에 내재화하였으며, 이벤트 유형, 정보성, 인도적 지원 여부 등 다양한 정보를 동시에 분류할 수 있습니다.

- **Performance Highlights**: 새로운 모델은 소셜 미디어 게시물에서 중요한 정보를 더 효과적으로 분류할 수 있도록 함으로써 재난 상황 인식과 긴급 대응 전략의 실시간 개선에 기여합니다. 이는 재난 관리 도구의 발전과 적응성을 높이는 연구의 기초를 마련합니다.



### Self-Regulated Data-Free Knowledge Amalgamation for Text Classification (https://arxiv.org/abs/2406.15476)
Comments:
          12 pages, 5 Figures, Proceedings of NAACL 2024

- **What's New**: 최근에 사전 학습된 텍스트 모델들이 많이 공개되면서 이를 활용하면 새 모델을 처음부터 훈련하는 비용을 크게 줄일 수 있게 되었습니다. 그러나 프라이버시, 보안, 지적 재산권 문제로 인해 이러한 데이터셋들이 공개되지 않는 경우도 많습니다. 이 논문에서는 원본 데이터를 사용하지 않고 여러 교사 모델로부터 학습할 수 있는 가벼운 학생 네트워크를 개발하는 것을 목표로 합니다. 이를 위해 데이터 없이도 지식을 전이하는 DFKA(데이터 프리 지식 병합)라는 과제를 탐구합니다. STRATANET이라는 새로운 프레임워크를 제안하며, 이는 각각의 교사 모델에 맞춤화된 텍스트 데이터를 생성하는 steerable data generator(스티어러블 데이터 생성기)와 교사들의 여러 계층에서 자가조절 전략으로 지식을 선택적으로 통합하는 모듈로 구성됩니다.

- **Technical Details**: STRATANET은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 각 사전에 학습된 교사 네트워크를 위한 유사 텍스트 데이터를 생성하는 유연한 생성 모듈입니다. 둘째, 학생 모델 훈련 중에 교사들의 지식을 자가조절 접목(self-regulated integration)하는 병합 모듈입니다. 이 병합 모듈은 교사의 중간 및 출력 상태의 신뢰도를 평가하는 교사별 out-of-distribution (OOD) 점수에 따라 작동합니다. 목표는 원본 학습 데이터에 접근하지 않고도 여러 교사 모델의 지식을 통합하여 학생 모델의 성능을 향상시키는 것입니다.

- **Performance Highlights**: 우리의 STRATANET으로 학습한 학생 모델은 세 가지 벤치마크 텍스트 분류 데이터셋(AG 뉴스, OhSumed 초록, 5 Abstracts Group)에서 여러 기준 모델을 상당히 능가하는 성능을 입증했습니다. 이 데이터셋들은 각각의 도메인이나 라벨이 다양하기 때문에, STRATANET의 성능이 다양한 상황에서 우수하다는 것을 확인할 수 있었습니다.



### Intertwining CP and NLP: The Generation of Unreasonably Constrained Sentences (https://arxiv.org/abs/2406.15473)
Comments:
          To appear at The 33rd International Joint Conference on Artificial Intelligence, IJCAI-24 (in press)

- **What's New**: 이번 논문에서는 NLP(자연어 처리)에서 어려운 제약사항을 다루는 텍스트 생성 문제를 해결하기 위한 Constraints First Framework를 소개합니다. 이 프레임워크는 텍스트 생성을 이산 조합 최적화 문제로 간주하며, 언어적 속성과 전통적인 제약사항을 결합한 제약 프로그래밍 방법으로 이를 해결합니다.

- **Technical Details**: 제약된 텍스트 생성 문제를 CSP(제약 만족 문제)로 정의하고, 이를 Multi-valued Decision Diagrams (MDDs) 자료구조를 사용해 최적화 문제로 변환합니다. 이를 통해 n-그램과 같은 언어적 속성과 문자 수, 음절 수, 단어 수와 같은 제약사항을 결합하여 해결합니다. 또한, 자동적으로 결과를 큐레이션하여 대형 언어 모델의 혼란도(perplexity)를 기준으로 최적의 문장을 선택합니다.

- **Performance Highlights**: 이 접근법의 효과는 RADNER 문장 생성을 통해 입증되었습니다. 본 제약 프로그래밍 기반 접근법을 통해 엄격한 제약사항을 충족하는 새로운 문장을 성공적으로 생성했습니다. 이는 비현실적으로 어려운 제약사항을 다루는 텍스트 생성 시나리오에서도 이 접근법의 가능성을 보여줍니다.



### Hyperbolic sentence representations for solving Textual Entailmen (https://arxiv.org/abs/2406.15472)
- **What's New**: 이번 연구에서는 계층적 데이터(hierarchical data)를 모델링하기에 적합한 쌍곡 공간(hyperbolic spaces)을 이용하여 문장을 Poincare ball에 임베딩(embedding)하는 방법을 제안합니다. 이를 통해 Textual Entailment 문제를 해결할 수 있는 가능성을 입증합니다.

- **Technical Details**: 연구에서 사용된 주요 작업은 두 가지 추가 데이터셋을 개발하고, 기존의 다양한 배경을 가진 모델들과 비교 평가하는 것입니다. 비교 대상에는 LSTM, Order Embeddings, Euclidean Averaging 등이 포함되며, Euclidean 공간에 문장을 표현하는 것과 자연스러운 대조를 이룹니다.

- **Performance Highlights**: 제안된 방법은 SICK 데이터셋에서 일관되게 높은 성능을 보였으며, SNLI 데이터셋의 이분류 버전(binary classification version)에서는 Order Embeddings에 이어 두 번째로 높은 성과를 기록했습니다.



### Improving Large Models with Small models: Lower Costs and Better Performanc (https://arxiv.org/abs/2406.15471)
Comments:
          11 pages

- **What's New**: 프리트레인된 대규모 모델(Pretrained Large Models, PLMs)인 ChatGPT와 같은 모델들이 다양한 작업에서 탁월한 성능을 보여주고 있습니다. 그러나 PLM의 높은 계산 요구사항 때문에 대부분의 제품 팀이 이러한 모델을 운영하거나 미세 조정(fine-tuning)하는 것을 꺼려합니다. 이를 해결하기 위해, 소형 모델과 대형 모델이 협력하는 새로운 패러다임인 Data Shunt$^+$ (DS$^+$)가 제안되었습니다. DS$^+$는 대형 모델을 쿼리하는 데 따른 비용을 크게 줄이는 동시에 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: DS$^+$는 특정 작업을 여러 하위 작업으로 나누고, 간단한 하위 작업은 소형 모델에 맡기고 복잡한 하위 작업만 대형 모델이 처리하도록 합니다. 이를 통해 전체적인 처리 비용을 줄이는 동시에 성능을 향상시킬 수 있습니다. 예를 들어, Amazon 제품의 감성 분석에서 ChatGPT는 94.43%의 정확도를 달성했으며, DS$^+$는 95.64%의 정확도를 달성하면서 비용을 31.18%로 줄였습니다.

- **Performance Highlights**: DS$^+$의 협력 기반 패러다임은 단순히 비용 절감뿐만 아니라 특정 작업 지식을 대형 모델에 더 효과적으로 주입할 수 있음을 실험을 통해 증명했습니다. 이로 인해 기존의 미세 조정보다 더 나은 성능을 발휘할 수 있습니다.



### Mental Disorder Classification via Temporal Representation of Tex (https://arxiv.org/abs/2406.15470)
Comments:
          RK and KM contributed equally to this work, 15 pages, 5 figures, 9 table

- **What's New**: 이번 연구는 소셜 미디어 게시물에서 정신 질환을 예측하는 새로운 프레임워크를 제안했습니다. 기존 방법들과 달리, 이 프레임워크는 연대기 순으로 정렬된 소셜 미디어 게시물을 숫자 시퀀스로 압축하고, 이를 이용해 정신 질환을 분류합니다. 이 방법은 우울증, 자해, 신경성 식욕부진증 세 가지 정신 질환에서 현재 SOTA(State-of-the-Art)를 약 5%의 F1 점수 향상으로 능가했습니다.

- **Technical Details**: 제안된 프레임워크는 소셜 미디어 게시물 데이터를 시간의 흐름에 따라 압축된 숫자 시퀀스로 변환합니다. 이를 통해 LLMs(posts from social media)의 문맥 길이 제한과 계산 비용 문제를 극복할 수 있습니다. 기존 방법은 데이터를 청크(chunks)로 나누어 처리하지만, 이는 시간 정보를 잃고 중요한 사이 결합성을 놓치는 단점이 있습니다. 새로운 접근법은 심층 학습 모델(deep learning models)을 활용하여 시간 변이 데이터를 통합합니다.

- **Performance Highlights**: 제안된 방법은 우울증, 자해, 신경성 식욕부진증 등 세 가지 정신 질환에 대해 현재 최고 성능(SOTA)을 약 5%의 F1 점수 향상으로 능가했습니다. 또한, 제안된 프레임워크를 사용하여 클래스 도메인 간 연구를 수행하고, 다른 분야의 데이터를 활용할 가능성을 탐구했습니다.



### Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models (https://arxiv.org/abs/2406.15468)
- **What's New**: MMLU-SR는 대형 언어 모델(LLMs)의 진정한 이해 능력을 측정하기 위해 고안된 새로운 데이터셋입니다. 표준화된 테스트 질문에서 주요 용어를 더미 단어와 정의로 대체하여 모델의 이해도를 평가합니다. 기존 MMLU 리더보드에서 높은 점수를 받은 LLM들이 이러한 대체 후 성능이 크게 감소한다는 결과를 발견했습니다.

- **Technical Details**: MMLU-SR은 세 가지 하위 집합인 ‘Question Only’, ‘Answer Only’, 그리고 ‘Question and Answer’로 구성됩니다. 중요한 용어를 GPT-3.5-turbo의 도움으로 추출하고 각 용어에 적절한 정의를 제공합니다. 생성된 사전을 이용해 질문과 답변의 용어를 무작위 더미 단어와 정의로 대체함으로써 모델이 암기된 용어나 어휘에 의존하지 않고 정의와 개념을 사용하여 추론할 수 있도록 합니다. 데이터 교체, 조합 및 최종 조정을 통해 데이터셋을 구성합니다.

- **Performance Highlights**: gpt-3.5-turbo, llama3-8b, 그리고 gemini-1.0-pro에 대한 평가 결과, 원래의 MMLU 데이터셋에 비해 MMLU-SR에서 성능이 크게 낮아졌습니다. 이는 모델이 암기된 데이터 대신 정의와 개념을 바탕으로 추론하는 데 어려움을 겪고 있음을 나타냅니다.



### RadEx: A Framework for Structured Information Extraction from Radiology Reports based on Large Language Models (https://arxiv.org/abs/2406.15465)
- **What's New**: 이 논문은 RadEx라는 프레임워크를 소개하며, 이는 방사선 보고서에서 자동으로 정보를 추출할 수 있는 시스템 개발을 위한 15개의 소프트웨어 구성 요소와 10개의 인공물을 포함합니다. 이 프레임워크는 모델 개발의 경계를 설정하고 일관된 일반 정보 모델을 제공하여 전체 프로세스를 포괄합니다.

- **Technical Details**: RadEx 프레임워크는 다양한 임상 도메인(예: 유방촬영술)에 대해 관련 정보를 정의하고 보고서 템플릿을 만들 수 있도록 지원합니다. 프레임워크는 생성 모델과 엔코더 전용 모델 둘 다를 지원하며, 정보 추출과 템플릿 채우기를 분리하여 독립적인 모델 개선을 가능하게 합니다. UIMA(Unstructured Information Management Architecture)를 사용해 정보 모델을 공식적으로 정의하며, 이 모델은 사실(facts), 앵커 엔티티(anchor entities), 수정자(modifiers)로 구성됩니다.

- **Performance Highlights**: RadEx 프레임워크를 사용하면 구성 요소가 쉽게 교체될 수 있어 구현 및 유지 관리가 용이하며, 표준화된 인공물 덕분에 구성 요소 간의 상호 운용성이 보장됩니다. 또한, 주어진 사용 사례에 대한 사실 스키마(fact schema)를 정의하고 이를 기반으로 보고서 템플릿을 채워 자동 추출된 정보를 표준화할 수 있습니다.



### Investigating the Robustness of LLMs on Math Word Problems (https://arxiv.org/abs/2406.15444)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 수학 문제 풀이(Math Word Problems, MWPs)에서 무관한 정보가 포함된 실제 문제들을 해결할 때 겪는 어려움을 해결하고자 새로운 프롬프트 프레임워크를 제안합니다. 이 프레임워크는 무관한 변수를 추가하여 해악성(adversarial) 변형 MWP를 생성하며, ProbleMATHIC이라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 해악성 및 비해악성 MWP를 모두 포함하고 있습니다.

- **Technical Details**: 우리는 '제약적 추가 접근법(Constrained Additive Approach)'을 제안하여 원래의 MWP에 무관한 수치 정보를 추가했습니다. 추가된 변수는 기존 변수와 관련이 없고, 같은 물리적 단위를 공유하지 않으며, 기존 변수에 대한 새로운 정보를 제공해서는 안 됩니다. 이를 통해 모델이 관련 정보를 식별하고 논리적으로 연관짓지 않고도 올바른 솔루션을 도출할 수 있는 능력을 평가할 수 있습니다. ProbleMATHIC 데이터셋은 단순 및 복잡한 문제 세트로 나뉘며, 각각 훈련 및 시험 샘플을 포함합니다.

- **Performance Highlights**: LLMs(Llama-2, Mistral) Fine-tuning 실험에서 해악성 샘플을 사용한 훈련이 해악성 MWP에 대한 성능을 약 8% 향상시켰습니다. 이는 모델이 노이즈에 대한 강건성을 높이고 관련 변수를 식별하는 능력을 향상시켰음을 나타냅니다. 또한 GSM-8K-Adv 벤치마크를 도입하여 프롬프트 프레임워크의 일반화 가능성을 평가한 결과, 여전히 해악성 정보에 직면했을 때 LLMs의 성능이 약 6% 감소함을 확인했습니다.



### ExU: AI Models for Examining Multilingual Disinformation Narratives and Understanding their Spread (https://arxiv.org/abs/2406.15443)
Comments:
          Accepted at The 25th Annual Conference of The European Association for Machine Translation (EAMT 24)

- **What's New**: ExU 프로젝트는 다국어로 확산되는 온라인 허위 정보를 분석하기 위한 AI 모델을 개발하는 데 중점을 두고 있습니다. 주요 작업으로는 소문 태도 분류(rumour stance classification)와 주장 검색(claim retrieval)이 있습니다. 이 프로젝트는 2023년 11월에 시작되어 18개월 간 진행될 예정입니다.

- **Technical Details**: ExU 프로젝트는 20개 이상의 언어를 대상으로, 특히 포르투갈어, 스페인어, 폴란드어, 슬로바키아어, 체코어, 힌디어 및 프랑스어를 평가 프레임워크에 포함할 것입니다. 사용자 요구 사항 조사 결과, 참가자들은 태도 분류 모델이 게시물의 주장에 대한 입장을 자동으로 예측하고 주요 논점을 하이라이트하는 기능을 중요하게 생각했습니다. 이 프로젝트는 다국어 데이터를 다루기 위해 다국어 트랜스포머 모델(Aya 모델 등)을 사용할 계획입니다.

- **Performance Highlights**: 이 프로젝트에서 사용되는 MultiClaim 데이터셋은 39개 언어로 된 293,169개의 사실 확인된 기사와 관련된 주장들을 포함하고 있습니다. 모델은 텍스트 주장을 분석하여 관련된 사실 확인된 주장의 목록, 언어 및 출처를 제공하고, 자연어로 요약된 중심 주장을 출력할 예정입니다.

- **Future Work**: 향후 작업으로는 저자원 언어(low-resource languages)에 대한 데이터 확보 및 번역을 통해 모델 성능을 개선할 계획입니다. 또한 다국어 환경에서 설명 가능성(explainability)을 보장하기 위해 기능 귀속(feature attribution) 및 추론 추출(rationale extraction)을 탐구할 예정입니다. 이와 함께 다국어 주장 검색 모델을 사용하여 사용자들이 사실과 허구를 효율적으로 구분할 수 있도록 지원할 계획입니다.



### Traitement quantique des langues : {\'e}tat de l'ar (https://arxiv.org/abs/2406.15370)
Comments:
          in French language

- **What's New**: 이 기사는 자연어 처리 (NLP) 분야에서의 양자 컴퓨팅 연구를 검토합니다. 이러한 연구는 현재 모델의 성능을 개선하고, 모호성 및 장거리 종속성과 같은 여러 언어적 현상을 더 잘 표현하는 것을 목표로 합니다.

- **Technical Details**: 여러 접근 방식을 소개하며, 상징적 도해적 접근 (symbolic diagrammatic approaches)과 하이브리드 신경망 (hybrid neural networks)을 포함합니다. 이러한 연구들은 실험적 연구가 이미 가능함을 보여주고 있으며, 새로운 모델의 설계 및 평가에 대한 연구 전망을 제시합니다.

- **Performance Highlights**: 이 연구들은 현재 모델의 성능을 개선하고 다양한 언어적 현상을 더 잘 표현할 수 있는 가능성을 제시합니다. 특히 모호성 (ambiguity) 및 장거리 종속성 (long range dependencies) 같은 언어적 현상에 대한 더 나은 표현력을 목표로 합니다.



### Exploring LLM Multi-Agents for ICD Coding (https://arxiv.org/abs/2406.15363)
- **What's New**: 이 논문에서는 LLMs(Large Language Models)가 임상 텍스트로부터 도메인 특화 훈련 없이도 정보 추출을 수행하는 능력을 설명하고 있습니다. 하지만, ICD 코드 작업에서는 중요한 세부 사항을 놓치고 높은 recall을 보이지만 precision이 낮다는 문제점이 있었습니다. 이를 해결하기 위해 다중 에이전트 기반의 새로운 방법을 제안하였습니다. 이 방법은 실제 코딩 프로세스를 모방하여 환자 에이전트, 의사 에이전트, 코더 에이전트, 리뷰어 에이전트, 조정자 에이전트의 다섯 가지 역할을 가지고 있습니다.

- **Technical Details**: 각 에이전트는 특정 기능을 수행하고 이를 위해 LLM 기반 모델을 사용합니다. 제안된 방법은 MIMIC-III 데이터셋을 사용하여 평가되었으며, 제안된 다중 에이전트 코딩 프레임워크가 Zero-shot Chain of Thought(CoT) 유도로와 CoT의 self-consistency와 비교하여 성능이 크게 향상된다는 것을 보여주었습니다. 또한, 에이전트 역할의 효능을 확인하는 ablation study도 수행되었습니다.

- **Performance Highlights**: 제안된 방법은 사전 훈련 또는 fine-tuning 없이 흔한 코드와 희귀 코드 모두에서 높은 정확도를 보였으며, 설명 가능성 측면에서도 최첨단의 ICD 코딩 방법과 일치하는 성과를 거두었습니다.



### Diverse Perspectives, Divergent Models: Cross-Cultural Evaluation of Depression Detection on Twitter (https://arxiv.org/abs/2406.15362)
Comments:
          6 pages, 2 figures, NAACL 2024 Main Conference

- **What's New**: 이 논문에서는 정신 질환, 특히 우울증이 있는 사용자들을 탐지하기 위해 소셜 미디어 데이터를 활용하는 최근 연구 동향을 살펴봅니다. 기존 모델들이 전 세계적으로 적용될 수 있는지 평가하기 위해 7개국의 지리적으로 위치한 우울증 사용자의 Twitter 데이터를 수집하였습니다. 그 결과, 우울증 탐지 모델이 전 세계적으로 일반화되지 못한다는 결론에 도달했습니다.

- **Technical Details**: 연구진은 Global South와 Global North 사용자들을 대상으로 모델의 성능을 비교했습니다. 사전 훈련된 언어 모델(Pre-trained language models)이 로지스틱 회귀(Logistic Regression) 모델보다 더 나은 일반화 성능을 보였지만, 여전히 비서구권 사용자와 우울증 사용자들에 대한 성능에서는 큰 차이를 보였습니다.

- **Performance Highlights**: Pre-trained language models이 가장 높은 일반화를 보였으나, Global South 사용자들에 대한 모델의 성능은 여전히 Global North 사용자들에 비해 낮았습니다. 이러한 성능 격차를 줄이기 위한 몇 가지 실행 가능한 제안을 함께 제공합니다.



### Constructing Multilingual Visual-Text Datasets Revealing Visual Multilingual Ability of Vision Language Models (https://arxiv.org/abs/2406.15359)
- **What's New**: 대형 언어 모델(LLMs)로 인해 이미지-텍스트 쌍을 처리하는 비전 언어 모델(VLMs)에 대한 관심이 증가하고 있습니다. 이전 연구들은 VLM의 시각적 이해 능력을 조사했지만, 기존 데이터셋은 여러 언어에 걸쳐 VLM의 세밀한 시각 언어 능력을 포괄적으로 평가하기에 한계가 있었습니다. 이를 위해 본 연구에서는 새로운 데이터셋을 개발하였고 이를 통해 VLM을 체계적으로 분석했습니다.

- **Technical Details**: 1) 목표 목표를 위해 우리는 개체 인식(Object Recognition), 이미지-텍스트 매칭(Image-Text Matching) 등의 9가지 VL 작업을 도입하였으며, 이를 위해 영어, 일본어, 스와힐리어, 우르두어로 된 다국어 시각-텍스트 데이터셋을 구축했습니다. 데이터셋은 GPT-4V를 사용하여 질문(questions), 답변(answers), 그리고 논리적 설명(rationales)을 생성하였습니다. 2) 새로운 VL 작업 'Unrelatedness'를 도입했습니다. 3) VLM의 추론 과정을 인간이 이해할 수 있도록 논리적 설명을 추가했습니다. 4) 제안된 데이터셋이 VL 작업에 적합한지 평가하기 위해 인간 평가(Human Evaluation)를 사용했습니다.

- **Performance Highlights**: 우리의 데이터셋으로 VLM을 미세 조정(Fine-tuning)할 수 있음을 보였으며, 이는 스와힐리어와 우르두어에서 처음으로 수행된 분석입니다. 또한, 본 연구는 VL 분석에 논리적 설명을 도입하여 평가에서 중요한 역할을 했습니다.



### Introducing Syllable Tokenization for Low-resource Languages: A Case Study with Swah (https://arxiv.org/abs/2406.15358)
- **What's New**: 새로운 논문에서 다국어 자연어 처리(NLP) 분야에서 사전 학습된 언어 모델(PLMs)이 저자원 언어에도 효과적으로 적용될 수 있도록 하는 방안을 연구했습니다. 특히, Swahili 언어에 적합한 음절 기반 토크나이저(syllable tokenizer)를 제안했습니다.

- **Technical Details**: 논문에서는 다양한 언어의 언어적 특성을 포착하는 단어 임베딩(word embeddings)을 만들기 위한 방법으로 토크나이제이션(tokenization)의 중요성을 강조합니다. 현재 대부분의 사전 학습된 언어 모델은 BPE, 워드피스(wordpiece), 유니그램(unigram)과 같은 일반적인 토크나이제이션 방법을 사용하고 있지만, 특정 언어에는 적합하지 않을 수 있습니다. 따라서, 해당 연구에서는 입력 텍스트의 음절을 기반으로 한 토크나이제이션 방식을 제안했습니다. 음절 토크나이저는 음절이 많은 언어에 특히 유효할 수 있으며, 예시로 Swahili 언어를 통해 이를 검증했습니다.

- **Performance Highlights**: GPT2를 사용한 텍스트 생성 실험을 통해 음절 토크나이저의 효과를 평가했으며, 제안된 음절 토크나이저가 Swahili 언어의 구조를 효과적으로 나타내는 음절 임베딩을 생성함을 확인했습니다. 이를 통해 음절 인지 언어 모델(syllable-aware language models)을 개발할 수 있는 가능성을 보여주었습니다.



### Ragnar\"ok: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track (https://arxiv.org/abs/2406.16828)
- **What's New**: 최근 몇 달 동안, 검색 기술이 Retrieval-Augmented Generation (RAG) 시스템을 통해 크게 발전하였습니다. 이 기술은 실시간 데이터를 대형 언어 모델(LLM)과 결합하여 정보를 제공합니다. 이러한 RAG 시스템을 테스트하고 평가하기 위해, TREC 2024 RAG 트랙을 제안했습니다.

- **Technical Details**: 우리는 'Ragnarök' 이라는 재사용 가능한 프레임워크를 개발했습니다. 이 프레임워크는 Pyserini 및 RankLLM과 같은 기존 파이썬 프레임워크와 깊이 통합되어 있으며, 사용자 친화적인 REST API와 WebUI를 제공합니다. 'Ragnarök'은 두 가지 주요 모듈로 구성됩니다: Retrieval와 Augmented Generation.

- **Performance Highlights**: 우리의 실험 결과 OpenAI의 GPT-4o는 Cohere의 Command R+에 비해 더 상세한 답변을 제공한다고 관찰했습니다. 이 프레임워크와 주요 베이스라인은 공개되어 있으며, TREC 2024 RAG 트랙에서 사용될 예정입니다. 향후, 더 다양한 LLMs 기준을 포함하고 프레임워크를 지속적으로 개선할 계획입니다.



### PISTOL: Dataset Compilation Pipeline for Structural Unlearning of LLMs (https://arxiv.org/abs/2406.16810)
- **What's New**: 최근 연구에서는 사전 학습된 모델 또는 파인 튜닝된 모델에서 특정 데이터를 삭제하려는 목적으로 기계적 잊어버림(machine unlearning)이 중요하게 떠오르고 있습니다. 그러나 지금까지 고려된 LLMs(대형 언어 모델)의 잊어버림 접근법은 독립된 데이터 포인트 제거에만 집중되어 있으며, 저장된 사실들이 논리적으로 연결되어 암묵적인 지식 그래프를 형성하고 있다는 점을 고려하지 않았습니다. 이를 해결하기 위해, 다중 시나리오 데이터셋을 컴파일하여 구조적 LLM 잊어버림(structural LLM unlearning) 방법을 벤치마킹하기 위한 파이프라인인 PISTOL을 제안합니다.

- **Technical Details**: PISTOL을 활용하여 합성된 샘플 데이터셋을 사용해, Llama2-7B와 Mistral-7B 모델에서 네 가지 다른 잊어버림 방법을 벤치마킹했습니다. 이 분석은 상호 연결된 데이터, 일괄 처리된 데이터 또는 특정 도메인에 치우친 데이터를 효과적이고 견고하게 제거하는 데 있어 발생하는 도전 과제를 설명하는 데 도움이 됩니다. 또한 사전 학습된 모델의 선택이 잊어버림 성과에 영향을 미칠 수 있다는 점을 강조합니다.

- **Performance Highlights**: 이 연구는 현재 LLMs 잊어버림 방법의 한계를 새롭게 인식하게 하며, 앞으로의 연구 방향도 제시합니다. 뿐만 아니라, 지속적인 탐색과 검증을 위한 반복 가능한 구조를 제공합니다.



### Beyond Thumbs Up/Down: Untangling Challenges of Fine-Grained Feedback for Text-to-Image Generation (https://arxiv.org/abs/2406.16807)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 보상 모델을 학습하고 개선하기 위해 인간의 피드백이 어떻게 활용될 수 있는지에 대해 탐구했습니다. 특히, 세밀한 피드백(fine-grained feedback)과 전통적인 대략적인 피드백(coarse-grained feedback)의 효과를 비교하며, 세밀한 피드백이 오히려 더 나쁜 모델을 만들 수 있다는 점을 지적합니다.

- **Technical Details**: 핵심적으로 보상 함수(reward function)의 학습에 중점을 두었으며, 세밀한 피드백이 더 나은 성능을 보인다고 가정했습니다. 실험은 실제 그리고 합성된 선호 데이터(preference data)를 사용해 다각도로 수행되었습니다. 모델 선택, 피드백 유형, 그리고 인간 판단과 컴퓨터 해석 간의 정렬이 중요한 요소로 작용함을 발견했습니다. 실험 결과, 통제된 환경에서는 세밀한 피드백이 효과적일 수 있으나, 실환경에서는 상황에 따라 결과가 달라질 수 있음이 드러났습니다.

- **Performance Highlights**: 주요 결과로는 고정된 예산 하에서 세밀한 피드백이 가끔 더 나쁜 모델 성능을 유도한다는 사실을 밝혀냈습니다. 그러나 특정 속성을 완전히 파악하고 있는 통제된 상황에서는 세밀한 피드백이 유용함이 확인되었습니다. 이를 통해 세밀한 피드백의 가치를 극대화하기 위해 피드백의 특성과 과제 특성을 신중하게 고려해야 함을 강조합니다.

- **Empirical Case Studies**: 실험을 통해 세밀한 피드백을 활용한 텍스트-이미지 보상 모델 훈련의 추가적인 가치는 매우 조건부적임이 입증되었습니다. 피드백 속성을 효과적으로 활용하기 위한 새로운 모델링 접근 방식의 필요성을 제기하며, 세밀한 피드백을 기반으로 한 텍스트-이미지 시스템의 보상 모델 구축 및 평가와 관련된 미해결 과제들을 논의합니다.



### OCALM: Object-Centric Assessment with Language Models (https://arxiv.org/abs/2406.16748)
Comments:
          Accepted at the RLBRew Workshop at RLC 2024

- **What's New**: 새로운 연구에서는 강화 학습 (RL) 에이전트를 훈련시키기 위한 보상 신호를 정의하는 문제를 해결하려고 합니다. Object-Centric Assessment with Language Models (OCALM)을 제안하여 자연어 태스크 설명에서 본질적으로 해석 가능한 보상 함수 (reward functions)를 도출합니다. 이는 대규모 언어 모델 (LLMs)의 광범위한 세계 지식을 활용하면서 많은 환경에서 공통적으로 존재하는 객체 중심적 (object-centric) 특성을 활용합니다.

- **Technical Details**: OCALM은 LLMs가 가진 세계 지식을 최대한 활용하고, 환경 내의 개체 중심적 성격을 고려하여 관계적 개념에 집중된 보상 함수를 도출합니다. 이를 통해 RL 에이전트가 자연어로 주어진 태스크 설명으로부터 정책을 도출할 수 있게 합니다. 기존의 블랙박스 보상 모델과는 달리, OCALM은 보상 함수가 본질적으로 해석 가능하도록 설계되어 디버깅의 어려움을 줄입니다.

- **Performance Highlights**: 이 방법은 복잡한 환경에서도 전문가의 도움 없이 비전문가가 에이전트의 목표를 지정할 수 있게 하여, LLMs와 결합된 객체 중심 접근법이 RL 에이전트의 훈련에 유망한 대안이 될 수 있음을 보여줍니다.



### The Responsible Foundation Model Development Cheatsheet: A Review of Tools & Resources (https://arxiv.org/abs/2406.16746)
- **What's New**: 기초 모델(Fondation Model) 개발에 관한 치트시트(Cheatsheet)가 소개되었습니다. 이 치트시트는 텍스트, 비전, 음성 모달리티를 포함하는 250개 이상의 도구와 자원을 담고 있으며, 데이터 선택, 처리 및 이해, 모델 훈련, 환경 영향, 모델 평가 및 배포 등에 관한 가이드를 제공합니다. 이를 통해 보다 책임감 있는 개발을 유도하고자 합니다.

- **Technical Details**: 이 자원 목록은 기존 연구를 기반으로 소프트웨어, 문서, 프레임워크, 가이드 및 실제 도구들을 조사하여 작성되었습니다. 데이터 소싱 및 처리, 모델 훈련 및 평가, 환경적 영향에 대한 인식, 모델의 릴리스 및 라이선스, 배포 관행 등을 지원하는 자원을 포함하고 있습니다. 또한, AI 개발 생태계를 검토하여 중요한 결함과 과용, 남용되는 도구들을 발견했습니다.

- **Performance Highlights**: 치트시트를 통해 발견된 주요 문제점은 다음과 같습니다: (i) 데이터 소싱, 모델 평가 및 모니터링 도구가 윤리적 및 현실적 요구를 충분히 충족시키지 못한다는 점, (ii) 모델 안전성, 기능 및 환경적 영향에 대한 평가가 재현성과 투명성이 부족하다는 점, (iii) 텍스트 중심, 특히 영어 중심의 분석이 다언어 및 다중 모달리티 분석을 지배하고 있다는 점, (iv) 모델뿐만 아니라 시스템 평가가 필요하다는 점입니다.



### ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models (https://arxiv.org/abs/2406.16635)
- **What's New**: 최근 논문에서는 입력에 의존한 '문맥적 스파시티'(contextual sparsity)를 모델링하기 위해 신경망을 통해 활성화 크기를 예측하고 있습니다. 이 논문에서는 기존의 크기 기반 가지치기 기준(magnitude-based pruning criteria)을 넘어, 대형 언어 모델(LLM)의 주의 헤드와 뉴런의 중요도를 평가하는 새로운 예측기 ShadowLLM을 개발했습니다. 이를 통해 기존 방법들에 비해 15% 이상의 정확도 향상을 달성하면서도 지연 시간(latency)을 증가시키지 않았습니다.

- **Technical Details**: ShadowLLM은 모델의 시작 부분에 통합된 예측기를 사용하여 전체 LLM 스파시티 패턴을 모델링합니다. 이는 입력에 따라 동적으로 스파시티 패턴을 변화시키는 방식으로, 더 나은 정확도-성능 트레이드오프를 제공합니다. 이전의 DejaVu 프레임워크는 각 층마다 신경망 예측기를 생성하여 지역 정보를 활용하지만, 레이어별 예측기는 높은 런타임 비용이 발생합니다. ShadowLLM은 입력에 따라 특정 불필요한 구조체를 가지치기하여 성능을 최적화합니다.

- **Performance Highlights**: ShadowLLM은 DejaVu 프레임워크에 비해 최대 20%의 속도 향상을 제공하면서도, 정확도는 15% 이상 향상시켰습니다. 이 방법은 최대 300억 개의 파라미터를 가진 모델에서도 유효성이 검증되었습니다. 다양한 기존 방법들을 검토하여 효과적인 가지치기 기준을 찾았고, 이는 성능을 유지하면서도 정확도를 향상시키는 데 기여했습니다.



### EvalAlign: Evaluating Text-to-Image Models through Precision Alignment of Multimodal Large Models with Supervised Fine-Tuning to Human Annotations (https://arxiv.org/abs/2406.16562)
Comments:
          Github Repository: this https URL

- **What's New**: 최근 텍스트-이미지 생성 모델(text-to-image generative models)의 진보가 주목받고 있습니다. 하지만 이러한 모델의 성능을 정확히 반영하는 평가 지표가 부족한 실정입니다. 특히, 모델 최적화를 이끄는 세밀한 지표가 부족합니다. 본 논문에서는 이를 해결하고자 EvalAlign이라는 정확하고 안정적이며 세밀한 평가 지표를 제안합니다.

- **Technical Details**: EvalAlign은 대규모 데이터셋으로 사전 훈련된 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 기능을 활용합니다. 우리는 이미지 충실도(image faithfulness)와 텍스트-이미지 정렬(text-image alignment)의 두 가지 주요 차원에 초점을 맞춘 평가 프로토콜을 개발했습니다. 이러한 프로토콜은 상세하고 세밀한 지침과 특정 점수 옵션을 포함하여 생성된 이미지의 정확한 수동 점수 매기기를 가능하게 합니다. 또한, MLLM을 사람의 평가 판단에 맞추기 위해 감독 하에 미세 조정(Supervised Fine-Tune, SFT)을 수행하여 강력한 평가 모델을 만들어냈습니다.

- **Performance Highlights**: 총 24개의 텍스트-이미지 생성 모델을 테스트한 결과, EvalAlign은 기존 지표보다 뛰어난 안정성을 제공할 뿐만 아니라 인간의 선호도와 더 잘 일치함을 확인했습니다. 이는 EvalAlign이 모델 평가에서의 효과성과 유용성을 입증함을 보여줍니다.



### DaLPSR: Leverage Degradation-Aligned Language Prompt for Real-World Image Super-Resolution (https://arxiv.org/abs/2406.16477)
- **What's New**: 새로운 연구는 이미지 super-resolution(초해상도 복원) 작업에서 텍스트 기반 제어를 활용해, 정확하고 세밀한 이미지 복원을 가능케 하는 다중모달 프레임워크(DaLPSR)를 제안했습니다. 이는 특히 심하게 저하된 이미지에서 중요한 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: 이 논문에서는 두 가지 주된 보조 정보를 사용합니다. 첫 번째는 이미지 복원을 위한 프롬프트 정렬 디코더(IRPAD)로, 이는 LR 이미지의 저하 정도를 자동으로 식별해 유용한 저하 프롬프트를 생성합니다. 이를 위해 저하 과정을 나타내는 트리플렛 데이터를 구성한 보조 데이터셋을 포함하여 저하 정도를 여러 구간으로 나누고, 저하 프롬프트 생성을 미세하게 검색하는 과정으로 처리합니다. 두 번째로, MLLM(multi-modal large language model)을 사용해 사람이 인식 가능한 고수준의 의미적 프롬프트를 확보합니다. 이를 위해 Recognize Anything Model(RAM)을 도입해 이미지를 기반으로 개체 라벨을 생성하여 MLLM의 프롬프트 지침으로 사용합니다.

- **Performance Highlights**: 제안한 방법은 정량적 및 정성적 분석에서 최신 기술과 비교해 새로운 최상의 지각적 품질 수준을 달성했습니다. 특히, 실제 사례에서 참고 없이 측정한 지표 기반으로 가장 높은 성능을 보였습니다.



### A Symmetry Property of Christoffel Words (https://arxiv.org/abs/2406.16408)
Comments:
          In Proceedings GASCom 2024, arXiv:2406.14588

- **What's New**: 이 논문에서는 이중 변수의 대칭성을 소개하면서 트라페조이드(words) 이론에 동기부여받아, 체리스토펠(Christoffel) 단어(characterize)의 대칭성을 연구했습니다.

- **Technical Details**: 단어의 길이에 따른 요소들(cardinality of factors)의 연속성이 대칭적인 트라페조이드 단어의 이론을 기반으로 합니다. 이중 변수의 대칭성을 통해 체리스토펠 단어의 성격을 밝히고, 관련된 여러 결과를 도출했습니다.

- **Performance Highlights**: 이번 연구는 체리스토펠 단어의 대칭적 특성을 명확히 하여 관련 수학적 구조의 이해를 높였습니다.



### DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task (https://arxiv.org/abs/2406.16332)
- **What's New**: 최근 대형 언어 모델(LLMs)을 제로샷 패시지 순위 산정(passage ranking)에 적용하는 데 대한 관심이 증가하고 있습니다. 그러나 적절한 맥락 내 시연(in-context demonstrations)을 어떻게 선택할 것인지에 대한 연구는 상대적으로 적었습니다. 이 논문은 이러한 시연 선택 문제를 '검색 후 재순위화(retrieve-then-rerank)' 프로세스로 정의하고, 새로운 프레임워크 DemoRank를 제안합니다.

- **Technical Details**: DemoRank 프레임워크는 두 단계로 구성됩니다. 먼저, LLM의 피드백을 사용하여 시연 검색기(demonstration retriever)를 훈련합니다. 그런 다음, 시연 간 종속성을 고려한 새로운 종속 인식(dependency-aware) 시연 재랭커(demonstration reranker)를 도입하여 검색된 시연을 재정렬합니다. 이는 단순히 각 시연의 개별 피드백을 사용하는 것보다 더 나은 성능을 발휘하도록 설계되었습니다. 특히, 데모랭크의 훈련 샘플을 효율적으로 구성하여 시간 소모를 줄이면서도 종속성을 반영합니다.

- **Performance Highlights**: 포괄적인 실험 결과 DemoRank는 도메인 내(in-domain) 시나리오에서의 효과를 입증했으며, 도메인 외(out-of-domain) 시나리오에서도 강력한 일반화를 보였습니다. 더불어, 제한된 훈련 데이터, 다양한 시연 수, 미지의 데이터셋 등 여러 상황에서도 성능 향상을 확인했습니다. DemoRank는 특히 몇 샷 인콘텍스트 학습(few-shot ICL)에서 우수한 성능을 보였습니다.



### Song Data Cleansing for End-to-End Neural Singer Diarization Using Neural Analysis and Synthesis Framework (https://arxiv.org/abs/2406.16315)
Comments:
          INTERSPEECH 2024 accepted

- **What's New**: 본 연구에서는 신경망 분석과 합성 프레임워크(NANSY++)를 활용한 데이터 정제 방법을 제안하여 종단간 신경 다이아리제이션 모델(EEND)을 가수 다이아리제이션에 적용했습니다. 제안된 모델은 일반적으로 대중 음악에 포함된 코럴 싱잉(choral singing) 데이터를 솔로 싱잉(solo singing) 데이터로 변환합니다. 이 데이터 정제 방법은 코럴 싱잉 데이터의 잘못된 레이블링을 줄이고, 솔로 싱잉 데이터로 변환함으로써 EEND 모델의 효과적인 학습을 돕습니다.

- **Technical Details**: NANSY++는 비중첩(non-overlapped) 오디오 신호 재구성을 위해 학습된 프레임워크로, 기존의 큰 고립 음성 데이터 세트를 사용하여 재구성 능력을 강화합니다. 이 연구에서는 NANSY++를 사용하여 코럴 싱잉 데이터를 깨끗한 솔로 싱잉 데이터로 변환합니다. 이렇게 변환된 데이터는 에너지 기반 음성 활동 감지와 결합되어 EEND 모델 학습용 시뮬레이션 데이터 세트를 생성합니다.

- **Performance Highlights**: 제안된 데이터 정제 방법을 사용하여 학습된 EEND 모델은 기존의 방법과 비교하여 다이아리제이션 오류율이 14.8 포인트 개선되었습니다. 총 91개의 인기 듀엣 노래에 대한 실험 결과, 제안된 방법이 코럴 싱잉을 솔로 싱잉으로 효과적으로 변환함을 확인했습니다.



### Anomaly Detection of Tabular Data Using LLMs (https://arxiv.org/abs/2406.16308)
Comments:
          accepted at the Anomaly Detection with Foundation Models workshop

- **What's New**: 이번 연구에서는 큰 언어 모델(LLM, Large Language Models)을 사용하여 테이블형 데이터에서 이상치를 감지하는 문제를 다루었습니다. 특히, 사전 학습된 LLM들이 분포 특정 모델 맞춤 없이 데이터 배치 내에서 이상치 탐지가 가능함을 보여줍니다. 이를 위해 LLM에 맞춘 데이터 생성 프로세스를 통해 합성 데이터셋을 만들고, LLM을 최적화하는 전략을 제안했습니다.

- **Technical Details**: LLM들은 자연어 텍스트 형식으로 입력과 출력을 처리하기 때문에 사용자 입장에서 설정이 간편합니다. 본 연구에서는 배치 수준 AD를 텍스트 기반 작업으로 변환하는 시리얼라이제이션(Serialization) 방식과 하이퍼파라미터 조정 없이 작동하는 방법을 제안했습니다. 또한 GPT-3.5, GPT-4, Llama2, Mistral 같은 모델을 사용해 합성 및 실제 데이터를 갖고 실험을 수행했습니다.

- **Performance Highlights**: ODDS 벤치마크 실험에서 GPT-4가 최첨단 전이 학습 기반 이상 탐지 방법과 동등한 성능을 보였으며, 미스트랄(Mistral) 기반으로 미세 조정된 탐지기가 GPT-3.5를 능가하여 LLM 조정 전략의 효과를 확인할 수 있었습니다. 이로써 LLM을 활용한 배치 수준의 이상 탐지 능력이 입증되었습니다.



### Confidence Regulation Neurons in Language Models (https://arxiv.org/abs/2406.16254)
Comments:
          25 pages, 14 figures

- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 불확실성을 어떻게 표현하고 조절하는지에 대한 메커니즘을 조사합니다. 특히, 최근 발견된 엔트로피 뉴런(entropy neurons)과 새로운 토큰 빈도 뉴런(token frequency neurons)에 초점을 맞추고 있습니다. 엔트로피 뉴런은 높은 가중치(norm weight)를 가지며, 출력 분포의 엔트로피를 효과적으로 조절합니다. 반면, 토큰 빈도 뉴런은 각 토큰의 로그 빈도에 비례하여 로그잇(logit)을 증강하거나 억제하여 출력 분포를 조절합니다.

- **Technical Details**: 엔트로피 뉴런은 언엠베딩 매트릭스의 널 스페이스(null space)에 작용하여 잔여 스트림(residual stream)의 노름(norm)에 영향을 미칩니다. 이들 뉴런은 GPT-2, Pythia, Phi-2, Gemma 2B 등 다양한 모델에서 관찰되었습니다. 토큰 빈도 뉴런은 각 토큰의 로그 빈도에 따라 로그잇을 조절하여 모델의 출력 분포를 유니그램(unigram) 분포로 전환시킵니다. 이 연구에서는 엔트로피 뉴런이 반복적인 서브시퀀스(subsequences)를 감지하고 계속할 때 자신감을 관리하는 방법을 케이스 스터디로 제시합니다.

- **Performance Highlights**: 연구 결과, 엔트로피 뉴런과 토큰 빈도 뉴런은 모델의 자신감을 조절하며 정확한 예측과 잘못된 예측에서의 성능을 각각 개선하고 악화시킵니다. 이는 LLM의 내부 메커니즘이 각 뉴런 계열의 출력을 조절하여 모델 예측의 불확실성을 관리하는 방법을 보여줍니다.



### Blind Baselines Beat Membership Inference Attacks for Foundation Models (https://arxiv.org/abs/2406.16201)
- **What's New**: 본 연구는 대규모 Foundation 모델에 대해 진행된 Membership Inference (MI) 공격 평가가 결함이 있다는 것을 밝혀냈습니다. 특히, 기존의 MI 공격 평가가 멤버와 비멤버 데이터를 서로 다른 분포에서 샘플링함으로 인해 본래의 목적을 제대로 이행하지 못한다고 주장합니다.

- **Technical Details**: 기존의 MI 공격 평가는 보통 훈련 데이터와 비훈련 데이터를 무작위로 선정합니다. 본 연구는 8개의 MI 평가 데이터셋을 분석하여, 실제 모델 훈련을 고려하지 않고 멤버와 비멤버 분포를 구분하는 'blind attacks' 즉, '블라인드 공격'이 최신 MI 공격보다도 뛰어나다는 것을 보였습니다. 이는 현재 MI 공격이 기껏해야 데이터의 특징을 사용하여 멤버쉽을 잘못 추론할 뿐, 모델에서 실제 멤버쉽 누출을 추출하지 못한다는 것을 의미합니다.

- **Performance Highlights**: 연구진은 단순한 'blind attacks'를 설계하여, 모든 최신 MI 공격의 보고된 결과를 능가하는 성능을 보였습니다. 예를 들어, 한 데이터셋에서 특정 날짜를 기준으로 멤버쉽을 구분하는 임계치(threshold)를 적용하거나, 다른 텍스트 또는 텍스트-비전 데이터셋에서는 bag-of-words 또는 n-gram 분류기를 사용하였습니다. 이러한 접근법을 통해 기존 MI 공격이 'chance' 수준 이하로 성능이 낮다는 것을 입증하였습니다.



### GraphEval2000: Benchmarking and Improving Large Language Models on Graph Datasets (https://arxiv.org/abs/2406.16176)
Comments:
          Submitted to NeurIPs 2024 Dataset and Benchmark track, under review

- **What's New**: 최근 연구에서는 LLMs(Large Language Models)의 복잡한 그래프 구조 데이터에 대한 제한된 성능을 지적하였습니다. 이를 해결하기 위해, 우리는 GraphEval2000을 소개합니다. GraphEval2000은 처음으로 그래프 데이터 구조 문제 40개와 2000개의 테스트 사례를 포함한 종합적인 그래프 데이터셋입니다. 이를 바탕으로 코드 챌린지를 통해 LLMs의 그래프 추론 능력을 평가하는 프레임워크를 제시합니다.

- **Technical Details**: GraphEval2000 데이터셋은 Sparse, Planar, Regular, Complete 그래프의 네 가지 주요 분류와 연결된, 비연결된, 순환, 비순환 그래프의 네 가지 하위 분류로 구성됩니다. 우리는 현재 여덟 가지 인기 있는 LLMs를 평가하여, directed graphs(방향성 그래프)에 대한 이해도가 undirected graphs(비방향성 그래프)보다 더 높다는 것을 밝혔습니다. 또한, 우리는 Structured Symbolic Decomposition (SSD)을 제안합니다. SSD는 복잡한 그래프 문제를 해결하기 위해 LLMs의 성능을 향상시키는 지시 기반 방법입니다.

- **Performance Highlights**: SSD는 GPT-3.5, GPT-4, GPT-4o의 복잡한 그래프 문제에서 11.11%, 33.37%, 33.37%의 성능 향상을 보였습니다. 또한, 개인 LLMs가 오픈 소스 모델보다 일관되게 더 나은 성능을 나타내었지만, 그 성능 격차는 점점 좁아지고 있습니다.



### Contextualized End-to-end Automatic Speech Recognition with Intermediate Biasing Loss (https://arxiv.org/abs/2406.16120)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: 이 논문은 엔드투엔드(End-to-End, E2E) 자동 음성 인식(ASR) 분야에서 맥락화와 관련된 새로운 방법을 제안합니다. 특히, 중간 레이어에서 명시적인 바이어싱 손실(explicit biasing loss)을 부가적인 작업으로 사용하는 것이 텍스트 토큰이나 오디오 프레임을 원하는 목표에 더 잘 맞추는 데 도움이 될 수 있다고 가정합니다. 이를 통해 네트워크의 정규화와 맥락화가 더 잘 이루어질 수 있습니다.

- **Technical Details**: 제안된 중간 바이어싱 손실(intermediate biasing loss)은 CTC(Connectionist Temporal Classification) 손실 기반의 새로운 부가 작업으로, 오디오 인코더의 중간 표현을 이용해 맥락화를 개선합니다. 오디오 인코더는 Conformer 아키텍처를 사용하며, Transformer 및 컨볼루션 레이어로 구성되어 있습니다. 이 방법은 CTC 및 RNN-transducer 기반 모델에 적용될 수 있으며, 특히 CTC 및 RNN-transducer 모델에 대한 적용 사례를 제시합니다.

- **Performance Highlights**: 제안된 방법은 LibriSpeech 코퍼스에서 기존의 컨텍스트 바이어싱 방식보다 성능이 우수하며, 바이어스된 단어 오류율(B-WER)에서 상대적으로 22.5%의 개선을 달성하였습니다. 비바이어스 단어 오류율(U-WER)도 RNN-transducer 기반의 조인트 디코딩(joint decoding)을 활용하여 추가적으로 감소시켰습니다. 바이어싱 리스트가 큰 경우에도 비바이어스 단어 오류율을 최소화하는 데 효과적입니다.



### Decoder-only Architecture for Streaming End-to-end Speech Recognition (https://arxiv.org/abs/2406.16107)
Comments:
          Accepted for Interspeech 2024

- **What's New**: 새로운 연구는 디코더 전용 모델(Decoder-only model)을 활용하여 블록 단위 스트리밍 자동 음성 인식(ASR)을 제안합니다. 연구에서는 CTC 출력을 사용하여 음성 특징을 압축하고, 컨텍스트 임베딩(Context embedding)을 활용해 디코더에 순차적으로 제공하는 방법을 제안합니다. 이를 통해 블록마다 출력 토큰을 즉시 예측할 수 있게 합니다.

- **Technical Details**: 제안된 방법에서는 음성 발화가 블록 단위 conformer 기반 음성 서브네트워크에 의해 처리됩니다. 각 블록은 음향 정보를 나타내는 프롬프트를 생성하며, 이 프롬프트는 불필요한 프레임을 제거하여 압축됩니다. 이와 함께 동일한 블록에서 이전 블록의 자유 정보를 상속받아 컨텍스트 임베딩을 생성하고, 이를 디코더에도 추가 프롬프트로 제공합니다. 또한, 새로운 학습 방법으로 랜덤 길이의 프리픽스 프롬프트를 사용해 블록 단위 처리 중 발생하는 프롬프트 단절에 대한 로버스트한 모델을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 디코더 전용 스트리밍 ASR은 LibriSpeech test-other 셋에서 단어 오류율(word error rate, WER)을 기존 모델에 비해 8% 감소시키며, 처리 속도는 두 배 빠릅니다. 이 방법은 기존 스트리밍 CTC/트랜스듀서(transducer) 모델보다 더 높은 정확도를 보이며, 일반적인 인코더-디코더 모델보다 더 빠른 추론 속도를 달성합니다.



### PORT: Preference Optimization on Reasoning Traces (https://arxiv.org/abs/2406.16061)
- **What's New**: 이 논문은 Chain-of-Thought(COT) 단계를 활용한 preference optimization 기술을 제안하여, 대형 언어 모델(LLMs)의 reasoning 성능을 향상시키는 방법을 소개합니다. 이 방법을 통해 reasoning traces를 포함한 데이터셋에서 나온 답변을 선택하고 잘못된 답변을 생성하는 두 가지 보완적인 스킴(숫자 손상, 약한 LLM 프롬프트)을 제시합니다.

- **Technical Details**: 제안된 접근법은 GSM8K, AQuA-RAT, ARC 벤치마크에서 Falcon2-11B와 Mistral-7B 모델의 정확도를 높였습니다. 예를 들어, GSM8K 벤치마크에서는 최대 상대 8.47%의 정확도 향상을 달성했습니다. 또한, 더 많은 reasoning traces 데이터셋을 생성하는 것이 LLM의 성능을 더욱 높일 수 있음을 시사합니다.

- **Performance Highlights**: 가장 간단한 접근법을 사용하여 Falcon2-11B 모델 기반으로 GSM8K 벤치마크에서 8.47% 정확도 향상, AQuA 벤치마크에서 18.73% 정확도 향상을 달성했습니다. Mistral-7B를 기반으로 한 결과에서도 일관된 성능 향상을 확인했습니다. 또한 다양한 preference optimization 기법을 비교한 결과, Direct Preference Optimization(DPO)이 가장 우수한 성능을 보였습니다.



### AudioBench: A Universal Benchmark for Audio Large Language Models (https://arxiv.org/abs/2406.16020)
Comments:
          20 pages; Preprint; Code: this https URL

- **What's New**: 이번 연구에서는 AudioBench라는 새로운 벤치마크를 소개하며, 이는 오디오 대형 언어 모델(AudioLLMs)을 평가하기 위해 설계되었습니다. AudioBench는 음성 이해, 음성 해석, 그리고 오디오 장면 이해에 중점을 둔 8개의 고유한 작업과 26개의 신중하게 선정한 또는 새롭게 편집된 데이터셋으로 구성되어 있습니다. 이 벤치마크는 기존에 부족했던 오디오 모델의 평가 틀을 보완하고자 합니다.

- **Technical Details**: AudioBench는 일반적인 지시를 따르는 오디오 언어 모델들을 전문적으로 평가하기 위해 설계되었습니다. 이 벤치마크는 음성 의미 이해, 환경 소리 해석, 그리고 비언어적 특성 인식을 포함한 다양한 작업을 아우릅니다. 공정한 비교를 위해 다양한 프롬프트 템플릿(prompt templates)을 사용하고 입력 길이를 초부터 분 단위까지 다양하게 조절하여 모델의 성능을 평가합니다. 'AudioBench' 툴킷은 오픈 소스로 제공되어, 향후 모델 개발 시 간단한 모델 추론 프로그램을 통해 평가할 수 있도록 합니다.

- **Performance Highlights**: AudioBench를 통해 네 개의 오픈소스 모델을 평가한 결과, 모든 작업에서 일관되게 뛰어난 성능을 보이는 단일 모델은 없었습니다. 이 결과는 AudioLLM의 미래 발전 가능성을 강조하며, 향후 연구의 방향을 제시합니다. 또한, 오디오 입력을 기반으로 하는 프롬프트 기반 모델의 출력을 평가하기 위한 메트릭스를 연구하였고, 오픈소스 모델 평가 방법을 탐구하였습니다. 특히, LLaMA-3 모델은 GPT-4 모델과 높은 상관관계를 보였습니다.



### Effectiveness of ChatGPT in explaining complex medical reports to patients (https://arxiv.org/abs/2406.15963)
Comments:
          under review

- **What's New**: 이번 연구는 ChatGPT (GPT 4)가 대장암 및 전립선암 환자들에게 다학제 팀(MDT) 보고서를 설명하는 데 얼마나 효과적인지를 평가합니다. 이 보고서는 전문적인 의학 용어로 작성되며 임상 지식을 전제로 하기 때문에 ChatGPT의 능력을 테스트하기에 좋은 자료입니다. 연구 결과, ChatGPT가 부정확한 정보 제공, 부적절한 언어 사용, 개인화의 한계, AI에 대한 불신, 임상 워크플로에의 통합 문제 등을 극복해야 한다고 결론지었습니다.

- **Technical Details**: 연구는 6개의 모의 MDT 보고서를 기반으로 진행되었으며, 이는 실제 환자 보고서를 정확히 모방한 것입니다. ChatGPT가 각 보고서에 대해 네 가지 시나리오 질문에 응답하도록 했습니다. 이러한 응답은 MDT 보고서 작성자, 다른 임상의 및 비전문가(일반인)에 의해 분석되었습니다. 또한, 암 환자, 간병인, 컴퓨터 과학자, 임상의가 포함된 세 개의 포커스 그룹에서 이러한 응답을 논의했습니다.

- **Performance Highlights**: ChatGPT의 응답은 여러 문제점을 드러냈습니다. GPT 4가 제공한 정보를 분석한 결과, 응답에서 부정확한 정보, 부적절하거나 이해하기 어려운 표현, 개인화된 상담의 부족, AI 시스템에 대한 신뢰 부족 등의 문제가 발견되었습니다. 이러한 문제들은 대형 언어 모델(LLMs)이 개인의 복잡한 의학 정보를 설명하는 데 사용되기 전에 해결되어야 합니다.



### Beyond Individual Facts: Investigating Categorical Knowledge Locality of Taxonomy and Meronomy Concepts in GPT Models (https://arxiv.org/abs/2406.15940)
Comments:
          27 pages, 23 figures, 12 tables

- **What's New**: 본 논문에서는 Generative Pre-trained Transformer (GPT) 모델 내 지식 위치에 관한 새로운 접근 방식을 제안합니다. 기존 연구와 달리 개별 사실이 아닌 개념과 연관된 지식의 클러스터에 초점을 맞추고 있습니다. 이를 위해 DARC라는 새로운 데이터셋을 만들었으며, 기존의 인과 매개 분석(causal mediation analysis) 방법을 개념 간의 관련성에 적용하여 조사했습니다.

- **Technical Details**: 본 연구에서는 두 가지 주요 계층적 구조인 분류 체계(taxonomy)와 부분 전체 관계(meronomy)를 분석했습니다. DARC 데이터셋은 총 34개의 개념과 12만 개 이상의 사실로 구성되며, 이를 토대로 지식이 모델 내 특정 영역에 존재하는지 여부를 확인했습니다. 기존의 인과 분석 방법을 사용하여 관련된 범주가 유사한 중요 영역을 나타내는지 조사했습니다.

- **Performance Highlights**: 실험 결과, 유사한 범주가 덜 유사한 범주와 비교했을 때 모델 내에서 비슷한 중요 영역을 나타낸다는 것을 발견했습니다. 그러나 개별 범주 하위 집합의 세밀한 지역화는 명확하지 않았습니다. 분류 체계 관련 그룹 간의 지식 지역성은 특히 강하게 나타났고, 부분 전체 관계에서도 유사한 경향을 보였지만 다소 약했습니다.



### Language Alignment via Nash-learning and Adaptive feedback (https://arxiv.org/abs/2406.15890)
Comments:
          Accepted at ICML 2024 Workshop on Models of Human Feedback for AI Alignment, Vienna, Austria

- **What's New**: 최근 연구는 인간 피드백(Human Feedback)을 통한 Nash 학습(Nash Learning)이 대형 언어 모델(Large Language Model, LLM) 정렬에 유망하다는 것을 보여주었습니다. 우리 연구팀은 이를 더욱 발전시켜 개선된 적응 피드백(adaptive feedback)을 활용한 미러 디센트 알고리즘(mirror descent algorithm) 방식으로 정렬 과정을 다루었습니다. 이를 통해 선호 모델(preference model) 학습이나 주석 데이터셋이 필요하지 않게 되었으며, 자가 정렬(self-alignment)을 가능하게 하는 새로운 알고리즘 LANA(Language Alignment via Nash-learning and Adaptive feedback)를 개발했습니다.

- **Technical Details**: LANA 알고리즘은 개선된 상대의 적응 피드백을 사용하는 미러 디센트 알고리즘(mirror descent algorithm)을 기반으로 합니다. 기존 연구들이 주석된 데이터셋이나 학습된 선호 모델에 의존하는 것과 달리, 우리의 방법은 해당 모델이나 데이터셋 없이도 개선된 정책을 통해 학습됩니다. 또한, KL 정규화를 포함하기 위한 수정된 Mirror Descent 알고리즘을 사용합니다. 특히 우리 방법은 참조 정책(reference policy) 없이 작동합니다.

- **Performance Highlights**: 다양한 실험과 수학적 검토를 통해 LANA 알고리즘이 인간 주석된 선호 데이터셋 없이도 성공적으로 자가 정렬이 가능함을 증명했습니다. 또한, 개선된 적응 피드백을 통해 발생할 수 있는 노이즈와 유틸리티 변화를 견딜 수 있는 견고한 자체 보상(training process)을 가지고 있음을 보여주었습니다.



### BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions (https://arxiv.org/abs/2406.15877)
Comments:
          44 pages, 14 figures, 7 tables, built with love by the BigCode community :)

- **What's New**: 최근 큰 언어 모델(LLMs)의 프로그래밍에 대한 발전으로 자동화된 소프트웨어 엔지니어링이 크게 강화되었습니다. 본 연구는 기존 벤치마크가 짧고 독립적인 알고리즘 작업에 제한되어 있다는 점을 지적하며, 더 복잡하고 실용적인 프로그래밍 작업을 해결할 수 있는 능력을 평가하기 위해 새로운 벤치마크인 'Bench'를 도입했습니다.

- **Technical Details**: Bench는 139개의 라이브러리와 7개의 도메인에서 1,140개의 세분화된 프로그래밍 작업을 해결하기 위해 다양한 함수 호출을 도구로 사용하는 능력을 평가합니다. 각 작업은 평균 5.6개의 테스트 케이스와 99%의 브랜치 커버리지를 포함합니다. 또한, 자연 언어 지향 변종인 'Benchi'를 제안하여 원래의 도크스트링을 핵심 정보만을 포함한 짧은 지시문으로 자동 변환합니다.

- **Performance Highlights**: 60개의 LLMs를 광범위하게 평가한 결과, LLMs는 복잡한 지시사항을 따라 함수 호출을 정확하게 사용할 수 있는 능력이 인간의 97% 대비 최대 60%에 불과하다는 점이 드러났습니다. 이는 이 분야에서 추가적인 발전이 필요함을 강조합니다.



### Intrinsic Dimension Correlation: uncovering nonlinear connections in multimodal representations (https://arxiv.org/abs/2406.15812)
- **What's New**: 이 논문은 고차원 데이터 매니폴드 간의 비선형 상관관계를 정량화하는 새 메트릭을 제안합니다. 이를 통해 고차원 및 비선형 상관관계를 탐지하기 위한 새로운 접근방식을 제공합니다. 특히, 시각 및 텍스트 임베딩 간의 상관관계를 명확히 드러내며, 기존 메트릭이 실패하는 데이터를 분석합니다.

- **Technical Details**: 제안된 메트릭은 '소개된 Intrinsic Dimension Correlation (IdCor)'입니다. Intrinsic Dimension (Id)는 데이터를 설명하는 데 필요한 최소 변수 수를 의미하며, 이 메트릭은 두 데이터 표현이 상관관계가 있으면 결합된 데이터셋의 Intrinsic Dimension이 감소한다는 개념에 기반합니다. 이를 통해 비선형 상관관계를 탐지할 수 있습니다.

- **Performance Highlights**: 제안된 메트릭은 합성 데이터와 제어된 환경에서 검증되었으며, 기존 기법들과의 비교에서 장점과 단점을 보여줍니다. 또한 대규모 신경망 표현에 대한 분석에서 뛰어난 성능을 보였으며, 기존 메트릭이 상관관계를 탐지하는 데 어려움을 겪는 멀티모달 데이터(시각 및 텍스트 임베딩)에서 강력한 상관관계를 밝혀냈습니다.



### What Matters in Transformers? Not All Attention is Needed (https://arxiv.org/abs/2406.15786)
Comments:
          15 pages, 13 figures, 6 tables

- **What's New**: 이번 연구에서는 Transformer 기반의 대규모 언어 모델(LLMs)의 다양한 모듈에서의 중복성을 조사했습니다. 특히, 블록, MLP (다층 퍼셉트론) 및 Attention 레이어의 중복성을 분석하고, 이를 통해 메모리 및 계산 비용을 줄이는 방법을 제안했습니다. 우리가 제안한 방법 중 하나는 'Attention Drop'으로, 이는 모델의 성능 저하 없이 많은 Attention 레이어들을 제거할 수 있게 합니다.

- **Technical Details**: 이번 연구에서는 유사성 기반의 측정 지표를 사용해 LLM 내 모듈의 중복성을 측정했습니다. 이 지표는 입력과 출력 간의 코사인 유사성을 계산하여 중복 모듈을 식별하고 제거합니다. 먼저, 각 트랜스포머 블록에서 높은 유사성을 보이는 모듈을 조사했으며, 특히 Attention 레이어가 높은 중복성을 가지는 것을 발견했습니다. 이로 인해 많은 Attention 레이어를 제거할 수 있었습니다. 또한, 'Joint Layer Drop'이라는 방법을 제안하여 Attention과 MLP 레이어를 함께 삭제하여 더 높은 성능 향상과 드롭 비율을 달성했습니다.

- **Performance Highlights**: Llama-3-70B 모델의 경우, Attention 레이어의 절반을 제거한 후에도 원래 모델과 비슷한 성능을 유지했습니다. 이는 LLM의 성능을 유지하면서도 메모리와 계산 비용을 크게 줄일 수 있는 가능성을 보여줍니다.



### Unveiling and Harnessing Hidden Attention Sinks: Enhancing Large Language Models without Training through Attention Calibration (https://arxiv.org/abs/2406.15765)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)에서 주목 메커니즘의 이해를 깊게 하기 위해 처음으로 주목 싱크(attention sinks) 현상을 탐구했습니다. 특히, 초점 토큰에서뿐만 아니라 후속 토큰에서도 주목 싱크가 발생한다는 것을 발견했습니다. 이를 기반으로, LLM의 성능을 향상시키기 위해 훈련 없이 주목 분포를 실시간으로 최적화하는 주목 보정 기술(Attention Calibration Technique, ACT)을 제안합니다.

- **Technical Details**: 연구는 다양한 입력과 작업에 대한 추론 동안 LLM의 주목 분포를 시각화하는 것에서 시작합니다. 이런 시각화를 통해 초기 토큰뿐 아니라 후속 토큰에서도 주목 싱크가 나타나는 것을 처음으로 발견했습니다. 또한, 모든 주목 싱크가 LLM의 정확도에 긍정적인 영향을 미치는 것은 아니며, 이를 기반으로 입력 적응적 방식으로 주목 분포를 실시간 최적화하는 ACT를 개발했습니다. ACT는 훈련(finetuning) 없이, 즉 모델의 가중치를 수정하지 않고도 작동합니다.

- **Performance Highlights**: ACT를 적용한 다양한 LLMs에서 평균 7.30%의 정확도 향상을 달성했습니다(Llama-30B 모델 기준). 다양한 데이터셋에서 성능 향상이 있었으며, 특히 복잡한 다중 회차 대화 작업에서도 성능이 개선되었습니다. 예를 들어, MT-Bench 데이터셋에서는 점수가 최대 0.13 향상되었습니다.



### Multimodal Segmentation for Vocal Tract Modeling (https://arxiv.org/abs/2406.15754)
Comments:
          Interspeech 2024

- **What's New**: 이번 연구에서는 발성 기관의 모델링을 위한 새로운 기법들을 소개합니다. 특히, RT-MRI 비디오의 분할(segmentation)을 위해 비전 기반 딥 레이블링 전략과 오디오를 추가로 활용한 멀티모달 알고리즘을 제안합니다. 이를 통해 RT-MRI 데이터셋의 레이블링을 크게 확장하여 새로운 벤치마크를 설정했습니다. 또한, 75명의 화자로 구성된 RT-MRI 데이터셋의 라벨을 공개하여, 기존 공개된 RT-MRI 데이터 양을 9배 이상 늘렸습니다.

- **Technical Details**: 이 연구에서는 크게 두 가지 주요 기법을 제안합니다. 첫째, speaker-independent 발성 기관 경계 분할을 위한 비전 기반 fully-convolutional neural network (U-Net)을 사용했습니다. 둘째, 음성 파형을 추가하여 발성 기관 분할의 정확도를 높이는 멀티모달 Transformer 모델을 도입했습니다. 기존의 RT-MRI 분할 알고리즘은 초기 프레임에 대한 수작업 경계 추적과 각 프레임에 대한 비선형 최적화를 필요로 하여 시간이 많이 소요되고, 최적화의 불확실성이 높았습니다.

- **Performance Highlights**: 새로운 멀티모달 접근 방식은 기존의 방법들에 비해 RT-MRI 분할에서 월등한 성능을 보여주었으며, 특히 75명의 다양한 연령, 성별, 악센트를 가진 화자들로 구성된 데이터셋에 대해 검증되었습니다. U-Net 기반 모델은 7명의 훈련 화자로부터 약 90분의 RT-MRI 비디오를 학습하여 각 프레임 별 95개의 주요 발성 기관 포인트를 추적했습니다. 이 모델은 2차원 확률 분포를 활용한 Kullback–Leibler 발산을 로스 함수로 사용하여 더욱 정확한 분할 결과를 도출했습니다.



### TacoLM: GaTed Attention Equipped Codec Language Model are Efficient Zero-Shot Text to Speech Synthesizers (https://arxiv.org/abs/2406.15752)
Comments:
          INTERSPEECH 2024

- **What's New**: 최근 neural codec language model(LM)은 zero-shot text-to-speech(TTS) 합성에서 강력한 성능을 보였지만, 여전히 속도와 안정성 면에서 한계가 있었습니다. 이를 해결하기 위해, TacoLM이라는 새로운 neural codec LM 변형을 도입했습니다. TacoLM은 gated attention 메커니즘을 활용하여 모델 크기를 줄이고 효율성을 높이며, 추가적인 gated cross-attention 레이어를 통해 합성된 음성의 정확도를 개선합니다.

- **Technical Details**: TacoLM은 Librispeech 코퍼스 평가에서 VALL-E보다 우수한 단어 오류율, 화자 유사성, 평균 의견 점수를 달성했습니다. TacoLM은 두 단계로 구성된 프레임워크로, AR(autogressive) 및 NAR(non-autoregressive) 언어 모델을 포함합니다. 이 모델은 텍스트 인코더, neural audio codec, autoregressive 언어 모델, non-autoregressive 언어 모델의 네 가지 주요 구성 요소로 이루어져 있습니다. 특히, 에너코드(EnCodec)라는 미리 훈련된 neural audio codec을 이용하여 입력 오디오를 디스크리트 토큰으로 변환합니다.

- **Performance Highlights**: TacoLM은 VALL-E와 비교하여 90% 더 적은 파라미터로 5.2배 더 빠른 속도를 자랑합니다. 또한, 실험 결과를 통해 TacoLM이 WER(word error rate), 화자 유사성, CMOS(Comparative Mean Opinion Score), SMOS(Speech Mean Opinion Score) 등 다양한 지표에서 VALL-E보다 뛰어난 성능을 보였습니다.



### Evaluating Large Vision-and-Language Models on Children's Mathematical Olympiads (https://arxiv.org/abs/2406.15736)
- **What's New**: 최근 몇 년간 대형 비전 및 언어 모델(LVLMs)의 발전이 두드러지고 있습니다. 이러한 모델들은 수학 및 알고리즘적 사고능력을 테스트하는 과정에서 인간 능력을 초과하는 경우도 나타났습니다. 본 연구는 이러한 LVLMs가 실제로 아이들의 수학적 문제 해결 능력을 얼마나 잘 따라잡을 수 있는지에 대한 체계적인 분석을 시도합니다. 이를 위해 국제적인 수학 올림피아드인 Mathematical Kangaroo (MK)에 사용된 문제들을 통해 AI의 성능을 평가했습니다.

- **Technical Details**: 본 연구에서는 2020년에서 2024년까지의 MK 올림피아드 문제들을 모아 SMART-840이라는 데이터셋을 구성하였습니다. 이 데이터셋을 활용하여 GPT-4o, Gemini, Claude-3 Opus, XGEN-MM 등 최신 LVLMs의 성능을 분석했습니다. 구체적으로, 문제들은 아이들이 실제로 문제를 푸는 성과와 비교할 수 있는 다양한 난이도로 구성되었습니다. 분석 결과, LVLMs는 고학년 문제를 잘 풀지만 저학년 문제에서는 성능이 부족함을 보였습니다. 또한 AI 모델과 어린이의 문제 해결 방식이 일치하지 않으며, 서로 다른 종류의 추론 방식을 취하고 있다는 점도 확인했습니다.

- **Performance Highlights**: LVLMs는 고학년 학생들이 푸는 문제에서는 10-20%가량의 성능 격차를 보였지만, 저학년 문제에서는 30-40% 정도로 더 큰 격차가 나타났습니다. 이는 AI 모델이 기초적인 추론 능력에서 부족함을 나타낸다고 결론지을 수 있습니다. 또한 서로 다른 프롬프트로 문제를 다시 풀어보았을 때 AI 모델의 응답에 상당한 변동성이 있음을 확인했습니다. 이러한 분석은 AI 모델이 인간의 인지 능력과 비교할 때 현재 어떤 한계가 있는지를 명확히 보여줍니다.



### SAIL: Self-Improving Efficient Online Alignment of Large Language Models (https://arxiv.org/abs/2406.15567)
Comments:
          24 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 현재의 오프라인 정렬 방식들이 고정된 선호 데이터셋에 과도하게 의존함으로써 최적의 성능을 발휘하지 못하는 문제를 해결하고자 한다. 온라인 RLHF(Reinforcement Learning from Human Feedback) 방법을 기반으로, 비레벨 최적화(bilevel optimization)을 효율적인 단일 레벨 최적화(single-level optimization) 방법으로 전환하여 새로운 응답을 생성하고, 모델 정렬을 반복적으로 정제하는 방법을 제안했다. 이를 통해 기존 온라인 RLHF 방법들을 통합하고, 은밀한 방식으로 성능을 향상시켰다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLM)의 온라인 정렬이 비레벨 최적화(bilevel optimization)로 뒷받침된다는 것을 증명했다. 이를 첫번째 레벨 최적화 방법으로 줄임으로써, 새로운 샘플을 생성하고 모델 정렬을 반복적으로 정제하는 방식으로 설계되었다. 이 과정에서 선호 레이블을 규제하고 응답을 탐색함으로써, 온라인 및 자기 개선 방식으로 작동할 수 있다. 또한, 기존 온라인 RLHF 방법을 특수한 경우로 일반화함으로써, 더욱 확장성 있는 프레임워크를 제안했다.

- **Performance Highlights**: 제안된 방법은 공개 데이터셋에서 최신 상태의 반복적 RLHF 방법들과 비교하여 상당히 향상된 정렬 성능을 보여준다. 프레퍼런스 오라클(preference oracle)에 접근 가능한지 여부와 상관없이, 새로운 응답 생성을 통해 정렬 성능을 최적화하고, 기존의 제한점을 극복하였다.



### Geneverse: A collection of Open-source Multimodal Large Language Models for Genomic and Proteomic Research (https://arxiv.org/abs/2406.15534)
Comments:
          8 pages

- **What's New**: Geneverse라는 이름의 새로운 LLM 및 MLLM 컬렉션이 제안되었습니다. 이 모델들은 유전자 기능 설명 생성, 단백질 구조로부터의 단백질 기능 추론, 공간 전사체 데이터에서 마커 유전자 선택 등 유전체 및 단백질체 연구의 세 가지 새로운 과제를 해결하기 위해 훈련되고 평가되었습니다.

- **Technical Details**: Geneverse 모델 컬렉션은 LLaMA2-7B, LLaMA2-13B, Mistral-7B, Gemma-7B와 같은 다양한 오픈 소스 기초 모델(foundation models)을 기반으로 합니다. 이를 위해 PEFT(Parameter-Efficient FineTuning) 기술과 전체 매개변수 미세 조정 기술이 사용되었습니다. PEFT 기술 중 하나인 LoRA(Low-rank Adaption)가 사용되었습니다.

- **Performance Highlights**: Geneverse는 유전자 및 단백질 기능 요약 생성에서 우수한 성능을 보여주었고, 폐쇄형 대규모 모델을 능가할 수 있음을 입증했습니다. 추가적으로 모델의 진실성과 구조적 정확성을 평가한 결과, Geneverse 모델이 현장에서 높은 성능을 보였습니다.



### Data Efficient Evaluation of Large Language Models and Text-to-Image Models via Adaptive Sampling (https://arxiv.org/abs/2406.15527)
- **What's New**: 최근 AI 모델 평가에서 SubLIME이라는 효율적인 평가 프레임워크가 소개되었습니다. 이 프레임워크는 적응형 샘플링(adaptive sampling) 기술을 사용하여 벤치마크 데이터셋의 대표 서브셋을 생성합니다. 이를 통해 전체 데이터셋과 비교할 때 통계적으로 일치하는 모델 랭킹을 보장하며, 평가 비용을 절감할 수 있습니다.

- **Technical Details**: SubLIME은 클러스터링(clustering) 및 품질 기반(quality-based) 샘플링 같은 여러 적응형 샘플링 기법을 활용합니다. 예를 들어, MMLU 벤치마크에서는 1% 샘플링 비율로도 모델 랭킹과 점수 분포를 유지할 수 있습니다. 텍스트에서 이미지로 변환하는 모델(text-to-image models)을 평가하기 위해서는 HEIM 리더보드 데이터를 활용하여 최적의 샘플링 기법을 동적으로 선택합니다.

- **Performance Highlights**: SubLIME은 6개의 NLP 벤치마크에 대한 실험에서 품질 기반 샘플링이 전체 데이터셋과 높은 상관관계(0.85에서 0.95)를 유지하면서도 샘플링 비율을 10%로 줄일 수 있다는 점을 보여주었습니다. 또한, SubLIME은 25개의 텍스트에서 이미지로 변환하는 모델을 17개의 다양한 벤치마크에서 평가할 때도 평가 비용을 크게 줄이면서 랭킹 무결성을 유지합니다.



### PKU-SafeRLHF: A Safety Alignment Preference Dataset for Llama Family Models (https://arxiv.org/abs/2406.15513)
Comments:
          a sibling project to SafeRLHF and BeaverTails

- **What's New**: PKU-SafeRLHF 데이터셋은 대규모 언어 모델(LLMs)의 안전성 정렬 연구를 촉진하기 위해 설계된 새로운 데이터셋입니다. BeaverTails와 같은 기존 프로젝트와 유사하며, 도움 되는(helpfulness)과 무해함(harmlessness) 평가를 별도로 제공합니다. 전체적으로 44.6k의 개선된 프롬프트와 265k 질문-답변 쌍에 대해 19개의 해로움 카테고리 및 세 가지 심각도 수준의 안전 메타 라벨이 제공됩니다.

- **Technical Details**: 데이터셋은 두 가지 타입의 annotations을 제공합니다: 265k Q-A 쌍의 안전 메타 라벨과 166.8k Q-A-B 쌍의 dual-preference 및 single-preference 데이터. 연구를 위해 Llama 패밀리 모델을 사용해 답변을 생성했으며, RLHF 및 심각도 민감 조정을 통해 LLMs의 위험 통제 및 안전 중심 RLHF 알고리즘을 학습했습니다.

- **Performance Highlights**: PKU-SafeRLHF 데이터셋은 연구자들이 LLM의 안전성을 높이는 데 중요한 자원이 될 것입니다. 특히 안전 메타 라벨은 19개 해로움 카테고리와 세 가지 심각도 수준을 포함하여 높은 일관성을 보입니다. 또한, RLHF를 통해 모델의 고품질 프리퍼런스(annotation) 데이터가 반영된 성능과 위험 통제의 효율성을 입증했습니다.



### CSRT: Evaluation and Analysis of LLMs using Code-Switching Red-Teaming Datas (https://arxiv.org/abs/2406.15481)
- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)의 다국어 능력과 안전성을 조명하면서도 기존 벤치마크는 이를 종합적으로 평가하지 못하고 있으며 수동 주석에 지나치게 의존하고 있습니다. 이 논문에서는 다국어 이해와 안전성을 동시에 테스트할 수 있는 간단하지만 효과적인 레드팀 기법(red-teaming technique)인 코드-스위칭 레드팀(CSRT)을 소개합니다. 또한 CSRT 데이터셋을 공개하며, 10개 언어를 결합한 315개의 코드-스위칭 쿼리를 포함하고 다양한 바람직하지 않은 행동을 유도합니다.

- **Technical Details**: 10개의 최첨단 LLMs를 사용한 광범위한 실험을 통해, CSRT가 기존의 다국어 레드팀 기법보다 유의미하게 더 많은 공격을 수행하며, 영어에서 기존 방법보다 46.7% 더 많은 공격을 달성한다는 것을 입증했습니다. 또한 16,000개의 샘플을 사용하여 여러 측면에 대한 평균 연구에서, 확대 법칙(scaling laws), 비안전한 행동의 범주, 최적의 데이터 생성을 위한 입력 조건 등을 포함하여 유해한 응답을 분석하였습니다.

- **Performance Highlights**: CSRT는 기존의 다국어 레드팀 기법들보다 46.7% 더 높은 공격률을 보였으며, 다양한 언어 조합을 통해 생성된 코드-스위칭 공격 프롬프트의 확장성을 검증했습니다.



### WundtGPT: Shaping Large Language Models To Be An Empathetic, Proactive Psychologis (https://arxiv.org/abs/2406.15474)
- **What's New**: WundtGPT는 정신 건강 분야를 위해 특별히 설계된 대형 언어 모델(LLM, Large Language Model)로, 심리학자의 공감(empathy)과 적극적인 지도(proactive guidance)를 통해 진정하고 효과적인 의사-환자 관계(DPR, Doctor-Patient Relationship)를 구축합니다. 이 모델은 심리학자와 환자 사이의 실제 대화를 통해 습득된 지식을 활용하여 환자가 대면 소통을 꺼릴 때 심리 상태를 이해하도록 돕습니다.

- **Technical Details**: WundtGPT는 심리 진단의 순서(Chain of Psychodiagnosis), 질문의 모음(Collection of Questions), 공감 제약(Empathy Constraints)을 포함하는 종합 프롬프트(prompt)를 사용하여 LLM의 질문 및 진단을 유도합니다. 최종 모델은 LLaMA3-8B-Chinese-Chat 베이스 모델을 기반으로 지도 학습을 통해 미세 조정되었습니다. 또한, 인지적 공감(cognitive empathy)과 정서적 공감(emotional empathy)을 기반으로 한 보상 모델을 제안하여 공감 기반 정신 건강 전문직과의 정렬을 촉진합니다.

- **Performance Highlights**: WundtGPT는 F1 점수와 정확도를 사용하여 감정 벤치마크 데이터셋에서 임상 진단 평가를 수행하였습니다. 이 모델은 프로액티브 진단(proactive diagnosis) 및 다중 턴 대화에서 따뜻한 심리 상담을 제공하는 능력을 평가받았고, 기존의 LLMs보다 전반적으로 우수한 성능을 보였습니다.



### D2O: Dynamic Discriminative Operations for Efficient Generative Inference of Large Language Models (https://arxiv.org/abs/2406.13035)
- **What's New**: 이번 연구에서는 Large Language Models(LLMs)의 긴 텍스트 생성 시 필요한 KV 캐싱 용량을 최적화하는 새로운 방법인 Dynamic Discriminative Operations(D2O) 기법을 소개합니다. 이 방법은 기존의 볼륨 큰 KV 캐싱의 문제를 해결하면서도 중요한 컨텍스트를 유지하여 텍스트 생성의 품질을 유지합니다.

- **Technical Details**: D2O 기법은 두 가지 주요 전략을 사용합니다. 첫째, 계층별로 주의 가중치의 밀도 차이를 고려하여 중요한 정보가 손실되지 않도록 하는 차별화된 캐시 비울(deletion) 전략을 적용합니다. 둘째, 계층 내비울 전략에서는 이전에 비워진 토큰의 중요도를 재판별(discriminate)하여 필요 시 캐시에 다시 합류합니다. 이를 위해 EMA(Exponential Moving Average) 임계값을 사용한 보상 메커니즘을 도입하여 유사한 토큰들과 병합합니다.

- **Performance Highlights**: D2O를 통해 실험한 결과, 메모리 절약과 고품질 긴 텍스트 생성이 동시에 가능했으며, 캐시 제약을 받는 상황에서도 성능이 크게 향상되었습니다. 다양한 LLM 아키텍처와 벤치마크에서 D2O가 뛰어난 성능을 보여주었습니다. 1) 추론 속도가 기존보다 최대 3배 이상 향상되고, 2) 이유 기반(task-based) 성능에서 현저한 성능 향상, 3) 긴 텍스트의 컨텍스트 유지 성능에서 우수한 성능을 확인하였습니다.



New uploads on arXiv(cs.IR)

### Ragnar\"ok: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track (https://arxiv.org/abs/2406.16828)
- **What's New**: 최근 몇 달 동안, 검색 기술이 Retrieval-Augmented Generation (RAG) 시스템을 통해 크게 발전하였습니다. 이 기술은 실시간 데이터를 대형 언어 모델(LLM)과 결합하여 정보를 제공합니다. 이러한 RAG 시스템을 테스트하고 평가하기 위해, TREC 2024 RAG 트랙을 제안했습니다.

- **Technical Details**: 우리는 'Ragnarök' 이라는 재사용 가능한 프레임워크를 개발했습니다. 이 프레임워크는 Pyserini 및 RankLLM과 같은 기존 파이썬 프레임워크와 깊이 통합되어 있으며, 사용자 친화적인 REST API와 WebUI를 제공합니다. 'Ragnarök'은 두 가지 주요 모듈로 구성됩니다: Retrieval와 Augmented Generation.

- **Performance Highlights**: 우리의 실험 결과 OpenAI의 GPT-4o는 Cohere의 Command R+에 비해 더 상세한 답변을 제공한다고 관찰했습니다. 이 프레임워크와 주요 베이스라인은 공개되어 있으며, TREC 2024 RAG 트랙에서 사용될 예정입니다. 향후, 더 다양한 LLMs 기준을 포함하고 프레임워크를 지속적으로 개선할 계획입니다.



### Meta-experiments: Improving experimentation through experimentation (https://arxiv.org/abs/2406.16629)
Comments:
          6 pages, 2 figures, 1 table

- **What's New**: 본 논문은 A/B testing을 최적화하기 위한 메타 실험(meta-experiments)의 응용에 대해 다룹니다. 즉, A/B testing의 과정을 개선하기 위해 A/B testing을 실시하는 것입니다.

- **Technical Details**: 연구에서는 메타 실험을 통해 실험자들이 더 충분히 강력한 A/B 테스트를 실행할 수 있도록 한 사례를 제시합니다. 또한 'dog fooding'의 이점을 강조합니다. 'dog fooding'은 스스로가 고객이 되어 자신의 제품이나 서비스를 테스트하는 과정을 의미합니다.

- **Performance Highlights**: 이 접근 방식은 실험자들이 자신의 실험 과정을 개선하고, A/B 테스트가 더욱 유효하고 강력하게 수행될 수 있도록 돕습니다.



### Star+: A New Multi-Domain Model for CTR Prediction (https://arxiv.org/abs/2406.16568)
- **What's New**: 이번 논문에서는 Star 모델에서 영감을 받은 새로운 멀티 도메인 클릭 예측 모델인 Star+를 소개합니다. 기존의 싱글 도메인 접근 방식과 일부 멀티태스크 학습 기법이 멀티 도메인 환경에서 도메인 별 데이터 분포와 복잡한 도메인 간 관계를 효과적으로 포착하는 데 어려움을 겪는 반면, Star+는 공유된 정보와 도메인별 정보를 다양한 융합 전략을 통해 최적의 균형을 찾습니다. 이 논문은 추천 시스템의 발전에 기여하며, 멀티 도메인 환경에서의 클릭 예측의 정확성과 효율성을 크게 향상시키는 방법을 제시합니다.

- **Technical Details**: Star+는 add, adaptive add, concatenation, gating fusions와 같은 다양한 융합 전략을 통해 공유 정보와 도메인별 정보 간의 상호작용을 향상시킵니다. 또한 layer normalization, batch normalization, partition normalization과 같은 다양한 정규화 기법이 모델의 성능에 미치는 영향을 조사합니다. Star+ 모델은 주요 산업과 공공 데이터셋에서 그 효과를 입증했으며, 이는 멀티 도메인 환경에서 적용 가능한 강력하고 확장 가능한 CTR 예측 솔루션을 제공합니다.

- **Performance Highlights**: Star+ 모델은 다양한 산업과 공공 데이터셋을 대상으로 한 광범위한 실험에서 예측 정확성과 효율성에 있어서 기존 모델들에 비해 상당한 개선을 보였습니다. 이는 멀티 도메인 환경에서 공유된 지식과 도메인별 특징을 잘 결합함으로써 이루어진 성과입니다.



### Cross-domain Transfer of Valence Preferences via a Meta-optimization Approach (https://arxiv.org/abs/2406.16494)
- **What's New**: 기존의 도메인 간 추천 시스템의 한계를 극복하기 위해 새로운 접근 방식인 CVPM(Cross-domain transfer of Valence Preferences via a Meta-optimization approach)을 제안합니다. CVPM은 매개변수 기반 메타 러닝(meta-learning)과 자가 지도 학습(self-supervised learning)을 결합한 하이브리드 아키텍처(hybrid architecture)로, 사용자의 긍정적 및 부정적 선호(positive and negative preferences)를 세분화하여 사용자의 개별적 선호를 더 잘 반영합니다. 또한, 겹치는 사용자가 적은 경우에도 신호 증강(signal enhancement)을 가능하게 합니다.

- **Technical Details**: CVPM은 사용자 선호의 세분화를 위해 긍정적 선호와 부정적 선호의 두 가지 분포를 학습하고, 다양한 모델 왜곡(model skew)과 패턴 손실을 방지하기 위해 그룹 및 개별 수준에서 대조 학습(contrastive tasks)을 추가로 설계합니다. 메타 러닝을 통해 각 사용자의 맞춤형 바이어스를 생성하여 사용자 고유의 행동을 고려한 개인 맞춤형 추천(personalized recommendation)을 제공합니다. 이와 함께, 사전 훈련된 모델(pre-trained model)과 아이템의 인기도(item popularity)를 이용해 의사 상호작용 항목(pseudo-interaction items)을 샘플링합니다.

- **Performance Highlights**: CVPM은 5개의 도메인 간 추천 과제와 1개의 시스템 간 추천 과제를 8개의 데이터 셋에서 테스트하였으며, 추운 시작(cold-start)과 따뜻한 시작(warm-start) 시나리오에서 모두 높은 성능을 보였습니다. 실험 결과, CVPM은 사용자의 표현 방식의 의미론적 향상과 추천 성능을 크게 개선하였습니다. 또한, 반복 가능성과 비교 가능성을 확보하기 위해 샘플 데이터 셋과 소스 코드를 GitHub에 공개하였습니다.



### Context-augmented Retrieval: A Novel Framework for Fast Information Retrieval based Response Generation using Large Language Mod (https://arxiv.org/abs/2406.16383)
- **What's New**: 이번 논문에서는 거대한 데이터베이스에서 정보를 신속하게 검색하고, 검색된 정보의 관련성을 보장하는 새로운 접근법인 CAR(Context Augmented Retrieval)을 제안합니다. CAR은 전통적인 텍스트 분류와 대형 언어 모델(LLM)을 결합하여 벡터 저장소에서 관련 정보를 빠르게 검색하고, 이를 통해 고품질 답변 생성을 가능하게 합니다.

- **Technical Details**: CAR 워크플로우는 다음과 같은 주요 단계를 포함합니다: 1) 사용자 쿼리 수신 및 분류, 2) 도메인별 라벨이 지정된 인덱스 로드, 3) BM25와 벡터 검색기를 결합한 하이브리드 검색, 4) LLM을 사용한 최종 응답 생성. 이러한 과정에서 DistilBERT 모델을 활용하여 더 효율적이고 가벼운 성능을 구현합니다.

- **Performance Highlights**: 실험 결과, CAR은 정보 검색 및 응답 생성 시간을 상당히 줄이면서도, 고품질의 응답을 일관되게 생성하는 데 성공했습니다. 이는 대규모 지식 도메인에서도 효과적으로 작동합니다.



### On the Role of Long-tail Knowledge in Retrieval Augmented Large Language Models (https://arxiv.org/abs/2406.16367)
- **What's New**: 이 논문은 Retrieval Augmented Generation (RAG)을 개선하기 위해 'long-tail knowledge' 탐지 방법을 제안합니다. 저자는 기존 RAG 시스템이 일반적인 지식에 대한 불필요한 컴퓨팅을 수행한다고 지적하면서, Long-tail knowledge가 중요하다는 것을 강조합니다. 이를 위해 Generative Expected Calibration Error (GECE)라는 새로운 지표를 도입했습니다.

- **Technical Details**: GECE는 기존의 Expected Calibration Error (ECE)를 확장하여 텍스트 생성 시나리오에도 적용할 수 있도록 디자인되었습니다. METEOR 점수와 LLM 출력 확률을 활용하여 'long-tailness'를 측정합니다. 추가로, 평균 단어 빈도와 특정 인스턴스의 그래디언트가 전체 데이터셋의 평균 그래디언트와 얼마나 다른지를 평가하는 도트 프로덕트를 결합하여 GECE 지표를 구성합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기존 RAG 파이프라인과 비교했을 때 평균 추론 시간에서 4배 이상의 속도 향상을 보였으며, downstream tasks에서도 일관된 성능 향상을 나타냈습니다.



### A Survey on Intent-aware Recommender Systems (https://arxiv.org/abs/2406.16350)
- **What's New**: 개인화 추천 시스템은 사용자의 특정 시점에서의 서비스 이용 의도를 반영해야 효과적이라는 최신 연구 동향을 중점적으로 다루고 있습니다. 이 논문은 Intent-Aware Recommender Systems (IARS) 구축을 위한 다양한 기술적 접근 방식을 조사하고 분류하였습니다. 또한, 현재 평가 관행을 분석하여 향후 연구 방향을 제시하며, 특히 상호작용 신호 및 맥락 정보를 추가로 고려할 필요성을 강조합니다.

- **Technical Details**: 기술적 접근 방식으로는 다양화 기법, 의도 예측 모델(intent prediction models), 잠재적 의도 모델링(latent intent modeling) 등이 포함됩니다. 이 논문은 최근 연구들에서 제안된 이러한 접근 방식을 체계적으로 분류하고 각각의 장단점을 논의합니다. 특히, IARS의 구축을 위한 다양한 데이터 유형, 평가 방법 및 응용 도메인을 중점적으로 살펴보고 있습니다.

- **Performance Highlights**: 산업 응용 분야에서 보고된 바에 따르면, 사용자의 예측된 의도에 맞춘 추천 시스템이 전환율(conversion rate) 등 주요 비즈니스 성과 지표를 크게 향상시키는 것으로 나타났습니다. 예를 들어, 전자 상거래 도메인에서는 동적인 전략 전환을 통해 구매율을 향상시켰으며, 음악 스트리밍 분야에서는 사용자의 잠재적 의도를 반영한 추천이 사용자 만족도를 예측하는 데 중요한 요소로 작용함을 확인했습니다.



### DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task (https://arxiv.org/abs/2406.16332)
- **What's New**: 최근 대형 언어 모델(LLMs)을 제로샷 패시지 순위 산정(passage ranking)에 적용하는 데 대한 관심이 증가하고 있습니다. 그러나 적절한 맥락 내 시연(in-context demonstrations)을 어떻게 선택할 것인지에 대한 연구는 상대적으로 적었습니다. 이 논문은 이러한 시연 선택 문제를 '검색 후 재순위화(retrieve-then-rerank)' 프로세스로 정의하고, 새로운 프레임워크 DemoRank를 제안합니다.

- **Technical Details**: DemoRank 프레임워크는 두 단계로 구성됩니다. 먼저, LLM의 피드백을 사용하여 시연 검색기(demonstration retriever)를 훈련합니다. 그런 다음, 시연 간 종속성을 고려한 새로운 종속 인식(dependency-aware) 시연 재랭커(demonstration reranker)를 도입하여 검색된 시연을 재정렬합니다. 이는 단순히 각 시연의 개별 피드백을 사용하는 것보다 더 나은 성능을 발휘하도록 설계되었습니다. 특히, 데모랭크의 훈련 샘플을 효율적으로 구성하여 시간 소모를 줄이면서도 종속성을 반영합니다.

- **Performance Highlights**: 포괄적인 실험 결과 DemoRank는 도메인 내(in-domain) 시나리오에서의 효과를 입증했으며, 도메인 외(out-of-domain) 시나리오에서도 강력한 일반화를 보였습니다. 더불어, 제한된 훈련 데이터, 다양한 시연 수, 미지의 데이터셋 등 여러 상황에서도 성능 향상을 확인했습니다. DemoRank는 특히 몇 샷 인콘텍스트 학습(few-shot ICL)에서 우수한 성능을 보였습니다.



### SimCE: Simplifying Cross-Entropy Loss for Collaborative Filtering (https://arxiv.org/abs/2406.16170)
- **What's New**: 새로 제안된 논문은 SimCE (Simplified Sampled Softmax Cross-Entropy)라는 새로운 손실 함수(loss function)를 소개합니다. 이는 기존의 Sampled Softmax Cross-Entropy (SSM)를 단순화하여 상한(upper bound) 최적화를 통합한 것입니다.

- **Technical Details**: 기존의 Bayesian Personalized Ranking (BPR) 손실 함수는 각 긍정 항목에 대해 하나의 부정 항목만 고려해 느린 수렴과 서브 최적(local optima) 문제를 겪습니다. SSM은 하나의 긍정 샘플과 여러 부정 샘플을 비교하여 더 나은 성능을 보입니다. SimCE는 SSM의 상한을 사용해 최적화 문제를 단순화하고, 이를 통해 더 효율적이고 확장 가능한 솔루션을 제안합니다.

- **Performance Highlights**: 12개의 벤치마크 데이터셋에서 MF와 LightGCN 백본을 사용한 종합 실험 결과, SimCE는 BPR과 SSM를 성능 면에서 일관되게 능가했습니다. 96개의 실증적 비교 중 93개에서 SimCE가 우수했으며, 최대 68.72%의 성능 향상을 달성했습니다.



### Evaluating Ensemble Methods for News Recommender Systems (https://arxiv.org/abs/2406.16106)
- **What's New**: 이 논문은 뉴스 추천 시스템 (NRS)에서 여러 최신 알고리즘을 결합하여 더욱 우수한 결과를 도출할 수 있는 합체 방법(ensemble methods)에 대해 다룹니다. Microsoft 뉴스 데이터셋(MIND)을 사용하여 다양한 알고리즘을 결합하면 개별 알고리즘보다 최대 5% 향상된 성능을 확인했습니다.

- **Technical Details**: 논문에서는 콘텐츠 기반 BERT 접근법과 협업 필터링 LSTUR 알고리즘을 포함한 여러 기본 학습 알고리즘(base learners)을 설명하였습니다. 이러한 알고리즘들은 MIND 데이터셋에서 다양한 기사와 사용자 상호작용을 분석하여 개인화된 뉴스를 추천합니다. 이 논문은 다양한 알고리즘의 출력 값을 조합하여 최적의 성능을 도출하는 랭크 집계 방법(rank aggregation methods)을 사용했습니다.

- **Performance Highlights**: 결과적으로, 서로 다른 기법의 알고리즘을 결합했을 때 개별 알고리즘보다 최대 5%까지 성능이 향상됨을 실험을 통해 입증하였습니다. 특히, 콘텐츠 기반 및 협업 필터링 방법이 충분히 다양할 때 이러한 개선이 두드러졌으며, 유사한 방법을 결합할 경우에는 성능 개선이 없었습니다.



### Evaluating D-MERIT of Partial-annotation on Information Retrieva (https://arxiv.org/abs/2406.16048)
Comments:
          Our dataset can be downloaded from this https URL

- **What's New**: 이번 연구에서는 부분적으로 주석된 데이터 세트로 인해 랭킹이 왜곡될 수 있음을 보여주고, Wikipedia에서 수집한 D-MERIT라는 새로운 평가 세트를 제안합니다. 이 세트는 각 쿼리에 대해 모든 관련 구절을 포함하려는 목표를 가지고 만들어졌습니다.

- **Technical Details**: 이번 연구에서는 대규모 코퍼스에서 일부 구절만을 주석하는 기존 방법론의 한계를 충돌합니다. D-MERIT는 '쿼리가 그룹을 설명'하고, '관련 구절이 해당 엔터티가 해당 그룹에 속한다는 증거'를 제공하는 형태입니다. 예를 들어, '언어학에 관한 저널'이라는 쿼리와 관련된 구절을 찾는 방식입니다. 이 세트는 각 쿼리에 대해 모든 관련 구절을 포함하려 하고, 자동 필터링 단계를 통해 후보 구절을 수집합니다.

- **Performance Highlights**: 데이터 세트가 부분적으로 주석된 상태에서 평가를 수행할 경우 시스템의 랭킹이 잘못될 수 있음을 발견했습니다. 그러나 시스템의 성능 차이가 유의미할 경우(single-relevant setup), 부분 주석된 데이터 세트도 정확한 랭킹을 제공할 수 있습니다. 더 많은 관련 구절을 포함시키면 평가의 신뢰도를 높일 수 있음을 확인했습니다.



### Learning k-Determinantal Point Processes for Personalized Ranking (https://arxiv.org/abs/2406.15983)
Comments:
          14 pages, accepted at ICDE 2024 (40th IEEE International Conference on Data Engineering)

- **What's New**: 사용자 선호도를 모델링하여 상품 추천 시스템의 개인화된 순위를 예측하는 새로운 최적화 기준 LkP을 제안합니다. 기존의 순위 최적화 방법들이 여러 항목 간의 상관관계를 충분히 활용하지 못하고 다양성 측면에서 부족함을 지적하며, 이를 극복하기 위해 DPP(Determinantal Point Process) 커널 분해를 통한 집합 확률 비교를 활용합니다.

- **Technical Details**: 이 논문은 DPP를 k-DPP로 확장하여 집합 수준의 관련성과 다양성 순위 비교를 공식화하였습니다. k-DPP는 DPP 분포 집합의 기수 k를 조건으로 하는 확장 입니다. 또한, 일반적인 확률적 경사 하강법(stochastic gradient descent)을 사용하여 LkP을 최적화하며, 이를 Matrix Factorization(MF) 및 신경망 접근 방식에도 적용하였습니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에서 LkP을 구현하여 관련성(relevance)과 다양성(diversity) 측면에서 성능을 개선하였습니다. 기존의 추천 모델에 적용했을 때도 강력한 성능 향상을 보여, 추천 시스템 분야에 중요한 가치를 지닌다는 것을 시사했습니다.



### LLM-Powered Explanations: Unraveling Recommendations Through Subgraph Reasoning (https://arxiv.org/abs/2406.15859)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)과 지식 그래프(Knowledge Graphs, KGs)를 결합한 새로운 추천 시스템을 소개합니다. 기존의 추천 시스템이 설명 가능성을 제공하지 못하는 한계를 극복하기 위해, LLMs을 활용해 사용자 리뷰를 새로운 트리플로 분해하여 KG를 풍부하게 하고, 이를 통해 추천 결과의 설명 가능성을 높입니다.

- **Technical Details**: LLM-SRR(Large Language Model powered Subgraph Reasoning Recommender)라는 새로운 아키텍처를 제안합니다. 이 프레임워크는 세 단계로 이루어집니다: 1) LLM을 이용한 정보 추출 및 KG 재구축, 2) 주의 메커니즘을 적용한 하위 그래프 추론(Subgraph Reasoning), 3) LLM을 통한 설명 생성입니다. 이 방법은 추천 시스템의 정확도와 투명성을 동시에 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 본 접근법은 네 개의 실세계 데이터셋에서 테스트되었으며, 기존 최첨단 방법들에 비해 평균 12%의 성능 향상을 보였습니다. 또한, 이 모델은 다국적 엔지니어링 및 기술 회사(METC)의 교차 판매(cross-selling) 추천 시스템에 적용되어 실용성을 입증하였습니다.



### FIRST: Faster Improved Listwise Reranking with Single Token Decoding (https://arxiv.org/abs/2406.15657)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 기존의 대형 언어 모델(LLM)을 활용한 리스트식(Liswise) 재랭킹의 비효율성을 해결하기 위해 FIRST라는 새로운 접근 방식을 도입했습니다. 기존의 방식은 후보 문단 식별자(passage identifier)를 생성하여 전체 순서를 얻는 데 시간이 많이 소요되었습니다. FIRST는 첫 번째 생성된 식별자의 로짓(logits)을 활용하여 직접 리스트식 재랭킹을 수행하며, 학습 과정에서 랭킹 손실(loss)을 사용해 잘못된 순위 평가를 최소화합니다.

- **Technical Details**: FIRST는 리스트식 재랭킹을 위해 대형 언어 모델의 첫 번째 생성된 식별자의 로짓만을 사용합니다. 일반적인 언어 모델링 목표 대신, 랭킹 정확성을 높이기 위해 학습-투-랭크(learning-to-rank) 손실을 포함시킵니다. 이러한 접근 방식은 후보 문단들에 대해 전반적인 중요도를 보다 빠르게 평가할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, FIRST는 추론 속도를 50% 향상시키면서도 높은 랭킹 성능을 유지했습니다. BEIR 벤치마크에서의 성능 향상을 기록했으며, 추론 시점에서의 관련성 피드백을 위한 재랭커로서 뛰어난 성능을 입증했습니다. 이는 크로스 인코더(cross-encoder)보다 우수한 재현율을 제공합니다.



### A Mechanism for Optimizing Media Recommender Systems (https://arxiv.org/abs/2406.16212)
Comments:
          Main Paper: 20 pages, Appendix with proofs and additional material: 26 pages

- **What's New**: 본 논문에서는 미디어 제작자와 소비자 간의 근본적인 상충 관계를 해결하는 메커니즘을 제안합니다. 특히 미디어 제작자는 도달 범위를 확대하고자 하는 반면, 소비자는 제공 받는 유틸리티(utility) 비율에 따라 주의를 집중합니다. 과도한 도달(overreach)은 이 비율에 부정적인 영향을 미치므로, 이를 고려한 비용 함수(cost function)를 통해 내용 배포의 최적 분배를 도출해 개별 소비자의 유틸리티와 참여를 극대화합니다.

- **Technical Details**: 이 메커니즘은 미디어 소스가 과도한 도달의 영향을 비용 함수에 포함시켜 최적의 분배를 결정함으로써, 각 소비자의 유틸리티(utility)를 극대화하고 참여를 증대시키는 방식을 취합니다. 이러한 접근은 제작자와 소비자 간의 Nash 균형(Nash equilibrium) 상태를 만들며, 동시에 Pareto 효율성(Pareto efficient)을 달성합니다. 또한, 추천 시스템(Recommender systems) 문헌과 비교해보면, 제안된 메커니즘은 소비자에게 최적의 콘텐츠 양을 도출하고, 여러 목표를 처리하는 방법을 개선할 수 있는 능력을 제시합니다.

- **Performance Highlights**: 제안된 메커니즘은 제작자와 소비자 모두에게 유익한 Nash 균형(Nash equilibrium)을 달성하며, 이는 또한 Pareto 효율성(Pareto efficient)을 보장합니다. 이로써 소비자의 유틸리티와 참여가 극대화되며, 추천 시스템(Recommender systems) 문헌의 한계를 초월합니다. 실용적인 알고리즘을 통해 각 소비자에게 최적 분배를 생성할 수 있습니다.



### Database-Augmented Query Representation for Information Retrieva (https://arxiv.org/abs/2406.16013)
- **What's New**: 새로운 정보 검색(Retrieval) 모델이 등장했습니다. 사용자가 제공하는 간단한 쿼리(query) 대신, 이 모델은 관계형 데이터베이스(Relational Database)의 다양한 메타데이터를 활용하여 쿼리를 확장합니다. 이 모델의 이름은 DAQu(Database-Augmented Query)입니다.

- **Technical Details**: DAQu는 원본 쿼리를 관계형 데이터베이스의 다양한 테이블에 있는 메타데이터(metadata)를 통해 확장합니다. 이러한 메타데이터는 특징(feature)들이 많고 순서가 없기 때문에, 그래프 기반의 집합 인코딩 전략(Graph-based Set Encoding Strategy)을 사용하여 특징들의 위계(hierarchies)를 고려해 인코딩합니다.

- **Performance Highlights**: DAQu는 다양한 메타데이터를 활용할 수 있는 검색 시나리오에서 기존의 쿼리 확장 방법들보다 월등히 높은 검색 성능을 보여줍니다.



### Food Pairing Unveiled: Exploring Recipe Creation Dynamics through Recommender Systems (https://arxiv.org/abs/2406.15533)
- **What's New**: 최근 연구는 Heston Blumenthal의 'food pairing' 가설을 GPT-3과 같은 최신 협업 필터링(Collaborative Filtering) 기술에 적용하여 레시피에 새로운 재료를 추천하고, 누락된 재료를 찾아내며, 특정 조합을 피하도록 도울 수 있는 도구를 개발했습니다.

- **Technical Details**: 이 연구는 기존 Ahn et al.의 연구 데이터를 활용하여 레시피 기반 모델과 맛 화합물(Flavor Compounds) 기반 모델을 비교했습니다. 아이템-아이템 협업 필터링(Item-Item Collaborative Filtering) 방식을 사용하여, 유사한 재료를 추천하는 시스템을 구성했습니다. LightGCN과 같은 Graph Convolutional Network도 도입되어 기존 유사도 기반 시스템과 비교되었습니다.

- **Performance Highlights**: 레시피 기반 모델이 맛 화합물 기반 모델보다 주로 높은 성능을 보였으나, 맛 화합물 기반 모델은 북미 요리에서 더 큰 효과를 보였습니다. LightGCN은 높은 정확도를 보였지만, 해석 가능성(Interpretability)이 떨어지는 단점이 있었습니다. 유사한 재료 간의 단순한 짝짓기(예: 'bell pepper'와 'green bell pepper')가 주로 food pairing을 성공적으로 이끄는 이유로 나타났으며, 이는 이 기법의 기초 원리를 재고하게 합니다.



