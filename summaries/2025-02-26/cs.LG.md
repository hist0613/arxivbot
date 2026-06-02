New uploads on arXiv(cs.CL)

### DRAMA: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers (https://arxiv.org/abs/2502.18460)
- **What's New**: 이번 연구는 DRAMA라는 훈련 프레임워크를 도입하여 대형 언어 모델(LLMs)을 사용하여 더 작고 일반화 가능한 조밀한 검색기를 생성하는 방법을 제안합니다. 기존의 상태에서도 조밀한 검색기는 대규모의 쿼리 처리에 대한 높은 컴퓨팅 비용과 지연 시간이 문제로 지적되었습니다. DRAMA는 LLM 기반 데이터 증강을 활용하여 훈련 데이터의 질을 높이는 동시에 조밀 검색기 성능을 극대화할 수 있습니다.

- **Technical Details**: DRAMA는 약 10억 개 미만의 매개변수를 가진 조밀한 검색기를 효과적으로 만들기 위해 LLM의 축소된 버전을 백본으로 사용합니다. 연구에서는 Llama3 모델을 기반으로 하여 다양한 LLM 증강 데이터를 단일 단계 대조 학습(setup of contrastive learning)에서 훈련합니다. 이러한 접근을 통해 다양한 언어와 긴 컨텍스트에 대한 우수한 능력을 발휘하며, 기존 인코더 기반 검색기의 성능을 넘어서는 결과를 얻었습니다.

- **Performance Highlights**: 실험 결과 DRAMA는 BEIR, MIRACL과 여러 다국어 검색 작업에서 높은 성능을 달성하며, 조밀한 검색기의 일반화 능력을 증명하였습니다. 또한, 이 프레임워크는 대형 언어 모델의 발전과 조화를 이루어 효율성과 일반화 간의 격차를 메워주는 기술적 가능성을 보여줍니다. 이 연구는 다양한 검색 작업에서 일관되게 우수한 성능을 보이는 소형 검색기의 잠재력을 강조합니다.



### FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Respons (https://arxiv.org/abs/2502.18452)
Comments:
          8 pages, 3 figures, 5 tables

- **What's New**: 대규모 언어 모델(LLMs)은 일반적인 추론 성능이 향상되고 있으나, 작은 모델들은 특정 추론 작업에서 상대적으로 낮은 성능을 보입니다. 이를 해결하기 위해, 저자들은 재난 분야에 특화된 작은 LLMs을 미세 조정(fine-tune)하여 정밀한 데이터 검색이 가능한 FRIDA(Field Ready Instruction Decoding Agent) 모델을 개발했습니다. FRIDA는 도메인 전문가와 언어학자의 지식을 활용하여 고품질의 합성 데이터를 생성하는 파이프라인을 갖추고 있습니다.

- **Technical Details**: FRIDA 모델은 130개의 시드 지침과 25,000개의 합성 명령문, 그리고 지진 관련 119개의 평가 지침을 모았으며, LLaMa와 Mistral 모델을 사용해 미세 조정하였습니다. 이 연구에서는 합성 데이터의 영향을 분석하기 위해 ablation 연구를 진행하였고, 물리적 상태와 객체 기능의 일반적인 상식 훈련이 FRIDA 모델보다 더 효과적임을 확인했습니다. 세부적으로, 재난 구호 작업의 특정 지식 없이는 성능 향상에 한계가 있음을 시사했습니다.

- **Performance Highlights**: FRIDA 모델들은 기본 모델들보다 전반적으로 뛰어난 성능을 보였으며, 특히 모델 크기나 아키텍처에 관계없이 성공적으로 일반 상식을 개선하였습니다. aFRIDA 모델은 일반적인 상식에 기반한 소규모 LLM로 실험하였고, 도메인별 특정 상식 지식에 비해 더 우수한 성과를 거두었습니다. 따라서 FRIDA 파이프라인은 전반적인 상식을 제공할 수 있으며, 추가적인 정보 검색이 도메인 지식을 충족시키는데 필요함을 결론지었습니다.



### Disambiguate First Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing (https://arxiv.org/abs/2502.18448)
- **What's New**: 본 논문에서는 ambiguity(모호성)와 underspecification(불완전 명세)를 처리하는 모듈식 접근 방식을 제안합니다. 이 방법은 자연 언어 해석을 통해 모호성을 해결한 후 이를 SQL 쿼리와 같은 논리적 형태로 매핑합니다. 기존의 LLM(대형 언어 모델)은 명확한 발화를 처리하는 데 뛰어나지만, 모호한 발화에 대해서는 편향된 결과를 보이는 것을 이용하여 초기 선호 해석 집합을 생성하고, 이후 specialized infilling model(특수화된 보간 모델)을 적용해 부족한 해석을 찾아냅니다.

- **Technical Details**: 모호한 질문에 대한 text-to-SQL parsing(텍스트-투-SQL 파싱)에 초점을 맞춰 두 단계 접근 방식인 "disambiguate first, parse later"(먼저 모호성을 해소하고, 그 후 파싱) 방식을 채택합니다. 이 접근 방식에서는 먼저 LLM을 통해 발화의 모든 가능한 의미를 생성한 후, 이를 바탕으로 unambiguous(명확한) 해석을 제공합니다. 또한, synthetic reference interpretations(합성 참조 해석)을 통해 AmbiQT 벤치마크에서의 학습 데이터 생성을 지원하며, SQL 쿼리를 실행하여 해석의 정확성을 검증합니다.

- **Performance Highlights**: 수행된 실험에서는 제안된 접근 방식이 모호한 질문에 대한 해석의 범위를 개선하고 다양한 데이터 세트 전반에서 일반화되는 것을 보여줍니다. 이러한 새로운 접근 방식은 다양한 주석 스타일, 데이터베이스 구조 및 모호성 유형에 대해 효과적으로 작동하며, 이러한 결과는 사용자 신뢰를 높이고 대화 시스템의 실질적인 유용성을 증가시킵니다.



### olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models (https://arxiv.org/abs/2502.18443)
- **What's New**: 본 논문에서는 PDFs를 클린하고 직렬화된 텍스트로 변환하는 오픈 소스 Python 툴킷인 olmOCR을 소개합니다. 이 툴킷은 섹션, 표 및 리스트와 같은 구조화된 콘텐츠를 보존하면서, 시각적 레이아웃이 다양한 260,000페이지 이상의 PDF 샘플에서 학습된 7B 비전 언어 모델(VLM)을 사용합니다. 또한 대규모 배치 처리에 최적화되어 있어 저렴한 비용으로 PDFs를 처리할 수 있습니다.

- **Technical Details**: olmOCR은 다양한 전자 문서 포맷의 텍스트와 메타데이터를 추출할 수 있는 document-anchoring 기법을 활용합니다. 이 기법은 이미지와 텍스트 정보를 함께 사용하여 문서의 정확한 텍스트 표현을 얻습니다. 또한 OCR 도구로서, PDFs 내의 텍스트 블록과 이미지를 추출하고 VLM에 입력하여 품질을 향상시킵니다.

- **Performance Highlights**: olmOCR은 비용 효율성이 뛰어나며, 1백만 페이지를 변환하는 데 드는 비용은 단 $190입니다. 문서 구조에 내재된 메타데이터가 없는 경우에도 높은 성능을 유지합니다. 최근 테스트에서 이 툴킷은 GPU에서 효율적으로 확장 가능하다는 것을 입증했습니다.



### Reversal Blessing: Thinking Backward May Outpace Thinking Forward in Multi-choice Questions (https://arxiv.org/abs/2502.18435)
- **What's New**: 이 논문에서는 일반적으로 사용되는 left-to-right (L2R) 자율 회귀(autoregressive) 모델 대신에 right-to-left (R2L) 방식의 효과를 조사합니다. R2L 모형은 MCQ(다중 선택 질문)와 지식 추출, 추론 작업에서 L2R을 능가하는 성과를 보여주며, 다양한 기준에서 다수의 실험을 통해 그 결과를 입증합니다. 또한, 최적의 텍스트 분포 인코딩 방법에 대한 이론적 통찰을 제공합니다.

- **Technical Details**: 이 연구는 L2R과 R2L 모델을 비교하기 위해 동일한 데이터와 컴퓨팅 자원으로 훈련된 모델을 사용합니다. R2L 방식은 예측 손실이 L2R와 유사한 결과를 도출할 수 있는 대칭성(symmetric) 구조를 가지고 있으며, 이로 인해 R2L은 더 적은 근사 오류(approximation errors)를 달성할 수 있습니다. 연구진은 R2L의 성과 차이를 결정하는 여러 요소들—캘리브레이션(calibration), 계산 가능성(computability), 방향 조건 엔트로피(direction conditional entropy)—이 어떻게 상관되어 있는지를 분석합니다.

- **Performance Highlights**: R2L 모델은 여러 MCQ 기준에서 L2R 모델 보다 일관되게 우수한 성능을 기록했습니다. 이 연구의 실험 결과에 따르면, R2L의 접근 방식이 특정 상황에서 더 나은 성과를 내는 이유를 규명하기 위해 다양한 요인들을 배제하고, 어떤 이유로 L2R과 R2L 선택이 결정되는지를 규명하고자 합니다. 결과적으로 이 연구는 LLM(대형 언어 모델)의 능력을 향상시키는 잠재력 있는 방향성을 제시합니다.



### Exploring Gender Disparities in Automatic Speech Recognition Technology (https://arxiv.org/abs/2502.18434)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 연구는 성별을 넘어 자동 음성 인식(ASR) 시스템의 공정성과 성능에 영향을 미치는 요인들을 조사합니다. 기존의 연구에서 다루어진 인구 통계학적 요소를 넘어, 훈련 데이터의 성별 비율 변화가 ASR 성능에 미치는 영향을 분석합니다. 연구 결과, 최적의 공정성은 단순한 성비(50-50 스플릿)에서가 아니라 특정한 성별 분포에서 발생한다는 것을 발견했습니다.

- **Technical Details**: 연구는 LibriSpeech 데이터셋과 Whisper 소형 모델을 사용하여 ASR 성능에 대한 다양한 성별 분포, 발화 내용의 가독성(읽기 난이도), 의미적 유사성 및 음역대 분포의 영향을 분석합니다. 총 11개의 훈련 부분집합을 만들어 훈련 데이터 세트에서 여성의 비율을 0%에서 100%까지 점진적으로 변화시키며 ASR 성능 향상을 관찰했습니다.

- **Performance Highlights**: 결과적으로워드 오류율(Word Error Rate, WER)을 계산하여 훈련 세트의 성별 분포가 ASR 모델 성능에 미치는 영향을 평가했습니다. 특히, 여성 비율 증가가 성별 간의 성능 격차에 미치는 효과를 분석하며 성별 비율 외에도 텍스트의 난이도, 의미적 유사성 및 음역대의 다양성이 ASR 정확도에 상당한 영향을 미침을 시사했습니다.



### TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning (https://arxiv.org/abs/2502.18431)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 논리적 추론 능력을 평가할 수 있도록 설계된 새로운 벤치마크인 TextGames를 소개합니다. TextGames는 패턴 인식, 공간 인식, 산술 및 논리적 사고와 같은 고급 기술을 요구하는 텍스트 기반 게임을 통해 LLMs의 성능을 측정합니다. 이 연구는 단일 턴(single-turn) 및 다중 턴(multi-turn) 추론에서의 LLMs의 성능과 그들이 피드백을 활용하여 자기 교정을 통해 후속 답변을 수정할 수 있는 능력을 분석합니다. LLMs는 대부분의 간단한 문제와 중간 수준의 문제에 대해서는 능숙하지만, 더욱 어려운 과제에서는 큰 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: TextGames 벤치마크는 총 8개의 퍼즐 게임으로 구성되며, 각 게임은 세 가지 난이도 수준을 가지고 있습니다. 게임의 규칙은 매우 복잡하여 LLMs가 세부 지침을 따르는 능력을 평가하는 데 용이합니다. 또한, LLMs가 피드백을 받을 때 이전 생성물에 대해 자기 반성을 통해 오류를 수정할 수 있는지를 조사합니다. 이 연구에서는 GPT-o3 Mini와 같은 추론 최적화 모델이 사전 훈련된 LLMs보다 어려운 과제에서 더욱 강력한 성능을 보임을 발견했습니다.

- **Performance Highlights**: 연구 결과, LLMs는 다중 턴 상호작용에서 피드백을 받았을 때 성능이 향상되고 이전 생성물에 대해 자기 반성할 수 있음을 확인했습니다. 그러나 LLMs는 여전히 시퀀싱, 카운팅, 복잡한 규칙의 일관된 적용에서 어려움을 겪고 있습니다. 특히, TextGames는 고급 추론 능력을 요구하므로, LLMs는 이러한 복잡한 문제 해결에 있어 충분한 능력을 갖추지 못한 것으로 보입니다. човеку, 반면 인간은 주어진 시간 내에 모든 문제를 해결할 수 있는 능력을 지니고 있습니다.



### Compressing Language Models for Specialized Domains (https://arxiv.org/abs/2502.18424)
Comments:
          Work in progress

- **What's New**:  본 연구에서는 압축된 언어 모델의 도메인 성능 향상을 위해 `cross-calibration`이라는 새로운 비학습(calibration) 방법을 제안합니다. 기존의 일반적인 모델 압축 방법은 이러한 도메인 간 성능 저하 문제를 해결하기 위해 전 매개변수(full-parameter) 미세 조정(fine-tuning) 과정이 필요했지만, 이는 상당한 계산 비용이 소모됩니다. 그러나 제안된 방법은 Hessian 기반의 민감도 분석을 통해 중요한 가중치를 식별하여 도메인별 성능을 최대화하면서도 일반 성능에 영향을 주지 않습니다.

- **Technical Details**: 압축된 언어 모델은 주로 `quantization`과 `pruning`과 같은 기술을 통해 효율성을 추구하는데, 본 연구에서는 Hessian 매트릭스를 활용하여 가중치의 중요성을 평가합니다. `cross-calibration` 방법은 도메인 특화 데이터와 일반 성능 간의 균형을 맞추어, 다양한 응용 분야에 걸쳐 성능을 극대화할 수 있는 접근법입니다. 이로 인해 모델 컴프레션의 계산 부하를 최소화하면서도 пост-training 환경에서 더욱 효과적인 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, `cross-calibration` 방법은 기존의 도메인 특화 및 일반 프루닝 방법들과 비교시 현저한 성능 향상을 보여주었습니다. 특히, 일반 성능을 희생하지 않고도 도메인 특정 작업에서 성능이 크게 증대되는 것을 확인할 수 있었습니다. 이 연구는 향후 다양한 도메인에서 압축된 전문 모델을 효과적으로 활용할 수 있는 잠재력이 있음을 보여줍니다.



### GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback (https://arxiv.org/abs/2502.18414)
- **What's New**: 이번 연구에서는 GLEAN을 제안하여 Generalized Category Discovery (GCD)의 기존 문제를 해결하고자 합니다. GLEAN은 다양한 방식의 LLM 피드백을 활용하여 데이터 레이블링 과정에서의 효율성을 높이고, 새로운 카테고리의 발견 및 기여를 도모합니다. 이러한 접근법은 기존의 GCD 접근법에 비해 더 나은 성능을 제공합니다.

- **Technical Details**: GLEAN의 핵심 개념은 LLM의 피드백을 세 가지 방식으로 활용하는 것입니다. 첫째, Similar Instance Selection을 통해 모호한 데이터 포인트 간의 유사한 인스턴스를 식별합니다. 둘째, Category Characterization을 통해 새로운 카테고리에 대한 설명을 생성합니다. 셋째, Pseudo Category Selection 및 Alignment을 통해 인스턴스 임베딩을 LLM이 선택한 카테고리 설명과 연결합니다.

- **Performance Highlights**: 실험 결과, GLEAN은 다양한 데이터셋과 성능 지표에서 기존의 최신 모델보다 우수한 성능을 보여주었습니다. 또한, 알려진 카테고리의 수에 따라 GLEAN의 성능을 분석하여 각 구성 요소와 하이퍼파라미터의 효과를 자세히 살펴보았습니다. 이러한 결과는 GLEAN이 GCD 분야에서 중요한 기여를 할 수 있음을 시사합니다.



### AgentRM: Enhancing Agent Generalization with Reward Modeling (https://arxiv.org/abs/2502.18407)
- **What's New**: 본 연구에서는 기존 LLM 기반 에이전트의 일반화 능력을 개선하기 위해, 정책 모델을 직접 미세 조정하는 것보다 보상 모델(reward model)을 미세 조정하여 정책 모델을 안내하는 것이 더 효과적임을 발견했습니다. 이를 통해 AgentRM이라는 일반화 가능한 보상 모델을 제안합니다. AgentRM은 test-time search에서 정책 모델을 효과적으로 가이드하여, 미지의 작업에서도 성능을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 에이전트 작업은 부분 관찰 가능한 마르코프 결정 과정(partially observable Markov decision process, POMDP)로 정형화되며, 다양한 작업 환경에서의 피드백을 기반으로 합니다. 연구에서는 보상 모델을 구축하기 위해 세 가지 접근 방식(1) 명시적 보상 모델링(explicit reward modeling), (2) 암묵적 보상 모델링(implicit reward modeling), (3) LLM을 평가자로 활용하는 방식(LLM-as-a-judge)을 조사하였습니다. AgentRM은 Best-of-N 샘플링과 단계 수준의 빔 검색(beam search)을 통해 응답 생성을 안내합니다.

- **Performance Highlights**: AgentRM은 아홉 개의 다양한 에이전트 작업에서 평균 8.8점의 성능 향상을 이루었으며, 이는 기존의 최고의 일반화 에이전트보다 4.0점 더 높은 성과입니다. 특히 LLaMA-3-70B 정책 모델에서는 12.6점의 더 큰 개선을 보여주어, 강한 일반화 능력을 입증했습니다. 또한, 미세 조정된 정책 모델을 신속하게 증강하여 세 가지 고정된 작업에서 상위 전문화된 에이전트보다 11.4점 더 나은 성능을 달성했습니다.



### KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation (https://arxiv.org/abs/2502.18397)
- **What's New**: 이번 논문에서는 multi-hop 질문 응답(multi-hop question answering, QA)을 위한 새로운 모델인 KiRAG를 제안합니다. KiRAG는 지식 기반의 반복 검색 모델(knowledge-driven iterative retriever model)로, 기존의 iRAG 모델에서 발생하는 정보 수집의 두 가지 주요 문제를 해결합니다. 이는 비관련 문서나 잘못된 사고의 연쇄에 의한 방해 요소들을 극복하고, 다단계 추론(multi-step reasoning)에서 필요한 정보를 동적으로 검출하고 수집하는 능력을 갖추고 있습니다.

- **Technical Details**: KiRAG는 문서를 지식 삼중(knoledge triples)으로 분해하고, 이를 기반으로 반복 검색(iterative retrieval)을 수행하여 신뢰할 수 있는 정보 수집 프로세스를 구현합니다. 이 모델은 정보의 격차(gaps)를 메우기 위해 추론(reasoning)을 통합하여 더욱 효과적으로 필요한 지식을 찾아냅니다. 이는 정보의 필요성이 변화하는 상황에서도 유연하게 적응할 수 있도록 도와줍니다.

- **Performance Highlights**: 실험 결과, KiRAG는 기존의 iRAG 모델에 비해 평균 9.40%의 R@3 및 5.14%의 F1에서 상당한 성능 향상을 보였습니다. 이 결과는 KiRAG가 다단계 질문 응답에 있어 효과적인 솔루션임을 입증합니다. 이러한 성과는 정보 검색 과정에서 지식 삼중의 활용이 얼마나 중요한지를 보여줍니다.



### Monte Carlo Temperature: a robust sampling strategy for LLM's uncertainty quantification methods (https://arxiv.org/abs/2502.18389)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 신뢰할 수 있는 배포를 위해 불확실성 정량화(Uncertainty Quantification, UQ)의 중요성을 강조합니다. 기존 UQ 방법은 고정된 온도(temperature) 샘플링의 선택에 대한 영향을 충분히 연구하지 않았으며, 저자들은 Monte Carlo Temperature (MCT)라는 새로운 샘플링 전략을 제안하여 이러한 문제를 해결합니다.

- **Technical Details**: MCT는 여러 문장의 생성 과정에서 온도를 동적으로 변화시켜 UQ 방법의 효과성을 향상시키는 접근 방식입니다. 이는 하이퍼파라미터 최적화(hyperparameter optimization, HPO)의 필요성을 줄이고, 다양한 모델-데이터셋 조합에 일반화할 수 있도록 설계되었습니다. MCT는 고정된 온도 전략과 비교하여 더 신뢰할 수 있는 불확실성 추정값을 제공합니다.

- **Performance Highlights**: MCT는 오라클(oracle) 온도를 기준으로 평가될 때, 일관되게 통계적 동등성을 기록하며 HPO의 부담을 제거합니다. 또한, MCT는 Best On Average Temperature 및 Fixed Random Temperature 접근 방식을 초과 성능을 보이며, 구조화된 온도 샘플링의 이점을 강조합니다.



### DBR: Divergence-Based Regularization for Debiasing Natural Language Understanding Models (https://arxiv.org/abs/2502.18353)
Comments:
          Accepted by SIGKDD Explorations

- **What's New**: 이 논문은 사전 훈련된 언어 모델(PLMs)이 자연어 이해(NLU) 작업에서 단순한 특징과 단축 경로(shortcut) 학습에 의존하는 경향을 보인다는 사실을 조명합니다. 이로 인해 모델이 도메인 외부 데이터(تماد의 OOD) 일반화에 어려움을 겪는다는 점을 강조하고, 이를 해결하기 위해 편차 기반 정규화(Divergence Based Regularization, DBR) 방법을 제안합니다. DBR은 원래 예제와 단축 토큰을 블라인드 처리한 예제 간의 분포 차이를 측정하여 단축 특징에 대한 의존성을 줄이는 접근법입니다.

- **Technical Details**: DBR 방법은 먼저 단축 토큰을 마스킹하여 모델 예측이 단축 특징의 영향을 받지 않도록 하며, 이어서 원래의 예제와 비슷한 예제들을 만들기 위해 정규화 손실을 추가합니다. 이 과정에서 편향 전용 모델을 사용하여 어떤 예제가 실제로 단축 특징에 의존하는지를 판단합니다. 또한 이 모델은 매 에포크마다 다른 마스킹된 예제를 생성하여 모델의 견고성을 향상시킵니다.

- **Performance Highlights**: 세 가지 NLU 작업에서 DBR을 평가한 결과, OOD 성능이 향상되었으며, 도메인 내 정확도에 거의 손실을 주지 않았습니다. 실험 결과는 단축 및 피상적인 특징에 대한 의존성을 줄임으로써 대규모 사전 훈련된 언어 모델의 일반화 능력이 개선될 수 있음을 보여줍니다.



### BRIDO: Bringing Democratic Order to Abstractive Summarization (https://arxiv.org/abs/2502.18342)
Comments:
          13 pages, 1 figure; AAAI-25 Workshop on PDLM camera ready

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 텍스트 요약 과정에서 발생하는 환각(hallucination) 문제를 다룹니다. 특히, 기존의 모델인 BRIO의 노출 편향(exposure bias) 완화 전략을 기반으로 하면서도 환각을 줄이는 데 중점을 둔 새로운 모델 BRIDO를 제안합니다. BRIDO는 후보 요약들의 유사성을 집합적으로 비교하여 후보들을 정렬하는 민주적인 방식을 도입합니다. 또한, XSum과 CNN/DM 데이터셋에서의 실험 결과를 통해 유의미한 성과를 보여주었습니다.

- **Technical Details**: BRIDO는 후보 요약의 유사성에 따라 정렬된 순위를 바탕으로 환각을 완화하는 추상 요약 모델입니다. 모델은 대조학습(contrastive learning)을 통해 높은 상호 후보 ROUGE 점수를 가진 후보를 유도하여 작동합니다. BRIDO의 주요 기여는 후보들 간의 유사성을 고려하여 환각이 적은 후보를 선택하도록 하는 것입니다. 이 접근법은 BRIO의 기법을 확장하며, 대규모 LLM을 탈피한 모델에서 효과적으로 활용될 수 있습니다.

- **Performance Highlights**: BRIDO는 XSum 및 CNN/DM 데이터셋에서 각각 6.25% 및 3.82% 향상된 일관성 G-Eval 점수를 달성했습니다. 이는 BRIO와 비교할 때 환각 측정에서 상당한 개선을 보여주며, 기존 모델에 비해 더 나은 결과를 제공합니다. 이 연구는 환각 문제를 해결하기 위한 새로운 방향성을 제시하며, 대조학습을 통한 접근 방식의 유효성을 입증하고 있습니다.



### Moderation Matters:Measuring Conversational Moderation Impact in English as a Second Language Group Discussion (https://arxiv.org/abs/2502.18341)
- **What's New**: 이 논문은 영어를 제2 외국어로 사용하는(ESL) 화자들이 그룹 토론에 참여할 때의 언어 장벽을 극복하는 방법에 대한 새로운 연구를 제시합니다. 17개의 온라인 ESL 대화 클럽 세션으로 구성된 첫 번째 데이터셋을 개발했습니다. 이 데이터셋은 조정된 토론과 조정되지 않은 토론 모두를 포함하여, 토론자들이 어떻게 참여하며 대화를 나누는지 분석할 수 있는 기초 데이터를 제공합니다.

- **Technical Details**: 논문에서는 ESL 담화 평가의 자동화된 접근 방법을 도입하고, 조정 전략을 10가지 유형으로 분류하는 프레임워크를 사용하여 연구를 진행했습니다. 참가자들의 대화 품질에 대한 영향을 수량화하기 위해 대화 기록을 직접 분석하였고, 조정자의 개입이 다자간 대화의 동태에 미치는 영향을 조사했습니다. 특히, 조정자가 제공하는 적극적인 인정과 격려가 가장 효과적인 전략으로 발견되었습니다.

- **Performance Highlights**: 연구 결과는 조정자가 대화의 주제 흐름을 개선하고 대화의 시작과 종료를 효과적으로 도와주며, 그들의 개입이 토론의 품질에 긍정적인 영향을 미친다는 것을 보여주었습니다. 반면 지나치게 많은 정보나 의견 공유는 부정적인 영향을 미치는 것으로 나타났습니다. 따라서 이 연구는 비원어민 대화 환경에서 ESL 그룹 토론과 조정자의 역할을 분석하는 토대를 마련합니다.



### Correlating and Predicting Human Evaluations of Language Models from Natural Language Processing Benchmarks (https://arxiv.org/abs/2502.18339)
- **What's New**: 이번 연구는 고성능 대화형 언어 모델(Chat Llama 2)의 평가 전략 변화에 주목하여, 기존의 정량적 NLP 벤치마크와 인간의 평가 간의 관계를 탐구하였습니다. 연구진은 160개의 표준 NLP 벤치마크에서 4개의 모델의 성능과 11,000개 이상의 단일 및 다중 턴 대화에서의 인간 선호도를 비교하였습니다. 그 결과, 많은 NLP 벤치마크가 인간 평가와 강력하게 상관관계를 보이며, 자동화된 지표가 인간의 선호를 예측하는 데 신뢰할 수 있음을 보여주었습니다.

- **Technical Details**: 이 논문에서는 7억, 13억, 34억, 70억 개의 매개변수를 가진 4개의 Chat Llama 2 모델을 평가하였습니다. 연구진은 2조 토큰에 대해 사전 훈련된 모델들을 사용하였고, 슈퍼바이즈드 파인튜닝 및 인간 피드백을 통한 강화 학습으로 후처리하였습니다. 단일 턴 및 다중 턴 대화의 다양성을 고려하여 대화 구조를 설정하였으며, 최첨단 인공지능 모델의 성능 평가를 위한 데이터셋을 구축하였습니다.

- **Performance Highlights**: 모델 간의 성능 비교에서 인간 신뢰성 평가와 여러 NLP 벤치마크 간의 상관관계를 분석했습니다. 인간 평가와 NLP 벤치마크 간의 관계를 선형 회귀 모델을 사용하여 예측했으며, 이로 인해 고비용의 인간 주석 의존도를 줄일 수 있는 가능성을 보여주었습니다. 전반적으로 고전적 벤치마크의 가치를 확립하고, 이러한 지표를 통해 실제 사용자 만족도를 예측할 수 있는 방법을 제시했습니다.



### BottleHumor: Self-Informed Humor Explanation using the Information Bottleneck Princip (https://arxiv.org/abs/2502.18331)
- **What's New**: 본 논문에서는 멀티모달 유머 해석을 위한 새로운 방법인 BottleHumor를 소개합니다. 이 방법은 정보 병목 원칙(information bottleneck principle)을 기반으로 하여, 비전 및 언어 모델에서 유용한 세계 지식을 반추하여 유머의 설명을 생성합니다. 특히, 다양한 유형의 지식을 요구하는 이 등장은 유머의 주제를 깊이 이해하는 데 도움이 됩니다.

- **Technical Details**: BottleHumor는 이미지와 텍스트를 통합하여 유머를 설명하는 방법입니다. 이 방법은 여러 단계의 추론을 사용하여 각각의 이미지와 자막에 대해 유용한 암시(implication)를 생성하고 선택합니다. 이를 통해 최종 후보 설명을 생성하는 과정에서 정보 병목 원칙을 적용하여 불필요한 중복을 최소화합니다.

- **Performance Highlights**: 세 가지 멀티모달 유머 데이터셋에서 평가한 결과, BottleHumor는 이전 방식에 비해 F1 점수가 각각 최대 8.2, 4.3, 2.8 포인트 향상되었습니다. 이는 기존의 자가 수정 방법(self-refine)보다 월등히 뛰어난 성능을 보이며, 다양한 세계 지식을 복합적인 추론 작업에 통합하는 방식의 중요성을 강조합니다.



### Mapping of Subjective Accounts into Interpreted Clusters (MOSAIC): Topic Modelling and LLM applied to Stroboscopic Phenomenology (https://arxiv.org/abs/2502.18318)
- **What's New**: 이번 연구는 스토로보스코픽 조명 자극(Stroboscopic Light Stimulation, SLS)에 의해 유도된 경험을 깊이 이해하기 위해 데이터 기반 접근 방식을 활용했습니다. 이를 통해 자연어 처리(Natural Language Processing)와 주제 모델링(Topic Modelling)을 사용하여 422개의 주관적인 보고서에서 862개의 문장을 분석하였습니다. 연구 결과 간단한 시각적 환각(Visual Hallucinations) 외에도 의식의 변화와 복잡한 환각을 경험할 수 있음을 확인했습니다.

- **Technical Details**: 이번 연구에서 사용한 주제 모델링(Topic Modelling)은 문장에서 단어의 출현 패턴을 파악하여 숨겨진 주제를 식별하는 통계적 방법입니다. 이를 통해 기존의 정형화된 설문조사가 간과할 수 있는 다양한 경험의 차원을 탐색할 수 있었습니다. 연구팀은 MOSAIC이라는 개방형 소스 자연어 처리 파이프라인을 구현하였으며, 이는 주관적인 텍스트 데이터의 체계적 분석을 가능하게 합니다.

- **Performance Highlights**: 연구는 스토로보스코픽 조명 자극에 의해 생성된 다양한 주관적 경험을 보다 명확하게 이해하는 데 기여했습니다. 주제 모델링을 통해 발견된 은닉 주제들은 일관된 패턴으로 나타났으며, 기존의 연구에서 간과되었던 새로운 경험적 차원을 조명했습니다. 본 연구는 다양한 연구 분야에서 복잡한 텍스트 보고서를 체계적으로 분석하는 방법론을 제시하여 향후 연구에도 기여할 것으로 기대됩니다.



### WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging (https://arxiv.org/abs/2502.18316)
- **What's New**: WiCkeD는 기존 다지선다형 벤치마크의 복잡성을 증가시키는 간단한 방법으로, 선택지를 'Above none of the above'로 무작위로 교체하여 생성됩니다. 이 방법은 자동으로 기존의 벤치마크에 적용될 수 있으며, 여러 인기 있는 벤치마크에 적용되어 LLM의 성능을 평가합니다. WiCkeD는 모델의 평균 성능이 12.1 포인트 떨어지도록 하여 새로운 도전 과제를 제공합니다.

- **Technical Details**: WiCkeD는 기존의 다지선다형 질문에서 무작위로 하나의 선택지를 'Above none of the above'로 교체하는 방식으로 작동합니다. 이 방법은 각 질문의 정답을 변경하지 않고도 질문의 난이도를 증가시킵니다. 이를 통해 6개의 인기 있는 벤치마크를 평가하였으며, 다수의 LLM에서 성능이 감소하는 경향을 보였습니다.

- **Performance Highlights**: 모델의 성능은 WiCkeD 변형을 사용할 경우 평균적으로 7.2-19.7 포인트 떨어졌고, 특히 chain-of-thought 방식을 사용할 경우에도 성능 저하가 나타났습니다. WiCkeD는 더 많은 reasoning 능력을 요구하는 쪽에서 모델의 민감도를 드러내어 원래 벤치마크와는 다른 정보를 제공합니다. 이로 인해 WiCkeD는 모델의 다양한 능력을 평가하는 데 유용한 도구로 자리 잡고 있습니다.



### Looking forward: Linguistic theory and methods (https://arxiv.org/abs/2502.18313)
- **What's New**: 이번 장에서는 계산(computational), 인지(cognitive), 진화적(evolutionary) 관점의 통합을 집중적으로 다루며, 현대 언어학의 주요 발전 경향을 조명합니다. 특히, 이론적 가설(hypotheses)을 명시적으로 검증하는 방법과 인공 신경망(artificial neural networks)이 언어학 이론에 미치는 영향을 강조합니다.

- **Technical Details**: 네 가지 주요 주제는 심볼 표상(symbolic representation)에 대한 가설의 테스트, 인공 신경망의 이론적 논의와 언어 분석에 미치는 영향, 언어학 이론에서의 상호주관성(intersubjectivity)의 중요성, 그리고 진화 언어학(evolutionary linguistics)의 성장입니다.

- **Performance Highlights**: 언어학을 컴퓨터 과학(computer science), 심리학(psychology), 신경과학(neuroscience), 생물학(biology)과 연결하면서 언어 연구의 변화하는 환경에 대한 미래지향적 관점을 제공합니다.



### RefuteBench 2.0 -- Agentic Benchmark for Dynamic Evaluation of LLM Responses to Refutation Instruction (https://arxiv.org/abs/2502.18308)
Comments:
          Work on progess

- **What's New**: 이 연구에서는 RefuteBench 2.0을 소개하며, 이는 기존의 RefuteBench를 크게 확장한 버전입니다. LLM 에이전트를 반론자(refuter) 및 평가자(evaluator)로 통합하여 보다 유연하고 포괄적인 평가를 가능하게 했습니다. 새로운 시스템은 사용자 반론 피드백을 효과적으로 통합하는 LLM의 능력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: RefuteBench 2.0은 LLM이 주어진 사용자 반론을 어떻게 이해하고 반영하는지를 평가하는 데 초점을 맞추고 있습니다. 연구에서는 transient refutation(일시적인 반론)과 persistent refutation(지속적인 반론)의 두 가지 유형을 구분하여 각각에 대한 모델의 반응을 평가합니다. LLM 기반의 평가자는 인간 평가자와 높은 상관관계를 보였으며, 반론자는 더 인간적인 반론을 생성하는 능력을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 현재의 LLM들은 반론에 대해 만족스러운 응답을 제공할 수 있지만, 대화가 길어질수록 반론 정보를 유지하는 데 어려움을 겪는 것으로 나타났습니다. 특히, 일시적인 반론의 수가 증가할수록 초기 작업의 성능이 감소하는 경향을 보였습니다. 이는 LLM이 긴 대화 맥락에서 이전 정보를 유지하고 적절히 활용하는 데 한계가 있음을 시사합니다.



### How Vital is the Jurisprudential Relevance: Law Article Intervened Legal Case Retrieval and Matching (https://arxiv.org/abs/2502.18292)
- **What's New**: 이 논문에서는 법적 사례 검색(LCM)과 유사 사례 매칭(LCM) 간의 문제를 보다 효과적으로 해결하기 위한 최첨단 모델인 LCM-LAI를 제안합니다. 이는 법률 문서의 예측 부과 작업(LAP)을 통해 법적-합리적 정보를 포착하며, 이전 연구에서의 가정에 의존하지 않고 진행됩니다. 또한, LCM-LAI는 법률 분배에 기반한 법적-합리적 유사성을 평가하기 위해 특별한 주의 메커니즘을 도입하여 기존의 의미적 유사성보다 더 효율적입니다.

- **Technical Details**: LCM-LAI는 종속 멀티 태스크 학습(framework) 구조를 채택하여 법률 사건의 사실 서술에서 법적-합리적 정보를 포착합니다. 이 구조는 효율적인 법률 문서의 예측 부과(sub-task) 기술을 포함하고 있어, 법률 기사가 입력으로 필요하지 않은 방식으로 작동합니다. LCM-LAI는 또한 법적 사건 간의 상호작용을 모델링하기 위해 혁신적인 article-aware attention 메커니즘을 도입하여, 사건 간 문장에서 법적-합리적 상관관계를 측정합니다.

- **Performance Highlights**: 다양한 실제 데이터 세트를 기반으로 한 실험 결과, LCM-LAI는 LCR 및 LCM 작업에서 최첨단 성능을 달성하였습니다. 특히, 법적 사항들을 처리하는 능력이 개선되어 기존 모델들보다 더 높은 정확도를 보여줍니다. 이러한 성과는 지능형 법률 시스템에서의 판결 지원 및 적절한 선례를 제공하는 데 큰 기여를 할 것으로 예상됩니다.



### Uncertainty Modeling in Multimodal Speech Analysis Across the Psychosis Spectrum (https://arxiv.org/abs/2502.18285)
- **What's New**: 이번 연구에서는 정신병 스펙트럼의 미세한 언어 장애를 포착하기 위한 불확실성 인식 모델을 개발하여 증상 심각도를 예측합니다. 연구팀은 독일어로 진행된 구조화된 인터뷰와 자서전 작업을 통해 수집된 114명의 음성 데이터를 분석했으며, 여기에는 초기 정신병 환자와 다양한 수준의 스키조타입 인구가 포함되었습니다. 이러한 접근 방식으로 증상 변화와 개인의 언어 패턴을 정량화하는 새로운 기회를 제공합니다.

- **Technical Details**: 모델은 악기적(acoustic) 및 언어적(linguistic) 특징을 통합하여 정교한 예측을 지원합니다. 음성 데이터의 불확실성을 정량화함으로써 모델은 음성 패턴의 변동성을 효과적으로 다룰 수 있으며, RMSE를 감소시키고 F1-score 83%를 달성했습니다. 구조적인 맥락에서는 음향 특징의 가중치를 높이고, 비구조적 맥락에서는 언어적 특징을 강조하여 다이나믹하게 조정됩니다.

- **Performance Highlights**: 모델은 다양한 상호작용 맥락에서 강력한 성능을 보여주었으며, 예측 정확성을 크게 향상시켰습니다. 불확실성 추정은 음성 지표의 신뢰도 차이를 식별함으로써 모델의 해석 가능성을 높였습니다. 이 연구는 정신병 스펙트럼 연구에서 조기 탐지 및 개인화된 평가, 임상 의사결정을 강화하는 방향으로 중요한 시사점을 제공합니다.



### Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases (https://arxiv.org/abs/2502.18282)
Comments:
          under review

- **What's New**: 이 연구는 대형 언어 모델(LLMs)과 인간의 의견 간의 정치적 경향성을 분석하는 새로운 방법론을 제시합니다. 특히, 사전 훈련 데이터에 내재된 가치와 편향이 모델의 출력을 어떻게 형성하는지를 실증적으로 조사합니다. 연구는 미국 대법원 사건을 사례로 사용하여 LLM의 정치적 경향과 인간 의견의 상관관계를 탐색하며, 이러한 상관관계가 존재하지 않음을 보여줍니다.

- **Technical Details**: 저자들은 SCOPE라는 데이터셋을 활용하여 32개의 미국 대법원 사건에 대한 LLM의 정치적 경향을 조사합니다. 이 데이터셋은 전문가들이 선정한 경우들로 구성되어 있으며, 각 사건에 대한 설문 응답을 수집하여 LLM의 응답과 비교합니다. 실험에서는 여러 개의 LLM 모델, 포함하여 Gemma-7b-it, Llama-3, OLMo-7B와 같은 오픈 소스 모델과 GPT-4o을 평가합니다.

- **Performance Highlights**: 연구의 결과는 LLMs가 훈련 데이터의 정치적 경향을 강하게 반영하고 있다는 것을 확인했습니다. 그러나, 인간 의견과의 상관관계가 약하다는 것을 발견했고, 이는 LLM의 훈련 데이터 수집 및 평가의 중요성을 강조합니다. 이러한 발견은 LLMs의 응답이 인간 중심의 가치와 얼마나 일치하는지를 보장하기 위해 기술적인 평가 기준이 필요함을 나타냅니다.



### Self-Adjust Softmax (https://arxiv.org/abs/2502.18277)
Comments:
          Tech Report

- **What's New**: 이 연구에서는 Transformer의 attention 메커니즘에서 자주 사용되는 softmax 함수의 한계를 해결하기 위해 Self-Adjust Softmax (SA-Softmax)를 제안합니다. 기존의 softmax가 극단적인 값에 취약하여 gradient vanishing 문제를 일으키는 반면, SA-Softmax는 gradient의 전달을 개선하면서도 기존의 확률적 속성과 순위를 유지합니다. 이 방법은 기존의 Transformer 아키텍처에 쉽게 통합될 수 있도록 설계되었습니다.

- **Technical Details**: SA-Softmax는 $softmax(x)$를 $rac{(x - min(x_{min},0))}{max(0,x_{max})-min(x_{min},0)} 	imes softmax(x)$로 수정하여 gradient의 흐름을 증대시킵니다. 이 수정된 함수는 attention 메커니즘의 핵심 속성인 입력 값의 상대적인 순서를 유지하며, 효과적인 attention 메커니즘의 작동에 필수적입니다. 제안된 SA-Softmax는 2.7억 개의 매개변수를 가진 대규모 모델에서도 효과적으로 작동하며, 다양한 데이터셋과 작업에서도 우수한 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, SA-Softmax는 traditional softmax보다 더 나은 Gradient 특성을 제공함을 보였습니다. 특히, 다양한 데이터셋과 언어 작업에서 SA-Softmax를 도입한 Transformer 모델이 더 우수한 학습 및 최적화를 경험하였고, 깊은 아키텍처에서도 안정적인 성능을 유지하였습니다. 이는 SA-Softmax가 대규모 모델에서 softmax의 단점을 보완할 수 있음을 시사합니다.



### Beyond In-Distribution Success: Scaling Curves of CoT Granularity for Language Model Generalization (https://arxiv.org/abs/2502.18273)
- **What's New**: 이 논문은 변환기 기반 언어 모델(Transformer-based LMs)이 배포 시 분포 이동을 겪는 복합 작업(compound tasks)에서 향상된 일반화(generalization) 능력을 어떻게 갖출 수 있는지를 조사합니다. Chain-of-Thought (CoT) 추론이 OOD(Out-of-Distribution) 일반화를 증대시키는 방법을 다루며, 이를 위해 여러 복합 작업에서의 실험을 통해 세 가지 주요 통찰(insight)을 발견했습니다. 이 연구 결과는 CoT가 언어 모델의 성능 향상에 중요한 역할을 할 수 있음을 보여줍니다.

- **Technical Details**: 실험을 통해 두 가지 데이터 패러다임(1) 결과 지향 질문-답변(Q-A) 쌍과 (2) 다양한 세분화(granularity) CoT 시퀀스를 기반으로 훈련된 언어 모델을 평가했습니다. 결과 지향 Q-A와 CoT 훈련된 모델이 높은 인 배포(in-distribution) 정확도를 달성하며, CoT 데이터를 세분화할수록 일반화 성능이 향상된다는 것을 발견했습니다. 모델의 성능은 다양한 조건을 포함한 긴 CoT 시퀀스에 따라 향상될 수 있습니다.

- **Performance Highlights**: 연구 결과 및 이론적 분석을 통해 CoT 훈련이 잘못된 학습(shortcut learning)을 완화하고 언어 모델의 일반화를 개선하는 데 도움이 된다는데 유의미한 증거를 제시합니다. CoT로 훈련된 언어 모델은 상대적으로 적은 데이터로도 Q-A 훈련과 유사한 성능을 달성하며, 이는 데이터 수집 방식이 모델의 일반화 능력에 미치는 영향을 강조합니다. 저자들은 수집된 고품질 CoT 데이터가 작은 양에서도 큰 효과를 발휘할 수 있음을 보여주며, 이는 실제 응용에 있어 중요한 시사점을 제공합니다.



### Debt Collection Negotiations with Large Language Models: An Evaluation System and Optimizing Decision Making with Multi-Agen (https://arxiv.org/abs/2502.18228)
Comments:
          21 pages

- **What's New**: 이 논문은 부채 수금 협상(DCN)에서 자동화를 위한 대형 언어 모델(LLMs)의 활용 가능성을 탐구하고 있습니다. 특히, LLMs가 인공지능 에이전트를 지원하여 더 복잡한 작업을 수행할 수 있는 잠재력을 보이는 가운데, DCN 수행을 위한 새로운 평가 프레임워크를 제안하고 있습니다. 이 프레임워크는 4가지 측면에서 13개의 메트릭을 포함하여 기존의 DCN 방식과 차별화됩니다.

- **Technical Details**: DCN은 채권자가 미지급 채무를 회수하고 채무자의 신용을 회복하기 위해 시작하는 협상을 의미합니다. 이 논문에서는 CTGAN을 활용하여 975개의 부채 기록을 포함하는 합성 데이터셋을 구축하였습니다. 또한, 다중 에이전트 부채 협상(MADeN) 프레임워크를 제안하여 계획 및 판단 모듈을 포함시켜 의사 결정을 개선하려고 하였습니다.

- **Performance Highlights**: 연구 결과, LLMs는 채무자의 금융 상태를 기반으로 적절한 결정을 내리지 못하고, 인간 협상자보다 과도한 양보 경향을 보이는 것으로 나타났습니다. 이를 개선하기 위해, 우리는 DPO(Decision Preference Optimization)와 리젝션 샘플링 기법을 적용하여 부채 회수율과 효율성에 대한 에이전트의 집중도를 높이고, Qwen2.5-7B 모델에서 다양한 메트릭에서 성능을 향상시켰습니다.



### Connecting Voices: LoReSpeech as a Low-Resource Speech Parallel Corpus (https://arxiv.org/abs/2502.18215)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 본 논문에서는 LoReSpeech라는 저자원 음성-음성 번역 코퍼스를 구축하기 위한 방법론을 제안합니다. 이 접근법은 LoReASR이라는 짧은 음성-전사 정렬의 하위 코퍼스를 생성하는 것으로 시작합니다. 또한 이 작업은 Tutlayt AI 프로젝트의 일환으로 진행되며, 저자원 언어의 음성 및 언어 자원을 개발하는 것을 목표로 하고 있습니다.

- **Technical Details**: LoReASR는 저자원 언어에 대한 ASR 코퍼스로, 짧고 정확한 정렬(음성-전사)을 제공합니다. 이 코퍼스는 Chechen, Cham, Comorian, Dzongkha, Kabyle, Inuktitut, Malagasy, Yucatec Maya, Navajo, Khumzari, Soninke의 10개 언어에 초점을 맞추고 있습니다. 우리의 방법론은 텍스트 음성을 위한 데이터 수집 및 전용 플랫폼을 활용하며, 오랜 음성 기록을 세분화된 번역과 정렬하는 데 방향성을 제공합니다.

- **Performance Highlights**: LoReSpeech는 다국어 ASR 시스템과 직관적인 음성-음성 번역 모델의 발전을 가능하게 합니다. 이 연구는 저자원 언어를 위한 인트라 및 인터 언어 정렬 음성 코퍼스를 창출하면서 음성 자원 측면에서 과잉 자원 언어와 저자원 언어 간의 격차를 해소하기 위한 기초 단계를 제공합니다.



### LAG: LLM agents for Leaderboard Auto Generation on Demanding (https://arxiv.org/abs/2502.18209)
- **What's New**: 이 논문은 Leaderboard Auto Generation (LAG)이라는 새로운 프레임워크를 소개하여 인공지능(AI)과 같은 빠르게 발전하는 분야 내의 연구 주제에 대한 자동 생성된 리더보드를 효율적으로 구축하는 방법을 제시합니다. 매일 대량의 AI 논문이 업데이트되면서 연구자들이 각 논문의 방법론, 실험 결과 및 설정을 추적하는 것이 점점 더 어려워지고 있습니다. LAG는 논문 수집, 실험 결과 추출 및 통합, 리더보드 생성, 품질 평가의 체계적인 접근 방식을 통해 이러한 문제를 해결합니다.

- **Technical Details**: LAG는 논문 수집, 테이블 추출 및 분류, 테이블 언팩킹 및 통합, 리더보드 생성 및 평가의 네 가지 단계로 구성됩니다. 이 프레임워크는 관련 있는 논문을 아카이브에서 자동으로 다운로드하고 신뢰성 있는 실험 설정을 제공하여 공정한 비교를 가능하게 합니다. 평가 측면에서 LAG는 주제 관련 품질과 내용 품질의 두 가지 품질 차원을 제시하여 생성된 리더보드의 효과성을 극대화합니다.

- **Performance Highlights**: LAG는 다채로운 리더보드 길이에서 실험을 통해 높은 주제 관련성 및 내용 품질 점수를 지속적으로 달성했습니다. 특히 20개의 항목이 포함된 리더보드를 생성했을 때, 주제 관련 품질에서 67.58%의 회수율과 70.33%의 정밀도를 기록했습니다. LAG는 수동으로 작성된 리더보드에 비해 효율성에서 우수하며, 인간의 평가와 높은 상관관계를 나타내는 결과를 도출하였습니다.



### Grandes modelos de lenguaje: de la predicción de palabras a la comprensión? (https://arxiv.org/abs/2502.18205)
Comments:
          26 pages, in Spanish. Chapter from book "La Inteligencia Artificial hoy y sus aplicaciones con Big Data", (Amparo Alonso Betanzos, Daniel Peña y Pilar Poncela, eds.). Publisher: Funcas. ISBN 978-84-17609-94-8

- **What's New**: AI 분야에서 대형 언어 모델(large language models)의 혁신이 일어나고 있으며, 특히 ChatGPT와 같은 모델이 주목받고 있습니다. 이 기술은 다양한 실제 응용 프로그램과 탐색되지 않은 잠재력을 가지고 있어서, 연관된 새로운 가능성을 열어주고 있습니다. 그러나 이러한 기술의 작동 원리와 언어 이해 능력에 대한 의문이 제기되어 있으며, 윤리적 문제도 함께 고려해야 합니다.

- **Technical Details**: 이 장에서는 대형 언어 모델의 발전 과정과 기본적인 작동 원리를 설명합니다. 이를 통해 이러한 모델의 능력과 한계를 더 잘 이해할 수 있게 되며, 기술 사용의 주요 논쟁점을 소개할 수 있습니다. 이러한 모델들은 데이터(데이터)를 통해 학습하며, 고급 자연어 처리(natural language processing) 기법들이 사용됩니다.

- **Performance Highlights**: 대형 언어 모델은 특정 작업에서 인간과 유사한 성능을 보여주는 가능성 때문에 많은 관심을 받고 있습니다. 그러나 이러한 모델의 결과물에서 발생할 수 있는 편향(bias)과 신뢰성(credibility) 문제는 여전히 중요한 이슈로 남아 있습니다. 앞으로 기술 발전과 더불어 이러한 논쟁을 해결하기 위한 연구가 필요합니다.



### Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs (https://arxiv.org/abs/2502.18179)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 이용한 레이아웃이 풍부한 문서에서 정보 추출(IE)의 설계 공간을 정의하고 탐색합니다. LLMs를 활용하는 레이아웃 인식 IE의 세 가지 핵심 도전 과제는 데이터 구조화, 모델 참여, 출력 세분화로 나뉩니다. 이 연구는 입력 표현, 청크(chunking), 프롬프트(prompting), LLM 및 멀티모달 모델의 선택 등과 같은 하위 문제를 탐구하고, 다양한 설계 선택의 결과를 새로운 테스트 수트를 통해 평가합니다.

- **Technical Details**: 레이아웃이 풍부한 문서에서의 IE는 텍스트와 복잡한 시각 레이아웃이 서로 얽힌 문서에서 정보를 식별하고 추출하여 구조화된 정보 인스턴스로 매핑하는 과정을 포함합니다. LLM은 문서의 레이아웃 정보를 잘 혼합하여 텍스트 기반 모델과 다르게 접근해야 하며, 다양한 프리프로세싱과 포스트프로세싱 기술을 평가할 필요가 있습니다. 연구에서는 LLMs의 성능을 다양한 모델과 비교하고 레이아웃 인식 모델인 LayoutLMv3와의 성능을 비교하여 인사이트를 제공합니다.

- **Performance Highlights**: 실험 결과, LLM은 특수한 모델과 경쟁할 수 있으며, 특히 데이터 훈련 없이도 유사한 성능을 발휘할 수 있습니다. 비교 연구에서는 기본 모델 대비 14.1 포인트 F1 점수 향상과 함께 다양한 조합 및 방법론의 효과를 분석했습니다. 그러나 멀티모달 LLM이 여전히 더 나은 성능을 보이지만, 비용과 투명성의 문제를 동반하고 있습니다.



### SECURA: Sigmoid-Enhanced CUR Decomposition with Uninterrupted Retention and Low-Rank Adaptation in Large Language Models (https://arxiv.org/abs/2502.18168)
Comments:
          New work on Parameter-Efficient Fine-Tuning (PEFT) for large language models. Includes new techniques SigNorm and CABR-LoRA for optimizing fine-tune performance and Knowledge retention

- **What's New**: 본 논문에서는 SECURA: Sigmoid-Enhanced CUR Decomposition LoRA를 제안하며, 파라미터 효율적인 미세 조정(PEFT) 접근 방식을 통해 파라미터 소실 문제를 완화하고 성능을 향상시킵니다. SECURA는 새로운 정규화 기법인 SigNorm을 도입하여 파라미터 보존과 전반적인 성능을 강화하는 방안을 제시합니다. 다양한 작업에서 SECURA가 기존의 LoRA 방법에 비해 고급 성능을 보여주었다는 점이 주목할 만합니다.

- **Technical Details**: SECURA의 두 가지 핵심 혁신은 CABR 분해와 SigNorm 정규화입니다. CABR 분해는 CUR-LoRA의 성능을 개선하기 위해 역 저순위 적응 행렬을 도입하여 추가 차원을 통합합니다. SigNorm은 시그모이드의 점진적 전이 속성을 활용하여 파라미터를 동적으로 조정하는 정규화 방법으로, 모델의 안정성과 성능을 균형 있게 유지하는 데 기여합니다.

- **Performance Highlights**: SECURA는 4개의 다중 선택 질문(MCQ) 작업에서 평균 3.63% 향상된 성능을 보였고, 5개의 질문 응답(QA) 작업에서도 2.56% 개선을 달성했습니다. 16개의 지속 학습 테스트에서 70% 이상의 정확도를 유지하며 기존의 다양한 방법보다 우수한 지식 보존 능력을 입증했습니다. 이러한 결과는 SECURA가 업무 적응 및 사전 훈련된 지식 보존에서 뛰어난 성능을 가진다는 것을 강조합니다.



### Can LLMs Explain Themselves Counterfactually? (https://arxiv.org/abs/2502.18156)
- **What's New**: 본 논문에서는 Self-Generated Counterfactual Explanations (SCEs)에 대한 연구를 소개합니다. LLM(대형 언어 모델)의 출력을 스스로 설명하도록 유도하는 새로운 패러다임이 떠오르고 있으며, 이 연구는 다양한 LLM이 SCE를 얼마나 잘 생성하는지를 평가합니다. 이를 위해 다양한 LLM과 데이터셋을 사용한 실험을 통해 모델의 내부 추론 과정에서 몇 가지 결함을 발견했습니다.

- **Technical Details**: 연구에서는 7B에서 70B 파라미터를 가진 7개의 LLM과 4개 고유 작업에 해당하는 6개의 데이터셋을 사용하여 실험을 진행하였습니다. 실험 절차는 첫째로 모델에게 예측을 요청하고, 그 다음 SCE를 생성하게 하며, 마지막으로 생성된 SCE에 대한 모델의 예측을 계산하는 방식으로 구성됩니다. 결과적으로, 대부분의 LLM은 SCE를 생성하지만, 그 예측 결과가 목표 레이블과 일치하지 않는 경우가 많습니다.

- **Performance Highlights**: LLM이 SCE를 생성하는 데 있어 내부 추론 과정의 결함이 드러났으며, 이는 원래 예측과 SCE 생성 지시가 모델의 최종 예측에 큰 영향을 미친다는 것을 시사합니다. 특히 GSM8K 수학 데이터셋에서 SCE의 유효성이 낮게 나타났습니다. 연구 결과는 현대 LLM이 스스로의 예측을 설명하는 능력에 있어 여전히 부족함을 보여주고 있습니다.



### NusaAksara: A Multimodal and Multilingual Benchmark for Preserving Indonesian Indigenous Scripts (https://arxiv.org/abs/2502.18148)
- **What's New**: 이번 논문에서는 인도네시아 언어와 스크립트에 대한 새로운 공공 벤치마크인 NusaAksara를 소개합니다. 기존의 NLP(자연어 처리) 진전은 주로 로마자 텍스트를 중심으로 이루어졌지만, NusaAksara는 원본 스크립트를 포함합니다. 이 벤치마크는 이미지 분할(image segmentation), OCR(광학 문자 인식), 음역(transliteration), 번역(translation), 언어 식별(language identification) 등의 다양한 작업을 포괄합니다.

- **Technical Details**: NusaAksara는 7개 언어, 8개 스크립트를 아우르며, NLP 벤치마크에서 잘 나타나지 않는 저자원(low-resource) 언어까지 포함합니다. 데이터는 전문가들이 엄격한 절차를 통해 생성하였으며, 유니코드(Unicode)로 지원되지 않는 람푼그(Lampung) 스크립트도 포함되어 있습니다. 이 데이터는 GPT-4o, Llama 3.2, Aya 23과 같은 LLM(대형 언어 모델) 및 VLM(비전 언어 모델), PP-OCR, LangID와 같은 작업 특정 시스템을 통해 벤치마킹되었습니다.

- **Performance Highlights**: 대부분의 NLP 기술이 인도네시아의 지역 스크립트를 처리하지 못하며, 많은 모델들이 거의 제로 성능에 가까운 결과를 보였습니다. NusaAksara는 인도네시아의 언어 다양성을 반영하고, 향후 연구와 개발에 중요한 기초 자료를 제공할 것입니다. 이 연구는 인도네시아 NLP 생태계의 발전에 기여할 중요한 이정표로 평가됩니다.



### LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers (https://arxiv.org/abs/2502.18139)
Comments:
          First submit

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)라는 새로운 접근 방식을 제안합니다. 기존의 RAG 방식은 쿼리 재작성을 통해 사용자 의도를 명확히 하고, 하이브리드 검색을 통해 검색 범위를 확장하는데 초점을 맞추었습니다. 그러나 이러한 방식은 쿼리 재작성이 밀접하게 연결되어 있어 하이브리드 검색의 호환성을 저해하고 있습니다. 이에 따라, LevelRAG는 복잡한 쿼리를 독립적인 원자 쿼리로 분해할 수 있는 고수준 검색기를 도입하여 이 문제를 해결합니다.

- **Technical Details**: LevelRAG는 고수준 검색기와 저수준 검색기(희소 검색기, 웹 검색기, 밀집 검색기)를 결합하여 정보 검색 로직을 최적화합니다. 고수준 검색기는 사용자의 쿼리를 원자 쿼리로 변환하여 저수준 검색기에게 전달하며, 이들은 해당 쿼리를 통해 데이터베이스에서 정보를 검색합니다. 특히 희소 검색기는 Lucene 구문을 사용하여 키워드 검색의 정확성을 높이며, 밀집 검색기는 복잡한 쿼리를 다루는 데 강점을 가지고 있습니다. 이 구성 요소들은 함께 작용하여 검색 프로세스의 완전성과 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, LevelRAG는 단일 및 다단계 질문 응답 작업에서 기존 RAG 방법보다 우수한 성능을 보였습니다. 특히, LevelRAG는 최신 고급 모델인 GPT4o를 능가하는 응답 품질을 보여 주었습니다. 또한, 제안된 희소 검색기만으로도 기존의 다양한 방법보다 더 나은 성능을 발휘한 것으로 나타났습니다.



### Uncertainty Quantification in Retrieval Augmented Question Answering (https://arxiv.org/abs/2502.18108)
- **What's New**: 이 논문에서는 Retrieval augmented Question Answering (QA) 방식을 통해 QA 모델이 제공된 지식을 활용하여 정확한 답변을 생성할 수 있도록 하고자 한다. 연구자들은 QA 모델이 질문에 대한 답변의 정확성을 예측하는 데 있어, 주어진 passage의 유용성을 정량화하는 새로운 방법을 제안하였다. 특히, 경량의 신경망 모델을 훈련시켜 passage의 유용성을 예측하고, 이를 통해 QA 모델의 불확실성을 측정하는 방식을 소개하고 있다.

- **Technical Details**: 이 연구는 Passage의 유용성을 판단하기 위해 정확도(accuracy)와 관련성(entailment) 지표를 결합하여 작은 신경망 모델을 훈련시킨다. 해당 방법은 간단한 정보 이론적 메트릭이 일정 정도까지는 답변의 정확성을 예측할 수 있음을 보여준다. 그러나 저자들은 자신의 접근 방식이 더 비싼 샘플링 기반 방법들과 비교하여 효율적인 성능을 보여준다고 주장하며, QA 시스템의 테스트 시간을 단축시킬 수 있는 가능성을 열어두고 있다.

- **Performance Highlights**: 여섯 개의 데이터 세트에 대한 평가 결과, 본 연구의 불확실성 추정기가 기존의 샘플링 기반 방법들보다 우수한 성능을 보이며, 특히 복잡한 추론 질문과 답변이 불가능한 상황에서도 더 나은 결과를 나타낸다. 이 연구의 결과는 샘플링 기반 솔루션들이 테스트 시간과 비용 측면에서 비효율적임을 강조하며, 실제 QA 시스템에서의 응답 성능을 개선하는 잠재력을 지니고 있다.



### Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning (https://arxiv.org/abs/2502.18080)
- **What's New**: 최근 연구에 따르면, Chain of Thoughts (CoT)의 길이를 늘리는 것이 LLMs의 복잡한 추론 작업에서 성능 향상에 기여할 수 있다는 점이 주목받고 있습니다. 그러나 본 논문에서는 CoT 길이를 지나치게 늘리는 것이 LLM의 추론 성능에 부정적인 영향을 미칠 수 있다는 우려를 제기합니다. 특정 도메인에서는 CoT 길이가 길어질수록 오히려 성능이 저하되는 경향이 관찰되었습니다.

- **Technical Details**: 연구진은 전체 모델 학습 과정에서 다양한 응답 길이 분포를 가진 소수의 시드 데이터 세트를 활용하여 모델이 깊이 있는 사고를 위한 다양한 추론 노력을 채택하도록 학습을 진행했습니다. 이후 모델은 주어진 문제에 대해 각기 다른 추론 노력을 기반으로 최적의 짧은 정답 응답을 선택하여 자기 개선(Self-improvement)을 시도합니다. 이 과정에서 Thinking-Optimal Scaling 전략(TOPS)이 도입되어, LLM이 문제를 해결하기 위해 필요한 토큰 수를 스스로 결정할 수 있도록 합니다.

- **Performance Highlights**: 이 연구의 결과로 Qwen2.5-32B-Instruct를 기반으로 한 자기 개선 모델이 다양한 수학 벤치마크에서 기존의 32B o1-like 모델을 초월하는 성과를 달성했습니다. 특히, GSM8K, MATH500 및 AIME2024를 포함한 여러 벤치마크에서 우수한 성능을 나타내어, QwQ-32B-Preview와 유사한 성능을 기록했습니다. 이는 다른 모델보다 적절한 추론 노력을 강조하는 개인화된 접근 방식의 중요성을 강조합니다.



### Uncertainty-aware abstention in medical diagnosis based on medical texts (https://arxiv.org/abs/2502.18050)
- **What's New**: 이 연구는 AI 지원 의료 진단의 신뢰성을 다루고 있습니다. 진단 시스템이 확신하지 못할 경우 결정을 내리지 않고 회피하는 선택 예측(selective prediction) 접근 방식을 초점을 맞추고 있습니다. 또한, HUQ-2라는 신뢰성을 높이는 새로운 방법을 도입하여 여러 불확실성 정량화 방법들을 비교했습니다.

- **Technical Details**: 의료 텍스트 분석을 위한 불확실성 정량화 방법에 대한 연구가 이루어졌습니다. 이 연구는 MIMIC-III와 MIMIC-IV 데이터셋을 활용한 사망률 예측 및 다중 라벨 의료 코드 예측을 포함하여 정신 건강 데이터셋을 분석하고 있습니다. HuQ-2 방법은 aleatoric 불확실성과 epistemic 불확실성을 결합하여 선택적 예측의 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 여러 의료 데이터셋을 활용한 실험에서 HUQ-2는 다른 불확실성 정량화 기법에 비해 탁월한 성능을 나타냈습니다. 다중 레이블 의료 코드 예측의 경우 라벨 수준에서 선택적 예측이 수행될 때 예측 정확도가 상당히 향상되었습니다. 이 결과들은 의료 텍스트 분석의 신뢰성과 해석 가능성을 높이기 위한 기초 자료가 될 것입니다.



### Harnessing Multiple Large Language Models: A Survey on LLM Ensemb (https://arxiv.org/abs/2502.18036)
Comments:
          9 pages, 2 figures, codebase: this https URL

- **What's New**: 최근 대규모 언어 모델(LLMs)의 활용이 증가하며, LLM Ensemble의 중요성이 부각되고 있습니다. 이 논문은 LLM Ensemble의 발전을 체계적으로 정리한 첫 번째 리뷰로, 분류 방법, 관련 문제 및 향후 연구 방향을 제안합니다. LLM의 개별적인 강점을 활용하기 위해 여러 모델을 병합하는 기법을 다루고 있습니다.

- **Technical Details**: LLM Ensemble은 세 가지 넓은 범주로 나뉘며, 각 카테고리는 'inference 전', 'inference 중', 'inference 후'로 구분됩니다. 'inference 전'에서는 적절한 모델을 라우팅하기 위한 알고리즘을 사용하고, 'inference 중'에서는 여러 LLM의 출력을 통합하여 역동적으로 응답을 생성합니다. 마지막으로 'inference 후'에서는 모든 모델의 완전한 응답을 통합합니다.

- **Performance Highlights**: 이 연구는 LLM Ensemble에 대한 포괄적인 분석을 제공하며, 이러한 앙상블 법들이 처리가능성과 정확성을 크게 향상시킬 수 있음을 보여줍니다. 다양한 조합 및 응답 통합 방법은 사용자 쿼리에 대해 더 나은 성능을 제공합니다. 향후 연구에서는 새로운 방법론과 벤치마크 개발을 통한 성능 향상 가능성을 제시하고 있습니다.



### Detecting Knowledge Boundary of Vision Large Language Models by Sampling-Based Inferenc (https://arxiv.org/abs/2502.18023)
Comments:
          Under review

- **What's New**: 이 연구에서는 Visual Large Language Models (VLLMs)의 지식 경계를 탐지하는 방법을 제안합니다. 기존 Retrieval Augmented Generation (RAG) 기술의 의존도를 완화하면서도 성능을 유지하거나 개선할 수 있는 방법을 개발했습니다. 이를 통해 VLLMs가 자신의 지식 범위를 초과하는 질문에 더 효율적으로 대응할 수 있도록 합니다.

- **Technical Details**: 제안하는 방법은 VLLM을 미세 조정하여 문제에 따라 하드 또는 소프트 지식 경계를 정의합니다. 두 가지 변형으로 구성된 이 방법은 VLLM의 출력을 샘플링하여 지식 경계를 추정하고, 이를 통해 RAG를 활용해 정확도를 검증합니다. 제안된 방법은 자동으로 생성된 데이터셋을 기반으로 하여 수작업 주석 없이 시행됩니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 RAG의 사용을 줄이는 동시에 다양한 유형의 데이터에서 VLLM의 성능을 유지하거나 개선한다는 것을 보여줍니다. 특히 혼합 데이터셋에서, RAG를 무차별적으로 사용하는 경우보다 50.67%의 검색 감소를 달성했습니다. 한 VLLM에 대한 지식 경계는 다른 VLLMs의 대체 경계 식별로도 활용될 수 있음을 보여줍니다.



### AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages (https://arxiv.org/abs/2502.18020)
- **What's New**: 이 논문에서는 다국어 모델을 위한 새로운 하이브리드 지식 증류(knowledge distillation) 접근법을 제안합니다. 기존의 방법들이 다국어 모델, 특히 저자원(low-resource) 언어에 대한 성능 유지를 어렵게 하는 문제를 해결하기 위해, 전통적인 지식 증류 방식과 단순화된 attention matching 메커니즘을 결합했습니다. 이로 인해, 전통적인 다국어 모델보다 유의미하게 작은 크기의 학생 모델 구조를 도입하여, 아프리카 언어 다섯 개에 대한 평가에서 효과를 입증했습니다.

- **Technical Details**: 우리가 제안한 하이브리드 증류 프레임워크는 지식 증류와 attention matching을 결합하여 학생 모델이 교사 모델의 출력 분포와 내부 attention 패턴을 모두 학습하도록 합니다. 특히, 매우 compact한 다국어 학생 모델을 설계하였으며, 이는 기존 모델보다 훨씬 작은 hidden dimension을 가집니다. 실험적으로, 아프리카어 저자원 언어인 Kinyarwanda, Swahili, Hausa, Igbo, Yoruba에 대해 이 접근법을 평가하고, 모델 크기를 85% 이상 줄이면서도 경쟁력 있는 성능을 달성했습니다.

- **Performance Highlights**: 이 연구의 실험 결과는 제안된 하이브리드 접근법이 교사 모델과 비교하여 성능 면에서 경쟁력을 유지함을 보여줍니다. 학생 모델은 원래 모델의 성능에서 85% 이내의 정확도를 유지하면서도, 연산 자원을 현저히 절감할 수 있었습니다. 이는 저자원 환경에서 다국어 모델을 배치하는 데 유용한 실용적인 프레임워크를 제공하며, 아프리카 언어와 관련된 응용 프로그램에서 큰 혜택을 제공합니다.



### Verdict: A Library for Scaling Judge-Time Compu (https://arxiv.org/abs/2502.18018)
- **What's New**: 이번 논문에서는 자동 판사 기능을 가진 LLM(Large Language Model)의 신뢰성 문제를 해결하기 위해 Verdict라는 오픈 소스 라이브러리를 소개합니다. 이 라이브러리는 모듈형 추론 단위(modular reasoning units)의 조합을 통해 정확성, 신뢰성, 해석 가능성을 향상시키는 데 초점을 맞추고 있습니다. 특히, Verification, Debate, Aggregation 같은 다양한 추론 단위를 활용하여 판사의 품질을 개선하고자 합니다.

- **Technical Details**: Verdict는 추론 시간(compute) 강화를 통해 LLM 판사의 성능을 최적화합니다. 이 라이브러리는 콘텐츠 조절(content moderation), 사실 확인(fact-checking), 환각 감지(hallucination detection)와 같은 다양한 과제를 처리하며, SOTA(state-of-the-art) 성능을 달성합니다. Verdict는 대규모로 조정된 판사보다 수십 배 더 우수한 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, Verdict는 기존의 미세 조정된 판사(fine-tuned judges) 및 논의 모델(reasoning models)을 초월하며, 다양한 복잡한 작업에서 뛰어난 성능을 보이고 있습니다. 이러한 성과는 연구자 및 실무자들이 스케일 가능한 해석 가능하고 신뢰할 수 있는 LLM 기반 평가자를 구축하는 데 유용한 프레임워크로 활용될 수 있기를 바랍니다.



### Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning (https://arxiv.org/abs/2502.18001)
- **What's New**: 이번 연구는 Chain-of-Thought (CoT) 능력을 Small Language Models (SLMs)로 증류하는 방법을 체계적으로 조사합니다. SLM이 CoT 추론을 독립적으로 생성하는 데 어려움을 겪기 때문에, CoT 데이터셋에서 슬픈 힘 시키기(teacher-annotated) 위한 미세 조정이 필요하다는 점을 강조하였습니다. 또한, CoT 사고 커널의 세 가지 주요 요인인 교사 선택, 세부 사항 수준, 설명 형식이 학생 모델의 학습 성과에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 연구의 방법론은 SLM의 학습 성능을 높이기 위한 실험을 포함하며, 4개의 교사 모델과 7개의 학생 모델을 사용하여 7개의 수학 및 상식 추론 데이터셋에서 평가하였습니다. 1-shot prompting 접근 방식을 채택하여 CoT 주석을 생성하였고, 이를 통해 일관된 교수 스타일을 유지하면서도 세부 사항 수준을 조정할 수 있음을 발견하였습니다. 논문은 CoT 감압 방법론을 위한 체계적인 프레임워크를 제시하고, SLM의 추론 능력 향상을 위한 기초를 마련하였습니다.

- **Performance Highlights**: 주요 발견으로는 SLM이 세부 사항 수준에 대해 비선형적인 관계를 보였다는 점입니다. 강한 학생 모델은 더 세밀한 이유로부터 이득을 얻는 반면, 약한 모델은 과도한 설명에 의해 압도될 수 있습니다. 또한, 사람이 주석한 CoT는 LLM이 생성한 CoT보다 성능이 떨어지는 경향이 있으며, 이는 교육 모델의 다양성과 복잡성이 정확도를 초월할 수 있음을 시사합니다.



### MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification (https://arxiv.org/abs/2502.17987)
- **What's New**: 이 논문에서 새로운 점은 저자들이 저자원 (low-resource) 반투 (Bantu) 언어에 특화된 텍스트 분류 모델 MAGE(Multi-Head Attention Guided Embeddings)를 소개한 것이다. 이 모델은 Language-Independent Data Augmentation (LiDA)을 통해 데이터 포인트를 선택적으로 향상시킴으로서 텍스트 분류 성능을 개선하는데 중점을 두고 있다. 특히, MAGE는 데이터 부족 문제를 해결하면서도 반투 언어의 고유한 구문론적(syntactic) 및 의미론적(semantic) 특성을 효과적으로 다루도록 설계되었다.

- **Technical Details**: MAGE는 LiDA 프레임워크를 기반으로 하여 중대한 혁신을 도입하고 있다. 전통적인 Denoising Autoencoder 대신 Variational Autoencoder (VAE)를 도입하여 더욱 표현력이 풍부하고 다양한 합성(augmented) 임베딩을 생성한다. 또한, Multi-Head Attention 메커니즘을 활용하여 임베딩에서 중요한 특징을 강조함으로써 저자원 언어에서 구문론적 및 의미론적 뉘앙스를 더 잘 포착할 수 있도록 한다.

- **Performance Highlights**: MAGE는 AfriSenti SemEval 데이터셋을 사용하여 감정 분류(sentiment classification) 성능 평가를 수행하였으며, 저자원 환경에서 기존의 기준 방법들보다 우수한 성능을 나타냈다. MAGE는 데이터 부족 문제를 해결함과 동시에 다른 저자원 언어 계열로의 텍스트 분류 능력을 확장할 수 있는 스케일러블한 프레임워크로 자리잡고 있다. 이 연구는 향후 저자원 언어 처리 및 분류 작업에 대한 연구의 기초를 제공하고 있으며, NLP 기술의 포괄성과 일반화 가능성을 높이는 방향으로 나아가고 있다.



### On Synthetic Data Strategies for Domain-Specific Generative Retrieva (https://arxiv.org/abs/2502.17957)
- **What's New**: 이 논문은 도메인 특화 코퍼스에 대한 생성 검색 모델을 개발하기 위한 합성 데이터 생성 전략을 조사합니다. 특히, 쿼리 해석을 위해 LLM(Large Language Model)로 생성된 다양한 쿼리를 활용하여 문서 요청을 다루고, 두 단계의 훈련 프레임워크에서 선호 학습을 통해 문서 순위를 개선하는 방법을 제시합니다. 실험을 통해 공개 데이터셋에서 합성 데이터 생성 및 하드 네거티브 샘플링 접근 방식의 효과를 입증하고 있습니다.

- **Technical Details**: 연구는 생성 검색 모델 훈련에서 데이터 전략의 중요성을 강조하며, 두 단계의 훈련 프레임워크를 도입하여 첫 단계에서는 문서 식별자 디코딩을 위한 합성 데이터를 주로 사용합니다. 두 번째 단계에서는 모델의 순위 향상을 위한 선호 학습을 진행하며, 이 과정에서 자기 생성된 데이터를 활용하여 네거티브 샘플을 수집합니다. 특히 고급의 하드 네거티브 후보 선택이 모델 성능에 미치는 영향을 검토합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 다중 세분화 및 제약 기반 쿼리가 문서 검색 성능을 크게 향상시킨다고 밝혔습니다. 또한, 문서 식별자 관련 데이터 선택 전략이 다른 유형의 식별자에도 잘 일반화됨을 시연하였습니다. 마지막으로, RPO(Regularized Preference Optimization)를 사용하여 순위 성능이 효과적으로 개선되며, 고품질 하드 네거티브 후보가 성능에 긍정적인 영향을 미친다는 사실도 확인되었습니다.



### Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments (https://arxiv.org/abs/2502.17956)
- **What's New**: 이 연구에서는 멀티스텝 추론(multi-step reasoning)의 중요성을 강조하며, 다국어 성능(multilingual performance)이 여전히 도전 과제로 남아 있음을 서술합니다. Chain-of-Thought (CoT) prompting이 추론을 개선하지만, 비영어 비문맥으로 인해 한계가 있음을 지적합니다. 새로운 접근 방식으로 Program-of-Thought (PoT) prompting을 제안하며, 이는 추론과 실행을 분리하여 다국어 환경에서의 모델 성능 향상 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 두 단계의 프레임워크를 통해 PoT를 평가하는 방법을 제시합니다. 첫 번째 단계는 질문(Q)에서 추론 단계(R)를 생성하는 것이며, 두 번째 단계는 외부 인터프리터가 R을 실행하여 최종 답변(A)을 얻는 것입니다. 이러한 접근 방식은 다국어 설정에서의 추론과 실행의 분리를 명확하게 하여, 다양한 언어에서도 성능 향상을 도모할 수 있습니다.

- **Performance Highlights**: 실험을 통해 PoT의 미세 조정(fine-tuning)이 다국어 추론의 정확성을 크게 향상시키며, CoT로 미세 조정된 모델보다 뛰어난 성능을 보였음을 확인했습니다. 또한, 추론 품질이 최종 답변의 정확성에 강한 상관관계를 보이며, 이는 모델의 성능 개선을 위한 귀중한 통찰력을 제공합니다. 제안된 기법은 다양한 언어에서의 정확도를 높여 cross-lingual 설정에서 성능이 31.6%에서 56.6%로 증가했습니다.



### Language Models' Factuality Depends on the Language of Inquiry (https://arxiv.org/abs/2502.17955)
- **What's New**: 이번 연구에서는 다국어 언어 모델(Multilingual Language Models, LMs)의 한계를 체계적으로 조사하기 위해 13개 언어에서 10,000개의 국가 관련 사실을 포함하는 데이터셋을 소개합니다. 연구진은 사실 기억(Factual Recall)과 지식 전이(Knowledge Transferability)를 측정하기 위해 세 가지 새로운 지표를 제안했습니다. 이 연구는 LMs가 언어별 특정 사실을 일관되게 기억하지 못하는 경향이 있음을 밝혀냈습니다.

- **Technical Details**: 연구에서는 LMs의 사실 기억을 평가하기 위해 Factual Recall Score (FRS), Knowledge Transferability Score (KTS), Cross-Lingual Factual Knowledge Transferability Score (X-FaKT)라는 세 가지 지표를 개발했습니다. FRS는 모델이 특정 언어에서 사실을 얼마나 정확하게 기억하는지를 측정하며, KTS는 언어 간 지식의 전이 정도를 정량화합니다. 이러한 지표들은 다국어 모델의 성능을 심층적으로 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, LMs는 언어별로 관련된 사실을 정확히 기억하는 반면, 이 지식을 다른 언어로 전이하는 데 어려움을 겪음을 보여주었습니다. LLM의 크기가 사실성과 지식 전이에 중요한 역할을 하며, 언어 자원에 따라 성능 차이가 두드러지게 나타났습니다. 이러한 결과는 다국어 AI 개발에 있어 LMs가 언어 간 지식 통합의 한계를 갖고 있음을 시사합니다.



### DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in Bilingual Complex Ophthalmology Reasoning (https://arxiv.org/abs/2502.17947)
Comments:
          29 pages, 4 figures, 1 table

- **What's New**: 이번 연구에서는 DeepSeek-R1과 최근에 출시된 세 가지 대형 언어 모델(LLMs)의 복합한 이중 언어 안과 사례에 대한 정확성과 추론 능력을 평가했습니다. 130개의 다지선다형 질문(MCQs)이 수집되어 중국 및 영어로 번역되었으며, 각 모델의 응답을 비교하였습니다. 이러한 접근을 통해 최신 LLM의 역량을 심도 있게 분석할 수 있는 기회를 제공하였습니다.

- **Technical Details**: 연구에 사용된 MCQs는 진단(n = 39) 및 관리(n = 91) 관련 질문으로 구성되어 있으며, 총 6개 주제로 분류되었습니다. DeepSeek-R1의 응답은 2025년 2월 15일부터 20일 사이에 기본 설정 하에 생성되었고, 정확도는 올바르게 대답한 질문의 비율로 계산되었습니다. 추론 능력은 추론 논리의 분석과 오류 원인을 통해 평가되었습니다.

- **Performance Highlights**: DeepSeek-R1은 전체적으로 가장 높은 정확도를 기록하였으며, 중국어 MCQs에서 0.862, 영어 MCQs에서 0.808의 정확도를 보였습니다. 다른 LLM 모델들은 중국어에서 각각 0.715, 0.685, 0.692의 정확도를 기록하여 DeepSeek-R1과 비교해 통계적으로 유의미한 차이를 보였습니다. 추론 오류의 주요 원인은 주요 긍정 이력을 무시하거나, 잘못된 의료 데이터를 해석하는 등의 것으로 나타났습니다.



### Assessing Large Language Models in Agentic Multilingual National Bias (https://arxiv.org/abs/2502.17945)
Comments:
          13 pages

- **What's New**: 본 연구는 다국어 자연어 처리의 최신 모델들이 복잡한 의사결정 작업을 수행하는 과정에서 어떻게 정치적, 국적적 편향을 드러내는지를 최초로 분석합니다. 연구는 대학 지원, 여행, 이주와 같은 세 가지 주요 시나리오를 통해 LLM의 적용 가능성과 역량을 테스트하며, 상이한 언어와 문화적 맥락에서의 편향 양상을 탐구합니다. 이를 통해 LLM이 특정 집단에 대한 편견을 강화하거나 특정 국가에 유리하도록 작용할 수 있음을 보입니다.

- **Technical Details**: 연구 방법론은 다국어 LLM의 추천 시스템에서 국적 편향을 포괄적인 평가 문제로 재구성하는 것을 포함합니다. 구체적으로, 다양한 언어 맥락에서 동일한 개체에 대한 LLM 평가를 실시하며, 편향의 차원을 수량화합니다. 대학과 도시, 여행지에 대한 평가에서의 편향을 주목하여 실험하는데, 이 과정에서 Chain-of-Thought (CoT) 전략이 편향에 미치는 영향을 탐구합니다.

- **Performance Highlights**: 연구 결과에서는 GPT-4와 Sonnet 모델이 영어 기반 국가에 대해 더 낮은 편향을 보였으나, 비영어권에서는 여전히 높은 편향을 나타내었습니다. 이는 LLM의 추천 결과의 공정성과 일관성에 부정적인 영향을 미치는 것으로 나타났습니다. 또한, 사용자 인구통계(성별, 언어 그룹)와 CoT 전략이 편향 패턴에 상당한 영향을 미친다는 것이 발견되어, 다국어 편향 완화 전략의 필요성을 강조합니다.



### CaseGen: A Benchmark for Multi-Stage Legal Case Documents Generation (https://arxiv.org/abs/2502.17943)
Comments:
          18 pages

- **What's New**: 본 연구에서는 중국 법률 분야에서 다단계 법적 문서 생성을 위한 벤치마크인 CaseGen을 처음으로 소개합니다. CaseGen은 법률 전문가에 의해 주석이 달린 500개의 실제 사례 샘플을 기반으로 하며, 7개의 필수 섹션으로 구성되어 있습니다. 이 시스템은 방어 진술서 작성, 재판 사실 기술, 법적 추론 구성, 판결 결과 생성 등 네 가지 주요 작업을 지원합니다. 이를 통해 LLM의 법적 문서 생성을 평가할 수 있는 포괄적인 플랫폼을 제공합니다.

- **Technical Details**: CaseGen은 다단계 생성 작업을 설계하여 법적 문서의 구조 및 작성 과정을 따릅니다. 각 작업(방어 진술서, 재판 사실, 법적 추론, 판결 결과)은 고유의 작성 논리와 평가 기준을 가지고 있어 더 세밀하고 미세한 평가가 가능합니다. 또한 LLM-as-a-judge 평가 프레임워크를 통해 자동화된 평가 방식을 채택하였으며, 이를 통해 효율성을 극대화할 수 있습니다. CaseGen은 LLM의 성능 할 수 있는 정확하고 체계적인 평가 도구를 제공합니다.

- **Performance Highlights**: 시스템 평가 결과, 현재 사용되는 LLM들은 법적 문서 생성에서 만족스럽지 않은 성능을 보였습니다. 이 논문은 다양한 상업적 및 오픈 소스 LLM을 체계적으로 평가 하여 이들의 한계점을 강조하고 있습니다. 또한 인간의 주석과 LLM의 평가 결과가 밀접하게 일치함을 보여주며, 앞으로의 개발 방향과 기회를 제시합니다. 주요 결과로는 기존 LLM들이 필요로 하는 법적인 전문성과 정확성을 충족하지 못하는 점이 강조됩니다.



### Advantage-Guided Distillation for Preference Alignment in Small Language Models (https://arxiv.org/abs/2502.17927)
Comments:
          Accepted by ICLR 2025(spotlight)

- **What's New**: 이번 연구에서는 Small Language Models (SLMs)의 alignment 문제를 해결하기 위해 Dual-Constrained Knowledge Distillation (DCKD)와 Advantage-Guided Distillation for Preference Alignment (ADPA)를 제안합니다. 이들은 정렬된 Teacher 모델에서 SLM으로의 지식을 효과적으로 전달하며, 학생 모델이 인간의 선호를 더 잘 반영하도록 돕습니다.

- **Technical Details**: DCKD는 알려진 Teacher와의 Knowledge Distillation 과정에 두 개의 KL-divergence 제약 조건을 추가하여, 선호하는 반응과 비선호하는 반응 모두에서 Teacher의 예측 행동을 학생 모델이 캡처하도록 합니다. ADPA는 보다 정교한 선호 정렬 메커니즘을 도입하여, 학생 모델이 세밀한 보상 신호를 활용하여 훈련 중에 Teacher의 지침을 효과적으로 따르도록 합니다.

- **Performance Highlights**: 실험 결과, DCKD와 ADPA 모두 SLM의 alignment 효과를 크게 향상시키고, 큰 모델들과의 성능 차이를 줄이는 데 기여했습니다. 특히 ADPA는 DCKD와 통합하여 성능을 더욱 높이며, SLM들이 인간의 선호를 보다 효과적으로 포착하게 만드는 성과를 보여줍니다.



### FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models (https://arxiv.org/abs/2502.17924)
- **What's New**: 이 논문은 FACT-AUDIT라는 새로운 평가 프레임워크를 소개하며, 이를 통해 대규모 언어 모델(LLM)의 사실 확인 능력을 동적으로 감사합니다. 기존 자동화된 사실 확인 평가 방법은 정적인 데이터셋과 분류 지표에 의존하여 한계가 있었지만, FACT-AUDIT는 모델의 반응에 따라 평가를 업데이트하는 방식으로 이러한 문제를 해결합니다. 중요 샘플링(importance sampling) 원칙과 다중 에이전트 협력을 활용하여 적응 가능하고 확장 가능한 데이터셋을 생성합니다.

- **Technical Details**: FACT-AUDIT는 두 가지 주요 특징에 중점을 둡니다: (1) 동적으로 업데이트되는 사실 확인 테스트 데이터 및 (2) 모델 생성 정당성의 심층 평가입니다. 이 프레임워크는 몬테카를로 샘플링 프로세스를 사용하여 실제 세계의 오라클 지식 공간에서 샘플링된 테스트 케이스를 생성합니다. 또한, 중요 샘플링 기법을 적용하여 LLM의 약점을 파악하고, 새로운 테스트 사례를 생성하는 반복적인 탐색 프로세스를 통해 LLM을 평가합니다.

- **Performance Highlights**: 광범위한 실험을 통해 FACT-AUDIT는 최신 LLM 간의 차별화를 효과적으로 수행하며, 사실 확인 성능에 대한 귀중한 통찰력을 제공합니다. 이 연구는 모델의 강점과 한계를 드러내어 LLM의 사실 확인 분석에서 중요한 기여를 합니다. 특히, FACT-AUDIT는 정당성 생성(justification production)과 판결 예측(verdict prediction)을 통합하여 전통적인 정확성 기반 자동 평가를 넘어서며, 보다 종합적인 평가를 제공합니다.



### Scaling LLM Pre-training with Vocabulary Curriculum (https://arxiv.org/abs/2502.17910)
Comments:
          under review

- **What's New**: 현대 언어 모델은 고정된 어휘(vocabulary)에 기반해 훈련되며, 이는 인간 언어 학습에서 관찰되는 적응형 어휘 습득과 대조됩니다. 이를 해결하기 위해 본 연구에서는 어휘 커리큘럼 학습(vocabulary curriculum learning) 접근법을 도입하여 훈련 효율성을 향상시킵니다. 이 방법은 엔트로피(entropy) 기반의 어휘 확장과 모델 최적화를 번갈아 수행함으로써 다양한 토큰화(tokenization) 세분화의 전이 가능한 표현(transferable representations)을 학습할 수 있게 합니다.

- **Technical Details**: 어휘 커리큘럼 학습은 기본 단위(문자)로 시작하여 점진적으로 더 복잡한 표현으로 확장되며, 높은 모델링 엔트로피(regions of high modeling entropy) 영역에 더 많은 용량을 할당하여 복잡한 언어 구조를 이해합니다. 기존 언어 모델의 한계를 극복하기 위해, 본 연구에서는 동적인 어휘 생성을 위한 프레임워크를 제안하고 이 과정에서 예측 가능성에 기반한 토큰 시퀀스를 병합(merge)하는 방식을 사용합니다. 또한, 어휘 업데이트는 목표 어휘 크기를 기준으로 증가 또는 감소할 수 있는 구조로 구성됩니다.

- **Performance Highlights**: 실험은 소규모 GPT 모델에서 수행되었으며, 어휘 커리큘럼 학습 접근법이 기존 고정형 어휘 훈련과 비교하여 낮은 bits-per-character(BPC) 성과를 낼 수 있음을 보여줍니다. 실험 결과, 어휘 커리큘럼을 통해 더 큰 어휘를 효과적으로 활용할 수 있으며, 이 과정에서 모델의 성능이 향상됨을 확인했습니다. 이러한 연구는 언어 모델링에 집중하지만, 점진적인 스케일링 효과가 다른 모달리티(modalities)와 도메인에도 일반화될 수 있을 것으로 기대됩니다.



### Can Large Language Models Identify Implicit Suicidal Ideation? An Empirical Evaluation (https://arxiv.org/abs/2502.17899)
- **What's New**: 이번 연구에서는 자살 예방을 위한 대규모 언어 모델(LLMs)의 역량을 평가하기 위한 포괄적인 평가 프레임워크를 소개합니다. 두 가지 중요한 측면인 암시적 자살 사상 식별(Identification of Implicit Suicidal ideation, IIS)과 적절한 지원 응답 제공(Provision of Appropriate Supportive responses, PAS)에 중점을 두고 있습니다. 이를 위해 1,308개의 테스트 사례로 구성된 새로운 데이터셋 DeepSuiMind를 개발하였습니다.

- **Technical Details**: DeepSuiMind 데이터셋은 D/S-IAT와 Negative Automatic Thinking 같은 심리적 프레임워크를 기반으로 하여 암시적 자살 사상을 평가할 수 있도록 설계되었습니다. 이 데이터셋은 다양한 표현과 맥락에서 암시적 자살 사상을 포괄적으로 다루며, LLM의 대응 능력을 평가하는 새로운 방법론을 제시합니다. 연구에서는 '경고 없음(No warning)'과 '경고 있음(Warning)'의 두 가지 실험 환경에서 8개의 널리 사용되는 LLM을 평가하였습니다.

- **Performance Highlights**: 결과적으로, LLM은 암시적 자살 사상을 정확하게 식별하는 데 있어 상당한 어려움을 겪는 것으로 나타났습니다. 이러한 결과는 정신 건강 맥락에서 LLM의 적용에 있어 홍보되는 기술적 한계를 강조하며, 보다 정교한 접근 방식이 필요하다는 점을 보여줍니다. LLM은 명시적 자살 관련 게시물에는 적절하게 대응할 수 있지만, 암시적 사상을 식별하는 데 있어서는 종종 상황을 악화시키거나 자해를 조장하는 방향으로 나아가기도 합니다.



### RankCoT: Refining Knowledge for Retrieval-Augmented Generation through Ranking Chain-of-Thoughts (https://arxiv.org/abs/2502.17888)
- **What's New**: 이 논문에서는 RankCoT라는 새로운 지식 정제 방법을 소개합니다. 이 방법은 외부 문서에 대한 reranking 신호를 통해 Chain-of-Thought (CoT) 기반 요약 생성에서 지식을 정제하는 데 초점을 맞춥니다. 또한, RankCoT는 자가 반성 메커니즘을 도입하여 CoT 출력을 더욱 개선합니다. 이러한 방식으로 RankCoT는 LLMs가 더 정확한 응답을 생성하도록 지원합니다.

- **Technical Details**: RankCoT는 Retrieval-Augmented Generation (RAG) 시스템에서 LLM을 최적화하는 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 쿼리와 문서 세트를 제공하여 LLM이 여러 개의 CoT 응답을 생성하도록 유도합니다. 두 번째 단계에서는 이러한 CoT 후보 중에서 최상의 출력을 유도하는 방법으로 LLM을 미세 조정하여, 의미 있는 요약 결과를 도출하는 데 중점을 둡니다. RankCoT는 모든 반환된 문서를 다루면서 비관련 문서를 필터링하여 CoT 스타일 요약을 생성합니다.

- **Performance Highlights**: 실험 결과, RankCoT는 모든 기준 모델보다 2% 이상의 향상된 성능을 보였습니다. 모든 규모의 LLM에서 효과적인 결과를 증명하며, 기존 reranking 및 요약 방법에 비해 더 짧지만 효과적인 정제 결과를 산출합니다. 또한, RankCoT는 반환된 문서에서 중요한 의미를 추출하고 내부 지식과 반환된 내용 간의 충돌을 완화합니다.



### Towards Enhanced Immersion and Agency for LLM-based Interactive Drama (https://arxiv.org/abs/2502.17878)
- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 한 인터랙티브 드라마(Interactive Drama)를 제안하며, 사용자가 이야기의 캐릭터로서 등장하여 LLM 에이전트와 대화하고 이야기를 경험할 수 있는 새로운 AI 기반의 대화 시나리오를 소개합니다. 특히, 몰입감(Immersion)과 주체성(Agency)이라는 두 가지 중요한 요소를 통해 사용자들이 즐거운 인터랙티브 경험을 할 수 있도록 연구하고 있습니다. 지난 연구들이 소홀히 여긴 이 두 가지 측면을 분석하고 개선하기 위한 방법론을 제안합니다.

- **Technical Details**: 본 연구에서는 드라마 대본을 기반으로 LLM 에이전트가 캐릭터를 시뮬레이션하여 대화를 진행합니다. 이를 통해 사용자는 scripted plot에 구애받지 않고 유연한 상호작용을 경험할 수 있습니다. 또한, 논문에서는 Playwriting-guided Generation 및 Plot-based Reflection이라는 두 가지 방법론을 제시하여 LLM의 대화 품질을 향상시키고, 플레이어의 의도에 맞추어 에이전트의 반응을 조정할 수 있는 방법을 소개합니다.

- **Performance Highlights**: 평가는 인간의 판단에 의존하여 몰입감과 주체성을 측정합니다. LLMs의 스토리텔링 품질이 크게 향상된 것으로 나타났으며, 사용자들이 스스로 만든 캐릭터로 이야기에 몰입할 수 있는 가능성이 증가했습니다. 이 연구는 드라마 캐릭터의 동적 발전을 통해 사용자와의 감정적 연결을 심화시키고, 더 나은 상호작용 경험을 제공하고자 합니다.



### SYNTHEMPATHY: A Scalable Empathy Corpus Generated Using LLMs Without Any Crowdsourcing (https://arxiv.org/abs/2502.17857)
Comments:
          10 pages

- **What's New**: 본 논문에서는 SYNTHEMPATHY라는 새로운 대화 데이터셋을 제안합니다. 기존의 동정적인 대화 데이터를 생성하기 위한 기존 방법론의 한계를 극복하고, 대화 시스템의 공감 능력을 강화하기 위한 새로운 데이터 생성 프레임워크를 소개합니다. 이 프레임워크는 105,578개의 독창적인 공감 응답을 포함하여, 대형 언어 모델(LLM) 생성 과정을 활용합니다.

- **Technical Details**: SYNTHEMPATHY 데이터셋은 LLM의 창의적인 가능성을 활용하여 구성된 스토리 생성, 설명 다시 쓰기 및 공감적 응답이라는 세 단계를 통해 제작됩니다. 생성된 데이터는 모델 다양성을 증대시키기 위해 Llama 2, Llama 3 및 Gemma와 같은 다양한 LLM을 활용하여 생산됩니다. 이 과정에서 정확한 중복 제거 및 무례한 언어의 제거를 포함한 단계적인 접근 방식이 적용됩니다.

- **Performance Highlights**: Mistral 7B 모델이 SYNTHEMPATHY 데이터셋에 대해 fine-tuning된 결과, 평균적인 공감 점수가 개선되었습니다. 이는 공감 능력의 향상뿐만 아니라, 대화 시스템의 신뢰성과 친밀감을 증가시킬 수 있는 가능성을 시사합니다. 데이터셋과 코드는 최종 논문이 발표될 때 공개될 예정입니다.



### LR${}^{2}$Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems (https://arxiv.org/abs/2502.17848)
- **What's New**: 최근 o1-like 모델에서의 발전은 대형 언어 모델(LLMs)의 추론 능력을 강화하여 반성적(reflection) 기능을 통해 복잡한 작업을 수행할 수 있도록 하고 있습니다. 그러나 이러한 반성적 능력을 효과적으로 평가하는 것이 적절한 벤치마크의 부족으로 인해 어렵습니다. 이를 해결하기 위해 우리는 LRs2Bench라는 새로운 벤치마크를 소개합니다.

- **Technical Details**: LRs2Bench는 반성적(reasoning) 추론 능력을 평가하기 위해 850개의 샘플로 구성되어 있으며, 여섯 가지 제약 만족 문제(Constraint Satisfaction Problems, CSPs)를 포함하고 있습니다. 각 작업 유형은 지식 기반, 논리적, 공간적 제약과 같은 다양한 constraint 패턴에 중점을 두어 다채로운 문제 해결 시나리오에 대한 종합적인 평가를 제공합니다. 이를 통해 반성적 사고를 필요로 하는 문제를 해결하는 능력을 측정할 수 있습니다.

- **Performance Highlights**: 실험 결과, DeepSeek-R1와 OpenAI o1-preview와 같은 최첨단 추론 특정 모델조차도 LRs2Bench에서 평균 Exact Match 점수가 각각 20.0%와 23.6%에 불과하여 이러한 작업에서 어려움을 겪고 있는 것으로 나타났습니다. 이 결과는 현재 LLM의 반성적(reasoning) 추론 능력에 상당한 개선 여지가 있음을 강조합니다. 우리의 벤치마크 리더보드는 해당 링크에서 확인할 수 있습니다.



### Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation (https://arxiv.org/abs/2502.17839)
Comments:
          16 pages, 2 figures

- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 프레임워크에 실용적인 원칙을 주입하여 검색된 컨텍스트의 유용성을 향상시키는 간단하고 비지도 학습 방식의 방법을 제안합니다. 이 방법은 RAG에 의해 검색된 문서 풀에서 질문과 가장 관련이 높은 문장을 식별하고, 입력 질문에서 다루어진 모든 주제를 포함하되 그 이상은 포함하지 않도록 하며, 이러한 문장을 원래의 맥락 내에서 강조 표시한 후 LLM에 제공합니다. 실험을 통해, 이 접근 방식이 세 가지 질문 답변 작업(ARC-Challenge, PubHealth, PopQA)에서 일관된 개선 효과를 보여준다는 사실을 입증하였습니다.

- **Technical Details**: RAG는 큰 언어 모델(LLMs)의 제한된 지식 수평을 해결하기 위해 등장하였습니다. 본 연구는 RAG가 종종 LLM에 너무 많은 정보를 제시하여, 이를 통해 효과적인 소통을 위한 Grice의 네 가지 격률을 위반하는 경우가 많다고 주장합니다. 제안된 방법은 비지도 학습 기반의 휴리스틱을 사용하여 Grice의 격률을 구현하고, RAG에서 주제에 맞는 문장들을 식별함으로써 성능을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PubHealth에서 19.7% 및 ARC-Challenge에서 10%의 상대적인 정확도 향상을 달성하였습니다. 제안된 접근 방식은 또한 작은 언어 모델인 AMD-OLMo-1B-SFT에서 10%의 개선 효과를 보였으며, RAG의 성능 향상에 있어 상용화된 검색기인 DPR과 함께 사용될 때 20%까지 성능을 개선할 수 있는 가능성을 제시합니다.



### Predicting Through Generation: Why Generation Is Better for Prediction (https://arxiv.org/abs/2502.17817)
Comments:
          Preprint paper

- **What's New**: 이번 논문은 output token 생성을 pooled representation보다 더 효과적인 예측 방법으로 제안합니다. LLMs가 massive text corpora에서 next-token prediction을 통해 훈련되었기 때문에, generation은 자연스럽게 이들의 학습 패러다임과 일치합니다. 또한, 저자들은 Data Processing Inequality (DPI)를 사용하여 이 주장을 이론적으로 증명하고, empirical evidence를 통해 token-level generation이 더 많은 mutual information을 유지한다는 것을 보여줍니다.

- **Technical Details**: 저자들은 exposure bias와 format mismatch와 같은 두 가지 주요 도전에 대한 해결책으로 PredGen(Predicting Through Generating)이라는 프레임워크를 소개합니다. 이 프레임워크는 scheduled sampling을 사용하여 exposure bias를 줄이고, task adapter를 통해 생성된 token을 구조화된 출력으로 변환합니다. 또한, Writer-Director Alignment Loss (WDAL)라는 특수 손실 함수를 도입하여 token generation과 최종 작업 예측 간의 일관성을 보장합니다.

- **Performance Highlights**: PredGen은 다양한 회귀 및 분류 벤치마크에서 평가되어, pooled representations 또는 전통적인 생성 접근법을 사용하는 기준선 모델보다 일관되게 우수한 성능을 나타냈습니다. 이 연구에서는 구조화된 예측 작업에서의 효과를 보여주며, generation 기반 훈련이 기존 분류 기반 예측 방법보다 우수하다는 다양한 증거를 제공합니다.



### Can Multimodal LLMs Perform Time Series Anomaly Detection? (https://arxiv.org/abs/2502.17812)
Comments:
          9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection of multimodal LLMs and time series analysis are on the website this https URL

- **What's New**: 이 논문은 다중 모달 대형 언어 모델(MLLMs)이 시계열 이상 탐지(TSAD)에 적합한지 조사합니다. 이를 위해, 시간을 나타내는 수치 데이터를 이미지 형태로 변환하여 MLLMs와 함께 사용하는 새로운 벤치마크인 VisualTimeAnomaly를 제안합니다. 연구 결과, MLLMs는 초기 단일 변수 사례에서 시작하여 다변량 및 불규칙 시계열을 포함하는 보다 복잡한 시나리오로 확장된다는 점에서 중요합니다. 이 논문은 MLLMs를 TSAD에 적용한 첫 번째 종합 연구로, 관련 데이터와 코드를 공개하여 추가 연구를 촉진합니다.

- **Technical Details**: VisualTimeAnomaly 데이터셋은 3가지 현실 세계 시나리오 및 3가지 이상 granularities로 구성된 12,400개의 시계열 이미지로 실험을 진행합니다. 이 데이터는 여러 MLLMs 모델 (GPT-4o, Gemini-1.5, LLaVA-NeXT, Qwen2-VL)에 공급되어 다목적 테스트를 가능하게 합니다. 논문은 초기 단일 변수에서 복잡한 다변량 및 불규칙 시계열로의 이동을 통해 MLLMs의 성능을 포괄적으로 평가합니다. 모델 간 성능 차이는 다변량 이상 탐지에서 두드러지게 나타납니다.

- **Performance Highlights**: MLLMs는 점 단위 이상의 이상(즉, 범위 및 변수 기반 이상)을 더 효과적으로 탐지하고, 25%의 데이터가 결측될 경우에도 높은 강건성을 보였습니다. 오픈 소스 MLLMs는 단일 변수 시나리오에서 뛰어난 성능을 발휘하며, proprietary 모델과 비교해도 TSAD에서 유사한 결과를 나타냅니다. 이러한 결과는 MLLMs가 인지 패턴과 관계를 시각적으로 분석하는 데 유리함을 나타내며, 앞으로의 연구에서 이점을 제공할 수 있습니다.



### URO-Bench: A Comprehensive Benchmark for End-to-End Spoken Dialogue Models (https://arxiv.org/abs/2502.17810)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전에 힘입어 종단 간 대화 모델(Spoken Dialogue Models, SDMs)이 큰 발전을 이루었습니다. 그러나 텍스트 기반 LLMs와 비교했을 때 음성 관련 평가가 부족하며, 이는 paralinguistic 정보와 음성 품질 등을 포함해야 합니다. 이러한 평가의 격차를 해소하기 위해 우리는 URO-Bench라는 광범위한 벤치마크를 제안합니다.

- **Technical Details**: URO-Bench는 다국어(multi-lingualism), 다회차 대화(multi-round dialogues), 그리고 paralinguistics에 대한 평가를 포함한 최초의 S2S 벤치마크입니다. 이 벤치마크는 모델의 이해(Understanding), 추론(Reasoning), 및 구술 대화(Oral conversation) 능력을 평가하는 기본 트랙(basic track)과 프로 트랙(pro track)으로 나뉘어 있으며, 각각 16개와 20개의 데이터셋으로 구성되어 있습니다.

- **Performance Highlights**: 제안한 벤치마크에 대한 평가 결과, 현재 오픈소스 SDMs는 일상적인 QA 작업에서 꽤 괜찮은 성능을 보여주지만, 지시 따르기 능력에서는 그들의 기반 LLMs에 비해 떨어지며, catastrophic forgetting 문제도 보입니다. 또한, paralinguistic 정보 및 오디오 이해에 대한 고급 평가에서 성능이 낮아, 이 분야에 대한 추가 연구의 필요성을 강조합니다.



### Your Language Model May Think Too Rigidly: Achieving Reasoning Consistency with Symmetry-Enhanced Training (https://arxiv.org/abs/2502.17800)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 성능을 향상시키기 위한 새로운 데이터 증강 방법인 syMmetry-ENhanceD (MEND)를 제안합니다. 기존 방법들이 주로 추론 체인의 증강에 초점을 맞춘 반면, MEND는 쿼리 변형의 대칭성을 인식하여 지식 추출 단계에서 모델 강건성을 개선하는 데 중점을 둡니다. 이러한 접근법은 데이터 효율적인 훈련과 아웃 오브 디스트리뷰션(Out-of-Distribution) 설정에 대한 일반화 능력을 향상시키는 데 기여합니다.

- **Technical Details**: MEND는 다양한 쿼리 변형에서 모델의 정보 추출 능력을 개선하기 위해 쿼리 증강(query augmentation) 기법을 사용합니다. 이 방법은 LLM이 기본적인 의미를 유지하는 쿼리의 미세한 변화를 인식하고 대응할 수 있도록 돕습니다. 논문은 논리적, 산술적 추론 과제에서 MEND의 효과를 증명하기 위해 포괄적인 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, MEND는 다양한 쿼리 변형에 걸쳐 추론 성능을 향상시켰습니다. 이를 통해 LLM의 강건성을 구조화된 데이터셋(curated dataset) 관리 방식을 통해 개선하는 새로운 통찰을 제공하게 됩니다. 이러한 접근은 LLM의 전반적인 성능을 높여 주며, 다양한 실세계 적용에 유용한 결과를 보여줍니다.



### Enhancing Human Evaluation in Machine Translation with Comparative Judgmen (https://arxiv.org/abs/2502.17797)
Comments:
          Preprint, 15 pages

- **What's New**: 이 연구에서는 기계 번역(Machine Translation, MT)에서 사람 주석(annotation)에 비교 판단(comparative judgment)을 통합하는 방법을 탐구하고, 세 가지 주석 설정(point-wise MQM, side-by-side MQM, SxS relative ranking, RR)을 평가합니다. SxS MQM은 앞서의 MQM을 확장하여 두 개의 번역을 쌍으로 비교하여 오류를 주석할 수 있도록 하였고, SxS RR은 오류 주석 없이 우수한 출력을 선택하는 데 집중합니다. 연구 결과, SxS 설정이 MQM보다 더 높은 주석자 간 동의도를 달성하고, 오류 주석의 일관성을 향상시키는 것으로 확인되었습니다.

- **Technical Details**: 이 연구에서는 세 가지 주석 방법론을 비교 분석합니다: 첫째, 포인트 단위의 MQM 방법으로, 이는 각 번역을 개별적으로 평가하고 오류의 범위와 심각도를 식별하는 것입니다. 둘째, SxS MQM 방법으로, 이를 통해 두 개의 번역을 나란히 비교하여 오류를 세분화하여 주석합니다. 셋째, SxS RR 방법은 두 개의 번역을 나란히 보여주고 어느 것이 더 우수한지를 평가하게 하여 상세한 오류 주석 없이도 유용한 평가를 가능하게 합니다.

- **Performance Highlights**: SxS MQM은 마주보는 방식으로 진행된 주석 작업에서 평균 38.5%의 오류 주석 일관성을 높여 주었고, SxS RR은 오류 주석 없이도 상당한 비용 효율성을 제공한다고 합니다. 모든 주석 설정은 시스템 순위를 안정적으로 반환하고, SxS RR은 비용 측면에서 MQM보다 약 1/3 가량 절감하는 효과가 있다고 보고합니다. 이 연구는 앞으로 발표될 세 가지 주석 데이터셋을 제공하여 추가적인 연구를 촉진할 것입니다.



### AIR: Complex Instruction Generation via Automatic Iterative Refinemen (https://arxiv.org/abs/2502.17787)
Comments:
          The first three authors contributed equally, 20 pages

- **What's New**: 이 논문에서는 복잡한 지침을 생성하는 새로운 자동 반복 정제 프레임워크인 AIR(Automatic Iterative Refinement)를 제안합니다. 기존의 방법들은 복잡한 지침을 성공적으로 생성하는 데 어려움을 겪고 있으며, 주로 단순 지침을 바탕으로 하거나 인적 자원이 많이 소모되는 방식입니다. AIR 프레임워크는 실제 요구 사항을 더 잘 반영하며, LLM(대형 언어 모델)의 복잡한 지침을 따르는 능력을 향상시킵니다.

- **Technical Details**: AIR 프레임워크는 두 단계로 구성됩니다: 첫 번째 단계는 문서에서 초기 지침을 생성하는 것이고, 두 번째 단계는 LLM-as-judge의 도움을 받아 지침을 반복적으로 정제하는 것입니다. 여기서 각 지침의 품질은 점수화되어 필터링되며, 복잡한 지침 생성을 위한 필수 원칙을 고려합니다. 백 번역(back-translation) 기법을 사용하여 기존 문서에서 새로운 지침을 생성합니다.

- **Performance Highlights**: AIR-10K 데이터셋은 10,000개의 복잡한 지침으로 구성되어 있으며, 제안된 방법으로 생성된 지침은 기존 방법들보다 모델의 복잡한 지침을 따르는 능력을 현저히 향상시킵니다. 실험 결과, AIR 프레임워크가 제공하는 정제 과정이 성능 개선에 크게 기여함을 보여줍니다. 이를 통해 복잡한 지침 생성을 위한 새로운 접근 방식을 제시합니다.



### Exploring the Potential of Large Language Models for Estimating the Reading Comprehension Question Difficulty (https://arxiv.org/abs/2502.17785)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 전통적인 방식으로 읽기 이해 질문의 난이도를 평가하는 데 필요한 대규모 인적 주석과 테스트가 까다로운 문제를 해결하기 위해, OpenAI의 GPT-4o와 o1와 같은 대형 언어 모델(LLMs)을 활용하는 방안을 제안합니다. 연구팀은 Study Aid and Reading Assessment (SARA) 데이터셋을 이용하여 LLM의 질문 난이도 추정 성능을 평가하고, 기존의 Item Response Theory (IRT)와 비교하였습니다. 그 결과, LLM은 IRT에서 유도한 파라미터와 의미 있게 일치하는 난이도 추정치를 제공하였지만, 특정 항목의 극단적 특성에 대한 민감도가 다르다는 점이 관찰되었습니다.

- **Technical Details**: 연구에 사용된 SARA 데이터셋은 다섯 개의 하위 테스트로 구성되어 있으며 각 하위 테스트는 핵심 읽기 기술을 목표로 하고 있습니다. 질문의 난이도를 추정하기 위해 IRT 분석과 함께 OpenAI의 GPT-4o와 o1 모델을 사용하였습니다. 특히, LLM을 통한 질문 난이도의 자동화된 평가 방법을 제안하며, 전통적인 심리측정 기법과 현대적인 AI 기반 방법의 융합이 이루어지고 있습니다.

- **Performance Highlights**: 연구 결과 LLM은 질문 난이도의 추정에 있어 기존 IRT 분석과의 유의미한 상관관계를 나타내었습니다. 또한, LLM은 더 동적인 교육 평가 시스템의 개발로 이어질 수 있는 가능성을 보여줍니다. 이러한 접근법은 맞춤형 학습 시스템의 설계와 교육 평가의 평등성을 증대시키기 위한 새로운 경로를 제시합니다.



### FoREST: Frame of Reference Evaluation in Spatial Reasoning Tasks (https://arxiv.org/abs/2502.17775)
Comments:
          9 pages

- **What's New**: 이 논문에서는 공간 추론에서 중요한 개념인 Frame of Reference (FoR)에 대한 이해를 개선하기 위해 FoREST 벤치마크를 소개합니다. 기존의 AI 모델이 FoR를 제대로 평가하지 못했던 점을 지적하며, 새로운 평가 체계를 통해 LLM의 공간적 이해능력을 체계적으로 분석할 수 있도록 합니다. 이 연구는 인간의 인지 기능과 유사한 방식으로 AI가 FoR를 이해해야 하는 필요성을 강조합니다.

- **Technical Details**: 저자들은 FoR을 구성하는 세 가지 주요 클래스, 즉 상대적(relative), 고유한(intrinsic), 절대적인(absolute) 이해를 기반으로 LLM을 평가합니다. 연구에서는 FoR의 모호성을 다루기 위한 QA 설정을 도입하고, 해당 설정에서 LLM의 응답 및 텍스트-이미지 모델의 레이아웃 생성을 평가합니다. 공간 관계를 명시하는 Spatial-Guided prompting 방법론을 제안하여 LLM의 방향, 위상(topological), 거리 관계 등을 아우르는 공간 정보를 추출하도록 합니다.

- **Performance Highlights**: 실험 결과, 다양한 FoR 클래스 간의 성능 차이를 발견하여 LLM이 특정 FoR에 유향(bias)됨을 보여주었습니다. 특히, 레이아웃-확산 모델에서 LLM이 생성한 레이아웃이 이미지 생성 파이프라인에서 중요하게 작용함을 강조합니다. Spatial-Guided prompting을 통한 성능 향상이 텍스트-이미지 생성 성능을 효과적으로 개선시키는 결과를 나타냈습니다.



### LLM Inference Acceleration via Efficient Operation Fusion (https://arxiv.org/abs/2502.17728)
- **What's New**: 최근 Transformer 기반 대형 언어 모델(LLMs)의 빠른 발전은 계속 증가하는 매개변수 수와 밀접하게 연결되어 있습니다. 본 논문에서는 Softmax와 Layernorm의 계산 시 발생하는 오버헤드를 완전히 차단할 수 있는 매우 효율적인 기술을 제안합니다. 이 기술을 통해 선형 계층과 비선형 계층의 작업을 병렬로 처리하여 전체 지연 시간을 줄이는 데 중점을 둡니다.

- **Technical Details**: Transformer 아키텍처의 주요 도전 과제 중 하나는 비선형 변환이며, 이에는 마스크 연산이 포함됩니다. 특정 연산, 예를 들어 Softmax 및 Layernorm은 집합적 집계 연산을 요구하며, 이는 서로 다른 하드웨어 엔진에서 계산을 수행할 때의 지연 요인이 됩니다. 본 연구에서는 이러한 집합적 연산을 피하기 위해 비선형 연산을 선형 방식으로 결합하여 계산 성능을 극대화했습니다.

- **Performance Highlights**: 제안된 기법은 전체 추론 지연 시간을 하드웨어 아키텍처에 따라 약 15-20% 줄이는 것으로 나타났습니다. 이는 우수한 모델 성능을 유지하면서도 비선형 및 선형 연산의 최적화를 통해 달성되었습니다. 따라서 이 접근법은 대규모 LLM의 실제 배치와 확장에 중요한 기여를 할 것입니다.



### Spontaneous Giving and Calculated Greed in Language Models (https://arxiv.org/abs/2502.17720)
- **What's New**: 이번 연구에서는 언어 모델의 추론 능력이 사회적 지능에 어떻게 영향을 미치는지 분석하였습니다. 특히, 언어 모델이 공공재 게임과 같은 사회적 딜레마에서 협력 결정에 미치는 영향을 살펴보았으며, 다양한 경제 게임에서 비추론 모델과 추론 모델의 성능을 비교했습니다. 연구 결과, 추론 모델이 사회적 협력을 줄이고 개인의 합리적인 선택을 우선시하는 경향이 있다는 사실을 확인했습니다.

- **Technical Details**: 연구에서는 두 가지 추론 기법인 chain-of-thought와 reflection이 공공재 게임에서 GPT-4o 모델의 협력 결정에 미치는 영향을 평가했습니다. 실험 결과, chain-of-thought 기법을 사용했을 때 협력 비율이 60% 크게 감소하였으며, 반영(reflection) 능력을 사용할 경우에도 협력이 현저히 줄어드는 경향이 나타났습니다. 여러 경제 게임에서 비추론 모델과 추론 모델을 비교했을 때, 추론 모델이 협력 및 처벌 결정에서 비추론 모델보다 낮은 협력 비율과 처벌 비율을 보였습니다.

- **Performance Highlights**: 결과적으로, 전부 GPT-4o 모델로 구성된 그룹은 협력 수준이 매우 높게 유지되는 반면, 그룹 내 추론 o1 모델의 비율이 증가함에 따라 협력 수준이 꾸준히 감소하는 경향을 보였습니다. 특히, 전부 추론 o1 모델로 구성된 그룹은 평균적으로 약 20%의 협력만을 나타냈습니다. 이러한 결과는 AI 모델에서 사회적 지능을 통합할 필요성을 시사하며, 인류의 협력 직관을 방해하지 않도록 설계해야 한다는 점을 강조합니다.



### Knowledge Distillation with Training Wheels (https://arxiv.org/abs/2502.17717)
- **What's New**: 본 논문에서는 지식 증류(Knowledge Distillation)의 새로운 프레임워크를 제안하여, 학생 모델이 교사 모델의 도움을 받을 수 있도록 하며, 테스트 시 자연어 명령으로 테스트 규칙을 따르도록 학습합니다. 이를 통해 기존의 지식 증류 방법에서 벗어나, 테스트 시간에 학생 모델이 교사 모델을 효과적으로 활용할 수 있는 새로운 방법론을 제공합니다. 이 방법은 학생 모델이 학습 자료를 이해하는 것뿐만 아니라, 다양한 섹션의 상대적 난이도를 파악하여 교사의 도움을 요청할 수 있도록 합니다.

- **Technical Details**: 논문에서는 지식 증류를 엔트로피 정규화된 가치 최적화 문제로 공식화하고, 이를 Path Consistency Learning(PCL) 기법으로 해결하여 새로운 지식 증류 알고리즘을 제안합니다. 또한, 제한된 강화 학습(constrained reinforcement learning)을 활용하여 교사 모델을 테스트 시간의 기준으로 활용할 수 있는 프레임워크로 확장합니다. 이 방식은 학생 모델이 교사의 도움을 요청하고, 주어진 규칙에 따라 필요한 경우에만 교사를 사용할 수 있도록 합니다.

- **Performance Highlights**: 번역과 요약 작업을 통해 실험한 결과, 모델들은 교사의 사용 규칙을 잘 준수하며 특정 토큰에 대해 스스로 생성하고 교사의 도움을 요청할 토큰을 우선순위로 두는 경향을 보였습니다. 교사의 사용이 허용됨에 따라 출력 품질이 향상되는 경향을 관찰했으며, 본 접근 방식은 Speculative Decoding과는 또 다른 유연한 정확도와 효율성의 균형을 제공합니다.



### Bridging Information Gaps with Comprehensive Answers: Improving the Diversity and Informativeness of Follow-Up Questions (https://arxiv.org/abs/2502.17715)
Comments:
          8 pages, 2 figures, submitted to ACL 2025

- **What's New**: 이 연구는 기존의 대화형 시스템이 사용자가 노출되지 않은 정보를 끌어내기 위한 다양한 맥락에서의 후속 질문을 생성할 수 있도록 하는 새로운 방법을 제안합니다. 저자들은 가상의 LLM(large language model) 생성 '포괄적 답변'을 기반으로 하여 응답되지 않은 정보를 겨냥한 질문을 생성하는 방안을 모색하고, 실험 결과 이러한 방법이 질문의 질과 다양성을 크게 향상시킨다는 것을 입증했습니다.

- **Technical Details**: 이 연구에서는 GPT-4를 사용하여 포괄적 답변과 정보 격차 기반 후속 질문을 생성합니다. 이후 원래 FollowupQG 훈련 세트를 25,000개의 합성 예시로 증강하고, 이 데이터를 바탕으로 여러 언어 모델을 미세 조정했습니다. 초기 질문-답변 쌍에서 도출된 격차에 대한 분석을 통해 인간의 인지 전략을 활용하며, 기존의 전통적 질문 생성 방법과는 달리 Missing Information을 재조명하여 질문을 생성하는 접근법을 취하고 있습니다.

- **Performance Highlights**: 실험 결과, 증강된 데이터셋에서 훈련된 모델이 기본선에 비해 질문의 품질(유효성, 관련성, 정보성 등)과 다양성에서 눈에 띄는 개선을 보였습니다. 이 방식은 정보 탐색 대화의 모호성을 줄이고 LLM의 답변 정확도를 향상시키는 데 기여할 수 있는 가능성을 보여줍니다.



### Semantics drives analogical change in Germanic strong verb paradigms: a phylogenetic study (https://arxiv.org/abs/2502.17670)
- **What's New**: 이 연구는 과거 분사의 서사가 어떻게 독일어 강동사들의 모음 교체 패턴에 영향을 미치는지를 탐구합니다. 14개의 고대 및 현대 독일어에서 107개의 관련 동사를 분석하여, 과거와 과거 분사 간의 의미적 대립에서 모음 교체 패턴의 유지를 밝혀내는 새로운 계통 발생 모델을 사용합니다. 이 연구는 언어의 변화에 대한 기존 이론에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 phylogenetic comparative methods를 사용하여, 현재 완료(constructions)와 과거 시제 구조의 기능적 중첩이 동사의 모음 교체 패턴에 미치는 영향을 조사합니다. 강동사의 모음 교체 패턴에 대한 회귀 모델을 적용하여, 기능적 겹침이 과거 형태와 과거 분사 간의 특정 패턴을 유지하는 방식처럼 작용함을 발견하였습니다. 결과적으로, 특정 조건에서 강동사는 기존 패턴을 유지하는 경향이 있지만, 새로운 패턴을 발전시키지는 않는 비대칭적 관계를 보여줍니다.

- **Performance Highlights**: 연구 결과는 의미적 변화(semantic changes)와 기능적 변화(functional changes)가 형태론적 패턴에 미치는 영향을 지지하며, 이러한 변화의 방향성을 이해할 수 있는 정량적 기반을 제공합니다. 이 연구는 언어가 진화하는 의미적 및 구문적 구조에 적응하는 방법과 관련하여 중요한 통찰력을 제공하며, 비정규성의 관찰 분포에 대한 기존 이론과 대조되는 발견을 제시합니다.



### Towards Human Cognition: Visual Context Guides Syntactic Priming in Fusion-Encoded Models (https://arxiv.org/abs/2502.17669)
Comments:
          8 pages, 9 figures

- **What's New**: PRISMATIC(프리즈매틱)이라는 멀티모달(혹은 다중모드) 구조적 프라이밍(data set) 데이터셋을 처음으로 소개했습니다. 이 데이터셋은 특정한 이미지와 문장이 짝을 이루고 있어, 언어 모델이 가지는 구문적 구조에 대한 민감성을 체계적으로 평가할 수 있는 도구를 제공합니다. 또한, 미리 정의된 목표 문장 없이 프라이밍 효과를 평가할 수 있는 기준 없는 척도를 제안했습니다.

- **Technical Details**: PRISMATIC 데이터셋은 Flickr30k에서 파생된 4,208개의 문장과 1,710개의 정렬된 이미지를 포함하며, 각 이미지는 다수의 설명 문장과 특정 구문 구조를 라벨링합니다. 연구에서는 dual encoder와 fusion encoder라는 두 가지 아키텍처에 대한 실험을 통해 그들의 구조적 보존 능력을 비교하고, tree kernel algorithm(트리 커널 알고리즘)을 기반으로 한 새로운 평가 지표를 사용하여 프라이밍 효과를 정량화하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 두 가지 인코딩 방식 모두 구조적 프라이밍 효과를 나타내며, 특히 fusion encoding 모델은 프라이밍 효과와 시각적 유사성 간의 강한 상관관계를 보여 주었습니다. 이는 멀티모달 언어 모델(MLLM)이 인간의 심리언어적 패턴을 더 잘 반영하고 있음을 제시하며, 이들 모델이 어떻게 시각 정보가 구문적 선택에 영향을 미치는지를 이해하는 데 기여합니다.



### Towards Typologically Aware Rescoring to Mitigate Unfaithfulness in Lower-Resource Languages (https://arxiv.org/abs/2502.17664)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 연구에서는 자원이 제한된 언어에서 다국어 대형 언어 모델(LLMs)의 신뢰성 문제를 해결하기 위한 방법으로, 경량의 보조 모델(auxiliary model)을 사용하여 LLM의 출력을 재점수(rescore)하는 접근 방식을 제시합니다. 미세 조정 없이 700MB 미만의 데이터로 처음부터 훈련된 단일 언어 4층 BERT 모델이 모폴로지적 복잡성이 다른 베트남어, 폴란드어, 조지아어에서 평균 88.33%의 정확도로 신뢰성 있는 요약을 식별할 수 있다는 점을 보여줍니다.

- **Technical Details**: 본 연구에서는 신뢰성 관련 환상(hallucination)을 완화하기 위한 재점수 방식을 구체적으로 논의합니다. 이 과정에서 작은 모델이 큰 모델의 출력을 재배열하는 방식으로 신뢰성을 최적화하여 시스템의 출력이 보다 신뢰성 있게 되는 것을 목표로 합니다. 이 방법은 Speech Processing과 같은 다른 분야에서도 성공적으로 적용되어 왔으며, 특히 자원을 제한받는 언어에 적합하다는 장점이 있습니다.

- **Performance Highlights**: BERT의 표준 훈련 목표가 다양한 언어에서 효과적이라는 사실을 밝혀냈습니다. 또한, 단층 아키텍처가 다양한 작업에서 더 견고한 성능을 발휘하며, 복잡한 형태의 언어는 더 깊은 모델이 필요하다는 것을 시사합니다. 이러한 내용을 바탕으로 다음 후속 연구에 대한 방향성을 제시하며, 보조 모델을 활용한 신뢰성 개선이 자원 제한 언어에서 유용할 수 있음을 강조합니다.



### Evaluating the Effect of Retrieval Augmentation on Social Biases (https://arxiv.org/abs/2502.17611)
Comments:
          18 pages

- **What's New**: 이 논문에서는 Retrieval Augmented Generation(RAG)의 사회적 편향(social bias)에 대한 영향을 체계적으로 연구하였습니다. RAG 시스템의 다양한 구성 요소와 생성되는 텍스트 간의 관계를 분석하여, 성별, 인종, 나이 및 종교와 같은 네 가지 사회적 편향 유형을 평가하였습니다. 여러 언어(영어, 일본어, 중국어)에서의 RAG 응답에서 이러한 사회적 편향이 증폭되는 현상을 발견했습니다.

- **Technical Details**: RAG 시스템은 외부 문서 컬렉션을 기반으로 운영되며, 이는 사전 훈련된 LLM에서 접근할 수 없는 정보를 포함할 수 있습니다. 이 시스템에서 문서는 다양한 검색 방법(희소(sparse) 및 밀집(dense) 검색)을 통해 색인화되어 쿼리에 대해 가장 관련성 높은 문서를 효율적으로 검색합니다. 우리는 BBQ(Bias Question Answering) 벤치마크 데이터를 사용하여 이러한 문서의 사회적 편향이 RAG 시스템의 응답에 미치는 영향을 조사하였습니다.

- **Performance Highlights**: RAG 시스템에서 사용되는 고정편향(stereotypically biased) 문서들은 사회적 편향이 증폭되는 결과를 도출하게 됩니다. 대규모 LLM에서는 사회적 편향이 상대적으로 덜 증가하는 경향이 있음을 발견했고, 다양한 문서의 검색 수에 따라 사회적 편향이 반드시 증가하지는 않는다는 놀라운 결과를 제공했습니다. 이러한 연구는 RAG 시스템의 사회적 편향 평가 방법에 대해 재고할 필요성을 제기합니다.



### PICASO: Permutation-Invariant Context Composition with State Space Models (https://arxiv.org/abs/2502.17605)
Comments:
          Published in The Thirteenth International Conference on Learning Representations, ICLR 2025

- **What's New**: 본 논문에서는 Large Language Models (LLMs)에 retrieval 기능을 추가하여 외부 지식을 효과적으로 활용할 수 있는 방법을 제안합니다. 특히, 여러 개의 pre-computed 상태를 효율적으로 조합하여 고품질 출력을 생성하는 방법인 PICASO(Permutation-Invariant Compositional Aggregation of States as Observations)를 도입합니다. 이 방법은 기존의 concatenation 방식과 비교하여 평균 5.4배 빠르며 성능 향상도 보여줍니다.

- **Technical Details**: PICASO는 State Space Models (SSMs)을 활용하여 정보가 포함된 여러 맥락들을 효과적으로 조합합니다. SSM의 동작에서 유도된 수학적 관계를 이용하여 여러 상태를 하나로 합치는 방법을 개발하였고, 이를 통해 맥락의 순서가 결과에 미치는 영향을 최소화합니다. Dynamic Programming을 통해 상태를 평균화하여 계산의 효율성을 높이고, 다소 근사치를 허용함으로써 선형 시간으로 계산할 수 있습니다.

- **Performance Highlights**: 실험 결과, PICASO는 WikiText와 MSMARCO 데이터셋에서 91%의 성능 이득을 보였으며, 기존의 최강 성능을 가진 baseline과 동일한 성능을 유지하면서도 더 빠른 처리 속도를 자랑합니다. 또한, 사전 훈련된 Mamba-2 2.7B 모델을 사용하여 단 하루의 fine-tuning으로 concatenation과 동등한 성능을 나타냈습니다.



### MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inferenc (https://arxiv.org/abs/2502.17599)
Comments:
          NAACL 2025 Main

- **What's New**: 본 논문에서는 MEDA(Multimodal Attention Entropy-Guided Dynamic KV Cache Allocation)라는 새로운 다층 KV 캐시 할당 방법을 제안합니다. 이 방법은 크로스 모달 주의 엔트로피(cross-modal attention entropy)를 활용하여 각 층의 KV 캐시 크기를 결정하며, 정보 손실 없이 KV 쌍을 선택하고 병합하는 전략을 사용합니다. 이를 통해 기존의 정적 할당 방식보다 높은 효율성을 달성합니다.

- **Technical Details**: MEDA는 각 레이어의 주의 밀도(attention density) 변화를 고려하여 동적으로 KV 캐시 크기를 할당하는 방법입니다. 선택된 KV 쌍과 선택되지 않은 KV 쌍을 병합하여 전체 문맥의 정보를 유지합니다. 이러한 동적 할당 방식을 통해 KV 캐시의 메모리 사용 효율성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: MEDA는 기존의 멀티모달 LLM과 텍스트 기반 KV 캐시 방법들보다 우수한 성능을 보입니다. 최대 72%의 KV 캐시 메모리 절감과 2.82배 더 빠른 디코딩 속도를 기록하며, 다양한 멀티모달 작업에서도 성능을 유지하거나 개선했습니다. 이 코드는 저자들이 제공하는 링크를 통해 확인할 수 있습니다.



### Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility (https://arxiv.org/abs/2502.17591)
Comments:
          ICLR'25 Poster. Project page and code is available at this https URL

- **What's New**: 이 논문에서는 Proactive Privacy Amnesia (PPA)라는 혁신적인 기법을 제안하여 대규모 언어 모델(LLMs)에서 개인 식별 정보(PII)를 보호하면서도 모델의 유용성을 유지할 수 있도록 하였습니다. 기존 방법들이 PII 보호와 모델 기능성 간의 균형을 맞추기 어려운 문제를 해결하기 위해, PPA는 정보의 민감성을 분석하고 주요 기억을 선택적으로 잊는 방식으로 작동합니다. 이를 통해 PPA는 모델 공격에 대한 방어력을 강화하는 한편, 기존 LLM의 성능 저하를 최소화할 수 있게 합니다.

- **Technical Details**: PPA는 세 가지 구성 요소로 이루어져 있습니다: (1) Sensitivity Analysis: PII에서 핵심 요소를 식별, (2) Selective Forgetting: 핵심 요소만을 선택적으로 잊는 과정, (3) Memory Implanting: 선택적 망각으로 인한 성능 손실을 보완하기 위한 전략입니다. 주요 메모리를 잊은 후 적절한 대체 기억을 삽입함으로써 LLM의 기능성을 유지하는 방식입니다. 또한, LLaMA2와 LLaMA3 모델에 대한 평가를 통해 PII 공격에 대한 방어 성능을 입증하였습니다.

- **Performance Highlights**: PPA 기법은 전화번호 노출 위험을 100% 제거하고, 물리적 주소 노출 위험을 9.8%에서 87.6%로 감소시켰습니다. 예를 들어 Enron 이메일 실험에서, PPA는 중간 모델 유틸리티 방식에 비해 모델의 성능을 372.7% 향상시키면서도 동일한 방어 수준을 유지하였습니다. 또한 PPA는 물리적 주소 방어에서 260.0%의 성능 향상을 보여주며, 위험 점수를 26.2% 감소시키는 결과를 보였습니다.



### End-to-End Chart Summarization via Visual Chain-of-Thought in Vision-Language Models (https://arxiv.org/abs/2502.17589)
- **What's New**: 이번 논문에서 소개하는 End-to-End Visual Chain-of-Thought (V-CoT) 방법은 차트 요약을 위한 새로운 접근 방식으로, 대형 비전-언어 모델(LVLM) 최적화를 목표로 합니다. 이 방법은 차트 이미지를 처리하고 텍스트 요약을 생성하기 위해 LVLM을 직접 교육시켜, 별도의 차트 파싱 모듈이 필요 없는 엔드 투 엔드 방식을 채택합니다. 또한, 시각적 Chain-of-Thought 메커니즘을 통합하여 LVLM이 요약 생성 중 시각적 추론 단계를 수행하도록 유도하는 방식으로 설계되었습니다.

- **Technical Details**: 제안된 V-CoT 방법은 Chart-Sum-QA 데이터셋을 활용하여 대규모 차트 요약 데이터에 대해 LVLM을 instruction fine-tuning 방식으로 학습합니다. 교육 중 LVLM은 최종 요약을 출력하기 전에 중간 시각적 추론 단계를 생성하는데, 이를 통해 시각적 분석 프로세스를 모방하도록 유도합니다. V-CoT 단계는 차트 유형 식별, 축 레이블 해석 및 트렌드 분석과 같은 요소들을 포함하여, 모델이 시각적 분석 능력을 내재화할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 End-to-End V-CoT 방법은 자동 평가 메트릭인 BLEU, BLEURT, CIDEr, CS에서 기존의 최신 모델들을 상당히 초월하여 새로운 성과를 세웠습니다. 또한, 인간 평가에서 매칭 정확도와 추론 정확도에서도 우수한 결과를 나타내어, 차트 요약 분야의 새로운 벤치마크로 자리잡게 됩니다. 이러한 성과는 V-CoT 접근 방식의 효과성과 강건성을 입증합니다.



### Towards Conditioning Clinical Text Generation for User Contro (https://arxiv.org/abs/2502.17571)
- **What's New**: 본 논문은 최신 Large Language Models(LLMs)를 활용하여 임상 환경에서의 자연어 생성 시스템을 개선하기 위한 자동화된 데이터셋 증대 방법을 탐구합니다. 여기서 LLMs는 인간의 역할을 대체하여 의료진의 통제를 가능하게 하고, 인지적 부담을 줄일 수 있도록 돕습니다. 논문에서는 BioNLP ACL'24 Discharge Me! 공동 과제에서 새로운 최신 기술을 달성하였으며, 데이터셋 증대 없이도 9%의 상대적 개선을 이루었습니다.

- **Technical Details**: 논문에서 제안하는 방법은 전통적인 데이터셋에 저자 가이드라인과 주제 세분화를 추가하여 LLMs의 스타일 및 내용 통제를 가능하게 합니다. 이는 생성 과정을 더 관리 가능한 하위 작업으로 나누어 의료진이 작성 가이드라인을 조정하고 과정을 동적으로 안내할 수 있도록 합니다. 이전의 접근 방식과 달리, 다양한 스타일로의 대응이 가능해져 풍부한 임상 텍스트 생성을 지원합니다.

- **Performance Highlights**: 사전 평가 결과, 제안된 접근 방식은 관련성, 정확성, 그리고 사실적 일관성을 현저히 향상시키는 것으로 나타났습니다. 데이터셋 증대 기법을 통해서는 기존의 최신 상태에 비해 34%의 상대적 개선을 이룰 수 있었습니다. 또한, 하이퍼파라미터 최적화를 통해 전통적인 지시 조정 방식이 무리 없이 개선될 수 있음을 보여주었습니다.



### Policy Learning with a Natural Language Action Space: A Causal Approach (https://arxiv.org/abs/2502.17538)
- **What's New**: 이 논문은 자연어 행동(space)에서 다단계 의사결정을 위한 새로운 인과적 프레임워크를 소개합니다. 기존의 Proximal Policy Optimization(PPO) 접근법에서 요구되는 여러 모델과 많은 훈련 데이터 대신, 단일 모델을 통해 Dynamic Treatment Regimes(DTR)를 이용하여 Q-learning을 적용합니다. 이를 통해 언어 임베딩에서의 그래디언트 상승(gradient ascent)을 통해 데이터 효율적인 정책 학습이 가능해집니다.

- **Technical Details**: 우리는 자연어 행동 공간에 대한 정책 학습 문제를 인과 추론(causal inference) 문제로 형태를 바꿉니다. Q-learning과 DTR을 활용한 프레임워크를 제안하여 단일 모델로 의사결정 시퀀스를 최적화합니다. 이 과정에서 텍스트 임베딩을 최적화하고, 이를 자연어로 디코딩하는 새로운 전략을 개발하여 실제 의사결정에 적용합니다.

- **Performance Highlights**: 정신 건강 중재, 혐오 발언 대응, 감정 스타일 전이 등의 세 가지 작업에서 우리의 접근법이 기존의 기준선보다 유의미한 향상을 보였습니다. 특히, 효과 전이(strength of transfer)에서 이전 방법보다 최대 30% 향상된 성과를 보여주었으며, 인간 평가에서도 유창성, 내용 보존, 효과 전이 분야에서 더 높은 균형 잡힌 성과를 기록했습니다.



### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (https://arxiv.org/abs/2502.18449)
- **What's New**: 이 논문은 SWE-RL이라는 새로운 접근법을 소개하며, 이는 소프트웨어 공학(SW engineering) 분야에서 대형 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 최초의 강화 학습(RL) 방법이다. SWE-RL은 주요 목적이 개발자들의 추론 과정 및 솔루션을 스스로 복구할 수 있도록 LLMs를 학습시키는 것이다. 이러한 기술은 대규모 오픈 소스 소프트웨어 진화 데이터(software evolution data)를 활용하여 현실 세계의 문제를 해결하는 데 초점을 맞추고 있다.

- **Technical Details**: SWE-RL은 규칙 기반 보상(rule-based reward)을 사용하여 LLM의 추론 능력을 개선하며, 코드 스냅샷, 코드 변경 및 문제(issue) 및 풀 리퀘스트(pull requests)와 같은 이벤트를 포함하는 소프트웨어의 모든 생애주기 기록을 학습에 활용한다. Llama3-SWE-RL-70B라는 모델을 구축하였으며, 이는 41.0%의 문제 해결율을 보이며 GitHub에서 인간 검증을 통과한 SWE-bench Verified 데이터셋에서 성능을 검증하였다. 본 논문에서 제시된 방법은 비록 소프트웨어 진화 데이터에만 RL을 수행했음에도 불구하고 일반화된 추론 능력을 보여 주목할 만하다.

- **Performance Highlights**: Llama3-SWE-RL-70B는 현재까지 중간 규모(<100B) LLM에서 보고된 최고의 성능을 기록하였으며, GPT-4o와 같은 선도적인 상용 LLM과도 비교 가능하다는 점에서 중요하다. 본 모델은 다양한 범위 밖의(out-of-domain) 작업에서 성능이 향상되었으며, 그 예로 함수 코딩, 라이브러리 사용, 코드 추론, 수학 및 일반 언어 이해가 있다. 기존의 감독 하에 미세 조정(supervised fine-tuning) 모델과 비교해도 Llama3-SWE-RL-70B가 더 나은 성능을 보여준다.



### Rank1: Test-Time Compute for Reranking in Information Retrieva (https://arxiv.org/abs/2502.18418)
- **What's New**: Rank1은 테스트 시간에 계산(compute)을 활용하여 훈련된 최초의 재순위 모델로, 이를 통해 작은 모델의 성능을 신속하게 개선할 수 있는 가능성을 보여줍니다. MS MARCO 데이터셋에서 60만 개 이상의 R1 추론 추적 예제를 수집하고 오픈소스로 공개하여 연구자들이 활용할 수 있게 하였습니다. 이 모델은 고급 추론 및 지시 준수 데이터셋에서 최첨단 성능을 기록하며, 유저 입력 프롬프트에 잘 반응하는 특성을 보여줍니다.

- **Technical Details**: Rank1은 OpenAI의 o1, Deepseek의 R1과 같은 추론 언어 모델을 사용하여, 정보 검색(Information Retrieval) 맥락에서 테스트 시간에 계산(compute)을 활용하도록 설계되었습니다. 모델은 쿼리와 문서를 동시에 추론할 수 있도록 최적화되었으며, 다양한 데이터셋에서 성능을 입증하였습니다. 특히, 이 모델은 Hard negatives와 Easy negatives의 조합을 활용하여 더 나은 학습을 이끌어냈습니다.

- **Performance Highlights**: Rank1은 브라이트(BRIGHT) 벤치마크에서 최첨단 성능을 달성하였고, 전통적인 IR 벤치마크에서 놀라운 성능을 보여주는 등 상당한 향상을 이뤄냈습니다. 사용자가 주는 프롬프트에 대한 적응성도 높아, 다양한 쿼리에 대해 효율적이고 설명 가능한 추론 체인을 제공합니다. 이는 더 나아가 사용자나 에이전트 기반 RAG 시스템이 활용할 수 있는 투명한 시스템으로 발전할 수 있게 합니다.



### AMPO: Active Multi-Preference Optimization (https://arxiv.org/abs/2502.18293)
- **What's New**: 이번 연구에서는 Active Multi-Preference Optimization (AMPO)라는 새로운 접근법을 소개합니다. AMPO는 대량의 후보 응답을 평가하고, 중요한 극단과 독특한 의미 군을 포괄하는 소규모 정보 집합을 선택하여 언어 모델의 선호 최적화를 가능하게 합니다. 이를 통해 다중 선호 최적화의 과정에서 모델이 학습하는 신호의 질을 높입니다.

- **Technical Details**: AMPO는 (a) 정책 기반 데이터 생성, (b) 그룹 기반 선호 학습, 그리고 (c) 능동적인 서브셋 선택을 통합하는 프레임워크입니다. 고유의 그룹 대조 손실 함수인 Refa를 채택하여 다수의 긍정적 및 부정적 응답을 단일 손실 항목으로 결합합니다. 또한, 선택된 응답의 다양성을 극대화하는 방법으로 베이직한 bottom-K에서 이론적으로 기반한 Opt-Select 방식까지 다양한 능동 선택 방식을 탐구합니다.

- **Performance Highlights**: AMPO는 Llama 8B 모델을 기반으로 한 AlpacaEval에서 최신 성과를 달성하였으며, Simpo와 같은 강력한 기준선 모델을 초월했습니다. 연구자들은 Hugging Face에 AMPO-Coreset-Selection 및 AMPO-Opt-Selection 데이터셋을 공개하여 다중 선호 정렬 연구를 촉진합니다.



### Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Suppor (https://arxiv.org/abs/2502.18274)
- **What's New**: 이 논문에서는 Citrus라는 의료 언어 모델을 소개합니다. 이 모델은 의료 전문가의 인지 과정을 모방하여 인공지능(AI) 추론과 임상 전문 지식 간의 간극을 해소합니다. Citrus는 전문가 수준의 질병 추론 데이터를 시뮬레이션하여 훈련되었으며, 이는 임상의 결정 경로를 정확하게 포착하는 혁신적인 접근 방식을 통해 수행되었습니다.

- **Technical Details**: Citrus는 대규모의 시뮬레이션된 전문가 질병 추론 데이터의 코퍼스에 대해 훈련됩니다. 이 모델은 복잡한 진단 및 치료 과정의 추론을 시뮬레이션하는 데 중점을 두고 있으며, 마지막 단계 훈련 데이터를 공개하여 의료 추론 작업을 위한 데이터 부족 문제도 해결합니다. 새로운 설계(architecture)와 대화형 데이터셋을 통해, 검증 및 반복 가능한 결과를 제공하고자 합니다.

- **Performance Highlights**: Citrus는 MedQA와 같은 권위 있는 벤치마크를 통한 평가에서 우수한 성능을 보여주며, 유사한 크기의 다른 모델들보다 더 나은 결과를 나타냈습니다. 이러한 성과는 Citrus가 의료 결정 지원 시스템을 크게 향상시킬 잠재력을 지닌 것으로, 임상 결정을 위한 더 정확하고 효율적인 도구를 제공할 수 있음을 강조합니다.



### Iterative Counterfactual Data Augmentation (https://arxiv.org/abs/2502.18249)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 반사실적 데이터 증강(Counterfactual Data Augmentation, CDA) 기법의 새로운 방법론인 반복적 반사실적 데이터 증강(Iterative Counterfactual Data Augmentation, ICDA)을 소개합니다. ICDA는 초기 고잡음 개입을 통해 훈련 데이터셋의 신호와 레이블 간의 상호정보(mutual information)를 높이고, 불필요한 신호의 정보를 줄이는 결과를 가져옵니다. 이 방식은 수동 규정 또는 기존 알고리즘 기반 방법론보다 더 효과적으로 작동합니다.

- **Technical Details**: ICDA 절차는 데이터셋 내에서 다루고자 하는 주요 신호와 불필요한 신호 간의 상호정보를 유지하면서, 훈련 데이터셋에서 높은 잡음을 가지던 데이터를 단계적으로 정제하는 과정을 포함합니다. 실험에서는 여섯 개의 인간이 생성한 데이터셋과 두 개의 대규모 언어 모델에 의해 생성된 데이터셋을 사용하였습니다. 훈련된 데이터셋은 인간의 주석과 더 잘 일치하는 합리적인 문서 내용을 생성합니다.

- **Performance Highlights**: 실험 결과는 ICDA를 통해 생성된 데이터셋에서 모형이 보다 효과적으로 학습하고, 결과적으로 인간 주석과 유사한 이해도를 가지는 문서가 생성됨을 보여주었습니다. 연구진은 이 기법이 학습의 질을 높이는 데 기여하며, 기존의 CDA 방식에 비해 명확한 성능 향상을 나타냈음을 강조했습니다.



### Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Though (https://arxiv.org/abs/2502.18186)
- **What's New**: 이번 연구에서는 C$^2$SER이라는 새로운 대형 오디오 언어 모델(ALM)을 제안하여 음성 감정 인식(SER)에서 발생하는 허위 정보 문제를 해결하고자 합니다. C$^2$SER는 컨텍스트 인식(contextual perception)과 사고의 연쇄(chain of thought, CoT)를 통해 SER의 안정성과 정확성을 강화합니다. 특히, Emotion2Vec-S 모델을 통합하여 감정 분별력을 향상시키고, 음성 내용과 말하기 스타일을 활용하여 인식을 더욱 정교하게 수행합니다.

- **Technical Details**: C$^2$SER는 Whisper 인코더를 사용하여 의미를 인식하고, Emotion2Vec-S는 음향적 인식을 위한 모듈로 작용합니다. Emotion2Vec-S는 반지도 학습(semi-supervised learning)을 통해 원래의 Emotion2Vec 모델을 확장하여 감정 관련 정보를 계량적으로 추출할 수 있게 설계되었습니다. CoT 방식을 적용하여 SER 작업을 단계적으로 처리함으로써 인간의 사고 방식을 모방하여 허위 정보 발생 가능성을 줄입니다.

- **Performance Highlights**: C$^2$SER는 여러 음성 데이터 세트를 활용한 광범위한 실험을 통해 기존의 ALM 모델들보다 우수한 성능을 보여주었습니다. 특히 Emo-Emilia라는 새로운 SER 테스트 세트를 통해서도 성능을 Validation하며, 가중 정확도(weighted accuracy), 비가중 정확도(unweighted accuracy), Macro F1 score 등에서 우수한 결과를 기록했습니다. 이를 통해 C$^2$SER가 다양한 맥락에서 신뢰할 수 있는 감정 인식을 제공할 수 있는 가능성을 확인했습니다.



### Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations (https://arxiv.org/abs/2502.18147)
- **What's New**: 이번 연구에서는 Sparse autoencoders (SAEs)를 넘어 Jacobian SAEs (JSAEs)를 제안합니다. JSAEs는 모델 구성 요소의 입력과 출력 활성화에서뿐만 아니라 이들을 연결하는 계산에서도 희소성(sparsity)을 제공합니다. 이를 통해 LLMs의 계산을 이해하는 데 기여할 수 있는 새로운 방향성을 제시합니다.

- **Technical Details**: JSAEs의 주요 기술 기여는 Jacobians를 효율적으로 계산하는 방법을 찾는 것입니다. LLM의 크기 때문에 Jacobians를 직접 계산하는 것은 실용적이지 않지만, 적절한 기초(JSAE basis)로 재작성할 경우 MLP는 대략적으로 선형(linear)으로 변환됩니다. 이 방법은 계산 희소성을 유지하면서도 전통적인 SAEs와 유사한 LLM 성능을 제공합니다.

- **Performance Highlights**: 실험 결과, JSAEs는 사전 훈련된 LLM에서 계산 희소성이 더 높게 나타났습니다. 이는 LLM이 훈련을 통해 학습한 계산 그래프의 희소성이라는 특성을 지니고 있음을 보여줍니다. 따라서 JSAEs는 표준 SAEs보다 학습된 transformer 계산을 이해하는 데 더 적합할 수 있다는 가능성을 제시합니다.



### HyperG: Hypergraph-Enhanced LLMs for Structured Knowledg (https://arxiv.org/abs/2502.18125)
- **What's New**: 본 논문에서는 LLMs의 구조적 지식 처리 능력을 향상시키기 위한 새로운 하이퍼그래프 기반 생성 프레임워크인 HyperG를 제안합니다. HyperG는 희소 데이터에 문맥 정보를 추가하고, 이를 통해 복잡한 구조적 관계를 인코딩할 수 있도록 설계되었습니다. 이를 통해 LLMs가 다양한 실제 작업에서 더 효율적으로 데이터의 구조를 이해하고 활용할 수 있도록 합니다.

- **Technical Details**: HyperG는 하이퍼그래프 구조를 활용하여 구조적 지식 내의 의미적 일관성(semantic consistency), 순서 불변성(order invariance), 계층적 종속성(hierarchical dependencies)을 캡처합니다. 이 시스템은 희소한 테이블 셀을 문맥 정보로 증강한 후, 태스크 특정 질문을 포함하여 임베디드된 의미적 지식을 하이퍼그래프를 통해 전파하는 Prompt-Attentive Hypergraph Learning(PHL) 모듈을 사용합니다. 이러한 접근 방식은 구조적 데이터에 대한 LLMs의 이해력과 추론 능력을 증대시키는 데 기여합니다.

- **Performance Highlights**: HyperG의 효과성과 일반화 가능성을 검증하기 위해, 본 연구에서는 다양한 하류 작업에서 시스템의 성능을 평가하는 광범위한 실험을 진행했습니다. 실시된 실험들은 기존의 방법들과 비교할 때 HyperG가 LLMs의 구조적 지식 처리 능력을 더욱 향상시킬 수 있음을 입증했습니다. 또한, 이 프레임워크는 희소한 데이터 문제를 해결하는 데 중점을 두어, 정보 전파를 통해 보다 풍부한 지식을 생성할 수 있는 가능성을 보여주었습니다.



### Bayesian Optimization for Controlled Image Editing via LLMs (https://arxiv.org/abs/2502.18116)
Comments:
          8 figures

- **What's New**: BayesGenie는 대형 언어 모델(LLMs)과 베이지안 최적화(Bayesian Optimization)를 통합하여 사용자가 이미지 편집을 쉽고 정밀하게 수행할 수 있도록 돕는 최신 접근법을 제안합니다. 기존의 수작업 영역 표시 없이 자연어로 이미지를 수정할 수 있는 기능을 제공하며, 원본 이미지의 의미적 일관성을 유지합니다. 이는 모델 미세 조정 또는 대규모 사전 훈련 없이도 다양한 LLM에 대해 뛰어난 적응성을 보여주는 모델 불가지론적(design) 설계를 특징으로 합니다.

- **Technical Details**: BayesGenie는 LLM의 의미 이해 능력을 활용하여 사용자의 요구에 따라 세부적인 프롬프트를 생성한 후 이를 스테이블 디퓨전(Stable Diffusion) 모델에 전달하여 정확한 이미지 수정을 지원합니다. 베이지안 최적화를 통해 매개변수 공간을 체계적으로 탐색하여 최적의 편집 품질을 달성합니다. 이 과정에서 ‘text_cfg_scale’와 ‘image_cfg_scale’와 같은 주요 매개변수를 조정하면서 결과물과 사용자 요구 사항 간의 정렬을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 BayesGenie는 편집 정확도 및 의미 보존 측면에서 기존 방법들을 크게 능가하는 성능을 입증하였습니다. Claude3와 GPT-4와 같은 다양한 LLM을 활용하여 프레임워크의 효과성을 검증하였으며, 이미지 수정의 직관성과 정확성을 향상시켰음을 보여줍니다. 실험 결과는 BayesGenie가 즉시 배치 가능하며 여러 시나리오에서 높은 품질의 결과를 제공할 수 있음을 시사합니다.



### Detecting Offensive Memes with Social Biases in Singapore Context Using Multimodal Large Language Models (https://arxiv.org/abs/2502.18101)
- **What's New**: 이 연구는 싱가포르의 현대적이고 다중 모달(multimodal) 의사소통 수단인 밈(meme)을 효과적으로 분류하기 위한 VLM(vision-language model)의 미세 조정(fine-tuning) 데이터셋을 소개합니다. 기존의 전통적인 온라인 콘텐츠 조정 시스템이 자원 부족 언어와 문화적 맥락의 필요성을 갖는 고도의 정보가 밀집된 매체인 밈을 분류하는 데 어려움을 겪고 있는 가운데, 112K의 밈 샘플로 구성된 대규모 데이터셋을 특수하게 제작했습니다. 이 데이터셋은 GPT-4V에 의해 레이블이 지정되어 있으며, 온라인 콘텐츠 조정에서 인공지능의 효과성을 극대화하고자 합니다.

- **Technical Details**: 연구에서는 두 가지 대표적인 비전-언어 모델(VLM), LLaVA-NeXT Mistral 7B와 Qwen2-VL 7B를 교육하고 평가했습니다. 또한 OCR(Optical Character Recognition)과 언어 번역(translation) 구성 요소의 필요성을 탐구하였습니다. 연구에서 제안하는 파이프라인은 두 가지 VLM을 사용하여 최첨단 성능을 보여주며, 인증된 데이터셋, 코드, 모델 가중치를 오픈 소스 방식으로 공개할 예정입니다.

- **Performance Highlights**: 제안된 솔루션은 공인된 테스트 세트에서 80.62%의 정확도와 0.8192 AUROC(Areas Under the Receiver Operating Characteristic Curve)를 달성하였고, 이는 온라인 콘텐츠 조정에 있어 인간의 도움을 크게 돕는 도구가 되리라 기대됩니다. 연구 결과는 싱가포르의 온라인 안전 시스템의 발전을 위해 제공되는 귀중한 로컬리제이션 리소스가 될 것입니다.



### Defining bias in AI-systems: Biased models are fair models (https://arxiv.org/abs/2502.18060)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 논문에서는 AI 시스템에서의 편향(bias) 개념에 대해 명확한 정의가 부족하다는 점을 지적하며, '공정성(fairness)'과의 대비 속에서 빈번히 사용되고 있는 편향 개념을 재조명합니다. 저자들은 편향을 부정적 현상으로 간주하는 전통적인 관점에서 벗어나, 편향과 차별(discrimination)의 구별이 중요하다고 주장합니다. 이를 통해 AI 시스템에서의 공정성에 대한 논의가 더 건설적으로 발전할 수 있다고 강조합니다.

- **Technical Details**: 신경망(neural networks)의 기본 구성 요소인 인공 기계 뉴런은 McCulloch-Pitts 뉴런(MCP neuron)에서 기원하며, 이는 입력을 처리해 0 또는 1의 출력을 생성하는 간단한 이진 뉴런 모델입니다. 이 논문에서는 아이디어가 진화하면서도 '편향(bias)'이 모델의 결정 경계를 조정하는 데 어떻게 사용되는지 설명합니다. 특히 ADALINE(Adaptive Linear Neuron)을 통해 편향 개념이 처음 포함되었음을 강조하며, 이는 모델의 훈련 과정에서 오류를 최소화하는 데 기여합니다.

- **Performance Highlights**: 편향의 정의와 공정성 간의 관계를 명확히 하므로써, AI 모델의 개발에 있어 실질적인 공정성을 추구하는 데 중요한 기여를 합니다. 예를 들어, 얼굴 인식 알고리즘이 특정 인종 데이터를 기반으로 훈련되면, 공정성에 부합하지 않는 것은 물론 특정 집단에 대한 성능 저하를 초래할 수 있습니다. 이와 같은 문제를 해결하기 위해, 대표성이 있는 훈련 데이터 사용이 필수적이며, 이는 편향 없는 모델로 이어질 수 있음을 제시합니다.



### ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents (https://arxiv.org/abs/2502.18017)
- **What's New**: 이번 논문에서는 시각적으로 풍부한 문서에서 정보를 이해하는 데 있어 기존의 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 데이터셋인 ViDoSeek을 소개합니다. ViDoSeek은 복잡한 추론을 요구하는 문서에서 RAG 성능을 평가하도록 설계되었습니다. 이를 통해 현재 RAG 접근 방식의 주요 한계도 식별하였으며, 특히 시각적 검색 방법과 추론 토큰 할당의 부족 문제가 강조되었습니다.

- **Technical Details**: ViDoRAG라는 새로운 다중 에이전트 RAG 프레임워크를 제안하여 시각적 문서에서 복잡한 추론을 위한 개선점을 제시합니다. 이 프레임워크는 Gaussian Mixture Model (GMM) 기반의 하이브리드 전략을 채택하여 다중 모달 검색(multi-modal retrieval)을 효과적으로 처리합니다. 또한 탐색, 요약, 반영 과정을 포함하는 반복적인 에이전트 워크플로우를 통해 모델의 추론 능력을 배양합니다.

- **Performance Highlights**: ViDoSeek에서 진행된 광범위한 실험 결과, ViDoRAG는 기존 방법들보다 10% 이상 성능이 향상된 것으로 나타났습니다. 이 결과는 ViDoRAG이 RAG 분야에서 테스트 시간 확장(test-time scaling) 연구에 대한 새로운 방향을 제공함을 나타냅니다. ViDoRAG은 경쟁적인 ViDoSeek 기준에서 기존 방법들과 비교하여 우수한 성능을 보였습니다.



### LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena (https://arxiv.org/abs/2502.17967)
- **What's New**: 이번 연구에서는 Agent Trading Arena라는 새로운 가상 환경을 소개하여 대규모 언어 모델(LLMs)의 수치적 추론 능력을 향상시키기 위한 방법을 제시합니다. 이 환경은 제로섬 게임(Zero-Sum Game) 방식으로 설계되어 에이전트들이 주식 포트폴리오에 투자하여 복잡한 경제 시스템을 시뮬레이션하는데, 이는 실제 상황에서의 문제 해결 능력을 더 잘 평가할 수 있게 해줍니다. 또한, 시각적 데이터를 사용할 때 LLMs의 수치적 추론 능력이 더 우수하다는 점을 강조합니다.

- **Technical Details**: Agent Trading Arena는 자산 가격 결정이 에이전트 간의 상호작용에 기반하여 자체적으로 이루어지도록 설계되었습니다. 외부 지식의 영향을 줄이기 위해 가격은 입찰-매도 시스템에 따라 결정되며, 모든 에이전트의 활동은 단기적인 최적 전략에 의존하지 않고 자율적으로 동적으로 변화합니다. 이러한 시스템은 에이전트가 시각적 표현을 통해 복잡한 데이터 관계를 추론하고 보다 전략적으로 의사 결정을 내릴 수 있도록 돕는 반사 모듈(Reflection Module)을 포함합니다.

- **Performance Highlights**: 실험 결과, LLMs는 텍스트 형식의 수치적 데이터에서 성능이 떨어지는 반면, 시각적으로 표현된 데이터에서는 월등한 성과를 보였습니다. 시각적 데이터로 처리된 경우, LLMs는 실험에서 높은 수익률(Return Rate)을 보여주어 구조화된 시각적 표현의 이점을 입증했습니다. 또한, NASDAQ 주식 데이터셋을 사용하여 LLM의 시각적 데이터 처리 능력이 텍스트 기반 데이터 처리보다 뛰어나다는 점을 확인했습니다.



### Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers (https://arxiv.org/abs/2502.17882)
- **What's New**: 과학 연구는 본질적으로 글로벌하지만, 대부분의 학술지는 영어로만 발행되어 비영어 원어민 연구자에게 장벽이 되고 있습니다. 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 JATS XML 형식을 유지한 채로 과학 기사를 번역하는 자동화된 방법을 개발하였습니다. 이로 인해 28개 언어로 다수의 과학 기사를 번역할 수 있게 되었으며, 번역 결과의 정확성을 평가하기 위한 새로운 질문-답변(QA) 벤치마킹 방법도 도입했습니다.

- **Technical Details**: 98%의 동료 심사 과학 기사가 영어로 발행되는 상황에서, LLM 기반의 번역 솔루션을 통해 언어 장벽을 낮추기 위한 방법이 제시됩니다. 자동 번역 프로세스에서는 JATS XML 포맷을 유지하며 원본 및 번역된 문서를 기반으로 번역 품질을 평가하는 방법을 사용합니다. LLM은 전통적인 기계 번역 시스템보다 성능이 우수하며, 특정 과학 커뮤니티의 요구에 맞춰 번역을 조정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 새로운 벤치마크 결과에 따르면, 번역된 기사의 평균 성능은 95.9%로, 주요 과학 정보가 정확히 전달되고 있음을 보여줍니다. 사용자 연구에서는 15명의 연구자가 자신의 기사를 번역했을 때 원본 정보를 잘 표현했다고 일관되게 평가했습니다. 흥미롭게도, 3분의 1의 저자는 많은 기술 용어가 '과도하게 번역'되었다고 느끼며, 영어 용어를 그대로 유지하길 선호한다고 밝혔습니다.



### A General Framework to Enhance Fine-tuning-based LLM Unlearning (https://arxiv.org/abs/2502.17823)
- **What's New**: 본 논문에서는 기존의 fine-tuning 기반 unlearning 방법을 개선하기 위해 Gated Representation UNlearning (GRUN)이라는 새로운 프레임워크를 제안합니다. GRUN은 target data를 구분하고 suppress하는 두 가지 모듈로 구성되어 있어 unlearning 성능을 크게 향상시킵니다. 또한, 이 시스템은 기존 방법보다 모델의 유틸리티(utility)를 유지하는 데 중점을 두고 있습니다.

- **Technical Details**: GRUN은 soft gate function과 Representation Fine-tuning (ReFT) 모듈로 구성됩니다. soft gate function은 target data의 구분을 지원하며, ReFT는 모델의 파라미터를 조정하는 대신 representation을 fine-tuning하여 성능 저하를 방지합니다. 이 논문에서는 GRUN의 효율성과 효과성을 입증하기 위해 다양한 실험을 통해 95% 이상의 훈련 시간 단축과 함께 near-perfect unlearning을 달성했습니다.

- **Performance Highlights**: GRUN은 LLM의 크기보다 적은 0.05% 이하의 추가 모듈을 요구하며, 다양한 모델(Llama 3.1, Mistral 등)과 데이터셋(TOFU, WMDP 등)에서 검증되었습니다. 이 방법은 효율적이며 sequential unlearning에 유망한 결과를 보이며, 기존 fine-tuning 방식에 일반적으로 적용 가능한 솔루션으로서 긍정적인 평가를 받고 있습니다.



### An Overview of Large Language Models for Statisticians (https://arxiv.org/abs/2502.17814)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 통계와 AI 분야의 교차점에서 어떻게 신뢰성과 투명성을 증진시키는데 기여할 수 있는지를 탐구합니다. 특히 불확실성 정량화(uncertainty quantification), 해석 가능성(interpretability), 공정성(fairness), 개인 정보 보호(privay), 워터마킹(watermarking) 및 모델 적응(model adaptation)과 같은 이슈에 초점을 맞추고 있습니다. 이러한 세부 사항들을 통해 통계학자들이 LLMs의 발전에 중요한 기여를 할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: LLMs는 자연어를 이해하고 생성하는 데 있어 본질적으로 단어 또는 단어 시퀀스의 확률을 할당하는 모델입니다. LLMs의 설계와 배치에서 통계학자들이 어떤 역할을 할 수 있는지에 대한 질문이 제기되며, 이는 통계 방법 및 이론의 진전을 요구합니다. 모델의 아키텍처가 신뢰할 수 있는 확률적 출력을 생성하는 방법과 알고리즘의 공정성과 신뢰성을 보장하기 위해 LLMs의 출력을 어떻게 활용할 수 있는지를 탐구하고 있습니다.

- **Performance Highlights**: LLMs는 데이터 수집, 청소 및 분석의 전통적인 통계 워크플로우를 향상시킬 수 있는 잠재력을 제공합니다. 의료 연구 및 생물 통계와 같은 분야에서 LLMs는 대표 데이터를 합성하고, 비정형 임상 노트에서 중요한 통찰을 추출하며, 높은 위험의 응용에서 예측 모델링을 지원할 수 있습니다. 이러한 상호 작용은 통계와 AI 분야 모두에서 새로운 기회를 제공하며, LLMs의 발전이 사회적 복잡성 문제 해결에 기여할 수 있는 가능성을 열어갑니다.



### MuCoS: Efficient Drug-Target Prediction through Multi-Context-Aware Sampling (https://arxiv.org/abs/2502.17784)
- **What's New**: 이번 연구에서는 MuCoS(Multi-Context-Aware Sampling)라는 새로운 방법을 제안하여 복잡한 생물학적 엔터티 간의 관계를 더 정확하게 예측할 수 있도록 하였습니다. MuCoS는 드러나지 않은 관계를 다룰 수 있도록 최적화된 이웃 샘플을 활용하고, BERT(Bidirectional Encoder Representations from Transformers)와의 통합을 통해 향상된 맥락적 임베딩을 제공합니다. 기존 모델에 비해 MuCoS는 13%의 MRR(Mean Reciprocal Rank) 향상 등을 보여주어 전체적으로 예측 성능이 개선되었습니다.

- **Technical Details**: MuCoS는 인접 엔터티로부터 맥락 정보를 수집하여 관계 예측을 수행하는 기술입니다. 이 방법은 헤드 엔티티와 관계로부터 최적화된 맥락 정보를 추출하고 이를 BERT 모델로 처리하여 예측을 수행합니다. 이 방법은 또한 주변 임베딩의 복잡성을 줄이고, 음성 샘플링을 제거하여 훈련 효율성 및 강 robustness를 향상시킵니다.

- **Performance Highlights**: 실험 결과, MuCoS는 KEGG50k 생물의학 데이터셋에서 기존 모델에 비해 MRR 13%, Hits@1 7%, Hits@3 4% 및 Hits@10 18% 향상을 보였습니다. 드러나지 않은 관계와 엔티티에 대한 예측 성능 또한 향상되어, 복잡한 DTI(Drug-Target Interaction) 예측에서 효과적인 방법으로 입증되었습니다.



### Tip of the Tongue Query Elicitation for Simulated Evaluation (https://arxiv.org/abs/2502.17776)
- **What's New**: 본 논문은 Tip-of-the-Tongue (TOT) 검색 시스템을 개선하기 위한 두 가지 새로운 방법을 제시합니다. GPT와 같은 대형 언어 모델(LLM)과 인간 참여자를 활용하여 TOT 검색 쿼리를 생성하는 방법을 소개하며, 이는 기존의 CQA 플랫폼에 의존하는 데이터 수집 방식을 극복하는 데 기여할 것입니다. 이 방법들은 영화, 랜드마크, 인물 등의 다양한 도메인에서 TOT 쿼리를 효율적으로 수집하고 평가할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 저자들은 LLM 기반의 TOT 사용자 시뮬레이터를 개발하여 합성 쿼리를 대규모로 생성하고, 이 쿼리들이 CQA 기반 TOT 쿼리와 어떤 연관성을 가지는지를 검증했습니다. 또한, 인간 참여자를 통한 쿼리 수집 인터페이스를 만들어 자연스러운 TOT 쿼리를 수집할 수 있는 방법을 제시하였습니다. 이러한 접근 방식은 CQA 데이터 의존도를 줄이고, 여러 도메인에서 데이터 수집의 범위를 확장하는 데 도움을 줍니다.

- **Performance Highlights**: TOT 쿼리와 관련된 시스템의 순위 상관관계 및 언어 유사성 분석을 통해 LLM을 활용한 합성 쿼리가 CQA 기반 쿼리와 효과적으로 일치함을 입증했습니다. 또한, 이 연구 결과는 TREC 2024 TOT 트랙과 TREC 2025 TOT 트랙에서 검증된 쿼리로 활용될 예정입니다. 저자들은 TOT 쿼리 생성을 위한 소스 코드를 공개하고, 사용자 인터페이스와 함께 유용한 시각적 자극 자료도 제공합니다.



### Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures (https://arxiv.org/abs/2502.17710)
Comments:
          40 pages, 49 figures

- **What's New**: 이번 연구에서는 다양한 문화적 해석을 포함하는 새로운 데이터셋인 Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS)를 소개합니다. 이 데이터셋은 25가지 제스처와 85개 국가를 포함하여 총 288개의 제스처-국가 쌍을 다루고 있으며, 각 제스처의 공격성, 문화적 중요성 및 맥락적 요소에 대한 주석이 포함되어 있습니다. 이는 AI 시스템이 비문화적 콘텐츠를 생성할 위험을 줄이기 위한 필수적인 노력을 보여줍니다.

- **Technical Details**: MC-SIGNS는 미국 중심의 편향을 발견하고, AI 모델의 비언어적 커뮤니케이션 해석에서 나타나는 한계를 드러냅니다. 기존의 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)은 문화적 맥락을 인식하지 못하고 미국에서 유래한 해석에 의존하며, 많은 제스처를 공격적으로 잘못 플래그(flag)합니다. 데이터셋은 공격적 제스처 감지와 비언어적 커뮤니케이션의 문화적 적합성을 평가하기 위한 테스트 베드(test bed)로 사용됩니다.

- **Performance Highlights**: MC-SIGNS 데이터셋을 활용한 평가 결과, text-to-image(T2I) 시스템은 공격적인 내용을 거의 거부하지 못하며, LLM은 제스처를 과도하게 공격적으로 표시하는 경향이 있음을 보여주었습니다. 연구 결과, AI 모델은 미국 중심의 해석을 자주 반복하며, 비 미국적 맥락에서 공격적인 제스처를 식별하는 정확도가 크게 낮아짐을 확인했습니다. 이는 AI 기술의 공평한 전 세계적 배포를 위해 더욱 포괄적이고 문화적으로 민감한 안전 메커니즘이 필요함을 강조합니다.



### Contrastive Visual Data Augmentation (https://arxiv.org/abs/2502.17709)
- **What's New**: 이 논문에서는 새로운 Contrastive Visual Data Augmentation (CoDA) 전략을 제안하여 대형 다중모달 모델(Large Multimodal Models, LMMs)이 새로운 개념을 인식하고 논리적으로 처리하는 능력을 향상시키는 방법을 설명합니다. CoDA는 대상 개념의 주요 대조적 텍스트 및 시각적 특징을 추출하고, 이를 통해 작성된 합성 데이터를 생성하여 LMM이 혼동할 수 있는 개념을 명확히 구분할 수 있도록 돕습니다.

- **Technical Details**: CoDA의 주된 과정은 대조적 특징 추출, 특징 필터링, 특징 제어 이미지 생성, 증강 이미지 필터링의 4단계로 구성됩니다. 이 과정에서 CoDA는 LMM이 잘못 인식하는 개념과 혼동하게 되는 개념을 식별하고, 해당 개념의 시각적 및 텍스트적 특징을 추출합니다. 이 특징들은 가시성이 뛰어나고 LMM이 이해할 수 있는 방식으로 생성되고 필터링을 거쳐 최종적으로 증강 이미지로 변환됩니다.

- **Performance Highlights**: CoDA는 iNaturalist와 SUN와 같은 다양한 데이터셋에서 성능이 크게 향상되는 것을 확인할 수 있었으며, NovelSpecies라는 새로운 데이터셋에서도 테스트 결과 기존의 시각적 데이터 증강 기법보다 절대적으로 12.3%, 5.1%, 6.0%의 정확도 향상을 보여주었습니다. 이로써 CoDA는 LMM의 새로운 개념 인식 능력을 효과적으로 개선하는 데 성공하였으며, 이에 따라 비전 커뮤니티에 중요한 기여를 하고 있습니다.



### From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs (https://arxiv.org/abs/2502.17701)
Comments:
          24 pages, 9 figures

- **What's New**: 이 논문에서는 FLARE라는 새로운 프레임워크를 도입하여 대형 언어 모델(LLM)을 활용한 산불 대피 결정 예측을 혁신적으로 개선하고 있습니다. FLARE는 인간의 복잡한 행동 논리를 보다 잘 이해할 수 있도록 심리학 및 행동 이론을 통합하여 Chain-of-Thought (CoT) 추론을 간소화하고 메모리 기반 강화 학습(RL) 모듈과 통합합니다. 기존 LLM의 한계를 극복하여 보다 정확한 대피 결정 예측을 가능하게 하며, 실험 결과 평균 20.47%의 성능 향상을 보여 주었습니다.

- **Technical Details**: FLARE 프레임워크는 위험 인식(risk perception) 및 위협 평가(threat assessment)를 주요 개념으로 사용하여 개인의 정신 상태를 나타냅니다. PADM(Protective Action Decision Model) 기반의 분류기를 사용하여 역사적 데이터와 경험적 행동 연구를 통해 가장 관련성이 높은 입력 변수를 선택하고, 이후 LLM이 선택된 추론 패턴으로부터 인식을 유추하여 점수를 부여합니다. 또한 이 시스템은 오류 기록과 자기 반성 메커니즘을 통합하여 모델의 추론 과정을 개선하는 동시에 개인의 대피 행동을 맞춤화합니다.

- **Performance Highlights**: FLARE는 자주 사용되는 기존 이론 기반 모델 대비 20.47%의 성능 향상을 기록하며, 강력한 교차 사건 일반화(cross-event generalizability)를 보여주고 있습니다. 이 프레임워크는 제한된 데이터와 불균형한 데이터셋에서도 효과적으로 작동하여 실제 대피 행동을 보다 잘 반영합니다. 또한, 독립적 실험을 통해 FLARE의 이유 기반 추론 능력을 입증하며, 대피 결정 예측에서 새로운 기준을 제시하고 있습니다.



### METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling (https://arxiv.org/abs/2502.17651)
- **What's New**: 이 논문에서는 효과적인 자동 챠트 생성을 위한 비전-언어 모델(Vision-Language Model, VLM)을 기반으로 하는 다중 에이전트 프레임워크인 METAL를 제안합니다. METAL는 각기 다른 전문 역할을 가지고 있는 에이전트들 간의 협업을 통해 복잡한 다중 모달 추론 과제를 분해합니다. 이를 통해 챠트 생성 정확도를 5.2% 향상시켜 기존에 비해 상당한 개선 효과를 보여줍니다.

- **Technical Details**: METAL 프레임워크는 주어진 참조 챠트 이미지를 기반으로 프로그래밍 사양을 학습하는 것을 목표로 합니다. 네 가지 전문 에이전트(Generation Agent, Visual Critique Agent, Code Critique Agent, Revision Agent)가 협력하여 직관적으로 진행되며 각 에이전트는 반복적으로 결과물을 개선합니다. 특히, 이 프레임워크는 시각적 평가와 코드 분석 시에 서로 다른 모달리티를 분리하는 방식을 채택하여 VLM의 자기 수정 능력을 향상시킵니다.

- **Performance Highlights**: METAL을 통해 얻은 실험 결과는 챠트 생성 정확도를 11.33% 이상 향상시키며, 이는 VLM이 시각적 이해와 코드 합성을 통합하는 능력을 크게 향상시킨다는 것을 보여줍니다. 또한, 테스트 시간 스케일링(timing scaling) 특성을 발견하여 컴퓨팅 예산이 증가함에 따라 METAL의 성능이 일관되게 개선된다는 것을 증명하였습니다. 이러한 성과는 VLM 기반의 비주얼 중심 코드 생성 향상에 있어 유망한 경로를 제공합니다.



### Synthetic Text Generation for Training Large Language Models via Gradient Matching (https://arxiv.org/abs/2502.17607)
Comments:
          15 pages, 5 figures, 4 tables

- **What's New**: 이번 연구에서는 실 데이터에서의 파인튜닝(fine-tuning) 시 LLM의 수렴성과 성능을 보장하는 인체 가독성이 있는 합성 텍스트를 생성하기 위한 최초의 이론적으로 엄밀한 방법론을 제안합니다. 이를 위해 Alternating Direction Method of Multipliers (ADMM) 기법을 적용하여 합성 예제의 임베딩을 반복적으로 최적화합니다. 최적화된 임베딩은 실제 데이터의 그라디언트와 유사한 텍스트 토큰 시퀀스로 매핑됩니다.

- **Technical Details**: 이 연구에서는 실 데이터의 그라디언트와 유사한 임베딩을 찾기 위해 이산 최적화 문제를 공식화합니다. 이 최적화 과정에서는 읽을 수 있는 텍스트를 보장하기 위해 낮은 perplexity를 요구하는 제약 조건을 추가합니다. ADMM을 통해 이산 최적화 문제를 해결함으로써, 실제 데이터에 대한 파인튜닝 결과에 가까운 솔루션으로 수렴함을 검증합니다.

- **Performance Highlights**: GrADmm 방식으로 생성된 합성 데이터가 실제 예제에 대해 최대 32.4% 향상된 성능을 보여줍니다. 또한, GrADmm을 통해 생성된 합성 데이터는 기존 LLM의 제로샷(zero-shot) 및 퓨샷(few-shot) 생성 방법보다도 최대 10.4% 더 나은 성능을 나타냅니다. 이 방식은 Llama-3.2-1B 및 OPT-1.3B와 같은 다른 LLM에 대한 파인튜닝에 있어서도 효용이 강조됩니다.



### Hallucination Detection in LLMs Using Spectral Features of Attention Maps (https://arxiv.org/abs/2502.17598)
Comments:
          Preprint, under review

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 환각(hallucination) 탐지를 위한 새로운 방법을 제안합니다. 기존의 attention map 기반 방법은 한계가 있었던 반면, 제안된 LapEigvals 방법은 Laplacian matrix의 상위-k 고유값을 활용하여 더욱 정확한 탐지가 가능합니다. 실험 결과, 이 방법은 최신의 환각 탐지 성능을 달성하였으며, 이를 통해 향후 연구의 방향성을 제시합니다.

- **Technical Details**: 제안된 LapEigvals 방법은 attention maps를 그래프 구조의 인접 행렬(adjacency matrix)로 해석하여 이를 통계적으로 분석합니다. 이를 통해 attention maps에서 유도된 Laplacian matrix의 고유값(eigenvalues)을 사용하여 환각을 탐지하는 Probe 모델의 입력 특성으로 활용합니다. 연구 결과, Laplacian의 고유값이 이전의 방법들보다 환각과 더 밀접한 관련을 가지고 있음을 보여주었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 LapEigvals는 기존 AttentionScore 방법보다 우수한 성능을 입증했으며, 다양한 데이터셋과 LLM에 걸쳐 이러한 결과가 나타났습니다. 또한, ablation study를 통해 자원의 견고성과 일반화 가능성을 강조하며, 향후 환각 탐지 방법의 발전에 기여할 논의의 기반을 마련했습니다.



### Training a Generally Curious Agen (https://arxiv.org/abs/2502.17543)
- **What's New**: 이 논문에서는 PAPRIKA라는 새로운 방법론을 소개합니다. PAPRIKA는 언어 모델이 특정 환경에 국한되지 않고 일반적인 의사결정 능력을 키울 수 있도록 하는 미세 조정(fine-tuning) 접근법입니다. 이 방법은 다양한 전략을 요구하는 여러 작업에서 생성된 합성(interaction data)을 기반으로 훈련하여 모델이 환경 피드백을 통해 새로운 작업에 적응할 수 있도록 학습합니다.

- **Technical Details**: PAPRIKA는 언어 모델이 다양한 과제에 대한 정보 추출 및 의사결정을 수행하도록 설계된 텍스트 기반 의사결정 작업의 식이로 구성됩니다. 이를 위해 기본 모델(base model)을 사용하여 상호작용 궤적(interaction trajectories)을 생성하고, 성공률에 따라 점수를 부여합니다. 또, Direct Preference Optimization의 변형을 통해 성공적인 궤적의 상대적 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, PAPRIKA로 미세 조정된 모델은 추가적인 훈련 없이도 완전히 새로운 작업에 학습된 의사결정 능력을 효과적으로 이전할 수 있음을 보여주었습니다. 이 연구는 합성 데이터 생성의 유용성을 강조하며, AI 시스템이 외부 세계와의 상호작용을 통해 새로운 순차적 의사결정 문제를 자율적으로 해결할 수 있는 가능성을 제시합니다.



### Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction (https://arxiv.org/abs/2502.17541)
- **What's New**: 이 연구에서는 데이터셋 기능 추출에 대한 새로운 접근 방식을 제안합니다. 기존의 단순한 기능 추출 방법들이 다양한 데이터셋에 대한 정확하고 다재다능한 설명을 생성하는 데 실패하는 반면, 이 방법은 신뢰성과 세밀한 제어를 제공합니다. 또한, 이는 전문가의 레이블링과 비교할 수 있는 압축되고 설명적인 표현을 유지하므로, 다양한 분야에 적용할 수 있습니다.

- **Technical Details**: 제안된 방법은 언어 모델(LLM)의 능력을 활용하여 데이터셋의 구조와 의미를 추출하는 이진 특징 집합을 최적화합니다. 이 방법은 무감독 자동 파이프라인을 통해 가능한 기능 집합을 생성하고, 데이터셋의 구조를 가장 잘 포착하는 서브셋을 추출합니다. 제공된 문맥에 따라 언어 모델이 원본 데이터를 재구성할 수 있는 능력을 평가하여 정보를 선택적으로 수집하는 방식으로 작동합니다.

- **Performance Highlights**: 이 연구는 두 가지 사례 연구를 통해 방법의 효과를 입증합니다. 첫 번째로, LLM jailbreak 전술에 대한 특징을 추출하여 인간이 제작한 전술들의 효율성과 다양성을 컴팩트하게 캡처했습니다. 두 번째로, 자동화된 피쳐 발견 방법을 통해 인간의 선호도와 일치하거나 그 성능을 초과하는 정확한 모델을 생성하는 데 성공하였습니다.



### PosterSum: A Multimodal Benchmark for Scientific Poster Summarization (https://arxiv.org/abs/2502.17540)
Comments:
          This paper includes a dataset of research posters with abstracts. We provide two cited examples ( arXiv:2211.11880 and arXiv:2210.07571 ) to illustrate reference summaries

- **What's New**: 이번 논문에서는 PosterSum이라는 새로운 멀티모달 벤치마크를 도입하여 시각적으로 복잡한 내용인 학술 포스터를 연구 논문 초록으로 요약하는 모델 개발을 지원하고자 합니다. PosterSum 데이터셋은 기계 학습 회의에서 발표된 16,305개의 포스터와 해당 초록을 포함하고 있으며, 이는 다양한 시각적 이해 도전과제를 제공합니다. 이 연구는 최신 Multimodal Large Language Models (MLLMs)가 학술 포스터 요약에서 직면하는 한계를 강조합니다.

- **Technical Details**: PosterSum 데이터셋은 각 포스터를 이미지 형식으로 제공하고 있으며, 포스터는 복잡한 레이아웃, 밀집한 텍스트 영역, 표 및 그림 등의 다양한 시각적 도전과제를 나타냅니다. 본 연구에서는 Segment & Summarize라는 계층적 방법을 제안하여 각 포스터를 일관된 영역으로 분할하고, 각 영역의 텍스트를 추출하여 지역 요약을 생성한 후, 이를 종합하여 포스터 전체를 아우르는 요약을 작성합니다. 이 과정에서는 추가 학습이나 미세 조정이 필요하지 않아 효율적입니다.

- **Performance Highlights**: 제안된 방법은 ROUGE-L 점수 24.18을 달성하여 기존의 MLLMs를 초과하며, 이는 학술 포스터 요약에서 새로운 기준을 설정합니다. PosterSum 데이터셋은 향후 멀티모달 과학 포스터 이해 연구에 기여할 수 있는 기초 자료가 됩니다. 또한, 이 연구는 MLLMs의 파인튜닝에의 유용성을 입증하여 제로샷 결과에 비해 유망한 개선 결과를 보여줍니다.



### The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? (https://arxiv.org/abs/2502.17535)
- **What's New**: 모델 압축 및 KV 캐시 압축이 LLM의 계산 및 저장 비용을 줄이기 위해 많은 관심을 받고 있습니다. 본 논문에서는 retrieval-augmented generation, multi-step reasoning, 외부 도구 사용과 같은 최근 LLM의 발전을 검토하고, 특정 LLM과 작업에 대해 더 작은 로또 LLM이 동일한 성능을 달성할 수 있다는 가설을 제시합니다.

- **Technical Details**: 현재 대부분의 LLM 압축 방법은 perplexity와 같은 기본적인 언어 작업에서만 성능을 보장하며, 실제 산업 상황에서는 충분한 성과를 보이지 않습니다. LLMs의 압축으로 인해 long-context retrieval 및 reasoning 능력이 감소할 수 있으며, KV 캐시 압축 또한 LLM의 긴 컨텍스트 이해 능력을 크게 제한합니다.

- **Performance Highlights**: 적응형 지식 검색(adaptive knowledge retrieval)에 관한 최근 연구는 LLM의 성능 향상과 모델 사이즈 및 지식 베이스 사이의 최적의 균형을 찾는 데 기여할 것으로 기대되고 있습니다. RAG(retrieval-augmented generation) 접근 방식은 특정 분야의 LLM 성능을 획기적으로 개선할 수 있음을 보여주며, 법률, 의료, 금융 분야에서의 LLM 적용 가능성을 높이고 있습니다.



### From Euler to AI: Unifying Formulas for Mathematical Constants (https://arxiv.org/abs/2502.17533)
Comments:
          50 pages, 6 figures

- **What's New**: 본 연구에서는수학 상수 $ank{	ext{π}}$와 관련된 방정식의 상관관계를 식별하고 통합하는 시스템을 제안했습니다. 이를 통해 457,145개의 arXiv 논문을 분석하여, 37%의 수학적 공식이 단일 수학 객체에서 유도될 수 있음을 제시했습니다. 이러한 접근은 AI 구동 발견을 위한 기초가 될 수 있습니다.

- **Technical Details**: 연구 방법론은 LaTeX 소스 코드를 활용하여 공식을 추출합니다. 278,242,582개의 고유한 문자열을 추출한 후, $	ext{π}$ 기符와 관계된 121,684개의 공식을 걸러내고 이들을 GPT-4o를 통해 분류합니다. 최종적으로, 시스템은 수학적 공식을 저비용으로 분석하고 유효성을 검증하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 이 연구에서는 $	ext{π}$와 같은 수학적 상수 뿐만 아니라, 그 외 다른 상수들도 다루어 넓은 적용 가능성을 보여주었습니다. 제안된 알고리즘은 기존 방식보다 더 효과적으로 수학적 발견을 수집하고 검증할 잠재력을 가지고 있습니다. 그 결과, 이 시스템은 방대한 수학 지식을 자율적으로 통합하는 가능성을 열어줍니다.



### Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation (https://arxiv.org/abs/2502.17521)
Comments:
          Github Link: this https URL

- **What's New**: 이번 논문에서는 데이터 오염(data contamination)의 위험을 줄이기 위해 고안된 정적(static)에서 동적(dynamic) 벤치마킹 방식의 변화에 대한 깊이 있는 분석을 수행합니다. 특히 동적 벤치마킹의 표준화된 평가 기준이 부족하다는 점을 강조하며, 이를 개선하기 위한 사례들을 제안합니다. 논문은 기존 연구의 한계점을 짚고, LLM(대규모 언어 모델)의 벤치마킹 방법론에 대한 종합적인 개요를 제공합니다.

- **Technical Details**: 본 연구는 정적 벤치마킹의 한계를 지적한 후, 시간에 따라 업데이트되는 벤치마킹 데이터세트를 사용하는 동적 벤치마킹 방법론을 제안합니다. 학습 단계에서 벤치마크 데이터가 모델의 훈련 데이터와 겹치지 않도록 하기 위해 다양한 기술적 접근 방식을 소개하며, 데이터 암호화(data encryption) 및 후속 오염 탐지(post-hoc contamination detection)와 같은 방법들이 있습니다. 하지만 이러한 정적 방법들의 한계로 인해 새로운 동적 벤치마킹 스킴이 도입되었습니다.

- **Performance Highlights**: 가장 주목할 만한 점은 기존의 동적 벤치마크가 제안된 평가 기준을 완전히 만족시키지 못한다는 것입니다. 이로 인해 기존의 평가 방법들이 모델의 실제 성능을 왜곡할 수 있는 문제점을 내포하고 있다는 것을 암시합니다. 이 논문에서는 데이터 오염의 위험을 줄이기 위한 동적 벤치마킹 방법에 대한 체계적인 조사를 실시하여, 향후 연구 방향에 대한 귀중한 인사이트를 제공합니다.



### SAE-V: Interpreting Multimodal Models for Enhanced Alignmen (https://arxiv.org/abs/2502.17514)
- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 해석 가능성을 높이기 위한 새로운 프레임워크인 SAE-V를 제안했습니다. SAE-V는 Sparse Autoencoders(SAEs) 개념을 확장하여 MLLM에 적용시키며, 이를 통해 모델의 동작과 데이터 품질을 세밀하게 해석할 수 있습니다. 이러한 접근 방식은 MLLM의 정렬 프로세스에서 데이터 필터링 기법을 활용하여 보다 효율적인 모델 정렬을 가능하게 합니다.

- **Technical Details**: SAE-V는 MLLM의 해석 가능성을 위해 설계된 메커니즘 해석 가능성 프레임워크입니다. 이 프레임워크는 텍스트와 이미지의 표현 공간을 융합하여 통합된 멀티모달 표현을 생성하는데 초점을 맞추고 있습니다. SAE-V는 MLLM의 교차 모달 상호작용을 어휘적으로 분석하여 모델의 품질과 정렬을 개선할 수 있는 기법을 제공합니다.

- **Performance Highlights**: 실험 결과, SAE-V 기반의 데이터 필터링 기법은 50%의 데이터로 110% 이상의 성능 향상을 달성했습니다. 이는 SAE-V가 MLLM의 해석 가능성과 정렬을 강화하는 데 효과적임을 입증합니다. 이와 함께 MLLM의 학습 과정에서의 특징 분포를 분석하여 멀티모달 이해 과제 성능과의 관계를 발견했습니다.



### Recurrent Knowledge Identification and Fusion for Language Model Continual Learning (https://arxiv.org/abs/2502.17510)
- **What's New**: Recurrent-KIF는 동적 매개변수 중요도 추정을 통해 지식 전이를 향상시키는 새로운 연속 학습(Continual Learning, CL) 프레임워크입니다. 이 접근법은 두 가지 루프 구조를 활용하여 새로운 작업에 신속하게 적응하고, 과거 지식을 관리하는 데 중점을 둡니다. 이러한 동적인 중요도 분포 추정을 기반으로 Recurrent-KIF는 복잡한 환경에서의 효과적인 학습을 실제로 구현합니다.

- **Technical Details**: Recurrent-KIF는 내부 루프(inner loop)와 외부 루프(outer loop)의 협업을 통해 작동합니다. 내부 루프는 새로운 작업에 적응하며 중요한 매개변수를 식별하는 역할을 하며, 외부 루프는 새로운 및 역사적인 지식의 융합을 관리합니다. 이 과정에서 중복 지식 가지치기(redundant knowledge pruning)와 핵심 지식 병합(key knowledge merging)을 통해 지식 융합이 이루어집니다.

- **Performance Highlights**: 실험 결과, Recurrent-KIF는 CF(변별적 망각)와 KT(지식 전이) 문제를 효과적으로 완화하는 것으로 나타났습니다. 다양한 모델 아키텍처와 크기(770M에서 13B까지)에서 우수한 성능을 발휘하며, CL 벤치마크에서 기존의 최첨단 방법들을 능가했습니다. Recurrent-KIF는 지식 융합 지식(fusion of knowledge)을 조정할 때 각 단계에서 최신 중요도 분포에 따라 적응적으로 최적화하여 모델 훈련 과정을 개선합니다.



### Protein Large Language Models: A Comprehensive Survey (https://arxiv.org/abs/2502.17504)
Comments:
          24 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 Protein LLMs(단백질 대형 언어 모델)에 대한 포괄적인 개요를 제공하는 최초의 연구로, 기존의 서베이들이 특정 측면이나 응용에 중점을 두었던 것과는 달리, 이 논문에서는 구조, 훈련 데이터셋, 평가 지표 및 다양한 응용 분야를 다루고 있습니다. 이는 단백질 과학에서의 혁신적인 발전에 기여할 것입니다.

- **Technical Details**: 저자들은 100편 이상의 연구 논문을 체계적으로 분석하여 최신 Protein LLMs의 구조적 분류법을 제안합니다. 이 모델들은 방대한 단백질 서열 데이터(large-scale protein sequence data)를 활용하여 더 높은 정확도를 얻는 방법을 분석하며, 단백질 공학 및 생물 의학 연구에서의 잠재력을 탐구합니다.

- **Performance Highlights**: Protein LLMs는 단백질 구조 예측, 기능 주석 및 디자인에서 더 효율적인 성능을 보여주며, 과학적 발견을 위한 필수 도구로 자리매김하고 있습니다. 논문은 단백질 과학 내 미래의 도전 과제와 방향성에 대해서도 논의합니다.



### Improving Value-based Process Verifier via Structural Prior Injection (https://arxiv.org/abs/2502.17498)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 Large Language Model(LLM) 추론 시나리오에서 Monte Carlo 샘플링을 통한 상태 가치 추정의 한계점을 극복하기 위해 구조적 사전 주입(structural prior injection) 방법을 제안합니다. Monte Carlo 추정이 가진 소음(noise)과 오차(errors)를 사전 정의된 범주형 분포의 기대값으로 매핑하여 해결합니다. 이 접근방식은 근본적으로 추정 오류를 분포 불일치 문제로 전환합니다.

- **Technical Details**: 논문에서는 Markov 결정 과정(Markov Decision Process, MDP) 및 Bellman 방정식을 통해 가치 기반 프로세스 검증기의 개념을 정립합니다. Monte Carlo 방법을 이용하여 상태 행동 가치를 추정하며, 샘플링의 한계로 인해 발생하는 오차를 통계 기반 거리(Statistics-based Distance)라는 새로운 미세 조정을 통해 측정합니다. 이 미세 조정은 다양한 목적 함수(mean-square error, cross-entropy)에 대한 최적화를 도와줍니다.

- **Performance Highlights**: 구조적 사전 주입을 통해 다양한 목표 함수에서 값 기반 프로세스 검증기의 성능을 향상시켜 약 1~2점의 성능 개선을 보여줍니다. 이는 적은 비용으로 실현되며, 동일한 최적 솔루션을 가지고도 서로 다른 구조적 사전(definition)에 따라 성능 차이가 크다는 것을 보여줍니다. 이러한 결과는 구조적 사전 주입이 미래 연구에서 중요한 방향임을 시사합니다.



### Brain-to-Text Decoding: A Non-invasive Approach via Typing (https://arxiv.org/abs/2502.17480)
Comments:
          15 pages, 5 figures

- **What's New**: 본 연구에서는 침습적인 뇌-컴퓨터 인터페이스(BCI)의 대안으로 비침습적인 Brain2Qwerty 모델을 도입합니다. 이 모델은 뇌 활동을 통해 문장 생성을 디코드할 수 있는 기능을 갖추고 있으며, 35명의 건강한 자원자와의 실험을 통해 그 효능을 입증했습니다. 특히, 이 연구는 기존 EEG 기반의 기술보다 우수한 성능을 보여주며 비침습적인 방법의 가능성을 열었습니다.

- **Technical Details**: Brain2Qwerty는 EEG(전기 뇌파)와 MEG(자기뇌파측정)를 사용하여 문장 생산을 디코드하는 새로운 딥 러닝 아키텍처입니다. 연구자들은 해당 모델을 훈련시키기 위해 35명의 참가자들에게 키보드로 간단한 문장을 입력하도록 하였고, 이를 통해 수집된 신호로부터 문자를 디코드했습니다. 실험 결과, MEG를 이용한 경우 평균 32%의 캐릭터 오류율(CER)을 기록하였고, 이는 EEG의 67%와 상당한 차이를 보였습니다.

- **Performance Highlights**: Brain2Qwerty는 MEG를 사용할 때 최고의 참가자에서 19%라는 낮은 CER을 달성했으며, 다양한 문장을 학습 세트 외에서도 완벽하게 디코딩할 수 있었습니다. 또한, EEGNet과의 비교에서 이 모델이 CER에서 1.14배, MEG에서 2.25배의 성과 향상을 이뤘음을 보였습니다. 이러한 결과는 비침습적인 브레인-컴퓨터 인터페이스의 안전하고 효과적인 개발 가능성을 제시합니다.



### ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis (https://arxiv.org/abs/2502.17475)
- **What's New**: ECG-Expert-QA는 ECG 해석에서 진단 능력을 평가하기 위한 포괄적인 멀티모달 데이터셋으로, 실제 임상 데이터와 체계적으로 생성된 합성 사례를 통합하였습니다. 이 데이터셋은 47,211개의 질문-답변 쌍으로 구성되어 있으며, 복잡한 사례 해석을 포함한 다양한 임상 시나리오를 다룹니다. 이를 통해 고유한 진단 작업을 통해 의료 언어 모델을 종합적으로 평가할 수 있는 기반이 마련되었습니다.

- **Technical Details**: 이 연구는 전통적인 평가 방법의 효율성 문제와 데이터셋의 복잡성 부족을 해결하기 위해 ECG-Expert-QA 데이터셋을 개발했습니다. 데이터셋은 기본 의료 지식 검증 모듈, 임상적 추론 평가 모듈, 위험 관리 모듈의 세 가지 핵심 평가 모듈로 구성되며, 이는 환자 예후 예측, 다중 모달 정보 통합 등을 포함합니다. 또한, 데이터셋은 윤리적 차원도 포함하여, 의료 AI의 결정 안전성을 평가하기 위한 기준을 제시합니다.

- **Performance Highlights**: 이 데이터셋은 기존의 의료 데이터셋에 비해 복잡한 진단 작업 비율이 15.3%에 달하며, 다양한 언어로 지원되어 cross-cultural 연구에 기여합니다. 특히, innovative evaluation dimensions인 반사적 추론과 기억 교정 메커니즘을 도입하여 모델의 의사 결정 논리의 임상적 합리성과 강건성을 평가합니다. 이러한 특성으로 인해 ECG-Expert-QA는 AI 보조 ECG 해석 발전의 중요한 벤치마크로 자리잡고 있습니다.



### Bridging Brain Signals and Language: A Deep Learning Approach to EEG-to-Text Decoding (https://arxiv.org/abs/2502.17465)
Comments:
          21 pages, 11 figures, and 6 tables

- **What's New**: 이 연구는 EEG(뇌파) 신호를 텍스트로 변환하는 전통적인 닫힌 어휘(closed-vocabulary) 접근 방식을 넘어서, 개인 특정 학습 모델을 자연어 처리(NLP) 방법과 통합하는 특별한 프레임워크를 소개합니다. 이로 인해 EEG 신호가 개별뇌 및 의미의 깊이를 포착하며, 더욱 풍부한 문장을 생성할 수 있도록 혁신적인 방안을 모색합니다. 연구는 신경망을 훈련시켜 매우 의미 있는 텍스트 생성을 가능하게 하며, ZuCo 데이터셋 분석을 통해 다른 방법들과 비교했을 때 향상된 성능을 보여줍니다.

- **Technical Details**: 이 프레임워크는 양방향 GRU와 변환기 인코더를 사용하여 개인 특화 피쳐를 추출하는 뇌 모듈, EEG에서 얻은 특성을 활용하여 개방된 어휘 텍스트를 생성하기 위해 BART를 통합한 언어 모듈, 의미적 및 문법적 수정을 위해 GPT-4를 활용하는 정제 모듈로 구성됩니다. 사용자 친화적인 웹 플랫폼을 통해 EEG 데이터를 업로드하고 텍스트 생성을 비음성화하는 등의 기능이 구현되어 있으며, 이는 다양한 EEG 데이터셋과 하드웨어와의 호환성을 보장합니다.

- **Performance Highlights**: 연구 결과는 BLEU, ROUGE 및 BERTScore와 같은 성능 지표에서 현재 사용되는 방법들에 비해 더 높은 성과를 달성했습니다. 이 접근법은 개인 간의 뇌 신호 변동성을 이해하고 의미있는 텍스트 생성을 효과적으로 수행할 수 있음을 입증합니다. 이 시스템은 어떤 환경에서도 효율적인 뇌-주관 텍스트 시스템을 개발하는 데 기여할 수 있으며, 혁신적인 보조 기술 개발과 개인화된 의사소통 시스템을 통해 인류와 컴퓨터 간의 상호작용 가능성을 확장하는 데 중점을 두고 있습니다.



### Polarized Online Discourse on Abortion: Frames and Hostile Expressions among Liberals and Conservatives (https://arxiv.org/abs/2311.16831)
- **What's New**: 이번 연구는 미국 내에서 낙태(abortion)를 둘러싼 정치적 분열이 시간이 지남에 따라 어떻게 반영되는지를 분석한 최초의 포괄적인 연구 중 하나입니다. 2022년 1월부터 2023년 1월까지의 1년 간 350만 개 이상의 트윗을 분석하여, 낙태에 대한 공개 담론에서 나타나는 적대적인 표현의 빈도와 정치적 프레임(frame)의 사용 양상을 조사했습니다. 특히 Roe v. Wade 판결의 뒤집힘과 같은 주요 사건들에 대한 반응과 관련하여, 진보 세력과 보수 세력 간의 긴밀한 상관관계를 발견했습니다.

- **Technical Details**: 연구팀은 최신 변환기 기반 분류기(transformer-based classifiers)를 사용하여 사용자들의 이념(ideology)을 추정하고, 적대적 표현을 식별하였습니다. 연구는 주로 분노(anger), 독성(toxicity), 모욕(insults), 저속한 언어(obscenities), 그리고 증오 발언(hate speech)으로 구성되어 있습니다. 이러한 데이터는 리버럴(liberals)과 컨서버티브(conservatives) 간의 이념적 비대칭성을 정량화하고, 그들이 사용하는 다섯 가지 주요 프레임(예: 종교, 태아의 법적 지위, 낙태 예외, 여성 건강 및 신체 자율성)을 추출하는 데 활용되었습니다.

- **Performance Highlights**: 연구 결과, 컨서버티브는 전반적으로 리버럴보다 더 많은 분노적이고 독성이 강한 발언을 했지만, 두 그룹은 주요 사건 이후 서로의 적대적 발언을 거울처럼 반영하는 경향이 있었습니다. 또한, 각 그룹은 낙태에 관한 다소 다른 관점을 드러내며, 특정 프레임이 한 쪽에서 선호될 때 다른 쪽에서 적대적 반응을 유도하는 경향이 있음을 발견했습니다. 이러한 발견은 디지털 시대의 양극화된 온라인 담론을 더 잘 이해하는 데 기여합니다.



### SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents (https://arxiv.org/abs/2502.15538)
Comments:
          26 pages, 5 figures, 23 tables

- **What's New**: 이번 연구는 인간의 사회적 전략이 사회적 에이전트에 통합되는 과정에 대한 부족한 연구를 해결하기 위해 SOTOPIA-Ω 프레임워크를 제안합니다. 이 프레임워크는 협상 이론에 영감을 받은 다단계 추론 전략과 두 가지 간단한 직접 전략을 전문가 에이전트에 동적으로 주입하여 고품질 사회적 대화 훈련 데이터셋을 자동으로 구축합니다.

- **Technical Details**: SOTOPIA-Ω 프레임워크에서는 Social Instruction Following (S-IF)라는 개념을 도입하고, 사회적 능력을 보완하는 두 가지 새로운 S-IF 평가 지표를 제안합니다. 이를 통해 에이전트가 사회적 목표를 달성하기 위한 성능을 정량화하고 분석할 수 있습니다.

- **Performance Highlights**: 7B 모델을 사용하여 고품질 데이터셋으로 훈련한 결과, 이 모델들은 전문가 에이전트인 GPT-4를 능가하여 사회적 목표를 더 효과적으로 달성하는 것으로 나타났습니다. 실험 분석을 통해 동적 구성의 이점이 입증되었으며, 이는 에이전트가 오랜 교착 상태를 타개하는 데 특히 효과적임을 보여줍니다.



New uploads on arXiv(cs.IR)

### Rank1: Test-Time Compute for Reranking in Information Retrieva (https://arxiv.org/abs/2502.18418)
- **What's New**: Rank1은 테스트 시간에 계산(compute)을 활용하여 훈련된 최초의 재순위 모델로, 이를 통해 작은 모델의 성능을 신속하게 개선할 수 있는 가능성을 보여줍니다. MS MARCO 데이터셋에서 60만 개 이상의 R1 추론 추적 예제를 수집하고 오픈소스로 공개하여 연구자들이 활용할 수 있게 하였습니다. 이 모델은 고급 추론 및 지시 준수 데이터셋에서 최첨단 성능을 기록하며, 유저 입력 프롬프트에 잘 반응하는 특성을 보여줍니다.

- **Technical Details**: Rank1은 OpenAI의 o1, Deepseek의 R1과 같은 추론 언어 모델을 사용하여, 정보 검색(Information Retrieval) 맥락에서 테스트 시간에 계산(compute)을 활용하도록 설계되었습니다. 모델은 쿼리와 문서를 동시에 추론할 수 있도록 최적화되었으며, 다양한 데이터셋에서 성능을 입증하였습니다. 특히, 이 모델은 Hard negatives와 Easy negatives의 조합을 활용하여 더 나은 학습을 이끌어냈습니다.

- **Performance Highlights**: Rank1은 브라이트(BRIGHT) 벤치마크에서 최첨단 성능을 달성하였고, 전통적인 IR 벤치마크에서 놀라운 성능을 보여주는 등 상당한 향상을 이뤄냈습니다. 사용자가 주는 프롬프트에 대한 적응성도 높아, 다양한 쿼리에 대해 효율적이고 설명 가능한 추론 체인을 제공합니다. 이는 더 나아가 사용자나 에이전트 기반 RAG 시스템이 활용할 수 있는 투명한 시스템으로 발전할 수 있게 합니다.



### A Unified Bayesian Perspective for Conventional and Robust Adaptive Filters (https://arxiv.org/abs/2502.18325)
- **What's New**: 이 논문에서는 전통적인 적응 필터(adaptive filters)의 기원과 해석에 대한 새로운 관점을 제시합니다. Bayesian 원칙을 사용하여 상태 공간 모델(state-space model)에서 반복적 추론을 적용하고, 해결책의 구조에 대한 일련의 간소화를 통해 여러 기존 적응 필터들을 통합된 프레임워크에서 유도할 수 있습니다. 특히, Gaussian 모델 하에 LMS, NLMS, Kalman 필터와 같은 잘 알려진 솔루션을 도출하고, 비-Gaussian 노이즈를 사용하는 경우 새로운 적응 필터 패밀리를 제시합니다.

- **Technical Details**: 적용된 상태 공간 모델에서는 관측 노이즈(ηt)가 0 평균의 랜덤 변수로 모델링되고, 독립된 0 평균 Gaussian 벡터들로 구성됩니다. 이 연구는 주어진 시간을 기준으로 정보를 바탕으로 posterior distribution을 추출하는 것을 목표로 하며, 다양한 복잡도의 적응 알고리즘을 체계적이고 명확한 방법으로 유도합니다. 특히 Laplacian 노이즈를 가정할 경우, 잘 알려진 signed-error LMS 알고리즘의 자연스러운 일반화를 이루는 알고리즘을 도출할 수 있습니다.

- **Performance Highlights**: 새로운 강건 적응 필터는 비-Gaussian 노이즈 환경에서 성능을 현저히 향상시킵니다. 실험은 합성 데이터를 사용하여 진행되었고, 결과는 이론적 도출이 유효함을 입증합니다. 특히, 간소화 가정이 늘어날수록 성능이 저하되는 경향이 관찰되었으며, 반면 강건 적응 필터는 노이즈의 비-Gaussian 분포 존재 하에서도 개선된 결과를 보여주었습니다.



### Neural Network Graph Similarity Computation Based on Graph Fusion (https://arxiv.org/abs/2502.18291)
Comments:
          9 pages, 4 figures, 4 tables

- **What's New**: 본 연구에서는 그래프 유사성 계산을 위한 새로운 접근 방식을 제안합니다. 그래프 쌍의 노드 시퀀스를 단일 대형 그래프로 결합하는 그래프 융합(graph fusion) 기법을 도입하여, 그래프 간의 상호 작용을 쉽게 수행하고, 전역적 주의 메커니즘을 통해 크로스 그래프 인사이트를 수집합니다. 이러한 방법은 이전의 복잡한 계산 부담을 줄이고, 두 그래프 간의 상호 작용을 효율적으로 처리할 수 있게 해줍니다.

- **Technical Details**: 연구에서는 그래프 융합 모듈을 통해 두 그래프의 노드 시퀀스를 하나의 큰 그래프로 통합하여, 트랜스포머 구조와 전역적 주의 메커니즘을 사용해 상호 작용을 인코딩합니다. 이 시스템은 그래프 쌍의 원래 노드 시퀀스에 따라 큰 그래프를 원래 두 그래프로 다시 나누고, 노드 간의 연관성을 고려하여 저차원 노드 간 상호 작용 계산을 수행합니다.

- **Performance Highlights**: 다섯 개의 공개 데이터 세트에 대한 광범위한 테스트 결과, 제안된 모델인 그래프 융합 모델(Graph Fusion Model, GFM)이 그래프-그래프 분류 및 회귀 작업에서 기존의 최첨단 기법들보다 우수한 성능을 보이며, 새로운 성능 및 효율성 기준을 설정했습니다. 이 모델은 그래프 간의 정보를 동시에 상호작용시키는 것이 특징입니다.



### HyperG: Hypergraph-Enhanced LLMs for Structured Knowledg (https://arxiv.org/abs/2502.18125)
- **What's New**: 본 논문에서는 LLMs의 구조적 지식 처리 능력을 향상시키기 위한 새로운 하이퍼그래프 기반 생성 프레임워크인 HyperG를 제안합니다. HyperG는 희소 데이터에 문맥 정보를 추가하고, 이를 통해 복잡한 구조적 관계를 인코딩할 수 있도록 설계되었습니다. 이를 통해 LLMs가 다양한 실제 작업에서 더 효율적으로 데이터의 구조를 이해하고 활용할 수 있도록 합니다.

- **Technical Details**: HyperG는 하이퍼그래프 구조를 활용하여 구조적 지식 내의 의미적 일관성(semantic consistency), 순서 불변성(order invariance), 계층적 종속성(hierarchical dependencies)을 캡처합니다. 이 시스템은 희소한 테이블 셀을 문맥 정보로 증강한 후, 태스크 특정 질문을 포함하여 임베디드된 의미적 지식을 하이퍼그래프를 통해 전파하는 Prompt-Attentive Hypergraph Learning(PHL) 모듈을 사용합니다. 이러한 접근 방식은 구조적 데이터에 대한 LLMs의 이해력과 추론 능력을 증대시키는 데 기여합니다.

- **Performance Highlights**: HyperG의 효과성과 일반화 가능성을 검증하기 위해, 본 연구에서는 다양한 하류 작업에서 시스템의 성능을 평가하는 광범위한 실험을 진행했습니다. 실시된 실험들은 기존의 방법들과 비교할 때 HyperG가 LLMs의 구조적 지식 처리 능력을 더욱 향상시킬 수 있음을 입증했습니다. 또한, 이 프레임워크는 희소한 데이터 문제를 해결하는 데 중점을 두어, 정보 전파를 통해 보다 풍부한 지식을 생성할 수 있는 가능성을 보여주었습니다.



### Tip of the Tongue Query Elicitation for Simulated Evaluation (https://arxiv.org/abs/2502.17776)
- **What's New**: 본 논문은 Tip-of-the-Tongue (TOT) 검색 시스템을 개선하기 위한 두 가지 새로운 방법을 제시합니다. GPT와 같은 대형 언어 모델(LLM)과 인간 참여자를 활용하여 TOT 검색 쿼리를 생성하는 방법을 소개하며, 이는 기존의 CQA 플랫폼에 의존하는 데이터 수집 방식을 극복하는 데 기여할 것입니다. 이 방법들은 영화, 랜드마크, 인물 등의 다양한 도메인에서 TOT 쿼리를 효율적으로 수집하고 평가할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 저자들은 LLM 기반의 TOT 사용자 시뮬레이터를 개발하여 합성 쿼리를 대규모로 생성하고, 이 쿼리들이 CQA 기반 TOT 쿼리와 어떤 연관성을 가지는지를 검증했습니다. 또한, 인간 참여자를 통한 쿼리 수집 인터페이스를 만들어 자연스러운 TOT 쿼리를 수집할 수 있는 방법을 제시하였습니다. 이러한 접근 방식은 CQA 데이터 의존도를 줄이고, 여러 도메인에서 데이터 수집의 범위를 확장하는 데 도움을 줍니다.

- **Performance Highlights**: TOT 쿼리와 관련된 시스템의 순위 상관관계 및 언어 유사성 분석을 통해 LLM을 활용한 합성 쿼리가 CQA 기반 쿼리와 효과적으로 일치함을 입증했습니다. 또한, 이 연구 결과는 TREC 2024 TOT 트랙과 TREC 2025 TOT 트랙에서 검증된 쿼리로 활용될 예정입니다. 저자들은 TOT 쿼리 생성을 위한 소스 코드를 공개하고, 사용자 인터페이스와 함께 유용한 시각적 자극 자료도 제공합니다.



### External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation (https://arxiv.org/abs/2502.17494)
Comments:
          Accepted by the ACM Web Conference (WWW) 2025 Industrial Track as Oral Presentation

- **What's New**: 본 논문에서 제안된 External Large Foundation Model (ExFM) 프레임워크는 산업 규모의 광고 추천 시스템에서 간과된 두 가지 주요 도전 과제를 해결하기 위해 설계되었습니다. 첫 번째 과제는 훈련 및 추론 지연이 제한되어 있는 것으로, 기존의 방법들이 대형 모델의 훈련과 추론 비용을 증가시키는 문제를 가지고 있었습니다. 두 번째 과제는 데이터 분포가 동적으로 변화하는 대량의 스트리밍 데이터가 사용되며, 이로 인해 모델이 항상 최신 데이터에 적합해야 하는 문제입니다.

- **Technical Details**: ExFM은 외부 증류(External Distillation) 기법을 활용하여 훈련 데이터 및 유지 비용을 절감하면서, 여러 학생 모델(Vertical Models, VMs)에 걸쳐 공통된 Teacher 모델(Foundation Model, FM)의 예측을 제공합니다. Auxiliary Head (AH)와 Student Adapter (SA)를 도입하여 FM가 VMs에 전이되는 데이터 분포의 격차를 완화하고, Freshness Gap을 감소시켜 모델의 성능을 향상시키는 방법론을 제안합니다. 이를 통해 ExFM은 재훈련을 최소화하면서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: ExFM 프레임워크를 도입한 실험 결과, 내부 산업 규모 데이터셋과 공개 데이터셋 모두에서 성능 향상이 확인되었습니다. ExFM은 수조 개의 파라미터를 가진 모델을 메타 플랫폼에 활용할 수 있도록 하여, 다양한 도메인 및 작업의 VMs에서 Promising한 성능 개선을 보여주었습니다. 또한, 하이퍼파라미터의 영향을 분석한 결과도 제공하여 이 접근법의 효용성을 강조합니다.



### DRAMA: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers (https://arxiv.org/abs/2502.18460)
- **What's New**: 이번 연구는 DRAMA라는 훈련 프레임워크를 도입하여 대형 언어 모델(LLMs)을 사용하여 더 작고 일반화 가능한 조밀한 검색기를 생성하는 방법을 제안합니다. 기존의 상태에서도 조밀한 검색기는 대규모의 쿼리 처리에 대한 높은 컴퓨팅 비용과 지연 시간이 문제로 지적되었습니다. DRAMA는 LLM 기반 데이터 증강을 활용하여 훈련 데이터의 질을 높이는 동시에 조밀 검색기 성능을 극대화할 수 있습니다.

- **Technical Details**: DRAMA는 약 10억 개 미만의 매개변수를 가진 조밀한 검색기를 효과적으로 만들기 위해 LLM의 축소된 버전을 백본으로 사용합니다. 연구에서는 Llama3 모델을 기반으로 하여 다양한 LLM 증강 데이터를 단일 단계 대조 학습(setup of contrastive learning)에서 훈련합니다. 이러한 접근을 통해 다양한 언어와 긴 컨텍스트에 대한 우수한 능력을 발휘하며, 기존 인코더 기반 검색기의 성능을 넘어서는 결과를 얻었습니다.

- **Performance Highlights**: 실험 결과 DRAMA는 BEIR, MIRACL과 여러 다국어 검색 작업에서 높은 성능을 달성하며, 조밀한 검색기의 일반화 능력을 증명하였습니다. 또한, 이 프레임워크는 대형 언어 모델의 발전과 조화를 이루어 효율성과 일반화 간의 격차를 메워주는 기술적 가능성을 보여줍니다. 이 연구는 다양한 검색 작업에서 일관되게 우수한 성능을 보이는 소형 검색기의 잠재력을 강조합니다.



### How Vital is the Jurisprudential Relevance: Law Article Intervened Legal Case Retrieval and Matching (https://arxiv.org/abs/2502.18292)
- **What's New**: 이 논문에서는 법적 사례 검색(LCM)과 유사 사례 매칭(LCM) 간의 문제를 보다 효과적으로 해결하기 위한 최첨단 모델인 LCM-LAI를 제안합니다. 이는 법률 문서의 예측 부과 작업(LAP)을 통해 법적-합리적 정보를 포착하며, 이전 연구에서의 가정에 의존하지 않고 진행됩니다. 또한, LCM-LAI는 법률 분배에 기반한 법적-합리적 유사성을 평가하기 위해 특별한 주의 메커니즘을 도입하여 기존의 의미적 유사성보다 더 효율적입니다.

- **Technical Details**: LCM-LAI는 종속 멀티 태스크 학습(framework) 구조를 채택하여 법률 사건의 사실 서술에서 법적-합리적 정보를 포착합니다. 이 구조는 효율적인 법률 문서의 예측 부과(sub-task) 기술을 포함하고 있어, 법률 기사가 입력으로 필요하지 않은 방식으로 작동합니다. LCM-LAI는 또한 법적 사건 간의 상호작용을 모델링하기 위해 혁신적인 article-aware attention 메커니즘을 도입하여, 사건 간 문장에서 법적-합리적 상관관계를 측정합니다.

- **Performance Highlights**: 다양한 실제 데이터 세트를 기반으로 한 실험 결과, LCM-LAI는 LCR 및 LCM 작업에서 최첨단 성능을 달성하였습니다. 특히, 법적 사항들을 처리하는 능력이 개선되어 기존 모델들보다 더 높은 정확도를 보여줍니다. 이러한 성과는 지능형 법률 시스템에서의 판결 지원 및 적절한 선례를 제공하는 데 큰 기여를 할 것으로 예상됩니다.



### LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers (https://arxiv.org/abs/2502.18139)
Comments:
          First submit

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)라는 새로운 접근 방식을 제안합니다. 기존의 RAG 방식은 쿼리 재작성을 통해 사용자 의도를 명확히 하고, 하이브리드 검색을 통해 검색 범위를 확장하는데 초점을 맞추었습니다. 그러나 이러한 방식은 쿼리 재작성이 밀접하게 연결되어 있어 하이브리드 검색의 호환성을 저해하고 있습니다. 이에 따라, LevelRAG는 복잡한 쿼리를 독립적인 원자 쿼리로 분해할 수 있는 고수준 검색기를 도입하여 이 문제를 해결합니다.

- **Technical Details**: LevelRAG는 고수준 검색기와 저수준 검색기(희소 검색기, 웹 검색기, 밀집 검색기)를 결합하여 정보 검색 로직을 최적화합니다. 고수준 검색기는 사용자의 쿼리를 원자 쿼리로 변환하여 저수준 검색기에게 전달하며, 이들은 해당 쿼리를 통해 데이터베이스에서 정보를 검색합니다. 특히 희소 검색기는 Lucene 구문을 사용하여 키워드 검색의 정확성을 높이며, 밀집 검색기는 복잡한 쿼리를 다루는 데 강점을 가지고 있습니다. 이 구성 요소들은 함께 작용하여 검색 프로세스의 완전성과 정확성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, LevelRAG는 단일 및 다단계 질문 응답 작업에서 기존 RAG 방법보다 우수한 성능을 보였습니다. 특히, LevelRAG는 최신 고급 모델인 GPT4o를 능가하는 응답 품질을 보여 주었습니다. 또한, 제안된 희소 검색기만으로도 기존의 다양한 방법보다 더 나은 성능을 발휘한 것으로 나타났습니다.



### AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages (https://arxiv.org/abs/2502.18020)
- **What's New**: 이 논문에서는 다국어 모델을 위한 새로운 하이브리드 지식 증류(knowledge distillation) 접근법을 제안합니다. 기존의 방법들이 다국어 모델, 특히 저자원(low-resource) 언어에 대한 성능 유지를 어렵게 하는 문제를 해결하기 위해, 전통적인 지식 증류 방식과 단순화된 attention matching 메커니즘을 결합했습니다. 이로 인해, 전통적인 다국어 모델보다 유의미하게 작은 크기의 학생 모델 구조를 도입하여, 아프리카 언어 다섯 개에 대한 평가에서 효과를 입증했습니다.

- **Technical Details**: 우리가 제안한 하이브리드 증류 프레임워크는 지식 증류와 attention matching을 결합하여 학생 모델이 교사 모델의 출력 분포와 내부 attention 패턴을 모두 학습하도록 합니다. 특히, 매우 compact한 다국어 학생 모델을 설계하였으며, 이는 기존 모델보다 훨씬 작은 hidden dimension을 가집니다. 실험적으로, 아프리카어 저자원 언어인 Kinyarwanda, Swahili, Hausa, Igbo, Yoruba에 대해 이 접근법을 평가하고, 모델 크기를 85% 이상 줄이면서도 경쟁력 있는 성능을 달성했습니다.

- **Performance Highlights**: 이 연구의 실험 결과는 제안된 하이브리드 접근법이 교사 모델과 비교하여 성능 면에서 경쟁력을 유지함을 보여줍니다. 학생 모델은 원래 모델의 성능에서 85% 이내의 정확도를 유지하면서도, 연산 자원을 현저히 절감할 수 있었습니다. 이는 저자원 환경에서 다국어 모델을 배치하는 데 유용한 실용적인 프레임워크를 제공하며, 아프리카 언어와 관련된 응용 프로그램에서 큰 혜택을 제공합니다.



### ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents (https://arxiv.org/abs/2502.18017)
- **What's New**: 이번 논문에서는 시각적으로 풍부한 문서에서 정보를 이해하는 데 있어 기존의 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 데이터셋인 ViDoSeek을 소개합니다. ViDoSeek은 복잡한 추론을 요구하는 문서에서 RAG 성능을 평가하도록 설계되었습니다. 이를 통해 현재 RAG 접근 방식의 주요 한계도 식별하였으며, 특히 시각적 검색 방법과 추론 토큰 할당의 부족 문제가 강조되었습니다.

- **Technical Details**: ViDoRAG라는 새로운 다중 에이전트 RAG 프레임워크를 제안하여 시각적 문서에서 복잡한 추론을 위한 개선점을 제시합니다. 이 프레임워크는 Gaussian Mixture Model (GMM) 기반의 하이브리드 전략을 채택하여 다중 모달 검색(multi-modal retrieval)을 효과적으로 처리합니다. 또한 탐색, 요약, 반영 과정을 포함하는 반복적인 에이전트 워크플로우를 통해 모델의 추론 능력을 배양합니다.

- **Performance Highlights**: ViDoSeek에서 진행된 광범위한 실험 결과, ViDoRAG는 기존 방법들보다 10% 이상 성능이 향상된 것으로 나타났습니다. 이 결과는 ViDoRAG이 RAG 분야에서 테스트 시간 확장(test-time scaling) 연구에 대한 새로운 방향을 제공함을 나타냅니다. ViDoRAG은 경쟁적인 ViDoSeek 기준에서 기존 방법들과 비교하여 우수한 성능을 보였습니다.



### MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification (https://arxiv.org/abs/2502.17987)
- **What's New**: 이 논문에서 새로운 점은 저자들이 저자원 (low-resource) 반투 (Bantu) 언어에 특화된 텍스트 분류 모델 MAGE(Multi-Head Attention Guided Embeddings)를 소개한 것이다. 이 모델은 Language-Independent Data Augmentation (LiDA)을 통해 데이터 포인트를 선택적으로 향상시킴으로서 텍스트 분류 성능을 개선하는데 중점을 두고 있다. 특히, MAGE는 데이터 부족 문제를 해결하면서도 반투 언어의 고유한 구문론적(syntactic) 및 의미론적(semantic) 특성을 효과적으로 다루도록 설계되었다.

- **Technical Details**: MAGE는 LiDA 프레임워크를 기반으로 하여 중대한 혁신을 도입하고 있다. 전통적인 Denoising Autoencoder 대신 Variational Autoencoder (VAE)를 도입하여 더욱 표현력이 풍부하고 다양한 합성(augmented) 임베딩을 생성한다. 또한, Multi-Head Attention 메커니즘을 활용하여 임베딩에서 중요한 특징을 강조함으로써 저자원 언어에서 구문론적 및 의미론적 뉘앙스를 더 잘 포착할 수 있도록 한다.

- **Performance Highlights**: MAGE는 AfriSenti SemEval 데이터셋을 사용하여 감정 분류(sentiment classification) 성능 평가를 수행하였으며, 저자원 환경에서 기존의 기준 방법들보다 우수한 성능을 나타냈다. MAGE는 데이터 부족 문제를 해결함과 동시에 다른 저자원 언어 계열로의 텍스트 분류 능력을 확장할 수 있는 스케일러블한 프레임워크로 자리잡고 있다. 이 연구는 향후 저자원 언어 처리 및 분류 작업에 대한 연구의 기초를 제공하고 있으며, NLP 기술의 포괄성과 일반화 가능성을 높이는 방향으로 나아가고 있다.



### On Synthetic Data Strategies for Domain-Specific Generative Retrieva (https://arxiv.org/abs/2502.17957)
- **What's New**: 이 논문은 도메인 특화 코퍼스에 대한 생성 검색 모델을 개발하기 위한 합성 데이터 생성 전략을 조사합니다. 특히, 쿼리 해석을 위해 LLM(Large Language Model)로 생성된 다양한 쿼리를 활용하여 문서 요청을 다루고, 두 단계의 훈련 프레임워크에서 선호 학습을 통해 문서 순위를 개선하는 방법을 제시합니다. 실험을 통해 공개 데이터셋에서 합성 데이터 생성 및 하드 네거티브 샘플링 접근 방식의 효과를 입증하고 있습니다.

- **Technical Details**: 연구는 생성 검색 모델 훈련에서 데이터 전략의 중요성을 강조하며, 두 단계의 훈련 프레임워크를 도입하여 첫 단계에서는 문서 식별자 디코딩을 위한 합성 데이터를 주로 사용합니다. 두 번째 단계에서는 모델의 순위 향상을 위한 선호 학습을 진행하며, 이 과정에서 자기 생성된 데이터를 활용하여 네거티브 샘플을 수집합니다. 특히 고급의 하드 네거티브 후보 선택이 모델 성능에 미치는 영향을 검토합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 다중 세분화 및 제약 기반 쿼리가 문서 검색 성능을 크게 향상시킨다고 밝혔습니다. 또한, 문서 식별자 관련 데이터 선택 전략이 다른 유형의 식별자에도 잘 일반화됨을 시연하였습니다. 마지막으로, RPO(Regularized Preference Optimization)를 사용하여 순위 성능이 효과적으로 개선되며, 고품질 하드 네거티브 후보가 성능에 긍정적인 영향을 미친다는 사실도 확인되었습니다.



### Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness (https://arxiv.org/abs/2502.17921)
- **What's New**: 이번 논문에서는 추천 시스템의 성별 편향을 정량화하기 위한 포괄적인 메트릭스를 새롭게 도입했습니다. 특히, 추천 항목의 범주를 고려하여 세분화된 공정성 평가의 중요성을 강조합니다. 기존의 성별 관련 편향 평가가 놓치고 있는 뉘앙스를 포착할 수 있는 방법에 대해 다루고 있습니다. 또한, Catagory-aware fairness metric을 훈련 시 정규화 항으로 사용하여 모델의 출력에서 편향을 효과적으로 최소화할 수 있음을 보여줍니다.

- **Technical Details**: 추천 시스템은 사용자 개개인의 선호도와 행동을 바탕으로 아이템을 개인화하여 추천합니다. 본 논문은 RMSE, NDCG, precision 등의 기존 성능 평가 지표만으로는 불충분하다는 점을 지적하며, 세분화된 평가 메트릭스의 필요성을 강조합니다. 성별 편향을 평가하기 위해 추천된 항목의 범주와 순위를 고려하는 방안을 제안하며, 이 메트릭스를 손실 함수의 일환으로 활용하여 공정성을 극대화합니다.

- **Performance Highlights**: 실제 데이터세트에 대한 실험을 통해 제안된 메트릭스의 효과성을 입증했습니다. 일반적인 추천 성능의 저하 없이, 다양한 범주에서 추천의 공정성을 크게 향상시켰습니다. 성별 관련 추천의 불균형 문제를 해결하는 데 도움을 주며, 추천 모델에 대한 편향 분석에 있어 향상된 통찰력을 제공합니다.



### The GigaMIDI Dataset with Features for Expressive Music Performance Detection (https://arxiv.org/abs/2502.17726)
Comments:
          Published at Transactions of the International Society for Music Information Retrieval (TISMIR), 8(1), 1-19

- **What's New**: 이 논문은 GigaMIDI 데이터셋을 소개하며, 140만 개의 유니크 MIDI 파일과 5천3백만 개의 트랙을 포함하고 있습니다. MIDI 파일에서 비표현적(non-expressive) 성능과 표현적(expressive) 성능을 구별하는 것은 도전적인 작업이며, 이를 위해 새로운 휴리스틱(heuristics)을 도입했습니다. 특히, Distinctive Note Velocity Ratio (DNVR)와 Note Onset Median Metric Level (NOMML) 등의 방법을 사용하여 MIDI 트랙을 구분할 수 있습니다.

- **Technical Details**: 디지털 음악 표현 방식은 오디오와 기호(symbolic) 두 가지로 나뉘며, MIDI는 기호 데이터를 효율적으로 저장하는 형식으로 인식되고 있습니다. MIDI 파일은 악기 연주와 같은 다양한 실시간 연출을 포함한 멀티트랙 아키텍처를 사용하여 음악 정보를 전달합니다. 이 논문에서는 새로운 휴리스틱을 통해 MIDI 트랙의 표현성을 분석하고 평가하는 방법론을 제안합니다.

- **Performance Highlights**: 이 연구를 통해 GigaMIDI 데이터셋에서 31%를 차지하는 표현적 MIDI 트랙을 포함하는 가장 큰 MIDI 데이터셋이 구축되었습니다. 논문에서 제안한 휴리스틱들은 MIDI 트랙의 표현적 차이를 효과적으로 구분할 수 있다는 충분한 평가 결과를 보여주고 있습니다. 최종적으로, 연구 결과는 MIDI 기반 음악 생성과 분석의 발전에 기여할 것으로 기대됩니다.



### Data Voids and Warning Banners on Google Search (https://arxiv.org/abs/2502.17542)
- **What's New**: 이 연구에서는 Google 검색에서 사용되는 경고 배너의 사용을 분석하고, 이를 통해 데이터 공백(data voids)을 식별하기 위한 딥 러닝 모델을 훈련했습니다. 연구진은 소셜 미디어에서 공유된 140만 개의 고유 검색 쿼리를 수집하여 Google의 경고 배너가 언제, 왜 적용되는지를 조사했습니다. 이 과정에서 발견된 데이터는 Google 검색의 컨텐츠 조정(practices)에 대한 투명성이 필요함을 강조합니다.

- **Technical Details**: 연구의 방법론에는 딥 러닝 모델을 활용하여 Google의 경고 배너의 존재를 예측하고, 이를 통해 라벨이 없는 데이터 공백을 식별하는 작업이 포함되었습니다. 연구진은 2023년 10월, 2024년 3월, 2024년 9월에 걸쳐 세 번의 데이터 수집 주기를 실시했고, 이 과정에서 약 1%의 검색 쿼리가 경고 배너를 생성했습니다. 특히, 저품질 배너는 총 배너의 2.1%를 차지했으며, 이는 조사된 검색 쿼리의 0.021%에 해당합니다.

- **Performance Highlights**: 이 연구에서는 Google이 고품질 결과를 위한 경고 배너를 사용하는 빈도가 낮다는 것을 발견했습니다. 또한, 저품질 배너는 특정 키워드가 포함된 검색 쿼리에서 더 자주 나타났으며, 특정 쿼리(예: ‘site:infowars.com’)에서는 전혀 나타나지 않았습니다. 연구진은 기존의 경고 배너 분류를 넘어 0.44%에서 1.16%의 SERP를 저품질 데이터 공백으로 분류하며, 이는 Google의 기준보다 29배에서 58배 더 많은 수치입니다.



### Contrastive Learning Augmented Social Recommendations (https://arxiv.org/abs/2502.15695)
- **What's New**: 본 연구에서는 전통적인 행동 기반 추천 시스템의 한계를 극복하기 위해 소셜 그래프(social graph)를 활용하는 새로운 접근 방식을 제안합니다. 특히, 콜드 유저(cold user)와의 상호작용을 개선하기 위해 사용자의 사회적 관계 정보를 통합하고, 이를 통해 사용자 관심을 보다 효과적으로 모델링합니다. 또한, 그래프 데이터의 노이즈를 줄이기 위한 이중 뷰 방식의 디노이징 프레임워크를 도입하여 소셜 그래프의 유효성을 높이고자 합니다.

- **Technical Details**: 이 방법론은 저차원 특이값 분해(low-rank singular value decomposition, SVD)를 사용하여 사용자-아이템 상호작용 행렬을 분석하고 필터링하여 노이즈를 제거합니다. 그리고 대조 학습(contrastive learning)을 활용하여 원래의 소셜 그래프와 복원된 소셜 그래프 간의 일치를 극대화합니다. 더불어, 상호 증류(mutual distillation) 메커니즘을 적용하여 사회적 관심과 행동적 관심을 각각 분리하여 효과적인 통합을 이룹니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 콜드 유저에 대한 추천의 정확성을 현저히 향상시켜 플랫폼의 성장을 도모할 수 있음을 입증하였습니다. 특히, 소셜 관계와 행동 정보를 효과적으로 융합하여 기존의 추천 시스템에서 나타나는 한계를 극복하고, 사용자 경험을 개선하는 데 큰 기여를 할 것으로 기대됩니다.



New uploads on arXiv(cs.CV)

### K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs (https://arxiv.org/abs/2502.18461)
- **What's New**: 이번 논문에서는 K-LoRA라는 새로운 접근법을 제안합니다. K-LoRA는 LoRA를 결합하여 학습된 주제와 스타일을 효과적으로 융합하는 비학습 기반 방법입니다. 이 방법은 각 주의(attention) 레이어에서 최적의 융합을 위한 Top-K 요소를 비교하여 선택하는 메커니즘을 통해 주제와 스타일의 대표적 특징을 유지합니다. 실험 결과, K-LoRA는 기존의 훈련 기반 접근법들보다 우수한 성능을 보였습니다.

- **Technical Details**: K-LoRA는 기본적으로 LoRA의 내재적 속성을 활용하여 확산 모델(diffusion models)에 적용합니다. 이는 각 주의 레이어가 통과할 때 Top-K 선택 과정을 포함하여, 스타일과 콘텐츠의 기여도를 균형 있게 유지합니다. 이 과정에서 선택 메커니즘이 스타일과 주제의 특성을 효율적으로 포착하도록 설계되었습니다. K-LoRA는 별도의 추가 훈련 없이도 사용 가능하며, 기존 LoRA 가중치에 바로 적용할 수 있습니다.

- **Performance Highlights**: K-LoRA의 실험 결과는 다양한 이미지 스타일화 작업에서 우수한 성능을 입증합니다. 특히, K-LoRA는 정밀한 스타일과 주제를 통합하는 데 효과적이며 생성된 이미지의 질적인 측면과 양적인 측면 모두에서 우수한 결과를 나타냅니다. 또한, K-LoRA는 직관적인 사용자 경험을 제공하여 추가적인 훈련 없이도 사용이 용이하다는 장점이 있습니다.



### GHOST 2.0: generative high-fidelity one shot transfer of heads (https://arxiv.org/abs/2502.18417)
- **What's New**: 논문에서는 최근 연구 커뮤니티에서 주목받고 있는 얼굴 스와핑( face swapping)과 관련된 문제인 헤드 스와핑(head swapping)에 대해 다루고 있습니다. GHOST 2.0이라는 새로운 프레임워크를 제안하며, 이는 두 개의 문제 특화 모듈로 구성되어 있습니다. 기존의 얼굴 스와핑과의 차별점은 전체 머리 구조를 보존해야 하는 추가적인 도전 과제가 있다는 점입니다.

- **Technical Details**: GHOST 2.0은 두 개의 주요 모듈, 즉 Aligner와 Blender로 구성되어 있습니다. Aligner 모델은 머리 재연출(head reenactment)을 위한 것이며, 이는 여러 스케일에서의 정체성 정보를 보존하고 극단적인 포즈 변환에 강력한 성능을 보여줍니다. Blender 모듈은 피부 색상을 전이시키고 스왑된 머리와 배경 간의 갭을 메우는 작업을 수행합니다.

- **Performance Highlights**: 각 모듈은 해당 작업에 대한 기준선 모델을 초과하며, 헤드 스와핑에서 최첨단 결과를 달성하는 데 기여합니다. 특히, 헤어 스타일의 큰 차이를 다루는 복잡한 경우에도 뛰어난 품질을 제공합니다. 연구팀은 특정 헤드 스와핑 작업을 위해 새로운 세분화(segmentation) 모델을 훈련시켰으며, 이는 올바른 색상 전이를 위해 수염과 얼굴 털을 별도의 클래스로 분리하는 방식으로 주목받고 있습니다.



### MedKAN: An Advanced Kolmogorov-Arnold Network for Medical Image Classification (https://arxiv.org/abs/2502.18416)
- **What's New**: 이번 연구에서는 의학 이미지 분류를 위한 새로운 프레임워크인 MedKAN을 소개합니다. MedKAN은 Kolmogorov-Arnold Networks(KAN)와 그 합성곱(convolutional) 확장을 기반으로 하며, 복잡한 비선형(Pathological) 구조를 효과적으로 캡처하고 특징 표현 능력을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: MedKAN은 두 가지 핵심 모듈인 Local Information KAN(LIK)과 Global Information KAN(GIK)으로 구성됩니다. LIK 모듈은 상세한 공간 특징을 추출하는 데 집중하기 위해 Local Grouped Convolution KAN(LGCK) 블록과 Spatial Feed-Forward Network(SFFN) 블록을 사용합니다. GIK 모듈은 장거리 의존성과 전역 맥락 관계를 모델링하여, 멀리 있는 영역 간의 정보를 통합하는 데 효과적입니다.

- **Performance Highlights**: 아홉 개의 공공 의학 이미지 분류 데이터셋에 대한 실험 결과, MedKAN은 CNN 및 Transformer 기반 모델들에 비해 뛰어난 성능을 보여주었습니다. 이를 통해 MedKAN의 효과성과 범용성이 입증되었으며, 다양한 의학 이미징 작업에 적용할 수 있는 가능성이 확인되었습니다.



### OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preferenc (https://arxiv.org/abs/2502.18411)
- **What's New**: 이 연구는 OmniAlign-V라는 새로운 데이터셋을 소개하며, 이는 20만 개의 고품질 훈련 샘플로 구성되어 있어 MLLMs의 인간 선호도 정렬을 개선하는 데 중점을 둡니다. 또한, MM-AlignBench라는 새로운 벤치마크를 개발하여 MLLMs의 인간 가치 정렬을 평가할 수 있도록 하고, 이를 통해 MLLMs의 성능을 향상시키기 위한 중요한 기초 데이터를 제공합니다.

- **Technical Details**: OmniAlign-V 데이터셋은 여러 가지 복잡한 질문과 다양한 응답 형식을 포함하는 데이터로 구성되어 있으며, 자연 이미지와 인포그래픽을 포함합니다. 연구는 Supervised Fine-Tuning (SFT) 및 Direct Preference Optimization (DPO) 기법을 활용하여 MLLMs의 성능을 극대화하는 방법을 탐구했으며, OmniAlign-V를 사용한 결과 MLLMs의 인간 선호도 정렬이 유의미하게 향상되었음을 결론지었습니다.

- **Performance Highlights**: OmniAlign-V로 세부 조정된 MLLMs는 기존 VQA 벤치마크에서 비교 가능하거나 우수한 성능을 보였고, MLLMs의 인간 선호도와의 정렬 능력이 향상되었습니다. DPO 단계에서 OmniAlign-V를 적용한 것도 높은 성능을 보여주었으며, 기존의 최첨단 모델보다 뛰어난 결과를 달성했습니다. 이 연구는 MLLMs의 인간 정렬 가능성을 높이는 데 기여할 수 있는 중요한 기틀을 제공합니다.



### EgoSim: An Egocentric Multi-view Simulator and Real Dataset for Body-worn Cameras during Motion and Activity (https://arxiv.org/abs/2502.18373)
- **What's New**: 최근 컴퓨터 비전 분야의 egocentric(자기중심적) 작업은 주로 헤드 마운트 카메라에 집중되어 왔으나, 저자는 카메라 기술의 소형화가 다양한 신체 착용 장치에 카메라 통합을 촉진할 것이라고 주장합니다. 이러한 변화는 사람의 움직임 추적, 바디 포즈 추정, 액션 인식 같은 기존 작업에 새로운 관점을 제공하고 특히 하체 부분을 다루는 데 중요한 기여를 할 것입니다. 본 논문에서는 여러 각도가 포함된 신체 착용 카메라의 현장 기반 성능을 구현하는 EgoSim 시뮬레이터와 MultiEgoView 데이터셋을 소개합니다.

- **Technical Details**: EgoSim은 여러 신체 착용 카메라로부터 현실적인 egocentric 렌더링을 생성하는 시뮬레이터로, 실제 모션 캡처 데이터를 사용하여 동작 아티팩트를 렌더링합니다. MultiEgoView 데이터셋은 6개의 신체 착용 카메라에서 촬영된 119시간의 egocentric 영상과 진실 위치를 포함하여 다양한 활동을 구성합니다. 또한, 13명의 참가자가 6개의 GoPro 카메라를 착용해 기록한 5시간의 실제 데이터도 포함되어 있으며, Xsens 모션 캡처 수트를 통해 전체 바디 3D 포즈 참조가 제공됩니다.

- **Performance Highlights**: EgoSim을 활용해 훈련한 비디오 기반 3D 포즈 추정 네트워크의 효과를 입증하였으며, 도메인 갭 분석을 통해 데이터셋과 시뮬레이터가 실제 데이터로의 추론에 얼마나 많은 도움이 되는지를 보여주었습니다. 이는 egocentric 인식 작업을 위한 개방형 연구의 발전에 기여할 것으로 예상됩니다. 전반적으로 EgoSim은 신체착용 카메라를 활용한 다양한 동작 인식을 위한 새로운 연구의 가능성을 열어줍니다.



### Near-Shore Mapping for Detection and Tracking of Vessels (https://arxiv.org/abs/2502.18368)
Comments:
          Submitted to FUSION 2025

- **What's New**: 이 논문에서는 자율 수면선박(ASV)이 도킹하기 위해 필수적인 초기 감지 기술을 제안합니다. 기존의 방법들은 주로 고정된 물체에 대한 추적을 수행했으나, 본 연구는 이동하는 물체와 정적인 물체를 구분하여 정확한 3D 맵을 구축하는 데 중점을 둡니다. 이 방식을 통해 기존 방법보다 더 정밀한 근해 물체 추적이 가능해졌습니다.

- **Technical Details**: 제안된 방법은 LiDAR 데이터와 이미지 데이터를 통합하여 환경을 정확하게 매핑합니다. LiDAR는 높은 정밀도로 거리 데이터를 제공하며, 딥러닝 기반의 인스턴스 세그멘테이션 기술을 이용하여 이미지에서 선박을 감지합니다. 또한, 이동할 가능성이 있는 물체를 필터링하여 정확한 추적을 가능하게 합니다.

- **Performance Highlights**: 실제 환경에서 수집된 데이터셋을 활용한 실험 결과, 제안된 방법이 정확한 맵을 기반으로 근해의 표적 추적 성능을 크게 향상시켰음을 보여줍니다. 이 논문에서 사용한 데이터셋은 카약과 선박이 자율 페리 프로토타입과 충돌 경로에 가까운 위치에서 움직이는 여러 시퀀스를 포함하고 있습니다.



### ART: Anonymous Region Transformer for Variable Multi-Layer Transparent Image Generation (https://arxiv.org/abs/2502.18364)
Comments:
          Project page: this https URL

- **What's New**: 새로운 연구에서 제안된 Anonymous Region Transformer (ART)는 사용자로 하여금 전역 텍스트 프롬프트와 익명 영역 레이아웃을 기반으로 다양한 다층 투명 이미지를 직접 생성할 수 있게 합니다. 이 시스템은 기존의 의미적 레이아웃 방식과는 달리, 어떤 시각적 토큰이 어떤 텍스트 토큰에 맞아야 할지를 생성 모델이 자율적으로 결정합니다. 이러한 접근은 텍스트와 이미지를 동일한 레벨에서 연계하여 더 나은 제어를 가능하게 합니다.

- **Technical Details**: ART는 익명 영역 레이아웃을 통해 다층 이미지를 생성하는데 필요한 비주얼 토큰을 효율적으로 선택하고, 각 레이어의 항목을 개별적으로 편집할 수 있는 기능을 제공합니다. 특히 레이어별 지역 크롭 메커니즘은 각 익명 지역에 해당하는 비주얼 토큰만을 선택하여 주목 계산 비용을 줄이고, 50개 이상의 독특한 레이어가 있는 이미지를 효율적으로 생성할 수 있게 합니다. 이 방식은 전체 주목 접근에 비해 12배 이상 빠르며 레이어 간의 충돌을 줄입니다.

- **Performance Highlights**: 실험 결과, ART는 텍스트 프롬프트와 시각적 토큰 간의 상호작용을 통해 각 레이어의 의미를 효과적으로 파악할 수 있으며, 레이어 간 일관성을 더욱 향상시킵니다. 이전의 방법들에 비해 더 높은 품질과 많은 수의 레이어를 가진 다층 투명 이미지를 생성할 수 있음을 입증했습니다. 이로 인해 그래픽 디자인 생성 등 다양한 분야에서의 활용 가능성이 열리게 됩니다.



### Self-Supervised Data Generation for Precision Agriculture: Blending Simulated Environments with Real Imagery (https://arxiv.org/abs/2502.18320)
Comments:
          Presented at 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)

- **What's New**: 정밀 농업에서 레이블이 붙은 데이터 부족과 변동성이 큰 데이터 차이에 대한 문제를 해결하기 위해, 저자들은 새로운 합성 데이터 생성 시스템을 제안했습니다. 이 시스템은 Unity 엔진을 기반으로 한 포도원 시뮬레이터를 활용하고, 기하학적 일관성을 고려한 컷앤페이스트 기법을 사용하여 리얼한 이미지를 생성합니다. 이는 다양한 관점과 조명 조건에서 고유한 데이터 샘플을 생성하여 탐지 알고리즘 훈련에 기여합니다. 저자들은 이 방법이 포도 재배에서 획기적인 성능 향상을 보여줌을 입증합니다.

- **Technical Details**: 제안된 시스템은 EU 프로젝트 CANOPIES의 일부로 개발된 3D 포도원 시뮬레이터를 사용합니다. 이 시뮬레이터는 전통적인 포도나무 재배 시스템을 재현하며, 깊이 카메라를 장착한 로봇이 시뮬레이터에서 작업하도록 설계되었습니다. 실제 이미지에서 세분화 마스크를 추출하고 이를 시뮬레이션 환경에 융합하여 더 다양하고 강력한 데이터 분포를 제공합니다. YOLOv5와 Segment Anything Model(SAM)을 활용하여 자동으로 세분화 마스크를 추출하고 이를 데이터 생성 파이프라인에 통합합니다.

- **Performance Highlights**: 이 연구에서는 제안된 방법을 표준 데이터와 증강 데이터를 이용하여 교육한 최첨단 탐지기와 비교 실험을 수행합니다. 실험 결과, 합성 데이터가 탐지 모델의 일반화 성능에 크게 기여하고, 수확 및 모니터링과 같은 농업 작업에서의 적용 가능성을 높임을 보여줍니다. 또한, 기술의 간소화로 인해 농민들도 쉽게 데이터를 생성하고 활용할 수 있도록 설계된 점이 강조됩니다.



### LDGen: Enhancing Text-to-Image Synthesis via Large Language Model-Driven Language Representation (https://arxiv.org/abs/2502.18302)
- **What's New**: 이번 연구에서는 LDGen이라는 새로운 방법을 소개합니다. LDGen은 기존의 텍스트-이미지 확산 모델에 대규모 언어 모델(LLM)을 통합할 수 있게 해주면서도 계산량을 최소화합니다. 기존의 CLIP 및 T5와 같은 텍스트 인코더들이 다국어 처리에 한계를 보였던 문제를 해결하기 위해, LLM의 고급 기능을 활용하였습니다.

- **Technical Details**: LDGen은 계층적 캡션 최적화와 인간 지침 기술을 활용하는 언어 표현 전략을 사용하여 텍스트 정보의 정확한 의미를 도출합니다. 또한 가벼운 어댑터(lightweight adapter) 및 교차 모드 리파이너(cross-modal refiner)를 도입하여 LLM과 이미지 기능 간의 효율적인 특징 정렬 및 상호 작용을 지원합니다. 이러한 접근 방식을 통해 트레이닝 시간을 줄이고 제로샷(Zero-Shot) 다국어 이미지 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 LDGen이 기본 모델 outperform하여 프롬프트 준수(prompt adherence) 및 이미지 미적 품질에서 우수한 성능을 보임을 보여줍니다. 특히, 다양한 언어를 매끄럽게 지원하면서도 LLM의 본질적인 기능을 활용하여 다국어 이미지 생성을 증가시켰습니다. 이 연구는 LLM과 기존 확산 모델 간의 효과적인 통합 방안을 제시하면서, 언어와 이미지 간의 의미 일치를 강화하였습니다.



### Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models (https://arxiv.org/abs/2502.18290)
- **What's New**: 이 논문에서는 Self-Supervised Learning (SSL) 비전 인코더의 보안 취약점을 다루고 있습니다. 특히, 이러한 인코더가 강력한 Backdoor 공격에 취약하다는 새로운 위협을 제기하며, 이를 통해 시각적 환각(visual hallucinations)을 유도할 수 있는 방법을 보여줍니다. 연구팀은 BadVision이라는 방법을 제안하여 LVLMs에 대한 공격을 효과적으로 수행할 수 있는 접근법을 제시했습니다.

- **Technical Details**: BadVision은 다양한 타입의 SSL 비전 인코더에 적용 가능한 통합 공격 프레임워크입니다. 이 방법은 유니버설 트리거 최적화(universal trigger optimization) 및 Backdoor 학습(backdoor learning) 기술을 활용하여 공격을 실행하며, 99% 이상의 성공률을 기록했습니다. 또한, 트리거-포커싱(trigger-focusing) 메커니즘을 통해 더욱 정교한 공격을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, BadVision은 8개의 벤치마크에서 LVLMs의 시각적 이해 오류를 77.6% 증가시켰습니다. 이 연구는 LVLM 시스템에서 스텔스(backdoor) 공격이 어떻게 수행될 수 있는지를 보여주며, 기존의 SoTA 방법들로는 이러한 공격을 효과적으로 탐지할 수 없음을 입증했습니다. 나아가, 이 공격의 높은 효율성과 실세계에서의 적용 가능성을 강조합니다.



### Multi-label out-of-distribution detection via evidential learning (https://arxiv.org/abs/2502.18224)
Comments:
          Accepted at Uncertainty Quantification for Computer Vision workshop (ECCVW 2024)

- **What's New**: 이번 연구에서는 multi-label 분류 설정에서 out-of-distribution (OOD) 데이터 탐지를 위해 Evidential Deep Learning (EDL) 접근 방식을 제안합니다. 기존 연구들은 일반적으로 single-label 데이터에 초점을 맞췄는데, 본 연구는 다양한레이블을 가진 데이터에서의 OOD 탐지의 필요성을 강조합니다. 제안된 방법은 두 가지 새로운 불확실성 기반 점수를 도입하여 OOD 데이터를 식별하는데 있어 유용성을 증대시킵니다.

- **Technical Details**: 제안된 CNN 아키텍처는 Beta Evidential Neural Network를 사용하여 샘플의 가능성과 예측 불확실성을 동시에 계산합니다. 이를 바탕으로 OOD 탐지를 위한 두 가지 새로운 점수를 제안하는데, 첫 번째는 최대 증거를 기반으로 한 OOD-score Max, 두 번째는 모든 출력의 증거를 고려한 OOD-score Sum입니다. 이 방법은 PASCAL-VOC, MS-COCO, NUS-WIDE 데이터셋을 통해 검증되었습니다.

- **Performance Highlights**: 제안된 방법은 세 가지 주요 multi-label 데이터셋에서 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. OOD 탐지에서의 효과적인 불확실성 추정이 실질적인 응용 프로그램에서 AI 시스템의 신뢰성을 높이는 데 중요한 역할을 할 수 있음을 입증합니다. 본 연구는 multi-label 분류 환경에서 EDL을 활용한 최초의 접근 방법으로, 향후 연구에 중요한 기초 자료로 작용할 것입니다.



### UASTrack: A Unified Adaptive Selection Framework with Modality-Customization in Single Object Tracking (https://arxiv.org/abs/2502.18220)
- **What's New**: 이번 연구에서는 모드 적응 인식을 통한 단일 객체 추적을 위한 통합 적응 선택 프레임워크인 UASTrack을 제안합니다. UASTrack은 Discriminative Auto-Selector (DAS)를 활용해 다양한 RGB-X 이미지 쌍에 따라 입력 모드 유형을 동적으로 식별할 수 있습니다. 또한, Task-Customized Optimization Adapter (TCOA)는 각 모드의 특성을 반영하여 최적화된 처리를 수행합니다. 이를 통해 다양한 다중 모드 추적 작업에서 균일하고 효율적인 성능을 달성할 수 있습니다.

- **Technical Details**: UASTrack은 RGB-X(여기서 X는 depth, event 또는 thermal 모드를 의미합니다) 이미지 쌍을 기반으로 하여, 다중 모드 작업 간 효율적인 통합 처리를 가능하게 합니다. DAS 모듈은 모드별로 적합한 네트워크 구조를 선택하도록 지침을 제공하며, 클래스 제약 손실(Classification Constraint Loss)을 추가하여 학습 능력을 향상시킵니다. TCOA는 각 모드별 특성을 고려하여 스타일을 최적화하여 더 나은 예측을 가능하게 합니다. 아울러, Transformer Encoder 블록 내에 양방향 어댑터를 도입해 RGB와 X 기능 간의 효과적인 상호작용을 지원합니다.

- **Performance Highlights**: UASTrack은 LasHeR, GTOT, RGBT234, VisEvent, DepthTrack 등에서 경쟁력 있는 성능을 입증했습니다. 특히, 모드별 특징에 최적화된 구조 덕분에, 필수적인 추가 파라미터는 1.87M, FLOPs는 1.95G에 불과하며, 기존 RGB-X 기준 대비 8.5%의 성공률 향상을 이뤘습니다. 다양한 벤치마크에서의 비교를 통해, 우리의 접근 방식이 최첨단 트래커들보다 효율적으로 작동함을 확인했습니다.



### Synthesizing Consistent Novel Views via 3D Epipolar Attention without Re-Training (https://arxiv.org/abs/2502.18219)
Comments:
          3DV 2025

- **What's New**: 이 연구에서는 단일 이미지로부터 새로운 뷰를 합성할 때, 시각적 일관성을 향상시키기 위해 epipolar geometry를 활용한 새로운 방법을 제안합니다. 기존의 방법들이 입력 이미지를 조건으로 사용해도 일관성을 보장하지 못하는 문제를 해결하기 위해, 입력 뷰와 참조 뷰 간의 중첩 정보를 원활하게 활용합니다. 이를 통해 모델은 별도의 학습이나 미세 조정 없이도 성능을 강화하여 3D 재구성과 같은 다운스트림 어플리케이션의 성능 또한 향상됩니다.

- **Technical Details**: 본 연구의 핵심 기술은 epipolar attention 모듈로, 이는 참조 이미지 내의 중첩 정보를 위치 지정 및 검색하기 위해 사용됩니다. 각 타겟 뷰의 포인트에 대해, 해당 포인트는 참조 이미지의 epipolar line에 제약을 두어 그에 따라 상응하는 지점을 찾아내고, 이를 통해 생성 과정에서 입력 이미지의 피처를 회수하게 됩니다. 이러한 방법은 메모리 요구사항도 감소시켜 GPU 사용 시 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 생성된 다중 뷰 이미지의 일관성을 크게 개선하는 효과를 보였으며, 학습이나 미세 조정이 필요하지 않았습니다. 또한, 이 합성된 이미지들은 3D 재구성 과제에도 적용하여 추가적인 성능 향상을 보여주었습니다. 따라서 이 방법은 다양한 실제 어플리케이션에서도 실용적으로 활용될 수 있을 것으로 예상됩니다.



### Learning Structure-Supporting Dependencies via Keypoint Interactive Transformer for General Mammal Pose Estimation (https://arxiv.org/abs/2502.18214)
Comments:
          accepted by IJCV 2025

- **What's New**: 이 논문에서는 일반 포유류 포즈 추정에 대한 새로운 모델인 Keypoint Interactive Transformer (KIT)를 제안합니다. KIT는 인스턴스 수준의 구조적 관계를 학습하여 포즈 추정에서의 어려움을 극복하는 것을 목표로 합니다. 특히, KITPose는 두 가지 연결된 컴포넌트로 구성되어 있으며, 키포인트 특징 추출과 이미지 맥락에 맞춘 바디 파트 프로프트 생성을 포함합니다.

- **Technical Details**: KIT 모듈은 고해상도 백본을 기반으로 하여 특징 맵을 획득한 후, 결정 단계에서 스택드 어텐션 작업을 수행하여 계층적 상호작용을 달성합니다. 또한, 모델의 일반화 성능 향상을 위해 cutmix 증강 기법을 적용합니다. 이를 통해 포즈가 많이 변동할 때 경험적으로 키포인트 간의 관계를 평가하고, 구조 지원 상호작용을 통해 특정 위치에 대한 결정을 집중시킵니다.

- **Performance Highlights**: KIT 모델은 제한된 양의 학습 데이터를 제공받아도 뛰어난 성능을 발휘할 수 있도록 설계되었습니다. 또한, Generalised Heatmap Regression Loss (GHRL)를 도입하여 중간 레이어가 지배적인 키포인트 특징을 생성하지 않도록 하고, 구조적 구별력을 강제합니다. 이는 다양한 포유류 종의 포즈 추정에서의 정확도를 크게 향상시킬 것으로 기대됩니다.



### Multi-Perspective Data Augmentation for Few-shot Object Detection (https://arxiv.org/abs/2502.18195)
Comments:
          ICLR 2025

- **What's New**: 이 연구에서는 최근의 몇 샷 물체 탐지(Few-Shot Object Detection) 방법에서의 주요 문제인 데이터 다양성 부족을 해결하기 위한 새로운 프레임워크인 Multi-Perspective Data Augmentation (MPAD)을 제안합니다. 기존의 기법들은 주로 전통적인 기하학적 변환이나 기존의 데이터 보강 기법에 의존하고 있어, 다양한 샘플을 생성하는 데 한계를 보였습니다. 특히, 본 연구는 이러한 문제를 극복하기 위해 In-Context Object Synthesis (ICOS), Harmonic Prompt Aggregation Scheduler (HPAS), Background Proposal method (BAP)와 같은 다양한 방법론을 도입했습니다.

- **Technical Details**: MPAD는 두 가지 주요 요소인 전경-전경 관계와 전경-배경 관계를 고려하여 데이터를 증강하는 방식을 채택합니다. ICOS는 대형 언어 모델(LLMs)로부터의 일반 지식을 활용하여 새로운 클래스의 속성을 다양화하죠. 또한, HPAS는 확산 모델의 생성 과정에서 프롬프트 임베딩을 혼합해 물체의 고급 특징과 저급 특징을 생성합니다. 마지막으로, BAP는 전경 물체와의 유사성을 고려하여 일반적이고 어려운 배경을 샘플링하는 방법을 제안합니다.

- **Performance Highlights**: 제안된 MPAD 프레임워크는 여러 FSOD 기준 평가에서 기존의 전통적인 방법들보다 훨씬 뛰어난 성능을 보였습니다. 특히 PASCAL VOC 데이터셋에서는 baseline에 비해 평균 17.5% 향상된 nAP50을 기록하며, 최첨단(pre-state-of-the-art) 성과를 달성했습니다. 이 결과는 기존의 물체 탐지 모델들이 직면한 극복하기 어려운 문제를 해결할 수 있는 강력한 솔루션임을 입증합니다.



### CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification (https://arxiv.org/abs/2502.18176)
Comments:
          accepted by ICLR 2025

- **What's New**: 이 논문은 적대적으로 강건한 제로샷 이미지 분류기를 구축하는 것을 목표로 합니다. 특히, CLIP 모델을 기반으로 하여, 적대적 공격에 대한 방어를 위해 'Purification' 방법을 활용합니다. CLIPure라는 새로운 접근법은 기존의 생성 모델에 의존하지 않으면서도 방어 효율성을 크게 향상시킨 방식으로, 제로샷 분류의 안정성을 크게 개선합니다.

- **Technical Details**: Purification 리스크는 적대적 샘플 정제 과정의 KL divergence를 통해 정의됩니다. 연구자들은 Bidirectional Stochastic Differential Equations (SDEs)를 활용하여 공격 프로세스와 정제 프로세스를 각각 모델링하고, 이 과정에서 KL divergence와 분포의 매끄러움과 같은 요소들이 purification 성능에 영향을 미친다고 강조합니다. CLIPure는 CLIP의 다중 모달 잠재 공간에서 수행되며, 두 가지 변종인 CLIPure-Diff와 CLIPure-Cos를 제안합니다.

- **Performance Highlights**: CLIPure의 실험 결과는 CIFAR-10에서 71.7%에서 91.1%로, ImageNet에서는 59.6%에서 72.6%로 조정되었으며, 13개 데이터셋에서는 평균적으로 108%의 상대적 개선을 보여주었습니다. 시장의 최고 성과(SOTA)를 상당히 초과하는 결과로 제로샷 분류의 강건성을 입증하며, CLIPure는 적대적 공격을 견디는 데 있어 중요한 진전을 이루었습니다.



### Monitoring snow avalanches from SAR data with deep learning (https://arxiv.org/abs/2502.18157)
- **What's New**: 이 논문에서는 눈사태 감지를 위한 최신 딥러닝 기술의 적용을 다루고 있습니다. 기존의 SAR(Synthetic Aperture Radar) 이미지 분석 방법과 비교하여, 최근의 연구들은 픽셀 수준에서의 세분화(segmentation)를 통해 훨씬 더 높은 정확도와 공간 해상도를 제공합니다. 특히, Sentinel-1 SAR 데이터를 활용한 사례 연구를 통해 딥러닝 모델이 전통적인 방법보다 더 뛰어난 성능을 발휘함을 보여주었습니다.

- **Technical Details**: 눈사태를 감지하기 위해, 딥러닝 기반의 CNN(Convolutional Neural Networks) 모델이 SAR 데이터에서 특징을 자동으로 추출할 수 있도록 설계되었습니다. 이는 고해상도 이미지를 처리하고 복잡한 패턴을 학습하는 데 유리하여, 관련 데이터를 효율적으로 분석할 수 있게 합니다. 연구에서는 4,500개 이상의 주석이 달린 SAR 이미지를 활용하여 다양한 최신 세분화 아키텍처들을 테스트하였고, 이를 통해 대규모 눈사태 감지에서의 효과적인 활용 가능성을 확인했습니다.

- **Performance Highlights**: 신뢰성 있는 눈사태 감지 및 예측을 위해 SAR 데이터를 통해 파생된 딥러닝 모델이 사용되었습니다. 이 모델은 노르웨이 전역에서 대규모 눈사태 감지에 적용되어 여러 겨울 시즌 동안 중요한 공간적 및 시간적 패턴을 드러냈습니다. 기존의 전통적인 방법들에 비해 깊이 있는 분석을 제공하는 이 모델은 향후 눈사태 위험 예측 및 관리에 중요한 역할을 할 것으로 기대됩니다.



### Joint Reconstruction of Spatially-Coherent and Realistic Clothed Humans and Objects from a Single Imag (https://arxiv.org/abs/2502.18150)
- **What's New**: 이 논문에서 제안된 SCHOR 프레임워크는 단일 이미지에서 실제적이고 비모수(clothed humans and objects)를 동시에 재구성하는 최초의 방법입니다. 기존의 매개변수(parametric) 기반 접근법이 인간과 객체의 세부 사항을 효과적으로 포착하지 못했던 반면, SCHOR는 새로운 attention-based neural implicit model을 통해 세밀한 디테일을 포함한 재구성을 가능하게 합니다. 이 방법은 인간-객체의 occlusions을 해결하고, 이미지 픽셀 정렬을 활용하여 3D 공간 인식을 가능하게 합니다.

- **Technical Details**: Schor 시스템은 한 장의 RGB 이미지에서 입력 받아, 인간과 객체를 분리하여 세그멘테이션(segmentation)을 수행하며, SMPL-H 모델을 사용하여 자세(pose) 매개변수를 추정합니다. occluded 영역은 generative diffusion model을 통해 복원되어 재구성합니다. 이 논문에서의 핵심 기술 요소는 self-attention 메커니즘을 포함한 neural implicit model로, 이를 통해 전체 맥락을 고려하고 3D 공간 일관성을 유지합니다.

- **Performance Highlights**: 평가 결과, synthetically 생성된 synHOR 데이터셋과 실제 BEHAVE 데이터셋에서 SCHOR의 성능이 최첨단 기술들을 초월하는 뛰어난 재구성 품질을 보였습니다. 이 연구는 단일 이미지로부터 비모수 인체와 객체 형태의 사실적인 재구성을 실현함으로써 향후 다양한 응용 분야에서의 발전 가능성을 제시합니다.



### LightFC-X: Lightweight Convolutional Tracker for RGB-X Tracking (https://arxiv.org/abs/2502.18143)
- **What's New**: 이 논문에서는 리소스 제약이 있는 장치에서 효과적으로 사용할 수 있는 경량 멀티모달 추적기 LightFC-X를 제안합니다. LightFC-X는 컨볼루션 기반의 RGB-X 추적기로, 효율적인 크로스 모달(feature) 모델링과 스페이셜-템포럴(spatiotemporal) 외형(feature) 조정을 통해 경량화를 이룹니다. 이 모델은 새로운 효율적인 크로스-어텐션 모듈과 스페이티템포럴 템플릿 집계 모듈을 포함하고 있으며, 실시간 성능을 유지합니다.

- **Technical Details**: 제안된 ECAM(효율적인 크로스 어텐션 모듈)은 단 0.08M 파라미터로 크로스 모달 상호 작용을 효율적으로 수행합니다. STAM(스페이티템포럴 템플릿 집계 모듈)은 모듈 미세 조정을 통해 시간 정보를 극대화하며 모델의 성능을 증가시킵니다. LightFC-X는 총 3개의 변형으로 구성되며, 데이터 세트 및 훈련 전략을 통해 스페이셜한(feature)의 특성을 명확히 한 후, STAM을 통해 시간-공간적 표현을 모델링합니다.

- **Performance Highlights**: LightFC-X는 LasHeR 벤치마크에서 기존 방법들보다 4.3%에서 5.7%까지 성능이 향상되었고, 파라미터는 2.6배, 속도는 2.7배 줄였습니다. CPU에서 실시간으로 22fps에 이르는 성능을 자랑합니다. LightFC-X는 여러 벤치마크에서 상태-of-the-art 성과를 보여주며, 멀티모달 추적기 간의 파라미터, 성능 및 속도 사이의 최적의 균형을 이룹니다.



### Personalized Federated Learning for Egocentric Video Gaze Estimation with Comprehensive Parameter Frezzing (https://arxiv.org/abs/2502.18123)
- **What's New**: 본 논문에서는 개인화된 연합학습(Personalized Federated Learning, PFL) 프레임워크를 통해 종합적인 파라미터 동결 전략(Comprehensive Parameters Freezing, FedCPF)을 제안합니다. 기존 방법들과 달리, FedCPF는 여러 훈련 반복에서 파라미터 변화율을 고려하여 개인화된 모델의 성능을 향상시키는 데 중점을 둡니다. 이를 통해 개인 사용자 데이터의 미세한 차이를 보다 효과적으로 포착할 수 있습니다.

- **Technical Details**: 제안된 Gaze Estimation 방법은 Global-Local Correlation (GLC) 모듈을 활용하여 에고센트릭 비디오에서 사용자의 시선을 예측합니다. GLC 모듈은 비디오 프레임의 지역 간 상관관계를 캡처하기 위해 자기 주의 메커니즘을 사용하여 비디오 표현 학습을 향상시킵니다. 개인화된 파라미터는 각 클라이언트에서 훈련 동안 동결되며, 나머지 글로벌 파라미터는 전역 모델에 집계됩니다.

- **Performance Highlights**: 실험 결과, FedCPF는 EGTEA Gaze+ 및 Ego4D 데이터셋에서 이전의 연합 학습 방법들보다 우수한 성능을 보여주었습니다. 특히 Recall, Precision 및 F1-score와 같은 지표에서 높은 결과를 기록하며, 모델 개인화의 효과를 입증하였습니다. 이러한 결과는 개인화가 필요한 작업에서 FedCPF의 유망한 접근임을 나타냅니다.



### Bayesian Optimization for Controlled Image Editing via LLMs (https://arxiv.org/abs/2502.18116)
Comments:
          8 figures

- **What's New**: BayesGenie는 대형 언어 모델(LLMs)과 베이지안 최적화(Bayesian Optimization)를 통합하여 사용자가 이미지 편집을 쉽고 정밀하게 수행할 수 있도록 돕는 최신 접근법을 제안합니다. 기존의 수작업 영역 표시 없이 자연어로 이미지를 수정할 수 있는 기능을 제공하며, 원본 이미지의 의미적 일관성을 유지합니다. 이는 모델 미세 조정 또는 대규모 사전 훈련 없이도 다양한 LLM에 대해 뛰어난 적응성을 보여주는 모델 불가지론적(design) 설계를 특징으로 합니다.

- **Technical Details**: BayesGenie는 LLM의 의미 이해 능력을 활용하여 사용자의 요구에 따라 세부적인 프롬프트를 생성한 후 이를 스테이블 디퓨전(Stable Diffusion) 모델에 전달하여 정확한 이미지 수정을 지원합니다. 베이지안 최적화를 통해 매개변수 공간을 체계적으로 탐색하여 최적의 편집 품질을 달성합니다. 이 과정에서 ‘text_cfg_scale’와 ‘image_cfg_scale’와 같은 주요 매개변수를 조정하면서 결과물과 사용자 요구 사항 간의 정렬을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 BayesGenie는 편집 정확도 및 의미 보존 측면에서 기존 방법들을 크게 능가하는 성능을 입증하였습니다. Claude3와 GPT-4와 같은 다양한 LLM을 활용하여 프레임워크의 효과성을 검증하였으며, 이미지 수정의 직관성과 정확성을 향상시켰음을 보여줍니다. 실험 결과는 BayesGenie가 즉시 배치 가능하며 여러 시나리오에서 높은 품질의 결과를 제공할 수 있음을 시사합니다.



### PromptMID: Modal Invariant Descriptors Based on Diffusion and Vision Foundation Models for Optical-SAR Image Matching (https://arxiv.org/abs/2502.18104)
Comments:
          15 pages, 8 figures

- **What's New**: PromptMID는 optical 및 SAR 이미지 매칭을 위한 새로운 접근 방식으로, 토지 이용 분류(land use classification)를 기반으로 한 텍스트 프롬프트(prompt)를 사용하여 모달리티 불변(descriptor) 서술자를 구축합니다. 이 방법은 고급 생상 모델과 비주얼 파운데이션 모델(VFM)을 활용하여 다중 스케일의 모달리티 불변 특징을 추출하고, 여러 게층(feature aggregation) 모듈을 디자인하여 서로 다른 해상도의 특징들을 효과적으로 융합합니다. 이 연구의 주요 목표는 기존의 방법의 제한을 넘어서, 시각 도메인의 불일치함을 해결하고 일반화 능력을 향상시키는 것입니다.

- **Technical Details**: PromptMID는 고차원 특징 추출을 위해 사전 학습된 확산 모델과 VFM을 이용하며, SAR 이미지와 optical 이미지를 텍스트와 매핑하는 과정에서 중간 특징을 활용하여 다중 스케일의 잠재 확산 특징을 개발합니다. 양식 조합의 특징으로는 다중 스케일 인식 집계 모듈(MSAA)을 사용하여 다양한 해상도(feature)를 가진 정보를 효과적으로 융합하고, 채널과 공간 차원에서 특징을 정제하는 컨볼루션 블록 주의 모듈(CBAM)을 적용하여 관련 없는 노이즈를 억제합니다.

- **Performance Highlights**: 실험 결과, PromptMID는 다양한 지역의 optical 및 SAR 영상 데이터셋에서 최첨단 매칭 방법들을 능가하는 성능을 보였으며, 보이는 도메인과 보이지 않는 도메인 모두 뛰어난 성능을 보였습니다. 특히, 이 방법은 해당 도메인 간의 일반화 능력에서 강력한 결과를 입증하였습니다. 또한, 연구진은 다양한 이미지 크기와 촬영 센서, 지리적 영역을 포함한 네 개의 데이터셋에서 실험을 수행하였으며, 모든 실험에서 뛰어난 일반화 능력을 확인했습니다.



### Detecting Offensive Memes with Social Biases in Singapore Context Using Multimodal Large Language Models (https://arxiv.org/abs/2502.18101)
- **What's New**: 이 연구는 싱가포르의 현대적이고 다중 모달(multimodal) 의사소통 수단인 밈(meme)을 효과적으로 분류하기 위한 VLM(vision-language model)의 미세 조정(fine-tuning) 데이터셋을 소개합니다. 기존의 전통적인 온라인 콘텐츠 조정 시스템이 자원 부족 언어와 문화적 맥락의 필요성을 갖는 고도의 정보가 밀집된 매체인 밈을 분류하는 데 어려움을 겪고 있는 가운데, 112K의 밈 샘플로 구성된 대규모 데이터셋을 특수하게 제작했습니다. 이 데이터셋은 GPT-4V에 의해 레이블이 지정되어 있으며, 온라인 콘텐츠 조정에서 인공지능의 효과성을 극대화하고자 합니다.

- **Technical Details**: 연구에서는 두 가지 대표적인 비전-언어 모델(VLM), LLaVA-NeXT Mistral 7B와 Qwen2-VL 7B를 교육하고 평가했습니다. 또한 OCR(Optical Character Recognition)과 언어 번역(translation) 구성 요소의 필요성을 탐구하였습니다. 연구에서 제안하는 파이프라인은 두 가지 VLM을 사용하여 최첨단 성능을 보여주며, 인증된 데이터셋, 코드, 모델 가중치를 오픈 소스 방식으로 공개할 예정입니다.

- **Performance Highlights**: 제안된 솔루션은 공인된 테스트 세트에서 80.62%의 정확도와 0.8192 AUROC(Areas Under the Receiver Operating Characteristic Curve)를 달성하였고, 이는 온라인 콘텐츠 조정에 있어 인간의 도움을 크게 돕는 도구가 되리라 기대됩니다. 연구 결과는 싱가포르의 온라인 안전 시스템의 발전을 위해 제공되는 귀중한 로컬리제이션 리소스가 될 것입니다.



### FwNet-ECA: Facilitating Window Attention with Global Receptive Fields through Fourier Filtering Operations (https://arxiv.org/abs/2502.18094)
- **What's New**: 이 논문에서는 FwNet-ECA라는 새로운 기법을 소개하고 있습니다. 이 방법은 푸리에 변환(Fourier transforms)과 학습 가능(weight matrices) 가중 행렬을 조합하여 이미지의 스펙트럼 특징을 향상시키는 데 중점을 둡니다. 이러한 전략은 inter-window connectivity를 촉진하여 수용 필드(receptive field)를 극대화하며, Efficient Channel Attention(ECA) 모듈을 포함시켜 다양한 채널 간의 통신을 개선합니다. 이 방법은 전통적인 shifted window 접근 방식에 비해 낮은 파라미터 수와 계산 오버헤드를 유지하면서 경쟁력 있는 정확도를 입증합니다.

- **Technical Details**: FwNet-ECA의 기본은 2차원 이산 푸리에 변환(2D DFT)에 기반하고 있습니다. 단순화를 위해 1차원 이산 푸리에 변환(1D DFT)으로 설명하자면, 길이 N의 1차원 시퀀스는 N 개의 주파수(complex amplitude) 구성 요소의 합으로 분해될 수 있습니다. 이 방법은 필터링 작업을 통해 윈도우 주의(window attention)의 전역 수용 필드를 설정하며, Fast Fourier Transform(FFT)을 사용하여 필터 증강(Filter Enhancement) 계층의 계산 복잡성을 O(NlogN)으로 줄입니다.

- **Performance Highlights**: 이 모델은 iCartoonFace 데이터셋과 ImageNet의 하위 작업에서 검증되었습니다. 경쟁력 있는 성능을 달성하면서도 ViT, ResMLP 및 ResNet과 같은 유명 모델들에 비해 상당히 적은 파라미터 수와 계산 비용을 자랑합니다. 이 연구는 비주얼 처리(visual processing) 작업에서 주의 메커니즘을 활용하는 보다 효율적이고 효과적인 대안을 제시하며, 윈도우 주의 모델의 한계를 완화합니다.



### A Fusion Model for Art Style and Author Recognition Based on Convolutional Neural Networks and Transformers (https://arxiv.org/abs/2502.18083)
- **What's New**: 이 논문은 CNN(Convolutional Neural Network)과 Transformer 모델을 결합한 새로운 융합 모델을 제안하여 예술 스타일과 작가 인식을 위한 이미지 분류 정확도를 향상시키는 방법을 다루고 있습니다. 전통적인 CNN은 지역적 특징 추출에 뛰어나지만, 복잡한 글로벌 의존성 포착이 부족하며, Transformer는 반대로 글로벌 컨텍스트에 강하지만 세부 지역 정보를 잘 다루지 못합니다. 이 연구는 CNN으로부터 지역 특징을 추출하고, Transformer를 통해 글로벌 컨텍스트를 모델링함으로써 이러한 한계를 극복하고자 합니다.

- **Technical Details**: 제안된 모델은 먼저 CNN을 통해 예술 작품의 지역적 특징을 추출하고, 이후 Transformer를 사용해 이 지역적 특징의 글로벌 정보를 캡처합니다. 모델의 특징 융합 메커니즘을 통해 분류 성능을 개선하여, 중국화 및 유화 데이터셋에서 각각 9.7% 및 7.1%의 정확도 향상을 보였습니다. 이러한 접근 방식은 특히 데이터가 적은 환경에서도 강력한 성능을 유지하며 복잡한 예술 작품을 효과적으로 처리할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 융합 모델은 CNN 및 Transformer 각각의 단독 모델보다 우수한 성능을 나타내어 F1 점수가 각각 0.06 및 0.05 향상되었습니다. 이 연구는 예술 인식에 있어 현재 기술의 한계를 넘어서기 위한 기반을 제공하며, 향후 다중 모달 통합 및 아키텍처 최적화를 통해 추가적인 개선 가능성을 보여줍니다.



### Examining the Threat Landscape: Foundation Models and Model Stealing (https://arxiv.org/abs/2502.18077)
Comments:
          Accepted to BMVC 2024

- **What's New**: 이 논문에서는 이미지 분류 API가 모델 도용 공격에 취약하다는 새로운 발견을 제시합니다. 특히, Foundation Models (FMs)에서 미세 조정(fine-tuning)된 모델이 전통적 비전 아키텍처보다 더 높은 도용 위험을 갖는다고 주장합니다. 이는 FMs가 시각적 패턴 및 기능을 폭넓게 잠재적 사용자와 공격자에게 제공하기 때문입니다.

- **Technical Details**: 연구팀은 세 가지 데이터셋과 여러 모델을 통해 FMs에 기반한 이미지 분류 API의 모델 도용 취약성을 체계적으로 조사했습니다. 결과적으로 ViT 모델에서 미세 조정된 모델들이 ResNet과 같은 소규모 모델에 비해 도용 위험이 더 크다는 것을 발견했습니다. 연구에서 도난당한 모델의 예측과 일치율도 보고되었고, ViT-L/16 모델이 가장 높은 일치율을 기록했습니다.

- **Performance Highlights**: 이 연구는 MLaaS 설정에서 상업적 API 배포 시 FMs 사용의 위험성을 경고합니다. 연구진은 높은 정확도와 보안 문제 사이의 트레이드오프를 강조하며, 이러한 모델의 안전성을 확보하기 위한 강력한 보안 프로토콜 및 대응책 필요성을 도출했습니다. 이 결과는 상업적으로 활용되는 API에서 모델 도용을 방지하기 위해 더 나은 선택을 요구합니다.



### Escaping The Big Data Paradigm in Self-Supervised Representation Learning (https://arxiv.org/abs/2502.18056)
Comments:
          Code and implementation available at: this https URL

- **What's New**: 이번 논문에서는 이미지에서의 자기 지도 학습(self-supervised learning)에서 방대한 데이터 세트에 의존하지 않고도 효과적인 표현 학습을 할 수 있는 방법을 제안합니다. 저자들은 SCOTT (Sparse Convolutional Tokenizer for Transformers)이라는 얕은 토크나이징 아키텍처를 도입하여 Vision Transformers (ViTs)의 성능을 향상시킵니다. 이 방법은 weak supervision을 기반으로 하며, 기존의 큰 데이터 세트에 의존하는 전통적인 접근 방식의 한계를 극복합니다.

- **Technical Details**: SCOTT은 CNN(Convolutional Neural Networks)의 유도 편견을 ViTs에 통합하는 토크나이징 아키텍처로, 이는 작은 규모의 데이터 환경에서도 효과적으로 작동하도록 설계되었습니다. 아울러, MIM (Masked Image Modeling) 프레임워크 내에서 Joint-Embedding Predictive Architecture (JEPA)를 활용하여 훈련됩니다. 이 방법은 대규모 사전 훈련 데이터 세트 없이는 학습이 어려운 기존 모델들과 달리, 정답 레이블 없이 대상 데이터 세트의 이미지만을 사용하여 훈련할 수 있습니다.

- **Performance Highlights**: SCOTT와 JEPA의 조합은 유한한 데이터 환경에서 ViT가 높은 성능을 발휘할 수 있도록 하며, 세밀한 비주얼 작업에서 기존 방법들보다 현저히 우수한 성능을 보입니다. 세 개의 소규모 데이터 세트, 즉 Oxford Flowers-102, Oxford IIIT Pets-37, ImageNet-100 등을 사용하여 검증하였으며, 결과적으로 대규모 사전 훈련, 복잡한 이미지 증강 및 더 큰 모델 크기에 의존하지 않고도 경쟁력 있는 결과를 얻었습니다. 이러한 발견은 의료 영상이나 로봇 공학과 같이 자원이 제한된 환경에서의 컴퓨터 응용 분야에 대한 새로운 길을 제시합니다.



### Progressive Local Alignment for Medical Multimodal Pre-training (https://arxiv.org/abs/2502.18047)
- **What's New**: 이 논문에서는 Progressive Local Alignment Network (PLAN)을 제안하여 의료 이미지와 텍스트 간의 국소 정렬(local alignment)을 개선합니다. PLAN은 대조 학습(contrastive learning)을 통해 단어와 픽셀의 관계를 형성하고 점진적 학습(progressive learning) 전략을 통해 이러한 관계를 반복적으로 정제하여 정렬의 정확성과 강인성을 높이는 방식입니다. 이는 기존의 의료 이미지 정렬 방법들이 극복하지 못했던 불규칙한 경계를 가진 구조를 더 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: PLAN의 접근 방식은 대조 학습을 통해 단어-영역 관계 학습을 두 개의 상호 연결된 하위 문제로 설정합니다. 첫 번째 하위 문제는 단어와 픽셀 간의 관계를 사용하여 지역 정렬(local alignment)을 수행하고, 두 번째는 방사선 전문의의 진단 추론에 영감을 받은 점진적 학습 전략으로, 지역 정렬 관계를 반복적으로 향상시키는 것입니다. 이 과정에서 다양한 의료 데이터셋을 사용하여 PLAN의 효과iveness를 평가하고 있습니다.

- **Performance Highlights**: 실험 결과, PLAN은 구문 기초(pharse grounding), 이미지-텍스트 검색(image-text retrieval), 객체 감지(object detection) 및 제로샷 분류(zero-shot classification)와 같은 여러 작업에서 기존 최첨단 방법들을 초월하여 새로운 기준을 세웠습니다. PLAN은 특히 세밀한 영역과 텍스트 간의 연관성을 강화하여 임상 결정 지원(clinical decision support)을 향상시킵니다. 이에 따라 의학 분야에서의 실세계 응용 가능성이 높아지고 있습니다.



### VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion (https://arxiv.org/abs/2502.18042)
- **What's New**: 이번 논문에서는 VLM-E2E라는 새로운 프레임워크를 제안하여 자율주행 시스템의 성능을 향상시키고자 하였습니다. 이 프레임워크는 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 시멘틱(semantic) 정보와 텍스트 표현을 Bird's-Eye-View (BEV) 특성에 통합함으로써, 운전자의 주의(attention) semantics를 명시적으로 포착합니다. 또한, BEV-Text 학습 가능한 가중치 융합 전략을 도입하여 시각 및 텍스트 정보를 효과적으로 활용하는 방법을 제안합니다.

- **Technical Details**: VLM-E2E는 E2E 자율주행 모델과 BEV 모듈의 통합을 통해 시각과 텍스트 특성이 결합된 풍부한 공간 인식을 가능하게 합니다. BEV-Text 가중치 융합 방식을 통해 각 모달리티의 중요성을 동적으로 조절하여, 상황에 맞는 시각 또는 텍스트 특성을 강조할 수 있습니다. 이 과정에서 운전 장면에 대한 텍스트 설명을 생성하고, 이를 통해 얻은 텍스트 정보를 CLIP 모델을 사용하여 밀집 표현으로 변환합니다.

- **Performance Highlights**: VLM-E2E는 nuScenes 데이터셋에서 평가될 때 기존 방법들보다 우수한 성능을 보였으며, 복잡한 주행 시나리오에서의 처리 능력을 크게 향상시켰습니다. 이 연구는 시각적 정밀성과 고수준 시멘틱 추론을 통합하여 보다 안전하고 해석 가능한 자율주행을 가능하게 합니다. VLM-E2E의 성능 개선은 E2E 자율주행 시스템의 결함을 극복하는데 기여할 수 있음을 보여줍니다.



### OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation (https://arxiv.org/abs/2502.18041)
- **What's New**: 이 논문에서는 Vision-Language Navigation (VLN)의 새로운 플랫폼인 OpenFly를 제안합니다. OpenFly는 다양한 도구와 대규모 벤치마크를 포함하여, 특히 실외 비행체(vin UAV)에 대한 VLN 연구를 촉진합니다. 기존의 데이터셋들이 실내 및 지상 기반이었던 것에 반해, OpenFly는 100,000개 경로와 18개의 풍부한 장면을 포괄합니다.

- **Technical Details**: OpenFly는 자동화된 데이터 수집을 위한 도구 체인을 개발하여, 포인트 클라우드 수집, 장면 의미 분할, 비행 경로 생성 및 지침 생성을 지원합니다. 이 플랫폼은 Unreal Engine, GTA V, Google Earth, 그리고 3D Gaussian Splatting(3D GS) 기술을 활용하여 다양한 고품질 시각 데이터를 생성합니다. OpenFly-Agent는 keyframe-aware VLN 모델로, 언어 지침, 현재 관찰 및 이력 keyframes를 입력으로 받아 직접 비행 작업을 출력합니다.

- **Performance Highlights**: OpenFly 플랫폼과 OpenFly-Agent는 광범위한 분석과 실험을 통해 그 우수함이 입증되었습니다. 특히 OpenFly는 자동화된 도구 체인과 대규모 데이터셋을 통해 데이터의 다양성과 효율성을 크게 개선하였습니다. 이 연구는 실외 항공 VLN의 새로운 기준을 제시하며, OpenFly의 도구와 코드는 오픈 소스로 제공될 예정입니다.



### ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents (https://arxiv.org/abs/2502.18017)
- **What's New**: 이번 논문에서는 시각적으로 풍부한 문서에서 정보를 이해하는 데 있어 기존의 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 데이터셋인 ViDoSeek을 소개합니다. ViDoSeek은 복잡한 추론을 요구하는 문서에서 RAG 성능을 평가하도록 설계되었습니다. 이를 통해 현재 RAG 접근 방식의 주요 한계도 식별하였으며, 특히 시각적 검색 방법과 추론 토큰 할당의 부족 문제가 강조되었습니다.

- **Technical Details**: ViDoRAG라는 새로운 다중 에이전트 RAG 프레임워크를 제안하여 시각적 문서에서 복잡한 추론을 위한 개선점을 제시합니다. 이 프레임워크는 Gaussian Mixture Model (GMM) 기반의 하이브리드 전략을 채택하여 다중 모달 검색(multi-modal retrieval)을 효과적으로 처리합니다. 또한 탐색, 요약, 반영 과정을 포함하는 반복적인 에이전트 워크플로우를 통해 모델의 추론 능력을 배양합니다.

- **Performance Highlights**: ViDoSeek에서 진행된 광범위한 실험 결과, ViDoRAG는 기존 방법들보다 10% 이상 성능이 향상된 것으로 나타났습니다. 이 결과는 ViDoRAG이 RAG 분야에서 테스트 시간 확장(test-time scaling) 연구에 대한 새로운 방향을 제공함을 나타냅니다. ViDoRAG은 경쟁적인 ViDoSeek 기준에서 기존 방법들과 비교하여 우수한 성능을 보였습니다.



### High-precision visual navigation device calibration method based on collimator (https://arxiv.org/abs/2502.18012)
- **What's New**: 이 연구는 정밀한 카메라 및 자세(calibration) 보정의 필요성을 해결하기 위해, 카메라 및 자세 보정을 위한 새로운 콜리메이터 기반 보정 방법과 시스템을 제안합니다. 기존의 복잡하고 시간 소모가 큰 카메라 보정 방법 대신, 이 연구는 단일 이미지 카메라 보정 알고리즘을 도입하여 시간을 대폭 단축시킵니다. 또한, 보정 프레임의 정밀 조정 메커니즘과 통합하여 효과적인 자세 보정이 가능하도록 합니다.

- **Technical Details**: 제안된 보정 시스템은 콜리메이터와 보정 프레임으로 구성되어 있으며, 안정적이고 신뢰할 수 있는 보정 데이터를 제공합니다. 콜리메이터는 광학 기기로, 균일한 조명을 제공하는 광원, 그라운드 글라스, 조준선, 렌즈로 구성됩니다. 이 시스템은 별도의 조정 기구를 통해 수평 및 정렬을 조정할 수 있어 높은 정확도의 보정을 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전통적인 다중 이미지 보정 기술과 비교하여 정확성과 안정성이 유사한 성과를 보였습니다. 재투사 오차(re-projection errors)는 0.1463 픽셀 이하이며, 평균 자세 각도 오차는 0.0586 도 이하, 표준 편차는 0.0257 도 이하로 나타났습니다. 이러한 결과는 이 방법의 높은 정확성과 강건성을 입증합니다.



### Shedding Light on the Polymer's Identity: Microplastic Detection and Identification Through Nile Red Staining and Multispectral Imaging (FIMAP) (https://arxiv.org/abs/2502.17997)
Comments:
          20 pages (with additional supplementary material), 5 Figures, 3 Tables

- **What's New**: 이 연구에서는 미세 플라스틱(microplastics, MPs)의 검출과 식별에 대한 새로운 접근 방법으로 Fluorescence Imaging Microplastic Analysis Platform (FIMAP)을 소개합니다. FIMAP은 네 개의 광학 필터와 다섯 개의 자극 파장을 갖춘 개선된 다중 분광 카메라로, 기존의 검출 기술보다 더 높은 정확도를 제공합니다. 이를 통해 다양한 미세 플라스틱 재질의 형광 행동을 포괄적으로 분석할 수 있게 되었습니다.

- **Technical Details**: FIMAP는 HDPE, LDPE, PP, PS, EPS, ABS, PVC, PC, PET, PA 등 10종의 Nile Red로 염색된 미세 플라스틱에 대한 정밀한 형광 분석을 수행합니다. 이 플랫폼은 K-means clustering을 사용하여 안정적인 세분화(Intersection over Union = 0.877)를 가능하게 하고, 20차원 색상 좌표 다변량 최근접 이웃(multi-dimensional nearest neighbor) 기법을 통해 미세 플라스틱을 분류합니다. 성능 지표로는 90%의 정밀도, 90%의 정확도, 100%의 재현율 및 F1 점수 94.7%를 달성했습니다.

- **Performance Highlights**: FIMAP는 특히 대규모 환경 샘플에서 미세 플라스틱을 식별하고 분류하기 위한 자동화된 고처리량(framework) 솔루션으로 제시됩니다. 그러나 35-104 마이크론 크기의 작은 미세 플라스틱에 대한 분류 정확도는 감소하여, 이는 염료 흡수 감소나 감지 가능한 픽셀 수의 부족과 관련이 있습니다. 앞으로 고배율 기기(예: 현미경)와 FIMAP의 통합이 미세 플라스틱 식별을 향상시킬 가능성이 있습니다.



### Improved YOLOv7x-Based Defect Detection Algorithm for Power Equipmen (https://arxiv.org/abs/2502.17961)
- **What's New**: 이 논문에서는 전력 장비의 이상 탐지를 위한 개선된 YOLOv7x 기반의 알고리즘을 제안합니다. 기존의 알고리즘에서 발생할 수 있는 배경 잡음과 불필요한 특징을 효과적으로 억제하기 위해 ACmix convolutional mixed attention mechanism 모듈을 도입했습니다. 또한 Biformer attention mechanism을 추가하여 핵심 특징에 대한 집중도를 높이고, 예측 및 실제 바운딩 박스 간의 관계를 종합적으로 평가할 수 있는 새로운 손실 함수인 MPDIoU를 적용했습니다.

- **Technical Details**: 제안된 알고리즘은 ACmix와 Biformer 어텐션 메커니즘을 기반으로 하여 네트워크의 특징 추출 능력을 향상시킵니다. ACmix 모듈은 배경 잡음 제거를 통해 신호 대 잡음비(signal-to-noise ratio)를 개선하였고, Biformer 모듈은 중요한 특징에 대한 인식을 강화하여 더욱 유연한 탐지를 가능하게 합니다. 마지막으로, MPDIoU 함수는 예측된 바운딩 박스와 실제 값 간의 불일치를 줄이는 데 기여했습니다.

- **Performance Highlights**: 개선된 알고리즘은 전반적인 탐지 정확도를 높여 모든 목표 범주에 대해 mAP@0.5/%가 93.5%에 달하며, 정밀도(precision)는 97.1%, 재현율(recall)은 97%에 도달했습니다. 이러한 성능 혁신은 전력 시스템에서의 이상 탐지에 매우 중요한 기여를 할 것으로 예상됩니다.



### Robust Polyp Detection and Diagnosis through Compositional Prompt-Guided Diffusion Models (https://arxiv.org/abs/2502.17951)
- **What's New**: 본 논문에서는 Progressive Spectrum Diffusion Model (PSDM)을 제안하여 다양한 임상 주석을 통합함으로써 합성 폴립 이미지를 생성하는 접근 방식을 소개합니다. 이 모델은 세그멘테이션 마스크, 바운딩 박스 및 대장내시경 보고서와 같은 다양한 정보를 조합하여 정교한 촉진(prompt)을 생성합니다. 이를 통해 모델이 넓은 공간 구조와 세부 정보를 효과적으로 캡쳐할 수 있도록 합니다.

- **Technical Details**: PSDM은 coarse와 fine 컴포넌트로 구성된 촉진 구조를 따라 다양한 임상 주석을 통합합니다. 이 방법은 원래의 세그멘테이션 마스크에서 제공되는 정보에 덧붙여 폴립 크기, 형태 및 조직학적 특성에 대한 텍스트 설명을 활용하여 출력 품질을 향상시킵니다. 이러한 접근 방식은 catastrophic forgetting 문제를 해결하기 위해 연속 학습(continual learning) 기법을 적용하여 모든 촉진으로부터 이미지를 생성할 수 있도록 합니다.

- **Performance Highlights**: PSDM을 통해 훈련 데이터를 증강함으로써 폴립 분류, 탐지 및 세그멘테이션 성능이 크게 향상되었습니다. PolypGen 데이터셋에서 PSDM은 F1 점수를 2.12% 증가시키고 평균 평균 정밀도를 3.09% 향상시켜 OOD 상황에서의 뛰어난 성능을 입증했습니다. 이러한 개선은 다양한 임상 시나리오에서의 일반화 가능성을 높이는 데 기여합니다.



### Optimal Brain Apoptosis (https://arxiv.org/abs/2502.17941)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문은 Convolutional Neural Networks (CNN)와 Transformers의 계산 효율성과 자원 요구 사항 문제를 다루기 위해 Optimal Brain Apoptosis (OBA)라는 혁신적인 pruning 방법을 제안합니다. 기존의 Hessian 행렬을 근사하는 방법 대신, 각 매개변수에 대한 Hessian-vector 곱을 직접 계산하여 매개변수 중요성을 추정합니다. 이 접근 방식은 Neural Network의 파라미터를 보다 정밀하게 다룰 수 있도록 하며, 다양한 데이터셋에서 CNN과 Transformers의 실험을 통해 검증되었습니다.

- **Technical Details**: OBA 방법은 매개변수의 2차 Taylor 전개를 효율적으로 계산하는 것을 목표로 합니다. 논문에서는 네트워크의 다양한 층 간 Hessian 하위 행렬을 분해하고, 이들 간의 비어있지 않은 조건을 식별하는 과정을 제시합니다. 이러한 방법을 통해 OBA는 구조적 및 비구조적 pruning 작업 모두에서 활용될 수 있을 뿐 아니라, 매개변수의 중요성을 정량화하는 데 더 정확한 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과, OBA는 VGG19, ResNet32, ResNet50, ViT-B/16 모델에서 CIFAR10, CIFAR100, Imagenet 데이터셋을 통해 기존 pruning 방법에 비해 성능 저하 없이 계산 효율성을 크게 향상시켰습니다. 이 연구는 효율적인 매개변수 중요성 추정의 새로운 접근법을 통해 deep learning 모델의 경량화와 최적화를 가능하게 했습니다. 논문에서 제공하는 코드를 통해 다른 연구자들도 이 방법을 쉽게 구현할 수 있도록 하고 있습니다.



### BD Currency Detection: A CNN Based Approach with Mobile App Integration (https://arxiv.org/abs/2502.17907)
- **What's New**: 이 연구에서는 전통적인 수동 검증 및 옵티컬 스캐닝 방식의 한계를 극복하기 위한 향상된 화폐 인식 시스템을 도입했습니다. Convolutional Neural Networks (CNNs)를 활용하여 방글라데시 은행권을 정확하게 분류할 수 있는 방법을 제시합니다. 이를 통해 50,334개의 이미지를 사용하여 훈련된 CNN 모델은 98.5%의 정확도를 기록하며 기존 방식보다 우수한 성능을 보였습니다.

- **Technical Details**: 데이터셋은 50,334개의 이미지로 구성되어 있으며, 모델 훈련을 위해 선처리(preprocessing) 및 CNN 모델 최적화를 통해 높은 성능을 발휘하도록 설계되었습니다. 모델은 TensorFlow Lite 형식으로 변환되어 Android 모바일 애플리케이션에 통합되어 실시간(real-time) 및 오프라인 기능을 지원합니다.

- **Performance Highlights**: 이 연구의 결과는 깊은 학습(deep learning)이 화폐 인식에 미치는 효과성을 강조하고 있습니다. 빠르고 안전하며 접근 가능한 솔루션을 제공하여 금융 거래 및 보조 기술을 향상시키는 데 기여하고 있습니다.



### From underwater to aerial: a novel multi-scale knowledge distillation approach for coral reef monitoring (https://arxiv.org/abs/2502.17883)
- **What's New**: 본 연구는 드론 기반 원격 감지와 AI 기반 방법론을 결합하여 산호초 생태계 모니터링을 위한 새로운 다중 규모 접근 방식을 제안합니다. 이 방법은 수중 이미지와 공중 이미지의 통합을 통해 정확하고 확장 가능한 산호초 평가를 가능하게 합니다. 특히 31개 산호 형태 및 관련 서식지를 탐지하는 변환기 모델을 활용하여, 고해상도의 뱅파 결과를 도출하고자 합니다.

- **Technical Details**: 자율 수상 차량(ASV)와 드론을 사용하여 라군에서 수집된 수중 및 공중 이미지를 결합합니다. 수중에서 촬영된 고밀도 이미지와 데이터 메타데이터를 통해 교차 정보를 이전하는 "가중 발자국" 방법을 적용합니다. 이 방법은 수중 이미지의 미세 데이터를 대규모 공중 이미지에 전이하며, 전반적으로 최종 분류 성능을 향상시킵니다.

- **Performance Highlights**: 연구 결과, 다중 규모 방법론을 통해 넓은 해역에서 높은 정확도로 산호 형태를 예측할 수 있음을 보여주었습니다. AUC (Area Under the Curve) 점수는 0.9251로, 고해상도 수중 데이터와 공중 데이터의 통합에 따른 정확한 결과를 나타냅니다. 이러한 접근 방식은 산호초 모니터링과 보존에 있어 혁신적인 가능성을 제공합니다.



### Dual Classification Head Self-training Network for Cross-scene Hyperspectral Image Classification (https://arxiv.org/abs/2502.17879)
- **What's New**: 이번 연구에서는 hyperspectral images (HSIs) 처리에서 cross-scene classification 문제를 해결하기 위해 dual classification head self-training network (DHSNet)을 제안합니다. 이는 source domain (SD)에서의 레이블이 있는 데이터와 target domain (TD)에서의 레이블이 없는 데이터를 활용하여 모델을 훈련하고, TD 데이터에 대한 추론을 수행하는 새로운 접근 방식입니다. 특히, 우리는 cross-scene HSI 분류 분야에서 처음으로 이중 분류 헤드(self-training) 전략을 도입하여 두 도메인 간의 클래스별 특성을 정렬합니다.

- **Technical Details**: DHSNet 모델은 도메인 간 특징 분포의 차이를 줄이고, 잘못된 pseudo-labels의 누적을 방지하는 방식으로 설계되었습니다. 이를 통해 TD 데이터의 다양한 클래스를 정확하게 분류할 수 있도록 훈련된 분류기가 개발됩니다. 또한, 중앙 특징 주의 메커니즘(central feature attention mechanism)을 통합하여 모델이 도메인 간 장면 불변 특징(scene-invariant features)을 더욱 효과적으로 학습할 수 있도록 하고 있습니다.

- **Performance Highlights**: 세 가지 cross-scene HSI 데이터셋에 대한 실험 결과, DHSNet은 기존의 최신 기법(state-of-the-art approaches)에 비해 월등한 성능 향상을 보여줍니다. 이 연구는 hyperspectral images의 cross-scene classification 성능 개선에 중요한 기여를 할 것으로 기대됩니다. DHSNet의 코드는 제공된 URL에서 확인할 수 있습니다.



### ASurvey: Spatiotemporal Consistency in Video Generation (https://arxiv.org/abs/2502.17863)
- **What's New**: 이 논문은 비디오 생성의 최근 발전을 체계적으로 검토하며, 공간 및 시간 일관성을 유지하는 데 중점을 두고 있는 다양한 연구를 포괄하고 있습니다. 기존 비디오 생성 관련 문헌에서 이 문제를 다룬 연구가 부족했음을 언급하며, 이로 인해 고품질 비디오 생성을 위한 기초적인 메커니즘을 깊이 이해하는 데 어려움이 있음을 알립니다. 이 설문조사는 비디오生成에 대한 기초 모델, 정보 표현 방법, 생성 방식, 후처리 기법 및 평가 메트릭의 다섯 가지 주요 측면을 다룹니다.

- **Technical Details**: 본 연구에서는 Generative Adversarial Networks (GAN), Autoregressive, Diffusion 및 Mask 모델을 포함한 여러 기초 모델을 요약하고, 이들이 어떻게 공간 및 시간 일관성을 유지하는지 설명합니다. GAN 모델은 생성자와 판별자로 구성되어 있으며, 생성자는 실제 데이터와 유사한 데이터를 생성하도록 훈련됩니다. Autoregressive 모델은 이전 프레임을 기반으로 현재 프레임을 생성하며, Diffusion 모델은 이미지 생성에서 비디오 생성으로 전이되어 활용됩니다.

- **Performance Highlights**: 공간 일관성(spatial consistency)과 시간 일관성(temporal consistency)을 유지하는 것은 비디오 생성의 핵심 과제입니다. 이 설문조사는 이러한 일관성을 달성하기 위한 최근의 접근 방식을 종합하였으며, 후속 연구를 위한 방향성과 도전 과제를 논의합니다. 미래 연구 개발이 비디오 생성 기술의 발전에 기여할 것으로 기대합니다.



### HRR: Hierarchical Retrospection Refinement for Generated Image Detection (https://arxiv.org/abs/2502.17862)
- **What's New**: 이번 연구에서는 이미지 생성 감지를 위한 새로운 프레임워크인 Hierarchical Retrospection Refinement (HRR)를 제안합니다. 기존의 생성 모델 감지 방법들이 특정 모델에 국한되어 있었던 반면, HRR는 다양한 생성 모델 및 이미지 크기에 대한 강력한 일반화 능력을 목표로 합니다. 이 프레임워크는 이미지의 스타일 정보 제거 및 다중 스케일 표현 생성을 통해 더욱 정교하고 현실감 넘치는 이미지 감지를 가능하게 합니다.

- **Technical Details**: HRR 프레임워크는 두 가지 주요 모듈로 구성됩니다: Multi-scale Style Retrospection (MSR)와 Additive Feature Refinement (AFR)입니다. MSR 모듈은 다양한 크기의 특징을 생성하며, 스타일 정보를 부드럽게 제거하여 모델의 강건성과 일반화 능력을 향상시킵니다. AFR 모듈은 correntropy sparse additive machine을 기반으로 하여 데이터의 본질적인 구조와 패턴을 포착하고, 불필요한 특징의 영향을 줄여 모델의 예측 성능을 개선합니다.

- **Performance Highlights**: HRR 프레임워크는 세 개의 기준 데이터세트에서 실험을 통해 기존 방법들보다 월등한 성능 향상을 보여주었습니다. 특히, 다양한 생성 모델에 대해 좋은 일반화 능력을 발휘하여 기존 감지기의 성능 저하 문제를 극복할 수 있음을 입증했습니다. 이번 연구는 이미지가 진짜인지 생성된 이미지인지를 판단하는 데 중요한 기여를 하며, 범위와 관련된 문제들을 효과적으로 해결했습니다.



### UniGS: Unified Language-Image-3D Pretraining with Gaussian Splatting (https://arxiv.org/abs/2502.17860)
Comments:
          ICLR 2025

- **What's New**: 본 논문에서는 UniGS라는 새로운 통합 텍스트-이미지-3D 사전 학습 프레임워크를 제안합니다. UniGS는 3D Gaussian Splatting(3DGS)을 3D 표현으로 활용하여 3D 인식 능력을 개선하는 데 초점을 맞추고 있습니다. 기존의 점 구름(point clouds) 기반 접근 방식이 가진 한계를 극복하기 위해, 3DGS의 이점을 최대한 활용하며 더욱 정교한 멀티모달 표현을 학습할 수 있도록 합니다.

- **Technical Details**: UniGS는 3DGS를 사용하여 3D 세계를 색상과 불투명성을 가진 3D 가우시안의 모음으로 모델링합니다. 이를 통해 3D 장면의 모든 정보를 포함하면서 2D 이미지와 강력한 연결을 형성합니다. 또한, Gaussian-Aware Guidance 모듈을 도입하여 3D 도메인에 대한 정밀한 표현 학습을 안내하며, 이를 통해 3D 인코더의 전역적인 명시적 3D 특징 추출을 용이하게 합니다.

- **Performance Highlights**: UniGS는 Objaverse, ABO, MVImgNet 및 SUN RGBD와 같은 다양한 데이터셋에서 일반화된 멀티모달 표현 학습의 효과를 입증하였습니다. 특히, 제로샷 분류에서 +9.36%, 텍스트 기반 검색에서 +4.3%, 오픈 월드 이해도에서 +7.92%의 remarkable improvements를 달성하였습니다. 이를 통해 UniGS는 다양한 3D 작업에서 최신 기술 수준의 성능을 이루어냅니다.



### Sketch-1-to-3: One Single Sketch to 3D Detailed Face Reconstruction (https://arxiv.org/abs/2502.17852)
- **What's New**: 이번 연구는 Sketch-1-to-3라는 새로운 프레임워크를 제안하여 단일 스케치에서 실제적인 3D 얼굴을 재구성하는 문제를 다룹니다. 이 프레임워크는 얼굴 스케치에서 지오메트릭 윤곽과 텍스처 세부 정보를 향상시키는 GCTD 모듈을 도입하고, 도메인 적응 모듈과 맞춤 손실 함수를 설계하여 3D 얼굴 공간과 스케치를 정렬합니다. 또한, 실손으로 그린 얼굴 스케치 데이터셋인 SketchFaces와 합성 스케치 데이터셋인 Syn-SketchFaces를 구축하여 평가 및 연구를 촉진합니다.

- **Technical Details**: 3D 얼굴 재구성의 주요 도전 과제는 2D 스케치와 3D 얼굴 구조 간의 큰 모달리티 차이입니다. 연구진은 2D 스케치에서 정확한 얼굴 키포인트를 추출하고, 다양한 얼굴 표정 및 세밀한 텍스처 세부 사항을 유지하는 것을 목표로 하고 있으며, 제한된 데이터로 고성능 모델을 훈련시키는 방법을 제안합니다. Sketch-1-to-3는 고충실도의 표현 및 텍스처 재구성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Sketch-1-to-3는 기존 스케치 기반 3D 얼굴 재구성 방법들에 비해 최첨단 성능을 달성했습니다. 이 연구는 단일 스케치에서 3D 얼굴 모델을 정확한 윤곽 및 세밀한 세부정보와 함께 재구성하는 데 명확히 초점을 맞추고 있습니다. 제안된 메소드와 데이터셋들은 스케치 기반 3D 얼굴 학습 분야의 연구를 크게 발전시킬 것으로 기대됩니다.



### A Novel Retinial Image Contrast Enhancement -- Fuzzy-Based Method (https://arxiv.org/abs/2502.17850)
Comments:
          This UPDATED version of the paper, accepted at the 2023 24th International Arab Conference on Information Technology (ACIT), includes corrections for typographical and grammatical errors, an joint authorship section with a detailed CRediT author statement, improvements in graphics and figure references, and refinements in citations

- **What's New**: 이 논문에서는 Fuzzy Contrast Enhancement(FCE)와 Contrast Limited Adaptive Histogram Equalization(CLAHE)를 결합하여 망막 혈관 구조 세분화를 위한 새로운 대비 향상 모델을 제안합니다. 이 모델은 특히 의료 진단과 관련된 망막 이미지의 정확성을 높이는 데 중요한 역할을 합니다. 제안된 방법은 기존 대비 향상 기술의 한계를 극복하여 더욱 선명한 혈관 구조를 구현할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 모델은 DRIVE 데이터세트를 기반으로 세 가지 버전으로 테스트되었습니다. 각 단계에서 FCE와 CLAHE의 강점을 활용하여 이미지를 선명하게 하는 방법론을 제시하고, Gaussian membership function을 사용하여 이미지의 밝기 수준을 퍼지화했습니다. FCE와 CLAHE의 출력을 선형 혼합 함수를 통해 조합하여 최적의 결과를 도출합니다.

- **Performance Highlights**: FCE와 CLAHE를 조합한 결과, 두 방법 모두 88%의 향상율을 보였으며, 이는 퍼지 논리를 통한 사전 처리의 효과를 증명합니다. 또한, 이 신규 방법은 기존의 Gray-scaling, Histogram Equalization(HE) 방법들과 비교할 때 혈관의 세밀한 구분에서 현저한 개선을 나타냈습니다. 이러한 성과는 망막 이미지의 정확한 분석과 진단 향상에 기여할 것으로 기대됩니다.



### Automatic Vehicle Detection using DETR: A Transformer-Based Approach for Navigating Treacherous Roads (https://arxiv.org/abs/2502.17843)
- **What's New**: 이 논문에서는 복잡하고 다양한 주행 환경에서 자동차 감지를 위한 최초의 Detection Transformer (DETR) 기술 실험을 진행하였습니다. 기존의 YOLO 및 Faster R-CNN 방법론이 도로 환경의 여러 복잡성 때문에 어려움을 겪는 반면, DETR는 Convolutional Neural Networks (CNNs)와 Transformer를 결합하여 감지 정확도와 효율성을 개선할 수 있는 새로운 가능성을 제공합니다.

- **Technical Details**: 연구팀은 Collaborative Hybrid Assignments Training (Co-DETR) 기법을 사용하여 DETR의 특징 학습 및 주의 메커니즘을 강화하였습니다. 이 방법은 레이블 할당 전략을 통해 효과적인 훈련 감독을 제공하고, 훈련 효율성을 높이는 긍정적인 좌표를 추출합니다. BadODD 데이터셋을 바탕으로 다양한 실험을 수행하였으며, DETR 변형 모델과 YOLO 모델을 비교하여 성능을 평가했습니다.

- **Performance Highlights**: 연구의 결과, 제안된 방법이 기존의 YOLO 및 Faster R-CNN 대비 우수한 성능을 보였으며, 다양한 조건에서도 개선된 정확도를 기록하였습니다. 이러한 연구는 자율 주행 기술의 발전에 기여하며, 복잡한 주행 환경에서 신뢰할 수 있는 객체 감지의 가능성을 열어줍니다. Co-DETR 기술의 활용으로 AVD를 위한 견고한 프레임워크를 제공할 것으로 기대됩니다.



### Weakly Supervised Pixel-Level Annotation with Visual Interpretability (https://arxiv.org/abs/2502.17824)
- **What's New**: 이 논문에서는 자동화된 설명 가능한 주석 시스템을 제안합니다. 이는 앙상블 학습(ensemble learning), 시각적 설명 가능성(visual explainability), 불확실성 정량화(uncertainty quantification)를 통합하여 의료 영상 주석의 정확성과 신뢰성을 크게 향상시킵니다. ResNet50, EfficientNet 및 DenseNet으로 구성된 세 가지 사전 훈련된 딥 러닝 모델을 결합하여 전문가의 진단 의사와 유사한 결과를 제공합니다.

- **Technical Details**: 제안하는 방식은 XGrad-CAM을 활용하여 시각적 설명을 제공하고 Monte Carlo Dropout을 통해 예측 불확실성을 정량화합니다. 이 시스템은 분류에 동의하는 모델의 saliency maps을 교차하여 픽셀 수준의 주석을 생성하며, 이를 통해 이미지 수준의 레이블만으로도 정확한 픽셀 주석을 얻을 수 있습니다. 불확실한 예측은 인간의 검토를 위해 플래그가 표시됩니다.

- **Performance Highlights**: TBX11K 의료 영상 데이터셋과 Fire 분할 데이터셋을 사용하여 평가한 결과, 우리 모델은 TBX11K에서 93.04%의 정확도, Fire 데이터셋에서 96.4%의 정확도를 달성했습니다. 또한, 이미지 수준의 레이블로 훈련한 모델임에도 불구하고 픽셀 수준 주석을 정밀하게 생성하여 각각 36.07% 및 64.7%의 Intersection over Union (IoU) 점수를 기록했습니다.



### Easy-Poly: A Easy Polyhedral Framework For 3D Multi-Object Tracking (https://arxiv.org/abs/2502.17822)
Comments:
          8 pages, 3 figures, 5 tables

- **What's New**: 이 연구에서는 Easy-Poly라는 새로운 실시간 3D 다중 객체 추적(3D MOT) 프레임워크를 소개합니다. Easy-Poly는 여러 개체 범주를 지원하며, 복잡한 환경에서 일반적인 트래킹 방법보다 상당한 개선을 보여줍니다. 고급 필터링과 데이터 연관 방식을 결합하여 높은 질의 탐지 성과를 달성하고, ID 전환 및 잘못된 종료를 줄이기 위해 조정 가능한 임계값을 활용합니다.

- **Technical Details**: Easy-Poly는 Augmented Proposal Generator를 통해 다중 모드 데이터 증강을 실시하고 SpConv 연산을 개선함으로써 탐지 정확도를 높입니다. 또한, 동적 추적 지향(Dynamic Track-Oriented, DTO) 데이터 연관 알고리즘을 도입하여 불확실성과 가림 문제를 효과적으로 관리합니다. 동적 모션 모델링(Dynamic Motion Modeling, DMM) 기술을 통해 칼만 필터에 대한 신뢰도 가중치를 접목시키고 적응형 노이즈 공분산을 활용하여 다양한 환경에서 성능을 향상시킵니다.

- **Performance Highlights**: Easy-Poly는 nuScenes 데이터셋에서 Poly-MOT 및 Fast-Poly와 비교하여 mAP 및 AMOTA에서 뛰어난 성과를 보였습니다. 예를 들어, CenterPoint의 mAP는 63.86%에서 64.89%로, LargeKernel3D는 63.30%에서 64.96%로 개선되었습니다. 이러한 개선을 통해 Easy-Poly는 자율 주행 및 관련 3D MOT 애플리케이션에서 효과적이고 신뢰할 수 있는 솔루션으로 자리잡을 것으로 기대됩니다.



### LAM: Large Avatar Model for One-shot Animatable Gaussian Head (https://arxiv.org/abs/2502.17796)
- **What's New**: 이번 연구에서는 LAM(Large Avatar Model)을 소개합니다. 이 모델은 단일 이미지로부터 즉시 애니메이션이 가능한 Gaussian 헤드를 생성하는 혁신적인 방법을 제시합니다. 기존 방법들과 달리 LAM은 추가적인 신경망 없이도 즉시 애니메이션 및 렌더링을 가능하게 하여 기존 렌더링 파이프라인과 원활하게 통합될 수 있습니다.

- **Technical Details**: LAM은 FLAME 모델의 정규화를 활용하여 초기 Gaussian 헤드의 형태를 수립합니다. Transformer 네트워크를 통해 다중 스케일 이미지 특징을 활용하여 정밀한 Gaussian 속성을 예측합니다. 이렇게 생성된 Gaussian 아바타는 표준 선형 블렌드 스키닝(Linear Blend Skinning, LBS) 기법을 이용해 애니메이션 할 수 있습니다.

- **Performance Highlights**: 실험 결과, LAM은 기존의 최첨단 기법들에 비해 뛰어난 성능을 보여줍니다. 이 접근 방식은 모바일 기기 등 다양한 플랫폼에서 실시간 애니메이션 및 렌더링을 지원하며, 단일 텍스트 프롬프트 또는 이미지로부터도 효율적인 생성 및 스타일화를 가능하게 합니다.



### Synthia: Novel Concept Design with Affordance Composition (https://arxiv.org/abs/2502.17793)
Comments:
          Code is available this https URL

- **What's New**: 본 논문은 기능적으로 일관된 디자인을 생성하기 위한 새로운 프레임워크인 SYNTHIA를 소개합니다. SYNTHIA는 사용자가 원하는 affordance(기능적 가능성)를 기반으로 하여 창의적인 개념 합성을 가능하게 하며, 이는 기존의 T2I 모델이 간과했던 기능적 일관성을 보장하는 데 중점을 둡니다. 또한, 계층적 개념 온톨로지를 활용하여 개념을 부분과 affordance로 분해하고, 이 구조적 접근을 통해 더욱 효과적인 디자인 제안이 가능합니다.

- **Technical Details**: SYNTHIA 프레임워크는 계층적 개념 온톨로지를 통한 affordance의 체계적인 구성이 주요 기술적 요소입니다. 이를 통해 모델은 일반적인 개념-기능 연관 외에도 복잡한 기능 조합을 학습하여, 긴밀하게 연결된 여러 기능을 하나의 일관된 형태로 통합하는 능력을 갖추게 됩니다. 더불어 교육 과정에 기반한 최적화를 통해 T2I 모델이 점진적으로 affordance 조합을 학습할 수 있도록 하며, 시각적 혁신을 유지하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, SYNTHIA는 기존의 T2I 모델을 크게 초월해 관찰 가능한 시각적 혁신성과 기능적 일관성 측면에서 각각 25.1% 및 14.7% 향상을 보였습니다. 이러한 결과는 인간 평가에서도 뚜렷하게 나타났으며, 새로운 디자인이 시각적으로 뿐만 아니라 기능적으로도 높은 실용성을 갖추었음을 증명합니다. SYNTHIA는 기존의 텍스트 기반 설명 의존에서 벗어나, 사용자가 지정한 affordance에 직접적으로 기반하여 창의적인 디자인을 생성할 수 있습니다.



### Improving Transformer Based Line Segment Detection with Matched Predicting and Re-ranking (https://arxiv.org/abs/2502.17766)
- **What's New**: 이번 논문에서는 RANK-LETR라는 새로운 Transformer 기반의 선분 탐지 방법을 제안합니다. 이 방법은 예측된 선분의 신뢰도 점수를 개선하여 최종 선택을 증가시키기 위해 학습 가능한 기하학적 정보를 활용합니다. 이를 통해 신뢰도 점수가 높은 고품질 선분의 순위를 정리합니다.

- **Technical Details**: 제안된 방법은 CNN 백본에서 추출한 다중 스케일 및 다중 레벨 이미지 특징을 변형 가능한 Transformer 인코더를 통해 처리합니다. 예측의 손실 함수를 정의할 필요 없이 신뢰도 및 위치 손실을 직접 적용하고, 신뢰도 점수가 높은 선분에 대해 더 높은 점수를 부여하는 선분 순위 손실(rank loss)을 도입합니다.

- **Performance Highlights**: 실험 결과 RANK-LETR는 다른 Transformer 및 CNN 기반 방법들보다 높은 예측 정확도를 보이며, 이전 Transformer 모델에 비해 훈련 에폭 수가 적어 성능을 향상시킵니다. 이는 선분 탐지 과정에서 훈련 효율성과 안정성을 크게 개선하는 것이 확인되었습니다.



### A digital eye-fixation biomarker using a deep anomaly scheme to classify Parkisonian patterns (https://arxiv.org/abs/2502.17762)
Comments:
          6 pages, 4 images

- **What's New**: 이 연구는 파킨슨병(Parkinson's disease, PD)의 안구 고정 패턴을 정량화하기 위한 새로운 동영상 분석 기법을 소개합니다. 전통적인 분류 모델과 달리, 이 접근법은 단일 클래스 학습(one-class learning)에 중점을 두어 대량의 데이터 없이 모든 다른 클래스를 이상(anomaly)으로 간주합니다. 이 방법은 13명의 건강한 대조군과 13명의 환자를 대상으로 평가되었으며, PD 환자의 경우 평균 감도(sensitivity) 0.97, 특이도(specificity) 0.63의 성능을 보였습니다.

- **Technical Details**: 이 연구에서 제안된 방법은 GANomaly 및 AnoGAN과 같은 두 가지 심층 생성 모델을 사용하여 PD 환자의 복잡한 안구 운동 패턴을 포착합니다. 이 모델은 병리학적 질환과 관련된 시공간적 영역을 식별하기 위해 파라메트릭 분포에 의존하지 않고 안구 고정 동영상을 분석합니다. 연구에서는 각 객체의 눈 영역을 수동으로 잘라내어 총 130개의 시퀀스를 기록하였고, 데이터의 다양성을 향상시키기 위해 여러 데이터 증가(data augmentation) 기법이 적용되었습니다.

- **Performance Highlights**: 제안된 디지털 바이오마커는 평균 AUC-ROC(Area Under the Receiver Operating Characteristic Curve) 0.95를 달성하며, 통계적 테스트를 통해 환자와 대조군 간에 유의미한 차이를 보였습니다(p < 0.05). 이는 PD의 조기 진단 및 병의 진행 추적에 유용한 결과를 나타냅니다. 이 연구는 안구 운동을 정량화하기 위한 새로운 접근법을 제공하며, 향후 PD의 생물학적 마커 개발에도 기여할 것으로 기대됩니다.



### AI-driven 3D Spatial Transcriptomics (https://arxiv.org/abs/2502.17761)
- **What's New**: 본 논문은 VORTEX라는 새로운 AI 프레임워크를 제안합니다. 이는 3D 조직 형태학과 최소한의 2D 공간 전사체 (spatial transcriptomics) 데이터를 활용하여 체적 (volumetric) 3D 전사체를 예측합니다. 기존의 3D 전사체 방법들이 가진 한계점들을 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: VORTEX는 다양한 조직 샘플에서 3D 형태-전사체(pair) 데이터를 사전 훈련 (pretraining)하고, 특정 관심 영역의 2D ST 데이터로 세밀하게 조정 (fine-tuning)하여 작동합니다. 이를 통해 기초적인 조직 관련 지식과 샘플 별 유전자 발현의 형태적 상관관계를 동시에 학습합니다. 이 기술은 고밀도, 고처리량 (high-throughput)의 빠른 3D ST을 가능하게 하여, 기존 기술의 한계를 넘어섭니다.

- **Performance Highlights**: VORTEX는 저비용 (cost-effective) 및 최소한의 파괴적인 방식으로 체적 분자의 통찰을 제공합니다. 이는 생체 표지자 발견 (biomarker discovery) 및 복잡한 조직 내 형상-분자 조합 (morphomolecular associations)과 세포 상태(cell states)에 대한 우리의 이해를 가속화하는데 기여할 것으로 기대됩니다.



### Task Graph Maximum Likelihood Estimation for Procedural Activity Understanding in Egocentric Videos (https://arxiv.org/abs/2502.17753)
Comments:
          arXiv admin note: text overlap with arXiv:2406.01486

- **What's New**: 본 연구에서는 절차적 활동에서 작업 그래프(task graphs)를 학습하기 위한 새로운 경량화 접근 방식을 제안합니다. 기존의 수작업으로 만든 방법들을 개선하여, 우리의 방법은 최대 우도(maximum likelihood)를 통해 직접적으로 엣지 가중치(edge weights)를 최적화하며, 이는 신경망 아키텍처(neural architectures)에 통합될 수 있습니다. 실험 결과는 CaptainCook4D, EgoPER, EgoProceL 데이터셋에서 각각 +14.5%, +10.2%, +13.6%의 F1-score 향상을 이루었습니다.

- **Technical Details**: 우리의 접근 방식은 절차적 활동의 키 스텝(key-step) 시퀀스를 바탕으로 작업 그래프를 학습합니다. 작업 그래프는 방향 비순환 그래프(Directed Acyclic Graph, DAG)로 정의되며, 각각의 노드는 주요 단계를 나타내고 방향성 엣지가 이러한 단계 간의 의존성을 정의합니다. 최대 우도 추정(maximum likelihood estimation) 프레임워크를 기반으로 하는 새로운 차별 가능(task graph maximum likelihood) 손실 함수를 제안하고, 이는 경량화 기법을 통해 그래디언트 하강법(gradient descent)으로 인접 행렬(adjacency matrix)을 직접 최적화할 수 있게 합니다.

- **Performance Highlights**: 제안된 모델은 CaptainCook4D 및 EgoPER 데이터셋에서 정확한 작업 그래프를 생성하는 능력을 평가받았으며, 6666개의 다운스트림(downstream) 작업에서도 유용성을 검증했습니다. Ego-Exo4D 절차 이해 벤치마크에서는 이전 키스텝, 선택적 키스텝, 절차적 오류, 누락된 키스텝, 미래 키스텝을 찾는 데 있어 실질적인 성과를 얻었고, Assembly101 및 EPIC-Tent의 온라인 오류 감지 작업에서도 각각 +19.8%와 +6.4%의 향상을 기록했습니다.



### Can Score-Based Generative Modeling Effectively Handle Medical Image Classification? (https://arxiv.org/abs/2502.17727)
Comments:
          Accepted at the International Symposium on Biomedical Imaging (ISBI) 2025

- **What's New**: 이 연구에서는 최신의 score-based generative 모델을 사용하여 의료 이미지를 분류하는 새로운 접근 방식을 제안합니다. 특히, 유방 촬영 이미지에 초점을 맞추어 CBIS-DDSM, INbreast 및 Vin-Dr Mammo 데이터 세트에서 우수한 분류 결과를 달성했습니다. 이는 기존의 분류 방법들과의 중요한 차별점을 보여주며, 의료 영상 분류의 넓은 맥락에서도 새로운 가능성을 제시합니다.

- **Technical Details**: 본 연구에서는 posterior approximation(후방 추정)과 likelihood approximation(우도 추정)의 두 가지 접근 방식을 통해 분류 작업을 수행합니다. 기존의 discriminative 모델(판별 모델)은 고차원 데이터에서의 유사성과 중첩된 분포 때문에 노이즈가 있는 이미지에서 신뢰성을 제한하는 경향이 있습니다. 이 연구는 score-based diffusion models(분산 모델)를 사용하여 이러한 문제를 해결하고, 클래스 간의 유사한 분포를 효과적으로 포착할 수 있는 잠재적 대안을 제시합니다.

- **Performance Highlights**: 제안된 generative classifier 모델은 CBIS-DDSM, INbreast, Vin-Dr Mammo 데이터 세트에서 경량화된 분류기 성능을 넘어서는 결과를 도출했습니다. 이는 기존의 discriminative 모델에 비해 상당한 이점을 보여줍니다. 코드 또한 공개되어 있어 더 많은 연구자들이 이 기술을 활용할 수 있도록 합니다.



### Contrastive Visual Data Augmentation (https://arxiv.org/abs/2502.17709)
- **What's New**: 이 논문에서는 새로운 Contrastive Visual Data Augmentation (CoDA) 전략을 제안하여 대형 다중모달 모델(Large Multimodal Models, LMMs)이 새로운 개념을 인식하고 논리적으로 처리하는 능력을 향상시키는 방법을 설명합니다. CoDA는 대상 개념의 주요 대조적 텍스트 및 시각적 특징을 추출하고, 이를 통해 작성된 합성 데이터를 생성하여 LMM이 혼동할 수 있는 개념을 명확히 구분할 수 있도록 돕습니다.

- **Technical Details**: CoDA의 주된 과정은 대조적 특징 추출, 특징 필터링, 특징 제어 이미지 생성, 증강 이미지 필터링의 4단계로 구성됩니다. 이 과정에서 CoDA는 LMM이 잘못 인식하는 개념과 혼동하게 되는 개념을 식별하고, 해당 개념의 시각적 및 텍스트적 특징을 추출합니다. 이 특징들은 가시성이 뛰어나고 LMM이 이해할 수 있는 방식으로 생성되고 필터링을 거쳐 최종적으로 증강 이미지로 변환됩니다.

- **Performance Highlights**: CoDA는 iNaturalist와 SUN와 같은 다양한 데이터셋에서 성능이 크게 향상되는 것을 확인할 수 있었으며, NovelSpecies라는 새로운 데이터셋에서도 테스트 결과 기존의 시각적 데이터 증강 기법보다 절대적으로 12.3%, 5.1%, 6.0%의 정확도 향상을 보여주었습니다. 이로써 CoDA는 LMM의 새로운 개념 인식 능력을 효과적으로 개선하는 데 성공하였으며, 이에 따라 비전 커뮤니티에 중요한 기여를 하고 있습니다.



### IBURD: Image Blending for Underwater Robotic Detection (https://arxiv.org/abs/2502.17706)
- **What's New**: IBURD(이미지 혼합 파이프라인)는 해양 쓰레기 탐지를 위한 심층 탐지기 훈련을 지원하고, 사실적인 합성 이미지를 생성하는 혁신적인 방법을 제시합니다. 이 방법은 쓰레기 객체의 이미지를 생성하고, 그 픽셀 수준의 주석을 제공하여 해양 환경의 배경 이미지와 결합하는 기술을 사용합니다. Poisson editing(포아송 편집) 및 스타일 전송 기법을 통해 IBURD는 투명한 객체를 임의의 배경에 강력하게 혼합할 수 있습니다.

- **Technical Details**: IBURD는 객체 이미지와 그 픽셀 수준의 주석, 그리고 목표 배경 이미지를 입력으로 받아 두 단계의 프로세스를 수행합니다. 첫 번째 단계인 Poisson editing을 통해 객체를 배경에 혼합하고, 두 번째 단계인 스타일 전송을 통해 객체의 외관을 변경합니다. 이 과정에서 주석을 동시에 업데이트하여 현실적인 합성 이미지를 생성하는 것이 가능합니다.

- **Performance Highlights**: IBURD를 통해 생성된 합성 이미지들은 데이터 부족과 데이터 다양성 문제를 해결하는 데 기여하며, 자율 수중 차량(AUV)이 자원 소모가 적은 환경에서도 효율적으로 객체 탐지를 수행할 수 있도록 합니다. 실제 환경에서 훈련된 탐지기의 성능을 정량적으로 평가하였고, IBURD에 의해 보강된 데이터셋을 통해 탐지 성능이 개선되었음을 입증하였습니다.



### Semi-Supervised Weed Detection in Vegetable Fields: In-domain and Cross-domain Experiments (https://arxiv.org/abs/2502.17673)
Comments:
          8 pages, 4 figure

- **What's New**: 이번 연구에서는 효율적인 잡초 감지를 위해 YOLOv8 기반의 반지도학적 물체 탐지(SSOD) 방법인 WeedTeacher를 도입했습니다. 이 방법은 라벨이 없는 데이터를 활용하여 기존의 SSOD 방법들과 비교하여 성능을 향상시키는 데 중점을 두고 있습니다. 실험은 19,931장의 다양한 필드 이미지를 포함한 새로운 잡초 데이터셋을 기반으로 진행되었습니다.

- **Technical Details**: 연구에 사용된 잡초 데이터셋은 두 개의 서로 다른 도메인을 나타내는 두 개의 하위 집합으로 구성됩니다. 첫 번째 하위 집합은 8,436장의 이미지로 구성된 공개 3SeasonWeedDet10 데이터셋이며, 두 번째 하위 집합은 특정 농장에 의해 수집된 11,496장의 라벨이 없는 이미지입니다. 연구는 교차 도메인 실험 및 In-domain 설정 하에 진행되어 각각의 효과성을 비교했습니다.

- **Performance Highlights**: YOLOv8 기반의 WeedTeacher는 In-domain 실험에서 모든 SSOD 방법 중 가장 높은 정확도를 기록했으며, 2.6% mAP@50 및 3.1% mAP@50:95의 개선을 보여주었습니다. 그러나 교차 도메인 실험에서는 모든 SSOD 방법이 감독 모델과 비교하여 유의미한 개선을 보이지 않았으며, 이는 더 많은 연구가 필요함을 시사합니다.



### METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling (https://arxiv.org/abs/2502.17651)
- **What's New**: 이 논문에서는 효과적인 자동 챠트 생성을 위한 비전-언어 모델(Vision-Language Model, VLM)을 기반으로 하는 다중 에이전트 프레임워크인 METAL를 제안합니다. METAL는 각기 다른 전문 역할을 가지고 있는 에이전트들 간의 협업을 통해 복잡한 다중 모달 추론 과제를 분해합니다. 이를 통해 챠트 생성 정확도를 5.2% 향상시켜 기존에 비해 상당한 개선 효과를 보여줍니다.

- **Technical Details**: METAL 프레임워크는 주어진 참조 챠트 이미지를 기반으로 프로그래밍 사양을 학습하는 것을 목표로 합니다. 네 가지 전문 에이전트(Generation Agent, Visual Critique Agent, Code Critique Agent, Revision Agent)가 협력하여 직관적으로 진행되며 각 에이전트는 반복적으로 결과물을 개선합니다. 특히, 이 프레임워크는 시각적 평가와 코드 분석 시에 서로 다른 모달리티를 분리하는 방식을 채택하여 VLM의 자기 수정 능력을 향상시킵니다.

- **Performance Highlights**: METAL을 통해 얻은 실험 결과는 챠트 생성 정확도를 11.33% 이상 향상시키며, 이는 VLM이 시각적 이해와 코드 합성을 통합하는 능력을 크게 향상시킨다는 것을 보여줍니다. 또한, 테스트 시간 스케일링(timing scaling) 특성을 발견하여 컴퓨팅 예산이 증가함에 따라 METAL의 성능이 일관되게 개선된다는 것을 증명하였습니다. 이러한 성과는 VLM 기반의 비주얼 중심 코드 생성 향상에 있어 유망한 경로를 제공합니다.



### CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR-Camera Calibration with Iterative and Attention-Driven Post-Refinemen (https://arxiv.org/abs/2502.17648)
Comments:
          Submitted to Transportation Research Part C: Emerging Technologies

- **What's New**: 이번 논문에서는 CalibRefine라는 새로운 자동 및 타겟리스(targetless) 온라인 캘리브레이션 프레임워크를 제안합니다. 이 시스템은 LiDAR 포인트 클라우드와 카메라 이미지를 직접 처리하여, 사람의 개입을 최소화하고 실시간으로 캘리브레이션을 수행할 수 있습니다. 기존 방법의 제한사항을 극복하고, 적은 노력으로도 높은 정확도를 달성할 수 있는 점이 혁신적입니다.

- **Technical Details**: CalibRefine는 네 개의 단계로 나뉘어 있습니다. 첫 번째 단계는 Common Feature Discriminator로, 자동으로 감지된 객체를 활용하여 신뢰할 수 있는 LiDAR-카메라 대응점을 생성합니다. 이어서 조잡한 호모그래피를 기반으로 한 캘리브레이션이 이루어지고, 추가 데이터 프레임이 입력될 때마다 반복적인 정제를 통해 정렬을 개선합니다. 마지막 단계에서는 Vision Transformer와 크로스 어텐션 메커니즘을 활용하여 비평면 왜곡을 처리하는 어텐션 기반의 정제가 진행됩니다.

- **Performance Highlights**: 다양한 도시 교통 데이터 세트에 대한 실험을 통해 CalibRefine는 기존의 타겟리스 방법보다 우수한 성능을 보이며, 수동으로 조정한 기준과 경쟁하거나 이를 초월하는 결과를 도출하였습니다. 이 연구 결과는 강력한 객체 수준 피처 매칭과 반복적이며 자가 감독된 어텐션 기반 조정이 실제 환경에서의 센서 융합을 일관되게 수행할 수 있음을 보여줍니다.



### A Priori Generalizability Estimate for a CNN (https://arxiv.org/abs/2502.17622)
- **What's New**: 이번 연구에서는 전체 합성곱 신경망(CNN)의 잘린 특이값 분해(truncated SVD)를 정의하고 이를 통해 CNN의 성능 저하가 예상되는 이미지를 식별하는 데 유용하다는 것을 입증하였습니다. 연구자들은 Right Projection Ratio와 Left Projection Ratio라는 두 가지 새로운 지표를 도입하여, 회귀 관점에서 이미지와 특이 벡터 간의 투영 충실도를 평가합니다. 이러한 접근은 기존의 진단 도구에서의 제한을 극복할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 연구에서 분석한 CNN은 컨볼루션, 다운샘플링, 업샘플링, ReLU 활성화 및 스킵 커넥션으로 구성되어 있습니다. 입력 텐서와 그에 대응하는 투영 행렬을 정의함으로써 CNN의 구조를 명확하게 설명하며, 이는 다양한 차원에서의 이미지 데이터를 처리할 수 있는 모델의 특성을 보여줍니다. 잘린 SVD 알고리즘은 CNN의 adjoint(켈리 변환)를 활용해 매트릭스 연산을 최적화하여 효율적인 라이트 프로젝션 비율을 계산합니다.

- **Performance Highlights**: Right Projection Ratio는 레이블이 없는 데이터만으로도 모델의 성능과 상관관계가 있는 것으로 나타났습니다. 이는 이미지 분할 문제에서 특정 이미지 세트에 대해 CNN의 신뢰성 있는 예측 지표로 작용할 수 있음을 시사합니다. 두 가지 데이터 세트에서 실험을 수행하여, 여전히 CNN의 성능을 저하시키는 클래스 불균형 문제를 효과적으로 식별할 수 있음을 보여주었습니다.



### PosterSum: A Multimodal Benchmark for Scientific Poster Summarization (https://arxiv.org/abs/2502.17540)
Comments:
          This paper includes a dataset of research posters with abstracts. We provide two cited examples ( arXiv:2211.11880 and arXiv:2210.07571 ) to illustrate reference summaries

- **What's New**: 이번 논문에서는 PosterSum이라는 새로운 멀티모달 벤치마크를 도입하여 시각적으로 복잡한 내용인 학술 포스터를 연구 논문 초록으로 요약하는 모델 개발을 지원하고자 합니다. PosterSum 데이터셋은 기계 학습 회의에서 발표된 16,305개의 포스터와 해당 초록을 포함하고 있으며, 이는 다양한 시각적 이해 도전과제를 제공합니다. 이 연구는 최신 Multimodal Large Language Models (MLLMs)가 학술 포스터 요약에서 직면하는 한계를 강조합니다.

- **Technical Details**: PosterSum 데이터셋은 각 포스터를 이미지 형식으로 제공하고 있으며, 포스터는 복잡한 레이아웃, 밀집한 텍스트 영역, 표 및 그림 등의 다양한 시각적 도전과제를 나타냅니다. 본 연구에서는 Segment & Summarize라는 계층적 방법을 제안하여 각 포스터를 일관된 영역으로 분할하고, 각 영역의 텍스트를 추출하여 지역 요약을 생성한 후, 이를 종합하여 포스터 전체를 아우르는 요약을 작성합니다. 이 과정에서는 추가 학습이나 미세 조정이 필요하지 않아 효율적입니다.

- **Performance Highlights**: 제안된 방법은 ROUGE-L 점수 24.18을 달성하여 기존의 MLLMs를 초과하며, 이는 학술 포스터 요약에서 새로운 기준을 설정합니다. PosterSum 데이터셋은 향후 멀티모달 과학 포스터 이해 연구에 기여할 수 있는 기초 자료가 됩니다. 또한, 이 연구는 MLLMs의 파인튜닝에의 유용성을 입증하여 제로샷 결과에 비해 유망한 개선 결과를 보여줍니다.



### From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithms (https://arxiv.org/abs/2502.18328)
- **What's New**: 이 논문에서는 최근의 시각적 이상 탐지(Visual Anomaly Detection, VAD) 기술을 오디오 이상 탐지(Audio Anomaly Detection, AAD) 분야에 적용하는 방법을 제시합니다. 전통적인 AAD 기법들은 이상 샘플을 주로 분류하지만, 이 연구에서는 스펙트로그램 내 이상 탐지의 세분화된 시간-주파수(localization) 정보를 제공합니다. 이러한 접근은 사용자에게 이상이 발생한 위치와 시간을 명확하게 제시하여 더욱 활용 가능한 결과를 제공합니다.

- **Technical Details**: 이 연구에서는 비지도학습(unsupervised setting)에서 VAD 알고리즘을 평가하는 프레임워크를 채택하였습니다. 특히, 스펙트로그램을 입력으로 사용하여 특징 추출기(feature extractor)를 통해 생성된 중간 표현을 활용합니다. 이 과정에서 여러 메모리-뱅크(memory-bank) 기술을 포함한 네 가지 유형의 특징 기반 VAD 알고리즘을 평가하여, 각각의 알고리즘이 스펙트로그램 내에서 이상 영역을 어떻게 식별하는지를 분석합니다.

- **Performance Highlights**: MIMMI와 EnvMix 두 가지 벤치마크를 통해 실험을 수행하여, 산업 응용 및 환경 응용에서의 알고리즘 성능을 비교합니다. 다양한 신호 대 잡음비(Signal-to-Noise Ratio, SNR) 레벨에서 모델의 이상 탐지 능력을 평가하며, 이를 통해 AAD 기법의 전반적인 효용성을 검증합니다. 연구 결과는 오디오 이상 탐지 시스템의 설명 가능성을 높이고, 실제 상황에서의 적용 가능성을 높이는 데 기여합니다.



### GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music (https://arxiv.org/abs/2502.18309)
- **What's New**: 이번 연구에서 제안하는 GCDance는 음악과 텍스트 프롬프트에 기반한 장르-specific 댄스 생성 프레임워크입니다. 기존의 방법들과 다르게, GCDance는 음악의 고수준 및 저수준 특징을 통합하여 다양한 장르의 댄스를 생성할 수 있는 가능성을 보여줍니다. 이를 통해, 동일한 음악에 대해서도 여러 가지 댄스 스타일을 일관성 있게 생성할 수 있습니다.

- **Technical Details**: GCDance는 고전적인 STFT(Short-Time Fourier Transform) 오디오 특성과 Wav2CLIP으로부터 추출한 딥 특징들을 융합하여 음악 특징을 효과적으로 표현합니다. 또한, CLIP 모델을 사용하여 장르 설명으로부터 텍스트 표현을 추출하고, 이를 밀접하게 결합하여 댄스 생성 파이프라인 내에서 장르 기반 제어를 수행합니다. 이 과정에서 FiLM(Feature-wise Linear Modulation) 레이어가 장르 정보를 기반으로 댄스 생성 프로세스를 조절합니다.

- **Performance Highlights**: GCDance는 FineDance 데이터셋을 통해 기존 상태의 접근 방식들과 비교하여 현저한 성능 향상을 이뤘습니다. 통계적으로, 여러 메트릭에서 최첨단 성과를 거두었으며, 손과 몸의 움직임에 대한 개별 평가에서도 뛰어난 결과를 보여주었습니다. AIST++ 데이터셋에서의 추가 결과를 통해, GCDance가 음악 기반 댄스 생성에 대한 효과적인 해결책임을 입증하였습니다.



### A Reverse Mamba Attention Network for Pathological Liver Segmentation (https://arxiv.org/abs/2502.18232)
Comments:
          16 pages, 3 figures

- **What's New**: RMA-Mamba는 시각적 상태 공간 모델의 능력을 향상시키는 혁신적인 아키텍처로, 특별히 설계된 Reverse Mamba Attention 모듈(RMA)을 도입합니다. 이 아키텍처는 장기적인 의존성을 포착하면서도 정밀한 지역적 특성을 유지하는 계층적 처리 파이프라인을 갖추고 있습니다.

- **Technical Details**: RMA-Mamba는 Vision Mamba(VMamba)의 효율적 시퀀스 모델링과 RMA의 특성 개선을 통합하여 다중 스케일에서 우수한 특성 학습을 달성합니다. RMA 모듈은 디코더 작업 중 진행적 특성 통합 전략을 구현하여 미세한 조직 변동과 대규모 해부학적 맥락을 동시에 포착합니다.

- **Performance Highlights**: RMA-Mamba는 T2 가중 MRI 스캔을 이용한 새로운 경화 간 데이터셋(CirrMRI600+)에 대해 92.08%의 Dice 계수와 87.36%의 평균 IoU를 달성하며, 기존의 간 분할 접근법에 비해 뛰어난 성능을 보여줍니다. 또한, RMA-Mamba의 일반화 능력은 CT 스캔의 간암 분할에서도 확인되어 Dice 점수 92.9%와 mIoU 88.99%를 기록했습니다.



### Liver Cirrhosis Stage Estimation from MRI with Deep Learning (https://arxiv.org/abs/2502.18225)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문에서는 다중 시퀀스 MRI를 이용한 자동 간경변 단계 추정의 최첨단 딥러닝 프레임워크를 제시합니다. 간경변은 간의 심각한 흉터(fibrosis)로, 여러 만성 간질환의 일반적인 결과입니다. 초기 진단은 합병증을 예방하기 위해 매우 중요하지만, 조기 간경변을 진단하는 소프트웨어나 데이터가 부족합니다.

- **Technical Details**: 우리는 CirrMRI600+라는 대규모 데이터셋을 활용하여 339명의 환자로부터 수집한 628개의 고해상도 MRI 스캔을 분석했습니다. 딥러닝 프레임워크는 시퀀스 특화된 피처 추출과 교차 시퀀스 주의 메커니즘을 통합하여 간경변의 미세한 조직 변화를 포착합니다. 또한, VGG-19, ResNet 변형, MambaVision, ConvNext 등 8가지 다양한 딥러닝 아키텍처를 평가하여 최상의 성능을 기록했습니다.

- **Performance Highlights**: 최고의 모델은 T1W 시퀀스에서 72.8%의 정확도를 달성하였고, T2W 시퀀스에서는 63.8%의 성능을 보였습니다. 전통적인 radiomics 접근 방식에 비해 월등히 뛰어난 성능으로, 자동 간경변 단계 분류의 새로운 기준을 세웠습니다. 본 연구는 간경변 조기 진단의 정확성을 향상시키고, 나아가 임상에서의 사용 가능성을 제시합니다.



### Training Consistency Models with Variational Noise Coupling (https://arxiv.org/abs/2502.18197)
Comments:
          23 pages, 11 figures

- **What's New**: 본 논문에서는 최근 이미지 생성 작업에서 경쟁력 있는 성능을 달성하고 있는 Consistency Training (CT)의 새로운 접근법을 소개합니다. 이는 Variational Autoencoders (VAE)의 구조에서 영감을 받은 노이즈 결합 (noise-coupling) 방식을 통해 데이터 의존적인 노이즈 방출 모델을 학습함으로써 이루어집니다. 이러한 방법은 고전적인 CT에서 고정된 전방 프로세스와 달리 노이즈-데이터 매핑의 기하학을 간접적으로 학습할 수 있게 합니다.

- **Technical Details**: 연구에서는 Flow Matching 프레임워크를 이용하여 노이즈 결합을 구현하는 새로운 CT 교육 방법을 제안합니다. 데이터에 따라 조정된 확률 분포를 학습하고, 추가적인 Kullback-Leibler divergence 손실로 정규화하여 end-to-end 훈련 절차를 개발합니다. 이러한 접근법은 데이터 의존적인 결합 분포를 통해 목표 ODE 흐름의 위치를 조정하여 모델의 학습 목표를 용이하게 만들어 줍니다.

- **Performance Highlights**: 실험 결과 다양한 이미지 데이터셋에서 상당한 생성 성능 향상이 나타났으며, 제안된 모델은 CIFAR-10의 비증류 CT FID에서 기존 최첨단 성능을 초과했습니다. 뿐만 아니라 ImageNet의 경우 $64 \times 64$ 해상도에서 2단계 생성 시 최첨단과 동등한 FID를 달성했습니다. 이러한 성과는 제안된 방법이 고차원 데이터에 대한 확장성과 효과적인 생성 성능을 제공함을 보여줍니다.



### VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with LoRA and Atrous Attention (https://arxiv.org/abs/2502.18185)
Comments:
          Submitted to IEEE JBHI

- **What's New**: 본 논문에서는 기계 학습을 기반으로 한 새로운 의료 영상 분할 모델인 VesselSAM을 제안합니다. VesselSAM은 aortic vessel segmentation을 위한 Segmentation Anything Model(SAM)의 수정된 버전입니다. AtrousLoRA라는 새로운 모듈을 도입하여 Atrous Attention과 Low-Rank Adaptation(LoRA)을 결합하여 영상 분할 성능을 개선합니다. 이 모델은 다중 스케일 컨텍스트 정보를 포착하여 지역 세부사항과 글로벌 컨텍스트를 모두 보존할 수 있는 성능이 특징입니다.

- **Technical Details**: VesselSAM은 Atrous Spatial Pyramid Pooling(ASPP)과 Attention 메커니즘을 통해 다중 스케일 정보를 효율적으로 캡처합니다. LoRA를 활용하여 프리트레인된 SAM의 이미지 인코더를 동결한 상태로 프로그램의 성능을 유지하면서 학습 가능한 파라미터 수를 줄일 수 있습니다. 이를 통해 기존의 대형 모델에 비해 계산 비용을 대폭 절감할 수 있으며, 보다 효율적인 fine-tuning을 가능하게 합니다.

- **Performance Highlights**: VesselSAM은 두 개의 도전적인 데이터셋, 즉 Aortic Vessel Tree(AVT) 데이터셋과 Type-B Aortic Dissection(TBAD) 데이터셋에서 평가되었습니다. 여러 의료 센터에서 DSC 점수 93.50%, 93.25%, 93.02%, 93.26%의 성과를 달성하며 최첨단 성능을 보여줍니다. 이러한 결과는 기계 학습 기반의 aortic vessel segmentation이 임상 환경에서 더욱 개선될 것으로 기대될 수 있음을 나타냅니다.



### SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inferenc (https://arxiv.org/abs/2502.18137)
- **What's New**: SpargeAttn은 다양한 생성 작업에 보편적으로 적용할 수 있는 새로운 sparse attention 기법입니다. 기존의 sparse attention이 특정 태스크에 맞춰 최적화된 반면, SpargeAttn은 기계 학습 모델의 end-to-end 성능을 유지하면서 다양한 모델에 대해 속도를 높일 수 있도록 설계되었습니다.

- **Technical Details**: SpargeAttn의 핵심 기술은 두 단계의 온라인 필터를 활용하여 sparse attention을 구현하는 것입니다. 첫 번째 단계에서는 입력의 attention 맵을 신속하고 정확하게 예측하여 필요하지 않은 행렬 곱셈을 생략합니다. 두 번째 단계에서는 온라인 softmax-aware 필터를 통해 추가적인 오버헤드 없이 좀 더 많은 행렬 곱셈을 생략합니다.

- **Performance Highlights**: SpargeAttn은 기존의 dense 및 sparse attention 모델에 비해 2.5배에서 5배 더 빠른 속도를 자랑합니다. 이 방법은 언어 모델링, 이미지 및 비디오 생성 등 다양한 작업에서 model의 end-to-end 성능을 유지하며 효율성을 극대화합니다.



### Enhancing Reusability of Learned Skills for Robot Manipulation via Gaze and Bottleneck (https://arxiv.org/abs/2502.18121)
- **What's New**: 본 연구에서는 Gaze 기반 병목 인식 로봇 조작(GazeBot)이라는 새로운 알고리즘을 제안합니다. GazeBot은 학습된 동작을 높은 재사용성으로 활용할 수 있게 해주며, 제공된 데모와 다르게 물체의 위치와 최종 작동기 자세가 변경되더라도 효과적으로 동작합니다. 이 알고리즘은 물체 조작에 필요한 시선 정보와 모션 병목(Eventual Bottleneck)의 두 가지 중요한 요소를 결합하여, 최신 모방 학습 방법들과 비교하여 뛰어난 일반화 성능을 달성합니다.

- **Technical Details**: GazeBot은 시선 중심의 포인트 클라우드(3D point cloud)를 사용하여 물체의 위치 변화에 강한 시각적 표현을 확보합니다. 또한, 모션을 두 가지 세그먼트로 구분하는 데이터 기반의 행동 세분화 방법을 사용하여, 학습된 동작이 기본적으로 재사용될 수 있는 방안을 제공합니다. 병목 지점의 예측은 3D 시선 위치와 gazed-centered point cloud로부터 이루어져, 이전에 보지 못한 객체 위치에서도 정확한 제어를 가능하게 합니다.

- **Performance Highlights**: GazeBot은 주어진 데모와 유사한 조건(인배급, ID)뿐 아니라, 새로운 조건(출계급, OOD)에서도 정확한 작업 수행을 보입니다. 연구결과 GazeBot은 과거의 최첨단 모델들과 비교했을 때, 물체 조작의 재사용 가능성을 크게 개선합니다. 이와 함께 GazeBot은 모델의 일반화性能을 유지하면서도 민첩성과 반응성을 손상시키지 않습니다.



### S-Graphs 2.0 -- A Hierarchical-Semantic Optimization and Loop Closure for SLAM (https://arxiv.org/abs/2502.18044)
Comments:
          8 pages, 9 figures, RAL submission

- **What's New**: 이 논문에서는 Situational Graphs 2.0 (S-Graphs 2.0)을 소개합니다. 이 모델은 3D 장면 그래프와 SLAM을 결합하여 로봇의 포즈(pose)와 지도(map) 최적화를 향상시키기 위한 효율적인 4계층 최적화 그래프를 생성합니다. 특히, 층(층)을 기반으로 한 루프 클로저(loop closure) 전략으로 시각적으로 유사한 영역에서의 잘못된 루프 클로저를 거부하는 기능이 포함되어 있습니다.

- **Technical Details**: S-Graphs 2.0은 환경을 Keyframes, Walls, Rooms, Floors의 네 가지 층으로 나누어 구성됩니다. 이 그래프에는 계층 구조를 활용하여 포즈와 지도의 정확성을 유지하면서도 계산 복잡성을 낮추는 알고리즘이 포함되어 있습니다. 새로운 바닥 탐지 모듈이 각 층에 바닥 수준의 의미론적 관계를 부여하여 더 나은 최적화를 제공합니다.

- **Performance Highlights**: 다양한 실제 다층 환경에서 알고리즘을 광범위하게 검증하였으며, S-Graphs 2.0은 대규모 다층 환경에서 뛰어난 성능을 보였습니다. 제안된 접근 방식은 여러 기존 기법들이 효율적으로 실행하지 못하는 계산 복잡성을 경계하면서 계층 지도를 생성할 수 있는 상태의 기술을 보여줍니다.



### InVDriver: Intra-Instance Aware Vectorized Query-Based Autonomous Driving Transformer (https://arxiv.org/abs/2502.17949)
Comments:
          Submitted to JICV (Journal of Intelligent and Connected Vehicles)

- **What's New**: 이번 연구에서는 InVDriver라는 새로운 vectorized query-based 시스템을 제안했습니다. 이 시스템은 masked self-attention layers를 통해 intra-instance spatial dependencies(인스턴스 내 공간 의존성)를 체계적으로 모델링하여 계획의 정확성과 경로의 부드러움을 향상시킵니다. 기존 시스템들이 공간적 상관관계를 간과했던 문제를 해결합니다.

- **Technical Details**: InVDriver는 perception(지각), prediction(예측), planning(계획) 등 모든 핵심 모듈에 masked self-attention 메커니즘을 통합합니다. 이를 통해 attention(주의)을 인스턴스 내 점 간 상호작용으로 제한하고, 구조적 요소의 조정과 함께 불필요한 inter-instance noise(인스턴스 간 잡음)를 억제합니다. 본 연구는 geometry coherence(기하학적 일관성)를 명확히 모델링하는 것이 vectorized 자율주행 시스템을 발전시키는 데 중요함을 검증했습니다.

- **Performance Highlights**: nuScenes benchmark에서의 실험 결과, InVDriver는 정확성과 안전성 측면에서 이전 방법들을 초월하며 state-of-the-art 성능을 달성하였습니다. 또한 높은 계산 효율성을 유지하고 있어 실용적인 배치 요구 사항을 충족합니다. 이는 end-to-end 시스템의 이론적 장점과 실제 적용 요구 사항 간의 격차를 메우는 중요한 발전으로 평가됩니다.



### Deep-JGAC: End-to-End Deep Joint Geometry and Attribute Compression for Dense Colored Point Clouds (https://arxiv.org/abs/2502.17939)
- **What's New**: 이 논문에서는 밀집한 컬러 포인트 클라우드를 위한 새로운 엔드 투 엔드 Deep Joint Geometry and Attribute Compression (Deep-JGAC) 프레임워크를 제안합니다. 이 프레임워크는 기하학(geometry)과 속성(attribute) 간의 상관관계를 이용하여 압축 효율성을 높입니다. 특히, 기하학과 속성 서브 인코더가 학습 기반의 인코더와 비학습 기반의 인코더 모두 호환될 수 있도록 설계되었습니다.

- **Technical Details**: Deep-JGAC 프레임워크는 attribute 정보를 기하학 인코딩에 융합하는 Attribute Information Fusion Module (AIFM)을 포함합니다. 또한, 기하학 압축 왜곡으로 인해 발생하는 포인트 클라우드의 기하학 및 속성 간의 불일치를 해결하기 위해 최적화된 다시 색칠하기 모듈이 제안됩니다. 이는 속성 코딩을 위한 기하학적으로 왜곡된 포인트 클라우드에 속성을 첨부하여 색상화를 향상시키고 계산 복잡성을 줄입니다.

- **Performance Highlights**: 실험 결과, Deep-JGAC는 기존의 G-PCC, V-PCC, GRASP, PCGCv2와 비교하여 D1-PSNR 기준으로 각각 평균 82.96%, 36.46%, 41.72%, 31.16%의 비트율 감소를 달성했습니다. 또한 인코딩/디코딩 시간은 V-PCC 및 IT-DL-PCC와 비교하여 평균 94.29% 및 24.70%, 96.75% 및 91.02% 감소했습니다. 이를 통해 제안하는 방법의 효과성과 효율성을 입증하였습니다.



### 3D Anatomical Structure-guided Deep Learning for Accurate Diffusion Microstructure Imaging (https://arxiv.org/abs/2502.17933)
- **What's New**: 본 연구에서는 생체 뇌의 미세구조를 탐구하기 위한 새로운 딥러닝 기반 프레임워크를 제안합니다. 이 프레임워크는 기존의 방법에 비해 고충실도(diffusion microstructure imaging) 및 빠른 이미징을 가능하게 하며, 해부학적 정보(anatomical information)와 파라미터 간의 상호 정보(mutual information)를 동시에 활용합니다. 이에 따라, 시간 효율성을 높이고 마이크로구조 추정의 정확도를 유지할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 T1w 이미지를 이용하여 얻은 세 가지 조직 확률 맵(tissue probability maps)을 통해 해부학적 priors를 활용합니다. 이는 뇌의 모호한 조직 경계를 정렬하여 매개변수 추정에서 발생할 수 있는 부정확성을 줄이는 데 기여합니다. 또한 이 프레임워크는 다양한 확산 모델을 수용할 수 있도록 유연한 네트워크 아키텍처를 채택하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 네 가지 최신 기술보다 우수한 성능을 보여주었습니다. 파라메트릭 맵(parametric maps) 추정 시, 피크 신호 대 잡음 비율(PSNR) 30.51$	ext{±}$0.58, 구조 유사도 지수(SSIM) 0.97$	ext{±}$0.004를 달성하였으며, 전통적인 촘촘한 샘플링 방법에 비해 15배의 시간 절약을 할 수 있음을 입증했습니다.



### AirCast: Improving Air Pollution Forecasting Through Multi-Variable Data Alignmen (https://arxiv.org/abs/2502.17919)
- **What's New**: 이번 논문에서는 대기 오염 예측을 위한 새로운 모델인 AirCast를 소개합니다. 이 모델은 날씨 및 공기 질 변수들을 통합하여 대기 조건과 오염 물질 농도를 동시에 예측할 수 있는 다중 작업 헤드 아키텍처를 사용합니다. 제안된 Frequency-weighted Mean Absolute Error (fMAE) 손실 함수를 통해 드문 오염 사건 예측의 어려움을 해결하고, 특정 변수의 선택을 통해 예측 정확도를 개선합니다.

- **Technical Details**: AirCast는 Vision Transformer (ViT) 아키텍처를 기반으로 하며, WeatherBench와 Copernicus Atmosphere Monitoring Service (CAMS) EAC4 데이터셋에서 획득한 날씨 및 공기 질 변수를 통합하여 대기 오염 예측을 수행합니다. 이 모델은 대량의 변수들을 효과적으로 처리하기 위해 변수 토큰화(variable tokenization) 및 변수 집계(variable aggregation) 모듈을 포함하고 있으며, 다중 작업 헤드 아키텍처를 통해 대기 및 오염 변수의 동시 예측이 가능합니다. fMAE 손실 함수는 오염 물질의 무거운 분포를 다루기 위해 설계되었습니다.

- **Performance Highlights**: 이 모델은 중동 및 북아프리카(MENA) 지역에서의 높은 PM 농도 예측에 집중하고 있으며, 해당 지역의 대기 질 저하를 방지하기 위한 연구입니다. AirCast는 기후 변화와 산업 발달로 악화된 대기 오염 문제를 해결하기 위한 혁신적인 접근 방식을 제공하며, 기존의 예측 방법보다 높은 정확도를 목표로 합니다. 이를 통해 정책 결정 및 오염 저감 전략 수립에 기여할 것으로 기대됩니다.



### FetchBot: Object Fetching in Cluttered Shelves via Zero-Shot Sim2Rea (https://arxiv.org/abs/2502.17894)
- **What's New**: 이번 연구에서는 FetchBot이라는 sim-to-real 프레임워크를 소개합니다. 이 프레임워크는 복잡한 선반에서 안전하고 일반화 가능한 객체 검색을 가능하게하는데 초점을 맞추고 있습니다. 특히, 제한된 움직임 공간과 복잡한 객체 역학이 있는 환경에서도 로봇이 안전하게 작업할 수 있도록 설계되었습니다. 이는 로봇이 출발점 없이도 기존 데이터를 기반으로 객체를 검색할 수 있도록 합니다.

- **Technical Details**: FetchBot은 고유의 voxel-based 장면 생성 방법인 UniVoxGen을 활용하여 다양한 복잡한 선반 장면을 대규모로 생성합니다. 이 과정에서 강화 학습(RL) 기반 정책을 사용하여 동적이고 안전한 객체 검색 경로를 생성합니다. 또한, 이 연구에서는 다중 뷰 표현 학습을 위한 새로운 아키텍처를 설계하여 주변 품목을 효과적으로 인식하고 충돌을 방지하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, FetchBot은 다양한 시뮬레이션 및 실제 환경에서 뛰어난 일반화 능력을 보여주었습니다. 특히 복잡한 선반 환경에서 다양한 객체를 효과적으로 처리하는 능력에서 두각을 나타냈습니다. 이 연구는 복잡한 객체 검색 과제를 해결하는 데 있어 큰 기여를 하며, 앞으로의 연구에 대한 통찰력을 제공합니다.



### A graph neural network-based multispectral-view learning model for diabetic macular ischemia detection from color fundus photographs (https://arxiv.org/abs/2502.17886)
- **What's New**: 이번 연구에서는 당뇨병성 황반 허혈(Diabetic Macular Ischemia, DMI)을 탐지하기 위한 그래프 신경망 기반 다채로운 뷰 학습(GNN-MSVL) 모델을 제안했습니다. 일반적인 색 망막 사진(Color Fundus Photographs, CFPs)을 활용하여 DMI를 탐지할 수 있는 가능성을 탐구하였으며, 이는 안과 의사들 사이에서의 회의론을 극복하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된 모델은 계산적 다채로운 영상(Calculation Multispectral Imaging, CMI)을 통해 CFPs로부터 24파장 다채로운 망막 이미지를 재구성합니다. 이후 ResNeXt101 구조를 사용하여 재구성된 이미지에서 특징을 추출하며, 맞춤형 점프 연결 전략을 가진 그래프 신경망(GNN)을 설계하여 교차 스펙트럼 관계를 강화합니다. 이는 포괄적이고 효율적인 다채로운 뷰 학습을 가능하게 합니다.

- **Performance Highlights**: 연구는 592명의 당뇨병 환자로부터 1,078개의 CFPs를 수집하였으며, 그 중 300명의 환자에서 DMI가 진단된 530개의 CFP를 분석하였습니다. 모델은 84.7%의 정확도와 0.900의 수신자 조작 특성 곡선 아래 면적(AUROC)을 달성하였으며, 이는 CFPs로부터 훈련된 기준 모델 및 인간 전문가들보다 높은 성능을 나타냈습니다. 이러한 결과는 AI 기반 CFP 분석이 DMI 탐지에 가능성이 있음을 시사하며, 조기 및 저비용의 검진에 기여할 것입니다.



### VVRec: Reconstruction Attacks on DL-based Volumetric Video Upstreaming via Latent Diffusion Model with Gamma Distribution (https://arxiv.org/abs/2502.17880)
- **What's New**: 이 논문에서는 VVRec이라는 최초의 DL 기반 Volumetric Video Reconstruction 공격 스킴을 설계했습니다. 기존의 여러 시스템의 중간 결과에서 원래의 포인트 클라우드를 재구성할 수 있는 능력을 보여줍니다. 또한, Gamma 분포와 정교화 알고리즘을 활용하여 재구성 품질을 크게 향상시킵니다.

- **Technical Details**: VVRec은 네 가지 잘 훈련된 신경망 모듈을 사용하는 Latent Diffusion Models를 기반으로 하여 설계되었습니다. 이 모델은 높은 품질의 포인트 클라우드를 재구성할 수 있으며, 기존의 방어 기법을 초월합니다. 또한, 우리의 접근 방식은 포인트 클라우드가 가진 높은 차원 기록 정보에 대한 분포 변화 문제를 다룹니다.

- **Performance Highlights**: VVRec은 세 가지 볼륨 비디오 데이터셋을 사용하여 평가되었으며, 64.70dB의 재구성 정확도를 달성했습니다. 또한 기존 기준선에 비해 46.39%의 왜곡 감소를 보여주었으며, 이는 VVRec의 우수한 성능을 입증합니다.



### TagGAN: A Generative Model for Data Tagging (https://arxiv.org/abs/2502.17836)
- **What's New**: 이 연구는 기존 AI 시스템이 가지는 투명성 부족 문제를 해결하기 위해 TagGAN이라는 새로운 Generative Adversarial Networks (GAN) 기반의 프레임워크를 제안합니다. TagGAN은 약한 감독 하에 질병의 픽셀 수준 지도를 생성할 수 있으며, 이는 라벨이 이미지 수준인 데이터만으로도 가능합니다. 이 연구는 의학 이미징에서 질병에 대한 정확한 시각화를 제공함으로써 진단 AI의 해석 가능성을 향상시킵니다.

- **Technical Details**: TagGAN은 도메인 변환 과정에서 비정상 이미지를 정상 표현으로 변환하면서 픽셀 수준의 질병 지도를 생성합니다. 이 후, 생성된 지도는 입력된 비정상 이미지에서 빼내어 모든 중요한 해부학적 세부 사항을 유지하면서 정상 이미지로 변환됩니다. TagGAN은 픽셀 수준 지도가 필요 없이 약한 감독 환경에서 잘 작동하는 최초의 모델로, 이는 의료 영상 분석에서 큰 발전을 의미합니다.

- **Performance Highlights**: CheXpert, TBX11K 및 COVID-19와 같은 기준 데이터셋에서 수행된 실험 평가를 통해 TagGAN은 질병 특정 픽셀을 정확하게 식별하는 데 있어 현재의 최상위 모델들을 능가하는 성능을 보여주었습니다. 이 결과는 TagGAN 모델이 의료 이미지를 태그할 수 있는 능력을 갖추었음을 강조하며, 훈련 중 이진 마스크의 필요성을 없애며 방사선 전문의의 작업 부담을 크게 줄일 수 있습니다.



### MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks (https://arxiv.org/abs/2502.17832)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서 제안하는 MM-PoisonRAG는 멀티모달 RAG 프레임워크에 대한 최초의 지식 오염 공격 프레임워크입니다. 공격자는 사실과 관련 없는 지식을 외부 지식 기반(KB)에 주입하여 모델이 잘못된 답변을 생성하도록 유도합니다. 두 가지 공격 전략인 Localized Poisoning Attack (LPA)와 Globalized Poisoning Attack (GPA)을 통해 특정 쿼리나 전반적인 조건에서 모델의 출력을 조작할 수 있습니다.

- **Technical Details**: MM-PoisonRAG는 LPA와 GPA 두 가지 공격 시나리오를 가지고 있으며, LPA는 쿼리와 관련된 잘못된 정보를 주입하여 특정 쿼리에 대한 조작을 목표로 합니다. 반면 GPA는 모든 쿼리에서 비관련 지식을 삽입하여 모델의 생성을 완전히 교란시킵니다. 이 연구에서는 각 공격이 모델의 응답 생성에 미치는 영향을 여러 작업과 설정에서 평가하였습니다.

- **Performance Highlights**: LPA는 최대 56%의 성공률로 공격자가 정의한 답변을 생성하는 데 성공했습니다. 반면 GPA는 단 한번의 비관련 지식 주입으로 모델의 정확도를 0%로 떨어뜨리는 결과를 보였습니다. 이러한 결과는 멀티모달 RAG 프레임워크를 보호하기 위해 강력한 방어책의 필요성을 강조합니다.



### Sample Selection via Contrastive Fragmentation for Noisy Label Regression (https://arxiv.org/abs/2502.17771)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 ConFrag라는 새로운 방법론을 제안합니다. 이 접근법은 레이블과 특성 간의 연속적인 상관관계를 활용하여 회귀 문제의 노이즈 레이블 문제를 해결하는 데 초점을 맞춥니다. 특히, 레이블 공간에서 가장 먼 조각들을 짝지어 대조적인 조각 쌍(contrastive fragment pairs)을 형성함으로써, 데이터 포인트 간 유사성에 기반한 훈련이 가능합니다.

- **Technical Details**: ConFrag(Contrastive Fragmentation) 프레임워크는 데이터셋을 대조적인 조각 쌍으로 나누고, 이 조각 쌍을 통해 클린 샘플을 선택하는 방식으로 작동합니다. 선택된 클린 샘플은 점진적인 학습을 통해 회귀 모델의 성능을 향상시킵니다. 이 프레임워크는 노이즈 비율에 무관하게 작동하며, 인접 조각의 혼합(mixture) 모델을 통해 주변 동의(neighborhood agreement)를 사용하여 노이즈 레이블을 식별합니다.

- **Performance Highlights**: ConFrag는 14개의 최첨단 기법들과 비교하여 일관되게 더 우수한 성능을 보여주었습니다. 다양한 도메인에서 수집한 6개의 벤치마크 데이터셋을 통해 실험을 수행하였으며, 이 과정에서 레이블 간의 노이즈 정도를 고려하는 Error Residual Ratio(ERR)라는 새로운 성능 지표를 도입했습니다. 이러한 방법론은 레이블 노이즈의 대칭 및 무작위 가우시안 노이즈에 강한 저항력을 가지고 있습니다.



### Label-free Prediction of Vascular Connectivity in Perfused Microvascular Networks in vitro (https://arxiv.org/abs/2502.17759)
- **What's New**: 이 연구에서는 레이블이 없는 혈관 연결성(vascular connectivity) 평가를 위한 VC-Net(Vessel Connectivity Network)을 개발했습니다. 기존의 형광 레이블이 적용된 방법들은 생체 적합성(biocompatibility) 문제를 일으키거나 세포 성장 과정에 방해가 될 수 있습니다. VC-Net은 이러한 문제를 해결하고, 혈관 세포 간의 연결성을 지속적으로 모니터링할 수 있는 방법을 제시합니다.

- **Technical Details**: VC-Net은 Vessel Queue Contrastive Learning (VQCL) 방법과 클래스 불균형(class imbalance) 알고리즘을 활용하여 훈련 데이터셋의 샘플 크기 제한, 모호한 클래스 특성 및 불균형 클래스 분포 문제를 해결합니다. 연구진은 다양한 배양 조건에서 미세혈관 네트워크(microvascular networks, MVNs)를 배양하고 그 미세 이미지를 수집하여 훈련 데이터셋으로 사용했습니다. 이 네트워크는 형광 이미징과 비교할 때 연결성을 정확하게 평가할 수 있음을 보여줍니다.

- **Performance Highlights**: VC-Net은 정상 미세혈관 네트워크와 종양 관련 미세혈관 네트워크 간의 연결성 특성을 성공적으로 구분했습니다. 종양 관련 미세환경에서 배양된 MVNs의 평균 연결성이 30.8% 감소한 반면, 비연결 영역(non-connected area)은 37.3% 증가하였습니다. 이 연구는 레이블 없는 지속적인 유기체(organoid) 또는 종양 혈관화 평가에 대한 새로운 경로를 제시합니다.



### Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures (https://arxiv.org/abs/2502.17710)
Comments:
          40 pages, 49 figures

- **What's New**: 이번 연구에서는 다양한 문화적 해석을 포함하는 새로운 데이터셋인 Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS)를 소개합니다. 이 데이터셋은 25가지 제스처와 85개 국가를 포함하여 총 288개의 제스처-국가 쌍을 다루고 있으며, 각 제스처의 공격성, 문화적 중요성 및 맥락적 요소에 대한 주석이 포함되어 있습니다. 이는 AI 시스템이 비문화적 콘텐츠를 생성할 위험을 줄이기 위한 필수적인 노력을 보여줍니다.

- **Technical Details**: MC-SIGNS는 미국 중심의 편향을 발견하고, AI 모델의 비언어적 커뮤니케이션 해석에서 나타나는 한계를 드러냅니다. 기존의 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)은 문화적 맥락을 인식하지 못하고 미국에서 유래한 해석에 의존하며, 많은 제스처를 공격적으로 잘못 플래그(flag)합니다. 데이터셋은 공격적 제스처 감지와 비언어적 커뮤니케이션의 문화적 적합성을 평가하기 위한 테스트 베드(test bed)로 사용됩니다.

- **Performance Highlights**: MC-SIGNS 데이터셋을 활용한 평가 결과, text-to-image(T2I) 시스템은 공격적인 내용을 거의 거부하지 못하며, LLM은 제스처를 과도하게 공격적으로 표시하는 경향이 있음을 보여주었습니다. 연구 결과, AI 모델은 미국 중심의 해석을 자주 반복하며, 비 미국적 맥락에서 공격적인 제스처를 식별하는 정확도가 크게 낮아짐을 확인했습니다. 이는 AI 기술의 공평한 전 세계적 배포를 위해 더욱 포괄적이고 문화적으로 민감한 안전 메커니즘이 필요함을 강조합니다.



### SynthRAD2025 Grand Challenge dataset: generating synthetic CTs for radiotherapy (https://arxiv.org/abs/2502.17609)
Comments:
          22 pages, 8 tables, 4 figures; Under submission to Medical Physics, as dataset paper for the SynhtRAD2025 Grand Challenge this https URL

- **What's New**: SynthRAD2025 데이터셋은 합성 의료 이미징(synthetic imaging) 분야에서 중요한 발전을 이루고 있습니다. 이 데이터셋은 유럽의 여러 대학 병원에서 수집된 2362개의 사례를 포함하고 있으며, 합성 컴퓨터 단층촬영(synthetic computed tomography, sCT) 생성의 벤치마킹 플랫폼을 제공합니다. 이를 통해 알고리즘의 성능을 테스트하고 향상시키는 기회를 제공합니다.

- **Technical Details**: 데이터셋은 다양한 스캐너와 프로토콜을 사용하여 획득된 MRI-CT 및 CBCT-CT 쌍을 포함합니다. 이미지 전처리 과정에서는 강체 및 변형 가능한 이미지 정합(rigid and deformable image registration)을 통해 높은 품질의 이미지를 보장합니다. 모든 이미지는 MetaImage (.mha) 형식으로 제공되어 의료 이미지 처리 도구와의 호환성을 유지하며, 메타데이터는 구조화된 CSV 파일로 제공됩니다.

- **Performance Highlights**: SynthRAD2025 데이터셋은 훈련(65%), 검증(10%), 테스트(25%) 세트로 나누어 데이터셋의 무결성을 유지합니다. 이 데이터셋은 MRI 전용 sCT 생성, MR 유도 광자 및 양성자 치료, CBCT 기반의 용적 계산 등 다양한 방면에서 응용됩니다. 이는 개인 맞춤형 암 치료 및 적응 방사선 치료 발전에 크게 기여할 것입니다.



### Data-Driven Pseudo-spectral Full Waveform Inversion via Deep Neural Networks (https://arxiv.org/abs/2502.17608)
Comments:
          11 pages, 6 pages, review paper

- **What's New**: 이 연구는 Deep Learning(딥러닝) 방법을 활용하여 지진학적 FWI(Full Waveform Inversion)에 대한 새로운 접근 방식을 제시합니다. 기존의 시간 영역(time-domain) 접근 대신에 pseudo-spectral(유사 스펙트럼) 방법을 통합하여 데이터 기반(data-driven) DNN(Deep Neural Networks) 모델을 제안하고 있습니다. 이를 통해 기존 FWI 기법의 한계를 극복하고, 더 깊은 영역과 오버스러스트 지역에서 더욱 우수한 성능을 보여 줍니다.

- **Technical Details**: FWI에서는 파형 방정식으로부터 유도된 매개변수를 최적화하여 묘사하기 위해 multivariate optimization(다변량 최적화) 방법이 적용됩니다. 연구에서는 pseudo-spectral 방법을 Deep Learning 프레임워크에 통합하기 위해 이론적으로 FWI 문제를 재구성하고, 이는 synthetics data(합성 데이터)에서 평가되었습니다. 뉴럴 네트워크의 구조 및 학습 과정에서 가중치를 결정하기 위한 제곱합 오차(Sum of Squared Errors)처럼 일반적으로 사용되는 cost function 이 사용됩니다.

- **Performance Highlights**: 제안된 DNN 프레임워크는 기존의 결정론적 모델 및 시간 기반(time-based) 접근 방식과 비교하여 유의미한 성능 향상을 보여 주었습니다. 특히, 더 깊은 지층과 오버스러스트 구역에서 classical FWI보다 우수한 결과를 도출하여 DNN의 가능성을 증명합니다. 향후 연구 방향으로는 pseudo-spectral DNN 방법의 한계 분석 및 추가적인 발전 가능성이 논의됩니다.



### Laplace-Beltrami Operator for Gaussian Splatting (https://arxiv.org/abs/2502.17531)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS)를 활용한 기하학적 처리를 위한 새로운 방법을 제안합니다. 특히, Mahalanobis distance를 사용하여 Laplace-Beltrami operator (LBO)를 직접 계산하는 접근 방식을 소개합니다. 기존의 방법들이 포인트 클라우드를 사용하여 정보를 손실시키는 반면, 본 연구는 분산 정보를 활용하여 정확도를 높였습니다.

- **Technical Details**: 3DGS는 장면을 3D Gaussian 분포의 집합으로 표현하며, 이와 관련된 기하학적 처리 기술적인 요구가 증가하고 있습니다. 본 연구의 핵심은 LBO의 정의를 Gaussian Splatting에서 직접 계산하고 이 과정에서 분산 정보를 포함하여 더 나은 표면 방향 추정을 수행하는 것입니다. 이를 통해 출력의 품질을 최적화 과정에서 평가할 수 있는 방법이 제시됩니다.

- **Performance Highlights**: 실험 결과, 제안된 LBO는 전통적인 기하학적 처리 애플리케이션인 거리 계산 및 형태 매칭에서 높은 성능을 보였습니다. 또한, 3DGS에서도 안정적인 기하학적 특성을 나타내며, 연구자는 이를 통해 다양한 후속 연구를 위한 데이터셋을 공유할 예정입니다. 이러한 방식은 기하학적 질감이 복잡한 상황에서도 효과적으로 사용될 수 있는 가능성을 보여줍니다.



### On Neural Inertial Classification Networks for Pedestrian Activity Recognition (https://arxiv.org/abs/2502.17520)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2501.01327

- **What's New**: 이번 논문은 보행자 행위 인식을 위한 신경 관성 분류 네트워크의 개선을 위해 10가지 데이터 기반 기법을 정의하고 분석했습니다. 심층 학습(Deep Learning) 기술의 발전을 통해 관성 센서의 인식 성능과 견고성이 크게 향상되었지만, 성과를 공정하게 비교하고 평가할 수 있는 공통 벤치마크가 부족했습니다. 이 연구는 네트워크 아키텍처, 데이터 증강, 데이터 전처리의 세 가지 측면에 중점을 두고 실험을 진행하였습니다.

- **Technical Details**: 우리는 네트워크 설계에 영감을 받아 CNN(Convolutional Neural Network) 층, Bi-LSTM(Bidirectional Long Short-Term Memory) 층, 완전 연결 층(FC)을 통합한 기본 네트워크 아키텍처를 사용했습니다. 입력 신호는 1차원 CNN을 통해 처리되며, 다양한 데이터 증강 기법을 적용하여 모델 성능을 개선했습니다. 네 가지 실제 기록된 데이터 세트에서 신경 네트워크 기술의 적용이 통합적으로 검토되었습니다.

- **Performance Highlights**: 회전(rotation) 및 다중 헤드 아키텍처(multi-head architecture)를 통한 데이터 증강이 관성 분류 네트워크의 성능 개선에 가장 일관된 효능을 보였으며, 이 연구는 신경 네트워크의 구현에 대한 실용적인 통찰력을 제공합니다. 신경 네트워크의 정확도를 지속적으로 개선할 수 있는 기법들을 평가하고, 각 기법이 특정 시나리오에서 효과적인지에 대한 통찰력을 제시합니다.



### Doctor-in-the-Loop: An Explainable, Multi-View Deep Learning Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer (https://arxiv.org/abs/2502.17503)
- **What's New**: 이번 연구에서는 비소세포 폐암(NSCLC) 환자의 병리학적 반응(predicted pathological response, pR) 예측의 정확성을 높이고 신뢰성을 확보하기 위해 'Doctor-in-the-Loop'라는 새로운 프레임워크를 제안합니다. 본 프레임워크는 전문적인 도메인 지식을 인공지능 기술에 통합하여, 임상적으로 중요한 해부학적 영역에 모델의 초점을 맞추며, 예측의 해석 가능성과 투명성을 향상시킵니다. 이 방법은 신경망의 훈련 과정에 내재적(explainable AI) 설명성을 부여하여, 예측의 신뢰성과 의료적 맥락의 중요성을 반영합니다.

- **Technical Details**: 연구에서 제안하는 'Doctor-in-the-Loop' 방법론은 다각적(multi-view) 접근 방식을 통해 광범위한 맥락에서 특정 병변(lesion) 세부사항으로 점진적으로 모델의 초점을 조정합니다. 이 과정에서 의료 전문가의 도메인 지식을 학습 과정에 통합하여 치료 예측의 핵심 요소를 더욱 명확히 할 수 있습니다. 또한, 신경망 훈련 시 임상적 인사이트를 반영함으로써 모델이 의료적 통찰과 밀접하게 연결될 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 NSCLC 환자 데이터를 활용하여 제안한 방법이 높은 예측 성능을 보임을 입증하였으며, 투명하고 정당한 결과를 제공합니다. 기존의 최신 연구와 비교했을 때, 'Doctor-in-the-Loop' 접근 방식은 더 높은 예측 정확도와 임상적 관련성을 달성하여, 향후 환자 맞춤형 치료법 개발에 기여할 것으로 기대됩니다. 이러한 결과는 임상적 해설이 가능한 인공지능의 발전에 중대한 이정표가 될 것입니다.



### Using Graph Convolutional Networks to Address fMRI Small Data Problems (https://arxiv.org/abs/2502.17489)
Comments:
          8 pages

- **What's New**: 이 연구는 그래프 신경망(graph neural networks)을 활용하여 소규모 데이터를 다루는 의학 이미징(medical imaging)의 예측 문제를 해결하고자 한다. 주로 치료 반응 예측(prognosis) 문제에 집중하여, 기능성 자기공명영상(fMRI) 데이터를 기반으로 치료에 대한 증상 개선을 예측하는 새로운 방법론을 제시한다. 기존의 패턴 인식 기법으로는 어려운 예측을 가능하게 하기 위하여, 환자의 뇌 활동 연결성 정보를 스펙트럼 표현(spectral representation)을 통해 효과적으로 전파하는 방식을 소개한다.

- **Technical Details**: 연구의 주요 기술적 초점은 t-fMRI 데이터의 작은 크기와 연결성 그래프(connected graphs) 정보의 복잡성을 극복하기 위해 설계된 심층 그래프 컨볼루션 학습 구조에 있다. 각 환자를 위한 새로운 표현 방식으로 스펙트럼 분해(spectral decomposition)를 활용하며, 이는 이전의 스펙트럼 분석과는 구별되는 접근법이다. GNN(그래프 신경망) 방법을 사용하여 기존의 일반 NN(신경망) 방법보다 더 나은 예측 성능을 가져오며, 이로 인해 72.2% ± 0.7%의 정확도를 달성했다.

- **Performance Highlights**: 실험 결과, 제안된 GNN 방법은 기존의 방법보다 약 12% 향상된 성능을 발휘함을 보였다. 데이터의 스무딩(smoothing) 효과를 통해 삼각 부등식(triangle inequality)의 위반을 줄임으로써 성능이 개선된 것으로 나타났으며, 이는 연결 데이터의 스펙트럼 임베딩(spectral embedding)을 더 잘 수행할 수 있게 함을 의미한다. 이러한 결과는 환자별 예측의 효율성을 높이고, 응용 가능한 치료 접근법을 제시하는 데 기여할 것이다.



### FantasyID: Face Knowledge Enhanced ID-Preserving Video Generation (https://arxiv.org/abs/2502.13995)
- **What's New**: 이 논문에서는 대규모 미리 훈련된 비디오 확산 모델을 기반으로 한 텍스트-비디오 생성(IPT2V)을 위한 새로운 조정 필요 없는 프레임워크인 FantasyID를 제안합니다. 이 모델은 3D 얼굴 기하학 프라이어를 도입하여 비디오 합성 과정에서 얼굴 구조의 일관성을 보장합니다. 또한, 단순한 복사-붙여넣기 기법을 방지하기 위해 다각도 얼굴 증강 전략을 활용하여 얼굴 표정과 머리 자세의 역동성을 증가시킵니다.

- **Technical Details**: FantasyID는 미리 훈련된 비디오 모델에 대한 입력 조건으로 다중 시점에서 추출된 얼굴 이미지를 사용합니다. 이를 통해 2D 얼굴 특성과 3D 구조 정보를 결합하여 얼굴 기술을 개선하고, 본 모델의 각 레이어에 적절한 신호를 주입하기 위한 레이어 인식 신호 주입 메커니즘을 도입합니다. 이러한 방법론을 통해 개별 ID 보존과 동작 역학을 균형 있게 모델링합니다.

- **Performance Highlights**: 실험 결과는 기존의 조정 필요 없는 IPT2V 방법들에 비해 FantasyID 모델의 우수성을 입증합니다. 이 모델은 높은 충실도의 인간 비디오를 생성하며, 시간 일관성과 ID 안정성을 동시에 유지하는 데 성공하여 개인화된 아바타 생성 또는 대화형 스토리텔링과 같은 실제 애플리케이션에 적용 가능성을 보여줍니다.



New uploads on arXiv(cs.AI)

### MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning (https://arxiv.org/abs/2502.18439)
- **What's New**: 이번 연구에서는 협동적인 멀티 에이전트 워크플로우를 구축하기 위해 MAPoRL(Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning)이라는 새로운 포스트 트레이닝 패러다임을 소개합니다. 대부분의 이전 연구는 기본 제공되는 LLM을 활용하여 협업 능력에 의존했으나, MAPoRL은 이를 넘어서 명시적으로 협력 행동을 유도합니다. 이 접근법은 여러 LLM이 독립적으로 응답을 생성하고 다중 턴 논의를 통해 최종 답변을 공동으로 개선하는 과정을 포함합니다.

- **Technical Details**: MAPoRL의 핵심 역량은 최종 답변과 논의 내용을 평가하는 MAPoRL verifier를 도입한 점입니다. 이 verifier는 답변의 정확성을 검증하고, 교정 및 설득적인 논의를 장려하기 위해 점수를 부여합니다. 이 점수는 공동 훈련(co-training) 보상으로 사용되며, 다중 에이전트 강화 학습(multi-agent RL)을 통해 최대화됩니다.

- **Performance Highlights**: 실험 결과, 개별 LLM을 단독으로 훈련하는 것이 효과적인 협업을 유도하기에는 충분하지 않다는 것을 확인했습니다. 반면에, 멀티 에이전트 공동 훈련은 다양한 벤치마크에서 협업 성능을 향상시킬 뿐만 아니라, 보지 못한 도메인에 대한 일반화 능력도 보여주었습니다.



### PyEvalAI: AI-assisted evaluation of Jupyter Notebooks for immediate personalized feedback (https://arxiv.org/abs/2502.18425)
- **What's New**: 제출된 학생 과제를 빠르게 채점하고 피드백을 제공하는 새로운 AI 지원 평가 시스템인 PyEvalAI를 도입했습니다. 이 시스템은 Jupyter 노트북을 자동으로 채점하며, 개인 정보 보호를 위해 로컬에서 호스팅된 언어 모델과 유닛 테스트를 결합하여 사용합니다. PyEvalAI는 무료이자 오픈소스로 제공되며, 과제 채점 과정에서 튜터들이 전권을 유지할 수 있도록 지원합니다.

- **Technical Details**: PyEvalAI는 Tornado 서버를 중심으로 구성되며, 사용자 인터페이스는 학생, 튜터 및 관리자용으로 나뉘어 있습니다. 학생들은 Jupyter 노트북 형식으로 과제를 제출할 수 있으며, 서버는 1-2분의 시간 내에 피드백과 점수를 제공합니다. 튜터들은 학생의 과제와 채점 결과를 종합적으로 확인할 수 있는 대시보드를 통해 원활한 피드백을 제공합니다.

- **Performance Highlights**: 사례 연구 결과, PyEvalAI는 피드백 속도와 채점 효율성을 향상시키는 데 효과적임을 입증했습니다. 학생들이 즉각적인 피드백을 받음으로써 학습 경험이 개선되었으며, 튜터들은 반복적인 채점 작업에서 벗어나 보다 효율적으로 수업을 진행할 수 있었습니다. 이 시스템의 도입을 통해 STEM 교육에서 AI의 활용 가능성이 더욱 확대되었습니다.



### The Gradient of Algebraic Model Counting (https://arxiv.org/abs/2502.18406)
Comments:
          Published at AAAI 2025

- **What's New**: 이 논문은 대수적 모델 카운팅(Algebraic Model Counting, AMC)의 관점을 활용하여 통계적 및 관계적 학습(statistical-relational learning)과 신경 유사 AI(neurosymbolic AI)에서의 학습 방법론을 통합하는 새로운 접근법을 제시합니다. 특히, 다양한 학습 알고리즘을 일반화된 그래디언트와 역전파(backpropagation) 기법을 통해 통합하여 좀 더 효율적인 메모리 사용과 더 빠른 학습 속도를 구현했습니다. 이러한 방식으로, 기존의 다양한 알고리즘보다 성능이 크게 향상된 알고리즘을 제공하며, 유연한 학습 도구를 제공합니다.

- **Technical Details**: 대수적 모델 카운팅은 논리 공식의 만족 가능성 문제를 세미링(semi-ring)으로 일반화한 것입니다. 이 연구에서는 그래디언트를 세미링에 따라 일반화하여 다양한 학습 알고리즘을 적용할 수 있는 새로운 대수적 그래디언트 도구를 제안합니다. 이 도구는 경량의 연산을 기반으로 하여 그래디언트 하강법(gradient descent), 기대 최대화(expectation-maximization), 엔트로피 극대화(entropy maximization) 등 여러 알고리즘을 구현할 수 있도록 합니다.

- **Performance Highlights**: 제안된 알고리즘인 대수적 역전파(algebraic backpropagation)는 기존 PyTorch와 Jax보다 몇 배 높은 성능을 보이며, 세미링 특성을 반영하여 더 나은 효율을 구현합니다. 특히, 기존의 알고리즘들이 가진 문제점들을 해결하기 위한 특화된 접근 방식을 제시하며, 실험 결과에서도 유의미한 성능 향상이 확인되었습니다. 이 연구는 또한 2차 정보에 기반한 학습 알고리즘이 어떻게 보다 효율적으로 구현될 수 있는지를 탐구하고 있습니다.



### How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities (https://arxiv.org/abs/2502.18387)
Comments:
          31 pages, 9 figures, 18 tables

- **What's New**: 이 논문은 탐색(search)과 대형 언어 모델(LLMs) 간의 상호 보완적 관계를 세 가지 관점에서 체계적으로 탐구합니다. 제안된 SeaL(Searching via Learning) 프레임워크는 LLM을 활용하여 효율적인 탐색을 가능하게 하며, SeaL-C는 효율성을 유지하면서 탐색의 완전성을 보장합니다. 세 가지 실세계 계획 작업에 대한 평가를 통해 SeaL은 전통적인 방법과 비교하여 탐색 공간을 최대 99.1%까지 줄이며 거의 완벽한 정확도를 달성합니다.

- **Technical Details**: 논문은 학습과 탐색이 문제 해결에서 어떻게 상호 작용하는지를 분석합니다. SeaL 프레임워크는 학습을 통합하여 탐색 프레임워크의 효율성을 높이며, SeaL-C는 두 단계의 랭킹과 완전 상태 분해를 통한 체계적인 탐색을 보장합니다. 연구는 Game of 24, 미니 크로스워드, Blocksworld에서 SeaL과 SeaL-C를 평가하여 그들의 효과성을 입증합니다.

- **Performance Highlights**: 실험 결과, SeaL은 거의 모든 설정에서 거의 완벽한 통과율을 기록하며, 전통적인 폭력 검색과 비교해 탐색 공간을 최대 99.1% 줄였습니다. SeaL-C 또한 효과적으로 완전성을 보장함으로써 SeaL의 효율성을 подтверж합니다. 이 결과는 LLM이 실제 문제 해결에서 더 효과적으로 사용되기 위해 탐색 능력을 개선해야 함을 시사합니다.



### MindMem: Multimodal for Predicting Advertisement Memorability Using LLMs and Deep Learning (https://arxiv.org/abs/2502.18371)
Comments:
          7 pages, 5 figures, 4 Tables, AAAI 2025 Economics of Modern ML: Markets, Incentives, and Generative AI Workshop

- **What's New**: 이번 논문에서는 MindMem이라는 다중 모달 프레임워크를 소개하며, 광고 기억력을 예측하는 데 기여합니다. 이 모델은 텍스트, 시각, 청각 데이터를 통합하여 광고 메시지를 최적화하는 MindMem-ReAd (MindMem-Driven Re-generated Advertisement) 기능을 통해 광고 기억력을 향상시킵니다. 이러한 접근 방식은 소비자 행동 모델링과 광고 최적화를 위해 인공지능(AI)의 변혁적인 잠재력을 강조합니다.

- **Technical Details**: 연구에서 사용된 데이터 세트는 Long-term Ad MemoraBility DAtaset (LAMBDA)와 Memento10K로, 각각 다양한 광고와 일반 비디오의 기억력 분석을 위한 보완적인 설정을 제공합니다. 두 데이터 세트 모두 광고 메시지의 다양한 특성, 예를 들어 장면 복잡성, 감정적 내용 등을 포괄적으로 평가하여 모델을 훈련하고 테스트하는 데 사용되었습니다. MindMem은 사전 훈련된 LLMs를 인지 모듈로 사용하여 비주얼, 오디오 및 텍스트의 표현을 통합하여 기억력을 예측합니다.

- **Performance Highlights**: MindMem은 LAMBDA 데이터 세트에서 Spearman의 상관계수 0.631, Memento10K에서 0.731에 도달하며 기존 방법들을 지속적으로 초월하는 성능을 보입니다. 추가적으로 광고 기억력에 긍정적인 영향을 미치는 요소들, 예를 들어 비디오의 속도와 감정적 공명 등이 밝혀졌습니다. MindMem-ReAd를 통해 최대 74.12%까지 광고 기억력을 향상시킬 수 있는 것으로 나타났습니다.



### GraphRank Pro+: Advancing Talent Analytics Through Knowledge Graphs and Sentiment-Enhanced Skill Profiling (https://arxiv.org/abs/2502.18315)
- **What's New**: 이 연구는 반구조화된 텍스트에서 정보 추출의 새로운 접근 방식을 제안합니다. Graph(그래프), Natural Language Processing (자연어 처리), 그리고 딥러닝(Deep Learning)을 활용하여, 기존의 복잡한 로직을 Graph 구조로 추상화하여 원시 데이터를 포괄적인 Knowledge Graph(지식 그래프)로 변환합니다. 이러한 혁신적인 프레임워크는 정교한 정보 추출 및 쿼리 작업을 가능하게 합니다.

- **Technical Details**: 우리의 방법론은 이력서에서 개인 프로필, 기술, 조직 소속 및 프로젝트 경험과 같은 세부 정보를 추출하여 해당 정보를 가중치가 있는 그래프 구조로 인코딩합니다. Skill-Project Edge Weighting(기술-프로젝트 엣지 가중치) 기법을 통해 개인의 전문 지식을 식별하고, 그래프 쿼리 기능을 통해 사용자가 효과적으로 후보자를 선별할 수 있도록 지원합니다.

- **Performance Highlights**: 이 시스템은 구직자에게는 타겟 쿼리 기반 필터링 및 정렬 기능을 제공하며, 채용담당자와 커리큘럼 디자이너에게도 혜택을 줍니다. 따라서, 다양한 이력서의 양에 압도되지 않고 정밀한 검색이 가능하게 되어, 인재 분석 및 채용 과정에 대한 혁신을 이룰 수 있습니다.



### Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Suppor (https://arxiv.org/abs/2502.18274)
- **What's New**: 이 논문에서는 Citrus라는 의료 언어 모델을 소개합니다. 이 모델은 의료 전문가의 인지 과정을 모방하여 인공지능(AI) 추론과 임상 전문 지식 간의 간극을 해소합니다. Citrus는 전문가 수준의 질병 추론 데이터를 시뮬레이션하여 훈련되었으며, 이는 임상의 결정 경로를 정확하게 포착하는 혁신적인 접근 방식을 통해 수행되었습니다.

- **Technical Details**: Citrus는 대규모의 시뮬레이션된 전문가 질병 추론 데이터의 코퍼스에 대해 훈련됩니다. 이 모델은 복잡한 진단 및 치료 과정의 추론을 시뮬레이션하는 데 중점을 두고 있으며, 마지막 단계 훈련 데이터를 공개하여 의료 추론 작업을 위한 데이터 부족 문제도 해결합니다. 새로운 설계(architecture)와 대화형 데이터셋을 통해, 검증 및 반복 가능한 결과를 제공하고자 합니다.

- **Performance Highlights**: Citrus는 MedQA와 같은 권위 있는 벤치마크를 통한 평가에서 우수한 성능을 보여주며, 유사한 크기의 다른 모델들보다 더 나은 결과를 나타냈습니다. 이러한 성과는 Citrus가 의료 결정 지원 시스템을 크게 향상시킬 잠재력을 지닌 것으로, 임상 결정을 위한 더 정확하고 효율적인 도구를 제공할 수 있음을 강조합니다.



### ChatMotion: A Multimodal Multi-Agent for Human Motion Analysis (https://arxiv.org/abs/2502.18180)
- **What's New**: 이번 논문에서는 인체 동작 분석을 위해 ChatMotion이라는 멀티모달 다중 에이전트 프레임워크를 소개하고 있습니다. ChatMotion은 사용자의 의도를 동적으로 해석하고, 복잡한 작업을 메타 태스크로 분해하며, 동작 이해를 위해 전문 기능 모듈을 활성화합니다. 이러한 접근 방식은 기존의 MLLMs의 제한 사항을 극복하고 다양한 분석 관점을 지원하는 데 초점을 맞추고 있습니다.

- **Technical Details**: ChatMotion은 MotionCore와 같은 여러 전문 모듈을 통합하여 인체 동작을 다양한 관점에서 분석합니다. 이 프레임워크는 사용자와 상호작용하며 태스크를 동적으로 조정할 수 있는 능력을 제공합니다. 이를 통해 인체 동작에 대한 보다 깊이 있는 이해를 가능하게 하며, 사용자 친화적인 경험을 제공합니다.

- **Performance Highlights**: 실험 결과, ChatMotion은 인체 동작 이해에서 높은 정확도와 적응성을 보였습니다. 사용자의 참여를 유도하는 데 효과적이며, 다양한 분석 요구를 충족할 수 있는 능력을 입증하였습니다. 이러한 성과는 ChatMotion이 MLLMs의 발전에 기여할 수 있는 잠재력을 보여줍니다.



### Defining bias in AI-systems: Biased models are fair models (https://arxiv.org/abs/2502.18060)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 논문에서는 AI 시스템에서의 편향(bias) 개념에 대해 명확한 정의가 부족하다는 점을 지적하며, '공정성(fairness)'과의 대비 속에서 빈번히 사용되고 있는 편향 개념을 재조명합니다. 저자들은 편향을 부정적 현상으로 간주하는 전통적인 관점에서 벗어나, 편향과 차별(discrimination)의 구별이 중요하다고 주장합니다. 이를 통해 AI 시스템에서의 공정성에 대한 논의가 더 건설적으로 발전할 수 있다고 강조합니다.

- **Technical Details**: 신경망(neural networks)의 기본 구성 요소인 인공 기계 뉴런은 McCulloch-Pitts 뉴런(MCP neuron)에서 기원하며, 이는 입력을 처리해 0 또는 1의 출력을 생성하는 간단한 이진 뉴런 모델입니다. 이 논문에서는 아이디어가 진화하면서도 '편향(bias)'이 모델의 결정 경계를 조정하는 데 어떻게 사용되는지 설명합니다. 특히 ADALINE(Adaptive Linear Neuron)을 통해 편향 개념이 처음 포함되었음을 강조하며, 이는 모델의 훈련 과정에서 오류를 최소화하는 데 기여합니다.

- **Performance Highlights**: 편향의 정의와 공정성 간의 관계를 명확히 하므로써, AI 모델의 개발에 있어 실질적인 공정성을 추구하는 데 중요한 기여를 합니다. 예를 들어, 얼굴 인식 알고리즘이 특정 인종 데이터를 기반으로 훈련되면, 공정성에 부합하지 않는 것은 물론 특정 집단에 대한 성능 저하를 초래할 수 있습니다. 이와 같은 문제를 해결하기 위해, 대표성이 있는 훈련 데이터 사용이 필수적이며, 이는 편향 없는 모델로 이어질 수 있음을 제시합니다.



### GNN-XAR: A Graph Neural Network for Explainable Activity Recognition in Smart Homes (https://arxiv.org/abs/2502.17999)
Comments:
          This is a preprint. Paper accepted for publication at the 21st EAI International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (Mobiquitous)

- **What's New**: 이번 연구는 스마트 홈 환경에서 ADLs(Activities of Daily Living) 인식을 위해 명확히 설계된 최초의 설명 가능한 그래프 신경망 시스템(GNN-XAR)을 소개합니다. GNN-XAR는 환경 센서 데이터의 시간 창에서 동적으로 그래프를 구성하여 공간적(spatial) 및 시간적(temporal) 특성을 고려합니다. 이 시스템은 각 예측에 대해 자연어 설명을 생성하며, 기존의 최첨단 방법들보다 우수한 설명을 제공합니다.

- **Technical Details**: GNN-XAR는 이진 환경 센서 데이터를 사용하여 공휴일(이벤트)와 그 상호작용의 맥락에서 활동을 인식합니다. 이 시스템은 그래프 구축 모듈을 통해 시간 창을 그래프 형태로 변환하며, 각 그래프는 그래프 신경망 모듈을 통해 분석됩니다. 설명 생성 모듈은 후속 XAI(설명 가능한 AI) 방법을 활용하여 노드와 간선의 중요성을 분석하고 이를 바탕으로 비전문 사용자가 이해할 수 있는 자연어 설명을 제공합니다.

- **Performance Highlights**: GNN-XAR는 두 개의 공개 데이터 세트에서 테스트되었으며, 기존의 설명 가능한 HAR 방법들보다 더 유용한 설명을 제공하면서도 인식률이 소폭 향상된 결과를 나타내었습니다. 이를 통해 스마트 홈 환경 내에서 인간 활동 인식의 신뢰성 및 투명성을 증대시킬 수 있는 가능성이 확인되었습니다.



### LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction (https://arxiv.org/abs/2502.17925)
- **What's New**: 이 논문은 LeanProofs에서 자동 정리 증명 과정의 진행 상황을 예측하는 방법인 LeanProgress를 소개합니다. 이를 통해 긴 증명이나 복잡한 수학적 공식화를 다룰 때의 한계를 극복하고자 합니다. LeanProgress는 남은 증명 단계 수를 예측하여 사용자가 증명 전략을 더욱 효과적으로 결정할 수 있도록 돕습니다.

- **Technical Details**: LeanProgress는 약 8만 개의 Lean 증명 경로로부터 생성된 균형 잡힌 데이터를 기반으로 학습합니다. 증명 단계의 예측을 수행하기 위해 DeepSeek Coder V1 1.3b 모델을 미세 조정하여 Mean Absolute Error (MAE) 3.29, 예측 정확도 75.1%를 달성했습니다. 이 모델은 베스트-퍼스트 검색 기법과 통합되어 Mathlib4에서 41.2%의 기본 성능에 비해 3.8%의 성능 개선을 보였습니다.

- **Performance Highlights**: LeanProgress는 선택한 최선의 검색 방법을 통해 증명 과정의 단계 수를 예측하여 사용자에게 즉각적인 피드백을 제공합니다. 이를 통해 증명 과정을 더 잘 이해하고, 긴 증명에서도 효율적으로 전략을 선택할 수 있게 됩니다. 이 연구 결과는 자동화된 정리 증명 및 대화형 정리 증명에서 중요한 진전을 보여줍니다.



### Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness (https://arxiv.org/abs/2502.17921)
- **What's New**: 이번 논문에서는 추천 시스템의 성별 편향을 정량화하기 위한 포괄적인 메트릭스를 새롭게 도입했습니다. 특히, 추천 항목의 범주를 고려하여 세분화된 공정성 평가의 중요성을 강조합니다. 기존의 성별 관련 편향 평가가 놓치고 있는 뉘앙스를 포착할 수 있는 방법에 대해 다루고 있습니다. 또한, Catagory-aware fairness metric을 훈련 시 정규화 항으로 사용하여 모델의 출력에서 편향을 효과적으로 최소화할 수 있음을 보여줍니다.

- **Technical Details**: 추천 시스템은 사용자 개개인의 선호도와 행동을 바탕으로 아이템을 개인화하여 추천합니다. 본 논문은 RMSE, NDCG, precision 등의 기존 성능 평가 지표만으로는 불충분하다는 점을 지적하며, 세분화된 평가 메트릭스의 필요성을 강조합니다. 성별 편향을 평가하기 위해 추천된 항목의 범주와 순위를 고려하는 방안을 제안하며, 이 메트릭스를 손실 함수의 일환으로 활용하여 공정성을 극대화합니다.

- **Performance Highlights**: 실제 데이터세트에 대한 실험을 통해 제안된 메트릭스의 효과성을 입증했습니다. 일반적인 추천 성능의 저하 없이, 다양한 범주에서 추천의 공정성을 크게 향상시켰습니다. 성별 관련 추천의 불균형 문제를 해결하는 데 도움을 주며, 추천 모델에 대한 편향 분석에 있어 향상된 통찰력을 제공합니다.



### Towards Sustainable Web Agents: A Plea for Transparency and Dedicated Metrics for Energy Consumption (https://arxiv.org/abs/2502.17903)
- **What's New**: 최근 대형 언어 모델(LLM)의 발전은 외부 도구를 사용하고 그 출력을 해석할 수 있는 모델 구축으로 전환되고 있습니다. 이러한 웹 에이전트(Web Agents)는 인터넷과 자율적으로 상호작용할 수 있는 능력을 가지고 있어, 사용자들의 일상적인 활동을 지원하며 시간 소모적인 반복 작업을 처리할 수 있는 강력한 도우미 역할을 합니다. 하지만 웹 에이전트 연구의 지속가능성 측면은 여전히 많이 다루어지지 않았습니다.

- **Technical Details**: 연구에서는 LASER와 MindAct라는 두 가지 웹 에이전트의 에너지 소비를 비교했습니다. LASER는 사용자가 요청한 제품을 자동으로 찾고 구매하는 기능을 가진 시스템이며, 사용 시 약 2930Wh의 에너지를 소모합니다. 반면 MindAct는 오픈 소스 LLM을 활용하여 에너지 소비를 대폭 줄일 수 있는 접근 방식을 사용하고 있으며, LASER에 비해 약 1500배 적은 에너지를 소모하는 것으로 추정되었습니다.

- **Performance Highlights**: 이 논문은 웹 에이전트의 성능을 평가하는 새로운 접근 방식과 함께 에너지 소비 및 환경적 영향을 고려해야 함을 제안하고 있습니다. 웹 에이전트의 설계 철학에 따른 에너지 소비 차이를 강조하며, 향후 새로운 웹 에이전트를 출시할 때는 모델 매개변수와 프로세스의 투명성을 확보할 것을 주문합니다. 이러한 평가 방식을 통해 웹 에이전트가 환경에 미치는 영향을 보다 쉽게 비교할 수 있는 메트릭스를 도입하자는 주장을 합니다.



### Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers (https://arxiv.org/abs/2502.17882)
- **What's New**: 과학 연구는 본질적으로 글로벌하지만, 대부분의 학술지는 영어로만 발행되어 비영어 원어민 연구자에게 장벽이 되고 있습니다. 이번 연구에서는 대형 언어 모델(LLMs)을 사용하여 JATS XML 형식을 유지한 채로 과학 기사를 번역하는 자동화된 방법을 개발하였습니다. 이로 인해 28개 언어로 다수의 과학 기사를 번역할 수 있게 되었으며, 번역 결과의 정확성을 평가하기 위한 새로운 질문-답변(QA) 벤치마킹 방법도 도입했습니다.

- **Technical Details**: 98%의 동료 심사 과학 기사가 영어로 발행되는 상황에서, LLM 기반의 번역 솔루션을 통해 언어 장벽을 낮추기 위한 방법이 제시됩니다. 자동 번역 프로세스에서는 JATS XML 포맷을 유지하며 원본 및 번역된 문서를 기반으로 번역 품질을 평가하는 방법을 사용합니다. LLM은 전통적인 기계 번역 시스템보다 성능이 우수하며, 특정 과학 커뮤니티의 요구에 맞춰 번역을 조정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 새로운 벤치마크 결과에 따르면, 번역된 기사의 평균 성능은 95.9%로, 주요 과학 정보가 정확히 전달되고 있음을 보여줍니다. 사용자 연구에서는 15명의 연구자가 자신의 기사를 번역했을 때 원본 정보를 잘 표현했다고 일관되게 평가했습니다. 흥미롭게도, 3분의 1의 저자는 많은 기술 용어가 '과도하게 번역'되었다고 느끼며, 영어 용어를 그대로 유지하길 선호한다고 밝혔습니다.



### A Combinatorial Identities Benchmark for Theorem Proving via Automated Theorem Generation (https://arxiv.org/abs/2502.17840)
- **What's New**: 이 논문에서는 LeanComb이라는 조합적 정체성의 벤치마크를 소개합니다. 이는 조합론에 대한 첫 번째 공식화된 정리 증명 벤치마크로, 418개의 정체성과 209개의 보조정리가 포함되어 있으며, 100개의 정체성을 테스트 세트로 가지고 있습니다. 또한, ATG4CI라는 자동 정리 생성기를 개발하여, 자가 개선 대형 언어 모델과 강화 학습 트리 탐색(Reinforcement Learning Tree Search) 방식을 결합하여 새로운 정리 및 그 증명을 생성합니다.

- **Technical Details**: LeanComb은 Lean 프로그램 언어를 바탕으로 수작업으로 공식화된 벤치마크로, 고전 및 현대 정체성의 폭넓은 범위를 다루며 조합적 정체성의 성능을 평가할 수 있습니다. Lean은 Microsoft Research에 의해 개발된 상호작용 정리 증명기이며, Mathlib4라는 풍부하게 유지되는 수학 라이브러리로 지원됩니다. LeanComb은 9898단계 이상을 포함하며, 조합론, 미적분학, 확률론 등 8개의 분류로 정리들을 나누고 있습니다.

- **Performance Highlights**: LeanComb-Enhanced 데이터를 이용하여 훈련된 모델은 자동 증명 성공률을 26%까지 높였습니다. 이 실험을 통해 각 모델이 기준 모델에 비해 17% 이상 성공률을 향상시키는 결과를 보였습니다. LeanComb 벤치마크와 LeanComb-Enhanced 데이터 세트를 통해 조합적 정체성에 대한 자동 정리 증명 시스템의 성능이 크게 개선되었음을 확인할 수 있습니다.



### DocPuzzle: A Process-Aware Benchmark for Evaluating Realistic Long-Context Reasoning Capabilities (https://arxiv.org/abs/2502.17807)
- **What's New**: DocPuzzle는 대형 언어 모델(LLMs)의 장기 맥락 reasoning 능력을 평가하기 위해 설계된 새로운 벤치마크입니다. 이 벤치마크는 다단계 reasoning이 필요한 100개의 전문가 수준의 QA 문제로 구성되어 있으며, 실제 문서에서 다루어집니다. DocPuzzle은 human-AI 협업 방식으로 품질과 난이도를 보장하며 추측 편향을 줄이기 위한 혁신적인 평가 프레임워크를 도입합니다.

- **Technical Details**: DocPuzzle은 다섯 가지 다양한 도메인에서 오는 실제 문서를 기반으로 하여 설계되었습니다. 평가 과정에서는 문서, 질문, 답변, 체크리스트가 포함되어 있어 reasoning 프로세스의 정확성을 확인할 수 있습니다. 이 문서에서는 다단계 reasoning 작업을 요구하는 질문과 함께 확인 절차가 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, o1-preview와 DeepSeek-R1같은 고급 reasoning 모델들이 Claude 3.5 Sonnet과 같은 일반 instruct 모델을 크게 초월하는 성능을 보였습니다. 또한, distillation 모델인 DeepSeek-R1-Distill-Qwen-32B는 교사 모델에 비해 상당히 낮은 성능을 보였으며, 이는 reasoning 능력의 일반화 유지에 있어 있는 도전 과제를 시사합니다.



### Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features (https://arxiv.org/abs/2502.17749)
- **What's New**: 이 연구는 코드 생성을 위한 대형 언어 모델(LLMs)이 독립적인 지식 재산 보호를 위협할 수 있다는 점을 강조합니다. 특히, LLM은 인간이 작성한 코드의 패러프레이즈 버전을 생성할 수 있는 잠재력을 지니고 있습니다. 이에 따라, LLM이 생성한 코드가 기존 코드의 패러프레이즈인지 감지하고, 어떤 LLM이 이를 수행했는지를 식별하기 위한 두 가지 과제를 제안합니다.

- **Technical Details**: 연구에서는 LPcode 데이터셋을 구축하여 인간이 작성한 코드와 LLM으로 패러프레이즈된 코드 쌍을 수집하고, 이를 통해 코딩 스타일의 중요한 차이점을 통계적으로 검증합니다. 감지 방법 LPcodedec을 개발하여, 코드의 네이밍 일관성, 코드 구조, 가독성 등의 세 가지 주요 측면에서 코딩 스타일 피처를 측정하였습니다. 이를 통해 인간과 LLM의 서로 다른 코딩 스타일의 특성을 잘 포착하고, 어떤 LLM이 패러프레이즈를 수행했는지를 분류할 수 있습니다.

- **Performance Highlights**: LPcodedec은 두 가지 태스크에서 기존의 최고 성능 기준을 초과 달성하며, F1 점수를 각각 2.64% 및 15.17% 향상시켰습니다. 또한, 속도는 각각 1,343배 및 213배 개선되어, 패러프레이즈 감지의 효율성을 크게 높였습니다. 이 결과는 LLM 기반 패러프레이즈 탐지 분야에 대한 귀중한 기여로 평가됩니다.



### Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures (https://arxiv.org/abs/2502.17710)
Comments:
          40 pages, 49 figures

- **What's New**: 이번 연구에서는 다양한 문화적 해석을 포함하는 새로운 데이터셋인 Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS)를 소개합니다. 이 데이터셋은 25가지 제스처와 85개 국가를 포함하여 총 288개의 제스처-국가 쌍을 다루고 있으며, 각 제스처의 공격성, 문화적 중요성 및 맥락적 요소에 대한 주석이 포함되어 있습니다. 이는 AI 시스템이 비문화적 콘텐츠를 생성할 위험을 줄이기 위한 필수적인 노력을 보여줍니다.

- **Technical Details**: MC-SIGNS는 미국 중심의 편향을 발견하고, AI 모델의 비언어적 커뮤니케이션 해석에서 나타나는 한계를 드러냅니다. 기존의 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)은 문화적 맥락을 인식하지 못하고 미국에서 유래한 해석에 의존하며, 많은 제스처를 공격적으로 잘못 플래그(flag)합니다. 데이터셋은 공격적 제스처 감지와 비언어적 커뮤니케이션의 문화적 적합성을 평가하기 위한 테스트 베드(test bed)로 사용됩니다.

- **Performance Highlights**: MC-SIGNS 데이터셋을 활용한 평가 결과, text-to-image(T2I) 시스템은 공격적인 내용을 거의 거부하지 못하며, LLM은 제스처를 과도하게 공격적으로 표시하는 경향이 있음을 보여주었습니다. 연구 결과, AI 모델은 미국 중심의 해석을 자주 반복하며, 비 미국적 맥락에서 공격적인 제스처를 식별하는 정확도가 크게 낮아짐을 확인했습니다. 이는 AI 기술의 공평한 전 세계적 배포를 위해 더욱 포괄적이고 문화적으로 민감한 안전 메커니즘이 필요함을 강조합니다.



### From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs (https://arxiv.org/abs/2502.17701)
Comments:
          24 pages, 9 figures

- **What's New**: 이 논문에서는 FLARE라는 새로운 프레임워크를 도입하여 대형 언어 모델(LLM)을 활용한 산불 대피 결정 예측을 혁신적으로 개선하고 있습니다. FLARE는 인간의 복잡한 행동 논리를 보다 잘 이해할 수 있도록 심리학 및 행동 이론을 통합하여 Chain-of-Thought (CoT) 추론을 간소화하고 메모리 기반 강화 학습(RL) 모듈과 통합합니다. 기존 LLM의 한계를 극복하여 보다 정확한 대피 결정 예측을 가능하게 하며, 실험 결과 평균 20.47%의 성능 향상을 보여 주었습니다.

- **Technical Details**: FLARE 프레임워크는 위험 인식(risk perception) 및 위협 평가(threat assessment)를 주요 개념으로 사용하여 개인의 정신 상태를 나타냅니다. PADM(Protective Action Decision Model) 기반의 분류기를 사용하여 역사적 데이터와 경험적 행동 연구를 통해 가장 관련성이 높은 입력 변수를 선택하고, 이후 LLM이 선택된 추론 패턴으로부터 인식을 유추하여 점수를 부여합니다. 또한 이 시스템은 오류 기록과 자기 반성 메커니즘을 통합하여 모델의 추론 과정을 개선하는 동시에 개인의 대피 행동을 맞춤화합니다.

- **Performance Highlights**: FLARE는 자주 사용되는 기존 이론 기반 모델 대비 20.47%의 성능 향상을 기록하며, 강력한 교차 사건 일반화(cross-event generalizability)를 보여주고 있습니다. 이 프레임워크는 제한된 데이터와 불균형한 데이터셋에서도 효과적으로 작동하여 실제 대피 행동을 보다 잘 반영합니다. 또한, 독립적 실험을 통해 FLARE의 이유 기반 추론 능력을 입증하며, 대피 결정 예측에서 새로운 기준을 제시하고 있습니다.



### Socratic: Enhancing Human Teamwork via AI-enabled Coaching (https://arxiv.org/abs/2502.17643)
Comments:
          Extended version of an identically-titled paper accepted at AAMAS 2025

- **What's New**: Socratic라는 새로운 AI 시스템은 인력 자원이 제한된 상황에서도 실시간으로 팀에게 지침을 제공하는 혁신적인 기능을 갖추고 있습니다. 이는 특히 의료, 재난 대응과 같은 생명 및 안전이 중요한 분야에서 팀워크를 개선하는 데 도움을 줄 수 있습니다. 연구 결과, Socratic는 팀의 성과를 최소한의 개입으로 크게 향상시킬 수 있음이 입증되었습니다.

- **Technical Details**: Socratic는 다중 에이전트 시스템(multi-agent systems)과 모방 학습(imitation learning)의 최신 발전을 활용하여 팀 행동을 모델링합니다. 이 시스템은 팀 수행 중의 불일치를 감지하고 팀 구성원들에게 재조정을 권장하여 협업 결정을 개선합니다. 또한, Dec-POMDPs(Decentralized partially observable Markov decision processes)를 통해 고정된 임무와 시간 제약을 가진 협업 작업을 모델링하는 데 필요한 수학적 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, Socratic는 실제 팀 작업에서 성과를 크게 향상시키는 것으로 나타났습니다. 참가자들은 비록 최소한의 개입으로 이루어졌지만, Socratic이 팀워크 개선에 유용하고 신뢰할 수 있는 도구로 인식하였습니다. 이러한 분석은 AI 연구와 실제 적용 가능성에 대한 희망적인 방향을 제시합니다.



### Representation Engineering for Large-Language Models: Survey and Research Challenges (https://arxiv.org/abs/2502.17601)
- **What's New**: 이 논문은 Representation Engineering(표현 엔지니어링)이라는 새로운 접근 방식을 통해 대형 언어 모델(LLMs)의 예측 가능성과 제어 가능성을 증대시키기 위한 방법론을 제시합니다. 기존의 기계적 해석 가능성(mechanistic interpretability), 프롬프트 엔지니어링(prompt-engineering), 파인튜닝(fine-tuning)과 비교하여, 표현 엔지니어링의 목표와 방법을 체계적으로 설명합니다. 고수준 개념의 감지 및 수정을 위한 대조 입력 샘플 활용의 필요성을 강조하며, 이는 LLM의 성능을 향상시킬 수 있는 잠재력을 가집니다.

- **Technical Details**: Representation Engineering은 크게 두 부분으로 나뉘며, 첫째는 Representation Reading(표현 읽기)으로, 네트워크의 잠재 공간에서 개념을 감지하고 추출하는 것입니다. 둘째는 Representation Control(표현 제어)로, 내부 표현을 수정하여 모델 출력을 조정하는 것을 목표로 합니다. 이 과정에서는 대조 쌍 입력을 활용하여 개념 간의 차이를 식별하고, 관련된 특징을 추출하는 데 중점을 두고 있습니다. 이는 모델의 복잡성과 대규모 파라미터의 문제를 해결할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: LLMs는 분산된 과제들에서 인간과 이전의 좁은 AI 시스템을 초월하는 성능을 보여주었습니다. 의료, 소프트웨어 개발, 과학 연구, 교육, 법률 및 금융 분야에서 복잡한 문제 해결, 프로그래밍 개념 이해 및 개인 맞춤형 튜터링 기능을 제공하는 등의 뛰어난 성능을 입증했습니다. 또한, LLMs는 다양한 도메인 간 지식 전이를 통해 새로운 문제를 다룰 수 있는 잠재력을 보이지만, 사실적 정확성과 신뢰성 보장에는 여전히 도전 과제가 존재합니다.



### Intention Recognition in Real-Time Interactive Navigation Maps (https://arxiv.org/abs/2502.17581)
- **What's New**: 이번 연구에서는 사용자의 의도를 인식하는 시스템 IntentRec4Maps를 개발했습니다. 이 시스템은 Google Maps Platform을 활용하여 실시간 내비게이션을 제공합니다. 특히, 실시간 의도 인식 기능을 실험하고 시연하기 위해 두 가지 경로 계획(Path-Planner)과 대규모 언어 모델(LLM)을 사용했습니다.

- **Technical Details**: IntentRec4Maps는 두 개의 주요 구성 요소로 구성됩니다: 실시간 의도 인식을 위한 Interactive Map Platform과 실시간 의도 인식 기능입니다. 이 시스템은 도로 네트워크를 기반으로 하며, 사용자의 위치를 위도와 경도로 표현하여 다양한 이동 경로를 분석합니다. IntentRec4Maps는 Mirroring이라는 온라인 인식 접근 방식을 활용하여 사용자의 의도를 실시간으로 인식합니다.

- **Performance Highlights**: 우리는 IntentRec4Maps의 효율성을 다양한 의도와 관측 문제를 통해 평가했습니다. 시스템은 두 가지 다른 경로 계획 도구를 사용하여 복잡한 인식 문제를 해결하는 능력을 보여주었습니다. 여러 예제를 통해 이 시스템의 효용성을 비디오로 시연했습니다.



### How Do Large Language Monkeys Get Their Power (Laws)? (https://arxiv.org/abs/2502.17578)
- **What's New**: 이 논문은 언어 모델의 다수의 시도에 대한 성공률을 분석하여, 각 문제의 실패율이 시도 횟수에 따라 지수적으로 감소해야 한다는 예측을 제시합니다. 그러나 연구 결과는 반대로, 전체적인 성공률이 다항식(power law)으로 변화함을 보여주며, 이는 개별 문제의 성공률이 헤비 테일(distributional perspective) 형태로 분포하여 나타나는 결과입니다.

- **Technical Details**: 연구진은 언어 모델이 각 문제에 대해 다수의 시도를 할 때, 개별적 성공 확률이 비대칭적인 분포를 가질 수 있음을 강조합니다. 이는 소수의 문제에서 극단적으로 낮은 성공 확률이 전체 성공 트렌드를 왜곡하게 만들어 다항식 형태의 분포를 생성하는데 기여합니다. 이 접근법은 또한 기존의 연구에서 관찰된 다항식 스케일링에서의 편차를 설명하게 됩니다.

- **Performance Highlights**: 실험 결과, 새로운 분포 관점을 통해 예측된 파워 로우(exponent)에 대한 상대적 오차가 크게 줄어들었으며, 이는 추론 계산을 약 2-4 배 낮출 수 있는 방법으로 나타났습니다. 결과적으로, 본 연구는 신경 언어 모델 성능의 향상과 이와 관련된 평가 발전에 기여할 수 있는 중요한 통찰을 제공합니다.



### Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction (https://arxiv.org/abs/2502.17541)
- **What's New**: 이 연구에서는 데이터셋 기능 추출에 대한 새로운 접근 방식을 제안합니다. 기존의 단순한 기능 추출 방법들이 다양한 데이터셋에 대한 정확하고 다재다능한 설명을 생성하는 데 실패하는 반면, 이 방법은 신뢰성과 세밀한 제어를 제공합니다. 또한, 이는 전문가의 레이블링과 비교할 수 있는 압축되고 설명적인 표현을 유지하므로, 다양한 분야에 적용할 수 있습니다.

- **Technical Details**: 제안된 방법은 언어 모델(LLM)의 능력을 활용하여 데이터셋의 구조와 의미를 추출하는 이진 특징 집합을 최적화합니다. 이 방법은 무감독 자동 파이프라인을 통해 가능한 기능 집합을 생성하고, 데이터셋의 구조를 가장 잘 포착하는 서브셋을 추출합니다. 제공된 문맥에 따라 언어 모델이 원본 데이터를 재구성할 수 있는 능력을 평가하여 정보를 선택적으로 수집하는 방식으로 작동합니다.

- **Performance Highlights**: 이 연구는 두 가지 사례 연구를 통해 방법의 효과를 입증합니다. 첫 번째로, LLM jailbreak 전술에 대한 특징을 추출하여 인간이 제작한 전술들의 효율성과 다양성을 컴팩트하게 캡처했습니다. 두 번째로, 자동화된 피쳐 발견 방법을 통해 인간의 선호도와 일치하거나 그 성능을 초과하는 정확한 모델을 생성하는 데 성공하였습니다.



### User Intent to Use DeekSeep for Healthcare Purposes and their Trust in the Large Language Model: Multinational Survey Study (https://arxiv.org/abs/2502.17487)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 헬스케어의 상호작용 자원으로 점점 더 활용되고 있는 가운데, 사용자 수용에 대한 심층적인 탐구가 부족하다는 점을 강조합니다. 연구는 DeepSeek이라는 LLM 기반 플랫폼의 헬스케어 용도 채택 의도를 형성하는 주요 요인으로 사용의 용이성, 인지된 유용성, 신뢰, 위험 인식의 상호작용을 조사했습니다.

- **Technical Details**: 인도, 영국, 미국의 556명을 대상으로 한 단면 조사(cross-sectional survey)를 통해 인식과 사용 패턴을 측정하였습니다. 구조 방정식 모델링(structural equation modeling)을 사용하여 직접적 및 간접적 영향을 평가했고, 잠재적 비선형 관계를 분석했습니다. 결과는 신뢰가 중재 역할을 중심으로, 사용의 용이성이 신뢰를 통해 채택 의도에 간접적으로 큰 영향을 미친다는 것을 보여주었습니다.

- **Performance Highlights**: 연구는 신뢰 구축이 헬스케어 분야에서 LLM의 수용을 장려하기 위한 중요한 요소임을 강조했습니다. 또한 사용의 용이성과 위험 인식에 대한 비선형 경로가 관찰되어, 특정 임계점(threshold) 효과가 있음을 시사했습니다. 이러한 발견은 기술 수용 및 건강 정보학 연구에 기여하며, 사용자 수용의 복합적 성격을 조명합니다.



### Scalable Equilibrium Sampling with Sequential Boltzmann Generators (https://arxiv.org/abs/2502.18462)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Boltzmann generator 프레임워크를 확장한 Sequential Boltzmann Generators (SBG)를 소개합니다. SBG는 Cartesian 좌표에서의 sampling을 향상시키기 위해 Transformer 기반의 비등가(normalizing flows) 구조를 활용하여 샘플 생성 및 likelihood 계산의 효율성을 극대화합니다. 추가적으로, annealed Langevin dynamics를 통해 표적 분포에 맞췄을 때 샘플의 variance를 줄여 더 정밀하게 모델링할 수 있도록 합니다.

- **Technical Details**: SBG 접근법은 기존의 Boltzmann generators가 아쉬운 성능을 보였던 높은 에너지 장벽을 극복합니다. 새로운 구조는 exact invertibility를 제공하고, 이는 샘플링 효율성을 증가시키며, 특히 복잡한 에너지 분포에서 작동할 수 있도록 개선되었습니다. 제안된 SBG는 또한 Sequential Monte Carlo (SMC) 기술을 통해 추론 시점을 조정하여, 중요한 물리적 양, 예를 들어 자유 에너지 차이를 계산하는 데 도움을 줍니다.

- **Performance Highlights**: SBG는 모든 측면에서 최첨단 성능을 달성하였으며, 기존의 Boltzmann generator에 비해 현저한 계산 효율성을 제공합니다. 특히, SBG는 tri, tetra, hexapeptides와 같은 복잡한 분자의 상태를 성공적으로 샘플링하며, 이로써 기존 방법으로는 불가능했던 비가역적인 메타스테이블 상태 샘플링의 새로운 가능성을 제시합니다.



### FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Respons (https://arxiv.org/abs/2502.18452)
Comments:
          8 pages, 3 figures, 5 tables

- **What's New**: 대규모 언어 모델(LLMs)은 일반적인 추론 성능이 향상되고 있으나, 작은 모델들은 특정 추론 작업에서 상대적으로 낮은 성능을 보입니다. 이를 해결하기 위해, 저자들은 재난 분야에 특화된 작은 LLMs을 미세 조정(fine-tune)하여 정밀한 데이터 검색이 가능한 FRIDA(Field Ready Instruction Decoding Agent) 모델을 개발했습니다. FRIDA는 도메인 전문가와 언어학자의 지식을 활용하여 고품질의 합성 데이터를 생성하는 파이프라인을 갖추고 있습니다.

- **Technical Details**: FRIDA 모델은 130개의 시드 지침과 25,000개의 합성 명령문, 그리고 지진 관련 119개의 평가 지침을 모았으며, LLaMa와 Mistral 모델을 사용해 미세 조정하였습니다. 이 연구에서는 합성 데이터의 영향을 분석하기 위해 ablation 연구를 진행하였고, 물리적 상태와 객체 기능의 일반적인 상식 훈련이 FRIDA 모델보다 더 효과적임을 확인했습니다. 세부적으로, 재난 구호 작업의 특정 지식 없이는 성능 향상에 한계가 있음을 시사했습니다.

- **Performance Highlights**: FRIDA 모델들은 기본 모델들보다 전반적으로 뛰어난 성능을 보였으며, 특히 모델 크기나 아키텍처에 관계없이 성공적으로 일반 상식을 개선하였습니다. aFRIDA 모델은 일반적인 상식에 기반한 소규모 LLM로 실험하였고, 도메인별 특정 상식 지식에 비해 더 우수한 성과를 거두었습니다. 따라서 FRIDA 파이프라인은 전반적인 상식을 제공할 수 있으며, 추가적인 정보 검색이 도메인 지식을 충족시키는데 필요함을 결론지었습니다.



### SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (https://arxiv.org/abs/2502.18449)
- **What's New**: 이 논문은 SWE-RL이라는 새로운 접근법을 소개하며, 이는 소프트웨어 공학(SW engineering) 분야에서 대형 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 최초의 강화 학습(RL) 방법이다. SWE-RL은 주요 목적이 개발자들의 추론 과정 및 솔루션을 스스로 복구할 수 있도록 LLMs를 학습시키는 것이다. 이러한 기술은 대규모 오픈 소스 소프트웨어 진화 데이터(software evolution data)를 활용하여 현실 세계의 문제를 해결하는 데 초점을 맞추고 있다.

- **Technical Details**: SWE-RL은 규칙 기반 보상(rule-based reward)을 사용하여 LLM의 추론 능력을 개선하며, 코드 스냅샷, 코드 변경 및 문제(issue) 및 풀 리퀘스트(pull requests)와 같은 이벤트를 포함하는 소프트웨어의 모든 생애주기 기록을 학습에 활용한다. Llama3-SWE-RL-70B라는 모델을 구축하였으며, 이는 41.0%의 문제 해결율을 보이며 GitHub에서 인간 검증을 통과한 SWE-bench Verified 데이터셋에서 성능을 검증하였다. 본 논문에서 제시된 방법은 비록 소프트웨어 진화 데이터에만 RL을 수행했음에도 불구하고 일반화된 추론 능력을 보여 주목할 만하다.

- **Performance Highlights**: Llama3-SWE-RL-70B는 현재까지 중간 규모(<100B) LLM에서 보고된 최고의 성능을 기록하였으며, GPT-4o와 같은 선도적인 상용 LLM과도 비교 가능하다는 점에서 중요하다. 본 모델은 다양한 범위 밖의(out-of-domain) 작업에서 성능이 향상되었으며, 그 예로 함수 코딩, 라이브러리 사용, 코드 추론, 수학 및 일반 언어 이해가 있다. 기존의 감독 하에 미세 조정(supervised fine-tuning) 모델과 비교해도 Llama3-SWE-RL-70B가 더 나은 성능을 보여준다.



### Disambiguate First Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing (https://arxiv.org/abs/2502.18448)
- **What's New**: 본 논문에서는 ambiguity(모호성)와 underspecification(불완전 명세)를 처리하는 모듈식 접근 방식을 제안합니다. 이 방법은 자연 언어 해석을 통해 모호성을 해결한 후 이를 SQL 쿼리와 같은 논리적 형태로 매핑합니다. 기존의 LLM(대형 언어 모델)은 명확한 발화를 처리하는 데 뛰어나지만, 모호한 발화에 대해서는 편향된 결과를 보이는 것을 이용하여 초기 선호 해석 집합을 생성하고, 이후 specialized infilling model(특수화된 보간 모델)을 적용해 부족한 해석을 찾아냅니다.

- **Technical Details**: 모호한 질문에 대한 text-to-SQL parsing(텍스트-투-SQL 파싱)에 초점을 맞춰 두 단계 접근 방식인 "disambiguate first, parse later"(먼저 모호성을 해소하고, 그 후 파싱) 방식을 채택합니다. 이 접근 방식에서는 먼저 LLM을 통해 발화의 모든 가능한 의미를 생성한 후, 이를 바탕으로 unambiguous(명확한) 해석을 제공합니다. 또한, synthetic reference interpretations(합성 참조 해석)을 통해 AmbiQT 벤치마크에서의 학습 데이터 생성을 지원하며, SQL 쿼리를 실행하여 해석의 정확성을 검증합니다.

- **Performance Highlights**: 수행된 실험에서는 제안된 접근 방식이 모호한 질문에 대한 해석의 범위를 개선하고 다양한 데이터 세트 전반에서 일반화되는 것을 보여줍니다. 이러한 새로운 접근 방식은 다양한 주석 스타일, 데이터베이스 구조 및 모호성 유형에 대해 효과적으로 작동하며, 이러한 결과는 사용자 신뢰를 높이고 대화 시스템의 실질적인 유용성을 증가시킵니다.



### ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies (https://arxiv.org/abs/2502.18438)
- **What's New**: 이 논문은 ToMCAT(Theory-of-Mind for Cooperative Agents in Teams)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 팀원들의 목표와 행동을 이해하고 예측하는 ToM(Theory of Mind) reasoning을 수행하는 메타-러닝 메커니즘과, 에이전트의 목표와 팀원의 특성에 따라 계획을 생성하는 multiagent denoising-diffusion 모델을 결합합니다. 가장 중요한 점은, 이 시스템이 동적으로 새로운 경로를 샘플링하여 최신 상태에 적응할 수 있도록 설계되었다는 것입니다.

- **Technical Details**: ToMCAT은 두 개의 모듈로 구성됩니다: ToMnet과 MADiff입니다. ToMnet은 다양한 에이전트의 데이터로부터 에이전트의 선호도와 행동에 대한 강력한 사전 정보를 학습하는 신경망입니다. MADiff는 확률적 denoising-diffusion 모델로, 에이전트와 팀원들의 행동을 이해하고 팀원들의 선호도에 적응한 다중 에이전트 계획을 생성합니다. 이러한 설계를 통해 ToMCAT은 에이전트가 자신뿐만 아니라 팀원들의 계획을 예측하고, 팀원들과의 상호작용에서 ToM reasoning을 활용할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, ToMCAT의 동적 재계획 메커니즘이 팀 성과를 유지하면서 자원 사용을 줄이는 데 중요하다는 점이 강조됩니다. 특히, 에이전트가 에피소드 동안 수집한 최근 관찰과 ToM 추론의 결합이 팀원에 적응하여 효과적인 계획을 생성하는 데 필수적입니다. 이는 기존에 제공받은 정보가 없는 상황에서도 효과적으로 팀과의 협력 및 상호작용을 가능하게 합니다.



### TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning (https://arxiv.org/abs/2502.18431)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)의 논리적 추론 능력을 평가할 수 있도록 설계된 새로운 벤치마크인 TextGames를 소개합니다. TextGames는 패턴 인식, 공간 인식, 산술 및 논리적 사고와 같은 고급 기술을 요구하는 텍스트 기반 게임을 통해 LLMs의 성능을 측정합니다. 이 연구는 단일 턴(single-turn) 및 다중 턴(multi-turn) 추론에서의 LLMs의 성능과 그들이 피드백을 활용하여 자기 교정을 통해 후속 답변을 수정할 수 있는 능력을 분석합니다. LLMs는 대부분의 간단한 문제와 중간 수준의 문제에 대해서는 능숙하지만, 더욱 어려운 과제에서는 큰 어려움을 겪는 것으로 나타났습니다.

- **Technical Details**: TextGames 벤치마크는 총 8개의 퍼즐 게임으로 구성되며, 각 게임은 세 가지 난이도 수준을 가지고 있습니다. 게임의 규칙은 매우 복잡하여 LLMs가 세부 지침을 따르는 능력을 평가하는 데 용이합니다. 또한, LLMs가 피드백을 받을 때 이전 생성물에 대해 자기 반성을 통해 오류를 수정할 수 있는지를 조사합니다. 이 연구에서는 GPT-o3 Mini와 같은 추론 최적화 모델이 사전 훈련된 LLMs보다 어려운 과제에서 더욱 강력한 성능을 보임을 발견했습니다.

- **Performance Highlights**: 연구 결과, LLMs는 다중 턴 상호작용에서 피드백을 받았을 때 성능이 향상되고 이전 생성물에 대해 자기 반성할 수 있음을 확인했습니다. 그러나 LLMs는 여전히 시퀀싱, 카운팅, 복잡한 규칙의 일관된 적용에서 어려움을 겪고 있습니다. 특히, TextGames는 고급 추론 능력을 요구하므로, LLMs는 이러한 복잡한 문제 해결에 있어 충분한 능력을 갖추지 못한 것으로 보입니다. човеку, 반면 인간은 주어진 시간 내에 모든 문제를 해결할 수 있는 능력을 지니고 있습니다.



### Comparative Analysis of MDL-VAE vs. Standard VAE on 202 Years of Gynecological Data (https://arxiv.org/abs/2502.18412)
Comments:
          12 pagas, 5 figures, 9th International Conference on Signal, Image Processing (SIPO 2025), Vancouver CA

- **What's New**: 이번 연구에서는 Minimum Description Length (MDL) 정규화가 강화된 Variational Autoencoder (VAE)를 표준 Autoencoder와 비교 평가하였습니다. MDL-VAE는 고차원 부인과 데이터 재구성에서 기존 모델보다 현저히 낮은 재구성 오류(MSE, MAE, RMSE)를 나타냈습니다. 또한, 효과적인 KL divergence 정규화를 통해 더 구조화된 잠재 표현(latent representations)을 드러냈습니다.

- **Technical Details**: 본 연구에서는 고차원 부인과 데이터의 재구성을 위해 MDL 원칙을 VAE 아키텍처에 통합했습니다. 이로 인해 MDL-VAE는 안정적인 훈련 및 검증 손실을 보여주고, 효율적인 추론(inference) 시간을 달성하여 강인성과 실용성을 잘 보여줍니다. 통계적 분석을 통해 이러한 성능 향상이 유의미함을 확인했습니다.

- **Performance Highlights**: MDL-VAE는 표준 Autoencoder에 비해 뚜렷한 성능 개선을 보였습니다. 이는 데이터 재구성 및 일반화에서 상당한 개선 효과가 있음을 시사하며, 의료 데이터 모델링 및 분석에 대한 고급 애플리케이션에 유망한 접근 방식이 될 것입니다.



### TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting (https://arxiv.org/abs/2502.18410)
Comments:
          8 pages, 4 figures, 7 tables and accepted at the AI4TS: AI for Time Series Analysis workshop, AAAI 2025

- **What's New**: 이번 연구에서는 새로운 신경망 구조인 TSKANMixer를 제안합니다. TSKANMixer는 Kolmogorov-Arnold Networks (KAN)를 TSMixer에 통합하여 시간 시계열 예측을 개선하는 데 중점을 두고 있습니다. 실험 결과, TSKANMixer는 기존 TSMixer보다 다양한 데이터셋에서 예측 정확도를 향상시키는 경향을 보였습니다.

- **Technical Details**: TSKANMixer는 KAN 레이어를 TSMixer 아키텍처에 추가하는 방식으로 설계되었습니다. KAN는 고정된 활성화 함수를 사용하는 전통적인 MLP와 달리, 학습 가능한 활성화 함수를 엣지에 적용하며, 노드에서 단순한 덧셈을 수행합니다. 이로 인해 KAN은 전통적인 MLP보다 성능과 해석 가능성을 개선할 수 있는 잠재력을 제공합니다.

- **Performance Highlights**: TSKANMixer는 여러 벤치마크에서 기존의 MLP 기반 모델보다 향상된 성능을 보이며, 특히 복잡하고 비선형적인 의존관계가 존재하는 다변량 시간 시계열 데이터에 대해 탁월한 예측 정확도를 보여주었습니다. 연구 결과, KAN이 시간 시계열 예측 분야에서 기존 방법에 비해 경쟁력 있는 대안이 될 수 있음을 입증했습니다.



### AgentRM: Enhancing Agent Generalization with Reward Modeling (https://arxiv.org/abs/2502.18407)
- **What's New**: 본 연구에서는 기존 LLM 기반 에이전트의 일반화 능력을 개선하기 위해, 정책 모델을 직접 미세 조정하는 것보다 보상 모델(reward model)을 미세 조정하여 정책 모델을 안내하는 것이 더 효과적임을 발견했습니다. 이를 통해 AgentRM이라는 일반화 가능한 보상 모델을 제안합니다. AgentRM은 test-time search에서 정책 모델을 효과적으로 가이드하여, 미지의 작업에서도 성능을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 에이전트 작업은 부분 관찰 가능한 마르코프 결정 과정(partially observable Markov decision process, POMDP)로 정형화되며, 다양한 작업 환경에서의 피드백을 기반으로 합니다. 연구에서는 보상 모델을 구축하기 위해 세 가지 접근 방식(1) 명시적 보상 모델링(explicit reward modeling), (2) 암묵적 보상 모델링(implicit reward modeling), (3) LLM을 평가자로 활용하는 방식(LLM-as-a-judge)을 조사하였습니다. AgentRM은 Best-of-N 샘플링과 단계 수준의 빔 검색(beam search)을 통해 응답 생성을 안내합니다.

- **Performance Highlights**: AgentRM은 아홉 개의 다양한 에이전트 작업에서 평균 8.8점의 성능 향상을 이루었으며, 이는 기존의 최고의 일반화 에이전트보다 4.0점 더 높은 성과입니다. 특히 LLaMA-3-70B 정책 모델에서는 12.6점의 더 큰 개선을 보여주어, 강한 일반화 능력을 입증했습니다. 또한, 미세 조정된 정책 모델을 신속하게 증강하여 세 가지 고정된 작업에서 상위 전문화된 에이전트보다 11.4점 더 나은 성능을 달성했습니다.



### EgoSim: An Egocentric Multi-view Simulator and Real Dataset for Body-worn Cameras during Motion and Activity (https://arxiv.org/abs/2502.18373)
- **What's New**: 최근 컴퓨터 비전 분야의 egocentric(자기중심적) 작업은 주로 헤드 마운트 카메라에 집중되어 왔으나, 저자는 카메라 기술의 소형화가 다양한 신체 착용 장치에 카메라 통합을 촉진할 것이라고 주장합니다. 이러한 변화는 사람의 움직임 추적, 바디 포즈 추정, 액션 인식 같은 기존 작업에 새로운 관점을 제공하고 특히 하체 부분을 다루는 데 중요한 기여를 할 것입니다. 본 논문에서는 여러 각도가 포함된 신체 착용 카메라의 현장 기반 성능을 구현하는 EgoSim 시뮬레이터와 MultiEgoView 데이터셋을 소개합니다.

- **Technical Details**: EgoSim은 여러 신체 착용 카메라로부터 현실적인 egocentric 렌더링을 생성하는 시뮬레이터로, 실제 모션 캡처 데이터를 사용하여 동작 아티팩트를 렌더링합니다. MultiEgoView 데이터셋은 6개의 신체 착용 카메라에서 촬영된 119시간의 egocentric 영상과 진실 위치를 포함하여 다양한 활동을 구성합니다. 또한, 13명의 참가자가 6개의 GoPro 카메라를 착용해 기록한 5시간의 실제 데이터도 포함되어 있으며, Xsens 모션 캡처 수트를 통해 전체 바디 3D 포즈 참조가 제공됩니다.

- **Performance Highlights**: EgoSim을 활용해 훈련한 비디오 기반 3D 포즈 추정 네트워크의 효과를 입증하였으며, 도메인 갭 분석을 통해 데이터셋과 시뮬레이터가 실제 데이터로의 추론에 얼마나 많은 도움이 되는지를 보여주었습니다. 이는 egocentric 인식 작업을 위한 개방형 연구의 발전에 기여할 것으로 예상됩니다. 전반적으로 EgoSim은 신체착용 카메라를 활용한 다양한 동작 인식을 위한 새로운 연구의 가능성을 열어줍니다.



### Which Contributions Deserve Credit? Perceptions of Attribution in Human-AI Co-Creation (https://arxiv.org/abs/2502.18357)
Comments:
          30 pages, 5 figures. In CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan

- **What's New**: 이번 연구에서는 AI와 인간의 공동 창작 과정에서 AI의 기여에 대한 인정이 어떻게 이루어지는지 조사했습니다. 155명의 지식 근로자들과의 설문 조사 결과, 인간 파트너와 동일한 기여를 한 AI 파트너에 대해 상대적으로 낮은 인정이 주어진다는 경향이 발견되었습니다. 연구 결과는 AI 기여에 대한 새로운 인정 접근 방식의 필요성을 강조하고 있습니다.

- **Technical Details**: 연구는 설문 조사를 통해 AI와 인간 파트너의 기여 유형, 양, 주도권 차이에 따라 저작권 인정 수준이 어떻게 변화하는지 분석했습니다. 참여자들은 AI 파트너의 기여에 대한 인정을 결정하는 다양한 요소를 고려했으며, 기여의 질과 개인적 가치가 중요한 역할을 했습니다. 이러한 분석을 통해 AI와 인간 파트너 간의 저작권 차이를 명확히 드러내고, AI 특정 저작권 프레임워크의 필요성을 강조했습니다.

- **Performance Highlights**: 연구 결과는 무엇보다 AI 파트너가 인간 파트너에 비해 동등한 기여에 대한 인정을 덜 받는 경향이 있음을 보여줍니다. 또한, 기여에 대한 인식은 동의 유지와 투명성 필요성에 대한 새로운 접근법을 위한 논의를 촉진할 수 있는 기반이 되고 있습니다. 이를 바탕으로 AI 기여가 진정성과 투명성을 갖춘 방식으로 인정받을 수 있도록 하기 위한 구체적인 디자인 전략도 제안됩니다.



### From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithms (https://arxiv.org/abs/2502.18328)
- **What's New**: 이 논문에서는 최근의 시각적 이상 탐지(Visual Anomaly Detection, VAD) 기술을 오디오 이상 탐지(Audio Anomaly Detection, AAD) 분야에 적용하는 방법을 제시합니다. 전통적인 AAD 기법들은 이상 샘플을 주로 분류하지만, 이 연구에서는 스펙트로그램 내 이상 탐지의 세분화된 시간-주파수(localization) 정보를 제공합니다. 이러한 접근은 사용자에게 이상이 발생한 위치와 시간을 명확하게 제시하여 더욱 활용 가능한 결과를 제공합니다.

- **Technical Details**: 이 연구에서는 비지도학습(unsupervised setting)에서 VAD 알고리즘을 평가하는 프레임워크를 채택하였습니다. 특히, 스펙트로그램을 입력으로 사용하여 특징 추출기(feature extractor)를 통해 생성된 중간 표현을 활용합니다. 이 과정에서 여러 메모리-뱅크(memory-bank) 기술을 포함한 네 가지 유형의 특징 기반 VAD 알고리즘을 평가하여, 각각의 알고리즘이 스펙트로그램 내에서 이상 영역을 어떻게 식별하는지를 분석합니다.

- **Performance Highlights**: MIMMI와 EnvMix 두 가지 벤치마크를 통해 실험을 수행하여, 산업 응용 및 환경 응용에서의 알고리즘 성능을 비교합니다. 다양한 신호 대 잡음비(Signal-to-Noise Ratio, SNR) 레벨에서 모델의 이상 탐지 능력을 평가하며, 이를 통해 AAD 기법의 전반적인 효용성을 검증합니다. 연구 결과는 오디오 이상 탐지 시스템의 설명 가능성을 높이고, 실제 상황에서의 적용 가능성을 높이는 데 기여합니다.



### Smart and Efficient IoT-Based Irrigation System Design: Utilizing a Hybrid Agent-Based and System Dynamics Approach (https://arxiv.org/abs/2502.18298)
Comments:
          50 pages, 22 figures

- **What's New**: 이 연구에서는 현대 기술을 활용하여 물 자원 부족 문제를 해결하기 위한 스마트 관개 시스템을 설계했습니다. IoT(Internet of Things)를 기반으로 하는 이 시스템은 Agent-Oriented Software Engineering (AOSE) 방법론을 이용해 설계되었습니다. 특히, Prometheus AOSE 방법론을 통해 토양 수분을 적절한 범위로 유지하여 물의 손실을 줄이는 것을 목표로 하고 있습니다.

- **Technical Details**: 설계된 스마트 관개 시스템은 센서(센서), 중앙 에이전트(central agent), 그리고 관개 노드(irrigation nodes)로 구성되어 있습니다. 이 시스템은 정의된 규칙을 따르며 협력적으로 토양 수분을 유지합니다. 또한, 하이브리드 에이전트 기반 시스템다이내믹스 모델을 활용하여 시뮬레이션이 이루어졌으며, 잔여 시간 내에 효율적인 관개를 보장합니다.

- **Performance Highlights**: 256회의 분수 인자 실험을 통하여 자동 관개 모드에서 시스템의 기능이 테스트되었습니다. 실험 결과, 이 시스템은 모든 테스트에서 거의 최적의 물 양으로 관개를 수행했습니다. 또한, 이 결과는 시스템의 에너지 소비를 줄여 운영 시간을 최소화하는 데에도 활용되었습니다.



### Mixing Any Cocktail with Limited Ingredients: On the Structure of Payoff Sets in Multi-Objective MDPs and its Impact on Randomised Strategies (https://arxiv.org/abs/2502.18296)
Comments:
          64 pages

- **What's New**: 본 논문에서는 마르코프 결정 프로세스(Markov decision processes)에서 다차원 보상 함수(multi-dimensional payoff functions)를 다룹니다. 특정한 기대 보상 벡터(expected payoff vector)가 달성 가능한지 여부를 탐구하고 있습니다. 본 연구는 순수 전략(pure strategies)만으로는 이 문제를 해결할 수 없다는 점을 강조합니다.

- **Technical Details**: 연구팀은 여러 전략의 예상 보상 벡터(expected payoff vectors) 집합의 구조를 분석하고, 전략에 대한 난수화(randomisation) 요구 사항의 결과를 살펴봅니다. 논문에서는 모든 전략에서 기대가 정립된 경우, 유한한 수의 순수 전략을 혼합(mix)하여 원하는 정밀도까지 어떤 기대 보상 벡터를 근사할 수 있음을 증명합니다.

- **Performance Highlights**: 또한, 모든 전략에서 기대 보상이 유한한 경우에는 유한한 전략들을 혼합하여 모든 기대 보상을 정확히 얻을 수 있음을 보여줍니다. 이 연구는 다차원 보상 함수를 활용한 의사결정 프로세스에서 전략 선택에 대한 더 깊은 통찰을 제공합니다.



### AMPO: Active Multi-Preference Optimization (https://arxiv.org/abs/2502.18293)
- **What's New**: 이번 연구에서는 Active Multi-Preference Optimization (AMPO)라는 새로운 접근법을 소개합니다. AMPO는 대량의 후보 응답을 평가하고, 중요한 극단과 독특한 의미 군을 포괄하는 소규모 정보 집합을 선택하여 언어 모델의 선호 최적화를 가능하게 합니다. 이를 통해 다중 선호 최적화의 과정에서 모델이 학습하는 신호의 질을 높입니다.

- **Technical Details**: AMPO는 (a) 정책 기반 데이터 생성, (b) 그룹 기반 선호 학습, 그리고 (c) 능동적인 서브셋 선택을 통합하는 프레임워크입니다. 고유의 그룹 대조 손실 함수인 Refa를 채택하여 다수의 긍정적 및 부정적 응답을 단일 손실 항목으로 결합합니다. 또한, 선택된 응답의 다양성을 극대화하는 방법으로 베이직한 bottom-K에서 이론적으로 기반한 Opt-Select 방식까지 다양한 능동 선택 방식을 탐구합니다.

- **Performance Highlights**: AMPO는 Llama 8B 모델을 기반으로 한 AlpacaEval에서 최신 성과를 달성하였으며, Simpo와 같은 강력한 기준선 모델을 초월했습니다. 연구자들은 Hugging Face에 AMPO-Coreset-Selection 및 AMPO-Opt-Selection 데이터셋을 공개하여 다중 선호 정렬 연구를 촉진합니다.



### A Reverse Mamba Attention Network for Pathological Liver Segmentation (https://arxiv.org/abs/2502.18232)
Comments:
          16 pages, 3 figures

- **What's New**: RMA-Mamba는 시각적 상태 공간 모델의 능력을 향상시키는 혁신적인 아키텍처로, 특별히 설계된 Reverse Mamba Attention 모듈(RMA)을 도입합니다. 이 아키텍처는 장기적인 의존성을 포착하면서도 정밀한 지역적 특성을 유지하는 계층적 처리 파이프라인을 갖추고 있습니다.

- **Technical Details**: RMA-Mamba는 Vision Mamba(VMamba)의 효율적 시퀀스 모델링과 RMA의 특성 개선을 통합하여 다중 스케일에서 우수한 특성 학습을 달성합니다. RMA 모듈은 디코더 작업 중 진행적 특성 통합 전략을 구현하여 미세한 조직 변동과 대규모 해부학적 맥락을 동시에 포착합니다.

- **Performance Highlights**: RMA-Mamba는 T2 가중 MRI 스캔을 이용한 새로운 경화 간 데이터셋(CirrMRI600+)에 대해 92.08%의 Dice 계수와 87.36%의 평균 IoU를 달성하며, 기존의 간 분할 접근법에 비해 뛰어난 성능을 보여줍니다. 또한, RMA-Mamba의 일반화 능력은 CT 스캔의 간암 분할에서도 확인되어 Dice 점수 92.9%와 mIoU 88.99%를 기록했습니다.



### Liver Cirrhosis Stage Estimation from MRI with Deep Learning (https://arxiv.org/abs/2502.18225)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문에서는 다중 시퀀스 MRI를 이용한 자동 간경변 단계 추정의 최첨단 딥러닝 프레임워크를 제시합니다. 간경변은 간의 심각한 흉터(fibrosis)로, 여러 만성 간질환의 일반적인 결과입니다. 초기 진단은 합병증을 예방하기 위해 매우 중요하지만, 조기 간경변을 진단하는 소프트웨어나 데이터가 부족합니다.

- **Technical Details**: 우리는 CirrMRI600+라는 대규모 데이터셋을 활용하여 339명의 환자로부터 수집한 628개의 고해상도 MRI 스캔을 분석했습니다. 딥러닝 프레임워크는 시퀀스 특화된 피처 추출과 교차 시퀀스 주의 메커니즘을 통합하여 간경변의 미세한 조직 변화를 포착합니다. 또한, VGG-19, ResNet 변형, MambaVision, ConvNext 등 8가지 다양한 딥러닝 아키텍처를 평가하여 최상의 성능을 기록했습니다.

- **Performance Highlights**: 최고의 모델은 T1W 시퀀스에서 72.8%의 정확도를 달성하였고, T2W 시퀀스에서는 63.8%의 성능을 보였습니다. 전통적인 radiomics 접근 방식에 비해 월등히 뛰어난 성능으로, 자동 간경변 단계 분류의 새로운 기준을 세웠습니다. 본 연구는 간경변 조기 진단의 정확성을 향상시키고, 나아가 임상에서의 사용 가능성을 제시합니다.



### UASTrack: A Unified Adaptive Selection Framework with Modality-Customization in Single Object Tracking (https://arxiv.org/abs/2502.18220)
- **What's New**: 이번 연구에서는 모드 적응 인식을 통한 단일 객체 추적을 위한 통합 적응 선택 프레임워크인 UASTrack을 제안합니다. UASTrack은 Discriminative Auto-Selector (DAS)를 활용해 다양한 RGB-X 이미지 쌍에 따라 입력 모드 유형을 동적으로 식별할 수 있습니다. 또한, Task-Customized Optimization Adapter (TCOA)는 각 모드의 특성을 반영하여 최적화된 처리를 수행합니다. 이를 통해 다양한 다중 모드 추적 작업에서 균일하고 효율적인 성능을 달성할 수 있습니다.

- **Technical Details**: UASTrack은 RGB-X(여기서 X는 depth, event 또는 thermal 모드를 의미합니다) 이미지 쌍을 기반으로 하여, 다중 모드 작업 간 효율적인 통합 처리를 가능하게 합니다. DAS 모듈은 모드별로 적합한 네트워크 구조를 선택하도록 지침을 제공하며, 클래스 제약 손실(Classification Constraint Loss)을 추가하여 학습 능력을 향상시킵니다. TCOA는 각 모드별 특성을 고려하여 스타일을 최적화하여 더 나은 예측을 가능하게 합니다. 아울러, Transformer Encoder 블록 내에 양방향 어댑터를 도입해 RGB와 X 기능 간의 효과적인 상호작용을 지원합니다.

- **Performance Highlights**: UASTrack은 LasHeR, GTOT, RGBT234, VisEvent, DepthTrack 등에서 경쟁력 있는 성능을 입증했습니다. 특히, 모드별 특징에 최적화된 구조 덕분에, 필수적인 추가 파라미터는 1.87M, FLOPs는 1.95G에 불과하며, 기존 RGB-X 기준 대비 8.5%의 성공률 향상을 이뤘습니다. 다양한 벤치마크에서의 비교를 통해, 우리의 접근 방식이 최첨단 트래커들보다 효율적으로 작동함을 확인했습니다.



### FLARE: A Framework for Stellar Flare Forecasting using Stellar Physical Properties and Historical Records (https://arxiv.org/abs/2502.18218)
- **What's New**: 이 논문에서는 별 플레어 예측에 대한 새로운 접근법을 제안하며, FLARE(Forecasting Light-curve-based Astronomical Records via features Ensemble)라는 대규모 모델을 소개합니다. FLARE는 별의 물리적 특성과 역사적 플레어 기록을 통합하여, 이러한 요소들이 플레어 예측에 중요한 역할을 한다는 실험 결과를 제시합니다.

- **Technical Details**: FLARE는 Soft Prompt Module과 Residual Record Fusion Module이라는 새로운 구성 요소들을 통해 별의 물리적 특성과 역사적 플레어 기록을 효과적으로 통합합니다. 효율적인 특징 추출을 위해 LoRA로 미세 조정된 다중 모달 모델을 사용하여, 플레어 예측의 정확도를 높입니다.

- **Performance Highlights**: Kepler 데이터를 사용한 실험 결과, FLARE는 모든 평가 메트릭에서 기존의 다른 방법들에 비해 뛰어난 성능을 나타냈습니다. 이를 통해 PLARE는 별 플레어 예측 분야에서 혁신적인 기여를 하고 있으며, 향후 천문학적 연구에 유용한 도구가 될 것으로 기대됩니다.



### LAG: LLM agents for Leaderboard Auto Generation on Demanding (https://arxiv.org/abs/2502.18209)
- **What's New**: 이 논문은 Leaderboard Auto Generation (LAG)이라는 새로운 프레임워크를 소개하여 인공지능(AI)과 같은 빠르게 발전하는 분야 내의 연구 주제에 대한 자동 생성된 리더보드를 효율적으로 구축하는 방법을 제시합니다. 매일 대량의 AI 논문이 업데이트되면서 연구자들이 각 논문의 방법론, 실험 결과 및 설정을 추적하는 것이 점점 더 어려워지고 있습니다. LAG는 논문 수집, 실험 결과 추출 및 통합, 리더보드 생성, 품질 평가의 체계적인 접근 방식을 통해 이러한 문제를 해결합니다.

- **Technical Details**: LAG는 논문 수집, 테이블 추출 및 분류, 테이블 언팩킹 및 통합, 리더보드 생성 및 평가의 네 가지 단계로 구성됩니다. 이 프레임워크는 관련 있는 논문을 아카이브에서 자동으로 다운로드하고 신뢰성 있는 실험 설정을 제공하여 공정한 비교를 가능하게 합니다. 평가 측면에서 LAG는 주제 관련 품질과 내용 품질의 두 가지 품질 차원을 제시하여 생성된 리더보드의 효과성을 극대화합니다.

- **Performance Highlights**: LAG는 다채로운 리더보드 길이에서 실험을 통해 높은 주제 관련성 및 내용 품질 점수를 지속적으로 달성했습니다. 특히 20개의 항목이 포함된 리더보드를 생성했을 때, 주제 관련 품질에서 67.58%의 회수율과 70.33%의 정밀도를 기록했습니다. LAG는 수동으로 작성된 리더보드에 비해 효율성에서 우수하며, 인간의 평가와 높은 상관관계를 나타내는 결과를 도출하였습니다.



### DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches (https://arxiv.org/abs/2502.18202)
- **What's New**: DenoMAE2.0는 전통적인 reconstruction loss와 함께 지역 패치 분류(objective) 목표를 통합하여 표현 학습 및 견고성을 개선한 향상된 denoising masked autoencoder입니다. 기존의 Masked Autoencoders(MAE)와는 달리 DenoMAE2.0은 마스킹되지 않은 패치에 대한 위치 인식을 포함하여 로컬 피쳐를 세밀하게 포착하는 동시에 전역 일관성을 유지합니다. 이 접근법은 특히 무선 통신의 반지도 학습(semisupervised learning)에서 높은 노이즈와 데이터 부족 문제가 심각한 상황에서도 이점을 제공합니다.

- **Technical Details**: DenoMAE2.0의 전체 프레임워크는 인코더, 재구성 denoising 디코더, 지역 패치 분류 헤드의 세 가지 구성 요소로 이루어져 있습니다. 첫 번째 단계에서는 입력 이미지를 비중첩(non-overlapping) 패치 시퀀스로 변환하며, 무작위로 75%의 패치를 마스킹합니다. 이렇게 생성된 가시 패치(visible patches)는 Vision Transformer(ViT) 인코더를 통해 패치 수준 특성을 생성하고, 이를 기반으로 지역 패치 분류의 목표를 설정하여 보다 구체적이고 유용한 표현을 학습합니다.

- **Performance Highlights**: DenoMAE2.0은 Deno-MAE와 기타 여러 기준선들과 비교하여 노이즈 제거 품질 및 다운스트림 분류 정확도 모두에서 우수한 성능을 보여줍니다. 데이터셋에서 DenoMAE에 비해 1.1%의 개선을 달성했으며, RadioML 벤치마크에서 별도의 분류 작업에서 11.83% 및 16.55%의 유의미한 정확도 향상을 기록했습니다. 이러한 결과는 DenoMAE2.0이 반지도 학습(ssemi-supervised learning) 환경에서 효과적으로 작동한다는 것을 보여줍니다.



### VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with LoRA and Atrous Attention (https://arxiv.org/abs/2502.18185)
Comments:
          Submitted to IEEE JBHI

- **What's New**: 본 논문에서는 기계 학습을 기반으로 한 새로운 의료 영상 분할 모델인 VesselSAM을 제안합니다. VesselSAM은 aortic vessel segmentation을 위한 Segmentation Anything Model(SAM)의 수정된 버전입니다. AtrousLoRA라는 새로운 모듈을 도입하여 Atrous Attention과 Low-Rank Adaptation(LoRA)을 결합하여 영상 분할 성능을 개선합니다. 이 모델은 다중 스케일 컨텍스트 정보를 포착하여 지역 세부사항과 글로벌 컨텍스트를 모두 보존할 수 있는 성능이 특징입니다.

- **Technical Details**: VesselSAM은 Atrous Spatial Pyramid Pooling(ASPP)과 Attention 메커니즘을 통해 다중 스케일 정보를 효율적으로 캡처합니다. LoRA를 활용하여 프리트레인된 SAM의 이미지 인코더를 동결한 상태로 프로그램의 성능을 유지하면서 학습 가능한 파라미터 수를 줄일 수 있습니다. 이를 통해 기존의 대형 모델에 비해 계산 비용을 대폭 절감할 수 있으며, 보다 효율적인 fine-tuning을 가능하게 합니다.

- **Performance Highlights**: VesselSAM은 두 개의 도전적인 데이터셋, 즉 Aortic Vessel Tree(AVT) 데이터셋과 Type-B Aortic Dissection(TBAD) 데이터셋에서 평가되었습니다. 여러 의료 센터에서 DSC 점수 93.50%, 93.25%, 93.02%, 93.26%의 성과를 달성하며 최첨단 성능을 보여줍니다. 이러한 결과는 기계 학습 기반의 aortic vessel segmentation이 임상 환경에서 더욱 개선될 것으로 기대될 수 있음을 나타냅니다.



### Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs (https://arxiv.org/abs/2502.18179)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 이용한 레이아웃이 풍부한 문서에서 정보 추출(IE)의 설계 공간을 정의하고 탐색합니다. LLMs를 활용하는 레이아웃 인식 IE의 세 가지 핵심 도전 과제는 데이터 구조화, 모델 참여, 출력 세분화로 나뉩니다. 이 연구는 입력 표현, 청크(chunking), 프롬프트(prompting), LLM 및 멀티모달 모델의 선택 등과 같은 하위 문제를 탐구하고, 다양한 설계 선택의 결과를 새로운 테스트 수트를 통해 평가합니다.

- **Technical Details**: 레이아웃이 풍부한 문서에서의 IE는 텍스트와 복잡한 시각 레이아웃이 서로 얽힌 문서에서 정보를 식별하고 추출하여 구조화된 정보 인스턴스로 매핑하는 과정을 포함합니다. LLM은 문서의 레이아웃 정보를 잘 혼합하여 텍스트 기반 모델과 다르게 접근해야 하며, 다양한 프리프로세싱과 포스트프로세싱 기술을 평가할 필요가 있습니다. 연구에서는 LLMs의 성능을 다양한 모델과 비교하고 레이아웃 인식 모델인 LayoutLMv3와의 성능을 비교하여 인사이트를 제공합니다.

- **Performance Highlights**: 실험 결과, LLM은 특수한 모델과 경쟁할 수 있으며, 특히 데이터 훈련 없이도 유사한 성능을 발휘할 수 있습니다. 비교 연구에서는 기본 모델 대비 14.1 포인트 F1 점수 향상과 함께 다양한 조합 및 방법론의 효과를 분석했습니다. 그러나 멀티모달 LLM이 여전히 더 나은 성능을 보이지만, 비용과 투명성의 문제를 동반하고 있습니다.



### CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification (https://arxiv.org/abs/2502.18176)
Comments:
          accepted by ICLR 2025

- **What's New**: 이 논문은 적대적으로 강건한 제로샷 이미지 분류기를 구축하는 것을 목표로 합니다. 특히, CLIP 모델을 기반으로 하여, 적대적 공격에 대한 방어를 위해 'Purification' 방법을 활용합니다. CLIPure라는 새로운 접근법은 기존의 생성 모델에 의존하지 않으면서도 방어 효율성을 크게 향상시킨 방식으로, 제로샷 분류의 안정성을 크게 개선합니다.

- **Technical Details**: Purification 리스크는 적대적 샘플 정제 과정의 KL divergence를 통해 정의됩니다. 연구자들은 Bidirectional Stochastic Differential Equations (SDEs)를 활용하여 공격 프로세스와 정제 프로세스를 각각 모델링하고, 이 과정에서 KL divergence와 분포의 매끄러움과 같은 요소들이 purification 성능에 영향을 미친다고 강조합니다. CLIPure는 CLIP의 다중 모달 잠재 공간에서 수행되며, 두 가지 변종인 CLIPure-Diff와 CLIPure-Cos를 제안합니다.

- **Performance Highlights**: CLIPure의 실험 결과는 CIFAR-10에서 71.7%에서 91.1%로, ImageNet에서는 59.6%에서 72.6%로 조정되었으며, 13개 데이터셋에서는 평균적으로 108%의 상대적 개선을 보여주었습니다. 시장의 최고 성과(SOTA)를 상당히 초과하는 결과로 제로샷 분류의 강건성을 입증하며, CLIPure는 적대적 공격을 견디는 데 있어 중요한 진전을 이루었습니다.



### SECURA: Sigmoid-Enhanced CUR Decomposition with Uninterrupted Retention and Low-Rank Adaptation in Large Language Models (https://arxiv.org/abs/2502.18168)
Comments:
          New work on Parameter-Efficient Fine-Tuning (PEFT) for large language models. Includes new techniques SigNorm and CABR-LoRA for optimizing fine-tune performance and Knowledge retention

- **What's New**: 본 논문에서는 SECURA: Sigmoid-Enhanced CUR Decomposition LoRA를 제안하며, 파라미터 효율적인 미세 조정(PEFT) 접근 방식을 통해 파라미터 소실 문제를 완화하고 성능을 향상시킵니다. SECURA는 새로운 정규화 기법인 SigNorm을 도입하여 파라미터 보존과 전반적인 성능을 강화하는 방안을 제시합니다. 다양한 작업에서 SECURA가 기존의 LoRA 방법에 비해 고급 성능을 보여주었다는 점이 주목할 만합니다.

- **Technical Details**: SECURA의 두 가지 핵심 혁신은 CABR 분해와 SigNorm 정규화입니다. CABR 분해는 CUR-LoRA의 성능을 개선하기 위해 역 저순위 적응 행렬을 도입하여 추가 차원을 통합합니다. SigNorm은 시그모이드의 점진적 전이 속성을 활용하여 파라미터를 동적으로 조정하는 정규화 방법으로, 모델의 안정성과 성능을 균형 있게 유지하는 데 기여합니다.

- **Performance Highlights**: SECURA는 4개의 다중 선택 질문(MCQ) 작업에서 평균 3.63% 향상된 성능을 보였고, 5개의 질문 응답(QA) 작업에서도 2.56% 개선을 달성했습니다. 16개의 지속 학습 테스트에서 70% 이상의 정확도를 유지하며 기존의 다양한 방법보다 우수한 지식 보존 능력을 입증했습니다. 이러한 결과는 SECURA가 업무 적응 및 사전 훈련된 지식 보존에서 뛰어난 성능을 가진다는 것을 강조합니다.



### iTrash: Incentivized Token Rewards for Automated Sorting and Handling (https://arxiv.org/abs/2502.18161)
Comments:
          Article submitted to IROS 2025

- **What's New**: 이 논문에서는 사무실과 같은 소규모 공간에서 재활용 효율성을 높이기 위해 iTrash라는 지능형 쓰레기통을 제안합니다. 연구팀은 5일 간의 실험을 통해 iTrash가 기존 쓰레기통에 비해 30% 이상의 효율성 향상을 보였음을 발견했습니다. iTrash는 사용자의 행동 데이터와 쓰레기통 사용 패턴을 수집하여 작업 예측 및 최적화에 도움을 주는 정보도 제공합니다. 마지막으로, 블록체인 기술을 활용하여 재활용을 위한 경제적 인센티브를 제공하는 가능성도 탐구하였습니다.

- **Technical Details**: iTrash는 블록체인 기반의 Waste-to-Reward 시스템을 통해 올바른 재활용을 유도하는 로봇 시스템으로 설계되었습니다. 사용자 행동에 따라 암호화폐 형태의 보상을 지급하며, 이는 XRP 네트워크를 통해 구현됩니다. 시스템은 근접 센서 및 이미지 분류 모델을 사용하여 쓰레기를 분류하고, LED 스트립을 통해 올바른 쓰레기통을 안내합니다. iTrash는 3D 프린팅된 기계 부품을 사용하여 다양한 쓰레기통에 쉽게 설치할 수 있도록 설계되었습니다.

- **Performance Highlights**: iTrash의 실험 결과, 전통적인 쓰레기통에 비해 재활용률이 30% 이상 증가하였고, 이는 사용자의 올바른 분리배출을 장려하는 효과를 나타냅니다. 추가적으로, 사용자는 쓰레기를 올바르게 처리할 때마다 블록체인 기술을 통해 얻는 보상을 실질적인 서비스나 상품 구매에 활용할 수 있는 새로운 환경을 제공합니다. 이러한 결과는 스마트한 공간에서의 재활용 효율성을 높이는 중요한 발전이 될 것입니다.



### Monitoring snow avalanches from SAR data with deep learning (https://arxiv.org/abs/2502.18157)
- **What's New**: 이 논문에서는 눈사태 감지를 위한 최신 딥러닝 기술의 적용을 다루고 있습니다. 기존의 SAR(Synthetic Aperture Radar) 이미지 분석 방법과 비교하여, 최근의 연구들은 픽셀 수준에서의 세분화(segmentation)를 통해 훨씬 더 높은 정확도와 공간 해상도를 제공합니다. 특히, Sentinel-1 SAR 데이터를 활용한 사례 연구를 통해 딥러닝 모델이 전통적인 방법보다 더 뛰어난 성능을 발휘함을 보여주었습니다.

- **Technical Details**: 눈사태를 감지하기 위해, 딥러닝 기반의 CNN(Convolutional Neural Networks) 모델이 SAR 데이터에서 특징을 자동으로 추출할 수 있도록 설계되었습니다. 이는 고해상도 이미지를 처리하고 복잡한 패턴을 학습하는 데 유리하여, 관련 데이터를 효율적으로 분석할 수 있게 합니다. 연구에서는 4,500개 이상의 주석이 달린 SAR 이미지를 활용하여 다양한 최신 세분화 아키텍처들을 테스트하였고, 이를 통해 대규모 눈사태 감지에서의 효과적인 활용 가능성을 확인했습니다.

- **Performance Highlights**: 신뢰성 있는 눈사태 감지 및 예측을 위해 SAR 데이터를 통해 파생된 딥러닝 모델이 사용되었습니다. 이 모델은 노르웨이 전역에서 대규모 눈사태 감지에 적용되어 여러 겨울 시즌 동안 중요한 공간적 및 시간적 패턴을 드러냈습니다. 기존의 전통적인 방법들에 비해 깊이 있는 분석을 제공하는 이 모델은 향후 눈사태 위험 예측 및 관리에 중요한 역할을 할 것으로 기대됩니다.



### Can LLMs Explain Themselves Counterfactually? (https://arxiv.org/abs/2502.18156)
- **What's New**: 본 논문에서는 Self-Generated Counterfactual Explanations (SCEs)에 대한 연구를 소개합니다. LLM(대형 언어 모델)의 출력을 스스로 설명하도록 유도하는 새로운 패러다임이 떠오르고 있으며, 이 연구는 다양한 LLM이 SCE를 얼마나 잘 생성하는지를 평가합니다. 이를 위해 다양한 LLM과 데이터셋을 사용한 실험을 통해 모델의 내부 추론 과정에서 몇 가지 결함을 발견했습니다.

- **Technical Details**: 연구에서는 7B에서 70B 파라미터를 가진 7개의 LLM과 4개 고유 작업에 해당하는 6개의 데이터셋을 사용하여 실험을 진행하였습니다. 실험 절차는 첫째로 모델에게 예측을 요청하고, 그 다음 SCE를 생성하게 하며, 마지막으로 생성된 SCE에 대한 모델의 예측을 계산하는 방식으로 구성됩니다. 결과적으로, 대부분의 LLM은 SCE를 생성하지만, 그 예측 결과가 목표 레이블과 일치하지 않는 경우가 많습니다.

- **Performance Highlights**: LLM이 SCE를 생성하는 데 있어 내부 추론 과정의 결함이 드러났으며, 이는 원래 예측과 SCE 생성 지시가 모델의 최종 예측에 큰 영향을 미친다는 것을 시사합니다. 특히 GSM8K 수학 데이터셋에서 SCE의 유효성이 낮게 나타났습니다. 연구 결과는 현대 LLM이 스스로의 예측을 설명하는 능력에 있어 여전히 부족함을 보여주고 있습니다.



### SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation (https://arxiv.org/abs/2502.18153)
- **What's New**: 이 논문은 근사적 2차 최적화 방법이 일반적으로 1차 접근 방식에 비해 낮은 일반화 성능을 보인다는 문제를 탐구합니다. 기존의 2차 방법들이 더 뚜렷한 샤프(minima)로 수렴하는 경향이 있다는 것을 발견하고, 솔루션의 샤프함을 명시적으로 줄이면서 일반화 성능을 향상시키기 위해 Sassha라는 새로운 방법을 제안합니다. 이를 통해 시간 효율성을 확보하고 안정성을 증대시키는 방법론을 제시합니다.

- **Technical Details**: Sassha는 'Sharpness-aware Adaptive Second-order optimizer with Stable Hessian Approximation'의 약자로, 기본적으로 Hessian의 대각선을 추정하는 2차 최적화 프레임워크에 SAM 방식의 샤프함 최소화 기법을 통합합니다. 이 기법은 샤프함 감소 프로세스를 시행할 때 수치적으로 불안정해질 수 있지만, 문헌에서 연구된 원칙에 따라 설계된 여러 가지 조치를 통해 안정성을 증대시킵니다. 이론적으로 Sassha는 예측된 커브를 부드럽게 조정하고, 이전에 계산된 Hessians의 효율적인 재사용을 가능함으로써 안정적이고 효율적인 알고리즘을 구현합니다.

- **Performance Highlights**: Sassha는 다양한 시각 및 자연어 작업에 대해 광범위하게 평가되었으며, 기존 2차 방법 및 SGD, AdamW, SAM과 같은 1차 방법에 비해 더 평평한 미니마와 강화된 일반화 성능을 꾸준히 달성하였습니다. 논문에서는 수렴성(convergence), 강건성(robustness), 안정성(stability), 효율성(efficiency), 비용(cost)에 대한 포괄적인 분석을 제공하여 Sassha의 성과를 심층적으로 연구하였습니다.



### A Real-time Spatio-Temporal Trajectory Planner for Autonomous Vehicles with Semantic Graph Optimization (https://arxiv.org/abs/2502.18151)
Comments:
          This work has been accepted for publication in IEEE Robotics and Automation Letters (RA-L). The final published version is available in IEEE Xplore (DOI: https://doi.org/10.1109/LRA.2024.3504239)

- **What's New**: 이번 논문에서는 복잡한 도시 환경에서 자율주행차의 안전하고 실행 가능한 경로를 실시간으로 계획할 수 있는 방법을 제안합니다. 제안된 방법은 그래프 최적화(graph optimization)에 기반을 둔 시공간 경로 계획(spatio-temporal trajectory planning) 방식입니다. 이는 정적(static) 및 동적(dynamic) 장애물 분리를 통해 구축된 의미적 시공간 지도(semantic spatio-temporal map)를 이용하여 감지 모듈의 다중 모드 정보(multi-modal information)를 효과적으로 추출합니다.

- **Technical Details**: 이 연구의 핵심 기술은 의미적 시공간 하이퍼그래프(semantic spatio-temporal hypergraph)를 기반으로 하는 희소 그래프 최적화(sparse graph optimization)입니다. 이를 통해 자율주행차는 동적인 도시 상황에서도 빠르고 효율적인 경로를 생성할 수 있습니다. 실험에서 제안된 방법은 복잡한 도시 공공 도로 시나리오를 효과적으로 처리할 수 있다는 것을 입증했습니다.

- **Performance Highlights**: 제안된 방법은 실시간(real-time)으로 전환되며, 복잡한 환경에서도 안정적인 경로 생성을 보여주었습니다. 또한, 연구 커뮤니티를 위한 벤치마킹(benchmarking) 지원을 위해 코드도 배포할 예정입니다. 이로 인해 다른 연구자들이 방법론을 쉽게 검증하고 개선할 수 있을 것입니다.



### Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations (https://arxiv.org/abs/2502.18147)
- **What's New**: 이번 연구에서는 Sparse autoencoders (SAEs)를 넘어 Jacobian SAEs (JSAEs)를 제안합니다. JSAEs는 모델 구성 요소의 입력과 출력 활성화에서뿐만 아니라 이들을 연결하는 계산에서도 희소성(sparsity)을 제공합니다. 이를 통해 LLMs의 계산을 이해하는 데 기여할 수 있는 새로운 방향성을 제시합니다.

- **Technical Details**: JSAEs의 주요 기술 기여는 Jacobians를 효율적으로 계산하는 방법을 찾는 것입니다. LLM의 크기 때문에 Jacobians를 직접 계산하는 것은 실용적이지 않지만, 적절한 기초(JSAE basis)로 재작성할 경우 MLP는 대략적으로 선형(linear)으로 변환됩니다. 이 방법은 계산 희소성을 유지하면서도 전통적인 SAEs와 유사한 LLM 성능을 제공합니다.

- **Performance Highlights**: 실험 결과, JSAEs는 사전 훈련된 LLM에서 계산 희소성이 더 높게 나타났습니다. 이는 LLM이 훈련을 통해 학습한 계산 그래프의 희소성이라는 특성을 지니고 있음을 보여줍니다. 따라서 JSAEs는 표준 SAEs보다 학습된 transformer 계산을 이해하는 데 더 적합할 수 있다는 가능성을 제시합니다.



### Large Language Model Driven Agents for Simulating Echo Chamber Formation (https://arxiv.org/abs/2502.18138)
- **What's New**: 최근 소셜 미디어 플랫폼에서 에코 챔버의 증가가 우려되고 있으며, 기존의 의견을 강화하는 경향이 있습니다. 본 논문에서는 기존의 정형화된 규칙과 수치적 시뮬레이션을 넘어, 대규모 언어 모델(LLM)을 이용하여 소셜 네트워크 내에서 에코 챔버의 역동성을 시뮬레이션하는 새로운 프레임워크를 제안합니다. LLM을 사용하여 의견 업데이트 및 네트워크 재연결 행동을 시뮬레이션함으로써, 더 많은 맥락을 인식하고 의미적으로 풍부한 사회적 상호작용을 포착할 수 있게 되었습니다.

- **Technical Details**: 이 연구에서는 사용자 의견의 진화와 사회 네트워크 내 구조적 변화를 포착하여 에코 챔버의 형성을 시뮬레이션합니다. 각 사용자는 이웃 사용자의 의견에 영향을 받아 자신의 의견을 업데이트하며, 의견 업데이트 규칙은 고전적인 DeGroot 모델에 영감을 받아 디자인됩니다. 또한, LLM을 이용하여 텍스트적 맥락을 통합한 방식으로 영향 함수와 호환성 함수를 정의하여, 보다 현실에 가까운 시뮬레이션을 진행합니다.

- **Performance Highlights**: 실제 소셜 네트워크 데이터와 LLM 기반 시뮬레이션을 비교 분석하여, 생성된 의견 추세의 정확성과 사실성을 파악했습니다. LLM을 사용한 에코 챔버 형성 모델링의 효과성을 보여주었으며, 구조적 및 의미적 차원에서 의견 클러스터링을 효과적으로 캡처할 수 있음을 증명했습니다. 이 연구는 온라인 커뮤니티의 양극화와 사회적 영향 역학에 대한 이해를 심화하는 도구로 기능할 것으로 기대됩니다.



### SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inferenc (https://arxiv.org/abs/2502.18137)
- **What's New**: SpargeAttn은 다양한 생성 작업에 보편적으로 적용할 수 있는 새로운 sparse attention 기법입니다. 기존의 sparse attention이 특정 태스크에 맞춰 최적화된 반면, SpargeAttn은 기계 학습 모델의 end-to-end 성능을 유지하면서 다양한 모델에 대해 속도를 높일 수 있도록 설계되었습니다.

- **Technical Details**: SpargeAttn의 핵심 기술은 두 단계의 온라인 필터를 활용하여 sparse attention을 구현하는 것입니다. 첫 번째 단계에서는 입력의 attention 맵을 신속하고 정확하게 예측하여 필요하지 않은 행렬 곱셈을 생략합니다. 두 번째 단계에서는 온라인 softmax-aware 필터를 통해 추가적인 오버헤드 없이 좀 더 많은 행렬 곱셈을 생략합니다.

- **Performance Highlights**: SpargeAttn은 기존의 dense 및 sparse attention 모델에 비해 2.5배에서 5배 더 빠른 속도를 자랑합니다. 이 방법은 언어 모델링, 이미지 및 비디오 생성 등 다양한 작업에서 model의 end-to-end 성능을 유지하며 효율성을 극대화합니다.



### EU-Nets: Enhanced, Explainable and Parsimonious U-Nets (https://arxiv.org/abs/2502.18122)
- **What's New**: 본 연구에서는 모든 U-Net 아키텍처에 적응 가능한 프레임워크인 MHEX+를 제안합니다. MHEX+를 기반으로 한 EU-Nets라는 새로운 U-Net 변형을 도입하여 전통적인 U-Net 모델의 한계를 극복하고 성능 및 안정성을 향상시키는 것을 목표로 합니다. Equivalent Convolutional Kernel과 협업 그라디언트 접근법은 해석 가능성을 높이고 불확실성 추정을 개선합니다.

- **Technical Details**: MHEX+는 MHEX 프레임워크를 확장한 것으로, 모든 U-Net 아키텍처에 적용 가능하도록 설계되었습니다. 이 연구에서 제안한 Equivalent Convolutional Kernel은 연속적인 합성곱 레이어를 통합하여 해석 가능성을 높이며, 협업 그라디언트 방법을 통해 디코더 레이어 간의 기울기 일관성을 측정하여 불확실성 추정을 합니다.

- **Performance Highlights**: EU-Nets는 실험에서 모든 네트워크와 데이터셋에 대해 평균 정확도를 1.389% 개선하고 분산을 0.83% 줄였습니다. MHEX+ 방식은 불과 0.1M 미만의 파라미터로 성능을 향상시킵니다. 이러한 성과는 MHEX+ 프레임워크가 U-Net 구조의 해석 가능성과 불확실성 처리를 더욱 강화하는 데 기여함을 보여줍니다.



### Bayesian Optimization for Controlled Image Editing via LLMs (https://arxiv.org/abs/2502.18116)
Comments:
          8 figures

- **What's New**: BayesGenie는 대형 언어 모델(LLMs)과 베이지안 최적화(Bayesian Optimization)를 통합하여 사용자가 이미지 편집을 쉽고 정밀하게 수행할 수 있도록 돕는 최신 접근법을 제안합니다. 기존의 수작업 영역 표시 없이 자연어로 이미지를 수정할 수 있는 기능을 제공하며, 원본 이미지의 의미적 일관성을 유지합니다. 이는 모델 미세 조정 또는 대규모 사전 훈련 없이도 다양한 LLM에 대해 뛰어난 적응성을 보여주는 모델 불가지론적(design) 설계를 특징으로 합니다.

- **Technical Details**: BayesGenie는 LLM의 의미 이해 능력을 활용하여 사용자의 요구에 따라 세부적인 프롬프트를 생성한 후 이를 스테이블 디퓨전(Stable Diffusion) 모델에 전달하여 정확한 이미지 수정을 지원합니다. 베이지안 최적화를 통해 매개변수 공간을 체계적으로 탐색하여 최적의 편집 품질을 달성합니다. 이 과정에서 ‘text_cfg_scale’와 ‘image_cfg_scale’와 같은 주요 매개변수를 조정하면서 결과물과 사용자 요구 사항 간의 정렬을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 BayesGenie는 편집 정확도 및 의미 보존 측면에서 기존 방법들을 크게 능가하는 성능을 입증하였습니다. Claude3와 GPT-4와 같은 다양한 LLM을 활용하여 프레임워크의 효과성을 검증하였으며, 이미지 수정의 직관성과 정확성을 향상시켰음을 보여줍니다. 실험 결과는 BayesGenie가 즉시 배치 가능하며 여러 시나리오에서 높은 품질의 결과를 제공할 수 있음을 시사합니다.



### The Built-In Robustness of Decentralized Federated Averaging to Bad Data (https://arxiv.org/abs/2502.18097)
Comments:
          Funding: SoBigData PPP (101079043), this http URL (PNRR IR0000013), FAIR (PNRR PE00000013), RESTART (PNRR PE00000001)

- **What's New**: 이번 연구는 분산형 연합 학습(DFL)이 저품질 또는 부패된 데이터와 어떻게 상호작용하는지를 심층적으로 조사합니다. 특히, 각 노드에서 로컬 데이터 분포에 대한 평가의 어려움과 중앙집중식 관리자 부재 시 발생할 수 있는 위험 요소들을 다룹니다. 이 연구는 두 가지 데이터 품질 저하 시나리오를 시뮬레이션하여, 분산형 학습이 저품질 데이터의 영향을 얼마나 잘 견딜 수 있는지를 탐구합니다.

- **Technical Details**: DFL 시스템은 여러 노드가 참여하여 협업적으로 모델을 학습할 수 있지만, 그 과정에서 데이터 품질의 변동이 영향을 미칠 수 있습니다. 본 연구에서는 FedAvg의 분산형 구현을 기반으로 하여, 부패된 데이터가 네트워크 내 특정 노드에 고르게 분포되거나 하나의 노드에 집중되는 경우를 시뮬레이션했습니다. 불완전한 데이터를 생성하기 위해 미리 훈련된 Generative Adversarial Network (GAN)에서 잠재 공간 내에서 보간 기법을 사용합니다.

- **Performance Highlights**: 연구 결과, 평균 기반의 분산 학습이 국소적인 부정확한 데이터에 대해 놀라울 정도로 강인함을 보였습니다. 특히, 부패된 데이터가 단일 노드에 집중될 경우, 그 노드의 중앙성에 관계없이 모델의 전반적인 학습 과정에 미치는 영향이 최소화되는 경향을 보였습니다. 또한, 분산형 학습이 중앙 집중화된 학습보다 데이터 부패에 더 강한 회복력을 보이며, 이는 추가적인 비교 분석을 필요로 한다는 점을 강조합니다.



### Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning (https://arxiv.org/abs/2502.18080)
- **What's New**: 최근 연구에 따르면, Chain of Thoughts (CoT)의 길이를 늘리는 것이 LLMs의 복잡한 추론 작업에서 성능 향상에 기여할 수 있다는 점이 주목받고 있습니다. 그러나 본 논문에서는 CoT 길이를 지나치게 늘리는 것이 LLM의 추론 성능에 부정적인 영향을 미칠 수 있다는 우려를 제기합니다. 특정 도메인에서는 CoT 길이가 길어질수록 오히려 성능이 저하되는 경향이 관찰되었습니다.

- **Technical Details**: 연구진은 전체 모델 학습 과정에서 다양한 응답 길이 분포를 가진 소수의 시드 데이터 세트를 활용하여 모델이 깊이 있는 사고를 위한 다양한 추론 노력을 채택하도록 학습을 진행했습니다. 이후 모델은 주어진 문제에 대해 각기 다른 추론 노력을 기반으로 최적의 짧은 정답 응답을 선택하여 자기 개선(Self-improvement)을 시도합니다. 이 과정에서 Thinking-Optimal Scaling 전략(TOPS)이 도입되어, LLM이 문제를 해결하기 위해 필요한 토큰 수를 스스로 결정할 수 있도록 합니다.

- **Performance Highlights**: 이 연구의 결과로 Qwen2.5-32B-Instruct를 기반으로 한 자기 개선 모델이 다양한 수학 벤치마크에서 기존의 32B o1-like 모델을 초월하는 성과를 달성했습니다. 특히, GSM8K, MATH500 및 AIME2024를 포함한 여러 벤치마크에서 우수한 성능을 나타내어, QwQ-32B-Preview와 유사한 성능을 기록했습니다. 이는 다른 모델보다 적절한 추론 노력을 강조하는 개인화된 접근 방식의 중요성을 강조합니다.



### MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration (https://arxiv.org/abs/2502.18072)
- **What's New**: 이 논문에서는 다중 로봇 작업 계획(Multi-Robot Task Planning)과 협업 문제를 해결하기 위한 새로운 알고리즘, Multi-Robot Behavior Tree Planning (MRBTP)을 제안합니다. MRBTP는 이론적으로 안전성(soundness)과 완전성(completeness)을 보장하며, 서로 다른 Behavior Tree (BT) 간의 이종 행동(heterogeneous actions)을 조정하기 위한 크로스 트리 확장(cross-tree expansion) 기능을 포함하고 있습니다. 또한, 동질적 행동(homogeneous actions)을 위한 백업 구조(backup structures)를 유지하여 로봇 팀의 목표 달성을 돕습니다.

- **Technical Details**: MRBTP는 다양한 액션 스페이스(action spaces)를 조정하는 복잡성으로 인해 다중 로봇 BT 계획 알고리즘 개발이 어려운 현황을 해결합니다. 이 알고리즘은 동질적 및 이종 로봇 팀을 위한 BT를 생성할 수 있으며, 대규모 언어 모델(Large Language Models, LLMs)이 사용될 때 목표 관련 행동을 유추하는 선택적 플러그인(plugin)을 제안합니다. 이를 통해 계획 속도(plan speed)와 협업 효율성(collaboration efficiency)을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: MRBTP는 창고 관리(warehouse management) 및 일상 서비스 시나리오에서 평가되었으며, 다양한 환경에서 강력한 실행 효율성(execution efficiency)을 나타났습니다. 또한, 미리 훈련된 LLM이 MRBTP를 위해 효과적인 태스크 전용 서브 트리(task-specific subtrees)를 생성할 수 있음을 보여주었습니다. 결과적으로, MRBTP는 다양한 설정 하에서도 강력함과 효율성을 입증하며, 다중 로봇 협업의 새로운 가능성을 열어줍니다.



### HEROS-GAN: Honed-Energy Regularized and Optimal Supervised GAN for Enhancing Accuracy and Range of Low-Cost Accelerometers (https://arxiv.org/abs/2502.18064)
Comments:
          AAAI Oral; AI for Sensors; Generative Deep Learning

- **What's New**: 본 논문에서는 저비용 가속도계의 정확도와 범위를 개선하기 위해 고안된 HEROS-GAN(하이온 에너지 정규화 및 최적 감독 GAN)을 제안합니다. 이 GAN은 저비용 센서 신호를 고비용 신호로 변환하여 정밀도와 범위의 제한을 극복하며, 최적 수송 이론(Optimal Transport Theory)을 활용하여 unpaired 데이터 간의 일관성을 탐색합니다. 또한, 새로운 Low-cost Accelerometer Signal Enhancement Dataset (LASED)를 구축하여 연구 커뮤니티에 공개했습니다.

- **Technical Details**: HEROS-GAN의 주요 기법으로는 Optimal Transport Supervision(OTS)과 Modulated Laplace Energy(MLE)가 있습니다. OTS는 비짝짓기 데이터 간의 잠재적 상관관계를 최대한 활용하도록 설계되어 Supervisory information을 극대화합니다. MLE는 모델이 범위 제한을 극복하고 세부 신호를 강화할 수 있도록 적절한 에너지를 주입합니다.

- **Performance Highlights**: 실험 결과, HEROS-GAN은 OTS 또는 MLE만 사용한 기존 SOTA(signal enhancement) 방법을 크게 초과하는 성능을 보임을 입증했습니다. OTS와 MLE를 통합한 HEROS-GAN은 가속도계의 범위를 두 배로 늘리고, 신호 잡음은 두 자릿수로 줄이는 놀라운 결과를 달성하여, 가속도계 신호 처리의 벤치마크를 수립했습니다.



### VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion (https://arxiv.org/abs/2502.18042)
- **What's New**: 이번 논문에서는 VLM-E2E라는 새로운 프레임워크를 제안하여 자율주행 시스템의 성능을 향상시키고자 하였습니다. 이 프레임워크는 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 시멘틱(semantic) 정보와 텍스트 표현을 Bird's-Eye-View (BEV) 특성에 통합함으로써, 운전자의 주의(attention) semantics를 명시적으로 포착합니다. 또한, BEV-Text 학습 가능한 가중치 융합 전략을 도입하여 시각 및 텍스트 정보를 효과적으로 활용하는 방법을 제안합니다.

- **Technical Details**: VLM-E2E는 E2E 자율주행 모델과 BEV 모듈의 통합을 통해 시각과 텍스트 특성이 결합된 풍부한 공간 인식을 가능하게 합니다. BEV-Text 가중치 융합 방식을 통해 각 모달리티의 중요성을 동적으로 조절하여, 상황에 맞는 시각 또는 텍스트 특성을 강조할 수 있습니다. 이 과정에서 운전 장면에 대한 텍스트 설명을 생성하고, 이를 통해 얻은 텍스트 정보를 CLIP 모델을 사용하여 밀집 표현으로 변환합니다.

- **Performance Highlights**: VLM-E2E는 nuScenes 데이터셋에서 평가될 때 기존 방법들보다 우수한 성능을 보였으며, 복잡한 주행 시나리오에서의 처리 능력을 크게 향상시켰습니다. 이 연구는 시각적 정밀성과 고수준 시멘틱 추론을 통합하여 보다 안전하고 해석 가능한 자율주행을 가능하게 합니다. VLM-E2E의 성능 개선은 E2E 자율주행 시스템의 결함을 극복하는데 기여할 수 있음을 보여줍니다.



### AutoCas: Autoregressive Cascade Predictor in Social Networks via Large Language Models (https://arxiv.org/abs/2502.18040)
Comments:
          12 pages

- **What's New**: 이 논문은 정보 전파에서의 인기 예측(Popularity prediction) 문제를 다루고 있으며, 기존의 방법들과 차별되는 부분으로 LLM(대규모 언어 모델)을 활용하여 정보 카스케이드(cascade)의 인기 예측을 위해 설계된 AutoCas라는 신모델을 소개합니다. AutoCas는 카스케이드 데이터를 토큰화하여 순차 모델링 원칙에 맞추고, 카스케이드 전파를 자기 회귀 모델링(task)으로 재구성함으로써 LLM의 아키텍처 장점을 최대한 활용합니다.

- **Technical Details**: AutoCas는 정보 카스케이드의 복잡한 지역(topology) 구조와 확산 맥락(diffusion context)을 이해하고, 이를 위해 카스케이드 데이터를 토큰으로 변환해 LLM 아키텍처와 호환되도록 시스템을 구축합니다. 또한, 객관성 있는 예측을 위해 프로프트 학습(prompt learning) 기법을 도입하여 LLM과 카스케이드 예측 간의 시너지를 극대화합니다. 이러한 접근 방식은 카스케이드의 전파 과정을 자기 회귀 모델링으로 reformulate하여, 기존의 방법들과는 다른 연속적인 관찰 시점에서도 재교육 없이 예측이 가능하도록 돕습니다.

- **Performance Highlights**: 대규모의 실제 데이터셋을 활용한 실험 결과, AutoCas는 기존 최첨단 모델들에 비해 카스케이드 인기 예측에서 월등한 성능을 보여줍니다. LLM의 스케일링 특성을 통해 AutoCas는 새로운 관찰 시간이 추가되더라도 굳이 재교육할 필요 없이 유연하게 대처할 수 있습니다. 이 연구는 정보 카스케이드 모델링 분야에서 LLM 기반의 최초의 프레임워크로서 의미 깊은 기여를 하고 있습니다.



### ExPath: Towards Explaining Targeted Pathways for Biological Knowledge Bases (https://arxiv.org/abs/2502.18026)
Comments:
          Under review

- **What's New**: 이번 논문에서는 생물학적 지식 기반에 실험 데이터를 통합하여 더 정확한 경로 추론(pathway inference)을 제공하는 새로운 프레임워크인 ExPath를 제안합니다. ExPath는 특히 아미노산 서열(AA-seqs)을 사용하여 바이오 네트워크(bio-networks)의 다양한 그래프를 분류하는 방식을 채택합니다. 경로 확인과 관련하여 분류에 기여하는 링크를 목표로 다루며, 이러한 경로는 더욱 구체적이고 타겟화된 정보로서 활용될 수 있습니다.

- **Technical Details**: ExPath는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 대형 단백질 언어 모델(pLM)이 아미노산 서열을 그래프에 인코딩하여 전통적인 데이터 처리 문제를 극복합니다; (2) PathMamba라는 하이브리드 아키텍처를 사용하여 그래프 신경망(GNNs)과 상태 공간 시퀀스 모델링(Mamba)을 결합, 로컬 상호작용과 전반적인 경로 수준 종속성을 동시에 학습합니다; (3) PathExplainer는 훈련 가능한 경로 마스크를 통해 기능적으로 중요한 노드와 엣지를 식별하는 서브그래프 학습 모듈입니다.

- **Performance Highlights**: ExPath의 실험 결과는 301개의 바이오 네트워크 평가를 포함하며, 이 평가에서 ExPath에 의해 추론된 경로가 생물학적으로 의미 있는 결과를 유지함을 보여줍니다. 또한, 우리는 기계 학습(ML) 중심의 생물학적 평가 방법과 새로운 지표를 제안하여 실제 실험 데이터가 어떻게 해석될 수 있는지를 명확히 하였습니다. 곧 301개의 큐레이션된 바이오 네트워크 데이터셋도 공개할 예정입니다.



### AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages (https://arxiv.org/abs/2502.18020)
- **What's New**: 이 논문에서는 다국어 모델을 위한 새로운 하이브리드 지식 증류(knowledge distillation) 접근법을 제안합니다. 기존의 방법들이 다국어 모델, 특히 저자원(low-resource) 언어에 대한 성능 유지를 어렵게 하는 문제를 해결하기 위해, 전통적인 지식 증류 방식과 단순화된 attention matching 메커니즘을 결합했습니다. 이로 인해, 전통적인 다국어 모델보다 유의미하게 작은 크기의 학생 모델 구조를 도입하여, 아프리카 언어 다섯 개에 대한 평가에서 효과를 입증했습니다.

- **Technical Details**: 우리가 제안한 하이브리드 증류 프레임워크는 지식 증류와 attention matching을 결합하여 학생 모델이 교사 모델의 출력 분포와 내부 attention 패턴을 모두 학습하도록 합니다. 특히, 매우 compact한 다국어 학생 모델을 설계하였으며, 이는 기존 모델보다 훨씬 작은 hidden dimension을 가집니다. 실험적으로, 아프리카어 저자원 언어인 Kinyarwanda, Swahili, Hausa, Igbo, Yoruba에 대해 이 접근법을 평가하고, 모델 크기를 85% 이상 줄이면서도 경쟁력 있는 성능을 달성했습니다.

- **Performance Highlights**: 이 연구의 실험 결과는 제안된 하이브리드 접근법이 교사 모델과 비교하여 성능 면에서 경쟁력을 유지함을 보여줍니다. 학생 모델은 원래 모델의 성능에서 85% 이내의 정확도를 유지하면서도, 연산 자원을 현저히 절감할 수 있었습니다. 이는 저자원 환경에서 다국어 모델을 배치하는 데 유용한 실용적인 프레임워크를 제공하며, 아프리카 언어와 관련된 응용 프로그램에서 큰 혜택을 제공합니다.



### ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents (https://arxiv.org/abs/2502.18017)
- **What's New**: 이번 논문에서는 시각적으로 풍부한 문서에서 정보를 이해하는 데 있어 기존의 Retrieval-Augmented Generation (RAG) 방법의 한계를 극복하기 위한 새로운 데이터셋인 ViDoSeek을 소개합니다. ViDoSeek은 복잡한 추론을 요구하는 문서에서 RAG 성능을 평가하도록 설계되었습니다. 이를 통해 현재 RAG 접근 방식의 주요 한계도 식별하였으며, 특히 시각적 검색 방법과 추론 토큰 할당의 부족 문제가 강조되었습니다.

- **Technical Details**: ViDoRAG라는 새로운 다중 에이전트 RAG 프레임워크를 제안하여 시각적 문서에서 복잡한 추론을 위한 개선점을 제시합니다. 이 프레임워크는 Gaussian Mixture Model (GMM) 기반의 하이브리드 전략을 채택하여 다중 모달 검색(multi-modal retrieval)을 효과적으로 처리합니다. 또한 탐색, 요약, 반영 과정을 포함하는 반복적인 에이전트 워크플로우를 통해 모델의 추론 능력을 배양합니다.

- **Performance Highlights**: ViDoSeek에서 진행된 광범위한 실험 결과, ViDoRAG는 기존 방법들보다 10% 이상 성능이 향상된 것으로 나타났습니다. 이 결과는 ViDoRAG이 RAG 분야에서 테스트 시간 확장(test-time scaling) 연구에 대한 새로운 방향을 제공함을 나타냅니다. ViDoRAG은 경쟁적인 ViDoSeek 기준에서 기존 방법들과 비교하여 우수한 성능을 보였습니다.



### NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms (https://arxiv.org/abs/2502.18008)
- **What's New**: NotaGen이라는 새로운 상징적 음악 생성 모델이 소개되었습니다. 이 모델은 160만 곡의 음악을 사전 학습하고, 9천 곡의 고품질 클래식 작품에 대해 'period-composer-instrumentation' 프롬프트를 조건으로 미세 조정되었습니다. 또한, CLaMP-DPO 방식을 통해 인류의 피드백 없이 생성 품질과 제어 가능성을 향상시킵니다. NotaGen은 인간 작곡에 대한 A/B 테스트에서 기초 모델보다 뛰어난 성능을 보였습니다.

- **Technical Details**: NotaGen은 수정된 ABC 표기법을 채택하여 서로 다른 음성을 정렬된 형식으로 표현합니다. 모델 아키텍처는 바 스트림 패칭과 Tunesformer를 기반으로 하며, 패치 수준 디코더와 문자 수준 디코더로 구성되어 있습니다. 사전 훈련을 통해 다양한 장르와 악기에서 기본적인 음악 구조와 패턴을 포착합니다. 데이터 정리를 통해 고품질의 음악 데이터셋을 확보하였습니다.

- **Performance Highlights**: CLaMP-DPO 방법을 활용한 NotaGen은 상징적 음악 생성 모델의 다양한 아키텍처와 인코딩 체계에서 음악성과 제어 가능성을 획기적으로 향상시킵니다. 이는 표준화된 음악 생성 및 시간적으로 배열된 패치를 통해 성과를 높이며, 인간의 주관적 평가에서도 우수한 성과를 입증했습니다. 이로써 상징적 음악 생성 모델의 새로운 가능성을 보여주고 있습니다.



### Radon-Nikodým Derivative: Re-imagining Anomaly Detection from a Measure Theoretic Perspectiv (https://arxiv.org/abs/2502.18002)
- **What's New**: 이 논문에서는 효과적인 이상 탐지 손실 함수 설계를 위한 핵심 원리로 Radon–Nikodým 정리를 제안합니다. 이를 통해 기존의 손실 함수에 Radon–Nikodým 도함수를 곱함으로써 성능을 크게 향상시킬 수 있음을 보여줍니다. 우리는 이 새로운 손실 함수를 RN-Loss라고 명명하고, PAC 학습 가능성(PAC learnability) 관점에서 이를 검증하였습니다.

- **Technical Details**: RN-Loss는 여러 형태의 데이터 분포(예: Weibull, Log-normal)에 효과적으로 적응하며, 기본적인 손실 함수(예: Binary Cross-Entropy)를 기반으로 계산 효율성을 유지합니다. 이 손실 함수는 클러스터링 알고리즘과 결합될 때도 우수한 성능을 발휘하며, 단순한 모델을 사용하여 정상 데이터로만 학습할 경우에도 이전에 보지 못한 이상치를 탐지하는 데 도움이 됩니다.

- **Performance Highlights**: RN-Loss는 다양한 평가 기준에서 최첨단 방법들을 초월하는 성능을 보여주었습니다. 다변량 데이터셋의 68%에서 F1 점수 증가를 달성하였고, 단일 변량 시계열 데이터 셋에서는 72%에서 최고의 F1 점수를 기록했습니다. 이 알고리즘은 기존의 CBLOF 및 ECBLOF 알고리즘에 비해 향상된 성능을 보이며, 특히 K-Means와 결합했을 때 93%의 단일 변량 데이터셋에서 우수한 성과를 거두었습니다.



### MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification (https://arxiv.org/abs/2502.17987)
- **What's New**: 이 논문에서 새로운 점은 저자들이 저자원 (low-resource) 반투 (Bantu) 언어에 특화된 텍스트 분류 모델 MAGE(Multi-Head Attention Guided Embeddings)를 소개한 것이다. 이 모델은 Language-Independent Data Augmentation (LiDA)을 통해 데이터 포인트를 선택적으로 향상시킴으로서 텍스트 분류 성능을 개선하는데 중점을 두고 있다. 특히, MAGE는 데이터 부족 문제를 해결하면서도 반투 언어의 고유한 구문론적(syntactic) 및 의미론적(semantic) 특성을 효과적으로 다루도록 설계되었다.

- **Technical Details**: MAGE는 LiDA 프레임워크를 기반으로 하여 중대한 혁신을 도입하고 있다. 전통적인 Denoising Autoencoder 대신 Variational Autoencoder (VAE)를 도입하여 더욱 표현력이 풍부하고 다양한 합성(augmented) 임베딩을 생성한다. 또한, Multi-Head Attention 메커니즘을 활용하여 임베딩에서 중요한 특징을 강조함으로써 저자원 언어에서 구문론적 및 의미론적 뉘앙스를 더 잘 포착할 수 있도록 한다.

- **Performance Highlights**: MAGE는 AfriSenti SemEval 데이터셋을 사용하여 감정 분류(sentiment classification) 성능 평가를 수행하였으며, 저자원 환경에서 기존의 기준 방법들보다 우수한 성능을 나타냈다. MAGE는 데이터 부족 문제를 해결함과 동시에 다른 저자원 언어 계열로의 텍스트 분류 능력을 확장할 수 있는 스케일러블한 프레임워크로 자리잡고 있다. 이 연구는 향후 저자원 언어 처리 및 분류 작업에 대한 연구의 기초를 제공하고 있으며, NLP 기술의 포괄성과 일반화 가능성을 높이는 방향으로 나아가고 있다.



### Broadening Discovery through Structural Models: Multimodal Combination of Local and Structural Properties for Predicting Chemical Features (https://arxiv.org/abs/2502.17986)
- **What's New**: 이 연구에서는 원자와 그 주변 환경에 대한 정보를 담고 있는 화학 지문(fingerprints)을 기반으로 하는 새로운 언어 모델을 개발하고자 합니다. 제안하는 방법론은 RoBERTa를 언어 모델로 사용하고 GIN, GCN, Graphormer와 같은 그래프 모델을 결합하여 bimodal architecture를 구성합니다. 이러한 통합된 접근법을 통해 화학 분자의 구조를 보다 효율적으로 예측할 수 있는 가능성을 제시합니다.

- **Technical Details**: Extended-Connectivity fingerprints(ECFP)는 각 분자에 대해 2차원 해시 배열을 할당하여 물리적, 화학적 특성을 암호화합니다. 이러한 표현은 그래프 신경망(Graph Neural Networks)과 자연어 처리(NLP) 모델의 결합을 통해 사용하여, 전통적인 방법에 비해 예측 성능이 향상될 수 있는 기회를 제공합니다. 또한, 이 모델은 화학 환경의 정의에 중요한 역할을 하는 분자 NMR 스펙트럼의 예측과 같은 작업에 유용합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 Quantitative Structure-Activity Relationship(QSAR) 및 핵자기공명(Nuclear Magnetic Resonance) 스펙트럼 예측 등의 특정 작업에서 기존의 방법 대비 상당한 성능 향상을 보여주었습니다. 이러한 성능 증가는 기존 접근 방식을 통해서는 발견하지 못했던 특정 화학적 관계와 물리적 속성들 간의 상관관계를 모델이 효과적으로 학습했음을 시사합니다.



### LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena (https://arxiv.org/abs/2502.17967)
- **What's New**: 이번 연구에서는 Agent Trading Arena라는 새로운 가상 환경을 소개하여 대규모 언어 모델(LLMs)의 수치적 추론 능력을 향상시키기 위한 방법을 제시합니다. 이 환경은 제로섬 게임(Zero-Sum Game) 방식으로 설계되어 에이전트들이 주식 포트폴리오에 투자하여 복잡한 경제 시스템을 시뮬레이션하는데, 이는 실제 상황에서의 문제 해결 능력을 더 잘 평가할 수 있게 해줍니다. 또한, 시각적 데이터를 사용할 때 LLMs의 수치적 추론 능력이 더 우수하다는 점을 강조합니다.

- **Technical Details**: Agent Trading Arena는 자산 가격 결정이 에이전트 간의 상호작용에 기반하여 자체적으로 이루어지도록 설계되었습니다. 외부 지식의 영향을 줄이기 위해 가격은 입찰-매도 시스템에 따라 결정되며, 모든 에이전트의 활동은 단기적인 최적 전략에 의존하지 않고 자율적으로 동적으로 변화합니다. 이러한 시스템은 에이전트가 시각적 표현을 통해 복잡한 데이터 관계를 추론하고 보다 전략적으로 의사 결정을 내릴 수 있도록 돕는 반사 모듈(Reflection Module)을 포함합니다.

- **Performance Highlights**: 실험 결과, LLMs는 텍스트 형식의 수치적 데이터에서 성능이 떨어지는 반면, 시각적으로 표현된 데이터에서는 월등한 성과를 보였습니다. 시각적 데이터로 처리된 경우, LLMs는 실험에서 높은 수익률(Return Rate)을 보여주어 구조화된 시각적 표현의 이점을 입증했습니다. 또한, NASDAQ 주식 데이터셋을 사용하여 LLM의 시각적 데이터 처리 능력이 텍스트 기반 데이터 처리보다 뛰어나다는 점을 확인했습니다.



### Language Models' Factuality Depends on the Language of Inquiry (https://arxiv.org/abs/2502.17955)
- **What's New**: 이번 연구에서는 다국어 언어 모델(Multilingual Language Models, LMs)의 한계를 체계적으로 조사하기 위해 13개 언어에서 10,000개의 국가 관련 사실을 포함하는 데이터셋을 소개합니다. 연구진은 사실 기억(Factual Recall)과 지식 전이(Knowledge Transferability)를 측정하기 위해 세 가지 새로운 지표를 제안했습니다. 이 연구는 LMs가 언어별 특정 사실을 일관되게 기억하지 못하는 경향이 있음을 밝혀냈습니다.

- **Technical Details**: 연구에서는 LMs의 사실 기억을 평가하기 위해 Factual Recall Score (FRS), Knowledge Transferability Score (KTS), Cross-Lingual Factual Knowledge Transferability Score (X-FaKT)라는 세 가지 지표를 개발했습니다. FRS는 모델이 특정 언어에서 사실을 얼마나 정확하게 기억하는지를 측정하며, KTS는 언어 간 지식의 전이 정도를 정량화합니다. 이러한 지표들은 다국어 모델의 성능을 심층적으로 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, LMs는 언어별로 관련된 사실을 정확히 기억하는 반면, 이 지식을 다른 언어로 전이하는 데 어려움을 겪음을 보여주었습니다. LLM의 크기가 사실성과 지식 전이에 중요한 역할을 하며, 언어 자원에 따라 성능 차이가 두드러지게 나타났습니다. 이러한 결과는 다국어 AI 개발에 있어 LMs가 언어 간 지식 통합의 한계를 갖고 있음을 시사합니다.



### Robust Polyp Detection and Diagnosis through Compositional Prompt-Guided Diffusion Models (https://arxiv.org/abs/2502.17951)
- **What's New**: 본 논문에서는 Progressive Spectrum Diffusion Model (PSDM)을 제안하여 다양한 임상 주석을 통합함으로써 합성 폴립 이미지를 생성하는 접근 방식을 소개합니다. 이 모델은 세그멘테이션 마스크, 바운딩 박스 및 대장내시경 보고서와 같은 다양한 정보를 조합하여 정교한 촉진(prompt)을 생성합니다. 이를 통해 모델이 넓은 공간 구조와 세부 정보를 효과적으로 캡쳐할 수 있도록 합니다.

- **Technical Details**: PSDM은 coarse와 fine 컴포넌트로 구성된 촉진 구조를 따라 다양한 임상 주석을 통합합니다. 이 방법은 원래의 세그멘테이션 마스크에서 제공되는 정보에 덧붙여 폴립 크기, 형태 및 조직학적 특성에 대한 텍스트 설명을 활용하여 출력 품질을 향상시킵니다. 이러한 접근 방식은 catastrophic forgetting 문제를 해결하기 위해 연속 학습(continual learning) 기법을 적용하여 모든 촉진으로부터 이미지를 생성할 수 있도록 합니다.

- **Performance Highlights**: PSDM을 통해 훈련 데이터를 증강함으로써 폴립 분류, 탐지 및 세그멘테이션 성능이 크게 향상되었습니다. PolypGen 데이터셋에서 PSDM은 F1 점수를 2.12% 증가시키고 평균 평균 정밀도를 3.09% 향상시켜 OOD 상황에서의 뛰어난 성능을 입증했습니다. 이러한 개선은 다양한 임상 시나리오에서의 일반화 가능성을 높이는 데 기여합니다.



### DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in Bilingual Complex Ophthalmology Reasoning (https://arxiv.org/abs/2502.17947)
Comments:
          29 pages, 4 figures, 1 table

- **What's New**: 이번 연구에서는 DeepSeek-R1과 최근에 출시된 세 가지 대형 언어 모델(LLMs)의 복합한 이중 언어 안과 사례에 대한 정확성과 추론 능력을 평가했습니다. 130개의 다지선다형 질문(MCQs)이 수집되어 중국 및 영어로 번역되었으며, 각 모델의 응답을 비교하였습니다. 이러한 접근을 통해 최신 LLM의 역량을 심도 있게 분석할 수 있는 기회를 제공하였습니다.

- **Technical Details**: 연구에 사용된 MCQs는 진단(n = 39) 및 관리(n = 91) 관련 질문으로 구성되어 있으며, 총 6개 주제로 분류되었습니다. DeepSeek-R1의 응답은 2025년 2월 15일부터 20일 사이에 기본 설정 하에 생성되었고, 정확도는 올바르게 대답한 질문의 비율로 계산되었습니다. 추론 능력은 추론 논리의 분석과 오류 원인을 통해 평가되었습니다.

- **Performance Highlights**: DeepSeek-R1은 전체적으로 가장 높은 정확도를 기록하였으며, 중국어 MCQs에서 0.862, 영어 MCQs에서 0.808의 정확도를 보였습니다. 다른 LLM 모델들은 중국어에서 각각 0.715, 0.685, 0.692의 정확도를 기록하여 DeepSeek-R1과 비교해 통계적으로 유의미한 차이를 보였습니다. 추론 오류의 주요 원인은 주요 긍정 이력을 무시하거나, 잘못된 의료 데이터를 해석하는 등의 것으로 나타났습니다.



### Optimal Brain Apoptosis (https://arxiv.org/abs/2502.17941)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문은 Convolutional Neural Networks (CNN)와 Transformers의 계산 효율성과 자원 요구 사항 문제를 다루기 위해 Optimal Brain Apoptosis (OBA)라는 혁신적인 pruning 방법을 제안합니다. 기존의 Hessian 행렬을 근사하는 방법 대신, 각 매개변수에 대한 Hessian-vector 곱을 직접 계산하여 매개변수 중요성을 추정합니다. 이 접근 방식은 Neural Network의 파라미터를 보다 정밀하게 다룰 수 있도록 하며, 다양한 데이터셋에서 CNN과 Transformers의 실험을 통해 검증되었습니다.

- **Technical Details**: OBA 방법은 매개변수의 2차 Taylor 전개를 효율적으로 계산하는 것을 목표로 합니다. 논문에서는 네트워크의 다양한 층 간 Hessian 하위 행렬을 분해하고, 이들 간의 비어있지 않은 조건을 식별하는 과정을 제시합니다. 이러한 방법을 통해 OBA는 구조적 및 비구조적 pruning 작업 모두에서 활용될 수 있을 뿐 아니라, 매개변수의 중요성을 정량화하는 데 더 정확한 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과, OBA는 VGG19, ResNet32, ResNet50, ViT-B/16 모델에서 CIFAR10, CIFAR100, Imagenet 데이터셋을 통해 기존 pruning 방법에 비해 성능 저하 없이 계산 효율성을 크게 향상시켰습니다. 이 연구는 효율적인 매개변수 중요성 추정의 새로운 접근법을 통해 deep learning 모델의 경량화와 최적화를 가능하게 했습니다. 논문에서 제공하는 코드를 통해 다른 연구자들도 이 방법을 쉽게 구현할 수 있도록 하고 있습니다.



### Integrating Boosted learning with Differential Evolution (DE) Optimizer: A Prediction of Groundwater Quality Risk Assessment in Odisha (https://arxiv.org/abs/2502.17929)
Comments:
          9 Figures (8 figs in paper and one additional graphical abstract), 9 Tables

- **What's New**: 본 연구에서는 머신 러닝 기반의 예측 모델인 LCBoost Fusion을 개발하여 지하수의 품질 지수(Groundwater Quality Index, GWQI)를 평가하고 주요 오염 물질을 식별하였습니다. LCBoost Fusion은 CatBoost와 LightGBM을 결합한 하이브리드 모델로, Differential Evolution (DE) 최적화를 통해 정확성을 높였습니다. 이 모델의 도입은 지속 가능한 지하수 관리에 기여할 것으로 기대됩니다.

- **Technical Details**: 모델의 성능 평가는 여러 단계를 거쳐 이루어졌으며, RMSE는 0.6829, MSE는 0.5102, MAE는 0.3147, R² 스코어는 0.9809로 각각 측정되었습니다. 특성 중요도 분석을 통해 칼륨(Potassium, K), 플루오라이드(Fluoride, F), 그리고 총 경도(Total Hardness, TH)가 지하수 오염에 가장 큰 영향을 미치는 요소로 평가되었습니다. 이러한 발전은 기계 학습이 지하수 품질 위험을 평가하는 데 어떻게 활용될 수 있는지를 보여줍니다.

- **Performance Highlights**: LCBoost Fusion 모델은 기존의 개별 모델인 CatBoost와 LightGBM보다 높은 예측 정확도를 기록하여, 지하수 질 향상에 기여할 수 있는 가능성을 보여줍니다. 이 연구에서는 오디샤 지역의 지하수를 대상으로 하였으며, 결과는 환경 기관과 정책 입안자들에게 목표 지역의 지속 가능한 관리를 위한 기초 자료로 활용될 수 있습니다. 향후 연구는 원격 탐지 데이터와 상호작용하는 의사결정 시스템을 개발하는 데 중점을 둘 것입니다.



### Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data (https://arxiv.org/abs/2502.17928)
- **What's New**: 이 논문에서는 SIDSL(Structure-prior Informed Diffusion model for Source Localization)이라는 새로운 프레임워크를 소개하고, 이를 통해 제한된 데이터 환경에서 소스 로컬라이제이션(source localization)의 세 가지 주요 문제를 해결합니다. 주목할 만한 문제는 알려지지 않은 전파 패턴, 복잡한 토폴로지-전파 관계 및 소스와 비소스 노드 간의 클래스 불균형입니다. 이러한 객체들을 해결하기 위해 SIDSL은 그래프 레이블 전파(graph label propagation)를 통한 토폴로지 인식(prior) 사전 정보를 통합하여 정확한 소스 예측을 지원합니다.

- **Technical Details**: SIDSL은 입력으로 그래프의 토폴로지와 관측된 노드 상태를 사용하며, 구조 기반의 사전 정보를 안내하는 조건부 디노이징(diffusion) 모델을 사용하여 소스 노드 분포를 예측합니다. 이를 통해 노드의 감염 경로를 추적하는 레이블 전파와 효율적인 토폴로지 인식 피쳐 추출을 위한 GNN(parameterized label propagation module, GNN-LP)을 결합하여 학습 효율성을 극대화합니다. 이러한 방법들은 실험적으로 네 가지 실제 데이터 세트에서 SIDSL이 최첨단 기법들보다 더 우수한 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: SIDSL은 네 가지 실제 데이터셋에서 F1 스코어가 7.5-13.3% 향상되는 성과를 기록하였으며, 합성 패턴의 시뮬레이션 데이터로 사전 훈련을 시행했을 때, 훈련 데이터의 10%만으로도 18.8% 이상의 성과를 얻었습니다. 이러한 결과는 SIDSL이 레이블 데이터가 부족한 실제 응용에서도 높은 효과성을 보여주며, 제한된 데이터로도 강력한 일반화 능력을 가진다는 것을 강조합니다. 이 연구는 SIDSL의 샘플 효율성을 향상시키고 훈련 시간을 단축할 수 있는 잠재력을 보여줍니다.



### Decoupled Graph Energy-based Model for Node Out-of-Distribution Detection on Heterophilic Graphs (https://arxiv.org/abs/2502.17912)
Comments:
          The first two authors contributed equally to this work; ICLR 2025

- **What's New**: 본 논문은 그래프 학습에서 OOD(Out-of-Distribution) 노드 탐지에 대한 연구를 다루고 있습니다. 이전의 방법들은 i.i.d. 데이터에 기반하여 설계되었으며, 이러한 접근법은 그래프의 의존성 때문에 직접적으로 적용할 수 없습니다. 저자는 Energy-based Models (EBMs)를 MLE(Maximum Likelihood Estimation)를 통해 훈련하고, heterophily 문제를 해결하기 위한 새로운 접근법인 DeGEM을 제안합니다.

- **Technical Details**: DeGEM은 그래프 인코더와 에너지 헤드 두 가지 구성 요소로 분해됩니다. 그래프 인코더는 Graph Contrastive Learning (GCL) 알고리즘으로 훈련되고, 에너지 헤드는 latent space에서 MLE로 훈련되어 노드의 OOD 점수를 제공합니다. 이러한 방법은 MCMC(Markov Chain Monte Carlo) 샘플링을 latent space에서 보다 효율적으로 수행할 수 있게 하여 계산 비용을 줄입니다.

- **Performance Highlights**: DeGEM은 기존의 최첨단 방법들보다 homophilic 및 heterophilic 그래프 모두에서 성능 향상을 보였습니다. 특히 평균 AUROC(Area Under the Receiver Operating Characteristic curve)에서 homophilic 그래프에서 6.71%, heterophilic 그래프에서는 20.29%의 개선을 보였으며, OOD 노출 없이도 성능이 우수함을 입증하였습니다. 기존 방법들이 heterophilic 그래프에서 낮은 성능을 보이는 반면, DeGEM은 이 문제를 해결하여 확실한 성능 우위를 확보했습니다.



### Enhancing Speech Quality through the Integration of BGRU and Transformer Architectures (https://arxiv.org/abs/2502.17911)
- **What's New**: 이 논문은 소음이 많은 환경에서 음성 신호의 품질 향상을 위한 최신 접근 방식을 제안합니다. Bidirectional Gated Recurrent Units (BGRU)와 Transformer 모델을 통합하여 음성 향상 작업에 효과적으로 사용한 결과를 제시합니다. 이러한 하이브리드 아키텍처는 기존 방법들보다 우수함을 입증하였습니다.

- **Technical Details**: BGRU-Transformer 프레임워크는 시간적 의존성(temporal dependencies)을 포착하고 복잡한 신호 패턴(signal patterns)을 학습하는 데 탁월합니다. 이 모델은 노이즈 감소(noise reduction)를 향상시키고 음성 품질(speech quality)을 개선하는 방향으로 설계되었습니다. 실험 평가를 통해 기존 접근 방식에 비해 상당한 성능 향상(performance gains)을 보여주었습니다.

- **Performance Highlights**: 결과적으로, BGRU와 Transformer의 통합은 시스템의 견고함(system robustness)을 향상시킬 뿐만 아니라 고급 음성 처리 기술(advanced speech processing techniques)로 나아갈 길을 열어줍니다. 이 연구는 음성 향상 기술(speech enhancement technology) 분야의 진전을 위한 기초를 마련하고 있으며, 향후 모델 아키텍처 최적화 및 응용 시나리오 탐색에 기여할 것입니다.



### Scaling LLM Pre-training with Vocabulary Curriculum (https://arxiv.org/abs/2502.17910)
Comments:
          under review

- **What's New**: 현대 언어 모델은 고정된 어휘(vocabulary)에 기반해 훈련되며, 이는 인간 언어 학습에서 관찰되는 적응형 어휘 습득과 대조됩니다. 이를 해결하기 위해 본 연구에서는 어휘 커리큘럼 학습(vocabulary curriculum learning) 접근법을 도입하여 훈련 효율성을 향상시킵니다. 이 방법은 엔트로피(entropy) 기반의 어휘 확장과 모델 최적화를 번갈아 수행함으로써 다양한 토큰화(tokenization) 세분화의 전이 가능한 표현(transferable representations)을 학습할 수 있게 합니다.

- **Technical Details**: 어휘 커리큘럼 학습은 기본 단위(문자)로 시작하여 점진적으로 더 복잡한 표현으로 확장되며, 높은 모델링 엔트로피(regions of high modeling entropy) 영역에 더 많은 용량을 할당하여 복잡한 언어 구조를 이해합니다. 기존 언어 모델의 한계를 극복하기 위해, 본 연구에서는 동적인 어휘 생성을 위한 프레임워크를 제안하고 이 과정에서 예측 가능성에 기반한 토큰 시퀀스를 병합(merge)하는 방식을 사용합니다. 또한, 어휘 업데이트는 목표 어휘 크기를 기준으로 증가 또는 감소할 수 있는 구조로 구성됩니다.

- **Performance Highlights**: 실험은 소규모 GPT 모델에서 수행되었으며, 어휘 커리큘럼 학습 접근법이 기존 고정형 어휘 훈련과 비교하여 낮은 bits-per-character(BPC) 성과를 낼 수 있음을 보여줍니다. 실험 결과, 어휘 커리큘럼을 통해 더 큰 어휘를 효과적으로 활용할 수 있으며, 이 과정에서 모델의 성능이 향상됨을 확인했습니다. 이러한 연구는 언어 모델링에 집중하지만, 점진적인 스케일링 효과가 다른 모달리티(modalities)와 도메인에도 일반화될 수 있을 것으로 기대됩니다.



### FactFlow: Automatic Fact Sheet Generation and Customization from Tabular Dataset via AI Chain Design & Implementation (https://arxiv.org/abs/2502.17909)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 논문은 데이터 분석 기술의 발전과 사용자 친화적인 자동화 도구의 필요성을 다룹니다. 특히, 	ool이라는 새로운 자동 사실 시트 생성 도구를 소개하며, 이 도구는 데이터의 의미를 잘 이해하고 사용자의 요구에 맞는 시나리오를 생성하는 데 중점을 둡니다. 이 시스템은 데이터셋을 시각적으로 매력적인 사실 시트로 변환하고 사용자가 자연어 명령을 통해 시트를 조정할 수 있게 합니다.

- **Technical Details**: FactFlow 시스템은 AI 체인(concept of AI chain)을 적용하여 사용자 요청 및 데이터를 기반으로 사실 시트를 생성합니다. 각 AI 워커는 특정 하위 작업을 효율적으로 처리하도록 설계되어 있으며, 이를 통해 전체적인 결과의 품질이 향상됩니다. 사용자는 새로운 사실을 추가하거나 콘텐츠를 재배열하여 자신만의 사실 시트를 맞춤 설정할 수 있는 기능도 포함되어 있습니다.

- **Performance Highlights**: 18명의 참가자를 대상으로 실시한 사용자 평가 결과, 	ool은 기존 자동 사실 시트 생성 방법들보다 뛰어난 성과를 보였으며, 사용자 맞춤 설정 작업에서도 긍정적인 경험을 제공한 것으로 확인되었습니다. 이러한 효과는 특히 LLM(대형 언어 모델)을 활용하여 자연어 기반으로 작업을 수행하는 데에서 기인합니다.



### Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs (https://arxiv.org/abs/2502.17900)
- **What's New**: 이 논문에서는 최신 심전도(ECG) 표현 학습 기법인 K-MERL을 제안합니다. K-MERL은 자유 형식의 보고서와 ECG 신호를 정렬하여, 구조화된 지식을 추출하여 자가 감독 학습을 향상시킵니다. 특히, K-MERL은 12리드 ECG의 독특한 공간적 및 시간적 특성을 포착하기 위해 리드 특정(tokenization) 기법을 활용합니다.

- **Technical Details**: K-MERL 프레임워크는 일반 목적의 오픈 소스 대형 언어 모델(LLM)을 활용하여 ECG 보고서에서 심장 관련 엔티티를 추출합니다. 또한, 리드 인식 ECG 인코더와 동적 리드 마스킹(dynamic lead masking) 전략을 설계하여 임의의 리드 입력을 처리하면서 리드 특정 공간 및 시간 패턴을 포착할 수 있습니다. 이러한 접근 방식으로 K-MERL은 다양한 리드 조합에서의 성능을 극대화합니다.

- **Performance Highlights**: K-MERL은 6개의 외부 ECG 데이터셋에 대한 평가에서 제로샷 분류(zero-shot classification) 및 선형 프로빙(linear probing) 작업에서 최고의 성능을 달성하였습니다. 기존 방법들에 비해 평균 16% AUC 개선을 달성하며, 부분 리드를 사용하는 제로샷 분류에서 특히 두드러진 성과를 보입니다.



### VeriPlan: Integrating Formal Verification and LLMs into End-User Planning (https://arxiv.org/abs/2502.17898)
Comments:
          In CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan. ACM, New York, NY, USA, 19 pages

- **What's New**: 이번 연구에서는 VeriPlan이라는 시스템을 소개하였으며, 이는 LLM(대형 언어 모델)의 신뢰성과 유연성을 향상시키기 위해 형식 검증 기법인 모델 체크를 적용하였다. VeriPlan은 사용자가 검증 과정에 참여할 수 있도록 하는 규칙 변환기(rule translator), 유연성 조정기(flexibility sliders), 그리고 모델 검사기(model checker)라는 세 가지 핵심 기능을 포함하고 있다. 사용자를 연구(n=12)를 통해 평가한 결과, LLM의 품질 인식, 사용성 및 사용자 만족도가 개선되었음을 보여주었다.

- **Technical Details**: 이번 연구는 LLM에 형식 검증 기법을 적용하여 사용자 중심의 계획 도구로서의 올바른 활용을 목표로 하였다. 특히, 모델 체크(model checking) 기법을 활용하여 LLM의 출력 결과를 사용자 정의 제약 조건과 비교해 검증할 수 있도록 하며, 이를 통해 사용자 통제(User control)와 유연한 적응(flexible adaptation) 기능을 지원하는 방법을 탐구하였다. 디자인 원칙으로는 검증 가능성과 사용자 참여를 통해 검증 과정에서의 실수를 교정할 수 있는 시스템 구조를 제안하였다.

- **Performance Highlights**: 사용자 경험을 향상시키기 위해 VeriPlan은 LLM의 계획 작업에서 출력 품질, 사용자 통제 및 투명성을 개선하였다. 모델 체크는 LLM의 엄격성을 강화하는 한편, 제약 조건의 강도를 조절할 수 있어 유연성과 창의적인 계획을 가능하게 한다. 이를 통해 일반 사용자가 더욱 쉽게 자동화된 계획 도구를 활용할 수 있는 방향을 제시한다.



### Sample-efficient diffusion-based control of complex nonlinear systems (https://arxiv.org/abs/2502.17893)
- **What's New**: 이번 논문은 복잡한 비선형 시스템을 제어하기 위한 새로운 프레임워크인 SEDC(Sample-Efficient Diffusion-based Control)를 제안합니다. SEDC는 샘플 효율성과 신뢰성을 높이기 위해 데이터 기반 제어 방식을 혁신적으로 개선합니다. 제안된 방법은 Decoupled State Diffusion, Dual-Mode Decomposition, Guided Self-finetuning을 통해 기존 방법보다 39.5%-49.4% 더 나은 제어 정확성을 달성하며, 훈련 샘플의 10%만 사용합니다.

- **Technical Details**: 이 연구는 비선형 시스템의 제어 문제를 디노이징 확산 프로세스로 재구성하여 최적의 제어 시퀀스를 샘플링합니다. 이를 통해 고차원 공간의 데이터 희소성 문제를 해결하기 위해 Decoupled State Diffusion(DSD) 접근 방식을 사용하여 물리학 기반 제어를 보장합니다. Dual-Mode Decomposition(DMD)은 시스템 동역학을 계층적 선형 및 비선형 구성 요소로 분해하여 비선형 제어 전략을 배우는 데 도움을 줍니다.

- **Performance Highlights**: SEDC는 세 가지 복잡한 비선형 동적 시스템에서 실험을 통해 기존의 방법과 비교하여 제어 정확성이 39.5%-49.4% 향상됨을 입증했습니다. 에너지 소비와 정확성 사이의 균형을 유지하면서도 훈련 샘플의 10%만으로 최신 성능을 달성합니다. 추가적인 압축 연구는 SEDC의 설계 요소들의 효과성을 확인합니다.



### Arrhythmia Classification from 12-Lead ECG Signals Using Convolutional and Transformer-Based Deep Learning Models (https://arxiv.org/abs/2502.17887)
Comments:
          34 pages, 17 figures

- **What's New**: 이번 연구는 루마니아의 심혈관 문제를 해결하기 위한 혁신적인 진단 방법을 모색합니다. 비용이 제한된 의료 환경에서 신속하고 효율적인 부정맥(arrhythmia) 진단 기술을 개발하는 데 중점을 두고 있습니다. 또한, 루마니아의 공공 의료 데이터가 부족한 점을 고려하여, 국제 공개 데이터셋을 사용하여 시스템을 훈련시켰습니다.

- **Technical Details**: 부정맥 분류 분야에서 일반적으로 사용되는 여러 데이터셋(PTB-XL, PTB 진단 ECG 데이터베이스, 중국 12-Lead ECG 챌린지 데이터베이스 등)을 결합하여 연구를 수행했습니다. 입력 데이터 처리를 위해 Pan-Tompkins 알고리즘 변형을 사용하였고, 이 알고리즘은 ECG 신호에서 QRS 복합체를 효율적으로 탐지하는데 강력한 성능을 보입니다. 기계 학습(machine learning) 기법으로는 1D CNNs, 2D CNNs, ResNet 및 비전 변환기(ViTs)가 포함되었습니다.

- **Performance Highlights**: 실험에서 GRU 기반의 1D CNN 모델이 93.4%의 최고 정확도를 기록하여 다양한 아키텍처 중 가장 높은 성능을 보였습니다. 또한, ECG 신호를 이미지로 변환하여 2D CNN 모델이 92.16%의 정확도를 달성하며 뛰어난 성과를 나타냈습니다.



### A graph neural network-based multispectral-view learning model for diabetic macular ischemia detection from color fundus photographs (https://arxiv.org/abs/2502.17886)
- **What's New**: 이번 연구에서는 당뇨병성 황반 허혈(Diabetic Macular Ischemia, DMI)을 탐지하기 위한 그래프 신경망 기반 다채로운 뷰 학습(GNN-MSVL) 모델을 제안했습니다. 일반적인 색 망막 사진(Color Fundus Photographs, CFPs)을 활용하여 DMI를 탐지할 수 있는 가능성을 탐구하였으며, 이는 안과 의사들 사이에서의 회의론을 극복하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된 모델은 계산적 다채로운 영상(Calculation Multispectral Imaging, CMI)을 통해 CFPs로부터 24파장 다채로운 망막 이미지를 재구성합니다. 이후 ResNeXt101 구조를 사용하여 재구성된 이미지에서 특징을 추출하며, 맞춤형 점프 연결 전략을 가진 그래프 신경망(GNN)을 설계하여 교차 스펙트럼 관계를 강화합니다. 이는 포괄적이고 효율적인 다채로운 뷰 학습을 가능하게 합니다.

- **Performance Highlights**: 연구는 592명의 당뇨병 환자로부터 1,078개의 CFPs를 수집하였으며, 그 중 300명의 환자에서 DMI가 진단된 530개의 CFP를 분석하였습니다. 모델은 84.7%의 정확도와 0.900의 수신자 조작 특성 곡선 아래 면적(AUROC)을 달성하였으며, 이는 CFPs로부터 훈련된 기준 모델 및 인간 전문가들보다 높은 성능을 나타냈습니다. 이러한 결과는 AI 기반 CFP 분석이 DMI 탐지에 가능성이 있음을 시사하며, 조기 및 저비용의 검진에 기여할 것입니다.



### From underwater to aerial: a novel multi-scale knowledge distillation approach for coral reef monitoring (https://arxiv.org/abs/2502.17883)
- **What's New**: 본 연구는 드론 기반 원격 감지와 AI 기반 방법론을 결합하여 산호초 생태계 모니터링을 위한 새로운 다중 규모 접근 방식을 제안합니다. 이 방법은 수중 이미지와 공중 이미지의 통합을 통해 정확하고 확장 가능한 산호초 평가를 가능하게 합니다. 특히 31개 산호 형태 및 관련 서식지를 탐지하는 변환기 모델을 활용하여, 고해상도의 뱅파 결과를 도출하고자 합니다.

- **Technical Details**: 자율 수상 차량(ASV)와 드론을 사용하여 라군에서 수집된 수중 및 공중 이미지를 결합합니다. 수중에서 촬영된 고밀도 이미지와 데이터 메타데이터를 통해 교차 정보를 이전하는 "가중 발자국" 방법을 적용합니다. 이 방법은 수중 이미지의 미세 데이터를 대규모 공중 이미지에 전이하며, 전반적으로 최종 분류 성능을 향상시킵니다.

- **Performance Highlights**: 연구 결과, 다중 규모 방법론을 통해 넓은 해역에서 높은 정확도로 산호 형태를 예측할 수 있음을 보여주었습니다. AUC (Area Under the Curve) 점수는 0.9251로, 고해상도 수중 데이터와 공중 데이터의 통합에 따른 정확한 결과를 나타냅니다. 이러한 접근 방식은 산호초 모니터링과 보존에 있어 혁신적인 가능성을 제공합니다.



### Contrastive Learning with Nasty Nois (https://arxiv.org/abs/2502.17872)
- **What's New**: 이 논문은 adversarial noise 아래에서 contrastive learning의 이론적 한계를 분석합니다. 연구는 PAC learning(Probably Approximately Correct) 및 VC-dimension(컬럼의 차원) 분석을 사용하여 adversarial 환경에서의 샘플 복잡성에 대한 하한 및 상한을 설정합니다. 또한, l2-distance 함수를 기반으로 한 데이터 의존적인 샘플 복잡성 경계를 도출합니다.

- **Technical Details**: contrastive learning에서는 
ho: V × V → ℝ와 같은 거리를 사용하는데, 이 거리는 d차원의 representation 함수 f에 의해 정의됩니다. 학습 작업은 boolean classifier의 가설 클래스에 의해 정의되며, PAC 모델을 사용하여 라벨이 지정된 예제 (x, y+, z−)에 접근합니다. 이를 통해 distance function ρ에 대한 가설 hρ를 정의하고, noise에 강한 학습 알고리즘의 존재를 입증합니다.

- **Performance Highlights**: П 이 연구는 adversarial noise의 존재 하에서 contrastive learning의 샘플 복잡성에 대한 최적 한계를 탐구합니다. 이는 adversary가 샘플을 수정 및 교체하여 노이즈 뷰 또는 노이즈 레이블을 도입하는 환경에서 효과적으로 동작하지 않게 하는 메커니즘을 설명합니다. 논문의 결과는 contrastive learning을 향상시키기 위한 전략적 방향을 제시합니다.



### ASurvey: Spatiotemporal Consistency in Video Generation (https://arxiv.org/abs/2502.17863)
- **What's New**: 이 논문은 비디오 생성의 최근 발전을 체계적으로 검토하며, 공간 및 시간 일관성을 유지하는 데 중점을 두고 있는 다양한 연구를 포괄하고 있습니다. 기존 비디오 생성 관련 문헌에서 이 문제를 다룬 연구가 부족했음을 언급하며, 이로 인해 고품질 비디오 생성을 위한 기초적인 메커니즘을 깊이 이해하는 데 어려움이 있음을 알립니다. 이 설문조사는 비디오生成에 대한 기초 모델, 정보 표현 방법, 생성 방식, 후처리 기법 및 평가 메트릭의 다섯 가지 주요 측면을 다룹니다.

- **Technical Details**: 본 연구에서는 Generative Adversarial Networks (GAN), Autoregressive, Diffusion 및 Mask 모델을 포함한 여러 기초 모델을 요약하고, 이들이 어떻게 공간 및 시간 일관성을 유지하는지 설명합니다. GAN 모델은 생성자와 판별자로 구성되어 있으며, 생성자는 실제 데이터와 유사한 데이터를 생성하도록 훈련됩니다. Autoregressive 모델은 이전 프레임을 기반으로 현재 프레임을 생성하며, Diffusion 모델은 이미지 생성에서 비디오 생성으로 전이되어 활용됩니다.

- **Performance Highlights**: 공간 일관성(spatial consistency)과 시간 일관성(temporal consistency)을 유지하는 것은 비디오 생성의 핵심 과제입니다. 이 설문조사는 이러한 일관성을 달성하기 위한 최근의 접근 방식을 종합하였으며, 후속 연구를 위한 방향성과 도전 과제를 논의합니다. 미래 연구 개발이 비디오 생성 기술의 발전에 기여할 것으로 기대합니다.



### Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation (https://arxiv.org/abs/2502.17839)
Comments:
          16 pages, 2 figures

- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 프레임워크에 실용적인 원칙을 주입하여 검색된 컨텍스트의 유용성을 향상시키는 간단하고 비지도 학습 방식의 방법을 제안합니다. 이 방법은 RAG에 의해 검색된 문서 풀에서 질문과 가장 관련이 높은 문장을 식별하고, 입력 질문에서 다루어진 모든 주제를 포함하되 그 이상은 포함하지 않도록 하며, 이러한 문장을 원래의 맥락 내에서 강조 표시한 후 LLM에 제공합니다. 실험을 통해, 이 접근 방식이 세 가지 질문 답변 작업(ARC-Challenge, PubHealth, PopQA)에서 일관된 개선 효과를 보여준다는 사실을 입증하였습니다.

- **Technical Details**: RAG는 큰 언어 모델(LLMs)의 제한된 지식 수평을 해결하기 위해 등장하였습니다. 본 연구는 RAG가 종종 LLM에 너무 많은 정보를 제시하여, 이를 통해 효과적인 소통을 위한 Grice의 네 가지 격률을 위반하는 경우가 많다고 주장합니다. 제안된 방법은 비지도 학습 기반의 휴리스틱을 사용하여 Grice의 격률을 구현하고, RAG에서 주제에 맞는 문장들을 식별함으로써 성능을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PubHealth에서 19.7% 및 ARC-Challenge에서 10%의 상대적인 정확도 향상을 달성하였습니다. 제안된 접근 방식은 또한 작은 언어 모델인 AMD-OLMo-1B-SFT에서 10%의 개선 효과를 보였으며, RAG의 성능 향상에 있어 상용화된 검색기인 DPR과 함께 사용될 때 20%까지 성능을 개선할 수 있는 가능성을 제시합니다.



### MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks (https://arxiv.org/abs/2502.17832)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서 제안하는 MM-PoisonRAG는 멀티모달 RAG 프레임워크에 대한 최초의 지식 오염 공격 프레임워크입니다. 공격자는 사실과 관련 없는 지식을 외부 지식 기반(KB)에 주입하여 모델이 잘못된 답변을 생성하도록 유도합니다. 두 가지 공격 전략인 Localized Poisoning Attack (LPA)와 Globalized Poisoning Attack (GPA)을 통해 특정 쿼리나 전반적인 조건에서 모델의 출력을 조작할 수 있습니다.

- **Technical Details**: MM-PoisonRAG는 LPA와 GPA 두 가지 공격 시나리오를 가지고 있으며, LPA는 쿼리와 관련된 잘못된 정보를 주입하여 특정 쿼리에 대한 조작을 목표로 합니다. 반면 GPA는 모든 쿼리에서 비관련 지식을 삽입하여 모델의 생성을 완전히 교란시킵니다. 이 연구에서는 각 공격이 모델의 응답 생성에 미치는 영향을 여러 작업과 설정에서 평가하였습니다.

- **Performance Highlights**: LPA는 최대 56%의 성공률로 공격자가 정의한 답변을 생성하는 데 성공했습니다. 반면 GPA는 단 한번의 비관련 지식 주입으로 모델의 정확도를 0%로 떨어뜨리는 결과를 보였습니다. 이러한 결과는 멀티모달 RAG 프레임워크를 보호하기 위해 강력한 방어책의 필요성을 강조합니다.



### CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems (https://arxiv.org/abs/2502.17821)
- **What's New**: 이번 연구에서 제안한 Collaborative Auxiliary Modality Learning (CAML)은 기존의 Auxiliary Modality Learning (AML)의 제한점을 극복하기 위해 다중 에이전트 간의 협업을 통한 다중 모달 데이터를 공유하는 새로운 프레임워크입니다. CAML은 테스트 시 각 에이전트가 감소된 모달리티로 추론할 수 있도록 하여, 자율 주행과 같이 동적 환경에서 더 나은 의사 결정을 지원합니다. 이 연구는 데이터 커버리지와 불확실성 감소 측면에서 CAML의 효과성을 분석하고, AML에 비해 이점들을 제공합니다.

- **Technical Details**: CAML은 다중 에이전트 시스템에서 훈련 기간 동안 모달 데이터를 공유하고, 테스트 시 각 에이전트가 감소된 모달리티로 추론하도록 설계되었습니다. 또한, 지식 전이(knowledge distillation)를 활용하여, 복잡한 모델에서 단순한 모델로 지식을 전이하여 모달리티가 결핍된 상태에서도 작동할 수 있도록 합니다. 예를 들어, 자율 주행 차량들이 훈련 중 LiDAR 및 RGB 이미지와 같은 센서 정보를 공유하여 더욱 견고한 표현을 구축하고, 운영 중에는 RGB 이미지만으로 추론을 수행하게 합니다.

- **Performance Highlights**: CAML은 사고 발생 가능성이 높은 자율 주행 시나리오에서 최대 58.13%의 사고 감지 성능 향상을 이뤄냈습니다. 추가적으로, 실제 공중-지상 로봇 데이터를 활용한 협업적 의미 분할에서 최대 10.61%의 mIoU 개선을 달성하였습니다. 이러한 결과는 CAML이 실세계 응용 프로그램에서 다중 에이전트 협업의 이점을 충분히 활용할 수 있음을 보여줍니다.



### An Overview of Large Language Models for Statisticians (https://arxiv.org/abs/2502.17814)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 통계와 AI 분야의 교차점에서 어떻게 신뢰성과 투명성을 증진시키는데 기여할 수 있는지를 탐구합니다. 특히 불확실성 정량화(uncertainty quantification), 해석 가능성(interpretability), 공정성(fairness), 개인 정보 보호(privay), 워터마킹(watermarking) 및 모델 적응(model adaptation)과 같은 이슈에 초점을 맞추고 있습니다. 이러한 세부 사항들을 통해 통계학자들이 LLMs의 발전에 중요한 기여를 할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: LLMs는 자연어를 이해하고 생성하는 데 있어 본질적으로 단어 또는 단어 시퀀스의 확률을 할당하는 모델입니다. LLMs의 설계와 배치에서 통계학자들이 어떤 역할을 할 수 있는지에 대한 질문이 제기되며, 이는 통계 방법 및 이론의 진전을 요구합니다. 모델의 아키텍처가 신뢰할 수 있는 확률적 출력을 생성하는 방법과 알고리즘의 공정성과 신뢰성을 보장하기 위해 LLMs의 출력을 어떻게 활용할 수 있는지를 탐구하고 있습니다.

- **Performance Highlights**: LLMs는 데이터 수집, 청소 및 분석의 전통적인 통계 워크플로우를 향상시킬 수 있는 잠재력을 제공합니다. 의료 연구 및 생물 통계와 같은 분야에서 LLMs는 대표 데이터를 합성하고, 비정형 임상 노트에서 중요한 통찰을 추출하며, 높은 위험의 응용에서 예측 모델링을 지원할 수 있습니다. 이러한 상호 작용은 통계와 AI 분야 모두에서 새로운 기회를 제공하며, LLMs의 발전이 사회적 복잡성 문제 해결에 기여할 수 있는 가능성을 열어갑니다.



### Research on Enhancing Cloud Computing Network Security using Artificial Intelligence Algorithms (https://arxiv.org/abs/2502.17801)
- **What's New**: 논문에서는 클라우드 컴퓨팅 환경에서의 보안 취약점을 해결하기 위해 딥러닝(Deep Learning)을 활용한 적응형 보안 프로텍션 프레임워크를 제안합니다. 전통적인 보안 메커니즘이 변화하는 공격 전략에 적응하는 데 어려움을 겪고 있어, 새로운 다층 방어 아키텍처를 통합하여 보안성을 높였습니다.

- **Technical Details**: 제안된 시스템은 실제 비즈니스 환경에서 평가되었으며, 97.3%의 탐지 정확도(detection accuracy)와 18 ms의 평균 응답 시간(average response time), 99.999%의 가용성(availability rate)을 기록했습니다. 이는 기존의 규칙 매칭(rule matching)과 특징 인식(feature recognition) 기반 메커니즘과의 비교에서 뛰어난 성능을 나타냅니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 탐지 정확도와 응답 효율성(response efficiency), 자원 활용(resource utilization)을 크게 개선시키는 것으로 나타났습니다. 이 연구는 클라우드 컴퓨팅 보안에 대한 새로운 접근 방식을 제시하며, 실제 환경에서의 유용성을 입증합니다.



### Synthia: Novel Concept Design with Affordance Composition (https://arxiv.org/abs/2502.17793)
Comments:
          Code is available this https URL

- **What's New**: 본 논문은 기능적으로 일관된 디자인을 생성하기 위한 새로운 프레임워크인 SYNTHIA를 소개합니다. SYNTHIA는 사용자가 원하는 affordance(기능적 가능성)를 기반으로 하여 창의적인 개념 합성을 가능하게 하며, 이는 기존의 T2I 모델이 간과했던 기능적 일관성을 보장하는 데 중점을 둡니다. 또한, 계층적 개념 온톨로지를 활용하여 개념을 부분과 affordance로 분해하고, 이 구조적 접근을 통해 더욱 효과적인 디자인 제안이 가능합니다.

- **Technical Details**: SYNTHIA 프레임워크는 계층적 개념 온톨로지를 통한 affordance의 체계적인 구성이 주요 기술적 요소입니다. 이를 통해 모델은 일반적인 개념-기능 연관 외에도 복잡한 기능 조합을 학습하여, 긴밀하게 연결된 여러 기능을 하나의 일관된 형태로 통합하는 능력을 갖추게 됩니다. 더불어 교육 과정에 기반한 최적화를 통해 T2I 모델이 점진적으로 affordance 조합을 학습할 수 있도록 하며, 시각적 혁신을 유지하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험 결과, SYNTHIA는 기존의 T2I 모델을 크게 초월해 관찰 가능한 시각적 혁신성과 기능적 일관성 측면에서 각각 25.1% 및 14.7% 향상을 보였습니다. 이러한 결과는 인간 평가에서도 뚜렷하게 나타났으며, 새로운 디자인이 시각적으로 뿐만 아니라 기능적으로도 높은 실용성을 갖추었음을 증명합니다. SYNTHIA는 기존의 텍스트 기반 설명 의존에서 벗어나, 사용자가 지정한 affordance에 직접적으로 기반하여 창의적인 디자인을 생성할 수 있습니다.



### AIR: Complex Instruction Generation via Automatic Iterative Refinemen (https://arxiv.org/abs/2502.17787)
Comments:
          The first three authors contributed equally, 20 pages

- **What's New**: 이 논문에서는 복잡한 지침을 생성하는 새로운 자동 반복 정제 프레임워크인 AIR(Automatic Iterative Refinement)를 제안합니다. 기존의 방법들은 복잡한 지침을 성공적으로 생성하는 데 어려움을 겪고 있으며, 주로 단순 지침을 바탕으로 하거나 인적 자원이 많이 소모되는 방식입니다. AIR 프레임워크는 실제 요구 사항을 더 잘 반영하며, LLM(대형 언어 모델)의 복잡한 지침을 따르는 능력을 향상시킵니다.

- **Technical Details**: AIR 프레임워크는 두 단계로 구성됩니다: 첫 번째 단계는 문서에서 초기 지침을 생성하는 것이고, 두 번째 단계는 LLM-as-judge의 도움을 받아 지침을 반복적으로 정제하는 것입니다. 여기서 각 지침의 품질은 점수화되어 필터링되며, 복잡한 지침 생성을 위한 필수 원칙을 고려합니다. 백 번역(back-translation) 기법을 사용하여 기존 문서에서 새로운 지침을 생성합니다.

- **Performance Highlights**: AIR-10K 데이터셋은 10,000개의 복잡한 지침으로 구성되어 있으며, 제안된 방법으로 생성된 지침은 기존 방법들보다 모델의 복잡한 지침을 따르는 능력을 현저히 향상시킵니다. 실험 결과, AIR 프레임워크가 제공하는 정제 과정이 성능 개선에 크게 기여함을 보여줍니다. 이를 통해 복잡한 지침 생성을 위한 새로운 접근 방식을 제시합니다.



### Uncertainty Quantification for LLM-Based Survey Simulations (https://arxiv.org/abs/2502.17773)
Comments:
          30 pages, 6 figures, 10 tables

- **What's New**: 본 논문은 대형 언어 모델(LLM)이 생성한 시뮬레이션 응답을 신뢰성 있게 활용하기 위한 방법을 제시합니다. 특히, 이러한 응답을 바탕으로 인구 통계의 신뢰 구간(confidence sets)을 구성하는 접근법을 개발하였습니다. 또한 샘플 크기를 적응적으로 선택함으로써, 과도한 오버피팅(overfitting)이나 비효율적인 추정치를 방지하려고 합니다.

- **Technical Details**: LLM 기반 설문 조사 시뮬레이션에서의 불확실성 정량화(unquantifying uncertainty)에 대한 엄격한 수학적 프레임워크를 제공합니다. 우리는 LLM과 실제 인구 간의 미스알라이먼트(misalignment)를 기반으로 시뮬레이션 샘플 크기를 적응적으로 선택하는 유연한 방법론(methodology)을 제안하고 있습니다. 이는 모든 LLM에 적용가능하며, 신뢰 구간을 구축하는데 드는 다양한 방법과 결합할 수 있습니다.

- **Performance Highlights**: 이 제안된 방법은 실제 데이터셋에 대한 일련의 실험을 통해 입증되었습니다. 결과적으로, 적절한 수의 시뮬레이션 샘플을 통해 신뢰할 수 있는 인구 통계 추정을 달성할 수 있었습니다. 우리의 방법은 LLM의 활용 가능성을 향상시키며, 인구 통계에 대한 이해도를 높이는 데 기여할 것입니다.



### Sample Selection via Contrastive Fragmentation for Noisy Label Regression (https://arxiv.org/abs/2502.17771)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 ConFrag라는 새로운 방법론을 제안합니다. 이 접근법은 레이블과 특성 간의 연속적인 상관관계를 활용하여 회귀 문제의 노이즈 레이블 문제를 해결하는 데 초점을 맞춥니다. 특히, 레이블 공간에서 가장 먼 조각들을 짝지어 대조적인 조각 쌍(contrastive fragment pairs)을 형성함으로써, 데이터 포인트 간 유사성에 기반한 훈련이 가능합니다.

- **Technical Details**: ConFrag(Contrastive Fragmentation) 프레임워크는 데이터셋을 대조적인 조각 쌍으로 나누고, 이 조각 쌍을 통해 클린 샘플을 선택하는 방식으로 작동합니다. 선택된 클린 샘플은 점진적인 학습을 통해 회귀 모델의 성능을 향상시킵니다. 이 프레임워크는 노이즈 비율에 무관하게 작동하며, 인접 조각의 혼합(mixture) 모델을 통해 주변 동의(neighborhood agreement)를 사용하여 노이즈 레이블을 식별합니다.

- **Performance Highlights**: ConFrag는 14개의 최첨단 기법들과 비교하여 일관되게 더 우수한 성능을 보여주었습니다. 다양한 도메인에서 수집한 6개의 벤치마크 데이터셋을 통해 실험을 수행하였으며, 이 과정에서 레이블 간의 노이즈 정도를 고려하는 Error Residual Ratio(ERR)라는 새로운 성능 지표를 도입했습니다. 이러한 방법론은 레이블 노이즈의 대칭 및 무작위 가우시안 노이즈에 강한 저항력을 가지고 있습니다.



### DeepSeek vs. ChatGPT: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks (https://arxiv.org/abs/2502.17764)
- **What's New**: 이번 연구에서는 최신 LLM(대형 언어 모델)인 ChatGPT o3-mini-high와 DeepSeek R1이 과학적 문제 해결에 어떻게 접근하는지를 비교합니다. 실험을 통해 이 모델들이 수치적 문제와 PDE(부분 미분 방정식) 기반 문제 해결에서의 성능 차이를 분석합니다. ChatGPT o3-mini-high는 일반적으로 더 높은 정확도와 응답 속도를 보여, 다양한 계산 작업에 있어 더 실용적인 선택으로 평가됩니다.

- **Technical Details**: 연구에서는 LLM의 성능을 평가하기 위해 DeepSeek V3, DeepSeek R1, ChatGPT 4o, ChatGPT o3-mini-high 네 가지 모델을 사용했습니다. 모델들은 전통적인 수치 방법, 예를 들어 유한 차분법(Finite Difference Method)과 유한 요소법(Finite Element Method)을 활용해 PDE를 해결하는 능력을 평가받았습니다. 실험은 LLM들이 적절한 입력 함수 공간을 정의해야 하는 비트리비얼(decision is required) 문제를 포함해 설계되었습니다.

- **Performance Highlights**: ChatGPT o3-mini-high는 수치적 알고리즘과 과학적 기계 학습에서 전반적으로 뛰어난 성능을 보였습니다. 특히, DeepSeek R1보다 속도와 정확성 면에서 우수한 결과를 도출했습니다. 실험을 통해 모델들은 수학적 논리의 깊이, 신뢰성 및 연구 수준 과학 문제로 일반화하는 능력도 평가되었습니다.



### Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM (https://arxiv.org/abs/2502.17763)
- **What's New**: 이 논문은 전통적인 보안 방법이 대규모 분산 시스템에서 복잡한 공격 벡터를 처리하는 데 한계를 가지고 있다는 문제를 다루고 있습니다. 이 새로운 보안 위협 탐지 시스템은 Federated Learning을 멀티모달 대형 언어 모델(multimodal large language models, LLMs)과 통합하여 데이터 프라이버시(data privacy)를 보장하면서도 다양한 데이터 소스를 처리할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: 제안된 시스템은 네트워크 트래픽, 시스템 로그, 이미지, 센서 데이터 등 이질적인 데이터 소스를 처리하기 위해 멀티모달 LLM을 사용하여 데이터를 분석합니다. 이 시스템은 10TB의 분산 데이터 세트에서 실험적으로 평가되었으며, 모델 훈련에는 180초, 위협 탐지에는 3.8초가 소요되어 분산 환경에서 효율적인 처리 능력을 유지합니다.

- **Performance Highlights**: 실험 결과, 이 시스템은 96.4%의 탐지 정확도를 기록하여 전통적인 기준 모델보다 4.1% 높은 성능을 보였습니다. 또한, 잘못된 양성(false positive) 및 잘못된 음성(false negative) 비율을 각각 1.8% 및 2.4% 감소시켰습니다. 이 결과들은 데이터 프라이버시를 유지하면서도 탐지 정확도와 계산 효율성이 크게 향상되었음을 보여주며, 대규모 보안 시스템의 실제 배치 가능성을 시사합니다.



### Graded Neural Networks (https://arxiv.org/abs/2502.17751)
- **What's New**: 이번 논문은 graded vector spaces $\,V_{\boldsymbol{w}}^{n}$를 기반으로 한 새로운 graded neural network (GNN) 프레임워크를 제시합니다. 전통적인 신경망 구조에 대수적 그레이딩을 접목시켜 기능의 중요성을 반영할 수 있는 신경세포, 층, 활성화 함수 및 손실 함수를 개발했습니다. 이는 머신 러닝 및 광자 시스템과 같은 다양한 응용 분야에 적용될 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 이 논문은 좌표별 그레이딩 구조를 활용하여 칼라르 액션 $\,\lambda\ast \mathbf{x}=(\lambda^{q_{i}}x_{i})$을 정의하고, graded neural network(GNN)를 설계했습니다. GNN은 각 입력의 중요성을 고려하여 대응하는 그레이드 $\,gr(x_{i}) \in I$를 부여하여 신경망 계층의 연산을 수행합니다. 이러한 구조는 전통적인 신경망 모델이 아닌, 그레이드가 있는 입력을 자연스럽게 처리할 수 있는 특징을 가지고 있습니다.

- **Performance Highlights**: 이 연구는 40%의 정확도로 시작한 이미지 자동군 예측하였으나, 가중된 불변량을 사용하는 경우 정확도가 99%로 상승하는 것을 발견했습니다. 이러한 성장은 체계적이며 가정적이지 않으며, GNN이 본질적으로 성능을 향상시킬 수 있는 가능성을 제시합니다. 평가된 다양한 시스템에서의 성능을 통해, GNN은 광자 기반 시스템에서 특히 고속 레이저 구현에 대한 가능성을 열어줍니다.



### LLM Inference Acceleration via Efficient Operation Fusion (https://arxiv.org/abs/2502.17728)
- **What's New**: 최근 Transformer 기반 대형 언어 모델(LLMs)의 빠른 발전은 계속 증가하는 매개변수 수와 밀접하게 연결되어 있습니다. 본 논문에서는 Softmax와 Layernorm의 계산 시 발생하는 오버헤드를 완전히 차단할 수 있는 매우 효율적인 기술을 제안합니다. 이 기술을 통해 선형 계층과 비선형 계층의 작업을 병렬로 처리하여 전체 지연 시간을 줄이는 데 중점을 둡니다.

- **Technical Details**: Transformer 아키텍처의 주요 도전 과제 중 하나는 비선형 변환이며, 이에는 마스크 연산이 포함됩니다. 특정 연산, 예를 들어 Softmax 및 Layernorm은 집합적 집계 연산을 요구하며, 이는 서로 다른 하드웨어 엔진에서 계산을 수행할 때의 지연 요인이 됩니다. 본 연구에서는 이러한 집합적 연산을 피하기 위해 비선형 연산을 선형 방식으로 결합하여 계산 성능을 극대화했습니다.

- **Performance Highlights**: 제안된 기법은 전체 추론 지연 시간을 하드웨어 아키텍처에 따라 약 15-20% 줄이는 것으로 나타났습니다. 이는 우수한 모델 성능을 유지하면서도 비선형 및 선형 연산의 최적화를 통해 달성되었습니다. 따라서 이 접근법은 대규모 LLM의 실제 배치와 확장에 중요한 기여를 할 것입니다.



### The GigaMIDI Dataset with Features for Expressive Music Performance Detection (https://arxiv.org/abs/2502.17726)
Comments:
          Published at Transactions of the International Society for Music Information Retrieval (TISMIR), 8(1), 1-19

- **What's New**: 이 논문은 GigaMIDI 데이터셋을 소개하며, 140만 개의 유니크 MIDI 파일과 5천3백만 개의 트랙을 포함하고 있습니다. MIDI 파일에서 비표현적(non-expressive) 성능과 표현적(expressive) 성능을 구별하는 것은 도전적인 작업이며, 이를 위해 새로운 휴리스틱(heuristics)을 도입했습니다. 특히, Distinctive Note Velocity Ratio (DNVR)와 Note Onset Median Metric Level (NOMML) 등의 방법을 사용하여 MIDI 트랙을 구분할 수 있습니다.

- **Technical Details**: 디지털 음악 표현 방식은 오디오와 기호(symbolic) 두 가지로 나뉘며, MIDI는 기호 데이터를 효율적으로 저장하는 형식으로 인식되고 있습니다. MIDI 파일은 악기 연주와 같은 다양한 실시간 연출을 포함한 멀티트랙 아키텍처를 사용하여 음악 정보를 전달합니다. 이 논문에서는 새로운 휴리스틱을 통해 MIDI 트랙의 표현성을 분석하고 평가하는 방법론을 제안합니다.

- **Performance Highlights**: 이 연구를 통해 GigaMIDI 데이터셋에서 31%를 차지하는 표현적 MIDI 트랙을 포함하는 가장 큰 MIDI 데이터셋이 구축되었습니다. 논문에서 제안한 휴리스틱들은 MIDI 트랙의 표현적 차이를 효과적으로 구분할 수 있다는 충분한 평가 결과를 보여주고 있습니다. 최종적으로, 연구 결과는 MIDI 기반 음악 생성과 분석의 발전에 기여할 것으로 기대됩니다.



### Solving the Traveling Salesman Problem via Different Quantum Computing Architectures (https://arxiv.org/abs/2502.17725)
Comments:
          13 pages, 21 figures, 32 citations

- **What's New**: 본 논문은 Traveling Salesman Problem (TSP)이라는 NP-hard 최적화 문제를 해결하기 위한 최신 광자 및 양자 컴퓨팅 아키텍처의 적용을 연구합니다. 다양한 접근 방식을 조사하여, Simulated Annealing (SA), Quadratic Unconstrained Binary Optimization (QUBO-Ising) 방법 및 Quantum Approximate Optimization Algorithm (QAOA) 등을 다룹니다. 이 연구는 IBM Quantum 플랫폼에서 QAOA와 Quantum Phase Estimation (QPE)을 시험하였으며, D-Wave 양자 어닐러를 사용하여 QUBO-Ising 방법도 탐색하였습니다.

- **Technical Details**: QAOA는 조합 최적화 문제를 풀기 위해 설계된 하이브리드 양자-고전적 변분 알고리즘입니다. 알고리즘의 성능을 이해하고 구현하기 위해서는 외부의 고전적 최적화를 수행하는 구성적 접근이 필수적입니다. QAOA는 특히 작은 스펙트럼 간격이 존재할 때 고전 제약 조건을 비선형 최적화로 해결할 수 있는 능력을 보여주었습니다. 이 연구는 양자 컴퓨터에서 TSP를 해결하기 위해 QUBO 문제로 정의하고 QAOA로 변환하는 기존 방법들을 탐구합니다.

- **Performance Highlights**: Ising 머신은 TSP 인스턴스 처리에서 고전적인 방법에 비해 유의미한 시간 이점을 제공합니다. SQUID 기반 Ising 머신은 최대 12 노드의 TSP 인스턴스를 처리할 수 있으며, 비선형 광전자 Ising 머신은 최대 18 노드까지 확장할 수 있습니다. 그러나 하드웨어의 한계로 인해 최적의 해를 제공하기 어려운 상황이 발생할 수 있으며, 문제의 크기가 증가할수록 접지 상태 수렴 문제도 나타납니다. 이러한 제한에도 불구하고, Ising 머신은 대규모 TSP를 효율적으로 해결할 수 있는 유망한 후보로 평가됩니다.



### Aligning Compound AI Systems via System-level DPO (https://arxiv.org/abs/2502.17721)
Comments:
          Accepted to workshops MARW and WMAC (Oral) at AAAI25

- **What's New**: 본 논문에서는 여러 상호작용하는 구성 요소로 이루어진 조합 AI 시스템의 일관성을 확보하기 위한 새로운 접근 방식을 제안합니다. 기존의 직접 선호 최적화(Direct Preference Optimization, DPO) 방법이 복합 AI 시스템에 직접 적용될 수 없다는 문제를 강조하고, 이러한 시스템을 Directed Acyclic Graph (DAG)로 구성하여 문제를 해결하려고 시도합니다.

- **Technical Details**: 조합 AI 시스템을 DAG로 모델링함으로써 구성 요소 간의 관계 및 데이터 생성 프로세스를 명확하게 포착할 수 있습니다. 이를 기반으로 시스템 수준 DPO(System-Level DPO, SysDPO)를 제안하여 조합 시스템 내의 최적화를 수행하고, 대형 언어 모델(LLM)과 확산 모델(diffusion model)의 공동 정렬을 연구합니다. 또한, DDPM(diffusion model)의 작동 원리와 최적화 방법에 대한 자세한 설명을 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 LLM과 확산 모델의 공동 정렬이 효과적임을 입증하였습니다. 이러한 발견은 조합 AI 시스템의 정렬 문제에 대한 깊은 통찰력을 제공하며, 향후 기술 발전의 토대를 마련합니다. 본 연구는 기존의 AI 시스템 정렬 방식의 한계를 보완하는 중요한 기여를 하고 있습니다.



### Spontaneous Giving and Calculated Greed in Language Models (https://arxiv.org/abs/2502.17720)
- **What's New**: 이번 연구에서는 언어 모델의 추론 능력이 사회적 지능에 어떻게 영향을 미치는지 분석하였습니다. 특히, 언어 모델이 공공재 게임과 같은 사회적 딜레마에서 협력 결정에 미치는 영향을 살펴보았으며, 다양한 경제 게임에서 비추론 모델과 추론 모델의 성능을 비교했습니다. 연구 결과, 추론 모델이 사회적 협력을 줄이고 개인의 합리적인 선택을 우선시하는 경향이 있다는 사실을 확인했습니다.

- **Technical Details**: 연구에서는 두 가지 추론 기법인 chain-of-thought와 reflection이 공공재 게임에서 GPT-4o 모델의 협력 결정에 미치는 영향을 평가했습니다. 실험 결과, chain-of-thought 기법을 사용했을 때 협력 비율이 60% 크게 감소하였으며, 반영(reflection) 능력을 사용할 경우에도 협력이 현저히 줄어드는 경향이 나타났습니다. 여러 경제 게임에서 비추론 모델과 추론 모델을 비교했을 때, 추론 모델이 협력 및 처벌 결정에서 비추론 모델보다 낮은 협력 비율과 처벌 비율을 보였습니다.

- **Performance Highlights**: 결과적으로, 전부 GPT-4o 모델로 구성된 그룹은 협력 수준이 매우 높게 유지되는 반면, 그룹 내 추론 o1 모델의 비율이 증가함에 따라 협력 수준이 꾸준히 감소하는 경향을 보였습니다. 특히, 전부 추론 o1 모델로 구성된 그룹은 평균적으로 약 20%의 협력만을 나타냈습니다. 이러한 결과는 AI 모델에서 사회적 지능을 통합할 필요성을 시사하며, 인류의 협력 직관을 방해하지 않도록 설계해야 한다는 점을 강조합니다.



### Bridging Information Gaps with Comprehensive Answers: Improving the Diversity and Informativeness of Follow-Up Questions (https://arxiv.org/abs/2502.17715)
Comments:
          8 pages, 2 figures, submitted to ACL 2025

- **What's New**: 이 연구는 기존의 대화형 시스템이 사용자가 노출되지 않은 정보를 끌어내기 위한 다양한 맥락에서의 후속 질문을 생성할 수 있도록 하는 새로운 방법을 제안합니다. 저자들은 가상의 LLM(large language model) 생성 '포괄적 답변'을 기반으로 하여 응답되지 않은 정보를 겨냥한 질문을 생성하는 방안을 모색하고, 실험 결과 이러한 방법이 질문의 질과 다양성을 크게 향상시킨다는 것을 입증했습니다.

- **Technical Details**: 이 연구에서는 GPT-4를 사용하여 포괄적 답변과 정보 격차 기반 후속 질문을 생성합니다. 이후 원래 FollowupQG 훈련 세트를 25,000개의 합성 예시로 증강하고, 이 데이터를 바탕으로 여러 언어 모델을 미세 조정했습니다. 초기 질문-답변 쌍에서 도출된 격차에 대한 분석을 통해 인간의 인지 전략을 활용하며, 기존의 전통적 질문 생성 방법과는 달리 Missing Information을 재조명하여 질문을 생성하는 접근법을 취하고 있습니다.

- **Performance Highlights**: 실험 결과, 증강된 데이터셋에서 훈련된 모델이 기본선에 비해 질문의 품질(유효성, 관련성, 정보성 등)과 다양성에서 눈에 띄는 개선을 보였습니다. 이 방식은 정보 탐색 대화의 모호성을 줄이고 LLM의 답변 정확도를 향상시키는 데 기여할 수 있는 가능성을 보여줍니다.



### On the usability of generative AI: Human generative AI (https://arxiv.org/abs/2502.17714)
- **What's New**: 이번 논문에서는 생성적 AI 시스템(Generative AI Systems)의 사용자 친화성(usability) 문제에 대한 심도 있는 분석을 제공합니다. 사용자 경험(user experience), 투명성(transparency), 통제(control), 인지 부하(cognitive load)와 같은 다양한 사용자 친화성 요인들이 논의됩니다. 생성적 AI의 활용도가 높아짐에 따라, 이에 대한 유용성과 접근성(accessibility)을 높이기 위한 새로운 접근 방식이 필요하다는 점을 강조합니다.

- **Technical Details**: 논문에서는 사용자 친화성의 평가 기준으로 효율성(efficiency), 학습 가능성(learnability), 만족도(satisfaction) 등을 검토하고, 다양한 분야에서의 모범 사례(best practices)를 분석합니다. 또한, 사용자 인터페이스(user interface)를 직관적으로 개선하고, 결과 해석 가능성(interpretability)을 높이는 방법이 논의됩니다. 생성적 AI 시스템에서의 출력 조정(fine-tuning)과 예측의 불확실성(unpredictability) 같은 공통 문제에 대해서도 다루고 있습니다.

- **Performance Highlights**: 사용자 피드백(user feedback)과 직관적인 인터페이스의 개선이 생성적 AI 시스템의 접근성을 더욱 높일 수 있음을 보여줍니다. 이 연구는 전체적인 사용자 경험을 개선하고, 생성적 AI의 활용성을 극대화하는 데 기여할 수 있는 방향성을 제시합니다. 실용적이고 효과적인 사용을 위한 다양한 전략들이 논의되며, 앞으로의 연구 방향에도 시사점을 제공합니다.



### Contrastive Visual Data Augmentation (https://arxiv.org/abs/2502.17709)
- **What's New**: 이 논문에서는 새로운 Contrastive Visual Data Augmentation (CoDA) 전략을 제안하여 대형 다중모달 모델(Large Multimodal Models, LMMs)이 새로운 개념을 인식하고 논리적으로 처리하는 능력을 향상시키는 방법을 설명합니다. CoDA는 대상 개념의 주요 대조적 텍스트 및 시각적 특징을 추출하고, 이를 통해 작성된 합성 데이터를 생성하여 LMM이 혼동할 수 있는 개념을 명확히 구분할 수 있도록 돕습니다.

- **Technical Details**: CoDA의 주된 과정은 대조적 특징 추출, 특징 필터링, 특징 제어 이미지 생성, 증강 이미지 필터링의 4단계로 구성됩니다. 이 과정에서 CoDA는 LMM이 잘못 인식하는 개념과 혼동하게 되는 개념을 식별하고, 해당 개념의 시각적 및 텍스트적 특징을 추출합니다. 이 특징들은 가시성이 뛰어나고 LMM이 이해할 수 있는 방식으로 생성되고 필터링을 거쳐 최종적으로 증강 이미지로 변환됩니다.

- **Performance Highlights**: CoDA는 iNaturalist와 SUN와 같은 다양한 데이터셋에서 성능이 크게 향상되는 것을 확인할 수 있었으며, NovelSpecies라는 새로운 데이터셋에서도 테스트 결과 기존의 시각적 데이터 증강 기법보다 절대적으로 12.3%, 5.1%, 6.0%의 정확도 향상을 보여주었습니다. 이로써 CoDA는 LMM의 새로운 개념 인식 능력을 효과적으로 개선하는 데 성공하였으며, 이에 따라 비전 커뮤니티에 중요한 기여를 하고 있습니다.



### To Patch or Not to Patch: Motivations, Challenges, and Implications for Cybersecurity (https://arxiv.org/abs/2502.17703)
Comments:
          7th International Conference HCI for Cybersecurity, Privacy and Trust (27th HCI International Conference)

- **What's New**: 이번 논문은 패치(patching)의 중요성과 복잡성을 다시 조사하며, 조직과 IT/보안팀이 패치를 선택하거나 실천하지 않는 이유를 비판적으로 탐구합니다. 특히, 인간 측면(human aspects)의 동기와 저항을 중점적으로 고려하여, 행동 결정에 영향을 미치는 여러 요인들을 종합하고 분석했습니다.

- **Technical Details**: 패치 관리 주기(patch management lifecycle)에 대한 논의는 중요한 핵심 사항입니다. 이는 취약성을 가진 기술의 식별, 패치의 배포 및 설치, 지속적인 패치 준수 모니터링을 포함합니다. IT 및 보안 팀은 취약성 스캐닝(vulnerability scanning), 패치 테스트(patch testing), 배포 후 검증(post-deployment verification) 등의 작업을 수행해야 하며, 이러한 과정은 시간에 민감합니다.

- **Performance Highlights**: 본 연구는 조직의 패치 결정에 대한 인센티브와 저항 요소를 구체적으로 분석하고, 특히 자원 제약이나 인적 오류와 같은 이유들이 패치 미적용의 주요 원인으로 작용함을 강조했습니다. 또한, 패치가 신속하게 적용되지 않을 경우 공격자에게 취약성이 노출되는 위험이 증가하는 점을 지적하고 있습니다.



### Yes, Q-learning Helps Offline In-Context RL (https://arxiv.org/abs/2502.17666)
- **What's New**: 이번 연구에서는 Reinforcement Learning (RL) 접근법을 확장 가능한 오프라인 In-Context RL (ICRL) 프레임워크와 통합하는 방법을 탐구합니다. 150개 이상의 데이터셋을 테스트하여 RL 목표 최적화가 성능을 평균적으로 40% 향상시킨다는 것을 입증하였습니다.

- **Technical Details**: 실험은 GridWorld와 MuJoCo 환경에서 파생된 다양한 데이터셋을 사용하여 진행되었습니다. RL 목표와 보상 극대화 goals를 정렬하는 것이 중요함을 보여주며, 오프라인 RL 기반 방법들이 온라인 접근법보다 성능이 더 뛰어난 것을 확인하였습니다.

- **Performance Highlights**: 이 연구의 결과는 다양한 데이터셋 커버리지 및 환경 복잡성에서도 오프라인 RL이 ICRL 설정에서 애플리케이션으로서의 가능성이 있음을 강조합니다. 또한, 알고리즘 증류 (Algorithm Distillation, AD)와 비교했을 때 상당한 성능 향상을 가져온다는 것을 보여줍니다.



### Effective Field Neural Network (https://arxiv.org/abs/2502.17665)
- **What's New**: 최근 머신 러닝(machine learning)의 급속한 발전에 따라, 물리학자들은 다체 문제(many-body problems)에서 차원의 저주(curse of dimensionality)를 해결하는 새로운 응용을 탐구하고 있습니다. 본 연구에서는 필드 이론(field theory)에 영감을 받아, 중요한 다체 상호작용(many-body interactions)을 자동으로 포착할 수 있는 새로운 머신 러닝 모델인 effective field neural networks (EFNNs)를 제안합니다.

- **Technical Details**: EFNN은 여러 자기 개선 프로세스(multiple self-refining processes)를 통해 다체 상호작용을 효율적으로 포착합니다. 이 논문은 고전적인 $3$-스핀 무한 범위 모델과 양자 더블 교환 모델(quantum double exchange model)을 사례로 들어, EFNN이 완전 연결 심층 신경망(fully-connected deep neural networks, DNNs)과 효과적인 모델보다 우수한 성능을 보임을 명시합니다.

- **Performance Highlights**: EFNN은 작은 시스템에서 학습된 후, 추가적인 훈련 없이도 더 큰 시스템에서 원활하게 사용될 수 있습니다. 이러한 과정에서 상대 오차(relative errors)가 감소할 수 있음을 보여주어, EFNN이 핵심 물리적 행동을 표현하는 데 효과적임을 추가적으로 입증합니다.



### StatLLM: A Dataset for Evaluating the Performance of Large Language Models in Statistical Analysis (https://arxiv.org/abs/2502.17657)
Comments:
          25 pages, 7 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 코딩 능력이 머신러닝과 데이터 과학에서 자동 통계 분석의 새로운 기회를 열어줄 수 있음을 강조합니다. 그러나 LLMs가 생성한 코드의 정확성을 평가하기 위한 벤치마크 데이터셋의 부족으로 인해 이들의 널리 사용되는 것은 어렵습니다. 이를 해결하기 위해, StatLLM이라는 오픈소스 데이터셋을 소개하여 통계 분석에서 LLMs의 성능을 평가할 수 있는 체계를 제공합니다.

- **Technical Details**: StatLLM 데이터셋은 세 가지 주요 구성 요소로 구성됩니다: (i) 통계 분석 과제, (ii) LLM에 의해 생성된 SAS 코드, (iii) 인간 평가 점수입니다. 첫 번째 요소는 다양한 통계 주제를 포함하는 문제 설명을 제공하며, 각 과제에는 데이터셋 세부사항과 인간이 검증한 SAS 코드가 첨부됩니다. 두 번째 요소는 ChatGPT 3.5, ChatGPT 4.0, Llama 3.1이 생성한 SAS 코드로, 각 모델에 대한 문제 설명과 데이터셋 정보가 제공되어 있습니다.

- **Performance Highlights**: StatLLM 데이터셋은 통계 프로그래밍의 요소에서 LLM의 성능을 평가하는 데 유용한 자료를 제공합니다. 전문가의 평가 점수는 LLM이 생성한 코드의 품질, 실행 가능성, 출력 정확성을 평가하여 LLM의 강점과 약점을 파악하는 데 도움을 줍니다. 이는 데이터 과학과 머신러닝 연구에서 신뢰할 수 있는 통계 소프트웨어 개발을 위한 기초 마련에 기여할 수 있습니다.



### METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling (https://arxiv.org/abs/2502.17651)
- **What's New**: 이 논문에서는 효과적인 자동 챠트 생성을 위한 비전-언어 모델(Vision-Language Model, VLM)을 기반으로 하는 다중 에이전트 프레임워크인 METAL를 제안합니다. METAL는 각기 다른 전문 역할을 가지고 있는 에이전트들 간의 협업을 통해 복잡한 다중 모달 추론 과제를 분해합니다. 이를 통해 챠트 생성 정확도를 5.2% 향상시켜 기존에 비해 상당한 개선 효과를 보여줍니다.

- **Technical Details**: METAL 프레임워크는 주어진 참조 챠트 이미지를 기반으로 프로그래밍 사양을 학습하는 것을 목표로 합니다. 네 가지 전문 에이전트(Generation Agent, Visual Critique Agent, Code Critique Agent, Revision Agent)가 협력하여 직관적으로 진행되며 각 에이전트는 반복적으로 결과물을 개선합니다. 특히, 이 프레임워크는 시각적 평가와 코드 분석 시에 서로 다른 모달리티를 분리하는 방식을 채택하여 VLM의 자기 수정 능력을 향상시킵니다.

- **Performance Highlights**: METAL을 통해 얻은 실험 결과는 챠트 생성 정확도를 11.33% 이상 향상시키며, 이는 VLM이 시각적 이해와 코드 합성을 통합하는 능력을 크게 향상시킨다는 것을 보여줍니다. 또한, 테스트 시간 스케일링(timing scaling) 특성을 발견하여 컴퓨팅 예산이 증가함에 따라 METAL의 성능이 일관되게 개선된다는 것을 증명하였습니다. 이러한 성과는 VLM 기반의 비주얼 중심 코드 생성 향상에 있어 유망한 경로를 제공합니다.



### Wearable Meets LLM for Stress Management: A Duoethnographic Study Integrating Wearable-Triggered Stressors and LLM Chatbots for Personalized Interventions (https://arxiv.org/abs/2502.17650)
Comments:
          In CHI '25 Proceedings of the CHI Conference on Human Factors in Computing Systems Yokohama, Japan

- **What's New**: 본 연구에서는 착용 가능한 장치 통합 LLM 채팅봇이 개인화된 스트레스 관리에서 어떤 방식으로 도움이 되는지를 연구하기 위해 듀오에스노그래픽(dyoethnographic) 접근법을 사용했습니다. 연구자들은 22일 동안 커스터마이즈된 채팅봇과 상호작용하며 스트레스 요인을 기록하고, 그에 대한 맞춤형 개입을 모색하는 과정을 밟았습니다. 결과적으로, 감지된 생리학적 신호에 대한 반응이 대부분 의미가 있었지만, 개입이 필요할 만큼 중요한 사건은 5건 중 1건에 불과하다는 것을 발견했습니다.

- **Technical Details**: 연구의 방법론으로는 듀오에스노그래피가 사용되었으며, 이는 두 명의 연구자가 공동으로 경험을 기록하고 해석하는 질적 연구 접근법입니다. 연구자들은 Samsung Galaxy Watch 6와 같은 착용 장치를 통해 스트레스와 관련된 생리학적 사건을 감지하고 CuesHub 앱을 통해 즉시 데이터를 전송했습니다. 분석 과정에서는 각 연구자의 스트레스 유발 요인에 대한 맞춤형 프롬프트 템플릿을 만들어 사용함으로써, 개별 사용자의 필요와 선호도에 맞춘 개입 방안을 탐구하였습니다.

- **Performance Highlights**: 착용 가능한 장치와 LLM 채팅봇의 통합을 통해 얻은 데이터는 스트레스 관리의 유효성을 향상시켰으며, 특히 짧은 사건 설명을 통합한 개입이 더 효과적이라는 결과를 보여주었습니다. 연구자들은 생리적 사건 감지시 실시간 개입을 제공하는 시스템의 필요성을 강조하며, 향후 통합된 AI 시스템을 통한 스트레스 관리에 대한 디자인 고려사항을 제안하고 있습니다. 이 연구는 개인 맞춤형 정신 건강 도구의 개발에 기여하기 위해 착용 장치와 생성 AI의 교차점을 탐구하는 첫 번째 단계로 자리 잡고 있습니다.



### Requirements for Quality Assurance of AI Models for Early Detection of Lung Cancer (https://arxiv.org/abs/2502.17639)
Comments:
          12 pages incl. 2 figures, 2 charts, and references, summary in English (page 2), article in German (original title: Anforderungen an die Qualitätssicherung von KI-Modellen für die Lungenkrebs-Früherkennung)

- **What's New**: 이 논문은 폐암 검출에서 AI의 역할을 강화하기 위한 시스템적 품질 보장 방안을 제안합니다. 기존 AI 시스템의 훈련 데이터와 성능 차이로 인해 소프트웨어 선택과 규제 평가가 복잡해진 점을 지적합니다. 특히, 유럽연합(EU) AI 법률에 따라, AI 기반 결절 탐지 및 특성화의 일관된 품질 보장이 필요하다고 강조합니다.

- **Technical Details**: 제안된 품질 보장 시스템은 검증된 참조 데이터셋에 기반하여 실제 스크리닝 사례와 가상 데이터(phantom data)를 포함합니다. 이는 부피(volume) 및 성장률(growth rate) 측정을 검증하기 위한 것입니다. 또한, AI 솔루션의 성능을 평가하기 위해 감도(sensitivity), 특이도(specificity), 부피 정확도(volumetric accuracy) 등의 기준에 기반한 표준화된 품질 평가가 필요하다고 언급합니다.

- **Performance Highlights**: 정기적인 데이터 업데이트는 인구 통계 변화와 기술 발전을 반영하여 AI의 지속적인 관련성을 보장합니다. 논문은 자가 학습 알고리즘(self-learning algorithms)과 그 업데이트에 대한 규제적 도전 과제를 다루고 있으며, 이는 기존 MDR 및 EU AI 법률이 충분히 다루지 않음을 강조합니다. 명확한 테스트 기준을 설정하고 참조 데이터를 체계적으로 활용함으로써, 성능 지표(performance metrics)를 비교 가능하게 만들고, 이에 따른 지침과 권고 사항을 마련할 수 있습니다.



### Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law (https://arxiv.org/abs/2502.17638)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 자연어 이해와 논리 기반 추론(logic-based reasoning)을 통합한 신경-상징적 접근법을 제안합니다. 이러한 방식은 법률 서비스에서 요구되는 높은 정확성, 반복성, 투명성을 해결하고자 하며, 주로 보험 계약에서의 적용 사례를 통해 그 가능성을 보여줍니다. LLM이 법률적 추론 능력을 개선했지만 여전히 복잡한 계약 분석을 위한 정확도와 일관성이 부족하다는 점이 강조됩니다.

- **Technical Details**: 연구에서는 세 가지 접근 방식을 테스트했습니다: 일반 LLM 방식, 비유도 방식, 그리고 유도 방식입니다. 비유도 방식에서는 LLM을 사용하여 보험 계약과 청구를 논리 표현(Logic Encoding)으로 변환하였습니다. 유도 방식에서는 법률 용어에 대한 구조적 프레임워크를 제공하여 LLM의 성능을 향상시키는 방법을 적용했습니다.

- **Performance Highlights**: 실험 결과, 유도 방식을 활용한 LLM과 논리 프로그램의 통합이 일반 LLM 방식보다 특정 법률 쿼리에 대한 성능이 뛰어남을 보여주었습니다. 특히, 보험 청구 보장 관련 질문에 대해 미리 정의된 문서에서 제공된 질문과 답변을 기준으로 한 성능 향상이 관찰되었습니다. 이 연구는 법률 쿼리를 해결하는 데 있어 의미 있는 개선을 이루는 방법을 제시하며, 향후 법률 AI 시스템의 발전에 기여할 가능성을 시사합니다.



### Theory-guided Pseudo-spectral Full Waveform Inversion via Deep Neural Networks (https://arxiv.org/abs/2502.17624)
Comments:
          26 pages, 23 figures, article paper

- **What's New**: 이 논문은 Deep Learning(딥 러닝) 기법을 활용하여 Full-Waveform Inversion(FWI)에 대해 새로운 접근 방식을 제안합니다. 기존의 시간 영역(time-domain) 기법의 장점을 기반으로 하면서, pseudo-spectral(위상 스펙트럼) 기법을 Deep Neural Network(DNN) 프레임워크 내에서 합성하여 활용합니다. 이는 이론 주도의 접근 방식으로, 기존 방법들과의 비교를 통해 다양한 이점을 확인하고자 합니다.

- **Technical Details**: FWI는 지진 데이터의 역문제를 해결하는 과정에서 다변량 최적화(multi-variate optimization)를 적용합니다. 이 과정에서 복잡한 파동 방정식(wave equation)을 기반으로 하여, 잔차 오차를 최소화하는 방향으로 모델을 업데이트합니다. 논문에서는 이러한 전통적인 접근 방식 대신, 신경망(Neural Network)을 도입하여 pseudo-spectral FWI 문제를 어떻게 딥러닝 알고리즘으로 재구성할 수 있는지를 설명합니다.

- **Performance Highlights**: 제안된 Recurrent Neural Network(RNN) 프레임워크는 기존의 FWI 방식보다 더 높은 정확도를 나타내었으며, 0.05의 오차 허용 범위와 1.45%의 상대적 오차를 기록했습니다. 또한, 이 방법은 결정을 더 안정적으로 진행하고, 단층(faults) 식별에 대해 더 나은 성능을 보여주었습니다. 이를 통해 획득한 데이터의 청정도는 기존의 방식 대비 뛰어난 성능을 입증하였으며, 특히 얕고 깊은 영역에서의 경계 탐지에 유리한 결과를 보여 주었습니다.



### Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations (https://arxiv.org/abs/2502.17618)
Comments:
          Extended version of an identically-titled paper accepted at AAMAS 2025

- **What's New**: 이번 연구에서는 복잡한 순차적 작업에서 팀원 간의 조정을 위해 'DTIL'이라는 새로운 계층적 Multi-Agent Imitation Learning (MAIL) 알고리즘을 도입했습니다. 기존의 MAIL 방법들이 동질적 시연(demonstrations)을 가정하는 데 비해, DTIL은 이질적 시연을 효과적으로 학습할 수 있도록 설계되었습니다. 이러한 접근법은 팀 구성원이 각기 다른 정책(policy)을 가지고 있음을 반영하며, 이로 인해 보다 유연한 팀 행동 모델링이 가능합니다.

- **Technical Details**: DTIL은 각 팀 구성원을 계층적 정책으로 표현하고, 이질적 팀 시연에서 이러한 정책을 분산 방식으로 학습합니다. 분포 정합(distribution-matching) 접근법을 사용하여 누적 오류(compounding errors)를 줄이며, 긴 시간(horizons)과 연속 상태(state) 표현에 효과적으로 확장할 수 있도록 합니다. 이는 복잡한 작업을 처리할 수 있는 능력을 증가시킵니다.

- **Performance Highlights**: 실험 결과, DTIL은 기존의 MAIL 기준선(baselines)에 비해 우수한 성능을 보였으며, 다양한 협업 시나리오에서 팀 행동을 정확하게 모델링했습니다. 이 알고리즘은 팀원의 협력적 수행에 있어 보다 나은 결과를 달성함으로써, 다중 에이전트 시스템과 인간-AI 팀을 위한 강력한 기초를 마련합니다.



### Flexible Counterfactual Explanations with Generative Models (https://arxiv.org/abs/2502.17613)
Comments:
          28 pages, 13 figures

- **What's New**: 이번 논문에서는 사용자들이 보다 유연하게 반사실적 설명을 적용할 수 있도록 지원하는 Flexible Counterfactual Explanations(FCEGAN) 프레임워크를 소개합니다. 기존 방법들이 고정된 변경 가능한 특성을 기반으로 하여 제한적임에 반해, FCEGAN에서는 사용자가 예측 시점에서 변경 가능한 특성을 동적으로 지정할 수 있는 기능을 포함하고 있습니다. 이는 사용자 요청에 따라 최소한의 수정으로 원하는 모델 예측 결과를 얻도록 합니다.

- **Technical Details**: FCEGAN은 Generative Adversarial Networks(GANs)를 활용하여 사용자 정의 제약 조건에 맞춰 설명을 조정할 수 있도록 설계되었습니다. 이 프레임워크는 반사실적 템플릿(counterfactual templates)을 도입함으로써 사용자가 어떤 특성을 변경할 수 있는지를 codify하고, 이를 통해 설명의 유효성과 실제 데이터의 밀접함을 보장합니다. 또한, FCEGAN은 블랙박스 환경에서도 작동 가능하여 모델 내부 접근 없이 예측 데이터를 활용하여 설명을 생성할 수 있습니다.

- **Performance Highlights**: FCEGAN의 성능은 경제 및 헬스케어 데이터셋을 통한 실험을 통해 기존의 전통적 방법들과 비교했을 때, 반사실적 설명의 유효성을 크게 향상시키는 것으로 나타났습니다. 특히, FCEGAN은 사용자 주도의 유연성과 블랙박스 호환성을 결합하여, 사용자 제약에 맞춘 개인화된 설명 생성을 지원합니다. 이러한 접근은 금융 및 헬스케어와 같은 분야에서 사용자 제약과 해석 가능성이 필수적인 상황에 매우 적합합니다.



### SynthRAD2025 Grand Challenge dataset: generating synthetic CTs for radiotherapy (https://arxiv.org/abs/2502.17609)
Comments:
          22 pages, 8 tables, 4 figures; Under submission to Medical Physics, as dataset paper for the SynhtRAD2025 Grand Challenge this https URL

- **What's New**: SynthRAD2025 데이터셋은 합성 의료 이미징(synthetic imaging) 분야에서 중요한 발전을 이루고 있습니다. 이 데이터셋은 유럽의 여러 대학 병원에서 수집된 2362개의 사례를 포함하고 있으며, 합성 컴퓨터 단층촬영(synthetic computed tomography, sCT) 생성의 벤치마킹 플랫폼을 제공합니다. 이를 통해 알고리즘의 성능을 테스트하고 향상시키는 기회를 제공합니다.

- **Technical Details**: 데이터셋은 다양한 스캐너와 프로토콜을 사용하여 획득된 MRI-CT 및 CBCT-CT 쌍을 포함합니다. 이미지 전처리 과정에서는 강체 및 변형 가능한 이미지 정합(rigid and deformable image registration)을 통해 높은 품질의 이미지를 보장합니다. 모든 이미지는 MetaImage (.mha) 형식으로 제공되어 의료 이미지 처리 도구와의 호환성을 유지하며, 메타데이터는 구조화된 CSV 파일로 제공됩니다.

- **Performance Highlights**: SynthRAD2025 데이터셋은 훈련(65%), 검증(10%), 테스트(25%) 세트로 나누어 데이터셋의 무결성을 유지합니다. 이 데이터셋은 MRI 전용 sCT 생성, MR 유도 광자 및 양성자 치료, CBCT 기반의 용적 계산 등 다양한 방면에서 응용됩니다. 이는 개인 맞춤형 암 치료 및 적응 방사선 치료 발전에 크게 기여할 것입니다.



### Data-Driven Pseudo-spectral Full Waveform Inversion via Deep Neural Networks (https://arxiv.org/abs/2502.17608)
Comments:
          11 pages, 6 pages, review paper

- **What's New**: 이 연구는 Deep Learning(딥러닝) 방법을 활용하여 지진학적 FWI(Full Waveform Inversion)에 대한 새로운 접근 방식을 제시합니다. 기존의 시간 영역(time-domain) 접근 대신에 pseudo-spectral(유사 스펙트럼) 방법을 통합하여 데이터 기반(data-driven) DNN(Deep Neural Networks) 모델을 제안하고 있습니다. 이를 통해 기존 FWI 기법의 한계를 극복하고, 더 깊은 영역과 오버스러스트 지역에서 더욱 우수한 성능을 보여 줍니다.

- **Technical Details**: FWI에서는 파형 방정식으로부터 유도된 매개변수를 최적화하여 묘사하기 위해 multivariate optimization(다변량 최적화) 방법이 적용됩니다. 연구에서는 pseudo-spectral 방법을 Deep Learning 프레임워크에 통합하기 위해 이론적으로 FWI 문제를 재구성하고, 이는 synthetics data(합성 데이터)에서 평가되었습니다. 뉴럴 네트워크의 구조 및 학습 과정에서 가중치를 결정하기 위한 제곱합 오차(Sum of Squared Errors)처럼 일반적으로 사용되는 cost function 이 사용됩니다.

- **Performance Highlights**: 제안된 DNN 프레임워크는 기존의 결정론적 모델 및 시간 기반(time-based) 접근 방식과 비교하여 유의미한 성능 향상을 보여 주었습니다. 특히, 더 깊은 지층과 오버스러스트 구역에서 classical FWI보다 우수한 결과를 도출하여 DNN의 가능성을 증명합니다. 향후 연구 방향으로는 pseudo-spectral DNN 방법의 한계 분석 및 추가적인 발전 가능성이 논의됩니다.



### PICASO: Permutation-Invariant Context Composition with State Space Models (https://arxiv.org/abs/2502.17605)
Comments:
          Published in The Thirteenth International Conference on Learning Representations, ICLR 2025

- **What's New**: 본 논문에서는 Large Language Models (LLMs)에 retrieval 기능을 추가하여 외부 지식을 효과적으로 활용할 수 있는 방법을 제안합니다. 특히, 여러 개의 pre-computed 상태를 효율적으로 조합하여 고품질 출력을 생성하는 방법인 PICASO(Permutation-Invariant Compositional Aggregation of States as Observations)를 도입합니다. 이 방법은 기존의 concatenation 방식과 비교하여 평균 5.4배 빠르며 성능 향상도 보여줍니다.

- **Technical Details**: PICASO는 State Space Models (SSMs)을 활용하여 정보가 포함된 여러 맥락들을 효과적으로 조합합니다. SSM의 동작에서 유도된 수학적 관계를 이용하여 여러 상태를 하나로 합치는 방법을 개발하였고, 이를 통해 맥락의 순서가 결과에 미치는 영향을 최소화합니다. Dynamic Programming을 통해 상태를 평균화하여 계산의 효율성을 높이고, 다소 근사치를 허용함으로써 선형 시간으로 계산할 수 있습니다.

- **Performance Highlights**: 실험 결과, PICASO는 WikiText와 MSMARCO 데이터셋에서 91%의 성능 이득을 보였으며, 기존의 최강 성능을 가진 baseline과 동일한 성능을 유지하면서도 더 빠른 처리 속도를 자랑합니다. 또한, 사전 훈련된 Mamba-2 2.7B 모델을 사용하여 단 하루의 fine-tuning으로 concatenation과 동등한 성능을 나타냈습니다.



### Hallucination Detection in LLMs Using Spectral Features of Attention Maps (https://arxiv.org/abs/2502.17598)
Comments:
          Preprint, under review

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 환각(hallucination) 탐지를 위한 새로운 방법을 제안합니다. 기존의 attention map 기반 방법은 한계가 있었던 반면, 제안된 LapEigvals 방법은 Laplacian matrix의 상위-k 고유값을 활용하여 더욱 정확한 탐지가 가능합니다. 실험 결과, 이 방법은 최신의 환각 탐지 성능을 달성하였으며, 이를 통해 향후 연구의 방향성을 제시합니다.

- **Technical Details**: 제안된 LapEigvals 방법은 attention maps를 그래프 구조의 인접 행렬(adjacency matrix)로 해석하여 이를 통계적으로 분석합니다. 이를 통해 attention maps에서 유도된 Laplacian matrix의 고유값(eigenvalues)을 사용하여 환각을 탐지하는 Probe 모델의 입력 특성으로 활용합니다. 연구 결과, Laplacian의 고유값이 이전의 방법들보다 환각과 더 밀접한 관련을 가지고 있음을 보여주었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 LapEigvals는 기존 AttentionScore 방법보다 우수한 성능을 입증했으며, 다양한 데이터셋과 LLM에 걸쳐 이러한 결과가 나타났습니다. 또한, ablation study를 통해 자원의 견고성과 일반화 가능성을 강조하며, 향후 환각 탐지 방법의 발전에 기여할 논의의 기반을 마련했습니다.



### Synergizing Deep Learning and Full-Waveform Inversion: Bridging Data-Driven and Theory-Guided Approaches for Enhanced Seismic Imaging (https://arxiv.org/abs/2502.17585)
Comments:
          20 pages, 14 images, literature review

- **What's New**: 이번 리뷰 논문은 Deep Learning(DL)과 Full Waveform Inversion(FWI)의 통합을 다루며, 수치해석에서의 더 나은 지진 이미징과 지하 구조 특성화 가능성을 탐구합니다. FWI와 DL의 기본 원리를 설명하고, 속도 추정, 복원, 단층 촬영 등 다양한 지구물리학적 응용을 검토합니다. 또한, 모델 복잡성, 데이터 품질과 같은 기존의 도전 과제와 함께, 하이브리드 및 물리 기반 모델을 통한 향후 연구 방향에 대해서도 논의합니다.

- **Technical Details**: FWI는 기록된 데이터와 일치하는 최적의 속도 모델 및 기타 암석 특성(밀도, 소산 흡수, 이방성)을 도출하기 위한 기법으로, 두 가지 주요 접근 방식인 전역 최적화(global optimization)와 직렬 해결(direct solving)을 사용합니다. 전역 최적화 방법으로는 몬테카를로(Monte Carlo) 방법, 유전 알고리즘(Genetic Algorithm), 시뮬레이티드 어닐링(Simulated Annealing) 등이 있으며, 이들은 각각 무작위 검색, 생물학적 진화의 유사, 통계 역학적 물리 등을 기반으로 하고 있습니다. 반면, 지역 최적화(local optimization) 방법은 1980년대에 소개되었으며, 관측된 데이터와 모델 데이터 간의 오차를 최소화하는 방식으로 적용됩니다.

- **Performance Highlights**: FWI와 DL의 통합은 지진 이미징 기술에 혁신적인 변화를 가져올 수 있는 잠재력을 지니고 있습니다. 이 두 접근 방식의 시너지를 통해 연구자들은 불완전한 데이터와 소음에 대한 강건성을 개선할 수 있으며, 계산의 효율성과 정확성을 동시에 높일 수 있는 기회를 갖게 됩니다. 이 리뷰는 FWI와 DL의 통합이 지하 정보를 보다 정확하고 효율적으로 추출할 수 있는 새로운 가능성을 열어줄 것이라고 강조하고 있습니다.



### Training a Generally Curious Agen (https://arxiv.org/abs/2502.17543)
- **What's New**: 이 논문에서는 PAPRIKA라는 새로운 방법론을 소개합니다. PAPRIKA는 언어 모델이 특정 환경에 국한되지 않고 일반적인 의사결정 능력을 키울 수 있도록 하는 미세 조정(fine-tuning) 접근법입니다. 이 방법은 다양한 전략을 요구하는 여러 작업에서 생성된 합성(interaction data)을 기반으로 훈련하여 모델이 환경 피드백을 통해 새로운 작업에 적응할 수 있도록 학습합니다.

- **Technical Details**: PAPRIKA는 언어 모델이 다양한 과제에 대한 정보 추출 및 의사결정을 수행하도록 설계된 텍스트 기반 의사결정 작업의 식이로 구성됩니다. 이를 위해 기본 모델(base model)을 사용하여 상호작용 궤적(interaction trajectories)을 생성하고, 성공률에 따라 점수를 부여합니다. 또, Direct Preference Optimization의 변형을 통해 성공적인 궤적의 상대적 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, PAPRIKA로 미세 조정된 모델은 추가적인 훈련 없이도 완전히 새로운 작업에 학습된 의사결정 능력을 효과적으로 이전할 수 있음을 보여주었습니다. 이 연구는 합성 데이터 생성의 유용성을 강조하며, AI 시스템이 외부 세계와의 상호작용을 통해 새로운 순차적 의사결정 문제를 자율적으로 해결할 수 있는 가능성을 제시합니다.



### PosterSum: A Multimodal Benchmark for Scientific Poster Summarization (https://arxiv.org/abs/2502.17540)
Comments:
          This paper includes a dataset of research posters with abstracts. We provide two cited examples ( arXiv:2211.11880 and arXiv:2210.07571 ) to illustrate reference summaries

- **What's New**: 이번 논문에서는 PosterSum이라는 새로운 멀티모달 벤치마크를 도입하여 시각적으로 복잡한 내용인 학술 포스터를 연구 논문 초록으로 요약하는 모델 개발을 지원하고자 합니다. PosterSum 데이터셋은 기계 학습 회의에서 발표된 16,305개의 포스터와 해당 초록을 포함하고 있으며, 이는 다양한 시각적 이해 도전과제를 제공합니다. 이 연구는 최신 Multimodal Large Language Models (MLLMs)가 학술 포스터 요약에서 직면하는 한계를 강조합니다.

- **Technical Details**: PosterSum 데이터셋은 각 포스터를 이미지 형식으로 제공하고 있으며, 포스터는 복잡한 레이아웃, 밀집한 텍스트 영역, 표 및 그림 등의 다양한 시각적 도전과제를 나타냅니다. 본 연구에서는 Segment & Summarize라는 계층적 방법을 제안하여 각 포스터를 일관된 영역으로 분할하고, 각 영역의 텍스트를 추출하여 지역 요약을 생성한 후, 이를 종합하여 포스터 전체를 아우르는 요약을 작성합니다. 이 과정에서는 추가 학습이나 미세 조정이 필요하지 않아 효율적입니다.

- **Performance Highlights**: 제안된 방법은 ROUGE-L 점수 24.18을 달성하여 기존의 MLLMs를 초과하며, 이는 학술 포스터 요약에서 새로운 기준을 설정합니다. PosterSum 데이터셋은 향후 멀티모달 과학 포스터 이해 연구에 기여할 수 있는 기초 자료가 됩니다. 또한, 이 연구는 MLLMs의 파인튜닝에의 유용성을 입증하여 제로샷 결과에 비해 유망한 개선 결과를 보여줍니다.



### On the Vulnerability of Concept Erasure in Diffusion Models (https://arxiv.org/abs/2502.17537)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 개인 정보 보호 및 보안 문제를 다루고 있습니다. 특히, 저작권이 있는 이미지나 해로운 이미지의 생성과 관련된 우려가 커지고 있는 가운데, 기존의 데이터 삭제 방법(즉, transfer learning을 통한 개념 지우기)에는 취약점이 있음을 보여줍니다. 그에 따라, RECORD라는 새로운 알고리즘을 도입하여 삭제된 내용을 환기할 수 있는 프롬프트를 발견하는 방법을 제안합니다.

- **Technical Details**: RECORD는 좌표 하강(Coordinate Descent) 기반의 알고리즘으로, 세밀한 프롬프트 최적화를 통해 삭제된 내용의 생성을 유도할 수 있는 입력을 찾아냅니다. 또한, 기존 방법에 비해 공격 성공률을 10배 높인 것으로 나타났습니다. 이 연구는 개념이 지워진 모델들이 적대적 공격(Adversarial Attack)에 n하기 보다 취약하다는 새로운 사실을 밝히고 있습니다, 이는 모델이 여전히 원치 않는 데이터의 정보를 보유하고 있다는 것을 시사합니다.

- **Performance Highlights**: RECORD는 현재의 최첨단 공격 방식에 비해 탁월한 성능을 보이며, 다양한 실험을 통해 지워진 모델의 동작을 탐구합니다. 또한, 유의미한 세밀한 활성화를 비교하면서, 기존의 ‘지우기’ 방법들이 실질적으로는 개념을 제거하기보다는 모델을 비정렬(misalignment)로 변환시키고 있음을 강조합니다. 이러한 위험은 저작권 관련 법률에 중대한 영향을 미칠 수 있습니다.



### The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? (https://arxiv.org/abs/2502.17535)
- **What's New**: 모델 압축 및 KV 캐시 압축이 LLM의 계산 및 저장 비용을 줄이기 위해 많은 관심을 받고 있습니다. 본 논문에서는 retrieval-augmented generation, multi-step reasoning, 외부 도구 사용과 같은 최근 LLM의 발전을 검토하고, 특정 LLM과 작업에 대해 더 작은 로또 LLM이 동일한 성능을 달성할 수 있다는 가설을 제시합니다.

- **Technical Details**: 현재 대부분의 LLM 압축 방법은 perplexity와 같은 기본적인 언어 작업에서만 성능을 보장하며, 실제 산업 상황에서는 충분한 성과를 보이지 않습니다. LLMs의 압축으로 인해 long-context retrieval 및 reasoning 능력이 감소할 수 있으며, KV 캐시 압축 또한 LLM의 긴 컨텍스트 이해 능력을 크게 제한합니다.

- **Performance Highlights**: 적응형 지식 검색(adaptive knowledge retrieval)에 관한 최근 연구는 LLM의 성능 향상과 모델 사이즈 및 지식 베이스 사이의 최적의 균형을 찾는 데 기여할 것으로 기대되고 있습니다. RAG(retrieval-augmented generation) 접근 방식은 특정 분야의 LLM 성능을 획기적으로 개선할 수 있음을 보여주며, 법률, 의료, 금융 분야에서의 LLM 적용 가능성을 높이고 있습니다.



### From Euler to AI: Unifying Formulas for Mathematical Constants (https://arxiv.org/abs/2502.17533)
Comments:
          50 pages, 6 figures

- **What's New**: 본 연구에서는수학 상수 $ank{	ext{π}}$와 관련된 방정식의 상관관계를 식별하고 통합하는 시스템을 제안했습니다. 이를 통해 457,145개의 arXiv 논문을 분석하여, 37%의 수학적 공식이 단일 수학 객체에서 유도될 수 있음을 제시했습니다. 이러한 접근은 AI 구동 발견을 위한 기초가 될 수 있습니다.

- **Technical Details**: 연구 방법론은 LaTeX 소스 코드를 활용하여 공식을 추출합니다. 278,242,582개의 고유한 문자열을 추출한 후, $	ext{π}$ 기符와 관계된 121,684개의 공식을 걸러내고 이들을 GPT-4o를 통해 분류합니다. 최종적으로, 시스템은 수학적 공식을 저비용으로 분석하고 유효성을 검증하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 이 연구에서는 $	ext{π}$와 같은 수학적 상수 뿐만 아니라, 그 외 다른 상수들도 다루어 넓은 적용 가능성을 보여주었습니다. 제안된 알고리즘은 기존 방식보다 더 효과적으로 수학적 발견을 수집하고 검증할 잠재력을 가지고 있습니다. 그 결과, 이 시스템은 방대한 수학 지식을 자율적으로 통합하는 가능성을 열어줍니다.



### Laplace-Beltrami Operator for Gaussian Splatting (https://arxiv.org/abs/2502.17531)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS)를 활용한 기하학적 처리를 위한 새로운 방법을 제안합니다. 특히, Mahalanobis distance를 사용하여 Laplace-Beltrami operator (LBO)를 직접 계산하는 접근 방식을 소개합니다. 기존의 방법들이 포인트 클라우드를 사용하여 정보를 손실시키는 반면, 본 연구는 분산 정보를 활용하여 정확도를 높였습니다.

- **Technical Details**: 3DGS는 장면을 3D Gaussian 분포의 집합으로 표현하며, 이와 관련된 기하학적 처리 기술적인 요구가 증가하고 있습니다. 본 연구의 핵심은 LBO의 정의를 Gaussian Splatting에서 직접 계산하고 이 과정에서 분산 정보를 포함하여 더 나은 표면 방향 추정을 수행하는 것입니다. 이를 통해 출력의 품질을 최적화 과정에서 평가할 수 있는 방법이 제시됩니다.

- **Performance Highlights**: 실험 결과, 제안된 LBO는 전통적인 기하학적 처리 애플리케이션인 거리 계산 및 형태 매칭에서 높은 성능을 보였습니다. 또한, 3DGS에서도 안정적인 기하학적 특성을 나타내며, 연구자는 이를 통해 다양한 후속 연구를 위한 데이터셋을 공유할 예정입니다. 이러한 방식은 기하학적 질감이 복잡한 상황에서도 효과적으로 사용될 수 있는 가능성을 보여줍니다.



### Perceptual Noise-Masking with Music through Deep Spectral Envelope Shaping (https://arxiv.org/abs/2502.17527)
- **What's New**: 이번 논문에서는 사람들의 음악 감상 환경에서 소음으로부터 자신을 분리하기 위한 방안을 제안합니다. 음악 신호가 소음의 주파수 성분을 마스킹하는 효과를 활용하여, 야구적 마스킹(psychoacoustic masking) 모델에 기반한 신경망(neural network)을 개발했습니다.

- **Technical Details**: 제안된 모델은 예측된 필터 주파수 응답(predicted filter frequency responses)을 통해 음악의 스펙트럼 환경(spectral envelope)을 재형성합니다. 이 모델은 원본 음악 믹스와 사용자가 선택한 리스닝 레벨을 유지하면서 소음을 효과적으로 마스킹하는 두 가지 제약 조건을 균형 있게 맞춘 지각 손실 함수(perceptual loss function)로 훈련됩니다.

- **Performance Highlights**: 시뮬레이션된 데이터로 사용자가 소음 환경에서 헤드폰으로 음악을 감상할 때의 경험을 재현하여 접근 방식을 평가했습니다. 정의된 객관적 메트릭(objective metrics)에 기반한 결과는 제안된 시스템이 기존 기술(state of the art)을 향상시킴을 보여줍니다.



### Multimodal Bearing Fault Classification Under Variable Conditions: A 1D CNN with Transfer Learning (https://arxiv.org/abs/2502.17524)
- **What's New**: 본 연구는 다중 모달 베어링 결함 분류 접근법을 제안하고 있으며, 이것은 진동(vibration) 신호와 모터 위상 전류(motor phase current) 신호를 1차원 합성곱 신경망(1D CNN) 프레임워크 내에서 결합하여 결함 감지의 정확성을 높이고자 합니다. 이 방법은 여러 신호에서 특징을 융합하여 정확도를 개선하며, L2 정규화의 추가로 모델은 96%의 정확도를 달성하였습니다. 이 연구는 다양한 운영 조건에서 강력한 성능을 보이는 것을 확인했습니다.

- **Technical Details**: 연구에서 다룬 1D CNN 모델은 진동과 전류 신호를 기반으로 하여 결함을 분류하는 방식으로, 최대 풀링(max pooling) 레이어까지의 파라미터를 보존하고 이후 레이어를 조정하는 전이 학습(transfer learning) 전략이 가장 높은 성능을 발휘했습니다. 또한, 이 접근법은 연산 자원이 제한된 환경에서도 적절한 대안을 제공하고 있습니다. 하지만 이는 더 많은 훈련 가능한 파라미터를 요구하여 연산 시간의 증가를 수반합니다.

- **Performance Highlights**: 이 다중 모달 1D CNN 프레임워크는 변동하는 작동 조건에서도 높은 정확도와 적응성을 제공하며, 산업 환경에서의 베어링 결함 분류의 정확도를 높이는 기반을 마련했습니다. 모델은 다양한 테스트에서 평균 96% 이상의 정확도를 달성하였으며, 기존의 단일 모달 신호 기반 방법보다 향상된 결과를 보여주었습니다. 본 연구 결과는 향후 베어링 결함 진단 시스템 발전에 기여할 것으로 기대됩니다.



### Spectral Theory for Edge Pruning in Asynchronous Recurrent Graph Neural Networks (https://arxiv.org/abs/2502.17522)
- **What's New**: 이 논문은 Asynchronous Recurrent Graph Neural Networks (ARGNNs)을 활용한 그래프 구조 데이터의 복잡한 의존성을 모델링하는 새로운 방법을 제안합니다. 특히, 그래프 스펙트럼 이론을 기반으로 하는 동적 가지치기(dynamic pruning) 기법을 이용하여 불필요한 에지를 효율적으로 제거함으로써 성능 저하 없이 모델의 계산 비용을 줄이는 것을 목표로 합니다.

- **Technical Details**: 제안하는 프루닝 방법은 네트워크 그래프의 라플라시안(Laplacian) 고유값의 허수 구성 요소를 활용하여 동적으로 처리됩니다. 이러한 접근법은 ARGNN의 복잡도를 줄여주고, 동적 그래프의 학습을 위한 효율성을 높이는 데 기여합니다. 결과적으로, 이 방법은 기존의 복잡한 모델링 요구를 감소시키며, 계산 시간을 단축하는 효과를 가져옵니다.

- **Performance Highlights**: 실험 결과는 제안된 동적 가지치기 방법이 ARGNN의 성능을 크게 저하시키지 않으면서도, 모델의 효율성을 현저히 향상시킴을 나타냅니다. 다양한 데이터셋에서 테스트한 결과, 이 방법의 효과는 여러 가지 상황에서 확인되었습니다. 따라서, 이 연구는 그래프 신경망 관련 연구 및 실제 응용에 있어 새로운 방향성을 제시합니다.



### Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation (https://arxiv.org/abs/2502.17521)
Comments:
          Github Link: this https URL

- **What's New**: 이번 논문에서는 데이터 오염(data contamination)의 위험을 줄이기 위해 고안된 정적(static)에서 동적(dynamic) 벤치마킹 방식의 변화에 대한 깊이 있는 분석을 수행합니다. 특히 동적 벤치마킹의 표준화된 평가 기준이 부족하다는 점을 강조하며, 이를 개선하기 위한 사례들을 제안합니다. 논문은 기존 연구의 한계점을 짚고, LLM(대규모 언어 모델)의 벤치마킹 방법론에 대한 종합적인 개요를 제공합니다.

- **Technical Details**: 본 연구는 정적 벤치마킹의 한계를 지적한 후, 시간에 따라 업데이트되는 벤치마킹 데이터세트를 사용하는 동적 벤치마킹 방법론을 제안합니다. 학습 단계에서 벤치마크 데이터가 모델의 훈련 데이터와 겹치지 않도록 하기 위해 다양한 기술적 접근 방식을 소개하며, 데이터 암호화(data encryption) 및 후속 오염 탐지(post-hoc contamination detection)와 같은 방법들이 있습니다. 하지만 이러한 정적 방법들의 한계로 인해 새로운 동적 벤치마킹 스킴이 도입되었습니다.

- **Performance Highlights**: 가장 주목할 만한 점은 기존의 동적 벤치마크가 제안된 평가 기준을 완전히 만족시키지 못한다는 것입니다. 이로 인해 기존의 평가 방법들이 모델의 실제 성능을 왜곡할 수 있는 문제점을 내포하고 있다는 것을 암시합니다. 이 논문에서는 데이터 오염의 위험을 줄이기 위한 동적 벤치마킹 방법에 대한 체계적인 조사를 실시하여, 향후 연구 방향에 대한 귀중한 인사이트를 제공합니다.



### Ensemble RL through Classifier Models: Enhancing Risk-Return Trade-offs in Trading Strategies (https://arxiv.org/abs/2502.17518)
Comments:
          16 pages,5 figures, 1 table

- **What's New**: 이 논문은 금융 거래 전략에서 앙상블 강화 학습(ensemble Reinforcement Learning, RL) 모델의 활용을 종합적으로 연구하며, 분류기(classifier) 모델을 통해 성능 향상을 도모합니다. A2C, PPO, SAC와 같은 RL 알고리즘을 전통적인 분류기인 서포트 벡터 머신(Support Vector Machines, SVM), 결정 트리(Decision Trees), 로지스틱 회귀(Logistic Regression)와 조합하여 위험-수익 무역을 개선하는 방법을 조사합니다. 다양한 앙상블 방법을 평가하여 핵심 재무 지표(Cumulative Returns, Sharpe Ratios, Calmar Ratios, Maximum Drawdown)에서 개별 RL 모델과 비교한 결과, 앙상블 방법이 항상 기초 모델을 초과하는 성과를 보이는 것을 확인했습니다.

- **Technical Details**: 위험 조정 수익(risk-adjusted returns) 측면에서 앙상블 방법이 기초 모델보다 뛰어난 성능을 나타내며, 드로우다운(drawdowns) 관리를 향상시키고 안정성을 제공합니다. 하지만 앙상블 성능은 분산 임계값(variance threshold) 선택에 민감하여, 최적 성과를 위해서는 동적인 분산 임계값 조정의 중요성을 강조합니다. 제안된 방법은 여러 RL 에이전트와 분류기로부터 신뢰도 점수를 집계하고, 분산 평가 메커니즘을 통한 신뢰할 수 없는 추정 필터링을 통해 의사 결정의 신뢰성을 향상시키기 위한 세 가지 핵심 요소로 구성됩니다.

- **Performance Highlights**: 이 연구에서는 제안된 앙상블 방법이 높은 신뢰도와 다양한 시장 환경에서의 적응 능력을 바탕으로 안정적인 의사 결정을 가능하게 한다고 주장합니다. 특히, 다양한 RL 에이전트의 출력 집계를 통해 탐색(exploration)과 활용(exploitation)을 개선하며, 신뢰도가 높은 의사 결정 시 소극적인 접근 방식을 채택함으로써 불확실성이 높은 상황에서의 신뢰성을 보장합니다. 이러한 점에서 이 논문은 금융 거래, 로봇 공학 및 기타 동적 환경에서 RL과 분류기의 결합이 가진 가치를 강조합니다.



### Attention-based UAV Trajectory Optimization for Wireless Power Transfer-assisted IoT Systems (https://arxiv.org/abs/2502.17517)
- **What's New**: 본 논문에서는 Attention 기반의 UAV Trajectory Optimization (AUTO) 프레임워크를 제안합니다. 이 프레임워크는 Attention Trajectory Optimization Model (ATOM)과 Actor-critic에 기반한 Trajectory lEarNing Method (TENMA)로 구성되어, UAV의 경로와 수를 최적화하는 혁신적인 방법을 제공합니다. 기존 연구들이 UAV와 WPT를 단편적으로 다룬 것과 달리, 본 연구는 두 가지 접근 방식을 통합하여 효과성을 높였습니다.

- **Technical Details**: ATOM 모델에서는 IoT 시스템을 그래픽 구조로 수학적으로 표현하고, 그래프 인코더를 사용하여 모든 IoTD의 self-attention 특징을 도출합니다. 이 모델은 UAV의 경로를 생성하는 디코더와 결합되어 있으며, 실제 보상을 기준으로 사용하여 critic 네트워크의 분산을 줄이고 TENMA의 안정성과 일반화를 향상시키는 Actor-Critic 학습 방법을 채택합니다.

- **Performance Highlights**: 제안된 AUTO 프레임워크는 대규모 다중 UAV 경로 계획을 지원하며, 다양한 실험과 하드웨어 필드 실험을 통해 그 효율성과 실행 가능성을 검증합니다. 자동으로 최적화된 UAV 경로는 데이터 전송 지연을 감소시키고, 전체 시스템에서의 에너지 사용 효율성을 증대시킵니다.



### A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models (https://arxiv.org/abs/2502.17516)
Comments:
          30 pages, 4 Figures, 10 Tables

- **What's New**: 이번 논문은 다중 모달 기반 모델(Multimodal Foundation Models, MMFMs)의 해석 가능성(Interpretability) 향상에 초점을 두고 있습니다. 저자들은 특히 대형 언어 모델(Large Language Models, LLMs)과 다중 모달 모델 간의 해석 가능성 차이를 줄이고자 다양한 접근 방식을 제시하고 있습니다. 이 연구는 LLM 해석 방법을 MMFM에 적용할 수 있는지, 그리고 이러한 방법이 유사한 통찰력을 제공하는지를 탐구합니다.

- **Technical Details**: 저자들은 다중 모달 모델의 메커니즘을 이해하기 위해 기계적 해석 가능성(Mechanistic Interpretability)에 대한 새로운 3차원 분류 체계를 제안하였습니다. 이 체계는 (1) 모델 가족(Model Family), (2) 해석 기법(Interpretability Techniques), (3) 응용(Application)을 포함하며, 각각의 카테고리에서 발췌된 통찰력을 실제 응용과 연결합니다. 예를 들어, 비생성적 VLM 모델(CLIP 등)과 생성적 VLM 모델(Stable-Diffusion 등) 간의 주요 차이점을 아우릅니다.

- **Performance Highlights**: 해석 가능성 연구는 LLM에서 아마도 가장 큰 발전을 이뤘지만, MMFM은 상대적으로 탐구가 부족합니다. 본 연구는 MMFM에서 발생하는 새로운 문제와 기존 비유형적 LLM 기반 해석 기법의 적용 가능성을 검토합니다. 이 연구는 MMFM 해석의 한계와 미래 연구 방향에 대한 통찰력을 제공하여 응용 문제인 모델 편집(Model Editing) 및 환각 억제(Hallucination Mitigation)의 발전을 도모합니다.



### Towards User-level Private Reinforcement Learning with Human Feedback (https://arxiv.org/abs/2502.17515)
- **What's New**: 이 연구에서는 기존의 RLHF(Reinforcement Learning with Human Feedback) 방법에서 사용자 프라이버시를 효과적으로 보호할 수 있는 새로운 프레임워크인 AUP-RLHF를 제안합니다. 기존 연구들은 주로 아이템 수준(item-level) 프라이버시 보호에 중점을 두었으며, 사용자 수준(user-level) 프라이버시는 제대로 다루지 못했습니다. 새로운 방법은 사용자 수준 레이블에 대한 차별적 프라이버시(Differential Privacy)를 통합하여 사용자 데이터의 개인 정보를 보호하는 동시에 개선된 추정 오차를 달성합니다.

- **Technical Details**: AUP-RLHF 알고리즘은 사용자 선호도에 대한 손실의 평균 그래디언트를 사용하여 파라미터 업데이트를 수행합니다. 전통적인 DP 알고리즘은 아이템 수준에 최적화되어 있어 사용자 수준에서는 유용성이 부족하다는 점을 지적하며, 고감도 데이터를 처리하기 위해 아웃라이어 제거 및 적응 샘플링 과정을 도입합니다. 알고리즘은 $(\\varepsilon, \\delta)$ 사용자 수준 프라이버시를 보장하며, 데이터에 노이즈를 추가하여 개인 정보를 보호합니다.

- **Performance Highlights**: 실험 결과, AUP-RLHF는 감정 생성(sentiment generation) 및 요약(summarization) 작업에서 기존의 다른 사용자 수준 DP 기법보다 뛰어난 성능을 보였습니다. 다양한 모델 크기와 다양한 프라이버시 파라미터 설정에 걸쳐 AUP-RLHF는 일관되게 우수한 프라이버시-유용성(trade-off) 균형을 달성했습니다. 이로써 이 연구는 사용자 프라이버시와 모델 유틸리티 간의 딜레마를 효과적으로 해결할 수 있는 가능성을 보여줍니다.



### SAE-V: Interpreting Multimodal Models for Enhanced Alignmen (https://arxiv.org/abs/2502.17514)
- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 해석 가능성을 높이기 위한 새로운 프레임워크인 SAE-V를 제안했습니다. SAE-V는 Sparse Autoencoders(SAEs) 개념을 확장하여 MLLM에 적용시키며, 이를 통해 모델의 동작과 데이터 품질을 세밀하게 해석할 수 있습니다. 이러한 접근 방식은 MLLM의 정렬 프로세스에서 데이터 필터링 기법을 활용하여 보다 효율적인 모델 정렬을 가능하게 합니다.

- **Technical Details**: SAE-V는 MLLM의 해석 가능성을 위해 설계된 메커니즘 해석 가능성 프레임워크입니다. 이 프레임워크는 텍스트와 이미지의 표현 공간을 융합하여 통합된 멀티모달 표현을 생성하는데 초점을 맞추고 있습니다. SAE-V는 MLLM의 교차 모달 상호작용을 어휘적으로 분석하여 모델의 품질과 정렬을 개선할 수 있는 기법을 제공합니다.

- **Performance Highlights**: 실험 결과, SAE-V 기반의 데이터 필터링 기법은 50%의 데이터로 110% 이상의 성능 향상을 달성했습니다. 이는 SAE-V가 MLLM의 해석 가능성과 정렬을 강화하는 데 효과적임을 입증합니다. 이와 함께 MLLM의 학습 과정에서의 특징 분포를 분석하여 멀티모달 이해 과제 성능과의 관계를 발견했습니다.



### Int2Int: a framework for mathematics with transformers (https://arxiv.org/abs/2502.17513)
- **What's New**: 이 논문은 수학 연구 문제, 특히 정수론과 관련된 문제를 해결하기 위한 Transformer 기반의 오픈 소스 코드 집합인 Int2Int를 소개합니다. Int2Int는 PyTorch로 완전하게 구현된 Transformer 아키텍처와 함께 교육 및 평가 루프를 제공하며, 일반적인 수학적 객체를 표현하고 생성하며 해독하는 클래스를 포함하고 있습니다. 공개된 데이터 준비 코드 및 실험 결과를 시각화하기 위한 Jupyter 노트북도 사용 가능합니다.

- **Technical Details**: Int2Int는 수학 문제를 문자 순서로 재작성하여 해결책을 찾아내는 방식으로 Transformer 아키텍처를 응용합니다. 특히, 두 개의 정수로 더하는 문제를 예시로, Int2Int는 입력과 출력을 위한 시퀀스를 모두 처리할 수 있는 기능을 제공합니다. Supervised learning 체계에 기반하여, 문제와 해결책의 쌍을 학습하여 예측을 수행하며, 이를 위해 cross-entropy 손실 함수를 사용합니다.

- **Performance Highlights**: Int2Int는 다양한 수학적 문제를 해결하는 데 사용할 수 있는 범용 프레임워크로, 직관적인 사용 설명서를 제공합니다. 특히 elliptic curve의 특성을 예측하거나 두 정수의 최대 공약수를 계산하는 문제에 대한 실용적인 튜토리얼을 포함합니다. 또한, MIT 라이센스 하에 제공되어 사용자가 자유롭게 활용하고 수정할 수 있는 장점이 있습니다.



### Recurrent Knowledge Identification and Fusion for Language Model Continual Learning (https://arxiv.org/abs/2502.17510)
- **What's New**: Recurrent-KIF는 동적 매개변수 중요도 추정을 통해 지식 전이를 향상시키는 새로운 연속 학습(Continual Learning, CL) 프레임워크입니다. 이 접근법은 두 가지 루프 구조를 활용하여 새로운 작업에 신속하게 적응하고, 과거 지식을 관리하는 데 중점을 둡니다. 이러한 동적인 중요도 분포 추정을 기반으로 Recurrent-KIF는 복잡한 환경에서의 효과적인 학습을 실제로 구현합니다.

- **Technical Details**: Recurrent-KIF는 내부 루프(inner loop)와 외부 루프(outer loop)의 협업을 통해 작동합니다. 내부 루프는 새로운 작업에 적응하며 중요한 매개변수를 식별하는 역할을 하며, 외부 루프는 새로운 및 역사적인 지식의 융합을 관리합니다. 이 과정에서 중복 지식 가지치기(redundant knowledge pruning)와 핵심 지식 병합(key knowledge merging)을 통해 지식 융합이 이루어집니다.

- **Performance Highlights**: 실험 결과, Recurrent-KIF는 CF(변별적 망각)와 KT(지식 전이) 문제를 효과적으로 완화하는 것으로 나타났습니다. 다양한 모델 아키텍처와 크기(770M에서 13B까지)에서 우수한 성능을 발휘하며, CL 벤치마크에서 기존의 최첨단 방법들을 능가했습니다. Recurrent-KIF는 지식 융합 지식(fusion of knowledge)을 조정할 때 각 단계에서 최신 중요도 분포에 따라 적응적으로 최적화하여 모델 훈련 과정을 개선합니다.



### C-3DPO: Constrained Controlled Classification for Direct Preference Optimization (https://arxiv.org/abs/2502.17507)
- **What's New**: 이 논문은 Direct Preference Optimization (DPO) 스타일 알고리즘을 암묵적 분류 알고리즘으로 재구성하는 새로운 관점을 제시한다. 이를 통해 다양한 DPO 스타일 알고리즘을 하나의 분류 프레임워크로 통합하고 확장하는 방법을 제안한다. 분석 결과, DPO 스타일 알고리즘의 기본 문제는 과소 지정(under-specified)되어 있어 승자-패자(winner-loser) 반응의 확률 붕괴(probability collapse)에 취약하다는 점을 강조한다. 이를 해결하기 위해 승자와 패자 간의 확률 질량 이동을 제어하는 제약 조건을 제안하며, 이를 통해 새로운 알고리즘인 C-3DPO를 개발하였다.

- **Technical Details**: 분류 프레임워크는 DPO 스타일 알고리즘을 한데 묶는 데 도움을 주며, 이 알고리즘들이 특정 분류 레이블과 손실 함수(loss function)의 선택에 따라 복원될 수 있음을 보여준다. 이 프레임워크는 기존의 이진 선호 쌍(preference pairs)뿐만 아니라 선호의 순위 리스트(ranked lists)와 같은 보다 풍부한 정보도 수용할 수 있다. 본문에서는 DPO 스타일 알고리즘이 두 개의 확률을 학습하기 위해 단일 제약 조건만 제공하여 과소 지정을 초래한다고 설명하고, 이를 해결할 수 있는 새로운 제약 조건 집합을 제안한다.

- **Performance Highlights**: C-3DPO는 여러 대형 언어 모델을 표준 선호 데이터셋으로 정렬하는 데 있어 기존의 DPO 알고리즘을 초월하는 성능 개선을 제공한다. 연구 결과, C-3DPO는 두 가지 표준 데이터셋과 최대 130억 개 파라미터를 가진 세 세트에서 vanilla DPO와 여러 다른 기준선(baseline)보다 뛰어난 성능을 보였다. 이는 최종 모델 평가에서 높은 품질을 보여주었다.



### RAG-Enhanced Collaborative LLM Agents for Drug Discovery (https://arxiv.org/abs/2502.17506)
Comments:
          Machine Learning, Drug Discovery

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 약물 발견에서 크게 활용될 가능성을 보여주고 있습니다. 그러나 이러한 생화학 데이터의 특수성 덕분에 비용이 많이 드는 도메인 특정 미세 조정이 필요하게 되어, 유연한 LLM의 적용을 방해하고 있습니다. 이에 대한 해결책으로 제안된 CLADD는 정보 검색을 통해 생화학 데이터의 도전 과제를 해결하고 효율적인 정보를 생산할 수 있는 시스템입니다.

- **Technical Details**: CLADD는 여러 LLM 에이전트가 협력하여 생물 의학 지식 기반에서 정보를 동적으로 검색하고, 관련 증거를 통합하여 응답을 생성하는 시스템으로, 도메인 특정 미세 조정이 필요하지 않습니다. 이 시스템은 데이터 이질성, 모호성 및 다원적 통합과 같은 핵심 장벽을 다룰 수 있는 구조를 가지고 있으며, 다른 팀들이 각기 다른 데이터 소스와 역할에 전문화되어 작업을 수행합니다.

- **Performance Highlights**: CLADD는 약물 발견의 다양한 작업에서 우수한 성능을 보여주며, 전통적인 딥러닝 접근 방식 및 특정 도메인 LLMs보다 뛰어난 결과를 시연했습니다. 특히, 이 프레임워크는 유연성과 설명 가능성을 강조하여, 과학자와 AI 간의 상호 작용을 개선합니다. 실험 결과에서는 CLADD의 효과성을 입증하며, 다양한 약물 발견 작업에 적용 가능성을 넓히고 있습니다.



### Inverse Surrogate Model of a Soft X-Ray Spectrometer using Domain Adaptation (https://arxiv.org/abs/2502.17505)
- **What's New**: 이번 연구에서는 소프트 X-Ray 분광계(soft X-ray spectrometer)에 대한 강력한 역 대리 모델(inverse surrogate model)을 개발했습니다. 실험적 조건을 최적화하기 위해 기계 학습(machine learning) 방법의 자동화를 추진하며, 실험의 출력인 감지기 이미지와 장비의 매개변수를 매핑하는 역 모델을 필요로 합니다. 제한된 실험 데이터 문제를 해결하기 위해 시뮬레이션 데이터(synthetic data)와 데이터 증강(data augmentation) 기법을 통해 최적의 자동 정렬 및 설정을 달성하고자 하였습니다.

- **Technical Details**: 우리의 연구에서는 BESSY II와 같은 전자 저장 링에서의 빔타임 동안 실험의 올바른 정렬 및 보정을 위해 다중 RZP(optical element) 모델을 지원하는 역 대리 모델을 제안합니다. 모델은 100,000개의 시뮬레이션을 통해 생성된 입력-출력 쌍으로 학습되며, 이를 통해 흥미로운 문제인 실험 이미지로부터 직접 매개 변수를 예측하는 방법을 가능하게 하였습니다. 강화학습(adversarial training) 방법을 통해 생성기(generator)와 분류기(discriminator) 두 개의 신경망이 동시에 학습하여 데이터의 특성 변화를 최소화합니다.

- **Performance Highlights**: 예비 실험에 따르면, 제안된 역 대리 모델은 실험에서 획득된 이미지를 효과적으로 해석하고, 소프트 X-Ray 분광계를 신속하게 조정하는 데 기여합니다. 데이터 증강 및 도메인 적응(domain adaptation) 기술을 통해, 시뮬레이션과 실험 간의 간극을 메우며 신뢰할 수 있는 출력 결과를 제공합니다. 궁극적으로, 이 방법은 자동화된 실험에서의 효율성을 크게 향상시키며, 기계 학습을 활용한 과학 장비의 새로운 가능성을 열어줍니다.



### Protein Large Language Models: A Comprehensive Survey (https://arxiv.org/abs/2502.17504)
Comments:
          24 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 Protein LLMs(단백질 대형 언어 모델)에 대한 포괄적인 개요를 제공하는 최초의 연구로, 기존의 서베이들이 특정 측면이나 응용에 중점을 두었던 것과는 달리, 이 논문에서는 구조, 훈련 데이터셋, 평가 지표 및 다양한 응용 분야를 다루고 있습니다. 이는 단백질 과학에서의 혁신적인 발전에 기여할 것입니다.

- **Technical Details**: 저자들은 100편 이상의 연구 논문을 체계적으로 분석하여 최신 Protein LLMs의 구조적 분류법을 제안합니다. 이 모델들은 방대한 단백질 서열 데이터(large-scale protein sequence data)를 활용하여 더 높은 정확도를 얻는 방법을 분석하며, 단백질 공학 및 생물 의학 연구에서의 잠재력을 탐구합니다.

- **Performance Highlights**: Protein LLMs는 단백질 구조 예측, 기능 주석 및 디자인에서 더 효율적인 성능을 보여주며, 과학적 발견을 위한 필수 도구로 자리매김하고 있습니다. 논문은 단백질 과학 내 미래의 도전 과제와 방향성에 대해서도 논의합니다.



### Doctor-in-the-Loop: An Explainable, Multi-View Deep Learning Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer (https://arxiv.org/abs/2502.17503)
- **What's New**: 이번 연구에서는 비소세포 폐암(NSCLC) 환자의 병리학적 반응(predicted pathological response, pR) 예측의 정확성을 높이고 신뢰성을 확보하기 위해 'Doctor-in-the-Loop'라는 새로운 프레임워크를 제안합니다. 본 프레임워크는 전문적인 도메인 지식을 인공지능 기술에 통합하여, 임상적으로 중요한 해부학적 영역에 모델의 초점을 맞추며, 예측의 해석 가능성과 투명성을 향상시킵니다. 이 방법은 신경망의 훈련 과정에 내재적(explainable AI) 설명성을 부여하여, 예측의 신뢰성과 의료적 맥락의 중요성을 반영합니다.

- **Technical Details**: 연구에서 제안하는 'Doctor-in-the-Loop' 방법론은 다각적(multi-view) 접근 방식을 통해 광범위한 맥락에서 특정 병변(lesion) 세부사항으로 점진적으로 모델의 초점을 조정합니다. 이 과정에서 의료 전문가의 도메인 지식을 학습 과정에 통합하여 치료 예측의 핵심 요소를 더욱 명확히 할 수 있습니다. 또한, 신경망 훈련 시 임상적 인사이트를 반영함으로써 모델이 의료적 통찰과 밀접하게 연결될 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 NSCLC 환자 데이터를 활용하여 제안한 방법이 높은 예측 성능을 보임을 입증하였으며, 투명하고 정당한 결과를 제공합니다. 기존의 최신 연구와 비교했을 때, 'Doctor-in-the-Loop' 접근 방식은 더 높은 예측 정확도와 임상적 관련성을 달성하여, 향후 환자 맞춤형 치료법 개발에 기여할 것으로 기대됩니다. 이러한 결과는 임상적 해설이 가능한 인공지능의 발전에 중대한 이정표가 될 것입니다.



### CoKV: Optimizing KV Cache Allocation via Cooperative Gam (https://arxiv.org/abs/2502.17501)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 주요 문제 중 하나인 메모리 소비를 효율적으로 해결하기 위해 CoKV라는 새로운 방법을 제안합니다. CoKV는 attention heads 간의 협업을 cooperative game으로 모델링하여, 각 head의 기여도를 평가하고 cache budget을 동적으로 할당합니다. 이러한 접근법은 기존의 방법들이 head의 중요성을 독립적으로 평가하는 것에서 벗어나, 협업의 중요성을 강조하는 점이 특징입니다.

- **Technical Details**: CoKV는 cooperative game theory에서 영감을 받아 Shapley value를 사용하여 각 attention head의 중요도를 평가합니다. 이 방법은 각 head의 기여도를 계산할 때, marginal contribution을 넘어서 complementary contribution을 사용합니다. 이러한 방식을 통해 전체 coalition을 평가하는 대신 몇 가지 선택된 coalition sizes에서 기여도를 계산하여 계산 비용을 상당히 줄일 수 있습니다.

- **Performance Highlights**: CoKV는 Llama-3-8B-Instruct와 Mistral-7B 모델을 사용하여 LongBench 벤치마크에서 최첨단 성능을 달성하였습니다. 실험 결과에 따르면, CoKV는 KV cache에 평균 128 KV pair를 유지하면서도 뛰어난 성능을 발휘하였으며, 기존의 기술들과도 잘 통합된다는 점을 보여주었습니다.



### Generalized Exponentiated Gradient Algorithms Using the Euler Two-Parameter Logarithm (https://arxiv.org/abs/2502.17500)
Comments:
          10 pages, preprint of Journal paper

- **What's New**: 이 논문은 Mirror Descent (MD) 접근법을 활용한 새로운 유형의 Generalized Exponentiated Gradient (GEG) 알고리즘을 제안하고, 두 매개변수가 있는 로그의 변형을 정규화 함수로 사용하는 방법을 조사합니다. 이 링크 함수는 일반화된 엔트로피(generalized entropies)와 밀접하게 연관되어 있습니다. 우리는 이 알고리즘이 훈련 데이터의 분포에 적응할 수 있도록 하며, 이를 통해 경량화된 기계 학습 업데이트를 가능하게 합니다.

- **Technical Details**: 제안된 GEG/MD 업데이트는 Euler 로그의 역함수를 근사화하는 일반화된 지수 함수를 추정하여 이끌어냅니다. 이 함수는 다양한 수치적 형태와 특성을 지닌 Euler 로그 및 변형된 지수 함수에 의해 조정 가능합니다. 두 개 이상의 하이퍼파라미터를 통한 학습을 통해 더욱 효과적인 성능 조정이 가능해집니다.

- **Performance Highlights**: 논문에서 개발된 알고리즘은 온라인 포트폴리오 선택(Online Portfolio Selection, OPLS) 문제에 적용되어 성능과 강인성을 개선하는 데 도움이 됩니다. 기존의 표준 EG 알고리즘과 비교할 때, 제안된 알고리즘은 다양한 시장 상황과 투자자의 선호에 효과적으로 적응할 수 있는 잠재력을 보여줍니다.



### Accuracy of Wearable ECG Parameter Calculation Method for Long QT and First-Degree A-V Block Detection: A Multi-Center Real-World Study with External Validations Compared to Standard ECG Machines and Cardiologist Assessments (https://arxiv.org/abs/2502.17499)
Comments:
          37 pages, 8 figures, 6 tables

- **What's New**: 최근 착용 가능한 장치가 심장 모니터링에 혁신을 가져왔습니다. 이 연구는 새로운 알고리즘인 FeatureDB를 통해 착용형 단일 리드 신호에서 ECG 매개변수를 자동으로 계산하는 방법을 평가했습니다. 이 연구는 기존 ECG 장치 및 전문가 의사 평가에 대한 정확성을 검증하기 위해 대규모 다기관 연구를 수행했습니다.

- **Technical Details**: 연구는 세 가지 다양한 데이터셋을 사용하였으며(AHMU-FH dataset, CSE dataset, HeartVoice-ECG-lite dataset), 모든 데이터는 두 명의 경험이 풍부한 심장 전문의에 의해 주석이 달렸습니다. FeatureDB는 PR interval, QRS duration, QT interval 등의 주요 ECG 매개변수와 표준 ECG 기계 및 임상의사들에 의해 계산된 결과와 통계적으로 유의관계를 나타냈습니다. Bland-Altman 분석을 통해 높은 정확도를 확인했습니다.

- **Performance Highlights**: FeatureDB는 Long QT syndrome (LQT) 및 atrioventricular block interval abnormalities (AVBI) 탐지에서 뛰어난 진단 성능을 보여주었습니다. LQT에서 ROC 곡선 아래 면적(AUC)은 0.836, AVBI는 0.861로 나타났으며, 정확도(accuracy)도 LQT 0.856, AVBI 0.845로 훌륭했습니다. 이러한 결과는 FeatureDB의 임상적 신뢰성을 뒷받침하며, 착용형 ECG 기술이 심혈관 질환 관리 및 조기 개입 전략에 통합될 수 있는 가능성을 강조합니다.



### Improving Value-based Process Verifier via Structural Prior Injection (https://arxiv.org/abs/2502.17498)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 Large Language Model(LLM) 추론 시나리오에서 Monte Carlo 샘플링을 통한 상태 가치 추정의 한계점을 극복하기 위해 구조적 사전 주입(structural prior injection) 방법을 제안합니다. Monte Carlo 추정이 가진 소음(noise)과 오차(errors)를 사전 정의된 범주형 분포의 기대값으로 매핑하여 해결합니다. 이 접근방식은 근본적으로 추정 오류를 분포 불일치 문제로 전환합니다.

- **Technical Details**: 논문에서는 Markov 결정 과정(Markov Decision Process, MDP) 및 Bellman 방정식을 통해 가치 기반 프로세스 검증기의 개념을 정립합니다. Monte Carlo 방법을 이용하여 상태 행동 가치를 추정하며, 샘플링의 한계로 인해 발생하는 오차를 통계 기반 거리(Statistics-based Distance)라는 새로운 미세 조정을 통해 측정합니다. 이 미세 조정은 다양한 목적 함수(mean-square error, cross-entropy)에 대한 최적화를 도와줍니다.

- **Performance Highlights**: 구조적 사전 주입을 통해 다양한 목표 함수에서 값 기반 프로세스 검증기의 성능을 향상시켜 약 1~2점의 성능 개선을 보여줍니다. 이는 적은 비용으로 실현되며, 동일한 최적 솔루션을 가지고도 서로 다른 구조적 사전(definition)에 따라 성능 차이가 크다는 것을 보여줍니다. 이러한 결과는 구조적 사전 주입이 미래 연구에서 중요한 방향임을 시사합니다.



### SpikeRL: A Scalable and Energy-efficient Framework for Deep Spiking Reinforcement Learning (https://arxiv.org/abs/2502.17496)
- **What's New**: 이번 연구에서는 SpikeRL이라는 새로운 프레임워크를 소개합니다. SpikeRL은 Spiking Neural Networks(SNNs)와 Deep Reinforcement Learning(DeepRL)의 시너지를 활용하여 복잡한 연속 제어 작업을 위한 효율적이고 확장 가능한 솔루션을 제공합니다. 특히, 우리는 PyTorch Distributed 패키지와 NCCL 백엔드를 사용하여 분산 훈련을 구현하고 혼합 정밀도 훈련(mixed precision training)을 최적화하여 모델의 성능을 개선했습니다.

- **Technical Details**: SpikeRL의 시스템 아키텍처는 DeepRL 기반 SNN 모델, MPI 및 NCCL 백엔드를 통한 분산 훈련, 혼합 정밀도 훈련의 세 가지 주요 구성 요소로 이루어져 있습니다. SNN 모델은 인구 인코딩(population encoding)과 디코딩을 통해 환경의 관측치를 스파이크로 변환하고, 깊은 비평가 네트워크(deep critic network)를 사용하여 액션을 평가합니다. 여기서 스파이크의 인코딩은 가우시안 분포를 사용해 연속 입력 값을 스파이크 트레인으로 변환하여 네트워크의 표현 용량과 연산 효율성을 최적화합니다.

- **Performance Highlights**: 새로운 SpikeRL 구현은 기존의 최신 DeepRL-SNN 방법들과 비교하여 4.26배 빠르고, 2.25배 더 에너지 효율적이라는 성과를 보여주었습니다. 이는 SNNs가 기존의 인공 신경망(ANNs)보다 향상된 성능을 발휘할 수 있도록 하여 복잡한 제어 시나리오에서의 적응성과 정밀도를 극대화하는 데 기여합니다. SpikeRL은 실제 응용 프로그램에서의 복잡한 연속 제어 작업에 진정한 확장 가능하고 지속 가능한 솔루션을 제공함을 입증하였습니다.



### External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation (https://arxiv.org/abs/2502.17494)
Comments:
          Accepted by the ACM Web Conference (WWW) 2025 Industrial Track as Oral Presentation

- **What's New**: 본 논문에서 제안된 External Large Foundation Model (ExFM) 프레임워크는 산업 규모의 광고 추천 시스템에서 간과된 두 가지 주요 도전 과제를 해결하기 위해 설계되었습니다. 첫 번째 과제는 훈련 및 추론 지연이 제한되어 있는 것으로, 기존의 방법들이 대형 모델의 훈련과 추론 비용을 증가시키는 문제를 가지고 있었습니다. 두 번째 과제는 데이터 분포가 동적으로 변화하는 대량의 스트리밍 데이터가 사용되며, 이로 인해 모델이 항상 최신 데이터에 적합해야 하는 문제입니다.

- **Technical Details**: ExFM은 외부 증류(External Distillation) 기법을 활용하여 훈련 데이터 및 유지 비용을 절감하면서, 여러 학생 모델(Vertical Models, VMs)에 걸쳐 공통된 Teacher 모델(Foundation Model, FM)의 예측을 제공합니다. Auxiliary Head (AH)와 Student Adapter (SA)를 도입하여 FM가 VMs에 전이되는 데이터 분포의 격차를 완화하고, Freshness Gap을 감소시켜 모델의 성능을 향상시키는 방법론을 제안합니다. 이를 통해 ExFM은 재훈련을 최소화하면서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: ExFM 프레임워크를 도입한 실험 결과, 내부 산업 규모 데이터셋과 공개 데이터셋 모두에서 성능 향상이 확인되었습니다. ExFM은 수조 개의 파라미터를 가진 모델을 메타 플랫폼에 활용할 수 있도록 하여, 다양한 도메인 및 작업의 VMs에서 Promising한 성능 개선을 보여주었습니다. 또한, 하이퍼파라미터의 영향을 분석한 결과도 제공하여 이 접근법의 효용성을 강조합니다.



### Pursuing Top Growth with Novel Loss Function (https://arxiv.org/abs/2502.17493)
Comments:
          30 pages, 7 figures, GitHub repo: this https URL

- **What's New**: 본 논문은 주식 시장에서 길게 지속 가능한 수익을 달성하기 위한 새로운 접근법을 제안합니다. 특히, return-weighted loss function을 통해 머신러닝 모델이 최적의 성장 기회를 발견할 수 있도록 제한된 정보만을 제공하는 방식입니다. 이 시스템은 상장 주식의 공개 데이터와 몇 가지 기술 지표를 기반으로 효율적인 일일 거래 시스템을 제시합니다.

- **Technical Details**: 연구에서 사용된 모델은 Convolutional Neural Networks (CNN)에 기반하며, 주가 변동, 거래량, 모멘텀, 변동성 및 트렌드와 같은 기술적 요소를 분석합니다. 우리는 성과 평과를 더 개선하기 위해 맞춤형 손실 함수와 주식 특징 정보를 결합하는 방식을 사용했습니다. 일일 평가 결과에 따른 투자 점수를 계산하여 거래 전략을 수립하며, 각 주식에 대한 2차원 행렬 형태로 피처를 구성합니다.

- **Performance Highlights**: 최고 모델은 2019년에서 2024년 사이에 연 61.73%의 수익률을 달성하였고, 샤프 비율(Sharpe Ratio)은 1.18에 달합니다. 또한, 2005년부터 2010년 사이의 데이터 분석에서는 연 37.61%의 수익률을 기록했습니다. 연구에서는 제안된 손실 함수가 전통적인 손실 함수들보다 어떻게 우수한지 여러 성과 지표 및 통계적 증거를 통해 입증하였습니다.



### A generalized dual potential for inelastic Constitutive Artificial Neural Networks: A JAX implementation at finite strains (https://arxiv.org/abs/2502.17490)
Comments:
          56 pages, 19 figures, 3 tables

- **What's New**: 본 논문에서는 비탄성 구성 인공지능 신경망인 iCANN을 위한 일반화된 이중 포텐셜(dual potential) 또는 가상 포텐셜(pseudo potential) 설계 방법론을 제시합니다. 새로운 포텐셜은 응력 불변량(stress invariants)으로 표현되어 대변형(large deformations)에 대한 열역학적 일관성(thermodynamic consistency)을 자연스럽게 만족합니다. 이전 연구와 비교하여, 새로운 포텐셜은 압력 민감 비탄성 압축(pressure-sensitive inelasticity)을 포함한 보다 넓은 스펙트럼의 재료 행동을 포착합니다.

- **Technical Details**: 논문에서는 유한 변형 비탄성(finite strain inelasticity)에 대한 iCANN의 열역학적 프레임워크를 재검토하고, 볼록(convex), 제로-값(zero-valued), 비부정적(non-negative) 이중 포텐셜을 구성하기 위한 조건을 도출합니다. 이를 신경망에 포함시키기 위해 아키텍처 설계를 상세히 설명하며, 미리 열역학에 대한 준수를 보장합니다. 이 프레임워크는 JAX에서 구현되어 있으며, 공개적으로 접근 가능합니다.

- **Performance Highlights**: 제안된 아키텍처의 성능을 평가하기 위해 비스코-탄성(visco-elastic) 재료 행동을 조사하였으며, 이 방법은 비스코-탄성에 한정되지 않습니다. 결과적으로, 새로운 아키텍처는 해석 가능한 모델과 매개변수를 강력히 발견하며, 비탄성도의 정도를 자율적으로 드러냅니다. 이를 통해 비탄성 재료 발견 전략에서 다양한 측면을 검토한 결과, 돋보이는 성능을 확인할 수 있었습니다.



### Using Graph Convolutional Networks to Address fMRI Small Data Problems (https://arxiv.org/abs/2502.17489)
Comments:
          8 pages

- **What's New**: 이 연구는 그래프 신경망(graph neural networks)을 활용하여 소규모 데이터를 다루는 의학 이미징(medical imaging)의 예측 문제를 해결하고자 한다. 주로 치료 반응 예측(prognosis) 문제에 집중하여, 기능성 자기공명영상(fMRI) 데이터를 기반으로 치료에 대한 증상 개선을 예측하는 새로운 방법론을 제시한다. 기존의 패턴 인식 기법으로는 어려운 예측을 가능하게 하기 위하여, 환자의 뇌 활동 연결성 정보를 스펙트럼 표현(spectral representation)을 통해 효과적으로 전파하는 방식을 소개한다.

- **Technical Details**: 연구의 주요 기술적 초점은 t-fMRI 데이터의 작은 크기와 연결성 그래프(connected graphs) 정보의 복잡성을 극복하기 위해 설계된 심층 그래프 컨볼루션 학습 구조에 있다. 각 환자를 위한 새로운 표현 방식으로 스펙트럼 분해(spectral decomposition)를 활용하며, 이는 이전의 스펙트럼 분석과는 구별되는 접근법이다. GNN(그래프 신경망) 방법을 사용하여 기존의 일반 NN(신경망) 방법보다 더 나은 예측 성능을 가져오며, 이로 인해 72.2% ± 0.7%의 정확도를 달성했다.

- **Performance Highlights**: 실험 결과, 제안된 GNN 방법은 기존의 방법보다 약 12% 향상된 성능을 발휘함을 보였다. 데이터의 스무딩(smoothing) 효과를 통해 삼각 부등식(triangle inequality)의 위반을 줄임으로써 성능이 개선된 것으로 나타났으며, 이는 연결 데이터의 스펙트럼 임베딩(spectral embedding)을 더 잘 수행할 수 있게 함을 의미한다. 이러한 결과는 환자별 예측의 효율성을 높이고, 응용 가능한 치료 접근법을 제시하는 데 기여할 것이다.



### Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework (https://arxiv.org/abs/2502.17481)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문은 SynthSleepNet이라는 다중 모달 하이브리드 자기 지도 학습(Self-Supervised Learning, SSL) 프레임워크를 소개합니다. 이 방법은 수면 분석을 위해 폴리솜노그래피(Polysomnography, PSG) 데이터를 효과적으로 통합합니다. 기존 방법과 차별화되는 점은 마스크된 예측(masked prediction)과 대조 학습(contrastive learning)을 융합하여 다양한 생리 신호 간의 보완 기능을 활용한다는 것입니다.

- **Technical Details**: SynthSleepNet은 뇌전도(Electroencephalogram, EEG), 안구전도(Electrooculography, EOG), 근전도(Electromyography, EMG), 심전도(Electrocardiogram, ECG) 등 여러 신호 모달리티에서 심층 표현 학습을 가능하게 합니다. 이 프레임워크는 Mamba 기반의 시간적 맥락 모듈을 통해 시간 축을 따라 신호 간의 맥락 정보를 효율적으로 포착하도록 설계되었습니다. 또한, LoRA(Low-Rank Adaptation)를 활용하여 정보를 정교하게 추출하며, 여러 개의 모달리티-specific backbone을 사용하여 신호 특성에 최적화된 인코더를 갖추고 있습니다.

- **Performance Highlights**: SynthSleepNet은 수면 단계 분류, 무호흡증 감지, 저호흡증 감지의 세 가지 후속 작업에서 각각 89.89%, 99.75%, 89.60%의 정확률을 기록하며 최신 기술 대비 우수한 성능을 보였습니다. 제한된 라벨이 있는 반지도 학습(semi-supervised learning) 환경에서도 87.98%, 99.37%, 77.52%의 정확성을 달성했습니다. 이 결과는 SynthSleepNet이 PSG 데이터의 종합 분석을 위한 기초 도구로서의 잠재력을 강조합니다.



### Brain-to-Text Decoding: A Non-invasive Approach via Typing (https://arxiv.org/abs/2502.17480)
Comments:
          15 pages, 5 figures

- **What's New**: 본 연구에서는 침습적인 뇌-컴퓨터 인터페이스(BCI)의 대안으로 비침습적인 Brain2Qwerty 모델을 도입합니다. 이 모델은 뇌 활동을 통해 문장 생성을 디코드할 수 있는 기능을 갖추고 있으며, 35명의 건강한 자원자와의 실험을 통해 그 효능을 입증했습니다. 특히, 이 연구는 기존 EEG 기반의 기술보다 우수한 성능을 보여주며 비침습적인 방법의 가능성을 열었습니다.

- **Technical Details**: Brain2Qwerty는 EEG(전기 뇌파)와 MEG(자기뇌파측정)를 사용하여 문장 생산을 디코드하는 새로운 딥 러닝 아키텍처입니다. 연구자들은 해당 모델을 훈련시키기 위해 35명의 참가자들에게 키보드로 간단한 문장을 입력하도록 하였고, 이를 통해 수집된 신호로부터 문자를 디코드했습니다. 실험 결과, MEG를 이용한 경우 평균 32%의 캐릭터 오류율(CER)을 기록하였고, 이는 EEG의 67%와 상당한 차이를 보였습니다.

- **Performance Highlights**: Brain2Qwerty는 MEG를 사용할 때 최고의 참가자에서 19%라는 낮은 CER을 달성했으며, 다양한 문장을 학습 세트 외에서도 완벽하게 디코딩할 수 있었습니다. 또한, EEGNet과의 비교에서 이 모델이 CER에서 1.14배, MEG에서 2.25배의 성과 향상을 이뤘음을 보였습니다. 이러한 결과는 비침습적인 브레인-컴퓨터 인터페이스의 안전하고 효과적인 개발 가능성을 제시합니다.



### ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis (https://arxiv.org/abs/2502.17475)
- **What's New**: ECG-Expert-QA는 ECG 해석에서 진단 능력을 평가하기 위한 포괄적인 멀티모달 데이터셋으로, 실제 임상 데이터와 체계적으로 생성된 합성 사례를 통합하였습니다. 이 데이터셋은 47,211개의 질문-답변 쌍으로 구성되어 있으며, 복잡한 사례 해석을 포함한 다양한 임상 시나리오를 다룹니다. 이를 통해 고유한 진단 작업을 통해 의료 언어 모델을 종합적으로 평가할 수 있는 기반이 마련되었습니다.

- **Technical Details**: 이 연구는 전통적인 평가 방법의 효율성 문제와 데이터셋의 복잡성 부족을 해결하기 위해 ECG-Expert-QA 데이터셋을 개발했습니다. 데이터셋은 기본 의료 지식 검증 모듈, 임상적 추론 평가 모듈, 위험 관리 모듈의 세 가지 핵심 평가 모듈로 구성되며, 이는 환자 예후 예측, 다중 모달 정보 통합 등을 포함합니다. 또한, 데이터셋은 윤리적 차원도 포함하여, 의료 AI의 결정 안전성을 평가하기 위한 기준을 제시합니다.

- **Performance Highlights**: 이 데이터셋은 기존의 의료 데이터셋에 비해 복잡한 진단 작업 비율이 15.3%에 달하며, 다양한 언어로 지원되어 cross-cultural 연구에 기여합니다. 특히, innovative evaluation dimensions인 반사적 추론과 기억 교정 메커니즘을 도입하여 모델의 의사 결정 논리의 임상적 합리성과 강건성을 평가합니다. 이러한 특성으로 인해 ECG-Expert-QA는 AI 보조 ECG 해석 발전의 중요한 벤치마크로 자리잡고 있습니다.



### MC2SleepNet: Multi-modal Cross-masking with Contrastive Learning for Sleep Stage Classification (https://arxiv.org/abs/2502.17470)
- **What's New**: 본 연구에서는 수면 단계 분류를 위한 새로운 깊이 학습 모델인 MC2SleepNet (Multi-modal Cross-masking with Contrastive learning for Sleep stage classification Network)을 소개합니다. 이 모델은 CNN(Convolutional Neural Networks)과 Transformer 아키텍처의 효과적인 협업을 통해 다중 모달 학습을 지원하여 수면 단계 분류의 정확성을 높이는 것을 목표로 합니다. MC2SleepNet은 SleepEDF-78 데이터셋에서 84.6%, Sleep Heart Health Study(SHHS) 데이터셋에서 88.6%의 우수한 정확도를 기록했습니다.

- **Technical Details**: MC2SleepNet은 두 가지 데이터 샘플의 특성을 결합한 다중 모달 모델로, 각기 다른 두 개의 백본을 사용합니다: (1) "Raw Signal View"와 (2) "Spectrogram View". 훈련 과정은 두 단계로 나뉘는데, 첫 번째는 Epoch-Level과 Sequence-Level을 동시에 사전 훈련(pre-training)하고, 두 번째는 모델을 미세 조정(fine-tuning)하는 과정입니다. 이 과정에서 InfoNCE 손실 및 Cross-Masking 기법이 도입되어 다양한 모달리티에서 추출된 특징을 상호 검토하여 성능을 향상시킵니다.

- **Performance Highlights**: MC2SleepNet은 최신 딥러닝 모델들과 비교했을 때 SleepEDF-78 데이터셋에서 84.6%의 정확도, SHHS 데이터셋에서 88.6%의 정확도를 달성하여 최첨단 성능을 기록했습니다. 이러한 결과는 소규모 및 대규모 데이터셋에 걸쳐 제안된 네트워크의 효과적인 일반성을 보여줍니다. 수면 단계 분류에서의 정확성 향상은 수면 부족이나 장애를 진단하는 데 있어 매우 중요한 기여를 할 것으로 기대됩니다.



### PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Lev (https://arxiv.org/abs/2502.17469)
- **What's New**: 이번 연구는 개인의 일상 생활과 건강에 대한 귀중한 통찰력을 제공하는 lifelog 데이터 분석 방법론을 제시합니다. 특히, PixleepFlow라는 이미지 기반의 수면 질 및 스트레스 수준 추정 모델을 개발하여 다양한 센서 데이터를 통합하여 다각적인 지표를 동시에 예측할 수 있는 가능성을 제시합니다. 이는 전통적인 시계열 데이터 분석과는 달리, 이미지 기반 입력 방식으로 데이터를 시각적으로 직관적으로 표현합니다.

- **Technical Details**: PixleepFlow는 다양한 센서 데이터를 조합하여 세 가지 RGB 이미지로 변환하는 과정에서 고차원 데이터를 단순화합니다. 이 접근 방식은 세분화된 세부사항보다는 전체적인 패턴과 이상 징후를 식별하는 데 유리합니다. 또한, Explainable Artificial Intelligence(XAI) 기술을 통해 예측 결과를 시각적으로 해석할 수 있도록 하여, 모델의 투명성과 설명 가능성을 높였습니다.

- **Performance Highlights**: 연구 결과, PixleepFlow는 각기 다른 데이터 형식보다 더 중요한 성과를 보여주었으며, 일상적인 데이터 수집을 통해 인간의 삶의 질을 이해하는 데 기여하고 있습니다. 이 모델은 수면 질 및 스트레스 수준을 동시에 예측할 수 있는 다중 레이블 분류 방법을 적용하여, 7개의 핵심 지표를 추정하고 높은 정확도를 달성하는 데 성공하였습니다.



### The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG (https://arxiv.org/abs/2502.17462)
Comments:
          Published at ICLR 2025, see this https URL. Code is available at this https URL

- **What's New**: 본 논문은 EEG(Scalp Electroencephalogram)와 iEEG(Intracranial Electroencephalogram)의 데이터 압축 성능을 향상시키는 새로운 신경 네트워크 모델인 BrainCodec을 소개합니다. BrainCodec은 두 가지 데이터 모달리티에 대해 높은 재구성 품질을 달성하며, 특히 iEEG를 이용해 훈련한 후 EEG에 전이할 경우 더 뛰어난 성능을 보입니다. 또한, EEG와 iEEG를 모두 사용하여 훈련 시 신뢰성을 높이고 재구성 성능을 개선하는 것을 보여줍니다.

- **Technical Details**: BrainCodec은 EEG와 iEEG 신호를 신경망 압축하여 최대 64배의 압축 비율을 달성합니다. 이 모델은 고품질의 신호에서 훈련되었을 때 지정된 성능 기준을 충족하는 높은 재구성 충실도를 보입니다. 구체적으로, 제안한 기준은 PRD(percentage root-mean-square difference) 30 이하 및 시퀀스 분류 성능이 1% 미만의 저하를 포함합니다.

- **Performance Highlights**: BrainCodec은 EEG와 iEEG 신호의 손실 압축에서 기존의 최첨단 압축 모델들을 초월한 성능을 보여줍니다. 시각적 평가를 통해 신경 학자가 BrainCodec의 높은 재구성 품질을 확인하였으며, 다운스트림 작업인 발작 탐지 및 운동 이미징 과제에 대해서도 성능 저하가 없음을 보였습니다. 결과적으로, BrainCodec은 의료 타임 시리즈 도메인에서 더 높은 SNR을 가진 데이터 소스가 더 높은 성능을 발휘할 수 있음을 뒷받침합니다.



### Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation (https://arxiv.org/abs/2502.17460)
Comments:
          7 pages, 1 figure, 5 tables, preprint

- **What's New**: 이번 연구는 Electroencephalogram (EEG) 데이터를 이용해 사전 훈련된 모델이 Electrocardiogram (ECG)와 Photoplethysmogram (PPG) 데이터에서 혈압(BP) 추정에 효과적으로 전이 학습될 수 있음을 실험적으로 입증한 최초의 연구입니다. 추가적인 대규모 사전 학습 없이 최소한의 세밀한 조정만으로도 가능하다는 점이 특징입니다. 이를 통해 혈압 모니터링 기술에서 다양한 생체 신호 간의 융합 가능성을 제시합니다.

- **Technical Details**: 이 연구는 기존 EEG 기반의 foundation model인 CEReBrO를 활용합니다. 이 모델은 EEG 신호를 기반으로 하여 미리 훈련된 후, ECG와 PPG 신호로 혈압을 예측하도록 세밀하게 조정됩니다. 그 과정에서 동적 INT8 양자화를 적용하여 모델 크기를 3.5배 이상 감소시켰고, 이는 자원 제약이 있는 wearable 장치에서의 실시간 혈압 모니터링을 가능하게 합니다.

- **Performance Highlights**: MIMIC-III 및 VitalDB 데이터셋에서 수행한 평가 결과, 이 접근 방식은 이완기 혈압의 평균 절대 오차 1.57 mmHg로 거의 최신 기술과 동등한 정확성을 달성했으며, 수축기 혈압의 경우 이전 작업보다 1.5배 높은 정확성을 보였습니다. 이러한 성과는 예측 정확성과 계산 비용의 절충안에서도 중요한 의미를 가지며, 혈압 모니터링 장치의 실용화에 기여할 수 있습니다.



### MoEMba: A Mamba-based Mixture of Experts for High-Density EMG-based Hand Gesture Recognition (https://arxiv.org/abs/2502.17457)
- **What's New**: 본 논문은 MoEMba 프레임워크를 통해 HDsEMG를 기반으로 한 제스처 인식을 개선하는 새로운 방법론을 제시합니다. MoEMba는 Selective State-Space Models (SSMs)을 활용하여 세션 간 및 주체 간 변동성 문제를 해결합니다. 이를 통해 시간 의존성과 채널 간 상호작용을 효과적으로 포착하며, 신호 표현을 향상시키기 위해 웨이브렛 변환 기능 조정(WTFM)을 통합했습니다.

- **Technical Details**: 제가 제안한 MoEMba 프레임워크는 다중 Mamba 전문가의 적응형 조합을 활용하여 짧은-장기 제스처 동역학을 포착합니다. 또한, 이 구조는 상대적으로 적은 계산량(FLOPS)으로 연산 효율성을 달성하며 고밀도 HD-sEMG 데이터의 세션 간 변동성에 강한 내성을 보입니다. MoEMba는 채널 주의(chanel attention)를 통해 채널 간 상호작용을 다룹니다.

- **Performance Highlights**: 실험 결과에 따르면, MoEMba는 CapgMyo HD-sEMG 데이터셋에서 56.9%의 균형 잡힌 정확도를 기록하며 기존 최첨단 모델들을 초월했습니다. 이는 MoEMba 프레임워크가 HD-sEMG 기반 HCI 시스템에서의 가능성을 강조하는 것이며, 프로세틱 제어 및 인간-컴퓨터 상호작용과 같은 실제 응용에 적합하다는 것을 보여줍니다.



### Survey on Recent Progress of AI for Chemistry: Methods, Applications, and Opportunities (https://arxiv.org/abs/2502.17456)
Comments:
          22 pages, 8 figures, 4 tables

- **What's New**: 인공지능(AI) 기술의 발전은 다양한 분야에서 혁신을 가져왔습니다. 특히 AI를 활용한 화학 연구의 가속화가 점차 성과를 내고 있으며, 많은 혁신적인 작업들이 이루어지고 있습니다. 이 논문에서는 현재 화학 분야에서 사용되는 AI 기법을 포괄적으로 검토하고, 데이터의 특성과 다양한 표현 방법을 소개하며, 주요 임무에 대한 모델을 개관합니다.

- **Technical Details**: AI는 일반적으로 데이터, 표현 및 모델의 세 가지 중요한 구성 요소로 구성됩니다. 머신러닝(ML)에서는 고품질의 다양한 데이터가 필수적이며, 데이터는 적절한 기계 인식 형식으로 변환되어야 합니다. 이 논문에서는 분자 수준 데이터와 반응 수준 데이터와 같은 다양한 화학 데이터셋을 소개하고, 이들을 활용하기 위한 분자 식별자와 설명자를 설명합니다.

- **Performance Highlights**: 최근의 연구들에서, 고속 처리 능력과 자동 합성 기술의 발전은 화학에서 머신러닝 방법의 개발에 대한 증가하는 관심을 불러일으켰습니다. 특히 대형 언어 모델(LLMs)의 활용은 화학 연구를 위한 에이전트 시스템 개발로 이어지고 있으며, 이들은 다운스트림 작업의 개선에도 기여하고 있습니다. 이 논문은 AI 기술을 화학에 적용하기 위한 여러 가지 중요한 문제들을 강조하며 향후 발전을 지원하는 기초 자료로 기능하고 있습니다.



### Smart Sampling Strategies for Wireless Industrial Data Acquisition (https://arxiv.org/abs/2502.17454)
Comments:
          17 pages, 11 figures

- **What's New**: 이 연구는 산업 환경에서 데이터 수집의 정확성을 최적화하는 방법을 탐구합니다. 특히 무선 텔레메트리 시스템에서의 높은 샘플링 주파수로 인해 발생하는 저장, 전송, 계산 자원 소비, 그리고 배터리 수명과 같은 문제를 해결하고자 합니다. 이 연구에서는 샘플링 주파수(Fs)를 줄이면서도 신호 측정의 품질을 유지하는 전략을 제안합니다.

- **Technical Details**: 연구의 주요 메트릭으로 상대 오류를 정의하고, 샘플링 주파수가 신호의 재구성 품질에 미치는 영향을 분석합니다. 이를 위해 데이터 전송에 소요되는 에너지를 고려한 비용 함수가 제안되며, Nyquist 기준에 따라 요구되는 최소 샘플링 주파수를 설정합니다. 또한, 샘플링 주파수에 따른 보여지는 신호의 변형이 적정 범위 이내에 포함될 수 있도록하였습니다.

- **Performance Highlights**: 샘플링 주파수를 80% 줄여도 측정 품질이 저하되지 않는 결과를 도출했습니다. 이를 통해 데이터 저장 및 전송 비용을 최적화할 수 있는 잠재력을 확인했습니다. 연구 결과는 산업 모니터링에 사용되는 예측 모델의 효율성을 유지하면서도 최소한의 자원을 사용하는 방법을 제시합니다.



### AirTag, You're It: Reverse Logistics and Last Mile Dynamics (https://arxiv.org/abs/2502.17447)
- **What's New**: 이 연구는 재난 구호 상황에서 last-mile delivery(최종 배송)의 필수 요소인 역물류(reverse logistics)의 도전 과제를 다룹니다. hub-and-spoke 네트워크가 장거리 확장성에서는 뛰어나지만, 밀접하게 위치한 spoke가 먼 hub에 의존할 경우 비효율성이 발생합니다. 20개의 Apple AirTag를 사용하여 이 연구는 물류 흐름에 대한 실증적인 통찰력을 제공합니다.

- **Technical Details**: 이 연구에서는 Bluetooth LE (BLE) 5 트래커와 Apple Find My 네트워크를 통합하여 세밀한 공간 및 시간 데이터를 수집합니다. 트래커는 동적인 화물 이동을 모니터링하는 데 있어 가치를 발휘해, 재난 구호와 같은 상황에서 모바일 허브 배치 및 경로 최적화의 실시간 조정을 가능하게 합니다. 또한, discrete event simulation (DES)을 활용하여 hub-spoke 구성에서 발생하는 saddle point를 탐구합니다.

- **Performance Highlights**: BLE 기술의 활용을 통해 역물류를 정제하고 지연 시간을 줄이며 운영 유연성을 향상시킬 수 있는 가능성을 발견하였습니다. 특히, 허리케인 헬렌과 같은 재난 구호 상황에서 이러한 기술적 접근이 크게 기여할 수 있음을 보여주고 있습니다. 이 연구는 정기적인 배송뿐만 아니라 위기 상황에서의 물류 관리에 중요한 개선을 제안합니다.



### DCentNet: Decentralized Multistage Biomedical Signal Classification using Early Exits (https://arxiv.org/abs/2502.17446)
- **What's New**: DCentNet는 IoT 웨어러블 센서에서 생체의료 데이터의 분산 멀티스테이지 신호 분류를 위한 혁신적인 접근 방식입니다. 기존의 중앙 집중식 처리 방식과 달리, DCentNet은 조기 종료 지점(EEP)을 도입하여 에너지 효율성과 처리 속도를 향상시킵니다. 이 시스템은 대형 CNN 모델을 여러 하위 네트워크로 나누어 데이터를 전송하기 전에 대형 특징 맵을 압축합니다.

- **Technical Details**: DCentNet은 EEP를 통해 CNN 모델을 여러 하위 네트워크로 분할하고, 각 하위 네트워크를 다른 노드에 배포하여 높은 신뢰성의 분류를 제공합니다. 유전 알고리즘을 사용하여 EEP 배치를 최적화하며, 초기 하위 네트워크에서 높은 정확성이 유지됩니다. 실험 결과, 한 개의 EEP을 사용할 경우 무선 데이터 전송이 94.54% 감소하고 복잡성이 21% 감소하며 원래의 정확도와 민감도를 유지합니다.

- **Performance Highlights**: DCentNet는 ARM Cortex-M4 MCU에서 구현되어 평균 73.6%의 전력 절약을 달성했습니다. 두 개의 EEP을 적용할 경우 민감도가 98.36%, 정확도가 97.74%에 도달하였으며, 추가적인 무선 데이터 전송 감소율은 91.86%에 달합니다. 분산 시스템의 장점으로 인해 시스템의 비용 효율성과 유연성이 크게 향상됩니다.



### Interpretable Dual-Filter Fuzzy Neural Networks for Affective Brain-Computer Interfaces (https://arxiv.org/abs/2502.17445)
- **What's New**: 이번 연구에서는 감정 상태를 감지하고 해석하는 새로운 컴퓨테이셔널 모델인 iFuzzyAffectDuo를 소개합니다. 이 모델은 이중 필터 퍼지 신경망 아키텍처를 통합하여 신경영상 데이터에서 감정 상태를 더 효과적으로 탐지하고 해석할 수 있도록 합니다. 기존의 접근 방식에 비해 더 높은 정확도와 해석 가능성을 달성하는 새로운 멤버십 함수(Membership Function, MF)를 도입했습니다.

- **Technical Details**: iFuzzyAffectDuo는 기능적 근적외선 분광법(Functional Near-Infrared Spectroscopy, fNIRS) 및 뇌전도(Electroencephalography, EEG) 데이터를 사용하는 세 가지 신경영상 데이터 세트에서 성능을 검증합니다. 이 모델은 EEGNet에서 영감을 받아 공간적 및 시간적 필터를 결합하여 특징 추출과 패턴 인식의 향상을 달성했습니다. 또한, 퍼지 주의 메커니즘(Fuzzy Attention Mechanism)을 통합하여 뇌 신호의 도메인 특이적 시공간 의존성을 포착하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, iFuzzyAffectDuo는 감정 인식 작업에서 최첨단 정확도와 해석 가능성을 달성하였으며, 정서적 신경 패턴을 효과적으로 포착하여 fNIRS 및 EEG 방식의 aBCI 시스템 성능을 향상시켰습니다. 이러한 결과는 감정의 신경적 기초를 이해하고 인간-컴퓨터 상호작용을 향상시키기 위한 새로운 경로를 열어줍니다.



### AI Agentic workflows and Enterprise APIs: Adapting API architectures for the age of AI agents (https://arxiv.org/abs/2502.17443)
- **What's New**: 이번 논문에서는 Generative AI의 발전이 자율 AI 에이전트의 출현을 촉진하고, 이로 인해 기업 컴퓨팅 인프라에서 새로운 도전 과제가 발생하고 있음을 강조합니다. 기존의 기업 API 아키텍처는 인간 주도의 정의된 상호작용 패턴을 위해 설계되어, AI 에이전트의 동적이고 목표 지향적인 행동을 지원하기에는 적합하지 않다고 지적합니다. 이를 해결하기 위해 기업 API의 아키텍처 개조를 체계적으로 검토하고 있습니다.

- **Technical Details**: 연구는 API 디자인 패러다임, 에이전트 상호작용 모델, 그리고 새로운 기술적 제약들을 포괄적으로 분석합니다. 또한, 표준화, 성능, 그리고 지능형 상호작용에 대한 주요 과제들을 해결하기 위해 이론적 모델링, 비교 분석, 탐색적 디자인 원칙을 결합한 혼합 방법론(mixed-method approach)을 사용합니다. 그 과정에서 API 변환을 위한 전략적 프레임워크를 개발하고 있습니다.

- **Performance Highlights**: 이 연구는 다음 세대 기업 API를 위한 개념 모델을 제안하여, 자율 AI 에이전트 생태계와 원활히 통합될 수 있도록 합니다. 이 모델은 미래의 기업 컴퓨팅 아키텍처에 중대한 영향을 미칠 것으로 기대됩니다. 기업 환경에서 지능형 에이전트의 도입을 효과적으로 지원하는 API 디자인이 될 것입니다.



### Thinking Before Running! Efficient Code Generation with Thorough Exploration and Optimal Refinemen (https://arxiv.org/abs/2502.17442)
Comments:
          14 pages, 10 figures

- **What's New**: 최근 연구 진행 상황에 따르면, ThinkCoder는 코드 생성을 효율적으로 수행하기 위한 새로운 프레임워크로 소개되었습니다. 이 프레임워크는 심층 탐색(thorough exploration)과 최적의 정제(optimal refinement)를 결합하여 최상의 코드 솔루션을 제공하는데 중점을 두고 있습니다. ThinkCoder는 두 개의 주요 에이전트로 구성되어 있으며, 탐색 에이전트는 다양한 코드와 테스트를 통해 탐색을 확장하고, 실행 에이전트는 결과를 독립적으로 검증합니다.

- **Technical Details**: ThinkCoder에서는 탐색과 최적화 과정에서 Reinforced Self-Training (ReST)을 이용한 선호 기반 최적화(preference-driven optimization)를 구현합니다. 이 방식은 성공적인 탐색 경로를 이용해 LLM의 학습을 지원하고, 탐색 중 최적의 솔루션을 더 효율적으로 생성할 수 있도록 돕습니다. 이러한 접근 방식은 계산 비용을 크게 절감하면서도 높은 정확성을 유지하게 합니다.

- **Performance Highlights**: ThinkCoder는 HumanEval 및 MBPP와 같은 벤치마크에서 뛰어난 성능을 보이며, 기존의 SOTA 모델 대비 21.7%의 계산 비용으로 Pass@1을 1.5% 개선한 결과를 나타내었습니다. AgentCoder와 비교했을 때, ThinkCoder는 두 번의 탐색 후 0.6% 높은 Pass@1을 달성했으며, LLaMA2-7B와 같은 모델도 20%의 계산 자원만으로 경쟁력을 갖춘 결과를 이끌어냈습니다.



### Large Language Models as Realistic Microservice Trace Generators (https://arxiv.org/abs/2502.17439)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 활용하여 마이크로서비스 호출 그래프와 같은 합성 워크로드 추적(workload traces)을 생성하는 방법을 제안합니다. 기존의 방법들이 특정 필드만 생성하거나 고정 구조의 추적을 다루던 반면, 이 연구는 계층 구조를 재귀적으로 생성하여 복잡한 제약조건을 만족하는 그래프를 생성하는 혁신적인 접근 방식을 사용합니다. 저자들은 이러한 방법을 통해 합성 추적의 유효성과 현실성을 크게 향상시켰다고 주장합니다.

- **Technical Details**: 저자들은 마이크로서비스 호출 그래프의 구조적 제약을 충족시키기 위해 LLM을 세부적으로 조정하여 각 계층을 재귀적으로 생성하는 방식을 채택했습니다. 이 과정에서 중간 단계를 통해 사용자가 요청한 속성을 유지하며 복잡한 계층 구조를 보다 쉽게 다룰 수 있도록 합니다. 논문에서는 Llama-2 7B 모델을 활용하여 마이크로서비스 추적 데이터를 학습시키고, 사용할 수 있는 코드도 공개하고 있습니다.

- **Performance Highlights**: 모델의 성능 평가 결과, 제안된 방식으로 생성된 합성 추적은 전체적인 유효성과 정확성 측면에서 기존 방법을 능가했습니다. 특히, 생성된 추적은 실제 데이터의 분포와 밀접하게 일치하며, 다운스트림 작업에서도 우수한 성능을 보였습니다. 저자들은 이 합성 추적이 마이크로서비스 관리 작업에서 실제 데이터 대신 효과적으로 사용될 수 있음을 입증합니다.



New uploads on arXiv(cs.LG)

### Allocating Variance to Maximize Expectation (https://arxiv.org/abs/2502.18463)
- **What's New**: 본 논문은 가우시안 랜덤 변수의 기대값 극대화를 위한 효율적인 근사 알고리즘을 설계하였습니다. 특히, 변수 집합의 크기 증가 시 최적 분산 배분을 특성화하며, $m=1$인 경우의 다항 시간 근사 알고리즘과 일반적인 경우 (m>1)에 대한 $O(	ext{log } n)$ 근사 알고리즘을 제공합니다. 이러한 극대화 문제는 경매 시장의 효용 극대화부터 정량 유전학의 혼합 모델 학습까지 다양한 응용에서 발생합니다.

- **Technical Details**: 문제는 n개의 독립 가우시안 랜덤 변수 (X1, ..., Xn)에 대한 최적 분산 배분을 찾는 것에 중점을 둡니다. 여기서 ${	ext{OPT}}$는 특정 제약 조건 하에서 최대 기대값을 계산하는 것을 포함합니다. 이 논문은 세 가지 기본 분산 배분 문제, 즉 독립 변수, 비독립 변수 및 그래프 변수를 고려하여 각각에 대한 해결책을 제시합니다.

- **Performance Highlights**: 개발된 알고리즘은 이론적으로 최적의 성능을 보장하며, 특히 변수의 수가 많아질수록 최적 분산 할당이 소수의 변수에 집중된다는 특징을 나타냅니다. 또한, 우리는 연속적이고 상관 관계가 없는 경우부터 상관 관계가 있는 경우까지 다양한 설정에서 성능을 비교하며, 이전의 광범위한 연구 결과와도 비교하여 이를 입증합니다.



### Scalable Equilibrium Sampling with Sequential Boltzmann Generators (https://arxiv.org/abs/2502.18462)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Boltzmann generator 프레임워크를 확장한 Sequential Boltzmann Generators (SBG)를 소개합니다. SBG는 Cartesian 좌표에서의 sampling을 향상시키기 위해 Transformer 기반의 비등가(normalizing flows) 구조를 활용하여 샘플 생성 및 likelihood 계산의 효율성을 극대화합니다. 추가적으로, annealed Langevin dynamics를 통해 표적 분포에 맞췄을 때 샘플의 variance를 줄여 더 정밀하게 모델링할 수 있도록 합니다.

- **Technical Details**: SBG 접근법은 기존의 Boltzmann generators가 아쉬운 성능을 보였던 높은 에너지 장벽을 극복합니다. 새로운 구조는 exact invertibility를 제공하고, 이는 샘플링 효율성을 증가시키며, 특히 복잡한 에너지 분포에서 작동할 수 있도록 개선되었습니다. 제안된 SBG는 또한 Sequential Monte Carlo (SMC) 기술을 통해 추론 시점을 조정하여, 중요한 물리적 양, 예를 들어 자유 에너지 차이를 계산하는 데 도움을 줍니다.

- **Performance Highlights**: SBG는 모든 측면에서 최첨단 성능을 달성하였으며, 기존의 Boltzmann generator에 비해 현저한 계산 효율성을 제공합니다. 특히, SBG는 tri, tetra, hexapeptides와 같은 복잡한 분자의 상태를 성공적으로 샘플링하며, 이로써 기존 방법으로는 불가능했던 비가역적인 메타스테이블 상태 샘플링의 새로운 가능성을 제시합니다.



### Supervised Reward Inferenc (https://arxiv.org/abs/2502.18447)
Comments:
          16 pages, 4 figures

- **What's New**: 이번 연구에서는 일반적인 행동에서 보상을 추론하는 새로운 접근법인 Supervised Reward Inference (SRI)를 제안합니다. 기존의 연구는 보상이 특정 행동 모델에 의해 생성된다고 가정하며, 이는 인간 행동의 복잡성을 충분히 설명하지 못합니다. 새로운 방법은 다양한 비최적 행동을 통해도 보상을 효율적으로 학습할 수 있음을 보여줍니다.

- **Technical Details**: SRI는 감독 학습(supervised learning)을 이용하여 행동과 보상 간의 직접적인 매핑을 학습합니다. 이는 행동 모델을 명시적으로 학습하는 기존 방법과는 달리, 데이터를 기반으로 보상 함수를 유도합니다. 연구는 다양한 비최적 데모를 사용하여 보상을 추론하고, 해당 접근법이 상대적으로 데이터 효율적이고 Bayes-optimal한 성질을 지닌다는 것을 입증합니다.

- **Performance Highlights**: 실험에서는 로봇 조작(task)을 통한 SRI의 성능을 평가하였으며, 다양한 비최적 데모를 바탕으로 한 보상 추론이 효과적임을 보여주었습니다. SRI는 기존의 비최적 행동을 활용하여 빠르고 효율적으로 보상을 추정할 수 있으며, 이는 여러 실제 적용 분야에서 유용할 것으로 기대됩니다.



### Comparative Analysis of MDL-VAE vs. Standard VAE on 202 Years of Gynecological Data (https://arxiv.org/abs/2502.18412)
Comments:
          12 pagas, 5 figures, 9th International Conference on Signal, Image Processing (SIPO 2025), Vancouver CA

- **What's New**: 이번 연구에서는 Minimum Description Length (MDL) 정규화가 강화된 Variational Autoencoder (VAE)를 표준 Autoencoder와 비교 평가하였습니다. MDL-VAE는 고차원 부인과 데이터 재구성에서 기존 모델보다 현저히 낮은 재구성 오류(MSE, MAE, RMSE)를 나타냈습니다. 또한, 효과적인 KL divergence 정규화를 통해 더 구조화된 잠재 표현(latent representations)을 드러냈습니다.

- **Technical Details**: 본 연구에서는 고차원 부인과 데이터의 재구성을 위해 MDL 원칙을 VAE 아키텍처에 통합했습니다. 이로 인해 MDL-VAE는 안정적인 훈련 및 검증 손실을 보여주고, 효율적인 추론(inference) 시간을 달성하여 강인성과 실용성을 잘 보여줍니다. 통계적 분석을 통해 이러한 성능 향상이 유의미함을 확인했습니다.

- **Performance Highlights**: MDL-VAE는 표준 Autoencoder에 비해 뚜렷한 성능 개선을 보였습니다. 이는 데이터 재구성 및 일반화에서 상당한 개선 효과가 있음을 시사하며, 의료 데이터 모델링 및 분석에 대한 고급 애플리케이션에 유망한 접근 방식이 될 것입니다.



### TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting (https://arxiv.org/abs/2502.18410)
Comments:
          8 pages, 4 figures, 7 tables and accepted at the AI4TS: AI for Time Series Analysis workshop, AAAI 2025

- **What's New**: 이번 연구에서는 새로운 신경망 구조인 TSKANMixer를 제안합니다. TSKANMixer는 Kolmogorov-Arnold Networks (KAN)를 TSMixer에 통합하여 시간 시계열 예측을 개선하는 데 중점을 두고 있습니다. 실험 결과, TSKANMixer는 기존 TSMixer보다 다양한 데이터셋에서 예측 정확도를 향상시키는 경향을 보였습니다.

- **Technical Details**: TSKANMixer는 KAN 레이어를 TSMixer 아키텍처에 추가하는 방식으로 설계되었습니다. KAN는 고정된 활성화 함수를 사용하는 전통적인 MLP와 달리, 학습 가능한 활성화 함수를 엣지에 적용하며, 노드에서 단순한 덧셈을 수행합니다. 이로 인해 KAN은 전통적인 MLP보다 성능과 해석 가능성을 개선할 수 있는 잠재력을 제공합니다.

- **Performance Highlights**: TSKANMixer는 여러 벤치마크에서 기존의 MLP 기반 모델보다 향상된 성능을 보이며, 특히 복잡하고 비선형적인 의존관계가 존재하는 다변량 시간 시계열 데이터에 대해 탁월한 예측 정확도를 보여주었습니다. 연구 결과, KAN이 시간 시계열 예측 분야에서 기존 방법에 비해 경쟁력 있는 대안이 될 수 있음을 입증했습니다.



### Enhancing DNA Foundation Models to Address Masking Inefficiencies (https://arxiv.org/abs/2502.18405)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 DNA 시퀀스 모델링을 위해 Masked Autoencoder Framework를 기반으로 한 수정된 인코더-디코더 아키텍처인 BarcodeMAE를 제안합니다. 기존의 Masked Language Modeling (MLM) 접근 방식으로 인한 배포 및 사전 훈련 간의 분포 차이를 극복하기 위해, 모델은 인코딩 중에 [MASK] 토큰을 제거하여 효율성을 향상시킵니다. 이로 인해 사전 훈련 중에 사용한 모든 가변성을 학습할 수 있습니다.

- **Technical Details**: BarcodeMAE는 BIOSCAN-5M 데이터셋을 활용하여 자가 지도 방식으로 사전 훈련을 수행합니다. 이 데이터셋은 2백만 개 이상의 고유한 DNA 바코드로 구성되어 있으며, 3개의 파티션으로 나누어져 있습니다. 인코더는 실제 토큰만을 처리하고, 디코더에서만 [MASK] 토큰의 예측을 수행하게 되어, 사전 훈련과 추론 간의 분포 차이를 해결합니다.

- **Performance Highlights**: 우리는 BarcodeMAE가 기존의 모델에 비해 분류 작업에서 10% 이상의 성능 향상을 보여주었음을 입증하였습니다. 특히, 생물 다양성 분석을 위해 설계된 이 모델은 모델들이 미세 조정 없이 특징 추출에 주로 사용되는 유전체 파이프라인에서 특히 효과적입니다. 전반적으로 BarcodeMAE는 평가 작업에서 평균적으로 우수한 성능을 보였습니다.



### The FFT Strikes Back: An Efficient Alternative to Self-Attention (https://arxiv.org/abs/2502.18394)
- **What's New**: 본 논문은 기존의 self-attention 메커니즘의 복잡도 문제를 해결하기 위해 FFT(빠른 푸리에 변환)를 활용한 FFTNet이라는 새로운 방법론을 도입합니다. 이 방법은 입력된 시퀀스를 주파수 영역으로 변환하고, 이를 통해 전 세계적인 토큰 혼합을 효율적으로 수행할 수 있습니다. 특히, 우리의 방법은 에너지 보존의 특성을 활용하여 훨씬 더 낮은 계산 복잡도인 \mathcal{O}(n\log n) 으로 긴 시퀀스의 의존성을 포착합니다.

- **Technical Details**: FFTNet은 입력 시퀀스를 주파수 영역으로 변환한 후, 학습 가능한 스펙트럼 필터를 적용하여 중요한 주파수 성분을 동적으로 강조합니다. 이 필터는 전역적인 컨텍스트 벡터에 근거하여 푸리에 계수를 재조정하며, 복잡한 패턴을 포착하는 데 필요한 고차 상관관계를 표현할 수 있도록 합니다. 이를 통해 우리는 FFT 기반의 변환에서 계산 효율성뿐만 아니라 적응적이고 맥락에 민감한 필터링과 비선형 처리를 결합하여 전통적인 self-attention에 대한 대안적인 접근법을 제시합니다.

- **Performance Highlights**: Long Range Arena 및 ImageNet 벤치마크에서의 실험 결과는 우리의 이론적 통찰이 실제로 우수한 성능을 발휘함을 입증합니다. 특정 환경에서도 기존의 고정 푸리에 및 표준 attention 모델에 비해 더 나은 성능을 보여주며, 이로 인해 장기 의존성을 더욱 효과적으로 모델링할 수 있는 가능성을 제시합니다. 따라서 FFTNet은 고차원 시퀀스 모델링 작업을 위한 매력적인 선택으로 대두될 것입니다.



### Mechanistic PDE Networks for Discovery of Governing Equations (https://arxiv.org/abs/2502.18377)
- **What's New**: 이번 논문에서는 Mechanistic PDE Networks를 소개하며 데이터에서 지배적인 편미분 방정식을 발견하는 모델을 개발했습니다. 이 네트워크는 시공간 데이터(spatiotemporal data)를 공간-시간 의존적인 선형 편미분 방정식(linear partial differential equations)으로 표현하고, 이를 특정 작업에 대해 해결하여 디코딩합니다. 논문에서는 NeuRLP-PDE라는 GPU 최적화된 선형 PDE 솔버를 비롯하여, 비선형 PDE를 다양한 환경에서 발견할 수 있는 아키텍처를 제안합니다.

- **Technical Details**: Mechanistic PDE Networks는 기계적 인코더(mechanistic encoder), NeuRLP-PDE로 구성된 미분 가능 솔버(differentiable solver), 선택적 디코더(decoder)로 이루어져 있습니다. 인코더는 데이터로부터 여러 개의 선형 편미분 방정식을 생성하며, 해당 방정식들은 Cartesian 격자에서 정의됩니다. NeuRLP-PDE는 메모리 효율적인 방식으로 PDE를 해결하도록 설계되어 있으며, 데이터의 간결함과 정확성을 보장하기 위해 여러 가지 최적화 기법이 적용됩니다.

- **Performance Highlights**: 제안된 Mechanistic PDE Networks는 다양한 편미분 방정식, 예를 들어 반응-확산(reaction-diffusion) 및 나비에-스토크스(Navier-Stokes) 방정식에서 유용함을 입증했습니다. 특히, 이 아키텍처는 잡음(noise)이나 결측 데이터에 대해 견고성을 제공하며, 내부에서 파생된 함수들만 사용하여 파라미터화된 선형 시스템을 효율적으로 해결할 수 있습니다. 이 기술은 데이터 기반의 방정식 발견을 위한 새로운 접근 방식을 제시하고, 과학 및 공학 분야에서의 응용 가능성을 높이고 있습니다.



### WebGames: Challenging General-Purpose Web-Browsing AI Agents (https://arxiv.org/abs/2502.18356)
- **What's New**: WebGames는 AI 에이전트의 웹 브라우징 능력을 평가하기 위해 설계된 포괄적인 벤치마크입니다. 50개 이상의 인터랙티브 챌린지를 포함하여, 인간에게는 간단하지만 현재 AI 시스템의 한계를 체계적으로 테스트합니다. 이 프레임워크는 외부 의존성을 제거하고 검증 가능한 정답으로 재현 가능한 평가를 보장합니다. 공개된 결과는 AI 성능이 인간 성과에 비해 심각한 격차가 있음을 보여줍니다.

- **Technical Details**: WebGames는 다섯 가지 핵심 설계 원칙을 기반으로 하여 AI 시스템을 평가합니다. 인간 중심 디자인, AI 도전 과제, 가벼운 구현, 검증 가능한 완료, 분리된 능력 테스트로 구성됩니다. 각 챌린지는 언제든지 독립적으로 설계되어 특정 브라우저 상호작용 능력을 평가할 수 있도록 되어 있습니다. 이러한 설계를 통해 WebGames는 현대 웹 애플리케이션의 다양한 상호작용 패턴을 포함합니다.

- **Performance Highlights**: 최고의 비전-언어 모델인 GPT-4o, Claude Computer-Use, Gemini-1.5-Pro, Qwen2-VL을 테스트한 결과, 인간의 평균 성공률 95.7%에 비해 AI 시스템은 최대 43.1%의 성공률을 기록했습니다. 이러한 결과는 AI 시스템이 직관적인 웹 상호작용 패턴을 처리하는 데 기초적인 한계를 드러냅니다. WebGames는 AI와 인간 상호작용의 차이를 정확하게 측정할 수 있는 강력한 도구로 자리잡고 있습니다.



### Structural Alignment Improves Graph Test-Time Adaptation (https://arxiv.org/abs/2502.18334)
- **What's New**: 그래프 기반 학습(Graph-based learning)은 추천 시스템(recommendation)부터 사기 탐지(fraud detection) 및 입자 물리학(particle physics)까지 다양한 분야에서 큰 성공을 거두었으나, 네트워크 연결성이나 상호작용 패턴의 변화가 있을 경우 일반화(generalization)에 어려움을 겪습니다. 본 연구에서는 이러한 문제를 해결하기 위한 새로운 알고리즘, 테스트-타임 구조 정렬(test-time structural alignment, TSA) 방식을 탐구하며, 이는 소스 도메인에 다시 접근하지 않고 그래프 구조를 추론할 수 있도록 합니다.

- **Technical Details**: TSA는 그래프 데이터 분포의 변화에 대한 이론적 기초에 구축된 알고리즘으로, 세 가지 주요 전략을 통합합니다. 첫째, 불확실성 인식 이웃 가중치(uncertainty-aware neighborhood weighting)를 사용하여 구조적 변화를 수용하며, 둘째, 노드 표현의 신호 대 잡음비(signal-to-noise ratio)를 기반으로 자기 노드(self-node)와 이웃 집계된 표현(neighborhood-aggregated representations)의 조정을 동적으로 수행합니다. 마지막으로, 잔여 라벨(label) 및 특징(feature) 변화를 보정하기 위해 결정 경계(decision boundary)를 정제합니다.

- **Performance Highlights**: TSA는 합성 데이터셋뿐만 아니라 여러 실제 데이터셋에 대한 실험을 통해 기존의 비그래프 테스트-타임 적응(non-graph TTA) 방법 및 최신 GTTA 기준선을 일관되게 초월하는 성과를 보여주었습니다. TSA는 합성 데이터셋에서 최대 12%의 성능 향상을 이루었고, 실제 데이터셋에 대해서는 평균 10%의 개선을 달성했습니다. 이러한 실험 결과는 TSA의 효과성을 입증하며, 다양한 Graph Neural Network (GNN) 구조에서도 우수한 성능을 발휘했습니다.



### Pretraining Frequency Predicts Compositional Generalization of CLIP on Real-World Tasks (https://arxiv.org/abs/2502.18326)
Comments:
          NeurIPS 2024 Workshop on Compositional Learning: Perspectives, Methods, and Paths Forward

- **What's New**: 이 논문은 CLIP 모델의 조합적 일반화(compositional generalization)에 대한 새로운 성과 조건을 탐구하고 있습니다. 이전 연구에 따르면, CLIP은 개별 개념에 대한 선형 성능 향상을 위해 기하급수적으로 더 많은 사전 학습 데이터(pretraining data)가 필요하다고 합니다. 이 연구는 CLIP 모델이 학습된 구성 요소의 조합으로 새로운 입력을 이해하고, 희귀한 관찰을 일반적인 개념으로 매핑할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구는 Udandarao et al.의 개념 추출 파이프라인을 활용하여, 사전 학습 데이터에 없는 객체 조합을 포함한 검색 테스트 세트를 생성합니다. CLIP 모델은 모든 아키텍처와 규모에 상관없이 이러한 커리어 테스트 세트에서 일관되게 뛰어난 성능을 발휘하는 것을 보여줍니다. 또한, 조합의 독립적인 사전 학습 빈도를 기반으로 CLIP 모델의 조합 능력을 예측하는 방법을 제안합니다.

- **Performance Highlights**: 결과적으로, CLIP 모델은 웹 규모의 데이터에서 객체 클래스에 대한 독립적인 이해를 얻었으며 그 조합 능력의 성공 조건을 파악할 수 있었습니다. 이러한 발견은 CLIP의 일반화 성능을 향상시키기 위해 데이터의 균형을 맞추는 것이 중요하다는 점을 암시하며, 이는 데이터 용량을 늘리지 않고도 효율성과 정확성을 개선하는 데 도움이 될 것입니다.



### Accelerated Training on Low-Power Edge Devices (https://arxiv.org/abs/2502.18323)
- **What's New**: 본 논문에서는 에지 디바이스(Edge Device)에서의 학습을 가속화하기 위해 GPU 주파수(GPU frequency)와 배치 크기(batch size)를 동시에 조정하는 새로운 방법론을 제안합니다. 이 방식은 전력 제약을 준수하면서도 학습 시간을 평균 2.4배 단축하고 에너지 소비를 크게 줄일 수 있음을 보여줍니다. 더불어, 모델의 성능에는 손상을 주지 않고 실제 하드웨어에서의 평가를 통해 이러한 성능 향상을 확인하였습니다.

- **Technical Details**: 연구에서는 시스템과 응용 파라미터의 공동 최적화를 통해 마이크로 컨트롤러에서의 GPU 주파수 및 배치 크기 조합을 최적화합니다. 제안된 방법론은 종단 간 프로파일링(off-line profiling)을 포함하여 각 조합의 전력 및 시간 요구 사항을 측정하고, 목표 정확도(target accuracy) 도달에 필요한 샘플 수 간의 관계를 추정합니다. 이를 통해 최적의 조합을 선택하여 전체 학습 시간을 최소화합니다.

- **Performance Highlights**: 실제 하드웨어에서 다양한 모델(CNN 및 Transformer)을 사용한 실험을 통해, 기존의 최첨단 기술에 비해 학습 속도가 평균 2.4배 향상되었습니다. 또한, 장치에서의 총 에너지 소비가 크게 감소하는 것을 관찰하였으며, 이로 인해 탄소 발자국(carbon footprint) 또한 줄어드는 효과를 보고합니다. 마지막으로, 프로토타입 데이터셋(proxy dataset) 선택에 대한 민감도 분석을 통해 이 방법론의 실용성과 효율성을 입증했습니다.



### Global-Decision-Focused Neural ODEs for Proactive Grid Resilience Managemen (https://arxiv.org/abs/2502.18321)
- **What's New**: 이 논문은 전력 시스템의 파괴적인 재해에 대한 resilience(회복력)를 향상시키기 위한 새로운 결정-making 패러다임인 PATOG(predict-all-then-optimize-globally)를 제안합니다. 기존의 예측 후 최적화 접근 방식(PTO)에서는 예측과 최적화 목표 간의 불일치로 인해 자원 배분의 비효율이 발생했습니다. PATOG는 모든 서비스 유닛의 예측적인 정전 모델을 통합하고, 이를 바탕으로 글로벌 결정을 최적화하는 방안을 제시합니다.

- **Technical Details**: PATOG의 핵심은 글로벌-결정 중심의(GDF) 신경 네트워크(OED) 모델로, 시스템 기능의 장기적 진화를 예측하면서도 전 세계적인 resilience 결정을 최적화합니다. 이 모델은 전송선 조건, 지역 날씨 및 사회 경제적 요인에 기반한 고유의 동적 시스템을 통해 정전 예측을 수행하며, 예측된 정전 발전을 통해 시간과 공간적으로 일관된 자원 배분을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, GDF 모델은 전력을 전략적이고 구조적으루 강화하는데 있어 예측의 일관성, 결정 효율성 및 전체 grid resilience에서 유의미한 향상을 보였습니다. 이 접근법은 기후 재해로 인한 인프라 관리에 있어 보다 스마트하고 선제적인 방법을 제시함으로써 전력 시스템의 복원력을 강화할 잠재력을 지니고 있습니다.



### Bayesian Computation in Deep Learning (https://arxiv.org/abs/2502.18300)
Comments:
          43 pages, 7 figures

- **What's New**: 이번 리뷰 논문은 Markov chain Monte Carlo의 새로운 편집판을 위한 것으로, 딥러닝 모델에 적용된 Bayesian 계산 기법인 근사 추론 기법을 소개합니다. 이 장에서는 Bayesian 신경망(Bayesian Neural Networks)과 심층 생성 모델(Deep Generative Models)을 다루며, 각 기법이 직면하는 후향 추론의 고유한 도전 과제를 설명합니다.

- **Technical Details**: Bayesian 신경망(BNNs)은 고차원 데이터에 대한 불확실성 정량화 문제를 해결하는 데 있어 중요한 역할을 합니다. 우리는 두 가지 주요 추론 방법인 마르코프 체인 몬테카를로(Markov Chain Monte Carlo, MCMC)와 변분 추론(Variational Inference, VI)에 중점을 둡니다. 특히 Stochastic Gradient 기반 최적화 기법을 활용하여 BNN computations을 수백만 개의 데이터 포인트에 맞게 확장할 수 있습니다.

- **Performance Highlights**: 딥러닝은 이미지 분류, 분할, 음성 인식 및 비디오 생성과 같은 다양한 작업에서 높은 정확도를 달성하고 있습니다. Bayesian 모델 평균(Bayesian Model Averaging)은 예측 정확도를 개선하는 데 기여할 수 있으며, 의료와 같은 안전이 중요한 분야에서 의사 결정 시 신뢰할 수 있는 불확실성 추정이 필요합니다. 이 논문에서 제시하는 근사(Bayesian) 추론 기법들은 이러한 요구를 충족하는 데 필수적인 기초 기술로 자리 잡고 있습니다.



### DeepCircuitX: A Comprehensive Repository-Level Dataset for RTL Code Understanding, Generation, and PPA Analysis (https://arxiv.org/abs/2502.18297)
Comments:
          8 pages, 3 figures

- **What's New**: 이번 논문에서는 RTL(레지스터 전송 레벨) 코드 이해, 생성 및 전력-성능-면적(PPA) 분석을 진전시키기 위해 설계된 포괄적인 저장소 수준의 데이터셋인 DeepCircuitX를 소개합니다. 이전 데이터셋들과는 달리, DeepCircuitX는 저장소, 파일, 모듈 및 블록 수준의 RTL 코드를 아우르는 체계적인 리소스를 제공합니다. 또한, Chain of Thought (CoT) 주석이 포함되어 각 수준에서의 기능 및 구조에 대한 세부 설명을 제공합니다.

- **Technical Details**: DeepCircuitX는 4,000개 이상의 회로 설계 프로젝트를 수집하여, 다양한 시나리오에 맞춰 파일, 모듈 및 블록으로 분할된 완전한 구조체로 이루어져 있습니다. 이 데이터셋은 또한 RTL 코드로부터 직접 PPA 예측을 가능케 하는 합성된 넷리스트와 PPA 메트릭스를 포함하며, 이는 조기 설계 탐색을 촉진합니다. CoT 주석 방법을 통해 각 수준에서 코드를 설명하는 주석과 질문-답변 쌍을 생성하여 LLMs(대형 언어 모델)의 학습 데이터 품질을 향상시킵니다.

- **Performance Highlights**: DeepCircuitX를 활용하여 훈련된 LLM들은 전반적으로 모든 메트릭에서 성능이 크게 향상되었습니다. 모델 규모(예: CodeT5의 220M, 7B, 16B)와 상관없이 LLM들이 Fine-tune 되었을 때 실질적인 이점이 드러났으며, PPA 예측의 정확성을 높이기 위한 도전 과제가 여전히 남아있음을 보여주었습니다. 이 데이터셋은 RTL 코드 이해, 생성, PPA 예측 등 EDA 작업에서의 능력 확장을 위한 중요한 자원으로 평가받고 있습니다.



### AMPO: Active Multi-Preference Optimization (https://arxiv.org/abs/2502.18293)
- **What's New**: 이번 연구에서는 Active Multi-Preference Optimization (AMPO)라는 새로운 접근법을 소개합니다. AMPO는 대량의 후보 응답을 평가하고, 중요한 극단과 독특한 의미 군을 포괄하는 소규모 정보 집합을 선택하여 언어 모델의 선호 최적화를 가능하게 합니다. 이를 통해 다중 선호 최적화의 과정에서 모델이 학습하는 신호의 질을 높입니다.

- **Technical Details**: AMPO는 (a) 정책 기반 데이터 생성, (b) 그룹 기반 선호 학습, 그리고 (c) 능동적인 서브셋 선택을 통합하는 프레임워크입니다. 고유의 그룹 대조 손실 함수인 Refa를 채택하여 다수의 긍정적 및 부정적 응답을 단일 손실 항목으로 결합합니다. 또한, 선택된 응답의 다양성을 극대화하는 방법으로 베이직한 bottom-K에서 이론적으로 기반한 Opt-Select 방식까지 다양한 능동 선택 방식을 탐구합니다.

- **Performance Highlights**: AMPO는 Llama 8B 모델을 기반으로 한 AlpacaEval에서 최신 성과를 달성하였으며, Simpo와 같은 강력한 기준선 모델을 초월했습니다. 연구자들은 Hugging Face에 AMPO-Coreset-Selection 및 AMPO-Opt-Selection 데이터셋을 공개하여 다중 선호 정렬 연구를 촉진합니다.



### Iterative Counterfactual Data Augmentation (https://arxiv.org/abs/2502.18249)
Comments:
          AAAI 2025

- **What's New**: 이 논문에서는 반사실적 데이터 증강(Counterfactual Data Augmentation, CDA) 기법의 새로운 방법론인 반복적 반사실적 데이터 증강(Iterative Counterfactual Data Augmentation, ICDA)을 소개합니다. ICDA는 초기 고잡음 개입을 통해 훈련 데이터셋의 신호와 레이블 간의 상호정보(mutual information)를 높이고, 불필요한 신호의 정보를 줄이는 결과를 가져옵니다. 이 방식은 수동 규정 또는 기존 알고리즘 기반 방법론보다 더 효과적으로 작동합니다.

- **Technical Details**: ICDA 절차는 데이터셋 내에서 다루고자 하는 주요 신호와 불필요한 신호 간의 상호정보를 유지하면서, 훈련 데이터셋에서 높은 잡음을 가지던 데이터를 단계적으로 정제하는 과정을 포함합니다. 실험에서는 여섯 개의 인간이 생성한 데이터셋과 두 개의 대규모 언어 모델에 의해 생성된 데이터셋을 사용하였습니다. 훈련된 데이터셋은 인간의 주석과 더 잘 일치하는 합리적인 문서 내용을 생성합니다.

- **Performance Highlights**: 실험 결과는 ICDA를 통해 생성된 데이터셋에서 모형이 보다 효과적으로 학습하고, 결과적으로 인간 주석과 유사한 이해도를 가지는 문서가 생성됨을 보여주었습니다. 연구진은 이 기법이 학습의 질을 높이는 데 기여하며, 기존의 CDA 방식에 비해 명확한 성능 향상을 나타냈음을 강조했습니다.



### Causal AI-based Root Cause Identification: Research to Practice at Sca (https://arxiv.org/abs/2502.18240)
- **What's New**: 이 논문에서는 전통적인 상관관계(correlation) 개념을 넘어 원인(causation)에 초점을 맞춘 새로운 Root Cause Identification (RCI) 알고리즘을 개발했음을 강조합니다. 이 알고리즘은 IBM Instana에 통합되어 현재 기업 고객에게 실제로 사용되고 있으며, 실시간으로 성능 문제를 진단하는 데 기여하고 있습니다.

- **Technical Details**: 논문의 핵심은 'causal AI'라는 개념을 활용하여 시스템의 신뢰성을 높이는 것입니다. RCI 알고리즘은 복잡한 분산 시스템에서의 실패를 진단하는 데 필요한 이론적 기초와 실제 구현 측면을 논의합니다. 이는 다양한 모듈과 팀, 데이터 센터로 구성된 현대 애플리케이션 환경에서 빅데이터(bigger data)를 측정하고 분석하는 데 유용합니다.

- **Performance Highlights**: Instana는 근 실시간(near real-time)으로 문제를 식별할 수 있는 뛰어난 진단 능력을 갖추고 있으며, 이는 기존의 Application Performance Management (APM) 도구들과의 차별점으로 작용합니다. 실제 사례를 통해, 본 알고리즘이 현재의 복잡한 시스템 환경에서 신뢰성과 성능을 향상시키는 방법이 설명됩니다.



### Unveiling and Causalizing CoT: A Causal Pespectiv (https://arxiv.org/abs/2502.18239)
- **What's New**: 이 논문은 Chain-of-Thought (CoT)의 이유를 명확히 하고 이해할 수 있도록 하는 접근 방식을 제안합니다. CoT는 대형 언어 모델(LLMs)의 추론 능력을 향상시키는 데 큰 성과를 올렸지만, 그 메커니즘은 여전히 불투명한 '블랙 박스'로 남아 있었습니다. 저자들은 인과적 모델링을 통해 CoT의 각 단계가 올바르고 이해 가능하게 만들기 위한 방법을 제시합니다.

- **Technical Details**: 논문에서는 구조적 인과 모델(Structural Causal Model, SCM)을 사용하여 CoT의 인과 관계를 모델링합니다. CoT에서의 각 단계 간 관계를 측정하기 위해 CoT 평균 인과 효과(CACE)를 정의하고, 첫 번째 단계에서의 인과 효과(First-Step Causal Effect, FSCE)를 도입하여 논리적 인과성을 정량화합니다. 이러한 방법을 통해 CoT에 내재된 인과 오류를 교정하고, 모든 단계에서의 유의미한 결과를 도출합니다.

- **Performance Highlights**: 저자들은 공개 소스 및 비공개 소스 LLM에 대한 실험 결과를 제시하며, 인과 오류가 효과적으로 수정되고 LLM의 추론 능력이 유의미하게 향상됐음을 보여주었습니다. 이 연구는 CoT의 메커니즘을 인과적으로 드러내어, 언어 모델의 추론이 더 정확하고 이해 가능하게 만드는데 기여할 것입니다. 결과적으로, CoT의 효용이 현실 세계의 문제 해결에 연결된 인과 관계를 반영한다는 가정을 기반으로 합니다.



### Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints (https://arxiv.org/abs/2502.18237)
Comments:
          Accepted at ICLR 2025

- **What's New**: 이 논문에서는 합성 (Synthetic) 표 형 데이터 생성의 한계를 극복하기 위해 새로운 레이어인 Disjunctive Refinement Layer (DRL)을 소개합니다. DRL은 생성된 데이터가 사용자 정의 제약 조건에 잘 부합하도록 보장하는 기능을 갖추고 있습니다. 이는 기존의 깊이 있는 생성 모델 (Deep Generative Models)에서 발생하는 비 현실적인 데이터 포인트 생성 문제를 해결하는 데 큰 기여를 합니다.

- **Technical Details**: DRL은 비구조적이거나 비연속적인 영역도 정의할 수 있는 제약 조건으로, 제약이 없는 선형 공식 (quantifier-free linear formulas)으로 표현된 복잡한 제약 조건들을 처리할 수 있는 첫 번째 방법입니다. 이 방법은 자동으로 깊이 있는 학습 모델이 제약 조건을 준수하도록 만듭니다. 실험 분석을 통해 DRL은 제약 준수 (constraint satisfaction)를 보장하며, 후속 작업 (downstream tasks)에서의 효율성을 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: DRL을 적용한 결과, 기존 깊이 있는 생성 모델에서 발생하던 제약 위반을 완전히 제거했습니다. 또한, F1 점수 (F1-score)를 21.4%까지, ROC 곡선 아래 면적 (Area Under the ROC Curve)을 20.9%까지 향상시키며 실제 데이터 생성에 대한 긍정적인 영향을 입증했습니다.



### Software implemented fault diagnosis of natural gas pumping unit based on feedforward neural network (https://arxiv.org/abs/2502.18233)
Comments:
          11 pages, 9 figures

- **What's New**: 최근 몇 년간 가스 펌핑 유닛(GPU) 진단을 위한 인공 신경망(ANN) 사용이 주목받고 있습니다. 기존의 ANN 훈련은 주로 GPU 작업 모델에 기반하였으나, 이 연구에서는 실제 GPU의 음향 및 진동 과정의 특성을 입력 데이터로 사용하여 ANN을 훈련시키는 방법을 제안합니다. 이는 GPU의 실제 상태를 평가할 수 있는 새로운 접근이 됩니다.

- **Technical Details**: 이 연구는 GTK-25-i 유형의 GPU에서 발생하는 실제 진동 및 음향 과정에 대한 기술적 분석을 수행하였습니다. ANN의 입력으로 들어오는 진단 신호 패킷을 생성하였으며, 이는 진동 및 음향 신호의 최대 진폭 성분과 표준 편차 값을 포함합니다. TensorFlow, Keras, NumPy 및 pandas와 같은 프레임워크를 활용하여 심층 완전 연결형 피드포워드 ANN 아키텍처를 개발하고, 오류 역전파 알고리즘을 통한 훈련을 진행했습니다.

- **Performance Highlights**: 훈련 및 테스트 결과, '정상' 상태의 1,475 신호 샘플에서 신호 분류 정확도는 1.0000으로 나타났습니다. '현재' 상태의 샘플 정확도는 0.9853, 그리고 '결함 있는' 상태에서는 정확도가 0.9091로 측정되었습니다. 이러한 결과는 개발된 ANN이 GPU의 기술적 상태를 실용적인 수준의 정확도로 분류할 수 있음을 시사하며, 이는 GPU 고장을 예방하는 데 기여할 수 있습니다.



### DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches (https://arxiv.org/abs/2502.18202)
- **What's New**: DenoMAE2.0는 전통적인 reconstruction loss와 함께 지역 패치 분류(objective) 목표를 통합하여 표현 학습 및 견고성을 개선한 향상된 denoising masked autoencoder입니다. 기존의 Masked Autoencoders(MAE)와는 달리 DenoMAE2.0은 마스킹되지 않은 패치에 대한 위치 인식을 포함하여 로컬 피쳐를 세밀하게 포착하는 동시에 전역 일관성을 유지합니다. 이 접근법은 특히 무선 통신의 반지도 학습(semisupervised learning)에서 높은 노이즈와 데이터 부족 문제가 심각한 상황에서도 이점을 제공합니다.

- **Technical Details**: DenoMAE2.0의 전체 프레임워크는 인코더, 재구성 denoising 디코더, 지역 패치 분류 헤드의 세 가지 구성 요소로 이루어져 있습니다. 첫 번째 단계에서는 입력 이미지를 비중첩(non-overlapping) 패치 시퀀스로 변환하며, 무작위로 75%의 패치를 마스킹합니다. 이렇게 생성된 가시 패치(visible patches)는 Vision Transformer(ViT) 인코더를 통해 패치 수준 특성을 생성하고, 이를 기반으로 지역 패치 분류의 목표를 설정하여 보다 구체적이고 유용한 표현을 학습합니다.

- **Performance Highlights**: DenoMAE2.0은 Deno-MAE와 기타 여러 기준선들과 비교하여 노이즈 제거 품질 및 다운스트림 분류 정확도 모두에서 우수한 성능을 보여줍니다. 데이터셋에서 DenoMAE에 비해 1.1%의 개선을 달성했으며, RadioML 벤치마크에서 별도의 분류 작업에서 11.83% 및 16.55%의 유의미한 정확도 향상을 기록했습니다. 이러한 결과는 DenoMAE2.0이 반지도 학습(ssemi-supervised learning) 환경에서 효과적으로 작동한다는 것을 보여줍니다.



### Training Consistency Models with Variational Noise Coupling (https://arxiv.org/abs/2502.18197)
Comments:
          23 pages, 11 figures

- **What's New**: 본 논문에서는 최근 이미지 생성 작업에서 경쟁력 있는 성능을 달성하고 있는 Consistency Training (CT)의 새로운 접근법을 소개합니다. 이는 Variational Autoencoders (VAE)의 구조에서 영감을 받은 노이즈 결합 (noise-coupling) 방식을 통해 데이터 의존적인 노이즈 방출 모델을 학습함으로써 이루어집니다. 이러한 방법은 고전적인 CT에서 고정된 전방 프로세스와 달리 노이즈-데이터 매핑의 기하학을 간접적으로 학습할 수 있게 합니다.

- **Technical Details**: 연구에서는 Flow Matching 프레임워크를 이용하여 노이즈 결합을 구현하는 새로운 CT 교육 방법을 제안합니다. 데이터에 따라 조정된 확률 분포를 학습하고, 추가적인 Kullback-Leibler divergence 손실로 정규화하여 end-to-end 훈련 절차를 개발합니다. 이러한 접근법은 데이터 의존적인 결합 분포를 통해 목표 ODE 흐름의 위치를 조정하여 모델의 학습 목표를 용이하게 만들어 줍니다.

- **Performance Highlights**: 실험 결과 다양한 이미지 데이터셋에서 상당한 생성 성능 향상이 나타났으며, 제안된 모델은 CIFAR-10의 비증류 CT FID에서 기존 최첨단 성능을 초과했습니다. 뿐만 아니라 ImageNet의 경우 $64 \times 64$ 해상도에서 2단계 생성 시 최첨단과 동등한 FID를 달성했습니다. 이러한 성과는 제안된 방법이 고차원 데이터에 대한 확장성과 효과적인 생성 성능을 제공함을 보여줍니다.



### Graph Augmentation for Cross Graph Domain Generalization (https://arxiv.org/abs/2502.18188)
- **What's New**: 이 논문은 그래프 신경망(GNN)을 활용한 교차 그래프 노드 분류 문제에 대해 새로운 데이터 증강(data augmentation) 기법을 제안합니다. 기존의 연구에서는 모델 훈련에 초점을 맞추었지만, 본 연구에서는 노이즈 엣지를 제거하고 디자인된 엣지 추가 방식을 통해 GNN의 일반화 능력을 개선하고 있습니다. 특히, 엣지 드롭핑(edge dropping) 기술과 클러스터링 기반의 엣지 추가(edge adding) 방식을 통해 구조의 불변 정보(invariant information)를 효과적으로 캡처할 수 있는 방법을 소개합니다.

- **Technical Details**: 제안된 방법은 먼저 저중량 엣지 드롭핑(low-weight edge dropping) 기술을 이용하여 그래프의 노이즈 엣지를 제거하고, 구조를 그에 맞게 조정합니다. 이후, 동일한 분포를 가진 노드 특징을 기반으로 클러스터링을 통해 새로운 엣지를 추가하여 불변 구조(invariant structures)를 생성합니다. 이 두 가지 기법을 통해 GNN은 다양한 환경에서 일반화 성능을 높일 수 있도록 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안한 접근법은 기존의 데이터 증강 방법들과 비교해 최고의 성능을 보였습니다. 특히, 인식하지 못하는 분포(OOD) 조건 하에서도 GNN의 분류 정확도가 크게 향상되었습니다. citation network 데이터셋에서의 실험을 통해 본 방법의 효과성과 일반화 능력이 입증되었습니다.



### Sharper Concentration Inequalities for Multi-Graph Dependent Variables (https://arxiv.org/abs/2502.18167)
Comments:
          34 pages

- **What's New**: 본 논문은 그래프에 의존하는 데이터가 포함된 Multi-Task Learning (MTL)에서 기존 이론 분석의 한계를 해결하기 위해 새로운 Bennett 불평등을 제안하고 있습니다. 이를 통해 기존에 비해 더 강력한 risk bound의 성과인 O(\frac{\log n}{n})을 얻을 수 있습니다. 또한, 이 새로운 Bennett 불평등을 바탕으로 Talagrand 불평등을 제안하고, 이를 통해 MTL의 이론적 일반화 분석을 위한 새로운 분석 프레임워크를 개발했습니다.

- **Technical Details**: 이 연구에서는 다중 그래프 의존(random variables with multi-graph dependence) 상황에서의 MTL 문제를 다루기 위해 새로운 Bennett concentration inequality를 도입했습니다. 이 불평등은 기존의 단일 그래프 의존 변수에 대한 결과(특히 Ralaivola & Amini, 2015)를 포괄할 수 있으며, 새로운 Talagrand 불평등을 제안하여 empirical process의 리스크 경계를 더 세부적으로 분석할 수 있게 합니다. 이를 통해 Local Rademacher Complexity (LRC) 개념을 활용하여 이론적 일반화 분석을 강화했습니다.

- **Performance Highlights**: 이론적 분석의 성공적인 적용 예로는 Macro-AUC Optimization이 있습니다. 실험 결과에 따르면, 새롭게 제안된 risk bound는 O(\frac{\log n}{n})로, Wu et al. (2023)로부터 도출된 O(\frac{1}{\sqrt{n}})의 결과보다 우수함을 보여줍니다. 또한, 이 연구에서 제시된 이론적 결과는 실험적으로도 뒷받침되며, 이는 MTL의 다중 그래프 의존 데이터에 대한 새로운 접근법의 유용성을 강조합니다.



### SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation (https://arxiv.org/abs/2502.18153)
- **What's New**: 이 논문은 근사적 2차 최적화 방법이 일반적으로 1차 접근 방식에 비해 낮은 일반화 성능을 보인다는 문제를 탐구합니다. 기존의 2차 방법들이 더 뚜렷한 샤프(minima)로 수렴하는 경향이 있다는 것을 발견하고, 솔루션의 샤프함을 명시적으로 줄이면서 일반화 성능을 향상시키기 위해 Sassha라는 새로운 방법을 제안합니다. 이를 통해 시간 효율성을 확보하고 안정성을 증대시키는 방법론을 제시합니다.

- **Technical Details**: Sassha는 'Sharpness-aware Adaptive Second-order optimizer with Stable Hessian Approximation'의 약자로, 기본적으로 Hessian의 대각선을 추정하는 2차 최적화 프레임워크에 SAM 방식의 샤프함 최소화 기법을 통합합니다. 이 기법은 샤프함 감소 프로세스를 시행할 때 수치적으로 불안정해질 수 있지만, 문헌에서 연구된 원칙에 따라 설계된 여러 가지 조치를 통해 안정성을 증대시킵니다. 이론적으로 Sassha는 예측된 커브를 부드럽게 조정하고, 이전에 계산된 Hessians의 효율적인 재사용을 가능함으로써 안정적이고 효율적인 알고리즘을 구현합니다.

- **Performance Highlights**: Sassha는 다양한 시각 및 자연어 작업에 대해 광범위하게 평가되었으며, 기존 2차 방법 및 SGD, AdamW, SAM과 같은 1차 방법에 비해 더 평평한 미니마와 강화된 일반화 성능을 꾸준히 달성하였습니다. 논문에서는 수렴성(convergence), 강건성(robustness), 안정성(stability), 효율성(efficiency), 비용(cost)에 대한 포괄적인 분석을 제공하여 Sassha의 성과를 심층적으로 연구하였습니다.



### Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations (https://arxiv.org/abs/2502.18147)
- **What's New**: 이번 연구에서는 Sparse autoencoders (SAEs)를 넘어 Jacobian SAEs (JSAEs)를 제안합니다. JSAEs는 모델 구성 요소의 입력과 출력 활성화에서뿐만 아니라 이들을 연결하는 계산에서도 희소성(sparsity)을 제공합니다. 이를 통해 LLMs의 계산을 이해하는 데 기여할 수 있는 새로운 방향성을 제시합니다.

- **Technical Details**: JSAEs의 주요 기술 기여는 Jacobians를 효율적으로 계산하는 방법을 찾는 것입니다. LLM의 크기 때문에 Jacobians를 직접 계산하는 것은 실용적이지 않지만, 적절한 기초(JSAE basis)로 재작성할 경우 MLP는 대략적으로 선형(linear)으로 변환됩니다. 이 방법은 계산 희소성을 유지하면서도 전통적인 SAEs와 유사한 LLM 성능을 제공합니다.

- **Performance Highlights**: 실험 결과, JSAEs는 사전 훈련된 LLM에서 계산 희소성이 더 높게 나타났습니다. 이는 LLM이 훈련을 통해 학습한 계산 그래프의 희소성이라는 특성을 지니고 있음을 보여줍니다. 따라서 JSAEs는 표준 SAEs보다 학습된 transformer 계산을 이해하는 데 더 적합할 수 있다는 가능성을 제시합니다.



### Actively Inferring Optimal Measurement Sequences (https://arxiv.org/abs/2502.18142)
- **What's New**: 본 논문에서는 저차원 표현을 활용하여 고차원 데이터 회수를 최적화하는 능동적 순차 추론 알고리즘을 제안합니다. 이는 변분 자동인코더 (Variational Autoencoder, VAE)의 저차원 표현(latent space)을 이용하여 다음 측정값을 선택하는 데에 집중합니다. 이러한 접근 방식은 최소한의 측정으로 최대의 정보를 획득하려는 목표를 가지고 있습니다.

- **Technical Details**: 제안된 알고리즘은 부분적으로 측정된 데이터를 수정된 VAE 인코더를 통해 매핑하여 전체 상태를 추론하도록 설계되었습니다. VAE는 이미지 재구성과 같은 다양한 작업에 적합하며, 이를 위해 저차원 가우시안 분포를 학습합니다. 알고리즘은 잠재 공간에서 샘플을 추출하고 생성된 데이터를 통해 조건부 값을 추정합니다.

- **Performance Highlights**: 패션 MNIST 데이터셋을 사용하여 알고리즘의 성능을 입증하였으며, 10단계 이내에 유용한 패턴을 선택할 수 있음을 보여주었습니다. 기존의 확률적 변분 추론(Stochastic Variational Inference)보다 더 효과적으로 배치 처리할 수 있어, 최소의 측정으로 우수한 결과를 달성하는 것을 확인했습니다.



### SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inferenc (https://arxiv.org/abs/2502.18137)
- **What's New**: SpargeAttn은 다양한 생성 작업에 보편적으로 적용할 수 있는 새로운 sparse attention 기법입니다. 기존의 sparse attention이 특정 태스크에 맞춰 최적화된 반면, SpargeAttn은 기계 학습 모델의 end-to-end 성능을 유지하면서 다양한 모델에 대해 속도를 높일 수 있도록 설계되었습니다.

- **Technical Details**: SpargeAttn의 핵심 기술은 두 단계의 온라인 필터를 활용하여 sparse attention을 구현하는 것입니다. 첫 번째 단계에서는 입력의 attention 맵을 신속하고 정확하게 예측하여 필요하지 않은 행렬 곱셈을 생략합니다. 두 번째 단계에서는 온라인 softmax-aware 필터를 통해 추가적인 오버헤드 없이 좀 더 많은 행렬 곱셈을 생략합니다.

- **Performance Highlights**: SpargeAttn은 기존의 dense 및 sparse attention 모델에 비해 2.5배에서 5배 더 빠른 속도를 자랑합니다. 이 방법은 언어 모델링, 이미지 및 비디오 생성 등 다양한 작업에서 model의 end-to-end 성능을 유지하며 효율성을 극대화합니다.



### EU-Nets: Enhanced, Explainable and Parsimonious U-Nets (https://arxiv.org/abs/2502.18122)
- **What's New**: 본 연구에서는 모든 U-Net 아키텍처에 적응 가능한 프레임워크인 MHEX+를 제안합니다. MHEX+를 기반으로 한 EU-Nets라는 새로운 U-Net 변형을 도입하여 전통적인 U-Net 모델의 한계를 극복하고 성능 및 안정성을 향상시키는 것을 목표로 합니다. Equivalent Convolutional Kernel과 협업 그라디언트 접근법은 해석 가능성을 높이고 불확실성 추정을 개선합니다.

- **Technical Details**: MHEX+는 MHEX 프레임워크를 확장한 것으로, 모든 U-Net 아키텍처에 적용 가능하도록 설계되었습니다. 이 연구에서 제안한 Equivalent Convolutional Kernel은 연속적인 합성곱 레이어를 통합하여 해석 가능성을 높이며, 협업 그라디언트 방법을 통해 디코더 레이어 간의 기울기 일관성을 측정하여 불확실성 추정을 합니다.

- **Performance Highlights**: EU-Nets는 실험에서 모든 네트워크와 데이터셋에 대해 평균 정확도를 1.389% 개선하고 분산을 0.83% 줄였습니다. MHEX+ 방식은 불과 0.1M 미만의 파라미터로 성능을 향상시킵니다. 이러한 성과는 MHEX+ 프레임워크가 U-Net 구조의 해석 가능성과 불확실성 처리를 더욱 강화하는 데 기여함을 보여줍니다.



### Stackelberg Game Preference Optimization for Data-Efficient Alignment of Language Models (https://arxiv.org/abs/2502.18099)
- **What's New**: 본 연구에서는 인간의 선호와 정렬된 언어 모델의 중요성을 강조하며, 데이터 효율적인 방식으로 이를 추구하는 Stackelberg Game Preference Optimization (SGPO) 방법을 제안합니다. SGPO는 정책(리더)이 인간의 선호와 일치하도록 최적화하는 과정에서 발생할 수 있는 최악의 선호 분포(팔로워)에 대응하게 됩니다. 이를 통해 기존의 Direct Preference Optimization (DPO)보다 최소한의 인적 주석으로도 강력한 결과를 낼 수 있음을 보여줍니다.

- **Technical Details**: SGPO 프레임워크는 두 개의 플레이어가 경쟁하는 Stackelberg 게임으로 모델링됩니다. 정책은 인간의 실제 선호를 충족하기 위해 작업하고, 적대적인 선호 분포는 정의된 Wasserstein 볼 내의 최악의 변화를 탐색합니다. 본 연구에서는 ϵ-Wasserstein 공을 사용하여 데이터 변화에 강인성을 확보하면서도 𝒪⁢(ϵ)로 정의된 후회 가드를 증명합니다.

- **Performance Highlights**: 실험 결과, Stackelberg Self-Annotated Preference Optimization (SSAPO) 알고리즘을 통해 단 2K seed preferences만으로도 Mistral-7B에서 35.82%, Llama3-8B-Instruct에서는 40.12%의 GPT-4 승률을 기록했습니다. 이는 기존의 다른 모델과 비슷한 성능으로, 적은 양의 인적 주석으로도 효과적인 선호 정렬을 이룰 수 있음을 보여줍니다.



### The Built-In Robustness of Decentralized Federated Averaging to Bad Data (https://arxiv.org/abs/2502.18097)
Comments:
          Funding: SoBigData PPP (101079043), this http URL (PNRR IR0000013), FAIR (PNRR PE00000013), RESTART (PNRR PE00000001)

- **What's New**: 이번 연구는 분산형 연합 학습(DFL)이 저품질 또는 부패된 데이터와 어떻게 상호작용하는지를 심층적으로 조사합니다. 특히, 각 노드에서 로컬 데이터 분포에 대한 평가의 어려움과 중앙집중식 관리자 부재 시 발생할 수 있는 위험 요소들을 다룹니다. 이 연구는 두 가지 데이터 품질 저하 시나리오를 시뮬레이션하여, 분산형 학습이 저품질 데이터의 영향을 얼마나 잘 견딜 수 있는지를 탐구합니다.

- **Technical Details**: DFL 시스템은 여러 노드가 참여하여 협업적으로 모델을 학습할 수 있지만, 그 과정에서 데이터 품질의 변동이 영향을 미칠 수 있습니다. 본 연구에서는 FedAvg의 분산형 구현을 기반으로 하여, 부패된 데이터가 네트워크 내 특정 노드에 고르게 분포되거나 하나의 노드에 집중되는 경우를 시뮬레이션했습니다. 불완전한 데이터를 생성하기 위해 미리 훈련된 Generative Adversarial Network (GAN)에서 잠재 공간 내에서 보간 기법을 사용합니다.

- **Performance Highlights**: 연구 결과, 평균 기반의 분산 학습이 국소적인 부정확한 데이터에 대해 놀라울 정도로 강인함을 보였습니다. 특히, 부패된 데이터가 단일 노드에 집중될 경우, 그 노드의 중앙성에 관계없이 모델의 전반적인 학습 과정에 미치는 영향이 최소화되는 경향을 보였습니다. 또한, 분산형 학습이 중앙 집중화된 학습보다 데이터 부패에 더 강한 회복력을 보이며, 이는 추가적인 비교 분석을 필요로 한다는 점을 강조합니다.



### HEROS-GAN: Honed-Energy Regularized and Optimal Supervised GAN for Enhancing Accuracy and Range of Low-Cost Accelerometers (https://arxiv.org/abs/2502.18064)
Comments:
          AAAI Oral; AI for Sensors; Generative Deep Learning

- **What's New**: 본 논문에서는 저비용 가속도계의 정확도와 범위를 개선하기 위해 고안된 HEROS-GAN(하이온 에너지 정규화 및 최적 감독 GAN)을 제안합니다. 이 GAN은 저비용 센서 신호를 고비용 신호로 변환하여 정밀도와 범위의 제한을 극복하며, 최적 수송 이론(Optimal Transport Theory)을 활용하여 unpaired 데이터 간의 일관성을 탐색합니다. 또한, 새로운 Low-cost Accelerometer Signal Enhancement Dataset (LASED)를 구축하여 연구 커뮤니티에 공개했습니다.

- **Technical Details**: HEROS-GAN의 주요 기법으로는 Optimal Transport Supervision(OTS)과 Modulated Laplace Energy(MLE)가 있습니다. OTS는 비짝짓기 데이터 간의 잠재적 상관관계를 최대한 활용하도록 설계되어 Supervisory information을 극대화합니다. MLE는 모델이 범위 제한을 극복하고 세부 신호를 강화할 수 있도록 적절한 에너지를 주입합니다.

- **Performance Highlights**: 실험 결과, HEROS-GAN은 OTS 또는 MLE만 사용한 기존 SOTA(signal enhancement) 방법을 크게 초과하는 성능을 보임을 입증했습니다. OTS와 MLE를 통합한 HEROS-GAN은 가속도계의 범위를 두 배로 늘리고, 신호 잡음은 두 자릿수로 줄이는 놀라운 결과를 달성하여, 가속도계 신호 처리의 벤치마크를 수립했습니다.



### A Market for Accuracy: Classification under Competition (https://arxiv.org/abs/2502.18052)
Comments:
          26 pages

- **What's New**: 이번 논문에서는 소비자 시장 내에서의 경쟁을 고려한 머신러닝 모델의 학습 방법을 제안합니다. 전통적인 접근 방식이 경쟁업체의 존재를 무시하고 정확도만을 강조하는 문제를 다루고 있으며, 시장 점유율을 극대화하는 방향으로 분류(classification) 방법을 발전시킵니다. 이 연구는 제공업체와 소비자, 시장 간의 상호작용을 이해하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자는 경쟁 구조를 기반으로 한 학습 설정을 도입하여, 사용자가 특징(feature) x와 레이블(label) y로 설명되는 구조를 가지며, 이로부터 얻은 데이터와 각 제공업체의 예측 성능 간의 상관관계를 분석합니다. Ben-Porat & Tennenholtz (2019)의 시장 모델을 바탕으로, 여러 제공업체 간의 상호작용에 대한 다이나믹스를 설명하며 최적의 분류기를 찾는 과정이 전통적인 분류 문제와 유사하다는 점을 강조합니다. 이를 통해, 특정 상황에서 표준 학습 기법을 활용하는 것이 효율적일 수 있음을 시사합니다.

- **Performance Highlights**: 경쟁적인 학습 환경에서의 실험 결과, 정확도 시장의 동역학이 어떻게 진행되는지를 보여줍니다. 많은 시장에서 경쟁이 소비자 복리를 개선하게 되며, 이는 시장의 효율성을 증가시키는 결과로 이어집니다. 연구의 실증 결과는 시장이 빠르게 수렴하여 균형을 이루며, 소비자와 제공업체 모두에게 유리한 결과를 도출한다는 점을 강조합니다.



### Enhancing 5G O-RAN Communication Efficiency Through AI-Based Latency Forecasting (https://arxiv.org/abs/2502.18046)
- **What's New**: 이 연구는 5G Open Radio Access Networks (O-RAN)에서 인공지능(AI) 기반의 실시간 대기 시간(latency) 예측 시스템을 소개합니다. 기존의 머신러닝(ML) 방법들이 실제적인 확장성 및 하드웨어 검증이 부족했던 반면, 이 논문은 FlexRIC 프레임워크를 사용하여 기능성 O-RAN 프로토타입에 통합된 시스템을 제안합니다. 이 시스템은 Bidirectional Long Short-Term Memory (LSTM) 모델을 활용하여 동적인 5G 환경 내에서 대기 시간을 실시간으로 예측합니다.

- **Technical Details**: 대기 시간 예측 아키텍처는 GPU 기반 작업 스테이션과 소프트웨어 정의 라디오 장치, Nvidia Jetson Nano 및 5G 모뎀으로 구성됩니다. 리눅스 컨테이너(LXC)를 활용하여 유연성과 확장성을 높였으며, 모든 도구는 오픈 소스 환경에서 사용되었습니다. Bidirectional LSTM 모델을 통해 56k 행의 초기 데이터 세트를 활용하여 대기 시간 예측 모델을 훈련했으며, 모델 성능은 0.04 이하의 손실 지표를 기록하며 실시간 예측을 가능하게 합니다.

- **Performance Highlights**: 시스템은 실시간으로 대기 시간을 예측하고 이 예측 결과와 실제 값을 비교하는 기능을 보여주었습니다. 실험 결과, 이 모델은 동적 자원 관리와 적응형 의사 결정을 통해 네트워크 성능을 최적화하는 데 기여했습니다. 이 연구는 5G O-RAN의 신뢰성과 효율성을 개선하는 다음 세대 무선 네트워크 최적화를 위한 기초를 마련하였으며, 향후 연구에서는 더 진보된 모델과 시스템 배출 CO2 감량 측정이 포함될 예정입니다.



### ExPath: Towards Explaining Targeted Pathways for Biological Knowledge Bases (https://arxiv.org/abs/2502.18026)
Comments:
          Under review

- **What's New**: 이번 논문에서는 생물학적 지식 기반에 실험 데이터를 통합하여 더 정확한 경로 추론(pathway inference)을 제공하는 새로운 프레임워크인 ExPath를 제안합니다. ExPath는 특히 아미노산 서열(AA-seqs)을 사용하여 바이오 네트워크(bio-networks)의 다양한 그래프를 분류하는 방식을 채택합니다. 경로 확인과 관련하여 분류에 기여하는 링크를 목표로 다루며, 이러한 경로는 더욱 구체적이고 타겟화된 정보로서 활용될 수 있습니다.

- **Technical Details**: ExPath는 세 가지 주요 구성 요소로 이루어져 있습니다: (1) 대형 단백질 언어 모델(pLM)이 아미노산 서열을 그래프에 인코딩하여 전통적인 데이터 처리 문제를 극복합니다; (2) PathMamba라는 하이브리드 아키텍처를 사용하여 그래프 신경망(GNNs)과 상태 공간 시퀀스 모델링(Mamba)을 결합, 로컬 상호작용과 전반적인 경로 수준 종속성을 동시에 학습합니다; (3) PathExplainer는 훈련 가능한 경로 마스크를 통해 기능적으로 중요한 노드와 엣지를 식별하는 서브그래프 학습 모듈입니다.

- **Performance Highlights**: ExPath의 실험 결과는 301개의 바이오 네트워크 평가를 포함하며, 이 평가에서 ExPath에 의해 추론된 경로가 생물학적으로 의미 있는 결과를 유지함을 보여줍니다. 또한, 우리는 기계 학습(ML) 중심의 생물학적 평가 방법과 새로운 지표를 제안하여 실제 실험 데이터가 어떻게 해석될 수 있는지를 명확히 하였습니다. 곧 301개의 큐레이션된 바이오 네트워크 데이터셋도 공개할 예정입니다.



### Patient Trajectory Prediction: Integrating Clinical Notes with Transformers (https://arxiv.org/abs/2502.18009)
- **What's New**: 본 논문에서는 전자 건강 기록(EHR)에서 질병의 경과를 예측하기 위한 접근 방식을 제안하고 있습니다. 기존의 구조화된 데이터에 의존하는 모델에서 벗어나 비구조화된 임상 메모를 통합함으로써 환자의 의료 이력을 더욱 충실히 반영할 수 있습니다. 이로 인해 진단 예측의 정확성이 향상될 것으로 기대됩니다.

- **Technical Details**: 제안된 방법론은 transformer 기반의 딥러닝 모델에 비구조화된 임상 메모를 통합하여 환자의 의료 이력을 풍부하게 표현합니다. 이는 환자의 다음 방문(N+1)의 진단을 예측하기 위한 임베딩(embedding) 생성을 포함하고, 이를 통해 예측의 오류를 줄입니다. MIMIC-IV 데이터셋을 활용하여 진행된 실험에서, 이 접근 방식은 기존의 데이터에만 의존하는 전통적인 모델을 초과하는 성능을 보였습니다.

- **Performance Highlights**: MIMIC-IV 데이터셋에서 진행된 실험 결과, 제안된 접근 방식은 환자의 의료 이력을 포괄적으로 반영함으로써 진단 예측의 정확성을 개선하는 데 성공했습니다. 이는 자동화된 의료 예측 시스템의 신뢰성을 높일 수 있는 잠재력을 보여줍니다. 또한, 다양한 환자 그룹과 복잡한 진단 상황을 다루는 능력을 갖춤으로써 임상적 적용 가능성을 높이고 있습니다.



### Radon-Nikodým Derivative: Re-imagining Anomaly Detection from a Measure Theoretic Perspectiv (https://arxiv.org/abs/2502.18002)
- **What's New**: 이 논문에서는 효과적인 이상 탐지 손실 함수 설계를 위한 핵심 원리로 Radon–Nikodým 정리를 제안합니다. 이를 통해 기존의 손실 함수에 Radon–Nikodým 도함수를 곱함으로써 성능을 크게 향상시킬 수 있음을 보여줍니다. 우리는 이 새로운 손실 함수를 RN-Loss라고 명명하고, PAC 학습 가능성(PAC learnability) 관점에서 이를 검증하였습니다.

- **Technical Details**: RN-Loss는 여러 형태의 데이터 분포(예: Weibull, Log-normal)에 효과적으로 적응하며, 기본적인 손실 함수(예: Binary Cross-Entropy)를 기반으로 계산 효율성을 유지합니다. 이 손실 함수는 클러스터링 알고리즘과 결합될 때도 우수한 성능을 발휘하며, 단순한 모델을 사용하여 정상 데이터로만 학습할 경우에도 이전에 보지 못한 이상치를 탐지하는 데 도움이 됩니다.

- **Performance Highlights**: RN-Loss는 다양한 평가 기준에서 최첨단 방법들을 초월하는 성능을 보여주었습니다. 다변량 데이터셋의 68%에서 F1 점수 증가를 달성하였고, 단일 변량 시계열 데이터 셋에서는 72%에서 최고의 F1 점수를 기록했습니다. 이 알고리즘은 기존의 CBLOF 및 ECBLOF 알고리즘에 비해 향상된 성능을 보이며, 특히 K-Means와 결합했을 때 93%의 단일 변량 데이터셋에서 우수한 성과를 거두었습니다.



### A Perspective on Symbolic Machine Learning in Physical Sciences (https://arxiv.org/abs/2502.17993)
Comments:
          Machine Learning and the Physical Sciences Workshop at NeurIPS 2024

- **What's New**: 이 논문에서는 머신러닝(ML)이 물리 과학에 어떻게 기여할 수 있는지를 논의합니다. 특히, 심볼릭 머신러닝(symbolic ML)과 숫자 기반 머신러닝(numerical ML)의 두 가지 접근 방식의 차이점을 강조하며, 물리학 문제 해결에 있어 두 방식이 동등하게 발전해야 한다고 주장합니다. 전통적인 심볼릭 ML은 해석 가능한 모델을 생성할 수 있지만, 현재는 상대적으로 초기 단계입니다.

- **Technical Details**: 심볼릭 머신러닝은 수치적 모델(즉, numerical ML)과 달리 수학적 기호를 포함하여 해석 가능한 표현을 생성하는 방식입니다. 이러한 방법은 물리 과학의 문제 해결에서 더 많은 해석 가능성을 제공하며, 이는 DNN(Deep Neural Networks)와 같은 블랙박스 모델의 비해 장점이 됩니다. 두 가지 ML 방식은 훈련 과정에서 모델의 파라미터를 수치적으로 최적화하는 데 의존합니다.

- **Performance Highlights**: 현재 심볼릭 ML의 발전은 물리학 연구의 여러 분야에서 나타나고 있으며, 예를 들어 고에너지 물리학에서의 입자 산란에 대한 기호적 회귀(symbolic regression) 응용이 있습니다. 이는 심볼릭 ML이 물리학 문제에 있어서 분석적으로 유용한 모델을 생성할 수 있음을 보여주며, 해석 가능성과 투명성을 높이는 데 기여합니다. 과거 몇 년 사이 심볼릭 ML의 연구가 증가하고 있으나, 여전히 그 응용은 물리학의 다른 분야에 비해 초기 단계에 머물러 있습니다.



### Broadening Discovery through Structural Models: Multimodal Combination of Local and Structural Properties for Predicting Chemical Features (https://arxiv.org/abs/2502.17986)
- **What's New**: 이 연구에서는 원자와 그 주변 환경에 대한 정보를 담고 있는 화학 지문(fingerprints)을 기반으로 하는 새로운 언어 모델을 개발하고자 합니다. 제안하는 방법론은 RoBERTa를 언어 모델로 사용하고 GIN, GCN, Graphormer와 같은 그래프 모델을 결합하여 bimodal architecture를 구성합니다. 이러한 통합된 접근법을 통해 화학 분자의 구조를 보다 효율적으로 예측할 수 있는 가능성을 제시합니다.

- **Technical Details**: Extended-Connectivity fingerprints(ECFP)는 각 분자에 대해 2차원 해시 배열을 할당하여 물리적, 화학적 특성을 암호화합니다. 이러한 표현은 그래프 신경망(Graph Neural Networks)과 자연어 처리(NLP) 모델의 결합을 통해 사용하여, 전통적인 방법에 비해 예측 성능이 향상될 수 있는 기회를 제공합니다. 또한, 이 모델은 화학 환경의 정의에 중요한 역할을 하는 분자 NMR 스펙트럼의 예측과 같은 작업에 유용합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 Quantitative Structure-Activity Relationship(QSAR) 및 핵자기공명(Nuclear Magnetic Resonance) 스펙트럼 예측 등의 특정 작업에서 기존의 방법 대비 상당한 성능 향상을 보여주었습니다. 이러한 성능 증가는 기존 접근 방식을 통해서는 발견하지 못했던 특정 화학적 관계와 물리적 속성들 간의 상관관계를 모델이 효과적으로 학습했음을 시사합니다.



### Generalized Decision Focused Learning under Imprecise Uncertainty--Theoretical Study (https://arxiv.org/abs/2502.17984)
Comments:
          13 pages

- **What's New**: 이 논문은 Decision Focused Learning (DFL) 의 새로운 접근법을 제안하며, 기존의 확률 모델에 의존하지 않고 비확률적 구조를 통해 인지적 불확실성을 모델링하는 방법을 제시합니다. 특히, 개념적으로 '간격(intervals)', '오염(contamination)' 및 '확률 상자(probability boxes)'와 같은 모델을 사용하여 불확실성을 처리하고, 제약 조건 내에서 불확실성을 통합하여 DFL의 유용성을 확장합니다. 이러한 방법론은 불확실한 환경에서도 적절한 의사결정을 지원하는데 중점을 둡니다.

- **Technical Details**: 논문에서는 에러의 주요 원인으로 작용할 수 있는 aleatoric (고유한 무작위성) 및 epistemic (데이터 한정으로 인한) 불확실성을 다루기 위해 세 가지 강력한 손실 함수(loss function)를 제안합니다. 이 손실 함수들은 실제 샘플에 대한 경험적 후회(empirical regret)를 더 나은 방식으로 근사하도록 설계되었으며, 복잡한 문제에 대한 효율성을 높입니다. 또한, 불확실성을 제약 조건 내에 통합함으로써 의사결정 시나리오를 보다 폭넓게 모델링할 수 있습니다.

- **Performance Highlights**: 실험적으로 확인된 바에 따르면, 제안된 강력한 손실 함수들은 대조군 샘플에서 경험적 후회를 개선하며, 추가적인 계산 시간을 필요로 하지 않습니다. 또한, DFL 방식은 다양한 응용 분야에서 전반적인 의사결정 품질과 저항성을 향상시키며, 특히 의료 자원 할당 및 물류 최적화 등의 분야에서 그 유용성을 입증하고 있습니다. 이를 통해 DFL은 전통적인 기계 학습 접근법의 한계를 극복하는 혁신적인 방법으로 자리매김하고 있습니다.



### Provable Performance Bounds for Digital Twin-driven Deep Reinforcement Learning in Wireless Networks: A Novel Digital-Twin Bisimulation Metric (https://arxiv.org/abs/2502.17983)
- **What's New**: 본 논문에서는 wireless network 최적화를 위한 디지털 트윈(digital twin, DT) 기반의 딥 강화 학습(deep reinforcement learning, DRL) 접근 방식에 대한 새로운 메트릭인 DT bisimulation metric (DT-BSM)을 제안합니다. 기존의 방법은 실제 배포 전 DRL 정책의 실제 성능을 보장하지 못하는데, 이는 DT가 신뢰할 수 있는 DRL 교육을 지원하는 능력을 평가하기 위한 보편적 메트릭이 부족하기 때문입니다. 이는 정책 탐색을 위한 안전하고 효율적인 훈련 환경을 제공하는 DT 기반의 새로운 측정 기준을 통해 해결하고자 합니다.

- **Technical Details**: DT-BSM은 Wasserstein 거리(Wasserstein distance)를 기반으로 한 새로운 메트릭으로, DT와 실제 wireless 네트워크 환경에서의 Markov decision process (MDP) 간의 불일치를 정량화합니다. 저자들은 DT에서 훈련된 정책의 성능 하락(regret)이 DT-BSM과 DT 내 MDP의 하위 최적성의 가중합으로 제한된다는 것을 증명합니다. 대규모 wireless 네트워크 시나리오에서 Wasserstein 거리의 계산 복잡성을 피하기 위해 총 변동 거리(total variation distance)를 기반으로 한 수정된 DT-BSM을 소개합니다.

- **Performance Highlights**: 실험 결과는 DT 기반 DRL의 이론적 성과 경계를 입증하는 최초의 사례로, DT-BSM 계산을 위한 MDP의 정확한 전이 확률을 얻는 도전과제를 해결하기 위해 통계적 샘플링에 기반한 경험적 DT-BSM 방법을 제안합니다. 경험적 DT-BSM이 항상 이론적인 값으로 수렴하며, 요구되는 샘플 크기와 목표 근사 정확도 수준 간의 관계를 정량적으로 establishment할 수 있음을 보여줍니다. 이러한 결과들은 DT 기반의 DRL의 실제 성능과 신뢰성을 확보하는 데 기여할 것입니다.



### XGBoost-Based Prediction of ICU Mortality in Sepsis-Associated Acute Kidney Injury Patients Using MIMIC-IV Database with Validation from eICU Databas (https://arxiv.org/abs/2502.17978)
- **What's New**: 이 연구에서는 Sepsis-Associated Acute Kidney Injury (SA-AKI) 환자의 집중 치료실(ICU) 사망률을 예측하기 위해 Medical Information Mart for Intensive Care IV (MIMIC-IV) 데이터베이스를 활용하여 기계 학습 모델을 개발하였습니다. 9,474명의 SA-AKI 환자로부터 24개의 예측 변수로 축소된 핵심 특징을 선별하였으며, XGBoost 모델을 활용하여 입원 중 사망률을 예측함으로써 기존 연구의 한계를 보완하고자 하였습니다.

- **Technical Details**: 연구에서는 MIMIC-IV에서 제공하는 비식별화된 전자 건강 기록을 활용하였으며, 연구 기간은 2008년부터 2019년까지입니다. 데이터셋은 환자의 인구 통계, 생체 신호, 실험실 결과, 약물 기록 등의 다양한 임상 데이터를 포함하고 있으며, HIPAA(Health Insurance Portability and Accountability Act) 규정을 준수하여 엄격히 관리되었습니다. 또한, eICU Collaborative Research Database의 데이터를 포함하여 외부 검증이 가능한 모델의 일반화를 도모하였습니다.

- **Performance Highlights**: 제안된 XGBoost 모델은 내부 Receiver Operating Characteristic(AUROC) 곡선에서 0.878의 성과를 거두었으며, 이는 임상적 결정 지원을 위한 높은 정확성과 해석 가능성을 제공합니다. SHAP 기법을 통해 중요한 사망 예측 인자로 SOFA 점수, 혈청 젖산, 호흡률 등이 identified 되었으며, LIME 분석에서는 혈청 젖산과 APACHE II 점수, 총 소변량 등의 특징이 강조되었습니다. 이 모델은 SA-AKI 환자에 대한 사망 예측을 개선하여 조기 환자 식별을 통해 임상적 의사 결정을 강화할 수 있습니다.



### Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation (https://arxiv.org/abs/2502.17972)
- **What's New**: 본 논문에서는 Tensor Network Purification (TNP)이라는 새로운 모델 프리 모델을 제안합니다. TNP는 특수 설계된 텐서 네트워크 분해 알고리즘에 기반하여 적대적 예시를 정화하는 방식으로, 사전 훈련된 생성 모델이나 특정 데이터셋에 의존하지 않습니다. 이로 인해 다양한 적대적 시나리오에서 강력한 강인성을 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: TNP는 고전적 분해 방법에서의 가우시안 노이즈 가정 완화를 중심으로 하며, 적대적 왜곡의 미지의 분포를 다루는 데 중점을 둡니다. 이 기법은 형식적으로 저랭크 표현 대신, 적대적 예시에서 관찰되지 않은 청정 예시를 재구성하는 것을 목표로 합니다. 이를 위해 점진적 다운샘플링과 새로운 적대적 최적화 목표를 도입하여 재구성 오류를 최소화하면서도 적대적 왜곡이 복원되지 않도록 합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, ImageNet에 대한 광범위한 실험 결과, TNP는 다양한 공격 유형과 작업에 걸쳐 효과적으로 일반화된 성능을 보였습니다. 특히 TNP는 다양한 노орм 위협에 대해 AT와 비교하여 평균 강인성 정확도가 26.45% 향상되었고, AP 방식에 비해 여러 공격에서 9.39%, 다양한 데이터셋에서도 6.47% 향상되었습니다. 이러한 결과는 TNP의 뛰어난 효과와 잠재력을 보여줍니다.



### LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena (https://arxiv.org/abs/2502.17967)
- **What's New**: 이번 연구에서는 Agent Trading Arena라는 새로운 가상 환경을 소개하여 대규모 언어 모델(LLMs)의 수치적 추론 능력을 향상시키기 위한 방법을 제시합니다. 이 환경은 제로섬 게임(Zero-Sum Game) 방식으로 설계되어 에이전트들이 주식 포트폴리오에 투자하여 복잡한 경제 시스템을 시뮬레이션하는데, 이는 실제 상황에서의 문제 해결 능력을 더 잘 평가할 수 있게 해줍니다. 또한, 시각적 데이터를 사용할 때 LLMs의 수치적 추론 능력이 더 우수하다는 점을 강조합니다.

- **Technical Details**: Agent Trading Arena는 자산 가격 결정이 에이전트 간의 상호작용에 기반하여 자체적으로 이루어지도록 설계되었습니다. 외부 지식의 영향을 줄이기 위해 가격은 입찰-매도 시스템에 따라 결정되며, 모든 에이전트의 활동은 단기적인 최적 전략에 의존하지 않고 자율적으로 동적으로 변화합니다. 이러한 시스템은 에이전트가 시각적 표현을 통해 복잡한 데이터 관계를 추론하고 보다 전략적으로 의사 결정을 내릴 수 있도록 돕는 반사 모듈(Reflection Module)을 포함합니다.

- **Performance Highlights**: 실험 결과, LLMs는 텍스트 형식의 수치적 데이터에서 성능이 떨어지는 반면, 시각적으로 표현된 데이터에서는 월등한 성과를 보였습니다. 시각적 데이터로 처리된 경우, LLMs는 실험에서 높은 수익률(Return Rate)을 보여주어 구조화된 시각적 표현의 이점을 입증했습니다. 또한, NASDAQ 주식 데이터셋을 사용하여 LLM의 시각적 데이터 처리 능력이 텍스트 기반 데이터 처리보다 뛰어나다는 점을 확인했습니다.



### Late Breaking Results: The Art of Beating the Odds with Predictor-Guided Random Design Space Exploration (https://arxiv.org/abs/2502.17936)
Comments:
          2 pages, 3 figures, conference, this research manuscript is currently under review for publication in an IEEE conference

- **What's New**: 이 논문은 MIG 기반 합성에서 랜덤 탐색을 통해 조합 디지털 회로를 개선하는 혁신적인 방법을 소개합니다. 이 방법은 다음 상태 예측(next-state prediction)과 반복 선택(iterative selection)을 통합하여 합성 프로세스를 크게 가속화하며, 기존의 최첨단 기술 대비 합성 속도를 최대 14배 향상시키고 MIG 최소화에서 20.94%의 개선을 달성합니다.

- **Technical Details**: 제안된 방법은 고유한 세 가지 모듈로 구성되어 있으며, Predictive Module(PrM)과 Policy Module(PoM)는 레시피(recipe) 선택을 최적화합니다. Inter-Iteration Selection Module(IISM)은 최적의 회로를 바탕으로 합성을 주기적으로 재시작합니다. 다양한 예측 모델을 평가하며, 이 과정에서 예측 정확도가 항상 결과의 합성 품질이나 속도 향상으로 이어지지 않는다는 점을 관찰했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 20.94%의 MIG 최소화 개선과 14배의 합성 속도 향상을 보여줍니다. 특정 회로를 대상으로 한 여러 실험을 통해, IISM 없이 두 개의 레시피를 선택하는 Two-Step Ahead Predictions(2SA) 방식이 1SA보다 월등한 속도를 달성하는 결과를 확인했습니다. 전체적으로, 높은 QoR(Quality of Results)를 유지하면서도 회로 최적화 속도를 크게 향상시킬 가능성이 확인되었습니다.



### Integrating Boosted learning with Differential Evolution (DE) Optimizer: A Prediction of Groundwater Quality Risk Assessment in Odisha (https://arxiv.org/abs/2502.17929)
Comments:
          9 Figures (8 figs in paper and one additional graphical abstract), 9 Tables

- **What's New**: 본 연구에서는 머신 러닝 기반의 예측 모델인 LCBoost Fusion을 개발하여 지하수의 품질 지수(Groundwater Quality Index, GWQI)를 평가하고 주요 오염 물질을 식별하였습니다. LCBoost Fusion은 CatBoost와 LightGBM을 결합한 하이브리드 모델로, Differential Evolution (DE) 최적화를 통해 정확성을 높였습니다. 이 모델의 도입은 지속 가능한 지하수 관리에 기여할 것으로 기대됩니다.

- **Technical Details**: 모델의 성능 평가는 여러 단계를 거쳐 이루어졌으며, RMSE는 0.6829, MSE는 0.5102, MAE는 0.3147, R² 스코어는 0.9809로 각각 측정되었습니다. 특성 중요도 분석을 통해 칼륨(Potassium, K), 플루오라이드(Fluoride, F), 그리고 총 경도(Total Hardness, TH)가 지하수 오염에 가장 큰 영향을 미치는 요소로 평가되었습니다. 이러한 발전은 기계 학습이 지하수 품질 위험을 평가하는 데 어떻게 활용될 수 있는지를 보여줍니다.

- **Performance Highlights**: LCBoost Fusion 모델은 기존의 개별 모델인 CatBoost와 LightGBM보다 높은 예측 정확도를 기록하여, 지하수 질 향상에 기여할 수 있는 가능성을 보여줍니다. 이 연구에서는 오디샤 지역의 지하수를 대상으로 하였으며, 결과는 환경 기관과 정책 입안자들에게 목표 지역의 지속 가능한 관리를 위한 기초 자료로 활용될 수 있습니다. 향후 연구는 원격 탐지 데이터와 상호작용하는 의사결정 시스템을 개발하는 데 중점을 둘 것입니다.



### Techniques for Enhancing Memory Capacity of Reservoir Computing (https://arxiv.org/abs/2502.17923)
- **What's New**: 이번 논문에서는 Reservoir Computing(RC) 모형의 메모리 용량을 향상시키기 위해 네트워크 구성을 수정하는 새로운 방법들을 제안합니다. 이 방법들은 기존의 RC 모형의 구조를 변경하지 않고도 메모리 용량을 높일 수 있도록 설계되었습니다. 특히, Delay, Pass through 및 Clustering 방법을 통해 입력 데이터에 대한 기억력을 개선하고, 이를 통해 다양한 시간 시퀀스 데이터 처리 작업에서의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 세 가지로 나뉘며, 각 방법의 메모리 용량은 입력 신호를 얼마나 오랫동안 저장할 수 있는지를 나타내는 지표인 MCMC로 정의됩니다. Delay 방법은 지정된 지연 스텝 수만큼의 지연 노드 체인을 추가하여 과거 입력을 유지합니다. Pass through 방법은 입력 값들을 직접 출력 레이어로 전달하고, Clustering 방법은 입력 및 reservoir 노드를 여러 부분으로 나누어 출력 레이어에서 통합합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법들은 echo state network(ESN)과 chaotic Boltzmann machine(CBM)-RC와 같은 대표적인 RC 모델을 통해 평가되었습니다. NARMA 과제를 통해 메모리 용량과 비선형성 사이의 무역 관계를 분석하고, 정보 처리 용량(IPC)을 측정하여 이들 방법의 유효성을 입증했습니다. 결과적으로, 각 방법이 어떻게 서로 보완되어 메모리 용량을 최적화하는지 확인할 수 있었습니다.



### C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models (https://arxiv.org/abs/2502.17920)
- **What's New**: C-LoRA(Continual Low-Rank Adaptation)는 전통적인 LoRA 사용법을 확장하여 지속적인 학습을 위한 새로운 접근 방식을 제안합니다. 이 방법은 learnable routing matrix를 소개하여 다양한 작업 간의 매개변수 업데이트를 동적으로 관리합니다. C-LoRA는 과거의 지식을 보존하고 방해를 최소화하면서 학습의 효율성을 높이기 위한 방법으로 작동합니다.

- **Technical Details**: C-LoRA는 기존의 LoRA 구조에서 발생하는 한계점을 해결하기 위해 고안된 방법입니다. 이는 orthogonality constraints를 적용하여 작업 간의 간섭을 줄임으로써, 지속적인 학습 환경에서도 효율성을 보장합니다. 또한, 이 방법은 다양한 작업에 대해 공통의 low-rank subspaces를 활용하여 메모리 및 컴퓨팅 자원의 소모를 최소화하도록 설계되었습니다.

- **Performance Highlights**: C-LoRA는 여러 벤치마크에서 기존의 최첨단 지속적 학습 방법들보다 우수한 성능을 보여주었습니다. C-LoRA의 성능 평가는 작업 적응을 위한 통합 접근 방식을 가능하게 하여, 순차적인 학습 시나리오에서 확장성과 매개변수 효율성을 동시에 달성할 수 있음을 나타냅니다. 추가적으로, 이 방법은 지식의 보존 및 전이와 관련된 이론적 통찰력을 제공합니다.



### AirCast: Improving Air Pollution Forecasting Through Multi-Variable Data Alignmen (https://arxiv.org/abs/2502.17919)
- **What's New**: 이번 논문에서는 대기 오염 예측을 위한 새로운 모델인 AirCast를 소개합니다. 이 모델은 날씨 및 공기 질 변수들을 통합하여 대기 조건과 오염 물질 농도를 동시에 예측할 수 있는 다중 작업 헤드 아키텍처를 사용합니다. 제안된 Frequency-weighted Mean Absolute Error (fMAE) 손실 함수를 통해 드문 오염 사건 예측의 어려움을 해결하고, 특정 변수의 선택을 통해 예측 정확도를 개선합니다.

- **Technical Details**: AirCast는 Vision Transformer (ViT) 아키텍처를 기반으로 하며, WeatherBench와 Copernicus Atmosphere Monitoring Service (CAMS) EAC4 데이터셋에서 획득한 날씨 및 공기 질 변수를 통합하여 대기 오염 예측을 수행합니다. 이 모델은 대량의 변수들을 효과적으로 처리하기 위해 변수 토큰화(variable tokenization) 및 변수 집계(variable aggregation) 모듈을 포함하고 있으며, 다중 작업 헤드 아키텍처를 통해 대기 및 오염 변수의 동시 예측이 가능합니다. fMAE 손실 함수는 오염 물질의 무거운 분포를 다루기 위해 설계되었습니다.

- **Performance Highlights**: 이 모델은 중동 및 북아프리카(MENA) 지역에서의 높은 PM 농도 예측에 집중하고 있으며, 해당 지역의 대기 질 저하를 방지하기 위한 연구입니다. AirCast는 기후 변화와 산업 발달로 악화된 대기 오염 문제를 해결하기 위한 혁신적인 접근 방식을 제공하며, 기존의 예측 방법보다 높은 정확도를 목표로 합니다. 이를 통해 정책 결정 및 오염 저감 전략 수립에 기여할 것으로 기대됩니다.



### Batch normalization does not improve initialization (https://arxiv.org/abs/2502.17913)
- **What's New**: 이번 논문에서는 배치 정규화(Batch Normalization, BN)가 초기화(initialization)를 개선하지 않는다는 반례(counterexample)를 제시합니다. 기존 연구에서는 BN이 초기화에 유리하다고 주장했으나, 저자들은 이러한 주장에 반하는 예를 통해 논의의 필요성을 강조합니다. 배치 정규화의 기초적인 이해와 그 작용 메커니즘에 대한 의문을 제기하고 있습니다.

- **Technical Details**: 신경망(neural networks)은 많은 매개변수(parameter)를 갖는 함수로, 이들은 훈련 초기에 무작위로 초기화되고 손실 함수(loss function)를 최소화하여 학습됩니다. 배치 정규화는 이러한 신경망 훈련을 개선하기 위한 정규화 기법입니다. 네트워크는 여러 층(layers)으로 구성되어 있으며, 각 층은 뉴런(neurons)과 활성화 함수(activation functions)를 포함합니다.

- **Performance Highlights**: 배치 정규화를 통해 훈련 속도가 빨라지고 안정성이 증가한다고 많은 응용 분야에서 증명되었습니다. 특히 데이터 과학자들 사이에서 BN의 효과는 기정사실로 받아들여지고 있으며, 이로 인해 심층 신경망(deep neural networks)의 최적화 최전선에서 광범위하게 활용되고 있습니다. 그러나 이 논문에서는 BN이 초기화에는 이점이 없다는 점이 새롭게 밝혀졌습니다.



### Decoupled Graph Energy-based Model for Node Out-of-Distribution Detection on Heterophilic Graphs (https://arxiv.org/abs/2502.17912)
Comments:
          The first two authors contributed equally to this work; ICLR 2025

- **What's New**: 본 논문은 그래프 학습에서 OOD(Out-of-Distribution) 노드 탐지에 대한 연구를 다루고 있습니다. 이전의 방법들은 i.i.d. 데이터에 기반하여 설계되었으며, 이러한 접근법은 그래프의 의존성 때문에 직접적으로 적용할 수 없습니다. 저자는 Energy-based Models (EBMs)를 MLE(Maximum Likelihood Estimation)를 통해 훈련하고, heterophily 문제를 해결하기 위한 새로운 접근법인 DeGEM을 제안합니다.

- **Technical Details**: DeGEM은 그래프 인코더와 에너지 헤드 두 가지 구성 요소로 분해됩니다. 그래프 인코더는 Graph Contrastive Learning (GCL) 알고리즘으로 훈련되고, 에너지 헤드는 latent space에서 MLE로 훈련되어 노드의 OOD 점수를 제공합니다. 이러한 방법은 MCMC(Markov Chain Monte Carlo) 샘플링을 latent space에서 보다 효율적으로 수행할 수 있게 하여 계산 비용을 줄입니다.

- **Performance Highlights**: DeGEM은 기존의 최첨단 방법들보다 homophilic 및 heterophilic 그래프 모두에서 성능 향상을 보였습니다. 특히 평균 AUROC(Area Under the Receiver Operating Characteristic curve)에서 homophilic 그래프에서 6.71%, heterophilic 그래프에서는 20.29%의 개선을 보였으며, OOD 노출 없이도 성능이 우수함을 입증하였습니다. 기존 방법들이 heterophilic 그래프에서 낮은 성능을 보이는 반면, DeGEM은 이 문제를 해결하여 확실한 성능 우위를 확보했습니다.



### Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs (https://arxiv.org/abs/2502.17900)
- **What's New**: 이 논문에서는 최신 심전도(ECG) 표현 학습 기법인 K-MERL을 제안합니다. K-MERL은 자유 형식의 보고서와 ECG 신호를 정렬하여, 구조화된 지식을 추출하여 자가 감독 학습을 향상시킵니다. 특히, K-MERL은 12리드 ECG의 독특한 공간적 및 시간적 특성을 포착하기 위해 리드 특정(tokenization) 기법을 활용합니다.

- **Technical Details**: K-MERL 프레임워크는 일반 목적의 오픈 소스 대형 언어 모델(LLM)을 활용하여 ECG 보고서에서 심장 관련 엔티티를 추출합니다. 또한, 리드 인식 ECG 인코더와 동적 리드 마스킹(dynamic lead masking) 전략을 설계하여 임의의 리드 입력을 처리하면서 리드 특정 공간 및 시간 패턴을 포착할 수 있습니다. 이러한 접근 방식으로 K-MERL은 다양한 리드 조합에서의 성능을 극대화합니다.

- **Performance Highlights**: K-MERL은 6개의 외부 ECG 데이터셋에 대한 평가에서 제로샷 분류(zero-shot classification) 및 선형 프로빙(linear probing) 작업에서 최고의 성능을 달성하였습니다. 기존 방법들에 비해 평균 16% AUC 개선을 달성하며, 부분 리드를 사용하는 제로샷 분류에서 특히 두드러진 성과를 보입니다.



### Arrhythmia Classification from 12-Lead ECG Signals Using Convolutional and Transformer-Based Deep Learning Models (https://arxiv.org/abs/2502.17887)
Comments:
          34 pages, 17 figures

- **What's New**: 이번 연구는 루마니아의 심혈관 문제를 해결하기 위한 혁신적인 진단 방법을 모색합니다. 비용이 제한된 의료 환경에서 신속하고 효율적인 부정맥(arrhythmia) 진단 기술을 개발하는 데 중점을 두고 있습니다. 또한, 루마니아의 공공 의료 데이터가 부족한 점을 고려하여, 국제 공개 데이터셋을 사용하여 시스템을 훈련시켰습니다.

- **Technical Details**: 부정맥 분류 분야에서 일반적으로 사용되는 여러 데이터셋(PTB-XL, PTB 진단 ECG 데이터베이스, 중국 12-Lead ECG 챌린지 데이터베이스 등)을 결합하여 연구를 수행했습니다. 입력 데이터 처리를 위해 Pan-Tompkins 알고리즘 변형을 사용하였고, 이 알고리즘은 ECG 신호에서 QRS 복합체를 효율적으로 탐지하는데 강력한 성능을 보입니다. 기계 학습(machine learning) 기법으로는 1D CNNs, 2D CNNs, ResNet 및 비전 변환기(ViTs)가 포함되었습니다.

- **Performance Highlights**: 실험에서 GRU 기반의 1D CNN 모델이 93.4%의 최고 정확도를 기록하여 다양한 아키텍처 중 가장 높은 성능을 보였습니다. 또한, ECG 신호를 이미지로 변환하여 2D CNN 모델이 92.16%의 정확도를 달성하며 뛰어난 성과를 나타냈습니다.



### Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning (https://arxiv.org/abs/2502.17874)
- **What's New**: 본 논문은 분자 기계 학습(molecular machine learning)에서 검색(탐색) 증강 생성(retrieval-augmented generation)을 효과적으로 통합하는 방법을 제안하고 있습니다. 특히, 그래프 신경망(graph neural networks)을 활용하여 쿼리 분자(query molecule)와 유사한 분자를 이해하기 위해 구조 정렬(structural alignment)을 개선하는 데 초점을 맞추었습니다. 이를 통해, 기존의 방식들에 비해 분자 구조 예측의 정확성을 크게 향상시키는 것이 가능합니다.

- **Technical Details**: 우리는 MARASON이라는 새로운 모델을 제안하며, 이를 통해 신경 그래프 매칭(neural graph matching)을 기반으로 한 분절(fragmentation) 기반 신경 네트워크(neural network)를 개선합니다. 이 모델은 분자 구조와 그 구조에 대한 특성 정보를 쌍으로 포함한 데이터베이스에서 효율적으로 레퍼런스를 검색하여, 구조-속성 관계 모델을 더욱 향상시킵니다. 우리가 제안한 방법은 노드와 엣지의 유사성을 명확히 모델링하고, 노이즈에 강한 일체형 신경 네트워크(end-to-end neural network)를 통해 애피니티(affinity) 메트릭스를 학습합니다.

- **Performance Highlights**: 실험 결과 MARASON은 28%의 top-1 정확도를 달성하며, 비검색 기본 상태인 19%의 정확도를 획기적으로 개선한 성과를 보여주었습니다. 또한 MARASON은 기존의 단순 검색 증강 생성 방법과 전통적인 그래프 매칭 기법보다 높은 성능을 발휘했습니다. 이는 분자 발견(molecular discovery)을 가속화하는 데 있어 새로운 가능성을 제시하며, 대규모 질량 스펙트럼 데이터베이스에 대한 제작 및 주석 과정을 혁신할 수 있음을 나타냅니다.



### EEGM2: An Efficient Mamba-2-Based Self-Supervised Framework for Long-Sequence EEG Modeling (https://arxiv.org/abs/2502.17873)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 논문에서는 EEGM2라는 새로운 자기 감독 프레임워크를 제안합니다. 이 프레임워크는 구조적 상태 공간의 이중성(SSD)에 기반하여 전통적인 Transformer 모델의 한계를 극복합니다. EEGM2는 세 가지 주요 혁신을 도입하여 EEG 신호 모델링의 성능을 향상시키고 메모리 효율성을 높이는 데 기여합니다.

- **Technical Details**: EEGM2는 Mamba-2 블록을 기반으로 하는 인코더-디코더 아키텍처로서, EEG 신호의 로컬 및 글로벌 특징을 효율적으로 캡처합니다. 이 모델은 다중 분기 수용 필드 입력 임베딩 전략을 사용하여 다양한 길이의 EEG 시퀀스에 대한 일반화 및 안정성을 개선하고, 시간적 정보의 모델링을 위한 매개 모듈을 포함합니다. 또한, spatiotemporal-aware 손실 함수를 통해 노이즈에 대한 강인성을 높이고 스펙트럼 정보를 보존합니다.

- **Performance Highlights**: 실험 결과, EEGM2는 기존의 사전 학습된 모델보다 18배 적은 크기와 선형 시간 복잡도로 새로운 작업에서 성능을 크게 향상시켰습니다. EEGM2는 6개의 EEG 데이터셋에서 최첨단의 교차 도메인 정확도를 달성하며, 자원 제한이 있는 BCI 장치에서의 효율적인 배치 솔루션으로서의 가능성을 보여줍니다.



### Contrastive Learning with Nasty Nois (https://arxiv.org/abs/2502.17872)
- **What's New**: 이 논문은 adversarial noise 아래에서 contrastive learning의 이론적 한계를 분석합니다. 연구는 PAC learning(Probably Approximately Correct) 및 VC-dimension(컬럼의 차원) 분석을 사용하여 adversarial 환경에서의 샘플 복잡성에 대한 하한 및 상한을 설정합니다. 또한, l2-distance 함수를 기반으로 한 데이터 의존적인 샘플 복잡성 경계를 도출합니다.

- **Technical Details**: contrastive learning에서는 
ho: V × V → ℝ와 같은 거리를 사용하는데, 이 거리는 d차원의 representation 함수 f에 의해 정의됩니다. 학습 작업은 boolean classifier의 가설 클래스에 의해 정의되며, PAC 모델을 사용하여 라벨이 지정된 예제 (x, y+, z−)에 접근합니다. 이를 통해 distance function ρ에 대한 가설 hρ를 정의하고, noise에 강한 학습 알고리즘의 존재를 입증합니다.

- **Performance Highlights**: П 이 연구는 adversarial noise의 존재 하에서 contrastive learning의 샘플 복잡성에 대한 최적 한계를 탐구합니다. 이는 adversary가 샘플을 수정 및 교체하여 노이즈 뷰 또는 노이즈 레이블을 도입하는 환경에서 효과적으로 동작하지 않게 하는 메커니즘을 설명합니다. 논문의 결과는 contrastive learning을 향상시키기 위한 전략적 방향을 제시합니다.



### Mitigating Attrition: Data-Driven Approach Using Machine Learning and Data Engineering (https://arxiv.org/abs/2502.17865)
Comments:
          7 pages

- **What's New**: 이 논문은 직원 이탈(employee attrition)을 감소시키기 위한 새로운 데이터 기반 접근 방식을 제안합니다. 머신 러닝(machine learning)과 데이터 엔지니어링(data engineering) 기술을 통합하여 인사 시스템으로부터의 데이터를 활용합니다. 이 연구는 이탈에 영향을 미치는 다양한 요인을 포괄적으로 수집하기 위해 고급 피처 엔지니어링(feature engineering)을 활용합니다.

- **Technical Details**: 제안된 프레임워크는 불균형 데이터셋(imbalanced datasets), 범주형 데이터(categorical data) 처리 및 모델 해석(model interpretation)과 같은 도전 과제를 해결하는 강력한 모델링 접근 방식을 설명합니다. 연구 방법론에는 훈련 및 테스트 전략, 기준 모델(baseline model) 구축, 보정된 예측 모델(calibrated predictive models) 개발에 대한 면밀한 고려가 포함됩니다. 또한 SHAP 값(SHAP values)과 같은 기술을 통한 모델 해석의 중요성이 강조됩니다.

- **Performance Highlights**: 이 접근 방식을 통해 조직은 이탈 위험을 능동적으로 식별하고 목표에 맞춘 유지 전략(targeted retention strategies)을 개발할 수 있습니다. 결과적으로 이는 직원의 이탈을 효과적으로 줄이는데 기여할 수 있습니다. 알고리즘 선택, 하이퍼파라미터 튜닝(hyperparameter tuning), 확률 보정(probability calibration) 등의 주요 설계 선택 사항에 대해서도 논의합니다.



### Armada: Memory-Efficient Distributed Training of Large-Scale Graph Neural Networks (https://arxiv.org/abs/2502.17846)
- **What's New**: 이 논문에서는 대규모 분산 GNN(Graph Neural Networks) 훈련을 위한 새로운 시스템인 Armada를 소개합니다. GNN 훈련의 병목 현상인 파티셔닝 문제를 해결하기 위해, GREM(Greedy plus Refinement for Edge-cut Minimization)이라는 새로운 min-edge-cut 파티셔닝 알고리즘을 개발하였습니다. GREM은 기존의 스트리밍 알고리즘의 성능을 개선하여 기하급수적으로 큰 그래프에 대해서도 효율적인 파티셔닝을 가능케 합니다.

- **Technical Details**: Armada는 CPU와 GPU 자원을 분리하여 GNN 훈련을 메모리 효율적이고 비용 효과적으로 할 수 있도록 설계된 분산 아키텍처입니다. 이 시스템의 핵심은 GREM 알고리즘이 이전의 정점 할당을 초기 선택 후에 얼리지 않고 지속적으로 수정할 수 있다는 점입니다. 이와 함께, GREM은 학습 가능한 그래프의 통계 정보를 활용하여 파티셔닝을 개선합니다.

- **Performance Highlights**: 실험 결과, Armada는 메모리 사용량을 8배 줄일 수 있으며, 훈련 속도를 4.5배 개선할 수 있음을 보여 줍니다. GREM의 상대적 성능은 기존의 METIS와 유사한 레벨의 엣지 컷을 유지하면서도 8-65배 적은 메모리 사용과 8-46배 빠른 실행 시간을 제공합니다. 이러한 성과는 분산 GNN 훈련에서 자원의 최적화와 효율성을 크게 향상시킬 것으로 기대됩니다.



### LeanKAN: A Parameter-Lean Kolmogorov-Arnold Network Layer with Improved Memory Efficiency and Convergence Behavior (https://arxiv.org/abs/2502.17844)
Comments:
          15 pages, 5 figures, and 1 table

- **What's New**: 최근 제안된 Kolmogorov-Arnold 네트워크(KAN)는 데이터 기반 모델링을 위한 유망한 대안으로, 기존의 다층 퍼셉트론(MLP)에 비해 혁신적인 접근 방식을 제공합니다. 기존 KAN 계층은 단순히 덧셈 연산자만 표현할 수 있었으나, 새롭게 제안된 MultKAN 계층은 덧셈과 곱셈을 결합하여 표현 성능을 진일보시켰습니다. 그러나 MultKAN 계층은 출력 계층에서의 제한된 적용 가능성, 부풀려진 매개변수화, 복잡한 하이퍼파라미터 등의 주요 단점이 존재합니다.

- **Technical Details**: 이 논문에서는 이러한 문제를 해결하기 위해 LeanKAN을 제안합니다. LeanKAN은 MultKAN 및 전통적인 AddKAN 계층을 직접적인 모듈형 대체물로 제공하며, 세 가지 주요 단점을 해결합니다. 첫째, LeanKAN은 출력 계층으로 일반적으로 적용 가능하며, 둘째, 주어진 네트워크 구조에 대한 매개변수 수를 상당히 줄이며, 셋째, 더 작은 하이퍼파라미터 집합을 필요로 합니다.

- **Performance Highlights**: LeanKAN은 전통적인 KAN 학습 문제 뿐만 아니라 KAN Ordinary Differential Equations(KAN-ODE), Deep Operator KANs(DeepOKAN)와 같은 증강 KAN 구조에서도 이점을 제공합니다. 특히, KAN의 간단함과 효율성을 보여주는 여러 시험을 통해 Sparse parameterization(희소 매개변수화)과 Compact structure(압축 구조)가 향상된 표현력과 학습 능력을 제공함을 입증하였습니다. 결과적으로 LeanKAN은 유사한 크기의 MultKAN보다 여러 작업 성능에서 우수함을 나타냈습니다.



### Task-Driven Semantic Quantization and Imitation Learning for Goal-Oriented Communications (https://arxiv.org/abs/2502.17842)
Comments:
          Accepted for publication in 2025 International Conference on Communications (IEEE ICC); 6 pages, 4 figures

- **What's New**: 이 논문에서는 고유한 목적 지향 통신 프레임워크인 Goal-Oriented Semantic Variational Autoencoder (GOS-VAE)를 제안합니다. 기존의 비트 단위 데이터 전송에서 의미 정보 전송으로의 패러다임 전환을 통해 대역폭을 줄이는 특징이 있습니다. GOS-VAE는 다운스트림 작업을 위한 중요한 의미 메시지를 추출하는 데 집중하여, 수신 측에서보다 효율적인 데이터 운반을 지원합니다.

- **Technical Details**: GOS-VAE는 Vector Quantized Variational Autoencoder (VQ-VAE) 구조를 활용하여 통신 장치의 기능을 최적화합니다. 이 모델은 저전력 송신기에서 고급 서버로 복잡한 작업 모델을 배치하여 계산 효율성을 높입니다. 또한, 모방 학습(imitation learning)을 활용해 의미 기반 데이터 복원을 수행하고, 다양한 채널 모델과 통신 프로토콜에 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과는 GOS-VAE가 목표 지향적인 의미를 잘 표현하고 대역폭 효율성을 갖춘다는 것을 보여주었습니다. 세심한 네트워크 깊이 조정과 맞춤형 CNN 구조를 통해 신호 복원 성능이 향상되며, 대역폭 소비가 감소합니다. 이는 자율주행과 같은 AI 기반 애플리케이션에 매우 유용한 기능을 제공합니다.



### MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks (https://arxiv.org/abs/2502.17832)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서 제안하는 MM-PoisonRAG는 멀티모달 RAG 프레임워크에 대한 최초의 지식 오염 공격 프레임워크입니다. 공격자는 사실과 관련 없는 지식을 외부 지식 기반(KB)에 주입하여 모델이 잘못된 답변을 생성하도록 유도합니다. 두 가지 공격 전략인 Localized Poisoning Attack (LPA)와 Globalized Poisoning Attack (GPA)을 통해 특정 쿼리나 전반적인 조건에서 모델의 출력을 조작할 수 있습니다.

- **Technical Details**: MM-PoisonRAG는 LPA와 GPA 두 가지 공격 시나리오를 가지고 있으며, LPA는 쿼리와 관련된 잘못된 정보를 주입하여 특정 쿼리에 대한 조작을 목표로 합니다. 반면 GPA는 모든 쿼리에서 비관련 지식을 삽입하여 모델의 생성을 완전히 교란시킵니다. 이 연구에서는 각 공격이 모델의 응답 생성에 미치는 영향을 여러 작업과 설정에서 평가하였습니다.

- **Performance Highlights**: LPA는 최대 56%의 성공률로 공격자가 정의한 답변을 생성하는 데 성공했습니다. 반면 GPA는 단 한번의 비관련 지식 주입으로 모델의 정확도를 0%로 떨어뜨리는 결과를 보였습니다. 이러한 결과는 멀티모달 RAG 프레임워크를 보호하기 위해 강력한 방어책의 필요성을 강조합니다.



### A General Framework to Enhance Fine-tuning-based LLM Unlearning (https://arxiv.org/abs/2502.17823)
- **What's New**: 본 논문에서는 기존의 fine-tuning 기반 unlearning 방법을 개선하기 위해 Gated Representation UNlearning (GRUN)이라는 새로운 프레임워크를 제안합니다. GRUN은 target data를 구분하고 suppress하는 두 가지 모듈로 구성되어 있어 unlearning 성능을 크게 향상시킵니다. 또한, 이 시스템은 기존 방법보다 모델의 유틸리티(utility)를 유지하는 데 중점을 두고 있습니다.

- **Technical Details**: GRUN은 soft gate function과 Representation Fine-tuning (ReFT) 모듈로 구성됩니다. soft gate function은 target data의 구분을 지원하며, ReFT는 모델의 파라미터를 조정하는 대신 representation을 fine-tuning하여 성능 저하를 방지합니다. 이 논문에서는 GRUN의 효율성과 효과성을 입증하기 위해 다양한 실험을 통해 95% 이상의 훈련 시간 단축과 함께 near-perfect unlearning을 달성했습니다.

- **Performance Highlights**: GRUN은 LLM의 크기보다 적은 0.05% 이하의 추가 모듈을 요구하며, 다양한 모델(Llama 3.1, Mistral 등)과 데이터셋(TOFU, WMDP 등)에서 검증되었습니다. 이 방법은 효율적이며 sequential unlearning에 유망한 결과를 보이며, 기존 fine-tuning 방식에 일반적으로 적용 가능한 솔루션으로서 긍정적인 평가를 받고 있습니다.



### PVBF: A Framework for Mitigating Parameter Variation Imbalance in Online Continual Learning (https://arxiv.org/abs/2502.17794)
Comments:
          27 pages, 11 figures

- **What's New**: 본 논문은 Online Continual Learning (OCL)에서 경험 재생(experience replay) 기반의 방법들이 직면하고 있는 예측 편향(prediction bias) 문제를 해결하기 위해 새로운 접근 방식을 제안합니다. 특히 파라미터 변동 불균형(parameter variation imbalance)이 예측 편향에 주는 영향을 강조하며, 두 가지 유형의 불균형(상관 유도 불균형(correlation-induced imbalance) 및 층별 불균형(layer-wise imbalance)을 구체적으로 살펴봅니다. 이러한 문제를 해결하기 위해 Parameter Variation Balancing Framework (PVBF)를 개발하였습니다.

- **Technical Details**: PVBF는 주어진 두 가지 불균형 문제를 해결하기 위해 세 가지 주요 구성 요소를 포함하고 있습니다. 첫째, Parameter Correlation Calculate (ParamCC)를 통해 이전 작업과 각 파라미터 간의 상관관계를 정량화합니다. 둘째, Encourage and Consolidate (E&C) 전략을 통해 학습 중 전반적인 파라미터의 기울기를 조정하여 잘못된 방향으로의 업데이트를 방지합니다. 셋째, Dual-layer Copy Weights with Reinit (D-CWR) 방법을 사용하여 출력 레이어의 파라미터 업데이트 속도를 조절합니다.

- **Performance Highlights**: 실험 결과 PVBF는 기존 경험 재생 기반 방법에 비해 평균 31%-47%의 정확도 향상을 보여주었으며, 특히 MiniImageNet 데이터셋에서 500개의 재생 샘플만 사용해 IID 방법의 97.5% 정확도를 달성했습니다. 이는 PVBF가 단순히 예측 편향을 줄이는 것뿐만 아니라 OCL 성능을 현저히 향상시킨다는 것을 나타냅니다. 또한 PVBF는 고전적인 오프라인 CL 시나리오에서도 일반적인 성능 향상을 보여줍니다.



### On-device edge learning for IoT data streams: a survey (https://arxiv.org/abs/2502.17788)
- **What's New**: 이 문헌 리뷰는 스마트 환경에서의 분류 작업을 위한 신경망(Neural Networks) 및 결정 트리(Decision Trees) 기반의 온디바이스 학습 방법에 대한 최신 내용을 탐색합니다. 데이터 아키텍처(배치 vs. 스트림) 및 네트워크 용량(클라우드 vs. 엣지)와 같은 주요 제약 조건이 TinyML 알고리즘 설계에 미치는 영향을 강조합니다. 논문에서는 자원 제약이 있는 엣지 디바이스에서의 딥러닝 배포에 대한 도전 과제를 자세히 설명하며, 특히 재앙적 망각(calamitous forgetting)과 IoT 테이블 데이터 처리의 어려움을 다룹니다.

- **Technical Details**: 온디바이스 학습 솔루션 개발을 위한 요구 사항을 재정의하며, 데이터 처리 방법의 차이(배치 또는 스트림 아키텍처)와 데이터가 생성되는 위치에 따른 네트워크 용량이 AI 알고리즘 설계에 미치는 영향을 논의합니다. IoT 환경에서는 정적 데이터셋에 대한 배치 처리, 이벤트 허브를 통한 스트림 데이터 처리, 그리고 배치와 스트림 데이터를 모두 처리할 수 있는 아키텍처가 적용됩니다. 특히, 배치 아키텍처는 즉각적인 응답이 필요하지 않은 애플리케이션에서 주로 사용되며, 대량의 데이터를 효율적으로 처리하기 위한 ETL 파이프라인과 데이터 레이크를 활용합니다.

- **Performance Highlights**: 결정 트리는 온디바이스 학습에 대해 메모리 효율적인 방법으로 알려져 있으나, 복잡한 패턴과 개념 변화에 적응하기 위해서는 동적 조정이 필요합니다. 동시에, 딥러닝 알고리즘은 더 많은 데이터를 학습하면서도 저조한 데이터 효율성과 느린 수렴 속도로 인해 도전에 직면해 있습니다. 따라서 이 논문은 메모리 최적화 및 IoT 데이터 스트림 처리 관련 과제를 다루며, 조직화된 데이터 스트림을 통해 기존 모델의 조건과 과거의 지식을 바탕으로 새로운 요구 사항에 대한 연구 질문에 답합니다.



### MuCoS: Efficient Drug-Target Prediction through Multi-Context-Aware Sampling (https://arxiv.org/abs/2502.17784)
- **What's New**: 이번 연구에서는 MuCoS(Multi-Context-Aware Sampling)라는 새로운 방법을 제안하여 복잡한 생물학적 엔터티 간의 관계를 더 정확하게 예측할 수 있도록 하였습니다. MuCoS는 드러나지 않은 관계를 다룰 수 있도록 최적화된 이웃 샘플을 활용하고, BERT(Bidirectional Encoder Representations from Transformers)와의 통합을 통해 향상된 맥락적 임베딩을 제공합니다. 기존 모델에 비해 MuCoS는 13%의 MRR(Mean Reciprocal Rank) 향상 등을 보여주어 전체적으로 예측 성능이 개선되었습니다.

- **Technical Details**: MuCoS는 인접 엔터티로부터 맥락 정보를 수집하여 관계 예측을 수행하는 기술입니다. 이 방법은 헤드 엔티티와 관계로부터 최적화된 맥락 정보를 추출하고 이를 BERT 모델로 처리하여 예측을 수행합니다. 이 방법은 또한 주변 임베딩의 복잡성을 줄이고, 음성 샘플링을 제거하여 훈련 효율성 및 강 robustness를 향상시킵니다.

- **Performance Highlights**: 실험 결과, MuCoS는 KEGG50k 생물의학 데이터셋에서 기존 모델에 비해 MRR 13%, Hits@1 7%, Hits@3 4% 및 Hits@10 18% 향상을 보였습니다. 드러나지 않은 관계와 엔티티에 대한 예측 성능 또한 향상되어, 복잡한 DTI(Drug-Target Interaction) 예측에서 효과적인 방법으로 입증되었습니다.



### Adaptive Nesterov Accelerated Distributional Deep Hedging for Efficient Volatility Risk Managemen (https://arxiv.org/abs/2502.17777)
- **What's New**: 본 연구에서는 전통적인 Vega 헤징 전략의 한계를 극복하기 위해 새로운 동적 Vega 헤징 프레임워크인 Adaptive Nesterov Accelerated Distributional Deep Hedging (ANADDH)를 소개합니다. 기존의 규칙 기반 모델에 의존하는 방법들과 달리, ANADDH는 적응형 Nesterov 가속화(Adaptive Nesterov Acceleration)를 기반으로 한 분포적 강화 학습(distributional reinforcement learning)을 결합하여 시장 변동성에 더 잘 적응할 수 있도록 합니다.

- **Technical Details**: ANADDH는 비즈니스 환경의 복잡성 속에서 헤징 효율성을 모델링하여, 보다 정확하고 즉각적인 헤징 전략을 제공합니다. 이 프레임워크는 액터-비평가 네트워크 구조를 이용해 누적 보상의 분포를 포착하며, 최적의 헤징 액션을 선택하는 액터 네트워크와 이 액션들을 평가하는 비평가 네트워크를 포함합니다. 또한, 실시간 데이터와 과거 데이터를 모두 활용하여 Vega 민감도를 측정하는 데 필요한 다양한 지표를 지속적으로 업데이트합니다.

- **Performance Highlights**: 실험적으로 ANADDH는 기존의 헤징 기술 대비 수익성과 안정성에서 상당한 성과 향상을 보여주었습니다. 이 방법은 시장의 변동성 변화에 빠르게 반응하면서도 결합된 최적화 기술을 적용하여 보다 정교한 위험 관리가 가능하다는 것을 입증했습니다. 결과적으로, ANADDH는 현대 금융 시장의 복잡한 역학을 효과적으로 처리하며 예측 가능한 금융 위험 관리에 중대한 진전을 가져옵니다.



### An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses (https://arxiv.org/abs/2502.17772)
Comments:
          18 pages, 2 figures, submitted for possible publication

- **What's New**: 본 논문은 차별적 프라이버시를 보장하는 확률적 경_gradient 하강법(Differentially Private Stochastic Gradient Descent; DPSGD)의 성능과 프라이버시 손실을 정확히 정량화하는 방법론을 제시하고 있습니다. 기존의 연구들은 일반적으로 제한된 조건과 가정을 사용하여 이론적 분석을 진행했던 반면, 본 연구에서는 커브 리니어 전환(Differential privacy) 가정을 활용하지 않고도 유용성을 분석하고 있습니다. 이를 통해 새로운 차별적 프라이버시 보장 메커니즘을 제안합니다.

- **Technical Details**: 본 연구에서는 부드러운 손실 함수에 대한 DPSGD의 프라이버시와 유용성을 체계적으로 설명합니다. 연구진은 우선 DPSGD를 통해 다루는 구간과 무제한의 경우에 모두 적용 가능한 다양한 통계적 경과를 제시했습니다. 또한, 이 연구는 GPSGD의 경향성 및 수학적 분석의 정밀도를 높였으며, 다양한 RDP(Rényi Differential Privacy) 경계 기반의 유용성 보장을 확립했습니다.

- **Performance Highlights**: 실험 결과는 본 논문에서 제안한 경계가 실제로 유효하다는 것을 보여줍니다. 특히, 더 작은 경계 지름의 경우, 프라이버시와 유용성을 동시에 개선할 수 있는 조건을 제시하였으며, 다양한 실험을 통해 이러한 가설을 뒷받침하고 있습니다. 기존과 다른 접근 방식을 통해 프라이버시 효용 분석의 새로운 방향성을 모색하였습니다.



### Sample Selection via Contrastive Fragmentation for Noisy Label Regression (https://arxiv.org/abs/2502.17771)
Comments:
          NeurIPS 2024

- **What's New**: 본 연구에서는 ConFrag라는 새로운 방법론을 제안합니다. 이 접근법은 레이블과 특성 간의 연속적인 상관관계를 활용하여 회귀 문제의 노이즈 레이블 문제를 해결하는 데 초점을 맞춥니다. 특히, 레이블 공간에서 가장 먼 조각들을 짝지어 대조적인 조각 쌍(contrastive fragment pairs)을 형성함으로써, 데이터 포인트 간 유사성에 기반한 훈련이 가능합니다.

- **Technical Details**: ConFrag(Contrastive Fragmentation) 프레임워크는 데이터셋을 대조적인 조각 쌍으로 나누고, 이 조각 쌍을 통해 클린 샘플을 선택하는 방식으로 작동합니다. 선택된 클린 샘플은 점진적인 학습을 통해 회귀 모델의 성능을 향상시킵니다. 이 프레임워크는 노이즈 비율에 무관하게 작동하며, 인접 조각의 혼합(mixture) 모델을 통해 주변 동의(neighborhood agreement)를 사용하여 노이즈 레이블을 식별합니다.

- **Performance Highlights**: ConFrag는 14개의 최첨단 기법들과 비교하여 일관되게 더 우수한 성능을 보여주었습니다. 다양한 도메인에서 수집한 6개의 벤치마크 데이터셋을 통해 실험을 수행하였으며, 이 과정에서 레이블 간의 노이즈 정도를 고려하는 Error Residual Ratio(ERR)라는 새로운 성능 지표를 도입했습니다. 이러한 방법론은 레이블 노이즈의 대칭 및 무작위 가우시안 노이즈에 강한 저항력을 가지고 있습니다.



### DeepSeek vs. ChatGPT: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks (https://arxiv.org/abs/2502.17764)
- **What's New**: 이번 연구에서는 최신 LLM(대형 언어 모델)인 ChatGPT o3-mini-high와 DeepSeek R1이 과학적 문제 해결에 어떻게 접근하는지를 비교합니다. 실험을 통해 이 모델들이 수치적 문제와 PDE(부분 미분 방정식) 기반 문제 해결에서의 성능 차이를 분석합니다. ChatGPT o3-mini-high는 일반적으로 더 높은 정확도와 응답 속도를 보여, 다양한 계산 작업에 있어 더 실용적인 선택으로 평가됩니다.

- **Technical Details**: 연구에서는 LLM의 성능을 평가하기 위해 DeepSeek V3, DeepSeek R1, ChatGPT 4o, ChatGPT o3-mini-high 네 가지 모델을 사용했습니다. 모델들은 전통적인 수치 방법, 예를 들어 유한 차분법(Finite Difference Method)과 유한 요소법(Finite Element Method)을 활용해 PDE를 해결하는 능력을 평가받았습니다. 실험은 LLM들이 적절한 입력 함수 공간을 정의해야 하는 비트리비얼(decision is required) 문제를 포함해 설계되었습니다.

- **Performance Highlights**: ChatGPT o3-mini-high는 수치적 알고리즘과 과학적 기계 학습에서 전반적으로 뛰어난 성능을 보였습니다. 특히, DeepSeek R1보다 속도와 정확성 면에서 우수한 결과를 도출했습니다. 실험을 통해 모델들은 수학적 논리의 깊이, 신뢰성 및 연구 수준 과학 문제로 일반화하는 능력도 평가되었습니다.



### Applications of deep reinforcement learning to urban transit network design (https://arxiv.org/abs/2502.17758)
Comments:
          This is a copy of my PhD thesis, which was successfully defended at McGill University in December of 2024. arXiv admin note: text overlap with arXiv:2404.05894

- **What's New**: 이 논문은 기존의 메타휴리스틱 최적화 알고리즘 대신 강화 학습(reinforcement learning)을 활용하여 대중 교통망 설계를 위한 신경망(neural network)을 훈련시키는 방법을 제안합니다. 특히, 이 연구는 Transit Network Design Problem (TNDP)을 Markov Decision Process (MDP)로 포맷하여 신경망 정책을 훈련하는 혁신적인 접근을 채택합니다. 이러한 접근법은 기존의 알고리즘들이 고려하지 않았던 새로운 가능성을 열어줍니다.

- **Technical Details**: TNDP 문제는 도시의 기존 도로 네트워크 및 여행 수요에 기초하여 비용 함수(cost function)를 최소화하면서 수요를 충족하는 대중 교통 노선 집합을 찾는 최적화 문제입니다. 본 논문에서는 강화 학습을 통해 신경망 정책을 훈련하고, 이를 메타휴리스틱 알고리즘과 결합하여 해답 공간(solution space)에서의 유망한 이동을 제안합니다. 이러한 하이브리드 알고리즘은 고전적인 메타휴리스틱 프레임워크 내에서 핵심 구성 요소로 작용하게 됩니다.

- **Performance Highlights**: 실제로, Laval, Quebec의 대중 교통망을 재설계하는 데 이 접근법을 적용했으며, 시뮬레이션 결과 기존 교통망에 비해 더 나은 서비스와 낮은 비용을 제공함을 보여주었습니다. 강화 학습을 통해 훈련된 신경 정책과 메타휴리스틱 알고리즘을 결합한 하이브리드 방식이 이전에 비해 우수한 교통망 계획을 가능하게 함을 증명했습니다.



### Robust and Efficient Deep Hedging via Linearized Objective Neural Network (https://arxiv.org/abs/2502.17757)
- **What's New**: 이번 논문에서는 금융 파생상품 위험 관리를 위한 혁신적인 접근법인 Deep Hedging with Linearized-objective Neural Network (DHLNN)을 제안합니다. DHLNN은 신경망의 훈련 과정을 강화하여 기존의 계산 비효율성과 잡음 데이터에 대한 민감성을 극복합니다. 이 프레임워크는 고정 구배 최적화 기법과 선형화된 훈련 동역학을 결합하여 안정적이고 빠른 수렴을 달성하며, 실제 시장 조건에 적절하게 적응할 수 있는 유연성을 제공합니다.

- **Technical Details**: DHLNN은 본질적으로 내장 최적화 접근법을 사용하여 훈련의 선형화를 실현합니다. 이 방법은 복잡한 최적화 경관을 단순화하고, 높은 차원 금융 환경에서의 전략 최적화에 대한 계산적 복잡성을 줄입니다. 또한, 블랙-숄즈 델타(anchor)와의 통합을 통해, 시장 변동성에 대한 모델의 강인성을 높이며, 파생상품의 가치 변동에 따른 전체 경로를 최적화합니다.

- **Performance Highlights**: 복잡한 시장 시나리오에서 DHLNN의 성능을 검증하기 위해, 합성 및 실제 시장 데이터를 바탕으로 광범위한 실험이 수행되었습니다. 그 결과, DHLNN은 빠른 수렴 속도, 향상된 안정성, 그리고 뛰어난 헤징(hedging) 성과를 보여주며, 금융 위험 관리에서 실용적이고 신뢰할 수 있는 솔루션을 제공하는 핵심 프로젝트로 자리잡고 있습니다.



### Graded Neural Networks (https://arxiv.org/abs/2502.17751)
- **What's New**: 이번 논문은 graded vector spaces $\,V_{\boldsymbol{w}}^{n}$를 기반으로 한 새로운 graded neural network (GNN) 프레임워크를 제시합니다. 전통적인 신경망 구조에 대수적 그레이딩을 접목시켜 기능의 중요성을 반영할 수 있는 신경세포, 층, 활성화 함수 및 손실 함수를 개발했습니다. 이는 머신 러닝 및 광자 시스템과 같은 다양한 응용 분야에 적용될 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: 이 논문은 좌표별 그레이딩 구조를 활용하여 칼라르 액션 $\,\lambda\ast \mathbf{x}=(\lambda^{q_{i}}x_{i})$을 정의하고, graded neural network(GNN)를 설계했습니다. GNN은 각 입력의 중요성을 고려하여 대응하는 그레이드 $\,gr(x_{i}) \in I$를 부여하여 신경망 계층의 연산을 수행합니다. 이러한 구조는 전통적인 신경망 모델이 아닌, 그레이드가 있는 입력을 자연스럽게 처리할 수 있는 특징을 가지고 있습니다.

- **Performance Highlights**: 이 연구는 40%의 정확도로 시작한 이미지 자동군 예측하였으나, 가중된 불변량을 사용하는 경우 정확도가 99%로 상승하는 것을 발견했습니다. 이러한 성장은 체계적이며 가정적이지 않으며, GNN이 본질적으로 성능을 향상시킬 수 있는 가능성을 제시합니다. 평가된 다양한 시스템에서의 성능을 통해, GNN은 광자 기반 시스템에서 특히 고속 레이저 구현에 대한 가능성을 열어줍니다.



### FinP: Fairness-in-Privacy in Federated Learning by Addressing Disparities in Privacy Risk (https://arxiv.org/abs/2502.17748)
- **What's New**: 이 논문은 기계 학습(ML)에서 개인 정보 보호의 공정한 분배를 다루는 FinP(공정한 개인정보 보호) 프레임워크를 소개합니다. 이는 특히 연합 학습(Federated Learning, FL) 환경에서 소스 추론 공격(SIA)에 대한 불균형한 노출을 완화하기 위해 설계되었습니다. FinP는 서버 측과 클라이언트 측의 두 가지 접근 방식을 사용하여 공정성을 달성하며, 개인 정보 보호 리스크를 공정하게 분배하는 것을 목표로 합니다.

- **Technical Details**: FinP 프레임워크는 클라이언트 기여의 불공정성을 해결하기 위해 서버 측에서 적응형 집계를 사용하고, 다양한 클라이언트의 취약성을 줄이기 위해 클라이언트 측 정규화를 적용합니다. 특히, PCA(주성분 분석) 거리로 추정된 개인 정보 보호 리스크에 따라 클라이언트 모델 업데이트의 가중치를 조정하는 방법을 사용하여 불균형을 해소합니다. 이러한 기술적 접근은 개인 정보 보호의 불평등한 위험 분포와 그 원인을 동시에 해결합니다.

- **Performance Highlights**: HAR(인간 활동 인식) 및 CIFAR-10 데이터셋에서 평가된 결과, FinP는 HAR에서 공정성을 약 20% 향상시키면서 모델의 유용성에 미치는 영향을 최소화합니다. 또한 CIFAR-10에서 SIA 리스크를 효과적으로 완화하며, FL 시스템에서의 개인 정보 보호 공정성을 향상시키는 능력을 보여주고 있습니다.



### Phoeni6: a Systematic Approach for Evaluating the Energy Consumption of Neural Networks (https://arxiv.org/abs/2502.17734)
Comments:
          The paper consists of 24 pages and 25 figures. It is currently under review at the journal Sustainable Computing: Informatics and Systems

- **What's New**: 이번 연구는 Phoeni6라는 체계적인 방법론을 제안하여 신경망의 에너지 소비를 평가하는 데 있어 공정한 비교(Fair Comparison, FC)와 결과 재현성(Result Reproducibility, RR)의 원칙을 준수하는 솔루션을 제공합니다. Phoeni6는 에너지 관련 데이터와 구성을 관리하여 평가 과정의 이동성과 투명성을 보장하며, 평가 자동화를 위한 컨테이너화된 도구들을 제공합니다. 특히, 다양한 이미지 파일 형식이 에너지 소비에 미치는 영향을 분석하는 두 가지 사례 연구를 통해 신경망의 에너지 효율성을 최적화할 필요성을 강조하고 있습니다.

- **Technical Details**: Phoeni6의 방법론은 신경망의 에너지 평가 과정에서의 투명성과 이식성, 그리고 FC와 RR 원칙을 준수하기 위해 설계되었습니다. 이 시스템은 11단계로 구성된 구조화된 과정을 통해 에너지 소비 평가를 수행하며, 각 단계를 유연하게 진행할 수 있도록 합니다. 등록된 디바이스 정보를 기반으로 에너지 소비와 성능을 평가하는데 필요한 모든 데이터를 데이터베이스에 저장하고, 최적화된 알고리즘을 통해 다양한 신경망의 성능을 비교할 수 있게 합니다.

- **Performance Highlights**: 사례 연구에서, AlexNet과 MobileNet의 에너지 소비 비교 결과, MobileNet이 원본 이미지에서 최대 6.25%, 리사이즈된 데이터셋에서는 2.32% 더 에너지 효율적임을 보여주었습니다. 또한, BMP 이미지 파일 형식이 PNG에 비해 에너지 소비를 최대 30% 줄일 수 있음을 자세히 설명하고 있습니다. 이러한 결과는 Phoeni6가 다양한 신경망 애플리케이션의 에너지 소비 최적화 및 지속 가능한 인공지능 관행 구축에 중요한 역할을 한다는 점을 강조합니다.



### Aligning Compound AI Systems via System-level DPO (https://arxiv.org/abs/2502.17721)
Comments:
          Accepted to workshops MARW and WMAC (Oral) at AAAI25

- **What's New**: 본 논문에서는 여러 상호작용하는 구성 요소로 이루어진 조합 AI 시스템의 일관성을 확보하기 위한 새로운 접근 방식을 제안합니다. 기존의 직접 선호 최적화(Direct Preference Optimization, DPO) 방법이 복합 AI 시스템에 직접 적용될 수 없다는 문제를 강조하고, 이러한 시스템을 Directed Acyclic Graph (DAG)로 구성하여 문제를 해결하려고 시도합니다.

- **Technical Details**: 조합 AI 시스템을 DAG로 모델링함으로써 구성 요소 간의 관계 및 데이터 생성 프로세스를 명확하게 포착할 수 있습니다. 이를 기반으로 시스템 수준 DPO(System-Level DPO, SysDPO)를 제안하여 조합 시스템 내의 최적화를 수행하고, 대형 언어 모델(LLM)과 확산 모델(diffusion model)의 공동 정렬을 연구합니다. 또한, DDPM(diffusion model)의 작동 원리와 최적화 방법에 대한 자세한 설명을 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 LLM과 확산 모델의 공동 정렬이 효과적임을 입증하였습니다. 이러한 발견은 조합 AI 시스템의 정렬 문제에 대한 깊은 통찰력을 제공하며, 향후 기술 발전의 토대를 마련합니다. 본 연구는 기존의 AI 시스템 정렬 방식의 한계를 보완하는 중요한 기여를 하고 있습니다.



### Learning Backbones: Sparsifying Graphs through Zero Forcing for Effective Graph-Based Learning (https://arxiv.org/abs/2502.17713)
Comments:
          13th International Conference on Complex Networks and their Applications

- **What's New**: 이 논문에서는 'learning backbones'라고 불리는 그래프 희소화의 새로운 프레임워크를 제안합니다. 이 프레임워크는 원본 그래프의 필수 학습 속성을 보존하면서 계산 효율성을 개선하고 학습 알고리즘의 복잡성을 줄입니다. 저자들은 zero-forcing (ZF) 현상을 활용하여, 원본 그래프에서 동적 속성을 유지하는 트리를 생성합니다.

- **Technical Details**: 이 프레임워크는 그래프를 다이나믹 시스템으로 간주하고, 최소한의 엣지 집합을 선택하여 그래프의 본질적인 동작을 포착하는 방법을 제안합니다. 다이나믹 특성을 보존하는 것이 정확한 분류와 예측을 위해 중요하며, control theory의 원칙을 통합하여 희소한 정보를 유지하는 것이 목표입니다. 또한, zero-forcing set (ZFS) 기반의 제어 백본을 제안하여, 기존 기술과 비교해 우수한 정밀도와 효율성을 입증합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 다양한 데이터셋과 기준 모델에 대해 그래프 분류 작업에서 기존 기술을 초월하는 성능을 보였습니다. 이 런닝 백본 방법론은 8개의 데이터셋에서 평가되었으며, 성능과 희소성 모두에서 뛰어난 결과를 나타냈습니다. 나아가, 노드 간 거리 메트릭을 활용하여 프레임워크의 활용성을 더욱 향상시킬 수 있는 확장 가능성을 탐색합니다.



### Robust Federated Learning with Global Sensitivity Estimation for Financial Risk Managemen (https://arxiv.org/abs/2502.17694)
- **What's New**: 이번 논문에서는 탈중앙화된 금융 시스템에 적합한 Federated Risk-Aware Learning with Central Sensitivity Estimation (FRAL-CSE)라는 새로운 Federated Learning (FL) 프레임워크를 제안합니다. 이 프레임워크는 중앙 집중식 감도의 변별성을 사용해 글로벌 모델 동태를 근사화하는 중앙 집중적 가속 메커니즘으로, 강력한 리스크 관리와 효율적인 학습을 목표로 합니다. FRAL-CSE는 그로 인한 부정확성을 최소화하며, 훈련 효율성을 높이고 최적화 안정성을 개선합니다.

- **Technical Details**: FRAL-CSE는 로컬 민감도 정보를 통해 두 번째 계수 정보를 효율적으로 포함하여 훈련 과정을 최적화합니다. 이 방법은 전통적인 모델 업데이트 방식과 다르게 자주 로컬 재평가를 요구하지 않고, 시장의 극단적인 변화에 대한 저항성을 강화하기 위해 왜곡 리스크 측정을 트레이닝 목표에 통합합니다. 이 프레임워크는 데이터의 이질성과 수정 불가능한 배열을 완화하며, 금융 결정 과정에서의 안정성을 높입니다.

- **Performance Highlights**: 실험 결과, FRAL-CSE는 기존의 현대적인 방법들과 비교하여 여러 이질적인 데이터 세트에서 수렴 속도를 높이고 회복 탄력성을 개선하는 데 효과적임을 입증했습니다. 이 논문은 또한 FRAL-CSE가 자원의 제약이 있는 동적 환경에서도 안정성과 확장성을 유지할 수 있는 가능성을 보여주었습니다.



### Predictive Response Optimization: Using Reinforcement Learning to Fight Online Social Network Abus (https://arxiv.org/abs/2502.17693)
Comments:
          To appear in USENIX Security 2025

- **What's New**: 이번 연구는 온라인 소셜 네트워크(OSNs)에서의 악용 감지의 목표가 단순한 이진 분류기를 만드는 것만이 아니라는 점을 강조합니다. 연구팀은 사용자가 경험하는 피해와 악용의 영향을 최적화하기 위한 행동 선택(actions)과 그에 따른 접근법을 제안합니다. 이 연구는 Predictive Response Optimization (PRO)라는 새로운 시스템을 도입하여 사용자 경험과 악용을 고려한 다차원적 트레이드오프를 해결하는 방식을 탐색합니다.

- **Technical Details**: PRO 시스템은 강화 학습(reinforcement learning)을 기반으로 하여 각 행동의 결과를 예측하고, 이를 바탕으로 사용자 경험에 미치는 영향을 최소화하면서 악용을 줄이는 방법을 모색합니다. 전통적인 이진 분류 방식의 한계를 극복하기 위해, 시스템은 다수의 행동을 선택하고, 'soft' enforcement 방식으로 안전한 사용자 경험을 유지할 수 있는 가능성을 탐구합니다. 이는 기존의 행동 선택 문제를 새로운 관점에서 바라보는 결과를 가져왔습니다.

- **Performance Highlights**: PRO 시스템을 Instagram과 Facebook에 적용한 실험 결과, 각각 59%와 4.5%의 악용량 감소가 기록되었습니다. 이 과정에서 사용자에게 부정적인 영향을 미치지 않으면서도 대규모로 악용 행위를 효과적으로 줄일 수 있음을 증명했습니다. 또한, 몇 가지 사례 연구를 통해 시스템이 변화하는 비즈니스 제약, 시스템 동작 및 적대적 전술에 어떻게 빠르게 적응할 수 있는지를 보여주었습니다.



### Yes, Q-learning Helps Offline In-Context RL (https://arxiv.org/abs/2502.17666)
- **What's New**: 이번 연구에서는 Reinforcement Learning (RL) 접근법을 확장 가능한 오프라인 In-Context RL (ICRL) 프레임워크와 통합하는 방법을 탐구합니다. 150개 이상의 데이터셋을 테스트하여 RL 목표 최적화가 성능을 평균적으로 40% 향상시킨다는 것을 입증하였습니다.

- **Technical Details**: 실험은 GridWorld와 MuJoCo 환경에서 파생된 다양한 데이터셋을 사용하여 진행되었습니다. RL 목표와 보상 극대화 goals를 정렬하는 것이 중요함을 보여주며, 오프라인 RL 기반 방법들이 온라인 접근법보다 성능이 더 뛰어난 것을 확인하였습니다.

- **Performance Highlights**: 이 연구의 결과는 다양한 데이터셋 커버리지 및 환경 복잡성에서도 오프라인 RL이 ICRL 설정에서 애플리케이션으로서의 가능성이 있음을 강조합니다. 또한, 알고리즘 증류 (Algorithm Distillation, AD)와 비교했을 때 상당한 성능 향상을 가져온다는 것을 보여줍니다.



### Architecting Digital Twins for Intelligent Transportation Systems (https://arxiv.org/abs/2502.17646)
- **What's New**: 이 논문은 Intelligent Transportation Systems (ITS)에서의 Digital Twin (DT) 플랫폼인 DigIT의 아키텍처를 제안합니다. 기존의 프레임워크의 한계를 극복하기 위해 모듈형 및 확장 가능한 솔루션을 제공하여 교통 관리를 개선합니다. 이 아키텍처는 Domain Concept Model (DCM)을 기반으로 하여 중요한 ITS 구성 요소를 체계적으로 모델링하여 예측 모델링과 시뮬레이션의 매끄러운 통합을 지원합니다.

- **Technical Details**: 제안된 아키텍처는 과거와 실시간 데이터에 기반하여 교통 패턴을 예측하는 머신러닝(Machine Learning) 모델을 활용하며, 이를 통해 교통 시나리오의 시뮬레이션과 결과 예측을 효과적으로 지원합니다. 또한, 변동하는 교통 패턴에 적응하기 위해 자동화된 Machine Learning Operations (MLOps)를 통합하여 예측 모델의 배포 및 생애주기 관리를 자동화합니다. 이 시스템은 일관된 예측과 계산 효율성을 제공하여 실제 ITS 응용 프로그램에 적합성을 증명합니다.

- **Performance Highlights**: 실제로 구현된 사례연구에서 이 디지털 트윈 시스템은 실시간 데이터 수집, 시뮬레이션을 통한 교통 시나리오 테스트 및 예측 모델링을 통해 효율성을 높였습니다. 높은 교통 밀도가 관찰된 하이데라바드의 교차로에 위치한 엣지 장치를 활용하여 차량 수, 밀도 및 혼잡 지표와 같은 실시간 트래픽 메트릭스를 수집했습니다. 결과적으로, 이 시스템은 정확한 예측과 실시간 적응력을 통해 스마트하고 효율적인 교통 관리 시스템으로서의 가능성을 보여줍니다.



### The Power of Graph Signal Processing for Chip Placement Acceleration (https://arxiv.org/abs/2502.17632)
Comments:
          ICCAD'24 conference

- **What's New**: 본 논문에서는 GiFt라는 새로운 매개변수 없는 접근 방식을 제안하며, 이는 반도체 칩 배치의 성능을 크게 향상시킬 수 있습니다. GiFt는 그래프 신호 처리(graph signal processing)에 기반하여 다중 해상도의 신호를 효과적으로 포착하여 최적화된 배치 솔루션을 생성합니다. 기존 딥러닝 기반 접근 방식이 가지는 고비용의 모델 학습 없이도 배치 프로세스를 가속화할 수 있습니다.

- **Technical Details**: GiFt는 회로 그래프에서의 스무딩(smoothness)을 촉진하는 다중 주파수 그래프 필터로 작동하여 저비용 계산 문제를 해결합니다. 또한 GiFt-Placer라는 현대의 분석적 배치기와 통합되어 수많은 최적화 반복을 최소화 합니다. 이 접근법은 그래프 신호 처리의 관점에서 GCN(그래프 합성곱 신경망) 기반 접근법이 가지는 한계를 극복하며, 적절한 그래프 구조를 포착하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, GiFt-Placer는 기존의 최첨단 분석 배치기보다 경쟁력 있는 성능을 보이며 배치 효율성을 현저하게 개선했습니다. 특히, GiFt-DREAMPlace는 최근에 제안된 GPU 가속 분석 배치기인 DREAMPlace에 비해 실행 시간을 45% 이상 단축시켰습니다. 이러한 결과는 GiFt의 효율성이 성능에 긍정적인 영향을 미친다는 것을 보여줍니다.



### Instance-Dependent Regret Bounds for Learning Two-Player Zero-Sum Games with Bandit Feedback (https://arxiv.org/abs/2502.17625)
- **What's New**: 본 논문에서는 두 플레이어 제로섬 게임에 대한 bandit feedback을 사용하여 no-regret self-play 학습 동역학의 수렴 속도를 가속화하는 방법을 제시합니다. 기존 연구들은 정확한 gradient feedback의 접근을 가정했으나, 본 연구는 Tsallis-INF 알고리즘을 통해 improved regret bound를 제공합니다. 이 알고리즘을 통해 얻어진 regret는 O(c_1 log T + √(c_2 T))와 같이 표현됩니다.

- **Technical Details**: Tsallis-INF 알고리즘은 Zimmert와 Seldin (2018)에서 제안되었으며, 두 플레이어가 이 알고리즘을 적용할 경우 regret를 게임의 복잡성에 따라 조절할 수 있습니다. 여기서 상수 c_1은 학습의 복잡성을 나타내며, c_2는 전략 공간의 경계 근처에 있을 때 매우 작을 수 있습니다. 특별히 순수 전략 Nash equilibrium이 존재할 경우, c_2 값은 0이 되어 최적의 regret bound를 형성하게 됩니다.

- **Performance Highlights**: 우리의 알고리즘은 평균 반복 수렴뿐만 아니라 마지막 반복 수렴(last-iterate convergence)을 보장하여 학습 동역학의 일상적인 행동에 대한 보다 선호되는 수렴 보장을 제공합니다. 또한, 알고리즘을 일정 횟수 수행한 후 가장 자주 선택된 행동의 쌍이 PSNE일 확률이 일정하게 유지되며, 이를 반복하여 더욱 높은 확률로 PSNE를 파악할 수 있습니다.



### Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations (https://arxiv.org/abs/2502.17618)
Comments:
          Extended version of an identically-titled paper accepted at AAMAS 2025

- **What's New**: 이번 연구에서는 복잡한 순차적 작업에서 팀원 간의 조정을 위해 'DTIL'이라는 새로운 계층적 Multi-Agent Imitation Learning (MAIL) 알고리즘을 도입했습니다. 기존의 MAIL 방법들이 동질적 시연(demonstrations)을 가정하는 데 비해, DTIL은 이질적 시연을 효과적으로 학습할 수 있도록 설계되었습니다. 이러한 접근법은 팀 구성원이 각기 다른 정책(policy)을 가지고 있음을 반영하며, 이로 인해 보다 유연한 팀 행동 모델링이 가능합니다.

- **Technical Details**: DTIL은 각 팀 구성원을 계층적 정책으로 표현하고, 이질적 팀 시연에서 이러한 정책을 분산 방식으로 학습합니다. 분포 정합(distribution-matching) 접근법을 사용하여 누적 오류(compounding errors)를 줄이며, 긴 시간(horizons)과 연속 상태(state) 표현에 효과적으로 확장할 수 있도록 합니다. 이는 복잡한 작업을 처리할 수 있는 능력을 증가시킵니다.

- **Performance Highlights**: 실험 결과, DTIL은 기존의 MAIL 기준선(baselines)에 비해 우수한 성능을 보였으며, 다양한 협업 시나리오에서 팀 행동을 정확하게 모델링했습니다. 이 알고리즘은 팀원의 협력적 수행에 있어 보다 나은 결과를 달성함으로써, 다중 에이전트 시스템과 인간-AI 팀을 위한 강력한 기초를 마련합니다.



### Provable Model-Parallel Distributed Principal Component Analysis with Parallel Deflation (https://arxiv.org/abs/2502.17615)
Comments:
          CPAL 2025

- **What's New**: 이 논문은 분산 주성분 분석(Principal Component Analysis, PCA) 프레임워크를 연구하며, 각 작업자가 고유한 고유벡터를 목표로 하고 동료의 중간 솔루션으로부터 이를 업데이트하여 그 결과를 정제합니다. 기존의 중심화된 고유값 문제의 디플레이션 방법에서 영감을 받아, 우리 접근 방식은 디플레이션 단계에서의 순차적 의존성을 해소하고 작업자의 비동기 업데이트를 허용하되, 소통 비용은 최소화합니다. 이는 기존 문헌에서 다루어지지 않았던 이론적 기초를 제시합니다.

- **Technical Details**: 이 연구에서는 협업 계산 프레임워크에 기반해 모델 병렬 분산 PCA를 혁신적으로 발전시킵니다. 우리는 작업자 간에 엄격한 순차적 의존성을 두지 않고 주성분을 병렬로 계산하는 것을 가능하게 하는 분산 PCA 프레임워크를 소개합니다. 또한, 데이터가 미니 배치로 들어올 경우를 대비한 알고리즘 수정 방법과 병렬 계산과 수렴 속성 간의 상호작용을 정형화하여 우리의 알고리즘이 효과적이라는 것을 증명합니다.

- **Performance Highlights**: 실험 결과, 우리의 분산 PCA 알고리즘은 ImageNet과 같은 대규모 데이터셋에서 기존 알고리즘과 동등한 성능을 보입니다. 이로써 우리는 제안한 이론을 입증하고, 작업자 간의 비순차적 상호작용이 실제 실행에 효과적임을 증명합니다. 이를 통해 PCA의 병렬 디플레이션 과정에 대한 이해가 증대되며, 이론적 벤치마크를 수립합니다.



### Scalable Graph Condensation with Evolving Capabilities (https://arxiv.org/abs/2502.17614)
Comments:
          16 pages, 6 figures

- **What's New**: 이 논문에서는 GECC (Graph Evolving Clustering Condensation)라는 새로운 그래프 응축 방법을 제안합니다. 이는 대규모 및 발전하는 그래프 데이터를 처리하기 위해 설계되었으며, 피쳐를 집계하여 클래스별 클러스터링을 수행합니다. 기존의 고정적인 훈련 집합을 다루는 방식과 다르게, GECC는 점진적으로 발전하는 그래프에 적응할 수 있는 가능성을 보여줍니다.

- **Technical Details**: GECC는 기존 그래프 응축 방법의 단점을 극복하기 위해, 노드 기능을 클러스터로 나누고 각 타임스텝에서 중심을 새로운 응축 노드 기능으로 사용합니다. 이 방법은 그래프의 변화를 수용하고, 과거 응축 결과를 초기화로 활용하여 점진적 클러스터링을 가능하게 합니다. 이러한 방식은 전통적인 GNN 훈련 프로세스에서 필요한 높은 비용의 그래디언트 계산을 피할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, GECC는 기존의 최신 그래프 응축 방법들보다 나은 성능을 보여주며, 특히 대규모 데이터셋에서 약 1,000배의 속도 향상을 달성했습니다. 이 방식은 높은 정확도를 유지하면서도 응축 프로세스를 보다 효율적으로 만들어, 실용적인 응용 분야에서의 적합성을 크게 향상시킵니다.



### Flexible Counterfactual Explanations with Generative Models (https://arxiv.org/abs/2502.17613)
Comments:
          28 pages, 13 figures

- **What's New**: 이번 논문에서는 사용자들이 보다 유연하게 반사실적 설명을 적용할 수 있도록 지원하는 Flexible Counterfactual Explanations(FCEGAN) 프레임워크를 소개합니다. 기존 방법들이 고정된 변경 가능한 특성을 기반으로 하여 제한적임에 반해, FCEGAN에서는 사용자가 예측 시점에서 변경 가능한 특성을 동적으로 지정할 수 있는 기능을 포함하고 있습니다. 이는 사용자 요청에 따라 최소한의 수정으로 원하는 모델 예측 결과를 얻도록 합니다.

- **Technical Details**: FCEGAN은 Generative Adversarial Networks(GANs)를 활용하여 사용자 정의 제약 조건에 맞춰 설명을 조정할 수 있도록 설계되었습니다. 이 프레임워크는 반사실적 템플릿(counterfactual templates)을 도입함으로써 사용자가 어떤 특성을 변경할 수 있는지를 codify하고, 이를 통해 설명의 유효성과 실제 데이터의 밀접함을 보장합니다. 또한, FCEGAN은 블랙박스 환경에서도 작동 가능하여 모델 내부 접근 없이 예측 데이터를 활용하여 설명을 생성할 수 있습니다.

- **Performance Highlights**: FCEGAN의 성능은 경제 및 헬스케어 데이터셋을 통한 실험을 통해 기존의 전통적 방법들과 비교했을 때, 반사실적 설명의 유효성을 크게 향상시키는 것으로 나타났습니다. 특히, FCEGAN은 사용자 주도의 유연성과 블랙박스 호환성을 결합하여, 사용자 제약에 맞춘 개인화된 설명 생성을 지원합니다. 이러한 접근은 금융 및 헬스케어와 같은 분야에서 사용자 제약과 해석 가능성이 필수적인 상황에 매우 적합합니다.



### Synthetic Text Generation for Training Large Language Models via Gradient Matching (https://arxiv.org/abs/2502.17607)
Comments:
          15 pages, 5 figures, 4 tables

- **What's New**: 이번 연구에서는 실 데이터에서의 파인튜닝(fine-tuning) 시 LLM의 수렴성과 성능을 보장하는 인체 가독성이 있는 합성 텍스트를 생성하기 위한 최초의 이론적으로 엄밀한 방법론을 제안합니다. 이를 위해 Alternating Direction Method of Multipliers (ADMM) 기법을 적용하여 합성 예제의 임베딩을 반복적으로 최적화합니다. 최적화된 임베딩은 실제 데이터의 그라디언트와 유사한 텍스트 토큰 시퀀스로 매핑됩니다.

- **Technical Details**: 이 연구에서는 실 데이터의 그라디언트와 유사한 임베딩을 찾기 위해 이산 최적화 문제를 공식화합니다. 이 최적화 과정에서는 읽을 수 있는 텍스트를 보장하기 위해 낮은 perplexity를 요구하는 제약 조건을 추가합니다. ADMM을 통해 이산 최적화 문제를 해결함으로써, 실제 데이터에 대한 파인튜닝 결과에 가까운 솔루션으로 수렴함을 검증합니다.

- **Performance Highlights**: GrADmm 방식으로 생성된 합성 데이터가 실제 예제에 대해 최대 32.4% 향상된 성능을 보여줍니다. 또한, GrADmm을 통해 생성된 합성 데이터는 기존 LLM의 제로샷(zero-shot) 및 퓨샷(few-shot) 생성 방법보다도 최대 10.4% 더 나은 성능을 나타냅니다. 이 방식은 Llama-3.2-1B 및 OPT-1.3B와 같은 다른 LLM에 대한 파인튜닝에 있어서도 효용이 강조됩니다.



### Hallucination Detection in LLMs Using Spectral Features of Attention Maps (https://arxiv.org/abs/2502.17598)
Comments:
          Preprint, under review

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 환각(hallucination) 탐지를 위한 새로운 방법을 제안합니다. 기존의 attention map 기반 방법은 한계가 있었던 반면, 제안된 LapEigvals 방법은 Laplacian matrix의 상위-k 고유값을 활용하여 더욱 정확한 탐지가 가능합니다. 실험 결과, 이 방법은 최신의 환각 탐지 성능을 달성하였으며, 이를 통해 향후 연구의 방향성을 제시합니다.

- **Technical Details**: 제안된 LapEigvals 방법은 attention maps를 그래프 구조의 인접 행렬(adjacency matrix)로 해석하여 이를 통계적으로 분석합니다. 이를 통해 attention maps에서 유도된 Laplacian matrix의 고유값(eigenvalues)을 사용하여 환각을 탐지하는 Probe 모델의 입력 특성으로 활용합니다. 연구 결과, Laplacian의 고유값이 이전의 방법들보다 환각과 더 밀접한 관련을 가지고 있음을 보여주었습니다.

- **Performance Highlights**: 광범위한 실험을 통해 LapEigvals는 기존 AttentionScore 방법보다 우수한 성능을 입증했으며, 다양한 데이터셋과 LLM에 걸쳐 이러한 결과가 나타났습니다. 또한, ablation study를 통해 자원의 견고성과 일반화 가능성을 강조하며, 향후 환각 탐지 방법의 발전에 기여할 논의의 기반을 마련했습니다.



### Training a Generally Curious Agen (https://arxiv.org/abs/2502.17543)
- **What's New**: 이 논문에서는 PAPRIKA라는 새로운 방법론을 소개합니다. PAPRIKA는 언어 모델이 특정 환경에 국한되지 않고 일반적인 의사결정 능력을 키울 수 있도록 하는 미세 조정(fine-tuning) 접근법입니다. 이 방법은 다양한 전략을 요구하는 여러 작업에서 생성된 합성(interaction data)을 기반으로 훈련하여 모델이 환경 피드백을 통해 새로운 작업에 적응할 수 있도록 학습합니다.

- **Technical Details**: PAPRIKA는 언어 모델이 다양한 과제에 대한 정보 추출 및 의사결정을 수행하도록 설계된 텍스트 기반 의사결정 작업의 식이로 구성됩니다. 이를 위해 기본 모델(base model)을 사용하여 상호작용 궤적(interaction trajectories)을 생성하고, 성공률에 따라 점수를 부여합니다. 또, Direct Preference Optimization의 변형을 통해 성공적인 궤적의 상대적 가능성을 높입니다.

- **Performance Highlights**: 실험 결과, PAPRIKA로 미세 조정된 모델은 추가적인 훈련 없이도 완전히 새로운 작업에 학습된 의사결정 능력을 효과적으로 이전할 수 있음을 보여주었습니다. 이 연구는 합성 데이터 생성의 유용성을 강조하며, AI 시스템이 외부 세계와의 상호작용을 통해 새로운 순차적 의사결정 문제를 자율적으로 해결할 수 있는 가능성을 제시합니다.



### On the Vulnerability of Concept Erasure in Diffusion Models (https://arxiv.org/abs/2502.17537)
- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 개인 정보 보호 및 보안 문제를 다루고 있습니다. 특히, 저작권이 있는 이미지나 해로운 이미지의 생성과 관련된 우려가 커지고 있는 가운데, 기존의 데이터 삭제 방법(즉, transfer learning을 통한 개념 지우기)에는 취약점이 있음을 보여줍니다. 그에 따라, RECORD라는 새로운 알고리즘을 도입하여 삭제된 내용을 환기할 수 있는 프롬프트를 발견하는 방법을 제안합니다.

- **Technical Details**: RECORD는 좌표 하강(Coordinate Descent) 기반의 알고리즘으로, 세밀한 프롬프트 최적화를 통해 삭제된 내용의 생성을 유도할 수 있는 입력을 찾아냅니다. 또한, 기존 방법에 비해 공격 성공률을 10배 높인 것으로 나타났습니다. 이 연구는 개념이 지워진 모델들이 적대적 공격(Adversarial Attack)에 n하기 보다 취약하다는 새로운 사실을 밝히고 있습니다, 이는 모델이 여전히 원치 않는 데이터의 정보를 보유하고 있다는 것을 시사합니다.

- **Performance Highlights**: RECORD는 현재의 최첨단 공격 방식에 비해 탁월한 성능을 보이며, 다양한 실험을 통해 지워진 모델의 동작을 탐구합니다. 또한, 유의미한 세밀한 활성화를 비교하면서, 기존의 ‘지우기’ 방법들이 실질적으로는 개념을 제거하기보다는 모델을 비정렬(misalignment)로 변환시키고 있음을 강조합니다. 이러한 위험은 저작권 관련 법률에 중대한 영향을 미칠 수 있습니다.



### The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? (https://arxiv.org/abs/2502.17535)
- **What's New**: 모델 압축 및 KV 캐시 압축이 LLM의 계산 및 저장 비용을 줄이기 위해 많은 관심을 받고 있습니다. 본 논문에서는 retrieval-augmented generation, multi-step reasoning, 외부 도구 사용과 같은 최근 LLM의 발전을 검토하고, 특정 LLM과 작업에 대해 더 작은 로또 LLM이 동일한 성능을 달성할 수 있다는 가설을 제시합니다.

- **Technical Details**: 현재 대부분의 LLM 압축 방법은 perplexity와 같은 기본적인 언어 작업에서만 성능을 보장하며, 실제 산업 상황에서는 충분한 성과를 보이지 않습니다. LLMs의 압축으로 인해 long-context retrieval 및 reasoning 능력이 감소할 수 있으며, KV 캐시 압축 또한 LLM의 긴 컨텍스트 이해 능력을 크게 제한합니다.

- **Performance Highlights**: 적응형 지식 검색(adaptive knowledge retrieval)에 관한 최근 연구는 LLM의 성능 향상과 모델 사이즈 및 지식 베이스 사이의 최적의 균형을 찾는 데 기여할 것으로 기대되고 있습니다. RAG(retrieval-augmented generation) 접근 방식은 특정 분야의 LLM 성능을 획기적으로 개선할 수 있음을 보여주며, 법률, 의료, 금융 분야에서의 LLM 적용 가능성을 높이고 있습니다.



### FedSV: Byzantine-Robust Federated Learning via Shapley Valu (https://arxiv.org/abs/2502.17526)
- **What's New**: 이 논문에서는 Federated Learning (FL)에서 악의적인 클라이언트에 대한 강력한 방어 메커니즘인 FedSV를 제안합니다. FedSV는 Shapley Value (SV)를 활용하여 사용자 기여를 측정하고, 각 클라이언트의 로컬 데이터가 모델의 평균 정확도에 주는 한계 기여를 계산합니다. 이러한 접근법은 악의적인 클라이언트를 신뢰성 있게 식별할 수 있도록 돕습니다.

- **Technical Details**: FedSV는 클라이언트가 속한 다양한 그룹을 고려하여 각 클라이언트의 기여도를 추정합니다. 이는 악의적인 클라이언트를 더욱 효과적으로 식별할 수 있도록 설계되었으며, FL의 학습 단계 동안 클라이언트의 예측 기여도를 평가합니다. 논문에서 제시된 실험은 MNIST 데이터셋을 사용하여 여러 공격 시나리오에서 FedSV의 성능을 입증합니다.

- **Performance Highlights**: 광범위한 MNIST 데이터셋 실험을 통해 FedSV의 효율성이 입증되었습니다. 다양한 공격 상황에서도 FedSV는 모델의 무결성을 보장하며, 공격으로 인한 오류를 최소화하는 데 도움을 줍니다. 이 결과는 FL 환경에서의 보안 강화를 위한 중요한 기초 자료가 될 것입니다.



### UNCA: A Neutrosophic-Based Framework for Robust Clustering and Enhanced Data Interpretation (https://arxiv.org/abs/2502.17523)
Comments:
          17 pages, 8 Figures, 1 Table

- **What's New**: 새롭게 제안된 통합 중립 클러스터링 알고리즘(UNCA)은 복잡한 데이터 클러스터링과 내재된 불확실성을 정확하게 표현하는 문제를 해결하고자 합니다. UNCA는 중립 논리를 기반으로 여러 전략을 결합하여 클러스터링 성능을 향상시킵니다.

- **Technical Details**: UNCA는 ":_lambda:-cutting matrix"를 사용하여 데이터 포인트 간의 의미 있는 관계를 필터링하는 전체적인 유사성 검사를 통해 시작됩니다. 그런 후에, 중립 K-평균 클러스터링을 위한 중심점을 초기화하며, 이때 멤버십 값은 진위, 불확정성 및 거짓의 정도를 기준으로 합니다. 최종적으로 클러스터링 결과를 확정짓기 위해 "defuzzification methods"를 활용합니다.

- **Performance Highlights**: 성능 평가 결과 UNCA는 여러 지표에서 기존 방법들을 초월했습니다. 예를 들어, Iris Dataset에서는 0.89의 Silhouette Score를, Wine Dataset에서는 0.59의 Davies-Bouldin Index를 달성했으며, Digits Dataset에서는 0.76의 Adjusted Rand Index (ARI)와 Customer Segmentation Dataset에서는 0.80의 Normalized Mutual Information (NMI)을 기록했습니다. 이러한 성과들은 UNCA가 클러스터링 정확도를 향상시키는 것뿐만 아니라 해석 가능성과 견고성도 높인다는 것을 보여줍니다.



### Spectral Theory for Edge Pruning in Asynchronous Recurrent Graph Neural Networks (https://arxiv.org/abs/2502.17522)
- **What's New**: 이 논문은 Asynchronous Recurrent Graph Neural Networks (ARGNNs)을 활용한 그래프 구조 데이터의 복잡한 의존성을 모델링하는 새로운 방법을 제안합니다. 특히, 그래프 스펙트럼 이론을 기반으로 하는 동적 가지치기(dynamic pruning) 기법을 이용하여 불필요한 에지를 효율적으로 제거함으로써 성능 저하 없이 모델의 계산 비용을 줄이는 것을 목표로 합니다.

- **Technical Details**: 제안하는 프루닝 방법은 네트워크 그래프의 라플라시안(Laplacian) 고유값의 허수 구성 요소를 활용하여 동적으로 처리됩니다. 이러한 접근법은 ARGNN의 복잡도를 줄여주고, 동적 그래프의 학습을 위한 효율성을 높이는 데 기여합니다. 결과적으로, 이 방법은 기존의 복잡한 모델링 요구를 감소시키며, 계산 시간을 단축하는 효과를 가져옵니다.

- **Performance Highlights**: 실험 결과는 제안된 동적 가지치기 방법이 ARGNN의 성능을 크게 저하시키지 않으면서도, 모델의 효율성을 현저히 향상시킴을 나타냅니다. 다양한 데이터셋에서 테스트한 결과, 이 방법의 효과는 여러 가지 상황에서 확인되었습니다. 따라서, 이 연구는 그래프 신경망 관련 연구 및 실제 응용에 있어 새로운 방향성을 제시합니다.



### Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation (https://arxiv.org/abs/2502.17521)
Comments:
          Github Link: this https URL

- **What's New**: 이번 논문에서는 데이터 오염(data contamination)의 위험을 줄이기 위해 고안된 정적(static)에서 동적(dynamic) 벤치마킹 방식의 변화에 대한 깊이 있는 분석을 수행합니다. 특히 동적 벤치마킹의 표준화된 평가 기준이 부족하다는 점을 강조하며, 이를 개선하기 위한 사례들을 제안합니다. 논문은 기존 연구의 한계점을 짚고, LLM(대규모 언어 모델)의 벤치마킹 방법론에 대한 종합적인 개요를 제공합니다.

- **Technical Details**: 본 연구는 정적 벤치마킹의 한계를 지적한 후, 시간에 따라 업데이트되는 벤치마킹 데이터세트를 사용하는 동적 벤치마킹 방법론을 제안합니다. 학습 단계에서 벤치마크 데이터가 모델의 훈련 데이터와 겹치지 않도록 하기 위해 다양한 기술적 접근 방식을 소개하며, 데이터 암호화(data encryption) 및 후속 오염 탐지(post-hoc contamination detection)와 같은 방법들이 있습니다. 하지만 이러한 정적 방법들의 한계로 인해 새로운 동적 벤치마킹 스킴이 도입되었습니다.

- **Performance Highlights**: 가장 주목할 만한 점은 기존의 동적 벤치마크가 제안된 평가 기준을 완전히 만족시키지 못한다는 것입니다. 이로 인해 기존의 평가 방법들이 모델의 실제 성능을 왜곡할 수 있는 문제점을 내포하고 있다는 것을 암시합니다. 이 논문에서는 데이터 오염의 위험을 줄이기 위한 동적 벤치마킹 방법에 대한 체계적인 조사를 실시하여, 향후 연구 방향에 대한 귀중한 인사이트를 제공합니다.



### On Neural Inertial Classification Networks for Pedestrian Activity Recognition (https://arxiv.org/abs/2502.17520)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2501.01327

- **What's New**: 이번 논문은 보행자 행위 인식을 위한 신경 관성 분류 네트워크의 개선을 위해 10가지 데이터 기반 기법을 정의하고 분석했습니다. 심층 학습(Deep Learning) 기술의 발전을 통해 관성 센서의 인식 성능과 견고성이 크게 향상되었지만, 성과를 공정하게 비교하고 평가할 수 있는 공통 벤치마크가 부족했습니다. 이 연구는 네트워크 아키텍처, 데이터 증강, 데이터 전처리의 세 가지 측면에 중점을 두고 실험을 진행하였습니다.

- **Technical Details**: 우리는 네트워크 설계에 영감을 받아 CNN(Convolutional Neural Network) 층, Bi-LSTM(Bidirectional Long Short-Term Memory) 층, 완전 연결 층(FC)을 통합한 기본 네트워크 아키텍처를 사용했습니다. 입력 신호는 1차원 CNN을 통해 처리되며, 다양한 데이터 증강 기법을 적용하여 모델 성능을 개선했습니다. 네 가지 실제 기록된 데이터 세트에서 신경 네트워크 기술의 적용이 통합적으로 검토되었습니다.

- **Performance Highlights**: 회전(rotation) 및 다중 헤드 아키텍처(multi-head architecture)를 통한 데이터 증강이 관성 분류 네트워크의 성능 개선에 가장 일관된 효능을 보였으며, 이 연구는 신경 네트워크의 구현에 대한 실용적인 통찰력을 제공합니다. 신경 네트워크의 정확도를 지속적으로 개선할 수 있는 기법들을 평가하고, 각 기법이 특정 시나리오에서 효과적인지에 대한 통찰력을 제시합니다.



### Ensemble RL through Classifier Models: Enhancing Risk-Return Trade-offs in Trading Strategies (https://arxiv.org/abs/2502.17518)
Comments:
          16 pages,5 figures, 1 table

- **What's New**: 이 논문은 금융 거래 전략에서 앙상블 강화 학습(ensemble Reinforcement Learning, RL) 모델의 활용을 종합적으로 연구하며, 분류기(classifier) 모델을 통해 성능 향상을 도모합니다. A2C, PPO, SAC와 같은 RL 알고리즘을 전통적인 분류기인 서포트 벡터 머신(Support Vector Machines, SVM), 결정 트리(Decision Trees), 로지스틱 회귀(Logistic Regression)와 조합하여 위험-수익 무역을 개선하는 방법을 조사합니다. 다양한 앙상블 방법을 평가하여 핵심 재무 지표(Cumulative Returns, Sharpe Ratios, Calmar Ratios, Maximum Drawdown)에서 개별 RL 모델과 비교한 결과, 앙상블 방법이 항상 기초 모델을 초과하는 성과를 보이는 것을 확인했습니다.

- **Technical Details**: 위험 조정 수익(risk-adjusted returns) 측면에서 앙상블 방법이 기초 모델보다 뛰어난 성능을 나타내며, 드로우다운(drawdowns) 관리를 향상시키고 안정성을 제공합니다. 하지만 앙상블 성능은 분산 임계값(variance threshold) 선택에 민감하여, 최적 성과를 위해서는 동적인 분산 임계값 조정의 중요성을 강조합니다. 제안된 방법은 여러 RL 에이전트와 분류기로부터 신뢰도 점수를 집계하고, 분산 평가 메커니즘을 통한 신뢰할 수 없는 추정 필터링을 통해 의사 결정의 신뢰성을 향상시키기 위한 세 가지 핵심 요소로 구성됩니다.

- **Performance Highlights**: 이 연구에서는 제안된 앙상블 방법이 높은 신뢰도와 다양한 시장 환경에서의 적응 능력을 바탕으로 안정적인 의사 결정을 가능하게 한다고 주장합니다. 특히, 다양한 RL 에이전트의 출력 집계를 통해 탐색(exploration)과 활용(exploitation)을 개선하며, 신뢰도가 높은 의사 결정 시 소극적인 접근 방식을 채택함으로써 불확실성이 높은 상황에서의 신뢰성을 보장합니다. 이러한 점에서 이 논문은 금융 거래, 로봇 공학 및 기타 동적 환경에서 RL과 분류기의 결합이 가진 가치를 강조합니다.



### A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models (https://arxiv.org/abs/2502.17516)
Comments:
          30 pages, 4 Figures, 10 Tables

- **What's New**: 이번 논문은 다중 모달 기반 모델(Multimodal Foundation Models, MMFMs)의 해석 가능성(Interpretability) 향상에 초점을 두고 있습니다. 저자들은 특히 대형 언어 모델(Large Language Models, LLMs)과 다중 모달 모델 간의 해석 가능성 차이를 줄이고자 다양한 접근 방식을 제시하고 있습니다. 이 연구는 LLM 해석 방법을 MMFM에 적용할 수 있는지, 그리고 이러한 방법이 유사한 통찰력을 제공하는지를 탐구합니다.

- **Technical Details**: 저자들은 다중 모달 모델의 메커니즘을 이해하기 위해 기계적 해석 가능성(Mechanistic Interpretability)에 대한 새로운 3차원 분류 체계를 제안하였습니다. 이 체계는 (1) 모델 가족(Model Family), (2) 해석 기법(Interpretability Techniques), (3) 응용(Application)을 포함하며, 각각의 카테고리에서 발췌된 통찰력을 실제 응용과 연결합니다. 예를 들어, 비생성적 VLM 모델(CLIP 등)과 생성적 VLM 모델(Stable-Diffusion 등) 간의 주요 차이점을 아우릅니다.

- **Performance Highlights**: 해석 가능성 연구는 LLM에서 아마도 가장 큰 발전을 이뤘지만, MMFM은 상대적으로 탐구가 부족합니다. 본 연구는 MMFM에서 발생하는 새로운 문제와 기존 비유형적 LLM 기반 해석 기법의 적용 가능성을 검토합니다. 이 연구는 MMFM 해석의 한계와 미래 연구 방향에 대한 통찰력을 제공하여 응용 문제인 모델 편집(Model Editing) 및 환각 억제(Hallucination Mitigation)의 발전을 도모합니다.



### Towards User-level Private Reinforcement Learning with Human Feedback (https://arxiv.org/abs/2502.17515)
- **What's New**: 이 연구에서는 기존의 RLHF(Reinforcement Learning with Human Feedback) 방법에서 사용자 프라이버시를 효과적으로 보호할 수 있는 새로운 프레임워크인 AUP-RLHF를 제안합니다. 기존 연구들은 주로 아이템 수준(item-level) 프라이버시 보호에 중점을 두었으며, 사용자 수준(user-level) 프라이버시는 제대로 다루지 못했습니다. 새로운 방법은 사용자 수준 레이블에 대한 차별적 프라이버시(Differential Privacy)를 통합하여 사용자 데이터의 개인 정보를 보호하는 동시에 개선된 추정 오차를 달성합니다.

- **Technical Details**: AUP-RLHF 알고리즘은 사용자 선호도에 대한 손실의 평균 그래디언트를 사용하여 파라미터 업데이트를 수행합니다. 전통적인 DP 알고리즘은 아이템 수준에 최적화되어 있어 사용자 수준에서는 유용성이 부족하다는 점을 지적하며, 고감도 데이터를 처리하기 위해 아웃라이어 제거 및 적응 샘플링 과정을 도입합니다. 알고리즘은 $(\\varepsilon, \\delta)$ 사용자 수준 프라이버시를 보장하며, 데이터에 노이즈를 추가하여 개인 정보를 보호합니다.

- **Performance Highlights**: 실험 결과, AUP-RLHF는 감정 생성(sentiment generation) 및 요약(summarization) 작업에서 기존의 다른 사용자 수준 DP 기법보다 뛰어난 성능을 보였습니다. 다양한 모델 크기와 다양한 프라이버시 파라미터 설정에 걸쳐 AUP-RLHF는 일관되게 우수한 프라이버시-유용성(trade-off) 균형을 달성했습니다. 이로써 이 연구는 사용자 프라이버시와 모델 유틸리티 간의 딜레마를 효과적으로 해결할 수 있는 가능성을 보여줍니다.



### SAE-V: Interpreting Multimodal Models for Enhanced Alignmen (https://arxiv.org/abs/2502.17514)
- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 해석 가능성을 높이기 위한 새로운 프레임워크인 SAE-V를 제안했습니다. SAE-V는 Sparse Autoencoders(SAEs) 개념을 확장하여 MLLM에 적용시키며, 이를 통해 모델의 동작과 데이터 품질을 세밀하게 해석할 수 있습니다. 이러한 접근 방식은 MLLM의 정렬 프로세스에서 데이터 필터링 기법을 활용하여 보다 효율적인 모델 정렬을 가능하게 합니다.

- **Technical Details**: SAE-V는 MLLM의 해석 가능성을 위해 설계된 메커니즘 해석 가능성 프레임워크입니다. 이 프레임워크는 텍스트와 이미지의 표현 공간을 융합하여 통합된 멀티모달 표현을 생성하는데 초점을 맞추고 있습니다. SAE-V는 MLLM의 교차 모달 상호작용을 어휘적으로 분석하여 모델의 품질과 정렬을 개선할 수 있는 기법을 제공합니다.

- **Performance Highlights**: 실험 결과, SAE-V 기반의 데이터 필터링 기법은 50%의 데이터로 110% 이상의 성능 향상을 달성했습니다. 이는 SAE-V가 MLLM의 해석 가능성과 정렬을 강화하는 데 효과적임을 입증합니다. 이와 함께 MLLM의 학습 과정에서의 특징 분포를 분석하여 멀티모달 이해 과제 성능과의 관계를 발견했습니다.



### Int2Int: a framework for mathematics with transformers (https://arxiv.org/abs/2502.17513)
- **What's New**: 이 논문은 수학 연구 문제, 특히 정수론과 관련된 문제를 해결하기 위한 Transformer 기반의 오픈 소스 코드 집합인 Int2Int를 소개합니다. Int2Int는 PyTorch로 완전하게 구현된 Transformer 아키텍처와 함께 교육 및 평가 루프를 제공하며, 일반적인 수학적 객체를 표현하고 생성하며 해독하는 클래스를 포함하고 있습니다. 공개된 데이터 준비 코드 및 실험 결과를 시각화하기 위한 Jupyter 노트북도 사용 가능합니다.

- **Technical Details**: Int2Int는 수학 문제를 문자 순서로 재작성하여 해결책을 찾아내는 방식으로 Transformer 아키텍처를 응용합니다. 특히, 두 개의 정수로 더하는 문제를 예시로, Int2Int는 입력과 출력을 위한 시퀀스를 모두 처리할 수 있는 기능을 제공합니다. Supervised learning 체계에 기반하여, 문제와 해결책의 쌍을 학습하여 예측을 수행하며, 이를 위해 cross-entropy 손실 함수를 사용합니다.

- **Performance Highlights**: Int2Int는 다양한 수학적 문제를 해결하는 데 사용할 수 있는 범용 프레임워크로, 직관적인 사용 설명서를 제공합니다. 특히 elliptic curve의 특성을 예측하거나 두 정수의 최대 공약수를 계산하는 문제에 대한 실용적인 튜토리얼을 포함합니다. 또한, MIT 라이센스 하에 제공되어 사용자가 자유롭게 활용하고 수정할 수 있는 장점이 있습니다.



### Learning multi-phase flow and transport in fractured porous media with auto-regressive and recurrent graph neural networks (https://arxiv.org/abs/2502.17512)
- **What's New**: 이 연구에서는 복잡한 다상 유체 흐름(multi-phase flow) 및 전송 동역학(transport dynamics) 모델링을 위해 그래프 신경망(Graph Neural Networks, GNN)을 사용하는 새로운 접근 방식을 제안합니다. 기존의 고도로 정밀한 conformal mesh 방식이 대규모 또는 복잡한 균열 네트워크에서 비효율적이라는 문제점을 인식하고, 이를 해결하기 위해 GNN을 활용한다는 점이 주목할 만합니다.

- **Technical Details**: 제안된 두 가지 딥러닝 아키텍처는 GNN과 재귀 GNN(recurrent GNN)입니다. 이 두 네트워크는 자가 회귀(autoregressive) 모델 롤아웃 후 전체 실제 시퀀스를 사용하여 모델을 미세 조정(fine-tuning)하는 두 단계의 훈련 전략을 따릅니다. Embedded Discrete Fracture Model (EDFM) 분산화로 인한 비구조적(topology) 계산 그리드에 적합하여, 균열이 있는 다공성 매질(fractured porous media)의 동역학을 효과적으로 모델링합니다.

- **Performance Highlights**: 시험 단계에서 자가 회귀 모델 롤아웃 동안 오류 누적(error accumulation)을 완화하는 데 두 단계 훈련 접근이 효과적임을 입증했습니다. GNN과 재귀 GNN 모두 이전에 보지 못한 균열 실현(unseen fracture realizations)에 잘 일반화되었으며, 포화 시퀀스 예측에서 유사한 성능을 보였습니다. 그러나 압력 시퀀스 예측에서는 재귀 GNN이 약간 더 우수한 성능을 보였습니다. 특히, 재귀 GNN은 긴 시퀀스 예측에서 정확도 측면에서 GNN보다 현저히 더 뛰어난 성능을 발휘했습니다.



### Recurrent Knowledge Identification and Fusion for Language Model Continual Learning (https://arxiv.org/abs/2502.17510)
- **What's New**: Recurrent-KIF는 동적 매개변수 중요도 추정을 통해 지식 전이를 향상시키는 새로운 연속 학습(Continual Learning, CL) 프레임워크입니다. 이 접근법은 두 가지 루프 구조를 활용하여 새로운 작업에 신속하게 적응하고, 과거 지식을 관리하는 데 중점을 둡니다. 이러한 동적인 중요도 분포 추정을 기반으로 Recurrent-KIF는 복잡한 환경에서의 효과적인 학습을 실제로 구현합니다.

- **Technical Details**: Recurrent-KIF는 내부 루프(inner loop)와 외부 루프(outer loop)의 협업을 통해 작동합니다. 내부 루프는 새로운 작업에 적응하며 중요한 매개변수를 식별하는 역할을 하며, 외부 루프는 새로운 및 역사적인 지식의 융합을 관리합니다. 이 과정에서 중복 지식 가지치기(redundant knowledge pruning)와 핵심 지식 병합(key knowledge merging)을 통해 지식 융합이 이루어집니다.

- **Performance Highlights**: 실험 결과, Recurrent-KIF는 CF(변별적 망각)와 KT(지식 전이) 문제를 효과적으로 완화하는 것으로 나타났습니다. 다양한 모델 아키텍처와 크기(770M에서 13B까지)에서 우수한 성능을 발휘하며, CL 벤치마크에서 기존의 최첨단 방법들을 능가했습니다. Recurrent-KIF는 지식 융합 지식(fusion of knowledge)을 조정할 때 각 단계에서 최신 중요도 분포에 따라 적응적으로 최적화하여 모델 훈련 과정을 개선합니다.



### C-3DPO: Constrained Controlled Classification for Direct Preference Optimization (https://arxiv.org/abs/2502.17507)
- **What's New**: 이 논문은 Direct Preference Optimization (DPO) 스타일 알고리즘을 암묵적 분류 알고리즘으로 재구성하는 새로운 관점을 제시한다. 이를 통해 다양한 DPO 스타일 알고리즘을 하나의 분류 프레임워크로 통합하고 확장하는 방법을 제안한다. 분석 결과, DPO 스타일 알고리즘의 기본 문제는 과소 지정(under-specified)되어 있어 승자-패자(winner-loser) 반응의 확률 붕괴(probability collapse)에 취약하다는 점을 강조한다. 이를 해결하기 위해 승자와 패자 간의 확률 질량 이동을 제어하는 제약 조건을 제안하며, 이를 통해 새로운 알고리즘인 C-3DPO를 개발하였다.

- **Technical Details**: 분류 프레임워크는 DPO 스타일 알고리즘을 한데 묶는 데 도움을 주며, 이 알고리즘들이 특정 분류 레이블과 손실 함수(loss function)의 선택에 따라 복원될 수 있음을 보여준다. 이 프레임워크는 기존의 이진 선호 쌍(preference pairs)뿐만 아니라 선호의 순위 리스트(ranked lists)와 같은 보다 풍부한 정보도 수용할 수 있다. 본문에서는 DPO 스타일 알고리즘이 두 개의 확률을 학습하기 위해 단일 제약 조건만 제공하여 과소 지정을 초래한다고 설명하고, 이를 해결할 수 있는 새로운 제약 조건 집합을 제안한다.

- **Performance Highlights**: C-3DPO는 여러 대형 언어 모델을 표준 선호 데이터셋으로 정렬하는 데 있어 기존의 DPO 알고리즘을 초월하는 성능 개선을 제공한다. 연구 결과, C-3DPO는 두 가지 표준 데이터셋과 최대 130억 개 파라미터를 가진 세 세트에서 vanilla DPO와 여러 다른 기준선(baseline)보다 뛰어난 성능을 보였다. 이는 최종 모델 평가에서 높은 품질을 보여주었다.



### RAG-Enhanced Collaborative LLM Agents for Drug Discovery (https://arxiv.org/abs/2502.17506)
Comments:
          Machine Learning, Drug Discovery

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 약물 발견에서 크게 활용될 가능성을 보여주고 있습니다. 그러나 이러한 생화학 데이터의 특수성 덕분에 비용이 많이 드는 도메인 특정 미세 조정이 필요하게 되어, 유연한 LLM의 적용을 방해하고 있습니다. 이에 대한 해결책으로 제안된 CLADD는 정보 검색을 통해 생화학 데이터의 도전 과제를 해결하고 효율적인 정보를 생산할 수 있는 시스템입니다.

- **Technical Details**: CLADD는 여러 LLM 에이전트가 협력하여 생물 의학 지식 기반에서 정보를 동적으로 검색하고, 관련 증거를 통합하여 응답을 생성하는 시스템으로, 도메인 특정 미세 조정이 필요하지 않습니다. 이 시스템은 데이터 이질성, 모호성 및 다원적 통합과 같은 핵심 장벽을 다룰 수 있는 구조를 가지고 있으며, 다른 팀들이 각기 다른 데이터 소스와 역할에 전문화되어 작업을 수행합니다.

- **Performance Highlights**: CLADD는 약물 발견의 다양한 작업에서 우수한 성능을 보여주며, 전통적인 딥러닝 접근 방식 및 특정 도메인 LLMs보다 뛰어난 결과를 시연했습니다. 특히, 이 프레임워크는 유연성과 설명 가능성을 강조하여, 과학자와 AI 간의 상호 작용을 개선합니다. 실험 결과에서는 CLADD의 효과성을 입증하며, 다양한 약물 발견 작업에 적용 가능성을 넓히고 있습니다.



### Doctor-in-the-Loop: An Explainable, Multi-View Deep Learning Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer (https://arxiv.org/abs/2502.17503)
- **What's New**: 이번 연구에서는 비소세포 폐암(NSCLC) 환자의 병리학적 반응(predicted pathological response, pR) 예측의 정확성을 높이고 신뢰성을 확보하기 위해 'Doctor-in-the-Loop'라는 새로운 프레임워크를 제안합니다. 본 프레임워크는 전문적인 도메인 지식을 인공지능 기술에 통합하여, 임상적으로 중요한 해부학적 영역에 모델의 초점을 맞추며, 예측의 해석 가능성과 투명성을 향상시킵니다. 이 방법은 신경망의 훈련 과정에 내재적(explainable AI) 설명성을 부여하여, 예측의 신뢰성과 의료적 맥락의 중요성을 반영합니다.

- **Technical Details**: 연구에서 제안하는 'Doctor-in-the-Loop' 방법론은 다각적(multi-view) 접근 방식을 통해 광범위한 맥락에서 특정 병변(lesion) 세부사항으로 점진적으로 모델의 초점을 조정합니다. 이 과정에서 의료 전문가의 도메인 지식을 학습 과정에 통합하여 치료 예측의 핵심 요소를 더욱 명확히 할 수 있습니다. 또한, 신경망 훈련 시 임상적 인사이트를 반영함으로써 모델이 의료적 통찰과 밀접하게 연결될 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 NSCLC 환자 데이터를 활용하여 제안한 방법이 높은 예측 성능을 보임을 입증하였으며, 투명하고 정당한 결과를 제공합니다. 기존의 최신 연구와 비교했을 때, 'Doctor-in-the-Loop' 접근 방식은 더 높은 예측 정확도와 임상적 관련성을 달성하여, 향후 환자 맞춤형 치료법 개발에 기여할 것으로 기대됩니다. 이러한 결과는 임상적 해설이 가능한 인공지능의 발전에 중대한 이정표가 될 것입니다.



### CoKV: Optimizing KV Cache Allocation via Cooperative Gam (https://arxiv.org/abs/2502.17501)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 주요 문제 중 하나인 메모리 소비를 효율적으로 해결하기 위해 CoKV라는 새로운 방법을 제안합니다. CoKV는 attention heads 간의 협업을 cooperative game으로 모델링하여, 각 head의 기여도를 평가하고 cache budget을 동적으로 할당합니다. 이러한 접근법은 기존의 방법들이 head의 중요성을 독립적으로 평가하는 것에서 벗어나, 협업의 중요성을 강조하는 점이 특징입니다.

- **Technical Details**: CoKV는 cooperative game theory에서 영감을 받아 Shapley value를 사용하여 각 attention head의 중요도를 평가합니다. 이 방법은 각 head의 기여도를 계산할 때, marginal contribution을 넘어서 complementary contribution을 사용합니다. 이러한 방식을 통해 전체 coalition을 평가하는 대신 몇 가지 선택된 coalition sizes에서 기여도를 계산하여 계산 비용을 상당히 줄일 수 있습니다.

- **Performance Highlights**: CoKV는 Llama-3-8B-Instruct와 Mistral-7B 모델을 사용하여 LongBench 벤치마크에서 최첨단 성능을 달성하였습니다. 실험 결과에 따르면, CoKV는 KV cache에 평균 128 KV pair를 유지하면서도 뛰어난 성능을 발휘하였으며, 기존의 기술들과도 잘 통합된다는 점을 보여주었습니다.



### Generalized Exponentiated Gradient Algorithms Using the Euler Two-Parameter Logarithm (https://arxiv.org/abs/2502.17500)
Comments:
          10 pages, preprint of Journal paper

- **What's New**: 이 논문은 Mirror Descent (MD) 접근법을 활용한 새로운 유형의 Generalized Exponentiated Gradient (GEG) 알고리즘을 제안하고, 두 매개변수가 있는 로그의 변형을 정규화 함수로 사용하는 방법을 조사합니다. 이 링크 함수는 일반화된 엔트로피(generalized entropies)와 밀접하게 연관되어 있습니다. 우리는 이 알고리즘이 훈련 데이터의 분포에 적응할 수 있도록 하며, 이를 통해 경량화된 기계 학습 업데이트를 가능하게 합니다.

- **Technical Details**: 제안된 GEG/MD 업데이트는 Euler 로그의 역함수를 근사화하는 일반화된 지수 함수를 추정하여 이끌어냅니다. 이 함수는 다양한 수치적 형태와 특성을 지닌 Euler 로그 및 변형된 지수 함수에 의해 조정 가능합니다. 두 개 이상의 하이퍼파라미터를 통한 학습을 통해 더욱 효과적인 성능 조정이 가능해집니다.

- **Performance Highlights**: 논문에서 개발된 알고리즘은 온라인 포트폴리오 선택(Online Portfolio Selection, OPLS) 문제에 적용되어 성능과 강인성을 개선하는 데 도움이 됩니다. 기존의 표준 EG 알고리즘과 비교할 때, 제안된 알고리즘은 다양한 시장 상황과 투자자의 선호에 효과적으로 적응할 수 있는 잠재력을 보여줍니다.



### Improving Value-based Process Verifier via Structural Prior Injection (https://arxiv.org/abs/2502.17498)
Comments:
          Preprint. Under review

- **What's New**: 이 논문에서는 Large Language Model(LLM) 추론 시나리오에서 Monte Carlo 샘플링을 통한 상태 가치 추정의 한계점을 극복하기 위해 구조적 사전 주입(structural prior injection) 방법을 제안합니다. Monte Carlo 추정이 가진 소음(noise)과 오차(errors)를 사전 정의된 범주형 분포의 기대값으로 매핑하여 해결합니다. 이 접근방식은 근본적으로 추정 오류를 분포 불일치 문제로 전환합니다.

- **Technical Details**: 논문에서는 Markov 결정 과정(Markov Decision Process, MDP) 및 Bellman 방정식을 통해 가치 기반 프로세스 검증기의 개념을 정립합니다. Monte Carlo 방법을 이용하여 상태 행동 가치를 추정하며, 샘플링의 한계로 인해 발생하는 오차를 통계 기반 거리(Statistics-based Distance)라는 새로운 미세 조정을 통해 측정합니다. 이 미세 조정은 다양한 목적 함수(mean-square error, cross-entropy)에 대한 최적화를 도와줍니다.

- **Performance Highlights**: 구조적 사전 주입을 통해 다양한 목표 함수에서 값 기반 프로세스 검증기의 성능을 향상시켜 약 1~2점의 성능 개선을 보여줍니다. 이는 적은 비용으로 실현되며, 동일한 최적 솔루션을 가지고도 서로 다른 구조적 사전(definition)에 따라 성능 차이가 크다는 것을 보여줍니다. 이러한 결과는 구조적 사전 주입이 미래 연구에서 중요한 방향임을 시사합니다.



### Hard constraint learning approaches with trainable influence functions for evolutionary equations (https://arxiv.org/abs/2502.17497)
- **What's New**: 이 논문은 진화 방정식(evolutionary equations)을 해결하기 위한 새로운 딥러닝 접근 방식을 개발했습니다. 이 방법은 순차적 학습 전략(sequential learning strategies)과 훈련 가능한 매개변수(trainable parameters)를 특징으로 하는 향상된 하드 제약 전략을 통합하여, 대규모 시간 영역에서 일반적인 Physics-Informed Neural Networks (PINNs)의 낮은 계산 정확도를 해결합니다. 특히, 시간 간격의 노드에서 PINN 솔루션의 연속성(continuity)과 부드러움(smoothness)을 엄격히 보장하여 초기 시간에서 먼 위치에서도 잘못된 해를 피할 수 있도록 설계되었습니다.

- **Technical Details**: 이 방법은 전체 시간 영역을 여러 개의 하위 구간(subintervals)으로 나누어 각 구간을 순차적으로 해결하는 방식으로, 자연스럽게 인과 원칙(causality)을 준수합니다. 또한, 하드 제약 기능(constrain influence functions)과 훈련 가능한 매개변수를 설정하여 제약 조건(hard constraints) 기술의 효과적인 구현을 위한 실용적인 전략을 제공합니다. 이러한 접근 방식은 진화 방정식의 다양한 요구 사항을 파악하는 데 중점을 두고 있으며, 이론적 및 기술적 지원을 통해 제안된 방법의 보편성(universality)과 계산 정확도(computational accuracy)를 크게 향상시킵니다.

- **Performance Highlights**: 제안된 방법의 성능은 수치 실험(numerical experiments)을 통해 검증되었습니다. 이 방법은 PINN 솔루션의 정확성을 개선함과 동시에 계산 효율성(computational efficiency)을 높이는 데 중요한 역할을 합니다. 또한, 이 연구는 순차 학습 전략에서 제시된 최초의 적응적인 시간 영역 분할 알고리즘(adaptive time-domain partitioning algorithm)을 포함하고 있으며, 이는 전체 시간 영역을 효과적으로 나누는 데 기여하게 됩니다.



### SpikeRL: A Scalable and Energy-efficient Framework for Deep Spiking Reinforcement Learning (https://arxiv.org/abs/2502.17496)
- **What's New**: 이번 연구에서는 SpikeRL이라는 새로운 프레임워크를 소개합니다. SpikeRL은 Spiking Neural Networks(SNNs)와 Deep Reinforcement Learning(DeepRL)의 시너지를 활용하여 복잡한 연속 제어 작업을 위한 효율적이고 확장 가능한 솔루션을 제공합니다. 특히, 우리는 PyTorch Distributed 패키지와 NCCL 백엔드를 사용하여 분산 훈련을 구현하고 혼합 정밀도 훈련(mixed precision training)을 최적화하여 모델의 성능을 개선했습니다.

- **Technical Details**: SpikeRL의 시스템 아키텍처는 DeepRL 기반 SNN 모델, MPI 및 NCCL 백엔드를 통한 분산 훈련, 혼합 정밀도 훈련의 세 가지 주요 구성 요소로 이루어져 있습니다. SNN 모델은 인구 인코딩(population encoding)과 디코딩을 통해 환경의 관측치를 스파이크로 변환하고, 깊은 비평가 네트워크(deep critic network)를 사용하여 액션을 평가합니다. 여기서 스파이크의 인코딩은 가우시안 분포를 사용해 연속 입력 값을 스파이크 트레인으로 변환하여 네트워크의 표현 용량과 연산 효율성을 최적화합니다.

- **Performance Highlights**: 새로운 SpikeRL 구현은 기존의 최신 DeepRL-SNN 방법들과 비교하여 4.26배 빠르고, 2.25배 더 에너지 효율적이라는 성과를 보여주었습니다. 이는 SNNs가 기존의 인공 신경망(ANNs)보다 향상된 성능을 발휘할 수 있도록 하여 복잡한 제어 시나리오에서의 적응성과 정밀도를 극대화하는 데 기여합니다. SpikeRL은 실제 응용 프로그램에서의 복잡한 연속 제어 작업에 진정한 확장 가능하고 지속 가능한 솔루션을 제공함을 입증하였습니다.



### Spatiotemporal Forecasting in Climate Data Using EOFs and Machine Learning Models: A Case Study in Ch (https://arxiv.org/abs/2502.17495)
Comments:
          25 pages, 6 figures

- **What's New**: 이번 연구는 기후 변화가 심한 칠레와 같은 지역에서 효과적인 자원 관리와 환경 계획을 위해 발전된 예측 도구를 제시하고 있습니다. 이 연구는 머신 러닝(Machine Learning, ML) 방법과 정립된 통계 기법을 통합한 혁신적이고 계산적으로 효율적인 하이브리드 방법론을 사용하여 기후 데이터 예측 문제에 접근합니다. 특히, 다양한 ML 모델을 활용하여 중기 범위 예측을 수행하고, 이들 데이터의 시공간 변수를 재구성하는 데 초점을 맞춥니다.

- **Technical Details**: 이 연구에서는 기후 자료를 다루기 위해 6355 지점으로 구성된 그리드를 사용하고 있습니다. 데이터는 1980년부터 2022년까지의 시계열로서, 예를 들어 일일 누적 강수량, 주간 온도 등의 기후 변수를 포함하고 있습니다. 연구의 주요 방법론으로는 임의 직교 함수(Empirical Orthogonal Functions, EOF)와 웨이브렛 분석, 그리고 심층 신경망(Deep Neural Networks, DNN)을 통합하여 시공간 데이터의 차원 감소 및 예측을 수행합니다.

- **Performance Highlights**: 이 방법론은 고차원 다변량 예측 문제를 저차원 단일 시계열 예측 문제로 전환하면서 계산 복잡성을 크게 줄이는 동시에 합리적인 정확도와 유용성을 지닌 예측을 생성합니다. 또한, 동적 시간 왜곡(Dynamic Time Warping, DTW)을 바탕으로 클러스터 분석을 통해 유사한 강수 시계열을 그룹화하며 모델 성능이 향상되는 지역을 효과적으로 식별합니다. 이 연구는 예측 수행이 실질적이고 효율적임을 입증하며, 기후 변동성을 효과적으로 설명하는 패턴을 포착하였습니다.



### Pursuing Top Growth with Novel Loss Function (https://arxiv.org/abs/2502.17493)
Comments:
          30 pages, 7 figures, GitHub repo: this https URL

- **What's New**: 본 논문은 주식 시장에서 길게 지속 가능한 수익을 달성하기 위한 새로운 접근법을 제안합니다. 특히, return-weighted loss function을 통해 머신러닝 모델이 최적의 성장 기회를 발견할 수 있도록 제한된 정보만을 제공하는 방식입니다. 이 시스템은 상장 주식의 공개 데이터와 몇 가지 기술 지표를 기반으로 효율적인 일일 거래 시스템을 제시합니다.

- **Technical Details**: 연구에서 사용된 모델은 Convolutional Neural Networks (CNN)에 기반하며, 주가 변동, 거래량, 모멘텀, 변동성 및 트렌드와 같은 기술적 요소를 분석합니다. 우리는 성과 평과를 더 개선하기 위해 맞춤형 손실 함수와 주식 특징 정보를 결합하는 방식을 사용했습니다. 일일 평가 결과에 따른 투자 점수를 계산하여 거래 전략을 수립하며, 각 주식에 대한 2차원 행렬 형태로 피처를 구성합니다.

- **Performance Highlights**: 최고 모델은 2019년에서 2024년 사이에 연 61.73%의 수익률을 달성하였고, 샤프 비율(Sharpe Ratio)은 1.18에 달합니다. 또한, 2005년부터 2010년 사이의 데이터 분석에서는 연 37.61%의 수익률을 기록했습니다. 연구에서는 제안된 손실 함수가 전통적인 손실 함수들보다 어떻게 우수한지 여러 성과 지표 및 통계적 증거를 통해 입증하였습니다.



### Rapid Parameter Inference with Uncertainty Quantification for a Radiological Plume Source Identification Problem (https://arxiv.org/abs/2502.17492)
- **What's New**: 이 논문에서는 핵사고나 방사선 분산 장치의 폭발과 같은 긴급 상황에서 방사성 물질의 출처를 신속하게 찾는 기법을 제안합니다. 특히, 신경망(neural networks)을 활용하여 방사선 센서로부터 수집된 데이터를 기반으로 방사성 물질의 방출 파라미터를 추정하는 방법을 개발했습니다. 제안된 연구는 Bayesian neural network와 categorical classification neural network의 두 가지 신경망 구조를 사용하여 예측에 대한 불확실성을 정량화합니다.

- **Technical Details**: 연구에서는 방사성 물질의 대기 중 전파를 모델링하기 위해 advection-diffusion equation을 활용합니다. 이를 통해 고정된 위치에서 측정된 방사선 농도를 바탕으로 방사성 오염물의 평균 농도를 계산하며, 재정의된 Gaussian plume 모델을 적용합니다. 이 과정에서 다양한 방출 파라미터와 바람 속도를 고려하여 신경망을 훈련하기 위한 데이터를 생성합니다.

- **Performance Highlights**: Bayesian neural network는 MCMC(Markov chain Monte Carlo) 방법에 비해 계산 비용이 낮습니다. 감지기 측정에 대한 aleatoric(무작위) 및 epistemic(지식적) 불확실성을 모두 고려하여 출처 위치와 방출량에 대한 밀도를 생성합니다. 본 연구 결과는 전달 거부 적응 메트로폴리스 알고리즘(Delayed Rejection Adaptive Metropolis Algorithm)과 비교되었으며, Bayesian 접근법의 효과와 효율성을 강조합니다.



### A generalized dual potential for inelastic Constitutive Artificial Neural Networks: A JAX implementation at finite strains (https://arxiv.org/abs/2502.17490)
Comments:
          56 pages, 19 figures, 3 tables

- **What's New**: 본 논문에서는 비탄성 구성 인공지능 신경망인 iCANN을 위한 일반화된 이중 포텐셜(dual potential) 또는 가상 포텐셜(pseudo potential) 설계 방법론을 제시합니다. 새로운 포텐셜은 응력 불변량(stress invariants)으로 표현되어 대변형(large deformations)에 대한 열역학적 일관성(thermodynamic consistency)을 자연스럽게 만족합니다. 이전 연구와 비교하여, 새로운 포텐셜은 압력 민감 비탄성 압축(pressure-sensitive inelasticity)을 포함한 보다 넓은 스펙트럼의 재료 행동을 포착합니다.

- **Technical Details**: 논문에서는 유한 변형 비탄성(finite strain inelasticity)에 대한 iCANN의 열역학적 프레임워크를 재검토하고, 볼록(convex), 제로-값(zero-valued), 비부정적(non-negative) 이중 포텐셜을 구성하기 위한 조건을 도출합니다. 이를 신경망에 포함시키기 위해 아키텍처 설계를 상세히 설명하며, 미리 열역학에 대한 준수를 보장합니다. 이 프레임워크는 JAX에서 구현되어 있으며, 공개적으로 접근 가능합니다.

- **Performance Highlights**: 제안된 아키텍처의 성능을 평가하기 위해 비스코-탄성(visco-elastic) 재료 행동을 조사하였으며, 이 방법은 비스코-탄성에 한정되지 않습니다. 결과적으로, 새로운 아키텍처는 해석 가능한 모델과 매개변수를 강력히 발견하며, 비탄성도의 정도를 자율적으로 드러냅니다. 이를 통해 비탄성 재료 발견 전략에서 다양한 측면을 검토한 결과, 돋보이는 성능을 확인할 수 있었습니다.



### LLM-Based Design Pattern Detection (https://arxiv.org/abs/2502.18458)
Comments:
          Submitted Version, that was accepted at PATTERNS 2025

- **What's New**: 이 논문에서는 익숙하지 않은 코드베이스에서 디자인 패턴 인스턴스를 자동으로 식별하는 혁신적인 접근법을 제시합니다. 기존의 정적 분석 도구들이 복잡도와 변동성으로 인해 어려움을 겪었으나, 본 연구는 Large Language Models를 활용하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 디자인 패턴 인스턴스 내에서 클래스가 수행하는 역할을 인식하는 데 중점을 둡니다. 이를 통해 코드 구조와 의도를 보다 명확하게 파악할 수 있으며, 코드 내의 암묵적 구현을 유추할 수 있는 기회를 제공합니다.

- **Performance Highlights**: 이 연구의 목표는 개발자에게 더 나은 이해를 지원하고, 리팩토링(refactoring) 및 유지보수(maintenance)와 같은 작업을 간소화하며, 모범 사례(best practices) 준수를 촉진하는 것입니다. 이러한 방식으로 소프트웨어의 품질과 유지 관리 가능성을 향상시키는 데 기여하고자 합니다.



### ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies (https://arxiv.org/abs/2502.18438)
- **What's New**: 이 논문은 ToMCAT(Theory-of-Mind for Cooperative Agents in Teams)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 팀원들의 목표와 행동을 이해하고 예측하는 ToM(Theory of Mind) reasoning을 수행하는 메타-러닝 메커니즘과, 에이전트의 목표와 팀원의 특성에 따라 계획을 생성하는 multiagent denoising-diffusion 모델을 결합합니다. 가장 중요한 점은, 이 시스템이 동적으로 새로운 경로를 샘플링하여 최신 상태에 적응할 수 있도록 설계되었다는 것입니다.

- **Technical Details**: ToMCAT은 두 개의 모듈로 구성됩니다: ToMnet과 MADiff입니다. ToMnet은 다양한 에이전트의 데이터로부터 에이전트의 선호도와 행동에 대한 강력한 사전 정보를 학습하는 신경망입니다. MADiff는 확률적 denoising-diffusion 모델로, 에이전트와 팀원들의 행동을 이해하고 팀원들의 선호도에 적응한 다중 에이전트 계획을 생성합니다. 이러한 설계를 통해 ToMCAT은 에이전트가 자신뿐만 아니라 팀원들의 계획을 예측하고, 팀원들과의 상호작용에서 ToM reasoning을 활용할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, ToMCAT의 동적 재계획 메커니즘이 팀 성과를 유지하면서 자원 사용을 줄이는 데 중요하다는 점이 강조됩니다. 특히, 에이전트가 에피소드 동안 수집한 최근 관찰과 ToM 추론의 결합이 팀원에 적응하여 효과적인 계획을 생성하는 데 필수적입니다. 이는 기존에 제공받은 정보가 없는 상황에서도 효과적으로 팀과의 협력 및 상호작용을 가능하게 합니다.



### Reversal Blessing: Thinking Backward May Outpace Thinking Forward in Multi-choice Questions (https://arxiv.org/abs/2502.18435)
- **What's New**: 이 논문에서는 일반적으로 사용되는 left-to-right (L2R) 자율 회귀(autoregressive) 모델 대신에 right-to-left (R2L) 방식의 효과를 조사합니다. R2L 모형은 MCQ(다중 선택 질문)와 지식 추출, 추론 작업에서 L2R을 능가하는 성과를 보여주며, 다양한 기준에서 다수의 실험을 통해 그 결과를 입증합니다. 또한, 최적의 텍스트 분포 인코딩 방법에 대한 이론적 통찰을 제공합니다.

- **Technical Details**: 이 연구는 L2R과 R2L 모델을 비교하기 위해 동일한 데이터와 컴퓨팅 자원으로 훈련된 모델을 사용합니다. R2L 방식은 예측 손실이 L2R와 유사한 결과를 도출할 수 있는 대칭성(symmetric) 구조를 가지고 있으며, 이로 인해 R2L은 더 적은 근사 오류(approximation errors)를 달성할 수 있습니다. 연구진은 R2L의 성과 차이를 결정하는 여러 요소들—캘리브레이션(calibration), 계산 가능성(computability), 방향 조건 엔트로피(direction conditional entropy)—이 어떻게 상관되어 있는지를 분석합니다.

- **Performance Highlights**: R2L 모델은 여러 MCQ 기준에서 L2R 모델 보다 일관되게 우수한 성능을 기록했습니다. 이 연구의 실험 결과에 따르면, R2L의 접근 방식이 특정 상황에서 더 나은 성과를 내는 이유를 규명하기 위해 다양한 요인들을 배제하고, 어떤 이유로 L2R과 R2L 선택이 결정되는지를 규명하고자 합니다. 결과적으로 이 연구는 LLM(대형 언어 모델)의 능력을 향상시키는 잠재력 있는 방향성을 제시합니다.



### Global law of conjugate kernel random matrices with heavy-tailed weights (https://arxiv.org/abs/2502.18428)
Comments:
          45 pages, 1 figure

- **What's New**: 이번 연구에서는 두 층 신경망 모델에서 발생하는 변환 커널 랜덤 행렬 $YY^	op$의 비대칭 스펙트럼 거동(asymptotic spectral behavior)을 분석합니다. 특히 랜덤 행렬 $W$와 $X$가 독립 동일 분포(i.i.d.)의 항목들을 가지며, $W$의 항목들은 heavy-tailed 분포를 따르는 점에 주목합니다.

- **Technical Details**: 연구에서 $W$는 대칭 $eta$ 안정법칙(symmetric $eta$-stable laws)과 같은 넓은 범위의 heavy-tailed 분포를 포함합니다. 또한, $X$의 항목들은 light-tailed를 따르며, 활성화 함수(activation function) $f$는 비선형(nonlinear)이며 부드럽고(odd) 홀수입니다. 이러한 조건 하에 $YY^	op$의 고유값 분포를 계산하여 그 지표를 도출하였습니다.

- **Performance Highlights**: 연구 결과, heavy-tailed 가중치(weight)가 $Y$의 항목들 간의 강한 상관관계를 유도하며, 이로 인해 light-tailed 가중치를 가진 모델들과는 fundamentally 다른 스펙트럼 거동을 나타냅니다. 이는 신경망 모델의 성능 향상에 기여할 수 있는 중요한 시사점을 제공합니다.



### Rank1: Test-Time Compute for Reranking in Information Retrieva (https://arxiv.org/abs/2502.18418)
- **What's New**: Rank1은 테스트 시간에 계산(compute)을 활용하여 훈련된 최초의 재순위 모델로, 이를 통해 작은 모델의 성능을 신속하게 개선할 수 있는 가능성을 보여줍니다. MS MARCO 데이터셋에서 60만 개 이상의 R1 추론 추적 예제를 수집하고 오픈소스로 공개하여 연구자들이 활용할 수 있게 하였습니다. 이 모델은 고급 추론 및 지시 준수 데이터셋에서 최첨단 성능을 기록하며, 유저 입력 프롬프트에 잘 반응하는 특성을 보여줍니다.

- **Technical Details**: Rank1은 OpenAI의 o1, Deepseek의 R1과 같은 추론 언어 모델을 사용하여, 정보 검색(Information Retrieval) 맥락에서 테스트 시간에 계산(compute)을 활용하도록 설계되었습니다. 모델은 쿼리와 문서를 동시에 추론할 수 있도록 최적화되었으며, 다양한 데이터셋에서 성능을 입증하였습니다. 특히, 이 모델은 Hard negatives와 Easy negatives의 조합을 활용하여 더 나은 학습을 이끌어냈습니다.

- **Performance Highlights**: Rank1은 브라이트(BRIGHT) 벤치마크에서 최첨단 성능을 달성하였고, 전통적인 IR 벤치마크에서 놀라운 성능을 보여주는 등 상당한 향상을 이뤄냈습니다. 사용자가 주는 프롬프트에 대한 적응성도 높아, 다양한 쿼리에 대해 효율적이고 설명 가능한 추론 체인을 제공합니다. 이는 더 나아가 사용자나 에이전트 기반 RAG 시스템이 활용할 수 있는 투명한 시스템으로 발전할 수 있게 합니다.



### GLEAN: Generalized Category Discovery with Diverse and Quality-Enhanced LLM Feedback (https://arxiv.org/abs/2502.18414)
- **What's New**: 이번 연구에서는 GLEAN을 제안하여 Generalized Category Discovery (GCD)의 기존 문제를 해결하고자 합니다. GLEAN은 다양한 방식의 LLM 피드백을 활용하여 데이터 레이블링 과정에서의 효율성을 높이고, 새로운 카테고리의 발견 및 기여를 도모합니다. 이러한 접근법은 기존의 GCD 접근법에 비해 더 나은 성능을 제공합니다.

- **Technical Details**: GLEAN의 핵심 개념은 LLM의 피드백을 세 가지 방식으로 활용하는 것입니다. 첫째, Similar Instance Selection을 통해 모호한 데이터 포인트 간의 유사한 인스턴스를 식별합니다. 둘째, Category Characterization을 통해 새로운 카테고리에 대한 설명을 생성합니다. 셋째, Pseudo Category Selection 및 Alignment을 통해 인스턴스 임베딩을 LLM이 선택한 카테고리 설명과 연결합니다.

- **Performance Highlights**: 실험 결과, GLEAN은 다양한 데이터셋과 성능 지표에서 기존의 최신 모델보다 우수한 성능을 보여주었습니다. 또한, 알려진 카테고리의 수에 따라 GLEAN의 성능을 분석하여 각 구성 요소와 하이퍼파라미터의 효과를 자세히 살펴보았습니다. 이러한 결과는 GLEAN이 GCD 분야에서 중요한 기여를 할 수 있음을 시사합니다.



### AgentRM: Enhancing Agent Generalization with Reward Modeling (https://arxiv.org/abs/2502.18407)
- **What's New**: 본 연구에서는 기존 LLM 기반 에이전트의 일반화 능력을 개선하기 위해, 정책 모델을 직접 미세 조정하는 것보다 보상 모델(reward model)을 미세 조정하여 정책 모델을 안내하는 것이 더 효과적임을 발견했습니다. 이를 통해 AgentRM이라는 일반화 가능한 보상 모델을 제안합니다. AgentRM은 test-time search에서 정책 모델을 효과적으로 가이드하여, 미지의 작업에서도 성능을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 에이전트 작업은 부분 관찰 가능한 마르코프 결정 과정(partially observable Markov decision process, POMDP)로 정형화되며, 다양한 작업 환경에서의 피드백을 기반으로 합니다. 연구에서는 보상 모델을 구축하기 위해 세 가지 접근 방식(1) 명시적 보상 모델링(explicit reward modeling), (2) 암묵적 보상 모델링(implicit reward modeling), (3) LLM을 평가자로 활용하는 방식(LLM-as-a-judge)을 조사하였습니다. AgentRM은 Best-of-N 샘플링과 단계 수준의 빔 검색(beam search)을 통해 응답 생성을 안내합니다.

- **Performance Highlights**: AgentRM은 아홉 개의 다양한 에이전트 작업에서 평균 8.8점의 성능 향상을 이루었으며, 이는 기존의 최고의 일반화 에이전트보다 4.0점 더 높은 성과입니다. 특히 LLaMA-3-70B 정책 모델에서는 12.6점의 더 큰 개선을 보여주어, 강한 일반화 능력을 입증했습니다. 또한, 미세 조정된 정책 모델을 신속하게 증강하여 세 가지 고정된 작업에서 상위 전문화된 에이전트보다 11.4점 더 나은 성능을 달성했습니다.



### The Gradient of Algebraic Model Counting (https://arxiv.org/abs/2502.18406)
Comments:
          Published at AAAI 2025

- **What's New**: 이 논문은 대수적 모델 카운팅(Algebraic Model Counting, AMC)의 관점을 활용하여 통계적 및 관계적 학습(statistical-relational learning)과 신경 유사 AI(neurosymbolic AI)에서의 학습 방법론을 통합하는 새로운 접근법을 제시합니다. 특히, 다양한 학습 알고리즘을 일반화된 그래디언트와 역전파(backpropagation) 기법을 통해 통합하여 좀 더 효율적인 메모리 사용과 더 빠른 학습 속도를 구현했습니다. 이러한 방식으로, 기존의 다양한 알고리즘보다 성능이 크게 향상된 알고리즘을 제공하며, 유연한 학습 도구를 제공합니다.

- **Technical Details**: 대수적 모델 카운팅은 논리 공식의 만족 가능성 문제를 세미링(semi-ring)으로 일반화한 것입니다. 이 연구에서는 그래디언트를 세미링에 따라 일반화하여 다양한 학습 알고리즘을 적용할 수 있는 새로운 대수적 그래디언트 도구를 제안합니다. 이 도구는 경량의 연산을 기반으로 하여 그래디언트 하강법(gradient descent), 기대 최대화(expectation-maximization), 엔트로피 극대화(entropy maximization) 등 여러 알고리즘을 구현할 수 있도록 합니다.

- **Performance Highlights**: 제안된 알고리즘인 대수적 역전파(algebraic backpropagation)는 기존 PyTorch와 Jax보다 몇 배 높은 성능을 보이며, 세미링 특성을 반영하여 더 나은 효율을 구현합니다. 특히, 기존의 알고리즘들이 가진 문제점들을 해결하기 위한 특화된 접근 방식을 제시하며, 실험 결과에서도 유의미한 성능 향상이 확인되었습니다. 이 연구는 또한 2차 정보에 기반한 학습 알고리즘이 어떻게 보다 효율적으로 구현될 수 있는지를 탐구하고 있습니다.



### Learning sparse generalized linear models with binary outcomes via iterative hard thresholding (https://arxiv.org/abs/2502.18393)
- **What's New**: 이번 연구에서는 이진 결과를 가진 희소 일반화 선형 모델(sparse generalized linear models, GLMs)에서 매개변수 추정을 위한 새로운 알고리즘, 이진 반복 경계 설정(Binary Iterative Hard Thresholding, BIHT)을 제안합니다. BIHT는 반복적인 경계 설정(projected gradient descent)을 사용하여 통계적으로 효율적이고 올바른 솔루션으로 수렴함을 증명합니다. 이 알고리즘은 GLM의 링크 함수에 대한 지식이 필요하지 않아 유연성과 일반성을 제공합니다.

- **Technical Details**: BIHT는 로그 손실(ReLU loss)에서의 선형 범위 설정을 통해 작동하며, 높은 차원의 데이터에서 저차원 구조를 활용해 희소성을 모델링하는 데 적합합니다. 이 알고리즘은 로지스틱 회귀(logistic regression)와 프로빗 회귀(probit regression)에 대해 추가 연구가 이루어졌으며, 두 경우 모두에서 BIHT의 성능을 평가하였습니다. 특히, 로지스틱 회귀에서는 알고리즘의 샘플 복잡도가 이전의 하한과 일치하는 통계적 최적성을 달성했습니다.

- **Performance Highlights**: BIHT는 로지스틱 회귀와 프로빗 회귀 모두에서 효과적인 성과를 보이며, 특히 로지스틱 회귀에 있어 모든 노이즈 레짐에서 통계적 최적성을 달성한 최초의 연구로 강조됩니다. 알고리즘의 성능은 기존의 최대 우도 추정(maximum likelihood estimation) 방법들보다 더 유리한 특성을 나타내며, 복잡한 데이터에도 잘 적용될 수 있습니다.



### EgoSim: An Egocentric Multi-view Simulator and Real Dataset for Body-worn Cameras during Motion and Activity (https://arxiv.org/abs/2502.18373)
- **What's New**: 최근 컴퓨터 비전 분야의 egocentric(자기중심적) 작업은 주로 헤드 마운트 카메라에 집중되어 왔으나, 저자는 카메라 기술의 소형화가 다양한 신체 착용 장치에 카메라 통합을 촉진할 것이라고 주장합니다. 이러한 변화는 사람의 움직임 추적, 바디 포즈 추정, 액션 인식 같은 기존 작업에 새로운 관점을 제공하고 특히 하체 부분을 다루는 데 중요한 기여를 할 것입니다. 본 논문에서는 여러 각도가 포함된 신체 착용 카메라의 현장 기반 성능을 구현하는 EgoSim 시뮬레이터와 MultiEgoView 데이터셋을 소개합니다.

- **Technical Details**: EgoSim은 여러 신체 착용 카메라로부터 현실적인 egocentric 렌더링을 생성하는 시뮬레이터로, 실제 모션 캡처 데이터를 사용하여 동작 아티팩트를 렌더링합니다. MultiEgoView 데이터셋은 6개의 신체 착용 카메라에서 촬영된 119시간의 egocentric 영상과 진실 위치를 포함하여 다양한 활동을 구성합니다. 또한, 13명의 참가자가 6개의 GoPro 카메라를 착용해 기록한 5시간의 실제 데이터도 포함되어 있으며, Xsens 모션 캡처 수트를 통해 전체 바디 3D 포즈 참조가 제공됩니다.

- **Performance Highlights**: EgoSim을 활용해 훈련한 비디오 기반 3D 포즈 추정 네트워크의 효과를 입증하였으며, 도메인 갭 분석을 통해 데이터셋과 시뮬레이터가 실제 데이터로의 추론에 얼마나 많은 도움이 되는지를 보여주었습니다. 이는 egocentric 인식 작업을 위한 개방형 연구의 발전에 기여할 것으로 예상됩니다. 전반적으로 EgoSim은 신체착용 카메라를 활용한 다양한 동작 인식을 위한 새로운 연구의 가능성을 열어줍니다.



### DBR: Divergence-Based Regularization for Debiasing Natural Language Understanding Models (https://arxiv.org/abs/2502.18353)
Comments:
          Accepted by SIGKDD Explorations

- **What's New**: 이 논문은 사전 훈련된 언어 모델(PLMs)이 자연어 이해(NLU) 작업에서 단순한 특징과 단축 경로(shortcut) 학습에 의존하는 경향을 보인다는 사실을 조명합니다. 이로 인해 모델이 도메인 외부 데이터(تماد의 OOD) 일반화에 어려움을 겪는다는 점을 강조하고, 이를 해결하기 위해 편차 기반 정규화(Divergence Based Regularization, DBR) 방법을 제안합니다. DBR은 원래 예제와 단축 토큰을 블라인드 처리한 예제 간의 분포 차이를 측정하여 단축 특징에 대한 의존성을 줄이는 접근법입니다.

- **Technical Details**: DBR 방법은 먼저 단축 토큰을 마스킹하여 모델 예측이 단축 특징의 영향을 받지 않도록 하며, 이어서 원래의 예제와 비슷한 예제들을 만들기 위해 정규화 손실을 추가합니다. 이 과정에서 편향 전용 모델을 사용하여 어떤 예제가 실제로 단축 특징에 의존하는지를 판단합니다. 또한 이 모델은 매 에포크마다 다른 마스킹된 예제를 생성하여 모델의 견고성을 향상시킵니다.

- **Performance Highlights**: 세 가지 NLU 작업에서 DBR을 평가한 결과, OOD 성능이 향상되었으며, 도메인 내 정확도에 거의 손실을 주지 않았습니다. 실험 결과는 단축 및 피상적인 특징에 대한 의존성을 줄임으로써 대규모 사전 훈련된 언어 모델의 일반화 능력이 개선될 수 있음을 보여줍니다.



### Graph Inference with Effective Resistance Queries (https://arxiv.org/abs/2502.18350)
- **What's New**: 이 논문은 그래프 추론 문제에서의 새로운 방법론을 제시합니다. 특히, 숨겨진 그래프에 대해 효과적인 저항(effective resistance, ER) 메트릭을 이용한 쿼리를 통해 다양한 그래프 특성을 추론할 수 있는 알고리즘을 개발하였습니다. 이러한 접근은 이전에 많이 연구되지 않았던 영역으로, 그래프 재구성과 속성 테스트에서 새로운 결과를 도출합니다.

- **Technical Details**: 연구에서는 O(n) 쿼리를 사용하여 그래프가 트리인지 여부를 판단하고, 두 그래프가 같은지를 결정하며, 특정 정점이 컷 정점(cut vertex)인지 테스트하는 알고리즘을 제안합니다. 또한, 그래프가 버텍스-이중 연결(vertex-biconnected)인지 또는 엣지-이중 연결(edge-biconnected)인지 검토하는 속성 테스트 알고리즘도 포함되어 있습니다. 마지막으로, 그래프 재구성과 관련된 여러 알고리즘을 제시합니다.

- **Performance Highlights**: 이 논문에서 제안된 ER 쿼리 기반 알고리즘은 각종 그래프 속성을 서브쿼드라틱(subquadratic) 수의 ER 쿼리로 테스트할 수 있게 합니다. 특히, 그래프가 버텍스-이중 연결인지 여부를 O(n/ε) 쿼리를 통해 결정할 수 있는 알고리즘을 제공합니다. 더 나아가, 그래프 재구성 과정에서의 효율성을 높이기 위한 여러 접근 방식을 검토하여, 기존의 알고리즘들과의 비교 분석을 통해 새로운 통찰을 얻고 이를 바탕으로 향후 연구 방향을 제시합니다.



### BRIDO: Bringing Democratic Order to Abstractive Summarization (https://arxiv.org/abs/2502.18342)
Comments:
          13 pages, 1 figure; AAAI-25 Workshop on PDLM camera ready

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 텍스트 요약 과정에서 발생하는 환각(hallucination) 문제를 다룹니다. 특히, 기존의 모델인 BRIO의 노출 편향(exposure bias) 완화 전략을 기반으로 하면서도 환각을 줄이는 데 중점을 둔 새로운 모델 BRIDO를 제안합니다. BRIDO는 후보 요약들의 유사성을 집합적으로 비교하여 후보들을 정렬하는 민주적인 방식을 도입합니다. 또한, XSum과 CNN/DM 데이터셋에서의 실험 결과를 통해 유의미한 성과를 보여주었습니다.

- **Technical Details**: BRIDO는 후보 요약의 유사성에 따라 정렬된 순위를 바탕으로 환각을 완화하는 추상 요약 모델입니다. 모델은 대조학습(contrastive learning)을 통해 높은 상호 후보 ROUGE 점수를 가진 후보를 유도하여 작동합니다. BRIDO의 주요 기여는 후보들 간의 유사성을 고려하여 환각이 적은 후보를 선택하도록 하는 것입니다. 이 접근법은 BRIO의 기법을 확장하며, 대규모 LLM을 탈피한 모델에서 효과적으로 활용될 수 있습니다.

- **Performance Highlights**: BRIDO는 XSum 및 CNN/DM 데이터셋에서 각각 6.25% 및 3.82% 향상된 일관성 G-Eval 점수를 달성했습니다. 이는 BRIO와 비교할 때 환각 측정에서 상당한 개선을 보여주며, 기존 모델에 비해 더 나은 결과를 제공합니다. 이 연구는 환각 문제를 해결하기 위한 새로운 방향성을 제시하며, 대조학습을 통한 접근 방식의 유효성을 입증하고 있습니다.



### Correlating and Predicting Human Evaluations of Language Models from Natural Language Processing Benchmarks (https://arxiv.org/abs/2502.18339)
- **What's New**: 이번 연구는 고성능 대화형 언어 모델(Chat Llama 2)의 평가 전략 변화에 주목하여, 기존의 정량적 NLP 벤치마크와 인간의 평가 간의 관계를 탐구하였습니다. 연구진은 160개의 표준 NLP 벤치마크에서 4개의 모델의 성능과 11,000개 이상의 단일 및 다중 턴 대화에서의 인간 선호도를 비교하였습니다. 그 결과, 많은 NLP 벤치마크가 인간 평가와 강력하게 상관관계를 보이며, 자동화된 지표가 인간의 선호를 예측하는 데 신뢰할 수 있음을 보여주었습니다.

- **Technical Details**: 이 논문에서는 7억, 13억, 34억, 70억 개의 매개변수를 가진 4개의 Chat Llama 2 모델을 평가하였습니다. 연구진은 2조 토큰에 대해 사전 훈련된 모델들을 사용하였고, 슈퍼바이즈드 파인튜닝 및 인간 피드백을 통한 강화 학습으로 후처리하였습니다. 단일 턴 및 다중 턴 대화의 다양성을 고려하여 대화 구조를 설정하였으며, 최첨단 인공지능 모델의 성능 평가를 위한 데이터셋을 구축하였습니다.

- **Performance Highlights**: 모델 간의 성능 비교에서 인간 신뢰성 평가와 여러 NLP 벤치마크 간의 상관관계를 분석했습니다. 인간 평가와 NLP 벤치마크 간의 관계를 선형 회귀 모델을 사용하여 예측했으며, 이로 인해 고비용의 인간 주석 의존도를 줄일 수 있는 가능성을 보여주었습니다. 전반적으로 고전적 벤치마크의 가치를 확립하고, 이러한 지표를 통해 실제 사용자 만족도를 예측할 수 있는 방법을 제시했습니다.



### Self-Supervised Data Generation for Precision Agriculture: Blending Simulated Environments with Real Imagery (https://arxiv.org/abs/2502.18320)
Comments:
          Presented at 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)

- **What's New**: 정밀 농업에서 레이블이 붙은 데이터 부족과 변동성이 큰 데이터 차이에 대한 문제를 해결하기 위해, 저자들은 새로운 합성 데이터 생성 시스템을 제안했습니다. 이 시스템은 Unity 엔진을 기반으로 한 포도원 시뮬레이터를 활용하고, 기하학적 일관성을 고려한 컷앤페이스트 기법을 사용하여 리얼한 이미지를 생성합니다. 이는 다양한 관점과 조명 조건에서 고유한 데이터 샘플을 생성하여 탐지 알고리즘 훈련에 기여합니다. 저자들은 이 방법이 포도 재배에서 획기적인 성능 향상을 보여줌을 입증합니다.

- **Technical Details**: 제안된 시스템은 EU 프로젝트 CANOPIES의 일부로 개발된 3D 포도원 시뮬레이터를 사용합니다. 이 시뮬레이터는 전통적인 포도나무 재배 시스템을 재현하며, 깊이 카메라를 장착한 로봇이 시뮬레이터에서 작업하도록 설계되었습니다. 실제 이미지에서 세분화 마스크를 추출하고 이를 시뮬레이션 환경에 융합하여 더 다양하고 강력한 데이터 분포를 제공합니다. YOLOv5와 Segment Anything Model(SAM)을 활용하여 자동으로 세분화 마스크를 추출하고 이를 데이터 생성 파이프라인에 통합합니다.

- **Performance Highlights**: 이 연구에서는 제안된 방법을 표준 데이터와 증강 데이터를 이용하여 교육한 최첨단 탐지기와 비교 실험을 수행합니다. 실험 결과, 합성 데이터가 탐지 모델의 일반화 성능에 크게 기여하고, 수확 및 모니터링과 같은 농업 작업에서의 적용 가능성을 높임을 보여줍니다. 또한, 기술의 간소화로 인해 농민들도 쉽게 데이터를 생성하고 활용할 수 있도록 설계된 점이 강조됩니다.



### GraphRank Pro+: Advancing Talent Analytics Through Knowledge Graphs and Sentiment-Enhanced Skill Profiling (https://arxiv.org/abs/2502.18315)
- **What's New**: 이 연구는 반구조화된 텍스트에서 정보 추출의 새로운 접근 방식을 제안합니다. Graph(그래프), Natural Language Processing (자연어 처리), 그리고 딥러닝(Deep Learning)을 활용하여, 기존의 복잡한 로직을 Graph 구조로 추상화하여 원시 데이터를 포괄적인 Knowledge Graph(지식 그래프)로 변환합니다. 이러한 혁신적인 프레임워크는 정교한 정보 추출 및 쿼리 작업을 가능하게 합니다.

- **Technical Details**: 우리의 방법론은 이력서에서 개인 프로필, 기술, 조직 소속 및 프로젝트 경험과 같은 세부 정보를 추출하여 해당 정보를 가중치가 있는 그래프 구조로 인코딩합니다. Skill-Project Edge Weighting(기술-프로젝트 엣지 가중치) 기법을 통해 개인의 전문 지식을 식별하고, 그래프 쿼리 기능을 통해 사용자가 효과적으로 후보자를 선별할 수 있도록 지원합니다.

- **Performance Highlights**: 이 시스템은 구직자에게는 타겟 쿼리 기반 필터링 및 정렬 기능을 제공하며, 채용담당자와 커리큘럼 디자이너에게도 혜택을 줍니다. 따라서, 다양한 이력서의 양에 압도되지 않고 정밀한 검색이 가능하게 되어, 인재 분석 및 채용 과정에 대한 혁신을 이룰 수 있습니다.



### Learning atomic forces from uncertainty-calibrated adversarial attacks (https://arxiv.org/abs/2502.18314)
- **What's New**: 본 논문에서는 사용자가 지정한 오류를 통해 적대적 구조를 발견하는 Calibrated Adversarial Geometry Optimization (CAGO) 알고리즘을 제안합니다. 이 알고리즘은 MLIPs의 추정된 불확실성을 실제 오류와 통합하여 신뢰할 수 있는 예측을 제공합니다. 야간학습(active learning) 파이프라인과 통합하여, CAGO가 훈련 구조 수를 수백으로 줄이면서 유체 이온 역학 및 금속-유기 프레임워크에서 물의 흡착을 학습하는 데 성공했음을 입증합니다.

- **Technical Details**: MLIPs를 신경망으로 표현하여 원자들의 잠재 에너지 표면을 학습하는 방식은 양자 역학적 계산을 수시간이 아닌 수 밀리초 만에 예측할 수 있게 해줍니다. Behler-Parinello 네트워크와 같은 선구적인 MLIP들은 이미 구조적 및 열역학적 속성을 잘 포착하고 있으며, NequIP와 같은 최신 방법들은 예측 오류를 현저히 줄이고 있습니다. 그러나 MLIP의 신뢰성을 높이기 위해서는 여전히 많은 실험과 경험이 필요하며, 이는 MLIP에 필요한 일반화 능력을 제한하는 요인이 됩니다.

- **Performance Highlights**: CAGO 알고리즘은 MLIP의 예측 오류를 적절하게 조절함으로써 고차원적이고 복잡한 시스템에서도 안정적인 결과를 보여줍니다. 연구 결과, CAGO가 적용된 MLIP는 액체 물의 구조적, 동역학적 및 열역학적 특성을 수백 개의 훈련 구조로 수렴할 수 있었으며, 이는 기존 연구들에서 필요하던 수천 개의 구조와 비교해 획기적인 개선을 나타냅니다. 따라서, CAGO는 MLIP의 사용을 통한 체계적인 접근 방식을 통해 높은 예측 정확도를 요구하는 문제에서 지속 가능한 해결책을 제공합니다.



### Exploring proteomic signatures in sepsis and non-infectious systemic inflammatory response syndrom (https://arxiv.org/abs/2502.18305)
- **What's New**: 이번 연구는 패혈증(sepsis)과 비감염성 전신 염증 반응 증후군(Non-Infectious Systemic Inflammatory Response Syndrome, NISIRS) 간의 차별적인 단백질 바이오마커(protein biomarker)를 식별하는 것을 목표로 합니다. 새로운 생체표지자(biomarker)를 찾아내어 패혈증의 조기 진단에 기여하고자 하며, 2016-2017년 동안의 관찰 연구 결과를 바탕으로 하고 있습니다.

- **Technical Details**: 연구는 세균 패혈증 환자를 대상으로 한 전향적 관찰 연구로, 질병의 성격을 파악하기 위해 질량 분석(mass spectrometry)을 이용하여 혈장 단백질을 분석했습니다. 이후, 재귀적 특징 제거(recursive feature elimination, RFE) 및 벡터 분류기(vector classifier)로 단백질과 질환의 상관관계를 평가하였습니다.

- **Performance Highlights**: 총 277명의 환자 중 141명은 패혈증을 앓고, 136명은 NISIRS에 해당합니다. RFE 분석 결과 25개의 단백질이 통계적 유의성이 있었으며, 정확도(accuracy) 0.960, 특이도(specificity) 0.920, 민감도(sensitivity) 0.973, AUC는 0.985의 성과를 기록했습니다. 이 중 14개 단백질은 패혈증과 더 큰 연관성을 보였고, 11개 단백질은 NISIRS에서 더 많이 발현되었습니다.



### Neural Network Graph Similarity Computation Based on Graph Fusion (https://arxiv.org/abs/2502.18291)
Comments:
          9 pages, 4 figures, 4 tables

- **What's New**: 본 연구에서는 그래프 유사성 계산을 위한 새로운 접근 방식을 제안합니다. 그래프 쌍의 노드 시퀀스를 단일 대형 그래프로 결합하는 그래프 융합(graph fusion) 기법을 도입하여, 그래프 간의 상호 작용을 쉽게 수행하고, 전역적 주의 메커니즘을 통해 크로스 그래프 인사이트를 수집합니다. 이러한 방법은 이전의 복잡한 계산 부담을 줄이고, 두 그래프 간의 상호 작용을 효율적으로 처리할 수 있게 해줍니다.

- **Technical Details**: 연구에서는 그래프 융합 모듈을 통해 두 그래프의 노드 시퀀스를 하나의 큰 그래프로 통합하여, 트랜스포머 구조와 전역적 주의 메커니즘을 사용해 상호 작용을 인코딩합니다. 이 시스템은 그래프 쌍의 원래 노드 시퀀스에 따라 큰 그래프를 원래 두 그래프로 다시 나누고, 노드 간의 연관성을 고려하여 저차원 노드 간 상호 작용 계산을 수행합니다.

- **Performance Highlights**: 다섯 개의 공개 데이터 세트에 대한 광범위한 테스트 결과, 제안된 모델인 그래프 융합 모델(Graph Fusion Model, GFM)이 그래프-그래프 분류 및 회귀 작업에서 기존의 최첨단 기법들보다 우수한 성능을 보이며, 새로운 성능 및 효율성 기준을 설정했습니다. 이 모델은 그래프 간의 정보를 동시에 상호작용시키는 것이 특징입니다.



### Nested Expectations with Kernel Quadratur (https://arxiv.org/abs/2502.18284)
- **What's New**: 이 논문에서는 중첩 기대값(nested expectations)을 추정하는 새로운 접근 방식을 제안합니다. 기존의 알고리즘인 nested Monte Carlo(NMC)와 multi-level Monte Carlo(MLMC)는 일관성을 지니고 있지만 수렴을 위해 많은 샘플이 필요합니다. 이에 반해, 저자들은 중첩 커널 적분(nested kernel quadrature) 추정기를 사용하는 새로운 추정기를 도입하여 매끄러움이 충분한 경우 기존 방법보다 더 빠른 수렴 속도를 입증하였습니다.

- **Technical Details**: 제안된 알고리즘인 중첩 커널 적분(NKQ)은 NMC의 내부 및 외부 Monte Carlo 추정기를 커널 적분 추정기로 교체합니다. 이는 두 가지 수준의 비가역성을 다루기 위해 필요하며, NKQ는 샘플 수를 크게 줄이는 효과를 얻습니다. 특히 NKQ는 $	ilde{	ext{O}}(	ext{Δ}^{-rac{d_{	ext{X}}}{s_{	ext{X}}}-rac{d_{	heta}}{s_{	heta}}})$의 함수 평가만으로도 원하는 오차를 달성할 수 있음을 보였습니다.

- **Performance Highlights**: 실험적으로 제안된 방법이 실제 애플리케이션, 예를 들어 베이지안 최적화(Bayesian optimisation), 옵션 가격 결정(option pricing), 건강 경제학(health economics) 분야에서 중첩 기대값을 추정하는 데 필요한 샘플 수가 줄어듦을 확인했습니다. 이는 기존 NMC 방법보다 최소한의 오차로도 필요한 계산량을 대폭 감소시킵니다. 따라서 제안된 방법은 다양한 분야에서 고품질의 추정 결과를 얻을 수 있게 합니다.



### Near-Optimal Approximations for Bayesian Inference in Function Spac (https://arxiv.org/abs/2502.18279)
Comments:
          59 pages (26 pages main paper + 33 pages appendices); 6 figures

- **What's New**: 본 연구에서는 reproducing kernel Hilbert space (RKHS)에서 정의된 베이즈 후행분포(Bayes posterior)를 위한 확장 가능한 추론 알고리즘을 제안합니다. 제안된 알고리즘에서는 likelihood function과 prior를 나타내는 Gaussian random element을 사용하여 RKHS 값의 Langevin diffusion의 정적 분포로부터 베이즈 후행분포를 추정합니다.

- **Technical Details**: 이 알고리즘은 무한 차원의 Langevin diffusion을 Kosambi-Karhunen-Loève 확장의 첫 번째 $M$ 성분으로 투영하는 방법을 포함합니다. 그런 다음, 총 확률 법칙과 충분조건을 이용하여 이 $M$ 성분에 대한 근사 후행분포를 기반으로 $	ext{Pi}_{	ext{B}}$의 추론을 수행합니다. 결과적으로 이 방법의 계산 복잡도는 $O(M^3+JM^2)$으로, 여기서 $J$는 후행 분포에서 생성된 샘플의 수를 나타냅니다.

- **Performance Highlights**: 이 알고리즘은 sparse variational Gaussian process (SVGP)의 후행분포를 특별한 경우로 회복합니다. 그러나 SVGP는 Gaussian process로 매개변수화가 제한되지만, 본 방법은 모든 확률 분포를 포함하는 비매개변수 변이 가족 $	ext{P}(	ext{R}^M)$을 기반으로 하고 있습니다. 결과적으로, 제안된 방법은 음의 로그 likelihood가 볼록(Convex)하고 Lipschitz 연속인 경우 Bayes posterior $	ext{Pi}_{	ext{B}}$의 최적 $M$ 차원 변이 근사에 수학적으로 가까워집니다.



### Near-optimal Active Regression of Single-Index Models (https://arxiv.org/abs/2502.18213)
- **What's New**: 이 연구는 활성 회귀 문제에 대한 새로운 접근 방식을 제시합니다. 일반적인 회귀 문제에 비해 레이블 접근 비용을 줄이면서도 문제를 해결하는 방법을 다룹니다. 특히, $f$가 Lipschitz 함수일 때도 $(1+	ext{ε})$-근사 솔루션을 제공하는 알고리즘이 처음으로 제안되었습니다.

- **Technical Details**: 작업은 $A$가 완전하게 접근 가능하고, 벡터 $b$는 항목 쿼리를 통해서만 접근할 수 있는 설정에서 수행됩니다. 쿼리 복잡도는 $	ilde{O}(d^{rac{p}{2}	ext{∨}1}/	ext{ε}^{p	ext{∨}2})$로, 이는 최적의 쿼리 수를 제공합니다. 다양한 $p$ 값에 따라 최적성을 보장하며, 특히 $p>2$일 경우 쿼리 복잡도 의존성이 최적임을 보여줍니다.

- **Performance Highlights**: 제안된 알고리즘은 활성 회귀 문제의 다양한 상황에서 성능을 제공하며, 특히 $p=2$ 및 $p=1$에 대한 기존 연구 결과와 비교하여 개선된 쿼리 복잡도를 보여줍니다. 이러한 결과는 레이블 접근을 최소화하려는 실제 문제에 중요하며, 특히 머신 러닝과 신경망 분야에 널리 적용될 수 있습니다.



### Recurrent Neural Networks for Dynamic VWAP Execution: Adaptive Trading Strategies with Temporal Kolmogorov-Arnold Networks (https://arxiv.org/abs/2502.18177)
- **What's New**: 이 논문에서는 현대 금융 시장에서 Volume Weighted Average Price (VWAP) 주문의 실행 문제를 해결하기 위한 새로운 동적 신경망 접근법을 제시합니다. 이전 연구에서 제안된 정적 모델의 한계를 극복하기 위해, 실시간으로 시장 조건에 적응할 수 있는 동적 네트워크 프레임워크를 개발하였습니다. 특히, 혁신적인 재귀 신경망(recurrent neural networks) 통합과 함께 시장 피드백 기반의 동적 조정 메커니즘을 도입하였습니다.

- **Technical Details**: 동적 VWAP 프레임워크는 시장의 복잡한 시간적 의존성을 포착하기 위해 재귀 신경망을 활용합니다. 이를 통해 VWAP 주문을 실행하는 과정에서 요구되는 의사결정을 지속적으로 최적화할 수 있습니다. 또한, 이 성과는 다섯 개 주요 암호화폐 시장에서의 경과적 분석을 통해 검증되었으며, 전통적인 실행 방법과 비교하여 실질적인 성능 향상을 이루었습니다.

- **Performance Highlights**: 이 동적 접근 방식은 유동성 있는 시장에서 10%에서 15%의 실행 성능 향상을 달성하며, 다양한 조건에서도 일관된 성과를 보여 주었습니다. 이러한 결과는 적응형 신경망 구조가 현대 VWAP 실행의 도전 과제를 효과적으로 해결할 수 있음을 시사하며, 실제 배포에 적합한 계산 효율성을 유지합니다.



### Inverse Materials Design by Large Language Model-Assisted Generative Framework (https://arxiv.org/abs/2502.18127)
- **What's New**: 이 논문에서는 효율성과 정확성을 개선하기 위해 AlloyGAN이라는 폐쇄 루프(framework)를 도입했습니다. AlloyGAN은 대형 언어 모델(LLM) 지원 텍스트 마이닝과 조건부 생성적 적대 신경망(CGAN)을 통합하여 데이터 다양성을 증대시키고 역설계(inverse design)를 개선합니다. 이를 통해 AlloyGAN은 새로운 합금(Tab) 물질의 발견을 가속화할 수 있는 확장 가능한 접근 방식을 제공합니다.

- **Technical Details**: AlloyGAN의 프레임워크는 네 가지 상호 연결된 구성 요소로 이루어져 있어 포괄적이고 반복적인 접근 방식을 제공합니다. 첫 번째는 LLM을 이용한 텍스트 마이닝 단계로, 화학 전문 프롬프트를 활용하여 관련 합금 조성 및 특성 데이터를 추출합니다. 두 번째로, 수집된 데이터는 화학 특성의 세트를 생성하는 데이터 전처리 과정을 거칩니다. 그런 다음, AlloyGAN에 최적화된 데이터셋을 입력하여 GAN 학습을 진행하고, 마지막으로 생성된 합금을 실험적으로 검증하여 피드백 루프를 생성합니다.

- **Performance Highlights**: AlloyGAN은 금속 유리(metallic glass)의 열역학적 특성을 예측하며 실험값과의 불일치를 8% 미만으로 유지하여 강력한 성능을 입증했습니다. CGAN을 활용하여 조건부 정보를 통합함으로써 합금 조성을 보다 정확하게 설계할 수 있으며, 검증된 데이터는 데이터베이스에 통합되어 지속적으로 모델의 예측 정확성을 향상시킵니다. 따라서 AlloyGAN은 합금 설계를 위한 중요한 플랫폼으로 자리 잡을 것으로 기대됩니다.



### Controlling dynamics of stochastic systems with deep reinforcement learning (https://arxiv.org/abs/2502.18111)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문에서는 심층 강화학습(deep reinforcement learning)을 활용하여 확률 시스템의 동적 제어를 위한 새로운 시뮬레이션 알고리즘을 제안합니다. 기존의 제어이론과 심층 강화학습 간의 간극을 좁히면서, 인공신경망(artificial neural networks)을 사용하여 시스템의 동작을 조절할 수 있는 가능성을 explore합니다. 이를 통해 확률 과정의 동적 제어에서 신경망이 어떻게 유용한지를 보여줍니다.

- **Technical Details**: 제안된 방법론은 에이전트 기반 시뮬레이션을 활용하여 신경망이 local state-to-state 전이를 유도하는 제어기로 작용한다는 점이 특징입니다. 파티클 융합(particle coalescence)과 비대칭 배제 과정(totally asymmetric exclusion process)을 포함한 두 가지 확률 프로세스에서 동적 제어의 효과를 확인합니다. 특히, 강화학습의 정책(policy) 구현 및 Markov 결정 프로세스(MDP) 프레임워크를 통한 보상 구조(reward structure) 최적화 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, 원래의 반응 메커니즘을 신경망 제어기로 대체함으로써 원하는 동적 행동을 유도하는 새로운 입자 간 상호작용을 개발한 것과 유사한 결과를 얻었습니다. 그러나 multi-particle 시스템에서 랜덤하게 선택된 에이전트의 업데이트로 인해 제어 효과성은 제한적이었습니다. 다양한 보상 구조를 적용함으로써 더욱 복잡한 제어 목표를 달성하는 성과를 보였습니다.



### Examining the Threat Landscape: Foundation Models and Model Stealing (https://arxiv.org/abs/2502.18077)
Comments:
          Accepted to BMVC 2024

- **What's New**: 이 논문에서는 이미지 분류 API가 모델 도용 공격에 취약하다는 새로운 발견을 제시합니다. 특히, Foundation Models (FMs)에서 미세 조정(fine-tuning)된 모델이 전통적 비전 아키텍처보다 더 높은 도용 위험을 갖는다고 주장합니다. 이는 FMs가 시각적 패턴 및 기능을 폭넓게 잠재적 사용자와 공격자에게 제공하기 때문입니다.

- **Technical Details**: 연구팀은 세 가지 데이터셋과 여러 모델을 통해 FMs에 기반한 이미지 분류 API의 모델 도용 취약성을 체계적으로 조사했습니다. 결과적으로 ViT 모델에서 미세 조정된 모델들이 ResNet과 같은 소규모 모델에 비해 도용 위험이 더 크다는 것을 발견했습니다. 연구에서 도난당한 모델의 예측과 일치율도 보고되었고, ViT-L/16 모델이 가장 높은 일치율을 기록했습니다.

- **Performance Highlights**: 이 연구는 MLaaS 설정에서 상업적 API 배포 시 FMs 사용의 위험성을 경고합니다. 연구진은 높은 정확도와 보안 문제 사이의 트레이드오프를 강조하며, 이러한 모델의 안전성을 확보하기 위한 강력한 보안 프로토콜 및 대응책 필요성을 도출했습니다. 이 결과는 상업적으로 활용되는 API에서 모델 도용을 방지하기 위해 더 나은 선택을 요구합니다.



### Golden Ratio Mixing of Real and Synthetic Data for Stabilizing Generative Model Training (https://arxiv.org/abs/2502.18049)
- **What's New**: 이번 논문에서는 모델 붕괴(model collapse)라는 현상을 다루고 있습니다. 이는 이전 모델이 생성한 데이터로 훈련된 모델들이 성능 저하를 겪는 문제로, 생성 모델 연구에서 중요한 도전 과제가 되고 있습니다. 제안된 프레임워크를 통해 실제 데이터와 합성 데이터(synthetic data)를 결합하여 반복적으로 훈련하는 방식이 탐구되었습니다.

- **Technical Details**: 우리는 여러 시나리오에서 가중(training scheme) 훈련 방식을 평가하여 최적의 훈련 전략을 개발하고자 했습니다. 이를 통해 합성 데이터의 혼합 비율(mixing proportion)과 가중치(weighting scheme)가 최종 모델 성능에 미치는 영향을 이론적으로 분석하였습니다. 이론적인 분석 결과, 합성 데이터의 비율에 따른 최적 가중치는 통일된 표현(unified expression)을 따라가는 경향이 있음을 발견했습니다.

- **Performance Highlights**: 특히 일부 경우에서, 실제 데이터에 할당된 최적 가중치는 황금비(golden ratio)의 역수에 상응하는 것으로 나타났습니다. 이러한 이론적 결과는 매우 다양한 시뮬레이션 데이터셋과 실제 테이블 데이터셋에서 검증되었습니다. 본 연구는 생성 모델 성능 향상과 합성 데이터 활용의 근본적인 균형을 드러내는 중요한 기초 자료가 됩니다.



### Progressive Local Alignment for Medical Multimodal Pre-training (https://arxiv.org/abs/2502.18047)
- **What's New**: 이 논문에서는 Progressive Local Alignment Network (PLAN)을 제안하여 의료 이미지와 텍스트 간의 국소 정렬(local alignment)을 개선합니다. PLAN은 대조 학습(contrastive learning)을 통해 단어와 픽셀의 관계를 형성하고 점진적 학습(progressive learning) 전략을 통해 이러한 관계를 반복적으로 정제하여 정렬의 정확성과 강인성을 높이는 방식입니다. 이는 기존의 의료 이미지 정렬 방법들이 극복하지 못했던 불규칙한 경계를 가진 구조를 더 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: PLAN의 접근 방식은 대조 학습을 통해 단어-영역 관계 학습을 두 개의 상호 연결된 하위 문제로 설정합니다. 첫 번째 하위 문제는 단어와 픽셀 간의 관계를 사용하여 지역 정렬(local alignment)을 수행하고, 두 번째는 방사선 전문의의 진단 추론에 영감을 받은 점진적 학습 전략으로, 지역 정렬 관계를 반복적으로 향상시키는 것입니다. 이 과정에서 다양한 의료 데이터셋을 사용하여 PLAN의 효과iveness를 평가하고 있습니다.

- **Performance Highlights**: 실험 결과, PLAN은 구문 기초(pharse grounding), 이미지-텍스트 검색(image-text retrieval), 객체 감지(object detection) 및 제로샷 분류(zero-shot classification)와 같은 여러 작업에서 기존 최첨단 방법들을 초월하여 새로운 기준을 세웠습니다. PLAN은 특히 세밀한 영역과 텍스트 간의 연관성을 강화하여 임상 결정 지원(clinical decision support)을 향상시킵니다. 이에 따라 의학 분야에서의 실세계 응용 가능성이 높아지고 있습니다.



### AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages (https://arxiv.org/abs/2502.18020)
- **What's New**: 이 논문에서는 다국어 모델을 위한 새로운 하이브리드 지식 증류(knowledge distillation) 접근법을 제안합니다. 기존의 방법들이 다국어 모델, 특히 저자원(low-resource) 언어에 대한 성능 유지를 어렵게 하는 문제를 해결하기 위해, 전통적인 지식 증류 방식과 단순화된 attention matching 메커니즘을 결합했습니다. 이로 인해, 전통적인 다국어 모델보다 유의미하게 작은 크기의 학생 모델 구조를 도입하여, 아프리카 언어 다섯 개에 대한 평가에서 효과를 입증했습니다.

- **Technical Details**: 우리가 제안한 하이브리드 증류 프레임워크는 지식 증류와 attention matching을 결합하여 학생 모델이 교사 모델의 출력 분포와 내부 attention 패턴을 모두 학습하도록 합니다. 특히, 매우 compact한 다국어 학생 모델을 설계하였으며, 이는 기존 모델보다 훨씬 작은 hidden dimension을 가집니다. 실험적으로, 아프리카어 저자원 언어인 Kinyarwanda, Swahili, Hausa, Igbo, Yoruba에 대해 이 접근법을 평가하고, 모델 크기를 85% 이상 줄이면서도 경쟁력 있는 성능을 달성했습니다.

- **Performance Highlights**: 이 연구의 실험 결과는 제안된 하이브리드 접근법이 교사 모델과 비교하여 성능 면에서 경쟁력을 유지함을 보여줍니다. 학생 모델은 원래 모델의 성능에서 85% 이내의 정확도를 유지하면서도, 연산 자원을 현저히 절감할 수 있었습니다. 이는 저자원 환경에서 다국어 모델을 배치하는 데 유용한 실용적인 프레임워크를 제공하며, 아프리카 언어와 관련된 응용 프로그램에서 큰 혜택을 제공합니다.



### GNN-XAR: A Graph Neural Network for Explainable Activity Recognition in Smart Homes (https://arxiv.org/abs/2502.17999)
Comments:
          This is a preprint. Paper accepted for publication at the 21st EAI International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (Mobiquitous)

- **What's New**: 이번 연구는 스마트 홈 환경에서 ADLs(Activities of Daily Living) 인식을 위해 명확히 설계된 최초의 설명 가능한 그래프 신경망 시스템(GNN-XAR)을 소개합니다. GNN-XAR는 환경 센서 데이터의 시간 창에서 동적으로 그래프를 구성하여 공간적(spatial) 및 시간적(temporal) 특성을 고려합니다. 이 시스템은 각 예측에 대해 자연어 설명을 생성하며, 기존의 최첨단 방법들보다 우수한 설명을 제공합니다.

- **Technical Details**: GNN-XAR는 이진 환경 센서 데이터를 사용하여 공휴일(이벤트)와 그 상호작용의 맥락에서 활동을 인식합니다. 이 시스템은 그래프 구축 모듈을 통해 시간 창을 그래프 형태로 변환하며, 각 그래프는 그래프 신경망 모듈을 통해 분석됩니다. 설명 생성 모듈은 후속 XAI(설명 가능한 AI) 방법을 활용하여 노드와 간선의 중요성을 분석하고 이를 바탕으로 비전문 사용자가 이해할 수 있는 자연어 설명을 제공합니다.

- **Performance Highlights**: GNN-XAR는 두 개의 공개 데이터 세트에서 테스트되었으며, 기존의 설명 가능한 HAR 방법들보다 더 유용한 설명을 제공하면서도 인식률이 소폭 향상된 결과를 나타내었습니다. 이를 통해 스마트 홈 환경 내에서 인간 활동 인식의 신뢰성 및 투명성을 증대시킬 수 있는 가능성이 확인되었습니다.



### Shedding Light on the Polymer's Identity: Microplastic Detection and Identification Through Nile Red Staining and Multispectral Imaging (FIMAP) (https://arxiv.org/abs/2502.17997)
Comments:
          20 pages (with additional supplementary material), 5 Figures, 3 Tables

- **What's New**: 이 연구에서는 미세 플라스틱(microplastics, MPs)의 검출과 식별에 대한 새로운 접근 방법으로 Fluorescence Imaging Microplastic Analysis Platform (FIMAP)을 소개합니다. FIMAP은 네 개의 광학 필터와 다섯 개의 자극 파장을 갖춘 개선된 다중 분광 카메라로, 기존의 검출 기술보다 더 높은 정확도를 제공합니다. 이를 통해 다양한 미세 플라스틱 재질의 형광 행동을 포괄적으로 분석할 수 있게 되었습니다.

- **Technical Details**: FIMAP는 HDPE, LDPE, PP, PS, EPS, ABS, PVC, PC, PET, PA 등 10종의 Nile Red로 염색된 미세 플라스틱에 대한 정밀한 형광 분석을 수행합니다. 이 플랫폼은 K-means clustering을 사용하여 안정적인 세분화(Intersection over Union = 0.877)를 가능하게 하고, 20차원 색상 좌표 다변량 최근접 이웃(multi-dimensional nearest neighbor) 기법을 통해 미세 플라스틱을 분류합니다. 성능 지표로는 90%의 정밀도, 90%의 정확도, 100%의 재현율 및 F1 점수 94.7%를 달성했습니다.

- **Performance Highlights**: FIMAP는 특히 대규모 환경 샘플에서 미세 플라스틱을 식별하고 분류하기 위한 자동화된 고처리량(framework) 솔루션으로 제시됩니다. 그러나 35-104 마이크론 크기의 작은 미세 플라스틱에 대한 분류 정확도는 감소하여, 이는 염료 흡수 감소나 감지 가능한 픽셀 수의 부족과 관련이 있습니다. 앞으로 고배율 기기(예: 현미경)와 FIMAP의 통합이 미세 플라스틱 식별을 향상시킬 가능성이 있습니다.



### MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification (https://arxiv.org/abs/2502.17987)
- **What's New**: 이 논문에서 새로운 점은 저자들이 저자원 (low-resource) 반투 (Bantu) 언어에 특화된 텍스트 분류 모델 MAGE(Multi-Head Attention Guided Embeddings)를 소개한 것이다. 이 모델은 Language-Independent Data Augmentation (LiDA)을 통해 데이터 포인트를 선택적으로 향상시킴으로서 텍스트 분류 성능을 개선하는데 중점을 두고 있다. 특히, MAGE는 데이터 부족 문제를 해결하면서도 반투 언어의 고유한 구문론적(syntactic) 및 의미론적(semantic) 특성을 효과적으로 다루도록 설계되었다.

- **Technical Details**: MAGE는 LiDA 프레임워크를 기반으로 하여 중대한 혁신을 도입하고 있다. 전통적인 Denoising Autoencoder 대신 Variational Autoencoder (VAE)를 도입하여 더욱 표현력이 풍부하고 다양한 합성(augmented) 임베딩을 생성한다. 또한, Multi-Head Attention 메커니즘을 활용하여 임베딩에서 중요한 특징을 강조함으로써 저자원 언어에서 구문론적 및 의미론적 뉘앙스를 더 잘 포착할 수 있도록 한다.

- **Performance Highlights**: MAGE는 AfriSenti SemEval 데이터셋을 사용하여 감정 분류(sentiment classification) 성능 평가를 수행하였으며, 저자원 환경에서 기존의 기준 방법들보다 우수한 성능을 나타냈다. MAGE는 데이터 부족 문제를 해결함과 동시에 다른 저자원 언어 계열로의 텍스트 분류 능력을 확장할 수 있는 스케일러블한 프레임워크로 자리잡고 있다. 이 연구는 향후 저자원 언어 처리 및 분류 작업에 대한 연구의 기초를 제공하고 있으며, NLP 기술의 포괄성과 일반화 가능성을 높이는 방향으로 나아가고 있다.



### Optimal Brain Apoptosis (https://arxiv.org/abs/2502.17941)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문은 Convolutional Neural Networks (CNN)와 Transformers의 계산 효율성과 자원 요구 사항 문제를 다루기 위해 Optimal Brain Apoptosis (OBA)라는 혁신적인 pruning 방법을 제안합니다. 기존의 Hessian 행렬을 근사하는 방법 대신, 각 매개변수에 대한 Hessian-vector 곱을 직접 계산하여 매개변수 중요성을 추정합니다. 이 접근 방식은 Neural Network의 파라미터를 보다 정밀하게 다룰 수 있도록 하며, 다양한 데이터셋에서 CNN과 Transformers의 실험을 통해 검증되었습니다.

- **Technical Details**: OBA 방법은 매개변수의 2차 Taylor 전개를 효율적으로 계산하는 것을 목표로 합니다. 논문에서는 네트워크의 다양한 층 간 Hessian 하위 행렬을 분해하고, 이들 간의 비어있지 않은 조건을 식별하는 과정을 제시합니다. 이러한 방법을 통해 OBA는 구조적 및 비구조적 pruning 작업 모두에서 활용될 수 있을 뿐 아니라, 매개변수의 중요성을 정량화하는 데 더 정확한 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과, OBA는 VGG19, ResNet32, ResNet50, ViT-B/16 모델에서 CIFAR10, CIFAR100, Imagenet 데이터셋을 통해 기존 pruning 방법에 비해 성능 저하 없이 계산 효율성을 크게 향상시켰습니다. 이 연구는 효율적인 매개변수 중요성 추정의 새로운 접근법을 통해 deep learning 모델의 경량화와 최적화를 가능하게 했습니다. 논문에서 제공하는 코드를 통해 다른 연구자들도 이 방법을 쉽게 구현할 수 있도록 하고 있습니다.



### Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data (https://arxiv.org/abs/2502.17928)
- **What's New**: 이 논문에서는 SIDSL(Structure-prior Informed Diffusion model for Source Localization)이라는 새로운 프레임워크를 소개하고, 이를 통해 제한된 데이터 환경에서 소스 로컬라이제이션(source localization)의 세 가지 주요 문제를 해결합니다. 주목할 만한 문제는 알려지지 않은 전파 패턴, 복잡한 토폴로지-전파 관계 및 소스와 비소스 노드 간의 클래스 불균형입니다. 이러한 객체들을 해결하기 위해 SIDSL은 그래프 레이블 전파(graph label propagation)를 통한 토폴로지 인식(prior) 사전 정보를 통합하여 정확한 소스 예측을 지원합니다.

- **Technical Details**: SIDSL은 입력으로 그래프의 토폴로지와 관측된 노드 상태를 사용하며, 구조 기반의 사전 정보를 안내하는 조건부 디노이징(diffusion) 모델을 사용하여 소스 노드 분포를 예측합니다. 이를 통해 노드의 감염 경로를 추적하는 레이블 전파와 효율적인 토폴로지 인식 피쳐 추출을 위한 GNN(parameterized label propagation module, GNN-LP)을 결합하여 학습 효율성을 극대화합니다. 이러한 방법들은 실험적으로 네 가지 실제 데이터 세트에서 SIDSL이 최첨단 기법들보다 더 우수한 성능을 발휘함을 보여줍니다.

- **Performance Highlights**: SIDSL은 네 가지 실제 데이터셋에서 F1 스코어가 7.5-13.3% 향상되는 성과를 기록하였으며, 합성 패턴의 시뮬레이션 데이터로 사전 훈련을 시행했을 때, 훈련 데이터의 10%만으로도 18.8% 이상의 성과를 얻었습니다. 이러한 결과는 SIDSL이 레이블 데이터가 부족한 실제 응용에서도 높은 효과성을 보여주며, 제한된 데이터로도 강력한 일반화 능력을 가진다는 것을 강조합니다. 이 연구는 SIDSL의 샘플 효율성을 향상시키고 훈련 시간을 단축할 수 있는 잠재력을 보여줍니다.



### BD Currency Detection: A CNN Based Approach with Mobile App Integration (https://arxiv.org/abs/2502.17907)
- **What's New**: 이 연구에서는 전통적인 수동 검증 및 옵티컬 스캐닝 방식의 한계를 극복하기 위한 향상된 화폐 인식 시스템을 도입했습니다. Convolutional Neural Networks (CNNs)를 활용하여 방글라데시 은행권을 정확하게 분류할 수 있는 방법을 제시합니다. 이를 통해 50,334개의 이미지를 사용하여 훈련된 CNN 모델은 98.5%의 정확도를 기록하며 기존 방식보다 우수한 성능을 보였습니다.

- **Technical Details**: 데이터셋은 50,334개의 이미지로 구성되어 있으며, 모델 훈련을 위해 선처리(preprocessing) 및 CNN 모델 최적화를 통해 높은 성능을 발휘하도록 설계되었습니다. 모델은 TensorFlow Lite 형식으로 변환되어 Android 모바일 애플리케이션에 통합되어 실시간(real-time) 및 오프라인 기능을 지원합니다.

- **Performance Highlights**: 이 연구의 결과는 깊은 학습(deep learning)이 화폐 인식에 미치는 효과성을 강조하고 있습니다. 빠르고 안전하며 접근 가능한 솔루션을 제공하여 금융 거래 및 보조 기술을 향상시키는 데 기여하고 있습니다.



### Sample-efficient diffusion-based control of complex nonlinear systems (https://arxiv.org/abs/2502.17893)
- **What's New**: 이번 논문은 복잡한 비선형 시스템을 제어하기 위한 새로운 프레임워크인 SEDC(Sample-Efficient Diffusion-based Control)를 제안합니다. SEDC는 샘플 효율성과 신뢰성을 높이기 위해 데이터 기반 제어 방식을 혁신적으로 개선합니다. 제안된 방법은 Decoupled State Diffusion, Dual-Mode Decomposition, Guided Self-finetuning을 통해 기존 방법보다 39.5%-49.4% 더 나은 제어 정확성을 달성하며, 훈련 샘플의 10%만 사용합니다.

- **Technical Details**: 이 연구는 비선형 시스템의 제어 문제를 디노이징 확산 프로세스로 재구성하여 최적의 제어 시퀀스를 샘플링합니다. 이를 통해 고차원 공간의 데이터 희소성 문제를 해결하기 위해 Decoupled State Diffusion(DSD) 접근 방식을 사용하여 물리학 기반 제어를 보장합니다. Dual-Mode Decomposition(DMD)은 시스템 동역학을 계층적 선형 및 비선형 구성 요소로 분해하여 비선형 제어 전략을 배우는 데 도움을 줍니다.

- **Performance Highlights**: SEDC는 세 가지 복잡한 비선형 동적 시스템에서 실험을 통해 기존의 방법과 비교하여 제어 정확성이 39.5%-49.4% 향상됨을 입증했습니다. 에너지 소비와 정확성 사이의 균형을 유지하면서도 훈련 샘플의 10%만으로 최신 성능을 달성합니다. 추가적인 압축 연구는 SEDC의 설계 요소들의 효과성을 확인합니다.



### Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation (https://arxiv.org/abs/2502.17839)
Comments:
          16 pages, 2 figures

- **What's New**: 이번 논문에서는 Retrieval-augmented generation (RAG) 프레임워크에 실용적인 원칙을 주입하여 검색된 컨텍스트의 유용성을 향상시키는 간단하고 비지도 학습 방식의 방법을 제안합니다. 이 방법은 RAG에 의해 검색된 문서 풀에서 질문과 가장 관련이 높은 문장을 식별하고, 입력 질문에서 다루어진 모든 주제를 포함하되 그 이상은 포함하지 않도록 하며, 이러한 문장을 원래의 맥락 내에서 강조 표시한 후 LLM에 제공합니다. 실험을 통해, 이 접근 방식이 세 가지 질문 답변 작업(ARC-Challenge, PubHealth, PopQA)에서 일관된 개선 효과를 보여준다는 사실을 입증하였습니다.

- **Technical Details**: RAG는 큰 언어 모델(LLMs)의 제한된 지식 수평을 해결하기 위해 등장하였습니다. 본 연구는 RAG가 종종 LLM에 너무 많은 정보를 제시하여, 이를 통해 효과적인 소통을 위한 Grice의 네 가지 격률을 위반하는 경우가 많다고 주장합니다. 제안된 방법은 비지도 학습 기반의 휴리스틱을 사용하여 Grice의 격률을 구현하고, RAG에서 주제에 맞는 문장들을 식별함으로써 성능을 향상시키는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 PubHealth에서 19.7% 및 ARC-Challenge에서 10%의 상대적인 정확도 향상을 달성하였습니다. 제안된 접근 방식은 또한 작은 언어 모델인 AMD-OLMo-1B-SFT에서 10%의 개선 효과를 보였으며, RAG의 성능 향상에 있어 상용화된 검색기인 DPR과 함께 사용될 때 20%까지 성능을 개선할 수 있는 가능성을 제시합니다.



### TagGAN: A Generative Model for Data Tagging (https://arxiv.org/abs/2502.17836)
- **What's New**: 이 연구는 기존 AI 시스템이 가지는 투명성 부족 문제를 해결하기 위해 TagGAN이라는 새로운 Generative Adversarial Networks (GAN) 기반의 프레임워크를 제안합니다. TagGAN은 약한 감독 하에 질병의 픽셀 수준 지도를 생성할 수 있으며, 이는 라벨이 이미지 수준인 데이터만으로도 가능합니다. 이 연구는 의학 이미징에서 질병에 대한 정확한 시각화를 제공함으로써 진단 AI의 해석 가능성을 향상시킵니다.

- **Technical Details**: TagGAN은 도메인 변환 과정에서 비정상 이미지를 정상 표현으로 변환하면서 픽셀 수준의 질병 지도를 생성합니다. 이 후, 생성된 지도는 입력된 비정상 이미지에서 빼내어 모든 중요한 해부학적 세부 사항을 유지하면서 정상 이미지로 변환됩니다. TagGAN은 픽셀 수준 지도가 필요 없이 약한 감독 환경에서 잘 작동하는 최초의 모델로, 이는 의료 영상 분석에서 큰 발전을 의미합니다.

- **Performance Highlights**: CheXpert, TBX11K 및 COVID-19와 같은 기준 데이터셋에서 수행된 실험 평가를 통해 TagGAN은 질병 특정 픽셀을 정확하게 식별하는 데 있어 현재의 최상위 모델들을 능가하는 성능을 보여주었습니다. 이 결과는 TagGAN 모델이 의료 이미지를 태그할 수 있는 능력을 갖추었음을 강조하며, 훈련 중 이진 마스크의 필요성을 없애며 방사선 전문의의 작업 부담을 크게 줄일 수 있습니다.



### Weakly Supervised Pixel-Level Annotation with Visual Interpretability (https://arxiv.org/abs/2502.17824)
- **What's New**: 이 논문에서는 자동화된 설명 가능한 주석 시스템을 제안합니다. 이는 앙상블 학습(ensemble learning), 시각적 설명 가능성(visual explainability), 불확실성 정량화(uncertainty quantification)를 통합하여 의료 영상 주석의 정확성과 신뢰성을 크게 향상시킵니다. ResNet50, EfficientNet 및 DenseNet으로 구성된 세 가지 사전 훈련된 딥 러닝 모델을 결합하여 전문가의 진단 의사와 유사한 결과를 제공합니다.

- **Technical Details**: 제안하는 방식은 XGrad-CAM을 활용하여 시각적 설명을 제공하고 Monte Carlo Dropout을 통해 예측 불확실성을 정량화합니다. 이 시스템은 분류에 동의하는 모델의 saliency maps을 교차하여 픽셀 수준의 주석을 생성하며, 이를 통해 이미지 수준의 레이블만으로도 정확한 픽셀 주석을 얻을 수 있습니다. 불확실한 예측은 인간의 검토를 위해 플래그가 표시됩니다.

- **Performance Highlights**: TBX11K 의료 영상 데이터셋과 Fire 분할 데이터셋을 사용하여 평가한 결과, 우리 모델은 TBX11K에서 93.04%의 정확도, Fire 데이터셋에서 96.4%의 정확도를 달성했습니다. 또한, 이미지 수준의 레이블로 훈련한 모델임에도 불구하고 픽셀 수준 주석을 정밀하게 생성하여 각각 36.07% 및 64.7%의 Intersection over Union (IoU) 점수를 기록했습니다.



### CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems (https://arxiv.org/abs/2502.17821)
- **What's New**: 이번 연구에서 제안한 Collaborative Auxiliary Modality Learning (CAML)은 기존의 Auxiliary Modality Learning (AML)의 제한점을 극복하기 위해 다중 에이전트 간의 협업을 통한 다중 모달 데이터를 공유하는 새로운 프레임워크입니다. CAML은 테스트 시 각 에이전트가 감소된 모달리티로 추론할 수 있도록 하여, 자율 주행과 같이 동적 환경에서 더 나은 의사 결정을 지원합니다. 이 연구는 데이터 커버리지와 불확실성 감소 측면에서 CAML의 효과성을 분석하고, AML에 비해 이점들을 제공합니다.

- **Technical Details**: CAML은 다중 에이전트 시스템에서 훈련 기간 동안 모달 데이터를 공유하고, 테스트 시 각 에이전트가 감소된 모달리티로 추론하도록 설계되었습니다. 또한, 지식 전이(knowledge distillation)를 활용하여, 복잡한 모델에서 단순한 모델로 지식을 전이하여 모달리티가 결핍된 상태에서도 작동할 수 있도록 합니다. 예를 들어, 자율 주행 차량들이 훈련 중 LiDAR 및 RGB 이미지와 같은 센서 정보를 공유하여 더욱 견고한 표현을 구축하고, 운영 중에는 RGB 이미지만으로 추론을 수행하게 합니다.

- **Performance Highlights**: CAML은 사고 발생 가능성이 높은 자율 주행 시나리오에서 최대 58.13%의 사고 감지 성능 향상을 이뤄냈습니다. 추가적으로, 실제 공중-지상 로봇 데이터를 활용한 협업적 의미 분할에서 최대 10.61%의 mIoU 개선을 달성하였습니다. 이러한 결과는 CAML이 실세계 응용 프로그램에서 다중 에이전트 협업의 이점을 충분히 활용할 수 있음을 보여줍니다.



### An Overview of Large Language Models for Statisticians (https://arxiv.org/abs/2502.17814)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 통계와 AI 분야의 교차점에서 어떻게 신뢰성과 투명성을 증진시키는데 기여할 수 있는지를 탐구합니다. 특히 불확실성 정량화(uncertainty quantification), 해석 가능성(interpretability), 공정성(fairness), 개인 정보 보호(privay), 워터마킹(watermarking) 및 모델 적응(model adaptation)과 같은 이슈에 초점을 맞추고 있습니다. 이러한 세부 사항들을 통해 통계학자들이 LLMs의 발전에 중요한 기여를 할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: LLMs는 자연어를 이해하고 생성하는 데 있어 본질적으로 단어 또는 단어 시퀀스의 확률을 할당하는 모델입니다. LLMs의 설계와 배치에서 통계학자들이 어떤 역할을 할 수 있는지에 대한 질문이 제기되며, 이는 통계 방법 및 이론의 진전을 요구합니다. 모델의 아키텍처가 신뢰할 수 있는 확률적 출력을 생성하는 방법과 알고리즘의 공정성과 신뢰성을 보장하기 위해 LLMs의 출력을 어떻게 활용할 수 있는지를 탐구하고 있습니다.

- **Performance Highlights**: LLMs는 데이터 수집, 청소 및 분석의 전통적인 통계 워크플로우를 향상시킬 수 있는 잠재력을 제공합니다. 의료 연구 및 생물 통계와 같은 분야에서 LLMs는 대표 데이터를 합성하고, 비정형 임상 노트에서 중요한 통찰을 추출하며, 높은 위험의 응용에서 예측 모델링을 지원할 수 있습니다. 이러한 상호 작용은 통계와 AI 분야 모두에서 새로운 기회를 제공하며, LLMs의 발전이 사회적 복잡성 문제 해결에 기여할 수 있는 가능성을 열어갑니다.



### Safe Multi-Agent Navigation guided by Goal-Conditioned Safe Reinforcement Learning (https://arxiv.org/abs/2502.17813)
Comments:
          Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract here is shorter than that in the PDF file

- **What's New**: 이 논문에서는 전통적인 계획 및 안전한 강화 학습(safe Reinforcement Learning, safe RL)의 장점을 통합하는 새로운 방법론을 제시합니다. 해당 방법은 목표 조건 강화 학습(goal-conditioned RL)과 안전한 RL을 결합하여 자율 에이전트가 안전하게 목표를 설정하고 길을 찾을 수 있도록 합니다. 자동화된 자기 훈련 알고리즘을 통해 누적 거리 및 안전 수준을 추정하며, 리플레이 버퍼에서 상태를 구축하여 안전하지 않은 경로를 제거하고 안전성을 보장하는 경로 계획을 생성합니다.

- **Technical Details**: 목표 조건 강화 학습(GCRL)은 향상된 마르코프 결정 프로세스(MDP)를 사용하여 에이전트가 목표에 맞는 정책을 학습합니다. 이 방법은 기존의 데이터 구조인 리플레이 버퍼를 활용하여 중간 그래프 구조를 동적으로 생성하고, 이를 통해 경로 계획의 유연성을 높입니다. 또한, 기존의 그래프 기반 방법에 비해 더욱 안전하고 빠른 경로를 균형 있게 제공합니다.

- **Performance Highlights**: 다수의 벤치마크 테스트 결과, 본 방법은 복잡하고 위험한 환경에서 여러 에이전트가 안전하게 목표 거리까지 도달할 수 있도록 하는 데 있어 유의미한 성과를 보여주었습니다. 특히, 다양한 에이전트를 추가할 때 새로운 훈련이 필요 없이 확장 가능한 점이 강조되며, 다중 에이전트 안전 내비게이션 문제 해결에 있어 효과적인 조정을 가능하게 합니다. 향후 연구 지원을 위한 코드도 공개될 예정입니다.



### Can Multimodal LLMs Perform Time Series Anomaly Detection? (https://arxiv.org/abs/2502.17812)
Comments:
          9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection of multimodal LLMs and time series analysis are on the website this https URL

- **What's New**: 이 논문은 다중 모달 대형 언어 모델(MLLMs)이 시계열 이상 탐지(TSAD)에 적합한지 조사합니다. 이를 위해, 시간을 나타내는 수치 데이터를 이미지 형태로 변환하여 MLLMs와 함께 사용하는 새로운 벤치마크인 VisualTimeAnomaly를 제안합니다. 연구 결과, MLLMs는 초기 단일 변수 사례에서 시작하여 다변량 및 불규칙 시계열을 포함하는 보다 복잡한 시나리오로 확장된다는 점에서 중요합니다. 이 논문은 MLLMs를 TSAD에 적용한 첫 번째 종합 연구로, 관련 데이터와 코드를 공개하여 추가 연구를 촉진합니다.

- **Technical Details**: VisualTimeAnomaly 데이터셋은 3가지 현실 세계 시나리오 및 3가지 이상 granularities로 구성된 12,400개의 시계열 이미지로 실험을 진행합니다. 이 데이터는 여러 MLLMs 모델 (GPT-4o, Gemini-1.5, LLaVA-NeXT, Qwen2-VL)에 공급되어 다목적 테스트를 가능하게 합니다. 논문은 초기 단일 변수에서 복잡한 다변량 및 불규칙 시계열로의 이동을 통해 MLLMs의 성능을 포괄적으로 평가합니다. 모델 간 성능 차이는 다변량 이상 탐지에서 두드러지게 나타납니다.

- **Performance Highlights**: MLLMs는 점 단위 이상의 이상(즉, 범위 및 변수 기반 이상)을 더 효과적으로 탐지하고, 25%의 데이터가 결측될 경우에도 높은 강건성을 보였습니다. 오픈 소스 MLLMs는 단일 변수 시나리오에서 뛰어난 성능을 발휘하며, proprietary 모델과 비교해도 TSAD에서 유사한 결과를 나타냅니다. 이러한 결과는 MLLMs가 인지 패턴과 관계를 시각적으로 분석하는 데 유리함을 나타내며, 앞으로의 연구에서 이점을 제공할 수 있습니다.



### Uncertainty Quantification for LLM-Based Survey Simulations (https://arxiv.org/abs/2502.17773)
Comments:
          30 pages, 6 figures, 10 tables

- **What's New**: 본 논문은 대형 언어 모델(LLM)이 생성한 시뮬레이션 응답을 신뢰성 있게 활용하기 위한 방법을 제시합니다. 특히, 이러한 응답을 바탕으로 인구 통계의 신뢰 구간(confidence sets)을 구성하는 접근법을 개발하였습니다. 또한 샘플 크기를 적응적으로 선택함으로써, 과도한 오버피팅(overfitting)이나 비효율적인 추정치를 방지하려고 합니다.

- **Technical Details**: LLM 기반 설문 조사 시뮬레이션에서의 불확실성 정량화(unquantifying uncertainty)에 대한 엄격한 수학적 프레임워크를 제공합니다. 우리는 LLM과 실제 인구 간의 미스알라이먼트(misalignment)를 기반으로 시뮬레이션 샘플 크기를 적응적으로 선택하는 유연한 방법론(methodology)을 제안하고 있습니다. 이는 모든 LLM에 적용가능하며, 신뢰 구간을 구축하는데 드는 다양한 방법과 결합할 수 있습니다.

- **Performance Highlights**: 이 제안된 방법은 실제 데이터셋에 대한 일련의 실험을 통해 입증되었습니다. 결과적으로, 적절한 수의 시뮬레이션 샘플을 통해 신뢰할 수 있는 인구 통계 추정을 달성할 수 있었습니다. 우리의 방법은 LLM의 활용 가능성을 향상시키며, 인구 통계에 대한 이해도를 높이는 데 기여할 것입니다.



### Conformal Prediction Under Generalized Covariate Shift with Posterior Drif (https://arxiv.org/abs/2502.17744)
Comments:
          Accepted to AISTATS 2025

- **What's New**: 이 연구에서는 전이 학습을 위한 새로운 분포 가정을 바탕으로 한 적합한 분류문제를 탐구합니다. 전이 학습(transfer learning)은 관련 소스 도메인(source domain)에서 지식을 활용하여 목표 도메인(target domain)에서 학습 성능을 향상시키는 것을 목표로 합니다. 제안된 방법은 가중치가 적용된 적합한 분류기를 도입하여 소스 및 타겟 샘플을 모두 고려합니다.

- **Technical Details**: 제안된 방법론은 covariate shift with posterior drift(CSPD)라는 새로운 분포 설정 하에 작동합니다. CSPD는 소스 데이터와 타겟 데이터 간의 특성 분포가 다르고, 예측 확률(posteriors) 또한 다르게 발생하는 상황을 포괄합니다. 이를 통해 각 데이터 인스턴스에 대해 신뢰할 수 있는 예측 영역을 도출하는 방법을 제공합니다.

- **Performance Highlights**: 이론적 연구를 통해 제안된 방법이 우수한 점근적(asymptotic) 성질을 보인다는 것을 입증합니다. 수치적 연구 결과는 제안된 방법의 유용성을 추가적으로 보여줍니다. 또한, 본 연구는 기존 방법론을 확장하여 전이 학습을 통해 실제 애플리케이션에 더욱 적합한 예측을 제공합니다.



### Toward 6-DOF Autonomous Underwater Vehicle Energy-Aware Position Control based on Deep Reinforcement Learning: Preliminary Results (https://arxiv.org/abs/2502.17742)
Comments:
          6 pages, 5 figures, submitted to 2024 IEEE OES AUV Symposium

- **What's New**: 이 논문은 Truncated Quantile Critics (TQC) 알고리즘을 기반으로 한 새로운 Deep Reinforcement Learning (DRL) 접근 방식을 제안하여 holonomic 6-DOF AUV의 제어를 수행합니다. 이 방법은 수동 조정 없이 thruster 구성에 대한 사전 지식 없이 직관적으로 명령을 thruster에 전달합니다. 또한, 에너지 소비를 보상 함수에 직접 통합하여 에너지 효율성을 달성합니다.

- **Technical Details**: DRL은 에이전트가 환경과 상호작용하여 상태를 행동으로 매핑하는 방법입니다. TQC 알고리즘은 과거의 경험을 바탕으로 실시간으로 임무의 효율성을 향상시키며, 높은 비선형 동작을 모델링 할 수 있는 능력을 갖추고 있습니다. 이 시스템은 배터리 전원이 필요한 AUV의 복잡한 행동을 수학적으로 모델링하여 6-DOF 동적 시스템으로 다루어집니다.

- **Performance Highlights**: 시뮬레이션 결과는 TQC High-Performance 방법이 목표 지점에 도달하는 데 있어 조정된 PID 컨트롤러보다 우수한 성능을 발휘함을 보여줍니다. 반면 TQC Energy-Aware 방법은 다소 낮은 성능을 보이지만 평균적으로 30% 더 적은 전력을 소비하는 것을 입증하였습니다. 이러한 결과는 AUV의 효율적인 에너지 사용 및 자율성 증가에 큰 장점을 제공합니다.



### Are GNNs doomed by the topology of their input graph? (https://arxiv.org/abs/2502.17739)
- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에서 입력 그래프의 구조가 학습 성능에 미치는 영향을 조사합니다. 특히, 지역적인 топологические особенности가 어떻게 메시지 전파 방식과 상호작용하여 과도한 평탄화(oversmoothing) 및 표현력 있는(representations) 표현을 생성하는지를 중점적으로 분석합니다. 새로운 개념인 $k$-hop similarity를 도입하여, 고유한 지역 구조를 가진 그래프가 GNN에서 일관된 노드 표현(nodal representations)을 이끌어내는지를 탐구합니다.

- **Technical Details**: GNN은 메시지 패싱(message passing) 메커니즘을 통해 노드의 표현을 업데이트합니다. 각 GNN 레이어에서 노드의 표현은 이웃 노드의 정보를 집계(aggregate)하여 업데이트되며, k 레이어를 가지는 GNN은 k-이웃에서 정보를 캡처할 수 있습니다. 그러나 GNN은 특정 비동형 그래프(non-isomorphic graphs)를 구분하는 데 한계를 가집니다. 이 논문은 $k$-hop 유사성을 통해 이러한 한계를 분석합니다.

- **Performance Highlights**: 실험 결과, 지역적인 연결 구조가 GNN의 학습 성능에 미치는 영향이 강조됩니다. $k$-hop 유사한 그래프에 대한 학습이 일관된 노드 표현을 생성하는 경향이 있으며, 이는 GNN 성능에 실질적인 함의를 갖습니다. 또한, 이 연구는 과도한 평탄화 문제에 대한 새로운 설명을 제공하며, 그래프의 토폴로지가 GNN 성능에 미치는 부정적인 영향을 보여줍니다.



### Knowledge Distillation with Training Wheels (https://arxiv.org/abs/2502.17717)
- **What's New**: 본 논문에서는 지식 증류(Knowledge Distillation)의 새로운 프레임워크를 제안하여, 학생 모델이 교사 모델의 도움을 받을 수 있도록 하며, 테스트 시 자연어 명령으로 테스트 규칙을 따르도록 학습합니다. 이를 통해 기존의 지식 증류 방법에서 벗어나, 테스트 시간에 학생 모델이 교사 모델을 효과적으로 활용할 수 있는 새로운 방법론을 제공합니다. 이 방법은 학생 모델이 학습 자료를 이해하는 것뿐만 아니라, 다양한 섹션의 상대적 난이도를 파악하여 교사의 도움을 요청할 수 있도록 합니다.

- **Technical Details**: 논문에서는 지식 증류를 엔트로피 정규화된 가치 최적화 문제로 공식화하고, 이를 Path Consistency Learning(PCL) 기법으로 해결하여 새로운 지식 증류 알고리즘을 제안합니다. 또한, 제한된 강화 학습(constrained reinforcement learning)을 활용하여 교사 모델을 테스트 시간의 기준으로 활용할 수 있는 프레임워크로 확장합니다. 이 방식은 학생 모델이 교사의 도움을 요청하고, 주어진 규칙에 따라 필요한 경우에만 교사를 사용할 수 있도록 합니다.

- **Performance Highlights**: 번역과 요약 작업을 통해 실험한 결과, 모델들은 교사의 사용 규칙을 잘 준수하며 특정 토큰에 대해 스스로 생성하고 교사의 도움을 요청할 토큰을 우선순위로 두는 경향을 보였습니다. 교사의 사용이 허용됨에 따라 출력 품질이 향상되는 경향을 관찰했으며, 본 접근 방식은 Speculative Decoding과는 또 다른 유연한 정확도와 효율성의 균형을 제공합니다.



### Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures (https://arxiv.org/abs/2502.17710)
Comments:
          40 pages, 49 figures

- **What's New**: 이번 연구에서는 다양한 문화적 해석을 포함하는 새로운 데이터셋인 Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS)를 소개합니다. 이 데이터셋은 25가지 제스처와 85개 국가를 포함하여 총 288개의 제스처-국가 쌍을 다루고 있으며, 각 제스처의 공격성, 문화적 중요성 및 맥락적 요소에 대한 주석이 포함되어 있습니다. 이는 AI 시스템이 비문화적 콘텐츠를 생성할 위험을 줄이기 위한 필수적인 노력을 보여줍니다.

- **Technical Details**: MC-SIGNS는 미국 중심의 편향을 발견하고, AI 모델의 비언어적 커뮤니케이션 해석에서 나타나는 한계를 드러냅니다. 기존의 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)은 문화적 맥락을 인식하지 못하고 미국에서 유래한 해석에 의존하며, 많은 제스처를 공격적으로 잘못 플래그(flag)합니다. 데이터셋은 공격적 제스처 감지와 비언어적 커뮤니케이션의 문화적 적합성을 평가하기 위한 테스트 베드(test bed)로 사용됩니다.

- **Performance Highlights**: MC-SIGNS 데이터셋을 활용한 평가 결과, text-to-image(T2I) 시스템은 공격적인 내용을 거의 거부하지 못하며, LLM은 제스처를 과도하게 공격적으로 표시하는 경향이 있음을 보여주었습니다. 연구 결과, AI 모델은 미국 중심의 해석을 자주 반복하며, 비 미국적 맥락에서 공격적인 제스처를 식별하는 정확도가 크게 낮아짐을 확인했습니다. 이는 AI 기술의 공평한 전 세계적 배포를 위해 더욱 포괄적이고 문화적으로 민감한 안전 메커니즘이 필요함을 강조합니다.



### Contrastive Visual Data Augmentation (https://arxiv.org/abs/2502.17709)
- **What's New**: 이 논문에서는 새로운 Contrastive Visual Data Augmentation (CoDA) 전략을 제안하여 대형 다중모달 모델(Large Multimodal Models, LMMs)이 새로운 개념을 인식하고 논리적으로 처리하는 능력을 향상시키는 방법을 설명합니다. CoDA는 대상 개념의 주요 대조적 텍스트 및 시각적 특징을 추출하고, 이를 통해 작성된 합성 데이터를 생성하여 LMM이 혼동할 수 있는 개념을 명확히 구분할 수 있도록 돕습니다.

- **Technical Details**: CoDA의 주된 과정은 대조적 특징 추출, 특징 필터링, 특징 제어 이미지 생성, 증강 이미지 필터링의 4단계로 구성됩니다. 이 과정에서 CoDA는 LMM이 잘못 인식하는 개념과 혼동하게 되는 개념을 식별하고, 해당 개념의 시각적 및 텍스트적 특징을 추출합니다. 이 특징들은 가시성이 뛰어나고 LMM이 이해할 수 있는 방식으로 생성되고 필터링을 거쳐 최종적으로 증강 이미지로 변환됩니다.

- **Performance Highlights**: CoDA는 iNaturalist와 SUN와 같은 다양한 데이터셋에서 성능이 크게 향상되는 것을 확인할 수 있었으며, NovelSpecies라는 새로운 데이터셋에서도 테스트 결과 기존의 시각적 데이터 증강 기법보다 절대적으로 12.3%, 5.1%, 6.0%의 정확도 향상을 보여주었습니다. 이로써 CoDA는 LMM의 새로운 개념 인식 능력을 효과적으로 개선하는 데 성공하였으며, 이에 따라 비전 커뮤니티에 중요한 기여를 하고 있습니다.



### From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs (https://arxiv.org/abs/2502.17701)
Comments:
          24 pages, 9 figures

- **What's New**: 이 논문에서는 FLARE라는 새로운 프레임워크를 도입하여 대형 언어 모델(LLM)을 활용한 산불 대피 결정 예측을 혁신적으로 개선하고 있습니다. FLARE는 인간의 복잡한 행동 논리를 보다 잘 이해할 수 있도록 심리학 및 행동 이론을 통합하여 Chain-of-Thought (CoT) 추론을 간소화하고 메모리 기반 강화 학습(RL) 모듈과 통합합니다. 기존 LLM의 한계를 극복하여 보다 정확한 대피 결정 예측을 가능하게 하며, 실험 결과 평균 20.47%의 성능 향상을 보여 주었습니다.

- **Technical Details**: FLARE 프레임워크는 위험 인식(risk perception) 및 위협 평가(threat assessment)를 주요 개념으로 사용하여 개인의 정신 상태를 나타냅니다. PADM(Protective Action Decision Model) 기반의 분류기를 사용하여 역사적 데이터와 경험적 행동 연구를 통해 가장 관련성이 높은 입력 변수를 선택하고, 이후 LLM이 선택된 추론 패턴으로부터 인식을 유추하여 점수를 부여합니다. 또한 이 시스템은 오류 기록과 자기 반성 메커니즘을 통합하여 모델의 추론 과정을 개선하는 동시에 개인의 대피 행동을 맞춤화합니다.

- **Performance Highlights**: FLARE는 자주 사용되는 기존 이론 기반 모델 대비 20.47%의 성능 향상을 기록하며, 강력한 교차 사건 일반화(cross-event generalizability)를 보여주고 있습니다. 이 프레임워크는 제한된 데이터와 불균형한 데이터셋에서도 효과적으로 작동하여 실제 대피 행동을 보다 잘 반영합니다. 또한, 독립적 실험을 통해 FLARE의 이유 기반 추론 능력을 입증하며, 대피 결정 예측에서 새로운 기준을 제시하고 있습니다.



### A Fokker-Planck-Based Loss Function that Bridges Dynamics with Density Estimation (https://arxiv.org/abs/2502.17690)
Comments:
          Under review by the ICML

- **What's New**: 본 논문에서는 Fokker-Planck 방정식에서 유도된 새로운 손실 함수(loss function)를 통해 동적 시스템 모델을 확률 밀도 함수(probability density functions)와 연결짓는 방법을 제시합니다. 이 손실 함수는 비연속 데이터에서 동적 파라미터를 추출하는 데 유용함을 보여주며, 특히 노이즈가 포함된 Lorenz 시스템이나 유전자 조절 네트워크와 같은 비시계열 데이터에 대한 응용 예시를 소개합니다. 또한 알려진 동적 모델을 기반으로 밀도를 추정하는 데에도 이 손실 함수가 유용하게 사용될 수 있습니다.

- **Technical Details**: 제안된 손실 함수는 Fokker-Planck 방정식에 근거하여 시스템의 동적 변화와 밀도 함수의 발전 간의 일치도를 평가합니다. L-1 노름을 사용하여 동적 모델과 밀도 추정 간의 차이를 정량화하며, 고전적인 방법론의 한계를 극복합니다. 이 손실 함수는 Gaussian Mixture Model(GMM)과 normalization flow 모델의 통합을 통해 밀도 추정의 정확성을 향상시키며, 신경망에서 문제를 해결하는 데 필요한 계산 효율성을 극대화합니다.

- **Performance Highlights**: Lorenz 시스템에서 비시계열 데이터만을 기반으로 세 가지 동적 파라미터를 성공적으로 추정했으며, 이는 기존 방법들과 비교했을 때 획기적인 성과입니다. 제안된 밀도 추정기는 실제 데이터와 동적 모델을 동시에 활용해 밀도, 에너지 및 스코어 함수를 추정하며, 클러스터링과 노이즈 제거와 같은 하위 응용에서도 효과적임을 실험을 통해 입증했습니다. 이로 인해 복잡한 동적 시스템을 분석하는 방법론이 한층 강화됨을 보여줍니다.



### Socratic: Enhancing Human Teamwork via AI-enabled Coaching (https://arxiv.org/abs/2502.17643)
Comments:
          Extended version of an identically-titled paper accepted at AAMAS 2025

- **What's New**: Socratic라는 새로운 AI 시스템은 인력 자원이 제한된 상황에서도 실시간으로 팀에게 지침을 제공하는 혁신적인 기능을 갖추고 있습니다. 이는 특히 의료, 재난 대응과 같은 생명 및 안전이 중요한 분야에서 팀워크를 개선하는 데 도움을 줄 수 있습니다. 연구 결과, Socratic는 팀의 성과를 최소한의 개입으로 크게 향상시킬 수 있음이 입증되었습니다.

- **Technical Details**: Socratic는 다중 에이전트 시스템(multi-agent systems)과 모방 학습(imitation learning)의 최신 발전을 활용하여 팀 행동을 모델링합니다. 이 시스템은 팀 수행 중의 불일치를 감지하고 팀 구성원들에게 재조정을 권장하여 협업 결정을 개선합니다. 또한, Dec-POMDPs(Decentralized partially observable Markov decision processes)를 통해 고정된 임무와 시간 제약을 가진 협업 작업을 모델링하는 데 필요한 수학적 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, Socratic는 실제 팀 작업에서 성과를 크게 향상시키는 것으로 나타났습니다. 참가자들은 비록 최소한의 개입으로 이루어졌지만, Socratic이 팀워크 개선에 유용하고 신뢰할 수 있는 도구로 인식하였습니다. 이러한 분석은 AI 연구와 실제 적용 가능성에 대한 희망적인 방향을 제시합니다.



### A Priori Generalizability Estimate for a CNN (https://arxiv.org/abs/2502.17622)
- **What's New**: 이번 연구에서는 전체 합성곱 신경망(CNN)의 잘린 특이값 분해(truncated SVD)를 정의하고 이를 통해 CNN의 성능 저하가 예상되는 이미지를 식별하는 데 유용하다는 것을 입증하였습니다. 연구자들은 Right Projection Ratio와 Left Projection Ratio라는 두 가지 새로운 지표를 도입하여, 회귀 관점에서 이미지와 특이 벡터 간의 투영 충실도를 평가합니다. 이러한 접근은 기존의 진단 도구에서의 제한을 극복할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 연구에서 분석한 CNN은 컨볼루션, 다운샘플링, 업샘플링, ReLU 활성화 및 스킵 커넥션으로 구성되어 있습니다. 입력 텐서와 그에 대응하는 투영 행렬을 정의함으로써 CNN의 구조를 명확하게 설명하며, 이는 다양한 차원에서의 이미지 데이터를 처리할 수 있는 모델의 특성을 보여줍니다. 잘린 SVD 알고리즘은 CNN의 adjoint(켈리 변환)를 활용해 매트릭스 연산을 최적화하여 효율적인 라이트 프로젝션 비율을 계산합니다.

- **Performance Highlights**: Right Projection Ratio는 레이블이 없는 데이터만으로도 모델의 성능과 상관관계가 있는 것으로 나타났습니다. 이는 이미지 분할 문제에서 특정 이미지 세트에 대해 CNN의 신뢰성 있는 예측 지표로 작용할 수 있음을 시사합니다. 두 가지 데이터 세트에서 실험을 수행하여, 여전히 CNN의 성능을 저하시키는 클래스 불균형 문제를 효과적으로 식별할 수 있음을 보여주었습니다.



### Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks (https://arxiv.org/abs/2502.17612)
- **What's New**: 이 논문은 분산형(Decentralized) 제어기를 위한 그래프 신경망(Graph Neural Network, GNN) 구조의 중요성을 강조합니다. 제어기 설계에 자연계에서 발견되는 자기 조직화(self-organization)를 영감을 얻어, 특히 '플로킹(flocking)'에서의 대칭(symmetries)을 활용하여 플로킹의 응집(cohesion)을 유지하는 데 집중합니다. 이 연구는 대칭을 강화한 GNN 제어기가 기존의 제어기보다 적은 데이터와 가중치로도 유사한 성능을 보임을 보여줍니다.

- **Technical Details**: 연구에서는 회전 불변성(rotation equivariance)과 변환 불변성(translation invariance) 대칭을 GNN 제어기에 적용하여 플로킹 제어를 향상시킵니다. 이 기법은 백분율로 표현하면 기존 GNN 제어기보다 70% 적은 훈련 데이터와 75% 적은 훈련 가중치를 사용하여 유사한 플로킹 통제를 구현합니다. 이를 위해 기존의 GNN 아키텍처에서 대칭적 특성을 지속적으로 활용하는 방식으로 설계되었습니다.

- **Performance Highlights**: 이 대칭 인지(controller) GNN 제어기는 기존 GNN 제어기보다 더 나은 일반화 성능을 보입니다. 또한, 이 연구에서 제안된 알고리즘은 다양한 환경에서도 효과적으로 작동하여 특히 자율 드론 군 제어나 센서 네트워크와 같은 복잡한 시스템에 큰 기여를 할 수 있습니다.



### Data-Driven Pseudo-spectral Full Waveform Inversion via Deep Neural Networks (https://arxiv.org/abs/2502.17608)
Comments:
          11 pages, 6 pages, review paper

- **What's New**: 이 연구는 Deep Learning(딥러닝) 방법을 활용하여 지진학적 FWI(Full Waveform Inversion)에 대한 새로운 접근 방식을 제시합니다. 기존의 시간 영역(time-domain) 접근 대신에 pseudo-spectral(유사 스펙트럼) 방법을 통합하여 데이터 기반(data-driven) DNN(Deep Neural Networks) 모델을 제안하고 있습니다. 이를 통해 기존 FWI 기법의 한계를 극복하고, 더 깊은 영역과 오버스러스트 지역에서 더욱 우수한 성능을 보여 줍니다.

- **Technical Details**: FWI에서는 파형 방정식으로부터 유도된 매개변수를 최적화하여 묘사하기 위해 multivariate optimization(다변량 최적화) 방법이 적용됩니다. 연구에서는 pseudo-spectral 방법을 Deep Learning 프레임워크에 통합하기 위해 이론적으로 FWI 문제를 재구성하고, 이는 synthetics data(합성 데이터)에서 평가되었습니다. 뉴럴 네트워크의 구조 및 학습 과정에서 가중치를 결정하기 위한 제곱합 오차(Sum of Squared Errors)처럼 일반적으로 사용되는 cost function 이 사용됩니다.

- **Performance Highlights**: 제안된 DNN 프레임워크는 기존의 결정론적 모델 및 시간 기반(time-based) 접근 방식과 비교하여 유의미한 성능 향상을 보여 주었습니다. 특히, 더 깊은 지층과 오버스러스트 구역에서 classical FWI보다 우수한 결과를 도출하여 DNN의 가능성을 증명합니다. 향후 연구 방향으로는 pseudo-spectral DNN 방법의 한계 분석 및 추가적인 발전 가능성이 논의됩니다.



### PICASO: Permutation-Invariant Context Composition with State Space Models (https://arxiv.org/abs/2502.17605)
Comments:
          Published in The Thirteenth International Conference on Learning Representations, ICLR 2025

- **What's New**: 본 논문에서는 Large Language Models (LLMs)에 retrieval 기능을 추가하여 외부 지식을 효과적으로 활용할 수 있는 방법을 제안합니다. 특히, 여러 개의 pre-computed 상태를 효율적으로 조합하여 고품질 출력을 생성하는 방법인 PICASO(Permutation-Invariant Compositional Aggregation of States as Observations)를 도입합니다. 이 방법은 기존의 concatenation 방식과 비교하여 평균 5.4배 빠르며 성능 향상도 보여줍니다.

- **Technical Details**: PICASO는 State Space Models (SSMs)을 활용하여 정보가 포함된 여러 맥락들을 효과적으로 조합합니다. SSM의 동작에서 유도된 수학적 관계를 이용하여 여러 상태를 하나로 합치는 방법을 개발하였고, 이를 통해 맥락의 순서가 결과에 미치는 영향을 최소화합니다. Dynamic Programming을 통해 상태를 평균화하여 계산의 효율성을 높이고, 다소 근사치를 허용함으로써 선형 시간으로 계산할 수 있습니다.

- **Performance Highlights**: 실험 결과, PICASO는 WikiText와 MSMARCO 데이터셋에서 91%의 성능 이득을 보였으며, 기존의 최강 성능을 가진 baseline과 동일한 성능을 유지하면서도 더 빠른 처리 속도를 자랑합니다. 또한, 사전 훈련된 Mamba-2 2.7B 모델을 사용하여 단 하루의 fine-tuning으로 concatenation과 동등한 성능을 나타냈습니다.



### A stochastic smoothing framework for nonconvex-nonconcave min-sum-max problems with applications to Wasserstein distributionally robust optimization (https://arxiv.org/abs/2502.17602)
Comments:
          35 pages

- **What's New**: 이번 논문에서 제안된 새로운 알고리즘 프레임워크는 비볼록 비오목(min-sum-max) 구조 문제에 대한 수렴 보장을 제공합니다. 이 접근법은 기존의 알고리즘들이 가진 제한사항을 극복하며, 딥 뉴럴 네트워크 훈련 같은 최신 기계 학습 문제를 해결하는 데 적용될 수 있습니다.

- **Technical Details**: 연구에서는 Log-Sum-Exp 함수에 기반한 확률론적 스무딩(stochastic smoothing) 프레임워크를 도입하여 max 연산자를 효율적으로 근사합니다. 제안된 알고리즘은 Clarke regularity를 기반으로 iterative smoothing 알고리즘을 개발하고, 이는 컴퓨터 성능 문제를 해결하며 순차적으로 수렴성을 보장합니다.

- **Performance Highlights**: 제안된 방법은 뉴스 벤더 문제(newsvendor problem), 딥 러닝 회귀(deep learning regression), 적대적 강건 딥 러닝(adversarially robust deep learning) 문제를 해결하는 데 있어 기존의 최첨단 방법들보다 더 높은 정확도와 강건성을 보이는 결과를 나타냈습니다. 이 연구는 어려운 문제 설정에서도 효과적으로 작동하는 방법론의 가능성을 시사합니다.



### Synergizing Deep Learning and Full-Waveform Inversion: Bridging Data-Driven and Theory-Guided Approaches for Enhanced Seismic Imaging (https://arxiv.org/abs/2502.17585)
Comments:
          20 pages, 14 images, literature review

- **What's New**: 이번 리뷰 논문은 Deep Learning(DL)과 Full Waveform Inversion(FWI)의 통합을 다루며, 수치해석에서의 더 나은 지진 이미징과 지하 구조 특성화 가능성을 탐구합니다. FWI와 DL의 기본 원리를 설명하고, 속도 추정, 복원, 단층 촬영 등 다양한 지구물리학적 응용을 검토합니다. 또한, 모델 복잡성, 데이터 품질과 같은 기존의 도전 과제와 함께, 하이브리드 및 물리 기반 모델을 통한 향후 연구 방향에 대해서도 논의합니다.

- **Technical Details**: FWI는 기록된 데이터와 일치하는 최적의 속도 모델 및 기타 암석 특성(밀도, 소산 흡수, 이방성)을 도출하기 위한 기법으로, 두 가지 주요 접근 방식인 전역 최적화(global optimization)와 직렬 해결(direct solving)을 사용합니다. 전역 최적화 방법으로는 몬테카를로(Monte Carlo) 방법, 유전 알고리즘(Genetic Algorithm), 시뮬레이티드 어닐링(Simulated Annealing) 등이 있으며, 이들은 각각 무작위 검색, 생물학적 진화의 유사, 통계 역학적 물리 등을 기반으로 하고 있습니다. 반면, 지역 최적화(local optimization) 방법은 1980년대에 소개되었으며, 관측된 데이터와 모델 데이터 간의 오차를 최소화하는 방식으로 적용됩니다.

- **Performance Highlights**: FWI와 DL의 통합은 지진 이미징 기술에 혁신적인 변화를 가져올 수 있는 잠재력을 지니고 있습니다. 이 두 접근 방식의 시너지를 통해 연구자들은 불완전한 데이터와 소음에 대한 강건성을 개선할 수 있으며, 계산의 효율성과 정확성을 동시에 높일 수 있는 기회를 갖게 됩니다. 이 리뷰는 FWI와 DL의 통합이 지하 정보를 보다 정확하고 효율적으로 추출할 수 있는 새로운 가능성을 열어줄 것이라고 강조하고 있습니다.



### Multi-Year-to-Decadal Temperature Prediction using a Machine Learning Model-Analog Framework (https://arxiv.org/abs/2502.17583)
Comments:
          14 pages, 10 figures (+ 8 supplemental figures)

- **What's New**: 이 논문은 다년에서 수십 년에 걸친 기후 예측을 위한 새로운 프레임워크를 제시하며, 머신러닝과 유사 예측(analog forecasting) 방법을 결합했다. 주요 기능은 지역과 예측 시점에 맞춘 마스크를 학습하여 예측 대상의 진화에 중요한 선행 정보(precursors)를 파악하는 것이다. 최적의 유사 상태와의 비교를 통해 얻은 예측 정보를 바탕으로 2미터 온도를 예측하는 데 있어 향상된 성능을 보였다.

- **Technical Details**: 이 연구에서는 CMIP6 데이터셋의 모델 시뮬레이션을 기반으로 하여, 다양한 지역과 리드타임에서의 기후 예측을 수행한다. 아날로그 예측 방식에서는 특정 예측 작업에 맞춘 가중치 마스크를 학습하여 중요한 선행 지역을 강조하며, 이를 통해 연간 평균 2미터 온도를 예측한다. 최종 데이터셋은 각각의 모델에서 역사적 기간과 SSP 예측을 포함한 총 285개의 시뮬레이션으로 구성된다.

- **Performance Highlights**: 이 연구는 기존 아날로그 방법과 초기화된 ESM(IESM) 예측 방법에 비해 다양한 지역과 리드타임에서 향상된 성능을 나타내었다. 특히, 머신러닝 요소가 추가됨으로써 유사 상태 선택에서 예측 작업에 특화된 정보를 효과적으로 반영하여, 기후 예측의 정확성을 높였다. 이로 인해 기후 변화에 대한 적응 및 완화 계획을 지원할 수 있는 유용한 도구가 될 것으로 기대된다.



### VANPY: Voice Analysis Framework (https://arxiv.org/abs/2502.17579)
- **What's New**: 최근 논문에서는 목소리 데이터의 자동 분석 및 특성화에 대한 종합적인 도구의 부족을 해결하기 위해 VANPY (Voice Analysis in Python) 프레임워크를 개발하였습니다. 이 오픈소스(end-to-end) 통합 프레임워크는 음성 데이터를 기반으로 한 화자 특성화를 위해 설계되었으며, 다양한 구성 요소를 쉽게 통합할 수 있도록 확장성을 고려하여 제작되었습니다.

- **Technical Details**: VANPY 프레임워크는 노래/음성 분리(music/speech separation), 음성 활동 감지(voice activity detection), 화자 임베딩(speaker embedding), 음성 특성 추출(vocal feature extraction) 등 15개 이상의 음성 분석 구성 요소를 포함합니다. 프레임워크의 특정 구성 요소는 성별 분류, 감정 분류, 나이 회귀, 신장 회귀와 같은 화자 특성화를 위한 기계 학습 모델로서, 여러 데이터셋에서 강력한 성능을 발휘하고 있습니다.

- **Performance Highlights**: VANPY 프레임워크는 'Pulp Fiction' 영화의 캐릭터 음성을 분석하는 사용 사례를 통해 화자의 성별, 나이, 신장, 감정 유형 및 감정 강도를 포함한 다양한 화자 특성을 효과적으로 추출할 수 있음을 보여줍니다. 이 연구는 음성 데이터 분석의 가능성을 강조하며, 다양한 실제 응용 프로그램에서 화자 특성화의 중요성을 명확히 하고 있습니다.



### How Do Large Language Monkeys Get Their Power (Laws)? (https://arxiv.org/abs/2502.17578)
- **What's New**: 이 논문은 언어 모델의 다수의 시도에 대한 성공률을 분석하여, 각 문제의 실패율이 시도 횟수에 따라 지수적으로 감소해야 한다는 예측을 제시합니다. 그러나 연구 결과는 반대로, 전체적인 성공률이 다항식(power law)으로 변화함을 보여주며, 이는 개별 문제의 성공률이 헤비 테일(distributional perspective) 형태로 분포하여 나타나는 결과입니다.

- **Technical Details**: 연구진은 언어 모델이 각 문제에 대해 다수의 시도를 할 때, 개별적 성공 확률이 비대칭적인 분포를 가질 수 있음을 강조합니다. 이는 소수의 문제에서 극단적으로 낮은 성공 확률이 전체 성공 트렌드를 왜곡하게 만들어 다항식 형태의 분포를 생성하는데 기여합니다. 이 접근법은 또한 기존의 연구에서 관찰된 다항식 스케일링에서의 편차를 설명하게 됩니다.

- **Performance Highlights**: 실험 결과, 새로운 분포 관점을 통해 예측된 파워 로우(exponent)에 대한 상대적 오차가 크게 줄어들었으며, 이는 추론 계산을 약 2-4 배 낮출 수 있는 방법으로 나타났습니다. 결과적으로, 본 연구는 신경 언어 모델 성능의 향상과 이와 관련된 평가 발전에 기여할 수 있는 중요한 통찰을 제공합니다.



### Utilizing Machine Learning to Predict Host Stars and the Key Elemental Abundances of Small Planets (https://arxiv.org/abs/2502.17563)
Comments:
          22 pages, 9 figures, 3 tables, accepted to AJ

- **What's New**: 이번 연구에서는 별의 조성과 작은 행성의 존재와의 관계를 머신러닝 알고리즘인 XGBoost를 활용하여 조사하였습니다. 특히 Mg, Si, Fe과 같은 원소들이 작은 행성 형성에 중요한 역할을 한다는 점에 주목했습니다. 연구진은 NASA Exoplanet Archive에 따르면 작은 행성을 포함하는 별들의 특징을 찾아내어, 작은 행성을 호스팅할 확률이 90% 이상인 별들의 목록을 작성했습니다.

- **Technical Details**: 조사는 세 가지 그룹의 외계 행성에 대해 진행되었습니다: (a) 모든 작은 행성, R$_{P}$ <$ 3.5 R_{⊕}$, (b) 서브 넵튠(sub-Neptunes), 2.0 $R_{⊕}$ <$ R_{P}$ <$ 3.5 $R_{⊕}$, (c) 수퍼 지구(super-Earths), 1.0 $R_{⊕}$ <$ R_{P}$ <$ 2.0 $R_{⊕}$입니다. 각 그룹은 서로 다른 특징 조합을 테스트하기 위해 7개의 앙상블로 세분화 되었습니다. 연구 결과, Na와 V가 어떤 행성의 반경과 관계없이 중요한 특징으로 밝혀졌습니다.

- **Performance Highlights**: 결과적으로, 이 연구는 별과 작은 행성 간의 화학적 상호작용을 나타내는 경향성을 발견하였고, 이로 인해 외계 행성 형성에서 원소의 중요성을 강조합니다. 또한, 머신러닝 기법은 향후 NASA의 작은 행성 탐지 미션에서 타겟을 선정하는 데 중요한 역할을 할 것으로 기대하고 있습니다. 연구 결과는 James Webb Space Telescope(JWST) 및 Nancy Grace Roman Space Telescope(NGRST)와 같은 미래의 탐사 임무에 큰 도움이 될 것입니다.



### Expressive equivalence of classical and quantum restricted Boltzmann machines (https://arxiv.org/abs/2502.17562)
Comments:
          11 pages, 4 figures; supplementary material 6 pages, 1 figure

- **What's New**: 이 연구에서는 semi-quantum restricted Boltzmann machine (sqRBM)이라는 새로운 모델을 제안합니다. sqRBM은 비가역적인 항을 포함하여 구성되어 있지만, 가시(unit) 하위 공간에서의 해밀토니안은 가역적입니다. 이는 출력 확률과 기울기를 명확히 계산할 수 있게 하여, 기존의 quantum restricted Boltzmann machines (QRBM)보다 효율적이고 훈련 가능성을 높입니다.

- **Technical Details**: sqRBM은 상관관계를 분석하여, 두 모델(RBM과 sqRBM) 간의 관계를 명확히 합니다. 또한, sqRBM은 RBM과 유사한 표현 능력을 가지며, 훈련 시 필요한 숨겨진 유닛 수는 RBM의 3분의 1로 줄일 수 있습니다. 이러한 특성 덕분에 sqRBM은 비효율적인 훈련 및 샘플링 문제를 완화할 것으로 예상됩니다.

- **Performance Highlights**: 우리는 최대 100개 유닛을 사용한 수치 시뮬레이션을 통해 이론적 발견을 검증하였습니다. 결과적으로, sqRBM은 훈련 및 샘플링의 계산 비용을 줄이며 실용적인 양자 기계 학습 애플리케이션을 가능하게 할 것으로 기대됩니다. 이러한 발전은 양자 기계 학습 분야에서의 응용 가능성을 크게 확대할 것입니다.



### CLEP-GAN: An Innovative Approach to Subject-Independent ECG Reconstruction from PPG Signals (https://arxiv.org/abs/2502.17536)
- **What's New**: 이 연구는 PPG(Photoplethysmography) 신호로부터 보이지 않는 ECG(Electrocardiogram) 신호를 재구성하는 도전 과제를 다룹니다. 기존의 여러 ECG-PPG 데이터셋은 다양성이 부족하며, 데이터 수집 과정에서 발생하는 노이즈로 인해 PPG로부터 ECG를 재구성하는 것이 어려워지고 있습니다. 이를 해결하기 위해 ODE(Ordinary Differential Equation) 모델을 활용한 새로운 합성 데이터 생성 기술과 주제에 독립적인 PPG-ECG 재구성 모델을 개발했습니다.

- **Technical Details**: 제안된 PPG-ECG 재구성 방법은 대조 학습(Contrastive Learning), 적대적 학습(Adversarial Learning), 주의 게이팅(Attention Gating) 기술을 통합하여 진행됩니다. 특히, 주제에 독립적인 접근법을 지향하며, GAN(Generative Adversarial Network) 아키텍처에 기반한 CLEP-GAN을 사용하여 PPG 신호로부터 ECG 신호를 재구성합니다. 이 모델은 다양한 전처리 기술 및 데이터 변동성을 관리하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존의 방법들과 비교할 때 보이지 않는 ECG 재구성에서 우수한 성능을 보여주었습니다. 특히, 성별 및 연령과 같은 요소들이 재구성 정확도에 미친 영향을 분석하여, 데이터셋 증강 시 인구학적 다양성을 고려하는 것의 중요성을 강조했습니다. 이 연구는 비침습적인 심장 모니터링의 새로운 가능성을 열어줍니다.



### A Machine Learning Approach for Design of Frequency Selective Surface based Radar Absorbing Material via Image Prediction (https://arxiv.org/abs/2502.17534)
- **What's New**: 이번 논문은 머신러닝(ML) 기법을 활용하여 주파수 선택적 표면(FSS) 기반의 레이더 흡수 재료를 설계하는 혁신적인 방법론을 제시합니다. 전통적인 전자기 설계에서는 FSS의 유닛 셀 치수를 입력 값으로 사용하여 흡수 계수를 예측하지만, 이 연구에서는 흡수 계수를 ML 모델의 입력으로 사용하고 FSS 유닛 셀 이미지를 예측합니다.

- **Technical Details**: 논문에서는 1GHz에서 30GHz까지의 넓은 주파수 대역에서 총 11개의 서로 다른 ML 모델이 연구되었습니다. 이 중 6개의 ML 모델(무작위 숲 분류(Random Forest classification), K-최근접 이웃 분류(K-Neighbors Classification), 그리드 검색 회귀(Grid search regression), 무작위 숲 회귀(Random Forest regression), 결정 트리 분류(Decision tree classification), 결정 트리 회귀(Decision tree regression))가 90% 이상의 훈련 정확도를 보여 주었습니다.

- **Performance Highlights**: 예측된 이미지의 다양한 주파수에 대한 흡수 계수는 상용 전자기 솔버를 사용하여 평가되었습니다. 이러한 ML 모델의 성능은 고무적이며, 향후 고성능 FSS 기반 레이더 흡수 재료의 설계 및 최적화를 가속화하는 데 유용하게 사용될 수 있습니다.



### Multimodal Bearing Fault Classification Under Variable Conditions: A 1D CNN with Transfer Learning (https://arxiv.org/abs/2502.17524)
- **What's New**: 본 연구는 다중 모달 베어링 결함 분류 접근법을 제안하고 있으며, 이것은 진동(vibration) 신호와 모터 위상 전류(motor phase current) 신호를 1차원 합성곱 신경망(1D CNN) 프레임워크 내에서 결합하여 결함 감지의 정확성을 높이고자 합니다. 이 방법은 여러 신호에서 특징을 융합하여 정확도를 개선하며, L2 정규화의 추가로 모델은 96%의 정확도를 달성하였습니다. 이 연구는 다양한 운영 조건에서 강력한 성능을 보이는 것을 확인했습니다.

- **Technical Details**: 연구에서 다룬 1D CNN 모델은 진동과 전류 신호를 기반으로 하여 결함을 분류하는 방식으로, 최대 풀링(max pooling) 레이어까지의 파라미터를 보존하고 이후 레이어를 조정하는 전이 학습(transfer learning) 전략이 가장 높은 성능을 발휘했습니다. 또한, 이 접근법은 연산 자원이 제한된 환경에서도 적절한 대안을 제공하고 있습니다. 하지만 이는 더 많은 훈련 가능한 파라미터를 요구하여 연산 시간의 증가를 수반합니다.

- **Performance Highlights**: 이 다중 모달 1D CNN 프레임워크는 변동하는 작동 조건에서도 높은 정확도와 적응성을 제공하며, 산업 환경에서의 베어링 결함 분류의 정확도를 높이는 기반을 마련했습니다. 모델은 다양한 테스트에서 평균 96% 이상의 정확도를 달성하였으며, 기존의 단일 모달 신호 기반 방법보다 향상된 결과를 보여주었습니다. 본 연구 결과는 향후 베어링 결함 진단 시스템 발전에 기여할 것으로 기대됩니다.



### Protein Large Language Models: A Comprehensive Survey (https://arxiv.org/abs/2502.17504)
Comments:
          24 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 Protein LLMs(단백질 대형 언어 모델)에 대한 포괄적인 개요를 제공하는 최초의 연구로, 기존의 서베이들이 특정 측면이나 응용에 중점을 두었던 것과는 달리, 이 논문에서는 구조, 훈련 데이터셋, 평가 지표 및 다양한 응용 분야를 다루고 있습니다. 이는 단백질 과학에서의 혁신적인 발전에 기여할 것입니다.

- **Technical Details**: 저자들은 100편 이상의 연구 논문을 체계적으로 분석하여 최신 Protein LLMs의 구조적 분류법을 제안합니다. 이 모델들은 방대한 단백질 서열 데이터(large-scale protein sequence data)를 활용하여 더 높은 정확도를 얻는 방법을 분석하며, 단백질 공학 및 생물 의학 연구에서의 잠재력을 탐구합니다.

- **Performance Highlights**: Protein LLMs는 단백질 구조 예측, 기능 주석 및 디자인에서 더 효율적인 성능을 보여주며, 과학적 발견을 위한 필수 도구로 자리매김하고 있습니다. 논문은 단백질 과학 내 미래의 도전 과제와 방향성에 대해서도 논의합니다.



### Accuracy of Wearable ECG Parameter Calculation Method for Long QT and First-Degree A-V Block Detection: A Multi-Center Real-World Study with External Validations Compared to Standard ECG Machines and Cardiologist Assessments (https://arxiv.org/abs/2502.17499)
Comments:
          37 pages, 8 figures, 6 tables

- **What's New**: 최근 착용 가능한 장치가 심장 모니터링에 혁신을 가져왔습니다. 이 연구는 새로운 알고리즘인 FeatureDB를 통해 착용형 단일 리드 신호에서 ECG 매개변수를 자동으로 계산하는 방법을 평가했습니다. 이 연구는 기존 ECG 장치 및 전문가 의사 평가에 대한 정확성을 검증하기 위해 대규모 다기관 연구를 수행했습니다.

- **Technical Details**: 연구는 세 가지 다양한 데이터셋을 사용하였으며(AHMU-FH dataset, CSE dataset, HeartVoice-ECG-lite dataset), 모든 데이터는 두 명의 경험이 풍부한 심장 전문의에 의해 주석이 달렸습니다. FeatureDB는 PR interval, QRS duration, QT interval 등의 주요 ECG 매개변수와 표준 ECG 기계 및 임상의사들에 의해 계산된 결과와 통계적으로 유의관계를 나타냈습니다. Bland-Altman 분석을 통해 높은 정확도를 확인했습니다.

- **Performance Highlights**: FeatureDB는 Long QT syndrome (LQT) 및 atrioventricular block interval abnormalities (AVBI) 탐지에서 뛰어난 진단 성능을 보여주었습니다. LQT에서 ROC 곡선 아래 면적(AUC)은 0.836, AVBI는 0.861로 나타났으며, 정확도(accuracy)도 LQT 0.856, AVBI 0.845로 훌륭했습니다. 이러한 결과는 FeatureDB의 임상적 신뢰성을 뒷받침하며, 착용형 ECG 기술이 심혈관 질환 관리 및 조기 개입 전략에 통합될 수 있는 가능성을 강조합니다.



### External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation (https://arxiv.org/abs/2502.17494)
Comments:
          Accepted by the ACM Web Conference (WWW) 2025 Industrial Track as Oral Presentation

- **What's New**: 본 논문에서 제안된 External Large Foundation Model (ExFM) 프레임워크는 산업 규모의 광고 추천 시스템에서 간과된 두 가지 주요 도전 과제를 해결하기 위해 설계되었습니다. 첫 번째 과제는 훈련 및 추론 지연이 제한되어 있는 것으로, 기존의 방법들이 대형 모델의 훈련과 추론 비용을 증가시키는 문제를 가지고 있었습니다. 두 번째 과제는 데이터 분포가 동적으로 변화하는 대량의 스트리밍 데이터가 사용되며, 이로 인해 모델이 항상 최신 데이터에 적합해야 하는 문제입니다.

- **Technical Details**: ExFM은 외부 증류(External Distillation) 기법을 활용하여 훈련 데이터 및 유지 비용을 절감하면서, 여러 학생 모델(Vertical Models, VMs)에 걸쳐 공통된 Teacher 모델(Foundation Model, FM)의 예측을 제공합니다. Auxiliary Head (AH)와 Student Adapter (SA)를 도입하여 FM가 VMs에 전이되는 데이터 분포의 격차를 완화하고, Freshness Gap을 감소시켜 모델의 성능을 향상시키는 방법론을 제안합니다. 이를 통해 ExFM은 재훈련을 최소화하면서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: ExFM 프레임워크를 도입한 실험 결과, 내부 산업 규모 데이터셋과 공개 데이터셋 모두에서 성능 향상이 확인되었습니다. ExFM은 수조 개의 파라미터를 가진 모델을 메타 플랫폼에 활용할 수 있도록 하여, 다양한 도메인 및 작업의 VMs에서 Promising한 성능 개선을 보여주었습니다. 또한, 하이퍼파라미터의 영향을 분석한 결과도 제공하여 이 접근법의 효용성을 강조합니다.



### Using Graph Convolutional Networks to Address fMRI Small Data Problems (https://arxiv.org/abs/2502.17489)
Comments:
          8 pages

- **What's New**: 이 연구는 그래프 신경망(graph neural networks)을 활용하여 소규모 데이터를 다루는 의학 이미징(medical imaging)의 예측 문제를 해결하고자 한다. 주로 치료 반응 예측(prognosis) 문제에 집중하여, 기능성 자기공명영상(fMRI) 데이터를 기반으로 치료에 대한 증상 개선을 예측하는 새로운 방법론을 제시한다. 기존의 패턴 인식 기법으로는 어려운 예측을 가능하게 하기 위하여, 환자의 뇌 활동 연결성 정보를 스펙트럼 표현(spectral representation)을 통해 효과적으로 전파하는 방식을 소개한다.

- **Technical Details**: 연구의 주요 기술적 초점은 t-fMRI 데이터의 작은 크기와 연결성 그래프(connected graphs) 정보의 복잡성을 극복하기 위해 설계된 심층 그래프 컨볼루션 학습 구조에 있다. 각 환자를 위한 새로운 표현 방식으로 스펙트럼 분해(spectral decomposition)를 활용하며, 이는 이전의 스펙트럼 분석과는 구별되는 접근법이다. GNN(그래프 신경망) 방법을 사용하여 기존의 일반 NN(신경망) 방법보다 더 나은 예측 성능을 가져오며, 이로 인해 72.2% ± 0.7%의 정확도를 달성했다.

- **Performance Highlights**: 실험 결과, 제안된 GNN 방법은 기존의 방법보다 약 12% 향상된 성능을 발휘함을 보였다. 데이터의 스무딩(smoothing) 효과를 통해 삼각 부등식(triangle inequality)의 위반을 줄임으로써 성능이 개선된 것으로 나타났으며, 이는 연결 데이터의 스펙트럼 임베딩(spectral embedding)을 더 잘 수행할 수 있게 함을 의미한다. 이러한 결과는 환자별 예측의 효율성을 높이고, 응용 가능한 치료 접근법을 제시하는 데 기여할 것이다.



### Multimodal Sleep Stage and Sleep Apnea Classification Using Vision Transformer: A Multitask Explainable Learning Approach (https://arxiv.org/abs/2502.17486)
- **What's New**: 이번 논문에서는 수면 단계( sleep stage )와 수면 장애( sleep disorder )를 동시에 분류할 수 있는 1D-Vision Transformer 모델을 제안합니다. 기존의 연구들은 일반적으로 단일 모드를 이용하여 수면 단계를 단독으로 분석하거나, 수면 장애를 별도로 분류하는 접근을 취했습니다. 그러나 본 연구는 수면 장애와 특정 수면 단계 간의 상관관계를 활용하여 두 요소를 동시에 식별할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 모델은 방향성이 있는 self-attention 메커니즘을 통해 지역적 및 전역적 종속성을 포착하고, 다양한 수면 장애와 수면 단계에서의 공통적인 특성을 활용합니다. 연구에서는 123명의 다양한 수면 장애가 있는 개인으로부터 수집된 다중 모드(multi-modal) 데이터를 이용하여 모델을 훈련하고 테스트하였으며, 기록된 데이터에는 photoplethysmogram (PPG), 호흡 흐름 신호( respiratory flow ), 호흡 노력 신호( respiratory effort )가 포함됩니다.

- **Performance Highlights**: 모델의 전체적인 분류 정확도는 5단계 수면 분류에서 78% (Cohen's Kappa 0.66), 수면 무호흡증 분류에서 74% (Cohen's Kappa 0.58)로 나타났습니다. 또한, 모델의 예측을 명확히 하기 위해 encoder attention weights를 분석하여 각 특성이 최종 분류에 미치는 영향을 조사하였습니다. 이를 통해 호흡의 저점과 정점을 포함한 특정 패턴들이 최종 분류 과정에 더 높은 기여를 한다는 결과를 얻었습니다.



### Urinary Tract Infection Detection in Digital Remote Monitoring: Strategies for Managing Participant-Specific Prediction Complexity (https://arxiv.org/abs/2502.17484)
- **What's New**: 이번 연구는 치매 환자에서의 요로 감염(Urinary Tract Infections, UTI) 조기 탐지를 위한 기존 기계 학습(Machine Learning, ML) 모델의 성능을 개선하는 데 중점을 두었습니다. 특히, 멀티레이어 퍼셉트론(Multilayer Perceptron, MLP)의 정교함을 높여 가정 환경의 변동성을 잘 처리하고 성별 공정성을 개선하였습니다. 이 연구는 특징 클러스터링, 손실 의존 클러스터링, 참여자 ID 삽입이라는 세 가지 주요 모델 설계를 구현하여 기존 MLP 모델 대비 그 성능을 비교했습니다.

- **Technical Details**: 연구에서는 손실 의존 MLP가 가장 큰 성과를 보였으며, 검증 정확도가 48.92%에서 72.60%로, 민감도가 27.44%에서 70.52%로 증가했습니다. 이 방법은 참가자별 데이터 변동을 해결하면서 모델의 공정성을 높였습니다. 연구는 다양한 데이터 출처에서 온 데이터를 처리하는 방법을 제안하며, 개별 가정을 공정하게 대표할 수 있도록 신뢰할 수 있는 모델을 개발하는 데 기여하고 있습니다.

- **Performance Highlights**: 연구 결과는 UTI 조기 발견에서의 신뢰성과 공정성을 높이는 보다 효과적인 접근법을 제공한다는 것을 시사합니다. 특히, 성별에 따른 모델의 공정성을 개선함으로써 치매 환자에 대한 보다 정확한 진단 및 치료 결정이 가능하게 됩니다. 이러한 결과는 임상 의사들이 UTI 위험을 더욱 효과적으로 감지하고 선별하는 데 도움을 주며, 조기 및 정확한 치료 결정을 쉽게 할 수 있도록 합니다.



### ConSense: Continually Sensing Human Activity with WiFi via Growing and Picking (https://arxiv.org/abs/2502.17483)
- **What's New**: 본 연구에서는 WiFi 기반 인간 활동 인식(HAR) 시스템을 위한 경량화된 동적 적응 학습 프레임워크인 ConSense를 제안합니다. ConSense는 기존 모델들이 기존 활동을 잊어버리는 문제를 해결하고 새로운 정보 통합을 지원하여 사용자 데이터를 저장할 필요 없이 지식 보존을 이룹니다. 이 프레임워크는 특히 transformer 아키텍처를 활용하여 공간적 및 시간적 특성을 효과적으로 캡처할 수 있습니다.

- **Technical Details**: ConSense는 multi-head self-attention (MHSA) 레이어 내에 각 작업의 데이터에 맞게 훈련된 작은 규모의 파라미터를 추가하였습니다. 이 작은 파라미터들은 prefixes로 불리며, 이전에 학습된 지식을 유지하면서도 새 정보를 통합하는 데 중요한 역할을 합니다. 또한, 다층 퍼셉트론(MLP)에서의 선택적 재훈련 전략을 통해 각 신경의 성능에 따라 가중치를 동적으로 조절함으로써 새로운 데이터에 적응하는 동시에 이전의 지식을 유지합니다.

- **Performance Highlights**: ConSense는 세 가지 공공 WiFi 데이터세트에서 평가되었으며, 기존의 여러 경쟁 모델보다 우수한 성능을 나타냈습니다. 또한, 더 적은 파라미터를 사용하여 효율성을 증명하였고, 이는 HAR의 클래스 증가 시나리오에서 실용성을 크게 향상시킵니다. 이 결과는 WiFi 기반 HAR에서 ConSense의 실용성을 잘 보여줍니다.



### Multi-View Contrastive Network (MCNet) for Motor Imagery Classification (https://arxiv.org/abs/2502.17482)
Comments:
          9 pages, 7 figures

- **What's New**: 이 논문은 BCI(Brain-Computer Interface)에서 MI(Motor Imagery) EEG 해독을 위한 새로운 접근법으로, 지식을 활용한 다중 뷰 대비 네트워크(MCNet)를 제안합니다. MCNet은 시간, 공간, 주파수 도메인의 지식을 통합하여 데이터 증진(data augmentation) 과정을 통해 더욱 구별력 있는 feature learning을 촉진합니다. 또한, 다양한 증강된 뷰의 데이터를 학습하는 교차 뷰 대비 모듈과, 지식 기반 및 데이터 기반 모델 간 기능 일관성을 향상시키는 교차 모델 대비 모듈을 도입하여 성능을 강화합니다.

- **Technical Details**: MCNet의 핵심은 다중 뷰 데이터 변환과 교차 뷰 대비 학습입니다. EEG 데이터를 시간, 공간, 주파수 Domain의 세 가지로 변환하여 다양한 관점에서의 데이터를 활용하고, 교차 모델 일관성 정규화를 통해 추출된 feature의 일관성을 보장합니다. 이러한 방법들을 통해 EEG 데이터의 특징을 효과적으로 캡처하고, 수업 정보(label information)를 활용하도록 설계되었습니다.

- **Performance Highlights**: MCNet은 네 가지 공용 MI 데이터셋과 세 가지 다른 아키텍처를 사용하여 총 10개의 기존 방법들보다 월등한 성능을 입증했습니다. 다양한 데이터 augmentation 전략과 함께 실험이 진행되었으며, 실증 결과는 MCNet이 있을 때 보다 우수한 디스크리미너티브 feature learning을 통해 EEG 분류 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework (https://arxiv.org/abs/2502.17481)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문은 SynthSleepNet이라는 다중 모달 하이브리드 자기 지도 학습(Self-Supervised Learning, SSL) 프레임워크를 소개합니다. 이 방법은 수면 분석을 위해 폴리솜노그래피(Polysomnography, PSG) 데이터를 효과적으로 통합합니다. 기존 방법과 차별화되는 점은 마스크된 예측(masked prediction)과 대조 학습(contrastive learning)을 융합하여 다양한 생리 신호 간의 보완 기능을 활용한다는 것입니다.

- **Technical Details**: SynthSleepNet은 뇌전도(Electroencephalogram, EEG), 안구전도(Electrooculography, EOG), 근전도(Electromyography, EMG), 심전도(Electrocardiogram, ECG) 등 여러 신호 모달리티에서 심층 표현 학습을 가능하게 합니다. 이 프레임워크는 Mamba 기반의 시간적 맥락 모듈을 통해 시간 축을 따라 신호 간의 맥락 정보를 효율적으로 포착하도록 설계되었습니다. 또한, LoRA(Low-Rank Adaptation)를 활용하여 정보를 정교하게 추출하며, 여러 개의 모달리티-specific backbone을 사용하여 신호 특성에 최적화된 인코더를 갖추고 있습니다.

- **Performance Highlights**: SynthSleepNet은 수면 단계 분류, 무호흡증 감지, 저호흡증 감지의 세 가지 후속 작업에서 각각 89.89%, 99.75%, 89.60%의 정확률을 기록하며 최신 기술 대비 우수한 성능을 보였습니다. 제한된 라벨이 있는 반지도 학습(semi-supervised learning) 환경에서도 87.98%, 99.37%, 77.52%의 정확성을 달성했습니다. 이 결과는 SynthSleepNet이 PSG 데이터의 종합 분석을 위한 기초 도구로서의 잠재력을 강조합니다.



### Frequency-Aware Masked Autoencoders for Human Activity Recognition using Accelerometers (https://arxiv.org/abs/2502.17477)
Comments:
          7 pages, 3 figures, submitted to 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)

- **What's New**: 본 논문에서는 신체 활동의 지속적 모니터링을 위해 널리 사용되는 wearable accelerometers에 대한 연구를 진행했습니다. 기존의 supervised machine learning과 deep learning 방법은 한정된 데이터로 인해 발전이 더디었으나, 자율 학습(self-supervised) 접근법을 통해 unlabeled 데이터 세트를 활용하는 방안이 제시되었습니다. 특히, 시간 시퀀스를 위한 transformer 기반의 masked autoencoder(MAE) 방법을 사용하여 새로운 log-scale mean magnitude(LMM) 손실 함수(loss function)를 제안합니다.

- **Technical Details**: 제안된 LMM 손실 함수는 기존의 mean squared error(MSE) 손실 함수와 비교하여 MAE 모델의 사전 훈련(pretraining)을 위한 새로운 수단을 제공합니다. 본 연구는 109,000명의 데이터가 포함된 UK Biobank accelerometry 데이터 세트를 활용하여 사전 훈련을 실시하였고, 이를 통해 작은 레이블이 있는 데이터세트에서 HAR 성능을 평가했습니다. LMM 손실로 사전 훈련된 모델은 MSE 손실로 훈련된 모델에 비해 성능이 개선된 것으로 나타났습니다.

- **Performance Highlights**: 사전 훈련을 통한 LMM 손실 사용은 조화 정확도(balanced accuracy)가 각각 0.848 및 0.709로 나타나는 등 성능이 유의미하게 향상되었습니다. LMM 손실의 더 나은 수렴(convergence)은 HAR의 다운스트림 성능 개선과 상당한 상관관계를 보였으며, 결국 LMM으로 사전 훈련된 MAE 모델이 최첨단 기술 대비 훨씬 뛰어난 성능을 보이는 것으로 확인되었습니다. 이러한 결과는 LMM 손실이 accelerometer 데이터에서 HAR를 위한 MAE 모델 사전 훈련에 있어 강력하고 효과적인 방법임을 보여줍니다.



### Fusion of ECG Foundation Model Embeddings to Improve Early Detection of Acute Coronary Syndromes (https://arxiv.org/abs/2502.17476)
- **What's New**: 이 연구는 급성 관상동맥 증후군(Acute Coronary Syndrome, ACS)의 조기 및 정확한 진단을 위한 ECG 기반 모델(ST-MEM 및 ECG-FM)의 활용을 탐구합니다. 특히, 구급차에서 수집된 ECG 데이터를 사용하여 ACS 위험 평가를 개선하는 방법을 제시합니다. 연구 결과는 이러한 모델들이 기존의 ResNet-50 모델을 초월하면서 fusion 방법이 최상의 성능을 나타낸다는 것을 보여줍니다.

- **Technical Details**: ST-MEM은 재구성 기반 접근법을 사용하고, ECG-FM은 대조 학습(contrastive learning)을 채택하여 독특한 공간적 및 시간적 ECG 특징을 캡처합니다. 두 모델 모두 자가 감독 학습(self-supervised learning, SSL)을 활용하며, 이들의 임베딩(embeddings)을 결합하여 예측 정확성을 향상시키는 방법도 평가됩니다. 이는 모델 간의 보완적인 특징을 극대화하기 위한 고급 융합(fusion) 전략의 필요성을 강조합니다.

- **Performance Highlights**: 모델의 성능 평가 결과, ST-MEM과 ECG-FM 모두 기준 모델인 ResNet-50보다 뛰어난 성능을 보였으며, 융합 기반 접근 방식이 가장 높은 성과를 기록했습니다. 특히, AUROC(Receiver Operating Characteristic Area Under Curve)는 0.843 +/- 0.006, AUCPR(Area Under the Precision-Recall Curve)은 0.674 +/- 0.012로 확인되었습니다. 이러한 결과는 ECG 기반 모델들이 조기 ACS 감지에 대한 잠재력을 지니고 있음을 강조합니다.



### ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis (https://arxiv.org/abs/2502.17475)
- **What's New**: ECG-Expert-QA는 ECG 해석에서 진단 능력을 평가하기 위한 포괄적인 멀티모달 데이터셋으로, 실제 임상 데이터와 체계적으로 생성된 합성 사례를 통합하였습니다. 이 데이터셋은 47,211개의 질문-답변 쌍으로 구성되어 있으며, 복잡한 사례 해석을 포함한 다양한 임상 시나리오를 다룹니다. 이를 통해 고유한 진단 작업을 통해 의료 언어 모델을 종합적으로 평가할 수 있는 기반이 마련되었습니다.

- **Technical Details**: 이 연구는 전통적인 평가 방법의 효율성 문제와 데이터셋의 복잡성 부족을 해결하기 위해 ECG-Expert-QA 데이터셋을 개발했습니다. 데이터셋은 기본 의료 지식 검증 모듈, 임상적 추론 평가 모듈, 위험 관리 모듈의 세 가지 핵심 평가 모듈로 구성되며, 이는 환자 예후 예측, 다중 모달 정보 통합 등을 포함합니다. 또한, 데이터셋은 윤리적 차원도 포함하여, 의료 AI의 결정 안전성을 평가하기 위한 기준을 제시합니다.

- **Performance Highlights**: 이 데이터셋은 기존의 의료 데이터셋에 비해 복잡한 진단 작업 비율이 15.3%에 달하며, 다양한 언어로 지원되어 cross-cultural 연구에 기여합니다. 특히, innovative evaluation dimensions인 반사적 추론과 기억 교정 메커니즘을 도입하여 모델의 의사 결정 논리의 임상적 합리성과 강건성을 평가합니다. 이러한 특성으로 인해 ECG-Expert-QA는 AI 보조 ECG 해석 발전의 중요한 벤치마크로 자리잡고 있습니다.



### PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Lev (https://arxiv.org/abs/2502.17469)
- **What's New**: 이번 연구는 개인의 일상 생활과 건강에 대한 귀중한 통찰력을 제공하는 lifelog 데이터 분석 방법론을 제시합니다. 특히, PixleepFlow라는 이미지 기반의 수면 질 및 스트레스 수준 추정 모델을 개발하여 다양한 센서 데이터를 통합하여 다각적인 지표를 동시에 예측할 수 있는 가능성을 제시합니다. 이는 전통적인 시계열 데이터 분석과는 달리, 이미지 기반 입력 방식으로 데이터를 시각적으로 직관적으로 표현합니다.

- **Technical Details**: PixleepFlow는 다양한 센서 데이터를 조합하여 세 가지 RGB 이미지로 변환하는 과정에서 고차원 데이터를 단순화합니다. 이 접근 방식은 세분화된 세부사항보다는 전체적인 패턴과 이상 징후를 식별하는 데 유리합니다. 또한, Explainable Artificial Intelligence(XAI) 기술을 통해 예측 결과를 시각적으로 해석할 수 있도록 하여, 모델의 투명성과 설명 가능성을 높였습니다.

- **Performance Highlights**: 연구 결과, PixleepFlow는 각기 다른 데이터 형식보다 더 중요한 성과를 보여주었으며, 일상적인 데이터 수집을 통해 인간의 삶의 질을 이해하는 데 기여하고 있습니다. 이 모델은 수면 질 및 스트레스 수준을 동시에 예측할 수 있는 다중 레이블 분류 방법을 적용하여, 7개의 핵심 지표를 추정하고 높은 정확도를 달성하는 데 성공하였습니다.



### CSSSTN: A Class-sensitive Subject-to-subject Semantic Style Transfer Network for EEG Classification in RSVP Tasks (https://arxiv.org/abs/2502.17468)
- **What's New**: 이 연구에서는 Rapid Serial Visual Presentation (RSVP) 패러다임에서 BCI를 지원하기 위한 Class-Sensitive Subject-to-Subject Semantic Style Transfer Network (CSSSTN)이 제안됩니다. CSSSTN은 BCI 전문가와 BCI 비문해 사용자 간의 특징 분포를 클래스별로 정렬하여 교차 주체 변동성을 해결하려고 합니다. 이는 것이 다양한 EEG 기반 BCI 응용 시스템의 이점을 극대화하는 혁신적인 방법입니다.

- **Technical Details**: CSSSTN은 세 가지 주요 구성 요소로 구성됩니다: (1) 주체 특화 분류기 훈련, (2) 수정된 내용 손실을 통해 의미 정보를 유지하면서 클래스 구별 특성을 전송하는 독특한 스타일 손실, (3) 출처 및 목표 도메인에서 예측을 통합하는 앙상블 접근 방식입니다. 실험 결과, CSSSTN은 Tsinghua 데이터세트에서 6.4%와 HDU 데이터세트에서 3.5%의 평균 균형 정확도 개선을 초래했습니다.

- **Performance Highlights**: CSSSTN은 적은 목표 데이터로 경쟁력 있는 결과를 도출하며, 훈련 데이터에 대한 의존성을 최소화하면서 BCI 비문해 사용자의 성과를 향상시킬 수 있는 잠재력을 지니고 있습니다. 특히 각 구성 요소의 효과가 입증되었으며, 클래스 별 전송과 하층 특징의 사용이 전송 성능을 향상시키고 부정적인 전송을 완화하는 데 기여했습니다.



### Bridging Brain Signals and Language: A Deep Learning Approach to EEG-to-Text Decoding (https://arxiv.org/abs/2502.17465)
Comments:
          21 pages, 11 figures, and 6 tables

- **What's New**: 이 연구는 EEG(뇌파) 신호를 텍스트로 변환하는 전통적인 닫힌 어휘(closed-vocabulary) 접근 방식을 넘어서, 개인 특정 학습 모델을 자연어 처리(NLP) 방법과 통합하는 특별한 프레임워크를 소개합니다. 이로 인해 EEG 신호가 개별뇌 및 의미의 깊이를 포착하며, 더욱 풍부한 문장을 생성할 수 있도록 혁신적인 방안을 모색합니다. 연구는 신경망을 훈련시켜 매우 의미 있는 텍스트 생성을 가능하게 하며, ZuCo 데이터셋 분석을 통해 다른 방법들과 비교했을 때 향상된 성능을 보여줍니다.

- **Technical Details**: 이 프레임워크는 양방향 GRU와 변환기 인코더를 사용하여 개인 특화 피쳐를 추출하는 뇌 모듈, EEG에서 얻은 특성을 활용하여 개방된 어휘 텍스트를 생성하기 위해 BART를 통합한 언어 모듈, 의미적 및 문법적 수정을 위해 GPT-4를 활용하는 정제 모듈로 구성됩니다. 사용자 친화적인 웹 플랫폼을 통해 EEG 데이터를 업로드하고 텍스트 생성을 비음성화하는 등의 기능이 구현되어 있으며, 이는 다양한 EEG 데이터셋과 하드웨어와의 호환성을 보장합니다.

- **Performance Highlights**: 연구 결과는 BLEU, ROUGE 및 BERTScore와 같은 성능 지표에서 현재 사용되는 방법들에 비해 더 높은 성과를 달성했습니다. 이 접근법은 개인 간의 뇌 신호 변동성을 이해하고 의미있는 텍스트 생성을 효과적으로 수행할 수 있음을 입증합니다. 이 시스템은 어떤 환경에서도 효율적인 뇌-주관 텍스트 시스템을 개발하는 데 기여할 수 있으며, 혁신적인 보조 기술 개발과 개인화된 의사소통 시스템을 통해 인류와 컴퓨터 간의 상호작용 가능성을 확장하는 데 중점을 두고 있습니다.



### Large Cognition Model: Towards Pretrained EEG Foundation Mod (https://arxiv.org/abs/2502.17464)
- **What's New**: 이번 연구에서는 대규모 EEG 기반 모델인 Large Cognition Model(LCM)을 제안합니다. LCM은 다양한 EEG 데이터셋 및 다운스트림 작업에 대해 일반화할 수 있도록 설계된 트랜스포머 기반의 모델입니다. 기존의 방법들과는 달리, LCM은 대규모 자기 지도 학습 기법을 적용하여 효율적인 미세 조정을 가능하게 하여 인지 상태 디코딩, 질병 분류, 신경 피드백 시스템 등 여러 응용 분야에서 우수한 성능을 보입니다.

- **Technical Details**: LCM은 온라인 인코더 fθ와 모멘텀 업데이트된 타겟 인코더 fξ로 구성된 자기 지도 대조 학습 프레임워크입니다. 이 모델은 대조 손실을 통한 표현 정렬, 마스킹된 기능 재구성 손실, 그리고 특징 학습의 안정성을 조정하는 적응형 최적화 계획을 포함합니다. MASK 방식으로 데이터의 일부분을 숨기고 잃어버린 구성 요소를 예측하는 과정을 통해 EEG 신호의 복잡한 시공간 관계를 포착합니다.

- **Performance Highlights**: LCM은 다양한 EEG 벤치마크에서 기존의 최첨단 방법들을 초월하는 성능을 보이며, 동일한 작업이나 데이터셋 간의 일반화에서 강한 결과를 나타냅니다. 또한 사전 훈련 없이도 특정 응용 프로그램에서 기존의 EEG 모델들을 초과하여 우수한 성능을 입증했습니다. 이러한 결과는 EEG 기초 모델의 발전이 신경 과학, 개인 맞춤형 의학 및 BCI(Brain-Computer Interface) 기술의 발전을 가속화할 수 있음을 보여줍니다.



### SincPD: An Explainable Method based on Sinc Filters to Diagnose Parkinson's Disease Severity by Gait Cycle Analysis (https://arxiv.org/abs/2502.17463)
- **What's New**: 이번 논문은 파킨슨병(Parkinson’s Disease, PD) 진단을 위한 설명 가능한 AI 기반 분류기 SincPD를 제안합니다. 이 시스템은 환자와 건강한 사람의 보행 주기를 분석하여 프로세스의 설명 가능성을 제공합니다. Sinc 레이어는 적응형 밴드패스 필터를 모델링하여 보행 주기에서 중요한 주파수 대역을 추출하고, 이를 통해 진단 결과의 근거를 설명합니다.

- **Technical Details**: 제안된 SincPD 방법은 착용 가능한 센서로 측정된 수직 지면 반응력(vertical Ground Reaction Force, vGRF)의 원시 데이터를 활용합니다. 모델은 먼저 많은 필터로 구성된 대형 모델을 훈련한 후, 클러스터링을 통해 불필요한 단위를 제거하고 최종 필터로 클러스터의 중심을 선택합니다. 이를 통해 각 센서에서 15개의 밴드패스 필터가 도출되며, 이러한 필터의 에너지를 비교하여 환자와 건강한 피실험자 간의 중요한 주파수를 분석합니다.

- **Performance Highlights**: 제안된 방법은 기존의 수작업으로 특징을 추출하는 기법에 비해 높은 분류 정확도를 제공합니다. Sinc 필터를 사용한 CNN 아키텍처는 복잡성을 줄여주면서도 중요한 주파수 대역을 효과적으로 학습합니다. 이러한 방식은 환자의 PD 진단 및 중증도 평가에 있어 임상적 관련성을 높이는 데 기여합니다.



### The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG (https://arxiv.org/abs/2502.17462)
Comments:
          Published at ICLR 2025, see this https URL. Code is available at this https URL

- **What's New**: 본 논문은 EEG(Scalp Electroencephalogram)와 iEEG(Intracranial Electroencephalogram)의 데이터 압축 성능을 향상시키는 새로운 신경 네트워크 모델인 BrainCodec을 소개합니다. BrainCodec은 두 가지 데이터 모달리티에 대해 높은 재구성 품질을 달성하며, 특히 iEEG를 이용해 훈련한 후 EEG에 전이할 경우 더 뛰어난 성능을 보입니다. 또한, EEG와 iEEG를 모두 사용하여 훈련 시 신뢰성을 높이고 재구성 성능을 개선하는 것을 보여줍니다.

- **Technical Details**: BrainCodec은 EEG와 iEEG 신호를 신경망 압축하여 최대 64배의 압축 비율을 달성합니다. 이 모델은 고품질의 신호에서 훈련되었을 때 지정된 성능 기준을 충족하는 높은 재구성 충실도를 보입니다. 구체적으로, 제안한 기준은 PRD(percentage root-mean-square difference) 30 이하 및 시퀀스 분류 성능이 1% 미만의 저하를 포함합니다.

- **Performance Highlights**: BrainCodec은 EEG와 iEEG 신호의 손실 압축에서 기존의 최첨단 압축 모델들을 초월한 성능을 보여줍니다. 시각적 평가를 통해 신경 학자가 BrainCodec의 높은 재구성 품질을 확인하였으며, 다운스트림 작업인 발작 탐지 및 운동 이미징 과제에 대해서도 성능 저하가 없음을 보였습니다. 결과적으로, BrainCodec은 의료 타임 시리즈 도메인에서 더 높은 SNR을 가진 데이터 소스가 더 높은 성능을 발휘할 수 있음을 뒷받침합니다.



### Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation (https://arxiv.org/abs/2502.17460)
Comments:
          7 pages, 1 figure, 5 tables, preprint

- **What's New**: 이번 연구는 Electroencephalogram (EEG) 데이터를 이용해 사전 훈련된 모델이 Electrocardiogram (ECG)와 Photoplethysmogram (PPG) 데이터에서 혈압(BP) 추정에 효과적으로 전이 학습될 수 있음을 실험적으로 입증한 최초의 연구입니다. 추가적인 대규모 사전 학습 없이 최소한의 세밀한 조정만으로도 가능하다는 점이 특징입니다. 이를 통해 혈압 모니터링 기술에서 다양한 생체 신호 간의 융합 가능성을 제시합니다.

- **Technical Details**: 이 연구는 기존 EEG 기반의 foundation model인 CEReBrO를 활용합니다. 이 모델은 EEG 신호를 기반으로 하여 미리 훈련된 후, ECG와 PPG 신호로 혈압을 예측하도록 세밀하게 조정됩니다. 그 과정에서 동적 INT8 양자화를 적용하여 모델 크기를 3.5배 이상 감소시켰고, 이는 자원 제약이 있는 wearable 장치에서의 실시간 혈압 모니터링을 가능하게 합니다.

- **Performance Highlights**: MIMIC-III 및 VitalDB 데이터셋에서 수행한 평가 결과, 이 접근 방식은 이완기 혈압의 평균 절대 오차 1.57 mmHg로 거의 최신 기술과 동등한 정확성을 달성했으며, 수축기 혈압의 경우 이전 작업보다 1.5배 높은 정확성을 보였습니다. 이러한 성과는 예측 정확성과 계산 비용의 절충안에서도 중요한 의미를 가지며, 혈압 모니터링 장치의 실용화에 기여할 수 있습니다.



### Study on Downlink CSI compression: Are Neural Networks the Only Solution? (https://arxiv.org/abs/2502.17459)
- **What's New**: 이번 연구에서는 Massive MIMO 시스템의 다운링크 채널 상태 정보를 효과적으로 전송할 수 있는 새로운 방법으로, AI/ML 기반이 아닌 주성분 분석(Principal Component Analysis, PCA)을 활용한 방법론을 제안합니다. 기존 AI/ML 모델들이 가지는 모델 복잡성, 일반화 문제, 그리고 다른 제조사 간의 호환성 문제를 해결할 수 있습니다. PCA를 통해 기본적인 차원 축소를 수행하여 CSI을 압축하고, 재구성하는 성능을 평가했습니다.

- **Technical Details**: 이 연구에서는 두 가지 서로 다른 무선 채널 표현인 각도-지연(domain) 표현과 고유벡터(eigenvector) 표현을 사용하였습니다. N개의 서브캐리어를 가진 OFDM 시스템을 모델로 하여, 채널 매핑을 수행하고 PCA를 적용하여 각기 다른 송수신 안테나에서의 독립적인 시간 주파수 채널을 찾아냈습니다. PCA는 독립된 다양한 채널을 활용하여 가장 적은 수의 성분을 선택함으로써, 채널의 각도 특성을 효과적으로 캡처합니다.

- **Performance Highlights**: 시뮬레이션 결과, PCA 기반의 CSI 압축 방법이 기존의 딥 뉴럴 네트워크 모델들과 비슷한 수준의 재구성 성능을 제공하는 것으로 나타났습니다. 정보 전송의 오버헤드를 줄이면서도 성능 저하 없이 유사한 성과를 이룰 수 있음을 보여줍니다. 따라서, PCA 접근 방식은 CSI 압축의 효율성을 증대시키는 잠재력이 있음을 확인하였습니다.



### MoEMba: A Mamba-based Mixture of Experts for High-Density EMG-based Hand Gesture Recognition (https://arxiv.org/abs/2502.17457)
- **What's New**: 본 논문은 MoEMba 프레임워크를 통해 HDsEMG를 기반으로 한 제스처 인식을 개선하는 새로운 방법론을 제시합니다. MoEMba는 Selective State-Space Models (SSMs)을 활용하여 세션 간 및 주체 간 변동성 문제를 해결합니다. 이를 통해 시간 의존성과 채널 간 상호작용을 효과적으로 포착하며, 신호 표현을 향상시키기 위해 웨이브렛 변환 기능 조정(WTFM)을 통합했습니다.

- **Technical Details**: 제가 제안한 MoEMba 프레임워크는 다중 Mamba 전문가의 적응형 조합을 활용하여 짧은-장기 제스처 동역학을 포착합니다. 또한, 이 구조는 상대적으로 적은 계산량(FLOPS)으로 연산 효율성을 달성하며 고밀도 HD-sEMG 데이터의 세션 간 변동성에 강한 내성을 보입니다. MoEMba는 채널 주의(chanel attention)를 통해 채널 간 상호작용을 다룹니다.

- **Performance Highlights**: 실험 결과에 따르면, MoEMba는 CapgMyo HD-sEMG 데이터셋에서 56.9%의 균형 잡힌 정확도를 기록하며 기존 최첨단 모델들을 초월했습니다. 이는 MoEMba 프레임워크가 HD-sEMG 기반 HCI 시스템에서의 가능성을 강조하는 것이며, 프로세틱 제어 및 인간-컴퓨터 상호작용과 같은 실제 응용에 적합하다는 것을 보여줍니다.



### Survey on Recent Progress of AI for Chemistry: Methods, Applications, and Opportunities (https://arxiv.org/abs/2502.17456)
Comments:
          22 pages, 8 figures, 4 tables

- **What's New**: 인공지능(AI) 기술의 발전은 다양한 분야에서 혁신을 가져왔습니다. 특히 AI를 활용한 화학 연구의 가속화가 점차 성과를 내고 있으며, 많은 혁신적인 작업들이 이루어지고 있습니다. 이 논문에서는 현재 화학 분야에서 사용되는 AI 기법을 포괄적으로 검토하고, 데이터의 특성과 다양한 표현 방법을 소개하며, 주요 임무에 대한 모델을 개관합니다.

- **Technical Details**: AI는 일반적으로 데이터, 표현 및 모델의 세 가지 중요한 구성 요소로 구성됩니다. 머신러닝(ML)에서는 고품질의 다양한 데이터가 필수적이며, 데이터는 적절한 기계 인식 형식으로 변환되어야 합니다. 이 논문에서는 분자 수준 데이터와 반응 수준 데이터와 같은 다양한 화학 데이터셋을 소개하고, 이들을 활용하기 위한 분자 식별자와 설명자를 설명합니다.

- **Performance Highlights**: 최근의 연구들에서, 고속 처리 능력과 자동 합성 기술의 발전은 화학에서 머신러닝 방법의 개발에 대한 증가하는 관심을 불러일으켰습니다. 특히 대형 언어 모델(LLMs)의 활용은 화학 연구를 위한 에이전트 시스템 개발로 이어지고 있으며, 이들은 다운스트림 작업의 개선에도 기여하고 있습니다. 이 논문은 AI 기술을 화학에 적용하기 위한 여러 가지 중요한 문제들을 강조하며 향후 발전을 지원하는 기초 자료로 기능하고 있습니다.



### DCentNet: Decentralized Multistage Biomedical Signal Classification using Early Exits (https://arxiv.org/abs/2502.17446)
- **What's New**: DCentNet는 IoT 웨어러블 센서에서 생체의료 데이터의 분산 멀티스테이지 신호 분류를 위한 혁신적인 접근 방식입니다. 기존의 중앙 집중식 처리 방식과 달리, DCentNet은 조기 종료 지점(EEP)을 도입하여 에너지 효율성과 처리 속도를 향상시킵니다. 이 시스템은 대형 CNN 모델을 여러 하위 네트워크로 나누어 데이터를 전송하기 전에 대형 특징 맵을 압축합니다.

- **Technical Details**: DCentNet은 EEP를 통해 CNN 모델을 여러 하위 네트워크로 분할하고, 각 하위 네트워크를 다른 노드에 배포하여 높은 신뢰성의 분류를 제공합니다. 유전 알고리즘을 사용하여 EEP 배치를 최적화하며, 초기 하위 네트워크에서 높은 정확성이 유지됩니다. 실험 결과, 한 개의 EEP을 사용할 경우 무선 데이터 전송이 94.54% 감소하고 복잡성이 21% 감소하며 원래의 정확도와 민감도를 유지합니다.

- **Performance Highlights**: DCentNet는 ARM Cortex-M4 MCU에서 구현되어 평균 73.6%의 전력 절약을 달성했습니다. 두 개의 EEP을 적용할 경우 민감도가 98.36%, 정확도가 97.74%에 도달하였으며, 추가적인 무선 데이터 전송 감소율은 91.86%에 달합니다. 분산 시스템의 장점으로 인해 시스템의 비용 효율성과 유연성이 크게 향상됩니다.



### Renaissance of Literate Programming in the Era of LLMs: Enhancing LLM-Based Code Generation in Large-Scale Projects (https://arxiv.org/abs/2502.17441)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models) 및 문서 지향 작업에서 ILP(Interoperable Literate Programming)의 활용을 통해 코드 생성의 효율성을 높이는 새로운 접근 방식을 제안합니다. 기존의 문서 지향 작업을 넘어서, 대규모 프로젝트의 관계를 강화할 수 있는 가능성을 탐구합니다. 나아가, 프롬프트 엔지니어링(prompt engineering) 방법론을 제시하여 LLM이 코드 생성을 보다 효과적으로 지원할 수 있도록 합니다.

- **Technical Details**: ILP는 문서 작성 및 대규모 프로젝트의 개발을 개선하기 위해 리터레이트 프로그래밍(Literate Programming) 원리를 활용합니다. 특히 이번 연구에서는 LLM이 ILP 스타일 지침 하에서 문서와 프로젝트를 수행하는 방식을 분석합니다. 또한, RepoBench 벤치마크를 통해 다양한 LLM이 Scheme 및 Python 코드를 생성하는 능력을 평가하며, LLM의 코드 생성 시 효용성을 강조합니다.

- **Performance Highlights**: 우리의 연구 결과는 ILP와 LLM의 결합이 대규모 프로젝트 개발에서 LLM 기반 코드 생성을 향상시킬 수 있음을 보여줍니다. 대형 코드베이스와 복잡한 상호 의존성으로 인해 코딩 작업이 어려운 상황에도 불구하고, 본 접근 방식이 효율적인 코드 작성과 이해를 가능하게 돕는다는 점에서 의미가 있습니다.



### GenAIOps for GenAI Model-Agility (https://arxiv.org/abs/2502.17440)
Comments:
          8 pages, 3 figures, 2 tables

- **What's New**: 이번 논문에서는 generative AI 애플리케이션의 개발 및 운영에서 AI 민첩성(AI Agility)이 요구됨을 강조합니다. 특히 'GenAI 모델 민첩성(GenAI Model Agility)'이라는 개념을 소개하며, 다양한 모델 제공자와 버전에 유연하게 적응할 수 있는 준비성을 정의합니다. GenAIOps라는 방법론을 정의하여, 기초 모델의 변화로 인한 애플리케이션 품질 저하 문제를 다루려고 합니다.

- **Technical Details**: 논문에서는 기존의 기계 학습(Machine Learning) 및 generative AI 모델을 관리하기 위한 프로세스인 GenAIOps를 소개합니다. 이 과정은 여러 단계로 나뉘며, 초기에 계획, 개발, 테스트, 출시, 관찰 단계로 진행됩니다. 또한 모델 전환 시 예상되는 문제와 이러한 문제를 해결하기 위한 기존 연구 및 도구들을 조사하고 이에 대한 솔루션과 한계점을 논의합니다.

- **Performance Highlights**: 기술적으로, Soft prompt tuning과 자동화된 프롬프트 엔지니어링(Automatic Prompt Engineering)이 Generative AI의 민첩성을 높이기 위한 중요한 기술로 언급됩니다. 이에 따른 성공적인 모델 전환과 사용자 피드백을 통해 애플리케이션 품질 유지가 가능함을 보여줍니다. 결과적으로 이 논문은 다양한 모델을 사용하는 빠른 응답성이 중요한 비즈니스 환경에서의 Generative AI 개발 프로세스 기틀을 제공합니다.



### ELLEN: Extremely Lightly Supervised Learning For Efficient Named Entity Recognition (https://arxiv.org/abs/2403.17385)
Comments:
          Accepted to LREC-COLING 2024

- **What's New**: 본 연구에서는 극히 적은 주석으로 구성된 세미-슈퍼바이즈드(named entity recognition, NER) 방법론을 제안하고 있습니다. 새로운 메서드 ELLEN은 10개의 예시로 구성된 사전을 사용하여 언어 모델과 언어 규칙을 융합하여 기존의 복잡한 방법들보다 성능이 뛰어난 결과를 보여줍니다. 연구진은 제한된 감독 하에서도 강력한 성능을 발휘할 수 있는 효율적인 방법을 제시하고 있습니다.

- **Technical Details**: ELLEN은 미세 조정된(language model) 언어 모델을 언어 규칙과 결합하여 작동하는 모듈형 신경-상징 방법입니다. 여기에는 'One Sense Per Discourse'와 같은 언어적 인사이트와 함께 Masked Language Model을 활용하여 비지도 NER을 수행하는 방식이 포함됩니다. 특히, Encoder only 전략을 사용하며, 훈련 시 여러 전략을 결합해 동작하게 설계되었습니다.

- **Performance Highlights**: 연구팀은 CoNLL-2003 데이터셋을 통해 제안한 방법이 76.87%의 F1 스코어를 달성했음을 보였습니다. ELLEN은 또한 WNUT-17 데이터셋에서 제로샷(zero-shot) 접근을 통해 GPT-3.5 및 GPT-4와 비교할 수 있는 성능을 기록했습니다. 이러한 결과는 ELLEN이 기존의 복잡한 세미-슈퍼바이즈드 NER 방법보다 우수한 성능을 보여준다는 것을 의미합니다.



