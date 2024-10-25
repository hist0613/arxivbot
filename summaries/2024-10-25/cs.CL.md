New uploads on arXiv(cs.CL)

### Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions (https://arxiv.org/abs/2410.18966)
Comments:
          2 tables and 1 figures in the main text. This is a preprint, under review

- **What's New**: 이번 연구는 데이터 오염(data contamination) 검사 방법에 대한 47개의 논문을 체계적으로 검토하고, 이와 관련된 가정을 평가합니다. 이는 기존의 접근 방식들이 다양한 환경에서 보편적으로 적용되지 않을 수 있는 특정 가정에 의존하고 있음을 강조합니다.

- **Technical Details**: 연구는 데이터 오염을 3단계로 분류하며, 인스턴스(instance) 수준과 데이터셋(dataset) 수준에서 일어나는 오염을 수학적으로 정의합니다. 각 접근 방식의 기본 가정과 요구사항을 체계적으로 정리하고, 그 가정들이 다양한 시나리오에서 검증되었는지를 평가합니다. 이 과정에서 3가지 주요 가정이 사례 연구를 통해 분석되었습니다.

- **Performance Highlights**: 현재 LLMs는 개별 인스턴스를 암기하는 것이 아니라 데이터 분포를 학습하는 경향이 있으며, 연구에서는 일부 검출 접근 방식이 무작위 추측에 가까운 성능을 보인다고 지적합니다. 이는 데이터 오염을 탐지할 때 더 명확한 방법론의 필요성을 강조합니다.



### Bridge-Coder: Unlocking LLMs' Potential to Overcome Language Gaps in Low-Resource Cod (https://arxiv.org/abs/2410.18957)
Comments:
          15 pages, 3 figures

- **What's New**: 이번 연구에서는 Low-Resource Programming Languages (LRPLs)에서의 성능 향상을 위해 Bridge-Coder라는 혁신적인 접근 방식을 제안합니다. LLMs는 High-Resource Programming Languages (HRPLs)에서는 뛰어난 성능을 보이지만, LRPLs에서는 성능 저하가 심각합니다. 이러한 성능 격차를 해결하기 위해 HRPLs의 데이터를 활용한 새로운 방법을 도입하였습니다.

- **Technical Details**: Bridge-Coder는 두 단계로 구성됩니다. 첫 번째 단계는 Bridge-Assisted Generation으로, LLM의 일반적인 추론 능력을 활용하여 HRPL에서 질 높은 답안을 생성하고, 이를 통해 LRPL의 요구 사항에 맞는 결과를 도출합니다. 두 번째 단계는 Bridged Alignment로, NL-PL Gap을 줄이기 위해 LLM이 직접 LRPLs에 대한 응답을 생성할 수 있도록 단계적으로 지원합니다.

- **Performance Highlights**: 실험 결과 LRPLs에서 Bridge-Coder의 사용이 LLM의 성능을 크게 향상시킨 것으로 나타났습니다. 특히, 다양한 LRPLs에서의 실험을 통해 모델의 지시 수행 능력이 향상된 것을 확인되었습니다. 이는 LRPLs 분야에서의 새로운 가능성을 열어주는 중요한 성과입니다.



### BioMistral-NLU: Towards More Generalizable Medical Language Understanding through Instruction Tuning (https://arxiv.org/abs/2410.18955)
Comments:
          3 figures an 5 tables

- **What's New**: 본 논문에서는 의학 분야의 자연어 이해(NLU) 작업에서 성능 향상을 위한 새로운 접근 방식을 제안합니다. 특히, BioMistral-NLU라는 모델을 개발하여 7가지 중요한 NLU 작업을 위한 통합 프롬프트 형식을 제시하고, MNLU-Instruct라는 데이터셋을 활용하여 개선된 성능을 보입니다.

- **Technical Details**: (1) 7가지 주요 NLU 작업을 위한 통합 프롬프트 포맷을 제안하고, (2) 기존 오픈 소스 의학 NLU 데이터를 활용한 MNLU-Instruct 데이터셋을 구축하며, (3) BioMistral-NLU 모델을 BioMistral에 대한 미세 조정을 통해 개발했습니다.

- **Performance Highlights**: BioMistral-NLU는 6개의 주요 NLU 작업에 대해 0-shot 환경에서 평가되었으며, 기존 BioMistral 및 상용 LLM인 ChatGPT, GPT-4보다 우수한 성능을 보였습니다. 다루는 작업의 다양성과 분명한 지침 조정이 LLM의 일반화 능력을 향상시켰음을 입증합니다.



### Dynamic Vocabulary Pruning in Early-Exit LLMs (https://arxiv.org/abs/2410.18952)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 효율적인 추론을 위한 새로운 접근 방식을 제안합니다. 특히, 'early-exiting' 기술을 통해 중간 층에서 다음 토큰 예측을 가능하게 하여 효율성을 향상시키고자 합니다. 기존의 복잡한 사전 확률 추정 문제를 해결하기 위해, 테스트 시간에 동적으로 어휘(vocabulary)를 가지치기(prune)하는 방법을 도입하였습니다.

- **Technical Details**: 어휘 공간의 크기를 줄임으로써 각 토큰에 대한 사후 동적 어휘 가지치기를 진행합니다. 이는 초기 몇 개의 후보 배출에서 모델의 은닉 표현(hidden representation)을 전체 어휘에 매핑하지 않고, 가장 가능성이 높은 K개(K) 토큰을 선별합니다. 이후 이들 토큰을 기반으로 가중치 행렬을 가지치기하고, 나머지 후보 배출에서도 이 가지치기된 가중치를 사용합니다. 이 방법은 통해 성능을 유지하면서도 더 적은 연산량으로 신뢰도 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 동적 어휘 가지치기는 early-exit LLM의 신뢰도 추정 효율성을 향상시켜 FLOPs와 시간 효율성을 개선하는 데 성공했습니다. 또한, 모델의 성능을 유지하면서도 효율성을 증가시키는 경량 설계를 구현하였습니다.



### From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems (https://arxiv.org/abs/2410.18921)
- **What's New**: 본 연구는 기존의 대형 언어 모델(LLMs)이 논리적 불일치를 식별할 수 있는 능력을 평가하기 위해 FaultyMath라는 벤치마크 데이터셋을 제안합니다. 이 데이터셋은 일반 상식, 애매한 진술, 수학적 모순 등 다양한 문제를 포함합니다.

- **Technical Details**: FaultyMath 데이터셋은 여러 수학 범주(예: algebra, geometry, number theory)와 난이도 수준을 포함하며, LLM의 성능을 평가하기 위해 Simple Prompt 및 Chain-of-thought prompt와 같은 다양한 평가 기법을 사용합니다. LLM의 성능을 측정하기 위한 주요 지표는 F1 Measure, Accuracy, True Positives, False Positives 등입니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM은 여전히 Blind Solver로 작동하며, 논리적 사고를 요구하는 상황에서 필요한 추론 능력이 부족함을 보여주었습니다. 특히 Simple Prompt를 사용한 GPT-4가 모든 지표에서 가장 뛰어난 성능을 보였으며, 이는 인간 평가 결과와 가장 근접한 것으로 평가됩니다.



### PRISM: A Methodology for Auditing Biases in Large Language Models (https://arxiv.org/abs/2410.18906)
- **What's New**: 본 논문은 PRISM이라는 유연하고 질문 기반의 방법론을 제안하며, LLM의 편견과 선호를 간접적으로 탐색하는 새로운 방식을 제시합니다.

- **Technical Details**: PRISM은 첫째, 정치적 성향을 평가하는 도구(예: Political Compass Test)를 선택하고, 둘째, 감사할 LLM을 선택한 후, 셋째, 모델에 역할을 할당하고, 넷째, 에세이를 작성하도록 지시하며, 다섯째, 평가자를 통해 에세이의 입장을 그에 따라 판단하는 방식으로 이루어집니다. 이러한 단계적 접근은 LLM의 기본 및 기꺼이 주장할 수 있는 위치를 매핑하는데 사용됩니다.

- **Performance Highlights**: 연구 결과, 감사된 21개의 LLM이 기본적으로 경제적으로 좌파적이며 사회적으로 자유주의적 성향을 갖고 있다는 것을 발견했습니다. PRISM은 LLM의 편향, 선호 및 제약을 보다 신뢰성 있게 탐색하는 데 유용함을 증명했습니다.



### LLMs for Extremely Low-Resource Finno-Ugric Languages (https://arxiv.org/abs/2410.18902)
- **What's New**: 이 논문은 Finno-Ugric 언어 가족의 저자원(low-resource) 언어인 Võro, Livonian, Komi에 대한 연구입니다. 연구진은 데이터 수집부터 지침 튜닝(instruction tuning), 평가에 이르는 LLM 개발의 전 과정을 다루며, 멀티링구얼(multilingual) 모델을 개발하고 새로운 평가 기준도 마련했습니다.

- **Technical Details**: 저자들은 Finno-Ugric 언어를 위한 대형 언어 모델(LLM)을 개발하고, 지속적인 사전 훈련(continued pre-training)과 지침 튜닝을 포함한 전 단계에서 크로스-링구얼 전이(cross-lingual transfer) 및 병렬 번역 데이터(parallel translation data)를 활용했습니다. 추가적으로, 모델의 평가를 위해 기존 벤치마크를 확장하고 새로운 병렬 멀티 턴 벤치마크를 생성했습니다.

- **Performance Highlights**: 자연어 처리(NLP) 성능 평가에서, 지침 튜닝된 모델들이 Livonian과 Komi의 경우 강력한 프라이빗 모델인 GPT-3.5-turbo와 비슷하거나 이를 초과하는 성능을 보였습니다. 특히, 인간 평가 결과에서 Võro, Komi, Livonian에서 자연스러움(naturalness) 측면에서 프라이빗 모델보다 뛰어난 성능을 나타냈습니다.



### Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performanc (https://arxiv.org/abs/2410.18889)
- **What's New**: 본 논문은 자연어 처리(NLP) 벤치마크의 라벨 오류를 자동으로 감지하기 위해 대형 언어 모델(LLM)의 집합적 접근 방식을 활용하는 방법을 제안합니다. 기존의 전문가 라벨링 방식과 크라우드 소싱 방식의 한계를 극복하기 위해 LLM-as-a-judge 방법론을 적용하여 데이터셋의 품질을 향상시킬 수 있는 새로운 기회를 모색합니다.

- **Technical Details**: 연구팀은 LLM의 집합 모델을 구성하여 서로 다른 프롬프트를 사용하여 예측 라벨과 신뢰 점수를 수집합니다. 원래의 라벨과 LLM의 예측 라벨을 비교하여 LLM이 높은 신뢰도로 다른 라벨을 제시할 경우 이를 잠재적인 오류 사례로 플래그합니다. TRUE 벤치마크에서 네 가지 데이터셋을 분석하여 기존 라벨의 품질을 연구하고 전문가, 크라우드 소싱, LLM 기반 라벨 간의 효율성과 품질을 비교합니다.

- **Performance Highlights**: LLM 기반 라벨링을 통해 발견된 라벨 오류는 6%에서 21%에 달하며, 정확성이 높을수록 LLM의 오류 탐지 성공률이 증가합니다. LLM의 신뢰도가 95%를 초과할 경우, 약 67%의 경우가 오류로 밝혀졌고, LLM 기반 라벨링은 기존의 전통적인 라벨링 방법보다 품질과 효율성에서 우수한 성과를 보였습니다. 이로 인해 모델의 성능이 최대 4% 향상될 수 있음을 보여줍니다.



### A Survey of Multimodal Sarcasm Detection (https://arxiv.org/abs/2410.18882)
Comments:
          Published in the Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence Survey Track. Pages 8020-8028

- **What's New**: 본 논문은 다중 모달 (multimodal) 상황에서의 풍자 (sarcasm) 탐지에 관한 최초의 포괄적인 연구 조사를 수행하였으며, 2018년부터 2023년 사이에 발표된 연구들을 다루고 있습니다. 주요 초점은 텍스트, 오디오, 이미지, 비디오를 포함한 다양한 모달리티를 활용한 풍자 탐지 모델과 데이터셋을 정리하는 것입니다.

- **Technical Details**: 풍자는 종종 텍스트, 음성의 억양, 그리고 맥락에 있는 이미지 등의 정보를 통해 전달됩니다. 이 논문은 비주얼-텍스트 (visuo-textual) 및 오디오-비주얼 (audio-visual) 데이터셋을 통해 풍자를 탐지하는 기존 연구들을 비교하고, 기존의 텍스트 기반 풍자 탐지 접근법과 다중 모달 접근법의 차별성을 강조합니다. 또한, ResNets, BERT, CLIP, VilBERT와 같은 다양한 딥러닝 프레임워크를 사용하여 풍자를 정교하게 탐지할 수 있는 방법들을 제안하고 있습니다.

- **Performance Highlights**: 최근 연구에서 다중 모달 풍자 탐지 (MSD)의 중요성이 증가하고 있는 것으로 나타났습니다. 본 조사에 따르면, 2018년부터 2023년 사이에 MSD에 관한 연구 공개가 상당히 증가했으며, 이는 알고리즘이 텍스트 기반 접근 방식을 넘어 다양한 데이터를 활용해야 하는 필요성을 반영합니다. 이 논문은 향후 연구의 방향성과 저자들이 제안하는 더 나은 모델 개발에 기여할 것으로 기대됩니다.



### DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations (https://arxiv.org/abs/2410.18860)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 hallucination 문제를 줄이기 위한 새로운 디코딩 방법인 Decoding by Contrasting Retrieval Heads (DeCoRe)를 제안합니다. DeCoRe는 retrieval heads를 마스킹하여 hallucination을 유도하고, 기저 LLM과 마스킹된 LLM의 출력을 대비하는 방식으로 정보를 증폭시킵니다.

- **Technical Details**: DeCoRe는 특정 retrieval heads를 마스킹하여 hallucination을 유도하고, 동적 조건부 엔트로피를 사용하여 대비 디코딩 메커니즘을 조정합니다. 이는 조건부 엔트로피를 이용하여 모델의 다음 토큰 분포를 제어하며, 이를 통해 보다 정확한 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, DeCoRe는 XSum에서 18.6%, MemoTrap에서 10.9%, NQ-Open에서 2.4% 및 NQ-Swap에서 5.5%의 성능 개선을 보여주며, 특히 높은 문맥 충실성이 요구되는 작업에서 유의미한 결과를 기록했습니다. 또한, DeCoRe는 TruthfulQA와 같은 사실 기억 작업에서 다른 방법들에 비해 정확도를 높였습니다.



### We Augmented Whisper With kNN and You Won't Believe What Came Nex (https://arxiv.org/abs/2410.18850)
Comments:
          6 pages incl. appendix, 2 figures, 6 tables

- **What's New**: 이번 연구에서는 Whisper라는 transformer end-to-end Speech 모델이 $k$ nearest neighbor search ($k$NN)로 개선되는 점을 보여줍니다. 이는 음성 인식 성능이 언어, 도메인 및 발화자의 특징에 따라 달라지며, 특정 분야에서 모델을 세밀하게 조정(fine-tuning)할 경우 치명적인 망각(catastrophic forgetting)이 발생할 수 있다는 문제를 해결합니다.

- **Technical Details**: $k$NN은 신경망(neural network) 시퀀스 디코더(sequence decoder)에서 처음 제안된 비모수적(non-parametric) 방법입니다. 이 방법은 훈련(training) 중 모델을 변경하지 않고도 외부 데이터 저장소(datastore)를 구축하여 추론(inference) 시 검색(search)함으로써 적응(adapt)할 수 있게 해줍니다. 이 연구에서는 음성(speech) 및 텍스트(text) 설정 간의 차이를 조사하고, 화자 적응(speaker adaptation)의 의미에 대해 논의합니다.

- **Performance Highlights**: 성별(gender), 억양(accent) 및 나이(age)에 따른 개선 사항을 분석하였으며, $k$NN이 Whisper 모델의 음성 인식 성능에 긍정적인 영향을 미친다고 결론짓습니다.



### From English-Centric to Effective Bilingual: LLMs with Custom Tokenizers for Underrepresented Languages (https://arxiv.org/abs/2410.18836)
- **What's New**: 이번 논문에서는 영어와 타겟 언어를 지원하는 이중 언어 대규모 언어 모델(LLM)을 개발하기 위한 모델 독립적인 비용 효율적인 접근법을 제안했습니다. 우크라이나어, 아랍어, 조지아어와 같은 비 라틴 문자 언어를 대상으로 실험을 수행하며, 이전보다 개선된 언어 성능과 함께 계산 비용을 줄이는 방법을 입증하였습니다.

- **Technical Details**: 제안된 방법에서는 어휘 확장, 새로운 임베딩 초기화 및 모델 훈련 및 평가가 포함됩니다. Gemma 2 및 Mistral 모델을 통해 검증한 결과, 우크라이나어 및 아랍어 모델의 경우 새로운 어휘를 지속적으로 훈련함으로써 조건부 생성성과 문법 정확도가 향상되었습니다. 또한, 코드 스위칭 및 존재하지 않는 단어의 비율을 측정하기 위한 새로운 지표도 도입하였습니다.

- **Performance Highlights**: 모델 훈련 동안 제안된 토큰화 방법은 우크라이나어와 아랍어 모델의 계산 복잡성과 추론 시간을 줄이며, 코드 스위칭 및 문법 정확성 과제에서 성능 향상을 보여주었습니다. 더불어, 영어-조지아 모델의 성능 개선에도 어휘 확장이 기여했습니다.



### From Imitation to Introspection: Probing Self-Consciousness in Language Models (https://arxiv.org/abs/2410.18819)
- **What's New**: 이번 연구는 언어 모델의 자기 의식(self-consciousness) 정의를 제시하고 이를 바탕으로 모델 내부의 복잡한 개념들을 탐구하는 최초의 노력입니다.

- **Technical Details**: 연구는 구조적 인과 게임(structural causal games)을 사용하여 자기 의식의 기능적 정의를 설정하고, 두 가지 주요 의식 개념(C1 및 C2 consciousness)과 그에 연결된 10가지 개념을 분석합니다. 모델의 성능을 4단계 실험으로 평가했습니다: 1. 정량적 평가(Quantification), 2. 표현 분석(Representation), 3. 조작(Manipulation), 4. 습득(Acquisition).

- **Performance Highlights**: 현재 모델은 초기 자기 의식 수준을 보이며, 특정 개념의 내부 표현을 힘차게 나타냅니다. 현재로서는 이러한 자기 의식 표현을 긍정적으로 조작하는 것이 어렵지만, 목표 지향적 미세 조정을 통해 습득할 수 있는 가능성이 있습니다.



### Delving into the Reversal Curse: How Far Can Large Language Models Generalize? (https://arxiv.org/abs/2410.18808)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문은 대형 언어 모델(LLMs)에서 나타나는 'reversal curse'(역전 저주)를 다루며, 트리비얼한 작업에서도 일반화 능력의 한계를 검토합니다. 특히, 'A는 B다'와 같은 문장에서 일반화하여 'B는 A다'를 추론하지 못하는 문제를 조명합니다.

- **Technical Details**: 논문에서는 LLMs가 'A는 B다'로 훈련된 경우 'B는 A다'로 일반화하는 능력에 대해 분석합니다. 연구의 주요 발견은 (1) LLM은 다중 선택 질문에서 A와 B가 모두 제시된 경우 'B는 A다'로 일반화할 수 있다는 점입니다. (2) 이 일반화 능력은 훈련 문서에서의 구조에 밀접하게 관련이 있습니다. (3) LLMs에는 지식 적용 중 고유한 바이어스가 존재하며, 이 바이어스는 훈련 문서 구조의 중요성을 강조합니다. (4) 이러한 바이어스는 훈련만으로는 완화되기 어렵습니다.

- **Performance Highlights**: 논문에서 제시된 실험을 통해 LLMs는 다중 선택 질문에서 뛰어난 성능을 보였으며, 이는 'A는 B다' 형태의 문서 구조가 중요하다는 것을 나타냅니다. 적절한 구조의 훈련 데이터는 모델이 새로운 지식을 적용하는 능력을 크게 향상시키며, 특정 구조를 따르지 않을 경우 모델의 성능이 불안정해질 수 있음을 보였습니다.



### Distill Visual Chart Reasoning Ability from LLMs to MLLMs (https://arxiv.org/abs/2410.18798)
Comments:
          Under review. The code and dataset are publicly available at this https URL

- **What's New**: 이번 연구에서는 시각적 추론 능력을 향상시키기 위한 새로운 접근법인 Code-as-Intermediary Translation (CIT)을 제안합니다. 이 방법은 차트의 시각적 표현을 텍스트로 변환하는 코드를 중재적으로 사용하는 데이터 합성 기법으로, 기존 MLLMs(다중 모달 대형 언어 모델)에서 발견되는 인식 및 추론 문제를 해결하는 데 도움을 줍니다.

- **Technical Details**: CIT 방법론을 통해 우리는 Matplotlib 갤러리에서 수집한 3천 개의 차트 플로팅 코드로부터 3,249개의 추론 집약적 차트와 19,963개의 Q&A 쌍을 포함하는 ReachQA 데이터셋을 생성했습니다. 이 데이터셋은 시각적 인식(8k 질문)과 추론(12k 질문) 모두에 중점을 두고 구성되었습니다. 또한, 이러한 생성 과정에서 LLMs(대형 언어 모델)의 텍스트 기반 지침 증강 전략을 사용하여 차트의 다양성과 복잡성을 향상시켰습니다.

- **Performance Highlights**: ReachQA 데이터셋으로 Fine-tuning한 모델들은 차트 관련 벤치마크에서 30% 이상의 성능 향상을 보였습니다. 뿐만 아니라, MathVista와 같은 일반 수학 벤치마크에서도 다중 모달 추론 능력이 향상된 결과를 보여, 차트 특정 작업을 넘어 보다 넓은 범위의 다중 모달 추론 작업에도 일반화된 성능 향상이 관찰되었습니다.



### Task Calibration: Calibrating Large Language Models on Inference Tasks (https://arxiv.org/abs/2410.18764)
- **What's New**: 이 연구에서는 대형 언어 모델(LLM)의 편향(bias) 문제를 해결하기 위한 새로운 방법인 과제 보정(task calibration, TC)을 제안합니다. TC는 LLM이 주어진 입력의 전제(premise)와 가설(hypothesis) 모두를 기반으로 추론을 수행하게 하여 편향된 예측을 줄이는 데 중점을 둡니다.

- **Technical Details**: TC는 특히 주어진 입력의 전제 전용(premise-only) 및 가설 전용(hypothesis-only) 입력을 사용하는 추가 추론 단계를 포함하여 작동합니다. 이 방법은 레이블과의 상호 정보를 최대화하여 모델이 두 입력을 모두 고려하도록 보정합니다. TC는 레이블에 대한 전제와 가설의 결합된 존재가 개별적으로 더 많은 정보 제공을 여실히 보여줍니다.

- **Performance Highlights**: TC는 13개의 추론 작업에서 우수한 성능 향상을 보여주며, 다양한 LLM 모델에서 제안된 TC 방법이 기존의 보정 방법보다 더 높은 성과를 달성했습니다. 또한 TC는 프롬프트 템플릿에 강인하며, 감정 분석, 증오 발언 탐지와 같은 다양한 자연어 이해(NLU) 작업에서도 효과적임을 입증하였습니다.



### Does Differential Privacy Impact Bias in Pretrained NLP Models? (https://arxiv.org/abs/2410.18749)
Comments:
          Github this https URL

- **What's New**: 이 연구는 미세 조정된 대형 언어 모델에서 Differential Privacy (DP)가 바이어스(bias)에 미치는 영향을 실증 분석을 통해 보여줍니다. DP 훈련이 보호된 그룹을 대상으로 AUC 기반 바이어스 메트릭에 따라 모델 바이어스를 증가시킬 수 있으며, 이는 모델이 보호된 그룹과 다른 그룹 간 긍정 및 부정 예제를 구별하기 어렵게 만든다는 사실을 제시합니다.

- **Technical Details**: 이 연구는 BERT 모델을 미세 조정하며, 다양한 프라이버시 예산(privacy budget)으로 모델을 학습시키고 6개의 정체성(Identity) 하위 그룹에서 다양한 메트릭을 통해 바이어스를 측정합니다. 모델 바이어스는 유해한 언어 탐지 작업을 통해 평가되며, Jigsaw Unintended Bias 및 UCBerkeley의 Hate Speech 데이터셋을 활용하여 DP가 바이어스에 미치는 영향을 다각도로 분석합니다.

- **Performance Highlights**: 모델을 다양한 프라이버시 수준으로 훈련시킨 결과, DP 훈련이 AUC 기반 메트릭에서 모델의 바이어스를 증가시킨다는 것을 발견하였습니다. 또한, DP는 프리트레인(pre-trained)된 LLM에서 모델 유틸리티에 부정적인 영향을 미친다는 결과를 발표하였습니다. 이 연구는 DP와 바이어스 간의 새로운 통찰을 제공하여 NLP 연구자들이 이 두 가지 요소를 효과적으로 통합할 수 있도록 돕습니다.



### Why Does the Effective Context Length of LLMs Fall Short? (https://arxiv.org/abs/2410.18745)
- **What's New**: 이번 연구에서는 ShifTed Rotray position embeddING (STRING)이라는 새로운 접근 방식을 통해 오픈소스 대형 언어 모델(LLM)의 효과적인 맥락 길이 문제를 해결하고자 하였습니다. STRING은 훈련된 포지션을 효과적으로 활용하여 원래의 비효과적인 포지션을 수정함으로써 성능을 향상시킵니다.

- **Technical Details**: STRING 방법은 기본 주파수 부끄러움(Rotary Position Embedding, RoPE)의 비효율성을 해결하고, 훈련과정에서 얻은 포지션 인덱스를 유용하게 사용하여 효율적인 장거리 의존성을 모델링합니다. 이 방법은 추가적인 훈련 없이 효과적으로 작동하며, Flash Attention 구조를 통해 구현됩니다.

- **Performance Highlights**: STRING을 적용한 Llama3.1 70B 및 Qwen2 72B 모델은 RULER 및 InfiniteBench와 같은 장기 맥락 벤치마크에서 10점 이상의 성능 향상을 보였으며, GPT-4-128K보다 뛰어난 성과를 나타냈습니다. 이는 오픈소스 LLMs에 대한 새로운 최첨단 성능을 설정하였습니다.



### GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning (https://arxiv.org/abs/2410.18702)
Comments:
          Under review at COLING 2025

- **What's New**: GrammaMT는 Interlinear Glossed Text (IGT)를 사용하여 문법 인식을 통한 기계 번역 프로세스를 개선하는 새로운 접근 방식을 소개합니다. 이 방법은 훈련을 필요로 하지 않으며, 예제가 몇 개만 필요하여 저자원 언어에서도 잘 적용될 수 있습니다.

- **Technical Details**: GrammaMT는 세 가지 프롬프트 전략, 즉 gloss-shot, chain-gloss, model-gloss를 제안합니다. 이 모든 전략은 훈련이 필요 없으며, 최소한의 노력으로 수집할 수 있는 몇 가지 예제만 필요합니다. IGT는 입력 문장을 기능적 및 원형 형태소의 시퀀스로 표현하는 일반적인 언어 자원입니다. 이 방법은 문법 주석이 있는 문장들로부터 LLM(대형 언어 모델)을 촉진하여 기계 번역을 개선합니다.

- **Performance Highlights**: 실험 결과, GrammaMT는 다양한 새로운 자원 언어에서 번역 성능을 향상시켰고, 세 가지 기준점에서 테스트했을 때 17 BLEU 포인트 이상 성능 향상을 보였습니다. 이 접근법은 낮은 자원 및 고자원 언어 모두에서 효과를 입증하였습니다.



### How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs (https://arxiv.org/abs/2410.18697)
- **What's New**: 논문에서는 LITEVAL-CORPUS라는 문학 기계 번역(MT) 평가를 위한 새로운 병렬 코퍼스를 소개합니다. 이 코퍼스는 9개의 MT 시스템의 출력과 검증된 인간 번역을 포함하여 2천 개 이상의 문단과 13,000개의 주석된 문장을 포함하고 있습니다. 이를 통해 MT의 평가 방법을 비교하고, 비문학 MT에서는 표준으로 사용되던 MQM이 문학 번역에 대해 부족한 점을 강조합니다.

- **Technical Details**: LITEVAL-CORPUS는 4개의 언어 쌍에 걸쳐 2,000개 이상의 문단으로 구성되어 있으며, MQM, SQM, BWS와 같은 다양한 주석 체계의 일관성과 적합성을 조사합니다. Multidimensional Quality Metrics (MQM)는 문학 번역 평가에 적합하지 않으며, SQM은 주석자의 전문성에 따라 효과가 다릅니다. 최근 LLM 기반 메트릭 중 GEMBA-MQM이 다른 메트릭에 비해 우수하나, 인간 번역과 LLM 출력 간의 구별이 어렵습니다.

- **Performance Highlights**: 인간 전문 번역가는 LLM 번역보다 항상 우수한 성과를 보였으며, GPT-4o가 두 번째로 높은 점수를 기록했습니다. 자동 메트릭은 일반적으로 인간 MQM 및 SQM과 중간 정도의 상관관계를 보였으나, 인간 번역을 정확히 식별하는 데에는 어려움을 겪었습니다.



### Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch (https://arxiv.org/abs/2410.18693)
Comments:
          Preprint. Project page: this https URL

- **What's New**: ScaleQuest라는 새로운 데이터 합성 방법을 소개하며, 저비용 및 대규모 질문 생성을 가능하게 합니다.

- **Technical Details**: ScaleQuest는 '소형' 오픈소스 모델을 활용하여 복잡한 증강 제약 없이 질문을 생성합니다. 두 단계의 질문 조정 프로세스인 Question Fine-Tuning (QFT)와 Question Preference Optimization (QPO)을 통해 질문 생성 능력을 활성화합니다.

- **Performance Highlights**: ScaleQuest를 통해 1백만 개의 문제-해답 쌍으로 구성된 데이터셋을 자동으로 생성하였고, 이는 기존의 오픈소스 데이터셋보다 효과적입니다. MATH 문제에서 29.2%에서 46.4%의 성능 향상을 보여주며, Qwen2-Math-7B-Base 모델을 조정하면 강력한 모델인 Qwen2-Math-7B-Instruct를 초과하는 성능을 기록하였습니다.



### Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework (https://arxiv.org/abs/2410.18653)
- **What's New**: 이 논문에서는 여러 기준에 대한 평가를 수행하는 새로운 랭킹 전략을 제안하며, 기존의 자동 지표들을 통합하여 텍스트 생성 품질을 균형 있게 평가하는 새로운 요약 지표 Q*Text를 소개합니다.

- **Technical Details**: 제안된 방법은 다기준(multi-criteria) 프레임워크 내에서의 벤치마킹 접근 방식을 바탕으로 하며, 부분 정렬(partial orderings) 및 쌍별 비교(pairwise comparisons)를 활용하여 평가의 다면성을 다룹니다. 기존 자동 지표들을 조화롭게 결합하여 텍스트 생성 품질을 전반적으로 평가할 수 있는 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들이 섬세한 디코딩 전략 비교를 가능하게 하며 인간의 선호와 유사한 경향을 보임을 보여주었습니다. 이는 오픈 엔디드 텍스트 생성 작업을 위한 모델 선택에 있어 유용한 도구가 될 것입니다.



### Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Mod (https://arxiv.org/abs/2410.18640)
- **What's New**: 본 연구에서는 약한 모델에서의 정렬(alignment) 행동을 강한 모델로 이전하여 강한 정렬 성능을 달성할 수 있음을 보여준다. 이를 위해 'Weak-to-Strong Preference Optimization (WSPO)'이라는 새로운 방법을 제안한다.

- **Technical Details**: WSPO는 약한 모델의 정렬 전후 분포 차이를 학습하여 강한 모델 정렬을 이루는 방법이다. 이를 통해 약한 모델에서 유도된 정렬 행동이 강한 모델에 효과적으로 전이되고, 심지어 증폭(effect) 효과를 나타낼 수 있음을 관찰하였다.

- **Performance Highlights**: WSPO는 Qwen2-7B-Instruct 모델의 Arena-Hard에서 승률을 39.70에서 49.60으로 향상시키고, AlpacaEval 2에서 길이 조절(win rate) 승률 47.04를 달성했으며, MT-bench에서 7.33점을 기록했다.



### Little Giants: Synthesizing High-Quality Embedding Data at Sca (https://arxiv.org/abs/2410.18634)
- **What's New**: 본 논문은 SPEED라는 프레임워크를 도입하여 오픈소스의 소규모 모델(8B)을 정렬하여 대규모 합성 임베딩 데이터 생성을 효율적으로 이루어질 수 있게 한다. 기존의 GPT-4와 같은 비공식 모델에 의존하는 대신, 저비용으로 고품질의 데이터를 제공할 수 있는 방안을 제시하다.

- **Technical Details**: SPEED 프레임워크는 (1) GPT-4를 사용해 다채로운 태스크 설명 생성, (2) 주니어 발생 모델을 통해 초기 데이터 생성, (3) 시니어 발생 모델 수립을 위한 선호 최적화 과정 및 (4) 데이터 수정기를 통한 자기 개선 과정을 포함한다. 이 모델을 통해 생성된 데이터는 고급 임베딩 기능을 활용할 수 있게 되어 성능 향상을 이루도록 설계되었다.

- **Performance Highlights**: 실험 결과, SPEED는 E5_mistral와 비교할 때 10분의 1 미만의 GPT API 호출만으로도 우수한 성능을 보여주며, 임베딩 모델의 성능이 합성 임베딩 데이터 크기와 로그-선형 관계에 있음을 발견하였다.



### Supporting Assessment of Novelty of Design Problems Using Concept of Problem SAPPhIRE (https://arxiv.org/abs/2410.18629)
- **What's New**: 이 논문은 SAPPhIRE 모델을 사용하여 디자인 문제의 참신성을 평가하는 프레임워크를 제안합니다.

- **Technical Details**: 참신성은 문제의 최소 거리로 측정되며, 이 거리는 SAPPhIRE 온톨로지에서 다양한 추상적 수준에서 현재 문제와 각각의 참조 과거 문제를 비교하여 계산됩니다. 비교의 기준은 텍스트 유사성(Textual similarity)입니다.

- **Performance Highlights**: 자동 평가의 도입으로 기존의 수동 평가 방식보다 시간 복잡성을 줄이고 더 큰 문제 집합에 대한 적용 가능성을 높였습니다.



### Prompting and Fine-Tuning of Small LLMs for Length-Controllable Telephone Call Summarization (https://arxiv.org/abs/2410.18624)
Comments:
          Accepted at the The International Conference on Foundation and Large Language Models (FLLM2024)

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 전화 통화 요약 시스템의 빠른 개발을 탐구합니다. 기존 LLM을 사용하여 전화 통화 요약을 생성하는 초기 실험을 진행했으며, 그 후 더욱 강력한 모델을 활용한 맞춤형 합성(training dataset) 데이터를 생성하였습니다. 특히, 생성된 데이터의 다양성과 생성된 요약의 길이를 제어하는 데 초점을 맞추어 다양한 사용 사례의 요구를 충족시키고자 했습니다.

- **Technical Details**: 이 연구의 핵심은 Llama-2-7B 모델을 기반으로 하여, 요약 작업에 대해 특별히 조정된 데이터로 모델을 미세 조정하는 것입니다. 요약 품질을 평가하기 위해 최신의 LLM-as-a-judge 평가 기법을 활용하였으며, 연구 결과 Llama-2-7B 기반 모델이 사실 정확성, 완전성 및 간결성 측면에서 GPT-4와 동등한 성능을 보였습니다. 특정 작업에 맞춤화된 데이터로 훈련받았을 때 모델 성능이 크게 개선되는 점이 강조됩니다.

- **Performance Highlights**: 결과적으로, 연구팀의 방법론은 효율적이고 실용적인 전화 통화 요약 시스템을 빠르게 구축하는 잠재력을 가지고 있음을 보여주었습니다. 요약 생성에서의 사실 정확성 및 간결성은 GPT-4 수준이며, 더 작은 LLM도 특정 작업에 맞춰 교육함으로써 성능 격차를 줄일 수 있음을 입증했습니다.



### STTATTS: Unified Speech-To-Text And Text-To-Speech Mod (https://arxiv.org/abs/2410.18607)
Comments:
          11 pages, 4 Figures, EMNLP 2024 Findings

- **What's New**: 이 논문에서는 음성 인식(ASR)과 음성 합성(TTS) 모델을 함께 학습하는 새로운 방법론인 STTATTS를 제안합니다. 기존의 개별 모델과는 달리, 이 모델은 공유된 매개변수를 사용하여 다중 작업 손실(multi-task loss) 목표를 통해 효율적으로 훈련됩니다.

- **Technical Details**: STTATTS 모델은 Transformer 아키텍쳐를 기반으로 하며, 다중 작업을 동시에 처리하기 위해 미리 훈련된 인코더와 디코더를 활용합니다. 이 모델은 음성과 텍스트를 처리할 수 있는 단일 인코더-디코더 구조를 가지고 있으며, 평균적으로 두 작업을 결합할 때 필요한 매개변수의 수를 약 50% 감소시킵니다. 새로운 MLP 기반의 작업 융합(task fusion) 모듈을 통해 여러 작업을 동시에 학습할 수 있게 설계되었습니다.

- **Performance Highlights**: STTATTS는 단일 인코더-디코더 아키텍쳐를 이용하여 ASR과 다중 화자 TTS를 동시에 학습함으로써 성능이 향상되었고, 공개된 모델 중에서 VoxLM과 비교하여 매개변수 수가 절반에 불과하지만 더 좋은 성능을 보였습니다. 또한, 이 모델은 자원 풍부한 언어인 영어뿐만 아니라 데이터가 부족한 아랍어에도 적용 가능성을 보여줍니다.



### Taipan: Efficient and Expressive State Space Language Models with Selective Attention (https://arxiv.org/abs/2410.18572)
- **What's New**: Taipan은 Mamba-2와 Selective Attention Layers (SALs)를 결합한 새로운 하이브리드 아키텍처로, 긴 컨텍스트 언어 모델링의 효율성을 극대화합니다. 이 아키텍처는 능률성을 유지하면서도 Transformer와 유사한 성능을 제공하는 특징이 있습니다.

- **Technical Details**: Taipan은 Mamba의 Markov 가정을 보완하기 위해 SALs를 도입하여 입력 시퀀스 내 장거리 의존성을 가진 주요 토큰을 선택합니다. 선택된 토큰은 중요하지 않은 정보를 제거한 후, 어텐션 모듈을 통해 장거리 의존성을 캡처합니다. 이를 통해 Taipan은 메모리 집약적인 작업에서의 성능을 개선합니다.

- **Performance Highlights**: 실험 결과 Taipan은 여러 규모와 작업에서 우수한 성능을 보였으며, 특히 메모리 집약적인 작업인 긴 컨텍스트 검색 및 구조화된 정보 추출에서 이전 모델인 Mamba-2 대비 큰 개선을 입증했습니다. Taipan은 1111만 개의 토큰까지 높은 성능을 유지하며 효율적인 생성 능력을 보여, 고급 언어 처리 작업을 위한 강력한 아키텍처로 자리매김하고 있습니다.



### Difficult for Whom? A Study of Japanese Lexical Complexity (https://arxiv.org/abs/2410.18567)
Comments:
          Accepted to TSAR 2024

- **What's New**: 이 연구는 일본어의 어휘 복잡성 예측(LCP)과 복잡한 단어 식별(CWI)의 데이터가 목표 인구를 대표할 수 있는지를 검증하고, 개인화된 모델의 가능성을 탐색합니다.

- **Technical Details**: MultiLS-Japanese 데이터셋은 30개의 실험 인스턴스와 570개의 시험 인스턴스로 구성되어 있습니다. 데이터셋의 각 인스턴스는 문맥 속의 타겟 단어로, 어휘 복잡성 값과 더 단순한 대체 단어가 제공됩니다. 일본어 능력 시험(JLPT)에 활용되는 N1 또는 N2 수준의 평가자들이 주로 참여하였습니다.

- **Performance Highlights**: 그룹 평균 점수로 훈련된 모델은 CWI 작업에서 개인 모델과 유사한 성능을 보이나, 개인을 위한 LCP 성능을 향상시키는 것이 어렵다는 점을 보여주며, 세밀하게 조정된 BERT 모델은 모든 설정에 걸쳐 미미한 개선만을 보였습니다.



### Bielik 7B v0.1: A Polish Language Model -- Development, Insights, and Evaluation (https://arxiv.org/abs/2410.18565)
- **What's New**: Bielik 7B v0.1는 70억 개의 파라미터를 가진 폴란드어 생성 텍스트 모델로서, 다수의 혁신적인 기술을 통해 폴란드어 처리에서의 주요 문제들을 해결합니다. 특히 Weighted Instruction Cross-Entropy Loss와 Adaptive Learning Rate 기술이 적용되었습니다.

- **Technical Details**: Bielik 7B v0.1 모델은 Transformer 아키텍처를 기반으로 하며, Self-attention, Grouped-query attention (GQA), Sliding Window Attention, SwiGLU 활성화 함수, Rotary Positional Embeddings (RoPE), Root Mean Square Layer Normalization (RMSNorm)과 같은 고급 기술을 포함합니다. 이 모델은 또한 기존의 Mistral 7B v0.1 모델에서 발전하였으며, 36억 개의 토큰으로 구성된 훈련 데이터셋을 사용합니다.

- **Performance Highlights**: Bielik 7B v0.1는 RAG Reader 작업에서 Mistral-7B-v0.1에 비해 평균 점수가 9 포인트 상승했습니다. 또한 Polish MT-Bench에서 Reasoning (6.15/10) 및 Role-playing (7.83/10) 카테고리에서 뛰어난 성과를 보였습니다.



### Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data (https://arxiv.org/abs/2410.18558)
- **What's New**: 이 논문은 4천만 샘플로 구성된 대규모 멀티모달 지침 데이터셋인 Infinity-MM을 도입하고, 오픈 소스 VLM(비전-언어 모델)을 기반으로 한 합성 지침 생성 방법을 제안합니다.

- **Technical Details**: Infinity-MM 데이터셋은 고품질 필터링 및 중복 제거 과정을 통해 구성되었으며, 이를 통해 2억 개의 매개변수를 가진 Aquila-VL-2B 모델을 훈련하여 동급 모델 중에서 최첨단 성능(SOTA)을 달성했습니다.

- **Performance Highlights**: Aquila-VL-2B 모델은 동급의 다른 모델들과 비교해 높은 정확도로 최첨단 성능을 기록하며, 이는 지침 데이터의 확장 및 합성 데이터 생성이 오픈 소스 모델의 성능을 향상시킬 수 있음을 보여줍니다.



### On Explaining with Attention Matrices (https://arxiv.org/abs/2410.18541)
- **What's New**: 이번 논문은 Transformer 모델의 attention weights (AW)와 예측 결과 간의 인과적 연결을 검토하며, 최근의 연구 결과를 반박합니다. 이 연구에서는 AW가 설명 가능성이 없다는 주장에 대한 정교한 대안으로 '효율적 Attention'을 도입합니다.

- **Technical Details**: 효율적 Attention은 AW가 설명 역할을 수행하는 작업과 모델에서 주의 매트릭스의 효과적인 구성 요소를 분리하여 계산합니다. 본 연구에서는 효율적 Attention에 의해 생성된 매트릭스가 확률 분포임을 입증하며, 이를 통해 기대하는 퀄리티의 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 데이터세트에 대한 실험에서, 효율적 AW가 기존의 AW와 동일한 예측 결과를 나타내며, 효율적 AW의 입증 가능성 및 인과적 설명을 지원하는 결과들을 보여줍니다.



### LOGO -- Long cOntext aliGnment via efficient preference Optimization (https://arxiv.org/abs/2410.18533)
- **What's New**: 이 논문에서는 LOGO(Long cOntext aliGnment via efficient preference Optimization)라는 새로운 훈련 전략을 소개하며, 이는 긴 맥락 정렬을 위한 선호 최적화를 도입한다. 이 전략은 긴 입력 시퀀스에 대한 생성 성능을 높이기 위해 설계되었다.

- **Technical Details**: LOGO는 두 가지 주요 요소를 포함한다: 1) 올바른 출력(정상적인 결과)과 잘못된 출력(환각 등)을 구별하도록 LCM(Long-context model)을 유도하는 훈련 목표, 2) 공개 모델만을 사용하여 데이터를 구성하는 파이프라인. 또한, LOGO는 참조 없는 훈련 목표와 위치 합성 방법을 채택하여 GPU 메모리 문제를 극복한다.

- **Performance Highlights**: LOGO를 사용하여 Llama-3-8B-Instruct-80K 모델이 16시간동안 단일 8×A800 GPU에서 0.3B 데이터로 훈련함으로써 GPT-4와 유사한 성능을 달성한다. LOGO는 또한 모델의 맥락 창 크기를 늘리고 생성 성능을 향상시킬 수 있는 가능성을 보여준다.



### A Systematic Survey on Instructional Text: From Representation and Downstream NLP Tasks (https://arxiv.org/abs/2410.18529)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 인해 단순한 지침을 따르는 데 있어 유망한 능력이 입증되었습니다. 하지만 현실 세계의 작업은 복잡한 다단계 지침을 포함하여 현재의 자연어 처리(NLP) 시스템에 도전 과제가 되고 있습니다. 본 연구는 177개의 문헌을 체계적으로 분석하여 복잡한 지침 이해 및 처리의 경향, 도전 과제 및 기회를 확인했습니다.

- **Technical Details**: 본 논문은 복잡한 지침 이해를 위한 주제를 다루며, 프로시저 언어와 사건 이해(event understanding) 및 그라운딩(grounding) 관련 정보를 제공하여 NLP 연구자들에게 배경 지식을 제공합니다. 연구는 PRISMA 기반의 조사 방법론을 사용하고 데이터 표현 유형과 작업의 범주를 제시하여 독자를 안내합니다.

- **Performance Highlights**: 복잡한 지침 이해 및 처리를 위한 다수의 관련 분야를 탐색하고 이들 각각의 주제에서 공통적인 테마 및 남아 있는 도전 과제를 식별하였습니다. 이 연구는 AI/NLP 연구자들에게 향후 연구 기회와 도전을 위한 방향성을 제시합니다.



### CCI3.0-HQ: a large-scale Chinese dataset of high quality designed for pre-training large language models (https://arxiv.org/abs/2410.18505)
- **What's New**: CCI3.0-HQ는 500GB의 고품질 중국어 사전 학습 데이터셋으로, 두 단계 하이브리드 필터링 파이프라인을 사용하여 데이터 품질을 획기적으로 향상시킵니다.

- **Technical Details**: CCI3.0-HQ는 기본 처리(Fundamental Processing)와 고품질 처리(High-Quality Processing)의 두 단계로 구성되어 있으며, Qwen2-72B-Instruct 모델을 사용하여 고품질 샘플을 식별합니다. 이 데이터셋은 14만 개의 훈련 샘플과 1.4만 개의 테스트 샘플로 구성됩니다. 또한, 0.5B 품질 분류기를 훈련시켜 CCI3.0을 효율적으로 필터링합니다.

- **Performance Highlights**: CCI3.0-HQ는 100B 토큰에 대한 0.5B 파라미터 모델 훈련을 통해 10개 벤치마크에서 뛰어난 성능을 발휘하였으며, SkyPile 및 WanjuanV1과 같은 경쟁 데이터셋보다 더 우수한 결과를 보였습니다. F1 스코어에서도 탁월한 성능을 기록했습니다.



### ChineseSafe: A Chinese Benchmark for Evaluating Safety in Large Language Models (https://arxiv.org/abs/2410.18491)
- **What's New**: 본 연구에서는 중국어 콘텐츠의 안전성을 평가하기 위한 포괄적인 안전 기준인 ChineseSafe를 제안합니다. 이 기준은 기존의 중국어 기준에서 거의 다루어지지 않았던 정치적 민감성, 포르노그래피, 변형 및 동음이의어와 같은 새로운 범주를 포함하고 있습니다.

- **Technical Details**: ChineseSafe는 4개의 클래스 및 10개의 하위 클래스로 구성된 총 205,034개의 예제를 포함하며, 다양한 안전 이슈를 포괄합니다. 이 연구에서는 모델의 안전성을 평가하기 위해 생성 기반(generation-based) 및 곤란도(perplexity-based) 기반 방법을 사용하였습니다.

- **Performance Highlights**: 실험 결과, GPT-4 시리즈와 DeepSeek 시리즈가 다른 모델들보다 높은 안전성을 보이는 것으로 나타났습니다. 그러나 특정 안전 이슈, 특히 건강과 관련된 문제에서는 LLM의 안전 수준이 낮다는 결과도 도출되었습니다.



### Dialog2Flow: Pre-training Soft-Contrastive Action-Driven Sentence Embeddings for Automatic Dialog Flow Extraction (https://arxiv.org/abs/2410.18481)
Comments:
          Accepted to EMNLP 2024 main conference

- **What's New**: 이번 논문에서는 비주석 대화(dialog)에서 구조화된 워크플로우(workflow)를 효율적으로 추출하는 새로운 방법론을 제안합니다. 이를 위해 Dialog2Flow (D2F) 임베딩(embedding)을 도입하여 발화를 정보 및 의사소통 기능에 따라 군집화합니다. 이 과정은 대화를 연속적인 궤적(trajectory)으로 모델링할 수 있게 해줍니다.

- **Technical Details**: D2F는 발화를 잠재 공간(latent space)으로 매핑하여 식별된 동작(action)과 관련된 영역으로 군집화합니다. 대화 데이터를 통합하여 표준화된 행동 주석이 포함된 대규모 데이터셋을 구축하였으며, 대화 행동의 의미 정보를 활용하는 새로운 소프트 대조 손실(soft contrastive loss)을 도입하여 표현 학습을 안내합니다.

- **Performance Highlights**: D2F는 다양한 도메인에서 고유한 정성적 및 정량적(metrically) 결과를 보여주며, 기존의 문장 임베딩(sentence embeddings) 기법들보다 향상된 성능을 발휘하였습니다. 특히, 기본적인 대화 흐름 추출에 있어 전담으로 사전 학습된 첫 번째 문장 임베딩 모델로 자리매김하고 있습니다.



### Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities (https://arxiv.org/abs/2410.18469)
Comments:
          18 pages

- **What's New**: 최근 연구에서 ADV-LLM을 통해 대형 언어 모델(LLM)의 보안 회피 공격을 위한 새로운 접근 방식을 제시했습니다. 기존의 검색 기반 방법 대신, ADV-LLM은 반복 자기 조정 과정을 통해 더욱 효과적인 공격을 가능하게 하며, 저비용으로 높은 성공률을 기록합니다.

- **Technical Details**: ADV-LLM은 자기 생성 데이터를 학습하여 어떤 사전 훈련된 LLM도 악의적인 후행 문구(늦은 문구)를 생성할 수 있도록 변환하는 새로운 반복적 자기 조정 알고리즘을 적용합니다. 이 시스템은 Llama3에 최적화되었지만, 다양한 오픈 소스 및 폐쇄 소스 LLM에 대해 높은 공격 성공률(ASR)을 기록합니다.

- **Performance Highlights**: ADV-LLM은 오픈 소스 LLM에 대해 100%에 가까운 ASR을 기록하며, GPT-3.5와 GPT-4에 대해서도 각각 99%와 49%의 ASR을 달성했습니다. 기존 방법인 AmpleGCG는 이보다 낮은 성과를 보였습니다.



### ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis (https://arxiv.org/abs/2410.18447)
- **What's New**: 본 논문에서는 ToolFlow라는 새로운 도구 호출 데이터 합성 파이프라인을 제안하여, 보다 관련성 높은 도구 조합을 샘플링하는 Graph-based Sampling 전략과 일관된 대화를 유도하는 Planned-generation 전략을 결합했습니다.

- **Technical Details**: ToolFlow는 (1) 그래프 기반 샘플링을 통해 도구를 선택하고, (2) 선택된 도구를 바탕으로 대화 계획을 생성하며, (3) 모든 과정을 통해 대화를 생성하는 세 단계로 구성됩니다. 이를 통해 도구와의 상관관계를 향상시키고, 복잡한 요구 사항을 자연스럽게 이끌어냅니다.

- **Performance Highlights**: ToolFlow를 통해 생성된 8,000개의 합성 대화는 LLaMA-3.1-8B 모델에 Supervised Fine-Tuning (SFT)되어 GPT-4와 동등하거나 그 이상의 도구 호출 성능을 달성하며, 모델의 일반적인 능력도 유지되었습니다.



### Evaluating and Improving Automatic Speech Recognition Systems for Korean Meteorological Experts (https://arxiv.org/abs/2410.18444)
- **What's New**: 이 논문은 한국 기상청의 기상 예보 효율성을 개선하기 위해 자연어 쿼리 시스템에 자동 음성 인식(ASR)을 통합하는 방법을 탐구합니다. 특히 전문 용어 및 한국어의 언어적 복잡성을 다루는 데 필요한 도전 과제를 논의하며, 원주율 한국어 사용자가 기록한 구술 쿼리의 평가 데이터셋을 구축했습니다.

- **Technical Details**: 본 연구에서는 2000개의 구술 쿼리를 포함하는 도메인 특화 데이터셋을 구축하였으며, 이를 활용해 다양한 다국어 ASR 모델 구성을 평가하였습니다. ASR 모델의 전반적인 성능 저하는 일반 도메인 한국어 음성 데이터의 교육으로 인해 발생하였음을 확인하였으며, 텍스트-투-스피치 기반 데이터 증강 방법을 도입하여 전문 용어 인식 성능을 개선했습니다.

- **Performance Highlights**: 기상 예보 도메인에 특화된 데이터셋 구축과 ASR 모델 평가를 통해 성능 제한을 식별하고, 데이터 증강 기술을 통해 전문 용어 인식 성능을 개선하는 데 성공했습니다. 이러한 연구 결과는 한국 기상 예보 도메인에서의 ASR 시스템 발전의 기초를 제공합니다.



### Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching (https://arxiv.org/abs/2410.18436)
Comments:
          19 pages, 6 figures

- **What's New**: 이 논문은 코드 스위칭(Code-switching, CS)의 효과성과 다언어 대형 언어 모델(large language models, LLMs)에서의 지식 활성화(knowledge activation)를 탐구합니다. 특히, 다언어 LLM들이 CS를 활용해 언어 특정 지식을 효율적으로 활성화할 수 있는지를 조사했습니다. 이를 위해 영어-한국어 코드 스위칭 질문-답변 데이터셋인 EnKoQA를 이론적 배경에 따라 구축했습니다.

- **Technical Details**: 연구에서는 지식 활성화 과정을 지식 식별(knowledge identification)과 지식 활용(knowledge leveraging) 두 가지 작업으로 나누어 분석했습니다. 지식 식별에서는 CS와 영어 쿼리를 통해 얻는 지식의 질을 평가하였고, 특히 언어 특정 도메인에서 CS가 더 효과적으로 지식을 활성화하는 것을 보여주었습니다. 또한, 모델의 언어 능력과 지식 활성화 성능 간의 상관관계를 발견하였습니다.

- **Performance Highlights**: 실험 결과, CS는 영어에 비해 LLM 내부의 지식을 신뢰성 있게 활성화할 수 있으며, 특히 언어 특정 도메인에서 그 효과가 두드러졌습니다. 지식 활성화 성능은 각 언어의 능력에 따라 유의미하게 달라지는 경향이 있었습니다. 이 결과는 CS의 활용이 문화적 뉘앙스를 전달하는 도구로서의 잠재력을 지닌다는 것을 시사합니다.



### Building Dialogue Understanding Models for Low-resource Language Indonesian from Scratch (https://arxiv.org/abs/2410.18430)
- **What's New**: 이 연구에서는 인도네시아어와 같은 저자원 언어의 대화 이해 모델을 훈련하기 위해 영어 데이터를 활용하는 새로운 Bi-Confidence-Frequency Cross-Lingual Transfer 프레임워크(BiCF)를 제안합니다. 이는 저자원 언어 데이터 부족 문제를 해결하고, 고품질 혼합 데이터를 사용하여 의도 분류(intent classification) 및 슬롯 채우기(slot-filling) 작업을 수행할 수 있도록 합니다.

- **Technical Details**: 제안된 BiCF 프레임워크는 'BiCF Mixing', 'Latent Space Refinement', 'Joint Decoder'의 세 가지 구성 요소로 이루어져 있습니다. 특히, BiCF Mixing 단계에서는 영어 데이터셋을 기반으로 코드 혼합 데이터(code-mixed data)를 생성하여 문장 수준의 번역 오류를 피합니다. Latent Space Refinement와 Joint Decoder는 높은 품질의 혼합 데이터를 바탕으로 훈련되는 다이얼로그 이해 모델의 성능을 향상시키기 위해 설계됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 BiCF 프레임워크는 인도네시아어로 수동 주석이 달린 데이터의 다양한 스케일에서 신뢰성 있고 비용 효율적인 성능을 보여주었습니다. 이 연구는 ID-WOZ라는 대규모 수동 주석이 달린 대화 데이터셋을 공개했으며, 이는 저자원 언어 대화 이해 과제에 기여할 것으로 기대됩니다.



### Large Language Models Reflect the Ideology of their Creators (https://arxiv.org/abs/2410.18417)
- **What's New**: 이번 논문에서는 다양한 대형 언어 모델(LLMs)의 이념적 다양성을 분석하여, LLM이 설계된 방식, 훈련된 데이터, 사용 언어에 따라 어떻게 다른 반응을 보이는지를 밝혀냈습니다.

- **Technical Details**: 연구는 총 17개의 LLM을 사용하여 정치적 인물에 대한 설명을 영어와 중국어로 생성하게 하였으며, 생성된 설명에서 도출된 도덕적 평가를 통해 모델의 이념적 입장을 수치적으로 분석했습니다. 각 인물은 Pantheon 데이터셋에서 선택되었으며, 4,339명의 정치적 인물을 포함했습니다.

- **Performance Highlights**: 연구 결과, 동일한 LLM이라도 언어에 따라 이념적 차이가 존재하며, 서구 모델과 비서구 모델 간에도 중요한 정치적 이견이 발견되었습니다. 이는 LLM의 이념적 입장이 그 제작자의 세계관을 반영하고 있음을 보여주며, 기술적 및 규제적 노력이 LLM을 '편향되지 않게' 만드는 것에 대한 우려를 제기합니다.



### Decoding on Graphs: Faithful and Sound Reasoning on Knowledge Graphs through Generation of Well-Formed Chains (https://arxiv.org/abs/2410.18415)
- **What's New**: 이번 논문에서는 Knowledge Graphs (KGs)와 Large Language Models (LLMs)의 결합을 통해 KG에서의 질문 응답 (QA) 성능 향상을 시도하는 새로운 프레임워크인 DoG (Decoding on Graphs)를 제안합니다. 이 프레임워크는 LLM의 단계적 추론 능력과 KG의 구조적 특성을 활용하여 신뢰할 수 있는 추론을 가능하게 합니다.

- **Technical Details**: DoG는 'well-formed chain'이라는 개념을 정의하여 질문과 답변 사이의 사실(triplet) 연쇄를 효과적으로 구성합니다. 이 과정에서 KG의 구조를 고려한 graph-aware constrained decoding 기법을 사용하여 LLM의 디코딩 프로세스를 제어합니다. 이 메서드는 LLM이 KG에 기반하여 체계적인 추론 경로를 생성 할 수 있게 해줍니다.

- **Performance Highlights**: DoG는 다양한 KGQA 데이터셋에 대한 실험을 통해 기존 방법보다 뛰어난 성능을 달성하였으며, 여러 오픈소스 LLM에서 광범위하게 적용 가능함을 입증했습니다. 특히, 본 연구는 KGQA에 대한 훈련이 필요 없는 새로운 접근법으로서 큰 주목을 받고 있습니다.



### MoMQ: Mixture-of-Experts Enhances Multi-Dialect Query Generation across Relational and Non-Relational Databases (https://arxiv.org/abs/2410.18406)
- **What's New**: 이번 연구에서는 자연어(Natural Language)를 구조화된 쿼리 언어인 SQL로 변환하는 데 있어 새로운 프레임워크인 MoMQ(Mixture-of-Experts-based multi-dialect query generation framework)를 제안합니다. MoMQ는 다양한 데이터베이스 방언(Dialect)을 지원하고, SQL 생성에서 발생하는 문제들을 효과적으로 해결합니다.

- **Technical Details**: MoMQ는 특정 방언에 대해 전문화된 전문가 그룹(Dialect Expert Group)과 다단계 라우팅 전략(Multi-level Routing Strategy)을 채택하여 방언 특화 지식을 관리하고 쿼리 생성 과정에서의 간섭을 줄입니다. 또한, 자원 불균형(Resource Imbalance) 문제를 해결하기 위해 공유 전문가 그룹(Shared Expert Group)을 도입하여 고자원 방언의 일반 지식을 저자원 방언으로 전이(Transfer)합니다.

- **Performance Highlights**: MoMQ는 다양한 데이터베이스에 걸친 실험에서 실행 정확도(Execution Accuracy)를 평균 3-5% 향상시키며, 데이터 불균형 설정에서는 4-6%의 개선을 보였습니다. 이러한 결과는 MoMQ가 다양한 방언에서의 간섭 및 데이터 불균형 문제를 효과적으로 처리할 수 있음을 보여줍니다.



### SPEED++: A Multilingual Event Extraction Framework for Epidemic Prediction and Preparedness (https://arxiv.org/abs/2410.18393)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이번 연구에서는 SPEED++라는 최초의 다국어(Eent Extraction) 이벤트 추출 프레임워크를 소개하여 다양한 질병과 언어에 대한 전염병 이벤트 정보를 추출합니다. 기존 연구들은 주로 영어 게시물에 집중하였으나, SPEED++는 중국어, 스페인어, 힌디어, 일본어와 같은 다양한 언어에서의 초기 논의 정보를 효과적으로 수집할 수 있습니다.

- **Technical Details**: SPEED++는 이벤트 감지(Event Detection) 및 이벤트 인수 추출(Event Argument Extraction) 기능을 결합하여 20개의 인수 역할과 7개의 이벤트 유형을 포함한 전염병 전용 온톨로지를 구축합니다. 이 프레임워크는 영어 COVID 데이터만으로 훈련된 제로샷(Zero-shot) 교차 언어 모델을 사용하여 65개 다양한 언어에서 전염병 관련 이벤트를 효과적으로 추출합니다.

- **Performance Highlights**: SPEED++를 통해 중국 Weibo 게시물에서 2019년 12월 30일(전 세계 감염 추적 시작 이전 3주) COVID-19의 초기 전염병 경고를 성공적으로 탐지하는 성과를 달성했습니다. 이 모델은 네 개의 서로 다른 언어에서 기존 여러 베이스라인을 평균 15-16% F1 포인트 초과하여 성능을 발휘했습니다.



### Monolingual and Multilingual Misinformation Detection for Low-Resource Languages: A Comprehensive Survey (https://arxiv.org/abs/2410.18390)
- **What's New**: 이번 연구는 정보의 오개념이 다양한 언어적 경계를 넘어 전 세계적으로 퍼지는 문제를 해결하기 위한 기여를 하고 있습니다. 특히, 낮은 리소스 언어(low-resource languages)에 대한 연구가 부족한 점을 강조하며, 이 분야에서의 데이터 수집, 모델 개발, 그리고 문화적 맥락 등의 과제를 다루고 있습니다.

- **Technical Details**: 이 논문에서는 낮은 리소스 언어를 위한 단일 언어 및 다중 언어에서의 허위 정보 감지에 대한 현재의 데이터셋, 방법론, 도구를 종합적으로 검토하였습니다. 연구는 크게 세 가지 단계로 구분되며: (1) 데이터 수집 및 주석 작업, (2) 데이터 처리, (3) 탐지 방법이 포함되어 있습니다. 방법론적으로는 기계 학습(classification), 임베딩(embedding) 생성 및 프롬프트 엔지니어링(prompt engineering)을 사용하는 생성형 대형 언어 모델(LLMs)을 활용합니다.

- **Performance Highlights**: 연구 결과는 낮은 리소스 언어에서의 허위 정보 감지를 위한 포괄적이고 강력한 시스템 구축 필요성을 확인했습니다. 또한, 데이터 수집 관행 개선, 학제 간 협력 및 사회적 책임을 고려한 AI 연구의 강화를 강조하였습니다.



### Improving Model Factuality with Fine-grained Critique-based Evaluator (https://arxiv.org/abs/2410.18359)
- **What's New**: 이 논문에서는 언어 모델의 Factuality(사실성) 평가를 개선하기 위해 FenCE(세밀한 비평 기반 평가기)를 훈련하여 제공하는 방식으로 언어 모델의 생성물에 대한 claim-level factuality feedback을 제공합니다.

- **Technical Details**: FenCE는 공공 데이터셋에서 인용된 데이터와 다양한 출처 문서들을 조합하여 훈련되며, 각 문서의 claim에 대한 텍스트 비평과 점수를 함께 제공하여 모델의 학습을 돕습니다. 이를 통해 생성된 응답을 평가하고 수정하여 트레이닝 데이터를 생성하고, 높게 평가된 수정 응답을 선호하여 언어 모델의 사실성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, FenCE의 도입으로 Llama3-8B-chat의 factuality 비율이 FActScore에서 14.45% 향상되었으며, 기존의 사실성 파인튜닝 방법보다 6.96% 뛰어난 성능을 보였습니다.



### AdaEDL: Early Draft Stopping for Speculative Decoding of Large Language Models via an Entropy-based Lower Bound on Token Acceptance Probability (https://arxiv.org/abs/2410.18351)
Comments:
          Workshop on Efficient Natural Language and Signal Processing at NeurIPS 2024

- **What's New**: 본 논문은 Adaptive Entropy-based Draft Length (AdaEDL)라는 새로운 기법을 제안합니다. 이 기법은 토큰 드래프팅 과정에서 드래프트 길이를 동적으로 조절할 수 있도록 해주며, 이는 기존의 정적 드래프트 길이 설정과 비교해 성능을 개선합니다.

- **Technical Details**: AdaEDL은 드래프트 모델의 로짓에서 관측된 엔트로피를 통해 드래프트 토큰의 수용 확률의 하한을 근사하여 초기에 드래프팅 과정을 중단할 수 있는 기준을 세웁니다. 이는 훈련이나 추가 파라미터 없이 작동하며, 다양한 LLM 시스템에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: AdaEDL은 기존의 정적 드래프트 길이 기반의 기법보다 10%에서 57% 더 나은 성능을 보였으며, 다른 훈련 없는 드래프트 중단 기법보다도 최대 10% 더 높은 성능을 기록했습니다. 또한 높은 샘플링 온도에서 성능을 일관되게 유지하며, 강건성이 뛰어난 점도 강조되었습니다.



### Aggregated Knowledge Model: Enhancing Domain-Specific QA with Fine-Tuned and Retrieval-Augmented Generation Models (https://arxiv.org/abs/2410.18344)
- **What's New**: 이 논문에서는 Lawrence Berkeley National Laboratory (LBL)의 ScienceIT 분야에서 닫힌 도메인 질문 응답(QA) 시스템을 향상시키기 위한 새로운 접근 방식을 소개합니다. 풍부한 데이터셋을 활용하여 세 가지 형태의 질문-답변 쌍을 생성하며, 집합 지식 모델(Aggregated Knowledge Model, AKM)을 제안하여 다양한 모델의 응답을 통합합니다.

- **Technical Details**: 논문은 AWS Bedrock, GCP PaLM2, Meta LLaMA2, OpenAI GPT-4 및 Google Gemini-Pro와 같은 최신 대형 언어 모델(LLMs)을 사용하여 QA 성능을 향상시키는 방법을 설명합니다. K-means clustering을 사용해 다양한 응답을 통합한 집합 지식 모델(AKM)은 보다 대표적인 답변을 선택합니다. 이 과정에서 데이터 처리 기술을 활용하여 양질의 질문-답변 쌍을 생성했습니다.

- **Performance Highlights**: 평가 결과는 다양한 지표를 통해 제안된 모델들이 LBL ScienceIT 환경에 적합하고 효과적임을 입증합니다. 특히, AKM은 파인튜닝과 검색 증강 전략을 통합하여 상당한 성능 개선을 보여주었습니다. 이 연구에서 얻은 통찰은 특정 도메인에 맞춘 전문 QA 시스템 개발에 기여할 수 있습니다.



### Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems (https://arxiv.org/abs/2410.18336)
- **What's New**: 본 연구는 AI 시스템이 단순히 정확한 수학 문제의 답변을 생성하는 것 이상으로, 새로운 해결책을 개발하거나 사람을 도와야 한다고 주장합니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)의 수학적 사고에서의 창의적 잠재력을 탐구합니다.

- **Technical Details**: 이 연구에서는 CreativeMath라는 새로운 프레임워크와 벤치마크를 도입합니다. 이 벤치마크는 중학교 수준의 문제부터 올림픽 수준의 경쟁 문제까지 다루며, LLM이 제공된 일부 알려진 해결책 이후에 혁신적인 해결책을 제안하는 능력을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 표준 수학 작업에서 좋은 성능을 보였으나 창의적 문제 해결 능력은 상당한 차이를 보였습니다. 특히 Gemini-1.5-Pro 모델이 독창적인 해결책을 생성하는 데 있어 다른 LLM보다 뛰어난 성과를 보였습니다.



### Measuring individual semantic networks: A simulation study (https://arxiv.org/abs/2410.18326)
- **What's New**: 이 연구는 개인 차이를 정확하게 포착하는 것이 의미 기억(semantic memory)의 기계적 이해를 발전시키는 데 기본적임을 강조합니다. 행동 패러다임으로부터 개인 차이를 측정하기 위한 개선된 디자인을 제안합니다.

- **Technical Details**: 이 연구는 두 가지 행동 패러다임인 자유 연상(free associations) 및 관련성 판단(relatedness judgment tasks)을 활용하여 개인 의미 네트워크의 심리 측정 속성을 조사하는 회복 시뮬레이션(recovery simulation)을 실시하였습니다. 시뮬레이션 결과, 절대 네트워크 특성에 대한 추정치는 심각한 편향(bias)에 취약하다는 것을 보여줍니다. 그러나 동일한 패러다임 및 디자인 구성에서 비교는 정확하고 일반화 가능합니다.

- **Performance Highlights**: 이 연구는 보통의 단서 수, 보통의 응답 수 및 다양한 단어가 포함된 단서 집합을 기반으로 하여 강력한 일반성을 지닌 준거에서 비교 가능성을 확보할 수 있음을 발견했습니다. 이러한 결과는 의미 네트워크의 구조에 대한 과거 연구 결과를 평가하고 개인 간 차이를 더욱 신뢰성 있게 드러낼 수 있는 새로운 연구를 설계하는 데 기여합니다.



### LEGO: Language Model Building Blocks (https://arxiv.org/abs/2410.18287)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)에서 작고 특정 작업에 맞춘 언어 모델(SLM)을 추출하여 조합하는 새로운 기술인 LEGO를 제안합니다. LEGO는 연합 학습(Federated Learning)과 혁신적인 집계 기법을 활용하여 LLM의 재구성을 가능하게 하면서 사용자 데이터 프라이버시를 유지합니다.

- **Technical Details**: LEGO는 LLM에서 SLM을 채택하는 과정에서 두 단계 접근 방식을 사용합니다. 우선 LLM을 가지치기(pruning)하여 다양한 크기의 SLM을 생성한 후, 이 SLM을 연합 학습 환경에 배치하여 최종적으로 LLM으로 집계합니다. 이 과정에서 LoRA(Hu et al., 2021) 기술을 활용하여 더 효율적인 미세 조정(fine-tuning)을 실현합니다.

- **Performance Highlights**: LEGO를 사용하여 SLM은 더 나은 학습 능력을 보이며, 데이터 이질성(data heterogeneity)에도 더 잘 적응합니다. 결과적으로 LEGO는 다양한 장치에 맞춘 모델을 생성하고, 해당 장치에서 더 강력하고 우수한 LLM을 구성할 수 있게 됩니다.



### Multilingual Hallucination Gaps in Large Language Models (https://arxiv.org/abs/2410.18270)
- **What's New**: 이번 연구에서는 다국어 자유 텍스트 생성에서 발생하는 환각(hallucination) 현상, 특히 다국어 환각 격차(multilingual hallucination gaps)에 대해 탐구합니다.

- **Technical Details**: 환각을 정량화하기 위해 FactScore 메트릭을 사용하고 이를 다국어 설정으로 확장했습니다. LLaMA, Qwen, Aya 모델을 사용하여 19개 언어로 전기(biography)를 생성하고 결과를 위키피디아 페이지와 비교했습니다.

- **Performance Highlights**: 연구 결과, 환각 비율에는 차이가 있었으며, 특히 고자원(high resource) 언어와 저자원(low resource) 언어 간에 나타나는 차이가 두드러졌습니다.



### Multi-Draft Speculative Sampling: Canonical Architectures and Theoretical Limits (https://arxiv.org/abs/2410.18234)
- **What's New**: 이번 연구는 다수의 초안 모델(draft model)에서 독립적으로 샘플링된 제안 시퀀스를 고려하는 multi-draft speculative sampling 접근 방식을 제안합니다. 최적의 토큰 선택 방식을 두 단계로 나누어 설명했으며, 이는 token-level draft selection과 single-draft speculative sampling을 포함합니다.

- **Technical Details**: 첫 번째 단계에서는 importance sampling (IS) 방식을 사용해 중간 토큰을 선택하고, 두 번째 단계에서는 선택된 토큰과 목표 모델(target model) 분포를 사용하여 최종 토큰을 생성하는 speculative sampling을 적용합니다. 두 개의 동일한 초안 모델을 사용하는 경우, 수용 확률이 1이도록 하는 충분조건과 필요조건을 설정하고, 최적 수용 확률에 대한 명시적인 표현을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 방식이 여러 시나리오에서 블록 효율(block efficiency) 및 토큰률(token rates)에서 기본 방식 대비 일관된 개선을 보임을 입증했습니다.



### Generalizations across filler-gap dependencies in neural language models (https://arxiv.org/abs/2410.18225)
Comments:
          accepted at CoNLL 2024

- **What's New**: 이번 연구에서는 Neural Language Model (NLM)이 filler-gap dependencies에 대한 공유된 표현을 posits하는지 여부를 조사합니다. 특히, NLM이 다양한 형태에서 같은 구조적 일반화를 나타내는지 확인합니다.

- **Technical Details**: 이 논문은 NLM에 clefting과 topicalization과 같은 특정한 예를 제공하여 filler-gap dependency를 인식하도록 하고, NLM이 이를 일반화하는지를 실험합니다. 실험은 Wh-movement, clefting, tough-movement, topicalization을 포함합니다.

- **Performance Highlights**: NLM은 문법적 filler-gap dependencies를 구별하는 데 성공적이지만, 입력의 표면적 특성에 의존하며, 공유된 일반화에는 의존하지 않는 것으로 관찰되었습니다. 이 연구는 언어 습득을 모델링하기 위한 특정 언어적 유도 편향이 필요함을 강조합니다.



### Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks (https://arxiv.org/abs/2410.18210)
Comments:
          14 pages, 6 figures, 7 tables

- **What's New**: 최근 다국어 대형 언어 모델(LLMs)에서 안전성 정렬(safety alignment) 공격이 언어 간 일반화(cross-lingual generalization)가 가능하다는 점이 확인되었습니다. 오직 하나의 언어로부터의 소수의 악성 지침-following 예를 사용해도 다국어 LLM이 손상될 수 있습니다.

- **Technical Details**: 이 연구에서는 Safety Information Localization (SIL) 방법을 통해 다국어 LLM 내의 안전 관련 정보를 국소화하고, 안전 관련 파라미터의 20%만 수정하면 모든 언어에서 안전성 정렬을 파괴할 수 있음을 밝혔습니다. 이는 악성 fine-tuning 공격을 통해 확인되었습니다.

- **Performance Highlights**: LLMs에서 단 100개의 악성 지침-following 예를 사용한 몇 단계의 fine-tuning으로, 다국어 LLM들의 안전성이 손상된 것으로 나타났습니다. 이 연구는 다국어 LLM의 안전성 문제를 해결하기 위한 중요한 통찰력을 제공합니다.



### CorrectionLM: Self-Corrections with SLM for Dialogue State Tracking (https://arxiv.org/abs/2410.18209)
- **What's New**: 이 논문에서는 CORRECTIONLM이라는 새로운 프레임워크를 소개하여 소형 언어 모델(SLM)이 대형 언어 모델(LLM)의 도움 없이 인콘텍스트 예제를 활용해 스스로를 수정할 수 있도록 합니다. 이 방법은 기존의 LLM에서 지식을 증류하는 접근 방식과는 달리 SLM의 자체 개선 기능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: CORRECTIONLM은 대화 상태 추적(Dialogue State Tracking, DST) 작업을 대상으로 하며, 저자들은 적은 자원 설정에서 SLM이 인콘텍스트 학습을 통해 생성한 예측 및 정답 레이블을 사용하여 파라미터 효율적으로 미세 조정(fine-tuning)합니다. 이 과정에서 기존 LLM를 필요로 하지 않고, 자체적으로 잘못된 예측을 수정할 수 있는 SLM을 생성합니다.

- **Performance Highlights**: 두 가지 낮은 자원 설정의 benchmark에서 실험한 결과, CORRECTIONLM은 인콘텍스트 예제를 활용하여 효과적인 수정을 수행하고, 이전의 최첨단 LLM에 비해 뛰어난 계산 효율성을 달성했습니다.



### Gazelle: An Instruction Dataset for Arabic Writing Assistanc (https://arxiv.org/abs/2410.18163)
Comments:
          EMNLP2024 Finding Camara-ready version

- **What's New**: 최근 생성형 AI의 발전으로, 특히 Large Language Models (LLMs)의 개발이 글쓰기 지원 도구의 구성을 변화시키고 있다. 그러나 아랍어와 같은 덜 대표되는 언어는 데이터 부족으로 인해 고급 AI 쓰기 도구 개발에 심각한 도전에 직면해 있다. 이를 해결하기 위해 Gazelle이라는 포괄적인 아랍어 글쓰기 지원 데이터 세트를 소개한다.

- **Technical Details**: Gazelle는 아랍어 글쓰기를 위해 특별히 선별된 수작업 데이터 세트로, 두 가지 주요 주제인 텍스트 리라이팅(text rewriting)과 글쓰기 조언(writing advice)을 중심으로 구성되어 있다. 텍스트 리라이팅 항목은 문법 오류 수정(Grammatical Error Correction, GEC), 은유 및 다중 단어 표현(Multi-word Expressions, MWEs) 수정, 텍스트 다듬기를 포함하고 있다. 글쓰기 조언 항목은 규칙 설명(Rule Explanation)과 아랍어의 변별(I’rab)을 다룬다.

- **Performance Highlights**: 최신 LLM인 GPT-4, GPT-4o, Cohere Command R+, Gemini 1.5 Pro를 포함하여 인간 평가를 수행한 결과, 이들 모델은 아랍어 글쓰기 도전 과제에 대해 각기 다른 강점과 한계를 보였다. 연구 결과는 아랍어 처리의 복잡성을 관리하기 위해 지속적인 모델 훈련과 데이터 세트 강화의 필요성을 강조하고 있다.



### Future Token Prediction -- Causal Language Modelling with Per-Token Semantic State Vector for Multi-Token Prediction (https://arxiv.org/abs/2410.18160)
Comments:
          15 pages, 7 figures, 3 tables

- **What's New**: 이 연구는 Future Token Prediction (FTP)이라는 새로운 사전 훈련 방법을 탐색합니다. 이 방법은 각 토큰 위치에 대한 임베딩 벡터를 생성하여 더 긴 텍스트의 의미를 포착하도록 돕습니다.

- **Technical Details**: FTP에서는 큰 Transformer encoder가 각 토큰 위치에 대한 상위 레이어 임베딩 벡터를 생성하고, 이를 언어 머리(language head)로 전달하는 대신 선형적으로 확장하여 의사 시퀀스(pseudo-sequence)로 투사합니다. 이후 작은 Transformer decoder가 시퀀스에서 그 위치로부터 N개의 다음 토큰을 예측하기 위해 교차 주의를 받습니다.

- **Performance Highlights**: FTP 모델에서 생성된 텍스트는 동일한 예측 당혹감(perplexity)으로 훈련된 표준 GPT 모델에 비해 주제 일관성이 개선되었습니다. 텍스트 분류 예제의 결과에 따라, FTP 모델은 텍스트의 주제를 더 잘 표현하는 벡터를 생성하였고, 복잡한 코딩 문제에서는 FTP 네트워크가 GPT 네트워크보다 현저히 더 나은 결과를 도출하였습니다.



### Meaning Typed Prompting: A Technique for Efficient, Reliable Structured Output Generation (https://arxiv.org/abs/2410.18146)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 고급 응용을 위해 구조화된 출력 생성을 신뢰할 수 있는 방법으로 개선한 새로운 기법인 의미 유형 프롬프팅(Meaning Typed Prompting, MTP)을 소개합니다.

- **Technical Details**: MTP는 변수(variable) 및 클래스(class)와 같은 유형(types), 의미(means), 추상화(abstraction)를 프롬프팅 과정에 통합하여 효율적인 구조화된 출력 생성을 가능하게 합니다. 표현력이 풍부한 유형 정의(type definitions)를 통해 MTP는 출력의 명확성을 강화하고 복잡한 추상화에 대한 의존도를 줄이며, 개발을 단순화하고 구현 효율성을 높입니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실증 분석을 통해 MTP는 기존 프레임워크에 비해 정확성(accuracy), 신뢰성(reliability), 일관성(consistency), 토큰 효율성(token efficiency)에서 우수한 성과를 보였습니다. 또한, MTP를 구현한 Semantix라는 프레임워크를 제시하여 실제 응용에 대한 통찰력을 제공합니다.



### Analyzing Nobel Prize Literature with Large Language Models (https://arxiv.org/abs/2410.18142)
- **What's New**: 이번 연구는 문학 분석의 맥락에서 고급 대형 언어 모델(LLMs), 특히 o1 모델의 능력을 검토합니다. 연구는 한강의 '아홉 개의 장'과 욘 포세의 '우정'이라는 두 개의 노벨 문학상을 수상한 단편 소설에 대해 인간 대학원생의 결과와 직접 비교합니다. 이 연구는 AI의 복잡한 문학 요소에의 관여 정도를 탐구하며, LLM이 문학적 해석에서 인간을 어떻게 보완하거나 도전할 수 있는지를 조명합니다.

- **Technical Details**: 연구는 LLM인 o1 모델이 주제를 분석하고, 상호텍스트성(intertextuality), 문화 및 역사적 맥락, 언어적 및 구조적 혁신, 캐릭터 개발 등 여러 복잡한 문학 차원에서 어떻게 기능하는지를 평가합니다. LLM과 인간 참가자의 출력을 비교함으로써 이 연구는 AI의 문학적 해석에서의 강점과 한계를 드러내고, 인간 지성의 복제를 향한 과제를 탐구합니다.

- **Performance Highlights**: o1 모델은 구조적 작업에서 강력한 분석 능력을 보여주었지만, 감정의 뉘앙스와 일관성 측면에서 인간 해석에 비해 부족함을 드러냈습니다. 이는 인문학 분야에서 인간과 AI의 협력 가능성을 강조하며, 문학 연구 및 기타 분야에서 새로운 기회를 열어줍니다.



### R2Gen-Mamba: A Selective State Space Model for Radiology Report Generation (https://arxiv.org/abs/2410.18135)
Comments:
          4 pages pages for ISBI2025

- **What's New**: 본 연구는 Mamba의 효율적인 시퀀스 처리와 Transformer 구조의 맥락적 이점을 활용한 새로운 자동 방사선 보고서 생성 방법인 R2Gen-Mamba를 제안합니다.

- **Technical Details**: R2Gen-Mamba는 Mamba 모델을 인코더로, Transformer를 디코더로 활용하며, 이는 낮은 계산 복잡도로 인해 학습 및 추론 효율성을 향상시킵니다. 이미지 패치의 특징을 입력으로 사용하고, 이에 대한 보고서를 생성하는 시퀀스-시퀀스 접근 방식을 채택합니다. Mamba의 선형 계산 복잡성 덕분에 높은 품질의 보고서를 생성할 수 있습니다.

- **Performance Highlights**: R2Gen-Mamba는 210,000개 이상의 X-ray 이미지-보고서 쌍을 포함하는 두 개의 벤치마크 데이터셋에서 전통적인 Transformer 기반 모델보다 높은 보고서 품질과 계산 효율성을 보였습니다. 이는 최신 기술(SOTA)들과 비교했을 때 리소스 효율적인 솔루션을 제공함을 나타냅니다.



### CAMEL-Bench: A Comprehensive Arabic LMM Benchmark (https://arxiv.org/abs/2410.18976)
Comments:
          10 pages, 5 figures, NAACL

- **What's New**: CAMEL-Bench라는 포괄적인 아랍어 대규모 멀티모달 모델 평가 벤치마크가 개발되었습니다. 이는 4억 명 이상의 아랍어 사용자를 위한 다양한 도메인과 하위 도메인으로 구성되어 있습니다.

- **Technical Details**: CAMEL-Bench는 8개의 다양한 도메인과 38개의 하위 도메인으로 구성되어 있으며, 총 29,036개의 질문이 포함되어 있습니다. 데이터는 GPT-4o를 이용해 아랍어로 번역되거나 수작업으로 수집되었습니다.

- **Performance Highlights**: GPT-4o 시리즈는 다양한 멀티모달 이해 작업에서 높은 성능을 보였으나, 아랍어 멀티모달 데이터 처리에서 개선이 필요함을 보여주었습니다.



### Unbounded: A Generative Infinite Game of Character Life Simulation (https://arxiv.org/abs/2410.18975)
Comments:
          18 pages; Project page: this https URL

- **What's New**: 전통적인 비디오 게임의 한계를 초월한 생성적 무한 게임(generative infinite game) 개념을 소개하고, 이를 통해 생성 AI를 활용한 새로운 게임인 Unbounded를 개발하였습니다. Unbounded는 플레이어가 자율 가상 캐릭터와 상호작용하고, 개방형 메커니즘을 통해 이야기를 생성하는 게임입니다.

- **Technical Details**: Unbounded는 다음과 같은 기술적 혁신을 포함합니다: (1) 게임 메커니즘, 내러티브 및 캐릭터 상호작용을 실시간으로 생성하는 특수화된 LLM(large language model) 및 (2) 다양한 환경에서 캐릭터를 일관되게 시각적으로 생성할 수 있는 새로운 동적 지역 이미지 프롬프트 어댑터(Regional Image Prompt Adapter)입니다. 이 기술들은 생동감 있는 캐릭터 시뮬레이션과 화면 간의 일관성 높은 시각적 생성이 가능합니다.

- **Performance Highlights**: 전통적인 접근 방식과 비교했을 때, 캐릭터 삶의 시뮬레이션, 사용자 지시 따르기, 내러티브 일관성 및 캐릭터와 환경의 시각적 일관성에서 유의미한 개선을 보여주었습니다. 특히, 우리가 개발한 distilled LLM은 대형 LLM과 비교했을 때도 상호작용 속도에서 비슷한 성능을 보여 줍니다.



### Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms (https://arxiv.org/abs/2410.18967)
- **What's New**: Ferret-UI 2가 다양한 플랫폼에서의 사용자 인터페이스(UI) 이해를 위한 혁신적인 멀티모달 대규모 언어 모델(MLLM)로 소개됩니다. 이 모델은 iPhone, Android, iPad, Webpage, AppleTV 등 여러 플랫폼을 지원하며, 여러 혁신적인 기능을 통해 사용자 중심의 복잡한 상호작용을 수행합니다.

- **Technical Details**: Ferret-UI 2는 세 가지 주요 혁신을 도입합니다: (1) 다양한 플랫폼 지원, (2) 적응형 스케일링을 통한 고해상도 인식, (3) GPT-4o를 활용한 시각적 프롬프트 기반 고품질 훈련 데이터 생성. 이러한 발전은 UI 요소에 대한 더 정확한 인식을 가능하게 하여 사용자의 의도에 맞춘 상호작용을 지원합니다.

- **Performance Highlights**: Ferret-UI 2는 여러 벤치마크와 실험 데이터셋에서 뛰어난 성능을 보여 주며, 9개 하위 작업과 5개의 플랫폼을 포함한 사용자 중심의 고급 작업에서 Ferret-UI보다 우수한 결과를 나타냅니다. GUIDE 및 GUI-World와 같은 최신 벤치마크에서도 강력한 성과를 기록했습니다.



### OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning (https://arxiv.org/abs/2410.18963)
Comments:
          Work in progress

- **What's New**: OSCAR는 다양한 데스크탑 및 모바일 애플리케이션을 통해 자율적으로 탐색하고 상호 작용할 수 있도록 설계된 범용 에이전트입니다. 이 에이전트는 마우스와 키보드 입력을 표준화하여 다양한 작업을 수행할 수 있으며, 화면 이미지를 처리하여 사용자 명령을 실행합니다.

- **Technical Details**: OSCAR는 상태 기계(state machine)로 운영되며, 오류 처리 메커니즘과 동적 작업 재계획(dynamic task re-planning) 기능을 갖추고 있어 실시간 피드백과 예외에 효율적으로 대응할 수 있습니다. OSCAR는 사용자 명령을 실행 가능한 Python 코드로 번역하여 GUI(Graphical User Interface)에 대한 정밀한 제어를 가능하게 합니다.

- **Performance Highlights**: OSCAR는 GAIA 벤치마크에서 평균 28.7%의 성공률을 기록하며, 가장 복잡한 Level 3 작업에서는 13.5%의 성공률을 보여 이전의 최고 성능을 거의 두 배로 증가시켰습니다. OSWorld 및 AndroidWorld 벤치마크에서도 각각 24.5% 및 61.6%의 성공률을 달성하여 실시간 동적 OS 작업에서의 우수한 적응성을 입증했습니다.



### Schema-Guided Culture-Aware Complex Event Simulation with Multi-Agent Role-Play (https://arxiv.org/abs/2410.18935)
Comments:
          Accepted as EMNLP 2024 Demo

- **What's New**: 이 논문은 정부와 사회가 빠르게 대응해야 하는 복잡한 뉴스 사건, 예를 들어 자연 재해와 사회 정치적 갈등에 대한 신속한 대응의 필요성을 강조합니다. 특히, 역사적 사건을 기반으로 미래를 예측하기에는 한계가 있으므로, 이러한 사건을 시뮬레이션할 수 있는 새로운 복합 뉴스 사건 시뮬레이터인 Miriam을 개발했습니다. Miriam은 사건 schema와 사용자 제공 가정을 기반으로 사용자 정의 가능한 복잡한 사건 시뮬레이션을 생성합니다.

- **Technical Details**: Miriam은 시나리오에 대한 도메인 지식을 나타내는 사건 schema와 사건 특정 조건을 나타내는 사용자 제공 가정을 결합하여, 사회 및 문화적 맥락에 따라 사건의 역학을 시뮬레이션합니다. 에이전트 기반 시뮬레이션을 통해 시뮬레이션 과정에서 개인 캐릭터의 상태, 계획 및 행동을 시뮬레이션하고, 이는 현실적이며 일관된 사건 생성을 가능하게 합니다. 시뮬레이터는 전체 사건 로그와 개요 문서 형태로 결과를 제공합니다.

- **Performance Highlights**: Miriam은 인도적 지원 조직의 참가자들에게 매우 긍정적인 반응을 얻었으며, 문화적 규범을 반영한 더 적합하고 일관된 시뮬레이션을 통해 미래 재해의 예방 및 관리에 기여할 수 있는 잠재력을 보여주고 있습니다.



### Provably Robust Watermarks for Open-Source Language Models (https://arxiv.org/abs/2410.18861)
- **What's New**: 이번 논문은 오픈소스 대형 언어 모델(LLM)을 위한 최초의 워터마크(watermark) 기법을 제시합니다. 기존의 방법들이 모델의 비밀 매개변수에 의존하는 반면, 본 연구에서는 모델의 매개변수를 수정하여 워터마크를 삽입하되 모델의 출력으로만 감지할 수 있는 방식으로 접근합니다.

- **Technical Details**: 저자들은 수식어 기반 워터마크(sampler-based watermark)와 달리, 모델의 가중치를 수정하여 워터마크를 삽입하고, 이 워터마크는 특정 제약 조건 하에서 제거할 수 없음을 입증했습니다. 특히, 이 워터마크는 입력된 텍스트(약 300 토큰)로부터도 충분히 감지할 수 있습니다. 공격자가 워터마크를 제거하기 위해서는 모델 품질 점수를 100점 만점에 0점으로 떨어뜨리면서 감지율을 50%로 낮춰야 합니다.

- **Performance Highlights**: OPT-6.7B 및 OPT-1.3B 모델을 사용한 실험 결과, 노출된 파라미터로부터도 이 워터마크를 감지할 수 있으며, 실제 언어 생성에서 높은 엔트로피 텍스트에 대해서도 성공적으로 작동함을 보여주었습니다.



### Demystifying Large Language Models for Medicine: A Primer (https://arxiv.org/abs/2410.18856)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 의료 분야에서 효과적으로 활용할 수 있는 단계별 가이드라인과 최고의 사례를 제공합니다.

- **Technical Details**: 이 연구는 LLM을 활용하는데 필요한 주요 단계인 작업 설정, LLM 선택, prompt engineering, fine-tuning, 및 배포를 포함하는 방법론을 설명합니다. 핵심 고려사항으로는 의료 작업과 LLM의 핵심 기능의 일치를 찾고, 선택된 작업 및 데이터, 성과 요구 사항, 모델 인터페이스에 따라 모델을 선택하는 것이 포함됩니다.

- **Performance Highlights**: 의료 전문가들이 LLM을 임상 실무에 안전하고 신뢰성 있게 통합할 수 있도록 돕기 위해 전략적인 접근 방식을 제시하며, 규제 준수, 윤리적 가이드라인, 공정성과 편향성에 대한 지속적인 모니터링을 강조합니다.



### A Combinatorial Approach to Neural Emergent Communication (https://arxiv.org/abs/2410.18806)
- **What's New**: Deep learning 기반의 emergent communication 연구에서 Lewis signaling game의 문제점을 지적하며, 성공적인 커뮤니케이션에 필요한 최소 심볼 수를 계산하는 알고리즘 SolveMinSym(SMS)을 제안합니다.

- **Technical Details**: 이 논문은 Lewis signaling game 설정 하에, 성공적인 커뮤니케이션을 위해 필요한 최소 심볼 수 min(|M|)를 이론적으로 분석하고, 다양한 min(|M|)을 갖는 데이터셋을 생성하여 유의미한 실험을 진행합니다. 이러한 접근법을 통해 훈련 데이터의 심볼 수가 emergent 언어의 효과적인 심볼 수에 미치는 영향을 조사합니다.

- **Performance Highlights**: 실험 결과, 훈련 데이터의 min(|M|)를 증가시켰을 때 emergent 언어의 효과적인 심볼 수가 증가함을 입증하였습니다. 이는 한두 개의 심볼만으로 이루어진 단순한 언어에서 벗어나, 더 긴 조합 언어의 생성을 위한 새로운 통찰을 제공합니다.



### An LLM Agent for Automatic Geospatial Data Analysis (https://arxiv.org/abs/2410.18792)
- **What's New**: 이 논문에서는 GeoAgent라는 새로운 프레임워크를 소개하여 대형 언어 모델(LLM)이 지리공간 데이터 처리에 보다 효과적으로 대응할 수 있도록 돕습니다. 이 프레임워크는 Monte Carlo Tree Search (MCTS) 알고리즘 내에서 코드 인터프리터, 정적 분석(static analysis) 및 Retrieval-Augmented Generation (RAG) 기술을 통합하여 지리공간 데이터 분석을 위한 혁신적인 접근법을 제공합니다.

- **Technical Details**: GeoAgent는 지리공간 데이터 분석의 복잡한 지침 이해를 해결하기 위해 MCTS 및 RAG 기술을 활용합니다. 이 프레임워크는 다양한 파이썬 라이브러리를 사용하여 데이터 수집, 처리 및 시각화와 같은 다양한 작업을 포함하는 기준 벤치마크를 제공합니다. GeoAgent는 LLM의 다단계 프로세스를 지원하여 여러 기능 호출을 정확히 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: GeoAgent는 기존 LLM과 비교하여 기능 호출 및 작업 완료에서 현저한 개선을 보여주었습니다. 1000개의 풍부한 컨텍스트의 지리공간 데이터 분석 작업을 포함하는 GeoCode라는 벤치마크에서 평가하면서, GeoAgent는 레스터 분석, 벡터 분석 및 혼합형 작업에서 탁월한 성능을 보였습니다.



### A Little Help Goes a Long Way: Efficient LLM Training by Leveraging Small LMs (https://arxiv.org/abs/2410.18779)
- **What's New**: 이 논문은 LLM(large language model) 개발에 있어 중요한 도전 과제인 사전 훈련(pre-training) 비용 절감을 위해 SLM(small language model)을 활용하는 새로운 패러다임을 제시합니다. SLM을 통해 추가적인 훈련 감독(supervision)을 제공하고, 가치 있는 훈련 예시를 선택함으로써 LLM의 훈련 효율성과 품질을 향상시키려는 전략이 중심입니다.

- **Technical Details**: SLM은 soft labels를 제공하여 효율적인 훈련을 지원하며, 'easy'하고 'hard'한 훈련 예시를 구분하여 선택하는 역할을 합니다. 이 연구에서는 SLM의 예측 분포를 LLM으로 효과적으로 전이하는 방법론을 제안하며, 이론적으로 KD(knowledge distillation)의 통계적 프레임워크를 개발하였습니다. 이를 통해 SLM이 LLM의 품질을 높일 수 있는 방법을 체계적으로 연구합니다.

- **Performance Highlights**: SALT(small model aided large model training) 방법론을 통해 2.8B 파라미터 LLM을 1.5B 파라미터 SLM을 이용해 훈련하였으며, 이는 표준 훈련 방법보다 훈련 시간과 효율성을 현저히 개선하여 28%의 시간을 절감했습니다. 또한, SALT 모델은 여러 도메인에서 후속 훈련(Supervised Fine-Tuning, SFT) 후에도 뛰어난 성능 향상을 보였습니다.



### Health Misinformation in Social Networks: A Survey of IT Approaches (https://arxiv.org/abs/2410.18670)
Comments:
          Preprint -- Under review in the ACM Transactions on Computing for Healthcare (HEALTH) journal

- **What's New**: 이번 연구는 정보기술 관점에서 소셜 네트워크의 의료 허위정보 문제에 대한 포괄적인 조사 결과를 제시합니다. 이 조사는 의료 허위정보와 관련된 연구에 대한 체계적인 리뷰를 제공하고 연구자와 실무자가 이 빠르게 변화하는 분야를 이해하도록 돕기 위해 구성되었습니다.

- **Technical Details**: 이 논문은 수동적 및 자동적 사실 확인(fact-checking) 접근 방식을 소개하고, 콘텐츠(content), 전파 특징(propagation features), 출처 특징(source features)을 이용한 가짜 뉴스 탐지(fake news detection) 방법을 탐색합니다. 또한 허위정보 확산을 저지하기 위한 완화 접근(mitigation approaches)도 논의합니다. 여러 의료 허위정보 데이터셋과 공개 사용 가능한 도구 목록도 제공됩니다.

- **Performance Highlights**: 이 연구는 200편 이상의 논문과 24개의 공개 데이터셋, 11개의 사실 확인 도구를 참조하여 의료 분야에서의 가짜 뉴스 탐지 기술의 현황을 조사합니다. 특히 COVID-19와 관련된 허위정보에 대한 대응을 위한 노력이 강조됩니다. 또한, 대부분의 의료 종사자들이 COVID-19 환자를 치료하는 데 있어 의료 허위정보가 크게 방해가 되었다고 언급했습니다.



### $C^2$: Scalable Auto-Feedback for LLM-based Chart Generation (https://arxiv.org/abs/2410.18652)
Comments:
          Preprint

- **What's New**: 본 논문에서는 인간 개입의 비용을 줄이기 위해 고품질 차트를 생성하는데 있어 새로운 자동 피드백 생성기(automatic feedback generator), ChartAF를 소개합니다. 이 시스템은 참고(reference) 없이 작동하며, 데이터 세트 ChartUIE-8K를 활용하여 데이터 다양성을 크게 개선합니다.

- **Technical Details**: 새로운 프레임워크 $C^2$는 (1) 자동 피드백 제공자(ChartAF)와 (2) 다양한 참고 없는 데이터 세트(ChartUIE-8K)로 구성됩니다. ChartUIE-8K는 기존 기준 대비 각각 5982%, 1936%, 91%의 쿼리, 데이터, 차트 유형을 증가시켜 데이터 다양성을 크게 향상시킵니다.

- **Performance Highlights**: 첫 번째 실험에서 74%의 응답자가 피드백 후 결과를 강하게 선호했으며, 10%는 선호한다고 답했습니다. 두 번째 후속 실험에서 ChartAF는 아홉 개의 기준선(Baseline)보다 우수한 성능을 보여주었습니다. 또한 LLM 사용자 연구에서 참여자의 94%가 ChartUIE-8K의 쿼리를 선호하고, 93%가 실제 사용 사례와 일치한다고 평가했습니다.



### Speech perception: a model of word recognition (https://arxiv.org/abs/2410.18590)
Comments:
          22 pages, 19 figures, 1 table

- **What's New**: 본 논문은 음소들 간의 상관관계를 고려하여 음성 인식 모델을 제시합니다. 이 모델에서는 적절하게 선택된 감소 동역학에 해당하는 매력점(attractor)으로서 단어에 연관되어 있으며, 짧은 단어에 대한 복구는 빠르게 이루어지지만 긴 단어의 경우는 영구적으로 잃어버릴 가능성이 존재함을 강조합니다.

- **Technical Details**: 모델은 음성의 기본 단위를 '소리'로 정의하며, 짧은 단어와 긴 단어의 암호 해독 과정에 대한 동역학을 논의합니다. 단어들은 스핀 시스템의 고정점으로 정의되며, 이를 통해 잘못 들은 단어의 복구를 시뮬레이션합니다. Gamma 분포를 사용하여 단어 길이 분포의 주요 특징을 모델링하며, 테크니컬한 요소로는 이징 스핀 체인을 활용합니다.

- **Performance Highlights**: 짧은 단어에서는 알고리즘이 빠르게 단어를 검색하거나 유효한 다른 단어를 제안하며, 긴 단어에서는 복구가 지속적으로 이루어지나 영구적으로 잃어버릴 확률이 존재합니다. 이러한 발견은 짧은 단어에 비해 긴 단어의 복구가 더 어려워짐을 보여줍니다.



### KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing (https://arxiv.org/abs/2410.18517)
Comments:
          Under Review by ICLR2025

- **What's New**: 최근의 대형 언어 모델 (LLMs)들은 Transformer 아키텍처를 기반으로 하여 뛰어난 성능을 보였지만, 이로 인해 상당한 GPU 메모리 요구가 발생하고 있습니다. 본 논문에서는 KV (key-value) 캐시를 층 간으로 공유하여 메모리 소비를 줄이고 성능을 유지하는 새로운 방법인 KVSharer를 제안합니다.

- **Technical Details**: KVSharer는 레이어 간 KV 캐시를 공유하여 레이어 단위의 압축을 수행하는 플러그-앤-플레이 방법입니다. 이는 직관적으로 유사한 KV 캐시를 공유하는 것이 아니라, 상반된 특성을 가진 KV 캐시를 공유할 때 모델 성능을 더 잘 유지할 수 있다는 경험적 발견에 기반합니다. 이 방법은 미세 조정 없이도 잘 훈련된 LLM에 바로 적용할 수 있습니다.

- **Performance Highlights**: KVSharer를 사용하면 KV 캐시 계산량을 30% 감소시킬 수 있으며, 이로 인해 메모리 소비가 줄어들고 성능에는 큰 영향을 미치지 않는 것으로 실험에서 나타났습니다. 또한, 최소 1.3배의 생성 속도 향상을 달성할 수 있으며, 기존의 intra-layer KV 캐시 압축 방법과의 호환성도 보장합니다.



### Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs (https://arxiv.org/abs/2410.18451)
- **What's New**: 이번 보고서에서는 LLM(대형 언어 모델)을 위한 보상 모델링을 개선하기 위한 데이터 중심의 기술들이 소개되었습니다. 특히, 높은 품질의 오픈 소스 선호 데이터셋을 구축하기 위한 데이터 선택 및 필터링 전략이 제안되어, 총 80K의 선호 쌍으로 이루어진 Skywork-Reward 데이터 컬렉션이 개발되었습니다. 이 데이터를 활용하여 Skywork-Reward 모델 시리즈를 개발하였으며, 현재 RewardBench 리더보드에서 1위 자리를 차지하고 있습니다.

- **Technical Details**: 제안된 기술들은 효과적인 데이터 선택 및 필터링 전략을 포함하며, 가벼우면서도 효과적인 선호 데이터 컬렉션을 제공합니다. 본 연구에서는 다양한 손실 함수에 대한 광범위한 연구를 진행하였고, 기본적인 Bradley-Terry 손실 함수가 보상 모델링 작업에서 일관되게 높은 성능을 보였습니다. Skywork-Reward 모델 시리즈는 이러한 훈련 기술을 집약하여 RewardBench 벤치마크에서의 성능이 크게 향상되었음을 입증하였습니다.

- **Performance Highlights**: Skywork-Reward 모델 시리즈는 RewardBench 리더보드에서 1위와 7위를 기록하고 있으며, 제안한 데이터 셋 및 기술이 많은 상위 모델들의 성능 향상에 기여하고 있습니다. 이러한 기여는 LLM의 사용자 선호 학습 응용에 있어 실질적인 효과를 강조합니다.



### WAFFLE: Multi-Modal Model for Automated Front-End Developmen (https://arxiv.org/abs/2410.18362)
- **What's New**: 본 논문에서는 UI 디자인 이미지를 HTML 코드로 변환할 때의 두 가지 주요 도전 과제를 해결하기 위한 새로운 미세 조정 전략인 Waffle을 소개합니다. Waffle은 구조 인식을 위한 주의 메커니즘과 대조적 미세 조정 접근 방식을 활용하여 LLMs의 이해도를 향상시킵니다.

- **Technical Details**: Waffle은 LLMs가 HTML 코드의 구조와 UI 이미지 간의 관계를 효과적으로 학습할 수 있도록 돕는 구조 인식 주의 메커니즘을 설계합니다. 이는 HTML 코드의 이전 세그먼트에 대한 주의를 허용하여 모델이 가장 관련성 높은 부분에 집중할 수 있도록 합니다. 또한, 대조적 학습 기법을 통해 모델이 UI 이미지의 미세한 시각적 차이를 인식하도록 돕습니다. 이 방식은 HTML 코드와 렌더링된 UI 디자인 간의 복잡한 관계를 명확히 이해하는 데 기여합니다.

- **Performance Highlights**: 새로운 벤치마크인 WebSight-Test와 기존 벤치마크인 Design2Code를 통해 Waffle을 적용한 모델은 기존 미세 조정 기법에 비해 9.00 pp의 HTML 일치도 향상, 0.0982의 CW-SSIM 증가, 32.99의 CLIP 점수 증가, 27.12 pp의 LLEM 향상을 기록했습니다. 이는 Waffle이 모델 독립적으로 작동하여 다양한 MLLMs을 개선하는 데 적용할 수 있음을 보여줍니다.



### CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation (https://arxiv.org/abs/2410.18311)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 CoreInfer라는 새로운 MLP-free adaptive sparse activation inference 방법을 제안합니다. 이 방법은 토큰이 아닌 문장 수준에서 activation을 예측하는 방식을 사용하여 inference 속도를 크게 높입니다.

- **Technical Details**: CoreInfer는 각 문장에 대해 가장 중요한 뉴런 집합인 core neurons를 정의하고, 이들 뉴런이 문장의 의미와 상관성이 크다는 것을 실험적으로 입증합니다. 이 방법은 pretrained stage에서 core neurons을 한 번만 예측하고, decoding 단계에서는 이 집합을 고정하여 반복적인 예측을 필요로 하지 않습니다.

- **Performance Highlights**: CoreInfer는 NVIDIA TITAN XP GPU에서 Huggingface 및 PowerInfer와 비교하여 각각 10.33배와 2.72배의 속도 향상을 달성했습니다. 이는 높은 성능 저하 없이 얻은 결과입니다.



### Robust and Explainable Depression Identification from Speech Using Vowel-Based Ensemble Learning Approaches (https://arxiv.org/abs/2410.18298)
Comments:
          accepted at the IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2024)

- **What's New**: 본 연구는 머신러닝을 활용하여 음성으로부터 우울증을 식별하기 위한 설명 가능한 알고리즘을 제안하고 있습니다. 이 연구는 발견된 정보를 바탕으로, 우울증 유무 및 그 심각도를 파악하는 데 있어 더욱 신뢰할 수 있는 도구가 될 수 있습니다.

- **Technical Details**: 음성 생산이 우울증에 영향을 미친다는 증거에 기반하여, 사전 훈련된 모음 기반 임베딩을 사용하고, 앙상블 학습(ensemble learning) 접근 방식을 통해 문제를 특정 우울증 증상 및 심각도 수준으로 분해합니다. '하향식(top-down)' 접근 방식과 '상향식(bottom-up)' 접근 방식을 사용하여 8개의 모델이 개별 PHQ-8 항목 점수를 예측합니다.

- **Performance Highlights**: 제안된 두 시스템 모두 여러 최신 기준선(state-of-the-art)과 비교했을 때 유사한 성능을 보이며, 데이터의 평균/중앙값에 덜 영향을 받는 것으로 나타났습니다. 이는 우울증 진단 및 스크리닝을 지원하는 데 유용할 수 있습니다.



### Kenyan Sign Language (KSL) Dataset: Using Artificial Intelligence (AI) in Bridging Communication Barrier among the Deaf Learners (https://arxiv.org/abs/2410.18295)
Comments:
          14 pages, to be published in 3rd International Conference on Artificial Intelligence and Robotics (MIRG-ICAIR 2023)

- **What's New**: 케냐 수어(Kenyan Sign Language, KSL)는 케냐의 청각 장애인 커뮤니티가 사용하는 주요 언어입니다. 이 프로젝트인 AI4KSL은 청각 장애인과 비장애인 간의 의사소통 장벽을 해소하기 위한 기술 혁신입니다.

- **Technical Details**: 본 연구는 2023년부터 2024년까지 진행되는 2년간의 연구 프로젝트로, 케냐 청각 장애인의 대표 샘플을 바탕으로 한 spontaneuous 및 elicited 데이터의 디지털 오픈 액세스 AI를 개발하는 것을 목표로 합니다. KSL 데이터셋은 영어 문장 14,000개와 연관된 KSL Gloss, 20,000개 이상의 서명된 KSL 비디오, 10,000개의 분할 및 세분화된 KSL 비디오를 포함하여 HamNoSys 시스템에 따라 4,000개의 단어가 5개의 발음 매개변수로 전사되었습니다.

- **Performance Highlights**: 이 연구 결과는 케냐의 청각 장애인 학습자를 위한 AI 보조 기술 데이터셋을 개발하는 데 기여하여, 언어 장벽을 해소하고 포용성을 증진시키는 데 큰 역할을 할 것입니다.



### Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models (https://arxiv.org/abs/2410.18252)
Comments:
          code at this https URL

- **What's New**: 이 논문에서는 비동기(asynchronous) 및 오프 정책(off-policy) 강화 학습(RL) 방법을 활용하여 RLHF(인간 피드백을 통한 강화 학습)의 효율성을 높이는 방안을 제안합니다. 이를 통해 더 빠른 학습을 가능케 하고, 성능 저하를 최소화하는 방법을 모색합니다.

- **Technical Details**: RLHF의 전통적인 접근법은 온라인(on-policy) 방식으로, 모델이 생성한 출력을 사용해 피드백을 받고 이를 통해 학습하는 것입니다. 연구에서는 비동기 방식으로 새로운 샘플을 생성하고, 이전의 샘플로부터 학습하는 오프 정책 강화 학습 방법을 도입하여 시간 효율성을 개선합니다.

- **Performance Highlights**: LLaMA 3.1 8B 모델을 사용한 실험에서는 비동기 RLHF 방식이 동기화(synchronous) 방법보다 약 40% 더 빠르게 학습하며, 동일한 성능을 유지하는 것을 확인하였습니다. 이 논문에서 제안된 온라인 DPO 방식은 오프 정책 데이터에 대한 강건성이 가장 높으며, 정책 모델의 크기가 커질수록 더 좋은 성능을 보여주었습니다.



### Optimizing the role of human evaluation in LLM-based spoken document summarization systems (https://arxiv.org/abs/2410.18218)
- **What's New**: 본 논문은 Generative AI의 콘텐츠에서 음성 문서 요약을 위한 평가 패러다임을 제안하며, 사회과학의 방법론을 활용하여 인간 평가의 신뢰성을 강화하고자 합니다.

- **Technical Details**: 저자는 네 가지 평가 기준: 양(Quantity), 질(Quality), 관련성(Relevance), 매너(Manner)를 기반으로 한 평가 방법론을 제시하고, 인간 기반 평가 체계 구현에 대한 구체적인 지침을 제공합니다.

- **Performance Highlights**: 미국의 한 기술 회사에서 이 평가 방법론을 사용하여 두 가지 기능을 평가하며, 이러한 방법이 실제로 적용되었음을 보여주는 사례 연구를 포함합니다.



### Advancing NLP Security by Leveraging LLMs as Adversarial Engines (https://arxiv.org/abs/2410.18215)
Comments:
          5 pages

- **What's New**: 이 논문에서는 Large Language Models (LLMs)를 활용하여 다채로운 적대적 공격을 생성하는 새로운 접근 방식을 제안하고 있습니다. 최근 연구 결과를 바탕으로 LLM가 단어 수준의 적대적 예제를 생성하는 데 효과적임을 보여주었으며, 이를 다양한 공격 유형으로 확장하려고 합니다.

- **Technical Details**: LLMs는 맥락에 맞는 텍스트 조각을 생성하여, 오분류를 일으킬 수 있는 적대적 패치(adversarial patches), 여러 입력에 걸쳐 효과적인 범용 변형(universal perturbations), 특정 misclassification을 겨냥한 타겟 공격(targeted attacks) 등의 다양한 적대적 공격을 생성할 수 있습니다.

- **Performance Highlights**: LLM을 사용하여 생성된 적대적 예제는 인간이 작성한 텍스트와 구별이 어렵고, 텍스트의 본래 의도(intent)를 유지하면서도 분류기를 속이는데 효과적일 것입니다. 이는 NLP 시스템의 신뢰성과 안전성을 향상시키는데 큰 기여를 할 것으로 기대됩니다.



### ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignmen (https://arxiv.org/abs/2410.18194)
- **What's New**: 이 논문에서는 ZIP-FIT이라는 데이터 선택 프레임워크를 소개하며, 이는 gzip 압축을 활용하여 잠재적인 훈련 데이터와 목표 작업 분포 간의 정렬을 직접 측정합니다.

- **Technical Details**: ZIP-FIT은 LZ77과 Huffman 코딩이라는 두 가지 주요 압축 기법을 사용하여 데이터 내에서 반복 패턴을 이용하여 압축을 수행하며, 수학적 언어 표기법인 Autoformalization과 Python 코드 생성 과제를 통해 평가되었습니다.

- **Performance Highlights**: ZIP-FIT은 기존의 데이터 선택 방식들보다 일관되게 모델 성능을 향상시키며, DSIR 및 D4와 같은 주요 기준선을 초과하여 85.1% 더 빠른 수렴 및 낮은 크로스 엔트로피 손실을 달성했습니다.



### SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback (https://arxiv.org/abs/2410.18141)
- **What's New**: 이 논문에서는 RAG 시스템의 모든 모듈이 개별적으로 최적화되는 대신 공동 최적화되어야 한다고 주장합니다. 이를 위해 'SmartRAG'라는 새로운 파이프라인을 설계하고, 강화 학습(rl)을 통해 시스템을 공동 최적화합니다.

- **Technical Details**: SmartRAG는 정책 네트워크(policy network)와 리트리버(retriever)를 포함하며, 정책 네트워크는 1) 언제 검색할지 결정, 2) 검색기와 가장 적합한 쿼리 생성, 3) 최종 응답 생성의 세 가지 역할을 합니다. 시스템의 최적화는 환경 피드백을 기반으로 이루어집니다.

- **Performance Highlights**: SmartRAG는 다양한 데이터셋에서 실험 결과, 개별적으로 최적화된 모듈보다 우수한 성능을 보였으며, retrieval 비용을 최소화하면서도 정확한 응답을 생성하는 데 성공했습니다.



### Tethering Broken Themes: Aligning Neural Topic Models with Labels and Authors (https://arxiv.org/abs/2410.18140)
- **What's New**: 본 논문에서는 FANToM이라는 새로운 방법론을 제안하여, 레이블(label) 및 저자(author) 정보와의 정렬을 통해 신경 주제 모델(neural topic models, NTMs)의 해석 가능성을 향상시킵니다.

- **Technical Details**: FANToM은 레이블 및 저자 메타데이터를 기존의 NTMs에 통합하여 학습된 주제를 더욱 해석 가능하게 만듭니다. 이 방법은 레이블, 주제, 저자 간의 정렬을 학습함으로써 더 큰 표현력을 제공합니다. 또한, FANToM은 삽입된 메타데이터를 사용하여 저자들의 관심사 및 유사성을 파악합니다.

- **Performance Highlights**: 실험 결과, FANToM은 기존 모델에 비해 주제 품질 및 정렬을 개선하며, 학습된 주제의 품질을 향상시키는 데 성공하였습니다. FANToM은 저자, 단어, 주제가 공유하는 임베딩 공간을 학습하도록 도와주어 저자들의 관심사와 유사성에 대한 통찰을 제공합니다.



### Graph Contrastive Learning via Cluster-refined Negative Sampling for Semi-supervised Text Classification (https://arxiv.org/abs/2410.18130)
Comments:
          7 pages, 3 figures

- **What's New**: 본 논문은 클러스터 정제된 negative sampling을 통해 오버 클러스터링 문제를 해결하는 새로운 그래프 대비 학습 방법인 ClusterText를 제안합니다.

- **Technical Details**: ClusterText는 Graph Neural Networks(GNNs)와 pretrained 모델인 BERT를 결합하여 텍스트 표현을 학습하고, 클러스터 정제 전략을 사용해 pseudo labels를 생성합니다. 이를 통해 다양한 클러스터에서 negative sample을 추출하고, self-correction 메커니즘을 도입하여 clustering의 불일치로 인해 발생하는 true negative samples 손실을 완화합니다.

- **Performance Highlights**: 실험 결과, ClusterText는 다섯 개의 인기 있는 벤치마크 데이터셋에서 텍스트 분류 작업에 대해 우수한 성능을 보여줍니다.



### Optimizing Preference Alignment with Differentiable NDCG Ranking (https://arxiv.org/abs/2410.18127)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 인간 선호와의 정렬을 새로운 방식으로 접근하여, Ranking 데이터 기반의 최적화 방법인 Direct Ranking Preference Optimization (DRPO)를 제안합니다. 이는 현재의 방법들(DPO)의 한계를 극복하고, NDCG와 같은 측정 지표를 직접 최적화합니다.

- **Technical Details**: DRPO는 Learning-to-Rank (LTR) 접근법을 통해 인간 선호 데이터를 활용하여 응답 목록의 순위를 최적화합니다. 이 과정에서 diffNDCG라는 차별 가능한 손실 함수를 도입하고, 이는 정렬된 목록에서의 응답 간 분류 정확도를 높여줍니다. 적응형 순위 정책 점수(Adaptive Rank Policy Score)를 통해 응답의 상대적 위치에 따라 점수 마진을 조정하며, 다양한 순위 모델과 함께 사용됩니다.

- **Performance Highlights**: DRPO는 기존의 방법들과 비교하여 생성된 응답의 질에서 우수한 성능을 보이며, 수많은 실험 결과에서 DRPO가 기존의 기준 선형 모델들을 능가함을 입증했습니다.



### Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models (https://arxiv.org/abs/2410.18122)
- **What's New**: 이 논문은 misinformation 모델의 out-of-distribution generalisation 능력을 평가하기 위한 benchmark dataset인 misinfo-general을 소개합니다. 기존의 잘 알려진 데이터셋들은 training과 inference 데이터 분포 간의 변화에 적합한 generalisation 능력이 부족한 문제가 있습니다.

- **Technical Details**: 이 논문에서는 generalisation을 위해 6가지 축(시간(time), 사건(event), 주제(topic), 출처(publisher), 정치적 편향(pollical bias), misinformation 유형(misinformation type))을 설정하고 평가 절차를 설계하였습니다. misinfo-general은 publisher 레벨의 메타데이터 주석을 통해 기사 수와 출처 다양성 및 라벨 품질의 균형을 맞춘 novel benchmark dataset을 제공합니다.

- **Performance Highlights**: 기본 모델을 통하여 misinfo-general 데이터셋의 성능을 심층적으로 분석하였고, 다양한 출처(class of publishers)로의 generalisation이 특히 도전적임을 분명히 밝혔습니다. 모델은 덜 빈번한 출처 또는 다른 정치적 편향에 대한 분류가 더 나쁨을 보였습니다. 데이터셋의 규모와 다양성이 generalisation 능력에 이점을 줄 수 있다는 초반 증거도 발견하였습니다.



### Improving Embedding Accuracy for Document Retrieval Using Entity Relationship Maps and Model-Aware Contrastive Sampling (https://arxiv.org/abs/2410.18105)
Comments:
          10 Pages, 9 Figures

- **What's New**: 이번 논문에서는 APEX-Embedding-7B라는 70억 매개변수의 디코더 전용 텍스트 피처 추출 모델을 소개합니다. 이 모델은 특히 Document Retrieval-Augmented Generation (RAG) 작업을 위해 설계되었습니다. 본 모델은 두 가지 훈련 기술을 사용하여 사실적 초점을 향상시키고 이러한 방식으로 RAG 시스템의 성능을 개선합니다.

- **Technical Details**: APEX-Embedding-7B은 (1) Structured Entity Relationship Maps를 이용한 사전 수렴 중단 세부 조정(pre-convergence interrupted fine-tuning)과 (2) Model-Aware Contrastive Sampling을 사용하는데, 이는 모델의 주의 편향을 사실적 콘텐츠로 전환하여 성능을 향상시킵니다. 이 메소드는 특히 부동산과 같이 방대한 데이터셋을 다루는 산업에 적합합니다.

- **Performance Highlights**: 본 모델은 평가에서 plain text 쿼리/document 쌍의 검색 정확도를 90.86%로 향상시키며(다음 우수 모델에 비해 6.26% 증가) 훈련 데이터 입력의 컨텍스트 크기를 평균 37.71% 감소시킵니다. APEX-Embedding-7B은 긴 문맥의 문서 검색 작업에서 새로운 최첨단 기준을 수립합니다.



### RingGesture: A Ring-Based Mid-Air Gesture Typing System Powered by a Deep-Learning Word Prediction Framework (https://arxiv.org/abs/2410.18100)
- **What's New**: 이 논문은 경량 증강 현실(AR) 안경을 위한 새로운 텍스트 입력 시스템인 RingGesture를 제안합니다. 이는 제스처 경로의 시작과 끝을 표시하는 전극과 손 추적을 위한 관성 측정 장치(IMU) 센서를 활용하여 공중 제스처 타이핑을 가능하게 합니다.

- **Technical Details**: RingGesture 시스템은 심층 학습 기반의 새로운 단어 예측 프레임워크인 Score Fusion을 통합하여 정확성과 입력 속도를 향상시킵니다. Score Fusion은 단어-제스처 디코딩 모델, 공간 철자 수정 모델, 경량 컨텍스트 언어 모델 등 세 가지 주요 구성 요소로 구성됩니다.

- **Performance Highlights**: RingGesture는 평균 27.3 단어/분(WPM)의 텍스트 입력 속도를 기록했으며, 최고 성능은 47.9 WPM입니다. Score Fusion 프레임워크는 전통적인 단어 예측 프레임워크인 Naive Correction보다 28.2% 더 나은 성능을 보여주었으며, RingGesture의 텍스트 입력 속도는 55.2% 향상되었습니다. 또한, 시스템 사용성 점수는 83점으로, 매우 우수한 사용성을 나타냅니다.



### Gesture2Text: A Generalizable Decoder for Word-Gesture Keyboards in XR Through Trajectory Coarse Discretization and Pre-training (https://arxiv.org/abs/2410.18099)
- **What's New**: 이 논문에서는 확장 현실(Extended Reality, XR) 환경에서의 직관적인 텍스트 입력을 위한 단어-제스처 키보드(Word-Gesture Keyboard, WGK) 시스템에 대한 새로운 신경 디코더(Neural Decoder)를 제안합니다. 새로운 접근법은 대규모의 간략히 이산화된(word-gesture trajectories) 데이터를 미리 학습하여 다양한 환경에서 일반화할 수 있는 성능을 보여줍니다.

- **Technical Details**: 합성곱 신경망(convolutional neural networks)과 사전 학습(pre-training) 기법을 통해 기존의 템플릿 일치(template-matching) 디코더인 SHARK^2의 한계를 극복하려고 하며, 입력 경로(trajectory)를 인코딩하는 구조적 표현 E(g)를 사용합니다. 이 시스템은 현실 증강(AR)과 가상 현실(VR)에서의 WGK 시스템에 적용 가능하며, 단순한 설치와 함께 높은 디코딩 정확도를 제공합니다.

- **Performance Highlights**: 이 새로운 사전 학습된 신경 디코더는 평균적으로 90.4%의 Top-4 정확도를 달성하였고, SHARK^2보다 37.2% 높은 성능을 보여주며, 전통적인 신경 디코더보다도 7.4% 향상된 결과를 보입니다. 또한, 이 디코더는 저용량(4 MB)으로 실시간(97ms) 처리 속도를 가지며 정확도를 희생하지 않았습니다.



### $M^3EL$: A Multi-task Multi-topic Dataset for Multi-modal Entity Linking (https://arxiv.org/abs/2410.18096)
- **What's New**: 본 논문에서는 Multi-modal Entity Linking (MEL) 를 위한 대규모 데이터셋 M^3EL을 제안한다. 이 데이터셋은 79,625개의 인스턴스를 포함하고 있으며, 5개의 다양한 주제와 9가지 다중 모달(Multi-modal) 작업을 포괄하고 있다. 기존 MEL 데이터셋의 한계를 극복하기 위해 데이터셋 구축 파이프라인을 수립하였다.

- **Technical Details**: M^3EL 데이터셋은 318,500개의 이미지를 포함하고 있으며, Text-Text, Text-Image, Image-Text 등 다양한 모달 작업을 지원한다. CLIP 모델을 활용하여 MODE(모달-증강) 훈련 전략을 제안하며, M^3EL 데이터셋을 기반으로 CLIP_{ND} 모델을 미세 조정(fine-tune)하였다.

- **Performance Highlights**: 실험 결과 기존 모델의 정확도가 49.4%에서 75.8%까지 다양하게 나타난 반면, M^3EL 데이터셋으로 학습한 CLIP_{ND} 모델은 다양한 작업에서 평균 9.3%에서 25%까지 성능이 향상되었다. 이 데이터셋은 MEL 알고리즘의 일반화 성능을 효과적으로 향상시킬 수 있는 우수한 품질의 사전 훈련 데이터셋으로 자리잡는다.



### Advancing Interpretability in Text Classification through Prototype Learning (https://arxiv.org/abs/2410.17546)
- **What's New**: ProtoLens는 텍스트 분류를 위한 새로운 프로토타입 기반 모델로, 서브-문장(sub-sentence) 수준의 해석 가능성을 제공하여 투명성이 중요한 응용 프로그램에 적합합니다.

- **Technical Details**: ProtoLens는 Prototype-aware Span Extraction 모듈을 사용하여 학습한 프로토타입과 관련된 텍스트 스팬을 식별하고, Prototype Alignment 메커니즘을 통해 훈련 과정 전반에 걸쳐 프로토타입이 의미론적으로 중요하도록 보장합니다.

- **Performance Highlights**: ProtoLens는 여러 텍스트 분류 벤치마크에서 프로토타입 기반 및 비해석적(baseline) 모델들보다 뛰어난 성능을 나타냅니다.



New uploads on arXiv(cs.IR)

### End-to-end Training for Recommendation with Language-based User Profiles (https://arxiv.org/abs/2410.18870)
- **What's New**: 이 논문에서는 사용자 프로필 생성을 위한 새로운 접근 방식을 제안합니다. LangPTune은 사용자 프로필을 효과적으로 생성하기 위해 대규모 언어 모델(LLM)을 end-to-end 방식으로 훈련하는 첫 번째 방법입니다.

- **Technical Details**: LangPTune은 Reinforcement Learning for System Optimization (RLSO)과 Contrastive Learning (CL)을 결합하여 LLM과 추천 시스템을 최적화합니다. 사용자와 상호작용한 아이템의 목록을 기반으로 사용자 프로필을 생성하며, 이를 통해 추천 모델의 성능을 극대화합니다.

- **Performance Highlights**: LangPTune은 다양한 추천 접근 방식을 대상으로 한 여러 공공 데이터셋에서 기존의 프로필 기반 방법들을 크게 초월하는 성능을 보였습니다. 또한, 전통적인 추천 모델과 유사한 수준의 성능을 발휘하여 해석 가능성과 투명성을 제공하는 강력하고 해석 가능한 대안을 제공합니다.



### Smart ETL and LLM-based contents classification: the European Smart Tourism Tools Observatory experienc (https://arxiv.org/abs/2410.18641)
- **What's New**: 이번 연구는 유럽 스마트 관광 도구(STTs) 관측소의 콘텐츠 업데이트를 개선하기 위해 STTs를 통합하고 분류하는 데 중점을 둡니다.

- **Technical Details**: 이 연구는 PDF 카탈로그에서 STTs에 대한 정보를 추출하기 위해 PDF-scraping 기술을 사용합니다. 추출된 정보는 QR 코드, 이미지, 링크 및 텍스트 정보를 포함합니다. 중복 STTs는 제거되고, 남은 항목은 Large Language Models (LLMs)을 사용하여 텍스트 정보를 기반으로 분류됩니다. 최종적으로, 데이터는 Dublin Core 메타데이터 구조에 맞게 변환됩니다.

- **Performance Highlights**: Smart ETL(procedure) 과정은 PDF-scraping 기술과 LLMs를 결합하여 텍스트 콘텐츠 기반 분류를 수행하는 데 성공적이었습니다. 초기 결과는 LLMs가 텍스트 콘텐츠 기반 분류에서 효과적임을 보여줍니다.



### Probing Ranking LLMs: Mechanistic Interpretability in Information Retrieva (https://arxiv.org/abs/2410.18527)
Comments:
          9 pages

- **What's New**: 이 연구는 최신 Transformer 네트워크를 활용하여 Passage Reranking의 메커니즘을 깊이 분석하는 혁신적인 접근 방식을 제안합니다. 특히, LLM의 활동을 레이어별로 분석하여 인간이 설계한 피처와의 상관관계를 밝혀냈습니다.

- **Technical Details**: 연구는 Llama2-7b/13b 아키텍처를 기반으로 하며, 특히 Passage Reranking 작업에 최적화된 LoRa Fine-tuned 변형인 RankLlama를 사용합니다. Neural 평점에서 MLP 유닛의 활성화를 추출하고, 이를 바탕으로 회귀 분석을 통해 피처와의 상관관계를 분석했습니다.

- **Performance Highlights**: 연구 결과, RankLlama 모델에서 여러 MSLR 피처의 뚜렷한 표현이 발견되었으며, 동일한 피처가 고유의 질의를 접할 때에도 일관성을 유지한다고 밝혀졌습니다. 이를 통해 LLM의 내부 작동 메커니즘을 이해하고, 정보 검색 커뮤니티에 중요한 통찰을 제공할 수 있습니다.



### NexusIndex: Integrating Advanced Vector Indexing and Multi-Model Embeddings for Robust Fake News Detection (https://arxiv.org/abs/2410.18294)
Comments:
          9 pages, 3 figures

- **What's New**: NexusIndex라는 새로운 프레임워크와 모델을 제안하여, 고급 언어 모델, 혁신적인 FAISSNexusIndex 레이어 및 주의 메커니즘을 통합하여 가짜 뉴스 감지를 강화했습니다.

- **Technical Details**: NexusIndex는 고차원 임베딩을 통해 뉴스 기사를 효율적으로 인덱싱하고, 다중 모델 임베딩을 활용하여 텍스트 해석 및 분류 정확도를 높입니다. 특히, FAISSNexusIndex 레이어를 통해 실시간 감지 및 시스템의 확장성을 최적화합니다.

- **Performance Highlights**: NexusIndex는 다양한 데이터셋에서 기존 최첨단 방법들보다 효율성과 정확성에서 더 우수한 성능을 보여주는 실험 결과를 보였습니다.



### SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback (https://arxiv.org/abs/2410.18141)
- **What's New**: 이 논문에서는 RAG 시스템의 모든 모듈이 개별적으로 최적화되는 대신 공동 최적화되어야 한다고 주장합니다. 이를 위해 'SmartRAG'라는 새로운 파이프라인을 설계하고, 강화 학습(rl)을 통해 시스템을 공동 최적화합니다.

- **Technical Details**: SmartRAG는 정책 네트워크(policy network)와 리트리버(retriever)를 포함하며, 정책 네트워크는 1) 언제 검색할지 결정, 2) 검색기와 가장 적합한 쿼리 생성, 3) 최종 응답 생성의 세 가지 역할을 합니다. 시스템의 최적화는 환경 피드백을 기반으로 이루어집니다.

- **Performance Highlights**: SmartRAG는 다양한 데이터셋에서 실험 결과, 개별적으로 최적화된 모듈보다 우수한 성능을 보였으며, retrieval 비용을 최소화하면서도 정확한 응답을 생성하는 데 성공했습니다.



### Tethering Broken Themes: Aligning Neural Topic Models with Labels and Authors (https://arxiv.org/abs/2410.18140)
- **What's New**: 본 논문에서는 FANToM이라는 새로운 방법론을 제안하여, 레이블(label) 및 저자(author) 정보와의 정렬을 통해 신경 주제 모델(neural topic models, NTMs)의 해석 가능성을 향상시킵니다.

- **Technical Details**: FANToM은 레이블 및 저자 메타데이터를 기존의 NTMs에 통합하여 학습된 주제를 더욱 해석 가능하게 만듭니다. 이 방법은 레이블, 주제, 저자 간의 정렬을 학습함으로써 더 큰 표현력을 제공합니다. 또한, FANToM은 삽입된 메타데이터를 사용하여 저자들의 관심사 및 유사성을 파악합니다.

- **Performance Highlights**: 실험 결과, FANToM은 기존 모델에 비해 주제 품질 및 정렬을 개선하며, 학습된 주제의 품질을 향상시키는 데 성공하였습니다. FANToM은 저자, 단어, 주제가 공유하는 임베딩 공간을 학습하도록 도와주어 저자들의 관심사와 유사성에 대한 통찰을 제공합니다.



### Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models (https://arxiv.org/abs/2410.18122)
- **What's New**: 이 논문은 misinformation 모델의 out-of-distribution generalisation 능력을 평가하기 위한 benchmark dataset인 misinfo-general을 소개합니다. 기존의 잘 알려진 데이터셋들은 training과 inference 데이터 분포 간의 변화에 적합한 generalisation 능력이 부족한 문제가 있습니다.

- **Technical Details**: 이 논문에서는 generalisation을 위해 6가지 축(시간(time), 사건(event), 주제(topic), 출처(publisher), 정치적 편향(pollical bias), misinformation 유형(misinformation type))을 설정하고 평가 절차를 설계하였습니다. misinfo-general은 publisher 레벨의 메타데이터 주석을 통해 기사 수와 출처 다양성 및 라벨 품질의 균형을 맞춘 novel benchmark dataset을 제공합니다.

- **Performance Highlights**: 기본 모델을 통하여 misinfo-general 데이터셋의 성능을 심층적으로 분석하였고, 다양한 출처(class of publishers)로의 generalisation이 특히 도전적임을 분명히 밝혔습니다. 모델은 덜 빈번한 출처 또는 다른 정치적 편향에 대한 분류가 더 나쁨을 보였습니다. 데이터셋의 규모와 다양성이 generalisation 능력에 이점을 줄 수 있다는 초반 증거도 발견하였습니다.



### Data Efficiency for Large Recommendation Models (https://arxiv.org/abs/2410.18111)
- **What's New**: 이 논문은 대형 추천 모델(Large Recommendation Models, LRM)의 훈련 데이터를 최적화하는 실행 가능한 원칙과 고수준 프레임워크를 소개합니다. 특히, 구글의 광고 클릭률(CTR) 예측 모델에서 성공적으로 적용된 전략들이 논의됩니다.

- **Technical Details**: 논문에서는 데이터 수렴(data convergence) 개념을 정의하고, 이를 가속화할 수 있는 방법과 훈련 데이터 볼륨과 모델 크기 간의 최적 균형을 맞추는 방법을 상세히 설명합니다. 초기 훈련(initial training)과 지속적인 온라인 훈련(continuous online training) 단계에서 데이터 효율성을 향상시키기 위한 전략들이 다루어집니다.

- **Performance Highlights**: 연구 결과, 연속적인 증류(continuous distillation)를 통해 모델이 35% 더 적은 데이터로 수렴할 수 있음을 보여줍니다. 이는 데이터와 모델 크기 간의 적절한 균형을 통해 훈련 시간을 줄이고 시스템 효율성을 향상시킬 수 있다는 것을 의미합니다.



### Improving Embedding Accuracy for Document Retrieval Using Entity Relationship Maps and Model-Aware Contrastive Sampling (https://arxiv.org/abs/2410.18105)
Comments:
          10 Pages, 9 Figures

- **What's New**: 이번 논문에서는 APEX-Embedding-7B라는 70억 매개변수의 디코더 전용 텍스트 피처 추출 모델을 소개합니다. 이 모델은 특히 Document Retrieval-Augmented Generation (RAG) 작업을 위해 설계되었습니다. 본 모델은 두 가지 훈련 기술을 사용하여 사실적 초점을 향상시키고 이러한 방식으로 RAG 시스템의 성능을 개선합니다.

- **Technical Details**: APEX-Embedding-7B은 (1) Structured Entity Relationship Maps를 이용한 사전 수렴 중단 세부 조정(pre-convergence interrupted fine-tuning)과 (2) Model-Aware Contrastive Sampling을 사용하는데, 이는 모델의 주의 편향을 사실적 콘텐츠로 전환하여 성능을 향상시킵니다. 이 메소드는 특히 부동산과 같이 방대한 데이터셋을 다루는 산업에 적합합니다.

- **Performance Highlights**: 본 모델은 평가에서 plain text 쿼리/document 쌍의 검색 정확도를 90.86%로 향상시키며(다음 우수 모델에 비해 6.26% 증가) 훈련 데이터 입력의 컨텍스트 크기를 평균 37.71% 감소시킵니다. APEX-Embedding-7B은 긴 문맥의 문서 검색 작업에서 새로운 최첨단 기준을 수립합니다.



### RRADistill: Distilling LLMs' Passage Ranking Ability for Document Re-Ranking of Long-Tail Queries in a Search Engin (https://arxiv.org/abs/2410.18097)
Comments:
          Accepted to EMNLP 2024 Industry Track. First two authors contributed equally

- **What's New**: 이 연구는 RRADistill: Re-Ranking Ability Distillation이라는 새로운 방법론을 제안하며, 문서 및 쿼리 간의 의미적 관계를 이해하는 대형언어모델(LLMs)의 효과성을 활용하는 동시에, 보다 작고 효율적인 모델(sLLMs)의 실용적인 훈련 기법을 개발합니다.

- **Technical Details**: 우리의 접근 방식은 크게 두 가지 단계로 구성됩니다. 첫째, LLM을 사용하여 랭킹 레이블을 생성하는 단계이며, 둘째, 이 레이블을 통해 SLM 랭커를 훈련하는 것입니다. 우리는 Pre-rank 기법을 도입하여 효율적인 레이블 생성 파이프라인을 만들었으며, Term Control Layer와 같은 새로운 기술을 통해 쿼리와 문서 간의 용어 매칭 신호를 효과적으로 활용합니다.

- **Performance Highlights**: 우리의 방법론은 한국 기반 검색 플랫폼에서 긴 쿼리에 대해 A/B 테스트를 통해 효과성을 입증했습니다. 실험 결과, RRA-BERT와 RRA-GPT 두 가지 랭커 모두 LLM의 관계적 랭킹을 흉내내도록 훈련되어, 긴 꼬리 쿼리 검색 품질을 획기적으로 개선하였습니다.



### $M^3EL$: A Multi-task Multi-topic Dataset for Multi-modal Entity Linking (https://arxiv.org/abs/2410.18096)
- **What's New**: 본 논문에서는 Multi-modal Entity Linking (MEL) 를 위한 대규모 데이터셋 M^3EL을 제안한다. 이 데이터셋은 79,625개의 인스턴스를 포함하고 있으며, 5개의 다양한 주제와 9가지 다중 모달(Multi-modal) 작업을 포괄하고 있다. 기존 MEL 데이터셋의 한계를 극복하기 위해 데이터셋 구축 파이프라인을 수립하였다.

- **Technical Details**: M^3EL 데이터셋은 318,500개의 이미지를 포함하고 있으며, Text-Text, Text-Image, Image-Text 등 다양한 모달 작업을 지원한다. CLIP 모델을 활용하여 MODE(모달-증강) 훈련 전략을 제안하며, M^3EL 데이터셋을 기반으로 CLIP_{ND} 모델을 미세 조정(fine-tune)하였다.

- **Performance Highlights**: 실험 결과 기존 모델의 정확도가 49.4%에서 75.8%까지 다양하게 나타난 반면, M^3EL 데이터셋으로 학습한 CLIP_{ND} 모델은 다양한 작업에서 평균 9.3%에서 25%까지 성능이 향상되었다. 이 데이터셋은 MEL 알고리즘의 일반화 성능을 효과적으로 향상시킬 수 있는 우수한 품질의 사전 훈련 데이터셋으로 자리잡는다.



### Liver Cancer Knowledge Graph Construction based on dynamic entity replacement and masking strategies RoBERTa-BiLSTM-CRF mod (https://arxiv.org/abs/2410.18090)
- **What's New**: 이번 연구는 한국에서 다섯 번째로 흔한 악성 종양이자 두 번째로 치명적인 간암의 진단과 치료를 위한 지식 그래프(knowledge graph)-기반 시스템을 제안합니다. 기존 데이터 소스와 실제 전자 의료 기록 간의 불일치 문제를 해결하며, 간암에 특화된 첫 번째 지식 그래프를 구축함으로써 의사가 진단하는 데 겪는 어려움을 경감할 수 있을 것으로 기대됩니다.

- **Technical Details**: 지식 그래프는 여섯 가지 단계로 거쳐 구축되었습니다: 개념 계층 설계(conceptual layer design), 데이터 전처리(data preprocessing), 개체 인식(entity identification), 개체 정규화(entity normalization), 지식 융합(knowledge fusion) 및 그래프 시각화(graph visualization). 이 과정에서는 새로운 동적 개체 변경 및 마스킹 전략(Dynamic Entity Replacement and Masking Strategy, DERM)이 도입되어 명명된 개체 인식(named entity recognition)을 개선하였습니다.

- **Performance Highlights**: 간암에 대한 지식 그래프는 1495개의 개체로 구성되어 있으며, 명명된 개체 인식 모델의 정확도는 93.23%, 재현율은 94.69%, F1 점수는 93.96%에 달합니다.



### CUPID: A Real-Time Session-Based Reciprocal Recommendation System for a One-on-One Social Discovery Platform (https://arxiv.org/abs/2410.18087)
Comments:
          The 2nd International Workshop on User Understanding from Big Data Workshop (DMU2 2024)

- **What's New**: CUPID는 실시간 1:1 소셜 탐색 플랫폼을 위한 새로운 세션 기반의 상호 추천 시스템을 소개합니다.

- **Technical Details**: CUPID는 사용자 세션 모델링과 실시간 매칭 프로세스를 분리하여 추론 시간을 줄이는 비동기 세션 모델링 접근 방식을 사용하며, 두 단계의 훈련 전략을 통해 임베딩 레이어와 예측 레이어의 훈련을 분리하여 계산 비용을 절감합니다. 이 시스템은 Azar 플랫폼의 대규모 실험 데이터에서 검증되었습니다.

- **Performance Highlights**: CUPID는 비동기 시스템에 비해 응답 지연을 76% 이상 줄이고 사용자 참여도를 현저하게 개선하여 실제 추천 성능을 향상시킵니다. 따뜻한 시작 사용자에 대해서는 평균 채팅 시간을 6.8% 증가시키고, 차가운 시작 사용자에서는 5.9% 증가시킵니다.



### Attention-based Citywide Electric Vehicle Charging Demand Prediction Approach Considering Urban Region and Dynamic Influences (https://arxiv.org/abs/2410.18766)
- **What's New**: 본 연구에서는 도심 내 전기차 충전 수요 예측을 위해 주목 기반 이종 다변량 데이터 융합 접근법(Attention-based Heterogeneous Multivariate Data Fusion, AHMDF)을 제안합니다. 이 방법은 지리 기반 클러스터링 하이퍼그래프 및 다변량 게이트 트랜스포머(Multivariate Gated Transformer)를 통합하여 정적 및 동적 영향을 모두 고려합니다.

- **Technical Details**: AHMDF는 1) 유사한 특성을 지닌 지역을 클러스터링하고 주의 깊은 하이퍼그래프를 통해 유사한 패턴을 학습하며, 2) 이웃 지역 간 정보 전파를 위한 그래프 주의 메커니즘을 활용하고, 3) 시계열 외부 변수를 융합하며 동적 특징을 효율적으로 캡처하기 위해 다변량 게이트 트랜스포머를 사용합니다.

- **Performance Highlights**: 상하이 충전 수요 예측 데이터셋을 통해 제안된 AHMDF 방법의 우수한 성능을 입증하였으며, 여러 시공간적 영향을 고려한 충전 수요 예측의 정확도를 향상시켰습니다.



### Little Giants: Synthesizing High-Quality Embedding Data at Sca (https://arxiv.org/abs/2410.18634)
- **What's New**: 본 논문은 SPEED라는 프레임워크를 도입하여 오픈소스의 소규모 모델(8B)을 정렬하여 대규모 합성 임베딩 데이터 생성을 효율적으로 이루어질 수 있게 한다. 기존의 GPT-4와 같은 비공식 모델에 의존하는 대신, 저비용으로 고품질의 데이터를 제공할 수 있는 방안을 제시하다.

- **Technical Details**: SPEED 프레임워크는 (1) GPT-4를 사용해 다채로운 태스크 설명 생성, (2) 주니어 발생 모델을 통해 초기 데이터 생성, (3) 시니어 발생 모델 수립을 위한 선호 최적화 과정 및 (4) 데이터 수정기를 통한 자기 개선 과정을 포함한다. 이 모델을 통해 생성된 데이터는 고급 임베딩 기능을 활용할 수 있게 되어 성능 향상을 이루도록 설계되었다.

- **Performance Highlights**: 실험 결과, SPEED는 E5_mistral와 비교할 때 10분의 1 미만의 GPT API 호출만으로도 우수한 성능을 보여주며, 임베딩 모델의 성능이 합성 임베딩 데이터 크기와 로그-선형 관계에 있음을 발견하였다.



### Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieva (https://arxiv.org/abs/2410.18385)
Comments:
          Accepted for publication at EMNLP 2024 Main Conference

- **What's New**: 이번 연구에서는 새로운 알고리즘인 Universal Document Linking (UDL)을 제안하여 제로샷(Zero-shot) 정보 검색(Information Retrieval, IR)의 도전 과제를 극복하고자 하였습니다. UDL은 유사한 문서를 연결하여 다양한 특성을 가진 여러 데이터셋에서 합성 쿼리 생성을 향상시키는 데 초점을 맞춥니다.

- **Technical Details**: UDL은 엔트로피(Entropy)를 활용해 유사성 모델을 선택하고 명명实体 인식(Named Entity Recognition, NER)을 통해 문서의 링크 결정을 내립니다. 알고리즘은 TF-IDF 및 사전 훈련된 언어 모델(Pre-trained Language Models)을 사용하여 문서 임베딩(Document Embeddings)을 생성하며, 유사성이 높은 문서들 간의 링크를 수립하기 위해 코사인 유사성(Cosine Similarity)을 계산합니다.

- **Performance Highlights**: UDL은 다양한 데이터셋과 IR 모델을 통해 검증되었으며, 제로샷 경우에서 기존의 최첨단 방법들을 초월하는 효과를 입증하였습니다. 이로 인해 UDL의 범용성과 유연성이 강화되었습니다.



### Context-Augmented Code Generation Using Programming Knowledge Graphs (https://arxiv.org/abs/2410.18251)
Comments:
          20 pages, Conference

- **What's New**: 이 논문은 Programming Knowledge Graph (PKG)를 활용하여 코드 검색 및 생성을 위한 새로운 프레임워크를 제안합니다. PKG는 외부 지식을 효과적으로 검색하고 통합하는 데 필수적인 역할을 합니다.

- **Technical Details**: 제안된 방법은 코드 조각의 관련성을 심층적으로 분석하여 검색 정밀도를 높이는 tree-pruning 기법을 포함합니다. 또한, re-ranking 메커니즘을 통해 비 관련성 솔루션을 선택적으로 통합하여 생성 중 발생할 수 있는 hallucination을 줄이는 데 기여합니다.

- **Performance Highlights**: HumanEval 및 MBPP 벤치마크에서 검증 결과, 제안된 방법은 pass@1 정확도를 최대 20%까지 향상시키고 MBPP에서 기존의 최첨단 모델보다 최대 34% 우수한 성능을 보였습니다.



New uploads on arXiv(cs.CV)

### PixelGaussian: Generalizable 3D Gaussian Reconstruction from Arbitrary Views (https://arxiv.org/abs/2410.18979)
Comments:
          Code is available at: this https URL

- **What's New**: 새로운 PixelGaussian 프레임워크를 제안하여 임의의 시점에서 일반화 가능한 3D Gaussian 재구성을 효율적으로 학습할 수 있도록 합니다.

- **Technical Details**: PixelGaussian은 지오메트릭 복잡도에 따라 Gaussian 분포와 개수를 동적으로 조정하여 효율적인 표현과 재구성 품질 향상을 달성합니다. 이를 위해 Cascade Gaussian Adapter(CGA)와 transformer 기반의 Iterative Gaussian Refiner(IGR) 모듈을 사용하여 Gaussian 표현을 정제합니다.

- **Performance Highlights**: PixelGaussian은 ACID 및 RealEstate10K 데이터셋에서 기존 방법들을 초월하는 성능을 보이며, 더 많은 입력 뷰에서도 일관된 성능을 발휘합니다.



### Framer: Interactive Frame Interpolation (https://arxiv.org/abs/2410.18978)
Comments:
          Project page: this https URL

- **What's New**: Framer는 사용자 창의성에 따라 두 이미지 간에 부드러운 전환 프레임을 생성하는 인터랙티브 프레임 보간(interpolation) 방법을 제안합니다. 사용자는 선택된 키포인트의 경로를 조정하여 전환 과정을 사용자 맞춤화할 수 있습니다.

- **Technical Details**: Framer는 대규모 사전 학습된 이미지-비디오 확산 모델을 미세 조정하여 비디오 프레임 보간을 수행합니다. 키포인트 기반 상호작용을 통합하여 프레임 간의 명확한 상관관계를 수립하며, 다양한 객체 모양과 스타일이 달라지는 도전적인 경우에서도 잘 작동합니다. 또한, 자동으로 키포인트 경로를 추정하고 미세 조정하는 '자율 비행(autopilot)' 모드를 도입하여 사용자 입력을 최소화합니다.

- **Performance Highlights**: Framer는 이미지 변형, 타임랩스 영상 생성 및 만화 보간과 같은 다양한 응용 프로그램에서 훌륭한 성능을 보여줍니다. 복잡한 움직임과 중요한 외관 변화가 있는 경우에서도 기존 방법들보다 더 부드럽고 시각적으로 매력적인 전환을 생성합니다.



### MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms (https://arxiv.org/abs/2410.18977)
Comments:
          MotionCLR v1 technical report

- **What's New**: 이 연구는 인체 움직임 생성을 위한 상호작용 편집 문제를 탐구합니다. MotionCLR로 명명된 새로운 어텐션 기반 모션 확산 모델을 제안하며, 이는 텍스트와 모션 간의 세부적인 대응 관계를 명확히 모델링합니다.

- **Technical Details**: MotionCLR 모델은 자기-어텐션(self-attention) 및 교차-어텐션(cross-attention) 메커니즘을 사용하여 모달 간 상호작용을 모델링하며, 이를 통해 프레임 간 시퀀스 유사성을 측정하고, 특정 동작의 단어 시퀀스와 타임스텝 간의 세부적인 대응 관계를 찾아냅니다.

- **Performance Highlights**: 실험 결과, MotionCLR은 우수한 생성 및 편집 능력을 보여주며, 설명력이 뛰어난 메커니즘을 통해 다양한 동작 편집 작업을 수행할 수 있습니다.



### CAMEL-Bench: A Comprehensive Arabic LMM Benchmark (https://arxiv.org/abs/2410.18976)
Comments:
          10 pages, 5 figures, NAACL

- **What's New**: CAMEL-Bench라는 포괄적인 아랍어 대규모 멀티모달 모델 평가 벤치마크가 개발되었습니다. 이는 4억 명 이상의 아랍어 사용자를 위한 다양한 도메인과 하위 도메인으로 구성되어 있습니다.

- **Technical Details**: CAMEL-Bench는 8개의 다양한 도메인과 38개의 하위 도메인으로 구성되어 있으며, 총 29,036개의 질문이 포함되어 있습니다. 데이터는 GPT-4o를 이용해 아랍어로 번역되거나 수작업으로 수집되었습니다.

- **Performance Highlights**: GPT-4o 시리즈는 다양한 멀티모달 이해 작업에서 높은 성능을 보였으나, 아랍어 멀티모달 데이터 처리에서 개선이 필요함을 보여주었습니다.



### Unbounded: A Generative Infinite Game of Character Life Simulation (https://arxiv.org/abs/2410.18975)
Comments:
          18 pages; Project page: this https URL

- **What's New**: 전통적인 비디오 게임의 한계를 초월한 생성적 무한 게임(generative infinite game) 개념을 소개하고, 이를 통해 생성 AI를 활용한 새로운 게임인 Unbounded를 개발하였습니다. Unbounded는 플레이어가 자율 가상 캐릭터와 상호작용하고, 개방형 메커니즘을 통해 이야기를 생성하는 게임입니다.

- **Technical Details**: Unbounded는 다음과 같은 기술적 혁신을 포함합니다: (1) 게임 메커니즘, 내러티브 및 캐릭터 상호작용을 실시간으로 생성하는 특수화된 LLM(large language model) 및 (2) 다양한 환경에서 캐릭터를 일관되게 시각적으로 생성할 수 있는 새로운 동적 지역 이미지 프롬프트 어댑터(Regional Image Prompt Adapter)입니다. 이 기술들은 생동감 있는 캐릭터 시뮬레이션과 화면 간의 일관성 높은 시각적 생성이 가능합니다.

- **Performance Highlights**: 전통적인 접근 방식과 비교했을 때, 캐릭터 삶의 시뮬레이션, 사용자 지시 따르기, 내러티브 일관성 및 캐릭터와 환경의 시각적 일관성에서 유의미한 개선을 보여주었습니다. 특히, 우리가 개발한 distilled LLM은 대형 LLM과 비교했을 때도 상호작용 속도에서 비슷한 성능을 보여 줍니다.



### 3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation (https://arxiv.org/abs/2410.18974)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 2D 네트워크 아키텍처의 한계를 극복하기 위해 프리트레인된 이미지 확산 모델에 3D 기하 인식을 주입하는 플러그인 모듈인 3D-Adapter를 소개합니다. 이 방법은 3D 피드백 증강(3D feedback augmentation)을 통해 다중 뷰 기능을 일관된 3D 표현으로 변환하여 기존 모델의 기하학적 품질을 크게 향상시킴을 입증합니다.

- **Technical Details**: 3D-Adapter는 프리트레인된 이미지 확산 모델의 샘플링 루프 내에서 각 복원 단계가 이루어질 때 중간 다중 뷰 기능을 복호화하고 이를 이용해 기하학적으로 일관된 3D 표현을 생성합니다. 주요 기술로는 Gaussian splatting을 기반으로 한 피드포워드 버전과 신경장(field) 및 메쉬를 사용하는 훈련이 필요 없는 다목적 버전이 포함되어 있습니다.

- **Performance Highlights**: 3D-Adapter는 Instant3D, Zero123++와 같은 텍스트-멀티뷰 모델의 기하학적 품질을 크게 향상시키며, 텍스트-이미지에 대한 기본적인 Stable Diffusion을 사용할 때도 고품질 3D 생성을 가능하게 합니다. 또한 텍스트-3D, 이미지-3D, 텍스트-텍스처 및 텍스트-아바타 작업에서도 뛰어난 결과를 보였으며, 3D-Adapter의 넓은 적용 가능성을 보여줍니다.



### Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms (https://arxiv.org/abs/2410.18967)
- **What's New**: Ferret-UI 2가 다양한 플랫폼에서의 사용자 인터페이스(UI) 이해를 위한 혁신적인 멀티모달 대규모 언어 모델(MLLM)로 소개됩니다. 이 모델은 iPhone, Android, iPad, Webpage, AppleTV 등 여러 플랫폼을 지원하며, 여러 혁신적인 기능을 통해 사용자 중심의 복잡한 상호작용을 수행합니다.

- **Technical Details**: Ferret-UI 2는 세 가지 주요 혁신을 도입합니다: (1) 다양한 플랫폼 지원, (2) 적응형 스케일링을 통한 고해상도 인식, (3) GPT-4o를 활용한 시각적 프롬프트 기반 고품질 훈련 데이터 생성. 이러한 발전은 UI 요소에 대한 더 정확한 인식을 가능하게 하여 사용자의 의도에 맞춘 상호작용을 지원합니다.

- **Performance Highlights**: Ferret-UI 2는 여러 벤치마크와 실험 데이터셋에서 뛰어난 성능을 보여 주며, 9개 하위 작업과 5개의 플랫폼을 포함한 사용자 중심의 고급 작업에서 Ferret-UI보다 우수한 결과를 나타냅니다. GUIDE 및 GUI-World와 같은 최신 벤치마크에서도 강력한 성과를 기록했습니다.



### Where Am I and What Will I See: An Auto-Regressive Model for Spatial Localization and View Prediction (https://arxiv.org/abs/2410.18962)
- **What's New**: Generative Spatial Transformer (GST) 모델을 통해 공간적인 위치 추정(spatial localization)과 시각적 예측(view prediction)을 동시에 해결하는 혁신적인 접근을 제안합니다. 기존 모델들이 이 두 문제를 분리하여 처리했던 것과 달리, GST는 이들을 통합하여 훈련하는 방식을 도입했습니다.

- **Technical Details**: GST는 Plücker coordinates를 활용하여 카메라를 맵으로 변환하고, 이를 토크나이징(tokenizing)하여 단일 이미지로부터 카메라 포즈를 추정하고 새로운 시각을 예측할 수 있도록 합니다. 이 모델은 두 개의 후방 확률 분포를 포괄하는 조인트 분포(joint distribution)를 모델링하여 최적화 과정의 안정성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, GST는 단일 색다른 시각을 합성하는 데 있어 최신 성능을 기록하고, 두 프레임 사이의 상대 카메라 포즈 추정에 대한 정밀도를 높이며, 기존의 기준을 뛰어넘은 성능을 발휘합니다.



### Large Spatial Model: End-to-end Unposed Images to Semantic 3D (https://arxiv.org/abs/2410.18956)
Comments:
          Project Website: this https URL

- **What's New**: 이번 연구에서는 기존의 복잡한 3D 구조 재구성을 단일 피드포워드 작업으로 처리할 수 있는 대형 공간 모델(Large Spatial Model, LSM)을 제안합니다. 이는 비정렬 RGB 이미지에서 직접적으로 의미론적 방사장(semantic radiance fields)을 생성하여 기하학, 외관 및 의미론을 동시에 추정하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: LSM은 Transformer 기반 아키텍처를 사용하여 픽셀 정렬(point-aligned)된 점 맵(pixel-aligned point maps)으로부터 전역 기하학(global geometry)을 통합합니다. 또한, 다중 스케일 융합(multi-scale fusion)과 지역 맥락 집합(local context aggregation)을 도입하여 공간 속성 회귀(spatial attribute regression)의 정확성을 강화하며, 3D 일관성(3D-consistent) 의미론적 특성 분야(semantic feature field)에 사전 훈련된 2D 언어 기반 분할(segmentation) 모델을 통합합니다. 이 효율적인 디코더는 의미론적 비등방성 가우시안(semantic anisotropic Gaussians)을 매개변수화하여 감독된(end-to-end) 학습을 용이하게 합니다.

- **Performance Highlights**: LSM은 비정렬 이미지에서 직접적으로 여러 3D 시각 작업을 통합하여 실시간 3D 의미론적 재구성을 달성합니다. 다양한 실험을 통해, LSM은 기존의 여러 방법 없이도 높은 효율을 발휘하며,  단일 GPU에서 실시간으로 복잡한 3D 작업을 수행하는 능력을 입증합니다.



### Sort-free Gaussian Splatting via Weighted Sum Rendering (https://arxiv.org/abs/2410.18931)
- **What's New**: 이 논문에서는 Weighted Sum Rendering(가중 합 렌더링) 방법을 제안하여, 3D Gaussian Splatting(3DGS)의 성능을 향상시키고 고유의 'popping' 아티팩트를 제거합니다. 이 새로운 접근 방식은 비순서적(alpha non-commutative) 혼합을 필요로 하지 않으므로 구현이 단순해집니다.

- **Technical Details**: Weighted Sum Rendering(GS-WSR) 방법은 각 Gaussian의 깊이를 사용하여 각 Gaussian splat에 대한 가중치를 계산하고, 이를 이용해 정렬 없이 간단한 합산 연산으로 이미지를 렌더링합니다. 추가적으로, 뷰-종속 불투명도(view-dependent opacity)가 이미지 품질을 크게 개선했습니다.

- **Performance Highlights**: 이 방법은 자원 제약이 있는 모바일 장치에서 평균적으로 1.23배 빠른 렌더링 속도를 달성했습니다. 실험 결과는 새로운 미분 가능 렌더링에 최적화된 일반화된 Gaussian splatting 공식이 경쟁력 있는 이미지 품질을 제공함을 보여주었습니다.



### SegLLM: Multi-round Reasoning Segmentation (https://arxiv.org/abs/2410.18923)
Comments:
          22 pages, 10 figures, 11 tables

- **What's New**: SegLLM은 대화형 메모리를 통해 시각 및 텍스트 출력의 이전 세분화 결과를 입력 스트림에 통합하여 복잡한 사용자 의도를 이해하고, 이미 지정된 엔티티(대상)의 위치, 상호작용 및 계층적 관계를 기반으로 개체를 세분화할 수 있는 새로운 다중 상호작용 세분화 모델입니다.

- **Technical Details**: SegLLM은 Mask-Encoding 및 Reference Mask-Decoding 기법을 통해 LLM이 세분화 마스크 출력을 인식하고, 마스크 디코더가 과거 대화 맥락을 인식할 수 있도록 설계되었습니다. 이를 통해 세분화 모델이 사용자 의도에 맞춘 복잡한 쿼리 처리 능력을 가질 수 있게 됩니다.

- **Performance Highlights**: MRSeg 벤치마크에서 SegLLM은 기존 모델들을 20% 이상 초과하는 성능을 보였고, 단일 라운드 REFCOCO 벤치마크에서도 시멘테이션 정확도가 5.5% 상승했습니다. 제안된 방법은 다양한 질문 템플릿에 대한 강인성도 향상되어 9.6%의 성능을 향상시켰습니다.



### Diff-Instruct++: Training One-step Text-to-image Generator Model to Align with Human Preferences (https://arxiv.org/abs/2410.18881)
- **What's New**: 이번 연구에서는 human preference(인간 선호)에 맞춘 one-step text-to-image generator 모델의 정렬 문제를 처음으로 다룹니다. 이를 위해 human feedback(인간 피드백)을 활용하는 reinforcement learning(RLHF)에서 영감을 받아 인간 보상 함수를 최대화하는 접근 방식을 채택합니다.

- **Technical Details**: Diff-Instruct++(DI++)라는 새로운 방법을 제안하며, 이는 이미지 데이터가 필요하지 않은 인체 선호 정렬 기법입니다. DI++는 기술적 도전 과제를 극복하고 효과적인 손실 함수 개발에 기여하며, 이 모델의 수렴 속도가 빠릅니다.

- **Performance Highlights**: DiT 기반의 one-step text-to-image 모델은 COCO 검증 프롬프트 데이터셋에서 Aesthetic Score(미적 점수) 6.19, Image Reward 1.24, Human preference Score (HPSv2.0) 28.48을 달성하여 기존의 오픈소스 모델을 초초과한 성과를 보였습니다.



### Multi-Class Abnormality Classification in Video Capsule Endoscopy Using Deep Learning (https://arxiv.org/abs/2410.18879)
- **What's New**: 팀 Seq2Cure의 연구는 Capsule Vision 2024 Challenge에서 다중 이상 징후 분류를 위해 CNN과 transformer 아키텍처를 결합한 모델을 적용했습니다.

- **Technical Details**: dataset은 50,000개 이상의 비디오 캡슐 내시경 프레임으로 구성되었으며, 10개 이상 징후 클래스에 레이블이 붙었습니다. CNN과 transformer 모델을 통합하여 이상 징후 분류의 정확도를 크게 향상시켰습니다. 구현된 데이터 증강 기법에는 이미지 크기 조정, 회전, 색상 변화 등이 포함됩니다.

- **Performance Highlights**: 모델은 검증 세트에서 86.34%의 균형 잡힌 정확도와 0.9908의 평균 AUC-ROC 점수를 기록하였으며, 복잡한 이상 징후 분류에서 상당한 개선을 보였습니다.



### Probabilistic Language-Image Pre-Training (https://arxiv.org/abs/2410.18857)
Comments:
          Code: this https URL 23 pages, 5.7 MB

- **What's New**: 이 논문에서는 이미지와 텍스트 쌍을 결합하는 Probabilistic Language-Image Pre-training (ProLIP)이라는 새로운 확률적 비전-언어 모델(VLM)을 소개합니다. ProLIP는 기존 VLM들이 가정하는 1:1 대응 관계 대신, 다대다(many-to-many) 관계를 모델링하여 더 나은 일치성과 해석 가능성을 제공합니다.

- **Technical Details**: ProLIP는 'uncertainty token'을 사용해 불확실성을 추정하며, 이는 추가적인 매개변수 없이도 이루어집니다. 최신 기술인 novel inclusion loss를 도입하여 이미지-텍스트 쌍 간의 분포적 포함 관계를 강화하고, 마스킹된 입력과 원본 입력 간의 관계도 더 효과적으로 모델링합니다.

- **Performance Highlights**: ProLIP는 74.6%의 ImageNet 제로 샷 정확도를 달성하며, 이는 동일한 샘플 수의 CLIP 모델이 보여준 73.5%보다 우수합니다. 이는 ProLIP가 적은 샘플에서도 75.8%로 성능이 향상된 것을 보여 주며, 불확실성 추정을 활용해 다운스트림 작업에서도 이점을 제공합니다.



### Multi-Scale Diffusion: Enhancing Spatial Layout in High-Resolution Panoramic Image Generation (https://arxiv.org/abs/2410.18830)
- **What's New**: 논문에서는 Multi-Scale Diffusion (MSD) 프레임워크를 소개하며, 이는 기존의 파노라마 이미지 생성을 위한 방법을 여러 해상도 수준으로 확장하는 모듈입니다. 이 새로운 접근 방식은 고해상도 파노라마의 공간 레이아웃 일관성을 높이고, 저해상도 이미지의 구조적 정보를 고해상도 출력으로 효과적으로 통합합니다.

- **Technical Details**: MSD 모듈은 저해상도 이미지에서 얻은 구조적 정보를 통해 고해상도 이미지를 생성하는 과정을 gradient descent 기술을 활용하여 구현합니다. 이를 통해 각 해상도 단계에서 MultiDiffusion 기법을 적절히 조합하며, 최종적으로 명확하고 구조적으로 일관된 파노라마 이미지를 생성합니다.

- **Performance Highlights**: MSD 모델은 FID, KID와 같은 정량적 및 질적 평가에서 기존 기준선 모델을 초과하여 다양성 및 현실감에서 월등한 성능을 보였습니다. 특히 다양한 텍스트 프롬프트에 대해 시각적으로 그리고 의미적으로 일관된 고해상도 파노라마 이미지를 생성하는 데 성공했습니다.



### Towards Visual Text Design Transfer Across Languages (https://arxiv.org/abs/2410.18823)
- **What's New**: 이번 연구에서는 시각적 텍스트 디자인을 언어 간 번역하는 Multimodal Style Translation (MuST-Bench)라는 새로운 작업을 소개합니다. 이는 디자인 의도를 유지하며 다양한 글쓰기 시스템 간에 번역할 수 있습니다.

- **Technical Details**: SIGIL (Style Integrity and Glyph Incentive Learning) 프레임워크를 도입하여 스타일 설명이 필요 없는 다중 모드 스타일 번역을 가능하게 합니다. 이 프레임워크는 다국어 설정을 위한 glyph latent, 안정적인 스타일 지침을 위한 사전 훈련된 VAE, 가독성 문자인식을 최적화하기 위한 강화 학습 피드백이 포함된 OCR 모델을 통해 이미지를 생성하는 모델을 개선합니다.

- **Performance Highlights**: SIGIL은 기존의 기초 모델들보다 우수한 스타일 일관성과 가독성을 달성하며, 시각적 충실성을 유지하는 데에도 뛰어납니다. MuST-Bench 베이스라인 데이터를 통해 성능이 입증되었으며, 영어에서 한국어를 포함한 다섯 가지 언어로의 전환 능력이 평가되었습니다.



### Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis (https://arxiv.org/abs/2410.18822)
Comments:
          Accepted by NeurIPS 2024. Project page: this https URL

- **What's New**: 이 논문에서는 외부의 supervision (감독) 없이 희소 뷰로부터 새로운 뷰를 합성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 disparity-guided image warping (불일치에 기반한 이미지 왜곡)을 통해 각각의 쌍의 이진 이미지를 구성하여 이들 간의 이진 스테레오 일관성(semi-supervision)을 탐색합니다. 더불어, Gaussian opacity constraint (가우시안 불투명도 제약)를 도입하여 3D 가우시안 탐색의 강건성과 효율성을 향상시킵니다.

- **Performance Highlights**: LLFF, DTU, Blender 데이터셋에서의 실험 결과, 제안 방법은 기존 최첨단 방법들보다 월등히 높은 성능을 보여주었습니다.



### Learning Global Object-Centric Representations via Disentangled Slot Attention (https://arxiv.org/abs/2410.18809)
Comments:
          Global Object-Centric Representations, Object Identification, Unsupervised Learning, Disentangled Learning

- **What's New**: 이 논문은 인공지능(AI) 시스템이 장면을 넘어 객체를 식별하고 특정 객체를 포함하는 다양한 장면을 생성할 수 있도록 하는 새로운 영역 중심 학습 방법인 GOLD(Global Object-centric Learning via Disentangled slot attention)를 소개합니다. 이 방법은 장면 의존성을 분리하여 전역 객체 중심 표현(global object-centric representation)을 학습합니다.

- **Technical Details**: GOLD는 두 가지 주요 구성요소로 이루어져 있습니다: 이미지 인코더-디코더 모듈과 글로벌 객체 중심 학습 모듈. 이미지 인코더-디코더 모듈은 DINO(ViT encoder)를 사용하여 이미지 특징을 추출하고 VQ-VAE 디코더를 통해 전체 이미지를 복원합니다. 글로벌 객체 중심 학습 모듈은 장면 의존성과 독립성을 분리하여 객체의 외부 속성과 객체 정체성 표현을 구분합니다.

- **Performance Highlights**: 실험 결과, GOLD는 다양한 데이터셋에서 객체 식별 및 특정 객체가 포함된 장면 생성에서 뛰어난 성능을 보이며, 기존 방법과 비교하여 글로벌 객체 중심 표현 학습에서 현저한 우수성을 입증하였습니다.



### Fast constrained sampling in pre-trained diffusion models (https://arxiv.org/abs/2410.18804)
- **What's New**: 이 논문에서는 대규모 사전 학습된 확산 모델, 특히 Stable Diffusion을 활용하여 빠른 제약 샘플링(fast-constrained sampling)을 구현하는 새로운 알고리즘을 제안합니다. 이 알고리즘은 기존 방법의 비싼 역전파(backpropagation) 연산 없이 샘플링을 가능하게 하여 결과적으로 신속한 이미지 생성 시간을 자랑합니다.

- **Technical Details**: 제안된 방법은 제약 조건 하에서 샘플링을 최적화하는 새로운 관점에 기초하고 있습니다. 연구팀은 샘플링 중 확산 잠재 변수(diffusion latents)에 대한 대체 그래디언트 업데이트(alternative gradient update)를 탐구하며, 이전 방법들과의 차별성을 두고, 기계 학습에서 일반적으로 사용되는 비싼 그래디언트 계산을 수치적 근사(numerical approximation)로 대체하여 속도를 크게 향상시킵니다.

- **Performance Highlights**: 이 방법은 기존 최고 성능의 튜닝 모델들과 비교할 때 비슷한 품질의 이미지를 제공하며, 상당히 짧은 추론(inference) 시간을 기록합니다. 기존의 다수의 처리 단계 및 반송통신이 요구되는 방법들과 비교했을 때, 새로운 접근 방식은 상당한 계산 속도 개선을 보여주며, 실제 응용에서도 뛰어난 성능을 입증합니다.



### Learning Geodesics of Geometric Shape Deformations From Images (https://arxiv.org/abs/2410.18797)
- **What's New**: 본 논문에서는 이미지에서 유도된 변형 필드의 지오데식 흐름을 학습하는 새로운 방법, 지오데식 변형 네트워크(GDN)를 소개합니다. GDN은 지오데식을 예측하는 능력이 있으며, 이는 이미지에서 나타나는 변형된 형태를 정량화하고 비교하는 데 중요합니다.

- **Technical Details**: GDN은 지오데식 손실을 최적화하여 네트워크의 규칙성과 일반성을 향상시키며, 수치 적분기(numercial integrators)를 통해 해결된 변형 과정을 통해 지오데식 매핑을 학습합니다. 네트워크의 각 숨겨진 층에서 이러한 매핑을 효과적으로 근사하는 적분 연산자(integral operators)와 부드러운 활성화 함수(smooth activation functions)의 조합이 형성됩니다.

- **Performance Highlights**: GDN은 2D 합성 데이터와 3D 실제 뇌 MRI에서 실험을 통해 기존 수치 솔루션에 비해 가까운 지오데식을 예측할 수 있음을 보여주며, OOD(out-of-distribution) 데이터에서 가장 최신의 딥러닝 기반 등록 네트워크보다 뛰어난 성능을 나타냅니다.



### WARP-LCA: Efficient Convolutional Sparse Coding with Locally Competitive Algorithm (https://arxiv.org/abs/2410.18794)
- **What's New**: 이번 논문은 Predictive Priming을 통한 Locally Competitive Algorithm (LCA) 개선 방법인 WARP-LCA를 제안하여, 성능과 수렴 속도를 크게 향상시켰습니다. 기존의 LCA는 비효율적인 최적화 사이클과 서브옵티멀(minima) 문제를 겪었으나, WARP-LCA는 예측 네트워크를 활용하여 초기 상태 추정을 통해 이러한 문제를 해결합니다.

- **Technical Details**: WARP-LCA는 예측 네트워크를 사용하여 LCA의 초기 상태를 현재 입력에 기반하여 제공하며, 이는 효율적인 최적화 수치를 도출합니다. 이 방법은 ℓ0(ell_0) 비선형 손실 함수를 회피하여 더 나은 섬세한 솔루션을 제시합니다. 논문에서는 WARP-LCA가 전통적인 LCA 대비 수렴 속도가 몇 배 향상되며, 더 우수한 최솟값에 도달한다는 결과를 보여줍니다.

- **Performance Highlights**: WARP-LCA는 이미지 적군(demoing) 작업에서 더 강력하고 실용적인 효과를 발휘하며, 노이즈와 기타 왜곡을 더 효과적으로 제거하는 성능을 보입니다. 실험 결과는 LCA의 한계점을 극복하고, 생물학적 영감을 받은 딥 러닝 분야의 진전을 이룬다는 것을 보여줍니다.



### Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances (https://arxiv.org/abs/2410.18775)
- **What's New**: 이번 연구에서는 W-Bench라는 최초의 종합 워터마크 평가 벤치마크를 제안하여 다양한 이미지 편집 기술에 대해 워터마크 방법의 강인성을 평가합니다. 이를 통해 여러 대표적인 워터마킹 방법들이 이러한 편집 후 워터마크를 인식하지 못하는 경우가 많다는 사실을 보여줍니다.

- **Technical Details**: 제안하는 VINE 방법은 두 가지 주요 혁신을 포함하여 이미지 편집에 대한 강인성을 크게 향상시킵니다. 첫째, 이미지 편집의 주파수 특성을 분석하여 블러링 왜곡이 유사한 주파수 특성을 갖는 것을 발견하고, 이를 훈련 중 대체 공격으로 사용하여 워터마크 강인성을 강화합니다. 둘째, 대규모로 사전 훈련된 확산 모델인 SDXL-Turbo를 워터마킹 작업에 적응시켜 더욱 인지되지 않으면서 강인한 워터마크 임베딩을 달성합니다.

- **Performance Highlights**: 실험 결과, VINE 방법은 다양한 이미지 편집 기술에 대해 뛰어난 워터마크 성능을 달성하여 기존 방법들에 비해 이미지 품질과 강인성 모두에서 성능이 우수함을 보여줍니다.



### Schedule Your Edit: A Simple yet Effective Diffusion Noise Schedule for Image Editing (https://arxiv.org/abs/2410.18756)
Comments:
          Accepted in NeurIPS 2024

- **What's New**: 최근 연구에서 텍스트 유도 확산 모델(Text-guided diffusion models)은 이미지 편집(Image editing)에서 고품질 및 다양성을 제공하는 데 크게 기여했습니다. 하지만 이미지의 원본을 잠재 공간(Latent space)으로 변환하는 과정에서는 예측 오류(Prediction errors)로 인해 문제점이 발생합니다. 본 연구는 이러한 오류를 줄이기 위해 새로운 로지스틱 스케줄(Logistic Schedule)을 제안합니다.

- **Technical Details**: 이 연구는 DDIM inversion에서 오류 누적(Error accumulation)의 주요 원인을 조사하고, 전통적인 노이즈 스케줄(Noise schedules)에서 발생하는 특이점(Singularity) 문제를 발견했습니다. 로지스틱 스케줄은 이 문제를 해결하기 위해 설계된 새로운 노이즈 스케줄로, 편집 안정성을 개선하고 노이즈 예측 오류를 줄이며, 기존 편집 방법들과의 호환성을 유지합니다.

- **Performance Highlights**: 여덟 가지 편집 작업을 통해 로지스틱 스케줄이 전통적인 노이즈 스케줄에 비해 원본 콘텐츠 보존(Content preservation) 및 편집 충실도(Edit fidelity) 측면에서 우수한 성능을 보임을 확인했습니다. 이러한 결과는 로지스틱 스케줄의 적응력과 효과성을 강조합니다.



### Cellpose+, a morphological analysis tool for feature extraction of stained cell images (https://arxiv.org/abs/2410.18738)
- **What's New**: 이 연구에서는 Cellpose라는 최첨단 세포 분할 프레임워크의 기능을 확장하여 형태학적 특성을 평가할 수 있는 기능 추출을 도입하고, DAPI 및 FITC 염색 세포의 새로운 데이터 세트를 소개합니다.

- **Technical Details**: 이 논문은 딥 러닝 기반의 자동화된 이미지 분석 방법을 사용하여 세포의 정확한 분할 및 특징 추출을 수행하는 방법에 대해 설명합니다. 특히, 다양한 신경망 구조를 활용해 세포의 핵 및 세포를 인식하는 과정을 pixelwise classification 문제로 설정하여 확률 맵을 생성합니다. 이를 통해 객체의 위치를 확인하고 비최대 억제(non-maximum suppression) 기법을 통해 정확도를 높입니다.

- **Performance Highlights**: Cellpose와 기능 추출 알고리즘을 결합하여 실험 데이터의 정확성과 효율성을 극대화하였으며, 새로운 데이터 세트를 통해 세포 분석의 가능성을 더욱 확장하고 있습니다.



### Rectified Diffusion Guidance for Conditional Generation (https://arxiv.org/abs/2410.18737)
- **What's New**: 본 논문에서는 Classifier-Free Guidance (CFG)의 이론적 결함을 재조명하고, 이 문제를 해결하기 위한 새로운 접근 방식인 ReCFG를 제안합니다. ReCFG는 지침 계수에서의 완화가 이루어져, 노이즈 제거가 확산 이론과 엄격히 일치하도록 합니다.

- **Technical Details**: CFG는 조건부 및 비조건부 점수 함수의 조합을 통해 생성적 분포의 기대값을 이동시키는 문제가 있습니다. 이 논문에서는 잘못된 조합 계수 구성이 기대값 이동의 원인임을 확인하고, 이를 해결하기 위해 두 개의 계수를 사용하는 ReCFG를 제안합니다. 이 접근방식은 관찰된 데이터를 통해 비율을 쉽게 사전 계산할 수 있도록 하며, 샘플링 속도에 거의 영향을 미치지 않습니다.

- **Performance Highlights**: 실험 결과, ReCFG는 기존 최첨단 확산 모델들과의 호환성을 보이며, 어떠한 재교육 없이도 클래스 조건화 및 텍스트 조건화된 샘플 생성이 가능함을 보여주었습니다. 향후 코드를 오픈 소스로 공개하여 추가 연구를 지원할 예정입니다.



### VoxelKeypointFusion: Generalizable Multi-View Multi-Person Pose Estimation (https://arxiv.org/abs/2410.18723)
- **What's New**: 본 연구에서는 다중 관점 다중 인물 자세 추정기(multi-view multi-person pose estimator)의 일반화 능력을 평가하고, 깊이 정보(depth information)를 추가로 활용하여 성능을 향상시키는 새로운 알고리즘을 제안합니다. 이 알고리즘은 보지 못한 데이터셋에 대한 일반화가 우수할 뿐만 아니라 다양한 키포인트(keypoints)에 대해서도 잘 작동하여 최초의 다중 관점 다중 인물 전체 신체 자세 추정기를 구현합니다.

- **Technical Details**: 새롭게 제안된 VoxelKeypointFusion 알고리즘은 두 단계로 구성됩니다. 첫 번째 단계에서는 각 이미지에 대해 2D 자세 예측을 수행하며, 여기에는 RTMPose와 같은 2D 자세 추정기가 통합됩니다. 두 번째 단계에서는 모든 열 관절 히트맵을 공유된 볼록화된 공간(voxelized space)으로 투영하고, 각 관절마다 임계치(threshold) 이상의 피크를 찾습니다. 이후 동일한 인물 ID를 가진 제안을 그룹화하고, 중심을 계산하여 최종적으로 최적의 키포인트 제안을 이용해 사람을 구성하게 됩니다.

- **Performance Highlights**: 새로운 알고리즘은 unseen datasets에 대해 뛰어난 일반화 성능을 보였으며, 이는 다양한 응용 프로그램에서 활용될 수 있을 것으로 기대됩니다. 전체 신체 자세 추정이 가능해지면서, 예를 들어 인간-로봇 협업에서 작업자의 동작 분석에 있어 손가락의 움직임 정보를 더 효과적으로 활용할 수 있는 가능성을 제시합니다.



### Low-Latency Video Anonymization for Crowd Anomaly Detection: Privacy vs. Performanc (https://arxiv.org/abs/2410.18717)
Comments:
          16pages, 8 figures, 9 tables

- **What's New**: 이번 연구에서는 기존의 깊은 학습 모델 기반의 개인 정보 보호 기법에 대한 한계를 극복하기 위해 경량 적응형 익명화(LA3D) 기법을 제안합니다. 이 기법은 실시간 비디오 이상 탐지(VAD) 애플리케이션에 특화되어 있으며, 개인의 프라이버시를 보호하면서 효율적인 비디오 분석을 가능하게 합니다.

- **Technical Details**: LA3D는 고속 세분화 모델을 사용하여 비디오 프레임에서 사람의 피사체 마스크를 식별하고, 각 감지된 신체부위에 대해 상대적인 특성에 따라 동적 모자이크를 적용합니다. 이 방법은 비디오 프레임의 깊이 변화에 따라 하이퍼파라미터를 동적으로 조정하여 개인 정보를 보호하는 데 집중합니다.

- **Performance Highlights**: 실험 결과, LA3D는 기존 알고리즘보다 개인 정보 익명화 능력을 크게 향상시키면서도 비디오 이상 탐지의 유효성을 저하시키지 않음을 입증하였습니다. 이전의 경량 전통 이미지 익명화 기법들을 재조명하고, 효과적인 개인 정보 보호 및 VAD 애플리케이션을 위한 새로운 전환점을 만들어가고 있습니다.



### ChatSearch: a Dataset and a Generative Retrieval Model for General Conversational Image Retrieva (https://arxiv.org/abs/2410.18715)
- **What's New**: 이번 연구에서는 오픈 도메인 이미지를 기반으로 한 일반적 대화형 이미지 검색(task of general conversational image retrieval) 문제를 다룹니다. 새로운 데이터셋 ChatSearch를 구축하였으며, 이 데이터셋은 다중 라운드의 다중 모달 대화(Multi-modal conversation) 컨텍스트 쿼리를 포함하여 이미지 검색의 정확성을 요구합니다.

- **Technical Details**: ChatSearcher라는 생성적 검색 모델(generative retrieval model)을 제안하며, 이는 이미지-텍스트 입력 및 출력을 혼합하여 수용하고 생성하도록 end-to-end 학습되었습니다. ChatSearcher는 대화형 이미지 검색에 최적화되어 있으며, 대화에 포함된 정보의 도출 및 이미지 특성을 동적으로 업데이트하도록 설계되었습니다. 또한, 대화형 검색 명령, 시각적 대화 명령 및 AI 생성 콘텐츠 조작을 포함한 다양한 명령 데이터로 instructive tuning을 시행합니다.

- **Performance Highlights**: ChatSearcher는 ChatSearch 데이터셋에서 우수한 성능을 보였으며, 제로샷 텍스트-이미지 검색(Zero-shot text-to-image retrieval) 및 제로샷 구성 이미지 검색(Zero-shot composed image retrieval)에서도 경쟁력 있는 성능을 보여줍니다. 또한 visual conversation task에서도 비슷한 성과를 나타내며, 멀티모달 대화(context)를 이해하고 추론하는 능력을 갖추고 있습니다.



### PESFormer: Boosting Macro- and Micro-expression Spotting with Direct Timestamp Encoding (https://arxiv.org/abs/2410.18695)
- **What's New**: 본 논문은 PESFormer라는 간단하면서도 효과적인 모델을 소개하여, 기존의 앵커 기반 방법의 한계를 극복하고 비디오 내 모든 훈련 간격을 효과적으로 활용할 수 있는 새로운 접근 방식을 제안합니다.

- **Technical Details**: PESFormer는 비전 트랜스포머 아키텍처를 기반으로 하며, 직접적인 타임스탬프 인코딩(DTE) 방법을 통해 각 타임스탬프를 이진 분류합니다. 이 모델은 불규칙한 비디오를 제로 패딩하여 정해진 길이의 균일한 비디오로 변환하여 모든 관련 훈련 간격을 유지합니다.

- **Performance Highlights**: PESFormer는 CAS(ME)², CAS(ME)³ 및 SAMM-LV 데이터셋에서 기존 기술보다 뛰어난 성능을 보여주며 최척 정도의 결과(State-of-the-art)를 달성했습니다.



### ODDN: Addressing Unpaired Data Challenges in Open-World Deepfake Detection on Online Social Networks (https://arxiv.org/abs/2410.18687)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문에서는 소셜 네트워크에서의 딥페이크 탐지의 새로운 접근 방식인 Open-world Deepfake Detection Network (ODDN)을 제안합니다. ODDN은 두 가지 핵심 모듈인 Open-world Data Aggregation (ODA)와 Compression-discard Gradient Correction (CGC)을 포함하여, 비공식 샘플과 공식 샘플 간의 관계를 강화하는 것을 목표로 합니다.

- **Technical Details**: ODDN은 ODA와 CGC라는 두 가지 모듈로 구성됩니다. ODA는 유사한 특성을 가진 압축 이미지와 원본 이미지 간의 미세 및 조잡한 분석을 통해 상관관계를 집계합니다. CGC는 압축을 제거할 때 발생하는 그래디언트 오차를 수정하여 다양한 압축 방법에서의 성능 향상을 도모합니다. 특히, 데이터의 비대칭성을 해결하기 위해 20%의 쌍데이터와 80%의 비쌍데이터를 사용하는 새로운 훈련 데이터 설정을 도입하여 성능을 검증하였습니다.

- **Performance Highlights**: 17개의 유명한 딥페이크 데이터셋에서 실시된 실험 결과, ODDN이 기존의 SOTA 모델에 비해 우수한 성능을 보였으며, 특히 여러 소셜 네트워크 플랫폼에서의 압축 상황을 고려한 딥페이크 탐지에서 뛰어난 효과를 나타냈습니다.



### Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks (https://arxiv.org/abs/2410.18684)
- **What's New**: 이번 논문에서는 기존의 semantic segmentation (의미 체계 분할) 평가를 다중 인스턴스 감지를 고려한 새로운 평가지표인 Connected-Component (CC)-Metrics를 제안합니다. 이를 통해 기존 평가 지표가 대형 연결 구성요소(connected component)에 편향되는 문제를 해결하고자 합니다. 특히, PET/CT 검사에서의 종양 분할 상황을 기반으로 하여 모든 종양에 동등한 중요성을 부여하는 방법론을 소개합니다.

- **Technical Details**: CC-Metrics는 각 인스턴스의 크기에 관계없이 동등하게 평가하여 기존의 overlap-based metrics (겹침 기반 지표)의 편향을 극복합니다. 예를 들어, Dice 또는 Surface Dice와 같은 지표의 경우, 대형 종양의 영향력이 작용하여 작은 변화가 간과되는 문제를 해결합니다. 이 방법은 Voronoi diagram을 활용하여 이미지 내에서 예측을 실제의 가장 가까운 연결 구성요소에 맞추고 기존 지표를 지역적으로 평가합니다.

- **Performance Highlights**: 실험 결과, CC-Metrics는 PET/CT 데이터셋에서 다양한 평가 시나리오를 통해 기존 지표보다 더 유의미한 평가 지표를 제공합니다. 여러 segmentation 모델을 평가한 결과, CC-Metrics를 사용했을 때 모델 간 성능 차이가 보다 뚜렷하게 나타났습니다. 이러한 점에서, CC-Metrics는 메타스타시스(segmentation)와 같은 의료 영상 분석의 경우 특히 유용한 도구가 될 것입니다.



### Rigid Single-Slice-in-Volume registration via rotation-equivariant 2D/3D feature matching (https://arxiv.org/abs/2410.18683)
- **What's New**: 이 논문에서는 2D 이미지를 3D 볼륨에 등록하기 위한 새로운 방법인 SLIV-Reg를 제안합니다. 이 방법은 기존의 수동적인 랜드마크 선택이나 초기 포즈 설정의 필요없이, 자가 지도(Self-supervised) 방식으로 2D 슬라이스를 3D 볼륨에 정렬할 수 있도록 돕습니다.

- **Technical Details**: SLIV-Reg는 group-equivariant CNNs를 사용하여 2D 슬라이스와 3D 볼륨의 특징을 추출하며, 이를 통해 2D 쿼리 슬라이스와 3D 후보 간의 대응 관계를 수립합니다. 이 방법은 또한 Canny edge detector를 활용하여 후보 포인트를 제한함으로써 높은 효율성을 보입니다.

- **Performance Highlights**: NSCLC-Radiomics CT 및 KIRBY21 MRI 데이터셋에서 실행된 실험 결과, SLIV-Reg는 절대 중앙 각도 오차가 2도 이하로 측정되었으며, 평균 일치 특징 정확도가 3픽셀 허용 수치에서 89%에 달하는 성능을 보였습니다.



### Ali-AUG: Innovative Approaches to Labeled Data Augmentation using One-Step Diffusion Mod (https://arxiv.org/abs/2410.18678)
- **What's New**: Ali-AUG는 산업 분야에서 효율적인 레이블된 데이터 증대(data augmentation)를 위한 새로운 단일 단계 확산(diffusion) 모델로, 효율적인 합성 레이블 이미지 생성을 가능하게 합니다.

- **Technical Details**: Ali-AUG는 안정적인 확산 아키텍처를 활용하여 마스크(mask)와 이미지를 효율적으로 통합하며, 정확한 기능 삽입(feature insertion)을 보장합니다. 또한, Low-Rank Adaptation (LoRA) 모듈을 통해 모델의 안정성과 효율성을 향상시킵니다.

- **Performance Highlights**: Ali-AUG는 다양한 산업 데이터셋에서 기존 증대 방법보다 31% 성능을 개선하고, 데이터 증대 없이 훈련한 모델보다 45% 향상된 성능을 보여줍니다. 훈련 시간 또한 32% 단축되며, 쌍(pair) 및 비쌍(unpaired) 데이터셋 모두에서 유연한 사용이 가능합니다.



### Enhancing pretraining efficiency for medical image segmentation via transferability metrics (https://arxiv.org/abs/2410.18677)
- **What's New**: 본 논문에서는 의료 이미지 세분화 작업을 위한 새로운 전이 가능성 지표(transferability metric)를 제안하며, 짧은 사전 훈련(pretraining)이 실제 작업에 더 나은 성능을 발휘할 수 있음을 발견하였습니다.

- **Technical Details**: 저자들은 ImageNet에서 사전 훈련된 모델을 사용하여 300개 이상의 조합의 모델, 데이터셋, 훈련 방법을 조사하였고, 이 과정에서 대비 학습(contrastive learning)을 기반으로 한 신뢰성 평가 지표를 개발하였습니다.

- **Performance Highlights**: 결과에 따르면, 사전 훈련 시간이 최적화된 경우 모델의 다운스트림 성능이 더 향상되었으며, 이는 사전 훈련에 소요되는 시간을 단축하면서도 기존의 방식보다 경미하게 더 나은 성능을 기록하였습니다.



### 3D Shape Completion with Test-Time Training (https://arxiv.org/abs/2410.18668)
- **What's New**: 이 연구는 	extit{shape completion} (형태 완성) 문제를 다루며, 불완전한 형태를 복원하는 데 필요한 누락된 부분을 예측하는 방법을 제시합니다. 이전의 연구들은 일반적으로 파손된 형태와 복원된 형태를 한 번에 예측했던 반면, 저자는 이를 파손된 부분과 새롭게 복원된 부분을 별도로 예측하되 두 예측이 서로 연결되도록 접근했습니다.

- **Technical Details**: 이 과정에서는 signed distance functions (SDF)의 예측에 기반한 디코더 네트워크를 사용하며, 이 표현을 통해 테스트 시간 훈련(test-time-training)을 고려합니다. 즉, 주어진 불완전한 형태에 더 정확하게 일치하도록 네트워크 매개변수를 미세 조정할 수 있습니다.

- **Performance Highlights**: 이 연구는 ShapeNet 데이터 세트의 여덟 가지 다른 형태 범주에 대한 복원 성능을 향상시키며, 기존 연구들이 파손 경계 주위에서 나타나는 아티팩트(artifacts) 문제를 해결하는 방법으로 특히 두드러진 개선을 보여줍니다.



### DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation (https://arxiv.org/abs/2410.18666)
Comments:
          Accepted by NeurIPS 2024

- **What's New**: 이 논문에서는 이미지 복원 (Image Restoration, IR) 분야의 실질적인 과제를 해결하기 위해 GenIR이라는 혁신적인 데이터 큐레이션 파이프라인과 DreamClear라는 최첨단 Diffusion Transformer (DiT) 기반 이미지 복원 모델을 소개합니다. GenIR은 다중 프롬프트 학습 전략으로, 기존 데이터셋의 제한을 극복하여 100만 개의 고품질 이미지를 생성합니다. DreamClear는 이 데이터를 활용하여 포토리얼리스틱 복원을 달성하는 데 목적을 둡니다.

- **Technical Details**: GenIR은 이미지-텍스트 쌍 구축, 이중 프롬프트 기반 미세 조정, 데이터 생성 및 필터링의 세 단계로 구성됩니다. 이 과정에서 MLLM(Gemini-1.5-Pro)을 활용하여 다양한 장면 설명을 생성하고, 저작권과 개인정보 보호 문제를 회피합니다. DreamClear 모델은 안정적인 확산 모델을 기반으로 하여, Mixture of Adaptive Modulator (MoAM)를 통해 다양한 손상 정도에 적응할 수 있는 능력을 강화합니다.

- **Performance Highlights**: 실험을 통해 DreamClear는 저수준 및 고수준 벤치마크에서 진행된 평가에서 최신 성능을 기록하였으며, 복잡한 실세계 손상을 효과적으로 처리할 수 있음을 입증하였습니다.



### A Cranial-Feature-Based Registration Scheme for Robotic Micromanipulation Using a Microscopic Stereo Camera System (https://arxiv.org/abs/2410.18630)
Comments:
          Accepted by Advanced Robotics, Vol. 38, Issue 21

- **What's New**: 본 연구는 생물학적 표본의 변동성을 극복하기 위한 마우스 두개골 창 만들기 작업을 중심으로 한 새로운 미세 스테레오 카메라 시스템(MSCS)과 정밀 등록 스킴을 소개합니다.

- **Technical Details**: MSCS는 깊이 인식을 위한 선형 모델로 보완되었으며, CNN 기반의 제약 및 색상 등록 전략을 통해 부분적으로 노출된 마우스 두개골 표면을 등록하는 정밀한 схем을 개발했습니다. 이 시스템은 3D 재구성을 위해 초당 30 프레임(FPS)으로 동작하며, 정밀도는 0.10 mm ± 0.02 mm로 측정되었습니다.

- **Performance Highlights**: MSCS는 105개의 지속적인 프레임에서 전이 오차 1.13 mm ± 0.31 mm, 회전 오차 3.38° ± 0.89°로 테스트되었습니다. 이 연구는 과학 및 외과적 환경에서 로봇 미세 조작의 정밀도를 높이는 데 기여할 혁신적인 방법을 제안합니다.



### Environment Maps Editing using Inverse Rendering and Adversarial Implicit Functions (https://arxiv.org/abs/2410.18622)
- **What's New**: 본 논문은 기존의 HDR (High Dynamic Range) 환경 맵 편집 과정을 혁신적으로 개선하는 방법을 제안합니다. 새로운 역진화 렌더링 아키텍처를 사용하여 희소성 문제를 해결하고 HDR 이미지에서의 픽셀 값 간의 변동성을 관리합니다.

- **Technical Details**: 새로운 접근법은 HDR 이미지의 자연스러운 편집을 가능하게 하며, 전통적인 pixel 공간에서의 최적화가 아닌Robust Implicit Function을 사용하여 각 환경 맵을 매개변수화합니다. 이 방법은 Adversarial Perturbations를 이용하여 최적화될 때 출력이 매끄럽고 안정적이도록 합니다. 또한, 기존 SIREN 기반의 암시적 표현을 HDR 이미지 처리에 맞게 조정하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 HDR 환경 맵의 원하는 조명 효과를 효과적으로 재구성하면서도, 시각적인 일관성을 유지하는 것으로 나타났습니다. 이 접근법은 3D 장면의 조명 조절을 가능하게 하여, 예술가들이 효율적으로 환경 맵을 조정할 수 있도록 합니다.



### FairQueue: Rethinking Prompt Learning for Fair Text-to-Image Generation (https://arxiv.org/abs/2410.18615)
Comments:
          Accepted in NeurIPS24

- **What's New**: 이번 연구는 텍스트-이미지 변환에서 공정한 생성(Fair T2I generation)을 달성하기 위해 기존의 프롬프트 학습(Prompt Learning) 방법이 샘플 질 저하라는 문제를 야기할 수 있음을 밝혔습니다.

- **Technical Details**: 연구자들은 프롬프트 학습 기반 접근 방식이 학습된 프롬프트와 참조 이미지 간의 임베딩 차이를 정렬하는 것을 목표로 했으나, 이로 인해 프롬프트가 왜곡되고 생성된 이미지 품질이 저하된다는 것을 발견했습니다. 이 과정에서 크로스-어텐션 맵(Cross-Attention Maps)을 분석하여 초기 디노이징 단계에서의 비정상성을 파악하였습니다.

- **Performance Highlights**: 연구에서 제안한 Prompt Queuing과 Attention Amplification 방법은 기존 SOTA 접근 방식보다 향상된 이미지 생성 품질을 보여주었고, 다양한 민감한 속성(tSA)에 대해 경쟁력 있는 공정성을 달성했습니다.



### On Model-Free Re-ranking for Visual Place Recognition with Deep Learned Local Features (https://arxiv.org/abs/2410.18573)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 논문은 모델 프리 (model-free) 리랭킹 (re-ranking) 접근 방식을 기반으로 한 새로운 세 가지 방법을 소개하며, 이 방법들이 장기 자율 시스템 (long-term autonomy systems)에서 어떻게 적용될 수 있는지를 다룹니다.

- **Technical Details**: 제안된 방법은 기본적으로 표준 로컬 외관 특징 (local visual features)을 사용하는데, 이는 이미지 변환에 대한 복잡한 모델 추정을 피할 수 있게 해주며, 계산적으로 비용이 낮습니다. D2-net 특징 감지기 (feature detector)와 결합하여 다양한 공공 데이터셋에서 실험을 수행하였습니다.

- **Performance Highlights**: 시험 결과, 제안된 모델 프리 접근 방식은 현재의 최첨단 (state-of-the-art) 방법들과 동등한 성능을 보였으며, 장기 비주얼 장소 인식 (visual place recognition)에 있어 모델 프리 접근이 실행 가능한 유망한 경로임을 확인하였습니다.



### Research on gesture recognition method based on SEDCNN-SVM (https://arxiv.org/abs/2410.18557)
- **What's New**: 이 논문에서는 sEMG(표면 전자 근육 신호)를 기반으로 한 제스처 인식에서 전통적인 수동 특징 추출 방식의 한계를 극복하고자 하는 새로운 접근법인 SEDCNN-SVM을 제안합니다.

- **Technical Details**: SEDCNN-SVM은 향상된 심층 합성곱 신경망(Deep Convolutional Neural Network, DCNN)과 서포트 벡터 머신(Support Vector Machine, SVM)으로 구성됩니다. DCNN은 합성곱 층의 합성곱 연산을 통해 sEMG의 특징 정보를 자동으로 추출하고 학습하여 복잡하고 고차원적인 데이터의 특징을 포착합니다. Squeeze and Excitation Networks(SE-Net) 및 잔차 모듈(residual module)을 모델에 추가하여 각 채널의 특징 표현을 개선하고, 합성곱 연산에서의 특징 정보 손실을 줄이며, 유용한 특징 정보를 포함시켰고, 네트워크의 기울기 소실 문제를 완화합니다. SVM은 특징 공간의 최적 초평면(optimal hyperplane)을 구축하여 모델의 일반화 능력과 분류 정확도를 향상시킵니다.

- **Performance Highlights**: SEDCNN-SVM의 인식 정확도는 0.955에 도달하였으며, 다른 분류 방법들에 비해 현저하게 개선되었습니다. 이 모델은 실시간으로 온라인 인식이 가능합니다.



### Local and Global Graph Modeling with Edge-weighted Graph Attention Network for Handwritten Mathematical Expression Recognition (https://arxiv.org/abs/2410.18555)
- **What's New**: 이 논문에서는 그래프 기반 모델링 기법을 활용하여 손글씨 수학 식 인식(Handwritten Mathematical Expression Recognition, HMER)을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: We propose an End-to-end model with an Edge-weighted Graph Attention Mechanism (EGAT) that performs simultaneous node and edge classification. The method includes stroke-level Local Graph Modeling (LGM) and Global Graph Modeling (GGM) to effectively capture both local and global graph features.

- **Performance Highlights**: Our proposed system shows strong performance in symbol detection, relation classification, and expression-level recognition in HMER tasks, demonstrating a comprehensive understanding of expression structures.



### Interpretable Representation Learning from Videos using Nonlinear Priors (https://arxiv.org/abs/2410.18539)
Comments:
          Accepted to BMVC 2024 (Oral)

- **What's New**: 이 논문에서는 비선형 Additive Noise Model (ANM)을 사용하는 새로운 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 비디오의 해석 가능한 잠재 변수를 학습하고, 학습 시 관찰되지 않은 가상 시나리오의 비디오를 생성할 수 있도록 합니다.

- **Technical Details**: Variational Auto-Encoder (VAE)에서 단순한 등방성 가우시안 프라이어를 비선형 ANM으로 확장하였으며, 이를 통해 객체의 위치와 같은 물리적 변수를 학습할 수 있습니다. 새로운 Gaussian Mixture Model (GMM) 근사 방법과 KL divergence를 이용한 데이터 로그 가능성 최적화 방법을 제안합니다.

- **Performance Highlights**: 낙하하는 물체, 진동하는 물체, 진자, 그리고 크랩 펄서 (pulsating Crab pulsar) 등 다양한 실세계 물리 비디오에 대해 유효성을 검증하였으며, 올바른 잠재 변수를 학습하고 훈련 분포 외부에서 현실적인 비디오를 생성해냈습니다.



### SMITE: Segment Me In TimE (https://arxiv.org/abs/2410.18538)
Comments:
          Technical report. Project page is at \url{this https URL}

- **What's New**: 이 논문은 비디오에서 객체를 세분화하는 데 있어 이전의 세그멘테이션 방식에 비해 향상된 접근 방식을 제시합니다. 특히, 몇 개의 참조 이미지를 통해 다양한 비디오 세그멘테이션 상황을 관리할 수 있는 SMITE 방법을 도입하여 기존 방법들을 능가하는 성능을 보여줍니다.

- **Technical Details**: SMITE 방법은 pretrained text-to-image diffusion model을 활용하고, 추가적인 temporal attention을 통해 시간적 일관성을 유지하며, temporal voting 메커니즘을 통해 각 픽셀의 레이블 일관성을 보장합니다. 이 방법은 레퍼런스 이미지에 따라 구조를 보존하며, noise와 flickering을 크게 줄이는 결과를 나타냅니다.

- **Performance Highlights**: SMITE는 기존의 비디오 세그멘테이션 방법들과 비교하여 더욱 향상된 정확성과 시간적 일관성을 보여주며, SMITE-50이라는 새로운 소규모 데이터셋을 통해 이 방법의 우수성을 입증하였습니다. 또한, 사용자 연구를 통해 세그멘테이션의 정확성과 일관성 측면에서 긍정적인 결과를 도출하였습니다.



### Beyond Color and Lines: Zero-Shot Style-Specific Image Variations with Coordinated Semantics (https://arxiv.org/abs/2410.18537)
Comments:
          13 pages,6 figures

- **What's New**: 이 연구에서는 스타일이 단순한 예술적 요소(예: 색상, 붓터치, 조명)뿐만 아니라, 의미(semantics)도 포함한다는 점을 강조하며, 이미지 변형을 위한 새로운 제로샷(zero-shot) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 이미지-텍스트-이미지(image-to-text-to-image) 문제로 변환하여, BLIP과 ChatGPT를 활용하여 설명을 생성하고 스타일 키워드를 연관 지어 이미지 생성을 수행합니다. 확산 모델(Diffusion model)을 사용해 텍스트 프롬프트(prompt)를 기반으로 이미지를 생성하며, 크로스-어텐션(cross-attention) 메커니즘을 조정하여 다양한 스타일을 다룰 수 있도록 튜닝합니다.

- **Performance Highlights**: 제안한 방법은 다양한 스타일의 이미지를 생성하는 데에 있어 높은 품질과 의미의 충실도(content fidelity)를 유지하며, 스타일 품질(stylized image quality)과 내용 일치(content matching)를 기반으로 두 가지 새로운 평가 지표를 제안합니다.



### Unsupervised semantic segmentation of urban high-density multispectral point clouds (https://arxiv.org/abs/2410.18520)
Comments:
          22 pages, 11 figures

- **What's New**: 이 연구는 새로운 고밀도 다중 스펙트럼 공중 레이저 스캔(ALS) 데이터의 의미론적 세분화(semantic segmentation)를 비지도 심층 클러스터링 방법인 GroupSP를 활용하여 제안합니다. 이는 드론을 통해 수집되는 ALS 데이터의 비용이 감소하면서 향후 더 널리 활용될 것으로 보입니다.

- **Technical Details**: GroupSP는 장면을 슈퍼포인트(superpoint)로 나누고, 이를 기반으로 신경망(neural network)을 반복적으로 훈련시켜 같은 클래스의 포인트들을 그룹화합니다. 이 과정에서 GroupSP는 비지도 방식으로 1200점/㎡의 고밀도 포인트 클라우드를 세분화하며 이를 통해 7개의 도시 객체 클래스를 분류하는 데 성공했습니다.

- **Performance Highlights**: GroupSP 방법을 통해 전체 정확도(overall accuracy) 97% 및 평균 교차 면적(Mean Intersection over Union, mIoU) 80%의 성능을 보였습니다. 또한, 적은 수의 주석(points)만으로도 95%의 정확도 및 75%의 mIoU를 달성하며, 다중 스펙트럼 정보가 성능 향상에 기여하는 것으로 확인되었습니다.



### A Note on Geometric Calibration of Multiple Cameras and Projectors (https://arxiv.org/abs/2410.18511)
- **What's New**: 본 연구에서는 여러 개의 카메라와 프로젝터 시스템의 기하학적 보정을 위한 새로운 접근 방식을 다루고 있습니다. 특히, 기존의 방법과 비교해 다중 카메라와 프로젝터의 동시 보정에 집중하고 있습니다.

- **Technical Details**: 본 연구의 핵심 요소로는 보정 객체(calibration object)의 설계와 이미지 처리 파이프라인(image processing pipeline)이 있습니다. 이를 통해 실험실에서 사용하는 표면 이미징 시스템을 보정하는 방법을 제시합니다.

- **Performance Highlights**: 보정 결과는 재투영 오차(reprojection errors) 형태로 제공되며, Zhang의 보정 방법과 같은 고전적인 접근법과 비교하여 유의미한 성과를 보여주고 있습니다.



### Synth4Seg -- Learning Defect Data Synthesis for Defect Segmentation using Bi-level Optimization (https://arxiv.org/abs/2410.18490)
- **What's New**: 이 논문은 결함 세분화(defect segmentation) 문제를 해결하기 위해 혁신적인 이층 최적화(bi-level optimization)에 기반한 합성 결함 데이터 생성(synthetic defect data generation) 프레임워크를 도입합니다. 기존의 방법들이 고정된 규칙에 따라 결함을 생성하는 데 그쳤다면, 이 연구에서는 온라인 합성 결함 생성 모듈을 활용하여 데이터 합성을 최적화합니다.

- **Technical Details**: 제안된 방법은 일반적으로 사용되는 Cut\&Paste 프레임워크를 기반으로 하며, 이층 최적화 문제를 해결하기 위해 효율적인 기울기 기반 최적화(gradient-based optimization) 알고리즘을 채택합니다. 이를 통해 결함 세분화 네트워크와 데이터 합성 모듈의 다양한 매개변수를 동시에 훈련하여, 훈련된 결함 세분화 네트워크의 검증 성능(validation performance)을 극대화합니다.

- **Performance Highlights**: 제한된 데이터 환경에서 실행된 실험 결과에 따르면, 제안된 이층 최적화 방법을 통해 합성 결함을 임의의 위치가 아닌 가장 효과적인 위치에 붙임으로써 세분화 성능이 최대 18.3% 향상되었습니다. 또한 증강(augmentation) 별 결함 데이터 출처에 대한 중요 가중치(importance weights)를 학습함으로써, 모든 데이터 출처에 동등한 중요성을 부여했을 때보다 최대 2.6%의 성능 향상 또한 보여주었습니다.



### Monge-Ampere Regularization for Learning Arbitrary Shapes from Point Clouds (https://arxiv.org/abs/2410.18477)
- **What's New**: 이번 논문에서는 다양한 표면 타입을 모델링할 수 있는 새로운 암시적 표면 표현인 스케일링 제곱 거리 함수(S$^{2}$DF)를 제안합니다. S$^{2}$DF는 내부와 외부 영역을 구분하지 않으면서도, 기존의 비정상적인 UDF의 문제를 해결할 수 있습니다.

- **Technical Details**: S$^{2}$DF는 Monge-Ampere 유형의 2차 편미분 방정식을 만족하며, 이를 통해 원시 방향이 없는 점 구름에서 S$^{2}$DF를 직접 학습하기 위한 새로운 Monge-Ampere 정규화를 개발합니다. 이 방법은 Ground-truth S$^{2}$DF 값에 대한 감독 없이도 학습을 가능하게 합니다.

- **Performance Highlights**: 여러 데이터셋에 걸친 광범위한 실험 결과, 본 방법이 Ground-truth 표면 정보를 감독으로 사용하는 최첨단 감독 방식들에 비해 현저하게 더 우수한 성능을 보여주었습니다. 특히, 본 방법은 더 정밀한 기하학적 세부 사항을 회복하는 것을 입증했습니다.



### Integrating Deep Feature Extraction and Hybrid ResNet-DenseNet Model for Multi-Class Abnormality Detection in Endoscopic Images (https://arxiv.org/abs/2410.18457)
Comments:
          10 pages, 5 figures, CVIP challenge report including the validation results

- **What's New**: 이 논문은 비디오 캡슐 내시경(Video Capsule Endoscopy, VCE) 프레임에서 위장관(Gastrointestinal, GI) 이상을 다중 클래스 분류하기 위한 딥러닝 프레임워크를 제시합니다. 10개 GI 이상 클래스(예: angioectasia, bleeding, ulcers)를 자동으로 식별하여 소화기내과 의사의 진단 부담을 줄이겠다는 목표를 가지고 있습니다.

- **Technical Details**: DenseNet과 ResNet 아키텍처의 앙상블(ensemble) 방법을 이용하여 모델을 구성하였으며, 이를 통해 94%의 평균 정확도를 달성했습니다. 모델은 정규화(normalization) 및 증강(augmentation)과 같은 데이터 전처리 기법을 활용해 다양한 비디오 프레임에 대한 일반화 능력을 향상시켰고, 10개 이상 클래스에 대한 명확한 구분을 목표로 하였습니다.

- **Performance Highlights**: 모델의 정밀도(precision)는 erythema에 대해 0.56에서 worms에 대해 1.00까지 다양하며, 정상 결과에 대한 재현율(recall)은 98%에 도달했습니다. 검토 결과, 전체 정확도는 94%였고, macro-average F1 score는 0.89, micro-average F1 score는 0.94로, 불균형 클래스의 처리 능력이 강하다는 것을 반영합니다.



### Segmentation-aware Prior Assisted Joint Global Information Aggregated 3D Building Reconstruction (https://arxiv.org/abs/2410.18433)
- **What's New**: 이 논문에서는 Multi-View Stereo(MVS) 알고리즘의 약한 텍스처 영역에 대한 새로운 해법을 제안하여 3D 건축 모델의 품질을 향상시키는 것을 목표로 합니다. Segment Anything Model(SAM)과 RANSAC 알고리즘을 기반으로 하는 새로운 알고리즘을 통해, 약한 텍스처 영역을 정확히 분할하고 이를 기반으로 신뢰할 수 있는 plane priors를 생성합니다.

- **Technical Details**: 제안된 알고리즘은 Segment Anything Model(SAM)을 활용하여 약한 텍스처 영역의 이미지를 세분화한 후, RANSAC(Schnabel et al., 2007) 방법과 Delaunay 삼각 분할법을 이용해 여러 plane priors를 생성하는 과정을 포함합니다. 또한, 새로운 글로벌 정보 집계 비용 함수를 도입하여 깊이 추정 업데이트 과정에서 기하학적 일관성을 유지합니다.

- **Performance Highlights**: ETH3D 벤치마크 데이터셋과 실제 상황에서의 실험 결과에 따르면, 본 연구의 방법이 다른 최신 방법들보다 3D 건축 모델 생성에서 뛰어난 성능을 보이는 것으로 나타났습니다. 이는 도시 계획 및 가상 현실 등 다양한 응용 분야에 긍정적인 영향을 미칠 것으로 기대됩니다.



### FreCaS: Efficient Higher-Resolution Image Generation via Frequency-aware Cascaded Sampling (https://arxiv.org/abs/2410.18410)
- **What's New**: 이 논문에서는 고해상도 이미지를 생성하기 위한 효율적인 주파수 인식 계단 샘플링 프레임워크(Frequency-aware Cascaded Sampling framework)인 FreCaS를 소개합니다. FreCaS는 샘플링 과정을 점진적으로 해상도를 높이는 단계로 분해하며, 각 단계에서 주파수 대역을 확장하면서 세부사항을 정제합니다.

- **Technical Details**: FreCaS는 고주파 세부사항 처리를 조기에 불필요한 계산을 피하기 위해, 저주파 내용부터 고주파 세부사항까지 단계적으로 생성하는 조정된 샘플링 전략을 수립합니다. 이를 위해, 주파수 인식 자가 분류기 없는 안내(FA-CFG) 전략을 도입하여 각 주파수 성분에 따라 다양한 안내 강도를 할당합니다. 또한, 이전 단계와 현재 단계의 교차 주의 맵(Cross-Attention Maps)을 융합하여 신뢰할 수 없는 레이아웃 생성을 방지합니다.

- **Performance Highlights**: FreCaS는 이미지 품질과 생성 속도에서 최첨단 방법들보다 월등한 성능을 보여줍니다. 특히, FreCaS는 사전 훈련된 SDXL 모델을 사용하여 2048×2048 이미지를 생성할 때, ScaleCrafter와 DemoFusion보다 각각 2.86배 및 6.07배 더 빠르며, FID 상대 개선은 각각 11.6과 3.7에 달합니다.



### Scale Propagation Network for Generalizable Depth Completion (https://arxiv.org/abs/2410.18408)
Comments:
          Major revision in IEEE Transactions on Pattern Analysis and Machine Intelligence

- **What's New**: 이번 논문에서는 일반화 가능한 깊이 완성을 위한 새로운 방법론인 SP-Norm (Scale Propagation Normalization)을 제안합니다. 기존의 normalization layer가 깊이 완성 모델의 일반화 능력을 제한한다고 분석하고, 입력에서 출력으로 스케일을 효과적으로 전파할 수 있도록 하는 새로운 네트워크 아키텍쳐를 개발했습니다.

- **Technical Details**: SP-Norm은 전통적인 normalization layer의 한계를 극복하기 위해 제안된 방법입니다. 입력 데이터를 정규화하는 대신, 입력의 정규화된 특성을 이용해 단일 레이어 퍼셉트론(Single-layer Perceptron, SLP)을 사용하여 입력을 재조정합니다. 또한, ConvNeXt V2 백본을 기반으로 하는 새로운 네트워크 아키텍쳐를 개발하여 기본 블록과 아키텍쳐의 조합을 탐색합니다.

- **Performance Highlights**: 실험 결과, 제안한 모델은 다양한 유형의 희소 깊이 맵에서 6개의 보지 못한 데이터셋을 대상으로 여타 최신 기법들보다 더 높은 정확도와 더 빠른 속도, 낮은 메모리 사용량을 기록했습니다. 이는 깊이 완성 문제에서의 새로운 기준을 제시합니다.



### DMVC: Multi-Camera Video Compression Network aimed at Improving Deep Learning Accuracy (https://arxiv.org/abs/2410.18400)
- **What's New**: 본 연구에서는 기계 학습 적용을 위해 특별히 설계된 최첨단 비디오 압축 프레임워크인 DMVC를 소개합니다. 기존의 인간 시각 인식을 우선하는 전통적 압축 방식과 달리, 우리의 접근법은 딥 러닝 정확성에 중요한 의미 정보를 보존하면서 데이터 사이즈를 효율적으로 줄이는 데 중점을 둡니다.

- **Technical Details**: DMVC는 다중 비디오 스트림을 동시에 처리할 수 있는 배치 기반으로 작동하며, 경량과 고정밀의 두 가지 복원 모드를 제공합니다. 이 프레임워크는 딥 러닝 알고리즘을 기반으로 필수 정보와 중복을 구분하여 기계 학습 작업에 가장 관련성 높은 데이터를 제공합니다.

- **Performance Highlights**: 다양한 데이터셋에서 얻은 실험 결과에 따르면, DMVC는 기계 학습 작업의 정확성을 유지하거나 개선하면서도 상당한 데이터 압축을 달성했습니다. 이는 스마트 시티 인프라부터 자율 시스템까지 다양한 응용 프로그램에서 큰 잠재력을 보여주며, 기계 학습과 비디오 압축의 통합에 대한 새로운 기준을 수립합니다.



### CloudEye: A New Paradigm of Video Analysis System for Mobile Visual Scenarios (https://arxiv.org/abs/2410.18399)
- **What's New**: 이 논문에서는 CloudEye라는 새로운 모바일 비주얼 인식 시스템을 제안합니다. 이 시스템은 엣지 서버에서의 콘텐츠 정보 마이닝(content information mining)을 이용하여, 클라우드 서버와 협력하여 실시간으로 효율적인 모바일 비주얼 처리(perception)를 수행합니다.

- **Technical Details**: CloudEye 시스템은 Fast Inference Module, Feature Mining Module, Quality Encode Module 세 가지 모듈로 구성되어 있으며, 각 모듈은 네트워크 대역폭 사용량을 69.50% 감소시키고, 추론 속도를 24.55% 증가시키며, 탐지 정확도를 67.30% 향상시킵니다. 이 시스템은 여러 비젼 기반 작업에서 ROI(Region of Interest) 예측과 공간-시간 상관관계를 고려한 비디오 프레임을 동적으로 압축합니다.

- **Performance Highlights**: CloudEye는 실시간 모바일 비주얼 인식 성능을 극대화하며, 네트워크 대역폭의 효율성을 높이는 방향으로 설계되었습니다. 연구를 통해, 이 시스템은 실제 환경에서의 모바일 비전 성능을 크게 개선하였음을 확인하였습니다.



### You Only Look Around: Learning Illumination Invariant Feature for Low-light Object Detection (https://arxiv.org/abs/2410.18398)
Comments:
          Accepted by NeurIPS2024

- **What's New**: 본 논문에서는 저조도 환경에서 물체 감지를 위한 새로운 프레임워크인 YOLA를 제안합니다. 기존 연구와 달리, 우리는 기능 학습(feature learning)의 관점에서 이 문제를 해결하려 합니다.

- **Technical Details**: Lambertian 이미지 형성 모델을 활용하여 조명 불변 특성을 학습하는 접근 방식을 제안합니다. 이 모델 아래에서는 이웃 색상 채널과 공간적으로 인접한 픽셀 간의 상호관계를 활용하여 조명 불변 특성 맵을 근사할 수 있습니다. 우리는 조명 불변 특성을 추출하기 위해 Illumination-Invariant Module (IIM)을 도입하였으며, 이는 기존의 물체 감지 프레임워크에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, IIM의 통합을 통해 저조도 물체 감지의 정확도가 크게 향상되었음을 보여주었습니다. YOLA는 기존 방법들과 비교하여 저조도의 이미지에서 물체 감지 성능을 상당히 개선할 수 있음을 입증했습니다.



### Irregular Tensor Low-Rank Representation for Hyperspectral Image Representation (https://arxiv.org/abs/2410.18388)
- **What's New**: 이번 연구에서는 불규칙한 하이퍼스펙트럼 이미지(HSI)를 처리하기 위한 새로운 불규칙 텐서 저랭크 표현 모델(ITLRR)을 제안합니다. 이 모델은 불규칙한 3D 데이터 큐브를 효율적으로 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 ITLRR 모델은 먼저 HSI 데이터를 몇 개의 불규칙한 동질 지역으로 분할하고, 각 불규칙한 3D 데이터 큐브를 정규 큐브 형태로 보완합니다. 이후, 비볼록 핵 노름(non-convex nuclear norm)을 사용하여 저랭크성을 추구하고, 부정적 글로벌 저랭크 항을 도입하여 전역 일관성을 향상시킵니다. 최종적으로 이 모델은 볼록-오목 최적화 문제로 공식화되고, 대체 증강 라그랑지안 방법을 통해 해결됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 4개의 공개 데이터셋에서 기존의 저랭크 기반 HSI 방법을 크게 초월하는 성능을 보였습니다. 이 연구는 불규칙 텐서의 저랭크 속성을 탐색한 최초의 방법으로, 지역 불규칙 3D 큐브의 저랭크 표현을 위한 비볼록 핵 노름을 도입하고 과매끄럽게 표현되는 문제를 피하기 위한 부정적 저랭크 항을 제안하였습니다.



### Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks (https://arxiv.org/abs/2410.18387)
Comments:
          Technical Report

- **What's New**: 본 연구에서는 MedRegA라는 최초의 이중언어 일반화 의료 AI 시스템을 소개하며, 이미지-레벨 및 지역-레벨 의료 비전-언어 작업을 동시에 처리할 수 있는 기능을 제공합니다. 이는 의료 MLLM에서 해부학적 영역을 이해할 수 있는 능력을 향상시키기 위한 것입니다.

- **Technical Details**: 정확한 지역 인식을 위해, 이 연구는 세 가지 지역 중심 과제를 설정하였습니다: (1) 입체 도형 확인 (Region-to-Text Identification), (2) 구조체 위치 탐지 (Text-to-Region Detection), (3) 기초 보고서 생성 (Grounded Report Generation). 이 과제를 수행하기 위해, 25K 개의 중검 데이터로 구성된 MedRegInstruct 데이터셋을 구축하였습니다. MedRegA는 다양한 의료 이미지를 대상으로 이미지 수준과 지역 수준의 작업을 동시에 수행할 수 있게 설계되었습니다.

- **Performance Highlights**: MedRegA는 기존 최고 성능 모델인 MedDr를 기반으로 8개 모달리티에서 시각 질문 응답, 보고서 생성 및 의료 이미지 분류 수행에서 최고의 성능을 보여주었습니다. 이는 상향성 및 해석가능성을 증가시키면서 의료 MLLM의 사용자 상호작용을 향상시킵니다.



### Real-time 3D-aware Portrait Video Relighting (https://arxiv.org/abs/2410.18355)
Comments:
          Accepted to CVPR 2024 (Highlight). Project page: this http URL

- **What's New**: 본 연구는 Neural Radiance Fields(NeRF)를 기반으로 한 실시간 3D 인식 포트레이트 비디오 리라이트 방법을 제안합니다. 이는 새로운 시점과 조명 조건에서 자연스러운 리라이트를 가능하게 하는 첫 번째 방법으로, 신속한 듀얼 인코더를 사용하여 알베도(tri-plane)와 shading 정보를 추론합니다.

- **Technical Details**: 알베도 트라이플레인과 이에 따른 조명 조건에 대한 shading 트라이플레인을 예측하며, 이를 통해 비디오의 각 프레임에서 빠르고 자연스러운 리라이트 결과를 생성합니다. 또한, 우리는 임시 일관성 네트워크를 활용하여 비디오 내에서의 깜박임 아티팩트를 줄이고 매끄러운 전환을 보장합니다.

- **Performance Highlights**: 소비자 수준의 하드웨어에서 32.98 fps로 실행되며, 재구성 품질, 조명 오류, 조명 불안정성, 임시 일관성 및 추론 속도 면에서 최신 기술에 대한 성능을 달성합니다.



### AVHBench: A Cross-Modal Hallucination Benchmark for Audio-Visual Large Language Models (https://arxiv.org/abs/2410.18325)
Comments:
          URL: this https URL

- **What's New**: 이번 연구에서는 오디오-비주얼 LLMs의 인식 및 이해 능력을 평가하기 위해 AVHBench를 제안합니다. 이는 오디오와 비주얼 신호 간의 미세한 관계를 이해하는 데 어려움을 겪는 현재의 LLMs의 한계를 강조합니다.

- **Technical Details**: AVHBench는 5,816개의 QnA 쌍과 1,238개의 오디오-비주얼 캡션으로 구성된 데이터셋을 포함하며, 주요 작업으로는 Audio-driven Video Hallucination, Video-driven Audio Hallucination, Audio-visual Matching, Audio-visual Captioning이 있습니다. 이 연구는 오디오-비주얼 LLMs가 다중 모드 신호 간의 교차 작용으로 인한 hallucination에 취약함을 나타냅니다.

- **Performance Highlights**: AVHBench를 사용한 평가 결과, 기존의 오디오-비주얼 LLMs는 교차 모드 hallucination에 대해 높은 취약성을 보였으나, 간단한 학습 방법을 통해 robust한 성능을 개선할 수 있음을 보여주었습니다. Low-Rank Adaptation (LoRA) 기법과 enhanced feature alignment의 결합이 효과적이었습니다.



### KhmerST: A Low-Resource Khmer Scene Text Detection and Recognition Benchmark (https://arxiv.org/abs/2410.18277)
Comments:
          Accepted at ACCV 2024

- **What's New**: 이번 논문에서는 캄보디아어(Khmer) 장면 텍스트 인식 및 감지를 위한 최초의 데이터 세트인 KhmerST를 소개합니다. 이 데이터 세트는 1,544개의 전문가 주석 이미지로 구성되어 있으며, 실내(997장) 및 실외(547장) 장면을 포함하고 있습니다.

- **Technical Details**: KhmerST 데이터 세트는 다양한 형태의 텍스트(평면 텍스트, 입체 텍스트, 조명이 부족한 텍스트, 먼 거리의 텍스트, 부분적으로 가려진 텍스트)를 포함하며 각 장면에 대해 선 수준의 텍스트와 다각형 경계 상자 좌표를 제공합니다. 데이터 세트의 문자는 총 33개의 자음, 16개의 의존 모음, 14개의 독립 모음, 13개의 발음 기호로 구성됩니다. 또한, 특유의 문자 조합으로 인해 Khmer 스크립트를 세밀하게 감지하고 인식하는 데 도전이 따릅니다.

- **Performance Highlights**: KhmerST 데이터 세트는 기존의 라틴 문자 벤치마크에 비해 현재의 최첨단 모델들이 Khmer 스크립트에 대해 효과적으로 작동하지 않음을 보여주며, 이 연구는 캄보디아어와 같은 저자원 언어에 대한 학습 알고리즘의 필요성을 강조합니다.



### CARLA2Real: a tool for reducing the sim2real gap in CARLA simulator (https://arxiv.org/abs/2410.18238)
Comments:
          22 pages

- **What's New**: 본 논문에서는 CARLA 시뮬레이터의 출력 결과를 향상시키기 위한 도구인 CARLA2Real을 개발하여, 실제 세계 데이터셋의 시각적 특성에 맞춰 시뮬레이션 데이터를 포토리얼리즘(photorealism)측면에서 개선하였다.

- **Technical Details**: CARLA2Real 도구는 시뮬레이터의 Geometry Buffers(G-Buffers)로부터 중간 정보를 처리하며, 이는 물체의 재질, 조명 및 기하학적 정보를 포함한다. 이 방법은 실시간 처리로서 13 FPS의 프레임 속도를 달성하며 Cityscapes, KITTI 및 Mapillary Vistas와 같은 데이터셋의 시각적 스타일과 현실감을 제공한다.

- **Performance Highlights**: 제안된 접근 방식은 강화된 합성 데이터에 대해 훈련된 특징 추출(feature extraction) 및 의미 세분화(semantic segmentation) 방법의 성능을 평가함으로써 시뮬레이터와 실제 세계 간의 시뮬레이션 갭(sim2real gap)이 줄어드는 것을 보였다.



### MsMorph: An Unsupervised pyramid learning network for brain image registration (https://arxiv.org/abs/2410.18228)
Comments:
          18 pages, 10 figures

- **What's New**: 의료 이미지 분석 분야에서 MsMorph이라는 새로운 딥러닝 기반 이미지 등록 프레임워크를 제안합니다. 이 방법은 수동 등록 프로세스를 모방하여 이미지 쌍의 변형을 더욱 유사하게 만드는 데 중점을 둡니다.

- **Technical Details**: MsMorph는 여러 측면에서 이미지 쌍 간의 특징 차이를 추출하고, 그래디언트(gradients)를 사용하여 다양한 스케일에서 의미(semantic) 정보를 디코딩합니다. 이 프레임워크는 예측된 변형 필드(deformation field)에 대해 지속적으로 보상하여 등록 정확도를 크게 향상시킵니다.

- **Performance Highlights**: 두 개의 공개된 뇌 MRI 데이터셋(LPBA와 Mindboggle)에서 여러 기존 등록 방법과 비교한 결과, Dice score, Hausdorff distance, average symmetric surface distance, non-Jacobian 등의 지표에서 우리의 방법이 일관되게 우수한 성능을 보였습니다.



### Automated Defect Detection and Grading of Piarom Dates Using Deep Learning (https://arxiv.org/abs/2410.18208)
- **What's New**: 본 연구는 Piarom 날짜의 실시간 탐지, 분류, 등급 측정을 위한 혁신적인 딥러닝 프레임워크를 제안합니다. 이 프레임워크는 고해상도 이미지 9,900장을 사용하여 11개의 결함 범주를 식별하며, 기존 AI 기반 솔루션의 한계를 극복합니다.

- **Technical Details**: 제안한 시스템은 Convolutional Neural Networks (CNN)와 최첨단 객체 탐지 알고리즘을 통합하여 결함 식별의 정확성과 신뢰성을 높입니다. 데이터 증강(data augmentation) 기술을 사용하여 모델 성능을 향상시키며, 각 날짜의 면적과 무게를 추정하는 첨단 분할(segmentation) 기법을 적용하여 등급 프로세스를 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안한 시스템은 기존 방법들에 비해 정확성과 계산 효율성을 크게 향상시킵니다. 이 시스템은 산업 요구에 맞춘 실시간 처리가 가능하여 Piarom 날짜 산업의 품질 제어 자동화를 위한 강력하고 확장 가능한 솔루션을 제공합니다.



### Rethinking Positive Pairs in Contrastive Learning (https://arxiv.org/abs/2410.18200)
- **What's New**: 이번 연구에서는 전통적인 대비 학습의 가정을 도전하며, 모든 샘플 쌍을 긍정적인 쌍으로 간주할 수 있는 새로운 접근 방식을 제안합니다. 이로 인해 시맨틱적으로 멀리 떨어진 쌍에서도 의미 있는 학습이 가능해졌습니다.

- **Technical Details**: Hydra라는 보편적인 대비 학습 프레임워크를 통해, 서로 다른 클래스 쌍에서 시각적 표현을 학습합니다. 이는 딥러닝 기법인 SimCLR을 기반으로 하며, 'Feature Filter'라는 새로운 모듈을 도입해 각 클래스 쌍의 하위공간을 정의하고, 이 공간 내에서만 잃을 수 있는 연산을 수행합니다.

- **Performance Highlights**: Hydra는 IN1K 데이터셋에서 500,500개의 클래스 쌍을 활용하여 학습한 결과, 차별적인 성능을 나타내었습니다. 이 접근법은 차원 붕괴를 예방하고, 서로 다른 쌍 간의 공통 특징을 발견하는 데 기여했습니다.



### Personalized Instance-based Navigation Toward User-Specific Objects in Realistic Environments (https://arxiv.org/abs/2410.18195)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track. Project page: this https URL

- **What's New**: 본 논문에서는 Personalized Instance-based Navigation (PIN)이라는 새로운 탐색 과제를 도입하고, 이를 지원하기 위한 전용 데이터셋인 PInNED를 제안합니다. PIN은 에이전트가 여러 유사 객체들 중 특정 개인 객체를 찾고 도달해야 할 때 사용자 개별적으로 해당 객체를 인식해야 하는 문제를 해결합니다.

- **Technical Details**: PInNED 데이터셋은 photo-realistic 환경에서 실제 3D 객체들이 추가된 장면들을 포함하고 있으며, 각 에피소드에서는 에이전트에게 제시된 특정 객체의 이미지와 텍스트 설명을 기반으로 탐색을 수행합니다. 이 데이터셋은 865.5k의 훈련 에피소드와 1.2k의 검증 에피소드를 포함하고 있으며, 에이전트는 주변 맥락에 대한 정보 없이 올바른 인스턴스를 구별해야 합니다.

- **Performance Highlights**: 현재의 탐색 방법들과 비교하여, 모듈형 접근 방식이 end-to-end 접근 방식에 비해 우수한 성능을 보였습니다. 그러나 PIN 과제는 여전히 해결되지 않은 문제로, 추가적인 연구가 필요함을 강조했습니다.



### Advancing Super-Resolution in Neural Radiance Fields via Variational Diffusion Strategies (https://arxiv.org/abs/2410.18137)
Comments:
          All our code is available at this https URL

- **What's New**: 이 논문은 Neural Rendering에서 view-consistent super-resolution(SR)을 위한 diffusion-guided 프레임워크에 대한 새로운 방법론을 제시합니다. 기존의 2D SR 모델과 Variational Score Distilling(VSD), LoRA fine-tuning helper와 같은 고급 기술을 결합하여 2D 이미지의 품질과 일관성을 대폭 향상시켰습니다.

- **Technical Details**: 이 방법론은 Iterative 3D Synchronization(I3DS)을 통합하여 독립적인 SR 2D 이미지 간의 불일치 문제를 해결합니다. 새로운 Variational Score Distillation(VSD) 기법을 도입하여 3D 장면 매개변수를 확정된 값들이 아니라 확률 분포로 모델링하며, 이 과정에서 low-rank adaptation(LoRA)을 사용하여 사전 훈련된 모델의 효율적인 미세 조정을 가능하게 합니다. 또한 UNet 아키텍처를 사용해 이미지 복원을 수행합니다.

- **Performance Highlights**: LLFF 데이터셋에 대한 정량적 벤치마크와 정성적 결과를 통해, 우리의 시스템이 기존의 DiSR-NeRF와 같은 방법들에 비해 우수한 성능을 보여주었다는 것을 입증하였습니다. 이로 인해 본 연구가 게임, 과학적 시각화 및 건축 설계 등 다양한 분야에서 고해상도 3D 장면 생성을 위한 새로운 기준이 될 것으로 기대됩니다.



### Point Cloud Compression with Bits-back Coding (https://arxiv.org/abs/2410.18115)
Comments:
          This paper is under reviewed in IEEE Robotics and Automation Letters

- **What's New**: 이 논문은 bits-back coding 방식을 활용하여 포인트 클라우드 데이터의 기하학적 속성을 손실 없이 압축하는 새로운 방법을 제안합니다. 이 방법은 심층 학습 기반 확률 모델을 사용하여 포인트 클라우드 정보의 Shannon 엔트로피를 추정하는 데 중점을 두고 있습니다.

- **Technical Details**: 포인트 클라우드 압축을 위한 새로운 접근 방식으로 convolutional variational autoencoder (CVAE)를 이용해 Shannon의 엔트로피를 추정합니다. bits-back coding 기법을 통해 CVAE의 학습된 잠재 변수 모델을 활용하여 데이터 포인트 간의 잠재적인 상관관계를 포착하고 압축 비율을 줄입니다.

- **Performance Highlights**: 제안하는 방법은 평균적으로 1.56 bits-per-point의 압축 비율을 달성하며, 구글의 Draco와 같은 기존 접근 방식의 1.83 bits-per-point에 비해 상당히 낮은 비율을 보입니다. 또한, 압축 코덱의 저장 및 통신 오버헤드를 크게 줄이는 데 성공하여 실용적인 응용 분야에 적용 가능성을 높입니다.



### NaVIP: An Image-Centric Indoor Navigation Solution for Visually Impaired Peop (https://arxiv.org/abs/2410.18109)
Comments:
          40 pages, 20 figures

- **What's New**: 본 논문에서는 시각장애인(VIPs)들을 위한 실내 내비게이션을 위한 이미지 데이터셋(NaVIP)과 이미지 중심 솔루션을 제안합니다. 이는 인프라 요구 없이 자동으로 환경을 이해할 수 있도록 돕는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: NaVIP 데이터셋은 연구 건물의 4개 층에서 수집된 300K개의 이미지를 포함하며, 정확한 6DoF 카메라 포즈, 실내 포인트 오브 인터레스(Points of Interest, PoIs) 정보 및 설명 캡션으로 라벨링되어 있습니다. 이 데이터는 실시간 추론과 훈련 확장성을 기준으로 벤치마킹되었습니다.

- **Performance Highlights**: NaVIP의 구현을 통해 카메라 포즈를 정확하게 추정하고, 이미지 캡셔닝 기법을 활용하여 VIPs의 탐색을 지원할 수 있음을 보여줍니다. 실제 건물 레이아웃에서 서브 미터 정확도로 위치 지정을 수행할 수 있는 성능을 보였습니다.



### A Deep Learning Approach to Estimate Canopy Height and Uncertainty by Integrating Seasonal Optical, SAR and Limited GEDI LiDAR Data over Northern Forests (https://arxiv.org/abs/2410.18108)
- **What's New**: 이 연구는 북부 고위도 지역의 식생 모니터링 및 탄소 저장량 평가를 위한 고해상도 캐노피 높이 추정 방법론을 제시합니다. 기존의 우주 기반 LiDAR 기술의 한계를 극복하고, 다양한 위성 데이터와 깊이 학습 회귀 모델을 통합하여 공간적으로 연속적인 캐노피 높이 및 불확실성 추정치를 생성하는 방법을 개발했습니다.

- **Technical Details**: 이 연구는 Sentinel-1, Landsat, ALOS-PALSAR-2의 다계절 위성 데이터와 우주 기반 GEDI LiDAR를 참조 데이터로 통합하여 캐노피 높이를 추정했습니다. 온타리오, 캐나다에서 시험 및 공중 LiDAR로 검증하였으며, 0.72의 R-square, 3.43 m의 RMSE, 2.44 m의 bias를 기록했습니다. 계절별 데이터를 활용하여 변동성을 10% 향상시키고 오류를 0.45 m 감소시켰습니다.

- **Performance Highlights**: 깊이 학습 모델의 가중치 전략은 높은 캐노피 높이 추정 시 오류를 크게 줄였습니다. 그러나 낮은 캐노피 높이는 과대 추정하였습니다. 불확실성 지도는 숲의 가장자리 근처에서 큰 불확실성을 강조했으며, 이는 GEDI 측정에서 오류가 발생하기 쉬운 구역입니다.



### RingGesture: A Ring-Based Mid-Air Gesture Typing System Powered by a Deep-Learning Word Prediction Framework (https://arxiv.org/abs/2410.18100)
- **What's New**: 이 논문은 경량 증강 현실(AR) 안경을 위한 새로운 텍스트 입력 시스템인 RingGesture를 제안합니다. 이는 제스처 경로의 시작과 끝을 표시하는 전극과 손 추적을 위한 관성 측정 장치(IMU) 센서를 활용하여 공중 제스처 타이핑을 가능하게 합니다.

- **Technical Details**: RingGesture 시스템은 심층 학습 기반의 새로운 단어 예측 프레임워크인 Score Fusion을 통합하여 정확성과 입력 속도를 향상시킵니다. Score Fusion은 단어-제스처 디코딩 모델, 공간 철자 수정 모델, 경량 컨텍스트 언어 모델 등 세 가지 주요 구성 요소로 구성됩니다.

- **Performance Highlights**: RingGesture는 평균 27.3 단어/분(WPM)의 텍스트 입력 속도를 기록했으며, 최고 성능은 47.9 WPM입니다. Score Fusion 프레임워크는 전통적인 단어 예측 프레임워크인 Naive Correction보다 28.2% 더 나은 성능을 보여주었으며, RingGesture의 텍스트 입력 속도는 55.2% 향상되었습니다. 또한, 시스템 사용성 점수는 83점으로, 매우 우수한 사용성을 나타냅니다.



### Gesture2Text: A Generalizable Decoder for Word-Gesture Keyboards in XR Through Trajectory Coarse Discretization and Pre-training (https://arxiv.org/abs/2410.18099)
- **What's New**: 이 논문에서는 확장 현실(Extended Reality, XR) 환경에서의 직관적인 텍스트 입력을 위한 단어-제스처 키보드(Word-Gesture Keyboard, WGK) 시스템에 대한 새로운 신경 디코더(Neural Decoder)를 제안합니다. 새로운 접근법은 대규모의 간략히 이산화된(word-gesture trajectories) 데이터를 미리 학습하여 다양한 환경에서 일반화할 수 있는 성능을 보여줍니다.

- **Technical Details**: 합성곱 신경망(convolutional neural networks)과 사전 학습(pre-training) 기법을 통해 기존의 템플릿 일치(template-matching) 디코더인 SHARK^2의 한계를 극복하려고 하며, 입력 경로(trajectory)를 인코딩하는 구조적 표현 E(g)를 사용합니다. 이 시스템은 현실 증강(AR)과 가상 현실(VR)에서의 WGK 시스템에 적용 가능하며, 단순한 설치와 함께 높은 디코딩 정확도를 제공합니다.

- **Performance Highlights**: 이 새로운 사전 학습된 신경 디코더는 평균적으로 90.4%의 Top-4 정확도를 달성하였고, SHARK^2보다 37.2% 높은 성능을 보여주며, 전통적인 신경 디코더보다도 7.4% 향상된 결과를 보입니다. 또한, 이 디코더는 저용량(4 MB)으로 실시간(97ms) 처리 속도를 가지며 정확도를 희생하지 않았습니다.



### TextureMeDefect: LLM-based Defect Texture Generation for Railway Components on Mobile Devices (https://arxiv.org/abs/2410.18085)
Comments:
          6 Pages, 8 figures

- **What's New**: 이번 연구에서는 산업 응용을 위한 컨텍스트 기반의 현실적인 텍스처 생성의 새로운 가능성을 제시합니다. 특히, 철도 부품의 결함 텍스처 생성을 위한 TextureMeDefect라는 모바일 친화적인 도구를 소개합니다.

- **Technical Details**: TextureMeDefect는 LLM(대형 언어 모델) 기반의 AI-Inferencing 엔진을 활용하여 사용자가 스마트폰이나 태블릿으로 촬영한 철도 부품 이미지에서 상호작용적으로 현실적인 결함 텍스처를 생성할 수 있게 합니다.

- **Performance Highlights**: TextureMeDefect는 전통적인 이미지 생성 도구를 초월하여 의미 있는 텍스처를 더 빠르게 생성하였으며, iOS 및 Android 플랫폼에서의 시간과 비용 효율성을 평가하였습니다. 또한, 세 가지 시나리오에서 소프트웨어의 사용성 점수(SUS)를 분석하였습니다.



### Stable Consistency Tuning: Understanding and Improving Consistency Models (https://arxiv.org/abs/2410.18958)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문은 일관성 모델의 새로운 이해 프레임워크를 제안하며, 이를 확산 모델의 denoising 과정을 Markov Decision Process (MDP)로 모델링하는 방식으로 접근합니다. 기존의 Easy Consistency Tuning (ECT)을 기반으로 한 Stable Consistency Tuning (SCT)도 소개되어, 성능 향상을 위한 다양한 개선 사항이 포함되어 있습니다.

- **Technical Details**: 일관성 모델의 훈련 방법은 카운터-수렴 조건(self-consistency condition)을 강화하는 것을 목표로 하며, 두 가지 방법인 consistency distillation (CD)과 consistency training/tuning (CT)을 활용합니다. 이 논문은 consistency 모델의 훈련을 Temporal Difference (TD) Learning의 관점에서 해석하는 새로운 방법론을 제시하며, SCT는 score identity를 이용한 분산 감소 학습을 포함합니다. 또한, multi-step sampling을 위해 deterministic multistep sampling 방법을 제안하고, classifier-free guidance의 효과성도 검증합니다.

- **Performance Highlights**: 테스트 결과 SCT는 CIFAR-10 및 ImageNet-64와 같은 벤치마크에서 상당한 성능 향상을 보이며, ImageNet-64에서 1-step FID 2.42 및 2-step FID 1.55로 일관성 모델의 신규 SoTA(최첨단)를 달성했습니다.



### ANAVI: Audio Noise Awareness using Visuals of Indoor environments for NAVIgation (https://arxiv.org/abs/2410.18932)
Comments:
          8th Conference on Robot Learning (CoRL) 2024

- **What's New**: 이번 연구에서는 로봇의 경로 계획에서 소음 인식을 위해 실내 비주얼을 활용하는 새로운 방법을 제안합니다. 기존의 로봇들이 소음을 인식하지 못하는 문제를 해결하기 위해, 로봇이 주변 환경의 시각적 관찰을 통해 소음 수준을 예측하고 이를 기반으로 보다 조용한 경로를 계획할 수 있도록 Acoustic Noise Predictor (ANP) 모델을 개발하였습니다.

- **Technical Details**: 로봇이 움직일 때 발생하는 소음(예: 모터 소리, 스피커의 소리)을 고려하여 소음의 세기를 예측하는 ANP 모델을 구축했습니다. 시뮬레이션된 환경에서 로봇의 위치에 따라 청취자 위치에서 소리의 세기가 어떻게 변하는지를 학습하며, 다양한 실내 환경에서의 3D 스캔(Matterport) 데이터를 활용하여 소음 예측을 위한 훈련을 진행했습니다. ANAVI 프레임워크를 통해 로봇은 자신의 행동이 주변에 미치는 소음을 인식하고 이에 적응하여 더 조용한 경로를 찾을 수 있습니다.

- **Performance Highlights**: ANP 모델은 거리 기반 추정 방식보다 더 나은 예측 정확도를 보였으며, 시뮬레이션과 실제 실험에서 일관된 결과를 나타내었습니다. ANP의 섬세한 예측은 소리의 반사, 회절, 흡수 등의 건축적 특성과 재료의 영향을 효과적으로 반영하였습니다. 이러한 연구는 로봇이 소음 문제를 해결하면서도 효율성을 유지할 수 있는 방향성을 제시합니다.



### SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deploymen (https://arxiv.org/abs/2410.18907)
- **What's New**: 이 논문에서는 몇 가지 인간 시연을 바탕으로 로봇 조작을 위한 대규모 데이터세트를 생성할 수 있는 자동화 시스템인 SkillMimicGen(SkillGen)을 제안합니다.

- **Technical Details**: SkillGen은 인간 시연을 조작 기술(manipulation skills)로 분할하고, 이러한 기술을 새로운 맥락에 적응시키며, 자유공간 전이(free-space transit) 및 전이 동작(transfer motion)을 통해 통합합니다. 또한, Hybrid Skill Policy(HSP) 프레임워크를 제안하여 SkillGen 데이터셋에서 기술의 시작, 제어 및 종료 컴포넌트를 학습할 수 있도록 합니다.

- **Performance Highlights**: SkillGen은 최신 데이터 생성 프레임워크에 비해 데이터 생성 및 정책 학습 성능을 크게 향상시키며, 다양한 장면 변형(scene variations)에 대한 데이터를 생성할 수 있는 능력을 보여줍니다. 24K 이상의 시연을 생성하고, 평균적으로 24% 더 성공적인 HSP 에이전트를 훈련시키는 데 성공했습니다. 또한 실제 조작 작업에 적용하여 긴 수평 조립 작업에서 제로샷 시뮬레이션-리얼 전이를 보여주었습니다.



### Highly efficient non-rigid registration in k-space with application to cardiac Magnetic Resonance Imaging (https://arxiv.org/abs/2410.18834)
- **What's New**: 본 연구에서는 새로운 자기지도(self-supervised) 딥러닝 기반의 프레임워크인 Local-All Pass Attention Network (LAPANet)를 소개합니다. 이 네트워크는 압축된 Fourier 공간(k-space)에서 비강체(non-rigid) 운동 추정을 직접 수행하여 높은 시간 해상도를 달성합니다.

- **Technical Details**: LAPANet는 국소적인 변위를 누적합으로 모델링하는 방식으로, Local All-Pass (LAP) 등록 기법을 따릅니다. 이 네트워크는 전체 k-space를 이용하여 다중 해상도 수준에서 정보를 통합함으로써 빠르고 정확한 운동 추정을 가능하게 하며, 고속 샘플링 경로에서도 효과를 보여줍니다.

- **Performance Highlights**: 전통적인 방법들과 비교하여, LAPANet는 Cartesian 궤적에서 시간 프레임당 22라인(R=78, 4.24 ms) 또는 비-Cartesian 궤적에서 33스포크(R=104, 4.99 ms)의 적은 데이터로도 높은 정확도의 비강체 운동 추정을 가능하게 하여 뛰어난 성능을 입증했습니다.



### Single-Shot Phase Diversity Wavefront Sensing in Deep Turbulence via Metasurface Optics (https://arxiv.org/abs/2410.18789)
- **What's New**: 이 논문에서는 자유공간 광 통신(FSOC) 시스템의 성능을 개선하기 위해 나노구조 이중굴절 메타서피스(optic) 기술을 이용한 새로운 파면 센서(wavefront sensor)를 소개합니다. 기존의 파면 센서는 긴 거리와 깊은 난기류(공기에서의 불규칙한 움직임)에서 성능 저하가 발생하였으나, 본 연구에서 제안하는 방법은 이러한 문제를 해결합니다.

- **Technical Details**: 나노구조 이중굴절 메타서피스(optic)를 통한 저지연(low-latency) 파면 감지 기술은 파 phase diversity를 활용하여 깊은 난기류 조건에서 유효한 파면을 복원할 수 있습니다. 이를 통해 FHOC 시스템에서의 통신 성능을 향상시킵니다.

- **Performance Highlights**: 모의실험과 실험 시연을 통해 평균 16배의 신호 증가(signal increase)를 확인하였습니다. 이 방법은 FSOC 시스템의 범위와 정확성을 향상시키며, 컴팩트(compact)하고 강력한 파면 감지 시스템을 가능하게 합니다.



### Transferring Knowledge from High-Quality to Low-Quality MRI for Adult Glioma Diagnosis (https://arxiv.org/abs/2410.18698)
Comments:
          Technical Report, MICCAI 2024 BraTS-SSA Challenge Runner Up

- **What's New**: 본 논문은 아프리카 사하라 이남에서의 신경영상 데이터의 한계를 극복하고 뇌종양 진단을 개선하기 위한 접근 방식을 제시합니다. BraTS-GLI 2021 대회의 기존 모델을 활용하여 세 가지 훈련 전략을 적용했습니다.

- **Technical Details**: 이 연구에서는 RSNA-ASNR-MICCAI BraTS-GLI 2021 데이터셋과 BraTS-Africa Challenge 데이터셋을 사용합니다. nnU-Net 기반의 세그멘테이션 모델을 적용하고, 크게 조정된 네트워크와 그룹 정규화를 통해 학습 성능을 향상시켰습니다. 두 가지 주요 수정 사항은 인코더의 필터 수를 두 배로 늘리고 배치 정규화를 그룹 정규화로 대체하는 것입니다.

- **Performance Highlights**: 최종 모델은 Dice 점수 0.882, 0.840, 0.926 및 Hausdorff 거리 15.324, 37.518, 13.971을 달성하여 종양, 종양 핵 및 전체 종양의 세분화 성능을 보여줬습니다. 대회 최종 단계에서 총 2위를 차지했습니다.



### Moving Object Segmentation in Point Cloud Data using Hidden Markov Models (https://arxiv.org/abs/2410.18638)
Comments:
          Accepted to the IEEE IROS 2024 workshop on Long-Term Perception for Autonomy in Dynamic Human-shared Environments: What Do Robots Need?

- **What's New**: 이 논문은 자율 에이전트가 환경 내에서 동적 물체를 식별할 수 있는 새로운 접근법을 제안합니다. 특히, 숨겨진 마르코프 모델(hidden Markov model, HMM)을 기반으로 하는 학습 필요 없는 방법을 통해 포인트 클라우드 데이터에서 이동 물체를 세그먼트화하는 방법을 제시합니다.

- **Technical Details**: 제안하는 알고리즘은 세 가지 단계로 구성됩니다. 먼저, HMM을 사용하여 각 복셀(voxel)의 점유 상태를 모델링하고, 관측값을 통해 점유 상태를 업데이트합니다. 또한, 복셀의 상태 전이 행렬은 높은 자기 전이 확률을 가져, 신뢰성이 충분해야 전이됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 센서 특성과 환경을 포함한 벤치마크 데이터셋에서 기존의 최첨단 방법들과 비교하여 일관되게 더 좋은 성능을 보여주었습니다. 이는 각각의 동적 물체에 대해 일반화된 성능을 입증하였으며, 오픈소스 프로젝트로 제공됩니다.



### Rethinking Softmax: Self-Attention with Polynomial Activations (https://arxiv.org/abs/2410.18613)
- **What's New**: 이 논문은 transformers 에서 softmax attention 의 효과가 attention allocation 을 위한 확률 분포 생성에만 국한되지 않음을 이론적으로 설명합니다. 대신, softmax attention 의 성공은 훈련 과정에서 attention matrix 의 Frobenius norm 을 암묵적으로 정규화하는 능력에 있음을 밝혔습니다.

- **Technical Details**: 이론적 프레임워크를 통해, 우리는 polynomial activations 를 탐구하여 이것이 attention matrix 의 Frobenius norm 을 정규화하는 데 효과적일 수 있음을 증명합니다. 이러한 대안적 activation 은 softmax 의 조건 몇 가지를 위반할 수 있지만, 여전히 훈련 중 Frobenius norm 을 정규화합니다.

- **Performance Highlights**: 다양한 컴퓨터 비전 및 자연어 처리(NLP) 과제를 통해, 새로운 activations 가 softmax 보다 동등하거나 더 나은 성능을 보임을 보여주며, softmax 를 넘어서는 attention 메커니즘에 대한 새로운 가능성을 시사합니다.



### A Joint Representation Using Continuous and Discrete Features for Cardiovascular Diseases Risk Prediction on Chest CT Scans (https://arxiv.org/abs/2410.18610)
Comments:
          23 pages, 9 figures

- **What's New**: 이 논문에서는 심혈관 질환(CVD)의 위험 예측을 개선하기 위해, 흉부 CT 스캔에서 추출한 연속적 특성과 이산적 정량적 바이오마커를 통합하는 새로운 방법론 DeepCVD를 제안합니다.

- **Technical Details**: DeepCVD는 CT 스캔 이미지에서 심층 학습(Deep Learning)으로 파생된 연속적 특성과 의학적으로 검증된 이산적 정량적 바이오마커를 통합하는 정보를 융합하는 방법론입니다. 이 과정에서 인스턴스별 특성 게이팅 메커니즘(Instance-wise Feature Gating Mechanism, IFGM)과 소프트 인스턴스별 특성 상호작용 메커니즘(Soft Instance-wise Feature Interaction Mechanism, SIFIM)을 사용하여 특성을 정렬하고 상호작용을 지원합니다.

- **Performance Highlights**: DeepCVD는 공공 데이터셋과 사설 데이터셋에서 우수한 예측 성능을 달성하였으며, LDCT-NLST 데이터셋에서 AUC 0.875을 달성하고 NERC-MBD 데이터셋에서 AUC 0.843을 기록했습니다. 이 모델은 의사가 특정 바이오마커의 기여도를 분석하는 데 유용하며, 심혈관 질환 위험 예측의 정확도를 크게 향상시키고 있습니다.



### Learn 2 Rage: Experiencing The Emotional Roller Coaster That Is Reinforcement Learning (https://arxiv.org/abs/2410.18462)
- **What's New**: 이번 논문은 2022 AIcrowd에서 주최한 Learn To Race Autonomous Racing Virtual Challenge에서 팀의 수상작에 대한 실험 및 솔루션 개요를 소개합니다. 이 대회의 목표는 자율주행 기술의 한계를 확장하고 안전성을 달성하는 데 중점을 두고 있습니다. 본 연구에서는 Soft Actor Critic (SAC) 변형을 구현하여 경쟁에 참가하였고, 시각적 및 기하학적 특성만을 기반으로 차량 제어를 배우는데 성공했습니다.

- **Technical Details**: 연구는 강화학습 (Reinforcement Learning, RL) 문제를 다루며, 초점은 RGB 이미지에서 직접 행동을 예측하는 것입니다. SAC 알고리즘을 사용하여 차량 제어 명령을 학습하고, VAE(Variational Auto-Encoder)를 통해 입력 이미지를 압축하여 주요 정보만 활용했습니다. 실시간 시뮬레이션 환경에서 향상된 보상 정책을 통해 부드러운 조향 및 가속 제어를 촉진했습니다.

- **Performance Highlights**: 이 시스템은 훈련이 적고 설명성이 뛰어나며 일반화 성능이 우수하여, 대회에서 다른 모든 에이전트를 큰 차이로 능가했습니다. 특히 첫 번째 카메라 입력 기준에서 1위를 기록하고, 전체 트랙 시간에서도 가장 빠른 기록을 달성했습니다.



### Uncertainty-Error correlations in Evidential Deep Learning models for biomedical segmentation (https://arxiv.org/abs/2410.18461)
Comments:
          15 pages

- **What's New**: 본 연구는 Evidential Deep Learning (EDL)라는 불확실성 정량화 (uncertainty quantification) 프레임워크가 생물의학 이미지 분할 (biomedical image segmentation)에 적용되었을 때의 효과를 분석하였습니다. 특히, U-Net 기반의 모델을 사용하여 모델의 예측 오차와 불확실성 간의 상관관계를 비교하였습니다.

- **Technical Details**: EDL 모델은 분할 레이블에 대한 사전 확률로 다이리클렛 분포 (Dirichlet distribution)를 사용하며, 다양한 모델 불확실성 정의를 가능하게 합니다. 연구에 사용된 데이터셋은 Medical Segmentation Decathlon에서 제공하는 심장 및 전립선 MRI 이미지입니다. EDL 모델은 훈련 후 불확실성 히트맵 (uncertainty heatmaps)을 생성하며, 이는 모델이 특정 예측에 대해 얼마나 신뢰하는지를 시각화합니다.

- **Performance Highlights**: EDL 모델은 전통적인 샤논 엔트로피 (Shannon entropy) 기반의 기법 대비 더 우수한 예측 오류-불확실성 상관관계를 나타내며, 유사한 Dice-Sorensen 지수를 달성했습니다. 이는 EDL 모델이 중요한 생물의학적 분할 작업에서 대규모 모델 오류 감지를 위한 높은 민감성을 제공함을 나타냅니다.



### Multi-Stage Airway Segmentation in Lung CT Based on Multi-scale Nested Residual UN (https://arxiv.org/abs/2410.18456)
- **What's New**: 이번 연구에서는 폐 CT 이미지에서 기도(segmentation)를 정확하고 완전하게 수행하기 위해 Multi-scale Nested Residual U-Net (MNR-UNet)이라는 새롭게 설계된 구조를 제안합니다. 이 구조는 다양한 스케일의 입력과 Residual Multi-scale Modules (RMM)를 포함하여 정보 흐름을 향상시킵니다.

- **Technical Details**: MNR-UNet은 다중 스케일 정보를 결합하고 잔여 연결(residual connections)을 통해 세밀한 기도 구조 디테일을 포착하며, gradient vanishing 문제를 감소시킵니다. 또한, Weighted Breakage-Aware Loss (wBAL)을 도입하여 기도의 연속성을 더욱 섬세하게 유지하도록 설계된 세 단계의 세그멘테이션 파이프라인을 따릅니다.

- **Performance Highlights**: 개발된 접근법이 공공 데이터셋과 자체 데이터셋 모두에서 실행된 검증 결과, 기존 방법 대비 기도 분할의 기술적 완전성과 세밀한 기도 구조 추출에서 크게 향상된 성과를 보여주었습니다. 또한, 기도의 브랜치를 더욱 정확하게 식별하는 데에 성공했습니다.



### WAFFLE: Multi-Modal Model for Automated Front-End Developmen (https://arxiv.org/abs/2410.18362)
- **What's New**: 본 논문에서는 UI 디자인 이미지를 HTML 코드로 변환할 때의 두 가지 주요 도전 과제를 해결하기 위한 새로운 미세 조정 전략인 Waffle을 소개합니다. Waffle은 구조 인식을 위한 주의 메커니즘과 대조적 미세 조정 접근 방식을 활용하여 LLMs의 이해도를 향상시킵니다.

- **Technical Details**: Waffle은 LLMs가 HTML 코드의 구조와 UI 이미지 간의 관계를 효과적으로 학습할 수 있도록 돕는 구조 인식 주의 메커니즘을 설계합니다. 이는 HTML 코드의 이전 세그먼트에 대한 주의를 허용하여 모델이 가장 관련성 높은 부분에 집중할 수 있도록 합니다. 또한, 대조적 학습 기법을 통해 모델이 UI 이미지의 미세한 시각적 차이를 인식하도록 돕습니다. 이 방식은 HTML 코드와 렌더링된 UI 디자인 간의 복잡한 관계를 명확히 이해하는 데 기여합니다.

- **Performance Highlights**: 새로운 벤치마크인 WebSight-Test와 기존 벤치마크인 Design2Code를 통해 Waffle을 적용한 모델은 기존 미세 조정 기법에 비해 9.00 pp의 HTML 일치도 향상, 0.0982의 CW-SSIM 증가, 32.99의 CLIP 점수 증가, 27.12 pp의 LLEM 향상을 기록했습니다. 이는 Waffle이 모델 독립적으로 작동하여 다양한 MLLMs을 개선하는 데 적용할 수 있음을 보여줍니다.



### Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared images (https://arxiv.org/abs/2410.18340)
Comments:
          Published in IEEE Robotics and Automation Letters (2024)

- **What's New**: 이 논문에서는 Thermal Chameleon Network (TCNet)라는 새로운 기술을 제안합니다. TCNet은 고유의 14비트 TIR 이미지를 특정 작업에 맞춰 어댑티브(Adaptive)하게 톤 맵핑(tone-mapping)할 수 있는 방법을 제공합니다.

- **Technical Details**: TCNet는 멀티채널 열 매핑(multi-channel thermal embedding)과 적응형 채널 압축(adaptive channel compression) 두 단계를 통해 동작합니다. 이는 하나의 RAW TIR 이미지를 여러 개의 열 표현으로 변환하여 각 작업에 최적화된 이미지를 생성합니다.

- **Performance Highlights**: TCNet은 객체 감지(object detection) 및 단안 깊이 추정(monocular depth estimation) 작업에서 성능을 크게 향상시켰습니다. 이 방법은 최소한의 계산 오버헤드와 기존 구조에 모듈식으로 통합할 수 있는 장점을 가지고 있습니다.



### Calibrating Deep Neural Network using Euclidean Distanc (https://arxiv.org/abs/2410.18321)
- **What's New**: 본 연구에서는 Focal Loss와 전통적인 보정 손실(Proper Loss)을 결합한 Focal Calibration Loss (FCL)을 제안하여 확률 보정을 개선하고, 어려운 샘플 처리에서 Focal Loss의 장점을 유지합니다.

- **Technical Details**: FCL은 클래스-사후 확률(class-posterior probabilities)을 올바르게 정렬하여 과신(overconfidence) 및 과소신(underconfidence)을 효과적으로 완화합니다. 이 손실을 최소화함으로써 생성된 분류기는 우수한 확률 및 분류 보정을 가지며, Focal Loss 단독 사용 시보다 낮은 사후 처리 격차(post-processing gap)를 보입니다.

- **Performance Highlights**: 여러 모델 및 데이터셋에 대한 광범위한 평가 결과, FCL은 보정(calibration) 및 정확도(accuracy) 메트릭 모두에서 SOTA(State Of The Art) 성능을 달성하였습니다. 또한, FCL은 Chest X-ray 영상 응용에서 이상 탐지(anomaly localization), 인식(recognition), 강건성(robustness)을 향상시킵니다.



### E2E-Swin-Unet++: An Enhanced End-to-End Swin-Unet Architecture With Dual Decoders For PTMC Segmentation (https://arxiv.org/abs/2410.18239)
- **What's New**: 이번 연구에서는 효율적인 유두암 미세암종 (PTMC) 관리를 위해 개발된 AI 기반의 세분화 모델 E2E-Swin-Unet++를 소개합니다. 이 모델은 초음파 B-모드 이미징을 개선하여 PTMC 종양을 실시간으로 식별하고 세분화할 수 있게 합니다.

- **Technical Details**: E2E-Swin-Unet++는 Swin-Unet 아키텍처의 향상된 엔드투엔드 확장으로, 갑상선 영역 정보를 통합하여 PTMC 세분화의 잘못된 가능성을 줄이면서 신속한 추론 기능을 제공합니다. 전통적인 초음파 B-모드 이미징 기술의 한계를 극복하기 위해 설계되었습니다.

- **Performance Highlights**: 실제 임상 RFA 데이터셋에 대한 실험 결과, E2E-Swin-Unet++는 관련 모델들과 비교하여 우수한 성능을 보여주었으며, 이러한 솔루션은 RFA 절제 치료의 정밀도 및 제어력을 크게 향상시킵니다.



### Bridging the Diagnostic Divide: Classical Computer Vision and Advanced AI methods for distinguishing ITB and CD through CTE Scans (https://arxiv.org/abs/2410.18161)
Comments:
          9 pages, 3 figures, 3 algorithms

- **What's New**: 이 연구에서는 Computed Tomography Enterography (CTE) 스캔과 딥러닝, 전통적인 컴퓨터 비전을 활용하여 Intestinal Tuberculosis (ITB)와 Crohn's Disease (CD)의 진단을 자동화하는 새로운 알고리즘을 제안합니다. 이 알고리즘은 피하 지방을 자동으로 분할하여 VF/SF 비율을 계산하고, 이전의 수동 계산 방식보다 효율성과 객관성을 향상시킵니다.

- **Technical Details**: 이 논문에서는 Hounsfield 단위(HU)를 기반으로 한 고전적인 컴퓨터 비전 알고리즘을 개발하여 피하 지방과 내장 지방의 분할을 수행하고, TotalSegmentator 및 수동 측정과 결과를 비교합니다. ResNet10 모델을 사용해 ITB와 CD를 식별하기 위한 데이터셋을 학습하고, Grad-CAM 기법을 활용하여 진단 결과를 설명합니다.

- **Performance Highlights**: 이 알고리즘은 75%의 정확도를 달성하였으며, 3D CT 이미지를 사용한 성능 비교도 수행하였습니다. 작은 데이터셋(100 사례)으로 인해 전문가들은 딥러닝 모델보다 기능 기반의 점수 시스템을 더 신뢰할 수 있다고 언급하였습니다.



### R2Gen-Mamba: A Selective State Space Model for Radiology Report Generation (https://arxiv.org/abs/2410.18135)
Comments:
          4 pages pages for ISBI2025

- **What's New**: 본 연구는 Mamba의 효율적인 시퀀스 처리와 Transformer 구조의 맥락적 이점을 활용한 새로운 자동 방사선 보고서 생성 방법인 R2Gen-Mamba를 제안합니다.

- **Technical Details**: R2Gen-Mamba는 Mamba 모델을 인코더로, Transformer를 디코더로 활용하며, 이는 낮은 계산 복잡도로 인해 학습 및 추론 효율성을 향상시킵니다. 이미지 패치의 특징을 입력으로 사용하고, 이에 대한 보고서를 생성하는 시퀀스-시퀀스 접근 방식을 채택합니다. Mamba의 선형 계산 복잡성 덕분에 높은 품질의 보고서를 생성할 수 있습니다.

- **Performance Highlights**: R2Gen-Mamba는 210,000개 이상의 X-ray 이미지-보고서 쌍을 포함하는 두 개의 벤치마크 데이터셋에서 전통적인 Transformer 기반 모델보다 높은 보고서 품질과 계산 효율성을 보였습니다. 이는 최신 기술(SOTA)들과 비교했을 때 리소스 효율적인 솔루션을 제공함을 나타냅니다.



### $M^3EL$: A Multi-task Multi-topic Dataset for Multi-modal Entity Linking (https://arxiv.org/abs/2410.18096)
- **What's New**: 본 논문에서는 Multi-modal Entity Linking (MEL) 를 위한 대규모 데이터셋 M^3EL을 제안한다. 이 데이터셋은 79,625개의 인스턴스를 포함하고 있으며, 5개의 다양한 주제와 9가지 다중 모달(Multi-modal) 작업을 포괄하고 있다. 기존 MEL 데이터셋의 한계를 극복하기 위해 데이터셋 구축 파이프라인을 수립하였다.

- **Technical Details**: M^3EL 데이터셋은 318,500개의 이미지를 포함하고 있으며, Text-Text, Text-Image, Image-Text 등 다양한 모달 작업을 지원한다. CLIP 모델을 활용하여 MODE(모달-증강) 훈련 전략을 제안하며, M^3EL 데이터셋을 기반으로 CLIP_{ND} 모델을 미세 조정(fine-tune)하였다.

- **Performance Highlights**: 실험 결과 기존 모델의 정확도가 49.4%에서 75.8%까지 다양하게 나타난 반면, M^3EL 데이터셋으로 학습한 CLIP_{ND} 모델은 다양한 작업에서 평균 9.3%에서 25%까지 성능이 향상되었다. 이 데이터셋은 MEL 알고리즘의 일반화 성능을 효과적으로 향상시킬 수 있는 우수한 품질의 사전 훈련 데이터셋으로 자리잡는다.



### BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping (https://arxiv.org/abs/2410.15430)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 기존의 training-required 및 training-free 방법들 사이의 연결고리를 찾고, 두 방법의 장점을 결합한 새로운 적응 전략인 BoostAdapter를 제안합니다. BoostAdapter는 테스트 데이터에서 유용한 특징을 추출하고, 효율적인 메모리 뱅크를 통해 테스트 샘플을 적응적으로 수정하는 방법을 탐구합니다.

- **Technical Details**: BoostAdapter는 instance-agnostic historical samples와 instance-aware boosting samples를 저장하는 경량 key-value 메모리를 유지합니다. 이 메모리는 테스트 데이터 스트림에서 필터링된 히스토리 샘플과 지역 부스트래핑에서 생성된 샘플로 구성되어 있습니다. 이 방법을 통해 두 가지 접근법의 장점을 활용할 수 있습니다.

- **Performance Highlights**: 실험 결과, BoostAdapter는 cross-domain 데이터셋과 out-of-distribution 데이터셋 모두에서 탁월한 성능을 보여주었으며, 기존 방법들과 비교하여 더 높은 일반화 능력을 입증했습니다.



New uploads on arXiv(cs.AI)

### ConceptDrift: Uncovering Biases through the Lens of Foundational Models (https://arxiv.org/abs/2410.18970)
- **What's New**: 본 연구에서는 기존의 데이터 분석에만 국한된 Bias 분석 방법에서 벗어나, Fine-tuned foundational models에서 나타나는 Bias를 식별하기 위한 새로운 접근법인 ConceptDrift를 제안합니다.

- **Technical Details**: ConceptDrift는 텍스트 표현의 임베딩으로부터 시작하여, 모델의 결정 프로세스에서 스피큘러스 피쳐를 검출하는 데 중점을 둡니다. 이는 모델의 클래스 치우침을 분석하여 결정-making에서 중요하게 작용하는 개념을 파악할 수 있도록 돕습니다. 이 방법은 이미지와 텍스트 데이터셋 모두에서 테스트되었습니다.

- **Performance Highlights**: 우리의 방법은 Waterbirds, CelebA, Nico++, CivilComments의 네 가지 데이터셋에서 이전의 Bias 식별 방법들과 비교하여 제로샷 성능을 유의미하게 향상시켰습니다.



### OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning (https://arxiv.org/abs/2410.18963)
Comments:
          Work in progress

- **What's New**: OSCAR는 다양한 데스크탑 및 모바일 애플리케이션을 통해 자율적으로 탐색하고 상호 작용할 수 있도록 설계된 범용 에이전트입니다. 이 에이전트는 마우스와 키보드 입력을 표준화하여 다양한 작업을 수행할 수 있으며, 화면 이미지를 처리하여 사용자 명령을 실행합니다.

- **Technical Details**: OSCAR는 상태 기계(state machine)로 운영되며, 오류 처리 메커니즘과 동적 작업 재계획(dynamic task re-planning) 기능을 갖추고 있어 실시간 피드백과 예외에 효율적으로 대응할 수 있습니다. OSCAR는 사용자 명령을 실행 가능한 Python 코드로 번역하여 GUI(Graphical User Interface)에 대한 정밀한 제어를 가능하게 합니다.

- **Performance Highlights**: OSCAR는 GAIA 벤치마크에서 평균 28.7%의 성공률을 기록하며, 가장 복잡한 Level 3 작업에서는 13.5%의 성공률을 보여 이전의 최고 성능을 거의 두 배로 증가시켰습니다. OSWorld 및 AndroidWorld 벤치마크에서도 각각 24.5% 및 61.6%의 성공률을 달성하여 실시간 동적 OS 작업에서의 우수한 적응성을 입증했습니다.



### Schema-Guided Culture-Aware Complex Event Simulation with Multi-Agent Role-Play (https://arxiv.org/abs/2410.18935)
Comments:
          Accepted as EMNLP 2024 Demo

- **What's New**: 이 논문은 정부와 사회가 빠르게 대응해야 하는 복잡한 뉴스 사건, 예를 들어 자연 재해와 사회 정치적 갈등에 대한 신속한 대응의 필요성을 강조합니다. 특히, 역사적 사건을 기반으로 미래를 예측하기에는 한계가 있으므로, 이러한 사건을 시뮬레이션할 수 있는 새로운 복합 뉴스 사건 시뮬레이터인 Miriam을 개발했습니다. Miriam은 사건 schema와 사용자 제공 가정을 기반으로 사용자 정의 가능한 복잡한 사건 시뮬레이션을 생성합니다.

- **Technical Details**: Miriam은 시나리오에 대한 도메인 지식을 나타내는 사건 schema와 사건 특정 조건을 나타내는 사용자 제공 가정을 결합하여, 사회 및 문화적 맥락에 따라 사건의 역학을 시뮬레이션합니다. 에이전트 기반 시뮬레이션을 통해 시뮬레이션 과정에서 개인 캐릭터의 상태, 계획 및 행동을 시뮬레이션하고, 이는 현실적이며 일관된 사건 생성을 가능하게 합니다. 시뮬레이터는 전체 사건 로그와 개요 문서 형태로 결과를 제공합니다.

- **Performance Highlights**: Miriam은 인도적 지원 조직의 참가자들에게 매우 긍정적인 반응을 얻었으며, 문화적 규범을 반영한 더 적합하고 일관된 시뮬레이션을 통해 미래 재해의 예방 및 관리에 기여할 수 있는 잠재력을 보여주고 있습니다.



### Improving Small-Scale Large Language Models Function Calling for Reasoning Tasks (https://arxiv.org/abs/2410.18890)
- **What's New**: 최근 발전된 대규모 언어 모델(LLMs)은 자연어 이해 및 생성에서 뛰어난 능력을 보여주고 있습니다. 하지만 수학적 문제 해결 및 논리적 추론에서 여전히 도전 과제를 안고 있으며, 이러한 한계를 극복하기 위해 함수 호출 기능을 탐색하고 있습니다.

- **Technical Details**: 본 연구에서는 LLM의 함수 호출 기능을 활용하여 작은 언어 모델을 훈련하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 문제와 사용 가능한 함수 집합을 제공받아, 단계별 추론 체인에서 LLM에게 추가적 예시와 설명을 주며 쿼리하는 에이전트를 사용합니다. 생성한 데이터셋은 인간 피드백 강화 학습(RLHF) 방법성을 통해 훈련됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 작은 모델에 대한 함수 호출 능력을 개선하며, 모델 크기와 성능 간의 균형을 조정하는 데 성공했다는 점을 보여줍니다.



### Guiding Empowerment Model: Liberating Neurodiversity in Online Higher Education (https://arxiv.org/abs/2410.18876)
Comments:
          9 pages, 1 Figure, 1 Table, Accepted in FIE 2024

- **What's New**: 이 논문은 신경 다양성(neurodiversity)과 상황적 제한이 있는 학습자를 위한 교육의 공평성 격차를 해결하기 위해 동적 요인을 식별하고, 학습과 기능에 영향을 미치는 여러 요인들을 종합적으로 분석합니다. 이를 통해 'Guiding Empowerment Model'을 소개하며, 이는 학습자의 능력에 영향을 미치는 주요 인지 및 상황적 요인을 contextualize합니다.

- **Technical Details**: 논문에서는 약 100개의 자료를 바탕으로 신경 다양성이 고려된 관점을 통합하고, UDL(Universal Design for Learning)과 같은 기존의 프레임워크와 함께 사용자 문제를 조명하는 세 가지 샘플 학생 프로필을 제공합니다. 또한, AI 기반의 기술 지원을 통해 학습자의 개인적 필요를 확인하고 효과적인 지원 시스템을 구현하는 과정을 설명합니다.

- **Performance Highlights**: 이 연구를 통해 제안된 접근 방식은 신경 다양성과 상황적 제한이 있는 학습자의 주요 학습 장벽을 해소하는 방식으로, 사용자 맞춤형 작업 관리(customizable task management), 다양한 콘텐츠 접근(increased varied content access), 다중 모달 협업(multi-modal collaboration) 등을 통해 향상된 학업 성취를 달성할 수 있음을 보여줍니다.



### The Cat and Mouse Game: The Ongoing Arms Race Between Diffusion Models and Detection Methods (https://arxiv.org/abs/2410.18866)
Comments:
          10 pages, 1 figure

- **What's New**: 확산 모델(Diffusion Models)의 발전이 합성 미디어 생성에서 혁신을 가져오고 있습니다. 이 기술은 예술, 디자인, 과학적 시각화 분야에서 새로운 가능성을 탐구하는 데 사용되고 있으며, 그 결과로 이미지 생성의 자유도와 사실성이 크게 증대되었습니다. 하지만 이러한 기술의 발전에는 심각한 윤리적, 사회적 문제도 수반됩니다.

- **Technical Details**: 확산 모델은 가우시안 노이즈를 반복적으로 추가하여 실제 이미지를 단계적으로 손상시킨 후, 이를 역전시켜 고화질의 합성 이미지를 생성합니다. 이러한 과정에서 생성된 이미지에는 고유한 특징들이 있으며, 특히 주파수(domain)와 공간(domain) 분석 기법을 통해 탐지가 가능합니다. 주요 탐지 기법으로는 CNN(CNN-based methods), 비전 트랜스포머(Vision Transformers) 기반 방법과 혼합 접근법(Hybrid Approaches)이 있으며, 이는 다양한 데이터셋과 평가 지표에 기초하여 탐지 정확도를 향상시키고 있습니다.

- **Performance Highlights**: 각 탐지 방법들은 저작권 보호(Copyright Protection), 허위 정보 방지(Combating Misinformation), 포렌식 분석(Forensic Analysis)에서 실질적인 응용 가능성을 보여주고 있습니다. 그러나 탐지의 정확성을 높이기 위해서는 더욱 다양한 데이터셋이 필요하며, 특히 고유한 모델 지문(Model Fingerprints)이나 독특한 저주파 및 고주파 특성(High-Frequency Features)을 활용한 접근이 요구됩니다.



### Demystifying Large Language Models for Medicine: A Primer (https://arxiv.org/abs/2410.18856)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 의료 분야에서 효과적으로 활용할 수 있는 단계별 가이드라인과 최고의 사례를 제공합니다.

- **Technical Details**: 이 연구는 LLM을 활용하는데 필요한 주요 단계인 작업 설정, LLM 선택, prompt engineering, fine-tuning, 및 배포를 포함하는 방법론을 설명합니다. 핵심 고려사항으로는 의료 작업과 LLM의 핵심 기능의 일치를 찾고, 선택된 작업 및 데이터, 성과 요구 사항, 모델 인터페이스에 따라 모델을 선택하는 것이 포함됩니다.

- **Performance Highlights**: 의료 전문가들이 LLM을 임상 실무에 안전하고 신뢰성 있게 통합할 수 있도록 돕기 위해 전략적인 접근 방식을 제시하며, 규제 준수, 윤리적 가이드라인, 공정성과 편향성에 대한 지속적인 모니터링을 강조합니다.



### Applying Neural Monte Carlo Tree Search to Unsignalized Multi-intersection Scheduling for Autonomous Vehicles (https://arxiv.org/abs/2410.18786)
- **What's New**: 본 논문에서는 자율 시스템의 공유 자원 접근 동적 스케줄링 문제를 다루고 있으며, Neural Monte Carlo Tree Search (NMCTS)를 활용하여 신호 없는 교차로를 통과하는 차량 무리(플래툰) 스케줄링을 최적화하는 방법을 제시합니다. 이 과정에서 교차로의 동적 복잡성을 보드 게임 형태로 변환하여 문제를 단순화하고, PNMCTS를 통해 해결합니다.

- **Technical Details**: 이 방법은 NMCTS를 사용하여 차량의 로드 스페이스 예약 요청을 보드 게임식 문제로 변환하고, 적절한 스케줄을 찾기 위해 왕복 교차로 시뮬레이션에서 PNMCTS를 적용합니다. 또한, 훈련 데이터의 품질 향상을 위해 우선 재샘플링 방법과 병렬 NMCTS(Parallel NMCTS, PNMCTS)를 통합하였습니다. 그리고 정책 학습을 위한 커리큘럼 학습 전략을 사용하여 훈련합니다.

- **Performance Highlights**: 시뮬레이션 결과, PNMCTS는 한 교차로의 복잡한 교통 상황에서 95%의 문제를 해결하였으며, 경량 교통에서는 교차 시간을 43%, 중량 교통에서는 52% 감소시켰습니다. 3x3 다중 교차로 네트워크에서도 평균 이동 시간을 74.5%, 총 처리량을 16% 향상시키는 성과를 보였습니다.



### Should We Really Edit Language Models? On the Evaluation of Edited Language Models (https://arxiv.org/abs/2410.18785)
Comments:
          NeurIPS 2024 this https URL

- **What's New**: 본 논문은 모델 편집 기술이 언어 모델의 일반적인 능력에 미치는 영향을 심층적으로 평가하고 분석합니다. 기존의 편집 방법은 제한된 범위의 지식 업데이트에 적합하며, 보다 정교하고 신뢰할 수 있는 편집 방법에 대한 필요성을 제기합니다.

- **Technical Details**: 모델 편집은 특정 사실에 대한 언어 모델의 행동을 정확하게 조정하는 것을 목표로 합니다. 이 과정에서 입력된 사실은 (s,r,o) 형태의 튜플로 이루어져 있으며, 해당 튜플을 새롭게 수정된 튜플 (s,r,o*)으로 교체하는 방식을 사용합니다. 편집 방법을 평가하는 주요 기준은 신뢰성(reliability), 일반화(generalization), 국소성(locality)입니다.

- **Performance Highlights**: 여러 연구 결과에 따르면, 현재의 모델 편집 방법들은 수십 개의 편집에 대해서만 모델의 일반 능력을 유지할 수 있으며, 수백 개 이상의 편집이 이루어질 경우 성능이 급격히 저하됩니다. 특히, 수천 개의 편집을 진행했을 때 모델의 지식 구조가 파괴되어 출력이 비어있는 문자열로 나타나는 'muting effect' 현상이 관찰되었습니다.



### AI Readiness in Healthcare through Storytelling XAI (https://arxiv.org/abs/2410.18725)
Comments:
          Pre-print of the accepted manuscript in EXPLIMED - First Workshop on Explainable Artificial Intelligence for the Medical Domain, European Conference on Artificial Intelligence (ECAI) - 2024, Santiago de Compostela, Spain

- **What's New**: 이번 연구는 이야기 중심의 설명 가능한 인공지능(Storytelling XAI) 프레임워크를 개발하여, 다양한 도메인 전문가를 위한 맞춤형 설명을 제공하고자 합니다. 특히, 의료 분야에서 AI 신뢰성을 높이기 위한 새로운 접근 방식을 소개합니다.

- **Technical Details**: 우리의 접근법은 다중 작업 지식 증류(multi-task distillation)와 해석 가능성 기법을 결합하여 대중의 요구에 맞는 AI 설명 가능성을 향상시킵니다. 다중 작업 지식 증류는 모델이 작업 간의 관계를 활용할 수 있도록 하여, 각 작업이 상호 지원하여 해석 가능성을 높입니다. 또한, 이 방법은 복잡한 대형 딥러닝 모델에 접근할 수 있게 합니다.

- **Performance Highlights**: 연구에서는 흉부 엑스레이(chest X-ray) 분석을 사용 사례로 하여 우리의 프레임워크가 의료 전문가들에게 유용한 XAI를 제공하는 효과를 증명합니다. 우리의 방법은 의료 분야에서의 책임감 있는 AI 구현을 목표로 하며, 도메인 전문가와 머신러닝 전문가 모두에게 신뢰를 구축하는 데 기여합니다.



### LLM-based Online Prediction of Time-varying Graph Signals (https://arxiv.org/abs/2410.18718)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)을 활용하여 시간에 따라 변하는 그래프 신호에서 결측치를 예측하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 공간적 및 시간적 매끄러움을 활용하여 LLM의 메시지 전달 체계를 통해 결측 노드를 예측합니다.

- **Technical Details**: 제안된 방법론은 시공간 차원에서의 매끄러움 가정을 활용하여 그래프 노드의 시간 변화 회귀 작업을 수행합니다. 각 노드에 대해 상대방의 신호와 이전 추정을 LLM에 입력하여 결측 관측치를 추론합니다. LLM은 각 노드의 이웃 신호와 시간에 따른 추세를 분석하여 결측 신호를 예측합니다.

- **Performance Highlights**: 실험 결과, GPT-3.5-turbo 모델을 사용하여 시간에 따라 변화하는 바람 속도 그래프 신호의 결측값을 예측했으며, 정확도 면에서 기존 온라인 그래프 필터링 알고리즘을 능가하는 성능을 보여주었습니다.



### GADT: Enhancing Transferable Adversarial Attacks through Gradient-guided Adversarial Data Transformation (https://arxiv.org/abs/2410.18648)
- **What's New**: 이번 논문은 데이터 증강(Data Augmentation) 파라미터와 적대적 노이즈(Adversarial Noise)를 최적화하여 공격 효율성을 높이는 새로운 공격 알고리즘인 GADT를 제안합니다. GADT는 반복적 적대성(iterative antagonism)을 통해 적절한 DA 파라미터를 식별하고, 그런 다음 이러한 파라미터에 따라 AN을 업데이트합니다.

- **Technical Details**: GADT는 데이터 증강 작업을 위한 미분 가능(differentiable) 라이브러리를 사용하여 적대적 DA 파라미터를 식별하며, DA 최적화 과정에서 새로운 손실 함수(loss function)를 도입합니다. 이 손실 항은 원본 이미지 내용을 유지하면서 적대적 효과를 향상시킵니다. Kornia 라이브러리를 활용하여 DA 파라미터에 대한 공격 메트릭과의 원시 기울기(raw gradient)를 계산합니다.

- **Performance Highlights**: 공공 데이터셋에 대한 광범위한 실험을 통해 GADT가 기존 전이 공격(Transferable Attacks) 방법들과 효과적으로 통합되며, 다양한 네트워크에서 DA 파라미터를 효과적으로 업데이트하고 AN 수식 전략을 유지하면서 성능을 향상시키는 것을 입증했습니다. 또한, GADT는 쿼리 기반 공격(query-based attacks)과 같은 다른 블랙박스 공격 시나리오에도 활용될 수 있습니다.



### Multi-agent cooperation through learning-aware policy gradients (https://arxiv.org/abs/2410.18636)
- **What's New**: 이 논문은 자율적으로 학습하는 에이전트 간의 협력을 촉진하는 새로운 방법을 제시합니다. 특히, 학습-aware 에이전트 간의 상호작용을 위한 첫 번째 비편향적이며 높은 차수 미분을 요구하지 않는 policy gradient 알고리즘을 소개합니다.

- **Technical Details**: 본 연구에서는 self-interested 에이전트를 대상으로 한 multi-agent reinforcement learning의 복잡성을 다루며, 협력을 이루기 위한 새로운 학습-aware reinforcement learning 규칙을 도입합니다. 이 알고리즘은 상대 에이전트의 학습 다이내믹스를 모델링하여 행동을 조정하며, 학습 알고리즘의 미니배치 처리와 같은 다양한 기술을 적용할 수 있습니다.

- **Performance Highlights**: 알고리즘의 성능은 전통적인 사회적 딜레마 환경에서 협력적인 행동과 높은 수익을 통한 성공적인 결과로 이어졌습니다. 특히, 장기적인 행동 조정이 요구되는 복잡한 사회적 딜레마 문제에서 첫 번째로 협력을 이끌어냈으며, 기존 방법보다 오랜 맥락 정책 훈련에서 큰 성과를 기록했습니다.



### AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistan (https://arxiv.org/abs/2410.18603)
- **What's New**: 본 논문에서는 다양한 이종 에이전트를 동적으로 통합하여 컴퓨터 작업을 자동화할 수 있는 확장 가능한 플랫폼인 AgentStore를 제안합니다. 이 플랫폼은 특히 개방형 컴퓨터 작업을 처리하는 데 있어 기존 시스템의 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: AgentStore는 세 가지 주요 구성 요소로 구성됩니다: AgentPool(에이전트 풀), AgentEnroll(에이전트 등록), MetaAgent(메타 에이전트). MetaAgent는 AgentToken 전략을 사용하여 적합한 에이전트를 선택하고, 다양한 에이전트의 전문성과 일반성을 활용하여 도메인 특정 작업 및 시스템 전반의 작업을 수행하도록 설계되었습니다. 또한, AgentToken을 이용한 자동화된 조정 과정이 포함되어 있어 실용성을 높입니다.

- **Performance Highlights**: AgentStore는 OSWorld 벤치마크에서 11.21%에서 23.85%로 성과를 두 배 이상 향상시킴으로써 기존 시스템의 성능 한계를 초월했습니다. 이는 일반화와 전문성 모두에서 에이전트 시스템의 능력을 향상시키는 데 기여함을 보여줍니다.



### Aligning CodeLLMs with Direct Preference Optimization (https://arxiv.org/abs/2410.18585)
- **What's New**: 본 연구는 CodeLLM의 정렬 (alignment) 단계에 대한 탐구 부족을 짚고 넘어가며, 기존의 PPO (Proximal Policy Optimization) 알고리즘의 한계를 지적하고 DPO (Direct Preference Optimization) 알고리즘을 이용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: DPO는 사용자로부터 수집된 선호 데이터 쌍 (preference data pairs)을 기반으로 작동하며, 각 응답의 품질에 따라 차별화된 보상 점수를 자동으로 부여합니다. DPO는 코딩 과제의 결과를 정량적으로 평가하는 외부 코드 실행기를 도입하여, 높은 유연성과 오류 감지에서의 장점을 제공합니다.

- **Performance Highlights**: DPO 방법론을 사용한 결과, CodeQwen 1.5 7B 모델의 MBPP와 HumanEval 기준에서 성능이 각각 0.783에서 0.804, 0.829에서 0.878로 향상되었습니다.



### SIKeD: Self-guided Iterative Knowledge Distillation for mathematical reasoning (https://arxiv.org/abs/2410.18574)
- **What's New**: 이번에 발표된 연구에서는 SIKeD(Self-guided Iterative Knowledge Distillation)라는 새로운 지식 증류 방법을 제안하여 대형 언어 모델(LLM)로부터 소형 모델이 다양한 추론 전략을 학습하도록 돕는다. 이 방법은 소형 모델이 스스로 생성한 출력을 기반으로 최적의 전략을 선택하는 자율적인 훈련 방식을 채택한다.

- **Technical Details**: SIKeD 방법론은 대형 모델이 다양한 추론 전략(예: Chain of Thought, Subquestion Decomposition 등)을 사용하여 소형 모델에게 문제를 해결하는 방법을 지도하는 방식으로 진행된다. 각 훈련 반복(iteration)마다 LLM 생성 데이터와 소형 모델의 자가 생성 출력을 결합하여 학습을 진행하며, 결과적으로 소형 모델이 다양한 전략을 적용할 수 있도록 해준다.

- **Performance Highlights**: SIKeD는 다양한 수학적 추론 데이터셋(GSM8K, SVAMP 등)에서 실험을 통해 기존의 증류 방법보다 최대 5점 이상의 향상된 성능을 보였다. 또한, 여러 번의 SIKeD 반복 훈련을 통해 소형 모델은 특정 문제에 대해 적절한 전략을 선택할 수 있는 능력을 가지게 된다.



### Explainable News Summarization -- Analysis and mitigation of Disagreement Problem (https://arxiv.org/abs/2410.18560)
- **What's New**: 이번 연구에서는 텍스트 요약의 Explainable AI (XAI) 기술에 있어서 발생하는 불일치 문제(disagreement problem)를 해결하기 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 방법은 sentence transformers와 k-means clustering 알고리즘을 활용하여 입력 기사를 세그먼트로 분할하고, 각 세그먼트에 대해 생성된 요약의 설명을 제공합니다. 이를 통해 종합적인 설명 대신 지역적(segmented) 설명을 제공함으로써 XAI 방법들 간의 불일치를 줄이는 것을 목표로 합니다.

- **Performance Highlights**: XSum 및 CNN-DailyMail이라는 두 개의 뉴스 요약 데이터셋에서 실험한 결과, XAI 방법들 간의 불일치가 크게 줄어드는 것을 확인했습니다. 또한, 사용자 친화적인 JavaScript 시각화 도구가 개발되어, 사용자가 입력 기사와 기계 생성 요약의 색상 코드화된 시각화를 인터랙티브하게 탐색할 수 있도록 하였습니다.



### PRACT: Optimizing Principled Reasoning and Acting of LLM Agen (https://arxiv.org/abs/2410.18528)
Comments:
          Accepted to SIG CoNLL 2024

- **What's New**: 본 논문에서는 Trajectory 데이터에서 행동 원칙(action principles)을 학습하고 강화하기 위한 새로운 방법인 PRAct(Principled Reasoning and Acting) 프레임워크를 소개합니다. 이 접근법의 중심에는 텍스트 그래디언트(text gradients)가 있으며, 이를 통해 행동 원칙을 도출합니다.

- **Technical Details**: PRAct 프레임워크는 환경 보상(rewards)을 활용하는 Reward-RPO와 외부 보상 없이 자가 반성을 수행하는 Self-RPO의 두 시나리오에서 개발됩니다. RPO의 두 가지 방법인 RPO-Traj와 RPO-Batch는 각각 다양한 설정에 적합하도록 도입되었습니다. 상태 그래프(state graph)를 통해 에이전트의 행동 원칙을 포뮬레이션하고, 전이를 만드는 방식으로 휴리스틱, 머신러닝(Machine Learning) 모델, LLM(대규모 언어 모델) 추론을 이용합니다.

- **Performance Highlights**: 네 가지 환경에서의 실험 결과, PRAct 에이전트는 RPO 프레임워크를 활용하여 행동 원칙을 효과적으로 학습하고 적용함으로써 성능을 향상시킵니다.



### Scaling up Masked Diffusion Models on Tex (https://arxiv.org/abs/2410.18514)
- **What's New**: 이 논문은 Masked Diffusion Models (MDMs)의 최초의 스케일링 법칙을 설정하고, 언어 모델의 필수 과제에서의 성능을 평가했습니다.

- **Technical Details**: MDMs는 1.1억(1.1B)개의 파라미터로 훈련되어 ARMs (Autoregressive Models)와 비교했을 때 스케일링 속도가 비슷하며, 상대적으로 작은 계산 격차를 보여줍니다. 논문에서는 비지도 분류기 없는 가이던스(unsupervised classifier-free guidance, CFG) 메커니즘을 제안하여, 편향되지 않은 대규모 데이터로부터 성능을 향상시킵니다.

- **Performance Highlights**: 1.1B MDM은 언어 이해에서 경쟁력 있는 성과를 보여주며, 여러 제로샷 벤치마크에서 더 큰 GPT-2 모델을 초과했습니다. MDM은 텍스트 생성에서 ARMs에 비해 1.4배 더 빠르며, 생성 품질이 높은 성과를 내는 가변적인 트레이드오프를 제공합니다. 이 모델은 ARMs의 고유한 한계를 효과적으로 해결합니다.



### A framework for GNSS-based solutions performance analysis in an ERTMS contex (https://arxiv.org/abs/2410.18510)
- **What's New**: 본 논문은 GNSS (Global Navigation Satellite System)를 철도 응용 프로그램에 도입하기 위한 진전을 다루고 있습니다. 유럽은 GNSS를 더 많은 응용 분야에 통합하여 철도의 탄소 발자국을 줄이는 도구로 사용하려고 합니다.

- **Technical Details**: GNSS는 철도에서의 위치 결정 기능을 제공하고, 전통적인 오도메트리(Odometry) 및 발리즈(Balises)를 보완하여 정확한 기차 위치 정보를 제공합니다. GNSS 수신기를 탑재한 온보드(localisation) 솔루션이 미래 시스템의 핵심이 될 것이며, 새로운 개념인 Moving Blocks, Virtual Coupling 및 자동화 개발을 지원할 것입니다.

- **Performance Highlights**: GNSS의 성능은 환경 조건, 신호 수신 방해 등으로 인해 영향을 받을 수 있습니다. 특히, 철도는 다양한 장애물이 존재하는 지역에서 운영되므로 성능 평가에 대한 새로운 질문들이 대두되고 있습니다. 성능을 평가하는 데 있어 동적 환경에서의 검증 방법과 실패의 영향 평가가 주요 이슈로 남아 있습니다.



### SFB-net for cardiac segmentation: Bridging the semantic gap with attention (https://arxiv.org/abs/2410.18503)
- **What's New**: 최근 몇 년 동안 심장 이미지 분할을 위한 딥러닝 알고리즘이 널리 사용되고 있습니다. 본 논문에서는 Swin Filtering Block 네트워크(SFB-net)를 소개하며, 이는 전통적인 합성곱(convolution)과 Swin Transformer 레이어를 결합하여 장기 의존성(long-range dependencies) 문제를 해결합니다.

- **Technical Details**: SFB-net은 네트워크의 하단에서 공간적 주의(spatial attention)를 도입하기 위해 전통적인 합성곱 층을 사용하고, 인코더(encoders)와 디코더(decoders) 사이에서 의미적으로 풍부한 특징(high level semantically rich features)에 집중하기 위해 Swin Transformer를 활용합니다.

- **Performance Highlights**: ACDC 데이터셋에서 평균 Dice 점수 92.4를 달성하였으며, 이는 해당 데이터셋에서의 다른 연구 결과를 초월하는 성과입니다. M&M 데이터셋에서는 평균 Dice 점수 87.99를 기록하여, 제안된 방법이 다양한 벤더(vendor)와 센터(center)의 데이터에 잘 일반화된다는 것을 보여줍니다.



### LLM as a code generator in Agile Model Driven Developmen (https://arxiv.org/abs/2410.18489)
- **What's New**: 본 연구는 큰 언어 모델(GPT-4)을 활용한 코드 자동 생성의 접근 방식을 진전시켰습니다. 특히, 모빌리티와 확장성을 강화하였으며, 소프트웨어 소스 모델을 근거로 코드 생성을 최적화하는 Agile Model Driven Development(AMDD) 접근법을 제안합니다.

- **Technical Details**: AMDD 접근법은 Unified Modeling Language(UML)를 사용하여 다중 에이전트 무인 자동차 군(UVF) 시스템을 모델링합니다. Object Constraint Language(OCL)와 FIPA 온톨로지 언어를 통합하여 코드 구조 메타 모델링을 수행하며, 다음으로 Java 및 Python 코드가 JADE 및 PADE 프레임워크와 호환되는 결과를 생성하도록 GPT-4의 자동 생성 기능을 활용합니다.

- **Performance Highlights**: 대셋하고 Cyclomatic Complexity를 활용한 구조적 안전성 평가를 실시하였으며, 다중 언어(Java, Python) 간의 코드 기능 비교를 통해 기대 동작과의 일치를 확인하였습니다. 결과적으로, OCL과 FIPA 온톨로지 메타 모델은 보다 복잡한 코드를 생성하였으나, 품질 유지가 가능함을 보여주었습니다.



### Gene-Metabolite Association Prediction with Interactive Knowledge Transfer Enhanced Graph for Metabolite Production (https://arxiv.org/abs/2410.18475)
Comments:
          10 PAGES, 4 FIGURES; bibm 2024

- **What's New**: 이 연구는 대사 공학(metabolic engineering) 분야에서 대사 생성물을 위한 효율적인 유전자(target gene) 선정을 자동화하기 위한 새로운 작업인 Gene-Metabolite Association Prediction을 제안합니다. 이 방법은 Saccharomyces cerevisiae(SC)와 Issatchenkia orientalis(IO)라는 두 가지 일반적인 미생물에 대한 2474개의 대사 생성물과 1947개의 유전자를 포함한 첫 번째 벤치마크를 소개합니다.

- **Technical Details**: 제안된 방법은 메타볼리즘 그래프(metabolism graph) 기반에서 대사 생성물과 후보 유전자의 연관 예측을 합니다. 이를 위해 Interactive Knowledge Transfer mechanism(IKT4Meta)을 도입하여 서로 다른 대사 그래프에서의 지식을 통합하여 연관 예측 정확도를 향상시킵니다. 이 방법은 사전학습된 언어 모델(Pretrained Language Models, PLMs)과 구조적 인코더(structural encoders)를 사용하여 대사 그래프 간의 지식 전송을 촉진합니다.

- **Performance Highlights**: 이 방법은 다양한 링크 예측(link prediction) 프레임워크에서 기존 방법들을 최대 12.3%까지 능가하는 실험 결과를 보였습니다. 이는 복잡한 대사 생성물 생산을 위한 목표 유전자의 예측 작업을 단순화하며, PLMs와 구조적 인코더의 결합된 강점을 이용하여 연관 예측의 견고성을 높였습니다.



### Beyond Multiple-Choice Accuracy: Real-World Challenges of Implementing Large Language Models in Healthcar (https://arxiv.org/abs/2410.18460)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 의료 분야에서의 가능성에 대해 다루고 있으며, 실제 적용에서의 여러 가지 도전 과제를 제시합니다.

- **Technical Details**: LLM의 의료 활용에 대한 논의는 운영 취약점, 윤리적 및 사회적 고려 사항, 성능 및 평가의 어려움, 법적 및 규제 준수의 네 가지 측면으로 나뉩니다.

- **Performance Highlights**: LLM의 잠재력을 극대화하고 의료 분야에 책임감 있게 통합하기 위해 이러한 도전 과제를 해결하는 것이 중요합니다.



### Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs (https://arxiv.org/abs/2410.18451)
- **What's New**: 이번 보고서에서는 LLM(대형 언어 모델)을 위한 보상 모델링을 개선하기 위한 데이터 중심의 기술들이 소개되었습니다. 특히, 높은 품질의 오픈 소스 선호 데이터셋을 구축하기 위한 데이터 선택 및 필터링 전략이 제안되어, 총 80K의 선호 쌍으로 이루어진 Skywork-Reward 데이터 컬렉션이 개발되었습니다. 이 데이터를 활용하여 Skywork-Reward 모델 시리즈를 개발하였으며, 현재 RewardBench 리더보드에서 1위 자리를 차지하고 있습니다.

- **Technical Details**: 제안된 기술들은 효과적인 데이터 선택 및 필터링 전략을 포함하며, 가벼우면서도 효과적인 선호 데이터 컬렉션을 제공합니다. 본 연구에서는 다양한 손실 함수에 대한 광범위한 연구를 진행하였고, 기본적인 Bradley-Terry 손실 함수가 보상 모델링 작업에서 일관되게 높은 성능을 보였습니다. Skywork-Reward 모델 시리즈는 이러한 훈련 기술을 집약하여 RewardBench 벤치마크에서의 성능이 크게 향상되었음을 입증하였습니다.

- **Performance Highlights**: Skywork-Reward 모델 시리즈는 RewardBench 리더보드에서 1위와 7위를 기록하고 있으며, 제안한 데이터 셋 및 기술이 많은 상위 모델들의 성능 향상에 기여하고 있습니다. 이러한 기여는 LLM의 사용자 선호 학습 응용에 있어 실질적인 효과를 강조합니다.



### Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieva (https://arxiv.org/abs/2410.18385)
Comments:
          Accepted for publication at EMNLP 2024 Main Conference

- **What's New**: 이번 연구에서는 새로운 알고리즘인 Universal Document Linking (UDL)을 제안하여 제로샷(Zero-shot) 정보 검색(Information Retrieval, IR)의 도전 과제를 극복하고자 하였습니다. UDL은 유사한 문서를 연결하여 다양한 특성을 가진 여러 데이터셋에서 합성 쿼리 생성을 향상시키는 데 초점을 맞춥니다.

- **Technical Details**: UDL은 엔트로피(Entropy)를 활용해 유사성 모델을 선택하고 명명实体 인식(Named Entity Recognition, NER)을 통해 문서의 링크 결정을 내립니다. 알고리즘은 TF-IDF 및 사전 훈련된 언어 모델(Pre-trained Language Models)을 사용하여 문서 임베딩(Document Embeddings)을 생성하며, 유사성이 높은 문서들 간의 링크를 수립하기 위해 코사인 유사성(Cosine Similarity)을 계산합니다.

- **Performance Highlights**: UDL은 다양한 데이터셋과 IR 모델을 통해 검증되었으며, 제로샷 경우에서 기존의 최첨단 방법들을 초월하는 효과를 입증하였습니다. 이로 인해 UDL의 범용성과 유연성이 강화되었습니다.



### Integrating Canonical Neural Units and Multi-Scale Training for Handwritten Text Recognition (https://arxiv.org/abs/2410.18374)
- **What's New**: 본 논문에서는 새로운 3차원(3D) attention 모듈과 글로벌-로컬(context information) 정보를 활용하여 필기체 인식 네트워크를 제안합니다. 기존의 segmentation-free 접근법인 HMM, CTC, ED에서 영감을 받아, 정보를 더욱 효율적으로 처리할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 네트워크는 세 가지 주요 구성 요소를 포함합니다: 3D attention 모듈, 글로벌-로컬 맥락 정보, 그리고 CTC와 크로스 엔트로피 손실을 동시 사용한 다중 스케일 학습 접근법입니다. 3D attention 모듈은 다양한 해상도를 갖는 차단된 feature maps의 2D 정보를 명확히 추출합니다.

- **Performance Highlights**: 중국어 필기체 데이터셋(SCUT-HCCDoc, SCUT-EPT)과 영어 필기체 데이터셋(IAM)에서 실험한 결과, 제안된 방법이 기존 최첨단 기술들에 비해 유사한 성능을 달성함을 보여주었습니다.



### Contextual Biasing to Improve Domain-specific Custom Vocabulary Audio Transcription without Explicit Fine-Tuning of Whisper Mod (https://arxiv.org/abs/2410.18363)
- **What's New**: 본 연구에서는 OpenAI의 Whisper 모델의 전사 정확성을 향상시키기 위한 새로운 방법론을 제시합니다. 기존의 fine-tuning(미세 조정) 없이도 상대적으로 적은 훈련 데이터셋으로 특정 어휘에 대한 모델 출력을 유도하는 contextual biasing(맥락 편향)을 활용합니다.

- **Technical Details**: 제안된 방법은 TCPGen(tree-constrained pointer generator)라는 구성 요소를 사용하여, 불린 리스트에서 도출된 어휘를 기반으로 Whisper 모델의 디코더 출력을 개선합니다. TCPGen은 신경망 구조와 형태적 prefix tree(접두어 트리)를 결합하여, 각 추론 단계에서 관련 서브트리에 집중하여 확률 분포를 계산합니다.

- **Performance Highlights**: 비교 실험 결과, 편향 모델은 원본 Whisper 모델들보다 전사 단어 오류율(Word Error Rate, WER)을 눈에 띄게 감소시켰으며, 다운스트림 애플리케이션의 성능도 향상되었습니다. 이는 제한된 어휘를 가진 도메인에서 스피치-투-텍스트(speech-to-text) 번역 성능 개선의 가능성을 시사합니다.



### Geometric Feature Enhanced Knowledge Graph Embedding and Spatial Reasoning (https://arxiv.org/abs/2410.18345)
Comments:
          4 pages, 1 figure, Accepted for the 7th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery

- **What's New**: 본 연구는 Geospatial Knowledge Graphs (GeoKGs)의 지리적 특성을 반영하여 지리적 직관을 내포한 새로운 KGE(Knowledge Graph Embedding) 기법을 개발했습니다. 이 기술은 지리적 개체와 공간 관계를 포함하는 새로운 전략을 통합하여 링크 예측 작업에서의 정확성을 향상시킵니다.

- **Technical Details**: GeoKG는 위키피디아에서 수집된 데이터로 구성되었으며, 공간 관계의 기하학적 특징(geometry features)을 통합하는 새로운 KGE 모델을 제안합니다. 기존의 KGE 기법들은 triplet의 의미 구조에만 의존하여 지리적 기하학적 특성을 인식하지 못하는 문제를 해결하기 위해, 기하학적 특성과 공간 관계 용어 간의 거리를 좁히는 방법을 사용합니다.

- **Performance Highlights**: 새롭게 제안된 KGE 모델은 링크 예측 작업에서 정점과 공간 관계 모두에 대해 예측 정확성을显著提高했습니다. 특히, topology와 direction을 포함한 기하학적 특성의 통합이 결과에 긍정적인 영향을 미쳤음을 보여주었습니다.



### Kenyan Sign Language (KSL) Dataset: Using Artificial Intelligence (AI) in Bridging Communication Barrier among the Deaf Learners (https://arxiv.org/abs/2410.18295)
Comments:
          14 pages, to be published in 3rd International Conference on Artificial Intelligence and Robotics (MIRG-ICAIR 2023)

- **What's New**: 케냐 수어(Kenyan Sign Language, KSL)는 케냐의 청각 장애인 커뮤니티가 사용하는 주요 언어입니다. 이 프로젝트인 AI4KSL은 청각 장애인과 비장애인 간의 의사소통 장벽을 해소하기 위한 기술 혁신입니다.

- **Technical Details**: 본 연구는 2023년부터 2024년까지 진행되는 2년간의 연구 프로젝트로, 케냐 청각 장애인의 대표 샘플을 바탕으로 한 spontaneuous 및 elicited 데이터의 디지털 오픈 액세스 AI를 개발하는 것을 목표로 합니다. KSL 데이터셋은 영어 문장 14,000개와 연관된 KSL Gloss, 20,000개 이상의 서명된 KSL 비디오, 10,000개의 분할 및 세분화된 KSL 비디오를 포함하여 HamNoSys 시스템에 따라 4,000개의 단어가 5개의 발음 매개변수로 전사되었습니다.

- **Performance Highlights**: 이 연구 결과는 케냐의 청각 장애인 학습자를 위한 AI 보조 기술 데이터셋을 개발하는 데 기여하여, 언어 장벽을 해소하고 포용성을 증진시키는 데 큰 역할을 할 것입니다.



### 1-2-3-Go! Policy Synthesis for Parameterized Markov Decision Processes via Decision-Tree Learning and Generalization (https://arxiv.org/abs/2410.18293)
Comments:
          Preprint. Under review

- **What's New**: 본 논문에서는 확률적 모델 검사(Probabilistic Model Checking)의 발전에도 불구하고, 검증 방법론의 확장성(Scalability) 한계를 극복하는 새로운 접근 방식인 학습 기반(policy synthesis using learning) 방법을 제안합니다. 특히 매개변수화된 마르코프 의사결정 프로세스(Parameterized Markov Decision Processes, MDPs)의 경우, 모델의 크기가 엄청나게 커지기 때문에 기존 도구의 범위를 넘어서는 정책을 합성하는 것이 어려웠습니다.

- **Technical Details**: 이 방법은 소규모 인스턴스에서 모델 검사를 통해 얻은 최적 정책을 결정 트리 학습(Decision-Tree Learning)을 사용하여 더 큰 인스턴스에 일반화하는 과정을 포함합니다. 이를 통해 대형 모델의 명시적 상태 공간 탐색(State-Space Exploration)을 회피할 수 있으며, 상태 공간 폭발 문제(State-Space Explosion Problem)에 대한 실질적인 해결책을 제공합니다.

- **Performance Highlights**: 논문에서는 양적 검증 벤치마크 세트(Quantitative Verification Benchmark Set)의 관련 모델에 대해 광범위한 실험을 수행하였으며, 실험 결과 우리의 정책이 최신 분석 도구(State-of-the-Art Analysis Tools)의 범위를 훨씬 초과하는 모델 크기에서도 잘 작동함을 보여줍니다.



### Backdoor in Seconds: Unlocking Vulnerabilities in Large Pre-trained Models via Model Editing (https://arxiv.org/abs/2410.18267)
- **What's New**: 이 논문은 대형 사전 훈련된 모델에 대한 백도어 공격(backdoor attack)의 새로운 접근 방식을 제안합니다. 구체적으로, EDT(Efficient, Data-free, Training-free) 모델을 도입하여 사전 훈련된 모델의 행동을 조작하는 고유한 도전 과제를 해결합니다.

- **Technical Details**: EDT 모델은 모델 편집(model editing) 기술에 영감을 받아 데이터에 의존하지 않고, 훈련 과정이 필요하지 않은 백도어 공격 방법입니다. 이 모델은 경량 코드북(codebook)을 사용하여 독성 이미지의 임베딩(embedding)을 목표 이미지로 대체하며, 훈련 데이터 세트를 오염시키지 않고 피해 모델의 훈련이 필요하지 않습니다.

- **Performance Highlights**: 실험 결과, EDT 모델은 이미지 분류(image classification), 이미지 캡셔닝(image captioning), 이미지 생성(image generation) 등의 다양한 다운스트림 작업에서 100%의 공격 성공률을 달성했으며, 깨끗한 모델의 정확도에 비해 거의 동일한 높은 정확도를 유지함을 보여주었습니다.



### Human-Agent Coordination in Games under Incomplete Information via Multi-Step Inten (https://arxiv.org/abs/2410.18242)
- **What's New**: 이 논문에서는 불완전한 정보 하에서 자율 에이전트와 인간 파트너 간의 전략적 조정 문제를 다룹니다. 특히, 여러 행동을 수행할 수 있는 턴 기반 협력 게임으로의 확장을 소개합니다. 이를 통해 에이전트는 멀티 스텝 의도를 활용하여 장기적인 작업 성능을 개선할 수 있다는 가설을 세웠습니다.

- **Technical Details**: 제안된 접근 방식은 메모리 모듈과 계획 모듈로 구성됩니다. 메모리 모듈은 파트너의 행동 이력을 기반으로 환경 동력학에 대한 확률적 신념을 업데이트하고, 계획 모듈은 IntentMCTS라는 온라인 계획 알고리즘을 통해 다음 행동을 전략적으로 선택합니다. IntentMCTS는 멀티 액션 Monte Carlo tree search (MCTS) 알고리즘을 기반으로 하며, 파트너의 의도를 보상으로 강화하여 검색 트리를 조정합니다.

- **Performance Highlights**: Gnomes at Night 테스트베드에서의 에이전트 간 시뮬레이션 결과에 따르면, IntentMCTS는 다른 MCTS 방법들보다 적은 단계와 제어 전환으로 성공적인 결과를 보여주었습니다. 인간-에이전트 사용자 연구에서는 IntentMCTS 에이전트가 18.52% 더 높은 성공률을 기록하였고, 인지 부하와 좌절감이 낮아졌으며, 사용자의 전반적인 만족도가 높아졌습니다.



### Data Augmentation for Automated Adaptive Rodent Training (https://arxiv.org/abs/2410.18221)
Comments:
          5 pages, 3 figures

- **What's New**: 본 논문에서는 실험실 동물, 특히 설치류에 대한 행동 훈련 프로토콜을 자동화하기 위해 데이터 기반(data-driven) 접근 방식을 이용하여 최적화한 내용을 다루고 있습니다. 자동화된 행동 훈련 시스템을 설계하여 훈련의 효율성과 정확성을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 훈련 프로세스의 개선을 위해 데이터 증강(data augmentation) 기법을 사용하였고, 인공 설치류 모델을 구축하여 실제 설치류의 행동과 유사성을 측정하는 새로운 유사성 메트릭(similarity metric)을 개발했습니다. 이 유사성 메트릭은 행동 확률 분포(action probability distribution) 기반으로 하여 인공 모델의 행동을 생물학적 설치류와 비교합니다.

- **Performance Highlights**: 기존의 훈련 프로토콜을 알고리즘적으로 수정하여 각 설치류의 성과에 따라 학습 속도를 단축시키는 것을 지향합니다. 이 연구는 다양한 인지적 행동을 요구하는 자동 훈련 시스템의 혁신적 발전을 모색하고 있으며, 데이터 제한이 있는 환경에서도 효과적인 훈련 프로토콜을 제시할 수 있는 가능성을 보여줍니다.



### Optimizing the role of human evaluation in LLM-based spoken document summarization systems (https://arxiv.org/abs/2410.18218)
- **What's New**: 본 논문은 Generative AI의 콘텐츠에서 음성 문서 요약을 위한 평가 패러다임을 제안하며, 사회과학의 방법론을 활용하여 인간 평가의 신뢰성을 강화하고자 합니다.

- **Technical Details**: 저자는 네 가지 평가 기준: 양(Quantity), 질(Quality), 관련성(Relevance), 매너(Manner)를 기반으로 한 평가 방법론을 제시하고, 인간 기반 평가 체계 구현에 대한 구체적인 지침을 제공합니다.

- **Performance Highlights**: 미국의 한 기술 회사에서 이 평가 방법론을 사용하여 두 가지 기능을 평가하며, 이러한 방법이 실제로 적용되었음을 보여주는 사례 연구를 포함합니다.



### Neural Cover Selection for Image Steganography (https://arxiv.org/abs/2410.18216)
- **What's New**: 최근 스테가노그래피(steganography) 분야에서 커버 이미지 선택(cover selection)의 중요성이 대두되고 있습니다. 기존 방법론은 지루한 탐색(exhaustive searches)에 의존했지만, 본 연구에서는 사전 훈련된 생성 모델(pretrained generative models)의 잠재 공간(latent space)을 최적화하여 보다 효과적인 커버 이미지를 선택할 수 있는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 제안된 프레임워크에서는 커버 이미지를 잠재 벡터(latent vector)로 변환한 후, 이를 사전 훈련된 생성 모델을 통해 재구성하는 방식으로 작동합니다. 이 과정에서 신경 스테가노그래픽 인코더(neural steganographic encoder)를 사용하여 비밀 메시지를 임베딩하고, 디코더를 통해 메시지를 복구합니다. 이를 통해 메시지 복구 오류를 최소화하며 이미지의 시각적 품질을 유지하는 커버 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 최적화된 이미지는 원본 이미지에 비해 메시지 복구 오류율이 크게 감소하였습니다. 또한 이미지 품질 측정 지표에서도 현저한 개선을 확인했으며, 스테가노그래피에 필요한 특정 비밀 메시지에 최적화된 커버 이미지를 생성할 수 있는 가능성을 보여주었습니다.



### Advancing NLP Security by Leveraging LLMs as Adversarial Engines (https://arxiv.org/abs/2410.18215)
Comments:
          5 pages

- **What's New**: 이 논문에서는 Large Language Models (LLMs)를 활용하여 다채로운 적대적 공격을 생성하는 새로운 접근 방식을 제안하고 있습니다. 최근 연구 결과를 바탕으로 LLM가 단어 수준의 적대적 예제를 생성하는 데 효과적임을 보여주었으며, 이를 다양한 공격 유형으로 확장하려고 합니다.

- **Technical Details**: LLMs는 맥락에 맞는 텍스트 조각을 생성하여, 오분류를 일으킬 수 있는 적대적 패치(adversarial patches), 여러 입력에 걸쳐 효과적인 범용 변형(universal perturbations), 특정 misclassification을 겨냥한 타겟 공격(targeted attacks) 등의 다양한 적대적 공격을 생성할 수 있습니다.

- **Performance Highlights**: LLM을 사용하여 생성된 적대적 예제는 인간이 작성한 텍스트와 구별이 어렵고, 텍스트의 본래 의도(intent)를 유지하면서도 분류기를 속이는데 효과적일 것입니다. 이는 NLP 시스템의 신뢰성과 안전성을 향상시키는데 큰 기여를 할 것으로 기대됩니다.



### Movement Control of Smart Mosque's Domes using CSRNet and Fuzzy Logic Techniques (https://arxiv.org/abs/2410.18123)
- **What's New**: 이 논문은 인공지능 기술을 활용하여 모스크의 공기를 신선하게 유지하고 태양광을 차단 없이 들어오게 하는 스마트 돔 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 날씨 조건과 모스크 내 혼잡도에 따라 돔의 움직임을 제어합니다. Saudi Arabia의 기상 데이터베이스와 Shanghai Technology Database에서 수집된 데이터를 기반으로 Congested Scene Recognition Network (CSRNet)와 Fuzzy 기법이 파이썬(Python) 프로그래밍 언어를 사용하여 적용됩니다.

- **Performance Highlights**: 기술의 적용 결과, 제안된 모델은 효과적으로 작동하며 매 시간마다 돔을 몇 분 동안 자동으로 열어 신선한 공기를 공급하는 정확한 지속 시간을 설정합니다.



### PixelGaussian: Generalizable 3D Gaussian Reconstruction from Arbitrary Views (https://arxiv.org/abs/2410.18979)
Comments:
          Code is available at: this https URL

- **What's New**: 새로운 PixelGaussian 프레임워크를 제안하여 임의의 시점에서 일반화 가능한 3D Gaussian 재구성을 효율적으로 학습할 수 있도록 합니다.

- **Technical Details**: PixelGaussian은 지오메트릭 복잡도에 따라 Gaussian 분포와 개수를 동적으로 조정하여 효율적인 표현과 재구성 품질 향상을 달성합니다. 이를 위해 Cascade Gaussian Adapter(CGA)와 transformer 기반의 Iterative Gaussian Refiner(IGR) 모듈을 사용하여 Gaussian 표현을 정제합니다.

- **Performance Highlights**: PixelGaussian은 ACID 및 RealEstate10K 데이터셋에서 기존 방법들을 초월하는 성능을 보이며, 더 많은 입력 뷰에서도 일관된 성능을 발휘합니다.



### CAMEL-Bench: A Comprehensive Arabic LMM Benchmark (https://arxiv.org/abs/2410.18976)
Comments:
          10 pages, 5 figures, NAACL

- **What's New**: CAMEL-Bench라는 포괄적인 아랍어 대규모 멀티모달 모델 평가 벤치마크가 개발되었습니다. 이는 4억 명 이상의 아랍어 사용자를 위한 다양한 도메인과 하위 도메인으로 구성되어 있습니다.

- **Technical Details**: CAMEL-Bench는 8개의 다양한 도메인과 38개의 하위 도메인으로 구성되어 있으며, 총 29,036개의 질문이 포함되어 있습니다. 데이터는 GPT-4o를 이용해 아랍어로 번역되거나 수작업으로 수집되었습니다.

- **Performance Highlights**: GPT-4o 시리즈는 다양한 멀티모달 이해 작업에서 높은 성능을 보였으나, 아랍어 멀티모달 데이터 처리에서 개선이 필요함을 보여주었습니다.



### Unbounded: A Generative Infinite Game of Character Life Simulation (https://arxiv.org/abs/2410.18975)
Comments:
          18 pages; Project page: this https URL

- **What's New**: 전통적인 비디오 게임의 한계를 초월한 생성적 무한 게임(generative infinite game) 개념을 소개하고, 이를 통해 생성 AI를 활용한 새로운 게임인 Unbounded를 개발하였습니다. Unbounded는 플레이어가 자율 가상 캐릭터와 상호작용하고, 개방형 메커니즘을 통해 이야기를 생성하는 게임입니다.

- **Technical Details**: Unbounded는 다음과 같은 기술적 혁신을 포함합니다: (1) 게임 메커니즘, 내러티브 및 캐릭터 상호작용을 실시간으로 생성하는 특수화된 LLM(large language model) 및 (2) 다양한 환경에서 캐릭터를 일관되게 시각적으로 생성할 수 있는 새로운 동적 지역 이미지 프롬프트 어댑터(Regional Image Prompt Adapter)입니다. 이 기술들은 생동감 있는 캐릭터 시뮬레이션과 화면 간의 일관성 높은 시각적 생성이 가능합니다.

- **Performance Highlights**: 전통적인 접근 방식과 비교했을 때, 캐릭터 삶의 시뮬레이션, 사용자 지시 따르기, 내러티브 일관성 및 캐릭터와 환경의 시각적 일관성에서 유의미한 개선을 보여주었습니다. 특히, 우리가 개발한 distilled LLM은 대형 LLM과 비교했을 때도 상호작용 속도에서 비슷한 성능을 보여 줍니다.



### 3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation (https://arxiv.org/abs/2410.18974)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 2D 네트워크 아키텍처의 한계를 극복하기 위해 프리트레인된 이미지 확산 모델에 3D 기하 인식을 주입하는 플러그인 모듈인 3D-Adapter를 소개합니다. 이 방법은 3D 피드백 증강(3D feedback augmentation)을 통해 다중 뷰 기능을 일관된 3D 표현으로 변환하여 기존 모델의 기하학적 품질을 크게 향상시킴을 입증합니다.

- **Technical Details**: 3D-Adapter는 프리트레인된 이미지 확산 모델의 샘플링 루프 내에서 각 복원 단계가 이루어질 때 중간 다중 뷰 기능을 복호화하고 이를 이용해 기하학적으로 일관된 3D 표현을 생성합니다. 주요 기술로는 Gaussian splatting을 기반으로 한 피드포워드 버전과 신경장(field) 및 메쉬를 사용하는 훈련이 필요 없는 다목적 버전이 포함되어 있습니다.

- **Performance Highlights**: 3D-Adapter는 Instant3D, Zero123++와 같은 텍스트-멀티뷰 모델의 기하학적 품질을 크게 향상시키며, 텍스트-이미지에 대한 기본적인 Stable Diffusion을 사용할 때도 고품질 3D 생성을 가능하게 합니다. 또한 텍스트-3D, 이미지-3D, 텍스트-텍스처 및 텍스트-아바타 작업에서도 뛰어난 결과를 보였으며, 3D-Adapter의 넓은 적용 가능성을 보여줍니다.



### Deep Insights into Cognitive Decline: A Survey of Leveraging Non-Intrusive Modalities with Deep Learning Techniques (https://arxiv.org/abs/2410.18972)
- **What's New**: 이번 연구는 비침습적(non-intrusive) 방법을 사용하여 인지 감소(cognitive decline)를 조기에 예측하는 딥러닝(deep learning) 기술의 활용을 초점을 맞춘 최초의 리뷰를 제시합니다.

- **Technical Details**: 연구에서는 오디오(audio), 텍스트(text), 비디오(video), 이미지(image)를 포함한 여러 비침습적 모달리티(modality)를 분석하고, 각 기술의 주요 특징과 장점을 논의합니다. 특히 Transformer 아키텍처(architecture)와 파운데이션 모델(foundation models) 같은 최첨단 접근 방법(state-of-the-art approaches)을 포함하여 다중모델(multimodal models)을 개발하는 작업도 심도 있게 다룹니다.

- **Performance Highlights**: 연구 결과, 일반적으로 텍스트 모달리티가 가장 우수한 성과를 보이며 인지 감소 감지에 가장 관련성이 높음을 입증하였습니다. 또한 다양한 개별 모달리티 접근 방식을 결합한 다중모델은 거의 모든 시나리오에서 성능을 일관되게 향상시키는 것으로 나타났습니다.



### Context is Key: A Benchmark for Forecasting with Essential Textual Information (https://arxiv.org/abs/2410.18959)
Comments:
          Preprint; under review. First two authors contributed equally

- **What's New**: 이 연구에서는 'Context is Key' (CiK)라는 새로운 시간 시계열 예측 벤치마크를 제안합니다. 이 벤치마크는 숫자 데이터와 다양한 종류의 텍스트 컨텍스트를 결합하여 예측 모델이 두 가지 모달리티를 통합하도록 요구합니다.

- **Technical Details**: CiK 벤치마크는 통계 모델, 시간 시계열 기초 모델, LLM(대형 언어 모델) 기반 예측기를 포함하여 여러 방법론을 평가합니다. 연구팀은 LLM을 활용한 간단한 프롬프팅(prompts) 방법을 제안하고, 이 방법이 테스트된 모든 다른 기법보다 뛰어난 성과를 보였음을 보고합니다.

- **Performance Highlights**: 실험 결과는 컨텍스트 정보를 포함하는 것이 매우 중요하다는 것을 보여주며, LLM 기반 예측 모델을 사용할 경우 예기치 않은 성과를 나타냈지만 몇 가지 중요한 단점도 드러났습니다.



### Dynamic Vocabulary Pruning in Early-Exit LLMs (https://arxiv.org/abs/2410.18952)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 효율적인 추론을 위한 새로운 접근 방식을 제안합니다. 특히, 'early-exiting' 기술을 통해 중간 층에서 다음 토큰 예측을 가능하게 하여 효율성을 향상시키고자 합니다. 기존의 복잡한 사전 확률 추정 문제를 해결하기 위해, 테스트 시간에 동적으로 어휘(vocabulary)를 가지치기(prune)하는 방법을 도입하였습니다.

- **Technical Details**: 어휘 공간의 크기를 줄임으로써 각 토큰에 대한 사후 동적 어휘 가지치기를 진행합니다. 이는 초기 몇 개의 후보 배출에서 모델의 은닉 표현(hidden representation)을 전체 어휘에 매핑하지 않고, 가장 가능성이 높은 K개(K) 토큰을 선별합니다. 이후 이들 토큰을 기반으로 가중치 행렬을 가지치기하고, 나머지 후보 배출에서도 이 가지치기된 가중치를 사용합니다. 이 방법은 통해 성능을 유지하면서도 더 적은 연산량으로 신뢰도 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 동적 어휘 가지치기는 early-exit LLM의 신뢰도 추정 효율성을 향상시켜 FLOPs와 시간 효율성을 개선하는 데 성공했습니다. 또한, 모델의 성능을 유지하면서도 효율성을 증가시키는 경량 설계를 구현하였습니다.



### ANAVI: Audio Noise Awareness using Visuals of Indoor environments for NAVIgation (https://arxiv.org/abs/2410.18932)
Comments:
          8th Conference on Robot Learning (CoRL) 2024

- **What's New**: 이번 연구에서는 로봇의 경로 계획에서 소음 인식을 위해 실내 비주얼을 활용하는 새로운 방법을 제안합니다. 기존의 로봇들이 소음을 인식하지 못하는 문제를 해결하기 위해, 로봇이 주변 환경의 시각적 관찰을 통해 소음 수준을 예측하고 이를 기반으로 보다 조용한 경로를 계획할 수 있도록 Acoustic Noise Predictor (ANP) 모델을 개발하였습니다.

- **Technical Details**: 로봇이 움직일 때 발생하는 소음(예: 모터 소리, 스피커의 소리)을 고려하여 소음의 세기를 예측하는 ANP 모델을 구축했습니다. 시뮬레이션된 환경에서 로봇의 위치에 따라 청취자 위치에서 소리의 세기가 어떻게 변하는지를 학습하며, 다양한 실내 환경에서의 3D 스캔(Matterport) 데이터를 활용하여 소음 예측을 위한 훈련을 진행했습니다. ANAVI 프레임워크를 통해 로봇은 자신의 행동이 주변에 미치는 소음을 인식하고 이에 적응하여 더 조용한 경로를 찾을 수 있습니다.

- **Performance Highlights**: ANP 모델은 거리 기반 추정 방식보다 더 나은 예측 정확도를 보였으며, 시뮬레이션과 실제 실험에서 일관된 결과를 나타내었습니다. ANP의 섬세한 예측은 소리의 반사, 회절, 흡수 등의 건축적 특성과 재료의 영향을 효과적으로 반영하였습니다. 이러한 연구는 로봇이 소음 문제를 해결하면서도 효율성을 유지할 수 있는 방향성을 제시합니다.



### SegLLM: Multi-round Reasoning Segmentation (https://arxiv.org/abs/2410.18923)
Comments:
          22 pages, 10 figures, 11 tables

- **What's New**: SegLLM은 대화형 메모리를 통해 시각 및 텍스트 출력의 이전 세분화 결과를 입력 스트림에 통합하여 복잡한 사용자 의도를 이해하고, 이미 지정된 엔티티(대상)의 위치, 상호작용 및 계층적 관계를 기반으로 개체를 세분화할 수 있는 새로운 다중 상호작용 세분화 모델입니다.

- **Technical Details**: SegLLM은 Mask-Encoding 및 Reference Mask-Decoding 기법을 통해 LLM이 세분화 마스크 출력을 인식하고, 마스크 디코더가 과거 대화 맥락을 인식할 수 있도록 설계되었습니다. 이를 통해 세분화 모델이 사용자 의도에 맞춘 복잡한 쿼리 처리 능력을 가질 수 있게 됩니다.

- **Performance Highlights**: MRSeg 벤치마크에서 SegLLM은 기존 모델들을 20% 이상 초과하는 성능을 보였고, 단일 라운드 REFCOCO 벤치마크에서도 시멘테이션 정확도가 5.5% 상승했습니다. 제안된 방법은 다양한 질문 템플릿에 대한 강인성도 향상되어 9.6%의 성능을 향상시켰습니다.



### From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems (https://arxiv.org/abs/2410.18921)
- **What's New**: 본 연구는 기존의 대형 언어 모델(LLMs)이 논리적 불일치를 식별할 수 있는 능력을 평가하기 위해 FaultyMath라는 벤치마크 데이터셋을 제안합니다. 이 데이터셋은 일반 상식, 애매한 진술, 수학적 모순 등 다양한 문제를 포함합니다.

- **Technical Details**: FaultyMath 데이터셋은 여러 수학 범주(예: algebra, geometry, number theory)와 난이도 수준을 포함하며, LLM의 성능을 평가하기 위해 Simple Prompt 및 Chain-of-thought prompt와 같은 다양한 평가 기법을 사용합니다. LLM의 성능을 측정하기 위한 주요 지표는 F1 Measure, Accuracy, True Positives, False Positives 등입니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM은 여전히 Blind Solver로 작동하며, 논리적 사고를 요구하는 상황에서 필요한 추론 능력이 부족함을 보여주었습니다. 특히 Simple Prompt를 사용한 GPT-4가 모든 지표에서 가장 뛰어난 성능을 보였으며, 이는 인간 평가 결과와 가장 근접한 것으로 평가됩니다.



### Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling (https://arxiv.org/abs/2410.18912)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구는 다중 시점 RGB 비디오에서 로봇의 동작 궤적과 장면 동역학에 미치는 영향을 명시적으로 고려하여 객체의 동역학을 학습하는 새로운 프레임워크를 제안합니다. 3D Gaussian Splatting(3DGS) 기반의 입자 동역학 모델을 훈련하기 위해 Graph Neural Networks(GNNs)를 활용합니다.

- **Technical Details**: 이 방법은 비디오에서 sparsely 추출된 control particles를 통해 객체의 3D 변환을 보간하여 예측된 미래 객체 상태를 렌더링하며, 행동 조건화(action-conditioned)된 비디오 예측을 가능하게 합니다. 학습된 동역학 모델은 개체 조작 작업을 위한 모델 기반 계획 프레임워크에도 적용될 수 있습니다.

- **Performance Highlights**: 연구는 로프, 옷감, 봉제 인형 등 다양한 변형 가능한 물체에 대해 실험을 수행하여, 제안된 프레임워크가 복잡한 형태와 동역학을 모델링할 수 있음을 입증합니다. 결과적으로, 제안된 방법은 기존 시스템 식별 접근 방식보다 동작 예측 정확도와 비디오 품질에서 우수한 성능을 보였습니다.



### SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning and Deploymen (https://arxiv.org/abs/2410.18907)
- **What's New**: 이 논문에서는 몇 가지 인간 시연을 바탕으로 로봇 조작을 위한 대규모 데이터세트를 생성할 수 있는 자동화 시스템인 SkillMimicGen(SkillGen)을 제안합니다.

- **Technical Details**: SkillGen은 인간 시연을 조작 기술(manipulation skills)로 분할하고, 이러한 기술을 새로운 맥락에 적응시키며, 자유공간 전이(free-space transit) 및 전이 동작(transfer motion)을 통해 통합합니다. 또한, Hybrid Skill Policy(HSP) 프레임워크를 제안하여 SkillGen 데이터셋에서 기술의 시작, 제어 및 종료 컴포넌트를 학습할 수 있도록 합니다.

- **Performance Highlights**: SkillGen은 최신 데이터 생성 프레임워크에 비해 데이터 생성 및 정책 학습 성능을 크게 향상시키며, 다양한 장면 변형(scene variations)에 대한 데이터를 생성할 수 있는 능력을 보여줍니다. 24K 이상의 시연을 생성하고, 평균적으로 24% 더 성공적인 HSP 에이전트를 훈련시키는 데 성공했습니다. 또한 실제 조작 작업에 적용하여 긴 수평 조립 작업에서 제로샷 시뮬레이션-리얼 전이를 보여주었습니다.



### PRISM: A Methodology for Auditing Biases in Large Language Models (https://arxiv.org/abs/2410.18906)
- **What's New**: 본 논문은 PRISM이라는 유연하고 질문 기반의 방법론을 제안하며, LLM의 편견과 선호를 간접적으로 탐색하는 새로운 방식을 제시합니다.

- **Technical Details**: PRISM은 첫째, 정치적 성향을 평가하는 도구(예: Political Compass Test)를 선택하고, 둘째, 감사할 LLM을 선택한 후, 셋째, 모델에 역할을 할당하고, 넷째, 에세이를 작성하도록 지시하며, 다섯째, 평가자를 통해 에세이의 입장을 그에 따라 판단하는 방식으로 이루어집니다. 이러한 단계적 접근은 LLM의 기본 및 기꺼이 주장할 수 있는 위치를 매핑하는데 사용됩니다.

- **Performance Highlights**: 연구 결과, 감사된 21개의 LLM이 기본적으로 경제적으로 좌파적이며 사회적으로 자유주의적 성향을 갖고 있다는 것을 발견했습니다. PRISM은 LLM의 편향, 선호 및 제약을 보다 신뢰성 있게 탐색하는 데 유용함을 증명했습니다.



### Creating and Repairing Robot Programs in Open-World Domains (https://arxiv.org/abs/2410.18893)
Comments:
          Under review at ACL Rolling Review

- **What's New**: 이 논문에서는 LLM(대규모 언어 모델)을 활용하여 자연 언어로부터 로봇 프로그램을 생성하는 방법에 대해 설명합니다. 하지만 LLM이 생성한 프로그램은 지시사항의 모호성이나 작업에 대한 오해, 세계 상태에 대한 정보 부족으로 인해 결함이 발생할 수 있습니다. 저자들은 이러한 문제를 해결하기 위한 RoboRepair 시스템을 제안하고 있습니다.

- **Technical Details**: RoboRepair 시스템은 프로그램 실행 중 오류가 발생한 시점까지의 실행을 추적하고, 그 후 LLM이 생성한 복구 프로그램을 실행하여 반복되는 작업을 최소화합니다. 이 시스템은 복구 프로그램의 효율성을 평가하기 위해 11개 작업과 다양한 오류 조건으로 구성된 벤치마크를 생성하였으며, 미래 오류에 대한 정보가 있는 오라클(oracle)과 비교하여 복구 프로그램의 효율성을 평가합니다.

- **Performance Highlights**: RoboRepair 시스템은 이전에 성공적으로 완료된 단계를 반복하지 않으면서 현재 세계 상태에서 복구할 수 있는 메커니즘을 제공합니다. 이 논문은 LLM이 생성한 복구 프로그램의 효율성을 기존 방법과 비교하여 검증하고 있습니다.



### Diff-Instruct++: Training One-step Text-to-image Generator Model to Align with Human Preferences (https://arxiv.org/abs/2410.18881)
- **What's New**: 이번 연구에서는 human preference(인간 선호)에 맞춘 one-step text-to-image generator 모델의 정렬 문제를 처음으로 다룹니다. 이를 위해 human feedback(인간 피드백)을 활용하는 reinforcement learning(RLHF)에서 영감을 받아 인간 보상 함수를 최대화하는 접근 방식을 채택합니다.

- **Technical Details**: Diff-Instruct++(DI++)라는 새로운 방법을 제안하며, 이는 이미지 데이터가 필요하지 않은 인체 선호 정렬 기법입니다. DI++는 기술적 도전 과제를 극복하고 효과적인 손실 함수 개발에 기여하며, 이 모델의 수렴 속도가 빠릅니다.

- **Performance Highlights**: DiT 기반의 one-step text-to-image 모델은 COCO 검증 프롬프트 데이터셋에서 Aesthetic Score(미적 점수) 6.19, Image Reward 1.24, Human preference Score (HPSv2.0) 28.48을 달성하여 기존의 오픈소스 모델을 초초과한 성과를 보였습니다.



### Provably Robust Watermarks for Open-Source Language Models (https://arxiv.org/abs/2410.18861)
- **What's New**: 이번 논문은 오픈소스 대형 언어 모델(LLM)을 위한 최초의 워터마크(watermark) 기법을 제시합니다. 기존의 방법들이 모델의 비밀 매개변수에 의존하는 반면, 본 연구에서는 모델의 매개변수를 수정하여 워터마크를 삽입하되 모델의 출력으로만 감지할 수 있는 방식으로 접근합니다.

- **Technical Details**: 저자들은 수식어 기반 워터마크(sampler-based watermark)와 달리, 모델의 가중치를 수정하여 워터마크를 삽입하고, 이 워터마크는 특정 제약 조건 하에서 제거할 수 없음을 입증했습니다. 특히, 이 워터마크는 입력된 텍스트(약 300 토큰)로부터도 충분히 감지할 수 있습니다. 공격자가 워터마크를 제거하기 위해서는 모델 품질 점수를 100점 만점에 0점으로 떨어뜨리면서 감지율을 50%로 낮춰야 합니다.

- **Performance Highlights**: OPT-6.7B 및 OPT-1.3B 모델을 사용한 실험 결과, 노출된 파라미터로부터도 이 워터마크를 감지할 수 있으며, 실제 언어 생성에서 높은 엔트로피 텍스트에 대해서도 성공적으로 작동함을 보여주었습니다.



### DeCoRe: Decoding by Contrasting Retrieval Heads to Mitigate Hallucinations (https://arxiv.org/abs/2410.18860)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)의 hallucination 문제를 줄이기 위한 새로운 디코딩 방법인 Decoding by Contrasting Retrieval Heads (DeCoRe)를 제안합니다. DeCoRe는 retrieval heads를 마스킹하여 hallucination을 유도하고, 기저 LLM과 마스킹된 LLM의 출력을 대비하는 방식으로 정보를 증폭시킵니다.

- **Technical Details**: DeCoRe는 특정 retrieval heads를 마스킹하여 hallucination을 유도하고, 동적 조건부 엔트로피를 사용하여 대비 디코딩 메커니즘을 조정합니다. 이는 조건부 엔트로피를 이용하여 모델의 다음 토큰 분포를 제어하며, 이를 통해 보다 정확한 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, DeCoRe는 XSum에서 18.6%, MemoTrap에서 10.9%, NQ-Open에서 2.4% 및 NQ-Swap에서 5.5%의 성능 개선을 보여주며, 특히 높은 문맥 충실성이 요구되는 작업에서 유의미한 결과를 기록했습니다. 또한, DeCoRe는 TruthfulQA와 같은 사실 기억 작업에서 다른 방법들에 비해 정확도를 높였습니다.



### DL-Polycube: Deep learning enhanced polycube method for high-quality hexahedral mesh generation and volumetric spline construction (https://arxiv.org/abs/2410.18852)
- **What's New**: 본 논문에서는 고품질의 헥사헤드ral (hex) 메쉬를 생성하기 위한 새로운 알고리즘, DL-Polycube를 제안합니다. 이 알고리즘은 surface triangular meshes과 polycube 구조 간의 연결을 세우고, deep neural network를 활용하여 메쉬 분류 및 자동 segmentation을 수행합니다.

- **Technical Details**: DL-Polycube 알고리즘은 triangular meshes를 polycube 구조로 분류하고, 이를 기반으로 unsupervised learning을 통해 surface segmentation을 수행합니다. 생성된 polycube 구조는 octree subdivison, parametric mapping 및 품질 개선 기법을 통해 헥스 메쉬로 변환됩니다. 최종적으로, 생성된 hex 메쉬에서 자르고 계층 구조의 B-splines를 구성하고, 이를 통해 trivariate Bézier 요소를 추출하여 isogeometric analysis에 사용합니다.

- **Performance Highlights**: DL-Polycube 알고리즘은 자동화된 surface segmentation과 polycube 구조 생성을 통해 헥스 메쉬 생성 속도를 크게 향상시키며, 여러 사례를 통해 알고리즘의 견고성을 입증하였습니다.



### Expanding AI Awareness Through Everyday Interactions with AI: A Reflective Journal Study (https://arxiv.org/abs/2410.18845)
Comments:
          Accepted and presented at the Frontiers in Education 2024 (FIE2024)

- **What's New**: 이 논문은 기술 교육 프로그램에서 학생들이 AI와의 일상적인 상호작용을 반영하는 방식에 대한 연구를 다룹니다. 구체적으로, 학생들은 AI에 대한 인식과 참여를 교실 내외의 경험을 통해 탐색합니다.

- **Technical Details**: 연구는 22명의 학부생을 대상으로 6주 동안 진행된 반영 저널 연구로, 학생들은 매주 AI와의 상호작용에 대한 저널 항목을 제출했습니다. 저널 프로프트는 학생들이 읽거나 관찰한 내용을 통해 자신의 관점과 지식을 반영하도록 설계되었습니다.

- **Performance Highlights**: 학생들은 저널 작성을 통해 AI의 日常적인 존재 및 잠재적 이점과 단점을 더욱 인식하게 되었다고 보고했습니다. 이 연구는 AI 인식 및 리터러시 분야의 지속적인 작업에 기여합니다.



### Learning to Explore with Lagrangians for Bandits under Unknown Linear Constraints (https://arxiv.org/abs/2410.18844)
- **What's New**: 이 논문에서는 다중 무장 배틀리아( Multi-Armed Bandit) 모델에서 안전, 자원 및 공정성을 고려하여 순수 탐색(pure exploration) 문제를 다룹니다. 구체적으로 알려지지 않은 선형 제약(linear constraints) 하에서 r-적합 feasible policy를 식별하는 방법을 제안합니다.

- **Technical Details**: 저자들은 제약 조건이 있는 순수 탐색을 위한 샘플 복잡도(sample complexity) 하한의 라그랑지 완화 (Lagrangian relaxation)를 도입하고, 이 하한이 제약의 순차적 추정(sequential estimation)과 어떻게 변화하는지를 보여줍니다. 또한, Track-and-Stop 및 Gamified Explorer 알고리즘의 두 가지 효율적인 확장인 LATS 및 LAGEX를 제안하고, 제약에 적응하는 정지 규칙을 적용하여 매 단계에서 가능한 세트의 비관적 추정을 사용합니다.

- **Performance Highlights**: LAGEX와 LATS는 다양한 보상 분포 및 제약 조건을 바탕으로 수치 실험을 통해 기본 알고리즘과 비교하여 효율적인 성능을 달성함을 입증하였습니다. 이를 통해 알고리즘들이 제약 조건에 의존하는 상한까지 점근적으로 최적의 샘플 복잡도를 나타낸다는 것을 보여주었습니다.



### From Efficiency to Equity: Measuring Fairness in Preference Learning (https://arxiv.org/abs/2410.18841)
- **What's New**: 이 논문은 AI 시스템에서 인간의 다양한 선호를 공정하게 반영하는 것을 보장하기 위해, 특히 preference learning 모델에서의 epistemic fairness를 평가하는 새로운 프레임워크를 제안합니다. 기존의 경제적 불평등 이론(inequality)과 Rawlsian 공정성(justice)에서 영감을 받아, 공정성을 정량화하는 지표를 제공합니다.

- **Technical Details**: 공정성을 측정하기 위해 Gini Coefficient, Atkinson Index, Kuznets Ratio와 같은 지표를 활용하였습니다. 이 접근법은 인공지능의 선택이 특정 집단을 우선시함으로써 발생할 수 있는 epistmological injustice를 해결하는 데 초점을 맞추고 있습니다. 논문에서는 사용자 그룹 간에 모델 성능 및 공정성의 차이를 분석하며, 다양한 선호를 아우르는 보상 모델(reward model)을 제안합니다.

- **Performance Highlights**: AI-EDI-Space와 Jester Jokes 데이터세트를 통해 검증된 우리의 접근법은 높은 성능을 보인 모델에서도 epistemic injustice가 여전히 나타날 수 있음을 보여줍니다. 따라서, 인공지능이 다양한 인간 경험을 공정하게 표현할 수 있는 시스템으로 발전하기 위한 방향성을 제시합니다.



### From English-Centric to Effective Bilingual: LLMs with Custom Tokenizers for Underrepresented Languages (https://arxiv.org/abs/2410.18836)
- **What's New**: 이번 논문에서는 영어와 타겟 언어를 지원하는 이중 언어 대규모 언어 모델(LLM)을 개발하기 위한 모델 독립적인 비용 효율적인 접근법을 제안했습니다. 우크라이나어, 아랍어, 조지아어와 같은 비 라틴 문자 언어를 대상으로 실험을 수행하며, 이전보다 개선된 언어 성능과 함께 계산 비용을 줄이는 방법을 입증하였습니다.

- **Technical Details**: 제안된 방법에서는 어휘 확장, 새로운 임베딩 초기화 및 모델 훈련 및 평가가 포함됩니다. Gemma 2 및 Mistral 모델을 통해 검증한 결과, 우크라이나어 및 아랍어 모델의 경우 새로운 어휘를 지속적으로 훈련함으로써 조건부 생성성과 문법 정확도가 향상되었습니다. 또한, 코드 스위칭 및 존재하지 않는 단어의 비율을 측정하기 위한 새로운 지표도 도입하였습니다.

- **Performance Highlights**: 모델 훈련 동안 제안된 토큰화 방법은 우크라이나어와 아랍어 모델의 계산 복잡성과 추론 시간을 줄이며, 코드 스위칭 및 문법 정확성 과제에서 성능 향상을 보여주었습니다. 더불어, 영어-조지아 모델의 성능 개선에도 어휘 확장이 기여했습니다.



### Towards Visual Text Design Transfer Across Languages (https://arxiv.org/abs/2410.18823)
- **What's New**: 이번 연구에서는 시각적 텍스트 디자인을 언어 간 번역하는 Multimodal Style Translation (MuST-Bench)라는 새로운 작업을 소개합니다. 이는 디자인 의도를 유지하며 다양한 글쓰기 시스템 간에 번역할 수 있습니다.

- **Technical Details**: SIGIL (Style Integrity and Glyph Incentive Learning) 프레임워크를 도입하여 스타일 설명이 필요 없는 다중 모드 스타일 번역을 가능하게 합니다. 이 프레임워크는 다국어 설정을 위한 glyph latent, 안정적인 스타일 지침을 위한 사전 훈련된 VAE, 가독성 문자인식을 최적화하기 위한 강화 학습 피드백이 포함된 OCR 모델을 통해 이미지를 생성하는 모델을 개선합니다.

- **Performance Highlights**: SIGIL은 기존의 기초 모델들보다 우수한 스타일 일관성과 가독성을 달성하며, 시각적 충실성을 유지하는 데에도 뛰어납니다. MuST-Bench 베이스라인 데이터를 통해 성능이 입증되었으며, 영어에서 한국어를 포함한 다섯 가지 언어로의 전환 능력이 평가되었습니다.



### Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances (https://arxiv.org/abs/2410.18775)
- **What's New**: 이번 연구에서는 W-Bench라는 최초의 종합 워터마크 평가 벤치마크를 제안하여 다양한 이미지 편집 기술에 대해 워터마크 방법의 강인성을 평가합니다. 이를 통해 여러 대표적인 워터마킹 방법들이 이러한 편집 후 워터마크를 인식하지 못하는 경우가 많다는 사실을 보여줍니다.

- **Technical Details**: 제안하는 VINE 방법은 두 가지 주요 혁신을 포함하여 이미지 편집에 대한 강인성을 크게 향상시킵니다. 첫째, 이미지 편집의 주파수 특성을 분석하여 블러링 왜곡이 유사한 주파수 특성을 갖는 것을 발견하고, 이를 훈련 중 대체 공격으로 사용하여 워터마크 강인성을 강화합니다. 둘째, 대규모로 사전 훈련된 확산 모델인 SDXL-Turbo를 워터마킹 작업에 적응시켜 더욱 인지되지 않으면서 강인한 워터마크 임베딩을 달성합니다.

- **Performance Highlights**: 실험 결과, VINE 방법은 다양한 이미지 편집 기술에 대해 뛰어난 워터마크 성능을 달성하여 기존 방법들에 비해 이미지 품질과 강인성 모두에서 성능이 우수함을 보여줍니다.



### Does Differential Privacy Impact Bias in Pretrained NLP Models? (https://arxiv.org/abs/2410.18749)
Comments:
          Github this https URL

- **What's New**: 이 연구는 미세 조정된 대형 언어 모델에서 Differential Privacy (DP)가 바이어스(bias)에 미치는 영향을 실증 분석을 통해 보여줍니다. DP 훈련이 보호된 그룹을 대상으로 AUC 기반 바이어스 메트릭에 따라 모델 바이어스를 증가시킬 수 있으며, 이는 모델이 보호된 그룹과 다른 그룹 간 긍정 및 부정 예제를 구별하기 어렵게 만든다는 사실을 제시합니다.

- **Technical Details**: 이 연구는 BERT 모델을 미세 조정하며, 다양한 프라이버시 예산(privacy budget)으로 모델을 학습시키고 6개의 정체성(Identity) 하위 그룹에서 다양한 메트릭을 통해 바이어스를 측정합니다. 모델 바이어스는 유해한 언어 탐지 작업을 통해 평가되며, Jigsaw Unintended Bias 및 UCBerkeley의 Hate Speech 데이터셋을 활용하여 DP가 바이어스에 미치는 영향을 다각도로 분석합니다.

- **Performance Highlights**: 모델을 다양한 프라이버시 수준으로 훈련시킨 결과, DP 훈련이 AUC 기반 메트릭에서 모델의 바이어스를 증가시킨다는 것을 발견하였습니다. 또한, DP는 프리트레인(pre-trained)된 LLM에서 모델 유틸리티에 부정적인 영향을 미친다는 결과를 발표하였습니다. 이 연구는 DP와 바이어스 간의 새로운 통찰을 제공하여 NLP 연구자들이 이 두 가지 요소를 효과적으로 통합할 수 있도록 돕습니다.



### Cellpose+, a morphological analysis tool for feature extraction of stained cell images (https://arxiv.org/abs/2410.18738)
- **What's New**: 이 연구에서는 Cellpose라는 최첨단 세포 분할 프레임워크의 기능을 확장하여 형태학적 특성을 평가할 수 있는 기능 추출을 도입하고, DAPI 및 FITC 염색 세포의 새로운 데이터 세트를 소개합니다.

- **Technical Details**: 이 논문은 딥 러닝 기반의 자동화된 이미지 분석 방법을 사용하여 세포의 정확한 분할 및 특징 추출을 수행하는 방법에 대해 설명합니다. 특히, 다양한 신경망 구조를 활용해 세포의 핵 및 세포를 인식하는 과정을 pixelwise classification 문제로 설정하여 확률 맵을 생성합니다. 이를 통해 객체의 위치를 확인하고 비최대 억제(non-maximum suppression) 기법을 통해 정확도를 높입니다.

- **Performance Highlights**: Cellpose와 기능 추출 알고리즘을 결합하여 실험 데이터의 정확성과 효율성을 극대화하였으며, 새로운 데이터 세트를 통해 세포 분석의 가능성을 더욱 확장하고 있습니다.



### GeoLoRA: Geometric integration for parameter efficient fine-tuning (https://arxiv.org/abs/2410.18720)
- **What's New**: 이 논문에서는 기존의 Low-Rank Adaptation (LoRA)의 한계를 극복하기 위해 GeoLoRA라는 새로운 동적 저순위 훈련 방법을 제안합니다. GeoLoRA는 매트릭스 미분 방정식에서의 동적 저순위 수렴 이론을 활용하여, 훈련 과정에서 파라미터 예산을 효율적으로 할당합니다.

- **Technical Details**: GeoLoRA는 작은 순위 어댑터에 대해 단일 역전파(backpropagation) 패스를 필요로 하여 계산 비용을 크게 절감하고, AdaLoRA와 같은 기존의 방법들보다 빠릅니다. 또한, GeoLoRA는 저순위 요소의 정확한 직교성을 유지하여 저순위 최적화에서 발생하는 문제를 피합니다.

- **Performance Highlights**: GLUE 벤치마크 및 Vision Transformers, Stable Diffusion과 같은 여러 테스트에서 GeoLoRA는 기존의 방법들보다 향상된 정확도와 계산 효율성을 보여주었습니다.



### Low-Latency Video Anonymization for Crowd Anomaly Detection: Privacy vs. Performanc (https://arxiv.org/abs/2410.18717)
Comments:
          16pages, 8 figures, 9 tables

- **What's New**: 이번 연구에서는 기존의 깊은 학습 모델 기반의 개인 정보 보호 기법에 대한 한계를 극복하기 위해 경량 적응형 익명화(LA3D) 기법을 제안합니다. 이 기법은 실시간 비디오 이상 탐지(VAD) 애플리케이션에 특화되어 있으며, 개인의 프라이버시를 보호하면서 효율적인 비디오 분석을 가능하게 합니다.

- **Technical Details**: LA3D는 고속 세분화 모델을 사용하여 비디오 프레임에서 사람의 피사체 마스크를 식별하고, 각 감지된 신체부위에 대해 상대적인 특성에 따라 동적 모자이크를 적용합니다. 이 방법은 비디오 프레임의 깊이 변화에 따라 하이퍼파라미터를 동적으로 조정하여 개인 정보를 보호하는 데 집중합니다.

- **Performance Highlights**: 실험 결과, LA3D는 기존 알고리즘보다 개인 정보 익명화 능력을 크게 향상시키면서도 비디오 이상 탐지의 유효성을 저하시키지 않음을 입증하였습니다. 이전의 경량 전통 이미지 익명화 기법들을 재조명하고, 효과적인 개인 정보 보호 및 VAD 애플리케이션을 위한 새로운 전환점을 만들어가고 있습니다.



### Uncovering the Genetic Basis of Glioblastoma Heterogeneity through Multimodal Analysis of Whole Slide Images and RNA Sequencing Data (https://arxiv.org/abs/2410.18710)
- **What's New**: 이번 연구에서는 다중 모달 딥러닝(multimodal deep learning) 기법을 활용하여 희귀하고 공격적인 뇌암인 신경교종(glioblastoma)의 이질성(heterogeneity)을 조사하였습니다.

- **Technical Details**: 전체 슬라이드 이미지(whole-slide images)와 RNA-seq(RNA 시퀀싱) 분석을 결합하여 유전자 프로파일(genetic profiles)을 식별했습니다. 새로운 RNA-seq 데이터 인코딩(encoding) 방법을 도입하여 신경교종의 진행 경과(patter of progression)를 설명할 수 있는 특정 유전적 프로파일을 발견하였습니다.

- **Performance Highlights**: 이 연구는 신경교종의 이질성을 구성하는 유전적 메커니즘(genetic mechanisms)에 새로운 통찰을 제공하며, 치료 개입(therapeutic intervention)을 위한 잠재적 타겟(potential targets)을 강조합니다.



### How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs (https://arxiv.org/abs/2410.18697)
- **What's New**: 논문에서는 LITEVAL-CORPUS라는 문학 기계 번역(MT) 평가를 위한 새로운 병렬 코퍼스를 소개합니다. 이 코퍼스는 9개의 MT 시스템의 출력과 검증된 인간 번역을 포함하여 2천 개 이상의 문단과 13,000개의 주석된 문장을 포함하고 있습니다. 이를 통해 MT의 평가 방법을 비교하고, 비문학 MT에서는 표준으로 사용되던 MQM이 문학 번역에 대해 부족한 점을 강조합니다.

- **Technical Details**: LITEVAL-CORPUS는 4개의 언어 쌍에 걸쳐 2,000개 이상의 문단으로 구성되어 있으며, MQM, SQM, BWS와 같은 다양한 주석 체계의 일관성과 적합성을 조사합니다. Multidimensional Quality Metrics (MQM)는 문학 번역 평가에 적합하지 않으며, SQM은 주석자의 전문성에 따라 효과가 다릅니다. 최근 LLM 기반 메트릭 중 GEMBA-MQM이 다른 메트릭에 비해 우수하나, 인간 번역과 LLM 출력 간의 구별이 어렵습니다.

- **Performance Highlights**: 인간 전문 번역가는 LLM 번역보다 항상 우수한 성과를 보였으며, GPT-4o가 두 번째로 높은 점수를 기록했습니다. 자동 메트릭은 일반적으로 인간 MQM 및 SQM과 중간 정도의 상관관계를 보였으나, 인간 번역을 정확히 식별하는 데에는 어려움을 겪었습니다.



### Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch (https://arxiv.org/abs/2410.18693)
Comments:
          Preprint. Project page: this https URL

- **What's New**: ScaleQuest라는 새로운 데이터 합성 방법을 소개하며, 저비용 및 대규모 질문 생성을 가능하게 합니다.

- **Technical Details**: ScaleQuest는 '소형' 오픈소스 모델을 활용하여 복잡한 증강 제약 없이 질문을 생성합니다. 두 단계의 질문 조정 프로세스인 Question Fine-Tuning (QFT)와 Question Preference Optimization (QPO)을 통해 질문 생성 능력을 활성화합니다.

- **Performance Highlights**: ScaleQuest를 통해 1백만 개의 문제-해답 쌍으로 구성된 데이터셋을 자동으로 생성하였고, 이는 기존의 오픈소스 데이터셋보다 효과적입니다. MATH 문제에서 29.2%에서 46.4%의 성능 향상을 보여주며, Qwen2-Math-7B-Base 모델을 조정하면 강력한 모델인 Qwen2-Math-7B-Instruct를 초과하는 성능을 기록하였습니다.



### Ali-AUG: Innovative Approaches to Labeled Data Augmentation using One-Step Diffusion Mod (https://arxiv.org/abs/2410.18678)
- **What's New**: Ali-AUG는 산업 분야에서 효율적인 레이블된 데이터 증대(data augmentation)를 위한 새로운 단일 단계 확산(diffusion) 모델로, 효율적인 합성 레이블 이미지 생성을 가능하게 합니다.

- **Technical Details**: Ali-AUG는 안정적인 확산 아키텍처를 활용하여 마스크(mask)와 이미지를 효율적으로 통합하며, 정확한 기능 삽입(feature insertion)을 보장합니다. 또한, Low-Rank Adaptation (LoRA) 모듈을 통해 모델의 안정성과 효율성을 향상시킵니다.

- **Performance Highlights**: Ali-AUG는 다양한 산업 데이터셋에서 기존 증대 방법보다 31% 성능을 개선하고, 데이터 증대 없이 훈련한 모델보다 45% 향상된 성능을 보여줍니다. 훈련 시간 또한 32% 단축되며, 쌍(pair) 및 비쌍(unpaired) 데이터셋 모두에서 유연한 사용이 가능합니다.



### Health Misinformation in Social Networks: A Survey of IT Approaches (https://arxiv.org/abs/2410.18670)
Comments:
          Preprint -- Under review in the ACM Transactions on Computing for Healthcare (HEALTH) journal

- **What's New**: 이번 연구는 정보기술 관점에서 소셜 네트워크의 의료 허위정보 문제에 대한 포괄적인 조사 결과를 제시합니다. 이 조사는 의료 허위정보와 관련된 연구에 대한 체계적인 리뷰를 제공하고 연구자와 실무자가 이 빠르게 변화하는 분야를 이해하도록 돕기 위해 구성되었습니다.

- **Technical Details**: 이 논문은 수동적 및 자동적 사실 확인(fact-checking) 접근 방식을 소개하고, 콘텐츠(content), 전파 특징(propagation features), 출처 특징(source features)을 이용한 가짜 뉴스 탐지(fake news detection) 방법을 탐색합니다. 또한 허위정보 확산을 저지하기 위한 완화 접근(mitigation approaches)도 논의합니다. 여러 의료 허위정보 데이터셋과 공개 사용 가능한 도구 목록도 제공됩니다.

- **Performance Highlights**: 이 연구는 200편 이상의 논문과 24개의 공개 데이터셋, 11개의 사실 확인 도구를 참조하여 의료 분야에서의 가짜 뉴스 탐지 기술의 현황을 조사합니다. 특히 COVID-19와 관련된 허위정보에 대한 대응을 위한 노력이 강조됩니다. 또한, 대부분의 의료 종사자들이 COVID-19 환자를 치료하는 데 있어 의료 허위정보가 크게 방해가 되었다고 언급했습니다.



### $C^2$: Scalable Auto-Feedback for LLM-based Chart Generation (https://arxiv.org/abs/2410.18652)
Comments:
          Preprint

- **What's New**: 본 논문에서는 인간 개입의 비용을 줄이기 위해 고품질 차트를 생성하는데 있어 새로운 자동 피드백 생성기(automatic feedback generator), ChartAF를 소개합니다. 이 시스템은 참고(reference) 없이 작동하며, 데이터 세트 ChartUIE-8K를 활용하여 데이터 다양성을 크게 개선합니다.

- **Technical Details**: 새로운 프레임워크 $C^2$는 (1) 자동 피드백 제공자(ChartAF)와 (2) 다양한 참고 없는 데이터 세트(ChartUIE-8K)로 구성됩니다. ChartUIE-8K는 기존 기준 대비 각각 5982%, 1936%, 91%의 쿼리, 데이터, 차트 유형을 증가시켜 데이터 다양성을 크게 향상시킵니다.

- **Performance Highlights**: 첫 번째 실험에서 74%의 응답자가 피드백 후 결과를 강하게 선호했으며, 10%는 선호한다고 답했습니다. 두 번째 후속 실험에서 ChartAF는 아홉 개의 기준선(Baseline)보다 우수한 성능을 보여주었습니다. 또한 LLM 사용자 연구에서 참여자의 94%가 ChartUIE-8K의 쿼리를 선호하고, 93%가 실제 사용 사례와 일치한다고 평가했습니다.



### Smart ETL and LLM-based contents classification: the European Smart Tourism Tools Observatory experienc (https://arxiv.org/abs/2410.18641)
- **What's New**: 이번 연구는 유럽 스마트 관광 도구(STTs) 관측소의 콘텐츠 업데이트를 개선하기 위해 STTs를 통합하고 분류하는 데 중점을 둡니다.

- **Technical Details**: 이 연구는 PDF 카탈로그에서 STTs에 대한 정보를 추출하기 위해 PDF-scraping 기술을 사용합니다. 추출된 정보는 QR 코드, 이미지, 링크 및 텍스트 정보를 포함합니다. 중복 STTs는 제거되고, 남은 항목은 Large Language Models (LLMs)을 사용하여 텍스트 정보를 기반으로 분류됩니다. 최종적으로, 데이터는 Dublin Core 메타데이터 구조에 맞게 변환됩니다.

- **Performance Highlights**: Smart ETL(procedure) 과정은 PDF-scraping 기술과 LLMs를 결합하여 텍스트 콘텐츠 기반 분류를 수행하는 데 성공적이었습니다. 초기 결과는 LLMs가 텍스트 콘텐츠 기반 분류에서 효과적임을 보여줍니다.



### Diffusion Attribution Score: Evaluating Training Data Influence in Diffusion Mod (https://arxiv.org/abs/2410.18639)
- **What's New**: 최근 아카이브 논문에서는 diffusion 모델의 훈련 샘플 기여도를 정확하게 평가하기 위한 새로운 방법인 Diffusion Attribution Score (DAS)를 제안하고 있습니다. 기존 방법들은 diffusion loss를 기반으로 샘플의 기여도를 측정했으나, 이는 정확한 평가를 제공하지 못합니다. DAS는 훈련 샘플의 포함 여부에 따라 예측 분포의 차이를 측정하여 기여도를 직접적으로 비교합니다.

- **Technical Details**: DAS는 기존의 data attribution 방법들이 갖는 한계를 극복하기 위해 설계되었습니다. 이전의 방법들은 predicted distribution과 ground truth distribution 간의 divergence를 측정하는데, 이는 예측 분포 간의 간접적인 비교를 초래합니다. DAS는 이와 다르게, 샘플의 포함 및 제외 시 예측 분포의 차이를 직접적으로 분석하여 모델 출력에 대한 훈련 샘플의 중요성을 평가합니다. 또한 DAS의 계산을 가속화하기 위한 다양한 전략도 모색하였습니다.

- **Performance Highlights**: DAS는 여러 벤치마크 테스트를 통해 이전의 기준을 뛰어넘는 성과를 보여주었으며, 특히 linear data-modelling score에서 탁월한 성능을 보였습니다. 이는 DAS가 다양한 데이터셋과 diffusion 모델에서도 효과적으로 작동함을 입증합니다.



### Little Giants: Synthesizing High-Quality Embedding Data at Sca (https://arxiv.org/abs/2410.18634)
- **What's New**: 본 논문은 SPEED라는 프레임워크를 도입하여 오픈소스의 소규모 모델(8B)을 정렬하여 대규모 합성 임베딩 데이터 생성을 효율적으로 이루어질 수 있게 한다. 기존의 GPT-4와 같은 비공식 모델에 의존하는 대신, 저비용으로 고품질의 데이터를 제공할 수 있는 방안을 제시하다.

- **Technical Details**: SPEED 프레임워크는 (1) GPT-4를 사용해 다채로운 태스크 설명 생성, (2) 주니어 발생 모델을 통해 초기 데이터 생성, (3) 시니어 발생 모델 수립을 위한 선호 최적화 과정 및 (4) 데이터 수정기를 통한 자기 개선 과정을 포함한다. 이 모델을 통해 생성된 데이터는 고급 임베딩 기능을 활용할 수 있게 되어 성능 향상을 이루도록 설계되었다.

- **Performance Highlights**: 실험 결과, SPEED는 E5_mistral와 비교할 때 10분의 1 미만의 GPT API 호출만으로도 우수한 성능을 보여주며, 임베딩 모델의 성능이 합성 임베딩 데이터 크기와 로그-선형 관계에 있음을 발견하였다.



### Wavetable Synthesis Using CVAE for Timbre Control Based on Semantic Lab (https://arxiv.org/abs/2410.18628)
Comments:
          6 pages, 4 figures, Accepted at APSIPA ASC 2024

- **What's New**: 본 연구는 직관적이고 합리적인 timbre 제어 방법을 제시하며, 특히 semantic labels를 활용하여 사용자가 원하는 timbre를 쉽게 정의할 수 있도록 합니다. 이를 통해 wavetable synthesis (WTS)에서의 timbre 제어 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구는 conditional variational autoencoder (CVAE)을 활용하여 timbre를 정의하는 데 사용됩니다. CVAE는 convolutional 및 upsampling layers를 포함하고 있으며, 사용자에게 제공된 semantic labels을 기반으로 wavetable을 선택하고 timbre를 정의할 수 있습니다. encoder와 decoder로 구성된 모델을 통해 입력된 waveform (x)와 조건 (c)을 잠재 공간 (z)으로 매핑합니다. 손실 함수는 reconstruction loss와 Kullback-Leibler (KL) divergence로 구성됩니다.

- **Performance Highlights**: 실험 결과, 본 연구의 접근 방식은 semantic 입력을 사용하여 wavetable의 timbre를 효과적으로 제어할 수 있으며, 실시간 성능을 보장합니다. 이를 통해 음악 제작 및 공연에서의 timbre 제어가 더 직관적이고 용이해집니다.



### SAMG: State-Action-Aware Offline-to-Online Reinforcement Learning with Offline Model Guidanc (https://arxiv.org/abs/2410.18626)
- **What's New**: 이번 논문에서는 SAMG(State-Action-Conditional Offline-to-Online Reinforcement Learning with Offline Model Guidance)라는 새로운 패러다임을 제안합니다. SAMG는 기존의 O2O 알고리즘과 달리 사전 훈련된 critic을 고정하여 각 상태-행동 쌍에 대한 오프라인 값을 제공함으로써 효율성을 높입니다.

- **Technical Details**: SAMG는 오프라인 데이터에 직접적으로 훈련하지 않고, 프리트레인된 오프라인 critic을 동결하여 오프라인 모델의 값을 이용합니다. 이 값들은 Bellman 방정식과 결합되며, 정책 상태-행동 인식 계수를 가중치로 사용합니다. 이 계수는 조건부 변량 자동 인코더(C-VAE)로부터 유도되며, 오프라인 데이터의 신뢰성을 상태-행동 수준에서 포착하는 데 목적이 있습니다.

- **Performance Highlights**: SAMG는 D4RL 벤치마크에서 네 개의 최첨단 O2O RL 알고리즘보다 우수한 성능을 나타냈으며, 이론적인 분석 결과 최적성과 낮은 추정 오류를 보였습니다.



### Prompting and Fine-Tuning of Small LLMs for Length-Controllable Telephone Call Summarization (https://arxiv.org/abs/2410.18624)
Comments:
          Accepted at the The International Conference on Foundation and Large Language Models (FLLM2024)

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 전화 통화 요약 시스템의 빠른 개발을 탐구합니다. 기존 LLM을 사용하여 전화 통화 요약을 생성하는 초기 실험을 진행했으며, 그 후 더욱 강력한 모델을 활용한 맞춤형 합성(training dataset) 데이터를 생성하였습니다. 특히, 생성된 데이터의 다양성과 생성된 요약의 길이를 제어하는 데 초점을 맞추어 다양한 사용 사례의 요구를 충족시키고자 했습니다.

- **Technical Details**: 이 연구의 핵심은 Llama-2-7B 모델을 기반으로 하여, 요약 작업에 대해 특별히 조정된 데이터로 모델을 미세 조정하는 것입니다. 요약 품질을 평가하기 위해 최신의 LLM-as-a-judge 평가 기법을 활용하였으며, 연구 결과 Llama-2-7B 기반 모델이 사실 정확성, 완전성 및 간결성 측면에서 GPT-4와 동등한 성능을 보였습니다. 특정 작업에 맞춤화된 데이터로 훈련받았을 때 모델 성능이 크게 개선되는 점이 강조됩니다.

- **Performance Highlights**: 결과적으로, 연구팀의 방법론은 효율적이고 실용적인 전화 통화 요약 시스템을 빠르게 구축하는 잠재력을 가지고 있음을 보여주었습니다. 요약 생성에서의 사실 정확성 및 간결성은 GPT-4 수준이며, 더 작은 LLM도 특정 작업에 맞춰 교육함으로써 성능 격차를 줄일 수 있음을 입증했습니다.



### FairQueue: Rethinking Prompt Learning for Fair Text-to-Image Generation (https://arxiv.org/abs/2410.18615)
Comments:
          Accepted in NeurIPS24

- **What's New**: 이번 연구는 텍스트-이미지 변환에서 공정한 생성(Fair T2I generation)을 달성하기 위해 기존의 프롬프트 학습(Prompt Learning) 방법이 샘플 질 저하라는 문제를 야기할 수 있음을 밝혔습니다.

- **Technical Details**: 연구자들은 프롬프트 학습 기반 접근 방식이 학습된 프롬프트와 참조 이미지 간의 임베딩 차이를 정렬하는 것을 목표로 했으나, 이로 인해 프롬프트가 왜곡되고 생성된 이미지 품질이 저하된다는 것을 발견했습니다. 이 과정에서 크로스-어텐션 맵(Cross-Attention Maps)을 분석하여 초기 디노이징 단계에서의 비정상성을 파악하였습니다.

- **Performance Highlights**: 연구에서 제안한 Prompt Queuing과 Attention Amplification 방법은 기존 SOTA 접근 방식보다 향상된 이미지 생성 품질을 보여주었고, 다양한 민감한 속성(tSA)에 대해 경쟁력 있는 공정성을 달성했습니다.



### TripCast: Pre-training of Masked 2D Transformers for Trip Time Series Forecasting (https://arxiv.org/abs/2410.18612)
Comments:
          Accepted by ICONIP 2024

- **What's New**: 이 연구에서는 관광 산업의 시간 시리즈 데이터가 고유한 2D 구조를 가지고 있다는 점에 착안하여, TripCast라는 새로운 모델링 패러다임을 제안합니다. 이 방식은 여행 시간 시리즈를 2D 데이터로 취급하여 마스킹(masking)과 재구성(reconstruction) 과정을 통해 표현을 학습합니다.

- **Technical Details**: TripCast 모델은 시간 시리즈의 이벤트 시간(event time)과 선행 시간(leading time)을 동시에 고려하여 지역적(local) 및 전역적(global) 의존성을 학습하는 방식으로 설계되었습니다. 데이터 전처리를 위해 입력을 고차원(latent vector)으로 변환하고, 마스킹을 통해 관측된 값과 관측되지 않은 값을 구분합니다. 모델은 대규모 실제 데이터를 바탕으로 사전 학습(pre-trained)되어 다른 최신 모델들보다 더 우수한 성능을 나타냅니다.

- **Performance Highlights**: 실험 결과, TripCast는 도메인 내(in-domain) 예측 시나리오에서 기존의 심층 학습(deep learning) 및 사전 학습 모델보다 월등한 성능을 보였고, 도메인 외(out-domain) 시나리오에서도 우수한 확장성(scalability)과 전이 가능성(transferability)을 입증했습니다.



### STTATTS: Unified Speech-To-Text And Text-To-Speech Mod (https://arxiv.org/abs/2410.18607)
Comments:
          11 pages, 4 Figures, EMNLP 2024 Findings

- **What's New**: 이 논문에서는 음성 인식(ASR)과 음성 합성(TTS) 모델을 함께 학습하는 새로운 방법론인 STTATTS를 제안합니다. 기존의 개별 모델과는 달리, 이 모델은 공유된 매개변수를 사용하여 다중 작업 손실(multi-task loss) 목표를 통해 효율적으로 훈련됩니다.

- **Technical Details**: STTATTS 모델은 Transformer 아키텍쳐를 기반으로 하며, 다중 작업을 동시에 처리하기 위해 미리 훈련된 인코더와 디코더를 활용합니다. 이 모델은 음성과 텍스트를 처리할 수 있는 단일 인코더-디코더 구조를 가지고 있으며, 평균적으로 두 작업을 결합할 때 필요한 매개변수의 수를 약 50% 감소시킵니다. 새로운 MLP 기반의 작업 융합(task fusion) 모듈을 통해 여러 작업을 동시에 학습할 수 있게 설계되었습니다.

- **Performance Highlights**: STTATTS는 단일 인코더-디코더 아키텍쳐를 이용하여 ASR과 다중 화자 TTS를 동시에 학습함으로써 성능이 향상되었고, 공개된 모델 중에서 VoxLM과 비교하여 매개변수 수가 절반에 불과하지만 더 좋은 성능을 보였습니다. 또한, 이 모델은 자원 풍부한 언어인 영어뿐만 아니라 데이터가 부족한 아랍어에도 적용 가능성을 보여줍니다.



### Taipan: Efficient and Expressive State Space Language Models with Selective Attention (https://arxiv.org/abs/2410.18572)
- **What's New**: Taipan은 Mamba-2와 Selective Attention Layers (SALs)를 결합한 새로운 하이브리드 아키텍처로, 긴 컨텍스트 언어 모델링의 효율성을 극대화합니다. 이 아키텍처는 능률성을 유지하면서도 Transformer와 유사한 성능을 제공하는 특징이 있습니다.

- **Technical Details**: Taipan은 Mamba의 Markov 가정을 보완하기 위해 SALs를 도입하여 입력 시퀀스 내 장거리 의존성을 가진 주요 토큰을 선택합니다. 선택된 토큰은 중요하지 않은 정보를 제거한 후, 어텐션 모듈을 통해 장거리 의존성을 캡처합니다. 이를 통해 Taipan은 메모리 집약적인 작업에서의 성능을 개선합니다.

- **Performance Highlights**: 실험 결과 Taipan은 여러 규모와 작업에서 우수한 성능을 보였으며, 특히 메모리 집약적인 작업인 긴 컨텍스트 검색 및 구조화된 정보 추출에서 이전 모델인 Mamba-2 대비 큰 개선을 입증했습니다. Taipan은 1111만 개의 토큰까지 높은 성능을 유지하며 효율적인 생성 능력을 보여, 고급 언어 처리 작업을 위한 강력한 아키텍처로 자리매김하고 있습니다.



### Zero-shot Object Navigation with Vision-Language Models Reasoning (https://arxiv.org/abs/2410.18570)
Comments:
          Accepted by the International Conference on Pattern Recognition (ICPR) for Oral presentation

- **What's New**: 본 논문에서는 Language-driven Zero-Shot Object Navigation (L-ZSON)에서 로봇의 탐색 및 상호작용을 안내하기 위한 자연어 명령을 포함하는 새로운 Vision Language 모델인 VLTNet을 제안합니다. 이 모델은 Tree-of-Thought(Module) 네트워크를 기반으로 합니다.

- **Technical Details**: VLTNet은 네 가지 주요 모듈로 구성됩니다: 비전 언어 모델 이해(vision language model understanding), 의미 맵(mapping), Tree-of-thought(ToT) 추론 및 탐색(tree-of-thought reasoning and exploration), 목표 식별(goal identification). 특히 ToT 추론 모듈은 로봇 탐색 중 탐색의 경계를 선택하는 데 혁신적으로 사용되며, 복잡한 경로 추론 과정과 필요 시 백트래킹을 포함합니다.

- **Performance Highlights**: PASTURE 및 RoboTHOR 벤치마크에서 실험 결과, L-ZSON에서 VLTNet의 성능이 뛰어남을 입증했습니다. 특히 복잡한 자연어 명령이 포함된 시나리오에서 우수한 성능을 보였습니다.



### Bielik 7B v0.1: A Polish Language Model -- Development, Insights, and Evaluation (https://arxiv.org/abs/2410.18565)
- **What's New**: Bielik 7B v0.1는 70억 개의 파라미터를 가진 폴란드어 생성 텍스트 모델로서, 다수의 혁신적인 기술을 통해 폴란드어 처리에서의 주요 문제들을 해결합니다. 특히 Weighted Instruction Cross-Entropy Loss와 Adaptive Learning Rate 기술이 적용되었습니다.

- **Technical Details**: Bielik 7B v0.1 모델은 Transformer 아키텍처를 기반으로 하며, Self-attention, Grouped-query attention (GQA), Sliding Window Attention, SwiGLU 활성화 함수, Rotary Positional Embeddings (RoPE), Root Mean Square Layer Normalization (RMSNorm)과 같은 고급 기술을 포함합니다. 이 모델은 또한 기존의 Mistral 7B v0.1 모델에서 발전하였으며, 36억 개의 토큰으로 구성된 훈련 데이터셋을 사용합니다.

- **Performance Highlights**: Bielik 7B v0.1는 RAG Reader 작업에서 Mistral-7B-v0.1에 비해 평균 점수가 9 포인트 상승했습니다. 또한 Polish MT-Bench에서 Reasoning (6.15/10) 및 Role-playing (7.83/10) 카테고리에서 뛰어난 성과를 보였습니다.



### Complexity Matters: Effective Dimensionality as a Measure for Adversarial Robustness (https://arxiv.org/abs/2410.18556)
- **What's New**: 이번 연구에서는 모델의 유효 차원(Effective Dimensionality)과 그 Robustness 간의 관계를 조사하였으며, 이는 모델 선택 및 Robustness 평가에서 유용한 기준으로 제안되었습니다.

- **Technical Details**: 연구에서는 YOLO, ResNet과 같은 상업적 규모의 모델들을 사용하여 유효 차원과 적대적(Adversarial) Robustness 사이의 거의 선형 반비례 관계를 발견했습니다. 적대적 훈련 기법이 유효 차원에 미치는 영향을 분석하여 동일한 반비례 관계가 존재함을 확인하였습니다.

- **Performance Highlights**: 모델의 유효 차원이 낮을수록 더욱 향상된 Robustness 특성을 보이며, 파라미터 수나 이전에 테스트된 측정 방법들보다 더 정교하고 효과적인 메트릭을 제공하는 것으로 나타났습니다.



### IMAN: An Adaptive Network for Robust NPC Mortality Prediction with Missing Modalities (https://arxiv.org/abs/2410.18551)
Comments:
          The paper has been accepted by BIBM 2024

- **What's New**: 본 논문에서는 누락된 모달리티에 대한 강력한 예측력을 가진 nasopharyngeal carcinoma (NPC)의 사망 예측을 위한 새롭게 개발된 IMAN 네트워크를 제안합니다. IMAN은 복잡한 다중 모달 데이터의 처리를 개선하기 위해 Dynamic Cross-Modal Calibration (DCMC), Spatial-Contextual Attention Integration (SCAI), Context-Aware Feature Acquisition (CAFA)라는 세 가지 핵심 모듈을 포함하고 있습니다.

- **Technical Details**: IMAN 네트워크는 다양한 데이터를 활용하여 예측력을 강화하는 missing-aware prompt 모듈로 사전 훈련됩니다. DCMC 모듈은 의료 이미지와 텍스트 데이터의 크기 조정 및 정렬을 통해 이질적인 입력의 정규화를 개선합니다. SCAI 모듈은 자기 주의 메커니즘에 위치 정보를 통합하여 특성 융합을 향상시키고, CAFA 모듈은 학습된 오프셋을 통해 다양한 비율과 방향에서의 적응형 특성 캡처를 수행합니다.

- **Performance Highlights**: IMAN은 NPC 데이터셋에서 광범위한 실험을 통해 누락된 데이터가 있는 상황에서도 뛰어난 예측 정확도를 보여줍니다. 이 통합 접근법은 보다 일관되고 정확한 치료 결과 예측을 가능하게 하여 NPC 진단 및 치료 계획의 중요한 발전을 나타냅니다.



### On Explaining with Attention Matrices (https://arxiv.org/abs/2410.18541)
- **What's New**: 이번 논문은 Transformer 모델의 attention weights (AW)와 예측 결과 간의 인과적 연결을 검토하며, 최근의 연구 결과를 반박합니다. 이 연구에서는 AW가 설명 가능성이 없다는 주장에 대한 정교한 대안으로 '효율적 Attention'을 도입합니다.

- **Technical Details**: 효율적 Attention은 AW가 설명 역할을 수행하는 작업과 모델에서 주의 매트릭스의 효과적인 구성 요소를 분리하여 계산합니다. 본 연구에서는 효율적 Attention에 의해 생성된 매트릭스가 확률 분포임을 입증하며, 이를 통해 기대하는 퀄리티의 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 데이터세트에 대한 실험에서, 효율적 AW가 기존의 AW와 동일한 예측 결과를 나타내며, 효율적 AW의 입증 가능성 및 인과적 설명을 지원하는 결과들을 보여줍니다.



### LOGO -- Long cOntext aliGnment via efficient preference Optimization (https://arxiv.org/abs/2410.18533)
- **What's New**: 이 논문에서는 LOGO(Long cOntext aliGnment via efficient preference Optimization)라는 새로운 훈련 전략을 소개하며, 이는 긴 맥락 정렬을 위한 선호 최적화를 도입한다. 이 전략은 긴 입력 시퀀스에 대한 생성 성능을 높이기 위해 설계되었다.

- **Technical Details**: LOGO는 두 가지 주요 요소를 포함한다: 1) 올바른 출력(정상적인 결과)과 잘못된 출력(환각 등)을 구별하도록 LCM(Long-context model)을 유도하는 훈련 목표, 2) 공개 모델만을 사용하여 데이터를 구성하는 파이프라인. 또한, LOGO는 참조 없는 훈련 목표와 위치 합성 방법을 채택하여 GPU 메모리 문제를 극복한다.

- **Performance Highlights**: LOGO를 사용하여 Llama-3-8B-Instruct-80K 모델이 16시간동안 단일 8×A800 GPU에서 0.3B 데이터로 훈련함으로써 GPT-4와 유사한 성능을 달성한다. LOGO는 또한 모델의 맥락 창 크기를 늘리고 생성 성능을 향상시킬 수 있는 가능성을 보여준다.



### KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing (https://arxiv.org/abs/2410.18517)
Comments:
          Under Review by ICLR2025

- **What's New**: 최근의 대형 언어 모델 (LLMs)들은 Transformer 아키텍처를 기반으로 하여 뛰어난 성능을 보였지만, 이로 인해 상당한 GPU 메모리 요구가 발생하고 있습니다. 본 논문에서는 KV (key-value) 캐시를 층 간으로 공유하여 메모리 소비를 줄이고 성능을 유지하는 새로운 방법인 KVSharer를 제안합니다.

- **Technical Details**: KVSharer는 레이어 간 KV 캐시를 공유하여 레이어 단위의 압축을 수행하는 플러그-앤-플레이 방법입니다. 이는 직관적으로 유사한 KV 캐시를 공유하는 것이 아니라, 상반된 특성을 가진 KV 캐시를 공유할 때 모델 성능을 더 잘 유지할 수 있다는 경험적 발견에 기반합니다. 이 방법은 미세 조정 없이도 잘 훈련된 LLM에 바로 적용할 수 있습니다.

- **Performance Highlights**: KVSharer를 사용하면 KV 캐시 계산량을 30% 감소시킬 수 있으며, 이로 인해 메모리 소비가 줄어들고 성능에는 큰 영향을 미치지 않는 것으로 실험에서 나타났습니다. 또한, 최소 1.3배의 생성 속도 향상을 달성할 수 있으며, 기존의 intra-layer KV 캐시 압축 방법과의 호환성도 보장합니다.



### Enhancing Graph Attention Neural Network Performance for Marijuana Consumption Classification through Large-scale Augmented Granger Causality (lsAGC) Analysis of Functional MR Images (https://arxiv.org/abs/2410.18506)
Comments:
          17 pages

- **What's New**: 이번 연구는 대규모 증강 그랜저 인과성(large-scale Augmented Granger Causality, lsAGC)을 활용하여 대마초 사용자와 일반 대조군 간의 뇌 네트워크 연결성을 구분하는 도구로서의 효과를 조사했습니다.

- **Technical Details**: 본 연구는 차원 축소와 특정 시간 시계열(Source Time-Series)의 증가를 통합한 모델을 통해 시계열(time-series) 예측을 수행하며, fMRI 데이터 간의 지향적 인과관계를 추정합니다. lsAGC는 다변량(multivariate) 접근법으로 모든 다른 시간 시계열을 고려하면서 내재된 동적 시스템의 연결을 밝혀냅니다. 연구는 ADHD 진단을 받은 60명의 성인 데이터를 활용했습니다.

- **Performance Highlights**: 상관계수 방법을 이용한 평균 정확도는 약 52.98%였으며, lsAGC 접근법은 평균 61.47%의 정확도와 함께 우수한 성능을 보였습니다. 제안된 방법은 뇌 네트워크 연결성 분석에서 지향적 인과 관계를 고려해야 할 필요성을 강조합니다.



### Dialog2Flow: Pre-training Soft-Contrastive Action-Driven Sentence Embeddings for Automatic Dialog Flow Extraction (https://arxiv.org/abs/2410.18481)
Comments:
          Accepted to EMNLP 2024 main conference

- **What's New**: 이번 논문에서는 비주석 대화(dialog)에서 구조화된 워크플로우(workflow)를 효율적으로 추출하는 새로운 방법론을 제안합니다. 이를 위해 Dialog2Flow (D2F) 임베딩(embedding)을 도입하여 발화를 정보 및 의사소통 기능에 따라 군집화합니다. 이 과정은 대화를 연속적인 궤적(trajectory)으로 모델링할 수 있게 해줍니다.

- **Technical Details**: D2F는 발화를 잠재 공간(latent space)으로 매핑하여 식별된 동작(action)과 관련된 영역으로 군집화합니다. 대화 데이터를 통합하여 표준화된 행동 주석이 포함된 대규모 데이터셋을 구축하였으며, 대화 행동의 의미 정보를 활용하는 새로운 소프트 대조 손실(soft contrastive loss)을 도입하여 표현 학습을 안내합니다.

- **Performance Highlights**: D2F는 다양한 도메인에서 고유한 정성적 및 정량적(metrically) 결과를 보여주며, 기존의 문장 임베딩(sentence embeddings) 기법들보다 향상된 성능을 발휘하였습니다. 특히, 기본적인 대화 흐름 추출에 있어 전담으로 사전 학습된 첫 번째 문장 임베딩 모델로 자리매김하고 있습니다.



### Multi-Stage Airway Segmentation in Lung CT Based on Multi-scale Nested Residual UN (https://arxiv.org/abs/2410.18456)
- **What's New**: 이번 연구에서는 폐 CT 이미지에서 기도(segmentation)를 정확하고 완전하게 수행하기 위해 Multi-scale Nested Residual U-Net (MNR-UNet)이라는 새롭게 설계된 구조를 제안합니다. 이 구조는 다양한 스케일의 입력과 Residual Multi-scale Modules (RMM)를 포함하여 정보 흐름을 향상시킵니다.

- **Technical Details**: MNR-UNet은 다중 스케일 정보를 결합하고 잔여 연결(residual connections)을 통해 세밀한 기도 구조 디테일을 포착하며, gradient vanishing 문제를 감소시킵니다. 또한, Weighted Breakage-Aware Loss (wBAL)을 도입하여 기도의 연속성을 더욱 섬세하게 유지하도록 설계된 세 단계의 세그멘테이션 파이프라인을 따릅니다.

- **Performance Highlights**: 개발된 접근법이 공공 데이터셋과 자체 데이터셋 모두에서 실행된 검증 결과, 기존 방법 대비 기도 분할의 기술적 완전성과 세밀한 기도 구조 추출에서 크게 향상된 성과를 보여주었습니다. 또한, 기도의 브랜치를 더욱 정확하게 식별하는 데에 성공했습니다.



### Verifying Non-friendly Formal Verification Designs: Can We Start Earlier? (https://arxiv.org/abs/2410.18454)
Comments:
          Published in DVCon Europe 2024

- **What's New**: 이 논문에서는 Systems on Chips (SoCs)의 복잡성이 증가함에 따라 발생하는 안전-critical 환경에서의 치명적 실패를 방지하기 위해 Formal Property Verification (FPV)과 High-level Equivalence Checking (HLEC) 도구를 사용하는 방법을 제안합니다.

- **Technical Details**: 제안된 자동화된 방법론은 두 단계로 구성됩니다. 첫 번째 단계에서는 C++로 작성된 untimed algorithmic description을 생성된 assertions를 사용하여 초기 검증을 수행합니다. 이 단계의 장점은 소프트웨어 레벨의 assertions가 몇 초 안에 실행되므로 RTL (Register Transfer Level) 디자인을 작성하기 전에 알고리즘에 대한 확정적인 결과를 얻을 수 있다는 것입니다. 두 번째 단계에서는 HLEC 및 관련 메타모델 매개변수를 사용하여 알고리즘 설명이 순차적 디자인과 검증됩니다.

- **Performance Highlights**: 이 방법론은 알고리즘 설명과 관련된 버그를 조기에 발견할 수 있으며, HLEC 검증을 위한 설정 준비를 도와줍니다. 이는 도구를 설정하고 속성을 수동으로 작성할 때 발생할 수 있는 오류를 줄여줍니다. 제안된 프레임워크는 데이터 경로에 작업 중인 팀들이 검증 흐름의 초기 단계에서 검증과 의사 결정을 지원할 수 있게 합니다.



### The Nature of Mathematical Modeling and Probabilistic Optimization Engineering in Generative AI (https://arxiv.org/abs/2410.18441)
Comments:
          19 pages, 3 figures

- **What's New**: 이 논문에서는 Transformer 모델의 핵심 요소에 대한 수학적 문제 공식화 및 확률적 최적화 방법의 심층 분석을 제공합니다. 새로운 서브워드 인코딩(SWE) 최적 솔루션과 하이퍼파라미터 최적화를 위한 크로스 엔트로피 최적화 방법이 제안됩니다.

- **Technical Details**: 제안된 SWE 솔루션은 byte-pair encoding (BPE) 알고리즘과 유사한 초기 설정을 기반으로 하며, WordPiece 접근 방식과 유사한 목표를 갖습니다. 또한, word2vec 모델을 위한 하이퍼파라미터 최적화에 크로스 엔트로피 방법을 도입합니다. RoPE (Rotary Positional Encoding)와 ALiBi (Attention with Linear Biases)의 조합을 통해 시퀀스 길이와 모델 성능을 향상시킵니다. 마지막으로 PrFlashAttention 방법을 통해 주의 계산에 참여할 가능성이 있는 블록을 결정합니다.

- **Performance Highlights**: 제안된 SAQ (Staircase Adaptive Quantization) 방법은 Multi-Query Attention (MQA)에서 적절한 모델 품질과 비용 절감을 이루면서 점진적인 양자화 저하를 달성합니다.



### MoMQ: Mixture-of-Experts Enhances Multi-Dialect Query Generation across Relational and Non-Relational Databases (https://arxiv.org/abs/2410.18406)
- **What's New**: 이번 연구에서는 자연어(Natural Language)를 구조화된 쿼리 언어인 SQL로 변환하는 데 있어 새로운 프레임워크인 MoMQ(Mixture-of-Experts-based multi-dialect query generation framework)를 제안합니다. MoMQ는 다양한 데이터베이스 방언(Dialect)을 지원하고, SQL 생성에서 발생하는 문제들을 효과적으로 해결합니다.

- **Technical Details**: MoMQ는 특정 방언에 대해 전문화된 전문가 그룹(Dialect Expert Group)과 다단계 라우팅 전략(Multi-level Routing Strategy)을 채택하여 방언 특화 지식을 관리하고 쿼리 생성 과정에서의 간섭을 줄입니다. 또한, 자원 불균형(Resource Imbalance) 문제를 해결하기 위해 공유 전문가 그룹(Shared Expert Group)을 도입하여 고자원 방언의 일반 지식을 저자원 방언으로 전이(Transfer)합니다.

- **Performance Highlights**: MoMQ는 다양한 데이터베이스에 걸친 실험에서 실행 정확도(Execution Accuracy)를 평균 3-5% 향상시키며, 데이터 불균형 설정에서는 4-6%의 개선을 보였습니다. 이러한 결과는 MoMQ가 다양한 방언에서의 간섭 및 데이터 불균형 문제를 효과적으로 처리할 수 있음을 보여줍니다.



### A Unimodal Speaker-Level Membership Inference Detector for Contrastive Pretraining (https://arxiv.org/abs/2410.18371)
- **What's New**: 이번 연구에서는 회원 추론 공격(member inference attack) 분야에 새로운 기법인 USMID를 제안합니다. 이 기법은 텍스트 데이터만을 사용하여 CLAP 모델의 프라이버시 누수를 탐지하는 도구로, 음성 데이터를 노출시키지 않고도 저비용으로 효율적인 탐지가 가능합니다.

- **Technical Details**: USMID는 CLAP 모델을 이용하여 텍스트 데이터에서 피처 벡터(feature vector)를 추출하고, 생성된 텍스트 가벼운 자극(gibberish)으로부터 여러 개의 이상 감지기(anomaly detector)를 훈련시켜, 각 테스트 텍스트의 피처 벡터를 입력받아 훈련 세트에 포함되어 있는지 여부를 판단합니다.

- **Performance Highlights**: 다양한 CLAP 모델 아키텍처 및 데이터셋에서 폭넓은 실험을 실시한 결과, USMID는 기존의 방법들보다 뛰어난 성능을 보이며, 오직 텍스트 데이터만으로도 프라이버시 누수를 효과적으로 탐지하는 것으로 나타났습니다.



### Data Publishing in Mechanics and Dynamics: Challenges, Guidelines, and Examples from Engineering Design (https://arxiv.org/abs/2410.18358)
Comments:
          21 pages, 8 figures

- **What's New**: 최근 엔지니어링 분야에서 데이터 기반 방법들이 더욱 중요해지고 있으며, 이는 특히 딥 러닝(Deep Learning) 인공 신경망의 성공 사례들에 의해 촉진되고 있습니다.

- **Technical Details**: 이 논문은 엔지니어링 설계 작업에서 데이터 출판(data publishing)의 가치와 도전에 대해 분석합니다. 특히 기계 및 역학 분야에서 발생하는 특정한 문제점을 다루며, 데이터 기반 방법이 요구하는 새로운 데이터 공유 관행을 제안합니다.

- **Performance Highlights**: 논문에서 제시된 연구 예제들은 데이터 출판의 실용적인 방법론을 설명하며, 기계 및 역학 분야의 과학자들이 데이터 기반 연구를 성공적으로 수행할 수 있도록 돕는 데 기여할 수 있습니다.



### The Impact of Generative Artificial Intelligence on Ideation and the performance of Innovation Teams (Preprint) (https://arxiv.org/abs/2410.18357)
Comments:
          24 pages, 5 figures, Author Contributions: Michael Gindert: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Project administration, Funding acquisition Marvin Lutz Müller: Validation, Investigation, Resources, Writing - Review & Editing, Supervision

- **What's New**: 이 논문은 Generative Artificial Intelligence (GenAI)가 혁신 팀의 아이디어 생성 단계에서의 역학과 성과에 미치는 영향을 조사합니다. AI 증강 아이디에이션 도구를 활용하여 연구하였으며, Knowledge Spill-over Theory of Entrepreneurship을 적용했습니다.

- **Technical Details**: 연구는 실험군과 대조군으로 나뉜 참가자들을 대상으로 한 구조화된 필드 실험(frame field experiment)을 통해 수행되었습니다. 결과적으로 AI 증강 팀이 더 높은 품질의 아이디어를 적은 시간 안에 생성했다는 것을 발견했습니다. GenAI의 적용은 효율성, 지식 교환, 만족도 및 참여도 증가와 더불어 아이디어 다양성 향상으로 이어졌습니다.

- **Performance Highlights**: 이 연구 결과는 AI가 혁신 관리 분야에서 변혁적인 역할을 하고 있음을 강조합니다. GenAI는 Entrepreneurship의 Knowledge Spillover Theory에서 중요한 요소에 긍정적인 영향을 미치며, 혁신 및 경제 성장에 대한 잠재적 영향을 부각시킵니다. 향후 연구는 GenAI와 창의적 프로세스 간의 역동적인 상호작용을 더 탐구할 필요가 있습니다.



### Aggregated Knowledge Model: Enhancing Domain-Specific QA with Fine-Tuned and Retrieval-Augmented Generation Models (https://arxiv.org/abs/2410.18344)
- **What's New**: 이 논문에서는 Lawrence Berkeley National Laboratory (LBL)의 ScienceIT 분야에서 닫힌 도메인 질문 응답(QA) 시스템을 향상시키기 위한 새로운 접근 방식을 소개합니다. 풍부한 데이터셋을 활용하여 세 가지 형태의 질문-답변 쌍을 생성하며, 집합 지식 모델(Aggregated Knowledge Model, AKM)을 제안하여 다양한 모델의 응답을 통합합니다.

- **Technical Details**: 논문은 AWS Bedrock, GCP PaLM2, Meta LLaMA2, OpenAI GPT-4 및 Google Gemini-Pro와 같은 최신 대형 언어 모델(LLMs)을 사용하여 QA 성능을 향상시키는 방법을 설명합니다. K-means clustering을 사용해 다양한 응답을 통합한 집합 지식 모델(AKM)은 보다 대표적인 답변을 선택합니다. 이 과정에서 데이터 처리 기술을 활용하여 양질의 질문-답변 쌍을 생성했습니다.

- **Performance Highlights**: 평가 결과는 다양한 지표를 통해 제안된 모델들이 LBL ScienceIT 환경에 적합하고 효과적임을 입증합니다. 특히, AKM은 파인튜닝과 검색 증강 전략을 통합하여 상당한 성능 개선을 보여주었습니다. 이 연구에서 얻은 통찰은 특정 도메인에 맞춘 전문 QA 시스템 개발에 기여할 수 있습니다.



### Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems (https://arxiv.org/abs/2410.18336)
- **What's New**: 본 연구는 AI 시스템이 단순히 정확한 수학 문제의 답변을 생성하는 것 이상으로, 새로운 해결책을 개발하거나 사람을 도와야 한다고 주장합니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)의 수학적 사고에서의 창의적 잠재력을 탐구합니다.

- **Technical Details**: 이 연구에서는 CreativeMath라는 새로운 프레임워크와 벤치마크를 도입합니다. 이 벤치마크는 중학교 수준의 문제부터 올림픽 수준의 경쟁 문제까지 다루며, LLM이 제공된 일부 알려진 해결책 이후에 혁신적인 해결책을 제안하는 능력을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 표준 수학 작업에서 좋은 성능을 보였으나 창의적 문제 해결 능력은 상당한 차이를 보였습니다. 특히 Gemini-1.5-Pro 모델이 독창적인 해결책을 생성하는 데 있어 다른 LLM보다 뛰어난 성과를 보였습니다.



### Search-Based Path Planning among Movable Obstacles (https://arxiv.org/abs/2410.18333)
- **What's New**: 이 논문은 Path planning Among Movable Obstacles (PAMO)를 조사하며, 로봇이 필요한 경우 경로상의 이동 가능한 장애물을 밀어낼 수 있도록 하여 정적 장애물 사이에서 최소 비용의 충돌 없는 경로를 찾는 방법을 제시합니다.

- **Technical Details**: PAMO 문제는 로봇과 이동 가능한 장애물 위치를 포함하는 방대한 상태 공간을 탐색해야 하며, 객체 수가 증가함에 따라 기하급수적으로 성장합니다. 저자들은 PAMO*와 하이브리드 상태 PAMO*를 제안하며, 이를 통해 2D 점유 그리드를 기반으로 한 다중 목표 및 자원 제약 문제를 해결합니다.

- **Performance Highlights**: PAMO*는 복잡한 환경에서 최대 400개의 객체가 있는 경우 1초 이내에 최적 솔루션을 찾을 수 있으며, 하이브리드 상태 PAMO*는 로봇과 객체 간의 높은 충실도를 갖춘 상호작용을 통해 연속 공간에서 계획할 수 있도록 합니다.



### Self-Supervised Learning for Time Series: A Review & Critique of FITS (https://arxiv.org/abs/2410.18318)
Comments:
arXiv:2307.03756v3 45 pages, 36 figures

- **What's New**: FITS라는 모델이 제안되었으며, 기존의 대규모 모델들과 비슷한 성능을 보여주지만 불과 10k-50k의 파라미터를 사용합니다. 특히 정기적인 패턴과 계절성을 잘 포착할 수 있으나 추세를 가지거나 불규칙한 데이터에 대해서는 한계를 보여줍니다.

- **Technical Details**: FITS는 1층의 신경망을 사용하여 복잡한 주파수 도메인에서 훈련을 수행합니다. DLinear와 결합하여 FITS의 약점을 보완한 두 가지 하이브리드 접근 방식을 통해 다변량 회귀(multivariate regression) 및 가격 데이터셋의 다중/선형 회귀에서 우수한 성과를 달성했습니다.

- **Performance Highlights**: FITS는 TimesNet(300.6M), Pyraformer(241.4M), FEDformer(20.68M), Autoformer(13.61M), PatchTST(1M) 등 기존의 최신 모델들과 비교하여 뛰어난 성능을 10k-50k 파라미터로 달성했습니다. 특히 다변량 회귀 및 가격 데이터셋에서 뛰어난 성능을 보였습니다.



### Countering Autonomous Cyber Threats (https://arxiv.org/abs/2410.18312)
Comments:
          76 pages, MPhil Thesis

- **What's New**: 본 논문은 Foundation Models (FMs)의 사이버 작전 수행 능력을 평가하며, 다운로드 가능한 모델이 독점 모델과 동등한 수준의 사이버 공격을 수행할 수 있음을 보여주고 있습니다. 또한, AI 기반 공격에 대한 효과적인 방어 메커니즘으로 간접 프롬프트 주입(Indirect Prompt Injection) 방법이 처음으로 제안되었습니다.

- **Technical Details**: 이 연구는 HackTheBox의 기계들을 대상으로 하여 다양한 시나리오에서 다운로드 가능한 최신 모델들이 독점 모델과 유사한 성능을 발휘하는지를 평가하였습니다. 악성 사이버 에이전트의 공격을 저지하기 위해 설계된 간접 프롬프트 주입 전략이 포함됩니다.

- **Performance Highlights**: 최신 다운로드 가능한 모델들은 일반적인 해킹 도구를 사용하여 알려진 취약점에 대한 간단한 사이버 공격을 수행하는 데 있어 독점 모델과 동등한 성능을 보였습니다. 간접 프롬프트 주입은 악성 사이버 에이전트의 목표를 무력화하는 효과적인 방법으로 입증되었습니다.



### Screw Geometry Meets Bandits: Incremental Acquisition of Demonstrations to Generate Manipulation Plans (https://arxiv.org/abs/2410.18275)
Comments:
          8 pages, 6 figures, under review in IEEE Robotics and Automation Letters

- **What's New**: 본 논문에서는 반복적으로 충분한 kinesthetic demonstration(운동 시연)을 얻고 로봇이 특정 작업 공간에서 복잡한 조작(task manipulation)을 성공적으로 수행할 수 있는 확신을 가질 수 있도록 하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 논문에서는 (i) screw geometric representation을 사용해 demonstration으로부터 manipulation plan을 생성하여 demonstration의 sufficiency(충분성)를 측정할 수 있도록 하고, (ii) PAC-learning(Probably Approximately Correct) 기반의 샘플링 전략을 활용해 작업 공간의 하위 영역에서 로봇의 manipulation plan 생성 능력을 평가하며, (iii) 약점 지역에서 추가 demonstration을 요청하는 heuristic(휴리스틱)을 제안합니다.

- **Performance Highlights**: 우리는 pouring(붓기) 및 scooping(퍼내기)라는 두 가지 복잡한 조작 작업에 대한 실험 결과를 제시하였으며, 지정된 작업 영역에서 성공적인 계획을 생성하는 데 있어 단지 몇 가지 예(최소 8개 미만)가 충분하다는 것을 보여줍니다.



### Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models (https://arxiv.org/abs/2410.18252)
Comments:
          code at this https URL

- **What's New**: 이 논문에서는 비동기(asynchronous) 및 오프 정책(off-policy) 강화 학습(RL) 방법을 활용하여 RLHF(인간 피드백을 통한 강화 학습)의 효율성을 높이는 방안을 제안합니다. 이를 통해 더 빠른 학습을 가능케 하고, 성능 저하를 최소화하는 방법을 모색합니다.

- **Technical Details**: RLHF의 전통적인 접근법은 온라인(on-policy) 방식으로, 모델이 생성한 출력을 사용해 피드백을 받고 이를 통해 학습하는 것입니다. 연구에서는 비동기 방식으로 새로운 샘플을 생성하고, 이전의 샘플로부터 학습하는 오프 정책 강화 학습 방법을 도입하여 시간 효율성을 개선합니다.

- **Performance Highlights**: LLaMA 3.1 8B 모델을 사용한 실험에서는 비동기 RLHF 방식이 동기화(synchronous) 방법보다 약 40% 더 빠르게 학습하며, 동일한 성능을 유지하는 것을 확인하였습니다. 이 논문에서 제안된 온라인 DPO 방식은 오프 정책 데이터에 대한 강건성이 가장 높으며, 정책 모델의 크기가 커질수록 더 좋은 성능을 보여주었습니다.



### Context-Augmented Code Generation Using Programming Knowledge Graphs (https://arxiv.org/abs/2410.18251)
Comments:
          20 pages, Conference

- **What's New**: 이 논문은 Programming Knowledge Graph (PKG)를 활용하여 코드 검색 및 생성을 위한 새로운 프레임워크를 제안합니다. PKG는 외부 지식을 효과적으로 검색하고 통합하는 데 필수적인 역할을 합니다.

- **Technical Details**: 제안된 방법은 코드 조각의 관련성을 심층적으로 분석하여 검색 정밀도를 높이는 tree-pruning 기법을 포함합니다. 또한, re-ranking 메커니즘을 통해 비 관련성 솔루션을 선택적으로 통합하여 생성 중 발생할 수 있는 hallucination을 줄이는 데 기여합니다.

- **Performance Highlights**: HumanEval 및 MBPP 벤치마크에서 검증 결과, 제안된 방법은 pass@1 정확도를 최대 20%까지 향상시키고 MBPP에서 기존의 최첨단 모델보다 최대 34% 우수한 성능을 보였습니다.



### Efficient Inference for Augmented Large Language Models (https://arxiv.org/abs/2410.18248)
- **What's New**: 이번 논문에서는 API 호출을 통해 외부 데이터 소스를 통합하여 단독 LLM의 기능을 향상시키는 새로운 LLM 추론 프레임워크인 LAMPS를 제안합니다. LAMPS는 요청 처리 시간을 최소화하는 통합 스케줄링 접근 방식을 채택하여 요청의 총 길이와 API 호출 중 처리 전략을 고려합니다.

- **Technical Details**: LAMPS는 요청 처리 시 메모리 소비를 예측하기 위해 opt-125m 언어 모델을 활용하여 입력 프롬프트로부터 API 전 출력 길이를 추정하고, 호출하는 API의 유형에 따라 API 지속 시간을 예측합니다. 요청에는 Preserve, Discard & Recompute, Swap이라는 세 가지 메모리 처리 전략이 있으며, 각 요청에 대해 최적의 전략을 할당합니다.

- **Performance Highlights**: LAMPS는 기존의 LLM 추론 시스템에 비해 종단 간 대기 시간(End-to-End Latency)을 27%에서 85%까지, TTFT(Throughput Time for Finish)를 4%에서 96%까지 감소시키며, vLLM에 대한 개선 효과가 더욱 두드러지는 성과를 냈습니다.



### Characterising Open Source Co-opetition in Company-hosted Open Source Software Projects: The Cases of PyTorch, TensorFlow, and Transformers (https://arxiv.org/abs/2410.18241)
Comments:
          26 pages, 2 figures, 9 tables

- **What's New**: 이번 연구는 Open Source Software (OSS) 프로젝트가 특정 회사에 의해 호스팅되고 관리되는 경우의 오픈 소스 협력 경쟁(open source co-opetition)에 대해 심층적으로 분석하였습니다. 특히 AI 분야의 세 가지 회사 호스팅 OSS 프로젝트인 Meta의 PyTorch, Google's TensorFlow, 그리고 Hugging Face의 Transformers를 다룹니다.

- **Technical Details**: Mixed-methods 분석을 통해 해당 OSS 프로젝트의 코드 저자 패턴, 협력 형태, 그리고 호스트와 외부 회사 간의 관계 유형을 조사하였습니다. 이를 통해 호스트와 외부 회사 간의 전략적, 비전략적, 계약적 협력이 어떻게 다르게 나타나는지를 분석했습니다. 또한 각각의 프로젝트에 대해 social network analysis (SNA)를 수행하였습니다.

- **Performance Highlights**: 결과적으로, 프로젝트들은 유사한 코드 저자 패턴을 보였지만 협력 구조는 다르게 나타났고, 단일 제공자 거버넌스 모델이 오픈 소스 협력 경쟁의 관행과 가능성에 미치는 영향도 식별하였습니다. 이러한 연구 결과는 AI 산업의 기술 및 경쟁 역학에 특화된 협력 형태를 이해하는 데 기여합니다.



### E2E-Swin-Unet++: An Enhanced End-to-End Swin-Unet Architecture With Dual Decoders For PTMC Segmentation (https://arxiv.org/abs/2410.18239)
- **What's New**: 이번 연구에서는 효율적인 유두암 미세암종 (PTMC) 관리를 위해 개발된 AI 기반의 세분화 모델 E2E-Swin-Unet++를 소개합니다. 이 모델은 초음파 B-모드 이미징을 개선하여 PTMC 종양을 실시간으로 식별하고 세분화할 수 있게 합니다.

- **Technical Details**: E2E-Swin-Unet++는 Swin-Unet 아키텍처의 향상된 엔드투엔드 확장으로, 갑상선 영역 정보를 통합하여 PTMC 세분화의 잘못된 가능성을 줄이면서 신속한 추론 기능을 제공합니다. 전통적인 초음파 B-모드 이미징 기술의 한계를 극복하기 위해 설계되었습니다.

- **Performance Highlights**: 실제 임상 RFA 데이터셋에 대한 실험 결과, E2E-Swin-Unet++는 관련 모델들과 비교하여 우수한 성능을 보여주었으며, 이러한 솔루션은 RFA 절제 치료의 정밀도 및 제어력을 크게 향상시킵니다.



### Bayesian optimization for robust robotic grasping using a sensorized compliant hand (https://arxiv.org/abs/2410.18237)
- **What's New**: 이 연구는 로봇 그리핑(Grasping)에 Bayesian Optimization (BO)을 적용하여 다양한 물체에 대한 안전하고 안정적인 그리프를 수행할 수 있는 새로운 방안을 제시합니다.

- **Technical Details**: Bayesian Optimization 기법을 사용하여 프로그래밍된 순서에 의존하지 않고, 촉각 센서(tactile sensors)를 활용하여 3D 공간에서 그리핑을 탐색하는 시스템을 개발하였습니다. 이 방법은 이전 시도의 지식을 활용하여 다양한 물체에 대한 안전한 그리프를 확보할 수 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 BO 접근 방식이 현실 세계의 노이즈와 불확실성 속에서도 미지의 물체에 대한 그리핑을 성공적으로 수행할 수 있다는 점이 입증되었습니다.



### Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks (https://arxiv.org/abs/2410.18210)
Comments:
          14 pages, 6 figures, 7 tables

- **What's New**: 최근 다국어 대형 언어 모델(LLMs)에서 안전성 정렬(safety alignment) 공격이 언어 간 일반화(cross-lingual generalization)가 가능하다는 점이 확인되었습니다. 오직 하나의 언어로부터의 소수의 악성 지침-following 예를 사용해도 다국어 LLM이 손상될 수 있습니다.

- **Technical Details**: 이 연구에서는 Safety Information Localization (SIL) 방법을 통해 다국어 LLM 내의 안전 관련 정보를 국소화하고, 안전 관련 파라미터의 20%만 수정하면 모든 언어에서 안전성 정렬을 파괴할 수 있음을 밝혔습니다. 이는 악성 fine-tuning 공격을 통해 확인되었습니다.

- **Performance Highlights**: LLMs에서 단 100개의 악성 지침-following 예를 사용한 몇 단계의 fine-tuning으로, 다국어 LLM들의 안전성이 손상된 것으로 나타났습니다. 이 연구는 다국어 LLM의 안전성 문제를 해결하기 위한 중요한 통찰력을 제공합니다.



### PyTSC: A Unified Platform for Multi-Agent Reinforcement Learning in Traffic Signal Contro (https://arxiv.org/abs/2410.18202)
Comments:
          13 pages

- **What's New**: 이 논문은 도시 환경에서의 신호 제어(traffic signal control, TSC)를 위한 Multi-Agent Reinforcement Learning(MARL)의 연구를 위해 설계된 새로운 시뮬레이션 환경인 PyTSC를 소개합니다. PyTSC는 여러 시뮬레이터(SUMO, CityFlow 등)를 통합하여 MARL 알고리즘의 훈련 및 평가를 용이하게 하고, 연구자들이 효율적으로 다양한 MARL 접근 방식을 탐색하고 실험할 수 있도록 돕습니다.

- **Technical Details**: PyTSC는 모듈화된 아키텍처와 통일된 API를 제공하여, 사용자들이 다양한 시뮬레이터를 쉽게 통합할 수 있으며, Centralized Training Decentralized Execution(CTDE) 프레임워크와 호환됩니다. 이를 통해 PyTSC는 실시간 교통 관리 문제를 해결하기 위한 연구의 가속화를 목표로 하며, 기존 시뮬레이터의 툴과 통합을 지원합니다.

- **Performance Highlights**: PyTSC는 다양한 MARL 기술을 실험하기 위한 10개의 오픈 소스 시나리오를 제공하며, 이 시나리오는 합성 및 실제 교통 네트워크를 포함하고 CityFlow와 SUMO 시뮬레이터와 호환됩니다. PyTSC는 실험 속도를 개선할 수 있는 모듈과 데이터 세트를 집계하는 기능을 제공하여, 연구자들이 빠르게 실험하고 보다 효과적인 TSC 전략을 개발할 수 있도록 지원합니다.



### ZIP-FIT: Embedding-Free Data Selection via Compression-Based Alignmen (https://arxiv.org/abs/2410.18194)
- **What's New**: 이 논문에서는 ZIP-FIT이라는 데이터 선택 프레임워크를 소개하며, 이는 gzip 압축을 활용하여 잠재적인 훈련 데이터와 목표 작업 분포 간의 정렬을 직접 측정합니다.

- **Technical Details**: ZIP-FIT은 LZ77과 Huffman 코딩이라는 두 가지 주요 압축 기법을 사용하여 데이터 내에서 반복 패턴을 이용하여 압축을 수행하며, 수학적 언어 표기법인 Autoformalization과 Python 코드 생성 과제를 통해 평가되었습니다.

- **Performance Highlights**: ZIP-FIT은 기존의 데이터 선택 방식들보다 일관되게 모델 성능을 향상시키며, DSIR 및 D4와 같은 주요 기준선을 초과하여 85.1% 더 빠른 수렴 및 낮은 크로스 엔트로피 손실을 달성했습니다.



### TabDPT: Scaling Tabular Foundation Models (https://arxiv.org/abs/2410.18164)
Comments:
          Minimal TabDPT interface to provide predictions on new datasets available at the following link: this https URL

- **What's New**: 본 논문에서는 Tabular Discriminative Pre-trained Transformer (TabDPT)라는 새로운 모델을 제안하며, 이를 통해 기존의 tabular 데이터에서의 성능 문제를 해결합니다. TabDPT는 기존의 tree-based 모델이나 대형 언어 모델의 한계를 극복하고, unseen 데이터에서 고속으로 예측을 수행할 수 있습니다.

- **Technical Details**: TabDPT는 self-supervised learning과 retrieval-based 방법을 결합하여, 실제 데이터를 기반으로 한 사전 훈련을 통해 tabular 데이터에 최적화된 ICL(인 컨텍스트 학습) 아키텍처를 사용합니다. 이 모델은 분류(classification) 및 회귀(regression) 문제 모두에 대한 작업에서 추가적인 훈련이나 하이퍼파라미터 튜닝 없이 높은 성능을 발휘합니다.

- **Performance Highlights**: TabDPT는 OpenML-CC18 및 OpenML-CTR23 벤치마크에서 state-of-the-art 성능을 달성하였으며, 훈련된 모델을 기반으로 빠른 추론 속도를 자랑합니다. 모델 크기와 훈련 데이터의 양이 증가함에 따라 성능이 향상되는 경향을 보이며, 이는 향후 더 큰 데이터셋의 수집 및 더 큰 모델 훈련을 통해 개선될 수 있음을 시사합니다.



### Physics-informed Neural Networks for Functional Differential Equations: Cylindrical Approximation and Its Convergence Guarantees (https://arxiv.org/abs/2410.18153)
Comments:
          Accepted at NeurIPS 2024. Both authors contributed equally. Some contents are omitted due to arXiv's storage limit. Please refer to the full paper at OpenReview (NeurIPS 2024) or this https URL

- **What's New**: 본 논문에서는 기능적 미분 방정식(Functional Differential Equations, FDEs)을 위한 최초의 학습 기법을 제안합니다. FDEs는 물리학, 수학 및 최적 제어에서 중요한 역할을 하지만, 수치해석이 현실적인 계산 비용으로 인해 어려움을 겪고 있습니다. 따라서 우리는 물리 정보 신경망(Physics-informed Neural Networks, PINNs)와 원통 근사(cylindrical approximation)를 결합한 하이브리드 접근법을 통해 이러한 문제를 해결하고자 했습니다.

- **Technical Details**: 제안하는 방법에서는 먼저 입력 함수를 직교 정규 기저 함수(orthonormal basis functions)로 확장하여 주어진 FDE를 확대된 차원 PDE로 변환합니다. 그런 다음, PINNs를 이용하여 고차원 PDE를 수치적으로 해결하는 방식을 취합니다. 이 과정에서 원통 근사의 수렴 정리를 증명하여 FDE 응용에 대한 신뢰성을 보장합니다.

- **Performance Highlights**: 실험 결과, 우리는 Burgers-Hopf 방정식(Burgers-Hopf equation)과 같은 두 개의 FDE에서 𝑂(𝑚𝑟) 복잡도로 정확도를 유지하며, 일반적인 L^1 상대 오차를 𝑃𝑖𝑛𝑛𝑠∼10^{-3} 수준으로 달성할 수 있음을 입증했습니다. 이는 기존의 알고리즘에 비해 상당한 성능 향상을 나타내며, 기능적 미분 방정식의 수치 해석이 더 많은 이에게 접근 가능하게 됩니다.



### Deep Autoencoder with SVD-Like Convergence and Flat Minima (https://arxiv.org/abs/2410.18148)
Comments:
          14 pages

- **What's New**: 이 논문에서는 고차원 복잡 물리 시스템을 위한 새로운 표현 학습 기법으로 learnable weighted hybrid autoencoder를 제안합니다. 이는 singular value decomposition (SVD)과 deep autoencoder의 장점을 결합하여 학습 가능한 가중치 프레임워크를 통해 improved convergence behavior를 제공합니다.

- **Technical Details**: 제안된 접근법은 POD(Proper Orthogonal Decomposition) 기반 인코더와 신경망 인코더를 결합하여, 학습 가능한 가중치 벡터를 사용하여 latent state를 구성합니다. 이를 통해 높은 차수에서도 안정적이고 효율적인 차원 축소를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 고전적인 혼돈 PDE 시스템에 대해 sharpness가 수천 배 작으며, 여러 경쟁 방법들에 비해 일반화 성능이 유의미하게 향상되는 것을 보여줍니다.



### MEC-IP: Efficient Discovery of Markov Equivalent Classes via Integer Programming (https://arxiv.org/abs/2410.18147)
- **What's New**: 본 논문은 관측 데이터를 이용해 Bayesian Networks (BNs)의 Markov Equivalent Class (MEC)를 발견하기 위한 새로운 Integer Programming (IP) 접근법인 MEC-IP 알고리즘을 제안합니다. 이 알고리즘은 고유한 clique-focusing 전략과 Extended Maximal Spanning Graphs (EMSG)를 활용하여 MEC 탐색을 효율화하며, 기존 알고리즘의 계산적 한계를 극복합니다.

- **Technical Details**: MEC-IP 알고리즘은 데이터 주도적인 targeted-clique-focusing 접근 방식을 채택하며, Exhaustive Search가 필요 없는 반복적인 네트워크 정제를 위한 Extended Maximal Spanning Graphs (EMSG)를 사용합니다. 이를 통해, 대규모 데이터셋에서도 시간이 절약되고 계산 자원이 효율적으로 활용되며, 조건부 독립성을 판단하는 데 필요한 통계적 테스트의 수를 줄여냅니다.

- **Performance Highlights**: MEC-IP 알고리즘은 계산 시간의 현저한 감소를 이루었으며, 다양한 데이터셋에서 인과 발견의 정확성 또한 개선되었습니다. 이는 연구자와 실무자에게 복잡한 데이터 구조의 효율적이고 정확한 분석을 위한 강력한 도구가 될 잠재력을 강조합니다.



### Meaning Typed Prompting: A Technique for Efficient, Reliable Structured Output Generation (https://arxiv.org/abs/2410.18146)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 고급 응용을 위해 구조화된 출력 생성을 신뢰할 수 있는 방법으로 개선한 새로운 기법인 의미 유형 프롬프팅(Meaning Typed Prompting, MTP)을 소개합니다.

- **Technical Details**: MTP는 변수(variable) 및 클래스(class)와 같은 유형(types), 의미(means), 추상화(abstraction)를 프롬프팅 과정에 통합하여 효율적인 구조화된 출력 생성을 가능하게 합니다. 표현력이 풍부한 유형 정의(type definitions)를 통해 MTP는 출력의 명확성을 강화하고 복잡한 추상화에 대한 의존도를 줄이며, 개발을 단순화하고 구현 효율성을 높입니다.

- **Performance Highlights**: 다양한 벤치마크에서의 실증 분석을 통해 MTP는 기존 프레임워크에 비해 정확성(accuracy), 신뢰성(reliability), 일관성(consistency), 토큰 효율성(token efficiency)에서 우수한 성과를 보였습니다. 또한, MTP를 구현한 Semantix라는 프레임워크를 제시하여 실제 응용에 대한 통찰력을 제공합니다.



### Analyzing Nobel Prize Literature with Large Language Models (https://arxiv.org/abs/2410.18142)
- **What's New**: 이번 연구는 문학 분석의 맥락에서 고급 대형 언어 모델(LLMs), 특히 o1 모델의 능력을 검토합니다. 연구는 한강의 '아홉 개의 장'과 욘 포세의 '우정'이라는 두 개의 노벨 문학상을 수상한 단편 소설에 대해 인간 대학원생의 결과와 직접 비교합니다. 이 연구는 AI의 복잡한 문학 요소에의 관여 정도를 탐구하며, LLM이 문학적 해석에서 인간을 어떻게 보완하거나 도전할 수 있는지를 조명합니다.

- **Technical Details**: 연구는 LLM인 o1 모델이 주제를 분석하고, 상호텍스트성(intertextuality), 문화 및 역사적 맥락, 언어적 및 구조적 혁신, 캐릭터 개발 등 여러 복잡한 문학 차원에서 어떻게 기능하는지를 평가합니다. LLM과 인간 참가자의 출력을 비교함으로써 이 연구는 AI의 문학적 해석에서의 강점과 한계를 드러내고, 인간 지성의 복제를 향한 과제를 탐구합니다.

- **Performance Highlights**: o1 모델은 구조적 작업에서 강력한 분석 능력을 보여주었지만, 감정의 뉘앙스와 일관성 측면에서 인간 해석에 비해 부족함을 드러냈습니다. 이는 인문학 분야에서 인간과 AI의 협력 가능성을 강조하며, 문학 연구 및 기타 분야에서 새로운 기회를 열어줍니다.



### SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback (https://arxiv.org/abs/2410.18141)
- **What's New**: 이 논문에서는 RAG 시스템의 모든 모듈이 개별적으로 최적화되는 대신 공동 최적화되어야 한다고 주장합니다. 이를 위해 'SmartRAG'라는 새로운 파이프라인을 설계하고, 강화 학습(rl)을 통해 시스템을 공동 최적화합니다.

- **Technical Details**: SmartRAG는 정책 네트워크(policy network)와 리트리버(retriever)를 포함하며, 정책 네트워크는 1) 언제 검색할지 결정, 2) 검색기와 가장 적합한 쿼리 생성, 3) 최종 응답 생성의 세 가지 역할을 합니다. 시스템의 최적화는 환경 피드백을 기반으로 이루어집니다.

- **Performance Highlights**: SmartRAG는 다양한 데이터셋에서 실험 결과, 개별적으로 최적화된 모듈보다 우수한 성능을 보였으며, retrieval 비용을 최소화하면서도 정확한 응답을 생성하는 데 성공했습니다.



### R2Gen-Mamba: A Selective State Space Model for Radiology Report Generation (https://arxiv.org/abs/2410.18135)
Comments:
          4 pages pages for ISBI2025

- **What's New**: 본 연구는 Mamba의 효율적인 시퀀스 처리와 Transformer 구조의 맥락적 이점을 활용한 새로운 자동 방사선 보고서 생성 방법인 R2Gen-Mamba를 제안합니다.

- **Technical Details**: R2Gen-Mamba는 Mamba 모델을 인코더로, Transformer를 디코더로 활용하며, 이는 낮은 계산 복잡도로 인해 학습 및 추론 효율성을 향상시킵니다. 이미지 패치의 특징을 입력으로 사용하고, 이에 대한 보고서를 생성하는 시퀀스-시퀀스 접근 방식을 채택합니다. Mamba의 선형 계산 복잡성 덕분에 높은 품질의 보고서를 생성할 수 있습니다.

- **Performance Highlights**: R2Gen-Mamba는 210,000개 이상의 X-ray 이미지-보고서 쌍을 포함하는 두 개의 벤치마크 데이터셋에서 전통적인 Transformer 기반 모델보다 높은 보고서 품질과 계산 효율성을 보였습니다. 이는 최신 기술(SOTA)들과 비교했을 때 리소스 효율적인 솔루션을 제공함을 나타냅니다.



### Optimizing Preference Alignment with Differentiable NDCG Ranking (https://arxiv.org/abs/2410.18127)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 인간 선호와의 정렬을 새로운 방식으로 접근하여, Ranking 데이터 기반의 최적화 방법인 Direct Ranking Preference Optimization (DRPO)를 제안합니다. 이는 현재의 방법들(DPO)의 한계를 극복하고, NDCG와 같은 측정 지표를 직접 최적화합니다.

- **Technical Details**: DRPO는 Learning-to-Rank (LTR) 접근법을 통해 인간 선호 데이터를 활용하여 응답 목록의 순위를 최적화합니다. 이 과정에서 diffNDCG라는 차별 가능한 손실 함수를 도입하고, 이는 정렬된 목록에서의 응답 간 분류 정확도를 높여줍니다. 적응형 순위 정책 점수(Adaptive Rank Policy Score)를 통해 응답의 상대적 위치에 따라 점수 마진을 조정하며, 다양한 순위 모델과 함께 사용됩니다.

- **Performance Highlights**: DRPO는 기존의 방법들과 비교하여 생성된 응답의 질에서 우수한 성능을 보이며, 수많은 실험 결과에서 DRPO가 기존의 기준 선형 모델들을 능가함을 입증했습니다.



### Towards Edge General Intelligence via Large Language Models: Opportunities and Challenges (https://arxiv.org/abs/2410.18125)
- **What's New**: 이 논문은 Edge Intelligence (EI)와 이를 통한 Edge General Intelligence (EGI)의 진화를 다루고 있습니다. LLMs (Large Language Models)의 통합이 EI를 더욱 발전시키는 방향으로 나아가고 있음을 강조합니다.

- **Technical Details**: EGI는 EI와의 차별점을 명확히 하며, LLM을 활용한 EGI를 중앙 집중형(centralized), 하이브리드(hybrid), 분산형(decentralized)으로 구분합니다. 각 시스템에 대한 프레임워크 설계와 기존 구현 사례를 상세히 설명합니다.

- **Performance Highlights**: 또한, Edge 장치에서 개발에 더 적합한 Small Language Models (SLMs)의 성능과 처리량(throughput)을 평가하여, 연구자들에게 EGI의 미래 발전을 위한 기초를 제공합니다.



### Point Cloud Compression with Bits-back Coding (https://arxiv.org/abs/2410.18115)
Comments:
          This paper is under reviewed in IEEE Robotics and Automation Letters

- **What's New**: 이 논문은 bits-back coding 방식을 활용하여 포인트 클라우드 데이터의 기하학적 속성을 손실 없이 압축하는 새로운 방법을 제안합니다. 이 방법은 심층 학습 기반 확률 모델을 사용하여 포인트 클라우드 정보의 Shannon 엔트로피를 추정하는 데 중점을 두고 있습니다.

- **Technical Details**: 포인트 클라우드 압축을 위한 새로운 접근 방식으로 convolutional variational autoencoder (CVAE)를 이용해 Shannon의 엔트로피를 추정합니다. bits-back coding 기법을 통해 CVAE의 학습된 잠재 변수 모델을 활용하여 데이터 포인트 간의 잠재적인 상관관계를 포착하고 압축 비율을 줄입니다.

- **Performance Highlights**: 제안하는 방법은 평균적으로 1.56 bits-per-point의 압축 비율을 달성하며, 구글의 Draco와 같은 기존 접근 방식의 1.83 bits-per-point에 비해 상당히 낮은 비율을 보입니다. 또한, 압축 코덱의 저장 및 통신 오버헤드를 크게 줄이는 데 성공하여 실용적인 응용 분야에 적용 가능성을 높입니다.



### Bridging Today and the Future of Humanity: AI Safety in 2024 and Beyond (https://arxiv.org/abs/2410.18114)
- **What's New**: 이번 논문은 AI 안전에 대한 기존 접근방식의 방향성을 재조명하고, 인류 문명의 진화와 AI의 상호작용에 대해 보다 광범위한 관점을 제시합니다. 특히, 현재의 AI 안전 노력들이 장기적인 필요와 얼마나 일치하는지를 검토하는 데 중점을 두고 있습니다.

- **Technical Details**: AI와 Large Language Models (LLMs)의 빠른 발전에도 불구하고, 현재 AI는 진정한 지능의 기준을 충족하지 못하며 대량의 데이터에 의존합니다. LLM의 추론 능력은 실제로는 훈련 데이터를 기반으로 한 근사적 검색의 형태입니다. 또한, AI의 높은 에너지 소비 문제는 지속 가능성에 대한 심각한 우려를 낳고 있습니다.

- **Performance Highlights**: AI 안전은 단순한 단기적인 문제 해결을 넘어, 포괄적인 품질 보증(AI quality assurance)으로 확장될 수 있으며, 다양한 분야의 전문가들과의 협력이 필수적입니다. 또한, AI 워크플로우의 맞춤화와 사용자-AI 상호작용을 개선하여 사용자 요구에 더욱 적합한 AI 시스템을 만드는 것이 중요합니다.



### NaVIP: An Image-Centric Indoor Navigation Solution for Visually Impaired Peop (https://arxiv.org/abs/2410.18109)
Comments:
          40 pages, 20 figures

- **What's New**: 본 논문에서는 시각장애인(VIPs)들을 위한 실내 내비게이션을 위한 이미지 데이터셋(NaVIP)과 이미지 중심 솔루션을 제안합니다. 이는 인프라 요구 없이 자동으로 환경을 이해할 수 있도록 돕는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: NaVIP 데이터셋은 연구 건물의 4개 층에서 수집된 300K개의 이미지를 포함하며, 정확한 6DoF 카메라 포즈, 실내 포인트 오브 인터레스(Points of Interest, PoIs) 정보 및 설명 캡션으로 라벨링되어 있습니다. 이 데이터는 실시간 추론과 훈련 확장성을 기준으로 벤치마킹되었습니다.

- **Performance Highlights**: NaVIP의 구현을 통해 카메라 포즈를 정확하게 추정하고, 이미지 캡셔닝 기법을 활용하여 VIPs의 탐색을 지원할 수 있음을 보여줍니다. 실제 건물 레이아웃에서 서브 미터 정확도로 위치 지정을 수행할 수 있는 성능을 보였습니다.



### In-Context Code-Text Learning for Bimodal Software Engineering (https://arxiv.org/abs/2410.18107)
- **What's New**: 이 논문은 소프트웨어 공학에서 코드와 텍스트의 이중 모드를 효과적으로 분석하기 위한 새로운 방법론인 InCTRL을 제안합니다. InCTRL은 사전 훈련된 CodeLLAMA 모델을 활용하여, 다양한 소프트웨어 엔지니어링 작업을 위한 통합적인 코드-텍스트 학습을 수행합니다.

- **Technical Details**: InCTRL은 코드와 텍스트 데이터를 단일 형식으로 통합하여 LLM(대형 언어 모델)의 학습 데이터를 확장합니다. 이 방법은 구성 가능한 프롬프트 템플릿을 통해 유용한 특징을 효과적으로 추출하며, 다양한 소프트웨어 엔지니어링 작업을 관통하는 프롬프트 학습을 통합합니다. 또한, Retrieval-Augmented Generation(RAG) 모듈을 통합하여 전통적인 인-컨텍스트 학습의 한계를 해결합니다.

- **Performance Highlights**: InCTRL은 코드 생성 및 텍스트 생성 작업에서 CodeLLAMA보다 17.36%에서 85.3%까지 성능을 향상시키며, 특히 프로그램 수리 작업에서 BLEU 점수를 85점 개선하고 클론 탐지 작업에서 69% 포인트 개선을 달성했습니다. 연구는 InCTRL 모델이 다양한 소프트웨어 엔지니어링 작업에서 최첨단 성능을 가지고 있으며, 특히 코드 생성 및 수리 분야에서 우수한 성과를 보인다고 강조합니다.



### Improving Embedding Accuracy for Document Retrieval Using Entity Relationship Maps and Model-Aware Contrastive Sampling (https://arxiv.org/abs/2410.18105)
Comments:
          10 Pages, 9 Figures

- **What's New**: 이번 논문에서는 APEX-Embedding-7B라는 70억 매개변수의 디코더 전용 텍스트 피처 추출 모델을 소개합니다. 이 모델은 특히 Document Retrieval-Augmented Generation (RAG) 작업을 위해 설계되었습니다. 본 모델은 두 가지 훈련 기술을 사용하여 사실적 초점을 향상시키고 이러한 방식으로 RAG 시스템의 성능을 개선합니다.

- **Technical Details**: APEX-Embedding-7B은 (1) Structured Entity Relationship Maps를 이용한 사전 수렴 중단 세부 조정(pre-convergence interrupted fine-tuning)과 (2) Model-Aware Contrastive Sampling을 사용하는데, 이는 모델의 주의 편향을 사실적 콘텐츠로 전환하여 성능을 향상시킵니다. 이 메소드는 특히 부동산과 같이 방대한 데이터셋을 다루는 산업에 적합합니다.

- **Performance Highlights**: 본 모델은 평가에서 plain text 쿼리/document 쌍의 검색 정확도를 90.86%로 향상시키며(다음 우수 모델에 비해 6.26% 증가) 훈련 데이터 입력의 컨텍스트 크기를 평균 37.71% 감소시킵니다. APEX-Embedding-7B은 긴 문맥의 문서 검색 작업에서 새로운 최첨단 기준을 수립합니다.



### ENWAR: A RAG-empowered Multi-Modal LLM Framework for Wireless Environment Perception (https://arxiv.org/abs/2410.18104)
- **What's New**: 본 논문에서는 6G 및 그 이후의 네트워크에서 네트워크 관리 및 조정을 향상시키기 위해 개발된 새로운 ENWAR 프레임워크를 소개합니다. ENWAR은 다중 모달 감각 데이터를 통합하여 복잡한 무선 환경을 인지하고 해석하는 능력을 갖추고 있습니다.

- **Technical Details**: ENWAR는 GPS, LiDAR, 카메라 데이터 등을 사용하여 다중 모달 센서 입력을 처리합니다. 이 프레임워크는 자연어 처리를 위한 LLM에 텍스트 기반 입력을 통합하고, RAG( retrieval-augmented generation) 기술을 활용하여 정확하고 맥락에 맞는 응답을 생성합니다.

- **Performance Highlights**: ENWAR는 70%의 관련성, 55%의 맥락 회상, 80%의 정확성 및 86%의 신뢰성이 발견되었으며, 이는 향후 무선 시스템에서 다중 모달 인식 및 해석 능력을 더욱 향상시킬 잠재력을 보여줍니다.



### A Hybrid Graph Neural Network for Enhanced EEG-Based Depression Detection (https://arxiv.org/abs/2410.18103)
- **What's New**: 이번 연구에서는 EEG 기반 우울증 탐지를 위한 새로운 하이브리드 그래프 신경망 모델인 Hybrid GNN (HGNN)을 제안합니다. 기존 GNN 기반 방법들이 우울증의 특성을 충분히 고려하지 못했던 문제를 해결하고자 합니다.

- **Technical Details**: HGNN은 고정된 연결을 사용하는 Common Graph Neural Network (CGNN)와 적응형 연결을 사용하는 Individualized Graph Neural Network (IGNN)을 결합하여 우울증의 공통 및 개인화된 패턴을 각각 포착합니다. 또한, IGNN 가지에는 개인화된 계층 정보를 추출하기 위한 Graph Pooling and Unpooling Module (GPUM)이 추가되었습니다.

- **Performance Highlights**: 두 개의 공개 데이터셋에 대한 광범위한 실험 결과, 제안한 모델은 최신 성능(state-of-the-art performance)을 달성했습니다.



### Multiple Global Peaks Big Bang-Big Crunch Algorithm for Multimodal Optimization (https://arxiv.org/abs/2410.18102)
Comments:
          23 pages

- **What's New**: 본 연구에서 제안된 'Multiple Global Peaks Big Bang-Big Crunch' 알고리즘(MGP-BBBC)은 다중 최적해가 존재하는 최적화 문제의 높은 정확도를 요구하는 다차원 탐색 공간에서 여러 개의 정점을 식별하는 데 중점을 두었습니다. 이 알고리즘은 진화론적 원리를 바탕으로 집단을 클러스터링하고 증가하는 방식으로 진동을 줄여 수렴을 보장합니다.

- **Technical Details**: MGP-BBBC 알고리즘은 비모수 클러스터링 방법을 사용하여 최적해의 수를 자동으로 결정하며, 'Big Bang-Big Crunch' 과정을 거쳐 개체들의 질량 중심을 유지하면서 정점을 찾습니다. 이 알고리즘은 생존 단계, 빅 크런치 연산자 및 빅 뱅 연산자를 통해 탐색과 활용의 균형을 맞추며, 개체의 적합성을 기준으로 부분 집합을 조정합니다.

- **Performance Highlights**: 실험 결과, MGP-BBBC는 20개의 다중 최적화 기준 테스트 함수에서 13개의 최신기법과 비교했을 때 10개의 알고리즘보다 우수한 성능을 나타내었으며, 나머지 3개의 알고리즘과도 경쟁력이 있음을 입증했습니다.



### Molecular Dynamics and Machine Learning Unlock Possibilities in Beauty Design -- A Perspectiv (https://arxiv.org/abs/2410.18101)
- **What's New**: 본 논문에서는 머신러닝과 분자 동역학(molecular dynamics) 접근법을 활용하여 스킨케어 제품의 혁신적인 디자인을 위한 연구를 다룹니다. 특히, 데이터가 부족한 상황에서 물리 기반 접근법을 통해 생리적 기능을 가진 단백질과의 상호작용을 모델링하며, 풍부한 데이터가 수집된 경우에는 QSAR(Quantitative Structure-Activity Relationship) 모델을 통해 실험 설계의 다음 단계를 안내할 수 있습니다.

- **Technical Details**: 컴퓨터 지원 분자 발견(Computer-Aided Molecular Discovery) 기술과 머신러닝의 새로운 애플리케이션을 통해 화장품 및 스킨케어 제품의 디자인에 대한 목표와 도전 과제를 검토합니다. 사용되는 기술에는 MD(분자 동역학) 시뮬레이션과 GNN(Graph Neural Network), 변분 추론(Variational Inference) 등이 포함되며, 이들은 분자 구조와 성능 예측을 위한 핵심 도구로 자리잡고 있습니다.

- **Performance Highlights**: 연구 결과, 머신러닝을 통해 분자 모델링의 효율성과 정확성을 대폭 향상시킬 수 있으며, 이는 새로운 스킨케어 제품의 디자인에 즉각적으로 활용될 수 있습니다. 또한, 다양한 사용자 요구를 충족시키기 위한 다학제적 연구 프로젝트도 제안되었습니다.



### RingGesture: A Ring-Based Mid-Air Gesture Typing System Powered by a Deep-Learning Word Prediction Framework (https://arxiv.org/abs/2410.18100)
- **What's New**: 이 논문은 경량 증강 현실(AR) 안경을 위한 새로운 텍스트 입력 시스템인 RingGesture를 제안합니다. 이는 제스처 경로의 시작과 끝을 표시하는 전극과 손 추적을 위한 관성 측정 장치(IMU) 센서를 활용하여 공중 제스처 타이핑을 가능하게 합니다.

- **Technical Details**: RingGesture 시스템은 심층 학습 기반의 새로운 단어 예측 프레임워크인 Score Fusion을 통합하여 정확성과 입력 속도를 향상시킵니다. Score Fusion은 단어-제스처 디코딩 모델, 공간 철자 수정 모델, 경량 컨텍스트 언어 모델 등 세 가지 주요 구성 요소로 구성됩니다.

- **Performance Highlights**: RingGesture는 평균 27.3 단어/분(WPM)의 텍스트 입력 속도를 기록했으며, 최고 성능은 47.9 WPM입니다. Score Fusion 프레임워크는 전통적인 단어 예측 프레임워크인 Naive Correction보다 28.2% 더 나은 성능을 보여주었으며, RingGesture의 텍스트 입력 속도는 55.2% 향상되었습니다. 또한, 시스템 사용성 점수는 83점으로, 매우 우수한 사용성을 나타냅니다.



### Gesture2Text: A Generalizable Decoder for Word-Gesture Keyboards in XR Through Trajectory Coarse Discretization and Pre-training (https://arxiv.org/abs/2410.18099)
- **What's New**: 이 논문에서는 확장 현실(Extended Reality, XR) 환경에서의 직관적인 텍스트 입력을 위한 단어-제스처 키보드(Word-Gesture Keyboard, WGK) 시스템에 대한 새로운 신경 디코더(Neural Decoder)를 제안합니다. 새로운 접근법은 대규모의 간략히 이산화된(word-gesture trajectories) 데이터를 미리 학습하여 다양한 환경에서 일반화할 수 있는 성능을 보여줍니다.

- **Technical Details**: 합성곱 신경망(convolutional neural networks)과 사전 학습(pre-training) 기법을 통해 기존의 템플릿 일치(template-matching) 디코더인 SHARK^2의 한계를 극복하려고 하며, 입력 경로(trajectory)를 인코딩하는 구조적 표현 E(g)를 사용합니다. 이 시스템은 현실 증강(AR)과 가상 현실(VR)에서의 WGK 시스템에 적용 가능하며, 단순한 설치와 함께 높은 디코딩 정확도를 제공합니다.

- **Performance Highlights**: 이 새로운 사전 학습된 신경 디코더는 평균적으로 90.4%의 Top-4 정확도를 달성하였고, SHARK^2보다 37.2% 높은 성능을 보여주며, 전통적인 신경 디코더보다도 7.4% 향상된 결과를 보입니다. 또한, 이 디코더는 저용량(4 MB)으로 실시간(97ms) 처리 속도를 가지며 정확도를 희생하지 않았습니다.



### RRADistill: Distilling LLMs' Passage Ranking Ability for Document Re-Ranking of Long-Tail Queries in a Search Engin (https://arxiv.org/abs/2410.18097)
Comments:
          Accepted to EMNLP 2024 Industry Track. First two authors contributed equally

- **What's New**: 이 연구는 RRADistill: Re-Ranking Ability Distillation이라는 새로운 방법론을 제안하며, 문서 및 쿼리 간의 의미적 관계를 이해하는 대형언어모델(LLMs)의 효과성을 활용하는 동시에, 보다 작고 효율적인 모델(sLLMs)의 실용적인 훈련 기법을 개발합니다.

- **Technical Details**: 우리의 접근 방식은 크게 두 가지 단계로 구성됩니다. 첫째, LLM을 사용하여 랭킹 레이블을 생성하는 단계이며, 둘째, 이 레이블을 통해 SLM 랭커를 훈련하는 것입니다. 우리는 Pre-rank 기법을 도입하여 효율적인 레이블 생성 파이프라인을 만들었으며, Term Control Layer와 같은 새로운 기술을 통해 쿼리와 문서 간의 용어 매칭 신호를 효과적으로 활용합니다.

- **Performance Highlights**: 우리의 방법론은 한국 기반 검색 플랫폼에서 긴 쿼리에 대해 A/B 테스트를 통해 효과성을 입증했습니다. 실험 결과, RRA-BERT와 RRA-GPT 두 가지 랭커 모두 LLM의 관계적 랭킹을 흉내내도록 훈련되어, 긴 꼬리 쿼리 검색 품질을 획기적으로 개선하였습니다.



### $M^3EL$: A Multi-task Multi-topic Dataset for Multi-modal Entity Linking (https://arxiv.org/abs/2410.18096)
- **What's New**: 본 논문에서는 Multi-modal Entity Linking (MEL) 를 위한 대규모 데이터셋 M^3EL을 제안한다. 이 데이터셋은 79,625개의 인스턴스를 포함하고 있으며, 5개의 다양한 주제와 9가지 다중 모달(Multi-modal) 작업을 포괄하고 있다. 기존 MEL 데이터셋의 한계를 극복하기 위해 데이터셋 구축 파이프라인을 수립하였다.

- **Technical Details**: M^3EL 데이터셋은 318,500개의 이미지를 포함하고 있으며, Text-Text, Text-Image, Image-Text 등 다양한 모달 작업을 지원한다. CLIP 모델을 활용하여 MODE(모달-증강) 훈련 전략을 제안하며, M^3EL 데이터셋을 기반으로 CLIP_{ND} 모델을 미세 조정(fine-tune)하였다.

- **Performance Highlights**: 실험 결과 기존 모델의 정확도가 49.4%에서 75.8%까지 다양하게 나타난 반면, M^3EL 데이터셋으로 학습한 CLIP_{ND} 모델은 다양한 작업에서 평균 9.3%에서 25%까지 성능이 향상되었다. 이 데이터셋은 MEL 알고리즘의 일반화 성능을 효과적으로 향상시킬 수 있는 우수한 품질의 사전 훈련 데이터셋으로 자리잡는다.



### Ethical Leadership in the Age of AI Challenges, Opportunities and Framework for Ethical Leadership (https://arxiv.org/abs/2410.18095)
Comments:
          15 pages, submitted to SAGE for review

- **What's New**: 이 연구는 인공지능(AI) 시대에 윤리적 리더십의 중요성을 탐구하며, AI 통합에서 마주하는 여러 윤리적 도전과 기회를 분석합니다.

- **Technical Details**: 윤리적 리더십은 공정함, 투명성, 지속 가능성을 포함한 핵심 요소로 구성된 프레임워크를 통해 AI가 직면한 윤리적 문제를 해결하는 접근 방식을 제안합니다. 머신러닝, 자연어 처리, 로봇공학과 같은 AI 기술이 조직과 비즈니스에 미치는 영향을 법적 규제를 넘어 탐색합니다.

- **Performance Highlights**: 윤리적 리더십이 조직의 문화, 직원의 도덕성, 신뢰성 및 대외적 평판에 미치는 긍정적인 영향을 강조하며, AI의 윤리적 과제를 해결함으로써 장기적으로 조직이 지속 가능한 성과를 달성할 수 있음을 시사합니다.



### Self-supervised inter-intra period-aware ECG representation learning for detecting atrial fibrillation (https://arxiv.org/abs/2410.18094)
Comments:
          Preprint submitted to Biomedical Signal Processing and Control

- **What's New**: 이 논문에서는 심방세동(atrial fibrillation) 관련 ECG(전자심전도) 신호의 대표성을 향상시키기 위해 inter-intra period-aware ECG representation learning 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 RR 간격의 불규칙성과 P파의 부재를 고찰하여, 심방세동 환자의 ECG 신호에서 특정 pre-training task를 개발합니다. 이는 단일 주기 안정적인 형태학적 표현을 학습하면서 중요한 인터주기 특징을 유지하는 것을 목표로 합니다.

- **Performance Highlights**: 본 방법은 BTCH 데이터셋에서 0.953/0.996의 AUC(Area Under the ROC Curve) 성능을 달성하였으며, CinC2017 및 CPSC2021과 같은 일반적으로 사용되는 벤치마크에서도 경쟁력 있는 결과를 보여줍니다.



### Two-Stage Radio Map Construction with Real Environments and Sparse Measurements (https://arxiv.org/abs/2410.18092)
- **What's New**: 이번 논문에서는 기존의 radio map 구축 방식의 단점을 보완하기 위해, Generative Adversarial Networks (GANs)를 활용한 새로운 FPTC(First-Predict-Then-Correct) 방법론을 제안합니다. 이 방법은 환경 정보를 기반으로 초기 radio map을 생성한 후, 불완전한 측정값을 통해 이를 수정하여 보다 정확한 결과를 도출합니다.

- **Technical Details**: 제안된 FPTC-GANs 방법은 두 개의 주요 네트워크인 RMP-GAN(라디오 지도 예측 GAN)과 RMC-GAN(라디오 지도 수정 GAN)으로 구성됩니다. RMP-GAN은 환경 정보를 입력받아 기초적인 라디오 지도를 예측하며, RMC-GAN은 희소한 측정값을 통해 이 예측 결과를 수정합니다. RMP-GAN은 self-attention 메커니즘을 이용하여 장거리 환경 의존성을 캡처하고, RMC-GAN은 residual-connection 블록을 도입하여 입력의 저수준 물리적 정보를 더 잘 보존합니다.

- **Performance Highlights**: 실험 결과에 따르면, FPTC-GANs 방법은 기존 최첨단 방법들과 비교할 때, radio map 구축 성능에서 가장 우수한 결과를 보여줍니다.



### Predicting Fine-grained Behavioral and Psychological Symptoms of Dementia Based on Machine Learning and Smart Wearable Devices (https://arxiv.org/abs/2410.18091)
- **What's New**: 이 연구는 스마트 웨어러블 기기에서 수집한 생리적 신호를 활용하여 BPSD(Behavioral and Psychological Symptoms of Dementia)를 예측하는 새로운 개인화된 프레임워크를 제시합니다. 이는 BPSD 예측을 위한 기계 학습 활용의 간극을 메우는 데 기여합니다.

- **Technical Details**: 제안된 개인화된 BPSD 예측 방법은 개별 행동 패턴을 추출하여 BPSD 발생을 정확하게 예측합니다. 또한, 일반화된 모델은 다양한 패턴을 식별하고 다양한 BPSD 증상 간의 차이를 구별합니다.

- **Performance Highlights**: 제안된 개인화된 방법과 기존의 일반화된 방법 간의 세부 비교에서, AUC(Area Under the Curve)가 16.0% 증가하는 등 모든 성능 지표에서 상당한 개선이 나타났습니다. 이는 우리 접근법이 실질적인 시나리오에서 환자 결과를 개선하고 예방적 개입을 가능하게 하는 데 잠재력을 가지음을 보여줍니다.



### Liver Cancer Knowledge Graph Construction based on dynamic entity replacement and masking strategies RoBERTa-BiLSTM-CRF mod (https://arxiv.org/abs/2410.18090)
- **What's New**: 이번 연구는 한국에서 다섯 번째로 흔한 악성 종양이자 두 번째로 치명적인 간암의 진단과 치료를 위한 지식 그래프(knowledge graph)-기반 시스템을 제안합니다. 기존 데이터 소스와 실제 전자 의료 기록 간의 불일치 문제를 해결하며, 간암에 특화된 첫 번째 지식 그래프를 구축함으로써 의사가 진단하는 데 겪는 어려움을 경감할 수 있을 것으로 기대됩니다.

- **Technical Details**: 지식 그래프는 여섯 가지 단계로 거쳐 구축되었습니다: 개념 계층 설계(conceptual layer design), 데이터 전처리(data preprocessing), 개체 인식(entity identification), 개체 정규화(entity normalization), 지식 융합(knowledge fusion) 및 그래프 시각화(graph visualization). 이 과정에서는 새로운 동적 개체 변경 및 마스킹 전략(Dynamic Entity Replacement and Masking Strategy, DERM)이 도입되어 명명된 개체 인식(named entity recognition)을 개선하였습니다.

- **Performance Highlights**: 간암에 대한 지식 그래프는 1495개의 개체로 구성되어 있으며, 명명된 개체 인식 모델의 정확도는 93.23%, 재현율은 94.69%, F1 점수는 93.96%에 달합니다.



### Empowering Cognitive Digital Twins with Generative Foundation Models: Developing a Low-Carbon Integrated Freight Transportation System (https://arxiv.org/abs/2410.18089)
- **What's New**: 본 논문은 디지털 트윈(Digital Twin)의 발전을 위해 생성적 AI(Generative AI)의 활용 가능성을 제시합니다. 특히, 도시 물류 최적화를 위한 새로운 패러다임을 제안하며, 여러 운송 수단을 아우르는 통합 물류 시스템을 효율적으로 관리할 수 있는 방법을 탐구합니다.

- **Technical Details**: 운송 안정성 향상을 위해 생성적 AI를 활용한 트랜스포머 기반 언어 모델(transformer-based language models)을 사용하여 개념적 프레임워크를 구성합니다. 이러한 모델은 데이터 엔지니어링(data engineering), 분석(analytics), 소프트웨어 개발(software development) 등의 워크플로우를 자동화하여 디지털 트윈의 능력을 확장하고자 합니다. 또한, 이 논문은 현재와 미래의 물류 시스템에 대해 지식 그래프(knowlledge graph)를 이용한 분석 및 작업 자동화를 제안합니다.

- **Performance Highlights**: 이 논문은 초기 프로토타입 결과를 공유하며, 물류 시스템의 최적화를 위한 인공지능 기반 의사결정 지원 도구의 개발을 목표로 하고 있습니다. 특히, 다양한 운송 수단의 융합과 실시간 의사결정이 가능하게 하는 차세대 물류 시스템에 대한 비전을 제시합니다.



### CUPID: A Real-Time Session-Based Reciprocal Recommendation System for a One-on-One Social Discovery Platform (https://arxiv.org/abs/2410.18087)
Comments:
          The 2nd International Workshop on User Understanding from Big Data Workshop (DMU2 2024)

- **What's New**: CUPID는 실시간 1:1 소셜 탐색 플랫폼을 위한 새로운 세션 기반의 상호 추천 시스템을 소개합니다.

- **Technical Details**: CUPID는 사용자 세션 모델링과 실시간 매칭 프로세스를 분리하여 추론 시간을 줄이는 비동기 세션 모델링 접근 방식을 사용하며, 두 단계의 훈련 전략을 통해 임베딩 레이어와 예측 레이어의 훈련을 분리하여 계산 비용을 절감합니다. 이 시스템은 Azar 플랫폼의 대규모 실험 데이터에서 검증되었습니다.

- **Performance Highlights**: CUPID는 비동기 시스템에 비해 응답 지연을 76% 이상 줄이고 사용자 참여도를 현저하게 개선하여 실제 추천 성능을 향상시킵니다. 따뜻한 시작 사용자에 대해서는 평균 채팅 시간을 6.8% 증가시키고, 차가운 시작 사용자에서는 5.9% 증가시킵니다.



### TextureMeDefect: LLM-based Defect Texture Generation for Railway Components on Mobile Devices (https://arxiv.org/abs/2410.18085)
Comments:
          6 Pages, 8 figures

- **What's New**: 이번 연구에서는 산업 응용을 위한 컨텍스트 기반의 현실적인 텍스처 생성의 새로운 가능성을 제시합니다. 특히, 철도 부품의 결함 텍스처 생성을 위한 TextureMeDefect라는 모바일 친화적인 도구를 소개합니다.

- **Technical Details**: TextureMeDefect는 LLM(대형 언어 모델) 기반의 AI-Inferencing 엔진을 활용하여 사용자가 스마트폰이나 태블릿으로 촬영한 철도 부품 이미지에서 상호작용적으로 현실적인 결함 텍스처를 생성할 수 있게 합니다.

- **Performance Highlights**: TextureMeDefect는 전통적인 이미지 생성 도구를 초월하여 의미 있는 텍스처를 더 빠르게 생성하였으며, iOS 및 Android 플랫폼에서의 시간과 비용 효율성을 평가하였습니다. 또한, 세 가지 시나리오에서 소프트웨어의 사용성 점수(SUS)를 분석하였습니다.



### Real-time Sub-milliwatt Epilepsy Detection Implemented on a Spiking Neural Network Edge Inference Processor (https://arxiv.org/abs/2410.16613)
- **What's New**: 이 연구는 스파이킹 신경망(Spiking Neural Network, SNN)을 이용해 간질 발작의 간헐적(interictal) 및 발작적(ictal) 기간을 실시간으로 감지하는 새로운 접근 방식을 제안합니다. 기존 기술들이 적시 진단을 제공하는 데 어려움을 겪고 있는 가운데, SNN을 활용한 진단 방안이 주목받고 있습니다.

- **Technical Details**: 연구에서 제안하는 방법은 Xylo라는 디지털 SNN 뉴로모픽 프로세서를 사용하여 EEG 신호를 분석합니다. 이 프로세서는 스파이크 방식의 누적 통합 및 발화(leaky integrate-and-fire, LIF) 뉴런을 시뮬레이션하며, 전통적인 방식보다 에너지 요구가 현저히 낮습니다. 이를 통해 93.3% 및 92.9%의 높은 테스트 정확도를 기록하였고 평균 전력 소비는 87.4 uW(IO power) + 287.9 uW(computational power)로 측정되었습니다.

- **Performance Highlights**: 제안된 방법은 여러 데이터셋에서 우수한 낮은 지연(latency) 성능을 보여주었으며, 향후 휴대용 및 웨어러블 장치에서의 널리 사용될 것으로 기대됩니다. 이 연구는 간질 진단이 더욱 효율적으로 진행될 수 있도록 도와주며, 신경 과학 연구 및 생체신호 모니터링에 중요한 기여를 할 것으로 보입니다.



### Advancing Interpretability in Text Classification through Prototype Learning (https://arxiv.org/abs/2410.17546)
- **What's New**: ProtoLens는 텍스트 분류를 위한 새로운 프로토타입 기반 모델로, 서브-문장(sub-sentence) 수준의 해석 가능성을 제공하여 투명성이 중요한 응용 프로그램에 적합합니다.

- **Technical Details**: ProtoLens는 Prototype-aware Span Extraction 모듈을 사용하여 학습한 프로토타입과 관련된 텍스트 스팬을 식별하고, Prototype Alignment 메커니즘을 통해 훈련 과정 전반에 걸쳐 프로토타입이 의미론적으로 중요하도록 보장합니다.

- **Performance Highlights**: ProtoLens는 여러 텍스트 분류 벤치마크에서 프로토타입 기반 및 비해석적(baseline) 모델들보다 뛰어난 성능을 나타냅니다.



### HyperspectralViTs: General Hyperspectral Models for On-board Remote Sensing (https://arxiv.org/abs/2410.17248)
Comments:
          13 pages, This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 hyperspectral 데이터 처리 용도로 특별히 설계된 새로운 머신러닝 모델인 HyperspectralViTs를 제안합니다. 이 모델은 기존의 깊이 학습 아키텍처를 기반으로 하며, 효율적인 성능과 빠른 추론 속도를 제공합니다.

- **Technical Details**: HyperspectralViTs는 SegFormer와 EfficientViT와 같은 최신 아키텍처를 활용하며, 전통적인 제품이나 스펙트럼 밴드 압축 전처리에 의존하지 않고 end-to-end 훈련을 지원합니다. 제안된 HyperSegFormer와 HyperEfficientViT 변형 모델을 사용하여의 F1 점수를 개선하였습니다.

- **Performance Highlights**: 메탄 탐지 작업에서, 새로운 합성 데이터셋과 대규모 벤치마크 데이터셋에서 F1 점수를 각각 27%와 13% 개선한 결과를 보여주었습니다. 또한, 추론 속도를 85% 향상시키며, EMIT 센서로부터의 단일 캡처를 30초 이내에 처리할 수 있는 성능을 자랑합니다.



